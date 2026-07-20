"""Numerical core for a gauge-fixed affine carrier with a positive NW residual.

The carrier is an explicit ``2 x 3`` RGB linear tail in image-normalized coordinates.  It is
evaluated
at every pixel while the Gaussian renderer carries only the per-sample residual from that plane::

    q(x, y) = [2 x / (W - 1) - 1, 2 y / (H - 1) - 1]
    r_i = c_i - q(mu_i) beta
    y(x) = q(x) beta + sum_i w_i(x) r_i / (sum_i w_i(x) + eps)

Omitting a constant row fixes the gauge: normalized constant colors already provide the intercept.
For a target ``t``, the unquantized tail is fitted through the actual reproduction defect
``D = Q_pixel - W Q_mu`` as ``argmin_beta ||t - W c - D beta||``.  The reference path forms an
independent dense float64 weight matrix.  The candidate path mirrors the
production float32 support enumeration and ``index_add`` order with exactly four local channels:
mass followed by the three residual numerators.  Neither path changes support membership, clips
the result, regularizes a global fit, or repairs a rank-deficient design.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import torch

from structsplat.render import _EPS, _support_weight, _tile_bounds, _tile_coords


SIGMA_CUTOFF: Final[float] = 3.0
CHANNEL_COUNT: Final[int] = 4
BETA_SHAPE: Final[tuple[int, int]] = (2, 3)
BETA_PAYLOAD_BYTES: Final[int] = 12
RAY_DESIGN_MARGIN: Final[float] = 1.75 / 255.0
RAY_GATE_MARGIN: Final[float] = 2.0 / 255.0
RAY_GRID_DENOMINATOR: Final[int] = 4096


@dataclass(frozen=True)
class FrozenContributors:
    """Detached production clipped-AABB contributor rows."""

    gid: torch.Tensor
    px: torch.Tensor
    py: torch.Tensor
    height: int
    width: int
    sample_count: int

    @property
    def flat(self) -> torch.Tensor:
        """Flattened row-major pixel index for every contributor row."""

        return self.py * int(self.width) + self.px

    @property
    def active_counts(self) -> torch.Tensor:
        """Number of production-support contributors at every pixel."""

        return torch.bincount(
            self.flat, minlength=int(self.height) * int(self.width)
        )


@dataclass(frozen=True)
class GlobalDesignDiagnostics:
    """Diagnostic-total rank and condition report for the global affine design."""

    design: torch.Tensor
    singular_values: torch.Tensor
    rank_threshold: float
    rank: int
    condition_number: float
    accepted: bool


@dataclass(frozen=True)
class AugmentedBasisDiagnostics:
    """Numerical ranks of ``W`` and the augmented basis ``[W,D]``."""

    base_singular_values: torch.Tensor
    augmented_singular_values: torch.Tensor
    common_rank_threshold: float
    base_rank_threshold: float
    augmented_rank_threshold: float
    base_rank: int
    augmented_rank: int
    rank_gain: int

    @property
    def full_tail_gain(self) -> bool:
        """Whether both gauge-fixed spatial columns add independent directions."""

        return self.rank_gain == 2


@dataclass(frozen=True)
class ReproductionDefectFit:
    """Unregularized full-rank reproduction-defect QR fit."""

    beta: torch.Tensor
    baseline: torch.Tensor
    reproduction_design: torch.Tensor
    target_residual: torch.Tensor
    diagnostics: GlobalDesignDiagnostics


@dataclass(frozen=True)
class ReproductionDefectSolution:
    """QR solution of an explicitly supplied reproduction-defect system."""

    beta: torch.Tensor
    diagnostics: GlobalDesignDiagnostics


@dataclass(frozen=True)
class RangeSafeRayProjection:
    """Largest feasible point found on the declared finite descending ray grid."""

    alpha_cap: torch.Tensor
    grid_index: torch.Tensor
    alpha: torch.Tensor
    payload: bytes
    decoded_beta: torch.Tensor
    output: torch.Tensor
    baseline_sse: torch.Tensor
    sse: torch.Tensor
    sse_limit: torch.Tensor
    target_minimum: torch.Tensor
    target_maximum: torch.Tensor
    output_minimum: torch.Tensor
    output_maximum: torch.Tensor
    excursion: torch.Tensor
    range_pass: torch.Tensor
    sse_pass: torch.Tensor
    active_constraints: tuple[str, ...]

    @property
    def feasible(self) -> bool:
        """Whether all post-quantization range and SSE constraints pass."""

        return bool(self.range_pass.all() and self.sse_pass.all())


@dataclass(frozen=True)
class ReferenceResult:
    """Independent float64 dense-oracle output and residual-weight diagnostics."""

    output: torch.Tensor
    mass: torch.Tensor
    residual_numerator: torch.Tensor
    normalized_residual: torch.Tensor
    residual_colors: torch.Tensor
    active_counts: torch.Tensor
    partition: torch.Tensor
    minimum_effective_weight: torch.Tensor
    nonnegative_pass: torch.Tensor
    first_moment: torch.Tensor
    amplification_l1: torch.Tensor
    effective_weights: torch.Tensor
    finite_pass: torch.Tensor
    contributors: FrozenContributors

    @property
    def a1(self) -> torch.Tensor:
        """Residual effective-weight L1 amplification."""

        return self.amplification_l1


@dataclass(frozen=True)
class CandidateResult:
    """Production-style float32 four-channel output and residual diagnostics."""

    output: torch.Tensor
    accumulators: torch.Tensor
    mass: torch.Tensor
    residual_numerator: torch.Tensor
    normalized_residual: torch.Tensor
    residual_colors: torch.Tensor
    active_counts: torch.Tensor
    partition: torch.Tensor
    minimum_effective_weight: torch.Tensor
    nonnegative_pass: torch.Tensor
    first_moment: torch.Tensor
    amplification_l1: torch.Tensor
    finite_pass: torch.Tensor
    contributors: FrozenContributors

    @property
    def a1(self) -> torch.Tensor:
        """Residual effective-weight L1 amplification."""

        return self.amplification_l1


def _validate_extent(height: int, width: int) -> tuple[int, int]:
    height = int(height)
    width = int(width)
    if height < 2 or width < 2:
        raise ValueError("affine carrier coordinates require height and width >= 2")
    return height, width


def _validate_float_tensor(
    name: str,
    value: torch.Tensor,
    shape: tuple[int, ...],
    *,
    dtype: torch.dtype | None = None,
) -> None:
    if value.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if not value.dtype.is_floating_point:
        raise ValueError(f"{name} must be floating point")
    if dtype is not None and value.dtype != dtype:
        raise ValueError(f"{name} must have dtype {dtype}")
    if value.device.type != "cpu":
        raise ValueError(f"{name} must be on CPU")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must be finite")


def _validate_radii(radii: torch.Tensor, count: int) -> None:
    integer_dtypes = (
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    )
    if radii.shape != (count, 2) or radii.dtype not in integer_dtypes:
        raise ValueError("radii must be an integer tensor with shape (N,2)")
    if radii.device.type != "cpu":
        raise ValueError("radii must be on CPU")
    if bool((radii < 0).any()):
        raise ValueError("radii must be nonnegative")


def _validate_beta(beta: torch.Tensor, *, dtype: torch.dtype | None = None) -> None:
    _validate_float_tensor("beta", beta, BETA_SHAPE, dtype=dtype)


def carrier_design(
    points: torch.Tensor,
    height: int,
    width: int,
) -> torch.Tensor:
    """Evaluate gauge-fixed ``[u,v]`` at pixel-coordinate points of shape ``(N,2)``."""

    height, width = _validate_extent(height, width)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must have shape (N,2)")
    if not points.dtype.is_floating_point:
        raise ValueError("points must be floating point")
    if points.device.type != "cpu":
        raise ValueError("points must be on CPU")
    if not bool(torch.isfinite(points).all()):
        raise ValueError("points must be finite")
    u = 2.0 * points[:, 0:1] / float(width - 1) - 1.0
    v = 2.0 * points[:, 1:2] / float(height - 1) - 1.0
    return torch.cat((u, v), dim=1)


def pixel_design(
    height: int,
    width: int,
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return the row-major ``(H*W,2)`` carrier design for all image pixels."""

    height, width = _validate_extent(height, width)
    if dtype not in (torch.float32, torch.float64):
        raise ValueError("pixel design dtype must be torch.float32 or torch.float64")
    yy, xx = torch.meshgrid(
        torch.arange(height, dtype=dtype),
        torch.arange(width, dtype=dtype),
        indexing="ij",
    )
    points = torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=1)
    return carrier_design(points, height, width)


def diagnose_reproduction_design(design: torch.Tensor) -> GlobalDesignDiagnostics:
    """Measure a ``(P,2)`` reproduction-defect rank without fitting or regularizing."""

    if design.ndim != 2 or design.shape[1] != 2:
        raise ValueError("reproduction design must have shape (P,2)")
    if design.dtype not in (torch.float32, torch.float64):
        raise ValueError("reproduction design must have dtype torch.float32 or torch.float64")
    if design.device.type != "cpu" or not bool(torch.isfinite(design).all()):
        raise ValueError("reproduction design must be finite and on CPU")
    singular = torch.linalg.svdvals(design)
    if singular.numel() < 2:
        singular = torch.cat(
            (singular, torch.zeros(2 - singular.numel(), dtype=design.dtype))
        )
    leading = float(singular[0]) if singular.numel() else 0.0
    threshold = max(int(design.shape[0]), 2) * torch.finfo(design.dtype).eps * leading
    rank = int((singular > threshold).sum())
    smallest = float(singular[1])
    condition = leading / smallest if smallest > 0.0 else float("inf")
    accepted = rank == 2 and bool(torch.isfinite(singular).all())
    return GlobalDesignDiagnostics(
        design=design,
        singular_values=singular,
        rank_threshold=float(threshold),
        rank=rank,
        condition_number=float(condition),
        accepted=accepted,
    )


def diagnose_augmented_basis(
    effective_weights: torch.Tensor,
    reproduction_design: torch.Tensor,
) -> AugmentedBasisDiagnostics:
    """Report the declared-epsilon rank gain from ``W`` to ``[W,D]``."""

    if effective_weights.ndim != 2:
        raise ValueError("effective_weights must have shape (P,N)")
    pixel_count = int(effective_weights.shape[0])
    if effective_weights.dtype not in (torch.float32, torch.float64):
        raise ValueError("effective_weights must have dtype torch.float32 or torch.float64")
    _validate_float_tensor(
        "effective_weights",
        effective_weights,
        (pixel_count, int(effective_weights.shape[1])),
        dtype=effective_weights.dtype,
    )
    _validate_float_tensor(
        "reproduction_design",
        reproduction_design,
        (pixel_count, 2),
        dtype=effective_weights.dtype,
    )
    base_singular = torch.linalg.svdvals(effective_weights)
    augmented = torch.cat((effective_weights, reproduction_design), dim=1)
    augmented_singular = torch.linalg.svdvals(augmented)
    augmented_leading = float(augmented_singular[0]) if augmented_singular.numel() else 0.0
    common_threshold = (
        max(int(augmented.shape[0]), int(augmented.shape[1]), 1)
        * torch.finfo(augmented.dtype).eps
        * augmented_leading
    )
    # A common augmented-system threshold is essential: subtracting ranks measured with two
    # independently scaled tolerances can manufacture a basis gain near numerical rank loss.
    base_rank = int((base_singular > common_threshold).sum())
    augmented_rank = int((augmented_singular > common_threshold).sum())
    return AugmentedBasisDiagnostics(
        base_singular_values=base_singular,
        augmented_singular_values=augmented_singular,
        common_rank_threshold=float(common_threshold),
        base_rank_threshold=float(common_threshold),
        augmented_rank_threshold=float(common_threshold),
        base_rank=base_rank,
        augmented_rank=augmented_rank,
        rank_gain=augmented_rank - base_rank,
    )


def solve_reproduction_design_qr(
    design: torch.Tensor,
    target_residual: torch.Tensor,
) -> ReproductionDefectSolution:
    """Solve a supplied full-rank defect system, rejecting rank loss without a fallback."""

    diagnostics = diagnose_reproduction_design(design)
    _validate_float_tensor(
        "target_residual",
        target_residual,
        (int(design.shape[0]), 3),
        dtype=design.dtype,
    )
    if not diagnostics.accepted:
        raise ValueError(
            "reproduction-defect design is rank deficient "
            f"(rank={diagnostics.rank}, condition={diagnostics.condition_number})"
        )
    q, r = torch.linalg.qr(design, mode="reduced")
    beta = torch.linalg.solve_triangular(
        r, q.transpose(0, 1) @ target_residual, upper=True
    )
    if not bool(torch.isfinite(beta).all()):
        raise RuntimeError("reproduction-defect QR solve produced non-finite beta")
    return ReproductionDefectSolution(beta=beta, diagnostics=diagnostics)


def encode_beta_f16(beta: torch.Tensor) -> bytes:
    """Encode a finite ``2 x 3`` carrier as canonical little-endian binary16 (12 bytes)."""

    _validate_beta(beta)
    source = beta.detach().cpu().numpy()
    if bool((np.abs(source) > np.finfo(np.float16).max).any()):
        raise ValueError("beta is outside the finite binary16 range")
    encoded = np.asarray(source, dtype="<f2")
    if not bool(np.isfinite(encoded).all()):
        raise ValueError("binary16 beta must remain finite after quantization")
    payload = encoded.tobytes(order="C")
    if len(payload) != BETA_PAYLOAD_BYTES:
        raise RuntimeError("internal binary16 beta payload size mismatch")
    return payload


def decode_beta_f16(payload: bytes | bytearray | memoryview) -> torch.Tensor:
    """Decode the canonical 12-byte carrier payload to a CPU float32 tensor."""

    try:
        view = memoryview(payload)
    except TypeError as exc:
        raise TypeError("beta payload must be bytes-like") from exc
    if view.nbytes != BETA_PAYLOAD_BYTES:
        raise ValueError(f"beta payload must contain exactly {BETA_PAYLOAD_BYTES} bytes")
    decoded = np.frombuffer(view, dtype="<f2", count=6).astype(np.float32, copy=True)
    if not bool(np.isfinite(decoded).all()):
        raise ValueError("decoded beta payload must be finite")
    return torch.from_numpy(decoded.reshape(BETA_SHAPE))


def quantize_beta_f16(beta: torch.Tensor) -> tuple[bytes, torch.Tensor]:
    """Return the exact payload and its float32 decoder realization."""

    payload = encode_beta_f16(beta)
    return payload, decode_beta_f16(payload)


def _quantize_ray_column(
    beta_column: torch.Tensor,
    alpha: float,
    channel: int,
) -> torch.Tensor:
    trial = torch.zeros(BETA_SHAPE, dtype=torch.float64)
    trial[:, channel] = alpha * beta_column
    return decode_beta_f16(encode_beta_f16(trial))[:, channel]


def _ray_channel_metrics(
    decoded_column: torch.Tensor,
    baseline: torch.Tensor,
    design: torch.Tensor,
    target: torch.Tensor,
    target_minimum: float,
    target_maximum: float,
    sse_limit: float,
) -> tuple[torch.Tensor, float, float, float, bool, bool]:
    output = baseline + design @ decoded_column.to(torch.float64)
    sse = float((output - target).square().sum())
    output_minimum = float(output.min())
    output_maximum = float(output.max())
    excursion = max(target_minimum - output_minimum, output_maximum - target_maximum, 0.0)
    return (
        output,
        sse,
        output_minimum,
        output_maximum,
        sse <= sse_limit,
        excursion <= RAY_GATE_MARGIN,
    )


def project_beta_ray_on_finite_grid(
    beta_ls: torch.Tensor,
    baseline: torch.Tensor,
    reproduction_design: torch.Tensor,
    target: torch.Tensor,
) -> RangeSafeRayProjection:
    """Select each RGB column on its frozen 4097-point descending ray grid.

    This is deliberately *not* claimed to find a globally largest feasible real-valued alpha.
    Each RGB channel is separable and receives its own analytic cap from the conservative
    ``1.75/255`` target-range envelope and finite binary16 range.  Its declared candidates are
    ``alpha_cap[c] * (1 - k/4096)`` for ``k=0..4095``, followed by exactly zero.  Every candidate
    column is binary16 encoded and float32 decoded before testing.  The three accepted columns are
    assembled into one 12-byte payload and rerendered together for final verification.  This avoids
    making one channel discard another channel's safe gain.  Decoder output is never clipped.
    """

    _validate_beta(beta_ls, dtype=torch.float64)
    if reproduction_design.ndim != 2 or reproduction_design.shape[1] != 2:
        raise ValueError("reproduction_design must have shape (P,2)")
    pixel_count = int(reproduction_design.shape[0])
    _validate_float_tensor(
        "reproduction_design",
        reproduction_design,
        (pixel_count, 2),
        dtype=torch.float64,
    )
    _validate_float_tensor("baseline", baseline, (pixel_count, 3), dtype=torch.float64)
    _validate_float_tensor("target", target, (pixel_count, 3), dtype=torch.float64)

    baseline_sse = (baseline - target).square().sum(dim=0)
    sse_limit = baseline_sse + torch.maximum(
        torch.full_like(baseline_sse, 1e-15), 1e-10 * baseline_sse
    )
    target_minimum = target.min(dim=0).values
    target_maximum = target.max(dim=0).values
    # The frozen analytic intersection uses the stricter design margin.  Rows whose ray
    # direction is zero still constrain feasibility, so assert alpha=0 against that declared
    # interval before reducing the inequalities to one-sided upper caps.
    analytic_lower = target_minimum - RAY_DESIGN_MARGIN
    analytic_upper = target_maximum + RAY_DESIGN_MARGIN
    if not bool(
        ((baseline >= analytic_lower[None, :]) & (baseline <= analytic_upper[None, :])).all()
    ):
        raise ValueError(
            "zero beta is not feasible under the frozen analytic range-ray intersection"
        )

    # Independently assert the frozen post-quantization fallback before considering any
    # nonzero column.
    zero_payload, zero_decoded = quantize_beta_f16(torch.zeros_like(beta_ls))
    del zero_payload
    zero_output = baseline + reproduction_design @ zero_decoded.to(torch.float64)
    zero_sse = (zero_output - target).square().sum(dim=0)
    zero_minimum = zero_output.min(dim=0).values
    zero_maximum = zero_output.max(dim=0).values
    zero_excursion = torch.maximum(
        torch.clamp(target_minimum - zero_minimum, min=0.0),
        torch.clamp(zero_maximum - target_maximum, min=0.0),
    )
    if not bool(((zero_excursion <= RAY_GATE_MARGIN) & (zero_sse <= sse_limit)).all()):
        raise ValueError("zero beta is not feasible under the frozen range/SSE constraints")

    alpha_caps = torch.zeros(3, dtype=torch.float64)
    grid_indices = torch.full((3,), RAY_GRID_DENOMINATOR, dtype=torch.int64)
    alphas = torch.zeros(3, dtype=torch.float64)
    selected_unquantized = torch.zeros(BETA_SHAPE, dtype=torch.float64)
    active: list[str] = []
    for channel in range(3):
        channel_baseline = baseline[:, channel]
        channel_target = target[:, channel]
        channel_direction = reproduction_design @ beta_ls[:, channel]
        lower_design = float(target_minimum[channel]) - RAY_DESIGN_MARGIN
        upper_design = float(target_maximum[channel]) + RAY_DESIGN_MARGIN
        cap = 1.0
        positive = channel_direction > 0.0
        if bool(positive.any()):
            cap = min(
                cap,
                float(
                    ((upper_design - channel_baseline[positive]) / channel_direction[positive]).min()
                ),
            )
        negative = channel_direction < 0.0
        if bool(negative.any()):
            cap = min(
                cap,
                float(
                    ((lower_design - channel_baseline[negative]) / channel_direction[negative]).min()
                ),
            )
        cap = max(0.0, min(1.0, cap))
        alpha_caps[channel] = cap

        cap_tolerance = 64.0 * torch.finfo(torch.float64).eps * max(1.0, abs(cap))
        analytic_labels: list[str] = []
        if abs(cap - 1.0) <= cap_tolerance:
            analytic_labels.append(f"unit_interval_rgb{channel}")
        if bool(positive.any()):
            upper_cap = float(
                ((upper_design - channel_baseline[positive]) / channel_direction[positive]).min()
            )
            if abs(cap - upper_cap) <= cap_tolerance:
                analytic_labels.append(f"design_upper_rgb{channel}")
        if bool(negative.any()):
            lower_cap = float(
                ((lower_design - channel_baseline[negative]) / channel_direction[negative]).min()
            )
            if abs(cap - lower_cap) <= cap_tolerance:
                analytic_labels.append(f"design_lower_rgb{channel}")

        previous_failures: list[str] = []
        selected = False
        for grid_index in range(RAY_GRID_DENOMINATOR):
            alpha = cap * (1.0 - grid_index / RAY_GRID_DENOMINATOR)
            try:
                decoded_column = _quantize_ray_column(
                    beta_ls[:, channel], alpha, channel
                )
            except ValueError as exc:
                # Overflow is a failed declared candidate, not a reason to alter the frozen
                # alpha grid or skip its explicit zero fallback.
                if "binary16 range" not in str(exc):
                    raise
                previous_failures = [f"binary16_range_rgb{channel}"]
                continue
            _, _, _, _, sse_pass, range_pass = _ray_channel_metrics(
                decoded_column,
                channel_baseline,
                reproduction_design,
                channel_target,
                float(target_minimum[channel]),
                float(target_maximum[channel]),
                float(sse_limit[channel]),
            )
            if range_pass and sse_pass:
                grid_indices[channel] = grid_index
                alphas[channel] = alpha
                selected_unquantized[:, channel] = alpha * beta_ls[:, channel]
                active.extend(previous_failures if grid_index > 0 else analytic_labels)
                selected = True
                break
            previous_failures = []
            if not range_pass:
                previous_failures.append(f"range_rgb{channel}")
            if not sse_pass:
                previous_failures.append(f"sse_rgb{channel}")
        if not selected:
            active.extend(previous_failures)

    payload, decoded = quantize_beta_f16(selected_unquantized)
    output = baseline + reproduction_design @ decoded.to(torch.float64)
    sse = (output - target).square().sum(dim=0)
    output_minimum = output.min(dim=0).values
    output_maximum = output.max(dim=0).values
    excursion = torch.maximum(
        torch.clamp(target_minimum - output_minimum, min=0.0),
        torch.clamp(output_maximum - target_maximum, min=0.0),
    )
    range_pass = excursion <= RAY_GATE_MARGIN
    sse_pass = sse <= sse_limit
    result = RangeSafeRayProjection(
        alpha_cap=alpha_caps,
        grid_index=grid_indices,
        alpha=alphas,
        payload=payload,
        decoded_beta=decoded,
        output=output,
        baseline_sse=baseline_sse,
        sse=sse,
        sse_limit=sse_limit,
        target_minimum=target_minimum,
        target_maximum=target_maximum,
        output_minimum=output_minimum,
        output_maximum=output_maximum,
        excursion=excursion,
        range_pass=range_pass,
        sse_pass=sse_pass,
        active_constraints=tuple(active),
    )
    if len(result.payload) != BETA_PAYLOAD_BYTES or not result.feasible:
        raise RuntimeError("internal finite-grid ray projection invariant failed")
    return result


def conics_from_parameters(
    log_scales: torch.Tensor,
    rotations: torch.Tensor,
) -> torch.Tensor:
    """Recompute zero-dilation rotation-scale conics without detaching parameters."""

    if log_scales.ndim != 2 or log_scales.shape[1] != 2:
        raise ValueError("log_scales must have shape (N,2)")
    if rotations.shape != (log_scales.shape[0],):
        raise ValueError("rotations must have shape (N,)")
    if log_scales.dtype != rotations.dtype:
        raise ValueError("log_scales and rotations must have the same dtype")
    if log_scales.device != rotations.device:
        raise ValueError("log_scales and rotations must share a device")
    if not bool(torch.isfinite(log_scales).all() and torch.isfinite(rotations).all()):
        raise ValueError("log_scales and rotations must be finite")
    scales = torch.exp(log_scales)
    inv_x = scales[:, 0].square().reciprocal()
    inv_y = scales[:, 1].square().reciprocal()
    cosine = torch.cos(rotations)
    sine = torch.sin(rotations)
    a = cosine.square() * inv_x + sine.square() * inv_y
    b = cosine * sine * (inv_x - inv_y)
    c = sine.square() * inv_x + cosine.square() * inv_y
    return torch.stack((a, b, c), dim=1)


def enumerate_contributors(
    means: torch.Tensor,
    radii: torch.Tensor,
    height: int,
    width: int,
) -> FrozenContributors:
    """Call production clipped-AABB enumeration, including its detached rounding."""

    height, width = _validate_extent(height, width)
    if means.ndim != 2 or means.shape[1] != 2:
        raise ValueError("means must have shape (N,2)")
    if not means.dtype.is_floating_point or means.device.type != "cpu":
        raise ValueError("means must be floating point and on CPU")
    if not bool(torch.isfinite(means).all()):
        raise ValueError("means must be finite")
    _validate_radii(radii, int(means.shape[0]))
    x0, y0, tile_width, tile_sizes = _tile_bounds(means, radii, height, width)
    gid, px, py = _tile_coords(
        x0,
        y0,
        tile_width,
        tile_sizes,
        0,
        int(means.shape[0]),
        means.device,
    )
    return FrozenContributors(
        gid=gid,
        px=px,
        py=py,
        height=height,
        width=width,
        sample_count=int(means.shape[0]),
    )


def _validate_contributors(
    contributors: FrozenContributors,
    count: int,
    height: int,
    width: int,
) -> None:
    if not isinstance(contributors, FrozenContributors):
        raise TypeError("contributors must be FrozenContributors")
    if (
        int(contributors.sample_count) != count
        or int(contributors.height) != height
        or int(contributors.width) != width
    ):
        raise ValueError("contributors do not match field or image dimensions")
    length = int(contributors.gid.numel())
    for name, value in (
        ("gid", contributors.gid),
        ("px", contributors.px),
        ("py", contributors.py),
    ):
        if value.shape != (length,) or value.dtype != torch.int64:
            raise ValueError(f"contributors.{name} must be a 1-D int64 tensor")
        if value.device.type != "cpu":
            raise ValueError(f"contributors.{name} must be on CPU")
    if length:
        if int(contributors.gid.min()) < 0 or int(contributors.gid.max()) >= count:
            raise ValueError("contributor Gaussian index is out of range")
        if int(contributors.px.min()) < 0 or int(contributors.px.max()) >= width:
            raise ValueError("contributor x coordinate is out of range")
        if int(contributors.py.min()) < 0 or int(contributors.py.max()) >= height:
            raise ValueError("contributor y coordinate is out of range")


def _validate_render_inputs(
    means: torch.Tensor,
    conics: torch.Tensor,
    colors: torch.Tensor,
    beta: torch.Tensor,
    radii: torch.Tensor,
    height: int,
    width: int,
    *,
    dtype: torch.dtype,
) -> tuple[int, int]:
    height, width = _validate_extent(height, width)
    if means.ndim != 2 or means.shape[1] != 2:
        raise ValueError("means must have shape (N,2)")
    count = int(means.shape[0])
    _validate_float_tensor("means", means, (count, 2), dtype=dtype)
    _validate_float_tensor("conics", conics, (count, 3), dtype=dtype)
    _validate_float_tensor("colors", colors, (count, 3), dtype=dtype)
    _validate_beta(beta, dtype=dtype)
    _validate_radii(radii, count)
    return height, width


def residual_colors(
    means: torch.Tensor,
    colors: torch.Tensor,
    beta: torch.Tensor,
    height: int,
    width: int,
) -> torch.Tensor:
    """Subtract the explicit carrier at each Gaussian mean from its stored RGB."""

    height, width = _validate_extent(height, width)
    if means.ndim != 2 or means.shape[1] != 2:
        raise ValueError("means must have shape (N,2)")
    count = int(means.shape[0])
    _validate_float_tensor("means", means, (count, 2))
    _validate_float_tensor("colors", colors, (count, 3), dtype=means.dtype)
    _validate_beta(beta, dtype=means.dtype)
    return colors - carrier_design(means, height, width) @ beta


def _contribution_weights(
    means: torch.Tensor,
    conics: torch.Tensor,
    contributors: FrozenContributors,
) -> torch.Tensor:
    gid = contributors.gid
    dx = contributors.px.to(means.dtype) - means[gid, 0]
    dy = contributors.py.to(means.dtype) - means[gid, 1]
    a = conics[gid, 0]
    b = conics[gid, 1]
    c = conics[gid, 2]
    quadratic = a * dx.square() + 2.0 * b * dx * dy + c * dy.square()
    return _support_weight(quadratic, SIGMA_CUTOFF, False)


def reference_render(
    means: torch.Tensor,
    conics: torch.Tensor,
    colors: torch.Tensor,
    beta: torch.Tensor,
    radii: torch.Tensor,
    height: int,
    width: int,
    *,
    contributors: FrozenContributors | None = None,
) -> ReferenceResult:
    """Render ordinary RGB after forming residuals from the decoded carrier."""

    residuals = residual_colors(means, colors, beta, height, width)
    return reference_render_residual(
        means,
        conics,
        residuals,
        beta,
        radii,
        height,
        width,
        contributors=contributors,
    )


def reference_render_residual(
    means: torch.Tensor,
    conics: torch.Tensor,
    residuals: torch.Tensor,
    beta: torch.Tensor,
    radii: torch.Tensor,
    height: int,
    width: int,
    *,
    contributors: FrozenContributors | None = None,
) -> ReferenceResult:
    """Render independently stored residual RGB with the float64 dense-weight oracle."""

    height, width = _validate_render_inputs(
        means, conics, residuals, beta, radii, height, width, dtype=torch.float64
    )
    if contributors is None:
        contributors = enumerate_contributors(means, radii, height, width)
    _validate_contributors(contributors, int(means.shape[0]), height, width)
    pixel_count = height * width
    count = int(means.shape[0])
    weights = _contribution_weights(means, conics, contributors)
    dense_weights = torch.zeros((pixel_count, count), dtype=torch.float64)
    if weights.numel():
        dense_weights[contributors.flat, contributors.gid] = weights
    mass = dense_weights.sum(dim=1)
    residual_numerator = dense_weights @ residuals
    normalized_residual = residual_numerator / (mass[:, None] + _EPS)
    pixels = pixel_design(height, width, dtype=torch.float64)
    output = pixels @ beta + normalized_residual
    effective = dense_weights / (mass[:, None] + _EPS)
    partition = effective.sum(dim=1)
    sample_coordinates = carrier_design(means, height, width)
    first_moment = effective @ sample_coordinates - partition[:, None] * pixels
    if count:
        minimum_effective_weight = effective.min(dim=1).values
    else:
        minimum_effective_weight = torch.zeros(pixel_count, dtype=torch.float64)
    nonnegative = minimum_effective_weight >= 0.0
    amplification = effective.abs().sum(dim=1)
    finite = (
        torch.isfinite(output).all(dim=1)
        & torch.isfinite(mass)
        & torch.isfinite(residual_numerator).all(dim=1)
        & torch.isfinite(normalized_residual).all(dim=1)
        & torch.isfinite(partition)
        & torch.isfinite(minimum_effective_weight)
        & torch.isfinite(first_moment).all(dim=1)
        & torch.isfinite(amplification)
    )
    return ReferenceResult(
        output=output.reshape(height, width, 3),
        mass=mass,
        residual_numerator=residual_numerator,
        normalized_residual=normalized_residual,
        residual_colors=residuals,
        active_counts=contributors.active_counts,
        partition=partition,
        minimum_effective_weight=minimum_effective_weight,
        nonnegative_pass=nonnegative,
        first_moment=first_moment,
        amplification_l1=amplification,
        effective_weights=effective,
        finite_pass=finite,
        contributors=contributors,
    )


def reproduction_defect_design(
    means: torch.Tensor,
    effective_weights: torch.Tensor,
    height: int,
    width: int,
) -> torch.Tensor:
    """Build ``D = Q_pixel - W Q_mu`` from an explicitly supplied normalized weight matrix."""

    height, width = _validate_extent(height, width)
    if means.ndim != 2 or means.shape[1] != 2:
        raise ValueError("means must have shape (N,2)")
    if means.dtype not in (torch.float32, torch.float64):
        raise ValueError("means must have dtype torch.float32 or torch.float64")
    if means.device.type != "cpu" or not bool(torch.isfinite(means).all()):
        raise ValueError("means must be finite and on CPU")
    expected = (height * width, int(means.shape[0]))
    _validate_float_tensor(
        "effective_weights", effective_weights, expected, dtype=means.dtype
    )
    if bool((effective_weights < 0.0).any()):
        raise ValueError("effective weights must be nonnegative")
    return pixel_design(height, width, dtype=means.dtype) - (
        effective_weights @ carrier_design(means, height, width)
    )


def fit_reproduction_defect_qr(
    means: torch.Tensor,
    conics: torch.Tensor,
    colors: torch.Tensor,
    radii: torch.Tensor,
    target: torch.Tensor,
    height: int,
    width: int,
    *,
    contributors: FrozenContributors | None = None,
) -> ReproductionDefectFit:
    """Fit ``argmin ||target - W colors - (Qpix - W Qmu) beta||`` in float64."""

    height, width = _validate_extent(height, width)
    _validate_float_tensor("target", target, (height, width, 3), dtype=torch.float64)
    zero_beta = torch.zeros(BETA_SHAPE, dtype=torch.float64)
    baseline_result = reference_render(
        means,
        conics,
        colors,
        zero_beta,
        radii,
        height,
        width,
        contributors=contributors,
    )
    baseline = baseline_result.output.reshape(-1, 3)
    design = reproduction_defect_design(
        means, baseline_result.effective_weights, height, width
    )
    target_residual = target.reshape(-1, 3) - baseline
    solution = solve_reproduction_design_qr(design, target_residual)
    return ReproductionDefectFit(
        beta=solution.beta,
        baseline=baseline,
        reproduction_design=design,
        target_residual=target_residual,
        diagnostics=solution.diagnostics,
    )


def candidate_render(
    means: torch.Tensor,
    conics: torch.Tensor,
    colors: torch.Tensor,
    beta: torch.Tensor,
    radii: torch.Tensor,
    height: int,
    width: int,
    *,
    contributors: FrozenContributors | None = None,
) -> CandidateResult:
    """Render ordinary RGB after forming residuals from the decoded carrier."""

    residuals = residual_colors(means, colors, beta, height, width)
    return candidate_render_residual(
        means,
        conics,
        residuals,
        beta,
        radii,
        height,
        width,
        contributors=contributors,
    )


def candidate_render_residual(
    means: torch.Tensor,
    conics: torch.Tensor,
    residuals: torch.Tensor,
    beta: torch.Tensor,
    radii: torch.Tensor,
    height: int,
    width: int,
    *,
    contributors: FrozenContributors | None = None,
) -> CandidateResult:
    """Render independently stored residual RGB with exactly four float32 channels."""

    height, width = _validate_render_inputs(
        means, conics, residuals, beta, radii, height, width, dtype=torch.float32
    )
    if contributors is None:
        contributors = enumerate_contributors(means, radii, height, width)
    _validate_contributors(contributors, int(means.shape[0]), height, width)
    pixel_count = height * width
    weights = _contribution_weights(means, conics, contributors)
    row_residual = residuals[contributors.gid]
    contributions = torch.cat((weights[:, None], weights[:, None] * row_residual), dim=1)
    if contributions.shape[1] != CHANNEL_COUNT:
        raise RuntimeError("internal four-channel accumulator layout mismatch")
    accumulators = torch.zeros((pixel_count, CHANNEL_COUNT), dtype=torch.float32)
    accumulators = accumulators.index_add(0, contributors.flat, contributions)
    mass = accumulators[:, 0]
    residual_numerator = accumulators[:, 1:4]
    normalized_residual = residual_numerator / (mass[:, None] + _EPS)
    pixels = pixel_design(height, width, dtype=torch.float32)
    output = pixels @ beta + normalized_residual

    element_probability = weights / (mass[contributors.flat] + _EPS)
    sample_coordinates = carrier_design(means, height, width)
    offsets = sample_coordinates[contributors.gid] - pixels[contributors.flat]
    partition = torch.zeros(pixel_count, dtype=torch.float32).index_add(
        0, contributors.flat, element_probability
    )
    first_moment = torch.zeros((pixel_count, 2), dtype=torch.float32).index_add(
        0, contributors.flat, element_probability[:, None] * offsets
    )
    amplification = torch.zeros(pixel_count, dtype=torch.float32).index_add(
        0, contributors.flat, element_probability.abs()
    )
    minimum_effective_weight = torch.full(
        (pixel_count,), float("inf"), dtype=torch.float32
    ).scatter_reduce(
        0, contributors.flat, element_probability, reduce="amin", include_self=True
    )
    minimum_effective_weight = torch.where(
        contributors.active_counts > 0,
        minimum_effective_weight,
        torch.zeros_like(minimum_effective_weight),
    )
    nonnegative = minimum_effective_weight >= 0.0
    finite = (
        torch.isfinite(accumulators).all(dim=1)
        & torch.isfinite(normalized_residual).all(dim=1)
        & torch.isfinite(output).all(dim=1)
        & torch.isfinite(partition)
        & torch.isfinite(minimum_effective_weight)
        & torch.isfinite(first_moment).all(dim=1)
        & torch.isfinite(amplification)
    )
    return CandidateResult(
        output=output.reshape(height, width, 3),
        accumulators=accumulators,
        mass=mass,
        residual_numerator=residual_numerator,
        normalized_residual=normalized_residual,
        residual_colors=residuals,
        active_counts=contributors.active_counts,
        partition=partition,
        minimum_effective_weight=minimum_effective_weight,
        nonnegative_pass=nonnegative,
        first_moment=first_moment,
        amplification_l1=amplification,
        finite_pass=finite,
        contributors=contributors,
    )


def reference_render_frozen(
    means: torch.Tensor,
    log_scales: torch.Tensor,
    rotations: torch.Tensor,
    colors: torch.Tensor,
    beta: torch.Tensor,
    radii: torch.Tensor,
    contributors: FrozenContributors,
) -> torch.Tensor:
    """Differentiable float64 carrier render with frozen production support."""

    return reference_render(
        means,
        conics_from_parameters(log_scales, rotations),
        colors,
        beta,
        radii,
        contributors.height,
        contributors.width,
        contributors=contributors,
    ).output


def candidate_render_frozen(
    means: torch.Tensor,
    log_scales: torch.Tensor,
    rotations: torch.Tensor,
    colors: torch.Tensor,
    beta: torch.Tensor,
    radii: torch.Tensor,
    contributors: FrozenContributors,
) -> torch.Tensor:
    """Differentiable float32 carrier render with frozen production support."""

    return candidate_render(
        means,
        conics_from_parameters(log_scales, rotations),
        colors,
        beta,
        radii,
        contributors.height,
        contributors.width,
        contributors=contributors,
    ).output


def reference_render_residual_frozen(
    means: torch.Tensor,
    log_scales: torch.Tensor,
    rotations: torch.Tensor,
    residuals: torch.Tensor,
    beta: torch.Tensor,
    radii: torch.Tensor,
    contributors: FrozenContributors,
) -> torch.Tensor:
    """Differentiable float64 render of independent residual parameters on frozen support."""

    return reference_render_residual(
        means,
        conics_from_parameters(log_scales, rotations),
        residuals,
        beta,
        radii,
        contributors.height,
        contributors.width,
        contributors=contributors,
    ).output


def candidate_render_residual_frozen(
    means: torch.Tensor,
    log_scales: torch.Tensor,
    rotations: torch.Tensor,
    residuals: torch.Tensor,
    beta: torch.Tensor,
    radii: torch.Tensor,
    contributors: FrozenContributors,
) -> torch.Tensor:
    """Differentiable float32 render of independent residual parameters on frozen support."""

    return candidate_render_residual(
        means,
        conics_from_parameters(log_scales, rotations),
        residuals,
        beta,
        radii,
        contributors.height,
        contributors.width,
        contributors=contributors,
    ).output


def squared_error_loss(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Finite shape-checked mean squared error for directional-gradient assays."""

    if output.shape != target.shape or output.ndim != 3 or output.shape[2] != 3:
        raise ValueError("output and target must have matching shape (H,W,3)")
    if output.dtype != target.dtype or output.device != target.device:
        raise ValueError("output and target must share dtype and device")
    if not bool(torch.isfinite(output).all() and torch.isfinite(target).all()):
        raise ValueError("output and target must be finite")
    return (output - target).square().mean()


__all__ = [
    "BETA_PAYLOAD_BYTES",
    "BETA_SHAPE",
    "CHANNEL_COUNT",
    "RAY_DESIGN_MARGIN",
    "RAY_GATE_MARGIN",
    "RAY_GRID_DENOMINATOR",
    "SIGMA_CUTOFF",
    "AugmentedBasisDiagnostics",
    "CandidateResult",
    "FrozenContributors",
    "GlobalDesignDiagnostics",
    "ReferenceResult",
    "RangeSafeRayProjection",
    "ReproductionDefectFit",
    "ReproductionDefectSolution",
    "candidate_render",
    "candidate_render_frozen",
    "candidate_render_residual",
    "candidate_render_residual_frozen",
    "carrier_design",
    "conics_from_parameters",
    "decode_beta_f16",
    "diagnose_augmented_basis",
    "diagnose_reproduction_design",
    "encode_beta_f16",
    "enumerate_contributors",
    "fit_reproduction_defect_qr",
    "pixel_design",
    "project_beta_ray_on_finite_grid",
    "quantize_beta_f16",
    "reference_render",
    "reference_render_frozen",
    "reference_render_residual",
    "reference_render_residual_frozen",
    "reproduction_defect_design",
    "residual_colors",
    "squared_error_loss",
    "solve_reproduction_design_qr",
]
