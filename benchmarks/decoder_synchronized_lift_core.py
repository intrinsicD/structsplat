"""Numerical core for a decoder-synchronized robust affine lift.

The mechanism derives a robust global affine trend only from decoded Gaussian means and ordinary
RGB colors. A target-blind smooth maximum of robust-plane-residual neighbor jumps attenuates the
complete per-channel affine correction, then the existing positive normalized Gaussian compositor
carries the residual::

    beta = robust_plane(decoded_means, decoded_colors)
    beta[:, j] *= confidence_j
    residual = colors - X_mu @ beta
    output = X_pixel @ beta + W @ residual

No coefficient, residual color, target statistic, or confidence is transmitted.  The functions
below deliberately keep the global solve and confidence construction explicit so the assay can
audit gradients, row-order invariance, and decoder work separately from Gaussian rendering.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Final

import numpy as np
import torch

from benchmarks import affine_carrier_core as affine_core


IRLS_ITERATIONS: Final[int] = 8
ROBUST_SCALE_FLOOR: Final[float] = 1.0 / 255.0
ROBUST_MAD_FACTOR: Final[float] = 1.4826
NEIGHBOR_COUNT: Final[int] = 4
EDGE_ABS_EPSILON: Final[float] = 1.0 / 4096.0
COLOR_SCALE_FLOOR: Final[float] = 1.0 / 255.0
SMOOTH_MAX_TEMPERATURE: Final[float] = 0.1
CONFIDENCE_LOW: Final[float] = 1.20
CONFIDENCE_HIGH: Final[float] = 1.50
MAXIMUM_DESIGN_CONDITION: Final[float] = 8.0


@dataclass(frozen=True)
class DesignDiagnostics:
    """Rank and condition diagnostics for the decoded-mean affine design."""

    singular_values: torch.Tensor
    rank_threshold: float
    rank: int
    condition_number: float
    accepted: bool


@dataclass(frozen=True)
class RobustPlaneResult:
    """Fixed-iteration per-channel Cauchy IRLS result."""

    beta: torch.Tensor
    residual: torch.Tensor
    scales: torch.Tensor
    weights: torch.Tensor
    diagnostics: DesignDiagnostics
    weighted_condition_numbers: torch.Tensor
    error_median_row_indices: torch.Tensor
    absolute_deviation_median_row_indices: torch.Tensor
    iterations: int


@dataclass(frozen=True)
class OrderStatisticTrace:
    """Central row identities for both IRLS linear-median evaluations."""

    error_median_row_indices: torch.Tensor
    absolute_deviation_median_row_indices: torch.Tensor


@dataclass(frozen=True)
class ConfidenceResult:
    """Smooth trend-residual-jump confidence on a deterministic directed graph."""

    alpha: torch.Tensor
    score: torch.Tensor
    color_scale: torch.Tensor
    neighbor_indices: torch.Tensor
    edge_coordinate_pairs: torch.Tensor


@dataclass(frozen=True)
class LiftState:
    """Complete decoder-derived affine lift state."""

    beta: torch.Tensor
    residual: torch.Tensor
    unconstrained_beta: torch.Tensor
    confidence: ConfidenceResult
    robust_plane: RobustPlaneResult


@dataclass(frozen=True)
class LiftRender:
    """Lift output plus the underlying positive normalized residual render."""

    output: torch.Tensor
    base: affine_core.ReferenceResult | affine_core.CandidateResult
    state: LiftState


@dataclass(frozen=True)
class EffectiveOperatorDiagnostics:
    """OLS lift operator diagnostics at fixed geometry and dense Gaussian weights."""

    left_inverse: torch.Tensor
    operator: torch.Tensor
    base_rank: int
    operator_rank: int
    common_rank_threshold: float
    reproduction_max_abs: float
    minimum_coefficient: float
    maximum_row_l1: float


def _validate_extent(height: int, width: int) -> tuple[int, int]:
    height = int(height)
    width = int(width)
    if height < 2 or width < 2:
        raise ValueError("decoder-synchronized coordinates require height and width >= 2")
    return height, width


def _validate_matrix(
    name: str,
    value: torch.Tensor,
    columns: int,
    *,
    dtype: torch.dtype | None = None,
    minimum_rows: int = 1,
) -> None:
    if value.ndim != 2 or value.shape[1] != columns or value.shape[0] < minimum_rows:
        raise ValueError(f"{name} must have shape (N,{columns}) with N >= {minimum_rows}")
    if not value.dtype.is_floating_point:
        raise ValueError(f"{name} must be floating point")
    if dtype is not None and value.dtype != dtype:
        raise ValueError(f"{name} must have dtype {dtype}")
    if value.device.type != "cpu":
        raise ValueError(f"{name} must be on CPU")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must be finite")


def affine_design(points: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Return ``[1, x_normalized, y_normalized]`` for ``(x,y)`` point rows."""

    height, width = _validate_extent(height, width)
    _validate_matrix("points", points, 2)
    one = torch.ones((points.shape[0],), dtype=points.dtype, device=points.device)
    x = 2.0 * points[:, 0] / float(width - 1) - 1.0
    y = 2.0 * points[:, 1] / float(height - 1) - 1.0
    return torch.stack((one, x, y), dim=1)


def pixel_design(height: int, width: int, *, dtype: torch.dtype) -> torch.Tensor:
    """Return the row-major full-pixel affine design."""

    height, width = _validate_extent(height, width)
    if not dtype.is_floating_point:
        raise ValueError("pixel design dtype must be floating point")
    yy, xx = torch.meshgrid(
        torch.arange(height, dtype=dtype),
        torch.arange(width, dtype=dtype),
        indexing="ij",
    )
    points = torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=1)
    return affine_design(points, height, width)


def diagnose_design(design: torch.Tensor) -> DesignDiagnostics:
    """Diagnose the full-rank three-column decoded-mean design."""

    _validate_matrix("design", design, 3, dtype=torch.float64, minimum_rows=3)
    singular_values = torch.linalg.svdvals(design)
    threshold_tensor = (
        max(int(design.shape[0]), 3) * torch.finfo(torch.float64).eps * singular_values[0]
    )
    rank = int((singular_values > threshold_tensor).sum().item())
    smallest = float(singular_values[-1].item())
    condition = math.inf if smallest <= 0.0 else float(singular_values[0].item()) / smallest
    accepted = rank == 3 and math.isfinite(condition) and condition <= MAXIMUM_DESIGN_CONDITION
    return DesignDiagnostics(
        singular_values=singular_values,
        rank_threshold=float(threshold_tensor.item()),
        rank=rank,
        condition_number=condition,
        accepted=accepted,
    )


def _qr_solve(design: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    """Reduced Householder QR solve with an explicit shared-rank precondition."""

    if design.ndim != 2 or design.shape[1] != 3:
        raise ValueError("QR design must have shape (N,3)")
    if rhs.ndim != 2 or rhs.shape[0] != design.shape[0]:
        raise ValueError("QR right-hand side must have shape (N,C)")
    singular_values = torch.linalg.svdvals(design)
    threshold = max(int(design.shape[0]), 3) * torch.finfo(design.dtype).eps * singular_values[0]
    if not bool(singular_values[-1] > threshold):
        raise ValueError("affine QR design is rank deficient")
    q, r = torch.linalg.qr(design, mode="reduced")
    solution = torch.linalg.solve_triangular(r, q.transpose(0, 1) @ rhs, upper=True)
    if not bool(torch.isfinite(solution).all()):
        raise RuntimeError("affine QR solve produced non-finite values")
    return solution


def _linear_median_with_support(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the frozen linear median and the exact rows carrying its autograd support."""

    if values.ndim != 1 or values.numel() < 1:
        raise ValueError("quantile trace values must be a nonempty vector")
    ordered, order = torch.sort(values, stable=True)
    lower = (int(values.numel()) - 1) // 2
    upper = int(values.numel()) // 2
    median = 0.5 * (ordered[lower] + ordered[upper])
    return median, torch.stack((order[lower], order[upper]))


def robust_plane_irls(
    means: torch.Tensor,
    colors: torch.Tensor,
    height: int,
    width: int,
) -> RobustPlaneResult:
    """Fit the frozen eight-update per-channel Cauchy IRLS affine plane."""

    _validate_matrix("means", means, 2, dtype=torch.float64, minimum_rows=5)
    _validate_matrix("colors", colors, 3, dtype=torch.float64, minimum_rows=5)
    if means.shape[0] != colors.shape[0]:
        raise ValueError("means and colors must have the same row count")
    design = affine_design(means, height, width)
    diagnostics = diagnose_design(design)
    if not diagnostics.accepted:
        raise ValueError("decoded-mean affine design failed rank/condition gates")

    beta = _qr_solve(design, colors)
    final_scales = torch.empty((3,), dtype=torch.float64)
    final_weights = torch.empty_like(colors)
    condition_history: list[torch.Tensor] = []
    error_median_history: list[torch.Tensor] = []
    deviation_median_history: list[torch.Tensor] = []
    for _ in range(IRLS_ITERATIONS):
        residual = colors - design @ beta
        channel_solutions: list[torch.Tensor] = []
        scales: list[torch.Tensor] = []
        weights: list[torch.Tensor] = []
        conditions: list[torch.Tensor] = []
        error_median_rows: list[torch.Tensor] = []
        deviation_median_rows: list[torch.Tensor] = []
        for channel in range(3):
            error = residual[:, channel]
            center, error_median_row_ids = _linear_median_with_support(error)
            absolute_deviation = torch.abs(error - center)
            deviation_median, deviation_median_row_ids = _linear_median_with_support(
                absolute_deviation
            )
            scale = torch.maximum(
                torch.as_tensor(ROBUST_SCALE_FLOOR, dtype=error.dtype),
                ROBUST_MAD_FACTOR * deviation_median,
            )
            weight = 1.0 / (1.0 + torch.square(error / scale))
            root_weight = torch.sqrt(weight)
            weighted_design = design * root_weight[:, None]
            weighted_rhs = colors[:, channel : channel + 1] * root_weight[:, None]
            singular_values = torch.linalg.svdvals(weighted_design)
            condition = singular_values[0] / singular_values[-1]
            if not bool(torch.isfinite(condition)):
                raise ValueError("weighted affine design has a non-finite condition number")
            channel_solutions.append(_qr_solve(weighted_design, weighted_rhs)[:, 0])
            scales.append(scale)
            weights.append(weight)
            conditions.append(condition)
            error_median_rows.append(error_median_row_ids)
            deviation_median_rows.append(deviation_median_row_ids)
        beta = torch.stack(channel_solutions, dim=1)
        final_scales = torch.stack(scales)
        final_weights = torch.stack(weights, dim=1)
        condition_history.append(torch.stack(conditions))
        error_median_history.append(torch.stack(error_median_rows))
        deviation_median_history.append(torch.stack(deviation_median_rows))

    residual = colors - design @ beta
    return RobustPlaneResult(
        beta=beta,
        residual=residual,
        scales=final_scales,
        weights=final_weights,
        diagnostics=diagnostics,
        weighted_condition_numbers=torch.stack(condition_history),
        error_median_row_indices=torch.stack(error_median_history),
        absolute_deviation_median_row_indices=torch.stack(deviation_median_history),
        iterations=IRLS_ITERATIONS,
    )


def robust_plane_order_statistic_trace(
    means: torch.Tensor,
    colors: torch.Tensor,
    height: int,
    width: int,
) -> OrderStatisticTrace:
    """Recompute and expose the frozen IRLS central order-statistic row identities."""

    result = robust_plane_irls(means, colors, height, width)
    return OrderStatisticTrace(
        error_median_row_indices=result.error_median_row_indices,
        absolute_deviation_median_row_indices=(result.absolute_deviation_median_row_indices),
    )


def ordinary_plane_qr(
    means: torch.Tensor,
    colors: torch.Tensor,
    height: int,
    width: int,
) -> tuple[torch.Tensor, DesignDiagnostics]:
    """Fit the non-robust decoded-color affine diagnostic by strict reduced QR."""

    _validate_matrix("means", means, 2, dtype=torch.float64, minimum_rows=5)
    _validate_matrix("colors", colors, 3, dtype=torch.float64, minimum_rows=5)
    if means.shape[0] != colors.shape[0]:
        raise ValueError("means and colors must have the same row count")
    design = affine_design(means, height, width)
    diagnostics = diagnose_design(design)
    if not diagnostics.accepted:
        raise ValueError("decoded-mean affine design failed rank/condition gates")
    return _qr_solve(design, colors), diagnostics


def dense_effective_operator_diagnostics(
    weights: torch.Tensor,
    means: torch.Tensor,
    height: int,
    width: int,
) -> EffectiveOperatorDiagnostics:
    """Build the ungated OLS color-to-pixel operator for nondecision diagnostics."""

    _validate_matrix("means", means, 2, dtype=torch.float64, minimum_rows=5)
    pixel_count = int(height) * int(width)
    count = int(means.shape[0])
    if weights.shape != (pixel_count, count):
        raise ValueError("weights must have shape (H*W,N)")
    if weights.dtype != torch.float64 or weights.device.type != "cpu":
        raise ValueError("weights must be a CPU float64 tensor")
    if not bool(torch.isfinite(weights).all()) or bool((weights < 0.0).any()):
        raise ValueError("weights must be finite and nonnegative")
    sample_design = affine_design(means, height, width)
    diagnostics = diagnose_design(sample_design)
    if not diagnostics.accepted:
        raise ValueError("decoded-mean affine design failed rank/condition gates")
    q, r = torch.linalg.qr(sample_design, mode="reduced")
    left_inverse = torch.linalg.solve_triangular(r, q.transpose(0, 1), upper=True)
    pixels = pixel_design(height, width, dtype=torch.float64)
    operator = weights + (pixels - weights @ sample_design) @ left_inverse
    common_sigma = torch.linalg.svdvals(operator)[0]
    threshold = max(pixel_count, count) * torch.finfo(torch.float64).eps * common_sigma
    base_rank = int((torch.linalg.svdvals(weights) > threshold).sum().item())
    operator_rank = int((torch.linalg.svdvals(operator) > threshold).sum().item())
    reproduction = operator @ sample_design
    return EffectiveOperatorDiagnostics(
        left_inverse=left_inverse,
        operator=operator,
        base_rank=base_rank,
        operator_rank=operator_rank,
        common_rank_threshold=float(threshold.item()),
        reproduction_max_abs=float(torch.max(torch.abs(reproduction - pixels)).item()),
        minimum_coefficient=float(torch.min(operator).item()),
        maximum_row_l1=float(torch.max(torch.sum(torch.abs(operator), dim=1)).item()),
    )


def dense_render_with_beta(
    weights: torch.Tensor,
    means: torch.Tensor,
    colors: torch.Tensor,
    beta: torch.Tensor,
    height: int,
    width: int,
) -> torch.Tensor:
    """Render a caller-supplied diagnostic plane with decoded ordinary colors."""

    _validate_matrix("means", means, 2, dtype=torch.float64, minimum_rows=5)
    _validate_matrix("colors", colors, 3, dtype=torch.float64, minimum_rows=5)
    if beta.shape != (3, 3) or beta.dtype != torch.float64:
        raise ValueError("beta must be a float64 tensor with shape (3,3)")
    if not bool(torch.isfinite(beta).all()):
        raise ValueError("beta must be finite")
    residual = colors - affine_design(means, height, width) @ beta
    output = pixel_design(height, width, dtype=torch.float64) @ beta + weights @ residual
    return output.reshape(int(height), int(width), 3)


def _neighbor_graph(
    means: torch.Tensor,
    height: int,
    width: int,
    *,
    neighbor_indices: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the frozen directed 4-NN graph and coordinate-pair ledger."""

    design = affine_design(means, height, width)
    coordinates = design[:, 1:3]
    count = int(coordinates.shape[0])
    if count <= NEIGHBOR_COUNT:
        raise ValueError("neighbor graph requires more than four samples")
    coordinate_array = coordinates.detach().cpu().numpy()
    if np.unique(coordinate_array, axis=0).shape[0] != count:
        raise ValueError("neighbor graph requires unique decoded mean coordinates")
    canonical_rows: list[np.ndarray] = []
    for source in range(count):
        difference = coordinate_array - coordinate_array[source]
        distance = np.sum(difference * difference, axis=1)
        distance[source] = np.inf
        # Distance is primary. Decoded x/y provide a permutation-invariant exact-tie break.
        order = np.lexsort((coordinate_array[:, 1], coordinate_array[:, 0], distance))
        canonical_rows.append(order[:NEIGHBOR_COUNT])
    canonical = torch.from_numpy(np.stack(canonical_rows).astype(np.int64, copy=False))
    if neighbor_indices is None:
        selected = canonical
    else:
        if neighbor_indices.shape != (count, NEIGHBOR_COUNT):
            raise ValueError("neighbor_indices must have shape (N,4)")
        if neighbor_indices.dtype != torch.int64 or neighbor_indices.device.type != "cpu":
            raise ValueError("neighbor_indices must be a CPU int64 tensor")
        if bool((neighbor_indices < 0).any()) or bool((neighbor_indices >= count).any()):
            raise ValueError("neighbor_indices are outside the sample range")
        if bool((neighbor_indices == torch.arange(count)[:, None]).any()):
            raise ValueError("neighbor graph may not contain self edges")
        if any(
            torch.unique(neighbor_indices[row]).numel() != NEIGHBOR_COUNT for row in range(count)
        ):
            raise ValueError("neighbor graph rows must contain four unique neighbors")
        if not torch.equal(neighbor_indices, canonical):
            raise ValueError("supplied neighbor graph is not the canonical decoded-mean 4-NN graph")
        selected = neighbor_indices
    sources = coordinates[:, None, :].expand(-1, NEIGHBOR_COUNT, -1)
    destinations = coordinates[selected]
    coordinate_pairs = torch.cat((sources, destinations), dim=2).reshape(-1, 4)
    return selected, coordinate_pairs


def smootherstep(value: torch.Tensor) -> torch.Tensor:
    """C2 unit-interval smootherstep with the frozen endpoint clamp."""

    clipped = torch.clamp(value, 0.0, 1.0)
    return clipped**3 * (clipped * (clipped * 6.0 - 15.0) + 10.0)


def discontinuity_confidence(
    means: torch.Tensor,
    colors: torch.Tensor,
    trend_residual: torch.Tensor,
    height: int,
    width: int,
    *,
    neighbor_indices: torch.Tensor | None = None,
) -> ConfidenceResult:
    """Compute the frozen log-mean-exp local-jump score and C2 confidence."""

    _validate_matrix("means", means, 2, dtype=torch.float64, minimum_rows=5)
    _validate_matrix("colors", colors, 3, dtype=torch.float64, minimum_rows=5)
    _validate_matrix("trend_residual", trend_residual, 3, dtype=torch.float64, minimum_rows=5)
    if means.shape[0] != colors.shape[0] or colors.shape != trend_residual.shape:
        raise ValueError("means, colors, and trend_residual must have aligned rows")
    neighbors, coordinate_pairs = _neighbor_graph(
        means, height, width, neighbor_indices=neighbor_indices
    )
    differences = trend_residual[:, None, :] - trend_residual[neighbors]
    smooth_absolute = torch.sqrt(torch.square(differences) + EDGE_ABS_EPSILON**2)
    centered = colors - torch.mean(colors, dim=0, keepdim=True)
    color_scale = torch.sqrt(torch.mean(torch.square(centered), dim=0) + COLOR_SCALE_FLOOR**2)
    normalized = smooth_absolute.reshape(-1, 3) / color_scale[None, :]
    scaled = normalized / SMOOTH_MAX_TEMPERATURE
    score = SMOOTH_MAX_TEMPERATURE * (
        torch.logsumexp(scaled, dim=0) - math.log(int(scaled.shape[0]))
    )
    unit = (score - CONFIDENCE_LOW) / (CONFIDENCE_HIGH - CONFIDENCE_LOW)
    alpha = 1.0 - smootherstep(unit)
    if not bool(torch.isfinite(score).all() and torch.isfinite(alpha).all()):
        raise RuntimeError("discontinuity confidence produced non-finite values")
    return ConfidenceResult(
        alpha=alpha,
        score=score,
        color_scale=color_scale,
        neighbor_indices=neighbors,
        edge_coordinate_pairs=coordinate_pairs,
    )


def derive_lift(
    means: torch.Tensor,
    colors: torch.Tensor,
    height: int,
    width: int,
    *,
    neighbor_indices: torch.Tensor | None = None,
) -> LiftState:
    """Derive robust plane, smooth confidence, and residual from decoded state only."""

    plane = robust_plane_irls(means, colors, height, width)
    confidence = discontinuity_confidence(
        means,
        colors,
        plane.residual,
        height,
        width,
        neighbor_indices=neighbor_indices,
    )
    beta = plane.beta * confidence.alpha.reshape(1, 3)
    residual = colors - affine_design(means, height, width) @ beta
    if not bool(torch.isfinite(beta).all() and torch.isfinite(residual).all()):
        raise RuntimeError("decoder-synchronized lift produced non-finite state")
    return LiftState(
        beta=beta,
        residual=residual,
        unconstrained_beta=plane.beta,
        confidence=confidence,
        robust_plane=plane,
    )


def dense_weight_render(
    weights: torch.Tensor,
    colors: torch.Tensor,
    height: int,
    width: int,
    *,
    lift: bool,
    means: torch.Tensor | None = None,
    neighbor_indices: torch.Tensor | None = None,
) -> tuple[torch.Tensor, LiftState | None]:
    """Render a supplied dense positive normalized weight matrix."""

    _validate_matrix("colors", colors, 3, dtype=torch.float64, minimum_rows=5)
    expected_pixels = int(height) * int(width)
    if weights.shape != (expected_pixels, colors.shape[0]):
        raise ValueError("weights must have shape (H*W,N)")
    if weights.dtype != torch.float64 or weights.device.type != "cpu":
        raise ValueError("weights must be a CPU float64 tensor")
    if not bool(torch.isfinite(weights).all()) or bool((weights < 0.0).any()):
        raise ValueError("weights must be finite and nonnegative")
    if not lift:
        return (weights @ colors).reshape(int(height), int(width), 3), None
    if means is None:
        raise ValueError("lift rendering requires decoded means")
    state = derive_lift(
        means,
        colors,
        height,
        width,
        neighbor_indices=neighbor_indices,
    )
    output = (
        pixel_design(height, width, dtype=torch.float64) @ state.beta + weights @ state.residual
    )
    return output.reshape(int(height), int(width), 3), state


def reference_render_lift(
    means: torch.Tensor,
    log_scales: torch.Tensor,
    rotations: torch.Tensor,
    colors: torch.Tensor,
    radii: torch.Tensor,
    height: int,
    width: int,
) -> LiftRender:
    """Independent float64 dense-oracle lift render from decoded ordinary state."""

    state = derive_lift(means, colors, height, width)
    conics = affine_core.conics_from_parameters(log_scales, rotations)
    base = affine_core.reference_render_residual(
        means,
        conics,
        state.residual,
        state.beta[1:3],
        radii,
        height,
        width,
    )
    output = base.output + state.beta[0]
    return LiftRender(output=output, base=base, state=state)


def candidate_render_lift(
    means: torch.Tensor,
    log_scales: torch.Tensor,
    rotations: torch.Tensor,
    colors: torch.Tensor,
    radii: torch.Tensor,
    height: int,
    width: int,
) -> LiftRender:
    """Production-order float32 render after one float64 decoder derivation and cast."""

    _validate_matrix("means", means, 2, dtype=torch.float32, minimum_rows=5)
    _validate_matrix("colors", colors, 3, dtype=torch.float32, minimum_rows=5)
    state64 = derive_lift(means.to(torch.float64), colors.to(torch.float64), height, width)
    beta32 = state64.beta.to(torch.float32)
    residual32 = state64.residual.to(torch.float32)
    conics = affine_core.conics_from_parameters(log_scales, rotations)
    base = affine_core.candidate_render_residual(
        means,
        conics,
        residual32,
        beta32[1:3],
        radii,
        height,
        width,
    )
    output = base.output + beta32[0]
    return LiftRender(output=output, base=base, state=state64)


__all__ = [
    "COLOR_SCALE_FLOOR",
    "CONFIDENCE_HIGH",
    "CONFIDENCE_LOW",
    "ConfidenceResult",
    "DesignDiagnostics",
    "EDGE_ABS_EPSILON",
    "EffectiveOperatorDiagnostics",
    "IRLS_ITERATIONS",
    "LiftRender",
    "LiftState",
    "MAXIMUM_DESIGN_CONDITION",
    "NEIGHBOR_COUNT",
    "OrderStatisticTrace",
    "ROBUST_MAD_FACTOR",
    "ROBUST_SCALE_FLOOR",
    "RobustPlaneResult",
    "SMOOTH_MAX_TEMPERATURE",
    "affine_design",
    "candidate_render_lift",
    "dense_weight_render",
    "dense_effective_operator_diagnostics",
    "dense_render_with_beta",
    "derive_lift",
    "diagnose_design",
    "discontinuity_confidence",
    "pixel_design",
    "reference_render_lift",
    "robust_plane_irls",
    "robust_plane_order_statistic_trace",
    "ordinary_plane_qr",
    "smootherstep",
]
