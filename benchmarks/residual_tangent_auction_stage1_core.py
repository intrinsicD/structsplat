"""Numerical core for the BENCH-009 Stage-1 residual tangent auction.

This module deliberately contains no target manifest, parent fitting, range-basis construction,
scientific gate, or result writer.  It provides the small renderer-local operations that those
layers need while keeping three quantities separate:

* dimensionless current-grammar parameter steps;
* leak-safe six-coefficient fits (discovery fit, held-out score); and
* exact finite packet realization under the normalized renderer.

The affine and carrier packets have six optimized RGB coefficients.  The two-birth packet has
two fixed geometries and six optimized RGB coefficients.  Packet-field counts are diagnostic;
nothing in this module is a byte, BPP, codec, or compression claim.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Callable, Sequence

import torch

from structsplat.gaussians import GaussianField
from structsplat.render import render_field


NORMALIZED_EPS = 1e-8
AFFINE_COEFFICIENT_ORDER = ("x_r", "x_g", "x_b", "y_r", "y_g", "y_b")
CARRIER_COEFFICIENT_ORDER = (
    "sin_r",
    "sin_g",
    "sin_b",
    "cos_r",
    "cos_g",
    "cos_b",
)
BIRTH_COEFFICIENT_ORDER = (
    "birth0_r",
    "birth0_g",
    "birth0_b",
    "birth1_r",
    "birth1_g",
    "birth1_b",
)


@dataclass(frozen=True)
class DimensionlessScaleSpec:
    """Frozen unit-fair chart for current-grammar and extension coefficients.

    Mean coordinates live in the parent's local principal-axis frame.  All remaining
    coordinates, including affine/carrier/birth RGB, use one physical unit directly.
    """

    mean_chart: str = "R_parent_diag_parent_scales"
    log_scale_units: float = 1.0
    rotation_radians: float = 1.0
    display_linear_rgb: float = 1.0

    def validate(self) -> None:
        if self.mean_chart != "R_parent_diag_parent_scales":
            raise ValueError("mean chart must use frozen parent principal axes")
        if (
            self.log_scale_units != 1.0
            or self.rotation_radians != 1.0
            or self.display_linear_rgb != 1.0
        ):
            raise ValueError("log-scale, rotation, and RGB must each use one physical unit")


@dataclass(frozen=True)
class FrozenPixelSupport:
    """Exact parent pixel membership used only by linearization diagnostics."""

    height: int
    width: int
    sigma_cutoff: float
    radii: torch.Tensor
    membership: torch.Tensor

    def validate_for(self, field: GaussianField) -> None:
        if self.radii.shape != (field.n, 2):
            raise ValueError("frozen support radii do not match the field")
        if self.membership.shape != (field.n, self.height, self.width):
            raise ValueError("frozen support membership does not match the field/image")


@dataclass(frozen=True)
class SupportMembershipChange:
    """Exact native-vs-parent AABB membership delta for current-grammar rows."""

    added_count: int
    removed_count: int
    xor_count: int
    parent_membership_sha256: str
    native_membership_sha256: str

    def as_record(self) -> dict[str, int | str]:
        return {
            "added_count": self.added_count,
            "removed_count": self.removed_count,
            "xor_count": self.xor_count,
            "parent_membership_sha256": self.parent_membership_sha256,
            "native_membership_sha256": self.native_membership_sha256,
        }


@dataclass(frozen=True)
class NoOpacityParameterization:
    """Frozen-support, no-opacity normalized-render parameterization."""

    template: GaussianField
    height: int
    width: int
    sigma_cutoff: float
    render_chunk: int
    physical_vector: torch.Tensor
    parent_mean_chart: torch.Tensor
    frozen_support: FrozenPixelSupport
    scale_spec: DimensionlessScaleSpec

    @property
    def parameter_count(self) -> int:
        return int(self.physical_vector.numel())

    def unpack_physical(self, vector: torch.Tensor) -> GaussianField:
        if vector.shape != self.physical_vector.shape:
            raise ValueError("physical parameter vector has the wrong shape")
        n = self.template.n
        cursor = 0

        def take(count: int) -> torch.Tensor:
            nonlocal cursor
            value = vector[cursor : cursor + count]
            cursor += count
            return value

        means = take(2 * n).reshape(n, 2)
        log_scales = take(2 * n).reshape(n, 2)
        rotations = take(n)
        colors = take(3 * n).reshape(n, 3)
        if cursor != self.parameter_count:
            raise RuntimeError("internal no-opacity parameter layout mismatch")
        return GaussianField(
            means,
            log_scales,
            rotations,
            colors,
            None,
            self.template.scale_max,
            None,
            self.template.background_mask,
            self.template.filter_variance,
        )

    def field_at(self, dimensionless_step: torch.Tensor) -> GaussianField:
        if dimensionless_step.shape != self.physical_vector.shape:
            raise ValueError("dimensionless parameter step has the wrong shape")
        n = self.template.n
        cursor = 0
        local_mean_step = dimensionless_step[cursor : cursor + 2 * n].reshape(n, 2)
        cursor += 2 * n
        log_scale_step = dimensionless_step[cursor : cursor + 2 * n].reshape(n, 2)
        cursor += 2 * n
        rotation_step = dimensionless_step[cursor : cursor + n]
        cursor += n
        color_step = dimensionless_step[cursor : cursor + 3 * n].reshape(n, 3)
        cursor += 3 * n
        if cursor != self.parameter_count:
            raise RuntimeError("internal dimensionless coordinate layout mismatch")
        means = self.template.means + torch.einsum(
            "nij,nj->ni", self.parent_mean_chart, local_mean_step
        )
        return GaussianField(
            means,
            self.template.log_scales + log_scale_step,
            self.template.rotations + rotation_step,
            self.template.colors + color_step,
            None,
            self.template.scale_max,
            None,
            self.template.background_mask,
            self.template.filter_variance,
        )

    @property
    def frozen_radii(self) -> torch.Tensor:
        """Compatibility view; membership, not radii alone, defines the frozen support."""

        return self.frozen_support.radii

    def render_step(self, dimensionless_step: torch.Tensor) -> torch.Tensor:
        field = self.field_at(dimensionless_step)
        return render_normalized_frozen_support(
            field,
            self.frozen_support,
        )

    def flat_render_function(self) -> Callable[[torch.Tensor], torch.Tensor]:
        return lambda step: self.render_step(step).reshape(-1)


@dataclass(frozen=True)
class CrossFitSixResult:
    """Discovery-only coefficient fit and held-out-only score."""

    coefficients: torch.Tensor
    trust_scale: float
    discovery_rank: int
    discovery_condition: float | None
    discovery_base_mse: float
    discovery_post_mse: float
    heldout_base_mse: float
    heldout_post_mse: float
    heldout_mse_gain: float


@dataclass(frozen=True)
class AllPixelSixOracle:
    """Separately labeled all-pixel fit; never a cross-fit score."""

    coefficients: torch.Tensor
    trust_scale: float
    rank: int
    condition: float | None
    base_mse: float
    post_mse: float
    mse_gain: float


@dataclass(frozen=True)
class SixCoefficientEvaluation:
    crossfit: CrossFitSixResult
    all_pixel_oracle: AllPixelSixOracle


@dataclass(frozen=True)
class BirthGeometry:
    mean_xy: tuple[float, float]
    scale_xy: tuple[float, float]
    angle_radians: float

    def validate(self) -> None:
        values = (*self.mean_xy, *self.scale_xy, self.angle_radians)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("birth geometry must be finite")
        if self.scale_xy[0] <= 0.0 or self.scale_xy[1] <= 0.0:
            raise ValueError("birth scales must be positive")


@dataclass(frozen=True)
class TwoBirthPacket:
    """Exact finite normalized action for two fixed-geometry births."""

    base_render: torch.Tensor
    zero_color_render: torch.Tensor
    offset: torch.Tensor
    design: torch.Tensor
    parent_denominator: torch.Tensor
    common_denominator: torch.Tensor
    birth_weights: torch.Tensor
    geometries: tuple[BirthGeometry, BirthGeometry]
    sigma_cutoff: float
    render_chunk: int
    frozen_parent_support: FrozenPixelSupport
    birth_support: FrozenPixelSupport


def _validate_no_opacity_field(field: GaussianField) -> None:
    if field.opacities is not None:
        raise ValueError("BENCH-009 Stage 1 primary parameterization forbids opacity")
    if field.color_grads is not None:
        raise ValueError("current-grammar parameterization requires constant colors")
    tensors = (field.means, field.log_scales, field.rotations, field.colors)
    if any(not bool(torch.isfinite(value).all()) for value in tensors):
        raise ValueError("field parameters must be finite")
    if field.n <= 0:
        raise ValueError("field must contain at least one Gaussian")


def freeze_pixel_support(
    field: GaussianField,
    height: int,
    width: int,
    *,
    sigma_cutoff: float = 3.0,
) -> FrozenPixelSupport:
    """Freeze exact AABB membership at the parent, including rounded centers."""

    _validate_no_opacity_field(field)
    if height <= 0 or width <= 0:
        raise ValueError("support dimensions must be positive")
    radii = field.radii(sigma_cutoff).detach().clone()
    device = field.means.device
    yy, xx = torch.meshgrid(
        torch.arange(height, device=device),
        torch.arange(width, device=device),
        indexing="ij",
    )
    ix = torch.round(field.means[:, 0].detach()).to(torch.long)
    iy = torch.round(field.means[:, 1].detach()).to(torch.long)
    x0 = (ix - radii[:, 0]).clamp(min=0)
    x1 = (ix + radii[:, 0]).clamp(max=width - 1)
    y0 = (iy - radii[:, 1]).clamp(min=0)
    y1 = (iy + radii[:, 1]).clamp(max=height - 1)
    membership = (
        (xx.unsqueeze(0) >= x0[:, None, None])
        & (xx.unsqueeze(0) <= x1[:, None, None])
        & (yy.unsqueeze(0) >= y0[:, None, None])
        & (yy.unsqueeze(0) <= y1[:, None, None])
    )
    return FrozenPixelSupport(
        height=int(height),
        width=int(width),
        sigma_cutoff=float(sigma_cutoff),
        radii=radii,
        membership=membership.detach().clone(),
    )


def pixel_membership_sha256(membership: torch.Tensor) -> str:
    """Hash exact boolean membership bytes with an explicit shape prefix."""

    value = membership.detach().to(device="cpu", dtype=torch.bool).contiguous()
    digest = hashlib.sha256()
    digest.update(str(tuple(value.shape)).encode("ascii"))
    digest.update(b"\0")
    digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def support_membership_change_summary(
    parent_support: FrozenPixelSupport,
    updated_field: GaussianField,
) -> SupportMembershipChange:
    """Report added, removed, XOR, and exact membership hashes separately."""

    updated = freeze_pixel_support(
        updated_field,
        parent_support.height,
        parent_support.width,
        sigma_cutoff=parent_support.sigma_cutoff,
    )
    parent_support.validate_for(updated_field)
    parent = parent_support.membership.to(device=updated.membership.device)
    native = updated.membership
    added = native & ~parent
    removed = parent & ~native
    return SupportMembershipChange(
        added_count=int(added.sum()),
        removed_count=int(removed.sum()),
        xor_count=int((added | removed).sum()),
        parent_membership_sha256=pixel_membership_sha256(parent),
        native_membership_sha256=pixel_membership_sha256(native),
    )


def support_membership_change_count(
    parent_support: FrozenPixelSupport,
    updated_field: GaussianField,
) -> int:
    """Count row/pixel membership changes made by the native recentered AABBs."""

    return support_membership_change_summary(parent_support, updated_field).xor_count


def _parent_mean_chart(field: GaussianField) -> torch.Tensor:
    """Return frozen ``R(theta_parent) diag(sx_parent, sy_parent)`` matrices."""

    scales = field.scales().detach()
    cosine = torch.cos(field.rotations.detach())
    sine = torch.sin(field.rotations.detach())
    chart = torch.zeros(
        field.n,
        2,
        2,
        device=field.means.device,
        dtype=field.means.dtype,
    )
    chart[:, 0, 0] = cosine * scales[:, 0]
    chart[:, 1, 0] = sine * scales[:, 0]
    chart[:, 0, 1] = -sine * scales[:, 1]
    chart[:, 1, 1] = cosine * scales[:, 1]
    return chart


def make_no_opacity_parameterization(
    field: GaussianField,
    height: int,
    width: int,
    *,
    sigma_cutoff: float = 3.0,
    render_chunk: int = 512,
    scale_spec: DimensionlessScaleSpec | None = None,
) -> NoOpacityParameterization:
    """Pack a field and freeze its discrete support for local JVPs.

    Mean steps are local principal-axis coordinates mapped through the frozen parent
    ``R diag(sx,sy)`` chart. Log-scale, rotation, and display-linear RGB use unit scale.
    """

    _validate_no_opacity_field(field)
    if height <= 0 or width <= 0:
        raise ValueError("render dimensions must be positive")
    if not math.isfinite(sigma_cutoff) or sigma_cutoff <= 0.0:
        raise ValueError("sigma_cutoff must be finite and positive")
    if render_chunk <= 0:
        raise ValueError("render_chunk must be positive")
    spec = scale_spec or DimensionlessScaleSpec()
    spec.validate()
    pieces = (
        field.means.reshape(-1),
        field.log_scales.reshape(-1),
        field.rotations.reshape(-1),
        field.colors.reshape(-1),
    )
    frozen_support = freeze_pixel_support(
        field,
        height,
        width,
        sigma_cutoff=sigma_cutoff,
    )
    return NoOpacityParameterization(
        template=field.detached(),
        height=int(height),
        width=int(width),
        sigma_cutoff=float(sigma_cutoff),
        render_chunk=int(render_chunk),
        physical_vector=torch.cat(pieces).detach().clone(),
        parent_mean_chart=_parent_mean_chart(field),
        frozen_support=frozen_support,
        scale_spec=spec,
    )


def render_normalized(
    field: GaussianField,
    height: int,
    width: int,
    *,
    sigma_cutoff: float = 3.0,
    render_chunk: int = 512,
    radii: torch.Tensor | None = None,
    color_grads: torch.Tensor | None = None,
) -> torch.Tensor:
    """Render the repository's exact normalized reference equation."""

    if field.opacities is not None:
        raise ValueError("this Stage-1 core supports no-opacity parents only")
    used_radii = field.radii(sigma_cutoff) if radii is None else radii
    return render_field(
        field.means,
        field.conics(),
        field.colors,
        used_radii,
        int(height),
        int(width),
        chunk=int(render_chunk),
        mode="normalized",
        opacities=None,
        scales=field.effective_scales(),
        rotations=field.rotations,
        sigma_cutoff=float(sigma_cutoff),
        color_grads=color_grads,
    )


def render_normalized_frozen_support(
    field: GaussianField,
    support: FrozenPixelSupport,
    *,
    color_grads: torch.Tensor | None = None,
) -> torch.Tensor:
    """Differentiable normalized render on exact parent-frozen pixel membership.

    This is the linearization/projector chart. Finite realized actions must use
    :func:`render_normalized`, whose native AABBs recenter and resize with the updated field.
    """

    _validate_no_opacity_field(field)
    support.validate_for(field)
    height, width = support.height, support.width
    device, dtype = field.means.device, field.means.dtype
    yy, xx = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype),
        torch.arange(width, device=device, dtype=dtype),
        indexing="ij",
    )
    dx = xx.unsqueeze(0) - field.means[:, 0, None, None]
    dy = yy.unsqueeze(0) - field.means[:, 1, None, None]
    conics = field.conics()
    q = (
        conics[:, 0, None, None] * dx.square()
        + 2.0 * conics[:, 1, None, None] * dx * dy
        + conics[:, 2, None, None] * dy.square()
    )
    weights = torch.where(
        support.membership.to(device=device),
        torch.exp(-0.5 * q),
        torch.zeros_like(q),
    )
    pixel_colors = field.colors[:, None, None, :].expand(-1, height, width, -1)
    if color_grads is not None:
        if color_grads.shape != (field.n, 2, 3):
            raise ValueError("color_grads must have shape (N,2,3)")
        angle = field.rotations[:, None, None]
        cosine, sine = torch.cos(angle), torch.sin(angle)
        scales = field.effective_scales().clamp_min(1e-6)
        local_x = (
            cosine * dx + sine * dy
        ) / scales[:, 0, None, None]
        local_y = (
            -sine * dx + cosine * dy
        ) / scales[:, 1, None, None]
        pixel_colors = pixel_colors + (
            color_grads[:, 0, None, None, :] * local_x[..., None]
            + color_grads[:, 1, None, None, :] * local_y[..., None]
        )
    numerator = (weights[..., None] * pixel_colors).sum(dim=0)
    denominator = weights.sum(dim=0)
    return numerator / (denominator[..., None] + NORMALIZED_EPS)


def centered_fd_jvp_validation(
    parameterization: NoOpacityParameterization,
    *,
    direction: torch.Tensor | None = None,
    seed: int = 0,
    epsilon: float = 1e-4,
) -> dict[str, float]:
    """Compare a centered finite difference with an autograd JVP."""

    if not math.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("epsilon must be finite and positive")
    origin = torch.zeros_like(parameterization.physical_vector)
    if direction is None:
        generator = torch.Generator(device=origin.device).manual_seed(int(seed))
        direction = torch.randn(
            origin.shape,
            generator=generator,
            device=origin.device,
            dtype=origin.dtype,
        )
    if direction.shape != origin.shape or not bool(torch.isfinite(direction).all()):
        raise ValueError("JVP direction is invalid")
    direction = direction / torch.linalg.vector_norm(direction).clamp_min(1e-15)
    function = parameterization.flat_render_function()
    _, jvp = torch.autograd.functional.jvp(
        function,
        origin,
        direction,
        create_graph=False,
        strict=True,
    )
    positive = function(epsilon * direction)
    negative = function(-epsilon * direction)
    finite_difference = (positive - negative) / (2.0 * epsilon)
    difference = finite_difference - jvp
    return {
        "epsilon": float(epsilon),
        "max_abs": float(difference.abs().max()),
        "relative_l2": float(
            torch.linalg.vector_norm(difference)
            / torch.linalg.vector_norm(finite_difference).clamp_min(1e-15)
        ),
    }


def checkerboard_fold_masks(
    height: int,
    width: int,
    fold: int,
    *,
    tile_size: int = 8,
    channels: int = 3,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return flattened discovery and held-out masks for 8x8 checkerboard tiles."""

    if height <= 0 or width <= 0 or tile_size <= 0 or channels <= 0:
        raise ValueError("mask dimensions, tile_size, and channels must be positive")
    if fold not in (0, 1):
        raise ValueError("the frozen checkerboard protocol has exactly folds 0 and 1")
    yy = torch.arange(height).reshape(height, 1)
    xx = torch.arange(width).reshape(1, width)
    parity = ((yy // tile_size) + (xx // tile_size)) % 2
    heldout_pixels = parity == int(fold)
    discovery_pixels = ~heldout_pixels
    discovery = discovery_pixels.unsqueeze(-1).expand(height, width, channels).reshape(-1)
    heldout = heldout_pixels.unsqueeze(-1).expand(height, width, channels).reshape(-1)
    return discovery, heldout


def uniform_linf_trust_scale(
    coefficients: torch.Tensor,
    radius: float,
) -> tuple[torch.Tensor, float]:
    """Apply one uniform scale so ``max(abs(coefficients)) <= radius``."""

    if not math.isfinite(radius) or radius <= 0.0:
        raise ValueError("trust radius must be finite and positive")
    if not bool(torch.isfinite(coefficients).all()):
        raise ValueError("coefficients must be finite")
    if coefficients.numel() == 0:
        return coefficients.clone(), 1.0
    maximum = float(coefficients.abs().max())
    factor = min(1.0, float(radius) / max(maximum, 1e-30))
    return coefficients * factor, factor


def _mask_vector(mask: torch.Tensor, rows: int, *, device: torch.device) -> torch.Tensor:
    value = torch.as_tensor(mask, device=device, dtype=torch.bool).reshape(-1)
    if int(value.numel()) != rows:
        raise ValueError("mask length does not match design rows")
    if not bool(value.any()):
        raise ValueError("mask selects no rows")
    return value


def _solve_six(
    design: torch.Tensor,
    rhs: torch.Tensor,
    mask: torch.Tensor,
    *,
    ridge: float,
    rcond: float,
    trust_radius: float | None,
) -> tuple[torch.Tensor, float, int, float | None]:
    if design.ndim != 2 or design.shape[1] != 6:
        raise ValueError("six-coefficient design must have shape (rows, 6)")
    if rhs.ndim != 1 or rhs.shape[0] != design.shape[0]:
        raise ValueError("rhs shape does not match design")
    if not bool(torch.isfinite(design).all()) or not bool(torch.isfinite(rhs).all()):
        raise ValueError("design and rhs must be finite")
    if not math.isfinite(ridge) or ridge < 0.0:
        raise ValueError("ridge must be finite and non-negative")
    if not math.isfinite(rcond) or rcond < 0.0:
        raise ValueError("rcond must be finite and non-negative")
    selected = _mask_vector(mask, design.shape[0], device=design.device)
    matrix = design[selected]
    target = rhs[selected]
    column_scales = torch.linalg.vector_norm(matrix, dim=0).clamp_min(1e-15)
    scaled = matrix / column_scales
    u, singular, vh = torch.linalg.svd(scaled, full_matrices=False)
    cutoff = float(rcond) * float(singular[0]) if singular.numel() else 0.0
    keep = singular > cutoff
    rank = int(keep.sum())
    if rank == 0:
        coefficients = rhs.new_zeros((6,))
        condition = None
    else:
        kept_singular = singular[keep]
        projected = u[:, keep].T @ target
        if ridge == 0.0:
            inverse = kept_singular.reciprocal()
        else:
            inverse = kept_singular / (kept_singular.square() + float(ridge) ** 2)
        scaled_coefficients = vh[keep].T @ (inverse * projected)
        coefficients = scaled_coefficients / column_scales
        condition = float(kept_singular[0] / kept_singular[-1])
    trust_scale = 1.0
    if trust_radius is not None:
        coefficients, trust_scale = uniform_linf_trust_scale(coefficients, trust_radius)
    return coefficients, trust_scale, rank, condition


def _masked_mse(value: torch.Tensor, mask: torch.Tensor) -> float:
    selected = _mask_vector(mask, value.numel(), device=value.device)
    return float(value[selected].square().mean())


def evaluate_six_coefficient_action(
    design: torch.Tensor,
    residual: torch.Tensor,
    discovery_mask: torch.Tensor,
    heldout_mask: torch.Tensor,
    *,
    ridge: float = 0.0,
    rcond: float = 1e-7,
    trust_radius: float | None = None,
) -> SixCoefficientEvaluation:
    """Fit on discovery, score on held-out, and emit a separate all-pixel oracle."""

    residual = residual.reshape(-1)
    discovery = _mask_vector(discovery_mask, design.shape[0], device=design.device)
    heldout = _mask_vector(heldout_mask, design.shape[0], device=design.device)
    if bool((discovery & heldout).any()):
        raise ValueError("discovery and held-out masks overlap")
    if not bool((discovery | heldout).all()):
        raise ValueError("discovery and held-out masks do not cover all rows")
    coefficients, scale, rank, condition = _solve_six(
        design,
        residual,
        discovery,
        ridge=ridge,
        rcond=rcond,
        trust_radius=trust_radius,
    )
    remaining = residual - design @ coefficients
    crossfit = CrossFitSixResult(
        coefficients=coefficients,
        trust_scale=scale,
        discovery_rank=rank,
        discovery_condition=condition,
        discovery_base_mse=_masked_mse(residual, discovery),
        discovery_post_mse=_masked_mse(remaining, discovery),
        heldout_base_mse=_masked_mse(residual, heldout),
        heldout_post_mse=_masked_mse(remaining, heldout),
        heldout_mse_gain=(
            _masked_mse(residual, heldout) - _masked_mse(remaining, heldout)
        ),
    )
    all_pixels = torch.ones_like(discovery)
    oracle_coefficients, oracle_scale, oracle_rank, oracle_condition = _solve_six(
        design,
        residual,
        all_pixels,
        ridge=ridge,
        rcond=rcond,
        trust_radius=trust_radius,
    )
    oracle_remaining = residual - design @ oracle_coefficients
    oracle_base_mse = float(residual.square().mean())
    oracle_post_mse = float(oracle_remaining.square().mean())
    oracle = AllPixelSixOracle(
        coefficients=oracle_coefficients,
        trust_scale=oracle_scale,
        rank=oracle_rank,
        condition=oracle_condition,
        base_mse=oracle_base_mse,
        post_mse=oracle_post_mse,
        mse_gain=oracle_base_mse - oracle_post_mse,
    )
    return SixCoefficientEvaluation(crossfit=crossfit, all_pixel_oracle=oracle)


def dense_gaussian_weights(
    field: GaussianField,
    height: int,
    width: int,
    *,
    sigma_cutoff: float = 3.0,
    radii: torch.Tensor | None = None,
    support: FrozenPixelSupport | None = None,
) -> torch.Tensor:
    """Dense no-opacity weights matching the reference renderer's compact AABBs."""

    _validate_no_opacity_field(field)
    if support is not None and radii is not None:
        raise ValueError("pass either exact frozen support or radii, not both")
    if support is not None:
        support.validate_for(field)
        if (height, width) != (support.height, support.width):
            raise ValueError("frozen support image dimensions do not match")
        used_radii = support.radii
    else:
        used_radii = field.radii(sigma_cutoff) if radii is None else radii
    if used_radii.shape != (field.n, 2):
        raise ValueError("radii must have shape (N, 2)")
    device, dtype = field.means.device, field.means.dtype
    yy, xx = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype),
        torch.arange(width, device=device, dtype=dtype),
        indexing="ij",
    )
    dx = xx.unsqueeze(0) - field.means[:, 0, None, None]
    dy = yy.unsqueeze(0) - field.means[:, 1, None, None]
    conics = field.conics()
    q = (
        conics[:, 0, None, None] * dx.square()
        + 2.0 * conics[:, 1, None, None] * dx * dy
        + conics[:, 2, None, None] * dy.square()
    )
    if support is None:
        ix = torch.round(field.means[:, 0].detach()).to(torch.long)
        iy = torch.round(field.means[:, 1].detach()).to(torch.long)
        x0 = (ix - used_radii[:, 0]).clamp(min=0)
        x1 = (ix + used_radii[:, 0]).clamp(max=width - 1)
        y0 = (iy - used_radii[:, 1]).clamp(min=0)
        y1 = (iy + used_radii[:, 1]).clamp(max=height - 1)
        membership = (
            (xx.unsqueeze(0) >= x0[:, None, None])
            & (xx.unsqueeze(0) <= x1[:, None, None])
            & (yy.unsqueeze(0) >= y0[:, None, None])
            & (yy.unsqueeze(0) <= y1[:, None, None])
        )
    else:
        membership = support.membership.to(device=device)
    return torch.where(membership, torch.exp(-0.5 * q), torch.zeros_like(q))


def normalized_responsibilities(
    field: GaussianField,
    height: int,
    width: int,
    *,
    sigma_cutoff: float = 3.0,
    radii: torch.Tensor | None = None,
    support: FrozenPixelSupport | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-row normalized ownership and the raw common denominator."""

    weights = dense_gaussian_weights(
        field,
        height,
        width,
        sigma_cutoff=sigma_cutoff,
        radii=radii,
        support=support,
    )
    denominator = weights.sum(dim=0)
    return weights / (denominator.unsqueeze(0) + NORMALIZED_EPS), denominator


def _six_rgb_columns(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    if first.shape != second.shape or first.ndim != 2:
        raise ValueError("basis images must be aligned 2-D tensors")
    height, width = first.shape
    columns: list[torch.Tensor] = []
    for basis in (first, second):
        for channel in range(3):
            image = basis.new_zeros((height, width, 3))
            image[..., channel] = basis
            columns.append(image.reshape(-1))
    return torch.stack(columns, dim=1)


def affine_six_columns(
    field: GaussianField,
    gid: int,
    height: int,
    width: int,
    *,
    sigma_cutoff: float = 3.0,
    radii: torch.Tensor | None = None,
    support: FrozenPixelSupport | None = None,
) -> torch.Tensor:
    """Analytic x/y RGB columns for one inherited Gaussian support."""

    if gid < 0 or gid >= field.n:
        raise ValueError("affine parent index is out of range")
    responsibility, _ = normalized_responsibilities(
        field,
        height,
        width,
        sigma_cutoff=sigma_cutoff,
        radii=radii,
        support=support,
    )
    device, dtype = field.means.device, field.means.dtype
    yy, xx = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype),
        torch.arange(width, device=device, dtype=dtype),
        indexing="ij",
    )
    dx = xx - field.means[gid, 0]
    dy = yy - field.means[gid, 1]
    angle = field.rotations[gid]
    cosine, sine = torch.cos(angle), torch.sin(angle)
    scales = field.effective_scales()[gid].clamp_min(1e-6)
    local_x = (cosine * dx + sine * dy) / scales[0]
    local_y = (-sine * dx + cosine * dy) / scales[1]
    return _six_rgb_columns(
        responsibility[gid] * local_x,
        responsibility[gid] * local_y,
    )


def realize_affine_six(
    field: GaussianField,
    gid: int,
    coefficients: torch.Tensor,
    height: int,
    width: int,
    *,
    sigma_cutoff: float = 3.0,
    render_chunk: int = 512,
) -> torch.Tensor:
    """Native finite realization with recentered/recomputed AABBs."""

    coefficients = coefficients.reshape(-1)
    if coefficients.numel() != 6 or not bool(torch.isfinite(coefficients).all()):
        raise ValueError("affine coefficients must contain six finite values")
    gradients = torch.zeros(
        field.n,
        2,
        3,
        device=field.means.device,
        dtype=field.means.dtype,
    )
    gradients[gid] = coefficients.reshape(2, 3)
    return render_normalized(
        field,
        height,
        width,
        sigma_cutoff=sigma_cutoff,
        render_chunk=render_chunk,
        color_grads=gradients,
    )


def carrier_six_columns(
    field: GaussianField,
    gid: int,
    frequency_xy: tuple[float, float],
    height: int,
    width: int,
    *,
    phase_radians: float = 0.0,
    sigma_cutoff: float = 3.0,
    radii: torch.Tensor | None = None,
    support: FrozenPixelSupport | None = None,
) -> torch.Tensor:
    """Analytic Gaussian-windowed sin/cos RGB columns."""

    if gid < 0 or gid >= field.n:
        raise ValueError("carrier parent index is out of range")
    if not all(math.isfinite(value) for value in (*frequency_xy, phase_radians)):
        raise ValueError("carrier frequency and phase must be finite")
    responsibility, _ = normalized_responsibilities(
        field,
        height,
        width,
        sigma_cutoff=sigma_cutoff,
        radii=radii,
        support=support,
    )
    device, dtype = field.means.device, field.means.dtype
    yy, xx = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype),
        torch.arange(width, device=device, dtype=dtype),
        indexing="ij",
    )
    dx = xx - field.means[gid, 0]
    dy = yy - field.means[gid, 1]
    angle = field.rotations[gid]
    cosine, sine = torch.cos(angle), torch.sin(angle)
    scales = field.effective_scales()[gid].clamp_min(1e-6)
    local_x = (cosine * dx + sine * dy) / scales[0]
    local_y = (-sine * dx + cosine * dy) / scales[1]
    phase = (
        2.0
        * math.pi
        * (float(frequency_xy[0]) * local_x + float(frequency_xy[1]) * local_y)
        + float(phase_radians)
    )
    return _six_rgb_columns(
        responsibility[gid] * torch.sin(phase),
        responsibility[gid] * torch.cos(phase),
    )


def realize_carrier_six(
    field: GaussianField,
    gid: int,
    frequency_xy: tuple[float, float],
    coefficients: torch.Tensor,
    height: int,
    width: int,
    *,
    phase_radians: float = 0.0,
    sigma_cutoff: float = 3.0,
    render_chunk: int = 512,
) -> torch.Tensor:
    """Native-base carrier realization with recentered/recomputed AABBs."""

    coefficients = coefficients.reshape(-1)
    if coefficients.numel() != 6 or not bool(torch.isfinite(coefficients).all()):
        raise ValueError("carrier coefficients must contain six finite values")
    base = render_normalized(
        field,
        height,
        width,
        sigma_cutoff=sigma_cutoff,
        render_chunk=render_chunk,
    )
    columns = carrier_six_columns(
        field,
        gid,
        frequency_xy,
        height,
        width,
        phase_radians=phase_radians,
        sigma_cutoff=sigma_cutoff,
    )
    return base + (columns @ coefficients).reshape_as(base)


def _birth_field(
    geometries: Sequence[BirthGeometry],
    colors: torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> GaussianField:
    if len(geometries) != 2:
        raise ValueError("exact control requires exactly two births")
    for geometry in geometries:
        geometry.validate()
    colors = colors.reshape(2, 3).to(device=device, dtype=dtype)
    means = torch.tensor(
        [geometry.mean_xy for geometry in geometries], device=device, dtype=dtype
    )
    scales = torch.tensor(
        [geometry.scale_xy for geometry in geometries], device=device, dtype=dtype
    )
    rotations = torch.tensor(
        [geometry.angle_radians for geometry in geometries], device=device, dtype=dtype
    )
    return GaussianField(means, torch.log(scales), rotations, colors)


def _append_births_without_detach(
    parent: GaussianField,
    births: GaussianField,
) -> GaussianField:
    """Append fixed births while preserving gradients through parent tensors."""

    if parent.opacities is not None or births.opacities is not None:
        raise ValueError("Stage-1 two-birth control is no-opacity only")
    filter_variance = None
    if parent.filter_variance is not None:
        filter_variance = torch.cat(
            (
                parent.filter_variance,
                torch.zeros(
                    births.n,
                    device=parent.filter_variance.device,
                    dtype=parent.filter_variance.dtype,
                ),
            )
        )
    return GaussianField(
        torch.cat((parent.means, births.means)),
        torch.cat((parent.log_scales, births.log_scales)),
        torch.cat((parent.rotations, births.rotations)),
        torch.cat((parent.colors, births.colors)),
        None,
        None,
        None,
        None,
        filter_variance,
    )


def build_two_birth_packet(
    parent: GaussianField,
    geometries: Sequence[BirthGeometry],
    height: int,
    width: int,
    *,
    sigma_cutoff: float = 3.0,
    render_chunk: int = 512,
    frozen_parent_support: FrozenPixelSupport | None = None,
) -> TwoBirthPacket:
    """Build the exact common-denominator offset and six RGB columns."""

    _validate_no_opacity_field(parent)
    if len(geometries) != 2:
        raise ValueError("two-birth packet requires exactly two geometries")
    pair = (geometries[0], geometries[1])
    for geometry in pair:
        geometry.validate()
    if frozen_parent_support is None:
        frozen_parent_support = freeze_pixel_support(
            parent,
            height,
            width,
            sigma_cutoff=sigma_cutoff,
        )
    parent_weights = dense_gaussian_weights(
        parent,
        height,
        width,
        sigma_cutoff=sigma_cutoff,
        support=frozen_parent_support,
    )
    parent_denominator = parent_weights.sum(dim=0)
    parent_numerator = torch.einsum("nhw,nc->hwc", parent_weights, parent.colors)
    base_render = parent_numerator / (parent_denominator[..., None] + NORMALIZED_EPS)

    zero_colors = parent.colors.new_zeros((2, 3))
    birth_field = _birth_field(
        pair,
        zero_colors,
        device=parent.means.device,
        dtype=parent.means.dtype,
    )
    birth_support = freeze_pixel_support(
        birth_field,
        height,
        width,
        sigma_cutoff=sigma_cutoff,
    )
    birth_weights = dense_gaussian_weights(
        birth_field,
        height,
        width,
        sigma_cutoff=sigma_cutoff,
        support=birth_support,
    )
    common_denominator = parent_denominator + birth_weights.sum(dim=0) + NORMALIZED_EPS
    zero_color_render = parent_numerator / common_denominator[..., None]
    offset = (zero_color_render - base_render).reshape(-1)
    design = _six_rgb_columns(
        birth_weights[0] / common_denominator,
        birth_weights[1] / common_denominator,
    )
    return TwoBirthPacket(
        base_render=base_render,
        zero_color_render=zero_color_render,
        offset=offset,
        design=design,
        parent_denominator=parent_denominator,
        common_denominator=common_denominator,
        birth_weights=birth_weights,
        geometries=pair,
        sigma_cutoff=float(sigma_cutoff),
        render_chunk=int(render_chunk),
        frozen_parent_support=frozen_parent_support,
        birth_support=birth_support,
    )


def realize_two_birth_six(
    parent: GaussianField,
    packet: TwoBirthPacket,
    coefficients: torch.Tensor,
) -> torch.Tensor:
    """Direct normalized render of the two finite births with their six RGB values."""

    coefficients = coefficients.reshape(-1)
    if coefficients.numel() != 6 or not bool(torch.isfinite(coefficients).all()):
        raise ValueError("two-birth coefficients must contain six finite values")
    births = _birth_field(
        packet.geometries,
        coefficients.reshape(2, 3),
        device=parent.means.device,
        dtype=parent.means.dtype,
    )
    combined = _append_births_without_detach(parent, births)
    return render_normalized(
        combined,
        packet.base_render.shape[0],
        packet.base_render.shape[1],
        sigma_cutoff=packet.sigma_cutoff,
        render_chunk=packet.render_chunk,
    )


def solve_two_birth_colors(
    packet: TwoBirthPacket,
    target: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
    ridge: float = 0.0,
    rcond: float = 1e-7,
    trust_radius: float | None = None,
) -> torch.Tensor:
    """Solve the exact finite common-denominator RGB least-squares problem."""

    if target.shape != packet.base_render.shape:
        raise ValueError("two-birth target shape does not match packet")
    rhs = (target - packet.base_render).reshape(-1) - packet.offset
    selected = (
        torch.ones(packet.design.shape[0], device=packet.design.device, dtype=torch.bool)
        if mask is None
        else mask
    )
    coefficients, _, _, _ = _solve_six(
        packet.design,
        rhs,
        selected,
        ridge=ridge,
        rcond=rcond,
        trust_radius=trust_radius,
    )
    return coefficients


def packet_linear_realization(
    base_render: torch.Tensor,
    design: torch.Tensor,
    coefficients: torch.Tensor,
    *,
    offset: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply a six-column packet equation, with an optional finite offset."""

    flat_offset = (
        base_render.new_zeros(base_render.numel())
        if offset is None
        else offset.reshape(-1)
    )
    if design.shape != (base_render.numel(), 6):
        raise ValueError("packet design has the wrong shape")
    coefficients = coefficients.reshape(-1)
    if coefficients.numel() != 6:
        raise ValueError("packet coefficients must contain six values")
    return (
        base_render.reshape(-1) + flat_offset + design @ coefficients
    ).reshape_as(base_render)
