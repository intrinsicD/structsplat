"""Gauge-lifted residual color dipoles for the FIT-032 benchmark.

For a normalized weighted-sum renderer, split one row into two co-located children carrying
half of its opacity mass. The render is exactly unchanged. Giving the children opposite mean
offsets ``+/- delta`` and opposite color contrasts ``+/- a`` produces, to leading order,

    delta I(p) = ((grad_mu w_i(p) . delta) / D(p)) a.

The useful term is bilinear: displacement alone and color contrast alone are both null at the
co-located split. This module solves the local rank-one residual projection and realizes that
joint direction with one net row. It is benchmark code, not a production primitive.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from structsplat.config import FitConfig
from structsplat.fit import _raw_weight_map_field
from structsplat.gaussians import GaussianField
from structsplat.render import (
    _EPS,
    _element_budget,
    _flat_tile_slices,
    _support_weight,
    _tile_bounds,
    _tile_coords,
)


@dataclass(frozen=True)
class DipoleSelection:
    """Selected parent rows and their jointly initialized dipole parameters."""

    parents: torch.Tensor
    displacement: torch.Tensor
    contrast: torch.Tensor
    score: torch.Tensor
    unclipped_score: torch.Tensor
    color_clip: torch.Tensor
    support_scale: torch.Tensor
    candidate_count: int
    rejected_background: int
    rejected_mask: int
    rejected_degenerate: int


def _exact_logit(probability: torch.Tensor) -> torch.Tensor:
    if not bool(torch.isfinite(probability).all()):
        raise ValueError("opacity probabilities must be finite")
    if bool(((probability <= 0.0) | (probability >= 1.0)).any()):
        raise ValueError("dipole splitting requires opacity probabilities strictly in (0, 1)")
    return torch.log(probability) - torch.log1p(-probability)


def _weight_translation_terms(
    field: GaussianField,
    cfg: FitConfig,
    height: int,
    width: int,
    denominator: torch.Tensor,
    residual: torch.Tensor | None,
    mask: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Accumulate ``H.T H`` and ``H.T residual`` for every parent.

    ``H`` is the two-column image Jacobian of the parent's raw weight with respect to its
    translation, divided by the full normalized-renderer denominator. The support-fade constant
    has zero derivative wherever the faded weight is positive; its zero-clamp is respected by
    masking the derivative outside that active support.
    """

    device = field.means.device
    dtype = field.means.dtype
    conics = field.conics(cfg.aa_dilation)
    radii = field.radii(cfg.sigma_cutoff, cfg.aa_dilation)
    opacity = field.opacity_values()
    if opacity is None:
        raise ValueError("FIT-032 requires explicit opacity logits")

    gram = torch.zeros(field.n, 2, 2, device=device, dtype=dtype)
    cross = torch.zeros(field.n, 2, 3, device=device, dtype=dtype)
    x0, y0, tile_width, tile_size = _tile_bounds(
        field.means, radii, height, width
    )
    budget = _element_budget(cfg.render_chunk)
    denominator = denominator.reshape(-1).clamp_min(_EPS)

    for start, end in _flat_tile_slices(tile_size, budget):
        gid, px, py = _tile_coords(
            x0, y0, tile_width, tile_size, start, end, device
        )
        dx = px.to(dtype) - field.means[gid, 0]
        dy = py.to(dtype) - field.means[gid, 1]
        a = conics[gid, 0]
        b = conics[gid, 1]
        c = conics[gid, 2]
        q = a * dx.square() + 2.0 * b * dx * dy + c * dy.square()
        support = _support_weight(
            q,
            cfg.sigma_cutoff,
            cfg.support_fade,
            support_fade_alpha=1.0,
        )
        active = support > 0.0
        base = torch.exp(-0.5 * q) * opacity[gid] * active.to(dtype)
        flat = py * width + px
        inv_denominator = denominator[flat].reciprocal()
        hx = base * (a * dx + b * dy) * inv_denominator
        hy = base * (b * dx + c * dy) * inv_denominator
        if mask is not None:
            allowed = mask[py, px]
            hx = hx * allowed.to(dtype)
            hy = hy * allowed.to(dtype)

        gram[:, 0, 0].index_add_(0, gid, hx.square())
        gram[:, 0, 1].index_add_(0, gid, hx * hy)
        gram[:, 1, 1].index_add_(0, gid, hy.square())
        if residual is not None:
            local_residual = residual[py, px]
            cross[:, 0].index_add_(0, gid, hx[:, None] * local_residual)
            cross[:, 1].index_add_(0, gid, hy[:, None] * local_residual)

    gram[:, 1, 0] = gram[:, 0, 1]
    return gram, cross


def _support_stable_scales(
    field: GaussianField,
    displacement: torch.Tensor,
    cfg: FitConfig,
    height: int,
    width: int,
) -> torch.Tensor:
    """Maximum joint lift that preserves each row's discrete support topology."""

    device = field.means.device
    dtype = field.means.dtype
    conics = field.conics(cfg.aa_dilation)
    radii = field.radii(cfg.sigma_cutoff, cfg.aa_dilation)
    x0, y0, tile_width, tile_size = _tile_bounds(
        field.means, radii, height, width
    )
    crossing = torch.full(
        (field.n,), float("inf"), device=device, dtype=dtype
    )
    cutoff_square = float(cfg.sigma_cutoff) ** 2
    budget = _element_budget(cfg.render_chunk)
    for start, end in _flat_tile_slices(tile_size, budget):
        gid, px, py = _tile_coords(
            x0, y0, tile_width, tile_size, start, end, device
        )
        dx = px.to(dtype) - field.means[gid, 0]
        dy = py.to(dtype) - field.means[gid, 1]
        a = conics[gid, 0]
        b = conics[gid, 1]
        c = conics[gid, 2]
        q = a * dx.square() + 2.0 * b * dx * dy + c * dy.square()
        qdx = a * dx + b * dy
        qdy = b * dx + c * dy
        delta = displacement[gid]
        linear = (qdx * delta[:, 0] + qdy * delta[:, 1]).abs()
        quadratic = (
            a * delta[:, 0].square()
            + 2.0 * b * delta[:, 0] * delta[:, 1]
            + c * delta[:, 1].square()
        ).clamp_min(1e-20)
        active = _support_weight(
            q,
            cfg.sigma_cutoff,
            cfg.support_fade,
            support_fade_alpha=1.0,
        ) > 0.0
        inside_gap = (cutoff_square - q).clamp_min(0.0)
        inside_root = (
            torch.sqrt(linear.square() + quadratic * inside_gap) - linear
        ) / quadratic
        outside_gap = (q - cutoff_square).clamp_min(0.0)
        discriminant = linear.square() - quadratic * outside_gap
        outside_root = torch.where(
            discriminant >= 0.0,
            (linear - torch.sqrt(discriminant.clamp_min(0.0))) / quadratic,
            torch.full_like(linear, float("inf")),
        )
        local_crossing = torch.where(active, inside_root, outside_root).clamp_min(0.0)
        crossing.scatter_reduce_(
            0, gid, local_crossing, reduce="amin", include_self=True
        )

    # Keeping rounded centers unchanged keeps the renderer's clipped AABB unchanged too.
    rounded = torch.round(field.means)
    lower_room = field.means - (rounded - 0.5)
    upper_room = (rounded + 0.5) - field.means
    round_room = torch.minimum(lower_room, upper_room).clamp_min(0.0)
    axis_crossing = torch.where(
        displacement.abs() > 1e-20,
        round_room / displacement.abs().clamp_min(1e-20),
        torch.full_like(displacement, float("inf")),
    ).amin(dim=1)
    root_scale = (0.95 * torch.minimum(crossing, axis_crossing)).clamp(0.0, 1.0)
    return root_scale.square()


@torch.no_grad()
def translation_jacobian(
    field: GaussianField,
    parent: int,
    cfg: FitConfig,
    height: int,
    width: int,
) -> torch.Tensor:
    """Return the normalized-renderer weight-translation Jacobian ``(H,W,2)`` for one row."""

    parent = int(parent)
    if parent < 0 or parent >= field.n:
        raise IndexError(f"parent {parent} is outside field with {field.n} rows")
    denominator = _raw_weight_map_field(
        field,
        cfg,
        height,
        width,
        support_fade_alpha=1.0,
    )
    device = field.means.device
    dtype = field.means.dtype
    result = torch.zeros(height * width, 2, device=device, dtype=dtype)
    row = field.subset(torch.tensor([parent], device=device))
    conic = row.conics(cfg.aa_dilation)[0]
    radius = row.radii(cfg.sigma_cutoff, cfg.aa_dilation)
    opacity = row.opacity_values()
    if opacity is None:
        raise ValueError("FIT-032 requires explicit opacity logits")
    x0, y0, tile_width, tile_size = _tile_bounds(
        row.means, radius, height, width
    )
    gid, px, py = _tile_coords(
        x0, y0, tile_width, tile_size, 0, 1, device
    )
    dx = px.to(dtype) - row.means[gid, 0]
    dy = py.to(dtype) - row.means[gid, 1]
    a, b, c = conic
    q = a * dx.square() + 2.0 * b * dx * dy + c * dy.square()
    support = _support_weight(
        q,
        cfg.sigma_cutoff,
        cfg.support_fade,
        support_fade_alpha=1.0,
    )
    base = (
        torch.exp(-0.5 * q)
        * opacity[gid]
        * (support > 0.0).to(dtype)
    )
    flat = py * width + px
    inv_denominator = denominator[flat].clamp_min(_EPS).reciprocal()
    values = torch.stack(
        [
            base * (a * dx + b * dy) * inv_denominator,
            base * (b * dx + c * dy) * inv_denominator,
        ],
        dim=1,
    )
    result.index_add_(0, flat, values)
    return result.view(height, width, 2)


def _generalized_rank_one_solution(
    gram: torch.Tensor,
    cross: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the best pixel-space direction and its unconstrained SSE reduction."""

    trace = gram[:, 0, 0] + gram[:, 1, 1]
    ridge = 1e-6 * trace.clamp_min(1e-12) + 1e-12
    regularized = gram.clone()
    regularized[:, 0, 0] += ridge
    regularized[:, 1, 1] += ridge
    values, vectors = torch.linalg.eigh(regularized)
    inv_sqrt = (
        vectors
        @ torch.diag_embed(values.clamp_min(1e-20).rsqrt())
        @ vectors.transpose(1, 2)
    )
    cross_gram = cross @ cross.transpose(1, 2)
    whitened = inv_sqrt @ cross_gram @ inv_sqrt
    scores, score_vectors = torch.linalg.eigh(whitened)
    direction = (inv_sqrt @ score_vectors[:, :, -1, None]).squeeze(-1)
    score = scores[:, -1].clamp_min(0.0)

    # Eigenvectors have arbitrary sign. Canonicalize using their largest absolute coordinate.
    dominant = direction.abs().argmax(dim=1)
    row = torch.arange(direction.shape[0], device=direction.device)
    sign = torch.where(
        direction[row, dominant] < 0.0,
        -torch.ones_like(score),
        torch.ones_like(score),
    )
    return direction * sign[:, None], score


@torch.no_grad()
def select_residual_dipoles(
    field: GaussianField,
    target: torch.Tensor,
    rendered: torch.Tensor,
    cfg: FitConfig,
    mask: torch.Tensor,
    count: int,
    *,
    trust_radius: float = 0.35,
    max_color_contrast: float = 0.5,
    minimum_spacing: float = 3.0,
    spacing_scale: float = 0.75,
) -> DipoleSelection:
    """Select residual-conditioned dipoles with a deterministic spatial exclusion rule."""

    if cfg.renderer not in {
        "normalized",
        "cuda",
        "cuda_normalized",
        "cuda_tiled",
        "cuda_tiled_normalized",
    }:
        raise ValueError("gauge-lifted dipoles require a normalized renderer")
    if field.color_grads is not None:
        raise ValueError("FIT-032 currently supports constant per-row colors only")
    if count <= 0:
        empty_long = torch.zeros(0, device=field.means.device, dtype=torch.long)
        empty_two = field.means.new_zeros((0, 2))
        empty_three = field.means.new_zeros((0, 3))
        empty_scalar = field.means.new_zeros(0)
        return DipoleSelection(
            empty_long,
            empty_two,
            empty_three,
            empty_scalar,
            empty_scalar,
            empty_scalar,
            empty_scalar,
            0,
            0,
            0,
            0,
        )

    height, width = target.shape[:2]
    denominator = _raw_weight_map_field(
        field,
        cfg,
        height,
        width,
        support_fade_alpha=1.0,
    )
    residual = target - rendered
    gram, cross = _weight_translation_terms(
        field,
        cfg,
        height,
        width,
        denominator,
        residual,
        mask,
    )
    direction, unconstrained_score = _generalized_rank_one_solution(gram, cross)

    conics = field.conics(cfg.aa_dilation)
    mahalanobis = torch.sqrt(
        conics[:, 0] * direction[:, 0].square()
        + 2.0 * conics[:, 1] * direction[:, 0] * direction[:, 1]
        + conics[:, 2] * direction[:, 1].square()
    )
    displacement = direction * (
        float(trust_radius) / mahalanobis.clamp_min(1e-12)
    )[:, None]
    projected_energy = torch.einsum(
        "ni,nij,nj->n", displacement, gram, displacement
    ).clamp_min(1e-20)
    contrast = torch.einsum("nic,ni->nc", cross, displacement)
    contrast = contrast / projected_energy[:, None]
    maximum = contrast.abs().amax(dim=1)
    color_clip = torch.minimum(
        torch.ones_like(maximum),
        torch.full_like(maximum, float(max_color_contrast))
        / maximum.clamp_min(1e-12),
    )
    contrast = contrast * color_clip[:, None]
    score = unconstrained_score * (2.0 * color_clip - color_clip.square())
    support_scale = _support_stable_scales(
        field, displacement, cfg, height, width
    )
    root_support_scale = torch.sqrt(support_scale)
    displacement = displacement * root_support_scale[:, None]
    contrast = contrast * root_support_scale[:, None]
    score = score * (2.0 * support_scale - support_scale.square())

    means = field.means
    child_a = means + displacement
    child_b = means - displacement
    inside_image = (
        (child_a[:, 0] >= 0.0)
        & (child_a[:, 0] <= width - 1)
        & (child_a[:, 1] >= 0.0)
        & (child_a[:, 1] <= height - 1)
        & (child_b[:, 0] >= 0.0)
        & (child_b[:, 0] <= width - 1)
        & (child_b[:, 1] >= 0.0)
        & (child_b[:, 1] <= height - 1)
    )
    ax = child_a[:, 0].round().long().clamp(0, width - 1)
    ay = child_a[:, 1].round().long().clamp(0, height - 1)
    bx = child_b[:, 0].round().long().clamp(0, width - 1)
    by = child_b[:, 1].round().long().clamp(0, height - 1)
    inside_mask = inside_image & mask[ay, ax] & mask[by, bx]
    background = (
        torch.zeros(field.n, device=means.device, dtype=torch.bool)
        if field.background_mask is None
        else field.background_mask.to(device=means.device, dtype=torch.bool)
    )
    degenerate = (
        ~torch.isfinite(score)
        | ~torch.isfinite(displacement).all(dim=1)
        | ~torch.isfinite(contrast).all(dim=1)
        | (score <= 0.0)
        | (projected_energy <= 1e-18)
    )
    eligible = inside_mask & ~background & ~degenerate

    # Stable CPU ordering makes row-id the deterministic tie breaker. K is small, so the
    # O(NK) spatial exclusion is intentional and transparent.
    score_cpu = score.detach().cpu()
    eligible_cpu = eligible.detach().cpu()
    order = sorted(
        range(field.n),
        key=lambda index: (-float(score_cpu[index]), index),
    )
    selected: list[int] = []
    effective_scales = field.effective_scales(cfg.aa_dilation)
    footprint = torch.sqrt(effective_scales.prod(dim=1)).detach()
    for index in order:
        if not bool(eligible_cpu[index]):
            continue
        if selected:
            selected_tensor = torch.as_tensor(
                selected,
                device=means.device,
                dtype=torch.long,
            )
            distance = torch.linalg.vector_norm(
                means[selected_tensor] - means[index],
                dim=1,
            )
            exclusion = torch.maximum(
                torch.full_like(distance, float(minimum_spacing)),
                float(spacing_scale)
                * (footprint[selected_tensor] + footprint[index]),
            )
            if bool((distance < exclusion).any()):
                continue
        selected.append(index)
        if len(selected) >= int(count):
            break

    parents = torch.as_tensor(selected, device=means.device, dtype=torch.long)
    return DipoleSelection(
        parents=parents,
        displacement=displacement[parents],
        contrast=contrast[parents],
        score=score[parents],
        unclipped_score=unconstrained_score[parents],
        color_clip=color_clip[parents],
        support_scale=support_scale[parents],
        candidate_count=int(eligible.sum()),
        rejected_background=int((background & inside_mask & ~degenerate).sum()),
        rejected_mask=int((~inside_mask & ~background & ~degenerate).sum()),
        rejected_degenerate=int(degenerate.sum()),
    )


@torch.no_grad()
def apply_dipole_split(
    field: GaussianField,
    selection: DipoleSelection,
    *,
    lift_scale: float = 1.0,
) -> GaussianField:
    """Apply a joint displacement/color lift; ``lift_scale=0`` is the exact gauge split."""

    if not math.isfinite(lift_scale) or lift_scale < 0.0:
        raise ValueError("lift_scale must be finite and nonnegative")
    parents = selection.parents
    if parents.numel() == 0:
        return field.detached()
    if int(torch.unique(parents).numel()) != int(parents.numel()):
        raise ValueError("dipole parents must be unique")
    if field.opacities is None:
        raise ValueError("FIT-032 requires explicit opacity logits")
    if field.color_grads is not None:
        raise ValueError("FIT-032 currently supports constant per-row colors only")

    root = math.sqrt(float(lift_scale))
    displacement = selection.displacement * root
    contrast = selection.contrast * root
    means = field.means.detach().clone()
    colors = field.colors.detach().clone()
    opacities = field.opacities.detach().clone()
    means[parents] += displacement
    colors[parents] += contrast
    parent_opacity = torch.sigmoid(field.opacities.detach()[parents])
    child_opacity = parent_opacity * 0.5
    child_logits = _exact_logit(child_opacity)
    opacities[parents] = child_logits

    child = GaussianField(
        means=field.means.detach()[parents].clone() - displacement,
        log_scales=field.log_scales.detach()[parents].clone(),
        rotations=field.rotations.detach()[parents].clone(),
        colors=field.colors.detach()[parents].clone() - contrast,
        opacities=child_logits.clone(),
        scale_max=(
            None
            if field.scale_max is None
            else field.scale_max.detach()[parents].clone()
        ),
        color_grads=None,
        background_mask=(
            None
            if field.background_mask is None
            else torch.zeros_like(field.background_mask.detach()[parents])
        ),
        filter_variance=(
            None
            if field.filter_variance is None
            else field.filter_variance.detach()[parents].clone()
        ),
    )
    parent = GaussianField(
        means=means,
        log_scales=field.log_scales.detach().clone(),
        rotations=field.rotations.detach().clone(),
        colors=colors,
        opacities=opacities,
        scale_max=None if field.scale_max is None else field.scale_max.detach().clone(),
        color_grads=None,
        background_mask=(
            None
            if field.background_mask is None
            else field.background_mask.detach().clone()
        ),
        filter_variance=(
            None
            if field.filter_variance is None
            else field.filter_variance.detach().clone()
        ),
    )
    return parent.append(child)
