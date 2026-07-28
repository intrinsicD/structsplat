"""Exact partial constant-plus-affine color solve for FIT-035."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from benchmarks.spectral_color_solve import (
    spectral_normal_observation,
    spectral_objective,
)
from structsplat.config import FitConfig
from structsplat.fit import _normalized_color_denominator
from structsplat.gaussians import GaussianField
from structsplat.render import (
    _EPS,
    _element_budget,
    _flat_tile_slices,
    _support_weight,
    _tile_bounds,
    _tile_coords,
)


_NORMALIZED_RENDERERS = {
    "normalized",
    "cuda",
    "cuda_normalized",
    "cuda_tiled",
    "cuda_tiled_normalized",
}


@dataclass(frozen=True)
class AffinePartialSolveResult:
    field: GaussianField
    iterations: int
    converged: bool
    initial_residual_norm: float
    final_residual_norm: float
    relative_residual: float
    initial_objective: float
    final_objective: float
    raw_color_min: float
    raw_color_max: float
    gradient_min: float
    gradient_max: float
    gradient_max_abs: float


def _validate_coefficients(field: GaussianField, coefficients: torch.Tensor) -> None:
    if coefficients.ndim != 3 or coefficients.shape[:2] != (field.n, 3):
        raise ValueError("coefficients must have shape (field.n, 3, channels)")


@torch.no_grad()
def affine_color_basis_apply(
    field: GaussianField,
    coefficients: torch.Tensor,
    cfg: FitConfig,
    height: int,
    width: int,
    denominator: torch.Tensor,
    *,
    support_fade_alpha: float = 1.0,
) -> torch.Tensor:
    """Apply constant/local-x/local-y color coefficients under the exact compositor."""

    _validate_coefficients(field, coefficients)
    device, dtype = field.means.device, field.means.dtype
    output = torch.zeros(
        height * width,
        coefficients.shape[2],
        device=device,
        dtype=dtype,
    )
    means = field.means.detach()
    conics = field.conics(cfg.aa_dilation).detach()
    radii = field.radii(cfg.sigma_cutoff, cfg.aa_dilation)
    opacities = field.opacity_values()
    if opacities is not None:
        opacities = opacities.detach()
    scales = field.effective_scales(0.0).detach().clamp_min(1e-6)
    rotations = field.rotations.detach()
    x0, y0, tiles_x, count = _tile_bounds(
        means, radii, height, width
    )
    budget = _element_budget(cfg.render_chunk)
    for start, end in _flat_tile_slices(count, budget):
        gid, px, py = _tile_coords(
            x0, y0, tiles_x, count, start, end, device
        )
        dx = px.to(dtype) - means[gid, 0]
        dy = py.to(dtype) - means[gid, 1]
        a, b, c = conics[gid, 0], conics[gid, 1], conics[gid, 2]
        quadratic = a * dx * dx + 2.0 * b * dx * dy + c * dy * dy
        weight = _support_weight(
            quadratic,
            cfg.sigma_cutoff,
            cfg.support_fade,
            support_fade_alpha,
        )
        if opacities is not None:
            weight = weight * opacities[gid]
        flat = py * width + px
        basis = weight[:, None] / (denominator[flat] + _EPS)
        cosine = torch.cos(rotations[gid])
        sine = torch.sin(rotations[gid])
        local_x = (
            cosine * dx + sine * dy
        ) / scales[gid, 0]
        local_y = (
            -sine * dx + cosine * dy
        ) / scales[gid, 1]
        values = (
            coefficients[gid, 0]
            + coefficients[gid, 1] * local_x[:, None]
            + coefficients[gid, 2] * local_y[:, None]
        )
        output.index_add_(0, flat, basis * values)
    return output.reshape(height, width, coefficients.shape[2])


@torch.no_grad()
def affine_color_basis_transpose(
    field: GaussianField,
    image: torch.Tensor,
    cfg: FitConfig,
    height: int,
    width: int,
    denominator: torch.Tensor,
    *,
    support_fade_alpha: float = 1.0,
) -> torch.Tensor:
    """Apply the exact transpose of :func:`affine_color_basis_apply`."""

    device, dtype = field.means.device, field.means.dtype
    output = torch.zeros(
        field.n,
        3,
        image.shape[2],
        device=device,
        dtype=dtype,
    )
    means = field.means.detach()
    conics = field.conics(cfg.aa_dilation).detach()
    radii = field.radii(cfg.sigma_cutoff, cfg.aa_dilation)
    opacities = field.opacity_values()
    if opacities is not None:
        opacities = opacities.detach()
    scales = field.effective_scales(0.0).detach().clamp_min(1e-6)
    rotations = field.rotations.detach()
    x0, y0, tiles_x, count = _tile_bounds(
        means, radii, height, width
    )
    budget = _element_budget(cfg.render_chunk)
    for start, end in _flat_tile_slices(count, budget):
        gid, px, py = _tile_coords(
            x0, y0, tiles_x, count, start, end, device
        )
        dx = px.to(dtype) - means[gid, 0]
        dy = py.to(dtype) - means[gid, 1]
        a, b, c = conics[gid, 0], conics[gid, 1], conics[gid, 2]
        quadratic = a * dx * dx + 2.0 * b * dx * dy + c * dy * dy
        weight = _support_weight(
            quadratic,
            cfg.sigma_cutoff,
            cfg.support_fade,
            support_fade_alpha,
        )
        if opacities is not None:
            weight = weight * opacities[gid]
        flat = py * width + px
        basis_image = (
            weight[:, None]
            / (denominator[flat] + _EPS)
            * image[py, px].to(dtype)
        )
        cosine = torch.cos(rotations[gid])
        sine = torch.sin(rotations[gid])
        local_x = (
            cosine * dx + sine * dy
        ) / scales[gid, 0]
        local_y = (
            -sine * dx + cosine * dy
        ) / scales[gid, 1]
        output[:, 0].index_add_(0, gid, basis_image)
        output[:, 1].index_add_(0, gid, basis_image * local_x[:, None])
        output[:, 2].index_add_(0, gid, basis_image * local_y[:, None])
    return output


def _objective(
    residual: torch.Tensor,
    detail_mask: torch.Tensor | None,
    raw_mask: torch.Tensor | None,
    *,
    sigma: float,
    raw_weight: float,
) -> torch.Tensor:
    if detail_mask is None:
        return residual.square().sum()
    assert raw_mask is not None
    return spectral_objective(
        residual,
        detail_mask,
        raw_mask,
        sigma=sigma,
        raw_weight=raw_weight,
    )


@torch.no_grad()
def solve_new_row_affine_colors(
    field: GaussianField,
    target: torch.Tensor,
    cfg: FitConfig,
    new_rows: torch.Tensor,
    *,
    detail_mask: torch.Tensor | None = None,
    raw_mask: torch.Tensor | None = None,
    sigma: float = 1.5,
    raw_weight: float = 0.1,
    color_ridge: float = 1e-4,
    gradient_ridge: float = 1e-3,
    max_iterations: int = 48,
    tolerance: float = 1e-7,
) -> AffinePartialSolveResult:
    """Solve only new constant RGB and local affine RGB coefficients."""

    if cfg.renderer not in _NORMALIZED_RENDERERS:
        raise ValueError("affine partial solve requires a normalized renderer")
    if field.color_grads is not None:
        raise ValueError("input field must not already contain affine colors")
    if (detail_mask is None) != (raw_mask is None):
        raise ValueError("detail_mask and raw_mask must be both present or both absent")
    for name, value in (
        ("color_ridge", color_ridge),
        ("gradient_ridge", gradient_ridge),
    ):
        if value < 0.0 or not math.isfinite(value):
            raise ValueError(f"{name} must be finite and nonnegative")
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive")
    rows = torch.unique(
        new_rows.detach().to(device=field.means.device, dtype=torch.long),
        sorted=True,
    )
    if rows.numel() == 0:
        raise ValueError("new_rows must be nonempty")
    if int(rows[0]) < 0 or int(rows[-1]) >= field.n:
        raise IndexError("new_rows contains an index outside the field")

    height, width = target.shape[:2]
    if detail_mask is not None:
        detail_mask = detail_mask.to(device=target.device, dtype=torch.bool)
        raw_mask = raw_mask.to(device=target.device, dtype=torch.bool)
    denominator = _normalized_color_denominator(
        field,
        cfg,
        height,
        width,
        support_fade_alpha=1.0,
    )
    initial = torch.zeros(
        rows.numel(),
        3,
        3,
        device=field.means.device,
        dtype=field.means.dtype,
    )
    initial[:, 0] = field.colors.detach()[rows]
    fixed_coefficients = torch.zeros(
        field.n,
        3,
        3,
        device=field.means.device,
        dtype=field.means.dtype,
    )
    fixed_coefficients[:, 0] = field.colors.detach()
    fixed_coefficients[rows] = 0.0
    fixed_image = affine_color_basis_apply(
        field,
        fixed_coefficients,
        cfg,
        height,
        width,
        denominator,
    )
    right_image = target - fixed_image

    def observe_normal(image: torch.Tensor) -> torch.Tensor:
        if detail_mask is None:
            return image
        assert raw_mask is not None
        return spectral_normal_observation(
            image,
            detail_mask,
            raw_mask,
            sigma=sigma,
            raw_weight=raw_weight,
        )

    observed_right = observe_normal(right_image)
    right = affine_color_basis_transpose(
        field,
        observed_right,
        cfg,
        height,
        width,
        denominator,
    )[rows]
    diagonal = field.means.new_tensor(
        [color_ridge, gradient_ridge, gradient_ridge]
    ).reshape(1, 3, 1)
    right = right + diagonal * initial

    def render_new(coefficients: torch.Tensor) -> torch.Tensor:
        full = torch.zeros(
            field.n,
            3,
            3,
            device=field.means.device,
            dtype=field.means.dtype,
        )
        full[rows] = coefficients
        return affine_color_basis_apply(
            field,
            full,
            cfg,
            height,
            width,
            denominator,
        )

    def normal_apply(coefficients: torch.Tensor) -> torch.Tensor:
        image = observe_normal(render_new(coefficients))
        transposed = affine_color_basis_transpose(
            field,
            image,
            cfg,
            height,
            width,
            denominator,
        )[rows]
        return transposed + diagonal * coefficients

    initial_residual_image = fixed_image + render_new(initial) - target
    initial_objective = _objective(
        initial_residual_image,
        detail_mask,
        raw_mask,
        sigma=sigma,
        raw_weight=raw_weight,
    )
    solution = initial.clone()
    residual = right - normal_apply(solution)
    direction = residual.clone()
    squared = residual.square().sum(dim=(0, 1))
    initial_norm = torch.sqrt(squared.sum()).clamp_min(1e-30)
    converged = bool(
        torch.sqrt(squared.sum()) <= float(tolerance) * initial_norm
    )
    iterations = 0
    for iteration in range(int(max_iterations)):
        if converged:
            break
        applied = normal_apply(direction)
        denominator_step = (direction * applied).sum(dim=(0, 1))
        alpha = torch.where(
            denominator_step.abs() > 1e-30,
            squared / denominator_step,
            torch.zeros_like(squared),
        )
        solution = solution + direction * alpha[None, None, :]
        residual = residual - applied * alpha[None, None, :]
        next_squared = residual.square().sum(dim=(0, 1))
        beta = torch.where(
            squared > 1e-30,
            next_squared / squared,
            torch.zeros_like(squared),
        )
        direction = residual + direction * beta[None, None, :]
        squared = next_squared
        iterations = iteration + 1
        converged = bool(
            torch.sqrt(squared.sum()) <= float(tolerance) * initial_norm
        )

    solved = field.detached().with_affine_colors()
    solved.colors[rows] = solution[:, 0]
    assert solved.color_grads is not None
    solved.color_grads[rows] = solution[:, 1:]
    final_residual_image = fixed_image + render_new(solution) - target
    regularization = (
        diagonal * (solution - initial).square()
    ).sum()
    final_objective = _objective(
        final_residual_image,
        detail_mask,
        raw_mask,
        sigma=sigma,
        raw_weight=raw_weight,
    ) + regularization
    gradients = solution[:, 1:]
    final_norm = torch.sqrt(squared.sum())
    return AffinePartialSolveResult(
        field=solved,
        iterations=iterations,
        converged=converged,
        initial_residual_norm=float(initial_norm),
        final_residual_norm=float(final_norm),
        relative_residual=float(final_norm / initial_norm),
        initial_objective=float(initial_objective),
        final_objective=float(final_objective),
        raw_color_min=float(solution[:, 0].min()),
        raw_color_max=float(solution[:, 0].max()),
        gradient_min=float(gradients.min()),
        gradient_max=float(gradients.max()),
        gradient_max_abs=float(gradients.abs().max()),
    )
