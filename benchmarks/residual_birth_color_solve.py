"""Frozen-base partial color solve for residual Gaussian births (FIT-033)."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from structsplat.config import FitConfig
from structsplat.fit import (
    _normalized_color_basis_apply,
    _normalized_color_basis_transpose,
    _normalized_color_denominator,
)
from structsplat.gaussians import GaussianField


_NORMALIZED_RENDERERS = {
    "normalized",
    "cuda",
    "cuda_normalized",
    "cuda_tiled",
    "cuda_tiled_normalized",
}


@dataclass(frozen=True)
class PartialColorSolveResult:
    field: GaussianField
    iterations: int
    converged: bool
    initial_residual_norm: float
    final_residual_norm: float
    relative_residual: float
    raw_color_min: float
    raw_color_max: float


@torch.no_grad()
def solve_new_row_colors(
    field: GaussianField,
    target: torch.Tensor,
    cfg: FitConfig,
    new_rows: torch.Tensor,
    *,
    ridge: float = 1e-4,
    max_iterations: int = 32,
    tolerance: float = 1e-7,
) -> PartialColorSolveResult:
    """Solve only ``new_rows`` under the exact normalized compositor.

    With fixed geometry and opacity, ``render = A c``. Inherited colors are held fixed, so the
    solved system is

        (A_new.T A_new + ridge I) x
          = A_new.T (target - A_old c_old) + ridge x_initial.

    The operator is applied implicitly with the renderer's exact support enumeration.
    """

    if cfg.renderer not in _NORMALIZED_RENDERERS:
        raise ValueError("partial color solve requires a normalized renderer")
    if field.color_grads is not None:
        raise ValueError("partial color solve currently supports constant colors only")
    if ridge < 0.0:
        raise ValueError("ridge must be nonnegative")
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
    denominator = _normalized_color_denominator(
        field,
        cfg,
        height,
        width,
        support_fade_alpha=1.0,
    )
    inherited_colors = field.colors.detach().clone()
    inherited_colors[rows] = 0.0
    fixed_image = _normalized_color_basis_apply(
        field,
        inherited_colors,
        cfg,
        height,
        width,
        denominator,
        support_fade_alpha=1.0,
    )
    right_image = target - fixed_image
    initial = field.colors.detach()[rows].clone()
    right = _normalized_color_basis_transpose(
        field,
        right_image,
        cfg,
        height,
        width,
        denominator,
        support_fade_alpha=1.0,
    )[rows]
    right = right + float(ridge) * initial

    def normal_apply(colors: torch.Tensor) -> torch.Tensor:
        full = torch.zeros_like(field.colors)
        full[rows] = colors
        image = _normalized_color_basis_apply(
            field,
            full,
            cfg,
            height,
            width,
            denominator,
            support_fade_alpha=1.0,
        )
        transposed = _normalized_color_basis_transpose(
            field,
            image,
            cfg,
            height,
            width,
            denominator,
            support_fade_alpha=1.0,
        )[rows]
        return transposed + float(ridge) * colors

    solution = initial.clone()
    residual = right - normal_apply(solution)
    direction = residual.clone()
    squared = residual.square().sum(dim=0)
    initial_norm = torch.sqrt(squared.sum()).clamp_min(1e-30)
    converged = bool(
        torch.sqrt(squared.sum()) <= float(tolerance) * initial_norm
    )
    iterations = 0
    for iteration in range(int(max_iterations)):
        if converged:
            break
        applied = normal_apply(direction)
        denominator_step = (direction * applied).sum(dim=0)
        alpha = torch.where(
            denominator_step.abs() > 1e-30,
            squared / denominator_step,
            torch.zeros_like(squared),
        )
        solution = solution + direction * alpha[None, :]
        residual = residual - applied * alpha[None, :]
        next_squared = residual.square().sum(dim=0)
        beta = torch.where(
            squared > 1e-30,
            next_squared / squared,
            torch.zeros_like(squared),
        )
        direction = residual + direction * beta[None, :]
        squared = next_squared
        iterations = iteration + 1
        converged = bool(
            torch.sqrt(squared.sum()) <= float(tolerance) * initial_norm
        )

    solved = field.detached()
    solved.colors[rows] = solution
    final_norm = torch.sqrt(squared.sum())
    return PartialColorSolveResult(
        field=solved,
        iterations=iterations,
        converged=converged,
        initial_residual_norm=float(initial_norm),
        final_residual_norm=float(final_norm),
        relative_residual=float(final_norm / initial_norm),
        raw_color_min=float(solution.min()),
        raw_color_max=float(solution.max()),
    )
