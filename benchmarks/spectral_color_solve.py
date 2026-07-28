"""High-pass-observation partial color solve for the FIT-034 benchmark."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F

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
class SpectralColorSolveResult:
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


def gaussian_highpass_zero(image: torch.Tensor, sigma: float) -> torch.Tensor:
    """Apply a symmetric zero-padded ``I - GaussianBlur`` operator to HxWxC."""

    if sigma <= 0.0 or not math.isfinite(sigma):
        raise ValueError("sigma must be finite and positive")
    radius = int(math.ceil(3.0 * float(sigma)))
    coordinate = torch.arange(
        -radius,
        radius + 1,
        device=image.device,
        dtype=image.dtype,
    )
    kernel_1d = torch.exp(-0.5 * (coordinate / float(sigma)).square())
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
    channels = image.shape[2]
    weight = kernel_2d.view(1, 1, 2 * radius + 1, 2 * radius + 1).expand(
        channels, 1, -1, -1
    )
    value = image.permute(2, 0, 1).unsqueeze(0)
    blurred = F.conv2d(value, weight, padding=radius, groups=channels)
    return image - blurred[0].permute(1, 2, 0)


def spectral_normal_observation(
    image: torch.Tensor,
    detail_mask: torch.Tensor,
    raw_mask: torch.Tensor,
    *,
    sigma: float,
    raw_weight: float,
) -> torch.Tensor:
    """Apply ``H M_detail H + raw_weight M_raw`` to an image.

    ``H`` is symmetric because :func:`gaussian_highpass_zero` uses a symmetric kernel and
    zero padding. The result is therefore the exact normal operator of the observation stack
    ``[sqrt(M_detail) H; sqrt(raw_weight M_raw)]``.
    """

    if raw_weight < 0.0 or not math.isfinite(raw_weight):
        raise ValueError("raw_weight must be finite and nonnegative")
    if detail_mask.shape != image.shape[:2] or raw_mask.shape != image.shape[:2]:
        raise ValueError("observation masks must match the image spatial shape")
    detail = detail_mask.to(device=image.device, dtype=image.dtype)[..., None]
    raw = raw_mask.to(device=image.device, dtype=image.dtype)[..., None]
    highpass = gaussian_highpass_zero(image, sigma)
    return gaussian_highpass_zero(detail * highpass, sigma) + float(
        raw_weight
    ) * raw * image


def spectral_objective(
    residual: torch.Tensor,
    detail_mask: torch.Tensor,
    raw_mask: torch.Tensor,
    *,
    sigma: float,
    raw_weight: float,
) -> torch.Tensor:
    """Return the unnormalized objective whose Hessian is above."""

    highpass = gaussian_highpass_zero(residual, sigma)
    detail = highpass[detail_mask].square().sum()
    raw = residual[raw_mask].square().sum()
    return detail + float(raw_weight) * raw


@torch.no_grad()
def solve_new_row_colors_spectral(
    field: GaussianField,
    target: torch.Tensor,
    cfg: FitConfig,
    new_rows: torch.Tensor,
    detail_mask: torch.Tensor,
    raw_mask: torch.Tensor,
    *,
    sigma: float = 1.5,
    raw_weight: float = 0.1,
    ridge: float = 1e-4,
    max_iterations: int = 48,
    tolerance: float = 1e-7,
) -> SpectralColorSolveResult:
    """Solve new-row colors under an exact high-pass-plus-RGB observation."""

    if cfg.renderer not in _NORMALIZED_RENDERERS:
        raise ValueError("spectral color solve requires a normalized renderer")
    if field.color_grads is not None:
        raise ValueError("spectral color solve currently supports constant colors only")
    if ridge < 0.0 or not math.isfinite(ridge):
        raise ValueError("ridge must be finite and nonnegative")
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
    detail_mask = detail_mask.to(device=target.device, dtype=torch.bool)
    raw_mask = raw_mask.to(device=target.device, dtype=torch.bool)
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
    observed_right = spectral_normal_observation(
        right_image,
        detail_mask,
        raw_mask,
        sigma=sigma,
        raw_weight=raw_weight,
    )
    initial = field.colors.detach()[rows].clone()
    right = _normalized_color_basis_transpose(
        field,
        observed_right,
        cfg,
        height,
        width,
        denominator,
        support_fade_alpha=1.0,
    )[rows]
    right = right + float(ridge) * initial

    def render_new(colors: torch.Tensor) -> torch.Tensor:
        full = torch.zeros_like(field.colors)
        full[rows] = colors
        return _normalized_color_basis_apply(
            field,
            full,
            cfg,
            height,
            width,
            denominator,
            support_fade_alpha=1.0,
        )

    def normal_apply(colors: torch.Tensor) -> torch.Tensor:
        image = render_new(colors)
        observed = spectral_normal_observation(
            image,
            detail_mask,
            raw_mask,
            sigma=sigma,
            raw_weight=raw_weight,
        )
        transposed = _normalized_color_basis_transpose(
            field,
            observed,
            cfg,
            height,
            width,
            denominator,
            support_fade_alpha=1.0,
        )[rows]
        return transposed + float(ridge) * colors

    initial_residual_image = fixed_image + render_new(initial) - target
    initial_objective = spectral_objective(
        initial_residual_image,
        detail_mask,
        raw_mask,
        sigma=sigma,
        raw_weight=raw_weight,
    )
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
    final_residual_image = fixed_image + render_new(solution) - target
    final_objective = spectral_objective(
        final_residual_image,
        detail_mask,
        raw_mask,
        sigma=sigma,
        raw_weight=raw_weight,
    ) + float(ridge) * (solution - initial).square().sum()
    final_norm = torch.sqrt(squared.sum())
    return SpectralColorSolveResult(
        field=solved,
        iterations=iterations,
        converged=converged,
        initial_residual_norm=float(initial_norm),
        final_residual_norm=float(final_norm),
        relative_residual=float(final_norm / initial_norm),
        initial_objective=float(initial_objective),
        final_objective=float(final_objective),
        raw_color_min=float(solution.min()),
        raw_color_max=float(solution.max()),
    )
