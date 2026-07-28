"""Sparse orthogonal pursuit for masked fine-detail residuals.

The pursuit tail adds ordinary constant-color Gaussian rows in small waves.  Each wave:

* finds spatially separated extrema of the deep-interior high-pass residual;
* appends fixed small isotropic geometry at those pixel sites;
* jointly re-solves the colors of every tail row under the exact normalized compositor while
  leaving every inherited row frozen.

The safe schedule owns transactionality and protected-metric acceptance.  This module contains
only the deterministic selector, the exact partial color solve, and the two predeclared
fine-detail metrics used by the stop rule.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

import torch
import torch.nn.functional as F

from .config import FitConfig
from .fit import (
    _MaskConstraint,
    _normalized_color_basis_apply,
    _normalized_color_basis_transpose,
    _normalized_color_denominator,
)
from .gaussians import GaussianField


_NORMALIZED_RENDERERS = {
    "normalized",
    "cuda",
    "cuda_normalized",
    "cuda_tiled",
    "cuda_tiled_normalized",
}


@dataclass(frozen=True)
class FineDetailMetrics:
    """Deep-interior residual energies used by the pursuit stop rule."""

    highpass_mse: float
    laplacian_mse: float
    pixels: int

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


@dataclass(frozen=True)
class PursuitBirthSelection:
    """One score-ranked pursuit cohort, or an insufficient-site result."""

    components: GaussianField | None
    sites: torch.Tensor
    scores: torch.Tensor
    metadata: dict[str, Any]


@dataclass(frozen=True)
class PartialColorSolveResult:
    """Result of an exact-compositor solve over a strict subset of field rows."""

    field: GaussianField
    iterations: int
    converged: bool
    initial_residual_norm: float
    final_residual_norm: float
    relative_residual: float
    raw_color_min: float
    raw_color_max: float

    def metadata(self) -> dict[str, float | int | bool]:
        return {
            "iterations": self.iterations,
            "converged": self.converged,
            "initial_residual_norm": self.initial_residual_norm,
            "final_residual_norm": self.final_residual_norm,
            "relative_residual": self.relative_residual,
            "raw_color_min": self.raw_color_min,
            "raw_color_max": self.raw_color_max,
        }


def gaussian_blur(image: torch.Tensor, sigma: float) -> torch.Tensor:
    """Return a separable reflect-padded RGB Gaussian blur."""

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must have shape HxWx3")
    if not math.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("sigma must be finite and positive")
    radius = int(math.ceil(3.0 * float(sigma)))
    if int(image.shape[0]) <= radius or int(image.shape[1]) <= radius:
        raise ValueError("image is too small for reflect-padded Gaussian blur")
    coordinate = torch.arange(
        -radius,
        radius + 1,
        device=image.device,
        dtype=image.dtype,
    )
    kernel = torch.exp(-0.5 * (coordinate / float(sigma)).square())
    kernel = kernel / kernel.sum()
    value = image.permute(2, 0, 1).unsqueeze(0)
    horizontal = kernel.view(1, 1, 1, -1).expand(3, 1, 1, -1)
    vertical = kernel.view(1, 1, -1, 1).expand(3, 1, -1, 1)
    value = F.conv2d(
        F.pad(value, (radius, radius, 0, 0), mode="reflect"),
        horizontal,
        groups=3,
    )
    value = F.conv2d(
        F.pad(value, (0, 0, radius, radius), mode="reflect"),
        vertical,
        groups=3,
    )
    return value[0].permute(1, 2, 0)


def _laplacian(image: torch.Tensor) -> torch.Tensor:
    kernel = image.new_tensor(
        [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]]
    )
    value = image.permute(2, 0, 1).unsqueeze(0)
    weights = kernel.view(1, 1, 3, 3).expand(3, 1, 3, 3)
    value = F.conv2d(
        F.pad(value, (1, 1, 1, 1), mode="reflect"),
        weights,
        groups=3,
    )
    return value[0].permute(1, 2, 0)


@torch.no_grad()
def fine_detail_metrics(
    rendered: torch.Tensor,
    target: torch.Tensor,
    constraint: _MaskConstraint,
    *,
    blur_sigma: float,
    deep_offset: float,
) -> FineDetailMetrics:
    """Measure high-pass and Laplacian residual energy in the reachable deep interior."""

    if rendered.shape != target.shape:
        raise ValueError("rendered and target must have matching shapes")
    if not math.isfinite(deep_offset) or deep_offset < 0.0:
        raise ValueError("deep_offset must be finite and nonnegative")
    height, width = target.shape[:2]
    deep = (
        constraint.sdf_flat.reshape(height, width)
        > float(constraint.margin) + float(deep_offset)
    )
    if not bool(deep.any()):
        deep = constraint.inside
    residual = rendered - target
    highpass = residual - gaussian_blur(residual, float(blur_sigma))
    laplacian = _laplacian(residual)
    return FineDetailMetrics(
        highpass_mse=float(highpass[deep].square().mean()),
        laplacian_mse=float(laplacian[deep].square().mean()),
        pixels=int(deep.sum()),
    )


def relative_detail_reductions(
    before: FineDetailMetrics,
    after: FineDetailMetrics,
) -> dict[str, float]:
    """Return positive-is-better relative reductions for the two stop metrics."""

    def reduction(old: float, new: float) -> float:
        if old <= 0.0:
            return 0.0 if new <= 0.0 else -float("inf")
        return 1.0 - new / old

    return {
        "highpass": reduction(before.highpass_mse, after.highpass_mse),
        "laplacian": reduction(before.laplacian_mse, after.laplacian_mse),
    }


@torch.no_grad()
def select_pursuit_births(
    field: GaussianField,
    target: torch.Tensor,
    rendered: torch.Tensor,
    constraint: _MaskConstraint,
    count: int,
    *,
    blur_sigma: float,
    nms_radius: int,
    deep_offset: float,
    scale: float,
    opacity: float,
    forbidden_mask: torch.Tensor | None = None,
) -> PursuitBirthSelection:
    """Select one NMS-separated wave of deep high-pass residual sites.

    ``forbidden_mask`` excludes only previously used sites in the selected FIT-039 mechanism.
    The NMS radius applies within the new wave, not across waves; adjacent later-wave sites are
    deliberately allowed because the killing ablation showed that they encode distinct fabric
    residual lobes efficiently.
    """

    if count <= 0:
        raise ValueError("count must be positive")
    if target.shape != rendered.shape or target.ndim != 3 or target.shape[2] != 3:
        raise ValueError("target and rendered must be matching HxWx3 tensors")
    if nms_radius < 0:
        raise ValueError("nms_radius must be nonnegative")
    if not math.isfinite(scale) or scale < 0.35:
        raise ValueError("scale must be finite and at least 0.35 pixels")
    if not math.isfinite(opacity) or not 0.0 < opacity < 1.0:
        raise ValueError("opacity must be finite and in (0, 1)")

    height, width = target.shape[:2]
    residual = rendered - target
    highpass = residual - gaussian_blur(residual, float(blur_sigma))
    raw_score = highpass.square().mean(dim=2)
    eligible = (
        constraint.sdf_flat.reshape(height, width)
        > float(constraint.margin) + float(deep_offset)
    )
    forbidden_pixels = 0
    if forbidden_mask is not None:
        forbidden = forbidden_mask.to(device=target.device, dtype=torch.bool)
        if forbidden.shape != (height, width):
            raise ValueError("forbidden_mask must match the image spatial shape")
        forbidden_pixels = int((eligible & forbidden).sum())
        eligible = eligible & ~forbidden

    negative = torch.full_like(raw_score, -float("inf"))
    score = torch.where(eligible, raw_score, negative)
    if nms_radius:
        pooled = F.max_pool2d(
            score[None, None],
            2 * int(nms_radius) + 1,
            stride=1,
            padding=int(nms_radius),
        )[0, 0]
        score = torch.where((score >= pooled) & eligible, score, negative)
    flat = score.reshape(-1)
    finite = int(torch.isfinite(flat).sum())
    selected = min(int(count), finite)
    if selected <= 0:
        empty_sites = torch.empty(0, device=target.device, dtype=torch.long)
        return PursuitBirthSelection(
            components=None,
            sites=empty_sites,
            scores=target.new_empty((0,)),
            metadata={
                "selected": 0,
                "requested": int(count),
                "finite_peaks": finite,
                "eligible_pixels": int(eligible.sum()),
                "forbidden_eligible_pixels": forbidden_pixels,
                "insufficient_sites": True,
            },
        )

    ranked = torch.topk(flat, k=selected, sorted=True)
    sites = ranked.indices
    y = torch.div(sites, width, rounding_mode="floor")
    x = sites - y * width
    means = torch.stack(
        [x.to(dtype=target.dtype), y.to(dtype=target.dtype)],
        dim=1,
    )
    log_scales = torch.full(
        (selected, 2),
        math.log(float(scale)),
        device=target.device,
        dtype=target.dtype,
    )
    opacity_logit = math.log(float(opacity) / (1.0 - float(opacity)))
    components = GaussianField(
        means=means,
        log_scales=log_scales,
        rotations=torch.zeros(
            selected,
            device=target.device,
            dtype=target.dtype,
        ),
        colors=target[y, x].detach().clone(),
        opacities=torch.full(
            (selected,),
            opacity_logit,
            device=target.device,
            dtype=target.dtype,
        ),
        scale_max=torch.full_like(log_scales, float("inf")),
        filter_variance=(
            None
            if field.filter_variance is None
            else torch.zeros(
                selected,
                device=target.device,
                dtype=target.dtype,
            )
        ),
    )
    return PursuitBirthSelection(
        components=components,
        sites=sites,
        scores=ranked.values,
        metadata={
            "selected": selected,
            "requested": int(count),
            "finite_peaks": finite,
            "eligible_pixels": int(eligible.sum()),
            "forbidden_eligible_pixels": forbidden_pixels,
            "insufficient_sites": selected < int(count),
            "score_rule": (
                f"RGB mean-square sigma-{float(blur_sigma):g} high-pass "
                "rendering residual"
            ),
            "nms_radius": int(nms_radius),
            "deep_offset": float(deep_offset),
            "scale": float(scale),
            "opacity": float(opacity),
            "score_min": float(ranked.values.min()),
            "score_mean": float(ranked.values.mean()),
            "score_max": float(ranked.values.max()),
        },
    )


@torch.no_grad()
def solve_partial_colors_normalized(
    field: GaussianField,
    target: torch.Tensor,
    cfg: FitConfig,
    rows_to_solve: torch.Tensor,
    *,
    ridge: float,
    max_iterations: int,
    tolerance: float,
) -> PartialColorSolveResult:
    """Jointly solve selected colors under the exact normalized compositor.

    Geometry, opacity, and every color outside ``rows_to_solve`` remain bit-for-bit frozen.
    """

    if cfg.renderer not in _NORMALIZED_RENDERERS:
        raise ValueError("partial color solve requires a normalized renderer")
    if field.color_grads is not None:
        raise ValueError("partial color solve supports constant colors only")
    if not math.isfinite(ridge) or ridge < 0.0:
        raise ValueError("ridge must be finite and nonnegative")
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive")
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be finite and positive")
    rows = torch.unique(
        rows_to_solve.detach().to(
            device=field.means.device,
            dtype=torch.long,
        ),
        sorted=True,
    )
    if rows.numel() == 0:
        raise ValueError("rows_to_solve must be nonempty")
    if int(rows[0]) < 0 or int(rows[-1]) >= field.n:
        raise IndexError("rows_to_solve contains an index outside the field")

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
        step_denominator = (direction * applied).sum(dim=0)
        alpha = torch.where(
            step_denominator.abs() > 1e-30,
            squared / step_denominator,
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
