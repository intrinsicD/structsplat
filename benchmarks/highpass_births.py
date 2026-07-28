"""Spatially separated high-pass residual births for the FIT-033 benchmark."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch
import torch.nn.functional as F

from structsplat.gaussians import GaussianField


@dataclass(frozen=True)
class HighpassBirthSelection:
    """A score-ranked, nested cohort of fine-detail birth components."""

    components: GaussianField
    sites: torch.Tensor
    scores: torch.Tensor
    metadata: dict[str, Any]


def gaussian_blur(image: torch.Tensor, sigma: float) -> torch.Tensor:
    """Return a separable reflect-padded RGB Gaussian blur."""

    radius = int(math.ceil(3.0 * float(sigma)))
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


@torch.no_grad()
def select_highpass_births(
    field: GaussianField,
    target: torch.Tensor,
    rendered: torch.Tensor,
    constraint: Any,
    count: int,
    *,
    blur_sigma: float = 1.5,
    nms_radius: int = 2,
    deep_offset: float = 6.0,
    scale: float = 0.35,
    opacity: float = 0.8,
    forbidden_mask: torch.Tensor | None = None,
) -> HighpassBirthSelection:
    """Select births at deep-interior high-pass residual maxima.

    The selector scores the high-pass component of the rendering residual, not target texture
    alone. Spatial NMS prevents the no-NMS clustering seen in FIT-031. Geometry and opacity are
    fixed here; the FIT-033 partial solve subsequently determines exact normalized-compositor
    color coefficients.
    """

    if count <= 0:
        raise ValueError("count must be positive")
    if target.shape != rendered.shape or target.ndim != 3 or target.shape[2] != 3:
        raise ValueError("target and rendered must be matching HxWx3 tensors")
    if blur_sigma <= 0.0 or not math.isfinite(blur_sigma):
        raise ValueError("blur_sigma must be finite and positive")
    if nms_radius < 0:
        raise ValueError("nms_radius must be nonnegative")
    if scale < 0.35 or not math.isfinite(scale):
        raise ValueError("scale must be finite and at least 0.35 pixels")
    if not 0.0 < opacity < 1.0 or not math.isfinite(opacity):
        raise ValueError("opacity must be finite and in (0, 1)")

    height, width = target.shape[:2]
    residual = rendered - target
    highpass = residual - gaussian_blur(residual, float(blur_sigma))
    score = highpass.square().mean(dim=2)
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
    negative = torch.full_like(score, -float("inf"))
    score = torch.where(eligible, score, negative)
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
    if selected < count:
        raise RuntimeError(
            f"high-pass selector has only {finite} finite peaks for {count} rows"
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
    return HighpassBirthSelection(
        components=components,
        sites=sites,
        scores=ranked.values,
        metadata={
            "selected": selected,
            "finite_peaks": finite,
            "eligible_pixels": int(eligible.sum()),
            "forbidden_eligible_pixels": forbidden_pixels,
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
