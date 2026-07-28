"""Residual-tangent anisotropic birth geometry for FIT-036."""

from __future__ import annotations

import math
from typing import Any

import torch

from benchmarks.highpass_births import (
    HighpassBirthSelection,
    gaussian_blur,
    select_highpass_births,
)
from structsplat.gaussians import GaussianField
from structsplat.safe_schedule import _local_structure


@torch.no_grad()
def select_highpass_ridge_births(
    field: GaussianField,
    target: torch.Tensor,
    rendered: torch.Tensor,
    constraint: Any,
    count: int,
    *,
    blur_sigma: float = 1.5,
    nms_radius: int = 2,
    deep_offset: float = 6.0,
    short_scale: float = 0.35,
    max_long_scale: float = 1.5,
    coherence_power: float = 1.0,
    opacity: float = 0.8,
) -> HighpassBirthSelection:
    """Reuse FIT-033 sites but elongate footprints along residual tangents."""

    if max_long_scale < short_scale or not math.isfinite(max_long_scale):
        raise ValueError("max_long_scale must be finite and at least short_scale")
    if coherence_power <= 0.0 or not math.isfinite(coherence_power):
        raise ValueError("coherence_power must be finite and positive")
    isotropic = select_highpass_births(
        field,
        target,
        rendered,
        constraint,
        count,
        blur_sigma=blur_sigma,
        nms_radius=nms_radius,
        deep_offset=deep_offset,
        scale=short_scale,
        opacity=opacity,
    )
    height, width = target.shape[:2]
    residual = rendered - target
    highpass = residual - gaussian_blur(residual, float(blur_sigma))
    tangent, coherence = _local_structure(highpass)
    y = torch.div(isotropic.sites, width, rounding_mode="floor")
    x = isotropic.sites - y * width
    selected_coherence = coherence[y, x]
    long_scale = float(short_scale) + (
        float(max_long_scale) - float(short_scale)
    ) * selected_coherence.pow(float(coherence_power))
    components = isotropic.components.detached()
    components.log_scales = torch.log(
        torch.stack(
            [
                long_scale,
                torch.full_like(long_scale, float(short_scale)),
            ],
            dim=1,
        )
    )
    components.rotations = tangent[y, x].detach().clone()
    metadata = {
        **isotropic.metadata,
        "geometry_rule": "high-pass residual tangent/coherence anisotropy",
        "short_scale": float(short_scale),
        "max_long_scale": float(max_long_scale),
        "coherence_power": float(coherence_power),
        "coherence_min": float(selected_coherence.min()),
        "coherence_mean": float(selected_coherence.mean()),
        "coherence_max": float(selected_coherence.max()),
        "long_scale_min": float(long_scale.min()),
        "long_scale_mean": float(long_scale.mean()),
        "long_scale_max": float(long_scale.max()),
        "fit_size": [width, height],
    }
    return HighpassBirthSelection(
        components=components,
        sites=isotropic.sites,
        scores=isotropic.scores,
        metadata=metadata,
    )
