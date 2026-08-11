"""Fail-closed RGB-tail refinement for fixed normalized Gaussian fields.

This module is a deterministic, default-off research reference.  Torch is imported lazily so
NumPy-only analysis imports do not acquire a torch dependency.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math
import time
from typing import TYPE_CHECKING

import numpy as np

from .progressive_residual_quadtree import progressive_artifact_metrics


if TYPE_CHECKING:
    from .config import FitConfig
    from .gaussians import GaussianField


_NORMALIZED_RENDERERS = {
    "normalized",
    "cuda",
    "cuda_normalized",
    "cuda_tiled",
    "cuda_tiled_normalized",
}


def _integer(value: object, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {result}")
    return result


def _finite(value: object, name: str, *, minimum: float = 0.0, strict: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise TypeError(f"{name} must be a finite number")
    result = float(value)
    invalid = result <= minimum if strict else result < minimum
    if not math.isfinite(result) or invalid:
        relation = ">" if strict else ">="
        raise ValueError(f"{name} must be finite and {relation} {minimum}, got {result}")
    return result


def _image(value: object, name: str) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a NumPy array")
    if value.ndim != 3 or value.shape[2] != 3 or min(value.shape[:2]) < 1:
        raise ValueError(f"{name} must have non-empty HWC RGB shape")
    if value.dtype.kind not in "fiu" or not np.isfinite(value).all():
        raise ValueError(f"{name} must contain finite numeric values")
    return np.array(value, dtype=np.float32, order="C", copy=True)


def _mask(value: object, shape: tuple[int, int]) -> np.ndarray:
    if not isinstance(value, np.ndarray) or value.dtype != np.bool_ or value.shape != shape:
        raise ValueError(f"mask must be a bool NumPy array with shape {shape}")
    result = np.array(value, dtype=bool, order="C", copy=True)
    if not result.any():
        raise ValueError("mask must contain at least one active pixel")
    return result


def _readonly(array: np.ndarray) -> np.ndarray:
    result = np.array(array, order="C", copy=True)
    result.flags.writeable = False
    return result


@dataclass(frozen=True)
class NormalizedTailRefinementConfig:
    """Color-only top-k tail objective and outer transaction settings."""

    steps: int = 100
    checkpoint_every: int = 5
    learning_rate: float = 0.01
    tail_fraction: float = 0.001
    tail_weight: float = 4.0
    max_color_shift: float = 1.0
    color_abs_limit: float = 8.0
    sse_relative_tolerance: float = 1e-8
    display_absolute_tolerance: float = 1e-12

    def __post_init__(self) -> None:
        for name in ("steps", "checkpoint_every"):
            object.__setattr__(
                self,
                name,
                _integer(getattr(self, name), name, minimum=1),
            )
        for name in (
            "learning_rate",
            "tail_fraction",
            "tail_weight",
            "max_color_shift",
            "color_abs_limit",
        ):
            object.__setattr__(
                self,
                name,
                _finite(getattr(self, name), name, strict=True),
            )
        if self.tail_fraction > 1.0:
            raise ValueError("tail_fraction must be <= 1")
        for name in ("sse_relative_tolerance", "display_absolute_tolerance"):
            object.__setattr__(self, name, _finite(getattr(self, name), name))


@dataclass(frozen=True)
class NormalizedTailRefinementCheckpoint:
    step: int
    selected: bool
    eligible: bool
    finite: bool
    raw_sse: float
    display_pixel_rmse_max: float
    display_patch7_rmse_max: float
    color_shift_max: float
    color_abs_max: float
    objective: float | None
    elapsed_seconds: float

    def to_record(self) -> dict[str, object]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class NormalizedTailRefinementResult:
    field: "GaussianField"
    reconstruction_raw: np.ndarray
    reconstruction: np.ndarray
    checkpoints: tuple[NormalizedTailRefinementCheckpoint, ...]
    selected_step: int
    tail_count: int
    initial_sse: float
    final_sse: float
    initial_display_pixel_rmse_max: float
    final_display_pixel_rmse_max: float
    initial_display_patch7_rmse_max: float
    final_display_patch7_rmse_max: float
    color_shift_max: float
    color_abs_max: float
    non_color_arrays_bit_exact: bool
    maintained_render_parity_max_abs: float
    elapsed_seconds: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "reconstruction_raw", _readonly(self.reconstruction_raw))
        object.__setattr__(self, "reconstruction", _readonly(self.reconstruction))
        if sum(checkpoint.selected for checkpoint in self.checkpoints) != 1:
            raise ValueError("exactly one normalized-tail checkpoint must be selected")

    def checkpoint_records(self) -> list[dict[str, object]]:
        return [checkpoint.to_record() for checkpoint in self.checkpoints]


def _non_color_equal(candidate: "GaussianField", source: "GaussianField") -> bool:
    names = (
        "means",
        "log_scales",
        "rotations",
        "opacities",
        "scale_max",
        "color_grads",
        "background_mask",
        "filter_variance",
    )
    for name in names:
        left = getattr(candidate, name)
        right = getattr(source, name)
        if (left is None) != (right is None):
            return False
        if left is not None and not bool((left.detach() == right.detach()).all()):
            return False
    return True


def refine_normalized_color_tail(
    field: "GaussianField",
    target: np.ndarray,
    mask: np.ndarray,
    render_config: "FitConfig",
    *,
    config: NormalizedTailRefinementConfig | None = None,
) -> NormalizedTailRefinementResult:
    """Optimize only RGB against a top-k residual tail and return a safe checkpoint.

    The input is never mutated.  Step zero competes with all checkpoints, and selected candidates
    may not regress raw SSE or exact displayed worst-pixel/7x7 maxima relative to that input.
    """

    started = time.perf_counter()

    import torch

    from .config import FitConfig
    from .fit import _render
    from .gaussians import GaussianField

    if not isinstance(field, GaussianField):
        raise TypeError("field must be GaussianField")
    if not isinstance(render_config, FitConfig):
        raise TypeError("render_config must be FitConfig")
    if field.n < 1:
        raise ValueError("field must contain at least one Gaussian")
    if render_config.renderer not in _NORMALIZED_RENDERERS:
        raise ValueError("tail refinement requires normalized renderer semantics")
    if render_config.color_basis != "constant" or field.color_grads is not None:
        raise ValueError("tail refinement currently requires constant colors")
    source = _image(target, "target")
    active = _mask(mask, source.shape[:2])
    cfg = config or NormalizedTailRefinementConfig()
    height, width = source.shape[:2]

    candidate = field.detached()
    candidate.colors.requires_grad_(True)
    anchor_colors = candidate.colors.detach().clone()
    target_tensor = torch.as_tensor(
        source,
        device=candidate.colors.device,
        dtype=candidate.colors.dtype,
    )
    mask_tensor = torch.as_tensor(active, device=candidate.colors.device, dtype=torch.bool)
    tail_count = max(1, int(math.ceil(cfg.tail_fraction * int(active.sum()))))
    optimizer = torch.optim.Adam([candidate.colors], lr=cfg.learning_rate)

    with torch.no_grad():
        initial_tensor = _render(candidate, render_config, height, width)
    initial_raw = initial_tensor.detach().cpu().numpy().astype(np.float32, copy=False)
    initial_metrics = progressive_artifact_metrics(
        initial_raw,
        source,
        active,
        pixel_threshold=0.02,
        patch7_threshold=0.01,
        displayed=True,
    )
    initial_raw_metrics = progressive_artifact_metrics(
        initial_raw,
        source,
        active,
        pixel_threshold=0.02,
        patch7_threshold=0.01,
        displayed=False,
    )
    initial_sse = float(initial_raw_metrics["sse"])
    initial_pixel = float(initial_metrics["pixel_rmse_max"])
    initial_patch = float(initial_metrics["patch7_rmse_max"])
    sse_tolerance = cfg.sse_relative_tolerance * max(initial_sse, 1.0)
    checkpoints: list[NormalizedTailRefinementCheckpoint] = [
        NormalizedTailRefinementCheckpoint(
            step=0,
            selected=False,
            eligible=True,
            finite=True,
            raw_sse=initial_sse,
            display_pixel_rmse_max=initial_pixel,
            display_patch7_rmse_max=initial_patch,
            color_shift_max=0.0,
            color_abs_max=float(torch.max(torch.abs(anchor_colors)).cpu()),
            objective=None,
            elapsed_seconds=time.perf_counter() - started,
        )
    ]
    best_index = 0
    best_key = (initial_pixel, initial_patch, initial_sse, 0)
    best_colors = anchor_colors.detach().clone()
    best_raw = np.array(initial_raw, copy=True)
    last_objective: float | None = None

    for step in range(1, cfg.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        rendered = _render(candidate, render_config, height, width)
        residual = rendered[mask_tensor] - target_tensor[mask_tensor]
        pixel_mse = torch.mean(residual.square(), dim=1)
        tail = torch.topk(
            pixel_mse,
            k=tail_count,
            largest=True,
            sorted=False,
        ).values.mean()
        loss = pixel_mse.mean() + cfg.tail_weight * tail
        if not bool(torch.isfinite(loss)):
            break
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            candidate.colors.copy_(
                torch.maximum(
                    torch.minimum(candidate.colors, anchor_colors + cfg.max_color_shift),
                    anchor_colors - cfg.max_color_shift,
                ).clamp(-cfg.color_abs_limit, cfg.color_abs_limit)
            )
        last_objective = float(loss.detach().cpu())
        if step % cfg.checkpoint_every != 0 and step != cfg.steps:
            continue

        with torch.no_grad():
            checkpoint_tensor = _render(candidate, render_config, height, width)
        checkpoint_raw = (
            checkpoint_tensor.detach().cpu().numpy().astype(np.float32, copy=False)
        )
        finite = bool(
            np.isfinite(checkpoint_raw).all()
            and torch.isfinite(candidate.colors).all()
        )
        raw_metrics = progressive_artifact_metrics(
            checkpoint_raw,
            source,
            active,
            pixel_threshold=0.02,
            patch7_threshold=0.01,
            displayed=False,
        )
        display_metrics = progressive_artifact_metrics(
            checkpoint_raw,
            source,
            active,
            pixel_threshold=0.02,
            patch7_threshold=0.01,
            displayed=True,
        )
        raw_sse = float(raw_metrics["sse"])
        pixel_max = float(display_metrics["pixel_rmse_max"])
        patch_max = float(display_metrics["patch7_rmse_max"])
        color_shift = float(
            torch.max(torch.abs(candidate.colors.detach() - anchor_colors)).cpu()
        )
        color_abs = float(torch.max(torch.abs(candidate.colors.detach())).cpu())
        eligible = bool(
            finite
            and color_shift <= cfg.max_color_shift + 1e-7
            and color_abs <= cfg.color_abs_limit + 1e-7
            and raw_sse <= initial_sse + sse_tolerance
            and pixel_max <= initial_pixel + cfg.display_absolute_tolerance
            and patch_max <= initial_patch + cfg.display_absolute_tolerance
        )
        checkpoints.append(
            NormalizedTailRefinementCheckpoint(
                step=step,
                selected=False,
                eligible=eligible,
                finite=finite,
                raw_sse=raw_sse,
                display_pixel_rmse_max=pixel_max,
                display_patch7_rmse_max=patch_max,
                color_shift_max=color_shift,
                color_abs_max=color_abs,
                objective=last_objective,
                elapsed_seconds=time.perf_counter() - started,
            )
        )
        key = (pixel_max, patch_max, raw_sse, step)
        if eligible and key < best_key:
            best_index = len(checkpoints) - 1
            best_key = key
            best_colors = candidate.colors.detach().clone()
            best_raw = np.array(checkpoint_raw, copy=True)

    checkpoints[best_index] = replace(checkpoints[best_index], selected=True)
    result_field = field.detached()
    with torch.no_grad():
        result_field.colors.copy_(best_colors)
        maintained_tensor = _render(result_field, render_config, height, width)
    maintained = maintained_tensor.detach().cpu().numpy().astype(np.float32, copy=False)
    parity = float(np.max(np.abs(maintained - best_raw)))
    selected = checkpoints[best_index]
    masked = np.where(active[:, :, None], best_raw, 0.0).astype(np.float32, copy=False)
    return NormalizedTailRefinementResult(
        field=result_field,
        reconstruction_raw=best_raw,
        reconstruction=masked,
        checkpoints=tuple(checkpoints),
        selected_step=selected.step,
        tail_count=tail_count,
        initial_sse=initial_sse,
        final_sse=selected.raw_sse,
        initial_display_pixel_rmse_max=initial_pixel,
        final_display_pixel_rmse_max=selected.display_pixel_rmse_max,
        initial_display_patch7_rmse_max=initial_patch,
        final_display_patch7_rmse_max=selected.display_patch7_rmse_max,
        color_shift_max=selected.color_shift_max,
        color_abs_max=selected.color_abs_max,
        non_color_arrays_bit_exact=_non_color_equal(result_field, field),
        maintained_render_parity_max_abs=parity,
        elapsed_seconds=time.perf_counter() - started,
    )


__all__ = [
    "NormalizedTailRefinementCheckpoint",
    "NormalizedTailRefinementConfig",
    "NormalizedTailRefinementResult",
    "refine_normalized_color_tail",
]
