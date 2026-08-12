"""Deterministic residual-pursuit tails for pure additive Gaussian fields.

This default-off research method appends fixed-scale Gaussian rows at the current worst raw RGB
residuals.  The encoder uses source pixels to construct the rows, but the returned endpoint is a
plain four-array Gaussian field rendered in one additive pass.  Torch remains a lazy dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import math
import time
from typing import TYPE_CHECKING

import numpy as np


if TYPE_CHECKING:
    from .gaussians import GaussianField


_ADDITIVE_RENDERERS = frozenset(("additive", "cuda_additive", "cuda_tiled_additive"))
_PARITY_LIMIT = 2e-5


def _integer(value: object, name: str, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {result}")
    return result


def _finite_positive(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise TypeError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and > 0, got {result}")
    return result


def _image(value: object, name: str) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a NumPy array")
    if value.ndim != 3 or value.shape[2] != 3 or min(value.shape[:2]) < 1:
        raise ValueError(f"{name} must have non-empty HWC RGB shape")
    if value.dtype.kind not in "fiu" or not np.isfinite(value).all():
        raise ValueError(f"{name} must contain finite numeric values")
    return np.array(value, dtype=np.float32, order="C", copy=True)


def _selection_mask(value: object, shape: tuple[int, int]) -> np.ndarray:
    if not isinstance(value, np.ndarray) or value.dtype != np.bool_ or value.shape != shape:
        raise ValueError(f"selection_mask must be a bool NumPy array with shape {shape}")
    result = np.array(value, dtype=bool, order="C", copy=True)
    if not result.any():
        raise ValueError("selection_mask must contain at least one active pixel")
    return result


def _readonly(value: np.ndarray) -> np.ndarray:
    result = np.array(value, dtype=np.float32, order="C", copy=True)
    result.flags.writeable = False
    return result


@dataclass(frozen=True)
class ResidualPursuitAdditiveConfig:
    """Frozen tail count, geometry, renderer, and safety controls."""

    tail_gaussians: int = 64
    scale_px: float = 0.35
    coefficient_abs_limit: float = 16.0
    sigma_cutoff: float = 3.0
    support_fade: bool = False
    render_chunk: int = 256
    renderer: str = "additive"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "tail_gaussians",
            _integer(self.tail_gaussians, "tail_gaussians"),
        )
        object.__setattr__(
            self,
            "render_chunk",
            _integer(self.render_chunk, "render_chunk"),
        )
        for name in ("scale_px", "coefficient_abs_limit", "sigma_cutoff"):
            object.__setattr__(
                self,
                name,
                _finite_positive(getattr(self, name), name),
            )
        if not isinstance(self.support_fade, bool):
            raise TypeError("support_fade must be bool")
        if self.renderer not in _ADDITIVE_RENDERERS:
            expected = ", ".join(sorted(_ADDITIVE_RENDERERS))
            raise ValueError(f"renderer must use additive semantics; expected one of {expected}")


@dataclass(frozen=True)
class ResidualPursuitStep:
    step: int
    x: int
    y: int
    selected_pixel_mse: float
    selected_pixel_rmse: float
    coefficient_rgb: tuple[float, float, float]
    post_step_pixel_rmse_max: float
    kernel_pixel_updates: int

    def to_record(self) -> dict[str, object]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class ResidualPursuitAdditiveResult:
    """Pure endpoint, encoder-side construction audit, and exact work accounting."""

    base_field: "GaussianField"
    tail_field: "GaussianField"
    field: "GaussianField"
    base_reconstruction_raw: np.ndarray
    analytic_reconstruction_raw: np.ndarray
    reconstruction_raw: np.ndarray
    trajectory: tuple[ResidualPursuitStep, ...]
    base_count: int
    tail_count: int
    total_count: int
    residual_scan_pixel_evaluations: int
    selection_mask_applied: bool
    selection_active_pixels: int
    tail_kernel_pixel_updates: int
    renderer_calls: int
    base_prefix_bit_exact: bool
    fixed_tail_geometry: bool
    training_payload_removed: bool
    analytic_render_parity_max_abs: float
    initial_pixel_rmse_max: float
    final_pixel_rmse_max: float
    coefficient_abs_max: float
    base_field_digest: str
    tail_field_digest: str
    endpoint_field_digest: str
    status: str
    elapsed_seconds: float

    def __post_init__(self) -> None:
        for name in (
            "base_reconstruction_raw",
            "analytic_reconstruction_raw",
            "reconstruction_raw",
        ):
            object.__setattr__(self, name, _readonly(getattr(self, name)))
        object.__setattr__(self, "trajectory", tuple(self.trajectory))
        if self.base_field.n != self.base_count:
            raise ValueError("base endpoint count does not match its accounting")
        if self.tail_field.n != self.tail_count:
            raise ValueError("tail endpoint count does not match its accounting")
        if self.field.n != self.total_count:
            raise ValueError("final endpoint count does not match its accounting")
        if self.total_count != self.base_count + self.tail_count:
            raise ValueError("base/tail count allocation is inconsistent")
        if len(self.trajectory) != self.tail_count:
            raise ValueError("trajectory length does not match tail count")
        if not all(
            _pure_payload(field) for field in (self.base_field, self.tail_field, self.field)
        ):
            raise ValueError("pursuit endpoints must contain exactly four Gaussian arrays")

    @property
    def completed(self) -> bool:
        return self.status == "completed"

    def trajectory_records(self) -> list[dict[str, object]]:
        return [point.to_record() for point in self.trajectory]


def _pure_payload(field: "GaussianField") -> bool:
    return all(
        getattr(field, name) is None
        for name in (
            "opacities",
            "scale_max",
            "color_grads",
            "background_mask",
            "filter_variance",
        )
    )


def _field_digest(field: "GaussianField") -> str:
    digest = sha256()
    for name in ("means", "log_scales", "rotations", "colors"):
        value = np.ascontiguousarray(getattr(field, name).detach().cpu().numpy())
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.tobytes())
    for name in ("filter_variance", "scale_max"):
        value = getattr(field, name)
        digest.update(name.encode())
        if value is None:
            digest.update(b"none")
        else:
            digest.update(np.ascontiguousarray(value.detach().cpu().numpy()).tobytes())
    return digest.hexdigest()


def _render(field: "GaussianField", height: int, width: int, config):
    from .render import render_field

    return render_field(
        field.means,
        field.conics(),
        field.colors,
        field.radii(config.sigma_cutoff),
        height,
        width,
        chunk=config.render_chunk,
        mode=config.renderer,
        scales=field.scales(),
        rotations=field.rotations,
        support_fade=config.support_fade,
        sigma_cutoff=config.sigma_cutoff,
    )


def _prefix_equal(candidate: "GaussianField", base: "GaussianField") -> bool:
    import torch

    return all(
        torch.equal(getattr(candidate, name)[: base.n], getattr(base, name))
        for name in ("means", "log_scales", "rotations", "colors")
    )


def append_residual_pursuit_gaussians(
    field: "GaussianField",
    target: np.ndarray,
    *,
    config: ResidualPursuitAdditiveConfig | None = None,
    selection_mask: np.ndarray | None = None,
) -> ResidualPursuitAdditiveResult:
    """Append deterministic worst-residual Gaussians and return a pure additive endpoint.

    ``selection_mask`` is encoder-only: when supplied, worst-pixel selection and the reported
    residual maxima are restricted to active pixels. The returned endpoint remains the same
    four-array Gaussian representation and does not retain the mask.
    """

    started = time.perf_counter()

    import torch

    from .gaussians import GaussianField

    if not isinstance(field, GaussianField):
        raise TypeError("field must be GaussianField")
    if field.n < 1:
        raise ValueError("field must contain at least one Gaussian")
    if not _pure_payload(field):
        raise ValueError("residual pursuit requires a pure four-array Gaussian field")
    source = _image(target, "target")
    cfg = config or ResidualPursuitAdditiveConfig()
    height, width = source.shape[:2]
    active = (
        np.ones((height, width), dtype=bool)
        if selection_mask is None
        else _selection_mask(selection_mask, (height, width))
    )
    base = field.detached()
    base_coefficient_max = float(torch.max(torch.abs(base.colors)).detach().cpu())
    if base_coefficient_max > cfg.coefficient_abs_limit:
        raise ValueError("base field exceeds coefficient_abs_limit")

    with torch.no_grad():
        base_tensor = _render(base, height, width, cfg)
    base_raw = base_tensor.detach().cpu().numpy().astype(np.float32, copy=False)
    if not np.isfinite(base_raw).all():
        raise RuntimeError("base additive render is non-finite")

    target64 = source.astype(np.float64)
    working = base_raw.astype(np.float64)
    initial_residual = target64 - working
    initial_pixel_max = float(
        np.sqrt(np.max(np.mean(initial_residual * initial_residual, axis=2)[active]))
    )
    means: list[tuple[float, float]] = []
    colors: list[np.ndarray] = []
    trajectory: list[ResidualPursuitStep] = []
    radius = max(1, int(math.ceil(cfg.sigma_cutoff * cfg.scale_px)))
    kernel_updates = 0

    for step in range(1, cfg.tail_gaussians + 1):
        residual = target64 - working
        pixel_mse = np.mean(residual * residual, axis=2)
        flat_index = int(
            np.argmax(pixel_mse)
            if selection_mask is None
            else np.argmax(np.where(active, pixel_mse, -np.inf))
        )
        y, x = divmod(flat_index, width)
        coefficient = np.array(residual[y, x], dtype=np.float64, copy=True)
        if not np.isfinite(coefficient).all():
            raise RuntimeError("residual pursuit produced a non-finite coefficient")
        if float(np.max(np.abs(coefficient))) > cfg.coefficient_abs_limit:
            raise RuntimeError("residual pursuit coefficient exceeds coefficient_abs_limit")

        y0, y1 = max(0, y - radius), min(height, y + radius + 1)
        x0, x1 = max(0, x - radius), min(width, x + radius + 1)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        weight = np.exp(
            -0.5 * ((xx - x) ** 2 + (yy - y) ** 2) / (cfg.scale_px * cfg.scale_px)
        )
        if cfg.support_fade:
            weight = np.maximum(
                weight - math.exp(-0.5 * cfg.sigma_cutoff * cfg.sigma_cutoff),
                0.0,
            )
        working[y0:y1, x0:x1] += weight[:, :, None] * coefficient
        updates = int((y1 - y0) * (x1 - x0))
        kernel_updates += updates
        post_residual = target64 - working
        post_pixel_max = float(
            np.sqrt(np.max(np.mean(post_residual * post_residual, axis=2)[active]))
        )
        means.append((float(x), float(y)))
        colors.append(coefficient)
        trajectory.append(
            ResidualPursuitStep(
                step=step,
                x=x,
                y=y,
                selected_pixel_mse=float(pixel_mse[y, x]),
                selected_pixel_rmse=float(math.sqrt(pixel_mse[y, x])),
                coefficient_rgb=tuple(float(value) for value in coefficient),
                post_step_pixel_rmse_max=post_pixel_max,
                kernel_pixel_updates=updates,
            )
        )

    tail = GaussianField.from_numpy(
        np.asarray(means, dtype=np.float32),
        np.full((cfg.tail_gaussians, 2), cfg.scale_px, dtype=np.float32),
        np.zeros(cfg.tail_gaussians, dtype=np.float32),
        np.asarray(colors, dtype=np.float32),
        device=base.means.device,
        dtype=base.means.dtype,
    )
    endpoint = base.append(tail)
    with torch.no_grad():
        final_tensor = _render(endpoint, height, width, cfg)
    final_raw = final_tensor.detach().cpu().numpy().astype(np.float32, copy=False)
    analytic_raw = working.astype(np.float32)
    parity = float(np.max(np.abs(final_raw.astype(np.float64) - analytic_raw.astype(np.float64))))
    if not np.isfinite(final_raw).all() or parity > _PARITY_LIMIT:
        raise RuntimeError(
            f"residual-pursuit cold render failed finite/parity safety ({parity:.6g})"
        )
    final_residual = source.astype(np.float64) - final_raw.astype(np.float64)
    final_pixel_max = float(
        np.sqrt(np.max(np.mean(final_residual * final_residual, axis=2)[active]))
    )
    coefficient_abs_max = float(torch.max(torch.abs(endpoint.colors)).detach().cpu())
    fixed_geometry = bool(
        torch.all(tail.means == torch.round(tail.means))
        and torch.all(tail.rotations == 0.0)
        and torch.allclose(
            tail.scales(),
            torch.full_like(tail.scales(), cfg.scale_px),
            atol=1e-7,
            rtol=0.0,
        )
    )
    return ResidualPursuitAdditiveResult(
        base_field=base,
        tail_field=tail,
        field=endpoint,
        base_reconstruction_raw=base_raw,
        analytic_reconstruction_raw=analytic_raw,
        reconstruction_raw=final_raw,
        trajectory=tuple(trajectory),
        base_count=base.n,
        tail_count=tail.n,
        total_count=endpoint.n,
        residual_scan_pixel_evaluations=cfg.tail_gaussians * height * width,
        selection_mask_applied=selection_mask is not None,
        selection_active_pixels=int(active.sum()),
        tail_kernel_pixel_updates=kernel_updates,
        renderer_calls=2,
        base_prefix_bit_exact=_prefix_equal(endpoint, base),
        fixed_tail_geometry=fixed_geometry,
        training_payload_removed=_pure_payload(endpoint),
        analytic_render_parity_max_abs=parity,
        initial_pixel_rmse_max=initial_pixel_max,
        final_pixel_rmse_max=final_pixel_max,
        coefficient_abs_max=coefficient_abs_max,
        base_field_digest=_field_digest(base),
        tail_field_digest=_field_digest(tail),
        endpoint_field_digest=_field_digest(endpoint),
        status="completed",
        elapsed_seconds=time.perf_counter() - started,
    )


__all__ = [
    "ResidualPursuitAdditiveConfig",
    "ResidualPursuitAdditiveResult",
    "ResidualPursuitStep",
    "append_residual_pursuit_gaussians",
]
