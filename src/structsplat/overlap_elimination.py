"""Overlap-prefiltered pixel lattices and feature-safe WSE elimination (HIER-008).

The module is a deterministic, default-off research reference.  Topology analysis and appearance
solves are NumPy/SciPy CPU work; torch is imported lazily only by the optional post-reduction
optimizer.  All fields use direct additive, peak-one Observation Field V2 semantics.

The byte counts exposed by callers are structural proxies.  This module implements no bitstream,
quantizer, entropy model, header, or cold decoder and therefore makes no compression-rate claim.
"""
from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
import time
from typing import Iterable

import numpy as np

from .observation_field import (
    AlphaSemantics,
    CanvasCropTransform,
    FieldSemantics,
    ObservationField2D,
    SupportSemantics,
    adapt_direct_additive,
    pack_alpha,
)
from .progressive_residual_quadtree import progressive_artifact_metrics
from .sampling import _neighbor_pairs
from .structure_tensor import compute as compute_structure_tensor
from .structure_tensor import gaussian_blur


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


def _integer(value: object, name: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {result}")
    return result


def _image(image: object) -> np.ndarray:
    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a NumPy array")
    if image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) < 1:
        raise ValueError("image must have non-empty HWC RGB shape")
    if image.dtype.kind not in "fiu" or not np.isfinite(image).all():
        raise ValueError("image must contain finite numeric values")
    result = np.asarray(image, dtype=np.float32)
    if (result < 0.0).any() or (result > 1.0).any():
        raise ValueError("image values must lie in [0, 1]")
    return np.array(result, dtype=np.float32, order="C", copy=True)


def _mask(mask: object, shape: tuple[int, int], name: str = "mask") -> np.ndarray:
    if not isinstance(mask, np.ndarray) or mask.shape != shape or mask.dtype != np.bool_:
        raise ValueError(f"{name} must be a bool NumPy array with shape {shape}")
    result = np.array(mask, dtype=bool, order="C", copy=True)
    if not result.any():
        raise ValueError(f"{name} must contain at least one active pixel")
    return result


@dataclass(frozen=True)
class AppearanceSolveConfig:
    """Matrix-free least-squares configuration for fixed Gaussian geometry."""

    tolerance: float = 1e-8
    max_iterations: int = 200
    ridge: float = 1e-8

    def __post_init__(self) -> None:
        object.__setattr__(self, "tolerance", _finite(self.tolerance, "tolerance", strict=True))
        object.__setattr__(
            self,
            "max_iterations",
            _integer(self.max_iterations, "max_iterations", minimum=1),
        )
        object.__setattr__(self, "ridge", _finite(self.ridge, "ridge"))


@dataclass(frozen=True)
class FeatureEliminationConfig:
    """Static feature/Schur price plus dynamic WSE crowding configuration."""

    target_count: int
    alpha: float = 8.0
    density_base: float = 0.20
    density_power: float = 0.50
    radius_min: float = 0.65
    radius_max: float = 2.25
    rgb_barrier: float = 0.10
    feature_protection: float = 4.0
    schur_ridge: float = 1e-6

    def __post_init__(self) -> None:
        object.__setattr__(self, "target_count", _integer(self.target_count, "target_count", minimum=1))
        for name in ("alpha", "density_power", "radius_min", "radius_max"):
            object.__setattr__(self, name, _finite(getattr(self, name), name, strict=True))
        for name in ("rgb_barrier", "feature_protection", "schur_ridge"):
            object.__setattr__(self, name, _finite(getattr(self, name), name))
        base = _finite(self.density_base, "density_base", strict=True)
        if base > 1.0:
            raise ValueError("density_base must be <= 1")
        object.__setattr__(self, "density_base", base)
        if self.radius_min > self.radius_max:
            raise ValueError("radius_min must be <= radius_max")


@dataclass(frozen=True)
class ProtectedLeafSelection:
    """Deterministic thin/high-feature pixel leaves reserved from contraction."""

    protected_mask: np.ndarray
    priority: np.ndarray
    structure_feature: np.ndarray
    highpass_feature: np.ndarray
    requested_count: int
    selected_count: int
    nms_selected_count: int


@dataclass(frozen=True)
class FieldOptimizerConfig:
    """Bounded common post-reduction optimizer used by every HIER-008 arm."""

    steps: int = 80
    checkpoint_every: int = 10
    lr_rgb: float = 0.01
    lr_means: float = 0.003
    lr_log_scales: float = 0.002
    max_mean_shift: float = 0.35
    max_log_scale_shift: float = 0.15
    error_smoothing_sigma: float = 1.5
    error_weight: float = 2.0
    feature_weight: float = 2.0
    tail_fraction: float = 0.01
    tail_weight: float = 2.0
    pixel_threshold: float = 0.02
    patch7_threshold: float = 0.01
    sigma_cutoff: float = 3.0
    support_fade_alpha: float = 0.0
    coefficient_limit: float = 16.0
    seed: int = 0
    device: str = "cpu"
    renderer: str = "additive"
    render_chunk: int = 256

    def __post_init__(self) -> None:
        object.__setattr__(self, "steps", _integer(self.steps, "steps", minimum=0))
        object.__setattr__(
            self,
            "checkpoint_every",
            _integer(self.checkpoint_every, "checkpoint_every", minimum=1),
        )
        for name in (
            "lr_rgb",
            "lr_means",
            "lr_log_scales",
            "max_mean_shift",
            "max_log_scale_shift",
            "tail_fraction",
            "pixel_threshold",
            "patch7_threshold",
            "sigma_cutoff",
            "coefficient_limit",
        ):
            object.__setattr__(self, name, _finite(getattr(self, name), name, strict=True))
        for name in (
            "error_smoothing_sigma",
            "error_weight",
            "feature_weight",
            "tail_weight",
            "support_fade_alpha",
        ):
            object.__setattr__(self, name, _finite(getattr(self, name), name))
        if self.tail_fraction > 1.0:
            raise ValueError("tail_fraction must be <= 1")
        if self.support_fade_alpha > 1.0:
            raise ValueError("support_fade_alpha must be <= 1")
        object.__setattr__(self, "seed", _integer(self.seed, "seed", minimum=0))
        object.__setattr__(self, "render_chunk", _integer(self.render_chunk, "render_chunk", minimum=1))
        if self.renderer not in ("additive", "cuda_additive", "cuda_tiled_additive"):
            raise ValueError("unsupported additive renderer")
        if self.renderer.startswith("cuda") and not self.device.startswith("cuda"):
            raise ValueError("a CUDA renderer requires a CUDA device")


@dataclass(frozen=True)
class AppearanceSolveDiagnostics:
    iterations: int
    converged: bool
    relative_normal_residual_max: float
    data_sse: float
    data_pixel_rmse_max: float
    coefficient_abs_max: float
    coefficient_l2_rms: float
    negative_coefficient_fraction: float
    elapsed_seconds: float

    def to_record(self) -> dict[str, object]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class FeatureEliminationResult:
    survivors_by_count: dict[int, np.ndarray]
    removal_order: np.ndarray
    feature_normalized: np.ndarray
    density_relative: np.ndarray
    target_radius: np.ndarray
    schur_cost: np.ndarray
    schur_residual_fraction: np.ndarray
    eligible_neighbor_count: np.ndarray
    initial_crowding: np.ndarray
    elapsed_seconds: float


@dataclass(frozen=True)
class OptimizerCheckpoint:
    step: int
    selected: bool
    selectable: bool
    objective: float
    raw_sse: float
    raw_pixel_rmse_max: float
    raw_patch7_rmse_max: float
    raw_normalized_violation: float
    coefficient_abs_max: float
    mean_shift_max: float
    log_scale_shift_max: float
    elapsed_seconds: float

    def to_record(self) -> dict[str, object]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class FieldOptimizationResult:
    field: ObservationField2D
    reconstruction_raw: np.ndarray
    checkpoints: tuple[OptimizerCheckpoint, ...]
    selected_step: int
    optimizer_sse_gain: float
    optimizer_psnr_gain_db: float
    coefficient_abs_max: float
    mean_shift_max: float
    mean_shift_rms: float
    log_scale_shift_max: float
    log_scale_shift_rms: float
    elapsed_seconds: float


def gaussian_stencil(
    scale_px: float,
    *,
    sigma_cutoff: float = 3.0,
    support_fade_alpha: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return integer `(dy, dx)` offsets and peak-one finite-support weights."""

    scale = _finite(scale_px, "scale_px", strict=True)
    cutoff = _finite(sigma_cutoff, "sigma_cutoff", strict=True)
    fade = _finite(support_fade_alpha, "support_fade_alpha")
    if fade > 1.0:
        raise ValueError("support_fade_alpha must be <= 1")
    radius = max(int(math.ceil(cutoff * scale)), 1)
    offsets: list[tuple[int, int]] = []
    weights: list[float] = []
    tail = fade * math.exp(-0.5 * cutoff**2)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            weight = max(math.exp(-0.5 * (dx * dx + dy * dy) / (scale * scale)) - tail, 0.0)
            if weight > 0.0:
                offsets.append((dy, dx))
                weights.append(weight)
    return np.asarray(offsets, dtype=np.int32), np.asarray(weights, dtype=np.float64)


def _shift_add(source: np.ndarray, offsets: np.ndarray, weights: np.ndarray) -> np.ndarray:
    height, width = source.shape[:2]
    result = np.zeros_like(source, dtype=np.float64)
    for (delta_y, delta_x), weight in zip(offsets.tolist(), weights.tolist()):
        source_y0 = max(0, -delta_y)
        source_y1 = min(height, height - delta_y)
        source_x0 = max(0, -delta_x)
        source_x1 = min(width, width - delta_x)
        if source_y1 <= source_y0 or source_x1 <= source_x0:
            continue
        result[
            source_y0 + delta_y : source_y1 + delta_y,
            source_x0 + delta_x : source_x1 + delta_x,
            ...,
        ] += weight * source[source_y0:source_y1, source_x0:source_x1, ...]
    return result


def render_fixed_lattice(
    coefficients: np.ndarray,
    basis_mask: np.ndarray,
    *,
    scale_px: float,
    sigma_cutoff: float = 3.0,
    support_fade_alpha: float = 0.0,
) -> np.ndarray:
    """Render fixed pixel-centred isotropic Gaussians through the discrete reference kernel."""

    if not isinstance(basis_mask, np.ndarray) or basis_mask.ndim != 2:
        raise ValueError("basis_mask must be a 2D bool NumPy array")
    active = _mask(basis_mask, basis_mask.shape, "basis_mask")
    values = np.asarray(coefficients, dtype=np.float64)
    if values.shape != (int(active.sum()), 3) or not np.isfinite(values).all():
        raise ValueError(
            f"coefficients must be finite with shape ({int(active.sum())}, 3), got {values.shape}"
        )
    source = np.zeros((*active.shape, 3), dtype=np.float64)
    source[active] = values
    offsets, weights = gaussian_stencil(
        scale_px,
        sigma_cutoff=sigma_cutoff,
        support_fade_alpha=support_fade_alpha,
    )
    return _shift_add(source, offsets, weights)


def solve_fixed_lattice_appearance(
    image: np.ndarray,
    observation_mask: np.ndarray,
    basis_mask: np.ndarray,
    *,
    scale_px: float,
    sigma_cutoff: float = 3.0,
    support_fade_alpha: float = 0.0,
    config: AppearanceSolveConfig | None = None,
    initial_coefficients: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, AppearanceSolveDiagnostics]:
    """Solve RGB coefficients for fixed lattice geometry with matrix-free preconditioned CG.

    The rectangular least-squares problem is solved through `(A.T A + ridge I)c = A.T y`, where
    `A` is exactly the finite-support CPU lattice renderer sampled on `observation_mask`.
    """

    started = time.perf_counter()
    source = _image(image)
    observed = _mask(observation_mask, source.shape[:2], "observation_mask")
    basis = _mask(basis_mask, source.shape[:2], "basis_mask")
    if np.any(basis & ~observed):
        raise ValueError("basis_mask must be a subset of observation_mask")
    cfg = config or AppearanceSolveConfig()
    offsets, weights = gaussian_stencil(
        scale_px,
        sigma_cutoff=sigma_cutoff,
        support_fade_alpha=support_fade_alpha,
    )
    count = int(basis.sum())

    def scatter(values: np.ndarray) -> np.ndarray:
        grid = np.zeros((*basis.shape, 3), dtype=np.float64)
        grid[basis] = values
        return grid

    def forward(values: np.ndarray) -> np.ndarray:
        return _shift_add(scatter(values), offsets, weights)

    def adjoint(grid: np.ndarray) -> np.ndarray:
        return _shift_add(grid, offsets, weights)[basis]

    target_grid = np.zeros_like(source, dtype=np.float64)
    target_grid[observed] = source[observed].astype(np.float64)
    rhs = adjoint(target_grid)

    squared_response = _shift_add(observed.astype(np.float64), offsets, weights * weights)
    diagonal = squared_response[basis] + cfg.ridge
    diagonal = np.maximum(diagonal, np.finfo(np.float64).tiny)

    def normal(values: np.ndarray) -> np.ndarray:
        rendered = forward(values)
        rendered[~observed] = 0.0
        return adjoint(rendered) + cfg.ridge * values

    if initial_coefficients is None:
        x = source[basis].astype(np.float64)
    else:
        x = np.asarray(initial_coefficients, dtype=np.float64).copy()
        if x.shape != (count, 3) or not np.isfinite(x).all():
            raise ValueError(
                f"initial_coefficients must be finite with shape ({count}, 3), got {x.shape}"
            )
    residual = rhs - normal(x)
    z = residual / diagonal[:, None]
    direction = z.copy()
    residual_dot = np.sum(residual * z, axis=0)
    rhs_norm = np.sqrt(np.sum(rhs * rhs, axis=0))
    rhs_norm = np.maximum(rhs_norm, np.finfo(np.float64).tiny)
    relative = np.sqrt(np.sum(residual * residual, axis=0)) / rhs_norm
    iterations = 0
    converged_channels = relative <= cfg.tolerance
    for iteration in range(1, cfg.max_iterations + 1):
        if bool(np.all(converged_channels)):
            break
        product = normal(direction)
        denominator = np.sum(direction * product, axis=0)
        alpha = np.zeros(3, dtype=np.float64)
        movable = (~converged_channels) & (np.abs(denominator) > np.finfo(np.float64).tiny)
        alpha[movable] = residual_dot[movable] / denominator[movable]
        x += direction * alpha[None, :]
        residual -= product * alpha[None, :]
        relative = np.sqrt(np.sum(residual * residual, axis=0)) / rhs_norm
        converged_channels |= relative <= cfg.tolerance
        z = residual / diagonal[:, None]
        updated_dot = np.sum(residual * z, axis=0)
        beta = np.zeros(3, dtype=np.float64)
        movable = (~converged_channels) & (np.abs(residual_dot) > np.finfo(np.float64).tiny)
        beta[movable] = updated_dot[movable] / residual_dot[movable]
        direction = z + direction * beta[None, :]
        direction[:, converged_channels] = 0.0
        residual_dot = updated_dot
        iterations = iteration

    reconstruction = forward(x)
    data_residual = reconstruction[observed] - source[observed].astype(np.float64)
    pixel_rmse = np.sqrt(np.mean(data_residual * data_residual, axis=1))
    diagnostics = AppearanceSolveDiagnostics(
        iterations=iterations,
        converged=bool(np.all(relative <= cfg.tolerance)),
        relative_normal_residual_max=float(np.max(relative)),
        data_sse=float(np.sum(data_residual * data_residual)),
        data_pixel_rmse_max=float(np.max(pixel_rmse)),
        coefficient_abs_max=float(np.max(np.abs(x))),
        coefficient_l2_rms=float(math.sqrt(float(np.mean(x * x)))),
        negative_coefficient_fraction=float(np.mean(x < 0.0)),
        elapsed_seconds=time.perf_counter() - started,
    )
    return x.astype(np.float32), reconstruction.astype(np.float32), diagnostics


def _kernel_correlations(offsets: np.ndarray, weights: np.ndarray) -> dict[tuple[int, int], float]:
    values = {tuple(offset): float(weight) for offset, weight in zip(offsets.tolist(), weights.tolist())}
    correlations: dict[tuple[int, int], float] = {}
    for ay, ax in values:
        for by, bx in values:
            delta = (by - ay, bx - ax)
            correlations[delta] = correlations.get(delta, 0.0) + values[(ay, ax)] * values[(by, bx)]
    return correlations


def _feature_and_density(
    image: np.ndarray,
    mask: np.ndarray,
    *,
    density_base: float,
    density_power: float,
) -> tuple[np.ndarray, np.ndarray]:
    tensor = compute_structure_tensor(image)
    reference = float(tensor.energy_ref or 1.0)
    feature = np.clip(tensor.energy.astype(np.float64) / reference, 0.0, 1.0)
    feature[~mask] = 0.0
    density = density_base + (1.0 - density_base) * np.power(feature, density_power)
    density[~mask] = 0.0
    density_mean = float(np.mean(density[mask]))
    density_relative = np.zeros_like(density)
    density_relative[mask] = density[mask] / max(density_mean, np.finfo(np.float64).tiny)
    return feature.astype(np.float32), density_relative.astype(np.float32)


def select_protected_feature_leaves(
    image: np.ndarray,
    mask: np.ndarray,
    count: int,
    *,
    highpass_sigma_px: float = 1.0,
    nms_radius_px: int = 1,
) -> ProtectedLeafSelection:
    """Select an exact deterministic reserve of structurally important pixel leaves.

    The score is the maximum of normalized structure-tensor energy and normalized local RGB
    high-pass magnitude.  Stable score/y/x ordering plus Chebyshev NMS spreads the reserve across
    thin structures; if NMS cannot supply the requested count, the remaining highest-score pixels
    are filled deterministically so the topology contract receives an exact budget.
    """

    source = _image(image)
    active_mask = _mask(mask, source.shape[:2])
    requested = _integer(count, "count", minimum=0)
    active_count = int(active_mask.sum())
    if requested > active_count:
        raise ValueError(
            f"count cannot exceed the number of active mask pixels: {requested} > {active_count}"
        )
    sigma = _finite(highpass_sigma_px, "highpass_sigma_px", strict=True)
    radius = _integer(nms_radius_px, "nms_radius_px", minimum=0)

    structure_feature, _ = _feature_and_density(
        source,
        active_mask,
        density_base=0.20,
        density_power=0.50,
    )
    highpass = np.sqrt(
        np.mean(
            np.square(
                source.astype(np.float64)
                - gaussian_blur(source, sigma).astype(np.float64)
            ),
            axis=2,
        )
    )
    reference = float(np.quantile(highpass[active_mask], 0.99))
    highpass_feature = np.zeros(active_mask.shape, dtype=np.float32)
    if reference > np.finfo(np.float64).tiny:
        highpass_feature[active_mask] = np.clip(
            highpass[active_mask] / reference, 0.0, 1.0
        ).astype(np.float32)
    priority = np.maximum(structure_feature, highpass_feature).astype(np.float32)
    priority[~active_mask] = 0.0

    protected = np.zeros(active_mask.shape, dtype=bool)
    if requested == 0:
        return ProtectedLeafSelection(
            protected_mask=protected,
            priority=priority,
            structure_feature=structure_feature,
            highpass_feature=highpass_feature,
            requested_count=0,
            selected_count=0,
            nms_selected_count=0,
        )

    flat = np.flatnonzero(active_mask)
    yy, xx = np.divmod(flat, active_mask.shape[1])
    scores = priority.reshape(-1)[flat]
    order = np.lexsort((xx, yy, -scores))
    selected: list[tuple[int, int]] = []
    selected_flat: set[int] = set()
    for ordered_index in order:
        y = int(yy[ordered_index])
        x = int(xx[ordered_index])
        y0 = max(0, y - radius)
        y1 = min(active_mask.shape[0], y + radius + 1)
        x0 = max(0, x - radius)
        x1 = min(active_mask.shape[1], x + radius + 1)
        if protected[y0:y1, x0:x1].any():
            continue
        protected[y, x] = True
        selected.append((y, x))
        selected_flat.add(int(flat[ordered_index]))
        if len(selected) == requested:
            break
    nms_selected_count = len(selected)
    if len(selected) < requested:
        for ordered_index in order:
            flat_index = int(flat[ordered_index])
            if flat_index in selected_flat:
                continue
            y = int(yy[ordered_index])
            x = int(xx[ordered_index])
            protected[y, x] = True
            selected_flat.add(flat_index)
            if len(selected_flat) == requested:
                break
    if int(protected.sum()) != requested:
        raise RuntimeError("feature reserve failed to materialize the requested exact count")
    return ProtectedLeafSelection(
        protected_mask=protected,
        priority=priority,
        structure_feature=structure_feature,
        highpass_feature=highpass_feature,
        requested_count=requested,
        selected_count=int(protected.sum()),
        nms_selected_count=nms_selected_count,
    )


def feature_wse_schur_eliminate(
    image: np.ndarray,
    mask: np.ndarray,
    coefficients: np.ndarray,
    target_counts: Iterable[int],
    *,
    scale_px: float,
    sigma_cutoff: float = 3.0,
    support_fade_alpha: float = 0.0,
    config: FeatureEliminationConfig,
) -> FeatureEliminationResult:
    """Return nested feature-protected survivor masks at the requested exact counts."""

    started = time.perf_counter()
    source = _image(image)
    active_mask = _mask(mask, source.shape[:2])
    count = int(active_mask.sum())
    values = np.asarray(coefficients, dtype=np.float64)
    if values.shape != (count, 3) or not np.isfinite(values).all():
        raise ValueError(f"coefficients must be finite with shape ({count}, 3)")
    counts = sorted({_integer(value, "target count", minimum=1) for value in target_counts}, reverse=True)
    if not counts:
        raise ValueError("target_counts must not be empty")
    if counts[0] > count or counts[-1] != config.target_count:
        raise ValueError(
            "target counts must not exceed the active count and the smallest must equal "
            "FeatureEliminationConfig.target_count"
        )

    yy, xx = np.nonzero(active_mask)
    points = np.stack([xx, yy], axis=1).astype(np.float64)
    point_rgb = source[active_mask].astype(np.float64)
    feature_grid, density_grid = _feature_and_density(
        source,
        active_mask,
        density_base=config.density_base,
        density_power=config.density_power,
    )
    feature = feature_grid[active_mask].astype(np.float64)
    density_relative = density_grid[active_mask].astype(np.float64)
    base_radius = math.sqrt(count / (math.pi * config.target_count))
    target_radius = np.clip(
        base_radius / np.sqrt(np.maximum(density_relative, np.finfo(np.float64).tiny)),
        config.radius_min,
        config.radius_max,
    )

    offsets, kernel_weights = gaussian_stencil(
        scale_px,
        sigma_cutoff=sigma_cutoff,
        support_fade_alpha=support_fade_alpha,
    )
    correlations = _kernel_correlations(offsets, kernel_weights)
    self_energy = correlations[(0, 0)]
    position_to_index = {(int(x), int(y)): index for index, (x, y) in enumerate(points.tolist())}
    coefficient_energy = np.sum(values * values, axis=1)
    positive_energy = coefficient_energy[coefficient_energy > np.finfo(np.float64).tiny]
    median_energy = float(np.median(positive_energy)) if positive_energy.size else 1.0
    schur_cost = np.empty(count, dtype=np.float64)
    schur_fraction = np.empty(count, dtype=np.float64)
    eligible_count = np.empty(count, dtype=np.int16)
    neighbor_offsets = tuple(
        (dy, dx)
        for dy in (-1, 0, 1)
        for dx in (-1, 0, 1)
        if dx != 0 or dy != 0
    )
    for index, (x_value, y_value) in enumerate(points.astype(np.int64).tolist()):
        neighbor_ids = []
        for dy, dx in neighbor_offsets:
            neighbor = position_to_index.get((x_value + dx, y_value + dy))
            if neighbor is None:
                continue
            if float(np.max(np.abs(point_rgb[index] - point_rgb[neighbor]))) > config.rgb_barrier:
                continue
            neighbor_ids.append(neighbor)
        eligible_count[index] = len(neighbor_ids)
        if neighbor_ids:
            neighbor_points = points[np.asarray(neighbor_ids, dtype=np.int64)]
            gram = np.empty((len(neighbor_ids), len(neighbor_ids)), dtype=np.float64)
            cross = np.empty(len(neighbor_ids), dtype=np.float64)
            for row, neighbor_point in enumerate(neighbor_points):
                dy = int(neighbor_point[1] - points[index, 1])
                dx = int(neighbor_point[0] - points[index, 0])
                cross[row] = correlations.get((dy, dx), 0.0)
                for column, other_point in enumerate(neighbor_points):
                    delta_y = int(other_point[1] - neighbor_point[1])
                    delta_x = int(other_point[0] - neighbor_point[0])
                    gram[row, column] = correlations.get((delta_y, delta_x), 0.0)
            gram.flat[:: len(neighbor_ids) + 1] += config.schur_ridge * self_energy
            try:
                projected = float(cross @ np.linalg.solve(gram, cross))
            except np.linalg.LinAlgError:
                projected = float(cross @ (np.linalg.pinv(gram) @ cross))
            residual_fraction = np.clip((self_energy - projected) / self_energy, 0.0, 1.0)
        else:
            residual_fraction = 1.0
        appearance_energy = coefficient_energy[index] + (
            median_energy * config.feature_protection * feature[index]
        )
        schur_fraction[index] = residual_fraction
        schur_cost[index] = max(appearance_energy * self_energy * residual_fraction, 0.0)

    recv, contributor, crowding_contribution = _neighbor_pairs(
        points,
        target_radius,
        metric=None,
        alpha=config.alpha,
    )
    crowding = np.bincount(recv, weights=crowding_contribution, minlength=count).astype(np.float64)
    initial_crowding = crowding.copy()
    positive_cost = schur_cost[schur_cost > np.finfo(np.float64).tiny]
    cost_scale = float(np.median(positive_cost)) if positive_cost.size else 1.0
    normalized_cost = schur_cost / max(cost_scale, np.finfo(np.float64).tiny)

    by_contributor = np.argsort(contributor, kind="stable")
    receiver_of = recv[by_contributor].tolist()
    contribution_of = crowding_contribution[by_contributor].tolist()
    sorted_contributor = contributor[by_contributor]
    indptr = np.searchsorted(sorted_contributor, np.arange(count + 1, dtype=np.int64)).tolist()

    alive = np.ones(count, dtype=bool)
    version = np.zeros(count, dtype=np.int64)

    def removal_priority(index: int) -> float:
        return float(crowding[index] / (0.05 + normalized_cost[index]))

    heap = [(-removal_priority(index), index, 0) for index in range(count)]
    heapq.heapify(heap)
    requested = set(counts)
    survivors_by_count: dict[int, np.ndarray] = {}
    if count in requested:
        survivors_by_count[count] = active_mask.copy()
    removal_order: list[int] = []
    remaining = count
    while remaining > counts[-1]:
        _, index, candidate_version = heapq.heappop(heap)
        if not alive[index] or candidate_version != version[index]:
            continue
        alive[index] = False
        removal_order.append(index)
        remaining -= 1
        for edge in range(indptr[index], indptr[index + 1]):
            receiver = receiver_of[edge]
            if not alive[receiver]:
                continue
            crowding[receiver] = max(crowding[receiver] - contribution_of[edge], 0.0)
            version[receiver] += 1
            heapq.heappush(
                heap,
                (-removal_priority(receiver), receiver, int(version[receiver])),
            )
        if remaining in requested:
            survivor_mask = np.zeros_like(active_mask)
            survivor_mask[yy[alive], xx[alive]] = True
            survivors_by_count[remaining] = survivor_mask

    missing = sorted(set(counts) - set(survivors_by_count))
    if missing:
        raise RuntimeError(f"feature elimination did not materialize requested counts {missing}")
    return FeatureEliminationResult(
        survivors_by_count=survivors_by_count,
        removal_order=np.asarray(removal_order, dtype=np.int32),
        feature_normalized=feature_grid,
        density_relative=density_grid,
        target_radius=target_radius.astype(np.float32),
        schur_cost=schur_cost.astype(np.float32),
        schur_residual_fraction=schur_fraction.astype(np.float32),
        eligible_neighbor_count=eligible_count,
        initial_crowding=initial_crowding.astype(np.float32),
        elapsed_seconds=time.perf_counter() - started,
    )


def lattice_observation_field(
    mask: np.ndarray,
    basis_mask: np.ndarray,
    coefficients: np.ndarray,
    *,
    scale_px: float,
    sigma_cutoff: float = 3.0,
    support_fade_alpha: float = 0.0,
) -> ObservationField2D:
    """Materialize fixed lattice survivors as a direct-additive Observation Field V2."""

    if not isinstance(mask, np.ndarray) or mask.ndim != 2:
        raise ValueError("mask must be a 2D bool NumPy array")
    scale = _finite(scale_px, "scale_px", strict=True)
    active_mask = _mask(mask, mask.shape)
    basis = _mask(basis_mask, active_mask.shape, "basis_mask")
    if np.any(basis & ~active_mask):
        raise ValueError("basis_mask must be a subset of mask")
    values = np.asarray(coefficients, dtype=np.float32)
    if values.shape != (int(basis.sum()), 3) or not np.isfinite(values).all():
        raise ValueError(f"coefficients must be finite with shape ({int(basis.sum())}, 3)")
    yy, xx = np.nonzero(basis)
    means = np.stack([xx, yy], axis=1).astype(np.float32)
    log_scales = np.full((len(means), 2), math.log(scale), dtype=np.float32)
    rotations = np.zeros(len(means), dtype=np.float32)
    packed_alpha = None if active_mask.all() else pack_alpha(active_mask)
    alpha = AlphaSemantics() if packed_alpha is None else AlphaSemantics(
        payload_encoding="binary_exact_packbits_little",
        matting_mode="multiply_alpha",
        boundary_policy="unconstrained",
    )
    semantics = FieldSemantics(
        coefficient_domain="signed",
        support=SupportSemantics(
            mode="axis_aligned_bbox",
            sigma_cutoff=sigma_cutoff,
            fade_alpha=support_fade_alpha,
            minimum_radius_px=1,
        ),
        alpha=alpha,
    )
    adaptation = adapt_direct_additive(
        means_xy=means,
        log_scales_xy=log_scales,
        rotations_rad=rotations,
        rgb_coeff=values,
        canvas_crop=CanvasCropTransform(
            canvas_width=active_mask.shape[1],
            canvas_height=active_mask.shape[0],
            crop_x=0,
            crop_y=0,
            crop_width=active_mask.shape[1],
            crop_height=active_mask.shape[0],
        ),
        semantics=semantics,
        packed_alpha=packed_alpha,
    )
    return adaptation.require_pixel_exact()


def _field_with_arrays(
    field: ObservationField2D,
    means: np.ndarray,
    log_scales: np.ndarray,
    coefficients: np.ndarray,
) -> ObservationField2D:
    adaptation = adapt_direct_additive(
        means_xy=np.asarray(means, dtype=np.float32),
        log_scales_xy=np.asarray(log_scales, dtype=np.float32),
        rotations_rad=np.asarray(field.rotations_rad, dtype=np.float32),
        rgb_coeff=np.asarray(coefficients, dtype=np.float32),
        canvas_crop=field.canvas_crop,
        semantics=field.semantics,
        packed_alpha=None if field.packed_alpha is None else np.array(field.packed_alpha, copy=True),
        background_rgb=(
            None if field.background_rgb is None else np.array(field.background_rgb, copy=True)
        ),
    )
    return adaptation.require_pixel_exact()


def _mask_aware_error_weight(
    reconstruction: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    feature: np.ndarray,
    config: FieldOptimizerConfig,
) -> np.ndarray:
    residual_energy = np.mean(
        (reconstruction.astype(np.float64) - target.astype(np.float64)) ** 2,
        axis=2,
    ).astype(np.float32)
    if config.error_smoothing_sigma > 0.0:
        mask_float = mask.astype(np.float32)
        numerator = gaussian_blur(residual_energy * mask_float, config.error_smoothing_sigma)
        denominator = gaussian_blur(mask_float, config.error_smoothing_sigma)
        smoothed = np.zeros_like(residual_energy)
        np.divide(numerator, denominator, out=smoothed, where=denominator > 1e-8)
    else:
        smoothed = residual_energy
    mean_error = float(np.mean(smoothed[mask]))
    relative_error = np.zeros_like(smoothed, dtype=np.float32)
    if mean_error > np.finfo(np.float32).tiny:
        relative_error[mask] = np.sqrt(np.maximum(smoothed[mask] / mean_error, 0.0))
    relative_error = np.clip(relative_error, 0.0, 4.0)
    weight = np.zeros_like(smoothed, dtype=np.float32)
    weight[mask] = (
        1.0
        + config.error_weight * relative_error[mask]
        + config.feature_weight * feature[mask]
    )
    weight[mask] /= max(float(np.mean(weight[mask])), np.finfo(np.float32).tiny)
    return weight


def _objective_value(
    reconstruction: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    weight: np.ndarray,
    tail_fraction: float,
    tail_weight: float,
) -> float:
    pixel_mse = np.mean(
        (reconstruction.astype(np.float64) - target.astype(np.float64)) ** 2,
        axis=2,
    )
    foreground = pixel_mse[mask]
    weighted = float(np.mean(foreground * weight[mask]))
    tail_count = max(1, int(math.ceil(tail_fraction * foreground.size)))
    tail = float(np.mean(np.partition(foreground, foreground.size - tail_count)[-tail_count:]))
    return weighted + tail_weight * tail


def optimize_observation_field(
    field: ObservationField2D,
    image: np.ndarray,
    mask: np.ndarray,
    *,
    feature_normalized: np.ndarray | None = None,
    config: FieldOptimizerConfig | None = None,
) -> FieldOptimizationResult:
    """Optimize all field rows inside fixed trust regions with artifact-safe checkpointing."""

    started = time.perf_counter()
    if not isinstance(field, ObservationField2D):
        raise TypeError("field must be an ObservationField2D")
    if field.semantics.renderer_equation != "additive_rgb_peak_one_v1":
        raise ValueError("the common optimizer requires a direct additive field")
    if field.semantics.coefficient_domain != "signed":
        raise ValueError("the common optimizer requires signed coefficients")
    target = _image(image)
    active_mask = _mask(mask, target.shape[:2])
    cfg = config or FieldOptimizerConfig()
    if (field.canvas_crop.crop_height, field.canvas_crop.crop_width) != target.shape[:2]:
        raise ValueError("field crop and target shape differ")
    if (
        field.canvas_crop.crop_x != 0
        or field.canvas_crop.crop_y != 0
        or field.canvas_crop.canvas_width != field.canvas_crop.crop_width
        or field.canvas_crop.canvas_height != field.canvas_crop.crop_height
    ):
        raise ValueError("the common optimizer currently requires a full-canvas zero-origin crop")
    support = field.semantics.support
    if not math.isclose(cfg.sigma_cutoff, support.sigma_cutoff, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("optimizer sigma_cutoff must match the field support semantics")
    if not math.isclose(
        cfg.support_fade_alpha,
        support.fade_alpha,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("optimizer support_fade_alpha must match the field support semantics")
    if field.background_rgb is not None or field.filter_variance_px2 is not None:
        raise ValueError("the common optimizer does not support background or covariance filtering")
    if field.semantics.filtering.aa_dilation_px2 != 0.0:
        raise ValueError("the common optimizer requires zero AA covariance dilation")
    if field.packed_alpha is not None and not np.array_equal(field.alpha_mask(), active_mask):
        raise ValueError("field alpha mask must match the optimizer observation mask")
    if feature_normalized is None:
        feature, _ = _feature_and_density(target, active_mask, density_base=0.2, density_power=0.5)
    else:
        feature = np.asarray(feature_normalized, dtype=np.float32)
        if feature.shape != active_mask.shape or not np.isfinite(feature).all():
            raise ValueError("feature_normalized must be finite and match the image")
        feature = np.clip(feature, 0.0, 1.0)

    import torch

    from .gaussians import GaussianField
    from .render import render_field

    torch.manual_seed(cfg.seed)
    if cfg.device.startswith("cuda"):
        torch.cuda.manual_seed_all(cfg.seed)
    gaussian = GaussianField.from_numpy(
        np.array(field.means_xy, copy=True),
        np.exp(np.array(field.log_scales_xy, copy=True)),
        np.array(field.rotations_rad, copy=True),
        np.array(field.rgb_coeff, copy=True),
        device=cfg.device,
    )
    initial_coefficient_abs_max = float(torch.max(torch.abs(gaussian.colors)).item())
    if initial_coefficient_abs_max > cfg.coefficient_limit:
        raise ValueError(
            "initial coefficient magnitude exceeds the optimizer stability limit: "
            f"{initial_coefficient_abs_max} > {cfg.coefficient_limit}"
        )
    gaussian.means.requires_grad_(True)
    gaussian.log_scales.requires_grad_(True)
    gaussian.colors.requires_grad_(True)
    base_means = gaussian.means.detach().clone()
    base_log_scales = gaussian.log_scales.detach().clone()
    optimizer = torch.optim.Adam(
        [
            {"params": [gaussian.colors], "lr": cfg.lr_rgb},
            {"params": [gaussian.means], "lr": cfg.lr_means},
            {"params": [gaussian.log_scales], "lr": cfg.lr_log_scales},
        ]
    )
    target_tensor = torch.as_tensor(target, device=cfg.device, dtype=torch.float32)
    mask_tensor = torch.as_tensor(active_mask, device=cfg.device, dtype=torch.bool)

    def render() -> "torch.Tensor":
        return render_field(
            gaussian.means,
            gaussian.conics(),
            gaussian.colors,
            gaussian.radii(cfg.sigma_cutoff),
            target.shape[0],
            target.shape[1],
            chunk=cfg.render_chunk,
            mode=cfg.renderer,
            opacities=None,
            scales=gaussian.scales(),
            rotations=gaussian.rotations,
            support_fade=cfg.support_fade_alpha > 0.0,
            sigma_cutoff=cfg.sigma_cutoff,
            support_fade_alpha=cfg.support_fade_alpha,
        )

    with torch.no_grad():
        initial_reconstruction = render().detach().cpu().numpy().astype(np.float32)
    weight = _mask_aware_error_weight(
        initial_reconstruction,
        target,
        active_mask,
        feature,
        cfg,
    )
    weight_tensor = torch.as_tensor(weight, device=cfg.device, dtype=torch.float32)
    initial_metrics = progressive_artifact_metrics(
        initial_reconstruction,
        target,
        active_mask,
        pixel_threshold=cfg.pixel_threshold,
        patch7_threshold=cfg.patch7_threshold,
        displayed=False,
    )
    initial_objective = _objective_value(
        initial_reconstruction,
        target,
        active_mask,
        weight,
        cfg.tail_fraction,
        cfg.tail_weight,
    )
    best_sse = float(initial_metrics["sse"])
    best_step = 0
    best_state = (
        gaussian.means.detach().clone(),
        gaussian.log_scales.detach().clone(),
        gaussian.colors.detach().clone(),
    )
    raw_checkpoint_data: list[dict[str, object]] = [
        {
            "step": 0,
            "selectable": True,
            "objective": initial_objective,
            "metrics": initial_metrics,
            "coefficient_abs_max": float(torch.max(torch.abs(gaussian.colors)).item()),
            "mean_shift_max": 0.0,
            "log_scale_shift_max": 0.0,
            "elapsed_seconds": time.perf_counter() - started,
        }
    ]
    comparison_tolerance = 32.0 * float(np.finfo(np.float32).eps)

    for step in range(1, cfg.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        reconstruction = render()
        pixel_mse = torch.mean((reconstruction - target_tensor) ** 2, dim=2)
        foreground = pixel_mse[mask_tensor]
        weighted_loss = torch.mean(foreground * weight_tensor[mask_tensor])
        tail_count = max(1, int(math.ceil(cfg.tail_fraction * int(active_mask.sum()))))
        tail_loss = torch.topk(foreground, tail_count, largest=True, sorted=False).values.mean()
        loss = weighted_loss + cfg.tail_weight * tail_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [gaussian.colors, gaussian.means, gaussian.log_scales],
            max_norm=10.0,
        )
        optimizer.step()
        with torch.no_grad():
            gaussian.means.copy_(
                torch.maximum(
                    torch.minimum(gaussian.means, base_means + cfg.max_mean_shift),
                    base_means - cfg.max_mean_shift,
                )
            )
            gaussian.log_scales.copy_(
                torch.maximum(
                    torch.minimum(
                        gaussian.log_scales,
                        base_log_scales + cfg.max_log_scale_shift,
                    ),
                    base_log_scales - cfg.max_log_scale_shift,
                )
            )
            gaussian.colors.clamp_(-cfg.coefficient_limit, cfg.coefficient_limit)

        if step % cfg.checkpoint_every != 0 and step != cfg.steps:
            continue
        with torch.no_grad():
            checkpoint_reconstruction = render().detach().cpu().numpy().astype(np.float32)
        metrics = progressive_artifact_metrics(
            checkpoint_reconstruction,
            target,
            active_mask,
            pixel_threshold=cfg.pixel_threshold,
            patch7_threshold=cfg.patch7_threshold,
            displayed=False,
        )
        coefficient_abs_max = float(torch.max(torch.abs(gaussian.colors)).item())
        mean_shift_max = float(torch.max(torch.abs(gaussian.means - base_means)).item())
        log_scale_shift_max = float(
            torch.max(torch.abs(gaussian.log_scales - base_log_scales)).item()
        )
        pixel_limit = float(initial_metrics["pixel_rmse_max"]) + comparison_tolerance * max(
            1.0, float(initial_metrics["pixel_rmse_max"])
        )
        patch_limit = float(initial_metrics["patch7_rmse_max"]) + comparison_tolerance * max(
            1.0, float(initial_metrics["patch7_rmse_max"])
        )
        selectable = bool(
            np.isfinite(checkpoint_reconstruction).all()
            and coefficient_abs_max <= cfg.coefficient_limit
            and mean_shift_max <= cfg.max_mean_shift + comparison_tolerance
            and log_scale_shift_max <= cfg.max_log_scale_shift + comparison_tolerance
            and float(metrics["pixel_rmse_max"]) <= pixel_limit
            and float(metrics["patch7_rmse_max"]) <= patch_limit
            and float(metrics["sse"]) < float(initial_metrics["sse"]) - comparison_tolerance
        )
        objective = _objective_value(
            checkpoint_reconstruction,
            target,
            active_mask,
            weight,
            cfg.tail_fraction,
            cfg.tail_weight,
        )
        raw_checkpoint_data.append(
            {
                "step": step,
                "selectable": selectable,
                "objective": objective,
                "metrics": metrics,
                "coefficient_abs_max": coefficient_abs_max,
                "mean_shift_max": mean_shift_max,
                "log_scale_shift_max": log_scale_shift_max,
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
        if selectable and float(metrics["sse"]) < best_sse - comparison_tolerance:
            best_sse = float(metrics["sse"])
            best_step = step
            best_state = (
                gaussian.means.detach().clone(),
                gaussian.log_scales.detach().clone(),
                gaussian.colors.detach().clone(),
            )
        if step < cfg.steps:
            weight = _mask_aware_error_weight(
                checkpoint_reconstruction,
                target,
                active_mask,
                feature,
                cfg,
            )
            weight_tensor = torch.as_tensor(weight, device=cfg.device, dtype=torch.float32)

    with torch.no_grad():
        gaussian.means.copy_(best_state[0])
        gaussian.log_scales.copy_(best_state[1])
        gaussian.colors.copy_(best_state[2])
        selected_reconstruction = render().detach().cpu().numpy().astype(np.float32)
    selected_field = _field_with_arrays(
        field,
        gaussian.means.detach().cpu().numpy(),
        gaussian.log_scales.detach().cpu().numpy(),
        gaussian.colors.detach().cpu().numpy(),
    )
    selected_means = gaussian.means.detach() - base_means
    selected_scales = gaussian.log_scales.detach() - base_log_scales
    mean_shift_max = float(torch.max(torch.abs(selected_means)).item())
    mean_shift_rms = float(torch.sqrt(torch.mean(selected_means * selected_means)).item())
    log_scale_shift_max = float(torch.max(torch.abs(selected_scales)).item())
    log_scale_shift_rms = float(torch.sqrt(torch.mean(selected_scales * selected_scales)).item())
    coefficient_abs_max = float(torch.max(torch.abs(gaussian.colors)).item())
    final_metrics = progressive_artifact_metrics(
        selected_reconstruction,
        target,
        active_mask,
        pixel_threshold=cfg.pixel_threshold,
        patch7_threshold=cfg.patch7_threshold,
        displayed=False,
    )
    initial_mse = float(initial_metrics["sse"]) / (3.0 * float(active_mask.sum()))
    final_mse = float(final_metrics["sse"]) / (3.0 * float(active_mask.sum()))
    psnr_gain = 10.0 * math.log10(max(initial_mse, 1e-12) / max(final_mse, 1e-12))
    checkpoints = tuple(
        OptimizerCheckpoint(
            step=int(record["step"]),
            selected=int(record["step"]) == best_step,
            selectable=bool(record["selectable"]),
            objective=float(record["objective"]),
            raw_sse=float(record["metrics"]["sse"]),
            raw_pixel_rmse_max=float(record["metrics"]["pixel_rmse_max"]),
            raw_patch7_rmse_max=float(record["metrics"]["patch7_rmse_max"]),
            raw_normalized_violation=float(record["metrics"]["normalized_violation"]),
            coefficient_abs_max=float(record["coefficient_abs_max"]),
            mean_shift_max=float(record["mean_shift_max"]),
            log_scale_shift_max=float(record["log_scale_shift_max"]),
            elapsed_seconds=float(record["elapsed_seconds"]),
        )
        for record in raw_checkpoint_data
    )
    return FieldOptimizationResult(
        field=selected_field,
        reconstruction_raw=selected_reconstruction,
        checkpoints=checkpoints,
        selected_step=best_step,
        optimizer_sse_gain=float(initial_metrics["sse"]) - float(final_metrics["sse"]),
        optimizer_psnr_gain_db=psnr_gain,
        coefficient_abs_max=coefficient_abs_max,
        mean_shift_max=mean_shift_max,
        mean_shift_rms=mean_shift_rms,
        log_scale_shift_max=log_scale_shift_max,
        log_scale_shift_rms=log_scale_shift_rms,
        elapsed_seconds=time.perf_counter() - started,
    )
