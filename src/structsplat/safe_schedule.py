"""Transactional, phase-ordered optimization for masked or unmasked 2D Gaussian fields.

The legacy fitter exposes all individual mechanisms (birth, split, merge, relocation, pruning),
but a fixed timer order can spend capacity before the operation with the largest useful delta
gets a chance to run.  This module implements the production schedule described by the Janelle
boundary investigation:

* fixed-topology bootstrap;
* coverage-first birth;
* detail birth / moment-preserving split;
* explicit boundary closure for masked inputs, or count-matched general closure otherwise;
* count-neutral redistribution at capacity;
* fixed-topology polish.

Every optimizer block and every topology proposal runs on a detached trial field.  A full-frame
metric vector decides whether both parameters and optimizer moments are committed or discarded.
Topology batches are reduced geometrically after a rejected trial, so the schedule can retain
safe members of a coarse proposal without falling back to per-Gaussian Python mutation.

Fixed-capacity storage is orthogonal to that policy (FIT-024/ADR-0021). In ``fixed_capacity``
mode, the same proposals and gates operate on a contiguous active prefix of capacity-shaped field
and Adam tensors; only activation metadata changes at growth commits, and compaction happens once
at the output boundary. Detached transaction/checkpoint snapshots remain capacity-shaped
allocations. FIT-021's triage policy remains a separate legacy mode.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, replace
import math
import time
from typing import Any, Callable, Iterable

import torch
import torch.nn.functional as F

from .config import FitConfig
from .gaussians import GaussianField
from .fit import (
    _MIN_DENSIFY_SCALE,
    _MaskConstraint,
    _image_gradients,
    _logit,
    _raw_weight_map_field,
    _render,
    _responsibility_error_density_scores,
    _solve_colors_normalized,
    fit,
)
from .triage import _mutual_nearest_pairs, _row_covariances, attribution_pass
from .pool import (
    geometric_capacity_schedule,
    grow_transactional_capacity,
    prepare_transactional_pool,
)


@dataclass(frozen=True)
class QualityMetrics:
    """Unweighted, reader-facing metrics used by the safe commit gate."""

    n_gaussians: int
    foreground_mse: float
    boundary_mse: float
    cvar99_mse: float
    p99_mse: float
    interior_hole_fraction: float
    boundary_hole_fraction: float
    outside_max_abs: float
    outside_coverage_max: float
    finite: bool

    @property
    def foreground_psnr_db(self) -> float:
        return -10.0 * math.log10(max(self.foreground_mse, 1e-12))

    @property
    def boundary_psnr_db(self) -> float:
        return -10.0 * math.log10(max(self.boundary_mse, 1e-12))

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["foreground_psnr_db"] = self.foreground_psnr_db
        value["boundary_psnr_db"] = self.boundary_psnr_db
        return value


@dataclass(frozen=True)
class CommitTolerances:
    """Numerical slack only; these are not quality trade-off budgets."""

    mse_relative: float = 2e-6
    mse_absolute: float = 1e-10
    tail_relative: float = 2e-5
    guard_p99: bool = False
    hole_absolute: float = 0.0
    outside_absolute: float = 1e-8
    minimum_relative_gain: float = 1e-8


@dataclass(frozen=True)
class PhaseBudget:
    name: str
    max_steps: int
    block_steps: int
    target_gaussians: int | None
    learning_rates: tuple[float, float, float, float, float]
    interior_floor_weight: float
    boundary_floor_weight: float
    lowpass_downsample: int = 1


@dataclass
class SafeScheduleConfig:
    """Schedule policy. Iteration budgets are ceilings; metric transitions may end earlier."""

    capacity: int = 11_000
    storage_policy: str = "dynamic"
    boundary_enabled: bool = True
    base_active_limit: int | None = None
    growth_factor: float = 2.0
    initial_capacity: int | None = None
    detail_tail_max_rows: int = 0
    detail_tail_batch_rows: int = 128
    detail_tail_min_gain_per_row: float = 0.0
    coverage_target_gaussians: int = 8_000
    detail_target_gaussians: int = 10_000
    coverage_tau: float = 0.05
    boundary_band: float = 4.0
    interior_hole_target: float = 1e-3
    boundary_hole_target: float = 1e-2
    stale_patience: int = 3
    event_min_count: int = 8
    recovery_steps: int = 80
    event_spacing_px: float = 5.0
    event_oversample: float = 8.0
    coverage_birth_count: int = 512
    detail_birth_count: int = 256
    detail_split_count: int = 128
    boundary_birth_count: int = 128
    redistribution_count: int = 64
    redistribution_min_responsibility: float = 0.25
    split_shrink: float = 0.60
    hole_opacity: float = 0.90
    covered_birth_opacity: float = 0.08
    merge_envelope_inflation: float = 1.05
    refinement_policy: str = "global"
    local_start_phase: str = "coverage"
    local_seed_count: int = 384
    local_neighbor_count: int = 8
    topology_neighbor_count: int = 8
    boundary_recycle_at_capacity: bool = False
    pareto_safe_checkpoints: bool = False
    pareto_checkpoint_every: int = 50
    event_color_solve: bool = False
    boundary_residual_mse_threshold: float = 1e-2
    boundary_residual_component_target: int = 0
    boundary_residual_min_pixels: int = 4
    tolerances: CommitTolerances = CommitTolerances()
    # ADR-0026: interior coverage a block may give up and still commit. 0.0 keeps the historical
    # strict gate; a positive value trades a bounded amount of interior hole fraction for the
    # pixel-error progress the strict gate discards. Boundary holes are never budgeted.
    hole_regression_budget: float = 0.0

    bootstrap: PhaseBudget = PhaseBudget(
        "bootstrap", 1_250, 250, 5_000,
        (3e-2, 2e-2, 8e-3, 3e-2, 1e-2),
        0.015, 0.025, 2,
    )
    coverage: PhaseBudget = PhaseBudget(
        "coverage_growth", 3_000, 250, 8_000,
        (3e-2, 2e-2, 8e-3, 3e-2, 1e-2),
        0.025, 0.040,
    )
    detail: PhaseBudget = PhaseBudget(
        "detail_growth", 3_000, 250, 10_000,
        (1.5e-2, 1e-2, 4e-3, 2e-2, 6e-3),
        0.010, 0.020,
    )
    boundary: PhaseBudget = PhaseBudget(
        "boundary_closure", 2_000, 250, 11_000,
        (1e-2, 8e-3, 3e-3, 1.5e-2, 5e-3),
        0.005, 0.050,
    )
    redistribution: PhaseBudget = PhaseBudget(
        "redistribution", 2_000, 250, 11_000,
        (8e-3, 6e-3, 2e-3, 1e-2, 3e-3),
        0.005, 0.015,
    )
    polish: PhaseBudget = PhaseBudget(
        "safe_polish", 2_000, 250, 11_000,
        (5e-3, 3e-3, 1e-3, 3e-3, 1e-3),
        0.002, 0.005,
    )

    def resolved_base_active_limit(self) -> int:
        """Return the logical ceiling used before the optional detail tail."""

        return (
            int(self.capacity)
            if self.base_active_limit is None
            else int(self.base_active_limit)
        )

    def resolved_initial_capacity(self, initial_n: int) -> int:
        """Physical row count a ``geometric`` fit starts at before growing toward ``capacity``."""

        if self.initial_capacity is None:
            return int(initial_n)
        return int(self.initial_capacity)

    def max_block_births(self) -> int:
        """Largest row count a single topology block can activate (the headroom to reserve)."""

        return max(
            int(self.coverage_birth_count),
            int(self.detail_birth_count),
            int(self.detail_split_count),
            int(self.boundary_birth_count),
            int(self.redistribution_count),
            int(self.event_min_count),
            1,
        )

    def validate(self, initial_n: int) -> None:
        if initial_n <= 0:
            raise ValueError("safe schedule needs a non-empty initial field")
        base_active_limit = self.resolved_base_active_limit()
        if not initial_n <= self.coverage_target_gaussians <= self.detail_target_gaussians:
            raise ValueError(
                "expected initial_n <= coverage_target_gaussians <= "
                "detail_target_gaussians"
            )
        if self.detail_target_gaussians > base_active_limit:
            raise ValueError(
                "detail_target_gaussians cannot exceed base_active_limit"
            )
        if self.capacity < initial_n:
            raise ValueError(
                f"capacity {self.capacity} is below initial field size {initial_n}"
            )
        if not initial_n <= base_active_limit <= self.capacity:
            raise ValueError(
                "base_active_limit must be between the initial field size and capacity"
            )
        if self.storage_policy not in ("dynamic", "fixed_capacity", "geometric"):
            raise ValueError(
                "storage_policy must be 'dynamic', 'fixed_capacity', or 'geometric'"
            )
        if self.storage_policy == "geometric":
            if not math.isfinite(self.growth_factor) or self.growth_factor <= 1.0:
                raise ValueError("growth_factor must be finite and greater than one")
            initial_capacity = self.resolved_initial_capacity(initial_n)
            if not initial_n <= initial_capacity <= self.capacity:
                raise ValueError(
                    "initial_capacity must be between the initial field size and capacity"
                )
        if self.detail_tail_max_rows < 0:
            raise ValueError("detail_tail_max_rows must be nonnegative")
        if self.detail_tail_batch_rows <= 0:
            raise ValueError("detail_tail_batch_rows must be positive")
        if self.detail_tail_min_gain_per_row < 0.0:
            raise ValueError(
                "detail_tail_min_gain_per_row must be nonnegative"
            )
        if base_active_limit + self.detail_tail_max_rows > self.capacity:
            raise ValueError(
                "base_active_limit + detail_tail_max_rows cannot exceed capacity"
            )
        if self.detail_tail_max_rows > 0:
            if self.storage_policy != "fixed_capacity":
                raise ValueError(
                    "detail tail reserve requires storage_policy='fixed_capacity'"
                )
            if self.detail_tail_max_rows < self.event_min_count:
                raise ValueError(
                    "detail_tail_max_rows must be at least event_min_count"
                )
            if self.detail_tail_batch_rows < self.event_min_count:
                raise ValueError(
                    "detail_tail_batch_rows must be at least event_min_count"
                )
        if self.coverage_tau <= 0.0:
            raise ValueError("coverage_tau must be positive")
        if self.event_min_count <= 0 or self.recovery_steps <= 0:
            raise ValueError("event_min_count and recovery_steps must be positive")
        if not 0.0 < self.redistribution_min_responsibility <= 1.0:
            raise ValueError("redistribution_min_responsibility must be in (0, 1]")
        if self.refinement_policy not in ("global", "local_neighborhood"):
            raise ValueError(
                "refinement_policy must be 'global' or 'local_neighborhood'"
            )
        if self.local_start_phase not in (
            "coverage",
            "detail",
            "boundary",
            "redistribution",
            "polish",
        ):
            raise ValueError(
                "local_start_phase must be coverage, detail, boundary, "
                "redistribution, or polish"
            )
        if (
            self.local_seed_count <= 0
            or self.local_neighbor_count < 0
            or self.topology_neighbor_count < 0
        ):
            raise ValueError(
                "local_seed_count must be positive and neighbour counts nonnegative"
            )
        if self.boundary_residual_mse_threshold < 0.0:
            raise ValueError("boundary_residual_mse_threshold must be nonnegative")
        if self.pareto_checkpoint_every <= 0:
            raise ValueError("pareto_checkpoint_every must be positive")
        if (
            self.boundary_residual_component_target < 0
            or self.boundary_residual_min_pixels <= 0
        ):
            raise ValueError(
                "boundary residual component target must be nonnegative and "
                "minimum pixels positive"
            )
        for phase in self.phases:
            if phase.max_steps < 0 or phase.block_steps <= 0:
                raise ValueError(
                    f"phase {phase.name} must have a nonnegative step budget "
                    "and positive block size"
                )

    @property
    def phases(self) -> tuple[PhaseBudget, ...]:
        return (
            self.bootstrap,
            self.coverage,
            self.detail,
            self.boundary,
            self.redistribution,
            self.polish,
        )


@dataclass
class TopologyProposal:
    kind: str
    field: GaussianField
    touched: torch.Tensor
    count: int
    metadata: dict[str, Any]
    active_n: int | None = None


@dataclass
class _Trial:
    kind: str
    field: GaussianField
    optimizer_state: dict | None
    metrics: QualityMetrics
    count: int
    metadata: dict[str, Any]
    attempted_count: int
    recovery_steps: int
    active_n: int | None


ScheduleObserver = Callable[[GaussianField, dict[str, Any]], None]


# torch.quantile rejects inputs above 2**24 elements. A mask ROI rarely reaches that, but the
# full-frame arm evaluates every pixel, so large frames do. Below the limit this is exactly
# torch.quantile, so previously recorded metrics stay comparable; above it, the sorted-index
# fallback is the nearest-rank quantile.
_QUANTILE_MAX_ELEMENTS = 2 ** 24


def _safe_quantile(values: torch.Tensor, q: float) -> torch.Tensor:
    flat = values.reshape(-1)
    if flat.numel() <= _QUANTILE_MAX_ELEMENTS:
        return torch.quantile(flat, q)
    k = int(min(flat.numel(), max(1, math.ceil(float(q) * flat.numel()))))
    return torch.kthvalue(flat, k).values


def _logical_n(field: GaussianField, active_n: int | None) -> int:
    return int(field.n if active_n is None else active_n)


def _active_field(field: GaussianField, active_n: int | None) -> GaussianField:
    return field if active_n is None else field.prefix_view(active_n)


@torch.no_grad()
def _apply_constraint(
    field: GaussianField,
    cfg: FitConfig,
    constraint: _MaskConstraint,
    active_n: int | None,
    *,
    refresh: bool = False,
) -> None:
    if not cfg.mask_contain:
        return
    active = _active_field(field, active_n)
    constraint.apply(active, cfg, refresh=refresh)
    if active_n is not None and active.scale_max is not None:
        if field.scale_max is None:
            field.scale_max = torch.full(
                (field.n, 2),
                float("inf"),
                device=field.means.device,
                dtype=field.means.dtype,
            )
        field.scale_max[:active_n] = active.scale_max


def _visible_boundary(mask: torch.Tensor) -> torch.Tensor:
    outside = (~mask).to(dtype=torch.float32)[None, None]
    near = F.max_pool2d(outside, kernel_size=5, stride=1, padding=2)[0, 0] > 0
    return mask & near


def _production_mutual_nearest_pairs(
    means: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Use a CPU k-d tree when available, with the dependency-free exact scan as fallback."""

    try:
        import numpy as np
        from scipy.spatial import cKDTree
    except ImportError:
        return _mutual_nearest_pairs(means)
    values = means.detach().cpu().numpy()
    tree = cKDTree(values)
    try:
        distances, neighbours = tree.query(values, k=2, workers=-1)
    except TypeError:  # scipy before the workers argument
        distances, neighbours = tree.query(values, k=2)
    nearest = np.asarray(neighbours[:, 1], dtype=np.int64)
    index = np.arange(values.shape[0], dtype=np.int64)
    mutual = nearest[nearest] == index
    first_np = index[mutual & (index < nearest)]
    second_np = nearest[first_np]
    device = means.device
    first = torch.as_tensor(first_np, device=device, dtype=torch.long)
    second = torch.as_tensor(second_np, device=device, dtype=torch.long)
    distance = torch.as_tensor(
        np.asarray(distances[first_np, 1]),
        device=device,
        dtype=means.dtype,
    )
    return first, second, distance


@torch.no_grad()
def _expand_spatial_neighborhood(
    field: GaussianField,
    seed_rows: torch.Tensor,
    neighbor_count: int,
    active_n: int | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Return seed rows plus their direct Euclidean neighbours as one batched row mask."""

    active = _active_field(field, active_n)
    seeds = torch.unique(
        seed_rows.detach().to(device=active.means.device, dtype=torch.long)
    )
    seeds = seeds[(seeds >= 0) & (seeds < active.n)]
    trainable = torch.zeros(
        field.n, device=field.means.device, dtype=torch.bool
    )
    if seeds.numel() == 0:
        return trainable, {
            "seed_rows": 0,
            "neighbor_count": int(neighbor_count),
            "trainable_rows": 0,
        }
    trainable[seeds] = True
    requested = max(0, int(neighbor_count))
    if requested > 0 and active.n > 1:
        k = min(active.n, requested + 1)
        try:
            import numpy as np
            from scipy.spatial import cKDTree

            values = active.means.detach().cpu().numpy()
            query = values[seeds.detach().cpu().numpy()]
            tree = cKDTree(values)
            try:
                _, neighbours = tree.query(query, k=k, workers=-1)
            except TypeError:  # scipy before the workers argument
                _, neighbours = tree.query(query, k=k)
            neighbour_rows = torch.as_tensor(
                np.asarray(neighbours).reshape(int(seeds.numel()), -1),
                device=field.means.device,
                dtype=torch.long,
            ).reshape(-1)
            trainable[neighbour_rows] = True
            backend = "scipy_ckdtree"
        except ImportError:
            # Dependency-free fallback. Chunking bounds the temporary seed-by-field matrix.
            means = active.means.detach()
            for chunk in seeds.split(256):
                distances = torch.cdist(means[chunk], means)
                neighbours = torch.topk(
                    distances, k=k, largest=False, dim=1
                ).indices
                trainable[neighbours.reshape(-1)] = True
            backend = "torch_cdist"
    else:
        backend = "seeds_only"
    if active.background_mask is not None:
        trainable[:active.n] &= ~active.background_mask.to(
            device=trainable.device, dtype=torch.bool
        )
        # A topology event must never lose its own non-background touched rows.
        trainable[seeds] = True
    return trainable, {
        "seed_rows": int(seeds.numel()),
        "neighbor_count": requested,
        "trainable_rows": int(trainable.sum()),
        "neighbor_backend": backend,
    }


@torch.no_grad()
def _local_residual_row_mask(
    field: GaussianField,
    target: torch.Tensor,
    cfg: FitConfig,
    constraint: _MaskConstraint,
    schedule: SafeScheduleConfig,
    phase_name: str,
    active_n: int | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Select high-error owners and spatial neighbours while freezing the accepted remainder."""

    active = _active_field(field, active_n)
    H, W = target.shape[:2]
    render = _render(active, cfg, H, W, support_fade_alpha=1.0)
    score, components = _responsibility_error_density_scores(
        active, target, render, cfg, support_fade_alpha=1.0
    )
    score = score.detach().clone()
    valid = torch.isfinite(score)
    boundary_only = phase_name == schedule.boundary.name
    if boundary_only:
        means = active.means.detach()
        ix = means[:, 0].round().long().clamp(0, W - 1)
        iy = means[:, 1].round().long().clamp(0, H - 1)
        sdf = constraint.sdf_flat[iy * W + ix]
        reach = (
            active.effective_scales(cfg.aa_dilation).detach().amax(dim=1)
            * float(cfg.sigma_cutoff)
        )
        valid &= (
            (sdf > 0.0)
            & (
                sdf
                <= float(constraint.margin)
                + float(schedule.boundary_band)
                + reach
            )
        )
    if active.background_mask is not None:
        valid &= ~active.background_mask.to(device=valid.device, dtype=torch.bool)
    candidates = valid.nonzero(as_tuple=False).reshape(-1)
    if candidates.numel() == 0 and boundary_only:
        # A pathological thin mask may have no centre in the band even though supports overlap it.
        valid = torch.isfinite(score)
        if active.background_mask is not None:
            valid &= ~active.background_mask.to(device=valid.device, dtype=torch.bool)
        candidates = valid.nonzero(as_tuple=False).reshape(-1)
    k = min(int(schedule.local_seed_count), int(candidates.numel()))
    if k <= 0:
        raise RuntimeError("local residual refinement found no trainable Gaussian rows")
    seeds = candidates[torch.topk(score[candidates], k=k).indices]
    trainable, neighborhood = _expand_spatial_neighborhood(
        field, seeds, schedule.local_neighbor_count, active_n=active_n
    )
    return trainable, {
        **neighborhood,
        "selection": "responsibility_error_density + spatial direct neighbours",
        "phase_boundary_restricted": boundary_only,
        "seed_score_mean": float(score[seeds].mean()),
        "seed_score_max": float(score[seeds].max()),
        "seed_responsibility_mass_mean": float(
            components["mass"][seeds].mean()
        ),
    }


def _binary_component_summary(
    binary: torch.Tensor,
    min_pixels: int,
) -> dict[str, int]:
    """Count 4-connected components without making scipy a required dependency."""

    import numpy as np

    array = binary.detach().to(device="cpu", dtype=torch.bool).numpy()
    try:
        from scipy import ndimage

        labels, count = ndimage.label(
            array, structure=np.asarray([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
        )
        sizes = np.bincount(labels.reshape(-1))[1:count + 1]
        kept = sizes[sizes >= int(min_pixels)]
        return {
            "components": int(kept.size),
            "pixels": int(kept.sum()) if kept.size else 0,
            "largest": int(kept.max()) if kept.size else 0,
        }
    except ImportError:
        height, width = array.shape
        remaining = set(np.flatnonzero(array.reshape(-1)).tolist())
        sizes: list[int] = []
        while remaining:
            start = remaining.pop()
            stack = [start]
            size = 0
            while stack:
                index = stack.pop()
                size += 1
                y, x = divmod(index, width)
                neighbours = []
                if x > 0:
                    neighbours.append(index - 1)
                if x + 1 < width:
                    neighbours.append(index + 1)
                if y > 0:
                    neighbours.append(index - width)
                if y + 1 < height:
                    neighbours.append(index + width)
                for neighbour in neighbours:
                    if neighbour in remaining:
                        remaining.remove(neighbour)
                        stack.append(neighbour)
            if size >= int(min_pixels):
                sizes.append(size)
        return {
            "components": len(sizes),
            "pixels": int(sum(sizes)),
            "largest": int(max(sizes)) if sizes else 0,
        }


@torch.no_grad()
def _boundary_defect_summary(
    field: GaussianField,
    target: torch.Tensor,
    cfg: FitConfig,
    constraint: _MaskConstraint,
    schedule: SafeScheduleConfig,
    active_n: int | None = None,
) -> dict[str, Any]:
    """Measure persistent reachable-band undercoverage and high-error components."""

    active = _active_field(field, active_n)
    H, W = target.shape[:2]
    render = _render(active, cfg, H, W, support_fade_alpha=1.0)
    denominator = _raw_weight_map_field(
        active, cfg, H, W, support_fade_alpha=1.0
    ).reshape(H, W)
    pixel_mse = (render - target).square().mean(dim=2)
    band = torch.zeros(H * W, device=target.device, dtype=torch.bool)
    band[constraint.band_flat] = True
    band = band.reshape(H, W)
    residual = band & (
        pixel_mse > float(schedule.boundary_residual_mse_threshold)
    )
    holes = band & (denominator < float(schedule.coverage_tau))
    minimum = int(schedule.boundary_residual_min_pixels)
    residual_summary = _binary_component_summary(residual, minimum)
    hole_summary = _binary_component_summary(holes, minimum)
    band_pixels = max(1, int(band.sum()))
    return {
        "band_pixels": int(band.sum()),
        "residual_mse_threshold": float(
            schedule.boundary_residual_mse_threshold
        ),
        "residual_fraction": float(residual.sum()) / band_pixels,
        "residual_components": residual_summary["components"],
        "residual_component_pixels": residual_summary["pixels"],
        "largest_residual_component": residual_summary["largest"],
        "hole_fraction": float(holes.sum()) / band_pixels,
        "hole_components": hole_summary["components"],
        "hole_component_pixels": hole_summary["pixels"],
        "largest_hole_component": hole_summary["largest"],
    }


@torch.no_grad()
def evaluate_quality(
    field: GaussianField,
    target: torch.Tensor,
    mask: torch.Tensor,
    cfg: FitConfig,
    constraint: _MaskConstraint,
    coverage_tau: float,
    active_n: int | None = None,
) -> tuple[QualityMetrics, torch.Tensor]:
    """Render and compute the complete safe-commit vector on the full target."""

    active = _active_field(field, active_n)
    H, W = target.shape[:2]
    render = _render(active, cfg, H, W, support_fade_alpha=1.0)
    pixel_mse = (render - target).square().mean(dim=2)
    fg_values = pixel_mse[mask]
    boundary = _visible_boundary(mask)
    boundary_values = pixel_mse[boundary]
    if boundary_values.numel() == 0:
        boundary_values = fg_values
    if fg_values.numel() == 0:
        raise ValueError("safe schedule mask contains no foreground pixels")
    tail_count = max(1, int(math.ceil(0.01 * int(fg_values.numel()))))
    tail = torch.topk(fg_values.reshape(-1), k=tail_count).values
    p99 = _safe_quantile(fg_values, 0.99)
    denominator = _raw_weight_map_field(
        active, cfg, H, W, support_fade_alpha=1.0
    ).reshape(H, W)

    def hole_fraction(indices: torch.Tensor) -> float:
        if indices.numel() == 0:
            return 0.0
        return float(
            (denominator.reshape(-1)[indices] < float(coverage_tau))
            .to(dtype=torch.float32)
            .mean()
        )

    outside_render = render[~mask].abs()
    outside_den = denominator[~mask].abs()
    metrics = QualityMetrics(
        n_gaussians=int(active.n),
        foreground_mse=float(fg_values.mean()),
        boundary_mse=float(boundary_values.mean()),
        cvar99_mse=float(tail.mean()),
        p99_mse=float(p99),
        interior_hole_fraction=hole_fraction(constraint.interior_flat),
        boundary_hole_fraction=hole_fraction(constraint.band_flat),
        outside_max_abs=(
            0.0 if outside_render.numel() == 0 else float(outside_render.max())
        ),
        outside_coverage_max=(
            0.0 if outside_den.numel() == 0 else float(outside_den.max())
        ),
        finite=bool(
            torch.isfinite(render).all()
            and torch.isfinite(denominator).all()
        ),
    )
    return metrics, render


def safe_commit_decision(
    before: QualityMetrics,
    candidate: QualityMetrics,
    tolerance: CommitTolerances,
    hole_regression_budget: float = 0.0,
) -> tuple[bool, list[str]]:
    """Return a Pareto-safe decision and explicit rejection reasons.

    ``hole_regression_budget`` (ADR-0026) is an explicit **interior** coverage trade-off budget,
    unlike every field of ``tolerance``, which is numerical slack. It is the interior hole
    fraction a block may add and still commit. The default 0.0 is the historical strict-
    monotonicity gate. Boundary holes are never budgeted: the masked arm's boundary closure is
    the one part of the schedule with confirmed behaviour, so it keeps its exact gate.
    """

    reasons: list[str] = []
    if not math.isfinite(hole_regression_budget) or hole_regression_budget < 0.0:
        raise ValueError(
            "hole_regression_budget must be finite and nonnegative, "
            f"got {hole_regression_budget}"
        )
    if not candidate.finite:
        reasons.append("non_finite")

    def nonworse(name: str, old: float, new: float, relative: float) -> None:
        limit = old * (1.0 + relative) + tolerance.mse_absolute
        if new > limit:
            reasons.append(f"{name}_regressed")

    nonworse("foreground_mse", before.foreground_mse, candidate.foreground_mse,
             tolerance.mse_relative)
    nonworse("boundary_mse", before.boundary_mse, candidate.boundary_mse,
             tolerance.mse_relative)
    nonworse("cvar99_mse", before.cvar99_mse, candidate.cvar99_mse,
             tolerance.tail_relative)
    if tolerance.guard_p99:
        nonworse("p99_mse", before.p99_mse, candidate.p99_mse,
                 tolerance.tail_relative)
    if (
        candidate.interior_hole_fraction
        > before.interior_hole_fraction
        + tolerance.hole_absolute
        + float(hole_regression_budget)
    ):
        reasons.append("interior_holes_regressed")
    if (
        candidate.boundary_hole_fraction
        > before.boundary_hole_fraction + tolerance.hole_absolute
    ):
        reasons.append("boundary_holes_regressed")
    outside_limit = max(
        tolerance.outside_absolute,
        before.outside_max_abs + tolerance.outside_absolute,
    )
    if candidate.outside_max_abs > outside_limit:
        reasons.append("outside_render_regressed")
    coverage_limit = max(
        tolerance.outside_absolute,
        before.outside_coverage_max + tolerance.outside_absolute,
    )
    if candidate.outside_coverage_max > coverage_limit:
        reasons.append("outside_coverage_regressed")

    gains = [
        (before.foreground_mse - candidate.foreground_mse)
        / max(before.foreground_mse, 1e-12),
        (before.boundary_mse - candidate.boundary_mse)
        / max(before.boundary_mse, 1e-12),
        (before.cvar99_mse - candidate.cvar99_mse)
        / max(before.cvar99_mse, 1e-12),
        before.interior_hole_fraction - candidate.interior_hole_fraction,
        before.boundary_hole_fraction - candidate.boundary_hole_fraction,
    ]
    if tolerance.guard_p99:
        gains.append(
            (before.p99_mse - candidate.p99_mse)
            / max(before.p99_mse, 1e-12)
        )
    if max(gains) < tolerance.minimum_relative_gain:
        reasons.append("no_material_gain")
    return not reasons, reasons


def _clone_optimizer_state(state: dict | None) -> dict | None:
    return None if state is None else copy.deepcopy(state)


def adapt_optimizer_state(
    state: dict | None,
    old_n: int,
    new_n: int,
    touched: torch.Tensor,
) -> dict | None:
    """Resize per-row moments and zero every topology-touched row."""

    if state is None:
        return None
    out = copy.deepcopy(state)
    touched_cpu = touched.detach().to(device="cpu", dtype=torch.long)
    for parameter_state in out.get("state", {}).values():
        for key, value in list(parameter_state.items()):
            if not torch.is_tensor(value) or value.ndim == 0 or value.shape[0] != old_n:
                continue
            if new_n < old_n:
                value = value[:new_n].clone()
            elif new_n > old_n:
                pad = value.new_zeros((new_n - old_n,) + value.shape[1:])
                value = torch.cat([value, pad], dim=0)
            valid = touched_cpu[(touched_cpu >= 0) & (touched_cpu < new_n)]
            if valid.numel() > 0:
                value[valid.to(device=value.device)] = 0
            parameter_state[key] = value
    return out


def _reserve_geometric_capacity(
    field: GaussianField,
    optimizer_state: dict | None,
    active_n: int | None,
    schedule: SafeScheduleConfig,
    migrations: list[dict[str, int]],
) -> tuple[GaussianField, dict | None]:
    """Grow physical storage geometrically ahead of a geometric-policy topology auction.

    No-op unless ``storage_policy == "geometric"``. When the contiguous active prefix comes within
    one block's birth demand of the current physical capacity, migrate the field tensors (append
    parked rows) and the Adam moments (``adapt_optimizer_state``, appended rows zeroed) up to the
    next geometric step, capped at the logical ceiling. This keeps every in-place birth inside
    physical storage while committing only O(log N) migrations over the whole fit. The live prefix
    is preserved bit-for-bit, so an accepted trajectory is identical to the equivalent
    ``fixed_capacity`` run once physical capacity has caught up to the ceiling.
    """
    if schedule.storage_policy != "geometric" or active_n is None:
        return field, optimizer_state
    ceiling = min(int(schedule.capacity), schedule.resolved_base_active_limit())
    old_n = int(field.n)
    if old_n >= ceiling:
        return field, optimizer_state
    required = min(ceiling, int(active_n) + schedule.max_block_births())
    if required <= old_n:
        return field, optimizer_state
    new_n = geometric_capacity_schedule(
        old_n, required, growth_factor=schedule.growth_factor, max_capacity=ceiling
    )
    if new_n <= old_n:
        return field, optimizer_state
    field = grow_transactional_capacity(field, new_n)
    optimizer_state = adapt_optimizer_state(
        optimizer_state, old_n, new_n, torch.empty(0, dtype=torch.long)
    )
    migrations.append(
        {"old_capacity": old_n, "new_capacity": new_n, "active_n": int(active_n)}
    )
    return field, optimizer_state


def restore_frozen_optimizer_rows(
    candidate: dict | None,
    incoming: dict | None,
    trainable_rows: torch.Tensor,
) -> dict | None:
    """Undo local-recovery moment decay on protected rows."""

    if candidate is None or incoming is None:
        return candidate
    out = copy.deepcopy(candidate)
    frozen = (~trainable_rows.detach().to(dtype=torch.bool)).to(device="cpu")
    incoming_states = incoming.get("state", {})
    for key, candidate_state in out.get("state", {}).items():
        source_state = incoming_states.get(key)
        if source_state is None:
            continue
        for name, value in candidate_state.items():
            source = source_state.get(name)
            if (
                not torch.is_tensor(value)
                or not torch.is_tensor(source)
                or value.shape != source.shape
                or value.ndim == 0
                or value.shape[0] != frozen.numel()
            ):
                continue
            index = frozen.to(device=value.device)
            value[index] = source.to(device=value.device)[index]
    return out


def _zero_color_optimizer_state(state: dict | None) -> dict | None:
    if state is None:
        return None
    out = copy.deepcopy(state)
    groups = out.get("param_groups", [])
    if len(groups) <= 3 or not groups[3].get("params"):
        return out
    color_key = groups[3]["params"][0]
    for name, value in out.get("state", {}).get(color_key, {}).items():
        if torch.is_tensor(value) and value.ndim > 0:
            value.zero_()
    return out


def _phase_fit_config(
    base: FitConfig,
    phase: PhaseBudget,
    steps: int,
    *,
    lr_scale: float = 1.0,
    boundary_enabled: bool = True,
) -> FitConfig:
    lm, ls, lr, lc, lo = phase.learning_rates
    lowpass = int(phase.lowpass_downsample)
    return replace(
        base,
        iters=int(steps),
        lr_means=lm * lr_scale,
        lr_scales=ls * lr_scale,
        lr_rot=lr * lr_scale,
        lr_color=lc * lr_scale,
        lr_opacity=lo * lr_scale,
        pixel_loss="l2",
        ssim_weight=0.0,
        loss_weighting="mask",
        loss_target_downsample=lowpass,
        loss_target_full_frac=1.0 if lowpass > 1 else 0.0,
        mask_undercoverage_weight=(
            float(phase.boundary_floor_weight) if boundary_enabled else 0.0
        ),
        mask_interior_undercoverage_weight=float(phase.interior_floor_weight),
        coverage_match_weight=0.0,
        geometry_loss_weight=0.0,
        color_solve_every=None,
        color_solve_schedule="none",
        checkpoint_policy="terminal",
        lr_schedule="none",
        prune_every=None,
        split_every=None,
        relocate_every=None,
        mask_boundary_add_every=None,
        adaptive_count=False,
        triage_every=None,
        target_file_bytes=None,
        pool_capacity=None,
        max_gaussians=None,
        early_stop_patience=None,
        log_every=max(1, min(50, int(steps))),
    )


def _event_record(
    *,
    phase: str,
    event: str,
    accepted: bool,
    before: QualityMetrics,
    candidate: QualityMetrics | None,
    selected: QualityMetrics,
    attempted_steps: int,
    accepted_steps: int,
    reasons: Iterable[str] = (),
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "phase": phase,
        "event": event,
        "accepted": bool(accepted),
        "attempted_steps": int(attempted_steps),
        "accepted_steps": int(accepted_steps),
        "before": before.to_dict(),
        "candidate": None if candidate is None else candidate.to_dict(),
        "selected": selected.to_dict(),
        "reasons": list(reasons),
        "metadata": {} if metadata is None else metadata,
    }


def _safe_color_solve(
    field: GaussianField,
    optimizer_state: dict | None,
    metrics: QualityMetrics,
    target: torch.Tensor,
    mask: torch.Tensor,
    base_cfg: FitConfig,
    constraint: _MaskConstraint,
    schedule: SafeScheduleConfig,
    active_n: int | None = None,
) -> tuple[GaussianField, dict | None, QualityMetrics, dict[str, Any]]:
    trial = field.detached()
    with torch.no_grad():
        solve_stats = _solve_colors_normalized(
            _active_field(trial, active_n),
            target,
            base_cfg,
            target.shape[0],
            target.shape[1],
            support_fade_alpha=1.0,
        )
        _apply_constraint(
            trial, base_cfg, constraint, active_n, refresh=True
        )
    candidate, _ = evaluate_quality(
        trial,
        target,
        mask,
        base_cfg,
        constraint,
        schedule.coverage_tau,
        active_n=active_n,
    )
    accepted, reasons = safe_commit_decision(
        metrics, candidate, schedule.tolerances, schedule.hole_regression_budget
    )
    if accepted:
        selected_field = trial
        selected_state = _zero_color_optimizer_state(optimizer_state)
        selected_metrics = candidate
    else:
        selected_field = field
        selected_state = optimizer_state
        selected_metrics = metrics
    record = _event_record(
        phase="color_solve",
        event="fixed_geometry_least_squares",
        accepted=accepted,
        before=metrics,
        candidate=candidate,
        selected=selected_metrics,
        attempted_steps=0,
        accepted_steps=0,
        reasons=reasons,
        metadata=solve_stats,
    )
    return selected_field, selected_state, selected_metrics, record


def _safe_fit_block(
    field: GaussianField,
    optimizer_state: dict | None,
    metrics: QualityMetrics,
    target: torch.Tensor,
    mask_cpu,
    mask: torch.Tensor,
    cfg: FitConfig,
    constraint: _MaskConstraint,
    schedule: SafeScheduleConfig,
    *,
    trainable_rows: torch.Tensor | None = None,
    post_color_solve: bool = False,
    active_n: int | None = None,
    verbose: bool = False,
) -> tuple[GaussianField, dict | None, QualityMetrics, dict[str, Any], dict[str, Any]]:
    incoming_state = _clone_optimizer_state(optimizer_state)
    trial = field.detached()
    output = fit(
        trial,
        target,
        cfg,
        mask=mask_cpu,
        verbose=verbose,
        optimizer_state=_clone_optimizer_state(optimizer_state),
        trainable_row_mask=trainable_rows,
        return_optimizer_state=True,
        mask_constraint_override=constraint,
        state_checkpoint_every=(
            int(schedule.pareto_checkpoint_every)
            if schedule.pareto_safe_checkpoints
            else 0
        ),
        active_row_count=active_n,
    )
    terminal_steps = int(output["iterations_run"])
    if schedule.pareto_safe_checkpoints:
        raw_snapshots = output["state_checkpoints"]
        if not raw_snapshots:
            raise RuntimeError("fit returned no state-matched Pareto checkpoints")
    else:
        raw_snapshots = [{
            "iter": terminal_steps,
            "field": output["field"],
            "optimizer_state": output["optimizer_state"],
            "terminal": True,
        }]

    evaluated: list[dict[str, Any]] = []
    terminal_raw: dict[str, Any] | None = None
    for snapshot in raw_snapshots:
        candidate_field = snapshot["field"].detached()
        candidate_state = _clone_optimizer_state(snapshot["optimizer_state"])
        _apply_constraint(
            candidate_field, cfg, constraint, active_n, refresh=True
        )
        if trainable_rows is not None:
            candidate_state = restore_frozen_optimizer_rows(
                candidate_state, incoming_state, trainable_rows
            )
        candidate, _ = evaluate_quality(
            candidate_field,
            target,
            mask,
            cfg,
            constraint,
            schedule.coverage_tau,
            active_n=active_n,
        )
        accepted, reasons = safe_commit_decision(
            metrics, candidate, schedule.tolerances, schedule.hole_regression_budget
        )
        raw_evaluation = {
            "iter": int(snapshot["iter"]),
            "field": candidate_field,
            "optimizer_state": candidate_state,
            "metrics": candidate,
            "accepted": bool(accepted),
            "reasons": reasons,
            "color_solved": False,
            "solve_stats": None,
        }
        evaluated.append(raw_evaluation)
        if bool(snapshot.get("terminal")) or int(snapshot["iter"]) == terminal_steps:
            terminal_raw = raw_evaluation

        if post_color_solve:
            solved_field = candidate_field.detached()
            with torch.no_grad():
                solve_stats = _solve_colors_normalized(
                    _active_field(solved_field, active_n),
                    target,
                    cfg,
                    target.shape[0],
                    target.shape[1],
                    support_fade_alpha=1.0,
                )
                _apply_constraint(
                    solved_field, cfg, constraint, active_n, refresh=True
                )
            solved_metrics, _ = evaluate_quality(
                solved_field,
                target,
                mask,
                cfg,
                constraint,
                schedule.coverage_tau,
                active_n=active_n,
            )
            solved_accepted, solved_reasons = safe_commit_decision(
                metrics, solved_metrics, schedule.tolerances,
                schedule.hole_regression_budget,
            )
            evaluated.append({
                "iter": int(snapshot["iter"]),
                "field": solved_field,
                "optimizer_state": _zero_color_optimizer_state(candidate_state),
                "metrics": solved_metrics,
                "accepted": bool(solved_accepted),
                "reasons": solved_reasons,
                "color_solved": True,
                "solve_stats": solve_stats,
            })

    if terminal_raw is None:
        raise RuntimeError("fit checkpoint set did not contain the terminal state")
    safe_candidates = [candidate for candidate in evaluated if candidate["accepted"]]
    if safe_candidates:
        selected = max(
            safe_candidates,
            key=lambda candidate: _trial_gain(metrics, candidate["metrics"]),
        )
        selected_field = selected["field"]
        selected_state = selected["optimizer_state"]
        selected_metrics = selected["metrics"]
        accepted = True
        reasons = []
        candidate = selected_metrics
        selected_steps = int(selected["iter"])
    else:
        selected_field = field
        selected_state = optimizer_state
        selected_metrics = metrics
        accepted = False
        reasons = terminal_raw["reasons"]
        candidate = terminal_raw["metrics"]
        selected_steps = 0

    candidate_evaluations = [{
        "iter": int(evaluation["iter"]),
        "color_solved": bool(evaluation["color_solved"]),
        "accepted": bool(evaluation["accepted"]),
        "reasons": list(evaluation["reasons"]),
        "gain": _trial_gain(metrics, evaluation["metrics"]),
        "metrics": evaluation["metrics"].to_dict(),
        "solve_stats": evaluation["solve_stats"],
    } for evaluation in evaluated]
    record = _event_record(
        phase="fit_block",
        event="local_recovery" if trainable_rows is not None else "global_fit",
        accepted=accepted,
        before=metrics,
        candidate=candidate,
        selected=selected_metrics,
        attempted_steps=terminal_steps,
        accepted_steps=selected_steps,
        reasons=reasons,
        metadata={
            "fit_psnr": float(output["psnr"]),
            "trainable_rows": (
                _logical_n(field, active_n)
                if trainable_rows is None
                else int(trainable_rows.sum())
            ),
            "pareto_safe_checkpoints": bool(schedule.pareto_safe_checkpoints),
            "pareto_checkpoint_every": int(schedule.pareto_checkpoint_every),
            "checkpoint_candidates": candidate_evaluations,
            "selected_iter": selected_steps if accepted else None,
            "selected_from_pareto_checkpoint": bool(
                accepted and selected_steps < terminal_steps
            ),
            "event_color_solve_enabled": bool(post_color_solve),
            "event_color_solve_applied": bool(
                accepted and selected["color_solved"]
            ) if safe_candidates else False,
        },
    )
    return selected_field, selected_state, selected_metrics, record, output


def _safe_fit_block_with_backtracking(
    field: GaussianField,
    optimizer_state: dict | None,
    metrics: QualityMetrics,
    target: torch.Tensor,
    mask_cpu,
    mask: torch.Tensor,
    base_cfg: FitConfig,
    phase: PhaseBudget,
    steps: int,
    lr_scale: float,
    constraint: _MaskConstraint,
    schedule: SafeScheduleConfig,
    *,
    trainable_rows: torch.Tensor | None = None,
    event: str | None = None,
    selection_metadata: dict[str, Any] | None = None,
    active_n: int | None = None,
    verbose: bool = False,
) -> tuple[
    GaussianField,
    dict | None,
    QualityMetrics,
    dict[str, Any],
    FitConfig,
    int,
]:
    """Retry a rejected global or local block with a shorter trust horizon and lower LR."""

    requested_steps = max(1, int(steps))
    trial_steps = requested_steps
    trial_lr_scale = float(lr_scale)
    minimum_steps = max(1, min(16, requested_steps))
    attempts: list[dict[str, Any]] = []
    total_attempted = 0
    selected_cfg = _phase_fit_config(
        base_cfg,
        phase,
        trial_steps,
        lr_scale=trial_lr_scale,
        boundary_enabled=schedule.boundary_enabled,
    )
    final_record: dict[str, Any] | None = None
    for attempt_index in range(4):
        selected_cfg = _phase_fit_config(
            base_cfg,
            phase,
            trial_steps,
            lr_scale=trial_lr_scale,
            boundary_enabled=schedule.boundary_enabled,
        )
        (
            selected_field,
            selected_state,
            selected_metrics,
            record,
            output,
        ) = _safe_fit_block(
            field,
            optimizer_state,
            metrics,
            target,
            mask_cpu,
            mask,
            selected_cfg,
            constraint,
            schedule,
            trainable_rows=trainable_rows,
            active_n=active_n,
            verbose=verbose,
        )
        if event is not None:
            record["event"] = event
        if selection_metadata is not None:
            record["metadata"]["selection"] = copy.deepcopy(selection_metadata)
        ran = int(output["iterations_run"])
        total_attempted += ran
        attempts.append({
            "attempt": attempt_index + 1,
            "steps": ran,
            "accepted_steps": int(record["accepted_steps"]),
            "lr_scale": trial_lr_scale,
            "accepted": bool(record["accepted"]),
            "candidate": record["candidate"],
            "reasons": record["reasons"],
        })
        final_record = record
        if record["accepted"]:
            record["attempted_steps"] = total_attempted
            record["metadata"]["backtracking_attempts"] = attempts
            return (
                selected_field,
                selected_state,
                selected_metrics,
                record,
                selected_cfg,
                total_attempted,
            )
        if trial_steps <= minimum_steps:
            break
        next_steps = max(minimum_steps, trial_steps // 2)
        if next_steps == trial_steps:
            break
        trial_steps = next_steps
        trial_lr_scale *= 0.5

    if final_record is None:
        raise RuntimeError("safe fit backtracking executed no attempts")
    final_record["attempted_steps"] = total_attempted
    final_record["accepted_steps"] = 0
    final_record["metadata"]["backtracking_attempts"] = attempts
    return (
        field,
        optimizer_state,
        metrics,
        final_record,
        selected_cfg,
        total_attempted,
    )


def _local_structure(target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    gx, gy = _image_gradients(target)
    jxx = F.avg_pool2d(gx.square()[None, None], 5, stride=1, padding=2)[0, 0]
    jyy = F.avg_pool2d(gy.square()[None, None], 5, stride=1, padding=2)[0, 0]
    jxy = F.avg_pool2d((gx * gy)[None, None], 5, stride=1, padding=2)[0, 0]
    discriminant = torch.sqrt((jxx - jyy).square() + 4.0 * jxy.square())
    coherence = discriminant / (jxx + jyy + 1e-8)
    gradient_angle = 0.5 * torch.atan2(2.0 * jxy, jxx - jyy)
    tangent_angle = gradient_angle + math.pi * 0.5
    return tangent_angle, coherence.clamp(0.0, 1.0)


def _replicate_box_blur(value: torch.Tensor, kernel: int) -> torch.Tensor:
    padding = int(kernel) // 2
    return F.avg_pool2d(
        F.pad(value, (padding, padding, padding, padding), mode="replicate"),
        int(kernel),
        stride=1,
    )


@torch.no_grad()
def _detail_tail_score(
    field: GaussianField,
    target: torch.Tensor,
    render_img: torch.Tensor,
    cfg: FitConfig,
    constraint: _MaskConstraint,
    schedule: SafeScheduleConfig,
    *,
    denominator: torch.Tensor | None = None,
    coherence: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Score persistent fine-detail error away from holes and the visible boundary.

    The tail reserve is not a second coverage phase. Sites must already have raw normalized-
    compositor coverage, sit beyond the configured boundary band, and carry signed residual
    energy that survives both a 3x3 and a 7x7 high-pass test. Target high-frequency energy and
    local coherence rank structure-aligned errors without making either a hard texture gate.
    """

    H, W = target.shape[:2]
    if denominator is None:
        denominator = _raw_weight_map_field(
            field, cfg, H, W, support_fade_alpha=1.0
        ).reshape(H, W)
    if coherence is None:
        _, coherence = _local_structure(target)

    signed_error = (render_img - target).permute(2, 0, 1)[None]
    target_chw = target.permute(2, 0, 1)[None]
    error_high_3 = (signed_error - _replicate_box_blur(signed_error, 3)).abs()
    error_high_7 = (signed_error - _replicate_box_blur(signed_error, 7)).abs()
    target_high_3 = (target_chw - _replicate_box_blur(target_chw, 3)).abs()
    target_high_7 = (target_chw - _replicate_box_blur(target_chw, 7)).abs()
    persistent_error = torch.minimum(error_high_3, error_high_7).mean(dim=1)[0]
    persistent_target = torch.minimum(target_high_3, target_high_7).mean(dim=1)[0]

    deep_interior = (
        constraint.eroded_flat
        & (constraint.sdf_flat > float(schedule.boundary_band))
    ).reshape(H, W)
    well_covered = denominator >= float(schedule.coverage_tau)
    allowed = deep_interior & well_covered
    negative = torch.full_like(persistent_error, -float("inf"))
    eligible_count = int(allowed.sum())
    if eligible_count <= 0:
        return negative, allowed, persistent_error, {
            "eligible_pixels": 0,
            "deep_interior_pixels": int(deep_interior.sum()),
            "well_covered_pixels": int(well_covered.sum()),
            "persistent_error_p95": 0.0,
            "target_detail_p95": 0.0,
            "score_rule": (
                "persistent signed residual high-pass; covered interior only"
            ),
        }

    error_ref = _safe_quantile(persistent_error[allowed], 0.95).clamp_min(1e-8)
    target_ref = _safe_quantile(persistent_target[allowed], 0.95).clamp_min(1e-8)
    error_n = (persistent_error / error_ref).clamp(0.0, 4.0)
    target_n = (persistent_target / target_ref).clamp(0.0, 4.0)
    score = error_n * (1.0 + 0.35 * target_n + 0.25 * coherence)
    score = torch.where(allowed & (persistent_error > 0.0), score, negative)
    return score, allowed, persistent_error, {
        "eligible_pixels": int(torch.isfinite(score).sum()),
        "deep_interior_pixels": int(deep_interior.sum()),
        "well_covered_pixels": int(well_covered.sum()),
        "persistent_error_p95": float(error_ref),
        "target_detail_p95": float(target_ref),
        "score_rule": (
            "min(abs(residual-highpass3), abs(residual-highpass7)) "
            "x target-detail/coherence; covered interior only"
        ),
    }


def _error_region_scale(
    residual: torch.Tensor,
    y: torch.Tensor,
    x: torch.Tensor,
    global_base: float,
) -> torch.Tensor:
    """Estimate absolute footprint size from coherent residual support, in one batched pyramid."""

    center = residual[y, x].clamp_min(1e-8)
    radii = (1.0, 2.0, 4.0, 7.0)
    samples = []
    for radius in radii:
        kernel = int(2 * radius + 1)
        mean = F.avg_pool2d(
            residual[None, None], kernel, stride=1, padding=kernel // 2
        )[0, 0]
        samples.append(mean[y, x])
    support = torch.stack(samples, dim=1) >= 0.35 * center[:, None]
    rung = support.to(dtype=residual.dtype).sum(dim=1).clamp_min(1.0) - 1.0
    radius_values = residual.new_tensor(radii)
    chosen = radius_values[rung.long().clamp(0, len(radii) - 1)]
    local = 0.5 * chosen
    return local.clamp(
        min=_MIN_DENSIFY_SCALE,
        max=max(2.5 * float(global_base), _MIN_DENSIFY_SCALE),
    )


@torch.no_grad()
def _birth_components(
    field: GaussianField,
    target: torch.Tensor,
    render_img: torch.Tensor,
    cfg: FitConfig,
    constraint: _MaskConstraint,
    schedule: SafeScheduleConfig,
    count: int,
    mode: str,
) -> tuple[GaussianField | None, dict[str, Any]]:
    """Create a feature/error-region-aware birth cohort without mutating ``field``."""

    if count <= 0:
        return None, {"selected": 0}
    H, W = target.shape[:2]
    den = _raw_weight_map_field(
        field, cfg, H, W, support_fade_alpha=1.0
    ).reshape(H, W)
    residual = (render_img - target).abs().mean(dim=2)
    residual_ref = _safe_quantile(residual[constraint.inside], 0.95).clamp_min(1e-8)
    residual_n = (residual / residual_ref).clamp(0.0, 4.0)
    tangent, coherence = _local_structure(target)
    deficit = ((float(schedule.coverage_tau) - den) / float(schedule.coverage_tau)).clamp(0, 1)
    tail_metadata: dict[str, Any] = {}

    if mode == "boundary":
        allowed = (
            (constraint.sdf_flat > 0.0)
            & (constraint.sdf_flat <= float(schedule.boundary_band))
        ).reshape(H, W)
        score = 4.0 * deficit + residual_n + 0.25 * coherence
    else:
        allowed = constraint.eroded_flat.reshape(H, W)
        if mode == "coverage":
            holes = den < float(schedule.coverage_tau)
            score = 8.0 * holes.to(residual.dtype) + 3.0 * deficit \
                + 0.20 * residual_n + 0.10 * coherence
        elif mode == "detail":
            score = residual_n * (1.0 + 0.5 * coherence) + 2.0 * deficit
        elif mode == "detail_tail":
            score, allowed, residual, tail_metadata = _detail_tail_score(
                field,
                target,
                render_img,
                cfg,
                constraint,
                schedule,
                denominator=den,
                coherence=coherence,
            )
        else:
            raise ValueError(f"unknown birth mode {mode!r}")
    negative = torch.full_like(score, -float("inf"))
    score = torch.where(allowed, score, negative)
    radius = max(1, int(round(float(schedule.event_spacing_px) * 0.5)))
    peaks = F.max_pool2d(
        score[None, None], 2 * radius + 1, stride=1, padding=radius
    )[0, 0]
    peak_score = torch.where((score >= peaks) & allowed, score, negative).reshape(-1)
    finite = int(torch.isfinite(peak_score).sum())
    k = min(int(count), finite)
    if k <= 0:
        return None, {"selected": 0, "finite_peaks": finite}
    sites = torch.topk(peak_score, k=k).indices
    source_y = torch.div(sites, W, rounding_mode="floor")
    source_x = sites - source_y * W
    if mode == "boundary":
        projected = constraint.nearest_inside_flat[sites].clamp_min(0)
        y = torch.div(projected, W, rounding_mode="floor")
        x = projected - y * W
        nx = constraint.normal_x_flat[projected]
        ny = constraint.normal_y_flat[projected]
        reliable = nx.square() + ny.square() > 0.25
        angle = torch.where(
            reliable,
            torch.atan2(nx, -ny),
            tangent[source_y, source_x],
        )
    else:
        y, x = source_y, source_x
        angle = tangent[y, x]

    global_base = max(
        math.sqrt((H * W) / max(field.n + k, 1)) * float(cfg.split_scale),
        _MIN_DENSIFY_SCALE,
    )
    local_base = _error_region_scale(residual, source_y, source_x, global_base)
    local_coherence = coherence[source_y, source_x]
    ratio = 1.0 + (float(cfg.densify_max_axis_ratio) - 1.0) * (
        local_coherence ** float(cfg.densify_coherence_power)
    )
    along = local_base * torch.sqrt(ratio)
    across = local_base / torch.sqrt(ratio)
    if mode == "boundary":
        depth = (
            constraint.sdf_flat[y * W + x] - float(constraint.margin)
        ).clamp_min(1e-3)
        across_cap = (depth / float(constraint.sigma_cutoff)).clamp_min(
            _MIN_DENSIFY_SCALE
        )
        across = torch.minimum(across, across_cap)
        along = torch.maximum(
            along,
            torch.full_like(along, 0.75 * float(schedule.event_spacing_px)),
        )

    means = torch.stack([x.to(target.dtype), y.to(target.dtype)], dim=1)
    log_scales = torch.log(
        torch.stack([along, across], dim=1).clamp_min(_MIN_DENSIFY_SCALE)
    )
    colors = target[source_y, source_x].detach().clone()
    is_hole = den[source_y, source_x] < float(schedule.coverage_tau)
    opacity = torch.where(
        is_hole,
        torch.full_like(local_base, float(schedule.hole_opacity)),
        torch.full_like(local_base, float(schedule.covered_birth_opacity)),
    )
    opacities = _logit(opacity)
    components = GaussianField(
        means,
        log_scales,
        angle,
        colors,
        opacities,
        scale_max=torch.full_like(log_scales, float("inf")),
        filter_variance=(
            None
            if field.filter_variance is None
            else torch.zeros(k, device=target.device, dtype=target.dtype)
        ),
    )
    metadata = {
        "selected": k,
        "finite_peaks": finite,
        "hole_births": int(is_hole.sum()),
        "mean_base_scale": float(local_base.mean()),
        "min_base_scale": float(local_base.min()),
        "max_base_scale": float(local_base.max()),
        "mean_axis_ratio": float(ratio.mean()),
        "site_mode": mode,
        "covariance_rule": "local residual support scale + local structure tensor",
        **tail_metadata,
    }
    return components, metadata


@torch.no_grad()
def propose_birth(
    field: GaussianField,
    target: torch.Tensor,
    render_img: torch.Tensor,
    cfg: FitConfig,
    constraint: _MaskConstraint,
    schedule: SafeScheduleConfig,
    count: int,
    mode: str,
    active_n: int | None = None,
) -> TopologyProposal | None:
    logical_n = _logical_n(field, active_n)
    active = _active_field(field, active_n)
    # Prefix storage (fixed_capacity/geometric) clamps births to the physical row headroom
    # ``field.n``: for fixed_capacity that equals ``schedule.capacity`` (unchanged), for geometric
    # it is the current, still-growing physical capacity. Dynamic storage appends, so it clamps to
    # the configured ceiling as before.
    room_ceiling = int(field.n) if active_n is not None else int(schedule.capacity)
    room = max(0, room_ceiling - logical_n)
    components, metadata = _birth_components(
        active,
        target,
        render_img,
        cfg,
        constraint,
        schedule,
        min(int(count), room),
        mode,
    )
    if components is None or components.n == 0:
        return None
    if active_n is None:
        trial = field.detached().append(components)
        proposal_active_n = None
        touched = torch.arange(
            field.n, trial.n, device=field.means.device, dtype=torch.long
        )
    else:
        if schedule.storage_policy == "geometric":
            if int(field.n) > int(schedule.capacity):
                raise ValueError(
                    "geometric proposal storage cannot exceed schedule.capacity"
                )
        elif int(field.n) != int(schedule.capacity):
            raise ValueError(
                "fixed-capacity proposal storage must equal schedule.capacity"
            )
        trial = field.detached()
        proposal_active_n = logical_n + int(components.n)
        touched = torch.arange(
            logical_n,
            proposal_active_n,
            device=field.means.device,
            dtype=torch.long,
        )
        _replace_rows(trial, touched, components)
    _apply_constraint(
        trial, cfg, constraint, proposal_active_n, refresh=True
    )
    return TopologyProposal(
        kind=f"{mode}_birth",
        field=trial,
        touched=touched,
        count=int(components.n),
        metadata=metadata,
        active_n=proposal_active_n,
    )


@torch.no_grad()
def propose_split(
    field: GaussianField,
    target: torch.Tensor,
    render_img: torch.Tensor,
    cfg: FitConfig,
    constraint: _MaskConstraint,
    schedule: SafeScheduleConfig,
    count: int,
    active_n: int | None = None,
    detail_tail_only: bool = False,
) -> TopologyProposal | None:
    logical_n = _logical_n(field, active_n)
    active = _active_field(field, active_n)
    # See propose_birth: prefix storage clamps to the physical headroom ``field.n``
    # (== schedule.capacity for fixed_capacity, still growing for geometric).
    room_ceiling = int(field.n) if active_n is not None else int(schedule.capacity)
    room = max(0, room_ceiling - logical_n)
    k = min(int(count), room, logical_n)
    if k <= 0:
        return None
    scores, components = _responsibility_error_density_scores(
        active, target, render_img, cfg, support_fade_alpha=1.0
    )
    # A split is for a broad, error-owning primitive. Tiny owners are left to sampled births.
    footprint = active.scales().detach().prod(dim=1)
    score = scores * torch.sqrt(
        footprint / torch.quantile(footprint, 0.75).clamp_min(1e-8)
    )
    if active.background_mask is not None:
        score = score.masked_fill(active.background_mask, -float("inf"))
    tail_metadata: dict[str, Any] = {}
    if detail_tail_only:
        tail_score, _, _, tail_metadata = _detail_tail_score(
            active,
            target,
            render_img,
            cfg,
            constraint,
            schedule,
        )
        H, W = target.shape[:2]
        ix = active.means.detach()[:, 0].round().long().clamp(0, W - 1)
        iy = active.means.detach()[:, 1].round().long().clamp(0, H - 1)
        parent_tail_score = tail_score[iy, ix]
        eligible = torch.isfinite(parent_tail_score)
        score = torch.where(
            eligible,
            score * (1.0 + parent_tail_score.clamp(0.0, 4.0)),
            torch.full_like(score, -float("inf")),
        )
        k = min(k, int(torch.isfinite(score).sum()))
        if k <= 0:
            return None
    parents = torch.topk(score, k=k).indices

    means = active.means.detach().clone()
    log_scales = active.log_scales.detach().clone()
    rotations = active.rotations.detach().clone()
    colors = active.colors.detach().clone()
    color_grads = (
        None
        if active.color_grads is None
        else active.color_grads.detach().clone()
    )
    scales = active.scales().detach()[parents]
    theta = rotations[parents]
    use_x = scales[:, 0] >= scales[:, 1]
    c, s = torch.cos(theta), torch.sin(theta)
    direction = torch.where(
        use_x[:, None],
        torch.stack([c, s], dim=1),
        torch.stack([-s, c], dim=1),
    )
    axis_scale = torch.where(use_x, scales[:, 0], scales[:, 1])
    shrink = min(max(float(schedule.split_shrink), 1e-4), 1.0 - 1e-4)
    offset = direction * (
        axis_scale * math.sqrt(max(1.0 - shrink * shrink, 0.0))
    )[:, None]
    child_means = means[parents] + offset
    means[parents] -= offset
    means[:, 0].clamp_(0, target.shape[1] - 1)
    means[:, 1].clamp_(0, target.shape[0] - 1)
    child_means[:, 0].clamp_(0, target.shape[1] - 1)
    child_means[:, 1].clamp_(0, target.shape[0] - 1)
    parent_log = log_scales[parents].clone()
    child_log = log_scales[parents].clone()
    axis_index = torch.where(use_x, 0, 1)
    row = torch.arange(k, device=parents.device)
    parent_log[row, axis_index] += math.log(shrink)
    child_log[row, axis_index] += math.log(shrink)
    log_scales[parents] = parent_log

    if active.opacities is None:
        opacities = torch.full(
            (logical_n,),
            10.0,
            device=active.means.device,
            dtype=active.means.dtype,
        )
        parent_opacity = torch.ones(
            k, device=active.means.device, dtype=active.means.dtype
        )
    else:
        opacities = active.opacities.detach().clone()
        parent_opacity = torch.sigmoid(opacities[parents])
    # Integral(o * G) is proportional to opacity * sx * sy. Shrinking one axis by
    # ``shrink`` therefore needs o_child = o_parent / (2*shrink), not o_parent/2, for the two
    # equal children to preserve the parent's raw integrated coverage mass.
    child_opacity = (parent_opacity / (2.0 * shrink)).clamp(1e-4, 1.0 - 1e-4)
    opacities[parents] = _logit(child_opacity)
    child = GaussianField(
        child_means,
        child_log,
        theta.clone(),
        colors[parents].clone(),
        _logit(child_opacity),
        (
            None
            if active.scale_max is None
            else active.scale_max.detach()[parents].clone()
        ),
        None if color_grads is None else color_grads[parents].clone(),
        filter_variance=(
            None
            if active.filter_variance is None
            else active.filter_variance.detach()[parents].clone()
        ),
    )
    parent_field = GaussianField(
        means,
        log_scales,
        rotations,
        colors,
        opacities,
        None if active.scale_max is None else active.scale_max.detach().clone(),
        color_grads,
        (
            None
            if active.background_mask is None
            else active.background_mask.detach().clone()
        ),
        (
            None
            if active.filter_variance is None
            else active.filter_variance.detach().clone()
        ),
    )
    if active_n is None:
        trial = parent_field.append(child)
        proposal_active_n = None
        children = torch.arange(field.n, trial.n, device=parents.device)
    else:
        if schedule.storage_policy == "geometric":
            if int(field.n) > int(schedule.capacity):
                raise ValueError(
                    "geometric proposal storage cannot exceed schedule.capacity"
                )
        elif int(field.n) != int(schedule.capacity):
            raise ValueError(
                "fixed-capacity proposal storage must equal schedule.capacity"
            )
        trial = field.detached()
        trial.means[:logical_n] = parent_field.means
        trial.log_scales[:logical_n] = parent_field.log_scales
        trial.rotations[:logical_n] = parent_field.rotations
        trial.colors[:logical_n] = parent_field.colors
        if trial.opacities is None:
            trial.opacities = torch.full(
                (field.n,),
                10.0,
                device=field.means.device,
                dtype=field.means.dtype,
            )
        trial.opacities[:logical_n] = parent_field.opacities
        proposal_active_n = logical_n + k
        children = torch.arange(
            logical_n, proposal_active_n, device=parents.device
        )
        _replace_rows(trial, children, child)
    _apply_constraint(
        trial, cfg, constraint, proposal_active_n, refresh=True
    )
    touched = torch.unique(torch.cat([parents, children]))
    return TopologyProposal(
        kind=(
            "detail_tail_split"
            if detail_tail_only
            else "moment_preserving_split"
        ),
        field=trial,
        touched=touched,
        count=k,
        metadata={
            "parents": k,
            "score_mean": float(score[parents].mean()),
            "responsibility_mass_mean": float(components["mass"][parents].mean()),
            "split_shrink": shrink,
            "mass_rule": "opacity divided by 2*axis_shrink (integrated raw mass preserving)",
            "detail_tail_only": bool(detail_tail_only),
            **tail_metadata,
        },
        active_n=proposal_active_n,
    )


def _replace_rows(field: GaussianField, rows: torch.Tensor, values: GaussianField) -> None:
    field.means[rows] = values.means
    field.log_scales[rows] = values.log_scales
    field.rotations[rows] = values.rotations
    field.colors[rows] = values.colors
    if field.opacities is None:
        field.opacities = torch.full(
            (field.n,), 10.0, device=field.means.device, dtype=field.means.dtype
        )
    if values.opacities is None:
        field.opacities[rows] = 10.0
    else:
        field.opacities[rows] = values.opacities
    if field.color_grads is not None:
        if values.color_grads is None:
            field.color_grads[rows] = 0.0
        else:
            field.color_grads[rows] = values.color_grads
    if field.scale_max is not None:
        field.scale_max[rows] = float("inf")
    if field.filter_variance is not None:
        if values.filter_variance is None:
            field.filter_variance[rows] = 0.0
        else:
            field.filter_variance[rows] = values.filter_variance


@torch.no_grad()
def propose_prune_rebirth(
    field: GaussianField,
    target: torch.Tensor,
    render_img: torch.Tensor,
    cfg: FitConfig,
    constraint: _MaskConstraint,
    schedule: SafeScheduleConfig,
    count: int,
    birth_mode: str = "detail",
    active_n: int | None = None,
) -> TopologyProposal | None:
    active = _active_field(field, active_n)
    stats = attribution_pass(
        active, target, render_img, cfg, support_fade_alpha=1.0
    )
    eligible = stats["max_responsibility"] < float(
        schedule.redistribution_min_responsibility
    )
    if active.background_mask is not None:
        eligible &= ~active.background_mask
    candidates = eligible.nonzero(as_tuple=False).reshape(-1)
    k = min(int(count), int(candidates.numel()))
    if k <= 0:
        return None
    donor_score = stats["activity"][candidates] + stats["error"][candidates]
    donors = candidates[torch.argsort(donor_score)[:k]]
    births, metadata = _birth_components(
        active,
        target,
        render_img,
        cfg,
        constraint,
        schedule,
        k,
        birth_mode,
    )
    if births is None:
        return None
    k = min(k, int(births.n))
    donors = donors[:k]
    trial = field.detached()
    _replace_rows(trial, donors, births.subset(slice(0, k)))
    _apply_constraint(trial, cfg, constraint, active_n, refresh=True)
    return TopologyProposal(
        kind=(
            "prune_rebirth"
            if birth_mode == "detail"
            else f"prune_{birth_mode}_rebirth"
        ),
        field=trial,
        touched=donors,
        count=k,
        metadata={
            **metadata,
            "donor_max_responsibility_mean": float(
                stats["max_responsibility"][donors].mean()
            ),
            "donor_activity_mean": float(stats["activity"][donors].mean()),
            "count_neutral": True,
        },
        active_n=active_n,
    )


@torch.no_grad()
def propose_merge_rebirth(
    field: GaussianField,
    target: torch.Tensor,
    render_img: torch.Tensor,
    cfg: FitConfig,
    constraint: _MaskConstraint,
    schedule: SafeScheduleConfig,
    count: int,
    birth_mode: str = "detail",
    active_n: int | None = None,
) -> TopologyProposal | None:
    """Batch midpoint/envelope merge with the absorbed row directly reborn at high error."""

    active = _active_field(field, active_n)
    if active.n < 4 or count <= 0:
        return None
    stats = attribution_pass(
        active, target, render_img, cfg, support_fade_alpha=1.0
    )
    first, second, distance = _production_mutual_nearest_pairs(
        active.means.detach()
    )
    if first.numel() == 0:
        return None
    colors = active.colors.detach()
    color_delta = torch.linalg.norm(colors[first] - colors[second], dim=1)
    sorted_scales = torch.sort(
        active.scales().detach(), dim=1
    ).values.clamp_min(1e-6)
    scale_delta = torch.linalg.norm(
        torch.log(sorted_scales[first]) - torch.log(sorted_scales[second]), dim=1
    )
    angle_delta = (
        active.rotations.detach()[first] - active.rotations.detach()[second]
    )
    axial_delta = 0.5 * torch.abs(
        torch.atan2(torch.sin(2 * angle_delta), torch.cos(2 * angle_delta))
    )
    minor = 0.5 * (sorted_scales[first, 0] + sorted_scales[second, 0])
    allowed_distance = torch.clamp(
        2.5 * minor, min=1.25, max=4.0
    )
    midpoint = 0.5 * (
        active.means.detach()[first] + active.means.detach()[second]
    )
    mx = torch.round(midpoint[:, 0]).long().clamp(0, constraint.W - 1)
    my = torch.round(midpoint[:, 1]).long().clamp(0, constraint.H - 1)
    redundant = torch.maximum(
        stats["max_responsibility"][first],
        stats["max_responsibility"][second],
    ) < float(schedule.redistribution_min_responsibility)
    valid = (
        (distance <= allowed_distance)
        & (color_delta <= 0.18)
        & (scale_delta <= 1.0)
        & (axial_delta <= 0.65)
        & constraint.eroded_flat[my * constraint.W + mx]
        & redundant
    )
    valid_idx = valid.nonzero(as_tuple=False).reshape(-1)
    k = min(int(count), int(valid_idx.numel()))
    if k <= 0:
        return None
    pair_score = (
        distance / allowed_distance.clamp_min(1e-6)
        + 2.0 * color_delta
        + 0.25 * scale_delta
        + 0.15 * axial_delta
        + 0.1 * (stats["error"][first] + stats["error"][second])
    )
    selected = valid_idx[torch.argsort(pair_score[valid_idx])[:k]]
    keep = first[selected]
    rebirth = second[selected]
    births, birth_metadata = _birth_components(
        active,
        target,
        render_img,
        cfg,
        constraint,
        schedule,
        k,
        birth_mode,
    )
    if births is None:
        return None
    k = min(k, int(births.n))
    keep, rebirth = keep[:k], rebirth[:k]
    births = births.subset(slice(0, k))

    trial = field.detached()
    means = active.means.detach()
    if trial.opacities is None:
        trial.opacities = torch.full(
            (field.n,), 10.0, device=means.device, dtype=means.dtype
        )
    amplitude = torch.sigmoid(trial.opacities.detach())
    wa, wb = amplitude[keep], amplitude[rebirth]
    total = (wa + wb).clamp_min(1e-6)
    merged_mean = 0.5 * (means[keep] + means[rebirth])
    cov_a = _row_covariances(active, keep)
    cov_b = _row_covariances(active, rebirth)
    da = means[keep] - merged_mean
    db = means[rebirth] - merged_mean
    spatial_a = cov_a + da[:, :, None] * da[:, None, :]
    spatial_b = cov_b + db[:, :, None] * db[:, None, :]
    moment = 0.5 * (spatial_a + spatial_b)
    values, vectors = torch.linalg.eigh(moment)
    inv_sqrt = (
        vectors
        @ torch.diag_embed(values.clamp_min(1e-8).rsqrt())
        @ vectors.transpose(1, 2)
    )
    envelope = torch.maximum(
        torch.linalg.eigvalsh(inv_sqrt @ spatial_a @ inv_sqrt)[:, -1],
        torch.linalg.eigvalsh(inv_sqrt @ spatial_b @ inv_sqrt)[:, -1],
    ).clamp_min(1.0)
    merged_cov = moment * envelope[:, None, None] * (
        float(schedule.merge_envelope_inflation) ** 2
    )
    eigenvalues, eigenvectors = torch.linalg.eigh(merged_cov)
    eigenvalues = eigenvalues.clamp_min(_MIN_DENSIFY_SCALE ** 2)
    major = eigenvectors[:, :, 1]
    trial.means[keep] = merged_mean
    trial.log_scales[keep] = torch.log(
        torch.stack(
            [torch.sqrt(eigenvalues[:, 1]), torch.sqrt(eigenvalues[:, 0])], dim=1
        )
    )
    trial.rotations[keep] = torch.atan2(major[:, 1], major[:, 0])
    trial.colors[keep] = (
        wa[:, None] * active.colors.detach()[keep]
        + wb[:, None] * active.colors.detach()[rebirth]
    ) / total[:, None]
    combined = (1.0 - (1.0 - wa) * (1.0 - wb)).clamp(1e-4, 1.0 - 1e-4)
    trial.opacities[keep] = _logit(combined)
    if trial.color_grads is not None:
        gradients = active.color_grads.detach()
        trial.color_grads[keep] = (
            wa[:, None, None] * gradients[keep]
            + wb[:, None, None] * gradients[rebirth]
        ) / total[:, None, None]
    if trial.filter_variance is not None:
        variance = active.filter_variance.detach()
        trial.filter_variance[keep] = (
            wa * variance[keep] + wb * variance[rebirth]
        ) / total
    if trial.scale_max is not None:
        trial.scale_max[keep] = float("inf")
    # Atomic identity: each absorbed partner is now the corresponding high-error birth. It is
    # never exposed through a shared free/teleport list.
    _replace_rows(trial, rebirth, births)
    _apply_constraint(trial, cfg, constraint, active_n, refresh=True)

    certified = _row_covariances(_active_field(trial, active_n), keep)
    cvalues, cvectors = torch.linalg.eigh(certified)
    cinv = (
        cvectors
        @ torch.diag_embed(cvalues.clamp_min(1e-8).rsqrt())
        @ cvectors.transpose(1, 2)
    )
    ratio = torch.maximum(
        torch.linalg.eigvalsh(cinv @ spatial_a @ cinv)[:, -1],
        torch.linalg.eigvalsh(cinv @ spatial_b @ cinv)[:, -1],
    )
    good = ratio <= 1.0 + 1e-4
    rejected = int((~good).sum())
    if rejected:
        bad_rows = torch.cat([keep[~good], rebirth[~good]])
        original = active.subset(bad_rows)
        _replace_rows(trial, bad_rows, original)
        keep, rebirth = keep[good], rebirth[good]
        _apply_constraint(trial, cfg, constraint, active_n, refresh=True)
    if keep.numel() == 0:
        return None
    touched = torch.unique(torch.cat([keep, rebirth]))
    return TopologyProposal(
        kind=(
            "merge_rebirth"
            if birth_mode == "detail"
            else f"merge_{birth_mode}_rebirth"
        ),
        field=trial,
        touched=touched,
        count=int(keep.numel()),
        metadata={
            **birth_metadata,
            "merged": int(keep.numel()),
            "containment_rejected": rejected,
            "count_neutral": True,
            "merge_rule": "exact midpoint + covariance envelope + direct partner rebirth",
        },
        active_n=active_n,
    )


@torch.no_grad()
def propose_funded_split(
    field: GaussianField,
    target: torch.Tensor,
    render_img: torch.Tensor,
    cfg: FitConfig,
    constraint: _MaskConstraint,
    schedule: SafeScheduleConfig,
    count: int,
    active_n: int | None = None,
) -> TopologyProposal | None:
    """Count-neutral moment split funded by low-responsibility donor rows."""

    active = _active_field(field, active_n)
    stats = attribution_pass(
        active, target, render_img, cfg, support_fade_alpha=1.0
    )
    donors_ok = stats["max_responsibility"] < float(
        schedule.redistribution_min_responsibility
    )
    if active.background_mask is not None:
        donors_ok &= ~active.background_mask
    donors_all = donors_ok.nonzero(as_tuple=False).reshape(-1)
    scores, _ = _responsibility_error_density_scores(
        active, target, render_img, cfg, support_fade_alpha=1.0
    )
    scores = scores.masked_fill(donors_ok, -float("inf"))
    finite_parents = torch.isfinite(scores).nonzero(as_tuple=False).reshape(-1)
    k = min(int(count), int(donors_all.numel()), int(finite_parents.numel()))
    if k <= 0:
        return None
    donors = donors_all[
        torch.argsort(stats["activity"][donors_all] + stats["error"][donors_all])[:k]
    ]
    parents = torch.topk(scores, k=k).indices
    trial = field.detached()
    if trial.opacities is None:
        trial.opacities = torch.full(
            (field.n,), 10.0, device=field.means.device, dtype=field.means.dtype
        )
    scales = active.scales().detach()[parents]
    theta = active.rotations.detach()[parents]
    use_x = scales[:, 0] >= scales[:, 1]
    c, s = torch.cos(theta), torch.sin(theta)
    direction = torch.where(
        use_x[:, None],
        torch.stack([c, s], dim=1),
        torch.stack([-s, c], dim=1),
    )
    axis_scale = torch.where(use_x, scales[:, 0], scales[:, 1])
    shrink = min(max(float(schedule.split_shrink), 1e-4), 1.0 - 1e-4)
    offset = direction * (
        axis_scale * math.sqrt(max(1.0 - shrink * shrink, 0.0))
    )[:, None]
    original_means = active.means.detach()[parents]
    trial.means[parents] = original_means - offset
    trial.means[donors] = original_means + offset
    parent_log = active.log_scales.detach()[parents].clone()
    axis = torch.where(use_x, 0, 1)
    row = torch.arange(k, device=parents.device)
    parent_log[row, axis] += math.log(shrink)
    trial.log_scales[parents] = parent_log
    trial.log_scales[donors] = parent_log
    trial.rotations[donors] = theta
    trial.colors[donors] = active.colors.detach()[parents]
    parent_opacity = torch.sigmoid(active.opacities.detach()[parents]) \
        if active.opacities is not None else torch.full(
            (k,), 1.0, device=parents.device, dtype=active.means.dtype
        )
    child_opacity = (parent_opacity / (2.0 * shrink)).clamp(1e-4, 1.0 - 1e-4)
    trial.opacities[parents] = _logit(child_opacity)
    trial.opacities[donors] = _logit(child_opacity)
    if trial.color_grads is not None:
        trial.color_grads[donors] = active.color_grads.detach()[parents]
    if trial.filter_variance is not None:
        trial.filter_variance[donors] = (
            active.filter_variance.detach()[parents]
        )
    if trial.scale_max is not None:
        trial.scale_max[torch.cat([parents, donors])] = float("inf")
    _apply_constraint(trial, cfg, constraint, active_n, refresh=True)
    return TopologyProposal(
        kind="funded_split",
        field=trial,
        touched=torch.unique(torch.cat([parents, donors])),
        count=k,
        metadata={
            "parents": k,
            "donors": k,
            "count_neutral": True,
            "split_shrink": shrink,
            "mass_rule": "opacity divided by 2*axis_shrink",
            "donor_max_responsibility_mean": float(
                stats["max_responsibility"][donors].mean()
            ),
        },
        active_n=active_n,
    )


def _trial_gain(before: QualityMetrics, candidate: QualityMetrics) -> float:
    """Rank only already-safe candidates; no regression can be hidden by this scalar."""

    return (
        (before.foreground_mse - candidate.foreground_mse)
        / max(before.foreground_mse, 1e-12)
        + 0.25
        * (before.boundary_mse - candidate.boundary_mse)
        / max(before.boundary_mse, 1e-12)
        + 0.10
        * (before.cvar99_mse - candidate.cvar99_mse)
        / max(before.cvar99_mse, 1e-12)
        + 0.05
        * (before.interior_hole_fraction - candidate.interior_hole_fraction)
        + 0.05
        * (before.boundary_hole_fraction - candidate.boundary_hole_fraction)
    )


def _topology_auction(
    field: GaussianField,
    optimizer_state: dict | None,
    metrics: QualityMetrics,
    target: torch.Tensor,
    mask_cpu,
    mask: torch.Tensor,
    fit_cfg: FitConfig,
    constraint: _MaskConstraint,
    schedule: SafeScheduleConfig,
    factories: list[tuple[str, int, Callable[[int, torch.Tensor], TopologyProposal | None]]],
    *,
    active_n: int | None = None,
    expand_neighborhood: bool | None = None,
    minimum_gain_per_row: float | None = None,
    verbose: bool,
) -> tuple[
    GaussianField,
    dict | None,
    QualityMetrics,
    dict[str, Any],
    int,
    int,
    int | None,
]:
    """Evaluate operator batches transactionally and commit the best Pareto-safe trial."""

    safe_trials: list[_Trial] = []
    attempts: list[dict[str, Any]] = []
    attempted_steps = 0
    # Every operator competes against the same accepted state. Render it once; trials never
    # mutate ``field`` before the winner is selected.
    _, current_render = evaluate_quality(
        field,
        target,
        mask,
        fit_cfg,
        constraint,
        schedule.coverage_tau,
        active_n=active_n,
    )
    for kind, requested, factory in factories:
        count = max(0, int(requested))
        while count >= int(schedule.event_min_count):
            proposal = factory(count, current_render)
            if proposal is None or proposal.count <= 0:
                attempts.append({
                    "kind": kind,
                    "requested": count,
                    "status": "no_proposal",
                })
                break
            adapted_state = adapt_optimizer_state(
                optimizer_state, field.n, proposal.field.n, proposal.touched
            )
            use_neighborhood = (
                schedule.refinement_policy == "local_neighborhood"
                if expand_neighborhood is None
                else bool(expand_neighborhood)
            )
            if use_neighborhood:
                trainable, neighborhood = _expand_spatial_neighborhood(
                    proposal.field,
                    proposal.touched,
                    schedule.topology_neighbor_count,
                    active_n=proposal.active_n,
                )
            else:
                trainable = torch.zeros(
                    proposal.field.n, device=target.device, dtype=torch.bool
                )
                trainable[proposal.touched] = True
                neighborhood = {
                    "seed_rows": int(proposal.touched.numel()),
                    "neighbor_count": 0,
                    "trainable_rows": int(proposal.touched.numel()),
                    "neighbor_backend": "disabled",
                }
            recovery_cfg = replace(
                fit_cfg,
                iters=int(schedule.recovery_steps),
                log_every=max(1, min(40, int(schedule.recovery_steps))),
            )
            recovered, recovered_state, recovered_metrics, record, output = _safe_fit_block(
                proposal.field,
                adapted_state,
                metrics,
                target,
                mask_cpu,
                mask,
                recovery_cfg,
                constraint,
                schedule,
                trainable_rows=trainable,
                post_color_solve=bool(schedule.event_color_solve),
                active_n=proposal.active_n,
                verbose=False,
            )
            attempted_steps += int(output["iterations_run"])
            trial_metadata = {
                **proposal.metadata,
                "recovery_neighborhood": neighborhood,
                "recovery": copy.deepcopy(record["metadata"]),
            }
            gain = (
                _trial_gain(metrics, recovered_metrics)
                if record["accepted"]
                else None
            )
            gain_per_row = (
                None
                if gain is None
                else float(gain) / max(int(proposal.count), 1)
            )
            effective = bool(
                record["accepted"]
                and (
                    minimum_gain_per_row is None
                    or gain_per_row >= float(minimum_gain_per_row)
                )
            )
            reasons = list(record["reasons"])
            if record["accepted"] and not effective:
                reasons.append("gain_per_row_below_minimum")
            attempts.append({
                "kind": kind,
                "requested": count,
                "proposed": proposal.count,
                "accepted": effective,
                "safe_gate_accepted": bool(record["accepted"]),
                "gain": gain,
                "gain_per_row": gain_per_row,
                "minimum_gain_per_row": minimum_gain_per_row,
                "reasons": reasons,
                "candidate": record["candidate"],
                "metadata": trial_metadata,
            })
            if effective:
                safe_trials.append(_Trial(
                    kind=proposal.kind,
                    field=recovered,
                    optimizer_state=recovered_state,
                    metrics=recovered_metrics,
                    count=proposal.count,
                    metadata=trial_metadata,
                    attempted_count=count,
                    recovery_steps=int(record["accepted_steps"]),
                    active_n=proposal.active_n,
                ))
                break
            next_count = count // 2
            if next_count == count:
                break
            count = next_count

    if not safe_trials:
        record = _event_record(
            phase="topology",
            event="auction",
            accepted=False,
            before=metrics,
            candidate=None,
            selected=metrics,
            attempted_steps=attempted_steps,
            accepted_steps=0,
            reasons=["no_safe_topology_trial"],
            metadata={"attempts": attempts},
        )
        return (
            field,
            optimizer_state,
            metrics,
            record,
            attempted_steps,
            0,
            active_n,
        )

    winner = max(safe_trials, key=lambda trial: _trial_gain(metrics, trial.metrics))
    winner_gain = _trial_gain(metrics, winner.metrics)
    record = _event_record(
        phase="topology",
        event=winner.kind,
        accepted=True,
        before=metrics,
        candidate=winner.metrics,
        selected=winner.metrics,
        attempted_steps=attempted_steps,
        accepted_steps=winner.recovery_steps,
        metadata={
            "winner_count": winner.count,
            "winner_requested_count": winner.attempted_count,
            "winner_gain": winner_gain,
            "winner_gain_per_row": winner_gain / max(winner.count, 1),
            "minimum_gain_per_row": minimum_gain_per_row,
            "winner_metadata": winner.metadata,
            "attempts": attempts,
        },
    )
    if verbose:
        print(
            f"  safe topology {winner.kind} x{winner.count}: "
            f"{metrics.foreground_psnr_db:.3f} -> "
            f"{winner.metrics.foreground_psnr_db:.3f} dB",
            flush=True,
        )
    return (
        winner.field,
        winner.optimizer_state,
        winner.metrics,
        record,
        attempted_steps,
        winner.recovery_steps,
        winner.active_n,
    )


def run_safe_schedule(
    initial_field: GaussianField,
    target: torch.Tensor,
    mask=None,
    base_cfg: FitConfig | None = None,
    schedule: SafeScheduleConfig | None = None,
    *,
    observer: ScheduleObserver | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Run the complete safe-commit schedule and return the best accepted field.

    ``mask=None`` selects the full-frame arm: every pixel is foreground, so the containment
    machinery degenerates rather than being bypassed by a separate code path. ``signed_distance``
    clips an empty complement to the image diagonal, so the projection interior is the whole frame
    and the per-row caps are inert. Boundary-specific initialization, loss, metrics, and proposals
    are disabled; the same closure slot and step budget use general coverage/detail proposals.
    Every other phase, the commit gate, and the metric vector are the masked arm's, unchanged
    (ADR-0025/0027).
    """

    schedule = SafeScheduleConfig() if schedule is None else schedule
    base_cfg = FitConfig() if base_cfg is None else base_cfg
    schedule.validate(initial_field.n)
    base_active_limit = schedule.resolved_base_active_limit()
    H_target, W_target = int(target.shape[0]), int(target.shape[1])
    full_frame = mask is None
    if full_frame:
        if schedule.boundary_enabled:
            schedule = replace(
                schedule,
                boundary_enabled=False,
                boundary=replace(
                    schedule.boundary, name="general_closure"
                ),
            )
        import numpy as _np

        mask_cpu = _np.ones((H_target, W_target), dtype=bool)
    else:
        mask_cpu = (
            mask.detach().cpu().numpy()
            if torch.is_tensor(mask)
            else mask
        )
    mask_tensor = torch.as_tensor(
        mask_cpu, device=target.device, dtype=torch.bool
    )
    cfg = replace(
        base_cfg,
        pixel_loss="l2",
        ssim_weight=0.0,
        loss_weighting="mask",
        mask_contain=schedule.boundary_enabled and not full_frame,
        mask_cap_mode="anisotropic",
        mask_undercoverage_band=(
            float(schedule.boundary_band)
            if schedule.boundary_enabled
            else 0.0
        ),
        mask_undercoverage_tau=float(schedule.coverage_tau),
        support_fade=True,
        coverage_match_weight=0.0,
        checkpoint_policy="terminal",
        triage_every=None,
        target_file_bytes=None,
        pool_capacity=None,
        split_every=None,
        relocate_every=None,
        prune_every=None,
        mask_boundary_add_every=None,
        adaptive_count=False,
        compute_lpips=False,
    )
    constraint = _MaskConstraint.from_mask(
        mask_cpu,
        target.device,
        target.dtype,
        cfg.sigma_cutoff,
        cfg.mask_margin,
        aa_dilation=cfg.aa_dilation,
        cap_mode="anisotropic",
        undercoverage_band=float(schedule.boundary_band),
    )
    field = initial_field.detached()
    # Keep the optimizer's parameter-group topology stable across the entire schedule. Births
    # need trainable opacity, so a legacy opaque field is made explicitly opaque before the first
    # moment state is created rather than adding a fifth optimizer group mid-run.
    if field.opacities is None:
        field.opacities = torch.full(
            (field.n,), 10.0, device=field.means.device, dtype=field.means.dtype
        )
    initial_n = int(field.n)
    active_n: int | None = None
    geometric_migrations: list[dict[str, int]] = []
    if schedule.storage_policy == "fixed_capacity":
        field, pool_state = prepare_transactional_pool(
            field, schedule.capacity
        )
        active_n = pool_state.active_n
    elif schedule.storage_policy == "geometric":
        # Start at a small physical capacity and grow geometrically toward ``capacity`` on demand.
        field, pool_state = prepare_transactional_pool(
            field, schedule.resolved_initial_capacity(initial_n)
        )
        active_n = pool_state.active_n
    initial_physical_n = int(field.n)
    _apply_constraint(field, cfg, constraint, active_n, refresh=True)
    metrics, _ = evaluate_quality(
        field,
        target,
        mask_tensor,
        cfg,
        constraint,
        schedule.coverage_tau,
        active_n=active_n,
    )
    optimizer_state: dict | None = None
    history: list[dict[str, Any]] = []
    attempted_steps = 0
    accepted_steps = 0
    started = time.perf_counter()

    def emit(record: dict[str, Any]) -> None:
        nonlocal attempted_steps, accepted_steps
        record = copy.deepcopy(record)
        attempted_steps += int(record.get("attempted_steps", 0))
        accepted_steps += int(record.get("accepted_steps", 0))
        record["global_attempted_steps"] = attempted_steps
        record["global_accepted_steps"] = accepted_steps
        record["elapsed_seconds"] = time.perf_counter() - started
        history.append(record)
        if observer is not None:
            observer(_active_field(field, active_n), record)

    emit(_event_record(
        phase="initialization",
        event="initial_safe_state",
        accepted=True,
        before=metrics,
        candidate=metrics,
        selected=metrics,
        attempted_steps=0,
        accepted_steps=0,
    ))

    # Remove color-fixable error before allocating any new topology.
    field, optimizer_state, metrics, color_record = _safe_color_solve(
        field,
        optimizer_state,
        metrics,
        target,
        mask_tensor,
        cfg,
        constraint,
        schedule,
        active_n=active_n,
    )
    emit(color_record)

    local_phase_order = {
        "coverage": 0,
        "detail": 1,
        "boundary": 2,
        "redistribution": 3,
        "polish": 4,
    }
    phase_alias = {
        schedule.coverage.name: "coverage",
        schedule.detail.name: "detail",
        schedule.boundary.name: "boundary",
        schedule.redistribution.name: "redistribution",
        schedule.polish.name: "polish",
    }

    def phase_uses_local_refinement(phase: PhaseBudget) -> bool:
        alias = phase_alias.get(phase.name)
        return bool(
            schedule.refinement_policy == "local_neighborhood"
            and alias is not None
            and local_phase_order[alias]
            >= local_phase_order[schedule.local_start_phase]
        )

    def run_phase(
        phase: PhaseBudget,
        topology: Callable[
            [FitConfig],
            list[tuple[str, int, Callable[[int, torch.Tensor], TopologyProposal | None]]],
        ] | None,
        exit_condition: Callable[[], bool],
        diagnostics: Callable[[], dict[str, Any]] | None = None,
    ) -> None:
        nonlocal field, optimizer_state, metrics, active_n
        phase_steps = 0
        stale = 0
        block_index = 0
        termination_reason = "step_budget"
        while phase_steps < int(phase.max_steps):
            if exit_condition():
                termination_reason = "metric_target"
                break
            block_index += 1
            steps = min(int(phase.block_steps), int(phase.max_steps) - phase_steps)
            lr_scale = 1.0
            if phase.name == schedule.polish.name:
                lr_scale = max(0.1, 1.0 - phase_steps / max(phase.max_steps, 1))
            trainable_rows = None
            selection_metadata = None
            fit_event = None
            local_refinement = phase_uses_local_refinement(phase)
            if local_refinement:
                selection_cfg = _phase_fit_config(
                    cfg,
                    phase,
                    steps,
                    lr_scale=lr_scale,
                    boundary_enabled=schedule.boundary_enabled,
                )
                trainable_rows, selection_metadata = _local_residual_row_mask(
                    field,
                    target,
                    selection_cfg,
                    constraint,
                    schedule,
                    phase.name,
                    active_n=active_n,
                )
                fit_event = "local_residual_fit"
            (
                field,
                optimizer_state,
                metrics,
                block_record,
                phase_cfg,
                block_attempted,
            ) = _safe_fit_block_with_backtracking(
                field,
                optimizer_state,
                metrics,
                target,
                mask_cpu,
                mask_tensor,
                cfg,
                phase,
                steps,
                lr_scale,
                constraint,
                schedule,
                trainable_rows=trainable_rows,
                event=fit_event,
                selection_metadata=selection_metadata,
                active_n=active_n,
                verbose=False,
            )
            block_record["phase"] = phase.name
            block_record["metadata"].update({
                "block": block_index,
                "phase_step_start": phase_steps,
                "lr_scale": lr_scale,
            })
            phase_steps += int(block_attempted)
            emit(block_record)
            progressed = bool(block_record["accepted"])

            if topology is not None and not exit_condition():
                factories = topology(phase_cfg)
                if factories:
                    field, optimizer_state = _reserve_geometric_capacity(
                        field, optimizer_state, active_n, schedule, geometric_migrations
                    )
                    (
                        field,
                        optimizer_state,
                        metrics,
                        topology_record,
                        _auction_attempted,
                        _auction_accepted,
                        active_n,
                    ) = _topology_auction(
                        field,
                        optimizer_state,
                        metrics,
                        target,
                        mask_cpu,
                        mask_tensor,
                        phase_cfg,
                        constraint,
                        schedule,
                        factories,
                        active_n=active_n,
                        expand_neighborhood=local_refinement,
                        verbose=verbose,
                    )
                    topology_record["phase"] = phase.name
                    emit(topology_record)
                    progressed = progressed or bool(topology_record["accepted"])

            if progressed:
                stale = 0
            else:
                stale += 1
            # Proposal generation, batch bisection, and fit backtracking are deterministic for
            # an unchanged accepted state. Repeating the identical rejected state would only
            # spend time; stale_patience remains recorded for compatibility but one completely
            # fruitless composite block is already a fixed point.
            if not progressed or stale >= int(schedule.stale_patience):
                termination_reason = "deterministic_fixed_point"
                break
            if verbose:
                print(
                    f"[{phase.name}] block={block_index} "
                    f"n={_logical_n(field, active_n)} "
                    f"fg={metrics.foreground_psnr_db:.3f}dB "
                    f"boundary={metrics.boundary_psnr_db:.3f}dB "
                    f"holes={100*metrics.interior_hole_fraction:.2f}%/"
                    f"{100*metrics.boundary_hole_fraction:.2f}%",
                    flush=True,
                )

        phase_metadata = {
            "phase_steps": phase_steps,
            "stale_blocks": stale,
            "target_gaussians": phase.target_gaussians,
            "termination_reason": termination_reason,
            "refinement_policy": (
                "local_neighborhood"
                if phase_uses_local_refinement(phase)
                else "global"
            ),
        }
        if diagnostics is not None:
            phase_metadata["diagnostics"] = diagnostics()
        emit(_event_record(
            phase=phase.name,
            event="phase_end",
            accepted=True,
            before=metrics,
            candidate=metrics,
            selected=metrics,
            attempted_steps=0,
            accepted_steps=0,
            metadata=phase_metadata,
        ))

    run_phase(
        schedule.bootstrap,
        topology=None,
        exit_condition=lambda: False,
    )

    def coverage_factories(phase_cfg: FitConfig):
        room = max(
            0,
            schedule.coverage_target_gaussians
            - _logical_n(field, active_n),
        )
        count = min(schedule.coverage_birth_count, room)
        return [(
            "coverage_birth",
            count,
            lambda k, image: propose_birth(
                field,
                target,
                image,
                phase_cfg,
                constraint,
                schedule,
                k,
                "coverage",
                active_n=active_n,
            ),
        )] if count > 0 else []

    run_phase(
        schedule.coverage,
        topology=coverage_factories,
        exit_condition=lambda: (
            _logical_n(field, active_n)
            >= schedule.coverage_target_gaussians
            or (
                metrics.interior_hole_fraction <= schedule.interior_hole_target
                and (
                    not schedule.boundary_enabled
                    or metrics.boundary_hole_fraction
                    <= schedule.boundary_hole_target
                )
            )
        ),
    )

    def detail_factories(phase_cfg: FitConfig):
        room = max(
            0,
            schedule.detail_target_gaussians
            - _logical_n(field, active_n),
        )
        if room <= 0:
            return []
        return [
            (
                "detail_birth",
                min(schedule.detail_birth_count, room),
                lambda k, image: propose_birth(
                    field,
                    target,
                    image,
                    phase_cfg,
                    constraint,
                    schedule,
                    k,
                    "detail",
                    active_n=active_n,
                ),
            ),
            (
                "moment_preserving_split",
                min(schedule.detail_split_count, room),
                lambda k, image: propose_split(
                    field,
                    target,
                    image,
                    phase_cfg,
                    constraint,
                    schedule,
                    k,
                    active_n=active_n,
                ),
            ),
            (
                "coverage_birth",
                min(schedule.detail_birth_count, room),
                lambda k, image: propose_birth(
                    field,
                    target,
                    image,
                    phase_cfg,
                    constraint,
                    schedule,
                    k,
                    "coverage",
                    active_n=active_n,
                ),
            ),
        ]

    run_phase(
        schedule.detail,
        topology=detail_factories,
        exit_condition=lambda: (
            _logical_n(field, active_n)
            >= schedule.detail_target_gaussians
        ),
    )

    def boundary_factories(phase_cfg: FitConfig):
        room = max(
            0, base_active_limit - _logical_n(field, active_n)
        )
        count = min(schedule.boundary_birth_count, room)
        if count > 0 and not schedule.boundary_enabled:
            return [
                (
                    "coverage_birth",
                    min(schedule.detail_birth_count, room),
                    lambda k, image: propose_birth(
                        field,
                        target,
                        image,
                        phase_cfg,
                        constraint,
                        schedule,
                        k,
                        "coverage",
                        active_n=active_n,
                    ),
                ),
                (
                    "detail_birth",
                    min(schedule.detail_birth_count, room),
                    lambda k, image: propose_birth(
                        field,
                        target,
                        image,
                        phase_cfg,
                        constraint,
                        schedule,
                        k,
                        "detail",
                        active_n=active_n,
                    ),
                ),
            ]
        if count > 0:
            factories = [
                (
                    "boundary_birth",
                    count,
                    lambda k, image: propose_birth(
                        field,
                        target,
                        image,
                        phase_cfg,
                        constraint,
                        schedule,
                        k,
                        "boundary",
                        active_n=active_n,
                    ),
                ),
            ]
            factories.append(
                (
                    "detail_birth",
                    min(schedule.detail_birth_count, room),
                    lambda k, image: propose_birth(
                        field,
                        target,
                        image,
                        phase_cfg,
                        constraint,
                        schedule,
                        k,
                        "detail",
                        active_n=active_n,
                    ),
                )
            )
            return factories
        if not schedule.boundary_recycle_at_capacity:
            return []
        recycle_count = int(schedule.redistribution_count)
        return [
            (
                "merge_boundary_rebirth",
                recycle_count,
                lambda k, image: propose_merge_rebirth(
                    field,
                    target,
                    image,
                    phase_cfg,
                    constraint,
                    schedule,
                    k,
                    birth_mode="boundary",
                    active_n=active_n,
                ),
            ),
            (
                "prune_boundary_rebirth",
                recycle_count,
                lambda k, image: propose_prune_rebirth(
                    field,
                    target,
                    image,
                    phase_cfg,
                    constraint,
                    schedule,
                    k,
                    birth_mode="boundary",
                    active_n=active_n,
                ),
            ),
        ]

    def boundary_diagnostics() -> dict[str, Any]:
        return _boundary_defect_summary(
            field,
            target,
            cfg,
            constraint,
            schedule,
            active_n=active_n,
        )

    def boundary_target_reached() -> bool:
        if not schedule.boundary_enabled:
            return _logical_n(field, active_n) >= base_active_limit
        if (
            not schedule.boundary_recycle_at_capacity
            and _logical_n(field, active_n) >= base_active_limit
        ):
            return True
        if metrics.boundary_hole_fraction > schedule.boundary_hole_target:
            return False
        diagnostics = boundary_diagnostics()
        return (
            int(diagnostics["residual_components"])
            <= int(schedule.boundary_residual_component_target)
        )

    run_phase(
        schedule.boundary,
        topology=boundary_factories,
        exit_condition=boundary_target_reached,
        diagnostics=(
            boundary_diagnostics if schedule.boundary_enabled else None
        ),
    )

    def redistribution_factories(phase_cfg: FitConfig):
        room = max(
            0, base_active_limit - _logical_n(field, active_n)
        )
        if room > 0:
            factories = [
                (
                    "coverage_birth",
                    min(schedule.detail_birth_count, room),
                    lambda k, image: propose_birth(
                        field,
                        target,
                        image,
                        phase_cfg,
                        constraint,
                        schedule,
                        k,
                        "coverage",
                        active_n=active_n,
                    ),
                ),
                (
                    "detail_birth",
                    min(schedule.detail_birth_count, room),
                    lambda k, image: propose_birth(
                        field,
                        target,
                        image,
                        phase_cfg,
                        constraint,
                        schedule,
                        k,
                        "detail",
                        active_n=active_n,
                    ),
                ),
            ]
            if schedule.boundary_enabled:
                factories.insert(
                    1,
                    (
                        "boundary_birth",
                        min(schedule.boundary_birth_count, room),
                        lambda k, image: propose_birth(
                            field,
                            target,
                            image,
                            phase_cfg,
                            constraint,
                            schedule,
                            k,
                            "boundary",
                            active_n=active_n,
                        ),
                    ),
                )
            return factories
        count = int(schedule.redistribution_count)
        factories = [
            (
                "merge_rebirth",
                count,
                lambda k, image: propose_merge_rebirth(
                    field,
                    target,
                    image,
                    phase_cfg,
                    constraint,
                    schedule,
                    k,
                    active_n=active_n,
                ),
            ),
            (
                "prune_rebirth",
                count,
                lambda k, image: propose_prune_rebirth(
                    field,
                    target,
                    image,
                    phase_cfg,
                    constraint,
                    schedule,
                    k,
                    active_n=active_n,
                ),
            ),
            (
                "funded_split",
                max(schedule.event_min_count, count // 2),
                lambda k, image: propose_funded_split(
                    field,
                    target,
                    image,
                    phase_cfg,
                    constraint,
                    schedule,
                    k,
                    active_n=active_n,
                ),
            ),
        ]
        if (
            schedule.boundary_enabled
            and
            schedule.boundary_recycle_at_capacity
            and metrics.boundary_hole_fraction > schedule.boundary_hole_target
        ):
            factories.extend([
                (
                    "merge_boundary_rebirth",
                    count,
                    lambda k, image: propose_merge_rebirth(
                        field,
                        target,
                        image,
                        phase_cfg,
                        constraint,
                        schedule,
                        k,
                        birth_mode="boundary",
                        active_n=active_n,
                    ),
                ),
                (
                    "prune_boundary_rebirth",
                    count,
                    lambda k, image: propose_prune_rebirth(
                        field,
                        target,
                        image,
                        phase_cfg,
                        constraint,
                        schedule,
                        k,
                        birth_mode="boundary",
                        active_n=active_n,
                    ),
                ),
            ])
        return factories

    run_phase(
        schedule.redistribution,
        topology=redistribution_factories,
        exit_condition=lambda: False,
    )

    # A second safe color solve separates final color error before the low-LR polish.
    field, optimizer_state, metrics, color_record = _safe_color_solve(
        field,
        optimizer_state,
        metrics,
        target,
        mask_tensor,
        cfg,
        constraint,
        schedule,
        active_n=active_n,
    )
    color_record["phase"] = "pre_polish_color_solve"
    emit(color_record)

    tail_start_n = _logical_n(field, active_n)
    tail_limit = min(
        int(schedule.capacity),
        int(base_active_limit) + int(schedule.detail_tail_max_rows),
        int(tail_start_n) + int(schedule.detail_tail_max_rows),
    )
    tail_wave_budget = (
        0
        if schedule.detail_tail_max_rows <= 0
        else math.ceil(
            int(schedule.detail_tail_max_rows)
            / int(schedule.detail_tail_batch_rows)
        )
    )
    tail_waves_attempted = 0
    tail_waves_accepted = 0
    tail_termination_reason = "disabled"
    if schedule.detail_tail_max_rows > 0:
        tail_termination_reason = "wave_budget"
        tail_fit_cfg = _phase_fit_config(
            cfg,
            schedule.polish,
            schedule.recovery_steps,
            boundary_enabled=schedule.boundary_enabled,
        )
        while (
            _logical_n(field, active_n) < tail_limit
            and tail_waves_attempted < tail_wave_budget
        ):
            tail_waves_attempted += 1
            room = tail_limit - _logical_n(field, active_n)
            requested = min(int(schedule.detail_tail_batch_rows), int(room))
            factories = [
                (
                    "detail_tail_birth",
                    requested,
                    lambda k, image: propose_birth(
                        field,
                        target,
                        image,
                        tail_fit_cfg,
                        constraint,
                        schedule,
                        k,
                        "detail_tail",
                        active_n=active_n,
                    ),
                ),
                (
                    "detail_tail_split",
                    requested,
                    lambda k, image: propose_split(
                        field,
                        target,
                        image,
                        tail_fit_cfg,
                        constraint,
                        schedule,
                        k,
                        active_n=active_n,
                        detail_tail_only=True,
                    ),
                ),
            ]
            (
                field,
                optimizer_state,
                metrics,
                topology_record,
                _auction_attempted,
                _auction_accepted,
                active_n,
            ) = _topology_auction(
                field,
                optimizer_state,
                metrics,
                target,
                mask_cpu,
                mask_tensor,
                tail_fit_cfg,
                constraint,
                schedule,
                factories,
                active_n=active_n,
                expand_neighborhood=False,
                minimum_gain_per_row=float(
                    schedule.detail_tail_min_gain_per_row
                ),
                verbose=verbose,
            )
            topology_record["phase"] = "detail_tail"
            topology_record["metadata"].update({
                "wave": tail_waves_attempted,
                "wave_budget": tail_wave_budget,
                "tail_start_n": tail_start_n,
                "tail_limit": tail_limit,
            })
            emit(topology_record)
            if not topology_record["accepted"]:
                tail_termination_reason = "no_safe_effective_winner"
                break
            tail_waves_accepted += 1
        else:
            if _logical_n(field, active_n) >= tail_limit:
                tail_termination_reason = "row_budget"

    tail_end_n = _logical_n(field, active_n)
    emit(_event_record(
        phase="detail_tail",
        event="phase_end",
        accepted=True,
        before=metrics,
        candidate=metrics,
        selected=metrics,
        attempted_steps=0,
        accepted_steps=0,
        metadata={
            "termination_reason": tail_termination_reason,
            "start_n": tail_start_n,
            "end_n": tail_end_n,
            "activated_rows": tail_end_n - tail_start_n,
            "configured_max_rows": int(schedule.detail_tail_max_rows),
            "batch_rows": int(schedule.detail_tail_batch_rows),
            "wave_budget": tail_wave_budget,
            "waves_attempted": tail_waves_attempted,
            "waves_accepted": tail_waves_accepted,
            "minimum_gain_per_row": float(
                schedule.detail_tail_min_gain_per_row
            ),
        },
    ))
    if tail_waves_accepted > 0:
        field, optimizer_state, metrics, color_record = _safe_color_solve(
            field,
            optimizer_state,
            metrics,
            target,
            mask_tensor,
            cfg,
            constraint,
            schedule,
            active_n=active_n,
        )
        color_record["phase"] = "post_detail_tail_color_solve"
        emit(color_record)

    run_phase(
        schedule.polish,
        topology=None,
        exit_condition=lambda: False,
    )

    final_active_n = _logical_n(field, active_n)
    physical_capacity = int(field.n)
    storage_record = {
        "policy": schedule.storage_policy,
        "capacity": int(schedule.capacity),
        "base_active_limit": int(base_active_limit),
        "detail_tail_max_rows": int(schedule.detail_tail_max_rows),
        "detail_tail_activated_rows": int(tail_end_n - tail_start_n),
        "initial_active_n": initial_n,
        "final_active_n": final_active_n,
        "initial_physical_rows": initial_physical_n,
        "final_physical_rows": physical_capacity,
        "fixed_shape_during_fit": active_n is not None
        and initial_physical_n == physical_capacity,
        "output_compacted": active_n is not None,
    }
    if schedule.storage_policy == "geometric":
        storage_record["growth_factor"] = float(schedule.growth_factor)
        storage_record["initial_capacity"] = int(
            schedule.resolved_initial_capacity(initial_n)
        )
        storage_record["geometric_migrations"] = len(geometric_migrations)
        storage_record["geometric_migration_events"] = list(geometric_migrations)
    if active_n is not None:
        empty = torch.empty(
            0, device=field.means.device, dtype=torch.long
        )
        optimizer_state = adapt_optimizer_state(
            optimizer_state,
            field.n,
            final_active_n,
            empty,
        )
        field = field.subset(slice(0, final_active_n))
        active_n = None

    return {
        "field": field,
        "arm": "full_frame" if full_frame else "masked",
        "metrics": metrics.to_dict(),
        "history": history,
        "attempted_steps": attempted_steps,
        "accepted_steps": accepted_steps,
        "optimizer_state": optimizer_state,
        "storage": storage_record,
        "schedule": asdict(schedule),
        "fit_config": asdict(cfg),
        "converged": bool(
            history
            and history[-1]["phase"] == schedule.polish.name
            and history[-1]["metadata"].get("termination_reason")
            == "deterministic_fixed_point"
        ),
        "seconds": time.perf_counter() - started,
    }
