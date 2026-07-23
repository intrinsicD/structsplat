"""Transactional, phase-ordered optimization for mask-contained 2D Gaussian fields.

The legacy fitter exposes all individual mechanisms (birth, split, merge, relocation, pruning),
but a fixed timer order can spend capacity before the operation with the largest useful delta
gets a chance to run.  This module implements the production schedule described by the Janelle
boundary investigation:

* fixed-topology bootstrap;
* coverage-first birth;
* detail birth / moment-preserving split;
* explicit boundary closure;
* count-neutral redistribution at capacity;
* fixed-topology polish.

Every optimizer block and every topology proposal runs on a detached trial field.  A full-frame
metric vector decides whether both parameters and optimizer moments are committed or discarded.
Topology batches are reduced geometrically after a rejected trial, so the schedule can retain
safe members of a coarse proposal without falling back to per-Gaussian Python mutation.

The module intentionally does not replace FIT-021's legacy pooled path.  It is an opt-in
orchestration layer whose strict acceptance semantics can be tested independently.
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

    def validate(self, initial_n: int) -> None:
        if initial_n <= 0:
            raise ValueError("safe schedule needs a non-empty initial field")
        if not initial_n <= self.coverage_target_gaussians <= self.detail_target_gaussians:
            raise ValueError(
                "expected initial_n <= coverage_target_gaussians <= "
                "detail_target_gaussians"
            )
        if self.detail_target_gaussians > self.capacity:
            raise ValueError("detail_target_gaussians cannot exceed capacity")
        if self.capacity < initial_n:
            raise ValueError(
                f"capacity {self.capacity} is below initial field size {initial_n}"
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
            if phase.max_steps <= 0 or phase.block_steps <= 0:
                raise ValueError(f"phase {phase.name} must have positive step budgets")

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


ScheduleObserver = Callable[[GaussianField, dict[str, Any]], None]


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
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Return seed rows plus their direct Euclidean neighbours as one batched row mask."""

    seeds = torch.unique(
        seed_rows.detach().to(device=field.means.device, dtype=torch.long)
    )
    seeds = seeds[(seeds >= 0) & (seeds < field.n)]
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
    if requested > 0 and field.n > 1:
        k = min(field.n, requested + 1)
        try:
            import numpy as np
            from scipy.spatial import cKDTree

            values = field.means.detach().cpu().numpy()
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
            means = field.means.detach()
            for chunk in seeds.split(256):
                distances = torch.cdist(means[chunk], means)
                neighbours = torch.topk(
                    distances, k=k, largest=False, dim=1
                ).indices
                trainable[neighbours.reshape(-1)] = True
            backend = "torch_cdist"
    else:
        backend = "seeds_only"
    if field.background_mask is not None:
        trainable &= ~field.background_mask.to(
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
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Select high-error owners and spatial neighbours while freezing the accepted remainder."""

    H, W = target.shape[:2]
    render = _render(field, cfg, H, W, support_fade_alpha=1.0)
    score, components = _responsibility_error_density_scores(
        field, target, render, cfg, support_fade_alpha=1.0
    )
    score = score.detach().clone()
    valid = torch.isfinite(score)
    boundary_only = phase_name == schedule.boundary.name
    if boundary_only:
        means = field.means.detach()
        ix = means[:, 0].round().long().clamp(0, W - 1)
        iy = means[:, 1].round().long().clamp(0, H - 1)
        sdf = constraint.sdf_flat[iy * W + ix]
        reach = (
            field.effective_scales(cfg.aa_dilation).detach().amax(dim=1)
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
    if field.background_mask is not None:
        valid &= ~field.background_mask.to(device=valid.device, dtype=torch.bool)
    candidates = valid.nonzero(as_tuple=False).reshape(-1)
    if candidates.numel() == 0 and boundary_only:
        # A pathological thin mask may have no centre in the band even though supports overlap it.
        valid = torch.isfinite(score)
        if field.background_mask is not None:
            valid &= ~field.background_mask.to(device=valid.device, dtype=torch.bool)
        candidates = valid.nonzero(as_tuple=False).reshape(-1)
    k = min(int(schedule.local_seed_count), int(candidates.numel()))
    if k <= 0:
        raise RuntimeError("local residual refinement found no trainable Gaussian rows")
    seeds = candidates[torch.topk(score[candidates], k=k).indices]
    trainable, neighborhood = _expand_spatial_neighborhood(
        field, seeds, schedule.local_neighbor_count
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
) -> dict[str, Any]:
    """Measure persistent reachable-band undercoverage and high-error components."""

    H, W = target.shape[:2]
    render = _render(field, cfg, H, W, support_fade_alpha=1.0)
    denominator = _raw_weight_map_field(
        field, cfg, H, W, support_fade_alpha=1.0
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
) -> tuple[QualityMetrics, torch.Tensor]:
    """Render and compute the complete safe-commit vector on the full target."""

    H, W = target.shape[:2]
    render = _render(field, cfg, H, W, support_fade_alpha=1.0)
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
    p99 = torch.quantile(fg_values, 0.99)
    denominator = _raw_weight_map_field(
        field, cfg, H, W, support_fade_alpha=1.0
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
        n_gaussians=int(field.n),
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
) -> tuple[bool, list[str]]:
    """Return a Pareto-safe decision and explicit rejection reasons."""

    reasons: list[str] = []
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
        > before.interior_hole_fraction + tolerance.hole_absolute
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
        mask_undercoverage_weight=float(phase.boundary_floor_weight),
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
) -> tuple[GaussianField, dict | None, QualityMetrics, dict[str, Any]]:
    trial = field.detached()
    with torch.no_grad():
        solve_stats = _solve_colors_normalized(
            trial, target, base_cfg, target.shape[0], target.shape[1],
            support_fade_alpha=1.0,
        )
        constraint.apply(trial, base_cfg, refresh=True)
    candidate, _ = evaluate_quality(
        trial, target, mask, base_cfg, constraint, schedule.coverage_tau
    )
    accepted, reasons = safe_commit_decision(metrics, candidate, schedule.tolerances)
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
        constraint.apply(candidate_field, cfg, refresh=True)
        if trainable_rows is not None:
            candidate_state = restore_frozen_optimizer_rows(
                candidate_state, incoming_state, trainable_rows
            )
        candidate, _ = evaluate_quality(
            candidate_field, target, mask, cfg, constraint, schedule.coverage_tau
        )
        accepted, reasons = safe_commit_decision(
            metrics, candidate, schedule.tolerances
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
                    solved_field,
                    target,
                    cfg,
                    target.shape[0],
                    target.shape[1],
                    support_fade_alpha=1.0,
                )
                constraint.apply(solved_field, cfg, refresh=True)
            solved_metrics, _ = evaluate_quality(
                solved_field,
                target,
                mask,
                cfg,
                constraint,
                schedule.coverage_tau,
            )
            solved_accepted, solved_reasons = safe_commit_decision(
                metrics, solved_metrics, schedule.tolerances
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
                int(field.n) if trainable_rows is None else int(trainable_rows.sum())
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
        base_cfg, phase, trial_steps, lr_scale=trial_lr_scale
    )
    final_record: dict[str, Any] | None = None
    for attempt_index in range(4):
        selected_cfg = _phase_fit_config(
            base_cfg, phase, trial_steps, lr_scale=trial_lr_scale
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
    residual_ref = torch.quantile(residual[constraint.inside], 0.95).clamp_min(1e-8)
    residual_n = (residual / residual_ref).clamp(0.0, 4.0)
    tangent, coherence = _local_structure(target)
    deficit = ((float(schedule.coverage_tau) - den) / float(schedule.coverage_tau)).clamp(0, 1)

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
) -> TopologyProposal | None:
    room = max(0, int(schedule.capacity) - int(field.n))
    components, metadata = _birth_components(
        field, target, render_img, cfg, constraint, schedule, min(int(count), room), mode
    )
    if components is None or components.n == 0:
        return None
    trial = field.detached().append(components)
    constraint.apply(trial, cfg, refresh=True)
    touched = torch.arange(
        field.n, trial.n, device=field.means.device, dtype=torch.long
    )
    return TopologyProposal(
        kind=f"{mode}_birth",
        field=trial,
        touched=touched,
        count=int(components.n),
        metadata=metadata,
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
) -> TopologyProposal | None:
    room = max(0, int(schedule.capacity) - int(field.n))
    k = min(int(count), room, int(field.n))
    if k <= 0:
        return None
    scores, components = _responsibility_error_density_scores(
        field, target, render_img, cfg, support_fade_alpha=1.0
    )
    # A split is for a broad, error-owning primitive. Tiny owners are left to sampled births.
    footprint = field.scales().detach().prod(dim=1)
    score = scores * torch.sqrt(
        footprint / torch.quantile(footprint, 0.75).clamp_min(1e-8)
    )
    if field.background_mask is not None:
        score = score.masked_fill(field.background_mask, -float("inf"))
    parents = torch.topk(score, k=k).indices

    means = field.means.detach().clone()
    log_scales = field.log_scales.detach().clone()
    rotations = field.rotations.detach().clone()
    colors = field.colors.detach().clone()
    color_grads = None if field.color_grads is None else field.color_grads.detach().clone()
    scales = field.scales().detach()[parents]
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

    if field.opacities is None:
        opacities = torch.full(
            (field.n,), 10.0, device=field.means.device, dtype=field.means.dtype
        )
        parent_opacity = torch.ones(k, device=field.means.device, dtype=field.means.dtype)
    else:
        opacities = field.opacities.detach().clone()
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
        None if field.scale_max is None else field.scale_max.detach()[parents].clone(),
        None if color_grads is None else color_grads[parents].clone(),
        filter_variance=(
            None
            if field.filter_variance is None
            else field.filter_variance.detach()[parents].clone()
        ),
    )
    parent_field = GaussianField(
        means,
        log_scales,
        rotations,
        colors,
        opacities,
        None if field.scale_max is None else field.scale_max.detach().clone(),
        color_grads,
        None if field.background_mask is None else field.background_mask.detach().clone(),
        None if field.filter_variance is None else field.filter_variance.detach().clone(),
    )
    trial = parent_field.append(child)
    constraint.apply(trial, cfg, refresh=True)
    children = torch.arange(field.n, trial.n, device=parents.device)
    touched = torch.unique(torch.cat([parents, children]))
    return TopologyProposal(
        kind="moment_preserving_split",
        field=trial,
        touched=touched,
        count=k,
        metadata={
            "parents": k,
            "score_mean": float(score[parents].mean()),
            "responsibility_mass_mean": float(components["mass"][parents].mean()),
            "split_shrink": shrink,
            "mass_rule": "opacity divided by 2*axis_shrink (integrated raw mass preserving)",
        },
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
) -> TopologyProposal | None:
    stats = attribution_pass(field, target, render_img, cfg, support_fade_alpha=1.0)
    eligible = stats["max_responsibility"] < float(
        schedule.redistribution_min_responsibility
    )
    if field.background_mask is not None:
        eligible &= ~field.background_mask
    candidates = eligible.nonzero(as_tuple=False).reshape(-1)
    k = min(int(count), int(candidates.numel()))
    if k <= 0:
        return None
    donor_score = stats["activity"][candidates] + stats["error"][candidates]
    donors = candidates[torch.argsort(donor_score)[:k]]
    births, metadata = _birth_components(
        field, target, render_img, cfg, constraint, schedule, k, birth_mode
    )
    if births is None:
        return None
    k = min(k, int(births.n))
    donors = donors[:k]
    trial = field.detached()
    _replace_rows(trial, donors, births.subset(slice(0, k)))
    constraint.apply(trial, cfg, refresh=True)
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
) -> TopologyProposal | None:
    """Batch midpoint/envelope merge with the absorbed row directly reborn at high error."""

    if field.n < 4 or count <= 0:
        return None
    stats = attribution_pass(field, target, render_img, cfg, support_fade_alpha=1.0)
    first, second, distance = _production_mutual_nearest_pairs(field.means.detach())
    if first.numel() == 0:
        return None
    colors = field.colors.detach()
    color_delta = torch.linalg.norm(colors[first] - colors[second], dim=1)
    sorted_scales = torch.sort(field.scales().detach(), dim=1).values.clamp_min(1e-6)
    scale_delta = torch.linalg.norm(
        torch.log(sorted_scales[first]) - torch.log(sorted_scales[second]), dim=1
    )
    angle_delta = field.rotations.detach()[first] - field.rotations.detach()[second]
    axial_delta = 0.5 * torch.abs(
        torch.atan2(torch.sin(2 * angle_delta), torch.cos(2 * angle_delta))
    )
    minor = 0.5 * (sorted_scales[first, 0] + sorted_scales[second, 0])
    allowed_distance = torch.clamp(
        2.5 * minor, min=1.25, max=4.0
    )
    midpoint = 0.5 * (field.means.detach()[first] + field.means.detach()[second])
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
        field, target, render_img, cfg, constraint, schedule, k, birth_mode
    )
    if births is None:
        return None
    k = min(k, int(births.n))
    keep, rebirth = keep[:k], rebirth[:k]
    births = births.subset(slice(0, k))

    trial = field.detached()
    means = field.means.detach()
    if trial.opacities is None:
        trial.opacities = torch.full(
            (field.n,), 10.0, device=means.device, dtype=means.dtype
        )
    amplitude = torch.sigmoid(trial.opacities.detach())
    wa, wb = amplitude[keep], amplitude[rebirth]
    total = (wa + wb).clamp_min(1e-6)
    merged_mean = 0.5 * (means[keep] + means[rebirth])
    cov_a = _row_covariances(field, keep)
    cov_b = _row_covariances(field, rebirth)
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
        wa[:, None] * field.colors.detach()[keep]
        + wb[:, None] * field.colors.detach()[rebirth]
    ) / total[:, None]
    combined = (1.0 - (1.0 - wa) * (1.0 - wb)).clamp(1e-4, 1.0 - 1e-4)
    trial.opacities[keep] = _logit(combined)
    if trial.color_grads is not None:
        gradients = field.color_grads.detach()
        trial.color_grads[keep] = (
            wa[:, None, None] * gradients[keep]
            + wb[:, None, None] * gradients[rebirth]
        ) / total[:, None, None]
    if trial.filter_variance is not None:
        variance = field.filter_variance.detach()
        trial.filter_variance[keep] = (
            wa * variance[keep] + wb * variance[rebirth]
        ) / total
    if trial.scale_max is not None:
        trial.scale_max[keep] = float("inf")
    # Atomic identity: each absorbed partner is now the corresponding high-error birth. It is
    # never exposed through a shared free/teleport list.
    _replace_rows(trial, rebirth, births)
    constraint.apply(trial, cfg, refresh=True)

    certified = _row_covariances(trial, keep)
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
        original = field.subset(bad_rows)
        _replace_rows(trial, bad_rows, original)
        keep, rebirth = keep[good], rebirth[good]
        constraint.apply(trial, cfg, refresh=True)
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
) -> TopologyProposal | None:
    """Count-neutral moment split funded by low-responsibility donor rows."""

    stats = attribution_pass(field, target, render_img, cfg, support_fade_alpha=1.0)
    donors_ok = stats["max_responsibility"] < float(
        schedule.redistribution_min_responsibility
    )
    if field.background_mask is not None:
        donors_ok &= ~field.background_mask
    donors_all = donors_ok.nonzero(as_tuple=False).reshape(-1)
    scores, _ = _responsibility_error_density_scores(
        field, target, render_img, cfg, support_fade_alpha=1.0
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
    scales = field.scales().detach()[parents]
    theta = field.rotations.detach()[parents]
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
    original_means = field.means.detach()[parents]
    trial.means[parents] = original_means - offset
    trial.means[donors] = original_means + offset
    parent_log = field.log_scales.detach()[parents].clone()
    axis = torch.where(use_x, 0, 1)
    row = torch.arange(k, device=parents.device)
    parent_log[row, axis] += math.log(shrink)
    trial.log_scales[parents] = parent_log
    trial.log_scales[donors] = parent_log
    trial.rotations[donors] = theta
    trial.colors[donors] = field.colors.detach()[parents]
    parent_opacity = torch.sigmoid(field.opacities.detach()[parents]) \
        if field.opacities is not None else torch.full(
            (k,), 1.0, device=parents.device, dtype=field.means.dtype
        )
    child_opacity = (parent_opacity / (2.0 * shrink)).clamp(1e-4, 1.0 - 1e-4)
    trial.opacities[parents] = _logit(child_opacity)
    trial.opacities[donors] = _logit(child_opacity)
    if trial.color_grads is not None:
        trial.color_grads[donors] = field.color_grads.detach()[parents]
    if trial.filter_variance is not None:
        trial.filter_variance[donors] = field.filter_variance.detach()[parents]
    if trial.scale_max is not None:
        trial.scale_max[torch.cat([parents, donors])] = float("inf")
    constraint.apply(trial, cfg, refresh=True)
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
    expand_neighborhood: bool | None = None,
    verbose: bool,
) -> tuple[GaussianField, dict | None, QualityMetrics, dict[str, Any], int, int]:
    """Evaluate operator batches transactionally and commit the best Pareto-safe trial."""

    safe_trials: list[_Trial] = []
    attempts: list[dict[str, Any]] = []
    attempted_steps = 0
    # Every operator competes against the same accepted state. Render it once; trials never
    # mutate ``field`` before the winner is selected.
    _, current_render = evaluate_quality(
        field, target, mask, fit_cfg, constraint, schedule.coverage_tau
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
                verbose=False,
            )
            attempted_steps += int(output["iterations_run"])
            trial_metadata = {
                **proposal.metadata,
                "recovery_neighborhood": neighborhood,
                "recovery": copy.deepcopy(record["metadata"]),
            }
            attempts.append({
                "kind": kind,
                "requested": count,
                "proposed": proposal.count,
                "accepted": bool(record["accepted"]),
                "reasons": record["reasons"],
                "candidate": record["candidate"],
                "metadata": trial_metadata,
            })
            if record["accepted"]:
                safe_trials.append(_Trial(
                    kind=proposal.kind,
                    field=recovered,
                    optimizer_state=recovered_state,
                    metrics=recovered_metrics,
                    count=proposal.count,
                    metadata=trial_metadata,
                    attempted_count=count,
                    recovery_steps=int(record["accepted_steps"]),
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
        return field, optimizer_state, metrics, record, attempted_steps, 0

    winner = max(safe_trials, key=lambda trial: _trial_gain(metrics, trial.metrics))
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
            "winner_gain": _trial_gain(metrics, winner.metrics),
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
    )


def run_safe_schedule(
    initial_field: GaussianField,
    target: torch.Tensor,
    mask,
    base_cfg: FitConfig,
    schedule: SafeScheduleConfig | None = None,
    *,
    observer: ScheduleObserver | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Run the complete safe-commit schedule and return the best accepted field."""

    schedule = SafeScheduleConfig() if schedule is None else schedule
    schedule.validate(initial_field.n)
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
        mask_contain=True,
        mask_cap_mode="anisotropic",
        mask_undercoverage_band=float(schedule.boundary_band),
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
    constraint.apply(field, cfg, refresh=True)
    metrics, _ = evaluate_quality(
        field, target, mask_tensor, cfg, constraint, schedule.coverage_tau
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
            observer(field, record)

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
        field, optimizer_state, metrics, target, mask_tensor, cfg, constraint, schedule
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
        nonlocal field, optimizer_state, metrics
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
                    cfg, phase, steps, lr_scale=lr_scale
                )
                trainable_rows, selection_metadata = _local_residual_row_mask(
                    field,
                    target,
                    selection_cfg,
                    constraint,
                    schedule,
                    phase.name,
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
                    (
                        field,
                        optimizer_state,
                        metrics,
                        topology_record,
                        _auction_attempted,
                        _auction_accepted,
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
                    f"[{phase.name}] block={block_index} n={field.n} "
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
        room = max(0, schedule.coverage_target_gaussians - field.n)
        count = min(schedule.coverage_birth_count, room)
        return [(
            "coverage_birth",
            count,
            lambda k, image: propose_birth(
                field, target, image, phase_cfg, constraint, schedule, k, "coverage"
            ),
        )] if count > 0 else []

    run_phase(
        schedule.coverage,
        topology=coverage_factories,
        exit_condition=lambda: (
            field.n >= schedule.coverage_target_gaussians
            or (
                metrics.interior_hole_fraction <= schedule.interior_hole_target
                and metrics.boundary_hole_fraction <= schedule.boundary_hole_target
            )
        ),
    )

    def detail_factories(phase_cfg: FitConfig):
        room = max(0, schedule.detail_target_gaussians - field.n)
        if room <= 0:
            return []
        return [
            (
                "detail_birth",
                min(schedule.detail_birth_count, room),
                lambda k, image: propose_birth(
                    field, target, image, phase_cfg, constraint, schedule, k, "detail"
                ),
            ),
            (
                "moment_preserving_split",
                min(schedule.detail_split_count, room),
                lambda k, image: propose_split(
                    field, target, image, phase_cfg, constraint, schedule, k
                ),
            ),
            (
                "coverage_birth",
                min(schedule.detail_birth_count, room),
                lambda k, image: propose_birth(
                    field, target, image, phase_cfg, constraint, schedule, k, "coverage"
                ),
            ),
        ]

    run_phase(
        schedule.detail,
        topology=detail_factories,
        exit_condition=lambda: field.n >= schedule.detail_target_gaussians,
    )

    def boundary_factories(phase_cfg: FitConfig):
        room = max(0, schedule.capacity - field.n)
        count = min(schedule.boundary_birth_count, room)
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
                ),
            ),
        ]

    def boundary_diagnostics() -> dict[str, Any]:
        return _boundary_defect_summary(
            field, target, cfg, constraint, schedule
        )

    def boundary_target_reached() -> bool:
        if not schedule.boundary_recycle_at_capacity and field.n >= schedule.capacity:
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
        diagnostics=boundary_diagnostics,
    )

    def redistribution_factories(phase_cfg: FitConfig):
        room = max(0, schedule.capacity - field.n)
        if room > 0:
            return [
                (
                    "coverage_birth",
                    min(schedule.detail_birth_count, room),
                    lambda k, image: propose_birth(
                        field, target, image, phase_cfg, constraint, schedule, k, "coverage"
                    ),
                ),
                (
                    "boundary_birth",
                    min(schedule.boundary_birth_count, room),
                    lambda k, image: propose_birth(
                        field, target, image, phase_cfg, constraint, schedule, k, "boundary"
                    ),
                ),
                (
                    "detail_birth",
                    min(schedule.detail_birth_count, room),
                    lambda k, image: propose_birth(
                        field, target, image, phase_cfg, constraint, schedule, k, "detail"
                    ),
                ),
            ]
        count = int(schedule.redistribution_count)
        factories = [
            (
                "merge_rebirth",
                count,
                lambda k, image: propose_merge_rebirth(
                    field, target, image, phase_cfg, constraint, schedule, k
                ),
            ),
            (
                "prune_rebirth",
                count,
                lambda k, image: propose_prune_rebirth(
                    field, target, image, phase_cfg, constraint, schedule, k
                ),
            ),
            (
                "funded_split",
                max(schedule.event_min_count, count // 2),
                lambda k, image: propose_funded_split(
                    field, target, image, phase_cfg, constraint, schedule, k
                ),
            ),
        ]
        if (
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
        field, optimizer_state, metrics, target, mask_tensor, cfg, constraint, schedule
    )
    color_record["phase"] = "pre_polish_color_solve"
    emit(color_record)
    run_phase(
        schedule.polish,
        topology=None,
        exit_condition=lambda: False,
    )

    return {
        "field": field,
        "metrics": metrics.to_dict(),
        "history": history,
        "attempted_steps": attempted_steps,
        "accepted_steps": accepted_steps,
        "optimizer_state": optimizer_state,
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
