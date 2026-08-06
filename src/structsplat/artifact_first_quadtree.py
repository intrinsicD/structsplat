"""Artifact-first parent-replacing frontier quadtree (HIER-007).

This default-off research reference keeps the HIER-006 quadtree as an allocation/index
structure, but not as an active Gaussian prefix.  Splitting a frontier parent deactivates that
parent, activates every mask-present child, and optionally reconciles the new child RGB values
with surviving active rows whose finite supports overlap the selected cells.  Geometry remains
fixed and all nonlocal coefficients are detached.

Torch is imported lazily inside optimization/render bridges.  Byte values labeled ``proxy`` are
not coded-stream rates: they omit a complete grammar, indices, headers, quantization, and entropy
coding.  COMP-013 remains authoritative for actual rate.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import time
from typing import Literal

import numpy as np

from .observation_field import ObservationField2D
from .pixel_contraction import (
    _mask_aware_smoothed_error,
    _normalized_error_update_weights,
)
from .progressive_residual_quadtree import (
    HierarchyRenderer,
    NodeKey,
    _base_keys,
    _cell_bounds,
    _cell_rgb_mean,
    _child_keys,
    _field_from_arrays,
    _finite,
    _geometry_rows,
    _integer,
    _lexicographic_improves,
    _validated_image,
    _validated_mask,
    progressive_artifact_metrics,
)


SelectionMode = Literal["energy", "artifact_first"]
ReconciliationScope = Literal["new_only", "overlap"]


def _array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class ArtifactFirstQuadtreeConfig:
    """Configuration for :func:`build_artifact_first_quadtree`.

    ``selection_mode`` and ``reconciliation_scope`` are the two experimental axes.  All geometry
    is reconstructed from the mask and node keys; ``estimated_row_bytes`` and every structural
    byte value are explicitly uncoded proxies.
    """

    selection_mode: SelectionMode = "artifact_first"
    reconciliation_scope: ReconciliationScope = "overlap"
    seed: int = 0
    start_level: int = 6
    max_gaussians: int = 8192
    leaf_scale_px: float = 0.18
    sigma_cutoff: float = 3.0
    support_fade_alpha: float = 0.0
    error_smoothing_sigma_px: float = 1.5
    error_weight_power: float = 0.5
    error_weight_floor: float = 0.05
    error_weight_ceiling: float = 4.0
    overlap_margin_px: int = 3
    max_child_rows_per_stage: int = 256
    base_steps: int = 400
    layer_steps: int = 50
    learning_rate: float = 0.05
    tail_fraction: float = 0.01
    tail_weight: float = 4.0
    checkpoint_every: int = 5
    pixel_rmse_threshold: float = 0.02
    patch7_rmse_threshold: float = 0.01
    estimated_row_bytes: int = 32
    device: str = "cpu"
    renderer: HierarchyRenderer = "additive"
    render_chunk: int = 256
    max_stages: int | None = None
    milestone_counts: tuple[int, ...] = (4096,)

    def __post_init__(self) -> None:
        if self.selection_mode not in ("energy", "artifact_first"):
            raise ValueError("selection_mode must be 'energy' or 'artifact_first'")
        if self.reconciliation_scope not in ("new_only", "overlap"):
            raise ValueError("reconciliation_scope must be 'new_only' or 'overlap'")
        for name, minimum in (
            ("seed", 0),
            ("start_level", 0),
            ("overlap_margin_px", 0),
        ):
            object.__setattr__(self, name, _integer(getattr(self, name), name, minimum=minimum))
        for name in (
            "max_gaussians",
            "max_child_rows_per_stage",
            "base_steps",
            "layer_steps",
            "checkpoint_every",
            "estimated_row_bytes",
            "render_chunk",
        ):
            object.__setattr__(self, name, _integer(getattr(self, name), name, minimum=1))
        if self.max_stages is not None:
            object.__setattr__(
                self,
                "max_stages",
                _integer(self.max_stages, "max_stages", minimum=1),
            )
        for name in (
            "leaf_scale_px",
            "sigma_cutoff",
            "error_weight_power",
            "error_weight_floor",
            "error_weight_ceiling",
            "learning_rate",
            "tail_weight",
            "pixel_rmse_threshold",
            "patch7_rmse_threshold",
        ):
            object.__setattr__(
                self,
                name,
                _finite(getattr(self, name), name, minimum=0.0, strict=True),
            )
        object.__setattr__(
            self,
            "error_smoothing_sigma_px",
            _finite(
                self.error_smoothing_sigma_px,
                "error_smoothing_sigma_px",
                minimum=0.0,
            ),
        )
        fade = _finite(self.support_fade_alpha, "support_fade_alpha", minimum=0.0)
        if fade > 1.0:
            raise ValueError("support_fade_alpha must be <= 1")
        object.__setattr__(self, "support_fade_alpha", fade)
        fraction = _finite(self.tail_fraction, "tail_fraction", minimum=0.0, strict=True)
        if fraction > 1.0:
            raise ValueError("tail_fraction must be <= 1")
        object.__setattr__(self, "tail_fraction", fraction)
        if not (self.error_weight_floor <= 1.0 <= self.error_weight_ceiling):
            raise ValueError("error-weight bounds must satisfy floor <= 1 <= ceiling")
        if not isinstance(self.device, str) or not self.device.strip():
            raise ValueError("device must be a non-empty torch device string")
        if self.renderer not in ("additive", "cuda_additive", "cuda_tiled_additive"):
            raise ValueError("unsupported renderer")
        if self.renderer.startswith("cuda") and not self.device.startswith("cuda"):
            raise ValueError("a CUDA renderer requires a CUDA device")
        if not isinstance(self.milestone_counts, tuple):
            raise TypeError("milestone_counts must be a tuple")
        milestones = []
        for value in self.milestone_counts:
            count = _integer(value, "milestone_counts item", minimum=1)
            if count <= self.max_gaussians:
                milestones.append(count)
        object.__setattr__(self, "milestone_counts", tuple(sorted(set(milestones))))


@dataclass(frozen=True)
class FrontierSplitCandidate:
    """One ranked active-parent split."""

    parent_key: NodeKey
    child_keys: tuple[NodeKey, ...]
    artifact_score: float
    energy_score: float
    net_active_rows: int


@dataclass(frozen=True)
class FrontierCheckpoint:
    """One evaluated optimizer checkpoint."""

    attempt_index: int
    step: int
    cumulative_optimizer_step: int
    candidate_count: int
    optimized_count: int
    raw_sse: float
    raw_pixel_rmse_max: float
    raw_patch7_rmse_max: float
    raw_normalized_violation: float
    objective: float
    elapsed_seconds: float

    def to_record(self) -> dict[str, object]:
        return {
            "attempt_index": self.attempt_index,
            "step": self.step,
            "cumulative_optimizer_step": self.cumulative_optimizer_step,
            "candidate_count": self.candidate_count,
            "optimized_count": self.optimized_count,
            "raw_sse": self.raw_sse,
            "raw_pixel_rmse_max": self.raw_pixel_rmse_max,
            "raw_patch7_rmse_max": self.raw_patch7_rmse_max,
            "raw_normalized_violation": self.raw_normalized_violation,
            "objective": self.objective,
            "elapsed_seconds": self.elapsed_seconds,
        }


@dataclass(frozen=True)
class FrontierStage:
    """One base fit or transactional split attempt."""

    attempt_index: int
    accepted_index: int | None
    kind: str
    status: str
    selection_mode: str
    reconciliation_scope: str
    parent_keys: tuple[NodeKey, ...]
    child_keys: tuple[NodeKey, ...]
    optimized_existing_keys: tuple[NodeKey, ...]
    count_before: int
    proposed_child_rows: int
    proposed_net_rows: int
    accepted_child_rows: int
    accepted_net_rows: int
    count_after: int
    stored_nodes_after: int
    inactive_nodes_after: int
    coefficient_event_rows_after: int
    artifact_score_max: float
    artifact_score_min: float
    energy_score_max: float
    energy_score_min: float
    optimized_rows: int
    frozen_rows: int
    selected_step: int
    attempted_optimizer_steps: int
    sse_before: float
    sse_after: float
    raw_violation_before: float
    raw_violation_after: float
    display_pixel_rmse_max: float
    display_patch7_rmse_max: float
    display_gate_pass: bool
    error_weight_min: float
    error_weight_mean: float
    error_weight_p50: float
    error_weight_p90: float
    error_weight_max: float
    error_weight_effective_rows: float
    raw_error_score_mean: float
    raw_error_score_max: float
    untouched_coefficients_bit_exact: bool
    rollback_bit_exact: bool
    frontier_partition_valid: bool
    optimized_vs_cold_max_abs: float
    selection_seconds: float
    attribution_seconds: float
    optimization_seconds: float
    cold_render_seconds: float
    cumulative_elapsed_seconds: float

    def to_record(self) -> dict[str, object]:
        record = dict(self.__dict__)
        for name in ("parent_keys", "child_keys", "optimized_existing_keys"):
            record[name] = [list(key) for key in record[name]]
        return record


@dataclass(frozen=True)
class FrontierSnapshot:
    """An independently persisted active-frontier field at one accepted stage."""

    label: str
    attempt_index: int
    active_keys: tuple[NodeKey, ...]
    field: ObservationField2D
    reconstruction_raw: np.ndarray
    stored_node_count: int
    coefficient_event_rows: int

    @property
    def active_count(self) -> int:
        return len(self.active_keys)


@dataclass(frozen=True)
class FrontierStartState:
    """Common fitted base cloned across factorial arms."""

    source_sha256: str
    mask_sha256: str
    base_signature: tuple[object, ...]
    active_keys: tuple[NodeKey, ...]
    colors: np.ndarray
    field: ObservationField2D
    reconstruction_raw: np.ndarray
    stage: FrontierStage
    checkpoints: tuple[FrontierCheckpoint, ...]
    initial_sse: float
    elapsed_seconds: float
    cumulative_optimizer_steps: int
    improved: bool


@dataclass(frozen=True)
class ArtifactFirstQuadtreeResult:
    """Final active field plus the complete HIER-007 diagnostic trajectory."""

    field: ObservationField2D
    reconstruction_raw: np.ndarray
    reconstruction: np.ndarray
    active_keys: tuple[NodeKey, ...]
    stored_keys: tuple[NodeKey, ...]
    stages: tuple[FrontierStage, ...]
    checkpoints: tuple[FrontierCheckpoint, ...]
    snapshots: tuple[FrontierSnapshot, ...]
    base_count: int
    final_count: int
    stored_node_count: int
    inactive_node_count: int
    accepted_split_count: int
    accepted_stage_count: int
    coefficient_event_rows: int
    initial_sse: float
    final_sse: float
    stop_reason: str
    shared_base_seconds: float
    arm_elapsed_seconds: float
    elapsed_seconds: float
    estimated_active_field_bytes: int
    canonical_active_raw_bytes: int
    active_coefficient_proxy_bytes: int
    tree_proxy_bits: int
    final_frontier_proxy_bytes: int
    progressive_event_coefficient_proxy_bytes: int
    progressive_event_proxy_bytes: int
    maintained_render_parity_max_abs: float

    def stage_records(self) -> list[dict[str, object]]:
        return [stage.to_record() for stage in self.stages]

    def checkpoint_records(self) -> list[dict[str, object]]:
        return [checkpoint.to_record() for checkpoint in self.checkpoints]


@dataclass(frozen=True)
class _ColorOptimization:
    colors: np.ndarray | None
    reconstruction_raw: np.ndarray | None
    selected_step: int
    selected_metrics: tuple[float, float, float, float]
    checkpoints: tuple[FrontierCheckpoint, ...]
    elapsed_seconds: float
    attribution_seconds: float
    attempted_steps: int
    weight_telemetry: dict[str, float]


def _base_signature(config: ArtifactFirstQuadtreeConfig) -> tuple[object, ...]:
    return (
        config.seed,
        config.start_level,
        config.leaf_scale_px,
        config.sigma_cutoff,
        config.support_fade_alpha,
        config.base_steps,
        config.learning_rate,
        config.tail_fraction,
        config.tail_weight,
        config.checkpoint_every,
        config.pixel_rmse_threshold,
        config.patch7_rmse_threshold,
        config.device,
        config.renderer,
        config.render_chunk,
    )


def _canonical_keys(keys: object) -> tuple[NodeKey, ...]:
    return tuple(sorted(keys, key=lambda key: (key[0], key[1], key[2])))


def _is_ancestor(ancestor: NodeKey, descendant: NodeKey) -> bool:
    level_delta = ancestor[0] - descendant[0]
    if level_delta <= 0:
        return False
    scale = 1 << level_delta
    return (
        descendant[1] // scale == ancestor[1]
        and descendant[2] // scale == ancestor[2]
    )


def frontier_partition_valid(mask: np.ndarray, keys: tuple[NodeKey, ...]) -> bool:
    """Return whether ``keys`` are an antichain that partitions every active mask pixel once."""

    if not isinstance(mask, np.ndarray) or mask.ndim != 2 or mask.dtype != np.bool_:
        raise ValueError("mask must be a 2D bool array")
    if len(set(keys)) != len(keys):
        return False
    for index, first in enumerate(keys):
        for second in keys[index + 1 :]:
            if _is_ancestor(first, second) or _is_ancestor(second, first):
                return False
    coverage = np.zeros(mask.shape, dtype=np.uint16)
    for key in keys:
        x0, y0, x1, y1 = _cell_bounds(key, mask.shape)
        if x0 >= x1 or y0 >= y1 or not mask[y0:y1, x0:x1].any():
            return False
        coverage[y0:y1, x0:x1] += mask[y0:y1, x0:x1]
    return bool(np.all(coverage[mask] == 1) and np.all(coverage[~mask] == 0))


def _field_for_frontier(
    mask: np.ndarray,
    keys: tuple[NodeKey, ...],
    colors_by_key: dict[NodeKey, np.ndarray],
    config: ArtifactFirstQuadtreeConfig,
) -> ObservationField2D:
    canonical = _canonical_keys(keys)
    means, log_scales, rotations = _geometry_rows(mask, canonical, config.leaf_scale_px)
    colors = np.stack([colors_by_key[key] for key in canonical], axis=0).astype(np.float32)
    return _field_from_arrays(means, log_scales, rotations, colors, mask, config)


def _cold_render(
    field: ObservationField2D,
    config: ArtifactFirstQuadtreeConfig,
) -> np.ndarray:
    from .pixel_contraction import render_observation_field

    return render_observation_field(
        field,
        device=config.device,
        renderer=config.renderer,
        render_chunk=config.render_chunk,
        apply_declared_alpha=False,
    ).astype(np.float32, copy=False)


def _render_key_block(
    mask: np.ndarray,
    keys: tuple[NodeKey, ...],
    colors_by_key: dict[NodeKey, np.ndarray],
    config: ArtifactFirstQuadtreeConfig,
) -> np.ndarray:
    if not keys:
        return np.zeros((*mask.shape, 3), dtype=np.float32)
    return _cold_render(_field_for_frontier(mask, keys, colors_by_key, config), config)


def _centered_patch_rmse_map(pixel_mse: np.ndarray, side: int = 7) -> np.ndarray:
    height, width = pixel_mse.shape
    effective = min(side, height, width)
    if effective % 2 == 0:
        effective -= 1
    effective = max(effective, 1)
    integral = np.pad(pixel_mse.astype(np.float64), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    sums = (
        integral[effective:, effective:]
        - integral[:-effective, effective:]
        - integral[effective:, :-effective]
        + integral[:-effective, :-effective]
    )
    result = np.zeros(pixel_mse.shape, dtype=np.float32)
    radius = effective // 2
    values = np.sqrt(np.maximum(sums, 0.0) / (effective * effective)).astype(np.float32)
    result[radius : height - radius, radius : width - radius] = values
    return result


def rank_frontier_splits(
    frontier: tuple[NodeKey, ...],
    blocked: set[NodeKey],
    mask: np.ndarray,
    reconstruction_raw: np.ndarray,
    target: np.ndarray,
    config: ArtifactFirstQuadtreeConfig,
) -> list[FrontierSplitCandidate]:
    """Rank splittable frontier nodes under the configured deterministic policy."""

    residual = np.asarray(reconstruction_raw, dtype=np.float64) - np.asarray(
        target, dtype=np.float64
    )
    pixel_mse = np.mean(residual * residual, axis=2)
    smoothed = _mask_aware_smoothed_error(
        pixel_mse,
        mask,
        config.error_smoothing_sigma_px,
    )
    pixel_rmse = np.sqrt(np.maximum(pixel_mse, 0.0))
    patch_rmse = _centered_patch_rmse_map(np.where(mask, pixel_mse, 0.0), 7)
    candidates: list[FrontierSplitCandidate] = []
    for key in frontier:
        if key in blocked or key[0] == 0:
            continue
        children = _child_keys(mask, key)
        if not children:
            continue
        x0, y0, x1, y1 = _cell_bounds(key, mask.shape)
        cell_mask = mask[y0:y1, x0:x1]
        pixel_max = float(np.max(pixel_rmse[y0:y1, x0:x1][cell_mask]))
        patch_max = float(np.max(patch_rmse[y0:y1, x0:x1]))
        artifact_score = max(
            pixel_max / config.pixel_rmse_threshold,
            patch_max / config.patch7_rmse_threshold,
        )
        net_rows = len(children) - 1
        energy_score = float(np.sum(smoothed[y0:y1, x0:x1][cell_mask])) / max(
            net_rows, 1
        )
        candidates.append(
            FrontierSplitCandidate(
                parent_key=key,
                child_keys=children,
                artifact_score=artifact_score,
                energy_score=energy_score,
                net_active_rows=net_rows,
            )
        )
    if config.selection_mode == "artifact_first":
        candidates.sort(
            key=lambda item: (
                -item.artifact_score,
                -item.energy_score,
                -item.parent_key[0],
                item.parent_key[1],
                item.parent_key[2],
            )
        )
    else:
        candidates.sort(
            key=lambda item: (
                -item.energy_score,
                -item.artifact_score,
                -item.parent_key[0],
                item.parent_key[1],
                item.parent_key[2],
            )
        )
    return candidates


def _support_box(
    mask: np.ndarray,
    key: NodeKey,
    leaf_scale_px: float,
    sigma_cutoff: float,
) -> tuple[float, float, float, float]:
    means, log_scales, rotations = _geometry_rows(mask, (key,), leaf_scale_px)
    scales_squared = np.exp(2.0 * log_scales[0].astype(np.float64))
    cosine = math.cos(float(rotations[0]))
    sine = math.sin(float(rotations[0]))
    var_x = cosine * cosine * scales_squared[0] + sine * sine * scales_squared[1]
    var_y = sine * sine * scales_squared[0] + cosine * cosine * scales_squared[1]
    radius_x = max(int(math.ceil(sigma_cutoff * math.sqrt(var_x))), 1)
    radius_y = max(int(math.ceil(sigma_cutoff * math.sqrt(var_y))), 1)
    return (
        float(means[0, 0] - radius_x),
        float(means[0, 1] - radius_y),
        float(means[0, 0] + radius_x),
        float(means[0, 1] + radius_y),
    )


def support_overlapping_keys(
    mask: np.ndarray,
    active_keys: tuple[NodeKey, ...],
    selected_parents: tuple[NodeKey, ...],
    *,
    leaf_scale_px: float,
    sigma_cutoff: float,
    margin_px: int,
) -> tuple[NodeKey, ...]:
    """Return surviving rows whose finite AABB intersects selected expanded supports."""

    margin = _integer(margin_px, "margin_px", minimum=0)
    parent_set = set(selected_parents)
    regions = []
    for key in selected_parents:
        left, top, right, bottom = _support_box(
            mask, key, leaf_scale_px, sigma_cutoff
        )
        regions.append(
            (left - margin, top - margin, right + margin, bottom + margin)
        )
    overlapping = []
    for key in active_keys:
        if key in parent_set:
            continue
        left, top, right, bottom = _support_box(
            mask, key, leaf_scale_px, sigma_cutoff
        )
        if any(
            left <= region_right
            and right >= region_left
            and top <= region_bottom
            and bottom >= region_top
            for region_left, region_top, region_right, region_bottom in regions
        ):
            overlapping.append(key)
    return _canonical_keys(overlapping)


def _weight_telemetry(weights: np.ndarray, raw_scores: np.ndarray) -> dict[str, float]:
    weights64 = np.asarray(weights, dtype=np.float64)
    scores64 = np.asarray(raw_scores, dtype=np.float64)
    weight_sum = float(np.sum(weights64))
    squared_sum = float(np.sum(weights64 * weights64))
    effective = weight_sum * weight_sum / squared_sum if squared_sum > 0.0 else 0.0
    return {
        "error_weight_min": float(np.min(weights64)),
        "error_weight_mean": float(np.mean(weights64)),
        "error_weight_p50": float(np.quantile(weights64, 0.5)),
        "error_weight_p90": float(np.quantile(weights64, 0.9)),
        "error_weight_max": float(np.max(weights64)),
        "error_weight_effective_rows": effective,
        "raw_error_score_mean": float(np.mean(scores64)),
        "raw_error_score_max": float(np.max(scores64)),
    }


def _row_error_update_weights(
    field,
    reference_raw: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    config: ArtifactFirstQuadtreeConfig,
) -> tuple[object, dict[str, float], float]:
    """Compute support-averaged smoothed-error multipliers with one renderer VJP."""

    import torch

    from .render import render_field

    started = time.perf_counter()
    residual = np.asarray(reference_raw, dtype=np.float64) - np.asarray(target, dtype=np.float64)
    residual_energy = np.mean(residual * residual, axis=2)
    smoothed = _mask_aware_smoothed_error(
        residual_energy,
        mask,
        config.error_smoothing_sigma_px,
    )
    error_tensor = torch.as_tensor(
        smoothed,
        device=field.means.device,
        dtype=field.means.dtype,
    )
    mask_tensor = torch.as_tensor(
        mask,
        device=field.means.device,
        dtype=field.means.dtype,
    )
    probe_colors = torch.zeros(
        (field.n, 3),
        device=field.means.device,
        dtype=field.means.dtype,
        requires_grad=True,
    )
    probe = render_field(
        field.means.detach(),
        field.conics().detach(),
        probe_colors,
        field.radii(config.sigma_cutoff),
        mask.shape[0],
        mask.shape[1],
        chunk=config.render_chunk,
        mode=config.renderer,
        opacities=None,
        scales=field.scales().detach(),
        rotations=field.rotations.detach(),
        support_fade=config.support_fade_alpha > 0.0,
        sigma_cutoff=config.sigma_cutoff,
        support_fade_alpha=config.support_fade_alpha,
    )
    probe_objective = torch.sum(probe[..., 0] * error_tensor)
    probe_objective = probe_objective + torch.sum(probe[..., 1] * mask_tensor)
    gradient = torch.autograd.grad(
        probe_objective,
        probe_colors,
        create_graph=False,
        retain_graph=False,
    )[0]
    numerator = gradient[:, 0].clamp_min(0.0)
    denominator = gradient[:, 1]
    raw_scores = torch.where(
        denominator > 1e-8,
        numerator / denominator.clamp_min(1e-8),
        torch.zeros_like(numerator),
    )
    raw_numpy = raw_scores.detach().cpu().numpy().astype(np.float64)
    weights_numpy = _normalized_error_update_weights(
        raw_numpy,
        power=config.error_weight_power,
        floor=config.error_weight_floor,
        ceiling=config.error_weight_ceiling,
    )
    weights = torch.as_tensor(
        weights_numpy,
        device=field.means.device,
        dtype=field.means.dtype,
    )
    return weights, _weight_telemetry(weights_numpy, raw_numpy), time.perf_counter() - started


def _unit_weight_telemetry(count: int) -> dict[str, float]:
    return {
        "error_weight_min": 1.0,
        "error_weight_mean": 1.0,
        "error_weight_p50": 1.0,
        "error_weight_p90": 1.0,
        "error_weight_max": 1.0,
        "error_weight_effective_rows": float(count),
        "raw_error_score_mean": 0.0,
        "raw_error_score_max": 0.0,
    }


def _optimize_color_block(
    *,
    means: np.ndarray,
    log_scales: np.ndarray,
    rotations: np.ndarray,
    initial_colors: np.ndarray,
    frozen_base: np.ndarray,
    reference_raw: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    config: ArtifactFirstQuadtreeConfig,
    steps: int,
    attempt_index: int,
    candidate_count: int,
    cumulative_step_offset: int,
    apply_error_weights: bool,
) -> _ColorOptimization:
    import torch
    import torch.nn.functional as torch_functional

    from .gaussians import GaussianField
    from .render import render_field

    started = time.perf_counter()
    device = torch.device(config.device)
    local = GaussianField.from_numpy(
        means,
        np.exp(log_scales),
        rotations,
        initial_colors,
        device=config.device,
    )
    local.colors.requires_grad_(True)
    optimizer = torch.optim.Adam([local.colors], lr=config.learning_rate)
    frozen_tensor = torch.as_tensor(
        np.ascontiguousarray(frozen_base), device=device, dtype=torch.float32
    )
    reference_tensor = torch.as_tensor(
        np.ascontiguousarray(reference_raw), device=device, dtype=torch.float32
    )
    target_tensor = torch.as_tensor(
        np.ascontiguousarray(target), device=device, dtype=torch.float32
    )
    mask_tensor = torch.as_tensor(mask, device=device, dtype=torch.bool)
    mask_float = mask_tensor.to(dtype=torch.float32)
    conics = local.conics()
    scales = local.scales()
    radii = local.radii(config.sigma_cutoff)
    patch_side = min(7, *mask.shape)
    if patch_side % 2 == 0:
        patch_side -= 1
    patch_side = max(patch_side, 1)
    if apply_error_weights:
        row_weights, telemetry, attribution_seconds = _row_error_update_weights(
            local,
            reference_raw,
            target,
            mask,
            config,
        )
    else:
        row_weights = torch.ones(
            local.n,
            device=local.means.device,
            dtype=local.means.dtype,
        )
        telemetry = _unit_weight_telemetry(local.n)
        attribution_seconds = 0.0

    def render_local():
        return render_field(
            local.means,
            conics,
            local.colors,
            radii,
            mask.shape[0],
            mask.shape[1],
            chunk=config.render_chunk,
            mode=config.renderer,
            opacities=None,
            scales=scales,
            rotations=local.rotations,
            support_fade=config.support_fade_alpha > 0.0,
            sigma_cutoff=config.sigma_cutoff,
            support_fade_alpha=config.support_fade_alpha,
        )

    def tensor_metrics(reconstruction):
        residual = reconstruction - target_tensor
        pixel_mse = torch.mean(residual.square(), dim=2)
        foreground = pixel_mse[mask_tensor]
        sse = torch.sum(residual[mask_tensor].square())
        pixel_max = torch.sqrt(torch.max(foreground))
        patch_mse = torch_functional.avg_pool2d(
            (pixel_mse * mask_float)[None, None],
            kernel_size=patch_side,
            stride=1,
            padding=0,
        )
        patch_max = torch.sqrt(torch.max(patch_mse))
        violation = torch.maximum(
            pixel_max / config.pixel_rmse_threshold,
            patch_max / config.patch7_rmse_threshold,
        )
        return sse, pixel_max, patch_max, violation, foreground

    with torch.no_grad():
        reference_values = tensor_metrics(reference_tensor)
    best_colors: np.ndarray | None = None
    best_reconstruction: np.ndarray | None = None
    best_step = -1
    best_metrics = (
        float(reference_values[0].item()),
        float(reference_values[1].item()),
        float(reference_values[2].item()),
        float(reference_values[3].item()),
    )
    checkpoints: list[FrontierCheckpoint] = []
    attempted_steps = 0
    for step in range(steps + 1):
        optimizer.zero_grad(set_to_none=True)
        reconstruction = frozen_tensor + render_local()
        sse, pixel_max, patch_max, violation, foreground = tensor_metrics(reconstruction)
        tail_count = max(1, int(math.ceil(config.tail_fraction * foreground.numel())))
        tail = torch.topk(foreground, k=tail_count, largest=True, sorted=False).values.mean()
        objective = foreground.mean() + config.tail_weight * tail
        if step % config.checkpoint_every == 0 or step == steps:
            candidate_violation = float(violation.detach().item())
            candidate_sse = float(sse.detach().item())
            checkpoints.append(
                FrontierCheckpoint(
                    attempt_index=attempt_index,
                    step=step,
                    cumulative_optimizer_step=cumulative_step_offset + step,
                    candidate_count=candidate_count,
                    optimized_count=local.n,
                    raw_sse=candidate_sse,
                    raw_pixel_rmse_max=float(pixel_max.detach().item()),
                    raw_patch7_rmse_max=float(patch_max.detach().item()),
                    raw_normalized_violation=candidate_violation,
                    objective=float(objective.detach().item()),
                    elapsed_seconds=time.perf_counter() - started,
                )
            )
            if _lexicographic_improves(
                candidate_violation,
                candidate_sse,
                best_metrics[3],
                best_metrics[0],
            ):
                best_step = step
                best_colors = local.colors.detach().cpu().numpy().copy()
                best_reconstruction = reconstruction.detach().cpu().numpy().copy()
                best_metrics = (
                    candidate_sse,
                    float(pixel_max.detach().item()),
                    float(patch_max.detach().item()),
                    candidate_violation,
                )
        if step == steps:
            break
        if not bool(torch.isfinite(objective)):
            break
        objective.backward()
        previous = local.colors.detach().clone()
        optimizer.step()
        with torch.no_grad():
            local.colors.copy_(
                previous + (local.colors - previous) * row_weights.reshape(-1, 1)
            )
        attempted_steps += 1
    return _ColorOptimization(
        colors=best_colors,
        reconstruction_raw=best_reconstruction,
        selected_step=best_step,
        selected_metrics=best_metrics,
        checkpoints=tuple(checkpoints),
        elapsed_seconds=time.perf_counter() - started,
        attribution_seconds=attribution_seconds,
        attempted_steps=attempted_steps,
        weight_telemetry=telemetry,
    )


def initialize_artifact_first_quadtree(
    image: np.ndarray,
    config: ArtifactFirstQuadtreeConfig,
    *,
    mask: np.ndarray | None = None,
) -> FrontierStartState:
    """Fit the common coarse base once for reuse by all factorial arms."""

    if not isinstance(config, ArtifactFirstQuadtreeConfig):
        raise TypeError("config must be ArtifactFirstQuadtreeConfig")
    source = _validated_image(image)
    active_mask = _validated_mask(mask, source.shape[:2])
    started = time.perf_counter()
    keys = _base_keys(active_mask, config.start_level)
    if not keys:
        raise RuntimeError("the start level produced no mask-present node")
    if len(keys) > config.max_gaussians:
        raise ValueError(
            f"base layer has {len(keys)} rows, exceeding max_gaussians={config.max_gaussians}"
        )
    means, log_scales, rotations = _geometry_rows(
        active_mask, keys, config.leaf_scale_px
    )
    initial_colors = np.stack(
        [_cell_rgb_mean(source, active_mask, key) for key in keys], axis=0
    )
    zero = np.zeros_like(source, dtype=np.float32)
    initial_metrics = progressive_artifact_metrics(
        zero,
        source,
        active_mask,
        pixel_threshold=config.pixel_rmse_threshold,
        patch7_threshold=config.patch7_rmse_threshold,
        displayed=False,
    )
    optimization = _optimize_color_block(
        means=means,
        log_scales=log_scales,
        rotations=rotations,
        initial_colors=initial_colors,
        frozen_base=zero,
        reference_raw=zero,
        target=source,
        mask=active_mask,
        config=config,
        steps=config.base_steps,
        attempt_index=0,
        candidate_count=len(keys),
        cumulative_step_offset=0,
        apply_error_weights=False,
    )
    if optimization.colors is None or optimization.reconstruction_raw is None:
        colors = np.zeros((len(keys), 3), dtype=np.float32)
        optimized_raw = zero
        selected_step = -1
    else:
        colors = optimization.colors.astype(np.float32, copy=True)
        optimized_raw = optimization.reconstruction_raw.astype(np.float32, copy=True)
        selected_step = optimization.selected_step
    colors_by_key = {key: colors[index].copy() for index, key in enumerate(keys)}
    field = _field_for_frontier(active_mask, keys, colors_by_key, config)
    cold_started = time.perf_counter()
    cold_raw = _cold_render(field, config)
    cold_seconds = time.perf_counter() - cold_started
    cold_metrics = progressive_artifact_metrics(
        cold_raw,
        source,
        active_mask,
        pixel_threshold=config.pixel_rmse_threshold,
        patch7_threshold=config.patch7_rmse_threshold,
        displayed=False,
    )
    display = progressive_artifact_metrics(
        cold_raw,
        source,
        active_mask,
        pixel_threshold=config.pixel_rmse_threshold,
        patch7_threshold=config.patch7_rmse_threshold,
        displayed=True,
    )
    improved = _lexicographic_improves(
        float(cold_metrics["normalized_violation"]),
        float(cold_metrics["sse"]),
        float(initial_metrics["normalized_violation"]),
        float(initial_metrics["sse"]),
    )
    elapsed = time.perf_counter() - started
    stage = FrontierStage(
        attempt_index=0,
        accepted_index=0 if improved else None,
        kind="base",
        status="accepted" if improved else "base_not_improved",
        selection_mode="shared_base",
        reconciliation_scope="shared_base",
        parent_keys=(),
        child_keys=keys,
        optimized_existing_keys=(),
        count_before=0,
        proposed_child_rows=len(keys),
        proposed_net_rows=len(keys),
        accepted_child_rows=len(keys),
        accepted_net_rows=len(keys),
        count_after=len(keys),
        stored_nodes_after=len(keys),
        inactive_nodes_after=0,
        coefficient_event_rows_after=len(keys),
        artifact_score_max=0.0,
        artifact_score_min=0.0,
        energy_score_max=0.0,
        energy_score_min=0.0,
        optimized_rows=len(keys),
        frozen_rows=0,
        selected_step=selected_step,
        attempted_optimizer_steps=optimization.attempted_steps,
        sse_before=float(initial_metrics["sse"]),
        sse_after=float(cold_metrics["sse"]),
        raw_violation_before=float(initial_metrics["normalized_violation"]),
        raw_violation_after=float(cold_metrics["normalized_violation"]),
        display_pixel_rmse_max=float(display["pixel_rmse_max"]),
        display_patch7_rmse_max=float(display["patch7_rmse_max"]),
        display_gate_pass=bool(display["gate_pass"]),
        error_weight_min=1.0,
        error_weight_mean=1.0,
        error_weight_p50=1.0,
        error_weight_p90=1.0,
        error_weight_max=1.0,
        error_weight_effective_rows=float(len(keys)),
        raw_error_score_mean=0.0,
        raw_error_score_max=0.0,
        untouched_coefficients_bit_exact=True,
        rollback_bit_exact=True,
        frontier_partition_valid=frontier_partition_valid(active_mask, keys),
        optimized_vs_cold_max_abs=float(np.max(np.abs(cold_raw - optimized_raw))),
        selection_seconds=0.0,
        attribution_seconds=0.0,
        optimization_seconds=optimization.elapsed_seconds,
        cold_render_seconds=cold_seconds,
        cumulative_elapsed_seconds=elapsed,
    )
    return FrontierStartState(
        source_sha256=_array_sha256(source),
        mask_sha256=_array_sha256(active_mask),
        base_signature=_base_signature(config),
        active_keys=keys,
        colors=colors,
        field=field,
        reconstruction_raw=cold_raw.copy(),
        stage=stage,
        checkpoints=optimization.checkpoints,
        initial_sse=float(initial_metrics["sse"]),
        elapsed_seconds=elapsed,
        cumulative_optimizer_steps=optimization.attempted_steps,
        improved=improved,
    )


def _validated_start_state(
    state: FrontierStartState,
    source: np.ndarray,
    mask: np.ndarray,
    config: ArtifactFirstQuadtreeConfig,
) -> None:
    if not isinstance(state, FrontierStartState):
        raise TypeError("start_state must be FrontierStartState")
    if state.source_sha256 != _array_sha256(source):
        raise ValueError("start_state was fitted to a different executed image")
    if state.mask_sha256 != _array_sha256(mask):
        raise ValueError("start_state was fitted to a different mask")
    if state.base_signature != _base_signature(config):
        raise ValueError("start_state base configuration differs from config")
    if state.field.n != len(state.active_keys) or state.colors.shape != (state.field.n, 3):
        raise ValueError("start_state row metadata is inconsistent")
    if state.field.n > config.max_gaussians:
        raise ValueError("start_state exceeds config.max_gaussians")
    if not frontier_partition_valid(mask, state.active_keys):
        raise ValueError("start_state active keys do not partition the mask")


def _select_batch(
    candidates: list[FrontierSplitCandidate],
    *,
    active_count: int,
    config: ArtifactFirstQuadtreeConfig,
) -> list[FrontierSplitCandidate]:
    selected = []
    child_rows = 0
    net_rows = 0
    remaining = config.max_gaussians - active_count
    for candidate in candidates:
        proposed_children = len(candidate.child_keys)
        proposed_net = candidate.net_active_rows
        if child_rows + proposed_children > config.max_child_rows_per_stage:
            continue
        if net_rows + proposed_net > remaining:
            continue
        selected.append(candidate)
        child_rows += proposed_children
        net_rows += proposed_net
        if child_rows == config.max_child_rows_per_stage or net_rows == remaining:
            break
    return selected


def _snapshot(
    label: str,
    attempt_index: int,
    active_keys: tuple[NodeKey, ...],
    field: ObservationField2D,
    reconstruction_raw: np.ndarray,
    stored_node_count: int,
    coefficient_event_rows: int,
) -> FrontierSnapshot:
    return FrontierSnapshot(
        label=label,
        attempt_index=attempt_index,
        active_keys=active_keys,
        field=field,
        reconstruction_raw=np.array(reconstruction_raw, dtype=np.float32, copy=True),
        stored_node_count=stored_node_count,
        coefficient_event_rows=coefficient_event_rows,
    )


def _deduplicated_snapshots(
    base: FrontierSnapshot,
    milestones: dict[int, FrontierSnapshot],
    passing: FrontierSnapshot | None,
    terminal: FrontierSnapshot,
) -> tuple[FrontierSnapshot, ...]:
    ordered = [base]
    ordered.extend(milestones[count] for count in sorted(milestones))
    if passing is not None:
        ordered.append(passing)
    intermediate = ordered
    deduplicated: list[FrontierSnapshot] = []
    seen: set[str] = set()
    for item in intermediate:
        digest = item.field.canonical_hash()
        if digest in seen:
            continue
        seen.add(digest)
        deduplicated.append(item)
    # A terminal record remains explicit even when every split rolled back and its field equals
    # the base.  The duplicate bytes are intentional: the terminal history includes failed work.
    deduplicated.append(terminal)
    return tuple(deduplicated)


def build_artifact_first_quadtree(
    image: np.ndarray,
    config: ArtifactFirstQuadtreeConfig,
    *,
    mask: np.ndarray | None = None,
    start_state: FrontierStartState | None = None,
) -> ArtifactFirstQuadtreeResult:
    """Build a parent-replacing active frontier with transactional local RGB reconciliation."""

    if not isinstance(config, ArtifactFirstQuadtreeConfig):
        raise TypeError("config must be ArtifactFirstQuadtreeConfig")
    source = _validated_image(image)
    active_mask = _validated_mask(mask, source.shape[:2])
    state = (
        initialize_artifact_first_quadtree(source, config, mask=active_mask)
        if start_state is None
        else start_state
    )
    _validated_start_state(state, source, active_mask, config)
    arm_started = time.perf_counter()
    active_keys = state.active_keys
    colors_by_key = {
        key: state.colors[index].copy() for index, key in enumerate(state.active_keys)
    }
    field = state.field
    current_raw = np.array(state.reconstruction_raw, dtype=np.float32, copy=True)
    stages: list[FrontierStage] = [state.stage]
    checkpoints: list[FrontierCheckpoint] = list(state.checkpoints)
    stored_keys = set(active_keys)
    blocked: set[NodeKey] = set()
    coefficient_event_rows = len(active_keys)
    cumulative_steps = state.cumulative_optimizer_steps
    accepted_split_count = 0
    accepted_stage_count = 0
    accepted_index = 1
    attempt_index = 1
    maintained_parity = state.stage.optimized_vs_cold_max_abs
    display_metrics = progressive_artifact_metrics(
        current_raw,
        source,
        active_mask,
        pixel_threshold=config.pixel_rmse_threshold,
        patch7_threshold=config.patch7_rmse_threshold,
        displayed=True,
    )
    base_snapshot = _snapshot(
        "base",
        0,
        active_keys,
        field,
        current_raw,
        len(stored_keys),
        coefficient_event_rows,
    )
    milestone_snapshots: dict[int, FrontierSnapshot] = {}
    for milestone in config.milestone_counts:
        if field.n <= milestone:
            milestone_snapshots[milestone] = _snapshot(
                f"milestone_{milestone}",
                0,
                active_keys,
                field,
                current_raw,
                len(stored_keys),
                coefficient_event_rows,
            )
    passing_snapshot = base_snapshot if display_metrics["gate_pass"] else None
    stop_reason = "display_gate_passed" if display_metrics["gate_pass"] else "frontier_exhausted"
    if not state.improved:
        stop_reason = "base_not_improved"

    while state.improved and not display_metrics["gate_pass"]:
        if config.max_stages is not None and accepted_stage_count >= config.max_stages:
            stop_reason = "max_stages_reached"
            break
        selection_started = time.perf_counter()
        candidates = rank_frontier_splits(
            active_keys,
            blocked,
            active_mask,
            current_raw,
            source,
            config,
        )
        selected = _select_batch(candidates, active_count=field.n, config=config)
        selection_seconds = time.perf_counter() - selection_started
        if not selected:
            if not candidates:
                stop_reason = "frontier_exhausted"
            elif field.n >= config.max_gaussians:
                stop_reason = "max_gaussians_reached"
            else:
                stop_reason = "no_split_fits_allowance"
            break

        retry = selected
        accepted_in_retry = False
        while retry:
            parent_keys = tuple(item.parent_key for item in retry)
            child_keys = tuple(child for item in retry for child in item.child_keys)
            parent_set = set(parent_keys)
            old_active_keys = active_keys
            old_field = field
            old_field_hash = field.canonical_hash()
            old_raw = current_raw
            old_colors = {key: value.copy() for key, value in colors_by_key.items()}
            before_metrics = progressive_artifact_metrics(
                current_raw,
                source,
                active_mask,
                pixel_threshold=config.pixel_rmse_threshold,
                patch7_threshold=config.patch7_rmse_threshold,
                displayed=False,
            )
            if config.reconciliation_scope == "overlap":
                neighbors = support_overlapping_keys(
                    active_mask,
                    active_keys,
                    parent_keys,
                    leaf_scale_px=config.leaf_scale_px,
                    sigma_cutoff=config.sigma_cutoff,
                    margin_px=config.overlap_margin_px,
                )
            else:
                neighbors = ()
            parent_render = _render_key_block(
                active_mask,
                parent_keys,
                colors_by_key,
                config,
            )
            neighbor_render = _render_key_block(
                active_mask,
                neighbors,
                colors_by_key,
                config,
            )
            frozen_base = current_raw - parent_render - neighbor_render
            residual_without_parent = source - (current_raw - parent_render)
            proposed_colors = {
                key: value.copy()
                for key, value in colors_by_key.items()
                if key not in parent_set
            }
            for child in child_keys:
                proposed_colors[child] = _cell_rgb_mean(
                    residual_without_parent,
                    active_mask,
                    child,
                )
            candidate_keys = _canonical_keys(proposed_colors)
            local_keys = _canonical_keys((*neighbors, *child_keys))
            local_means, local_log_scales, local_rotations = _geometry_rows(
                active_mask,
                local_keys,
                config.leaf_scale_px,
            )
            local_colors = np.stack(
                [proposed_colors[key] for key in local_keys], axis=0
            ).astype(np.float32)
            candidate_count = len(candidate_keys)
            optimization = _optimize_color_block(
                means=local_means,
                log_scales=local_log_scales,
                rotations=local_rotations,
                initial_colors=local_colors,
                frozen_base=frozen_base,
                reference_raw=current_raw,
                target=source,
                mask=active_mask,
                config=config,
                steps=config.layer_steps,
                attempt_index=attempt_index,
                candidate_count=candidate_count,
                cumulative_step_offset=cumulative_steps,
                apply_error_weights=True,
            )
            cumulative_steps += optimization.attempted_steps
            checkpoints.extend(optimization.checkpoints)
            accepted = False
            candidate_metrics = before_metrics
            candidate_display = display_metrics
            candidate_parity = 0.0
            cold_seconds = 0.0
            candidate_field: ObservationField2D | None = None
            candidate_raw: np.ndarray | None = None
            untouched_exact = True
            if optimization.colors is not None:
                for index, key in enumerate(local_keys):
                    proposed_colors[key] = optimization.colors[index].astype(
                        np.float32, copy=True
                    )
                untouched_keys = set(candidate_keys) - set(local_keys)
                untouched_exact = all(
                    np.array_equal(old_colors[key], proposed_colors[key])
                    for key in untouched_keys
                )
                candidate_field = _field_for_frontier(
                    active_mask,
                    candidate_keys,
                    proposed_colors,
                    config,
                )
                cold_started = time.perf_counter()
                candidate_raw = _cold_render(candidate_field, config)
                cold_seconds = time.perf_counter() - cold_started
                if optimization.reconstruction_raw is not None:
                    candidate_parity = float(
                        np.max(
                            np.abs(
                                candidate_raw
                                - optimization.reconstruction_raw.astype(np.float32)
                            )
                        )
                    )
                candidate_metrics = progressive_artifact_metrics(
                    candidate_raw,
                    source,
                    active_mask,
                    pixel_threshold=config.pixel_rmse_threshold,
                    patch7_threshold=config.patch7_rmse_threshold,
                    displayed=False,
                )
                accepted = _lexicographic_improves(
                    float(candidate_metrics["normalized_violation"]),
                    float(candidate_metrics["sse"]),
                    float(before_metrics["normalized_violation"]),
                    float(before_metrics["sse"]),
                )
                if accepted:
                    candidate_display = progressive_artifact_metrics(
                        candidate_raw,
                        source,
                        active_mask,
                        pixel_threshold=config.pixel_rmse_threshold,
                        patch7_threshold=config.patch7_rmse_threshold,
                        displayed=True,
                    )

            proposed_net_rows = len(child_keys) - len(parent_keys)
            if accepted:
                if candidate_field is None or candidate_raw is None:
                    raise RuntimeError("accepted candidate is missing its cold field")
                active_keys = candidate_keys
                colors_by_key = proposed_colors
                field = candidate_field
                current_raw = candidate_raw
                display_metrics = candidate_display
                stored_keys.update(child_keys)
                coefficient_event_rows += len(child_keys) + len(neighbors)
                accepted_split_count += len(parent_keys)
                accepted_stage_count += 1
                maintained_parity = max(maintained_parity, candidate_parity)
                partition_valid = frontier_partition_valid(active_mask, active_keys)
                if not partition_valid:
                    raise RuntimeError("accepted active frontier no longer partitions the mask")
                rollback_exact = True
                status = "accepted"
                accepted_in_retry = True
            else:
                partition_valid = frontier_partition_valid(active_mask, old_active_keys)
                rollback_exact = (
                    active_keys == old_active_keys
                    and field is old_field
                    and field.canonical_hash() == old_field_hash
                    and np.array_equal(current_raw, old_raw)
                    and set(colors_by_key) == set(old_colors)
                    and all(
                        np.array_equal(colors_by_key[key], old_colors[key])
                        for key in old_colors
                    )
                )
                status = "rolled_back_backoff" if len(retry) > 1 else "rolled_back_blocked"
                candidate_metrics = before_metrics
                candidate_display = display_metrics

            weights = optimization.weight_telemetry
            stages.append(
                FrontierStage(
                    attempt_index=attempt_index,
                    accepted_index=accepted_index if accepted else None,
                    kind="parent_replacement",
                    status=status,
                    selection_mode=config.selection_mode,
                    reconciliation_scope=config.reconciliation_scope,
                    parent_keys=parent_keys,
                    child_keys=child_keys,
                    optimized_existing_keys=neighbors,
                    count_before=len(old_active_keys),
                    proposed_child_rows=len(child_keys),
                    proposed_net_rows=proposed_net_rows,
                    accepted_child_rows=len(child_keys) if accepted else 0,
                    accepted_net_rows=proposed_net_rows if accepted else 0,
                    count_after=field.n,
                    stored_nodes_after=len(stored_keys),
                    inactive_nodes_after=len(stored_keys) - field.n,
                    coefficient_event_rows_after=coefficient_event_rows,
                    artifact_score_max=max(item.artifact_score for item in retry),
                    artifact_score_min=min(item.artifact_score for item in retry),
                    energy_score_max=max(item.energy_score for item in retry),
                    energy_score_min=min(item.energy_score for item in retry),
                    optimized_rows=len(local_keys),
                    frozen_rows=max(candidate_count - len(local_keys), 0),
                    selected_step=optimization.selected_step,
                    attempted_optimizer_steps=optimization.attempted_steps,
                    sse_before=float(before_metrics["sse"]),
                    sse_after=float(candidate_metrics["sse"]),
                    raw_violation_before=float(before_metrics["normalized_violation"]),
                    raw_violation_after=float(candidate_metrics["normalized_violation"]),
                    display_pixel_rmse_max=float(candidate_display["pixel_rmse_max"]),
                    display_patch7_rmse_max=float(candidate_display["patch7_rmse_max"]),
                    display_gate_pass=bool(candidate_display["gate_pass"]),
                    error_weight_min=weights["error_weight_min"],
                    error_weight_mean=weights["error_weight_mean"],
                    error_weight_p50=weights["error_weight_p50"],
                    error_weight_p90=weights["error_weight_p90"],
                    error_weight_max=weights["error_weight_max"],
                    error_weight_effective_rows=weights[
                        "error_weight_effective_rows"
                    ],
                    raw_error_score_mean=weights["raw_error_score_mean"],
                    raw_error_score_max=weights["raw_error_score_max"],
                    untouched_coefficients_bit_exact=untouched_exact,
                    rollback_bit_exact=rollback_exact,
                    frontier_partition_valid=partition_valid,
                    optimized_vs_cold_max_abs=candidate_parity,
                    selection_seconds=selection_seconds,
                    attribution_seconds=optimization.attribution_seconds,
                    optimization_seconds=optimization.elapsed_seconds,
                    cold_render_seconds=cold_seconds,
                    cumulative_elapsed_seconds=(
                        state.elapsed_seconds + time.perf_counter() - arm_started
                    ),
                )
            )
            selection_seconds = 0.0
            if accepted:
                accepted_index += 1
                for milestone in config.milestone_counts:
                    if field.n <= milestone:
                        milestone_snapshots[milestone] = _snapshot(
                            f"milestone_{milestone}",
                            attempt_index,
                            active_keys,
                            field,
                            current_raw,
                            len(stored_keys),
                            coefficient_event_rows,
                        )
                if display_metrics["gate_pass"] and passing_snapshot is None:
                    passing_snapshot = _snapshot(
                        "first_passing",
                        attempt_index,
                        active_keys,
                        field,
                        current_raw,
                        len(stored_keys),
                        coefficient_event_rows,
                    )
                attempt_index += 1
                break
            if len(retry) == 1:
                blocked.add(retry[0].parent_key)
                attempt_index += 1
                break
            retry = retry[: max(1, len(retry) // 2)]
            attempt_index += 1

        if accepted_in_retry:
            if display_metrics["gate_pass"]:
                stop_reason = "display_gate_passed"
                break
            continue
        if len(blocked) >= len([key for key in active_keys if key[0] > 0]):
            stop_reason = "no_progress"
            break

    final_metrics = progressive_artifact_metrics(
        current_raw,
        source,
        active_mask,
        pixel_threshold=config.pixel_rmse_threshold,
        patch7_threshold=config.patch7_rmse_threshold,
        displayed=False,
    )
    if display_metrics["gate_pass"]:
        stop_reason = "display_gate_passed"
    terminal = _snapshot(
        "terminal",
        stages[-1].attempt_index,
        active_keys,
        field,
        current_raw,
        len(stored_keys),
        coefficient_event_rows,
    )
    snapshots = _deduplicated_snapshots(
        base_snapshot,
        milestone_snapshots,
        passing_snapshot,
        terminal,
    )
    reconstruction = np.asarray(current_raw, dtype=np.float32)
    if field.packed_alpha is not None and field.semantics.alpha.matting_mode == "multiply_alpha":
        reconstruction = reconstruction * active_mask[:, :, None]
    alpha_bytes = 0 if field.packed_alpha is None else int(field.packed_alpha.nbytes)
    active_coefficient_bytes = field.n * 3 * np.dtype(np.float32).itemsize
    tree_bits = len(stored_keys)
    final_frontier_proxy = active_coefficient_bytes + (tree_bits + 7) // 8 + alpha_bytes
    event_coefficient_bytes = coefficient_event_rows * 3 * np.dtype(np.float32).itemsize
    progressive_proxy = event_coefficient_bytes + (tree_bits + 7) // 8 + alpha_bytes
    canonical_raw_bytes = int(sum(array.nbytes for array in field._array_items().values()))
    arm_elapsed = time.perf_counter() - arm_started
    return ArtifactFirstQuadtreeResult(
        field=field,
        reconstruction_raw=np.asarray(current_raw, dtype=np.float32),
        reconstruction=np.asarray(reconstruction, dtype=np.float32),
        active_keys=active_keys,
        stored_keys=_canonical_keys(stored_keys),
        stages=tuple(stages),
        checkpoints=tuple(checkpoints),
        snapshots=snapshots,
        base_count=len(state.active_keys),
        final_count=field.n,
        stored_node_count=len(stored_keys),
        inactive_node_count=len(stored_keys) - field.n,
        accepted_split_count=accepted_split_count,
        accepted_stage_count=accepted_stage_count,
        coefficient_event_rows=coefficient_event_rows,
        initial_sse=state.initial_sse,
        final_sse=float(final_metrics["sse"]),
        stop_reason=stop_reason,
        shared_base_seconds=state.elapsed_seconds,
        arm_elapsed_seconds=arm_elapsed,
        elapsed_seconds=state.elapsed_seconds + arm_elapsed,
        estimated_active_field_bytes=field.n * config.estimated_row_bytes + alpha_bytes,
        canonical_active_raw_bytes=canonical_raw_bytes,
        active_coefficient_proxy_bytes=active_coefficient_bytes,
        tree_proxy_bits=tree_bits,
        final_frontier_proxy_bytes=final_frontier_proxy,
        progressive_event_coefficient_proxy_bytes=event_coefficient_bytes,
        progressive_event_proxy_bytes=progressive_proxy,
        maintained_render_parity_max_abs=maintained_parity,
    )
