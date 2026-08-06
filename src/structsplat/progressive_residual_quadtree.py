"""Parent-preserving progressive residual quadtree for direct additive fields (HIER-006).

The module is a default-off NumPy-first research reference.  A coarse mask-moment quadtree layer
is fitted first.  High-error frontier cells retain their parent and append deterministic child
geometry whose signed RGB coefficients fit the residual against an immutable prefix.  Torch is
imported lazily only inside the coefficient optimizer and maintained-render bridge.

The structured byte count emitted here is deliberately a proxy: it assumes mask/tree-derived
geometry and float32 RGB coefficients, but it is not a self-contained coded stream.  COMP-013 and
FIT-030 remain authoritative for actual rate.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Literal

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
from .pixel_contraction import _mask_aware_smoothed_error


HierarchyRenderer = Literal["additive", "cuda_additive", "cuda_tiled_additive"]
NodeKey = tuple[int, int, int]


def _integer(value: object, name: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {result}")
    return result


def _finite(value: object, name: str, *, minimum: float, strict: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise TypeError(f"{name} must be a finite number")
    result = float(value)
    invalid = result <= minimum if strict else result < minimum
    if not math.isfinite(result) or invalid:
        relation = ">" if strict else ">="
        raise ValueError(f"{name} must be finite and {relation} {minimum}, got {result}")
    return result


def _lexicographic_improves(
    candidate_violation: float,
    candidate_sse: float,
    reference_violation: float,
    reference_sse: float,
) -> bool:
    """Compare artifact violation then SSE with a float32 roundoff tie band.

    Optimizer checkpoints and cold maintained renders pass through different reduction kernels.
    Treating their last-bit noise as a real local-artifact regression can roll back a layer whose
    mathematical maximum is unchanged.  The 32-ULP-scale band is far below one displayed 8-bit
    step and only activates the SSE tie-break inside that numerical equivalence class.
    """

    epsilon = float(np.finfo(np.float32).eps)
    violation_tolerance = 32.0 * epsilon * max(
        1.0, abs(candidate_violation), abs(reference_violation)
    )
    if candidate_violation < reference_violation - violation_tolerance:
        return True
    if candidate_violation > reference_violation + violation_tolerance:
        return False
    sse_tolerance = 32.0 * epsilon * max(1.0, abs(candidate_sse), abs(reference_sse))
    return candidate_sse < reference_sse - sse_tolerance


@dataclass(frozen=True)
class ProgressiveResidualConfig:
    """Configuration for :func:`build_progressive_residual_quadtree`.

    ``estimated_structured_proxy_bytes`` in the result is not an actual stream size.  It assumes
    float32 RGB coefficients, one split/leaf decision bit per retained node, and shared mask/config
    side information while omitting a real header, entropy model, and cold decoder.
    """

    start_level: int = 6
    max_gaussians: int = 8192
    leaf_scale_px: float = 0.18
    sigma_cutoff: float = 3.0
    support_fade_alpha: float = 0.0
    error_smoothing_sigma_px: float = 1.5
    max_rows_per_stage: int = 256
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
        object.__setattr__(self, "start_level", _integer(self.start_level, "start_level", minimum=0))
        for name in (
            "max_gaussians",
            "max_rows_per_stage",
            "base_steps",
            "layer_steps",
            "checkpoint_every",
            "estimated_row_bytes",
            "render_chunk",
        ):
            object.__setattr__(self, name, _integer(getattr(self, name), name, minimum=1))
        if self.max_stages is not None:
            object.__setattr__(self, "max_stages", _integer(self.max_stages, "max_stages", minimum=1))
        for name in (
            "leaf_scale_px",
            "sigma_cutoff",
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
        tail_fraction = _finite(self.tail_fraction, "tail_fraction", minimum=0.0, strict=True)
        if tail_fraction > 1.0:
            raise ValueError("tail_fraction must be <= 1")
        object.__setattr__(self, "tail_fraction", tail_fraction)
        if not isinstance(self.device, str) or not self.device.strip():
            raise ValueError("device must be a non-empty torch device string")
        if self.renderer not in ("additive", "cuda_additive", "cuda_tiled_additive"):
            raise ValueError("unsupported renderer")
        if self.renderer.startswith("cuda") and not self.device.startswith("cuda"):
            raise ValueError("a CUDA renderer requires a CUDA device")
        if not isinstance(self.milestone_counts, tuple):
            raise TypeError("milestone_counts must be a tuple of positive integers")
        validated_milestones = []
        for value in self.milestone_counts:
            milestone = _integer(value, "milestone_counts item", minimum=1)
            if milestone <= self.max_gaussians:
                validated_milestones.append(milestone)
        milestones = tuple(sorted(set(validated_milestones)))
        object.__setattr__(self, "milestone_counts", milestones)


@dataclass(frozen=True)
class HierarchyCheckpoint:
    """One evaluated coefficient checkpoint for a proposed hierarchy stage."""

    attempt_index: int
    step: int
    cumulative_optimizer_step: int
    candidate_count: int
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
            "raw_sse": self.raw_sse,
            "raw_pixel_rmse_max": self.raw_pixel_rmse_max,
            "raw_patch7_rmse_max": self.raw_patch7_rmse_max,
            "raw_normalized_violation": self.raw_normalized_violation,
            "objective": self.objective,
            "elapsed_seconds": self.elapsed_seconds,
        }


@dataclass(frozen=True)
class HierarchyStage:
    """One accepted or rolled-back coarse/base or child-refinement stage."""

    attempt_index: int
    accepted_index: int | None
    kind: str
    status: str
    parent_keys: tuple[NodeKey, ...]
    child_keys: tuple[NodeKey, ...]
    count_before: int
    proposed_rows: int
    accepted_rows: int
    count_after: int
    selection_score_max: float
    selection_score_min: float
    selected_step: int
    sse_before: float
    sse_after: float
    raw_violation_before: float
    raw_violation_after: float
    display_pixel_rmse_max: float
    display_patch7_rmse_max: float
    display_gate_pass: bool
    prefix_bit_exact: bool
    accumulated_vs_cold_max_abs: float
    selection_seconds: float
    optimization_seconds: float
    cold_render_seconds: float
    cumulative_elapsed_seconds: float

    def to_record(self) -> dict[str, object]:
        return {
            "attempt_index": self.attempt_index,
            "accepted_index": self.accepted_index,
            "kind": self.kind,
            "status": self.status,
            "parent_keys": [list(key) for key in self.parent_keys],
            "child_keys": [list(key) for key in self.child_keys],
            "count_before": self.count_before,
            "proposed_rows": self.proposed_rows,
            "accepted_rows": self.accepted_rows,
            "count_after": self.count_after,
            "selection_score_max": self.selection_score_max,
            "selection_score_min": self.selection_score_min,
            "selected_step": self.selected_step,
            "sse_before": self.sse_before,
            "sse_after": self.sse_after,
            "raw_violation_before": self.raw_violation_before,
            "raw_violation_after": self.raw_violation_after,
            "display_pixel_rmse_max": self.display_pixel_rmse_max,
            "display_patch7_rmse_max": self.display_patch7_rmse_max,
            "display_gate_pass": self.display_gate_pass,
            "prefix_bit_exact": self.prefix_bit_exact,
            "accumulated_vs_cold_max_abs": self.accumulated_vs_cold_max_abs,
            "selection_seconds": self.selection_seconds,
            "optimization_seconds": self.optimization_seconds,
            "cold_render_seconds": self.cold_render_seconds,
            "cumulative_elapsed_seconds": self.cumulative_elapsed_seconds,
        }


@dataclass(frozen=True)
class ProgressiveResidualResult:
    """Result and trajectory of a parent-preserving residual quadtree build."""

    field: ObservationField2D
    reconstruction_raw: np.ndarray
    reconstruction: np.ndarray
    stages: tuple[HierarchyStage, ...]
    checkpoints: tuple[HierarchyCheckpoint, ...]
    base_count: int
    final_count: int
    accepted_split_count: int
    initial_sse: float
    final_sse: float
    stop_reason: str
    elapsed_seconds: float
    estimated_field_bytes: int
    canonical_raw_bytes: int
    coefficient_proxy_bytes: int
    tree_proxy_bits: int
    estimated_structured_proxy_bytes: int
    prefix_bit_exact: bool
    maintained_render_parity_max_abs: float
    snapshot_counts: tuple[int, ...]

    def stage_records(self) -> list[dict[str, object]]:
        return [stage.to_record() for stage in self.stages]

    def checkpoint_records(self) -> list[dict[str, object]]:
        return [checkpoint.to_record() for checkpoint in self.checkpoints]


@dataclass(frozen=True)
class _LayerOptimization:
    colors: np.ndarray | None
    residual_render: np.ndarray | None
    selected_step: int
    selected_metrics: tuple[float, float, float, float]
    checkpoints: tuple[HierarchyCheckpoint, ...]
    elapsed_seconds: float
    attempted_steps: int


def _validated_image(image: object) -> np.ndarray:
    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a NumPy array")
    if image.ndim != 3 or image.shape[2] != 3 or image.shape[0] < 1 or image.shape[1] < 1:
        raise ValueError("image must have non-empty HWC shape (H, W, 3)")
    if image.dtype.kind not in "fiu" or not np.isfinite(image).all():
        raise ValueError("image must contain finite numeric values")
    result = np.asarray(image, dtype=np.float32)
    if (result < 0.0).any() or (result > 1.0).any():
        raise ValueError("image values must lie in [0, 1]")
    return np.array(result, dtype=np.float32, order="C", copy=True)


def _validated_mask(mask: object | None, shape: tuple[int, int]) -> np.ndarray:
    if mask is None:
        return np.ones(shape, dtype=bool)
    if not isinstance(mask, np.ndarray) or mask.shape != shape or mask.dtype != np.bool_:
        raise ValueError(f"mask must be a bool NumPy array with shape {shape}")
    result = np.array(mask, dtype=bool, order="C", copy=True)
    if not result.any():
        raise ValueError("mask must contain at least one active pixel")
    return result


def _cell_bounds(key: NodeKey, shape: tuple[int, int]) -> tuple[int, int, int, int]:
    level, cell_y, cell_x = key
    height, width = shape
    side = 1 << level
    x0 = cell_x * side
    y0 = cell_y * side
    return x0, y0, min(x0 + side, width), min(y0 + side, height)


def _node_present(mask: np.ndarray, key: NodeKey) -> bool:
    x0, y0, x1, y1 = _cell_bounds(key, mask.shape)
    return x0 < x1 and y0 < y1 and bool(mask[y0:y1, x0:x1].any())


def _base_keys(mask: np.ndarray, level: int) -> tuple[NodeKey, ...]:
    side = 1 << level
    height, width = mask.shape
    keys = []
    for cell_y in range((height + side - 1) // side):
        for cell_x in range((width + side - 1) // side):
            key = (level, cell_y, cell_x)
            if _node_present(mask, key):
                keys.append(key)
    return tuple(keys)


def _child_keys(mask: np.ndarray, key: NodeKey) -> tuple[NodeKey, ...]:
    level, cell_y, cell_x = key
    if level == 0:
        return ()
    children = []
    for offset_y in (0, 1):
        for offset_x in (0, 1):
            child = (level - 1, 2 * cell_y + offset_y, 2 * cell_x + offset_x)
            if _node_present(mask, child):
                children.append(child)
    return tuple(children)


def _node_geometry(
    mask: np.ndarray,
    key: NodeKey,
    leaf_scale_px: float,
) -> tuple[np.ndarray, np.ndarray, np.float32]:
    """Return deterministic mask-moment mean, log scales, and canonical RS angle."""

    x0, y0, x1, y1 = _cell_bounds(key, mask.shape)
    local_y, local_x = np.nonzero(mask[y0:y1, x0:x1])
    if local_x.size == 0:
        raise ValueError(f"quadtree node {key} contains no active pixel")
    points = np.stack([local_x + x0, local_y + y0], axis=1).astype(np.float64)
    mean = points.mean(axis=0)
    centered = points - mean[None, :]
    covariance = centered.T @ centered / points.shape[0]
    covariance += np.eye(2, dtype=np.float64) * leaf_scale_px**2
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    eigenvalues = np.maximum(eigenvalues, leaf_scale_px**2)
    first_axis = eigenvectors[:, 0]
    angle = math.atan2(float(first_axis[1]), float(first_axis[0]))
    angle = (angle + math.pi / 2.0) % math.pi - math.pi / 2.0
    return (
        mean.astype(np.float32),
        (0.5 * np.log(eigenvalues)).astype(np.float32),
        np.float32(angle),
    )


def _geometry_rows(
    mask: np.ndarray,
    keys: tuple[NodeKey, ...],
    leaf_scale_px: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    means = np.empty((len(keys), 2), dtype=np.float32)
    log_scales = np.empty((len(keys), 2), dtype=np.float32)
    rotations = np.empty(len(keys), dtype=np.float32)
    for index, key in enumerate(keys):
        means[index], log_scales[index], rotations[index] = _node_geometry(
            mask, key, leaf_scale_px
        )
    return means, log_scales, rotations


def _cell_rgb_mean(values: np.ndarray, mask: np.ndarray, key: NodeKey) -> np.ndarray:
    x0, y0, x1, y1 = _cell_bounds(key, mask.shape)
    cell_mask = mask[y0:y1, x0:x1]
    return np.mean(values[y0:y1, x0:x1][cell_mask], axis=0, dtype=np.float64).astype(np.float32)


def _field_from_arrays(
    means: np.ndarray,
    log_scales: np.ndarray,
    rotations: np.ndarray,
    coefficients: np.ndarray,
    mask: np.ndarray,
    config: ProgressiveResidualConfig,
) -> ObservationField2D:
    height, width = mask.shape
    packed_alpha = None
    alpha_semantics = AlphaSemantics()
    if not mask.all():
        packed_alpha = pack_alpha(mask)
        alpha_semantics = AlphaSemantics(
            payload_encoding="binary_exact_packbits_little",
            matting_mode="multiply_alpha",
            boundary_policy="unconstrained",
        )
    semantics = FieldSemantics(
        coefficient_domain="signed",
        support=SupportSemantics(
            mode="axis_aligned_bbox",
            sigma_cutoff=config.sigma_cutoff,
            fade_alpha=config.support_fade_alpha,
            minimum_radius_px=1,
        ),
        alpha=alpha_semantics,
    )
    adaptation = adapt_direct_additive(
        means_xy=np.asarray(means, dtype=np.float32),
        log_scales_xy=np.asarray(log_scales, dtype=np.float32),
        rotations_rad=np.asarray(rotations, dtype=np.float32),
        rgb_coeff=np.asarray(coefficients, dtype=np.float32),
        canvas_crop=CanvasCropTransform(
            canvas_width=width,
            canvas_height=height,
            crop_x=0,
            crop_y=0,
            crop_width=width,
            crop_height=height,
        ),
        semantics=semantics,
        packed_alpha=packed_alpha,
    )
    return adaptation.require_pixel_exact()


def progressive_prefix_field(field: ObservationField2D, count: int) -> ObservationField2D:
    """Return a lossless prefix view as an independently renderable observation field."""

    if not isinstance(field, ObservationField2D):
        raise TypeError("field must be ObservationField2D")
    prefix_count = _integer(count, "count", minimum=1)
    if prefix_count > field.n:
        raise ValueError(f"count cannot exceed field.n ({field.n}), got {prefix_count}")
    return ObservationField2D(
        means_xy=np.array(field.means_xy[:prefix_count], copy=True),
        log_scales_xy=np.array(field.log_scales_xy[:prefix_count], copy=True),
        rotations_rad=np.array(field.rotations_rad[:prefix_count], copy=True),
        rgb_coeff=np.array(field.rgb_coeff[:prefix_count], copy=True),
        canvas_crop=field.canvas_crop,
        semantics=field.semantics,
        background_rgb=field.background_rgb,
        packed_alpha=field.packed_alpha,
        camera=field.camera,
        schema_version=field.schema_version,
    )


def _patch_rmse_max(pixel_mse: np.ndarray, side: int) -> tuple[float, int]:
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
    return float(math.sqrt(max(float(sums.max()), 0.0) / (effective * effective))), effective


def progressive_artifact_metrics(
    reconstruction: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    *,
    pixel_threshold: float,
    patch7_threshold: float,
    displayed: bool,
) -> dict[str, float | bool | int]:
    """Compute raw or exact displayed local metrics used by the hierarchy gate."""

    candidate = np.asarray(reconstruction, dtype=np.float64)
    source = np.asarray(target, dtype=np.float64)
    if candidate.shape != source.shape or candidate.ndim != 3 or candidate.shape[2] != 3:
        raise ValueError("reconstruction and target must have matching HWC RGB shapes")
    if mask.shape != candidate.shape[:2] or mask.dtype != np.bool_ or not mask.any():
        raise ValueError("mask must be a non-empty bool array matching the image")
    if displayed:
        candidate = np.rint(np.clip(candidate, 0.0, 1.0) * 255.0) / 255.0
        source = np.rint(np.clip(source, 0.0, 1.0) * 255.0) / 255.0
    residual = candidate - source
    pixel_mse = np.mean(residual * residual, axis=2)
    foreground = pixel_mse[mask]
    pixel_max = float(math.sqrt(max(float(foreground.max()), 0.0)))
    black_matted = np.where(mask, pixel_mse, 0.0)
    patch_max, effective_side = _patch_rmse_max(black_matted, 7)
    sse = float(np.sum(residual[mask] * residual[mask]))
    violation = max(pixel_max / pixel_threshold, patch_max / patch7_threshold)
    return {
        "sse": sse,
        "pixel_rmse_max": pixel_max,
        "patch7_rmse_max": patch_max,
        "patch7_effective_side": effective_side,
        "normalized_violation": violation,
        "gate_pass": pixel_max <= pixel_threshold and patch_max <= patch7_threshold,
    }


def _cold_render(
    field: ObservationField2D,
    config: ProgressiveResidualConfig,
) -> np.ndarray:
    from .pixel_contraction import render_observation_field

    return render_observation_field(
        field,
        device=config.device,
        renderer=config.renderer,
        render_chunk=config.render_chunk,
        apply_declared_alpha=False,
    ).astype(np.float32, copy=False)


def _optimize_layer(
    *,
    means: np.ndarray,
    log_scales: np.ndarray,
    rotations: np.ndarray,
    initial_colors: np.ndarray,
    prefix_raw: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    config: ProgressiveResidualConfig,
    steps: int,
    attempt_index: int,
    candidate_count: int,
    cumulative_step_offset: int,
) -> _LayerOptimization:
    import torch
    import torch.nn.functional as torch_functional

    from .gaussians import GaussianField
    from .render import render_field

    started = time.perf_counter()
    device = torch.device(config.device)
    layer = GaussianField.from_numpy(
        means,
        np.exp(log_scales),
        rotations,
        initial_colors,
        device=config.device,
    )
    layer.colors.requires_grad_(True)
    optimizer = torch.optim.Adam([layer.colors], lr=config.learning_rate)
    prefix_tensor = torch.as_tensor(
        np.ascontiguousarray(prefix_raw), device=device, dtype=torch.float32
    )
    target_tensor = torch.as_tensor(
        np.ascontiguousarray(target), device=device, dtype=torch.float32
    )
    mask_tensor = torch.as_tensor(mask, device=device, dtype=torch.bool)
    mask_float = mask_tensor.to(dtype=torch.float32)
    conics = layer.conics()
    scales = layer.scales()
    radii = layer.radii(config.sigma_cutoff)
    patch_side = min(7, *mask.shape)
    if patch_side % 2 == 0:
        patch_side -= 1
    patch_side = max(patch_side, 1)

    def render_layer():
        return render_field(
            layer.means,
            conics,
            layer.colors,
            radii,
            mask.shape[0],
            mask.shape[1],
            chunk=config.render_chunk,
            mode=config.renderer,
            opacities=None,
            scales=scales,
            rotations=layer.rotations,
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
        prefix_sse, prefix_pixel, prefix_patch, prefix_violation, _ = tensor_metrics(
            prefix_tensor
        )
    best_colors: np.ndarray | None = None
    best_render: np.ndarray | None = None
    best_step = -1
    best_metrics = (
        float(prefix_sse.item()),
        float(prefix_pixel.item()),
        float(prefix_patch.item()),
        float(prefix_violation.item()),
    )
    records: list[HierarchyCheckpoint] = []
    attempted_steps = 0
    for step in range(steps + 1):
        optimizer.zero_grad(set_to_none=True)
        layer_raw = render_layer()
        reconstruction = prefix_tensor + layer_raw
        sse, pixel_max, patch_max, violation, foreground = tensor_metrics(reconstruction)
        tail_count = max(1, int(math.ceil(config.tail_fraction * foreground.numel())))
        tail = torch.topk(foreground, k=tail_count, largest=True, sorted=False).values.mean()
        objective = foreground.mean() + config.tail_weight * tail
        if step % config.checkpoint_every == 0 or step == steps:
            candidate_key = (float(violation.detach().item()), float(sse.detach().item()))
            records.append(
                HierarchyCheckpoint(
                    attempt_index=attempt_index,
                    step=step,
                    cumulative_optimizer_step=cumulative_step_offset + step,
                    candidate_count=candidate_count,
                    raw_sse=candidate_key[1],
                    raw_pixel_rmse_max=float(pixel_max.detach().item()),
                    raw_patch7_rmse_max=float(patch_max.detach().item()),
                    raw_normalized_violation=candidate_key[0],
                    objective=float(objective.detach().item()),
                    elapsed_seconds=time.perf_counter() - started,
                )
            )
            if _lexicographic_improves(
                candidate_key[0],
                candidate_key[1],
                best_metrics[3],
                best_metrics[0],
            ):
                best_step = step
                best_colors = layer.colors.detach().cpu().numpy().copy()
                best_render = layer_raw.detach().cpu().numpy().copy()
                best_metrics = (
                    candidate_key[1],
                    float(pixel_max.detach().item()),
                    float(patch_max.detach().item()),
                    candidate_key[0],
                )
        if step == steps:
            break
        if not torch.isfinite(objective):
            break
        objective.backward()
        optimizer.step()
        attempted_steps += 1
    return _LayerOptimization(
        colors=best_colors,
        residual_render=best_render,
        selected_step=best_step,
        selected_metrics=best_metrics,
        checkpoints=tuple(records),
        elapsed_seconds=time.perf_counter() - started,
        attempted_steps=attempted_steps,
    )


def _candidate_splits(
    frontier: set[NodeKey],
    blocked: set[NodeKey],
    mask: np.ndarray,
    smoothed_error: np.ndarray,
) -> list[tuple[float, NodeKey, tuple[NodeKey, ...]]]:
    candidates = []
    for key in frontier:
        if key in blocked or key[0] == 0:
            continue
        children = _child_keys(mask, key)
        if not children:
            continue
        x0, y0, x1, y1 = _cell_bounds(key, mask.shape)
        cell_mask = mask[y0:y1, x0:x1]
        score = float(np.sum(smoothed_error[y0:y1, x0:x1][cell_mask])) / len(children)
        candidates.append((score, key, children))
    candidates.sort(key=lambda item: (-item[0], -item[1][0], item[1][1], item[1][2]))
    return candidates


def _snapshot_counts(
    stages: list[HierarchyStage],
    milestones: tuple[int, ...],
    final_count: int,
) -> tuple[int, ...]:
    accepted_counts = [stage.count_after for stage in stages if stage.accepted_rows > 0]
    if not accepted_counts:
        return (final_count,)
    selected = {accepted_counts[0], final_count}
    for milestone in milestones:
        below = [count for count in accepted_counts if count <= milestone]
        if below:
            selected.add(max(below))
    passing = [stage.count_after for stage in stages if stage.display_gate_pass]
    if passing:
        selected.add(passing[0])
    return tuple(sorted(selected))


def build_progressive_residual_quadtree(
    image: np.ndarray,
    config: ProgressiveResidualConfig,
    *,
    mask: np.ndarray | None = None,
) -> ProgressiveResidualResult:
    """Build a coarse-to-fine additive field with immutable parent/residual prefixes."""

    if not isinstance(config, ProgressiveResidualConfig):
        raise TypeError("config must be ProgressiveResidualConfig")
    source = _validated_image(image)
    active_mask = _validated_mask(mask, source.shape[:2])
    started = time.perf_counter()
    base_keys = _base_keys(active_mask, config.start_level)
    if not base_keys:
        raise RuntimeError("the start level produced no mask-present quadtree node")
    if len(base_keys) > config.max_gaussians:
        raise ValueError(
            f"base layer has {len(base_keys)} rows, exceeding max_gaussians={config.max_gaussians}"
        )

    initial_raw = np.zeros_like(source, dtype=np.float32)
    initial_metrics = progressive_artifact_metrics(
        initial_raw,
        source,
        active_mask,
        pixel_threshold=config.pixel_rmse_threshold,
        patch7_threshold=config.patch7_rmse_threshold,
        displayed=False,
    )
    means, log_scales, rotations = _geometry_rows(
        active_mask, base_keys, config.leaf_scale_px
    )
    base_colors = np.stack(
        [_cell_rgb_mean(source, active_mask, key) for key in base_keys], axis=0
    )
    cumulative_steps = 0
    base_optimization = _optimize_layer(
        means=means,
        log_scales=log_scales,
        rotations=rotations,
        initial_colors=base_colors,
        prefix_raw=initial_raw,
        target=source,
        mask=active_mask,
        config=config,
        steps=config.base_steps,
        attempt_index=0,
        candidate_count=len(base_keys),
        cumulative_step_offset=0,
    )
    cumulative_steps += base_optimization.attempted_steps
    checkpoints: list[HierarchyCheckpoint] = list(base_optimization.checkpoints)
    if base_optimization.colors is None or base_optimization.residual_render is None:
        coefficients = np.zeros((len(base_keys), 3), dtype=np.float32)
        accumulated_raw = initial_raw
        base_selected_step = -1
    else:
        coefficients = base_optimization.colors.astype(np.float32, copy=True)
        accumulated_raw = base_optimization.residual_render.astype(np.float32, copy=True)
        base_selected_step = base_optimization.selected_step
    field = _field_from_arrays(
        means, log_scales, rotations, coefficients, active_mask, config
    )
    cold_started = time.perf_counter()
    cold_raw = _cold_render(field, config)
    cold_seconds = time.perf_counter() - cold_started
    base_parity = float(np.max(np.abs(cold_raw - accumulated_raw)))
    cold_metrics = progressive_artifact_metrics(
        cold_raw,
        source,
        active_mask,
        pixel_threshold=config.pixel_rmse_threshold,
        patch7_threshold=config.patch7_rmse_threshold,
        displayed=False,
    )
    display_metrics = progressive_artifact_metrics(
        cold_raw,
        source,
        active_mask,
        pixel_threshold=config.pixel_rmse_threshold,
        patch7_threshold=config.patch7_rmse_threshold,
        displayed=True,
    )
    base_improved = _lexicographic_improves(
        float(cold_metrics["normalized_violation"]),
        float(cold_metrics["sse"]),
        float(initial_metrics["normalized_violation"]),
        float(initial_metrics["sse"]),
    )
    current_raw = cold_raw
    stages: list[HierarchyStage] = [
        HierarchyStage(
            attempt_index=0,
            accepted_index=0 if base_improved else None,
            kind="base",
            status="accepted" if base_improved else "base_not_improved",
            parent_keys=(),
            child_keys=base_keys,
            count_before=0,
            proposed_rows=len(base_keys),
            accepted_rows=len(base_keys),
            count_after=len(base_keys),
            selection_score_max=0.0,
            selection_score_min=0.0,
            selected_step=base_selected_step,
            sse_before=float(initial_metrics["sse"]),
            sse_after=float(cold_metrics["sse"]),
            raw_violation_before=float(initial_metrics["normalized_violation"]),
            raw_violation_after=float(cold_metrics["normalized_violation"]),
            display_pixel_rmse_max=float(display_metrics["pixel_rmse_max"]),
            display_patch7_rmse_max=float(display_metrics["patch7_rmse_max"]),
            display_gate_pass=bool(display_metrics["gate_pass"]),
            prefix_bit_exact=True,
            accumulated_vs_cold_max_abs=base_parity,
            selection_seconds=0.0,
            optimization_seconds=base_optimization.elapsed_seconds,
            cold_render_seconds=cold_seconds,
            cumulative_elapsed_seconds=time.perf_counter() - started,
        )
    ]
    frontier = set(base_keys)
    blocked: set[NodeKey] = set()
    accepted_split_count = 0
    accepted_stage_count = 0
    accepted_index = 1
    attempt_index = 1
    stop_reason = "display_gate_passed" if display_metrics["gate_pass"] else "frontier_exhausted"
    maintained_parity = base_parity
    prefix_bit_exact = True
    if not base_improved:
        stop_reason = "base_not_improved"

    while base_improved and not display_metrics["gate_pass"] and field.n < config.max_gaussians:
        if config.max_stages is not None and accepted_stage_count >= config.max_stages:
            stop_reason = "max_stages_reached"
            break
        selection_started = time.perf_counter()
        residual_energy = np.mean((current_raw - source) ** 2, axis=2)
        smoothed_error = _mask_aware_smoothed_error(
            residual_energy,
            active_mask,
            config.error_smoothing_sigma_px,
        )
        candidates = _candidate_splits(frontier, blocked, active_mask, smoothed_error)
        remaining = config.max_gaussians - field.n
        allowance = min(config.max_rows_per_stage, remaining)
        selected: list[tuple[float, NodeKey, tuple[NodeKey, ...]]] = []
        rows = 0
        for candidate in candidates:
            child_count = len(candidate[2])
            if rows + child_count <= allowance:
                selected.append(candidate)
                rows += child_count
            if rows == allowance:
                break
        selection_seconds = time.perf_counter() - selection_started
        if not selected:
            stop_reason = "frontier_exhausted" if not candidates else "no_split_fits_allowance"
            break

        parent_keys = tuple(item[1] for item in selected)
        child_keys = tuple(child for item in selected for child in item[2])
        layer_means, layer_log_scales, layer_rotations = _geometry_rows(
            active_mask, child_keys, config.leaf_scale_px
        )
        residual = source - current_raw
        layer_colors = np.stack(
            [_cell_rgb_mean(residual, active_mask, key) for key in child_keys], axis=0
        )
        before_metrics = progressive_artifact_metrics(
            current_raw,
            source,
            active_mask,
            pixel_threshold=config.pixel_rmse_threshold,
            patch7_threshold=config.patch7_rmse_threshold,
            displayed=False,
        )
        optimization = _optimize_layer(
            means=layer_means,
            log_scales=layer_log_scales,
            rotations=layer_rotations,
            initial_colors=layer_colors,
            prefix_raw=current_raw,
            target=source,
            mask=active_mask,
            config=config,
            steps=config.layer_steps,
            attempt_index=attempt_index,
            candidate_count=field.n + len(child_keys),
            cumulative_step_offset=cumulative_steps,
        )
        cumulative_steps += optimization.attempted_steps
        checkpoints.extend(optimization.checkpoints)
        accepted = False
        cold_candidate = current_raw
        candidate_metrics = before_metrics
        candidate_display = display_metrics
        candidate_parity = 0.0
        candidate_cold_seconds = 0.0
        old_means = means
        old_log_scales = log_scales
        old_rotations = rotations
        old_coefficients = coefficients
        if optimization.colors is not None and optimization.residual_render is not None:
            proposed_means = np.concatenate([means, layer_means], axis=0)
            proposed_log_scales = np.concatenate([log_scales, layer_log_scales], axis=0)
            proposed_rotations = np.concatenate([rotations, layer_rotations], axis=0)
            proposed_coefficients = np.concatenate(
                [coefficients, optimization.colors.astype(np.float32)], axis=0
            )
            candidate_field = _field_from_arrays(
                proposed_means,
                proposed_log_scales,
                proposed_rotations,
                proposed_coefficients,
                active_mask,
                config,
            )
            cold_started = time.perf_counter()
            cold_candidate = _cold_render(candidate_field, config)
            candidate_cold_seconds = time.perf_counter() - cold_started
            accumulated_candidate = current_raw + optimization.residual_render
            candidate_parity = float(np.max(np.abs(cold_candidate - accumulated_candidate)))
            candidate_metrics = progressive_artifact_metrics(
                cold_candidate,
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
                means = proposed_means
                log_scales = proposed_log_scales
                rotations = proposed_rotations
                coefficients = proposed_coefficients
                field = candidate_field
                current_raw = cold_candidate
                candidate_display = progressive_artifact_metrics(
                    current_raw,
                    source,
                    active_mask,
                    pixel_threshold=config.pixel_rmse_threshold,
                    patch7_threshold=config.patch7_rmse_threshold,
                    displayed=True,
                )
                display_metrics = candidate_display
                for parent in parent_keys:
                    frontier.remove(parent)
                    frontier.update(_child_keys(active_mask, parent))
                accepted_split_count += len(parent_keys)
                accepted_stage_count += 1
                maintained_parity = max(maintained_parity, candidate_parity)
                prefix_bit_exact = prefix_bit_exact and all(
                    np.array_equal(before, after[: before.shape[0]])
                    for before, after in (
                        (old_means, means),
                        (old_log_scales, log_scales),
                        (old_rotations, rotations),
                        (old_coefficients, coefficients),
                    )
                )
        if not accepted:
            blocked.update(parent_keys)
            candidate_metrics = before_metrics
            candidate_display = display_metrics

        stages.append(
            HierarchyStage(
                attempt_index=attempt_index,
                accepted_index=accepted_index if accepted else None,
                kind="residual_children",
                status="accepted" if accepted else "rolled_back",
                parent_keys=parent_keys,
                child_keys=child_keys,
                count_before=old_means.shape[0],
                proposed_rows=len(child_keys),
                accepted_rows=len(child_keys) if accepted else 0,
                count_after=field.n,
                selection_score_max=max(item[0] for item in selected),
                selection_score_min=min(item[0] for item in selected),
                selected_step=optimization.selected_step,
                sse_before=float(before_metrics["sse"]),
                sse_after=float(candidate_metrics["sse"]),
                raw_violation_before=float(before_metrics["normalized_violation"]),
                raw_violation_after=float(candidate_metrics["normalized_violation"]),
                display_pixel_rmse_max=float(candidate_display["pixel_rmse_max"]),
                display_patch7_rmse_max=float(candidate_display["patch7_rmse_max"]),
                display_gate_pass=bool(candidate_display["gate_pass"]),
                prefix_bit_exact=prefix_bit_exact,
                accumulated_vs_cold_max_abs=candidate_parity,
                selection_seconds=selection_seconds,
                optimization_seconds=optimization.elapsed_seconds,
                cold_render_seconds=candidate_cold_seconds,
                cumulative_elapsed_seconds=time.perf_counter() - started,
            )
        )
        if accepted:
            accepted_index += 1
            if display_metrics["gate_pass"]:
                stop_reason = "display_gate_passed"
                break
        attempt_index += 1
    else:
        if display_metrics["gate_pass"]:
            stop_reason = "display_gate_passed"
        elif field.n >= config.max_gaussians:
            stop_reason = "max_gaussians_reached"

    final_metrics = progressive_artifact_metrics(
        current_raw,
        source,
        active_mask,
        pixel_threshold=config.pixel_rmse_threshold,
        patch7_threshold=config.patch7_rmse_threshold,
        displayed=False,
    )
    reconstruction = np.asarray(current_raw, dtype=np.float32)
    if field.packed_alpha is not None and field.semantics.alpha.matting_mode == "multiply_alpha":
        reconstruction = reconstruction * active_mask[:, :, None]
    alpha_bytes = 0 if field.packed_alpha is None else int(field.packed_alpha.nbytes)
    coefficient_proxy_bytes = field.n * 3 * np.dtype(np.float32).itemsize
    tree_proxy_bits = field.n
    structured_proxy = coefficient_proxy_bytes + (tree_proxy_bits + 7) // 8 + alpha_bytes
    canonical_raw_bytes = int(sum(array.nbytes for array in field._array_items().values()))
    return ProgressiveResidualResult(
        field=field,
        reconstruction_raw=np.asarray(current_raw, dtype=np.float32),
        reconstruction=np.asarray(reconstruction, dtype=np.float32),
        stages=tuple(stages),
        checkpoints=tuple(checkpoints),
        base_count=len(base_keys),
        final_count=field.n,
        accepted_split_count=accepted_split_count,
        initial_sse=float(initial_metrics["sse"]),
        final_sse=float(final_metrics["sse"]),
        stop_reason=stop_reason,
        elapsed_seconds=time.perf_counter() - started,
        estimated_field_bytes=field.n * config.estimated_row_bytes + alpha_bytes,
        canonical_raw_bytes=canonical_raw_bytes,
        coefficient_proxy_bytes=coefficient_proxy_bytes,
        tree_proxy_bits=tree_proxy_bits,
        estimated_structured_proxy_bytes=structured_proxy,
        prefix_bit_exact=prefix_bit_exact,
        maintained_render_parity_max_abs=maintained_parity,
        snapshot_counts=_snapshot_counts(stages, config.milestone_counts, field.n),
    )
