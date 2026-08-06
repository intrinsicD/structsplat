"""Packet-derived occlusion-aware ray-posterior initialization for realtime-gs.

The module is intentionally optional and import-lazy: importing it does not import torch,
realtime-gs, DINOv2, or OpenCV.  At execution time it uses the existing codec-native structural
field only to propose source rays.  Continuous packet appearance supplies descriptors and
radiance; calibrated source-excluded views supply a bounded depth likelihood; an explicit dustbin
absorbs occluded or incompatible views; and reciprocal ray agreement rejects isolated modes before
constructing a realtime-gs ``Gaussians3D`` field.

This is a default-off research adapter, not part of StructSplat's maintained 2D conversion path.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import math
from pathlib import Path
import subprocess
import time
from typing import Any, Callable, Sequence

from .realtime_gs_adapter import RealtimeGSCodecNativeView


@dataclass(frozen=True)
class RayPosteriorConfig:
    """Controls owned by the packet-feature and ray-posterior layer."""

    feature_model: str = "dinov2_vits14"
    feature_max_side: int = 518
    feature_patch_size: int = 14
    feature_device: str = "cuda"
    feature_query_chunk: int = 65_536
    feature_storage_dtype: str = "float16"
    target_views: int = 4
    target_baseline_deg: float = 18.0
    min_baseline_deg: float = 3.0
    max_baseline_deg: float = 65.0
    best_view_count: int = 2
    min_evidence_views: int = 2
    dustbin_cost: float = 0.65
    view_dispersion_weight: float = 0.25
    posterior_temperature: float = 0.08
    dino_weight: float = 1.0
    detail_weight: float = 0.25
    fine_samples: int = 9
    fine_half_width_steps: float = 1.0
    score_batch_rays: int = 1_024
    reciprocal_pixel_radius: float = 18.0
    reciprocal_depth_extent_fraction: float = 0.04
    reciprocal_world_extent_fraction: float = 0.06
    min_reciprocal_views: int = 1
    apply_reciprocal: bool = True
    allow_confidence_fallback: bool = True
    min_primary_fraction: float = 0.75
    min_depth_sigma_extent_fraction: float = 0.002
    max_depth_sigma_extent_fraction: float = 0.05
    apply_surface_cover: bool = True
    surface_cover_isotropic: bool = False
    surface_cover_min_planarity: float = 0.0

    def __post_init__(self) -> None:
        integer_positive = (
            "feature_max_side",
            "feature_patch_size",
            "feature_query_chunk",
            "target_views",
            "best_view_count",
            "min_evidence_views",
            "fine_samples",
            "score_batch_rays",
        )
        for name in integer_positive:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.fine_samples < 3 or self.fine_samples % 2 == 0:
            raise ValueError("fine_samples must be an odd integer of at least three")
        if self.feature_storage_dtype not in {"float16", "float32"}:
            raise ValueError("feature_storage_dtype must be 'float16' or 'float32'")
        if self.best_view_count > self.target_views:
            raise ValueError("best_view_count cannot exceed target_views")
        if self.min_evidence_views > self.best_view_count:
            raise ValueError("min_evidence_views cannot exceed best_view_count")
        finite_positive = (
            "target_baseline_deg",
            "max_baseline_deg",
            "dustbin_cost",
            "posterior_temperature",
            "fine_half_width_steps",
            "reciprocal_pixel_radius",
            "reciprocal_depth_extent_fraction",
            "reciprocal_world_extent_fraction",
            "max_depth_sigma_extent_fraction",
        )
        for name in finite_positive:
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if (
            not math.isfinite(self.min_baseline_deg)
            or self.min_baseline_deg < 0.0
            or self.min_baseline_deg >= self.max_baseline_deg
        ):
            raise ValueError("min_baseline_deg must be finite, non-negative, and below max")
        for name in ("dino_weight", "detail_weight", "view_dispersion_weight"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.dino_weight + self.detail_weight <= 0.0:
            raise ValueError("at least one descriptor weight must be positive")
        if (
            isinstance(self.min_reciprocal_views, bool)
            or not isinstance(self.min_reciprocal_views, int)
            or self.min_reciprocal_views < 0
        ):
            raise ValueError("min_reciprocal_views must be a non-negative integer")
        if self.min_reciprocal_views > self.target_views:
            raise ValueError("min_reciprocal_views cannot exceed target_views")
        if not 0.0 <= self.min_primary_fraction <= 1.0:
            raise ValueError("min_primary_fraction must lie in [0,1]")
        if (
            not math.isfinite(self.min_depth_sigma_extent_fraction)
            or self.min_depth_sigma_extent_fraction <= 0.0
            or self.min_depth_sigma_extent_fraction
            > self.max_depth_sigma_extent_fraction
        ):
            raise ValueError("depth sigma extent fractions must be positive and ordered")
        if not 0.0 <= self.surface_cover_min_planarity < 1.0:
            raise ValueError("surface_cover_min_planarity must lie in [0,1)")
        for name in (
            "apply_reciprocal",
            "allow_confidence_fallback",
            "apply_surface_cover",
            "surface_cover_isotropic",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be bool")


@dataclass(frozen=True)
class PacketFeatureView:
    """One packet crop represented by a semantic and a local-detail feature map."""

    semantic: Any  # (C,Hs,Ws), L2-normalized
    detail: Any  # (D,H,W), L2-normalized
    alpha: Any  # (1,H,W), float in {0,1}
    crop: tuple[int, int, int, int]
    canvas: tuple[int, int]
    semantic_input_shape: tuple[int, int]


@dataclass(frozen=True)
class PacketFeatureSet:
    """Shared packet-derived features and their complete build receipt."""

    views: tuple[PacketFeatureView, ...]
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class DepthPosteriorSummary:
    """Robust per-ray discrete posterior and confidence diagnostics."""

    posterior: Any
    aggregate_cost: Any
    evidence_views: Any
    eligible_depth: Any
    best_index: Any
    best_cost: Any
    margin: Any
    normalized_entropy: Any


@dataclass(frozen=True)
class RayPosteriorLiftResult:
    """Realtime-gs initialization plus candidate-complete causal diagnostics."""

    initialization: Any
    raw_initialization: Any
    selected_candidate_indices: Any
    selected_fallback_mask: Any
    candidate_depths: Any
    candidate_scores: Any
    candidate_reciprocal_support: Any
    diagnostics: dict[str, Any]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_revision(path: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _distribution(values: Any) -> dict[str, float]:
    values = values.detach().double().reshape(-1)
    if values.numel() == 0:
        return {name: 0.0 for name in ("min", "p10", "median", "p90", "max", "mean")}
    return {
        "min": float(values.min()),
        "p10": float(values.quantile(0.10)),
        "median": float(values.median()),
        "p90": float(values.quantile(0.90)),
        "max": float(values.max()),
        "mean": float(values.mean()),
    }


def robust_depth_posterior(
    costs: Any,
    valid: Any,
    *,
    dustbin_cost: float,
    best_view_count: int,
    min_evidence_views: int,
    temperature: float,
    view_dispersion_weight: float = 0.25,
) -> DepthPosteriorSummary:
    """Aggregate source-excluded view costs with an explicit missed-observation state.

    ``costs`` and ``valid`` have shape ``(..., S, V)``.  Invalid or more expensive observations
    choose the dustbin.  The best ``K`` observations (including dustbins) determine a depth's cost;
    at least ``min_evidence_views`` real observations must beat the dustbin.  Ties are deterministic:
    ``argmin`` returns the first depth sample.
    """
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - package dependency in normal environments
        raise RuntimeError("ray-posterior scoring requires torch") from exc
    if not torch.is_tensor(costs) or not torch.is_tensor(valid):
        raise TypeError("costs and valid must be torch tensors")
    if costs.shape != valid.shape or costs.ndim < 2:
        raise ValueError("costs and valid must share shape (...,S,V)")
    if not costs.is_floating_point() or valid.dtype != torch.bool:
        raise TypeError("costs must float and valid must bool")
    if not bool(torch.isfinite(costs[valid]).all()):
        raise ValueError("valid costs must be finite")
    if not 1 <= best_view_count <= costs.shape[-1]:
        raise ValueError("best_view_count must lie in [1,V]")
    if not 1 <= min_evidence_views <= best_view_count:
        raise ValueError("min_evidence_views must lie in [1,best_view_count]")
    if not math.isfinite(dustbin_cost) or dustbin_cost <= 0.0:
        raise ValueError("dustbin_cost must be finite and positive")
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be finite and positive")
    if not math.isfinite(view_dispersion_weight) or view_dispersion_weight < 0.0:
        raise ValueError("view_dispersion_weight must be finite and non-negative")

    dustbin = torch.full_like(costs, dustbin_cost)
    competed = torch.where(valid, torch.minimum(costs, dustbin), dustbin)
    top = torch.topk(competed, k=best_view_count, dim=-1, largest=False, sorted=True).values
    evidence = (top < dustbin_cost).sum(dim=-1)
    aggregate = top.mean(dim=-1) + view_dispersion_weight * (
        top[..., -1] - top[..., 0]
    )
    eligible = evidence >= min_evidence_views
    logits = -aggregate / temperature
    logits = torch.where(eligible, logits, torch.full_like(logits, -torch.inf))
    any_eligible = eligible.any(dim=-1, keepdim=True)
    safe_logits = torch.where(any_eligible, logits, torch.zeros_like(logits))
    posterior = torch.softmax(safe_logits, dim=-1)
    posterior = torch.where(any_eligible, posterior, torch.zeros_like(posterior))
    ranked = torch.where(eligible, aggregate, torch.full_like(aggregate, torch.inf))
    best_index = ranked.argmin(dim=-1)
    best_cost = ranked.gather(-1, best_index[..., None])[..., 0]
    if aggregate.shape[-1] > 1:
        two = torch.topk(ranked, k=2, dim=-1, largest=False, sorted=True).values
        margin = two[..., 1] - two[..., 0]
        margin = torch.where(torch.isfinite(margin), margin, torch.zeros_like(margin))
    else:
        margin = torch.zeros_like(best_cost)
    entropy = -(posterior * posterior.clamp_min(torch.finfo(posterior.dtype).tiny).log()).sum(
        dim=-1
    )
    if aggregate.shape[-1] > 1:
        entropy = entropy / math.log(aggregate.shape[-1])
    entropy = torch.where(any_eligible[..., 0], entropy, torch.ones_like(entropy))
    return DepthPosteriorSummary(
        posterior=posterior,
        aggregate_cost=aggregate,
        evidence_views=evidence,
        eligible_depth=eligible,
        best_index=best_index,
        best_cost=best_cost,
        margin=margin,
        normalized_entropy=entropy,
    )


def _detail_descriptor(rgb: Any) -> Any:
    import torch
    import torch.nn.functional as functional

    gray = (
        0.2126 * rgb[:, 0:1]
        + 0.7152 * rgb[:, 1:2]
        + 0.0722 * rgb[:, 2:3]
    )
    local_mean = functional.avg_pool2d(gray, 5, stride=1, padding=2)
    local_square = functional.avg_pool2d(gray.square(), 5, stride=1, padding=2)
    local_std = (local_square - local_mean.square()).clamp_min(1e-6).sqrt()
    kernel_x = gray.new_tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
    )[None, None] / 8.0
    kernel_y = kernel_x.transpose(-1, -2)
    dx = functional.conv2d(gray, kernel_x, padding=1)
    dy = functional.conv2d(gray, kernel_y, padding=1)
    laplace = functional.conv2d(
        gray,
        gray.new_tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]])[
            None, None
        ],
        padding=1,
    )
    chroma = rgb / rgb.square().sum(dim=1, keepdim=True).clamp_min(1e-6).sqrt()
    descriptor = torch.cat(
        [
            chroma,
            (gray - local_mean) / local_std,
            dx / local_std,
            dy / local_std,
            (dx.square() + dy.square()).sqrt() / local_std,
            laplace / local_std,
        ],
        dim=1,
    )
    return functional.normalize(descriptor, dim=1, eps=1e-6)


def _query_packet_crop(view: RealtimeGSCodecNativeView, config: RayPosteriorConfig, device: Any):
    import torch

    backend = view.query_backend
    if not hasattr(backend, "query_appearance"):
        raise TypeError("codec-native query backend lacks query_appearance")
    x, y, width, height = (int(value) for value in view.structural_field.fit_window)
    yy, xx = torch.meshgrid(
        torch.arange(height, device=device, dtype=torch.float32) + y + 0.5,
        torch.arange(width, device=device, dtype=torch.float32) + x + 0.5,
        indexing="ij",
    )
    points = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)
    colors = []
    alphas = []
    valids = []
    for start in range(0, points.shape[0], config.feature_query_chunk):
        color, alpha, valid = backend.query_appearance(
            points[start : start + config.feature_query_chunk]
        )
        colors.append(color.to(device))
        alphas.append(alpha.to(device))
        valids.append(valid.to(device))
    rgb = torch.cat(colors).reshape(height, width, 3).permute(2, 0, 1)[None]
    alpha = (torch.cat(alphas) & torch.cat(valids)).reshape(height, width)[None, None]
    rgb = rgb.clamp(0.0, 1.0) * alpha.to(rgb)
    return rgb, alpha, (x, y, width, height)


def _load_dino_model(config: RayPosteriorConfig, device: Any):
    import torch

    model = torch.hub.load(
        "facebookresearch/dinov2",
        config.feature_model,
        pretrained=True,
        trust_repo=True,
    )
    return model.eval().to(device)


def _model_receipt(config: RayPosteriorConfig, injected: bool) -> dict[str, Any]:
    import torch

    hub = Path(torch.hub.get_dir())
    checkpoint_names = {
        "dinov2_vits14": "dinov2_vits14_pretrain.pth",
        "dinov2_vitb14": "dinov2_vitb14_pretrain.pth",
        "dinov2_vitl14": "dinov2_vitl14_pretrain.pth",
        "dinov2_vitg14": "dinov2_vitg14_pretrain.pth",
    }
    checkpoint = hub / "checkpoints" / checkpoint_names.get(config.feature_model, "")
    repository = hub / "facebookresearch_dinov2_main"
    license_path = repository / "LICENSE"
    return {
        "model": config.feature_model,
        "injected_model": injected,
        "checkpoint": (
            {
                "path": str(checkpoint),
                "bytes": checkpoint.stat().st_size,
                "sha256": _sha256(checkpoint),
            }
            if checkpoint.is_file()
            else None
        ),
        "repository": (
            {"path": str(repository), "revision": _git_revision(repository)}
            if repository.is_dir()
            else None
        ),
        "license": (
            {
                "spdx": "Apache-2.0",
                "path": str(license_path),
                "bytes": license_path.stat().st_size,
                "sha256": _sha256(license_path),
            }
            if license_path.is_file()
            else None
        ),
    }


def build_packet_feature_pyramids(
    views: Sequence[RealtimeGSCodecNativeView],
    config: RayPosteriorConfig | None = None,
    *,
    model: Any | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> PacketFeatureSet:
    """Decode continuous packet crops and build shared semantic/detail feature maps."""
    config = config or RayPosteriorConfig()
    if not views:
        raise ValueError("views must be non-empty")
    for index, view in enumerate(views):
        if not isinstance(view, RealtimeGSCodecNativeView):
            raise TypeError(f"views[{index}] must be RealtimeGSCodecNativeView")
    try:
        import torch
        import torch.nn.functional as functional
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("packet feature extraction requires torch") from exc
    device = torch.device(config.feature_device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("feature_device='cuda' requires CUDA")
    injected = model is not None
    if model is None:
        model = _load_dino_model(config, device)
    else:
        model = model.eval().to(device)
    storage_dtype = torch.float16 if config.feature_storage_dtype == "float16" else torch.float32
    started = time.perf_counter()
    outputs = []
    per_view = []
    with torch.inference_mode():
        for index, view in enumerate(views):
            view_started = time.perf_counter()
            rgb, alpha, crop = _query_packet_crop(view, config, device)
            height, width = rgb.shape[-2:]
            scale = min(1.0, config.feature_max_side / max(height, width))
            input_height = max(
                config.feature_patch_size,
                int(round(height * scale / config.feature_patch_size))
                * config.feature_patch_size,
            )
            input_width = max(
                config.feature_patch_size,
                int(round(width * scale / config.feature_patch_size))
                * config.feature_patch_size,
            )
            resized = functional.interpolate(
                rgb,
                size=(input_height, input_width),
                mode="bicubic",
                align_corners=False,
                antialias=True,
            ).clamp(0.0, 1.0)
            normalized = (resized - resized.new_tensor([0.485, 0.456, 0.406])[None, :, None, None])
            normalized = normalized / resized.new_tensor([0.229, 0.224, 0.225])[
                None, :, None, None
            ]
            model_output = model.forward_features(normalized)
            if not isinstance(model_output, dict) or "x_norm_patchtokens" not in model_output:
                raise TypeError("feature model must return x_norm_patchtokens from forward_features")
            tokens = model_output["x_norm_patchtokens"]
            grid_height = input_height // config.feature_patch_size
            grid_width = input_width // config.feature_patch_size
            if tokens.shape[0] != 1 or tokens.shape[1] != grid_height * grid_width:
                raise ValueError("feature model patch-token shape does not match configured patch size")
            semantic = tokens.reshape(1, grid_height, grid_width, -1).permute(0, 3, 1, 2)
            semantic = functional.normalize(semantic.float(), dim=1, eps=1e-6)[0]
            detail = _detail_descriptor(rgb.float())[0]
            feature_view = PacketFeatureView(
                semantic=semantic.to(storage_dtype),
                detail=detail.to(storage_dtype),
                alpha=alpha[0].to(storage_dtype),
                crop=crop,
                canvas=(view.structural_field.width, view.structural_field.height),
                semantic_input_shape=(input_height, input_width),
            )
            outputs.append(feature_view)
            per_view.append(
                {
                    "index": index,
                    "crop": list(crop),
                    "canvas": list(feature_view.canvas),
                    "semantic_input_shape": [input_height, input_width],
                    "semantic_shape": list(semantic.shape),
                    "detail_shape": list(detail.shape),
                    "foreground_fraction": float(alpha.float().mean()),
                    "seconds": time.perf_counter() - view_started,
                }
            )
            if progress_callback is not None:
                progress_callback(index + 1, len(views))
            del rgb, resized, normalized, model_output, tokens, semantic, detail
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    diagnostics = {
        "schema": "structsplat.packet_feature_pyramids.v1",
        "ownership": "derived_from_cold_packet_appearance_and_alpha",
        "source_rgb_opened": False,
        "device": str(device),
        "storage_dtype": config.feature_storage_dtype,
        "preprocessing": {
            "image_range": "packet_query_clamped_[0,1]_and_alpha_matted",
            "normalization_mean": [0.485, 0.456, 0.406],
            "normalization_std": [0.229, 0.224, 0.225],
            "max_side": config.feature_max_side,
            "patch_size": config.feature_patch_size,
            "resize": "bicubic_antialiased_to_nearest_patch_multiple",
            "detail": "unit_chroma_plus_local_standardized_sobel_magnitude_laplacian",
        },
        "model": _model_receipt(config, injected),
        "view_count": len(outputs),
        "seconds": time.perf_counter() - started,
        "views": per_view,
    }
    return PacketFeatureSet(tuple(outputs), diagnostics)


def _sample_feature_map(feature: PacketFeatureView, xy: Any, member: str):
    import torch
    import torch.nn.functional as functional

    value = getattr(feature, member)
    x, y, width, height = feature.crop
    grid_x = 2.0 * (xy[:, 0] - x) / width - 1.0
    grid_y = 2.0 * (xy[:, 1] - y) / height - 1.0
    grid = torch.stack([grid_x, grid_y], dim=-1).reshape(1, -1, 1, 2)
    sampled = functional.grid_sample(
        value[None].float(),
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )[0, :, :, 0].T
    sampled = functional.normalize(sampled, dim=-1, eps=1e-6)
    inside = (
        (xy[:, 0] >= x + 0.5)
        & (xy[:, 0] <= x + width - 0.5)
        & (xy[:, 1] >= y + 0.5)
        & (xy[:, 1] <= y + height - 0.5)
    )
    alpha = functional.grid_sample(
        feature.alpha[None].float(),
        grid,
        mode="nearest",
        padding_mode="zeros",
        align_corners=False,
    )[0, 0, :, 0]
    return sampled, inside & (alpha > 0.5)


def _choose_neighbors(cameras: Sequence[Any], center: Any, config: RayPosteriorConfig):
    import torch

    positions = torch.stack([camera.position for camera in cameras]).double()
    directions = torch.nn.functional.normalize(positions - center.double()[None], dim=-1)
    result = []
    for source in range(len(cameras)):
        cosine = (directions[source][None] * directions).sum(dim=-1).clamp(-1.0, 1.0)
        angles = torch.rad2deg(torch.acos(cosine))
        candidates = []
        for target in range(len(cameras)):
            if target == source:
                continue
            angle = float(angles[target])
            in_band = config.min_baseline_deg <= angle <= config.max_baseline_deg
            candidates.append(
                (
                    0 if in_band else 1,
                    abs(angle - config.target_baseline_deg),
                    target,
                )
            )
        candidates.sort()
        chosen = tuple(item[2] for item in candidates[: config.target_views])
        if len(chosen) != config.target_views:
            raise ValueError("not enough source-excluded cameras for target_views")
        result.append(chosen)
    return tuple(result)


def _score_depth_grid(
    world: Any,
    source_view: int,
    source_xy: Any,
    cameras: Sequence[Any],
    neighbors: Sequence[int],
    features: PacketFeatureSet,
    config: RayPosteriorConfig,
):
    import torch

    batch, samples = world.shape[:2]
    source_semantic, source_valid_semantic = _sample_feature_map(
        features.views[source_view], source_xy, "semantic"
    )
    source_detail, source_valid_detail = _sample_feature_map(
        features.views[source_view], source_xy, "detail"
    )
    source_valid = source_valid_semantic & source_valid_detail
    costs = []
    valids = []
    flat_world = world.reshape(-1, 3)
    total_weight = config.dino_weight + config.detail_weight
    for target in neighbors:
        uv, depth = cameras[target].project(flat_world)
        target_semantic, valid_semantic = _sample_feature_map(
            features.views[target], uv, "semantic"
        )
        target_detail, valid_detail = _sample_feature_map(features.views[target], uv, "detail")
        semantic_cost = 1.0 - (
            source_semantic[:, None, :] * target_semantic.reshape(batch, samples, -1)
        ).sum(dim=-1)
        detail_cost = 1.0 - (
            source_detail[:, None, :] * target_detail.reshape(batch, samples, -1)
        ).sum(dim=-1)
        combined = (
            config.dino_weight * semantic_cost + config.detail_weight * detail_cost
        ) / total_weight
        valid = (
            source_valid[:, None]
            & valid_semantic.reshape(batch, samples)
            & valid_detail.reshape(batch, samples)
            & (depth.reshape(batch, samples) > 0.0)
        )
        costs.append(combined)
        valids.append(valid)
    return torch.stack(costs, dim=-1), torch.stack(valids, dim=-1)


def _candidate_reciprocal_support(
    means: Any,
    depths: Any,
    valid: Any,
    view_ids: Any,
    xy: Any,
    cameras: Sequence[Any],
    neighbors: Sequence[Sequence[int]],
    extent: float,
    config: RayPosteriorConfig,
):
    import torch

    support = torch.zeros_like(view_ids, dtype=torch.long)
    for source, targets in enumerate(neighbors):
        source_indices = (view_ids == source).nonzero(as_tuple=True)[0]
        if source_indices.numel() == 0:
            continue
        for target in targets:
            target_indices = (view_ids == target).nonzero(as_tuple=True)[0]
            if target_indices.numel() == 0:
                continue
            projected_target, target_depth = cameras[target].project(means[source_indices])
            forward_distance = torch.cdist(projected_target, xy[target_indices])
            nearest_distance, nearest_local = forward_distance.min(dim=1)
            nearest_target = target_indices[nearest_local]
            projected_source, _ = cameras[source].project(means[nearest_target])
            cycle_distance = (projected_source - xy[source_indices]).norm(dim=-1)
            target_candidate_depth = depths[nearest_target]
            depth_consistent = (
                (target_candidate_depth - target_depth).abs()
                <= config.reciprocal_depth_extent_fraction * extent
            )
            world_consistent = (
                (means[nearest_target] - means[source_indices]).norm(dim=-1)
                <= config.reciprocal_world_extent_fraction * extent
            )
            consistent = (
                valid[source_indices]
                & valid[nearest_target]
                & (nearest_distance <= config.reciprocal_pixel_radius)
                & (cycle_distance <= config.reciprocal_pixel_radius)
                & depth_consistent
                & world_consistent
            )
            support[source_indices] += consistent.long()
    return support


def initialize_occlusion_aware_ray_posterior(
    inputs: Any,
    views: Sequence[RealtimeGSCodecNativeView],
    carve_config: Any,
    features: PacketFeatureSet,
    config: RayPosteriorConfig | None = None,
    *,
    progress_callback: Callable[[int, int], None] | None = None,
) -> RayPosteriorLiftResult:
    """Place an exact-count realtime-gs field from packet-derived ray-depth posteriors."""
    config = config or RayPosteriorConfig()
    if len(views) != int(inputs.n_views) or len(features.views) != int(inputs.n_views):
        raise ValueError("inputs, views, and features must contain the same ordered cameras")
    for index, (view, field) in enumerate(zip(views, inputs.observations, strict=True)):
        if not isinstance(view, RealtimeGSCodecNativeView):
            raise TypeError(f"views[{index}] must be RealtimeGSCodecNativeView")
        if view.structural_field is not field:
            raise ValueError(f"views[{index}] does not own inputs.observations[{index}]")
    try:
        import torch
        from rtgs.core.gaussians3d import Gaussians3D
        from rtgs.lift.base import lift_covariance
        from rtgs.lift.compact_carve import (
            CompactCarveConfig,
            CompactInitializationResult,
            CompactLineage,
            _balanced_topk,
            _bounds_source,
            _center_and_extent,
            _component_covariances,
            _propose_anchors,
            _ray_box,
            _validate_cpu_inputs,
        )
        from rtgs.lift.surfel_init import SurfelInitConfig, reconcile_covariances
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "ray-posterior lifting requires torch and the optional realtime-gs package"
        ) from exc
    if not isinstance(carve_config, CompactCarveConfig):
        raise TypeError("carve_config must be rtgs CompactCarveConfig")
    _validate_cpu_inputs(inputs)
    if carve_config.samples_per_ray < 2:
        raise ValueError("carve_config.samples_per_ray must be at least two")
    device = features.views[0].semantic.device
    if any(feature.semantic.device != device or feature.detail.device != device for feature in features.views):
        raise ValueError("all packet features must share one device")

    started = time.perf_counter()
    generator = torch.Generator(device="cpu").manual_seed(carve_config.seed)
    view_ids_cpu, component_ids_cpu, xy_cpu, anchor_attempts, proposed_per_view = _propose_anchors(
        inputs, carve_config, generator
    )
    candidate_count = int(view_ids_cpu.numel())
    if candidate_count < carve_config.n_init_3d:
        raise ValueError("ray-posterior proposal count is below n_init_3d")
    dtype = inputs.observations[0].dtype
    center_cpu, extent = _center_and_extent(inputs, dtype)
    half = extent * carve_config.bounds_scale
    lo = center_cpu.to(device) - half
    hi = center_cpu.to(device) + half
    cameras = tuple(camera.to(device) for camera in inputs.cameras)
    neighbors = _choose_neighbors(inputs.cameras, center_cpu, config)
    view_ids = view_ids_cpu.to(device)
    xy = xy_cpu.to(device=device, dtype=torch.float32)
    candidate_depths = torch.zeros(candidate_count, device=device, dtype=torch.float32)
    candidate_sigmas = torch.zeros_like(candidate_depths)
    candidate_means = torch.zeros(candidate_count, 3, device=device, dtype=torch.float32)
    candidate_scores = torch.zeros_like(candidate_depths)
    candidate_valid = torch.zeros(candidate_count, device=device, dtype=torch.bool)
    candidate_entropy = torch.ones_like(candidate_depths)
    candidate_margin = torch.zeros_like(candidate_depths)
    candidate_evidence = torch.zeros(candidate_count, device=device, dtype=torch.long)
    coarse_steps = (
        torch.arange(carve_config.samples_per_ray, device=device, dtype=torch.float32) + 0.5
    ) / carve_config.samples_per_ray
    processed = 0
    scoring_started = time.perf_counter()
    for source in range(inputs.n_views):
        source_indices = (view_ids == source).nonzero(as_tuple=True)[0]
        for offset in range(0, source_indices.numel(), config.score_batch_rays):
            selected_indices = source_indices[offset : offset + config.score_batch_rays]
            source_xy = xy[selected_indices]
            origin, direction = cameras[source].pixel_rays(source_xy)
            origins = origin.expand(selected_indices.numel(), -1)
            t0, t1 = _ray_box(origins, direction, lo, hi)
            t0 = t0.clamp_min(carve_config.near)
            ray_valid = t1 > t0
            coarse_depth = t0[:, None] + (t1 - t0).clamp_min(0.0)[:, None] * coarse_steps
            coarse_world = origins[:, None, :] + coarse_depth[..., None] * direction[:, None, :]
            coarse_costs, coarse_valid = _score_depth_grid(
                coarse_world,
                source,
                source_xy,
                cameras,
                neighbors[source],
                features,
                config,
            )
            coarse = robust_depth_posterior(
                coarse_costs,
                coarse_valid,
                dustbin_cost=config.dustbin_cost,
                best_view_count=config.best_view_count,
                min_evidence_views=config.min_evidence_views,
                temperature=config.posterior_temperature,
                view_dispersion_weight=config.view_dispersion_weight,
            )
            row = torch.arange(selected_indices.numel(), device=device)
            coarse_best = coarse_depth[row, coarse.best_index]
            coarse_step = (t1 - t0).clamp_min(0.0) / carve_config.samples_per_ray
            fine_offsets = torch.linspace(
                -config.fine_half_width_steps,
                config.fine_half_width_steps,
                config.fine_samples,
                device=device,
            )
            fine_depth = coarse_best[:, None] + coarse_step[:, None] * fine_offsets[None]
            fine_depth = torch.maximum(torch.minimum(fine_depth, t1[:, None]), t0[:, None])
            fine_world = origins[:, None, :] + fine_depth[..., None] * direction[:, None, :]
            fine_costs, fine_valid = _score_depth_grid(
                fine_world,
                source,
                source_xy,
                cameras,
                neighbors[source],
                features,
                config,
            )
            fine = robust_depth_posterior(
                fine_costs,
                fine_valid,
                dustbin_cost=config.dustbin_cost,
                best_view_count=config.best_view_count,
                min_evidence_views=config.min_evidence_views,
                temperature=config.posterior_temperature,
                view_dispersion_weight=config.view_dispersion_weight,
            )
            best_depth = fine_depth[row, fine.best_index]
            best_mean = fine_world[row, fine.best_index]
            best_evidence = fine.evidence_views[row, fine.best_index]
            eligible = ray_valid & fine.eligible_depth.any(dim=-1)
            variance = (
                fine.posterior * (fine_depth - best_depth[:, None]).square()
            ).sum(dim=-1)
            sigma = variance.clamp_min(0.0).sqrt().clamp(
                min=config.min_depth_sigma_extent_fraction * extent,
                max=config.max_depth_sigma_extent_fraction * extent,
            )
            margin_confidence = torch.sigmoid(
                fine.margin / config.posterior_temperature
            )
            confidence = (
                (1.0 - fine.normalized_entropy).clamp(0.0, 1.0)
                * margin_confidence
                * (best_evidence.to(torch.float32) / config.best_view_count)
            )
            candidate_depths[selected_indices] = best_depth
            candidate_sigmas[selected_indices] = sigma
            candidate_means[selected_indices] = best_mean
            candidate_scores[selected_indices] = confidence
            candidate_valid[selected_indices] = eligible
            candidate_entropy[selected_indices] = fine.normalized_entropy
            candidate_margin[selected_indices] = fine.margin
            candidate_evidence[selected_indices] = best_evidence
            processed += int(selected_indices.numel())
            if progress_callback is not None:
                progress_callback(processed, candidate_count)
            del coarse_costs, coarse_valid, coarse, fine_costs, fine_valid, fine
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    scoring_seconds = time.perf_counter() - scoring_started

    reciprocal_started = time.perf_counter()
    reciprocal_support = _candidate_reciprocal_support(
        candidate_means,
        candidate_depths,
        candidate_valid,
        view_ids,
        xy,
        cameras,
        neighbors,
        extent,
        config,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    reciprocal_seconds = time.perf_counter() - reciprocal_started
    primary = candidate_valid
    if config.apply_reciprocal:
        primary = primary & (reciprocal_support >= config.min_reciprocal_views)
    base_eligible = candidate_valid
    if int(base_eligible.sum()) < carve_config.n_init_3d:
        raise ValueError(
            "ray posterior found fewer descriptor-supported candidates than n_init_3d"
        )
    selection_score = (
        candidate_scores
        + 0.05 * reciprocal_support.to(candidate_scores)
        + 10.0 * primary.to(candidate_scores)
    )
    if config.apply_reciprocal and not config.allow_confidence_fallback:
        selection_eligible = primary
    else:
        selection_eligible = base_eligible
    if int(selection_eligible.sum()) < carve_config.n_init_3d:
        raise ValueError("ray-posterior primary/fallback policy cannot fill n_init_3d")
    selected = _balanced_topk(
        selection_score.cpu(),
        selection_eligible.cpu(),
        view_ids_cpu,
        carve_config.n_init_3d,
        inputs.n_views,
    ).to(device)
    selected_fallback = ~primary[selected]
    primary_fraction = 1.0 - float(selected_fallback.float().mean())
    if primary_fraction + 1e-12 < config.min_primary_fraction:
        primary_count = int((~selected_fallback).sum())
        raise ValueError(
            "ray-posterior selected primary fraction "
            f"{primary_count}/{selected.numel()}={primary_fraction:.6f} falls below "
            f"min_primary_fraction={config.min_primary_fraction:.6f}"
        )

    color_parts = []
    for source in range(inputs.n_views):
        output_mask = view_ids[selected] == source
        local_indices = output_mask.nonzero(as_tuple=True)[0]
        if local_indices.numel() == 0:
            continue
        color, alpha, valid = views[source].query_backend.query_appearance(
            xy[selected[local_indices]]
        )
        if not bool((alpha & valid).all()):
            raise RuntimeError("selected source radiance query left packet alpha/valid support")
        color_parts.append((local_indices, color.to(device).clamp(0.0, 1.0)))
    colors = torch.empty(selected.numel(), 3, device=device, dtype=torch.float32)
    for local_indices, color in color_parts:
        colors[local_indices] = color

    selected_cpu = selected.cpu()
    means_cpu = candidate_means[selected].cpu().to(dtype)
    depths_cpu = candidate_depths[selected].cpu().to(dtype)
    sigmas_cpu = candidate_sigmas[selected].cpu().to(dtype)
    selected_covariances = torch.empty(selected.numel(), 3, 3, dtype=dtype)
    selected_ray_sigmas = torch.empty(selected.numel(), dtype=dtype)
    for source in view_ids_cpu[selected_cpu].unique(sorted=True).tolist():
        output_mask = view_ids_cpu[selected_cpu] == source
        selected_candidates = selected_cpu[output_mask]
        local_xy = xy_cpu[selected_candidates]
        covariance_2d = _component_covariances(
            inputs.observations[source], component_ids_cpu[selected_candidates]
        )
        _, directions = inputs.cameras[source].pixel_rays(local_xy)
        selected_ray_sigmas[output_mask] = sigmas_cpu[output_mask] * directions.to(dtype).norm(
            dim=-1
        )
        selected_covariances[output_mask] = lift_covariance(
            inputs.cameras[source],
            local_xy,
            covariance_2d.to(local_xy),
            depths_cpu[output_mask].to(local_xy),
            selected_ray_sigmas[output_mask].to(local_xy),
        ).to(dtype)
    opacity = torch.full(
        (selected.numel(),), carve_config.init_opacity, dtype=dtype
    )
    gaussians = Gaussians3D.from_means_covs(
        means=means_cpu,
        covs=selected_covariances,
        colors=colors.cpu().to(dtype),
        opacity=opacity,
        sh_degree=carve_config.sh_degree,
    )
    cover_diagnostics = None
    cover_seconds = 0.0
    raw_gaussians = gaussians
    if config.apply_surface_cover:
        cover_started = time.perf_counter()
        covered = reconcile_covariances(
            gaussians,
            SurfelInitConfig(
                isotropic=config.surface_cover_isotropic,
                use_resolution_floor=False,
                min_planarity=config.surface_cover_min_planarity,
            ),
        )
        cover_seconds = time.perf_counter() - cover_started
        cover_diagnostics = covered.diagnostics
        gaussians = covered.gaussians
        if not torch.equal(gaussians.means, raw_gaussians.means):
            raise RuntimeError("surface-cover reconciliation changed posterior means")
        if not torch.equal(gaussians.sh, raw_gaussians.sh):
            raise RuntimeError("surface-cover reconciliation changed posterior radiance")

    lineage = CompactLineage(
        source_view_indices=view_ids_cpu[selected_cpu].clone(),
        source_component_indices=component_ids_cpu[selected_cpu].clone(),
        source_xy=xy_cpu[selected_cpu].clone(),
    )
    diagnostics = {
        "schema": "structsplat.occlusion_aware_ray_posterior.v1",
        "ownership": {
            "ray_proposals": "codec_sparse_structural_measure",
            "depth_likelihood": "source_excluded_packet_features_with_dustbin",
            "consistency": "independently_selected_reciprocal_ray_modes",
            "radiance": "selected_source_packet_continuous_appearance",
            "render_extent": (
                "local_surface_cover"
                if config.apply_surface_cover
                else "posterior_localization_covariance"
            ),
        },
        "bounds_source": _bounds_source(inputs),
        "bounds_center": center_cpu.tolist(),
        "bounds_extent": extent,
        "candidate_count": candidate_count,
        "proposed_candidates_per_view": proposed_per_view,
        "anchor_attempt_count": anchor_attempts,
        "selected_count": int(selected.numel()),
        "primary_selected_count": int((~selected_fallback).sum()),
        "fallback_selected_count": int(selected_fallback.sum()),
        "primary_selected_fraction": primary_fraction,
        "neighbor_indices": [list(row) for row in neighbors],
        "coarse_samples": carve_config.samples_per_ray,
        "fine_samples": config.fine_samples,
        "source_excluded_target_views": config.target_views,
        "best_view_count": config.best_view_count,
        "min_evidence_views": config.min_evidence_views,
        "dustbin_cost": config.dustbin_cost,
        "view_dispersion_weight": config.view_dispersion_weight,
        "posterior_temperature": config.posterior_temperature,
        "apply_reciprocal": config.apply_reciprocal,
        "min_reciprocal_views": config.min_reciprocal_views,
        "candidate_valid_count": int(candidate_valid.sum()),
        "candidate_primary_count": int(primary.sum()),
        "candidate_entropy": _distribution(candidate_entropy[candidate_valid]),
        "candidate_margin": _distribution(candidate_margin[candidate_valid]),
        "candidate_evidence_views": _distribution(
            candidate_evidence[candidate_valid].float()
        ),
        "candidate_reciprocal_support": _distribution(
            reciprocal_support[candidate_valid].float()
        ),
        "selected_depth": _distribution(candidate_depths[selected]),
        "selected_depth_sigma": _distribution(candidate_sigmas[selected]),
        "selected_score": _distribution(candidate_scores[selected]),
        "feature_receipt": features.diagnostics,
        "feature_seconds_shared": features.diagnostics.get("seconds"),
        "scoring_seconds": scoring_seconds,
        "reciprocal_seconds": reciprocal_seconds,
        "surface_cover_seconds": cover_seconds,
        "surface_cover": cover_diagnostics,
        "placement_seconds_excluding_shared_features": time.perf_counter() - started,
        "structural_index_pairs_evaluated": 0,
    }
    initialization = CompactInitializationResult(
        gaussians=gaussians,
        lineage=lineage,
        depths=depths_cpu,
        depth_sigmas=sigmas_cpu,
        ray_sigmas=selected_ray_sigmas,
        scores=candidate_scores[selected].cpu().to(dtype),
        diagnostics=diagnostics,
    )
    raw_initialization = replace(initialization, gaussians=raw_gaussians)
    return RayPosteriorLiftResult(
        initialization=initialization,
        raw_initialization=raw_initialization,
        selected_candidate_indices=selected_cpu,
        selected_fallback_mask=selected_fallback.cpu(),
        candidate_depths=candidate_depths.detach().cpu(),
        candidate_scores=candidate_scores.detach().cpu(),
        candidate_reciprocal_support=reciprocal_support.detach().cpu(),
        diagnostics=diagnostics,
    )


__all__ = [
    "DepthPosteriorSummary",
    "PacketFeatureSet",
    "PacketFeatureView",
    "RayPosteriorConfig",
    "RayPosteriorLiftResult",
    "build_packet_feature_pyramids",
    "initialize_occlusion_aware_ray_posterior",
    "robust_depth_posterior",
]
