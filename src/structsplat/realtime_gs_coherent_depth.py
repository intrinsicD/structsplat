"""Packet-only calibrated coherent-depth initialization for realtime-gs (CORE-019).

This module is deliberately optional and default-off. Importing it requires only NumPy and the
Python standard library; torch, scipy, safetensors, VGGT, and realtime-gs are imported lazily by
the functions that execute those boundaries.

The geometry contract is intentionally narrower than a general reconstruction system:

* four-view feed-forward predictions provide one spatially coherent depth field per group;
* one Sim(3) per group transfers only metric scale to the known calibrated cameras;
* known camera rays own every back-projected point;
* fused target depth classifies support, compatible occlusion, and free-space contradiction;
* feature-aware weighted sample elimination compiles redundant proposals to an exact budget,
  after which bounded cross-view micro-contraction may move selected survivors continuously; and
* realtime-gs owns the final Gaussian container while a compatibility-gated depth-normal cover
  sets render extent without geometry-only cross-surface neighborhoods.

No reporting image, render, optimizer state, or source RGB is accepted by this API. Packet
appearance and calibration are the complete construction inputs.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import heapq
import inspect
import math
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Callable, Sequence

import numpy as np

from .realtime_gs_adapter import RealtimeGSCodecNativeView


VGGT_SOURCE_REVISION = "a288dd0f14786c93483e45524328726ab7b1b4ce"
VGGT_MODEL_REVISION = "860abec7937da0a4c03c41d3c269c366e82abdf9"
VGGT_MODEL_BYTES = 5_026_367_224
VGGT_MODEL_SHA256 = "f164acf60724910d8fe1578bb499d800850c7bb0948db7555c413f9fbe60467e"


def _positive_float(value: object, name: str, *, allow_zero: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise TypeError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or (result < 0.0 if allow_zero else result <= 0.0):
        relation = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be finite and {relation}")
    return result


def _integer(value: object, name: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


@dataclass(frozen=True)
class CoherentDepthConfig:
    """Frozen controls for inference, fusion, evidence, contraction, and elimination."""

    target_count: int = 10_000
    candidate_multiplier: int = 4
    structural_fraction: float = 0.80
    seed: int = 0
    max_anchor_rounds: int = 8

    group_size: int = 4
    inference_max_side: int = 392
    oom_fallback_max_side: int = 336
    patch_size: int = 14
    inference_device: str = "cuda"
    feature_query_chunk: int = 262_144

    vggt_source_root: str = "/home/alex/Documents/vggt"
    vggt_source_revision: str = VGGT_SOURCE_REVISION
    checkpoint_bytes: int = VGGT_MODEL_BYTES
    checkpoint_sha256: str = VGGT_MODEL_SHA256

    max_group_center_rmse_fraction: float = 0.08
    max_group_loo_median_fraction: float = 0.15
    max_group_orientation_median_deg: float = 25.0
    max_group_focal_relative_median: float = 0.30
    min_group_finite_depth_fraction: float = 0.99

    fusion_relative_outlier: float = 0.15
    fusion_mad_multiplier: float = 3.0
    fusion_relative_sigma_floor: float = 0.005
    min_fusion_observations: int = 1

    support_views: int = 6
    support_relative_tolerance: float = 0.08
    support_uncertainty_multiplier: float = 2.0
    min_source_confidence: float = 0.05
    min_target_confidence: float = 0.05
    min_support_views: int = 1
    max_contradictions: int = 1
    min_normal_cosine: float = 0.50
    bounds_half_extent_multiplier: float = 0.65
    allow_fallback: bool = True
    min_primary_fraction: float = 0.75

    apply_contraction: bool = True
    contraction_radius_diameter_fraction: float = 0.001
    contraction_normal_cosine: float = 0.95
    contraction_rgb_barrier: float = 0.08
    contraction_max_cluster_size: int = 2

    apply_wse: bool = True
    wse_neighbors: int = 24
    wse_alpha: float = 8.0
    wse_rgb_barrier: float = 0.15
    wse_normal_cosine: float = 0.70
    wse_feature_protection: float = 3.0
    wse_view_floor_fraction: float = 0.50
    wse_anchor_fraction: float = 0.15
    wse_anchor_radius_diameter_fraction: float = 0.002

    init_opacity: float = 0.10
    normal_flatness: float = 0.30
    apply_surface_cover: bool = True
    surface_cover_neighbors: int = 24
    surface_cover_spacing_neighbors: int = 3
    surface_cover_sigma_ratio: float = 0.50
    surface_cover_target_alpha: float = 0.90
    surface_cover_min_opacity: float = 0.02
    surface_cover_max_opacity: float = 0.95
    surface_cover_max_pixel_sigma: float = 2.0

    def __post_init__(self) -> None:
        for name, minimum in (
            ("target_count", 1),
            ("candidate_multiplier", 1),
            ("seed", 0),
            ("max_anchor_rounds", 1),
            ("group_size", 3),
            ("inference_max_side", 14),
            ("oom_fallback_max_side", 14),
            ("patch_size", 1),
            ("feature_query_chunk", 1),
            ("checkpoint_bytes", 1),
            ("min_fusion_observations", 1),
            ("support_views", 1),
            ("min_support_views", 0),
            ("max_contradictions", 0),
            ("contraction_max_cluster_size", 1),
            ("wse_neighbors", 1),
            ("surface_cover_neighbors", 1),
            ("surface_cover_spacing_neighbors", 1),
        ):
            object.__setattr__(self, name, _integer(getattr(self, name), name, minimum=minimum))
        if self.group_size != 4:
            raise ValueError("CORE-019 freezes group_size at four")
        if self.inference_max_side % self.patch_size != 0:
            raise ValueError("inference_max_side must be divisible by patch_size")
        if self.oom_fallback_max_side % self.patch_size != 0:
            raise ValueError("oom_fallback_max_side must be divisible by patch_size")
        if self.oom_fallback_max_side > self.inference_max_side:
            raise ValueError("OOM fallback cannot exceed the primary inference size")
        if self.inference_device not in {"cuda"}:
            raise ValueError("the pinned VGGT path currently requires inference_device='cuda'")
        for name in (
            "structural_fraction",
            "min_group_finite_depth_fraction",
            "min_source_confidence",
            "min_target_confidence",
            "min_normal_cosine",
            "min_primary_fraction",
            "contraction_normal_cosine",
            "contraction_rgb_barrier",
            "wse_rgb_barrier",
            "wse_normal_cosine",
            "wse_view_floor_fraction",
            "wse_anchor_fraction",
            "init_opacity",
            "normal_flatness",
            "surface_cover_target_alpha",
            "surface_cover_min_opacity",
            "surface_cover_max_opacity",
        ):
            value = _positive_float(getattr(self, name), name, allow_zero=True)
            if value > 1.0:
                raise ValueError(f"{name} must lie in [0,1]")
            object.__setattr__(self, name, value)
        if not 0.0 < self.structural_fraction <= 1.0:
            raise ValueError("structural_fraction must lie in (0,1]")
        if not 0.0 < self.init_opacity < 1.0:
            raise ValueError("init_opacity must lie in (0,1)")
        if not 0.0 < self.surface_cover_target_alpha < 1.0:
            raise ValueError("surface_cover_target_alpha must lie in (0,1)")
        if not 0.0 < self.surface_cover_min_opacity <= self.surface_cover_max_opacity < 1.0:
            raise ValueError("surface-cover opacity bounds must satisfy 0 < min <= max < 1")
        for name in (
            "max_group_center_rmse_fraction",
            "max_group_loo_median_fraction",
            "max_group_orientation_median_deg",
            "max_group_focal_relative_median",
            "fusion_relative_outlier",
            "fusion_mad_multiplier",
            "fusion_relative_sigma_floor",
            "support_relative_tolerance",
            "support_uncertainty_multiplier",
            "bounds_half_extent_multiplier",
            "contraction_radius_diameter_fraction",
            "wse_alpha",
            "wse_feature_protection",
            "wse_anchor_radius_diameter_fraction",
            "surface_cover_sigma_ratio",
            "surface_cover_max_pixel_sigma",
        ):
            object.__setattr__(self, name, _positive_float(getattr(self, name), name))
        if self.normal_flatness <= 0.0:
            raise ValueError("normal_flatness must be positive")
        if self.surface_cover_spacing_neighbors > self.surface_cover_neighbors:
            raise ValueError("surface_cover_spacing_neighbors cannot exceed surface_cover_neighbors")
        for name in ("allow_fallback", "apply_contraction", "apply_wse", "apply_surface_cover"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be bool")
        if not isinstance(self.vggt_source_root, str) or not self.vggt_source_root:
            raise ValueError("vggt_source_root must be a non-empty path string")
        digest_lengths = {"vggt_source_revision": 40, "checkpoint_sha256": 64}
        for name, length in digest_lengths.items():
            value = getattr(self, name)
            if not isinstance(value, str) or len(value) != length:
                raise ValueError(f"{name} has the wrong digest length")


@dataclass(frozen=True)
class SimilarityAlignment:
    scale: float
    rotation: np.ndarray
    translation: np.ndarray
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class CoherentDepthField:
    """Fused packet-derived construction state, stored on CPU."""

    images: Any  # torch (V,3,H,W)
    depth: Any  # torch (V,H,W), known-camera z depth
    uncertainty: Any  # torch (V,H,W), robust absolute z uncertainty
    confidence: Any  # torch (V,H,W), normalized [0,1]
    normals: Any  # torch (V,H,W,3), known-world directions
    groups: tuple[tuple[int, ...], ...]
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class ProjectiveClassification:
    support: Any
    occluded: Any
    contradiction: Any
    invalid: Any
    tolerance: Any


@dataclass(frozen=True)
class WeightedEliminationResult:
    selected_indices: np.ndarray
    initial_crowding: np.ndarray
    final_crowding: np.ndarray
    removal_order: np.ndarray
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class CoherentDepthLiftResult:
    initialization: Any
    raw_initialization: Any
    field: CoherentDepthField
    selected_candidate_indices: Any
    selected_fallback_mask: Any
    diagnostics: dict[str, Any]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_revision(path: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _distribution(values: Any) -> dict[str, float | int | None]:
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = array[np.isfinite(array)]
    result: dict[str, float | int | None] = {
        "count": int(array.size),
        "finite_count": int(finite.size),
        "min": None,
        "median": None,
        "mean": None,
        "p90": None,
        "p99": None,
        "max": None,
    }
    if finite.size:
        result.update(
            {
                "min": float(finite.min()),
                "median": float(np.median(finite)),
                "mean": float(finite.mean()),
                "p90": float(np.quantile(finite, 0.90)),
                "p99": float(np.quantile(finite, 0.99)),
                "max": float(finite.max()),
            }
        )
    return result


def umeyama_similarity(source: np.ndarray, target: np.ndarray) -> SimilarityAlignment:
    """Least-squares positive-scale Sim(3) from corresponding ``source`` to ``target`` points."""

    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("source and target must share shape (N,3)")
    if source.shape[0] < 3 or not np.isfinite(source).all() or not np.isfinite(target).all():
        raise ValueError("similarity alignment needs at least three finite correspondences")
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    covariance = target_centered.T @ source_centered / source.shape[0]
    u, singular, vh = np.linalg.svd(covariance)
    sign = np.ones(3, dtype=np.float64)
    if np.linalg.det(u @ vh) < 0.0:
        sign[-1] = -1.0
    rotation = u @ np.diag(sign) @ vh
    variance = float(np.square(source_centered).sum() / source.shape[0])
    if not math.isfinite(variance) or variance <= 1e-12:
        raise ValueError("source camera centers are degenerate")
    scale = float(np.dot(singular, sign) / variance)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("similarity scale must be finite and positive")
    translation = target_mean - scale * (rotation @ source_mean)
    aligned = scale * (source @ rotation.T) + translation
    residual = np.linalg.norm(aligned - target, axis=1)
    diameter = float(np.linalg.norm(target[:, None] - target[None, :], axis=-1).max())
    if not math.isfinite(diameter) or diameter <= 1e-12:
        raise ValueError("target camera centers are degenerate")
    return SimilarityAlignment(
        scale=scale,
        rotation=rotation,
        translation=translation,
        diagnostics={
            "center_residual": _distribution(residual),
            "center_rmse": float(np.sqrt(np.mean(np.square(residual)))),
            "center_rmse_over_diameter": float(np.sqrt(np.mean(np.square(residual))) / diameter),
            "target_diameter": diameter,
        },
    )


def _rotation_error_degrees(first: np.ndarray, second: np.ndarray) -> float:
    cosine = np.clip((np.trace(first.T @ second) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def align_predicted_group(
    predicted_extrinsics: np.ndarray,
    predicted_intrinsics: np.ndarray,
    cameras: Sequence[Any],
) -> SimilarityAlignment:
    """Align one predicted camera group and attach leave-one-out/orientation/focal diagnostics."""

    extrinsics = np.asarray(predicted_extrinsics, dtype=np.float64)
    intrinsics = np.asarray(predicted_intrinsics, dtype=np.float64)
    if extrinsics.shape != (len(cameras), 3, 4):
        raise ValueError("predicted_extrinsics must have shape (V,3,4)")
    if intrinsics.shape != (len(cameras), 3, 3):
        raise ValueError("predicted_intrinsics must have shape (V,3,3)")
    predicted_rotations = extrinsics[:, :3, :3]
    predicted_translations = extrinsics[:, :3, 3]
    predicted_centers = np.stack(
        [
            -rotation.T @ translation
            for rotation, translation in zip(predicted_rotations, predicted_translations, strict=True)
        ]
    )
    known_centers = np.stack([camera.position.detach().double().cpu().numpy() for camera in cameras])
    alignment = umeyama_similarity(predicted_centers, known_centers)

    orientation = []
    for predicted_rotation, camera in zip(predicted_rotations, cameras, strict=True):
        aligned_camera_to_world = alignment.rotation @ predicted_rotation.T
        known_camera_to_world = camera.R.detach().double().cpu().numpy().T
        orientation.append(_rotation_error_degrees(aligned_camera_to_world, known_camera_to_world))
    focal = []
    for intrinsic, camera in zip(intrinsics, cameras, strict=True):
        focal.extend(
            [
                abs(float(intrinsic[0, 0]) - float(camera.fx)) / float(camera.fx),
                abs(float(intrinsic[1, 1]) - float(camera.fy)) / float(camera.fy),
            ]
        )
    loo = []
    if len(cameras) >= 4:
        for heldout in range(len(cameras)):
            retained = [index for index in range(len(cameras)) if index != heldout]
            local = umeyama_similarity(predicted_centers[retained], known_centers[retained])
            estimate = (
                local.scale * (predicted_centers[heldout] @ local.rotation.T) + local.translation
            )
            loo.append(float(np.linalg.norm(estimate - known_centers[heldout])))
    diameter = float(alignment.diagnostics["target_diameter"])
    diagnostics = {
        **alignment.diagnostics,
        "leave_one_out_center_error": _distribution(loo),
        "leave_one_out_median_over_diameter": (
            float(np.median(loo) / diameter) if loo else None
        ),
        "orientation_error_degrees": _distribution(orientation),
        "focal_relative_error": _distribution(focal),
        "predicted_centers": predicted_centers.tolist(),
        "known_centers": known_centers.tolist(),
    }
    return replace(alignment, diagnostics=diagnostics)


def build_calibration_groups(cameras: Sequence[Any], group_size: int = 4) -> tuple[tuple[int, ...], ...]:
    """Return one deterministic local-baseline group per construction camera."""

    if isinstance(group_size, bool) or not isinstance(group_size, int) or group_size != 4:
        raise ValueError("group_size must be exactly four")
    if len(cameras) < group_size:
        raise ValueError("at least four construction cameras are required")
    centers = np.stack([camera.position.detach().double().cpu().numpy() for camera in cameras])
    if not np.isfinite(centers).all():
        raise ValueError("camera centers must be finite")
    distance = np.linalg.norm(centers[:, None] - centers[None, :], axis=-1)
    groups = []
    for anchor in range(len(cameras)):
        order = sorted(
            (index for index in range(len(cameras)) if index != anchor),
            key=lambda index: (float(distance[anchor, index]), index),
        )
        groups.append((anchor, *order[: group_size - 1]))
    if set(index for group in groups for index in group) != set(range(len(cameras))):
        raise RuntimeError("calibration grouping failed to cover every construction camera")
    return tuple(groups)


def _target_shape(width: int, height: int, max_side: int, patch_size: int) -> tuple[int, int]:
    scale = max_side / max(width, height)
    output_width = max(patch_size, round(width * scale / patch_size) * patch_size)
    output_height = max(patch_size, round(height * scale / patch_size) * patch_size)
    return output_width, output_height


def _rescaled_camera(camera: Any, width: int, height: int) -> Any:
    """Return the same calibrated pose with intrinsics expressed on the inference raster."""

    scale_x = width / int(camera.width)
    scale_y = height / int(camera.height)
    return type(camera)(
        fx=float(camera.fx) * scale_x,
        fy=float(camera.fy) * scale_y,
        cx=float(camera.cx) * scale_x,
        cy=float(camera.cy) * scale_y,
        width=width,
        height=height,
        R=camera.R,
        t=camera.t,
    )


def build_packet_inference_images(
    views: Sequence[RealtimeGSCodecNativeView],
    config: CoherentDepthConfig,
    *,
    max_side: int | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Query packet appearance directly onto the bounded VGGT grid without structural-index work."""

    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("packet inference-image construction requires torch") from exc
    if not views:
        raise ValueError("views must be non-empty")
    for index, view in enumerate(views):
        if not isinstance(view, RealtimeGSCodecNativeView):
            raise TypeError(f"views[{index}] must be RealtimeGSCodecNativeView")
        if not hasattr(view.query_backend, "query_appearance"):
            raise TypeError(f"views[{index}] lacks the packet appearance-only query seam")
    selected_max_side = config.inference_max_side if max_side is None else int(max_side)
    canvas = [(int(view.structural_field.width), int(view.structural_field.height)) for view in views]
    shapes = [_target_shape(width, height, selected_max_side, config.patch_size) for width, height in canvas]
    if len(set(shapes)) != 1:
        raise ValueError("all packet views must resolve to one common inference shape")
    output_width, output_height = shapes[0]
    outputs = []
    per_view = []
    pairs_before = [
        int(getattr(view.query_backend.structural_backend, "total_pairs_evaluated", 0))
        for view in views
    ]
    started = time.perf_counter()
    for index, (view, (width, height)) in enumerate(zip(views, canvas, strict=True)):
        query_device = view.query_backend._appearance.device
        yy, xx = torch.meshgrid(
            torch.arange(output_height, device=query_device, dtype=torch.float32),
            torch.arange(output_width, device=query_device, dtype=torch.float32),
            indexing="ij",
        )
        xy = torch.stack(
            [
                (xx.reshape(-1) + 0.5) * width / output_width,
                (yy.reshape(-1) + 0.5) * height / output_height,
            ],
            dim=-1,
        )
        colors = []
        alphas = []
        valids = []
        view_started = time.perf_counter()
        for start in range(0, xy.shape[0], config.feature_query_chunk):
            color, alpha, valid = view.query_backend.query_appearance(
                xy[start : start + config.feature_query_chunk]
            )
            colors.append(color.detach().cpu())
            alphas.append(alpha.detach().cpu())
            valids.append(valid.detach().cpu())
        rgb = torch.cat(colors).reshape(output_height, output_width, 3)
        alpha = (torch.cat(alphas) & torch.cat(valids)).reshape(output_height, output_width)
        rgb = torch.where(alpha[..., None], rgb, torch.zeros_like(rgb)).clamp(0.0, 1.0)
        outputs.append(rgb.permute(2, 0, 1).contiguous())
        per_view.append(
            {
                "index": index,
                "canvas": [width, height],
                "inference_shape": [output_height, output_width],
                "foreground_fraction": float(alpha.float().mean()),
                "seconds": time.perf_counter() - view_started,
            }
        )
    pairs_after = [
        int(getattr(view.query_backend.structural_backend, "total_pairs_evaluated", 0))
        for view in views
    ]
    if pairs_after != pairs_before:
        raise RuntimeError("packet inference-image construction queried the structural index")
    images = torch.stack(outputs).float()
    return images, {
        "schema": "structsplat.packet_vggt_images.v1",
        "source_rgb_opened": False,
        "selected_max_side": selected_max_side,
        "shape": list(images.shape),
        "structural_pairs_before": pairs_before,
        "structural_pairs_after": pairs_after,
        "seconds": time.perf_counter() - started,
        "views": per_view,
    }


class _PinnedVGGTPredictor:
    """One verified, mixed-precision VGGT instance reused across all groups."""

    def __init__(self, checkpoint: Path, config: CoherentDepthConfig) -> None:
        try:
            import torch
            from safetensors import safe_open
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("VGGT inference requires torch and safetensors") from exc
        if not torch.cuda.is_available():
            raise RuntimeError("VGGT inference requires CUDA")
        source_root = Path(config.vggt_source_root).expanduser().resolve()
        if _git_revision(source_root) != config.vggt_source_revision:
            raise RuntimeError("VGGT source revision does not match the frozen receipt")
        checkpoint = checkpoint.expanduser().resolve()
        if not checkpoint.is_file() or checkpoint.stat().st_size != config.checkpoint_bytes:
            raise RuntimeError("VGGT checkpoint byte count does not match the frozen receipt")
        digest_started = time.perf_counter()
        checkpoint_digest = _sha256(checkpoint)
        digest_seconds = time.perf_counter() - digest_started
        if checkpoint_digest != config.checkpoint_sha256:
            raise RuntimeError("VGGT checkpoint SHA-256 does not match the frozen receipt")

        source_text = str(source_root)
        inserted = source_text not in sys.path
        if inserted:
            sys.path.insert(0, source_text)
        try:
            from vggt.models.vggt import VGGT
            from vggt.utils.pose_enc import pose_encoding_to_extri_intri
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("the pinned VGGT source is not importable") from exc
        finally:
            if inserted:
                sys.path.remove(source_text)
        implementation = Path(inspect.getfile(VGGT)).resolve()
        if source_root not in implementation.parents:
            raise RuntimeError("imported VGGT implementation is outside the pinned source root")

        construction_started = time.perf_counter()
        model = VGGT(enable_point=False, enable_track=False).eval()
        construction_seconds = time.perf_counter() - construction_started
        model.aggregator.to(dtype=torch.bfloat16)
        model.camera_head.to(dtype=torch.float32)
        model.depth_head.to(dtype=torch.float32)
        state = model.state_dict()
        load_started = time.perf_counter()
        with torch.no_grad(), safe_open(checkpoint, framework="pt", device="cpu") as stream:
            keys = set(stream.keys())
            missing = sorted(set(state) - keys)
            if missing:
                raise RuntimeError(f"VGGT checkpoint is missing {len(missing)} enabled tensors")
            for name, destination in state.items():
                source = stream.get_tensor(name)
                if source.shape != destination.shape:
                    raise RuntimeError(f"VGGT checkpoint tensor shape mismatch for {name}")
                destination.copy_(source)
            unused = sorted(keys - set(state))
        load_seconds = time.perf_counter() - load_started
        model = model.cuda()
        torch.cuda.synchronize()
        dtype_parameters: dict[str, int] = {}
        dtype_bytes: dict[str, int] = {}
        for parameter in model.parameters():
            name = str(parameter.dtype).removeprefix("torch.")
            dtype_parameters[name] = dtype_parameters.get(name, 0) + parameter.numel()
            dtype_bytes[name] = dtype_bytes.get(name, 0) + parameter.numel() * parameter.element_size()
        self._torch = torch
        self._pose_decoder = pose_encoding_to_extri_intri
        self.model = model
        self.receipt = {
            "source_root": str(source_root),
            "source_revision": config.vggt_source_revision,
            "source_implementation": str(implementation),
            "model_id": "facebook/VGGT-1B",
            "model_revision": VGGT_MODEL_REVISION,
            "checkpoint": {
                "path": str(checkpoint),
                "bytes": checkpoint.stat().st_size,
                "sha256": checkpoint_digest,
                "hash_seconds": digest_seconds,
            },
            "license": "CC-BY-NC-4.0 public research checkpoint",
            "enabled_tensor_count": len(state),
            "unused_disabled_head_tensor_count": len(unused),
            "construction_seconds": construction_seconds,
            "load_seconds": load_seconds,
            "parameter_count_by_dtype": dtype_parameters,
            "parameter_bytes_by_dtype": dtype_bytes,
            "precision": "bf16 aggregator; fp32 camera/depth heads; bf16 CUDA autocast",
        }

    def __call__(self, images: Any, group: tuple[int, ...]) -> dict[str, Any]:
        del group
        torch = self._torch
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        value = images.cuda(non_blocking=False)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            prediction = self.model(value)
        torch.cuda.synchronize()
        extrinsic, intrinsic = self._pose_decoder(prediction["pose_enc"].float(), value.shape[-2:])
        output = {
            "depth": prediction["depth"].float().cpu().numpy().squeeze(0).squeeze(-1),
            "confidence": prediction["depth_conf"].float().cpu().numpy().squeeze(0),
            "extrinsic": extrinsic.float().cpu().numpy().squeeze(0),
            "intrinsic": intrinsic.float().cpu().numpy().squeeze(0),
            "seconds": time.perf_counter() - started,
            "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        }
        del prediction, value, extrinsic, intrinsic
        torch.cuda.empty_cache()
        return output

    def close(self) -> None:
        torch = self._torch
        self.model = self.model.cpu()
        del self.model
        torch.cuda.empty_cache()


def _normalize_confidence(confidence: np.ndarray) -> np.ndarray:
    value = np.asarray(confidence, dtype=np.float64)
    result = np.zeros_like(value, dtype=np.float32)
    finite = np.isfinite(value) & (value > 0.0)
    if not finite.any():
        return result
    transformed = np.log1p(np.maximum(value, 0.0))
    for index in range(value.shape[0]):
        local = transformed[index][finite[index]]
        if not local.size:
            continue
        lo, hi = np.quantile(local, [0.05, 0.95])
        if float(hi - lo) <= np.finfo(np.float32).eps:
            result[index][finite[index]] = 1.0
            continue
        denominator = float(hi - lo)
        normalized = np.clip((transformed[index] - lo) / denominator, 0.0, 1.0)
        result[index] = np.where(finite[index], normalized, 0.0).astype(np.float32)
    return result


def _group_gate(alignment: SimilarityAlignment, depth: np.ndarray, config: CoherentDepthConfig):
    diagnostics = alignment.diagnostics
    finite_fraction = float(np.isfinite(depth).mean())
    checks = {
        "finite_depth_fraction": finite_fraction >= config.min_group_finite_depth_fraction,
        "center_rmse": float(diagnostics["center_rmse_over_diameter"])
        <= config.max_group_center_rmse_fraction,
        "leave_one_out_center": float(diagnostics["leave_one_out_median_over_diameter"])
        <= config.max_group_loo_median_fraction,
        "orientation": float(diagnostics["orientation_error_degrees"]["median"])
        <= config.max_group_orientation_median_deg,
        "focal": float(diagnostics["focal_relative_error"]["median"])
        <= config.max_group_focal_relative_median,
    }
    return checks, all(checks.values())


def _weighted_median(values: Any, weights: Any, dimension: int = 0) -> Any:
    import torch

    ranked, order = torch.sort(values, dim=dimension, stable=True)
    ranked_weights = torch.gather(weights, dimension, order)
    cumulative = torch.cumsum(ranked_weights, dim=dimension)
    total = ranked_weights.sum(dim=dimension, keepdim=True)
    selector = cumulative >= 0.5 * total
    first = selector.to(torch.int64).argmax(dim=dimension, keepdim=True)
    median = torch.gather(ranked, dimension, first).squeeze(dimension)
    return median


def fuse_overlapping_depths(
    estimates: Sequence[Sequence[tuple[Any, Any]]],
    config: CoherentDepthConfig,
) -> tuple[Any, Any, Any, dict[str, Any]]:
    """Robustly fuse ``(depth, confidence)`` maps supplied for every construction view."""

    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("depth fusion requires torch") from exc
    if not estimates or any(not view for view in estimates):
        raise ValueError("every construction view needs at least one accepted group estimate")
    fused_depth = []
    fused_uncertainty = []
    fused_confidence = []
    per_view = []
    for view_index, view_estimates in enumerate(estimates):
        depth = torch.stack([torch.as_tensor(item[0]).float() for item in view_estimates])
        weight = torch.stack([torch.as_tensor(item[1]).float() for item in view_estimates])
        valid = torch.isfinite(depth) & (depth > 0.0) & torch.isfinite(weight) & (weight > 0.0)
        safe_depth = torch.where(valid, depth, torch.full_like(depth, torch.inf))
        safe_weight = torch.where(valid, weight, torch.zeros_like(weight))
        count = valid.sum(dim=0)
        median = _weighted_median(safe_depth, safe_weight)
        absolute = (depth - median[None]).abs()
        mad = _weighted_median(
            torch.where(valid, absolute, torch.full_like(absolute, torch.inf)), safe_weight
        )
        tolerance = torch.maximum(
            config.fusion_relative_outlier * median.abs(),
            config.fusion_mad_multiplier * mad,
        )
        inlier = valid & (absolute <= tolerance[None])
        inlier_weight = torch.where(inlier, weight, torch.zeros_like(weight))
        denominator = inlier_weight.sum(dim=0)
        mean = (torch.where(inlier, depth, torch.zeros_like(depth)) * inlier_weight).sum(dim=0)
        mean = mean / denominator.clamp_min(torch.finfo(mean.dtype).tiny)
        residual = torch.where(inlier, (depth - mean[None]).abs(), torch.zeros_like(depth))
        robust_sigma = (residual * inlier_weight).sum(dim=0) / denominator.clamp_min(
            torch.finfo(mean.dtype).tiny
        )
        robust_sigma = torch.maximum(
            robust_sigma,
            config.fusion_relative_sigma_floor * mean.abs(),
        )
        observation_fraction = inlier.sum(dim=0).float() / float(len(view_estimates))
        normalized_weight = denominator / max(float(len(view_estimates)), 1.0)
        confidence = (observation_fraction * normalized_weight).clamp(0.0, 1.0)
        valid_output = (
            (count >= config.min_fusion_observations)
            & (denominator > 0.0)
            & torch.isfinite(mean)
            & (mean > 0.0)
        )
        mean = torch.where(valid_output, mean, torch.zeros_like(mean))
        robust_sigma = torch.where(valid_output, robust_sigma, torch.zeros_like(robust_sigma))
        confidence = torch.where(valid_output, confidence, torch.zeros_like(confidence))
        fused_depth.append(mean)
        fused_uncertainty.append(robust_sigma)
        fused_confidence.append(confidence)
        per_view.append(
            {
                "view_index": view_index,
                "estimate_count": len(view_estimates),
                "valid_fraction": float(valid_output.float().mean()),
                "depth": _distribution(mean[valid_output]),
                "relative_uncertainty": _distribution(
                    robust_sigma[valid_output] / mean[valid_output].clamp_min(1e-8)
                ),
                "confidence": _distribution(confidence[valid_output]),
            }
        )
    return (
        torch.stack(fused_depth),
        torch.stack(fused_uncertainty),
        torch.stack(fused_confidence),
        {"views": per_view},
    )


def _depth_normals(depth: Any, camera: Any, canvas: tuple[int, int]) -> Any:
    import torch
    import torch.nn.functional as functional

    height, width = depth.shape
    canvas_width, canvas_height = canvas
    yy, xx = torch.meshgrid(
        torch.arange(height, dtype=torch.float32),
        torch.arange(width, dtype=torch.float32),
        indexing="ij",
    )
    uv = torch.stack(
        [
            (xx.reshape(-1) + 0.5) * canvas_width / width,
            (yy.reshape(-1) + 0.5) * canvas_height / height,
        ],
        dim=-1,
    )
    points = camera.unproject(uv, depth.reshape(-1)).reshape(height, width, 3)
    padded = functional.pad(points.permute(2, 0, 1)[None], (1, 1, 1, 1), mode="replicate")[0]
    dx = padded[:, 1:-1, 2:] - padded[:, 1:-1, :-2]
    dy = padded[:, 2:, 1:-1] - padded[:, :-2, 1:-1]
    normal = torch.linalg.cross(dx.permute(1, 2, 0), dy.permute(1, 2, 0), dim=-1)
    norm = normal.norm(dim=-1, keepdim=True)
    normal = normal / norm.clamp_min(1e-8)
    towards_camera = camera.position[None, None] - points
    flip = (normal * towards_camera).sum(dim=-1) < 0.0
    normal = torch.where(flip[..., None], -normal, normal)
    valid = torch.isfinite(normal).all(dim=-1) & torch.isfinite(depth) & (depth > 0.0) & (norm[..., 0] > 1e-8)
    return torch.where(valid[..., None], normal, torch.zeros_like(normal))


def infer_coherent_depth_field(
    inputs: Any,
    views: Sequence[RealtimeGSCodecNativeView],
    checkpoint: str | Path | None,
    config: CoherentDepthConfig | None = None,
    *,
    predictor: Callable[[Any, tuple[int, ...]], dict[str, Any]] | None = None,
) -> CoherentDepthField:
    """Predict, align, and fuse packet-only depth for every construction view."""

    config = config or CoherentDepthConfig()
    if len(views) != int(inputs.n_views):
        raise ValueError("views must contain one codec-native packet per construction input")
    for index, (view, field) in enumerate(zip(views, inputs.observations, strict=True)):
        if not isinstance(view, RealtimeGSCodecNativeView):
            raise TypeError(f"views[{index}] must be RealtimeGSCodecNativeView")
        if view.structural_field is not field:
            raise ValueError(f"views[{index}] does not own inputs.observations[{index}]")
    groups = build_calibration_groups(inputs.cameras, config.group_size)
    owned_predictor = predictor is None
    model_receipt: dict[str, Any] | None = None
    if predictor is None:
        if checkpoint is None:
            raise ValueError("checkpoint is required when predictor is not injected")
        predictor_object = _PinnedVGGTPredictor(Path(checkpoint), config)
        predictor = predictor_object
        model_receipt = predictor_object.receipt
    else:
        predictor_object = None
        model_receipt = {"injected_predictor": True}

    selected_max_side = config.inference_max_side
    fallback_reason = None
    inference_started = time.perf_counter()
    try:
        images, image_receipt = build_packet_inference_images(views, config)
        while True:
            estimates: list[list[tuple[Any, Any]]] = [[] for _ in views]
            group_records = []
            try:
                for group_index, group in enumerate(groups):
                    prediction = predictor(images[list(group)], group)
                    depth = np.asarray(prediction["depth"], dtype=np.float32)
                    confidence = np.asarray(prediction["confidence"], dtype=np.float32)
                    extrinsic = np.asarray(prediction["extrinsic"], dtype=np.float64)
                    intrinsic = np.asarray(prediction["intrinsic"], dtype=np.float64)
                    expected_depth_shape = (config.group_size, images.shape[-2], images.shape[-1])
                    if depth.shape != expected_depth_shape or confidence.shape != expected_depth_shape:
                        raise ValueError("predictor depth/confidence shape does not match its group")
                    inference_cameras = [
                        _rescaled_camera(
                            inputs.cameras[index], images.shape[-1], images.shape[-2]
                        )
                        for index in group
                    ]
                    alignment = align_predicted_group(
                        extrinsic,
                        intrinsic,
                        inference_cameras,
                    )
                    checks, accepted = _group_gate(alignment, depth, config)
                    normalized_confidence = _normalize_confidence(confidence)
                    record = {
                        "group_index": group_index,
                        "members": list(group),
                        "scale": alignment.scale,
                        "alignment": alignment.diagnostics,
                        "checks": checks,
                        "accepted": accepted,
                        "seconds": float(prediction.get("seconds", 0.0)),
                        "peak_cuda_allocated_bytes": int(
                            prediction.get("peak_cuda_allocated_bytes", 0)
                        ),
                        "peak_cuda_reserved_bytes": int(
                            prediction.get("peak_cuda_reserved_bytes", 0)
                        ),
                    }
                    group_records.append(record)
                    if not accepted:
                        continue
                    for local, view_index in enumerate(group):
                        estimates[view_index].append(
                            (alignment.scale * depth[local], normalized_confidence[local])
                        )
                missing = [index for index, item in enumerate(estimates) if not item]
                if missing:
                    raise RuntimeError(
                        f"accepted VGGT groups leave construction views without depth: {missing}"
                    )
                break
            except Exception as error:
                try:
                    import torch
                except ImportError:  # pragma: no cover
                    raise
                if (
                    isinstance(error, torch.OutOfMemoryError)
                    and selected_max_side == config.inference_max_side
                ):
                    fallback_reason = str(error)
                    torch.cuda.empty_cache()
                    selected_max_side = config.oom_fallback_max_side
                    images, image_receipt = build_packet_inference_images(
                        views, config, max_side=selected_max_side
                    )
                    continue
                raise
        depth, uncertainty, confidence, fusion_diagnostics = fuse_overlapping_depths(
            estimates, config
        )
        canvas = [
            (int(view.structural_field.width), int(view.structural_field.height)) for view in views
        ]
        normals = torch_stack(
            [_depth_normals(depth[index], inputs.cameras[index], canvas[index]) for index in range(len(views))]
        )
    finally:
        if owned_predictor and predictor_object is not None:
            predictor_object.close()
    diagnostics = {
        "schema": "structsplat.coherent_depth_field.v1",
        "source_rgb_opened": False,
        "reporting_views_present": False,
        "groups": group_records,
        "group_count": len(groups),
        "accepted_group_count": sum(bool(record["accepted"]) for record in group_records),
        "selected_max_side": selected_max_side,
        "oom_fallback_reason": fallback_reason,
        "packet_images": image_receipt,
        "model": model_receipt,
        "fusion": fusion_diagnostics,
        "total_seconds": time.perf_counter() - inference_started,
        "depth": _distribution(depth[depth > 0.0]),
        "relative_uncertainty": _distribution(
            uncertainty[depth > 0.0] / depth[depth > 0.0].clamp_min(1e-8)
        ),
        "confidence": _distribution(confidence),
    }
    return CoherentDepthField(
        images=images.cpu(),
        depth=depth.cpu(),
        uncertainty=uncertainty.cpu(),
        confidence=confidence.cpu(),
        normals=normals.cpu(),
        groups=groups,
        diagnostics=diagnostics,
    )


def torch_stack(values: Sequence[Any]) -> Any:
    """Keep torch optional at module import while avoiding repeated local boilerplate."""

    import torch

    return torch.stack(list(values))


def classify_projective_depth(
    projected_depth: Any,
    target_depth: Any,
    source_uncertainty: Any,
    target_uncertainty: Any,
    valid: Any,
    *,
    relative_tolerance: float,
    uncertainty_multiplier: float,
) -> ProjectiveClassification:
    """Classify calibrated target evidence without treating compatible occlusion as failure."""

    import torch

    tensors = (projected_depth, target_depth, source_uncertainty, target_uncertainty, valid)
    if not all(torch.is_tensor(value) for value in tensors):
        raise TypeError("projective classification inputs must be torch tensors")
    shape = projected_depth.shape
    if any(value.shape != shape for value in tensors[1:]):
        raise ValueError("projective classification inputs must share one shape")
    if valid.dtype != torch.bool:
        raise TypeError("valid must be bool")
    relative = _positive_float(relative_tolerance, "relative_tolerance")
    uncertainty = _positive_float(uncertainty_multiplier, "uncertainty_multiplier")
    finite = (
        valid
        & torch.isfinite(projected_depth)
        & torch.isfinite(target_depth)
        & torch.isfinite(source_uncertainty)
        & torch.isfinite(target_uncertainty)
        & (projected_depth > 0.0)
        & (target_depth > 0.0)
        & (source_uncertainty >= 0.0)
        & (target_uncertainty >= 0.0)
    )
    tolerance = torch.maximum(
        relative * torch.maximum(projected_depth.abs(), target_depth.abs()),
        uncertainty * (source_uncertainty + target_uncertainty),
    )
    difference = target_depth - projected_depth
    support = finite & (difference.abs() <= tolerance)
    occluded = finite & (difference < -tolerance)
    contradiction = finite & (difference > tolerance)
    return ProjectiveClassification(
        support=support,
        occluded=occluded,
        contradiction=contradiction,
        invalid=~finite,
        tolerance=tolerance,
    )


def _sample_map(value: Any, xy: Any, canvas: tuple[int, int], *, mode: str = "bilinear") -> Any:
    import torch
    import torch.nn.functional as functional

    width, height = canvas
    grid = torch.stack(
        [2.0 * xy[:, 0] / width - 1.0, 2.0 * xy[:, 1] / height - 1.0], dim=-1
    ).reshape(1, -1, 1, 2)
    if value.ndim == 2:
        source = value[None, None]
        sampled = functional.grid_sample(
            source.float(), grid, mode=mode, padding_mode="zeros", align_corners=False
        )[0, 0, :, 0]
    elif value.ndim == 3 and value.shape[-1] == 3:
        source = value.permute(2, 0, 1)[None]
        sampled = functional.grid_sample(
            source.float(), grid, mode=mode, padding_mode="zeros", align_corners=False
        )[0, :, :, 0].T
    elif value.ndim == 3:
        source = value[None]
        sampled = functional.grid_sample(
            source.float(), grid, mode=mode, padding_mode="zeros", align_corners=False
        )[0, :, :, 0].T
    else:
        raise ValueError("sampled map must be HW, HWC3, or CHW")
    return sampled


def _balanced_counts(total: int, groups: int) -> list[int]:
    base, remainder = divmod(total, groups)
    return [base + (index < remainder) for index in range(groups)]


def _sample_field_anchors(field: Any, count: int, generator: Any, max_rounds: int):
    import torch

    masses = field.amplitudes * (2.0 * math.pi) * field.effective_variances().prod(dim=1).sqrt()
    total_mass = masses.sum()
    if not bool(torch.isfinite(total_mass)) or float(total_mass) <= 0.0:
        raise ValueError("structural packet has no finite positive proposal mass")
    cumulative = masses.cumsum(0)
    points = []
    component_rows = []
    attempts = 0
    remaining = count
    for _ in range(max_rounds):
        if remaining <= 0:
            break
        draw_count = max(2 * remaining, 8)
        uniforms = torch.rand(draw_count, generator=generator, dtype=field.dtype) * total_mass
        components = torch.searchsorted(cumulative, uniforms, right=False).clamp_max(field.n - 1)
        normal = torch.randn(draw_count, 2, generator=generator, dtype=field.dtype)
        scales = field.effective_variances()[components].sqrt()
        local = normal * scales
        angle = field.rotations[components]
        cosine, sine = torch.cos(angle), torch.sin(angle)
        offsets = torch.stack(
            [
                cosine * local[:, 0] - sine * local[:, 1],
                sine * local[:, 0] + cosine * local[:, 1],
            ],
            dim=-1,
        )
        if field.mean_residuals is None:
            xy = field.means[components] + offsets
        else:
            fit_x, fit_y, _, _ = field.fit_window
            origin = torch.tensor([fit_x + 0.5, fit_y + 0.5], dtype=torch.float64)
            xy = field.local_means(components).double() + offsets.double() + origin
        base_weight = field.amplitudes[components] * torch.exp(-0.5 * normal.square().sum(dim=1))
        exact_weight = field.component_weight(xy, components)
        acceptance = (exact_weight / base_weight.clamp_min(torch.finfo(field.dtype).tiny)).clamp(
            0.0, 1.0
        )
        active = torch.rand(draw_count, generator=generator, dtype=field.dtype) < acceptance
        selected = active.nonzero(as_tuple=True)[0][:remaining]
        if selected.numel():
            points.append(xy[selected].float())
            component_rows.append(components[selected].long())
            remaining -= int(selected.numel())
        attempts += draw_count
    if remaining:
        raise RuntimeError("structural proposal sampler exhausted its bounded anchor rounds")
    return torch.cat(points), torch.cat(component_rows), attempts


def _image_feature_maps(images: Any) -> Any:
    import torch
    import torch.nn.functional as functional

    gray = 0.2126 * images[:, 0:1] + 0.7152 * images[:, 1:2] + 0.0722 * images[:, 2:3]
    kernel_x = images.new_tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
    )[None, None] / 8.0
    kernel_y = kernel_x.transpose(-1, -2)
    dx = functional.conv2d(gray, kernel_x, padding=1)
    dy = functional.conv2d(gray, kernel_y, padding=1)
    gradient = torch.sqrt(dx.square() + dy.square())[:, 0]
    flat = gradient.flatten(1)
    reference = torch.quantile(flat, 0.95, dim=1).clamp_min(1e-6)
    return (gradient / reference[:, None, None]).clamp(0.0, 1.0)


def _proposal_validity(
    inputs: Any,
    field: CoherentDepthField,
    view_index: int,
    xy: Any,
    config: CoherentDepthConfig,
) -> Any:
    """Cheap source-depth/bounds gate used to spend the fixed proposal budget productively."""

    import torch

    camera = inputs.cameras[view_index]
    canvas = (int(camera.width), int(camera.height))
    depth = _sample_map(field.depth[view_index], xy, canvas)
    confidence = _sample_map(field.confidence[view_index], xy, canvas)
    normals = _sample_map(field.normals[view_index], xy, canvas)
    means = camera.unproject(xy.float(), depth)
    center, extent = inputs.bounds_hint
    half = float(extent) * config.bounds_half_extent_multiplier
    return (
        torch.isfinite(depth)
        & (depth > 0.0)
        & torch.isfinite(confidence)
        & (confidence >= config.min_source_confidence)
        & torch.isfinite(normals).all(dim=-1)
        & (normals.norm(dim=-1) > 0.5)
        & torch.isfinite(means).all(dim=-1)
        & ((means - center.float()[None]).abs() <= half).all(dim=-1)
    )


def _propose_candidate_rays(inputs: Any, field: CoherentDepthField, config: CoherentDepthConfig):
    import torch

    total = config.target_count * config.candidate_multiplier
    structural_total = int(round(total * config.structural_fraction))
    structural_total = min(max(structural_total, config.target_count), total)
    cover_total = total - structural_total
    generator = torch.Generator(device="cpu").manual_seed(config.seed)
    view_rows = []
    component_rows = []
    point_rows = []
    kind_rows = []
    attempts = 0
    source_prefilter_rejected = 0
    structural_points: list[list[Any]] = [[] for _ in range(inputs.n_views)]
    structural_components: list[list[Any]] = [[] for _ in range(inputs.n_views)]
    structural_per_view = [0 for _ in range(inputs.n_views)]
    remaining_structural = structural_total
    for _ in range(config.max_anchor_rounds):
        if remaining_structural <= 0:
            break
        # The fixed total budget is balanced whenever views have evidence, but a camera whose
        # structural mass mostly observes room geometry outside the explicit reconstruction bounds
        # cannot veto the scene. Its shortfall is deterministically reallocated on later rounds.
        round_quota = max(math.ceil(remaining_structural / inputs.n_views), 1)
        for view_index, observation in enumerate(inputs.observations):
            if remaining_structural <= 0:
                break
            requested = min(round_quota, remaining_structural)
            draw_count = max(4 * requested, 128)
            xy, components, draw_attempts = _sample_field_anchors(
                observation,
                draw_count,
                generator,
                config.max_anchor_rounds,
            )
            valid = _proposal_validity(inputs, field, view_index, xy, config)
            selected = valid.nonzero(as_tuple=True)[0][:requested]
            if selected.numel():
                structural_points[view_index].append(xy[selected])
                structural_components[view_index].append(components[selected])
                accepted_count = int(selected.numel())
                structural_per_view[view_index] += accepted_count
                remaining_structural -= accepted_count
            source_prefilter_rejected += int((~valid).sum())
            attempts += draw_attempts
    if remaining_structural:
        raise RuntimeError(
            "construction views cannot fill the global structural coherent-depth proposal budget "
            f"after {config.max_anchor_rounds} bounded reallocation rounds; "
            f"missing={remaining_structural}"
        )
    for view_index, count in enumerate(structural_per_view):
        if count == 0:
            continue
        xy = torch.cat(structural_points[view_index])
        components = torch.cat(structural_components[view_index])
        view_rows.append(torch.full((count,), view_index, dtype=torch.long))
        component_rows.append(components)
        point_rows.append(xy)
        kind_rows.append(torch.zeros(count, dtype=torch.long))

    cover_counts = _balanced_counts(cover_total, inputs.n_views)
    feature_maps = _image_feature_maps(field.images)
    for view_index, count in enumerate(cover_counts):
        if count == 0:
            continue
        confidence = field.confidence[view_index]
        feature = feature_maps[view_index]
        probability = (0.25 + 0.50 * confidence + 0.25 * feature).flatten()
        height, width = confidence.shape
        yy, xx = torch.meshgrid(
            torch.arange(height, dtype=torch.float32),
            torch.arange(width, dtype=torch.float32),
            indexing="ij",
        )
        canvas_width = int(inputs.cameras[view_index].width)
        canvas_height = int(inputs.cameras[view_index].height)
        grid_xy = torch.stack(
            [
                (xx.reshape(-1) + 0.5) * canvas_width / width,
                (yy.reshape(-1) + 0.5) * canvas_height / height,
            ],
            dim=-1,
        )
        valid = _proposal_validity(inputs, field, view_index, grid_xy, config)
        valid &= torch.isfinite(probability)
        probability = torch.where(valid, probability, torch.zeros_like(probability))
        available = int((probability > 0.0).sum())
        if available < count:
            raise RuntimeError(
                f"view {view_index} cannot supply its bounded flat-cover proposal count"
            )
        indices = torch.multinomial(probability, count, replacement=False, generator=generator)
        yy = torch.div(indices, width, rounding_mode="floor").float()
        xx = (indices % width).float()
        jitter = torch.rand(count, 2, generator=generator) - 0.5
        xy = torch.stack(
            [
                (xx + 0.5 + 0.70 * jitter[:, 0]) * canvas_width / width,
                (yy + 0.5 + 0.70 * jitter[:, 1]) * canvas_height / height,
            ],
            dim=-1,
        )
        xy[:, 0].clamp_(0.5, canvas_width - 0.5)
        xy[:, 1].clamp_(0.5, canvas_height - 0.5)
        view_rows.append(torch.full((count,), view_index, dtype=torch.long))
        component_rows.append(torch.full((count,), -1, dtype=torch.long))
        point_rows.append(xy)
        kind_rows.append(torch.ones(count, dtype=torch.long))
    return (
        torch.cat(view_rows),
        torch.cat(component_rows),
        torch.cat(point_rows),
        torch.cat(kind_rows),
        {
            "total": total,
            "structural": structural_total,
            "cover": cover_total,
            "structural_anchor_attempts": attempts,
            "source_prefilter_rejected": source_prefilter_rejected,
            "structural_per_view": structural_per_view,
        },
    )


def _camera_neighbors(cameras: Sequence[Any], count: int) -> tuple[tuple[int, ...], ...]:
    centers = np.stack([camera.position.detach().double().cpu().numpy() for camera in cameras])
    distance = np.linalg.norm(centers[:, None] - centers[None, :], axis=-1)
    result = []
    for source in range(len(cameras)):
        order = sorted(
            (target for target in range(len(cameras)) if target != source),
            key=lambda target: (float(distance[source, target]), target),
        )
        result.append(tuple(order[: min(count, len(order))]))
    return tuple(result)


def _candidate_evidence(
    inputs: Any,
    views: Sequence[RealtimeGSCodecNativeView],
    field: CoherentDepthField,
    source_view: Any,
    source_component: Any,
    source_xy: Any,
    source_kind: Any,
    config: CoherentDepthConfig,
    *,
    apply_projective_support: bool,
) -> dict[str, Any]:
    import torch

    count = int(source_view.numel())
    means = torch.zeros(count, 3, dtype=torch.float32)
    depth = torch.zeros(count, dtype=torch.float32)
    uncertainty = torch.zeros(count, dtype=torch.float32)
    confidence = torch.zeros(count, dtype=torch.float32)
    normals = torch.zeros(count, 3, dtype=torch.float32)
    colors = torch.zeros(count, 3, dtype=torch.float32)
    color_mad = torch.zeros(count, dtype=torch.float32)
    feature = torch.zeros(count, dtype=torch.float32)
    support = torch.zeros(count, dtype=torch.long)
    occluded = torch.zeros(count, dtype=torch.long)
    contradiction = torch.zeros(count, dtype=torch.long)
    evidence = torch.zeros(count, dtype=torch.long)
    normal_sum = torch.zeros(count, dtype=torch.float32)
    color_samples = torch.full(
        (count, config.support_views + 1, 3),
        torch.nan,
        dtype=torch.float32,
    )
    feature_maps = _image_feature_maps(field.images)
    canvases = [(int(camera.width), int(camera.height)) for camera in inputs.cameras]
    neighbors = _camera_neighbors(inputs.cameras, config.support_views)

    for view_index in range(inputs.n_views):
        rows = (source_view == view_index).nonzero(as_tuple=True)[0]
        if not rows.numel():
            continue
        xy = source_xy[rows].float()
        canvas = canvases[view_index]
        local_depth = _sample_map(field.depth[view_index], xy, canvas)
        local_uncertainty = _sample_map(field.uncertainty[view_index], xy, canvas)
        local_confidence = _sample_map(field.confidence[view_index], xy, canvas)
        local_normals = _sample_map(field.normals[view_index], xy, canvas)
        local_colors = _sample_map(field.images[view_index], xy, canvas)
        local_feature = _sample_map(feature_maps[view_index], xy, canvas)
        local_means = inputs.cameras[view_index].unproject(xy, local_depth)
        means[rows] = local_means
        depth[rows] = local_depth
        uncertainty[rows] = local_uncertainty
        confidence[rows] = local_confidence
        normals[rows] = local_normals
        colors[rows] = local_colors
        color_samples[rows, 0] = local_colors
        feature[rows] = local_feature
        if not apply_projective_support:
            continue
        for neighbor_slot, target in enumerate(neighbors[view_index], start=1):
            projected, projected_depth = inputs.cameras[target].project(local_means)
            in_image = inputs.cameras[target].in_image(projected) & (projected_depth > 0.0)
            target_depth = _sample_map(field.depth[target], projected, canvases[target])
            target_uncertainty = _sample_map(
                field.uncertainty[target], projected, canvases[target]
            )
            target_confidence = _sample_map(field.confidence[target], projected, canvases[target])
            target_normal = _sample_map(field.normals[target], projected, canvases[target])
            target_color = _sample_map(field.images[target], projected, canvases[target])
            valid = (
                in_image
                & (target_confidence >= config.min_target_confidence)
                & torch.isfinite(target_normal).all(dim=-1)
                & (target_normal.norm(dim=-1) > 0.5)
            )
            classification = classify_projective_depth(
                projected_depth,
                target_depth,
                local_uncertainty,
                target_uncertainty,
                valid,
                relative_tolerance=config.support_relative_tolerance,
                uncertainty_multiplier=config.support_uncertainty_multiplier,
            )
            agreement = (local_normals * target_normal).sum(dim=-1).abs()
            compatible_support = classification.support & (agreement >= config.min_normal_cosine)
            support[rows] += compatible_support.long()
            occluded[rows] += classification.occluded.long()
            contradiction[rows] += (
                classification.contradiction | (classification.support & ~compatible_support)
            ).long()
            evidence[rows] += (~classification.invalid).long()
            normal_sum[rows] += torch.where(compatible_support, agreement, torch.zeros_like(agreement))
            supported_rows = rows[compatible_support]
            color_samples[supported_rows, neighbor_slot] = target_color[compatible_support]
    colors = torch.nanmedian(color_samples, dim=1).values
    color_mad = torch.nanmedian((color_samples - colors[:, None]).abs(), dim=1).values.max(dim=1).values
    center, extent = inputs.bounds_hint
    half = float(extent) * config.bounds_half_extent_multiplier
    in_bounds = ((means - center.float()[None]).abs() <= half).all(dim=-1)
    finite = (
        torch.isfinite(means).all(dim=-1)
        & torch.isfinite(depth)
        & (depth > 0.0)
        & torch.isfinite(uncertainty)
        & (uncertainty >= 0.0)
        & torch.isfinite(confidence)
        & (confidence >= config.min_source_confidence)
        & torch.isfinite(normals).all(dim=-1)
        & (normals.norm(dim=-1) > 0.5)
        & torch.isfinite(colors).all(dim=-1)
        & in_bounds
    )
    if apply_projective_support:
        primary = (
            finite
            & (support >= config.min_support_views)
            & (contradiction <= config.max_contradictions)
        )
    else:
        primary = finite
    uncertainty_quality = torch.exp(
        -uncertainty / (0.10 * depth.abs()).clamp_min(1e-6)
    ).clamp(0.0, 1.0)
    support_quality = support.float() / evidence.clamp_min(1).float()
    normal_quality = normal_sum / support.clamp_min(1).float()
    cover_floor = torch.where(source_kind == 1, torch.full_like(feature, 0.5), torch.zeros_like(feature))
    importance = (
        0.25 * confidence
        + 0.25 * support_quality
        + 0.20 * uncertainty_quality
        + 0.20 * feature
        + 0.05 * normal_quality
        + 0.05 * cover_floor
    )
    importance = torch.where(finite, importance, torch.zeros_like(importance))
    return {
        "means": means,
        "depth": depth,
        "uncertainty": uncertainty,
        "confidence": confidence,
        "normals": torch.nn.functional.normalize(normals, dim=-1),
        "colors": colors.clamp(0.0, 1.0),
        "color_mad": color_mad,
        "feature": feature,
        "support": support,
        "occluded": occluded,
        "contradiction": contradiction,
        "evidence": evidence,
        "finite": finite,
        "primary": primary,
        "importance": importance,
        "source_view": source_view,
        "source_component": source_component,
        "source_xy": source_xy,
        "source_kind": source_kind,
    }


def contract_candidates(
    candidates: dict[str, Any],
    selected_indices: np.ndarray,
    diameter: float,
    config: CoherentDepthConfig,
):
    """Absorb only close cross-view duplicates without changing the selected exact budget.

    WSE owns decimation. Each eliminated proposal may move its nearest compatible survivor toward
    a weighted continuous centroid, but a survivor has bounded capacity and displacement. This
    prevents contraction from becoming an implicit second, uncontrolled elimination algorithm.
    Appearance and lineage remain those of the selected survivor.
    """

    import torch
    from scipy.spatial import cKDTree

    points = candidates["means"].detach().cpu().numpy().astype(np.float64)
    colors = candidates["colors"].detach().cpu().numpy().astype(np.float64)
    normals = candidates["normals"].detach().cpu().numpy().astype(np.float64)
    importance = candidates["importance"].detach().cpu().numpy().astype(np.float64)
    source_view = candidates["source_view"].detach().cpu().numpy().astype(np.int64)
    source_kind = candidates["source_kind"].detach().cpu().numpy().astype(np.int64)
    support = candidates["support"].detach().cpu().numpy().astype(np.int64)
    contradiction = candidates["contradiction"].detach().cpu().numpy().astype(np.int64)
    selected = np.asarray(selected_indices, dtype=np.int64)
    if selected.ndim != 1 or selected.size == 0:
        raise ValueError("selected_indices must be one non-empty vector")
    if np.unique(selected).size != selected.size or (selected < 0).any() or (selected >= len(points)).any():
        raise ValueError("selected_indices must be unique in-range rows")

    radius = config.contraction_radius_diameter_fraction * diameter
    survivor_points = points[selected]
    survivor_tree = cKDTree(survivor_points)
    unselected = np.flatnonzero(~np.isin(np.arange(len(points)), selected, assume_unique=True))
    query_k = min(8, len(selected))
    if unselected.size:
        distances, neighbors = survivor_tree.query(points[unselected], k=query_k, workers=-1)
        distances = np.asarray(distances).reshape(len(unselected), query_k)
        neighbors = np.asarray(neighbors).reshape(len(unselected), query_k)
    else:
        distances = np.empty((0, query_k), dtype=np.float64)
        neighbors = np.empty((0, query_k), dtype=np.int64)

    proposals: list[tuple[float, int, int]] = []
    for row, candidate_index in enumerate(unselected.tolist()):
        for distance, survivor_local in zip(
            distances[row].tolist(), neighbors[row].tolist(), strict=True
        ):
            if distance > radius:
                break
            survivor_index = int(selected[survivor_local])
            if source_view[candidate_index] == source_view[survivor_index]:
                continue
            if source_kind[candidate_index] != source_kind[survivor_index]:
                continue
            if abs(int(support[candidate_index]) - int(support[survivor_index])) > 1:
                continue
            if contradiction[candidate_index] != contradiction[survivor_index]:
                continue
            color_difference = float(
                np.max(np.abs(colors[candidate_index] - colors[survivor_index]))
            )
            normal_agreement = float(
                abs(np.dot(normals[candidate_index], normals[survivor_index]))
            )
            if (
                color_difference <= config.contraction_rgb_barrier
                and normal_agreement >= config.contraction_normal_cosine
            ):
                proposals.append((float(distance), candidate_index, int(survivor_local)))
                break
    proposals.sort(key=lambda item: (item[0], item[1], item[2]))

    capacity = np.full(
        len(selected), config.contraction_max_cluster_size - 1, dtype=np.int64
    )
    members: list[list[int]] = [[int(index)] for index in selected.tolist()]
    for _distance, candidate_index, survivor_local in proposals:
        if capacity[survivor_local] <= 0:
            continue
        members[survivor_local].append(candidate_index)
        capacity[survivor_local] -= 1

    contracted_points = survivor_points.copy()
    displacement_cap = 0.5 * radius
    for survivor_local, cluster in enumerate(members):
        if len(cluster) == 1:
            continue
        member_indices = np.asarray(cluster, dtype=np.int64)
        weights = np.maximum(importance[member_indices], 1e-6)
        centroid = np.average(points[member_indices], axis=0, weights=weights)
        delta = centroid - survivor_points[survivor_local]
        length = float(np.linalg.norm(delta))
        if length > displacement_cap:
            delta *= displacement_cap / length
        contracted_points[survivor_local] += delta

    selected_tensor = torch.from_numpy(selected)
    output = {name: value[selected_tensor].clone() for name, value in candidates.items()}
    output["means"] = torch.from_numpy(contracted_points.astype(np.float32))
    displacement = np.linalg.norm(contracted_points - survivor_points, axis=1)
    cluster_sizes = np.asarray([len(cluster) for cluster in members], dtype=np.int64)
    absorbed = int(cluster_sizes.sum() - len(selected))
    diagnostics = {
        "input_count": int(len(points)),
        "output_count": int(len(selected)),
        "radius": float(radius),
        "max_displacement": float(displacement_cap),
        "compatible_assignment_count": int(len(proposals)),
        "absorbed_proposal_count": absorbed,
        "capacity_rejected_count": int(len(proposals) - absorbed),
        "cluster_size": _distribution(cluster_sizes),
        "selected_displacement": _distribution(displacement),
        "continuous_nonzero_displacement_fraction": float(np.mean(displacement > 1e-9)),
        "cross_view_only": True,
        "appearance_changed": False,
        "lineage_changed": False,
    }
    return output, diagnostics


def select_feature_anchors(
    points: np.ndarray,
    colors: np.ndarray,
    normals: np.ndarray,
    feature: np.ndarray,
    importance: np.ndarray,
    view_indices: np.ndarray,
    target_count: int,
    *,
    fraction: float,
    radius: float,
    rgb_barrier: float,
    normal_cosine: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Choose a balanced, feature-ranked hard-protection set with compatible 3D NMS."""

    points = np.asarray(points, dtype=np.float64)
    colors = np.asarray(colors, dtype=np.float64)
    normals = np.asarray(normals, dtype=np.float64)
    feature = np.asarray(feature, dtype=np.float64)
    importance = np.asarray(importance, dtype=np.float64)
    view_indices = np.asarray(view_indices, dtype=np.int64)
    count = len(points)
    if points.shape != (count, 3) or colors.shape != (count, 3) or normals.shape != (count, 3):
        raise ValueError("feature-anchor points, colors, and normals must have shape (N,3)")
    if feature.shape != (count,) or importance.shape != (count,) or view_indices.shape != (count,):
        raise ValueError("feature-anchor scalars must have shape (N,)")
    if not all(np.isfinite(value).all() for value in (points, colors, normals, feature, importance)):
        raise ValueError("feature-anchor inputs must be finite")
    target = _integer(target_count, "target_count", minimum=1)
    if target > count:
        raise ValueError("target_count cannot exceed the candidate count")
    anchor_fraction = _positive_float(fraction, "fraction", allow_zero=True)
    if anchor_fraction > 1.0:
        raise ValueError("fraction must lie in [0,1]")
    protected_target = min(int(round(target * anchor_fraction)), target, count)
    protected = np.zeros(count, dtype=bool)
    if protected_target == 0:
        return protected, {"target_count": 0, "selected_count": 0, "nms_selected_count": 0}
    radius = _positive_float(radius, "radius")
    views = np.unique(view_indices)
    quotas = dict(zip(views.tolist(), _balanced_counts(protected_target, len(views)), strict=True))
    selected_per_view = {int(view): 0 for view in views}
    indices = np.arange(count, dtype=np.int64)
    order = np.lexsort((indices, -importance, -feature))
    cells = np.floor(points / radius).astype(np.int64)
    buckets: dict[tuple[int, int, int], list[int]] = {}

    def compatible_neighbor_exists(index: int) -> bool:
        cell = cells[index]
        for offset_x in (-1, 0, 1):
            for offset_y in (-1, 0, 1):
                for offset_z in (-1, 0, 1):
                    key = (
                        int(cell[0] + offset_x),
                        int(cell[1] + offset_y),
                        int(cell[2] + offset_z),
                    )
                    for other in buckets.get(key, []):
                        if np.linalg.norm(points[index] - points[other]) >= radius:
                            continue
                        if np.max(np.abs(colors[index] - colors[other])) > rgb_barrier:
                            continue
                        if abs(float(np.dot(normals[index], normals[other]))) < normal_cosine:
                            continue
                        return True
        return False

    for index in order.tolist():
        view = int(view_indices[index])
        if selected_per_view[view] >= quotas[view] or compatible_neighbor_exists(index):
            continue
        protected[index] = True
        selected_per_view[view] += 1
        buckets.setdefault(tuple(int(value) for value in cells[index]), []).append(index)
        if int(protected.sum()) == protected_target:
            break
    nms_selected = int(protected.sum())

    # Extremely small or duplicate-only views may not fill their NMS quota. Fill by the same
    # frozen feature order so the hard-count contract remains deterministic and auditable.
    for index in order.tolist():
        if protected[index]:
            continue
        view = int(view_indices[index])
        if selected_per_view[view] >= quotas[view]:
            continue
        protected[index] = True
        selected_per_view[view] += 1
        if int(protected.sum()) == protected_target:
            break
    for index in order.tolist():
        if int(protected.sum()) == protected_target:
            break
        if not protected[index]:
            protected[index] = True
            selected_per_view[int(view_indices[index])] += 1
    selected = np.flatnonzero(protected)
    return protected, {
        "target_count": protected_target,
        "selected_count": int(selected.size),
        "nms_selected_count": nms_selected,
        "forced_fill_count": int(selected.size - nms_selected),
        "radius": float(radius),
        "feature": _distribution(feature[selected]),
        "importance": _distribution(importance[selected]),
        "selected_per_view": {str(view): int(selected_per_view[int(view)]) for view in views},
    }


def dynamic_weighted_sample_elimination(
    points: np.ndarray,
    colors: np.ndarray,
    normals: np.ndarray,
    importance: np.ndarray,
    view_indices: np.ndarray,
    target_count: int,
    *,
    neighbors: int = 24,
    alpha: float = 8.0,
    rgb_barrier: float = 0.15,
    normal_cosine: float = 0.70,
    feature_protection: float = 3.0,
    view_floor_fraction: float = 0.50,
    protected: np.ndarray | None = None,
) -> WeightedEliminationResult:
    """Dynamic kNN WSE with hard anchors, compatibility barriers, and per-view floors."""

    from scipy.spatial import cKDTree

    points = np.asarray(points, dtype=np.float64)
    colors = np.asarray(colors, dtype=np.float64)
    normals = np.asarray(normals, dtype=np.float64)
    importance = np.asarray(importance, dtype=np.float64)
    view_indices = np.asarray(view_indices, dtype=np.int64)
    count = points.shape[0]
    target = _integer(target_count, "target_count", minimum=1)
    if not target <= count:
        raise ValueError("target_count cannot exceed the candidate count")
    if points.shape != (count, 3) or colors.shape != (count, 3) or normals.shape != (count, 3):
        raise ValueError("points, colors, and normals must all have shape (N,3)")
    if importance.shape != (count,) or view_indices.shape != (count,):
        raise ValueError("importance and view_indices must have shape (N,)")
    if not all(np.isfinite(value).all() for value in (points, colors, normals, importance)):
        raise ValueError("WSE inputs must be finite")
    if (view_indices < 0).any():
        raise ValueError("view_indices must be non-negative")
    protected_mask = (
        np.zeros(count, dtype=bool) if protected is None else np.asarray(protected, dtype=bool)
    )
    if protected_mask.shape != (count,):
        raise ValueError("protected must have shape (N,)")
    if int(protected_mask.sum()) > target:
        raise ValueError("protected rows cannot exceed target_count")
    if target == count:
        zeros = np.zeros(count, dtype=np.float64)
        return WeightedEliminationResult(
            selected_indices=np.arange(count, dtype=np.int64),
            initial_crowding=zeros,
            final_crowding=zeros,
            removal_order=np.empty(0, dtype=np.int64),
            diagnostics={
                "input_count": count,
                "target_count": target,
                "removed_count": 0,
                "protected_count": int(protected_mask.sum()),
                "protected_survivor_count": int(protected_mask.sum()),
            },
        )

    k = min(_integer(neighbors, "neighbors", minimum=1), count - 1)
    tree = cKDTree(points)
    distances, indices = tree.query(points, k=k + 1, workers=-1)
    redundancy_rank = min(max(int(math.ceil(count / target)), 1), k)
    local_radius = distances[:, redundancy_rank].clip(min=1e-12)
    first = np.repeat(np.arange(count, dtype=np.int64), k)
    second = indices[:, 1:].reshape(-1).astype(np.int64)
    pairs = np.stack([np.minimum(first, second), np.maximum(first, second)], axis=1)
    pairs = np.unique(pairs, axis=0)
    delta = points[pairs[:, 0]] - points[pairs[:, 1]]
    distance = np.linalg.norm(delta, axis=1)
    radius = np.sqrt(local_radius[pairs[:, 0]] * local_radius[pairs[:, 1]])
    color_difference = np.max(np.abs(colors[pairs[:, 0]] - colors[pairs[:, 1]]), axis=1)
    normal_agreement = np.abs(np.sum(normals[pairs[:, 0]] * normals[pairs[:, 1]], axis=1))
    compatible = (color_difference <= rgb_barrier) & (normal_agreement >= normal_cosine)
    color_similarity = np.clip(1.0 - color_difference / max(rgb_barrier, 1e-12), 0.0, 1.0)
    normal_similarity = np.clip(
        (normal_agreement - normal_cosine) / max(1.0 - normal_cosine, 1e-12), 0.0, 1.0
    )
    contribution = np.exp(-alpha * np.square(distance / radius))
    contribution *= compatible * (0.25 + 0.75 * color_similarity) * (0.25 + 0.75 * normal_similarity)
    nonzero = contribution > np.finfo(np.float64).tiny
    pairs = pairs[nonzero]
    contribution = contribution[nonzero]

    adjacency: list[list[tuple[int, float]]] = [[] for _ in range(count)]
    crowding = np.zeros(count, dtype=np.float64)
    for (first_index, second_index), value in zip(pairs.tolist(), contribution.tolist(), strict=True):
        adjacency[first_index].append((second_index, value))
        adjacency[second_index].append((first_index, value))
        crowding[first_index] += value
        crowding[second_index] += value
    initial_crowding = crowding.copy()
    normalized_importance = np.clip(importance, 0.0, None)
    positive = normalized_importance[normalized_importance > 0.0]
    scale = float(np.median(positive)) if positive.size else 1.0
    normalized_importance /= max(scale, np.finfo(np.float64).tiny)

    alive = np.ones(count, dtype=bool)
    version = np.zeros(count, dtype=np.int64)
    views = np.unique(view_indices)
    view_count = np.bincount(view_indices, minlength=int(views.max()) + 1).astype(np.int64)
    floor = max(1, int(math.floor(view_floor_fraction * target / max(len(views), 1))))
    floors = np.minimum(view_count, floor)

    crowding_floor = 1.0 / float(k)

    def priority(index: int) -> float:
        quality = 0.05 + normalized_importance[index]
        return float((crowding[index] + crowding_floor) / quality**feature_protection)

    heap = [(-priority(index), index, 0) for index in range(count)]
    heapq.heapify(heap)
    removal = []
    alive_count = count
    while alive_count > target:
        selected = None
        deferred = []
        while heap:
            negative, index, current_version = heapq.heappop(heap)
            if not alive[index] or current_version != version[index]:
                continue
            if protected_mask[index]:
                continue
            if view_count[view_indices[index]] <= floors[view_indices[index]]:
                deferred.append((negative, index, current_version))
                continue
            selected = index
            break
        for item in deferred:
            heapq.heappush(heap, item)
        if selected is None:
            raise RuntimeError(
                "protected rows and per-view WSE floors prevent reaching the exact target count"
            )
        alive[selected] = False
        alive_count -= 1
        removal.append(selected)
        view_count[view_indices[selected]] -= 1
        for neighbor, value in adjacency[selected]:
            if not alive[neighbor]:
                continue
            crowding[neighbor] = max(crowding[neighbor] - value, 0.0)
            version[neighbor] += 1
            heapq.heappush(heap, (-priority(neighbor), neighbor, int(version[neighbor])))
    selected_indices = np.flatnonzero(alive).astype(np.int64)
    if selected_indices.size != target:
        raise RuntimeError("weighted sample elimination violated its exact-count contract")
    return WeightedEliminationResult(
        selected_indices=selected_indices,
        initial_crowding=initial_crowding,
        final_crowding=crowding,
        removal_order=np.asarray(removal, dtype=np.int64),
        diagnostics={
            "input_count": count,
            "target_count": target,
            "removed_count": len(removal),
            "neighbor_k": k,
            "pair_count": int(len(pairs)),
            "crowding_floor": crowding_floor,
            "protected_count": int(protected_mask.sum()),
            "protected_survivor_count": int(protected_mask[selected_indices].sum()),
            "per_view_floor": int(floor),
            "survivors_per_view": {
                str(int(view)): int(np.sum(view_indices[selected_indices] == view)) for view in views
            },
            "initial_crowding": _distribution(initial_crowding),
            "selected_final_crowding": _distribution(crowding[selected_indices]),
            "selected_importance": _distribution(importance[selected_indices]),
            "local_radius": _distribution(local_radius),
        },
    )


def _balanced_topk(scores: Any, view_indices: Any, target: int) -> Any:
    import torch

    selected = []
    counts = _balanced_counts(target, int(view_indices.max()) + 1)
    for view, count in enumerate(counts):
        rows = (view_indices == view).nonzero(as_tuple=True)[0]
        if rows.numel() < count:
            selected.extend(rows.tolist())
            continue
        order = torch.argsort(scores[rows], descending=True, stable=True)
        selected.extend(rows[order[:count]].tolist())
    selected_set = set(selected)
    if len(selected) < target:
        order = torch.argsort(scores, descending=True, stable=True).tolist()
        for index in order:
            if index in selected_set:
                continue
            selected.append(index)
            selected_set.add(index)
            if len(selected) == target:
                break
    if len(selected) != target:
        raise RuntimeError("balanced exact-budget selection could not fill the target")
    return torch.tensor(selected, dtype=torch.long)


def _oriented_surfel_covariance(
    points: Any,
    normals: Any,
    uncertainty: Any,
    sigma_tangent: Any,
    diameter: float,
    config: CoherentDepthConfig,
):
    import torch

    sigma_normal = torch.maximum(
        torch.full_like(sigma_tangent, 1e-5 * diameter),
        torch.minimum(uncertainty.float(), config.normal_flatness * sigma_tangent),
    )
    normal = torch.nn.functional.normalize(normals.float(), dim=-1)
    identity = torch.eye(3, dtype=points.dtype)[None]
    projector = normal[:, :, None] * normal[:, None, :]
    covariance = sigma_tangent.square()[:, None, None] * (identity - projector)
    covariance += sigma_normal.square()[:, None, None] * projector
    return covariance, sigma_normal


def _construction_pixel_footprints(
    points: Any,
    depth: Any,
    source_view: Any,
    cameras: Sequence[Any],
    raster_shape: tuple[int, int],
):
    import torch

    height, width = raster_shape
    source_footprint = torch.zeros_like(depth, dtype=torch.float32)
    for view_index, camera in enumerate(cameras):
        rows = source_view == view_index
        if not bool(rows.any()):
            continue
        scaled_fx = float(camera.fx) * width / int(camera.width)
        scaled_fy = float(camera.fy) * height / int(camera.height)
        focal = math.sqrt(scaled_fx * scaled_fy)
        source_footprint[rows] = depth[rows].float() / focal
    if not bool(torch.isfinite(source_footprint).all()) or bool((source_footprint <= 0.0).any()):
        raise RuntimeError("selected source pixel footprints must be finite and positive")

    footprint = torch.full_like(source_footprint, torch.inf)
    visible_count = torch.zeros_like(source_view, dtype=torch.long)
    for camera in cameras:
        projected, projected_depth = camera.project(points)
        visible = camera.in_image(projected) & (projected_depth > 0.0)
        scaled_fx = float(camera.fx) * width / int(camera.width)
        scaled_fy = float(camera.fy) * height / int(camera.height)
        candidate = projected_depth.float() / math.sqrt(scaled_fx * scaled_fy)
        footprint = torch.where(visible, torch.minimum(footprint, candidate), footprint)
        visible_count += visible.long()
    footprint = torch.where(torch.isfinite(footprint), footprint, source_footprint)
    if not bool(torch.isfinite(footprint).all()) or bool((footprint <= 0.0).any()):
        raise RuntimeError("construction-view pixel footprints must be finite and positive")
    return footprint, visible_count


def _local_surfel_covariances(
    points: Any,
    normals: Any,
    uncertainty: Any,
    pixel_footprint: Any,
    diameter: float,
    config: CoherentDepthConfig,
):
    import torch
    from scipy.spatial import cKDTree

    points_np = points.detach().cpu().numpy().astype(np.float64)
    if len(points_np) == 1:
        distances = (2.0 * pixel_footprint.detach().cpu().numpy()).astype(np.float64)
    else:
        distances = cKDTree(points_np).query(points_np, k=2, workers=-1)[0][:, 1]
    spacing = torch.from_numpy(distances.astype(np.float32)).clamp_min(1e-5 * diameter)
    uncapped = 0.5 * spacing
    maximum = (config.surface_cover_max_pixel_sigma * pixel_footprint).clamp_min(
        1e-5 * diameter
    )
    sigma_tangent = torch.minimum(uncapped, maximum)
    covariance, sigma_normal = _oriented_surfel_covariance(
        points, normals, uncertainty, sigma_tangent, diameter, config
    )
    return covariance, spacing, sigma_tangent, sigma_normal


def _compatible_surface_cover(
    points: Any,
    colors: Any,
    normals: Any,
    uncertainty: Any,
    pixel_footprint: Any,
    diameter: float,
    config: CoherentDepthConfig,
):
    """Build cover covariances from color/normal-compatible neighbors and depth normals."""

    import torch
    from scipy.spatial import cKDTree

    points_np = points.detach().cpu().numpy().astype(np.float64)
    colors_np = colors.detach().cpu().numpy().astype(np.float64)
    normals_np = normals.detach().cpu().numpy().astype(np.float64)
    count = len(points_np)
    k = min(config.surface_cover_neighbors, count - 1)
    if k == 0:
        requested = 0
        compatible_count = np.zeros(1, dtype=np.int64)
        fallback = np.ones(1, dtype=bool)
        spacing_np = (
            pixel_footprint.detach().cpu().numpy() / config.surface_cover_sigma_ratio
        ).astype(np.float64)
    else:
        distances, indices = cKDTree(points_np).query(points_np, k=k + 1, workers=-1)
        distances = distances[:, 1:]
        indices = indices[:, 1:]
        color_difference = np.max(np.abs(colors_np[:, None] - colors_np[indices]), axis=-1)
        normal_agreement = np.abs(
            np.sum(normals_np[:, None] * normals_np[indices], axis=-1)
        )
        compatible = (
            (color_difference <= config.wse_rgb_barrier)
            & (normal_agreement >= config.wse_normal_cosine)
        )
        compatible_distances = np.where(compatible, distances, np.inf)
        compatible_distances.sort(axis=1)
        requested = min(config.surface_cover_spacing_neighbors, k)
        nearest = compatible_distances[:, :requested]
        finite = np.isfinite(nearest)
        compatible_count = finite.sum(axis=1)
        spacing_np = np.divide(
            np.where(finite, nearest, 0.0).sum(axis=1),
            np.maximum(compatible_count, 1),
        )
        fallback = compatible_count == 0
        spacing_np[fallback] = distances[fallback, 0]
    spacing = torch.from_numpy(spacing_np.astype(np.float32)).clamp_min(1e-5 * diameter)
    uncapped = config.surface_cover_sigma_ratio * spacing
    maximum = (config.surface_cover_max_pixel_sigma * pixel_footprint).clamp_min(
        1e-5 * diameter
    )
    sigma_tangent = torch.minimum(uncapped, maximum)
    covariance, sigma_normal = _oriented_surfel_covariance(
        points, normals, uncertainty, sigma_tangent, diameter, config
    )

    hex_cell_area = math.sqrt(3.0) / 2.0
    overlap = (
        2.0 * math.pi * sigma_tangent.double().square()
        / (hex_cell_area * spacing.double().square()).clamp_min(1e-30)
    ).clamp_min(1e-6)
    opacity = 1.0 - (1.0 - config.surface_cover_target_alpha) ** (1.0 / overlap)
    opacity = opacity.clamp(
        config.surface_cover_min_opacity, config.surface_cover_max_opacity
    ).float()
    diagnostics = {
        "schema": "structsplat.compatible_depth_surface_cover.v1",
        "neighbor_k": int(k),
        "spacing_neighbors": int(requested),
        "orientation": "fused_depth_normal",
        "compatibility": "packet_color_and_fused_depth_normal",
        "fallback_nearest_geometry_count": int(fallback.sum()),
        "fallback_nearest_geometry_fraction": float(fallback.mean()),
        "compatible_neighbors_used": _distribution(compatible_count),
        "spacing": _distribution(spacing),
        "pixel_footprint": _distribution(pixel_footprint),
        "uncapped_sigma_tangent": _distribution(uncapped),
        "max_pixel_sigma": config.surface_cover_max_pixel_sigma,
        "capped_count": int((uncapped > maximum).sum()),
        "capped_fraction": float((uncapped > maximum).float().mean()),
        "sigma_tangent": _distribution(sigma_tangent),
        "sigma_normal": _distribution(sigma_normal),
        "overlap_kernel_units": _distribution(overlap),
        "opacity": _distribution(opacity),
    }
    return covariance, opacity, spacing, sigma_tangent, sigma_normal, diagnostics


def initialize_calibrated_coherent_depth(
    inputs: Any,
    views: Sequence[RealtimeGSCodecNativeView],
    checkpoint: str | Path | None,
    config: CoherentDepthConfig | None = None,
    *,
    field: CoherentDepthField | None = None,
    predictor: Callable[[Any, tuple[int, ...]], dict[str, Any]] | None = None,
    apply_projective_support: bool = True,
) -> CoherentDepthLiftResult:
    """Compile a fused coherent-depth field into an exact-budget realtime-gs initialization."""

    config = config or CoherentDepthConfig()
    if field is None:
        field = infer_coherent_depth_field(inputs, views, checkpoint, config, predictor=predictor)
    if len(views) != int(inputs.n_views) or field.depth.shape[0] != int(inputs.n_views):
        raise ValueError("field/view/input counts do not match")
    try:
        import torch
        from rtgs.core.gaussians3d import Gaussians3D
        from rtgs.lift.compact_carve import CompactInitializationResult, CompactLineage
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("coherent-depth lifting requires torch and realtime-gs") from exc
    started = time.perf_counter()
    source_view, source_component, source_xy, source_kind, proposal_diagnostics = (
        _propose_candidate_rays(inputs, field, config)
    )
    candidates = _candidate_evidence(
        inputs,
        views,
        field,
        source_view,
        source_component,
        source_xy,
        source_kind,
        config,
        apply_projective_support=apply_projective_support,
    )
    primary = candidates["primary"]
    finite = candidates["finite"]
    primary_count = int(primary.sum())
    fallback_needed = max(config.target_count - primary_count, 0)
    accepted = primary.clone()
    fallback = torch.zeros_like(primary)
    if fallback_needed:
        if not config.allow_fallback:
            raise RuntimeError("projective support cannot fill the exact budget and fallback is closed")
        available = finite & ~primary
        order = torch.argsort(candidates["importance"], descending=True, stable=True)
        chosen = order[available[order]][:fallback_needed]
        fallback[chosen] = True
        accepted |= fallback
    if int(accepted.sum()) < config.target_count:
        raise RuntimeError("finite coherent-depth proposals cannot fill the exact target count")
    primary_fraction_floor = min(primary_count, config.target_count) / config.target_count
    if apply_projective_support and primary_fraction_floor < config.min_primary_fraction:
        raise RuntimeError(
            "projectively supported candidates fail the frozen primary-fraction floor: "
            f"{primary_fraction_floor:.6f} < {config.min_primary_fraction:.6f}"
        )
    accepted_indices = accepted.nonzero(as_tuple=True)[0]
    working = {name: value[accepted_indices] for name, value in candidates.items()}
    working_fallback = fallback[accepted_indices]
    original_indices = accepted_indices.clone()
    center, extent = inputs.bounds_hint
    del center
    diameter = 2.0 * float(extent) * math.sqrt(3.0)

    if working["means"].shape[0] < config.target_count:
        raise RuntimeError("accepted candidates cannot fill the exact output budget")

    anchor_diagnostics = {"applied": False}
    wse_diagnostics = {"applied": False}
    if config.apply_wse and apply_projective_support:
        protected, anchor_diagnostics = select_feature_anchors(
            working["means"].numpy(),
            working["colors"].numpy(),
            working["normals"].numpy(),
            working["feature"].numpy(),
            working["importance"].numpy(),
            working["source_view"].numpy(),
            config.target_count,
            fraction=config.wse_anchor_fraction,
            radius=config.wse_anchor_radius_diameter_fraction * diameter,
            rgb_barrier=config.wse_rgb_barrier,
            normal_cosine=config.wse_normal_cosine,
        )
        anchor_diagnostics["applied"] = True
        wse = dynamic_weighted_sample_elimination(
            working["means"].numpy(),
            working["colors"].numpy(),
            working["normals"].numpy(),
            working["importance"].numpy(),
            working["source_view"].numpy(),
            config.target_count,
            neighbors=config.wse_neighbors,
            alpha=config.wse_alpha,
            rgb_barrier=config.wse_rgb_barrier,
            normal_cosine=config.wse_normal_cosine,
            feature_protection=config.wse_feature_protection,
            view_floor_fraction=config.wse_view_floor_fraction,
            protected=protected,
        )
        selected_np = wse.selected_indices
        wse_diagnostics = {**wse.diagnostics, "applied": True}
    else:
        selected_np = _balanced_topk(
            working["importance"], working["source_view"], config.target_count
        ).numpy()
    selected = torch.from_numpy(selected_np)
    selected_fallback = working_fallback[selected]
    selected_original = original_indices[selected]

    contraction_diagnostics = {"applied": False}
    if config.apply_contraction and apply_projective_support:
        selected_working, contraction_diagnostics = contract_candidates(
            working, selected_np, diameter, config
        )
        contraction_diagnostics["applied"] = True
    else:
        selected_working = {name: value[selected].clone() for name, value in working.items()}

    pixel_footprint, footprint_visible_views = _construction_pixel_footprints(
        selected_working["means"],
        selected_working["depth"],
        selected_working["source_view"],
        inputs.cameras,
        (int(field.depth.shape[-2]), int(field.depth.shape[-1])),
    )
    covariance, spacing, sigma_tangent, sigma_normal = _local_surfel_covariances(
        selected_working["means"],
        selected_working["normals"],
        selected_working["uncertainty"],
        pixel_footprint,
        diameter,
        config,
    )
    opacity = torch.full((config.target_count,), config.init_opacity, dtype=torch.float32)
    raw_gaussians = Gaussians3D.from_means_covs(
        selected_working["means"],
        covariance,
        selected_working["colors"],
        opacity,
        sh_degree=0,
    )
    gaussians = raw_gaussians
    surface_diagnostics = None
    surface_seconds = 0.0
    if config.apply_surface_cover:
        surface_started = time.perf_counter()
        (
            cover_covariance,
            cover_opacity,
            _cover_spacing,
            _cover_sigma_tangent,
            _cover_sigma_normal,
            surface_diagnostics,
        ) = _compatible_surface_cover(
            selected_working["means"],
            selected_working["colors"],
            selected_working["normals"],
            selected_working["uncertainty"],
            pixel_footprint,
            diameter,
            config,
        )
        gaussians = Gaussians3D.from_means_covs(
            selected_working["means"],
            cover_covariance,
            selected_working["colors"],
            cover_opacity,
            sh_degree=0,
        )
        surface_seconds = time.perf_counter() - surface_started
        if not torch.equal(gaussians.means, raw_gaussians.means):
            raise RuntimeError("surface-cover reconciliation changed coherent-depth means")
        if not torch.equal(gaussians.sh, raw_gaussians.sh):
            raise RuntimeError("surface-cover reconciliation changed coherent-depth appearance")

    lineage = CompactLineage(
        source_view_indices=selected_working["source_view"].clone(),
        source_component_indices=selected_working["source_component"].clone(),
        source_xy=selected_working["source_xy"].clone(),
    )
    if config.apply_wse and apply_projective_support:
        topology = "hard_anchor_dynamic_feature_wse"
    else:
        topology = "balanced_exact_budget"
    if config.apply_contraction and apply_projective_support:
        topology += "_then_bounded_micro_contraction"
    diagnostics = {
        "schema": "structsplat.calibrated_coherent_depth_lift.v1",
        "ownership": {
            "appearance": "codec_native_packet_query",
            "coherent_depth": "pinned_vggt_over_packet_appearance",
            "metric_scale": "group_sim3_to_known_calibration",
            "rays": "known_calibrated_cameras",
            "surface_acceptance": (
                "occlusion_aware_projective_support"
                if apply_projective_support
                else "raw_source_depth"
            ),
            "topology": topology,
            "render_extent": (
                "compatible_depth_normal_surface_cover"
                if config.apply_surface_cover
                else "local_surfel"
            ),
        },
        "source_rgb_opened": False,
        "reporting_views_present": False,
        "proposal": proposal_diagnostics,
        "candidate_count": int(source_view.numel()),
        "finite_candidate_count": int(finite.sum()),
        "primary_candidate_count": primary_count,
        "primary_fraction_floor": primary_fraction_floor,
        "fallback_candidate_count": int(fallback.sum()),
        "accepted_before_selection": int(accepted.sum()),
        "feature_anchors": anchor_diagnostics,
        "contraction": contraction_diagnostics,
        "wse": wse_diagnostics,
        "selected_fallback_count": int(selected_fallback.sum()),
        "selected_fallback_fraction": float(selected_fallback.float().mean()),
        "selected_original_candidate_indices": selected_original.tolist(),
        "selected_support": _distribution(selected_working["support"]),
        "selected_occlusion": _distribution(selected_working["occluded"]),
        "selected_contradiction": _distribution(selected_working["contradiction"]),
        "selected_confidence": _distribution(selected_working["confidence"]),
        "selected_feature": _distribution(selected_working["feature"]),
        "selected_color_mad": _distribution(selected_working["color_mad"]),
        "selected_cover_visible_views": _distribution(footprint_visible_views),
        "selected_per_view": {
            str(view): int((selected_working["source_view"] == view).sum())
            for view in range(int(inputs.n_views))
        },
        "selected_relative_uncertainty": _distribution(
            selected_working["uncertainty"] / selected_working["depth"].clamp_min(1e-8)
        ),
        "initial_spacing": _distribution(spacing),
        "initial_sigma_tangent": _distribution(sigma_tangent),
        "initial_sigma_normal": _distribution(sigma_normal),
        "surface_cover_seconds": surface_seconds,
        "surface_cover": surface_diagnostics,
        "total_lift_seconds": time.perf_counter() - started,
        "coherent_depth_field": field.diagnostics,
    }
    raw = CompactInitializationResult(
        gaussians=raw_gaussians,
        lineage=lineage,
        depths=selected_working["depth"].clone(),
        depth_sigmas=selected_working["uncertainty"].clone(),
        ray_sigmas=selected_working["uncertainty"].clone(),
        scores=selected_working["importance"].clone(),
        diagnostics=diagnostics,
    )
    initialization = replace(raw, gaussians=gaussians)
    return CoherentDepthLiftResult(
        initialization=initialization,
        raw_initialization=raw,
        field=field,
        selected_candidate_indices=selected_original,
        selected_fallback_mask=selected_fallback,
        diagnostics=diagnostics,
    )


__all__ = [
    "CoherentDepthConfig",
    "CoherentDepthField",
    "CoherentDepthLiftResult",
    "ProjectiveClassification",
    "SimilarityAlignment",
    "WeightedEliminationResult",
    "align_predicted_group",
    "build_calibration_groups",
    "build_packet_inference_images",
    "classify_projective_depth",
    "dynamic_weighted_sample_elimination",
    "fuse_overlapping_depths",
    "infer_coherent_depth_field",
    "initialize_calibrated_coherent_depth",
    "select_feature_anchors",
    "umeyama_similarity",
]
