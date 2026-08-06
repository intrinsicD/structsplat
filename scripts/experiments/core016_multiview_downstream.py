#!/usr/bin/env python3
"""Run CORE-016's exposed source-grounded Janelle multiview diagnostic.

This driver is intentionally not a BENCH-019 executor. It uses one exposed frame and one seed to
answer a development question: does the complete dual-plane packet remain useful after the paired
appearance backend is propagated through realtime-gs CompactCarve and ordinary 3DGS refinement?
Every arm is evaluated against the same calibrated source RGB/masks; no teacher render is a target.

Reproduce from the StructSplat root with::

    PYTHONPATH=src:/home/alex/Documents/realtime-gs/src \
      /home/alex/Documents/realtime-gs/.venv/bin/python \
      scripts/experiments/core016_multiview_downstream.py \
      --out results/core016_multiview_downstream_janelle_2026-08-06_v1

The output directory is immutable: the driver refuses to overwrite it.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import datetime as dt
import gc
import hashlib
from html import escape
import importlib.metadata
import io
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import traceback
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
import torch

from rtgs.core.metrics import image_metrics, masked_crop
from rtgs.data.calibrated import load_calibrated_scene
from rtgs.data.compact_views import CompactDataset
from rtgs.data.reconstruction_inputs import ReconstructionInputs
from rtgs.lift.compact_carve import (
    CompactCarveConfig,
    CompactCarveInitializer,
    build_query_backends,
    make_placement_progress_printer,
)
from rtgs.optim.density import DensityConfig
from rtgs.optim.trainer import TrainConfig, Trainer
from rtgs.render.base import get_rasterizer

from structsplat.codec_native_field import (
    CodecNativeField,
    CodecNativeFieldConfig,
    build_codec_native_field,
)
from structsplat.metrics import LPIPS, ms_ssim
from structsplat.observation_field import CanvasCropTransform
from structsplat.realtime_gs_adapter import make_realtime_gs_view


STRUCTSPLAT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RTGS_ROOT = Path("/home/alex/Documents/realtime-gs")
DEFAULT_FRAME = Path(
    "/home/alex/Dropbox/Work/Janelle/2025_03_07_stage_with_fabric/frame_00008"
)
TRAIN_IDS = (
    "C0001",
    "C0006",
    "C0012",
    "C0019",
    "C0022",
    "C0028",
    "C0031",
    "C0039",
)
HELDOUT_IDS = ("C0004", "C0025", "C1004")
FULL_TRAIN_IDS = (
    "C0001",
    "C0005",
    "C0006",
    "C0008",
    "C0009",
    "C0012",
    "C0014",
    "C0018",
    "C0019",
    "C0020",
    "C0021",
    "C0022",
    "C0026",
    "C0028",
    "C0029",
    "C0030",
    "C0031",
    "C0034",
    "C0037",
    "C0039",
    "C1000",
    "C1001",
    "C1002",
)
PACKET_DOWNSCALE = 4
EVALUATION_DOWNSCALE = 8
PACKET_CROP_MARGIN = 16
SEED = 0
ARM_LABELS = {
    "rtgsv_control": "RTGSV control (≈5.3k rows/view)",
    "dual_webp92_n512": "dual-plane WebP q92 · 512 structure/view",
    "dual_webp92_n2048": "dual-plane WebP q92 · 2,048 structure/view",
}


@dataclasses.dataclass(frozen=True)
class DiagnosticProfile:
    """One predeclared execution profile; profile selection happens before any arm runs."""

    name: str
    arms: tuple[str, ...]
    iterations: int
    eval_every: int
    densify: bool
    scope: str
    train_ids: tuple[str, ...]
    heldout_ids: tuple[str, ...]
    n_init_3d: int
    density_max_gaussians: int
    mask_alpha_lambda: float
    outside_alpha_lambda: float
    polish_iterations: int = 0
    polish_mask_alpha_lambda: float | None = None
    polish_outside_alpha_lambda: float | None = None
    polish_lr_factor: float | None = None
    reference_candidate_psnr_min_db: float | None = None
    reference_candidate_gradient_mae_max: float | None = None


PROFILES = {
    "fixed": DiagnosticProfile(
        name="fixed",
        arms=("rtgsv_control", "dual_webp92_n512", "dual_webp92_n2048"),
        iterations=1_000,
        eval_every=50,
        densify=False,
        scope="fixed-topology representation and initialization isolation",
        train_ids=TRAIN_IDS,
        heldout_ids=HELDOUT_IDS,
        n_init_3d=835,
        density_max_gaussians=20_000,
        mask_alpha_lambda=0.05,
        outside_alpha_lambda=0.01,
    ),
    "density": DiagnosticProfile(
        name="density",
        arms=("rtgsv_control", "dual_webp92_n512"),
        iterations=2_000,
        eval_every=100,
        densify=True,
        scope="post-v1 variable-topology visual-recovery development follow-up",
        train_ids=TRAIN_IDS,
        heldout_ids=HELDOUT_IDS,
        n_init_3d=835,
        density_max_gaussians=20_000,
        mask_alpha_lambda=0.05,
        outside_alpha_lambda=0.01,
    ),
    "full": DiagnosticProfile(
        name="full",
        arms=("rtgsv_control", "dual_webp92_n512"),
        iterations=2_000,
        eval_every=100,
        densify=True,
        scope="23-view capture-regime geometry and visual-quality development assay",
        train_ids=FULL_TRAIN_IDS,
        heldout_ids=HELDOUT_IDS,
        n_init_3d=5_000,
        density_max_gaussians=20_000,
        mask_alpha_lambda=0.05,
        outside_alpha_lambda=0.01,
    ),
    "matched10k": DiagnosticProfile(
        name="matched10k",
        arms=("rtgsv_control", "dual_webp92_n512"),
        iterations=2_000,
        eval_every=100,
        densify=True,
        scope="post-v3 matched 10k-topology efficiency and visual-quality assay",
        train_ids=FULL_TRAIN_IDS,
        heldout_ids=HELDOUT_IDS,
        n_init_3d=5_000,
        density_max_gaussians=10_000,
        mask_alpha_lambda=0.05,
        outside_alpha_lambda=0.01,
    ),
    "silhouette": DiagnosticProfile(
        name="silhouette",
        arms=("rtgsv_control", "dual_webp92_n512"),
        iterations=2_000,
        eval_every=100,
        densify=True,
        scope="post-v4 exact-mask silhouette-supervision artifact assay",
        train_ids=FULL_TRAIN_IDS,
        heldout_ids=HELDOUT_IDS,
        n_init_3d=5_000,
        density_max_gaussians=10_000,
        mask_alpha_lambda=0.20,
        outside_alpha_lambda=0.05,
        reference_candidate_psnr_min_db=24.988010533650714,
        reference_candidate_gradient_mae_max=0.012926414298514525,
    ),
    "latepolish": DiagnosticProfile(
        name="latepolish",
        arms=("rtgsv_control", "dual_webp92_n512"),
        iterations=2_000,
        eval_every=100,
        densify=True,
        scope="post-v5 fixed-topology low-rate silhouette-polish artifact assay",
        train_ids=FULL_TRAIN_IDS,
        heldout_ids=HELDOUT_IDS,
        n_init_3d=5_000,
        density_max_gaussians=10_000,
        mask_alpha_lambda=0.05,
        outside_alpha_lambda=0.01,
        polish_iterations=250,
        polish_mask_alpha_lambda=0.20,
        polish_outside_alpha_lambda=0.05,
        polish_lr_factor=0.20,
        reference_candidate_psnr_min_db=24.988010533650714,
        reference_candidate_gradient_mae_max=0.012926414298514525,
    ),
}

STRUCTSPLAT_SOURCES = (
    "src/structsplat/codec_native_field.py",
    "src/structsplat/realtime_gs_adapter.py",
    "src/structsplat/observation_field.py",
    "src/structsplat/density.py",
    "src/structsplat/sampling.py",
    "src/structsplat/structure_tensor.py",
    "src/structsplat/metrics.py",
    "scripts/experiments/core016_multiview_downstream.py",
    "tasks/CORE-016-codec-native-dual-plane-field.md",
)
RTGS_SOURCES = (
    "src/rtgs/core/camera.py",
    "src/rtgs/core/gaussians3d.py",
    "src/rtgs/core/metrics.py",
    "src/rtgs/core/observation2d.py",
    "src/rtgs/core/observation2d_cuda.py",
    "src/rtgs/data/calibrated.py",
    "src/rtgs/data/compact_views.py",
    "src/rtgs/data/reconstruction_inputs.py",
    "src/rtgs/data/scene.py",
    "src/rtgs/lift/compact_carve.py",
    "src/rtgs/optim/trainer.py",
    "src/rtgs/render/base.py",
    "src/rtgs/render/gsplat_backend.py",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bytes_sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _tensor_sha256(value: torch.Tensor) -> str:
    array = value.detach().contiguous().cpu().numpy()
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("utf-8"))
    digest.update(b"\0")
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("utf-8"))
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(("git", *args), cwd=root, text=True).strip()


def _repository_record(root: Path) -> dict[str, Any]:
    return {
        "path": str(root),
        "head": _git(root, "rev-parse", "HEAD"),
        "branch": _git(root, "branch", "--show-current"),
        "status_short": _git(root, "status", "--short"),
        "diff_sha256": _bytes_sha256(
            subprocess.check_output(("git", "diff", "--binary"), cwd=root)
        ),
    }


def _environment() -> dict[str, Any]:
    packages: dict[str, str | None] = {}
    for name in ("torch", "gsplat", "numpy", "Pillow", "lpips", "matplotlib"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "packages": packages,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_total_bytes": (
            int(torch.cuda.get_device_properties(0).total_memory)
            if torch.cuda.is_available()
            else None
        ),
        "ld_preload": os.environ.get("LD_PRELOAD"),
    }


def _snapshot_sources(out: Path, rtgs_root: Path) -> list[dict[str, Any]]:
    records = []
    for repository, root, paths in (
        ("structsplat", STRUCTSPLAT_ROOT, STRUCTSPLAT_SOURCES),
        ("realtime-gs", rtgs_root, RTGS_SOURCES),
    ):
        for relative in paths:
            source = root / relative
            if not source.is_file():
                raise FileNotFoundError(f"missing bound source: {source}")
            target = out / "executed_sources" / repository / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            records.append(
                {
                    "repository": repository,
                    "source_path": str(source),
                    "snapshot_path": str(target.relative_to(out)),
                    "bytes": target.stat().st_size,
                    "sha256": _sha256(target),
                }
            )
    return records


def _artifact(path: Path, out: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve().relative_to(out.resolve())),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _pil_rgb(value: torch.Tensor | np.ndarray) -> Image.Image:
    array = value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)
    pixels = np.rint(np.clip(array, 0.0, 1.0) * 255.0).astype(np.uint8)
    return Image.fromarray(pixels, mode="RGB")


def _canonical_png(value: np.ndarray) -> bytes:
    stream = io.BytesIO()
    _pil_rgb(value).save(stream, format="PNG", optimize=False)
    return stream.getvalue()


def _save_rgb(path: Path, value: torch.Tensor | np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _pil_rgb(value).save(path)


def _labeled(value: torch.Tensor | np.ndarray, label: str, header: int = 26) -> Image.Image:
    image = _pil_rgb(value)
    result = Image.new("RGB", (image.width, image.height + header), "white")
    result.paste(image, (0, header))
    ImageDraw.Draw(result).text((5, 7), label, fill="black")
    return result


def _sheet(rows: list[list[Image.Image]]) -> Image.Image:
    row_images: list[Image.Image] = []
    for panels in rows:
        width = sum(panel.width for panel in panels)
        height = max(panel.height for panel in panels)
        row = Image.new("RGB", (width, height), "white")
        x = 0
        for panel in panels:
            row.paste(panel, (x, 0))
            x += panel.width
        row_images.append(row)
    width = max(row.width for row in row_images)
    height = sum(row.height for row in row_images)
    result = Image.new("RGB", (width, height), "white")
    y = 0
    for row in row_images:
        result.paste(row, (0, y))
        y += row.height
    return result


def _find_rgb(frame: Path, view_id: str) -> Path:
    matches = [
        path
        for path in (frame / "rgb").iterdir()
        if path.stem.upper() == view_id and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one source RGB for {view_id}, found {matches}")
    return matches[0]


def _crop_from_mask(
    image: torch.Tensor,
    mask: torch.Tensor,
    margin: int,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int]]:
    active = mask.detach().cpu().numpy() > 0.5
    yy, xx = np.nonzero(active)
    if yy.size == 0:
        raise ValueError("packet mask is empty")
    height, width = active.shape
    x0 = max(0, int(xx.min()) - margin)
    x1 = min(width, int(xx.max()) + 1 + margin)
    y0 = max(0, int(yy.min()) - margin)
    y1 = min(height, int(yy.max()) + 1 + margin)
    rgb = np.ascontiguousarray(image.detach().cpu().numpy()[y0:y1, x0:x1], dtype=np.float32)
    alpha = np.ascontiguousarray(active[y0:y1, x0:x1])
    return rgb, alpha, (x0, y0, x1 - x0, y1 - y0)


def _camera_scaling_error(reference: Any, scaled: Any, factor: int) -> float:
    values = (
        abs(float(reference.fx) / factor - float(scaled.fx)),
        abs(float(reference.fy) / factor - float(scaled.fy)),
        abs(float(reference.cx) / factor - float(scaled.cx)),
        abs(float(reference.cy) / factor - float(scaled.cy)),
        float((reference.R - scaled.R).abs().max()),
        float((reference.t - scaled.t).abs().max()),
    )
    return max(values)


def _carve_config(profile: DiagnosticProfile) -> CompactCarveConfig:
    return CompactCarveConfig(
        n_init_3d=profile.n_init_3d,
        candidate_multiplier=4,
        anchor_mode="mass_random",
        samples_per_ray=48,
        query_batch_size=4096,
        seed=SEED,
        bounds_scale=0.5,
        min_views=2,
        hull_fraction=0.85,
        coverage_scale=1.0,
        coverage_threshold=0.40,
        color_std_sigma=0.20,
        min_score=0.05,
    )


def _density_config(profile: DiagnosticProfile) -> DensityConfig:
    return DensityConfig(
        start_iter=100,
        stop_iter=1_000,
        every=100,
        grad_threshold=8e-4,
        absgrad=True,
        split_scale_frac=0.01,
        split_factor=1.6,
        prune_opacity=0.005,
        prune_scale_frac=0.1,
        max_gaussians=profile.density_max_gaussians,
        opacity_reset_every=1_000,
        opacity_reset_value=0.011,
        revised_opacity=True,
    )


def _train_config(profile: DiagnosticProfile) -> TrainConfig:
    return TrainConfig(
        iterations=profile.iterations,
        rasterizer="gsplat",
        device="cuda",
        densify=profile.densify,
        density_strategy="gsplat-default" if profile.densify else "classic",
        density=_density_config(profile),
        eval_every=profile.eval_every,
        checkpoint_policy="final",
        target_sh_degree=3,
        sh_degree_interval=250,
        mask_alpha_lambda=profile.mask_alpha_lambda,
        outside_alpha_lambda=profile.outside_alpha_lambda,
        use_masks=True,
        random_background=True,
        packed=False,
        antialiased=True,
        record_train_metrics=True,
        validate_render_finite=True,
        seed=SEED,
    )


def _polish_config(profile: DiagnosticProfile) -> TrainConfig | None:
    if profile.polish_iterations <= 0:
        return None
    if (
        profile.polish_mask_alpha_lambda is None
        or profile.polish_outside_alpha_lambda is None
        or profile.polish_lr_factor is None
    ):
        raise ValueError("polish profiles require mask, outside-alpha, and LR factors")
    base = _train_config(profile)
    factor = profile.polish_lr_factor
    return TrainConfig(
        iterations=profile.polish_iterations,
        lr_means=base.lr_means * factor,
        lr_quats=base.lr_quats * factor,
        lr_scales=base.lr_scales * factor,
        lr_opacity=base.lr_opacity * factor,
        lr_sh=base.lr_sh * factor,
        lr_sh_rest=base.lr_sh_rest * factor,
        ssim_lambda=base.ssim_lambda,
        rasterizer="gsplat",
        device="cuda",
        densify=False,
        density_strategy="classic",
        density=_density_config(profile),
        eval_every=50,
        checkpoint_policy="final",
        target_sh_degree=3,
        sh_degree_interval=250,
        mask_alpha_lambda=profile.polish_mask_alpha_lambda,
        outside_alpha_lambda=profile.polish_outside_alpha_lambda,
        use_masks=True,
        random_background=True,
        packed=False,
        antialiased=True,
        record_train_metrics=True,
        validate_render_finite=True,
        seed=SEED + 1,
        iteration_offset=profile.iterations,
        schedule_iterations=profile.iterations + profile.polish_iterations,
    )


def _gradient_mae(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> float:
    valid_x = (mask[:, 1:] > 0.5) & (mask[:, :-1] > 0.5)
    valid_y = (mask[1:, :] > 0.5) & (mask[:-1, :] > 0.5)
    errors = []
    if bool(valid_x.any()):
        pred_dx = prediction[:, 1:] - prediction[:, :-1]
        target_dx = target[:, 1:] - target[:, :-1]
        errors.append((pred_dx - target_dx).abs()[valid_x].reshape(-1))
    if bool(valid_y.any()):
        pred_dy = prediction[1:] - prediction[:-1]
        target_dy = target[1:] - target[:-1]
        errors.append((pred_dy - target_dy).abs()[valid_y].reshape(-1))
    return float(torch.cat(errors).mean()) if errors else 0.0


def _evaluate_indices(
    scene: Any,
    model: Any,
    renderer: Any,
    indices: list[int],
) -> dict[str, Any]:
    per_view: list[dict[str, Any]] = []
    device = model.means.device
    with torch.no_grad():
        for index in indices:
            target = scene.images[index].to(device)
            mask = scene.masks[index].to(device).clamp(0.0, 1.0)
            camera = scene.cameras[index].to(device)
            output = renderer.render(model, camera)
            prediction = output.color.clamp(0.0, 1.0)
            values: dict[str, Any] = image_metrics(prediction, target, mask)
            pred_crop = masked_crop(prediction, mask)
            target_crop = masked_crop(target, mask)
            values["ms_ssim"] = ms_ssim(pred_crop, target_crop)
            try:
                values["lpips"] = LPIPS.distance(pred_crop, target_crop)
            except Exception as error:  # diagnostic keeps an explicit unavailable metric
                values["lpips"] = None
                values["lpips_error"] = f"{type(error).__name__}: {error}"[:300]
            foreground = mask > 0.5
            error = prediction - target
            active = error[foreground].reshape(-1)
            absolute = active.abs()
            values.update(
                {
                    "mse_fg": float(active.square().mean()),
                    "mae_fg": float(absolute.mean()),
                    "p95_abs_fg": float(torch.quantile(absolute, 0.95)),
                    "p99_abs_fg": float(torch.quantile(absolute, 0.99)),
                    "max_abs_fg": float(absolute.max()),
                    "gradient_mae_fg": _gradient_mae(prediction, target, mask),
                }
            )
            predicted_alpha = output.alpha > 0.5
            intersection = (predicted_alpha & foreground).sum()
            union = (predicted_alpha | foreground).sum().clamp_min(1)
            values.update(
                {
                    "alpha_iou": float(intersection / union),
                    "alpha_inside": float(output.alpha[foreground].mean()),
                    "alpha_outside": float(output.alpha[~foreground].mean()),
                    "view_id": scene.view_names[index],
                }
            )
            per_view.append(values)
    numeric_keys = sorted(
        {
            key
            for record in per_view
            for key, value in record.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
    )
    aggregate = {}
    for key in numeric_keys:
        present = [float(record[key]) for record in per_view if record.get(key) is not None]
        if present:
            aggregate[key] = float(np.mean(present))
    if any(record.get("lpips") is None for record in per_view):
        aggregate["lpips"] = None
    return {"aggregate": aggregate, "per_view": per_view}


def _evaluate_model(scene: Any, model: Any, renderer: Any) -> dict[str, Any]:
    return {
        "train": _evaluate_indices(scene, model, renderer, list(scene.training_views)),
        "heldout": _evaluate_indices(scene, model, renderer, list(scene.testing_views)),
    }


def _build_candidate_inputs(
    arm: str,
    structural_count: int,
    packet_scene: Any,
    bounds_hint: tuple[torch.Tensor, float],
    frame: Path,
    out: Path,
    train_ids: tuple[str, ...],
) -> tuple[ReconstructionInputs, list[Any], dict[str, Any]]:
    arm_root = out / "arms" / arm / "inputs"
    records = []
    views = []
    started = time.perf_counter()
    for local_index, (view_id, image, mask, camera) in enumerate(
        zip(
            train_ids,
            packet_scene.images,
            packet_scene.masks,
            packet_scene.cameras,
            strict=True,
        )
    ):
        source_path = _find_rgb(frame, view_id)
        source_payload = source_path.read_bytes()
        crop, alpha, (x, y, width, height) = _crop_from_mask(
            image, mask, PACKET_CROP_MARGIN
        )
        transform = CanvasCropTransform(
            camera.width,
            camera.height,
            x,
            y,
            width,
            height,
        )
        config = CodecNativeFieldConfig(
            appearance_codec="webp",
            appearance_quality=92,
            lattice_sigma_px=0.45,
            lattice_radius_px=3,
            lattice_prefilter_steps=8,
            structural_count=structural_count,
            structural_seed=local_index,
        )
        encode_started = time.perf_counter()
        packet = build_codec_native_field(
            crop,
            config=config,
            mask=alpha,
            canvas_crop=transform,
            source_payload=source_payload,
        )
        encode_seconds = time.perf_counter() - encode_started
        view_root = arm_root / view_id
        view_root.mkdir(parents=True, exist_ok=True)
        packet_path = view_root / f"{view_id}.sgdp"
        ledger = packet.save(packet_path)
        decode_started = time.perf_counter()
        cold = CodecNativeField.load(packet_path)
        decode_seconds = time.perf_counter() - decode_started
        adapter_started = time.perf_counter()
        paired = make_realtime_gs_view(
            cold,
            device="cpu",
            query_device="cuda",
        )
        torch.cuda.synchronize()
        adapter_seconds = time.perf_counter() - adapter_started
        views.append(paired)

        source_crop_path = view_root / "source_crop.png"
        decoded_path = view_root / "decoded_matted.png"
        error_path = view_root / "decoded_error_x8.png"
        _save_rgb(source_crop_path, crop * alpha[..., None])
        decoded_matted = cold.decoded_appearance * alpha[..., None]
        _save_rgb(decoded_path, decoded_matted)
        _save_rgb(error_path, np.abs(decoded_matted - crop * alpha[..., None]) * 8.0)
        canonical = _canonical_png(crop)
        pixel_error = cold.decoded_appearance.astype(np.float64) - crop.astype(np.float64)
        mse = float(np.square(pixel_error[alpha]).mean())
        records.append(
            {
                "view_id": view_id,
                "source_rgb": {
                    "path": str(source_path),
                    "bytes": len(source_payload),
                    "sha256": _bytes_sha256(source_payload),
                },
                "loaded_image_sha256": _tensor_sha256(image),
                "loaded_mask_sha256": _tensor_sha256(mask),
                "canvas": [camera.width, camera.height],
                "crop": [x, y, width, height],
                "foreground_pixels": int(alpha.sum()),
                "canonical_crop_png_bytes": len(canonical),
                "canonical_crop_png_sha256": _bytes_sha256(canonical),
                "config": dataclasses.asdict(config),
                "packet": _artifact(packet_path, out),
                "packet_ledger": dataclasses.asdict(ledger),
                "source_crop": _artifact(source_crop_path, out),
                "decoded_matted": _artifact(decoded_path, out),
                "decoded_error_x8": _artifact(error_path, out),
                "decoded_crop_psnr_db": (
                    120.0 if mse <= 1e-12 else float(-10.0 * np.log10(mse))
                ),
                "encode_seconds": encode_seconds,
                "cold_decode_seconds": decode_seconds,
                "adapter_index_seconds": adapter_seconds,
                "index_entries": int(paired.query_backend.n_entries),
                "index_payload_bytes": int(paired.query_backend.payload_bytes),
            }
        )
    inputs = ReconstructionInputs(
        observations=[view.structural_field for view in views],
        cameras=list(packet_scene.cameras),
        view_names=list(train_ids),
        bounds_hint=(bounds_hint[0].clone(), bounds_hint[1]),
        name=f"core016-{arm}",
    )
    complete_bytes = sum(record["packet"]["bytes"] for record in records)
    return inputs, [view.query_backend for view in views], {
        "kind": "codec_native_dual_plane_v2",
        "structural_count_per_view": structural_count,
        "view_count": len(records),
        "complete_input_bytes": complete_bytes,
        "canonical_crop_png_bytes": sum(
            record["canonical_crop_png_bytes"] for record in records
        ),
        "raw_source_rgb_bytes": sum(record["source_rgb"]["bytes"] for record in records),
        "input_build_seconds": time.perf_counter() - started,
        "appearance_encode_seconds": sum(record["encode_seconds"] for record in records),
        "cold_decode_seconds": sum(record["cold_decode_seconds"] for record in records),
        "adapter_index_seconds": sum(record["adapter_index_seconds"] for record in records),
        "index_entries": sum(record["index_entries"] for record in records),
        "index_payload_bytes": sum(record["index_payload_bytes"] for record in records),
        "views": records,
    }


def _build_control_inputs(
    compact: CompactDataset,
    carve: CompactCarveConfig,
    frame: Path,
    train_ids: tuple[str, ...],
) -> tuple[ReconstructionInputs, list[Any], dict[str, Any]]:
    by_name = {view.view_id: view for view in compact.views}
    selected = [by_name[name] for name in train_ids]
    inputs = ReconstructionInputs(
        observations=[view.observation for view in selected],
        cameras=[view.camera for view in selected],
        view_names=list(train_ids),
        bounds_hint=(compact.bounds_hint[0].clone(), compact.bounds_hint[1]),
        name="core016-rtgsv-control",
    )
    started = time.perf_counter()
    backends = build_query_backends(inputs.observations, carve, device="cuda")
    torch.cuda.synchronize()
    index_seconds = time.perf_counter() - started
    records = []
    for view, backend in zip(selected, backends, strict=True):
        path = frame / "gaussians2d" / f"{view.view_id}.rtgsv"
        records.append(
            {
                "view_id": view.view_id,
                "container": {
                    "path": str(path),
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                },
                "n_gaussians": view.observation.n,
                "fit_window": list(view.observation.fit_window),
                "index_entries": int(backend.n_entries),
                "index_payload_bytes": int(backend.payload_bytes),
            }
        )
    return inputs, backends, {
        "kind": "existing_rtgsv_control",
        "view_count": len(records),
        "complete_input_bytes": sum(record["container"]["bytes"] for record in records),
        "cold_decode_seconds": None,
        "adapter_index_seconds": index_seconds,
        "index_entries": sum(record["index_entries"] for record in records),
        "index_payload_bytes": sum(record["index_payload_bytes"] for record in records),
        "views": records,
    }


def _save_heldout_visuals(
    arm: str,
    scene: Any,
    initial: Any,
    final: Any,
    renderer: Any,
    out: Path,
) -> dict[str, Any]:
    root = out / "arms" / arm / "visuals"
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    artifacts = {}
    initial_cuda = initial.to("cuda")
    final_cuda = final.to("cuda")
    with torch.no_grad():
        for index in scene.testing_views:
            view_id = scene.view_names[index]
            target = scene.images[index].cuda()
            mask = scene.masks[index].cuda().clamp(0.0, 1.0)
            target_matted = target * mask[..., None]
            camera = scene.cameras[index].to("cuda")
            init_render = renderer.render(initial_cuda, camera).color.clamp(0.0, 1.0)
            final_render = renderer.render(final_cuda, camera).color.clamp(0.0, 1.0)
            error = (final_render - target_matted).abs().mul(4.0).clamp(0.0, 1.0)
            paths = {
                "target": root / f"{view_id}_target.png",
                "initial": root / f"{view_id}_initial.png",
                "final": root / f"{view_id}_final.png",
                "error_x4": root / f"{view_id}_error_x4.png",
            }
            _save_rgb(paths["target"], target_matted)
            _save_rgb(paths["initial"], init_render)
            _save_rgb(paths["final"], final_render)
            _save_rgb(paths["error_x4"], error)
            artifacts[view_id] = {
                name: _artifact(path, out) for name, path in paths.items()
            }
            rows.append(
                [
                    _labeled(target_matted, f"{view_id} target"),
                    _labeled(init_render, "initial"),
                    _labeled(final_render, "final"),
                    _labeled(error, "|final-target| x4"),
                ]
            )
    sheet_path = root / "heldout_contact_sheet.png"
    _sheet(rows).save(sheet_path)
    return {"contact_sheet": _artifact(sheet_path, out), "views": artifacts}


def _run_arm(
    profile: DiagnosticProfile,
    arm: str,
    inputs: ReconstructionInputs,
    backends: list[Any],
    input_record: dict[str, Any],
    scene: Any,
    renderer: Any,
    out: Path,
) -> dict[str, Any]:
    root = out / "arms" / arm
    root.mkdir(parents=True, exist_ok=True)
    carve = _carve_config(profile)
    print(f"[lift] {arm}", flush=True)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    lift_started = time.perf_counter()
    initialization = CompactCarveInitializer(carve).initialize(
        inputs,
        backends=backends,
        progress_callback=make_placement_progress_printer(
            every_batches=10,
            every_seconds=20.0,
        ),
    )
    torch.cuda.synchronize()
    lift_seconds = time.perf_counter() - lift_started
    initial = initialization.gaussians.to("cpu")
    initial_npz = root / "gaussians_init.npz"
    initial_ply = root / "gaussians_init.ply"
    initial.save_npz(initial_npz)
    initial.save_ply(initial_ply)

    curves = [
        {
            "step": 0,
            "optimization_elapsed_seconds": 0.0,
            "metrics": _evaluate_model(scene, initial.to("cuda"), renderer),
        }
    ]

    def checkpoint(snapshot: Any, step: int) -> None:
        print(f"  [eval] {arm} step={step}", flush=True)
        curves.append(
            {
                "step": step,
                "optimization_elapsed_seconds": None,
                "metrics": _evaluate_model(scene, snapshot, renderer),
            }
        )

    train_config = _train_config(profile)
    print(f"[refine] {arm} N={initial.n} steps={profile.iterations}", flush=True)
    train_started = time.perf_counter()
    final, history = Trainer(train_config).train(
        scene,
        initialization.gaussians,
        checkpoint_callback=checkpoint,
    )
    main_native_seconds = float(history["elapsed"][-1][1])
    polish_config = _polish_config(profile)
    polish_history = None
    if polish_config is not None:
        print(
            f"[polish] {arm} N={final.n} steps={profile.polish_iterations}",
            flush=True,
        )
        final, polish_history = Trainer(polish_config).train(
            scene,
            final,
            checkpoint_callback=checkpoint,
        )
    torch.cuda.synchronize()
    train_wall_seconds = time.perf_counter() - train_started
    final_cpu = final.to("cpu")
    elapsed_by_step = {int(step): float(value) for step, value in history["elapsed"]}
    if polish_history is not None:
        elapsed_by_step.update(
            {
                int(step): main_native_seconds + float(value)
                for step, value in polish_history["elapsed"]
            }
        )
    for record in curves:
        if record["step"] in elapsed_by_step:
            record["optimization_elapsed_seconds"] = elapsed_by_step[record["step"]]
    history_path = root / "training_history.json"
    curves_path = root / "checkpoint_metrics.json"
    history_record = dict(history)
    history_record["polish_stage"] = polish_history
    _write_json(history_path, history_record)
    _write_json(curves_path, curves)
    final_npz = root / "gaussians_final.npz"
    final_ply = root / "gaussians_final.ply"
    final_cpu.save_npz(final_npz)
    final_cpu.save_ply(final_ply)
    visuals = _save_heldout_visuals(
        arm,
        scene,
        initial,
        final_cpu,
        renderer,
        out,
    )
    query_records = []
    for index, backend in enumerate(backends):
        structural = getattr(backend, "structural_backend", backend)
        query_records.append(
            {
                "view_id": profile.train_ids[index],
                "backend_kind": type(backend).__name__,
                "structural_backend_kind": type(structural).__name__,
                "n_entries": int(getattr(backend, "n_entries", 0)),
                "payload_bytes": int(getattr(backend, "payload_bytes", 0)),
                "total_pairs_evaluated": int(
                    getattr(backend, "total_pairs_evaluated", 0)
                ),
                "peak_pair_chunk": int(getattr(backend, "peak_pair_chunk", 0)),
            }
        )
    return {
        "arm": arm,
        "label": ARM_LABELS[arm],
        "status": "ok",
        "inputs": input_record,
        "carve_config": dataclasses.asdict(carve),
        "train_config": dataclasses.asdict(train_config),
        "polish_config": (
            None if polish_config is None else dataclasses.asdict(polish_config)
        ),
        "lift_seconds": lift_seconds,
        "lift_diagnostics": initialization.diagnostics,
        "query_backends": query_records,
        "init_n_gaussians": initial.n,
        "final_n_gaussians": final_cpu.n,
        "training_wall_seconds_including_metrics": train_wall_seconds,
        "training_native_elapsed_seconds": main_native_seconds
        + (
            0.0
            if polish_history is None
            else float(polish_history["elapsed"][-1][1])
        ),
        "peak_vram_gb": max(
            float(history["peak_vram_gb"]),
            0.0
            if polish_history is None
            else float(polish_history["peak_vram_gb"]),
        ),
        "initial_metrics": curves[0]["metrics"],
        "final_metrics": curves[-1]["metrics"],
        "curves": _artifact(curves_path, out),
        "history": _artifact(history_path, out),
        "models": {
            "initial_npz": _artifact(initial_npz, out),
            "initial_ply": _artifact(initial_ply, out),
            "final_npz": _artifact(final_npz, out),
            "final_ply": _artifact(final_ply, out),
        },
        "visuals": visuals,
        "curve_rows": curves,
    }


def _curve_metric(record: dict[str, Any], split: str, metric: str) -> float | None:
    value = record["metrics"][split]["aggregate"].get(metric)
    return None if value is None else float(value)


def _terminal_row(record: dict[str, Any]) -> dict[str, Any]:
    if record["status"] != "ok":
        return {
            "arm": record["arm"],
            "label": record["label"],
            "status": record["status"],
            "error": record.get("error"),
        }
    heldout = record["final_metrics"]["heldout"]["aggregate"]
    train = record["final_metrics"]["train"]["aggregate"]
    return {
        "arm": record["arm"],
        "label": record["label"],
        "status": "ok",
        "input_bytes": record["inputs"]["complete_input_bytes"],
        "input_mib": record["inputs"]["complete_input_bytes"] / 1024**2,
        "input_rows_per_view": (
            record["inputs"].get("structural_count_per_view")
            or float(
                np.mean([view["n_gaussians"] for view in record["inputs"]["views"]])
            )
        ),
        "lift_seconds": record["lift_seconds"],
        "training_native_seconds": record["training_native_elapsed_seconds"],
        "pipeline_available_seconds": (
            float(record["inputs"].get("input_build_seconds") or 0.0)
            + record["lift_seconds"]
            + record["training_native_elapsed_seconds"]
        ),
        "final_n_gaussians": record["final_n_gaussians"],
        "final_npz_bytes": record["models"]["final_npz"]["bytes"],
        "final_ply_bytes": record["models"]["final_ply"]["bytes"],
        "train_psnr_fg": train.get("psnr_fg"),
        "train_ssim_crop": train.get("ssim_crop"),
        "train_ms_ssim": train.get("ms_ssim"),
        "train_lpips": train.get("lpips"),
        "heldout_psnr_fg": heldout.get("psnr_fg"),
        "heldout_psnr_crop": heldout.get("psnr_crop"),
        "heldout_ssim_crop": heldout.get("ssim_crop"),
        "heldout_ms_ssim": heldout.get("ms_ssim"),
        "heldout_lpips": heldout.get("lpips"),
        "heldout_mse_fg": heldout.get("mse_fg"),
        "heldout_mae_fg": heldout.get("mae_fg"),
        "heldout_p99_abs_fg": heldout.get("p99_abs_fg"),
        "heldout_max_abs_fg": heldout.get("max_abs_fg"),
        "heldout_gradient_mae_fg": heldout.get("gradient_mae_fg"),
        "heldout_alpha_iou": heldout.get("alpha_iou"),
        "peak_vram_gb": record["peak_vram_gb"],
    }


def _decorate_convergence(records: list[dict[str, Any]], rows: list[dict[str, Any]]) -> None:
    by_arm = {row["arm"]: row for row in rows if row.get("status") == "ok"}
    control_target = by_arm.get("rtgsv_control", {}).get("heldout_psnr_fg")
    if control_target is None:
        return
    for record in records:
        if record["status"] != "ok":
            continue
        row = by_arm[record["arm"]]
        first = next(
            (
                curve
                for curve in record["curve_rows"]
                if (_curve_metric(curve, "heldout", "psnr_fg") or -float("inf"))
                >= control_target
            ),
            None,
        )
        row["step_to_control_final_heldout_psnr"] = None if first is None else first["step"]
        row["seconds_to_control_final_heldout_psnr"] = (
            None if first is None else first["optimization_elapsed_seconds"]
        )
        steps = np.asarray([curve["step"] for curve in record["curve_rows"]], dtype=np.float64)
        values = np.asarray(
            [_curve_metric(curve, "heldout", "psnr_fg") for curve in record["curve_rows"]],
            dtype=np.float64,
        )
        row["heldout_psnr_auc_over_steps"] = float(
            np.trapezoid(values, steps) / max(float(steps[-1]), 1.0)
        )


def _decision(
    profile: DiagnosticProfile,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    by_arm = {row["arm"]: row for row in rows if row.get("status") == "ok"}
    missing = [name for name in profile.arms if name not in by_arm]
    if missing:
        return {"advance": False, "reason": f"missing successful arms: {missing}", "gates": {}}
    control = by_arm["rtgsv_control"]
    compact = by_arm["dual_webp92_n512"]
    if profile.name in {"density", "full", "matched10k", "silhouette", "latepolish"}:
        full_capture = profile.name in {"full", "matched10k", "silhouette", "latepolish"}
        absolute_psnr_floor = 24.0 if full_capture else 22.5
        absolute_alpha_floor = (
            0.95
            if profile.name in {"silhouette", "latepolish"}
            else (0.93 if full_capture else 0.90)
        )
        lpips_control = control.get("heldout_lpips")
        lpips_candidate = compact.get("heldout_lpips")
        gates = {
            "input_at_least_3x_smaller": (
                control["input_bytes"] / compact["input_bytes"] >= 3.0
            ),
            "heldout_psnr_within_0_5db": (
                compact["heldout_psnr_fg"] >= control["heldout_psnr_fg"] - 0.5
            ),
            "heldout_ms_ssim_within_0_01": (
                compact["heldout_ms_ssim"] >= control["heldout_ms_ssim"] - 0.01
            ),
            "heldout_lpips_within_0_02": (
                lpips_control is not None
                and lpips_candidate is not None
                and lpips_candidate <= lpips_control + 0.02
            ),
            "heldout_alpha_iou_within_0_02": (
                compact["heldout_alpha_iou"] >= control["heldout_alpha_iou"] - 0.02
            ),
            f"absolute_heldout_psnr_at_least_{absolute_psnr_floor:g}db": (
                compact["heldout_psnr_fg"] >= absolute_psnr_floor
            ),
            f"absolute_heldout_alpha_iou_at_least_{absolute_alpha_floor:.2f}": (
                compact["heldout_alpha_iou"] >= absolute_alpha_floor
            ),
            f"both_final_counts_at_most_{profile.density_max_gaussians}": all(
                row["final_n_gaussians"] <= profile.density_max_gaussians
                for row in by_arm.values()
            ),
            "candidate_final_count_within_10_percent_of_control": (
                compact["final_n_gaussians"] <= 1.10 * control["final_n_gaussians"]
            ),
        }
        if profile.reference_candidate_psnr_min_db is not None:
            gates["candidate_psnr_retains_frozen_v4_within_0_2db"] = (
                compact["heldout_psnr_fg"]
                >= profile.reference_candidate_psnr_min_db
            )
        if profile.reference_candidate_gradient_mae_max is not None:
            gates["candidate_gradient_mae_no_worse_than_frozen_v4"] = (
                compact["heldout_gradient_mae_fg"]
                <= profile.reference_candidate_gradient_mae_max
            )
        return {
            # Native-pixel visual review is deliberately a separate mandatory disposition.
            "advance": False,
            "scalar_pass": all(gates.values()),
            "manual_visual_review_required": True,
            "gates": gates,
            "control_over_n512_input_ratio": (
                control["input_bytes"] / compact["input_bytes"]
            ),
            "n512_minus_control_heldout_psnr_db": (
                compact["heldout_psnr_fg"] - control["heldout_psnr_fg"]
            ),
            "n512_minus_control_heldout_ms_ssim": (
                compact["heldout_ms_ssim"] - control["heldout_ms_ssim"]
            ),
            "n512_minus_control_heldout_lpips": (
                None
                if lpips_control is None or lpips_candidate is None
                else lpips_candidate - lpips_control
            ),
            "n512_minus_control_heldout_alpha_iou": (
                compact["heldout_alpha_iou"] - control["heldout_alpha_iou"]
            ),
        }

    candidate = by_arm["dual_webp92_n2048"]
    lpips_control = control.get("heldout_lpips")
    lpips_candidate = candidate.get("heldout_lpips")
    gates = {
        "input_at_least_3x_smaller": (
            control["input_bytes"] / candidate["input_bytes"] >= 3.0
        ),
        "heldout_psnr_within_1db": (
            candidate["heldout_psnr_fg"] >= control["heldout_psnr_fg"] - 1.0
        ),
        "heldout_ms_ssim_within_0_02": (
            candidate["heldout_ms_ssim"] >= control["heldout_ms_ssim"] - 0.02
        ),
        "heldout_lpips_within_0_03": (
            lpips_control is not None
            and lpips_candidate is not None
            and lpips_candidate <= lpips_control + 0.03
        ),
        "fixed_topology_cardinality_preserved": all(
            row["final_n_gaussians"] == profile.n_init_3d for row in by_arm.values()
        ),
    }
    return {
        "advance": all(gates.values()),
        "gates": gates,
        "control_over_n2048_input_ratio": control["input_bytes"] / candidate["input_bytes"],
        "n2048_minus_control_heldout_psnr_db": (
            candidate["heldout_psnr_fg"] - control["heldout_psnr_fg"]
        ),
        "n2048_minus_control_heldout_ms_ssim": (
            candidate["heldout_ms_ssim"] - control["heldout_ms_ssim"]
        ),
        "n2048_minus_control_heldout_lpips": (
            None
            if lpips_control is None or lpips_candidate is None
            else lpips_candidate - lpips_control
        ),
        "prefer_n512": (
            compact["heldout_psnr_fg"] >= candidate["heldout_psnr_fg"] - 0.25
        ),
        "n512_minus_n2048_heldout_psnr_db": (
            compact["heldout_psnr_fg"] - candidate["heldout_psnr_fg"]
        ),
    }


def _write_metric_tables(out: Path, rows: list[dict[str, Any]]) -> None:
    _write_json(out / "metrics.json", {"schema": "core016.multiview.metrics.v1", "rows": rows})
    with (out / "metrics.jsonl").open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
    keys = sorted({key for row in rows for key in row})
    with (out / "metrics.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _plot_curves(records: list[dict[str, Any]], out: Path) -> Path:
    specs = (
        ("heldout", "psnr_fg", "reporting-view foreground PSNR (dB)"),
        ("heldout", "ms_ssim", "reporting-view MS-SSIM"),
        ("heldout", "lpips", "reporting-view LPIPS (lower better)"),
        ("heldout", "gradient_mae_fg", "reporting-view gradient MAE (lower better)"),
        ("heldout", "alpha_iou", "reporting-view alpha IoU"),
        ("train", "psnr_fg", "training foreground PSNR (dB)"),
    )
    figure, axes = plt.subplots(3, 2, figsize=(13, 13), constrained_layout=True)
    for axis, (split, metric, title) in zip(axes.flat, specs, strict=True):
        for record in records:
            if record["status"] != "ok":
                continue
            x = [curve["step"] for curve in record["curve_rows"]]
            y = [_curve_metric(curve, split, metric) for curve in record["curve_rows"]]
            if any(value is None for value in y):
                continue
            axis.plot(x, y, marker="o", markersize=3, label=ARM_LABELS[record["arm"]])
        axis.set_title(title)
        axis.set_xlabel("attempted optimization steps")
        axis.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=8)
    path = out / "all_metric_curves.png"
    figure.savefig(path, dpi=170)
    plt.close(figure)
    return path


def _plot_rate_quality(
    profile: DiagnosticProfile,
    rows: list[dict[str, Any]],
    out: Path,
) -> Path:
    figure, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for row in rows:
        if row.get("status") != "ok":
            continue
        axis.scatter(row["input_mib"], row["heldout_psnr_fg"], s=70)
        axis.annotate(row["label"], (row["input_mib"], row["heldout_psnr_fg"]), xytext=(5, 5), textcoords="offset points", fontsize=8)
    axis.set_xlabel(f"complete {len(profile.train_ids)}-view reconstruction input (MiB)")
    axis.set_ylabel("final reporting-view foreground PSNR (dB)")
    axis.set_title("CORE-016 downstream rate-quality diagnostic")
    axis.grid(alpha=0.25)
    path = out / "rate_quality.png"
    figure.savefig(path, dpi=170)
    plt.close(figure)
    return path


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _write_html(
    profile: DiagnosticProfile,
    out: Path,
    rows: list[dict[str, Any]],
    records: list[dict[str, Any]],
    decision: dict[str, Any],
) -> None:
    table_rows = []
    by_record = {record["arm"]: record for record in records}
    for row in rows:
        record = by_record[row["arm"]]
        if row.get("status") != "ok":
            table_rows.append(
                f"<tr><td>{escape(row['label'])}</td><td colspan='11'>ERROR: "
                f"{escape(str(row.get('error')))}</td></tr>"
            )
            continue
        sheet = record["visuals"]["contact_sheet"]["path"]
        table_rows.append(
            "<tr>"
            f"<td>{escape(row['label'])}</td>"
            f"<td>{int(row['input_bytes']):,}</td>"
            f"<td>{_fmt(row['heldout_psnr_fg'], 3)}</td>"
            f"<td>{_fmt(row['heldout_ssim_crop'])}</td>"
            f"<td>{_fmt(row['heldout_ms_ssim'])}</td>"
            f"<td>{_fmt(row['heldout_lpips'])}</td>"
            f"<td>{_fmt(row['heldout_gradient_mae_fg'], 5)}</td>"
            f"<td>{_fmt(row['heldout_alpha_iou'])}</td>"
            f"<td>{_fmt(row['lift_seconds'], 2)}</td>"
            f"<td>{_fmt(row['training_native_seconds'], 2)}</td>"
            f"<td>{int(row['final_n_gaussians'])}</td>"
            f"<td><a href='{escape(sheet)}'>target / init / final / error</a></td>"
            "</tr>"
        )
    gate_rows = "".join(
        f"<li class='{('pass' if passed else 'fail')}'>{escape(name)}: "
        f"{('PASS' if passed else 'FAIL')}</li>"
        for name, passed in decision.get("gates", {}).items()
    )
    cards = []
    for record in records:
        if record["status"] != "ok":
            continue
        sheet = record["visuals"]["contact_sheet"]["path"]
        cards.append(
            f"<section><h3>{escape(record['label'])}</h3>"
            f"<a href='{escape(sheet)}'><img src='{escape(sheet)}'></a>"
            f"<p><a href='{escape(record['curves']['path'])}'>checkpoint metrics JSON</a> · "
            f"<a href='{escape(record['history']['path'])}'>trainer history</a> · "
            f"<a href='{escape(record['models']['final_ply']['path'])}'>final PLY</a> · "
            f"<a href='{escape(record['models']['final_npz']['path'])}'>final NPZ</a></p></section>"
        )
    html = f"""<!doctype html>
<meta charset='utf-8'><title>CORE-016 multiview downstream diagnostic</title>
<style>
body{{font:15px system-ui,sans-serif;max-width:1500px;margin:28px auto;padding:0 20px;color:#18212b}}
.warning{{background:#fff4d6;border-left:5px solid #d98b00;padding:12px}} .pass{{color:#087d40}} .fail{{color:#b42318}}
table{{border-collapse:collapse;width:100%;font-size:13px}} th,td{{border:1px solid #ccd3da;padding:6px;text-align:right}} th:first-child,td:first-child{{text-align:left}}
img{{max-width:100%;height:auto;border:1px solid #ccd3da}} section{{margin:30px 0}} a{{color:#005bbb}}
</style>
<h1>CORE-016: source-grounded multiview downstream diagnostic ({escape(profile.name)})</h1>
<p class='warning'><strong>Diagnostic only.</strong> One exposed Janelle frame, one seed, packet input downscale 4, reporting downscale 8, {escape(profile.scope)}. This cannot promote a default or satisfy BENCH-019. Every displayed target comes from common calibrated RGB/masks, never from a candidate or control teacher. The density profile cannot advance until native-pixel visual review is recorded separately.</p>
<p><a href='manifest.json'>manifest</a> · <a href='metrics.json'>metrics JSON</a> · <a href='metrics.jsonl'>JSONL</a> · <a href='metrics.csv'>CSV</a> · <a href='decision.json'>gate decision</a> · <a href='plan.json'>frozen run plan</a></p>
<h2>Development gate: {('ADVANCE' if decision.get('advance') else 'DO NOT ADVANCE')}</h2><ul>{gate_rows}</ul>
<h2>Terminal results</h2>
<table><thead><tr><th>arm</th><th>input B</th><th>held PSNR-FG</th><th>held SSIM</th><th>held MS-SSIM</th><th>held LPIPS</th><th>gradient MAE</th><th>alpha IoU</th><th>lift s</th><th>train s</th><th>3D N</th><th>visual</th></tr></thead><tbody>{''.join(table_rows)}</tbody></table>
<h2>Curves and rate-quality</h2><p><a href='all_metric_curves.png'><img src='all_metric_curves.png'></a></p><p><a href='rate_quality.png'><img src='rate_quality.png'></a></p>
<h2>Held-out visuals</h2>{''.join(cards)}
"""
    (out / "index.html").write_text(html, encoding="utf-8")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--profile", choices=tuple(PROFILES), default="fixed")
    parser.add_argument("--frame", type=Path, default=DEFAULT_FRAME)
    parser.add_argument("--realtime-gs-root", type=Path, default=DEFAULT_RTGS_ROOT)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    profile = PROFILES[args.profile]
    all_ids = profile.train_ids + profile.heldout_ids
    out = args.out.expanduser().resolve()
    frame = args.frame.expanduser().resolve()
    rtgs_root = args.realtime_gs_root.expanduser().resolve()
    calibration = frame.parent / "calibration_dome.json"
    compact_root = frame / "gaussians2d"
    if out.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {out}")
    if not frame.is_dir() or not calibration.is_file() or not compact_root.is_dir():
        raise FileNotFoundError("Janelle frame, calibration, or compact control is missing")
    if not torch.cuda.is_available():
        raise RuntimeError("the frozen diagnostic requires CUDA")
    out.mkdir(parents=True)
    started = time.perf_counter()
    repositories = {
        "structsplat": _repository_record(STRUCTSPLAT_ROOT),
        "realtime_gs": _repository_record(rtgs_root),
    }
    source_snapshots = _snapshot_sources(out, rtgs_root)
    compact_load_started = time.perf_counter()
    compact = CompactDataset.load(compact_root, device="cpu")
    compact_load_seconds = time.perf_counter() - compact_load_started
    if compact.bounds_hint is None:
        raise RuntimeError("the compact control must provide explicit bounds")
    packet_scene = load_calibrated_scene(
        frame,
        calibration_path=calibration,
        downscale=PACKET_DOWNSCALE,
        view_ids=profile.train_ids,
        test_every=0,
        load_masks=True,
        undistort=True,
    )
    scene = load_calibrated_scene(
        frame,
        calibration_path=calibration,
        downscale=EVALUATION_DOWNSCALE,
        view_ids=all_ids,
        test_every=0,
        load_masks=True,
        undistort=True,
    )
    if packet_scene.masks is None or scene.masks is None:
        raise RuntimeError("the frozen diagnostic requires complete source masks")
    scene.train_indices = list(range(len(profile.train_ids)))
    scene.test_indices = list(range(len(profile.train_ids), len(all_ids)))
    scene.bounds_hint = (compact.bounds_hint[0].clone(), compact.bounds_hint[1])
    scene.name = "frame_00008-core016-source-grounded"
    scene.validate()
    packet_scene.bounds_hint = (compact.bounds_hint[0].clone(), compact.bounds_hint[1])
    by_name = {view.view_id: view for view in compact.views}
    camera_errors = {
        name: {
            "packet_downscale_error": _camera_scaling_error(
                by_name[name].camera, packet_scene.cameras[index], PACKET_DOWNSCALE
            ),
            "evaluation_downscale_error": _camera_scaling_error(
                by_name[name].camera, scene.cameras[index], EVALUATION_DOWNSCALE
            ),
        }
        for index, name in enumerate(profile.train_ids)
    }
    if max(value for row in camera_errors.values() for value in row.values()) != 0.0:
        raise RuntimeError(f"calibrated camera scaling drifted: {camera_errors}")
    input_records = {}
    for index, name in enumerate(all_ids):
        rgb = _find_rgb(frame, name)
        mask_path = frame / "mask" / f"mask_{name}.png"
        input_records[name] = {
            "role": "train" if name in profile.train_ids else "reporting_only",
            "rgb": {"path": str(rgb), "bytes": rgb.stat().st_size, "sha256": _sha256(rgb)},
            "mask": {
                "path": str(mask_path),
                "bytes": mask_path.stat().st_size,
                "sha256": _sha256(mask_path),
            },
            "evaluation_image_tensor_sha256": _tensor_sha256(scene.images[index]),
            "evaluation_mask_tensor_sha256": _tensor_sha256(scene.masks[index]),
        }
    plan = {
        "schema": "core016.multiview.plan.v1",
        "created_utc": dt.datetime.now(dt.UTC).isoformat(),
        "scope": "exposed single-frame single-seed reduced-resolution diagnostic only",
        "profile": dataclasses.asdict(profile),
        "repositories": repositories,
        "environment": _environment(),
        "source_snapshots": source_snapshots,
        "frame": str(frame),
        "calibration": {
            "path": str(calibration),
            "bytes": calibration.stat().st_size,
            "sha256": _sha256(calibration),
        },
        "train_ids": list(profile.train_ids),
        "heldout_ids": list(profile.heldout_ids),
        "packet_downscale": PACKET_DOWNSCALE,
        "evaluation_downscale": EVALUATION_DOWNSCALE,
        "packet_crop_margin": PACKET_CROP_MARGIN,
        "camera_scaling_errors": camera_errors,
        "bounds_hint": {
            "center": compact.bounds_hint[0].tolist(),
            "extent": compact.bounds_hint[1],
        },
        "inputs": input_records,
        "arms": list(profile.arms),
        "carve_config": dataclasses.asdict(_carve_config(profile)),
        "train_config": dataclasses.asdict(_train_config(profile)),
        "polish_config": (
            None
            if _polish_config(profile) is None
            else dataclasses.asdict(_polish_config(profile))
        ),
        "development_gate": (
            {
                "n2048_control_input_ratio_min": 3.0,
                "n2048_control_heldout_psnr_delta_min_db": -1.0,
                "n2048_control_heldout_ms_ssim_delta_min": -0.02,
                "n2048_control_heldout_lpips_delta_max": 0.03,
                "n512_n2048_heldout_psnr_delta_min_db_for_preference": -0.25,
            }
            if profile.name == "fixed"
            else {
                "n512_control_input_ratio_min": 3.0,
                "n512_control_heldout_psnr_delta_min_db": -0.5,
                "n512_control_heldout_ms_ssim_delta_min": -0.01,
                "n512_control_heldout_lpips_delta_max": 0.02,
                "n512_control_heldout_alpha_iou_delta_min": -0.02,
                "n512_absolute_heldout_psnr_min_db": (
                    24.0
                    if profile.name in {"full", "matched10k", "silhouette", "latepolish"}
                    else 22.5
                ),
                "n512_absolute_heldout_alpha_iou_min": (
                    0.95
                    if profile.name in {"silhouette", "latepolish"}
                    else (
                        0.93 if profile.name in {"full", "matched10k"} else 0.90
                    )
                ),
                "max_final_gaussians": profile.density_max_gaussians,
                "n512_final_count_over_control_max": 1.10,
                "reference_candidate_psnr_min_db": (
                    profile.reference_candidate_psnr_min_db
                ),
                "reference_candidate_gradient_mae_max": (
                    profile.reference_candidate_gradient_mae_max
                ),
                "manual_native_pixel_visual_review_required": True,
            }
        ),
    }
    _write_json(out / "plan.json", plan)

    renderer = get_rasterizer(
        "gsplat",
        device=torch.device("cuda"),
        packed=False,
        antialiased=True,
    )
    records = []
    for arm in profile.arms:
        inputs = None
        backends = None
        try:
            print(f"[input] {arm}", flush=True)
            if arm == "rtgsv_control":
                inputs, backends, input_record = _build_control_inputs(
                    compact,
                    _carve_config(profile),
                    frame,
                    profile.train_ids,
                )
                input_record["compact_dataset_load_seconds_all_26_views"] = compact_load_seconds
            else:
                count = 512 if arm.endswith("n512") else 2048
                inputs, backends, input_record = _build_candidate_inputs(
                    arm,
                    count,
                    packet_scene,
                    compact.bounds_hint,
                    frame,
                    out,
                    profile.train_ids,
                )
            record = _run_arm(
                profile,
                arm,
                inputs,
                backends,
                input_record,
                scene,
                renderer,
                out,
            )
        except Exception as error:
            record = {
                "arm": arm,
                "label": ARM_LABELS[arm],
                "status": "error",
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
            }
            print(record["traceback"], file=sys.stderr, flush=True)
        records.append(record)
        _write_json(out / "partial_records.json", records)
        del inputs, backends
        gc.collect()
        torch.cuda.empty_cache()

    rows = [_terminal_row(record) for record in records]
    _decorate_convergence(records, rows)
    decision = _decision(profile, rows)
    _write_metric_tables(out, rows)
    _write_json(out / "decision.json", decision)
    curves_path = _plot_curves(records, out)
    rate_quality_path = _plot_rate_quality(profile, rows, out)
    _write_html(profile, out, rows, records, decision)
    # Avoid duplicating outcome-sized curve rows inside the root manifest.
    manifest_records = []
    for record in records:
        retained = dict(record)
        retained.pop("curve_rows", None)
        manifest_records.append(retained)
    manifest = {
        "schema": "core016.multiview.manifest.v1",
        "status": "ok" if all(record["status"] == "ok" for record in records) else "partial",
        "scope": plan["scope"],
        "plan": _artifact(out / "plan.json", out),
        "records": manifest_records,
        "decision": decision,
        "metrics": {
            name: _artifact(out / name, out)
            for name in ("metrics.json", "metrics.jsonl", "metrics.csv")
        },
        "plots": {
            "all_metric_curves": _artifact(curves_path, out),
            "rate_quality": _artifact(rate_quality_path, out),
        },
        "report": _artifact(out / "index.html", out),
        "total_wall_seconds": time.perf_counter() - started,
    }
    _write_json(out / "manifest.json", manifest)
    print(json.dumps({"decision": decision, "rows": rows}, indent=2), flush=True)
    return 0 if manifest["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
