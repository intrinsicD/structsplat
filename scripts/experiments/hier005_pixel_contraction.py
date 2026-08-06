#!/usr/bin/env python3
"""HIER-005 implicit pixel-field contraction diagnostic.

This is an implementation diagnostic, not a preregistered comparison and not compression
evidence.  It cold-loads the lossless Observation Field V2 output, renders it through the
maintained additive renderer, and writes stable rows that can later be joined to BENCH-020/021
controls.  Estimated payload bytes are explicitly not actual codec bytes.

Exact invocation example::

    PYTHONPATH=src python scripts/experiments/hier005_pixel_contraction.py \
        --images results/datasets/abl004/kodak24/01.png \
        --out results/hier005_kodak01_diagnostic \
        --target-gaussians 512 1024 2048 --device cpu --renderer additive

The method is deterministic and has no random seed.  See
``tasks/HIER-005-implicit-pixel-contraction.md`` for scope and evidence limits.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
from html import escape
import importlib.metadata
import json
import math
import platform
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"})
REPORT_SCHEMA = "structsplat.hier005_pixel_contraction.diagnostic.v1"

# Every non-provenance scalar outcome written by ``_run_one`` is represented here.  The report
# plots these values against the achieved Gaussian count; inputs and fixed configuration values
# (dimensions, source bytes, SSIM window) remain in the tidy ledgers rather than becoming
# uninformative constant curves.
CURVE_SPECS = (
    ("psnr_db", "Foreground-mask PSNR (dB)", False),
    ("ssim", "Full-matted-raster SSIM", False),
    ("ms_ssim", "Full-matted-raster MS-SSIM", False),
    ("lpips", "Full-matted-raster LPIPS (lower is better)", False),
    ("masked_mse", "Masked MSE (lower is better)", True),
    ("artifact_pixel_rmse_q99", "Display PNG foreground pixel RMSE q99", True),
    ("artifact_pixel_rmse_q999", "Display PNG foreground pixel RMSE q99.9", True),
    ("artifact_pixel_rmse_max", "Display PNG foreground pixel RMSE maximum", True),
    (
        "artifact_pixel_rmse_fraction_gt_005",
        "Display PNG foreground fraction with pixel RMSE > 0.05",
        True,
    ),
    (
        "artifact_pixel_rmse_fraction_gt_010",
        "Display PNG foreground fraction with pixel RMSE > 0.10",
        True,
    ),
    ("artifact_patch_rmse_max_3", "Display PNG maximum 3x3 patch RMSE", True),
    ("artifact_patch_rmse_max_7", "Display PNG maximum 7x7 patch RMSE", True),
    ("artifact_patch_rmse_max_15", "Display PNG maximum 15x15 patch RMSE", True),
    ("artifact_patch_rmse_max_31", "Display PNG maximum 31x31 patch RMSE", True),
    ("initial_sse", "Initial SSE", True),
    ("final_sse", "Final SSE (lower is better)", True),
    ("estimated_field_bytes", "Estimated uncoded field bytes", True),
    ("canonical_raw_bytes", "Canonical raw field bytes", True),
    ("lossless_reference_bytes", "Lossless reference NPZ bytes", True),
    ("estimated_bits_per_pixel", "Estimated uncoded field bits/pixel", True),
    ("canonical_raw_bits_per_pixel", "Canonical raw field bits/pixel", True),
    ("lossless_reference_bits_per_pixel", "Lossless reference NPZ bits/pixel", True),
    (
        "estimated_bits_per_active_pixel",
        "Estimated uncoded field bits/active foreground pixel",
        True,
    ),
    (
        "canonical_raw_bits_per_active_pixel",
        "Canonical raw field bits/active foreground pixel",
        True,
    ),
    (
        "lossless_reference_bits_per_active_pixel",
        "Lossless reference NPZ bits/active foreground pixel",
        True,
    ),
    ("source_over_estimated_ratio", "Original source bytes / estimated field bytes", True),
    (
        "source_over_canonical_raw_ratio",
        "Original source bytes / canonical raw field bytes",
        True,
    ),
    (
        "source_over_lossless_reference_ratio",
        "Original source bytes / lossless reference NPZ bytes",
        True,
    ),
    (
        "evaluation_png_over_estimated_ratio",
        "Evaluation PNG bytes / estimated field bytes",
        True,
    ),
    (
        "evaluation_png_over_canonical_raw_ratio",
        "Evaluation PNG bytes / canonical raw field bytes",
        True,
    ),
    (
        "evaluation_png_over_lossless_reference_ratio",
        "Evaluation PNG bytes / lossless reference NPZ bytes",
        True,
    ),
    ("contraction_seconds", "Contraction time (seconds)", True),
    ("topology_seconds", "Topology-only time estimate (seconds)", True),
    ("recovery_seconds", "Recovery time (seconds)", True),
    ("recovery_attribution_seconds", "Error-attribution time (seconds)", True),
    ("cold_decode_seconds", "Cold field decode time (seconds)", True),
    ("first_render_seconds", "First maintained render time (seconds)", True),
    ("render_seconds", "Immediate-repeat maintained render time (seconds)", True),
    ("metric_seconds", "Metric evaluation time (seconds)", True),
    ("total_seconds", "Total artifact-to-metrics wall time (seconds)", True),
    ("contraction_actions", "Committed contraction actions", True),
    ("recovery_checkpoints", "Recovery checkpoints", True),
    ("recovery_accepted_checkpoints", "Accepted recovery checkpoints", True),
    ("recovery_optimizer_steps", "Attempted recovery optimizer steps", True),
    ("recovery_sse_gain", "Cumulative recovery SSE gain", True),
    ("recovery_optimized_rows_max", "Maximum rows optimized at a checkpoint", True),
    (
        "recovery_error_weight_effective_rows_mean",
        "Mean effective error-weighted rows",
        True,
    ),
    ("recovery_error_weight_p90_mean", "Mean checkpoint error-weight p90", True),
    ("recovery_error_weight_max_peak", "Peak checkpoint error weight", True),
    ("touched_active_rows", "Final active rows touched by contractions", True),
    ("untouched_active_rows", "Final active rows never touched by contractions", True),
    (
        "maintained_render_parity_max_abs",
        "Maintained/reference render max-absolute difference",
        True,
    ),
    (
        "repeated_render_parity_max_abs",
        "First/repeated maintained render max-absolute difference",
        True,
    ),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonable(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return value


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(_jsonable(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _snapshot_executed_sources(output_root: Path) -> list[dict[str, object]]:
    """Preserve task-local untracked sources so a dirty diagnostic remains inspectable."""
    source_paths = (
        Path(__file__).resolve(),
        ROOT / "src" / "structsplat" / "pixel_contraction.py",
    )
    records: list[dict[str, object]] = []
    for source in source_paths:
        relative = source.relative_to(ROOT)
        destination = output_root / "source_snapshot" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        records.append(
            {
                "repository_path": str(relative),
                "snapshot_path": str(destination.relative_to(output_root)),
                "bytes": destination.stat().st_size,
                "sha256": _sha256(destination),
            }
        )
    return records


def _discover_images(raw_paths: list[Path]) -> list[Path]:
    images: list[Path] = []
    for path in raw_paths:
        if path.is_dir():
            images.extend(
                candidate
                for candidate in sorted(path.iterdir())
                if candidate.is_file() and candidate.suffix.lower() in IMAGE_SUFFIXES
            )
        elif path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            images.append(path)
        else:
            raise SystemExit(f"image path is missing or unsupported: {path}")
    resolved: list[Path] = []
    seen: set[Path] = set()
    for image in images:
        absolute = image.resolve()
        if absolute not in seen:
            seen.add(absolute)
            resolved.append(absolute)
    if not resolved:
        raise SystemExit("no supported images found")
    return resolved


def _load_evaluation_raster(
    image_path: Path,
    mask_path: Path | None,
    *,
    max_side: int | None,
    mask_threshold: float,
) -> tuple[np.ndarray, np.ndarray | None, dict[str, object]]:
    """Load one native source and optional mask into a deterministic evaluation raster."""
    from PIL import Image

    with Image.open(image_path) as source:
        image = source.convert("RGB")
        original_width, original_height = image.size
        evaluation_width, evaluation_height = original_width, original_height
        if max_side is not None and max(image.size) > max_side:
            scale = float(max_side) / float(max(image.size))
            evaluation_width = max(1, round(original_width * scale))
            evaluation_height = max(1, round(original_height * scale))
            image = image.resize(
                (evaluation_width, evaluation_height), Image.Resampling.LANCZOS
            )
        rgb = np.asarray(image, dtype=np.float32) / 255.0

    mask: np.ndarray | None = None
    mask_record: dict[str, object] | None = None
    if mask_path is not None:
        with Image.open(mask_path) as source:
            if source.size != (original_width, original_height):
                raise ValueError(
                    f"mask shape {(source.height, source.width)} does not match original image "
                    f"shape {(original_height, original_width)}"
                )
            if source.mode in {"RGBA", "LA"} or "transparency" in source.info:
                alpha = source.convert("RGBA").getchannel("A")
            else:
                alpha = source.convert("L")
            if alpha.size != (evaluation_width, evaluation_height):
                alpha = alpha.resize(
                    (evaluation_width, evaluation_height), Image.Resampling.NEAREST
                )
            mask = np.asarray(alpha, dtype=np.float32) / 255.0 >= mask_threshold
        if not mask.any():
            raise ValueError("thresholded mask contains no active pixels")
        mask_record = {
            "path": str(mask_path),
            "bytes": mask_path.stat().st_size,
            "sha256": _sha256(mask_path),
            "threshold": mask_threshold,
        }

    resized = (evaluation_width, evaluation_height) != (original_width, original_height)
    record: dict[str, object] = {
        "original_width": original_width,
        "original_height": original_height,
        "evaluation_width": evaluation_width,
        "evaluation_height": evaluation_height,
        "scale_x": evaluation_width / original_width,
        "scale_y": evaluation_height / original_height,
        "max_side": max_side,
        "rgb_resampling": "pillow_lanczos" if resized else "none",
        "mask_resampling": "pillow_nearest" if resized and mask is not None else "none",
        "mask": mask_record,
    }
    return np.ascontiguousarray(rgb), mask, record


def _git_record() -> dict[str, object]:
    def run(*args: str) -> str | None:
        completed = subprocess.run(
            ["git", *args],
            cwd=ROOT,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return completed.stdout.strip() if completed.returncode == 0 else None

    status = run("status", "--porcelain")
    return {
        "revision": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "dirty": None if status is None else bool(status),
        "status_porcelain": status,
    }


def _installed_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _metric_values(
    reconstruction: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    *,
    device: str,
    compute_lpips: bool,
) -> dict[str, object]:
    import torch
    import torch.nn.functional as torch_functional

    from structsplat import metrics

    metric_device = torch.device(device)
    prediction_tensor = torch.from_numpy(np.ascontiguousarray(reconstruction)).to(metric_device)
    target_tensor = torch.from_numpy(np.ascontiguousarray(target)).to(metric_device)
    residual = reconstruction.astype(np.float64) - target.astype(np.float64)
    masked_mse = float(np.mean((residual[mask]) ** 2))
    psnr_db = float(10.0 * math.log10(1.0 / max(masked_mse, 1e-12)))
    minimum_side = min(target.shape[0], target.shape[1])
    ssim_window = min(11, minimum_side if minimum_side % 2 == 1 else minimum_side - 1)
    ssim_value: float | None = None
    ms_ssim_value: float | None = None
    if ssim_window >= 3:
        ssim_value = float(metrics.ssim(prediction_tensor, target_tensor, win=ssim_window))
        ms_ssim_value = float(
            metrics.ms_ssim(prediction_tensor, target_tensor, win=ssim_window)
        )
    lpips_value: float | None = None
    lpips_error: str | None = None
    if compute_lpips:
        try:
            lpips_value = metrics.LPIPS.distance(prediction_tensor, target_tensor)
        except Exception as exc:  # optional package, weights, and backend are environment-bound
            lpips_error = f"{type(exc).__name__}: {exc}"[:300]

    # These metrics describe the exact 8-bit, display-referred PNGs shown by the report.  Global
    # fidelity stays on the raw float reconstruction above; keeping the domains separate avoids
    # pretending a display artifact gate is an optimizer objective or a codec metric.
    display_prediction = np.rint(np.clip(reconstruction, 0.0, 1.0) * 255.0) / 255.0
    display_target = np.rint(np.clip(target, 0.0, 1.0) * 255.0) / 255.0
    display_residual = display_prediction - display_target
    pixel_rmse = np.sqrt(np.mean(display_residual * display_residual, axis=2))
    foreground_rmse = pixel_rmse[mask]
    artifact_metrics: dict[str, object] = {
        "artifact_metric_domain": "display_png_8bit_black_matted_rgb",
        "artifact_pixel_rmse_q99": float(np.quantile(foreground_rmse, 0.99)),
        "artifact_pixel_rmse_q999": float(np.quantile(foreground_rmse, 0.999)),
        "artifact_pixel_rmse_max": float(np.max(foreground_rmse)),
        "artifact_pixel_rmse_fraction_gt_005": float(np.mean(foreground_rmse > 0.05)),
        "artifact_pixel_rmse_fraction_gt_010": float(np.mean(foreground_rmse > 0.10)),
    }
    squared_pixel_error = torch.as_tensor(
        np.mean(display_residual * display_residual, axis=2).astype(np.float32),
        device=metric_device,
    )[None, None]
    for patch_side in (3, 7, 15, 31):
        effective_side = min(patch_side, target.shape[0], target.shape[1])
        if effective_side % 2 == 0:
            effective_side -= 1
        effective_side = max(effective_side, 1)
        pooled = torch_functional.avg_pool2d(
            squared_pixel_error,
            kernel_size=effective_side,
            stride=1,
            padding=0,
        )
        artifact_metrics[f"artifact_patch_rmse_max_{patch_side}"] = float(
            torch.sqrt(torch.max(pooled)).item()
        )
        artifact_metrics[f"artifact_patch_effective_side_{patch_side}"] = effective_side
    artifact_metrics["artifact_gate_pixel_max_threshold"] = 0.02
    artifact_metrics["artifact_gate_patch7_max_threshold"] = 0.01
    artifact_metrics["artifact_gate_pixel_max_pass"] = bool(
        artifact_metrics["artifact_pixel_rmse_max"] <= 0.02
    )
    artifact_metrics["artifact_gate_patch7_max_pass"] = bool(
        artifact_metrics["artifact_patch_rmse_max_7"] <= 0.01
    )
    artifact_metrics["artifact_gate_pass"] = bool(
        artifact_metrics["artifact_gate_pixel_max_pass"]
        and artifact_metrics["artifact_gate_patch7_max_pass"]
    )
    return {
        "masked_mse": masked_mse,
        "psnr_db": psnr_db,
        "ssim": ssim_value,
        "ms_ssim": ms_ssim_value,
        "ssim_window": ssim_window if ssim_window >= 3 else None,
        "lpips": lpips_value,
        "lpips_error": lpips_error,
        **artifact_metrics,
    }


def _run_one(
    image_path: Path,
    image: np.ndarray,
    target_count: int,
    mask: np.ndarray | None,
    raster_record: dict[str, object],
    args: argparse.Namespace,
    output_root: Path,
) -> dict[str, object]:
    from structsplat.cli import save_error_heatmap, save_image
    from structsplat.observation_field import ObservationField2D
    from structsplat.pixel_contraction import (
        PixelContractionConfig,
        contract_image,
        render_observation_field,
    )

    active_mask = np.ones(image.shape[:2], dtype=bool) if mask is None else mask
    active_pixels = int(active_mask.sum())
    if target_count > active_pixels:
        raise ValueError(
            f"target {target_count} exceeds {active_pixels} active pixels for {image_path.name}"
        )
    key = f"{image_path.stem}__n{target_count}"
    artifact_dir = output_root / "artifacts" / key
    artifact_dir.mkdir(parents=True, exist_ok=False)
    target = image * active_mask[:, :, None]
    evaluation_source_path = artifact_dir / "source.png"
    save_image(str(evaluation_source_path), target)
    evaluation_source_png_bytes = evaluation_source_path.stat().st_size
    config = PixelContractionConfig(
        target_gaussians=target_count,
        leaf_scale_px=args.leaf_scale,
        sigma_cutoff=args.sigma_cutoff,
        support_fade_alpha=args.support_fade_alpha,
        coefficient_domain=args.coefficient_domain,
        estimated_row_bytes=args.estimated_row_bytes,
        proposal_batch_size=args.proposal_batch_size,
        merge_batch_size=args.merge_batch_size,
        pair_shortlist=args.pair_shortlist,
        exact_option_shortlist=args.exact_option_shortlist,
        pair_policy=args.pair_policy,
        recovery_steps=args.recovery_steps,
        recovery_scope=args.recovery_scope,
        recovery_schedule=args.recovery_schedule,
        recovery_progress_checkpoints=args.recovery_progress_checkpoints,
        recovery_every_actions=args.recovery_every_actions,
        recovery_device=args.recovery_device or args.device,
        recovery_renderer=args.recovery_renderer or args.renderer,
        recovery_render_chunk=args.recovery_render_chunk,
        recovery_lr_means=args.recovery_lr_means,
        recovery_lr_scales=args.recovery_lr_scales,
        recovery_lr_rotations=args.recovery_lr_rotations,
        recovery_lr_coefficients=args.recovery_lr_coefficients,
        recovery_max_mean_shift_px=args.recovery_max_mean_shift,
        recovery_max_log_scale_shift=args.recovery_max_log_scale_shift,
        recovery_max_rotation_shift_rad=args.recovery_max_rotation_shift,
        recovery_error_smoothing_sigma_px=args.recovery_error_smoothing_sigma,
        recovery_error_weight_power=args.recovery_error_weight_power,
        recovery_error_weight_floor=args.recovery_error_weight_floor,
        recovery_error_weight_ceiling=args.recovery_error_weight_ceiling,
    )

    total_started = time.perf_counter()
    contraction = contract_image(image, config, mask=mask)
    field_path = artifact_dir / "field.observation.npz"
    contraction.field.save_lossless(field_path)
    lossless_bytes = field_path.stat().st_size
    decode_started = time.perf_counter()
    cold_field = ObservationField2D.load_lossless(field_path)
    cold_decode_seconds = time.perf_counter() - decode_started
    first_render_started = time.perf_counter()
    first_reconstruction = render_observation_field(
        cold_field,
        device=args.device,
        renderer=args.renderer,
        render_chunk=args.render_chunk,
    )
    first_render_seconds = time.perf_counter() - first_render_started
    render_started = time.perf_counter()
    reconstruction = render_observation_field(
        cold_field,
        device=args.device,
        renderer=args.renderer,
        render_chunk=args.render_chunk,
    )
    render_seconds = time.perf_counter() - render_started
    repeated_render_parity_max_abs = float(
        np.max(np.abs(reconstruction - first_reconstruction))
    )

    metric_started = time.perf_counter()
    metrics = _metric_values(
        reconstruction,
        target,
        active_mask,
        device=args.device,
        compute_lpips=args.lpips,
    )
    metric_seconds = time.perf_counter() - metric_started
    total_seconds = time.perf_counter() - total_started
    maintained_parity_max_abs = float(
        np.max(np.abs(reconstruction - contraction.reconstruction))
    )
    save_image(str(artifact_dir / "reconstruction.png"), reconstruction)
    save_error_heatmap(
        str(artifact_dir / "error.png"), reconstruction - target, scale=args.error_scale
    )
    _write_json(artifact_dir / "history.json", contraction.history_records())
    _write_json(artifact_dir / "recovery_history.json", contraction.recovery_records())

    source_bytes = image_path.stat().st_size
    pixel_count = image.shape[0] * image.shape[1]
    estimated_bytes = contraction.estimated_field_bytes
    canonical_raw_bytes = contraction.canonical_raw_bytes
    row: dict[str, object] = {
        "schema": REPORT_SCHEMA,
        "status": "diagnostic",
        "method": (
            "implicit_pixel_contraction"
            if args.recovery_steps == 0
            else (
                "implicit_pixel_contraction_all_error_weighted_recovery"
                if args.recovery_scope == "all_error_weighted"
                else "implicit_pixel_contraction_selective_recovery"
            )
        ),
        "image": image_path.name,
        "source_path": str(image_path),
        "source_sha256": _sha256(image_path),
        "source_file_bytes": source_bytes,
        "original_width": raster_record["original_width"],
        "original_height": raster_record["original_height"],
        "width": image.shape[1],
        "height": image.shape[0],
        "pixels": pixel_count,
        "active_pixels": active_pixels,
        "mask": mask is not None,
        "mask_source_path": (
            None if raster_record["mask"] is None else raster_record["mask"]["path"]
        ),
        "mask_source_sha256": (
            None if raster_record["mask"] is None else raster_record["mask"]["sha256"]
        ),
        "evaluation_scale_x": raster_record["scale_x"],
        "evaluation_scale_y": raster_record["scale_y"],
        "rgb_resampling": raster_record["rgb_resampling"],
        "mask_resampling": raster_record["mask_resampling"],
        "evaluation_source_png_bytes": evaluation_source_png_bytes,
        "target_gaussians": target_count,
        "n_gaussians": contraction.final_count,
        "stop_reason": contraction.stop_reason,
        "contraction_actions": len(contraction.history),
        "recovery_steps_per_checkpoint": args.recovery_steps,
        "recovery_scope": args.recovery_scope,
        "recovery_schedule": args.recovery_schedule,
        "recovery_progress_checkpoints": args.recovery_progress_checkpoints,
        "recovery_every_actions": args.recovery_every_actions,
        "recovery_device": args.recovery_device or args.device,
        "recovery_renderer": args.recovery_renderer or args.renderer,
        "recovery_error_smoothing_sigma_px": args.recovery_error_smoothing_sigma,
        "recovery_error_weight_power": args.recovery_error_weight_power,
        "recovery_error_weight_floor": args.recovery_error_weight_floor,
        "recovery_error_weight_ceiling": args.recovery_error_weight_ceiling,
        "recovery_determinism": (
            "cuda_atomic_numerically_nondeterministic"
            if args.recovery_steps > 0
            and (args.recovery_device or args.device).startswith("cuda")
            else "cpu_bit_deterministic_or_recovery_disabled"
        ),
        "recovery_checkpoints": len(contraction.recovery_history),
        "recovery_accepted_checkpoints": sum(
            event.accepted for event in contraction.recovery_history
        ),
        "recovery_optimizer_steps": sum(
            event.attempted_steps for event in contraction.recovery_history
        ),
        "recovery_sse_gain": sum(
            event.sse_before - event.sse_after for event in contraction.recovery_history
        ),
        "recovery_attribution_seconds": sum(
            event.attribution_seconds for event in contraction.recovery_history
        ),
        "recovery_optimized_rows_max": max(
            (event.optimized_count for event in contraction.recovery_history),
            default=0,
        ),
        "recovery_error_weight_effective_rows_mean": (
            float(
                np.mean(
                    [
                        event.error_weight_effective_rows
                        for event in contraction.recovery_history
                    ]
                )
            )
            if contraction.recovery_history
            else 0.0
        ),
        "recovery_error_weight_p90_mean": (
            float(
                np.mean(
                    [event.error_weight_p90 for event in contraction.recovery_history]
                )
            )
            if contraction.recovery_history
            else 0.0
        ),
        "recovery_error_weight_max_peak": max(
            (event.error_weight_max for event in contraction.recovery_history),
            default=0.0,
        ),
        "touched_active_rows": contraction.touched_active_rows,
        "untouched_active_rows": contraction.untouched_active_rows,
        "initial_sse": contraction.initial_sse,
        "final_sse": contraction.final_sse,
        "estimated_field_bytes": estimated_bytes,
        "canonical_raw_bytes": canonical_raw_bytes,
        "lossless_reference_bytes": lossless_bytes,
        "estimated_bits_per_pixel": 8.0 * estimated_bytes / pixel_count,
        "canonical_raw_bits_per_pixel": 8.0 * canonical_raw_bytes / pixel_count,
        "lossless_reference_bits_per_pixel": 8.0 * lossless_bytes / pixel_count,
        "estimated_bits_per_active_pixel": 8.0 * estimated_bytes / active_pixels,
        "canonical_raw_bits_per_active_pixel": 8.0 * canonical_raw_bytes / active_pixels,
        "lossless_reference_bits_per_active_pixel": 8.0 * lossless_bytes / active_pixels,
        "source_over_estimated_ratio": source_bytes / max(estimated_bytes, 1),
        "source_over_canonical_raw_ratio": source_bytes / max(canonical_raw_bytes, 1),
        "source_over_lossless_reference_ratio": source_bytes / max(lossless_bytes, 1),
        "evaluation_png_over_estimated_ratio": (
            evaluation_source_png_bytes / max(estimated_bytes, 1)
        ),
        "evaluation_png_over_canonical_raw_ratio": (
            evaluation_source_png_bytes / max(canonical_raw_bytes, 1)
        ),
        "evaluation_png_over_lossless_reference_ratio": (
            evaluation_source_png_bytes / max(lossless_bytes, 1)
        ),
        "contraction_seconds": contraction.elapsed_seconds,
        "recovery_seconds": sum(
            event.elapsed_seconds for event in contraction.recovery_history
        ),
        "topology_seconds": contraction.elapsed_seconds
        - sum(event.elapsed_seconds for event in contraction.recovery_history),
        "cold_decode_seconds": cold_decode_seconds,
        "first_render_seconds": first_render_seconds,
        "render_seconds": render_seconds,
        "repeated_render_parity_max_abs": repeated_render_parity_max_abs,
        "metric_seconds": metric_seconds,
        "total_seconds": total_seconds,
        "maintained_render_parity_max_abs": maintained_parity_max_abs,
        "field_canonical_sha256": cold_field.canonical_hash(),
        "field_file_sha256": _sha256(field_path),
        "artifact_dir": str(artifact_dir.relative_to(output_root)),
        **metrics,
    }
    _write_json(artifact_dir / "row.json", row)
    return row


def _format(value: object, digits: int = 4) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        if abs(value) >= 10000 or (value != 0.0 and abs(value) < 0.001):
            return f"{value:.3e}"
        return f"{value:.{digits}f}"
    return str(value)


def _finite_number(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def _axis_format(value: float) -> str:
    magnitude = abs(value)
    if magnitude >= 10000 or (magnitude != 0.0 and magnitude < 0.001):
        return f"{value:.2e}"
    if magnitude >= 100:
        return f"{value:.0f}"
    if magnitude >= 10:
        return f"{value:.2f}"
    return f"{value:.4f}".rstrip("0").rstrip(".")


def _metric_curve_svg(
    rows: list[dict[str, object]],
    metric: str,
    label: str,
    prefer_log_y: bool,
) -> str | None:
    groups: dict[str, list[tuple[float, float]]] = {}
    for row in rows:
        count = row.get("n_gaussians")
        value = row.get(metric)
        if not _finite_number(count) or not _finite_number(value) or float(count) <= 0.0:
            continue
        groups.setdefault(str(row["image"]), []).append((float(count), float(value)))
    if not groups:
        return None
    for points in groups.values():
        points.sort()

    all_points = [point for points in groups.values() for point in points]
    counts = sorted({point[0] for point in all_points})
    values = [point[1] for point in all_points]
    use_log_x = len(counts) > 1
    use_log_y = prefer_log_y and all(value > 0.0 for value in values)

    def tx(value: float) -> float:
        return math.log2(value) if use_log_x else value

    def ty(value: float) -> float:
        return math.log10(value) if use_log_y else value

    x_values = [tx(value) for value in counts]
    y_values = [ty(value) for value in values]
    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = min(y_values), max(y_values)
    if x_max <= x_min:
        x_pad = max(abs(x_min) * 0.05, 0.5)
        x_min, x_max = x_min - x_pad, x_max + x_pad
    if y_max <= y_min:
        y_pad = 0.2 if use_log_y else max(abs(y_min) * 0.05, 0.5)
        y_min, y_max = y_min - y_pad, y_max + y_pad
    else:
        y_pad = 0.06 * (y_max - y_min)
        y_min, y_max = y_min - y_pad, y_max + y_pad

    width, height = 640, 350
    left, right, top, bottom = 88, 24, 48, 58
    plot_width, plot_height = width - left - right, height - top - bottom

    def px(value: float) -> float:
        return left + (tx(value) - x_min) / (x_max - x_min) * plot_width

    def py(value: float) -> float:
        return top + (y_max - ty(value)) / (y_max - y_min) * plot_height

    title = escape(label)
    elements = [
        f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 {width} {height}' "
        f"role='img' aria-label='{title} versus Gaussian count'>",
        "<style>text{font-family:system-ui,sans-serif;fill:#25313b;font-size:11px}"
        ".grid{stroke:#dfe6eb;stroke-width:1}.axis{stroke:#667784;stroke-width:1.2}"
        ".curve{fill:none;stroke-width:2.2}.point{stroke:white;stroke-width:1.2}</style>",
        f"<text x='{width / 2}' y='22' text-anchor='middle' "
        f"style='font-size:15px;font-weight:600'>{title}</text>",
    ]
    for index in range(5):
        fraction = index / 4
        y_position = top + fraction * plot_height
        transformed = y_max - fraction * (y_max - y_min)
        tick_value = 10.0**transformed if use_log_y else transformed
        elements.extend(
            (
                f"<line x1='{left}' y1='{y_position:.2f}' x2='{left + plot_width}' "
                f"y2='{y_position:.2f}' class='grid'/>",
                f"<text x='{left - 8}' y='{y_position + 4:.2f}' text-anchor='end'>"
                f"{escape(_axis_format(tick_value))}</text>",
            )
        )
    x_ticks = counts
    if len(x_ticks) > 8:
        indices = sorted({round(index * (len(x_ticks) - 1) / 7) for index in range(8)})
        x_ticks = [x_ticks[index] for index in indices]
    for value in x_ticks:
        x_position = px(value)
        elements.extend(
            (
                f"<line x1='{x_position:.2f}' y1='{top}' x2='{x_position:.2f}' "
                f"y2='{top + plot_height}' class='grid'/>",
                f"<text x='{x_position:.2f}' y='{top + plot_height + 19}' "
                f"text-anchor='middle'>{int(value):,}</text>",
            )
        )
    elements.extend(
        (
            f"<line x1='{left}' y1='{top + plot_height}' x2='{left + plot_width}' "
            f"y2='{top + plot_height}' class='axis'/>",
            f"<line x1='{left}' y1='{top}' x2='{left}' y2='{top + plot_height}' "
            "class='axis'/>",
            f"<text x='{left + plot_width / 2}' y='{height - 10}' text-anchor='middle'>"
            f"achieved Gaussian count N{' (log₂ scale)' if use_log_x else ''}</text>",
            f"<text x='18' y='{top + plot_height / 2}' text-anchor='middle' "
            f"transform='rotate(-90 18 {top + plot_height / 2})'>"
            f"{title}{' (log₁₀ scale)' if use_log_y else ''}</text>",
        )
    )
    palette = ("#1769aa", "#d97706", "#16825d", "#9c3aa5", "#c33d4c")
    for group_index, (image_name, points) in enumerate(sorted(groups.items())):
        color = palette[group_index % len(palette)]
        coordinates = " ".join(f"{px(x):.2f},{py(y):.2f}" for x, y in points)
        if len(points) > 1:
            elements.append(
                f"<polyline points='{coordinates}' class='curve' stroke='{color}'/>"
            )
        for count, value in points:
            point_title = escape(
                f"{image_name}: N={int(count):,}, {metric}={_axis_format(value)}"
            )
            elements.append(
                f"<circle cx='{px(count):.2f}' cy='{py(value):.2f}' r='4.5' "
                f"fill='{color}' class='point'><title>{point_title}</title></circle>"
            )
    elements.append("</svg>")
    return "".join(elements)


def _write_metric_curves(
    output_root: Path, rows: list[dict[str, object]]
) -> tuple[str, list[dict[str, object]]]:
    curve_root = output_root / "curves"
    # ``main`` refuses non-empty outputs. ``exist_ok=True`` additionally permits a packaging-only
    # report regeneration from already persisted rows without rerunning the expensive field fit.
    curve_root.mkdir(parents=True, exist_ok=True)
    figures: list[str] = []
    catalog: list[dict[str, object]] = []
    for metric, label, prefer_log_y in CURVE_SPECS:
        svg = _metric_curve_svg(rows, metric, label, prefer_log_y)
        if svg is None:
            continue
        relative_path = Path("curves") / f"{metric}.svg"
        (output_root / relative_path).write_text(svg + "\n", encoding="utf-8")
        catalog.append(
            {
                "metric": metric,
                "label": label,
                "path": str(relative_path),
                "preferred_y_scale": "log10" if prefer_log_y else "linear",
            }
        )
        figures.append(
            "<figure class='curve-card'>"
            f"{svg}<figcaption><code>{escape(metric)}</code> · "
            f"<a href='{escape(str(relative_path))}'>standalone SVG</a></figcaption></figure>"
        )
    _write_json(
        curve_root / "catalog.json",
        {"schema": REPORT_SCHEMA, "x": "n_gaussians", "curves": catalog},
    )
    return "".join(figures), catalog


def _rd_svg(rows: list[dict[str, object]]) -> str:
    points = [
        row
        for row in rows
        if isinstance(row.get("estimated_bits_per_pixel"), (int, float))
        and isinstance(row.get("psnr_db"), (int, float))
    ]
    if not points:
        return "<p>No completed points.</p>"
    width, height = 720, 330
    left, right, top, bottom = 70, 25, 25, 55
    xs = np.asarray([float(row["estimated_bits_per_pixel"]) for row in points])
    ys = np.asarray([float(row["psnr_db"]) for row in points])
    x_min, x_max = float(xs.min()), float(xs.max())
    y_min, y_max = float(ys.min()), float(ys.max())
    if x_max <= x_min:
        x_min, x_max = max(0.0, x_min - 0.5), x_max + 0.5
    if y_max <= y_min:
        y_min, y_max = y_min - 0.5, y_max + 0.5
    plot_width, plot_height = width - left - right, height - top - bottom

    def px(value: float) -> float:
        return left + (value - x_min) / (x_max - x_min) * plot_width

    def py(value: float) -> float:
        return top + (y_max - value) / (y_max - y_min) * plot_height

    elements = [
        f"<svg viewBox='0 0 {width} {height}' role='img' "
        "aria-label='Estimated raw payload bits per pixel versus foreground-mask PSNR'>",
        f"<line x1='{left}' y1='{top + plot_height}' x2='{left + plot_width}' "
        f"y2='{top + plot_height}' class='axis'/>",
        f"<line x1='{left}' y1='{top}' x2='{left}' y2='{top + plot_height}' class='axis'/>",
        f"<text x='{left + plot_width / 2}' y='{height - 12}' text-anchor='middle'>"
        "estimated raw payload (bits/pixel; not codec rate)</text>",
        f"<text x='18' y='{top + plot_height / 2}' text-anchor='middle' "
        "transform='rotate(-90 18 150)'>foreground-mask PSNR (dB)</text>",
        f"<text x='{left}' y='{top + plot_height + 22}' text-anchor='middle'>{x_min:.2f}</text>",
        f"<text x='{left + plot_width}' y='{top + plot_height + 22}' "
        f"text-anchor='middle'>{x_max:.2f}</text>",
        f"<text x='{left - 8}' y='{top + plot_height}' text-anchor='end'>{y_min:.2f}</text>",
        f"<text x='{left - 8}' y='{top + 5}' text-anchor='end'>{y_max:.2f}</text>",
    ]
    for row in points:
        x = px(float(row["estimated_bits_per_pixel"]))
        y = py(float(row["psnr_db"]))
        label = f"{row['image']} N={row['n_gaussians']}"
        elements.append(
            f"<circle cx='{x:.2f}' cy='{y:.2f}' r='5'><title>{escape(label)}</title></circle>"
        )
    elements.append("</svg>")
    return "".join(elements)


def _write_report(
    output_root: Path, rows: list[dict[str, object]], command: str
) -> None:
    curve_html, curve_catalog = _write_metric_curves(output_root, rows)
    table_rows = []
    cards = []
    for row in rows:
        artifact = escape(str(row["artifact_dir"]))
        table_rows.append(
            "<tr>"
            f"<td>{escape(str(row['image']))}</td>"
            f"<td>{row['original_width']}×{row['original_height']} → "
            f"{row['width']}×{row['height']}</td>"
            f"<td>{row['n_gaussians']}</td>"
            f"<td>{_format(row['psnr_db'], 3)}</td>"
            f"<td>{_format(row['ms_ssim'], 5)}</td>"
            f"<td>{_format(row['lpips'], 5)}</td>"
            f"<td>{_format(row['artifact_pixel_rmse_max'], 4)}</td>"
            f"<td>{_format(row['artifact_patch_rmse_max_7'], 4)}</td>"
            f"<td>{'pass' if row['artifact_gate_pass'] else 'FAIL'}</td>"
            f"<td>{_format(row['estimated_bits_per_pixel'], 3)}</td>"
            f"<td>{_format(row['source_over_estimated_ratio'], 3)}</td>"
            f"<td>{_format(row['evaluation_png_over_estimated_ratio'], 3)}</td>"
            f"<td>{_format(row['contraction_seconds'], 3)}</td>"
            f"<td>{_format(row['recovery_seconds'], 3)}</td>"
            f"<td>{escape(str(row['stop_reason']))}</td>"
            "</tr>"
        )
        cards.append(
            "<article class='card'>"
            f"<h3>{escape(str(row['image']))} · N={row['n_gaussians']}</h3>"
            "<div class='images'>"
            f"<figure><img src='{artifact}/source.png'><figcaption>source</figcaption></figure>"
            f"<figure><img src='{artifact}/reconstruction.png'><figcaption>reconstruction</figcaption></figure>"
            f"<figure><img src='{artifact}/error.png'><figcaption>fixed-scale error</figcaption></figure>"
            "</div>"
            f"<p>foreground PSNR {_format(row['psnr_db'], 3)} dB · raw-estimate "
            f"{_format(row['estimated_bits_per_pixel'], 3)} bpp · "
            f"{_format(row['contraction_seconds'], 3)} s · {escape(str(row['recovery_scope']))} "
            "recovery "
            f"{row['recovery_accepted_checkpoints']}/{row['recovery_checkpoints']} checkpoints, "
            f"{row['recovery_optimized_rows_max']} rows max, "
            f"{_format(row['recovery_seconds'], 3)} s</p>"
            f"<p>artifact gate <strong>{'pass' if row['artifact_gate_pass'] else 'FAIL'}</strong> · "
            f"pixel max {_format(row['artifact_pixel_rmse_max'], 4)} / 0.0200 · "
            f"7×7 max {_format(row['artifact_patch_rmse_max_7'], 4)} / 0.0100</p>"
            f"<p><a href='{artifact}/field.observation.npz'>field</a> · "
            f"<a href='{artifact}/history.json'>contraction history</a> · "
            f"<a href='{artifact}/recovery_history.json'>recovery history</a> · "
            f"<a href='{artifact}/row.json'>row ledger</a></p></article>"
        )
    document = f"""<!doctype html>
<html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width'>
<title>HIER-005 pixel contraction diagnostic</title>
<style>
:root{{--ink:#17212b;--muted:#5c6b78;--line:#d7dfe5;--paper:#f7f9fb;--blue:#1769aa;}}
body{{font-family:system-ui,sans-serif;color:var(--ink);margin:0;background:var(--paper);}}
header,main{{max-width:1180px;margin:auto;padding:24px;}} header{{padding-bottom:8px;}}
.warning{{border-left:5px solid #b45309;background:#fff7ed;padding:12px 16px;}}
code{{overflow-wrap:anywhere}} table{{border-collapse:collapse;width:100%;background:white;}}
th,td{{border-bottom:1px solid var(--line);padding:8px;text-align:right;}}
th:first-child,td:first-child{{text-align:left}} svg{{width:100%;background:white;}}
.axis{{stroke:#6b7884;stroke-width:1}} circle{{fill:var(--blue)}}
.card{{background:white;border:1px solid var(--line);border-radius:8px;padding:14px;margin:18px 0;}}
.curve-grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px;}}
.curve-card{{background:white;border:1px solid var(--line);border-radius:8px;margin:0;padding:8px;}}
.images{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px;}}
figure{{margin:0}} img{{width:100%;height:auto;image-rendering:auto}} figcaption{{color:var(--muted)}}
@media(max-width:800px){{.images,.curve-grid{{grid-template-columns:1fr}} table{{font-size:12px}}}}
</style></head><body><header><h1>HIER-005 implicit pixel contraction</h1>
<p class='warning'><strong>Diagnostic only.</strong> Estimated/raw/reference-container bytes are
not a compressed codec rate. This page cannot select Field V2 semantics, change defaults, or
support a comparative claim. Original-source byte ratios retain the native file denominator even
when the evaluation raster is resized; use the separately labeled evaluation-PNG ratios for a
same-raster file-size comparison.</p><p><code>{escape(command)}</code></p>
<p>When recovery runs on CUDA, atomic-gradient order can change the optimized field hash
and produce small run-to-run metric differences; this is numerical repeatability, not bit
determinism.</p>
<p><a href='metrics.json'>metrics.json</a> · <a href='metrics.jsonl'>metrics.jsonl</a> ·
<a href='metrics.csv'>metrics.csv</a> ·
<a href='config.json'>config.json</a> · <a href='curves/catalog.json'>curve catalog</a> ·
<a href='manifest.json'>manifest.json</a></p></header>
<main><h2>Estimated payload–quality view</h2>{_rd_svg(rows)}
<h2>All outcome metrics versus Gaussian count</h2>
<p>{len(curve_catalog)} curves. Log axes are labeled in each panel; missing optional metrics are
omitted rather than imputed.</p><div class='curve-grid'>{curve_html}</div>
<h2>Rows</h2><div style='overflow:auto'><table><thead><tr><th>image</th>
<th>native → eval</th><th>N</th><th>foreground PSNR dB</th><th>matted MS-SSIM</th>
<th>matted LPIPS</th>
<th>pixel max</th><th>7×7 max</th><th>artifact gate</th>
<th>est. bpp</th><th>original/est.</th><th>eval PNG/est.</th>
<th>contract s</th><th>recovery s</th><th>stop</th></tr></thead>
<tbody>{''.join(table_rows)}</tbody></table></div>
<h2>Visuals</h2>{''.join(cards)}</main></body></html>"""
    (output_root / "index.html").write_text(document, encoding="utf-8")


def _write_tables(output_root: Path, rows: list[dict[str, object]]) -> None:
    payload = {
        "schema": REPORT_SCHEMA,
        "status": "diagnostic",
        "claim_ready": False,
        "row_count": len(rows),
        "rows": rows,
        "rate_warning": (
            "estimated_field_bytes and canonical_raw_bytes are uncoded payload references; "
            "lossless_reference_bytes is a deterministic interchange container, not COMP-013"
        ),
        "metric_domains": {
            "masked_mse": "thresholded foreground mask only",
            "psnr_db": "thresholded foreground mask only",
            "ssim": "full evaluation raster after black matting outside the mask",
            "ms_ssim": "full evaluation raster after black matting outside the mask",
            "lpips": "full evaluation raster after black matting outside the mask",
            "artifact_pixel_rmse_*": (
                "exact displayed 8-bit PNG values; pixel quantiles/max over foreground mask"
            ),
            "artifact_patch_rmse_max_*": (
                "maximum complete in-canvas black-matted RGB patch RMSE on exact displayed "
                "8-bit PNG values; requested odd patch side clips to the largest fitting odd "
                "side on tiny diagnostic rasters"
            ),
        },
    }
    _write_json(output_root / "metrics.json", payload)
    with (output_root / "metrics.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(_jsonable(row), sort_keys=True) + "\n")
    columns = sorted({key for row in rows for key in row})
    with (output_root / "metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_manifest(output_root: Path) -> None:
    entries = []
    for path in sorted(output_root.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            entries.append(
                {
                    "path": str(path.relative_to(output_root)),
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
            )
    _write_json(
        output_root / "manifest.json",
        {"schema": REPORT_SCHEMA, "status": "diagnostic", "files": entries},
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--target-gaussians", type=int, nargs="+", required=True)
    parser.add_argument("--mask", type=Path)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument(
        "--max-side",
        type=int,
        help=(
            "deterministically resize the evaluation RGB with Pillow LANCZOS and the mask "
            "with nearest-neighbor while retaining native source provenance"
        ),
    )
    parser.add_argument("--leaf-scale", type=float, default=0.18)
    parser.add_argument("--sigma-cutoff", type=float, default=3.0)
    parser.add_argument("--support-fade-alpha", type=float, default=0.0)
    parser.add_argument("--coefficient-domain", choices=("signed", "nonnegative"), default="signed")
    parser.add_argument("--estimated-row-bytes", type=int, default=32)
    parser.add_argument("--proposal-batch-size", type=int, default=64)
    parser.add_argument("--merge-batch-size", type=int, default=8)
    parser.add_argument("--pair-shortlist", type=int, default=3)
    parser.add_argument("--exact-option-shortlist", type=int, default=2)
    parser.add_argument("--pair-policy", choices=("exact_count", "always"), default="exact_count")
    parser.add_argument(
        "--recovery-steps",
        type=int,
        default=0,
        help="Adam steps at each opt-in recovery checkpoint (0 disables recovery)",
    )
    parser.add_argument(
        "--recovery-scope",
        choices=("touched", "all_error_weighted"),
        default="touched",
        help=(
            "touched preserves never-touched pixel rows; all_error_weighted optimizes every "
            "active row with smoothed-error Adam update multipliers"
        ),
    )
    parser.add_argument(
        "--recovery-schedule",
        choices=("progress", "actions"),
        default="progress",
        help=(
            "progress uses a fixed number of row-reduction checkpoints; actions retains the "
            "legacy accepted-action cadence"
        ),
    )
    parser.add_argument("--recovery-progress-checkpoints", type=int, default=16)
    parser.add_argument("--recovery-every-actions", type=int, default=128)
    parser.add_argument(
        "--recovery-device",
        help="torch device for recovery (defaults to --device)",
    )
    parser.add_argument(
        "--recovery-renderer",
        choices=("additive", "cuda_additive", "cuda_tiled_additive"),
        help="renderer for recovery (defaults to --renderer)",
    )
    parser.add_argument("--recovery-render-chunk", type=int, default=256)
    parser.add_argument("--recovery-lr-means", type=float, default=0.005)
    parser.add_argument("--recovery-lr-scales", type=float, default=0.003)
    parser.add_argument("--recovery-lr-rotations", type=float, default=0.001)
    parser.add_argument("--recovery-lr-coefficients", type=float, default=0.003)
    parser.add_argument("--recovery-max-mean-shift", type=float, default=1.5)
    parser.add_argument("--recovery-max-log-scale-shift", type=float, default=0.35)
    parser.add_argument("--recovery-max-rotation-shift", type=float, default=0.35)
    parser.add_argument("--recovery-error-smoothing-sigma", type=float, default=1.5)
    parser.add_argument("--recovery-error-weight-power", type=float, default=0.5)
    parser.add_argument("--recovery-error-weight-floor", type=float, default=0.05)
    parser.add_argument("--recovery-error-weight-ceiling", type=float, default=4.0)
    parser.add_argument(
        "--renderer",
        choices=("additive", "cuda_additive", "cuda_tiled_additive"),
        default="additive",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--render-chunk", type=int, default=4096)
    parser.add_argument("--error-scale", type=float, default=4.0)
    parser.add_argument("--lpips", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if any(target <= 0 for target in args.target_gaussians):
        raise SystemExit("all --target-gaussians values must be positive")
    if not 0.0 <= args.mask_threshold <= 1.0:
        raise SystemExit("--mask-threshold must lie in [0, 1]")
    if args.max_side is not None and args.max_side <= 0:
        raise SystemExit("--max-side must be positive")
    if args.error_scale <= 0.0:
        raise SystemExit("--error-scale must be positive")
    if args.recovery_steps < 0:
        raise SystemExit("--recovery-steps must be nonnegative")
    images = _discover_images(args.images)
    if args.mask is not None and len(images) != 1:
        raise SystemExit("--mask is supported only with one resolved input image")
    output_root = args.out.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise SystemExit(f"refusing to overwrite nonempty output directory: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    source_snapshot = _snapshot_executed_sources(output_root)

    resolved_mask = args.mask.resolve() if args.mask is not None else None

    import torch

    import structsplat

    cuda_devices = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            cuda_devices.append(
                {
                    "index": index,
                    "name": properties.name,
                    "capability": [properties.major, properties.minor],
                    "total_memory_bytes": properties.total_memory,
                }
            )

    command = shlex.join([sys.executable, str(Path(__file__).relative_to(ROOT)), *(argv or sys.argv[1:])])
    config_record = {
        "schema": REPORT_SCHEMA,
        "task": "HIER-005",
        "status": "diagnostic",
        "claim_ready": False,
        "command": command,
        "args": vars(args),
        "resolved_images": [str(path) for path in images],
        "resolved_mask": None if resolved_mask is None else str(resolved_mask),
        "executed_source_snapshot": source_snapshot,
        "evaluation_raster": {
            "rgb_resampling": "Pillow LANCZOS when --max-side reduces the native raster",
            "mask_resampling": "Pillow nearest-neighbor when --max-side reduces the native raster",
            "metrics_domain": "resized RGB multiplied by the thresholded resized mask",
            "metric_partition": (
                "MSE/PSNR use foreground-mask pixels; SSIM/MS-SSIM/LPIPS use the full "
                "black-matted evaluation raster"
            ),
            "original_source_bytes_retained": True,
        },
        "resolved_recovery": {
            "enabled": args.recovery_steps > 0,
            "device": args.recovery_device or args.device,
            "renderer": args.recovery_renderer or args.renderer,
            "scope": args.recovery_scope,
            "schedule": args.recovery_schedule,
            "progress_checkpoints": args.recovery_progress_checkpoints,
            "freeze_contract": (
                "all active rows are trainable with matrix-free smoothed-error update weights; "
                "inactive rows remain fixed"
                if args.recovery_scope == "all_error_weighted"
                else (
                    "only active rows ever produced by a contraction are trainable; "
                    "never-touched pixel rows and all inactive rows remain fixed in a "
                    "detached base"
                )
            ),
            "error_weighting": {
                "smoothing_sigma_px": args.recovery_error_smoothing_sigma,
                "power": args.recovery_error_weight_power,
                "floor": args.recovery_error_weight_floor,
                "ceiling": args.recovery_error_weight_ceiling,
                "attribution": (
                    "smoothed mask-aware residual MSE averaged under each Gaussian by one "
                    "additive-renderer color VJP; applied after Adam preconditioning"
                ),
            },
        },
        "method_determinism": (
            "topology has no random source; CPU recovery is bit-deterministic in the focused "
            "test, while CUDA recovery can vary through atomic-gradient accumulation order"
            if args.recovery_steps > 0
            else "no random source or seed; deterministic topology and field"
        ),
        "git": _git_record(),
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "pillow": _installed_version("Pillow"),
            "lpips": _installed_version("lpips"),
            "structsplat": getattr(structsplat, "__version__", "unknown"),
        },
        "cuda_devices": cuda_devices,
        "evidence_limits": [
            "No preregistered controls or independent protocol review.",
            "Estimated and raw bytes are not complete codec bytes.",
            "Lossless Observation Field NPZ is reference interchange, not compression.",
            (
                "Original-source ratios compare a native file with a resized field when "
                "--max-side is active; evaluation-PNG ratios are the same-raster reference."
            ),
            (
                "CUDA recovery is numerically nondeterministic because gradient "
                "atomics can change optimizer trajectories; repeat quality and parity must be "
                "reported with the field hash."
            ),
            "This diagnostic cannot select semantics, promote claims, or change defaults.",
        ],
    }
    _write_json(output_root / "config.json", config_record)

    rows: list[dict[str, object]] = []
    for image_path in images:
        try:
            image, mask, raster_record = _load_evaluation_raster(
                image_path,
                resolved_mask,
                max_side=args.max_side,
                mask_threshold=args.mask_threshold,
            )
        except (OSError, ValueError) as exc:
            raise SystemExit(f"cannot prepare {image_path}: {exc}") from exc
        for target_count in sorted(set(args.target_gaussians)):
            row = _run_one(
                image_path,
                image,
                target_count,
                mask,
                raster_record,
                args,
                output_root,
            )
            rows.append(row)
            print(
                f"{image_path.name} N={target_count}: {row['psnr_db']:.3f} dB, "
                f"{row['contraction_seconds']:.3f} s, {row['stop_reason']}",
                flush=True,
            )
    _write_tables(output_root, rows)
    _write_report(output_root, rows, command)
    _write_manifest(output_root)
    print(f"wrote diagnostic report: {output_root / 'index.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
