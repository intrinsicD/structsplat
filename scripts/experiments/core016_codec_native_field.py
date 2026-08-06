#!/usr/bin/env python3
"""Run the CORE-016 codec-native dual-plane field diagnostic.

This is an exposed-data killing test, not a formal benchmark.  It charges the complete cold
packet, separates appearance rate from structural capacity, and preserves every requested cell.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import csv
import hashlib
from html import escape
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import sys
import time
import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[2]
for candidate in (ROOT, ROOT / "src"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from scripts.experiments import hier005_pixel_contraction as report_utils  # noqa: E402
from structsplat.cli import save_error_heatmap, save_image, save_rgba  # noqa: E402
from structsplat.codec_native_field import (  # noqa: E402
    AppearanceCodec,
    CodecNativeField,
    CodecNativeFieldConfig,
    build_structural_field,
    encode_appearance,
)
from structsplat.observation_field import CanvasCropTransform, ObservationField2D  # noqa: E402
from structsplat.structure_tensor import gaussian_blur  # noqa: E402


REPORT_SCHEMA = "structsplat.core016_codec_native_field.diagnostic.v1"
SOURCE_FILES = (
    "scripts/experiments/core016_codec_native_field.py",
    "src/structsplat/codec_native_field.py",
    "src/structsplat/realtime_gs_adapter.py",
    "src/structsplat/observation_field.py",
    "tasks/CORE-016-codec-native-dual-plane-field.md",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonable(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return value


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_jsonable(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _git_record() -> dict[str, object]:
    def run(*arguments: str) -> str | None:
        result = subprocess.run(
            ["git", *arguments],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() if result.returncode == 0 else None

    status = run("status", "--porcelain")
    return {
        "revision": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "dirty": None if status is None else bool(status),
        "status_porcelain": status,
    }


def _version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _snapshot_sources(output_root: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for relative_text in SOURCE_FILES:
        relative = Path(relative_text)
        source = ROOT / relative
        destination = output_root / "source_snapshot" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        records.append(
            {
                "repository_path": relative.as_posix(),
                "snapshot_path": str(destination.relative_to(output_root)),
                "bytes": destination.stat().st_size,
                "sha256": _sha256(destination),
            }
        )
    return records


def _parse_cell(value: str) -> tuple[AppearanceCodec, int]:
    try:
        codec_text, quality_text = value.split(":", 1)
        quality = int(quality_text)
    except (ValueError, TypeError) as exc:
        raise argparse.ArgumentTypeError("appearance cells must use CODEC:QUALITY") from exc
    if codec_text not in {"jpeg", "webp", "webp_lossless"}:
        raise argparse.ArgumentTypeError(
            "appearance codec must be jpeg, webp, or webp_lossless"
        )
    if not 1 <= quality <= 100:
        raise argparse.ArgumentTypeError("appearance quality/effort must lie in [1,100]")
    return codec_text, quality  # type: ignore[return-value]


def _cell_name(codec: str, quality: int) -> str:
    marker = "e" if codec == "webp_lossless" else "q"
    return f"{codec}_{marker}{quality}"


def _load_crop(
    image_path: Path,
    mask_path: Path,
    *,
    max_side: int | None,
    mask_threshold: float,
    crop_margin: int,
) -> tuple[np.ndarray, np.ndarray, CanvasCropTransform, dict[str, object]]:
    start = time.perf_counter()
    with Image.open(image_path) as source:
        image = source.convert("RGB")
        original_width, original_height = image.size
        evaluation_width, evaluation_height = image.size
        if max_side is not None and max(image.size) > max_side:
            scale = max_side / max(image.size)
            evaluation_width = max(1, round(original_width * scale))
            evaluation_height = max(1, round(original_height * scale))
            image = image.resize(
                (evaluation_width, evaluation_height), Image.Resampling.LANCZOS
            )

        with Image.open(mask_path) as raw_mask:
            if raw_mask.size != (original_width, original_height):
                raise ValueError("mask dimensions do not match the source image")
            if raw_mask.mode in {"RGBA", "LA"} or "transparency" in raw_mask.info:
                alpha = raw_mask.convert("RGBA").getchannel("A")
            else:
                alpha = raw_mask.convert("L")
            if alpha.size != (evaluation_width, evaluation_height):
                alpha = alpha.resize(
                    (evaluation_width, evaluation_height), Image.Resampling.NEAREST
                )
            mask = np.asarray(alpha, dtype=np.float32) / 255.0 >= mask_threshold
        if not bool(mask.any()):
            raise ValueError("thresholded mask is empty")

        ys, xs = np.nonzero(mask)
        x0 = max(0, int(xs.min()) - crop_margin)
        y0 = max(0, int(ys.min()) - crop_margin)
        x1 = min(evaluation_width, int(xs.max()) + 1 + crop_margin)
        y1 = min(evaluation_height, int(ys.max()) + 1 + crop_margin)
        crop_rgb = np.asarray(image.crop((x0, y0, x1, y1)), dtype=np.float32) / 255.0
        crop_mask = mask[y0:y1, x0:x1]

    transform = CanvasCropTransform(
        evaluation_width,
        evaluation_height,
        x0,
        y0,
        x1 - x0,
        y1 - y0,
    )
    record = {
        "image_path": str(image_path),
        "image_sha256": _sha256(image_path),
        "image_bytes": image_path.stat().st_size,
        "mask_path": str(mask_path),
        "mask_sha256": _sha256(mask_path),
        "mask_bytes": mask_path.stat().st_size,
        "original_width": original_width,
        "original_height": original_height,
        "evaluation_width": evaluation_width,
        "evaluation_height": evaluation_height,
        "resized": (evaluation_width, evaluation_height) != (original_width, original_height),
        "max_side": max_side,
        "rgb_resampling": (
            "none"
            if (evaluation_width, evaluation_height) == (original_width, original_height)
            else "pillow_lanczos"
        ),
        "mask_resampling": (
            "none"
            if (evaluation_width, evaluation_height) == (original_width, original_height)
            else "pillow_nearest"
        ),
        "mask_threshold": mask_threshold,
        "crop_margin": crop_margin,
        "crop_x": x0,
        "crop_y": y0,
        "crop_width": x1 - x0,
        "crop_height": y1 - y0,
        "crop_pixels": (x1 - x0) * (y1 - y0),
        "active_pixels": int(crop_mask.sum()),
        "active_fraction": float(crop_mask.mean()),
        "preprocess_seconds": time.perf_counter() - start,
    }
    return (
        np.ascontiguousarray(crop_rgb),
        np.ascontiguousarray(crop_mask),
        transform,
        record,
    )


def _frequency_metrics(
    reconstruction: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float]:
    residual = np.asarray(reconstruction - target, dtype=np.float32)
    result: dict[str, float] = {}
    for sigma in (0.75, 1.5, 3.0):
        residual_highpass = residual - gaussian_blur(residual, sigma)
        target_highpass = target - gaussian_blur(target, sigma)
        error_mse = float(np.mean(np.square(residual_highpass[mask], dtype=np.float64)))
        target_mse = float(np.mean(np.square(target_highpass[mask], dtype=np.float64)))
        label = str(sigma).replace(".", "_")
        result[f"highpass_sigma_{label}_error_mse"] = error_mse
        result[f"highpass_sigma_{label}_target_mse"] = target_mse
        result[f"highpass_sigma_{label}_error_psnr_db"] = float(
            10.0 * math.log10(1.0 / max(error_mse, 1e-12))
        )
        result[f"highpass_sigma_{label}_retained_fraction"] = float(
            1.0 - error_mse / max(target_mse, 1e-12)
        )

    gradient_terms: list[np.ndarray] = []
    for channel in range(3):
        dy, dx = np.gradient(residual[..., channel])
        gradient_terms.extend((np.square(dx), np.square(dy)))
    gradient_mse = float(np.mean(np.stack(gradient_terms, axis=-1)[mask]))
    result["gradient_error_mse"] = gradient_mse
    result["gradient_error_rmse"] = math.sqrt(max(gradient_mse, 0.0))
    return result


def _continuous_query_metrics(
    packet: CodecNativeField,
    mask: np.ndarray,
    *,
    maximum_points: int,
    seed: int = 20260806,
) -> dict[str, float | int | str]:
    """Compare the Gaussian extension to a decoded-raster bilinear reference.

    Bilinear interpolation is not continuous-scene ground truth.  It is a familiar texture
    sampling control that exposes near-nearest transitions, excessive derivatives, and ringing.
    Only points whose complete 2x2 footprint is inside alpha are sampled.
    """
    valid = mask[:-1, :-1].copy()
    valid &= mask[:-1, 1:]
    valid &= mask[1:, :-1]
    valid &= mask[1:, 1:]
    y, x = np.nonzero(valid)
    if x.size == 0:
        return {
            "continuous_reference": "decoded_raster_bilinear_deep_alpha",
            "continuous_query_points": 0,
        }
    rng = np.random.default_rng(seed)
    if x.size > maximum_points:
        selected = rng.choice(x.size, size=maximum_points, replace=False)
        x = x[selected]
        y = y[selected]
    fraction = rng.uniform(0.05, 0.95, size=(x.size, 2))
    fx = fraction[:, 0]
    fy = fraction[:, 1]
    points = np.stack([x + fx, y + fy], axis=1).astype(np.float64)
    samples = packet.decoded_appearance
    p00 = samples[y, x].astype(np.float64)
    p10 = samples[y, x + 1].astype(np.float64)
    p01 = samples[y + 1, x].astype(np.float64)
    p11 = samples[y + 1, x + 1].astype(np.float64)
    bilinear = (
        (1.0 - fx)[:, None] * (1.0 - fy)[:, None] * p00
        + fx[:, None] * (1.0 - fy)[:, None] * p10
        + (1.0 - fx)[:, None] * fy[:, None] * p01
        + fx[:, None] * fy[:, None] * p11
    )
    queried = packet.query_appearance(points, apply_alpha=False).astype(np.float64)
    residual = queried - bilinear
    mse = float(np.mean(np.square(residual)))
    pixel_rmse = np.sqrt(np.mean(np.square(residual), axis=1))

    epsilon = 1.0 / 256.0
    probes = np.concatenate(
        [
            points + [epsilon, 0.0],
            points - [epsilon, 0.0],
            points + [0.0, epsilon],
            points - [0.0, epsilon],
        ],
        axis=0,
    )
    probe_values = packet.query_appearance(probes, apply_alpha=False).astype(np.float64)
    plus_x, minus_x, plus_y, minus_y = np.split(probe_values, 4)
    gradient_x = (plus_x - minus_x) / (2.0 * epsilon)
    gradient_y = (plus_y - minus_y) / (2.0 * epsilon)
    bilinear_x = (1.0 - fy)[:, None] * (p10 - p00) + fy[:, None] * (p11 - p01)
    bilinear_y = (1.0 - fx)[:, None] * (p01 - p00) + fx[:, None] * (p11 - p10)
    gradient_norm = np.sqrt(
        np.mean(np.square(gradient_x) + np.square(gradient_y), axis=1)
    )
    bilinear_gradient_norm = np.sqrt(
        np.mean(np.square(bilinear_x) + np.square(bilinear_y), axis=1)
    )
    local_min = np.minimum.reduce((p00, p10, p01, p11))
    local_max = np.maximum.reduce((p00, p10, p01, p11))
    overshoot = (queried < local_min - 1e-6) | (queried > local_max + 1e-6)
    return {
        "continuous_reference": "decoded_raster_bilinear_deep_alpha",
        "continuous_query_points": int(points.shape[0]),
        "continuous_bilinear_mse": mse,
        "continuous_bilinear_psnr_db": float(
            10.0 * math.log10(1.0 / max(mse, 1e-12))
        ),
        "continuous_bilinear_pixel_rmse_q99": float(np.quantile(pixel_rmse, 0.99)),
        "continuous_bilinear_max_abs": float(np.max(np.abs(residual))),
        "continuous_gradient_rms": float(np.sqrt(np.mean(np.square(gradient_norm)))),
        "continuous_gradient_q99": float(np.quantile(gradient_norm, 0.99)),
        "bilinear_gradient_rms": float(
            np.sqrt(np.mean(np.square(bilinear_gradient_norm)))
        ),
        "continuous_over_bilinear_gradient_rms_ratio": float(
            np.sqrt(np.mean(np.square(gradient_norm)))
            / max(float(np.sqrt(np.mean(np.square(bilinear_gradient_norm)))), 1e-12)
        ),
        "continuous_local_range_overshoot_fraction": float(np.mean(overshoot)),
        "continuous_value_outside_0_1_fraction": float(
            np.mean((queried < 0.0) | (queried > 1.0))
        ),
        "continuous_value_min": float(np.min(queried)),
        "continuous_value_max": float(np.max(queried)),
    }


def _sample_active_points(mask: np.ndarray, maximum: int) -> np.ndarray:
    y, x = np.nonzero(mask)
    if x.size > maximum:
        indices = np.linspace(0, x.size - 1, maximum, dtype=np.int64)
        x = x[indices]
        y = y[indices]
    return np.stack([x, y], axis=1).astype(np.float64)


def _structural_metrics(
    field: ObservationField2D,
    mask: np.ndarray,
    *,
    maximum_points: int,
) -> dict[str, float | int]:
    from scipy.spatial import cKDTree

    points = _sample_active_points(mask, maximum_points)
    distances = cKDTree(field.means_xy.astype(np.float64)).query(points, workers=1)[0]
    densities: list[np.ndarray] = []
    for start in range(0, points.shape[0], 256):
        densities.append(
            field.structural_density(points[start : start + 256], apply_alpha=True)
        )
    density = np.concatenate(densities) if densities else np.zeros(0, dtype=np.float64)
    scales = np.exp(field.log_scales_xy.astype(np.float64))
    aspect = np.maximum(scales[:, 0], scales[:, 1]) / np.minimum(scales[:, 0], scales[:, 1])
    spacing = math.sqrt(int(mask.sum()) / field.n)
    return {
        "structural_sample_points": int(points.shape[0]),
        "structural_nearest_distance_q50_px": float(np.quantile(distances, 0.50)),
        "structural_nearest_distance_q95_px": float(np.quantile(distances, 0.95)),
        "structural_nearest_distance_q99_px": float(np.quantile(distances, 0.99)),
        "structural_nearest_distance_max_px": float(np.max(distances)),
        "structural_nearest_q99_over_nominal_spacing": float(
            np.quantile(distances, 0.99) / max(spacing, 1e-12)
        ),
        "structural_support_coverage_fraction": float(np.mean(density > 0.0)),
        "structural_density_q01": float(np.quantile(density, 0.01)),
        "structural_density_q50": float(np.quantile(density, 0.50)),
        "structural_density_q99": float(np.quantile(density, 0.99)),
        "structural_density_coefficient_of_variation": float(
            np.std(density) / max(float(np.mean(density)), 1e-12)
        ),
        "structural_scale_q10_px": float(np.quantile(scales, 0.10)),
        "structural_scale_q50_px": float(np.quantile(scales, 0.50)),
        "structural_scale_q90_px": float(np.quantile(scales, 0.90)),
        "structural_aspect_q50": float(np.quantile(aspect, 0.50)),
        "structural_aspect_q95": float(np.quantile(aspect, 0.95)),
    }


def _worst_bounds(
    reconstruction: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    side: int,
) -> tuple[int, int, int, int]:
    error = np.sqrt(np.mean(np.square(reconstruction - target), axis=2))
    error = np.where(mask, error, -1.0)
    y, x = np.unravel_index(int(np.argmax(error)), error.shape)
    width = min(side, error.shape[1])
    height = min(side, error.shape[0])
    x0 = min(max(0, x - width // 2), error.shape[1] - width)
    y0 = min(max(0, y - height // 2), error.shape[0] - height)
    return x0, y0, x0 + width, y0 + height


def _save_crop(path: Path, image: np.ndarray, bounds: tuple[int, int, int, int]) -> None:
    x0, y0, x1, y1 = bounds
    save_image(str(path), image[y0:y1, x0:x1])


def _save_centers(path: Path, target: np.ndarray, field: ObservationField2D) -> None:
    pixels = np.rint(np.clip(target, 0.0, 1.0) * 255.0).astype(np.uint8)
    image = Image.fromarray(pixels, mode="RGB")
    draw = ImageDraw.Draw(image)
    radius = max(1, round(max(image.size) / 800))
    for x, y in field.means_xy:
        draw.ellipse(
            (float(x) - radius, float(y) - radius, float(x) + radius, float(y) + radius),
            outline=(45, 255, 110),
            width=1,
        )
    image.save(path)


def _benchmark_realtime_gs(
    packet: CodecNativeField,
    points: np.ndarray,
    *,
    realtime_gs_root: Path | None,
    device: str,
    payload_bytes: int,
    query_repeats: int,
) -> dict[str, object]:
    if realtime_gs_root is None:
        return {
            "realtime_gs_query_compatible": None,
            "realtime_gs_skip_reason": "--realtime-gs-root not supplied",
        }
    source_root = realtime_gs_root / "src"
    if not source_root.is_dir():
        return {
            "realtime_gs_query_compatible": False,
            "realtime_gs_skip_reason": f"missing realtime-gs source root: {source_root}",
        }
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    try:
        import torch

        from structsplat.realtime_gs_adapter import make_realtime_gs_view

        start = time.perf_counter()
        view = make_realtime_gs_view(
            packet,
            device=device,
            payload_bytes=payload_bytes,
        )
        if str(device).startswith("cuda"):
            torch.cuda.synchronize()
        build_seconds = time.perf_counter() - start
        offset = np.asarray(
            [packet.canvas_crop.crop_x + 0.5, packet.canvas_crop.crop_y + 0.5],
            dtype=np.float32,
        )
        full_points = points.astype(np.float32) + offset
        roundtrip_points = full_points.astype(np.float64) - offset.astype(np.float64)
        xy = torch.from_numpy(full_points).to(
            device=view.structural_field.device,
            dtype=view.structural_field.dtype,
        )
        if str(device).startswith("cuda"):
            torch.cuda.synchronize()
        start = time.perf_counter()
        cold_result = view.query_backend.query(xy)
        if str(device).startswith("cuda"):
            torch.cuda.synchronize()
        cold_query_seconds = time.perf_counter() - start
        query_timings = []
        result = None
        for _ in range(query_repeats):
            start = time.perf_counter()
            result = view.query_backend.query(xy)
            if str(device).startswith("cuda"):
                torch.cuda.synchronize()
            query_timings.append(time.perf_counter() - start)
        assert result is not None
        query_seconds = float(np.median(query_timings))
        expected = packet.query(roundtrip_points)
        ideal = packet.query(points)
        color = result.color.detach().cpu().numpy()
        weight = result.weight_sum.detach().cpu().numpy()
        color_error = float(np.max(np.abs(color - expected.color)))
        weight_error = float(np.max(np.abs(weight - expected.structural_density)))
        compatible = bool(color_error <= 2e-5 and weight_error <= 2e-4)
        record = {
            "realtime_gs_query_compatible": compatible,
            "realtime_gs_skip_reason": None,
            "realtime_gs_adapter_build_seconds": build_seconds,
            "realtime_gs_cold_first_query_seconds": cold_query_seconds,
            "realtime_gs_query_seconds": query_seconds,
            "realtime_gs_query_seconds_q10": float(np.quantile(query_timings, 0.10)),
            "realtime_gs_query_seconds_q90": float(np.quantile(query_timings, 0.90)),
            "realtime_gs_query_repeats": query_repeats,
            "realtime_gs_query_points": int(points.shape[0]),
            "realtime_gs_query_points_per_second": float(
                points.shape[0] / max(query_seconds, 1e-12)
            ),
            "realtime_gs_color_parity_max_abs": color_error,
            "realtime_gs_weight_parity_max_abs": weight_error,
            "realtime_gs_coordinate_roundtrip_max_abs_px": float(
                np.max(np.abs(roundtrip_points - points))
            ),
            "realtime_gs_roundtrip_vs_ideal_color_max_abs": float(
                np.max(np.abs(expected.color - ideal.color))
            ),
            "realtime_gs_roundtrip_vs_ideal_weight_max_abs": float(
                np.max(np.abs(expected.structural_density - ideal.structural_density))
            ),
            "realtime_gs_index_entries": int(view.query_backend.n_entries),
            "realtime_gs_index_max_candidates": int(view.query_backend.max_candidates),
            "realtime_gs_structural_rows": int(view.structural_field.n),
        }
        del cold_result, result, xy, view
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()
        return record
    except Exception as exc:  # dependency and local-checkout failures are diagnostic outcomes
        return {
            "realtime_gs_query_compatible": False,
            "realtime_gs_skip_reason": f"{type(exc).__name__}: {exc}"[:500],
        }


def _load_contextual_controls(
    *,
    rtgsv_path: Path | None,
    fit_json_path: Path | None,
    hier_metrics_path: Path | None,
) -> dict[str, object]:
    result: dict[str, object] = {
        "comparison_semantics": (
            "Context only. The RTGSV fit is calibrated/undistorted with a different fit window; "
            "HIER metrics use an explicit-row field and full evaluation raster. Do not relabel "
            "their rates or full-raster perceptual metrics as like-for-like packet results."
        )
    }
    if rtgsv_path is not None:
        result["rtgsv"] = {
            "path": str(rtgsv_path),
            "exists": rtgsv_path.is_file(),
            "bytes": rtgsv_path.stat().st_size if rtgsv_path.is_file() else None,
            "sha256": _sha256(rtgsv_path) if rtgsv_path.is_file() else None,
        }
    if fit_json_path is not None and fit_json_path.is_file():
        payload = json.loads(fit_json_path.read_text(encoding="utf-8"))
        metrics = payload.get("metrics", {})
        timing = payload.get("timing", {})
        output = payload.get("output", {})
        fit = payload.get("fit", {})
        result["iterative_fit"] = {
            "path": str(fit_json_path),
            "sha256": _sha256(fit_json_path),
            "complete_bytes": output.get("bytes"),
            "serialized_gaussians": output.get("serialized_gaussians"),
            "iterations_run": fit.get("iterations_run"),
            "selected_iter": fit.get("selected_iter"),
            "foreground_psnr_raw": metrics.get("foreground_psnr_raw"),
            "matted_crop_psnr_raw": metrics.get("matted_crop_psnr_raw"),
            "matted_crop_ms_ssim_raw": metrics.get("matted_crop_ms_ssim_raw"),
            "lpips_alex_clamped_matted_crop": metrics.get(
                "lpips_alex_clamped_matted_crop"
            ),
            "total_seconds": timing.get("total_seconds"),
            "fit_seconds": timing.get("fit_seconds"),
            "metric_domain_warning": (
                "calibrated undistorted native crop; contextual only"
            ),
        }
    if hier_metrics_path is not None and hier_metrics_path.is_file():
        payload = json.loads(hier_metrics_path.read_text(encoding="utf-8"))
        rows = []
        for row in payload.get("rows", []):
            rows.append(
                {
                    name: row.get(name)
                    for name in (
                        "arm",
                        "n_gaussians",
                        "estimated_field_bytes",
                        "lossless_reference_bytes",
                        "psnr_db",
                        "ssim",
                        "ms_ssim",
                        "lpips",
                        "artifact_pixel_rmse_max",
                        "artifact_patch_rmse_max_7",
                        "artifact_gate_pass",
                        "total_seconds",
                    )
                }
            )
        result["hier009"] = {
            "path": str(hier_metrics_path),
            "sha256": _sha256(hier_metrics_path),
            "rows": rows,
            "metric_domain_warning": (
                "same exposed image if max-side matches, but uncoded explicit-row rate and "
                "full-raster SSIM domain are not packet comparisons"
            ),
        }
    return result


def _numeric(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _plot_curves(
    output_root: Path,
    rows: list[dict[str, object]],
    *,
    curve_mode: str,
) -> list[dict[str, object]]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    curve_root = output_root / "curves"
    curve_root.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    excluded = {
        "source_original_bytes",
        "source_plus_mask_bytes",
        "canonical_crop_png_bytes",
        "canonical_matted_png_bytes",
        "canonical_rgba_png_bytes",
        "crop_width",
        "crop_height",
        "crop_pixels",
        "active_pixels",
        "source_bytes",
        "source_sha256",
    }
    summary_metrics = {
        "psnr_db",
        "ms_ssim",
        "lpips",
        "artifact_pixel_rmse_max",
        "artifact_patch_rmse_max_7",
        "highpass_sigma_1_5_error_mse",
        "gradient_error_rmse",
        "packet_complete_bytes",
        "source_over_packet_ratio",
        "canonical_crop_png_over_packet_ratio",
        "end_to_end_encode_estimated_seconds",
        "cold_decode_seconds",
        "appearance_query_points_per_second",
        "realtime_gs_query_points_per_second",
        "structural_support_coverage_fraction",
        "structural_nearest_distance_q99_px",
        "continuous_bilinear_psnr_db",
        "continuous_bilinear_pixel_rmse_q99",
        "continuous_over_bilinear_gradient_rms_ratio",
        "continuous_local_range_overshoot_fraction",
    }

    series_specs = (
        (
            "appearance_rate",
            [row for row in rows if bool(row["appearance_rate_cell"])],
            "packet_complete_bytes",
            "appearance_codec",
        ),
        (
            "structure_capacity",
            [row for row in rows if bool(row["structure_capacity_cell"])],
            "structural_count",
            "appearance_cell",
        ),
    )
    palette = ("#1769aa", "#d97706", "#16825d", "#9c3aa5", "#c33d4c")
    for series, data, x_name, group_name in series_specs:
        metrics = sorted(
            {
                name
                for row in data
                for name, value in row.items()
                if name != x_name and name not in excluded and _numeric(value)
            }
        )
        if curve_mode == "summary":
            metrics = [name for name in metrics if name in summary_metrics]
        for metric in metrics:
            values = [float(row[metric]) for row in data if _numeric(row.get(metric))]
            if len(values) < 2 or max(values) == min(values):
                continue
            figure, axis = plt.subplots(figsize=(6.5, 3.8), constrained_layout=True)
            plotted = False
            groups = sorted({str(row[group_name]) for row in data})
            for index, group in enumerate(groups):
                points = sorted(
                    (
                        (float(row[x_name]), float(row[metric]))
                        for row in data
                        if str(row[group_name]) == group
                        and _numeric(row.get(x_name))
                        and _numeric(row.get(metric))
                    ),
                    key=lambda item: item[0],
                )
                if not points:
                    continue
                x, y = zip(*points)
                axis.plot(
                    x,
                    y,
                    marker="o",
                    linewidth=1.6,
                    color=palette[index % len(palette)],
                    label=group,
                )
                plotted = True
            if not plotted:
                plt.close(figure)
                continue
            if x_name == "packet_complete_bytes":
                axis.set_xscale("log")
            if x_name == "structural_count":
                axis.set_xscale("log", base=2)
            axis.set_xlabel(x_name.replace("_", " "))
            axis.set_ylabel(metric.replace("_", " "))
            axis.set_title(f"{series}: {metric.replace('_', ' ')}")
            axis.grid(True, alpha=0.28)
            axis.legend(fontsize=7)
            path = curve_root / f"{series}__{metric}.svg"
            figure.savefig(path, format="svg")
            plt.close(figure)
            records.append(
                {"series": series, "metric": metric, "path": str(path.relative_to(output_root))}
            )
    _write_json(
        curve_root / "catalog.json",
        {"schema": REPORT_SCHEMA, "curve_count": len(records), "curves": records},
    )
    return records


def _write_tables(output_root: Path, rows: list[dict[str, object]]) -> None:
    _write_json(
        output_root / "metrics.json",
        {
            "schema": REPORT_SCHEMA,
            "status": "diagnostic",
            "claim_ready": False,
            "row_count": len(rows),
            "rows": rows,
        },
    )
    with (output_root / "metrics.jsonl").open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(_jsonable(row), sort_keys=True) + "\n")
    columns = sorted({name for row in rows for name in row})
    with (output_root / "metrics.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _write_report(
    output_root: Path,
    rows: list[dict[str, object]],
    curves: list[dict[str, object]],
    controls: dict[str, object],
) -> None:
    passing = [row for row in rows if bool(row.get("killing_gate_pass"))]
    verdict = (
        f"SURVIVES this exposed killing test ({len(passing)}/{len(rows)} cells pass)."
        if passing
        else "KILLED by this exposed test (no cell passes all frozen gates)."
    )
    table_rows: list[str] = []
    cards: list[str] = []
    for row in sorted(
        rows,
        key=lambda value: (int(value["structural_count"]), int(value["packet_complete_bytes"])),
    ):
        artifact = str(row["artifact_dir"])
        gate = "PASS" if row["artifact_gate_pass"] else "FAIL"
        killing = "PASS" if row["killing_gate_pass"] else "FAIL"
        adapter = row.get("realtime_gs_query_compatible")
        adapter_text = "n/a" if adapter is None else ("PASS" if adapter else "FAIL")
        table_rows.append(
            "<tr>"
            f"<td>{escape(str(row['appearance_cell']))}</td>"
            f"<td>{int(row['structural_count']):,}</td>"
            f"<td>{int(row['packet_complete_bytes']):,}</td>"
            f"<td>{float(row['source_over_packet_ratio']):.2f}×</td>"
            f"<td>{float(row['canonical_crop_png_over_packet_ratio']):.2f}×</td>"
            f"<td>{float(row['psnr_db']):.3f}</td>"
            f"<td>{float(row['ms_ssim']):.6f}</td>"
            f"<td>{float(row['artifact_pixel_rmse_max']):.5f}</td>"
            f"<td>{float(row['artifact_patch_rmse_max_7']):.5f}</td>"
            f"<td class='{gate.lower()}'>{gate}</td>"
            f"<td>{float(row['end_to_end_encode_estimated_seconds']):.3f}</td>"
            f"<td>{adapter_text}</td>"
            f"<td class='{killing.lower()}'>{killing}</td>"
            "</tr>"
        )
        cards.append(
            f"<section class='card'><h3>{escape(str(row['cell']))}</h3>"
            f"<p>{int(row['packet_complete_bytes']):,} B · {float(row['psnr_db']):.3f} dB · "
            f"artifact {gate} · killing gate {killing}</p><div class='images'>"
            f"<figure><img src='reference/source_matted.png'><figcaption>target</figcaption></figure>"
            f"<figure><img src='{artifact}/reconstruction.png'><figcaption>reconstruction</figcaption></figure>"
            f"<figure><img src='{artifact}/error.png'><figcaption>error ×{row['error_scale']}</figcaption></figure>"
            f"<figure><img src='{artifact}/centers.png'><figcaption>structural centers</figcaption></figure>"
            f"<figure><img src='{artifact}/source_crop.png'><figcaption>worst-area target</figcaption></figure>"
            f"<figure><img src='{artifact}/reconstruction_crop.png'><figcaption>worst-area reconstruction</figcaption></figure>"
            f"<figure><img src='{artifact}/error_crop.png'><figcaption>worst-area error</figcaption></figure>"
            "</div></section>"
        )
    curve_links = "".join(
        f"<li><a href='{record['path']}'>{escape(str(record['series']))}: "
        f"{escape(str(record['metric']))}</a></li>"
        for record in curves
    )
    control_text = escape(json.dumps(controls, indent=2, sort_keys=True))
    document = f"""<!doctype html><html><head><meta charset='utf-8'>
<title>CORE-016 codec-native dual-plane diagnostic</title><style>
body{{font-family:system-ui,sans-serif;margin:0;background:#f3f6f8;color:#17242d}}main{{max-width:1600px;margin:auto;padding:24px}}
.warning{{background:#fff3cd;border:1px solid #d5ae45;padding:12px;border-radius:8px}}.verdict{{font-size:20px;font-weight:700}}
table{{border-collapse:collapse;width:100%;background:white;font-size:12px}}th,td{{border:1px solid #dce3e8;padding:7px;text-align:right}}th:first-child,td:first-child{{text-align:left}}
.pass{{color:#087a48;font-weight:700}}.fail{{color:#b62929;font-weight:700}}.card{{background:white;border:1px solid #dce3e8;border-radius:9px;padding:14px;margin:18px 0}}
.images{{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:10px}}figure{{margin:0}}img{{width:100%;height:auto;background:#101417}}figcaption{{font-size:12px;color:#52616c}}.links{{columns:3;font-size:12px}}pre{{white-space:pre-wrap;background:#162029;color:#dce7ed;padding:12px;border-radius:8px}}
</style></head><body><main><h1>CORE-016: codec-native dual-plane Gaussian field</h1>
<p class='warning'><strong>Diagnostic only.</strong> Exposed C0001 data, a dirty source snapshot, and contextual controls with different preprocessing. Every packet byte—including appearance, exact alpha, sparse structure, metadata, and container framing—is charged. Original-file ratio also benefits from storing only the mask crop; use the crop-PNG ratio to isolate that effect.</p>
<p class='verdict'>{escape(verdict)}</p><p><a href='manifest.json'>manifest</a> · <a href='metrics.json'>metrics</a> · <a href='metrics.csv'>CSV</a> · <a href='config.json'>config</a> · <a href='controls.json'>controls</a> · <a href='curves/catalog.json'>all curves</a></p>
<h2>Outcomes</h2><table><thead><tr><th>appearance</th><th>structural N</th><th>packet B</th><th>original/packet</th><th>crop PNG/packet</th><th>foreground PSNR</th><th>matted MS-SSIM</th><th>pixel max</th><th>patch7 max</th><th>artifact</th><th>encode s</th><th>RTGS query</th><th>kill gate</th></tr></thead><tbody>{''.join(table_rows)}</tbody></table>
<h2>Visuals</h2>{''.join(cards)}<h2>Metric curves</h2><ul class='links'>{curve_links}</ul>
<h2>Contextual controls (not metric-equivalent)</h2><pre>{control_text}</pre>
</main></body></html>"""
    (output_root / "index.html").write_text(document, encoding="utf-8")


def _write_manifest(output_root: Path) -> None:
    files = []
    for path in sorted(output_root.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            files.append(
                {
                    "path": str(path.relative_to(output_root)),
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
            )
    _write_json(
        output_root / "manifest.json",
        {"schema": REPORT_SCHEMA, "status": "diagnostic", "files": files},
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--mask", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-side", type=int)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--crop-margin", type=int, default=0)
    parser.add_argument(
        "--appearance-cells",
        nargs="+",
        type=_parse_cell,
        default=None,
        help="CODEC:QUALITY cells; lossless WebP QUALITY is compression effort",
    )
    parser.add_argument("--structural-counts", nargs="+", type=int, default=[512, 1024, 2048, 4096])
    parser.add_argument("--primary-structural-count", type=int, default=2048)
    parser.add_argument("--anchor-cell", type=_parse_cell, default=("webp_lossless", 100))
    parser.add_argument("--structural-seed", type=int, default=0)
    parser.add_argument("--lattice-sigma", type=float, default=0.25)
    parser.add_argument("--lattice-radius", type=int, default=2)
    parser.add_argument("--lattice-prefilter-steps", type=int, default=0)
    parser.add_argument("--appearance-query-points", type=int, default=20_000)
    parser.add_argument("--dual-query-points", type=int, default=512)
    parser.add_argument("--continuous-query-points", type=int, default=4096)
    parser.add_argument("--structural-metric-points", type=int, default=4096)
    parser.add_argument("--query-repeats", type=int, default=7)
    parser.add_argument("--metric-device", default="cuda")
    parser.add_argument("--lpips", action="store_true")
    parser.add_argument("--realtime-gs-root", type=Path)
    parser.add_argument("--realtime-gs-device", default="cuda")
    parser.add_argument("--rtgsv", type=Path)
    parser.add_argument("--fit-json", type=Path)
    parser.add_argument("--hier-metrics", type=Path)
    parser.add_argument("--error-scale", type=float, default=4.0)
    parser.add_argument("--worst-crop-side", type=int, default=192)
    parser.add_argument("--curve-mode", choices=("all", "summary"), default="all")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    args.image = args.image.resolve()
    args.mask = args.mask.resolve()
    args.out = args.out.resolve()
    if not args.image.is_file() or not args.mask.is_file():
        raise SystemExit("image and mask must be existing files")
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite existing output: {args.out}")
    if args.crop_margin < 0:
        raise SystemExit("crop margin must be non-negative")
    if not 0.0 < args.mask_threshold <= 1.0:
        raise SystemExit("mask threshold must lie in (0,1]")
    if args.error_scale <= 0.0:
        raise SystemExit("error scale must be positive")
    if args.query_repeats < 1:
        raise SystemExit("query repeats must be positive")
    counts = sorted(set(args.structural_counts))
    if any(count < 1 for count in counts):
        raise SystemExit("structural counts must be positive")
    if args.primary_structural_count not in counts:
        raise SystemExit("primary structural count must occur in --structural-counts")
    appearance_cells = args.appearance_cells or [
        ("jpeg", 60),
        ("jpeg", 80),
        ("jpeg", 92),
        ("jpeg", 98),
        ("webp", 60),
        ("webp", 80),
        ("webp", 92),
        ("webp_lossless", 75),
        ("webp_lossless", 100),
    ]
    appearance_cells = list(dict.fromkeys(appearance_cells))
    if args.anchor_cell not in appearance_cells:
        appearance_cells.append(args.anchor_cell)

    args.out.mkdir(parents=True)
    (args.out / "reference").mkdir()
    (args.out / "artifacts").mkdir()
    source_snapshot = _snapshot_sources(args.out)
    source_payload = args.image.read_bytes()
    image, mask, transform, source_record = _load_crop(
        args.image,
        args.mask,
        max_side=args.max_side,
        mask_threshold=args.mask_threshold,
        crop_margin=args.crop_margin,
    )
    target = np.ascontiguousarray(image * mask[..., None])
    save_image(str(args.out / "reference" / "source_crop.png"), image)
    save_image(str(args.out / "reference" / "source_matted.png"), target)
    save_rgba(str(args.out / "reference" / "source_rgba.png"), image, mask.astype(np.float32))
    Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(
        args.out / "reference" / "alpha.png"
    )
    reference_bytes = {
        "canonical_crop_png_bytes": (args.out / "reference" / "source_crop.png").stat().st_size,
        "canonical_matted_png_bytes": (
            args.out / "reference" / "source_matted.png"
        ).stat().st_size,
        "canonical_rgba_png_bytes": (args.out / "reference" / "source_rgba.png").stat().st_size,
    }

    controls = _load_contextual_controls(
        rtgsv_path=args.rtgsv.resolve() if args.rtgsv is not None else None,
        fit_json_path=args.fit_json.resolve() if args.fit_json is not None else None,
        hier_metrics_path=args.hier_metrics.resolve() if args.hier_metrics is not None else None,
    )
    _write_json(args.out / "controls.json", controls)
    iterative_control = controls.get("iterative_fit")
    iterative_seconds = (
        iterative_control.get("total_seconds")
        if isinstance(iterative_control, dict)
        else None
    )
    rtgsv_control = controls.get("rtgsv")
    rtgsv_bytes = rtgsv_control.get("bytes") if isinstance(rtgsv_control, dict) else None

    structures: dict[int, ObservationField2D] = {}
    structure_times: dict[int, float] = {}
    structure_metrics: dict[int, dict[str, float | int]] = {}
    for count in counts:
        config = CodecNativeFieldConfig(
            appearance_codec=args.anchor_cell[0],
            appearance_quality=args.anchor_cell[1],
            lattice_sigma_px=args.lattice_sigma,
            lattice_radius_px=args.lattice_radius,
            lattice_prefilter_steps=args.lattice_prefilter_steps,
            structural_count=count,
            structural_seed=args.structural_seed,
        )
        start = time.perf_counter()
        structure = build_structural_field(
            image,
            config=config,
            canvas_crop=transform,
            mask=mask,
        )
        structure_times[count] = time.perf_counter() - start
        structures[count] = structure
        structure_metrics[count] = _structural_metrics(
            structure,
            mask,
            maximum_points=args.structural_metric_points,
        )

    appearance_payloads: dict[tuple[str, int], bytes] = {}
    appearance_times: dict[tuple[str, int], float] = {}
    for codec, quality in appearance_cells:
        config = CodecNativeFieldConfig(
            appearance_codec=codec,
            appearance_quality=quality,
            lattice_sigma_px=args.lattice_sigma,
            lattice_radius_px=args.lattice_radius,
            lattice_prefilter_steps=args.lattice_prefilter_steps,
            structural_count=args.primary_structural_count,
            structural_seed=args.structural_seed,
        )
        start = time.perf_counter()
        appearance_payloads[(codec, quality)] = encode_appearance(image, config)
        appearance_times[(codec, quality)] = time.perf_counter() - start

    cells = {
        (args.primary_structural_count, codec, quality)
        for codec, quality in appearance_cells
    }
    cells.update((count, args.anchor_cell[0], args.anchor_cell[1]) for count in counts)
    rng = np.random.default_rng(20260806)
    appearance_points = np.stack(
        [
            rng.uniform(0.0, max(image.shape[1] - 1.0, 0.0), args.appearance_query_points),
            rng.uniform(0.0, max(image.shape[0] - 1.0, 0.0), args.appearance_query_points),
        ],
        axis=1,
    ).astype(np.float64)
    active_points = _sample_active_points(mask, args.dual_query_points)
    active_points = active_points + rng.uniform(-0.45, 0.45, active_points.shape)

    rows: list[dict[str, object]] = []
    quality_cache: dict[tuple[str, int], tuple[np.ndarray, dict[str, object], float]] = {}
    source_digest = hashlib.sha256(source_payload).hexdigest()
    for count, codec, quality in sorted(cells, key=lambda cell: (cell[0], cell[1], cell[2])):
        config = CodecNativeFieldConfig(
            appearance_codec=codec,  # type: ignore[arg-type]
            appearance_quality=quality,
            lattice_sigma_px=args.lattice_sigma,
            lattice_radius_px=args.lattice_radius,
            lattice_prefilter_steps=args.lattice_prefilter_steps,
            structural_count=count,
            structural_seed=args.structural_seed,
        )
        packet = CodecNativeField(
            appearance_payload=appearance_payloads[(codec, quality)],
            structure=structures[count],
            config=config,
            source_sha256=source_digest,
            source_bytes=len(source_payload),
        )
        appearance_name = _cell_name(codec, quality)
        cell_name = f"{appearance_name}__n{count}"
        artifact_dir = args.out / "artifacts" / cell_name
        artifact_dir.mkdir()
        packet_path = artifact_dir / f"field{'.sgdp'}"
        start = time.perf_counter()
        ledger = packet.save(packet_path)
        serialize_seconds = time.perf_counter() - start
        start = time.perf_counter()
        cold = CodecNativeField.load(packet_path)
        cold_decode_seconds = time.perf_counter() - start
        start = time.perf_counter()
        reconstruction = cold.render()
        render_seconds = time.perf_counter() - start

        quality_key = (codec, quality)
        if quality_key not in quality_cache:
            start = time.perf_counter()
            metrics = report_utils._metric_values(
                reconstruction,
                target,
                mask,
                device=args.metric_device,
                compute_lpips=args.lpips,
            )
            metrics.update(_frequency_metrics(reconstruction, target, mask))
            metrics.update(
                _continuous_query_metrics(
                    cold,
                    mask,
                    maximum_points=args.continuous_query_points,
                )
            )
            raw_pixel_rmse = np.sqrt(
                np.mean(np.square(reconstruction.astype(np.float64) - target), axis=2)
            )[mask]
            decoded_matted = cold.decoded_appearance * mask[..., None]
            decoded_residual = decoded_matted.astype(np.float64) - target
            replay_residual = reconstruction.astype(np.float64) - decoded_matted
            metrics.update(
                {
                    "raw_foreground_pixel_rmse_q99": float(
                        np.quantile(raw_pixel_rmse, 0.99)
                    ),
                    "raw_foreground_pixel_rmse_q999": float(
                        np.quantile(raw_pixel_rmse, 0.999)
                    ),
                    "raw_foreground_pixel_rmse_max": float(np.max(raw_pixel_rmse)),
                    "decoded_source_foreground_mse": float(
                        np.mean(np.square(decoded_residual[mask]))
                    ),
                    "decoded_source_foreground_max_abs": float(
                        np.max(np.abs(decoded_residual[mask]))
                    ),
                    "pixel_center_replay_foreground_mse": float(
                        np.mean(np.square(replay_residual[mask]))
                    ),
                    "pixel_center_replay_foreground_max_abs": float(
                        np.max(np.abs(replay_residual[mask]))
                    ),
                }
            )
            metric_seconds = time.perf_counter() - start
            quality_cache[quality_key] = (reconstruction.copy(), metrics, metric_seconds)
        cached_reconstruction, metrics, metric_seconds = quality_cache[quality_key]
        if not np.array_equal(reconstruction, cached_reconstruction):
            raise RuntimeError("appearance replay changed across structural-count cells")

        appearance_timings = []
        for _ in range(args.query_repeats):
            start = time.perf_counter()
            _ = cold.query_appearance(appearance_points, apply_alpha=True)
            appearance_timings.append(time.perf_counter() - start)
        appearance_query_seconds = float(np.median(appearance_timings))
        dual_timings = []
        dual = None
        for _ in range(args.query_repeats):
            start = time.perf_counter()
            dual = cold.query(active_points)
            dual_timings.append(time.perf_counter() - start)
        assert dual is not None
        dual_query_seconds = float(np.median(dual_timings))
        if not np.isfinite(dual.color).all() or not np.isfinite(dual.structural_density).all():
            raise RuntimeError("non-finite dual query result")
        realtime = _benchmark_realtime_gs(
            cold,
            active_points,
            realtime_gs_root=(
                args.realtime_gs_root.resolve() if args.realtime_gs_root is not None else None
            ),
            device=args.realtime_gs_device,
            payload_bytes=ledger.complete_bytes,
            query_repeats=args.query_repeats,
        )

        save_image(str(artifact_dir / "reconstruction.png"), reconstruction)
        save_error_heatmap(
            str(artifact_dir / "error.png"),
            reconstruction - target,
            scale=args.error_scale,
        )
        _save_centers(artifact_dir / "centers.png", target, structures[count])
        bounds = _worst_bounds(reconstruction, target, mask, args.worst_crop_side)
        _save_crop(artifact_dir / "source_crop.png", target, bounds)
        _save_crop(artifact_dir / "reconstruction_crop.png", reconstruction, bounds)
        x0, y0, x1, y1 = bounds
        save_error_heatmap(
            str(artifact_dir / "error_crop.png"),
            (reconstruction - target)[y0:y1, x0:x1],
            scale=args.error_scale,
        )

        end_to_end_seconds = float(source_record["preprocess_seconds"]) + structure_times[count]
        end_to_end_seconds += appearance_times[(codec, quality)] + serialize_seconds
        adapter_value = realtime.get("realtime_gs_query_compatible")
        query_gate = adapter_value is True
        speed_gate = (
            isinstance(iterative_seconds, (int, float))
            and end_to_end_seconds <= 0.25 * float(iterative_seconds)
        )
        smaller_gate = ledger.complete_bytes < len(source_payload)
        artifact_gate = bool(metrics["artifact_gate_pass"])
        row: dict[str, object] = {
            "schema": REPORT_SCHEMA,
            "status": "diagnostic",
            "cell": cell_name,
            "appearance_cell": appearance_name,
            "appearance_codec": codec,
            "appearance_quality_or_effort": quality,
            "appearance_lossless": codec == "webp_lossless",
            "structural_count": count,
            "structural_seed": args.structural_seed,
            "appearance_rate_cell": count == args.primary_structural_count,
            "structure_capacity_cell": (codec, quality) == args.anchor_cell,
            "artifact_dir": str(artifact_dir.relative_to(args.out)),
            "packet_path": str(packet_path.relative_to(args.out)),
            "packet_sha256": _sha256(packet_path),
            "source_sha256": source_digest,
            "source_original_bytes": len(source_payload),
            "source_plus_mask_bytes": len(source_payload) + args.mask.stat().st_size,
            **reference_bytes,
            "crop_width": image.shape[1],
            "crop_height": image.shape[0],
            "crop_pixels": image.shape[0] * image.shape[1],
            "active_pixels": int(mask.sum()),
            "packet_complete_bytes": ledger.complete_bytes,
            "packet_manifest_bytes": ledger.manifest_bytes,
            "packet_appearance_bytes": ledger.appearance_bytes,
            "packet_structure_bytes": ledger.structure_bytes,
            "packet_container_overhead_bytes": ledger.container_overhead_bytes,
            "packet_bits_per_crop_pixel": 8.0 * ledger.complete_bytes / mask.size,
            "packet_bits_per_active_pixel": 8.0 * ledger.complete_bytes / int(mask.sum()),
            "source_over_packet_ratio": len(source_payload) / ledger.complete_bytes,
            "source_plus_mask_over_packet_ratio": (
                len(source_payload) + args.mask.stat().st_size
            )
            / ledger.complete_bytes,
            "canonical_crop_png_over_packet_ratio": (
                reference_bytes["canonical_crop_png_bytes"] / ledger.complete_bytes
            ),
            "canonical_matted_png_over_packet_ratio": (
                reference_bytes["canonical_matted_png_bytes"] / ledger.complete_bytes
            ),
            "rtgsv_bytes": rtgsv_bytes,
            "rtgsv_over_packet_ratio": (
                float(rtgsv_bytes) / ledger.complete_bytes
                if isinstance(rtgsv_bytes, (int, float))
                else None
            ),
            "preprocess_seconds": source_record["preprocess_seconds"],
            "appearance_encode_seconds": appearance_times[(codec, quality)],
            "structural_allocate_seconds": structure_times[count],
            "packet_serialize_seconds": serialize_seconds,
            "end_to_end_encode_estimated_seconds": end_to_end_seconds,
            "optimizer_iterations": 0,
            "cold_decode_seconds": cold_decode_seconds,
            "render_seconds": render_seconds,
            "metric_seconds": metric_seconds,
            "appearance_query_points": args.appearance_query_points,
            "appearance_query_seconds": appearance_query_seconds,
            "appearance_query_seconds_q10": float(np.quantile(appearance_timings, 0.10)),
            "appearance_query_seconds_q90": float(np.quantile(appearance_timings, 0.90)),
            "appearance_query_repeats": args.query_repeats,
            "appearance_query_points_per_second": (
                args.appearance_query_points / max(appearance_query_seconds, 1e-12)
            ),
            "numpy_dual_query_points": int(active_points.shape[0]),
            "numpy_dual_query_seconds": dual_query_seconds,
            "numpy_dual_query_seconds_q10": float(np.quantile(dual_timings, 0.10)),
            "numpy_dual_query_seconds_q90": float(np.quantile(dual_timings, 0.90)),
            "numpy_dual_query_repeats": args.query_repeats,
            "numpy_dual_query_points_per_second": (
                active_points.shape[0] / max(dual_query_seconds, 1e-12)
            ),
            "smaller_than_exact_source_gate": smaller_gate,
            "materially_faster_than_iterative_gate": speed_gate,
            "query_compatible_gate": query_gate,
            "artifact_safe_gate": artifact_gate,
            "killing_gate_pass": bool(smaller_gate and speed_gate and query_gate and artifact_gate),
            "error_scale": args.error_scale,
            "worst_crop_x0": x0,
            "worst_crop_y0": y0,
            "worst_crop_x1": x1,
            "worst_crop_y1": y1,
            **metrics,
            **structure_metrics[count],
            **realtime,
        }
        _write_json(artifact_dir / "config.json", asdict(config))
        _write_json(artifact_dir / "row.json", row)
        rows.append(row)

    _write_tables(args.out, rows)
    curves = _plot_curves(args.out, rows, curve_mode=args.curve_mode)
    _write_report(args.out, rows, curves, controls)
    config_record = {
        "schema": REPORT_SCHEMA,
        "status": "diagnostic",
        "claim_ready": False,
        "argv": sys.argv if argv is None else [str(Path(__file__)), *argv],
        "command": shlex.join(sys.argv if argv is None else [str(Path(__file__)), *argv]),
        "arguments": vars(args),
        "appearance_cells": appearance_cells,
        "structural_counts": counts,
        "source": source_record,
        "reference_bytes": reference_bytes,
        "source_snapshot": source_snapshot,
        "repository": _git_record(),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pillow": _version("pillow"),
            "torch": _version("torch"),
            "scipy": _version("scipy"),
            "matplotlib": _version("matplotlib"),
        },
        "evidence_limits": [
            "one exposed development image",
            "dirty source snapshot",
            "contextual controls have incompatible preprocessing/rate semantics",
            "no multi-view 3D reconstruction or held-out confirmation",
        ],
    }
    _write_json(args.out / "config.json", config_record)
    _write_manifest(args.out)
    print(args.out / "index.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
