#!/usr/bin/env python3
"""Run and package the HIER-006 parent-preserving residual-quadtree diagnostic.

The report deliberately distinguishes three sizes: canonical float32 field bytes, a lossless
Observation Field reference container, and a tree/coefficients-only structured proxy.  The last
assumes that mask-derived geometry and the hierarchy are shared with the decoder; it is not an
implemented codec and must not be reported as compressed file size.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import csv
import hashlib
from html import escape
import importlib.metadata
import json
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import sys
import time

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import hier005_pixel_contraction as report_utils  # noqa: E402

from structsplat.observation_field import ObservationField2D  # noqa: E402
from structsplat.pixel_contraction import render_observation_field  # noqa: E402
from structsplat.progressive_residual_quadtree import (  # noqa: E402
    ProgressiveResidualConfig,
    build_progressive_residual_quadtree,
    progressive_artifact_metrics,
    progressive_prefix_field,
)


METHOD = "progressive_residual_quadtree"
WORKFLOW_SCHEMA = "structsplat.current_pipeline.workflow.v1"
METRIC_SCHEMA = "structsplat.current_pipeline.metric.v1"
RUN_SCHEMA = "structsplat.current_pipeline.run.v1"
TRAJECTORY_SCHEMA = "structsplat.hier006.trajectory.v1"

SNAPSHOT_CURVES = (
    ("psnr_db", "Foreground-mask PSNR (dB)", False),
    ("ssim", "Full black-matted raster SSIM", False),
    ("ms_ssim", "Full black-matted raster MS-SSIM", False),
    ("lpips", "Full black-matted raster LPIPS (lower is better)", True),
    ("masked_mse", "Foreground-mask MSE", True),
    ("artifact_pixel_rmse_q99", "Displayed foreground pixel RMSE q99", True),
    ("artifact_pixel_rmse_q999", "Displayed foreground pixel RMSE q99.9", True),
    ("artifact_pixel_rmse_max", "Displayed foreground pixel RMSE maximum", True),
    (
        "artifact_pixel_rmse_fraction_gt_005",
        "Displayed foreground fraction with pixel RMSE > 0.05",
        True,
    ),
    (
        "artifact_pixel_rmse_fraction_gt_010",
        "Displayed foreground fraction with pixel RMSE > 0.10",
        True,
    ),
    ("artifact_patch_rmse_max_3", "Displayed maximum 3x3 patch RMSE", True),
    ("artifact_patch_rmse_max_7", "Displayed maximum 7x7 patch RMSE", True),
    ("artifact_patch_rmse_max_15", "Displayed maximum 15x15 patch RMSE", True),
    ("artifact_patch_rmse_max_31", "Displayed maximum 31x31 patch RMSE", True),
    ("raw_artifact_pixel_rmse_max", "Raw foreground pixel RMSE maximum", True),
    ("raw_artifact_patch7_rmse_max", "Raw maximum 7x7 patch RMSE", True),
    ("raw_artifact_normalized_violation", "Raw normalized artifact violation", True),
    ("raw_sse", "Raw foreground SSE", True),
    ("estimated_field_bytes", "Estimated uncoded full-field bytes", True),
    ("canonical_raw_bytes", "Canonical float32 field bytes", True),
    ("lossless_reference_bytes", "Lossless reference NPZ bytes", True),
    ("structured_proxy_bytes", "Tree plus float32 RGB coefficient proxy bytes", True),
    ("estimated_bits_per_pixel", "Estimated uncoded full-field bits/pixel", True),
    ("structured_proxy_bits_per_pixel", "Structured proxy bits/pixel", True),
    (
        "estimated_bits_per_active_pixel",
        "Estimated uncoded full-field bits/active pixel",
        True,
    ),
    (
        "structured_proxy_bits_per_active_pixel",
        "Structured proxy bits/active pixel",
        True,
    ),
    (
        "source_over_estimated_ratio",
        "Original JPEG bytes / estimated uncoded full-field bytes",
        False,
    ),
    (
        "source_over_structured_proxy_ratio",
        "Original JPEG bytes / structured proxy bytes",
        False,
    ),
    (
        "evaluation_png_over_estimated_ratio",
        "Evaluation PNG bytes / estimated uncoded full-field bytes",
        False,
    ),
    (
        "evaluation_png_over_structured_proxy_ratio",
        "Evaluation PNG bytes / structured proxy bytes",
        False,
    ),
    ("accepted_stage_count", "Accepted hierarchy stages", False),
    ("accepted_rows_through_prefix", "Accepted rows through prefix", False),
    ("level0_rows", "Pixel-scale level-0 rows", False),
    ("retained_ancestor_rows", "Retained non-leaf ancestor rows", False),
    ("negative_coefficient_fraction", "Negative RGB coefficient fraction", False),
    ("build_seconds_to_prefix", "Hierarchy build seconds through prefix", False),
    ("full_hierarchy_seconds", "Complete hierarchy build seconds", False),
    ("cold_decode_seconds", "Cold field decode seconds", False),
    ("render_seconds", "Cold maintained-render seconds", False),
    ("repeated_render_seconds", "Immediate repeated-render seconds", False),
    ("metric_seconds", "Metric evaluation seconds", False),
    (
        "repeated_render_parity_max_abs",
        "Repeated maintained-render parity maximum absolute error",
        True,
    ),
    ("total_seconds", "Build plus snapshot evaluation seconds", False),
)

STAGE_CURVES = (
    ("raw_sse", "Accepted-stage raw foreground SSE", True),
    ("raw_normalized_violation", "Accepted-stage raw artifact violation", True),
    ("display_pixel_rmse_max", "Accepted-stage displayed pixel RMSE maximum", True),
    ("display_patch7_rmse_max", "Accepted-stage displayed 7x7 RMSE maximum", True),
    ("rows_added", "Rows appended by accepted stage", False),
    ("optimizer_selected_step", "Selected optimizer step", False),
    ("optimization_seconds", "Stage coefficient optimization seconds", False),
    ("selection_seconds", "Stage error-selection seconds", False),
    ("cold_render_seconds", "Stage cold-validation render seconds", False),
    ("cumulative_stage_seconds", "Cumulative hierarchy stage seconds", False),
)

CHECKPOINT_CURVES = (
    ("raw_sse", "Optimizer-checkpoint raw foreground SSE", True),
    ("raw_normalized_violation", "Optimizer-checkpoint raw artifact violation", True),
    ("raw_pixel_rmse_max", "Optimizer-checkpoint raw pixel RMSE maximum", True),
    ("raw_patch7_rmse_max", "Optimizer-checkpoint raw 7x7 RMSE maximum", True),
    ("objective", "Optimizer-checkpoint training objective", True),
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
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(_jsonable(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _git(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    value = completed.stdout.strip()
    return value if completed.returncode == 0 and value else "unavailable"


def _repository_identity() -> dict[str, object]:
    completed = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    status = completed.stdout if completed.returncode == 0 else "status unavailable\n"
    return {
        "commit": _git("rev-parse", "HEAD"),
        "branch": _git("branch", "--show-current"),
        "dirty": bool(status),
        "status_sha256": hashlib.sha256(status.encode()).hexdigest(),
    }


def _snapshot_sources(output_root: Path) -> list[dict[str, object]]:
    sources = (
        Path(__file__).resolve(),
        ROOT / "src" / "structsplat" / "progressive_residual_quadtree.py",
        ROOT / "src" / "structsplat" / "pixel_contraction.py",
        ROOT / "scripts" / "experiments" / "hier005_pixel_contraction.py",
    )
    records = []
    for source in sources:
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


def _canonical_raw_bytes(field: ObservationField2D) -> int:
    return int(sum(array.nbytes for array in field._array_items().values()))


def _tree_depth_image(
    shape: tuple[int, int],
    stages: list[dict[str, object]],
    *,
    attempt_limit: int,
    start_level: int,
) -> np.ndarray:
    height, width = shape
    depth = np.zeros(shape, dtype=np.int16)
    for stage in stages:
        if stage["attempt_index"] > attempt_limit or stage["status"] != "accepted":
            continue
        if stage["kind"] != "residual_children":
            continue
        for level, cell_y, cell_x in stage["child_keys"]:
            side = 1 << int(level)
            x0 = int(cell_x) * side
            y0 = int(cell_y) * side
            x1 = min(width, x0 + side)
            y1 = min(height, y0 + side)
            depth[y0:y1, x0:x1] = np.maximum(
                depth[y0:y1, x0:x1], start_level - int(level)
            )
    palette = np.asarray(
        [
            [35, 46, 58],
            [44, 105, 154],
            [38, 150, 160],
            [61, 183, 118],
            [174, 203, 73],
            [245, 183, 48],
            [231, 92, 43],
            [180, 39, 77],
        ],
        dtype=np.uint8,
    )
    return palette[np.clip(depth, 0, len(palette) - 1)]


def _save_visuals(
    artifact_dir: Path,
    target: np.ndarray,
    reconstruction: np.ndarray,
    mask: np.ndarray,
    depth_rgb: np.ndarray,
    *,
    error_scale: float,
) -> dict[str, object]:
    from structsplat.cli import save_error_heatmap, save_image

    save_image(str(artifact_dir / "target.png"), target)
    save_image(str(artifact_dir / "reconstruction.png"), reconstruction)
    save_error_heatmap(
        str(artifact_dir / "error.png"), reconstruction - target, scale=error_scale
    )
    Image.fromarray(depth_rgb, mode="RGB").save(artifact_dir / "hierarchy_depth.png")

    display_target = np.rint(np.clip(target, 0.0, 1.0) * 255.0) / 255.0
    display_reconstruction = np.rint(np.clip(reconstruction, 0.0, 1.0) * 255.0) / 255.0
    pixel_rmse = np.sqrt(np.mean((display_reconstruction - display_target) ** 2, axis=2))
    pixel_rmse[~mask] = -1.0
    worst_y, worst_x = np.unravel_index(int(np.argmax(pixel_rmse)), pixel_rmse.shape)
    crop_side = min(96, target.shape[0], target.shape[1])
    x0 = min(max(0, int(worst_x) - crop_side // 2), target.shape[1] - crop_side)
    y0 = min(max(0, int(worst_y) - crop_side // 2), target.shape[0] - crop_side)
    x1, y1 = x0 + crop_side, y0 + crop_side
    save_image(str(artifact_dir / "target_crop.png"), target[y0:y1, x0:x1])
    save_image(
        str(artifact_dir / "reconstruction_crop.png"),
        reconstruction[y0:y1, x0:x1],
    )
    save_error_heatmap(
        str(artifact_dir / "error_crop.png"),
        reconstruction[y0:y1, x0:x1] - target[y0:y1, x0:x1],
        scale=error_scale,
    )
    Image.fromarray(depth_rgb[y0:y1, x0:x1], mode="RGB").save(
        artifact_dir / "hierarchy_depth_crop.png"
    )
    return {
        "worst_display_pixel_x": int(worst_x),
        "worst_display_pixel_y": int(worst_y),
        "worst_crop_xyxy": [x0, y0, x1, y1],
    }


def _attempt_limit(
    stages: list[dict[str, object]], snapshot_count: int, final_count: int
) -> int:
    if snapshot_count == final_count:
        return max(int(stage["attempt_index"]) for stage in stages)
    accepted = [
        int(stage["attempt_index"])
        for stage in stages
        if stage["accepted_rows"] > 0 and stage["count_after"] <= snapshot_count
    ]
    return max(accepted)


def _build_seconds(stages: list[dict[str, object]], attempt_limit: int) -> float:
    included = [
        stage for stage in stages if int(stage["attempt_index"]) <= attempt_limit
    ]
    return float(included[-1]["cumulative_elapsed_seconds"])


def _load_comparison(report_root: Path | None) -> dict[str, object]:
    if report_root is None:
        return {"available": False, "reason": "no comparison report requested"}
    metrics_path = report_root.resolve() / "metrics.json"
    if not metrics_path.is_file():
        return {
            "available": False,
            "reason": f"comparison metrics are missing: {metrics_path}",
        }
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    rows = payload.get("rows") if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        return {"available": False, "reason": "comparison metrics have no row list"}
    selected = []
    keys = (
        "method",
        "image",
        "n_gaussians",
        "psnr_db",
        "ssim",
        "ms_ssim",
        "lpips",
        "artifact_pixel_rmse_max",
        "artifact_patch_rmse_max_7",
        "artifact_gate_pass",
        "estimated_field_bytes",
        "canonical_raw_bytes",
        "lossless_reference_bytes",
        "source_over_estimated_ratio",
        "total_seconds",
        "stop_reason",
    )
    for row in rows:
        if isinstance(row, dict) and row.get("n_gaussians") in (4096, 8192):
            selected.append({key: row.get(key) for key in keys})
    return {
        "available": bool(selected),
        "source_report": str(report_root.resolve()),
        "metrics_sha256": _sha256(metrics_path),
        "warning": "external dirty diagnostic copied for context; not a controlled arm",
        "rows": selected,
    }


def _prefix_history(
    stages: list[dict[str, object]],
    checkpoints: list[dict[str, object]],
    *,
    attempt_limit: int,
    snapshot_count: int,
) -> dict[str, object]:
    return {
        "schema": TRAJECTORY_SCHEMA,
        "snapshot_count": snapshot_count,
        "attempt_limit": attempt_limit,
        "stages": [
            stage for stage in stages if int(stage["attempt_index"]) <= attempt_limit
        ],
        "checkpoints": [
            checkpoint
            for checkpoint in checkpoints
            if int(checkpoint["attempt_index"]) <= attempt_limit
        ],
    }


def _run_image(
    image_path: Path,
    image: np.ndarray,
    mask: np.ndarray | None,
    raster_record: dict[str, object],
    config: ProgressiveResidualConfig,
    args: argparse.Namespace,
    output_root: Path,
    repository: dict[str, object],
    source_id: str,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    active_mask = np.ones(image.shape[:2], dtype=bool) if mask is None else mask
    target = image * active_mask[:, :, None]
    result = build_progressive_residual_quadtree(image, config, mask=mask)
    stages = result.stage_records()
    checkpoints = result.checkpoint_records()
    trajectory = {
        "schema": TRAJECTORY_SCHEMA,
        "source_id": source_id,
        "config": asdict(config),
        "base_count": result.base_count,
        "final_count": result.final_count,
        "accepted_split_count": result.accepted_split_count,
        "stop_reason": result.stop_reason,
        "elapsed_seconds": result.elapsed_seconds,
        "initial_sse": result.initial_sse,
        "final_sse": result.final_sse,
        "prefix_bit_exact": result.prefix_bit_exact,
        "maintained_render_parity_max_abs": result.maintained_render_parity_max_abs,
        "snapshot_counts": result.snapshot_counts,
        "stages": stages,
        "checkpoints": checkpoints,
    }
    trajectory_path = output_root / f"trajectory_{image_path.stem}.json"
    _write_json(trajectory_path, trajectory)

    rows: list[dict[str, object]] = []
    for snapshot_count in result.snapshot_counts:
        variant = f"prefix_n{snapshot_count}"
        key = f"{image_path.stem}__{variant}"
        artifact_dir = output_root / "artifacts" / key
        artifact_dir.mkdir(parents=True, exist_ok=False)
        prefix = progressive_prefix_field(result.field, snapshot_count)
        field_path = artifact_dir / "field.observation.npz"
        prefix.save_lossless(field_path)
        decode_started = time.perf_counter()
        cold_field = ObservationField2D.load_lossless(field_path)
        cold_decode_seconds = time.perf_counter() - decode_started

        render_started = time.perf_counter()
        raw_reconstruction = render_observation_field(
            cold_field,
            device=config.device,
            renderer=config.renderer,
            render_chunk=config.render_chunk,
            apply_declared_alpha=False,
        )
        reconstruction = raw_reconstruction * active_mask[:, :, None]
        render_seconds = time.perf_counter() - render_started
        repeat_started = time.perf_counter()
        repeat = render_observation_field(
            cold_field,
            device=config.device,
            renderer=config.renderer,
            render_chunk=config.render_chunk,
            apply_declared_alpha=True,
        )
        repeated_render_seconds = time.perf_counter() - repeat_started
        repeated_render_parity = float(np.max(np.abs(repeat - reconstruction)))

        metric_started = time.perf_counter()
        metrics = report_utils._metric_values(
            reconstruction,
            target,
            active_mask,
            device=config.device,
            compute_lpips=args.lpips,
        )
        raw_metrics = progressive_artifact_metrics(
            raw_reconstruction,
            image,
            active_mask,
            pixel_threshold=config.pixel_rmse_threshold,
            patch7_threshold=config.patch7_rmse_threshold,
            displayed=False,
        )
        metric_seconds = time.perf_counter() - metric_started
        limit = _attempt_limit(stages, snapshot_count, result.final_count)
        build_seconds = _build_seconds(stages, limit)
        depth_rgb = _tree_depth_image(
            image.shape[:2],
            stages,
            attempt_limit=limit,
            start_level=config.start_level,
        )
        visual_record = _save_visuals(
            artifact_dir,
            target,
            reconstruction,
            active_mask,
            depth_rgb,
            error_scale=args.error_scale,
        )
        evaluation_png_bytes = (artifact_dir / "target.png").stat().st_size
        history_path = artifact_dir / "history.json"
        _write_json(
            history_path,
            _prefix_history(
                stages,
                checkpoints,
                attempt_limit=limit,
                snapshot_count=snapshot_count,
            ),
        )
        run_config_path = artifact_dir / "config.json"
        run_config = {
            "schema": RUN_SCHEMA,
            "method": METHOD,
            "variant": variant,
            "seed": 0,
            "source": {
                "relative": source_id,
                "path": str(image_path),
                "sha256": _sha256(image_path),
                "bytes": image_path.stat().st_size,
            },
            "mask": raster_record["mask"],
            "repository": repository,
            "progressive_config": asdict(config),
            "evaluation_raster": raster_record,
            "selection_rule": (
                "mask-aware sigma-smoothed residual energy per child row; stable ties"
            ),
            "acceptance_rule": (
                "cold-rendered candidate lexicographically improves raw normalized local "
                "artifact violation and then raw foreground SSE; otherwise exact rollback"
            ),
            "rate_warning": (
                "structured_proxy_bytes assumes shared mask/tree-derived geometry and is not "
                "an implemented coded stream"
            ),
        }
        _write_json(run_config_path, run_config)

        source_bytes = image_path.stat().st_size
        pixel_count = image.shape[0] * image.shape[1]
        active_pixels = int(active_mask.sum())
        alpha_bytes = 0 if prefix.packed_alpha is None else int(prefix.packed_alpha.nbytes)
        estimated_bytes = snapshot_count * config.estimated_row_bytes + alpha_bytes
        structured_proxy_bytes = snapshot_count * 3 * 4 + (snapshot_count + 7) // 8
        structured_proxy_bytes += alpha_bytes
        canonical_raw_bytes = _canonical_raw_bytes(prefix)
        lossless_bytes = field_path.stat().st_size
        accepted_prefix_stages = [
            stage
            for stage in stages
            if int(stage["attempt_index"]) <= limit and int(stage["accepted_rows"]) > 0
        ]
        level0_rows = sum(
            1
            for stage in accepted_prefix_stages
            for child in stage["child_keys"]
            if int(child[0]) == 0
        )
        if snapshot_count == result.final_count:
            build_seconds = result.elapsed_seconds
        total_seconds = (
            build_seconds
            + cold_decode_seconds
            + render_seconds
            + repeated_render_seconds
            + metric_seconds
        )
        artifact_relative = artifact_dir.relative_to(output_root)
        row: dict[str, object] = {
            "schema": METRIC_SCHEMA,
            "status": "ok",
            "method": METHOD,
            "variant": variant,
            "seed": 0,
            "source_id": source_id,
            "image": image_path.name,
            "n_gaussians": snapshot_count,
            "base_gaussians": result.base_count,
            "final_gaussians": result.final_count,
            "accepted_split_count": result.accepted_split_count,
            "accepted_stage_count": len(accepted_prefix_stages),
            "accepted_rows_through_prefix": snapshot_count,
            "level0_rows": level0_rows,
            "retained_ancestor_rows": snapshot_count - level0_rows,
            "stop_reason": result.stop_reason,
            "snapshot_attempt_limit": limit,
            "prefix_bit_exact": result.prefix_bit_exact,
            "maintained_render_parity_max_abs": (
                result.maintained_render_parity_max_abs
            ),
            "repeated_render_parity_max_abs": repeated_render_parity,
            "source_file_bytes": source_bytes,
            "evaluation_png_bytes": evaluation_png_bytes,
            "width": image.shape[1],
            "height": image.shape[0],
            "pixels": pixel_count,
            "active_pixels": active_pixels,
            "raw_sse": float(raw_metrics["sse"]),
            "raw_artifact_pixel_rmse_max": float(raw_metrics["pixel_rmse_max"]),
            "raw_artifact_patch7_rmse_max": float(raw_metrics["patch7_rmse_max"]),
            "raw_artifact_normalized_violation": float(
                raw_metrics["normalized_violation"]
            ),
            "estimated_field_bytes": estimated_bytes,
            "canonical_raw_bytes": canonical_raw_bytes,
            "lossless_reference_bytes": lossless_bytes,
            "coefficient_proxy_bytes": snapshot_count * 3 * 4,
            "tree_proxy_bits": snapshot_count,
            "structured_proxy_bytes": structured_proxy_bytes,
            "negative_coefficient_fraction": float(np.mean(prefix.rgb_coeff < 0.0)),
            "coefficient_min": float(np.min(prefix.rgb_coeff)),
            "coefficient_max": float(np.max(prefix.rgb_coeff)),
            "estimated_bits_per_pixel": 8.0 * estimated_bytes / pixel_count,
            "canonical_raw_bits_per_pixel": 8.0 * canonical_raw_bytes / pixel_count,
            "lossless_reference_bits_per_pixel": 8.0 * lossless_bytes / pixel_count,
            "structured_proxy_bits_per_pixel": 8.0
            * structured_proxy_bytes
            / pixel_count,
            "estimated_bits_per_active_pixel": 8.0 * estimated_bytes / active_pixels,
            "structured_proxy_bits_per_active_pixel": 8.0
            * structured_proxy_bytes
            / active_pixels,
            "source_over_estimated_ratio": source_bytes / max(estimated_bytes, 1),
            "source_over_canonical_raw_ratio": source_bytes
            / max(canonical_raw_bytes, 1),
            "source_over_lossless_reference_ratio": source_bytes
            / max(lossless_bytes, 1),
            "source_over_structured_proxy_ratio": source_bytes
            / max(structured_proxy_bytes, 1),
            "evaluation_png_over_estimated_ratio": evaluation_png_bytes
            / max(estimated_bytes, 1),
            "evaluation_png_over_canonical_raw_ratio": evaluation_png_bytes
            / max(canonical_raw_bytes, 1),
            "evaluation_png_over_lossless_reference_ratio": evaluation_png_bytes
            / max(lossless_bytes, 1),
            "evaluation_png_over_structured_proxy_ratio": evaluation_png_bytes
            / max(structured_proxy_bytes, 1),
            "build_seconds_to_prefix": build_seconds,
            "full_hierarchy_seconds": result.elapsed_seconds,
            "cold_decode_seconds": cold_decode_seconds,
            "render_seconds": render_seconds,
            "repeated_render_seconds": repeated_render_seconds,
            "metric_seconds": metric_seconds,
            "total_seconds": total_seconds,
            "psnr": float(metrics["psnr_db"]),
            **metrics,
            **visual_record,
            "target_png": str(artifact_relative / "target.png"),
            "reconstruction_png": str(artifact_relative / "reconstruction.png"),
            "error_png": str(artifact_relative / "error.png"),
            "field_npz": str(field_path.relative_to(output_root)),
            "history_json": str(history_path.relative_to(output_root)),
            "config_json": str(run_config_path.relative_to(output_root)),
            "field_sha256": _sha256(field_path),
            "field_canonical_sha256": cold_field.canonical_hash(),
            "artifact_dir": str(artifact_relative),
            "trajectory_json": str(trajectory_path.relative_to(output_root)),
        }
        _write_json(artifact_dir / "row.json", row)
        rows.append(row)
        print(
            f"{source_id} N={snapshot_count}: {row['psnr_db']:.3f} dB, "
            f"pixel={row['artifact_pixel_rmse_max']:.6f}, "
            f"patch7={row['artifact_patch_rmse_max_7']:.6f}, "
            f"gate={row['artifact_gate_pass']}",
            flush=True,
        )
    return rows, trajectory


def _write_tables(output_root: Path, rows: list[dict[str, object]]) -> None:
    _write_json(output_root / "metrics.json", rows)
    with (output_root / "metrics.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(_jsonable(row), sort_keys=True) + "\n")
    columns = sorted(
        {key for row in rows for key in row if key not in {"curves", "snapshots"}}
    )
    with (output_root / "metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: (
                        json.dumps(_jsonable(value), sort_keys=True)
                        if isinstance(value, (dict, list, tuple))
                        else "" if value is None else value
                    )
                    for key, value in row.items()
                }
            )


def _curve(
    rows: list[dict[str, object]],
    metric: str,
    label: str,
    prefer_log_y: bool,
    *,
    x_field: str = "n_gaussians",
    x_label: str = "achieved Gaussian count N",
) -> str | None:
    transformed = []
    for row in rows:
        x = row.get(x_field)
        if isinstance(x, (int, float)) and not isinstance(x, bool) and float(x) > 0:
            transformed.append({**row, "n_gaussians": x})
    svg = report_utils._metric_curve_svg(transformed, metric, label, prefer_log_y)
    if svg is None:
        return None
    return svg.replace("achieved Gaussian count N", x_label).replace("N=", f"{x_label}=")


def _write_curves(
    output_root: Path,
    rows: list[dict[str, object]],
    trajectories: list[dict[str, object]],
) -> list[dict[str, object]]:
    curve_root = output_root / "curves"
    curve_root.mkdir(parents=True, exist_ok=True)
    catalog: list[dict[str, object]] = []

    def emit(
        name: str,
        svg: str | None,
        *,
        metric: str,
        x: str,
        y_scale: str,
    ) -> None:
        if svg is None:
            return
        path = curve_root / f"{name}.svg"
        path.write_text(svg + "\n", encoding="utf-8")
        catalog.append(
            {
                "path": str(path.relative_to(output_root)),
                "metric": metric,
                "x": x,
                "preferred_y_scale": y_scale,
            }
        )

    for metric, label, log_y in SNAPSHOT_CURVES:
        emit(
            f"snapshot_{metric}",
            _curve(rows, metric, label, log_y),
            metric=metric,
            x="n_gaussians",
            y_scale="log10" if log_y else "linear",
        )

    stage_rows = []
    checkpoint_rows = []
    for trajectory in trajectories:
        image = str(trajectory["source_id"])
        for stage in trajectory["stages"]:
            if stage["accepted_rows"] <= 0:
                continue
            stage_rows.append(
                {
                    "image": image,
                    "n_gaussians": stage["count_after"],
                    "raw_sse": stage["sse_after"],
                    "raw_normalized_violation": stage["raw_violation_after"],
                    "display_pixel_rmse_max": stage["display_pixel_rmse_max"],
                    "display_patch7_rmse_max": stage["display_patch7_rmse_max"],
                    "rows_added": stage["accepted_rows"],
                    "optimizer_selected_step": stage["selected_step"],
                    "optimization_seconds": stage["optimization_seconds"],
                    "selection_seconds": stage["selection_seconds"],
                    "cold_render_seconds": stage["cold_render_seconds"],
                    "cumulative_stage_seconds": stage["cumulative_elapsed_seconds"],
                }
            )
        for checkpoint in trajectory["checkpoints"]:
            step = int(checkpoint["cumulative_optimizer_step"])
            if step <= 0:
                continue
            checkpoint_rows.append(
                {
                    "image": image,
                    "optimizer_step": step,
                    "raw_sse": checkpoint["raw_sse"],
                    "raw_normalized_violation": checkpoint[
                        "raw_normalized_violation"
                    ],
                    "raw_pixel_rmse_max": checkpoint["raw_pixel_rmse_max"],
                    "raw_patch7_rmse_max": checkpoint["raw_patch7_rmse_max"],
                    "objective": checkpoint["objective"],
                }
            )

    for metric, label, log_y in STAGE_CURVES:
        emit(
            f"stage_{metric}",
            _curve(stage_rows, metric, label, log_y),
            metric=metric,
            x="n_gaussians",
            y_scale="log10" if log_y else "linear",
        )
    for metric, label, log_y in CHECKPOINT_CURVES:
        emit(
            f"optimizer_{metric}",
            _curve(
                checkpoint_rows,
                metric,
                label,
                log_y,
                x_field="optimizer_step",
                x_label="cumulative optimizer step",
            ),
            metric=metric,
            x="cumulative_optimizer_step",
            y_scale="log10" if log_y else "linear",
        )
    _write_json(
        curve_root / "catalog.json",
        {"schema": "structsplat.hier006.curves.v1", "curves": catalog},
    )
    return catalog


def _format(value: object, digits: int = 4) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "pass" if value else "FAIL"
    if isinstance(value, float):
        if value != 0.0 and (abs(value) < 0.001 or abs(value) >= 10000):
            return f"{value:.3e}"
        return f"{value:.{digits}f}"
    return str(value)


def _write_index(
    output_root: Path,
    rows: list[dict[str, object]],
    curves: list[dict[str, object]],
    comparison: dict[str, object],
    command: str,
) -> None:
    table_rows = []
    cards = []
    for row in rows:
        artifact = str(row["artifact_dir"])
        table_rows.append(
            "<tr>"
            f"<td>{escape(str(row['variant']))}</td>"
            f"<td>{int(row['n_gaussians']):,}</td>"
            f"<td>{_format(row['psnr_db'], 3)}</td>"
            f"<td>{_format(row['ms_ssim'], 6)}</td>"
            f"<td>{_format(row.get('lpips'), 6)}</td>"
            f"<td>{_format(row['artifact_pixel_rmse_max'], 6)}</td>"
            f"<td>{_format(row['artifact_patch_rmse_max_7'], 6)}</td>"
            f"<td>{_format(row['artifact_gate_pass'])}</td>"
            f"<td>{int(row['estimated_field_bytes']):,}</td>"
            f"<td>{int(row['structured_proxy_bytes']):,}</td>"
            f"<td>{_format(row['source_over_estimated_ratio'], 3)}×</td>"
            f"<td>{_format(row['source_over_structured_proxy_ratio'], 3)}×</td>"
            f"<td>{_format(row['evaluation_png_over_estimated_ratio'], 3)}×</td>"
            f"<td>{_format(row['evaluation_png_over_structured_proxy_ratio'], 3)}×</td>"
            "</tr>"
        )
        links = (
            ("field", row["field_npz"]),
            ("history", row["history_json"]),
            ("config", row["config_json"]),
            ("target", row["target_png"]),
            ("reconstruction", row["reconstruction_png"]),
            ("error", row["error_png"]),
            ("row", f"{artifact}/row.json"),
            ("trajectory", row["trajectory_json"]),
        )
        cards.append(
            "<article class='card'>"
            f"<h3>{escape(str(row['variant']))}</h3>"
            "<div class='visuals'>"
            f"<figure><img src='{escape(str(row['target_png']))}'><figcaption>target</figcaption></figure>"
            f"<figure><img src='{escape(str(row['reconstruction_png']))}'><figcaption>reconstruction</figcaption></figure>"
            f"<figure><img src='{escape(str(row['error_png']))}'><figcaption>error × scale</figcaption></figure>"
            f"<figure><img src='{escape(artifact + '/hierarchy_depth.png')}'><figcaption>accepted depth</figcaption></figure>"
            "</div><div class='visuals crop'>"
            f"<figure><img src='{escape(artifact + '/target_crop.png')}'><figcaption>worst target crop</figcaption></figure>"
            f"<figure><img src='{escape(artifact + '/reconstruction_crop.png')}'><figcaption>worst reconstruction crop</figcaption></figure>"
            f"<figure><img src='{escape(artifact + '/error_crop.png')}'><figcaption>worst error crop</figcaption></figure>"
            f"<figure><img src='{escape(artifact + '/hierarchy_depth_crop.png')}'><figcaption>crop depth</figcaption></figure>"
            "</div><p>" + " · ".join(
                f"<a href='{escape(str(path))}'>{escape(label)}</a>" for label, path in links
            ) + "</p></article>"
        )

    comparison_rows = []
    if comparison.get("available"):
        for row in comparison["rows"]:
            comparison_rows.append(
                "<tr>"
                f"<td>{escape(str(row.get('method')))}</td>"
                f"<td>{int(row['n_gaussians']):,}</td>"
                f"<td>{_format(row.get('psnr_db'), 3)}</td>"
                f"<td>{_format(row.get('artifact_pixel_rmse_max'), 6)}</td>"
                f"<td>{_format(row.get('artifact_patch_rmse_max_7'), 6)}</td>"
                f"<td>{_format(row.get('artifact_gate_pass'))}</td>"
                "</tr>"
            )
    curve_cards = "".join(
        "<figure class='curve'><img src='{}'><figcaption><a href='{}'>{}</a></figcaption></figure>".format(
            escape(str(item["path"])),
            escape(str(item["path"])),
            escape(str(item["metric"])),
        )
        for item in curves
    )
    document = f"""<!doctype html>
<html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width'>
<title>HIER-006 progressive residual quadtree diagnostic</title>
<style>
body{{font-family:system-ui,sans-serif;margin:0;background:#f4f6f8;color:#1d2935}}
main{{max-width:1500px;margin:auto;padding:28px}} code{{word-break:break-all}}
.warning{{background:#fff4ce;border-left:5px solid #db8b00;padding:12px 16px}}
table{{border-collapse:collapse;width:100%;background:white}} th,td{{padding:8px;border:1px solid #d8e0e7;text-align:right}}
th:first-child,td:first-child{{text-align:left}} .card{{background:white;padding:16px;margin:20px 0;border-radius:8px}}
.visuals{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px}} figure{{margin:0}} img{{max-width:100%;height:auto}}
.visuals figure img{{width:100%;image-rendering:auto}} .crop figure img{{image-rendering:pixelated}}
.curves{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px}} .curve{{background:white;padding:8px}}
@media(max-width:850px){{.visuals,.curves{{grid-template-columns:1fr}}}}
</style></head><body><main>
<h1>HIER-006: parent-preserving progressive residual quadtree</h1>
<p class='warning'><strong>Diagnostic only.</strong> Parents are retained and child RGB residuals are signed. Geometry is fixed and mask-derived. The structured byte proxy assumes shared geometry/tree side information; it is not a self-contained codec size. CUDA atomics can make optimizer trajectories numerically nondeterministic.</p>
<p><strong>Executed command:</strong> <code>{escape(command)}</code></p>
<p><a href='manifest.json'>manifest</a> · <a href='metrics.json'>metrics JSON</a> · <a href='metrics.jsonl'>JSONL</a> · <a href='metrics.csv'>CSV</a> · <a href='comparison.json'>comparison context</a> · <a href='curves/catalog.json'>curve catalog</a></p>
<h2>Snapshot metrics</h2><div style='overflow:auto'><table><thead><tr>
<th>prefix</th><th>N</th><th>PSNR dB</th><th>MS-SSIM</th><th>LPIPS</th><th>pixel max</th><th>7×7 max</th><th>gate</th><th>full bytes</th><th>proxy bytes</th><th>JPEG/full</th><th>JPEG/proxy</th><th>eval PNG/full</th><th>eval PNG/proxy</th>
</tr></thead><tbody>{''.join(table_rows)}</tbody></table></div>
<h2>Existing HIER-005 context (not a controlled arm)</h2>
<table><thead><tr><th>method</th><th>N</th><th>PSNR dB</th><th>pixel max</th><th>7×7 max</th><th>gate</th></tr></thead><tbody>{''.join(comparison_rows) or '<tr><td colspan="6">unavailable</td></tr>'}</tbody></table>
<h2>Full images and worst-error crops</h2>{''.join(cards)}
<h2>Metric, stage, and optimizer curves</h2><div class='curves'>{curve_cards}</div>
</main></body></html>"""
    (output_root / "index.html").write_text(document, encoding="utf-8")


def _installed_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, nargs="+", required=True)
    parser.add_argument("--mask", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-side", type=int, default=512)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--start-level", type=int, default=6)
    parser.add_argument("--max-gaussians", type=int, default=8192)
    parser.add_argument("--leaf-scale", type=float, default=0.18)
    parser.add_argument("--sigma-cutoff", type=float, default=3.0)
    parser.add_argument("--support-fade-alpha", type=float, default=0.0)
    parser.add_argument("--error-smoothing-sigma", type=float, default=1.5)
    parser.add_argument("--max-rows-per-stage", type=int, default=256)
    parser.add_argument("--base-steps", type=int, default=400)
    parser.add_argument("--layer-steps", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--tail-fraction", type=float, default=0.01)
    parser.add_argument("--tail-weight", type=float, default=4.0)
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--pixel-threshold", type=float, default=0.02)
    parser.add_argument("--patch7-threshold", type=float, default=0.01)
    parser.add_argument("--estimated-row-bytes", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--renderer",
        choices=("additive", "cuda_additive", "cuda_tiled_additive"),
        default="cuda_additive",
    )
    parser.add_argument("--render-chunk", type=int, default=256)
    parser.add_argument("--max-stages", type=int)
    parser.add_argument("--milestone-counts", type=int, nargs="*", default=[4096])
    parser.add_argument("--lpips", action="store_true")
    parser.add_argument("--error-scale", type=float, default=4.0)
    parser.add_argument(
        "--comparison-report",
        type=Path,
        default=Path("results/hier005_janelle_artifact_hard3_touched_2026-08-05"),
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    output_root = args.out.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise SystemExit(f"refusing non-empty output directory: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    images = report_utils._discover_images(args.images)
    resolved_mask = None if args.mask is None else args.mask.resolve()
    if resolved_mask is not None and not resolved_mask.is_file():
        raise SystemExit(f"mask does not exist: {resolved_mask}")
    repository = _repository_identity()
    command = shlex.join([sys.executable, *sys.argv])
    source_snapshots = _snapshot_sources(output_root)
    comparison = _load_comparison(args.comparison_report)
    _write_json(output_root / "comparison.json", comparison)

    config = ProgressiveResidualConfig(
        start_level=args.start_level,
        max_gaussians=args.max_gaussians,
        leaf_scale_px=args.leaf_scale,
        sigma_cutoff=args.sigma_cutoff,
        support_fade_alpha=args.support_fade_alpha,
        error_smoothing_sigma_px=args.error_smoothing_sigma,
        max_rows_per_stage=args.max_rows_per_stage,
        base_steps=args.base_steps,
        layer_steps=args.layer_steps,
        learning_rate=args.learning_rate,
        tail_fraction=args.tail_fraction,
        tail_weight=args.tail_weight,
        checkpoint_every=args.checkpoint_every,
        pixel_rmse_threshold=args.pixel_threshold,
        patch7_rmse_threshold=args.patch7_threshold,
        estimated_row_bytes=args.estimated_row_bytes,
        device=args.device,
        renderer=args.renderer,
        render_chunk=args.render_chunk,
        max_stages=args.max_stages,
        milestone_counts=tuple(args.milestone_counts),
    )
    rows: list[dict[str, object]] = []
    trajectories = []
    image_manifest = []
    for image_index, image_path in enumerate(images):
        image, mask, raster_record = report_utils._load_evaluation_raster(
            image_path,
            resolved_mask,
            max_side=args.max_side,
            mask_threshold=args.mask_threshold,
        )
        source_id = image_path.name
        if any(item["relative"] == source_id for item in image_manifest):
            source_id = f"{image_index:03d}_{source_id}"
        image_manifest.append(
            {
                "relative": source_id,
                "path": str(image_path),
                "bytes": image_path.stat().st_size,
                "sha256": _sha256(image_path),
            }
        )
        image_rows, trajectory = _run_image(
            image_path,
            image,
            mask,
            raster_record,
            config,
            args,
            output_root,
            repository,
            source_id,
        )
        rows.extend(image_rows)
        trajectories.append(trajectory)
    _write_tables(output_root, rows)
    curves = _write_curves(output_root, rows, trajectories)
    _write_index(output_root, rows, curves, comparison, command)
    manifest = {
        "schema": WORKFLOW_SCHEMA,
        "status": "diagnostic",
        "claim_ready": False,
        "task": "HIER-006",
        "method": METHOD,
        "command": command,
        "repository": repository,
        "variants": sorted({str(row["variant"]) for row in rows}),
        "seeds": [0],
        "images": image_manifest,
        "source_snapshot": source_snapshots,
        "protocol": asdict(config),
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": _installed_version("torch"),
            "pillow": _installed_version("Pillow"),
            "lpips": _installed_version("lpips"),
        },
        "evidence_limits": [
            "single-image dirty-source diagnostic without an independent preregistration review",
            "CUDA additive gradients can vary through atomic accumulation order",
            "structured_proxy_bytes omits a real header, entropy model, and cold decoder",
            "lossless Observation Field NPZ is reference interchange, not compression",
            "original-file ratios compare the native JPEG against a resized evaluation field",
            "existing HIER-005 values are contextual and not a jointly executed controlled arm",
        ],
    }
    _write_json(output_root / "manifest.json", manifest)
    print(f"wrote diagnostic report: {output_root / 'index.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
