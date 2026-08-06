#!/usr/bin/env python3
"""Run the frozen HIER-007 artifact-first frontier-quadtree diagnostic.

Reproduce the frozen exposed-image run with the command recorded in
``tasks/HIER-007-artifact-first-frontier-quadtree.md``.  The four arms share one fitted base and
factor selection priority (energy/artifact-first) against reconciliation scope
(new-only/overlap).  Full active-field bytes, final-frontier proxies, and progressive-event
proxies are separate ledgers; neither proxy is an implemented codec rate.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import csv
from html import escape
import json
from pathlib import Path
import platform
import shlex
import shutil
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import hier005_pixel_contraction as report_utils  # noqa: E402
import hier006_progressive_residual_quadtree as report_helpers  # noqa: E402

from structsplat.artifact_first_quadtree import (  # noqa: E402
    ArtifactFirstQuadtreeConfig,
    ArtifactFirstQuadtreeResult,
    FrontierSnapshot,
    build_artifact_first_quadtree,
    initialize_artifact_first_quadtree,
)
from structsplat.observation_field import ObservationField2D  # noqa: E402
from structsplat.pixel_contraction import render_observation_field  # noqa: E402
from structsplat.progressive_residual_quadtree import (  # noqa: E402
    _cell_bounds,
    progressive_artifact_metrics,
)


METHOD = "artifact_first_frontier_quadtree"
WORKFLOW_SCHEMA = "structsplat.current_pipeline.workflow.v1"
METRIC_SCHEMA = "structsplat.current_pipeline.metric.v1"
RUN_SCHEMA = "structsplat.current_pipeline.run.v1"
TRAJECTORY_SCHEMA = "structsplat.hier007.trajectory.v1"
CURVE_SCHEMA = "structsplat.hier007.curves.v1"
ARM_CHOICES = (
    "energy__new_only",
    "artifact_first__new_only",
    "energy__overlap",
    "artifact_first__overlap",
)

SNAPSHOT_CURVES = (
    ("psnr_db", "Foreground-mask PSNR (dB)", False),
    ("ssim", "Full black-matted raster SSIM", False),
    ("ms_ssim", "Full black-matted raster MS-SSIM", False),
    ("lpips", "Full black-matted raster LPIPS (lower is better)", True),
    ("masked_mse", "Foreground-mask MSE", True),
    ("artifact_pixel_rmse_q99", "Displayed foreground pixel RMSE q99", True),
    ("artifact_pixel_rmse_q999", "Displayed foreground pixel RMSE q99.9", True),
    ("artifact_pixel_rmse_max", "Displayed foreground pixel RMSE maximum", True),
    ("artifact_patch_rmse_max_3", "Displayed maximum 3x3 patch RMSE", True),
    ("artifact_patch_rmse_max_7", "Displayed maximum 7x7 patch RMSE", True),
    ("artifact_patch_rmse_max_15", "Displayed maximum 15x15 patch RMSE", True),
    ("artifact_patch_rmse_max_31", "Displayed maximum 31x31 patch RMSE", True),
    ("raw_artifact_pixel_rmse_max", "Raw foreground pixel RMSE maximum", True),
    ("raw_artifact_patch7_rmse_max", "Raw maximum 7x7 patch RMSE", True),
    ("raw_artifact_normalized_violation", "Raw normalized artifact violation", True),
    ("raw_sse", "Raw foreground SSE", True),
    ("active_nodes", "Active Gaussian/frontier nodes", False),
    ("stored_nodes", "All accepted hierarchy nodes", False),
    ("inactive_nodes", "Inactive hierarchy ancestors", False),
    ("coefficient_event_rows", "Progressive coefficient-event rows", False),
    ("estimated_active_field_bytes", "Estimated uncoded active-field bytes", True),
    ("canonical_active_raw_bytes", "Canonical active float32 field bytes", True),
    ("lossless_reference_bytes", "Lossless reference NPZ bytes", True),
    ("final_frontier_proxy_bytes", "Final-frontier structural proxy bytes", True),
    ("progressive_event_proxy_bytes", "Progressive-event structural proxy bytes", True),
    ("estimated_active_bits_per_pixel", "Estimated active-field bits/pixel", True),
    ("final_frontier_proxy_bits_per_pixel", "Final-frontier proxy bits/pixel", True),
    ("progressive_event_proxy_bits_per_pixel", "Progressive-event proxy bits/pixel", True),
    ("source_over_estimated_active_ratio", "Native JPEG / active-field bytes", False),
    ("source_over_final_frontier_proxy_ratio", "Native JPEG / frontier proxy", False),
    ("source_over_progressive_event_proxy_ratio", "Native JPEG / event proxy", False),
    ("evaluation_png_over_estimated_active_ratio", "Evaluation PNG / active-field bytes", False),
    ("evaluation_png_over_final_frontier_proxy_ratio", "Evaluation PNG / frontier proxy", False),
    ("evaluation_png_over_progressive_event_proxy_ratio", "Evaluation PNG / event proxy", False),
    ("accepted_split_count", "Accepted parent splits", False),
    ("accepted_stage_count", "Accepted topology stages", False),
    ("level0_active_rows", "Active pixel-level rows", False),
    ("active_nonleaf_rows", "Active nonleaf rows", False),
    ("shared_base_seconds", "Shared base fit seconds", False),
    ("arm_seconds_to_snapshot", "Arm build seconds through snapshot", False),
    ("cold_decode_seconds", "Cold field decode seconds", False),
    ("render_seconds", "Cold maintained-render seconds", False),
    ("metric_seconds", "Metric evaluation seconds", False),
    ("total_seconds", "Build plus snapshot evaluation seconds", False),
)

STAGE_CURVES = (
    ("raw_sse", "Accepted-stage raw foreground SSE", True),
    ("raw_normalized_violation", "Accepted-stage raw artifact violation", True),
    ("display_pixel_rmse_max", "Accepted-stage displayed pixel RMSE maximum", True),
    ("display_patch7_rmse_max", "Accepted-stage displayed 7x7 RMSE maximum", True),
    ("stored_nodes", "Accepted-stage stored hierarchy nodes", False),
    ("inactive_nodes", "Accepted-stage inactive hierarchy nodes", False),
    ("coefficient_event_rows", "Accepted-stage coefficient-event rows", False),
    ("optimized_rows", "Locally optimized rows per accepted stage", False),
    ("frozen_rows", "Frozen rows per accepted stage", False),
    ("error_weight_min", "Minimum local update weight", False),
    ("error_weight_p50", "Median local update weight", False),
    ("error_weight_p90", "p90 local update weight", False),
    ("error_weight_max", "Maximum local update weight", False),
    ("error_weight_effective_rows", "Effective weighted local rows", False),
    ("raw_error_score_max", "Maximum support-averaged error score", True),
    ("selected_step", "Selected optimizer step", False),
    ("attribution_seconds", "Row-attribution seconds", False),
    ("optimization_seconds", "Local optimization seconds", False),
    ("cold_render_seconds", "Transactional cold-render seconds", False),
    ("cumulative_seconds", "Cumulative build seconds", False),
)

CHECKPOINT_CURVES = (
    ("raw_sse", "Optimizer-checkpoint raw foreground SSE", True),
    ("raw_normalized_violation", "Optimizer-checkpoint raw artifact violation", True),
    ("raw_pixel_rmse_max", "Optimizer-checkpoint raw pixel RMSE maximum", True),
    ("raw_patch7_rmse_max", "Optimizer-checkpoint raw 7x7 RMSE maximum", True),
    ("objective", "Optimizer-checkpoint training objective", True),
    ("optimized_count", "Optimizer-checkpoint locally active rows", False),
)


def _arm_config(args: argparse.Namespace, arm: str) -> ArtifactFirstQuadtreeConfig:
    selection_mode, reconciliation_scope = arm.split("__", maxsplit=1)
    return ArtifactFirstQuadtreeConfig(
        selection_mode=selection_mode,
        reconciliation_scope=reconciliation_scope,
        seed=0,
        start_level=args.start_level,
        max_gaussians=args.max_gaussians,
        leaf_scale_px=args.leaf_scale,
        sigma_cutoff=args.sigma_cutoff,
        support_fade_alpha=args.support_fade_alpha,
        error_smoothing_sigma_px=args.error_smoothing_sigma,
        error_weight_power=args.error_weight_power,
        error_weight_floor=args.error_weight_floor,
        error_weight_ceiling=args.error_weight_ceiling,
        overlap_margin_px=args.overlap_margin,
        max_child_rows_per_stage=args.max_child_rows_per_stage,
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


def _snapshot_sources(output_root: Path) -> list[dict[str, object]]:
    sources = (
        Path(__file__).resolve(),
        ROOT / "src" / "structsplat" / "artifact_first_quadtree.py",
        ROOT / "src" / "structsplat" / "progressive_residual_quadtree.py",
        ROOT / "src" / "structsplat" / "pixel_contraction.py",
        ROOT / "scripts" / "experiments" / "hier005_pixel_contraction.py",
        ROOT / "scripts" / "experiments" / "hier006_progressive_residual_quadtree.py",
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
                "sha256": report_helpers._sha256(destination),
            }
        )
    return records


def _frontier_depth_image(
    shape: tuple[int, int],
    active_keys: tuple[tuple[int, int, int], ...],
    start_level: int,
) -> np.ndarray:
    depth = np.zeros(shape, dtype=np.int16)
    for key in active_keys:
        x0, y0, x1, y1 = _cell_bounds(key, shape)
        depth[y0:y1, x0:x1] = start_level - key[0]
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


def _trajectory_record(
    result: ArtifactFirstQuadtreeResult,
    config: ArtifactFirstQuadtreeConfig,
    source_id: str,
    arm: str,
) -> dict[str, object]:
    return {
        "schema": TRAJECTORY_SCHEMA,
        "source_id": source_id,
        "arm": arm,
        "config": asdict(config),
        "base_count": result.base_count,
        "final_count": result.final_count,
        "stored_node_count": result.stored_node_count,
        "inactive_node_count": result.inactive_node_count,
        "accepted_split_count": result.accepted_split_count,
        "accepted_stage_count": result.accepted_stage_count,
        "coefficient_event_rows": result.coefficient_event_rows,
        "stop_reason": result.stop_reason,
        "shared_base_seconds": result.shared_base_seconds,
        "arm_elapsed_seconds": result.arm_elapsed_seconds,
        "elapsed_seconds": result.elapsed_seconds,
        "initial_sse": result.initial_sse,
        "final_sse": result.final_sse,
        "maintained_render_parity_max_abs": result.maintained_render_parity_max_abs,
        "snapshots": [
            {
                "label": snapshot.label,
                "attempt_index": snapshot.attempt_index,
                "active_count": snapshot.active_count,
                "active_keys": [list(key) for key in snapshot.active_keys],
                "stored_node_count": snapshot.stored_node_count,
                "coefficient_event_rows": snapshot.coefficient_event_rows,
            }
            for snapshot in result.snapshots
        ],
        "stages": result.stage_records(),
        "checkpoints": result.checkpoint_records(),
    }


def _stage_through(
    result: ArtifactFirstQuadtreeResult,
    attempt_index: int,
) -> list[dict[str, object]]:
    return [
        stage.to_record()
        for stage in result.stages
        if stage.attempt_index <= attempt_index
    ]


def _checkpoint_through(
    result: ArtifactFirstQuadtreeResult,
    attempt_index: int,
) -> list[dict[str, object]]:
    return [
        checkpoint.to_record()
        for checkpoint in result.checkpoints
        if checkpoint.attempt_index <= attempt_index
    ]


def _snapshot_build_seconds(
    result: ArtifactFirstQuadtreeResult,
    attempt_index: int,
) -> float:
    values = [
        stage.cumulative_elapsed_seconds
        for stage in result.stages
        if stage.attempt_index <= attempt_index
    ]
    return max(values) if values else result.shared_base_seconds


def _write_snapshot_row(
    *,
    image_path: Path,
    image: np.ndarray,
    active_mask: np.ndarray,
    raster_record: dict[str, object],
    source_id: str,
    arm: str,
    config: ArtifactFirstQuadtreeConfig,
    result: ArtifactFirstQuadtreeResult,
    snapshot: FrontierSnapshot,
    args: argparse.Namespace,
    output_root: Path,
    repository: dict[str, object],
    trajectory_path: Path,
) -> dict[str, object]:
    variant = (
        f"{arm}__{snapshot.label}__n{snapshot.active_count}"
        f"__a{snapshot.attempt_index}"
    )
    artifact_dir = output_root / "artifacts" / f"{image_path.stem}__{variant}"
    artifact_dir.mkdir(parents=True, exist_ok=False)
    field_path = artifact_dir / "field.observation.npz"
    snapshot.field.save_lossless(field_path)
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
    repeated = render_observation_field(
        cold_field,
        device=config.device,
        renderer=config.renderer,
        render_chunk=config.render_chunk,
        apply_declared_alpha=True,
    )
    repeated_render_seconds = time.perf_counter() - repeat_started
    repeated_parity = float(np.max(np.abs(repeated - reconstruction)))
    target = image * active_mask[:, :, None]
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
    depth = _frontier_depth_image(
        image.shape[:2], snapshot.active_keys, config.start_level
    )
    visual_record = report_helpers._save_visuals(
        artifact_dir,
        target,
        reconstruction,
        active_mask,
        depth,
        error_scale=args.error_scale,
    )
    evaluation_png_bytes = (artifact_dir / "target.png").stat().st_size
    history_path = artifact_dir / "history.json"
    history = {
        "schema": TRAJECTORY_SCHEMA,
        "source_id": source_id,
        "arm": arm,
        "snapshot_label": snapshot.label,
        "snapshot_attempt_index": snapshot.attempt_index,
        "active_keys": [list(key) for key in snapshot.active_keys],
        "stages": _stage_through(result, snapshot.attempt_index),
        "checkpoints": _checkpoint_through(result, snapshot.attempt_index),
    }
    report_helpers._write_json(history_path, history)
    run_config_path = artifact_dir / "config.json"
    run_config = {
        "schema": RUN_SCHEMA,
        "method": METHOD,
        "variant": variant,
        "seed": config.seed,
        "source": {
            "relative": source_id,
            "path": str(image_path),
            "sha256": report_helpers._sha256(image_path),
            "bytes": image_path.stat().st_size,
        },
        "mask": raster_record["mask"],
        "repository": repository,
        "arm": arm,
        "frontier_config": asdict(config),
        "evaluation_raster": raster_record,
        "selection_rule": (
            "energy: mask-aware smoothed residual energy per net active row; "
            "artifact_first: raw pixel/centered-complete-7x7 violation then energy"
        ),
        "reconciliation_rule": (
            "new_only or new children plus surviving active rows whose finite-support AABBs "
            "intersect selected parent supports expanded by the gate radius; all other rows "
            "detached and geometry fixed"
        ),
        "acceptance_rule": (
            "cold full-field candidate lexicographically improves raw normalized local "
            "artifact violation then foreground SSE with the HIER-006 float32 tie band"
        ),
        "rate_warning": (
            "final_frontier_proxy_bytes and progressive_event_proxy_bytes are uncoded, "
            "non-self-contained structural proxies"
        ),
    }
    report_helpers._write_json(run_config_path, run_config)

    stages = [
        stage
        for stage in result.stages
        if stage.attempt_index <= snapshot.attempt_index and stage.status == "accepted"
    ]
    accepted_replacements = [stage for stage in stages if stage.kind == "parent_replacement"]
    accepted_split_count = sum(len(stage.parent_keys) for stage in accepted_replacements)
    source_bytes = image_path.stat().st_size
    active_count = snapshot.active_count
    stored_count = snapshot.stored_node_count
    inactive_count = stored_count - active_count
    alpha_bytes = 0 if cold_field.packed_alpha is None else int(cold_field.packed_alpha.nbytes)
    estimated_active_bytes = active_count * config.estimated_row_bytes + alpha_bytes
    canonical_active_bytes = int(
        sum(array.nbytes for array in cold_field._array_items().values())
    )
    active_coefficient_bytes = active_count * 3 * np.dtype(np.float32).itemsize
    tree_bits = stored_count
    final_frontier_proxy = active_coefficient_bytes + (tree_bits + 7) // 8 + alpha_bytes
    event_coefficient_bytes = snapshot.coefficient_event_rows * 3 * np.dtype(np.float32).itemsize
    progressive_event_proxy = event_coefficient_bytes + (tree_bits + 7) // 8 + alpha_bytes
    lossless_bytes = field_path.stat().st_size
    pixel_count = image.shape[0] * image.shape[1]
    active_pixels = int(active_mask.sum())
    build_seconds = _snapshot_build_seconds(result, snapshot.attempt_index)
    arm_seconds = max(build_seconds - result.shared_base_seconds, 0.0)
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
        "arm": arm,
        "selection_mode": config.selection_mode,
        "reconciliation_scope": config.reconciliation_scope,
        "snapshot_label": snapshot.label,
        "snapshot_attempt_index": snapshot.attempt_index,
        "seed": config.seed,
        "source_id": source_id,
        "image": f"{source_id} · {arm}",
        "image_file": image_path.name,
        "n_gaussians": active_count,
        "active_nodes": active_count,
        "stored_nodes": stored_count,
        "inactive_nodes": inactive_count,
        "base_gaussians": result.base_count,
        "final_gaussians": result.final_count,
        "accepted_split_count": accepted_split_count,
        "accepted_stage_count": len(accepted_replacements),
        "coefficient_event_rows": snapshot.coefficient_event_rows,
        "level0_active_rows": sum(key[0] == 0 for key in snapshot.active_keys),
        "active_nonleaf_rows": sum(key[0] > 0 for key in snapshot.active_keys),
        "stop_reason": result.stop_reason,
        "frontier_partition_valid": all(stage.frontier_partition_valid for stage in stages),
        "untouched_coefficients_bit_exact": all(
            stage.untouched_coefficients_bit_exact for stage in stages
        ),
        "rollback_bit_exact": all(stage.rollback_bit_exact for stage in result.stages),
        "maintained_render_parity_max_abs": result.maintained_render_parity_max_abs,
        "snapshot_cold_parity_max_abs": float(
            np.max(np.abs(raw_reconstruction - snapshot.reconstruction_raw))
        ),
        "repeated_render_parity_max_abs": repeated_parity,
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
        "estimated_active_field_bytes": estimated_active_bytes,
        "canonical_active_raw_bytes": canonical_active_bytes,
        "lossless_reference_bytes": lossless_bytes,
        "active_coefficient_proxy_bytes": active_coefficient_bytes,
        "tree_proxy_bits": tree_bits,
        "final_frontier_proxy_bytes": final_frontier_proxy,
        "progressive_event_coefficient_proxy_bytes": event_coefficient_bytes,
        "progressive_event_proxy_bytes": progressive_event_proxy,
        "estimated_active_bits_per_pixel": 8.0 * estimated_active_bytes / pixel_count,
        "canonical_active_raw_bits_per_pixel": 8.0 * canonical_active_bytes / pixel_count,
        "lossless_reference_bits_per_pixel": 8.0 * lossless_bytes / pixel_count,
        "final_frontier_proxy_bits_per_pixel": 8.0 * final_frontier_proxy / pixel_count,
        "progressive_event_proxy_bits_per_pixel": 8.0
        * progressive_event_proxy
        / pixel_count,
        "estimated_active_bits_per_active_pixel": 8.0
        * estimated_active_bytes
        / active_pixels,
        "final_frontier_proxy_bits_per_active_pixel": 8.0
        * final_frontier_proxy
        / active_pixels,
        "progressive_event_proxy_bits_per_active_pixel": 8.0
        * progressive_event_proxy
        / active_pixels,
        "source_over_estimated_active_ratio": source_bytes / max(estimated_active_bytes, 1),
        "source_over_canonical_active_ratio": source_bytes / max(canonical_active_bytes, 1),
        "source_over_lossless_reference_ratio": source_bytes / max(lossless_bytes, 1),
        "source_over_final_frontier_proxy_ratio": source_bytes
        / max(final_frontier_proxy, 1),
        "source_over_progressive_event_proxy_ratio": source_bytes
        / max(progressive_event_proxy, 1),
        "evaluation_png_over_estimated_active_ratio": evaluation_png_bytes
        / max(estimated_active_bytes, 1),
        "evaluation_png_over_canonical_active_ratio": evaluation_png_bytes
        / max(canonical_active_bytes, 1),
        "evaluation_png_over_lossless_reference_ratio": evaluation_png_bytes
        / max(lossless_bytes, 1),
        "evaluation_png_over_final_frontier_proxy_ratio": evaluation_png_bytes
        / max(final_frontier_proxy, 1),
        "evaluation_png_over_progressive_event_proxy_ratio": evaluation_png_bytes
        / max(progressive_event_proxy, 1),
        "negative_coefficient_fraction": float(np.mean(cold_field.rgb_coeff < 0.0)),
        "coefficient_min": float(np.min(cold_field.rgb_coeff)),
        "coefficient_max": float(np.max(cold_field.rgb_coeff)),
        "shared_base_seconds": result.shared_base_seconds,
        "arm_seconds_to_snapshot": arm_seconds,
        "build_seconds_to_snapshot": build_seconds,
        "full_arm_seconds": result.arm_elapsed_seconds,
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
        "field_sha256": report_helpers._sha256(field_path),
        "field_canonical_sha256": cold_field.canonical_hash(),
        "artifact_dir": str(artifact_relative),
        "trajectory_json": str(trajectory_path.relative_to(output_root)),
    }
    report_helpers._write_json(artifact_dir / "row.json", row)
    print(
        f"{source_id} {arm} {snapshot.label} N={active_count}: "
        f"{row['psnr_db']:.3f} dB, pixel={row['artifact_pixel_rmse_max']:.6f}, "
        f"patch7={row['artifact_patch_rmse_max_7']:.6f}, "
        f"gate={row['artifact_gate_pass']}",
        flush=True,
    )
    return row


def _run_image(
    *,
    image_path: Path,
    image: np.ndarray,
    mask: np.ndarray | None,
    raster_record: dict[str, object],
    source_id: str,
    args: argparse.Namespace,
    output_root: Path,
    repository: dict[str, object],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    active_mask = np.ones(image.shape[:2], dtype=bool) if mask is None else mask
    first_config = _arm_config(args, args.arms[0])
    shared_base = initialize_artifact_first_quadtree(
        image,
        first_config,
        mask=active_mask,
    )
    rows = []
    trajectories = []
    for arm in args.arms:
        config = _arm_config(args, arm)
        result = build_artifact_first_quadtree(
            image,
            config,
            mask=active_mask,
            start_state=shared_base,
        )
        trajectory = _trajectory_record(result, config, source_id, arm)
        trajectory_path = output_root / f"trajectory_{image_path.stem}__{arm}.json"
        report_helpers._write_json(trajectory_path, trajectory)
        trajectories.append(trajectory)
        for snapshot in result.snapshots:
            rows.append(
                _write_snapshot_row(
                    image_path=image_path,
                    image=image,
                    active_mask=active_mask,
                    raster_record=raster_record,
                    source_id=source_id,
                    arm=arm,
                    config=config,
                    result=result,
                    snapshot=snapshot,
                    args=args,
                    output_root=output_root,
                    repository=repository,
                    trajectory_path=trajectory_path,
                )
            )
    return rows, trajectories


def _write_tables(output_root: Path, rows: list[dict[str, object]]) -> None:
    report_helpers._write_json(output_root / "metrics.json", rows)
    with (output_root / "metrics.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(report_helpers._jsonable(row), sort_keys=True) + "\n"
            )
    columns = sorted(
        {key for row in rows for key in row if key not in {"curves", "snapshots"}}
    )
    with (output_root / "metrics.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: (
                        json.dumps(report_helpers._jsonable(value), sort_keys=True)
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
    x_label: str = "active Gaussian count N",
) -> str | None:
    transformed = []
    for row in rows:
        x = row.get(x_field)
        if isinstance(x, (int, float)) and not isinstance(x, bool) and float(x) > 0:
            transformed.append({**row, "n_gaussians": x})
    svg = report_utils._metric_curve_svg(transformed, metric, label, prefer_log_y)
    if svg is None:
        return None
    return svg.replace("achieved Gaussian count N", x_label)


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
            x="active_count",
            y_scale="log10" if log_y else "linear",
        )

    stage_rows = []
    checkpoint_rows = []
    attempt_rows = []
    for trajectory in trajectories:
        group = f"{trajectory['source_id']} · {trajectory['arm']}"
        for stage in trajectory["stages"]:
            if stage["kind"] != "parent_replacement":
                continue
            attempt_rows.append(
                {
                    "image": group,
                    "attempt_index": int(stage["attempt_index"]) + 1,
                    "accepted": 1.0 if stage["status"] == "accepted" else 0.0,
                    "proposed_child_rows": stage["proposed_child_rows"],
                    "proposed_net_rows": stage["proposed_net_rows"],
                    "optimized_rows": stage["optimized_rows"],
                    "frozen_rows": stage["frozen_rows"],
                    "attempted_optimizer_steps": stage["attempted_optimizer_steps"],
                    "optimization_seconds": stage["optimization_seconds"],
                    "attribution_seconds": stage["attribution_seconds"],
                    "cold_render_seconds": stage["cold_render_seconds"],
                }
            )
            if stage["status"] != "accepted":
                continue
            stage_rows.append(
                {
                    "image": group,
                    "n_gaussians": stage["count_after"],
                    "raw_sse": stage["sse_after"],
                    "raw_normalized_violation": stage["raw_violation_after"],
                    "display_pixel_rmse_max": stage["display_pixel_rmse_max"],
                    "display_patch7_rmse_max": stage["display_patch7_rmse_max"],
                    "stored_nodes": stage["stored_nodes_after"],
                    "inactive_nodes": stage["inactive_nodes_after"],
                    "coefficient_event_rows": stage["coefficient_event_rows_after"],
                    "optimized_rows": stage["optimized_rows"],
                    "frozen_rows": stage["frozen_rows"],
                    "error_weight_min": stage["error_weight_min"],
                    "error_weight_p50": stage["error_weight_p50"],
                    "error_weight_p90": stage["error_weight_p90"],
                    "error_weight_max": stage["error_weight_max"],
                    "error_weight_effective_rows": stage[
                        "error_weight_effective_rows"
                    ],
                    "raw_error_score_max": stage["raw_error_score_max"],
                    "selected_step": stage["selected_step"],
                    "attribution_seconds": stage["attribution_seconds"],
                    "optimization_seconds": stage["optimization_seconds"],
                    "cold_render_seconds": stage["cold_render_seconds"],
                    "cumulative_seconds": stage["cumulative_elapsed_seconds"],
                }
            )
        for checkpoint in trajectory["checkpoints"]:
            step = int(checkpoint["cumulative_optimizer_step"])
            if step <= 0:
                continue
            checkpoint_rows.append(
                {
                    "image": group,
                    "optimizer_step": step,
                    "raw_sse": checkpoint["raw_sse"],
                    "raw_normalized_violation": checkpoint[
                        "raw_normalized_violation"
                    ],
                    "raw_pixel_rmse_max": checkpoint["raw_pixel_rmse_max"],
                    "raw_patch7_rmse_max": checkpoint["raw_patch7_rmse_max"],
                    "objective": checkpoint["objective"],
                    "optimized_count": checkpoint["optimized_count"],
                }
            )

    for metric, label, log_y in STAGE_CURVES:
        emit(
            f"stage_{metric}",
            _curve(stage_rows, metric, label, log_y),
            metric=metric,
            x="active_count",
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
    for metric, label, log_y in (
        ("accepted", "Transactional acceptance (1=yes)", False),
        ("proposed_child_rows", "Proposed child rows per attempt", False),
        ("proposed_net_rows", "Proposed net active rows per attempt", False),
        ("optimized_rows", "Optimized rows per attempt", False),
        ("frozen_rows", "Frozen rows per attempt", False),
        ("attempted_optimizer_steps", "Optimizer steps per attempt", False),
        ("optimization_seconds", "Optimization seconds per attempt", False),
        ("attribution_seconds", "Attribution seconds per attempt", False),
        ("cold_render_seconds", "Cold-render seconds per attempt", False),
    ):
        emit(
            f"attempt_{metric}",
            _curve(
                attempt_rows,
                metric,
                label,
                log_y,
                x_field="attempt_index",
                x_label="split attempt index",
            ),
            metric=metric,
            x="attempt_index",
            y_scale="log10" if log_y else "linear",
        )
    report_helpers._write_json(
        curve_root / "catalog.json",
        {"schema": CURVE_SCHEMA, "curves": catalog},
    )
    return catalog


def _load_context(path: Path | None, label: str) -> list[dict[str, object]]:
    if path is None:
        return []
    metrics_path = path / "metrics.json"
    if not metrics_path.is_file():
        return []
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    rows = payload.get("rows") if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        return []
    selected = []
    for row in rows:
        if not isinstance(row, dict) or row.get("status") not in {"ok", "diagnostic"}:
            continue
        count = row.get("n_gaussians")
        if not isinstance(count, int):
            continue
        if count < 3500 and count != min(
            [
                item.get("n_gaussians", count)
                for item in rows
                if isinstance(item, dict) and isinstance(item.get("n_gaussians"), int)
            ],
            default=count,
        ):
            continue
        selected.append(
            {
                "context": label,
                "method": row.get("method"),
                "variant": row.get("variant"),
                "n_gaussians": count,
                "psnr_db": row.get("psnr_db", row.get("psnr")),
                "ms_ssim": row.get("ms_ssim"),
                "lpips": row.get("lpips"),
                "artifact_pixel_rmse_max": row.get("artifact_pixel_rmse_max"),
                "artifact_patch_rmse_max_7": row.get("artifact_patch_rmse_max_7"),
                "artifact_gate_pass": row.get("artifact_gate_pass"),
            }
        )
    return selected


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
    context: list[dict[str, object]],
    command: str,
) -> None:
    terminal = [row for row in rows if row["snapshot_label"] == "terminal"]
    terminal_rows = "".join(
        "<tr>"
        f"<td>{escape(str(row['arm']))}</td>"
        f"<td>{int(row['n_gaussians']):,}</td>"
        f"<td>{int(row['stored_nodes']):,}</td>"
        f"<td>{int(row['inactive_nodes']):,}</td>"
        f"<td>{_format(row['psnr_db'], 3)}</td>"
        f"<td>{_format(row['ms_ssim'], 6)}</td>"
        f"<td>{_format(row.get('lpips'), 6)}</td>"
        f"<td>{_format(row['artifact_pixel_rmse_max'], 6)}</td>"
        f"<td>{_format(row['artifact_patch_rmse_max_7'], 6)}</td>"
        f"<td>{_format(row['artifact_gate_pass'])}</td>"
        f"<td>{int(row['canonical_active_raw_bytes']):,}</td>"
        f"<td>{int(row['final_frontier_proxy_bytes']):,}</td>"
        f"<td>{int(row['progressive_event_proxy_bytes']):,}</td>"
        f"<td>{_format(row['full_arm_seconds'], 3)}</td>"
        "</tr>"
        for row in terminal
    )
    all_rows = "".join(
        "<tr>"
        f"<td>{escape(str(row['arm']))}</td>"
        f"<td>{escape(str(row['snapshot_label']))}</td>"
        f"<td>{int(row['n_gaussians']):,}</td>"
        f"<td>{_format(row['psnr_db'], 3)}</td>"
        f"<td>{_format(row['artifact_pixel_rmse_max'], 6)}</td>"
        f"<td>{_format(row['artifact_patch_rmse_max_7'], 6)}</td>"
        f"<td>{_format(row['artifact_gate_pass'])}</td>"
        f"<td>{int(row['accepted_split_count']):,}</td>"
        f"<td>{int(row['coefficient_event_rows']):,}</td>"
        "</tr>"
        for row in rows
    )
    cards = []
    for row in rows:
        artifact = str(row["artifact_dir"])
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
            f"<h3>{escape(str(row['arm']))} · {escape(str(row['snapshot_label']))} · "
            f"N={int(row['n_gaussians']):,}</h3>"
            "<div class='visuals'>"
            f"<figure><a href='{escape(str(row['target_png']))}'><img src='{escape(str(row['target_png']))}'></a><figcaption>target</figcaption></figure>"
            f"<figure><a href='{escape(str(row['reconstruction_png']))}'><img src='{escape(str(row['reconstruction_png']))}'></a><figcaption>reconstruction</figcaption></figure>"
            f"<figure><a href='{escape(str(row['error_png']))}'><img src='{escape(str(row['error_png']))}'></a><figcaption>error × scale</figcaption></figure>"
            f"<figure><a href='{escape(artifact + '/hierarchy_depth.png')}'><img src='{escape(artifact + '/hierarchy_depth.png')}'></a><figcaption>active frontier depth</figcaption></figure>"
            "</div><div class='visuals crop'>"
            f"<figure><a href='{escape(artifact + '/target_crop.png')}'><img src='{escape(artifact + '/target_crop.png')}'></a><figcaption>worst target crop</figcaption></figure>"
            f"<figure><a href='{escape(artifact + '/reconstruction_crop.png')}'><img src='{escape(artifact + '/reconstruction_crop.png')}'></a><figcaption>worst reconstruction crop</figcaption></figure>"
            f"<figure><a href='{escape(artifact + '/error_crop.png')}'><img src='{escape(artifact + '/error_crop.png')}'></a><figcaption>worst error crop</figcaption></figure>"
            f"<figure><a href='{escape(artifact + '/hierarchy_depth_crop.png')}'><img src='{escape(artifact + '/hierarchy_depth_crop.png')}'></a><figcaption>crop frontier depth</figcaption></figure>"
            "</div><p>"
            + " · ".join(
                f"<a href='{escape(str(path))}'>{escape(label)}</a>"
                for label, path in links
            )
            + "</p></article>"
        )
    context_rows = "".join(
        "<tr>"
        f"<td>{escape(str(row['context']))}</td>"
        f"<td>{escape(str(row.get('variant')))}</td>"
        f"<td>{int(row['n_gaussians']):,}</td>"
        f"<td>{_format(row.get('psnr_db'), 3)}</td>"
        f"<td>{_format(row.get('artifact_pixel_rmse_max'), 6)}</td>"
        f"<td>{_format(row.get('artifact_patch_rmse_max_7'), 6)}</td>"
        f"<td>{_format(row.get('artifact_gate_pass'))}</td>"
        "</tr>"
        for row in context
    )
    curve_cards = "".join(
        "<figure class='curve'><a href='{0}'><img src='{0}'></a><figcaption>{1}</figcaption></figure>".format(
            escape(str(item["path"])), escape(str(item["metric"]))
        )
        for item in curves
    )
    document = f"""<!doctype html>
<html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width'>
<title>HIER-007 artifact-first frontier-quadtree diagnostic</title>
<style>
body{{font-family:system-ui,sans-serif;margin:0;background:#f4f6f8;color:#1d2935}}
main{{max-width:1550px;margin:auto;padding:28px}} code{{word-break:break-all}}
.warning{{background:#fff4ce;border-left:5px solid #db8b00;padding:12px 16px}}
table{{border-collapse:collapse;width:100%;background:white}}th,td{{padding:8px;border:1px solid #d8e0e7;text-align:right}}
th:first-child,td:first-child{{text-align:left}}.card{{background:white;padding:16px;margin:20px 0;border-radius:8px}}
.visuals{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px}}figure{{margin:0}}img{{max-width:100%;height:auto}}
.visuals figure img{{width:100%}}.crop figure img{{image-rendering:pixelated}}.curves{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px}}
.curve{{background:white;padding:8px}}@media(max-width:850px){{.visuals,.curves{{grid-template-columns:1fr}}}}
</style></head><body><main>
<h1>HIER-007: artifact-first parent-replacing frontier quadtree</h1>
<p class='warning'><strong>Diagnostic only.</strong> C0001 is exposed and this run is source-dirty, single-image, and single-seed. Parents are inactive after splitting; geometry is fixed. Frontier/event byte values are non-self-contained proxies, not codec rates. CUDA optimizer paths are device/source-bound and can vary through atomic accumulation.</p>
<p>The 2×2 arms share one fitted base. Axes are split priority (<code>energy</code> vs <code>artifact_first</code>) and RGB reconciliation (<code>new_only</code> vs finite-support <code>overlap</code>). Every trial is accepted only by a cold full-field local-artifact/SSE transaction.</p>
<p><strong>Executed command:</strong> <code>{escape(command)}</code></p>
<p><a href='manifest.json'>manifest</a> · <a href='metrics.json'>metrics JSON</a> · <a href='metrics.jsonl'>JSONL</a> · <a href='metrics.csv'>CSV</a> · <a href='context.json'>context</a> · <a href='curves/catalog.json'>curve catalog</a></p>
<h2>Terminal factorial comparison</h2><div style='overflow:auto'><table><thead><tr><th>arm</th><th>active N</th><th>stored</th><th>inactive</th><th>PSNR dB</th><th>MS-SSIM</th><th>LPIPS</th><th>pixel max</th><th>7×7 max</th><th>gate</th><th>canonical active bytes</th><th>frontier proxy</th><th>event proxy</th><th>arm sec</th></tr></thead><tbody>{terminal_rows}</tbody></table></div>
<h2>All saved snapshots</h2><div style='overflow:auto'><table><thead><tr><th>arm</th><th>snapshot</th><th>N</th><th>PSNR dB</th><th>pixel max</th><th>7×7 max</th><th>gate</th><th>splits</th><th>coefficient events</th></tr></thead><tbody>{all_rows}</tbody></table></div>
<h2>Existing context (not jointly executed controls)</h2><table><thead><tr><th>context</th><th>variant</th><th>N</th><th>PSNR dB</th><th>pixel max</th><th>7×7 max</th><th>gate</th></tr></thead><tbody>{context_rows or '<tr><td colspan="7">unavailable</td></tr>'}</tbody></table>
<h2>Full images and worst-error crops</h2>{''.join(cards)}
<h2>Quality, artifact, topology, byte, work, and convergence curves</h2><div class='curves'>{curve_cards}</div>
</main></body></html>"""
    (output_root / "index.html").write_text(document, encoding="utf-8")


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
    parser.add_argument("--error-weight-power", type=float, default=0.5)
    parser.add_argument("--error-weight-floor", type=float, default=0.05)
    parser.add_argument("--error-weight-ceiling", type=float, default=4.0)
    parser.add_argument("--overlap-margin", type=int, default=3)
    parser.add_argument("--max-child-rows-per-stage", type=int, default=256)
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
    parser.add_argument(
        "--arms",
        nargs="+",
        choices=ARM_CHOICES,
        default=list(ARM_CHOICES),
    )
    parser.add_argument("--lpips", action="store_true")
    parser.add_argument("--error-scale", type=float, default=4.0)
    parser.add_argument(
        "--hier005-report",
        type=Path,
        default=Path("results/hier005_janelle_artifact_hard3_touched_2026-08-05"),
    )
    parser.add_argument(
        "--hier006-report",
        type=Path,
        default=Path(
            "results/hier006_janelle_progressive_residual_quadtree_corrected_2026-08-05"
        ),
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    if len(set(args.arms)) != len(args.arms):
        raise SystemExit("--arms must not contain duplicates")
    output_root = args.out.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise SystemExit(f"refusing non-empty output directory: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    images = report_utils._discover_images(args.images)
    resolved_mask = None if args.mask is None else args.mask.resolve()
    if resolved_mask is not None and not resolved_mask.is_file():
        raise SystemExit(f"mask does not exist: {resolved_mask}")
    repository = report_helpers._repository_identity()
    command = shlex.join([sys.executable, *sys.argv])
    source_snapshots = _snapshot_sources(output_root)
    context = [
        *_load_context(args.hier005_report, "HIER-005 contextual"),
        *_load_context(args.hier006_report, "HIER-006 contextual"),
    ]
    report_helpers._write_json(output_root / "context.json", context)
    rows: list[dict[str, object]] = []
    trajectories: list[dict[str, object]] = []
    image_manifest = []
    raster_manifest = []
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
                "sha256": report_helpers._sha256(image_path),
            }
        )
        raster_manifest.append({"source_id": source_id, **raster_record})
        image_rows, image_trajectories = _run_image(
            image_path=image_path,
            image=image,
            mask=mask,
            raster_record=raster_record,
            source_id=source_id,
            args=args,
            output_root=output_root,
            repository=repository,
        )
        rows.extend(image_rows)
        trajectories.extend(image_trajectories)
    _write_tables(output_root, rows)
    curves = _write_curves(output_root, rows, trajectories)
    _write_index(output_root, rows, curves, context, command)
    protocol = {
        "arms": args.arms,
        "shared_base": True,
        "common": {
            key: value
            for key, value in asdict(_arm_config(args, args.arms[0])).items()
            if key not in {"selection_mode", "reconciliation_scope"}
        },
        "factor_axes": {
            arm: {
                "selection_mode": arm.split("__", maxsplit=1)[0],
                "reconciliation_scope": arm.split("__", maxsplit=1)[1],
            }
            for arm in args.arms
        },
    }
    manifest = {
        "schema": WORKFLOW_SCHEMA,
        "status": "diagnostic",
        "claim_ready": False,
        "task": "HIER-007",
        "method": METHOD,
        "command": command,
        "repository": repository,
        "variants": sorted({str(row["variant"]) for row in rows}),
        "arms": args.arms,
        "seeds": [0],
        "images": image_manifest,
        "evaluation_rasters": raster_manifest,
        "source_snapshot": source_snapshots,
        "protocol": protocol,
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": report_helpers._installed_version("torch"),
            "pillow": report_helpers._installed_version("Pillow"),
            "lpips": report_helpers._installed_version("lpips"),
        },
        "evidence_limits": [
            "single exposed-image dirty-source diagnostic without distinct prospective review",
            "single seed and CUDA additive gradients can vary through atomic accumulation",
            "frontier/event proxy bytes omit a complete grammar, indices, headers, quantization, entropy coding, and cold decoder",
            "lossless Observation Field NPZ is reference interchange, not compression",
            "native JPEG ratios compare a native-resolution source with a resized evaluation field",
            "HIER-005/HIER-006 rows are retained context, not jointly executed equal-work arms",
            "the diagnostic cannot support a novelty, general-quality, actual-rate, production, or default claim",
        ],
    }
    report_helpers._write_json(output_root / "manifest.json", manifest)
    print(f"wrote diagnostic report: {output_root / 'index.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
