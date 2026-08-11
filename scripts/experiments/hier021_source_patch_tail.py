#!/usr/bin/env python3
"""Run HIER-021's prospective low-coverage 7x7 RGB exception diagnostic."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from html import escape
import json
import math
import platform
from pathlib import Path
import shlex
import shutil
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
for candidate in (ROOT, ROOT / "src"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from scripts.experiments import hier005_pixel_contraction as report_utils  # noqa: E402
from scripts.experiments import hier015_geometry_escape as h15  # noqa: E402
from scripts.experiments import hier017_normalization_epsilon as h17  # noqa: E402
from scripts.experiments import hier018_counted_background as h18  # noqa: E402
from scripts.experiments import hier019_confidence_tail as h19  # noqa: E402
from scripts.experiments import hier020_sparse_pixel_safe_tail as h20  # noqa: E402
from structsplat.cli import save_error_heatmap, save_image  # noqa: E402
from structsplat.config import StructureTensorConfig  # noqa: E402
from structsplat.fit import _normalized_color_denominator  # noqa: E402
from structsplat.gaussians import GaussianField  # noqa: E402
from structsplat.init import build_field  # noqa: E402
from structsplat.pixel_contraction import contract_image  # noqa: E402
from structsplat.source_patch_tail import (  # noqa: E402
    SourcePatchConfig,
    SourcePatchPayload,
    apply_source_patch_payload,
    render_source_patch_payload,
    select_source_patch_tail,
)


REPORT_SCHEMA = "structsplat.hier021_source_patch_tail.diagnostic.v1"
DEVELOPMENT_BINDINGS = {
    "COCO_train2014_000000233566.jpg": (
        "064d1aa1f712c2c9850813b7745b0c77c57ab1aa0a5716574157dcfe90e5ec56"
    ),
    "COCO_train2014_000000455444.jpg": (
        "f1e87849fed2d66758df0fdb1970b4d3a21c013314adae154aa10e73565da340"
    ),
    "COCO_train2014_000000552149.jpg": (
        "c6d43328bede507e143a443711133df6312dd71566a39d5002014dd61f4c0c50"
    ),
    "COCO_train2014_000000439248.jpg": (
        "585f84581a42a57f6ba5bee526088c0e45b6694d20321a46fc3042a1a0628582"
    ),
}
SELECTION_DIGESTS = {
    "COCO_train2014_000000233566.jpg": (
        "000011281017d254c9c240e2c98a24bc0dc31e0c6506b49710f2d7a5a32424d3"
    ),
    "COCO_train2014_000000455444.jpg": (
        "0003568a57cd479208dcbdb645ac09f259d577df98745860df05d7b064d9b50a"
    ),
    "COCO_train2014_000000552149.jpg": (
        "0004a29dcfd3a3e88dbca2c44d4cebb2a9d4a4226d367cae0bd3f3831c8736e5"
    ),
    "COCO_train2014_000000439248.jpg": (
        "00055b45e8344269f5b0a00c3c556114f2828f4ac4e6cdd664ef265c4dff450f"
    ),
}
CONTROL_ARM = h20.CONTROL_ARM
DIRECT_ARM = h20.DIRECT_ARM
CANDIDATE_MODE = "low_coverage_rgb_patch_spt1"
BASELINE_MODE = h20.BASELINE_MODE
METRIC_KEYS = h20.METRIC_KEYS


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path)
    parser.add_argument("--review-from", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--visual-disposition", choices=("pending", "pass", "fail"), default="pending"
    )
    parser.add_argument("--target-gaussians", type=int, default=7000)
    parser.add_argument("--max-side", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--direct-fit-steps", type=int, default=750)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--additive-renderer", default="cuda_additive")
    parser.add_argument("--direct-renderer", default="cuda")
    parser.add_argument("--render-chunk", type=int, default=256)
    parser.add_argument("--lpips", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--patch-radius", type=int, default=3)
    parser.add_argument("--coverage-threshold", type=float, default=1e-8)
    parser.add_argument("--error-scale", type=float, default=4.0)
    parser.set_defaults(phase="development")
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    frozen = {
        "target_gaussians": 7000,
        "max_side": 512,
        "seed": 0,
        "direct_fit_steps": 750,
        "device": "cuda",
        "additive_renderer": "cuda_additive",
        "direct_renderer": "cuda",
        "render_chunk": 256,
        "lpips": True,
        "patch_radius": 3,
        "coverage_threshold": 1e-8,
    }
    for name, expected in frozen.items():
        if getattr(args, name) != expected:
            raise SystemExit(
                f"frozen HIER-021 protocol requires --{name.replace('_', '-')} {expected}"
            )
    if not math.isfinite(args.error_scale) or args.error_scale <= 0.0:
        raise SystemExit("--error-scale must be finite and positive")
    if (args.images is None) == (args.review_from is None):
        raise SystemExit("pass exactly one of --images or --review-from")
    if args.images is not None and args.visual_disposition != "pending":
        raise SystemExit("record the visual verdict only with --review-from")


def _configs(args: argparse.Namespace):
    init_config, _, fit_config = h18._configs(args)
    patch_config = SourcePatchConfig(
        radius=args.patch_radius,
        coverage_threshold=args.coverage_threshold,
    )
    return init_config, fit_config, patch_config


def _snapshot_sources(output_root: Path) -> list[dict[str, object]]:
    paths = (
        Path(__file__).resolve(),
        ROOT / "src" / "structsplat" / "source_patch_tail.py",
        ROOT / "src" / "structsplat" / "tail_recovery.py",
        ROOT / "scripts" / "experiments" / "hier020_sparse_pixel_safe_tail.py",
        ROOT / "tasks" / "HIER-021-low-coverage-rgb-patch-tail.md",
    )
    records: list[dict[str, object]] = []
    for source in paths:
        relative = source.relative_to(ROOT)
        destination = output_root / "source_snapshot" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        records.append(
            {
                "repository_path": str(relative),
                "snapshot_path": str(destination.relative_to(output_root)),
                "bytes": destination.stat().st_size,
                "sha256": report_utils._sha256(destination),
            }
        )
    return records


def _metric_subset(row: dict[str, object]) -> dict[str, float]:
    return {key: float(row[key]) for key in METRIC_KEYS}


def _selection(
    baseline: dict[str, float],
    candidate: dict[str, float],
    *,
    payload_count: int,
    finite: bool,
    outside_identity_max_abs: float,
    baseline_parity_max_abs: float,
    repeated_parity_max_abs: float,
    payload_roundtrip_exact: bool,
    pointwise_raw_delta_max: float,
    pointwise_display_delta_max: float,
) -> tuple[str, dict[str, bool], dict[str, float]]:
    ratio = candidate["masked_mse"] / max(baseline["masked_mse"], 1e-30)
    deltas = {
        "mse_ratio": ratio,
        "psnr_delta_db": candidate["psnr_db"] - baseline["psnr_db"],
        "ms_ssim_delta": candidate["ms_ssim"] - baseline["ms_ssim"],
        "lpips_delta": candidate["lpips"] - baseline["lpips"],
        "pixel_max_delta": candidate["artifact_pixel_rmse_max"]
        - baseline["artifact_pixel_rmse_max"],
        "patch7_max_delta": candidate["artifact_patch_rmse_max_7"]
        - baseline["artifact_patch_rmse_max_7"],
    }
    clauses = {
        "candidate_finite": finite,
        "outside_payload_bit_exact": outside_identity_max_abs == 0.0,
        "baseline_cold_parity_le_2e_5": baseline_parity_max_abs <= 2e-5,
        "candidate_repeated_parity_le_2e_5": repeated_parity_max_abs <= 2e-5,
        "payload_roundtrip_exact": payload_roundtrip_exact,
        "pointwise_raw_strict": payload_count == 0 or pointwise_raw_delta_max < 0.0,
        "pointwise_display_strict": payload_count == 0
        or pointwise_display_delta_max < 0.0,
        "mse_noninferior": ratio <= 1.0 + 1e-8,
        "pixel_max_noninferior": deltas["pixel_max_delta"] <= 1e-12,
        "patch7_max_noninferior": deltas["patch7_max_delta"] <= 1e-12,
        "ms_ssim_noninferior": deltas["ms_ssim_delta"] >= -1e-7,
        "lpips_noninferior": deltas["lpips_delta"] <= 1e-7,
    }
    material = (
        ratio < 1.0 - 1e-8
        or deltas["pixel_max_delta"] < -1e-12
        or deltas["patch7_max_delta"] < -1e-12
    )
    selected = (
        CANDIDATE_MODE
        if payload_count > 0 and all(clauses.values()) and material
        else BASELINE_MODE
    )
    return selected, clauses, deltas


def _augment_direct_row(
    *,
    output_root: Path,
    row: dict[str, object],
    image: np.ndarray,
    mask: np.ndarray,
    baseline_render: np.ndarray,
    fit_config,
    patch_config: SourcePatchConfig,
    args: argparse.Namespace,
) -> dict[str, object]:
    import torch

    from structsplat.fit import _render

    artifact_dir = output_root / str(row["artifact_dir"])
    field_path = artifact_dir / "field.gaussian.npz"
    field_hash_before = report_utils._sha256(field_path)
    field_bytes = field_path.stat().st_size
    height, width = image.shape[:2]
    field = GaussianField.load(str(field_path), device=args.device)
    target = torch.as_tensor(image, device=args.device, dtype=torch.float32).contiguous()

    torch.cuda.synchronize()
    baseline_started = time.perf_counter()
    ordinary = _render(field, fit_config, height, width)
    torch.cuda.synchronize()
    ordinary_seconds = time.perf_counter() - baseline_started

    torch.cuda.synchronize()
    encoder_started = time.perf_counter()
    denominator = _normalized_color_denominator(field, fit_config, height, width).reshape(
        height, width
    )
    result = select_source_patch_tail(
        ordinary,
        denominator,
        target,
        fit_config.normalization_eps,
        patch_config,
    )
    encoded = result.payload.to_bytes()
    torch.cuda.synchronize()
    candidate_construction_seconds = time.perf_counter() - encoder_started
    candidate_path = artifact_dir / "candidate.spt1"
    candidate_path.write_bytes(encoded)
    parsed = SourcePatchPayload.from_bytes(candidate_path.read_bytes())
    payload_roundtrip_exact = parsed.to_bytes() == encoded
    cold_candidate = apply_source_patch_payload(ordinary, parsed)

    repeated_field = GaussianField.load(str(field_path), device=args.device)
    torch.cuda.synchronize()
    repeated_started = time.perf_counter()
    repeated = render_source_patch_payload(
        repeated_field, fit_config, height, width, parsed
    )
    torch.cuda.synchronize()
    repeated_seconds = time.perf_counter() - repeated_started

    ordinary_np = ordinary.detach().cpu().numpy().astype(np.float32, copy=False)
    candidate = cold_candidate.detach().cpu().numpy().astype(np.float32, copy=False)
    repeated_np = repeated.detach().cpu().numpy().astype(np.float32, copy=False)
    seed = result.seed_mask.detach().cpu().numpy().astype(bool, copy=False)
    expanded = result.expanded_mask.detach().cpu().numpy().astype(bool, copy=False)
    selected = result.selected_mask.detach().cpu().numpy().astype(bool, copy=False)
    denominator_np = denominator.detach().cpu().numpy().astype(np.float64)
    baseline_parity = max(
        float(np.max(np.abs(ordinary_np - baseline_render))),
        float(row["maintained_render_parity_max_abs"]),
    )
    repeated_parity = float(np.max(np.abs(repeated_np - candidate)))
    outside = np.abs(candidate - ordinary_np)[~selected]
    outside_identity = float(outside.max()) if outside.size else 0.0

    metric_started = time.perf_counter()
    candidate_metrics = report_utils._metric_values(
        candidate, image, mask, device=args.device, compute_lpips=args.lpips
    )
    metric_seconds = time.perf_counter() - metric_started
    encoder_rdo_seconds = candidate_construction_seconds + metric_seconds
    candidate_subset = _metric_subset(candidate_metrics)
    baseline_subset = _metric_subset(row)
    finite = bool(
        np.isfinite(candidate).all()
        and np.isfinite(denominator_np).all()
        and all(math.isfinite(value) for value in candidate_subset.values())
    )
    selected_mode, clauses, deltas = _selection(
        baseline_subset,
        candidate_subset,
        payload_count=result.selected_count,
        finite=finite,
        outside_identity_max_abs=outside_identity,
        baseline_parity_max_abs=baseline_parity,
        repeated_parity_max_abs=repeated_parity,
        payload_roundtrip_exact=payload_roundtrip_exact,
        pointwise_raw_delta_max=result.pointwise_raw_sse_delta_max,
        pointwise_display_delta_max=result.pointwise_display_sse_delta_max,
    )
    selected_metrics = (
        candidate_subset if selected_mode == CANDIDATE_MODE else baseline_subset
    )
    selected_bytes = len(encoded) if selected_mode == CANDIDATE_MODE else 0
    selected_path: str | None = None
    selected_hash: str | None = None
    if selected_mode == CANDIDATE_MODE:
        path = artifact_dir / "tail.spt1"
        path.write_bytes(encoded)
        selected_path = str(path.relative_to(output_root))
        selected_hash = report_utils._sha256(path)

    candidate_dir = artifact_dir / "source_patch_candidate"
    candidate_dir.mkdir(parents=True, exist_ok=False)
    h15._save_visuals(
        candidate_dir, image, candidate, ordinary_np, mask, args.error_scale
    )
    save_error_heatmap(
        str(candidate_dir / "candidate_delta.png"),
        candidate - ordinary_np,
        scale=args.error_scale,
    )
    overlay = np.clip(ordinary_np, 0.0, 1.0).copy()
    overlay[selected] = 0.35 * overlay[selected] + 0.65 * np.asarray(
        [1.0, 0.0, 0.0], dtype=np.float32
    )
    save_image(str(candidate_dir / "selected_sites.png"), overlay)
    with np.load(artifact_dir / "analysis.npz", allow_pickle=False) as analysis:
        common_bounds = tuple(int(value) for value in analysis["crop_bounds"])
    for name, value in (
        ("source_common_crop.png", image),
        ("baseline_common_crop.png", ordinary_np),
        ("candidate_common_crop.png", candidate),
    ):
        h15.viz_utils._save_crop(candidate_dir / name, value, common_bounds)
    np.savez_compressed(
        candidate_dir / "patch_analysis.npz",
        seed_mask=seed,
        expanded_mask=expanded,
        selected_mask=selected,
        denominator=denominator_np.astype(np.float32),
        candidate_delta=(candidate - ordinary_np).astype(np.float32),
        common_crop_bounds=np.asarray(common_bounds, dtype=np.int32),
    )
    telemetry = {
        "schema": REPORT_SCHEMA,
        "status": "diagnostic",
        "semantic_family": "low_coverage_rgb8_patch_spt1_v1",
        "coverage_threshold": result.coverage_threshold,
        "patch_radius": result.radius,
        "seed_count": result.seed_count,
        "expanded_count": result.expanded_count,
        "selected_count": result.selected_count,
        "pointwise_raw_sse_delta_max": result.pointwise_raw_sse_delta_max,
        "pointwise_display_sse_delta_max": result.pointwise_display_sse_delta_max,
        "outside_identity_max_abs": outside_identity,
        "baseline_cold_parity_max_abs": baseline_parity,
        "candidate_repeated_parity_max_abs": repeated_parity,
        "payload_roundtrip_exact": payload_roundtrip_exact,
        "candidate_payload_bytes": len(encoded),
        "candidate_payload_sha256": report_utils._sha256(candidate_path),
        "selected_payload_bytes": selected_bytes,
        "selected_payload_path": selected_path,
        "selected_payload_sha256": selected_hash,
        "candidate_metrics": candidate_subset,
        "metric_deltas_vs_baseline": deltas,
        "selection_clauses": clauses,
        "selected_mode": selected_mode,
        "selected_metrics": selected_metrics,
        "ordinary_render_seconds": ordinary_seconds,
        "candidate_construction_seconds": candidate_construction_seconds,
        "encoder_rdo_seconds": encoder_rdo_seconds,
        "selected_decode_seconds": (
            repeated_seconds if selected_mode == CANDIDATE_MODE else ordinary_seconds
        ),
        "candidate_metric_seconds": metric_seconds,
        "field_file_sha256_before": field_hash_before,
        "field_file_sha256_after": report_utils._sha256(field_path),
    }
    report_utils._write_json(candidate_dir / "patch_recovery.json", telemetry)
    selected_decode_seconds = float(telemetry["selected_decode_seconds"])
    row.update(
        {
            "spt1_semantic_family": telemetry["semantic_family"],
            "spt1_coverage_threshold": result.coverage_threshold,
            "spt1_patch_radius": result.radius,
            "spt1_seed_count": result.seed_count,
            "spt1_expanded_count": result.expanded_count,
            "spt1_selected_count": result.selected_count,
            "spt1_pointwise_raw_sse_delta_max": result.pointwise_raw_sse_delta_max,
            "spt1_pointwise_display_sse_delta_max": result.pointwise_display_sse_delta_max,
            "spt1_outside_identity_max_abs": outside_identity,
            "spt1_baseline_cold_parity_max_abs": baseline_parity,
            "spt1_candidate_repeated_parity_max_abs": repeated_parity,
            "spt1_payload_roundtrip_exact": payload_roundtrip_exact,
            "spt1_candidate_finite": finite,
            "spt1_candidate_metrics": candidate_subset,
            "spt1_metric_deltas_vs_baseline": deltas,
            "spt1_selection_clauses": clauses,
            "spt1_selected_mode": selected_mode,
            "spt1_selected_metrics": selected_metrics,
            "spt1_candidate_payload_bytes": len(encoded),
            "spt1_candidate_payload_sha256": telemetry["candidate_payload_sha256"],
            "spt1_selected_payload_bytes": selected_bytes,
            "spt1_selected_payload_path": selected_path,
            "spt1_selected_payload_sha256": selected_hash,
            "spt1_field_plus_candidate_reference_bytes": field_bytes + len(encoded),
            "spt1_field_plus_selected_reference_bytes": field_bytes + selected_bytes,
            "spt1_ordinary_render_seconds": ordinary_seconds,
            "spt1_candidate_construction_seconds": candidate_construction_seconds,
            "spt1_encoder_seconds": encoder_rdo_seconds,
            "spt1_selected_decode_seconds": selected_decode_seconds,
            "spt1_candidate_metric_seconds": metric_seconds,
            "spt1_decode_time_ratio": selected_decode_seconds
            / max(ordinary_seconds, 1e-12),
            "spt1_pipeline_time_ratio": (
                (float(row["pipeline_algorithm_seconds"]) + encoder_rdo_seconds)
                / max(float(row["pipeline_algorithm_seconds"]), 1e-12)
            ),
            "spt1_field_file_sha256_before": field_hash_before,
            "spt1_field_file_sha256_after": report_utils._sha256(field_path),
        }
    )
    report_utils._write_json(artifact_dir / "row.json", row)
    return row


def _pairs(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    controls = {str(row["image"]): row for row in rows if row["arm"] == CONTROL_ARM}
    pairs: list[dict[str, object]] = []
    for row in rows:
        if row["arm"] != DIRECT_ARM or str(row["image"]) not in controls:
            continue
        control = controls[str(row["image"])]
        selected = row["spt1_selected_metrics"]
        pairs.append(
            {
                "image": row["image"],
                "selected_mode": row["spt1_selected_mode"],
                "selected_count": row["spt1_selected_count"],
                "psnr_delta_db": float(selected["psnr_db"]) - float(control["psnr_db"]),
                "mse_ratio": float(selected["masked_mse"])
                / max(float(control["masked_mse"]), 1e-30),
                "ms_ssim_delta": float(selected["ms_ssim"])
                - float(control["ms_ssim"]),
                "lpips_delta": float(selected["lpips"]) - float(control["lpips"]),
                "pixel_max_delta": float(selected["artifact_pixel_rmse_max"])
                - float(control["artifact_pixel_rmse_max"]),
                "patch7_max_delta": float(selected["artifact_patch_rmse_max_7"])
                - float(control["artifact_patch_rmse_max_7"]),
                "baseline_pixel_max_delta": float(row["artifact_pixel_rmse_max"])
                - float(control["artifact_pixel_rmse_max"]),
                "baseline_patch7_max_delta": float(row["artifact_patch_rmse_max_7"])
                - float(control["artifact_patch_rmse_max_7"]),
            }
        )
    return pairs


def _decision(
    rows: list[dict[str, object]],
    attempts: list[dict[str, object]],
    *,
    visual_disposition: str,
) -> dict[str, object]:
    direct = [row for row in rows if row["arm"] == DIRECT_ARM]
    pairs = _pairs(rows)
    failures = [
        pair
        for pair in pairs
        if float(pair["baseline_pixel_max_delta"]) > 1e-12
        or float(pair["baseline_patch7_max_delta"]) > 1e-12
    ]
    gates = {
        "complete_four_direct_rows": len(direct) == 4,
        "complete_four_h005_pairs": len(pairs) == 4,
        "complete_attempt_ledger": len(attempts) == 8,
        "zero_failures": all(record.get("status") == "ok" for record in attempts),
        "all_exact_count": all(int(row["n_gaussians"]) == 7000 for row in direct),
        "all_field_bytes_unchanged": all(
            row["spt1_field_file_sha256_before"] == row["spt1_field_file_sha256_after"]
            for row in direct
        ),
        "all_payloads_canonical": all(
            bool(row["spt1_payload_roundtrip_exact"]) for row in direct
        ),
        "all_candidates_finite": all(bool(row["spt1_candidate_finite"]) for row in direct),
        "all_outside_payload_bit_exact": all(
            float(row["spt1_outside_identity_max_abs"]) == 0.0 for row in direct
        ),
        "all_parity_le_2e_5": all(
            float(row["spt1_baseline_cold_parity_max_abs"]) <= 2e-5
            and float(row["spt1_candidate_repeated_parity_max_abs"]) <= 2e-5
            for row in direct
        ),
        "all_selected_transactions_safe": all(
            row["spt1_selected_mode"] == BASELINE_MODE
            or all(bool(value) for value in row["spt1_selection_clauses"].values())
            for row in direct
        ),
        "all_psnr_gain_vs_h005_ge_2_db": all(
            float(pair["psnr_delta_db"]) >= 2.0 for pair in pairs
        ),
        "all_mse_noninferior_vs_h005": all(
            float(pair["mse_ratio"]) <= 1.0 + 1e-8 for pair in pairs
        ),
        "all_ms_ssim_noninferior_vs_h005": all(
            float(pair["ms_ssim_delta"]) >= -1e-7 for pair in pairs
        ),
        "all_lpips_noninferior_vs_h005": all(
            float(pair["lpips_delta"]) <= 1e-7 for pair in pairs
        ),
        "all_pixel_max_noninferior_vs_h005": all(
            float(pair["pixel_max_delta"]) <= 1e-12 for pair in pairs
        ),
        "all_patch7_max_noninferior_vs_h005": all(
            float(pair["patch7_max_delta"]) <= 1e-12 for pair in pairs
        ),
        "all_baseline_local_failures_repaired": all(
            float(pair["pixel_max_delta"]) <= 1e-12
            and float(pair["patch7_max_delta"]) <= 1e-12
            for pair in failures
        ),
        "median_pipeline_time_ratio_le_1_25": (
            float(np.median([row["spt1_pipeline_time_ratio"] for row in direct])) <= 1.25
            if direct
            else False
        ),
        "median_selected_decode_ratio_le_2": (
            float(np.median([row["spt1_decode_time_ratio"] for row in direct])) <= 2.0
            if direct
            else False
        ),
    }
    numeric_pass = all(gates.values())
    return {
        "schema": REPORT_SCHEMA,
        "status": "diagnostic",
        "phase": "development",
        "gates": gates,
        "selected_vs_h005_pairs": pairs,
        "baseline_local_failures": failures,
        "baseline_local_failure_count": len(failures),
        "selected_image_count": sum(
            row["spt1_selected_mode"] == CANDIDATE_MODE for row in direct
        ),
        "selected_pixel_count": sum(
            int(row["spt1_selected_count"])
            for row in direct
            if row["spt1_selected_mode"] == CANDIDATE_MODE
        ),
        "candidate_side_bytes": sum(
            int(row["spt1_candidate_payload_bytes"]) for row in direct
        ),
        "selected_side_bytes": sum(
            int(row["spt1_selected_payload_bytes"]) for row in direct
        ),
        "attempt_count": len(attempts),
        "failure_count": sum(record.get("status") != "ok" for record in attempts),
        "numeric_bank_pass": numeric_pass,
        "visual_review_required": True,
        "visual_disposition": visual_disposition,
        "bounded_bank_pass": numeric_pass and visual_disposition == "pass",
        "numeric_candidates": [CANDIDATE_MODE] if numeric_pass else [],
        "interpretation": (
            "Fresh SPT1 screen passed numeric and recorded visual gates."
            if numeric_pass and visual_disposition == "pass"
            else (
                "Fresh SPT1 screen is numerically safe and awaits native visual review."
                if numeric_pass
                else "The frozen SPT1 screen failed; do not replay or retune."
            )
        ),
    }


def _write_report(
    output_root: Path,
    rows: list[dict[str, object]],
    decision: dict[str, object],
    command: str,
) -> None:
    table: list[str] = []
    cards: list[str] = []
    for row in rows:
        artifact = escape(str(row["artifact_dir"]))
        if row["arm"] == CONTROL_ARM:
            table.append(
                "<tr>"
                f"<td>{escape(str(row['image']))}</td><td>{CONTROL_ARM}</td>"
                f"<td>{float(row['psnr_db']):.3f}</td><td>{float(row['ms_ssim']):.5f}</td>"
                f"<td>{float(row['lpips']):.5f}</td>"
                f"<td>{float(row['artifact_pixel_rmse_max']):.4f}</td>"
                f"<td>{float(row['artifact_patch_rmse_max_7']):.4f}</td>"
                "<td>—</td><td>—</td><td>—</td>"
                f"<td><a href='{artifact}/reconstruction.png'>full</a> · "
                f"<a href='{artifact}/reconstruction_crop.png'>crop</a></td></tr>"
            )
        else:
            candidate = row["spt1_candidate_metrics"]
            table.append(
                "<tr>"
                f"<td>{escape(str(row['image']))}</td><td>{DIRECT_ARM}</td>"
                f"<td>{float(row['psnr_db']):.3f} / {float(candidate['psnr_db']):.3f}</td>"
                f"<td>{float(row['ms_ssim']):.5f} / {float(candidate['ms_ssim']):.5f}</td>"
                f"<td>{float(row['lpips']):.5f} / {float(candidate['lpips']):.5f}</td>"
                f"<td>{float(row['artifact_pixel_rmse_max']):.4f} / "
                f"{float(candidate['artifact_pixel_rmse_max']):.4f}</td>"
                f"<td>{float(row['artifact_patch_rmse_max_7']):.4f} / "
                f"{float(candidate['artifact_patch_rmse_max_7']):.4f}</td>"
                f"<td>{int(row['spt1_seed_count'])}</td>"
                f"<td>{int(row['spt1_selected_count'])}</td>"
                f"<td>{escape(str(row['spt1_selected_mode']))}</td>"
                f"<td><a href='{artifact}/reconstruction.png'>base</a> · "
                f"<a href='{artifact}/source_patch_candidate/reconstruction.png'>patch</a> · "
                f"<a href='{artifact}/source_patch_candidate/candidate_common_crop.png'>crop</a> · "
                f"<a href='{artifact}/source_patch_candidate/selected_sites.png'>sites</a> · "
                f"<a href='{artifact}/candidate.spt1'>SPT1</a></td></tr>"
            )
        cards.append(
            f"<section><h3>{escape(str(row['image']))} — {escape(str(row['arm']))}</h3>"
            f"<a href='{artifact}/source.png'><img src='{artifact}/source.png'></a>"
            f"<a href='{artifact}/reconstruction.png'><img src='{artifact}/reconstruction.png'></a>"
            f"<a href='{artifact}/error.png'><img src='{artifact}/error.png'></a>"
            f"<a href='{artifact}/reconstruction_crop.png'>"
            f"<img src='{artifact}/reconstruction_crop.png'></a>"
            + (
                f"<a href='{artifact}/source_patch_candidate/reconstruction.png'>"
                f"<img src='{artifact}/source_patch_candidate/reconstruction.png'></a>"
                f"<a href='{artifact}/source_patch_candidate/candidate_delta.png'>"
                f"<img src='{artifact}/source_patch_candidate/candidate_delta.png'></a>"
                f"<a href='{artifact}/source_patch_candidate/selected_sites.png'>"
                f"<img src='{artifact}/source_patch_candidate/selected_sites.png'></a>"
                if row["arm"] == DIRECT_ARM
                else ""
            )
            + "</section>"
        )
    html = f"""<!doctype html><html><head><meta charset='utf-8'><title>HIER-021</title>
<style>body{{font-family:system-ui;margin:2rem;max-width:1700px}}table{{border-collapse:collapse}}
th,td{{border:1px solid #bbb;padding:.35rem}}img{{max-width:300px;max-height:240px;margin:.25rem}}
section{{border-top:1px solid #aaa;margin-top:1rem}}pre,code{{white-space:pre-wrap}}</style>
</head><body><h1>HIER-021 low-coverage RGB exception patches</h1>
<p>Dirty-source diagnostic. Direct cells show <strong>ordinary / SPT1 candidate</strong>.
SPT1 is an explicit source-RGB residual layer, not part of the 7,000-Gaussian field.</p>
<p><code>{escape(command)}</code></p><p><a href='config.json'>config</a> ·
<a href='decision.json'>decision</a> · <a href='metrics.json'>JSON</a> ·
<a href='metrics.jsonl'>JSONL</a> · <a href='metrics.csv'>CSV</a> ·
<a href='attempts.json'>attempts</a> · <a href='manifest.json'>manifest</a></p>
<h2>Decision</h2><pre>{escape(json.dumps(decision, indent=2, sort_keys=True))}</pre>
<h2>Cells</h2><table><tr><th>image</th><th>arm</th><th>PSNR</th><th>MS-SSIM</th>
<th>LPIPS</th><th>pixel max</th><th>7x7 max</th><th>seeds</th><th>records</th>
<th>selected</th><th>visuals/data</th></tr>{''.join(table)}</table>
<h2>Visual audit</h2>{''.join(cards)}</body></html>"""
    (output_root / "index.html").write_text(html, encoding="utf-8")


def _review(args: argparse.Namespace, output_root: Path, command: str) -> bool:
    if args.review_from is None:
        return False
    source_root = args.review_from.resolve()
    required = [source_root / name for name in ("metrics.json", "attempts.json", "decision.json")]
    if not source_root.is_dir() or not all(path.is_file() for path in required):
        raise SystemExit("--review-from is missing fresh ledgers")
    rows = json.loads(required[0].read_text(encoding="utf-8"))["rows"]
    attempts = json.loads(required[1].read_text(encoding="utf-8"))["attempts"]
    prior = json.loads(required[2].read_text(encoding="utf-8"))
    if not prior.get("numeric_bank_pass") or prior.get("visual_disposition") != "pending":
        raise SystemExit("--review-from is not a pending numeric-pass HIER-021 run")
    shutil.copytree(source_root, output_root)
    decision = _decision(rows, attempts, visual_disposition=args.visual_disposition)
    decision.update(
        {
            "reviewed_from": str(source_root),
            "review_only": True,
            "cell_computation_rerun": False,
            "source_decision_sha256": report_utils._sha256(required[2]),
        }
    )
    report_utils._write_json(output_root / "decision.json", decision)
    config = json.loads((output_root / "config.json").read_text(encoding="utf-8"))
    config["command"] = command
    config["visual_review"] = {
        "source_path": str(source_root),
        "source_decision_sha256": report_utils._sha256(required[2]),
        "disposition": args.visual_disposition,
        "cell_computation_rerun": False,
    }
    report_utils._write_json(output_root / "config.json", config)
    report_utils._write_json(
        output_root / "visual_review.json",
        {"schema": REPORT_SCHEMA, "status": "diagnostic", **config["visual_review"]},
    )
    _write_report(output_root, rows, decision, command)
    h15._write_manifest(output_root, schema=REPORT_SCHEMA)
    print(json.dumps(decision, indent=2, sort_keys=True))
    return True


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    _validate_args(args)
    output_root = args.out.resolve()
    command = shlex.join(
        [sys.executable, str(Path(__file__).resolve()), *(argv or sys.argv[1:])]
    )
    if _review(args, output_root, command):
        return 0
    assert args.images is not None
    images = h19._bound_paths(args.images, DEVELOPMENT_BINDINGS)

    import torch

    output_root.mkdir(parents=True, exist_ok=False)
    contraction_config = h15._contraction_config(args)
    init_config, fit_config, patch_config = _configs(args)
    config = {
        "schema": REPORT_SCHEMA,
        "status": "diagnostic",
        "phase": "development",
        "command": command,
        "args": vars(args),
        "development_selection_digests": SELECTION_DIGESTS,
        "sources": [
            {"path": str(path), "sha256": report_utils._sha256(path)} for path in images
        ],
        "contraction": asdict(contraction_config),
        "direct_init": asdict(init_config),
        "fit": asdict(fit_config),
        "source_patch": asdict(patch_config),
        "spt1": {
            "magic": "SPT1",
            "header_bytes": 16,
            "record_bytes": 7,
            "records": "strictly increasing raster-flat uint32le plus RGB8",
        },
        "source_snapshots": _snapshot_sources(output_root),
        "git": report_utils._git_record(),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        },
        "limitations": [
            "Dirty-source one-seed diagnostic without distinct review.",
            "SPT1 is an extra source-derived residual layer, not a pure-Gaussian repair.",
            "NPZ plus SPT1 is reference accounting, not a production codec rate.",
            "Target-known mode selection is encoder-side RDO.",
        ],
    }
    report_utils._write_json(output_root / "config.json", config)
    rows: list[dict[str, object]] = []
    attempts: list[dict[str, object]] = []

    def record(
        image_path: Path,
        arm: str,
        started: float,
        error: Exception | None = None,
    ) -> None:
        item: dict[str, object] = {
            "image": image_path.stem,
            "arm": arm,
            "status": "ok" if error is None else "error",
            "elapsed_seconds": time.perf_counter() - started,
        }
        if error is not None:
            item["error"] = f"{type(error).__name__}: {error}"[:1000]
        attempts.append(item)
        report_utils._write_json(
            output_root / "attempts.json",
            {"schema": REPORT_SCHEMA, "status": "diagnostic", "attempts": attempts},
        )

    run_started = time.perf_counter()
    for image_path in images:
        image, loaded_mask, raster = report_utils._load_evaluation_raster(
            image_path, None, max_side=args.max_side, mask_threshold=0.5
        )
        if loaded_mask is not None:
            raise RuntimeError("HIER-021 requires a generated full-frame mask")
        mask = np.ones(image.shape[:2], dtype=bool)
        control_reconstruction = image

        started = time.perf_counter()
        try:
            h17._seed_everything(args.seed)
            torch.cuda.reset_peak_memory_stats()
            control = contract_image(image, contraction_config, mask=mask)
            seconds = time.perf_counter() - started
            row = h15._write_observation_cell(
                output_root=output_root,
                image_path=image_path,
                image=image,
                mask=mask,
                raster=raster,
                arm=CONTROL_ARM,
                field=control.field,
                control_field=control.field,
                control_reconstruction=control.reconstruction,
                expected=control.reconstruction,
                contraction_seconds=seconds,
                method_seconds=0.0,
                projection=None,
                alternating=None,
                peak_cuda_bytes=int(torch.cuda.max_memory_allocated()),
                args=args,
                schema=REPORT_SCHEMA,
            )
            rows.append(row)
            control_reconstruction = control.reconstruction
            record(image_path, CONTROL_ARM, started)
        except Exception as exc:
            record(image_path, CONTROL_ARM, started, exc)

        started = time.perf_counter()
        try:
            h17._seed_everything(args.seed)
            init_started = time.perf_counter()
            initial = build_field(
                image, init_config, StructureTensorConfig(), device=args.device
            )
            init_seconds = time.perf_counter() - init_started
            init_hash = h15._gaussian_content_hash(initial)
            field, fit_result, peak = h17._run_fit(initial, image, fit_config)
            row, baseline_render = h17._write_cell(
                output_root=output_root,
                image_path=image_path,
                image=image,
                mask=mask,
                raster=raster,
                arm=DIRECT_ARM,
                field=field,
                fit_result=fit_result,
                init_seconds=init_seconds,
                control_reconstruction=control_reconstruction,
                peak_cuda_bytes=peak,
                fit_config=fit_config,
                init_hash=init_hash,
                args=args,
                extra_row={
                    "spt1_is_explicit_source_rgb_residual": True,
                    "tail_added_gaussians": 0,
                    "tail_added_learned_parameters": 0,
                },
                schema=REPORT_SCHEMA,
            )
            row = _augment_direct_row(
                output_root=output_root,
                row=row,
                image=image,
                mask=mask,
                baseline_render=baseline_render,
                fit_config=fit_config,
                patch_config=patch_config,
                args=args,
            )
            rows.append(row)
            record(image_path, DIRECT_ARM, started)
        except Exception as exc:
            record(image_path, DIRECT_ARM, started, exc)
        h15._write_tables(output_root, rows, schema=REPORT_SCHEMA)

    decision = _decision(rows, attempts, visual_disposition="pending")
    decision["elapsed_seconds"] = time.perf_counter() - run_started
    report_utils._write_json(output_root / "decision.json", decision)
    h15._write_tables(output_root, rows, schema=REPORT_SCHEMA)
    _write_report(output_root, rows, decision, command)
    h15._write_manifest(output_root, schema=REPORT_SCHEMA)
    print(json.dumps(decision, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
