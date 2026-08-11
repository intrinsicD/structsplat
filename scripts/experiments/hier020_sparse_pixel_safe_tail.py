#!/usr/bin/env python3
"""Run HIER-020's source-bound sparse pixel-safe tail diagnostic."""
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
from scripts.experiments import hier013_global_projection_development as h13  # noqa: E402
from scripts.experiments import hier015_geometry_escape as h15  # noqa: E402
from scripts.experiments import hier017_normalization_epsilon as h17  # noqa: E402
from scripts.experiments import hier019_confidence_tail as h19  # noqa: E402
from structsplat.cli import save_error_heatmap, save_image  # noqa: E402
from structsplat.config import StructureTensorConfig  # noqa: E402
from structsplat.gaussians import GaussianField  # noqa: E402
from structsplat.init import build_field  # noqa: E402
from structsplat.pixel_contraction import contract_image  # noqa: E402
from structsplat.tail_recovery import (  # noqa: E402
    ConfidenceTailConfig,
    SparseTailPayload,
    apply_sparse_tail_payload,
    render_confidence_gated_self_prior,
    select_pixel_safe_tail,
)


REPORT_SCHEMA = "structsplat.hier020_sparse_pixel_safe_tail.diagnostic.v1"
DEVELOPMENT_BINDINGS = {
    "COCO_train2014_000000046728.jpg": (
        "72b147fd1ccf3e29e0e85ac556447009901841a98755bc0caee1d35d7de9d07c"
    ),
    "COCO_train2014_000000036289.jpg": (
        "bd1d848a1d12ab05546b36d46be4090fb7dd28acea7f4c759ffb2b818faac8bb"
    ),
    "COCO_train2014_000000466403.jpg": (
        "e3be2be68738ade64381301532276fef62d513684cf3e7fcdc2a94572d30c980"
    ),
    "COCO_train2014_000000072902.jpg": (
        "118efde8863cbab7a350ff3d6f4cc3ce73130f5c85c797ba40271f1b558bb81d"
    ),
}
SELECTION_DIGESTS = {
    "COCO_train2014_000000046728.jpg": (
        "00000931c0b426b1e276a951bc8aa1c50203aa2a50411c9ee2072d9aa1e6992e"
    ),
    "COCO_train2014_000000036289.jpg": (
        "0001a21eccc35740069a5b8f008c77f0f91df5d09eacbb51b8d40308f254cd54"
    ),
    "COCO_train2014_000000466403.jpg": (
        "00022135883480d3bd755a22b3d08b6070940afe685d1a3d5d5661f6d6044d38"
    ),
    "COCO_train2014_000000072902.jpg": (
        "00029855fe2d51bc41b0eee30f1e7abbbe0d36e4a55b8f3b06b8aec4ae32db65"
    ),
}
CONTROL_ARM = "h005_control"
DIRECT_ARM = "direct_no_recovery"
CANDIDATE_MODE = "sparse_pixel_safe_tail_sst1"
BASELINE_MODE = "ordinary_normalized"
METRIC_KEYS = h19.METRIC_KEYS


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase", choices=("development", "replay_tests"), required=True
    )
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--development-decision", type=Path)
    parser.add_argument("--recover-from", type=Path)
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
    parser.add_argument("--tail-scale-multiplier", type=float, default=2.0)
    parser.add_argument("--tail-coverage-threshold", type=float, default=1e-8)
    parser.add_argument("--error-scale", type=float, default=4.0)
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
        "tail_scale_multiplier": 2.0,
        "tail_coverage_threshold": 1e-8,
    }
    for name, expected in frozen.items():
        if getattr(args, name) != expected:
            raise SystemExit(
                f"frozen HIER-020 protocol requires --{name.replace('_', '-')} {expected}"
            )
    if not math.isfinite(args.error_scale) or args.error_scale <= 0.0:
        raise SystemExit("--error-scale must be finite and positive")
    replay = args.phase != "development"
    if replay != (args.development_decision is not None):
        raise SystemExit("replay requires --development-decision; development rejects it")
    if args.recover_from is not None and args.phase != "development":
        raise SystemExit("--recover-from is valid only for development")
    if args.visual_disposition != "pending" and args.recover_from is None:
        raise SystemExit("a visual verdict may only be recorded during immutable recovery")


def _validate_development_decision(args: argparse.Namespace) -> None:
    if args.phase == "development":
        return
    decision = json.loads(args.development_decision.read_text(encoding="utf-8"))
    if (
        decision.get("schema") != REPORT_SCHEMA
        or decision.get("phase") != "development"
        or CANDIDATE_MODE not in decision.get("numeric_candidates", [])
        or decision.get("visual_disposition") != "pass"
    ):
        raise SystemExit("HIER-020 did not pass numeric and visual development review")


def _discover_images(args: argparse.Namespace) -> list[Path]:
    if args.phase == "development":
        return h19._bound_paths(args.images, DEVELOPMENT_BINDINGS)
    return h13._discover_sources([args.images])


def _configs(args: argparse.Namespace):
    return h19._configs(args)


def _snapshot_sources(output_root: Path) -> list[dict[str, object]]:
    paths = (
        Path(__file__).resolve(),
        ROOT / "src" / "structsplat" / "tail_recovery.py",
        ROOT / "src" / "structsplat" / "fit.py",
        ROOT / "src" / "structsplat" / "render.py",
        ROOT / "scripts" / "experiments" / "hier019_confidence_tail.py",
        ROOT / "tasks" / "HIER-020-sparse-pixel-safe-tail.md",
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
    selected_count: int,
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
        "pixel_max_delta": (
            candidate["artifact_pixel_rmse_max"]
            - baseline["artifact_pixel_rmse_max"]
        ),
        "patch7_max_delta": (
            candidate["artifact_patch_rmse_max_7"]
            - baseline["artifact_patch_rmse_max_7"]
        ),
    }
    pointwise_vacuous_or_raw_strict = (
        selected_count == 0 or pointwise_raw_delta_max < 0.0
    )
    pointwise_vacuous_or_display_safe = (
        selected_count == 0 or pointwise_display_delta_max <= 0.0
    )
    clauses = {
        "candidate_finite": finite,
        "outside_payload_bit_exact": outside_identity_max_abs == 0.0,
        "baseline_cold_parity_le_2e_5": baseline_parity_max_abs <= 2e-5,
        "candidate_repeated_parity_le_2e_5": repeated_parity_max_abs <= 2e-5,
        "payload_roundtrip_exact": payload_roundtrip_exact,
        "pointwise_raw_strict": pointwise_vacuous_or_raw_strict,
        "pointwise_display_noninferior": pointwise_vacuous_or_display_safe,
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
        if selected_count > 0 and all(clauses.values()) and material
        else BASELINE_MODE
    )
    return selected, clauses, deltas


def _pointwise_deltas(
    baseline: np.ndarray,
    candidate: np.ndarray,
    target: np.ndarray,
    selected: np.ndarray,
) -> tuple[float, float]:
    if not selected.any():
        return 0.0, 0.0
    raw_base = np.square(baseline.astype(np.float64) - target).sum(axis=2)
    raw_candidate = np.square(candidate.astype(np.float64) - target).sum(axis=2)
    q_target = np.rint(np.clip(target, 0.0, 1.0) * 255.0)
    q_base = np.rint(np.clip(baseline, 0.0, 1.0) * 255.0)
    q_candidate = np.rint(np.clip(candidate, 0.0, 1.0) * 255.0)
    display_base = np.square(q_base - q_target).sum(axis=2)
    display_candidate = np.square(q_candidate - q_target).sum(axis=2)
    return (
        float(np.max((raw_candidate - raw_base)[selected])),
        float(np.max((display_candidate - display_base)[selected])),
    )


def _augment_direct_row(
    *,
    output_root: Path,
    row: dict[str, object],
    image: np.ndarray,
    mask: np.ndarray,
    baseline_render: np.ndarray,
    fit_config,
    tail_config: ConfidenceTailConfig,
    args: argparse.Namespace,
) -> dict[str, object]:
    import torch

    artifact_dir = output_root / str(row["artifact_dir"])
    field_path = artifact_dir / "field.gaussian.npz"
    field_hash_before = report_utils._sha256(field_path)
    field_bytes = field_path.stat().st_size
    cold_field = GaussianField.load(str(field_path), device=args.device)
    height, width = image.shape[:2]

    torch.cuda.synchronize()
    baseline_started = time.perf_counter()
    ordinary = h17._render_numpy(cold_field, fit_config, height, width)
    torch.cuda.synchronize()
    baseline_render_seconds = time.perf_counter() - baseline_started

    target = torch.as_tensor(image, device=args.device, dtype=torch.float32).contiguous()
    torch.cuda.synchronize()
    encoder_started = time.perf_counter()
    first = render_confidence_gated_self_prior(
        cold_field, fit_config, height, width, tail_config
    )
    sparse = select_pixel_safe_tail(first, target)
    encoded = sparse.payload.to_bytes()
    torch.cuda.synchronize()
    encoder_seconds = time.perf_counter() - encoder_started

    candidate_path = artifact_dir / "candidate.sst1"
    candidate_path.write_bytes(encoded)
    parsed = SparseTailPayload.from_bytes(candidate_path.read_bytes())
    payload_roundtrip_exact = parsed.to_bytes() == encoded

    repeated_field = GaussianField.load(str(field_path), device=args.device)
    torch.cuda.synchronize()
    decode_started = time.perf_counter()
    repeated = render_confidence_gated_self_prior(
        repeated_field, fit_config, height, width, tail_config
    )
    replay_tensor = apply_sparse_tail_payload(repeated, parsed)
    torch.cuda.synchronize()
    decode_seconds = time.perf_counter() - decode_started

    cold_baseline = first.baseline.detach().cpu().numpy().astype(np.float32, copy=False)
    proposal = first.candidate.detach().cpu().numpy().astype(np.float32, copy=False)
    candidate = sparse.candidate.detach().cpu().numpy().astype(np.float32, copy=False)
    replay = replay_tensor.detach().cpu().numpy().astype(np.float32, copy=False)
    denominator = first.denominator.detach().cpu().numpy().astype(np.float64)
    missing_mass = first.missing_mass.detach().cpu().numpy().astype(np.float64)
    active = first.activation_mask.detach().cpu().numpy().astype(bool, copy=False)
    selected = sparse.selected_mask.detach().cpu().numpy().astype(bool, copy=False)
    baseline_parity = float(np.max(np.abs(ordinary - baseline_render)))
    tail_baseline_parity = float(np.max(np.abs(cold_baseline - ordinary)))
    repeated_parity = float(np.max(np.abs(replay - candidate)))
    outside_delta = np.abs(candidate - cold_baseline)[~selected]
    outside_identity = float(outside_delta.max()) if outside_delta.size else 0.0
    raw_delta_max, display_delta_max = _pointwise_deltas(
        cold_baseline, candidate, image, selected
    )

    metric_started = time.perf_counter()
    candidate_metrics = report_utils._metric_values(
        candidate, image, mask, device=args.device, compute_lpips=args.lpips
    )
    metric_seconds = time.perf_counter() - metric_started
    candidate_subset = _metric_subset(candidate_metrics)
    baseline_subset = _metric_subset(row)
    finite = bool(
        np.isfinite(candidate).all()
        and np.isfinite(proposal).all()
        and np.isfinite(denominator).all()
        and np.isfinite(missing_mass).all()
        and all(math.isfinite(value) for value in candidate_subset.values())
    )
    selected_mode, clauses, deltas = _selection(
        baseline_subset,
        candidate_subset,
        selected_count=sparse.selected_count,
        finite=finite,
        outside_identity_max_abs=outside_identity,
        baseline_parity_max_abs=max(baseline_parity, tail_baseline_parity),
        repeated_parity_max_abs=repeated_parity,
        payload_roundtrip_exact=payload_roundtrip_exact,
        pointwise_raw_delta_max=raw_delta_max,
        pointwise_display_delta_max=display_delta_max,
    )
    selected_metrics = (
        candidate_subset if selected_mode == CANDIDATE_MODE else baseline_subset
    )
    selected_payload_bytes = len(encoded) if selected_mode == CANDIDATE_MODE else 0
    selected_payload_path: str | None = None
    selected_payload_sha256: str | None = None
    if selected_mode == CANDIDATE_MODE:
        selected_path = artifact_dir / "tail.sst1"
        selected_path.write_bytes(encoded)
        selected_payload_path = str(selected_path.relative_to(output_root))
        selected_payload_sha256 = report_utils._sha256(selected_path)

    candidate_dir = artifact_dir / "sparse_tail_candidate"
    candidate_dir.mkdir(parents=True, exist_ok=False)
    h15._save_visuals(
        candidate_dir, image, candidate, cold_baseline, mask, args.error_scale
    )
    save_image(str(candidate_dir / "unmasked_proposal.png"), proposal)
    save_error_heatmap(
        str(candidate_dir / "candidate_delta.png"),
        candidate - cold_baseline,
        scale=args.error_scale,
    )
    overlay = np.clip(cold_baseline, 0.0, 1.0).copy()
    overlay[selected] = 0.35 * overlay[selected] + 0.65 * np.asarray(
        [1.0, 0.0, 0.0], dtype=np.float32
    )
    save_image(str(candidate_dir / "selected_sites.png"), overlay)
    with np.load(artifact_dir / "analysis.npz", allow_pickle=False) as analysis:
        common_bounds = tuple(int(value) for value in analysis["crop_bounds"])
    h15.viz_utils._save_crop(
        candidate_dir / "source_common_crop.png", image, common_bounds
    )
    h15.viz_utils._save_crop(
        candidate_dir / "baseline_common_crop.png", cold_baseline, common_bounds
    )
    h15.viz_utils._save_crop(
        candidate_dir / "candidate_common_crop.png", candidate, common_bounds
    )

    baseline_error = np.sqrt(
        np.mean((cold_baseline.astype(np.float64) - image) ** 2, axis=2)
    )
    candidate_error = np.sqrt(
        np.mean((candidate.astype(np.float64) - image) ** 2, axis=2)
    )
    coordinates = np.argwhere(selected).astype(np.int32)
    np.savez_compressed(
        candidate_dir / "tail_analysis.npz",
        activation_mask=active,
        selected_mask=selected,
        selected_yx=coordinates,
        denominator=denominator.astype(np.float32),
        missing_mass=missing_mass.astype(np.float32),
        proposal_delta=(proposal - cold_baseline).astype(np.float32),
        candidate_delta=(candidate - cold_baseline).astype(np.float32),
        baseline_pixel_rmse=baseline_error.astype(np.float32),
        candidate_pixel_rmse=candidate_error.astype(np.float32),
        common_crop_bounds=np.asarray(common_bounds, dtype=np.int32),
    )
    telemetry: dict[str, object] = {
        "schema": REPORT_SCHEMA,
        "status": "diagnostic",
        "semantic_family": "sparse_pixel_safe_same_field_prior_sst1_v1",
        "scale_multiplier": first.scale_multiplier,
        "coverage_threshold": first.coverage_threshold,
        "normalization_eps": first.normalization_eps,
        "activation_count": first.activation_count,
        "selected_count": sparse.selected_count,
        "rejected_count": first.activation_count - sparse.selected_count,
        "selected_yx": coordinates.tolist(),
        "denominator_quantiles": h19._quantiles(denominator),
        "missing_mass_quantiles": h19._quantiles(missing_mass),
        "pointwise_raw_sse_delta_max": raw_delta_max,
        "pointwise_display_sse_delta_max": display_delta_max,
        "outside_identity_max_abs": outside_identity,
        "baseline_cold_parity_max_abs": max(baseline_parity, tail_baseline_parity),
        "candidate_repeated_parity_max_abs": repeated_parity,
        "payload_roundtrip_exact": payload_roundtrip_exact,
        "candidate_payload_bytes": len(encoded),
        "candidate_payload_sha256": report_utils._sha256(candidate_path),
        "field_plus_candidate_reference_bytes": field_bytes + len(encoded),
        "selected_payload_bytes": selected_payload_bytes,
        "selected_payload_path": selected_payload_path,
        "selected_payload_sha256": selected_payload_sha256,
        "field_plus_selected_reference_bytes": field_bytes + selected_payload_bytes,
        "candidate_finite": finite,
        "candidate_metrics": candidate_subset,
        "metric_deltas_vs_baseline": deltas,
        "selection_clauses": clauses,
        "selected_mode": selected_mode,
        "selected_metrics": selected_metrics,
        "ordinary_render_seconds": baseline_render_seconds,
        "encoder_tail_seconds": encoder_seconds,
        "selected_tail_decode_seconds": decode_seconds,
        "candidate_metric_seconds": metric_seconds,
        "field_file_sha256_before": field_hash_before,
        "field_file_sha256_after": report_utils._sha256(field_path),
    }
    report_utils._write_json(candidate_dir / "tail_recovery.json", telemetry)

    row.update(
        {
            "sst1_semantic_family": telemetry["semantic_family"],
            "sst1_scale_multiplier": first.scale_multiplier,
            "sst1_coverage_threshold": first.coverage_threshold,
            "sst1_activation_count": first.activation_count,
            "sst1_selected_count": sparse.selected_count,
            "sst1_rejected_count": first.activation_count - sparse.selected_count,
            "sst1_selected_yx": coordinates.tolist(),
            "sst1_outside_identity_max_abs": outside_identity,
            "sst1_baseline_cold_parity_max_abs": max(
                baseline_parity, tail_baseline_parity
            ),
            "sst1_candidate_repeated_parity_max_abs": repeated_parity,
            "sst1_payload_roundtrip_exact": payload_roundtrip_exact,
            "sst1_pointwise_raw_sse_delta_max": raw_delta_max,
            "sst1_pointwise_display_sse_delta_max": display_delta_max,
            "sst1_candidate_finite": finite,
            "sst1_candidate_metrics": candidate_subset,
            "sst1_metric_deltas_vs_baseline": deltas,
            "sst1_selection_clauses": clauses,
            "sst1_selected_mode": selected_mode,
            "sst1_selected_metrics": selected_metrics,
            "sst1_candidate_payload_bytes": len(encoded),
            "sst1_candidate_payload_sha256": telemetry["candidate_payload_sha256"],
            "sst1_selected_payload_bytes": selected_payload_bytes,
            "sst1_selected_payload_path": selected_payload_path,
            "sst1_selected_payload_sha256": selected_payload_sha256,
            "sst1_field_plus_candidate_reference_bytes": field_bytes + len(encoded),
            "sst1_field_plus_selected_reference_bytes": field_bytes
            + selected_payload_bytes,
            "sst1_ordinary_render_seconds": baseline_render_seconds,
            "sst1_encoder_tail_seconds": encoder_seconds,
            "sst1_selected_tail_decode_seconds": decode_seconds,
            "sst1_candidate_metric_seconds": metric_seconds,
            "sst1_decode_time_ratio": decode_seconds
            / max(baseline_render_seconds, 1e-12),
            "sst1_pipeline_time_ratio": (
                (float(row["pipeline_algorithm_seconds"]) + encoder_seconds)
                / max(float(row["pipeline_algorithm_seconds"]), 1e-12)
            ),
            "sst1_field_file_sha256_before": field_hash_before,
            "sst1_field_file_sha256_after": report_utils._sha256(field_path),
        }
    )
    report_utils._write_json(artifact_dir / "row.json", row)
    return row


def _retime_selected_decode(
    *,
    output_root: Path,
    row: dict[str, object],
    fit_config,
    tail_config: ConfidenceTailConfig,
    device: str,
) -> dict[str, object]:
    """Replace full-proposal diagnostic timing with the actually selected decode path."""
    import torch

    from structsplat.fit import _render
    from structsplat.tail_recovery import render_sparse_tail_payload

    artifact_dir = output_root / str(row["artifact_dir"])
    field_path = artifact_dir / "field.gaussian.npz"
    height, width = int(row["height"]), int(row["width"])
    field = GaussianField.load(str(field_path), device=device)
    torch.cuda.synchronize()
    baseline_started = time.perf_counter()
    baseline = _render(field, fit_config, height, width)
    torch.cuda.synchronize()
    baseline_seconds = time.perf_counter() - baseline_started

    if row["sst1_selected_mode"] == CANDIDATE_MODE:
        payload = SparseTailPayload.from_bytes((artifact_dir / "candidate.sst1").read_bytes())
    else:
        payload = SparseTailPayload(height, width)
    selected_field = GaussianField.load(str(field_path), device=device)
    torch.cuda.synchronize()
    decode_started = time.perf_counter()
    decoded = render_sparse_tail_payload(
        selected_field, fit_config, height, width, payload, tail_config
    )
    torch.cuda.synchronize()
    decode_seconds = time.perf_counter() - decode_started
    if payload.count:
        full = render_confidence_gated_self_prior(
            field, fit_config, height, width, tail_config
        )
        expected = apply_sparse_tail_payload(full, payload)
    else:
        expected = baseline
    torch.cuda.synchronize()
    parity = float((decoded - expected).abs().max().detach().cpu())
    row.update(
        {
            "sst1_full_frame_decode_seconds_v1": row["sst1_selected_tail_decode_seconds"],
            "sst1_full_frame_decode_time_ratio_v1": row["sst1_decode_time_ratio"],
            "sst1_ordinary_render_seconds": baseline_seconds,
            "sst1_selected_tail_decode_seconds": decode_seconds,
            "sst1_decode_time_ratio": decode_seconds / max(baseline_seconds, 1e-12),
            "sst1_selected_decode_payload_count": payload.count,
            "sst1_optimized_decode_parity_max_abs": parity,
            "sst1_decode_implementation": "ordinary_plus_coordinate_only_tail_v2",
        }
    )
    report_utils._write_json(artifact_dir / "row.json", row)
    return row


def _selected_pairs(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    controls = {
        str(row["image"]): row for row in rows if row.get("arm") == CONTROL_ARM
    }
    pairs: list[dict[str, object]] = []
    for row in rows:
        if row.get("arm") != DIRECT_ARM:
            continue
        control = controls.get(str(row["image"]))
        if control is None:
            continue
        selected = row["sst1_selected_metrics"]
        assert isinstance(selected, dict)
        pairs.append(
            {
                "image": row["image"],
                "selected_mode": row["sst1_selected_mode"],
                "selected_count": row["sst1_selected_count"],
                "psnr_delta_db": float(selected["psnr_db"])
                - float(control["psnr_db"]),
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
    args: argparse.Namespace,
) -> dict[str, object]:
    expected = 4 if args.phase == "development" else 16
    direct = [row for row in rows if row.get("arm") == DIRECT_ARM]
    pairs = _selected_pairs(rows)
    baseline_local_failures = [
        pair
        for pair in pairs
        if float(pair["baseline_pixel_max_delta"]) > 1e-12
        or float(pair["baseline_patch7_max_delta"]) > 1e-12
    ]
    gates = {
        "complete_direct_rows": len(direct) == expected,
        "complete_h005_pairs": len(pairs) == expected,
        "complete_attempt_ledger": len(attempts) == expected * 2,
        "zero_failures": all(record.get("status") == "ok" for record in attempts),
        "all_exact_count": all(int(row["n_gaussians"]) == 7000 for row in direct),
        "all_field_bytes_unchanged": all(
            row["sst1_field_file_sha256_before"]
            == row["sst1_field_file_sha256_after"]
            for row in direct
        ),
        "all_payloads_canonical": all(
            bool(row["sst1_payload_roundtrip_exact"]) for row in direct
        ),
        "all_candidates_finite": all(bool(row["sst1_candidate_finite"]) for row in direct),
        "all_outside_payload_bit_exact": all(
            float(row["sst1_outside_identity_max_abs"]) == 0.0 for row in direct
        ),
        "all_parity_le_2e_5": all(
            float(row["sst1_baseline_cold_parity_max_abs"]) <= 2e-5
            and float(row["sst1_candidate_repeated_parity_max_abs"]) <= 2e-5
            for row in direct
        ),
        "all_optimized_selected_decode_parity_le_2e_5": all(
            float(
                row.get(
                    "sst1_optimized_decode_parity_max_abs",
                    row["sst1_candidate_repeated_parity_max_abs"],
                )
            )
            <= 2e-5
            for row in direct
        ),
        "all_selected_transactions_safe": all(
            row["sst1_selected_mode"] == BASELINE_MODE
            or all(bool(value) for value in row["sst1_selection_clauses"].values())
            for row in direct
        ),
        "all_selected_mse_noninferior_vs_baseline": all(
            row["sst1_selected_mode"] == BASELINE_MODE
            or float(row["sst1_metric_deltas_vs_baseline"]["mse_ratio"])
            <= 1.0 + 1e-8
            for row in direct
        ),
        "all_selected_pixel_noninferior_vs_baseline": all(
            row["sst1_selected_mode"] == BASELINE_MODE
            or float(row["sst1_metric_deltas_vs_baseline"]["pixel_max_delta"])
            <= 1e-12
            for row in direct
        ),
        "all_selected_patch_noninferior_vs_baseline": all(
            row["sst1_selected_mode"] == BASELINE_MODE
            or float(row["sst1_metric_deltas_vs_baseline"]["patch7_max_delta"])
            <= 1e-12
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
            for pair in baseline_local_failures
        ),
        "median_pipeline_time_ratio_le_1_25": (
            float(np.median([row["sst1_pipeline_time_ratio"] for row in direct]))
            <= 1.25
            if direct
            else False
        ),
        "median_selected_tail_decode_ratio_le_5": (
            float(np.median([row["sst1_decode_time_ratio"] for row in direct])) <= 5.0
            if direct
            else False
        ),
    }
    candidate = all(gates.values())
    visual_disposition = getattr(args, "visual_disposition", "pending")
    return {
        "schema": REPORT_SCHEMA,
        "status": "diagnostic",
        "phase": args.phase,
        "gates": gates,
        "selected_vs_h005_pairs": pairs,
        "baseline_local_failure_count": len(baseline_local_failures),
        "sst1_selected_image_count": sum(
            row["sst1_selected_mode"] == CANDIDATE_MODE for row in direct
        ),
        "sst1_proposed_safe_pixel_count": sum(
            int(row["sst1_selected_count"]) for row in direct
        ),
        "sst1_selected_pixel_count": sum(
            int(row["sst1_selected_count"])
            for row in direct
            if row["sst1_selected_mode"] == CANDIDATE_MODE
        ),
        "candidate_side_bytes": sum(
            int(row["sst1_candidate_payload_bytes"]) for row in direct
        ),
        "selected_side_bytes": sum(
            int(row["sst1_selected_payload_bytes"]) for row in direct
        ),
        "attempt_count": len(attempts),
        "failure_count": sum(record.get("status") != "ok" for record in attempts),
        "numeric_candidates": [CANDIDATE_MODE] if candidate else [],
        "numeric_disposition": CANDIDATE_MODE if candidate else "no_robust_tail_candidate",
        "numeric_bank_pass": candidate,
        "bounded_bank_pass": candidate and visual_disposition == "pass",
        "visual_review_required": True,
        "visual_disposition": visual_disposition,
        "interpretation": (
            "Numeric portfolio requires native visual review before any replay."
            if candidate
            else "The frozen sparse portfolio misses a numeric gate; do not replay or retune."
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
                f"<td>{float(row['psnr_db']):.3f}</td>"
                f"<td>{float(row['ms_ssim']):.5f}</td>"
                f"<td>{float(row['lpips']):.5f}</td>"
                f"<td>{float(row['artifact_pixel_rmse_max']):.4f}</td>"
                f"<td>{float(row['artifact_patch_rmse_max_7']):.4f}</td>"
                "<td>—</td><td>—</td><td>—</td>"
                f"<td><a href='{artifact}/reconstruction.png'>full</a> · "
                f"<a href='{artifact}/reconstruction_crop.png'>crop</a></td></tr>"
            )
        else:
            candidate = row["sst1_candidate_metrics"]
            assert isinstance(candidate, dict)
            table.append(
                "<tr>"
                f"<td>{escape(str(row['image']))}</td><td>{DIRECT_ARM}</td>"
                f"<td>{float(row['psnr_db']):.3f} / {float(candidate['psnr_db']):.3f}</td>"
                f"<td>{float(row['ms_ssim']):.5f} / "
                f"{float(candidate['ms_ssim']):.5f}</td>"
                f"<td>{float(row['lpips']):.5f} / {float(candidate['lpips']):.5f}</td>"
                f"<td>{float(row['artifact_pixel_rmse_max']):.4f} / "
                f"{float(candidate['artifact_pixel_rmse_max']):.4f}</td>"
                f"<td>{float(row['artifact_patch_rmse_max_7']):.4f} / "
                f"{float(candidate['artifact_patch_rmse_max_7']):.4f}</td>"
                f"<td>{int(row['sst1_activation_count'])}</td>"
                f"<td>{int(row['sst1_selected_count'])}</td>"
                f"<td>{escape(str(row['sst1_selected_mode']))}</td>"
                f"<td><a href='{artifact}/reconstruction.png'>base</a> · "
                f"<a href='{artifact}/reconstruction_crop.png'>base crop</a> · "
                f"<a href='{artifact}/sparse_tail_candidate/reconstruction.png'>tail</a> · "
                f"<a href='{artifact}/sparse_tail_candidate/candidate_common_crop.png'>"
                "common crop</a> · "
                f"<a href='{artifact}/sparse_tail_candidate/selected_sites.png'>sites</a> · "
                f"<a href='{artifact}/candidate.sst1'>SST1</a></td></tr>"
            )
        cards.append(
            f"<section><h3>{escape(str(row['image']))} — "
            f"{escape(str(row['arm']))}</h3>"
            f"<a href='{artifact}/source.png'><img src='{artifact}/source.png'></a>"
            f"<a href='{artifact}/reconstruction.png'>"
            f"<img src='{artifact}/reconstruction.png'></a>"
            f"<a href='{artifact}/error.png'><img src='{artifact}/error.png'></a>"
            f"<a href='{artifact}/reconstruction_crop.png'>"
            f"<img src='{artifact}/reconstruction_crop.png'></a>"
            + (
                f"<a href='{artifact}/sparse_tail_candidate/reconstruction.png'>"
                f"<img src='{artifact}/sparse_tail_candidate/reconstruction.png'></a>"
                f"<a href='{artifact}/sparse_tail_candidate/candidate_delta.png'>"
                f"<img src='{artifact}/sparse_tail_candidate/candidate_delta.png'></a>"
                f"<a href='{artifact}/sparse_tail_candidate/selected_sites.png'>"
                f"<img src='{artifact}/sparse_tail_candidate/selected_sites.png'></a>"
                if row["arm"] == DIRECT_ARM
                else ""
            )
            + "</section>"
        )
    html = f"""<!doctype html><html><head><meta charset='utf-8'><title>HIER-020</title>
<style>body{{font-family:system-ui;margin:2rem;max-width:1700px}}table{{border-collapse:collapse}}
th,td{{border:1px solid #bbb;padding:.35rem}}img{{max-width:300px;max-height:240px;margin:.25rem}}
section{{border-top:1px solid #aaa;margin-top:1rem}}pre,code{{white-space:pre-wrap}}</style>
</head><body><h1>HIER-020 sparse pixel-safe confidence tail —
{escape(str(decision['phase']))}</h1>
<p>Dirty-source diagnostic. Direct cells show <strong>ordinary / sparse candidate</strong>.
SST1 stores sorted raster indices; the source image is unavailable to decode.</p>
<p><code>{escape(command)}</code></p><p><a href='config.json'>config</a> ·
<a href='decision.json'>decision</a> · <a href='metrics.json'>JSON</a> ·
<a href='metrics.jsonl'>JSONL</a> · <a href='metrics.csv'>CSV</a> ·
<a href='attempts.json'>attempts</a> · <a href='manifest.json'>manifest</a></p>
<h2>Decision</h2><pre>{escape(json.dumps(decision, indent=2, sort_keys=True))}</pre>
<h2>Cells</h2><table><tr><th>image</th><th>arm</th><th>PSNR</th><th>MS-SSIM</th>
<th>LPIPS</th><th>pixel max</th><th>7x7 max</th><th>proposed</th><th>stored</th>
<th>selected</th><th>visuals/data</th></tr>{''.join(table)}</table>
<h2>Visual audit</h2>{''.join(cards)}</body></html>"""
    (output_root / "index.html").write_text(html, encoding="utf-8")


def _recover(
    args: argparse.Namespace,
    images: list[Path],
    output_root: Path,
    command: str,
) -> bool:
    """Recover the raw fresh cells with selected-mode sparse decode timing only."""
    if args.recover_from is None:
        return False

    import torch

    from structsplat.fit import _render
    from structsplat.tail_recovery import render_sparse_tail_payload

    source_root = args.recover_from.resolve()
    ledger_paths = [
        source_root / name
        for name in ("metrics.json", "attempts.json", "decision.json", "config.json")
    ]
    if not source_root.is_dir() or not all(path.is_file() for path in ledger_paths):
        raise SystemExit("recovery source is missing its raw ledgers")
    source_metrics = json.loads(ledger_paths[0].read_text(encoding="utf-8"))
    attempts_record = json.loads(ledger_paths[1].read_text(encoding="utf-8"))
    raw_decision = json.loads(ledger_paths[2].read_text(encoding="utf-8"))
    rows = source_metrics.get("rows", [])
    attempts = attempts_record.get("attempts", [])
    if (
        source_metrics.get("schema") != REPORT_SCHEMA
        or raw_decision.get("schema") != REPORT_SCHEMA
        or len(rows) != 8
        or len(attempts) != 8
        or any(record.get("status") != "ok" for record in attempts)
    ):
        raise SystemExit("recovery source is not a complete successful HIER-020 raw run")

    recovery_started = time.perf_counter()
    shutil.copytree(source_root, output_root)
    recovery_snapshot_dir = output_root / "recovery_source_snapshot"
    snapshot_sources = (
        Path(__file__).resolve(),
        ROOT / "src" / "structsplat" / "tail_recovery.py",
        ROOT / "tasks" / "HIER-020-sparse-pixel-safe-tail.md",
    )
    recovery_snapshots: list[dict[str, object]] = []
    for source in snapshot_sources:
        relative = source.relative_to(ROOT)
        destination = recovery_snapshot_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        recovery_snapshots.append(
            {
                "repository_path": str(relative),
                "snapshot_path": str(destination.relative_to(output_root)),
                "bytes": destination.stat().st_size,
                "sha256": report_utils._sha256(destination),
            }
        )

    _, fit_config, tail_config = _configs(args)
    direct = [row for row in rows if row.get("arm") == DIRECT_ARM]
    first_row = direct[0]
    first_field = GaussianField.load(
        str(output_root / str(first_row["artifact_dir"]) / "field.gaussian.npz"),
        device=args.device,
    )
    warmup_started = time.perf_counter()
    _render(first_field, fit_config, int(first_row["height"]), int(first_row["width"]))
    torch.cuda.synchronize()
    warmup_seconds = time.perf_counter() - warmup_started

    for row in direct:
        artifact_dir = output_root / str(row["artifact_dir"])
        field_path = artifact_dir / "field.gaussian.npz"
        height, width = int(row["height"]), int(row["width"])
        field = GaussianField.load(str(field_path), device=args.device)
        torch.cuda.synchronize()
        baseline_started = time.perf_counter()
        baseline = _render(field, fit_config, height, width)
        torch.cuda.synchronize()
        baseline_seconds = time.perf_counter() - baseline_started

        if row["sst1_selected_mode"] == CANDIDATE_MODE:
            selected_bytes = (artifact_dir / "candidate.sst1").read_bytes()
            payload = SparseTailPayload.from_bytes(selected_bytes)
        else:
            payload = SparseTailPayload(height, width)
        selected_field = GaussianField.load(str(field_path), device=args.device)
        torch.cuda.synchronize()
        decode_started = time.perf_counter()
        decoded = render_sparse_tail_payload(
            selected_field,
            fit_config,
            height,
            width,
            payload,
            tail_config,
        )
        torch.cuda.synchronize()
        selected_decode_seconds = time.perf_counter() - decode_started

        if payload.count:
            full = render_confidence_gated_self_prior(
                field, fit_config, height, width, tail_config
            )
            expected = apply_sparse_tail_payload(full, payload)
        else:
            expected = baseline
        torch.cuda.synchronize()
        parity = float((decoded - expected).abs().max().detach().cpu())
        old_decode_seconds = float(row["sst1_selected_tail_decode_seconds"])
        old_decode_ratio = float(row["sst1_decode_time_ratio"])
        row.update(
            {
                "sst1_full_frame_decode_seconds_v1": old_decode_seconds,
                "sst1_full_frame_decode_time_ratio_v1": old_decode_ratio,
                "sst1_ordinary_render_seconds": baseline_seconds,
                "sst1_selected_tail_decode_seconds": selected_decode_seconds,
                "sst1_decode_time_ratio": selected_decode_seconds
                / max(baseline_seconds, 1e-12),
                "sst1_selected_decode_payload_count": payload.count,
                "sst1_optimized_decode_parity_max_abs": parity,
                "sst1_decode_implementation": "ordinary_plus_coordinate_only_tail_v2",
                "sst1_timing_recovered_without_refit": True,
            }
        )
        report_utils._write_json(artifact_dir / "row.json", row)

    decision = _decision(rows, attempts, args)
    decision.update(
        {
            "raw_cell_elapsed_seconds": raw_decision.get("elapsed_seconds"),
            "recovery_elapsed_seconds": time.perf_counter() - recovery_started,
            "cuda_extension_warmup_seconds": warmup_seconds,
            "recovered_from_complete_raw_run": True,
            "cell_computation_rerun": False,
            "quality_metrics_recomputed": False,
            "selection_rule_changed": False,
            "decoder_implementation_changed": True,
        }
    )
    report_utils._write_json(output_root / "decision.json", decision)

    config = json.loads((output_root / "config.json").read_text(encoding="utf-8"))
    config.update(
        {
            "command": command,
            "recovery": {
                "source_path": str(source_root),
                "source_metrics_sha256": report_utils._sha256(ledger_paths[0]),
                "source_attempts_sha256": report_utils._sha256(ledger_paths[1]),
                "source_decision_sha256": report_utils._sha256(ledger_paths[2]),
                "cell_computation_rerun": False,
                "quality_metrics_recomputed": False,
                "selection_rule_changed": False,
                "decoder_implementation_changed": True,
                "visual_disposition": args.visual_disposition,
            },
            "recovery_source_snapshots": recovery_snapshots,
        }
    )
    report_utils._write_json(output_root / "config.json", config)
    report_utils._write_json(
        output_root / "recovery.json",
        {
            "schema": REPORT_SCHEMA,
            "status": "diagnostic",
            "command": command,
            **config["recovery"],
            "recovery_source_snapshots": recovery_snapshots,
            "source_count": len(images),
        },
    )
    h15._write_tables(output_root, rows, schema=REPORT_SCHEMA)
    _write_report(output_root, rows, decision, command)
    h15._write_manifest(output_root, schema=REPORT_SCHEMA)
    print(json.dumps(decision, indent=2, sort_keys=True))
    return True


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    _validate_args(args)
    _validate_development_decision(args)
    images = _discover_images(args)
    output_root = args.out.resolve()

    import torch

    command = shlex.join(
        [sys.executable, str(Path(__file__).resolve()), *(argv or sys.argv[1:])]
    )
    if _recover(args, images, output_root, command):
        return 0
    output_root.mkdir(parents=True, exist_ok=False)
    contraction_config = h15._contraction_config(args)
    init_config, fit_config, tail_config = _configs(args)
    config = {
        "schema": REPORT_SCHEMA,
        "status": "diagnostic",
        "phase": args.phase,
        "command": command,
        "args": vars(args),
        "development_selection_digests": SELECTION_DIGESTS,
        "sources": [
            {"path": str(path), "sha256": report_utils._sha256(path)} for path in images
        ],
        "contraction": asdict(contraction_config),
        "direct_init": asdict(init_config),
        "fit": asdict(fit_config),
        "tail": asdict(tail_config),
        "sst1": {
            "magic": "SST1",
            "header_bytes": 16,
            "index_bytes": 4,
            "index_order": "strictly_increasing_raster_flat_uint32_le",
            "encoder_raw_rule": "recovered_rgb_sse < baseline_rgb_sse",
            "encoder_display_rule": "q8_recovered_rgb_sse <= q8_baseline_rgb_sse",
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
            "Target-known pixel selection and perceptual rollback are encoder-side RDO.",
            "NPZ plus SST1 is reference accounting, not a production codec rate.",
            "CUDA accumulation is numerically, not bit, reproducible.",
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

    def persist() -> None:
        h15._write_tables(output_root, rows, schema=REPORT_SCHEMA)

    run_started = time.perf_counter()
    for image_path in images:
        load_started = time.perf_counter()
        try:
            image, loaded_mask, raster = report_utils._load_evaluation_raster(
                image_path, None, max_side=args.max_side, mask_threshold=0.5
            )
            if loaded_mask is not None:
                raise RuntimeError("HIER-020 requires a generated full-frame mask")
        except Exception as exc:
            record(image_path, CONTROL_ARM, load_started, exc)
            record(image_path, DIRECT_ARM, load_started, exc)
            continue
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
            persist()
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
                    "tail_candidate_uses_same_field": True,
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
                tail_config=tail_config,
                args=args,
            )
            row = _retime_selected_decode(
                output_root=output_root,
                row=row,
                fit_config=fit_config,
                tail_config=tail_config,
                device=args.device,
            )
            rows.append(row)
            record(image_path, DIRECT_ARM, started)
            persist()
        except Exception as exc:
            record(image_path, DIRECT_ARM, started, exc)

    decision = _decision(rows, attempts, args)
    decision["elapsed_seconds"] = time.perf_counter() - run_started
    report_utils._write_json(output_root / "decision.json", decision)
    h15._write_tables(output_root, rows, schema=REPORT_SCHEMA)
    _write_report(output_root, rows, decision, command)
    h15._write_manifest(output_root, schema=REPORT_SCHEMA)
    print(json.dumps(decision, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
