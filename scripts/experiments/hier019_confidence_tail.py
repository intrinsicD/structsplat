#!/usr/bin/env python3
"""Run HIER-019's source-bound confidence-gated same-field tail diagnostic."""
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
from scripts.experiments import hier016_normalized_tail_repair as h16  # noqa: E402
from scripts.experiments import hier017_normalization_epsilon as h17  # noqa: E402
from scripts.experiments import hier018_counted_background as h18  # noqa: E402
from structsplat.config import StructureTensorConfig  # noqa: E402
from structsplat.gaussians import GaussianField  # noqa: E402
from structsplat.init import build_field  # noqa: E402
from structsplat.pixel_contraction import contract_image  # noqa: E402
from structsplat.tail_recovery import (  # noqa: E402
    ConfidenceTailConfig,
    render_confidence_gated_self_prior,
)


REPORT_SCHEMA = "structsplat.hier019_confidence_tail.diagnostic.v1"
DEVELOPMENT_BINDINGS = {
    "COCO_train2014_000000489983.jpg": (
        "be167d03370237f18b6121d760c20e7c6ac42269d842165e868174f7299c8bf1"
    ),
    "COCO_train2014_000000568599.jpg": (
        "1510b72c75a7eb0f7583be727bd64312382b8a05825a6b3dd09503923c90fa2e"
    ),
    "COCO_train2014_000000078213.jpg": (
        "fe89777b6b42252a44108d7491f6104872fd667941ae9cf1258c67b8c065fb5a"
    ),
    "COCO_train2014_000000564341.jpg": (
        "b4808c36dbbb8ddd835b052be392a7659a672277a262f962196bfa6b372bb209"
    ),
}
SELECTION_DIGESTS = {
    "COCO_train2014_000000489983.jpg": (
        "0000b25ae4eee1321e1371aee79521a5579847c8135de72feb81b20417c86a2f"
    ),
    "COCO_train2014_000000568599.jpg": (
        "0000ca76903c8227e2a2e6d8b994f627c71ab3dd18556e8b61a5d2285e784a0f"
    ),
    "COCO_train2014_000000078213.jpg": (
        "00018c7ff0b18668d93f606ebe0f7186687d8d7d212d7a2a55741e07795d3947"
    ),
    "COCO_train2014_000000564341.jpg": (
        "00018fbe03e7a72c12514ebc9ef3b04b1c0ed4e7fb6bb7587df31ce7e32f68ae"
    ),
}
CONTROL_ARM = "h005_control"
DIRECT_ARM = "direct_no_recovery"
CANDIDATE_MODE = "confidence_self_prior2"
BASELINE_MODE = "ordinary_normalized"
METRIC_KEYS = (
    "masked_mse",
    "psnr_db",
    "ssim",
    "ms_ssim",
    "lpips",
    "artifact_pixel_rmse_max",
    "artifact_patch_rmse_max_7",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=(
            "development",
            "replay_h15",
            "replay_h16",
            "replay_h17",
            "replay_h18",
            "replay_tests",
        ),
        required=True,
    )
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--development-decision", type=Path)
    parser.add_argument("--control-metrics", type=Path)
    parser.add_argument("--recover-from", type=Path)
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
                f"frozen HIER-019 protocol requires --{name.replace('_', '-')} {expected}"
            )
    if not math.isfinite(args.error_scale) or args.error_scale <= 0.0:
        raise SystemExit("--error-scale must be finite and positive")
    replay = args.phase != "development"
    if replay != (args.development_decision is not None):
        raise SystemExit("replay requires --development-decision; development rejects it")
    needs_external = args.phase in (
        "replay_h15",
        "replay_h16",
        "replay_h17",
        "replay_h18",
    )
    if needs_external != (args.control_metrics is not None):
        raise SystemExit("COCO replays require --control-metrics; other phases reject it")
    if args.recover_from is not None and args.phase != "development":
        raise SystemExit("--recover-from is valid only for development")


def _validate_development_decision(args: argparse.Namespace) -> None:
    if args.phase == "development":
        return
    decision = json.loads(args.development_decision.read_text(encoding="utf-8"))
    if (
        decision.get("schema") != REPORT_SCHEMA
        or decision.get("phase") != "development"
        or CANDIDATE_MODE not in decision.get("numeric_candidates", [])
    ):
        raise SystemExit("HIER-019 did not pass fresh development")


def _bound_paths(directory: Path, bindings: dict[str, str]) -> list[Path]:
    paths = [directory / name for name in bindings]
    actual = {
        path.name: report_utils._sha256(path)
        for path in paths
        if path.is_file()
    }
    if actual != bindings:
        raise SystemExit(f"source binding mismatch: expected {bindings}, got {actual}")
    return [path.resolve() for path in paths]


def _discover_images(args: argparse.Namespace) -> list[Path]:
    if args.phase == "development":
        return _bound_paths(args.images, DEVELOPMENT_BINDINGS)
    if args.phase == "replay_h15":
        return _bound_paths(args.images, h15.DEVELOPMENT_BINDINGS)
    if args.phase == "replay_h16":
        return _bound_paths(args.images, h16.DEVELOPMENT_BINDINGS)
    if args.phase == "replay_h17":
        return _bound_paths(args.images, h17.DEVELOPMENT_BINDINGS)
    if args.phase == "replay_h18":
        return _bound_paths(args.images, h18.DEVELOPMENT_BINDINGS)
    return h13._discover_sources([args.images])


def _configs(args: argparse.Namespace):
    baseline_init, _, fit_config = h18._configs(args)
    tail_config = ConfidenceTailConfig(
        scale_multiplier=args.tail_scale_multiplier,
        coverage_threshold=args.tail_coverage_threshold,
    )
    return baseline_init, fit_config, tail_config


def _snapshot_sources(output_root: Path) -> list[dict[str, object]]:
    paths = (
        Path(__file__).resolve(),
        ROOT / "src" / "structsplat" / "tail_recovery.py",
        ROOT / "src" / "structsplat" / "fit.py",
        ROOT / "src" / "structsplat" / "render.py",
        ROOT / "scripts" / "experiments" / "hier017_normalization_epsilon.py",
        ROOT / "tasks" / "HIER-019-confidence-gated-self-prior.md",
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


def _quantiles(values: np.ndarray) -> dict[str, float]:
    return {
        f"q{label}": float(np.quantile(values, quantile))
        for label, quantile in (
            ("000", 0.0),
            ("001", 0.001),
            ("010", 0.01),
            ("500", 0.5),
            ("990", 0.99),
            ("999", 0.999),
            ("1000", 1.0),
        )
    }


def _selection(
    baseline: dict[str, float],
    candidate: dict[str, float],
    *,
    activation_count: int,
    finite: bool,
    outside_identity_max_abs: float,
    baseline_parity_max_abs: float,
    repeated_parity_max_abs: float,
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
    clauses = {
        "candidate_finite": finite,
        "outside_activation_bit_exact": outside_identity_max_abs == 0.0,
        "baseline_cold_parity_le_2e_5": baseline_parity_max_abs <= 2e-5,
        "candidate_repeated_parity_le_2e_5": repeated_parity_max_abs <= 2e-5,
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
        if activation_count > 0 and all(clauses.values()) and material
        else BASELINE_MODE
    )
    return selected, clauses, deltas


def _active_quantiles(values: np.ndarray, active: np.ndarray) -> dict[str, float] | None:
    selected = values[active]
    return _quantiles(selected) if selected.size else None


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
    cold_field = GaussianField.load(str(field_path), device=args.device)

    tail_started = time.perf_counter()
    first = render_confidence_gated_self_prior(
        cold_field, fit_config, image.shape[0], image.shape[1], tail_config
    )
    torch.cuda.synchronize()
    tail_seconds = time.perf_counter() - tail_started
    repeated_started = time.perf_counter()
    repeated = render_confidence_gated_self_prior(
        cold_field, fit_config, image.shape[0], image.shape[1], tail_config
    )
    torch.cuda.synchronize()
    repeated_seconds = time.perf_counter() - repeated_started

    cold_baseline = first.baseline.detach().cpu().numpy().astype(np.float32, copy=False)
    candidate = first.candidate.detach().cpu().numpy().astype(np.float32, copy=False)
    prior = first.prior.detach().cpu().numpy().astype(np.float32, copy=False)
    denominator = first.denominator.detach().cpu().numpy().astype(np.float64)
    missing_mass = first.missing_mass.detach().cpu().numpy().astype(np.float64)
    active = first.activation_mask.detach().cpu().numpy().astype(bool, copy=False)
    repeated_candidate = repeated.candidate.detach().cpu().numpy().astype(np.float32, copy=False)
    baseline_parity = float(np.max(np.abs(cold_baseline - baseline_render)))
    repeated_parity = float(np.max(np.abs(repeated_candidate - candidate)))

    metric_started = time.perf_counter()
    candidate_metrics = report_utils._metric_values(
        candidate, image, mask, device=args.device, compute_lpips=args.lpips
    )
    metric_seconds = time.perf_counter() - metric_started
    candidate_subset = _metric_subset(candidate_metrics)
    baseline_subset = _metric_subset(row)
    finite = bool(
        np.isfinite(candidate).all()
        and np.isfinite(prior).all()
        and np.isfinite(denominator).all()
        and np.isfinite(missing_mass).all()
        and all(math.isfinite(value) for value in candidate_subset.values())
    )
    selected_mode, clauses, deltas = _selection(
        baseline_subset,
        candidate_subset,
        activation_count=first.activation_count,
        finite=finite,
        outside_identity_max_abs=first.outside_identity_max_abs,
        baseline_parity_max_abs=baseline_parity,
        repeated_parity_max_abs=repeated_parity,
    )
    selected_metrics = (
        candidate_subset if selected_mode == CANDIDATE_MODE else baseline_subset
    )

    candidate_dir = artifact_dir / "tail_candidate"
    candidate_dir.mkdir(parents=True, exist_ok=False)
    bounds = h15._save_visuals(
        candidate_dir,
        image,
        candidate,
        cold_baseline,
        mask,
        args.error_scale,
    )
    h15.save_error_heatmap(
        str(candidate_dir / "candidate_delta.png"),
        candidate - cold_baseline,
        scale=args.error_scale,
    )

    baseline_error = np.sqrt(
        np.mean((cold_baseline.astype(np.float64) - image.astype(np.float64)) ** 2, axis=2)
    )
    candidate_error = np.sqrt(
        np.mean((candidate.astype(np.float64) - image.astype(np.float64)) ** 2, axis=2)
    )
    error_improvement = baseline_error - candidate_error
    coordinates = np.argwhere(active).astype(np.int32)
    np.savez_compressed(
        candidate_dir / "tail_analysis.npz",
        activation_mask=active,
        activation_yx=coordinates,
        denominator=denominator.astype(np.float32),
        missing_mass=missing_mass.astype(np.float32),
        prior=prior,
        candidate_delta=(candidate - cold_baseline).astype(np.float32),
        baseline_pixel_rmse=baseline_error.astype(np.float32),
        candidate_pixel_rmse=candidate_error.astype(np.float32),
        crop_bounds=np.asarray(bounds, dtype=np.int32),
    )
    active_improvement = error_improvement[active]
    telemetry: dict[str, object] = {
        "schema": REPORT_SCHEMA,
        "status": "diagnostic",
        "semantic_family": "confidence_gated_same_field_prior_v1",
        "scale_multiplier": first.scale_multiplier,
        "coverage_threshold": first.coverage_threshold,
        "normalization_eps": first.normalization_eps,
        "activation_count": first.activation_count,
        "activation_yx": coordinates.tolist(),
        "denominator_quantiles": _quantiles(denominator),
        "missing_mass_quantiles": _quantiles(missing_mass),
        "active_denominator_quantiles": _active_quantiles(denominator, active),
        "active_missing_mass_quantiles": _active_quantiles(missing_mass, active),
        "active_error_improvement_quantiles": (
            _quantiles(active_improvement) if active_improvement.size else None
        ),
        "active_error_improvement_max": (
            float(active_improvement.max()) if active_improvement.size else 0.0
        ),
        "active_error_improvement_min": (
            float(active_improvement.min()) if active_improvement.size else 0.0
        ),
        "outside_identity_max_abs": first.outside_identity_max_abs,
        "baseline_cold_parity_max_abs": baseline_parity,
        "candidate_repeated_parity_max_abs": repeated_parity,
        "candidate_finite": finite,
        "candidate_metrics": candidate_subset,
        "metric_deltas_vs_baseline": deltas,
        "selection_clauses": clauses,
        "selected_mode": selected_mode,
        "selected_metrics": selected_metrics,
        "tail_render_seconds": tail_seconds,
        "tail_repeated_render_seconds": repeated_seconds,
        "tail_metric_seconds": metric_seconds,
        "field_file_sha256_before": field_hash_before,
        "field_file_sha256_after": report_utils._sha256(field_path),
    }
    report_utils._write_json(candidate_dir / "tail_recovery.json", telemetry)

    baseline_single_render = max(float(row["render_seconds"]) / 2.0, 1e-12)
    row.update(
        {
            "tail_semantic_family": telemetry["semantic_family"],
            "tail_scale_multiplier": first.scale_multiplier,
            "tail_coverage_threshold": first.coverage_threshold,
            "tail_activation_count": first.activation_count,
            "tail_activation_yx": coordinates.tolist(),
            "tail_outside_identity_max_abs": first.outside_identity_max_abs,
            "tail_baseline_cold_parity_max_abs": baseline_parity,
            "tail_candidate_repeated_parity_max_abs": repeated_parity,
            "tail_candidate_finite": finite,
            "tail_candidate_metrics": candidate_subset,
            "tail_metric_deltas_vs_baseline": deltas,
            "tail_selection_clauses": clauses,
            "tail_selected_mode": selected_mode,
            "tail_selected_metrics": selected_metrics,
            "tail_render_seconds": tail_seconds,
            "tail_repeated_render_seconds": repeated_seconds,
            "tail_metric_seconds": metric_seconds,
            "tail_render_time_ratio": tail_seconds / baseline_single_render,
            "tail_pipeline_time_ratio": (
                (float(row["pipeline_algorithm_seconds"]) + tail_seconds)
                / max(float(row["pipeline_algorithm_seconds"]), 1e-12)
            ),
            "tail_field_file_sha256_before": field_hash_before,
            "tail_field_file_sha256_after": report_utils._sha256(field_path),
            "tail_active_error_improvement_max": telemetry[
                "active_error_improvement_max"
            ],
            "tail_active_error_improvement_min": telemetry[
                "active_error_improvement_min"
            ],
        }
    )
    report_utils._write_json(artifact_dir / "row.json", row)
    return row


def _control_rows_from_payload(path: Path | None) -> dict[str, dict[str, object]]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        str(row["image"]): row
        for row in payload["rows"]
        if row.get("arm") == CONTROL_ARM
    }


def _selected_pairs(
    rows: list[dict[str, object]],
    external_controls: dict[str, dict[str, object]] | None = None,
) -> list[dict[str, object]]:
    controls = {
        str(row["image"]): row for row in rows if row.get("arm") == CONTROL_ARM
    }
    if external_controls:
        controls.update(external_controls)
    pairs: list[dict[str, object]] = []
    for row in rows:
        if row.get("arm") != DIRECT_ARM:
            continue
        control = controls.get(str(row["image"]))
        if control is None:
            continue
        selected = row["tail_selected_metrics"]
        assert isinstance(selected, dict)
        pairs.append(
            {
                "image": row["image"],
                "selected_mode": row["tail_selected_mode"],
                "activation_count": row["tail_activation_count"],
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
                "n_gaussians": row["n_gaussians"],
                "tail_pipeline_time_ratio": row["tail_pipeline_time_ratio"],
                "tail_render_time_ratio": row["tail_render_time_ratio"],
            }
        )
    return pairs


def _direct_records(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    return [row for row in rows if row.get("arm") == DIRECT_ARM]


def _base_selection_gates(direct: list[dict[str, object]], expected: int) -> dict[str, bool]:
    return {
        "complete_direct_rows": len(direct) == expected,
        "all_exact_count": all(int(row["n_gaussians"]) == 7000 for row in direct),
        "all_field_bytes_unchanged": all(
            row["tail_field_file_sha256_before"] == row["tail_field_file_sha256_after"]
            for row in direct
        ),
        "all_selected_transactions_safe": all(
            row["tail_selected_mode"] == BASELINE_MODE
            or all(bool(value) for value in row["tail_selection_clauses"].values())
            for row in direct
        ),
        "all_outside_activation_bit_exact": all(
            float(row["tail_outside_identity_max_abs"]) == 0.0 for row in direct
        ),
        "all_candidate_finite": all(bool(row["tail_candidate_finite"]) for row in direct),
        "all_tail_parity_le_2e_5": all(
            float(row["tail_baseline_cold_parity_max_abs"]) <= 2e-5
            and float(row["tail_candidate_repeated_parity_max_abs"]) <= 2e-5
            for row in direct
        ),
        "all_selected_mse_noninferior_vs_baseline": all(
            row["tail_selected_mode"] == BASELINE_MODE
            or float(row["tail_metric_deltas_vs_baseline"]["mse_ratio"]) <= 1.0 + 1e-8
            for row in direct
        ),
        "all_selected_pixel_noninferior_vs_baseline": all(
            row["tail_selected_mode"] == BASELINE_MODE
            or float(row["tail_metric_deltas_vs_baseline"]["pixel_max_delta"]) <= 1e-12
            for row in direct
        ),
        "all_selected_patch_noninferior_vs_baseline": all(
            row["tail_selected_mode"] == BASELINE_MODE
            or float(row["tail_metric_deltas_vs_baseline"]["patch7_max_delta"]) <= 1e-12
            for row in direct
        ),
    }


def _pair_gates(pairs: list[dict[str, object]], expected: int) -> dict[str, bool]:
    return {
        "complete_h005_pairs": len(pairs) == expected,
        "all_psnr_gain_vs_h005_ge_2_db": all(
            float(pair["psnr_delta_db"]) >= 2.0 for pair in pairs
        ),
        "all_mse_noninferior_vs_h005": all(
            float(pair["mse_ratio"]) <= 1.0 + 1e-8 for pair in pairs
        ),
        "all_pixel_max_noninferior_vs_h005": all(
            float(pair["pixel_max_delta"]) <= 1e-12 for pair in pairs
        ),
        "all_patch7_max_noninferior_vs_h005": all(
            float(pair["patch7_max_delta"]) <= 1e-12 for pair in pairs
        ),
        "mean_ms_ssim_noninferior_vs_h005": (
            float(np.mean([pair["ms_ssim_delta"] for pair in pairs])) >= -1e-7
            if pairs else False
        ),
        "mean_lpips_noninferior_vs_h005": (
            float(np.mean([pair["lpips_delta"] for pair in pairs])) <= 1e-7
            if pairs else False
        ),
    }


def _decision(
    rows: list[dict[str, object]],
    attempts: list[dict[str, object]],
    args: argparse.Namespace,
) -> dict[str, object]:
    expected = 16 if args.phase == "replay_tests" else 4
    direct = _direct_records(rows)
    external = _control_rows_from_payload(args.control_metrics)
    pairs = _selected_pairs(rows, external)
    gates = {
        **_base_selection_gates(direct, expected),
        **_pair_gates(pairs, expected),
        "complete_attempt_ledger": len(attempts) == (
            expected * 2 if args.phase in ("development", "replay_tests") else expected
        ),
        "zero_failures": all(record.get("status") == "ok" for record in attempts),
        "median_pipeline_time_ratio_le_1_25": (
            float(np.median([row["tail_pipeline_time_ratio"] for row in direct])) <= 1.25
            if direct else False
        ),
        "median_tail_render_ratio_le_5": (
            float(np.median([row["tail_render_time_ratio"] for row in direct])) <= 5.0
            if direct else False
        ),
    }
    if args.phase != "development":
        gates.update(
            {
                "all_ms_ssim_noninferior_vs_h005": all(
                    float(pair["ms_ssim_delta"]) >= -1e-7 for pair in pairs
                ),
                "all_lpips_noninferior_vs_h005": all(
                    float(pair["lpips_delta"]) <= 1e-7 for pair in pairs
                ),
            }
        )
    candidate = all(gates.values())
    selected_count = sum(
        row["tail_selected_mode"] == CANDIDATE_MODE for row in direct
    )
    return {
        "schema": REPORT_SCHEMA,
        "status": "diagnostic",
        "phase": args.phase,
        "gates": gates,
        "selected_vs_h005_pairs": pairs,
        "tail_candidate_selected_count": selected_count,
        "images_with_low_coverage": sum(
            int(row["tail_activation_count"]) > 0 for row in direct
        ),
        "attempt_count": len(attempts),
        "failure_count": sum(record.get("status") != "ok" for record in attempts),
        "numeric_candidates": [CANDIDATE_MODE] if candidate else [],
        "numeric_disposition": (
            "guarded_confidence_tail" if candidate else "no_robust_tail_candidate"
        ),
        "bounded_bank_pass": candidate,
        "visual_review_required": True,
        "interpretation": (
            "Numeric portfolio requires native visual review before further replay."
            if candidate
            else "The guarded portfolio misses a frozen gate; do not advance or retune."
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
                "<td>—</td><td>—</td>"
                f"<td><a href='{artifact}/reconstruction.png'>full</a> · "
                f"<a href='{artifact}/reconstruction_crop.png'>crop</a></td></tr>"
            )
        else:
            candidate = row["tail_candidate_metrics"]
            assert isinstance(candidate, dict)
            selected = escape(str(row["tail_selected_mode"]))
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
                f"<td>{int(row['tail_activation_count'])}</td><td>{selected}</td>"
                f"<td><a href='{artifact}/reconstruction.png'>base</a> · "
                f"<a href='{artifact}/reconstruction_crop.png'>base crop</a> · "
                f"<a href='{artifact}/tail_candidate/reconstruction.png'>tail</a> · "
                f"<a href='{artifact}/tail_candidate/reconstruction_crop.png'>tail crop</a> · "
                f"<a href='{artifact}/tail_candidate/candidate_delta.png'>delta</a></td></tr>"
            )
        cards.append(
            f"<section><h3>{escape(str(row['image']))} — {escape(str(row['arm']))}</h3>"
            f"<a href='{artifact}/source.png'><img src='{artifact}/source.png'></a>"
            f"<a href='{artifact}/reconstruction.png'><img src='{artifact}/reconstruction.png'></a>"
            f"<a href='{artifact}/error.png'><img src='{artifact}/error.png'></a>"
            f"<a href='{artifact}/reconstruction_crop.png'>"
            f"<img src='{artifact}/reconstruction_crop.png'></a>"
            + (
                f"<a href='{artifact}/tail_candidate/reconstruction.png'>"
                f"<img src='{artifact}/tail_candidate/reconstruction.png'></a>"
                f"<a href='{artifact}/tail_candidate/candidate_delta.png'>"
                f"<img src='{artifact}/tail_candidate/candidate_delta.png'></a>"
                if row["arm"] == DIRECT_ARM
                else ""
            )
            + "</section>"
        )
    html = f"""<!doctype html><html><head><meta charset='utf-8'><title>HIER-019</title>
<style>body{{font-family:system-ui;margin:2rem;max-width:1700px}}table{{border-collapse:collapse}}
th,td{{border:1px solid #bbb;padding:.35rem}}img{{max-width:300px;max-height:240px;margin:.25rem}}
section{{border-top:1px solid #aaa;margin-top:1rem}}pre,code{{white-space:pre-wrap}}</style>
</head><body><h1>HIER-019 confidence-gated same-field tail —
{escape(str(decision['phase']))}</h1>
<p>Dirty-source diagnostic. Direct cells show <strong>baseline / candidate</strong>; one selected
mode applies to the whole image and every stored field remains exact N=7,000.</p>
<p><code>{escape(command)}</code></p><p><a href='config.json'>config</a> ·
<a href='decision.json'>decision</a> · <a href='metrics.json'>JSON</a> ·
<a href='metrics.jsonl'>JSONL</a> · <a href='metrics.csv'>CSV</a> ·
<a href='attempts.json'>attempts</a> · <a href='manifest.json'>manifest</a></p>
<h2>Decision</h2><pre>{escape(json.dumps(decision, indent=2, sort_keys=True))}</pre>
<h2>Cells</h2><table><tr><th>image</th><th>arm</th><th>PSNR</th><th>MS-SSIM</th>
<th>LPIPS</th><th>pixel max</th><th>7x7 max</th><th>active px</th><th>selected</th>
<th>visuals</th></tr>{''.join(table)}</table><h2>Visual audit</h2>{''.join(cards)}
</body></html>"""
    (output_root / "index.html").write_text(html, encoding="utf-8")


def _recover(
    args: argparse.Namespace,
    images: list[Path],
    output_root: Path,
    command: str,
) -> bool:
    if args.recover_from is None:
        return False
    source_root = args.recover_from.resolve()
    paths = [source_root / name for name in ("metrics.json", "attempts.json", "decision.json")]
    if not source_root.is_dir() or not all(path.is_file() for path in paths):
        raise SystemExit("recovery source is missing its ledgers")
    rows = json.loads(paths[0].read_text(encoding="utf-8")).get("rows", [])
    attempts = json.loads(paths[1].read_text(encoding="utf-8")).get("attempts", [])
    decision = json.loads(paths[2].read_text(encoding="utf-8"))
    expected = len(DEVELOPMENT_BINDINGS) * 2
    if len(rows) != expected or len(attempts) != expected or any(
        record.get("status") != "ok" for record in attempts
    ):
        raise SystemExit("recovery source is not a complete successful HIER-019 run")
    shutil.copytree(source_root, output_root)
    snapshot = output_root / "recovery_source_snapshot" / Path(__file__).name
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(__file__).resolve(), snapshot)
    decision.update({"recovered_from_complete_raw_run": True, "cell_computation_rerun": False})
    report_utils._write_json(output_root / "decision.json", decision)
    report_utils._write_json(
        output_root / "recovery.json",
        {
            "schema": REPORT_SCHEMA,
            "status": "diagnostic",
            "command": command,
            "source_path": str(source_root),
            "source_metrics_sha256": report_utils._sha256(paths[0]),
            "source_attempts_sha256": report_utils._sha256(paths[1]),
            "source_decision_sha256": report_utils._sha256(paths[2]),
            "cell_computation_rerun": False,
            "recovery_driver_snapshot": str(snapshot.relative_to(output_root)),
            "recovery_driver_sha256": report_utils._sha256(snapshot),
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
    include_control = args.phase in ("development", "replay_tests")
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
        "source_snapshots": _snapshot_sources(output_root),
        "control_metrics": (
            None
            if args.control_metrics is None
            else {
                "path": str(args.control_metrics.resolve()),
                "sha256": report_utils._sha256(args.control_metrics),
            }
        ),
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
            "The target-known transaction is encoder-side model selection, not held-out scoring.",
            "A selected mode bit is not yet integrated into the maintained codec.",
            "CUDA accumulation is numerically, not bit, reproducible.",
        ],
    }
    report_utils._write_json(output_root / "config.json", config)
    rows: list[dict[str, object]] = []
    attempts: list[dict[str, object]] = []

    def record(image_path: Path, arm: str, started: float, error: Exception | None = None) -> None:
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
                raise RuntimeError("HIER-019 requires a generated full-frame mask")
        except Exception as exc:
            if include_control:
                record(image_path, CONTROL_ARM, load_started, exc)
            record(image_path, DIRECT_ARM, load_started, exc)
            continue
        mask = np.ones(image.shape[:2], dtype=bool)
        control_reconstruction = image

        if include_control:
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
