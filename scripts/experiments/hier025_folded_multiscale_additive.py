#!/usr/bin/env python3
"""Run HIER-025's frozen folded multiscale pure-additive diagnostic."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import csv
from hashlib import sha256
from html import escape
import json
import math
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
for candidate in (ROOT, ROOT / "src"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from scripts.experiments import hier022_additive_continuation as h22  # noqa: E402
from scripts.experiments import hier023_unit_gauge_continuation as h23  # noqa: E402
from scripts.experiments import hier024_gauge_geometry_projection as h24  # noqa: E402
from structsplat.config import StructureTensorConfig  # noqa: E402
from structsplat.folded_multiscale_additive import (  # noqa: E402
    FoldedMultiscaleAdditiveConfig,
    FoldedMultiscaleAdditiveResult,
    fit_folded_multiscale_additive,
)
from structsplat.init import build_field  # noqa: E402


REPORT_SCHEMA = "structsplat.hier025_folded_multiscale_additive.diagnostic.v1"
ARMS = (
    "normalized_plain",
    "additive_plain",
    "additive_projected_safe",
    "folded_multiscale_additive",
    "folded_multiscale_projected_safe",
)
PROJECTED_ARMS = frozenset(
    ("additive_projected_safe", "folded_multiscale_projected_safe")
)
FOLDED_ARMS = frozenset(
    ("folded_multiscale_additive", "folded_multiscale_projected_safe")
)
EXCLUDED_FILENAMES = frozenset(
    (
        "0001.png",
        "0343.png",
        "0685.png",
        "0534.png",
        "0002.png",
        "0268.png",
        "0800.png",
        "0571.png",
    )
)
SELECTION_SALT = "HIER-025-v1:"
SOURCE_BINDINGS = {
    "0115.png": "b08214ed8a205d5ff148eb14541de6117f282350bc3e4fc46d2efa8c848073e1",
    "0457.png": "565bb5b65c50abd4b0715b9318851de400cae1475db9c44a138a3bae275d2a05",
    "0229.png": "e985cdadc0861ae47a76ae66a46290b7aa322b4d2596727634b144cb205c2d18",
    "0799.png": "ad42d7e2fe2ee15461e6999e7673a1f96b1be791b4be8c01baca26812f5667db",
}
SELECTION_BINDINGS = {
    "0115.png": "1d9d1f9fc7d952b603b0ea635477bbaba4865650f501f14a9ed7d65cf9095cfe",
    "0457.png": "9071b79f6d8b7cd3f8988fdd5ba475da831141ab173c0d7d885be085666aefd1",
    "0229.png": "c6dd4fde23e17f3572009eaef56ebc1f3adf02ab2b643f628691d6f96a2962e1",
    "0799.png": "fae7587fcfbbd2cd988a7af0b4bef936cb8dab2bd8631e0108dddec6bca42460",
}
COEFFICIENT_LIMIT = 16.0
PARITY_LIMIT = 2e-5


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--max-side", type=int, default=160)
    parser.add_argument("--budgets", type=int, nargs="+", default=[640])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--iters", type=int, default=500)
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--lpips", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--render-chunk", type=int, default=256)
    parser.add_argument("--error-scale", type=float, default=4.0)
    parser.add_argument("--resume", action="store_true")
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    frozen = {
        "max_side": 160,
        "budgets": [640],
        "seeds": [0, 1],
        "iters": 500,
        "arms": list(ARMS),
        "device": "cuda",
        "lpips": True,
        "render_chunk": 256,
    }
    for name, expected in frozen.items():
        if getattr(args, name) != expected:
            raise SystemExit(
                f"frozen HIER-025 protocol requires {name}={expected!r}, "
                f"got {getattr(args, name)!r}"
            )
    if not math.isfinite(args.error_scale) or args.error_scale <= 0.0:
        raise SystemExit("--error-scale must be finite and positive")
    if not args.images.is_dir():
        raise SystemExit(f"image directory does not exist: {args.images}")


def _command() -> str:
    return " ".join(shlex.quote(value) for value in sys.argv)


def _write_json(path: Path, value: object) -> None:
    h22._write_json(path, value)


def _discover_sources(root: Path) -> list[Path]:
    candidates = sorted(
        path
        for path in root.iterdir()
        if path.is_file() and path.name not in EXCLUDED_FILENAMES
    )
    ranked = sorted(
        (sha256(f"{SELECTION_SALT}{path.name}".encode()).hexdigest(), path)
        for path in candidates
    )
    selection = {path.name: digest for digest, path in ranked}
    hashes = {path.name: h22.report_utils._sha256(path) for _, path in ranked}
    if len(candidates) != 4 or selection != SELECTION_BINDINGS or hashes != SOURCE_BINDINGS:
        raise SystemExit(
            "HIER-025 source selection or hash binding differs: "
            f"candidate_count={len(candidates)}, selection={selection}, hashes={hashes}"
        )
    return [path.resolve() for _, path in ranked]


def _snapshot_sources(output_root: Path) -> list[dict[str, object]]:
    paths = (
        Path(__file__).resolve(),
        ROOT / "scripts" / "experiments" / "hier023_unit_gauge_continuation.py",
        ROOT / "scripts" / "experiments" / "hier024_gauge_geometry_projection.py",
        ROOT / "src" / "structsplat" / "folded_multiscale_additive.py",
        ROOT / "src" / "structsplat" / "endpoint_appearance_projection.py",
        ROOT / "src" / "structsplat" / "contraction_refinement.py",
        ROOT / "tests" / "test_folded_multiscale_additive.py",
        ROOT / "tests" / "test_endpoint_appearance_projection.py",
        ROOT / "tasks" / "HIER-025-folded-multiscale-additive.md",
        ROOT / "scripts" / "check_report_bundle.py",
    )
    records = []
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
                "sha256": h22.report_utils._sha256(destination),
            }
        )
    return records


def _folded_config(args: argparse.Namespace) -> FoldedMultiscaleAdditiveConfig:
    return FoldedMultiscaleAdditiveConfig(
        total_gaussians=640,
        coarse_gaussians=16,
        coarse_steps=100,
        detail_steps=300,
        joint_steps=100,
        checkpoint_every=25,
        lr_means=5e-2,
        lr_scales=3e-2,
        lr_rotations=1e-2,
        lr_coefficients=3e-2,
        detail_feature_cap_px=12.0,
        coefficient_abs_limit=COEFFICIENT_LIMIT,
        sigma_cutoff=3.0,
        render_chunk=args.render_chunk,
        renderer="cuda_additive",
    )


def _run_folded(
    target: np.ndarray, seed: int, args: argparse.Namespace, torch
) -> dict[str, object]:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.reset_peak_memory_stats()
    wall_started = time.perf_counter()
    result = fit_folded_multiscale_additive(
        target,
        seed=seed,
        config=_folded_config(args),
        device=args.device,
        verbose=False,
    )
    torch.cuda.synchronize()
    wall_seconds = time.perf_counter() - wall_started
    records = [
        {"step": float(point.step), "psnr_db": point.psnr_db}
        for point in result.trajectory
    ]
    trajectory = h22._normalize_trajectory(records, args.iters)
    with torch.no_grad():
        coverage = h22._coverage_record(
            h22._unit_coverage(
                result.field,
                target.shape[0],
                target.shape[1],
                "cuda_additive",
                args.render_chunk,
            )
        )
    return {
        "field": result.field,
        "expected": result.reconstruction_raw,
        "trajectory": trajectory,
        "hold_psnr_db": None,
        "renderer_calls": result.fit_renderer_calls,
        "normalized_calls": 0,
        "additive_numerator_calls": result.fit_renderer_calls,
        "additive_denominator_calls": 0,
        "renderer_calls_coverage_diagnostic": 1,
        "selected_step": result.selected_steps["joint"],
        "completed": result.completed,
        "method_status": result.status,
        "history": {
            "stages": result.stage_histories,
            "trajectory": result.trajectory_records(),
            "selected_steps": result.selected_steps,
        },
        "optimizer_reset_count": 0,
        "optimizer_reset_step": None,
        "hold_optimizer_reset_count": 0,
        "endpoint_parity": max(
            result.fold_parity_max_abs, result.endpoint_parity_max_abs
        ),
        "semantic_family": "additive_rgb_peak_one_v1",
        "renderer": "cuda_additive",
        "fit_seconds": result.fit_seconds,
        "wall_fit_seconds": wall_seconds,
        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "coverage": coverage,
        "folded_result": result,
    }


def _projection_record(method: dict[str, object]) -> dict[str, object]:
    projection = method["projection_result"]
    if projection is None:
        return {
            "selected_iteration": None,
            "initial_sse": None,
            "final_sse": None,
            "forward_applications": 0,
            "transpose_applications": 0,
            "relative_normal_residual_max": None,
            "adjoint_relative_error": None,
            "initial_operator_parity_max_abs": None,
            "maintained_render_parity_max_abs": None,
            "normal_diagonal_min": None,
            "normal_diagonal_max": None,
            "geometry_exact": True,
            "checkpoints": [],
        }
    receipt = projection.projection
    return {
        "selected_iteration": receipt.selected_iteration,
        "initial_sse": receipt.initial_sse,
        "final_sse": receipt.final_sse,
        "forward_applications": receipt.forward_applications,
        "transpose_applications": receipt.transpose_applications,
        "relative_normal_residual_max": receipt.relative_normal_residual_max,
        "adjoint_relative_error": receipt.adjoint_relative_error,
        "initial_operator_parity_max_abs": receipt.initial_operator_parity_max_abs,
        "maintained_render_parity_max_abs": receipt.maintained_render_parity_max_abs,
        "normal_diagonal_min": receipt.normal_diagonal_min,
        "normal_diagonal_max": receipt.normal_diagonal_max,
        "geometry_exact": projection.geometry_exact,
        "checkpoints": receipt.checkpoint_records(),
    }


def _write_cell(
    *,
    output_root: Path,
    image_path: Path,
    target: np.ndarray,
    raster: dict[str, object],
    seed: int,
    budget: int,
    arm: str,
    initial_field_sha256: str,
    init_seconds: float,
    method: dict[str, object],
    args: argparse.Namespace,
    torch,
) -> dict[str, object]:
    row = h23._write_cell(
        output_root=output_root,
        image_path=image_path,
        target=target,
        raster=raster,
        seed=seed,
        budget=budget,
        arm=arm,
        initial_field_sha256=initial_field_sha256,
        init_seconds=init_seconds,
        method=method,
        args=args,
        torch=torch,
    )
    artifact_dir = output_root / str(row["artifact_dir"])
    incoming_path = artifact_dir / "incoming.field.gaussian.npz"
    proposal_path = artifact_dir / "proposal.field.gaussian.npz"
    method["incoming_field"].save(str(incoming_path))
    method["proposal_field"].save(str(proposal_path))
    projection = _projection_record(method)
    _write_json(artifact_dir / "projection_history.json", projection)
    folded: FoldedMultiscaleAdditiveResult | None = method.get("folded_result")
    geometry_record = (
        {
            "coarse_count": folded.coarse_count,
            "detail_count": folded.detail_count,
            "coarse_geometry_exact": folded.coarse_geometry_exact,
            "coarse_geometry_digest": folded.coarse_geometry_digest,
            "fold_parity_max_abs": folded.fold_parity_max_abs,
            "endpoint_parity_max_abs": folded.endpoint_parity_max_abs,
            "training_mask_removed": folded.training_mask_removed,
            "lowpass_sha256": folded.lowpass_sha256,
            "residual_sha256": folded.residual_sha256,
            "coarse_field_digest": folded.coarse_field_digest,
            "detail_field_digest": folded.detail_field_digest,
            "folded_initial_digest": folded.folded_initial_digest,
            "endpoint_field_digest": folded.endpoint_field_digest,
        }
        if folded is not None
        else {}
    )
    _write_json(artifact_dir / "geometry_history.json", geometry_record)
    _write_json(
        artifact_dir / "config.json",
        {
            "schema": REPORT_SCHEMA,
            "arm": arm,
            "seed": seed,
            "budget": budget,
            "init": (
                asdict(h22._init_config(budget, seed))
                if arm not in FOLDED_ARMS
                else None
            ),
            "fit": (
                asdict(h23._fit_config(args, arm))
                if arm in ("normalized_plain", "additive_plain", "additive_projected_safe")
                else asdict(_folded_config(args))
            ),
            "projection": (
                asdict(h24._projection_config(args)) if arm in PROJECTED_ARMS else None
            ),
            "safety": asdict(h24._safety_config()) if arm in PROJECTED_ARMS else None,
        },
    )
    row.update(
        {
            "schema": REPORT_SCHEMA,
            "selection_sha256": SELECTION_BINDINGS[image_path.name],
            "selected_lambda": None if arm == "normalized_plain" else 0.0,
            "projection_applied": method["projection_applied"],
            "projection_selected": method["projection_selected"],
            "projection_reason": method["projection_reason"],
            "projection_clauses": method["projection_clauses"],
            "projection_seconds": method["projection_seconds"],
            "projection_metric_seconds": method["projection_metric_seconds"],
            "incoming_field_digest": method["incoming_field_digest"],
            "proposal_field_digest": method["proposal_field_digest"],
            "final_field_digest": method["final_field_digest"],
            "incoming_field_file_sha256": h22.report_utils._sha256(incoming_path),
            "proposal_field_file_sha256": h22.report_utils._sha256(proposal_path),
            "projection_selected_iteration": projection["selected_iteration"],
            "projection_initial_sse": projection["initial_sse"],
            "projection_final_sse": projection["final_sse"],
            "projection_forward_applications": projection["forward_applications"],
            "projection_transpose_applications": projection["transpose_applications"],
            "projection_relative_normal_residual_max": projection[
                "relative_normal_residual_max"
            ],
            "projection_adjoint_relative_error": projection["adjoint_relative_error"],
            "projection_initial_operator_parity_max_abs": projection[
                "initial_operator_parity_max_abs"
            ],
            "projection_maintained_render_parity_max_abs": projection[
                "maintained_render_parity_max_abs"
            ],
            "projection_geometry_exact": projection["geometry_exact"],
            "coarse_count": 0 if folded is None else folded.coarse_count,
            "detail_count": budget if folded is None else folded.detail_count,
            "coarse_geometry_exact": True if folded is None else folded.coarse_geometry_exact,
            "training_mask_removed": True if folded is None else folded.training_mask_removed,
            "fold_parity_max_abs": 0.0 if folded is None else folded.fold_parity_max_abs,
            "folded_endpoint_parity_max_abs": (
                0.0 if folded is None else folded.endpoint_parity_max_abs
            ),
            "observer_renderer_calls": (
                0 if folded is None else folded.observer_renderer_calls
            ),
            "diagnostic_renderer_calls_fit": (
                0 if folded is None else folded.diagnostic_renderer_calls
            ),
            "lowpass_sha256": None if folded is None else folded.lowpass_sha256,
            "residual_sha256": None if folded is None else folded.residual_sha256,
            "coarse_field_digest": None if folded is None else folded.coarse_field_digest,
            "detail_field_digest": None if folded is None else folded.detail_field_digest,
            "folded_initial_digest": None if folded is None else folded.folded_initial_digest,
            "folded_endpoint_digest": None if folded is None else folded.endpoint_field_digest,
            "coarse_geometry_digest": (
                None if folded is None else folded.coarse_geometry_digest
            ),
            "level_payload_present": any(
                token in key.lower()
                for key in row["field_npz_keys"]
                for token in ("background", "level", "residual", "lowpass", "scaler")
            ),
        }
    )
    for prefix, values in (
        ("incoming", method["incoming_selection_metrics"]),
        ("proposal", method["proposal_selection_metrics"]),
    ):
        for key in ("raw_mse", "ms_ssim", "lpips", "pixel_max", "patch7_max"):
            row[f"{prefix}_{key}"] = None if values is None else values[key]
    row["pipeline_algorithm_seconds"] = float(row["pipeline_algorithm_seconds"]) + float(
        method["projection_seconds"]
    )
    row["total_seconds"] = float(row["total_seconds"]) + float(
        method["projection_seconds"]
    ) + float(method["projection_metric_seconds"])
    _write_json(artifact_dir / "row.json", row)
    return row


def _write_tables(output_root: Path, rows: list[dict[str, object]]) -> None:
    _write_json(
        output_root / "metrics.json",
        {"schema": REPORT_SCHEMA, "status": "diagnostic", "rows": rows},
    )
    with (output_root / "metrics.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    columns = sorted({key for row in rows for key in row})
    with (output_root / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _mean(rows: list[dict[str, object]], key: str) -> float:
    return float(np.mean([float(row[key]) for row in rows]))


def _decision(rows: list[dict[str, object]]) -> dict[str, object]:
    expected_count = len(SOURCE_BINDINGS) * 2
    by_arm = {arm: [row for row in rows if row["arm"] == arm] for arm in ARMS}
    complete = all(len(by_arm[arm]) == expected_count for arm in ARMS)
    aggregates = {
        arm: {
            "cell_count": len(by_arm[arm]),
            "mean_psnr_db": _mean(by_arm[arm], "psnr_db") if by_arm[arm] else None,
            "mean_ms_ssim": _mean(by_arm[arm], "ms_ssim") if by_arm[arm] else None,
            "mean_lpips": _mean(by_arm[arm], "lpips") if by_arm[arm] else None,
            "mean_pixel_max": _mean(by_arm[arm], "artifact_pixel_rmse_max")
            if by_arm[arm]
            else None,
            "mean_patch7_max": _mean(by_arm[arm], "artifact_patch_rmse_max_7")
            if by_arm[arm]
            else None,
            "mean_psnr_auc": _mean(by_arm[arm], "psnr_auc_attempted_step")
            if by_arm[arm]
            else None,
            "mean_fit_seconds": _mean(by_arm[arm], "fit_seconds")
            if by_arm[arm]
            else None,
            "mean_projection_seconds": _mean(by_arm[arm], "projection_seconds")
            if by_arm[arm]
            else None,
            "mean_renderer_calls": _mean(by_arm[arm], "renderer_calls_fit")
            if by_arm[arm]
            else None,
            "projection_selected_count": sum(
                bool(row["projection_selected"]) for row in by_arm[arm]
            ),
        }
        for arm in ARMS
    }
    gates: dict[str, bool] = {"all_cells_present": complete}
    if complete:
        candidate = by_arm["folded_multiscale_projected_safe"]
        folded = by_arm["folded_multiscale_additive"]
        additive_projected = by_arm["additive_projected_safe"]
        additive = by_arm["additive_plain"]
        normalized = by_arm["normalized_plain"]
        key = lambda row: (row["image"], row["seed"])
        additive_projected_by_key = {key(row): row for row in additive_projected}
        gates.update(
            {
                "candidate_endpoint_integrity": all(
                    row["completed"]
                    and row["method_status"] == "completed"
                    and row["selected_lambda"] == 0.0
                    and row["n_gaussians"] == row["target_gaussians"] == 640
                    and row["coarse_count"] == 16
                    and row["detail_count"] == 624
                    and row["coarse_geometry_exact"]
                    and row["training_mask_removed"]
                    and float(row["fold_parity_max_abs"]) <= PARITY_LIMIT
                    and float(row["folded_endpoint_parity_max_abs"]) <= PARITY_LIMIT
                    and row["finite_reconstruction"]
                    and float(row["coefficient_abs_max"]) <= COEFFICIENT_LIMIT
                    and float(row["maintained_render_parity_max_abs"]) <= PARITY_LIMIT
                    and row["projection_geometry_exact"]
                    and not row["mass_payload_present"]
                    and not row["denominator_payload_present"]
                    and not row["optimizer_payload_present"]
                    and not row["auxiliary_rgb_payload_present"]
                    and not row["level_payload_present"]
                    for row in candidate
                ),
                "projection_transactions_fail_closed": all(
                    (
                        row["projection_selected"]
                        and all(row["projection_clauses"].values())
                    )
                    or (
                        not row["projection_selected"]
                        and row["final_field_digest"] == row["incoming_field_digest"]
                    )
                    for arm in PROJECTED_ARMS
                    for row in by_arm[arm]
                ),
                "unprojected_folded_at_least_0p05_db_above_additive": (
                    _mean(folded, "psnr_db") >= _mean(additive, "psnr_db") + 0.05
                ),
                "candidate_at_least_0p10_db_above_projected_additive": (
                    _mean(candidate, "psnr_db")
                    >= _mean(additive_projected, "psnr_db") + 0.10
                ),
                "closes_half_positive_normalized_projected_additive_gap": (
                    _mean(candidate, "psnr_db") - _mean(additive_projected, "psnr_db")
                    >= 0.5
                    * max(
                        0.0,
                        _mean(normalized, "psnr_db")
                        - _mean(additive_projected, "psnr_db"),
                    )
                ),
                "candidate_mean_ms_ssim_noninferior": (
                    _mean(candidate, "ms_ssim") >= _mean(additive_projected, "ms_ssim")
                ),
                "candidate_mean_lpips_noninferior": (
                    _mean(candidate, "lpips") <= _mean(additive_projected, "lpips")
                ),
                "candidate_mean_pixel_max_noninferior": (
                    _mean(candidate, "artifact_pixel_rmse_max")
                    <= _mean(additive_projected, "artifact_pixel_rmse_max")
                ),
                "candidate_mean_patch7_max_noninferior": (
                    _mean(candidate, "artifact_patch_rmse_max_7")
                    <= _mean(additive_projected, "artifact_patch_rmse_max_7")
                ),
                "all_candidate_lpips_within_projected_additive_plus_0p01": all(
                    float(row["lpips"])
                    <= float(additive_projected_by_key[key(row)]["lpips"]) + 0.01
                    for row in candidate
                ),
                "all_candidate_local_max_within_projected_additive_plus_0p005": all(
                    float(row["artifact_pixel_rmse_max"])
                    <= float(
                        additive_projected_by_key[key(row)]["artifact_pixel_rmse_max"]
                    )
                    + 0.005
                    and float(row["artifact_patch_rmse_max_7"])
                    <= float(
                        additive_projected_by_key[key(row)][
                            "artifact_patch_rmse_max_7"
                        ]
                    )
                    + 0.005
                    for row in candidate
                ),
                "candidate_full_target_auc_exceeds_additive": (
                    _mean(candidate, "psnr_auc_attempted_step")
                    > _mean(additive, "psnr_auc_attempted_step")
                ),
            }
        )
    numeric_pass = bool(gates and all(gates.values()))
    if not complete:
        failure_class = "incomplete_execution"
    elif not gates.get("candidate_endpoint_integrity", False):
        failure_class = "endpoint_integrity"
    elif not gates.get("projection_transactions_fail_closed", False):
        failure_class = "selection_integrity"
    elif not numeric_pass:
        failure_class = "basis_geometry"
    else:
        failure_class = None
    return {
        "schema": REPORT_SCHEMA,
        "status": "diagnostic",
        "phase": "development",
        "candidate_arm": "folded_multiscale_projected_safe",
        "coefficient_solve_control": "additive_projected_safe",
        "aggregates": aggregates,
        "gates": gates,
        "numeric_pass": numeric_pass,
        "phase_c_numeric_eligible": numeric_pass,
        "phase_c_authorized": False,
        "visual_review": "pending",
        "overall_pass": False,
        "failure_class_if_numeric": failure_class,
        "formal_claim_ready": False,
        "interpretation": (
            "Numeric gates pass; native visual audit is required before Phase C."
            if numeric_pass
            else "The frozen folded-basis gate failed; retain the bank and do not tune it."
        ),
    }


def _write_report(
    output_root: Path, rows: list[dict[str, object]], decision: dict[str, object]
) -> None:
    table_rows = []
    cards = []
    for row in rows:
        artifact = escape(str(row["artifact_dir"]))
        table_rows.append(
            "<tr>"
            f"<td>{escape(str(row['image']))}</td><td>{int(row['seed'])}</td>"
            f"<td>{escape(str(row['arm']))}</td><td>{float(row['psnr_db']):.3f}</td>"
            f"<td>{float(row['ms_ssim']):.5f}</td><td>{float(row['lpips']):.5f}</td>"
            f"<td>{float(row['artifact_pixel_rmse_max']):.4f}</td>"
            f"<td>{float(row['artifact_patch_rmse_max_7']):.4f}</td>"
            f"<td>{'yes' if row['projection_selected'] else 'no'}</td>"
            f"<td>{escape(str(row['projection_reason']))}</td>"
            f"<td><a href='{artifact}/reconstruction.png'>full</a> · "
            f"<a href='{artifact}/reconstruction_crop.png'>crop</a> · "
            f"<a href='{artifact}/error.png'>error</a> · "
            f"<a href='{artifact}/learning_curve.svg'>curve</a></td></tr>"
        )
        cards.append(
            f"<section><h3>{escape(str(row['image']))} · seed {int(row['seed'])} · "
            f"{escape(str(row['arm']))}</h3>"
            f"<a href='{artifact}/source.png'><img src='{artifact}/source.png'></a>"
            f"<a href='{artifact}/reconstruction.png'><img src='{artifact}/reconstruction.png'></a>"
            f"<a href='{artifact}/error.png'><img src='{artifact}/error.png'></a>"
            f"<a href='{artifact}/reconstruction_crop.png'>"
            f"<img src='{artifact}/reconstruction_crop.png'></a></section>"
        )
    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>HIER-025 folded multiscale additive</title><style>
body{{font-family:system-ui;margin:2rem;max-width:1800px}}table{{border-collapse:collapse}}
th,td{{border:1px solid #bbb;padding:.35rem}}img{{max-width:300px;max-height:240px;margin:.25rem}}
section{{border-top:1px solid #aaa;margin-top:1rem}}pre,code{{white-space:pre-wrap}}
</style></head><body><h1>HIER-025 folded multiscale residual Gaussian sum</h1>
<p><strong>Consumed development diagnostic.</strong> Historically consumed sources, dirty code,
and producer review prohibit confirmation, semantic/default, codec/rate, or novelty claims.</p>
<p><code>{escape(_command())}</code></p>
<p><a href="config.json">config</a> · <a href="decision.json">decision</a> ·
<a href="metrics.json">JSON</a> · <a href="metrics.jsonl">JSONL</a> ·
<a href="metrics.csv">CSV</a> · <a href="attempts.json">attempts</a> ·
<a href="manifest.json">manifest</a></p>
<h2>Decision</h2><pre>{escape(json.dumps(decision, indent=2, sort_keys=True))}</pre>
<h2>Cells</h2><table><tr><th>image</th><th>seed</th><th>arm</th><th>PSNR</th>
<th>MS-SSIM</th><th>LPIPS</th><th>pixel max</th><th>7x7 max</th><th>projection</th>
<th>transaction</th><th>artifacts</th></tr>{''.join(table_rows)}</table>
<h2>Visual audit</h2>{''.join(cards)}</body></html>"""
    (output_root / "index.html").write_text(html, encoding="utf-8")


def _write_manifest(output_root: Path) -> None:
    files = []
    for path in sorted(output_root.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            files.append(
                {
                    "path": str(path.relative_to(output_root)),
                    "bytes": path.stat().st_size,
                    "sha256": h22.report_utils._sha256(path),
                }
            )
    _write_json(
        output_root / "manifest.json",
        {"schema": REPORT_SCHEMA, "status": "diagnostic", "files": files},
    )


def main() -> None:
    args = _parser().parse_args()
    _validate_args(args)
    if (args.out / "COMPLETED").is_file():
        raise SystemExit(f"completed HIER-025 bundle is immutable: {args.out}")
    if args.out.exists() and any(args.out.iterdir()) and not args.resume:
        raise SystemExit(f"output directory is non-empty; use --resume: {args.out}")
    args.out.mkdir(parents=True, exist_ok=True)

    import torch

    if not torch.cuda.is_available():
        raise SystemExit("frozen HIER-025 protocol requires CUDA")
    sources = _discover_sources(args.images)
    _write_json(args.out / "environment.json", h22._environment(torch))
    snapshots = _snapshot_sources(args.out)
    _write_json(
        args.out / "config.json",
        {
            "schema": REPORT_SCHEMA,
            "status": "diagnostic",
            "command": _command(),
            "git": h22._git_record(),
            "source_snapshots": snapshots,
            "source_selection": {
                "salt": SELECTION_SALT,
                "excluded": sorted(EXCLUDED_FILENAMES),
                "remaining_candidate_count": 4,
                "selection_bindings": SELECTION_BINDINGS,
                "source_bindings": SOURCE_BINDINGS,
                "historically_consumed": True,
            },
            "arguments": vars(args),
            "baseline_init": asdict(h22._init_config(640, 0)),
            "structure_tensor": asdict(StructureTensorConfig()),
            "folded_multiscale": asdict(_folded_config(args)),
            "projection": asdict(h24._projection_config(args)),
            "safety": asdict(h24._safety_config()),
            "paired_fit_reuse": (
                "additive_plain/additive_projected_safe share one exact fit; "
                "folded_multiscale_additive/folded_multiscale_projected_safe share one exact fit"
            ),
            "procedural_preflight": {
                "fixtures": ["constant", "ramp", "edge", "checker", "isolated_detail"],
                "integrity_pass": True,
                "quality_used_for_selection": False,
                "observed_psnr_delta_range_db": [-6.138814, -0.267108],
            },
        },
    )
    with (args.out / "git.diff").open("wb") as handle:
        subprocess.run(
            ["git", "diff", "--binary", "HEAD"],
            cwd=ROOT,
            check=False,
            stdout=handle,
        )
    (args.out / "NATURAL_STARTED").write_text(
        "HIER-025 source selection consumed; no in-place tuning.\n", encoding="utf-8"
    )

    rows: list[dict[str, object]] = []
    attempts: list[dict[str, object]] = []
    metrics_path = args.out / "metrics.json"
    attempts_path = args.out / "attempts.json"
    if args.resume and metrics_path.is_file():
        rows = json.loads(metrics_path.read_text(encoding="utf-8")).get("rows", [])
    if args.resume and attempts_path.is_file():
        attempts = json.loads(attempts_path.read_text(encoding="utf-8")).get("attempts", [])
    row_keys = {(row["image"], row["seed"], row["arm"]) for row in rows}
    tensor_config = StructureTensorConfig()
    for image_path in sources:
        target, mask, raster = h22.report_utils._load_evaluation_raster(
            image_path, None, max_side=args.max_side, mask_threshold=0.5
        )
        if mask is not None:
            raise RuntimeError("HIER-025 requires an unmasked full-frame source")
        for seed in args.seeds:
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            init_started = time.perf_counter()
            initial = build_field(
                target,
                h22._init_config(args.budgets[0], seed),
                tensor_config,
                device=args.device,
            )
            init_seconds = time.perf_counter() - init_started
            initial_path = (
                args.out / "initial_fields" / f"{image_path.stem}__s{seed}__n640.npz"
            )
            initial_path.parent.mkdir(parents=True, exist_ok=True)
            if not initial_path.exists():
                initial.save(str(initial_path))
            initial_sha = h22.report_utils._sha256(initial_path)
            methods: dict[str, dict[str, object]] = {}
            fit_error = None
            try:
                methods["normalized_plain"] = h23._run_method(
                    initial, target, "normalized_plain", args, torch
                )
                methods["additive_plain"] = h23._run_method(
                    initial, target, "additive_plain", args, torch
                )
                methods["folded_multiscale_additive"] = _run_folded(
                    target, seed, args, torch
                )
            except Exception as exc:
                fit_error = exc
            for arm in args.arms:
                stable_key = (image_path.stem, seed, arm)
                if stable_key in row_keys:
                    continue
                cell_started = time.perf_counter()
                try:
                    if fit_error is not None:
                        raise RuntimeError(f"paired fit failed: {fit_error}")
                    if arm == "additive_projected_safe":
                        method = h24._project_method(
                            methods["additive_plain"], target, args
                        )
                    elif arm == "folded_multiscale_projected_safe":
                        method = h24._project_method(
                            methods["folded_multiscale_additive"], target, args
                        )
                    else:
                        method = h24._base_method(methods[arm])
                    folded = method.get("folded_result")
                    method_initial_sha = (
                        folded.folded_initial_digest
                        if folded is not None
                        else initial_sha
                    )
                    row = _write_cell(
                        output_root=args.out,
                        image_path=image_path,
                        target=target,
                        raster=raster,
                        seed=seed,
                        budget=args.budgets[0],
                        arm=arm,
                        initial_field_sha256=method_initial_sha,
                        init_seconds=0.0 if folded is not None else init_seconds,
                        method=method,
                        args=args,
                        torch=torch,
                    )
                    rows.append(row)
                    row_keys.add(stable_key)
                    attempts.append(
                        {
                            "image": image_path.stem,
                            "seed": seed,
                            "arm": arm,
                            "status": "ok",
                            "elapsed_seconds": time.perf_counter() - cell_started,
                        }
                    )
                except Exception as exc:
                    attempts.append(
                        {
                            "image": image_path.stem,
                            "seed": seed,
                            "arm": arm,
                            "status": "error",
                            "error": f"{type(exc).__name__}: {exc}"[:1000],
                            "elapsed_seconds": time.perf_counter() - cell_started,
                        }
                    )
                finally:
                    _write_tables(args.out, rows)
                    _write_json(
                        attempts_path,
                        {"schema": REPORT_SCHEMA, "status": "diagnostic", "attempts": attempts},
                    )
                    torch.cuda.empty_cache()

    decision = _decision(rows)
    _write_json(args.out / "decision.json", decision)
    _write_report(args.out, rows, decision)
    (args.out / "COMPLETED").write_text(
        "HIER-025 consumed development diagnostic; do not overwrite.\n", encoding="utf-8"
    )
    _write_manifest(args.out)
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
