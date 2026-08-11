#!/usr/bin/env python3
"""Run HIER-023's frozen mass-free unit-gauge continuation diagnostic."""
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
from structsplat.config import FitConfig, StructureTensorConfig  # noqa: E402
from structsplat.fit import fit  # noqa: E402
from structsplat.gaussians import GaussianField  # noqa: E402
from structsplat.init import build_field  # noqa: E402
from structsplat.unit_gauge_continuation import (  # noqa: E402
    UnitGaugeContinuationConfig,
    UnitGaugeContinuationResult,
    fit_unit_gauge_continuation,
)


REPORT_SCHEMA = "structsplat.hier023_unit_gauge_continuation.diagnostic.v1"
ARMS = (
    "normalized_plain",
    "additive_plain",
    "gauge_locked_no_reset",
    "gauge_locked_endpoint_reset",
)
CONTINUATION_ARMS = ARMS[2:]
SELECTION_SALT = "HIER-023-v1:"
SOURCE_BINDINGS = {
    "0001.png": "cdb20d7a462744c269d8e197f735c7bc42e7cda367a940a9b7bc27803b1c8619",
    "0343.png": "f70f775deb82a5744fae0640b5b095e35374f7228893dead5750a4b9d7ef8781",
    "0685.png": "c42e9a8e92f57ed8ebff3ba247c7578aa85b59785021123f673c56d895e63364",
    "0534.png": "c605f2a1092cafc85280d618eb55344c58830313dc75b0469a8f7321f11aa4d3",
}
SELECTION_BINDINGS = {
    "0001.png": "10083e2041d2c0bc6f03e615d1d0492274e07ad4444db36ab98a0bb7f1598aeb",
    "0343.png": "2a383550912212efa3c76a17623f2e2ed033d2b4197e86ba191d7bcf1c65f899",
    "0685.png": "2a826a2dab4101c14069a055da21800cc3493803493d3abb92d623c80d458528",
    "0534.png": "3171ef416d1a74d0fa4e69988adb20492a3ab0b860d247a11d397749b62f4c15",
}
CHECKPOINT_EVERY = 25
FEATURE_CAP_PX = 12.0
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
                f"frozen HIER-023 protocol requires {name}={expected!r}, "
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
    candidates = sorted(path for path in root.iterdir() if path.is_file())
    ranked = sorted(
        (sha256(f"{SELECTION_SALT}{path.name}".encode()).hexdigest(), path)
        for path in candidates
    )
    selected = ranked[:4]
    selection = {path.name: digest for digest, path in selected}
    hashes = {path.name: h22.report_utils._sha256(path) for _, path in selected}
    if selection != SELECTION_BINDINGS or hashes != SOURCE_BINDINGS:
        raise SystemExit(
            "HIER-023 source selection or hash binding differs: "
            f"selection={selection}, hashes={hashes}"
        )
    return [path.resolve() for _, path in selected]


def _snapshot_sources(output_root: Path) -> list[dict[str, object]]:
    paths = (
        Path(__file__).resolve(),
        ROOT / "scripts" / "experiments" / "hier022_additive_continuation.py",
        ROOT / "src" / "structsplat" / "unit_gauge_continuation.py",
        ROOT / "tests" / "test_unit_gauge_continuation.py",
        ROOT / "tasks" / "HIER-023-unit-gauge-additive-continuation.md",
        ROOT / "docs" / "research" / "2026-08-11-pure-gaussian-additive-continuation.md",
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


def _fit_config(args: argparse.Namespace, arm: str) -> FitConfig:
    renderer = "cuda" if arm == "normalized_plain" else "cuda_additive"
    return FitConfig(
        iters=args.iters,
        lr_means=5e-2,
        lr_scales=3e-2,
        lr_rot=1e-2,
        lr_color=3e-2,
        lr_opacity=1e-2,
        optimizer="adam",
        pixel_loss="l1",
        ssim_weight=0.3,
        log_every=CHECKPOINT_EVERY,
        checkpoint_policy="best_psnr_final_count",
        sigma_cutoff=3.0,
        support_fade=False,
        aa_dilation=0.0,
        render_chunk=args.render_chunk,
        renderer=renderer,
        color_basis="constant",
        compute_lpips=False,
        max_gaussians=args.budgets[0],
    )


def _continuation_config(args: argparse.Namespace, arm: str) -> UnitGaugeContinuationConfig:
    return UnitGaugeContinuationConfig(
        steps=args.iters,
        checkpoint_every=CHECKPOINT_EVERY,
        lr_means=5e-2,
        lr_scales=3e-2,
        lr_rotations=1e-2,
        lr_coefficients=3e-2,
        pixel_loss="l1",
        ssim_weight=0.3,
        normalization_eps=1e-8,
        coefficient_abs_limit=COEFFICIENT_LIMIT,
        min_scale_px=0.35,
        sigma_cutoff=3.0,
        aa_dilation=0.0,
        render_chunk=args.render_chunk,
        normalized_renderer="cuda",
        additive_renderer="cuda_additive",
        support_fade=False,
        reset_optimizer_at_endpoint=arm == "gauge_locked_endpoint_reset",
    )


def _trajectory_continuation(
    result: UnitGaugeContinuationResult, steps: int
) -> list[dict[str, float]]:
    records = [
        {"step": float(checkpoint.step), "psnr_db": checkpoint.raw_psnr_db}
        for checkpoint in result.checkpoints
    ]
    return h22._normalize_trajectory(records, steps)


def _step_value(trajectory: list[dict[str, float]], step: int) -> float:
    matches = [row["psnr_db"] for row in trajectory if int(row["step"]) == step]
    if len(matches) != 1:
        raise RuntimeError(f"trajectory does not contain exactly one step {step}")
    return float(matches[0])


def _selected_coverage(result: UnitGaugeContinuationResult) -> dict[str, float]:
    selected = next(checkpoint for checkpoint in result.checkpoints if checkpoint.selected)
    return {
        "coverage_loss": selected.coverage_loss,
        "denominator_min": selected.denominator_min,
        "denominator_q01": selected.denominator_q01,
        "denominator_q05": selected.denominator_q05,
        "denominator_q50": selected.denominator_q50,
        "denominator_q95": selected.denominator_q95,
        "denominator_q99": selected.denominator_q99,
        "denominator_max": selected.denominator_max,
    }


def _run_method(
    initial: GaussianField,
    target: np.ndarray,
    arm: str,
    args: argparse.Namespace,
    torch,
) -> dict[str, object]:
    target_tensor = torch.as_tensor(target, device=args.device, dtype=torch.float32)
    torch.cuda.reset_peak_memory_stats()
    wall_started = time.perf_counter()
    if arm in ("normalized_plain", "additive_plain"):
        config = _fit_config(args, arm)
        hold_values: list[float] = []

        def observe_hold(field_view, iteration: int, _loss: float) -> None:
            if iteration != 175:
                return
            observed = h22._field_render(
                field_view,
                target.shape[0],
                target.shape[1],
                config.renderer,
                args.render_chunk,
            )
            mse = float((observed - target_tensor).square().mean().cpu())
            hold_values.append(-10.0 * math.log10(max(mse, 1e-12)))

        result = fit(
            initial.detached(),
            target_tensor,
            config,
            verbose=False,
            iteration_observer=observe_hold,
            observer_every=175,
        )
        torch.cuda.synchronize()
        if len(hold_values) != 1:
            raise RuntimeError("read-only baseline observer did not capture step 175 exactly once")
        field = result["field"]
        expected = result["render"].detach().cpu().numpy().astype(np.float32, copy=False)
        trajectory = h22._trajectory_baseline(result, args.iters)
        renderer_calls = (
            args.iters
            + len(result["checkpoint_history"]["iter"])
            + int(result["selected_from_checkpoint"])
            + 1
        )
        normalized_calls = renderer_calls if arm == "normalized_plain" else 0
        numerator_calls = renderer_calls if arm == "additive_plain" else 0
        denominator_calls = 0
        selected_step = int(result["selected_iter"])
        completed = bool(result["iterations_run"] == args.iters)
        status = "completed" if completed else "incomplete"
        history = {
            "history": result["history"],
            "checkpoint_history": result["checkpoint_history"],
        }
        reset_count = 0
        reset_step = None
        hold_reset_count = 0
        endpoint_parity = 0.0
        renderer = config.renderer
        semantic = (
            "normalized_weighted_sum_v1"
            if arm == "normalized_plain"
            else "additive_rgb_peak_one_v1"
        )
        fit_seconds = float(result["fit_seconds"])
        with torch.no_grad():
            coverage = h22._coverage_record(
                h22._unit_coverage(
                    field,
                    target.shape[0],
                    target.shape[1],
                    "cuda_additive",
                    args.render_chunk,
                )
            )
        coverage_calls = 1
        hold_psnr_db = hold_values[0]
    else:
        config = _continuation_config(args, arm)
        result = fit_unit_gauge_continuation(
            initial.detached(), target, config=config, verbose=False
        )
        torch.cuda.synchronize()
        field = result.field
        expected = result.reconstruction_raw
        trajectory = _trajectory_continuation(result, args.iters)
        renderer_calls = result.renderer_calls
        normalized_calls = result.normalized_calls
        numerator_calls = result.additive_numerator_calls
        denominator_calls = result.additive_denominator_calls
        selected_step = result.selected_step
        completed = result.completed
        status = result.status
        history = result.checkpoint_records()
        reset_count = result.optimizer_reset_count
        reset_step = result.optimizer_reset_step
        hold_checkpoint = next(item for item in result.checkpoints if item.step == 175)
        hold_reset_count = hold_checkpoint.optimizer_reset_count
        endpoint_parity = result.endpoint_parity_max_abs
        renderer = config.additive_renderer
        semantic = "additive_rgb_peak_one_v1"
        fit_seconds = result.elapsed_seconds
        coverage = _selected_coverage(result)
        coverage_calls = 0
        hold_psnr_db = _step_value(trajectory, 175)
    peak = int(torch.cuda.max_memory_allocated())
    return {
        "field": field,
        "expected": expected,
        "trajectory": trajectory,
        "hold_psnr_db": hold_psnr_db,
        "renderer_calls": renderer_calls,
        "normalized_calls": normalized_calls,
        "additive_numerator_calls": numerator_calls,
        "additive_denominator_calls": denominator_calls,
        "renderer_calls_coverage_diagnostic": coverage_calls,
        "selected_step": selected_step,
        "completed": completed,
        "method_status": status,
        "history": history,
        "optimizer_reset_count": reset_count,
        "optimizer_reset_step": reset_step,
        "hold_optimizer_reset_count": hold_reset_count,
        "endpoint_parity": endpoint_parity,
        "semantic_family": semantic,
        "renderer": renderer,
        "fit_seconds": fit_seconds,
        "wall_fit_seconds": time.perf_counter() - wall_started,
        "peak_cuda_allocated_bytes": peak,
        "coverage": coverage,
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
    artifact_dir = output_root / "artifacts" / f"{image_path.stem}__s{seed}__n{budget}__{arm}"
    artifact_dir.mkdir(parents=True, exist_ok=False)
    field: GaussianField = method["field"]
    field_path = artifact_dir / "field.gaussian.npz"
    field.save(str(field_path))
    with np.load(field_path) as payload:
        field_keys = sorted(payload.files)
    decode_started = time.perf_counter()
    cold_field = GaussianField.load(str(field_path), device=args.device)
    decode_seconds = time.perf_counter() - decode_started
    render_started = time.perf_counter()
    with torch.no_grad():
        cold_tensor = h22._field_render(
            cold_field,
            target.shape[0],
            target.shape[1],
            method["renderer"],
            args.render_chunk,
        )
        repeated_tensor = h22._field_render(
            cold_field,
            target.shape[0],
            target.shape[1],
            method["renderer"],
            args.render_chunk,
        )
    torch.cuda.synchronize()
    render_seconds = time.perf_counter() - render_started
    cold = cold_tensor.detach().cpu().numpy().astype(np.float32, copy=False)
    repeated = repeated_tensor.detach().cpu().numpy().astype(np.float32, copy=False)
    expected = np.asarray(method["expected"], dtype=np.float32)
    metric_started = time.perf_counter()
    metrics = h22.report_utils._metric_values(
        cold,
        target,
        np.ones(target.shape[:2], dtype=bool),
        device=args.device,
        compute_lpips=args.lpips,
    )
    if metrics["lpips"] is None:
        raise RuntimeError(f"LPIPS is required but unavailable: {metrics['lpips_error']}")
    metric_seconds = time.perf_counter() - metric_started
    bounds = h22._save_visuals(artifact_dir, target, cold, args.error_scale)
    trajectory = method["trajectory"]
    h22._write_curve(
        artifact_dir / "learning_curve.svg", trajectory, f"{image_path.stem} {arm}"
    )
    _write_json(artifact_dir / "fit_history.json", method["history"])
    _write_json(artifact_dir / "projection_history.json", [])
    _write_json(artifact_dir / "geometry_history.json", [])
    _write_json(
        artifact_dir / "config.json",
        {
            "schema": REPORT_SCHEMA,
            "arm": arm,
            "seed": seed,
            "budget": budget,
            "init": asdict(h22._init_config(budget, seed)),
            "fit": asdict(
                _fit_config(args, arm)
                if arm in ("normalized_plain", "additive_plain")
                else _continuation_config(args, arm)
            ),
        },
    )
    np.savez_compressed(
        artifact_dir / "analysis.npz",
        crop_bounds=np.asarray(bounds, dtype=np.int32),
        reconstruction_raw=cold,
        error_raw=cold.astype(np.float32) - target.astype(np.float32),
        trajectory_step=np.asarray([row["step"] for row in trajectory], dtype=np.float32),
        trajectory_psnr_db=np.asarray(
            [row["psnr_db"] for row in trajectory], dtype=np.float32
        ),
    )
    cold_parity = float(np.max(np.abs(cold.astype(np.float64) - expected.astype(np.float64))))
    repeated_parity = float(
        np.max(np.abs(repeated.astype(np.float64) - cold.astype(np.float64)))
    )
    total_seconds = (
        init_seconds
        + float(method["wall_fit_seconds"])
        + decode_seconds
        + render_seconds
        + metric_seconds
    )
    row: dict[str, object] = {
        "schema": REPORT_SCHEMA,
        "status": "diagnostic",
        "phase": "development",
        "image": image_path.stem,
        "arm": arm,
        "seed": seed,
        "semantic_family": method["semantic_family"],
        "artifact_dir": str(artifact_dir.relative_to(output_root)),
        "source_path": str(image_path),
        "source_sha256": h22.report_utils._sha256(image_path),
        "source_file_bytes": image_path.stat().st_size,
        "selection_sha256": SELECTION_BINDINGS.get(image_path.name),
        "original_width": raster["original_width"],
        "original_height": raster["original_height"],
        "width": target.shape[1],
        "height": target.shape[0],
        "active_pixels": int(target.shape[0] * target.shape[1]),
        "target_gaussians": budget,
        "n_gaussians": field.n,
        "initial_field_sha256": initial_field_sha256,
        "field_file_sha256": h22.report_utils._sha256(field_path),
        "field_file_bytes": field_path.stat().st_size,
        "field_npz_keys": field_keys,
        "mass_payload_present": any("mass" in key.lower() for key in field_keys),
        "denominator_payload_present": any("denom" in key.lower() for key in field_keys),
        "optimizer_payload_present": any("optimizer" in key.lower() for key in field_keys),
        "auxiliary_rgb_payload_present": any(
            key in field_keys for key in ("color_grads", "opacities")
        ),
        "method_status": method["method_status"],
        "completed": method["completed"],
        "selected_step": method["selected_step"],
        "selected_lambda": 0.0 if arm in CONTINUATION_ARMS else None,
        "hold_psnr_db": method["hold_psnr_db"],
        "optimizer_reset_count": method["optimizer_reset_count"],
        "optimizer_reset_step": method["optimizer_reset_step"],
        "hold_optimizer_reset_count": method["hold_optimizer_reset_count"],
        "endpoint_internal_parity_max_abs": method["endpoint_parity"],
        "attempted_steps": args.iters,
        "psnr_auc_attempted_step": h22._psnr_auc(trajectory, args.iters),
        "renderer_calls_fit": method["renderer_calls"],
        "normalized_calls_fit": method["normalized_calls"],
        "additive_numerator_calls_fit": method["additive_numerator_calls"],
        "additive_denominator_calls_fit": method["additive_denominator_calls"],
        "renderer_calls_coverage_diagnostic": method[
            "renderer_calls_coverage_diagnostic"
        ],
        "init_seconds": init_seconds,
        "fit_seconds": method["fit_seconds"],
        "wall_fit_seconds": method["wall_fit_seconds"],
        "cold_decode_seconds": decode_seconds,
        "render_seconds": render_seconds,
        "metric_seconds": metric_seconds,
        "pipeline_algorithm_seconds": init_seconds + float(method["fit_seconds"]),
        "total_seconds": total_seconds,
        "peak_cuda_allocated_bytes": method["peak_cuda_allocated_bytes"],
        "maintained_render_parity_max_abs": cold_parity,
        "repeated_render_parity_max_abs": repeated_parity,
        "finite_reconstruction": bool(np.isfinite(cold).all()),
        "raw_mse": metrics["masked_mse"],
        **h22._display_metrics(cold, target),
        **h22._coefficient_record(field),
        **method["coverage"],
        **metrics,
    }
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


def _integrity_eligible(rows: list[dict[str, object]]) -> bool:
    return bool(
        len(rows) == len(SOURCE_BINDINGS) * 2
        and all(
            row["completed"]
            and row["method_status"] == "completed"
            and row["selected_lambda"] == 0.0
            and row["n_gaussians"] == row["target_gaussians"] == 640
            and row["finite_reconstruction"]
            and float(row["coefficient_abs_max"]) <= COEFFICIENT_LIMIT
            and float(row["maintained_render_parity_max_abs"]) <= PARITY_LIMIT
            and not row["mass_payload_present"]
            and not row["denominator_payload_present"]
            and not row["optimizer_payload_present"]
            and not row["auxiliary_rgb_payload_present"]
            for row in rows
        )
    )


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
            "mean_renderer_calls": _mean(by_arm[arm], "renderer_calls_fit")
            if by_arm[arm]
            else None,
        }
        for arm in ARMS
    }
    eligible = {
        arm: _integrity_eligible(by_arm[arm]) if complete else False
        for arm in CONTINUATION_ARMS
    }
    selected_arm = None
    selector_reason = "no integrity-eligible continuation arm"
    if complete and any(eligible.values()):
        candidates = [arm for arm in CONTINUATION_ARMS if eligible[arm]]
        if len(candidates) == 1:
            selected_arm = candidates[0]
            selector_reason = "only integrity-eligible continuation arm"
        else:
            no_reset = aggregates["gauge_locked_no_reset"]["mean_psnr_db"]
            reset = aggregates["gauge_locked_endpoint_reset"]["mean_psnr_db"]
            if abs(float(reset) - float(no_reset)) <= 0.02:
                selected_arm = "gauge_locked_no_reset"
                selector_reason = "mean PSNR difference <= 0.02 dB; chose simpler no-reset arm"
            elif float(reset) > float(no_reset):
                selected_arm = "gauge_locked_endpoint_reset"
                selector_reason = "reset arm has higher mean endpoint PSNR"
            else:
                selected_arm = "gauge_locked_no_reset"
                selector_reason = "no-reset arm has higher mean endpoint PSNR"

    gates: dict[str, bool] = {"all_cells_present": complete}
    if complete and selected_arm is not None:
        selected = by_arm[selected_arm]
        normalized = by_arm["normalized_plain"]
        additive = by_arm["additive_plain"]
        keys = lambda row: (row["image"], row["seed"])
        normalized_by_key = {keys(row): row for row in normalized}
        additive_by_key = {keys(row): row for row in additive}
        all_continuations = [row for arm in CONTINUATION_ARMS for row in by_arm[arm]]
        mean_selected_psnr = _mean(selected, "psnr_db")
        mean_additive_psnr = _mean(additive, "psnr_db")
        gates.update(
            {
                "selected_endpoint_integrity": eligible[selected_arm],
                "all_hold_psnr_within_0p05_db_normalized": all(
                    abs(
                        float(row["hold_psnr_db"])
                        - float(normalized_by_key[keys(row)]["hold_psnr_db"])
                    )
                    <= 0.05
                    for row in all_continuations
                ),
                "reset_telemetry_exact": all(
                    row["optimizer_reset_count"] == 0
                    and row["optimizer_reset_step"] is None
                    and row["hold_optimizer_reset_count"] == 0
                    for row in by_arm["gauge_locked_no_reset"]
                )
                and all(
                    row["optimizer_reset_count"] == 1
                    and row["optimizer_reset_step"] == 251
                    and row["hold_optimizer_reset_count"] == 0
                    for row in by_arm["gauge_locked_endpoint_reset"]
                ),
                "mean_psnr_no_more_than_0p05_db_below_additive": (
                    mean_selected_psnr >= mean_additive_psnr - 0.05
                ),
                "closes_half_positive_normalized_additive_gap": (
                    mean_selected_psnr - mean_additive_psnr
                    >= 0.5
                    * max(0.0, _mean(normalized, "psnr_db") - mean_additive_psnr)
                ),
                "mean_lpips_within_additive_plus_0p002": (
                    _mean(selected, "lpips") <= _mean(additive, "lpips") + 0.002
                ),
                "all_lpips_within_additive_plus_0p01": all(
                    float(row["lpips"])
                    <= float(additive_by_key[keys(row)]["lpips"]) + 0.01
                    for row in selected
                ),
                "mean_pixel_max_within_additive_plus_0p005": (
                    _mean(selected, "artifact_pixel_rmse_max")
                    <= _mean(additive, "artifact_pixel_rmse_max") + 0.005
                ),
                "mean_patch7_max_within_additive_plus_0p005": (
                    _mean(selected, "artifact_patch_rmse_max_7")
                    <= _mean(additive, "artifact_patch_rmse_max_7") + 0.005
                ),
                "at_least_one_local_mean_noninferior_to_additive": (
                    _mean(selected, "artifact_pixel_rmse_max")
                    <= _mean(additive, "artifact_pixel_rmse_max")
                    or _mean(selected, "artifact_patch_rmse_max_7")
                    <= _mean(additive, "artifact_patch_rmse_max_7")
                ),
                "mean_psnr_auc_exceeds_additive": (
                    _mean(selected, "psnr_auc_attempted_step")
                    > _mean(additive, "psnr_auc_attempted_step")
                ),
            }
        )
    numeric_pass = bool(gates and all(gates.values()))
    if not complete:
        failure_class = "incomplete_execution"
    elif selected_arm is None or not gates.get("selected_endpoint_integrity", False):
        failure_class = "endpoint_integrity"
    elif not gates.get("all_hold_psnr_within_0p05_db_normalized", False):
        failure_class = "path_identity"
    elif not numeric_pass:
        failure_class = "endpoint_adaptation"
    else:
        failure_class = None
    return {
        "schema": REPORT_SCHEMA,
        "status": "diagnostic",
        "phase": "development",
        "selector": {
            "eligible": eligible,
            "selected_arm": selected_arm,
            "reason": selector_reason,
        },
        "aggregates": aggregates,
        "gates": gates,
        "numeric_pass": numeric_pass,
        "visual_review": "pending",
        "overall_pass": False,
        "failure_class_if_numeric": failure_class,
        "formal_claim_ready": False,
        "interpretation": (
            "Numeric gates pass; native full-frame and worst-crop review remains required."
            if numeric_pass
            else "The frozen mechanism gate failed; retain the bank and do not tune it in place."
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
            f"<td>{float(row['hold_psnr_db']):.3f}</td>"
            f"<td>{int(row['optimizer_reset_count'])}</td>"
            f"<td>{int(row['renderer_calls_fit'])}</td>"
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
<title>HIER-023 unit-gauge continuation</title><style>
body{{font-family:system-ui;margin:2rem;max-width:1700px}}table{{border-collapse:collapse}}
th,td{{border:1px solid #bbb;padding:.35rem}}img{{max-width:300px;max-height:240px;margin:.25rem}}
section{{border-top:1px solid #aaa;margin-top:1rem}}pre,code{{white-space:pre-wrap}}
</style></head><body><h1>HIER-023 unit-gauge normalized-to-additive continuation</h1>
<p><strong>Consumed development diagnostic.</strong> This dirty, historically consumed-source
report is not formal confirmation, semantic/codec selection, a default change, or a novelty claim.</p>
<p><code>{escape(_command())}</code></p>
<p><a href="config.json">config</a> · <a href="decision.json">decision</a> ·
<a href="metrics.json">JSON</a> · <a href="metrics.jsonl">JSONL</a> ·
<a href="metrics.csv">CSV</a> · <a href="attempts.json">attempts</a> ·
<a href="manifest.json">manifest</a></p>
<h2>Decision</h2><pre>{escape(json.dumps(decision, indent=2, sort_keys=True))}</pre>
<h2>Cells</h2><table><tr><th>image</th><th>seed</th><th>arm</th><th>PSNR</th>
<th>MS-SSIM</th><th>LPIPS</th><th>pixel max</th><th>7x7 max</th><th>step-175 PSNR</th>
<th>resets</th><th>fit renderer calls</th><th>artifacts</th></tr>{''.join(table_rows)}</table>
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
        raise SystemExit(f"completed HIER-023 bundle is immutable: {args.out}")
    if args.out.exists() and any(args.out.iterdir()) and not args.resume:
        raise SystemExit(f"output directory is non-empty; use --resume after interruption: {args.out}")
    args.out.mkdir(parents=True, exist_ok=True)

    import torch

    if not torch.cuda.is_available():
        raise SystemExit("frozen HIER-023 protocol requires CUDA")
    sources = _discover_sources(args.images)
    git = h22._git_record()
    snapshots = _snapshot_sources(args.out)
    _write_json(args.out / "environment.json", h22._environment(torch))
    _write_json(
        args.out / "config.json",
        {
            "schema": REPORT_SCHEMA,
            "status": "diagnostic",
            "command": _command(),
            "git": git,
            "source_snapshots": snapshots,
            "source_selection": {
                "salt": SELECTION_SALT,
                "candidate_count": 12,
                "selection_bindings": SELECTION_BINDINGS,
                "source_bindings": SOURCE_BINDINGS,
                "historically_consumed": True,
            },
            "arguments": vars(args),
            "init": asdict(h22._init_config(args.budgets[0], args.seeds[0])),
            "structure_tensor": asdict(StructureTensorConfig()),
            "continuation_schedule": {
                "hold_fraction": 0.35,
                "anneal_fraction": 0.15,
                "endpoint_fraction": 0.50,
                "lengths_at_500": [175, 75, 250],
                "last_anneal_strictly_positive": True,
            },
            "selector": (
                "higher mean exact-endpoint PSNR among integrity-eligible continuation arms; "
                "difference <= 0.02 dB selects no-reset"
            ),
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
        "HIER-023 source selection consumed; no in-place tuning.\n", encoding="utf-8"
    )

    rows: list[dict[str, object]] = []
    attempts: list[dict[str, object]] = []
    metrics_path = args.out / "metrics.json"
    if args.resume and metrics_path.is_file():
        rows = json.loads(metrics_path.read_text(encoding="utf-8")).get("rows", [])
    attempts_path = args.out / "attempts.json"
    if args.resume and attempts_path.is_file():
        attempts = json.loads(attempts_path.read_text(encoding="utf-8")).get("attempts", [])
    row_keys = {(row["image"], row["seed"], row["arm"]) for row in rows}
    tensor_config = StructureTensorConfig()
    for image_path in sources:
        target, mask, raster = h22.report_utils._load_evaluation_raster(
            image_path, None, max_side=args.max_side, mask_threshold=0.5
        )
        if mask is not None:
            raise RuntimeError("HIER-023 requires an unmasked full-frame source")
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
            initial_path = args.out / "initial_fields" / f"{image_path.stem}__s{seed}__n640.npz"
            initial_path.parent.mkdir(parents=True, exist_ok=True)
            if not initial_path.exists():
                initial.save(str(initial_path))
            initial_sha = h22.report_utils._sha256(initial_path)
            for arm in args.arms:
                stable_key = (image_path.stem, seed, arm)
                if stable_key in row_keys:
                    continue
                cell_started = time.perf_counter()
                try:
                    method = _run_method(initial, target, arm, args, torch)
                    row = _write_cell(
                        output_root=args.out,
                        image_path=image_path,
                        target=target,
                        raster=raster,
                        seed=seed,
                        budget=args.budgets[0],
                        arm=arm,
                        initial_field_sha256=initial_sha,
                        init_seconds=init_seconds,
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
        "HIER-023 consumed development diagnostic; do not overwrite.\n", encoding="utf-8"
    )
    _write_manifest(args.out)
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
