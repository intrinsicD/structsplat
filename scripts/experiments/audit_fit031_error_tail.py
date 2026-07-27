#!/usr/bin/env python3
"""Independent artifact audit for the FIT-031 Janelle development run."""

from __future__ import annotations

import argparse
from dataclasses import fields
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from structsplat.config import FitConfig
from structsplat.gaussians import GaussianField
from structsplat.safe_schedule import (
    CommitTolerances,
    QualityMetrics,
    _MaskConstraint,
    evaluate_quality,
    safe_commit_decision,
)
from structsplat.workflows import _metric_bundle, _prepare_source


PROTECTED_FIELDS = {item.name for item in fields(QualityMetrics)}
MARKER_EVENTS = {
    "initial_safe_state",
    "phase_end",
    "stage_end",
    "error_tail_estimate",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _row(root: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    rows = _json(root / "metrics.json")
    if len(rows) != 1 or rows[0].get("status") != "ok":
        raise ValueError(f"expected one successful row in {root}")
    row = rows[0]
    config = _json(Path(row["config_json"]))
    history = _json(Path(row["history_json"]))
    return row, config, history


def _quality(payload: dict[str, Any]) -> QualityMetrics:
    return QualityMetrics(**{name: payload[name] for name in PROTECTED_FIELDS})


def _history_audit(
    row: dict[str, Any],
    config: dict[str, Any],
    history_payload: dict[str, Any],
) -> dict[str, Any]:
    history = history_payload["schedule_history"]
    tolerance = CommitTolerances(**config["schedule"]["tolerances"])
    hole_budget = float(config["schedule"]["hole_regression_budget"])
    continuity_failures: list[int] = []
    accepted_gate_failures: list[dict[str, Any]] = []
    attempt_gate_failures: list[dict[str, Any]] = []
    accepted_checked = 0
    rejected_attempts_checked = 0

    for index, record in enumerate(history):
        if index and record["before"] != history[index - 1]["selected"]:
            continuity_failures.append(index)
        candidate = record.get("candidate")
        if (
            record.get("event") not in MARKER_EVENTS
            and record.get("accepted")
            and candidate is not None
        ):
            accepted_checked += 1
            accepted, reasons = safe_commit_decision(
                _quality(record["before"]),
                _quality(candidate),
                tolerance,
                hole_budget,
            )
            if not accepted:
                accepted_gate_failures.append(
                    {
                        "index": index,
                        "phase": record["phase"],
                        "event": record["event"],
                        "reasons": reasons,
                    }
                )

        for attempt in (record.get("metadata") or {}).get("attempts", []):
            attempt_candidate = attempt.get("candidate")
            if attempt_candidate is None or "safe_gate_accepted" not in attempt:
                continue
            accepted, reasons = safe_commit_decision(
                _quality(record["before"]),
                _quality(attempt_candidate),
                tolerance,
                hole_budget,
            )
            expected = bool(attempt["safe_gate_accepted"])
            if not expected:
                rejected_attempts_checked += 1
            if accepted != expected:
                attempt_gate_failures.append(
                    {
                        "index": index,
                        "phase": record["phase"],
                        "event": record["event"],
                        "requested": attempt.get("requested"),
                        "recorded": expected,
                        "replayed": accepted,
                        "reasons": reasons,
                    }
                )

    attempted_sum = sum(int(record.get("attempted_steps", 0)) for record in history)
    accepted_sum = sum(int(record.get("accepted_steps", 0)) for record in history)
    return {
        "records": len(history),
        "continuity_failures": continuity_failures,
        "accepted_gate_records_checked": accepted_checked,
        "accepted_gate_failures": accepted_gate_failures,
        "rejected_attempts_checked": rejected_attempts_checked,
        "attempt_gate_failures": attempt_gate_failures,
        "attempted_steps_sum": attempted_sum,
        "attempted_steps_matches": attempted_sum == int(row["attempted_steps"]),
        "accepted_steps_sum": accepted_sum,
        "accepted_steps_matches": accepted_sum == int(row["accepted_steps"]),
    }


def _estimator_audit(
    fine_root: Path,
    row: dict[str, Any],
    history_payload: dict[str, Any],
) -> dict[str, Any]:
    tail = row["error_tail"]
    recomputed = (
        0
        if float(tail["residual_l2_square_sum"]) <= 0.0
        else math.ceil(
            float(tail["residual_l1_sum"]) ** 2
            / float(tail["residual_l2_square_sum"])
        )
    )
    requested = math.ceil(float(tail["fraction"]) * recomputed)

    estimate_snapshot = next(
        snapshot
        for snapshot in history_payload["snapshots"]
        if snapshot["phase"] == "error_tail"
        and snapshot["event"] == "error_tail_estimate"
    )
    render_u8 = (
        np.asarray(Image.open(estimate_snapshot["reconstruction"]).convert("RGB"))
        .astype(np.float64)
        / 255.0
    )
    target_u8 = (
        np.asarray(Image.open(row["target_png"]).convert("RGB"))
        .astype(np.float64)
        / 255.0
    )
    config = _json(Path(row["config_json"]))
    mask_path = Path(config["source"]["mask_path"])
    mask = np.asarray(
        Image.open(mask_path)
        .convert("L")
        .resize(
            (render_u8.shape[1], render_u8.shape[0]),
            Image.Resampling.NEAREST,
        )
    ) > 127
    error = np.abs(render_u8 - target_u8).mean(axis=2)[mask]
    error_sum = float(error.sum())
    error_square_sum = float(np.square(error).sum())
    png_estimate = (
        0
        if error_square_sum <= 0.0
        else math.ceil(error_sum * error_sum / error_square_sum)
    )
    png_relative_delta = abs(png_estimate - recomputed) / max(recomputed, 1)

    tail_records = [
        record
        for record in history_payload["schedule_history"]
        if record["phase"] == "error_tail"
        and record["event"] == "error_tail_birth"
        and record["accepted"]
    ]
    counts = [int(record["metadata"]["winner_count"]) for record in tail_records]
    geometry_failures = []
    for index, record in enumerate(tail_records):
        metadata = record["metadata"]["winner_metadata"]
        if (
            metadata.get("score_rule") != "foreground per-pixel RGB MAE only"
            or metadata.get("covariance_rule")
            != "small isotropic residual-support scale"
            or float(metadata.get("mean_axis_ratio", float("nan"))) != 1.0
            or float(metadata.get("max_base_scale", float("inf")))
            > float(tail["max_scale"])
        ):
            geometry_failures.append(index)

    return {
        "formula_recomputed_rows": recomputed,
        "formula_matches": recomputed == int(tail["estimated_complete_rows"]),
        "request_recomputed_rows": requested,
        "request_matches": requested == int(tail["requested_rows"]),
        "png_quantized_estimate": png_estimate,
        "png_quantized_relative_delta": png_relative_delta,
        "png_quantized_within_one_percent": png_relative_delta <= 0.01,
        "accepted_wave_counts": counts,
        "accepted_rows_sum": sum(counts),
        "activated_rows_matches": sum(counts) == int(tail["activated_rows"]),
        "all_batches_bounded": all(
            int(tail["minimum_batch_rows"])
            <= count
            <= int(tail["batch_rows"])
            for count in counts
        ),
        "geometry_failures": geometry_failures,
        "fine_root": str(fine_root),
    }


def _cold_rescore(
    row: dict[str, Any],
    config: dict[str, Any],
    device: str,
) -> dict[str, Any]:
    import torch

    source = config["source"]
    prepared = _prepare_source(
        Path(source["path"]),
        Path(source["relative"]),
        mask_root=None,
        direct_mask=Path(source["mask_path"]),
        mask_invert=False,
        max_side=max(source["fit_size"]),
    )
    target = torch.as_tensor(
        prepared["target"], device=device, dtype=torch.float32
    ).contiguous()
    mask_tensor = torch.as_tensor(
        prepared["mask"], device=device, dtype=torch.bool
    )
    fit_config = FitConfig(**config["fit_config"])
    constraint = _MaskConstraint.from_mask(
        prepared["mask"],
        target.device,
        target.dtype,
        fit_config.sigma_cutoff,
        fit_config.mask_margin,
        aa_dilation=fit_config.aa_dilation,
        cap_mode="anisotropic",
        undercoverage_band=float(config["schedule"]["boundary_band"]),
    )
    field = GaussianField.load(row["field_npz"], device=device)
    with torch.no_grad():
        protected, render = evaluate_quality(
            field,
            target,
            mask_tensor,
            fit_config,
            constraint,
            float(config["schedule"]["coverage_tau"]),
        )
        display = _metric_bundle(
            render,
            target,
            prepared["mask"],
            lpips=row.get("lpips") is not None,
        )
    stored_protected = row["error_tail"]["after"]
    protected_delta = {
        name: abs(float(protected.to_dict()[name]) - float(stored_protected[name]))
        for name in PROTECTED_FIELDS
        if name != "finite"
    }
    display_names = ("psnr", "ssim", "ms_ssim", "lpips", "mse", "mae", "max_abs")
    display_delta = {
        name: abs(float(display[name]) - float(row[name]))
        for name in display_names
        if display.get(name) is not None and row.get(name) is not None
    }
    return {
        "device": device,
        "field_sha256": _sha256(Path(row["field_npz"])),
        "field_rows": field.n,
        "protected_metrics": protected.to_dict(),
        "protected_abs_delta": protected_delta,
        "protected_max_abs_delta": max(protected_delta.values(), default=0.0),
        "display_metrics": display,
        "display_abs_delta": display_delta,
        "display_max_abs_delta": max(display_delta.values(), default=0.0),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline_root", type=Path)
    parser.add_argument("fine_root", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--source-patch", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    baseline_root = args.baseline_root.resolve()
    fine_root = args.fine_root.resolve()
    repository_root = Path(__file__).resolve().parents[2]
    baseline, baseline_config, _ = _row(baseline_root)
    fine, fine_config, fine_history = _row(fine_root)
    tail = fine["error_tail"]

    source_equal = baseline_config["source"] == fine_config["source"]
    environment_equal = baseline_config["environment"] == fine_config["environment"]
    initialization_equal = (
        baseline_config["initialization"] == fine_config["initialization"]
    )
    fit_config_equal = baseline_config["fit_config"] == fine_config["fit_config"]
    within_run_deltas = {
        "foreground_psnr_db": (
            float(tail["after"]["foreground_psnr_db"])
            - float(tail["before"]["foreground_psnr_db"])
        ),
        "boundary_psnr_db": (
            float(tail["after"]["boundary_psnr_db"])
            - float(tail["before"]["boundary_psnr_db"])
        ),
        "cvar99_relative": (
            float(tail["after"]["cvar99_mse"])
            / float(tail["before"]["cvar99_mse"])
            - 1.0
        ),
        "p99_relative": (
            float(tail["after"]["p99_mse"])
            / float(tail["before"]["p99_mse"])
            - 1.0
        ),
        "boundary_hole_percentage_points": 100.0
        * (
            float(tail["after"]["boundary_hole_fraction"])
            - float(tail["before"]["boundary_hole_fraction"])
        ),
    }
    baseline_context = {
        "baseline_rows": int(baseline["n_gaussians"]),
        "fine_pre_tail_rows": int(tail["start_n"]),
        "final_rows": int(fine["n_gaussians"]),
        "baseline_psnr": float(baseline["psnr"]),
        "fine_final_psnr": float(fine["psnr"]),
        "display_psnr_delta": float(fine["psnr"]) - float(baseline["psnr"]),
        "baseline_total_seconds": float(baseline["total_seconds"]),
        "fine_total_seconds": float(fine["total_seconds"]),
        "count_matched": int(baseline["n_gaussians"]) == int(fine["n_gaussians"]),
        "ordinary_terminal_count_matched": int(baseline["n_gaussians"])
        == int(tail["start_n"]),
    }
    audit = {
        "schema": "structsplat.fit031.error_tail_audit.v1",
        "valid": True,
        "scope": "one-image one-seed RTX-4090 exposed-development terminal-tail assay",
        "controlled_equalities": {
            "source": source_equal,
            "target_pixel_sha256": baseline["target_pixel_sha256"]
            == fine["target_pixel_sha256"],
            "seed": baseline["seed"] == fine["seed"],
            "environment": environment_equal,
            "initialization_config": initialization_equal,
            "fit_config": fit_config_equal,
        },
        "history": _history_audit(fine, fine_config, fine_history),
        "estimator_and_geometry": _estimator_audit(
            fine_root, fine, fine_history
        ),
        "cold_rescore": _cold_rescore(fine, fine_config, args.device),
        "within_run_deltas": within_run_deltas,
        "baseline_context": baseline_context,
        "results": {
            "baseline": {
                name: baseline[name]
                for name in (
                    "n_gaussians",
                    "psnr",
                    "ssim",
                    "ms_ssim",
                    "lpips",
                    "mse",
                    "mae",
                    "attempted_steps",
                    "accepted_steps",
                    "fit_seconds",
                    "total_seconds",
                )
            },
            "fine_detail": {
                name: fine[name]
                for name in (
                    "n_gaussians",
                    "psnr",
                    "ssim",
                    "ms_ssim",
                    "lpips",
                    "mse",
                    "mae",
                    "attempted_steps",
                    "accepted_steps",
                    "fit_seconds",
                    "total_seconds",
                )
            },
            "tail": tail,
        },
        "artifact_sha256": {
            "baseline_metrics": _sha256(baseline_root / "metrics.json"),
            "baseline_manifest": _sha256(baseline_root / "manifest.json"),
            "fine_metrics": _sha256(fine_root / "metrics.json"),
            "fine_manifest": _sha256(fine_root / "manifest.json"),
            "fine_index": _sha256(fine_root / "index.html"),
        },
        "executed_source": {
            "base_commit": fine_config["repository"]["commit"],
            "branch": fine_config["repository"]["branch"],
            "dirty": fine_config["repository"]["dirty"],
            "status_sha256": fine_config["repository"]["status_sha256"],
            "implementation_sha256": {
                relative: _sha256(repository_root / relative)
                for relative in (
                    "src/structsplat/pipeline.py",
                    "src/structsplat/safe_schedule.py",
                    "src/structsplat/workflows.py",
                )
            },
            "patch_path": (
                None
                if args.source_patch is None
                else str(args.source_patch.resolve())
            ),
            "patch_sha256": (
                None
                if args.source_patch is None
                else _sha256(args.source_patch.resolve())
            ),
        },
        "limitations": [
            "one exposed development image, one seed, one RTX 4090 execution",
            "CUDA atomic accumulation makes the ordinary trajectories tolerance-reproducible, not bit-exact",
            "the existing baseline stopped below 11,000 rows while this replay reached 11,000 before the tail",
            "the final field is neither count-matched nor rate-matched to the default",
            "the estimator snapshot is persisted as an 8-bit PNG, so its independent pixel replay is quantized",
            "no default, generality, codec-rate, or efficiency claim is authorized",
        ],
    }
    checks = [
        *audit["controlled_equalities"].values(),
        not audit["history"]["continuity_failures"],
        not audit["history"]["accepted_gate_failures"],
        not audit["history"]["attempt_gate_failures"],
        audit["history"]["attempted_steps_matches"],
        audit["history"]["accepted_steps_matches"],
        audit["estimator_and_geometry"]["formula_matches"],
        audit["estimator_and_geometry"]["request_matches"],
        audit["estimator_and_geometry"]["png_quantized_within_one_percent"],
        audit["estimator_and_geometry"]["activated_rows_matches"],
        audit["estimator_and_geometry"]["all_batches_bounded"],
        not audit["estimator_and_geometry"]["geometry_failures"],
        audit["cold_rescore"]["field_rows"] == int(fine["n_gaussians"]),
        audit["cold_rescore"]["protected_max_abs_delta"] <= 1e-6,
        audit["cold_rescore"]["display_max_abs_delta"] <= 1e-5,
        args.source_patch is not None and args.source_patch.is_file(),
    ]
    audit["valid"] = all(checks)
    payload = json.dumps(audit, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
        print(args.output)
    return 0 if audit["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
