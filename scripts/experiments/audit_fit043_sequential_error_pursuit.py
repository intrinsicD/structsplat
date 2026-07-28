#!/usr/bin/env python3
"""Independent cold-field and artifact audit for FIT-043."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
from typing import Any

import numpy as np
import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
for _root in (
    REPOSITORY_ROOT,
    REPOSITORY_ROOT / "src",
    REPOSITORY_ROOT / "deprecated_scripts",
):
    _text = str(_root)
    if _text not in sys.path:
        sys.path.insert(0, _text)

from scripts.experiments.fit032_janelle_dipole_screen import (  # noqa: E402
    _base_config,
    _prepare_janelle,
)
from scripts.experiments.fit033_janelle_highpass_solve import (  # noqa: E402
    _evaluate_all,
)
from scripts.experiments.fit043_sequential_error_pursuit import (  # noqa: E402
    DEFAULT_CAPTURE_ROOT,
    DEFAULT_INPUT,
    DEFAULT_OUT,
    DEFAULT_REALTIME_ROOT,
    DEFAULT_REPORT_OUT,
    FROZEN_INPUT_HASHES,
    SCHEMA,
)
from scripts.experiments.run_janelle_cross_view_tail_diagnostic import (  # noqa: E402
    _constraint,
    _prefix_exact,
)
from structsplat.gaussians import GaussianField  # noqa: E402
from structsplat.safe_schedule import (  # noqa: E402
    SafeScheduleConfig,
    safe_commit_decision,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _reduction(before: float, after: float) -> float:
    return 0.0 if before <= 0.0 else 1.0 - after / before


def _independent_adjusted_target(base: float, entry: float, target: float) -> float:
    threshold = base * (1.0 - target)
    if entry <= threshold or entry <= 0.0:
        return 0.0
    return min(1.0, max(0.0, 1.0 - threshold / entry))


def _close(left: float, right: float) -> bool:
    return bool(
        np.isfinite(left) and np.isfinite(right) and abs(left - right) <= 1e-6 + 1e-5 * abs(right)
    )


def _metrics_match(
    observed: dict[str, Any],
    stored: dict[str, Any],
) -> tuple[bool, float]:
    deltas = []
    valid = True
    for key, stored_value in stored.items():
        if isinstance(stored_value, bool) or not isinstance(stored_value, (int, float)):
            continue
        if key not in observed:
            continue
        delta = abs(float(observed[key]) - float(stored_value))
        deltas.append(delta)
        valid = bool(valid and _close(float(observed[key]), float(stored_value)))
    return valid, max(deltas, default=0.0)


def _stats(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p25": float(np.quantile(array, 0.25)),
        "p75": float(np.quantile(array, 0.75)),
    }


def _stats_match(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return all(_close(float(left[key]), float(right[key])) for key in left)


def _audit_cell(
    args: argparse.Namespace,
    frame: str,
    view: str,
    device: torch.device,
) -> dict[str, Any]:
    input_path = args.input / "cells" / frame / view / "result.json"
    result_path = args.run / "cells" / frame / view / "result.json"
    source = _load_json(input_path)
    result = _load_json(result_path)
    if result.get("schema") != f"{SCHEMA}.cell":
        raise RuntimeError(f"unexpected FIT-043 cell schema: {result_path}")
    bindings = result["input_binding"]
    binding_checks = {
        "input_cell": _sha256(input_path) == bindings["cell_result_sha256"],
        "base": _sha256(Path(bindings["base_field"])) == bindings["base_field_sha256"],
        "error": _sha256(Path(bindings["error_field"])) == bindings["error_field_sha256"],
        "pursuit": _sha256(Path(bindings["pursuit_field"])) == bindings["pursuit_field_sha256"],
        "target": _sha256(Path(bindings["target"])) == bindings["target_sha256"],
        "mask": _sha256(Path(bindings["mask"])) == bindings["mask_sha256"],
        "final": _sha256(Path(result["field"]["path"])) == result["field"]["sha256"],
    }
    if not all(binding_checks.values()):
        raise RuntimeError(f"binding failure for {frame}/{view}: {binding_checks}")

    source_record = source["source"]
    prepare_args = argparse.Namespace(
        realtime_root=args.realtime_root,
        capture_root=args.capture_root,
        frame=frame,
        view_id=view,
        max_side=1200,
        field=Path(source_record["archive"]),
    )
    prepared = _prepare_janelle(prepare_args)
    target = torch.as_tensor(
        np.asarray(prepared["target"], dtype=np.float32),
        device=device,
        dtype=torch.float32,
    ).contiguous()
    mask_cpu = np.asarray(prepared["mask"], dtype=bool)
    mask = torch.as_tensor(mask_cpu, device=device, dtype=torch.bool)
    cfg_args = argparse.Namespace(
        renderer="cuda",
        mask_margin=0.75,
        boundary_band=4.0,
        coverage_tau=0.05,
    )
    cfg = replace(
        _base_config(cfg_args),
        color_solve_maxiter=1,
        color_solve_lambda=1e30,
    )
    if json.loads(json.dumps(asdict(cfg), default=str)) != result["fit_config"]:
        raise RuntimeError(f"fit config mismatch for {frame}/{view}")
    constraint = _constraint(mask_cpu, target, cfg, 4.0)
    base = GaussianField.load(bindings["base_field"], device=device)
    error = GaussianField.load(bindings["error_field"], device=device)
    final = GaussianField.load(result["field"]["path"], device=device)
    base_metrics, _, base_quality = _evaluate_all(base, target, mask, cfg, constraint, 0.05)
    error_metrics, _, error_quality = _evaluate_all(error, target, mask, cfg, constraint, 0.05)
    final_metrics, _, final_quality = _evaluate_all(final, target, mask, cfg, constraint, 0.05)
    metric_checks = {
        "base": _metrics_match(base_metrics, result["baseline"]),
        "error": _metrics_match(error_metrics, result["stage_entry"]),
        "final": _metrics_match(final_metrics, result["final"]),
    }
    metrics_match = all(value[0] for value in metric_checks.values())
    prefix_exact, prefix_checks = _prefix_exact(error, final)
    stage_safe, stage_reasons = safe_commit_decision(
        error_quality,
        final_quality,
        SafeScheduleConfig().tolerances,
        0.0,
    )
    base_safe, base_reasons = safe_commit_decision(
        base_quality,
        final_quality,
        SafeScheduleConfig().tolerances,
        0.0,
    )
    hp = _reduction(
        float(base_metrics["detail_highpass_sigma_1_5_mse"]),
        float(final_metrics["detail_highpass_sigma_1_5_mse"]),
    )
    lap = _reduction(
        float(base_metrics["detail_laplacian_mse"]),
        float(final_metrics["detail_laplacian_mse"]),
    )
    incremental_hp = _reduction(
        float(error_metrics["detail_highpass_sigma_1_5_mse"]),
        float(final_metrics["detail_highpass_sigma_1_5_mse"]),
    )
    incremental_lap = _reduction(
        float(error_metrics["detail_laplacian_mse"]),
        float(final_metrics["detail_laplacian_mse"]),
    )
    controller = result["controller"]
    expected_hp_target = _independent_adjusted_target(
        float(base_metrics["detail_highpass_sigma_1_5_mse"]),
        float(error_metrics["detail_highpass_sigma_1_5_mse"]),
        0.25,
    )
    expected_lap_target = _independent_adjusted_target(
        float(base_metrics["detail_laplacian_mse"]),
        float(error_metrics["detail_laplacian_mse"]),
        0.20,
    )
    controller_checks = {
        "highpass_target": _close(
            float(controller["adjusted_stage_highpass_target"]),
            expected_hp_target,
        ),
        "laplacian_target": _close(
            float(controller["adjusted_stage_laplacian_target"]),
            expected_lap_target,
        ),
        "row_arithmetic": (
            int(result["rows"]["combined_tail_added"])
            == int(result["rows"]["error_only_added"])
            + int(result["rows"]["sequential_pursuit_added"])
            and int(result["field"]["rows"]) == int(final.n)
        ),
        "stored_prefix": result["error_prefix_exact"] == prefix_exact,
        "stored_stage_gate": result["stage_protected_safe"] == bool(stage_safe),
        "stored_base_gate": result["original_base_protected_safe"] == bool(base_safe),
        "stored_target": (
            result["target_reached_common_25hp_20lap"] == bool(hp >= 0.25 and lap >= 0.20)
        ),
        "stored_reductions": (
            _close(
                float(result["cumulative_reductions"]["detail_highpass_sigma_1_5_mse"]),
                hp,
            )
            and _close(
                float(result["cumulative_reductions"]["detail_laplacian_mse"]),
                lap,
            )
            and _close(
                float(result["incremental_reductions"]["detail_highpass_sigma_1_5_mse"]),
                incremental_hp,
            )
            and _close(
                float(result["incremental_reductions"]["detail_laplacian_mse"]),
                incremental_lap,
            )
        ),
    }
    if not metrics_match or not all(controller_checks.values()):
        raise RuntimeError(
            f"metric/controller audit failure for {frame}/{view}: "
            f"metrics={metric_checks}, controller={controller_checks}"
        )
    return {
        "frame": frame,
        "view": view,
        "disposition": controller["disposition"],
        "bindings": binding_checks,
        "metric_max_abs_delta": max(value[1] for value in metric_checks.values()),
        "controller_checks": controller_checks,
        "prefix_exact": prefix_exact,
        "prefix_checks": prefix_checks,
        "stage_safe": bool(stage_safe),
        "stage_reasons": list(stage_reasons),
        "base_safe": bool(base_safe),
        "base_reasons": list(base_reasons),
        "outside_exact_zero": bool(
            float(final_metrics["outside_max_abs"]) == 0.0
            and float(final_metrics["outside_coverage_max"]) == 0.0
        ),
        "target_reached": bool(hp >= 0.25 and lap >= 0.20),
        "highpass_reduction": hp,
        "laplacian_reduction": lap,
        "incremental_highpass_reduction": incremental_hp,
        "incremental_laplacian_reduction": incremental_lap,
        "sequential_rows": int(result["rows"]["sequential_pursuit_added"]),
        "pursuit_only_rows": int(result["rows"]["pursuit_only_added_reused"]),
        "combined_minus_error_psnr": float(final_metrics["foreground_psnr_db"])
        - float(error_metrics["foreground_psnr_db"]),
        "retention": float(result["quality_comparison"]["foreground_gain_retention_fraction"]),
    }


def run(args: argparse.Namespace) -> None:
    args.input = args.input.resolve()
    args.run = args.run.resolve()
    args.report = args.report.resolve()
    args.capture_root = args.capture_root.resolve()
    args.realtime_root = args.realtime_root.resolve()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA but torch.cuda.is_available() is false")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    audit_source = Path(__file__).resolve()
    audit_source_snapshot = args.run / "audit_source_snapshot.py"
    if audit_source_snapshot.is_file():
        if _sha256(audit_source_snapshot) != _sha256(audit_source):
            raise RuntimeError(
                "existing FIT-043 audit source snapshot differs from the executed auditor"
            )
    else:
        shutil.copy2(audit_source, audit_source_snapshot)
    manifest = _load_json(args.run / "manifest.json")
    summary = _load_json(args.run / "summary.json")
    input_manifest = _load_json(args.input / "manifest.json")
    frozen_hashes_ok = all(
        _sha256(args.input / name) == expected for name, expected in FROZEN_INPUT_HASHES.items()
    )
    snapshot_checks = []
    for record in manifest["source_snapshot"]:
        captured = args.run / "source_snapshot" / record["path"]
        snapshot_checks.append(
            captured.is_file()
            and _sha256(captured) == record["sha256"]
            and captured.stat().st_size == record["bytes"]
        )
    requested = [(cell["frame"], cell["view_id"]) for cell in input_manifest["requested_cells"]]
    cells = []
    for index, (frame, view) in enumerate(requested, start=1):
        print(f"[{index}/51] audit {frame}/{view}", flush=True)
        cells.append(_audit_cell(args, frame, view, device))
        if device.type == "cuda":
            torch.cuda.empty_cache()

    executed = [cell for cell in cells if cell["disposition"] == "pursuit_executed"]
    skipped = [cell for cell in cells if cell["disposition"] == "already_satisfied"]
    recomputed = {
        "completed_cells": len(cells),
        "pursuit_executed_cells": len(executed),
        "already_satisfied_cells": len(skipped),
        "target_reached_cells": sum(cell["target_reached"] for cell in cells),
        "stage_protected_safe_cells": sum(cell["stage_safe"] for cell in cells),
        "original_base_protected_safe_cells": sum(cell["base_safe"] for cell in cells),
        "outside_exact_zero_cells": sum(cell["outside_exact_zero"] for cell in cells),
        "executed_prefix_exact_cells": sum(cell["prefix_exact"] for cell in executed),
        "sequential_rows": _stats([float(cell["sequential_rows"]) for cell in cells]),
        "pursuit_only_rows": _stats([float(cell["pursuit_only_rows"]) for cell in cells]),
        "highpass": _stats([float(cell["highpass_reduction"]) for cell in cells]),
        "laplacian": _stats([float(cell["laplacian_reduction"]) for cell in cells]),
        "combined_minus_error_psnr": _stats(
            [float(cell["combined_minus_error_psnr"]) for cell in cells]
        ),
    }
    aggregate_checks = {
        "completed": recomputed["completed_cells"] == summary["completed_cells"] == 51,
        "executed": recomputed["pursuit_executed_cells"] == summary["pursuit_executed_cells"],
        "skipped": recomputed["already_satisfied_cells"] == summary["already_satisfied_cells"],
        "target": recomputed["target_reached_cells"] == summary["target_reached_cells"],
        "stage_safe": recomputed["stage_protected_safe_cells"]
        == summary["stage_protected_safe_cells"],
        "base_safe": recomputed["original_base_protected_safe_cells"]
        == summary["original_base_protected_safe_cells"],
        "outside": recomputed["outside_exact_zero_cells"] == summary["outside_exact_zero_cells"],
        "prefix": recomputed["executed_prefix_exact_cells"]
        == summary["executed_prefix_exact_cells"],
        "sequential_rows": _stats_match(
            recomputed["sequential_rows"],
            summary["rows"]["sequential_pursuit_added_all_cells"],
        ),
        "pursuit_rows": _stats_match(
            recomputed["pursuit_only_rows"],
            summary["rows"]["pursuit_only_added_reused"],
        ),
        "highpass": _stats_match(
            recomputed["highpass"],
            summary["cumulative"]["highpass_reduction"],
        ),
        "laplacian": _stats_match(
            recomputed["laplacian"],
            summary["cumulative"]["laplacian_reduction"],
        ),
        "psnr_delta": _stats_match(
            recomputed["combined_minus_error_psnr"],
            summary["cumulative"]["combined_minus_error_foreground_psnr_db"],
        ),
    }
    rule_1 = bool(
        len(cells) == 51
        and all(
            cell["target_reached"]
            and cell["base_safe"]
            and cell["outside_exact_zero"]
            and (cell["prefix_exact"] if cell["disposition"] == "pursuit_executed" else True)
            for cell in cells
        )
    )
    rule_2 = bool(
        all(cell["retention"] >= 0.95 for cell in cells)
        and recomputed["combined_minus_error_psnr"]["median"] >= -0.05
    )
    rule_3 = all(
        cell["incremental_highpass_reduction"] > 0.0
        and cell["incremental_laplacian_reduction"] > 0.0
        for cell in executed
    )
    rule_4 = bool(
        recomputed["sequential_rows"]["median"] <= recomputed["pursuit_only_rows"]["median"]
    )
    decision_checks = {
        "rule_1": summary["decision"]["rule_1_all_cells_target_protected_prefix_zero"] == rule_1,
        "rule_2": summary["decision"]["rule_2_global_gain_retained"] == rule_2,
        "rule_3": summary["decision"]["rule_3_every_executed_stage_improves_both_detail_metrics"]
        == rule_3,
        "rule_4": summary["decision"]["rule_4_median_incremental_rows_no_more_than_pursuit_only"]
        == rule_4,
        "viable": summary["decision"]["viable_dual_objective_exposed_data_option"]
        == bool(rule_1 and rule_2 and rule_3 and rule_4),
    }

    with (args.report / "comparison.csv").open("r", encoding="utf-8", newline="") as stream:
        csv_rows = list(csv.DictReader(stream))
    index_path = args.report / "index.html"
    index_text = index_path.read_text(encoding="utf-8")
    references = re.findall(r'(?:src|href)="([^"]+)"', index_text)
    local_references = [
        reference
        for reference in references
        if not reference.startswith(("http://", "https://", "#"))
    ]
    missing_links = [
        reference
        for reference in local_references
        if reference != "audit.json" and not (args.report / reference).is_file()
    ]
    report_checks = {
        "csv_rows": len(csv_rows) == 51,
        "cards": index_text.count('<article class="cell') == 51,
        "local_links": len(local_references),
        "missing_links": missing_links,
        "audit_output_link_declared": "audit.json" in local_references,
        "summary_copy": _sha256(args.report / "summary.json") == _sha256(args.run / "summary.json"),
    }
    checks = {
        "manifest_complete": (
            manifest["status"] == "complete"
            and manifest["completed_cells"] == 51
            and manifest["failed_cells"] == 0
        ),
        "frozen_input_hashes": frozen_hashes_ok,
        "source_snapshot": all(snapshot_checks) and len(snapshot_checks) > 0,
        "inventory": len(cells) == 51,
        "bindings": all(all(cell["bindings"].values()) for cell in cells),
        "cold_metrics": all(cell["metric_max_abs_delta"] <= 1e-6 + 1e-5 for cell in cells),
        "controller_math": all(all(cell["controller_checks"].values()) for cell in cells),
        "aggregates": all(aggregate_checks.values()),
        "decision": all(decision_checks.values()),
        "report": (
            report_checks["csv_rows"]
            and report_checks["cards"]
            and not report_checks["missing_links"]
            and report_checks["audit_output_link_declared"]
            and report_checks["summary_copy"]
        ),
    }
    audit = {
        "schema": f"{SCHEMA}.audit",
        "task": "FIT-043",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "aggregate_checks": aggregate_checks,
        "decision_checks": decision_checks,
        "report_checks": report_checks,
        "recomputed": recomputed,
        "maximum_cold_metric_abs_delta": max(cell["metric_max_abs_delta"] for cell in cells),
        "cells": cells,
        "scope": (
            "cold replay on the same RTX-4090 environment; exposed correlated "
            "Janelle cells; no FIT-042, actual-rate, or default claim"
        ),
        "audit_source": {
            "path": str(audit_source_snapshot.resolve()),
            "sha256": _sha256(audit_source_snapshot),
            "bytes": audit_source_snapshot.stat().st_size,
        },
    }
    _atomic_json(args.run / "audit.json", audit)
    _atomic_json(args.report / "audit.json", audit)
    artifact_path = args.report / "artifact.json"
    artifact = _load_json(artifact_path)
    artifact["audit_sha256"] = _sha256(args.report / "audit.json")
    artifact["audit_status"] = audit["status"]
    _atomic_json(artifact_path, artifact)
    print(json.dumps({**checks, "status": audit["status"]}, indent=2))
    if audit["status"] != "PASS":
        raise SystemExit(1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--run", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_OUT)
    parser.add_argument("--capture-root", type=Path, default=DEFAULT_CAPTURE_ROOT)
    parser.add_argument("--realtime-root", type=Path, default=DEFAULT_REALTIME_ROOT)
    parser.add_argument("--device", default="cuda:0")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
