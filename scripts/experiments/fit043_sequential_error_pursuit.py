#!/usr/bin/env python3
"""Run FIT-043's cumulative error-only -> orthogonal-pursuit sequence.

The 51 base, error-only, and pursuit-only cells are reused from the audited
Janelle cross-view diagnostic. Only the missing sequential arm is executed.
This is exposed, correlated mechanism evidence and never FIT-042 confirmation.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import html
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import traceback
from typing import Any

import numpy as np
from PIL import __version__ as PILLOW_VERSION
from PIL import Image
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
from scripts.experiments.fit040_janelle_production_pursuit import (  # noqa: E402
    _disabled_phase,
)
from scripts.experiments.run_janelle_cross_view_tail_diagnostic import (  # noqa: E402
    _constraint,
    _prefix_exact,
    _save_crop,
    _save_residual_crop,
    _save_rgb,
)
from structsplat.gaussians import GaussianField  # noqa: E402
from structsplat.safe_schedule import (  # noqa: E402
    SafeScheduleConfig,
    run_safe_schedule,
    safe_commit_decision,
)


SCHEMA = "structsplat.fit043_sequential_error_pursuit.v1"
DEFAULT_INPUT = REPOSITORY_ROOT / "runs/janelle_cross_view_tail_diagnostic_20260728"
DEFAULT_OUT = REPOSITORY_ROOT / "runs/fit043_sequential_error_pursuit_20260728"
DEFAULT_REPORT_OUT = (
    REPOSITORY_ROOT / "ara/evidence/fit043-sequential-error-pursuit-janelle-2026-07-28"
)
DEFAULT_CAPTURE_ROOT = Path("/home/alex/Dropbox/Work/Janelle/2025_03_07_stage_with_fabric")
DEFAULT_REALTIME_ROOT = Path("/home/alex/Documents/realtime-gs")
FROZEN_INPUT_HASHES = {
    "manifest.json": "f8958a583c238cf649b9662b84130d75d2bf3afad7760f43ce22da9245ee6976",
    "summary.json": "c1cafe794fc73e8e32212d00b4edc9e8e48fe863d71c97195f8bd9db64bcc638",
    "comparison.csv": "bdde1bd87adba10bd9ee1e33bc189c963d597a4fc81ae4311739d85c4a24c564",
}
DETAIL_KEYS = (
    "detail_highpass_sigma_0_75_mse",
    "detail_highpass_sigma_1_5_mse",
    "detail_highpass_sigma_3_mse",
    "detail_laplacian_mse",
    "detail_residual_mse",
    "detail_sobel_mse",
)
SOURCE_FILES = (
    "scripts/experiments/fit043_sequential_error_pursuit.py",
    "scripts/experiments/audit_fit043_sequential_error_pursuit.py",
    "scripts/experiments/run_janelle_cross_view_tail_diagnostic.py",
    "scripts/experiments/fit032_janelle_dipole_screen.py",
    "scripts/experiments/fit033_janelle_highpass_solve.py",
    "scripts/experiments/fit040_janelle_production_pursuit.py",
    "src/structsplat/detail_pursuit.py",
    "src/structsplat/safe_schedule.py",
    "tests/test_fit043_sequential_error_pursuit.py",
    "tasks/FIT-043-sequential-error-pursuit-tail.md",
)
REPORT_IMAGE_NAMES = (
    "target.png",
    "base.png",
    "base_error.png",
    "error_only.png",
    "error_only_error.png",
    "pursuit.png",
    "pursuit_error.png",
    "error_then_pursuit.png",
    "error_then_pursuit_error.png",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def _atomic_json(path: Path, payload: Any) -> None:
    _atomic_text(
        path,
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
    )


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _git(*arguments: str) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPOSITORY_ROOT), *arguments],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _verify_sha256(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} is missing: {path}")
    actual = _sha256(path)
    if actual != expected:
        raise RuntimeError(f"{label} SHA-256 mismatch: expected {expected}, observed {actual}")


def _snapshot_sources(out: Path) -> list[dict[str, Any]]:
    records = []
    for relative in SOURCE_FILES:
        source = REPOSITORY_ROOT / relative
        if not source.is_file():
            raise FileNotFoundError(f"required FIT-043 source is missing: {source}")
        destination = out / "source_snapshot" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        records.append(
            {
                "path": relative,
                "sha256": _sha256(source),
                "bytes": source.stat().st_size,
            }
        )
    return records


def adjusted_stage_target(
    base_error: float,
    stage_entry_error: float,
    cumulative_target: float,
) -> float:
    """Map an original-base reduction target to the stage-entry reference."""

    values = (base_error, stage_entry_error, cumulative_target)
    if not all(np.isfinite(value) for value in values):
        raise ValueError("cumulative target inputs must be finite")
    if base_error < 0.0 or stage_entry_error < 0.0:
        raise ValueError("detail errors must be nonnegative")
    if not 0.0 <= cumulative_target <= 1.0:
        raise ValueError("cumulative_target must be in [0, 1]")
    threshold = base_error * (1.0 - cumulative_target)
    if stage_entry_error <= threshold or stage_entry_error <= 0.0:
        return 0.0
    return float(min(1.0, max(0.0, 1.0 - threshold / stage_entry_error)))


def _reduction(before: float, after: float) -> float:
    return 0.0 if before <= 0.0 else 1.0 - after / before


def _already_satisfied(
    highpass_reduction: float,
    laplacian_reduction: float,
    protected_safe: bool,
) -> bool:
    return bool(protected_safe and highpass_reduction >= 0.25 and laplacian_reduction >= 0.20)


def _normalized_json(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True, default=str))


def _metric_delta(
    observed: dict[str, Any],
    expected: dict[str, Any],
) -> tuple[bool, float, dict[str, float]]:
    deltas: dict[str, float] = {}
    valid = True
    for key, expected_value in expected.items():
        if key not in observed or isinstance(expected_value, bool):
            continue
        if not isinstance(expected_value, (int, float)):
            continue
        observed_value = float(observed[key])
        expected_float = float(expected_value)
        delta = abs(observed_value - expected_float)
        deltas[key] = delta
        tolerance = 1e-6 + 1e-5 * abs(expected_float)
        valid = bool(valid and np.isfinite(observed_value) and delta <= tolerance)
    return valid, max(deltas.values(), default=0.0), deltas


def _schedule(
    stage_entry_rows: int,
    highpass_target: float,
    laplacian_target: float,
    args: argparse.Namespace,
) -> SafeScheduleConfig:
    defaults = SafeScheduleConfig()
    return SafeScheduleConfig(
        capacity=stage_entry_rows,
        storage_policy="dynamic",
        boundary_enabled=True,
        coverage_target_gaussians=stage_entry_rows,
        detail_target_gaussians=stage_entry_rows,
        coverage_tau=float(args.coverage_tau),
        boundary_band=float(args.boundary_band),
        pursuit_tail_enabled=True,
        pursuit_tail_batch_rows=int(args.pursuit_batch_rows),
        pursuit_tail_max_rows=int(args.pursuit_max_rows),
        pursuit_tail_highpass_target=float(highpass_target),
        pursuit_tail_laplacian_target=float(laplacian_target),
        bootstrap=_disabled_phase(defaults.bootstrap, stage_entry_rows),
        coverage=_disabled_phase(defaults.coverage, stage_entry_rows),
        detail=_disabled_phase(defaults.detail, stage_entry_rows),
        boundary=_disabled_phase(defaults.boundary, stage_entry_rows),
        redistribution=_disabled_phase(defaults.redistribution, stage_entry_rows),
        polish=_disabled_phase(defaults.polish, stage_entry_rows),
    )


def _verify_fixed_input(args: argparse.Namespace) -> dict[str, Any]:
    for name, expected in FROZEN_INPUT_HASHES.items():
        _verify_sha256(args.input / name, expected, f"frozen input {name}")
    manifest = _load_json(args.input / "manifest.json")
    expected_protocol = {
        "status": "complete",
        "completed_cells": 51,
        "failed_cells": 0,
        "max_side": 1200,
        "seed": 0,
        "shared_renderer": "cuda",
    }
    mismatches = {
        key: (manifest.get(key), value)
        for key, value in expected_protocol.items()
        if manifest.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"frozen input protocol mismatch: {mismatches}")
    environment = manifest["environment"]
    current_gpu = (
        torch.cuda.get_device_name(torch.device(args.device))
        if torch.device(args.device).type == "cuda"
        else None
    )
    environment_checks = {
        "torch": torch.__version__ == environment["torch"],
        "torch_cuda": torch.version.cuda == environment["torch_cuda"],
        "pillow": PILLOW_VERSION == environment["pillow"],
        "gpu": current_gpu == environment["gpu"],
        "renderer": args.renderer == manifest["shared_renderer"],
        "seed": int(args.seed) == int(manifest["seed"]),
        "max_side": int(args.max_side) == int(manifest["max_side"]),
    }
    if not all(environment_checks.values()):
        raise RuntimeError(f"FIT-043 environment differs from frozen input: {environment_checks}")
    return manifest


def _prepare_cell(
    args: argparse.Namespace,
    frame: str,
    view_id: str,
    device: torch.device,
) -> dict[str, Any]:
    input_cell_dir = args.input / "cells" / frame / view_id
    input_cell_path = input_cell_dir / "result.json"
    input_cell = _load_json(input_cell_path)
    if input_cell.get("schema") != ("structsplat.janelle_cross_view_tail_diagnostic.v1.cell"):
        raise RuntimeError(f"unexpected input cell schema: {input_cell_path}")
    if not input_cell.get("eligible"):
        raise RuntimeError(f"frozen FIT-043 cell is not eligible: {frame}/{view_id}")
    source = input_cell["source"]
    base_record = input_cell["base"]
    error_record = input_cell["arms"]["error_only"]
    pursuit_record = input_cell["arms"]["pursuit"]
    paths = {
        "base": Path(base_record["path"]),
        "error_only": Path(error_record["field"]["path"]),
        "pursuit": Path(pursuit_record["field"]["path"]),
        "target": Path(source["target"]),
        "mask": Path(source["materialized_mask"]),
    }
    expected_hashes = {
        "base": base_record["sha256"],
        "error_only": error_record["field"]["sha256"],
        "pursuit": pursuit_record["field"]["sha256"],
        "target": source["target_sha256"],
        "mask": source["materialized_mask_sha256"],
    }
    for label, path in paths.items():
        _verify_sha256(path, expected_hashes[label], f"{frame}/{view_id} {label}")

    prepare_args = argparse.Namespace(
        realtime_root=args.realtime_root,
        capture_root=args.capture_root,
        frame=frame,
        view_id=view_id,
        max_side=int(args.max_side),
        field=Path(source["archive"]),
    )
    prepared = _prepare_janelle(prepare_args)
    source_checks = {
        "image_sha256": _sha256(Path(prepared["image_path"])) == source["image_sha256"],
        "mask_sha256": _sha256(Path(prepared["mask_path"])) == source["mask_sha256"],
        "fit_size": list(prepared["fit_size"]) == list(source["fit_size"]),
        "fit_window": list(prepared["fit_window"]) == list(source["fit_window"]),
        "pillow": PILLOW_VERSION == source["pillow"],
    }
    target_cpu = np.asarray(prepared["target"], dtype=np.float32)
    mask_cpu = np.asarray(prepared["mask"], dtype=bool)
    stored_target = np.asarray(Image.open(paths["target"]).convert("RGB"))
    stored_mask = np.asarray(Image.open(paths["mask"]).convert("L")) > 127
    source_checks["target_materialization"] = bool(
        np.array_equal(
            stored_target,
            np.rint(np.clip(target_cpu, 0.0, 1.0) * 255.0).astype(np.uint8),
        )
    )
    source_checks["mask_materialization"] = bool(np.array_equal(stored_mask, mask_cpu))
    if not all(source_checks.values()):
        raise RuntimeError(f"source replay mismatch for {frame}/{view_id}: {source_checks}")

    target = torch.as_tensor(
        target_cpu,
        device=device,
        dtype=torch.float32,
    ).contiguous()
    mask = torch.as_tensor(mask_cpu, device=device, dtype=torch.bool)
    cfg = replace(
        _base_config(args),
        color_solve_maxiter=1,
        color_solve_lambda=1e30,
    )
    if _normalized_json(asdict(cfg)) != _normalized_json(error_record["fit_config"]):
        raise RuntimeError(f"fit config mismatch for {frame}/{view_id}")
    constraint = _constraint(
        mask_cpu,
        target,
        cfg,
        float(args.boundary_band),
    )
    base = GaussianField.load(str(paths["base"]), device=device)
    error_field = GaussianField.load(str(paths["error_only"]), device=device)
    base_metrics, _, base_quality = _evaluate_all(
        base,
        target,
        mask,
        cfg,
        constraint,
        float(args.coverage_tau),
    )
    error_metrics, error_render, error_quality = _evaluate_all(
        error_field,
        target,
        mask,
        cfg,
        constraint,
        float(args.coverage_tau),
    )
    base_match, base_max_delta, base_deltas = _metric_delta(
        base_metrics,
        base_record["baseline"],
    )
    error_match, error_max_delta, error_deltas = _metric_delta(
        error_metrics,
        error_record["final"],
    )
    if not base_match or not error_match:
        raise RuntimeError(
            f"cold input rescore mismatch for {frame}/{view_id}: "
            f"base={base_max_delta}, error={error_max_delta}"
        )
    base_safe_error, base_error_reasons = safe_commit_decision(
        base_quality,
        error_quality,
        SafeScheduleConfig().tolerances,
        0.0,
    )
    return {
        "input_cell_path": input_cell_path,
        "input_cell_sha256": _sha256(input_cell_path),
        "input_cell": input_cell,
        "source": source,
        "base_record": base_record,
        "error_record": error_record,
        "pursuit_record": pursuit_record,
        "paths": paths,
        "base": base,
        "error_field": error_field,
        "target": target,
        "mask": mask,
        "cfg": cfg,
        "constraint": constraint,
        "base_metrics": base_metrics,
        "base_quality": base_quality,
        "error_metrics": error_metrics,
        "error_render": error_render,
        "error_quality": error_quality,
        "input_rescore": {
            "base_matches": base_match,
            "base_max_abs_delta": base_max_delta,
            "base_metric_deltas": base_deltas,
            "error_matches": error_match,
            "error_max_abs_delta": error_max_delta,
            "error_metric_deltas": error_deltas,
            "error_protected_safe_vs_base": bool(base_safe_error),
            "error_protected_reasons_vs_base": list(base_error_reasons),
        },
    }


def _run_cell(
    args: argparse.Namespace,
    frame: str,
    view_id: str,
    device: torch.device,
) -> dict[str, Any]:
    out_cell = args.out / "cells" / frame / view_id
    result_path = out_cell / "result.json"
    input_cell_path = args.input / "cells" / frame / view_id / "result.json"
    input_cell_sha256 = _sha256(input_cell_path)
    if result_path.is_file():
        cached = _load_json(result_path)
        if (
            cached.get("schema") != f"{SCHEMA}.cell"
            or cached.get("input_binding", {}).get("cell_result_sha256") != input_cell_sha256
        ):
            raise RuntimeError(f"stale cached FIT-043 result: {result_path}")
        return cached

    prepared = _prepare_cell(args, frame, view_id, device)
    base_metrics = prepared["base_metrics"]
    error_metrics = prepared["error_metrics"]
    error_record = prepared["error_record"]
    pursuit_record = prepared["pursuit_record"]
    base_hp = float(base_metrics["detail_highpass_sigma_1_5_mse"])
    base_lap = float(base_metrics["detail_laplacian_mse"])
    error_hp = float(error_metrics["detail_highpass_sigma_1_5_mse"])
    error_lap = float(error_metrics["detail_laplacian_mse"])
    error_hp_reduction = _reduction(base_hp, error_hp)
    error_lap_reduction = _reduction(base_lap, error_lap)
    error_protected = bool(
        prepared["input_rescore"]["error_protected_safe_vs_base"]
        and error_record["protected_safe"]
        and error_record["outside_exact_zero"]
    )
    skip = _already_satisfied(
        error_hp_reduction,
        error_lap_reduction,
        error_protected,
    )
    hp_stage_target = adjusted_stage_target(base_hp, error_hp, 0.25)
    lap_stage_target = adjusted_stage_target(base_lap, error_lap, 0.20)
    history: list[dict[str, Any]] = []
    schedule_dict = None
    schedule_seconds = 0.0
    attempted_steps = 0
    accepted_steps = 0
    if skip:
        disposition = "already_satisfied"
        final_field = prepared["error_field"]
        tail = {
            "enabled": False,
            "target_reached": True,
            "termination_reason": "already_satisfied_cumulative_target",
            "activated_rows": 0,
            "waves_attempted": 0,
            "waves_accepted": 0,
            "highpass_target": hp_stage_target,
            "laplacian_target": lap_stage_target,
            "seconds": 0.0,
            "waves": [],
        }
        elapsed = 0.0
    else:
        disposition = "pursuit_executed"
        schedule = _schedule(
            prepared["error_field"].n,
            hp_stage_target,
            lap_stage_target,
            args,
        )
        schedule_dict = asdict(schedule)
        torch.manual_seed(int(args.seed))
        started = time.perf_counter()
        scheduled = run_safe_schedule(
            prepared["error_field"],
            prepared["target"],
            prepared["mask"],
            prepared["cfg"],
            schedule,
            verbose=not args.quiet,
        )
        elapsed = time.perf_counter() - started
        final_field = scheduled["field"]
        tail = scheduled["pursuit_tail"]
        history = scheduled["history"]
        schedule_seconds = float(scheduled["seconds"])
        attempted_steps = int(scheduled["attempted_steps"])
        accepted_steps = int(scheduled["accepted_steps"])

    final_metrics, final_render, final_quality = _evaluate_all(
        final_field,
        prepared["target"],
        prepared["mask"],
        prepared["cfg"],
        prepared["constraint"],
        float(args.coverage_tau),
    )
    stage_safe, stage_reasons = safe_commit_decision(
        prepared["error_quality"],
        final_quality,
        SafeScheduleConfig().tolerances,
        0.0,
    )
    original_safe, original_reasons = safe_commit_decision(
        prepared["base_quality"],
        final_quality,
        SafeScheduleConfig().tolerances,
        0.0,
    )
    prefix_exact, prefix_checks = _prefix_exact(
        prepared["error_field"],
        final_field,
    )
    outside_exact = bool(
        float(final_metrics["outside_max_abs"]) == 0.0
        and float(final_metrics["outside_coverage_max"]) == 0.0
    )
    cumulative_reductions = {
        key: _reduction(
            float(base_metrics[key]),
            float(final_metrics[key]),
        )
        for key in DETAIL_KEYS
    }
    incremental_reductions = {
        key: _reduction(
            float(error_metrics[key]),
            float(final_metrics[key]),
        )
        for key in DETAIL_KEYS
    }
    target_reached = bool(
        cumulative_reductions["detail_highpass_sigma_1_5_mse"] >= 0.25
        and cumulative_reductions["detail_laplacian_mse"] >= 0.20
    )
    base_rows = int(prepared["base"].n)
    error_rows = int(prepared["error_field"].n - base_rows)
    incremental_rows = int(final_field.n - prepared["error_field"].n)
    total_tail_rows = int(final_field.n - base_rows)
    error_gain = float(error_metrics["foreground_psnr_db"]) - float(
        base_metrics["foreground_psnr_db"]
    )
    combined_gain = float(final_metrics["foreground_psnr_db"]) - float(
        base_metrics["foreground_psnr_db"]
    )
    retention = (
        combined_gain / error_gain
        if error_gain > 0.0
        else (float("inf") if combined_gain >= error_gain else float("-inf"))
    )
    overall_pass = bool(
        target_reached and stage_safe and original_safe and outside_exact and prefix_exact
    )

    field_path = out_cell / "field.npz"
    field_path.parent.mkdir(parents=True, exist_ok=True)
    final_field.save(str(field_path))
    _save_rgb(out_cell / "images/full/error_then_pursuit.png", final_render)
    bounds = tuple(int(value) for value in prepared["base_record"]["detail_crop_bounds_xyxy"])
    _save_crop(
        out_cell / "images/detail/error_then_pursuit.png",
        final_render,
        bounds,
    )
    _save_residual_crop(
        out_cell / "images/detail/error_then_pursuit_error.png",
        final_render,
        prepared["target"],
        bounds,
        float(prepared["base_record"]["residual_heatmap_p99_scale"]),
    )
    _atomic_json(out_cell / "history.json", history)
    payload = {
        "schema": f"{SCHEMA}.cell",
        "task": "FIT-043",
        "frame": frame,
        "view_id": view_id,
        "input_binding": {
            "cell_result": str(prepared["input_cell_path"].resolve()),
            "cell_result_sha256": prepared["input_cell_sha256"],
            "base_field": str(prepared["paths"]["base"].resolve()),
            "base_field_sha256": prepared["base_record"]["sha256"],
            "error_field": str(prepared["paths"]["error_only"].resolve()),
            "error_field_sha256": error_record["field"]["sha256"],
            "pursuit_field": str(prepared["paths"]["pursuit"].resolve()),
            "pursuit_field_sha256": pursuit_record["field"]["sha256"],
            "target": str(prepared["paths"]["target"].resolve()),
            "target_sha256": prepared["source"]["target_sha256"],
            "mask": str(prepared["paths"]["mask"].resolve()),
            "mask_sha256": prepared["source"]["materialized_mask_sha256"],
            "input_rescore": prepared["input_rescore"],
        },
        "controller": {
            "order": ["error_only", "orthogonal_pursuit"],
            "disposition": disposition,
            "cumulative_highpass_target": 0.25,
            "cumulative_laplacian_target": 0.20,
            "adjusted_stage_highpass_target": hp_stage_target,
            "adjusted_stage_laplacian_target": lap_stage_target,
            "target_transform": "max(0, 1 - base*(1-T)/stage_entry)",
            "error_already_satisfied": skip,
        },
        "fit_config": asdict(prepared["cfg"]),
        "schedule": schedule_dict,
        "tail": tail,
        "baseline": base_metrics,
        "stage_entry": error_metrics,
        "final": final_metrics,
        "cumulative_reductions": cumulative_reductions,
        "incremental_reductions": incremental_reductions,
        "target_reached_common_25hp_20lap": target_reached,
        "stage_protected_safe": bool(stage_safe),
        "stage_protected_reasons": list(stage_reasons),
        "original_base_protected_safe": bool(original_safe),
        "original_base_protected_reasons": list(original_reasons),
        "outside_exact_zero": outside_exact,
        "error_prefix_exact": prefix_exact,
        "error_prefix_checks": prefix_checks,
        "overall_pass": overall_pass,
        "rows": {
            "base": base_rows,
            "error_only_added": error_rows,
            "pursuit_only_added_reused": int(pursuit_record["activated_rows"]),
            "sequential_pursuit_added": incremental_rows,
            "combined_tail_added": total_tail_rows,
            "final": int(final_field.n),
        },
        "quality_comparison": {
            "error_only_foreground_psnr_gain_db": error_gain,
            "combined_foreground_psnr_gain_db": combined_gain,
            "combined_minus_error_foreground_psnr_db": (
                float(final_metrics["foreground_psnr_db"])
                - float(error_metrics["foreground_psnr_db"])
            ),
            "foreground_gain_retention_fraction": retention,
            "pursuit_only_foreground_psnr_gain_db": float(
                pursuit_record["foreground_psnr_gain_db"]
            ),
            "pursuit_only_highpass_reduction": float(
                pursuit_record["relative_reductions"]["detail_highpass_sigma_1_5_mse"]
            ),
            "pursuit_only_laplacian_reduction": float(
                pursuit_record["relative_reductions"]["detail_laplacian_mse"]
            ),
        },
        "field": {
            "path": str(field_path.resolve()),
            "sha256": _sha256(field_path),
            "rows": int(final_field.n),
        },
        "images": {
            "full": str((out_cell / "images/full/error_then_pursuit.png").resolve()),
            "detail": str((out_cell / "images/detail/error_then_pursuit.png").resolve()),
            "detail_error": str(
                (out_cell / "images/detail/error_then_pursuit_error.png").resolve()
            ),
        },
        "seconds": elapsed,
        "schedule_seconds": schedule_seconds,
        "attempted_steps": attempted_steps,
        "accepted_steps": accepted_steps,
    }
    _atomic_json(result_path, payload)
    return payload


def _stats(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {"mean": 0.0, "median": 0.0, "p25": 0.0, "p75": 0.0}
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p25": float(np.quantile(array, 0.25)),
        "p75": float(np.quantile(array, 0.75)),
    }


def _aggregate(
    cells: list[dict[str, Any]],
    failures: list[dict[str, Any]],
) -> dict[str, Any]:
    completed = [cell for cell in cells if cell.get("schema") == f"{SCHEMA}.cell"]
    executed = [
        cell for cell in completed if cell["controller"]["disposition"] == "pursuit_executed"
    ]
    skipped = [
        cell for cell in completed if cell["controller"]["disposition"] == "already_satisfied"
    ]
    retention_all = all(
        float(cell["quality_comparison"]["foreground_gain_retention_fraction"]) >= 0.95
        for cell in completed
    )
    median_delta = _stats(
        [
            float(cell["quality_comparison"]["combined_minus_error_foreground_psnr_db"])
            for cell in completed
        ]
    )["median"]
    detail_improves_all_executed = all(
        float(cell["incremental_reductions"]["detail_highpass_sigma_1_5_mse"]) > 0.0
        and float(cell["incremental_reductions"]["detail_laplacian_mse"]) > 0.0
        for cell in executed
    )
    sequential_rows = _stats(
        [float(cell["rows"]["sequential_pursuit_added"]) for cell in completed]
    )
    pursuit_only_rows = _stats(
        [float(cell["rows"]["pursuit_only_added_reused"]) for cell in completed]
    )
    rule_1 = bool(
        len(completed) == 51
        and not failures
        and all(
            cell["target_reached_common_25hp_20lap"]
            and cell["original_base_protected_safe"]
            and cell["outside_exact_zero"]
            and (
                cell["error_prefix_exact"]
                if cell["controller"]["disposition"] == "pursuit_executed"
                else True
            )
            for cell in completed
        )
    )
    rule_2 = bool(retention_all and median_delta >= -0.05)
    rule_3 = bool(detail_improves_all_executed)
    rule_4 = bool(sequential_rows["median"] <= pursuit_only_rows["median"])
    viable = bool(rule_1 and rule_2 and rule_3 and rule_4)
    decision = {
        "rule_1_all_cells_target_protected_prefix_zero": rule_1,
        "rule_2_global_gain_retained": rule_2,
        "rule_2_every_cell_retains_95_percent": retention_all,
        "rule_2_median_combined_minus_error_psnr_db": median_delta,
        "rule_3_every_executed_stage_improves_both_detail_metrics": rule_3,
        "rule_4_median_incremental_rows_no_more_than_pursuit_only": rule_4,
        "viable_dual_objective_exposed_data_option": viable,
        "production_interface_authorized": False,
        "default_change_authorized": False,
        "fit042_confirmation": False,
        "verdict": (
            "separate error-only then pursuit stages are viable on the exposed "
            "Janelle cross-view diagnostic"
            if viable
            else "reject the frozen sequential controller without retuning"
        ),
    }
    by_frame: dict[str, Any] = {}
    for frame in sorted({cell["frame"] for cell in completed}):
        subset = [cell for cell in completed if cell["frame"] == frame]
        by_frame[frame] = {
            "cells": len(subset),
            "target_reached": sum(
                bool(cell["target_reached_common_25hp_20lap"]) for cell in subset
            ),
            "skipped": sum(
                cell["controller"]["disposition"] == "already_satisfied" for cell in subset
            ),
            "incremental_rows": _stats(
                [float(cell["rows"]["sequential_pursuit_added"]) for cell in subset]
            ),
        }
    return {
        "schema": f"{SCHEMA}.summary",
        "task": "FIT-043",
        "scope": (
            "exposed correlated 51-cell Janelle diagnostic; one seed; "
            "natural unequal counts; not FIT-042"
        ),
        "requested_cells": 51,
        "completed_cells": len(completed),
        "failed_cells": len(failures),
        "pursuit_executed_cells": len(executed),
        "already_satisfied_cells": len(skipped),
        "target_reached_cells": sum(
            bool(cell["target_reached_common_25hp_20lap"]) for cell in completed
        ),
        "stage_protected_safe_cells": sum(bool(cell["stage_protected_safe"]) for cell in completed),
        "original_base_protected_safe_cells": sum(
            bool(cell["original_base_protected_safe"]) for cell in completed
        ),
        "outside_exact_zero_cells": sum(bool(cell["outside_exact_zero"]) for cell in completed),
        "executed_prefix_exact_cells": sum(bool(cell["error_prefix_exact"]) for cell in executed),
        "rows": {
            "error_only_added": _stats(
                [float(cell["rows"]["error_only_added"]) for cell in completed]
            ),
            "pursuit_only_added_reused": pursuit_only_rows,
            "sequential_pursuit_added_all_cells": sequential_rows,
            "sequential_pursuit_added_executed_cells": _stats(
                [float(cell["rows"]["sequential_pursuit_added"]) for cell in executed]
            ),
            "combined_tail_added": _stats(
                [float(cell["rows"]["combined_tail_added"]) for cell in completed]
            ),
        },
        "cumulative": {
            "highpass_reduction": _stats(
                [
                    float(cell["cumulative_reductions"]["detail_highpass_sigma_1_5_mse"])
                    for cell in completed
                ]
            ),
            "laplacian_reduction": _stats(
                [float(cell["cumulative_reductions"]["detail_laplacian_mse"]) for cell in completed]
            ),
            "foreground_psnr_gain_db": _stats(
                [
                    float(cell["quality_comparison"]["combined_foreground_psnr_gain_db"])
                    for cell in completed
                ]
            ),
            "combined_minus_error_foreground_psnr_db": _stats(
                [
                    float(cell["quality_comparison"]["combined_minus_error_foreground_psnr_db"])
                    for cell in completed
                ]
            ),
            "foreground_gain_retention_fraction": _stats(
                [
                    float(cell["quality_comparison"]["foreground_gain_retention_fraction"])
                    for cell in completed
                ]
            ),
        },
        "incremental_executed": {
            "highpass_reduction": _stats(
                [
                    float(cell["incremental_reductions"]["detail_highpass_sigma_1_5_mse"])
                    for cell in executed
                ]
            ),
            "laplacian_reduction": _stats(
                [float(cell["incremental_reductions"]["detail_laplacian_mse"]) for cell in executed]
            ),
        },
        "seconds_new_stage": _stats([float(cell["seconds"]) for cell in completed]),
        "by_frame": by_frame,
        "decision": decision,
        "failures": failures,
    }


def _csv_rows(cells: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for cell in cells:
        rows.append(
            {
                "frame": cell["frame"],
                "view_id": cell["view_id"],
                "disposition": cell["controller"]["disposition"],
                "target_reached": cell["target_reached_common_25hp_20lap"],
                "stage_protected_safe": cell["stage_protected_safe"],
                "original_base_protected_safe": cell["original_base_protected_safe"],
                "outside_exact_zero": cell["outside_exact_zero"],
                "error_prefix_exact": cell["error_prefix_exact"],
                "error_only_added_rows": cell["rows"]["error_only_added"],
                "pursuit_only_added_rows": cell["rows"]["pursuit_only_added_reused"],
                "sequential_pursuit_added_rows": cell["rows"]["sequential_pursuit_added"],
                "combined_tail_added_rows": cell["rows"]["combined_tail_added"],
                "adjusted_highpass_target": cell["controller"]["adjusted_stage_highpass_target"],
                "adjusted_laplacian_target": cell["controller"]["adjusted_stage_laplacian_target"],
                "combined_highpass_reduction": cell["cumulative_reductions"][
                    "detail_highpass_sigma_1_5_mse"
                ],
                "combined_laplacian_reduction": cell["cumulative_reductions"][
                    "detail_laplacian_mse"
                ],
                "incremental_highpass_reduction": cell["incremental_reductions"][
                    "detail_highpass_sigma_1_5_mse"
                ],
                "incremental_laplacian_reduction": cell["incremental_reductions"][
                    "detail_laplacian_mse"
                ],
                "error_only_foreground_psnr_gain_db": cell["quality_comparison"][
                    "error_only_foreground_psnr_gain_db"
                ],
                "combined_foreground_psnr_gain_db": cell["quality_comparison"][
                    "combined_foreground_psnr_gain_db"
                ],
                "combined_minus_error_foreground_psnr_db": cell["quality_comparison"][
                    "combined_minus_error_foreground_psnr_db"
                ],
                "foreground_gain_retention_fraction": cell["quality_comparison"][
                    "foreground_gain_retention_fraction"
                ],
                "seconds_new_stage": cell["seconds"],
                "termination_reason": cell["tail"]["termination_reason"],
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        _atomic_text(path, "")
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def _copy_report_images(
    args: argparse.Namespace,
    cells: list[dict[str, Any]],
) -> dict[tuple[str, str], dict[str, str]]:
    references = {}
    for cell in cells:
        frame = cell["frame"]
        view = cell["view_id"]
        destination = args.report_out / "images" / frame / view
        destination.mkdir(parents=True, exist_ok=True)
        input_detail = args.input / "cells" / frame / view / "images/detail"
        output_detail = args.out / "cells" / frame / view / "images/detail"
        per_cell = {}
        for name in REPORT_IMAGE_NAMES:
            source = (
                output_detail / name
                if name.startswith("error_then_pursuit")
                else input_detail / name
            )
            if not source.is_file():
                raise FileNotFoundError(f"report image is missing: {source}")
            target = destination / name
            shutil.copy2(source, target)
            per_cell[name] = str(target.relative_to(args.report_out))
        references[(frame, view)] = per_cell
    return references


def _report_html(
    summary: dict[str, Any],
    cells: list[dict[str, Any]],
    images: dict[tuple[str, str], dict[str, str]],
) -> str:
    decision = summary["decision"]
    rows = summary["rows"]
    cumulative = summary["cumulative"]
    verdict_class = "pass" if decision["viable_dual_objective_exposed_data_option"] else "fail"
    cards = []
    for cell in cells:
        frame = cell["frame"]
        view = cell["view_id"]
        refs = images[(frame, view)]
        disposition = cell["controller"]["disposition"]
        status = "pass" if cell["overall_pass"] else "fail"
        visual_items = (
            ("target.png", "Target"),
            ("base.png", "Base"),
            ("error_only.png", "Error-only"),
            ("pursuit.png", "Pursuit only"),
            ("error_then_pursuit.png", "Error → pursuit"),
        )
        visuals = "".join(
            f'<figure><a href="{html.escape(refs[name])}">'
            f'<img loading="lazy" src="{html.escape(refs[name])}" '
            f'alt="{html.escape(label)} {frame}/{view}"></a>'
            f"<figcaption>{html.escape(label)}</figcaption></figure>"
            for name, label in visual_items
        )
        error_items = (
            ("base_error.png", "Base |error|"),
            ("error_only_error.png", "Error-only |error|"),
            ("pursuit_error.png", "Pursuit-only |error|"),
            ("error_then_pursuit_error.png", "Combined |error|"),
        )
        error_visuals = "".join(
            f'<figure><a href="{html.escape(refs[name])}">'
            f'<img loading="lazy" src="{html.escape(refs[name])}" '
            f'alt="{html.escape(label)} {frame}/{view}"></a>'
            f"<figcaption>{html.escape(label)}</figcaption></figure>"
            for name, label in error_items
        )
        cards.append(
            f"""
<article class="cell {status}" data-disposition="{disposition}">
  <header><h3>{frame} / {view}</h3>
    <span class="tag">{disposition.replace("_", " ")}</span></header>
  <div class="numbers">
    <span><b>{cell["rows"]["error_only_added"]:,}</b> error rows</span>
    <span><b>+{cell["rows"]["sequential_pursuit_added"]:,}</b> pursuit rows</span>
    <span><b>{_pct(cell["cumulative_reductions"]["detail_highpass_sigma_1_5_mse"])}</b> HP</span>
    <span><b>{_pct(cell["cumulative_reductions"]["detail_laplacian_mse"])}</b> Lap</span>
    <span><b>{cell["quality_comparison"]["combined_foreground_psnr_gain_db"]:+.3f} dB</b> FG gain</span>
  </div>
  <div class="visuals">{visuals}</div>
  <details><summary>Matched residual views and checks</summary>
    <div class="visuals errors">{error_visuals}</div>
    <p>Adjusted pursuit targets:
      HP {cell["controller"]["adjusted_stage_highpass_target"]:.4f},
      Lap {cell["controller"]["adjusted_stage_laplacian_target"]:.4f}.
      Gain retained:
      {_pct(cell["quality_comparison"]["foreground_gain_retention_fraction"])}.
      Stage/base protected: {cell["stage_protected_safe"]}/{cell["original_base_protected_safe"]};
      prefix exact: {cell["error_prefix_exact"]}; outside zero: {cell["outside_exact_zero"]}.
    </p>
  </details>
</article>"""
        )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>FIT-043 · Error → pursuit sequence</title>
<style>
:root {{ color-scheme: dark; --bg:#0b1017; --panel:#121a24; --line:#26364a;
  --text:#e7edf5; --muted:#9dacbd; --cyan:#71e5d1; --amber:#f3bf64; --red:#ff8b8b; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--text);
  font:15px/1.5 system-ui,sans-serif; }}
main {{ max-width:1540px; margin:auto; padding:28px; }}
h1 {{ font-size:clamp(2rem,5vw,4.5rem); margin:.1em 0; letter-spacing:-.04em; }}
h2 {{ margin-top:2.5rem; }} .lede {{ max-width:1000px; color:var(--muted); font-size:1.1rem; }}
.verdict {{ border:1px solid var(--cyan); background:#102924; padding:18px 22px;
  border-radius:14px; margin:24px 0; }} .verdict.fail {{ border-color:var(--red); background:#301a20; }}
.metrics {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:10px; }}
.metric {{ background:var(--panel); border:1px solid var(--line); border-radius:12px; padding:16px; }}
.metric strong {{ display:block; color:var(--cyan); font-size:1.65rem; }}
.metric span {{ color:var(--muted); }}
table {{ width:100%; border-collapse:collapse; background:var(--panel); }}
th,td {{ border-bottom:1px solid var(--line); padding:10px; text-align:right; }}
th:first-child,td:first-child {{ text-align:left; }}
.controls {{ position:sticky; top:0; z-index:4; padding:12px 0; background:rgba(11,16,23,.94); }}
button {{ background:var(--panel); color:var(--text); border:1px solid var(--line);
  border-radius:999px; padding:8px 13px; margin-right:6px; cursor:pointer; }}
.grid {{ display:grid; gap:18px; }}
.cell {{ border:1px solid var(--line); background:var(--panel); border-radius:15px; padding:16px; }}
.cell.fail {{ border-color:var(--red); }} .cell header {{ display:flex; justify-content:space-between; }}
.cell h3 {{ margin:0; }} .tag {{ color:var(--amber); }}
.numbers {{ display:flex; flex-wrap:wrap; gap:14px; color:var(--muted); margin:8px 0 13px; }}
.numbers b {{ color:var(--text); }}
.visuals {{ display:grid; grid-template-columns:repeat(5,minmax(0,1fr)); gap:8px; }}
.visuals.errors {{ grid-template-columns:repeat(4,minmax(0,1fr)); margin-top:12px; }}
figure {{ margin:0; }} img {{ width:100%; display:block; border-radius:7px; image-rendering:auto; }}
figcaption {{ color:var(--muted); font-size:.82rem; margin-top:3px; }}
details {{ margin-top:10px; color:var(--muted); }} summary {{ cursor:pointer; color:var(--text); }}
.hidden {{ display:none; }} code {{ color:var(--cyan); }}
@media (max-width:900px) {{ .visuals,.visuals.errors {{ grid-template-columns:repeat(2,1fr); }} }}
</style>
</head>
<body><main>
<p>STRUCTSPLAT / FIT-043 / EXPOSED-DATA DIAGNOSTIC</p>
<h1>Error first. Detail second.</h1>
<p class="lede">The natural FIT-031 error tail is followed by FIT-040 pursuit only for the
remaining cumulative fine-detail deficit. Existing single-stage results are reused; only the
missing sequential arm was executed. The 51 cells are correlated Janelle views, not independent
FIT-042 confirmation.</p>
<section class="verdict {verdict_class}">
  <strong>Verdict: {html.escape(decision["verdict"])}</strong><br>
  This tests separate ordered stages—not a merged score. Production interface and default changes
  remain unauthorized.
</section>
<section class="metrics">
  <div class="metric"><strong>{summary["target_reached_cells"]}/51</strong><span>cumulative target hits</span></div>
  <div class="metric"><strong>{summary["already_satisfied_cells"]}</strong><span>pursuit stages skipped</span></div>
  <div class="metric"><strong>{rows["sequential_pursuit_added_all_cells"]["median"]:,.0f}</strong><span>median added pursuit rows</span></div>
  <div class="metric"><strong>{rows["combined_tail_added"]["median"]:,.0f}</strong><span>median total tail rows</span></div>
  <div class="metric"><strong>{_pct(cumulative["highpass_reduction"]["median"])}</strong><span>median cumulative HP reduction</span></div>
  <div class="metric"><strong>{_pct(cumulative["laplacian_reduction"]["median"])}</strong><span>median cumulative Lap reduction</span></div>
  <div class="metric"><strong>{cumulative["foreground_psnr_gain_db"]["median"]:+.3f} dB</strong><span>median foreground gain</span></div>
</section>
<h2>Stage comparison</h2>
<table>
<thead><tr><th>Arm</th><th>Target hits</th><th>Median added rows</th><th>Median HP</th><th>Median Lap</th><th>Median FG gain</th></tr></thead>
<tbody>
<tr><td>FIT-040 pursuit only (reused)</td><td>51/51</td>
<td>{rows["pursuit_only_added_reused"]["median"]:,.0f}</td><td>26.71%</td><td>26.90%</td><td>+0.051 dB</td></tr>
<tr><td>FIT-031 error only (reused)</td><td>7/51</td>
<td>{rows["error_only_added"]["median"]:,.0f}</td><td>12.78%</td><td>14.81%</td><td>+3.587 dB</td></tr>
<tr><td>Error → pursuit</td><td>{summary["target_reached_cells"]}/51</td>
<td>{rows["combined_tail_added"]["median"]:,.0f} total (+{rows["sequential_pursuit_added_all_cells"]["median"]:,.0f} pursuit)</td>
<td>{_pct(cumulative["highpass_reduction"]["median"])}</td>
<td>{_pct(cumulative["laplacian_reduction"]["median"])}</td>
<td>{cumulative["foreground_psnr_gain_db"]["median"]:+.3f} dB</td></tr>
</tbody></table>
<p class="lede">The combination is a dual-objective quality path, not a row-efficiency winner:
it carries the error tail's large natural row count. Pursuit-only remains the fine-detail choice;
error-only remains the global-fit choice.</p>
<h2>All 51 paired visual cells</h2>
<div class="controls">
  <button data-filter="all">All</button>
  <button data-filter="pursuit_executed">Pursuit executed</button>
  <button data-filter="already_satisfied">Already satisfied</button>
</div>
<section class="grid">{"".join(cards)}</section>
<h2>Scope and provenance</h2>
<p class="lede">Exact source/field hashes, adjusted targets, metrics, timings, prefix checks, and
termination records are in <a href="comparison.csv">comparison.csv</a>,
<a href="summary.json">summary.json</a>, <a href="audit.json">audit.json</a>, and
<a href="run.md">run.md</a>. Historical single-arm timings are not presented as newly measured.
No actual-byte, work-efficiency, independent-scene, or default claim follows.</p>
</main>
<script>
document.querySelectorAll("button[data-filter]").forEach(button => {{
  button.addEventListener("click", () => {{
    const wanted = button.dataset.filter;
    document.querySelectorAll(".cell").forEach(card => {{
      card.classList.toggle("hidden", wanted !== "all" && card.dataset.disposition !== wanted);
    }});
  }});
}});
</script>
</body></html>
"""


def _run_markdown(
    args: argparse.Namespace,
    manifest: dict[str, Any],
    summary: dict[str, Any],
) -> str:
    decision = summary["decision"]
    return f"""# FIT-043 sequential error-then-pursuit diagnostic

## Scope

This preregistered task reuses the audited base, FIT-031 error-only, and FIT-040 pursuit-only
outputs for all 51 non-reference Janelle cells and executes only the previously missing
`error_then_pursuit` arm. The cells are correlated views from two adjacent frames of one capture,
with one seed and natural unequal row counts. This is not FIT-042 independent confirmation,
equal-row evidence, actual-rate evidence, or a default decision.

## Frozen controller

1. Run error-only first by loading its exact persisted field.
2. Skip pursuit when error-only already reaches cumulative 25% high-pass and 20% Laplacian
   reduction relative to the original base.
3. Otherwise convert each cumulative target with
   `max(0, 1 - base*(1-T)/stage_entry)` and run the unchanged 128-row FIT-040 waves up to 2,048.
4. Freeze every base+error row and require both the stage-entry and original-base protected gates,
   exact outside zero, and the cumulative detail targets.

## Result

- Completed: {summary["completed_cells"]}/51; failures: {summary["failed_cells"]}.
- Pursuit executed: {summary["pursuit_executed_cells"]}; skipped as already satisfied:
  {summary["already_satisfied_cells"]}.
- Cumulative target hits: {summary["target_reached_cells"]}/51.
- Original-base protected safe / outside exact zero / executed prefix exact:
  {summary["original_base_protected_safe_cells"]}/51 /
  {summary["outside_exact_zero_cells"]}/51 /
  {summary["executed_prefix_exact_cells"]}/{summary["pursuit_executed_cells"]}.
- Median incremental pursuit rows:
  {summary["rows"]["sequential_pursuit_added_all_cells"]["median"]:,.0f};
  median total combined tail rows:
  {summary["rows"]["combined_tail_added"]["median"]:,.0f}.
- Median cumulative high-pass/Laplacian reductions:
  {_pct(summary["cumulative"]["highpass_reduction"]["median"])} /
  {_pct(summary["cumulative"]["laplacian_reduction"]["median"])}.
- Median combined foreground-PSNR gain:
  {summary["cumulative"]["foreground_psnr_gain_db"]["median"]:+.6f} dB;
  median change from error-only:
  {summary["cumulative"]["combined_minus_error_foreground_psnr_db"]["median"]:+.6f} dB.

Frozen decision: **{decision["verdict"]}**. All four rules:
`{decision["rule_1_all_cells_target_protected_prefix_zero"]}`,
`{decision["rule_2_global_gain_retained"]}`,
`{decision["rule_3_every_executed_stage_improves_both_detail_metrics"]}`,
`{decision["rule_4_median_incremental_rows_no_more_than_pursuit_only"]}`.

Even on a pass, the result supports only the feasibility of separate ordered stages on this
exposed capture. Pursuit-only remains the row-efficient fine-detail arm; error-only remains the
global-fit arm. A production opt-in combination needs a later interface task and ADR amendment.

## Provenance

- Frozen input: `{args.input.resolve()}`
- Input hashes: `{FROZEN_INPUT_HASHES}`
- Raw output: `{args.out.resolve()}`
- Command: `{manifest["command"]}`
- Environment: `{manifest["environment"]}`
- Executed source snapshot: `{manifest["source_snapshot"]}`

## Reproduction

```bash
PYTHONPATH=src:. python scripts/experiments/fit043_sequential_error_pursuit.py --quiet
PYTHONPATH=src:. python scripts/experiments/audit_fit043_sequential_error_pursuit.py
```
"""


def _build_report(
    args: argparse.Namespace,
    manifest: dict[str, Any],
    summary: dict[str, Any],
    cells: list[dict[str, Any]],
) -> None:
    args.report_out.mkdir(parents=True, exist_ok=True)
    _write_csv(args.report_out / "comparison.csv", _csv_rows(cells))
    _atomic_json(args.report_out / "summary.json", summary)
    images = _copy_report_images(args, cells)
    _atomic_text(
        args.report_out / "index.html",
        _report_html(summary, cells, images),
    )
    _atomic_text(
        args.report_out / "run.md",
        _run_markdown(args, manifest, summary),
    )
    artifact = {
        "schema": f"{SCHEMA}.artifact",
        "title": "FIT-043 sequential error-then-pursuit Janelle diagnostic",
        "source_run": str(args.out.resolve()),
        "source_manifest_sha256": _sha256(args.out / "manifest.json"),
        "summary_sha256": _sha256(args.report_out / "summary.json"),
        "comparison_csv_sha256": _sha256(args.report_out / "comparison.csv"),
        "index_sha256": _sha256(args.report_out / "index.html"),
        "run_markdown_sha256": _sha256(args.report_out / "run.md"),
        "cells": len(cells),
        "report_images": len(cells) * len(REPORT_IMAGE_NAMES),
    }
    _atomic_json(args.report_out / "artifact.json", artifact)


def _manifest(
    args: argparse.Namespace,
    input_manifest: dict[str, Any],
    source_snapshot: list[dict[str, Any]],
) -> dict[str, Any]:
    device = torch.device(args.device)
    return {
        "schema": f"{SCHEMA}.manifest",
        "task": "FIT-043",
        "status": "in_progress",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_status": "preregistered_exposed_data_sequence_diagnostic",
        "fit042_confirmation": False,
        "command": [sys.executable, *sys.argv],
        "input": str(args.input.resolve()),
        "input_hashes": FROZEN_INPUT_HASHES,
        "input_requested_cells": input_manifest["requested_cells"],
        "newly_executed_arms": ["error_then_pursuit"],
        "reused_arms": ["base", "error_only", "pursuit_only"],
        "controller": {
            "order": ["error_only", "orthogonal_pursuit"],
            "cumulative_highpass_target": 0.25,
            "cumulative_laplacian_target": 0.20,
            "target_transform": "max(0, 1 - base*(1-T)/stage_entry)",
            "batch_rows": int(args.pursuit_batch_rows),
            "max_rows": int(args.pursuit_max_rows),
        },
        "source_snapshot": source_snapshot,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "device": str(device),
            "gpu": (torch.cuda.get_device_name(device) if device.type == "cuda" else None),
            "pillow": PILLOW_VERSION,
            "renderer": args.renderer,
            "git_commit": _git("rev-parse", "HEAD"),
            "git_dirty": bool(_git("status", "--porcelain")),
        },
    }


def run(args: argparse.Namespace) -> None:
    args.input = args.input.resolve()
    args.out = args.out.resolve()
    args.report_out = args.report_out.resolve()
    args.capture_root = args.capture_root.resolve()
    args.realtime_root = args.realtime_root.resolve()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA but torch.cuda.is_available() is false")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    input_manifest = _verify_fixed_input(args)
    requested = [
        (record["frame"], record["view_id"]) for record in input_manifest["requested_cells"]
    ]
    if len(requested) != 51:
        raise RuntimeError(f"FIT-043 requires exactly 51 cells, observed {len(requested)}")
    if args.limit is not None:
        requested = requested[: int(args.limit)]
    args.out.mkdir(parents=True, exist_ok=True)
    snapshot_path = args.out / "source_snapshot.json"
    if snapshot_path.is_file():
        source_snapshot = _load_json(snapshot_path)
        for record in source_snapshot:
            _verify_sha256(
                args.out / "source_snapshot" / record["path"],
                record["sha256"],
                f"executed source snapshot {record['path']}",
            )
    else:
        source_snapshot = _snapshot_sources(args.out)
        _atomic_json(snapshot_path, source_snapshot)
    manifest_path = args.out / "manifest.json"
    manifest = _manifest(args, input_manifest, source_snapshot)
    if manifest_path.is_file():
        prior = _load_json(manifest_path)
        if (
            prior.get("input_hashes") != manifest["input_hashes"]
            or prior.get("controller") != manifest["controller"]
        ):
            raise RuntimeError("existing FIT-043 run has a different frozen protocol")
        manifest["created_at_utc"] = prior["created_at_utc"]
    _atomic_json(manifest_path, manifest)

    completed: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for index, (frame, view_id) in enumerate(requested, start=1):
        print(f"[{index}/{len(requested)}] {frame}/{view_id}", flush=True)
        try:
            cell = _run_cell(args, frame, view_id, device)
            completed.append(cell)
            print(
                f"  {cell['controller']['disposition']}: "
                f"+{cell['rows']['sequential_pursuit_added']} rows, "
                f"HP={_pct(cell['cumulative_reductions']['detail_highpass_sigma_1_5_mse'])}, "
                f"Lap={_pct(cell['cumulative_reductions']['detail_laplacian_mse'])}",
                flush=True,
            )
        except Exception as error:  # noqa: BLE001 - preserve per-cell failures
            failure = {
                "frame": frame,
                "view_id": view_id,
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            }
            failures.append(failure)
            failure_path = args.out / "cells" / frame / view_id / "failure.json"
            _atomic_json(failure_path, failure)
            print(f"  FAILED: {error}", file=sys.stderr, flush=True)
            if args.fail_fast:
                raise
        _atomic_json(
            args.out / "progress.json",
            {
                "requested": len(requested),
                "completed": len(completed),
                "failures": failures,
                "last_cell": {"frame": frame, "view_id": view_id},
            },
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()

    summary = _aggregate(completed, failures)
    _atomic_json(args.out / "summary.json", summary)
    _write_csv(args.out / "comparison.csv", _csv_rows(completed))
    manifest["status"] = (
        "complete" if not failures and len(completed) == len(requested) == 51 else "partial"
    )
    manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["completed_cells"] = len(completed)
    manifest["failed_cells"] = len(failures)
    manifest["summary_sha256"] = _sha256(args.out / "summary.json")
    _atomic_json(manifest_path, manifest)
    _build_report(args, manifest, summary, completed)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--report-out", type=Path, default=DEFAULT_REPORT_OUT)
    parser.add_argument("--capture-root", type=Path, default=DEFAULT_CAPTURE_ROOT)
    parser.add_argument("--realtime-root", type=Path, default=DEFAULT_REALTIME_ROOT)
    parser.add_argument("--max-side", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--renderer", default="cuda")
    parser.add_argument("--mask-margin", type=float, default=0.75)
    parser.add_argument("--boundary-band", type=float, default=4.0)
    parser.add_argument("--coverage-tau", type=float, default=0.05)
    parser.add_argument("--pursuit-batch-rows", type=int, default=128)
    parser.add_argument("--pursuit-max-rows", type=int, default=2048)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
