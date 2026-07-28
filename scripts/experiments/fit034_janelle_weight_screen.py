#!/usr/bin/env python3
"""Bounded raw-RGB-weight screen for FIT-034 on exposed Janelle C0001."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
for _root in (REPOSITORY_ROOT, REPOSITORY_ROOT / "src"):
    _text = str(_root)
    if _text not in sys.path:
        sys.path.insert(0, _text)

from benchmarks.highpass_births import select_highpass_births  # noqa: E402
from benchmarks.spectral_color_solve import (  # noqa: E402
    solve_new_row_colors_spectral,
)
from scripts.experiments.fit032_janelle_dipole_screen import (  # noqa: E402
    _base_config,
    _prepare_current_job,
    _scaled_field,
)
from scripts.experiments.fit033_janelle_highpass_solve import (  # noqa: E402
    _constraint_delta,
    _evaluate_all,
)
from structsplat.fit import _MaskConstraint  # noqa: E402
from structsplat.safe_schedule import (  # noqa: E402
    SafeScheduleConfig,
    safe_commit_decision,
)


DEFAULT_BASE_JOB = (
    REPOSITORY_ROOT
    / "runs/fit032_current_base_20260728/runs/current/C0001/seed_0"
)
DEFAULT_OUT = REPOSITORY_ROOT / "runs/fit034_janelle_weight_screen_20260728"
RAW_WEIGHTS = (0.0, 0.01, 0.03, 0.1, 0.3, 1.0)
SOURCE_FILES = (
    "benchmarks/highpass_births.py",
    "benchmarks/spectral_color_solve.py",
    "scripts/experiments/fit034_janelle_weight_screen.py",
    "tasks/FIT-034-spectral-partial-color-solve.md",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def run(args: argparse.Namespace) -> None:
    if args.out.exists() and any(args.out.iterdir()):
        raise RuntimeError(f"output directory is not empty: {args.out}")
    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    device = torch.device(args.device)
    prepared = _prepare_current_job(args.base_job)
    target = torch.as_tensor(
        prepared["target"], device=device, dtype=torch.float32
    ).contiguous()
    mask = torch.as_tensor(
        prepared["mask"], device=device, dtype=torch.bool
    )
    cfg = _base_config(args)
    base = _scaled_field(prepared["field_path"], device, 1.0, 1.0)
    constraint = _MaskConstraint.from_mask(
        prepared["mask"],
        device,
        target.dtype,
        cfg.sigma_cutoff,
        cfg.mask_margin,
        aa_dilation=cfg.aa_dilation,
        min_scale=0.35,
        cap_mode=cfg.mask_cap_mode,
        undercoverage_band=cfg.mask_undercoverage_band,
    )
    _constraint_delta(base, cfg, constraint)
    baseline, base_render, base_quality = _evaluate_all(
        base, target, mask, cfg, constraint, args.coverage_tau
    )
    selection = select_highpass_births(
        base,
        target,
        base_render,
        constraint,
        128,
        blur_sigma=1.5,
        nms_radius=2,
        deep_offset=6.0,
        scale=0.35,
        opacity=0.8,
    )
    proposal = base.append(selection.components)
    _constraint_delta(proposal, cfg, constraint)
    new_rows = torch.arange(
        base.n, proposal.n, device=device, dtype=torch.long
    )
    deep = (
        constraint.sdf_flat.reshape(mask.shape)
        > float(constraint.margin) + 6.0
    )
    schedule = SafeScheduleConfig(
        capacity=proposal.n,
        coverage_tau=args.coverage_tau,
        boundary_band=args.boundary_band,
    )

    rows = []
    for raw_weight in RAW_WEIGHTS:
        started = time.perf_counter()
        solve = solve_new_row_colors_spectral(
            proposal,
            target,
            cfg,
            new_rows,
            deep,
            mask,
            sigma=1.5,
            raw_weight=raw_weight,
            ridge=1e-4,
            max_iterations=48,
            tolerance=1e-7,
        )
        metrics, _, quality = _evaluate_all(
            solve.field,
            target,
            mask,
            cfg,
            constraint,
            args.coverage_tau,
        )
        safe, reasons = safe_commit_decision(
            base_quality,
            quality,
            schedule.tolerances,
            schedule.hole_regression_budget,
        )
        row = {
            "raw_weight": raw_weight,
            "seconds": time.perf_counter() - started,
            "protected_safe": safe,
            "protected_reasons": reasons,
            "color_min": solve.raw_color_min,
            "color_max": solve.raw_color_max,
            "iterations": solve.iterations,
            "converged": solve.converged,
            "relative_residual": solve.relative_residual,
            "initial_objective": solve.initial_objective,
            "final_objective": solve.final_objective,
            "metrics": metrics,
            "sigma_1_5_reduction": 1.0
            - metrics["detail_highpass_sigma_1_5_mse"]
            / baseline["detail_highpass_sigma_1_5_mse"],
            "laplacian_reduction": 1.0
            - metrics["detail_laplacian_mse"]
            / baseline["detail_laplacian_mse"],
        }
        rows.append(row)
        _atomic_json(args.out / "partial_results.json", rows)
        print(
            f"raw_weight={raw_weight:g} "
            f"HP={100 * row['sigma_1_5_reduction']:.3f}% "
            f"Lap={100 * row['laplacian_reduction']:.3f}% "
            f"colors=[{solve.raw_color_min:.3f},{solve.raw_color_max:.3f}] "
            f"safe={safe}"
        )

    eligible = [
        row
        for row in rows
        if row["protected_safe"]
        and row["color_min"] >= -2.0
        and row["color_max"] <= 2.0
    ]
    selected = (
        None
        if not eligible
        else max(eligible, key=lambda row: row["sigma_1_5_reduction"])
    )
    result = {
        "schema": "structsplat.fit034.janelle-weight-screen.v1",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "development_only": True,
        "base_job": str(args.base_job.resolve()),
        "base_field_sha256": _sha256(prepared["field_path"]),
        "protocol": {
            "budget": 128,
            "raw_weights": RAW_WEIGHTS,
            "sigma": 1.5,
            "ridge": 1e-4,
            "max_iterations": 48,
            "selection": selection.metadata,
            "fit_config": asdict(cfg),
        },
        "baseline": baseline,
        "rows": rows,
        "selection_rule": (
            "maximum sigma-1.5 reduction among protected-safe rows with "
            "new colors within [-2,2]"
        ),
        "selected_raw_weight": (
            None if selected is None else selected["raw_weight"]
        ),
        "sources": [
            {
                "path": relative,
                "sha256": _sha256(REPOSITORY_ROOT / relative),
            }
            for relative in SOURCE_FILES
        ],
    }
    _atomic_json(args.out / "result.json", result)
    print(f"selected_raw_weight={result['selected_raw_weight']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-job", type=Path, default=DEFAULT_BASE_JOB)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--renderer", default="cuda_tiled")
    parser.add_argument("--coverage-tau", type=float, default=0.05)
    parser.add_argument("--boundary-band", type=float, default=4.0)
    parser.add_argument("--mask-margin", type=float, default=0.75)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
