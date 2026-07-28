#!/usr/bin/env python3
"""Nested minimum-row detail-target curve for FIT-037 on Janelle C0001."""

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
from benchmarks.residual_birth_color_solve import (  # noqa: E402
    solve_new_row_colors,
)
from scripts.experiments.fit032_janelle_dipole_screen import (  # noqa: E402
    _base_config,
    _prepare_current_job,
    _scaled_field,
)
from scripts.experiments.fit033_janelle_highpass_solve import (  # noqa: E402
    _constraint_delta,
    _evaluate_all,
    _prefix_components,
    _solve_metadata,
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
DEFAULT_OUT = REPOSITORY_ROOT / "runs/fit037_janelle_minimum_detail_rows_20260728"
BUDGETS = (128, 256, 384, 512, 768, 1024, 1536, 2048)
SOURCE_FILES = (
    "benchmarks/highpass_births.py",
    "benchmarks/residual_birth_color_solve.py",
    "scripts/experiments/fit037_janelle_minimum_detail_rows.py",
    "tasks/FIT-037-minimum-row-detail-target.md",
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
        max(BUDGETS),
        blur_sigma=1.5,
        nms_radius=2,
        deep_offset=6.0,
        scale=0.35,
        opacity=0.8,
    )
    schedule = SafeScheduleConfig(
        capacity=base.n + max(BUDGETS),
        coverage_tau=args.coverage_tau,
        boundary_band=args.boundary_band,
    )
    rows = []
    for budget in BUDGETS:
        started = time.perf_counter()
        proposal = base.append(
            _prefix_components(selection.components, budget)
        )
        constraint_delta = _constraint_delta(proposal, cfg, constraint)
        solve = solve_new_row_colors(
            proposal,
            target,
            cfg,
            torch.arange(
                base.n,
                proposal.n,
                device=device,
                dtype=torch.long,
            ),
            ridge=1e-4,
            max_iterations=48,
            tolerance=1e-7,
        )
        metrics, rendered, quality = _evaluate_all(
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
        sigma_reduction = 1.0 - (
            metrics["detail_highpass_sigma_1_5_mse"]
            / baseline["detail_highpass_sigma_1_5_mse"]
        )
        laplacian_reduction = 1.0 - (
            metrics["detail_laplacian_mse"]
            / baseline["detail_laplacian_mse"]
        )
        passed = (
            safe
            and sigma_reduction >= 0.25
            and laplacian_reduction >= 0.20
        )
        row = {
            "budget": budget,
            "n_before": base.n,
            "n_after": solve.field.n,
            "seconds": time.perf_counter() - started,
            "protected_safe": safe,
            "protected_reasons": reasons,
            "constraint_delta": constraint_delta,
            "render_min": float(rendered.min()),
            "render_max": float(rendered.max()),
            "solve": _solve_metadata(solve),
            "prefix_score_min": float(selection.scores[:budget].min()),
            "prefix_score_mean": float(selection.scores[:budget].mean()),
            "metrics": metrics,
            "sigma_1_5_reduction": sigma_reduction,
            "laplacian_reduction": laplacian_reduction,
            "target_passed": passed,
            "rows_fraction_of_fit031_additions": budget / 4608.0,
            "rows_fraction_of_base": budget / base.n,
        }
        rows.append(row)
        _atomic_json(args.out / "partial_results.json", rows)
        print(
            f"K={budget:4d} HP={100 * sigma_reduction:.3f}% "
            f"Lap={100 * laplacian_reduction:.3f}% "
            f"PSNR={metrics['foreground_psnr_db']:.6f} "
            f"safe={safe} target={passed}"
        )

    first_pass = next(
        (row for row in rows if row["target_passed"]),
        None,
    )
    result = {
        "schema": "structsplat.fit037.janelle-minimum-detail-rows.v1",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "development_only": True,
        "base_job": str(args.base_job.resolve()),
        "base_field_sha256": _sha256(prepared["field_path"]),
        "protocol": {
            "budgets": BUDGETS,
            "sigma_1_5_target": 0.25,
            "laplacian_target": 0.20,
            "required_protected_safe": True,
            "fit031_accepted_additions": 4608,
            "selection": selection.metadata,
            "solve_ridge": 1e-4,
            "solve_max_iterations": 48,
            "fit_config": asdict(cfg),
        },
        "baseline": baseline,
        "rows": rows,
        "decision": {
            "target_reached": first_pass is not None,
            "minimum_rows": (
                None if first_pass is None else first_pass["budget"]
            ),
            "minimum_rows_fraction_of_fit031": (
                None
                if first_pass is None
                else first_pass["rows_fraction_of_fit031_additions"]
            ),
            "production_promotion_authorized": False,
        },
        "sources": [
            {
                "path": relative,
                "sha256": _sha256(REPOSITORY_ROOT / relative),
            }
            for relative in SOURCE_FILES
        ],
    }
    _atomic_json(args.out / "result.json", result)
    print(json.dumps(result["decision"], indent=2))


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
