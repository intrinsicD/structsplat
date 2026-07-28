#!/usr/bin/env python3
"""Bounded anisotropy screen for FIT-036 high-pass residual-ridge births."""

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

from benchmarks.highpass_ridge_births import (  # noqa: E402
    select_highpass_ridge_births,
)
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
DEFAULT_OUT = REPOSITORY_ROOT / "runs/fit036_janelle_ridge_screen_20260728"
MAX_LONG_SCALES = (0.5, 0.75, 1.0, 1.5)
COHERENCE_POWERS = (0.5, 1.0, 2.0)
SOURCE_FILES = (
    "benchmarks/highpass_ridge_births.py",
    "benchmarks/residual_birth_color_solve.py",
    "scripts/experiments/fit036_janelle_ridge_screen.py",
    "tasks/FIT-036-highpass-residual-ridge-births.md",
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
    schedule = SafeScheduleConfig(
        capacity=base.n + 128,
        coverage_tau=args.coverage_tau,
        boundary_band=args.boundary_band,
    )
    rows = []
    for max_long_scale in MAX_LONG_SCALES:
        for coherence_power in COHERENCE_POWERS:
            started = time.perf_counter()
            selection = select_highpass_ridge_births(
                base,
                target,
                base_render,
                constraint,
                128,
                blur_sigma=1.5,
                nms_radius=2,
                deep_offset=6.0,
                short_scale=0.35,
                max_long_scale=max_long_scale,
                coherence_power=coherence_power,
                opacity=0.8,
            )
            proposal = base.append(selection.components)
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
                max_iterations=32,
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
            row = {
                "max_long_scale": max_long_scale,
                "coherence_power": coherence_power,
                "seconds": time.perf_counter() - started,
                "protected_safe": safe,
                "protected_reasons": reasons,
                "constraint_delta": constraint_delta,
                "render_min": float(rendered.min()),
                "render_max": float(rendered.max()),
                "solve": _solve_metadata(solve),
                "selection": selection.metadata,
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
                f"long={max_long_scale:g} power={coherence_power:g} "
                f"HP={100 * row['sigma_1_5_reduction']:.3f}% "
                f"Lap={100 * row['laplacian_reduction']:.3f}% "
                f"safe={safe}"
            )

    eligible = [row for row in rows if row["protected_safe"]]
    selected = (
        None
        if not eligible
        else max(eligible, key=lambda row: row["sigma_1_5_reduction"])
    )
    result = {
        "schema": "structsplat.fit036.janelle-ridge-screen.v1",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "development_only": True,
        "base_job": str(args.base_job.resolve()),
        "base_field_sha256": _sha256(prepared["field_path"]),
        "protocol": {
            "budget": 128,
            "max_long_scales": MAX_LONG_SCALES,
            "coherence_powers": COHERENCE_POWERS,
            "short_scale": 0.35,
            "opacity": 0.8,
            "fit_config": asdict(cfg),
        },
        "baseline": baseline,
        "rows": rows,
        "selection_rule": (
            "maximum sigma-1.5 reduction among protected-safe rows"
        ),
        "selected": (
            None
            if selected is None
            else {
                key: selected[key]
                for key in (
                    "max_long_scale",
                    "coherence_power",
                    "sigma_1_5_reduction",
                    "laplacian_reduction",
                )
            }
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
    print(json.dumps(result["selected"], indent=2))


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
