#!/usr/bin/env python3
"""Bounded scale/ridge/objective screen for FIT-035 sparse affine births."""

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

from benchmarks.affine_partial_color_solve import (  # noqa: E402
    solve_new_row_affine_colors,
)
from benchmarks.highpass_births import select_highpass_births  # noqa: E402
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
DEFAULT_OUT = REPOSITORY_ROOT / "runs/fit035_janelle_affine_screen_20260728"
SCALES = (0.35, 0.5, 0.75)
GRADIENT_RIDGES = (1e-4, 1e-3, 1e-2)
OBJECTIVES = ("raw", "spectral")
SOURCE_FILES = (
    "benchmarks/affine_partial_color_solve.py",
    "benchmarks/highpass_births.py",
    "scripts/experiments/fit035_janelle_affine_screen.py",
    "tasks/FIT-035-sparse-affine-detail-births.md",
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
    if cfg.renderer != "normalized":
        raise ValueError("FIT-035 screen requires the exact reference normalized renderer")
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
    affine_base = base.detached().with_affine_colors()
    affine_metrics, affine_render, affine_quality = _evaluate_all(
        affine_base,
        target,
        mask,
        cfg,
        constraint,
        args.coverage_tau,
    )
    aa_safe, aa_reasons = safe_commit_decision(
        base_quality,
        affine_quality,
        SafeScheduleConfig().tolerances,
    )
    renderer_aa = {
        "max_abs_render_delta": float((affine_render - base_render).abs().max()),
        "foreground_mse_delta": (
            affine_metrics["foreground_mse"] - baseline["foreground_mse"]
        ),
        "protected_safe": aa_safe,
        "protected_reasons": aa_reasons,
    }
    if renderer_aa["max_abs_render_delta"] > 2e-6 or not aa_safe:
        raise RuntimeError(f"affine zero-gradient renderer A/A failed: {renderer_aa}")

    deep = (
        constraint.sdf_flat.reshape(mask.shape)
        > float(constraint.margin) + 6.0
    )
    schedule = SafeScheduleConfig(
        capacity=base.n + 128,
        coverage_tau=args.coverage_tau,
        boundary_band=args.boundary_band,
    )
    rows = []
    for scale in SCALES:
        selection = select_highpass_births(
            base,
            target,
            base_render,
            constraint,
            128,
            blur_sigma=1.5,
            nms_radius=2,
            deep_offset=6.0,
            scale=scale,
            opacity=0.8,
        )
        proposal = base.append(selection.components)
        constraint_delta = _constraint_delta(proposal, cfg, constraint)
        new_rows = torch.arange(
            base.n, proposal.n, device=device, dtype=torch.long
        )
        for gradient_ridge in GRADIENT_RIDGES:
            for objective in OBJECTIVES:
                started = time.perf_counter()
                solve = solve_new_row_affine_colors(
                    proposal,
                    target,
                    cfg,
                    new_rows,
                    detail_mask=(deep if objective == "spectral" else None),
                    raw_mask=(mask if objective == "spectral" else None),
                    sigma=1.5,
                    raw_weight=0.1,
                    color_ridge=1e-4,
                    gradient_ridge=gradient_ridge,
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
                row = {
                    "scale": scale,
                    "gradient_ridge": gradient_ridge,
                    "objective": objective,
                    "seconds": time.perf_counter() - started,
                    "protected_safe": safe,
                    "protected_reasons": reasons,
                    "constraint_delta": constraint_delta,
                    "render_min": float(rendered.min()),
                    "render_max": float(rendered.max()),
                    "raw_color_min": solve.raw_color_min,
                    "raw_color_max": solve.raw_color_max,
                    "gradient_min": solve.gradient_min,
                    "gradient_max": solve.gradient_max,
                    "gradient_max_abs": solve.gradient_max_abs,
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
                    f"scale={scale:g} ridge={gradient_ridge:g} "
                    f"obj={objective:8s} "
                    f"HP={100 * row['sigma_1_5_reduction']:.3f}% "
                    f"Lap={100 * row['laplacian_reduction']:.3f}% "
                    f"|grad|={solve.gradient_max_abs:.3f} safe={safe}"
                )

    eligible = [
        row
        for row in rows
        if row["protected_safe"]
        and row["gradient_max_abs"] <= 2.0
        and row["converged"]
    ]
    selected = (
        None
        if not eligible
        else max(eligible, key=lambda row: row["sigma_1_5_reduction"])
    )
    result = {
        "schema": "structsplat.fit035.janelle-affine-screen.v1",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "development_only": True,
        "base_job": str(args.base_job.resolve()),
        "base_field_sha256": _sha256(prepared["field_path"]),
        "renderer_aa": renderer_aa,
        "protocol": {
            "budget": 128,
            "scales": SCALES,
            "gradient_ridges": GRADIENT_RIDGES,
            "objectives": OBJECTIVES,
            "spectral_raw_weight": 0.1,
            "color_ridge": 1e-4,
            "max_iterations": 48,
            "fit_config": asdict(cfg),
        },
        "baseline": baseline,
        "rows": rows,
        "selection_rule": (
            "maximum sigma-1.5 reduction among converged protected-safe rows "
            "with max absolute local gradient <=2"
        ),
        "selected": (
            None
            if selected is None
            else {
                key: selected[key]
                for key in (
                    "scale",
                    "gradient_ridge",
                    "objective",
                    "sigma_1_5_reduction",
                    "laplacian_reduction",
                    "gradient_max_abs",
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
    parser.add_argument("--renderer", default="normalized")
    parser.add_argument("--coverage-tau", type=float, default=0.05)
    parser.add_argument("--boundary-band", type=float, default=4.0)
    parser.add_argument("--mask-margin", type=float, default=0.75)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
