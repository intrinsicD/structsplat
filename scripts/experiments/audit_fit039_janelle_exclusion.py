#!/usr/bin/env python3
"""Cold replay and scientist-pass audit for the selected FIT-039 Janelle field."""

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

from benchmarks.highpass_births import gaussian_blur  # noqa: E402
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
    _crop,
    _evaluate_all,
    _save_rgb,
    _save_scalar,
    _solve_metadata,
)
from structsplat.fit import _MaskConstraint  # noqa: E402
from structsplat.gaussians import GaussianField  # noqa: E402
from structsplat.safe_schedule import (  # noqa: E402
    SafeScheduleConfig,
    safe_commit_decision,
)


DEFAULT_RESULT = (
    REPOSITORY_ROOT / "runs/fit039_janelle_exclusion_screen_20260728/result.json"
)
DEFAULT_FIELD = (
    REPOSITORY_ROOT
    / "runs/fit039_janelle_exclusion_screen_20260728/fields/radius_0_selected.npz"
)
DEFAULT_OUT = (
    REPOSITORY_ROOT / "runs/fit039_janelle_exclusion_screen_20260728/audit"
)
METRIC_KEYS = (
    "foreground_mse",
    "boundary_mse",
    "cvar99_mse",
    "p99_mse",
    "interior_hole_fraction",
    "boundary_hole_fraction",
    "outside_max_abs",
    "outside_coverage_max",
    "detail_residual_mse",
    "detail_highpass_sigma_0_75_mse",
    "detail_highpass_sigma_1_5_mse",
    "detail_highpass_sigma_3_mse",
    "detail_laplacian_mse",
    "detail_sobel_mse",
)
SOURCE_FILES = (
    "scripts/experiments/audit_fit039_janelle_exclusion.py",
    "scripts/experiments/fit039_janelle_exclusion_screen.py",
    "benchmarks/highpass_births.py",
    "benchmarks/residual_birth_color_solve.py",
    "tasks/FIT-039-detail-pursuit-exclusion-radius.md",
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
    result = json.loads(args.result.read_text(encoding="utf-8"))
    if result["decision"]["selected_exclusion_radius"] != 0:
        raise RuntimeError("audit is bound to the selected exact-site exclusion arm")
    arm = next(
        item for item in result["arms"] if item["exclusion_radius"] == 0
    )
    selected_record = next(
        row for row in arm["rows"] if row["target_passed"]
    )
    base_job = Path(result["base_job"])
    prepared = _prepare_current_job(base_job)
    device = torch.device(args.device)
    target = torch.as_tensor(
        prepared["target"], device=device, dtype=torch.float32
    ).contiguous()
    mask = torch.as_tensor(
        prepared["mask"], device=device, dtype=torch.bool
    )
    cfg = _base_config(args)
    base = _scaled_field(prepared["field_path"], device, 1.0, 1.0)
    candidate = GaussianField.load(str(args.field), device=device)
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
    base_constraint_delta = _constraint_delta(base, cfg, constraint)
    candidate_constraint_delta = _constraint_delta(
        candidate, cfg, constraint
    )
    baseline, base_render, base_quality = _evaluate_all(
        base, target, mask, cfg, constraint, args.coverage_tau
    )
    cold, candidate_render, candidate_quality = _evaluate_all(
        candidate,
        target,
        mask,
        cfg,
        constraint,
        args.coverage_tau,
    )
    schedule = SafeScheduleConfig(
        capacity=candidate.n,
        coverage_tau=args.coverage_tau,
        boundary_band=args.boundary_band,
    )
    cold_safe, cold_reasons = safe_commit_decision(
        base_quality,
        candidate_quality,
        schedule.tolerances,
        schedule.hole_regression_budget,
    )
    stored = selected_record["metrics"]
    cold_deltas = {
        key: float(cold[key]) - float(stored[key]) for key in METRIC_KEYS
    }
    max_cold_delta = max(abs(value) for value in cold_deltas.values())

    second_started = time.perf_counter()
    second = solve_new_row_colors(
        candidate,
        target,
        cfg,
        torch.arange(
            base.n,
            candidate.n,
            device=device,
            dtype=torch.long,
        ),
        ridge=1e-4,
        max_iterations=64,
        tolerance=1e-7,
    )
    second_metrics, second_render, second_quality = _evaluate_all(
        second.field,
        target,
        mask,
        cfg,
        constraint,
        args.coverage_tau,
    )
    second_safe, second_reasons = safe_commit_decision(
        base_quality,
        second_quality,
        schedule.tolerances,
        schedule.hole_regression_budget,
    )
    fixed_point = {
        "seconds": time.perf_counter() - second_started,
        "solve": _solve_metadata(second),
        "max_abs_color_delta": float(
            (
                second.field.colors[base.n :]
                - candidate.colors[base.n :]
            )
            .abs()
            .max()
        ),
        "max_abs_render_delta": float(
            (second_render - candidate_render).abs().max()
        ),
        "metric_deltas": {
            key: float(second_metrics[key]) - float(cold[key])
            for key in METRIC_KEYS
        },
        "protected_safe": second_safe,
        "protected_reasons": second_reasons,
    }

    relative_reductions = {
        key: 1.0 - float(cold[key]) / float(baseline[key])
        for key in (
            "foreground_mse",
            "cvar99_mse",
            "p99_mse",
            "detail_residual_mse",
            "detail_highpass_sigma_0_75_mse",
            "detail_highpass_sigma_1_5_mse",
            "detail_highpass_sigma_3_mse",
            "detail_laplacian_mse",
            "detail_sobel_mse",
        )
    }
    all_sites = [
        int(site)
        for row in arm["rows"]
        if int(row["added_rows"]) <= int(selected_record["added_rows"])
        for site in row["batch_sites_flat"]
    ]
    unique_sites = len(set(all_sites))
    first_site = all_sites[0]
    width = target.shape[1]
    crop_center = (first_site // width, first_site % width)
    deep = (
        constraint.sdf_flat.reshape(mask.shape)
        > float(constraint.margin) + 6.0
    )
    base_error = (base_render - target).square().mean(dim=2).sqrt()
    candidate_error = (
        (candidate_render - target).square().mean(dim=2).sqrt()
    )
    base_highpass = (
        (base_render - target) - gaussian_blur(base_render - target, 1.5)
    ).square().mean(dim=2).sqrt()
    candidate_highpass = (
        (candidate_render - target)
        - gaussian_blur(candidate_render - target, 1.5)
    ).square().mean(dim=2).sqrt()
    error_scale = float(torch.quantile(base_error[mask], 0.99))
    highpass_scale = float(torch.quantile(base_highpass[deep], 0.99))
    _save_rgb(args.out / "images/target.png", target)
    _save_rgb(args.out / "images/baseline.png", base_render)
    _save_rgb(args.out / "images/candidate.png", candidate_render)
    _save_scalar(
        args.out / "images/baseline_error.png",
        base_error,
        error_scale,
    )
    _save_scalar(
        args.out / "images/candidate_error.png",
        candidate_error,
        error_scale,
    )
    _save_scalar(
        args.out / "images/baseline_highpass_error.png",
        base_highpass,
        highpass_scale,
    )
    _save_scalar(
        args.out / "images/candidate_highpass_error.png",
        candidate_highpass,
        highpass_scale,
    )
    _save_rgb(args.out / "crops/target.png", _crop(target, crop_center))
    _save_rgb(
        args.out / "crops/baseline.png",
        _crop(base_render, crop_center),
    )
    _save_rgb(
        args.out / "crops/candidate.png",
        _crop(candidate_render, crop_center),
    )

    fixed_point_hp_delta = fixed_point["metric_deltas"][
        "detail_highpass_sigma_1_5_mse"
    ]
    checks = {
        "field_count_exact": candidate.n == base.n + 768,
        "site_count_exact": len(all_sites) == 768,
        "sites_unique": unique_sites == 768,
        "cold_metrics_match": max_cold_delta <= 5e-8,
        "cold_protected_safe": cold_safe,
        "constraint_noop": max(
            candidate_constraint_delta.values()
        ) == 0.0,
        "outside_exact_zero": (
            cold["outside_max_abs"] == 0.0
            and cold["outside_coverage_max"] == 0.0
        ),
        "fixed_point_protected_safe": second_safe,
        "fixed_point_highpass_nonworse": fixed_point_hp_delta <= 1e-10,
        "primary_target": (
            relative_reductions[
                "detail_highpass_sigma_1_5_mse"
            ]
            >= 0.25
        ),
        "laplacian_target": (
            relative_reductions["detail_laplacian_mse"] >= 0.20
        ),
    }
    audit = {
        "schema": "structsplat.fit039.janelle-exclusion-audit.v1",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "inputs": {
            "result": str(args.result.resolve()),
            "result_sha256": _sha256(args.result),
            "field": str(args.field.resolve()),
            "field_sha256": _sha256(args.field),
            "base_field": str(prepared["field_path"]),
            "base_field_sha256": _sha256(prepared["field_path"]),
        },
        "protocol": {
            "fit_config": asdict(cfg),
            "metric_keys": METRIC_KEYS,
            "cold_metric_tolerance": 5e-8,
            "crop_center_yx": crop_center,
            "visualization_error_p99": error_scale,
            "visualization_highpass_p99": highpass_scale,
        },
        "baseline": baseline,
        "cold_candidate": cold,
        "stored_candidate": stored,
        "cold_metric_deltas": cold_deltas,
        "max_abs_cold_metric_delta": max_cold_delta,
        "relative_reductions": relative_reductions,
        "protected": {
            "safe": cold_safe,
            "reasons": cold_reasons,
        },
        "constraint": {
            "base_delta": base_constraint_delta,
            "candidate_delta": candidate_constraint_delta,
        },
        "sites": {
            "count": len(all_sites),
            "unique": unique_sites,
            "sha256": hashlib.sha256(
                torch.as_tensor(all_sites, dtype=torch.int64)
                .numpy()
                .astype("<i8", copy=False)
                .tobytes()
            ).hexdigest(),
        },
        "fixed_point": fixed_point,
        "cost": {
            "base_rows": base.n,
            "added_rows": candidate.n - base.n,
            "final_rows": candidate.n,
            "added_fraction_of_base": (candidate.n - base.n) / base.n,
            "fraction_of_fit031_accepted_additions": (
                (candidate.n - base.n) / 4608.0
            ),
            "row_reduction_factor_vs_fit031": (
                4608.0 / (candidate.n - base.n)
            ),
            "raw_added_scalars": (candidate.n - base.n) * 9,
            "fit031_raw_added_scalars_at_same_schema": 4608 * 9,
            "screen_stage_seconds_sum": sum(
                float(row["seconds"])
                for row in arm["rows"]
                if int(row["added_rows"])
                <= int(selected_record["added_rows"])
            ),
        },
        "checks": checks,
        "passed": all(checks.values()),
        "sources": [
            {
                "path": relative,
                "sha256": _sha256(REPOSITORY_ROOT / relative),
            }
            for relative in SOURCE_FILES
        ],
    }
    _atomic_json(args.out / "audit.json", audit)
    print(json.dumps({"passed": audit["passed"], "checks": checks}, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--field", type=Path, default=DEFAULT_FIELD)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--renderer", default="cuda_tiled")
    parser.add_argument("--coverage-tau", type=float, default=0.05)
    parser.add_argument("--boundary-band", type=float, default=4.0)
    parser.add_argument("--mask-margin", type=float, default=0.75)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
