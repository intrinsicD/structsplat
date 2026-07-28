#!/usr/bin/env python3
"""Prior-site exclusion killing ablation for FIT-039 on Janelle C0001."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F


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
DEFAULT_OUT = REPOSITORY_ROOT / "runs/fit039_janelle_exclusion_screen_20260728"
RADIUS2_RESULT = (
    REPOSITORY_ROOT / "runs/fit038_janelle_detail_pursuit_20260728/result.json"
)
BATCH_ROWS = 128
MAX_ROWS = 2048
RADII = (0, 1)
SOURCE_FILES = (
    "benchmarks/highpass_births.py",
    "benchmarks/residual_birth_color_solve.py",
    "scripts/experiments/fit039_janelle_exclusion_screen.py",
    "tasks/FIT-039-detail-pursuit-exclusion-radius.md",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _site_sha256(sites: list[int]) -> str:
    values = np.asarray(sites, dtype="<i8")
    return hashlib.sha256(values.tobytes()).hexdigest()


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
    height, width = mask.shape
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
        capacity=base.n + MAX_ROWS,
        coverage_tau=args.coverage_tau,
        boundary_band=args.boundary_band,
    )

    arms = []
    for radius in RADII:
        current = base
        current_render = base_render
        site_mask = torch.zeros(
            height, width, device=device, dtype=torch.bool
        )
        all_sites: list[int] = []
        stage_rows = []
        reached = None
        for stage in range(1, MAX_ROWS // BATCH_ROWS + 1):
            started = time.perf_counter()
            kernel = 2 * radius + 1
            forbidden = F.max_pool2d(
                site_mask[None, None].to(dtype=target.dtype),
                kernel,
                stride=1,
                padding=radius,
            )[0, 0] > 0.0
            selection = select_highpass_births(
                current,
                target,
                current_render,
                constraint,
                BATCH_ROWS,
                blur_sigma=1.5,
                nms_radius=2,
                deep_offset=6.0,
                scale=0.35,
                opacity=0.8,
                forbidden_mask=forbidden,
            )
            proposal = current.append(selection.components)
            constraint_delta = _constraint_delta(
                proposal, cfg, constraint
            )
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
                max_iterations=64,
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
            batch_sites = [int(value) for value in selection.sites.cpu()]
            candidate_sites = all_sites + batch_sites
            row = {
                "stage": stage,
                "added_rows": stage * BATCH_ROWS,
                "n_after": solve.field.n,
                "seconds": time.perf_counter() - started,
                "protected_safe": safe,
                "protected_reasons": reasons,
                "constraint_delta": constraint_delta,
                "render_min": float(rendered.min()),
                "render_max": float(rendered.max()),
                "solve": _solve_metadata(solve),
                "selection": selection.metadata,
                "batch_sites_flat": batch_sites,
                "all_sites_sha256": _site_sha256(candidate_sites),
                "unique_sites": len(set(candidate_sites)),
                "metrics": metrics,
                "sigma_1_5_reduction": sigma_reduction,
                "laplacian_reduction": laplacian_reduction,
                "target_passed": passed,
            }
            stage_rows.append(row)
            _atomic_json(
                args.out / f"radius_{radius}_partial.json",
                stage_rows,
            )
            print(
                f"radius={radius} stage={stage:2d} "
                f"K={stage * BATCH_ROWS:4d} "
                f"HP={100 * sigma_reduction:.3f}% "
                f"Lap={100 * laplacian_reduction:.3f}% "
                f"safe={safe} target={passed}"
            )
            if not safe:
                break
            current = solve.field
            current_render = rendered
            all_sites = candidate_sites
            y = torch.div(
                selection.sites, width, rounding_mode="floor"
            )
            x = selection.sites - y * width
            site_mask[y, x] = True
            if passed:
                reached = row
                (args.out / "fields").mkdir(parents=True, exist_ok=True)
                current.save(
                    str(args.out / f"fields/radius_{radius}_selected.npz")
                )
                break
        arms.append(
            {
                "exclusion_radius": radius,
                "rows": stage_rows,
                "target_reached": reached is not None,
                "minimum_rows": (
                    None if reached is None else reached["added_rows"]
                ),
            }
        )

    radius2 = json.loads(args.radius2_result.read_text(encoding="utf-8"))
    passing = [arm for arm in arms if arm["target_reached"]]
    selected = (
        None
        if not passing
        else min(
            passing,
            key=lambda arm: (
                int(arm["minimum_rows"]),
                -int(arm["exclusion_radius"]),
            ),
        )
    )
    result = {
        "schema": "structsplat.fit039.janelle-exclusion-screen.v1",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "development_only": True,
        "base_job": str(args.base_job.resolve()),
        "base_field_sha256": _sha256(prepared["field_path"]),
        "radius2_control": {
            "path": str(args.radius2_result.resolve()),
            "sha256": _sha256(args.radius2_result),
            "target_reached": radius2["decision"]["target_reached"],
            "minimum_rows": radius2["decision"]["minimum_rows"],
            "terminal": radius2["rows"][-1],
        },
        "protocol": {
            "exclusion_radii": RADII,
            "within_batch_nms_radius": 2,
            "batch_rows": BATCH_ROWS,
            "max_rows": MAX_ROWS,
            "sigma_1_5_target": 0.25,
            "laplacian_target": 0.20,
            "fit_config": asdict(cfg),
        },
        "baseline": baseline,
        "arms": arms,
        "decision": {
            "target_reached": selected is not None,
            "selected_exclusion_radius": (
                None
                if selected is None
                else selected["exclusion_radius"]
            ),
            "minimum_rows": (
                None if selected is None else selected["minimum_rows"]
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
    parser.add_argument(
        "--radius2-result", type=Path, default=RADIUS2_RESULT
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--renderer", default="cuda_tiled")
    parser.add_argument("--coverage-tau", type=float, default=0.05)
    parser.add_argument("--boundary-band", type=float, default=4.0)
    parser.add_argument("--mask-margin", type=float, default=0.75)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
