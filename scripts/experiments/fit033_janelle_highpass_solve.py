#!/usr/bin/env python3
"""Frozen equal-row FIT-033 screen on the current-pipeline Janelle C0001 state.

The result-bearing arms separate two decisions:

* allocation: FIT-031 per-pixel residual rank versus spatially separated high-pass peaks;
* coefficient initialization: target-pixel RGB versus an exact frozen-base partial color solve.

The strongest immediate FIT-032 dipole is included as an equal-row control. No optimizer
recovery is run: the partial least-squares solution is itself the terminal fixed-geometry action.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
from typing import Any

from PIL import Image
import torch
import torch.nn.functional as F


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
for _root in (
    REPOSITORY_ROOT,
    REPOSITORY_ROOT / "src",
    REPOSITORY_ROOT / "deprecated_scripts",
):
    _text = str(_root)
    if _text not in sys.path:
        sys.path.insert(0, _text)

from benchmarks.gauge_lifted_dipole import select_residual_dipoles  # noqa: E402
from benchmarks.highpass_births import (  # noqa: E402
    gaussian_blur,
    select_highpass_births,
)
from benchmarks.residual_birth_color_solve import (  # noqa: E402
    PartialColorSolveResult,
    solve_new_row_colors,
)
from scripts.experiments.fit032_janelle_dipole_screen import (  # noqa: E402
    _base_config,
    _dipole_proposal,
    _evaluate,
    _prepare_current_job,
    _scaled_field,
    _selection_prefix,
)
from structsplat.fit import _MaskConstraint  # noqa: E402
from structsplat.gaussians import GaussianField  # noqa: E402
from structsplat.safe_schedule import (  # noqa: E402
    SafeScheduleConfig,
    _birth_components,
    safe_commit_decision,
)


DEFAULT_BASE_JOB = (
    REPOSITORY_ROOT
    / "runs/fit032_current_base_20260728/runs/current/C0001/seed_0"
)
DEFAULT_OUT = REPOSITORY_ROOT / "runs/fit033_janelle_highpass_solve_20260728"
SOURCE_FILES = (
    "benchmarks/gauge_lifted_dipole.py",
    "benchmarks/highpass_births.py",
    "benchmarks/residual_birth_color_solve.py",
    "scripts/experiments/fit032_janelle_dipole_screen.py",
    "scripts/experiments/fit033_janelle_highpass_solve.py",
    "tasks/FIT-033-residual-birth-partial-color-solve.md",
)
ARMS = (
    "fit031_error_target",
    "fit031_error_solved",
    "highpass_target",
    "highpass_solved",
    "fit032_dipole",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(*arguments: str) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPOSITORY_ROOT), *arguments],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _snapshot_sources(out: Path) -> list[dict[str, Any]]:
    records = []
    for relative in SOURCE_FILES:
        source = REPOSITORY_ROOT / relative
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


def _save_rgb(path: Path, image: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    array = (
        image.detach()
        .cpu()
        .clamp(0.0, 1.0)
        .mul(255.0)
        .round()
        .to(torch.uint8)
        .numpy()
    )
    Image.fromarray(array, mode="RGB").save(path)


def _save_scalar(path: Path, value: torch.Tensor, scale: float) -> None:
    normalized = (value.detach().cpu() / max(float(scale), 1e-12)).clamp(0.0, 1.0)
    image = torch.stack(
        [
            normalized,
            torch.sqrt(normalized),
            0.15 * (1.0 - normalized),
        ],
        dim=2,
    )
    _save_rgb(path, image)


def _crop(image: torch.Tensor, center: tuple[int, int], half_size: int = 96) -> torch.Tensor:
    height, width = image.shape[:2]
    y, x = center
    y0 = max(0, min(height - 2 * half_size, y - half_size))
    x0 = max(0, min(width - 2 * half_size, x - half_size))
    return image[y0 : min(height, y0 + 2 * half_size), x0 : min(width, x0 + 2 * half_size)]


def _laplacian(image: torch.Tensor) -> torch.Tensor:
    kernel = image.new_tensor(
        [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]]
    )
    value = image.permute(2, 0, 1).unsqueeze(0)
    weights = kernel.view(1, 1, 3, 3).expand(3, 1, 3, 3)
    value = F.conv2d(F.pad(value, (1, 1, 1, 1), mode="reflect"), weights, groups=3)
    return value[0].permute(1, 2, 0)


def _sobel_energy(image: torch.Tensor) -> torch.Tensor:
    kernel_x = image.new_tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
    ) / 8.0
    kernel_y = kernel_x.T
    value = image.permute(2, 0, 1).unsqueeze(0)
    padded = F.pad(value, (1, 1, 1, 1), mode="reflect")
    weight_x = kernel_x.view(1, 1, 3, 3).expand(3, 1, 3, 3)
    weight_y = kernel_y.view(1, 1, 3, 3).expand(3, 1, 3, 3)
    grad_x = F.conv2d(padded, weight_x, groups=3)[0].permute(1, 2, 0)
    grad_y = F.conv2d(padded, weight_y, groups=3)[0].permute(1, 2, 0)
    return 0.5 * (grad_x.square() + grad_y.square())


@torch.no_grad()
def _evaluate_all(
    field: GaussianField,
    target: torch.Tensor,
    mask: torch.Tensor,
    cfg,
    constraint: _MaskConstraint,
    coverage_tau: float,
) -> tuple[dict[str, Any], torch.Tensor, Any]:
    metrics, rendered, quality = _evaluate(
        field,
        target,
        mask,
        cfg,
        constraint,
        coverage_tau,
    )
    residual = rendered - target
    deep = (
        constraint.sdf_flat.reshape(mask.shape)
        > float(constraint.margin) + 6.0
    )
    if not bool(deep.any()):
        deep = mask
    metrics.update(
        {
            "detail_deep_pixels": int(deep.sum()),
            "detail_residual_mse": float(residual[deep].square().mean()),
            "detail_highpass_sigma_0_75_mse": float(
                (residual - gaussian_blur(residual, 0.75))[deep].square().mean()
            ),
            "detail_highpass_sigma_1_5_mse": float(
                (residual - gaussian_blur(residual, 1.5))[deep].square().mean()
            ),
            "detail_highpass_sigma_3_mse": float(
                (residual - gaussian_blur(residual, 3.0))[deep].square().mean()
            ),
            "detail_laplacian_mse": float(_laplacian(residual)[deep].square().mean()),
            "detail_sobel_mse": float(_sobel_energy(residual)[deep].mean()),
        }
    )
    return metrics, rendered, quality


def _solve_metadata(result: PartialColorSolveResult) -> dict[str, Any]:
    return {
        "iterations": result.iterations,
        "converged": result.converged,
        "initial_residual_norm": result.initial_residual_norm,
        "final_residual_norm": result.final_residual_norm,
        "relative_residual": result.relative_residual,
        "raw_color_min": result.raw_color_min,
        "raw_color_max": result.raw_color_max,
    }


def _prefix_components(components: GaussianField, count: int) -> GaussianField:
    rows = torch.arange(count, device=components.means.device, dtype=torch.long)
    return components.subset(rows)


def _constraint_delta(
    field: GaussianField,
    cfg,
    constraint: _MaskConstraint,
) -> dict[str, float]:
    means = field.means.detach().clone()
    log_scales = field.log_scales.detach().clone()
    constraint.apply(field, cfg, refresh=True)
    return {
        "mean_max_abs": float((field.means - means).abs().max()),
        "log_scale_max_abs": float((field.log_scales - log_scales).abs().max()),
    }


def _decision(
    baseline: dict[str, Any],
    rows: list[dict[str, Any]],
    budgets: tuple[int, ...],
) -> dict[str, Any]:
    by_key = {(row["arm"], int(row["budget"])): row for row in rows}
    records = []
    factor_passes = 0
    detail_passes = 0
    all_safe = True
    for budget in budgets:
        candidate = by_key[("highpass_solved", budget)]
        target_control = by_key[("highpass_target", budget)]
        candidate_reduction = float(baseline["foreground_mse"]) - float(
            candidate["metrics"]["foreground_mse"]
        )
        control_reduction = float(baseline["foreground_mse"]) - float(
            target_control["metrics"]["foreground_mse"]
        )
        if control_reduction > 0.0:
            factor = candidate_reduction / control_reduction
        elif candidate_reduction > 0.0:
            factor = float("inf")
        else:
            factor = float("-inf")
        detail_reduction = 1.0 - (
            float(candidate["metrics"]["detail_highpass_sigma_1_5_mse"])
            / float(baseline["detail_highpass_sigma_1_5_mse"])
        )
        factor_pass = candidate_reduction > 0.0 and factor >= 2.0
        detail_pass = detail_reduction >= 0.05
        safe = bool(candidate["protected_safe"])
        factor_passes += int(factor_pass)
        detail_passes += int(detail_pass)
        all_safe = all_safe and safe
        strongest_control_reduction = max(
            float(baseline["foreground_mse"])
            - float(by_key[(arm, budget)]["metrics"]["foreground_mse"])
            for arm in ARMS
            if arm != "highpass_solved"
        )
        records.append(
            {
                "budget": budget,
                "candidate_foreground_mse_reduction": candidate_reduction,
                "same_geometry_target_color_reduction": control_reduction,
                "candidate_to_target_color_factor": factor,
                "factor_pass": factor_pass,
                "sigma_1_5_highpass_relative_reduction": detail_reduction,
                "detail_pass": detail_pass,
                "protected_safe": safe,
                "strongest_other_foreground_mse_reduction": strongest_control_reduction,
            }
        )
    advance = factor_passes >= 2 and detail_passes >= 1 and all_safe
    return {
        "rule": (
            "advance to independent-image confirmation iff highpass_solved has >=2x "
            "same-geometry target-color foreground-MSE reduction at >=2 budgets, >=5% "
            "deep sigma-1.5 high-pass reduction at >=1 budget <=128, and is protected-safe "
            "at every budget"
        ),
        "budget_records": records,
        "factor_budgets_passed": factor_passes,
        "detail_budgets_passed": detail_passes,
        "all_candidate_budgets_protected_safe": all_safe,
        "advance_to_independent_confirmation": advance,
        "production_promotion_authorized": False,
    }


def run(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise RuntimeError("requested CUDA but torch.cuda.is_available() is false")
    budgets = tuple(sorted(set(int(value) for value in args.budgets)))
    if budgets != (32, 64, 128):
        raise ValueError("the frozen FIT-033 protocol requires budgets 32 64 128")
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        raise RuntimeError(f"output directory is not empty: {out}")
    out.mkdir(parents=True, exist_ok=True)
    source_records = _snapshot_sources(out)

    torch.manual_seed(0)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)

    prepared = _prepare_current_job(args.base_job)
    width, height = prepared["fit_size"]
    target = torch.as_tensor(
        prepared["target"], device=device, dtype=torch.float32
    ).contiguous()
    mask_cpu = torch.as_tensor(prepared["mask"], dtype=torch.bool)
    mask = mask_cpu.to(device=device)
    cfg = _base_config(args)
    base = _scaled_field(prepared["field_path"], device, 1.0, 1.0)
    if base.opacities is None or base.color_grads is not None:
        raise RuntimeError("FIT-033 requires explicit opacity and constant colors")
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
    baseline, base_render, base_quality = _evaluate_all(
        base, target, mask, cfg, constraint, args.coverage_tau
    )
    _save_rgb(out / "images/target.png", target)
    _save_rgb(out / "images/baseline.png", base_render)

    schedule = SafeScheduleConfig(
        capacity=base.n + max(budgets),
        coverage_tau=float(args.coverage_tau),
        boundary_band=float(args.boundary_band),
        error_tail_max_scale=1.25,
    )
    standard_components, standard_metadata = _birth_components(
        base,
        target,
        base_render,
        cfg,
        constraint,
        schedule,
        max(budgets),
        "error_tail",
    )
    if standard_components is None or standard_components.n != max(budgets):
        raise RuntimeError("FIT-031 control did not produce the frozen maximum budget")

    highpass_selection = select_highpass_births(
        base,
        target,
        base_render,
        constraint,
        max(budgets),
        blur_sigma=1.5,
        nms_radius=2,
        deep_offset=6.0,
        scale=0.35,
        opacity=0.8,
    )
    dipole_selection = select_residual_dipoles(
        base,
        target,
        base_render,
        cfg,
        mask,
        max(budgets),
        trust_radius=0.35,
        max_color_contrast=0.5,
        minimum_spacing=3.0,
        spacing_scale=0.75,
    )

    deep = (
        constraint.sdf_flat.reshape(mask.shape)
        > float(constraint.margin) + 6.0
    )
    base_error = (base_render - target).square().mean(dim=2).sqrt()
    base_highpass = (
        (base_render - target) - gaussian_blur(base_render - target, 1.5)
    ).square().mean(dim=2).sqrt()
    error_scale = float(torch.quantile(base_error[mask], 0.99))
    highpass_scale = float(torch.quantile(base_highpass[deep], 0.99))
    _save_scalar(out / "images/baseline_error.png", base_error, error_scale)
    _save_scalar(
        out / "images/baseline_highpass_error.png",
        base_highpass,
        highpass_scale,
    )
    top_site = int(highpass_selection.sites[0])
    crop_center = (top_site // width, top_site % width)
    _save_rgb(out / "crops/target.png", _crop(target, crop_center))
    _save_rgb(out / "crops/baseline.png", _crop(base_render, crop_center))

    rows: list[dict[str, Any]] = []
    for budget in budgets:
        for arm in ARMS:
            started = time.perf_counter()
            solve_metadata = None
            proposal_metadata: dict[str, Any]
            if arm.startswith("fit031_error"):
                proposal = base.append(
                    _prefix_components(standard_components, budget)
                )
                constraint_delta = _constraint_delta(proposal, cfg, constraint)
                proposal_metadata = {
                    **standard_metadata,
                    "prefix_rows": budget,
                }
                if arm.endswith("solved"):
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
                    proposal = solve.field
                    solve_metadata = _solve_metadata(solve)
            elif arm.startswith("highpass"):
                proposal = base.append(
                    _prefix_components(highpass_selection.components, budget)
                )
                constraint_delta = _constraint_delta(proposal, cfg, constraint)
                proposal_metadata = {
                    **highpass_selection.metadata,
                    "prefix_rows": budget,
                    "prefix_score_min": float(
                        highpass_selection.scores[:budget].min()
                    ),
                    "prefix_score_mean": float(
                        highpass_selection.scores[:budget].mean()
                    ),
                }
                if arm.endswith("solved"):
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
                    proposal = solve.field
                    solve_metadata = _solve_metadata(solve)
            else:
                (
                    proposal,
                    _,
                    proposal_metadata,
                    _,
                    _,
                ) = _dipole_proposal(
                    base,
                    _selection_prefix(dipole_selection, budget),
                    target,
                    mask,
                    cfg,
                    constraint,
                    args.coverage_tau,
                )
                constraint_delta = {
                    "mean_max_abs": max(
                        float(row["constraint_mean_max_abs"])
                        for row in proposal_metadata["line_search"]
                    ),
                    "log_scale_max_abs": max(
                        float(row["constraint_log_scale_max_abs"])
                        for row in proposal_metadata["line_search"]
                    ),
                }
            proposal_seconds = time.perf_counter() - started
            if proposal.n != base.n + budget:
                raise RuntimeError(
                    f"{arm}/{budget}: expected {base.n + budget} rows, got {proposal.n}"
                )
            metrics, rendered, quality = _evaluate_all(
                proposal,
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
            fixed_point = None
            if arm == "highpass_solved" and budget == max(budgets):
                second = solve_new_row_colors(
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
                second_metrics, second_render, _ = _evaluate_all(
                    second.field,
                    target,
                    mask,
                    cfg,
                    constraint,
                    args.coverage_tau,
                )
                fixed_point = {
                    "second_solve": _solve_metadata(second),
                    "max_abs_color_delta": float(
                        (
                            second.field.colors[base.n :]
                            - proposal.colors[base.n :]
                        )
                        .abs()
                        .max()
                    ),
                    "max_abs_render_delta": float(
                        (second_render - rendered).abs().max()
                    ),
                    "foreground_mse_delta": (
                        float(second_metrics["foreground_mse"])
                        - float(metrics["foreground_mse"])
                    ),
                    "highpass_mse_delta": (
                        float(second_metrics["detail_highpass_sigma_1_5_mse"])
                        - float(metrics["detail_highpass_sigma_1_5_mse"])
                    ),
                }
                (out / "fields").mkdir(parents=True, exist_ok=True)
                proposal.save(str(out / "fields/highpass_solved_128.npz"))

            _save_rgb(out / f"images/{arm}_{budget:03d}.png", rendered)
            error = (rendered - target).square().mean(dim=2).sqrt()
            highpass_error = (
                (rendered - target) - gaussian_blur(rendered - target, 1.5)
            ).square().mean(dim=2).sqrt()
            _save_scalar(
                out / f"images/{arm}_{budget:03d}_error.png",
                error,
                error_scale,
            )
            _save_scalar(
                out / f"images/{arm}_{budget:03d}_highpass_error.png",
                highpass_error,
                highpass_scale,
            )
            if budget == max(budgets):
                _save_rgb(
                    out / f"crops/{arm}.png",
                    _crop(rendered, crop_center),
                )
            row = {
                "arm": arm,
                "budget": budget,
                "n_before": base.n,
                "n_after": proposal.n,
                "proposal_seconds": proposal_seconds,
                "metrics": metrics,
                "protected_safe": safe,
                "protected_reasons": reasons,
                "constraint_delta": constraint_delta,
                "proposal_metadata": proposal_metadata,
                "solve_metadata": solve_metadata,
                "fixed_point": fixed_point,
            }
            rows.append(row)
            _atomic_json(out / "partial_results.json", rows)
            print(
                f"{arm:23s} K={budget:3d} "
                f"PSNR={metrics['foreground_psnr_db']:.6f} "
                f"HP={metrics['detail_highpass_sigma_1_5_mse']:.9g} "
                f"safe={safe}"
            )

    decision = _decision(baseline, rows, budgets)
    repository_status = _git("status", "--short")
    result = {
        "schema": "structsplat.fit033.janelle-highpass-partial-solve.v1",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": {
            "source_kind": prepared["source_kind"],
            "base_job": str(prepared["base_job"]),
            "image_path": str(prepared["image_path"]),
            "mask_path": str(prepared["mask_path"]),
            "image_sha256": _sha256(prepared["image_path"]),
            "mask_sha256": _sha256(prepared["mask_path"]),
            "fit_size": prepared["fit_size"],
            "foreground_pixels": int(mask.sum()),
        },
        "base_field": {
            "path": str(prepared["field_path"]),
            "sha256": _sha256(prepared["field_path"]),
            "n_gaussians": base.n,
            "constraint_delta": base_constraint_delta,
        },
        "protocol": {
            "budgets": budgets,
            "arms": ARMS,
            "optimizer_recovery_steps": 0,
            "selection_blur_sigma": 1.5,
            "selection_nms_radius": 2,
            "selection_deep_offset": 6.0,
            "birth_scale": 0.35,
            "birth_opacity": 0.8,
            "solve_ridge": 1e-4,
            "solve_max_iterations": 32,
            "solve_tolerance": 1e-7,
            "fit_config": asdict(cfg),
            "detail_metrics": {
                "primary": (
                    "MSE of sigma-1.5 high-pass RGB rendering residual on "
                    "SDF > margin+6 pixels"
                ),
                "orthogonal": [
                    "deep residual MSE",
                    "sigma-0.75 high-pass residual MSE",
                    "sigma-3 high-pass residual MSE",
                    "Laplacian residual MSE",
                    "Sobel residual energy",
                ],
            },
        },
        "baseline": baseline,
        "selection": {
            "fit031_error": standard_metadata,
            "highpass": highpass_selection.metadata,
            "dipole": {
                "candidate_count": dipole_selection.candidate_count,
                "rejected_background": dipole_selection.rejected_background,
                "rejected_mask": dipole_selection.rejected_mask,
                "rejected_degenerate": dipole_selection.rejected_degenerate,
                "selected": int(dipole_selection.parents.numel()),
            },
            "crop_center_yx": crop_center,
        },
        "visualization_scales": {
            "error_p99": error_scale,
            "highpass_error_p99": highpass_scale,
        },
        "rows": rows,
        "decision": decision,
        "repository": {
            "commit": _git("rev-parse", "HEAD"),
            "branch": _git("branch", "--show-current"),
            "status": repository_status,
            "status_sha256": hashlib.sha256(
                repository_status.encode()
            ).hexdigest(),
            "source_snapshot": source_records,
        },
        "environment": {
            "platform": platform.platform(),
            "python": sys.version,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "device": str(device),
            "gpu": (
                torch.cuda.get_device_name(device)
                if device.type == "cuda"
                else None
            ),
            "peak_cuda_bytes": (
                int(torch.cuda.max_memory_allocated(device))
                if device.type == "cuda"
                else None
            ),
        },
    }
    _atomic_json(out / "result.json", result)
    print(json.dumps(decision, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-job", type=Path, default=DEFAULT_BASE_JOB)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--renderer", default="cuda_tiled")
    parser.add_argument("--budgets", type=int, nargs="+", default=[32, 64, 128])
    parser.add_argument("--coverage-tau", type=float, default=0.05)
    parser.add_argument("--boundary-band", type=float, default=4.0)
    parser.add_argument("--mask-margin", type=float, default=0.75)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
