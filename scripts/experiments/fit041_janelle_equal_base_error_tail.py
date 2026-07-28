#!/usr/bin/env python3
"""Run FIT-031's error-only tail on FIT-040's exact persisted Janelle base."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
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

from scripts.experiments.fit032_janelle_dipole_screen import (  # noqa: E402
    _base_config,
    _prepare_current_job,
    _scaled_field,
)
from scripts.experiments.fit033_janelle_highpass_solve import (  # noqa: E402
    _constraint_delta,
    _evaluate_all,
)
from scripts.experiments.fit040_janelle_production_pursuit import (  # noqa: E402
    _disabled_phase,
    _sha256,
)
from structsplat.fit import _MaskConstraint  # noqa: E402
from structsplat.safe_schedule import (  # noqa: E402
    SafeScheduleConfig,
    run_safe_schedule,
)


DEFAULT_BASE_JOB = (
    REPOSITORY_ROOT
    / "runs/fit032_current_base_20260728/runs/current/C0001/seed_0"
)
DEFAULT_PURSUIT_RESULT = (
    REPOSITORY_ROOT / "runs/fit040_janelle_production_pursuit_20260728/result.json"
)
DEFAULT_OUT = REPOSITORY_ROOT / "runs/fit041_janelle_equal_base_error_tail_20260728"
DETAIL_KEYS = (
    "detail_highpass_sigma_0_75_mse",
    "detail_highpass_sigma_1_5_mse",
    "detail_highpass_sigma_3_mse",
    "detail_laplacian_mse",
    "detail_residual_mse",
    "detail_sobel_mse",
)


def _atomic_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _reduction(before: float, after: float) -> float:
    return 0.0 if before <= 0.0 else 1.0 - after / before


def _placement_summary(field, start_n: int, constraint, height: int, width: int):
    means = field.means[start_n:]
    x = means[:, 0].round().long().clamp(0, width - 1)
    y = means[:, 1].round().long().clamp(0, height - 1)
    sdf = constraint.sdf_flat.reshape(height, width)[y, x]
    deep_threshold = float(constraint.margin) + 6.0
    quantiles = torch.quantile(
        sdf.to(dtype=torch.float32),
        sdf.new_tensor([0.0, 0.1, 0.5, 0.9, 1.0]),
    )
    deep_rows = int((sdf > deep_threshold).sum())
    return {
        "rows": int(sdf.numel()),
        "deep_threshold": deep_threshold,
        "deep_rows": deep_rows,
        "deep_fraction": deep_rows / max(int(sdf.numel()), 1),
        "sdf_quantiles_p0_p10_p50_p90_p100": [
            float(value) for value in quantiles
        ],
    }


def run(args: argparse.Namespace) -> None:
    if args.out.exists() and any(args.out.iterdir()):
        raise RuntimeError(f"output directory is not empty: {args.out}")
    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA but torch.cuda.is_available() is false")
    torch.manual_seed(0)

    prepared = _prepare_current_job(args.base_job)
    target = torch.as_tensor(
        prepared["target"],
        device=device,
        dtype=torch.float32,
    ).contiguous()
    mask = torch.as_tensor(
        prepared["mask"],
        device=device,
        dtype=torch.bool,
    )
    base = _scaled_field(prepared["field_path"], device, 1.0, 1.0)
    cfg = replace(
        _base_config(args),
        color_solve_maxiter=1,
        color_solve_lambda=1e30,
    )
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
    constraint_delta = _constraint_delta(base, cfg, constraint)
    baseline, _, _ = _evaluate_all(
        base,
        target,
        mask,
        cfg,
        constraint,
        args.coverage_tau,
    )

    defaults = SafeScheduleConfig()
    schedule = SafeScheduleConfig(
        capacity=base.n,
        storage_policy="dynamic",
        boundary_enabled=True,
        coverage_target_gaussians=base.n,
        detail_target_gaussians=base.n,
        coverage_tau=float(args.coverage_tau),
        boundary_band=float(args.boundary_band),
        error_tail_fraction=float(args.fraction),
        bootstrap=_disabled_phase(defaults.bootstrap, base.n),
        coverage=_disabled_phase(defaults.coverage, base.n),
        detail=_disabled_phase(defaults.detail, base.n),
        boundary=_disabled_phase(defaults.boundary, base.n),
        redistribution=_disabled_phase(defaults.redistribution, base.n),
        polish=_disabled_phase(defaults.polish, base.n),
    )
    started = time.perf_counter()
    result = run_safe_schedule(
        base,
        target,
        mask,
        cfg,
        schedule,
        verbose=not args.quiet,
    )
    elapsed = time.perf_counter() - started
    field_path = args.out / "field.npz"
    result["field"].save(str(field_path))
    final, _, _ = _evaluate_all(
        result["field"],
        target,
        mask,
        cfg,
        constraint,
        args.coverage_tau,
    )
    reductions = {
        key: _reduction(float(baseline[key]), float(final[key]))
        for key in DETAIL_KEYS
    }

    pursuit = json.loads(args.pursuit_result.read_text(encoding="utf-8"))
    height, width = mask.shape
    error_placement = _placement_summary(
        result["field"],
        base.n,
        constraint,
        height,
        width,
    )
    pursuit_field = type(base).load(
        pursuit["field"]["path"],
        device=device,
    )
    pursuit_placement = _placement_summary(
        pursuit_field,
        base.n,
        constraint,
        height,
        width,
    )
    tail = result["error_tail"]
    same_base_checks = {
        "base_field_sha256": (
            _sha256(prepared["field_path"])
            == pursuit["source"]["base_field_sha256"]
        ),
        "target_pixel_sha256": (
            _sha256(args.base_job / "target.png")
            == pursuit["source"]["target_pixel_sha256"]
        ),
        "mask_sha256": (
            _sha256(prepared["mask_path"]) == pursuit["source"]["mask_sha256"]
        ),
        "base_rows": int(base.n) == int(pursuit["pursuit_tail"]["start_n"]),
        "constraint_noop": (
            constraint_delta["mean_max_abs"] == 0.0
            and constraint_delta["log_scale_max_abs"] == 0.0
        ),
    }
    comparison = {
        "same_base_checks": same_base_checks,
        "same_base": all(same_base_checks.values()),
        "error_tail_added_rows": int(tail["activated_rows"]),
        "pursuit_added_rows": int(pursuit["pursuit_tail"]["activated_rows"]),
        "error_tail_to_pursuit_row_ratio": (
            float(tail["activated_rows"])
            / float(pursuit["pursuit_tail"]["activated_rows"])
        ),
        "error_tail_highpass_reduction": reductions[
            "detail_highpass_sigma_1_5_mse"
        ],
        "pursuit_highpass_reduction": float(
            pursuit["pursuit_tail"]["highpass_reduction"]
        ),
        "error_tail_laplacian_reduction": reductions[
            "detail_laplacian_mse"
        ],
        "pursuit_laplacian_reduction": float(
            pursuit["pursuit_tail"]["laplacian_reduction"]
        ),
        "error_tail_foreground_psnr_gain_db": float(
            tail["foreground_psnr_gain_db"]
        ),
        "pursuit_foreground_psnr_gain_db": float(
            pursuit["pursuit_tail"]["foreground_psnr_gain_db"]
        ),
        "error_tail_seconds": elapsed,
        "pursuit_tail_seconds": float(pursuit["pursuit_tail"]["seconds"]),
        "error_tail_deep_rows": error_placement["deep_rows"],
        "pursuit_deep_rows": pursuit_placement["deep_rows"],
    }
    comparison["pursuit_uses_fewer_rows"] = (
        comparison["pursuit_added_rows"] < comparison["error_tail_added_rows"]
    )
    comparison["pursuit_has_larger_highpass_reduction"] = (
        comparison["pursuit_highpass_reduction"]
        > comparison["error_tail_highpass_reduction"]
    )
    comparison["pursuit_has_larger_laplacian_reduction"] = (
        comparison["pursuit_laplacian_reduction"]
        > comparison["error_tail_laplacian_reduction"]
    )

    payload = {
        "schema": "structsplat.fit041.equal-base-error-tail.v1",
        "task": "FIT-041",
        "source": {
            "base_job": str(args.base_job.resolve()),
            "base_field": str(prepared["field_path"]),
            "base_field_sha256": _sha256(prepared["field_path"]),
            "target_pixel_sha256": _sha256(args.base_job / "target.png"),
            "mask": str(prepared["mask_path"]),
            "mask_sha256": _sha256(prepared["mask_path"]),
            "pursuit_result": str(args.pursuit_result.resolve()),
            "pursuit_result_sha256": _sha256(args.pursuit_result),
        },
        "environment": {
            "device": str(device),
            "gpu": (
                torch.cuda.get_device_name(device)
                if device.type == "cuda"
                else None
            ),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
        },
        "fit_config": asdict(cfg),
        "replay_adapter": (
            "entry global color solve made an exact no-op because base_job is post-schedule"
        ),
        "schedule": asdict(schedule),
        "constraint_delta": constraint_delta,
        "baseline": baseline,
        "final": final,
        "relative_reductions": reductions,
        "placement": {
            "error_tail": error_placement,
            "pursuit_tail": pursuit_placement,
        },
        "error_tail": tail,
        "storage": result["storage"],
        "converged": result["converged"],
        "seconds": elapsed,
        "field": {
            "path": str(field_path),
            "sha256": _sha256(field_path),
        },
        "comparison": comparison,
    }
    _atomic_json(args.out / "history.json", result["history"])
    _atomic_json(args.out / "result.json", payload)
    print(json.dumps(comparison, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-job", type=Path, default=DEFAULT_BASE_JOB)
    parser.add_argument(
        "--pursuit-result",
        type=Path,
        default=DEFAULT_PURSUIT_RESULT,
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--fraction", type=float, default=0.5)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--renderer", default="cuda_tiled")
    parser.add_argument("--mask-margin", type=float, default=0.75)
    parser.add_argument("--boundary-band", type=float, default=4.0)
    parser.add_argument("--coverage-tau", type=float, default=0.05)
    parser.add_argument("--quiet", action="store_true")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
