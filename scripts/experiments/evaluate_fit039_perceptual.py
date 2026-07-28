#!/usr/bin/env python3
"""BENCH-001 metric readout for the selected FIT-039 Janelle field."""

from __future__ import annotations

import argparse
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

from scripts.experiments.fit032_janelle_dipole_screen import (  # noqa: E402
    _base_config,
    _prepare_current_job,
    _scaled_field,
)
from scripts.experiments.fit033_janelle_highpass_solve import (  # noqa: E402
    _constraint_delta,
    _evaluate_all,
)
from structsplat import metrics  # noqa: E402
from structsplat.fit import _MaskConstraint  # noqa: E402
from structsplat.gaussians import GaussianField  # noqa: E402


DEFAULT_RESULT = (
    REPOSITORY_ROOT / "runs/fit039_janelle_exclusion_screen_20260728/result.json"
)
DEFAULT_FIELD = (
    REPOSITORY_ROOT
    / "runs/fit039_janelle_exclusion_screen_20260728/fields/radius_0_selected.npz"
)
DEFAULT_OUT = (
    REPOSITORY_ROOT
    / "runs/fit039_janelle_exclusion_screen_20260728/audit/perceptual.json"
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


def _readout(rendered: torch.Tensor, target: torch.Tensor) -> dict:
    raw = rendered
    clamped = rendered.clamp(0.0, 1.0)
    return {
        "raw": {
            "psnr": metrics.psnr(raw, target),
            "ssim": float(metrics.ssim(raw, target)),
            "ms_ssim": metrics.ms_ssim(raw, target),
            "lpips": metrics.LPIPS.distance(raw, target),
        },
        "display_clamped": {
            "psnr": metrics.psnr(clamped, target),
            "ssim": float(metrics.ssim(clamped, target)),
            "ms_ssim": metrics.ms_ssim(clamped, target),
            "lpips": metrics.LPIPS.distance(clamped, target),
        },
    }


def run(args: argparse.Namespace) -> None:
    result = json.loads(args.result.read_text(encoding="utf-8"))
    prepared = _prepare_current_job(Path(result["base_job"]))
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
    _constraint_delta(base, cfg, constraint)
    _constraint_delta(candidate, cfg, constraint)
    _, base_render, _ = _evaluate_all(
        base, target, mask, cfg, constraint, args.coverage_tau
    )
    _, candidate_render, _ = _evaluate_all(
        candidate, target, mask, cfg, constraint, args.coverage_tau
    )
    base_metrics = _readout(base_render, target)
    candidate_metrics = _readout(candidate_render, target)
    payload = {
        "schema": "structsplat.fit039.bench001-metrics.v1",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "inputs": {
            "result": str(args.result.resolve()),
            "result_sha256": _sha256(args.result),
            "field": str(args.field.resolve()),
            "field_sha256": _sha256(args.field),
        },
        "protocol": {
            "metric_module": "src/structsplat/metrics.py",
            "mask_is_matted_black_in_both_target_and_render": True,
            "lpips_optional": True,
        },
        "baseline": base_metrics,
        "candidate": candidate_metrics,
        "delta_candidate_minus_baseline": {
            range_name: {
                metric_name: (
                    None
                    if candidate_metrics[range_name][metric_name] is None
                    or base_metrics[range_name][metric_name] is None
                    else candidate_metrics[range_name][metric_name]
                    - base_metrics[range_name][metric_name]
                )
                for metric_name in base_metrics[range_name]
            }
            for range_name in base_metrics
        },
    }
    _atomic_json(args.out, payload)
    print(json.dumps(payload, indent=2))


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
