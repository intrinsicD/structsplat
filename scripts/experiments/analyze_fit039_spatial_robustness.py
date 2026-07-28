#!/usr/bin/env python3
"""Spatial concentration audit for the selected FIT-039 Janelle field."""

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

from benchmarks.highpass_births import gaussian_blur  # noqa: E402
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
    / "runs/fit039_janelle_exclusion_screen_20260728/audit/spatial.json"
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


def _quantiles(values: torch.Tensor) -> dict[str, float]:
    return {
        f"p{int(100 * fraction):02d}": float(
            torch.quantile(values, fraction)
        )
        for fraction in (0.1, 0.25, 0.5, 0.75, 0.9)
    }


def run(args: argparse.Namespace) -> None:
    result = json.loads(args.result.read_text(encoding="utf-8"))
    arm = next(
        item for item in result["arms"] if item["exclusion_radius"] == 0
    )
    selected_rows = [
        row for row in arm["rows"] if int(row["added_rows"]) <= 768
    ]
    all_sites = [
        int(site)
        for row in selected_rows
        for site in row["batch_sites_flat"]
    ]
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
    deep = (
        constraint.sdf_flat.reshape(mask.shape)
        > float(constraint.margin) + 6.0
    )
    base_residual = base_render - target
    candidate_residual = candidate_render - target
    base_hp = (
        base_residual - gaussian_blur(base_residual, 1.5)
    ).square().mean(dim=2)
    candidate_hp = (
        candidate_residual - gaussian_blur(candidate_residual, 1.5)
    ).square().mean(dim=2)
    improvement = base_hp - candidate_hp
    deep_improvement = improvement[deep]
    positive = deep_improvement.clamp_min(0.0)
    negative = (-deep_improvement).clamp_min(0.0)

    tile_size = 32
    tile_rows = []
    for y0 in range(0, target.shape[0], tile_size):
        for x0 in range(0, target.shape[1], tile_size):
            tile_deep = deep[
                y0 : y0 + tile_size,
                x0 : x0 + tile_size,
            ]
            count = int(tile_deep.sum())
            if count < 64:
                continue
            before = base_hp[
                y0 : y0 + tile_size,
                x0 : x0 + tile_size,
            ][tile_deep].mean()
            after = candidate_hp[
                y0 : y0 + tile_size,
                x0 : x0 + tile_size,
            ][tile_deep].mean()
            tile_rows.append(
                {
                    "y0": y0,
                    "x0": x0,
                    "deep_pixels": count,
                    "before": float(before),
                    "after": float(after),
                    "relative_reduction": float(1.0 - after / before),
                    "absolute_reduction_sum": float(
                        (before - after) * count
                    ),
                }
            )
    tile_reductions = torch.tensor(
        [row["relative_reduction"] for row in tile_rows]
    )
    positive_tile_reduction = sum(
        max(0.0, row["absolute_reduction_sum"]) for row in tile_rows
    )
    top_tile_share = (
        0.0
        if positive_tile_reduction <= 0.0
        else max(
            max(0.0, row["absolute_reduction_sum"])
            for row in tile_rows
        )
        / positive_tile_reduction
    )

    width = target.shape[1]
    sites = torch.as_tensor(all_sites, dtype=torch.long)
    y = torch.div(sites, width, rounding_mode="floor")
    x = sites - y * width
    coordinates = torch.stack([x, y], dim=1)
    distance = (
        coordinates[:, None] - coordinates[None, :]
    ).abs().amax(dim=2)
    diagonal = torch.eye(len(all_sites), dtype=torch.bool)
    neighbor_1 = ((distance <= 1) & ~diagonal).any(dim=1)
    neighbor_2 = ((distance <= 2) & ~diagonal).any(dim=1)
    occupied_8x8 = len(
        {
            (int(y_value) // 8, int(x_value) // 8)
            for x_value, y_value in coordinates
        }
    )
    incremental = []
    previous_sigma = 0.0
    previous_laplacian = 0.0
    for row in selected_rows:
        incremental.append(
            {
                "added_rows": row["added_rows"],
                "sigma_1_5_reduction": row["sigma_1_5_reduction"],
                "sigma_1_5_increment": (
                    row["sigma_1_5_reduction"] - previous_sigma
                ),
                "laplacian_reduction": row["laplacian_reduction"],
                "laplacian_increment": (
                    row["laplacian_reduction"] - previous_laplacian
                ),
            }
        )
        previous_sigma = row["sigma_1_5_reduction"]
        previous_laplacian = row["laplacian_reduction"]

    payload = {
        "schema": "structsplat.fit039.spatial-robustness.v1",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "inputs": {
            "result": str(args.result.resolve()),
            "result_sha256": _sha256(args.result),
            "field": str(args.field.resolve()),
            "field_sha256": _sha256(args.field),
        },
        "pixel_attribution": {
            "deep_pixels": int(deep.sum()),
            "improved_fraction": float(
                (deep_improvement > 0.0).float().mean()
            ),
            "worsened_fraction": float(
                (deep_improvement < 0.0).float().mean()
            ),
            "unchanged_fraction": float(
                (deep_improvement == 0.0).float().mean()
            ),
            "positive_reduction_sum": float(positive.sum()),
            "negative_reduction_sum": float(negative.sum()),
            "net_reduction_sum": float(deep_improvement.sum()),
            "positive_to_negative_ratio": float(
                positive.sum() / negative.sum().clamp_min(1e-30)
            ),
        },
        "tiles_32": {
            "minimum_deep_pixels": 64,
            "count": len(tile_rows),
            "positive_fraction": float(
                (tile_reductions > 0.0).float().mean()
            ),
            "reduction_quantiles": _quantiles(tile_reductions),
            "top_positive_tile_share": top_tile_share,
            "rows": tile_rows,
        },
        "site_dispersion": {
            "count": len(all_sites),
            "unique": len(set(all_sites)),
            "fraction_with_neighbor_chebyshev_1": float(
                neighbor_1.float().mean()
            ),
            "fraction_with_neighbor_chebyshev_2": float(
                neighbor_2.float().mean()
            ),
            "occupied_8x8_cells": occupied_8x8,
        },
        "incremental_batches": incremental,
    }
    _atomic_json(args.out, payload)
    print(
        json.dumps(
            {
                "pixel_attribution": payload["pixel_attribution"],
                "tiles_32": {
                    key: value
                    for key, value in payload["tiles_32"].items()
                    if key != "rows"
                },
                "site_dispersion": payload["site_dispersion"],
            },
            indent=2,
        )
    )


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
