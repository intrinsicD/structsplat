#!/usr/bin/env python3
"""Run HIER-013's frozen exact-7k global-projection development screen.

Reproduce from the repository root::

    PYTHONPATH=src python scripts/experiments/hier013_global_projection_development.py \
      --images tests/test_images \
      --out results/hier013_global_projection_test_images_development_2026-08-10 \
      --seeds 0 1 2 --target-gaussians 7000 --max-side 512 \
      --projection-ridge 1e-8 --projection-tolerance 1e-6 \
      --projection-max-iterations 48 --projection-coefficient-limit 16 \
      --max-exchanges 128 --site-count 96 --site-nms-radius 1 \
      --donor-count 64 --proposal-frontier 24 --coefficient-limit 16 \
      --device cuda --renderer cuda_additive --render-chunk 256 --lpips

This is an explicitly dirty-source-compatible development diagnostic. It cannot authorize a
default, Field V2 semantics, FIT-046 completion, compression rate, or production confirmation.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import csv
from html import escape
import hashlib
import json
import math
import platform
from pathlib import Path
import shlex
import shutil
import sys
import time
import traceback

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
for candidate in (ROOT, ROOT / "src"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from scripts.experiments import hier005_pixel_contraction as report_utils  # noqa: E402
from scripts.experiments import hier010_residual_anchor_projection as viz_utils  # noqa: E402
from scripts.experiments.hier011_guarded_residual_column_exchange import (  # noqa: E402
    CANDIDATE_SHAPES,
)
from structsplat.cli import save_error_heatmap, save_image  # noqa: E402
from structsplat.contraction_refinement import (  # noqa: E402
    CoefficientProjectionConfig,
    CoefficientProjectionResult,
    project_contracted_coefficients,
)
from structsplat.observation_field import ObservationField2D  # noqa: E402
from structsplat.pixel_contraction import (  # noqa: E402
    PixelContractionConfig,
    contract_image,
    render_observation_field,
)
from structsplat.residual_exchange import (  # noqa: E402
    ResidualExchangeConfig,
    ResidualExchangeResult,
    exchange_residual_columns,
)


REPORT_SCHEMA = "structsplat.hier013_global_projection_development.diagnostic.v1"
ARMS = (
    "h005_control",
    "touched_projection",
    "global_projection",
    "exchange_global_projection",
)
SEEDS = (0, 1, 2)
EXPECTED_SOURCE_SET_SHA256 = "18ae31a25a0d168d876955da13a05eb3e7b0409e05c5d07998120f4ef1ea9350"
EXPECTED_SOURCES = {
    "tests/test_images/COCO_train2014_000000000009.jpg":
        "35cdfe8259aca40d564baf33ee749d82ce852446bd9574f0c47551d8bfffda99",
    "tests/test_images/COCO_train2014_000000000025.jpg":
        "d8f12a26d8803701cabac80494b080f998e5ed9bafaf61a2825ce6212c85487a",
    "tests/test_images/COCO_train2014_000000000030.jpg":
        "0444b10826d376ad9075805061405f6071a62b80eda29c5f284ed77b093d5b1d",
    "tests/test_images/COCO_train2014_000000000034.jpg":
        "2c46871034fa901ae795a8bb916ba7f2f728507cab9e511cced0986bd083d193",
    "tests/test_images/DIV2K_train_HR/0001.png":
        "cdb20d7a462744c269d8e197f735c7bc42e7cda367a940a9b7bc27803b1c8619",
    "tests/test_images/DIV2K_train_HR/0002.png":
        "82325cea74c2cd4681f69a10e36ba15c896d99ec47dc2c687ef07f7497781e09",
    "tests/test_images/DIV2K_train_HR/0115.png":
        "b08214ed8a205d5ff148eb14541de6117f282350bc3e4fc46d2efa8c848073e1",
    "tests/test_images/DIV2K_train_HR/0229.png":
        "e985cdadc0861ae47a76ae66a46290b7aa322b4d2596727634b144cb205c2d18",
    "tests/test_images/DIV2K_train_HR/0268.png":
        "455a05afcc60e0638259bb6dd98018606786cd73ee7118049cff94b48b5d4e7b",
    "tests/test_images/DIV2K_train_HR/0343.png":
        "f70f775deb82a5744fae0640b5b095e35374f7228893dead5750a4b9d7ef8781",
    "tests/test_images/DIV2K_train_HR/0457.png":
        "565bb5b65c50abd4b0715b9318851de400cae1475db9c44a138a3bae275d2a05",
    "tests/test_images/DIV2K_train_HR/0534.png":
        "c605f2a1092cafc85280d618eb55344c58830313dc75b0469a8f7321f11aa4d3",
    "tests/test_images/DIV2K_train_HR/0571.png":
        "6de58e0706300b3496f538dca3b80d478062f4c4396990b3b5e6479300ed71ef",
    "tests/test_images/DIV2K_train_HR/0685.png":
        "c42e9a8e92f57ed8ebff3ba247c7578aa85b59785021123f673c56d895e63364",
    "tests/test_images/DIV2K_train_HR/0799.png":
        "ad42d7e2fe2ee15461e6999e7673a1f96b1be791b4be8c01baca26812f5667db",
    "tests/test_images/DIV2K_train_HR/0800.png":
        "eb6df5bfeacd04334062b6103f6ee8f33af1abd3e1375a7f2c2a4831fa701221",
}
REQUIRED_ARTIFACT_FILES = (
    "source.png",
    "initial_lattice.png",
    "initial_error.png",
    "reconstruction.png",
    "error.png",
    "feature_priority.png",
    "protected.png",
    "centers.png",
    "source_crop.png",
    "reconstruction_crop.png",
    "error_crop.png",
    "field.observation.npz",
    "history.json",
    "recovery_history.json",
    "analysis.npz",
    "config.json",
    "row.json",
)


def _source_set_sha256(bindings: dict[str, str]) -> str:
    payload = "".join(f"{path}\t{bindings[path]}\n" for path in sorted(bindings))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _discover_sources(raw_paths: list[Path]) -> list[Path]:
    discovered: set[Path] = set()
    suffixes = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    for raw in raw_paths:
        path = raw.resolve()
        if path.is_dir():
            discovered.update(
                candidate.resolve()
                for candidate in path.rglob("*")
                if candidate.is_file() and candidate.suffix.lower() in suffixes
            )
        elif path.is_file() and path.suffix.lower() in suffixes:
            discovered.add(path)
        else:
            raise SystemExit(f"image path is missing or unsupported: {raw}")
    actual: dict[str, str] = {}
    for path in sorted(discovered):
        try:
            relative = path.relative_to(ROOT).as_posix()
        except ValueError as exc:
            raise SystemExit(f"HIER-013 inputs must be inside the repository: {path}") from exc
        actual[relative] = report_utils._sha256(path)
    if actual != EXPECTED_SOURCES:
        missing = sorted(set(EXPECTED_SOURCES) - set(actual))
        extra = sorted(set(actual) - set(EXPECTED_SOURCES))
        changed = sorted(
            path for path in set(actual) & set(EXPECTED_SOURCES)
            if actual[path] != EXPECTED_SOURCES[path]
        )
        raise SystemExit(
            "HIER-013 source binding mismatch: "
            f"missing={missing}, extra={extra}, changed={changed}"
        )
    if _source_set_sha256(actual) != EXPECTED_SOURCE_SET_SHA256:
        raise SystemExit("HIER-013 source-set digest mismatch")
    return [ROOT / path for path in sorted(EXPECTED_SOURCES)]


def _source_identity(path: Path) -> tuple[str, str]:
    relative = path.relative_to(ROOT)
    if "DIV2K_train_HR" in relative.parts:
        return "DIV2K", f"DIV2K/{path.stem}"
    return "COCO", f"COCO/{path.stem}"


def _snapshot_sources(output_root: Path) -> list[dict[str, object]]:
    sources = (
        Path(__file__).resolve(),
        ROOT / "src" / "structsplat" / "pixel_contraction.py",
        ROOT / "src" / "structsplat" / "contraction_refinement.py",
        ROOT / "src" / "structsplat" / "residual_exchange.py",
        ROOT / "scripts" / "experiments" / "hier005_pixel_contraction.py",
        ROOT / "scripts" / "experiments" / "hier010_residual_anchor_projection.py",
        ROOT / "scripts" / "experiments" / "hier011_guarded_residual_column_exchange.py",
        ROOT / "scripts" / "check_report_bundle.py",
        ROOT / "tasks" / "HIER-013-global-projection-development-screen.md",
    )
    records: list[dict[str, object]] = []
    for source in sources:
        relative = source.relative_to(ROOT)
        destination = output_root / "source_snapshot" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        records.append(
            {
                "repository_path": str(relative),
                "snapshot_path": str(destination.relative_to(output_root)),
                "bytes": destination.stat().st_size,
                "sha256": report_utils._sha256(destination),
            }
        )
    return records


def _non_rgb_equal(first: ObservationField2D, second: ObservationField2D) -> bool:
    if first.semantic_record() != second.semantic_record():
        return False
    first_arrays = first._array_items()
    second_arrays = second._array_items()
    names = (set(first_arrays) | set(second_arrays)) - {"rgb_coeff"}
    return all(
        name in first_arrays
        and name in second_arrays
        and np.array_equal(first_arrays[name], second_arrays[name])
        for name in names
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", nargs="+", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--target-gaussians", type=int, default=7000)
    parser.add_argument("--max-side", type=int, default=512)
    parser.add_argument("--projection-ridge", type=float, default=1e-8)
    parser.add_argument("--projection-tolerance", type=float, default=1e-6)
    parser.add_argument("--projection-max-iterations", type=int, default=48)
    parser.add_argument("--projection-coefficient-limit", type=float, default=16.0)
    parser.add_argument("--max-exchanges", type=int, default=128)
    parser.add_argument("--site-count", type=int, default=96)
    parser.add_argument("--site-nms-radius", type=int, default=1)
    parser.add_argument("--donor-count", type=int, default=64)
    parser.add_argument("--proposal-frontier", type=int, default=24)
    parser.add_argument("--coefficient-limit", type=float, default=16.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--renderer",
        choices=("additive", "cuda_additive", "cuda_tiled_additive"),
        default="cuda_additive",
    )
    parser.add_argument("--render-chunk", type=int, default=256)
    parser.add_argument("--error-scale", type=float, default=4.0)
    parser.add_argument("--lpips", action="store_true")
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    expected_ints = {
        "target_gaussians": 7000,
        "max_side": 512,
        "projection_max_iterations": 48,
        "max_exchanges": 128,
        "site_count": 96,
        "site_nms_radius": 1,
        "donor_count": 64,
        "proposal_frontier": 24,
        "render_chunk": 256,
    }
    for name, expected in expected_ints.items():
        if getattr(args, name) != expected:
            raise SystemExit(f"frozen protocol requires --{name.replace('_', '-')} {expected}")
    expected_floats = {
        "projection_ridge": 1e-8,
        "projection_tolerance": 1e-6,
        "projection_coefficient_limit": 16.0,
        "coefficient_limit": 16.0,
        "error_scale": 4.0,
    }
    for name, expected in expected_floats.items():
        if float(getattr(args, name)) != expected:
            raise SystemExit(f"frozen protocol requires --{name.replace('_', '-')} {expected}")
    if tuple(args.seeds) != SEEDS:
        raise SystemExit("frozen protocol requires --seeds 0 1 2 in that order")
    if args.device != "cuda" or args.renderer != "cuda_additive" or not args.lpips:
        raise SystemExit(
            "frozen protocol requires --device cuda --renderer cuda_additive --lpips"
        )


def _contraction_config(args: argparse.Namespace) -> PixelContractionConfig:
    return PixelContractionConfig(
        target_gaussians=args.target_gaussians,
        leaf_scale_px=0.18,
        sigma_cutoff=3.0,
        support_fade_alpha=0.0,
        coefficient_domain="signed",
        estimated_row_bytes=32,
        proposal_batch_size=64,
        merge_batch_size=8,
        pair_shortlist=3,
        exact_option_shortlist=2,
        pair_policy="exact_count",
        recovery_steps=50,
        recovery_scope="touched",
        recovery_schedule="progress",
        recovery_progress_checkpoints=16,
        recovery_device=args.device,
        recovery_renderer=args.renderer,
        recovery_render_chunk=args.render_chunk,
        recovery_lr_means=0.005,
        recovery_lr_scales=0.003,
        recovery_lr_rotations=0.001,
        recovery_lr_coefficients=0.003,
        recovery_max_mean_shift_px=1.5,
        recovery_max_log_scale_shift=0.35,
        recovery_max_rotation_shift_rad=0.35,
    )


def _write_tables(output_root: Path, rows: list[dict[str, object]]) -> None:
    payload = {
        "schema": REPORT_SCHEMA,
        "status": "diagnostic",
        "claim_ready": False,
        "row_count": len(rows),
        "expected_row_count": len(EXPECTED_SOURCES) * len(SEEDS) * len(ARMS),
        "rows": rows,
        "rate_warning": (
            "complete_reference_stream_bytes is the self-contained lossless Observation Field "
            "NPZ, not a selected production codec or compression-rate claim"
        ),
    }
    report_utils._write_json(output_root / "metrics.json", payload)
    with (output_root / "metrics.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(report_utils._jsonable(row), sort_keys=True) + "\n")
    columns = sorted({key for row in rows for key in row})
    with (output_root / "metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _write_manifest(output_root: Path) -> None:
    files = []
    for path in sorted(output_root.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            files.append(
                {
                    "path": str(path.relative_to(output_root)),
                    "bytes": path.stat().st_size,
                    "sha256": report_utils._sha256(path),
                }
            )
    report_utils._write_json(
        output_root / "manifest.json",
        {"schema": REPORT_SCHEMA, "status": "diagnostic", "files": files},
    )


def _paired_records(rows: list[dict[str, object]], arm: str) -> list[dict[str, object]]:
    lookup = {
        (str(row["source_image"]), int(row["seed"]), str(row["arm"])): row
        for row in rows
    }
    pairs: list[dict[str, object]] = []
    for source_image in sorted({str(row["source_image"]) for row in rows}):
        for seed in SEEDS:
            control = lookup.get((source_image, seed, "h005_control"))
            candidate = lookup.get((source_image, seed, arm))
            if control is None or candidate is None:
                continue
            control_mse = float(control["masked_mse"])
            candidate_mse = float(candidate["masked_mse"])
            pairs.append(
                {
                    "source_image": source_image,
                    "family": candidate["family"],
                    "seed": seed,
                    "mse_ratio": candidate_mse / control_mse,
                    "psnr_delta_db": float(candidate["psnr_db"]) - float(control["psnr_db"]),
                    "ms_ssim_delta": float(candidate["ms_ssim"]) - float(control["ms_ssim"]),
                    "lpips_delta": (
                        None
                        if candidate["lpips"] is None or control["lpips"] is None
                        else float(candidate["lpips"]) - float(control["lpips"])
                    ),
                    "pixel_max_delta": float(candidate["artifact_pixel_rmse_max"])
                    - float(control["artifact_pixel_rmse_max"]),
                    "patch7_max_delta": float(candidate["artifact_patch_rmse_max_7"])
                    - float(control["artifact_patch_rmse_max_7"]),
                    "pipeline_algorithm_seconds": float(candidate["pipeline_algorithm_seconds"]),
                    "control_algorithm_seconds": float(control["pipeline_algorithm_seconds"]),
                    "projection_seconds": float(candidate["projection_seconds"]),
                    "complete_reference_stream_bytes": int(
                        candidate["complete_reference_stream_bytes"]
                    ),
                }
            )
    return pairs


def _mean_or_none(values: list[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None]
    return None if len(finite) != len(values) or not finite else float(np.mean(finite))


def _arm_aggregate(rows: list[dict[str, object]], arm: str) -> dict[str, object]:
    pairs = _paired_records(rows, arm) if arm != "h005_control" else []
    arm_rows = [row for row in rows if row["arm"] == arm]
    if arm == "h005_control":
        return {
            "arm": arm,
            "paired_cells": len(arm_rows),
            "geometric_mean_mse_ratio": 1.0,
            "mse_reduction_percent": 0.0,
            "mean_psnr_delta_db": 0.0,
            "mean_ms_ssim_delta": 0.0,
            "mean_lpips_delta": 0.0,
            "mean_pixel_max_delta": 0.0,
            "mean_patch7_max_delta": 0.0,
            "median_pipeline_algorithm_seconds": float(
                np.median([float(row["pipeline_algorithm_seconds"]) for row in arm_rows])
            ),
            "median_complete_reference_stream_bytes": float(
                np.median([int(row["complete_reference_stream_bytes"]) for row in arm_rows])
            ),
        }
    ratio = float(np.exp(np.mean(np.log([float(pair["mse_ratio"]) for pair in pairs]))))
    return {
        "arm": arm,
        "paired_cells": len(pairs),
        "geometric_mean_mse_ratio": ratio,
        "mse_reduction_percent": 100.0 * (1.0 - ratio),
        "mean_psnr_delta_db": float(np.mean([float(pair["psnr_delta_db"]) for pair in pairs])),
        "mean_ms_ssim_delta": float(
            np.mean([float(pair["ms_ssim_delta"]) for pair in pairs])
        ),
        "mean_lpips_delta": _mean_or_none([pair["lpips_delta"] for pair in pairs]),
        "mean_pixel_max_delta": float(
            np.mean([float(pair["pixel_max_delta"]) for pair in pairs])
        ),
        "mean_patch7_max_delta": float(
            np.mean([float(pair["patch7_max_delta"]) for pair in pairs])
        ),
        "median_pipeline_algorithm_seconds": float(
            np.median([float(pair["pipeline_algorithm_seconds"]) for pair in pairs])
        ),
        "median_complete_reference_stream_bytes": float(
            np.median([int(pair["complete_reference_stream_bytes"]) for pair in pairs])
        ),
    }


def _image_mean_records(pairs: list[dict[str, object]]) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for source_image in sorted({str(pair["source_image"]) for pair in pairs}):
        group = [pair for pair in pairs if pair["source_image"] == source_image]
        records.append(
            {
                "source_image": source_image,
                "family": group[0]["family"],
                "replicates": len(group),
                "geometric_mean_mse_ratio": float(
                    np.exp(np.mean(np.log([float(pair["mse_ratio"]) for pair in group])))
                ),
                "mean_psnr_delta_db": float(
                    np.mean([float(pair["psnr_delta_db"]) for pair in group])
                ),
                "mean_ms_ssim_delta": float(
                    np.mean([float(pair["ms_ssim_delta"]) for pair in group])
                ),
                "mean_lpips_delta": _mean_or_none(
                    [pair["lpips_delta"] for pair in group]
                ),
                "mean_pixel_max_delta": float(
                    np.mean([float(pair["pixel_max_delta"]) for pair in group])
                ),
                "mean_patch7_max_delta": float(
                    np.mean([float(pair["patch7_max_delta"]) for pair in group])
                ),
            }
        )
    return records


def _bootstrap_interval(image_records: list[dict[str, object]]) -> dict[str, float]:
    logs = np.log(
        np.asarray(
            [float(record["geometric_mean_mse_ratio"]) for record in image_records],
            dtype=np.float64,
        )
    )
    psnr = np.asarray(
        [float(record["mean_psnr_delta_db"]) for record in image_records], dtype=np.float64
    )
    rng = np.random.default_rng(13013)
    indices = rng.integers(0, logs.size, size=(20_000, logs.size))
    ratios = np.exp(np.mean(logs[indices], axis=1))
    psnr_means = np.mean(psnr[indices], axis=1)
    return {
        "mse_ratio_low_95": float(np.quantile(ratios, 0.025)),
        "mse_ratio_high_95": float(np.quantile(ratios, 0.975)),
        "psnr_delta_low_95_db": float(np.quantile(psnr_means, 0.025)),
        "psnr_delta_high_95_db": float(np.quantile(psnr_means, 0.975)),
        "resamples": 20_000,
        "seed": 13013,
    }


def _decision_from_rows(
    rows: list[dict[str, object]], attempts: list[dict[str, object]]
) -> tuple[dict[str, object], dict[str, object]]:
    expected_cells = len(EXPECTED_SOURCES) * len(SEEDS) * len(ARMS)
    arm_aggregates = [_arm_aggregate(rows, arm) for arm in ARMS if any(row["arm"] == arm for row in rows)]
    global_pairs = _paired_records(rows, "global_projection")
    global_images = _image_mean_records(global_pairs)
    bootstrap = _bootstrap_interval(global_images) if len(global_images) == len(EXPECTED_SOURCES) else None
    global_ratio = (
        float(np.exp(np.mean(np.log([float(pair["mse_ratio"]) for pair in global_pairs]))))
        if global_pairs
        else math.inf
    )
    family_ratios = {
        family: float(
            np.exp(
                np.mean(
                    np.log(
                        [
                            float(pair["mse_ratio"])
                            for pair in global_pairs
                            if pair["family"] == family
                        ]
                    )
                )
            )
        )
        for family in ("COCO", "DIV2K")
        if any(pair["family"] == family for pair in global_pairs)
    }
    complete = len(rows) == expected_cells and len(attempts) == expected_cells and all(
        attempt["status"] == "ok" for attempt in attempts
    )
    integrity = complete and all(
        int(row["n_gaussians"]) == 7000
        and bool(row["non_rgb_arrays_bit_exact"])
        and float(row["maintained_render_parity_max_abs"]) <= 2e-6
        and float(row["repeated_render_parity_max_abs"]) <= 2e-6
        and float(row["projection_internal_render_parity_max_abs"]) <= 2e-6
        and float(row["projection_adjoint_relative_error"]) <= 2e-6
        and bool(row["projection_transaction_pass"])
        and float(row["projection_coefficient_abs_max"]) <= 16.0
        and float(row["exchange_internal_render_parity_max_abs"]) <= 2e-6
        and float(row["exchange_pricing_error_max_abs"]) <= 2e-6
        for row in rows
    )
    all_lpips = bool(global_pairs) and all(pair["lpips_delta"] is not None for pair in global_pairs)
    mean_ms_ssim_delta = (
        float(np.mean([float(pair["ms_ssim_delta"]) for pair in global_pairs]))
        if global_pairs
        else -math.inf
    )
    mean_lpips_delta = (
        float(np.mean([float(pair["lpips_delta"]) for pair in global_pairs]))
        if all_lpips
        else math.inf
    )
    no_mse_regression = bool(global_pairs) and all(
        float(pair["mse_ratio"]) <= 1.0 + 1e-8 for pair in global_pairs
    )
    no_local_regression = bool(global_pairs) and all(
        float(pair["pixel_max_delta"]) <= 0.0
        and float(pair["patch7_max_delta"]) <= 0.0
        for pair in global_pairs
    )
    overheads = [
        float(pair["projection_seconds"]) / max(float(pair["control_algorithm_seconds"]), 1e-12)
        for pair in global_pairs
    ]
    median_overhead_ratio = float(np.median(overheads)) if overheads else math.inf
    clauses = {
        "complete_192_cells": complete,
        "integrity": integrity,
        "geometric_mean_mse_ratio_le_0_80": global_ratio <= 0.80,
        "bootstrap_upper_mse_ratio_lt_1": bool(
            bootstrap is not None and bootstrap["mse_ratio_high_95"] < 1.0
        ),
        "both_families_improve": set(family_ratios) == {"COCO", "DIV2K"}
        and all(ratio < 1.0 for ratio in family_ratios.values()),
        "no_paired_mse_regression": no_mse_regression,
        "aggregate_ms_ssim_noninferior": mean_ms_ssim_delta >= -1e-7,
        "aggregate_lpips_noninferior": all_lpips and mean_lpips_delta <= 1e-7,
        "no_local_max_regression": no_local_regression,
        "median_projection_overhead_le_0_25": median_overhead_ratio <= 0.25,
    }
    gate_pass = all(clauses.values())

    exchange_pairs = _paired_records(rows, "exchange_global_projection")
    comparison = "unavailable"
    exchange_vs_global_ratio = None
    if len(exchange_pairs) == len(global_pairs) and global_pairs:
        global_lookup = {
            (pair["source_image"], pair["seed"]): pair for pair in global_pairs
        }
        exchange_over_global = []
        lpips_exchange_minus_global = []
        pixel_exchange_minus_global = []
        patch_exchange_minus_global = []
        for pair in exchange_pairs:
            global_pair = global_lookup[(pair["source_image"], pair["seed"])]
            exchange_over_global.append(
                float(pair["mse_ratio"]) / float(global_pair["mse_ratio"])
            )
            if pair["lpips_delta"] is not None and global_pair["lpips_delta"] is not None:
                lpips_exchange_minus_global.append(
                    float(pair["lpips_delta"]) - float(global_pair["lpips_delta"])
                )
            pixel_exchange_minus_global.append(
                float(pair["pixel_max_delta"]) - float(global_pair["pixel_max_delta"])
            )
            patch_exchange_minus_global.append(
                float(pair["patch7_max_delta"]) - float(global_pair["patch7_max_delta"])
            )
        exchange_vs_global_ratio = float(
            np.exp(np.mean(np.log(exchange_over_global)))
        )
        direct_lower_mse = exchange_vs_global_ratio >= 1.0
        direct_guardrails = (
            lpips_exchange_minus_global
            and float(np.mean(lpips_exchange_minus_global)) >= 0.0
            and float(np.mean(pixel_exchange_minus_global)) >= 0.0
            and float(np.mean(patch_exchange_minus_global)) >= 0.0
        )
        exchange_lower_mse = exchange_vs_global_ratio < 1.0
        exchange_guardrails = (
            lpips_exchange_minus_global
            and float(np.mean(lpips_exchange_minus_global)) <= 0.0
            and float(np.mean(pixel_exchange_minus_global)) <= 0.0
            and float(np.mean(patch_exchange_minus_global)) <= 0.0
        )
        if direct_lower_mse and direct_guardrails:
            comparison = "global_projection"
        elif exchange_lower_mse and exchange_guardrails:
            comparison = "exchange_global_projection"
        else:
            comparison = "heterogeneous_pareto"

    aggregates = {
        "schema": REPORT_SCHEMA,
        "arms": arm_aggregates,
        "global_projection_image_means": global_images,
        "global_projection_bootstrap": bootstrap,
        "global_projection_family_geometric_mean_mse_ratio": family_ratios,
    }
    decision = {
        "schema": REPORT_SCHEMA,
        "status": "diagnostic",
        "claim_ready": False,
        "development_gate_pass": gate_pass,
        "clauses": clauses,
        "expected_cells": expected_cells,
        "successful_cells": len(rows),
        "error_cells": sum(attempt["status"] == "error" for attempt in attempts),
        "global_projection_geometric_mean_mse_ratio": global_ratio,
        "global_projection_mse_reduction_percent": 100.0 * (1.0 - global_ratio),
        "global_projection_mean_psnr_delta_db": (
            float(np.mean([float(pair["psnr_delta_db"]) for pair in global_pairs]))
            if global_pairs
            else None
        ),
        "global_projection_mean_ms_ssim_delta": mean_ms_ssim_delta,
        "global_projection_mean_lpips_delta": mean_lpips_delta if all_lpips else None,
        "global_projection_median_overhead_ratio": median_overhead_ratio,
        "bootstrap": bootstrap,
        "family_geometric_mean_mse_ratio": family_ratios,
        "global_vs_exchange_selection": comparison,
        "exchange_over_global_geometric_mean_mse_ratio": exchange_vs_global_ratio,
        "summary": (
            "The frozen all-row projection clears every development effect, perceptual/local, "
            "work, completeness, and integrity clause on the requested 16-image bank. This "
            "retains it as a FIT-046 development candidate only."
            if gate_pass
            else "The frozen all-row projection misses at least one preregistered development "
            "clause. Preserve the result without retuning these images; no promotion follows."
        ),
    }
    return aggregates, decision


def _plot_curves(
    output_root: Path, rows: list[dict[str, object]], aggregates: dict[str, object]
) -> list[dict[str, object]]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    curve_root = output_root / "curves"
    curve_root.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    image_names = [
        str(record["source_image"])
        for record in aggregates["global_projection_image_means"]
    ]
    for metric, ylabel in (
        ("mean_psnr_delta_db", "mean paired PSNR delta (dB)"),
        ("geometric_mean_mse_ratio", "geometric-mean MSE ratio"),
        ("mean_lpips_delta", "mean paired LPIPS delta"),
    ):
        figure, axis = plt.subplots(figsize=(12.5, 4.2), constrained_layout=True)
        for arm in ARMS[1:]:
            image_records = _image_mean_records(_paired_records(rows, arm))
            lookup = {record["source_image"]: record for record in image_records}
            values = [lookup[name][metric] for name in image_names]
            axis.plot(range(len(image_names)), values, marker="o", linewidth=1.0, label=arm)
        axis.axhline(1.0 if "ratio" in metric else 0.0, color="#333", linewidth=0.8)
        axis.set_xticks(range(len(image_names)), image_names, rotation=55, ha="right", fontsize=7)
        axis.set_ylabel(ylabel)
        axis.grid(True, alpha=0.25)
        axis.legend(fontsize=8)
        path = curve_root / f"per_image__{metric}.svg"
        figure.savefig(path, format="svg")
        plt.close(figure)
        records.append({"kind": "per_image", "metric": metric, "path": str(path.relative_to(output_root))})
    report_utils._write_json(
        curve_root / "catalog.json",
        {"schema": REPORT_SCHEMA, "curve_count": len(records), "curves": records},
    )
    return records


def _write_report(
    output_root: Path,
    rows: list[dict[str, object]],
    attempts: list[dict[str, object]],
    aggregates: dict[str, object],
    decision: dict[str, object],
    curves: list[dict[str, object]],
) -> None:
    aggregate_rows = []
    for arm in aggregates["arms"]:
        lpips = arm["mean_lpips_delta"]
        aggregate_rows.append(
            "<tr>"
            f"<td>{escape(str(arm['arm']))}</td><td>{int(arm['paired_cells'])}</td>"
            f"<td>{float(arm['geometric_mean_mse_ratio']):.6f}</td>"
            f"<td>{float(arm['mse_reduction_percent']):+.2f}%</td>"
            f"<td>{float(arm['mean_psnr_delta_db']):+.4f}</td>"
            f"<td>{float(arm['mean_ms_ssim_delta']):+.8f}</td>"
            f"<td>{'n/a' if lpips is None else f'{float(lpips):+.8f}'}</td>"
            f"<td>{float(arm['mean_pixel_max_delta']):+.6f}</td>"
            f"<td>{float(arm['mean_patch7_max_delta']):+.6f}</td>"
            f"<td>{float(arm['median_pipeline_algorithm_seconds']):.3f}</td>"
            f"<td>{float(arm['median_complete_reference_stream_bytes']):.0f}</td></tr>"
        )
    image_rows = []
    for record in aggregates["global_projection_image_means"]:
        image_rows.append(
            "<tr>"
            f"<td>{escape(str(record['source_image']))}</td><td>{escape(str(record['family']))}</td>"
            f"<td>{float(record['geometric_mean_mse_ratio']):.6f}</td>"
            f"<td>{float(record['mean_psnr_delta_db']):+.4f}</td>"
            f"<td>{float(record['mean_ms_ssim_delta']):+.8f}</td>"
            f"<td>{float(record['mean_lpips_delta']):+.8f}</td>"
            f"<td>{float(record['mean_pixel_max_delta']):+.6f}</td>"
            f"<td>{float(record['mean_patch7_max_delta']):+.6f}</td></tr>"
        )
    clause_items = "".join(
        f"<li class='{'pass' if passed else 'fail'}'>{'PASS' if passed else 'FAIL'} — "
        f"{escape(name)}</li>"
        for name, passed in decision["clauses"].items()
    )
    error_items = "".join(
        f"<li>{escape(str(attempt['image']))} seed {attempt['seed']} / "
        f"{escape(str(attempt['arm']))}: {escape(str(attempt.get('error', 'unknown error')))}</li>"
        for attempt in attempts
        if attempt["status"] == "error"
    ) or "<li>None</li>"
    cards = []
    for row in rows:
        if int(row["seed"]) != 0 or row["arm"] != "global_projection":
            continue
        artifact = str(row["artifact_dir"])
        cards.append(
            f"<section class='card'><h3>{escape(str(row['source_image']))}</h3>"
            f"<p>{float(row['psnr_db']):.3f} dB, {float(row['psnr_delta_vs_control_db']):+.3f} dB; "
            f"MSE reduction {100.0 * (1.0 - float(row['masked_mse_ratio_vs_control'])):.2f}%</p>"
            "<div class='images'>"
            f"<figure><a href='{artifact}/source.png'><img src='{artifact}/source.png'></a><figcaption>source</figcaption></figure>"
            f"<figure><a href='{artifact}/reconstruction.png'><img src='{artifact}/reconstruction.png'></a><figcaption>global projection</figcaption></figure>"
            f"<figure><a href='{artifact}/error.png'><img src='{artifact}/error.png'></a><figcaption>absolute error ×4</figcaption></figure>"
            f"<figure><a href='{artifact}/error_crop.png'><img src='{artifact}/error_crop.png'></a><figcaption>worst-area error ×4</figcaption></figure>"
            "</div></section>"
        )
    artifact_links = []
    for row in rows:
        artifact = str(row["artifact_dir"])
        title = f"{row['source_image']} / seed {row['seed']} / {row['arm']}"
        for filename in REQUIRED_ARTIFACT_FILES + (
            "projection_history.json",
            "exchange_history.json",
        ):
            artifact_links.append(
                f"<li><a href='{artifact}/{filename}'>{escape(title)} / {filename}</a></li>"
            )
    curve_links = "".join(
        f"<li><a href='{curve['path']}'>{escape(str(curve['metric']))}</a></li>"
        for curve in curves
    )
    snapshot_links = "".join(
        f"<li><a href='{path.relative_to(output_root)}'>{escape(str(path.relative_to(output_root)))}</a></li>"
        for path in sorted((output_root / "source_snapshot").rglob("*"))
        if path.is_file()
    )
    verdict = "PASS" if decision["development_gate_pass"] else "FAIL"
    document = f"""<!doctype html><html><head><meta charset='utf-8'>
<meta name='viewport' content='width=device-width,initial-scale=1'>
<title>HIER-013 global projection development screen</title><style>
body{{font-family:system-ui,sans-serif;margin:0;background:#f2f5f7;color:#17232c}}main{{max-width:1600px;margin:auto;padding:24px}}
.warning{{background:#fff3cd;border:1px solid #d3ad45;padding:13px;border-radius:8px}}.verdict{{background:{'#e4f6ec' if decision['development_gate_pass'] else '#fee9e7'};border:1px solid {'#2e8b57' if decision['development_gate_pass'] else '#b64335'};padding:16px;border-radius:8px}}
table{{border-collapse:collapse;width:100%;background:white;font-size:12px}}th,td{{border:1px solid #d9e1e6;padding:7px;text-align:right}}th:first-child,td:first-child{{text-align:left}}.pass{{color:#087a48}}.fail{{color:#b62929;font-weight:700}}
.card{{background:white;border:1px solid #d9e1e6;border-radius:9px;padding:14px;margin:18px 0}}.images{{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:10px}}figure{{margin:0}}img{{width:100%;height:auto;background:#101417}}figcaption{{font-size:12px;color:#52616c}}.links{{columns:3;font-size:11px}}
</style></head><body><main><p>STRUCTSPLAT / HIER-013 / DEVELOPMENT DIAGNOSTIC</p>
<h1>Global appearance projection across 16 repository images</h1>
<p class='warning'><strong>Not confirmation.</strong> All COCO/DIV2K inputs are exposed development data. The source tree is dirty and no distinct prospective reviewer approved the protocol. CUDA atomics are numerically nondeterministic. Lossless NPZ bytes are a complete reference stream, not production codec rate. No default, semantic, speed, or FIT-046 conclusion follows.</p>
<section class='verdict'><h2>Frozen development gate: {verdict}</h2><p>{escape(str(decision['summary']))}</p><ul>{clause_items}</ul></section>
<p><a href='manifest.json'>manifest</a> · <a href='metrics.json'>metrics JSON</a> · <a href='metrics.jsonl'>JSONL</a> · <a href='metrics.csv'>CSV</a> · <a href='aggregates.json'>aggregates</a> · <a href='decision.json'>decision</a> · <a href='attempts.json'>attempts/errors</a> · <a href='config.json'>config</a></p>
<h2>Arm aggregates</h2><table><thead><tr><th>arm</th><th>paired cells</th><th>geo MSE ratio</th><th>MSE reduction</th><th>ΔPSNR</th><th>ΔMS-SSIM</th><th>ΔLPIPS</th><th>Δpixel max</th><th>Δ7×7 max</th><th>median algorithm s</th><th>median NPZ bytes</th></tr></thead><tbody>{''.join(aggregate_rows)}</tbody></table>
<h2>Global projection by image</h2><table><thead><tr><th>image</th><th>family</th><th>geo MSE ratio</th><th>ΔPSNR</th><th>ΔMS-SSIM</th><th>ΔLPIPS</th><th>Δpixel max</th><th>Δ7×7 max</th></tr></thead><tbody>{''.join(image_rows)}</tbody></table>
<h2>Error cells</h2><ul>{error_items}</ul><h2>Representative seed-0 visuals</h2>{''.join(cards)}
<h2>Curves</h2><ul>{curve_links}</ul><h2>All raw cell artifacts</h2><ul class='links'>{''.join(artifact_links)}</ul>
<h2>Executed source snapshot</h2><ul>{snapshot_links}</ul></main></body></html>"""
    (output_root / "index.html").write_text(document, encoding="utf-8")


def _projection_values(
    result: CoefficientProjectionResult | None,
) -> dict[str, object]:
    if result is None:
        return {
            "projection_scope": "none",
            "projection_trainable_rows": 0,
            "projection_selected_iteration": 0,
            "projection_checkpoint_count": 0,
            "projection_initial_sse": 0.0,
            "projection_final_sse": 0.0,
            "projection_forward_applications": 0,
            "projection_transpose_applications": 0,
            "projection_adjoint_relative_error": 0.0,
            "projection_internal_render_parity_max_abs": 0.0,
            "projection_relative_normal_residual_max": 0.0,
            "projection_coefficient_abs_max": 0.0,
            "projection_transaction_pass": True,
            "projection_seconds": 0.0,
        }
    selected = next(checkpoint for checkpoint in result.checkpoints if checkpoint.selected)
    return {
        "projection_trainable_rows": result.trainable_rows,
        "projection_selected_iteration": result.selected_iteration,
        "projection_checkpoint_count": len(result.checkpoints),
        "projection_initial_sse": result.initial_sse,
        "projection_final_sse": result.final_sse,
        "projection_forward_applications": result.forward_applications,
        "projection_transpose_applications": result.transpose_applications,
        "projection_adjoint_relative_error": result.adjoint_relative_error,
        "projection_internal_render_parity_max_abs": result.maintained_render_parity_max_abs,
        "projection_relative_normal_residual_max": result.relative_normal_residual_max,
        "projection_coefficient_abs_max": selected.coefficient_abs_max,
        "projection_transaction_pass": bool(
            selected.selectable
            and selected.raw_sse <= result.initial_sse + 1e-8 * max(result.initial_sse, 1.0)
        ),
        "projection_seconds": result.elapsed_seconds,
    }


def _exchange_values(result: ResidualExchangeResult | None) -> dict[str, object]:
    if result is None:
        return {
            "exchange_accepted": 0,
            "exchange_stop_reason": "not_run",
            "exchange_proposed_pairs": 0,
            "exchange_cold_rendered_pairs": 0,
            "exchange_initial_sse": 0.0,
            "exchange_final_sse": 0.0,
            "exchange_pricing_error_max_abs": 0.0,
            "exchange_internal_render_parity_max_abs": 0.0,
            "exchange_repeated_render_parity_max_abs": 0.0,
            "exchange_seconds": 0.0,
        }
    return {
        "exchange_accepted": result.accepted_exchanges,
        "exchange_stop_reason": result.stop_reason,
        "exchange_proposed_pairs": result.proposed_pairs,
        "exchange_cold_rendered_pairs": result.cold_rendered_pairs,
        "exchange_initial_sse": result.initial_sse,
        "exchange_final_sse": result.final_sse,
        "exchange_pricing_error_max_abs": result.maximum_pricing_error_abs,
        "exchange_internal_render_parity_max_abs": result.maintained_render_parity_max_abs,
        "exchange_repeated_render_parity_max_abs": result.repeated_render_parity_max_abs,
        "exchange_seconds": result.elapsed_seconds,
    }


def _write_cell(
    *,
    output_root: Path,
    image_path: Path,
    image: np.ndarray,
    mask: np.ndarray,
    raster_record: dict[str, object],
    family: str,
    source_image: str,
    seed: int,
    arm: str,
    field: ObservationField2D,
    expected_reconstruction: np.ndarray,
    geometry_source: ObservationField2D,
    base_reconstruction: np.ndarray,
    control_history: list[dict[str, object]],
    recovery_history: list[dict[str, object]],
    touched_mask: np.ndarray,
    projection: CoefficientProjectionResult | None,
    projection_scope: str,
    exchange: ResidualExchangeResult | None,
    pipeline_algorithm_seconds: float,
    peak_cuda_bytes: int,
    args: argparse.Namespace,
) -> dict[str, object]:
    cell_started = time.perf_counter()
    slug = source_image.replace("/", "__")
    artifact_dir = (
        output_root
        / "artifacts"
        / f"{slug}__seed{seed}__{arm}__n{args.target_gaussians}"
    )
    artifact_dir.mkdir(parents=True, exist_ok=False)
    field_path = artifact_dir / "field.observation.npz"
    field.save_lossless(field_path)
    decode_started = time.perf_counter()
    cold_field = ObservationField2D.load_lossless(field_path)
    cold_decode_seconds = time.perf_counter() - decode_started
    render_started = time.perf_counter()
    cold = render_observation_field(
        cold_field,
        device=args.device,
        renderer=args.renderer,
        render_chunk=args.render_chunk,
    )
    render_seconds = time.perf_counter() - render_started
    repeated = render_observation_field(
        cold_field,
        device=args.device,
        renderer=args.renderer,
        render_chunk=args.render_chunk,
    )
    maintained_parity = float(np.max(np.abs(cold - expected_reconstruction)))
    repeated_parity = float(np.max(np.abs(repeated - cold)))
    metric_started = time.perf_counter()
    metrics = report_utils._metric_values(
        cold,
        image,
        mask,
        device=args.device,
        compute_lpips=args.lpips,
    )
    metric_seconds = time.perf_counter() - metric_started

    save_image(str(artifact_dir / "source.png"), image)
    save_image(str(artifact_dir / "initial_lattice.png"), base_reconstruction)
    save_error_heatmap(
        str(artifact_dir / "initial_error.png"),
        base_reconstruction - image,
        scale=args.error_scale,
    )
    save_image(str(artifact_dir / "reconstruction.png"), cold)
    save_error_heatmap(
        str(artifact_dir / "error.png"), cold - image, scale=args.error_scale
    )
    initial_score = np.mean(
        (base_reconstruction.astype(np.float64) - image.astype(np.float64)) ** 2,
        axis=2,
    ).astype(np.float32)
    viz_utils._save_scalar(artifact_dir / "feature_priority.png", initial_score, mask)
    exchange_means = (
        np.empty((0, 2), dtype=np.float32)
        if exchange is None
        else exchange.field.means_xy[exchange.replaced_row_mask]
    )
    viz_utils._save_centers(
        artifact_dir / "protected.png",
        image,
        mask,
        exchange_means,
        color=(1.0, 0.0, 0.2),
    )
    viz_utils._save_centers(
        artifact_dir / "centers.png",
        image,
        mask,
        cold_field.means_xy,
        color=(0.0, 1.0, 0.2),
    )
    crop_bounds = viz_utils._worst_crop_bounds(cold, image, mask)
    viz_utils._save_crop(artifact_dir / "source_crop.png", image, crop_bounds)
    viz_utils._save_crop(artifact_dir / "reconstruction_crop.png", cold, crop_bounds)
    error_visual = np.repeat(
        np.clip(np.mean(np.abs(cold - image), axis=2) * args.error_scale, 0.0, 1.0)[
            :, :, None
        ],
        3,
        axis=2,
    )
    viz_utils._save_crop(artifact_dir / "error_crop.png", error_visual, crop_bounds)
    report_utils._write_json(artifact_dir / "history.json", control_history)
    report_utils._write_json(artifact_dir / "recovery_history.json", recovery_history)
    projection_history = [] if projection is None else projection.checkpoint_records()
    exchange_history = [] if exchange is None else exchange.checkpoint_records()
    report_utils._write_json(artifact_dir / "projection_history.json", projection_history)
    report_utils._write_json(artifact_dir / "exchange_history.json", exchange_history)
    trainable_mask = (
        np.zeros(field.n, dtype=bool)
        if projection_scope == "none"
        else (touched_mask if projection_scope == "touched" else np.ones(field.n, dtype=bool))
    )
    np.savez_compressed(
        artifact_dir / "analysis.npz",
        residual_score=initial_score,
        projection_trainable_row_mask=trainable_mask,
        exchange_replaced_row_mask=(
            np.zeros(field.n, dtype=bool)
            if exchange is None
            else exchange.replaced_row_mask
        ),
        exchange_means=exchange_means,
        crop_bounds=np.asarray(crop_bounds, dtype=np.int32),
        full_frame_mask=mask,
    )
    report_utils._write_json(
        artifact_dir / "config.json",
        {
            "schema": REPORT_SCHEMA,
            "status": "diagnostic",
            "source_image": source_image,
            "family": family,
            "seed": seed,
            "arm": arm,
            "projection_scope": projection_scope,
            "pipeline_algorithm_seconds": pipeline_algorithm_seconds,
        },
    )
    canonical_raw_bytes = int(
        sum(array.nbytes for array in cold_field._array_items().values())
    )
    projection_values = _projection_values(projection)
    projection_values["projection_scope"] = projection_scope
    exchange_values = _exchange_values(exchange)
    row: dict[str, object] = {
        "schema": REPORT_SCHEMA,
        "status": "diagnostic",
        "image": f"{source_image}__seed{seed}",
        "source_image": source_image,
        "family": family,
        "seed": seed,
        "arm": arm,
        "artifact_dir": str(artifact_dir.relative_to(output_root)),
        "source_path": str(image_path),
        "source_sha256": report_utils._sha256(image_path),
        "source_file_bytes": image_path.stat().st_size,
        "mask_policy": "full_frame_all_true",
        "evaluation_mask_sha256": hashlib.sha256(mask.tobytes()).hexdigest(),
        "original_width": raster_record["original_width"],
        "original_height": raster_record["original_height"],
        "width": image.shape[1],
        "height": image.shape[0],
        "pixels": int(mask.size),
        "active_pixels": int(mask.sum()),
        "target_gaussians": args.target_gaussians,
        "n_gaussians": cold_field.n,
        "reduction_factor": int(mask.sum()) / cold_field.n,
        "geometry_source_canonical_sha256": geometry_source.canonical_hash(),
        "non_rgb_arrays_bit_exact": _non_rgb_equal(cold_field, geometry_source),
        "pipeline_algorithm_seconds": pipeline_algorithm_seconds,
        "cold_decode_seconds": cold_decode_seconds,
        "render_seconds": render_seconds,
        "metric_seconds": metric_seconds,
        "total_seconds": pipeline_algorithm_seconds
        + cold_decode_seconds
        + 2.0 * render_seconds
        + metric_seconds,
        "peak_cuda_allocated_bytes": peak_cuda_bytes,
        "canonical_raw_bytes": canonical_raw_bytes,
        "lossless_reference_bytes": field_path.stat().st_size,
        "complete_reference_stream_bytes": field_path.stat().st_size,
        "complete_reference_stream_bpp": 8.0 * field_path.stat().st_size / mask.size,
        "maintained_render_parity_max_abs": maintained_parity,
        "repeated_render_parity_max_abs": repeated_parity,
        "field_canonical_sha256": cold_field.canonical_hash(),
        "field_file_sha256": report_utils._sha256(field_path),
        "cell_packaging_seconds": time.perf_counter() - cell_started,
        **projection_values,
        **exchange_values,
        **metrics,
    }
    report_utils._write_json(artifact_dir / "row.json", row)
    return row


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    _validate_args(args)
    images = _discover_sources(args.images)
    output_root = args.out.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise SystemExit(f"output directory must be absent or empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    command = shlex.join(
        [sys.executable, str(Path(__file__).relative_to(ROOT)), *(argv or sys.argv[1:])]
    )
    source_snapshot = _snapshot_sources(output_root)
    contraction_config = _contraction_config(args)
    projection_config = CoefficientProjectionConfig(
        ridge=args.projection_ridge,
        tolerance=args.projection_tolerance,
        max_iterations=args.projection_max_iterations,
        coefficient_abs_limit=args.projection_coefficient_limit,
    )
    exchange_config = ResidualExchangeConfig(
        candidate_shapes=CANDIDATE_SHAPES,
        max_exchanges=args.max_exchanges,
        site_count=args.site_count,
        site_nms_radius_px=args.site_nms_radius,
        donor_count=args.donor_count,
        proposal_frontier=args.proposal_frontier,
        coefficient_abs_limit=args.coefficient_limit,
    )

    import torch

    import structsplat

    if not torch.cuda.is_available():
        raise SystemExit("HIER-013 frozen protocol requires an available CUDA device")
    protocol_record = {
        "sources": EXPECTED_SOURCES,
        "source_set_sha256": EXPECTED_SOURCE_SET_SHA256,
        "seeds": list(SEEDS),
        "arms": list(ARMS),
        "contraction": asdict(contraction_config),
        "projection": asdict(projection_config),
        "exchange": asdict(exchange_config),
        "decision": {
            "geometric_mean_mse_ratio_max": 0.80,
            "bootstrap_upper_mse_ratio_max_exclusive": 1.0,
            "bootstrap_resamples": 20_000,
            "bootstrap_seed": 13013,
            "mse_cell_relative_tolerance": 1e-8,
            "ms_ssim_mean_delta_tolerance": -1e-7,
            "lpips_mean_delta_tolerance": 1e-7,
            "local_max_delta_tolerance": 0.0,
            "median_projection_overhead_ratio_max": 0.25,
        },
    }
    protocol_digest = hashlib.sha256(
        json.dumps(report_utils._jsonable(protocol_record), sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    config: dict[str, object] = {
        "schema": REPORT_SCHEMA,
        "task": "HIER-013",
        "status": "diagnostic",
        "claim_ready": False,
        "command": command,
        "args": vars(args),
        "protocol": protocol_record,
        "protocol_digest": protocol_digest,
        "protocol_review": None,
        "git": report_utils._git_record(),
        "executed_source_snapshot": source_snapshot,
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "structsplat": getattr(structsplat, "__version__", "unknown"),
        },
        "cuda_device": torch.cuda.get_device_name(0),
        "evidence_limits": [
            "All requested COCO/DIV2K sources are exposed development data, not confirmation.",
            "Dirty-source snapshot and no distinct prospective protocol review.",
            "CUDA recovery/render/projection atomics are numerically, not bit, reproducible.",
            "Arms spend unequal work; no speed conclusion.",
            "Lossless Observation Field NPZ is a complete reference stream, not codec rate.",
            "No default, Field V2 semantic, FIT-046, or BENCH-020 decision.",
        ],
    }
    report_utils._write_json(output_root / "config.json", config)

    rows: list[dict[str, object]] = []
    attempts: list[dict[str, object]] = []
    projection_rows: list[dict[str, object]] = []
    exchange_rows: list[dict[str, object]] = []
    journal_path = output_root / "metrics.journal.jsonl"
    run_started = time.perf_counter()
    for image_path in images:
        family, source_image = _source_identity(image_path)
        image, loaded_mask, raster_record = report_utils._load_evaluation_raster(
            image_path,
            None,
            max_side=args.max_side,
            mask_threshold=0.5,
        )
        if loaded_mask is not None:
            raise RuntimeError("HIER-013 requires internally generated full-frame masks")
        mask = np.ones(image.shape[:2], dtype=bool)
        for seed in SEEDS:
            completed_arms: set[str] = set()
            try:
                np.random.seed(seed)
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                torch.cuda.reset_peak_memory_stats()
                control_started = time.perf_counter()
                control = contract_image(image, contraction_config, mask=mask)
                control_seconds = time.perf_counter() - control_started
                all_rows = np.ones(control.field.n, dtype=bool)
                no_rows = np.zeros(control.field.n, dtype=bool)
                touched = project_contracted_coefficients(
                    control.field,
                    image,
                    mask,
                    control.touched_row_mask,
                    control.protected_row_mask,
                    config=projection_config,
                    device=args.device,
                    renderer=args.renderer,
                    render_chunk=args.render_chunk,
                )
                global_projection = project_contracted_coefficients(
                    control.field,
                    image,
                    mask,
                    all_rows,
                    no_rows,
                    config=projection_config,
                    device=args.device,
                    renderer=args.renderer,
                    render_chunk=args.render_chunk,
                )
                exchange = exchange_residual_columns(
                    control.field,
                    image,
                    mask,
                    config=exchange_config,
                    device=args.device,
                    renderer=args.renderer,
                    render_chunk=args.render_chunk,
                )
                exchange_global = project_contracted_coefficients(
                    exchange.field,
                    image,
                    mask,
                    np.ones(exchange.field.n, dtype=bool),
                    np.zeros(exchange.field.n, dtype=bool),
                    config=projection_config,
                    device=args.device,
                    renderer=args.renderer,
                    render_chunk=args.render_chunk,
                )
                peak_cuda_bytes = int(torch.cuda.max_memory_allocated())
                arm_values = {
                    "h005_control": (
                        control.field,
                        control.reconstruction,
                        control.field,
                        None,
                        "none",
                        None,
                        control_seconds,
                    ),
                    "touched_projection": (
                        touched.field,
                        touched.reconstruction,
                        control.field,
                        touched,
                        "touched",
                        None,
                        control_seconds + touched.elapsed_seconds,
                    ),
                    "global_projection": (
                        global_projection.field,
                        global_projection.reconstruction,
                        control.field,
                        global_projection,
                        "all",
                        None,
                        control_seconds + global_projection.elapsed_seconds,
                    ),
                    "exchange_global_projection": (
                        exchange_global.field,
                        exchange_global.reconstruction,
                        exchange.field,
                        exchange_global,
                        "all",
                        exchange,
                        control_seconds + exchange.elapsed_seconds + exchange_global.elapsed_seconds,
                    ),
                }
                for arm in ARMS:
                    (
                        field,
                        expected_reconstruction,
                        geometry_source,
                        projection,
                        projection_scope,
                        arm_exchange,
                        pipeline_seconds,
                    ) = arm_values[arm]
                    row = _write_cell(
                        output_root=output_root,
                        image_path=image_path,
                        image=image,
                        mask=mask,
                        raster_record=raster_record,
                        family=family,
                        source_image=source_image,
                        seed=seed,
                        arm=arm,
                        field=field,
                        expected_reconstruction=expected_reconstruction,
                        geometry_source=geometry_source,
                        base_reconstruction=control.reconstruction,
                        control_history=control.history_records(),
                        recovery_history=control.recovery_records(),
                        touched_mask=control.touched_row_mask,
                        projection=projection,
                        projection_scope=projection_scope,
                        exchange=arm_exchange,
                        pipeline_algorithm_seconds=pipeline_seconds,
                        peak_cuda_bytes=peak_cuda_bytes,
                        args=args,
                    )
                    rows.append(row)
                    completed_arms.add(arm)
                    attempts.append(
                        {
                            "schema": REPORT_SCHEMA,
                            "status": "ok",
                            "image": source_image,
                            "family": family,
                            "seed": seed,
                            "arm": arm,
                            "artifact_dir": row["artifact_dir"],
                        }
                    )
                    with journal_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(report_utils._jsonable(row), sort_keys=True) + "\n")
                for arm, result in (
                    ("touched_projection", touched),
                    ("global_projection", global_projection),
                    ("exchange_global_projection", exchange_global),
                ):
                    for checkpoint in result.checkpoints:
                        projection_rows.append(
                            {
                                "schema": REPORT_SCHEMA,
                                "image": source_image,
                                "family": family,
                                "seed": seed,
                                "arm": arm,
                                **checkpoint.to_record(),
                            }
                        )
                for checkpoint in exchange.checkpoints:
                    exchange_rows.append(
                        {
                            "schema": REPORT_SCHEMA,
                            "image": source_image,
                            "family": family,
                            "seed": seed,
                            **checkpoint.to_record(),
                        }
                    )
                print(
                    f"{source_image} seed={seed}: "
                    + ", ".join(
                        f"{row['arm']}={float(row['psnr_db']):.3f}dB"
                        for row in rows
                        if row["source_image"] == source_image and row["seed"] == seed
                    ),
                    flush=True,
                )
            except Exception as exc:  # preserve the frozen matrix and continue
                error = f"{type(exc).__name__}: {exc}"[:500]
                trace = traceback.format_exc(limit=12)
                for arm in ARMS:
                    if arm in completed_arms:
                        continue
                    attempts.append(
                        {
                            "schema": REPORT_SCHEMA,
                            "status": "error",
                            "image": source_image,
                            "family": family,
                            "seed": seed,
                            "arm": arm,
                            "error": error,
                            "traceback": trace,
                        }
                    )
                print(f"ERROR {source_image} seed={seed}: {error}", flush=True)
                torch.cuda.empty_cache()

    by_key = {
        (str(row["source_image"]), int(row["seed"]), str(row["arm"])): row
        for row in rows
    }
    for row in rows:
        control = by_key.get((str(row["source_image"]), int(row["seed"]), "h005_control"))
        if control is None:
            continue
        row["psnr_delta_vs_control_db"] = float(row["psnr_db"]) - float(control["psnr_db"])
        row["masked_mse_ratio_vs_control"] = float(row["masked_mse"]) / float(
            control["masked_mse"]
        )
        row["pixel_max_delta_vs_control"] = float(row["artifact_pixel_rmse_max"]) - float(
            control["artifact_pixel_rmse_max"]
        )
        row["patch7_max_delta_vs_control"] = float(
            row["artifact_patch_rmse_max_7"]
        ) - float(control["artifact_patch_rmse_max_7"])
        report_utils._write_json(output_root / str(row["artifact_dir"]) / "row.json", row)

    report_utils._write_json(
        output_root / "attempts.json",
        {
            "schema": REPORT_SCHEMA,
            "expected_cells": len(EXPECTED_SOURCES) * len(SEEDS) * len(ARMS),
            "attempts": attempts,
        },
    )
    report_utils._write_json(
        output_root / "projection_checkpoints.json",
        {"schema": REPORT_SCHEMA, "rows": projection_rows},
    )
    report_utils._write_json(
        output_root / "exchange_checkpoints.json",
        {"schema": REPORT_SCHEMA, "rows": exchange_rows},
    )
    aggregates, decision = _decision_from_rows(rows, attempts)
    report_utils._write_json(output_root / "aggregates.json", aggregates)
    report_utils._write_json(output_root / "decision.json", decision)
    _write_tables(output_root, rows)
    curves = _plot_curves(output_root, rows, aggregates) if rows else []
    _write_report(output_root, rows, attempts, aggregates, decision, curves)
    config["elapsed_seconds"] = time.perf_counter() - run_started
    config["decision"] = decision
    report_utils._write_json(output_root / "config.json", config)
    _write_manifest(output_root)
    print(f"wrote diagnostic report: {output_root / 'index.html'}", flush=True)
    return 0 if len(rows) == len(EXPECTED_SOURCES) * len(SEEDS) * len(ARMS) else 2


if __name__ == "__main__":
    raise SystemExit(main())
