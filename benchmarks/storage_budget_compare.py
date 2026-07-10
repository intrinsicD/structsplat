"""Run the frozen 168 KiB equal-capacity convergence benchmark.

``168 KiB`` means exactly 172,032 bytes. The matched-policy lane counts a frozen
constant-RGB rotation/scale Gaussian as eight float32 values (32 bytes), so the exact cap is
5,376 Gaussians. This analytical representation payload is deliberately separate from source,
prepared-target, reconstruction, checkpoint, and actual SSPL1 codec sizes.

The method tuple below is an explicit snapshot of every method registered by the fair-density
harness when this protocol was defined. Keeping it literal makes later registry additions fail a
test instead of silently changing an already named benchmark lane.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

from benchmarks import fair_density_control_compare as F
from benchmarks import results_index
from benchmarks import storage_budget as SB
from benchmarks.common import resolve_seeds, write_json


DEFAULT_IMAGES = (
    Path("tests/test_images/COCO_train2014_000000000009.jpg"),
    Path("tests/test_images/COCO_train2014_000000000025.jpg"),
    Path("tests/test_images/COCO_train2014_000000000030.jpg"),
    Path("tests/test_images/COCO_train2014_000000000034.jpg"),
)

STORAGE_BENCHMARK_METHODS = (
    "structsplat_best_default",
    "structsplat_best_cosine",
    "structsplat_best_cosine_tail",
    "structsplat_best_checkpoint",
    "structsplat_best_checkpoint_lowpass2x_f10",
    "structsplat_best_generation_filter",
    "structsplat_best_generation_filter_a18pi",
    "structsplat_best_generation_filter_a36pi",
    "structsplat_best_ssim010",
    "structsplat_best_l1_only",
    "structsplat_best_charbonnier",
    "structsplat_best_tensor_loss",
    "structsplat_best_gcr015",
    "structsplat_best_gcr015_e2",
    "structsplat_best_gcr015_e4",
    "structsplat_best_gcr030",
    "structsplat_best_gcr060",
    "structsplat_best_color_final",
    "structsplat_best_relocate",
    "structsplat_best_adaptive_1p5x",
    "gaussianimage_fixed_full",
    "gaussianimage_plus_residual",
    "image_gs_residual",
    "instant_gi_quadtree_fixed",
    "structsplat_onedge_residual",
    "structsplat_onedge_residual_relocate",
    "structsplat_onedge_residual_featurecap",
    "structsplat_onedge_residual_feature_rel",
    "structsplat_onedge_tensor",
    "structsplat_onedge_tensor_featurecap",
    "structsplat_onedge_tensor_feature_rel",
    "structsplat_flanking_tensor",
    "structsplat_quadtree_wse_residual",
    "structsplat_quadtree_wse_residual_relocate",
    "structsplat_quadtree_wse_residual_featurecap",
    "structsplat_quadtree_wse_residual_feature_rel",
    "structsplat_quadtree_wse_tensor",
    "structsplat_quadtree_wse_tensor_featurecap",
    "structsplat_quadtree_wse_tensor_feature_rel",
    "structsplat_quadtree_hybrid_tensor",
    "floyd_steinberg_tensor",
)

DEFAULT_OUTDIR = Path("results/storage_budget_168k_all_methods")
DEFAULT_ROOT_INDEX = Path("results/index.html")
DEFAULT_ADDITIONAL_REPORTS = (
    Path("results/fair_gaussian_variants_20260710_full_external_same_hparams"),
    Path("results/fit016_lowpass_coco4_500"),
    Path("results/fit015_checkpoint_kodak4_r160_5000"),
    Path("results/native_gaussianimage_matched_500_official_two_seed"),
    Path("results/native_image_gs_fixedn_500_official_two_seed"),
)
DEFAULT_MAX_SIDE = 160
DEFAULT_ITERS = 10_000
DEFAULT_LOG_EVERY = 100
DEFAULT_EARLY_STOP_PATIENCE = 6
DEFAULT_EARLY_STOP_MIN_DELTA = 0.005
DEFAULT_EARLY_STOP_MIN_ITERS = 6_500
DEFAULT_SEEDS = (0, 1)


def fair_args(args: argparse.Namespace) -> SimpleNamespace:
    """Translate the frozen storage protocol into the fair harness namespace."""
    return SimpleNamespace(
        images=[Path(path) for path in args.images],
        outdir=Path(args.outdir),
        budgets=[SB.FIXED_STORAGE_GAUSSIAN_CAP],
        methods=list(args.methods),
        max_side=int(args.max_side),
        start_fraction=0.5,
        growth_waves=4,
        iters=int(args.iters),
        log_every=int(args.log_every),
        early_stop_patience=int(args.early_stop_patience),
        early_stop_min_delta=float(args.early_stop_min_delta),
        early_stop_min_iters=int(args.early_stop_min_iters),
        target_psnr=35.0,
        target_psnrs=[22.0, 24.0, 26.0, 28.0, 30.0, 32.0, 35.0, 40.0],
        flat_frac=0.02,
        corner_frac=0.15,
        render_chunk=int(args.render_chunk),
        renderer="cuda",
        support_fade=False,
        pixel_loss="l1",
        ssim_weight=0.3,
        lpips=bool(args.lpips),
        relocate_fraction=0.25,
        relocate_downsample=4,
        feature_cap=12.0,
        feature_cap_reference_side=160.0,
        storage_budget_bytes=SB.FIXED_STORAGE_BUDGET_BYTES,
        compute_codec_metrics=bool(args.compute_codec_metrics),
        resume=bool(args.resume),
        max_new_cells=args.max_new_cells,
        device=args.device,
        seed=0,
        seeds=[int(seed) for seed in args.seeds],
    )


def _protocol_hash(outdir: Path, rows: list[dict[str, Any]]) -> str | None:
    hashes = {
        str(row["run_protocol_sha256"])
        for row in rows
        if row.get("run_protocol_sha256") is not None
    }
    if len(hashes) == 1:
        return next(iter(hashes))
    config_path = outdir / "config.json"
    if not config_path.is_file():
        return None
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    resolved = payload.get("resolved", payload) if isinstance(payload, dict) else {}
    value = resolved.get("run_protocol_sha256") if isinstance(resolved, dict) else None
    return None if value is None else str(value)


def write_report_manifest(
    outdir: Path,
    rows: list[dict[str, Any]],
    *,
    images: Sequence[Path],
    methods: Sequence[str],
    seeds: Sequence[int],
    max_side: int,
    iters: int,
    log_every: int,
    early_stop_patience: int,
    early_stop_min_delta: float,
    early_stop_min_iters: int,
    lpips_enabled: bool,
    codec_metrics_enabled: bool,
) -> Path:
    """Write the explicit manifest consumed by the portable root dashboard."""
    expected = len(images) * len(methods) * len(seeds)
    completed = sum(row.get("status") == "ok" for row in rows)
    errors = sum(row.get("status") != "ok" for row in rows)
    exact = sum(
        row.get("status") == "ok" and row.get("storage_exact_budget") is True for row in rows
    )
    plateau = sum(
        row.get("status") == "ok"
        and row.get("convergence_stop_reason") == "psnr_plateau"
        for row in rows
    )
    max_horizon = sum(
        row.get("status") == "ok"
        and row.get("convergence_stop_reason") == "max_horizon"
        for row in rows
    )
    adaptive_stop = sum(
        row.get("status") == "ok"
        and str(row.get("convergence_stop_reason", "")).startswith("adaptive_")
        for row in rows
    )
    lpips_completed = sum(
        row.get("status") == "ok" and row.get("lpips") is not None for row in rows
    )
    codec_completed = sum(
        row.get("status") == "ok" and row.get("actual_codec_bytes") is not None
        for row in rows
    )
    all_methods = tuple(methods) == STORAGE_BENCHMARK_METHODS
    title = (
        "168 KiB all-method convergence benchmark"
        if all_methods
        else f"168 KiB {len(methods)}-method convergence benchmark"
    )
    status = (
        "right_censored"
        if completed + errors == expected and not errors and max_horizon
        else None
    )
    link_candidates = [
        ("Open per-image HTML report", "index.html"),
        ("Image byte sizes", "image_storage.csv"),
        ("Per-image method metrics", "image_metrics.csv"),
        ("Method summary", "storage_method_summary.csv"),
        ("Convergence curves", "convergence_curves.csv"),
        ("Protocol config", "config.json"),
    ]
    manifest = {
        "schema_version": 1,
        "title": title,
        "kind": "matched-policy local analogues",
        "lane": "168_kib",
        "priority": 0,
        "description": (
            f"{len(methods)} fair-harness method{'s' if len(methods) != 1 else ''} at a "
            f"5,376-Gaussian cap, up to {iters:,} "
            f"iterations; after iteration {early_stop_min_iters:,}, plateau stopping requires "
            f"{early_stop_patience} consecutive {log_every}-iteration evaluations without a "
            f"{early_stop_min_delta:g} dB gain. Benchmark side {max_side}px."
        ),
        "report_dir": ".",
        "completeness": {
            "expected_cells": expected,
            "completed_cells": completed,
            "error_cells": errors,
        },
        "coverage": {
            "images": len(images),
            "methods": len(methods),
            "seeds": len(seeds),
            "exact_capacity_cells": exact,
            "plateau_converged_cells": plateau,
            "max_horizon_cells": max_horizon,
            "adaptive_stop_cells": adaptive_stop,
            "lpips_enabled": lpips_enabled,
            "lpips_completed_cells": lpips_completed,
            "codec_metrics_enabled": codec_metrics_enabled,
            "codec_metrics_completed_cells": codec_completed,
        },
        "storage": {
            "budget_bytes": SB.FIXED_STORAGE_BUDGET_BYTES,
            "bytes_per_gaussian": SB.CONSTANT_RGB_RS_BYTES_PER_GAUSSIAN,
            "gaussian_cap": SB.FIXED_STORAGE_GAUSSIAN_CAP,
            "accounting": SB.STORAGE_ACCOUNTING_NAME,
        },
        "protocol_sha256": _protocol_hash(outdir, rows),
        "status": status,
        "caveats": [
            "168 KiB is 172,032 bytes; it is not the SI quantity 168 kB.",
            (
                "The equal-capacity rate is an analytical float32 frozen payload (mean, "
                "log-scales, rotation, RGB), not a checkpoint or codec byte count."
            ),
            (
                "Actual SSPL1 stream bytes and cold-decode quality are separate columns when "
                f"enabled ({codec_metrics_enabled}); source, prepared-target, and reconstruction "
                "file sizes are always reported separately."
            ),
            (
                "These are local matched-policy analogues, not native external repository "
                "pipelines. Capacity-mismatched rows are excluded from strict equal-rate winners."
            ),
            (
                "Early-stopped convergence curves carry the scored reconstruction endpoint to "
                "the nominal horizon, and AUC uses the same hold-last convention. Max-horizon "
                "cells are right-censored rather than silently called converged."
            ),
            (
                "Instant-GI under-allocation is filled with deterministic random target samples "
                "so its storage-lane field has exactly 5,376 Gaussians; native and fill counts "
                "remain visible."
            ),
        ],
        "links": [
            {"label": label, "path": path}
            for label, path in link_candidates
            if (outdir / path).is_file()
        ],
    }
    path = outdir / results_index.MANIFEST_NAME
    write_json(path, manifest)
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run all fair-harness methods at the exact 168 KiB analytical payload cap"
    )
    parser.add_argument("--images", nargs="+", type=Path, default=list(DEFAULT_IMAGES))
    parser.add_argument(
        "--methods", nargs="+", choices=F.DEFAULT_METHODS, default=list(STORAGE_BENCHMARK_METHODS)
    )
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--root-index", type=Path, default=DEFAULT_ROOT_INDEX)
    parser.add_argument(
        "--additional-report",
        action="append",
        type=Path,
        default=list(DEFAULT_ADDITIONAL_REPORTS),
        help="explicit existing report to retain in the unified root index",
    )
    parser.add_argument("--runtime-helper", type=Path, default=None)
    parser.add_argument("--max-side", type=int, default=DEFAULT_MAX_SIDE)
    parser.add_argument("--iters", type=int, default=DEFAULT_ITERS)
    parser.add_argument("--log-every", type=int, default=DEFAULT_LOG_EVERY)
    parser.add_argument("--early-stop-patience", type=int, default=DEFAULT_EARLY_STOP_PATIENCE)
    parser.add_argument("--early-stop-min-delta", type=float, default=DEFAULT_EARLY_STOP_MIN_DELTA)
    parser.add_argument("--early-stop-min-iters", type=int, default=DEFAULT_EARLY_STOP_MIN_ITERS)
    parser.add_argument("--render-chunk", type=int, default=512)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-new-cells", type=int, default=None)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--no-lpips", dest="lpips", action="store_false")
    parser.add_argument(
        "--no-codec-metrics",
        dest="compute_codec_metrics",
        action="store_false",
    )
    parser.set_defaults(resume=True, lpips=True, compute_codec_metrics=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.early_stop_min_iters >= args.iters:
        raise ValueError("early-stop-min-iters must be smaller than iters")
    if "instant_gi_quadtree_fixed" in args.methods and not os.environ.get("STRUCTSPLAT_INSTANT_GI"):
        candidate = Path("/home/alex/Documents/Instant-GI/quard_image.py")
        if candidate.is_file():
            os.environ["STRUCTSPLAT_INSTANT_GI"] = str(candidate)

    harness_args = fair_args(args)
    rows = F.run(harness_args)
    seeds = resolve_seeds(harness_args.seed, harness_args.seeds)
    write_report_manifest(
        Path(args.outdir),
        rows,
        images=harness_args.images,
        methods=harness_args.methods,
        seeds=seeds,
        max_side=harness_args.max_side,
        iters=harness_args.iters,
        log_every=harness_args.log_every,
        early_stop_patience=harness_args.early_stop_patience,
        early_stop_min_delta=harness_args.early_stop_min_delta,
        early_stop_min_iters=harness_args.early_stop_min_iters,
        lpips_enabled=harness_args.lpips,
        codec_metrics_enabled=harness_args.compute_codec_metrics,
    )
    report_specs = [Path(args.outdir)]
    report_specs.extend(
        path for path in args.additional_report if path.exists() and path not in report_specs
    )
    results_index.build_results_index(
        report_specs,
        output=Path(args.root_index),
        runtime_helper=args.runtime_helper,
    )


if __name__ == "__main__":
    main()
