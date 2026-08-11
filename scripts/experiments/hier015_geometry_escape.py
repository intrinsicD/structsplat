#!/usr/bin/env python3
"""Run HIER-015's exact-7k geometry-escape/direct-fit development diagnostic.

Development screen::

    PYTHONPATH=src python scripts/experiments/hier015_geometry_escape.py \
      --phase development --images /home/alex/Documents/datasets/train2014 \
      --out results/hier015_coco_geometry_escape_2026-08-10

Consumed-bank replay is allowed only after the development disposition and visual review are
frozen.  Its exact invocation is emitted by the development report.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import csv
import hashlib
from html import escape
import json
import math
import platform
from pathlib import Path
import shlex
import shutil
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
for candidate in (ROOT, ROOT / "src"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from scripts.experiments import hier005_pixel_contraction as report_utils  # noqa: E402
from scripts.experiments import hier010_residual_anchor_projection as viz_utils  # noqa: E402
from scripts.experiments import hier013_global_projection_development as h13  # noqa: E402
from structsplat.cli import save_error_heatmap, save_image  # noqa: E402
from structsplat.config import FitConfig, InitConfig, StructureTensorConfig  # noqa: E402
from structsplat.contraction_refinement import (  # noqa: E402
    AlternatingGeometryConfig,
    AlternatingGeometryResult,
    CoefficientProjectionConfig,
    CoefficientProjectionResult,
    GeometryRelaxationConfig,
    alternate_projected_geometry,
    project_contracted_coefficients,
)
from structsplat.gaussians import GaussianField  # noqa: E402
from structsplat.init import build_field  # noqa: E402
from structsplat.observation_field import ObservationField2D  # noqa: E402
from structsplat.pixel_contraction import (  # noqa: E402
    PixelContractionConfig,
    contract_image,
    render_observation_field,
)


REPORT_SCHEMA = "structsplat.hier015_geometry_escape.diagnostic.v1"
DEVELOPMENT_BINDINGS = {
    "COCO_train2014_000000371955.jpg": (
        "24c86916356edf9c00c17d74cd4f767f5e3fc33f1e5b56b239c05e914d87dfff"
    ),
    "COCO_train2014_000000012379.jpg": (
        "82fa9d25824b7dd43480b4f64651d3106a91f3b8f7e6d474da221733d289ca90"
    ),
    "COCO_train2014_000000090218.jpg": (
        "7789b17db08cd18831f615bafa0abf2a602297a6554810ce1c133a214d921c90"
    ),
    "COCO_train2014_000000237851.jpg": (
        "05451ba10a92a5009889773abda7c042b254c0f2f54d7b1bab9b158154e8172b"
    ),
}
DEVELOPMENT_ARMS = (
    "h005_control",
    "conditioned_transaction",
    "relax_1x400",
    "relax_2x200",
    "direct_normalized_fixed7k",
)
HIERARCHY_ARMS = ("relax_1x400", "relax_2x200")
DIRECT_ARM = "direct_normalized_fixed7k"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("development", "replay"), required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--disposition",
        choices=("relax_1x400", "relax_2x200", "direct_normalized_fixed7k"),
        help="frozen Phase-B disposition; required only for replay",
    )
    parser.add_argument("--development-decision", type=Path)
    parser.add_argument("--target-gaussians", type=int, default=7000)
    parser.add_argument("--max-side", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--projection-max-iterations", type=int, default=96)
    parser.add_argument("--coefficient-limit", type=float, default=16.0)
    parser.add_argument("--geometry-steps", type=int, default=400)
    parser.add_argument("--direct-fit-steps", type=int, default=750)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--additive-renderer", default="cuda_additive")
    parser.add_argument("--direct-renderer", default="cuda")
    parser.add_argument("--render-chunk", type=int, default=256)
    parser.add_argument("--lpips", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--error-scale", type=float, default=4.0)
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    frozen = {
        "target_gaussians": 7000,
        "max_side": 512,
        "seed": 0,
        "projection_max_iterations": 96,
        "coefficient_limit": 16.0,
        "geometry_steps": 400,
        "direct_fit_steps": 750,
        "device": "cuda",
        "additive_renderer": "cuda_additive",
        "direct_renderer": "cuda",
        "render_chunk": 256,
        "lpips": True,
    }
    for name, expected in frozen.items():
        if getattr(args, name) != expected:
            option = name.replace("_", "-")
            raise SystemExit(f"frozen HIER-015 protocol requires --{option} {expected}")
    if not math.isfinite(args.error_scale) or args.error_scale <= 0.0:
        raise SystemExit("--error-scale must be finite and positive")
    if args.phase == "development":
        if args.disposition is not None or args.development_decision is not None:
            raise SystemExit("development does not accept replay disposition arguments")
    elif args.disposition is None or args.development_decision is None:
        raise SystemExit("replay requires --disposition and --development-decision")


def _discover_images(args: argparse.Namespace) -> list[Path]:
    if args.phase == "development":
        paths = [args.images / name for name in DEVELOPMENT_BINDINGS]
        actual = {
            path.name: report_utils._sha256(path)
            for path in paths
            if path.is_file()
        }
        if actual != DEVELOPMENT_BINDINGS:
            raise SystemExit(
                "HIER-015 development source bank is missing or hash-mismatched: "
                f"expected {DEVELOPMENT_BINDINGS}, got {actual}"
            )
        return [path.resolve() for path in paths]
    decision = json.loads(args.development_decision.read_text(encoding="utf-8"))
    if decision.get("schema") != REPORT_SCHEMA or decision.get("phase") != "development":
        raise SystemExit("--development-decision is not a HIER-015 development decision")
    candidates = decision.get("numeric_candidates", [])
    if args.disposition not in candidates:
        raise SystemExit(
            f"replay disposition {args.disposition!r} is not an eligible numeric candidate: "
            f"{candidates}"
        )
    return h13._discover_sources([args.images])


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
        recovery_renderer=args.additive_renderer,
        recovery_render_chunk=args.render_chunk,
        recovery_lr_means=0.005,
        recovery_lr_scales=0.003,
        recovery_lr_rotations=0.001,
        recovery_lr_coefficients=0.003,
        recovery_max_mean_shift_px=1.5,
        recovery_max_log_scale_shift=0.35,
        recovery_max_rotation_shift_rad=0.35,
    )


def _projection_config(args: argparse.Namespace, *, intermediate: bool) -> CoefficientProjectionConfig:
    return CoefficientProjectionConfig(
        tolerance=1e-6,
        max_iterations=args.projection_max_iterations,
        ridge=1e-8,
        coefficient_abs_limit=args.coefficient_limit,
        regularization_center="zero",
        solver_start="zero",
        frozen_base_mode="explicit",
        allow_unsafe_stage_zero_reconditioning=True,
        selection_mode="bounded_intermediate" if intermediate else "transaction",
    )


def _alternating_config(args: argparse.Namespace, arm: str) -> AlternatingGeometryConfig:
    if arm == "relax_1x400":
        rounds, steps = 1, args.geometry_steps
    elif arm == "relax_2x200":
        rounds, steps = 2, args.geometry_steps // 2
    else:
        raise ValueError(f"unknown alternating arm {arm}")
    return AlternatingGeometryConfig(
        rounds=rounds,
        geometry=GeometryRelaxationConfig(
            steps=steps,
            checkpoint_every=25,
            lr_means=0.01,
            lr_scales=0.006,
            lr_rotations=0.002,
            max_mean_shift_px=4.0,
            max_log_scale_shift=0.7,
            max_rotation_shift_rad=0.7,
        ),
        projection=_projection_config(args, intermediate=True),
    )


def _direct_configs(args: argparse.Namespace) -> tuple[InitConfig, FitConfig]:
    return (
        InitConfig(
            strategy="aniso_onedge",
            num_gaussians=args.target_gaussians,
            seed=args.seed,
            sampling_mode="wse",
            flank_offset_frac=0.0,
            scale_cap_mode="feature",
            scale_cap_max=38.4,
        ),
        FitConfig(
            iters=args.direct_fit_steps,
            renderer=args.direct_renderer,
            render_chunk=args.render_chunk,
            pixel_loss="l1",
            ssim_weight=0.3,
            checkpoint_policy="best_psnr_final_count",
            compute_lpips=False,
            log_every=25,
            max_gaussians=args.target_gaussians,
        ),
    )


def _snapshot_sources(output_root: Path) -> list[dict[str, object]]:
    paths = (
        Path(__file__).resolve(),
        ROOT / "src" / "structsplat" / "contraction_refinement.py",
        ROOT / "src" / "structsplat" / "pixel_contraction.py",
        ROOT / "src" / "structsplat" / "fit.py",
        ROOT / "src" / "structsplat" / "init.py",
        ROOT / "tasks" / "HIER-015-geometry-escape-direct-fit.md",
    )
    records: list[dict[str, object]] = []
    for source in paths:
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


def _gaussian_content_hash(field: GaussianField) -> str:
    digest = hashlib.sha256()
    arrays = {
        "colors": field.colors.detach().cpu().numpy(),
        "log_scales": field.log_scales.detach().cpu().numpy(),
        "means": field.means.detach().cpu().numpy(),
        "rotations": field.rotations.detach().cpu().numpy(),
    }
    for name, array in sorted(arrays.items()):
        contiguous = np.ascontiguousarray(array)
        digest.update(name.encode("utf-8"))
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _geometry_delta(field: ObservationField2D, control: ObservationField2D) -> dict[str, object]:
    mean_shift = np.linalg.norm(
        field.means_xy.astype(np.float64) - control.means_xy.astype(np.float64), axis=1
    )
    scale_shift = np.abs(
        field.log_scales_xy.astype(np.float64) - control.log_scales_xy.astype(np.float64)
    )
    rotation_shift = np.abs(
        field.rotations_rad.astype(np.float64) - control.rotations_rad.astype(np.float64)
    )
    return {
        "geometry_changed": bool(
            np.any(mean_shift > 0.0)
            or np.any(scale_shift > 0.0)
            or np.any(rotation_shift > 0.0)
        ),
        "mean_shift_max_px": float(np.max(mean_shift)),
        "mean_shift_median_px": float(np.median(mean_shift)),
        "log_scale_shift_max": float(np.max(scale_shift)),
        "rotation_shift_max_rad": float(np.max(rotation_shift)),
    }


def _save_visuals(
    artifact_dir: Path,
    image: np.ndarray,
    reconstruction: np.ndarray,
    control: np.ndarray,
    mask: np.ndarray,
    error_scale: float,
) -> tuple[int, int, int, int]:
    save_image(str(artifact_dir / "source.png"), image)
    save_image(str(artifact_dir / "control.png"), control)
    save_image(str(artifact_dir / "reconstruction.png"), reconstruction)
    save_error_heatmap(
        str(artifact_dir / "error.png"), reconstruction - image, scale=error_scale
    )
    bounds = viz_utils._worst_crop_bounds(reconstruction, image, mask)
    viz_utils._save_crop(artifact_dir / "source_crop.png", image, bounds)
    viz_utils._save_crop(artifact_dir / "reconstruction_crop.png", reconstruction, bounds)
    shown_error = np.repeat(
        np.clip(
            np.mean(np.abs(reconstruction - image), axis=2) * error_scale,
            0.0,
            1.0,
        )[:, :, None],
        3,
        axis=2,
    )
    viz_utils._save_crop(artifact_dir / "error_crop.png", shown_error, bounds)
    return bounds


def _write_observation_cell(
    *,
    output_root: Path,
    image_path: Path,
    image: np.ndarray,
    mask: np.ndarray,
    raster: dict[str, object],
    arm: str,
    field: ObservationField2D,
    control_field: ObservationField2D,
    control_reconstruction: np.ndarray,
    expected: np.ndarray,
    contraction_seconds: float,
    method_seconds: float,
    projection: CoefficientProjectionResult | None,
    alternating: AlternatingGeometryResult | None,
    peak_cuda_bytes: int,
    args: argparse.Namespace,
    schema: str = REPORT_SCHEMA,
    extra_row: dict[str, object] | None = None,
) -> dict[str, object]:
    artifact_dir = output_root / "artifacts" / f"{image_path.stem}__{arm}__n7000"
    artifact_dir.mkdir(parents=True, exist_ok=False)
    field_path = artifact_dir / "field.observation.npz"
    field.save_lossless(field_path)
    decode_started = time.perf_counter()
    cold_field = ObservationField2D.load_lossless(field_path)
    decode_seconds = time.perf_counter() - decode_started
    render_started = time.perf_counter()
    cold = render_observation_field(
        cold_field,
        device=args.device,
        renderer=args.additive_renderer,
        render_chunk=args.render_chunk,
    )
    render_seconds = time.perf_counter() - render_started
    repeated = render_observation_field(
        cold_field,
        device=args.device,
        renderer=args.additive_renderer,
        render_chunk=args.render_chunk,
    )
    metric_started = time.perf_counter()
    metrics = report_utils._metric_values(
        cold, image, mask, device=args.device, compute_lpips=args.lpips
    )
    metric_seconds = time.perf_counter() - metric_started
    bounds = _save_visuals(
        artifact_dir,
        image,
        cold,
        control_reconstruction,
        mask,
        args.error_scale,
    )
    projection_history: object = []
    geometry_history: object = []
    selected_projection_iteration = 0
    selected_geometry_steps = 0
    transaction_safe = True
    forward_calls = 0
    transpose_calls = 0
    if projection is not None:
        projection_history = projection.checkpoint_records()
        selected_projection_iteration = projection.selected_iteration
        selected_checkpoint = next(
            checkpoint for checkpoint in projection.checkpoints if checkpoint.selected
        )
        transaction_safe = selected_checkpoint.transaction_safe
        forward_calls = projection.forward_applications
        transpose_calls = projection.transpose_applications
    if alternating is not None:
        projection_history = [
            result.checkpoint_records() for result in alternating.projection_results
        ]
        geometry_history = {
            "outer": alternating.checkpoint_records(),
            "blocks": [
                result.checkpoint_records() for result in alternating.geometry_results
            ],
        }
        selected_projection_iteration = next(
            checkpoint.projection_iteration
            for checkpoint in alternating.checkpoints
            if checkpoint.selected
        )
        selected_geometry_steps = alternating.total_geometry_steps
        transaction_safe = next(
            checkpoint.transaction_safe
            for checkpoint in alternating.checkpoints
            if checkpoint.selected
        )
        forward_calls = alternating.forward_applications
        transpose_calls = alternating.transpose_applications
    report_utils._write_json(artifact_dir / "projection_history.json", projection_history)
    report_utils._write_json(artifact_dir / "geometry_history.json", geometry_history)
    np.savez_compressed(
        artifact_dir / "analysis.npz",
        crop_bounds=np.asarray(bounds, dtype=np.int32),
        mask=mask,
        coefficient_abs=np.abs(cold_field.rgb_coeff),
        mean_shift=np.linalg.norm(
            cold_field.means_xy.astype(np.float64)
            - control_field.means_xy.astype(np.float64),
            axis=1,
        ),
    )
    coefficient_abs_max = float(np.max(np.abs(cold_field.rgb_coeff)))
    geometry = _geometry_delta(cold_field, control_field)
    row: dict[str, object] = {
        "schema": schema,
        "status": "diagnostic",
        "phase": args.phase,
        "image": image_path.stem,
        "arm": arm,
        "semantic_family": "additive_rgb_peak_one_v1",
        "artifact_dir": str(artifact_dir.relative_to(output_root)),
        "source_path": str(image_path),
        "source_sha256": report_utils._sha256(image_path),
        "source_file_bytes": image_path.stat().st_size,
        "original_width": raster["original_width"],
        "original_height": raster["original_height"],
        "width": image.shape[1],
        "height": image.shape[0],
        "active_pixels": int(mask.sum()),
        "seed": args.seed,
        "n_gaussians": cold_field.n,
        "target_gaussians": args.target_gaussians,
        "field_content_sha256": cold_field.canonical_hash(),
        "field_file_sha256": report_utils._sha256(field_path),
        "non_rgb_arrays_bit_exact": h13._non_rgb_equal(cold_field, control_field),
        "coefficient_abs_max": coefficient_abs_max,
        "coefficient_abs_median": float(np.median(np.abs(cold_field.rgb_coeff))),
        "coefficient_abs_q99": float(np.quantile(np.abs(cold_field.rgb_coeff), 0.99)),
        "contraction_seconds": contraction_seconds,
        "method_seconds": method_seconds,
        "pipeline_algorithm_seconds": contraction_seconds + method_seconds,
        "method_overhead_ratio": method_seconds / max(contraction_seconds, 1e-12),
        "fit_seconds": 0.0,
        "cold_decode_seconds": decode_seconds,
        "render_seconds": render_seconds,
        "metric_seconds": metric_seconds,
        "peak_cuda_allocated_bytes": peak_cuda_bytes,
        "lossless_reference_bytes": field_path.stat().st_size,
        "maintained_render_parity_max_abs": float(np.max(np.abs(cold - expected))),
        "repeated_render_parity_max_abs": float(np.max(np.abs(repeated - cold))),
        "selected_projection_iteration": selected_projection_iteration,
        "selected_geometry_steps": selected_geometry_steps,
        "selected_transaction_safe": transaction_safe,
        "projection_forward_applications": forward_calls,
        "projection_transpose_applications": transpose_calls,
        **geometry,
        **metrics,
    }
    if extra_row is not None:
        row.update(extra_row)
    report_utils._write_json(artifact_dir / "row.json", row)
    return row


def _write_direct_cell(
    *,
    output_root: Path,
    image_path: Path,
    image: np.ndarray,
    mask: np.ndarray,
    raster: dict[str, object],
    field: GaussianField,
    fit_result: dict[str, object],
    init_seconds: float,
    control_reconstruction: np.ndarray,
    peak_cuda_bytes: int,
    fit_config: FitConfig,
    args: argparse.Namespace,
    arm: str = DIRECT_ARM,
    schema: str = REPORT_SCHEMA,
    extra_row: dict[str, object] | None = None,
) -> dict[str, object]:
    import torch

    from structsplat.fit import _render

    artifact_dir = output_root / "artifacts" / f"{image_path.stem}__{arm}__n7000"
    artifact_dir.mkdir(parents=True, exist_ok=False)
    field_path = artifact_dir / "field.gaussian.npz"
    field.save(str(field_path))
    decode_started = time.perf_counter()
    cold_field = GaussianField.load(str(field_path), device=args.device)
    decode_seconds = time.perf_counter() - decode_started
    render_started = time.perf_counter()
    with torch.no_grad():
        cold_tensor = _render(cold_field, fit_config, image.shape[0], image.shape[1])
        repeated_tensor = _render(cold_field, fit_config, image.shape[0], image.shape[1])
    render_seconds = time.perf_counter() - render_started
    cold = cold_tensor.detach().cpu().numpy().astype(np.float32, copy=False)
    repeated = repeated_tensor.detach().cpu().numpy().astype(np.float32, copy=False)
    expected = fit_result["render"].detach().cpu().numpy().astype(np.float32, copy=False)
    metric_started = time.perf_counter()
    metrics = report_utils._metric_values(
        cold, image, mask, device=args.device, compute_lpips=args.lpips
    )
    metric_seconds = time.perf_counter() - metric_started
    bounds = _save_visuals(
        artifact_dir,
        image,
        cold,
        control_reconstruction,
        mask,
        args.error_scale,
    )
    history = fit_result["history"]
    report_utils._write_json(artifact_dir / "fit_history.json", history)
    report_utils._write_json(artifact_dir / "projection_history.json", [])
    report_utils._write_json(artifact_dir / "geometry_history.json", [])
    colors = cold_field.colors.detach().cpu().numpy()
    np.savez_compressed(
        artifact_dir / "analysis.npz",
        crop_bounds=np.asarray(bounds, dtype=np.int32),
        mask=mask,
        coefficient_abs=np.abs(colors),
    )
    fit_seconds = float(fit_result["fit_seconds"])
    row: dict[str, object] = {
        "schema": schema,
        "status": "diagnostic",
        "phase": args.phase,
        "image": image_path.stem,
        "arm": arm,
        "semantic_family": "normalized_weighted_sum_v1",
        "artifact_dir": str(artifact_dir.relative_to(output_root)),
        "source_path": str(image_path),
        "source_sha256": report_utils._sha256(image_path),
        "source_file_bytes": image_path.stat().st_size,
        "original_width": raster["original_width"],
        "original_height": raster["original_height"],
        "width": image.shape[1],
        "height": image.shape[0],
        "active_pixels": int(mask.sum()),
        "seed": args.seed,
        "n_gaussians": cold_field.n,
        "target_gaussians": args.target_gaussians,
        "field_content_sha256": _gaussian_content_hash(cold_field),
        "field_file_sha256": report_utils._sha256(field_path),
        "non_rgb_arrays_bit_exact": None,
        "coefficient_abs_max": float(np.max(np.abs(colors))),
        "coefficient_abs_median": float(np.median(np.abs(colors))),
        "coefficient_abs_q99": float(np.quantile(np.abs(colors), 0.99)),
        "contraction_seconds": 0.0,
        "method_seconds": init_seconds + fit_seconds,
        "pipeline_algorithm_seconds": init_seconds + fit_seconds,
        "method_overhead_ratio": None,
        "initialization_seconds": init_seconds,
        "fit_seconds": fit_seconds,
        "iterations_run": int(fit_result["iterations_run"]),
        "selected_iteration": int(fit_result["selected_iter"]),
        "cold_decode_seconds": decode_seconds,
        "render_seconds": render_seconds,
        "metric_seconds": metric_seconds,
        "peak_cuda_allocated_bytes": peak_cuda_bytes,
        "lossless_reference_bytes": field_path.stat().st_size,
        "maintained_render_parity_max_abs": float(np.max(np.abs(cold - expected))),
        "repeated_render_parity_max_abs": float(np.max(np.abs(repeated - cold))),
        "selected_projection_iteration": 0,
        "selected_geometry_steps": int(fit_result["iterations_run"]),
        "selected_transaction_safe": True,
        "projection_forward_applications": 0,
        "projection_transpose_applications": 0,
        "geometry_changed": True,
        "mean_shift_max_px": None,
        "mean_shift_median_px": None,
        "log_scale_shift_max": None,
        "rotation_shift_max_rad": None,
        **metrics,
    }
    if extra_row is not None:
        row.update(extra_row)
    report_utils._write_json(artifact_dir / "row.json", row)
    return row


def _paired(rows: list[dict[str, object]], arm: str) -> list[dict[str, object]]:
    controls = {str(row["image"]): row for row in rows if row["arm"] == "h005_control"}
    pairs: list[dict[str, object]] = []
    for row in rows:
        if row["arm"] != arm:
            continue
        control = controls.get(str(row["image"]))
        if control is None:
            continue
        pairs.append(
            {
                "image": row["image"],
                "mse_ratio": float(row["masked_mse"]) / float(control["masked_mse"]),
                "psnr_delta_db": float(row["psnr_db"]) - float(control["psnr_db"]),
                "ms_ssim_delta": float(row["ms_ssim"]) - float(control["ms_ssim"]),
                "lpips_delta": float(row["lpips"]) - float(control["lpips"]),
                "pixel_max_delta": float(row["artifact_pixel_rmse_max"])
                - float(control["artifact_pixel_rmse_max"]),
                "patch7_max_delta": float(row["artifact_patch_rmse_max_7"])
                - float(control["artifact_patch_rmse_max_7"]),
                "n_gaussians": row["n_gaussians"],
                "coefficient_abs_max": row["coefficient_abs_max"],
                "geometry_changed": row["geometry_changed"],
                "selected_transaction_safe": row["selected_transaction_safe"],
                "maintained_render_parity_max_abs": row[
                    "maintained_render_parity_max_abs"
                ],
                "repeated_render_parity_max_abs": row[
                    "repeated_render_parity_max_abs"
                ],
                "method_overhead_ratio": row["method_overhead_ratio"],
            }
        )
    return pairs


def _arm_aggregate(pairs: list[dict[str, object]]) -> dict[str, object]:
    if not pairs:
        return {
            "pairs": [],
            "pair_count": 0,
            "geometric_mean_mse_ratio": 1.0,
            "mean_psnr_delta_db": 0.0,
            "mean_ms_ssim_delta": 0.0,
            "mean_lpips_delta": 0.0,
            "maximum_pixel_max_delta": 0.0,
            "maximum_patch7_max_delta": 0.0,
        }
    ratios = np.asarray([float(pair["mse_ratio"]) for pair in pairs], dtype=np.float64)
    return {
        "pairs": pairs,
        "pair_count": len(pairs),
        "geometric_mean_mse_ratio": float(np.exp(np.mean(np.log(ratios)))),
        "mean_psnr_delta_db": float(np.mean([pair["psnr_delta_db"] for pair in pairs])),
        "mean_ms_ssim_delta": float(np.mean([pair["ms_ssim_delta"] for pair in pairs])),
        "mean_lpips_delta": float(np.mean([pair["lpips_delta"] for pair in pairs])),
        "maximum_pixel_max_delta": max(float(pair["pixel_max_delta"]) for pair in pairs),
        "maximum_patch7_max_delta": max(float(pair["patch7_max_delta"]) for pair in pairs),
    }


def _aggregate(
    rows: list[dict[str, object]],
    args: argparse.Namespace,
    attempts: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    attempt_records = attempts or []
    failures = [record for record in attempt_records if record.get("status") != "ok"]
    if args.phase == "replay":
        expected_rows = 16 if args.disposition == DIRECT_ARM else 32
        return {
            "schema": REPORT_SCHEMA,
            "status": "diagnostic",
            "phase": "replay",
            "disposition": args.disposition,
            "row_count": len(rows),
            "complete": len(rows) == expected_rows,
            "attempt_count": len(attempt_records),
            "failure_count": len(failures),
            "all_exact_count": all(int(row["n_gaussians"]) == 7000 for row in rows),
            "all_finite": all(
                math.isfinite(float(row[metric]))
                for row in rows
                for metric in ("masked_mse", "psnr_db", "ms_ssim", "lpips")
            ),
            "interpretation": "Consumed-bank reporting replay; no retuning or held-out claim.",
        }
    aggregates = {
        arm: _arm_aggregate(_paired(rows, arm)) for arm in DEVELOPMENT_ARMS[1:]
    }
    hierarchy_gates: dict[str, dict[str, bool]] = {}
    for arm in HIERARCHY_ARMS:
        aggregate = aggregates[arm]
        pairs = aggregate["pairs"]
        hierarchy_gates[arm] = {
            "complete_four_pairs": int(aggregate["pair_count"]) == 4,
            "all_finite": all(
                math.isfinite(float(pair[key]))
                for pair in pairs
                for key in (
                    "mse_ratio",
                    "psnr_delta_db",
                    "ms_ssim_delta",
                    "lpips_delta",
                    "pixel_max_delta",
                    "patch7_max_delta",
                    "coefficient_abs_max",
                )
            ),
            "all_exact_count": all(int(pair["n_gaussians"]) == 7000 for pair in pairs),
            "all_coefficients_bounded": all(
                float(pair["coefficient_abs_max"]) <= 16.0 for pair in pairs
            ),
            "all_final_transactions_safe": all(
                bool(pair["selected_transaction_safe"]) for pair in pairs
            ),
            "all_geometry_moved": all(bool(pair["geometry_changed"]) for pair in pairs),
            "all_parity_le_2e_5": all(
                float(pair["maintained_render_parity_max_abs"]) <= 2e-5
                and float(pair["repeated_render_parity_max_abs"]) <= 2e-5
                for pair in pairs
            ),
            "all_mse_noninferior": all(
                float(pair["mse_ratio"]) <= 1.0 + 1e-8 for pair in pairs
            ),
            "all_pixel_max_noninferior": all(
                float(pair["pixel_max_delta"]) <= 1e-12 for pair in pairs
            ),
            "all_patch7_max_noninferior": all(
                float(pair["patch7_max_delta"]) <= 1e-12 for pair in pairs
            ),
            "geometric_mean_mse_ratio_le_0_80": (
                float(aggregate["geometric_mean_mse_ratio"]) <= 0.80
            ),
            "mean_ms_ssim_noninferior": float(aggregate["mean_ms_ssim_delta"]) >= -1e-7,
            "mean_lpips_noninferior": float(aggregate["mean_lpips_delta"]) <= 1e-7,
            "median_overhead_le_0_50": float(
                np.median([float(pair["method_overhead_ratio"]) for pair in pairs])
            )
            <= 0.50,
        }
    direct = aggregates[DIRECT_ARM]
    direct_pairs = direct["pairs"]
    direct_gate = {
        "complete_four_pairs": int(direct["pair_count"]) == 4,
        "all_exact_count": all(int(pair["n_gaussians"]) == 7000 for pair in direct_pairs),
        "all_finite": all(
            math.isfinite(float(pair[key]))
            for pair in direct_pairs
            for key in (
                "mse_ratio",
                "psnr_delta_db",
                "ms_ssim_delta",
                "lpips_delta",
                "pixel_max_delta",
                "patch7_max_delta",
            )
        ),
        "all_parity_le_2e_5": all(
            float(pair["maintained_render_parity_max_abs"]) <= 2e-5
            and float(pair["repeated_render_parity_max_abs"]) <= 2e-5
            for pair in direct_pairs
        ),
        "all_psnr_gain_ge_2_db": all(
            float(pair["psnr_delta_db"]) >= 2.0 for pair in direct_pairs
        ),
        "mean_ms_ssim_noninferior": float(direct["mean_ms_ssim_delta"]) >= -1e-7,
        "mean_lpips_noninferior": float(direct["mean_lpips_delta"]) <= 1e-7,
        "all_pixel_max_noninferior": all(
            float(pair["pixel_max_delta"]) <= 1e-12 for pair in direct_pairs
        ),
        "all_patch7_max_noninferior": all(
            float(pair["patch7_max_delta"]) <= 1e-12 for pair in direct_pairs
        ),
    }
    passing_hierarchy = [arm for arm in HIERARCHY_ARMS if all(hierarchy_gates[arm].values())]
    numeric_candidates: list[str] = []
    if passing_hierarchy:
        numeric_candidates.append(
            min(
                passing_hierarchy,
                key=lambda arm: float(aggregates[arm]["geometric_mean_mse_ratio"]),
            )
        )
    if all(direct_gate.values()):
        numeric_candidates.append(DIRECT_ARM)
    numeric_disposition = numeric_candidates[0] if numeric_candidates else "no_robust_7k_candidate"
    return {
        "schema": REPORT_SCHEMA,
        "status": "diagnostic",
        "phase": "development",
        "arm_aggregates": aggregates,
        "hierarchy_gates": hierarchy_gates,
        "direct_gate": direct_gate,
        "numeric_candidates": numeric_candidates,
        "numeric_disposition": numeric_disposition,
        "attempt_count": len(attempt_records),
        "failure_count": len(failures),
        "visual_review_required": True,
        "interpretation": (
            "Numeric candidate(s) must pass frozen full-frame/worst-crop visual review before replay."
            if numeric_candidates
            else "No arm clears its frozen numeric gate; do not tune this source bank."
        ),
    }


def _write_tables(
    output_root: Path,
    rows: list[dict[str, object]],
    *,
    schema: str = REPORT_SCHEMA,
) -> None:
    report_utils._write_json(
        output_root / "metrics.json",
        {"schema": schema, "status": "diagnostic", "rows": rows},
    )
    with (output_root / "metrics.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    columns = sorted({key for row in rows for key in row})
    with (output_root / "metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        if not columns:
            return
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _write_report(
    output_root: Path,
    rows: list[dict[str, object]],
    decision: dict[str, object],
    command: str,
) -> None:
    table_rows: list[str] = []
    for row in rows:
        table_rows.append(
            "<tr>"
            f"<td>{escape(str(row['image']))}</td><td>{escape(str(row['arm']))}</td>"
            f"<td>{escape(str(row['semantic_family']))}</td>"
            f"<td>{int(row['n_gaussians'])}</td><td>{float(row['psnr_db']):.3f}</td>"
            f"<td>{float(row['ms_ssim']):.5f}</td><td>{float(row['lpips']):.5f}</td>"
            f"<td>{float(row['artifact_pixel_rmse_max']):.4f}</td>"
            f"<td>{float(row['artifact_patch_rmse_max_7']):.4f}</td>"
            f"<td>{float(row['coefficient_abs_max']):.2f}</td>"
            f"<td>{float(row['pipeline_algorithm_seconds']):.1f}</td>"
            f"<td><a href='{escape(str(row['artifact_dir']))}/reconstruction.png'>full</a> · "
            f"<a href='{escape(str(row['artifact_dir']))}/reconstruction_crop.png'>crop</a> · "
            f"<a href='{escape(str(row['artifact_dir']))}/error.png'>error</a></td></tr>"
        )
    cards: list[str] = []
    for row in rows:
        artifact = escape(str(row["artifact_dir"]))
        cards.append(
            f"<section><h3>{escape(str(row['image']))} — {escape(str(row['arm']))}</h3>"
            f"<a href='{artifact}/source.png'><img src='{artifact}/source.png'></a>"
            f"<a href='{artifact}/reconstruction.png'><img src='{artifact}/reconstruction.png'></a>"
            f"<a href='{artifact}/error.png'><img src='{artifact}/error.png'></a>"
            f"<a href='{artifact}/reconstruction_crop.png'><img src='{artifact}/reconstruction_crop.png'></a>"
            "</section>"
        )
    decision_text = escape(json.dumps(decision, indent=2, sort_keys=True))
    html = f"""<!doctype html><html><head><meta charset='utf-8'>
<title>HIER-015 geometry escape</title><style>
body{{font-family:system-ui;margin:2rem;max-width:1600px}}table{{border-collapse:collapse}}
th,td{{border:1px solid #bbb;padding:.35rem}}img{{max-width:300px;max-height:240px;margin:.25rem}}
section{{border-top:1px solid #aaa;margin-top:1rem}}pre,code{{white-space:pre-wrap}}
</style></head><body><h1>HIER-015 exact-7k geometry escape — {escape(str(decision['phase']))}</h1>
<p>Development/reporting-only evidence. Additive and normalized arms have explicitly different
renderer semantics. Visual review is mandatory before any replay disposition.</p>
<p><code>{escape(command)}</code></p>
<p><a href='config.json'>config</a> · <a href='decision.json'>decision</a> ·
<a href='metrics.json'>JSON</a> · <a href='metrics.jsonl'>JSONL</a> ·
<a href='metrics.csv'>CSV</a> · <a href='attempts.json'>attempts</a> ·
<a href='manifest.json'>manifest</a></p>
<h2>Decision</h2><pre>{decision_text}</pre>
<h2>Cells</h2><table><tr><th>image</th><th>arm</th><th>semantics</th><th>N</th>
<th>PSNR</th><th>MS-SSIM</th><th>LPIPS</th><th>pixel max</th><th>7x7 max</th>
<th>|c|max</th><th>algorithm s</th><th>artifacts</th></tr>{''.join(table_rows)}</table>
<h2>Visual audit</h2>{''.join(cards)}</body></html>"""
    (output_root / "index.html").write_text(html, encoding="utf-8")


def _write_manifest(output_root: Path, *, schema: str = REPORT_SCHEMA) -> None:
    files: list[dict[str, object]] = []
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
        {"schema": schema, "status": "diagnostic", "files": files},
    )


def _run_direct(
    image: np.ndarray,
    args: argparse.Namespace,
) -> tuple[GaussianField, dict[str, object], float, FitConfig, int]:
    import torch

    from structsplat.fit import fit

    init_config, fit_config = _direct_configs(args)
    torch.cuda.reset_peak_memory_stats()
    init_started = time.perf_counter()
    field = build_field(
        image, init_config, StructureTensorConfig(), device=args.device
    )
    init_seconds = time.perf_counter() - init_started
    target = torch.as_tensor(image, device=args.device, dtype=torch.float32).contiguous()
    result = fit(field, target, fit_config, verbose=False)
    peak = int(torch.cuda.max_memory_allocated())
    return result["field"], result, init_seconds, fit_config, peak


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    _validate_args(args)
    images = _discover_images(args)
    output_root = args.out.resolve()
    output_root.mkdir(parents=True, exist_ok=False)

    import torch

    command = shlex.join([sys.executable, str(Path(__file__).resolve()), *(argv or sys.argv[1:])])
    contraction_config = _contraction_config(args)
    transaction_config = _projection_config(args, intermediate=False)
    alternating_configs = {arm: _alternating_config(args, arm) for arm in HIERARCHY_ARMS}
    init_config, fit_config = _direct_configs(args)
    arms = DEVELOPMENT_ARMS
    if args.phase == "replay":
        arms = (
            (DIRECT_ARM,)
            if args.disposition == DIRECT_ARM
            else ("h005_control", str(args.disposition))
        )
    source_snapshots = _snapshot_sources(output_root)
    config = {
        "schema": REPORT_SCHEMA,
        "status": "diagnostic",
        "phase": args.phase,
        "command": command,
        "args": vars(args),
        "arms": list(arms),
        "sources": [
            {"path": str(path), "sha256": report_utils._sha256(path)} for path in images
        ],
        "contraction": asdict(contraction_config),
        "conditioned_transaction": asdict(transaction_config),
        "alternating": {arm: asdict(cfg) for arm, cfg in alternating_configs.items()},
        "direct_init": asdict(init_config),
        "direct_fit": asdict(fit_config),
        "source_snapshots": source_snapshots,
        "git": report_utils._git_record(),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        },
        "limitations": [
            "Dirty-source, one-seed diagnostic without distinct prospective review.",
            "Additive and normalized arms are explicit different-semantic controls.",
            "CUDA atomic accumulation is numerically, not bit, reproducible.",
            "Lossless NPZ artifacts are reference persistence, not production codec rate.",
        ],
    }
    report_utils._write_json(output_root / "config.json", config)

    rows: list[dict[str, object]] = []
    attempts: list[dict[str, object]] = []

    def record_attempt(
        image_path: Path,
        arm: str,
        started: float,
        error: Exception | None = None,
    ) -> None:
        record: dict[str, object] = {
            "image": image_path.stem,
            "arm": arm,
            "status": "ok" if error is None else "error",
            "elapsed_seconds": time.perf_counter() - started,
        }
        if error is not None:
            record["error"] = f"{type(error).__name__}: {error}"[:1000]
        attempts.append(record)
        report_utils._write_json(
            output_root / "attempts.json",
            {"schema": REPORT_SCHEMA, "status": "diagnostic", "attempts": attempts},
        )

    def persist_rows() -> None:
        _write_tables(output_root, rows)

    run_started = time.perf_counter()
    for image_path in images:
        load_started = time.perf_counter()
        try:
            image, loaded_mask, raster = report_utils._load_evaluation_raster(
                image_path, None, max_side=args.max_side, mask_threshold=0.5
            )
            if loaded_mask is not None:
                raise RuntimeError("HIER-015 requires an internally generated full-frame mask")
        except Exception as exc:
            for arm in arms:
                record_attempt(image_path, arm, load_started, exc)
            continue
        mask = np.ones(image.shape[:2], dtype=bool)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

        control = None
        contraction_seconds = 0.0
        contraction_peak = 0
        additive_arms = tuple(arm for arm in arms if arm != DIRECT_ARM)
        if additive_arms:
            contraction_started = time.perf_counter()
            try:
                torch.cuda.reset_peak_memory_stats()
                control = contract_image(image, contraction_config, mask=mask)
                contraction_seconds = time.perf_counter() - contraction_started
                contraction_peak = int(torch.cuda.max_memory_allocated())
            except Exception as exc:
                for arm in additive_arms:
                    record_attempt(image_path, arm, contraction_started, exc)

        if control is not None and "h005_control" in arms:
            cell_started = time.perf_counter()
            try:
                row = _write_observation_cell(
                    output_root=output_root,
                    image_path=image_path,
                    image=image,
                    mask=mask,
                    raster=raster,
                    arm="h005_control",
                    field=control.field,
                    control_field=control.field,
                    control_reconstruction=control.reconstruction,
                    expected=control.reconstruction,
                    contraction_seconds=contraction_seconds,
                    method_seconds=0.0,
                    projection=None,
                    alternating=None,
                    peak_cuda_bytes=contraction_peak,
                    args=args,
                )
                rows.append(row)
                record_attempt(image_path, "h005_control", cell_started)
                persist_rows()
            except Exception as exc:
                record_attempt(image_path, "h005_control", cell_started, exc)

        if control is not None and "conditioned_transaction" in arms:
            cell_started = time.perf_counter()
            try:
                all_rows = np.ones(control.field.n, dtype=bool)
                torch.cuda.reset_peak_memory_stats()
                method_started = time.perf_counter()
                projection = project_contracted_coefficients(
                    control.field,
                    image,
                    mask,
                    all_rows,
                    config=transaction_config,
                    device=args.device,
                    renderer=args.additive_renderer,
                    render_chunk=args.render_chunk,
                )
                method_seconds = time.perf_counter() - method_started
                row = _write_observation_cell(
                    output_root=output_root,
                    image_path=image_path,
                    image=image,
                    mask=mask,
                    raster=raster,
                    arm="conditioned_transaction",
                    field=projection.field,
                    control_field=control.field,
                    control_reconstruction=control.reconstruction,
                    expected=projection.reconstruction_raw,
                    contraction_seconds=contraction_seconds,
                    method_seconds=method_seconds,
                    projection=projection,
                    alternating=None,
                    peak_cuda_bytes=int(torch.cuda.max_memory_allocated()),
                    args=args,
                )
                rows.append(row)
                record_attempt(image_path, "conditioned_transaction", cell_started)
                persist_rows()
            except Exception as exc:
                record_attempt(image_path, "conditioned_transaction", cell_started, exc)

        if control is not None:
            for arm in (candidate for candidate in HIERARCHY_ARMS if candidate in arms):
                cell_started = time.perf_counter()
                try:
                    torch.cuda.reset_peak_memory_stats()
                    method_started = time.perf_counter()
                    alternating = alternate_projected_geometry(
                        control.field,
                        image,
                        mask,
                        config=alternating_configs[arm],
                        device=args.device,
                        renderer=args.additive_renderer,
                        render_chunk=args.render_chunk,
                    )
                    method_seconds = time.perf_counter() - method_started
                    row = _write_observation_cell(
                        output_root=output_root,
                        image_path=image_path,
                        image=image,
                        mask=mask,
                        raster=raster,
                        arm=arm,
                        field=alternating.field,
                        control_field=control.field,
                        control_reconstruction=control.reconstruction,
                        expected=alternating.reconstruction_raw,
                        contraction_seconds=contraction_seconds,
                        method_seconds=method_seconds,
                        projection=None,
                        alternating=alternating,
                        peak_cuda_bytes=int(torch.cuda.max_memory_allocated()),
                        args=args,
                    )
                    rows.append(row)
                    record_attempt(image_path, arm, cell_started)
                    persist_rows()
                except Exception as exc:
                    record_attempt(image_path, arm, cell_started, exc)

        if DIRECT_ARM in arms:
            cell_started = time.perf_counter()
            try:
                direct_field, fit_result, init_seconds, direct_fit_config, peak = _run_direct(
                    image, args
                )
                row = _write_direct_cell(
                    output_root=output_root,
                    image_path=image_path,
                    image=image,
                    mask=mask,
                    raster=raster,
                    field=direct_field,
                    fit_result=fit_result,
                    init_seconds=init_seconds,
                    control_reconstruction=(
                        image if control is None else control.reconstruction
                    ),
                    peak_cuda_bytes=peak,
                    fit_config=direct_fit_config,
                    args=args,
                )
                rows.append(row)
                record_attempt(image_path, DIRECT_ARM, cell_started)
                persist_rows()
            except Exception as exc:
                record_attempt(image_path, DIRECT_ARM, cell_started, exc)

    decision = _aggregate(rows, args, attempts)
    decision["elapsed_seconds"] = time.perf_counter() - run_started
    report_utils._write_json(output_root / "decision.json", decision)
    _write_tables(output_root, rows)
    _write_report(output_root, rows, decision, command)
    _write_manifest(output_root)
    print(json.dumps(decision, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
