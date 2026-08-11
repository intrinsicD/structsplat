#!/usr/bin/env python3
"""Run HIER-014's conditioned exact-7k appearance-projection diagnostic.

Kodak selection::

    PYTHONPATH=src python scripts/experiments/hier014_conditioned_projection.py \
      --phase kodak --images /home/alex/Documents/datasets/kodak24 \
      --out results/hier014_kodak_conditioned_projection_2026-08-10

Consumed-bank robustness replay (only after the Kodak recipe is frozen)::

    PYTHONPATH=src python scripts/experiments/hier014_conditioned_projection.py \
      --phase replay --images tests/test_images \
      --out results/hier014_test_images_conditioned_projection_2026-08-10

Both phases are development diagnostics.  The protocol authority is
``tasks/HIER-014-conditioned-minimum-norm-projection.md``.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import csv
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


REPORT_SCHEMA = "structsplat.hier014_conditioned_projection.diagnostic.v1"
KODAK_BINDINGS = {
    "kodim01.png": "a56e27cbf5f843c048b6af1d6e090760e9c92fadba88b7dee0205918a37523bd",
    "kodim07.png": "b77d3f006f42414bb242222e0482e750c0fb9e5ee8d4bed2f6f11c5605fe54a4",
    "kodim13.png": "bc34a3ce58dea09dce1704c997171602de90cb34d0c8503a988b77f473d39b08",
    "kodim19.png": "b7450b264b1b0a411390d8931b112c27905a992520fc90569dc4b920aa32bbdc",
}
KODAK_ARMS = (
    "h005_control",
    "legacy_input_subtract",
    "origin_subtract",
    "origin_explicit",
)
REPLAY_ARMS = (
    "h005_control",
    "legacy_input_subtract",
    "origin_explicit",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("kodak", "replay"), required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--target-gaussians", type=int, default=7000)
    parser.add_argument("--max-side", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--projection-ridge", type=float, default=1e-8)
    parser.add_argument("--projection-tolerance", type=float, default=1e-6)
    parser.add_argument("--projection-max-iterations", type=int, default=96)
    parser.add_argument("--coefficient-limit", type=float, default=16.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--renderer", default="cuda_additive")
    parser.add_argument("--render-chunk", type=int, default=256)
    parser.add_argument("--error-scale", type=float, default=4.0)
    parser.add_argument("--lpips", action=argparse.BooleanOptionalAction, default=True)
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    frozen = {
        "target_gaussians": 7000,
        "max_side": 512,
        "seed": 0,
        "projection_ridge": 1e-8,
        "projection_tolerance": 1e-6,
        "projection_max_iterations": 96,
        "coefficient_limit": 16.0,
        "device": "cuda",
        "renderer": "cuda_additive",
        "render_chunk": 256,
        "lpips": True,
    }
    for name, expected in frozen.items():
        if getattr(args, name) != expected:
            raise SystemExit(f"frozen HIER-014 protocol requires --{name.replace('_', '-')} {expected}")
    if args.error_scale <= 0.0 or not math.isfinite(args.error_scale):
        raise SystemExit("--error-scale must be finite and positive")


def _discover_images(args: argparse.Namespace) -> list[Path]:
    if args.phase == "kodak":
        images = [args.images / name for name in KODAK_BINDINGS]
        actual = {
            path.name: report_utils._sha256(path)
            for path in images
            if path.is_file()
        }
        if actual != KODAK_BINDINGS:
            raise SystemExit(
                "Kodak source bank is missing or hash-mismatched: "
                f"expected {KODAK_BINDINGS}, got {actual}"
            )
        return [path.resolve() for path in images]
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


def _projection_config(args: argparse.Namespace, arm: str) -> CoefficientProjectionConfig:
    origin = arm.startswith("origin_")
    return CoefficientProjectionConfig(
        ridge=args.projection_ridge,
        tolerance=args.projection_tolerance,
        max_iterations=args.projection_max_iterations,
        coefficient_abs_limit=args.coefficient_limit,
        regularization_center="zero" if origin else "input",
        solver_start="zero" if origin else "input",
        frozen_base_mode="explicit" if arm == "origin_explicit" else "subtract",
        allow_unsafe_stage_zero_reconditioning=origin,
    )


def _snapshot_sources(output_root: Path) -> list[dict[str, object]]:
    paths = (
        Path(__file__).resolve(),
        ROOT / "src" / "structsplat" / "contraction_refinement.py",
        ROOT / "src" / "structsplat" / "pixel_contraction.py",
        ROOT / "scripts" / "experiments" / "hier005_pixel_contraction.py",
        ROOT / "scripts" / "experiments" / "hier010_residual_anchor_projection.py",
        ROOT / "scripts" / "experiments" / "hier013_global_projection_development.py",
        ROOT / "tasks" / "HIER-014-conditioned-minimum-norm-projection.md",
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


def _projection_values(result: CoefficientProjectionResult | None) -> dict[str, object]:
    if result is None:
        return {
            "projection_selected_iteration": 0,
            "projection_checkpoint_count": 0,
            "projection_stage_zero_selectable": True,
            "projection_selected_selectable": True,
            "projection_initial_sse": 0.0,
            "projection_final_sse": 0.0,
            "projection_forward_applications": 0,
            "projection_transpose_applications": 0,
            "projection_relative_normal_residual_max": 0.0,
            "projection_adjoint_relative_error": 0.0,
            "projection_initial_operator_parity_max_abs": 0.0,
            "projection_internal_render_parity_max_abs": 0.0,
            "projection_normal_diagonal_min": 0.0,
            "projection_normal_diagonal_max": 0.0,
            "projection_normal_diagonal_ratio": 1.0,
            "projection_coefficient_abs_max": 0.0,
            "projection_seconds": 0.0,
        }
    selected = next(checkpoint for checkpoint in result.checkpoints if checkpoint.selected)
    diagonal_ratio = result.normal_diagonal_max / max(
        result.normal_diagonal_min, np.finfo(np.float64).tiny
    )
    return {
        "projection_selected_iteration": result.selected_iteration,
        "projection_checkpoint_count": len(result.checkpoints),
        "projection_stage_zero_selectable": result.checkpoints[0].selectable,
        "projection_selected_selectable": selected.selectable,
        "projection_initial_sse": result.initial_sse,
        "projection_final_sse": result.final_sse,
        "projection_forward_applications": result.forward_applications,
        "projection_transpose_applications": result.transpose_applications,
        "projection_relative_normal_residual_max": result.relative_normal_residual_max,
        "projection_adjoint_relative_error": result.adjoint_relative_error,
        "projection_initial_operator_parity_max_abs": result.initial_operator_parity_max_abs,
        "projection_internal_render_parity_max_abs": result.maintained_render_parity_max_abs,
        "projection_normal_diagonal_min": result.normal_diagonal_min,
        "projection_normal_diagonal_max": result.normal_diagonal_max,
        "projection_normal_diagonal_ratio": diagonal_ratio,
        "projection_coefficient_abs_max": selected.coefficient_abs_max,
        "projection_seconds": result.elapsed_seconds,
    }


def _write_cell(
    *,
    output_root: Path,
    image_path: Path,
    image: np.ndarray,
    mask: np.ndarray,
    raster: dict[str, object],
    arm: str,
    field: ObservationField2D,
    geometry_source: ObservationField2D,
    expected: np.ndarray,
    control_reconstruction: np.ndarray,
    contraction_seconds: float,
    projection: CoefficientProjectionResult | None,
    peak_cuda_bytes: int,
    args: argparse.Namespace,
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
    save_image(str(artifact_dir / "control.png"), control_reconstruction)
    save_image(str(artifact_dir / "reconstruction.png"), cold)
    save_error_heatmap(
        str(artifact_dir / "error.png"),
        cold - image,
        scale=args.error_scale,
    )
    bounds = viz_utils._worst_crop_bounds(cold, image, mask)
    viz_utils._save_crop(artifact_dir / "source_crop.png", image, bounds)
    viz_utils._save_crop(artifact_dir / "reconstruction_crop.png", cold, bounds)
    shown_error = np.repeat(
        np.clip(np.mean(np.abs(cold - image), axis=2) * args.error_scale, 0.0, 1.0)[
            :, :, None
        ],
        3,
        axis=2,
    )
    viz_utils._save_crop(artifact_dir / "error_crop.png", shown_error, bounds)
    report_utils._write_json(
        artifact_dir / "projection_history.json",
        [] if projection is None else projection.checkpoint_records(),
    )
    np.savez_compressed(
        artifact_dir / "analysis.npz",
        crop_bounds=np.asarray(bounds, dtype=np.int32),
        mask=mask,
        coefficient_abs=np.abs(cold_field.rgb_coeff),
    )

    projection_values = _projection_values(projection)
    coefficient_abs_max = float(np.max(np.abs(cold_field.rgb_coeff)))
    row: dict[str, object] = {
        "schema": REPORT_SCHEMA,
        "status": "diagnostic",
        "phase": args.phase,
        "image": image_path.stem,
        "arm": arm,
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
        "field_canonical_sha256": cold_field.canonical_hash(),
        "field_file_sha256": report_utils._sha256(field_path),
        "geometry_source_canonical_sha256": geometry_source.canonical_hash(),
        "non_rgb_arrays_bit_exact": h13._non_rgb_equal(cold_field, geometry_source),
        "coefficient_abs_max": coefficient_abs_max,
        "coefficient_abs_median": float(np.median(np.abs(cold_field.rgb_coeff))),
        "coefficient_abs_q99": float(np.quantile(np.abs(cold_field.rgb_coeff), 0.99)),
        "contraction_seconds": contraction_seconds,
        "pipeline_algorithm_seconds": contraction_seconds
        + float(projection_values["projection_seconds"]),
        "projection_overhead_ratio": float(projection_values["projection_seconds"])
        / max(contraction_seconds, 1e-12),
        "cold_decode_seconds": decode_seconds,
        "render_seconds": render_seconds,
        "metric_seconds": metric_seconds,
        "peak_cuda_allocated_bytes": peak_cuda_bytes,
        "lossless_reference_bytes": field_path.stat().st_size,
        "maintained_render_parity_max_abs": float(np.max(np.abs(cold - expected))),
        "repeated_render_parity_max_abs": float(np.max(np.abs(repeated - cold))),
        **projection_values,
        **metrics,
    }
    report_utils._write_json(artifact_dir / "row.json", row)
    return row


def _paired(rows: list[dict[str, object]], arm: str) -> list[dict[str, object]]:
    controls = {
        str(row["image"]): row for row in rows if row["arm"] == "h005_control"
    }
    result: list[dict[str, object]] = []
    for row in rows:
        if row["arm"] != arm:
            continue
        control = controls[str(row["image"])]
        result.append(
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
                "projection_selected_iteration": row["projection_selected_iteration"],
                "coefficient_abs_max": row["coefficient_abs_max"],
                "maintained_render_parity_max_abs": row[
                    "maintained_render_parity_max_abs"
                ],
                "repeated_render_parity_max_abs": row[
                    "repeated_render_parity_max_abs"
                ],
                "projection_overhead_ratio": row["projection_overhead_ratio"],
                "non_rgb_arrays_bit_exact": row["non_rgb_arrays_bit_exact"],
                "n_gaussians": row["n_gaussians"],
            }
        )
    return result


def _aggregate(rows: list[dict[str, object]], args: argparse.Namespace) -> dict[str, object]:
    arms = KODAK_ARMS if args.phase == "kodak" else REPLAY_ARMS
    arm_records: dict[str, object] = {}
    for arm in arms[1:]:
        pairs = _paired(rows, arm)
        ratios = np.asarray([float(pair["mse_ratio"]) for pair in pairs])
        arm_records[arm] = {
            "pairs": pairs,
            "geometric_mean_mse_ratio": float(np.exp(np.mean(np.log(ratios)))),
            "mean_psnr_delta_db": float(
                np.mean([float(pair["psnr_delta_db"]) for pair in pairs])
            ),
            "mean_ms_ssim_delta": float(
                np.mean([float(pair["ms_ssim_delta"]) for pair in pairs])
            ),
            "mean_lpips_delta": float(
                np.mean([float(pair["lpips_delta"]) for pair in pairs])
            ),
            "maximum_pixel_max_delta": max(
                float(pair["pixel_max_delta"]) for pair in pairs
            ),
            "maximum_patch7_max_delta": max(
                float(pair["patch7_max_delta"]) for pair in pairs
            ),
            "nonzero_solves": sum(
                int(pair["projection_selected_iteration"]) > 0 for pair in pairs
            ),
            "pair_count": len(pairs),
        }

    selected = arm_records["origin_explicit"]
    pairs = selected["pairs"]
    common = {
        "all_nonzero": all(int(pair["projection_selected_iteration"]) > 0 for pair in pairs),
        "all_exact_count": all(int(pair["n_gaussians"]) == 7000 for pair in pairs),
        "all_non_rgb_exact": all(bool(pair["non_rgb_arrays_bit_exact"]) for pair in pairs),
        "all_coefficients_bounded": all(
            float(pair["coefficient_abs_max"]) <= 16.0 for pair in pairs
        ),
        "all_parity_le_2e_6": all(
            float(pair["maintained_render_parity_max_abs"]) <= 2e-6
            and float(pair["repeated_render_parity_max_abs"]) <= 2e-6
            for pair in pairs
        ),
        "all_mse_noninferior": all(float(pair["mse_ratio"]) <= 1.0 + 1e-8 for pair in pairs),
        "all_pixel_max_noninferior": all(
            float(pair["pixel_max_delta"]) <= 1e-12 for pair in pairs
        ),
        "all_patch7_max_noninferior": all(
            float(pair["patch7_max_delta"]) <= 1e-12 for pair in pairs
        ),
        "geometric_mean_mse_ratio_le_0_90": float(
            selected["geometric_mean_mse_ratio"]
        )
        <= 0.90,
    }
    if args.phase == "kodak":
        common.update(
            {
                "mean_ms_ssim_noninferior": float(selected["mean_ms_ssim_delta"])
                >= -1e-7,
                "mean_lpips_noninferior": float(selected["mean_lpips_delta"]) <= 1e-7,
                "median_overhead_le_0_25": float(
                    np.median([float(pair["projection_overhead_ratio"]) for pair in pairs])
                )
                <= 0.25,
            }
        )
    decision = "pass" if all(common.values()) else "fail"
    return {
        "schema": REPORT_SCHEMA,
        "status": "diagnostic",
        "phase": args.phase,
        "arm_aggregates": arm_records,
        "gate_predicates": common,
        "decision": decision,
        "interpretation": (
            "The frozen conditioned projection clears every measured gate."
            if decision == "pass"
            else "The frozen conditioned projection misses at least one measured gate; do not tune this bank in place."
        ),
    }


def _write_tables(output_root: Path, rows: list[dict[str, object]]) -> None:
    report_utils._write_json(
        output_root / "metrics.json",
        {"schema": REPORT_SCHEMA, "status": "diagnostic", "rows": rows},
    )
    with (output_root / "metrics.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    columns = sorted({key for row in rows for key in row})
    with (output_root / "metrics.csv").open("w", encoding="utf-8", newline="") as handle:
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
            f"<td>{int(row['n_gaussians'])}</td><td>{float(row['psnr_db']):.4f}</td>"
            f"<td>{float(row['masked_mse']):.7g}</td><td>{float(row['ms_ssim']):.6f}</td>"
            f"<td>{float(row['lpips']):.6f}</td>"
            f"<td>{float(row['coefficient_abs_max']):.3f}</td>"
            f"<td>{int(row['projection_selected_iteration'])}</td>"
            f"<td>{float(row['maintained_render_parity_max_abs']):.3g}</td>"
            f"<td><a href='{escape(str(row['artifact_dir']))}/reconstruction.png'>full</a> · "
            f"<a href='{escape(str(row['artifact_dir']))}/reconstruction_crop.png'>crop</a> · "
            f"<a href='{escape(str(row['artifact_dir']))}/projection_history.json'>solver</a></td>"
            "</tr>"
        )
    cards: list[str] = []
    for row in rows:
        artifact = escape(str(row["artifact_dir"]))
        cards.append(
            "<section><h3>"
            f"{escape(str(row['image']))} — {escape(str(row['arm']))}</h3>"
            f"<a href='{artifact}/source.png'><img src='{artifact}/source.png'></a>"
            f"<a href='{artifact}/reconstruction.png'><img src='{artifact}/reconstruction.png'></a>"
            f"<a href='{artifact}/error.png'><img src='{artifact}/error.png'></a>"
            f"<a href='{artifact}/reconstruction_crop.png'><img src='{artifact}/reconstruction_crop.png'></a>"
            "</section>"
        )
    gate_rows = "".join(
        f"<tr><td>{escape(name)}</td><td>{'PASS' if value else 'FAIL'}</td></tr>"
        for name, value in decision["gate_predicates"].items()
    )
    html = f"""<!doctype html><html><head><meta charset='utf-8'>
<title>HIER-014 conditioned projection</title><style>
body{{font-family:system-ui;margin:2rem;max-width:1500px}}table{{border-collapse:collapse}}
th,td{{border:1px solid #bbb;padding:.35rem}}img{{max-width:310px;max-height:250px;margin:.25rem}}
section{{border-top:1px solid #aaa;margin-top:1rem}}code{{white-space:pre-wrap}}
</style></head><body><h1>HIER-014 conditioned exact-7k projection — {escape(str(decision['phase']))}</h1>
<p><strong>Diagnostic decision:</strong> {escape(str(decision['decision']))}. {escape(str(decision['interpretation']))}</p>
<p>Development/reporting-only evidence. No default, semantic, codec, rate, or held-out claim.</p>
<p><code>{escape(command)}</code></p>
<p><a href='config.json'>config</a> · <a href='decision.json'>decision</a> ·
<a href='metrics.json'>JSON</a> · <a href='metrics.jsonl'>JSONL</a> ·
<a href='metrics.csv'>CSV</a> · <a href='manifest.json'>manifest</a></p>
<h2>Gate</h2><table><tr><th>predicate</th><th>result</th></tr>{gate_rows}</table>
<h2>Cells</h2><table><tr><th>image</th><th>arm</th><th>N</th><th>PSNR</th><th>MSE</th>
<th>MS-SSIM</th><th>LPIPS</th><th>|c|max</th><th>PCG</th><th>cold parity</th><th>artifacts</th></tr>
{''.join(table_rows)}</table><h2>Visual audit</h2>{''.join(cards)}</body></html>"""
    (output_root / "index.html").write_text(html, encoding="utf-8")


def _write_manifest(output_root: Path) -> None:
    records: list[dict[str, object]] = []
    for path in sorted(output_root.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            records.append(
                {
                    "path": str(path.relative_to(output_root)),
                    "bytes": path.stat().st_size,
                    "sha256": report_utils._sha256(path),
                }
            )
    report_utils._write_json(
        output_root / "manifest.json",
        {"schema": REPORT_SCHEMA, "status": "diagnostic", "files": records},
    )


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    _validate_args(args)
    images = _discover_images(args)
    output_root = args.out.resolve()
    output_root.mkdir(parents=True, exist_ok=False)

    import torch

    command = shlex.join([sys.executable, str(Path(__file__).resolve()), *(argv or sys.argv[1:])])
    contraction_config = _contraction_config(args)
    arms = KODAK_ARMS if args.phase == "kodak" else REPLAY_ARMS
    projection_configs = {
        arm: _projection_config(args, arm) for arm in arms if arm != "h005_control"
    }
    source_snapshots = _snapshot_sources(output_root)
    config = {
        "schema": REPORT_SCHEMA,
        "status": "diagnostic",
        "phase": args.phase,
        "command": command,
        "args": vars(args),
        "arms": list(arms),
        "contraction": asdict(contraction_config),
        "projections": {arm: asdict(cfg) for arm, cfg in projection_configs.items()},
        "sources": [
            {"path": str(path), "sha256": report_utils._sha256(path)} for path in images
        ],
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
            "Diagnostic dirty-source-capable execution without distinct prospective review.",
            "Kodak is development selection; HIER-013's bank is consumed reporting-only data.",
            "CUDA atomic accumulation is numerically, not bit, reproducible.",
            "Lossless NPZ bytes are reference persistence, not a production codec rate.",
        ],
    }
    report_utils._write_json(output_root / "config.json", config)

    rows: list[dict[str, object]] = []
    run_started = time.perf_counter()
    for image_path in images:
        image, loaded_mask, raster = report_utils._load_evaluation_raster(
            image_path,
            None,
            max_side=args.max_side,
            mask_threshold=0.5,
        )
        if loaded_mask is not None:
            raise RuntimeError("HIER-014 requires an internally generated full-frame mask")
        mask = np.ones(image.shape[:2], dtype=bool)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.reset_peak_memory_stats()
        contraction_started = time.perf_counter()
        control = contract_image(image, contraction_config, mask=mask)
        contraction_seconds = time.perf_counter() - contraction_started
        all_rows = np.ones(control.field.n, dtype=bool)
        no_rows = np.zeros(control.field.n, dtype=bool)
        projections: dict[str, CoefficientProjectionResult] = {}
        for arm, projection_config in projection_configs.items():
            projections[arm] = project_contracted_coefficients(
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
        peak_cuda_bytes = int(torch.cuda.max_memory_allocated())
        arm_values: dict[
            str,
            tuple[ObservationField2D, np.ndarray, CoefficientProjectionResult | None],
        ] = {
            "h005_control": (control.field, control.reconstruction, None),
            **{
                arm: (projection.field, projection.reconstruction, projection)
                for arm, projection in projections.items()
            },
        }
        for arm in arms:
            field, expected, projection = arm_values[arm]
            row = _write_cell(
                output_root=output_root,
                image_path=image_path,
                image=image,
                mask=mask,
                raster=raster,
                arm=arm,
                field=field,
                geometry_source=control.field,
                expected=expected,
                control_reconstruction=control.reconstruction,
                contraction_seconds=contraction_seconds,
                projection=projection,
                peak_cuda_bytes=peak_cuda_bytes,
                args=args,
            )
            rows.append(row)
            _write_tables(output_root, rows)

    decision = _aggregate(rows, args)
    decision["elapsed_seconds"] = time.perf_counter() - run_started
    report_utils._write_json(output_root / "decision.json", decision)
    _write_tables(output_root, rows)
    _write_report(output_root, rows, decision, command)
    _write_manifest(output_root)
    print(json.dumps(decision, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
