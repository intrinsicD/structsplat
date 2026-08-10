#!/usr/bin/env python3
"""Run the frozen HIER-010 residual-anchor/projection diagnostic.

The protocol authority is ``tasks/HIER-010-residual-anchor-projection.md``.  This is an exposed-
view, dirty-source-capable diagnostic, not a formal benchmark, semantic decision, default change,
or compression claim.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import csv
from html import escape
import json
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
from structsplat.cli import save_error_heatmap, save_image  # noqa: E402
from structsplat.contraction_refinement import (  # noqa: E402
    CoefficientProjectionConfig,
    project_contracted_coefficients,
    select_residual_anchor_leaves,
)
from structsplat.observation_field import ObservationField2D  # noqa: E402
from structsplat.overlap_elimination import lattice_observation_field  # noqa: E402
from structsplat.pixel_contraction import (  # noqa: E402
    PixelContractionConfig,
    contract_image,
    render_observation_field,
)


REPORT_SCHEMA = "structsplat.hier010_residual_anchor_projection.diagnostic.v1"
ARMS = (
    "h005_control",
    "control_projection",
    "residual_anchor",
    "anchor_projection",
)
EXPECTED_SOURCES = {
    "C0001": {
        "rgb": "ae24fe99d3f8edbd04cd2c85ebc4fe9bfd95abe878c22abb7691cadcfc5c411b",
        "mask": "94dcbf7005dbeb1d183e259a569d783aa5df900255e763385bed91f02d3b80c3",
    },
    "C0004": {
        "rgb": "26eb4cf24a034eb830198df6e7a6ac409ccb7cf4814ff645c71d0b6966b7070e",
        "mask": "4702bfa9df354f38e35a63207a37d4ec1b753afc4d0668bd905f3cdab320f35d",
    },
}


def _snapshot_sources(output_root: Path) -> list[dict[str, object]]:
    sources = (
        Path(__file__).resolve(),
        ROOT / "src" / "structsplat" / "contraction_refinement.py",
        ROOT / "src" / "structsplat" / "pixel_contraction.py",
        ROOT / "scripts" / "experiments" / "hier005_pixel_contraction.py",
        ROOT / "scripts" / "check_report_bundle.py",
        ROOT / "tasks" / "HIER-010-residual-anchor-projection.md",
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


def _canonical_raw_bytes(field: ObservationField2D) -> int:
    return int(sum(array.nbytes for array in field._array_items().values()))


def _save_scalar(path: Path, values: np.ndarray, mask: np.ndarray) -> None:
    shown = np.zeros((*mask.shape, 3), dtype=np.float32)
    active_values = values[mask].astype(np.float64)
    reference = max(float(np.quantile(active_values, 0.99)), np.finfo(np.float64).eps)
    normalized = np.clip(values.astype(np.float64) / reference, 0.0, 1.0)
    shown[mask] = normalized[mask, None]
    save_image(str(path), shown)


def _save_centers(
    path: Path,
    source: np.ndarray,
    mask: np.ndarray,
    means: np.ndarray,
    *,
    color: tuple[float, float, float],
) -> None:
    shown = np.asarray(source * mask[:, :, None], dtype=np.float32).copy()
    height, width = mask.shape
    for x_value, y_value in np.rint(means).astype(np.int64):
        x = int(np.clip(x_value, 0, width - 1))
        y = int(np.clip(y_value, 0, height - 1))
        shown[max(0, y - 1) : min(height, y + 2), x] = color
        shown[y, max(0, x - 1) : min(width, x + 2)] = color
    save_image(str(path), shown)


def _worst_crop_bounds(
    reconstruction: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    side: int = 96,
) -> tuple[int, int, int, int]:
    error = np.mean(np.square(reconstruction.astype(np.float64) - target), axis=2)
    error = np.where(mask, error, -1.0)
    y, x = np.unravel_index(int(np.argmax(error)), error.shape)
    crop_width = min(side, error.shape[1])
    crop_height = min(side, error.shape[0])
    x0 = min(max(0, x - crop_width // 2), error.shape[1] - crop_width)
    y0 = min(max(0, y - crop_height // 2), error.shape[0] - crop_height)
    return x0, y0, x0 + crop_width, y0 + crop_height


def _save_crop(path: Path, array: np.ndarray, bounds: tuple[int, int, int, int]) -> None:
    x0, y0, x1, y1 = bounds
    save_image(str(path), array[y0:y1, x0:x1])


def _write_tables(output_root: Path, rows: list[dict[str, object]]) -> None:
    payload = {
        "schema": REPORT_SCHEMA,
        "status": "diagnostic",
        "claim_ready": False,
        "row_count": len(rows),
        "rows": rows,
        "rate_warning": "Canonical and NPZ bytes are references, not a complete codec rate.",
    }
    report_utils._write_json(output_root / "metrics.json", payload)
    with (output_root / "metrics.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(report_utils._jsonable(row), sort_keys=True) + "\n")
    columns = sorted({name for row in rows for name in row})
    with (output_root / "metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _plot_curves(
    output_root: Path,
    rows: list[dict[str, object]],
    projection_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    curve_root = output_root / "curves"
    curve_root.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    arm_x = {arm: index for index, arm in enumerate(ARMS)}
    for metric in (
        "psnr_db",
        "masked_mse",
        "ms_ssim",
        "lpips",
        "artifact_pixel_rmse_max",
        "artifact_patch_rmse_max_7",
        "pipeline_cumulative_seconds",
    ):
        figure, axis = plt.subplots(figsize=(7.2, 3.8), constrained_layout=True)
        plotted = False
        for image_name in sorted({str(row["image"]) for row in rows}):
            points = [
                row
                for row in rows
                if row["image"] == image_name and isinstance(row.get(metric), (int, float))
            ]
            if not points:
                continue
            points.sort(key=lambda row: arm_x[str(row["arm"])])
            axis.plot(
                [arm_x[str(row["arm"])] for row in points],
                [float(row[metric]) for row in points],
                marker="o",
                label=image_name,
            )
            plotted = True
        if not plotted:
            plt.close(figure)
            continue
        axis.set_xticks(range(len(ARMS)), ARMS, rotation=18, ha="right")
        axis.set_ylabel(metric.replace("_", " "))
        axis.grid(True, alpha=0.28)
        axis.legend()
        path = curve_root / f"arms__{metric}.svg"
        figure.savefig(path, format="svg")
        plt.close(figure)
        records.append({"kind": "arms", "metric": metric, "path": str(path.relative_to(output_root))})

    for metric in ("raw_sse", "display_normalized_violation", "coefficient_abs_max"):
        figure, axis = plt.subplots(figsize=(7.2, 3.8), constrained_layout=True)
        plotted = False
        series_names = sorted({str(row["series"]) for row in projection_rows})
        for series in series_names:
            points = sorted(
                (row for row in projection_rows if row["series"] == series),
                key=lambda row: int(row["iteration"]),
            )
            if not points:
                continue
            axis.plot(
                [int(row["iteration"]) for row in points],
                [float(row[metric]) for row in points],
                marker=".",
                linewidth=1.2,
                label=series,
            )
            selected = [row for row in points if row["selected"]]
            if selected:
                axis.scatter(
                    [int(selected[0]["iteration"])],
                    [float(selected[0][metric])],
                    s=50,
                    marker="*",
                    zorder=4,
                )
            plotted = True
        if not plotted:
            plt.close(figure)
            continue
        axis.set_xlabel("PCG iteration")
        axis.set_ylabel(metric.replace("_", " "))
        if metric == "raw_sse":
            axis.set_yscale("log")
        axis.grid(True, alpha=0.28)
        axis.legend(fontsize=6, ncol=2)
        path = curve_root / f"projection__{metric}.svg"
        figure.savefig(path, format="svg")
        plt.close(figure)
        records.append(
            {"kind": "projection", "metric": metric, "path": str(path.relative_to(output_root))}
        )

    report_utils._write_json(
        curve_root / "catalog.json",
        {"schema": REPORT_SCHEMA, "curve_count": len(records), "curves": records},
    )
    return records


def _write_report(
    output_root: Path,
    rows: list[dict[str, object]],
    curves: list[dict[str, object]],
    decision: dict[str, object],
) -> None:
    table_rows: list[str] = []
    cards: list[str] = []
    artifact_links: list[str] = []
    for row in rows:
        gate = "PASS" if row["artifact_gate_pass"] else "FAIL"
        mechanism = "PASS" if row.get("full_mechanism_cell_pass") else "—"
        lpips_text = "n/a" if row.get("lpips") is None else f"{float(row['lpips']):.6f}"
        table_rows.append(
            "<tr>"
            f"<td>{escape(str(row['image']))}</td><td>{escape(str(row['arm']))}</td>"
            f"<td>{int(row['n_gaussians']):,}</td><td>{float(row['psnr_db']):.3f}</td>"
            f"<td>{float(row['psnr_delta_vs_control_db']):+.3f}</td>"
            f"<td>{float(row['ms_ssim']):.6f}</td>"
            f"<td>{lpips_text}</td>"
            f"<td>{float(row['artifact_pixel_rmse_max']):.5f}</td>"
            f"<td>{float(row['artifact_patch_rmse_max_7']):.5f}</td>"
            f"<td class='{gate.lower()}'>{gate}</td>"
            f"<td>{int(row['projection_selected_iteration'])}</td>"
            f"<td>{float(row['pipeline_cumulative_seconds']):.2f}</td>"
            f"<td>{mechanism}</td></tr>"
        )
        artifact = str(row["artifact_dir"])
        title = f"{row['image']} / {row['arm']}"
        cards.append(
            f"<section class='card'><h3>{escape(title)}</h3>"
            f"<p>{float(row['psnr_db']):.3f} dB ({float(row['psnr_delta_vs_control_db']):+.3f}); "
            f"pixel/7×7 {float(row['artifact_pixel_rmse_max']):.5f}/"
            f"{float(row['artifact_patch_rmse_max_7']):.5f}; gate {gate}; "
            f"projection step {int(row['projection_selected_iteration'])}</p><div class='images'>"
            f"<figure><img src='{artifact}/source.png'><figcaption>source</figcaption></figure>"
            f"<figure><img src='{artifact}/reconstruction.png'><figcaption>result</figcaption></figure>"
            f"<figure><img src='{artifact}/error.png'><figcaption>error ×4</figcaption></figure>"
            f"<figure><img src='{artifact}/feature_priority.png'><figcaption>residual anchor score</figcaption></figure>"
            f"<figure><img src='{artifact}/protected.png'><figcaption>protected leaves</figcaption></figure>"
            f"<figure><img src='{artifact}/centers.png'><figcaption>final centers</figcaption></figure>"
            f"<figure><img src='{artifact}/source_crop.png'><figcaption>worst-area source</figcaption></figure>"
            f"<figure><img src='{artifact}/reconstruction_crop.png'><figcaption>worst-area result</figcaption></figure>"
            f"<figure><img src='{artifact}/error_crop.png'><figcaption>worst-area error ×4</figcaption></figure>"
            "</div></section>"
        )
        for filename in (
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
            "projection_history.json",
            "analysis.npz",
            "config.json",
            "row.json",
        ):
            artifact_links.append(
                f"<li><a href='{artifact}/{filename}'>{escape(title)} / {filename}</a></li>"
            )
    curve_links = "".join(
        f"<li><a href='{record['path']}'>{escape(str(record['kind']))}: "
        f"{escape(str(record['metric']))}</a></li>"
        for record in curves
    )
    snapshots = "".join(
        f"<li><a href='{path.relative_to(output_root)}'>"
        f"{escape(str(path.relative_to(output_root)))}</a></li>"
        for path in sorted((output_root / "source_snapshot").rglob("*"))
        if path.is_file()
    )
    verdict = "PASS" if decision["advance_mechanism"] else "FAIL"
    document = f"""<!doctype html><html><head><meta charset='utf-8'>
<meta name='viewport' content='width=device-width,initial-scale=1'>
<title>HIER-010 residual anchor + projection</title><style>
body{{font-family:system-ui,sans-serif;margin:0;background:#f3f6f8;color:#18242d}}
main{{max-width:1500px;margin:auto;padding:24px}}.warning{{background:#fff3cd;border:1px solid #e0ba54;padding:12px;border-radius:8px}}
.verdict{{background:{'#e4f6ec' if decision['advance_mechanism'] else '#fee9e7'};border:1px solid {'#2e8b57' if decision['advance_mechanism'] else '#b64335'};padding:16px;border-radius:8px}}
table{{border-collapse:collapse;width:100%;background:white;font-size:13px}}th,td{{border:1px solid #dce3e8;padding:7px;text-align:right}}th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){{text-align:left}}
.pass{{color:#087a48;font-weight:700}}.fail{{color:#b62929;font-weight:700}}.card{{background:white;border:1px solid #dce3e8;border-radius:9px;padding:14px;margin:18px 0}}
.images{{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:10px}}figure{{margin:0}}img{{width:100%;height:auto;background:#101417}}figcaption{{font-size:12px;color:#52616c}}.links{{columns:2;font-size:12px}}
</style></head><body><main><p>STRUCTSPLAT / HIER-010 / EXPOSED-VIEW DIAGNOSTIC</p>
<h1>Residual anchors, then a safe linear finish</h1>
<p class='warning'><strong>Diagnostic only.</strong> Two exposed, correlated Janelle views; one numerically nondeterministic CUDA trajectory; dirty-source snapshot. Exact N is matched, work is not. Bytes are not codec rate and this report cannot change semantics or defaults.</p>
<section class='verdict'><h2>Frozen mechanism gate: {verdict}</h2><p>{escape(str(decision['summary']))}</p></section>
<p><a href='manifest.json'>manifest</a> · <a href='metrics.json'>metrics JSON</a> · <a href='metrics.jsonl'>JSONL</a> · <a href='metrics.csv'>CSV</a> · <a href='config.json'>config</a> · <a href='decision.json'>decision</a> · <a href='projection_checkpoints.json'>projection checkpoints</a> · <a href='recovery_checkpoints.json'>recovery checkpoints</a> · <a href='curves/catalog.json'>curves</a></p>
<h2>Exact-7k outcomes</h2><table><thead><tr><th>image</th><th>arm</th><th>N</th><th>PSNR</th><th>Δ control</th><th>MS-SSIM</th><th>LPIPS</th><th>pixel max</th><th>7×7 max</th><th>artifact gate</th><th>PCG step</th><th>pipeline s</th><th>full rule</th></tr></thead><tbody>{''.join(table_rows)}</tbody></table>
<h2>Visual audit</h2>{''.join(cards)}<h2>Metric and solver curves</h2><ul class='links'>{curve_links}</ul>
<h2>Raw cell artifacts</h2><ul class='links'>{''.join(artifact_links)}</ul><h2>Executed source snapshot</h2><ul class='links'>{snapshots}</ul>
</main></body></html>"""
    (output_root / "index.html").write_text(document, encoding="utf-8")


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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", nargs="+", type=Path, required=True)
    parser.add_argument("--masks", nargs="+", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--target-gaussians", type=int, default=7000)
    parser.add_argument("--max-side", type=int, default=512)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--anchor-count", type=int, default=350)
    parser.add_argument("--anchor-patch-side", type=int, default=7)
    parser.add_argument("--anchor-nms-radius", type=int, default=1)
    parser.add_argument("--projection-ridge", type=float, default=1e-8)
    parser.add_argument("--projection-tolerance", type=float, default=1e-6)
    parser.add_argument("--projection-max-iterations", type=int, default=48)
    parser.add_argument("--projection-coefficient-limit", type=float, default=16.0)
    parser.add_argument("--leaf-scale", type=float, default=0.18)
    parser.add_argument("--sigma-cutoff", type=float, default=3.0)
    parser.add_argument("--support-fade-alpha", type=float, default=0.0)
    parser.add_argument("--estimated-row-bytes", type=int, default=32)
    parser.add_argument("--proposal-batch-size", type=int, default=64)
    parser.add_argument("--merge-batch-size", type=int, default=8)
    parser.add_argument("--pair-shortlist", type=int, default=3)
    parser.add_argument("--exact-option-shortlist", type=int, default=2)
    parser.add_argument("--recovery-steps", type=int, default=50)
    parser.add_argument("--recovery-progress-checkpoints", type=int, default=16)
    parser.add_argument("--recovery-lr-means", type=float, default=0.005)
    parser.add_argument("--recovery-lr-scales", type=float, default=0.003)
    parser.add_argument("--recovery-lr-rotations", type=float, default=0.001)
    parser.add_argument("--recovery-lr-coefficients", type=float, default=0.003)
    parser.add_argument("--recovery-max-mean-shift", type=float, default=1.5)
    parser.add_argument("--recovery-max-log-scale-shift", type=float, default=0.35)
    parser.add_argument("--recovery-max-rotation-shift", type=float, default=0.35)
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


def _contraction_config(args: argparse.Namespace) -> PixelContractionConfig:
    return PixelContractionConfig(
        target_gaussians=args.target_gaussians,
        leaf_scale_px=args.leaf_scale,
        sigma_cutoff=args.sigma_cutoff,
        support_fade_alpha=args.support_fade_alpha,
        coefficient_domain="signed",
        estimated_row_bytes=args.estimated_row_bytes,
        proposal_batch_size=args.proposal_batch_size,
        merge_batch_size=args.merge_batch_size,
        pair_shortlist=args.pair_shortlist,
        exact_option_shortlist=args.exact_option_shortlist,
        pair_policy="exact_count",
        recovery_steps=args.recovery_steps,
        recovery_scope="touched",
        recovery_schedule="progress",
        recovery_progress_checkpoints=args.recovery_progress_checkpoints,
        recovery_device=args.device,
        recovery_renderer=args.renderer,
        recovery_render_chunk=args.render_chunk,
        recovery_lr_means=args.recovery_lr_means,
        recovery_lr_scales=args.recovery_lr_scales,
        recovery_lr_rotations=args.recovery_lr_rotations,
        recovery_lr_coefficients=args.recovery_lr_coefficients,
        recovery_max_mean_shift_px=args.recovery_max_mean_shift,
        recovery_max_log_scale_shift=args.recovery_max_log_scale_shift,
        recovery_max_rotation_shift_rad=args.recovery_max_rotation_shift,
    )


def _validate_protocol_inputs(
    images: list[Path], masks: list[Path], args: argparse.Namespace
) -> None:
    if len(images) != len(masks) or len(images) != 2:
        raise SystemExit("the frozen HIER-010 command requires exactly two image/mask pairs")
    if {path.stem for path in images} != set(EXPECTED_SOURCES):
        raise SystemExit("the frozen HIER-010 command requires C0001 and C0004")
    for image, mask in zip(images, masks, strict=True):
        expected = EXPECTED_SOURCES[image.stem]
        if report_utils._sha256(image) != expected["rgb"]:
            raise SystemExit(f"source hash mismatch for {image}")
        if report_utils._sha256(mask) != expected["mask"]:
            raise SystemExit(f"mask hash mismatch for {mask}")
    expected_scalars = {
        "target_gaussians": 7000,
        "max_side": 512,
        "anchor_count": 350,
        "anchor_patch_side": 7,
        "anchor_nms_radius": 1,
        "projection_max_iterations": 48,
        "estimated_row_bytes": 32,
        "proposal_batch_size": 64,
        "merge_batch_size": 8,
        "pair_shortlist": 3,
        "exact_option_shortlist": 2,
        "recovery_steps": 50,
        "recovery_progress_checkpoints": 16,
        "render_chunk": 256,
    }
    for name, expected in expected_scalars.items():
        if getattr(args, name) != expected:
            raise SystemExit(f"frozen protocol requires --{name.replace('_', '-')} {expected}")
    expected_floats = {
        "mask_threshold": 0.5,
        "projection_ridge": 1e-8,
        "projection_tolerance": 1e-6,
        "projection_coefficient_limit": 16.0,
        "leaf_scale": 0.18,
        "sigma_cutoff": 3.0,
        "support_fade_alpha": 0.0,
        "recovery_lr_means": 0.005,
        "recovery_lr_scales": 0.003,
        "recovery_lr_rotations": 0.001,
        "recovery_lr_coefficients": 0.003,
        "recovery_max_mean_shift": 1.5,
        "recovery_max_log_scale_shift": 0.35,
        "recovery_max_rotation_shift": 0.35,
        "error_scale": 4.0,
    }
    for name, expected in expected_floats.items():
        if float(getattr(args, name)) != expected:
            raise SystemExit(f"frozen protocol requires --{name.replace('_', '-')} {expected}")
    if args.renderer != "cuda_additive" or args.device != "cuda":
        raise SystemExit("frozen protocol requires --device cuda --renderer cuda_additive")
    if not args.lpips:
        raise SystemExit("frozen protocol requires --lpips")


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    images = [path.resolve() for path in args.images]
    masks = [path.resolve() for path in args.masks]
    _validate_protocol_inputs(images, masks, args)
    output_root = args.out.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise SystemExit(f"output directory must be absent or empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    command = shlex.join(
        [sys.executable, str(Path(__file__).relative_to(ROOT)), *(argv or sys.argv[1:])]
    )
    source_snapshot = _snapshot_sources(output_root)
    projection_config = CoefficientProjectionConfig(
        tolerance=args.projection_tolerance,
        max_iterations=args.projection_max_iterations,
        ridge=args.projection_ridge,
        coefficient_abs_limit=args.projection_coefficient_limit,
    )
    contraction_config = _contraction_config(args)

    import torch

    config: dict[str, object] = {
        "schema": REPORT_SCHEMA,
        "task": "HIER-010",
        "status": "diagnostic",
        "claim_ready": False,
        "command": command,
        "args": vars(args),
        "git": report_utils._git_record(),
        "executed_source_snapshot": source_snapshot,
        "contraction": asdict(contraction_config),
        "projection": asdict(projection_config),
        "anchor_rule": {
            "score": "max(pixel_mse/q99, mask_aware_box7_mse/q99)",
            "ranking": "stable_descending_score_then_row_major",
            "nms": "chebyshev_radius_1_then_stable_fill",
        },
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "cuda_device": torch.cuda.get_device_name(0),
        "evidence_limits": [
            "Both views are exposed and correlated; neither is confirmation.",
            "No distinct prospective reviewer; diagnostic only.",
            "CUDA recovery and projection accumulation are numerically nondeterministic.",
            "Two-pass arms spend more work; no equal-work speed conclusion.",
            "Canonical/raw/NPZ bytes are not complete codec rate.",
        ],
    }
    report_utils._write_json(output_root / "config.json", config)

    rows: list[dict[str, object]] = []
    projection_rows: list[dict[str, object]] = []
    recovery_rows: list[dict[str, object]] = []
    run_started = time.perf_counter()
    for image_path, mask_path in zip(images, masks, strict=True):
        image, loaded_mask, raster_record = report_utils._load_evaluation_raster(
            image_path,
            mask_path,
            max_side=args.max_side,
            mask_threshold=args.mask_threshold,
        )
        if loaded_mask is None:
            raise RuntimeError("HIER-010 requires a mask")
        mask = loaded_mask
        target = image * mask[:, :, None]
        active_pixels = int(mask.sum())
        if args.target_gaussians > active_pixels:
            raise RuntimeError("target count exceeds active pixels")

        torch.cuda.reset_peak_memory_stats()
        first_started = time.perf_counter()
        control = contract_image(image, contraction_config, mask=mask)
        first_seconds = time.perf_counter() - first_started
        selection_started = time.perf_counter()
        selection = select_residual_anchor_leaves(
            control.reconstruction,
            target,
            mask,
            args.anchor_count,
            patch_side=args.anchor_patch_side,
            nms_radius_px=args.anchor_nms_radius,
        )
        selection_seconds = time.perf_counter() - selection_started
        anchor_started = time.perf_counter()
        anchored = contract_image(
            image,
            contraction_config,
            mask=mask,
            protected_leaf_mask=selection.protected_mask,
        )
        anchor_seconds = time.perf_counter() - anchor_started
        control_projection = project_contracted_coefficients(
            control.field,
            target,
            mask,
            control.touched_row_mask,
            control.protected_row_mask,
            config=projection_config,
            device=args.device,
            renderer=args.renderer,
            render_chunk=args.render_chunk,
        )
        anchor_projection = project_contracted_coefficients(
            anchored.field,
            target,
            mask,
            anchored.touched_row_mask,
            anchored.protected_row_mask,
            config=projection_config,
            device=args.device,
            renderer=args.renderer,
            render_chunk=args.render_chunk,
        )
        peak_cuda_bytes = int(torch.cuda.max_memory_allocated())

        initial_field = lattice_observation_field(
            mask,
            mask,
            image[mask].astype(np.float32),
            scale_px=args.leaf_scale,
            sigma_cutoff=args.sigma_cutoff,
            support_fade_alpha=args.support_fade_alpha,
        )
        initial_raw = render_observation_field(
            initial_field,
            device=args.device,
            renderer=args.renderer,
            render_chunk=args.render_chunk,
            apply_declared_alpha=False,
        )

        arm_values = {
            "h005_control": (control, None, first_seconds),
            "control_projection": (
                control,
                control_projection,
                first_seconds + control_projection.elapsed_seconds,
            ),
            "residual_anchor": (
                anchored,
                None,
                first_seconds + selection_seconds + anchor_seconds,
            ),
            "anchor_projection": (
                anchored,
                anchor_projection,
                first_seconds
                + selection_seconds
                + anchor_seconds
                + anchor_projection.elapsed_seconds,
            ),
        }
        image_rows: list[dict[str, object]] = []
        for arm in ARMS:
            cell_started = time.perf_counter()
            contraction, projection, cumulative_seconds = arm_values[arm]
            final_field = contraction.field if projection is None else projection.field
            expected_reconstruction = (
                contraction.reconstruction if projection is None else projection.reconstruction
            )
            artifact_dir = output_root / "artifacts" / f"{image_path.stem}__{arm}__n7000"
            artifact_dir.mkdir(parents=True, exist_ok=False)
            field_path = artifact_dir / "field.observation.npz"
            final_field.save_lossless(field_path)
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
                target,
                mask,
                device=args.device,
                compute_lpips=args.lpips,
            )
            metric_seconds = time.perf_counter() - metric_started

            save_image(str(artifact_dir / "source.png"), target)
            save_image(str(artifact_dir / "initial_lattice.png"), initial_raw * mask[:, :, None])
            save_error_heatmap(
                str(artifact_dir / "initial_error.png"),
                initial_raw * mask[:, :, None] - target,
                scale=args.error_scale,
            )
            save_image(str(artifact_dir / "reconstruction.png"), cold)
            save_error_heatmap(
                str(artifact_dir / "error.png"), cold - target, scale=args.error_scale
            )
            _save_scalar(artifact_dir / "feature_priority.png", selection.score, mask)
            protected_y, protected_x = np.nonzero(selection.protected_mask)
            protected_means = np.stack([protected_x, protected_y], axis=1).astype(np.float32)
            shown_protected = protected_means if "anchor" in arm else np.empty((0, 2), np.float32)
            _save_centers(
                artifact_dir / "protected.png",
                image,
                mask,
                shown_protected,
                color=(1.0, 0.0, 0.2),
            )
            _save_centers(
                artifact_dir / "centers.png",
                image,
                mask,
                cold_field.means_xy,
                color=(0.0, 1.0, 0.2),
            )
            crop_bounds = _worst_crop_bounds(cold, target, mask)
            _save_crop(artifact_dir / "source_crop.png", target, crop_bounds)
            _save_crop(artifact_dir / "reconstruction_crop.png", cold, crop_bounds)
            error_visual = np.repeat(
                np.clip(np.mean(np.abs(cold - target), axis=2) * args.error_scale, 0.0, 1.0)[
                    :, :, None
                ],
                3,
                axis=2,
            )
            _save_crop(artifact_dir / "error_crop.png", error_visual, crop_bounds)
            report_utils._write_json(
                artifact_dir / "history.json", contraction.history_records()
            )
            report_utils._write_json(
                artifact_dir / "recovery_history.json", contraction.recovery_records()
            )
            projection_history = [] if projection is None else projection.checkpoint_records()
            report_utils._write_json(
                artifact_dir / "projection_history.json", projection_history
            )
            np.savez_compressed(
                artifact_dir / "analysis.npz",
                residual_anchor_score=selection.score,
                residual_pixel_mse=selection.pixel_mse,
                residual_patch_mse=selection.patch_mse,
                residual_protected_mask=selection.protected_mask,
                touched_row_mask=contraction.touched_row_mask,
                protected_row_mask=contraction.protected_row_mask,
                crop_bounds=np.asarray(crop_bounds, dtype=np.int32),
            )
            report_utils._write_json(
                artifact_dir / "config.json",
                {
                    "schema": REPORT_SCHEMA,
                    "status": "diagnostic",
                    "image": image_path.stem,
                    "arm": arm,
                    "contraction": asdict(contraction_config),
                    "projection": asdict(projection_config),
                    "projection_enabled": projection is not None,
                    "residual_anchor_enabled": "anchor" in arm,
                    "anchor_count": args.anchor_count,
                    "pipeline_cumulative_seconds": cumulative_seconds,
                },
            )

            canonical_bytes = _canonical_raw_bytes(cold_field)
            lossless_bytes = field_path.stat().st_size
            estimated_bytes = (
                cold_field.n * args.estimated_row_bytes
                + (0 if cold_field.packed_alpha is None else cold_field.packed_alpha.nbytes)
            )
            projection_selected = 0 if projection is None else projection.selected_iteration
            projection_initial_sse = (
                contraction.final_sse if projection is None else projection.initial_sse
            )
            projection_final_sse = (
                contraction.final_sse if projection is None else projection.final_sse
            )
            row: dict[str, object] = {
                "schema": REPORT_SCHEMA,
                "status": "diagnostic",
                "image": image_path.stem,
                "arm": arm,
                "artifact_dir": str(artifact_dir.relative_to(output_root)),
                "source_path": str(image_path),
                "source_sha256": report_utils._sha256(image_path),
                "source_file_bytes": image_path.stat().st_size,
                "mask_source_path": str(mask_path),
                "mask_source_sha256": report_utils._sha256(mask_path),
                "original_width": raster_record["original_width"],
                "original_height": raster_record["original_height"],
                "width": image.shape[1],
                "height": image.shape[0],
                "pixels": int(mask.size),
                "active_pixels": active_pixels,
                "target_gaussians": args.target_gaussians,
                "n_gaussians": cold_field.n,
                "reduction_factor": active_pixels / cold_field.n,
                "residual_anchor_enabled": "anchor" in arm,
                "projection_enabled": projection is not None,
                "anchor_requested_rows": args.anchor_count if "anchor" in arm else 0,
                "anchor_nms_selected_rows": selection.nms_selected_count if "anchor" in arm else 0,
                "anchor_pixel_reference_q99": selection.pixel_reference_q99,
                "anchor_patch_reference_q99": selection.patch_reference_q99,
                "stop_reason": contraction.stop_reason,
                "contraction_actions": len(contraction.history),
                "recovery_checkpoints": len(contraction.recovery_history),
                "recovery_accepted_checkpoints": sum(
                    event.accepted for event in contraction.recovery_history
                ),
                "recovery_optimizer_steps": sum(
                    event.attempted_steps for event in contraction.recovery_history
                ),
                "touched_active_rows": contraction.touched_active_rows,
                "untouched_active_rows": contraction.untouched_active_rows,
                "protected_active_rows": contraction.protected_active_rows,
                "blocked_regions": contraction.blocked_regions,
                "projection_selected_iteration": projection_selected,
                "projection_checkpoint_count": len(projection_history),
                "projection_trainable_rows": 0 if projection is None else projection.trainable_rows,
                "projection_frozen_rows": cold_field.n if projection is None else projection.frozen_rows,
                "projection_initial_sse": projection_initial_sse,
                "projection_final_sse": projection_final_sse,
                "projection_sse_gain": projection_initial_sse - projection_final_sse,
                "projection_forward_applications": 0 if projection is None else projection.forward_applications,
                "projection_transpose_applications": 0 if projection is None else projection.transpose_applications,
                "projection_relative_normal_residual_max": 0.0 if projection is None else projection.relative_normal_residual_max,
                "projection_adjoint_relative_error": 0.0 if projection is None else projection.adjoint_relative_error,
                "projection_internal_render_parity_max_abs": 0.0 if projection is None else projection.maintained_render_parity_max_abs,
                "first_pass_seconds": first_seconds,
                "anchor_selection_seconds": selection_seconds if "anchor" in arm else 0.0,
                "second_pass_seconds": anchor_seconds if "anchor" in arm else 0.0,
                "projection_seconds": 0.0 if projection is None else projection.elapsed_seconds,
                "pipeline_cumulative_seconds": cumulative_seconds,
                "cold_decode_seconds": cold_decode_seconds,
                "render_seconds": render_seconds,
                "metric_seconds": metric_seconds,
                "total_seconds": cumulative_seconds
                + cold_decode_seconds
                + 2.0 * render_seconds
                + metric_seconds,
                "peak_cuda_allocated_bytes": peak_cuda_bytes,
                "estimated_field_bytes": int(estimated_bytes),
                "canonical_raw_bytes": canonical_bytes,
                "lossless_reference_bytes": lossless_bytes,
                "estimated_bits_per_pixel": 8.0 * estimated_bytes / mask.size,
                "maintained_render_parity_max_abs": maintained_parity,
                "repeated_render_parity_max_abs": repeated_parity,
                "field_canonical_sha256": cold_field.canonical_hash(),
                "field_file_sha256": report_utils._sha256(field_path),
                "cell_packaging_seconds": time.perf_counter() - cell_started,
                **metrics,
            }
            image_rows.append(row)
            for event in contraction.recovery_history:
                recovery_rows.append(
                    {
                        "schema": REPORT_SCHEMA,
                        "series": f"{image_path.stem}__{arm}",
                        "image": image_path.stem,
                        "arm": arm,
                        **event.to_record(),
                    }
                )
            if projection is not None:
                for checkpoint in projection.checkpoints:
                    projection_rows.append(
                        {
                            "schema": REPORT_SCHEMA,
                            "series": f"{image_path.stem}__{arm}",
                            "image": image_path.stem,
                            "arm": arm,
                            **checkpoint.to_record(),
                        }
                    )

        control_row = next(row for row in image_rows if row["arm"] == "h005_control")
        for row in image_rows:
            row["psnr_delta_vs_control_db"] = float(row["psnr_db"]) - float(
                control_row["psnr_db"]
            )
            row["masked_mse_ratio_vs_control"] = float(row["masked_mse"]) / float(
                control_row["masked_mse"]
            )
            row["pixel_max_delta_vs_control"] = float(
                row["artifact_pixel_rmse_max"]
            ) - float(control_row["artifact_pixel_rmse_max"])
            row["patch7_max_delta_vs_control"] = float(
                row["artifact_patch_rmse_max_7"]
            ) - float(control_row["artifact_patch_rmse_max_7"])
            row["full_mechanism_cell_pass"] = bool(
                row["arm"] == "anchor_projection"
                and row["n_gaussians"] == args.target_gaussians
                and float(row["masked_mse"]) < float(control_row["masked_mse"])
                and float(row["artifact_pixel_rmse_max"])
                <= float(control_row["artifact_pixel_rmse_max"])
                and float(row["artifact_patch_rmse_max_7"])
                <= float(control_row["artifact_patch_rmse_max_7"])
                and float(row["maintained_render_parity_max_abs"]) <= 2e-6
                and float(row["repeated_render_parity_max_abs"]) <= 2e-6
                and float(row["projection_internal_render_parity_max_abs"]) <= 2e-6
                and float(row["projection_adjoint_relative_error"]) <= 2e-6
            )
            report_utils._write_json(
                output_root / str(row["artifact_dir"]) / "row.json", row
            )
        rows.extend(image_rows)
        print(
            f"{image_path.stem}: "
            + ", ".join(
                f"{row['arm']}={float(row['psnr_db']):.3f}dB/"
                f"{float(row['artifact_pixel_rmse_max']):.4f}/"
                f"{float(row['artifact_patch_rmse_max_7']):.4f}"
                for row in image_rows
            ),
            flush=True,
        )

    full_rows = [row for row in rows if row["arm"] == "anchor_projection"]
    advance = len(full_rows) == len(images) and all(
        bool(row["full_mechanism_cell_pass"]) for row in full_rows
    )
    decision = {
        "schema": REPORT_SCHEMA,
        "status": "diagnostic",
        "advance_mechanism": advance,
        "required_images": [path.stem for path in images],
        "passing_images": [row["image"] for row in full_rows if row["full_mechanism_cell_pass"]],
        "summary": (
            "The full count-neutral composition satisfies the frozen per-image MSE, local-max, "
            "exact-count, and parity rule on both exposed views. This motivates a fresh reviewed "
            "study only; it does not change a default."
            if advance
            else "The full composition misses at least one frozen per-image requirement. HIER-005 "
            "remains unchanged and these consumed views must not be retuned."
        ),
    }
    report_utils._write_json(output_root / "decision.json", decision)
    report_utils._write_json(
        output_root / "projection_checkpoints.json",
        {"schema": REPORT_SCHEMA, "rows": projection_rows},
    )
    report_utils._write_json(
        output_root / "recovery_checkpoints.json",
        {"schema": REPORT_SCHEMA, "rows": recovery_rows},
    )
    _write_tables(output_root, rows)
    curves = _plot_curves(output_root, rows, projection_rows)
    _write_report(output_root, rows, curves, decision)
    config["elapsed_seconds"] = time.perf_counter() - run_started
    config["decision"] = decision
    report_utils._write_json(output_root / "config.json", config)
    _write_manifest(output_root)
    print(f"wrote diagnostic report: {output_root / 'index.html'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
