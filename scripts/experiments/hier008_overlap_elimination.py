#!/usr/bin/env python3
"""Run the frozen HIER-008 overlap-lattice x simplification diagnostic.

This is an exposed-data, dirty-source diagnostic.  Estimated row payloads and lossless NPZ sizes
are reported separately and are not complete codec rates.  The exact frozen command and protocol
live in ``tasks/HIER-008-overlap-lattice-feature-elimination.md``.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import csv
from html import escape
import json
import math
from pathlib import Path
import shlex
import shutil
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import hier005_pixel_contraction as report_utils  # noqa: E402

from structsplat.cli import save_error_heatmap, save_image  # noqa: E402
from structsplat.observation_field import ObservationField2D  # noqa: E402
from structsplat.overlap_elimination import (  # noqa: E402
    AppearanceSolveConfig,
    FeatureEliminationConfig,
    FieldOptimizerConfig,
    feature_wse_schur_eliminate,
    lattice_observation_field,
    optimize_observation_field,
    solve_fixed_lattice_appearance,
)
from structsplat.pixel_contraction import (  # noqa: E402
    PixelContractionConfig,
    contract_image,
    render_observation_field,
)
from structsplat.progressive_residual_quadtree import (  # noqa: E402
    progressive_artifact_metrics,
)


REPORT_SCHEMA = "structsplat.hier008_overlap_elimination.diagnostic.v1"
SCHEDULERS = ("quadtree", "feature_wse_schur")


def _support_arms(values: list[str]) -> dict[str, float]:
    arms: dict[str, float] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"support arm must have NAME=SCALE form, got {value!r}")
        name, raw_scale = value.split("=", maxsplit=1)
        if not name or name in arms:
            raise ValueError(f"support arm names must be non-empty and unique, got {name!r}")
        scale = float(raw_scale)
        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError(f"support scale must be finite and positive, got {scale}")
        arms[name] = scale
    if not arms:
        raise ValueError("at least one support arm is required")
    return arms


def _snapshot_sources(output_root: Path) -> list[dict[str, object]]:
    sources = (
        Path(__file__).resolve(),
        ROOT / "src" / "structsplat" / "overlap_elimination.py",
        ROOT / "src" / "structsplat" / "pixel_contraction.py",
        ROOT / "src" / "structsplat" / "sampling.py",
        ROOT / "src" / "structsplat" / "structure_tensor.py",
        ROOT / "scripts" / "experiments" / "hier005_pixel_contraction.py",
        ROOT / "tasks" / "HIER-008-overlap-lattice-feature-elimination.md",
    )
    records = []
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


def _feature_geometry_metrics(
    feature: np.ndarray,
    mask: np.ndarray,
    means: np.ndarray,
) -> dict[str, float]:
    from scipy.spatial import cKDTree

    yy, xx = np.nonzero(mask)
    foreground_points = np.stack([xx, yy], axis=1).astype(np.float64)
    points = np.asarray(means, dtype=np.float64)
    tree = cKDTree(points)
    distances, _ = tree.query(foreground_points, k=1, workers=1)
    rounded = np.rint(points).astype(np.int64)
    rounded[:, 0] = np.clip(rounded[:, 0], 0, mask.shape[1] - 1)
    rounded[:, 1] = np.clip(rounded[:, 1], 0, mask.shape[0] - 1)
    center_feature = feature[rounded[:, 1], rounded[:, 0]]
    foreground_feature = feature[mask]
    threshold = float(np.quantile(foreground_feature, 0.90))
    top_feature = foreground_feature >= threshold
    return {
        "feature_at_centers_mean": float(np.mean(center_feature)),
        "feature_at_centers_q90": float(np.quantile(center_feature, 0.90)),
        "top_feature_coverage_within_1_5px": float(np.mean(distances[top_feature] <= 1.5)),
        "nearest_center_distance_mean": float(np.mean(distances)),
        "nearest_center_distance_q99": float(np.quantile(distances, 0.99)),
        "nearest_center_distance_max": float(np.max(distances)),
    }


def _save_feature(path: Path, feature: np.ndarray, mask: np.ndarray) -> None:
    from PIL import Image
    from structsplat.visualize import scalar_heatmap

    values = np.where(mask, np.clip(feature, 0.0, 1.0), 0.0)
    Image.fromarray(scalar_heatmap(values.astype(np.float32)), mode="RGB").save(path)


def _save_centers(path: Path, source: np.ndarray, mask: np.ndarray, means: np.ndarray) -> None:
    from PIL import Image

    canvas = np.rint(np.clip(source * mask[:, :, None], 0.0, 1.0) * 255.0).astype(np.uint8)
    canvas = np.rint(canvas.astype(np.float32) * 0.45).astype(np.uint8)
    rounded = np.rint(np.asarray(means)).astype(np.int64)
    valid = (
        (rounded[:, 0] >= 0)
        & (rounded[:, 0] < source.shape[1])
        & (rounded[:, 1] >= 0)
        & (rounded[:, 1] < source.shape[0])
    )
    rounded = rounded[valid]
    canvas[rounded[:, 1], rounded[:, 0], 0] = 32
    canvas[rounded[:, 1], rounded[:, 0], 1] = 245
    canvas[rounded[:, 1], rounded[:, 0], 2] = 225
    Image.fromarray(canvas, mode="RGB").save(path)


def _worst_crop_bounds(
    reconstruction: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    side: int = 96,
) -> tuple[int, int, int, int]:
    pixel_mse = np.mean((reconstruction.astype(np.float64) - target.astype(np.float64)) ** 2, axis=2)
    pixel_mse[~mask] = -1.0
    y, x = np.unravel_index(int(np.argmax(pixel_mse)), pixel_mse.shape)
    width = min(side, target.shape[1])
    height = min(side, target.shape[0])
    x0 = min(max(x - width // 2, 0), target.shape[1] - width)
    y0 = min(max(y - height // 2, 0), target.shape[0] - height)
    return x0, y0, x0 + width, y0 + height


def _save_crop(path: Path, array: np.ndarray, bounds: tuple[int, int, int, int]) -> None:
    x0, y0, x1, y1 = bounds
    save_image(str(path), array[y0:y1, x0:x1])


def _plot_curves(
    output_root: Path,
    rows: list[dict[str, object]],
    checkpoint_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    curve_root = output_root / "curves"
    curve_root.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    excluded = {
        "schema",
        "status",
        "image",
        "arm",
        "support_arm",
        "scheduler",
        "artifact_dir",
        "source_path",
        "source_sha256",
        "mask_source_path",
        "mask_source_sha256",
        "field_canonical_sha256",
        "field_file_sha256",
        "lpips_error",
        "stop_reason",
        "artifact_metric_domain",
    }

    def numeric_metric_names(data: list[dict[str, object]], x_name: str) -> list[str]:
        names = set()
        for row in data:
            for name, value in row.items():
                if name in excluded or name == x_name or isinstance(value, bool):
                    continue
                if isinstance(value, (int, float)) and math.isfinite(float(value)):
                    names.add(name)
        return sorted(names)

    palette = ("#1769aa", "#d97706", "#16825d", "#9c3aa5", "#c33d4c", "#59636e")

    for metric in numeric_metric_names(rows, "n_gaussians"):
        figure, axis = plt.subplots(figsize=(6.4, 3.7), constrained_layout=True)
        plotted = False
        for index, arm in enumerate(sorted({str(row["arm"]) for row in rows})):
            points = sorted(
                (
                    (int(row["n_gaussians"]), float(row[metric]))
                    for row in rows
                    if row["arm"] == arm
                    and isinstance(row.get(metric), (int, float))
                    and not isinstance(row.get(metric), bool)
                    and math.isfinite(float(row[metric]))
                ),
                key=lambda value: value[0],
            )
            if not points:
                continue
            x, y = zip(*points)
            axis.plot(x, y, marker="o", linewidth=1.8, label=arm, color=palette[index % len(palette)])
            plotted = True
        if not plotted:
            plt.close(figure)
            continue
        axis.set_xscale("log", base=2)
        axis.set_xlabel("Gaussian count N")
        axis.set_ylabel(metric.replace("_", " "))
        axis.set_title(metric.replace("_", " "))
        axis.grid(True, alpha=0.28)
        axis.legend(fontsize=7)
        path = curve_root / f"snapshot__{metric}.svg"
        figure.savefig(path, format="svg")
        plt.close(figure)
        records.append({"kind": "snapshot", "metric": metric, "path": str(path.relative_to(output_root))})

    for metric in numeric_metric_names(checkpoint_rows, "step"):
        figure, axis = plt.subplots(figsize=(6.4, 3.7), constrained_layout=True)
        plotted = False
        groups = sorted({str(row["series"]) for row in checkpoint_rows})
        for index, series in enumerate(groups):
            points = sorted(
                (
                    (int(row["step"]), float(row[metric]))
                    for row in checkpoint_rows
                    if row["series"] == series
                    and isinstance(row.get(metric), (int, float))
                    and not isinstance(row.get(metric), bool)
                    and math.isfinite(float(row[metric]))
                ),
                key=lambda value: value[0],
            )
            if not points:
                continue
            x, y = zip(*points)
            axis.plot(x, y, marker=".", linewidth=1.3, label=series, color=palette[index % len(palette)])
            plotted = True
        if not plotted:
            plt.close(figure)
            continue
        axis.set_xlabel("Optimizer step")
        axis.set_ylabel(metric.replace("_", " "))
        axis.set_title(f"optimizer {metric.replace('_', ' ')}")
        axis.grid(True, alpha=0.28)
        axis.legend(fontsize=5, ncol=2)
        path = curve_root / f"optimizer__{metric}.svg"
        figure.savefig(path, format="svg")
        plt.close(figure)
        records.append({"kind": "optimizer", "metric": metric, "path": str(path.relative_to(output_root))})

    report_utils._write_json(
        curve_root / "catalog.json",
        {"schema": REPORT_SCHEMA, "curve_count": len(records), "curves": records},
    )
    return records


def _write_tables(output_root: Path, rows: list[dict[str, object]]) -> None:
    payload = {
        "schema": REPORT_SCHEMA,
        "status": "diagnostic",
        "claim_ready": False,
        "row_count": len(rows),
        "rows": rows,
        "rate_warning": (
            "estimated_field_bytes and canonical_raw_bytes are uncoded payload references; "
            "lossless_reference_bytes is an interchange container, not a COMP-013 stream"
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
        for row in rows:
            writer.writerow(row)


def _write_report(
    output_root: Path,
    rows: list[dict[str, object]],
    curve_records: list[dict[str, object]],
) -> None:
    table_rows = []
    cards = []
    artifact_links = []
    for row in sorted(rows, key=lambda value: (str(value["arm"]), -int(value["n_gaussians"]))):
        artifact = str(row["artifact_dir"])
        gate = "PASS" if row["artifact_gate_pass"] else "FAIL"
        table_rows.append(
            "<tr>"
            f"<td>{escape(str(row['arm']))}</td><td>{int(row['n_gaussians']):,}</td>"
            f"<td>{float(row['reduction_factor']):.2f}×</td>"
            f"<td>{float(row['psnr_db']):.3f}</td>"
            f"<td>{float(row['ms_ssim']):.5f}</td>"
            f"<td>{float(row['artifact_pixel_rmse_max']):.5f}</td>"
            f"<td>{float(row['artifact_patch_rmse_max_7']):.5f}</td>"
            f"<td class='{gate.lower()}'>{gate}</td>"
            f"<td>{int(row['optimizer_selected_step'])}</td>"
            f"<td>{float(row['optimizer_psnr_gain_db']):.3f}</td>"
            f"<td>{float(row['source_over_estimated_ratio']):.2f}×</td>"
            "</tr>"
        )
        card_id = f"{row['arm']} N={row['n_gaussians']}"
        cards.append(
            f"<section class='card'><h3>{escape(card_id)}</h3>"
            f"<p>PSNR {float(row['psnr_db']):.3f} dB · local gate {gate} · optimizer step "
            f"{int(row['optimizer_selected_step'])} · safe gain "
            f"{float(row['optimizer_psnr_gain_db']):.3f} dB</p>"
            "<div class='images'>"
            f"<figure><img src='{artifact}/source.png'><figcaption>source</figcaption></figure>"
            f"<figure><img src='{artifact}/preoptimization.png'><figcaption>before optimizer</figcaption></figure>"
            f"<figure><img src='{artifact}/reconstruction.png'><figcaption>selected result</figcaption></figure>"
            f"<figure><img src='{artifact}/error.png'><figcaption>selected error ×4</figcaption></figure>"
            f"<figure><img src='{artifact}/feature.png'><figcaption>structure feature</figcaption></figure>"
            f"<figure><img src='{artifact}/survivors.png'><figcaption>Gaussian centres</figcaption></figure>"
            f"<figure><img src='{artifact}/source_crop.png'><figcaption>worst-area source</figcaption></figure>"
            f"<figure><img src='{artifact}/reconstruction_crop.png'><figcaption>worst-area result</figcaption></figure>"
            f"<figure><img src='{artifact}/error_crop.png'><figcaption>worst-area error ×4</figcaption></figure>"
            "</div></section>"
        )
        for filename in (
            "source.png",
            "preoptimization.png",
            "preoptimization_error.png",
            "reconstruction.png",
            "error.png",
            "feature.png",
            "survivors.png",
            "source_crop.png",
            "reconstruction_crop.png",
            "error_crop.png",
            "field.observation.npz",
            "history.json",
            "optimizer_history.json",
            "analysis.npz",
            "config.json",
            "row.json",
        ):
            artifact_links.append(f"<li><a href='{artifact}/{filename}'>{escape(card_id)} / {filename}</a></li>")
    curve_links = "".join(
        f"<li><a href='{record['path']}'>{escape(str(record['kind']))}: "
        f"{escape(str(record['metric']))}</a></li>"
        for record in curve_records
    )
    snapshot_links = "".join(
        f"<li><a href='{path.relative_to(output_root)}'>{escape(str(path.relative_to(output_root)))}</a></li>"
        for path in sorted((output_root / "source_snapshot").rglob("*"))
        if path.is_file()
    )
    document = f"""<!doctype html><html><head><meta charset='utf-8'>
<title>HIER-008 overlap lattice × feature elimination</title>
<style>
body{{font-family:system-ui,sans-serif;margin:0;background:#f3f6f8;color:#18242d}}
main{{max-width:1500px;margin:auto;padding:24px}}h1,h2,h3{{margin-bottom:.4rem}}
.warning{{background:#fff3cd;border:1px solid #e0ba54;padding:12px;border-radius:7px}}
table{{border-collapse:collapse;width:100%;background:white;font-size:13px}}
th,td{{border:1px solid #dce3e8;padding:7px;text-align:right}}th:first-child,td:first-child{{text-align:left}}
.pass{{color:#087a48;font-weight:700}}.fail{{color:#b62929;font-weight:700}}
.card{{background:white;border:1px solid #dce3e8;border-radius:9px;padding:14px;margin:18px 0}}
.images{{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:10px}}
figure{{margin:0}}img{{width:100%;height:auto;image-rendering:auto;background:#101417}}
figcaption{{font-size:12px;color:#52616c}}.links{{columns:2;font-size:12px}}
</style></head><body><main>
<h1>HIER-008: overlap lattice × feature-safe elimination</h1>
<p class='warning'><strong>Diagnostic only.</strong> One exposed resized C0001 image, one seed, and a dirty source tree. Estimated rows and NPZ files are not complete codec rates. Negative cells are retained.</p>
<p><a href='manifest.json'>manifest</a> · <a href='metrics.json'>metrics JSON</a> ·
<a href='metrics.jsonl'>JSONL</a> · <a href='metrics.csv'>CSV</a> ·
<a href='config.json'>run config</a> · <a href='curves/catalog.json'>curve catalog</a></p>
<h2>Outcome table</h2><table><thead><tr><th>arm</th><th>N</th><th>reduction</th><th>PSNR</th>
<th>MS-SSIM</th><th>pixel max</th><th>7×7 max</th><th>gate</th><th>opt step</th>
<th>opt ΔPSNR</th><th>native JPEG / estimated</th></tr></thead><tbody>{''.join(table_rows)}</tbody></table>
<h2>Visual results</h2>{''.join(cards)}
<h2>Metric and optimizer curves</h2><ul class='links'>{curve_links}</ul>
<h2>Raw per-cell artifacts</h2><ul class='links'>{''.join(artifact_links)}</ul>
<h2>Executed source snapshot</h2><ul class='links'>{snapshot_links}</ul>
</main></body></html>"""
    (output_root / "index.html").write_text(document, encoding="utf-8")


def _write_manifest(output_root: Path) -> None:
    entries = []
    for path in sorted(output_root.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            entries.append(
                {
                    "path": str(path.relative_to(output_root)),
                    "bytes": path.stat().st_size,
                    "sha256": report_utils._sha256(path),
                }
            )
    report_utils._write_json(
        output_root / "manifest.json",
        {"schema": REPORT_SCHEMA, "status": "diagnostic", "files": entries},
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, nargs="+", required=True)
    parser.add_argument("--mask", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-side", type=int, default=512)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--target-gaussians", type=int, nargs="+", required=True)
    parser.add_argument("--support-arms", nargs="+", required=True)
    parser.add_argument("--schedulers", nargs="+", choices=SCHEDULERS, required=True)
    parser.add_argument("--cg-tolerance", type=float, default=1e-8)
    parser.add_argument("--cg-max-iterations", type=int, default=200)
    parser.add_argument("--cg-ridge", type=float, default=1e-8)
    parser.add_argument("--wse-alpha", type=float, default=8.0)
    parser.add_argument("--density-base", type=float, default=0.20)
    parser.add_argument("--density-power", type=float, default=0.50)
    parser.add_argument("--radius-min", type=float, default=0.65)
    parser.add_argument("--radius-max", type=float, default=2.25)
    parser.add_argument("--rgb-barrier", type=float, default=0.10)
    parser.add_argument("--feature-protection", type=float, default=4.0)
    parser.add_argument("--schur-ridge", type=float, default=1e-6)
    parser.add_argument("--optimizer-steps", type=int, default=80)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--lr-rgb", type=float, default=0.01)
    parser.add_argument("--lr-means", type=float, default=0.003)
    parser.add_argument("--lr-log-scales", type=float, default=0.002)
    parser.add_argument("--max-mean-shift", type=float, default=0.35)
    parser.add_argument("--max-log-scale-shift", type=float, default=0.15)
    parser.add_argument("--error-smoothing-sigma", type=float, default=1.5)
    parser.add_argument("--error-weight", type=float, default=2.0)
    parser.add_argument("--feature-weight", type=float, default=2.0)
    parser.add_argument("--tail-fraction", type=float, default=0.01)
    parser.add_argument("--tail-weight", type=float, default=2.0)
    parser.add_argument("--pixel-threshold", type=float, default=0.02)
    parser.add_argument("--patch7-threshold", type=float, default=0.01)
    parser.add_argument("--sigma-cutoff", type=float, default=3.0)
    parser.add_argument("--support-fade-alpha", type=float, default=0.0)
    parser.add_argument("--estimated-row-bytes", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--renderer",
        choices=("additive", "cuda_additive", "cuda_tiled_additive"),
        default="additive",
    )
    parser.add_argument("--render-chunk", type=int, default=256)
    parser.add_argument("--lpips", action="store_true")
    parser.add_argument("--error-scale", type=float, default=4.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if len(args.images) != 1:
        raise ValueError("the frozen HIER-008 diagnostic requires exactly one image")
    support_arms = _support_arms(args.support_arms)
    targets = sorted(set(args.target_gaussians), reverse=True)
    if any(value <= 0 for value in targets):
        raise ValueError("target counts must be positive")
    if len(set(args.schedulers)) != len(args.schedulers):
        raise ValueError("schedulers must be unique")
    output_root = args.out.resolve()
    output_root.mkdir(parents=True, exist_ok=False)
    source_snapshot = _snapshot_sources(output_root)
    command = " ".join(shlex.quote(value) for value in [sys.executable, *sys.argv])
    git = report_utils._git_record()
    solve_config = AppearanceSolveConfig(
        tolerance=args.cg_tolerance,
        max_iterations=args.cg_max_iterations,
        ridge=args.cg_ridge,
    )
    optimizer_config = FieldOptimizerConfig(
        steps=args.optimizer_steps,
        checkpoint_every=args.checkpoint_every,
        lr_rgb=args.lr_rgb,
        lr_means=args.lr_means,
        lr_log_scales=args.lr_log_scales,
        max_mean_shift=args.max_mean_shift,
        max_log_scale_shift=args.max_log_scale_shift,
        error_smoothing_sigma=args.error_smoothing_sigma,
        error_weight=args.error_weight,
        feature_weight=args.feature_weight,
        tail_fraction=args.tail_fraction,
        tail_weight=args.tail_weight,
        pixel_threshold=args.pixel_threshold,
        patch7_threshold=args.patch7_threshold,
        sigma_cutoff=args.sigma_cutoff,
        support_fade_alpha=args.support_fade_alpha,
        seed=args.seed,
        device=args.device,
        renderer=args.renderer,
        render_chunk=args.render_chunk,
    )
    image_path = args.images[0].resolve()
    mask_path = args.mask.resolve()
    image, loaded_mask, raster_record = report_utils._load_evaluation_raster(
        image_path,
        mask_path,
        max_side=args.max_side,
        mask_threshold=args.mask_threshold,
    )
    if loaded_mask is None:
        raise RuntimeError("the frozen HIER-008 diagnostic requires a mask")
    mask = loaded_mask
    target = image * mask[:, :, None]
    active_pixels = int(mask.sum())
    if targets[0] > active_pixels:
        raise ValueError("a target count exceeds the number of foreground pixels")
    rows: list[dict[str, object]] = []
    checkpoint_rows: list[dict[str, object]] = []
    run_started = time.perf_counter()

    for support_name, scale in support_arms.items():
        full_coefficients, full_reconstruction, full_solve = solve_fixed_lattice_appearance(
            image,
            mask,
            mask,
            scale_px=scale,
            sigma_cutoff=args.sigma_cutoff,
            support_fade_alpha=args.support_fade_alpha,
            config=solve_config,
        )
        if full_solve.coefficient_abs_max > optimizer_config.coefficient_limit:
            raise RuntimeError(
                f"{support_name} full prefit exceeds coefficient limit: "
                f"{full_solve.coefficient_abs_max}"
            )
        elimination_config = FeatureEliminationConfig(
            target_count=min(targets),
            alpha=args.wse_alpha,
            density_base=args.density_base,
            density_power=args.density_power,
            radius_min=args.radius_min,
            radius_max=args.radius_max,
            rgb_barrier=args.rgb_barrier,
            feature_protection=args.feature_protection,
            schur_ridge=args.schur_ridge,
        )
        elimination = feature_wse_schur_eliminate(
            image,
            mask,
            full_coefficients,
            targets,
            scale_px=scale,
            sigma_cutoff=args.sigma_cutoff,
            support_fade_alpha=args.support_fade_alpha,
            config=elimination_config,
        )
        coefficient_grid = np.zeros((*mask.shape, 3), dtype=np.float32)
        coefficient_grid[mask] = full_coefficients

        for scheduler in args.schedulers:
            for target_count in targets:
                cell_started = time.perf_counter()
                topology_history: list[dict[str, object]]
                reduced_solve_record: dict[str, object] | None
                stop_reason: str
                if scheduler == "quadtree":
                    contraction = contract_image(
                        image,
                        PixelContractionConfig(
                            target_gaussians=target_count,
                            leaf_scale_px=scale,
                            sigma_cutoff=args.sigma_cutoff,
                            support_fade_alpha=args.support_fade_alpha,
                            coefficient_domain="signed",
                            estimated_row_bytes=args.estimated_row_bytes,
                            recovery_steps=0,
                        ),
                        mask=mask,
                        initial_coefficients=full_coefficients,
                    )
                    preoptimization_field = contraction.field
                    topology_seconds = contraction.elapsed_seconds
                    topology_history = contraction.history_records()
                    reduced_solve_record = None
                    stop_reason = contraction.stop_reason
                else:
                    survivor_mask = elimination.survivors_by_count[target_count]
                    initial_survivors = coefficient_grid[survivor_mask]
                    topology_started = time.perf_counter()
                    coefficients, _, reduced_solve = solve_fixed_lattice_appearance(
                        image,
                        mask,
                        survivor_mask,
                        scale_px=scale,
                        sigma_cutoff=args.sigma_cutoff,
                        support_fade_alpha=args.support_fade_alpha,
                        config=solve_config,
                        initial_coefficients=initial_survivors,
                    )
                    preoptimization_field = lattice_observation_field(
                        mask,
                        survivor_mask,
                        coefficients,
                        scale_px=scale,
                        sigma_cutoff=args.sigma_cutoff,
                        support_fade_alpha=args.support_fade_alpha,
                    )
                    topology_seconds = time.perf_counter() - topology_started
                    topology_history = [
                        {
                            "kind": "feature_wse_schur",
                            "initial_count": active_pixels,
                            "target_count": target_count,
                            "removed_count": active_pixels - target_count,
                            "removal_order_prefix": elimination.removal_order[
                                : active_pixels - target_count
                            ].tolist(),
                        }
                    ]
                    reduced_solve_record = reduced_solve.to_record()
                    stop_reason = "target_reached"
                if preoptimization_field.n != target_count:
                    raise RuntimeError(
                        f"{support_name}/{scheduler} missed exact count {target_count}: "
                        f"{preoptimization_field.n}"
                    )
                preoptimization_raw = render_observation_field(
                    preoptimization_field,
                    device=args.device,
                    renderer=args.renderer,
                    render_chunk=args.render_chunk,
                    apply_declared_alpha=False,
                )
                preoptimization_display = preoptimization_raw * mask[:, :, None]
                pre_raw_metrics = progressive_artifact_metrics(
                    preoptimization_raw,
                    image,
                    mask,
                    pixel_threshold=args.pixel_threshold,
                    patch7_threshold=args.patch7_threshold,
                    displayed=False,
                )
                optimization = optimize_observation_field(
                    preoptimization_field,
                    image,
                    mask,
                    feature_normalized=elimination.feature_normalized,
                    config=optimizer_config,
                )
                arm = f"{support_name}__{scheduler}"
                series = f"{arm}__n{target_count}"
                for checkpoint in optimization.checkpoints:
                    checkpoint_rows.append(
                        {
                            "series": series,
                            "arm": arm,
                            "n_gaussians": target_count,
                            **checkpoint.to_record(),
                        }
                    )

                artifact_key = f"{image_path.stem}__{support_name}__{scheduler}__n{target_count}"
                artifact_dir = output_root / "artifacts" / artifact_key
                artifact_dir.mkdir(parents=True, exist_ok=False)
                field_path = artifact_dir / "field.observation.npz"
                optimization.field.save_lossless(field_path)
                lossless_bytes = field_path.stat().st_size
                decode_started = time.perf_counter()
                cold_field = ObservationField2D.load_lossless(field_path)
                cold_decode_seconds = time.perf_counter() - decode_started
                render_started = time.perf_counter()
                cold_raw = render_observation_field(
                    cold_field,
                    device=args.device,
                    renderer=args.renderer,
                    render_chunk=args.render_chunk,
                    apply_declared_alpha=False,
                )
                render_seconds = time.perf_counter() - render_started
                repeated_started = time.perf_counter()
                repeated_raw = render_observation_field(
                    cold_field,
                    device=args.device,
                    renderer=args.renderer,
                    render_chunk=args.render_chunk,
                    apply_declared_alpha=False,
                )
                repeated_render_seconds = time.perf_counter() - repeated_started
                reconstruction = cold_raw * mask[:, :, None]
                maintained_parity = float(np.max(np.abs(cold_raw - optimization.reconstruction_raw)))
                repeated_parity = float(np.max(np.abs(repeated_raw - cold_raw)))
                if maintained_parity >= 2e-6 or repeated_parity >= 2e-6:
                    raise RuntimeError(
                        f"maintained render parity failed for {artifact_key}: "
                        f"{maintained_parity}, {repeated_parity}"
                    )
                metric_started = time.perf_counter()
                metrics = report_utils._metric_values(
                    reconstruction,
                    target,
                    mask,
                    device=args.device,
                    compute_lpips=args.lpips,
                )
                metric_seconds = time.perf_counter() - metric_started
                raw_metrics = progressive_artifact_metrics(
                    cold_raw,
                    image,
                    mask,
                    pixel_threshold=args.pixel_threshold,
                    patch7_threshold=args.patch7_threshold,
                    displayed=False,
                )
                display_metrics = progressive_artifact_metrics(
                    reconstruction,
                    target,
                    mask,
                    pixel_threshold=args.pixel_threshold,
                    patch7_threshold=args.patch7_threshold,
                    displayed=True,
                )
                geometry_metrics = _feature_geometry_metrics(
                    elimination.feature_normalized,
                    mask,
                    cold_field.means_xy,
                )

                save_image(str(artifact_dir / "source.png"), target)
                save_image(str(artifact_dir / "preoptimization.png"), preoptimization_display)
                save_error_heatmap(
                    str(artifact_dir / "preoptimization_error.png"),
                    preoptimization_display - target,
                    scale=args.error_scale,
                )
                save_image(str(artifact_dir / "reconstruction.png"), reconstruction)
                save_error_heatmap(
                    str(artifact_dir / "error.png"),
                    reconstruction - target,
                    scale=args.error_scale,
                )
                _save_feature(
                    artifact_dir / "feature.png",
                    elimination.feature_normalized,
                    mask,
                )
                _save_centers(
                    artifact_dir / "survivors.png",
                    image,
                    mask,
                    cold_field.means_xy,
                )
                crop_bounds = _worst_crop_bounds(reconstruction, target, mask)
                _save_crop(artifact_dir / "source_crop.png", target, crop_bounds)
                _save_crop(artifact_dir / "reconstruction_crop.png", reconstruction, crop_bounds)
                error_visual = np.clip(np.mean(np.abs(reconstruction - target), axis=2) * args.error_scale, 0.0, 1.0)
                error_rgb = np.repeat(error_visual[:, :, None], 3, axis=2)
                _save_crop(artifact_dir / "error_crop.png", error_rgb, crop_bounds)

                history_payload = {
                    "schema": REPORT_SCHEMA,
                    "support_arm": support_name,
                    "scheduler": scheduler,
                    "target_count": target_count,
                    "full_lattice_solve": full_solve.to_record(),
                    "reduced_lattice_solve": reduced_solve_record,
                    "topology_history": topology_history,
                }
                report_utils._write_json(artifact_dir / "history.json", history_payload)
                report_utils._write_json(
                    artifact_dir / "optimizer_history.json",
                    {
                        "schema": REPORT_SCHEMA,
                        "selected_step": optimization.selected_step,
                        "checkpoints": [checkpoint.to_record() for checkpoint in optimization.checkpoints],
                    },
                )
                np.savez_compressed(
                    artifact_dir / "analysis.npz",
                    feature_normalized=elimination.feature_normalized,
                    density_relative=elimination.density_relative,
                    target_radius=elimination.target_radius,
                    schur_cost=elimination.schur_cost,
                    schur_residual_fraction=elimination.schur_residual_fraction,
                    eligible_neighbor_count=elimination.eligible_neighbor_count,
                    initial_crowding=elimination.initial_crowding,
                    removal_order=elimination.removal_order,
                    crop_bounds=np.asarray(crop_bounds, dtype=np.int32),
                )
                per_cell_config = {
                    "schema": REPORT_SCHEMA,
                    "status": "diagnostic",
                    "claim_ready": False,
                    "arm": arm,
                    "support_scale_px": scale,
                    "scheduler": scheduler,
                    "target_count": target_count,
                    "appearance_solve": asdict(solve_config),
                    "feature_elimination": asdict(elimination_config),
                    "optimizer": asdict(optimizer_config),
                }
                report_utils._write_json(artifact_dir / "config.json", per_cell_config)

                source_bytes = image_path.stat().st_size
                evaluation_source_bytes = (artifact_dir / "source.png").stat().st_size
                canonical_bytes = _canonical_raw_bytes(cold_field)
                alpha_bytes = 0 if cold_field.packed_alpha is None else int(cold_field.packed_alpha.nbytes)
                estimated_bytes = target_count * args.estimated_row_bytes + alpha_bytes
                pixel_count = int(mask.size)
                coefficient_values = np.asarray(cold_field.rgb_coeff)
                optimizer_selectable = sum(
                    int(checkpoint.selectable and checkpoint.step > 0)
                    for checkpoint in optimization.checkpoints
                )
                cell_total_seconds = time.perf_counter() - cell_started
                row: dict[str, object] = {
                    "schema": REPORT_SCHEMA,
                    "status": "diagnostic",
                    "image": image_path.stem,
                    "arm": arm,
                    "support_arm": support_name,
                    "scheduler": scheduler,
                    "support_scale_px": scale,
                    "axis_neighbor_peak_weight": math.exp(-0.5 / (scale * scale)),
                    "target_gaussians": target_count,
                    "n_gaussians": cold_field.n,
                    "active_pixels": active_pixels,
                    "reduction_factor": active_pixels / target_count,
                    "width": image.shape[1],
                    "height": image.shape[0],
                    "pixels": pixel_count,
                    "original_width": raster_record["original_width"],
                    "original_height": raster_record["original_height"],
                    "source_path": str(image_path),
                    "source_sha256": report_utils._sha256(image_path),
                    "source_file_bytes": source_bytes,
                    "mask_source_path": str(mask_path),
                    "mask_source_sha256": report_utils._sha256(mask_path),
                    "artifact_dir": str(artifact_dir.relative_to(output_root)),
                    "stop_reason": stop_reason,
                    "full_prefit_iterations": full_solve.iterations,
                    "full_prefit_converged": full_solve.converged,
                    "full_prefit_normal_residual": full_solve.relative_normal_residual_max,
                    "full_prefit_sse": full_solve.data_sse,
                    "full_prefit_pixel_rmse_max": full_solve.data_pixel_rmse_max,
                    "full_prefit_coefficient_abs_max": full_solve.coefficient_abs_max,
                    "full_prefit_negative_fraction": full_solve.negative_coefficient_fraction,
                    "full_prefit_seconds": full_solve.elapsed_seconds,
                    "feature_elimination_seconds": elimination.elapsed_seconds,
                    "schur_cost_mean": float(np.mean(elimination.schur_cost)),
                    "schur_cost_q90": float(np.quantile(elimination.schur_cost, 0.90)),
                    "schur_residual_fraction_mean": float(np.mean(elimination.schur_residual_fraction)),
                    "eligible_neighbor_count_mean": float(np.mean(elimination.eligible_neighbor_count)),
                    "target_radius_mean": float(np.mean(elimination.target_radius)),
                    "target_radius_min": float(np.min(elimination.target_radius)),
                    "target_radius_max": float(np.max(elimination.target_radius)),
                    "preoptimizer_raw_sse": float(pre_raw_metrics["sse"]),
                    "preoptimizer_raw_pixel_rmse_max": float(pre_raw_metrics["pixel_rmse_max"]),
                    "preoptimizer_raw_patch7_rmse_max": float(pre_raw_metrics["patch7_rmse_max"]),
                    "optimizer_selected_step": optimization.selected_step,
                    "optimizer_selectable_later_checkpoints": optimizer_selectable,
                    "optimizer_sse_gain": optimization.optimizer_sse_gain,
                    "optimizer_psnr_gain_db": optimization.optimizer_psnr_gain_db,
                    "optimizer_coefficient_abs_max": optimization.coefficient_abs_max,
                    "optimizer_mean_shift_max": optimization.mean_shift_max,
                    "optimizer_mean_shift_rms": optimization.mean_shift_rms,
                    "optimizer_log_scale_shift_max": optimization.log_scale_shift_max,
                    "optimizer_log_scale_shift_rms": optimization.log_scale_shift_rms,
                    "optimizer_seconds": optimization.elapsed_seconds,
                    "raw_sse": float(raw_metrics["sse"]),
                    "raw_artifact_pixel_rmse_max": float(raw_metrics["pixel_rmse_max"]),
                    "raw_artifact_patch7_rmse_max": float(raw_metrics["patch7_rmse_max"]),
                    "raw_artifact_normalized_violation": float(raw_metrics["normalized_violation"]),
                    "display_artifact_sse": float(display_metrics["sse"]),
                    "field_coefficient_abs_max": float(np.max(np.abs(coefficient_values))),
                    "field_negative_coefficient_fraction": float(np.mean(coefficient_values < 0.0)),
                    "estimated_field_bytes": estimated_bytes,
                    "canonical_raw_bytes": canonical_bytes,
                    "lossless_reference_bytes": lossless_bytes,
                    "estimated_bits_per_pixel": 8.0 * estimated_bytes / pixel_count,
                    "estimated_bits_per_active_pixel": 8.0 * estimated_bytes / active_pixels,
                    "source_over_estimated_ratio": source_bytes / estimated_bytes,
                    "source_over_canonical_raw_ratio": source_bytes / canonical_bytes,
                    "source_over_lossless_reference_ratio": source_bytes / lossless_bytes,
                    "evaluation_png_over_estimated_ratio": evaluation_source_bytes / estimated_bytes,
                    "field_canonical_sha256": cold_field.canonical_hash(),
                    "field_file_sha256": report_utils._sha256(field_path),
                    "maintained_render_parity_max_abs": maintained_parity,
                    "repeated_render_parity_max_abs": repeated_parity,
                    "topology_seconds": topology_seconds,
                    "cold_decode_seconds": cold_decode_seconds,
                    "render_seconds": render_seconds,
                    "repeated_render_seconds": repeated_render_seconds,
                    "metric_seconds": metric_seconds,
                    "total_seconds": cell_total_seconds,
                    **geometry_metrics,
                    **metrics,
                }
                report_utils._write_json(artifact_dir / "row.json", row)
                rows.append(row)
                print(
                    f"{arm} N={target_count}: PSNR={row['psnr_db']:.3f} dB, "
                    f"pixel={row['artifact_pixel_rmse_max']:.5f}, "
                    f"patch7={row['artifact_patch_rmse_max_7']:.5f}, "
                    f"gate={'PASS' if row['artifact_gate_pass'] else 'FAIL'}, "
                    f"opt_step={optimization.selected_step}, "
                    f"opt_gain={optimization.optimizer_psnr_gain_db:.3f} dB"
                )

    config_payload = {
        "schema": REPORT_SCHEMA,
        "status": "diagnostic",
        "claim_ready": False,
        "command": command,
        "args": vars(args),
        "support_arms": support_arms,
        "appearance_solve": asdict(solve_config),
        "optimizer": asdict(optimizer_config),
        "git": git,
        "executed_source_snapshot": source_snapshot,
        "evaluation_raster": raster_record,
        "run_seconds": time.perf_counter() - run_started,
    }
    report_utils._write_json(output_root / "config.json", config_payload)
    _write_tables(output_root, rows)
    curve_records = _plot_curves(output_root, rows, checkpoint_rows)
    _write_report(output_root, rows, curve_records)
    _write_manifest(output_root)
    print(f"wrote {output_root / 'index.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
