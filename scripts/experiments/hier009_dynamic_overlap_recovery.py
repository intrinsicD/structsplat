#!/usr/bin/env python3
"""Run the frozen HIER-009 dynamic overlap/neighborhood diagnostic.

The exact exposed-image protocol and reproduction command live in
``tasks/HIER-009-dynamic-overlap-neighborhood-recovery.md``.  This is a dirty-source diagnostic,
not a formal benchmark or compression claim.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import csv
import json
import math
import platform
from pathlib import Path
import shlex
import shutil
import sys
import time
from html import escape

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
for candidate in (ROOT, ROOT / "src"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from scripts.experiments import hier005_pixel_contraction as report_utils  # noqa: E402
from structsplat.cli import save_error_heatmap, save_image  # noqa: E402
from structsplat.observation_field import ObservationField2D  # noqa: E402
from structsplat.overlap_elimination import (  # noqa: E402
    AppearanceSolveConfig,
    ProtectedLeafSelection,
    lattice_observation_field,
    select_protected_feature_leaves,
    solve_fixed_lattice_appearance,
)
from structsplat.pixel_contraction import (  # noqa: E402
    PixelContractionConfig,
    contract_image,
    render_observation_field,
)


REPORT_SCHEMA = "structsplat.hier009_dynamic_overlap_recovery.diagnostic.v1"
ARMS = (
    "delta_touched",
    "overlap_touched",
    "overlap_halo",
    "overlap_halo_protected",
)


def _snapshot_sources(output_root: Path) -> list[dict[str, object]]:
    sources = (
        Path(__file__).resolve(),
        ROOT / "src" / "structsplat" / "pixel_contraction.py",
        ROOT / "src" / "structsplat" / "overlap_elimination.py",
        ROOT / "src" / "structsplat" / "structure_tensor.py",
        ROOT / "scripts" / "experiments" / "hier005_pixel_contraction.py",
        ROOT / "tasks" / "HIER-009-dynamic-overlap-neighborhood-recovery.md",
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


def _save_scalar(path: Path, values: np.ndarray, mask: np.ndarray) -> None:
    shown = np.zeros((*mask.shape, 3), dtype=np.float32)
    clipped = np.where(mask, np.clip(values, 0.0, 1.0), 0.0)
    shown[..., 0] = clipped
    shown[..., 1] = clipped
    shown[..., 2] = clipped
    save_image(str(path), shown)


def _save_centers(
    path: Path,
    source: np.ndarray,
    mask: np.ndarray,
    means: np.ndarray,
    *,
    color: tuple[float, float, float] = (0.0, 1.0, 0.2),
) -> None:
    shown = np.asarray(source * mask[:, :, None], dtype=np.float32).copy()
    height, width = mask.shape
    for x_value, y_value in np.rint(means).astype(np.int64):
        x = int(np.clip(x_value, 0, width - 1))
        y = int(np.clip(y_value, 0, height - 1))
        shown[max(0, y - 1) : min(height, y + 2), x, :] = color
        shown[y, max(0, x - 1) : min(width, x + 2), :] = color
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


def _feature_geometry_metrics(
    feature: np.ndarray,
    mask: np.ndarray,
    means: np.ndarray,
) -> dict[str, float]:
    from scipy.spatial import cKDTree

    yy, xx = np.nonzero(mask)
    points = np.stack([xx, yy], axis=1).astype(np.float64)
    distances = cKDTree(np.asarray(means, dtype=np.float64)).query(points, workers=1)[0]
    values = feature[mask]
    threshold = float(np.quantile(values, 0.90))
    top = values >= threshold
    return {
        "nearest_center_distance_q99": float(np.quantile(distances, 0.99)),
        "nearest_center_distance_max": float(np.max(distances)),
        "top_feature_coverage_within_1_5px": float(np.mean(distances[top] <= 1.5)),
    }


def _canonical_raw_bytes(field: ObservationField2D) -> int:
    return int(sum(array.nbytes for array in field._array_items().values()))


def _psnr_from_sse(sse: float, active_pixels: int) -> float:
    mse = float(sse) / max(3 * active_pixels, 1)
    return float(10.0 * math.log10(1.0 / max(mse, 1e-12)))


def _plot_curves(
    output_root: Path,
    rows: list[dict[str, object]],
    checkpoints: list[dict[str, object]],
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
        "series",
        "artifact_dir",
        "source_path",
        "source_sha256",
        "mask_source_path",
        "mask_source_sha256",
        "field_file_sha256",
        "stop_reason",
        "lpips_error",
        "artifact_metric_domain",
        "recovery_scope",
        "prefit_kind",
    }

    def metric_names(data: list[dict[str, object]], x_name: str) -> list[str]:
        result: set[str] = set()
        for row in data:
            for name, value in row.items():
                if name in excluded or name == x_name or isinstance(value, bool):
                    continue
                if isinstance(value, (int, float)) and math.isfinite(float(value)):
                    result.add(name)
        return sorted(result)

    palette = ("#1769aa", "#d97706", "#16825d", "#9c3aa5", "#c33d4c", "#59636e")
    for kind, data, x_name, group_name in (
        ("snapshot", rows, "n_gaussians", "arm"),
        ("recovery", checkpoints, "checkpoint_index", "series"),
    ):
        for metric in metric_names(data, x_name):
            figure, axis = plt.subplots(figsize=(6.4, 3.7), constrained_layout=True)
            plotted = False
            groups = sorted({str(row[group_name]) for row in data})
            for index, group in enumerate(groups):
                points = sorted(
                    (
                        (float(row[x_name]), float(row[metric]))
                        for row in data
                        if str(row[group_name]) == group
                        and isinstance(row.get(metric), (int, float))
                        and not isinstance(row.get(metric), bool)
                        and math.isfinite(float(row[metric]))
                    ),
                    key=lambda value: value[0],
                )
                if not points:
                    continue
                x, y = zip(*points)
                axis.plot(
                    x,
                    y,
                    marker="o" if kind == "snapshot" else ".",
                    linewidth=1.6,
                    label=group,
                    color=palette[index % len(palette)],
                )
                plotted = True
            if not plotted:
                plt.close(figure)
                continue
            if kind == "snapshot":
                axis.set_xscale("log", base=2)
            axis.set_xlabel(x_name.replace("_", " "))
            axis.set_ylabel(metric.replace("_", " "))
            axis.set_title(f"{kind} {metric.replace('_', ' ')}")
            axis.grid(True, alpha=0.28)
            axis.legend(fontsize=6, ncol=2)
            path = curve_root / f"{kind}__{metric}.svg"
            figure.savefig(path, format="svg")
            plt.close(figure)
            records.append(
                {"kind": kind, "metric": metric, "path": str(path.relative_to(output_root))}
            )
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
        "rate_warning": "All field byte values are uncoded references, not a complete codec.",
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


def _write_report(
    output_root: Path,
    rows: list[dict[str, object]],
    curves: list[dict[str, object]],
) -> None:
    table_rows: list[str] = []
    cards: list[str] = []
    artifact_links: list[str] = []
    for row in sorted(rows, key=lambda value: (str(value["arm"]), -int(value["n_gaussians"]))):
        artifact = str(row["artifact_dir"])
        gate = "PASS" if row["artifact_gate_pass"] else "FAIL"
        lpips_text = (
            "n/a" if row.get("lpips") is None else f"{float(row['lpips']):.6f}"
        )
        table_rows.append(
            "<tr>"
            f"<td>{escape(str(row['arm']))}</td><td>{int(row['n_gaussians']):,}</td>"
            f"<td>{float(row['reduction_factor']):.2f}×</td>"
            f"<td>{float(row['psnr_db']):.3f}</td>"
            f"<td>{float(row['ms_ssim']):.6f}</td>"
            f"<td>{lpips_text}</td>"
            f"<td>{float(row['artifact_pixel_rmse_max']):.5f}</td>"
            f"<td>{float(row['artifact_patch_rmse_max_7']):.5f}</td>"
            f"<td class='{gate.lower()}'>{gate}</td>"
            f"<td>{int(row['recovery_accepted_checkpoints'])}/"
            f"{int(row['recovery_checkpoints'])}</td>"
            f"<td>{float(row['recovery_checkpoint_psnr_gain_sum_db']):.3f}</td>"
            f"<td>{int(row['recovery_optimized_rows_max'])}</td>"
            f"<td>{int(row['protected_active_rows'])}</td>"
            "</tr>"
        )
        title = f"{row['arm']} N={row['n_gaussians']}"
        cards.append(
            f"<section class='card'><h3>{escape(title)}</h3>"
            f"<p>PSNR {float(row['psnr_db']):.3f} dB · gate {gate} · recovery "
            f"{int(row['recovery_accepted_checkpoints'])}/{int(row['recovery_checkpoints'])} · "
            f"optimized up to {int(row['recovery_optimized_rows_max'])} rows · protected "
            f"{int(row['protected_active_rows'])}</p><div class='images'>"
            f"<figure><img src='{artifact}/source.png'><figcaption>source</figcaption></figure>"
            f"<figure><img src='{artifact}/initial_lattice.png'><figcaption>initial lattice</figcaption></figure>"
            f"<figure><img src='{artifact}/reconstruction.png'><figcaption>result</figcaption></figure>"
            f"<figure><img src='{artifact}/error.png'><figcaption>error ×4</figcaption></figure>"
            f"<figure><img src='{artifact}/feature_priority.png'><figcaption>feature priority</figcaption></figure>"
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
    document = f"""<!doctype html><html><head><meta charset='utf-8'>
<title>HIER-009 dynamic overlap recovery</title><style>
body{{font-family:system-ui,sans-serif;margin:0;background:#f3f6f8;color:#18242d}}
main{{max-width:1500px;margin:auto;padding:24px}}.warning{{background:#fff3cd;border:1px solid #e0ba54;padding:12px;border-radius:7px}}
table{{border-collapse:collapse;width:100%;background:white;font-size:13px}}th,td{{border:1px solid #dce3e8;padding:7px;text-align:right}}th:first-child,td:first-child{{text-align:left}}
.pass{{color:#087a48;font-weight:700}}.fail{{color:#b62929;font-weight:700}}.card{{background:white;border:1px solid #dce3e8;border-radius:9px;padding:14px;margin:18px 0}}
.images{{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:10px}}figure{{margin:0}}img{{width:100%;height:auto;background:#101417}}figcaption{{font-size:12px;color:#52616c}}.links{{columns:2;font-size:12px}}
</style></head><body><main><h1>HIER-009: dynamic overlap + 3×3 recovery halo</h1>
<p class='warning'><strong>Diagnostic only.</strong> One exposed resized C0001 raster, one CUDA trajectory, and a dirty source tree. Attempted optimizer steps are matched but optimized-row work is not. Field bytes are not codec bytes; all negative cells are retained.</p>
<p><a href='manifest.json'>manifest</a> · <a href='metrics.json'>metrics JSON</a> · <a href='metrics.jsonl'>JSONL</a> · <a href='metrics.csv'>CSV</a> · <a href='config.json'>config</a> · <a href='curves/catalog.json'>curves</a></p>
<h2>Outcomes</h2><table><thead><tr><th>arm</th><th>N</th><th>reduction</th><th>PSNR</th><th>MS-SSIM</th><th>LPIPS</th><th>pixel max</th><th>7×7 max</th><th>gate</th><th>accepted opt</th><th>Σ checkpoint ΔPSNR</th><th>max opt rows</th><th>protected</th></tr></thead><tbody>{''.join(table_rows)}</tbody></table>
<h2>Visuals</h2>{''.join(cards)}<h2>All snapshot and recovery curves</h2><ul class='links'>{curve_links}</ul>
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
    parser.add_argument("--images", type=Path, nargs="+", required=True)
    parser.add_argument("--mask", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-side", type=int, default=512)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--target-gaussians", type=int, nargs="+", required=True)
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    parser.add_argument("--delta-scale", type=float, default=0.18)
    parser.add_argument("--overlap-scale", type=float, default=0.50)
    parser.add_argument("--cg-tolerance", type=float, default=1e-8)
    parser.add_argument("--cg-max-iterations", type=int, default=200)
    parser.add_argument("--cg-ridge", type=float, default=1e-8)
    parser.add_argument("--protected-fraction", type=float, default=0.05)
    parser.add_argument("--protected-highpass-sigma", type=float, default=1.0)
    parser.add_argument("--protected-nms-radius", type=int, default=1)
    parser.add_argument("--proposal-batch-size", type=int, default=64)
    parser.add_argument("--merge-batch-size", type=int, default=8)
    parser.add_argument("--pair-shortlist", type=int, default=3)
    parser.add_argument("--exact-option-shortlist", type=int, default=2)
    parser.add_argument("--recovery-checkpoints", type=int, default=16)
    parser.add_argument("--recovery-steps", type=int, default=50)
    parser.add_argument("--lr-coefficients", type=float, default=0.003)
    parser.add_argument("--lr-means", type=float, default=0.005)
    parser.add_argument("--lr-scales", type=float, default=0.003)
    parser.add_argument("--lr-rotations", type=float, default=0.001)
    parser.add_argument("--max-mean-shift", type=float, default=1.5)
    parser.add_argument("--max-log-scale-shift", type=float, default=0.35)
    parser.add_argument("--max-rotation-shift", type=float, default=0.35)
    parser.add_argument("--sigma-cutoff", type=float, default=3.0)
    parser.add_argument("--support-fade-alpha", type=float, default=0.0)
    parser.add_argument("--estimated-row-bytes", type=int, default=32)
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


def _arm_values(
    arm: str,
    args: argparse.Namespace,
) -> tuple[float, bool, str, bool]:
    if arm == "delta_touched":
        return args.delta_scale, False, "touched", False
    if arm == "overlap_touched":
        return args.overlap_scale, True, "touched", False
    if arm == "overlap_halo":
        return args.overlap_scale, True, "touched_neighborhood", False
    if arm == "overlap_halo_protected":
        return args.overlap_scale, True, "touched_neighborhood", True
    raise ValueError(f"unknown arm {arm!r}")


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if len(set(args.arms)) != len(args.arms):
        raise SystemExit("--arms must be unique")
    if any(value <= 0 for value in args.target_gaussians):
        raise SystemExit("all target counts must be positive")
    if not 0.0 <= args.protected_fraction < 1.0:
        raise SystemExit("--protected-fraction must lie in [0, 1)")
    output_root = args.out.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise SystemExit(f"output directory must be absent or empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    images = report_utils._discover_images(args.images)
    command = shlex.join(
        [sys.executable, str(Path(__file__).relative_to(ROOT)), *(argv or sys.argv[1:])]
    )
    source_snapshot = _snapshot_sources(output_root)
    solve_config = AppearanceSolveConfig(
        tolerance=args.cg_tolerance,
        max_iterations=args.cg_max_iterations,
        ridge=args.cg_ridge,
    )
    config = {
        "schema": REPORT_SCHEMA,
        "task": "HIER-009",
        "status": "diagnostic",
        "claim_ready": False,
        "command": command,
        "args": vars(args),
        "git": report_utils._git_record(),
        "executed_source_snapshot": source_snapshot,
        "appearance_solve": asdict(solve_config),
        "versions": {"python": platform.python_version(), "numpy": np.__version__},
        "evidence_limits": [
            "Exposed single resized image and one CUDA trajectory.",
            "Dirty-source diagnostic without prospective distinct review.",
            "Attempted optimizer steps are matched, optimized-row FLOPs are not.",
            "Estimated/raw/NPZ field bytes are not complete codec bytes.",
        ],
    }
    report_utils._write_json(output_root / "config.json", config)

    rows: list[dict[str, object]] = []
    checkpoint_rows: list[dict[str, object]] = []
    run_started = time.perf_counter()
    for image_path in images:
        image, loaded_mask, raster_record = report_utils._load_evaluation_raster(
            image_path,
            args.mask.resolve(),
            max_side=args.max_side,
            mask_threshold=args.mask_threshold,
        )
        if loaded_mask is None:
            raise RuntimeError("HIER-009 requires a mask")
        mask = loaded_mask
        target = image * mask[:, :, None]
        active_pixels = int(mask.sum())
        if max(args.target_gaussians) > active_pixels:
            raise ValueError("a target count exceeds the active mask count")
        overlap_coefficients, overlap_initial_raw, overlap_solve = solve_fixed_lattice_appearance(
            image,
            mask,
            mask,
            scale_px=args.overlap_scale,
            sigma_cutoff=args.sigma_cutoff,
            support_fade_alpha=args.support_fade_alpha,
            config=solve_config,
        )
        delta_coefficients = image[mask].astype(np.float32)
        delta_field = lattice_observation_field(
            mask,
            mask,
            delta_coefficients,
            scale_px=args.delta_scale,
            sigma_cutoff=args.sigma_cutoff,
            support_fade_alpha=args.support_fade_alpha,
        )
        delta_initial_raw = render_observation_field(
            delta_field,
            device=args.device,
            renderer=args.renderer,
            render_chunk=args.render_chunk,
            apply_declared_alpha=False,
        )
        feature_only = select_protected_feature_leaves(
            image,
            mask,
            0,
            highpass_sigma_px=args.protected_highpass_sigma,
            nms_radius_px=args.protected_nms_radius,
        )
        for arm in args.arms:
            scale, use_prefit, recovery_scope, use_protection = _arm_values(arm, args)
            for target_count in sorted(set(args.target_gaussians), reverse=True):
                cell_started = time.perf_counter()
                protected_count = (
                    int(round(args.protected_fraction * target_count))
                    if use_protection
                    else 0
                )
                selection: ProtectedLeafSelection = (
                    select_protected_feature_leaves(
                        image,
                        mask,
                        protected_count,
                        highpass_sigma_px=args.protected_highpass_sigma,
                        nms_radius_px=args.protected_nms_radius,
                    )
                    if use_protection
                    else feature_only
                )
                initial_coefficients = (
                    overlap_coefficients if use_prefit else delta_coefficients
                )
                initial_raw = overlap_initial_raw if use_prefit else delta_initial_raw
                contraction_config = PixelContractionConfig(
                    target_gaussians=target_count,
                    leaf_scale_px=scale,
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
                    recovery_scope=recovery_scope,
                    recovery_schedule="progress",
                    recovery_progress_checkpoints=args.recovery_checkpoints,
                    recovery_neighborhood_radius_px=1,
                    recovery_device=args.device,
                    recovery_renderer=args.renderer,
                    recovery_render_chunk=args.render_chunk,
                    recovery_lr_means=args.lr_means,
                    recovery_lr_scales=args.lr_scales,
                    recovery_lr_rotations=args.lr_rotations,
                    recovery_lr_coefficients=args.lr_coefficients,
                    recovery_max_mean_shift_px=args.max_mean_shift,
                    recovery_max_log_scale_shift=args.max_log_scale_shift,
                    recovery_max_rotation_shift_rad=args.max_rotation_shift,
                )
                contraction = contract_image(
                    image,
                    contraction_config,
                    mask=mask,
                    initial_coefficients=initial_coefficients,
                    protected_leaf_mask=(
                        selection.protected_mask if use_protection else None
                    ),
                )
                artifact_key = f"{image_path.stem}__{arm}__n{target_count}"
                artifact_dir = output_root / "artifacts" / artifact_key
                artifact_dir.mkdir(parents=True, exist_ok=False)
                field_path = artifact_dir / "field.observation.npz"
                contraction.field.save_lossless(field_path)
                lossless_bytes = field_path.stat().st_size
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
                maintained_parity = float(np.max(np.abs(cold - contraction.reconstruction)))
                repeated_parity = float(np.max(np.abs(repeated - cold)))
                metrics = report_utils._metric_values(
                    cold,
                    target,
                    mask,
                    device=args.device,
                    compute_lpips=args.lpips,
                )
                geometry_metrics = _feature_geometry_metrics(
                    selection.priority,
                    mask,
                    cold_field.means_xy,
                )
                protected_means = np.stack(
                    np.nonzero(selection.protected_mask)[::-1], axis=1
                ).astype(np.float32)
                protected_geometry_error = 0.0
                if protected_means.size:
                    from scipy.spatial import cKDTree

                    protected_geometry_error = float(
                        np.max(cKDTree(cold_field.means_xy).query(protected_means, workers=1)[0])
                    )

                initial_display = initial_raw * mask[:, :, None]
                save_image(str(artifact_dir / "source.png"), target)
                save_image(str(artifact_dir / "initial_lattice.png"), initial_display)
                save_error_heatmap(
                    str(artifact_dir / "initial_error.png"),
                    initial_display - target,
                    scale=args.error_scale,
                )
                save_image(str(artifact_dir / "reconstruction.png"), cold)
                save_error_heatmap(
                    str(artifact_dir / "error.png"), cold - target, scale=args.error_scale
                )
                _save_scalar(
                    artifact_dir / "feature_priority.png", selection.priority, mask
                )
                _save_centers(
                    artifact_dir / "protected.png",
                    image,
                    mask,
                    protected_means,
                    color=(1.0, 0.0, 0.2),
                )
                _save_centers(
                    artifact_dir / "centers.png", image, mask, cold_field.means_xy
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
                np.savez_compressed(
                    artifact_dir / "analysis.npz",
                    feature_priority=selection.priority,
                    structure_feature=selection.structure_feature,
                    highpass_feature=selection.highpass_feature,
                    protected_mask=selection.protected_mask,
                    crop_bounds=np.asarray(crop_bounds, dtype=np.int32),
                )
                report_utils._write_json(
                    artifact_dir / "config.json",
                    {
                        "schema": REPORT_SCHEMA,
                        "status": "diagnostic",
                        "arm": arm,
                        "target_count": target_count,
                        "contraction": asdict(contraction_config),
                        "selection": {
                            "requested_count": selection.requested_count,
                            "nms_selected_count": selection.nms_selected_count,
                            "protected_fraction": args.protected_fraction,
                            "highpass_sigma_px": args.protected_highpass_sigma,
                            "nms_radius_px": args.protected_nms_radius,
                        },
                    },
                )
                source_bytes = image_path.stat().st_size
                evaluation_png_bytes = (artifact_dir / "source.png").stat().st_size
                canonical_bytes = _canonical_raw_bytes(cold_field)
                alpha_bytes = 0 if cold_field.packed_alpha is None else cold_field.packed_alpha.nbytes
                estimated_bytes = cold_field.n * args.estimated_row_bytes + alpha_bytes
                recovery_gain_db = sum(
                    10.0 * math.log10(event.sse_before / event.sse_after)
                    for event in contraction.recovery_history
                    if event.accepted and event.sse_after > 0.0
                )
                row: dict[str, object] = {
                    "schema": REPORT_SCHEMA,
                    "status": "diagnostic",
                    "image": image_path.stem,
                    "arm": arm,
                    "artifact_dir": str(artifact_dir.relative_to(output_root)),
                    "source_path": str(image_path),
                    "source_sha256": report_utils._sha256(image_path),
                    "source_file_bytes": source_bytes,
                    "mask_source_path": str(args.mask.resolve()),
                    "mask_source_sha256": report_utils._sha256(args.mask.resolve()),
                    "original_width": raster_record["original_width"],
                    "original_height": raster_record["original_height"],
                    "width": image.shape[1],
                    "height": image.shape[0],
                    "pixels": int(mask.size),
                    "active_pixels": active_pixels,
                    "target_gaussians": target_count,
                    "n_gaussians": cold_field.n,
                    "reduction_factor": active_pixels / cold_field.n,
                    "stop_reason": contraction.stop_reason,
                    "prefit_kind": "exact_overlap_pcg" if use_prefit else "source_rgb_delta",
                    "support_scale_px": scale,
                    "initial_lattice_sse": float(
                        np.sum(np.square((initial_raw - image)[mask], dtype=np.float64))
                    ),
                    "prefit_iterations": overlap_solve.iterations if use_prefit else 0,
                    "prefit_converged": overlap_solve.converged if use_prefit else True,
                    "prefit_coefficient_abs_max": (
                        overlap_solve.coefficient_abs_max
                        if use_prefit
                        else float(np.max(np.abs(delta_coefficients)))
                    ),
                    "contraction_actions": len(contraction.history),
                    "contraction_seconds": contraction.elapsed_seconds,
                    "recovery_scope": recovery_scope,
                    "recovery_checkpoints": len(contraction.recovery_history),
                    "recovery_accepted_checkpoints": sum(
                        event.accepted for event in contraction.recovery_history
                    ),
                    "recovery_checkpoint_psnr_gain_sum_db": recovery_gain_db,
                    "recovery_sse_gain_sum": sum(
                        event.sse_before - event.sse_after
                        for event in contraction.recovery_history
                    ),
                    "recovery_optimized_rows_max": max(
                        (event.optimized_count for event in contraction.recovery_history),
                        default=0,
                    ),
                    "recovery_neighborhood_rows_max": max(
                        (event.neighborhood_count for event in contraction.recovery_history),
                        default=0,
                    ),
                    "recovery_accepted_new_neighbors": sum(
                        event.accepted_new_neighborhood_count
                        for event in contraction.recovery_history
                    ),
                    "recovery_neighbor_active_rows": contraction.recovery_neighbor_active_rows,
                    "protected_requested_rows": protected_count,
                    "protected_initial_rows": contraction.protected_initial_rows,
                    "protected_active_rows": contraction.protected_active_rows,
                    "protected_nms_selected_rows": selection.nms_selected_count,
                    "protected_geometry_error_max_px": protected_geometry_error,
                    "blocked_regions": contraction.blocked_regions,
                    "final_sse": contraction.final_sse,
                    "estimated_field_bytes": int(estimated_bytes),
                    "canonical_raw_bytes": canonical_bytes,
                    "lossless_reference_bytes": lossless_bytes,
                    "estimated_bits_per_pixel": 8.0 * estimated_bytes / mask.size,
                    "source_over_estimated_ratio": source_bytes / estimated_bytes,
                    "evaluation_png_over_estimated_ratio": evaluation_png_bytes / estimated_bytes,
                    "cold_decode_seconds": cold_decode_seconds,
                    "render_seconds": render_seconds,
                    "total_seconds": time.perf_counter() - cell_started,
                    "maintained_render_parity_max_abs": maintained_parity,
                    "repeated_render_parity_max_abs": repeated_parity,
                    "field_file_sha256": report_utils._sha256(field_path),
                    **geometry_metrics,
                    **metrics,
                }
                report_utils._write_json(artifact_dir / "row.json", row)
                rows.append(row)
                for event in contraction.recovery_history:
                    checkpoint_rows.append(
                        {
                            "schema": REPORT_SCHEMA,
                            "series": f"{arm}__n{target_count}",
                            "arm": arm,
                            "n_gaussians": target_count,
                            "checkpoint_index": event.checkpoint_index,
                            "action_count": event.action_count,
                            "active_count": event.active_count,
                            "psnr_before_db": _psnr_from_sse(event.sse_before, active_pixels),
                            "psnr_after_db": _psnr_from_sse(event.sse_after, active_pixels),
                            **event.to_record(),
                        }
                    )
                print(
                    f"{arm} N={target_count}: {row['psnr_db']:.3f} dB, "
                    f"pixel/patch7={row['artifact_pixel_rmse_max']:.5f}/"
                    f"{row['artifact_patch_rmse_max_7']:.5f}, {contraction.stop_reason}",
                    flush=True,
                )

    report_utils._write_json(
        output_root / "recovery_checkpoints.json",
        {"schema": REPORT_SCHEMA, "rows": checkpoint_rows},
    )
    _write_tables(output_root, rows)
    curves = _plot_curves(output_root, rows, checkpoint_rows)
    _write_report(output_root, rows, curves)
    config["elapsed_seconds"] = time.perf_counter() - run_started
    report_utils._write_json(output_root / "config.json", config)
    _write_manifest(output_root)
    print(f"wrote diagnostic report: {output_root / 'index.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
