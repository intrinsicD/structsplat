#!/usr/bin/env python3
"""Package the HIER-012 global safeguarded appearance-projection diagnostic."""
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
from scripts.experiments import hier010_residual_anchor_projection as viz_utils  # noqa: E402
from structsplat.cli import save_error_heatmap, save_image  # noqa: E402
from structsplat.contraction_refinement import (  # noqa: E402
    CoefficientProjectionConfig,
    CoefficientProjectionResult,
    project_contracted_coefficients,
)
from structsplat.observation_field import ObservationField2D  # noqa: E402
from structsplat.pixel_contraction import render_observation_field  # noqa: E402


REPORT_SCHEMA = "structsplat.hier012_global_appearance_projection.diagnostic.v1"
HIER010_MANIFEST_SHA256 = "80b84bce9b5ec72e9369fd61474d761c8ecd3f2a9f6ed9495f7cb67f14dd81ba"
HIER011_MANIFEST_SHA256 = "c15d18ee3b1eca4782c4400e2f94ffe35dca6a0b383ba960c6260d396c849bf9"
ARMS = (
    "h005_control",
    "touched_projection",
    "guarded_exchange",
    "exchange_global_projection",
    "global_projection",
)
EXPECTED = {
    "C0001": {
        "rgb": "ae24fe99d3f8edbd04cd2c85ebc4fe9bfd95abe878c22abb7691cadcfc5c411b",
        "mask": "94dcbf7005dbeb1d183e259a569d783aa5df900255e763385bed91f02d3b80c3",
        "base_file": "cfa05c3cc5bfe5f747e14bae2cfe254283593123a4eb92177f73b95a071d1dac",
        "base_canonical": "9bfbe941b90bac66a7c6ce3166fffcd76224520f40edc0800a9de4a1ea9cfb5b",
        "touched_file": "b15cadb6b5211dcd0cb70cf074534a8f3e8d7650b47cb3ac5f98d656c07a7595",
        "touched_canonical": "21a93aef6d249eb788a29a54d246d118031929d41bd4557e7059e78aa5c9a59c",
        "exchange_file": "7a0260beed61742dad65beb7fe951a3dbe17be356bee73909e5ebf907819433c",
        "exchange_canonical": "8bc2b7b1896912335164ca19331e9885f7dd873dbe4ea53b3ca9fd995e9909f3",
    },
    "C0004": {
        "rgb": "26eb4cf24a034eb830198df6e7a6ac409ccb7cf4814ff645c71d0b6966b7070e",
        "mask": "4702bfa9df354f38e35a63207a37d4ec1b753afc4d0668bd905f3cdab320f35d",
        "base_file": "6743d9141791c8932532708085815157e7c918ee398c1ab5fbcbc0342b111f5b",
        "base_canonical": "8c4406059c5bc68254fd2b16740019ae550230aa6778369f99ac9d606e9635e8",
        "touched_file": "7404c470ba90b806ebff5e9cd20eb623ab62f30dcff12e5baca3c3ca72f9da3c",
        "touched_canonical": "177987a89c891d278ed84d5458fe350abbd1e6343a0c91fcf695842b0c952ac6",
        "exchange_file": "69bfd813a4edd4fe0d547413d7355f5af02d4178db7733436f20619c9d7fd0da",
        "exchange_canonical": "385961330433cf4d3175bfccc12429a527ce236efc0a9a06e825e7eb062794ca",
    },
}


def _snapshot_sources(output_root: Path) -> list[dict[str, object]]:
    sources = (
        Path(__file__).resolve(),
        ROOT / "src" / "structsplat" / "contraction_refinement.py",
        ROOT / "src" / "structsplat" / "pixel_contraction.py",
        ROOT / "scripts" / "experiments" / "hier005_pixel_contraction.py",
        ROOT / "scripts" / "experiments" / "hier010_residual_anchor_projection.py",
        ROOT / "scripts" / "check_report_bundle.py",
        ROOT / "tasks" / "HIER-012-global-appearance-projection.md",
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
        figure, axis = plt.subplots(figsize=(8.2, 3.8), constrained_layout=True)
        for image_name in ("C0001", "C0004"):
            points = sorted(
                (row for row in rows if row["image"] == image_name),
                key=lambda row: arm_x[str(row["arm"])],
            )
            axis.plot(
                [arm_x[str(row["arm"])] for row in points],
                [float(row[metric]) for row in points],
                marker="o",
                label=image_name,
            )
        axis.set_xticks(range(len(ARMS)), ARMS, rotation=18, ha="right")
        axis.set_ylabel(metric.replace("_", " "))
        axis.grid(True, alpha=0.28)
        axis.legend()
        path = curve_root / f"arms__{metric}.svg"
        figure.savefig(path, format="svg")
        plt.close(figure)
        records.append({"kind": "arms", "metric": metric, "path": str(path.relative_to(output_root))})

    for metric in ("raw_sse", "display_pixel_rmse_max", "display_patch7_rmse_max"):
        figure, axis = plt.subplots(figsize=(8.2, 3.8), constrained_layout=True)
        for series in sorted({str(row["series"]) for row in projection_rows}):
            points = sorted(
                (row for row in projection_rows if row["series"] == series),
                key=lambda row: int(row["iteration"]),
            )
            axis.plot(
                [int(row["iteration"]) for row in points],
                [float(row[metric]) for row in points],
                linewidth=1.0,
                label=series,
            )
            selected = [row for row in points if row["selected"]]
            if selected:
                axis.scatter(
                    [int(selected[0]["iteration"])],
                    [float(selected[0][metric])],
                    marker="*",
                    s=45,
                    zorder=4,
                )
        axis.set_xlabel("PCG iteration")
        axis.set_ylabel(metric.replace("_", " "))
        axis.grid(True, alpha=0.28)
        axis.legend(fontsize=6, ncol=2)
        path = curve_root / f"projection__{metric}.svg"
        figure.savefig(path, format="svg")
        plt.close(figure)
        records.append({"kind": "projection", "metric": metric, "path": str(path.relative_to(output_root))})
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
        selected = "SELECTED" if row["arm"] == "global_projection" else "—"
        lpips_text = "n/a" if row["lpips"] is None else f"{float(row['lpips']):.8f}"
        table_rows.append(
            "<tr>"
            f"<td>{escape(str(row['image']))}</td><td>{escape(str(row['arm']))}</td>"
            f"<td>{int(row['n_gaussians']):,}</td><td>{float(row['psnr_db']):.4f}</td>"
            f"<td>{float(row['psnr_delta_vs_control_db']):+.4f}</td>"
            f"<td>{float(row['masked_mse']):.8g}</td>"
            f"<td>{float(row['ms_ssim']):.7f}</td><td>{lpips_text}</td>"
            f"<td>{float(row['artifact_pixel_rmse_max']):.5f}</td>"
            f"<td>{float(row['artifact_patch_rmse_max_7']):.5f}</td>"
            f"<td class='{gate.lower()}'>{gate}</td>"
            f"<td>{escape(str(row['projection_scope']))}</td>"
            f"<td>{int(row['projection_selected_iteration'])}</td>"
            f"<td>{float(row['pipeline_cumulative_seconds']):.2f}</td>"
            f"<td>{selected}</td></tr>"
        )
        artifact = str(row["artifact_dir"])
        title = f"{row['image']} / {row['arm']}"
        cards.append(
            f"<section class='card'><h3>{escape(title)}</h3>"
            f"<p>{float(row['psnr_db']):.4f} dB ({float(row['psnr_delta_vs_control_db']):+.4f}); "
            f"pixel/7×7 {float(row['artifact_pixel_rmse_max']):.5f}/"
            f"{float(row['artifact_patch_rmse_max_7']):.5f}; projection "
            f"{escape(str(row['projection_scope']))} step "
            f"{int(row['projection_selected_iteration'])}; gate {gate}</p><div class='images'>"
            f"<figure><img src='{artifact}/source.png'><figcaption>source</figcaption></figure>"
            f"<figure><img src='{artifact}/reconstruction.png'><figcaption>result</figcaption></figure>"
            f"<figure><img src='{artifact}/error.png'><figcaption>error ×4</figcaption></figure>"
            f"<figure><img src='{artifact}/feature_priority.png'><figcaption>HIER-005 residual energy</figcaption></figure>"
            f"<figure><img src='{artifact}/protected.png'><figcaption>exchange geometry changes</figcaption></figure>"
            f"<figure><img src='{artifact}/centers.png'><figcaption>fixed centers</figcaption></figure>"
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
    verdict = "PASS" if decision["selected_pipeline_pass"] else "FAIL"
    color = "#e4f6ec" if decision["selected_pipeline_pass"] else "#fee9e7"
    border = "#2e8b57" if decision["selected_pipeline_pass"] else "#b64335"
    document = f"""<!doctype html><html><head><meta charset='utf-8'>
<meta name='viewport' content='width=device-width,initial-scale=1'>
<title>HIER-012 global safeguarded appearance projection</title><style>
body{{font-family:system-ui,sans-serif;margin:0;background:#f3f6f8;color:#18242d}}main{{max-width:1550px;margin:auto;padding:24px}}
.warning{{background:#fff3cd;border:1px solid #e0ba54;padding:12px;border-radius:8px}}.verdict{{background:{color};border:1px solid {border};padding:16px;border-radius:8px}}
table{{border-collapse:collapse;width:100%;background:white;font-size:12px}}th,td{{border:1px solid #dce3e8;padding:7px;text-align:right}}th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){{text-align:left}}
.pass{{color:#087a48;font-weight:700}}.fail{{color:#b62929;font-weight:700}}.card{{background:white;border:1px solid #dce3e8;border-radius:9px;padding:14px;margin:18px 0}}
.images{{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:10px}}figure{{margin:0}}img{{width:100%;height:auto;background:#101417}}figcaption{{font-size:12px;color:#52616c}}.links{{columns:2;font-size:12px}}
</style></head><body><main><p>STRUCTSPLAT / HIER-012 / EXPOSED SUCCESSOR DIAGNOSTIC</p>
<h1>Let every retained Gaussian explain its color</h1>
<p class='warning'><strong>Development evidence only.</strong> Both correlated Janelle views informed this successor choice. The run is dirty-source snapshotted and lacks distinct prospective review. Exact N is matched, work is not, and reference bytes are not codec rate. No default or FIT-046 decision follows.</p>
<section class='verdict'><h2>Selected pipeline integrity/effect gate: {verdict}</h2><p>{escape(str(decision['summary']))}</p></section>
<p><a href='manifest.json'>manifest</a> · <a href='metrics.json'>metrics JSON</a> · <a href='metrics.jsonl'>JSONL</a> · <a href='metrics.csv'>CSV</a> · <a href='config.json'>config</a> · <a href='decision.json'>decision</a> · <a href='projection_checkpoints.json'>projection checkpoints</a> · <a href='curves/catalog.json'>curves</a></p>
<h2>Exact-7k attribution</h2><table><thead><tr><th>image</th><th>arm</th><th>N</th><th>PSNR</th><th>Δ control</th><th>MSE</th><th>MS-SSIM</th><th>LPIPS</th><th>pixel max</th><th>7×7 max</th><th>gate</th><th>scope</th><th>PCG</th><th>pipeline s</th><th>choice</th></tr></thead><tbody>{''.join(table_rows)}</tbody></table>
<h2>Visual audit</h2>{''.join(cards)}<h2>Curves</h2><ul class='links'>{curve_links}</ul>
<h2>Raw cell artifacts</h2><ul class='links'>{''.join(artifact_links)}</ul><h2>Executed source snapshot</h2><ul class='links'>{snapshots}</ul>
</main></body></html>"""
    (output_root / "index.html").write_text(document, encoding="utf-8")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", nargs="+", type=Path, required=True)
    parser.add_argument("--masks", nargs="+", type=Path, required=True)
    parser.add_argument("--hier010-root", type=Path, required=True)
    parser.add_argument("--hier011-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--projection-ridge", type=float, default=1e-8)
    parser.add_argument("--projection-tolerance", type=float, default=1e-6)
    parser.add_argument("--projection-max-iterations", type=int, default=48)
    parser.add_argument("--projection-coefficient-limit", type=float, default=16.0)
    parser.add_argument("--max-side", type=int, default=512)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
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


def _paths(
    hier010_root: Path, hier011_root: Path, image_name: str
) -> dict[str, Path]:
    h10 = hier010_root / "artifacts"
    h11 = hier011_root / "artifacts"
    return {
        "base": h10 / f"{image_name}__h005_control__n7000",
        "touched": h10 / f"{image_name}__control_projection__n7000",
        "exchange": h11 / f"{image_name}__guarded_exchange__n7000",
    }


def _validate_protocol(
    images: list[Path],
    masks: list[Path],
    hier010_root: Path,
    hier011_root: Path,
    args: argparse.Namespace,
) -> None:
    if len(images) != 2 or len(masks) != 2 or len(images) != len(masks):
        raise SystemExit("the HIER-012 diagnostic requires exactly two image/mask pairs")
    if [path.stem for path in images] != ["C0001", "C0004"]:
        raise SystemExit("the HIER-012 order is C0001 then C0004")
    if report_utils._sha256(hier010_root / "manifest.json") != HIER010_MANIFEST_SHA256:
        raise SystemExit("HIER-010 manifest hash mismatch")
    if report_utils._sha256(hier011_root / "manifest.json") != HIER011_MANIFEST_SHA256:
        raise SystemExit("HIER-011 manifest hash mismatch")
    for image, mask in zip(images, masks, strict=True):
        expected = EXPECTED[image.stem]
        if report_utils._sha256(image) != expected["rgb"]:
            raise SystemExit(f"source hash mismatch for {image}")
        if report_utils._sha256(mask) != expected["mask"]:
            raise SystemExit(f"mask hash mismatch for {mask}")
        paths = _paths(hier010_root, hier011_root, image.stem)
        for key, source in (
            ("base", paths["base"] / "field.observation.npz"),
            ("touched", paths["touched"] / "field.observation.npz"),
            ("exchange", paths["exchange"] / "field.observation.npz"),
        ):
            if report_utils._sha256(source) != expected[f"{key}_file"]:
                raise SystemExit(f"{key} field file hash mismatch for {image.stem}")
            field = ObservationField2D.load_lossless(source)
            if field.canonical_hash() != expected[f"{key}_canonical"]:
                raise SystemExit(f"{key} canonical hash mismatch for {image.stem}")
            if field.n != 7000:
                raise SystemExit(f"{key} field is not exact N=7,000 for {image.stem}")
    required = {
        "projection_ridge": 1e-8,
        "projection_tolerance": 1e-6,
        "projection_coefficient_limit": 16.0,
        "mask_threshold": 0.5,
        "error_scale": 4.0,
    }
    for name, value in required.items():
        if float(getattr(args, name)) != value:
            raise SystemExit(f"diagnostic requires --{name.replace('_', '-')} {value}")
    if args.projection_max_iterations != 48 or args.max_side != 512 or args.render_chunk != 256:
        raise SystemExit("diagnostic requires PCG 48, max-side 512, and render-chunk 256")
    if args.device != "cuda" or args.renderer != "cuda_additive" or not args.lpips:
        raise SystemExit("diagnostic requires --device cuda --renderer cuda_additive --lpips")


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    images = [path.resolve() for path in args.images]
    masks = [path.resolve() for path in args.masks]
    hier010_root = args.hier010_root.resolve()
    hier011_root = args.hier011_root.resolve()
    _validate_protocol(images, masks, hier010_root, hier011_root, args)
    output_root = args.out.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise SystemExit(f"output directory must be absent or empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    command = shlex.join(
        [sys.executable, str(Path(__file__).relative_to(ROOT)), *(argv or sys.argv[1:])]
    )
    source_snapshot = _snapshot_sources(output_root)
    projection_config = CoefficientProjectionConfig(
        ridge=args.projection_ridge,
        tolerance=args.projection_tolerance,
        max_iterations=args.projection_max_iterations,
        coefficient_abs_limit=args.projection_coefficient_limit,
    )

    import torch

    config: dict[str, object] = {
        "schema": REPORT_SCHEMA,
        "task": "HIER-012",
        "status": "diagnostic",
        "claim_ready": False,
        "command": command,
        "args": vars(args),
        "git": report_utils._git_record(),
        "executed_source_snapshot": source_snapshot,
        "input_bindings": EXPECTED,
        "hier010_manifest_sha256": HIER010_MANIFEST_SHA256,
        "hier011_manifest_sha256": HIER011_MANIFEST_SHA256,
        "projection": asdict(projection_config),
        "selection_disclosure": (
            "Both images were feasibility-probed before this packaging run; this is descriptive "
            "pipeline selection, not prospective confirmation."
        ),
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "cuda_device": torch.cuda.get_device_name(0),
        "evidence_limits": [
            "Both correlated views informed the successor choice.",
            "Dirty-source snapshot and no distinct prospective reviewer.",
            "CUDA accumulation is numerically, not bit, reproducible.",
            "Unequal work; no speed conclusion.",
            "Reference field bytes are not complete codec rate.",
            "No default, semantic, FIT-046, or BENCH-020 decision.",
        ],
    }
    report_utils._write_json(output_root / "config.json", config)

    rows: list[dict[str, object]] = []
    projection_rows: list[dict[str, object]] = []
    run_started = time.perf_counter()
    for image_path, mask_path in zip(images, masks, strict=True):
        image, loaded_mask, raster_record = report_utils._load_evaluation_raster(
            image_path,
            mask_path,
            max_side=args.max_side,
            mask_threshold=args.mask_threshold,
        )
        if loaded_mask is None:
            raise RuntimeError("HIER-012 requires masks")
        mask = loaded_mask
        target = image * mask[:, :, None]
        active_pixels = int(mask.sum())
        paths = _paths(hier010_root, hier011_root, image_path.stem)
        base_field = ObservationField2D.load_lossless(
            paths["base"] / "field.observation.npz"
        )
        touched_field = ObservationField2D.load_lossless(
            paths["touched"] / "field.observation.npz"
        )
        exchange_field = ObservationField2D.load_lossless(
            paths["exchange"] / "field.observation.npz"
        )
        base_row = json.loads((paths["base"] / "row.json").read_text(encoding="utf-8"))
        touched_row = json.loads(
            (paths["touched"] / "row.json").read_text(encoding="utf-8")
        )
        exchange_row = json.loads(
            (paths["exchange"] / "row.json").read_text(encoding="utf-8")
        )
        with np.load(paths["base"] / "analysis.npz", allow_pickle=False) as analysis:
            inherited_touched = np.array(analysis["touched_row_mask"], dtype=bool, copy=True)
        base_reconstruction = render_observation_field(
            base_field,
            device=args.device,
            renderer=args.renderer,
            render_chunk=args.render_chunk,
        )
        touched_reconstruction = render_observation_field(
            touched_field,
            device=args.device,
            renderer=args.renderer,
            render_chunk=args.render_chunk,
        )
        exchange_reconstruction = render_observation_field(
            exchange_field,
            device=args.device,
            renderer=args.renderer,
            render_chunk=args.render_chunk,
        )
        all_rows = np.ones(7000, dtype=bool)
        no_rows = np.zeros(7000, dtype=bool)
        torch.cuda.reset_peak_memory_stats()
        global_projection = project_contracted_coefficients(
            base_field,
            target,
            mask,
            all_rows,
            no_rows,
            config=projection_config,
            device=args.device,
            renderer=args.renderer,
            render_chunk=args.render_chunk,
        )
        exchange_global_projection = project_contracted_coefficients(
            exchange_field,
            target,
            mask,
            all_rows,
            no_rows,
            config=projection_config,
            device=args.device,
            renderer=args.renderer,
            render_chunk=args.render_chunk,
        )
        peak_cuda_bytes = int(torch.cuda.max_memory_allocated())
        for label, result in (
            ("global_projection", global_projection),
            ("exchange_global_projection", exchange_global_projection),
        ):
            for checkpoint in result.checkpoints:
                projection_rows.append(
                    {
                        "schema": REPORT_SCHEMA,
                        "image": image_path.stem,
                        "arm": label,
                        "series": f"{image_path.stem}__{label}",
                        **checkpoint.to_record(),
                    }
                )

        arm_values: dict[
            str,
            tuple[
                ObservationField2D,
                np.ndarray,
                ObservationField2D,
                str,
                CoefficientProjectionResult | None,
                float,
                list[dict[str, object]],
                list[dict[str, object]],
            ],
        ] = {
            "h005_control": (
                base_field,
                base_reconstruction,
                base_field,
                "none",
                None,
                float(base_row["pipeline_cumulative_seconds"]),
                json.loads((paths["base"] / "history.json").read_text(encoding="utf-8")),
                [],
            ),
            "touched_projection": (
                touched_field,
                touched_reconstruction,
                base_field,
                "touched",
                None,
                float(touched_row["pipeline_cumulative_seconds"]),
                json.loads((paths["base"] / "history.json").read_text(encoding="utf-8")),
                json.loads(
                    (paths["touched"] / "projection_history.json").read_text(encoding="utf-8")
                ),
            ),
            "guarded_exchange": (
                exchange_field,
                exchange_reconstruction,
                exchange_field,
                "none",
                None,
                float(exchange_row["pipeline_cumulative_seconds"]),
                json.loads((paths["exchange"] / "history.json").read_text(encoding="utf-8")),
                [],
            ),
            "exchange_global_projection": (
                exchange_global_projection.field,
                exchange_global_projection.reconstruction,
                exchange_field,
                "all",
                exchange_global_projection,
                float(exchange_row["pipeline_cumulative_seconds"])
                + exchange_global_projection.elapsed_seconds,
                json.loads((paths["exchange"] / "history.json").read_text(encoding="utf-8")),
                exchange_global_projection.checkpoint_records(),
            ),
            "global_projection": (
                global_projection.field,
                global_projection.reconstruction,
                base_field,
                "all",
                global_projection,
                float(base_row["pipeline_cumulative_seconds"])
                + global_projection.elapsed_seconds,
                json.loads((paths["base"] / "history.json").read_text(encoding="utf-8")),
                global_projection.checkpoint_records(),
            ),
        }
        initial_score = np.mean(
            (base_reconstruction.astype(np.float64) - target.astype(np.float64)) ** 2,
            axis=2,
        ).astype(np.float32)
        with np.load(paths["exchange"] / "analysis.npz", allow_pickle=False) as analysis:
            exchange_means = np.array(analysis["incoming_means"], copy=True)
        image_rows: list[dict[str, object]] = []
        for arm in ARMS:
            cell_started = time.perf_counter()
            (
                final_field,
                expected_reconstruction,
                geometry_source,
                scope,
                projection,
                cumulative_seconds,
                topology_history,
                projection_history,
            ) = arm_values[arm]
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
            save_image(str(artifact_dir / "initial_lattice.png"), base_reconstruction)
            save_error_heatmap(
                str(artifact_dir / "initial_error.png"),
                base_reconstruction - target,
                scale=args.error_scale,
            )
            save_image(str(artifact_dir / "reconstruction.png"), cold)
            save_error_heatmap(
                str(artifact_dir / "error.png"), cold - target, scale=args.error_scale
            )
            viz_utils._save_scalar(artifact_dir / "feature_priority.png", initial_score, mask)
            shown_exchange = (
                exchange_means
                if arm in ("guarded_exchange", "exchange_global_projection")
                else np.empty((0, 2), dtype=np.float32)
            )
            viz_utils._save_centers(
                artifact_dir / "protected.png",
                image,
                mask,
                shown_exchange,
                color=(1.0, 0.0, 0.2),
            )
            viz_utils._save_centers(
                artifact_dir / "centers.png",
                image,
                mask,
                cold_field.means_xy,
                color=(0.0, 1.0, 0.2),
            )
            crop_bounds = viz_utils._worst_crop_bounds(cold, target, mask)
            viz_utils._save_crop(artifact_dir / "source_crop.png", target, crop_bounds)
            viz_utils._save_crop(
                artifact_dir / "reconstruction_crop.png", cold, crop_bounds
            )
            error_visual = np.repeat(
                np.clip(np.mean(np.abs(cold - target), axis=2) * args.error_scale, 0.0, 1.0)[
                    :, :, None
                ],
                3,
                axis=2,
            )
            viz_utils._save_crop(
                artifact_dir / "error_crop.png", error_visual, crop_bounds
            )
            report_utils._write_json(artifact_dir / "history.json", topology_history)
            report_utils._write_json(
                artifact_dir / "recovery_history.json", projection_history
            )
            report_utils._write_json(
                artifact_dir / "projection_history.json", projection_history
            )
            non_rgb_equal = _non_rgb_equal(cold_field, geometry_source)
            np.savez_compressed(
                artifact_dir / "analysis.npz",
                initial_residual_score=initial_score,
                projection_trainable_row_mask=(
                    np.ones(7000, dtype=bool)
                    if scope == "all"
                    else (
                        inherited_touched
                        if scope == "touched"
                        else np.zeros(7000, dtype=bool)
                    )
                ),
                exchange_means=shown_exchange,
                crop_bounds=np.asarray(crop_bounds, dtype=np.int32),
            )
            report_utils._write_json(
                artifact_dir / "config.json",
                {
                    "schema": REPORT_SCHEMA,
                    "status": "diagnostic",
                    "image": image_path.stem,
                    "arm": arm,
                    "projection_scope": scope,
                    "projection": asdict(projection_config),
                    "pipeline_cumulative_seconds": cumulative_seconds,
                },
            )
            raw_sse = float(metrics["masked_mse"]) * active_pixels * 3
            selected_iteration = (
                int(touched_row["projection_selected_iteration"])
                if arm == "touched_projection"
                else (0 if projection is None else projection.selected_iteration)
            )
            if arm == "touched_projection":
                projection_trainable_rows = int(touched_row["projection_trainable_rows"])
                projection_initial_sse = float(touched_row["projection_initial_sse"])
                projection_final_sse = float(touched_row["projection_final_sse"])
                projection_forward_applications = int(
                    touched_row["projection_forward_applications"]
                )
                projection_transpose_applications = int(
                    touched_row["projection_transpose_applications"]
                )
                projection_adjoint_error = float(
                    touched_row["projection_adjoint_relative_error"]
                )
                projection_internal_parity = float(
                    touched_row["projection_internal_render_parity_max_abs"]
                )
                projection_relative_residual = float(
                    touched_row["projection_relative_normal_residual_max"]
                )
                projection_seconds = float(touched_row["projection_seconds"])
            elif projection is None:
                projection_trainable_rows = 0
                projection_initial_sse = raw_sse
                projection_final_sse = raw_sse
                projection_forward_applications = 0
                projection_transpose_applications = 0
                projection_adjoint_error = 0.0
                projection_internal_parity = 0.0
                projection_relative_residual = 0.0
                projection_seconds = 0.0
            else:
                projection_trainable_rows = projection.trainable_rows
                projection_initial_sse = projection.initial_sse
                projection_final_sse = projection.final_sse
                projection_forward_applications = projection.forward_applications
                projection_transpose_applications = projection.transpose_applications
                projection_adjoint_error = projection.adjoint_relative_error
                projection_internal_parity = projection.maintained_render_parity_max_abs
                projection_relative_residual = projection.relative_normal_residual_max
                projection_seconds = projection.elapsed_seconds
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
                "target_gaussians": 7000,
                "n_gaussians": cold_field.n,
                "reduction_factor": active_pixels / cold_field.n,
                "geometry_source_canonical_sha256": geometry_source.canonical_hash(),
                "non_rgb_arrays_bit_exact": non_rgb_equal,
                "projection_scope": scope,
                "projection_trainable_rows": projection_trainable_rows,
                "projection_selected_iteration": selected_iteration,
                "projection_checkpoint_count": len(projection_history),
                "projection_initial_sse": projection_initial_sse,
                "projection_final_sse": projection_final_sse,
                "projection_sse_gain": projection_initial_sse - projection_final_sse,
                "projection_forward_applications": projection_forward_applications,
                "projection_transpose_applications": projection_transpose_applications,
                "projection_adjoint_relative_error": projection_adjoint_error,
                "projection_internal_render_parity_max_abs": projection_internal_parity,
                "projection_relative_normal_residual_max": projection_relative_residual,
                "pipeline_cumulative_seconds": cumulative_seconds,
                "projection_seconds": projection_seconds,
                "cold_decode_seconds": cold_decode_seconds,
                "render_seconds": render_seconds,
                "metric_seconds": metric_seconds,
                "total_seconds": cumulative_seconds
                + cold_decode_seconds
                + 2.0 * render_seconds
                + metric_seconds,
                "peak_cuda_allocated_bytes": peak_cuda_bytes,
                "canonical_raw_bytes": int(
                    sum(array.nbytes for array in cold_field._array_items().values())
                ),
                "lossless_reference_bytes": field_path.stat().st_size,
                "maintained_render_parity_max_abs": maintained_parity,
                "repeated_render_parity_max_abs": repeated_parity,
                "field_canonical_sha256": cold_field.canonical_hash(),
                "field_file_sha256": report_utils._sha256(field_path),
                "cell_packaging_seconds": time.perf_counter() - cell_started,
                **metrics,
            }
            image_rows.append(row)

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
            row["selected_pipeline_cell_pass"] = bool(
                row["arm"] == "global_projection"
                and row["n_gaussians"] == 7000
                and row["non_rgb_arrays_bit_exact"]
                and float(row["psnr_delta_vs_control_db"]) >= 1.5
                and float(row["masked_mse"]) < float(control_row["masked_mse"])
                and bool(row["artifact_gate_pass"])
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
                f"{row['arm']}={float(row['psnr_db']):.4f}dB/"
                f"{float(row['artifact_pixel_rmse_max']):.5f}/"
                f"{float(row['artifact_patch_rmse_max_7']):.5f}"
                for row in image_rows
            ),
            flush=True,
        )

    selected_rows = [row for row in rows if row["arm"] == "global_projection"]
    exchange_global_rows = [
        row for row in rows if row["arm"] == "exchange_global_projection"
    ]
    selected_pass = len(selected_rows) == 2 and all(
        bool(row["selected_pipeline_cell_pass"]) for row in selected_rows
    )
    selected_geomean = float(
        np.exp(np.mean(np.log([float(row["masked_mse"]) for row in selected_rows])))
    )
    exchange_geomean = float(
        np.exp(np.mean(np.log([float(row["masked_mse"]) for row in exchange_global_rows])))
    )
    decision = {
        "schema": REPORT_SCHEMA,
        "status": "diagnostic",
        "selected_pipeline": "h005_control_then_global_projection",
        "selected_pipeline_pass": selected_pass,
        "minimum_psnr_gain_db": 1.5,
        "passing_images": [
            row["image"] for row in selected_rows if row["selected_pipeline_cell_pass"]
        ],
        "global_projection_geometric_mean_mse": selected_geomean,
        "exchange_global_projection_geometric_mean_mse": exchange_geomean,
        "global_projection_beats_exchange_global": selected_geomean < exchange_geomean,
        "summary": (
            "HIER-005 plus global safeguarded RGB projection clears the exact-count, +1.5 dB, "
            "local-artifact, non-RGB identity, renderer, and adjoint checks on both exposed "
            "views, and has lower geometric-mean MSE than exchange plus the same solve. This is "
            "the selected development pipeline, not a default or confirmation claim."
            if selected_pass and selected_geomean < exchange_geomean
            else "The selected global-projection pipeline or its simplicity attribution misses "
            "a diagnostic clause; no pipeline/default promotion follows."
        ),
    }
    report_utils._write_json(output_root / "decision.json", decision)
    report_utils._write_json(
        output_root / "projection_checkpoints.json",
        {"schema": REPORT_SCHEMA, "rows": projection_rows},
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
