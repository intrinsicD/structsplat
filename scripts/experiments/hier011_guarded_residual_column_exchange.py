#!/usr/bin/env python3
"""Run the frozen HIER-011 exact-count residual column-exchange diagnostic."""
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
from structsplat.cli import save_error_heatmap, save_image  # noqa: E402
from structsplat.contraction_refinement import (  # noqa: E402
    CoefficientProjectionConfig,
    project_contracted_coefficients,
)
from structsplat.observation_field import ObservationField2D  # noqa: E402
from structsplat.pixel_contraction import render_observation_field  # noqa: E402
from structsplat.residual_exchange import (  # noqa: E402
    ResidualExchangeConfig,
    exchange_residual_columns,
)


REPORT_SCHEMA = "structsplat.hier011_guarded_residual_column_exchange.diagnostic.v1"
HIER010_MANIFEST_SHA256 = "80b84bce9b5ec72e9369fd61474d761c8ecd3f2a9f6ed9495f7cb67f14dd81ba"
ARMS = (
    "h005_control",
    "control_projection",
    "guarded_exchange",
    "exchange_projection",
)
CANDIDATE_SHAPES = (
    (0.30, 0.30, 0.0),
    (0.45, 0.45, 0.0),
    (0.60, 0.60, 0.0),
    (0.75, 0.75, 0.0),
    (0.75, 0.30, 0.0),
    (0.75, 0.30, math.pi / 4.0),
    (0.75, 0.30, math.pi / 2.0),
    (0.75, 0.30, 3.0 * math.pi / 4.0),
)
EXPECTED = {
    "C0001": {
        "rgb": "ae24fe99d3f8edbd04cd2c85ebc4fe9bfd95abe878c22abb7691cadcfc5c411b",
        "mask": "94dcbf7005dbeb1d183e259a569d783aa5df900255e763385bed91f02d3b80c3",
        "h005_file": "cfa05c3cc5bfe5f747e14bae2cfe254283593123a4eb92177f73b95a071d1dac",
        "h005_canonical": "9bfbe941b90bac66a7c6ce3166fffcd76224520f40edc0800a9de4a1ea9cfb5b",
        "projection_file": "b15cadb6b5211dcd0cb70cf074534a8f3e8d7650b47cb3ac5f98d656c07a7595",
        "projection_canonical": "21a93aef6d249eb788a29a54d246d118031929d41bd4557e7059e78aa5c9a59c",
        "analysis_file": "bb03c256a99959689f15b11b1122ccf671b91653cebcd26f584537f1bc0b48a5",
    },
    "C0004": {
        "rgb": "26eb4cf24a034eb830198df6e7a6ac409ccb7cf4814ff645c71d0b6966b7070e",
        "mask": "4702bfa9df354f38e35a63207a37d4ec1b753afc4d0668bd905f3cdab320f35d",
        "h005_file": "6743d9141791c8932532708085815157e7c918ee398c1ab5fbcbc0342b111f5b",
        "h005_canonical": "8c4406059c5bc68254fd2b16740019ae550230aa6778369f99ac9d606e9635e8",
        "projection_file": "7404c470ba90b806ebff5e9cd20eb623ab62f30dcff12e5baca3c3ca72f9da3c",
        "projection_canonical": "177987a89c891d278ed84d5458fe350abbd1e6343a0c91fcf695842b0c952ac6",
        "analysis_file": "b6dda6626c0bdd1a0a52a7465a7bbfdf4e10da4debc7c0d96a4bde1864791a90",
    },
}


def _snapshot_sources(output_root: Path) -> list[dict[str, object]]:
    sources = (
        Path(__file__).resolve(),
        ROOT / "src" / "structsplat" / "residual_exchange.py",
        ROOT / "src" / "structsplat" / "contraction_refinement.py",
        ROOT / "src" / "structsplat" / "pixel_contraction.py",
        ROOT / "scripts" / "experiments" / "hier005_pixel_contraction.py",
        ROOT / "scripts" / "experiments" / "hier010_residual_anchor_projection.py",
        ROOT / "scripts" / "check_report_bundle.py",
        ROOT / "tasks" / "HIER-011-guarded-residual-column-exchange.md",
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
    columns = sorted({name for row in rows for name in row})
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


def _plot_curves(
    output_root: Path,
    rows: list[dict[str, object]],
    exchange_rows: list[dict[str, object]],
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
        for image_name in sorted({str(row["image"]) for row in rows}):
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

    for metric in ("raw_sse", "psnr_db", "display_pixel_rmse_max", "display_patch7_rmse_max"):
        figure, axis = plt.subplots(figsize=(7.2, 3.8), constrained_layout=True)
        for image_name in sorted({str(row["image"]) for row in exchange_rows}):
            points = sorted(
                (row for row in exchange_rows if row["image"] == image_name),
                key=lambda row: int(row["accepted_count"]),
            )
            axis.plot(
                [int(row["accepted_count"]) for row in points],
                [float(row[metric]) for row in points],
                linewidth=1.2,
                label=image_name,
            )
        axis.set_xlabel("accepted one-for-one exchanges")
        axis.set_ylabel(metric.replace("_", " "))
        axis.grid(True, alpha=0.28)
        axis.legend()
        path = curve_root / f"exchange__{metric}.svg"
        figure.savefig(path, format="svg")
        plt.close(figure)
        records.append({"kind": "exchange", "metric": metric, "path": str(path.relative_to(output_root))})

    figure, axis = plt.subplots(figsize=(7.2, 3.8), constrained_layout=True)
    for image_name in sorted({str(row["image"]) for row in projection_rows}):
        points = sorted(
            (row for row in projection_rows if row["image"] == image_name),
            key=lambda row: int(row["iteration"]),
        )
        axis.plot(
            [int(row["iteration"]) for row in points],
            [float(row["raw_sse"]) for row in points],
            linewidth=1.2,
            label=image_name,
        )
    axis.set_xlabel("PCG iteration")
    axis.set_ylabel("raw SSE")
    axis.grid(True, alpha=0.28)
    axis.legend()
    path = curve_root / "projection__raw_sse.svg"
    figure.savefig(path, format="svg")
    plt.close(figure)
    records.append({"kind": "projection", "metric": "raw_sse", "path": str(path.relative_to(output_root))})

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
        lpips_text = "n/a" if row.get("lpips") is None else f"{float(row['lpips']):.7f}"
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
            f"<td>{int(row['exchange_accepted'])}</td>"
            f"<td>{int(row['projection_selected_iteration'])}</td>"
            f"<td>{float(row['pipeline_cumulative_seconds']):.2f}</td>"
            f"<td>{mechanism}</td></tr>"
        )
        artifact = str(row["artifact_dir"])
        title = f"{row['image']} / {row['arm']}"
        cards.append(
            f"<section class='card'><h3>{escape(title)}</h3>"
            f"<p>{float(row['psnr_db']):.4f} dB ({float(row['psnr_delta_vs_control_db']):+.4f}); "
            f"pixel/7×7 {float(row['artifact_pixel_rmse_max']):.5f}/"
            f"{float(row['artifact_patch_rmse_max_7']):.5f}; {int(row['exchange_accepted'])} "
            f"exchanges; PCG step {int(row['projection_selected_iteration'])}; gate {gate}</p>"
            "<div class='images'>"
            f"<figure><img src='{artifact}/source.png'><figcaption>source</figcaption></figure>"
            f"<figure><img src='{artifact}/reconstruction.png'><figcaption>result</figcaption></figure>"
            f"<figure><img src='{artifact}/error.png'><figcaption>error ×4</figcaption></figure>"
            f"<figure><img src='{artifact}/feature_priority.png'><figcaption>initial residual energy</figcaption></figure>"
            f"<figure><img src='{artifact}/protected.png'><figcaption>entering atom centers</figcaption></figure>"
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
    color = "#e4f6ec" if decision["advance_mechanism"] else "#fee9e7"
    border = "#2e8b57" if decision["advance_mechanism"] else "#b64335"
    document = f"""<!doctype html><html><head><meta charset='utf-8'>
<meta name='viewport' content='width=device-width,initial-scale=1'>
<title>HIER-011 guarded residual column exchange</title><style>
body{{font-family:system-ui,sans-serif;margin:0;background:#f3f6f8;color:#18242d}}
main{{max-width:1500px;margin:auto;padding:24px}}.warning{{background:#fff3cd;border:1px solid #e0ba54;padding:12px;border-radius:8px}}
.verdict{{background:{color};border:1px solid {border};padding:16px;border-radius:8px}}
table{{border-collapse:collapse;width:100%;background:white;font-size:12px}}th,td{{border:1px solid #dce3e8;padding:7px;text-align:right}}th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){{text-align:left}}
.pass{{color:#087a48;font-weight:700}}.fail{{color:#b62929;font-weight:700}}.card{{background:white;border:1px solid #dce3e8;border-radius:9px;padding:14px;margin:18px 0}}
.images{{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:10px}}figure{{margin:0}}img{{width:100%;height:auto;background:#101417}}figcaption{{font-size:12px;color:#52616c}}.links{{columns:2;font-size:12px}}
</style></head><body><main><p>STRUCTSPLAT / HIER-011 / EXPOSED-VIEW DIAGNOSTIC</p>
<h1>Pay for every residual atom</h1>
<p class='warning'><strong>Diagnostic only.</strong> C0001 selected the atom bank and C0004 is a correlated, previously exposed transfer view. The repository is dirty and there was no distinct prospective reviewer. Exact row count is matched; work is not. Bytes are reference storage, not codec rate.</p>
<section class='verdict'><h2>Frozen full-pipeline gate: {verdict}</h2><p>{escape(str(decision['summary']))}</p></section>
<p><a href='manifest.json'>manifest</a> · <a href='metrics.json'>metrics JSON</a> · <a href='metrics.jsonl'>JSONL</a> · <a href='metrics.csv'>CSV</a> · <a href='config.json'>config</a> · <a href='decision.json'>decision</a> · <a href='exchange_checkpoints.json'>exchange checkpoints</a> · <a href='projection_checkpoints.json'>projection checkpoints</a> · <a href='curves/catalog.json'>curves</a></p>
<h2>Exact-7k outcomes</h2><table><thead><tr><th>image</th><th>arm</th><th>N</th><th>PSNR</th><th>Δ control</th><th>MSE</th><th>MS-SSIM</th><th>LPIPS</th><th>pixel max</th><th>7×7 max</th><th>artifact gate</th><th>swaps</th><th>PCG</th><th>pipeline s</th><th>full rule</th></tr></thead><tbody>{''.join(table_rows)}</tbody></table>
<h2>Visual audit</h2>{''.join(cards)}<h2>Metric and solver curves</h2><ul class='links'>{curve_links}</ul>
<h2>Raw cell artifacts</h2><ul class='links'>{''.join(artifact_links)}</ul><h2>Executed source snapshot</h2><ul class='links'>{snapshots}</ul>
</main></body></html>"""
    (output_root / "index.html").write_text(document, encoding="utf-8")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", nargs="+", type=Path, required=True)
    parser.add_argument("--masks", nargs="+", type=Path, required=True)
    parser.add_argument("--hier010-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-exchanges", type=int, default=128)
    parser.add_argument("--site-count", type=int, default=96)
    parser.add_argument("--site-nms-radius", type=int, default=1)
    parser.add_argument("--donor-count", type=int, default=64)
    parser.add_argument("--proposal-frontier", type=int, default=24)
    parser.add_argument("--coefficient-limit", type=float, default=16.0)
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


def _validate_protocol(
    images: list[Path], masks: list[Path], hier010_root: Path, args: argparse.Namespace
) -> None:
    if len(images) != 2 or len(masks) != 2 or len(images) != len(masks):
        raise SystemExit("the frozen HIER-011 command requires two image/mask pairs")
    if [path.stem for path in images] != ["C0001", "C0004"]:
        raise SystemExit("the frozen HIER-011 order is C0001 then C0004")
    if report_utils._sha256(hier010_root / "manifest.json") != HIER010_MANIFEST_SHA256:
        raise SystemExit("HIER-010 manifest hash mismatch")
    for image, mask in zip(images, masks, strict=True):
        expected = EXPECTED[image.stem]
        if report_utils._sha256(image) != expected["rgb"]:
            raise SystemExit(f"source hash mismatch for {image}")
        if report_utils._sha256(mask) != expected["mask"]:
            raise SystemExit(f"mask hash mismatch for {mask}")
        artifact_root = hier010_root / "artifacts"
        h005_root = artifact_root / f"{image.stem}__h005_control__n7000"
        projection_root = artifact_root / f"{image.stem}__control_projection__n7000"
        if report_utils._sha256(h005_root / "field.observation.npz") != expected["h005_file"]:
            raise SystemExit(f"HIER-005 field hash mismatch for {image.stem}")
        if report_utils._sha256(projection_root / "field.observation.npz") != expected["projection_file"]:
            raise SystemExit(f"projection field hash mismatch for {image.stem}")
        if report_utils._sha256(h005_root / "analysis.npz") != expected["analysis_file"]:
            raise SystemExit(f"analysis hash mismatch for {image.stem}")
        h005_field = ObservationField2D.load_lossless(h005_root / "field.observation.npz")
        projection_field = ObservationField2D.load_lossless(
            projection_root / "field.observation.npz"
        )
        if h005_field.canonical_hash() != expected["h005_canonical"]:
            raise SystemExit(f"HIER-005 canonical hash mismatch for {image.stem}")
        if projection_field.canonical_hash() != expected["projection_canonical"]:
            raise SystemExit(f"projection canonical hash mismatch for {image.stem}")
    expected_ints = {
        "max_exchanges": 128,
        "site_count": 96,
        "site_nms_radius": 1,
        "donor_count": 64,
        "proposal_frontier": 24,
        "projection_max_iterations": 48,
        "max_side": 512,
        "render_chunk": 256,
    }
    for name, expected in expected_ints.items():
        if getattr(args, name) != expected:
            raise SystemExit(f"frozen protocol requires --{name.replace('_', '-')} {expected}")
    expected_floats = {
        "coefficient_limit": 16.0,
        "projection_ridge": 1e-8,
        "projection_tolerance": 1e-6,
        "projection_coefficient_limit": 16.0,
        "mask_threshold": 0.5,
        "error_scale": 4.0,
    }
    for name, expected in expected_floats.items():
        if float(getattr(args, name)) != expected:
            raise SystemExit(f"frozen protocol requires --{name.replace('_', '-')} {expected}")
    if args.device != "cuda" or args.renderer != "cuda_additive" or not args.lpips:
        raise SystemExit(
            "frozen protocol requires --device cuda --renderer cuda_additive --lpips"
        )


def _input_paths(hier010_root: Path, image_name: str) -> tuple[Path, Path, Path, Path]:
    artifact_root = hier010_root / "artifacts"
    base = artifact_root / f"{image_name}__h005_control__n7000"
    projected = artifact_root / f"{image_name}__control_projection__n7000"
    return (
        base / "field.observation.npz",
        projected / "field.observation.npz",
        base / "analysis.npz",
        base / "row.json",
    )


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    images = [path.resolve() for path in args.images]
    masks = [path.resolve() for path in args.masks]
    hier010_root = args.hier010_root.resolve()
    _validate_protocol(images, masks, hier010_root, args)
    output_root = args.out.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise SystemExit(f"output directory must be absent or empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    command = shlex.join(
        [sys.executable, str(Path(__file__).relative_to(ROOT)), *(argv or sys.argv[1:])]
    )
    source_snapshot = _snapshot_sources(output_root)
    exchange_config = ResidualExchangeConfig(
        candidate_shapes=CANDIDATE_SHAPES,
        max_exchanges=args.max_exchanges,
        site_count=args.site_count,
        site_nms_radius_px=args.site_nms_radius,
        donor_count=args.donor_count,
        proposal_frontier=args.proposal_frontier,
        coefficient_abs_limit=args.coefficient_limit,
    )
    projection_config = CoefficientProjectionConfig(
        ridge=args.projection_ridge,
        tolerance=args.projection_tolerance,
        max_iterations=args.projection_max_iterations,
        coefficient_abs_limit=args.projection_coefficient_limit,
    )

    import torch

    config: dict[str, object] = {
        "schema": REPORT_SCHEMA,
        "task": "HIER-011",
        "status": "diagnostic",
        "claim_ready": False,
        "command": command,
        "args": vars(args),
        "git": report_utils._git_record(),
        "executed_source_snapshot": source_snapshot,
        "hier010_manifest_sha256": HIER010_MANIFEST_SHA256,
        "input_bindings": EXPECTED,
        "development_bank_selection": {
            "view": "C0001",
            "accepted_pivots": 32,
            "winner": "oriented",
            "compact_final_sse": 0.44133125930395634,
            "multiscale_final_sse": 0.43752487644524096,
            "oriented_final_sse": 0.4316094323398037,
        },
        "exchange": asdict(exchange_config),
        "projection": asdict(projection_config),
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "cuda_device": torch.cuda.get_device_name(0),
        "evidence_limits": [
            "C0001 selected the atom bank; C0004 is correlated and previously exposed.",
            "No distinct prospective reviewer; dirty-source diagnostic only.",
            "CUDA accumulation is numerically, not bit, reproducible.",
            "Arms spend unequal work; no speed conclusion is permitted.",
            "Canonical/raw/NPZ bytes are reference storage, not complete codec rate.",
        ],
    }
    report_utils._write_json(output_root / "config.json", config)

    rows: list[dict[str, object]] = []
    exchange_rows: list[dict[str, object]] = []
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
            raise RuntimeError("HIER-011 requires a mask")
        mask = loaded_mask
        target = image * mask[:, :, None]
        active_pixels = int(mask.sum())
        base_path, projected_path, analysis_path, original_row_path = _input_paths(
            hier010_root, image_path.stem
        )
        base_field = ObservationField2D.load_lossless(base_path)
        control_projection_field = ObservationField2D.load_lossless(projected_path)
        with np.load(analysis_path, allow_pickle=False) as analysis:
            inherited_touched = np.array(analysis["touched_row_mask"], dtype=bool, copy=True)
        original_row = json.loads(original_row_path.read_text(encoding="utf-8"))
        projected_original_row = json.loads(
            (projected_path.parent / "row.json").read_text(encoding="utf-8")
        )
        if base_field.n != 7000 or control_projection_field.n != 7000:
            raise RuntimeError("persisted inputs must contain exactly 7,000 rows")
        if inherited_touched.shape != (7000,):
            raise RuntimeError("persisted touched-row provenance is misaligned")

        torch.cuda.reset_peak_memory_stats()
        base_reconstruction = render_observation_field(
            base_field,
            device=args.device,
            renderer=args.renderer,
            render_chunk=args.render_chunk,
        )
        persisted_projection_reconstruction = render_observation_field(
            control_projection_field,
            device=args.device,
            renderer=args.renderer,
            render_chunk=args.render_chunk,
        )
        exchange = exchange_residual_columns(
            base_field,
            target,
            mask,
            config=exchange_config,
            device=args.device,
            renderer=args.renderer,
            render_chunk=args.render_chunk,
        )
        projection_trainable = inherited_touched | exchange.replaced_row_mask
        projection = project_contracted_coefficients(
            exchange.field,
            target,
            mask,
            projection_trainable,
            np.zeros(exchange.field.n, dtype=bool),
            config=projection_config,
            device=args.device,
            renderer=args.renderer,
            render_chunk=args.render_chunk,
        )
        peak_cuda_bytes = int(torch.cuda.max_memory_allocated())
        base_seconds = float(original_row["pipeline_cumulative_seconds"])
        control_projection_seconds = float(
            projected_original_row["pipeline_cumulative_seconds"]
        )
        arm_values = {
            "h005_control": (base_field, base_reconstruction, 0, None, base_seconds),
            "control_projection": (
                control_projection_field,
                persisted_projection_reconstruction,
                0,
                None,
                control_projection_seconds,
            ),
            "guarded_exchange": (
                exchange.field,
                exchange.reconstruction,
                exchange.accepted_exchanges,
                None,
                base_seconds + exchange.elapsed_seconds,
            ),
            "exchange_projection": (
                projection.field,
                projection.reconstruction,
                exchange.accepted_exchanges,
                projection,
                base_seconds + exchange.elapsed_seconds + projection.elapsed_seconds,
            ),
        }
        initial_score = np.mean(
            (base_reconstruction.astype(np.float64) - target.astype(np.float64)) ** 2,
            axis=2,
        ).astype(np.float32)
        incoming_means = exchange.field.means_xy[exchange.replaced_row_mask]
        image_rows: list[dict[str, object]] = []
        for arm in ARMS:
            cell_started = time.perf_counter()
            final_field, expected_reconstruction, accepted_count, arm_projection, cumulative = (
                arm_values[arm]
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
            shown_incoming = (
                incoming_means
                if arm in ("guarded_exchange", "exchange_projection")
                else np.empty((0, 2), dtype=np.float32)
            )
            viz_utils._save_centers(
                artifact_dir / "protected.png",
                image,
                mask,
                shown_incoming,
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
            exchange_history = (
                exchange.checkpoint_records()
                if arm in ("guarded_exchange", "exchange_projection")
                else []
            )
            projection_history = (
                [] if arm_projection is None else arm_projection.checkpoint_records()
            )
            report_utils._write_json(artifact_dir / "history.json", exchange_history)
            report_utils._write_json(
                artifact_dir / "recovery_history.json", projection_history
            )
            report_utils._write_json(
                artifact_dir / "projection_history.json", projection_history
            )
            np.savez_compressed(
                artifact_dir / "analysis.npz",
                initial_residual_score=initial_score,
                inherited_touched_row_mask=inherited_touched,
                replaced_row_mask=(
                    exchange.replaced_row_mask
                    if arm in ("guarded_exchange", "exchange_projection")
                    else np.zeros(7000, dtype=bool)
                ),
                incoming_means=shown_incoming,
                crop_bounds=np.asarray(crop_bounds, dtype=np.int32),
            )
            report_utils._write_json(
                artifact_dir / "config.json",
                {
                    "schema": REPORT_SCHEMA,
                    "status": "diagnostic",
                    "image": image_path.stem,
                    "arm": arm,
                    "exchange": asdict(exchange_config),
                    "projection": asdict(projection_config),
                    "exchange_enabled": arm in ("guarded_exchange", "exchange_projection"),
                    "projection_enabled": arm in ("control_projection", "exchange_projection"),
                    "pipeline_cumulative_seconds": cumulative,
                },
            )

            canonical_bytes = int(
                sum(array.nbytes for array in cold_field._array_items().values())
            )
            projection_selected = (
                int(projected_original_row["projection_selected_iteration"])
                if arm == "control_projection"
                else (0 if arm_projection is None else arm_projection.selected_iteration)
            )
            raw_sse = float(metrics["masked_mse"]) * active_pixels * 3
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
                "base_field_source_path": str(base_path),
                "base_field_source_sha256": report_utils._sha256(base_path),
                "original_width": raster_record["original_width"],
                "original_height": raster_record["original_height"],
                "width": image.shape[1],
                "height": image.shape[0],
                "pixels": int(mask.size),
                "active_pixels": active_pixels,
                "target_gaussians": 7000,
                "n_gaussians": cold_field.n,
                "reduction_factor": active_pixels / cold_field.n,
                "exchange_enabled": arm in ("guarded_exchange", "exchange_projection"),
                "exchange_accepted": accepted_count,
                "exchange_stop_reason": exchange.stop_reason if accepted_count else "not_run",
                "exchange_proposed_pairs": exchange.proposed_pairs if accepted_count else 0,
                "exchange_cold_rendered_pairs": exchange.cold_rendered_pairs if accepted_count else 0,
                "exchange_initial_sse": exchange.initial_sse,
                "exchange_final_sse": exchange.final_sse if accepted_count else exchange.initial_sse,
                "exchange_sse_gain": (
                    exchange.initial_sse - exchange.final_sse if accepted_count else 0.0
                ),
                "exchange_maximum_pricing_error_abs": (
                    exchange.maximum_pricing_error_abs if accepted_count else 0.0
                ),
                "exchange_internal_render_parity_max_abs": (
                    exchange.maintained_render_parity_max_abs if accepted_count else 0.0
                ),
                "exchange_repeated_render_parity_max_abs": (
                    exchange.repeated_render_parity_max_abs if accepted_count else 0.0
                ),
                "projection_enabled": arm in ("control_projection", "exchange_projection"),
                "projection_selected_iteration": projection_selected,
                "projection_checkpoint_count": len(projection_history),
                "projection_trainable_rows": (
                    0 if arm_projection is None else arm_projection.trainable_rows
                ),
                "projection_frozen_rows": (
                    cold_field.n if arm_projection is None else arm_projection.frozen_rows
                ),
                "projection_initial_sse": (
                    raw_sse
                    if arm_projection is None
                    else arm_projection.initial_sse
                ),
                "projection_final_sse": (
                    raw_sse
                    if arm_projection is None
                    else arm_projection.final_sse
                ),
                "projection_sse_gain": (
                    0.0
                    if arm_projection is None
                    else arm_projection.initial_sse - arm_projection.final_sse
                ),
                "projection_forward_applications": (
                    0 if arm_projection is None else arm_projection.forward_applications
                ),
                "projection_transpose_applications": (
                    0 if arm_projection is None else arm_projection.transpose_applications
                ),
                "projection_adjoint_relative_error": (
                    0.0 if arm_projection is None else arm_projection.adjoint_relative_error
                ),
                "projection_internal_render_parity_max_abs": (
                    0.0
                    if arm_projection is None
                    else arm_projection.maintained_render_parity_max_abs
                ),
                "inherited_touched_rows": int(inherited_touched.sum()),
                "pipeline_cumulative_seconds": cumulative,
                "exchange_seconds": exchange.elapsed_seconds if accepted_count else 0.0,
                "projection_seconds": (
                    0.0 if arm_projection is None else arm_projection.elapsed_seconds
                ),
                "cold_decode_seconds": cold_decode_seconds,
                "render_seconds": render_seconds,
                "metric_seconds": metric_seconds,
                "total_seconds": cumulative + cold_decode_seconds + 2.0 * render_seconds + metric_seconds,
                "peak_cuda_allocated_bytes": peak_cuda_bytes,
                "canonical_raw_bytes": canonical_bytes,
                "lossless_reference_bytes": field_path.stat().st_size,
                "maintained_render_parity_max_abs": maintained_parity,
                "repeated_render_parity_max_abs": repeated_parity,
                "field_canonical_sha256": cold_field.canonical_hash(),
                "field_file_sha256": report_utils._sha256(field_path),
                "cell_packaging_seconds": time.perf_counter() - cell_started,
                **metrics,
            }
            image_rows.append(row)

        for checkpoint in exchange.checkpoints:
            exchange_rows.append(
                {
                    "schema": REPORT_SCHEMA,
                    "image": image_path.stem,
                    **checkpoint.to_record(),
                }
            )
        for checkpoint in projection.checkpoints:
            projection_rows.append(
                {
                    "schema": REPORT_SCHEMA,
                    "image": image_path.stem,
                    "arm": "exchange_projection",
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
                row["arm"] == "exchange_projection"
                and row["n_gaussians"] == 7000
                and float(row["psnr_delta_vs_control_db"]) >= 0.10
                and float(row["masked_mse"]) < float(control_row["masked_mse"])
                and bool(row["artifact_gate_pass"])
                and float(row["artifact_pixel_rmse_max"])
                <= float(control_row["artifact_pixel_rmse_max"])
                and float(row["artifact_patch_rmse_max_7"])
                <= float(control_row["artifact_patch_rmse_max_7"])
                and float(row["maintained_render_parity_max_abs"]) <= 2e-6
                and float(row["repeated_render_parity_max_abs"]) <= 2e-6
                and float(row["exchange_internal_render_parity_max_abs"]) <= 2e-6
                and float(row["exchange_maximum_pricing_error_abs"]) <= 2e-6
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

    full_rows = [row for row in rows if row["arm"] == "exchange_projection"]
    advance = len(full_rows) == 2 and all(
        bool(row["full_mechanism_cell_pass"]) for row in full_rows
    )
    decision = {
        "schema": REPORT_SCHEMA,
        "status": "diagnostic",
        "advance_mechanism": advance,
        "required_images": ["C0001", "C0004"],
        "passing_images": [row["image"] for row in full_rows if row["full_mechanism_cell_pass"]],
        "minimum_psnr_gain_db": 0.10,
        "summary": (
            "The exact-count exchange-plus-projection pipeline clears the frozen material-gain, "
            "artifact, local-nonregression, count, pricing, and renderer-integrity clauses on "
            "both exposed views. This motivates a fresh reviewed/independent study only; it does "
            "not change a default."
            if advance
            else "The full pipeline misses at least one frozen per-image clause. HIER-005 remains "
            "unchanged and these consumed views cannot be retuned."
        ),
    }
    report_utils._write_json(output_root / "decision.json", decision)
    report_utils._write_json(
        output_root / "exchange_checkpoints.json",
        {"schema": REPORT_SCHEMA, "rows": exchange_rows},
    )
    report_utils._write_json(
        output_root / "projection_checkpoints.json",
        {"schema": REPORT_SCHEMA, "rows": projection_rows},
    )
    _write_tables(output_root, rows)
    curves = _plot_curves(output_root, rows, exchange_rows, projection_rows)
    _write_report(output_root, rows, curves, decision)
    config["elapsed_seconds"] = time.perf_counter() - run_started
    config["decision"] = decision
    report_utils._write_json(output_root / "config.json", config)
    _write_manifest(output_root)
    print(f"wrote diagnostic report: {output_root / 'index.html'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
