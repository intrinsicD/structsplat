#!/usr/bin/env python3
"""Bounded local-residual repair ladder for a persisted HIER-005 field.

This task-local diagnostic starts every nonzero ladder row from the exact same cold-loaded base
field. It never optimizes base geometry or coefficients: fixed 0.75 px signed rescue Gaussians
are seeded by stable residual ranking plus Chebyshev-radius NMS, and only their RGB coefficients
are optimized. Displayed 8-bit PNG artifact metrics are the final gate authority.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
from html import escape
from html.parser import HTMLParser
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time
from urllib.parse import unquote, urlsplit

import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import hier005_pixel_contraction as base_report  # noqa: E402


REPORT_SCHEMA = "structsplat.hier005_artifact_repair.diagnostic.v1"
CURVE_SPECS = (
    ("psnr_db", "Foreground-mask PSNR (dB)", False),
    ("ssim", "Full-matted-raster SSIM", False),
    ("ms_ssim", "Full-matted-raster MS-SSIM", False),
    ("lpips", "Full-matted-raster LPIPS (lower is better)", True),
    ("masked_mse", "Foreground-mask MSE", True),
    ("artifact_pixel_rmse_q99", "Display PNG foreground pixel RMSE q99", True),
    ("artifact_pixel_rmse_q999", "Display PNG foreground pixel RMSE q99.9", True),
    ("artifact_pixel_rmse_max", "Display PNG foreground pixel RMSE maximum", True),
    (
        "artifact_pixel_rmse_fraction_gt_005",
        "Display PNG foreground fraction with pixel RMSE > 0.05",
        True,
    ),
    (
        "artifact_pixel_rmse_fraction_gt_010",
        "Display PNG foreground fraction with pixel RMSE > 0.10",
        True,
    ),
    ("artifact_patch_rmse_max_3", "Display PNG maximum 3x3 patch RMSE", True),
    ("artifact_patch_rmse_max_7", "Display PNG maximum 7x7 patch RMSE", True),
    ("artifact_patch_rmse_max_15", "Display PNG maximum 15x15 patch RMSE", True),
    ("artifact_patch_rmse_max_31", "Display PNG maximum 31x31 patch RMSE", True),
    ("raw_artifact_pixel_rmse_max", "Raw foreground pixel RMSE maximum", True),
    ("raw_artifact_patch7_rmse_max", "Raw black-matted 7x7 patch RMSE maximum", True),
    ("raw_artifact_normalized_violation", "Raw normalized local-artifact violation", True),
    ("raw_sse", "Raw foreground SSE", True),
    ("estimated_field_bytes", "Estimated uncoded field bytes", True),
    ("canonical_raw_bytes", "Canonical raw field bytes", True),
    ("lossless_reference_bytes", "Lossless reference NPZ bytes", True),
    ("estimated_bits_per_pixel", "Estimated uncoded field bits/pixel", True),
    ("canonical_raw_bits_per_pixel", "Canonical raw field bits/pixel", True),
    ("lossless_reference_bits_per_pixel", "Lossless reference NPZ bits/pixel", True),
    ("source_over_estimated_ratio", "Original source bytes / estimated field bytes", True),
    (
        "evaluation_png_over_estimated_ratio",
        "Evaluation PNG bytes / estimated field bytes",
        True,
    ),
    ("rescue_rows_added", "Accepted rescue rows", False),
    ("repair_selected_step", "Selected rescue optimizer step", False),
    ("repair_seconds", "Local repair wall time (seconds)", True),
    ("cold_decode_seconds", "Cold field decode time (seconds)", True),
    ("first_render_seconds", "First maintained render time (seconds)", True),
    ("render_seconds", "Immediate-repeat maintained render time (seconds)", True),
    ("metric_seconds", "Metric evaluation time (seconds)", True),
    ("total_seconds", "Artifact-to-metrics wall time (seconds)", True),
    (
        "maintained_render_parity_max_abs",
        "Maintained/in-memory reconstruction max-absolute difference",
        True,
    ),
    (
        "repeated_render_parity_max_abs",
        "First/repeated maintained render max-absolute difference",
        True,
    ),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonable(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(_jsonable(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _raw_artifact_values(
    reconstruction: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    *,
    pixel_threshold: float,
    patch_threshold: float,
) -> dict[str, float]:
    import torch
    import torch.nn.functional as torch_functional

    residual = reconstruction.astype(np.float64) - target.astype(np.float64)
    pixel_mse = np.mean(residual * residual, axis=2)
    pixel_max = float(np.sqrt(np.max(pixel_mse[mask])))
    black_matted = np.where(mask, pixel_mse, 0.0).astype(np.float32)
    side = min(7, *mask.shape)
    if side % 2 == 0:
        side -= 1
    side = max(side, 1)
    pooled = torch_functional.avg_pool2d(
        torch.from_numpy(black_matted)[None, None],
        kernel_size=side,
        stride=1,
        padding=0,
    )
    patch_max = float(torch.sqrt(torch.max(pooled)).item())
    return {
        "raw_sse": float(np.sum((residual[mask]) ** 2)),
        "raw_artifact_pixel_rmse_max": pixel_max,
        "raw_artifact_patch7_rmse_max": patch_max,
        "raw_artifact_normalized_violation": max(
            pixel_max / pixel_threshold,
            patch_max / patch_threshold,
        ),
    }


def _save_crops(
    artifact_dir: Path,
    target: np.ndarray,
    reconstruction: np.ndarray,
    mask: np.ndarray,
    selected_xy: np.ndarray,
    *,
    error_scale: float,
) -> tuple[int, int, list[int]]:
    from structsplat.cli import save_error_heatmap, save_image

    displayed_target = np.rint(np.clip(target, 0.0, 1.0) * 255.0) / 255.0
    displayed_reconstruction = np.rint(np.clip(reconstruction, 0.0, 1.0) * 255.0) / 255.0
    pixel_rmse = np.sqrt(
        np.mean((displayed_reconstruction - displayed_target) ** 2, axis=2)
    )
    pixel_rmse[~mask] = -1.0
    worst_y, worst_x = np.unravel_index(int(np.argmax(pixel_rmse)), pixel_rmse.shape)
    crop_side = min(96, target.shape[0], target.shape[1])
    x0 = min(max(0, worst_x - crop_side // 2), target.shape[1] - crop_side)
    y0 = min(max(0, worst_y - crop_side // 2), target.shape[0] - crop_side)
    x1, y1 = x0 + crop_side, y0 + crop_side
    crop_box = [x0, y0, x1, y1]
    save_image(str(artifact_dir / "source_crop.png"), target[y0:y1, x0:x1])
    save_image(
        str(artifact_dir / "reconstruction_crop.png"),
        reconstruction[y0:y1, x0:x1],
    )
    save_error_heatmap(
        str(artifact_dir / "error_crop.png"),
        reconstruction[y0:y1, x0:x1] - target[y0:y1, x0:x1],
        scale=error_scale,
    )
    overlay = Image.fromarray(
        np.rint(np.clip(target, 0.0, 1.0) * 255.0).astype(np.uint8), mode="RGB"
    )
    draw = ImageDraw.Draw(overlay)
    for x, y in selected_xy.tolist():
        draw.ellipse((x - 1, y - 1, x + 1, y + 1), fill=(255, 196, 0))
    overlay.save(artifact_dir / "rescue_centers.png")
    return int(worst_x), int(worst_y), crop_box


def _write_tables(output_root: Path, rows: list[dict[str, object]]) -> None:
    payload = {
        "schema": REPORT_SCHEMA,
        "status": "diagnostic",
        "claim_ready": False,
        "row_count": len(rows),
        "selection_rule": (
            "independent rows start from the exact same base field; selected checkpoint "
            "lexicographically minimizes raw normalized local-artifact violation, then raw SSE"
        ),
        "metric_domains": {
            "psnr_db": "thresholded foreground mask only",
            "ssim_ms_ssim_lpips": "full evaluation raster after black matting",
            "artifact_pixel_rmse_*": "exact displayed 8-bit PNG foreground values",
            "artifact_patch_rmse_max_*": (
                "maximum complete in-canvas black-matted patch RMSE on displayed 8-bit PNGs"
            ),
            "raw_artifact_*": "unclipped float reconstruction used only for checkpoint audit",
            "rate": "uncoded/reference-container diagnostics, not a compressed codec rate",
        },
        "rows": rows,
    }
    _write_json(output_root / "metrics.json", payload)
    with (output_root / "metrics.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(_jsonable(row), sort_keys=True) + "\n")
    columns = sorted({key for row in rows for key in row})
    with (output_root / "metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: (
                        json.dumps(_jsonable(value), sort_keys=True)
                        if isinstance(value, (dict, list, tuple))
                        else "" if value is None else value
                    )
                    for key, value in row.items()
                }
            )


def _write_curves(output_root: Path, rows: list[dict[str, object]]) -> list[dict[str, object]]:
    curve_root = output_root / "curves"
    # ``main`` refuses an existing output directory. ``exist_ok=True`` also permits a
    # packaging-only regeneration from the persisted result rows.
    curve_root.mkdir(parents=True, exist_ok=True)
    catalog: list[dict[str, object]] = []
    for metric, label, prefer_log_y in CURVE_SPECS:
        svg = base_report._metric_curve_svg(rows, metric, label, prefer_log_y)
        if svg is None:
            continue
        path = curve_root / f"{metric}.svg"
        path.write_text(svg + "\n", encoding="utf-8")
        catalog.append(
            {
                "metric": metric,
                "label": label,
                "path": str(path.relative_to(output_root)),
                "preferred_y_scale": "log10" if prefer_log_y else "linear",
            }
        )
    _write_json(
        curve_root / "catalog.json",
        {"schema": REPORT_SCHEMA, "x": "n_gaussians", "curves": catalog},
    )
    return catalog


def _format(value: object, digits: int = 4) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        if value != 0.0 and (abs(value) < 0.001 or abs(value) >= 10000):
            return f"{value:.3e}"
        return f"{value:.{digits}f}"
    return str(value)


def _write_report(
    output_root: Path,
    rows: list[dict[str, object]],
    curves: list[dict[str, object]],
    *,
    command: str,
) -> None:
    table_rows: list[str] = []
    cards: list[str] = []
    for row in rows:
        artifact = escape(str(row["artifact_dir"]))
        table_rows.append(
            "<tr>"
            f"<td>{row['rescue_limit']}</td><td>{row['rescue_rows_added']}</td>"
            f"<td>{row['n_gaussians']}</td><td>{row['repair_selected_step']}</td>"
            f"<td>{_format(row['psnr_db'], 3)}</td><td>{_format(row['ms_ssim'], 6)}</td>"
            f"<td>{_format(row['lpips'], 6)}</td>"
            f"<td>{_format(row['artifact_pixel_rmse_max'], 5)}</td>"
            f"<td>{_format(row['artifact_patch_rmse_max_7'], 5)}</td>"
            f"<td><strong>{'pass' if row['artifact_gate_pass'] else 'FAIL'}</strong></td>"
            f"<td>{_format(row['raw_artifact_normalized_violation'], 3)}</td>"
            f"<td>{_format(row['estimated_bits_per_pixel'], 4)}</td>"
            f"<td>{_format(row['evaluation_png_over_estimated_ratio'], 3)}</td>"
            f"<td>{_format(row['repair_seconds'], 2)}</td></tr>"
        )
        cards.append(
            "<article class='card'>"
            f"<h3>limit {row['rescue_limit']} · accepted {row['rescue_rows_added']} · "
            f"N={row['n_gaussians']}</h3>"
            "<div class='images'>"
            f"<figure><img src='{artifact}/source.png'><figcaption>source</figcaption></figure>"
            f"<figure><img src='{artifact}/reconstruction.png'><figcaption>cold reconstruction</figcaption></figure>"
            f"<figure><img src='{artifact}/error.png'><figcaption>4× fixed-scale error</figcaption></figure>"
            "</div><h4>Worst displayed-error neighborhood</h4><div class='images crops'>"
            f"<figure><img src='{artifact}/source_crop.png'><figcaption>source crop</figcaption></figure>"
            f"<figure><img src='{artifact}/reconstruction_crop.png'><figcaption>reconstruction crop</figcaption></figure>"
            f"<figure><img src='{artifact}/error_crop.png'><figcaption>4× error crop</figcaption></figure>"
            "</div><div class='centers'>"
            f"<figure><img src='{artifact}/rescue_centers.png'><figcaption>rescue centers (yellow)</figcaption></figure>"
            "</div>"
            f"<p>PSNR {_format(row['psnr_db'], 3)} dB · MS-SSIM "
            f"{_format(row['ms_ssim'], 6)} · LPIPS {_format(row['lpips'], 6)} · "
            f"display gate <strong>{'pass' if row['artifact_gate_pass'] else 'FAIL'}</strong> "
            f"(pixel {_format(row['artifact_pixel_rmse_max'], 5)} / 0.02000; "
            f"7×7 {_format(row['artifact_patch_rmse_max_7'], 5)} / 0.01000).</p>"
            f"<p>Base prefix bit-exact: {row['base_prefix_bit_exact']} · repeated-render "
            f"parity {_format(row['repeated_render_parity_max_abs'])} · worst pixel "
            f"({row['worst_display_x']}, {row['worst_display_y']}).</p>"
            f"<p><a href='{artifact}/field.observation.npz'>field</a> · "
            f"<a href='{artifact}/repair_history.json'>repair history</a> · "
            f"<a href='{artifact}/row.json'>row ledger</a></p></article>"
        )
    curve_cards = "".join(
        "<figure class='curve'><img src='{}'><figcaption><code>{}</code></figcaption></figure>".format(
            escape(str(curve["path"])), escape(str(curve["metric"]))
        )
        for curve in curves
    )
    document = f"""<!doctype html><html lang='en'><head><meta charset='utf-8'>
<meta name='viewport' content='width=device-width'><title>HIER-005 local artifact repair</title>
<style>:root{{--ink:#17212b;--muted:#5c6b78;--line:#d7dfe5;--paper:#f7f9fb}}
body{{font-family:system-ui,sans-serif;color:var(--ink);margin:0;background:var(--paper)}}
header,main{{max-width:1220px;margin:auto;padding:24px}}header{{padding-bottom:8px}}
.warning{{border-left:5px solid #b45309;background:#fff7ed;padding:12px 16px}}
table{{border-collapse:collapse;width:100%;background:white}}th,td{{padding:8px;border-bottom:1px solid var(--line);text-align:right}}
.card,.curve{{background:white;border:1px solid var(--line);border-radius:8px;padding:12px;margin:16px 0}}
.images{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}}figure{{margin:0}}
img{{width:100%;height:auto}}.crops img{{image-rendering:pixelated}}.centers{{max-width:390px}}
.curves{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}}.curve{{margin:0}}
figcaption{{color:var(--muted)}}code{{overflow-wrap:anywhere}}
@media(max-width:800px){{.images,.curves{{grid-template-columns:1fr}}table{{font-size:12px}}}}</style></head>
<body><header><h1>HIER-005 bounded local artifact repair</h1>
<p class='warning'><strong>Diagnostic only.</strong> Every nonzero row forks the exact same
cold 4,096-row hard-3σ terminal field. Base parameters remain bit-exact; fixed rescue geometry
is added and only rescue RGB is optimized. The displayed 8-bit PNG pixel/patch gate is final.
Payload values are uncoded diagnostics, not codec rates.</p>
<p><code>{escape(command)}</code></p><p><a href='metrics.json'>metrics.json</a> ·
<a href='metrics.jsonl'>metrics.jsonl</a> · <a href='metrics.csv'>metrics.csv</a> ·
<a href='config.json'>config.json</a> · <a href='curves/catalog.json'>curve catalog</a> ·
<a href='manifest.json'>manifest</a> · <a href='verification.json'>verification</a></p></header><main>
<h2>Outcomes</h2><div style='overflow:auto'><table><thead><tr><th>row limit</th>
<th>added</th><th>N</th><th>step</th><th>PSNR dB</th><th>MS-SSIM</th><th>LPIPS</th>
<th>pixel max</th><th>7×7 max</th><th>gate</th><th>raw violation</th><th>est. bpp</th>
<th>eval PNG / est.</th><th>repair s</th></tr></thead><tbody>{''.join(table_rows)}</tbody></table></div>
<h2>Visuals</h2>{''.join(cards)}<h2>All outcome curves</h2>
<p>{len(curves)} standalone SVG curves versus achieved Gaussian count.</p>
<div class='curves'>{curve_cards}</div></main></body></html>"""
    (output_root / "index.html").write_text(document, encoding="utf-8")


def _snapshot_sources(output_root: Path) -> list[dict[str, object]]:
    paths = (
        Path(__file__).resolve(),
        ROOT / "scripts" / "experiments" / "hier005_pixel_contraction.py",
        ROOT / "src" / "structsplat" / "pixel_contraction.py",
    )
    records: list[dict[str, object]] = []
    for source in paths:
        relative = source.relative_to(ROOT)
        destination = output_root / "source_snapshot" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        records.append(
            {
                "path": str(relative),
                "sha256": _sha256(destination),
                "bytes": destination.stat().st_size,
            }
        )
    return records


def _write_manifest(output_root: Path) -> None:
    files = []
    for path in sorted(output_root.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            files.append(
                {
                    "path": str(path.relative_to(output_root)),
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
            )
    _write_json(
        output_root / "manifest.json",
        {"schema": REPORT_SCHEMA, "status": "diagnostic", "files": files},
    )


class _LinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        del tag
        for name, value in attrs:
            if name in {"href", "src"} and value:
                self.links.append(value)


def _verify_bundle(output_root: Path) -> dict[str, object]:
    manifest = json.loads((output_root / "manifest.json").read_text(encoding="utf-8"))
    for record in manifest["files"]:
        path = output_root / record["path"]
        if not path.is_file():
            raise RuntimeError(f"manifest path is missing: {record['path']}")
        if path.stat().st_size != record["bytes"] or _sha256(path) != record["sha256"]:
            raise RuntimeError(f"manifest identity differs: {record['path']}")
    parser = _LinkParser()
    parser.feed((output_root / "index.html").read_text(encoding="utf-8"))
    resolved_links: set[str] = set()
    for raw in parser.links:
        split = urlsplit(raw)
        if split.scheme or split.netloc or raw.startswith("/"):
            raise RuntimeError(f"report contains a nonportable link: {raw}")
        if not split.path:
            continue
        target = (output_root / unquote(split.path)).resolve()
        try:
            target.relative_to(output_root.resolve())
        except ValueError as exc:
            raise RuntimeError(f"report link escapes the bundle: {raw}") from exc
        if not target.is_file():
            raise RuntimeError(f"report link is missing: {raw}")
        resolved_links.add(str(target.relative_to(output_root.resolve())))
    for required in ("metrics.json", "metrics.jsonl", "metrics.csv", "manifest.json"):
        if required not in resolved_links:
            raise RuntimeError(f"report does not expose required ledger: {required}")
    metrics = json.loads((output_root / "metrics.json").read_text(encoding="utf-8"))
    jsonl_rows = [
        json.loads(line)
        for line in (output_root / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if jsonl_rows != metrics["rows"]:
        raise RuntimeError("metrics JSON and JSONL rows differ")
    with (output_root / "metrics.csv").open(newline="", encoding="utf-8") as handle:
        csv_rows = list(csv.DictReader(handle))
    if len(csv_rows) != len(metrics["rows"]):
        raise RuntimeError("metrics CSV row count differs")
    if not all(row["base_prefix_bit_exact"] for row in metrics["rows"]):
        raise RuntimeError("a repair row changed the base-field prefix")
    if not all(
        row["maintained_render_parity_max_abs"] < 2e-6 for row in metrics["rows"]
    ):
        raise RuntimeError("a repair row failed maintained-render parity")
    return {
        "schema": REPORT_SCHEMA,
        "status": "verified_diagnostic",
        "manifest_entries_checked": len(manifest["files"]),
        "html_links_checked": len(parser.links),
        "metric_rows_checked": len(metrics["rows"]),
        "base_prefix_bit_exact": True,
        "maintained_render_parity_threshold": 2e-6,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-artifact", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rescue-rows", type=int, nargs="+", default=(102, 205, 410))
    parser.add_argument("--scale-px", type=float, default=0.75)
    parser.add_argument("--nms-radius", type=int, default=1)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--tail-fraction", type=float, default=0.01)
    parser.add_argument("--tail-weight", type=float, default=4.0)
    parser.add_argument("--pixel-threshold", type=float, default=0.02)
    parser.add_argument("--patch7-threshold", type=float, default=0.01)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--renderer",
        choices=("additive", "cuda_additive", "cuda_tiled_additive"),
        default="additive",
    )
    parser.add_argument("--render-chunk", type=int, default=256)
    parser.add_argument("--error-scale", type=float, default=4.0)
    parser.add_argument("--lpips", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite existing output directory: {args.out}")
    if any(rows <= 0 for rows in args.rescue_rows):
        raise SystemExit("all --rescue-rows values must be positive")
    if len(set(args.rescue_rows)) != len(args.rescue_rows):
        raise SystemExit("--rescue-rows values must be unique")
    base_field_path = args.base_artifact / "field.observation.npz"
    target_path = args.base_artifact / "source.png"
    base_row_path = args.base_artifact / "row.json"
    for required in (base_field_path, target_path, base_row_path):
        if not required.is_file():
            raise SystemExit(f"missing base artifact input: {required}")

    from structsplat.cli import save_error_heatmap, save_image
    from structsplat.observation_field import ObservationField2D
    from structsplat.pixel_contraction import (
        LocalRescueConfig,
        render_observation_field,
        rescue_observation_field,
    )

    args.out.mkdir(parents=True)
    (args.out / "artifacts").mkdir()
    base_field = ObservationField2D.load_lossless(base_field_path)
    base_row = json.loads(base_row_path.read_text(encoding="utf-8"))
    target = np.asarray(Image.open(target_path).convert("RGB"), dtype=np.float32) / 255.0
    mask = (
        np.ones(base_field.crop_shape, dtype=bool)
        if base_field.packed_alpha is None
        else base_field.alpha_mask()
    )
    if target.shape[:2] != base_field.crop_shape:
        raise SystemExit("base source PNG and field crop dimensions disagree")

    command = " ".join(shlex.quote(token) for token in ([sys.executable, __file__] + sys.argv[1:]))
    rows: list[dict[str, object]] = []
    ladder = [0, *sorted(args.rescue_rows)]
    for rescue_limit in ladder:
        row_started = time.perf_counter()
        artifact_dir = args.out / "artifacts" / f"rescue_{rescue_limit:04d}"
        artifact_dir.mkdir()
        if rescue_limit == 0:
            repair = None
            candidate_field = base_field
            in_memory_reconstruction = render_observation_field(
                base_field,
                device=args.device,
                renderer=args.renderer,
                render_chunk=args.render_chunk,
            )
            selected_xy = np.empty((0, 2), dtype=np.int64)
            repair_record: dict[str, object] = {
                "rows_added": 0,
                "selected_step": -1,
                "elapsed_seconds": 0.0,
            }
        else:
            repair = rescue_observation_field(
                base_field,
                target,
                LocalRescueConfig(
                    max_rows=rescue_limit,
                    scale_px=args.scale_px,
                    nms_radius_px=args.nms_radius,
                    steps=args.steps,
                    learning_rate=args.learning_rate,
                    tail_fraction=args.tail_fraction,
                    tail_weight=args.tail_weight,
                    pixel_rmse_threshold=args.pixel_threshold,
                    patch7_rmse_threshold=args.patch7_threshold,
                    device=args.device,
                    renderer=args.renderer,
                    render_chunk=args.render_chunk,
                ),
                mask=mask,
            )
            candidate_field = repair.field
            in_memory_reconstruction = repair.reconstruction
            selected_xy = repair.selected_xy
            repair_record = repair.to_record()
        _write_json(artifact_dir / "repair_history.json", repair_record)
        field_path = artifact_dir / "field.observation.npz"
        candidate_field.save_lossless(field_path)
        lossless_bytes = field_path.stat().st_size
        decode_started = time.perf_counter()
        cold_field = ObservationField2D.load_lossless(field_path)
        cold_decode_seconds = time.perf_counter() - decode_started
        first_render_started = time.perf_counter()
        first_reconstruction = render_observation_field(
            cold_field,
            device=args.device,
            renderer=args.renderer,
            render_chunk=args.render_chunk,
        )
        first_render_seconds = time.perf_counter() - first_render_started
        render_started = time.perf_counter()
        reconstruction = render_observation_field(
            cold_field,
            device=args.device,
            renderer=args.renderer,
            render_chunk=args.render_chunk,
        )
        render_seconds = time.perf_counter() - render_started
        metric_started = time.perf_counter()
        metric_values = base_report._metric_values(
            reconstruction,
            target,
            mask,
            device=args.device,
            compute_lpips=args.lpips,
        )
        raw_values = _raw_artifact_values(
            reconstruction,
            target,
            mask,
            pixel_threshold=args.pixel_threshold,
            patch_threshold=args.patch7_threshold,
        )
        metric_seconds = time.perf_counter() - metric_started
        save_image(str(artifact_dir / "source.png"), target)
        save_image(str(artifact_dir / "reconstruction.png"), reconstruction)
        save_error_heatmap(
            str(artifact_dir / "error.png"), reconstruction - target, scale=args.error_scale
        )
        worst_x, worst_y, crop_box = _save_crops(
            artifact_dir,
            target,
            reconstruction,
            mask,
            selected_xy,
            error_scale=args.error_scale,
        )
        arrays = cold_field._array_items()
        canonical_raw_bytes = int(sum(array.nbytes for array in arrays.values()))
        alpha_bytes = 0 if cold_field.packed_alpha is None else cold_field.packed_alpha.nbytes
        estimated_field_bytes = int(cold_field.n * 32 + alpha_bytes)
        pixel_count = target.shape[0] * target.shape[1]
        active_pixels = int(mask.sum())
        prefix_exact = all(
            np.array_equal(getattr(cold_field, name)[: base_field.n], getattr(base_field, name))
            for name in ("means_xy", "log_scales_xy", "rotations_rad", "rgb_coeff")
        )
        row: dict[str, object] = {
            "schema": REPORT_SCHEMA,
            "status": "diagnostic",
            "method": "hard3_terminal_frozen_base_signed_local_rescue",
            "image": str(base_row["image"]),
            "source_path": str(base_row["source_path"]),
            "source_sha256": str(base_row["source_sha256"]),
            "source_file_bytes": int(base_row["source_file_bytes"]),
            "evaluation_source_png_bytes": target_path.stat().st_size,
            "original_width": int(base_row["original_width"]),
            "original_height": int(base_row["original_height"]),
            "width": target.shape[1],
            "height": target.shape[0],
            "pixels": pixel_count,
            "active_pixels": active_pixels,
            "base_gaussians": base_field.n,
            "rescue_limit": rescue_limit,
            "rescue_rows_added": int(repair_record["rows_added"]),
            "n_gaussians": cold_field.n,
            "repair_selected_step": int(repair_record["selected_step"]),
            "repair_seconds": float(repair_record["elapsed_seconds"]),
            "base_prefix_bit_exact": prefix_exact,
            "base_field_canonical_sha256": base_field.canonical_hash(),
            "field_canonical_sha256": cold_field.canonical_hash(),
            "field_file_sha256": _sha256(field_path),
            "estimated_field_bytes": estimated_field_bytes,
            "canonical_raw_bytes": canonical_raw_bytes,
            "lossless_reference_bytes": lossless_bytes,
            "estimated_bits_per_pixel": 8.0 * estimated_field_bytes / pixel_count,
            "canonical_raw_bits_per_pixel": 8.0 * canonical_raw_bytes / pixel_count,
            "lossless_reference_bits_per_pixel": 8.0 * lossless_bytes / pixel_count,
            "estimated_bits_per_active_pixel": 8.0 * estimated_field_bytes / active_pixels,
            "source_over_estimated_ratio": int(base_row["source_file_bytes"])
            / estimated_field_bytes,
            "source_over_canonical_raw_ratio": int(base_row["source_file_bytes"])
            / canonical_raw_bytes,
            "source_over_lossless_reference_ratio": int(base_row["source_file_bytes"])
            / lossless_bytes,
            "evaluation_png_over_estimated_ratio": target_path.stat().st_size
            / estimated_field_bytes,
            "cold_decode_seconds": cold_decode_seconds,
            "first_render_seconds": first_render_seconds,
            "render_seconds": render_seconds,
            "metric_seconds": metric_seconds,
            "maintained_render_parity_max_abs": float(
                np.max(np.abs(reconstruction - in_memory_reconstruction))
            ),
            "repeated_render_parity_max_abs": float(
                np.max(np.abs(reconstruction - first_reconstruction))
            ),
            "worst_display_x": worst_x,
            "worst_display_y": worst_y,
            "worst_crop_xyxy": crop_box,
            "artifact_dir": str(artifact_dir.relative_to(args.out)),
            **raw_values,
            **metric_values,
        }
        row["total_seconds"] = time.perf_counter() - row_started
        _write_json(artifact_dir / "row.json", row)
        rows.append(row)

    _write_tables(args.out, rows)
    curves = _write_curves(args.out, rows)
    source_snapshots = _snapshot_sources(args.out)
    base_report_path = args.base_artifact.parents[1]
    git_head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=False, capture_output=True, text=True
    ).stdout.strip()
    config = {
        "schema": REPORT_SCHEMA,
        "status": "diagnostic",
        "claim_ready": False,
        "task": "HIER-005",
        "command": command,
        "args": vars(args) | {"base_artifact": str(args.base_artifact), "out": str(args.out)},
        "base_report": str(base_report_path),
        "base_artifact": str(args.base_artifact),
        "base_field_sha256": _sha256(base_field_path),
        "base_field_canonical_sha256": base_field.canonical_hash(),
        "target_png_sha256": _sha256(target_path),
        "mask_sha256": base_row.get("mask_source_sha256"),
        "git_head": git_head,
        "git_dirty": bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=ROOT,
                check=False,
                capture_output=True,
                text=True,
            ).stdout
        ),
        "method_determinism": (
            "cuda_atomic_numerically_nondeterministic"
            if args.device.startswith("cuda")
            else "cpu_bit_deterministic"
        ),
        "source_snapshot": source_snapshots,
    }
    _write_json(args.out / "config.json", config)
    _write_report(
        args.out,
        rows,
        curves,
        command=command,
    )
    _write_json(
        args.out / "verification.json",
        {"schema": REPORT_SCHEMA, "status": "pending"},
    )
    _write_manifest(args.out)
    verification = _verify_bundle(args.out)
    _write_json(args.out / "verification.json", verification)
    _write_manifest(args.out)
    _verify_bundle(args.out)
    print(args.out / "index.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
