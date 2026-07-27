#!/usr/bin/env python3
"""BENCH-017 exploratory pass: run the full-frame arm of ADR-0025 over an image set.

This is **not** the preregistered BENCH-017 screen. It runs one arm
(``run_pipeline(image, mask=None)`` at the shipped recipe) over a directory of unmasked images
and emits per-image reconstructions, error maps, a metric table, and a browsable ``index.html``.
It has no control arm, no frozen gate, and no multi-seed replication, so nothing it produces may
promote a claim on its own. Its job is visual inspection and a first look at where the arm lands.

Invocation (the one this run used)::

    PYTHONPATH=src /home/alex/miniconda3/bin/python \
        scripts/experiments/bench017_fullframe_screen.py \
        --images results/datasets/abl004/kodak24 \
        --out results/bench017_fullframe_kodak24 \
        --capacity 5000 --seed 0 --device cuda --renderer cuda

Re-render the page from an interrupted or finished run without refitting::

    ... bench017_fullframe_screen.py --out <same outdir> --report-only

Per BENCH-002 the outdir carries ``config.json`` (resolved args, device, versions). Records are
appended to ``records.jsonl`` as each image completes, so the run is resumable and a partial run
is still viewable. GPU renders are not bit-reproducible (atomic accumulation); the recorded
device and versions bound the variation.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from html import escape
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

# Sequential blue ramp (dataviz reference palette), light -> dark, for the error maps.
_ERROR_RAMP = [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
    "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
]
# Absolute mean-channel error mapped to the top of the ramp. Fixed so maps are comparable
# across images; errors above it saturate.
ERROR_SCALE = 0.15


def _ramp_lut() -> np.ndarray:
    """Return a (256, 3) uint8 LUT interpolating the sequential ramp from white."""
    stops = np.array(
        [[int(h[i:i + 2], 16) for i in (1, 3, 5)] for h in _ERROR_RAMP], dtype=np.float64
    )
    stops = np.vstack([np.array([255.0, 255.0, 255.0]), stops])
    xs = np.linspace(0.0, 1.0, len(stops))
    grid = np.linspace(0.0, 1.0, 256)
    return np.stack(
        [np.interp(grid, xs, stops[:, c]) for c in range(3)], axis=1
    ).round().astype(np.uint8)


def discover_images(paths: list[str]) -> list[Path]:
    out: list[Path] = []
    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            out.extend(sorted(q for q in p.iterdir() if q.suffix.lower() in IMAGE_SUFFIXES))
        elif p.is_file():
            out.append(p)
        else:
            raise SystemExit(f"no such image or directory: {p}")
    if not out:
        raise SystemExit("no images found")
    return out


def write_config(outdir: Path, args: argparse.Namespace, images: list[Path], device: str) -> None:
    import torch

    import structsplat

    cfg = {
        "script": "scripts/experiments/bench017_fullframe_screen.py",
        "task": "BENCH-017 (exploratory single-arm pass, not the preregistered screen)",
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "images": [str(p) for p in images],
        "resolved_device": device,
        "versions": {
            "structsplat": getattr(structsplat, "__version__", "unknown"),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "python": platform.python_version(),
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "error_map_scale": ERROR_SCALE,
        "notes": [
            "Single arm: run_pipeline(image, mask=None) at the shipped recipe.",
            "No control arm, no frozen gate, single seed -- cannot promote a claim.",
            "CUDA renders are not bit-reproducible (atomic accumulation).",
        ],
    }
    (outdir / "config.json").write_text(json.dumps(cfg, indent=2))


def run_one(image_path: Path, args: argparse.Namespace, outdir: Path, lut: np.ndarray) -> dict:
    """Fit one image, write its artifacts, and return the tidy record."""
    import torch

    from structsplat import metrics as M
    from structsplat.cli import load_image, save_image
    from structsplat.config import FitConfig
    from structsplat.fit import _render
    from structsplat.pipeline import PipelineConfig, run_pipeline

    img = load_image(str(image_path))
    H, W = img.shape[:2]
    stem = image_path.stem

    cfg = PipelineConfig(
        capacity=args.capacity,
        seed=args.seed,
        device=args.device,
        renderer=args.renderer,
        step_scale=args.step_scale,
    )
    t0 = time.perf_counter()
    result = run_pipeline(img, None, cfg, verbose=args.verbose)
    wall = time.perf_counter() - t0

    field = result["field"]
    fit_cfg = FitConfig(**{
        k: v for k, v in result["fit_config"].items() if k in FitConfig.__dataclass_fields__
    })
    with torch.no_grad():
        rendered = _render(field, fit_cfg, H, W, support_fade_alpha=1.0)
    recon = rendered.detach().cpu().numpy()

    # Metrics on the raw render, one convention per row (BENCH-002).
    dev = torch.device("cuda" if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    pred = torch.from_numpy(np.ascontiguousarray(recon)).to(dev)
    targ = torch.from_numpy(np.ascontiguousarray(img)).to(dev)
    row_metrics = {
        "psnr_db": float(M.psnr(pred, targ)),
        "ssim": float(M.ssim(pred, targ)),
        "ms_ssim": float(M.ms_ssim(pred, targ)),
    }
    try:
        lp = M.LPIPS.distance(pred, targ)  # None when the optional `lpips` dep is absent
        row_metrics["lpips"] = float(lp) if lp is not None else None
    except Exception as exc:  # weight download or backend failure
        row_metrics["lpips"] = None
        row_metrics["lpips_error"] = str(exc)[:200]

    assets = outdir / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    save_image(str(assets / f"{stem}_recon.png"), recon)
    save_image(str(assets / f"{stem}_source.png"), img)
    err = np.abs(recon - img).mean(axis=2)
    idx = np.clip(err / ERROR_SCALE, 0.0, 1.0)
    err_rgb = lut[(idx * 255.0).round().astype(np.uint8)]
    save_image(str(assets / f"{stem}_error.png"), err_rgb.astype(np.float32) / 255.0)
    field.save(str(assets / f"{stem}.npz"))
    (assets / f"{stem}_pipeline.json").write_text(
        json.dumps({k: v for k, v in result.items() if k not in ("field", "optimizer_state")},
                   indent=2, default=str)
    )

    pipeline_metrics = result["metrics"]
    return {
        "image": image_path.name,
        "source_path": str(image_path),
        "height": H,
        "width": W,
        "arm": result["arm"],
        "recipe": f"{result['recipe']['name']}@{result['recipe']['version']}",
        "capacity": args.capacity,
        "n_gaussians": int(field.n),
        "seed": args.seed,
        "step_scale": args.step_scale,
        "device": result["device"],
        "renderer": args.renderer or "auto",
        "seconds": float(result["seconds"]),
        "wall_seconds": wall,
        "attempted_steps": int(result.get("attempted_steps", 0)),
        "accepted_steps": int(result.get("accepted_steps", 0)),
        "converged": bool(result.get("converged", False)),
        "pipeline_foreground_psnr_db": pipeline_metrics.get("foreground_psnr_db"),
        "interior_hole_fraction": pipeline_metrics.get("interior_hole_fraction"),
        **row_metrics,
        "status": "ok",
    }


# ---------------------------------------------------------------------------- report


def _fmt(value, digits: int = 3, dash: str = "--") -> str:
    if value is None:
        return dash
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _summary(rows: list[dict]) -> dict:
    ok = [r for r in rows if r.get("status") == "ok"]

    def agg(key: str) -> dict:
        vals = [r[key] for r in ok if isinstance(r.get(key), (int, float))]
        if not vals:
            return {"mean": None, "min": None, "max": None, "std": None, "n": 0}
        arr = np.asarray(vals, dtype=float)
        return {
            "mean": float(arr.mean()), "min": float(arr.min()), "max": float(arr.max()),
            "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0, "n": int(arr.size),
        }

    return {k: agg(k) for k in (
        "psnr_db", "ms_ssim", "ssim", "lpips", "n_gaussians", "seconds",
        "attempted_steps", "accepted_steps",
    )}


def _psnr_chart(rows: list[dict]) -> str:
    """Horizontal bars, one series, sorted by PSNR. Single series -> no legend box."""
    ok = [r for r in rows if r.get("status") == "ok" and isinstance(r.get("psnr_db"), float)]
    if not ok:
        return "<p class='muted'>No completed rows yet.</p>"
    ok = sorted(ok, key=lambda r: r["psnr_db"])
    row_h, gap, left, right, top = 22, 6, 88, 56, 8
    width, height = 720, top + len(ok) * (row_h + gap)
    lo = 0.0
    hi = max(r["psnr_db"] for r in ok) * 1.08
    plot_w = width - left - right

    parts = [
        f"<svg viewBox='0 0 {width} {height}' class='chart' role='img' "
        f"aria-label='PSNR by image, decibels'>"
    ]
    for i, r in enumerate(ok):
        y = top + i * (row_h + gap)
        w = max(2.0, (r["psnr_db"] - lo) / (hi - lo) * plot_w)
        label = escape(Path(r["image"]).stem)
        parts.append(
            f"<g class='bar-g'><title>{label}: {r['psnr_db']:.2f} dB PSNR, "
            f"{_fmt(r.get('ms_ssim'), 4)} MS-SSIM</title>"
            f"<text x='{left - 10}' y='{y + row_h * 0.72}' class='cat' "
            f"text-anchor='end'>{label}</text>"
            f"<rect x='{left}' y='{y}' width='{w:.2f}' height='{row_h}' rx='4' "
            f"class='bar'/>"
            f"<text x='{left + w + 8:.2f}' y='{y + row_h * 0.72}' class='val'>"
            f"{r['psnr_db']:.2f}</text></g>"
        )
    parts.append("</svg>")
    return "".join(parts)


def _cards(rows: list[dict]) -> str:
    out = []
    for r in rows:
        stem = Path(r["image"]).stem
        if r.get("status") != "ok":
            out.append(
                f"<section class='card'><h3>{escape(r['image'])}</h3>"
                f"<p class='fail'>failed: {escape(str(r.get('error', ''))[:400])}</p></section>"
            )
            continue
        lpips = _fmt(r.get("lpips"), 4)
        out.append(f"""
<section class="card" id="{escape(stem)}">
  <div class="card-head">
    <h3>{escape(r['image'])}</h3>
    <div class="chips">
      <span class="chip"><b>{r['psnr_db']:.2f}</b> dB</span>
      <span class="chip">MS-SSIM <b>{_fmt(r.get('ms_ssim'), 4)}</b></span>
      <span class="chip">LPIPS <b>{lpips}</b></span>
      <span class="chip">{r['n_gaussians']:,} rows</span>
      <span class="chip">{r['seconds']:.0f}s</span>
    </div>
  </div>
  <div class="triptych">
    <figure><a href="assets/{escape(stem)}_source.png" target="_blank" rel="noopener">
      <img loading="lazy" src="assets/{escape(stem)}_source.png" alt="source {escape(stem)}"></a>
      <figcaption>source &middot; {r['width']}&times;{r['height']}</figcaption></figure>
    <figure><a href="assets/{escape(stem)}_recon.png" target="_blank" rel="noopener">
      <img loading="lazy" src="assets/{escape(stem)}_recon.png"
           alt="reconstruction {escape(stem)}"></a>
      <figcaption>reconstruction</figcaption></figure>
    <figure><a href="assets/{escape(stem)}_error.png" target="_blank" rel="noopener">
      <img loading="lazy" src="assets/{escape(stem)}_error.png"
           alt="error map {escape(stem)}"></a>
      <figcaption>|error| &middot; 0 to {ERROR_SCALE}</figcaption></figure>
  </div>
  <p class="meta">accepted {r['accepted_steps']:,} of {r['attempted_steps']:,} attempted steps
     &middot; interior holes {_fmt(r.get('interior_hole_fraction'), 5)}
     &middot; <a href="assets/{escape(stem)}_pipeline.json">pipeline.json</a></p>
</section>""")
    return "".join(out)


def _table(rows: list[dict]) -> str:
    head = ("image", "PSNR (dB)", "MS-SSIM", "SSIM", "LPIPS", "rows", "attempted",
            "accepted", "seconds")
    body = []
    for r in rows:
        if r.get("status") != "ok":
            body.append(
                f"<tr><td>{escape(r['image'])}</td>"
                f"<td colspan='8' class='fail'>failed</td></tr>"
            )
            continue
        body.append(
            "<tr>"
            f"<td>{escape(r['image'])}</td>"
            f"<td class='num'>{r['psnr_db']:.2f}</td>"
            f"<td class='num'>{_fmt(r.get('ms_ssim'), 4)}</td>"
            f"<td class='num'>{_fmt(r.get('ssim'), 4)}</td>"
            f"<td class='num'>{_fmt(r.get('lpips'), 4)}</td>"
            f"<td class='num'>{r['n_gaussians']:,}</td>"
            f"<td class='num'>{r['attempted_steps']:,}</td>"
            f"<td class='num'>{r['accepted_steps']:,}</td>"
            f"<td class='num'>{r['seconds']:.0f}</td>"
            "</tr>"
        )
    return (
        "<table><thead><tr>"
        + "".join(f"<th>{escape(h)}</th>" for h in head)
        + "</tr></thead><tbody>" + "".join(body) + "</tbody></table>"
    )


def build_report(outdir: Path) -> Path:
    records_path = outdir / "records.jsonl"
    rows = [json.loads(line) for line in records_path.read_text().splitlines() if line.strip()]
    config = json.loads((outdir / "config.json").read_text())
    summary = _summary(rows)
    (outdir / "summary.json").write_text(
        json.dumps({"summary": summary, "n_rows": len(rows)}, indent=2)
    )

    ok = [r for r in rows if r.get("status") == "ok"]
    recipe = ok[0]["recipe"] if ok else "n/a"
    arm = ok[0]["arm"] if ok else "n/a"
    versions = config.get("versions", {})
    gpu = versions.get("gpu") or config.get("resolved_device", "?")
    mean_psnr = summary["psnr_db"]["mean"]
    hero = f"{mean_psnr:.2f}" if mean_psnr is not None else "--"
    std_psnr = summary["psnr_db"]["std"]

    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>BENCH-017 full-frame arm &middot; {escape(Path(config['args']['out']).name)}</title>
<style>
  :root {{ color-scheme: light dark;
    --surface-0:#f4f3f0; --surface-1:#fcfcfb; --border:#dedcd5;
    --text-primary:#0b0b0b; --text-secondary:#52514e; --text-muted:#77756e;
    --series-1:#2a78d6; --warn-bg:#fdf6e3; --warn-bd:#eda100; --fail:#e34948; }}
  @media (prefers-color-scheme: dark) {{ :root:where(:not([data-theme="light"])) {{
    --surface-0:#111110; --surface-1:#1a1a19; --border:#33322f;
    --text-primary:#ffffff; --text-secondary:#c3c2b7; --text-muted:#8f8e85;
    --series-1:#3987e5; --warn-bg:#2a2415; --warn-bd:#c98500; --fail:#e66767; }} }}
  :root[data-theme="dark"] {{
    --surface-0:#111110; --surface-1:#1a1a19; --border:#33322f;
    --text-primary:#ffffff; --text-secondary:#c3c2b7; --text-muted:#8f8e85;
    --series-1:#3987e5; --warn-bg:#2a2415; --warn-bd:#c98500; --fail:#e66767; }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; padding:32px 24px 72px; background:var(--surface-0);
    color:var(--text-primary); font:15px/1.55 ui-sans-serif,system-ui,-apple-system,
    "Segoe UI",Roboto,sans-serif; }}
  .wrap {{ max-width:1180px; margin:0 auto; }}
  h1 {{ font-size:22px; margin:0 0 4px; letter-spacing:-.01em; }}
  h2 {{ font-size:16px; margin:36px 0 12px; letter-spacing:-.01em; }}
  h3 {{ font-size:15px; margin:0; font-family:ui-monospace,SFMono-Regular,Menlo,monospace; }}
  .sub {{ color:var(--text-secondary); margin:0 0 20px; }}
  .muted {{ color:var(--text-muted); }}
  .fail {{ color:var(--fail); }}
  .caveat {{ background:var(--warn-bg); border-left:3px solid var(--warn-bd);
    padding:12px 16px; border-radius:0 8px 8px 0; margin:0 0 24px;
    color:var(--text-secondary); font-size:14px; }}
  .caveat b {{ color:var(--text-primary); }}
  .tiles {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(148px,1fr));
    gap:12px; margin-bottom:8px; }}
  .tile {{ background:var(--surface-1); border:1px solid var(--border);
    border-radius:10px; padding:14px 16px; }}
  .tile .k {{ font-size:12px; color:var(--text-muted); text-transform:uppercase;
    letter-spacing:.04em; }}
  .tile .v {{ font-size:26px; font-variant-numeric:tabular-nums; letter-spacing:-.02em;
    margin-top:2px; }}
  .tile .n {{ font-size:12px; color:var(--text-secondary); }}
  .panel {{ background:var(--surface-1); border:1px solid var(--border);
    border-radius:10px; padding:18px; overflow-x:auto; }}
  .chart {{ width:100%; height:auto; display:block; min-width:520px; }}
  .chart .bar {{ fill:var(--series-1); }}
  .chart .bar-g:hover .bar {{ opacity:.82; }}
  .chart .cat {{ fill:var(--text-secondary); font-size:12px;
    font-family:ui-monospace,SFMono-Regular,Menlo,monospace; }}
  .chart .val {{ fill:var(--text-primary); font-size:12px; font-variant-numeric:tabular-nums; }}
  table {{ border-collapse:collapse; width:100%; font-size:14px; min-width:720px; }}
  th,td {{ text-align:left; padding:7px 12px; border-bottom:1px solid var(--border);
    white-space:nowrap; }}
  th {{ color:var(--text-muted); font-weight:600; font-size:12px; text-transform:uppercase;
    letter-spacing:.04em; }}
  td.num {{ text-align:right; font-variant-numeric:tabular-nums; }}
  td:first-child {{ font-family:ui-monospace,SFMono-Regular,Menlo,monospace; }}
  .card {{ background:var(--surface-1); border:1px solid var(--border); border-radius:10px;
    padding:16px; margin-bottom:16px; }}
  .card-head {{ display:flex; flex-wrap:wrap; gap:10px; align-items:baseline;
    justify-content:space-between; margin-bottom:12px; }}
  .chips {{ display:flex; flex-wrap:wrap; gap:6px; }}
  .chip {{ font-size:12px; color:var(--text-secondary); background:var(--surface-0);
    border:1px solid var(--border); border-radius:999px; padding:2px 10px;
    font-variant-numeric:tabular-nums; }}
  .chip b {{ color:var(--text-primary); }}
  .triptych {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(240px,1fr)); gap:12px; }}
  figure {{ margin:0; }}
  figure img {{ width:100%; height:auto; display:block; border-radius:6px;
    border:1px solid var(--border); background:var(--surface-0); }}
  figcaption {{ font-size:12px; color:var(--text-muted); margin-top:6px; }}
  .meta {{ font-size:12px; color:var(--text-muted); margin:12px 0 0; }}
  a {{ color:var(--series-1); }}
  code {{ font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:13px;
    color:var(--text-secondary); }}
</style></head><body><div class="wrap">

<h1>Full-frame arm on unmasked images</h1>
<p class="sub"><code>{escape(arm)}</code> &middot; recipe <code>{escape(recipe)}</code>
  &middot; {len(ok)} of {len(rows)} images completed &middot; {escape(str(gpu))}</p>

<p class="caveat"><b>Exploratory, not a screen.</b> One arm, one seed, no control and no frozen
gate, so nothing here promotes a claim. BENCH-017 is the preregistered comparison
(full-frame arm vs. <code>plain_fit_shipped</code> vs. the C12 pinned default, &ge;3 seeds, gate
declared before the first fit). Numbers below say where this arm lands on these images, not that
it wins. CUDA renders are not bit-reproducible.</p>

<div class="tiles">
  <div class="tile"><div class="k">Mean PSNR</div><div class="v">{hero}</div>
    <div class="n">dB &middot; &sigma; {_fmt(std_psnr, 2)} over {summary['psnr_db']['n']}</div></div>
  <div class="tile"><div class="k">Mean MS-SSIM</div>
    <div class="v">{_fmt(summary['ms_ssim']['mean'], 4)}</div>
    <div class="n">min {_fmt(summary['ms_ssim']['min'], 4)}</div></div>
  <div class="tile"><div class="k">Mean LPIPS</div>
    <div class="v">{_fmt(summary['lpips']['mean'], 4)}</div>
    <div class="n">lower is better</div></div>
  <div class="tile"><div class="k">Rows</div>
    <div class="v">{_fmt(summary['n_gaussians']['mean'], 0)}</div>
    <div class="n">capacity {config['args'].get('capacity')}</div></div>
  <div class="tile"><div class="k">Mean time</div>
    <div class="v">{_fmt(summary['seconds']['mean'], 0)}</div><div class="n">seconds/image</div></div>
  <div class="tile"><div class="k">Accepted steps</div>
    <div class="v">{_fmt(summary['accepted_steps']['mean'], 0)}</div>
    <div class="n">of {_fmt(summary['attempted_steps']['mean'], 0)} attempted</div></div>
</div>

<h2>PSNR by image</h2>
<div class="panel">{_psnr_chart(rows)}</div>

<h2>Per-image metrics</h2>
<div class="panel">{_table(rows)}</div>

<h2>Visual inspection</h2>
<p class="sub muted">Source, reconstruction, and |error| at a fixed 0&ndash;{ERROR_SCALE} scale so
maps are comparable across images. Click any panel for full resolution.</p>
{_cards(rows)}

<h2>Provenance</h2>
<div class="panel"><p class="muted" style="margin:0">
  <code>config.json</code> (resolved args, device, versions) &middot;
  <code>records.jsonl</code> (one row per image) &middot;
  <code>summary.json</code> &middot; per-image <code>.npz</code> fields and
  <code>_pipeline.json</code> histories under <code>assets/</code>.<br>
  torch {escape(str(versions.get('torch')))} &middot; numpy
  {escape(str(versions.get('numpy')))} &middot; CUDA {escape(str(versions.get('cuda')))}
  &middot; seed {config['args'].get('seed')}
</p></div>

</div></body></html>"""
    path = outdir / "index.html"
    path.write_text(html)
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--images", nargs="*", default=[],
                    help="image files or directories of unmasked images")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--capacity", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--step-scale", type=float, default=1.0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--renderer", default=None, choices=[None, "normalized", "cuda", "cuda_tiled"])
    ap.add_argument("--limit", type=int, default=None, help="fit at most this many images")
    ap.add_argument("--report-only", action="store_true",
                    help="rebuild index.html from an existing records.jsonl")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.report_only:
        print(f"wrote {build_report(outdir)}")
        return 0

    images = discover_images(args.images)
    if args.limit:
        images = images[: args.limit]

    import torch

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    write_config(outdir, args, images, device)

    records_path = outdir / "records.jsonl"
    done = set()
    if records_path.exists():
        for line in records_path.read_text().splitlines():
            if line.strip():
                done.add(json.loads(line)["image"])

    lut = _ramp_lut()
    for i, path in enumerate(images, 1):
        if path.name in done:
            print(f"[{i}/{len(images)}] {path.name}: already recorded, skipping", flush=True)
            continue
        print(f"[{i}/{len(images)}] {path.name}: fitting", flush=True)
        try:
            record = run_one(path, args, outdir, lut)
            print(f"    PSNR {record['psnr_db']:.2f} dB  MS-SSIM "
                  f"{_fmt(record.get('ms_ssim'), 4)}  {record['seconds']:.0f}s", flush=True)
        except Exception as exc:  # isolate per-image failure; the sweep finishes (BENCH-002)
            record = {"image": path.name, "source_path": str(path), "status": "error",
                      "error": f"{type(exc).__name__}: {exc}"}
            print(f"    FAILED: {record['error']}", flush=True)
        with records_path.open("a") as fh:
            fh.write(json.dumps(record) + "\n")
        build_report(outdir)  # keep the page viewable mid-run

    print(f"wrote {build_report(outdir)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
