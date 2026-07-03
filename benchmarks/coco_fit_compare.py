"""Legacy matched visual fit comparison on four COCO images.

This script intentionally uses StructSplat's reference renderer/fitter for every row. The
comparison isolates placement/growth policy instead of mixing repo-specific CUDA kernels, losses,
and codec definitions. Native cross-repo runs require additional extension builds/checkpoints.
It is superseded for new comparisons by ``cross_repo_matrix_compare.py``, which carries the same
four-image protocol forward with multiple resolutions, seeds, iteration counts, and extra metrics.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from benchmarks.common import ANALOGUE_LABELS, ANALOGUE_METHODS, build_comparison_analogue
from benchmarks.common import load_image as _load_image
from benchmarks.common import resolve_seeds, run_config, save_image as _save_image
from benchmarks.common import target_tensor as _target_tensor
from benchmarks.common import write_config, write_csv, write_json
from structsplat.config import FitConfig, InitConfig, StructureTensorConfig
from structsplat.fit import fit
from structsplat.init import build_field


DEFAULT_IMAGES = [
    "COCO_train2014_000000000009.jpg",
    "COCO_train2014_000000000025.jpg",
    "COCO_train2014_000000000030.jpg",
    "COCO_train2014_000000000034.jpg",
]

DEFAULT_DATASET_DIR = Path("tests/test_images")


METHODS = [
    "structsplat",
    *ANALOGUE_METHODS,
]


METHOD_LABELS = {
    "structsplat": "StructSplat",
    **ANALOGUE_LABELS,
}


METHOD_NOTES = {
    "structsplat": "aniso_flanking + tensor density + WSE",
    "gaussianimage": "random fixed-count init, GaussianImage-style baseline",
    "gaussianimage_plus": "random half-budget init + residual top-error additions",
    "image_gs": "gradient-density init + residual progressive additions",
    "instant_gi_quadtree": "Instant-GI quadtree fallback; learned checkpoint absent locally",
}


def _fit_method(
    method: str,
    img: np.ndarray,
    image_path: Path,
    target: torch.Tensor,
    budget: int,
    seed: int,
    base_fit: FitConfig,
    scfg: StructureTensorConfig,
    device: str,
) -> dict:
    start_n = budget
    fcfg = base_fit

    if method == "structsplat":
        field = build_field(
            img,
            InitConfig(strategy="aniso_flanking", num_gaussians=budget, seed=seed),
            scfg,
            device=device,
        )
    elif method in ANALOGUE_METHODS:
        field, fcfg, _init_seconds, start_n = build_comparison_analogue(
            method, img, image_path, budget, seed, device, base_fit, scfg
        )
    else:
        raise ValueError(f"unknown method {method!r}")

    out = fit(field, target, fcfg, verbose=False)
    return {
        "method": method,
        "method_label": METHOD_LABELS[method],
        "method_note": METHOD_NOTES[method],
        "start_gaussians": start_n,
        "n_gaussians": int(out["n_gaussians"]),
        "psnr": round(float(out["psnr"]), 4),
        "ssim": round(float(out["ssim"]), 5),
        "ms_ssim": round(float(out["ms_ssim"]), 5),
        "fit_seconds": round(float(out["fit_seconds"]), 4),
        "render": out["render"].detach().clamp(0, 1).cpu().numpy(),
        "history": out.get("history", {}),
    }


def _tile_with_label(img: Image.Image, title: str, subtitle: str, w: int, h: int) -> Image.Image:
    label_h = 42
    canvas = Image.new("RGB", (w, h + label_h), "white")
    canvas.paste(img.resize((w, h), Image.Resampling.LANCZOS), (0, 0))
    draw = ImageDraw.Draw(canvas)
    try:
        font_bold = ImageFont.truetype("DejaVuSans-Bold.ttf", 13)
        font = ImageFont.truetype("DejaVuSans.ttf", 11)
    except Exception:
        font_bold = ImageFont.load_default()
        font = ImageFont.load_default()
    draw.text((6, h + 4), title, fill=(0, 0, 0), font=font_bold)
    draw.text((6, h + 22), subtitle, fill=(60, 60, 60), font=font)
    return canvas


def _make_grid(rows: list[dict], selected: list[dict], outdir: Path, seed: int,
               *, include_seed_in_name: bool) -> None:
    methods = METHODS
    thumb_w = 180
    thumb_h = 135
    gap = 8
    cols = 1 + len(methods)
    cell_h = thumb_h + 42
    grid = Image.new("RGB", (cols * thumb_w + (cols - 1) * gap, len(selected) * cell_h + (len(selected) - 1) * gap), "white")

    by_image_method = {(r["image"], r["method"]): r for r in rows if r["seed"] == seed}
    for ridx, item in enumerate(selected):
        y = ridx * (cell_h + gap)
        original = Image.open(item["selected_path"]).convert("RGB")
        label = item["image"].replace("COCO_train2014_", "COCO ")
        original_tile = _tile_with_label(original, label, "target", thumb_w, thumb_h)
        grid.paste(original_tile, (0, y))
        for midx, method in enumerate(methods, 1):
            row = by_image_method[(item["image"], method)]
            rec = Image.open(row["reconstruction_path"]).convert("RGB")
            title = METHOD_LABELS[method]
            subtitle = f"{row['psnr']:.2f} dB | {row['n_gaussians']} G"
            grid.paste(_tile_with_label(rec, title, subtitle, thumb_w, thumb_h), (midx * (thumb_w + gap), y))
    name = f"comparison_grid_seed{seed}.png" if include_seed_in_name else "comparison_grid.png"
    grid.save(outdir / name)


def _write_summary(rows: list[dict], selected: list[dict], outdir: Path, budget: int,
                   iters: int, max_side: int, seeds: list[int]) -> None:
    lines = [
        "# COCO Fit Comparison",
        "",
        "Matched reference comparison using StructSplat's renderer/fitter for all methods.",
        "",
        f"- Images: {len(selected)} COCO train2014 images",
        f"- Max side: {max_side}px",
        f"- Budget: {budget} Gaussians cap",
        f"- Iterations: {iters}",
        f"- Seeds: {', '.join(str(s) for s in seeds)}",
        f"- Device: {'cuda' if torch.cuda.is_available() else 'cpu'}",
        "- Aggregate rows report mean and population std over image x seed runs.",
        "",
        "## Method Mapping",
        "",
    ]
    for method in METHODS:
        lines.append(f"- **{METHOD_LABELS[method]}**: {METHOD_NOTES[method]}")
    lines += [
        "",
        "## Mean Metrics",
        "",
        "| Method | Runs | Mean PSNR | PSNR Std | Mean MS-SSIM | MS-SSIM Std | Mean seconds | Seconds Std |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    from benchmarks.common import fmt_mean_std

    def _ms(vals, key, digits):
        return fmt_mean_std((r[key] for r in vals), digits)

    for method in METHODS:
        vals = [r for r in rows if r["method"] == method]
        # a method with zero ok rows (e.g. Instant-GI unavailable) must not raise StatisticsError
        # and void the whole run (BENCH-002)
        psnr_m, psnr_s = _ms(vals, "psnr", 3)
        ms_m, ms_s = _ms(vals, "ms_ssim", 5)
        sec_m, sec_s = _ms(vals, "fit_seconds", 2)
        lines.append(
            f"| {METHOD_LABELS[method]} | {len(vals)} | {psnr_m} | {psnr_s} | "
            f"{ms_m} | {ms_s} | {sec_m} | {sec_s} |"
        )
    lines += ["", "## Per-Image PSNR", "", "| Image | " + " | ".join(METHOD_LABELS[m] for m in METHODS) + " |"]
    lines.append("|---|" + "---:|" * len(METHODS))
    for item in selected:
        cells = []
        for method in METHODS:
            vals = [r for r in rows if r["image"] == item["image"] and r["method"] == method]
            mean_cell, std_cell = fmt_mean_std((r["psnr"] for r in vals), 2)
            cells.append("-" if mean_cell == "-" else f"{mean_cell} ({std_cell})")
        lines.append(f"| {item['image']} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "## Caveat",
        "",
        "This is not a native CUDA-codec benchmark. Native runs were blocked locally by missing or",
        "unbuilt dependencies/checkpoints, so this isolates initialization/growth policy under one",
        "renderer and loss.",
        "",
        "Visual grid: `comparison_grid.png`.",
    ]
    (outdir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(
    dataset_dir: Path,
    outdir: Path,
    image_names: list[str],
    budget: int,
    iters: int,
    max_side: int,
    seed: int | None = None,
    device: str | None = None,
    seeds: list[int] | tuple[int, ...] | None = None,
) -> list[dict]:
    seeds = resolve_seeds(seed, seeds)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    outdir.mkdir(parents=True, exist_ok=True)
    write_config(str(outdir), run_config({
        "dataset_dir": str(dataset_dir), "image_names": image_names, "budget": budget,
        "iters": iters, "max_side": max_side, "seeds": seeds,
    }, device=device))
    selected_dir = outdir / "selected"
    recon_dir = outdir / "reconstructions"
    selected_dir.mkdir(exist_ok=True)
    recon_dir.mkdir(exist_ok=True)

    base_fit = FitConfig(
        iters=iters,
        render_chunk=384,
        log_every=max(1, iters // 4),
        target_psnrs=[25.0, 28.0, 30.0],
        compute_lpips=False,
    )
    scfg = StructureTensorConfig()

    rows: list[dict] = []
    selected: list[dict] = []
    for image_name in image_names:
        src = dataset_dir / image_name
        img = _load_image(src, max_side)
        stem = Path(image_name).stem
        resized_path = selected_dir / f"{stem}.png"
        _save_image(img, resized_path)
        selected.append({"image": stem, "source_path": str(src), "selected_path": str(resized_path)})
        target = _target_tensor(img, device)
        for seed_value in seeds:
            for method in METHODS:
                print(
                    f"[{stem}] seed={seed_value} {METHOD_LABELS[method]} "
                    f"budget={budget} iters={iters}",
                    flush=True,
                )
                try:
                    rec = _fit_method(
                        method, img, resized_path, target, budget, seed_value, base_fit, scfg,
                        device,
                    )
                    recon_path = recon_dir / f"{stem}_seed{seed_value}_{method}.png"
                    _save_image(rec.pop("render"), recon_path)
                    hist = rec.pop("history")
                    row = {
                        "image": stem,
                        "source_path": str(src),
                        "width": img.shape[1],
                        "height": img.shape[0],
                        "budget": budget,
                        "iters": iters,
                        "seed": seed_value,
                        **rec,
                        "reconstruction_path": str(recon_path),
                        "history": hist,
                        "status": "ok",
                        "error": "",
                    }
                except Exception as exc:
                    row = {
                        "image": stem,
                        "source_path": str(src),
                        "width": img.shape[1],
                        "height": img.shape[0],
                        "budget": budget,
                        "iters": iters,
                        "seed": seed_value,
                        "method": method,
                        "method_label": METHOD_LABELS[method],
                        "method_note": METHOD_NOTES[method],
                        "start_gaussians": 0,
                        "n_gaussians": 0,
                        "psnr": float("nan"),
                        "ssim": float("nan"),
                        "ms_ssim": float("nan"),
                        "fit_seconds": 0.0,
                        "reconstruction_path": "",
                        "history": {},
                        "status": "error",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                    print(f"  ERROR {row['error']}", flush=True)
                rows.append(row)
                if row["status"] == "ok":
                    print(
                        f"  psnr={row['psnr']:.3f} ms_ssim={row['ms_ssim']:.5f} "
                        f"n={row['n_gaussians']} sec={row['fit_seconds']:.2f}",
                        flush=True,
                    )

    json_rows = [{k: v for k, v in r.items() if k != "history"} for r in rows]
    write_json(outdir / "metrics.json", json_rows)
    if json_rows:
        write_csv(outdir / "metrics.csv", json_rows, fieldnames=[k for k in json_rows[0].keys()])
    ok_rows = [r for r in rows if r["status"] == "ok"]
    for seed_value in seeds:
        seed_rows = [r for r in ok_rows if r["seed"] == seed_value]
        if len(seed_rows) == len(selected) * len(METHODS):
            _make_grid(ok_rows, selected, outdir, seed_value, include_seed_in_name=len(seeds) > 1)
    _write_summary(ok_rows, selected, outdir, budget, iters, max_side, seeds)
    return rows


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Matched COCO fit comparison for 2D Gaussian approaches")
    p.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR,
                   help="directory containing the four benchmark images")
    p.add_argument("--outdir", type=Path, default=Path("results/coco_fit_compare"))
    p.add_argument("--images", nargs="+", default=DEFAULT_IMAGES)
    p.add_argument("--budget", type=int, default=768)
    p.add_argument("--iters", type=int, default=120)
    p.add_argument("--max-side", type=int, default=192)
    from benchmarks.common import add_seed_args

    add_seed_args(p)
    p.add_argument("--device", default=None)
    args = p.parse_args()
    run(
        args.dataset_dir, args.outdir, args.images, args.budget, args.iters, args.max_side,
        args.seed, args.device, resolve_seeds(args.seed, args.seeds),
    )


if __name__ == "__main__":
    main()
