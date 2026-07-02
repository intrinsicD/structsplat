"""Matched visual fit comparison on four COCO images.

This script intentionally uses StructSplat's reference renderer/fitter for every row. The
comparison isolates placement/growth policy instead of mixing repo-specific CUDA kernels, losses,
and codec definitions. Native cross-repo runs require additional extension builds/checkpoints.
"""
from __future__ import annotations

import csv
import importlib.util
import json
import math
import os
from dataclasses import asdict
from pathlib import Path
from statistics import mean
from typing import Callable

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from structsplat.config import FitConfig, InitConfig, StructureTensorConfig
from structsplat.fit import fit
from structsplat.gaussians import GaussianField
from structsplat.init import build_field
DEFAULT_IMAGES = [
    "COCO_train2014_000000000009.jpg",
    "COCO_train2014_000000000025.jpg",
    "COCO_train2014_000000000030.jpg",
    "COCO_train2014_000000000034.jpg",
]


METHODS = [
    "structsplat",
    "gaussianimage",
    "gaussianimage_plus",
    "image_gs",
    "instant_gi_quadtree",
]


METHOD_LABELS = {
    "structsplat": "StructSplat",
    "gaussianimage": "GaussianImage",
    "gaussianimage_plus": "GaussianImage++",
    "image_gs": "Image-GS",
    "instant_gi_quadtree": "Instant-GI qt",
}


METHOD_NOTES = {
    "structsplat": "aniso_flanking + tensor density + WSE",
    "gaussianimage": "random fixed-count init, GaussianImage-style baseline",
    "gaussianimage_plus": "random half-budget init + residual top-error additions",
    "image_gs": "gradient-density init + residual progressive additions",
    "instant_gi_quadtree": "Instant-GI quadtree fallback; learned checkpoint absent locally",
}


def _load_image(path: Path, max_side: int) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    scale = max_side / max(img.size)
    if scale < 1.0:
        img = img.resize((round(img.size[0] * scale), round(img.size[1] * scale)), Image.Resampling.LANCZOS)
    return np.asarray(img, dtype=np.float32) / 255.0


def _save_image(arr: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = np.clip(arr * 255.0 + 0.5, 0, 255).astype(np.uint8)
    Image.fromarray(img).save(path)


def _target_tensor(img: np.ndarray, device: str) -> torch.Tensor:
    return torch.as_tensor(img, dtype=torch.float32, device=device)


def _residual_cfg(base: FitConfig, budget: int, start_n: int, add_times: int = 4) -> FitConfig:
    add_total = max(0, budget - start_n)
    split_count = max(1, math.ceil(add_total / max(1, add_times))) if add_total else 0
    split_every = max(1, base.iters // (add_times + 1)) if add_total else None
    return FitConfig(
        **{
            **asdict(base),
            "split_every": split_every,
            "split_count": split_count,
            "split_mode": "residual_add",
            "max_gaussians": budget,
        }
    )


def _build_quadtree_field(img: np.ndarray, image_path: Path, budget: int, seed: int, device: str) -> GaussianField:
    qpath = Path("/home/alex/Documents/Instant-GI/quard_image.py")
    spec = importlib.util.spec_from_file_location("instant_gi_quard_image", qpath)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {qpath}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    points, _elapsed = mod.QuardImage(image_path).split()
    if points.shape[0] > budget:
        rng = np.random.default_rng(seed)
        keep = np.sort(rng.choice(points.shape[0], size=budget, replace=False))
        points = points[keep]

    H, W = img.shape[:2]
    means = np.empty((points.shape[0], 2), dtype=np.float32)
    means[:, 0] = (points[:, 0] + 1.0) * 0.5 * (W - 1)
    means[:, 1] = (points[:, 1] + 1.0) * 0.5 * (H - 1)
    scales = np.maximum(points[:, 2:4].astype(np.float32), 0.5)
    rotations = (points[:, 4].astype(np.float32) * (2.0 * np.pi)) % np.pi
    colors = _bilinear(img, means)
    return GaussianField.from_numpy(means, scales, rotations, colors, None, device=device)


def _bilinear(img: np.ndarray, pts: np.ndarray) -> np.ndarray:
    H, W = img.shape[:2]
    x = np.clip(pts[:, 0], 0, W - 1.001)
    y = np.clip(pts[:, 1], 0, H - 1.001)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    x1, y1 = np.clip(x0 + 1, 0, W - 1), np.clip(y0 + 1, 0, H - 1)
    fx, fy = (x - x0)[:, None], (y - y0)[:, None]
    return (
        img[y0, x0] * (1 - fx) * (1 - fy)
        + img[y0, x1] * fx * (1 - fy)
        + img[y1, x0] * (1 - fx) * fy
        + img[y1, x1] * fx * fy
    ).astype(np.float32)


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
    elif method == "gaussianimage":
        field = build_field(
            img,
            InitConfig(strategy="random", num_gaussians=budget, seed=seed),
            scfg,
            device=device,
        )
    elif method == "gaussianimage_plus":
        start_n = max(16, budget // 2)
        field = build_field(
            img,
            InitConfig(strategy="random", num_gaussians=start_n, seed=seed),
            scfg,
            device=device,
        )
        fcfg = _residual_cfg(base_fit, budget, start_n)
    elif method == "image_gs":
        start_n = max(16, budget // 2)
        field = build_field(
            img,
            InitConfig(
                strategy="iso_blue_noise",
                density_mode="gradient",
                sampling_mode="density_random",
                num_gaussians=start_n,
                seed=seed,
            ),
            scfg,
            device=device,
        )
        fcfg = _residual_cfg(base_fit, budget, start_n)
    elif method == "instant_gi_quadtree":
        field = _build_quadtree_field(img, image_path, budget, seed, device)
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


def _make_grid(rows: list[dict], selected: list[dict], outdir: Path) -> None:
    methods = METHODS
    thumb_w = 180
    thumb_h = 135
    gap = 8
    cols = 1 + len(methods)
    cell_h = thumb_h + 42
    grid = Image.new("RGB", (cols * thumb_w + (cols - 1) * gap, len(selected) * cell_h + (len(selected) - 1) * gap), "white")

    by_image_method = {(r["image"], r["method"]): r for r in rows}
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
    grid.save(outdir / "comparison_grid.png")


def _write_summary(rows: list[dict], selected: list[dict], outdir: Path, budget: int, iters: int, max_side: int) -> None:
    lines = [
        "# COCO Fit Comparison",
        "",
        "Matched reference comparison using StructSplat's renderer/fitter for all methods.",
        "",
        f"- Images: {len(selected)} COCO train2014 images",
        f"- Max side: {max_side}px",
        f"- Budget: {budget} Gaussians cap",
        f"- Iterations: {iters}",
        f"- Device: {'cuda' if torch.cuda.is_available() else 'cpu'}",
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
        "| Method | Mean PSNR | Mean MS-SSIM | Mean seconds |",
        "|---|---:|---:|---:|",
    ]
    for method in METHODS:
        vals = [r for r in rows if r["method"] == method]
        lines.append(
            f"| {METHOD_LABELS[method]} | {mean(r['psnr'] for r in vals):.3f} | "
            f"{mean(r['ms_ssim'] for r in vals):.5f} | {mean(r['fit_seconds'] for r in vals):.2f} |"
        )
    lines += ["", "## Per-Image PSNR", "", "| Image | " + " | ".join(METHOD_LABELS[m] for m in METHODS) + " |"]
    lines.append("|---|" + "---:|" * len(METHODS))
    for item in selected:
        cells = []
        for method in METHODS:
            row = next(r for r in rows if r["image"] == item["image"] and r["method"] == method)
            cells.append(f"{row['psnr']:.2f}")
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
    seed: int,
    device: str | None = None,
) -> list[dict]:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    outdir.mkdir(parents=True, exist_ok=True)
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
        for method in METHODS:
            print(f"[{stem}] {METHOD_LABELS[method]} budget={budget} iters={iters}", flush=True)
            try:
                rec = _fit_method(method, img, resized_path, target, budget, seed, base_fit, scfg, device)
                recon_path = recon_dir / f"{stem}_{method}.png"
                _save_image(rec.pop("render"), recon_path)
                hist = rec.pop("history")
                row = {
                    "image": stem,
                    "source_path": str(src),
                    "width": img.shape[1],
                    "height": img.shape[0],
                    "budget": budget,
                    "iters": iters,
                    "seed": seed,
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
                    "seed": seed,
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
    (outdir / "metrics.json").write_text(json.dumps(json_rows, indent=2), encoding="utf-8")
    with (outdir / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        fields = [k for k in json_rows[0].keys()]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(json_rows)
    ok_rows = [r for r in rows if r["status"] == "ok"]
    if len(ok_rows) == len(rows):
        _make_grid(ok_rows, selected, outdir)
    _write_summary(ok_rows, selected, outdir, budget, iters, max_side)
    return rows


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Matched COCO fit comparison for 2D Gaussian approaches")
    p.add_argument("--dataset-dir", type=Path, default=Path("/home/alex/Documents/datasets/train2014"))
    p.add_argument("--outdir", type=Path, default=Path("results/coco_fit_compare"))
    p.add_argument("--images", nargs="+", default=DEFAULT_IMAGES)
    p.add_argument("--budget", type=int, default=768)
    p.add_argument("--iters", type=int, default=120)
    p.add_argument("--max-side", type=int, default=192)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    args = p.parse_args()
    run(args.dataset_dir, args.outdir, args.images, args.budget, args.iters, args.max_side, args.seed, args.device)


if __name__ == "__main__":
    main()
