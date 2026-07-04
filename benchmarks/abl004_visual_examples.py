"""Generate ABL-004 visual comparison sheets.

The sheets are intentionally small-run visuals, not a replacement for the staged numerical
screen. They compare the current ABL-004 finalist/control variants across the six visual levels
used by the cross-repo matrix: 160/240/320 px x 80/200 fit iterations.
"""
from __future__ import annotations

import argparse
import time
from dataclasses import asdict
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from benchmarks.common import (
    json_safe_rows,
    load_image,
    run_config,
    save_image,
    write_config,
    write_csv,
    write_json,
)
from benchmarks.cross_repo_matrix_compare import _budget_for_size
from structsplat import metrics as M
from structsplat.config import FitConfig, InitConfig, StructureTensorConfig
from structsplat.fit import fit
from structsplat.init import build_field


DEFAULT_DATASET_DIR = Path("results/datasets/abl004/kodak24")
DEFAULT_IMAGES = ["kodim01.png", "kodim07.png", "kodim10.png", "kodim13.png", "kodim22.png"]
DEFAULT_MAX_SIDES = [160, 240, 320]
DEFAULT_ITERS = [80, 200]

VARIANTS = [
    "aniso_onedge",
    "aniso_flanking",
    "quadtree_wse",
    "quadtree_hybrid",
    "iso_blue_noise",
    "floyd_steinberg",
]

VARIANT_LABELS = {
    "aniso_onedge": "Aniso on-edge",
    "aniso_flanking": "Aniso flanking",
    "quadtree_wse": "Quadtree WSE",
    "quadtree_hybrid": "Quadtree hybrid",
    "iso_blue_noise": "Iso blue noise",
    "floyd_steinberg": "Floyd-Steinberg",
}

VARIANT_NOTES = {
    "aniso_onedge": "anisotropic WSE centers placed on features",
    "aniso_flanking": "anisotropic WSE centers flanking features, offset 0.5",
    "quadtree_wse": "quadtree allocation with local WSE/flanking samples",
    "quadtree_hybrid": "aggregate smooth cells and sampled detailed cells",
    "iso_blue_noise": "isotropic weighted sample elimination baseline",
    "floyd_steinberg": "same flanking thesis config with Floyd-Steinberg placement",
}


def _variant_init(variant: str, budget: int, seed: int) -> InitConfig:
    if variant == "floyd_steinberg":
        return InitConfig(
            strategy="aniso_flanking",
            num_gaussians=budget,
            seed=seed,
            sampling_mode="floyd_steinberg",
            flank_offset_frac=0.5,
        )
    if variant == "aniso_flanking":
        return InitConfig(
            strategy="aniso_flanking",
            num_gaussians=budget,
            seed=seed,
            sampling_mode="wse",
            flank_offset_frac=0.5,
        )
    if variant == "aniso_onedge":
        return InitConfig(
            strategy="aniso_onedge",
            num_gaussians=budget,
            seed=seed,
            sampling_mode="wse",
            flank_offset_frac=0.0,
        )
    if variant in {"quadtree_wse", "quadtree_hybrid", "iso_blue_noise"}:
        return InitConfig(
            strategy=variant,
            num_gaussians=budget,
            seed=seed,
            sampling_mode="wse",
            flank_offset_frac=0.0,
        )
    raise ValueError(f"unknown ABL-004 visual variant {variant!r}")


def _fit_config(iters: int, renderer: str) -> FitConfig:
    return FitConfig(
        iters=iters,
        log_every=max(1, iters // 10),
        render_chunk=512,
        renderer=renderer,
        pixel_loss="charbonnier",
        compute_lpips=False,
        target_psnrs=[22.0, 24.0, 26.0, 28.0, 30.0],
    )


def _grid_metric(row: dict[str, Any], key: str, digits: int) -> str:
    value = row.get(key)
    if value in (None, "", "None"):
        return "-"
    return f"{float(value):.{digits}f}"


def _tile_with_label(img: Image.Image, title: str, subtitle: str, w: int, h: int) -> Image.Image:
    label_h = 48
    canvas = Image.new("RGB", (w, h + label_h), "white")
    canvas.paste(img.resize((w, h), Image.Resampling.LANCZOS), (0, 0))
    draw = ImageDraw.Draw(canvas)
    try:
        font_bold = ImageFont.truetype("DejaVuSans-Bold.ttf", 12)
        font = ImageFont.truetype("DejaVuSans.ttf", 10)
    except Exception:
        font_bold = ImageFont.load_default()
        font = ImageFont.load_default()
    draw.text((5, h + 4), title, fill=(0, 0, 0), font=font_bold)
    draw.text((5, h + 22), subtitle, fill=(55, 55, 55), font=font)
    return canvas


def _blank_tile(w: int, h: int, text: str = "not available") -> Image.Image:
    img = Image.new("RGB", (w, h + 48), (242, 242, 242))
    draw = ImageDraw.Draw(img)
    draw.rectangle((0, 0, w - 1, h + 47), outline=(190, 190, 190))
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 12)
    except Exception:
        font = ImageFont.load_default()
    box = draw.textbbox((0, 0), text, font=font)
    draw.text(((w - (box[2] - box[0])) / 2, (h - (box[3] - box[1])) / 2),
              text, fill=(90, 90, 90), font=font)
    return img


def _make_image_grid(
    image_stem: str,
    selected: dict[tuple[int, int], dict[str, Any]],
    rows: list[dict[str, Any]],
    outdir: Path,
    max_sides: list[int],
    iters_list: list[int],
    seed: int,
) -> Path:
    by_key = {
        (int(r["max_side"]), int(r["iters"]), int(r["seed"]), r["variant"]): r
        for r in rows
        if r["image"] == image_stem
    }
    thumb_w, thumb_h, gap = 220, 132, 8
    cols = 1 + len(VARIANTS)
    levels = [(m, it) for m in max_sides for it in iters_list]
    cell_h = thumb_h + 48
    grid = Image.new(
        "RGB",
        (cols * thumb_w + (cols - 1) * gap, len(levels) * cell_h + (len(levels) - 1) * gap),
        "white",
    )
    for ridx, (max_side, iters) in enumerate(levels):
        y = ridx * (cell_h + gap)
        info = selected[(max_side, seed)]
        target = Image.open(info["selected_path"]).convert("RGB")
        title = f"{image_stem} | {max_side}px / {iters}it"
        subtitle = f"target | cap {info['budget_cap']}G | seed {seed}"
        grid.paste(_tile_with_label(target, title, subtitle, thumb_w, thumb_h), (0, y))
        for vidx, variant in enumerate(VARIANTS, 1):
            row = by_key.get((max_side, iters, seed, variant))
            x = vidx * (thumb_w + gap)
            if row is None or row.get("status") != "ok":
                grid.paste(_blank_tile(thumb_w, thumb_h), (x, y))
                continue
            rec = Image.open(row["reconstruction_path"]).convert("RGB")
            subtitle = (
                f"P {_grid_metric(row, 'psnr', 2)} | "
                f"MS {_grid_metric(row, 'ms_ssim', 4)} | "
                f"{int(row['n_gaussians'])}G"
            )
            grid.paste(
                _tile_with_label(rec, VARIANT_LABELS[variant], subtitle, thumb_w, thumb_h),
                (x, y),
            )
    grid_dir = outdir / "grids"
    grid_dir.mkdir(parents=True, exist_ok=True)
    path = grid_dir / f"{image_stem}_all_levels.png"
    grid.save(path)
    return path


def _write_summary(
    outdir: Path,
    rows: list[dict[str, Any]],
    grids: dict[str, str],
    image_names: list[str],
    max_sides: list[int],
    iters_list: list[int],
    base_side: int,
    base_budget: int,
    seed: int,
    renderer: str,
) -> None:
    ok = [r for r in rows if r["status"] == "ok"]
    lines = [
        "# ABL-004 Visual Examples",
        "",
        "Small matched visual sheets for inspecting the current ABL-004 finalist/control variants.",
        "",
        "## Run Controls",
        "",
        f"- Images: {', '.join(Path(name).stem for name in image_names)}",
        f"- Levels: {'/'.join(str(x) for x in max_sides)} px x {'/'.join(str(x) for x in iters_list)} iterations",
        f"- Budgets: base {base_budget} at {base_side}px, area-scaled per level",
        f"- Seed: {seed}",
        f"- Renderer: {renderer}",
        "- LPIPS: disabled",
        "",
        "## Variants",
        "",
    ]
    for variant in VARIANTS:
        lines.append(f"- **{VARIANT_LABELS[variant]}**: {VARIANT_NOTES[variant]}")

    lines += [
        "",
        "## Example Sheets",
        "",
        "| Image | Grid |",
        "|---|---|",
    ]
    for name in image_names:
        stem = Path(name).stem
        rel = Path(grids[stem]).relative_to(outdir)
        lines.append(f"| {stem} | `{rel}` |")

    lines += [
        "",
        "## Mean Metrics By Level",
        "",
        "| Max side | Iterations | Variant | Budget | PSNR | MS-SSIM | Fit s |",
        "|---:|---:|---|---:|---:|---:|---:|",
    ]
    for max_side in max_sides:
        for iters in iters_list:
            for variant in VARIANTS:
                vals = [
                    r for r in ok
                    if int(r["max_side"]) == max_side
                    and int(r["iters"]) == iters
                    and r["variant"] == variant
                ]
                if not vals:
                    continue
                lines.append(
                    f"| {max_side} | {iters} | {VARIANT_LABELS[variant]} | "
                    f"{int(mean(r['budget_cap'] for r in vals))} | "
                    f"{mean(r['psnr'] for r in vals):.4f} | "
                    f"{mean(r['ms_ssim'] for r in vals):.5f} | "
                    f"{mean(r['fit_seconds'] for r in vals):.3f} |"
                )

    errors = [r for r in rows if r["status"] != "ok"]
    if errors:
        lines += ["", "## Errors", "", "| Cell | Error |", "|---|---|"]
        for row in errors:
            lines.append(
                f"| {row['image']} {row['max_side']}px {row['iters']}it "
                f"{row['variant']} | `{row['error']}` |"
            )

    lines += [
        "",
        "Per-cell reconstructions are under `reconstructions/`; visual sheets are under `grids/`.",
    ]
    (outdir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(
    dataset_dir: Path,
    outdir: Path,
    image_names: list[str],
    max_sides: list[int],
    iters_list: list[int],
    base_budget: int,
    base_side: int,
    seed: int,
    renderer: str,
    device: str | None = None,
) -> list[dict[str, Any]]:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    outdir.mkdir(parents=True, exist_ok=True)
    write_config(str(outdir), run_config({
        "dataset_dir": str(dataset_dir),
        "image_names": image_names,
        "max_sides": max_sides,
        "iters_list": iters_list,
        "base_budget": base_budget,
        "base_side": base_side,
        "seed": seed,
        "variants": VARIANTS,
        "renderer": renderer,
    }, device=device))

    selected_dir = outdir / "selected"
    recon_dir = outdir / "reconstructions"
    selected_dir.mkdir(exist_ok=True)
    recon_dir.mkdir(exist_ok=True)

    selected: dict[str, dict[tuple[int, int], dict[str, Any]]] = {}
    loaded: dict[tuple[str, int], tuple[np.ndarray, Path, int]] = {}
    for image_name in image_names:
        image_path = dataset_dir / image_name
        stem = Path(image_name).stem
        selected[stem] = {}
        for max_side in max_sides:
            budget_cap = _budget_for_size(max_side, base_side, base_budget)
            img = load_image(image_path, max_side)
            resized_path = selected_dir / f"{stem}_{max_side}px.png"
            save_image(img, resized_path)
            selected[stem][(max_side, seed)] = {
                "selected_path": str(resized_path),
                "budget_cap": int(budget_cap),
            }
            loaded[(stem, max_side)] = (img, resized_path, budget_cap)

    rows: list[dict[str, Any]] = []
    total = len(image_names) * len(max_sides) * len(iters_list) * len(VARIANTS)
    done = 0
    scfg = StructureTensorConfig(flat_frac=0.02, corner_frac=0.15)
    for image_name in image_names:
        stem = Path(image_name).stem
        source_path = dataset_dir / image_name
        for max_side in max_sides:
            img, resized_path, budget_cap = loaded[(stem, max_side)]
            target = torch.as_tensor(img, dtype=torch.float32, device=device)
            height, width = img.shape[:2]
            for iters in iters_list:
                fcfg = _fit_config(iters, renderer)
                for variant in VARIANTS:
                    done += 1
                    print(
                        f"[{done}/{total}] {stem} {max_side}px {iters}it "
                        f"seed={seed} {VARIANT_LABELS[variant]} cap={budget_cap}",
                        flush=True,
                    )
                    row_base: dict[str, Any] = {
                        "image": stem,
                        "source_path": str(source_path),
                        "selected_path": str(resized_path),
                        "height": int(height),
                        "width": int(width),
                        "max_side": int(max_side),
                        "iters": int(iters),
                        "budget_cap": int(budget_cap),
                        "seed": int(seed),
                        "renderer": renderer,
                        "variant": variant,
                        "variant_label": VARIANT_LABELS[variant],
                        "variant_note": VARIANT_NOTES[variant],
                    }
                    try:
                        icfg = _variant_init(variant, budget_cap, seed)
                        t0 = time.time()
                        field = build_field(img, icfg, scfg, device=device)
                        init_seconds = time.time() - t0
                        out = fit(field, target, fcfg, verbose=False)
                        render = out["render"].detach().clamp(0, 1)
                        render_np = render.cpu().numpy()
                        recon_path = (
                            recon_dir
                            / stem
                            / f"{max_side}px_{iters}it"
                            / f"{variant}.png"
                        )
                        save_image(render_np, recon_path)
                        row = {
                            **row_base,
                            "status": "ok",
                            "error": "",
                            "init_strategy": icfg.strategy,
                            "sampling_mode": icfg.sampling_mode,
                            "flank_offset_frac": float(icfg.flank_offset_frac),
                            "max_axis_ratio": float(icfg.max_axis_ratio),
                            "coherence_power": float(icfg.coherence_power),
                            "flat_frac": float(scfg.flat_frac),
                            "corner_frac": float(scfg.corner_frac),
                            "init_seconds": float(init_seconds),
                            "fit_seconds": float(out["fit_seconds"]),
                            "total_seconds": float(init_seconds + out["fit_seconds"]),
                            "iterations_run": int(out.get("iterations_run", iters)),
                            "stopped_early": bool(out.get("stopped_early", False)),
                            "n_gaussians": int(out["n_gaussians"]),
                            "psnr": M.psnr(render, target),
                            "ssim": float(M.ssim(render, target)),
                            "ms_ssim": M.ms_ssim(render, target),
                            "iters_to_targets": out.get("iters_to_targets", {}),
                            "fit_config": asdict(fcfg),
                            "init_config": asdict(icfg),
                            "reconstruction_path": str(recon_path),
                        }
                        print(
                            f"  psnr={row['psnr']:.3f} ms={row['ms_ssim']:.5f} "
                            f"n={row['n_gaussians']} total={row['total_seconds']:.3f}s",
                            flush=True,
                        )
                    except Exception as exc:
                        row = {
                            **row_base,
                            "status": "error",
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                        print(f"  ERROR {row['error']}", flush=True)
                    rows.append(row)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

    grids: dict[str, str] = {}
    for image_name in image_names:
        stem = Path(image_name).stem
        grids[stem] = str(
            _make_image_grid(stem, selected[stem], rows, outdir, max_sides, iters_list, seed)
        )

    json_rows = json_safe_rows(rows)
    write_json(outdir / "metrics.json", json_rows)
    if json_rows:
        fieldnames = sorted({k for row in json_rows for k in row.keys() if k not in {"fit_config", "init_config"}})
        write_csv(outdir / "metrics.csv", json_rows, fieldnames=fieldnames, extrasaction="ignore")
    write_json(outdir / "grids.json", grids)
    _write_summary(
        outdir,
        rows,
        grids,
        image_names,
        max_sides,
        iters_list,
        base_side,
        base_budget,
        seed,
        renderer,
    )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate ABL-004 visual comparison sheets")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--outdir", type=Path, default=Path("results/abl004_visual_examples_5img"))
    parser.add_argument("--images", nargs="+", default=DEFAULT_IMAGES)
    parser.add_argument("--max-sides", nargs="+", type=int, default=DEFAULT_MAX_SIDES)
    parser.add_argument("--iters", nargs="+", type=int, default=DEFAULT_ITERS)
    parser.add_argument("--base-budget", type=int, default=640)
    parser.add_argument("--base-side", type=int, default=160)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--renderer", default="cuda")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    run(
        args.dataset_dir,
        args.outdir,
        args.images,
        args.max_sides,
        args.iters,
        args.base_budget,
        args.base_side,
        args.seed,
        args.renderer,
        args.device,
    )


if __name__ == "__main__":
    main()
