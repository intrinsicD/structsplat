"""Multi-resolution matched comparison against local 2D Gaussian image repos.

The rows are repo-inspired method analogues run through StructSplat's current fitter and
exact CUDA renderer. This keeps the comparison executable and isolates initialization /
growth policy; it is not a native codec benchmark for the external repositories.
"""
from __future__ import annotations

import argparse
import math
import time
from dataclasses import asdict, replace
from pathlib import Path
from statistics import mean, pstdev
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from benchmarks.common import build_comparison_analogue
from benchmarks.common import load_image as _load_image
from benchmarks.common import psnr_auc as _psnr_auc
from benchmarks.common import residual_fit_cfg as _residual_fit_cfg
from benchmarks.common import resolve_seeds, run_config, save_image as _save_image, write_config
from benchmarks.common import write_csv, write_json
from structsplat import metrics as M
from structsplat.config import FitConfig, InitConfig, StructureTensorConfig
from structsplat.fit import fit
from structsplat.init import build_field

if TYPE_CHECKING:
    from structsplat.gaussians import GaussianField


DEFAULT_IMAGES = [
    "COCO_train2014_000000000009.jpg",
    "COCO_train2014_000000000025.jpg",
    "COCO_train2014_000000000030.jpg",
    "COCO_train2014_000000000034.jpg",
]

DEFAULT_DATASET_DIR = Path("tests/test_images")

DIFFICULT_IMAGES = {
    "COCO_train2014_000000000025",
    "COCO_train2014_000000000034",
}

METHODS = [
    "structsplat_current",
    "structsplat_shipped_defaults",
    "gaussianimage",
    "gaussianimage_plus",
    "image_gs",
    "instant_gi_quadtree",
]

METHOD_LABELS = {
    "structsplat_current": "StructSplat current",
    "structsplat_shipped_defaults": "StructSplat shipped defaults",
    "gaussianimage": "GaussianImage",
    "gaussianimage_plus": "GaussianImage++",
    "image_gs": "Image-GS",
    "instant_gi_quadtree": "Instant-GI qt",
}

METHOD_NOTES = {
    "structsplat_current": (
        "best-searched StructSplat policy in this harness: aniso_flanking, scharr/rgb tensor, "
        "WSE, two-sided colors, feature12 cap, residual_tensor_add"
    ),
    "structsplat_shipped_defaults": (
        "public InitConfig/FitConfig defaults with only benchmark-control fields overridden "
        "(iterations, renderer, logging, target tracking)"
    ),
    "gaussianimage": "fixed random GaussianImage-style baseline",
    "gaussianimage_plus": "random half-budget init plus high-residual additions",
    "image_gs": "gradient-density init plus residual progressive additions",
    "instant_gi_quadtree": "Instant-GI quadtree/Delaunay fallback; no learned checkpoint",
}


def _growth_count(final_budget: int, fraction: float = 0.20) -> int:
    return max(16, int(round(final_budget * fraction / 16.0)) * 16)


def _base_fit(iters: int, renderer: str, lpips: bool) -> FitConfig:
    return FitConfig(
        iters=iters,
        render_chunk=512,
        log_every=max(1, iters // 20),
        renderer=renderer,
        pixel_loss="charbonnier",
        optimizer="adam",
        lr_schedule="none",
        target_psnrs=[22.0, 24.0, 26.0, 28.0, 30.0],
        compute_lpips=False,  # handled safely outside fit timing
        max_gaussians=None,
    )


def _shipped_defaults_fit(base_fit: FitConfig) -> FitConfig:
    return replace(
        FitConfig(),
        iters=base_fit.iters,
        render_chunk=base_fit.render_chunk,
        log_every=base_fit.log_every,
        renderer=base_fit.renderer,
        target_psnr=base_fit.target_psnr,
        target_psnrs=list(base_fit.target_psnrs),
        compute_lpips=False,
    )


def _build_method(
    method: str,
    img: np.ndarray,
    image_path: Path,
    final_budget: int,
    seed: int,
    device: str,
    base_fit: FitConfig,
) -> tuple[GaussianField, FitConfig, float, int]:
    scfg_default = StructureTensorConfig()
    init_start = time.time()

    if method == "structsplat_current":
        growth = _growth_count(final_budget, 0.20)
        start_budget = max(16, final_budget - growth)
        scfg = StructureTensorConfig(gradient_operator="scharr", color_space="rgb")
        field = build_field(
            img,
            InitConfig(
                strategy="aniso_flanking",
                num_gaussians=start_budget,
                density_mode="structure",
                sampling_mode="wse",
                color_mode="two_sided",
                scale_mode="spacing",
                scale_cap_mode="feature",
                scale_cap_max=12.0,
                opacity_mode="none",
                seed=seed,
            ),
            scfg,
            device=device,
        )
        init_seconds = time.time() - init_start
        fcfg = _residual_fit_cfg(
            base_fit, final_budget, start_budget, "residual_tensor_add", add_times=2
        )
        return field, fcfg, init_seconds, start_budget

    if method == "structsplat_shipped_defaults":
        field = build_field(
            img,
            InitConfig(num_gaussians=final_budget, seed=seed),
            scfg_default,
            device=device,
        )
        return field, _shipped_defaults_fit(base_fit), time.time() - init_start, final_budget

    if method != "structsplat_current":
        return build_comparison_analogue(
            method, img, image_path, final_budget, seed, device, base_fit, scfg_default
        )

    raise ValueError(f"unknown method {method!r}")


def _extra_metrics(render: torch.Tensor, target: torch.Tensor, want_lpips: bool) -> dict[str, float | None]:
    err = (render - target).detach()
    abs_err = err.abs()
    mse = torch.mean(err * err).clamp_min(1e-12)
    luma = torch.tensor([0.2126, 0.7152, 0.0722], device=target.device, dtype=target.dtype)
    render_luma = (render * luma).sum(dim=2)
    target_luma = (target * luma).sum(dim=2)
    luma_mse = torch.mean((render_luma - target_luma) ** 2).clamp_min(1e-12)

    gx = torch.zeros_like(target_luma)
    gy = torch.zeros_like(target_luma)
    if target_luma.shape[1] > 1:
        gx[:, 1:-1] = 0.5 * (target_luma[:, 2:] - target_luma[:, :-2])
        gx[:, 0] = target_luma[:, 1] - target_luma[:, 0]
        gx[:, -1] = target_luma[:, -1] - target_luma[:, -2]
    if target_luma.shape[0] > 1:
        gy[1:-1, :] = 0.5 * (target_luma[2:, :] - target_luma[:-2, :])
        gy[0, :] = target_luma[1, :] - target_luma[0, :]
        gy[-1, :] = target_luma[-1, :] - target_luma[-2, :]
    edge = torch.sqrt(gx * gx + gy * gy)
    edge_weight = 0.1 + edge / edge.mean().clamp_min(1e-6)
    luma_abs = (render_luma - target_luma).abs()

    lpips_val = None
    if want_lpips:
        try:
            lpips_val = M.LPIPS.distance(render, target)
        except Exception as exc:  # keep long benchmark alive if model download/cache fails
            print(f"  LPIPS skipped: {type(exc).__name__}: {exc}", flush=True)
            lpips_val = None

    return {
        "mse": float(mse),
        "mae": float(abs_err.mean()),
        "p95_abs_error": float(torch.quantile(abs_err.reshape(-1), 0.95)),
        "luma_psnr": float(10.0 * torch.log10(1.0 / luma_mse)),
        "edge_mae": float((luma_abs * edge_weight).mean()),
        "lpips": lpips_val,
    }


def _scale_stats(field: GaussianField) -> dict[str, float]:
    with torch.no_grad():
        scales = field.scales().detach().reshape(-1)
        return {
            "final_scale_max": float(scales.max()),
            "final_scale_p95": float(torch.quantile(scales, 0.95)),
            "final_scale_mean": float(scales.mean()),
        }


def _budget_for_size(max_side: int, base_side: int, base_budget: int) -> int:
    raw = base_budget * (max_side / base_side) ** 2
    return max(16, int(round(raw / 16.0)) * 16)


def _row_without_heavy(row: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in row.items() if k not in {"history"}}


def _fit_one(
    method: str,
    img: np.ndarray,
    image_path: Path,
    target: torch.Tensor,
    final_budget: int,
    seed: int,
    base_fit: FitConfig,
    device: str,
    want_lpips: bool,
) -> tuple[dict[str, Any], np.ndarray]:
    field, fcfg, init_seconds, start_gaussians = _build_method(
        method, img, image_path, final_budget, seed, device, base_fit
    )
    out = fit(field, target, fcfg, verbose=False)
    render = out["render"].detach().clamp(0, 1)
    extras = _extra_metrics(render, target, want_lpips)
    scales = _scale_stats(out["field"])
    # ONE metric convention per row (BENCH-002): every final-quality column is display-referred,
    # i.e. computed on the SAME clamped render as mae/edge_mae. Previously psnr/ssim/ms_ssim came
    # from fit()'s eval on the *unclamped* render, mixing conventions within a single row.
    # (auc_psnr stays a training-trajectory metric over the unclamped fit history, by design.)
    row = {
        "method": method,
        "method_label": METHOD_LABELS[method],
        "method_note": METHOD_NOTES[method],
        "start_gaussians": int(start_gaussians),
        "n_gaussians": int(out["n_gaussians"]),
        "init_seconds": float(init_seconds),
        "fit_seconds": float(out["fit_seconds"]),
        "total_seconds": float(init_seconds + out["fit_seconds"]),
        "iterations_run": int(out.get("iterations_run", base_fit.iters)),
        "stopped_early": bool(out.get("stopped_early", False)),
        "psnr": M.psnr(render, target),
        "ssim": float(M.ssim(render, target)),
        "ms_ssim": M.ms_ssim(render, target),
        "auc_psnr": _psnr_auc(out.get("history", {})),
        "iters_to_targets": out.get("iters_to_targets", {}),
        "history": out.get("history", {}),
        **extras,
        **scales,
        "fit_config": asdict(fcfg),
    }
    return row, render.cpu().numpy()


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


def _blank_tile(w: int, h: int) -> Image.Image:
    img = Image.new("RGB", (w, h), (242, 242, 242))
    draw = ImageDraw.Draw(img)
    draw.rectangle((0, 0, w - 1, h - 1), outline=(190, 190, 190))
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 12)
    except Exception:
        font = ImageFont.load_default()
    text = "not available"
    box = draw.textbbox((0, 0), text, font=font)
    draw.text(((w - (box[2] - box[0])) / 2, (h - (box[3] - box[1])) / 2),
              text, fill=(90, 90, 90), font=font)
    return img


def _grid_metric(row: dict[str, Any], key: str, digits: int) -> str:
    value = row.get(key)
    if value in (None, "", "None"):
        return "-"
    return f"{float(value):.{digits}f}"


def _make_grid(rows: list[dict[str, Any]], selected: list[dict[str, Any]], outdir: Path,
               max_side: int, iters: int, seed: int, *, include_seed_in_name: bool) -> None:
    subset_all = [
        r for r in rows
        if int(r["max_side"]) == max_side and int(r["iters"]) == iters and int(r["seed"]) == seed
    ]
    subset_ok = [r for r in subset_all if r["status"] == "ok"]
    if not subset_ok:
        return
    by_key = {(r["image"], r["method"]): r for r in subset_all}
    thumb_w, thumb_h, gap = 220, 132, 8
    cols = 1 + len(METHODS)
    cell_h = thumb_h + 48
    grid = Image.new(
        "RGB",
        (cols * thumb_w + (cols - 1) * gap, len(selected) * cell_h + (len(selected) - 1) * gap),
        "white",
    )
    for ridx, item in enumerate(selected):
        y = ridx * (cell_h + gap)
        original = Image.open(item[f"selected_{max_side}"]).convert("RGB")
        name = item["image"].replace("COCO_train2014_", "COCO ")
        diff = "difficult" if item["difficulty"] == "difficult" else "regular"
        grid.paste(_tile_with_label(original, name, f"target | {diff}", thumb_w, thumb_h), (0, y))
        for midx, method in enumerate(METHODS, 1):
            row = by_key.get((item["image"], method))
            if row is None or row.get("status") != "ok":
                rec = _blank_tile(thumb_w, thumb_h)
                sub = "error | see summary" if row is not None else "missing"
            else:
                rec = Image.open(row["reconstruction_path"]).convert("RGB")
                lpips = _grid_metric(row, "lpips", 3)
                sub = (
                    f"P {_grid_metric(row, 'psnr', 2)} | "
                    f"MS {_grid_metric(row, 'ms_ssim', 4)} | "
                    f"LP {lpips} | {int(row['n_gaussians'])}G"
                )
            grid.paste(
                _tile_with_label(rec, METHOD_LABELS[method], sub, thumb_w, thumb_h),
                (midx * (thumb_w + gap), y),
            )
    grid_dir = outdir / "grids"
    grid_dir.mkdir(exist_ok=True)
    suffix = f"_seed{seed}" if include_seed_in_name else ""
    grid.save(grid_dir / f"comparison_{max_side}px_{iters}it{suffix}.png")


def _mean_or_none(vals: list[float | None]) -> float | None:
    clean = [v for v in vals if v is not None and not math.isnan(float(v))]
    return mean(clean) if clean else None


def _std_or_none(vals: list[float | None]) -> float | None:
    clean = [v for v in vals if v is not None and not math.isnan(float(v))]
    if not clean:
        return None
    return pstdev(clean) if len(clean) > 1 else 0.0


def _fmt(v: float | None, digits: int = 4) -> str:
    if v is None:
        return "-"
    return f"{v:.{digits}f}"


def _groups(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    out: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        if row["status"] != "ok":
            continue
        key = tuple(row[k] for k in keys)
        out.setdefault(key, []).append(row)
    return out


def _write_summary(rows: list[dict[str, Any]], selected: list[dict[str, Any]], outdir: Path) -> None:
    ok = [r for r in rows if r["status"] == "ok"]
    lines = [
        "# Cross-Repo Matrix Comparison",
        "",
        "Matched executable comparison using StructSplat's current fitter and exact CUDA renderer for all rows.",
        "External repositories are represented by local repo-inspired placement/growth policies, not native codec pipelines.",
        f"Seeds: {', '.join(str(s) for s in sorted({int(r['seed']) for r in rows if 'seed' in r})) or '-'}",
        "Aggregate rows report mean and population std over image x seed runs.",
        "",
        "## Images",
        "",
        "| Image | Difficulty | Source |",
        "|---|---|---|",
    ]
    for item in selected:
        lines.append(f"| {item['image']} | {item['difficulty']} | `{item['source_path']}` |")

    lines += [
        "",
        "## Method Mapping",
        "",
    ]
    for method in METHODS:
        lines.append(f"- **{METHOD_LABELS[method]}**: {METHOD_NOTES[method]}")

    lines += [
        "",
        "## Overall Means",
        "",
        "| Method | Runs | PSNR | PSNR Std | SSIM | MS-SSIM | MS-SSIM Std | LPIPS | AUC | MAE | Edge MAE | Fit s | Total s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        vals = [r for r in ok if r["method"] == method]
        # this loop covers ALL methods, so a method with zero ok rows would raise StatisticsError
        # and void the whole summary; _mean_or_none keeps it to a "-" cell (BENCH-002)
        def _m(key, digits):
            return _fmt(_mean_or_none([r[key] for r in vals]), digits)
        def _s(key, digits):
            return _fmt(_std_or_none([r[key] for r in vals]), digits)
        lines.append(
            f"| {METHOD_LABELS[method]} | {len(vals)} | "
            f"{_m('psnr', 4)} | {_s('psnr', 4)} | {_m('ssim', 5)} | "
            f"{_m('ms_ssim', 5)} | {_s('ms_ssim', 5)} | "
            f"{_m('lpips', 4)} | {_m('auc_psnr', 3)} | "
            f"{_m('mae', 5)} | {_m('edge_mae', 5)} | "
            f"{_m('fit_seconds', 3)} | {_m('total_seconds', 3)} |"
        )

    lines += [
        "",
        "## Means By Resolution And Iterations",
        "",
        "| Max side | Iterations | Method | Budget cap | PSNR | PSNR Std | MS-SSIM | MS-SSIM Std | LPIPS | AUC | Fit s | Total s |",
        "|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for (max_side, iters, method), vals in sorted(_groups(ok, ("max_side", "iters", "method")).items()):
        lines.append(
            f"| {max_side} | {iters} | {METHOD_LABELS[method]} | {int(mean(r['budget_cap'] for r in vals))} | "
            f"{mean(r['psnr'] for r in vals):.4f} | {_fmt(_std_or_none([r['psnr'] for r in vals]), 4)} | "
            f"{mean(r['ms_ssim'] for r in vals):.5f} | "
            f"{_fmt(_std_or_none([r['ms_ssim'] for r in vals]), 5)} | "
            f"{_fmt(_mean_or_none([r['lpips'] for r in vals]), 4)} | "
            f"{_fmt(_mean_or_none([r['auc_psnr'] for r in vals]), 3)} | "
            f"{mean(r['fit_seconds'] for r in vals):.3f} | {mean(r['total_seconds'] for r in vals):.3f} |"
        )

    lines += [
        "",
        "## Difficult-Image Split",
        "",
        "| Difficulty | Method | Runs | PSNR | PSNR Std | MS-SSIM | MS-SSIM Std | LPIPS | Fit s |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for (difficulty, method), vals in sorted(_groups(ok, ("difficulty", "method")).items()):
        lines.append(
            f"| {difficulty} | {METHOD_LABELS[method]} | {len(vals)} | "
            f"{mean(r['psnr'] for r in vals):.4f} | {_fmt(_std_or_none([r['psnr'] for r in vals]), 4)} | "
            f"{mean(r['ms_ssim'] for r in vals):.5f} | "
            f"{_fmt(_std_or_none([r['ms_ssim'] for r in vals]), 5)} | "
            f"{_fmt(_mean_or_none([r['lpips'] for r in vals]), 4)} | "
            f"{mean(r['fit_seconds'] for r in vals):.3f} |"
        )

    lines += [
        "",
        "## Winners",
        "",
        "| Max side | Iterations | Best PSNR | Best MS-SSIM | Best LPIPS | Fastest fit |",
        "|---:|---:|---|---|---|---|",
    ]
    for (max_side, iters), vals in sorted(_groups(ok, ("max_side", "iters")).items()):
        by_method = []
        for method in METHODS:
            sub = [r for r in vals if r["method"] == method]
            if not sub:
                continue
            by_method.append({
                "method": method,
                "psnr": mean(r["psnr"] for r in sub),
                "ms_ssim": mean(r["ms_ssim"] for r in sub),
                "lpips": _mean_or_none([r["lpips"] for r in sub]),
                "fit_seconds": mean(r["fit_seconds"] for r in sub),
            })
        best_psnr = max(by_method, key=lambda r: r["psnr"])
        best_ms = max(by_method, key=lambda r: r["ms_ssim"])
        lpips_vals = [r for r in by_method if r["lpips"] is not None]
        best_lpips = min(lpips_vals, key=lambda r: r["lpips"]) if lpips_vals else None
        fastest = min(by_method, key=lambda r: r["fit_seconds"])
        lines.append(
            f"| {max_side} | {iters} | {METHOD_LABELS[best_psnr['method']]} ({best_psnr['psnr']:.3f}) | "
            f"{METHOD_LABELS[best_ms['method']]} ({best_ms['ms_ssim']:.5f}) | "
            f"{METHOD_LABELS[best_lpips['method']]} ({best_lpips['lpips']:.4f})"
            if best_lpips is not None else f"| {max_side} | {iters} | {METHOD_LABELS[best_psnr['method']]} ({best_psnr['psnr']:.3f}) | {METHOD_LABELS[best_ms['method']]} ({best_ms['ms_ssim']:.5f}) | -"
        )
        if best_lpips is not None:
            lines[-1] += f" | {METHOD_LABELS[fastest['method']]} ({fastest['fit_seconds']:.3f}s) |"
        else:
            lines[-1] += f" | {METHOD_LABELS[fastest['method']]} ({fastest['fit_seconds']:.3f}s) |"

    errors = [r for r in rows if r["status"] != "ok"]
    if errors:
        lines += ["", "## Errors", "", "| Row | Error |", "|---|---|"]
        for r in errors:
            lines.append(f"| {r['image']} {r['max_side']}px {r['iters']}it {r['method_label']} | `{r['error']}` |")

    lines += [
        "",
        "## Caveat",
        "",
        "This is a matched reference-policy benchmark. It does not exercise each repository's native CUDA renderer, entropy coder, or checkpointed learned components.",
        "Visual grids are under `grids/`; per-row reconstructions are under `reconstructions/`.",
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
    seed: int | None,
    renderer: str,
    lpips: bool,
    device: str | None,
    seeds: list[int] | tuple[int, ...] | None = None,
) -> list[dict[str, Any]]:
    seeds = resolve_seeds(seed, seeds)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    outdir.mkdir(parents=True, exist_ok=True)
    write_config(str(outdir), run_config({
        "dataset_dir": str(dataset_dir), "image_names": image_names, "max_sides": max_sides,
        "iters_list": iters_list, "base_budget": base_budget, "base_side": base_side,
        "seeds": seeds, "renderer": renderer, "lpips": lpips,
    }, device=device))
    selected_dir = outdir / "selected"
    recon_dir = outdir / "reconstructions"
    selected_dir.mkdir(exist_ok=True)
    recon_dir.mkdir(exist_ok=True)

    selected: list[dict[str, Any]] = []
    loaded: dict[tuple[str, int], tuple[np.ndarray, Path]] = {}
    for image_name in image_names:
        stem = Path(image_name).stem
        item = {
            "image": stem,
            "source_path": str(dataset_dir / image_name),
            "difficulty": "difficult" if stem in DIFFICULT_IMAGES else "regular",
        }
        for max_side in max_sides:
            img = _load_image(dataset_dir / image_name, max_side)
            resized_path = selected_dir / f"{stem}_{max_side}px.png"
            _save_image(img, resized_path)
            item[f"selected_{max_side}"] = str(resized_path)
            loaded[(stem, max_side)] = (img, resized_path)
        selected.append(item)

    rows: list[dict[str, Any]] = []
    total = len(image_names) * len(max_sides) * len(iters_list) * len(seeds) * len(METHODS)
    done = 0
    for max_side in max_sides:
        budget_cap = _budget_for_size(max_side, base_side, base_budget)
        for iters in iters_list:
            base_fit = _base_fit(iters, renderer, lpips)
            for item in selected:
                stem = item["image"]
                img, resized_path = loaded[(stem, max_side)]
                target = torch.as_tensor(img, dtype=torch.float32, device=device)
                for seed_value in seeds:
                    for method in METHODS:
                        done += 1
                        label = METHOD_LABELS[method]
                        print(
                            f"[{done}/{total}] {stem} {max_side}px {iters}it seed={seed_value} "
                            f"{label} cap={budget_cap}",
                            flush=True,
                        )
                        row_base = {
                            "image": stem,
                            "difficulty": item["difficulty"],
                            "source_path": item["source_path"],
                            "selected_path": str(resized_path),
                            "width": int(img.shape[1]),
                            "height": int(img.shape[0]),
                            "max_side": int(max_side),
                            "iters": int(iters),
                            "budget_cap": int(budget_cap),
                            "seed": int(seed_value),
                            "renderer": renderer,
                            "method": method,
                            "method_label": label,
                            "method_note": METHOD_NOTES[method],
                        }
                        try:
                            row, render_np = _fit_one(
                                method, img, resized_path, target, budget_cap, seed_value,
                                base_fit, device, lpips,
                            )
                            recon_path = (
                                recon_dir
                                / f"{max_side}px_{iters}it"
                                / f"{stem}_seed{seed_value}_{method}.png"
                            )
                            _save_image(render_np, recon_path)
                            row.update(row_base)
                            row["reconstruction_path"] = str(recon_path)
                            row["status"] = "ok"
                            row["error"] = ""
                            print(
                                f"  psnr={row['psnr']:.3f} ms={row['ms_ssim']:.5f} "
                                f"lpips={_fmt(row['lpips'], 4)} n={row['n_gaussians']} "
                                f"fit={row['fit_seconds']:.3f}s total={row['total_seconds']:.3f}s",
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
            for seed_value in seeds:
                _make_grid(
                    rows, selected, outdir, max_side, iters, seed_value,
                    include_seed_in_name=len(seeds) > 1,
                )

    json_rows = [_row_without_heavy(r) for r in rows]
    write_json(outdir / "metrics.json", json_rows)
    if json_rows:
        fieldnames = sorted({k for r in json_rows for k in r.keys() if k != "fit_config"})
        write_csv(
            outdir / "metrics.csv",
            json_rows,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
    write_json(outdir / "selected.json", selected)
    _write_summary(rows, selected, outdir)
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description="Cross-repo matched matrix comparison")
    p.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR,
                   help="directory containing the four benchmark images")
    p.add_argument("--outdir", type=Path, default=Path("results/cross_repo_matrix_current"))
    p.add_argument("--images", nargs="+", default=DEFAULT_IMAGES)
    p.add_argument("--max-sides", nargs="+", type=int, default=[160, 240, 320])
    p.add_argument("--iters", nargs="+", type=int, default=[80, 200])
    p.add_argument("--base-budget", type=int, default=640)
    p.add_argument("--base-side", type=int, default=160)
    from benchmarks.common import add_seed_args

    add_seed_args(p)
    p.add_argument("--renderer", default="cuda")
    p.add_argument("--lpips", action="store_true")
    p.add_argument("--device", default=None)
    args = p.parse_args()
    seeds = resolve_seeds(args.seed, args.seeds)
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
        args.lpips,
        args.device,
        seeds,
    )


if __name__ == "__main__":
    main()
