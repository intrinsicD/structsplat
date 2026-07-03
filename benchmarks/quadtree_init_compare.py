"""Focused comparison for quadtree aggregate Gaussian initialization."""
from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import replace
from pathlib import Path
from statistics import mean, pstdev

import numpy as np
import torch

from benchmarks.optimization_followup import (
    BASELINE,
    TOP1,
    Candidate,
    _load_image,
    _make_configs,
    _psnr_auc,
    _select_images,
)
from structsplat import metrics as M
from structsplat.fit import fit
from structsplat.init import build_field
from structsplat.render import render_field


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _candidates() -> list[Candidate]:
    return [
        BASELINE,
        TOP1,
        replace(
            TOP1,
            label="quadtree_agg_variance",
            strategy="quadtree_aggregate",
            color="aggregate",
        ),
        replace(
            TOP1,
            label="quadtree_bilinear_variance",
            strategy="quadtree_aggregate",
            color="bilinear",
        ),
        replace(
            TOP1,
            label="quadtree_agg_structure",
            strategy="quadtree_aggregate",
            density="structure",
            color="aggregate",
        ),
        replace(
            TOP1,
            label="quadtree_hybrid_agg_variance",
            strategy="quadtree_hybrid",
            color="aggregate",
        ),
        replace(
            TOP1,
            label="quadtree_hybrid_bilinear_variance",
            strategy="quadtree_hybrid",
            color="bilinear",
        ),
        replace(
            TOP1,
            label="quadtree_wse_variance",
            strategy="quadtree_wse",
            color="bilinear",
        ),
        replace(
            TOP1,
            label="stage_top1_hard_cap8",
            scale_cap_mode="hard",
            scale_cap_max=8.0,
        ),
        replace(
            TOP1,
            label="stage_top1_feature_cap12",
            scale_cap_mode="feature",
            scale_cap_max=12.0,
        ),
        replace(
            TOP1,
            label="quadtree_wse_hard_cap8",
            strategy="quadtree_wse",
            color="bilinear",
            scale_cap_mode="hard",
            scale_cap_max=8.0,
        ),
        replace(
            TOP1,
            label="quadtree_wse_feature_cap12",
            strategy="quadtree_wse",
            color="bilinear",
            scale_cap_mode="feature",
            scale_cap_max=12.0,
        ),
        replace(
            TOP1,
            label="quadtree_hybrid_agg_feature_cap12",
            strategy="quadtree_hybrid",
            color="aggregate",
            scale_cap_mode="feature",
            scale_cap_max=12.0,
        ),
    ]


def _scale_stats(field) -> dict:
    scales = field.scales().detach().cpu().numpy()
    major = scales.max(axis=1)
    return {
        "max_scale": round(float(major.max()), 4),
        "p95_scale": round(float(np.percentile(major, 95)), 4),
    }


def run_candidate(image_path: Path, c: Candidate, args, seed: int | None = None) -> dict:
    seed = int(args.seed if seed is None else seed)
    img = _load_image(image_path, args.max_side)
    target = torch.as_tensor(img, dtype=torch.float32, device=args.device)
    icfg, fcfg, scfg = _make_configs(
        c, args.budget, seed, args.iters, args.render_chunk,
        args.split_count, args.max_gaussians,
    )
    start = time.time()
    field = build_field(img, icfg, scfg, device=args.device)
    _sync()
    init_seconds = time.time() - start
    init_scales = _scale_stats(field)
    with torch.no_grad():
        init_img = render_field(
            field.means,
            field.conics(fcfg.aa_dilation),
            field.colors,
            field.radii(fcfg.sigma_cutoff, fcfg.aa_dilation),
            target.shape[0],
            target.shape[1],
            fcfg.render_chunk,
            fcfg.renderer,
            field.opacity_values(),
        )
        init_psnr = float(M.psnr(init_img, target))
        init_ms_ssim = float(M.ms_ssim(init_img, target))
    out = fit(field, target, fcfg, verbose=False)
    _sync()
    total_seconds = time.time() - start
    final_scales = _scale_stats(out["field"])
    hist = out.get("history", {})
    iterations_run = int(out.get("iterations_run") or ((max(hist.get("iter", [args.iters - 1])) + 1)
                                                       if hist else args.iters))
    return {
        "image": image_path.stem,
        "source_path": str(image_path),
        "candidate": c.label,
        "budget": args.budget,
        "seed": seed,
        "strategy": c.strategy,
        "density": c.density,
        "sampling": c.sampling,
        "color": c.color,
        "scale": c.scale,
        "scale_cap_mode": c.scale_cap_mode,
        "scale_cap_max": c.scale_cap_max,
        "opacity": c.opacity,
        "loss": c.loss,
        "optimizer": c.optimizer,
        "renderer": c.renderer,
        "width": img.shape[1],
        "height": img.shape[0],
        "init_psnr": round(init_psnr, 4),
        "init_ms_ssim": round(init_ms_ssim, 5),
        "psnr": round(float(out["psnr"]), 4),
        "ssim": round(float(out["ssim"]), 5),
        "ms_ssim": round(float(out["ms_ssim"]), 5),
        "auc_psnr": _psnr_auc(hist),
        "n_gaussians": int(out["n_gaussians"]),
        "init_max_scale": init_scales["max_scale"],
        "init_p95_scale": init_scales["p95_scale"],
        "final_max_scale": final_scales["max_scale"],
        "final_p95_scale": final_scales["p95_scale"],
        "init_seconds": round(float(init_seconds), 4),
        "fit_seconds": round(float(out.get("fit_seconds", 0.0)), 4),
        "total_seconds": round(float(total_seconds), 4),
        "iterations_run": iterations_run,
    }


def _write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.with_suffix(".json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    if not rows:
        return
    with path.with_suffix(".csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _aggregate(rows: list[dict]) -> list[dict]:
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(str(row["candidate"]), []).append(row)
    out = []
    for label, vals in groups.items():
        ps = [float(v["psnr"]) for v in vals]
        init_ps = [float(v["init_psnr"]) for v in vals]
        out.append({
            "candidate": label,
            "runs": len(vals),
            "mean_init_psnr": mean(init_ps),
            "mean_psnr": mean(ps),
            "std_psnr": pstdev(ps) if len(ps) > 1 else 0.0,
            "mean_ms_ssim": mean(float(v["ms_ssim"]) for v in vals),
            "mean_auc_psnr": mean(float(v["auc_psnr"]) for v in vals if v["auc_psnr"] is not None),
            "mean_final_max_scale": mean(float(v["final_max_scale"]) for v in vals),
            "mean_final_p95_scale": mean(float(v["final_p95_scale"]) for v in vals),
            "mean_init_seconds": mean(float(v["init_seconds"]) for v in vals),
            "mean_fit_seconds": mean(float(v["fit_seconds"]) for v in vals),
            "mean_total_seconds": mean(float(v["total_seconds"]) for v in vals),
        })
    return sorted(out, key=lambda r: r["mean_psnr"], reverse=True)


def write_summary(outdir: Path, rows: list[dict], images: list[Path], args) -> None:
    agg = _aggregate(rows)
    lines = [
        "# Quadtree Init Compare",
        "",
        f"- Images: {len(images)} held-out COCO train2014 files",
        f"- Budget: {args.budget}",
        f"- Iterations: {args.iters}",
        f"- Seeds: {', '.join(str(s) for s in args.seeds)}",
        f"- Max side: {args.max_side}",
        f"- Device: {args.device}",
        "- Aggregate rows report mean and population std over image x seed runs.",
        "",
        "## Images",
        "",
        *[f"- `{p.name}`" for p in images],
        "",
        "## Aggregate",
        "",
        "| candidate | Runs | Init PSNR | Final PSNR | PSNR Std | MS-SSIM | AUC | Final max scale | Final p95 scale | Init s | Fit s | Total s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in agg:
        lines.append(
            f"| `{r['candidate']}` | {r['runs']} | {r['mean_init_psnr']:.4f} | "
            f"{r['mean_psnr']:.4f} | {r['std_psnr']:.4f} | {r['mean_ms_ssim']:.5f} | "
            f"{r['mean_auc_psnr']:.4f} | {r['mean_final_max_scale']:.3f} | "
            f"{r['mean_final_p95_scale']:.3f} | {r['mean_init_seconds']:.4f} | "
            f"{r['mean_fit_seconds']:.3f} | {r['mean_total_seconds']:.3f} |"
        )
    lines += [
        "",
        "## Per Image",
        "",
        "| image | seed | candidate | Init PSNR | Final PSNR | MS-SSIM | Total s |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['image']}` | {row['seed']} | `{row['candidate']}` | "
            f"{row['init_psnr']:.4f} | {row['psnr']:.4f} | {row['ms_ssim']:.5f} | "
            f"{row['total_seconds']:.3f} |"
        )
    (outdir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description="Compare quadtree aggregate initialization")
    p.add_argument("--dataset-dir", type=Path, default=None,
                   help="COCO train2014 directory (required; no personal-path default, BENCH-002)")
    p.add_argument("--outdir", type=Path, default=Path("results/quadtree_init_compare"))
    p.add_argument("--image-count", type=int, default=8)
    p.add_argument("--budget", type=int, default=512)
    p.add_argument("--iters", type=int, default=80)
    p.add_argument("--max-side", type=int, default=160)
    from benchmarks._util import add_seed_args, resolve_seeds

    add_seed_args(p)
    p.add_argument("--render-chunk", type=int, default=512)
    p.add_argument("--split-count", type=int, default=128)
    p.add_argument("--max-gaussians", type=int, default=None)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    if args.dataset_dir is None:
        raise SystemExit("--dataset-dir is required (path to the COCO train2014 images)")
    args.seeds = resolve_seeds(args.seed, args.seeds)
    args.seed = args.seeds[0]
    args.device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    args.outdir.mkdir(parents=True, exist_ok=True)
    from benchmarks._util import run_config, write_config
    write_config(str(args.outdir), run_config(vars(args), device=args.device))
    images = _select_images(args.dataset_dir, args.image_count)
    rows = []
    for image in images:
        for seed in args.seeds:
            for c in _candidates():
                print(f"[quadtree] {image.stem} seed={seed} {c.label}", flush=True)
                row = run_candidate(image, c, args, seed)
                rows.append(row)
                print(
                    f"  init={row['init_psnr']:.3f} final={row['psnr']:.3f} "
                    f"total={row['total_seconds']:.2f}s",
                    flush=True,
                )
    _write_rows(args.outdir / "quadtree_init_compare", rows)
    write_summary(args.outdir, rows, images, args)
    print(f"\nwrote quadtree comparison to {args.outdir}")


if __name__ == "__main__":
    main()
