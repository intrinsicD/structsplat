"""ABL-001: the init-strategy x budget sweep, and the co-scientist fitness hook.

Holds everything constant (fitter, iters, images) and varies one thing — the init strategy — across
budgets and seeds. Emits tidy per-cell records (JSON + CSV) and a markdown summary. The scorer reads
only the metrics dict, never a method's internals, so it stays task-agnostic (mirrors the
IntrinsicEngine core/harness split). Torch is imported lazily.
"""
from __future__ import annotations
import csv
import json
import os
import glob
from statistics import mean, pstdev

from structsplat.config import InitConfig, FitConfig
from structsplat.init import STRATEGIES

DEFAULT_STRATEGIES = list(STRATEGIES)


def _iter_images(images):
    files = []
    for item in images:
        if os.path.isdir(item):
            for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp"):
                files += glob.glob(os.path.join(item, ext))
        else:
            files.append(item)
    return sorted(files)


def run_ablation(images, budgets=(2000, 5000, 10000, 20000), strategies=None, seeds=(0, 1, 2),
                 iters=1500, target_psnr=35.0, outdir="results", device=None):
    import torch
    import numpy as np
    from structsplat.cli import load_image
    from structsplat import init as _init
    from structsplat.fit import fit

    strategies = strategies or DEFAULT_STRATEGIES
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(outdir, exist_ok=True)
    files = _iter_images(images)
    if not files:
        raise SystemExit("no images found")

    rows = []
    for path in files:
        img = load_image(path)
        target = torch.as_tensor(img, device=device)
        name = os.path.splitext(os.path.basename(path))[0]
        for budget in budgets:
            for strat in strategies:
                for seed in seeds:
                    icfg = InitConfig(strategy=strat, num_gaussians=budget, seed=seed)
                    fcfg = FitConfig(iters=iters, target_psnr=target_psnr)
                    field = _init.build_field(img, icfg, device=device)
                    out = fit(field, target, fcfg, verbose=False)
                    rec = {"image": name, "strategy": strat, "budget": budget, "seed": seed,
                           "psnr": round(out["psnr"], 4), "ssim": round(out["ssim"], 5),
                           "ms_ssim": round(out["ms_ssim"], 5),
                           "lpips": out["lpips"], "iters_to_target": out["iters_to_target"],
                           "n_gaussians": out["n_gaussians"]}
                    rows.append(rec)
                    print(rec)

    _write(rows, outdir)
    return rows


def _write(rows, outdir):
    with open(os.path.join(outdir, "ablation.json"), "w") as f:
        json.dump(rows, f, indent=2)
    if rows:
        with open(os.path.join(outdir, "ablation.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    with open(os.path.join(outdir, "summary.md"), "w") as f:
        f.write(summarize(rows))
    print(f"\nwrote ablation.json / ablation.csv / summary.md to {outdir}")


def summarize(rows) -> str:
    """Mean PSNR per (strategy, budget) across images+seeds -> markdown table."""
    budgets = sorted({r["budget"] for r in rows})
    strategies = sorted({r["strategy"] for r in rows})
    lines = ["# Ablation summary (mean PSNR, dB)\n",
             "| strategy \\ budget | " + " | ".join(str(b) for b in budgets) + " |",
             "|" + "---|" * (len(budgets) + 1)]
    for s in strategies:
        cells = []
        for b in budgets:
            vals = [r["psnr"] for r in rows if r["strategy"] == s and r["budget"] == b]
            cells.append(f"{mean(vals):.2f}" if vals else "-")
        lines.append(f"| {s} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def fitness(rows, strategy: str, budget: int) -> float:
    """Scalar a co-scientist maximizes: mean PSNR for a strategy at a target budget.

    Swap for area-under PSNR-vs-iters, or a rate-distortion point, as the discovery target evolves.
    """
    vals = [r["psnr"] for r in rows if r["strategy"] == strategy and r["budget"] == budget]
    return mean(vals) if vals else float("-inf")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="StructSplat init-strategy ablation (ABL-001)")
    p.add_argument("images", nargs="+")
    p.add_argument("--budgets", type=int, nargs="+", default=[2000, 5000, 10000, 20000])
    p.add_argument("--iters", type=int, default=1500)
    p.add_argument("--target-psnr", type=float, default=35.0, dest="target_psnr")
    p.add_argument("--outdir", default="results")
    p.add_argument("--device", default=None)
    a = p.parse_args()
    run_ablation(a.images, budgets=a.budgets, iters=a.iters, target_psnr=a.target_psnr,
                 outdir=a.outdir, device=a.device)
