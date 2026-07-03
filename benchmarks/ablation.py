"""ABL-001: the init-strategy x budget sweep, and the co-scientist fitness hook.

Holds everything constant (fitter, iters, images) and varies one thing — the init strategy — across
budgets and seeds. Emits tidy per-cell records (JSON + CSV) and a markdown summary. The scorer reads
only the metrics dict, never a method's internals, so it stays task-agnostic (mirrors the
IntrinsicEngine core/harness split). Torch is imported lazily.
"""
from __future__ import annotations
import os
import glob
import time
from statistics import mean, pstdev

from benchmarks.common import run_config, write_config, write_csv, write_json
from structsplat.config import InitConfig, FitConfig, StructureTensorConfig
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


def _one(x):
    return tuple(x) if isinstance(x, (list, tuple)) else (x,)


def _strategy_sweep_values(strategy, flank_offsets, flat_fracs, corner_fracs,
                           max_axis_ratios, coherence_powers):
    if strategy == "aniso_flanking":
        return (flank_offsets, flat_fracs, corner_fracs, max_axis_ratios, coherence_powers)
    if strategy == "aniso_onedge":
        return ((0.0,), (flat_fracs[0],), (corner_fracs[0],), max_axis_ratios, coherence_powers)
    return ((0.0,), (flat_fracs[0],), (corner_fracs[0],), (max_axis_ratios[0],), (coherence_powers[0],))


def run_ablation(images, budgets=(2000, 5000, 10000, 20000), strategies=None, seeds=(0, 1, 2),
                 iters=1500, target_psnr=35.0, target_psnrs=None,
                 flank_offsets=(0.5,), flat_fracs=(0.02,), corner_fracs=(0.15,),
                 max_axis_ratios=(6.0,), coherence_powers=(1.0,),
                 render_chunk=512, pixel_loss="l1", ssim_weight=0.3, compute_lpips=False,
                 outdir="results", device=None, write_plots=True):
    import torch
    from structsplat.cli import load_image
    from structsplat import init as _init
    from structsplat.fit import fit

    strategies = strategies or DEFAULT_STRATEGIES
    flank_offsets = _one(flank_offsets)
    flat_fracs = _one(flat_fracs)
    corner_fracs = _one(corner_fracs)
    max_axis_ratios = _one(max_axis_ratios)
    coherence_powers = _one(coherence_powers)
    target_psnrs = list(target_psnrs or [])
    if target_psnr is not None:
        target_psnrs.append(float(target_psnr))

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(outdir, exist_ok=True)
    files = _iter_images(images)
    if not files:
        raise SystemExit("no images found")

    # reproducible from its own artifacts (invariant 5 / BENCH-002)
    write_config(outdir, run_config({
        "images": files, "budgets": list(budgets), "strategies": list(strategies),
        "seeds": list(seeds), "iters": iters, "target_psnr": target_psnr,
        "target_psnrs": target_psnrs, "flank_offsets": list(flank_offsets),
        "flat_fracs": list(flat_fracs), "corner_fracs": list(corner_fracs),
        "max_axis_ratios": list(max_axis_ratios), "coherence_powers": list(coherence_powers),
        "render_chunk": render_chunk, "pixel_loss": pixel_loss, "ssim_weight": ssim_weight,
        "compute_lpips": compute_lpips,
    }, device=device))

    rows = []
    for path in files:
        img = load_image(path)
        target = torch.as_tensor(img, device=device)
        name = os.path.splitext(os.path.basename(path))[0]
        for budget in budgets:
            for strat in strategies:
                vals = _strategy_sweep_values(strat, flank_offsets, flat_fracs, corner_fracs,
                                               max_axis_ratios, coherence_powers)
                for flank in vals[0]:
                    for flat in vals[1]:
                        for corner in vals[2]:
                            for axis_ratio in vals[3]:
                                for coh_power in vals[4]:
                                    for seed in seeds:
                                        icfg = InitConfig(
                                            strategy=strat,
                                            num_gaussians=budget,
                                            seed=seed,
                                            flank_offset_frac=float(flank),
                                            max_axis_ratio=float(axis_ratio),
                                            coherence_power=float(coh_power),
                                        )
                                        scfg = StructureTensorConfig(
                                            flat_frac=float(flat),
                                            corner_frac=float(corner),
                                        )
                                        fcfg = FitConfig(
                                            iters=iters,
                                            target_psnr=target_psnr,
                                            target_psnrs=target_psnrs,
                                            render_chunk=render_chunk,
                                            pixel_loss=pixel_loss,
                                            ssim_weight=ssim_weight,
                                            compute_lpips=compute_lpips,
                                        )
                                        t0 = time.time()
                                        field = _init.build_field(img, icfg, scfg, device=device)
                                        init_seconds = time.time() - t0
                                        out = fit(field, target, fcfg, verbose=False)
                                        rec = {
                                            "image": name,
                                            "strategy": strat,
                                            "budget": budget,
                                            "seed": seed,
                                            "flank_offset_frac": float(flank),
                                            "flat_frac": float(flat),
                                            "corner_frac": float(corner),
                                            "max_axis_ratio": float(axis_ratio),
                                            "coherence_power": float(coh_power),
                                            "psnr": round(out["psnr"], 4),
                                            "ssim": round(out["ssim"], 5),
                                            "ms_ssim": round(out["ms_ssim"], 5),
                                            "lpips": out["lpips"],
                                            "iters_to_target": out["iters_to_target"],
                                            "iters_to_targets": out.get("iters_to_targets", {}),
                                            "n_gaussians": out["n_gaussians"],
                                            "init_seconds": init_seconds,
                                            "fit_seconds": out.get("fit_seconds"),
                                            "history": out.get("history", {}),
                                        }
                                        rows.append(rec)
                                        print({k: rec[k] for k in rec if k != "history"})

    _write(rows, outdir, write_plots=write_plots)
    return rows


def _write(rows, outdir, write_plots=True):
    write_json(os.path.join(outdir, "ablation.json"), rows)
    if rows:
        # nested dicts stay in the JSON; str()-ified dict cells make the CSV unusable
        fields = [k for k in rows[0] if k not in {"history", "iters_to_targets"}]
        write_csv(
            os.path.join(outdir, "ablation.csv"),
            [{k: row.get(k) for k in fields} for row in rows],
            fieldnames=fields,
        )
    with open(os.path.join(outdir, "summary.md"), "w") as f:
        f.write(summarize(rows))
    if write_plots:
        _write_plots(rows, outdir)
    print(f"\nwrote ablation.json / ablation.csv / summary.md to {outdir}")


def summarize(rows) -> str:
    """Mean PSNR per strategy/config/budget across images+seeds -> markdown table."""
    budgets = sorted({r["budget"] for r in rows})
    keys = sorted({
        (r["strategy"], r["flank_offset_frac"], r["flat_frac"], r["corner_frac"],
         r["max_axis_ratio"], r["coherence_power"])
        for r in rows
    })
    lines = ["# Ablation summary (mean PSNR ± std, dB)\n",
             "| config \\ budget | " + " | ".join(str(b) for b in budgets) + " |",
             "|" + "---|" * (len(budgets) + 1)]
    for key in keys:
        s, flank, flat, corner, axis_ratio, coh_power = key
        label = (f"{s} flank={flank:g} flat={flat:g} corner={corner:g} "
                 f"ratio={axis_ratio:g} cpow={coh_power:g}")
        cells = []
        for b in budgets:
            vals = [
                r["psnr"] for r in rows
                if (r["strategy"], r["flank_offset_frac"], r["flat_frac"], r["corner_frac"],
                    r["max_axis_ratio"], r["coherence_power"]) == key and r["budget"] == b
            ]
            if vals:
                cells.append(f"{mean(vals):.2f} ± {pstdev(vals):.2f}")
            else:
                cells.append("-")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")

    lines += ["", "## Time to target", "",
              "| config | budget | target | reached | mean iters |",
              "|---|---:|---:|---:|---:|"]
    for key in keys:
        s, flank, flat, corner, axis_ratio, coh_power = key
        label = (f"{s} flank={flank:g} flat={flat:g} corner={corner:g} "
                 f"ratio={axis_ratio:g} cpow={coh_power:g}")
        for b in budgets:
            subset = [
                r for r in rows
                if (r["strategy"], r["flank_offset_frac"], r["flat_frac"], r["corner_frac"],
                    r["max_axis_ratio"], r["coherence_power"]) == key and r["budget"] == b
            ]
            targets = sorted({float(t) for r in subset for t in r.get("iters_to_targets", {})})
            for t in targets:
                vals = [r.get("iters_to_targets", {}).get(str(t)) for r in subset]
                reached = [v for v in vals if v is not None]
                mean_iters = f"{mean(reached):.1f}" if reached else "-"
                lines.append(f"| {label} | {b} | {t:g} | {len(reached)}/{len(vals)} | {mean_iters} |")
    return "\n".join(lines) + "\n"


def _write_plots(rows, outdir):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    groups = sorted({r["strategy"] for r in rows})
    budgets = sorted({r["budget"] for r in rows})
    plt.figure(figsize=(7, 4))
    for strategy in groups:
        ys = []
        xs = []
        for b in budgets:
            # best-config-per-strategy, not the pooled mean over hyperparameter variants (BENCH-002)
            best = _best_config_mean(rows, strategy, b)
            if best != float("-inf"):
                xs.append(b)
                ys.append(best)
        if xs:
            plt.plot(xs, ys, marker="o", label=strategy)
    plt.xlabel("Gaussians")
    plt.ylabel("PSNR (dB)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "psnr_vs_budget.png"), dpi=160)
    plt.close()

    plt.figure(figsize=(7, 4))
    for strategy in groups:
        for b in budgets:  # pooling budgets would average curves of different capacity
            curves = [r.get("history", {}) for r in rows
                      if r["strategy"] == strategy and r["budget"] == b]
            by_iter = {}
            for hist in curves:
                for it, psnr in zip(hist.get("iter", []), hist.get("psnr", [])):
                    by_iter.setdefault(it, []).append(psnr)
            if by_iter:
                xs = sorted(by_iter)
                ys = [mean(by_iter[it]) for it in xs]
                plt.plot(xs, ys, label=f"{strategy} @ {b}")
    plt.xlabel("Iteration")
    plt.ylabel("PSNR (dB)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "psnr_vs_iters.png"), dpi=160)
    plt.close()


def _config_key(r):
    """Full config identity of a row (strategy + every swept hyperparameter)."""
    return (r["strategy"], r["flank_offset_frac"], r["flat_frac"], r["corner_frac"],
            r["max_axis_ratio"], r["coherence_power"])


def _best_config_mean(rows, strategy: str, budget: int) -> float:
    """Best (not pooled) mean PSNR for a strategy at a budget: max over its config variants.

    Pooling all flank/ratio/threshold variants of a strategy into one mean biases the headline
    comparison whenever those sweeps are active (a strategy with a wide, mostly-bad sweep looks
    worse than its actual best config). We aggregate per full config key across images+seeds,
    then take the best config for the strategy (BENCH-002).
    """
    per_config = {}
    for r in rows:
        if r["strategy"] == strategy and r["budget"] == budget:
            per_config.setdefault(_config_key(r), []).append(r["psnr"])
    if not per_config:
        return float("-inf")
    return max(mean(v) for v in per_config.values())


def fitness(rows, strategy: str, budget: int) -> float:
    """Scalar a co-scientist maximizes: best-config mean PSNR for a strategy at a target budget.

    Uses the best config per strategy (not the pooled mean over hyperparameter variants) so the
    signal is not diluted by an active flank/ratio/threshold sweep. Swap for area-under
    PSNR-vs-iters, or a rate-distortion point, as the discovery target evolves.
    """
    return _best_config_mean(rows, strategy, budget)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="StructSplat init-strategy ablation (ABL-001)")
    p.add_argument("images", nargs="+")
    p.add_argument("--budgets", type=int, nargs="+", default=[2000, 5000, 10000, 20000])
    p.add_argument("--iters", type=int, default=1500)
    p.add_argument("--target-psnr", type=float, default=35.0, dest="target_psnr")
    p.add_argument("--target-psnrs", type=float, nargs="*", default=[])
    p.add_argument("--strategies", nargs="*", default=None)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--flank-offsets", type=float, nargs="+", default=[0.5])
    p.add_argument("--flat-fracs", type=float, nargs="+", default=[0.02])
    p.add_argument("--corner-fracs", type=float, nargs="+", default=[0.15])
    p.add_argument("--max-axis-ratios", type=float, nargs="+", default=[6.0])
    p.add_argument("--coherence-powers", type=float, nargs="+", default=[1.0])
    p.add_argument("--render-chunk", type=int, default=512)
    p.add_argument("--pixel-loss", choices=["l1", "l2"], default="l1")
    p.add_argument("--ssim-weight", type=float, default=0.3)
    p.add_argument("--lpips", action="store_true", help="compute LPIPS for each run")
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--outdir", default="results")
    p.add_argument("--device", default=None)
    a = p.parse_args()
    run_ablation(a.images, budgets=a.budgets, strategies=a.strategies, seeds=a.seeds,
                 iters=a.iters, target_psnr=a.target_psnr, target_psnrs=a.target_psnrs,
                 flank_offsets=a.flank_offsets, flat_fracs=a.flat_fracs,
                 corner_fracs=a.corner_fracs, max_axis_ratios=a.max_axis_ratios,
                 coherence_powers=a.coherence_powers, render_chunk=a.render_chunk,
                 pixel_loss=a.pixel_loss, ssim_weight=a.ssim_weight, compute_lpips=a.lpips,
                 outdir=a.outdir, device=a.device, write_plots=not a.no_plots)
