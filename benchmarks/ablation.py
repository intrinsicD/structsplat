"""ABL-001: the init-strategy x budget sweep, and the co-scientist fitness hook.

Holds everything constant (fitter, iters, images) and varies one thing — the init strategy — across
budgets and seeds. Emits tidy per-cell records (JSON + CSV) and a markdown summary. The scorer reads
only the metrics dict, never a method's internals, so it stays task-agnostic (mirrors the
IntrinsicEngine core/harness split). Torch is imported lazily.
"""
from __future__ import annotations
import os
import glob
import json
import time
from statistics import mean, pstdev

from benchmarks.common import load_image as _load_image
from benchmarks.common import run_config, write_config, write_csv, write_json
from structsplat.config import InitConfig, FitConfig, StructureTensorConfig
from structsplat.init import STRATEGIES

CONTROL_ARMS = {
    # Same thesis configuration as aniso_flanking, but replace WSE with the O(HW)
    # Floyd-Steinberg placement primitive used as ABL-004's killer control.
    "floyd_steinberg": {
        "init_strategy": "aniso_flanking",
        "sampling_mode": "floyd_steinberg",
        "fit_control": "none",
    },
    # Image-GS-style gradient/density-weighted random placement control.
    "density_random": {
        "init_strategy": "iso_blue_noise",
        "sampling_mode": "density_random",
        "fit_control": "none",
    },
    # 3DGS/MCMC-style control: random init plus count-preserving relocation during fitting.
    "random_relocate": {
        "init_strategy": "random",
        "sampling_mode": "wse",
        "fit_control": "relocate",
    },
}

DEFAULT_STRATEGIES = list(STRATEGIES) + list(CONTROL_ARMS)
_CELL_KEY_FIELDS = (
    "source_path", "max_side", "strategy", "budget", "seed", "flank_offset_frac",
    "flat_frac", "corner_frac", "max_axis_ratio", "coherence_power", "init_strategy",
    "sampling_mode", "fit_control", "relocate_every", "relocate_count", "iters",
    "target_psnr", "target_psnrs", "render_chunk", "pixel_loss", "ssim_weight",
    "renderer", "compute_lpips",
)


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


def _resolve_arm(label: str, budget: int, iters: int,
                 relocate_every: int | None, relocate_count: int | None) -> dict:
    if label in CONTROL_ARMS:
        arm = dict(CONTROL_ARMS[label])
    elif label in STRATEGIES:
        arm = {
            "init_strategy": label,
            "sampling_mode": "wse",
            "fit_control": "none",
        }
    else:
        valid = ", ".join([*STRATEGIES, *CONTROL_ARMS])
        raise ValueError(f"unknown ablation arm {label!r}; expected one of: {valid}")

    arm["fit_kwargs"] = {}
    if arm["fit_control"] == "relocate":
        every = relocate_every if relocate_every is not None else max(1, iters // 2)
        count = relocate_count if relocate_count is not None else 64
        arm["fit_kwargs"] = {
            "relocate_every": int(every),
            "relocate_count": min(int(count), int(budget)),
        }
    return arm


def _cell_key(row: dict) -> tuple:
    return tuple(
        tuple(v) if isinstance(v, list) else v
        for v in (row.get(k) for k in _CELL_KEY_FIELDS)
    )


def _load_jsonl(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def run_ablation(images, budgets=(2000, 5000, 10000, 20000), strategies=None, seeds=(0, 1, 2),
                 iters=1500, target_psnr=35.0, target_psnrs=None,
                 flank_offsets=(0.5,), flat_fracs=(0.02,), corner_fracs=(0.15,),
                 max_axis_ratios=(6.0,), coherence_powers=(1.0,),
                 render_chunk=512, renderer="normalized",
                 pixel_loss="l1", ssim_weight=0.3, compute_lpips=False,
                 relocate_every=None, relocate_count=64,
                 max_side: int | None = None, resume: bool = False,
                 max_new_cells: int | None = None,
                 outdir="results", device=None, write_plots=True):
    import torch
    from structsplat import init as _init
    from structsplat.fit import fit

    strategies = list(strategies or DEFAULT_STRATEGIES)
    unknown = sorted(set(strategies) - set(DEFAULT_STRATEGIES))
    if unknown:
        valid = ", ".join(DEFAULT_STRATEGIES)
        raise ValueError(f"unknown ablation arm(s) {unknown}; expected one of: {valid}")
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
    jsonl_path = os.path.join(outdir, "ablation.jsonl")
    rows = _load_jsonl(jsonl_path) if resume else []
    if not resume and os.path.exists(jsonl_path):
        os.remove(jsonl_path)
    done = {_cell_key(r) for r in rows}
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
        "render_chunk": render_chunk, "renderer": renderer, "pixel_loss": pixel_loss,
        "ssim_weight": ssim_weight, "compute_lpips": compute_lpips, "control_arms": CONTROL_ARMS,
        "relocate_every": relocate_every, "relocate_count": relocate_count,
        "max_side": max_side, "resume": resume, "max_new_cells": max_new_cells,
    }, device=device))

    new_cells = 0
    for path in files:
        img = _load_image(path, max_side=max_side)
        target = torch.as_tensor(img, device=device)
        name = os.path.splitext(os.path.basename(path))[0]
        H, W = img.shape[:2]
        for budget in budgets:
            for strat in strategies:
                arm = _resolve_arm(strat, budget, iters, relocate_every, relocate_count)
                init_strategy = arm["init_strategy"]
                vals = _strategy_sweep_values(init_strategy, flank_offsets, flat_fracs, corner_fracs,
                                               max_axis_ratios, coherence_powers)
                for flank in vals[0]:
                    for flat in vals[1]:
                        for corner in vals[2]:
                            for axis_ratio in vals[3]:
                                for coh_power in vals[4]:
                                    for seed in seeds:
                                        key_row = {
                                            "source_path": path,
                                            "max_side": max_side,
                                            "strategy": strat,
                                            "init_strategy": init_strategy,
                                            "sampling_mode": arm["sampling_mode"],
                                            "fit_control": arm["fit_control"],
                                            "relocate_every": arm["fit_kwargs"].get("relocate_every"),
                                            "relocate_count": arm["fit_kwargs"].get("relocate_count"),
                                            "budget": budget,
                                            "seed": seed,
                                            "flank_offset_frac": float(flank),
                                            "flat_frac": float(flat),
                                            "corner_frac": float(corner),
                                            "max_axis_ratio": float(axis_ratio),
                                            "coherence_power": float(coh_power),
                                            "iters": iters,
                                            "target_psnr": target_psnr,
                                            "target_psnrs": target_psnrs,
                                            "render_chunk": render_chunk,
                                            "renderer": renderer,
                                            "pixel_loss": pixel_loss,
                                            "ssim_weight": ssim_weight,
                                            "compute_lpips": compute_lpips,
                                        }
                                        if _cell_key(key_row) in done:
                                            print(f"skip existing {key_row}", flush=True)
                                            continue
                                        icfg = InitConfig(
                                            strategy=init_strategy,
                                            num_gaussians=budget,
                                            seed=seed,
                                            sampling_mode=arm["sampling_mode"],
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
                                            renderer=renderer,
                                            pixel_loss=pixel_loss,
                                            ssim_weight=ssim_weight,
                                            compute_lpips=compute_lpips,
                                            **arm["fit_kwargs"],
                                        )
                                        t0 = time.time()
                                        field = _init.build_field(img, icfg, scfg, device=device)
                                        init_seconds = time.time() - t0
                                        out = fit(field, target, fcfg, verbose=False)
                                        rec = {
                                            "image": name,
                                            "source_path": path,
                                            "height": H,
                                            "width": W,
                                            "max_side": max_side,
                                            "strategy": strat,
                                            "init_strategy": init_strategy,
                                            "sampling_mode": arm["sampling_mode"],
                                            "fit_control": arm["fit_control"],
                                            "relocate_every": arm["fit_kwargs"].get("relocate_every"),
                                            "relocate_count": arm["fit_kwargs"].get("relocate_count"),
                                            "iters": iters,
                                            "target_psnr": target_psnr,
                                            "target_psnrs": target_psnrs,
                                            "render_chunk": render_chunk,
                                            "renderer": renderer,
                                            "pixel_loss": pixel_loss,
                                            "ssim_weight": ssim_weight,
                                            "compute_lpips": compute_lpips,
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
                                        done.add(_cell_key(rec))
                                        with open(jsonl_path, "a", encoding="utf-8") as jf:
                                            jf.write(json.dumps(rec, default=str) + "\n")
                                        print({k: rec[k] for k in rec if k != "history"})
                                        new_cells += 1
                                        if max_new_cells is not None and new_cells >= max_new_cells:
                                            _write(rows, outdir, write_plots=write_plots)
                                            return rows

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
         r["max_axis_ratio"], r["coherence_power"], r.get("renderer", "normalized"))
        for r in rows
    })
    lines = ["# Ablation summary (mean PSNR ± std, dB)\n",
             "| config \\ budget | " + " | ".join(str(b) for b in budgets) + " |",
             "|" + "---|" * (len(budgets) + 1)]
    for key in keys:
        s, flank, flat, corner, axis_ratio, coh_power, renderer = key
        label = (f"{s} flank={flank:g} flat={flat:g} corner={corner:g} "
                 f"ratio={axis_ratio:g} cpow={coh_power:g} renderer={renderer}")
        cells = []
        for b in budgets:
            vals = [
                r["psnr"] for r in rows
                if (r["strategy"], r["flank_offset_frac"], r["flat_frac"], r["corner_frac"],
                    r["max_axis_ratio"], r["coherence_power"],
                    r.get("renderer", "normalized")) == key and r["budget"] == b
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
        s, flank, flat, corner, axis_ratio, coh_power, renderer = key
        label = (f"{s} flank={flank:g} flat={flat:g} corner={corner:g} "
                 f"ratio={axis_ratio:g} cpow={coh_power:g} renderer={renderer}")
        for b in budgets:
            subset = [
                r for r in rows
                if (r["strategy"], r["flank_offset_frac"], r["flat_frac"], r["corner_frac"],
                    r["max_axis_ratio"], r["coherence_power"],
                    r.get("renderer", "normalized")) == key and r["budget"] == b
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
            r["max_axis_ratio"], r["coherence_power"], r.get("sampling_mode", "wse"),
            r.get("fit_control", "none"), r.get("renderer", "normalized"))


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
    p.add_argument("--renderer", default="normalized",
                   choices=["normalized", "additive", "cuda", "cuda_normalized",
                            "cuda_additive", "gsplat", "cuda_gsplat"])
    p.add_argument("--pixel-loss", choices=["l1", "l2"], default="l1")
    p.add_argument("--ssim-weight", type=float, default=0.3)
    p.add_argument("--relocate-every", type=int, default=None)
    p.add_argument("--relocate-count", type=int, default=64)
    p.add_argument("--max-side", type=int, default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--max-new-cells", type=int, default=None,
                   help="run at most this many new cells, then rewrite outputs and exit")
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
                 renderer=a.renderer, pixel_loss=a.pixel_loss, ssim_weight=a.ssim_weight,
                 compute_lpips=a.lpips,
                 relocate_every=a.relocate_every, relocate_count=a.relocate_count,
                 max_side=a.max_side, resume=a.resume, max_new_cells=a.max_new_cells,
                 outdir=a.outdir, device=a.device, write_plots=not a.no_plots)
