"""Stage-wise StructSplat search across init, density, fitting, refinement, and pyramid choices."""
from __future__ import annotations

import argparse
import csv
import glob
import itertools
import json
import os
import time
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import numpy as np

from structsplat.config import FitConfig, InitConfig, PyramidConfig, StructureTensorConfig


def _iter_images(images):
    files = []
    for item in images:
        if os.path.isdir(item):
            for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp"):
                files += glob.glob(os.path.join(item, ext))
        else:
            files.append(item)
    return sorted(files)


def _load_image(path: str, max_side: int | None = None) -> np.ndarray:
    from PIL import Image

    img = Image.open(path).convert("RGB")
    if max_side is not None and max(img.size) > max_side:
        scale = max_side / max(img.size)
        img = img.resize((round(img.size[0] * scale), round(img.size[1] * scale)),
                         Image.Resampling.LANCZOS)
    return np.asarray(img, dtype=np.float32) / 255.0


def _one(x):
    return tuple(x) if isinstance(x, (list, tuple)) else (x,)


def _config_label(c: dict[str, Any]) -> str:
    keys = [
        "strategy", "tensor", "density", "sampling", "color", "scale", "renderer",
        "opacity", "loss", "optimizer", "lr_schedule", "refine", "pyramid",
    ]
    return "|".join(f"{k}={c[k]}" for k in keys)


def _iter_configs(
    strategies,
    tensor_operators,
    density_modes,
    sampling_modes,
    color_modes,
    scale_modes,
    opacity_modes,
    renderers,
    pixel_losses,
    optimizers,
    lr_schedules,
    refine_modes,
    pyramid_modes,
):
    fields = [
        ("strategy", strategies),
        ("tensor", tensor_operators),
        ("density", density_modes),
        ("sampling", sampling_modes),
        ("color", color_modes),
        ("scale", scale_modes),
        ("opacity", opacity_modes),
        ("renderer", renderers),
        ("loss", pixel_losses),
        ("optimizer", optimizers),
        ("lr_schedule", lr_schedules),
        ("refine", refine_modes),
        ("pyramid", pyramid_modes),
    ]
    names = [f[0] for f in fields]
    for values in itertools.product(*[f[1] for f in fields]):
        cfg = dict(zip(names, values, strict=True))
        cfg["label"] = _config_label(cfg)
        yield cfg


def _refine_kwargs(mode: str, split_every: int | None, split_count: int,
                   prune_every: int | None, prune_min_activity: float) -> dict[str, Any]:
    if mode == "none":
        return {}
    if mode == "prune":
        return {
            "prune_every": prune_every,
            "prune_min_activity": prune_min_activity,
        }
    if mode in ("duplicate", "residual_add"):
        return {
            "split_every": split_every,
            "split_count": split_count,
            "split_mode": mode,
        }
    if mode == "prune_residual_add":
        return {
            "prune_every": prune_every,
            "prune_min_activity": prune_min_activity,
            "split_every": split_every,
            "split_count": split_count,
            "split_mode": "residual_add",
        }
    raise ValueError(
        f"unknown refine mode {mode!r}; expected none, prune, duplicate, residual_add, prune_residual_add"
    )


def _run_one(img, target, cfg, *, budget, seed, iters, render_chunk, ssim_weight,
             flank_offset, max_axis_ratio, coherence_power, init_scale_mult,
             density_base, density_power, flat_frac, corner_frac, grad_sigma, tensor_sigma,
             color_radius, init_opacity, lr_decay_every, lr_decay_gamma, split_every, split_count,
             prune_every, prune_min_activity, max_gaussians, pyramid_levels,
             pyramid_fractions, pyramid_iters_per_level, compute_lpips, verbose):
    from structsplat import init as _init
    from structsplat.fit import fit
    from structsplat.pyramid import fit_pyramid

    scfg = StructureTensorConfig(
        grad_sigma=grad_sigma,
        tensor_sigma=tensor_sigma,
        gradient_operator=cfg["tensor"],
        flat_frac=flat_frac,
        corner_frac=corner_frac,
    )
    icfg = InitConfig(
        strategy=cfg["strategy"],
        num_gaussians=budget,
        density_base=density_base,
        density_power=density_power,
        density_mode=cfg["density"],
        sampling_mode=cfg["sampling"],
        max_axis_ratio=max_axis_ratio,
        coherence_power=coherence_power,
        scale_mode=cfg["scale"],
        init_scale_mult=init_scale_mult,
        flank_offset_frac=flank_offset,
        color_mode=cfg["color"],
        color_radius=color_radius,
        opacity_mode=cfg["opacity"],
        init_opacity=init_opacity,
        seed=seed,
    )
    refine = _refine_kwargs(cfg["refine"], split_every, split_count, prune_every, prune_min_activity)
    fcfg = FitConfig(
        iters=iters,
        render_chunk=render_chunk,
        ssim_weight=ssim_weight,
        compute_lpips=compute_lpips,
        pixel_loss=cfg["loss"],
        optimizer=cfg["optimizer"],
        lr_schedule=cfg["lr_schedule"],
        lr_decay_every=lr_decay_every,
        lr_decay_gamma=lr_decay_gamma,
        renderer=cfg["renderer"],
        max_gaussians=max_gaussians,
        log_every=max(1, iters // 4),
        **refine,
    )

    start = time.time()
    if cfg["pyramid"] == "single":
        field = _init.build_field(img, icfg, scfg, device=target.device)
        init_seconds = time.time() - start
        out = fit(field, target, fcfg, verbose=verbose)
        elapsed = time.time() - start
    elif cfg["pyramid"] == "pyramid":
        init_seconds = 0.0
        pcfg = PyramidConfig(
            levels=pyramid_levels,
            level_fractions=list(pyramid_fractions),
            iters_per_level=pyramid_iters_per_level,
        )
        out = fit_pyramid(img, target, icfg, fcfg, pcfg, scfg, verbose=verbose)
        elapsed = time.time() - start
    else:
        raise ValueError(f"unknown pyramid mode {cfg['pyramid']!r}; expected single or pyramid")

    return {
        "psnr": round(float(out["psnr"]), 4),
        "ssim": round(float(out["ssim"]), 5),
        "ms_ssim": round(float(out["ms_ssim"]), 5),
        "lpips": out.get("lpips"),
        "n_gaussians": int(out["n_gaussians"]),
        "init_seconds": init_seconds,
        "fit_seconds": float(out.get("fit_seconds", 0.0)) if cfg["pyramid"] == "single" else elapsed,
        "total_seconds": elapsed,
        "history": out.get("history", {}),
        "prefix_metrics": out.get("prefix_metrics"),
    }


def run_stage_search(
    images,
    budgets=(1024, 2048),
    seeds=(0,),
    iters=300,
    max_side: int | None = 320,
    strategies=("aniso_flanking",),
    tensor_operators=("central", "scharr"),
    density_modes=("structure", "hybrid"),
    sampling_modes=("wse",),
    color_modes=("bilinear", "local_mean", "two_sided"),
    scale_modes=("spacing",),
    opacity_modes=("none",),
    renderers=("normalized",),
    pixel_losses=("l1", "charbonnier"),
    optimizers=("adam",),
    lr_schedules=("none", "cosine"),
    refine_modes=("none", "residual_add"),
    pyramid_modes=("single",),
    render_chunk=512,
    ssim_weight=0.3,
    flank_offset=0.5,
    max_axis_ratio=6.0,
    coherence_power=1.0,
    init_scale_mult=1.0,
    density_base=0.05,
    density_power=1.0,
    flat_frac=0.02,
    corner_frac=0.15,
    grad_sigma=1.0,
    tensor_sigma=2.0,
    color_radius=1.5,
    init_opacity=0.9,
    lr_decay_every=None,
    lr_decay_gamma=0.5,
    split_every=None,
    split_count=0,
    prune_every=None,
    prune_min_activity=0.0,
    max_gaussians=None,
    pyramid_levels=2,
    pyramid_fractions=(0.35, 0.65),
    pyramid_iters_per_level=None,
    compute_lpips=False,
    outdir="results/stage_search",
    device=None,
    max_configs: int | None = None,
    shuffle_configs=False,
    config_seed=0,
    verbose=False,
):
    import torch

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_path = Path(outdir)
    out_path.mkdir(parents=True, exist_ok=True)
    files = _iter_images(images)
    if not files:
        raise SystemExit("no images found")
    if pyramid_iters_per_level is None:
        pyramid_iters_per_level = max(1, iters // max(1, pyramid_levels))
    if split_every is None:
        split_every = max(1, iters // 2)
    if split_count <= 0:
        split_count = 64
    if prune_every is None:
        prune_every = max(1, iters // 2)

    configs = list(_iter_configs(
        _one(strategies), _one(tensor_operators), _one(density_modes), _one(sampling_modes),
        _one(color_modes), _one(scale_modes), _one(opacity_modes), _one(renderers),
        _one(pixel_losses), _one(optimizers), _one(lr_schedules), _one(refine_modes),
        _one(pyramid_modes),
    ))
    if shuffle_configs:
        rng = np.random.default_rng(config_seed)
        rng.shuffle(configs)
    if max_configs is not None:
        configs = configs[:max_configs]

    rows = []
    for image_path in files:
        img = _load_image(image_path, max_side)
        target = torch.as_tensor(img, device=device)
        image_name = Path(image_path).stem
        for budget in budgets:
            for seed in seeds:
                for config_idx, cfg in enumerate(configs):
                    print(
                        f"[{image_name}] budget={budget} seed={seed} "
                        f"config={config_idx + 1}/{len(configs)} {cfg['label']}",
                        flush=True,
                    )
                    metrics = _run_one(
                        img, target, cfg, budget=budget, seed=seed, iters=iters,
                        render_chunk=render_chunk, ssim_weight=ssim_weight,
                        flank_offset=flank_offset, max_axis_ratio=max_axis_ratio,
                        coherence_power=coherence_power, init_scale_mult=init_scale_mult,
                        density_base=density_base, density_power=density_power,
                        flat_frac=flat_frac, corner_frac=corner_frac,
                        grad_sigma=grad_sigma, tensor_sigma=tensor_sigma,
                        color_radius=color_radius, init_opacity=init_opacity,
                        lr_decay_every=lr_decay_every,
                        lr_decay_gamma=lr_decay_gamma, split_every=split_every,
                        split_count=split_count, prune_every=prune_every,
                        prune_min_activity=prune_min_activity, max_gaussians=max_gaussians,
                        pyramid_levels=pyramid_levels, pyramid_fractions=pyramid_fractions,
                        pyramid_iters_per_level=pyramid_iters_per_level,
                        compute_lpips=compute_lpips, verbose=verbose,
                    )
                    row = {
                        "image": image_name,
                        "budget": budget,
                        "seed": seed,
                        **{k: cfg[k] for k in cfg if k != "label"},
                        "config_label": cfg["label"],
                        **metrics,
                    }
                    rows.append(row)
                    print({k: row[k] for k in row if k not in {"history", "prefix_metrics"}}, flush=True)

    _write(rows, out_path)
    return rows


def _config_key(row):
    keys = [
        "strategy", "tensor", "density", "sampling", "color", "scale", "renderer",
        "opacity", "loss", "optimizer", "lr_schedule", "refine", "pyramid", "budget",
    ]
    return tuple(row[k] for k in keys)


def summarize(rows, top_k: int = 20) -> str:
    groups: dict[tuple, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(_config_key(row), []).append(row)
    ranked = []
    for key, vals in groups.items():
        psnrs = [v["psnr"] for v in vals]
        ranked.append((mean(psnrs), key, vals))
    ranked.sort(reverse=True, key=lambda x: x[0])

    lines = [
        "# StructSplat Stage Search",
        "",
        "| Rank | Budget | Mean PSNR | Std | Mean MS-SSIM | Mean seconds | Config |",
        "|---:|---:|---:|---:|---:|---:|---|",
    ]
    for rank, (_, _key, vals) in enumerate(ranked[:top_k], 1):
        psnrs = [v["psnr"] for v in vals]
        ms = [v["ms_ssim"] for v in vals]
        sec = [v["fit_seconds"] for v in vals]
        label = vals[0]["config_label"]
        budget = vals[0]["budget"]
        lines.append(
            f"| {rank} | {budget} | {mean(psnrs):.4f} | {pstdev(psnrs):.4f} | "
            f"{mean(ms):.5f} | {mean(sec):.2f} | `{label}` |"
        )
    return "\n".join(lines) + "\n"


def _write(rows, outdir: Path):
    (outdir / "stage_search.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    if rows:
        with (outdir / "stage_search.csv").open("w", newline="", encoding="utf-8") as f:
            fields = [k for k in rows[0].keys() if k not in {"history", "prefix_metrics"}]
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for row in rows:
                w.writerow({k: row.get(k) for k in fields})
    (outdir / "summary.md").write_text(summarize(rows), encoding="utf-8")
    print(f"\nwrote stage_search.json / stage_search.csv / summary.md to {outdir}")


def main():
    p = argparse.ArgumentParser(description="Search StructSplat stage alternatives")
    p.add_argument("images", nargs="+")
    p.add_argument("--budgets", type=int, nargs="+", default=[1024, 2048])
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--iters", type=int, default=300)
    p.add_argument("--max-side", type=int, default=320)
    p.add_argument("--strategies", nargs="+", default=["aniso_flanking"])
    p.add_argument("--tensor-operators", nargs="+", default=["central", "scharr"])
    p.add_argument("--density-modes", nargs="+", default=["structure", "hybrid"])
    p.add_argument("--sampling-modes", nargs="+", default=["wse"])
    p.add_argument("--color-modes", nargs="+", default=["bilinear", "local_mean", "two_sided"])
    p.add_argument("--scale-modes", nargs="+", default=["spacing"])
    p.add_argument("--opacity-modes", nargs="+", default=["none"])
    p.add_argument("--renderers", nargs="+", default=["normalized"])
    p.add_argument("--pixel-losses", nargs="+", default=["l1", "charbonnier"])
    p.add_argument("--optimizers", nargs="+", default=["adam"])
    p.add_argument("--lr-schedules", nargs="+", default=["none", "cosine"])
    p.add_argument("--refine-modes", nargs="+", default=["none", "residual_add"])
    p.add_argument("--pyramid-modes", nargs="+", default=["single"])
    p.add_argument("--chunk", type=int, default=512)
    p.add_argument("--ssim-weight", type=float, default=0.3)
    p.add_argument("--split-every", type=int, default=None)
    p.add_argument("--split-count", type=int, default=64)
    p.add_argument("--prune-every", type=int, default=None)
    p.add_argument("--prune-min-activity", type=float, default=0.0)
    p.add_argument("--max-gaussians", type=int, default=None)
    p.add_argument("--pyramid-levels", type=int, default=2)
    p.add_argument("--pyramid-fractions", type=float, nargs="+", default=[0.35, 0.65])
    p.add_argument("--pyramid-iters-per-level", type=int, default=None)
    p.add_argument("--lpips", action="store_true")
    p.add_argument("--max-configs", type=int, default=None)
    p.add_argument("--shuffle-configs", action="store_true")
    p.add_argument("--config-seed", type=int, default=0)
    p.add_argument("--outdir", default="results/stage_search")
    p.add_argument("--device", default=None)
    p.add_argument("--verbose", action="store_true")
    a = p.parse_args()
    run_stage_search(
        a.images, budgets=a.budgets, seeds=a.seeds, iters=a.iters,
        max_side=a.max_side, strategies=a.strategies, tensor_operators=a.tensor_operators,
        density_modes=a.density_modes, sampling_modes=a.sampling_modes,
        color_modes=a.color_modes, scale_modes=a.scale_modes, opacity_modes=a.opacity_modes,
        renderers=a.renderers, pixel_losses=a.pixel_losses, optimizers=a.optimizers,
        lr_schedules=a.lr_schedules, refine_modes=a.refine_modes,
        pyramid_modes=a.pyramid_modes, render_chunk=a.chunk,
        ssim_weight=a.ssim_weight, split_every=a.split_every, split_count=a.split_count,
        prune_every=a.prune_every, prune_min_activity=a.prune_min_activity,
        max_gaussians=a.max_gaussians, pyramid_levels=a.pyramid_levels,
        pyramid_fractions=a.pyramid_fractions, pyramid_iters_per_level=a.pyramid_iters_per_level,
        compute_lpips=a.lpips, max_configs=a.max_configs, shuffle_configs=a.shuffle_configs,
        config_seed=a.config_seed, outdir=a.outdir, device=a.device, verbose=a.verbose,
    )


if __name__ == "__main__":
    main()
