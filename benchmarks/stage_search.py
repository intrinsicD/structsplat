"""Stage-wise StructSplat search across init, density, fitting, refinement, and pyramid choices.

Two modes (ABL-002 / ADR-0010):
  * factorial — the original full product of the requested per-stage options, for finding the
    best complete configuration. Configs that only differ in a stage that provably cannot
    affect the output (e.g. tensor operator under strategy=random) are canonicalized and
    deduplicated so they neither waste compute nor confound the marginal statistics.
  * influence — one-factor-at-a-time around a baseline: the FIRST value of every stage axis is
    the baseline; each remaining value is run with every other stage pinned to baseline. The
    summary reports *paired* deltas (per image x budget x seed) against the baseline, which is
    the direct answer to "what is the influence of this stage?" for quality (PSNR/MS-SSIM/
    LPIPS), convergence (iters-to-target, PSNR AUC), and speed (init/fit seconds).
"""
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

# stage axes, in label order; values = the swappable options each stage exposes
STAGE_KEYS = [
    "strategy", "tensor", "tensor_color", "density", "sampling", "orientation", "color",
    "scale", "scale_cap", "opacity", "renderer", "loss", "optimizer", "lr_schedule",
    "refine", "pyramid",
]

FACTORIAL_DEFAULTS: dict[str, tuple[str, ...]] = {
    "strategies": ("aniso_flanking",),
    "tensor_operators": ("central", "scharr"),
    "tensor_colors": ("luma",),
    "density_modes": ("structure", "hybrid"),
    "sampling_modes": ("wse",),
    "orientation_modes": ("tensor",),
    "color_modes": ("bilinear", "local_mean", "two_sided"),
    "scale_modes": ("spacing",),
    "scale_cap_modes": ("feature12",),
    "opacity_modes": ("none",),
    "renderers": ("normalized",),
    "pixel_losses": ("l1", "charbonnier"),
    "optimizers": ("adam",),
    "lr_schedules": ("none", "cosine"),
    "refine_modes": ("none", "residual_add"),
    "pyramid_modes": ("single",),
}

# influence mode: first value per axis = the baseline (ADR-0009 defaults), rest = the variants
INFLUENCE_DEFAULTS: dict[str, tuple[str, ...]] = {
    "strategies": (
        "aniso_flanking", "quadtree_wse", "quadtree_hybrid", "quadtree_aggregate",
        "aniso_onedge", "iso_blue_noise", "grid", "random"
    ),
    "tensor_operators": ("central", "sobel", "scharr"),
    "tensor_colors": ("luma", "rgb"),
    "density_modes": ("structure", "gradient", "variance", "hybrid", "uniform"),
    "sampling_modes": ("wse", "dart_throwing", "halton", "cvt", "farthest_point",
                       "density_random", "jittered_grid"),
    "orientation_modes": ("tensor", "random", "zero"),
    "color_modes": ("bilinear", "local_mean", "two_sided"),
    "scale_modes": ("spacing", "uniform", "knn"),
    "scale_cap_modes": ("feature12", "none", "hard8"),
    "opacity_modes": ("none", "constant"),
    "renderers": ("normalized", "additive", "cuda", "cuda_additive"),
    "pixel_losses": ("l1", "l2", "charbonnier"),
    "optimizers": ("adam", "adamw"),
    "lr_schedules": ("none", "cosine", "step"),
    "refine_modes": (
        "none", "prune", "duplicate", "support_duplicate", "residual_add",
        "residual_tensor_add", "prune_residual_add", "prune_residual_tensor_add"
    ),
    "pyramid_modes": ("single", "pyramid"),
}

_AXIS_TO_KEY = {
    "strategies": "strategy", "tensor_operators": "tensor", "tensor_colors": "tensor_color",
    "density_modes": "density", "sampling_modes": "sampling",
    "orientation_modes": "orientation", "color_modes": "color", "scale_modes": "scale",
    "scale_cap_modes": "scale_cap", "opacity_modes": "opacity", "renderers": "renderer",
    "pixel_losses": "loss",
    "optimizers": "optimizer", "lr_schedules": "lr_schedule", "refine_modes": "refine",
    "pyramid_modes": "pyramid",
}


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
    return "|".join(f"{k}={c[k]}" for k in STAGE_KEYS)


def _canonicalize(cfg: dict[str, Any], canonical: dict[str, str]) -> dict[str, Any]:
    """Pin stage fields that provably produce the IDENTICAL initial field for this config.

    Two configs that only differ in a pinned field are the *same experiment*; running both
    would double-count one cell and bias any per-stage marginal statistics. Only exact
    init-level equivalences are pinned — in particular, orientation is NOT pinned for
    isotropic inits (equal initial axes still break symmetry through fitting: the rotation
    decides which axis each scale gradient feeds), except where the angles are exactly equal:
      * random/grid do not read density/sampling. Tensor/tensor_color are inert there unless
        the scale-cap stage is feature-based, because that cap computes tensor run lengths.
        orientation 'tensor' == 'zero' (both give zero angles; 'random' stays distinct).
      * jittered_grid placement never reads the density map (angles/ratios come from the
        tensor, not the density), so the density stage is inert under it.
      * two_sided color sampling only diverges from bilinear inside the aniso_flanking branch.
    """
    c = dict(cfg)
    strat = c["strategy"]
    if strat in ("random", "grid"):
        feature_cap = str(c.get("scale_cap", "none")).startswith("feature")
        inert = ("density", "sampling") if feature_cap else (
            "tensor", "tensor_color", "density", "sampling"
        )
        for k in inert:
            c[k] = canonical[k]
        if c["orientation"] in ("tensor", "zero"):
            c["orientation"] = "tensor" if canonical["orientation"] in ("tensor", "zero") \
                else "zero"
    if strat not in ("random", "grid") and c["sampling"] == "jittered_grid":
        c["density"] = canonical["density"]
    if strat != "aniso_flanking" and c["color"] == "two_sided":
        c["color"] = "bilinear"
    return c


def _iter_configs(axes: dict[str, tuple]):
    names = [_AXIS_TO_KEY[a] for a in axes]
    for values in itertools.product(*axes.values()):
        yield dict(zip(names, values, strict=True))


def _influence_configs(axes: dict[str, tuple]):
    base = {_AXIS_TO_KEY[a]: vals[0] for a, vals in axes.items()}
    yield dict(base)
    for a, vals in axes.items():
        key = _AXIS_TO_KEY[a]
        for v in vals[1:]:
            yield {**base, key: v}


def _refine_kwargs(mode: str, split_every: int | None, split_count: int,
                   prune_every: int | None, prune_min_activity: float) -> dict[str, Any]:
    if mode == "none":
        return {}
    if mode == "prune":
        return {
            "prune_every": prune_every,
            "prune_min_activity": prune_min_activity,
        }
    if mode in ("duplicate", "support_duplicate", "residual_add", "residual_tensor_add"):
        return {
            "split_every": split_every,
            "split_count": split_count,
            "split_mode": mode,
        }
    if mode in ("prune_residual_add", "prune_residual_tensor_add"):
        return {
            "prune_every": prune_every,
            "prune_min_activity": prune_min_activity,
            "split_every": split_every,
            "split_count": split_count,
            "split_mode": "residual_tensor_add"
            if mode == "prune_residual_tensor_add" else "residual_add",
        }
    raise ValueError(
        f"unknown refine mode {mode!r}; expected none, prune, duplicate, support_duplicate, "
        "residual_add, residual_tensor_add, prune_residual_add, or prune_residual_tensor_add"
    )


def _scale_cap_kwargs(mode: str) -> dict[str, Any]:
    if mode in ("none", "uncapped"):
        return {"scale_cap_mode": "none", "scale_cap_max": None}
    aliases = {
        "hard8": ("hard", 8.0),
        "hard12": ("hard", 12.0),
        "feature8": ("feature", 8.0),
        "feature12": ("feature", 12.0),
        "feature_cap8": ("feature", 8.0),
        "feature_cap12": ("feature", 12.0),
    }
    if mode in aliases:
        cap_mode, cap = aliases[mode]
        return {"scale_cap_mode": cap_mode, "scale_cap_max": cap}
    for prefix, cap_mode in (("feature_cap", "feature"), ("feature", "feature"), ("hard", "hard")):
        if mode.startswith(prefix):
            suffix = mode[len(prefix):].lstrip("_")
            try:
                cap = float(suffix)
            except ValueError as exc:
                raise ValueError(f"cannot parse scale cap mode {mode!r}") from exc
            return {"scale_cap_mode": cap_mode, "scale_cap_max": cap}
    raise ValueError(
        f"unknown scale_cap mode {mode!r}; expected none, hard8, hard12, feature8, or feature12"
    )


def _psnr_auc(history: dict) -> float | None:
    """Mean PSNR over the training trajectory (trapezoid over the logged history).

    A single number that rewards both converging fast and converging high; complements
    iters-to-target, which saturates once every config reaches the target.
    """
    its, ps = history.get("iter", []), history.get("psnr", [])
    if len(its) < 2:
        return float(ps[0]) if ps else None
    trapezoid = getattr(np, "trapezoid", None) or np.trapz  # numpy<2 fallback
    return float(trapezoid(ps, its) / max(its[-1] - its[0], 1))


def _seconds_to_target(history: dict, iters_to_target) -> float | None:
    """Wall seconds at the iteration where the target PSNR was first reached (interpolated)."""
    if iters_to_target is None:
        return None
    its, el = history.get("iter", []), history.get("elapsed", [])
    if not its:
        return None
    return float(np.interp(iters_to_target, its, el))


def _run_one(img, target, cfg, *, budget, seed, iters, render_chunk, ssim_weight,
             flank_offset, max_axis_ratio, coherence_power, init_scale_mult,
             density_base, density_power, flat_frac, corner_frac, grad_sigma, tensor_sigma,
             color_radius, init_opacity, lr_decay_every, lr_decay_gamma, split_every, split_count,
             prune_every, prune_min_activity, max_gaussians, pyramid_levels,
             pyramid_fractions, pyramid_iters_per_level, compute_lpips,
             target_psnr, target_psnrs, log_every, verbose):
    from structsplat import init as _init
    from structsplat.fit import fit
    from structsplat.pyramid import fit_pyramid

    scfg = StructureTensorConfig(
        grad_sigma=grad_sigma,
        tensor_sigma=tensor_sigma,
        gradient_operator=cfg["tensor"],
        color_space=cfg["tensor_color"],
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
        orientation_mode=cfg["orientation"],
        scale_mode=cfg["scale"],
        **_scale_cap_kwargs(cfg["scale_cap"]),
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
        # only step reads it; leaking it into schedule="none" configs would silently
        # re-enable step decay through fit's backward-compat fallback
        lr_decay_every=lr_decay_every if cfg["lr_schedule"] == "step" else None,
        lr_decay_gamma=lr_decay_gamma,
        renderer=cfg["renderer"],
        max_gaussians=max_gaussians,
        target_psnr=target_psnr,
        target_psnrs=list(target_psnrs),
        log_every=log_every,
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

    history = out.get("history", {})
    iters_to_target = out.get("iters_to_target")
    fit_seconds = float(out.get("fit_seconds", 0.0))
    if cfg["pyramid"] == "pyramid":
        # fit_pyramid aggregates per-level fit time; the rest of the wall clock is the
        # interleaved init/density/tensor work, i.e. this mode's init cost
        init_seconds = max(elapsed - fit_seconds, 0.0)
    return {
        "psnr": round(float(out["psnr"]), 4),
        "ssim": round(float(out["ssim"]), 5),
        "ms_ssim": round(float(out["ms_ssim"]), 5),
        "lpips": out.get("lpips"),
        "auc_psnr": _psnr_auc(history),
        "iters_to_target": iters_to_target,
        "iters_to_targets": out.get("iters_to_targets", {}),
        "seconds_to_target": _seconds_to_target(history, iters_to_target),
        "n_gaussians": int(out["n_gaussians"]),
        "init_seconds": init_seconds,
        "fit_seconds": fit_seconds,
        "total_seconds": elapsed,
        "history": history,
        "prefix_metrics": out.get("prefix_metrics"),
    }


def run_stage_search(
    images,
    budgets=(1024, 2048),
    seeds=(0,),
    iters=300,
    max_side: int | None = 320,
    mode: str = "factorial",
    strategies=None,
    tensor_operators=None,
    tensor_colors=None,
    density_modes=None,
    sampling_modes=None,
    orientation_modes=None,
    color_modes=None,
    scale_modes=None,
    scale_cap_modes=None,
    opacity_modes=None,
    renderers=None,
    pixel_losses=None,
    optimizers=None,
    lr_schedules=None,
    refine_modes=None,
    pyramid_modes=None,
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
    target_psnr: float | None = None,
    target_psnrs=(),
    log_every: int | None = None,
    dedupe: bool = True,
    outdir="results/stage_search",
    device=None,
    max_configs: int | None = None,
    shuffle_configs=False,
    config_seed=0,
    verbose=False,
):
    import torch

    if mode not in ("factorial", "influence"):
        raise ValueError(f"unknown mode {mode!r}; expected factorial or influence")
    defaults = INFLUENCE_DEFAULTS if mode == "influence" else FACTORIAL_DEFAULTS
    supplied = {
        "strategies": strategies, "tensor_operators": tensor_operators,
        "tensor_colors": tensor_colors, "density_modes": density_modes,
        "sampling_modes": sampling_modes, "orientation_modes": orientation_modes,
        "color_modes": color_modes, "scale_modes": scale_modes,
        "scale_cap_modes": scale_cap_modes,
        "opacity_modes": opacity_modes, "renderers": renderers,
        "pixel_losses": pixel_losses, "optimizers": optimizers,
        "lr_schedules": lr_schedules, "refine_modes": refine_modes,
        "pyramid_modes": pyramid_modes,
    }
    axes = {a: _one(v) if v is not None else defaults[a] for a, v in supplied.items()}

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
    if prune_min_activity <= 0.0:
        # 0 makes the prune refine modes silent no-ops (fit disables pruning at <=0);
        # a run that claims to test pruning must actually prune something prunable
        prune_min_activity = 1e-2
    if lr_decay_every is None:
        # FitConfig's step fallback is 500, a no-op at screening iteration counts
        lr_decay_every = max(1, iters // 3)
    if log_every is None:
        log_every = max(1, iters // 20)  # fine enough for a meaningful PSNR AUC

    canonical = {_AXIS_TO_KEY[a]: vals[0] for a, vals in axes.items()}
    raw = _influence_configs(axes) if mode == "influence" else _iter_configs(axes)
    configs, seen = [], set()
    n_dropped = 0
    for cfg in raw:
        c = _canonicalize(cfg, canonical) if dedupe else dict(cfg)
        c["label"] = _config_label(c)
        if c["label"] in seen:
            n_dropped += 1
            continue
        seen.add(c["label"])
        configs.append(c)
    if n_dropped:
        print(f"deduplicated {n_dropped} configs equivalent to an already-scheduled one")
    if shuffle_configs:
        rng = np.random.default_rng(config_seed)
        rng.shuffle(configs)
    if max_configs is not None:
        configs = configs[:max_configs]

    baseline_label = None
    if mode == "influence":
        base = _canonicalize({_AXIS_TO_KEY[a]: vals[0] for a, vals in axes.items()}, canonical)
        baseline_label = _config_label(base)

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
                        compute_lpips=compute_lpips, target_psnr=target_psnr,
                        target_psnrs=target_psnrs, log_every=log_every, verbose=verbose,
                    )
                    row = {
                        "image": image_name,
                        "budget": budget,
                        "seed": seed,
                        **{k: cfg[k] for k in cfg if k != "label"},
                        "config_label": cfg["label"],
                        "is_baseline": cfg["label"] == baseline_label,
                        **metrics,
                    }
                    rows.append(row)
                    print({k: row[k] for k in row if k not in {"history", "prefix_metrics"}}, flush=True)

    _write(rows, out_path, mode=mode, baseline_label=baseline_label)
    return rows


def _config_key(row):
    return tuple(row[k] for k in STAGE_KEYS) + (row["budget"],)


def _fmt(v, spec=".4f"):
    return format(v, spec) if v is not None else "-"


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
        "| Rank | Budget | Mean PSNR | Std | Mean MS-SSIM | Mean AUC | Iters→target | Mean fit s | Config |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for rank, (_, _key, vals) in enumerate(ranked[:top_k], 1):
        psnrs = [v["psnr"] for v in vals]
        ms = [v["ms_ssim"] for v in vals]
        sec = [v["fit_seconds"] for v in vals]
        aucs = [v["auc_psnr"] for v in vals if v.get("auc_psnr") is not None]
        itt = [v["iters_to_target"] for v in vals if v.get("iters_to_target") is not None]
        label = vals[0]["config_label"]
        budget = vals[0]["budget"]
        lines.append(
            f"| {rank} | {budget} | {mean(psnrs):.4f} | {pstdev(psnrs):.4f} | "
            f"{mean(ms):.5f} | {_fmt(mean(aucs) if aucs else None, '.3f')} | "
            f"{_fmt(mean(itt) if itt else None, '.0f')} ({len(itt)}/{len(vals)}) | "
            f"{mean(sec):.2f} | `{label}` |"
        )
    lines += ["", stage_effects(rows)]
    return "\n".join(lines) + "\n"


def stage_effects(rows) -> str:
    """Marginal means per stage level, over all runs that share that level.

    For factorial runs these are observational marginals (levels co-vary with the other
    stages that were swept); for influence-mode runs prefer `summarize_influence`, which
    reports paired deltas against the baseline instead.
    """
    lines = [
        "## Per-stage marginal means",
        "",
        "| Stage | Level | Runs | PSNR | MS-SSIM | AUC | Iters→target | Fit s |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for stage in STAGE_KEYS:
        levels = sorted({r[stage] for r in rows})
        if len(levels) < 2:
            continue
        for lv in levels:
            sub = [r for r in rows if r[stage] == lv]
            psnrs = [r["psnr"] for r in sub]
            ms = [r["ms_ssim"] for r in sub]
            aucs = [r["auc_psnr"] for r in sub if r.get("auc_psnr") is not None]
            itt = [r["iters_to_target"] for r in sub if r.get("iters_to_target") is not None]
            sec = [r["fit_seconds"] for r in sub]
            lines.append(
                f"| {stage} | {lv} | {len(sub)} | {mean(psnrs):.3f} ± {pstdev(psnrs):.3f} | "
                f"{mean(ms):.5f} | {_fmt(mean(aucs) if aucs else None, '.3f')} | "
                f"{_fmt(mean(itt) if itt else None, '.0f')} ({len(itt)}/{len(sub)}) | "
                f"{mean(sec):.2f} |"
            )
    return "\n".join(lines)


def summarize_influence(rows, baseline_label: str) -> str:
    """Paired per-stage deltas vs the baseline config: the stage-influence answer.

    Each variant row is matched with the baseline row of the same (image, budget, seed) and
    the metric differences are aggregated. Positive ΔPSNR/ΔAUC = variant better; negative
    Δiters-to-target / Δseconds = variant faster.
    """
    base = {(r["image"], r["budget"], r["seed"]): r for r in rows if r["config_label"] == baseline_label}
    if not base:
        return "# Stage influence\n\n(no baseline rows found)\n"

    variants: dict[tuple[str, str], list[dict]] = {}
    for r in rows:
        if r["config_label"] == baseline_label:
            continue
        b = base.get((r["image"], r["budget"], r["seed"]))
        if b is None:
            continue
        diff = [k for k in STAGE_KEYS if r[k] != b[k]]
        stage = diff[0] if len(diff) == 1 else "+".join(diff)
        variants.setdefault((stage, "|".join(f"{k}={r[k]}" for k in diff)), []).append(
            {"r": r, "b": b})

    def dstat(pairs, key):
        ds = [p["r"][key] - p["b"][key] for p in pairs
              if p["r"].get(key) is not None and p["b"].get(key) is not None]
        if not ds:
            return "-"
        return f"{mean(ds):+.3f} ± {pstdev(ds):.3f}"

    def treach(pairs):
        rv = sum(1 for p in pairs if p["r"].get("iters_to_target") is not None)
        rb = sum(1 for p in pairs if p["b"].get("iters_to_target") is not None)
        return f"{rv}/{rb}/{len(pairs)}"

    b0 = base[next(iter(base))]
    lines = [
        "# Stage influence (paired deltas vs baseline)",
        "",
        f"Baseline: `{baseline_label}`",
        f"Baseline means: PSNR {mean(r['psnr'] for r in base.values()):.3f}, "
        f"MS-SSIM {mean(r['ms_ssim'] for r in base.values()):.5f}, "
        f"AUC {_fmt(b0.get('auc_psnr'), '.3f')}, "
        f"fit {mean(r['fit_seconds'] for r in base.values()):.2f}s over {len(base)} cells.",
        "",
        "Positive ΔPSNR/ΔMS-SSIM/ΔAUC = variant better than baseline; negative Δiters/Δs = faster.",
        "reached = target reached (variant/baseline/cells).",
        "",
        "| Stage | Variant | Cells | ΔPSNR | ΔMS-SSIM | ΔAUC | Δiters→target | reached | Δinit s | Δfit s |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    order = {k: i for i, k in enumerate(STAGE_KEYS)}
    for (stage, label), pairs in sorted(variants.items(),
                                        key=lambda kv: (order.get(kv[0][0], 99), kv[0][1])):
        lines.append(
            f"| {stage} | `{label}` | {len(pairs)} | {dstat(pairs, 'psnr')} | "
            f"{dstat(pairs, 'ms_ssim')} | {dstat(pairs, 'auc_psnr')} | "
            f"{dstat(pairs, 'iters_to_target')} | {treach(pairs)} | "
            f"{dstat(pairs, 'init_seconds')} | {dstat(pairs, 'fit_seconds')} |"
        )
    return "\n".join(lines) + "\n"


def _write(rows, outdir: Path, mode: str = "factorial", baseline_label: str | None = None):
    (outdir / "stage_search.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    if rows:
        with (outdir / "stage_search.csv").open("w", newline="", encoding="utf-8") as f:
            fields = [k for k in rows[0].keys() if k not in {"history", "prefix_metrics"}]
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for row in rows:
                w.writerow({k: row.get(k) for k in fields})
    (outdir / "summary.md").write_text(summarize(rows), encoding="utf-8")
    wrote = "stage_search.json / stage_search.csv / summary.md"
    if mode == "influence" and baseline_label is not None:
        (outdir / "influence.md").write_text(summarize_influence(rows, baseline_label),
                                             encoding="utf-8")
        wrote += " / influence.md"
    print(f"\nwrote {wrote} to {outdir}")


def main():
    p = argparse.ArgumentParser(description="Search StructSplat stage alternatives")
    p.add_argument("images", nargs="+")
    p.add_argument("--mode", choices=["factorial", "influence"], default="factorial",
                   help="factorial: full product of the given options; influence: "
                        "one-factor-at-a-time deltas around the baseline (first value of "
                        "each axis)")
    p.add_argument("--budgets", type=int, nargs="+", default=[1024, 2048])
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--iters", type=int, default=300)
    p.add_argument("--max-side", type=int, default=320)
    p.add_argument("--strategies", nargs="+", default=None)
    p.add_argument("--tensor-operators", nargs="+", default=None)
    p.add_argument("--tensor-colors", nargs="+", default=None)
    p.add_argument("--density-modes", nargs="+", default=None)
    p.add_argument("--sampling-modes", nargs="+", default=None)
    p.add_argument("--orientation-modes", nargs="+", default=None)
    p.add_argument("--color-modes", nargs="+", default=None)
    p.add_argument("--scale-modes", nargs="+", default=None)
    p.add_argument("--scale-cap-modes", nargs="+", default=None,
                   help="none, hard8/hard12, feature8/feature12, or feature_cap<N>")
    p.add_argument("--opacity-modes", nargs="+", default=None)
    p.add_argument("--renderers", nargs="+", default=None)
    p.add_argument("--pixel-losses", nargs="+", default=None)
    p.add_argument("--optimizers", nargs="+", default=None)
    p.add_argument("--lr-schedules", nargs="+", default=None)
    p.add_argument("--refine-modes", nargs="+", default=None)
    p.add_argument("--pyramid-modes", nargs="+", default=None)
    p.add_argument("--chunk", type=int, default=512)
    p.add_argument("--ssim-weight", type=float, default=0.3)
    p.add_argument("--target-psnr", type=float, default=None,
                   help="record iters/seconds-to-target for convergence-rate comparisons")
    p.add_argument("--target-psnrs", type=float, nargs="*", default=[])
    p.add_argument("--log-every", type=int, default=None)
    p.add_argument("--no-dedupe", action="store_true",
                   help="keep configs that are provably equivalent (not recommended)")
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
        a.images, budgets=a.budgets, seeds=a.seeds, iters=a.iters, mode=a.mode,
        max_side=a.max_side, strategies=a.strategies, tensor_operators=a.tensor_operators,
        tensor_colors=a.tensor_colors, density_modes=a.density_modes,
        sampling_modes=a.sampling_modes, orientation_modes=a.orientation_modes,
        color_modes=a.color_modes, scale_modes=a.scale_modes,
        scale_cap_modes=a.scale_cap_modes, opacity_modes=a.opacity_modes,
        renderers=a.renderers, pixel_losses=a.pixel_losses, optimizers=a.optimizers,
        lr_schedules=a.lr_schedules, refine_modes=a.refine_modes,
        pyramid_modes=a.pyramid_modes, render_chunk=a.chunk,
        ssim_weight=a.ssim_weight, split_every=a.split_every, split_count=a.split_count,
        prune_every=a.prune_every, prune_min_activity=a.prune_min_activity,
        max_gaussians=a.max_gaussians, pyramid_levels=a.pyramid_levels,
        pyramid_fractions=a.pyramid_fractions, pyramid_iters_per_level=a.pyramid_iters_per_level,
        compute_lpips=a.lpips, target_psnr=a.target_psnr, target_psnrs=a.target_psnrs,
        log_every=a.log_every, dedupe=not a.no_dedupe,
        max_configs=a.max_configs, shuffle_configs=a.shuffle_configs,
        config_seed=a.config_seed, outdir=a.outdir, device=a.device, verbose=a.verbose,
    )


if __name__ == "__main__":
    main()
