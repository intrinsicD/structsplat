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
import glob
import itertools
import json
import os
import time
from html import escape
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import numpy as np

from benchmarks.common import load_image as _load_image
from benchmarks.common import psnr_auc as _psnr_auc
from benchmarks.common import run_config, write_config, write_csv, write_json
from structsplat.config import FitConfig, InitConfig, PyramidConfig, StructureTensorConfig

# stage axes, in label order; values = the swappable options each stage exposes
STAGE_KEYS = [
    "strategy", "tensor", "tensor_color", "density", "sampling", "orientation", "color",
    "scale", "scale_cap", "opacity", "renderer", "aa", "color_basis", "color_solve", "loss",
    "optimizer", "lr_schedule", "refine", "pyramid",
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
    # baseline matches the shipped default (config.py scale_cap_mode='none', ADR-0009); it had
    # silently diverged to feature12 with no held-out justification (BENCH-002)
    "scale_cap_modes": ("none",),
    "opacity_modes": ("none",),
    "renderers": ("normalized",),
    "aa_dilations": (0.0,),
    "color_basis_modes": ("constant",),
    "color_solve_modes": ("none",),
    "pixel_losses": ("l1", "charbonnier"),
    "optimizers": ("adam",),
    "lr_schedules": ("none", "cosine"),
    "refine_modes": ("none", "residual_add"),
    "pyramid_modes": ("single",),
}

# influence mode: FIRST value per axis = the baseline (the shipped ADR-0009 defaults), rest = the
# variants. Every first value must equal config.py's default for that stage (BENCH-002).
INFLUENCE_DEFAULTS: dict[str, tuple[str, ...]] = {
    "strategies": (
        "aniso_flanking", "quadtree_wse", "quadtree_hybrid", "quadtree_aggregate",
        "aniso_onedge", "iso_blue_noise", "grid", "random"
    ),
    "tensor_operators": ("central", "sobel", "scharr"),
    "tensor_colors": ("luma", "rgb"),
    "density_modes": ("structure", "gradient", "variance", "hybrid", "uniform"),
    "sampling_modes": ("wse", "floyd_steinberg", "dart_throwing", "halton", "cvt",
                       "farthest_point", "density_random", "jittered_grid"),
    "orientation_modes": ("tensor", "random", "zero"),
    "color_modes": ("bilinear", "local_mean", "two_sided"),
    "scale_modes": ("spacing", "uniform", "knn"),
    "scale_cap_modes": ("none", "feature12", "hard8"),
    "opacity_modes": ("none", "constant"),
    "renderers": (
        "normalized", "additive", "cuda", "cuda_additive",
        "cuda_tiled", "cuda_tiled_additive",
    ),
    "aa_dilations": (0.0, 0.3),
    "color_basis_modes": ("constant", "affine"),
    "color_solve_modes": ("none", "every10"),
    "pixel_losses": ("l1", "l2", "charbonnier"),
    "optimizers": ("adam", "adamw", "adan"),
    "lr_schedules": ("none", "cosine", "step"),
    "refine_modes": (
        "none", "prune", "duplicate", "fp_duplicate", "moment_preserving",
        "support_duplicate", "residual_add", "residual_tensor_add", "ranked_wave",
        "absgrad_wave", "freq_violation", "relocate",
        "residual_add_nms", "residual_tensor_add_nms",
        "residual_add_residual_color", "residual_tensor_add_residual_color",
        "residual_add_nms_residual_color", "residual_tensor_add_nms_residual_color",
        "prune_residual_add", "prune_residual_tensor_add",
    ),
    "pyramid_modes": ("single", "pyramid"),
}

# strategies whose placement actually routes through _blue_noise_positions and therefore reads
# the top-level sampling_mode (init.py else-branch). Quadtree strategies have their own placement
# and ignore sampling_mode, so a jittered_grid pin must NOT touch them (BENCH-002).
_BLUE_NOISE_STRATEGIES = ("iso_blue_noise", "aniso_onedge", "aniso_flanking")

_AXIS_TO_KEY = {
    "strategies": "strategy", "tensor_operators": "tensor", "tensor_colors": "tensor_color",
    "density_modes": "density", "sampling_modes": "sampling",
    "orientation_modes": "orientation", "color_modes": "color", "scale_modes": "scale",
    "scale_cap_modes": "scale_cap", "opacity_modes": "opacity", "renderers": "renderer",
    "aa_dilations": "aa", "color_basis_modes": "color_basis",
    "color_solve_modes": "color_solve", "pixel_losses": "loss",
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
    # jittered_grid placement ignores the density map — but only for strategies that actually
    # place via _blue_noise_positions. Quadtree strategies DO read density (their leaves are
    # density-prioritized), so pinning density there wrongly deduped genuinely distinct cells.
    if strat in _BLUE_NOISE_STRATEGIES and c["sampling"] == "jittered_grid":
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
    split_variants: dict[str, dict[str, Any]] = {
        "duplicate": {"split_mode": "duplicate"},
        "fp_duplicate": {"split_mode": "fp_duplicate"},
        "moment_preserving": {"split_mode": "moment_preserving"},
        "support_duplicate": {"split_mode": "support_duplicate"},
        "residual_add": {"split_mode": "residual_add"},
        "residual_tensor_add": {"split_mode": "residual_tensor_add"},
        "ranked_wave": {"split_mode": "ranked_wave"},
        "absgrad_wave": {"split_mode": "absgrad_wave"},
        "freq_violation": {"split_mode": "freq_violation"},
        "residual_add_nms": {
            "split_mode": "residual_add",
            "split_min_spacing": 1.0,
            "split_oversample": 8.0,
        },
        "residual_tensor_add_nms": {
            "split_mode": "residual_tensor_add",
            "split_min_spacing": 1.0,
            "split_oversample": 8.0,
        },
        "residual_add_residual_color": {
            "split_mode": "residual_add",
            "split_color_init": "residual",
        },
        "residual_tensor_add_residual_color": {
            "split_mode": "residual_tensor_add",
            "split_color_init": "residual",
        },
        "residual_add_nms_residual_color": {
            "split_mode": "residual_add",
            "split_min_spacing": 1.0,
            "split_oversample": 8.0,
            "split_color_init": "residual",
        },
        "residual_tensor_add_nms_residual_color": {
            "split_mode": "residual_tensor_add",
            "split_min_spacing": 1.0,
            "split_oversample": 8.0,
            "split_color_init": "residual",
        },
    }
    if mode == "none":
        return {}
    if mode == "prune":
        return {
            "prune_every": prune_every,
            "prune_min_activity": prune_min_activity,
        }
    if mode == "relocate":
        return {
            "relocate_every": split_every,
            "relocate_count": split_count,
        }
    if mode in split_variants:
        return {
            "split_every": split_every,
            "split_count": split_count,
            **split_variants[mode],
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
    expected = ", ".join(["none", "prune", "relocate", *split_variants,
                          "prune_residual_add", "prune_residual_tensor_add"])
    raise ValueError(
        f"unknown refine mode {mode!r}; expected one of: {expected}"
    )


def _refine_adds_capacity(mode: str) -> bool:
    return mode in {
        "duplicate", "fp_duplicate", "moment_preserving", "support_duplicate",
        "residual_add", "residual_tensor_add", "ranked_wave", "absgrad_wave",
        "freq_violation", "residual_add_nms",
        "residual_tensor_add_nms", "residual_add_residual_color",
        "residual_tensor_add_residual_color", "residual_add_nms_residual_color",
        "residual_tensor_add_nms_residual_color", "prune_residual_add",
        "prune_residual_tensor_add",
    }


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


def _color_solve_kwargs(mode: str) -> dict[str, Any]:
    if mode in ("none", "off"):
        return {"color_solve_every": None}
    if mode in ("every10", "cg10"):
        return {"color_solve_every": 10}
    if mode.startswith("every"):
        suffix = mode[len("every"):]
        try:
            every = int(suffix)
        except ValueError as exc:
            raise ValueError(f"cannot parse color_solve mode {mode!r}") from exc
        if every <= 0:
            raise ValueError(f"color_solve interval must be positive, got {mode!r}")
        return {"color_solve_every": every}
    raise ValueError(
        f"unknown color_solve mode {mode!r}; expected none, every10, or every<N>"
    )


def _seconds_to_target(history: dict, iters_to_target) -> float | None:
    """Wall seconds at the iteration where the target PSNR was first reached (interpolated)."""
    if iters_to_target is None:
        return None
    its, el = history.get("iter", []), history.get("elapsed", [])
    if not its:
        return None
    return float(np.interp(iters_to_target, its, el))


def _run_one(img, target, cfg, *, budget, seed, iters, render_chunk, ssim_weight,
             ssim_backend, flank_offset, max_axis_ratio, coherence_power, init_scale_mult,
             density_base, density_power, flat_frac, corner_frac, grad_sigma, tensor_sigma,
             color_radius, init_opacity, lr_decay_every, lr_decay_gamma, split_every, split_count,
             prune_every, prune_min_activity, max_gaussians, pyramid_levels,
             pyramid_fractions, pyramid_iters_per_level, compute_lpips,
             target_psnr, target_psnrs, target_ms_ssim, target_bpp, adaptive_count,
             adaptive_growth_every, adaptive_growth_count, adaptive_split_mode,
             adaptive_min_delta_psnr, adaptive_patience, log_every, verbose):
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
        ssim_backend=ssim_backend,
        compute_lpips=compute_lpips,
        pixel_loss=cfg["loss"],
        optimizer=cfg["optimizer"],
        lr_schedule=cfg["lr_schedule"],
        # only step reads it; leaking it into schedule="none" configs would silently
        # re-enable step decay through fit's backward-compat fallback
        lr_decay_every=lr_decay_every if cfg["lr_schedule"] == "step" else None,
        lr_decay_gamma=lr_decay_gamma,
        renderer=cfg["renderer"],
        aa_dilation=float(cfg["aa"]),
        color_basis=cfg["color_basis"],
        **_color_solve_kwargs(cfg["color_solve"]),
        max_gaussians=max_gaussians,
        target_psnr=target_psnr,
        target_psnrs=list(target_psnrs),
        target_ms_ssim=target_ms_ssim,
        target_bpp=target_bpp,
        adaptive_count=adaptive_count,
        adaptive_growth_every=adaptive_growth_every,
        adaptive_growth_count=adaptive_growth_count,
        adaptive_split_mode=adaptive_split_mode,
        adaptive_min_delta_psnr=adaptive_min_delta_psnr,
        adaptive_patience=adaptive_patience,
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
    split_events = history.get("split_events", [])
    ranked_events = [e for e in split_events if e.get("mode") == "ranked_wave"]
    absgrad_events = [e for e in split_events if e.get("mode") == "absgrad_wave"]
    freq_events = [e for e in split_events if e.get("mode") == "freq_violation"]
    color_solve_events = history.get("color_solve_events", [])
    adaptive_events = history.get("adaptive_events", [])

    def event_mean(events: list[dict], key: str) -> float | None:
        vals = [float(e[key]) for e in events if key in e]
        return None if not vals else round(float(np.mean(vals)), 6)

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
        "split_event_count": len(split_events),
        "ranked_wave_score_mean": event_mean(ranked_events, "score_mean"),
        "ranked_wave_residual_support_mean": event_mean(ranked_events, "residual_support_mean"),
        "ranked_wave_activity_mean": event_mean(ranked_events, "activity_mean"),
        "ranked_wave_footprint_mean": event_mean(ranked_events, "footprint_mean"),
        "absgrad_score_mean": event_mean(absgrad_events, "absgrad_score_mean"),
        "absgrad_score_max": event_mean(absgrad_events, "absgrad_score_max"),
        "freq_violation_score_mean": event_mean(freq_events, "freq_violation_score_mean"),
        "freq_violation_score_max": event_mean(freq_events, "freq_violation_score_max"),
        "freq_violation_axis0_count": event_mean(freq_events, "freq_violation_axis0_count"),
        "freq_violation_axis1_count": event_mean(freq_events, "freq_violation_axis1_count"),
        "freq_violation_freq_mean": event_mean(freq_events, "freq_violation_freq_mean"),
        "color_solve_every": fcfg.color_solve_every,
        "color_solve_lambda": fcfg.color_solve_lambda,
        "color_solve_maxiter": fcfg.color_solve_maxiter,
        "color_solve_event_count": len(color_solve_events),
        "color_solve_relative_residual_mean": event_mean(
            color_solve_events, "relative_residual"
        ),
        "relocate_event_count": len(history.get("relocate_events", [])),
        "adaptive_count": bool(adaptive_count),
        "adaptive_event_count": len(adaptive_events),
        "adaptive_growth_count": sum(1 for e in adaptive_events if e.get("action") == "grow"),
        "adaptive_stop_reason": out.get("adaptive_stop_reason"),
        "adaptive_selected_n": out.get("adaptive_selected_n"),
        "estimated_bpp": round(float(out.get("estimated_bpp", 0.0)), 4),
        "target_ms_ssim": target_ms_ssim,
        "target_bpp": target_bpp,
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
    aa_dilations=None,
    color_basis_modes=None,
    color_solve_modes=None,
    pixel_losses=None,
    optimizers=None,
    lr_schedules=None,
    refine_modes=None,
    pyramid_modes=None,
    render_chunk=512,
    ssim_weight=0.3,
    ssim_backend="builtin",
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
    target_ms_ssim: float | None = None,
    target_bpp: float | None = None,
    adaptive_count: bool = False,
    adaptive_growth_every: int = 50,
    adaptive_growth_count: int = 64,
    adaptive_split_mode: str = "residual_tensor_add",
    adaptive_min_delta_psnr: float = 0.02,
    adaptive_patience: int = 2,
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
        "aa_dilations": aa_dilations,
        "color_basis_modes": color_basis_modes,
        "color_solve_modes": color_solve_modes,
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

    write_config(str(out_path), run_config({
        "images": files, "mode": mode, "budgets": list(budgets), "seeds": list(seeds),
        "iters": iters, "max_side": max_side, "axes": {k: list(v) for k, v in axes.items()},
        "render_chunk": render_chunk, "ssim_weight": ssim_weight, "split_every": split_every,
        "ssim_backend": ssim_backend,
        "split_count": split_count, "prune_every": prune_every,
        "prune_min_activity": prune_min_activity, "max_gaussians": max_gaussians,
        "target_psnr": target_psnr, "target_ms_ssim": target_ms_ssim,
        "target_bpp": target_bpp, "adaptive_count": adaptive_count,
        "adaptive_growth_every": adaptive_growth_every,
        "adaptive_growth_count": adaptive_growth_count,
        "adaptive_split_mode": adaptive_split_mode,
        "adaptive_min_delta_psnr": adaptive_min_delta_psnr,
        "adaptive_patience": adaptive_patience, "dedupe": dedupe,
    }, device=device))
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

    jsonl_path = out_path / "stage_search.jsonl"
    jsonl_path.unlink(missing_ok=True)   # fresh incremental log for this run

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
                    # equal-budget arms (BENCH-002): cap at the cell budget, and let adding-refine
                    # arms START below budget so their planned additions land AT budget — otherwise
                    # they ran with up to +split_count more capacity than the baselines they rank
                    # against, confounding "refine wins" with extra capacity.
                    cap = budget if max_gaussians is None else max_gaussians
                    adds = _refine_adds_capacity(cfg["refine"])
                    init_budget = max(1, budget - split_count) if adds else budget
                    base_row = {
                        "image": image_name, "budget": budget, "seed": seed,
                        "init_budget": init_budget, "max_gaussians": cap,
                        **{k: cfg[k] for k in cfg if k != "label"},
                        "config_label": cfg["label"],
                        "is_baseline": cfg["label"] == baseline_label,
                    }
                    try:
                        metrics = _run_one(
                            img, target, cfg, budget=init_budget, seed=seed, iters=iters,
                            render_chunk=render_chunk, ssim_weight=ssim_weight,
                            ssim_backend=ssim_backend,
                            flank_offset=flank_offset, max_axis_ratio=max_axis_ratio,
                            coherence_power=coherence_power, init_scale_mult=init_scale_mult,
                            density_base=density_base, density_power=density_power,
                            flat_frac=flat_frac, corner_frac=corner_frac,
                            grad_sigma=grad_sigma, tensor_sigma=tensor_sigma,
                            color_radius=color_radius, init_opacity=init_opacity,
                            lr_decay_every=lr_decay_every,
                            lr_decay_gamma=lr_decay_gamma, split_every=split_every,
                            split_count=split_count, prune_every=prune_every,
                            prune_min_activity=prune_min_activity, max_gaussians=cap,
                            pyramid_levels=pyramid_levels, pyramid_fractions=pyramid_fractions,
                            pyramid_iters_per_level=pyramid_iters_per_level,
                            compute_lpips=compute_lpips, target_psnr=target_psnr,
                            target_psnrs=target_psnrs, target_ms_ssim=target_ms_ssim,
                            target_bpp=target_bpp, adaptive_count=adaptive_count,
                            adaptive_growth_every=adaptive_growth_every,
                            adaptive_growth_count=adaptive_growth_count,
                            adaptive_split_mode=adaptive_split_mode,
                            adaptive_min_delta_psnr=adaptive_min_delta_psnr,
                            adaptive_patience=adaptive_patience,
                            log_every=log_every, verbose=verbose,
                        )
                        row = {**base_row, "status": "ok", **metrics}
                    except Exception as exc:  # one broken arm must not void the whole sweep
                        row = {**base_row, "status": "error", "error": repr(exc)}
                        print(f"  ERROR in cell: {exc!r}", flush=True)
                    rows.append(row)
                    with jsonl_path.open("a", encoding="utf-8") as jf:  # crash-recoverable
                        jf.write(json.dumps({k: v for k, v in row.items()
                                             if k not in {"history", "prefix_metrics"}}) + "\n")
                    print({k: row[k] for k in row
                           if k not in {"history", "prefix_metrics"}}, flush=True)

    _write(rows, out_path, mode=mode, baseline_label=baseline_label)
    return rows


def _config_key(row):
    return tuple(row[k] for k in STAGE_KEYS) + (row["budget"],)


def _fmt(v, spec=".4f"):
    return format(v, spec) if v is not None else "-"


def summarize(rows, top_k: int = 20) -> str:
    rows = [r for r in rows if r.get("status") != "error"]  # broken cells never enter the ranking
    if not rows:
        return "# StructSplat Stage Search\n\n(no successful cells)\n"
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
    rows = [r for r in rows if r.get("status") != "error"]
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
    rows = [r for r in rows if r.get("status") != "error"]
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

    base_auc = [r["auc_psnr"] for r in base.values() if r.get("auc_psnr") is not None]
    lines = [
        "# Stage influence (paired deltas vs baseline)",
        "",
        f"Baseline: `{baseline_label}`",
        f"Baseline means: PSNR {mean(r['psnr'] for r in base.values()):.3f}, "
        f"MS-SSIM {mean(r['ms_ssim'] for r in base.values()):.5f}, "
        # mean AUC over all baseline cells, not one arbitrary cell (BENCH-002)
        f"AUC {_fmt(mean(base_auc) if base_auc else None, '.3f')}, "
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
    write_json(outdir / "stage_search.json", rows)
    if rows:
        fields = [k for k in rows[0].keys() if k not in {"history", "prefix_metrics"}]
        write_csv(
            outdir / "stage_search.csv",
            [{k: row.get(k) for k in fields} for row in rows],
            fieldnames=fields,
        )
    (outdir / "summary.md").write_text(summarize(rows), encoding="utf-8")
    wrote = "stage_search.json / stage_search.csv / summary.md"
    if mode == "influence" and baseline_label is not None:
        (outdir / "influence.md").write_text(summarize_influence(rows, baseline_label),
                                             encoding="utf-8")
        wrote += " / influence.md"
    _write_index_html(rows, outdir, mode=mode)
    wrote += " / index.html"
    print(f"\nwrote {wrote} to {outdir}")


def _write_index_html(rows: list[dict[str, Any]], outdir: Path, *, mode: str) -> None:
    ok_rows = [r for r in rows if r.get("status") != "error"]
    error_rows = [r for r in rows if r.get("status") == "error"]
    configs = {_config_label({k: r[k] for k in STAGE_KEYS}) for r in rows} if rows else set()
    images = sorted({str(r.get("image", "")) for r in rows if r.get("image")})
    budgets = sorted({r.get("budget") for r in rows if r.get("budget") is not None})
    seeds = sorted({r.get("seed") for r in rows if r.get("seed") is not None})

    links = [
        ("summary.md", "summary.md"),
        ("stage_search.csv", "stage_search.csv"),
        ("stage_search.json", "stage_search.json"),
        ("stage_search.jsonl", "stage_search.jsonl"),
        ("config.json", "config.json"),
    ]
    if (outdir / "influence.md").exists():
        links.insert(1, ("influence.md", "influence.md"))

    groups: dict[tuple, list[dict[str, Any]]] = {}
    for row in ok_rows:
        groups.setdefault(_config_key(row), []).append(row)
    ranked = []
    for key, vals in groups.items():
        psnrs = [v["psnr"] for v in vals]
        ranked.append((mean(psnrs), key, vals))
    ranked.sort(reverse=True, key=lambda x: x[0])

    def row_mean(vals: list[dict[str, Any]], key: str) -> float | None:
        xs = [v[key] for v in vals if v.get(key) is not None]
        return mean(xs) if xs else None

    def fmt(value: Any, spec: str = ".4f") -> str:
        if value is None:
            return "-"
        if isinstance(value, (int, np.integer)):
            return str(int(value))
        if isinstance(value, (float, np.floating)):
            return format(float(value), spec)
        return escape(str(value))

    html = [
        "<!doctype html>",
        '<html lang="en">',
        "<head>",
        '<meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        "<title>StructSplat Stage Search</title>",
        "<style>",
        "body{font-family:system-ui,-apple-system,Segoe UI,sans-serif;margin:24px;line-height:1.35;color:#171717;background:#fafafa}",
        "h1,h2{margin:0.9rem 0 0.55rem}",
        ".meta{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:8px;margin:16px 0}",
        ".card{border:1px solid #d4d4d4;background:#fff;padding:10px;border-radius:6px}",
        ".label{font-size:12px;color:#525252;text-transform:uppercase;letter-spacing:.03em}",
        ".value{font-size:20px;font-weight:650}",
        "table{border-collapse:collapse;width:100%;background:#fff;margin:10px 0 22px}",
        "th,td{border:1px solid #d4d4d4;padding:6px 8px;text-align:left;vertical-align:top}",
        "th{background:#f0f0f0}",
        "td.num{text-align:right;font-variant-numeric:tabular-nums}",
        "code{font-size:12px;word-break:break-word}",
        ".links a{margin-right:14px}",
        ".note{color:#525252}",
        "</style>",
        "</head>",
        "<body>",
        "<h1>StructSplat Stage Search</h1>",
        f'<p class="note">Mode: <code>{escape(mode)}</code>. Scalar benchmark overview; this harness does not emit reconstructions or plots.</p>',
        '<p class="links">' + " ".join(
            f'<a href="{escape(href)}">{escape(label)}</a>'
            for label, href in links if (outdir / href).exists()
        ) + "</p>",
        '<div class="meta">',
        f'<div class="card"><div class="label">Cells</div><div class="value">{len(rows)}</div></div>',
        f'<div class="card"><div class="label">OK</div><div class="value">{len(ok_rows)}</div></div>',
        f'<div class="card"><div class="label">Errors</div><div class="value">{len(error_rows)}</div></div>',
        f'<div class="card"><div class="label">Configs</div><div class="value">{len(configs)}</div></div>',
        f'<div class="card"><div class="label">Images</div><div class="value">{len(images)}</div></div>',
        f'<div class="card"><div class="label">Budgets</div><div class="value">{len(budgets)}</div></div>',
        f'<div class="card"><div class="label">Seeds</div><div class="value">{len(seeds)}</div></div>',
        "</div>",
        "<h2>Top Configs</h2>",
        "<table><thead><tr><th>Rank</th><th>Budget</th><th>Mean PSNR</th><th>Mean MS-SSIM</th><th>Mean AUC</th><th>Mean fit s</th><th>Config</th></tr></thead><tbody>",
    ]
    for rank, (_score, _key, vals) in enumerate(ranked[:20], 1):
        label = vals[0]["config_label"]
        html.append(
            "<tr>"
            f'<td class="num">{rank}</td>'
            f'<td class="num">{fmt(vals[0].get("budget"), ".0f")}</td>'
            f'<td class="num">{fmt(row_mean(vals, "psnr"))}</td>'
            f'<td class="num">{fmt(row_mean(vals, "ms_ssim"), ".5f")}</td>'
            f'<td class="num">{fmt(row_mean(vals, "auc_psnr"), ".3f")}</td>'
            f'<td class="num">{fmt(row_mean(vals, "fit_seconds"), ".3f")}</td>'
            f"<td><code>{escape(label)}</code></td>"
            "</tr>"
        )
    if not ranked:
        html.append('<tr><td colspan="7">(no successful cells)</td></tr>')
    html.append("</tbody></table>")

    stage_rows = []
    for stage in STAGE_KEYS:
        levels = sorted({r[stage] for r in ok_rows}) if ok_rows else []
        if len(levels) < 2:
            continue
        for level in levels:
            vals = [r for r in ok_rows if r[stage] == level]
            stage_rows.append((stage, level, vals))
    if stage_rows:
        html += [
            "<h2>Per-Stage Marginals</h2>",
            "<table><thead><tr><th>Stage</th><th>Level</th><th>Runs</th><th>PSNR</th><th>MS-SSIM</th><th>AUC</th><th>Fit s</th></tr></thead><tbody>",
        ]
        for stage, level, vals in stage_rows:
            psnrs = [v["psnr"] for v in vals]
            html.append(
                "<tr>"
                f"<td>{escape(str(stage))}</td>"
                f"<td><code>{escape(str(level))}</code></td>"
                f'<td class="num">{len(vals)}</td>'
                f'<td class="num">{mean(psnrs):.3f} +/- {pstdev(psnrs):.3f}</td>'
                f'<td class="num">{fmt(row_mean(vals, "ms_ssim"), ".5f")}</td>'
                f'<td class="num">{fmt(row_mean(vals, "auc_psnr"), ".3f")}</td>'
                f'<td class="num">{fmt(row_mean(vals, "fit_seconds"), ".3f")}</td>'
                "</tr>"
            )
        html.append("</tbody></table>")

    if error_rows:
        html += [
            "<h2>Error Cells</h2>",
            "<table><thead><tr><th>Image</th><th>Budget</th><th>Seed</th><th>Config</th><th>Error</th></tr></thead><tbody>",
        ]
        for row in error_rows[:50]:
            html.append(
                "<tr>"
                f"<td>{escape(str(row.get('image', '-')))}</td>"
                f'<td class="num">{fmt(row.get("budget"), ".0f")}</td>'
                f'<td class="num">{fmt(row.get("seed"), ".0f")}</td>'
                f"<td><code>{escape(str(row.get('config_label', '-')))}</code></td>"
                f"<td>{escape(str(row.get('error', '-')))}</td>"
                "</tr>"
            )
        html.append("</tbody></table>")

    html += ["</body>", "</html>"]
    (outdir / "index.html").write_text("\n".join(html) + "\n", encoding="utf-8")


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
    p.add_argument("--aa-dilations", type=float, nargs="+", default=None)
    p.add_argument("--color-basis-modes", nargs="+", default=None,
                   help="constant or affine")
    p.add_argument("--color-solve-modes", nargs="+", default=None,
                   help="none, every10, or every<N>; normalized renderer only")
    p.add_argument("--pixel-losses", nargs="+", default=None)
    p.add_argument("--optimizers", nargs="+", default=None)
    p.add_argument("--lr-schedules", nargs="+", default=None)
    p.add_argument("--refine-modes", nargs="+", default=None)
    p.add_argument("--pyramid-modes", nargs="+", default=None)
    p.add_argument("--chunk", type=int, default=512)
    p.add_argument("--ssim-weight", type=float, default=0.3)
    p.add_argument("--ssim-backend", choices=["builtin", "fused", "auto"], default="builtin")
    p.add_argument("--target-psnr", type=float, default=None,
                   help="record iters/seconds-to-target for convergence-rate comparisons")
    p.add_argument("--target-psnrs", type=float, nargs="*", default=[])
    p.add_argument("--target-ms-ssim", type=float, default=None)
    p.add_argument("--target-bpp", type=float, default=None)
    p.add_argument("--adaptive-count", action="store_true",
                   help="enable FIT-008 self-adaptive Gaussian count controller")
    p.add_argument("--adaptive-growth-every", type=int, default=50)
    p.add_argument("--adaptive-growth-count", type=int, default=64)
    p.add_argument("--adaptive-split-mode",
                   choices=[
                       "residual_add", "residual_tensor_add", "ranked_wave",
                       "freq_violation", "fp_duplicate", "moment_preserving",
                       "support_duplicate",
                   ],
                   default="residual_tensor_add")
    p.add_argument("--adaptive-min-delta-psnr", type=float, default=0.02)
    p.add_argument("--adaptive-patience", type=int, default=2)
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
        renderers=a.renderers, aa_dilations=a.aa_dilations,
        color_basis_modes=a.color_basis_modes,
        color_solve_modes=a.color_solve_modes,
        pixel_losses=a.pixel_losses, optimizers=a.optimizers,
        lr_schedules=a.lr_schedules, refine_modes=a.refine_modes,
        pyramid_modes=a.pyramid_modes, render_chunk=a.chunk,
        ssim_weight=a.ssim_weight, ssim_backend=a.ssim_backend,
        split_every=a.split_every, split_count=a.split_count,
        prune_every=a.prune_every, prune_min_activity=a.prune_min_activity,
        max_gaussians=a.max_gaussians, pyramid_levels=a.pyramid_levels,
        pyramid_fractions=a.pyramid_fractions, pyramid_iters_per_level=a.pyramid_iters_per_level,
        compute_lpips=a.lpips, target_psnr=a.target_psnr, target_psnrs=a.target_psnrs,
        target_ms_ssim=a.target_ms_ssim, target_bpp=a.target_bpp,
        adaptive_count=a.adaptive_count,
        adaptive_growth_every=a.adaptive_growth_every,
        adaptive_growth_count=a.adaptive_growth_count,
        adaptive_split_mode=a.adaptive_split_mode,
        adaptive_min_delta_psnr=a.adaptive_min_delta_psnr,
        adaptive_patience=a.adaptive_patience,
        log_every=a.log_every, dedupe=not a.no_dedupe,
        max_configs=a.max_configs, shuffle_configs=a.shuffle_configs,
        config_seed=a.config_seed, outdir=a.outdir, device=a.device, verbose=a.verbose,
    )


if __name__ == "__main__":
    main()
