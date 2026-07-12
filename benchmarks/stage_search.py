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
import math
import os
import time
from html import escape
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import numpy as np
import torch

from benchmarks.common import load_image as _load_image
from benchmarks.common import HEADLINE_TARGET_PSNRS, headline_target_psnrs
from benchmarks.common import psnr_auc as _psnr_auc
from benchmarks.common import run_config, write_config, write_csv, write_json
from benchmarks.common import target_hit_stats, target_iters, target_label
from structsplat.config import (
    DEFAULT_INIT_STRATEGY,
    FitConfig,
    InitConfig,
    PyramidConfig,
    StructureTensorConfig,
    refine_alias_to_axes,
    refine_axes_to_alias,
)

# stage axes, in label order; values = the swappable options each stage exposes
STAGE_KEYS = [
    "strategy", "tensor", "tensor_color", "density", "sampling", "orientation", "color",
    "scale", "scale_cap", "background", "opacity", "renderer", "aa", "color_basis",
    "color_solve", "loss", "loss_weight", "optimizer", "lr_schedule", "refine_site",
    "refine_primitive", "refine_nms", "refine_color", "refine_score", "refine_prune",
    "refine_relocate",
    "state_seed", "row_temper", "support_fade", "pyramid",
]

FACTORIAL_DEFAULTS: dict[str, tuple[str, ...]] = {
    "strategies": (DEFAULT_INIT_STRATEGY,),
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
    "background_modes": ("off",),
    "opacity_modes": ("none",),
    "renderers": ("normalized",),
    "aa_dilations": (0.0,),
    "color_basis_modes": ("constant",),
    "color_solve_modes": ("none",),
    "pixel_losses": ("l1", "charbonnier"),
    "loss_weight_modes": ("none",),
    "optimizers": ("adam",),
    "lr_schedules": ("none", "cosine"),
    "refine_sites": ("none", "residual"),
    "refine_primitives": ("sampled_add",),
    "refine_nms_modes": ("off",),
    "refine_color_inits": ("target",),
    "refine_score_modes": ("legacy_abs",),
    "refine_prune_modes": ("off",),
    "refine_relocate_modes": ("off",),
    "state_seed_modes": ("off",),
    "row_temper_modes": ("off",),
    "support_fade_modes": ("off",),
    "pyramid_modes": ("single",),
}

# influence mode: FIRST value per axis = the baseline (the shipped ADR-0013 defaults), rest = the
# variants. Every first value must equal config.py's default for that stage (BENCH-002).
INFLUENCE_DEFAULTS: dict[str, tuple[str, ...]] = {
    "strategies": (
        DEFAULT_INIT_STRATEGY, "aniso_onedge", "quadtree_hybrid", "quadtree_aggregate",
        "aniso_flanking", "iso_blue_noise", "grid", "random"
    ),
    "tensor_operators": ("central", "sobel", "scharr"),
    "tensor_colors": ("luma", "rgb"),
    "density_modes": ("structure", "gradient", "variance", "hybrid", "uniform"),
    "sampling_modes": ("wse", "floyd_steinberg", "dart_throwing", "halton", "cvt",
                       "farthest_point", "density_random", "jittered_grid"),
    "orientation_modes": ("tensor", "random", "zero"),
    "color_modes": ("bilinear", "local_mean", "two_sided"),
    "scale_modes": ("spacing", "uniform", "knn"),
    "scale_cap_modes": ("none", "feature12", "feature_rel", "hard8"),
    "background_modes": ("off", "frac0.05_grid8", "frac0.10_grid16"),
    "opacity_modes": ("none", "constant"),
    "renderers": (
        "normalized", "additive", "cuda", "cuda_additive",
        "cuda_tiled", "cuda_tiled_additive",
    ),
    "aa_dilations": (0.0, 0.3),
    "color_basis_modes": ("constant", "affine"),
    "color_solve_modes": ("none", "every10"),
    "pixel_losses": ("l1", "l2", "charbonnier"),
    "loss_weight_modes": ("none", "tensor"),
    "optimizers": ("adam", "adamw", "adan"),
    "lr_schedules": ("none", "cosine", "step"),
    "refine_sites": (
        "none", "residual", "residual_tensor", "support", "ranked", "absgrad",
        "freq_violation",
    ),
    "refine_primitives": ("fp", "duplicate", "moment_preserving", "sampled_add"),
    "refine_nms_modes": ("off", "on"),
    "refine_color_inits": ("target", "residual"),
    "refine_score_modes": ("legacy_abs", "gaussian_abs", "signed_gaussian"),
    "refine_prune_modes": ("off", "on"),
    "refine_relocate_modes": ("off", "on"),
    "state_seed_modes": ("off", "on"),
    "row_temper_modes": ("off", "warmup5"),
    "support_fade_modes": ("off",),
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
    "background_modes": "background",
    "aa_dilations": "aa", "color_basis_modes": "color_basis",
    "color_solve_modes": "color_solve", "pixel_losses": "loss",
    "loss_weight_modes": "loss_weight",
    "optimizers": "optimizer", "lr_schedules": "lr_schedule",
    "refine_modes": "refine",
    "refine_sites": "refine_site", "refine_primitives": "refine_primitive",
    "refine_nms_modes": "refine_nms", "refine_color_inits": "refine_color",
    "refine_score_modes": "refine_score",
    "refine_prune_modes": "refine_prune", "refine_relocate_modes": "refine_relocate",
    "state_seed_modes": "state_seed", "row_temper_modes": "row_temper",
    "support_fade_modes": "support_fade",
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


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _cell_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("image"),
        int(row.get("budget")),
        int(row.get("seed")),
        row.get("config_label"),
    )


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
    canonical = _normalize_refine_config(canonical)
    c = _normalize_refine_config(cfg)
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
    if c.get("refine_site") == "none":
        for k in ("refine_primitive", "refine_nms", "refine_color", "refine_score"):
            c[k] = canonical[k]
    if c.get("refine_primitive") != "sampled_add":
        # NMS and residual-color initialization are sampled-add controls; duplicate-style
        # primitives ignore them, so pin them to avoid duplicate equivalent cells.
        c["refine_nms"] = canonical["refine_nms"]
        c["refine_color"] = canonical["refine_color"]
        c["refine_score"] = canonical["refine_score"]
    if c.get("refine_site") == "none" and c.get("refine_relocate") != "on":
        c["state_seed"] = canonical["state_seed"]
        c["row_temper"] = canonical["row_temper"]
    c = _normalize_refine_config(c)
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


LEGACY_REFINE_MODES = (
    "none", "prune", "duplicate", "fp_duplicate", "moment_preserving",
    "support_duplicate", "residual_add", "residual_tensor_add", "ranked_wave",
    "absgrad_wave", "freq_violation", "relocate", "residual_add_nms",
    "residual_tensor_add_nms", "residual_add_residual_color",
    "residual_tensor_add_residual_color", "residual_add_nms_residual_color",
    "residual_tensor_add_nms_residual_color", "prune_residual_add",
    "prune_residual_tensor_add",
)


def _base_refine_config() -> dict[str, Any]:
    return {
        "refine": "none",
        "refine_site": "none",
        "refine_primitive": "duplicate",
        "refine_nms": "off",
        "refine_color": "target",
        "refine_score": "legacy_abs",
        "refine_prune": "off",
        "refine_relocate": "off",
    }


def _legacy_refine_config(mode: str) -> dict[str, Any]:
    c = _base_refine_config()
    c["refine"] = mode
    if mode == "none":
        return c
    if mode == "prune":
        c["refine_prune"] = "on"
        return c
    if mode == "relocate":
        c["refine_relocate"] = "on"
        return c
    core = mode
    if core.startswith("prune_"):
        c["refine_prune"] = "on"
        core = core[len("prune_"):]
    if core.endswith("_residual_color"):
        c["refine_color"] = "residual"
        core = core[:-len("_residual_color")]
    site, primitive, nms = refine_alias_to_axes(core)
    c.update({
        "refine_site": site,
        "refine_primitive": primitive,
        "refine_nms": nms,
    })
    return c


def _refine_label(cfg: dict[str, Any]) -> str:
    site = cfg["refine_site"]
    primitive = cfg["refine_primitive"]
    nms = cfg["refine_nms"]
    if site == "none":
        label = "none"
    else:
        label = refine_axes_to_alias(site, primitive, nms)
    if cfg.get("refine_color") == "residual" and site != "none" and primitive == "sampled_add":
        label += "_residual_color"
    if cfg.get("refine_prune") == "on":
        label = "prune" if label == "none" else f"prune_{label}"
    if cfg.get("refine_relocate") == "on":
        label = "relocate" if label == "none" else f"{label}_relocate"
    return label


def _normalize_refine_config(cfg: dict[str, Any]) -> dict[str, Any]:
    c = dict(cfg)
    if "refine" in c and not all(k in c for k in (
        "refine_site", "refine_primitive", "refine_nms", "refine_color",
        "refine_score", "refine_prune", "refine_relocate",
    )):
        c.update(_legacy_refine_config(c["refine"]))
    else:
        c.setdefault("refine_site", "none")
        c.setdefault("refine_primitive", "duplicate")
        c.setdefault("refine_nms", "off")
        c.setdefault("refine_color", "target")
        c.setdefault("refine_score", "legacy_abs")
        c.setdefault("refine_prune", "off")
        c.setdefault("refine_relocate", "off")
    c.setdefault("state_seed", "off")
    c.setdefault("row_temper", "off")
    c.setdefault("support_fade", "off")
    c.setdefault("loss_weight", "none")
    c.setdefault("background", "off")
    c["refine"] = _refine_label(c)
    return c


def _refine_kwargs_from_config(cfg: dict[str, Any], split_every: int | None,
                               split_count: int, prune_every: int | None,
                               prune_min_activity: float) -> dict[str, Any]:
    refine_cfg = _normalize_refine_config(cfg)
    out: dict[str, Any] = {}
    if refine_cfg["refine_prune"] == "on":
        out.update({
            "prune_every": prune_every,
            "prune_min_activity": prune_min_activity,
        })
    if refine_cfg["refine_relocate"] == "on":
        out.update({
            "relocate_every": split_every,
            "relocate_count": split_count,
        })
    if refine_cfg["refine_site"] != "none":
        out.update({
            "split_every": split_every,
            "split_count": split_count,
            "split_mode": refine_axes_to_alias(
                refine_cfg["refine_site"],
                refine_cfg["refine_primitive"],
                "off",
            ),
            "refine_site": refine_cfg["refine_site"],
            "refine_primitive": refine_cfg["refine_primitive"],
            "refine_nms": refine_cfg["refine_nms"],
            "split_color_init": refine_cfg["refine_color"],
            "sampled_add_score": refine_cfg["refine_score"],
        })
        if refine_cfg["refine_nms"] == "on":
            out.update({"split_min_spacing": 1.0, "split_oversample": 8.0})
    return out


def _refine_kwargs(mode: str, split_every: int | None, split_count: int,
                   prune_every: int | None, prune_min_activity: float) -> dict[str, Any]:
    if mode not in LEGACY_REFINE_MODES:
        expected = ", ".join(LEGACY_REFINE_MODES)
        raise ValueError(f"unknown refine mode {mode!r}; expected one of: {expected}")
    out = _refine_kwargs_from_config(
        _legacy_refine_config(mode), split_every, split_count,
        prune_every, prune_min_activity,
    )
    for key in ("refine_site", "refine_primitive", "refine_nms"):
        out.pop(key, None)
    if out.get("sampled_add_score") == "legacy_abs":
        out.pop("sampled_add_score")
    if out.get("split_color_init") == "target":
        out.pop("split_color_init", None)
    return out


def _refine_adds_capacity(refine: str | dict[str, Any]) -> bool:
    cfg = _legacy_refine_config(refine) if isinstance(refine, str) else _normalize_refine_config(refine)
    return cfg["refine_site"] != "none"


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
    if mode in ("feature_rel", "feature_relative", "rel_feature"):
        return {"scale_cap_mode": "feature_rel", "scale_cap_max": None}
    for prefix, cap_mode in (("feature_cap", "feature"), ("feature", "feature"), ("hard", "hard")):
        if mode.startswith(prefix):
            suffix = mode[len(prefix):].lstrip("_")
            try:
                cap = float(suffix)
            except ValueError as exc:
                raise ValueError(f"cannot parse scale cap mode {mode!r}") from exc
            return {"scale_cap_mode": cap_mode, "scale_cap_max": cap}
    raise ValueError(
        f"unknown scale_cap mode {mode!r}; expected none, hard8, hard12, feature8, "
        "feature12, or feature_rel"
    )


def _color_solve_kwargs(mode: str) -> dict[str, Any]:
    if mode in ("none", "off"):
        return {"color_solve_every": None, "color_solve_schedule": "none"}
    tokens = mode.split("+")
    schedule: list[str] = []
    every: int | None = None
    for token in tokens:
        if token == "cg10":
            token = "every10"
        if token.startswith("every"):
            suffix = token[len("every"):]
            try:
                every = int(suffix)
            except ValueError as exc:
                raise ValueError(f"cannot parse color_solve mode {mode!r}") from exc
            if every <= 0:
                raise ValueError(f"color_solve interval must be positive, got {mode!r}")
            schedule.append("every")
        elif token in ("init", "final", "on_split"):
            schedule.append(token)
        else:
            raise ValueError(
                f"unknown color_solve mode {mode!r}; expected none, every<N>, init, "
                "final, on_split, or a + composition"
            )
    # Preserve the user's trigger order while removing duplicates.
    deduped = list(dict.fromkeys(schedule))
    return {
        "color_solve_every": every,
        "color_solve_schedule": "+".join(deduped) if deduped else "none",
    }


def _state_seed_kwargs(mode: str) -> dict[str, Any]:
    if mode in ("off", "none", "false", "0"):
        return {"seed_new_row_optimizer_state": False}
    if mode in ("on", "seed", "parent", "true", "1"):
        return {"seed_new_row_optimizer_state": True}
    raise ValueError(f"unknown state_seed mode {mode!r}; expected off or on")


def _row_temper_kwargs(mode: str) -> dict[str, Any]:
    if mode in ("off", "none", "false", "0"):
        return {"new_row_temper_iters": 0}
    for prefix in ("warmup", "ramp", "temper"):
        if mode.startswith(prefix):
            suffix = mode[len(prefix):].lstrip("_")
            try:
                iters = int(suffix)
            except ValueError as exc:
                raise ValueError(f"cannot parse row_temper mode {mode!r}") from exc
            if iters <= 0:
                raise ValueError(f"row_temper warmup must be positive, got {mode!r}")
            return {"new_row_temper_iters": iters}
    raise ValueError(f"unknown row_temper mode {mode!r}; expected off or warmup<N>")


def _support_fade_kwargs(mode: str) -> dict[str, Any]:
    if mode in ("off", "none", "false", "0"):
        return {"support_fade": False, "support_fade_until_frac": None}
    if mode in ("on", "full", "true", "1"):
        return {"support_fade": True, "support_fade_until_frac": None}
    for prefix in ("until", "early", "schedule"):
        if mode.startswith(prefix):
            suffix = mode[len(prefix):].lstrip("_")
            try:
                frac = float(suffix)
            except ValueError as exc:
                raise ValueError(f"cannot parse support_fade mode {mode!r}") from exc
            if not 0.0 <= frac <= 1.0:
                raise ValueError(f"support_fade schedule fraction must be in [0, 1], got {mode!r}")
            return {"support_fade": False, "support_fade_until_frac": frac}
    raise ValueError(f"unknown support_fade mode {mode!r}; expected off, on, or until<F>")


def _loss_weight_kwargs(mode: str) -> dict[str, Any]:
    if mode in ("off", "none", "false", "0"):
        return {"loss_weighting": "none"}
    if mode in ("tensor", "edge", "structure", "on", "true", "1"):
        return {"loss_weighting": "tensor", "loss_weight_beta": 1.0}
    for prefix in ("tensor", "edge", "structure"):
        if mode.startswith(prefix):
            suffix = mode[len(prefix):].lstrip("_")
            try:
                beta = float(suffix)
            except ValueError as exc:
                raise ValueError(f"cannot parse loss_weight mode {mode!r}") from exc
            if beta < 0.0:
                raise ValueError(f"loss_weight beta must be >= 0, got {mode!r}")
            return {"loss_weighting": "tensor", "loss_weight_beta": beta}
    raise ValueError(f"unknown loss_weight mode {mode!r}; expected none or tensor[_beta]")


def _background_kwargs(mode: str) -> dict[str, Any]:
    if mode in ("off", "none", "false", "0"):
        return {"background_fraction": 0.0, "background_grid": 0}
    frac = None
    grid = None
    for token in mode.replace("-", "_").split("_"):
        if token.startswith("frac"):
            suffix = token[len("frac"):]
            try:
                frac = float(suffix)
            except ValueError as exc:
                raise ValueError(f"cannot parse background mode {mode!r}") from exc
        elif token.startswith("f") and len(token) > 1:
            try:
                frac = float(token[1:])
            except ValueError as exc:
                raise ValueError(f"cannot parse background mode {mode!r}") from exc
        elif token.startswith("grid"):
            suffix = token[len("grid"):]
            try:
                grid = int(suffix)
            except ValueError as exc:
                raise ValueError(f"cannot parse background mode {mode!r}") from exc
        elif token.startswith("g") and len(token) > 1:
            try:
                grid = int(token[1:])
            except ValueError as exc:
                raise ValueError(f"cannot parse background mode {mode!r}") from exc
    if frac is None or grid is None:
        raise ValueError(
            f"unknown background mode {mode!r}; expected off or frac<F>_grid<N>")
    return {"background_fraction": frac, "background_grid": grid}


def _split_recovery_stats(history: dict[str, Any]) -> dict[str, float | None]:
    events = history.get("split_events", [])
    its = history.get("iter", [])
    psnrs = history.get("psnr", [])
    if not events or not its or not psnrs:
        return {"post_split_delta_mean": None, "split_recovery_iters_mean": None}
    deltas: list[float] = []
    recoveries: list[float] = []
    for event in events:
        eiter = event.get("iter")
        if eiter is None:
            continue
        before = [(it, p) for it, p in zip(its, psnrs, strict=False) if it <= eiter]
        after = [(it, p) for it, p in zip(its, psnrs, strict=False) if it > eiter]
        if not before or not after:
            continue
        base_iter, base_psnr = before[-1]
        post_iter, post_psnr = after[0]
        deltas.append(float(post_psnr) - float(base_psnr))
        recovered = next((it for it, p in after if p >= base_psnr), None)
        if recovered is not None:
            recoveries.append(float(recovered) - float(base_iter))
        elif post_iter is not None:
            recoveries.append(float(its[-1]) - float(base_iter))
    return {
        "post_split_delta_mean": None if not deltas else round(float(np.mean(deltas)), 6),
        "split_recovery_iters_mean": (
            None if not recoveries else round(float(np.mean(recoveries)), 6)
        ),
    }


def _seconds_to_target(history: dict, iters_to_target) -> float | None:
    """Wall seconds at the iteration where the target PSNR was first reached (interpolated)."""
    if iters_to_target is None:
        return None
    its, el = history.get("iter", []), history.get("elapsed", [])
    if not its:
        return None
    return float(np.interp(iters_to_target, its, el))


def _edge_mae(render, target) -> float:
    render = render.detach().clamp(0, 1)
    target = target.detach().clamp(0, 1)
    luma = target.new_tensor([0.2126, 0.7152, 0.0722])
    err = ((render - target).abs() * luma).sum(dim=2)
    gray = (target * luma).sum(dim=2)
    gx = gray.new_zeros(gray.shape)
    gy = gray.new_zeros(gray.shape)
    gx[:, 1:-1] = 0.5 * (gray[:, 2:] - gray[:, :-2])
    gx[:, 0] = gray[:, 1] - gray[:, 0] if gray.shape[1] > 1 else 0.0
    gx[:, -1] = gray[:, -1] - gray[:, -2] if gray.shape[1] > 1 else 0.0
    gy[1:-1, :] = 0.5 * (gray[2:, :] - gray[:-2, :])
    gy[0, :] = gray[1, :] - gray[0, :] if gray.shape[0] > 1 else 0.0
    gy[-1, :] = gray[-1, :] - gray[-2, :] if gray.shape[0] > 1 else 0.0
    edge = torch.sqrt(gx * gx + gy * gy)
    weight = 0.1 + edge / edge.mean().clamp_min(1e-6)
    return float((err * weight).mean().detach().cpu())


def _large_support_stats(field, cfg: FitConfig, H: int, W: int) -> dict[str, float | int]:
    with torch.no_grad():
        radii = field.radii(cfg.sigma_cutoff, cfg.aa_dilation).to(device=field.means.device)
        area = ((2 * radii[:, 0] + 1) * (2 * radii[:, 1] + 1)).to(torch.float32)
        large = area > (0.25 * float(H * W))
        bg = getattr(field, "background_mask", None)
        if bg is None:
            bg = torch.zeros(field.n, device=field.means.device, dtype=torch.bool)
        else:
            bg = bg.to(device=field.means.device, dtype=torch.bool)
        detail = ~bg
        count = int(large.sum().detach().cpu())
        detail_count = int((large & detail).sum().detach().cpu())
        n_detail = max(1, int(detail.sum().detach().cpu()))
        return {
            "large_support_count": count,
            "large_support_fraction": round(count / max(1, int(field.n)), 6),
            "large_support_detail_count": detail_count,
            "large_support_detail_fraction": round(detail_count / n_detail, 6),
        }


def _run_one(img, target, cfg, *, budget, seed, iters, render_chunk, ssim_weight,
             ssim_backend, flank_offset, max_axis_ratio, coherence_power, init_scale_mult,
             density_base, density_power, flat_frac, corner_frac, grad_sigma, tensor_sigma,
             color_radius, init_opacity, lr_decay_every, lr_decay_gamma, split_every, split_count,
             prune_every, prune_min_activity, max_gaussians, pyramid_levels,
             pyramid_fractions, pyramid_iters_per_level, pyramid_level_iters, compute_lpips,
             target_psnr, target_psnrs, target_ms_ssim, target_bpp, adaptive_count,
             adaptive_growth_every, adaptive_growth_count, adaptive_split_mode,
             adaptive_min_delta_psnr, adaptive_patience, early_stop_patience,
             early_stop_min_delta, early_stop_min_iters, log_every, verbose):
    from structsplat import init as _init
    from structsplat import structure_tensor as _st
    from structsplat.fit import fit
    from structsplat.pyramid import fit_pyramid

    H, W = img.shape[:2]
    scfg = StructureTensorConfig(
        grad_sigma=grad_sigma,
        tensor_sigma=tensor_sigma,
        gradient_operator=cfg["tensor"],
        color_space=cfg["tensor_color"],
        flat_frac=flat_frac,
        corner_frac=corner_frac,
    )
    loss_tensor = None
    if cfg["loss_weight"] not in ("none", "off"):
        loss_tensor = _st.compute(img, scfg)
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
        **_background_kwargs(cfg["background"]),
        init_scale_mult=init_scale_mult,
        flank_offset_frac=flank_offset,
        color_mode=cfg["color"],
        color_radius=color_radius,
        opacity_mode=cfg["opacity"],
        init_opacity=init_opacity,
        seed=seed,
    )
    refine = _refine_kwargs_from_config(
        cfg, split_every, split_count, prune_every, prune_min_activity
    )
    fcfg = FitConfig(
        iters=iters,
        render_chunk=render_chunk,
        ssim_weight=ssim_weight,
        ssim_backend=ssim_backend,
        compute_lpips=compute_lpips,
        pixel_loss=cfg["loss"],
        **_loss_weight_kwargs(cfg["loss_weight"]),
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
        **_state_seed_kwargs(cfg["state_seed"]),
        **_row_temper_kwargs(cfg["row_temper"]),
        **_support_fade_kwargs(cfg["support_fade"]),
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
        early_stop_patience=early_stop_patience,
        early_stop_min_delta=early_stop_min_delta,
        early_stop_min_iters=early_stop_min_iters,
        log_every=log_every,
        **refine,
    )

    start = time.time()
    if cfg["pyramid"] == "single":
        field = _init.build_field(
            img, icfg, scfg, tensor=loss_tensor, device=target.device
        )
        init_seconds = time.time() - start
        out = fit(
            field, target, fcfg, verbose=verbose,
            loss_weight_map=None if loss_tensor is None else loss_tensor.energy,
        )
        elapsed = time.time() - start
    elif cfg["pyramid"] == "pyramid":
        init_seconds = 0.0
        pcfg = PyramidConfig(
            levels=pyramid_levels,
            level_fractions=list(pyramid_fractions),
            iters_per_level=pyramid_iters_per_level,
            level_iters=None if pyramid_level_iters is None else list(pyramid_level_iters),
        )
        out = fit_pyramid(
            img, target, icfg, fcfg, pcfg, scfg, verbose=verbose,
            loss_weight_map=None if loss_tensor is None else loss_tensor.energy,
        )
        elapsed = time.time() - start
    else:
        raise ValueError(f"unknown pyramid mode {cfg['pyramid']!r}; expected single or pyramid")

    history = out.get("history", {})
    iters_to_target = out.get("iters_to_target")
    fit_seconds = float(out.get("fit_seconds", 0.0))
    split_events = history.get("split_events", [])
    ranked_events = [
        e for e in split_events
        if e.get("mode") == "ranked_wave" or e.get("refine_site") == "ranked"
    ]
    absgrad_events = [
        e for e in split_events
        if e.get("mode") == "absgrad_wave" or e.get("refine_site") == "absgrad"
    ]
    freq_events = [
        e for e in split_events
        if e.get("mode") == "freq_violation" or e.get("refine_site") == "freq_violation"
    ]
    color_solve_events = history.get("color_solve_events", [])
    adaptive_events = history.get("adaptive_events", [])
    split_recovery = _split_recovery_stats(history)

    def event_mean(events: list[dict], key: str) -> float | None:
        vals = [float(e[key]) for e in events if key in e]
        return None if not vals else round(float(np.mean(vals)), 6)

    if cfg["pyramid"] == "pyramid":
        # fit_pyramid aggregates per-level fit time; the rest of the wall clock is the
        # interleaved init/density/tensor work, i.e. this mode's init cost
        init_seconds = max(elapsed - fit_seconds, 0.0)
    background_count = int(out.get("background_count", out["field"].background_count))
    detail_count = int(out.get("detail_count", out["field"].n - background_count))
    return {
        "psnr": round(float(out["psnr"]), 4),
        "ssim": round(float(out["ssim"]), 5),
        "ms_ssim": round(float(out["ms_ssim"]), 5),
        "lpips": out.get("lpips"),
        "edge_mae": round(_edge_mae(out["render"], target), 6),
        "auc_psnr": _psnr_auc(
            history,
            nominal_iters=iters if early_stop_patience is not None else None,
        ),
        "auc_psnr_horizon": (
            "nominal_hold_last" if early_stop_patience is not None else "observed"
        ),
        "iters_to_target": iters_to_target,
        "iters_to_targets": out.get("iters_to_targets", {}),
        "seconds_to_target": _seconds_to_target(history, iters_to_target),
        "n_gaussians": int(out["n_gaussians"]),
        "background_fraction": float(icfg.background_fraction),
        "background_grid": int(icfg.background_grid),
        "background_count": background_count,
        "detail_count": detail_count,
        "background_actual_fraction": round(background_count / max(1, int(out["n_gaussians"])), 6),
        **_large_support_stats(out["field"], fcfg, H, W),
        "iterations_run": int(out.get("iterations_run", iters)),
        "stopped_early": bool(out.get("stopped_early", False)),
        "stopped_at": out.get("stopped_at"),
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
        "color_solve_schedule": fcfg.color_solve_schedule,
        "color_solve_lambda": fcfg.color_solve_lambda,
        "color_solve_maxiter": fcfg.color_solve_maxiter,
        "color_solve_event_count": len(color_solve_events),
        "color_solve_relative_residual_mean": event_mean(
            color_solve_events, "relative_residual"
        ),
        "loss_weighting": fcfg.loss_weighting,
        "loss_weight_beta": float(fcfg.loss_weight_beta),
        "loss_weight_mean": out.get("loss_weight_mean"),
        "loss_weight_max": out.get("loss_weight_max"),
        "relocate_event_count": len(history.get("relocate_events", [])),
        "seed_new_row_optimizer_state": bool(fcfg.seed_new_row_optimizer_state),
        "new_row_temper_iters": int(fcfg.new_row_temper_iters),
        "new_row_temper_start": float(fcfg.new_row_temper_start),
        "tempered_new_rows_mean": event_mean(
            [{"tempered": v} for v in history.get("tempered_new_rows", [])], "tempered"
        ),
        "support_fade_static": bool(fcfg.support_fade),
        "support_fade_until_frac": fcfg.support_fade_until_frac,
        "support_fade_crossfade_iters": int(fcfg.support_fade_crossfade_iters),
        "support_fade_alpha_mean": event_mean(
            [{"alpha": v} for v in history.get("support_fade_alpha", [])], "alpha"
        ),
        **split_recovery,
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
        "level_iters": out.get("level_iters"),
        "level_budgets": out.get("level_budgets"),
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
    background_modes=None,
    opacity_modes=None,
    renderers=None,
    aa_dilations=None,
    color_basis_modes=None,
    color_solve_modes=None,
    pixel_losses=None,
    loss_weight_modes=None,
    optimizers=None,
    lr_schedules=None,
    refine_modes=None,
    refine_sites=None,
    refine_primitives=None,
    refine_nms_modes=None,
    refine_color_inits=None,
    refine_score_modes=None,
    refine_prune_modes=None,
    refine_relocate_modes=None,
    state_seed_modes=None,
    row_temper_modes=None,
    support_fade_modes=None,
    pyramid_modes=None,
    render_chunk=512,
    ssim_weight=0.3,
    ssim_backend="builtin",
    flank_offset=None,
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
    pyramid_level_iters=None,
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
    early_exit: bool = False,
    early_exit_window: int = 150,
    early_exit_min_delta: float = 0.02,
    early_exit_min_iters: int | None = None,
    log_every: int | None = None,
    dedupe: bool = True,
    outdir="results/stage_search",
    device=None,
    max_configs: int | None = None,
    resume: bool = False,
    max_new_cells: int | None = None,
    shuffle_configs=False,
    config_seed=0,
    verbose=False,
):
    import torch

    if mode not in ("factorial", "influence"):
        raise ValueError(f"unknown mode {mode!r}; expected factorial or influence")
    explicit_refine = (
        refine_sites, refine_primitives, refine_nms_modes, refine_color_inits,
        refine_score_modes,
        refine_prune_modes, refine_relocate_modes,
    )
    if refine_modes is not None and any(v is not None for v in explicit_refine):
        raise ValueError("use either legacy refine_modes or explicit refine axes, not both")
    defaults = INFLUENCE_DEFAULTS if mode == "influence" else FACTORIAL_DEFAULTS
    supplied = {
        "strategies": strategies, "tensor_operators": tensor_operators,
        "tensor_colors": tensor_colors, "density_modes": density_modes,
        "sampling_modes": sampling_modes, "orientation_modes": orientation_modes,
        "color_modes": color_modes, "scale_modes": scale_modes,
        "scale_cap_modes": scale_cap_modes,
        "background_modes": background_modes,
        "opacity_modes": opacity_modes, "renderers": renderers,
        "aa_dilations": aa_dilations,
        "color_basis_modes": color_basis_modes,
        "color_solve_modes": color_solve_modes,
        "pixel_losses": pixel_losses, "loss_weight_modes": loss_weight_modes,
        "optimizers": optimizers,
        "lr_schedules": lr_schedules,
        "state_seed_modes": state_seed_modes,
        "row_temper_modes": row_temper_modes,
        "support_fade_modes": support_fade_modes,
        "pyramid_modes": pyramid_modes,
    }
    if refine_modes is not None:
        supplied["refine_modes"] = refine_modes
    else:
        supplied.update({
            "refine_sites": refine_sites,
            "refine_primitives": refine_primitives,
            "refine_nms_modes": refine_nms_modes,
            "refine_color_inits": refine_color_inits,
            "refine_score_modes": refine_score_modes,
            "refine_prune_modes": refine_prune_modes,
            "refine_relocate_modes": refine_relocate_modes,
        })
    axes = {a: _one(v) if v is not None else defaults[a] for a, v in supplied.items()}

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_path = Path(outdir)
    out_path.mkdir(parents=True, exist_ok=True)
    files = _iter_images(images)
    if not files:
        raise SystemExit("no images found")

    if pyramid_iters_per_level is None:
        pyramid_iters_per_level = max(1, iters // max(1, pyramid_levels))
    if pyramid_level_iters is not None:
        pyramid_level_iters = [int(v) for v in pyramid_level_iters]
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

    write_config(str(out_path), run_config({
        "images": files, "mode": mode, "budgets": list(budgets), "seeds": list(seeds),
        "iters": iters, "max_side": max_side, "axes": {k: list(v) for k, v in axes.items()},
        "render_chunk": render_chunk, "ssim_weight": ssim_weight, "split_every": split_every,
        "ssim_backend": ssim_backend,
        "split_count": split_count, "prune_every": prune_every,
        "prune_min_activity": prune_min_activity, "max_gaussians": max_gaussians,
        "pyramid_levels": pyramid_levels, "pyramid_fractions": list(pyramid_fractions),
        "pyramid_iters_per_level": pyramid_iters_per_level,
        "pyramid_level_iters": pyramid_level_iters,
        "target_psnr": target_psnr, "target_psnrs": list(target_psnrs),
        "target_ms_ssim": target_ms_ssim, "target_bpp": target_bpp,
        "adaptive_count": adaptive_count,
        "adaptive_growth_every": adaptive_growth_every,
        "adaptive_growth_count": adaptive_growth_count,
        "adaptive_split_mode": adaptive_split_mode,
        "adaptive_min_delta_psnr": adaptive_min_delta_psnr,
        "adaptive_patience": adaptive_patience,
        "early_exit": early_exit,
        "early_exit_window": early_exit_window,
        "early_exit_min_delta": early_exit_min_delta,
        "early_exit_min_iters": early_exit_min_iters,
        "log_every": log_every,
        "resume": resume,
        "max_new_cells": max_new_cells,
        "dedupe": dedupe,
    }, device=device))

    canonical = _normalize_refine_config({_AXIS_TO_KEY[a]: vals[0] for a, vals in axes.items()})
    raw = _influence_configs(axes) if mode == "influence" else _iter_configs(axes)
    configs, seen = [], set()
    n_dropped = 0
    for cfg in raw:
        normalized = _normalize_refine_config(cfg)
        c = _canonicalize(normalized, canonical) if dedupe else normalized
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
        base = _normalize_refine_config({_AXIS_TO_KEY[a]: vals[0] for a, vals in axes.items()})
        base = _canonicalize(base, canonical)
        baseline_label = _config_label(base)

    jsonl_path = out_path / "stage_search.jsonl"
    rows = _load_jsonl(jsonl_path) if resume else []
    if not resume:
        jsonl_path.unlink(missing_ok=True)   # fresh incremental log for this run
    done = {_cell_key(row) for row in rows}

    new_cells = 0
    stop_requested = False
    for image_path in files:
        if stop_requested:
            break
        img = _load_image(image_path, max_side)
        target = torch.as_tensor(img, device=device)
        image_name = Path(image_path).stem
        for budget in budgets:
            if stop_requested:
                break
            for seed in seeds:
                if stop_requested:
                    break
                for config_idx, cfg in enumerate(configs):
                    if max_new_cells is not None and new_cells >= max_new_cells:
                        stop_requested = True
                        break
                    # equal-budget arms (BENCH-002): cap at the cell budget, and let adding-refine
                    # arms START below budget so their planned additions land AT budget — otherwise
                    # they ran with up to +split_count more capacity than the baselines they rank
                    # against, confounding "refine wins" with extra capacity.
                    cap = budget if max_gaussians is None else max_gaussians
                    adds = _refine_adds_capacity(cfg)
                    init_budget = max(1, budget - split_count) if adds else budget
                    base_row = {
                        "image": image_name, "budget": budget, "seed": seed,
                        "init_budget": init_budget, "max_gaussians": cap,
                        **{k: cfg[k] for k in cfg if k != "label"},
                        "config_label": cfg["label"],
                        "is_baseline": cfg["label"] == baseline_label,
                    }
                    if _cell_key(base_row) in done:
                        continue
                    print(
                        f"[{image_name}] budget={budget} seed={seed} "
                        f"config={config_idx + 1}/{len(configs)} {cfg['label']}",
                        flush=True,
                    )
                    early_stop_patience = None
                    early_stop_min_iters_resolved = 0
                    if early_exit:
                        early_stop_patience = max(
                            1,
                            int(math.ceil(max(1, early_exit_window) / max(1, log_every))),
                        )
                        early_stop_min_iters_resolved = int(early_exit_min_iters or 0)
                        if adds:
                            early_stop_min_iters_resolved = max(
                                early_stop_min_iters_resolved,
                                int(split_every),
                            )
                        if adaptive_count:
                            early_stop_min_iters_resolved = max(
                                early_stop_min_iters_resolved,
                                int(adaptive_growth_every),
                            )
                    base_row.update({
                        "early_exit": bool(early_exit),
                        "early_stop_patience": early_stop_patience,
                        "early_stop_min_delta": (
                            early_exit_min_delta if early_stop_patience is not None else None
                        ),
                        "early_stop_min_iters": early_stop_min_iters_resolved
                        if early_stop_patience is not None else None,
                    })
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
                            pyramid_level_iters=pyramid_level_iters,
                            compute_lpips=compute_lpips, target_psnr=target_psnr,
                            target_psnrs=target_psnrs, target_ms_ssim=target_ms_ssim,
                            target_bpp=target_bpp, adaptive_count=adaptive_count,
                            adaptive_growth_every=adaptive_growth_every,
                            adaptive_growth_count=adaptive_growth_count,
                            adaptive_split_mode=adaptive_split_mode,
                            adaptive_min_delta_psnr=adaptive_min_delta_psnr,
                            adaptive_patience=adaptive_patience,
                            early_stop_patience=early_stop_patience,
                            early_stop_min_delta=early_exit_min_delta,
                            early_stop_min_iters=early_stop_min_iters_resolved,
                            log_every=log_every, verbose=verbose,
                        )
                        row = {**base_row, "status": "ok", **metrics}
                    except Exception as exc:  # one broken arm must not void the whole sweep
                        row = {**base_row, "status": "error", "error": repr(exc)}
                        print(f"  ERROR in cell: {exc!r}", flush=True)
                    rows.append(row)
                    done.add(_cell_key(row))
                    new_cells += 1
                    with jsonl_path.open("a", encoding="utf-8") as jf:  # crash-recoverable
                        jf.write(json.dumps({k: v for k, v in row.items()
                                             if k not in {"history", "prefix_metrics"}}) + "\n")
                    print({k: row[k] for k in row
                           if k not in {"history", "prefix_metrics"}}, flush=True)

    _write(rows, out_path, mode=mode, baseline_label=baseline_label)
    return rows


def _config_key(row):
    r = _normalize_refine_config(row)
    return tuple(r[k] for k in STAGE_KEYS) + (row["budget"],)


def _fmt(v, spec=".4f"):
    return format(v, spec) if v is not None else "-"


def _target_table_columns(targets: list[float]) -> tuple[str, str]:
    header = "".join(
        f" | Hit {target_label(t)} | Iter {target_label(t)}"
        for t in targets
    )
    align = "|---:" * (2 * len(targets))
    return header, align


def _target_cells(rows: list[dict[str, Any]], targets: list[float]) -> str:
    cells = []
    for target in targets:
        stats = target_hit_stats(rows, target)
        cells.append(f"{stats['hits']}/{stats['runs']}")
        cells.append(_fmt(stats["mean_iter"], ".0f"))
    return " | ".join(cells)


def _target_delta_stat(pairs: list[dict], target: float) -> str:
    ds = []
    for p in pairs:
        rv = target_iters(p["r"], target)
        bv = target_iters(p["b"], target)
        if rv is not None and bv is not None:
            ds.append(rv - bv)
    if not ds:
        return "-"
    return f"{mean(ds):+.1f} ± {pstdev(ds):.1f}"


def _target_reach_stat(pairs: list[dict], target: float) -> str:
    rv = sum(1 for p in pairs if target_iters(p["r"], target) is not None)
    bv = sum(1 for p in pairs if target_iters(p["b"], target) is not None)
    return f"{rv}/{bv}/{len(pairs)}"


def summarize(rows, top_k: int = 20) -> str:
    rows = [
        _normalize_refine_config(r) for r in rows if r.get("status") != "error"
    ]  # broken cells never enter the ranking
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
    targets = headline_target_psnrs(rows, defaults=HEADLINE_TARGET_PSNRS)
    target_header, target_align = _target_table_columns(targets)

    lines = [
        "# StructSplat Stage Search",
        "",
        "| Rank | Budget | Mean PSNR | Std | Mean MS-SSIM | Mean AUC"
        f"{target_header} | Mean fit s | Config |",
        "|---:|---:|---:|---:|---:|---:"
        f"{target_align}|---:|---|",
    ]
    for rank, (_, _key, vals) in enumerate(ranked[:top_k], 1):
        psnrs = [v["psnr"] for v in vals]
        ms = [v["ms_ssim"] for v in vals]
        sec = [v["fit_seconds"] for v in vals]
        aucs = [v["auc_psnr"] for v in vals if v.get("auc_psnr") is not None]
        label = vals[0]["config_label"]
        budget = vals[0]["budget"]
        lines.append(
            f"| {rank} | {budget} | {mean(psnrs):.4f} | {pstdev(psnrs):.4f} | "
            f"{mean(ms):.5f} | {_fmt(mean(aucs) if aucs else None, '.3f')} | "
            f"{_target_cells(vals, targets)} | "
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
    rows = [_normalize_refine_config(r) for r in rows if r.get("status") != "error"]
    targets = headline_target_psnrs(rows, defaults=HEADLINE_TARGET_PSNRS)
    target_header, target_align = _target_table_columns(targets)
    lines = [
        "## Per-stage marginal means",
        "",
        "| Stage | Level | Runs | PSNR | MS-SSIM | AUC"
        f"{target_header} | Fit s |",
        "|---|---|---:|---:|---:|---:"
        f"{target_align}|---:|",
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
            sec = [r["fit_seconds"] for r in sub]
            lines.append(
                f"| {stage} | {lv} | {len(sub)} | {mean(psnrs):.3f} ± {pstdev(psnrs):.3f} | "
                f"{mean(ms):.5f} | {_fmt(mean(aucs) if aucs else None, '.3f')} | "
                f"{_target_cells(sub, targets)} | "
                f"{mean(sec):.2f} |"
            )
    return "\n".join(lines)


def summarize_influence(rows, baseline_label: str) -> str:
    """Paired per-stage deltas vs the baseline config: the stage-influence answer.

    Each variant row is matched with the baseline row of the same (image, budget, seed) and
    the metric differences are aggregated. Positive ΔPSNR/ΔAUC = variant better; negative
    Δiters-to-target / Δseconds = variant faster.
    """
    rows = [_normalize_refine_config(r) for r in rows if r.get("status") != "error"]
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

    base_auc = [r["auc_psnr"] for r in base.values() if r.get("auc_psnr") is not None]
    targets = headline_target_psnrs(rows, defaults=HEADLINE_TARGET_PSNRS)
    target_delta_header = "".join(
        f" | Δiter@{target_label(t)} | reach@{target_label(t)}"
        for t in targets
    )
    target_delta_align = "|---:" * (2 * len(targets))
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
        "reach@target = target reached (variant/baseline/cells).",
        "",
        "| Stage | Variant | Cells | ΔPSNR | ΔMS-SSIM | ΔAUC"
        f"{target_delta_header} | Δinit s | Δfit s |",
        "|---|---|---:|---:|---:|---:"
        f"{target_delta_align}|---:|---:|",
    ]
    order = {k: i for i, k in enumerate(STAGE_KEYS)}
    for (stage, label), pairs in sorted(variants.items(),
                                        key=lambda kv: (order.get(kv[0][0], 99), kv[0][1])):
        lines.append(
            f"| {stage} | `{label}` | {len(pairs)} | {dstat(pairs, 'psnr')} | "
            f"{dstat(pairs, 'ms_ssim')} | {dstat(pairs, 'auc_psnr')} | "
            + " | ".join(
                f"{_target_delta_stat(pairs, t)} | {_target_reach_stat(pairs, t)}"
                for t in targets
            )
            + " | "
            f"{dstat(pairs, 'init_seconds')} | {dstat(pairs, 'fit_seconds')} |"
        )
    return "\n".join(lines) + "\n"


def _influence_variant_summaries(rows: list[dict[str, Any]],
                                 baseline_label: str | None) -> list[dict[str, Any]]:
    if baseline_label is None:
        bases = {r.get("config_label") for r in rows if r.get("is_baseline")}
        if len(bases) == 1:
            baseline_label = next(iter(bases))
    if baseline_label is None:
        return []
    ok = [_normalize_refine_config(r) for r in rows if r.get("status") != "error"]
    base = {
        (r["image"], r["budget"], r["seed"]): r
        for r in ok
        if r["config_label"] == baseline_label
    }
    variants: dict[tuple[str, str], list[dict[str, dict]]] = {}
    for r in ok:
        if r["config_label"] == baseline_label:
            continue
        b = base.get((r["image"], r["budget"], r["seed"]))
        if b is None:
            continue
        diff = [k for k in STAGE_KEYS if r[k] != b[k]]
        stage = diff[0] if len(diff) == 1 else "+".join(diff)
        label = "|".join(f"{k}={r[k]}" for k in diff)
        variants.setdefault((stage, label), []).append({"r": r, "b": b})

    def delta(pairs: list[dict[str, dict]], key: str) -> tuple[float | None, float | None]:
        ds = [
            p["r"][key] - p["b"][key]
            for p in pairs
            if p["r"].get(key) is not None and p["b"].get(key) is not None
        ]
        if not ds:
            return None, None
        return float(mean(ds)), float(pstdev(ds))

    out = []
    for (stage, label), pairs in variants.items():
        dpsnr, spsnr = delta(pairs, "psnr")
        dms, sms = delta(pairs, "ms_ssim")
        dauc, sauc = delta(pairs, "auc_psnr")
        dinit, sinit = delta(pairs, "init_seconds")
        dfit, sfit = delta(pairs, "fit_seconds")
        out.append({
            "stage": stage,
            "variant": label,
            "cells": len(pairs),
            "dpsnr": dpsnr,
            "dpsnr_std": spsnr,
            "dms_ssim": dms,
            "dms_ssim_std": sms,
            "dauc": dauc,
            "dauc_std": sauc,
            "dinit_seconds": dinit,
            "dinit_seconds_std": sinit,
            "dfit_seconds": dfit,
            "dfit_seconds_std": sfit,
        })
    order = {k: i for i, k in enumerate(STAGE_KEYS)}
    out.sort(key=lambda r: (order.get(r["stage"], 99), str(r["variant"])))
    return out


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
    _write_index_html(rows, outdir, mode=mode, baseline_label=baseline_label)
    wrote += " / index.html"
    print(f"\nwrote {wrote} to {outdir}")


def _write_index_html(rows: list[dict[str, Any]], outdir: Path, *, mode: str,
                      baseline_label: str | None = None) -> None:
    rows = [_normalize_refine_config(r) for r in rows]
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

    def fmt_delta(value: float | None, std: float | None = None, spec: str = ".3f") -> str:
        if value is None:
            return "-"
        text = format(float(value), f"+{spec}")
        if std is not None:
            text += f" +/- {float(std):.3f}"
        return text

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
        ".best-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:10px;margin:10px 0 22px}",
        ".badge{display:inline-block;border:1px solid #a3a3a3;background:#f5f5f5;border-radius:999px;padding:2px 7px;margin:1px 3px 1px 0;font-size:12px}",
        ".badge.best{border-color:#047857;background:#ecfdf5;color:#064e3b;font-weight:650}",
        "tr.best-row{background:#f0fdf4}",
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

    influence_summaries = (
        _influence_variant_summaries(ok_rows, baseline_label)
        if mode == "influence" else []
    )
    if influence_summaries:
        winners: dict[str, tuple[str, str]] = {}

        def best_max(key: str) -> dict[str, Any] | None:
            vals = [r for r in influence_summaries if r.get(key) is not None]
            return max(vals, key=lambda r: r[key]) if vals else None

        def best_min(key: str) -> dict[str, Any] | None:
            vals = [r for r in influence_summaries if r.get(key) is not None]
            return min(vals, key=lambda r: r[key]) if vals else None

        best_cards = [
            ("Best PSNR", best_max("dpsnr"), "dpsnr", ".3f"),
            ("Best MS-SSIM", best_max("dms_ssim"), "dms_ssim", ".5f"),
            ("Best AUC", best_max("dauc"), "dauc", ".3f"),
            ("Fastest fit", best_min("dfit_seconds"), "dfit_seconds", ".3f"),
        ]
        for label, summary, _key, _spec in best_cards:
            if summary is not None:
                winners[label] = (summary["stage"], summary["variant"])
        html += [
            "<h2>Best Influence Variants</h2>",
            '<p class="note">Influence-mode leaders are paired deltas against the baseline. '
            "They identify the best variant inside this run; they are not default-promotion "
            "claims by themselves.</p>",
            '<div class="best-grid">',
        ]
        for label, summary, key, spec in best_cards:
            if summary is None:
                continue
            value = fmt_delta(summary.get(key), None, spec)
            html.append(
                '<div class="card">'
                f'<div class="label">{escape(label)}</div>'
                f'<div class="value">{escape(value)}</div>'
                f'<div><span class="badge best">{escape(str(summary["variant"]))}</span></div>'
                f'<div class="note">stage {escape(str(summary["stage"]))}, '
                f'{int(summary["cells"])} paired cells</div>'
                "</div>"
            )
        html.append("</div>")

        html += [
            "<h2>Influence Paired Deltas</h2>",
            "<table><thead><tr><th>Rank</th><th>Best</th><th>Stage</th><th>Variant</th><th>Cells</th>"
            "<th>ΔPSNR</th><th>ΔMS-SSIM</th><th>ΔAUC</th><th>Δinit s</th>"
            "<th>Δfit s</th></tr></thead><tbody>",
        ]
        ranked_influence = sorted(
            influence_summaries,
            key=lambda r: (
                -float("-inf") if r["dpsnr"] is None else -r["dpsnr"],
                -float("-inf") if r["dms_ssim"] is None else -r["dms_ssim"],
                -float("-inf") if r["dauc"] is None else -r["dauc"],
                str(r["stage"]),
                str(r["variant"]),
            ),
        )
        for rank, summary in enumerate(ranked_influence, 1):
            marks = [
                name for name, ident in winners.items()
                if ident == (summary["stage"], summary["variant"])
            ]
            row_class = ' class="best-row"' if marks else ""
            mark_html = " ".join(
                f'<span class="badge best">{escape(name)}</span>' for name in marks
            )
            if not mark_html:
                mark_html = "-"
            html.append(
                f"<tr{row_class}>"
                f'<td class="num">{rank}</td>'
                f"<td>{mark_html}</td>"
                f"<td>{escape(str(summary['stage']))}</td>"
                f"<td><code>{escape(str(summary['variant']))}</code></td>"
                f'<td class="num">{int(summary["cells"])}</td>'
                f'<td class="num">{fmt_delta(summary["dpsnr"], summary["dpsnr_std"])}</td>'
                f'<td class="num">{fmt_delta(summary["dms_ssim"], summary["dms_ssim_std"], ".5f")}</td>'
                f'<td class="num">{fmt_delta(summary["dauc"], summary["dauc_std"])}</td>'
                f'<td class="num">{fmt_delta(summary["dinit_seconds"], summary["dinit_seconds_std"])}</td>'
                f'<td class="num">{fmt_delta(summary["dfit_seconds"], summary["dfit_seconds_std"])}</td>'
                "</tr>"
            )
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
                   help="none, hard8/hard12, feature8/feature12, feature_cap<N>, or feature_rel")
    p.add_argument("--background-modes", nargs="+", default=None,
                   help="off or frac<F>_grid<N>, e.g. frac0.10_grid16")
    p.add_argument("--opacity-modes", nargs="+", default=None)
    p.add_argument("--renderers", nargs="+", default=None)
    p.add_argument("--aa-dilations", type=float, nargs="+", default=None)
    p.add_argument("--color-basis-modes", nargs="+", default=None,
                   help="constant or affine")
    p.add_argument("--color-solve-modes", nargs="+", default=None,
                   help="none, every10, or every<N>; normalized renderer only")
    p.add_argument("--pixel-losses", nargs="+", default=None)
    p.add_argument("--loss-weight-modes", nargs="+", default=None,
                   help="none, tensor, or tensor_<beta>; weights only the pixel-loss term")
    p.add_argument("--optimizers", nargs="+", default=None)
    p.add_argument("--lr-schedules", nargs="+", default=None)
    p.add_argument("--refine-modes", nargs="+", default=None,
                   help="legacy flat refine aliases; prefer the explicit refine axes")
    p.add_argument("--refine-sites", nargs="+", default=None,
                   help="none, residual, residual_tensor, support, ranked, absgrad, freq_violation")
    p.add_argument("--refine-primitives", nargs="+", default=None,
                   help="duplicate, fp, moment_preserving, sampled_add")
    p.add_argument("--refine-nms-modes", nargs="+", default=None,
                   help="off or on")
    p.add_argument("--refine-color-inits", nargs="+", default=None,
                   help="target or residual")
    p.add_argument("--refine-score-modes", nargs="+", default=None,
                   help="legacy_abs, gaussian_abs, or signed_gaussian; sampled-add only")
    p.add_argument("--refine-prune-modes", nargs="+", default=None,
                   help="off or on")
    p.add_argument("--refine-relocate-modes", nargs="+", default=None,
                   help="off or on")
    p.add_argument("--state-seed-modes", nargs="+", default=None,
                   help="off or on; seed new-row optimizer moments from parent/median")
    p.add_argument("--row-temper-modes", nargs="+", default=None,
                   help="off or warmup<N>; post-insert update ramp")
    p.add_argument("--support-fade-modes", nargs="+", default=None,
                   help="off, on, or until<F>; scheduled compact-support fade")
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
    p.add_argument("--early-exit", action="store_true",
                   help="opt-in plateau early exit; AUC holds last PSNR to the nominal horizon")
    p.add_argument("--early-exit-window", type=int, default=150,
                   help="plateau window in iterations; translated to logged PSNR samples")
    p.add_argument("--early-exit-min-delta", type=float, default=0.02,
                   help="minimum PSNR improvement over the window to keep running")
    p.add_argument("--early-exit-min-iters", type=int, default=None,
                   help="absolute iteration floor before early exit may trigger")
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
    p.add_argument("--pyramid-level-iters", type=int, nargs="+", default=None,
                   help="explicit coarse-to-fine pyramid iteration counts")
    p.add_argument("--lpips", action="store_true")
    p.add_argument("--max-configs", type=int, default=None)
    p.add_argument("--resume", action="store_true",
                   help="resume from stage_search.jsonl in --outdir and skip completed cells")
    p.add_argument("--max-new-cells", type=int, default=None,
                   help="stop after this many newly executed cells; useful for GPU shards")
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
        scale_cap_modes=a.scale_cap_modes, background_modes=a.background_modes,
        opacity_modes=a.opacity_modes,
        renderers=a.renderers, aa_dilations=a.aa_dilations,
        color_basis_modes=a.color_basis_modes,
        color_solve_modes=a.color_solve_modes,
        pixel_losses=a.pixel_losses, loss_weight_modes=a.loss_weight_modes,
        optimizers=a.optimizers,
        lr_schedules=a.lr_schedules, refine_modes=a.refine_modes,
        refine_sites=a.refine_sites,
        refine_primitives=a.refine_primitives,
        refine_nms_modes=a.refine_nms_modes,
        refine_color_inits=a.refine_color_inits,
        refine_score_modes=a.refine_score_modes,
        refine_prune_modes=a.refine_prune_modes,
        refine_relocate_modes=a.refine_relocate_modes,
        state_seed_modes=a.state_seed_modes,
        row_temper_modes=a.row_temper_modes,
        support_fade_modes=a.support_fade_modes,
        pyramid_modes=a.pyramid_modes, render_chunk=a.chunk,
        ssim_weight=a.ssim_weight, ssim_backend=a.ssim_backend,
        split_every=a.split_every, split_count=a.split_count,
        prune_every=a.prune_every, prune_min_activity=a.prune_min_activity,
        max_gaussians=a.max_gaussians, pyramid_levels=a.pyramid_levels,
        pyramid_fractions=a.pyramid_fractions, pyramid_iters_per_level=a.pyramid_iters_per_level,
        pyramid_level_iters=a.pyramid_level_iters,
        compute_lpips=a.lpips, target_psnr=a.target_psnr, target_psnrs=a.target_psnrs,
        target_ms_ssim=a.target_ms_ssim, target_bpp=a.target_bpp,
        adaptive_count=a.adaptive_count,
        adaptive_growth_every=a.adaptive_growth_every,
        adaptive_growth_count=a.adaptive_growth_count,
        adaptive_split_mode=a.adaptive_split_mode,
        adaptive_min_delta_psnr=a.adaptive_min_delta_psnr,
        adaptive_patience=a.adaptive_patience,
        early_exit=a.early_exit,
        early_exit_window=a.early_exit_window,
        early_exit_min_delta=a.early_exit_min_delta,
        early_exit_min_iters=a.early_exit_min_iters,
        log_every=a.log_every, dedupe=not a.no_dedupe,
        max_configs=a.max_configs, resume=a.resume, max_new_cells=a.max_new_cells,
        shuffle_configs=a.shuffle_configs,
        config_seed=a.config_seed, outdir=a.outdir, device=a.device, verbose=a.verbose,
    )


if __name__ == "__main__":
    main()
