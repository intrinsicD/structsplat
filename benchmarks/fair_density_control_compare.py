"""Fair density-control comparison against repo-inspired 2D Gaussian baselines.

This is the density-control-aware companion to ``cross_repo_matrix_compare.py``. It keeps the
renderer, fitter, loss, target tracking, initial Gaussian count, final Gaussian cap, and growth
schedule matched across growth rows, then varies the placement/growth policy:

* GaussianImage-style fixed full-count random control.
* GaussianImage++ / Image-GS inspired residual-growth analogues.
* StructSplat initializers under the same residual-growth schedule.
* StructSplat initializers under tensor-aware residual growth.

The rows are still executable matched-policy analogues, not native external-repo pipelines. Native
repo runs remain a separate practical benchmark because those repositories use different renderers,
optimizers, metrics, codecs, and checkpoint assumptions.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, replace
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, TYPE_CHECKING

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps

from benchmarks.common import (
    HEADLINE_TARGET_PSNRS,
    build_comparison_analogue,
    json_safe_rows,
    load_image,
    psnr_auc,
    resolve_seeds,
    run_config,
    save_image,
    target_tensor,
    target_label,
    write_config,
    write_csv,
    write_json,
)
from structsplat import metrics as M
from structsplat.config import FitConfig, InitConfig, StructureTensorConfig
from structsplat.fit import fit
from structsplat.init import build_field

if TYPE_CHECKING:
    from structsplat.gaussians import GaussianField


DEFAULT_IMAGES = [
    "results/datasets/abl004/kodak24/kodim01.png",
    "results/datasets/abl004/kodak24/kodim07.png",
    "results/datasets/abl004/kodak24/kodim13.png",
    "results/datasets/abl004/kodak24/kodim19.png",
]

DEFAULT_METHODS = [
    "gaussianimage_fixed_full",
    "gaussianimage_plus_residual",
    "image_gs_residual",
    "instant_gi_quadtree_fixed",
    "structsplat_onedge_residual",
    "structsplat_onedge_residual_relocate",
    "structsplat_onedge_residual_featurecap",
    "structsplat_onedge_residual_feature_rel",
    "structsplat_onedge_tensor",
    "structsplat_onedge_tensor_featurecap",
    "structsplat_onedge_tensor_feature_rel",
    "structsplat_flanking_tensor",
    "structsplat_quadtree_wse_residual",
    "structsplat_quadtree_wse_residual_relocate",
    "structsplat_quadtree_wse_residual_featurecap",
    "structsplat_quadtree_wse_residual_feature_rel",
    "structsplat_quadtree_wse_tensor",
    "structsplat_quadtree_wse_tensor_featurecap",
    "structsplat_quadtree_wse_tensor_feature_rel",
    "structsplat_quadtree_hybrid_tensor",
    "floyd_steinberg_tensor",
]

METHOD_LABELS = {
    "gaussianimage_fixed_full": "GaussianImage fixed",
    "gaussianimage_plus_residual": "GaussianImage++ residual",
    "image_gs_residual": "Image-GS residual",
    "instant_gi_quadtree_fixed": "Instant-GI quadtree",
    "structsplat_onedge_residual": "SS on-edge + residual",
    "structsplat_onedge_residual_relocate": "SS on-edge + residual relocate",
    "structsplat_onedge_residual_featurecap": "SS on-edge + residual feature cap",
    "structsplat_onedge_residual_feature_rel": "SS on-edge + residual feature-rel cap",
    "structsplat_onedge_tensor": "SS on-edge + tensor",
    "structsplat_onedge_tensor_featurecap": "SS on-edge + tensor feature cap",
    "structsplat_onedge_tensor_feature_rel": "SS on-edge + tensor feature-rel cap",
    "structsplat_flanking_tensor": "SS flanking + tensor",
    "structsplat_quadtree_wse_residual": "SS qt-WSE + residual",
    "structsplat_quadtree_wse_residual_relocate": "SS qt-WSE + residual relocate",
    "structsplat_quadtree_wse_residual_featurecap": "SS qt-WSE + residual feature cap",
    "structsplat_quadtree_wse_residual_feature_rel": "SS qt-WSE + residual feature-rel cap",
    "structsplat_quadtree_wse_tensor": "SS qt-WSE + tensor",
    "structsplat_quadtree_wse_tensor_featurecap": "SS qt-WSE + tensor feature cap",
    "structsplat_quadtree_wse_tensor_feature_rel": "SS qt-WSE + tensor feature-rel cap",
    "structsplat_quadtree_hybrid_tensor": "SS qt-hybrid + tensor",
    "floyd_steinberg_tensor": "Floyd + tensor",
}

METHOD_NOTES = {
    "gaussianimage_fixed_full": (
        "GaussianImage-style random fixed-count control; starts at the final cap and does not grow."
    ),
    "gaussianimage_plus_residual": (
        "GaussianImage++-style analogue: random half-budget start plus residual-add growth."
    ),
    "image_gs_residual": (
        "Image-GS-style analogue: gradient-density random half-budget start plus residual-add growth."
    ),
    "instant_gi_quadtree_fixed": (
        "Instant-GI quadtree/Delaunay fallback if STRUCTSPLAT_INSTANT_GI is configured; fixed count."
    ),
    "structsplat_onedge_residual": (
        "StructSplat on-edge initializer under the same residual-add growth as external analogues."
    ),
    "structsplat_onedge_residual_relocate": (
        "StructSplat on-edge residual-add growth plus split-scheduled residual relocation."
    ),
    "structsplat_onedge_residual_featurecap": (
        "StructSplat on-edge residual-add growth with feature-adaptive per-Gaussian scale caps."
    ),
    "structsplat_onedge_residual_feature_rel": (
        "StructSplat on-edge residual-add growth with feature-relative local-radius scale caps."
    ),
    "structsplat_onedge_tensor": (
        "StructSplat on-edge initializer plus tensor-aware residual growth."
    ),
    "structsplat_onedge_tensor_featurecap": (
        "StructSplat on-edge tensor-aware residual growth with feature-adaptive scale caps."
    ),
    "structsplat_onedge_tensor_feature_rel": (
        "StructSplat on-edge tensor-aware residual growth with feature-relative local-radius caps."
    ),
    "structsplat_flanking_tensor": (
        "StructSplat flanking initializer plus tensor-aware residual growth."
    ),
    "structsplat_quadtree_wse_residual": (
        "StructSplat quadtree-WSE initializer under the same residual-add growth as external analogues."
    ),
    "structsplat_quadtree_wse_residual_relocate": (
        "StructSplat quadtree-WSE residual-add growth plus split-scheduled residual relocation."
    ),
    "structsplat_quadtree_wse_residual_featurecap": (
        "StructSplat quadtree-WSE residual-add growth with feature-adaptive scale caps."
    ),
    "structsplat_quadtree_wse_residual_feature_rel": (
        "StructSplat quadtree-WSE residual-add growth with feature-relative local-radius caps."
    ),
    "structsplat_quadtree_wse_tensor": (
        "StructSplat quadtree-WSE initializer plus tensor-aware residual growth."
    ),
    "structsplat_quadtree_wse_tensor_featurecap": (
        "StructSplat quadtree-WSE tensor-aware residual growth with feature-adaptive scale caps."
    ),
    "structsplat_quadtree_wse_tensor_feature_rel": (
        "StructSplat quadtree-WSE tensor-aware residual growth with feature-relative local-radius caps."
    ),
    "structsplat_quadtree_hybrid_tensor": (
        "StructSplat quadtree-hybrid initializer plus tensor-aware residual growth."
    ),
    "floyd_steinberg_tensor": (
        "Floyd-Steinberg placement control plus tensor-aware residual growth."
    ),
}

METHOD_TRACKS = {
    "gaussianimage_fixed_full": "fixed-full",
    "gaussianimage_plus_residual": "repo-growth",
    "image_gs_residual": "repo-growth",
    "instant_gi_quadtree_fixed": "fixed-full",
    "structsplat_onedge_residual": "same-growth",
    "structsplat_onedge_residual_relocate": "same-growth+relocate",
    "structsplat_onedge_residual_featurecap": "same-growth+feature-cap",
    "structsplat_onedge_residual_feature_rel": "same-growth+feature-rel-cap",
    "structsplat_onedge_tensor": "tensor-growth",
    "structsplat_onedge_tensor_featurecap": "tensor-growth+feature-cap",
    "structsplat_onedge_tensor_feature_rel": "tensor-growth+feature-rel-cap",
    "structsplat_flanking_tensor": "tensor-growth",
    "structsplat_quadtree_wse_residual": "same-growth",
    "structsplat_quadtree_wse_residual_relocate": "same-growth+relocate",
    "structsplat_quadtree_wse_residual_featurecap": "same-growth+feature-cap",
    "structsplat_quadtree_wse_residual_feature_rel": "same-growth+feature-rel-cap",
    "structsplat_quadtree_wse_tensor": "tensor-growth",
    "structsplat_quadtree_wse_tensor_featurecap": "tensor-growth+feature-cap",
    "structsplat_quadtree_wse_tensor_feature_rel": "tensor-growth+feature-rel-cap",
    "structsplat_quadtree_hybrid_tensor": "tensor-growth",
    "floyd_steinberg_tensor": "tensor-growth-control",
}

STRUCTSPLAT_INIT = {
    "structsplat_onedge_residual": ("aniso_onedge", "wse", 0.0),
    "structsplat_onedge_residual_relocate": ("aniso_onedge", "wse", 0.0),
    "structsplat_onedge_residual_featurecap": ("aniso_onedge", "wse", 0.0),
    "structsplat_onedge_residual_feature_rel": ("aniso_onedge", "wse", 0.0),
    "structsplat_onedge_tensor": ("aniso_onedge", "wse", 0.0),
    "structsplat_onedge_tensor_featurecap": ("aniso_onedge", "wse", 0.0),
    "structsplat_onedge_tensor_feature_rel": ("aniso_onedge", "wse", 0.0),
    "structsplat_flanking_tensor": ("aniso_flanking", "wse", 0.5),
    "structsplat_quadtree_wse_residual": ("quadtree_wse", "wse", 0.0),
    "structsplat_quadtree_wse_residual_relocate": ("quadtree_wse", "wse", 0.0),
    "structsplat_quadtree_wse_residual_featurecap": ("quadtree_wse", "wse", 0.0),
    "structsplat_quadtree_wse_residual_feature_rel": ("quadtree_wse", "wse", 0.0),
    "structsplat_quadtree_wse_tensor": ("quadtree_wse", "wse", 0.0),
    "structsplat_quadtree_wse_tensor_featurecap": ("quadtree_wse", "wse", 0.0),
    "structsplat_quadtree_wse_tensor_feature_rel": ("quadtree_wse", "wse", 0.0),
    "structsplat_quadtree_hybrid_tensor": ("quadtree_hybrid", "wse", 0.0),
    "floyd_steinberg_tensor": ("aniso_flanking", "floyd_steinberg", 0.5),
}

STRUCTSPLAT_SPLIT_MODE = {
    "structsplat_onedge_residual": "residual_add",
    "structsplat_onedge_residual_relocate": "residual_add",
    "structsplat_onedge_residual_featurecap": "residual_add",
    "structsplat_onedge_residual_feature_rel": "residual_add",
    "structsplat_onedge_tensor": "residual_tensor_add",
    "structsplat_onedge_tensor_featurecap": "residual_tensor_add",
    "structsplat_onedge_tensor_feature_rel": "residual_tensor_add",
    "structsplat_flanking_tensor": "residual_tensor_add",
    "structsplat_quadtree_wse_residual": "residual_add",
    "structsplat_quadtree_wse_residual_relocate": "residual_add",
    "structsplat_quadtree_wse_residual_featurecap": "residual_add",
    "structsplat_quadtree_wse_residual_feature_rel": "residual_add",
    "structsplat_quadtree_wse_tensor": "residual_tensor_add",
    "structsplat_quadtree_wse_tensor_featurecap": "residual_tensor_add",
    "structsplat_quadtree_wse_tensor_feature_rel": "residual_tensor_add",
    "structsplat_quadtree_hybrid_tensor": "residual_tensor_add",
    "floyd_steinberg_tensor": "residual_tensor_add",
}

RELOCATION_METHODS = {
    "structsplat_onedge_residual_relocate",
    "structsplat_quadtree_wse_residual_relocate",
}

FEATURE_CAP_METHODS = {
    "structsplat_onedge_residual_featurecap",
    "structsplat_onedge_tensor_featurecap",
    "structsplat_quadtree_wse_residual_featurecap",
    "structsplat_quadtree_wse_tensor_featurecap",
}

FEATURE_REL_METHODS = {
    "structsplat_onedge_residual_feature_rel",
    "structsplat_onedge_tensor_feature_rel",
    "structsplat_quadtree_wse_residual_feature_rel",
    "structsplat_quadtree_wse_tensor_feature_rel",
}

DEFAULT_FEATURE_CAP_REFERENCE_SIDE = 160.0


def _one(x):
    return tuple(x) if isinstance(x, (list, tuple)) else (x,)


def _start_budget(final_budget: int, start_fraction: float) -> int:
    if not 0.0 < start_fraction <= 1.0:
        raise ValueError(f"start_fraction must be in (0, 1], got {start_fraction}")
    return max(16, min(int(final_budget), int(round(final_budget * start_fraction))))


def _growth_fit_cfg(
    base: FitConfig,
    final_budget: int,
    start_budget: int,
    split_mode: str,
    growth_waves: int,
) -> FitConfig:
    add_total = max(0, int(final_budget) - int(start_budget))
    split_count = max(1, int(math.ceil(add_total / max(1, growth_waves)))) if add_total else 0
    split_every = max(1, base.iters // (growth_waves + 1)) if add_total else None
    return replace(
        base,
        split_every=split_every,
        split_count=split_count,
        split_mode=split_mode,
        refine_site=None,
        refine_primitive=None,
        refine_nms=None,
        max_gaussians=int(final_budget),
    )


def _feature_cap_pixels(img: np.ndarray, feature_cap: float,
                        reference_side: float) -> float:
    if feature_cap <= 0.0:
        raise ValueError(f"feature_cap must be > 0, got {feature_cap}")
    if reference_side <= 0.0:
        raise ValueError(f"feature_cap_reference_side must be > 0, got {reference_side}")
    return float(feature_cap) * (float(max(img.shape[:2])) / float(reference_side))


def _relocation_growth_fit_cfg(
    base: FitConfig,
    final_budget: int,
    start_budget: int,
    split_mode: str,
    growth_waves: int,
    relocate_fraction: float,
    relocate_downsample: int,
) -> FitConfig:
    if relocate_fraction < 0.0:
        raise ValueError(f"relocate_fraction must be >= 0, got {relocate_fraction}")
    cfg = _growth_fit_cfg(base, final_budget, start_budget, split_mode, growth_waves)
    if cfg.split_count <= 0 or relocate_fraction == 0.0:
        return replace(
            cfg,
            relocate_at_split=False,
            relocate_every=None,
            relocate_count=0,
            relocate_residual_downsample=max(1, int(relocate_downsample)),
        )
    relocate_count = max(1, int(math.ceil(cfg.split_count * relocate_fraction)))
    return replace(
        cfg,
        relocate_at_split=True,
        relocate_every=None,
        relocate_count=relocate_count,
        relocate_residual_downsample=max(1, int(relocate_downsample)),
    )


def _base_fit(args: argparse.Namespace) -> FitConfig:
    target_psnrs = sorted(set(float(x) for x in args.target_psnrs + [args.target_psnr]))
    return FitConfig(
        iters=args.iters,
        target_psnr=args.target_psnr,
        target_psnrs=target_psnrs,
        render_chunk=args.render_chunk,
        renderer=args.renderer,
        support_fade=bool(getattr(args, "support_fade", False)),
        pixel_loss=args.pixel_loss,
        ssim_weight=args.ssim_weight,
        compute_lpips=False,
        log_every=max(1, args.iters // 20),
    )


def _structsplat_field(
    img: np.ndarray,
    method: str,
    start_budget: int,
    seed: int,
    scfg: StructureTensorConfig,
    device: str,
    feature_cap: float | None = None,
    feature_rel: bool = False,
) -> tuple[GaussianField, InitConfig, float]:
    strategy, sampling_mode, flank = STRUCTSPLAT_INIT[method]
    scale_cap_mode = "feature_rel" if feature_rel else (
        "feature" if feature_cap is not None else "none"
    )
    icfg = InitConfig(
        strategy=strategy,
        num_gaussians=start_budget,
        seed=seed,
        sampling_mode=sampling_mode,
        flank_offset_frac=flank,
        scale_cap_mode=scale_cap_mode,
        scale_cap_max=feature_cap if not feature_rel else None,
    )
    t0 = time.time()
    field = build_field(img, icfg, scfg, device=device)
    return field, icfg, time.time() - t0


def _build_method(
    method: str,
    img: np.ndarray,
    image_path: Path,
    final_budget: int,
    start_budget: int,
    seed: int,
    base_fit: FitConfig,
    scfg: StructureTensorConfig,
    growth_waves: int,
    device: str,
    relocate_fraction: float = 0.25,
    relocate_downsample: int = 4,
    feature_cap: float = 12.0,
    feature_cap_reference_side: float = DEFAULT_FEATURE_CAP_REFERENCE_SIDE,
) -> tuple[GaussianField, FitConfig, float, int, dict[str, Any]]:
    if method == "gaussianimage_fixed_full":
        t0 = time.time()
        icfg = InitConfig(strategy="random", num_gaussians=final_budget, seed=seed)
        field = build_field(img, icfg, StructureTensorConfig(), device=device)
        return field, base_fit, time.time() - t0, final_budget, {
            "init_config": asdict(icfg),
            "growth_rule": "none",
        }

    if method == "gaussianimage_plus_residual":
        t0 = time.time()
        icfg = InitConfig(strategy="random", num_gaussians=start_budget, seed=seed)
        field = build_field(img, icfg, scfg, device=device)
        init_seconds = time.time() - t0
        fcfg = _growth_fit_cfg(base_fit, final_budget, start_budget, "residual_add", growth_waves)
        return field, fcfg, init_seconds, start_budget, {
            "init_config": asdict(icfg),
            "growth_rule": "residual_add",
        }

    if method == "image_gs_residual":
        t0 = time.time()
        icfg = InitConfig(
            strategy="iso_blue_noise",
            num_gaussians=start_budget,
            density_mode="gradient",
            sampling_mode="density_random",
            scale_mode="spacing",
            seed=seed,
        )
        field = build_field(img, icfg, scfg, device=device)
        init_seconds = time.time() - t0
        fcfg = _growth_fit_cfg(base_fit, final_budget, start_budget, "residual_add", growth_waves)
        return field, fcfg, init_seconds, start_budget, {
            "init_config": asdict(icfg),
            "growth_rule": "residual_add",
        }

    if method == "instant_gi_quadtree_fixed":
        field, fcfg, init_seconds, actual_start = build_comparison_analogue(
            "instant_gi_quadtree", img, image_path, final_budget, seed, device, base_fit, scfg
        )
        return field, fcfg, init_seconds, actual_start, {
            "init_config": {"strategy": "instant_gi_quadtree", "seed": seed},
            "growth_rule": "none",
        }

    if method in STRUCTSPLAT_INIT:
        feature_cap_px = (
            _feature_cap_pixels(img, feature_cap, feature_cap_reference_side)
            if method in FEATURE_CAP_METHODS else None
        )
        field, icfg, init_seconds = _structsplat_field(
            img,
            method,
            start_budget,
            seed,
            scfg,
            device,
            feature_cap=feature_cap_px,
            feature_rel=method in FEATURE_REL_METHODS,
        )
        split_mode = STRUCTSPLAT_SPLIT_MODE[method]
        if method in RELOCATION_METHODS:
            fcfg = _relocation_growth_fit_cfg(
                base_fit,
                final_budget,
                start_budget,
                split_mode,
                growth_waves,
                relocate_fraction,
                relocate_downsample,
            )
            return field, fcfg, init_seconds, start_budget, {
                "init_config": asdict(icfg),
                "growth_rule": f"{split_mode}+relocate",
                "relocate_rule": "at_split",
                "relocate_count_per_event": fcfg.relocate_count,
                "relocate_fraction": relocate_fraction,
                "relocate_residual_downsample": fcfg.relocate_residual_downsample,
            }
        fcfg = _growth_fit_cfg(base_fit, final_budget, start_budget, split_mode, growth_waves)
        extra = {
            "init_config": asdict(icfg),
            "growth_rule": split_mode,
        }
        if method in FEATURE_CAP_METHODS:
            extra.update({
                "scale_cap_rule": "feature",
                "scale_cap_input": feature_cap,
                "scale_cap_reference_side": feature_cap_reference_side,
                "scale_cap_max": feature_cap_px,
                "feature_cap_px": feature_cap_px,
            })
        elif method in FEATURE_REL_METHODS:
            extra.update({
                "scale_cap_rule": "feature_rel",
                "scale_cap_input": None,
                "scale_cap_reference_side": None,
                "scale_cap_max": None,
                "feature_cap_px": None,
            })
        return field, fcfg, init_seconds, start_budget, extra

    valid = ", ".join(DEFAULT_METHODS)
    raise ValueError(f"unknown fair comparison method {method!r}; expected one of: {valid}")


def _extra_metrics(render: torch.Tensor, target: torch.Tensor, want_lpips: bool) -> dict[str, float | None]:
    err = (render - target).detach()
    abs_err = err.abs()
    mse = torch.mean(err * err).clamp_min(1e-12)
    luma = torch.tensor([0.2126, 0.7152, 0.0722], device=target.device, dtype=target.dtype)
    render_luma = (render * luma).sum(dim=2)
    target_luma = (target * luma).sum(dim=2)
    luma_mse = torch.mean((render_luma - target_luma) ** 2).clamp_min(1e-12)
    lpips_val = None
    if want_lpips:
        try:
            lpips_val = M.LPIPS.distance(render, target)
        except Exception as exc:
            print(f"  LPIPS skipped: {type(exc).__name__}: {exc}", flush=True)
            lpips_val = None
    return {
        "mse": float(mse),
        "mae": float(abs_err.mean()),
        "p95_abs_error": float(torch.quantile(abs_err.reshape(-1), 0.95)),
        "luma_psnr": float(10.0 * torch.log10(1.0 / luma_mse)),
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


def _cell_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("image"),
        row.get("source_path"),
        int(row.get("max_side")),
        int(row.get("final_budget")),
        int(row.get("start_budget")),
        float(row.get("start_fraction")),
        int(row.get("growth_waves")),
        int(row.get("seed")),
        row.get("method"),
        int(row.get("iters")),
        row.get("renderer"),
        bool(row.get("support_fade", False)),
        row.get("pixel_loss"),
        float(row.get("ssim_weight")),
        row.get("feature_cap_px") if str(row.get("method", "")).endswith("_featurecap") else None,
    )


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


def _fit_one(
    method: str,
    img: np.ndarray,
    image_path: Path,
    target: torch.Tensor,
    final_budget: int,
    start_budget: int,
    seed: int,
    base_fit: FitConfig,
    scfg: StructureTensorConfig,
    growth_waves: int,
    device: str,
    want_lpips: bool,
    relocate_fraction: float = 0.25,
    relocate_downsample: int = 4,
    feature_cap: float = 12.0,
    feature_cap_reference_side: float = DEFAULT_FEATURE_CAP_REFERENCE_SIDE,
) -> tuple[dict[str, Any], np.ndarray]:
    field, fcfg, init_seconds, actual_start, extra = _build_method(
        method,
        img,
        image_path,
        final_budget,
        start_budget,
        seed,
        base_fit,
        scfg,
        growth_waves,
        device,
        relocate_fraction,
        relocate_downsample,
        feature_cap,
        feature_cap_reference_side,
    )
    out = fit(field, target, fcfg, verbose=False)
    render = out["render"].detach().clamp(0, 1)
    row = {
        "start_gaussians": int(actual_start),
        "n_gaussians": int(out["n_gaussians"]),
        "growth_rule": extra["growth_rule"],
        "init_seconds": float(init_seconds),
        "fit_seconds": float(out["fit_seconds"]),
        "total_seconds": float(init_seconds + out["fit_seconds"]),
        "iterations_run": int(out.get("iterations_run", base_fit.iters)),
        "stopped_early": bool(out.get("stopped_early", False)),
        "psnr": M.psnr(render, target),
        "ssim": float(M.ssim(render, target, backend=fcfg.ssim_backend)),
        "ms_ssim": M.ms_ssim(render, target),
        "auc_psnr": psnr_auc(out.get("history", {})),
        "iters_to_targets": out.get("iters_to_targets", {}),
        "history": out.get("history", {}),
        "init_config": extra["init_config"],
        "fit_config": asdict(fcfg),
        **_extra_metrics(render, target, want_lpips),
        **_scale_stats(out["field"]),
    }
    for key in (
        "relocate_rule",
        "relocate_count_per_event",
        "relocate_fraction",
        "relocate_residual_downsample",
        "scale_cap_rule",
        "scale_cap_input",
        "scale_cap_reference_side",
        "scale_cap_max",
        "feature_cap_px",
    ):
        if key in extra:
            row[key] = extra[key]
    return row, render.cpu().numpy()


def _mean_or_none(vals: list[float | None]) -> float | None:
    clean = [float(v) for v in vals if v is not None and not math.isnan(float(v))]
    return mean(clean) if clean else None


def _std_or_none(vals: list[float | None]) -> float | None:
    clean = [float(v) for v in vals if v is not None and not math.isnan(float(v))]
    if not clean:
        return None
    return pstdev(clean) if len(clean) > 1 else 0.0


def _fmt(v: float | None, digits: int = 4) -> str:
    if v is None:
        return "-"
    return f"{float(v):.{digits}f}"


def _fmt_pct(v: float | None) -> str:
    if v is None:
        return "-"
    return f"{100.0 * float(v):.0f}%"


def _write_outputs(rows: list[dict[str, Any]], outdir: Path, methods: list[str]) -> None:
    json_rows = json_safe_rows(rows, skip={"history"})
    write_json(outdir / "metrics.json", json_rows)
    if json_rows:
        fieldnames = sorted({k for r in json_rows for k in r.keys() if k not in {"fit_config", "init_config", "iters_to_targets"}})
        write_csv(outdir / "metrics.csv", json_rows, fieldnames=fieldnames, extrasaction="ignore")
    _write_convergence_tables(rows, outdir, methods)
    _write_summary(rows, outdir, methods)
    _write_plots(rows, outdir, methods)
    _write_grids(rows, outdir, methods)
    _write_index(outdir, methods)


def _groups(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    out: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("status") != "ok":
            continue
        key = tuple(row[k] for k in keys)
        out.setdefault(key, []).append(row)
    return out


def _history_pairs(row: dict[str, Any]) -> list[tuple[int, float]]:
    history = row.get("history") or {}
    iters = history.get("iter") or []
    psnrs = history.get("psnr") or []
    out: list[tuple[int, float]] = []
    for it, psnr in zip(iters, psnrs):
        try:
            out.append((int(it), float(psnr)))
        except (TypeError, ValueError):
            continue
    return out


def _history_elapsed(row: dict[str, Any]) -> list[tuple[int, float]]:
    history = row.get("history") or {}
    iters = history.get("iter") or []
    elapsed = history.get("elapsed") or []
    out: list[tuple[int, float]] = []
    for it, seconds in zip(iters, elapsed):
        try:
            out.append((int(it), float(seconds)))
        except (TypeError, ValueError):
            continue
    return out


def _psnr_at_or_after(row: dict[str, Any], target_iter: int) -> float | None:
    for it, psnr in _history_pairs(row):
        if it >= target_iter:
            return psnr
    pairs = _history_pairs(row)
    return pairs[-1][1] if pairs else None


def _target_iter(row: dict[str, Any], target: float) -> int | None:
    targets = row.get("iters_to_targets") or {}
    raw = targets.get(str(float(target)))
    if raw is None:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def _target_seconds(row: dict[str, Any], target: float) -> float | None:
    target_iter = _target_iter(row, target)
    if target_iter is None:
        return None
    for it, seconds in _history_elapsed(row):
        if it >= target_iter:
            return seconds
    elapsed = _history_elapsed(row)
    return elapsed[-1][1] if elapsed else None


def _target_values(rows: list[dict[str, Any]]) -> list[float]:
    vals: set[float] = set()
    for row in rows:
        for key in (row.get("iters_to_targets") or {}).keys():
            try:
                vals.add(float(key))
            except (TypeError, ValueError):
                continue
    return sorted(vals)


def _target_summary(vals: list[dict[str, Any]], target: float) -> dict[str, float | int | None]:
    hit_iters = [_target_iter(r, target) for r in vals]
    hit_iters = [v for v in hit_iters if v is not None]
    hit_seconds = [_target_seconds(r, target) for r in vals]
    hit_seconds = [v for v in hit_seconds if v is not None]
    runs = len(vals)
    hits = len(hit_iters)
    return {
        "runs": runs,
        "hits": hits,
        "hit_rate": hits / runs if runs else None,
        "mean_iter": mean(hit_iters) if hit_iters else None,
        "mean_seconds": mean(hit_seconds) if hit_seconds else None,
    }


def _write_convergence_tables(rows: list[dict[str, Any]], outdir: Path, methods: list[str]) -> None:
    ok = [r for r in rows if r.get("status") == "ok"]
    if not ok:
        return
    curve_rows = []
    for (budget, method), vals in sorted(_groups(ok, ("final_budget", "method")).items()):
        by_iter: dict[int, list[float]] = {}
        for row in vals:
            for it, psnr in _history_pairs(row):
                by_iter.setdefault(it, []).append(psnr)
        for it, psnrs in sorted(by_iter.items()):
            curve_rows.append({
                "final_budget": int(budget),
                "method": method,
                "method_label": METHOD_LABELS[method],
                "iter": it,
                "mean_psnr": mean(psnrs),
                "std_psnr": _std_or_none(psnrs),
                "runs": len(psnrs),
            })
    if curve_rows:
        write_csv(
            outdir / "convergence_curves.csv",
            curve_rows,
            fieldnames=["final_budget", "method", "method_label", "iter", "mean_psnr", "std_psnr", "runs"],
        )

    targets = _target_values(ok)
    target_rows = []
    scopes: list[tuple[str, list[dict[str, Any]]]] = [("all", ok)]
    scopes += [(str(b), [r for r in ok if int(r["final_budget"]) == b]) for b in sorted({int(r["final_budget"]) for r in ok})]
    for budget_scope, scope_vals in scopes:
        for method in methods:
            vals = [r for r in scope_vals if r["method"] == method]
            for target in targets:
                stats = _target_summary(vals, target)
                target_rows.append({
                    "budget": budget_scope,
                    "method": method,
                    "method_label": METHOD_LABELS[method],
                    "target_psnr": target,
                    **stats,
                })
    if target_rows:
        write_csv(
            outdir / "target_hit_rates.csv",
            target_rows,
            fieldnames=[
                "budget",
                "method",
                "method_label",
                "target_psnr",
                "runs",
                "hits",
                "hit_rate",
                "mean_iter",
                "mean_seconds",
            ],
        )


def _write_summary(rows: list[dict[str, Any]], outdir: Path, methods: list[str]) -> None:
    ok = [r for r in rows if r.get("status") == "ok"]
    lines = [
        "# Fair Density-Control Comparison",
        "",
        "Matched-policy comparison against repo-inspired 2D Gaussian baselines.",
        "",
        "Growth rows share the same initial Gaussian count, final cap, growth wave count, fitter, renderer, loss, target tracking, and iteration budget.",
        "This is not a native external-repo benchmark; it isolates placement/growth policies inside StructSplat's fitter and exact renderer.",
        "",
        "## Methods",
        "",
        "| Method | Track | Description |",
        "|---|---|---|",
    ]
    for method in methods:
        lines.append(f"| {METHOD_LABELS[method]} | {METHOD_TRACKS[method]} | {METHOD_NOTES[method]} |")

    lines += [
        "",
        "## Overall Means",
        "",
        "| Method | Runs | PSNR | PSNR Std | MS-SSIM | MS-SSIM Std | AUC | LPIPS | Init s | Fit s | Total s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in methods:
        vals = [r for r in ok if r["method"] == method]
        lines.append(
            f"| {METHOD_LABELS[method]} | {len(vals)} | "
            f"{_fmt(_mean_or_none([r['psnr'] for r in vals]), 4)} | "
            f"{_fmt(_std_or_none([r['psnr'] for r in vals]), 4)} | "
            f"{_fmt(_mean_or_none([r['ms_ssim'] for r in vals]), 5)} | "
            f"{_fmt(_std_or_none([r['ms_ssim'] for r in vals]), 5)} | "
            f"{_fmt(_mean_or_none([r['auc_psnr'] for r in vals]), 3)} | "
            f"{_fmt(_mean_or_none([r['lpips'] for r in vals]), 4)} | "
            f"{_fmt(_mean_or_none([r['init_seconds'] for r in vals]), 3)} | "
            f"{_fmt(_mean_or_none([r['fit_seconds'] for r in vals]), 3)} | "
            f"{_fmt(_mean_or_none([r['total_seconds'] for r in vals]), 3)} |"
        )

    targets = [t for t in _target_values(ok) if t in set(HEADLINE_TARGET_PSNRS)]
    if targets:
        target_header = "".join(
            f" | Hit {target_label(t)} | Iter {target_label(t)}"
            for t in targets
        )
        target_align = "|---:" * (2 * len(targets))
        lines += [
            "",
            "## Convergence",
            "",
            "AUC is the area under the logged PSNR-over-iteration curve; higher means better quality earlier in the same 1500-iteration budget.",
            "",
            "| Method | AUC | PSNR@0 | PSNR@375 | PSNR@750 | PSNR@1125 | Final PSNR |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for method in methods:
            vals = [r for r in ok if r["method"] == method]
            lines.append(
                f"| {METHOD_LABELS[method]} | "
                f"{_fmt(_mean_or_none([r.get('auc_psnr') for r in vals]), 3)} | "
                f"{_fmt(_mean_or_none([_psnr_at_or_after(r, 0) for r in vals]), 3)} | "
                f"{_fmt(_mean_or_none([_psnr_at_or_after(r, 375) for r in vals]), 3)} | "
                f"{_fmt(_mean_or_none([_psnr_at_or_after(r, 750) for r in vals]), 3)} | "
                f"{_fmt(_mean_or_none([_psnr_at_or_after(r, 1125) for r in vals]), 3)} | "
                f"{_fmt(_mean_or_none([r.get('psnr') for r in vals]), 3)} |"
            )

        lines += [
            "",
            "Target-hit cells report hit rate across all image/budget cells and mean hit iteration among cells that reached the target.",
            "",
            f"| Method{target_header} |",
            f"|---{target_align}|",
        ]
        for method in methods:
            vals = [r for r in ok if r["method"] == method]
            cells: list[str] = []
            for target in targets:
                stats = _target_summary(vals, target)
                cells.append(_fmt_pct(stats["hit_rate"]))
                cells.append(_fmt(stats["mean_iter"], 1))
            lines.append(f"| {METHOD_LABELS[method]} | " + " | ".join(cells) + " |")

    lines += [
        "",
        "## Means By Budget",
        "",
        "| Final budget | Method | Start G | Final G | PSNR | PSNR Std | MS-SSIM | AUC | Fit s |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for (budget, method), vals in sorted(_groups(ok, ("final_budget", "method")).items()):
        lines.append(
            f"| {int(budget)} | {METHOD_LABELS[method]} | "
            f"{int(round(mean(r['start_gaussians'] for r in vals)))} | "
            f"{int(round(mean(r['n_gaussians'] for r in vals)))} | "
            f"{mean(r['psnr'] for r in vals):.4f} | {_fmt(_std_or_none([r['psnr'] for r in vals]), 4)} | "
            f"{mean(r['ms_ssim'] for r in vals):.5f} | "
            f"{_fmt(_mean_or_none([r['auc_psnr'] for r in vals]), 3)} | "
            f"{mean(r['fit_seconds'] for r in vals):.3f} |"
        )

    lines += [
        "",
        "## Winners By Image/Budget",
        "",
        "| Image | Budget | Best PSNR | Best MS-SSIM |",
        "|---|---:|---|---|",
    ]
    for (image, budget), vals in sorted(_groups(ok, ("image", "final_budget")).items()):
        best_p = max(vals, key=lambda r: r["psnr"])
        best_m = max(vals, key=lambda r: r["ms_ssim"])
        lines.append(
            f"| {image} | {int(budget)} | {METHOD_LABELS[best_p['method']]} ({best_p['psnr']:.3f}) | "
            f"{METHOD_LABELS[best_m['method']]} ({best_m['ms_ssim']:.5f}) |"
        )

    errors = [r for r in rows if r.get("status") != "ok"]
    if errors:
        lines += ["", "## Errors", "", "| Cell | Error |", "|---|---|"]
        for row in errors:
            lines.append(
                f"| {row.get('image')} {row.get('final_budget')} {row.get('method_label')} | "
                f"`{row.get('error')}` |"
            )

    lines += [
        "",
        f"Plots are under `plots/`; visual grids are under `grids/`; per-cell reconstructions are under `reconstructions/`; amplified x{DIFF_GAIN:g} absolute-difference maps are under `diffs/`.",
    ]
    (outdir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_plots(rows: list[dict[str, Any]], outdir: Path, methods: list[str]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    ok = [r for r in rows if r.get("status") == "ok"]
    if not ok:
        return
    plot_dir = outdir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    budgets = sorted({int(r["final_budget"]) for r in ok})

    def series(metric: str):
        plt.figure(figsize=(10, 5))
        for method in methods:
            xs, ys = [], []
            for budget in budgets:
                vals = [r[metric] for r in ok if r["method"] == method and int(r["final_budget"]) == budget]
                if vals:
                    xs.append(budget)
                    ys.append(mean(vals))
            if xs:
                plt.plot(xs, ys, marker="o", label=METHOD_LABELS[method])
        plt.xlabel("Final Gaussian cap")
        plt.ylabel(metric)
        plt.title(f"Mean {metric} by final cap")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(plot_dir / f"mean_{metric}_by_budget.png", dpi=160)
        plt.close()

    for metric in ("psnr", "ms_ssim", "auc_psnr", "fit_seconds"):
        series(metric)

    ncols = 1
    nrows = len(budgets)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(11, max(4, 3.6 * nrows)), squeeze=False)
    for ax, budget in zip(axes.flat, budgets):
        for method in methods:
            vals = [r for r in ok if r["method"] == method and int(r["final_budget"]) == budget]
            by_iter: dict[int, list[float]] = {}
            for row in vals:
                for it, psnr in _history_pairs(row):
                    by_iter.setdefault(it, []).append(psnr)
            if by_iter:
                xs = sorted(by_iter)
                ys = [mean(by_iter[it]) for it in xs]
                ax.plot(xs, ys, linewidth=1.8, label=METHOD_LABELS[method])
        ax.set_title(f"Mean PSNR convergence, {budget}G cap")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("PSNR")
        ax.grid(True, alpha=0.25)
    axes.flat[0].legend(fontsize=8, ncols=2)
    fig.tight_layout()
    fig.savefig(plot_dir / "mean_psnr_curve_by_budget.png", dpi=160)
    plt.close(fig)

    targets = [t for t in _target_values(ok) if t in {22.0, 24.0, 26.0, 28.0, 30.0, 32.0}]
    if targets:
        mat = []
        for method in methods:
            vals = [r for r in ok if r["method"] == method]
            mat.append([float(_target_summary(vals, t)["hit_rate"] or 0.0) for t in targets])
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(np.array(mat), vmin=0.0, vmax=1.0, cmap="viridis")
        ax.set_xticks(np.arange(len(targets)), [f"{t:g}" for t in targets])
        ax.set_yticks(np.arange(len(methods)), [METHOD_LABELS[m] for m in methods])
        ax.set_xlabel("Target PSNR")
        ax.set_title("Target-hit rates across image/budget cells")
        for y, row in enumerate(mat):
            for x, val in enumerate(row):
                ax.text(x, y, f"{100.0 * val:.0f}%", ha="center", va="center", color="white" if val < 0.55 else "black", fontsize=8)
        fig.colorbar(im, ax=ax, label="Hit rate")
        fig.tight_layout()
        fig.savefig(plot_dir / "target_hit_rate_heatmap.png", dpi=160)
        plt.close(fig)

    baseline = "gaussianimage_plus_residual"
    if any(r["method"] == baseline for r in ok):
        plt.figure(figsize=(10, 5))
        labels = [m for m in methods if m != baseline]
        x = np.arange(len(labels))
        width = 0.8 / max(1, len(budgets))
        for bidx, budget in enumerate(budgets):
            deltas = []
            for method in labels:
                per_unit = []
                for (image, seed), vals in _groups(
                    [r for r in ok if int(r["final_budget"]) == budget],
                    ("image", "seed"),
                ).items():
                    by_method = {r["method"]: r for r in vals}
                    if method in by_method and baseline in by_method:
                        per_unit.append(by_method[method]["psnr"] - by_method[baseline]["psnr"])
                deltas.append(mean(per_unit) if per_unit else 0.0)
            plt.bar(x + (bidx - (len(budgets) - 1) / 2) * width, deltas, width, label=str(budget))
        plt.axhline(0.0, color="black", linewidth=0.8)
        plt.xticks(x, [METHOD_LABELS[m] for m in labels], rotation=35, ha="right")
        plt.ylabel("PSNR delta vs GaussianImage++ residual (dB)")
        plt.title("Paired mean PSNR deltas by budget")
        plt.legend(title="Budget")
        plt.tight_layout()
        plt.savefig(plot_dir / "paired_delta_vs_gaussianimage_plus.png", dpi=160)
        plt.close()


def _font(name: str, size: int):
    try:
        return ImageFont.truetype(name, size)
    except Exception:
        return ImageFont.load_default()


FONT = _font("DejaVuSans.ttf", 11)
FONT_B = _font("DejaVuSans-Bold.ttf", 11)
FONT_TITLE = _font("DejaVuSans-Bold.ttf", 14)
DIFF_GAIN = 6.0


def _fit_thumb(path: Path, size: tuple[int, int]) -> Image.Image:
    img = Image.open(path).convert("RGB")
    thumb = ImageOps.contain(img, size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, (248, 248, 248))
    canvas.paste(thumb, ((size[0] - thumb.width) // 2, (size[1] - thumb.height) // 2))
    return canvas


def _write_abs_diff_image(
    target_path: Path,
    recon_path: Path,
    out_path: Path,
    gain: float = DIFF_GAIN,
) -> Path | None:
    if not target_path.exists() or not recon_path.exists():
        return None
    target = Image.open(target_path).convert("RGB")
    recon = Image.open(recon_path).convert("RGB")
    if recon.size != target.size:
        recon = recon.resize(target.size, Image.Resampling.BILINEAR)
    target_arr = np.asarray(target, dtype=np.float32)
    recon_arr = np.asarray(recon, dtype=np.float32)
    diff = np.clip(np.abs(recon_arr - target_arr) * float(gain), 0.0, 255.0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(diff.astype(np.uint8), mode="RGB").save(out_path)
    return out_path


def _write_zero_diff_image(target_path: Path, out_path: Path) -> Path:
    size = Image.open(target_path).size if target_path.exists() else (8, 8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, (0, 0, 0)).save(out_path)
    return out_path


def _tile(path: Path | None, title: str, subtitle: str, size: tuple[int, int], label_h: int) -> Image.Image:
    canvas = Image.new("RGB", (size[0], size[1] + label_h), "white")
    draw = ImageDraw.Draw(canvas)
    if path is not None and path.exists():
        canvas.paste(_fit_thumb(path, size), (0, 0))
    else:
        draw.rectangle((0, 0, size[0] - 1, size[1] - 1), fill=(238, 238, 238), outline=(180, 180, 180))
        draw.text((8, 8), "missing/error", fill=(90, 90, 90), font=FONT_B)
    draw.text((5, size[1] + 4), title[:34], fill=(0, 0, 0), font=FONT_B)
    draw.text((5, size[1] + 23), subtitle[:46], fill=(55, 55, 55), font=FONT)
    return canvas


def _write_grids(rows: list[dict[str, Any]], outdir: Path, methods: list[str]) -> None:
    ok = [r for r in rows if r.get("status") == "ok"]
    if not ok:
        return
    grid_image_dir = outdir / "grids" / "by_image"
    grid_budget_dir = outdir / "grids" / "by_budget"
    diff_dir = outdir / "diffs"
    grid_image_dir.mkdir(parents=True, exist_ok=True)
    grid_budget_dir.mkdir(parents=True, exist_ok=True)
    by_cell = {
        (r["image"], int(r["final_budget"]), int(r["seed"]), r["method"]): r
        for r in rows
    }
    images = sorted({r["image"] for r in ok})
    budgets = sorted({int(r["final_budget"]) for r in ok})
    seeds = sorted({int(r["seed"]) for r in ok})
    tile_size = (178, 132)
    label_h = 47
    gap = 8
    row_label_w = 82
    header_h = 32
    cell_w, cell_h = tile_size[0], tile_size[1] + label_h
    diff_gap = 4
    pair_h = cell_h * 2 + diff_gap
    diff_subtitle = f"|target-recon| x{DIFF_GAIN:g}"

    for seed in seeds:
        for image in images:
            cols = ["target", *methods]
            W = row_label_w + len(cols) * cell_w + (len(cols) - 1) * gap
            H = header_h + len(budgets) * pair_h + (len(budgets) - 1) * gap
            grid = Image.new("RGB", (W, H), "white")
            draw = ImageDraw.Draw(grid)
            draw.text((0, 6), f"{image} seed {seed}: budgets x methods", fill=(0, 0, 0), font=FONT_TITLE)
            target_path = Path(next(r["target_path"] for r in ok if r["image"] == image))
            target_diff_path = _write_zero_diff_image(target_path, diff_dir / image / f"seed{seed}_target.png")
            for ridx, budget in enumerate(budgets):
                y = header_h + ridx * (pair_h + gap)
                y_diff = y + cell_h + diff_gap
                draw.text((0, y + 6), f"{budget}G", fill=(0, 0, 0), font=FONT_TITLE)
                draw.text((0, y_diff + 6), "diff", fill=(70, 70, 70), font=FONT_TITLE)
                grid.paste(_tile(target_path, "Target", image, tile_size, label_h), (row_label_w, y))
                grid.paste(_tile(target_diff_path, "Diff: target", "zero", tile_size, label_h), (row_label_w, y_diff))
                for midx, method in enumerate(methods, 1):
                    rec = by_cell.get((image, budget, seed, method))
                    path = Path(rec["reconstruction_path"]) if rec and rec.get("status") == "ok" else None
                    diff_path = (
                        _write_abs_diff_image(
                            target_path,
                            path,
                            diff_dir / image / str(budget) / f"seed{seed}_{method}.png",
                        )
                        if path is not None
                        else None
                    )
                    if rec and rec.get("status") == "ok":
                        subtitle = f"P {rec['psnr']:.2f} | MS {rec['ms_ssim']:.4f} | {rec['n_gaussians']}G"
                    else:
                        subtitle = "error/missing"
                    x = row_label_w + midx * (cell_w + gap)
                    grid.paste(_tile(path, METHOD_LABELS[method], subtitle, tile_size, label_h), (x, y))
                    grid.paste(
                        _tile(diff_path, f"Diff: {METHOD_LABELS[method]}", diff_subtitle, tile_size, label_h),
                        (x, y_diff),
                    )
            grid.save(grid_image_dir / f"{image}_seed{seed}_budgets_methods.png")

        for budget in budgets:
            cols = ["target", *methods]
            W = row_label_w + len(cols) * cell_w + (len(cols) - 1) * gap
            H = header_h + len(images) * pair_h + (len(images) - 1) * gap
            grid = Image.new("RGB", (W, H), "white")
            draw = ImageDraw.Draw(grid)
            draw.text((0, 6), f"{budget}G seed {seed}: images x methods", fill=(0, 0, 0), font=FONT_TITLE)
            for ridx, image in enumerate(images):
                y = header_h + ridx * (pair_h + gap)
                y_diff = y + cell_h + diff_gap
                draw.text((0, y + 6), image, fill=(0, 0, 0), font=FONT_TITLE)
                draw.text((0, y_diff + 6), "diff", fill=(70, 70, 70), font=FONT_TITLE)
                target_path = Path(next(r["target_path"] for r in ok if r["image"] == image))
                target_diff_path = _write_zero_diff_image(target_path, diff_dir / image / f"seed{seed}_target.png")
                grid.paste(_tile(target_path, "Target", image, tile_size, label_h), (row_label_w, y))
                grid.paste(_tile(target_diff_path, "Diff: target", "zero", tile_size, label_h), (row_label_w, y_diff))
                for midx, method in enumerate(methods, 1):
                    rec = by_cell.get((image, budget, seed, method))
                    path = Path(rec["reconstruction_path"]) if rec and rec.get("status") == "ok" else None
                    diff_path = (
                        _write_abs_diff_image(
                            target_path,
                            path,
                            diff_dir / image / str(budget) / f"seed{seed}_{method}.png",
                        )
                        if path is not None
                        else None
                    )
                    if rec and rec.get("status") == "ok":
                        subtitle = f"P {rec['psnr']:.2f} | MS {rec['ms_ssim']:.4f} | {rec['n_gaussians']}G"
                    else:
                        subtitle = "error/missing"
                    x = row_label_w + midx * (cell_w + gap)
                    grid.paste(_tile(path, METHOD_LABELS[method], subtitle, tile_size, label_h), (x, y))
                    grid.paste(
                        _tile(diff_path, f"Diff: {METHOD_LABELS[method]}", diff_subtitle, tile_size, label_h),
                        (x, y_diff),
                    )
            grid.save(grid_budget_dir / f"budget_{budget}_seed{seed}_images_methods.png")


def _write_index(outdir: Path, methods: list[str]) -> None:
    plots = [
        ("Mean PSNR by budget", "plots/mean_psnr_by_budget.png"),
        ("Mean MS-SSIM by budget", "plots/mean_ms_ssim_by_budget.png"),
        ("Mean AUC PSNR by budget", "plots/mean_auc_psnr_by_budget.png"),
        ("Mean PSNR convergence", "plots/mean_psnr_curve_by_budget.png"),
        ("Target-hit rates", "plots/target_hit_rate_heatmap.png"),
        ("Mean fit seconds by budget", "plots/mean_fit_seconds_by_budget.png"),
        ("Paired delta vs GaussianImage++", "plots/paired_delta_vs_gaussianimage_plus.png"),
    ]
    image_grids = sorted((outdir / "grids" / "by_image").glob("*.png"))
    budget_grids = sorted((outdir / "grids" / "by_budget").glob("*.png"))
    html = [
        "<!doctype html>",
        '<html lang="en"><head><meta charset="utf-8">',
        "<title>Fair Density-Control Comparison</title>",
        "<style>",
        "body{font-family:system-ui,-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;margin:24px;color:#111}",
        "h1{font-size:26px;margin:0 0 8px} h2{font-size:20px;margin:28px 0 10px}",
        "p{max-width:980px;line-height:1.45}.note{background:#f5f5f5;border-left:4px solid #666;padding:10px 12px;max-width:1040px}",
        "img{max-width:100%;height:auto;border:1px solid #ddd} figure{margin:18px 0 28px} figcaption{font-weight:650;margin-bottom:8px}",
        ".grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:20px;align-items:start}",
        "table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:4px 7px;text-align:left}",
        "</style></head><body>",
        "<h1>Fair Density-Control Comparison</h1>",
        '<p class="note">Matched-policy benchmark: growth rows share the same initial count, final cap, growth waves, fitter, renderer, loss, target tracking, and iteration budget. External repos are represented by local analogues here; this is not a native external-pipeline benchmark.</p>',
        f'<p class="note">Visual grids show each reconstruction row followed by an amplified absolute difference row: |target - reconstruction| x{DIFF_GAIN:g}, clipped for display.</p>',
        "<h2>Files</h2>",
        '<p><a href="summary.md">summary.md</a> · <a href="metrics.csv">metrics.csv</a> · <a href="metrics.json">metrics.json</a> · <a href="convergence_curves.csv">convergence_curves.csv</a> · <a href="target_hit_rates.csv">target_hit_rates.csv</a> · <a href="config.json">config.json</a></p>',
        "<h2>Methods</h2><table><tr><th>Method</th><th>Track</th></tr>",
    ]
    for method in methods:
        html.append(f"<tr><td>{METHOD_LABELS[method]}</td><td>{METHOD_TRACKS[method]}</td></tr>")
    html += ["</table>", "<h2>Metric Diagrams</h2>", '<div class="grid">']
    for title, src in plots:
        if (outdir / src).exists():
            html.append(f'<figure><figcaption>{title}</figcaption><a href="{src}"><img src="{src}" alt="{title}"></a></figure>')
    html.append("</div><h2>Visual Grids By Image</h2>")
    for p in image_grids:
        rel = p.relative_to(outdir)
        html.append(f'<figure><figcaption>{p.stem}</figcaption><a href="{rel}"><img src="{rel}" alt="{p.stem}"></a></figure>')
    html.append("<h2>Visual Grids By Budget</h2>")
    for p in budget_grids:
        rel = p.relative_to(outdir)
        html.append(f'<figure><figcaption>{p.stem}</figcaption><a href="{rel}"><img src="{rel}" alt="{p.stem}"></a></figure>')
    html.append("</body></html>")
    (outdir / "index.html").write_text("\n".join(html) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> list[dict[str, Any]]:
    methods = list(args.methods or DEFAULT_METHODS)
    unknown = sorted(set(methods) - set(DEFAULT_METHODS))
    if unknown:
        valid = ", ".join(DEFAULT_METHODS)
        raise ValueError(f"unknown methods {unknown}; expected one of: {valid}")

    seeds = resolve_seeds(args.seed, args.seeds)
    budgets = [int(b) for b in args.budgets]
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    target_dir = outdir / "targets"
    recon_dir = outdir / "reconstructions"
    target_dir.mkdir(exist_ok=True)
    recon_dir.mkdir(exist_ok=True)

    write_config(str(outdir), run_config({
        "images": [str(p) for p in args.images],
        "budgets": budgets,
        "methods": methods,
        "seeds": seeds,
        "start_fraction": args.start_fraction,
        "growth_waves": args.growth_waves,
        "max_side": args.max_side,
        "iters": args.iters,
        "target_psnr": args.target_psnr,
        "target_psnrs": args.target_psnrs,
        "renderer": args.renderer,
        "support_fade": args.support_fade,
        "render_chunk": args.render_chunk,
        "pixel_loss": args.pixel_loss,
        "ssim_weight": args.ssim_weight,
        "lpips": args.lpips,
        "relocate_fraction": args.relocate_fraction,
        "relocate_downsample": args.relocate_downsample,
        "feature_cap": args.feature_cap,
        "feature_cap_reference_side": args.feature_cap_reference_side,
        "resume": args.resume,
        "max_new_cells": args.max_new_cells,
    }, device=device))

    jsonl_path = outdir / "metrics.jsonl"
    existing = _load_jsonl(jsonl_path) if args.resume else []
    if not args.resume and jsonl_path.exists():
        jsonl_path.unlink()
    done = {_cell_key(r) for r in existing if r.get("status") == "ok"}
    rows = list(existing)

    loaded: dict[str, tuple[np.ndarray, Path]] = {}
    for image_path in args.images:
        image_path = Path(image_path)
        img = load_image(image_path, max_side=args.max_side)
        stem = image_path.stem
        target_path = target_dir / f"{stem}.png"
        save_image(img, target_path)
        loaded[str(image_path)] = (img, target_path)

    scfg = StructureTensorConfig(flat_frac=args.flat_frac, corner_frac=args.corner_frac)
    base_fit = _base_fit(args)
    total = len(args.images) * len(budgets) * len(seeds) * len(methods)
    new_cells = 0
    cell_idx = 0
    for image_path in args.images:
        image_path = Path(image_path)
        img, target_path = loaded[str(image_path)]
        image = image_path.stem
        target = target_tensor(img, device)
        for final_budget in budgets:
            start_budget = _start_budget(final_budget, args.start_fraction)
            for seed in seeds:
                for method in methods:
                    cell_idx += 1
                    key_row = {
                        "image": image,
                        "source_path": str(image_path),
                        "max_side": args.max_side,
                        "final_budget": final_budget,
                        "start_budget": start_budget,
                        "start_fraction": args.start_fraction,
                        "growth_waves": args.growth_waves,
                        "seed": seed,
                        "method": method,
                        "iters": args.iters,
                        "renderer": args.renderer,
                        "support_fade": args.support_fade,
                        "pixel_loss": args.pixel_loss,
                        "ssim_weight": args.ssim_weight,
                        "feature_cap_px": (
                            _feature_cap_pixels(img, args.feature_cap, args.feature_cap_reference_side)
                            if method in FEATURE_CAP_METHODS else None
                        ),
                    }
                    if _cell_key(key_row) in done:
                        print(f"[{cell_idx}/{total}] skip existing {image} {final_budget} {method}", flush=True)
                        continue
                    print(
                        f"[{cell_idx}/{total}] fit {image} cap={final_budget} start={start_budget} "
                        f"seed={seed} {METHOD_LABELS[method]}",
                        flush=True,
                    )
                    row_base = {
                        **key_row,
                        "height": int(img.shape[0]),
                        "width": int(img.shape[1]),
                        "target_path": str(target_path),
                        "method_label": METHOD_LABELS[method],
                        "method_note": METHOD_NOTES[method],
                        "method_track": METHOD_TRACKS[method],
                    }
                    try:
                        row, render_np = _fit_one(
                            method,
                            img,
                            target_path,
                            target,
                            final_budget,
                            start_budget,
                            seed,
                            base_fit,
                            scfg,
                            args.growth_waves,
                            device,
                            args.lpips,
                            args.relocate_fraction,
                            args.relocate_downsample,
                            args.feature_cap,
                            args.feature_cap_reference_side,
                        )
                        recon_path = recon_dir / image / str(final_budget) / f"seed{seed}_{method}.png"
                        save_image(render_np, recon_path)
                        rec = {
                            **row_base,
                            **row,
                            "status": "ok",
                            "error": "",
                            "reconstruction_path": str(recon_path),
                        }
                        print(
                            f"  psnr={rec['psnr']:.3f} ms={rec['ms_ssim']:.5f} "
                            f"auc={_fmt(rec['auc_psnr'], 3)} n={rec['n_gaussians']} "
                            f"fit={rec['fit_seconds']:.2f}s",
                            flush=True,
                        )
                        done.add(_cell_key(rec))
                    except Exception as exc:
                        rec = {
                            **row_base,
                            "status": "error",
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                        print(f"  ERROR {rec['error']}", flush=True)
                    with jsonl_path.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(rec, default=str) + "\n")
                    rows.append(rec)
                    new_cells += 1
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    if args.max_new_cells is not None and new_cells >= args.max_new_cells:
                        _write_outputs(rows, outdir, methods)
                        return rows

    _write_outputs(rows, outdir, methods)
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description="Fair density-control comparison")
    p.add_argument("--images", nargs="+", type=Path, default=[Path(p) for p in DEFAULT_IMAGES])
    p.add_argument("--outdir", type=Path, default=Path("results/fair_density_control_difficult4"))
    p.add_argument("--budgets", nargs="+", type=int, default=[2000, 5000, 10000])
    p.add_argument("--methods", nargs="+", default=None)
    p.add_argument("--max-side", type=int, default=768)
    p.add_argument("--start-fraction", type=float, default=0.5)
    p.add_argument("--growth-waves", type=int, default=4)
    p.add_argument("--iters", type=int, default=1500)
    p.add_argument("--target-psnr", type=float, default=35.0)
    p.add_argument("--target-psnrs", type=float, nargs="*", default=[22.0, 24.0, 26.0, 28.0, 30.0, 32.0])
    p.add_argument("--flat-frac", type=float, default=0.02)
    p.add_argument("--corner-frac", type=float, default=0.15)
    p.add_argument("--render-chunk", type=int, default=512)
    p.add_argument("--renderer", default="cuda")
    p.add_argument("--support-fade", action="store_true",
                   help="enable C0 compact-support fade in the renderer and fit residual scoring")
    p.add_argument("--pixel-loss", choices=["l1", "l2", "charbonnier"], default="l1")
    p.add_argument("--ssim-weight", type=float, default=0.3)
    p.add_argument("--lpips", action="store_true")
    p.add_argument("--relocate-fraction", type=float, default=0.25,
                   help="fraction of split_count moved by relocation rows on each growth event")
    p.add_argument("--relocate-downsample", type=int, default=4,
                   help="coarse residual max-pool factor for relocation rows")
    p.add_argument("--feature-cap", type=float, default=12.0,
                   help="feature cap value at --feature-cap-reference-side for scale-cap rows")
    p.add_argument("--feature-cap-reference-side", type=float,
                   default=DEFAULT_FEATURE_CAP_REFERENCE_SIDE,
                   help="image side length where --feature-cap is interpreted literally")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--max-new-cells", type=int, default=None)
    p.add_argument("--device", default=None)
    from benchmarks.common import add_seed_args

    add_seed_args(p)
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
