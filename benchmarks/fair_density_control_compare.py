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
import hashlib
import json
import math
import os
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
from benchmarks._util import package_versions, repository_state
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


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pixel_sha256(image: np.ndarray) -> str:
    pixels = np.rint(np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)
    digest = hashlib.sha256()
    digest.update(np.asarray(pixels.shape, dtype="<i8").tobytes())
    digest.update(pixels.tobytes(order="C"))
    return digest.hexdigest()


def _artifact_source_id(path: Path, source_sha256: str) -> str:
    """Return a readable, collision-resistant name for source-specific artifacts."""
    canonical = path.resolve()
    digest = hashlib.sha256()
    digest.update(str(canonical).encode("utf-8"))
    digest.update(b"\0")
    digest.update(source_sha256.encode("ascii"))
    return f"{canonical.stem}_{digest.hexdigest()[:16]}"


def _source_fingerprint() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    paths = {
        "benchmark_source_sha256": Path(__file__).resolve(),
        "fit_source_sha256": root / "src" / "structsplat" / "fit.py",
        "config_source_sha256": root / "src" / "structsplat" / "config.py",
        "gaussians_source_sha256": root / "src" / "structsplat" / "gaussians.py",
        "render_source_sha256": root / "src" / "structsplat" / "render.py",
        "metrics_source_sha256": root / "src" / "structsplat" / "metrics.py",
        "init_source_sha256": root / "src" / "structsplat" / "init.py",
    }
    return {name: _file_sha256(path) for name, path in paths.items()}


def _run_protocol_sha256(
    scientific_protocol: dict[str, Any],
    *,
    device: str,
    versions: dict[str, Any],
    repo_state: dict[str, Any],
    source_fingerprint: dict[str, Any],
) -> str:
    payload = {
        "scientific_protocol": scientific_protocol,
        "device": device,
        "versions": versions,
        "repository_commit": repo_state.get("commit"),
        "repository_tracked_diff_sha256": repo_state.get("tracked_diff_sha256"),
        **source_fingerprint,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()

BEST_DEFAULT_METHOD = "structsplat_best_default"
BEST_COSINE_METHOD = "structsplat_best_cosine"
BEST_COSINE_TAIL_METHOD = "structsplat_best_cosine_tail"
BEST_CHECKPOINT_METHOD = "structsplat_best_checkpoint"
BEST_CHECKPOINT_LOWPASS_METHOD = "structsplat_best_checkpoint_lowpass2x_f10"
BEST_CAF_METHOD = "structsplat_best_generation_filter"
BEST_CAF_WEAK_METHOD = "structsplat_best_generation_filter_a18pi"
BEST_CAF_MILD_METHOD = "structsplat_best_generation_filter_a36pi"
BEST_DEFAULT_BASE_METHOD = "structsplat_onedge_tensor_featurecap"
BEST_DEFAULT_FEATURE_CAP = 12.0
BEST_DEFAULT_GROWTH_WAVES = 5
ADAPTIVE_EXTRA_CAP_MULT = 1.5
DOMINANCE_BOOTSTRAP_SAMPLES = 5000
DOMINANCE_BOOTSTRAP_SEED = 0

# Positive gains always mean that the candidate is better than the pinned default.
# Timing metrics therefore use the opposite sign from quality/convergence metrics.
DOMINANCE_METRICS = {
    "psnr": 1.0,
    "ms_ssim": 1.0,
    "auc_psnr": 1.0,
    "fit_seconds": -1.0,
    "total_seconds": -1.0,
}

DEFAULT_METHODS = [
    BEST_DEFAULT_METHOD,
    BEST_COSINE_METHOD,
    BEST_COSINE_TAIL_METHOD,
    BEST_CHECKPOINT_METHOD,
    BEST_CHECKPOINT_LOWPASS_METHOD,
    BEST_CAF_METHOD,
    BEST_CAF_WEAK_METHOD,
    BEST_CAF_MILD_METHOD,
    "structsplat_best_ssim010",
    "structsplat_best_l1_only",
    "structsplat_best_charbonnier",
    "structsplat_best_tensor_loss",
    "structsplat_best_gcr015",
    "structsplat_best_gcr015_e2",
    "structsplat_best_gcr015_e4",
    "structsplat_best_gcr030",
    "structsplat_best_gcr060",
    "structsplat_best_color_final",
    "structsplat_best_relocate",
    "structsplat_best_adaptive_1p5x",
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
    BEST_DEFAULT_METHOD: "SS best default",
    BEST_COSINE_METHOD: "SS best + cosine LR",
    BEST_COSINE_TAIL_METHOD: "SS best + post-growth cosine LR",
    BEST_CHECKPOINT_METHOD: "SS best + full-count checkpoint",
    BEST_CHECKPOINT_LOWPASS_METHOD: "SS checkpoint + 2x low-pass warmup",
    BEST_CAF_METHOD: "SS best + generation CAF",
    BEST_CAF_WEAK_METHOD: "SS best + generation CAF 18pi",
    BEST_CAF_MILD_METHOD: "SS best + generation CAF 36pi",
    "structsplat_best_ssim010": "SS best + SSIM 0.10",
    "structsplat_best_l1_only": "SS best + L1 only",
    "structsplat_best_charbonnier": "SS best + Charbonnier",
    "structsplat_best_tensor_loss": "SS best + tensor loss",
    "structsplat_best_gcr015": "SS best + GCR 0.015",
    "structsplat_best_gcr015_e2": "SS best + GCR 0.015 every2",
    "structsplat_best_gcr015_e4": "SS best + GCR 0.015 every4",
    "structsplat_best_gcr030": "SS best + GCR 0.030",
    "structsplat_best_gcr060": "SS best + GCR 0.060",
    "structsplat_best_color_final": "SS best + final color solve",
    "structsplat_best_relocate": "SS best + split relocate",
    "structsplat_best_adaptive_1p5x": "SS best + adaptive 1.5x cap",
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
    BEST_DEFAULT_METHOD: (
        "Pinned current Gaussian-image winner: aniso_onedge + WSE, feature cap 12@160, "
        "tensor-aware residual growth, 5 growth waves, L1 + 0.3 SSIM."
    ),
    BEST_COSINE_METHOD: (
        "Best default with one horizon-wide cosine learning-rate decay to stabilize long fits."
    ),
    BEST_COSINE_TAIL_METHOD: (
        "Best default held at full LR through five growth waves, then cosine-decayed over "
        "the final one-sixth of the horizon."
    ),
    BEST_CHECKPOINT_METHOD: (
        "Best default with post-transition PSNR checkpoint selection restricted to states "
        "at the terminal Gaussian count."
    ),
    BEST_CHECKPOINT_LOWPASS_METHOD: (
        "Full-count checkpoint control plus a 2x area-lowpass loss target cosine-blended "
        "to the full image by 10% of the global fit horizon."
    ),
    BEST_CAF_METHOD: (
        "Best default plus GaussianImage++-style birth-cohort covariance filtering with "
        "s=min(300, HW/(9*pi*N_after)) px^2."
    ),
    BEST_CAF_WEAK_METHOD: (
        "Weaker generation-density covariance filter with alpha=18*pi (half the native "
        "GaussianImage++ variance)."
    ),
    BEST_CAF_MILD_METHOD: (
        "Mild generation-density covariance filter with alpha=36*pi (quarter the native "
        "GaussianImage++ variance)."
    ),
    "structsplat_best_ssim010": (
        "Best default geometry/growth with SSIM weight lowered to 0.10 for lower pixel diff."
    ),
    "structsplat_best_l1_only": (
        "Best default geometry/growth with pure L1 pixel loss and no SSIM term."
    ),
    "structsplat_best_charbonnier": (
        "Best default geometry/growth with Charbonnier pixel loss and SSIM weight 0.10."
    ),
    "structsplat_best_tensor_loss": (
        "Best default geometry/growth with tensor-weighted pixel loss to emphasize edges."
    ),
    "structsplat_best_gcr015": (
        "Best default plus ground-truth-gradient-weighted Sobel consistency at weight 0.015."
    ),
    "structsplat_best_gcr015_e2": (
        "GCR 0.015 evaluated every 2 steps with active weight scaled to preserve its mean."
    ),
    "structsplat_best_gcr015_e4": (
        "GCR 0.015 evaluated every 4 steps with active weight scaled to preserve its mean."
    ),
    "structsplat_best_gcr030": (
        "Best default plus ground-truth-gradient-weighted Sobel consistency at weight 0.030."
    ),
    "structsplat_best_gcr060": (
        "Best default plus ground-truth-gradient-weighted Sobel consistency at weight 0.060."
    ),
    "structsplat_best_color_final": (
        "Best default geometry/growth with a final fixed-geometry RGB least-squares color solve."
    ),
    "structsplat_best_relocate": (
        "Best default geometry/growth plus split-scheduled residual relocation."
    ),
    "structsplat_best_adaptive_1p5x": (
        "Best default warm start with adaptive residual growth allowed up to 1.5x the requested cap."
    ),
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
    BEST_DEFAULT_METHOD: "best-default",
    BEST_COSINE_METHOD: "best-long-horizon",
    BEST_COSINE_TAIL_METHOD: "best-long-horizon",
    BEST_CHECKPOINT_METHOD: "best-long-horizon",
    BEST_CHECKPOINT_LOWPASS_METHOD: "best-multiscale-loss",
    BEST_CAF_METHOD: "best-covariance-filter",
    BEST_CAF_WEAK_METHOD: "best-covariance-filter",
    BEST_CAF_MILD_METHOD: "best-covariance-filter",
    "structsplat_best_ssim010": "best-loss-sweep",
    "structsplat_best_l1_only": "best-loss-sweep",
    "structsplat_best_charbonnier": "best-loss-sweep",
    "structsplat_best_tensor_loss": "best-edge-loss",
    "structsplat_best_gcr015": "best-geometry-loss",
    "structsplat_best_gcr015_e2": "best-geometry-loss",
    "structsplat_best_gcr015_e4": "best-geometry-loss",
    "structsplat_best_gcr030": "best-geometry-loss",
    "structsplat_best_gcr060": "best-geometry-loss",
    "structsplat_best_color_final": "best-color-polish",
    "structsplat_best_relocate": "best-relocate",
    "structsplat_best_adaptive_1p5x": "best-adaptive-capacity",
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
    BEST_DEFAULT_METHOD: ("aniso_onedge", "wse", 0.0),
    BEST_COSINE_METHOD: ("aniso_onedge", "wse", 0.0),
    BEST_COSINE_TAIL_METHOD: ("aniso_onedge", "wse", 0.0),
    BEST_CHECKPOINT_METHOD: ("aniso_onedge", "wse", 0.0),
    BEST_CHECKPOINT_LOWPASS_METHOD: ("aniso_onedge", "wse", 0.0),
    BEST_CAF_METHOD: ("aniso_onedge", "wse", 0.0),
    BEST_CAF_WEAK_METHOD: ("aniso_onedge", "wse", 0.0),
    BEST_CAF_MILD_METHOD: ("aniso_onedge", "wse", 0.0),
    "structsplat_best_ssim010": ("aniso_onedge", "wse", 0.0),
    "structsplat_best_l1_only": ("aniso_onedge", "wse", 0.0),
    "structsplat_best_charbonnier": ("aniso_onedge", "wse", 0.0),
    "structsplat_best_tensor_loss": ("aniso_onedge", "wse", 0.0),
    "structsplat_best_gcr015": ("aniso_onedge", "wse", 0.0),
    "structsplat_best_gcr015_e2": ("aniso_onedge", "wse", 0.0),
    "structsplat_best_gcr015_e4": ("aniso_onedge", "wse", 0.0),
    "structsplat_best_gcr030": ("aniso_onedge", "wse", 0.0),
    "structsplat_best_gcr060": ("aniso_onedge", "wse", 0.0),
    "structsplat_best_color_final": ("aniso_onedge", "wse", 0.0),
    "structsplat_best_relocate": ("aniso_onedge", "wse", 0.0),
    "structsplat_best_adaptive_1p5x": ("aniso_onedge", "wse", 0.0),
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
    BEST_DEFAULT_METHOD: "residual_tensor_add",
    BEST_COSINE_METHOD: "residual_tensor_add",
    BEST_COSINE_TAIL_METHOD: "residual_tensor_add",
    BEST_CHECKPOINT_METHOD: "residual_tensor_add",
    BEST_CHECKPOINT_LOWPASS_METHOD: "residual_tensor_add",
    BEST_CAF_METHOD: "residual_tensor_add",
    BEST_CAF_WEAK_METHOD: "residual_tensor_add",
    BEST_CAF_MILD_METHOD: "residual_tensor_add",
    "structsplat_best_ssim010": "residual_tensor_add",
    "structsplat_best_l1_only": "residual_tensor_add",
    "structsplat_best_charbonnier": "residual_tensor_add",
    "structsplat_best_tensor_loss": "residual_tensor_add",
    "structsplat_best_gcr015": "residual_tensor_add",
    "structsplat_best_gcr015_e2": "residual_tensor_add",
    "structsplat_best_gcr015_e4": "residual_tensor_add",
    "structsplat_best_gcr030": "residual_tensor_add",
    "structsplat_best_gcr060": "residual_tensor_add",
    "structsplat_best_color_final": "residual_tensor_add",
    "structsplat_best_relocate": "residual_tensor_add",
    "structsplat_best_adaptive_1p5x": "residual_tensor_add",
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
    "structsplat_best_relocate",
    "structsplat_onedge_residual_relocate",
    "structsplat_quadtree_wse_residual_relocate",
}

FEATURE_CAP_METHODS = {
    BEST_DEFAULT_METHOD,
    BEST_COSINE_METHOD,
    BEST_COSINE_TAIL_METHOD,
    BEST_CHECKPOINT_METHOD,
    BEST_CHECKPOINT_LOWPASS_METHOD,
    BEST_CAF_METHOD,
    BEST_CAF_WEAK_METHOD,
    BEST_CAF_MILD_METHOD,
    "structsplat_best_ssim010",
    "structsplat_best_l1_only",
    "structsplat_best_charbonnier",
    "structsplat_best_tensor_loss",
    "structsplat_best_gcr015",
    "structsplat_best_gcr015_e2",
    "structsplat_best_gcr015_e4",
    "structsplat_best_gcr030",
    "structsplat_best_gcr060",
    "structsplat_best_color_final",
    "structsplat_best_relocate",
    "structsplat_best_adaptive_1p5x",
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
BEST_VARIANT_METHODS = {
    BEST_DEFAULT_METHOD,
    BEST_COSINE_METHOD,
    BEST_COSINE_TAIL_METHOD,
    BEST_CHECKPOINT_METHOD,
    BEST_CHECKPOINT_LOWPASS_METHOD,
    BEST_CAF_METHOD,
    BEST_CAF_WEAK_METHOD,
    BEST_CAF_MILD_METHOD,
    "structsplat_best_ssim010",
    "structsplat_best_l1_only",
    "structsplat_best_charbonnier",
    "structsplat_best_tensor_loss",
    "structsplat_best_gcr015",
    "structsplat_best_gcr015_e2",
    "structsplat_best_gcr015_e4",
    "structsplat_best_gcr030",
    "structsplat_best_gcr060",
    "structsplat_best_color_final",
    "structsplat_best_relocate",
    "structsplat_best_adaptive_1p5x",
}
BEST_SSIM010_METHODS = {"structsplat_best_ssim010"}
BEST_L1_ONLY_METHODS = {"structsplat_best_l1_only"}
BEST_CHARBONNIER_METHODS = {"structsplat_best_charbonnier"}
BEST_TENSOR_LOSS_METHODS = {"structsplat_best_tensor_loss"}
BEST_GCR_METHODS = {
    "structsplat_best_gcr015": (0.015, 1),
    "structsplat_best_gcr015_e2": (0.015, 2),
    "structsplat_best_gcr015_e4": (0.015, 4),
    "structsplat_best_gcr030": (0.030, 1),
    "structsplat_best_gcr060": (0.060, 1),
}
BEST_COLOR_FINAL_METHODS = {"structsplat_best_color_final"}
BEST_ADAPTIVE_METHODS = {"structsplat_best_adaptive_1p5x"}
BEST_CHECKPOINT_METHODS = {BEST_CHECKPOINT_METHOD, BEST_CHECKPOINT_LOWPASS_METHOD}
BEST_LOWPASS_TARGET_METHODS = {
    BEST_CHECKPOINT_LOWPASS_METHOD: (2, 0.10),
}
BEST_COSINE_METHODS = {
    BEST_COSINE_METHOD: 0.0,
    BEST_COSINE_TAIL_METHOD: BEST_DEFAULT_GROWTH_WAVES / (BEST_DEFAULT_GROWTH_WAVES + 1),
}
BEST_CAF_METHODS = {
    BEST_CAF_METHOD: 9.0 * math.pi,
    BEST_CAF_WEAK_METHOD: 18.0 * math.pi,
    BEST_CAF_MILD_METHOD: 36.0 * math.pi,
}


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


def _round_budget(value: float) -> int:
    return max(16, int(round(float(value) / 16.0)) * 16)


def _method_growth_waves(method: str, growth_waves: int) -> int:
    return BEST_DEFAULT_GROWTH_WAVES if method in BEST_VARIANT_METHODS else growth_waves


def _method_feature_cap(method: str, feature_cap: float) -> float:
    return BEST_DEFAULT_FEATURE_CAP if method in BEST_VARIANT_METHODS else feature_cap


def _method_feature_cap_reference_side(method: str, reference_side: float) -> float:
    return (
        DEFAULT_FEATURE_CAP_REFERENCE_SIDE
        if method in BEST_VARIANT_METHODS
        else reference_side
    )


def _apply_method_fit_overrides(
    method: str,
    cfg: FitConfig,
    final_budget: int,
) -> tuple[FitConfig, dict[str, Any]]:
    """Apply explicit candidate-variant overrides after the shared growth schedule."""
    overrides: dict[str, Any] = {}
    if method in BEST_VARIANT_METHODS:
        cfg = replace(
            cfg,
            pixel_loss="l1",
            ssim_weight=0.3,
            loss_weighting="none",
            loss_weight_beta=1.0,
            geometry_loss_weight=0.0,
            geometry_loss_every=1,
            color_basis="constant",
            color_solve_every=None,
            color_solve_schedule="none",
            lr_schedule="none",
            loss_target_downsample=1,
            loss_target_full_frac=0.0,
        )
        overrides.update({
            "pinned_best_default": True,
            "pixel_loss": "l1",
            "ssim_weight": 0.3,
            "geometry_loss_weight": 0.0,
            "growth_waves": BEST_DEFAULT_GROWTH_WAVES,
        })
    if method in BEST_SSIM010_METHODS:
        cfg = replace(cfg, ssim_weight=0.1)
        overrides["ssim_weight"] = 0.1
    if method in BEST_L1_ONLY_METHODS:
        cfg = replace(cfg, ssim_weight=0.0)
        overrides["ssim_weight"] = 0.0
    if method in BEST_CHARBONNIER_METHODS:
        cfg = replace(cfg, pixel_loss="charbonnier", ssim_weight=0.1)
        overrides.update({"pixel_loss": "charbonnier", "ssim_weight": 0.1})
    if method in BEST_TENSOR_LOSS_METHODS:
        cfg = replace(cfg, loss_weighting="tensor", loss_weight_beta=1.0)
        overrides.update({"loss_weighting": "tensor", "loss_weight_beta": 1.0})
    if method in BEST_COSINE_METHODS:
        start_frac = BEST_COSINE_METHODS[method]
        cfg = replace(
            cfg,
            lr_schedule="cosine",
            lr_cosine_start_frac=start_frac,
        )
        overrides.update({
            "lr_schedule": "cosine",
            "lr_cosine_start_frac": start_frac,
        })
    if method in BEST_CHECKPOINT_METHODS:
        cfg = replace(cfg, checkpoint_policy="best_psnr_final_count")
        overrides["checkpoint_policy"] = "best_psnr_final_count"
    if method in BEST_LOWPASS_TARGET_METHODS:
        downsample, full_frac = BEST_LOWPASS_TARGET_METHODS[method]
        cfg = replace(
            cfg,
            loss_target_downsample=downsample,
            loss_target_full_frac=full_frac,
        )
        overrides.update({
            "loss_target_downsample": downsample,
            "loss_target_full_frac": full_frac,
        })
    if method in BEST_GCR_METHODS:
        weight, every = BEST_GCR_METHODS[method]
        cfg = replace(
            cfg,
            geometry_loss_weight=weight,
            geometry_loss_every=every,
        )
        overrides.update({
            "geometry_loss_weight": weight,
            "geometry_loss_every": every,
        })
    if method in BEST_CAF_METHODS:
        cfg = replace(
            cfg,
            covariance_filter_mode="generation_density",
            covariance_filter_alpha=BEST_CAF_METHODS[method],
        )
        overrides.update({
            "covariance_filter_mode": "generation_density",
            "covariance_filter_alpha": float(cfg.covariance_filter_alpha),
            "covariance_filter_max_variance": float(cfg.covariance_filter_max_variance),
        })
    if method in BEST_COLOR_FINAL_METHODS:
        cfg = replace(cfg, color_solve_every=None, color_solve_schedule="final")
        overrides.update({"color_solve_every": None, "color_solve_schedule": "final"})
    if method in BEST_ADAPTIVE_METHODS:
        adaptive_cap = _round_budget(final_budget * ADAPTIVE_EXTRA_CAP_MULT)
        cfg = replace(
            cfg,
            adaptive_count=True,
            max_gaussians=adaptive_cap,
            adaptive_growth_every=max(25, cfg.iters // 10),
            adaptive_growth_count=_round_budget(final_budget * 0.10),
            adaptive_split_mode="residual_tensor_add",
        )
        overrides.update({
            "adaptive_count": True,
            "adaptive_cap_multiplier": ADAPTIVE_EXTRA_CAP_MULT,
            "variant_max_gaussians": adaptive_cap,
            "adaptive_growth_every": cfg.adaptive_growth_every,
            "adaptive_growth_count": cfg.adaptive_growth_count,
        })
    return cfg, overrides


def _method_fit_signature(
    method: str,
    base_fit: FitConfig,
    final_budget: int,
    start_budget: int,
    growth_waves: int,
) -> FitConfig:
    if method in STRUCTSPLAT_INIT:
        split_mode = STRUCTSPLAT_SPLIT_MODE[method]
        resolved_growth_waves = _method_growth_waves(method, growth_waves)
        cfg = _growth_fit_cfg(base_fit, final_budget, start_budget, split_mode, resolved_growth_waves)
        cfg, _overrides = _apply_method_fit_overrides(method, cfg, final_budget)
        return cfg
    return base_fit


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
        color_solve_every=None,
        color_solve_schedule="none",
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
        resolved_growth_waves = _method_growth_waves(method, growth_waves)
        resolved_feature_cap = _method_feature_cap(method, feature_cap)
        resolved_feature_cap_reference_side = _method_feature_cap_reference_side(
            method, feature_cap_reference_side
        )
        feature_cap_px = (
            _feature_cap_pixels(
                img,
                resolved_feature_cap,
                resolved_feature_cap_reference_side,
            )
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
                resolved_growth_waves,
                relocate_fraction,
                relocate_downsample,
            )
            fcfg, variant_overrides = _apply_method_fit_overrides(method, fcfg, final_budget)
            extra = {
                "init_config": asdict(icfg),
                "growth_rule": f"{split_mode}+relocate",
                "configured_growth_waves": resolved_growth_waves,
                "relocate_rule": "at_split",
                "relocate_count_per_event": fcfg.relocate_count,
                "relocate_fraction": relocate_fraction,
                "relocate_residual_downsample": fcfg.relocate_residual_downsample,
                "variant_base": BEST_DEFAULT_BASE_METHOD if method in BEST_VARIANT_METHODS else None,
                "variant_overrides": variant_overrides,
            }
            if method in FEATURE_CAP_METHODS:
                extra.update({
                    "scale_cap_rule": "feature",
                    "scale_cap_input": resolved_feature_cap,
                    "scale_cap_reference_side": resolved_feature_cap_reference_side,
                    "scale_cap_max": feature_cap_px,
                    "feature_cap_px": feature_cap_px,
                })
            return field, fcfg, init_seconds, start_budget, extra
        fcfg = _growth_fit_cfg(
            base_fit, final_budget, start_budget, split_mode, resolved_growth_waves
        )
        fcfg, variant_overrides = _apply_method_fit_overrides(method, fcfg, final_budget)
        extra = {
            "init_config": asdict(icfg),
            "growth_rule": split_mode,
            "configured_growth_waves": resolved_growth_waves,
        }
        if method in BEST_VARIANT_METHODS:
            extra.update({
                "variant_base": BEST_DEFAULT_BASE_METHOD,
                "variant_overrides": variant_overrides,
            })
        if method in FEATURE_CAP_METHODS:
            extra.update({
                "scale_cap_rule": "feature",
                "scale_cap_input": resolved_feature_cap,
                "scale_cap_reference_side": resolved_feature_cap_reference_side,
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
        base_scales = field.scales().detach().reshape(-1)
        scales = field.effective_scales().detach().reshape(-1)
        return {
            "final_scale_max": float(scales.max()),
            "final_scale_p95": float(torch.quantile(scales, 0.95)),
            "final_scale_mean": float(scales.mean()),
            "final_base_scale_max": float(base_scales.max()),
            "final_base_scale_p95": float(torch.quantile(base_scales, 0.95)),
            "final_base_scale_mean": float(base_scales.mean()),
        }


def _cell_key(row: dict[str, Any]) -> tuple[Any, ...]:
    fit_config = row.get("fit_config") if isinstance(row.get("fit_config"), dict) else {}
    geometry_loss_weight = row.get(
        "geometry_loss_weight", fit_config.get("geometry_loss_weight", 0.0)
    )
    geometry_loss_every = row.get(
        "geometry_loss_every", fit_config.get("geometry_loss_every", 1)
    )
    checkpoint_policy = row.get(
        "checkpoint_policy", fit_config.get("checkpoint_policy", "terminal")
    )
    loss_target_downsample = row.get(
        "loss_target_downsample", fit_config.get("loss_target_downsample", 1)
    )
    loss_target_full_frac = row.get(
        "loss_target_full_frac", fit_config.get("loss_target_full_frac", 0.0)
    )
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
        row.get("feature_cap_px") if row.get("method") in FEATURE_CAP_METHODS else None,
        row.get("loss_weighting", "none"),
        float(geometry_loss_weight),
        int(geometry_loss_every),
        row.get("color_solve_schedule", "none"),
        checkpoint_policy,
        int(loss_target_downsample),
        float(loss_target_full_frac),
        bool(row.get("adaptive_count", False)),
        row.get("variant_max_gaussians"),
        row.get("source_sha256"),
        row.get("target_pixel_sha256"),
        row.get("repository_commit"),
        row.get("repository_tracked_diff_sha256"),
        row.get("benchmark_source_sha256"),
        row.get("fit_source_sha256"),
        row.get("config_source_sha256"),
        row.get("gaussians_source_sha256"),
        row.get("render_source_sha256"),
        row.get("metrics_source_sha256"),
        row.get("init_source_sha256"),
        row.get("run_protocol_sha256"),
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
    if fcfg.checkpoint_policy != "terminal" and want_lpips:
        # LPIPS is post-fit and excluded from fit_seconds; enabling it here preserves the
        # terminal-vs-selected perceptual audit for checkpoint candidates.
        fcfg = replace(fcfg, compute_lpips=True)
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
        "checkpoint_policy": out.get("checkpoint_policy", "terminal"),
        "checkpoint_history": out.get("checkpoint_history"),
        "trajectory_terminal_iter": out.get("trajectory_terminal_iter"),
        "trajectory_terminal_n_gaussians": out.get("trajectory_terminal_n_gaussians"),
        "trajectory_terminal_psnr": out.get("trajectory_terminal_psnr"),
        "trajectory_terminal_ssim": out.get("trajectory_terminal_ssim"),
        "trajectory_terminal_ms_ssim": out.get("trajectory_terminal_ms_ssim"),
        "trajectory_terminal_lpips": out.get("trajectory_terminal_lpips"),
        "selected_iter": out.get("selected_iter"),
        "selected_n_gaussians": out.get("selected_n_gaussians"),
        "selected_checkpoint_psnr": out.get("selected_checkpoint_psnr"),
        "selected_fit_psnr": out.get("selected_psnr"),
        "selected_fit_ssim": out.get("ssim"),
        "selected_fit_ms_ssim": out.get("ms_ssim"),
        "selected_fit_lpips": out.get("lpips"),
        "selected_from_checkpoint": out.get("selected_from_checkpoint", False),
        "init_config": extra["init_config"],
        "fit_config": asdict(fcfg),
        "covariance_filter_mode": out.get("covariance_filter_mode"),
        "covariance_filter_active": out.get("covariance_filter_active"),
        "covariance_filter_alpha": out.get("covariance_filter_alpha"),
        "covariance_filter_max_variance": out.get("covariance_filter_max_variance"),
        "covariance_filter_cohorts": out.get("covariance_filter_cohorts"),
        "covariance_filter_variance_min": out.get("covariance_filter_variance_min"),
        "covariance_filter_variance_max": out.get("covariance_filter_variance_max"),
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


def _fmt_gain_ci(row: dict[str, Any], metric: str, digits: int) -> str:
    center = row.get(f"gain_{metric}")
    low = row.get(f"ci_low_{metric}")
    high = row.get(f"ci_high_{metric}")
    if center is None or low is None or high is None:
        return "-"
    return f"{float(center):+.{digits}f} [{float(low):+.{digits}f}, {float(high):+.{digits}f}]"


def _fmt_pct(v: float | None) -> str:
    if v is None:
        return "-"
    return f"{100.0 * float(v):.0f}%"


def _write_outputs(rows: list[dict[str, Any]], outdir: Path, methods: list[str]) -> None:
    # Resume journals are an execution aid, not an append-only evidence ledger.  Compact them to
    # the exact requested/current cells so stale source hashes or superseded duplicate rows cannot
    # leak into a later summary under the newly written config provenance.
    with (outdir / "metrics.jsonl").open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, default=str) + "\n")
    json_rows = json_safe_rows(rows, skip={"history", "checkpoint_history"})
    write_json(outdir / "metrics.json", json_rows)
    if json_rows:
        fieldnames = sorted({k for r in json_rows for k in r.keys() if k not in {"fit_config", "init_config", "iters_to_targets"}})
        write_csv(outdir / "metrics.csv", json_rows, fieldnames=fieldnames, extrasaction="ignore")
    else:
        (outdir / "metrics.csv").unlink(missing_ok=True)
    _write_default_dominance(rows, outdir, methods)
    checkpoint_audit = _checkpoint_selection_audit(rows)
    if checkpoint_audit:
        write_csv(
            outdir / "checkpoint_selection.csv",
            checkpoint_audit,
            fieldnames=list(checkpoint_audit[0]),
        )
    else:
        (outdir / "checkpoint_selection.csv").unlink(missing_ok=True)
    lowpass_pairs, lowpass_summary = _paired_method_audit(
        rows,
        baseline_method=BEST_CHECKPOINT_METHOD,
        candidate_method=BEST_CHECKPOINT_LOWPASS_METHOD,
    )
    if lowpass_pairs:
        write_csv(
            outdir / "lowpass_vs_checkpoint.csv",
            lowpass_pairs,
            fieldnames=list(lowpass_pairs[0]),
        )
    else:
        (outdir / "lowpass_vs_checkpoint.csv").unlink(missing_ok=True)
    if lowpass_summary is not None:
        write_csv(
            outdir / "lowpass_vs_checkpoint_summary.csv",
            [lowpass_summary],
            fieldnames=list(lowpass_summary),
        )
    else:
        (outdir / "lowpass_vs_checkpoint_summary.csv").unlink(missing_ok=True)
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


def _promotion_pair_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("source_path") or row.get("image"),
        row.get("source_sha256"),
        row.get("target_pixel_sha256"),
        row.get("repository_commit"),
        row.get("repository_tracked_diff_sha256"),
        row.get("benchmark_source_sha256"),
        row.get("fit_source_sha256"),
        row.get("config_source_sha256"),
        row.get("gaussians_source_sha256"),
        row.get("render_source_sha256"),
        row.get("metrics_source_sha256"),
        row.get("init_source_sha256"),
        row.get("run_protocol_sha256"),
        int(row.get("max_side", 0) or 0),
        int(row.get("final_budget", 0) or 0),
        int(row.get("start_budget", 0) or 0),
        float(row.get("start_fraction", 0.0) or 0.0),
        int(row.get("growth_waves", 0) or 0),
        int(row.get("seed", 0) or 0),
        int(row.get("iters", 0) or 0),
        row.get("renderer"),
        bool(row.get("support_fade", False)),
    )


def _over_budget_methods(rows: list[dict[str, Any]], methods: list[str]) -> set[str]:
    over_budget: set[str] = set()
    for method in methods:
        for row in rows:
            if row.get("status") != "ok" or row.get("method") != method:
                continue
            cap = row.get("final_budget")
            achieved = row.get("n_gaussians")
            if cap and achieved and float(achieved) > float(cap) * 1.02:
                over_budget.add(method)
                break
    return over_budget


def _bootstrap_mean_ci(
    values: list[float],
    *,
    samples: int = DOMINANCE_BOOTSTRAP_SAMPLES,
    seed: int = DOMINANCE_BOOTSTRAP_SEED,
    alpha: float = 0.05,
) -> tuple[float | None, float | None, float | None]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return None, None, None
    center = float(arr.mean())
    if arr.size == 1 or samples <= 0:
        return center, float(arr[0]), float(arr[0])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(samples, arr.size))
    means = arr[idx].mean(axis=1)
    low, high = np.quantile(means, [alpha / 2.0, 1.0 - alpha / 2.0])
    return center, float(low), float(high)


def _clustered_gain_ci(
    pairs: list[tuple[dict[str, Any], dict[str, Any]]],
    metric: str,
    direction: float,
    *,
    seed: int,
    alpha: float = 0.05,
    baseline_field: str | None = None,
    candidate_field: str | None = None,
) -> tuple[float | None, float | None, float | None, int, int]:
    """Return an image-clustered paired gain and bootstrap CI.

    Seeds and budgets from the same source image are correlated, so first average their
    paired gains within image and bootstrap images rather than treating every row as an
    independent unit. This keeps small multi-seed proxy runs from overstating certainty.
    """
    by_image: dict[str, list[float]] = {}
    pair_count = 0
    baseline_field = baseline_field or metric
    candidate_field = candidate_field or metric
    for base, candidate in pairs:
        base_value = base.get(baseline_field)
        candidate_value = candidate.get(candidate_field)
        if base_value is None or candidate_value is None:
            continue
        try:
            gain = direction * (float(candidate_value) - float(base_value))
        except (TypeError, ValueError):
            continue
        if not math.isfinite(gain):
            continue
        image = str(candidate.get("source_path") or candidate.get("image") or "unknown")
        by_image.setdefault(image, []).append(gain)
        pair_count += 1
    image_means = [mean(values) for values in by_image.values()]
    center, low, high = _bootstrap_mean_ci(image_means, seed=seed, alpha=alpha)
    return center, low, high, pair_count, len(image_means)


def _default_dominance_audit(
    rows: list[dict[str, Any]],
    methods: list[str],
    over_budget: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Compare every method with the pinned default on paired multi-objective gains."""
    over_budget = over_budget if over_budget is not None else _over_budget_methods(rows, methods)
    default_rows = [
        row for row in rows
        if row.get("status") == "ok" and row.get("method") == BEST_DEFAULT_METHOD
    ]
    default_by_key = {_promotion_pair_key(row): row for row in default_rows}
    out: list[dict[str, Any]] = []
    for method_idx, method in enumerate(methods):
        if method == BEST_DEFAULT_METHOD:
            continue
        pairs = []
        for row in rows:
            if row.get("status") != "ok" or row.get("method") != method:
                continue
            base = default_by_key.get(_promotion_pair_key(row))
            if base is not None:
                pairs.append((base, row))
        if not pairs:
            continue

        # Strict dominance is a joint statement, so every core metric must use the same
        # complete paired cells. Optional LPIPS remains separately missing-safe.
        complete_pairs = []
        for base, candidate in pairs:
            values = [base.get(metric) for metric in DOMINANCE_METRICS]
            values += [candidate.get(metric) for metric in DOMINANCE_METRICS]
            try:
                complete = all(value is not None and math.isfinite(float(value)) for value in values)
            except (TypeError, ValueError):
                complete = False
            if complete:
                complete_pairs.append((base, candidate))

        record: dict[str, Any] = {
            "method": method,
            "method_label": METHOD_LABELS[method],
            "track": METHOD_TRACKS[method],
            "pairs": len(complete_pairs),
            "budget_matched": method not in over_budget,
        }
        core_intervals: list[tuple[float, float, float]] = []
        familywise_intervals: list[tuple[float, float, float]] = []
        paired_images: list[int] = []
        for metric_idx, (metric, direction) in enumerate(DOMINANCE_METRICS.items()):
            center, low, high, metric_pairs, images = _clustered_gain_ci(
                complete_pairs,
                metric,
                direction,
                seed=DOMINANCE_BOOTSTRAP_SEED + method_idx * 101 + metric_idx,
            )
            fw_center, fw_low, fw_high, _fw_pairs, _fw_images = _clustered_gain_ci(
                complete_pairs,
                metric,
                direction,
                seed=DOMINANCE_BOOTSTRAP_SEED + method_idx * 101 + metric_idx,
                alpha=0.05 / len(DOMINANCE_METRICS),
            )
            record[f"gain_{metric}"] = center
            record[f"ci_low_{metric}"] = low
            record[f"ci_high_{metric}"] = high
            record[f"pairs_{metric}"] = metric_pairs
            if center is not None and low is not None and high is not None:
                core_intervals.append((center, low, high))
                paired_images.append(images)
            if fw_center is not None and fw_low is not None and fw_high is not None:
                familywise_intervals.append((fw_center, fw_low, fw_high))

        lpips_center, lpips_low, lpips_high, lpips_pairs, lpips_images = _clustered_gain_ci(
            pairs,
            "lpips",
            -1.0,
            seed=DOMINANCE_BOOTSTRAP_SEED + method_idx * 101 + len(DOMINANCE_METRICS),
        )
        record.update({
            "gain_lpips": lpips_center,
            "ci_low_lpips": lpips_low,
            "ci_high_lpips": lpips_high,
            "pairs_lpips": lpips_pairs,
            "images_lpips": lpips_images,
            "paired_images": min(paired_images) if paired_images else 0,
        })

        means = [interval[0] for interval in core_intervals]
        if method in over_budget:
            sample_relation = "over_budget"
            supported_relation = "not_comparable"
        elif len(means) != len(DOMINANCE_METRICS):
            sample_relation = "incomplete"
            supported_relation = "incomplete"
        elif all(value > 0.0 for value in means):
            sample_relation = "candidate_dominates"
            supported_relation = (
                "candidate_dominates"
                if len(familywise_intervals) == len(DOMINANCE_METRICS)
                and all(low > 0.0 for _center, low, _high in familywise_intervals)
                else "inconclusive"
            )
        elif all(value < 0.0 for value in means):
            sample_relation = "default_dominates"
            supported_relation = (
                "default_dominates"
                if len(familywise_intervals) == len(DOMINANCE_METRICS)
                and all(high < 0.0 for _center, _low, high in familywise_intervals)
                else "inconclusive"
            )
        else:
            sample_relation = "tradeoff"
            supported_relation = "tradeoff"
        record["sample_relation"] = sample_relation
        record["supported_relation_95ci"] = supported_relation
        out.append(record)
    return out


def _write_default_dominance(
    rows: list[dict[str, Any]],
    outdir: Path,
    methods: list[str],
) -> None:
    audit = _default_dominance_audit(rows, methods)
    if not audit:
        (outdir / "default_dominance.csv").unlink(missing_ok=True)
        return
    metric_fields = []
    for metric in [*DOMINANCE_METRICS, "lpips"]:
        metric_fields.extend([
            f"gain_{metric}",
            f"ci_low_{metric}",
            f"ci_high_{metric}",
            f"pairs_{metric}",
        ])
    write_csv(
        outdir / "default_dominance.csv",
        audit,
        fieldnames=[
            "method",
            "method_label",
            "track",
            "pairs",
            "paired_images",
            "budget_matched",
            *metric_fields,
            "images_lpips",
            "sample_relation",
            "supported_relation_95ci",
        ],
    )


def _promotion_checks(
    rows: list[dict[str, Any]],
    methods: list[str],
    over_budget: set[str],
) -> list[dict[str, Any]]:
    default_rows = [r for r in rows if r.get("method") == BEST_DEFAULT_METHOD]
    default_by_key = {_promotion_pair_key(r): r for r in default_rows}
    out: list[dict[str, Any]] = []
    for method in methods:
        if (
            method == BEST_DEFAULT_METHOD
            or method not in BEST_VARIANT_METHODS
            or method in over_budget
        ):
            continue
        pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
        for row in rows:
            if row.get("method") != method:
                continue
            base = default_by_key.get(_promotion_pair_key(row))
            if base is not None:
                pairs.append((base, row))
        if not pairs:
            continue
        d_psnr = [float(row["psnr"]) - float(base["psnr"]) for base, row in pairs]
        d_ms = [float(row["ms_ssim"]) - float(base["ms_ssim"]) for base, row in pairs]
        d_auc = [float(row["auc_psnr"]) - float(base["auc_psnr"]) for base, row in pairs]
        d_fit = [float(row["fit_seconds"]) - float(base["fit_seconds"]) for base, row in pairs]
        d_total = [
            float(row["total_seconds"]) - float(base["total_seconds"])
            for base, row in pairs
        ]
        promote = (
            mean(d_psnr) > 0.0
            and mean(d_ms) > 0.0
            and mean(d_auc) > 0.0
            and mean(d_fit) < 0.0
            and mean(d_total) < 0.0
        )
        out.append({
            "method": method,
            "pairs": len(pairs),
            "d_psnr": mean(d_psnr),
            "d_ms_ssim": mean(d_ms),
            "d_auc": mean(d_auc),
            "d_fit_seconds": mean(d_fit),
            "d_total_seconds": mean(d_total),
            "psnr_wins": sum(1 for v in d_psnr if v > 0.0),
            "ms_ssim_wins": sum(1 for v in d_ms if v > 0.0),
            "auc_wins": sum(1 for v in d_auc if v > 0.0),
            "faster_fit": sum(1 for v in d_fit if v < 0.0),
            "promote": promote,
        })
    return out


def _checkpoint_selection_audit(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Within-trajectory gain from selecting a same-final-count checkpoint."""
    audit = []
    for row in rows:
        if (
            row.get("status") != "ok"
            or row.get("checkpoint_policy") != "best_psnr_final_count"
        ):
            continue
        selected_n = row.get("selected_n_gaussians")
        terminal_n = row.get("trajectory_terminal_n_gaussians")
        if selected_n is None or terminal_n is None or int(selected_n) != int(terminal_n):
            continue

        def gain(selected_name: str, terminal_name: str, direction: float = 1.0):
            selected = row.get(selected_name)
            terminal = row.get(terminal_name)
            if selected is None or terminal is None:
                return None
            return direction * (float(selected) - float(terminal))

        audit.append({
            "image": row.get("image"),
            "source_path": row.get("source_path"),
            "final_budget": row.get("final_budget"),
            "seed": row.get("seed"),
            "method": row.get("method"),
            "terminal_iter": row.get("trajectory_terminal_iter"),
            "selected_iter": row.get("selected_iter"),
            "terminal_n_gaussians": int(terminal_n),
            "selected_n_gaussians": int(selected_n),
            "selected_from_checkpoint": bool(row.get("selected_from_checkpoint", False)),
            "terminal_psnr": row.get("trajectory_terminal_psnr"),
            "selected_psnr": row.get("selected_fit_psnr"),
            "gain_psnr": gain("selected_fit_psnr", "trajectory_terminal_psnr"),
            "terminal_ssim": row.get("trajectory_terminal_ssim"),
            "selected_ssim": row.get("selected_fit_ssim"),
            "gain_ssim": gain("selected_fit_ssim", "trajectory_terminal_ssim"),
            "terminal_ms_ssim": row.get("trajectory_terminal_ms_ssim"),
            "selected_ms_ssim": row.get("selected_fit_ms_ssim"),
            "gain_ms_ssim": gain(
                "selected_fit_ms_ssim", "trajectory_terminal_ms_ssim"
            ),
            "terminal_lpips": row.get("trajectory_terminal_lpips"),
            "selected_lpips": row.get("selected_fit_lpips"),
            "gain_lpips": gain(
                "selected_fit_lpips", "trajectory_terminal_lpips", direction=-1.0
            ),
        })
    return audit


def _paired_method_audit(
    rows: list[dict[str, Any]],
    *,
    baseline_method: str,
    candidate_method: str,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    """Direct paired gains for one candidate over its causal method control."""
    baseline_by_key = {
        _promotion_pair_key(row): row
        for row in rows
        if row.get("status") == "ok" and row.get("method") == baseline_method
    }
    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    records: list[dict[str, Any]] = []
    metric_directions = {**DOMINANCE_METRICS, "lpips": -1.0}
    terminal_metrics = {
        "terminal_psnr": ("trajectory_terminal_psnr", 1.0),
        "terminal_ssim": ("trajectory_terminal_ssim", 1.0),
        "terminal_ms_ssim": ("trajectory_terminal_ms_ssim", 1.0),
        "terminal_lpips": ("trajectory_terminal_lpips", -1.0),
    }

    def causal_invariant_errors(
        baseline: dict[str, Any], candidate: dict[str, Any]
    ) -> list[str]:
        errors = []
        for label, row in (("baseline", baseline), ("candidate", candidate)):
            if row.get("checkpoint_policy") != "best_psnr_final_count":
                errors.append(f"{label} checkpoint policy is not best_psnr_final_count")
            requested_start = row.get("start_budget")
            actual_start = row.get("start_gaussians")
            requested_final = row.get("final_budget")
            actual_final = row.get("n_gaussians")
            terminal_n = row.get("trajectory_terminal_n_gaussians")
            selected_n = row.get("selected_n_gaussians")
            if requested_start is None or actual_start is None or int(actual_start) != int(
                requested_start
            ):
                errors.append(f"{label} actual start count does not match the request")
            if requested_final is None or actual_final is None or int(actual_final) != int(
                requested_final
            ):
                errors.append(f"{label} actual final count does not match the cap")
            if (
                requested_final is None
                or terminal_n is None
                or selected_n is None
                or int(terminal_n) != int(requested_final)
                or int(selected_n) != int(requested_final)
            ):
                errors.append(f"{label} terminal/selected counts do not match the cap")

        baseline_cfg = baseline.get("fit_config")
        candidate_cfg = candidate.get("fit_config")
        if not isinstance(baseline_cfg, dict) or not isinstance(candidate_cfg, dict):
            errors.append("both rows must carry resolved fit_config dictionaries")
        else:
            baseline_treatment = (
                baseline_cfg.get("loss_target_downsample", 1),
                baseline_cfg.get("loss_target_full_frac", 0.0),
            )
            candidate_treatment = (
                candidate_cfg.get("loss_target_downsample", 1),
                candidate_cfg.get("loss_target_full_frac", 0.0),
            )
            expected_candidate = BEST_LOWPASS_TARGET_METHODS.get(candidate_method)
            if baseline_treatment != (1, 0.0):
                errors.append(f"baseline treatment fields are {baseline_treatment!r}, not neutral")
            if expected_candidate is None or not (
                int(candidate_treatment[0]) == int(expected_candidate[0])
                and math.isclose(
                    float(candidate_treatment[1]),
                    float(expected_candidate[1]),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
            ):
                errors.append(
                    f"candidate treatment fields are {candidate_treatment!r}, not "
                    f"{expected_candidate!r}"
                )
            baseline_control = dict(baseline_cfg)
            candidate_control = dict(candidate_cfg)
            for name in ("loss_target_downsample", "loss_target_full_frac"):
                baseline_control.pop(name, None)
                candidate_control.pop(name, None)
            if json.dumps(baseline_control, sort_keys=True, default=str) != json.dumps(
                candidate_control, sort_keys=True, default=str
            ):
                errors.append("resolved fit configs differ outside the two treatment fields")
        return errors

    for candidate in rows:
        if candidate.get("status") != "ok" or candidate.get("method") != candidate_method:
            continue
        baseline = baseline_by_key.get(_promotion_pair_key(candidate))
        if baseline is None:
            continue
        invariant_errors = causal_invariant_errors(baseline, candidate)
        if invariant_errors:
            raise ValueError(
                "invalid causal low-pass/checkpoint pair for "
                f"{candidate.get('source_path')} seed={candidate.get('seed')}: "
                + "; ".join(invariant_errors)
            )
        pairs.append((baseline, candidate))
        record: dict[str, Any] = {
            "image": candidate.get("image"),
            "source_path": candidate.get("source_path"),
            "target_pixel_sha256": candidate.get("target_pixel_sha256"),
            "max_side": candidate.get("max_side"),
            "height": candidate.get("height"),
            "width": candidate.get("width"),
            "final_budget": candidate.get("final_budget"),
            "start_budget": candidate.get("start_budget"),
            "seed": candidate.get("seed"),
            "iters": candidate.get("iters"),
            "baseline_method": baseline_method,
            "candidate_method": candidate_method,
            "causal_invariants_verified": True,
            "baseline_selected_iter": baseline.get("selected_iter"),
            "candidate_selected_iter": candidate.get("selected_iter"),
        }
        for metric, direction in metric_directions.items():
            baseline_value = baseline.get(metric)
            candidate_value = candidate.get(metric)
            record[f"baseline_{metric}"] = baseline_value
            record[f"candidate_{metric}"] = candidate_value
            try:
                gain = (
                    None
                    if baseline_value is None or candidate_value is None
                    else direction * (float(candidate_value) - float(baseline_value))
                )
            except (TypeError, ValueError):
                gain = None
            record[f"gain_{metric}"] = gain
        for output_name, (row_name, direction) in terminal_metrics.items():
            baseline_value = baseline.get(row_name)
            candidate_value = candidate.get(row_name)
            record[f"baseline_{output_name}"] = baseline_value
            record[f"candidate_{output_name}"] = candidate_value
            try:
                gain = (
                    None
                    if baseline_value is None or candidate_value is None
                    else direction * (float(candidate_value) - float(baseline_value))
                )
            except (TypeError, ValueError):
                gain = None
            record[f"gain_{output_name}"] = gain
        records.append(record)
    if not pairs:
        return [], None

    complete_pairs = []
    for baseline, candidate in pairs:
        values = [baseline.get(metric) for metric in DOMINANCE_METRICS]
        values += [candidate.get(metric) for metric in DOMINANCE_METRICS]
        try:
            complete = all(
                value is not None and math.isfinite(float(value)) for value in values
            )
        except (TypeError, ValueError):
            complete = False
        if complete:
            complete_pairs.append((baseline, candidate))

    summary: dict[str, Any] = {
        "baseline_method": baseline_method,
        "candidate_method": candidate_method,
        "pairs": len(complete_pairs),
    }
    core_images: list[int] = []
    for metric_idx, (metric, direction) in enumerate(metric_directions.items()):
        metric_pairs = pairs if metric == "lpips" else complete_pairs
        center, low, high, count, images = _clustered_gain_ci(
            metric_pairs,
            metric,
            direction,
            seed=DOMINANCE_BOOTSTRAP_SEED + 700 + metric_idx,
        )
        summary[f"gain_{metric}"] = center
        summary[f"ci_low_{metric}"] = low
        summary[f"ci_high_{metric}"] = high
        summary[f"pairs_{metric}"] = count
        summary[f"images_{metric}"] = images
        if metric != "lpips":
            core_images.append(images)
    for metric_idx, (output_name, (row_name, direction)) in enumerate(
        terminal_metrics.items()
    ):
        center, low, high, count, images = _clustered_gain_ci(
            pairs,
            output_name,
            direction,
            seed=DOMINANCE_BOOTSTRAP_SEED + 800 + metric_idx,
            baseline_field=row_name,
            candidate_field=row_name,
        )
        summary[f"gain_{output_name}"] = center
        summary[f"ci_low_{output_name}"] = low
        summary[f"ci_high_{output_name}"] = high
        summary[f"pairs_{output_name}"] = count
        summary[f"images_{output_name}"] = images
    summary["paired_images"] = min(core_images) if core_images else 0
    return records, summary


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
    else:
        (outdir / "convergence_curves.csv").unlink(missing_ok=True)

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
    else:
        (outdir / "target_hit_rates.csv").unlink(missing_ok=True)


def _write_summary(rows: list[dict[str, Any]], outdir: Path, methods: list[str]) -> None:
    ok = [r for r in rows if r.get("status") == "ok"]

    # Convergence sampling adapts to the iteration budget actually logged in this run
    # (the fitter logs up to iters-1). Hardcoded sample points silently degrade to the
    # final-iter value when a run uses fewer iters than the template assumed, producing
    # duplicate columns and a misstated budget in the caption.
    last_iters = [pairs[-1][0] for r in ok if (pairs := _history_pairs(r))]
    budget_iter = max(last_iters) if last_iters else 0
    conv_points = sorted({0, *(int(round(budget_iter * f)) for f in (0.25, 0.5, 0.75))})

    # Methods whose achieved Gaussian count exceeds the shared final cap are not
    # rate-matched to the equal-budget rows (e.g. adaptive extra-capacity variants),
    # so their quality numbers are not directly comparable. Flag them with a dagger and
    # keep them out of the per-cell winners. Empty for a standard equal-budget run.
    over_budget = _over_budget_methods(ok, methods)

    def _label(method: str) -> str:
        return METHOD_LABELS[method] + (" †" if method in over_budget else "")

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
            f"| {_label(method)} | {len(vals)} | "
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

    if over_budget:
        lines += [
            "",
            "† Not budget-matched: mean final Gaussian count exceeds the shared final cap "
            "(adaptive extra capacity). These rows spend more primitives — more rate for an "
            "image codec — so their PSNR/MS-SSIM is not directly comparable to the equal-budget "
            "rows, and they are excluded from the per-cell winners below.",
        ]

    checkpoint_audit = _checkpoint_selection_audit(ok)
    if checkpoint_audit:
        lines += [
            "",
            "## Within-Trajectory Checkpoint Audit",
            "",
            "Each delta compares the selected state with that same run's terminal state; both "
            "have the terminal Gaussian count. Positive means the selected state is better, "
            "including LPIPS (positive means lower). This avoids attributing independent CUDA "
            "trajectory divergence to checkpoint selection. Per-cell values are in "
            "`checkpoint_selection.csv`.",
            "",
            "| Method | Runs | Earlier states selected | PSNR gain | SSIM gain | MS-SSIM gain | LPIPS gain |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for method in sorted({str(row["method"]) for row in checkpoint_audit}):
            method_rows = [row for row in checkpoint_audit if row["method"] == method]
            selected = sum(bool(row["selected_from_checkpoint"]) for row in method_rows)
            lines.append(
                f"| {_label(method)} | {len(method_rows)} | {selected} | "
                f"{_fmt(_mean_or_none([r['gain_psnr'] for r in method_rows]), 4)} | "
                f"{_fmt(_mean_or_none([r['gain_ssim'] for r in method_rows]), 5)} | "
                f"{_fmt(_mean_or_none([r['gain_ms_ssim'] for r in method_rows]), 5)} | "
                f"{_fmt(_mean_or_none([r['gain_lpips'] for r in method_rows]), 4)} |"
            )

    _lowpass_pairs, lowpass_summary = _paired_method_audit(
        ok,
        baseline_method=BEST_CHECKPOINT_METHOD,
        candidate_method=BEST_CHECKPOINT_LOWPASS_METHOD,
    )
    if lowpass_summary is not None:
        lines += [
            "",
            "## Low-Pass Curriculum vs Checkpoint Control",
            "",
            "This is the causal method comparison: both arms use terminal-count checkpoint "
            "selection and differ only in the two loss-target curriculum fields. Positive gains "
            "favor the low-pass candidate; positive time gains mean faster and positive LPIPS "
            "gain means lower. Intervals bootstrap source images after averaging their correlated "
            "seeds/budgets. Per-cell and aggregate values are in `lowpass_vs_checkpoint.csv` "
            "and `lowpass_vs_checkpoint_summary.csv`.",
            "",
            "| Pairs / images | PSNR gain [95% CI] | MS-SSIM gain [95% CI] | AUC gain [95% CI] | Fit gain s [95% CI] | Total gain s [95% CI] | LPIPS gain [95% CI] |",
            "|---:|---:|---:|---:|---:|---:|---:|",
            f"| {lowpass_summary['pairs']} / {lowpass_summary['paired_images']} | "
            f"{_fmt_gain_ci(lowpass_summary, 'psnr', 4)} | "
            f"{_fmt_gain_ci(lowpass_summary, 'ms_ssim', 5)} | "
            f"{_fmt_gain_ci(lowpass_summary, 'auc_psnr', 4)} | "
            f"{_fmt_gain_ci(lowpass_summary, 'fit_seconds', 4)} | "
            f"{_fmt_gain_ci(lowpass_summary, 'total_seconds', 4)} | "
            f"{_fmt_gain_ci(lowpass_summary, 'lpips', 4)} |",
            "",
            "Terminal (unselected) endpoint deltas from the same paired trajectories:",
            "",
            "| Terminal PSNR gain [95% CI] | Terminal SSIM gain [95% CI] | Terminal MS-SSIM gain [95% CI] | Terminal LPIPS gain [95% CI] |",
            "|---:|---:|---:|---:|",
            f"| {_fmt_gain_ci(lowpass_summary, 'terminal_psnr', 4)} | "
            f"{_fmt_gain_ci(lowpass_summary, 'terminal_ssim', 5)} | "
            f"{_fmt_gain_ci(lowpass_summary, 'terminal_ms_ssim', 5)} | "
            f"{_fmt_gain_ci(lowpass_summary, 'terminal_lpips', 4)} |",
        ]

    dominance_rows = _default_dominance_audit(ok, methods, over_budget)
    headline_methods = BEST_VARIANT_METHODS | {
        "gaussianimage_fixed_full",
        "gaussianimage_plus_residual",
        "image_gs_residual",
        "instant_gi_quadtree_fixed",
    }
    headline_dominance = [row for row in dominance_rows if row["method"] in headline_methods]
    if headline_dominance:
        lines += [
            "",
            "## Paired Strict-Dominance Audit",
            "",
            "Every delta is a candidate gain over `SS best default`; positive is better, "
            "including fit/total-time gains (positive means faster) and LPIPS gain "
            "(positive means lower LPIPS). Confidence intervals bootstrap source images after "
            "averaging correlated seeds/budgets within each image. A tradeoff is not a dominance "
            "result, and over-budget rows are not comparable. Displayed intervals are marginal "
            "95% image-bootstrap intervals; the relation column uses Bonferroni-adjusted bounds "
            "for 95% familywise coverage across the five core metrics. Full rows are in "
            "`default_dominance.csv`.",
            "",
            "| Candidate | Pairs / images | PSNR gain [95% CI] | MS-SSIM gain [95% CI] | AUC gain [95% CI] | Fit gain s [95% CI] | Total gain s [95% CI] | LPIPS gain [95% CI] | Sample relation | Familywise 95% relation |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---|---|",
        ]
        for row in headline_dominance:
            sample_relation = str(row["sample_relation"]).replace("_", " ")
            supported_relation = str(row["supported_relation_95ci"]).replace("_", " ")
            lines.append(
                f"| {_label(row['method'])} | {row['pairs']} / {row['paired_images']} | "
                f"{_fmt_gain_ci(row, 'psnr', 4)} | "
                f"{_fmt_gain_ci(row, 'ms_ssim', 5)} | "
                f"{_fmt_gain_ci(row, 'auc_psnr', 4)} | "
                f"{_fmt_gain_ci(row, 'fit_seconds', 4)} | "
                f"{_fmt_gain_ci(row, 'total_seconds', 4)} | "
                f"{_fmt_gain_ci(row, 'lpips', 4)} | "
                f"{sample_relation} | {supported_relation} |"
            )
        lines += [
            "",
            "The strict-dominance core uses PSNR, MS-SSIM, AUC, fit seconds, and total "
            "seconds. LPIPS is reported when enabled but is not silently imputed or added to "
            "the gate. These rows compare policies under StructSplat's harness; analogue labels "
            "do not make them native external-repository results.",
        ]

    promotion_rows = _promotion_checks(ok, methods, over_budget)
    if promotion_rows:
        lines += [
            "",
            "## Default Promotion Check",
            "",
            "A best-default candidate is promotable only when its paired mean deltas beat "
            "`SS best default` on quality (PSNR and MS-SSIM), convergence (AUC), and "
            "performance (fit and total seconds). Over-budget rows are excluded.",
            "",
            "| Candidate | Pairs | ΔPSNR | ΔMS-SSIM | ΔAUC | ΔFit s | ΔTotal s | PSNR wins | MS wins | AUC wins | Faster fit | Promote |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
        for row in promotion_rows:
            pairs = int(row["pairs"])
            lines.append(
                f"| {METHOD_LABELS[row['method']]} | {pairs} | "
                f"{row['d_psnr']:+.4f} | "
                f"{row['d_ms_ssim']:+.5f} | "
                f"{row['d_auc']:+.4f} | "
                f"{row['d_fit_seconds']:+.4f} | "
                f"{row['d_total_seconds']:+.4f} | "
                f"{row['psnr_wins']}/{pairs} | "
                f"{row['ms_ssim_wins']}/{pairs} | "
                f"{row['auc_wins']}/{pairs} | "
                f"{row['faster_fit']}/{pairs} | "
                f"{'yes' if row['promote'] else 'no'} |"
            )

    targets = [t for t in _target_values(ok) if t in set(HEADLINE_TARGET_PSNRS)]
    if targets:
        target_header = "".join(
            f" | Hit {target_label(t)} | Iter {target_label(t)}"
            for t in targets
        )
        target_align = "|---:" * (2 * len(targets))
        conv_header = " | ".join(f"PSNR@{pt}" for pt in conv_points)
        conv_align = "|---:" * (len(conv_points) + 2)  # AUC + sample points + Final
        lines += [
            "",
            "## Convergence",
            "",
            "AUC is the area under the logged PSNR-over-iteration curve; higher means better "
            f"quality earlier in the same {budget_iter}-iteration budget.",
            "",
            f"| Method | AUC | {conv_header} | Final PSNR |",
            f"|---{conv_align}|",
        ]
        for method in methods:
            vals = [r for r in ok if r["method"] == method]
            cols = " | ".join(
                _fmt(_mean_or_none([_psnr_at_or_after(r, pt) for r in vals]), 3)
                for pt in conv_points
            )
            lines.append(
                f"| {_label(method)} | "
                f"{_fmt(_mean_or_none([r.get('auc_psnr') for r in vals]), 3)} | "
                f"{cols} | "
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
            lines.append(f"| {_label(method)} | " + " | ".join(cells) + " |")

    lines += [
        "",
        "## Means By Budget",
        "",
        "| Final budget | Method | Start G | Final G | PSNR | PSNR Std | MS-SSIM | AUC | Fit s |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for (budget, method), vals in sorted(_groups(ok, ("final_budget", "method")).items()):
        lines.append(
            f"| {int(budget)} | {_label(method)} | "
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
    ]
    if over_budget:
        lines += [
            "Winners are taken among budget-matched methods only; † rows (over the shared cap) are excluded.",
            "",
        ]
    lines += [
        "| Image | Budget | Best PSNR | Best MS-SSIM |",
        "|---|---:|---|---|",
    ]
    for (image, budget), vals in sorted(_groups(ok, ("image", "final_budget")).items()):
        matched = [r for r in vals if r["method"] not in over_budget] or vals
        best_p = max(matched, key=lambda r: r["psnr"])
        best_m = max(matched, key=lambda r: r["ms_ssim"])
        lines.append(
            f"| {image} | {int(budget)} | {_label(best_p['method'])} ({best_p['psnr']:.3f}) | "
            f"{_label(best_m['method'])} ({best_m['ms_ssim']:.5f}) |"
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
    file_links = [
        '<a href="summary.md">summary.md</a>',
        '<a href="metrics.csv">metrics.csv</a>',
        '<a href="metrics.json">metrics.json</a>',
    ]
    for name in ("default_dominance.csv", "convergence_curves.csv", "target_hit_rates.csv"):
        if (outdir / name).exists():
            file_links.append(f'<a href="{name}">{name}</a>')
    if (outdir / "checkpoint_selection.csv").exists():
        file_links.append('<a href="checkpoint_selection.csv">checkpoint_selection.csv</a>')
    if (outdir / "lowpass_vs_checkpoint.csv").exists():
        file_links += [
            '<a href="lowpass_vs_checkpoint.csv">lowpass_vs_checkpoint.csv</a>',
            '<a href="lowpass_vs_checkpoint_summary.csv">lowpass_vs_checkpoint_summary.csv</a>',
        ]
    file_links.append('<a href="config.json">config.json</a>')
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
        f"<p>{' · '.join(file_links)}</p>",
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

    repo_state = repository_state()
    source_fingerprint = _source_fingerprint()
    scientific_protocol = {
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
        "flat_frac": args.flat_frac,
        "corner_frac": args.corner_frac,
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
    }
    instant_gi_source = os.environ.get("STRUCTSPLAT_INSTANT_GI")
    if any(method.startswith("instant_gi_") for method in methods):
        instant_gi_path = None if not instant_gi_source else Path(instant_gi_source).resolve()
        scientific_protocol["instant_gi_source"] = (
            None if instant_gi_path is None else str(instant_gi_path)
        )
        scientific_protocol["instant_gi_source_sha256"] = (
            None
            if instant_gi_path is None or not instant_gi_path.is_file()
            else _file_sha256(instant_gi_path)
        )
    run_protocol_sha256 = _run_protocol_sha256(
        scientific_protocol,
        device=device,
        versions=package_versions(),
        repo_state=repo_state,
        source_fingerprint=source_fingerprint,
    )
    row_provenance = {
        "repository_commit": repo_state.get("commit"),
        "repository_tracked_diff_sha256": repo_state.get("tracked_diff_sha256"),
        **source_fingerprint,
        "run_protocol_sha256": run_protocol_sha256,
    }
    write_config(str(outdir), run_config({
        **scientific_protocol,
        "resume": args.resume,
        "max_new_cells": args.max_new_cells,
        **row_provenance,
    }, device=device))

    jsonl_path = outdir / "metrics.jsonl"
    existing = _load_jsonl(jsonl_path) if args.resume else []
    if not args.resume and jsonl_path.exists():
        jsonl_path.unlink()
    existing_by_key = {
        _cell_key(row): row for row in existing if row.get("status") == "ok"
    }
    rows: list[dict[str, Any]] = []
    selected_keys: set[tuple[Any, ...]] = set()

    loaded: dict[str, tuple[np.ndarray, Path, str, str, str]] = {}
    for image_path in args.images:
        image_path = Path(image_path).resolve()
        img = load_image(image_path, max_side=args.max_side)
        source_sha256 = _file_sha256(image_path)
        source_id = _artifact_source_id(image_path, source_sha256)
        target_path = target_dir / f"{source_id}.png"
        save_image(img, target_path)
        loaded[str(image_path)] = (
            img,
            target_path,
            source_sha256,
            _pixel_sha256(img),
            source_id,
        )

    scfg = StructureTensorConfig(flat_frac=args.flat_frac, corner_frac=args.corner_frac)
    base_fit = _base_fit(args)
    total = len(args.images) * len(budgets) * len(seeds) * len(methods)
    new_cells = 0
    cell_idx = 0
    for image_path in args.images:
        image_path = Path(image_path).resolve()
        img, target_path, source_sha256, target_pixel_sha256, source_id = loaded[
            str(image_path)
        ]
        image = source_id
        target = target_tensor(img, device)
        for final_budget in budgets:
            start_budget = _start_budget(final_budget, args.start_fraction)
            for seed in seeds:
                for method in methods:
                    resolved_growth_waves = _method_growth_waves(method, args.growth_waves)
                    fit_sig = _method_fit_signature(
                        method, base_fit, final_budget, start_budget, args.growth_waves
                    )
                    feature_cap_px = (
                        _feature_cap_pixels(
                            img,
                            _method_feature_cap(method, args.feature_cap),
                            _method_feature_cap_reference_side(
                                method, args.feature_cap_reference_side
                            ),
                        )
                        if method in FEATURE_CAP_METHODS else None
                    )
                    cell_idx += 1
                    key_row = {
                        **row_provenance,
                        "image": image,
                        "image_stem": image_path.stem,
                        "source_path": str(image_path),
                        "source_sha256": source_sha256,
                        "target_pixel_sha256": target_pixel_sha256,
                        "max_side": args.max_side,
                        "final_budget": final_budget,
                        "start_budget": start_budget,
                        "start_fraction": args.start_fraction,
                        "growth_waves": resolved_growth_waves,
                        "seed": seed,
                        "method": method,
                        "iters": args.iters,
                        "renderer": args.renderer,
                        "support_fade": args.support_fade,
                        "pixel_loss": fit_sig.pixel_loss,
                        "ssim_weight": fit_sig.ssim_weight,
                        "loss_weighting": fit_sig.loss_weighting,
                        "geometry_loss_weight": fit_sig.geometry_loss_weight,
                        "geometry_loss_every": fit_sig.geometry_loss_every,
                        "color_solve_schedule": fit_sig.color_solve_schedule,
                        "checkpoint_policy": fit_sig.checkpoint_policy,
                        "loss_target_downsample": fit_sig.loss_target_downsample,
                        "loss_target_full_frac": fit_sig.loss_target_full_frac,
                        "adaptive_count": fit_sig.adaptive_count,
                        "variant_max_gaussians": fit_sig.max_gaussians
                        if fit_sig.adaptive_count else None,
                        "feature_cap_px": feature_cap_px,
                    }
                    key = _cell_key(key_row)
                    if key in selected_keys:
                        print(
                            f"[{cell_idx}/{total}] skip duplicate requested cell "
                            f"{image} {final_budget} {method}",
                            flush=True,
                        )
                        continue
                    cached = existing_by_key.get(key)
                    if cached is not None:
                        rows.append(cached)
                        selected_keys.add(key)
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
                        "variant_max_gaussians": key_row["variant_max_gaussians"],
                        "loss_weighting": fit_sig.loss_weighting,
                        "loss_weight_beta": fit_sig.loss_weight_beta,
                        "geometry_loss_weight": fit_sig.geometry_loss_weight,
                        "geometry_loss_every": fit_sig.geometry_loss_every,
                        "color_solve_every": fit_sig.color_solve_every,
                        "color_solve_schedule": fit_sig.color_solve_schedule,
                        "adaptive_count": fit_sig.adaptive_count,
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
                        recon_path = (
                            recon_dir / source_id / str(final_budget) / f"seed{seed}_{method}.png"
                        )
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
                    selected_keys.add(key)
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
