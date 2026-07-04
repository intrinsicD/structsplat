"""Shared benchmark helpers (BENCH-003).

This module owns pure benchmark plumbing that had started to drift across harnesses:
image IO, trajectory AUC, simple row writers, and the shared comparison analogues used by
the COCO/cross-repo scripts. It intentionally re-exports the BENCH-002 config/stat helpers
so new benchmark code has one import surface.
"""
from __future__ import annotations

import csv
import importlib.util
import json
import os
import time
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable

import numpy as np
from PIL import Image

from benchmarks._util import (
    add_seed_args,
    fmt_mean_std,
    mean_or_none,
    mean_std_or_none,
    resolve_seeds,
    run_config,
    write_config,
)

if TYPE_CHECKING:
    import torch

    from structsplat.config import FitConfig, StructureTensorConfig
    from structsplat.gaussians import GaussianField

__all__ = [
    "add_seed_args",
    "fmt_mean_std",
    "mean_or_none",
    "mean_std_or_none",
    "resolve_seeds",
    "run_config",
    "write_config",
    "ANALOGUE_METHODS",
    "ANALOGUE_LABELS",
    "ANALOGUE_NOTES",
    "load_image",
    "save_image",
    "target_tensor",
    "bilinear_sample",
    "psnr_auc",
    "json_safe_rows",
    "write_json",
    "write_csv",
    "write_rows",
    "residual_fit_cfg",
    "instant_gi_quard_path",
    "load_instant_gi_module",
    "build_instant_gi_field",
    "build_comparison_analogue",
]


ANALOGUE_METHODS = (
    "gaussianimage",
    "gaussianimage_plus",
    "image_gs",
    "instant_gi_quadtree",
)

ANALOGUE_LABELS = {
    "gaussianimage": "GaussianImage",
    "gaussianimage_plus": "GaussianImage++",
    "image_gs": "Image-GS",
    "instant_gi_quadtree": "Instant-GI qt",
}

ANALOGUE_NOTES = {
    "gaussianimage": "fixed random GaussianImage-style baseline",
    "gaussianimage_plus": "random half-budget init plus high-residual additions",
    "image_gs": "gradient-density init plus residual progressive additions",
    "instant_gi_quadtree": "Instant-GI quadtree/Delaunay fallback; no learned checkpoint",
}


def load_image(path: str | Path, max_side: int | None = None) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    if max_side is not None and max(img.size) > max_side:
        scale = max_side / max(img.size)
        img = img.resize(
            (round(img.size[0] * scale), round(img.size[1] * scale)),
            Image.Resampling.LANCZOS,
        )
    return np.asarray(img, dtype=np.float32) / 255.0


def save_image(arr: np.ndarray, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    img = np.clip(arr * 255.0 + 0.5, 0, 255).astype(np.uint8)
    Image.fromarray(img).save(path)


def target_tensor(img: np.ndarray, device: str) -> torch.Tensor:
    import torch

    return torch.as_tensor(img, dtype=torch.float32, device=device)


def bilinear_sample(img: np.ndarray, pts: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    x = np.clip(pts[:, 0], 0, w - 1.001)
    y = np.clip(pts[:, 1], 0, h - 1.001)
    x0 = np.floor(x).astype(int)
    y0 = np.floor(y).astype(int)
    x1 = np.clip(x0 + 1, 0, w - 1)
    y1 = np.clip(y0 + 1, 0, h - 1)
    fx = (x - x0)[:, None]
    fy = (y - y0)[:, None]
    return (
        img[y0, x0] * (1 - fx) * (1 - fy)
        + img[y0, x1] * fx * (1 - fy)
        + img[y1, x0] * (1 - fx) * fy
        + img[y1, x1] * fx * fy
    ).astype(np.float32)


def psnr_auc(history: dict[str, Any], round_digits: int | None = None) -> float | None:
    """Mean PSNR over the logged fit trajectory."""
    its = history.get("iter", [])
    ps = history.get("psnr", [])
    if not ps:
        return None
    if len(ps) == 1 or len(its) < 2:
        value = float(ps[0])
    else:
        trapezoid = getattr(np, "trapezoid", None) or np.trapz
        value = float(trapezoid(ps, its) / max(its[-1] - its[0], 1))
    return round(value, round_digits) if round_digits is not None else value


def json_safe_rows(rows: Iterable[dict[str, Any]], skip: set[str] | None = None) -> list[dict[str, Any]]:
    skip = skip or set()
    return [{k: v for k, v in row.items() if k not in skip} for row in rows]


def write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def write_csv(
    path: str | Path,
    rows: list[dict[str, Any]],
    *,
    fieldnames: list[str] | None = None,
    sort_fieldnames: bool = False,
    extrasaction: str = "raise",
) -> None:
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        if sort_fieldnames:
            fieldnames = sorted({k for row in rows for k in row.keys()})
        else:
            fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            extrasaction=extrasaction,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def write_rows(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    write_json(path.with_suffix(".json"), rows)
    write_csv(path.with_suffix(".csv"), rows)


def residual_fit_cfg(
    base: FitConfig,
    final_budget: int,
    start_budget: int,
    split_mode: str = "residual_add",
    add_times: int = 4,
) -> FitConfig:
    add_total = max(0, final_budget - start_budget)
    split_count = max(1, int(np.ceil(add_total / max(1, add_times)))) if add_total else 0
    split_every = max(1, base.iters // (add_times + 1)) if add_total else None
    return replace(
        base,
        split_every=split_every,
        split_count=split_count,
        split_mode=split_mode,
        max_gaussians=final_budget,
    )


def instant_gi_quard_path() -> Path:
    env = os.environ.get("STRUCTSPLAT_INSTANT_GI")
    if not env:
        raise RuntimeError(
            "Instant-GI methods require STRUCTSPLAT_INSTANT_GI=/path/to/quard_image.py; "
            "unset -> this method is skipped."
        )
    return Path(env)


def load_instant_gi_module():
    qpath = instant_gi_quard_path()
    spec = importlib.util.spec_from_file_location("instant_gi_quard_image", qpath)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {qpath}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_instant_gi_field(
    img: np.ndarray,
    image_path: str | Path,
    final_budget: int,
    seed: int,
    device: str,
) -> tuple[GaussianField, float]:
    from structsplat.gaussians import GaussianField

    start = time.time()
    points, _elapsed = load_instant_gi_module().QuardImage(image_path).split()
    if points.shape[0] > final_budget:
        rng = np.random.default_rng(seed)
        keep = np.sort(rng.choice(points.shape[0], size=final_budget, replace=False))
        points = points[keep]

    h, w = img.shape[:2]
    means = np.empty((points.shape[0], 2), dtype=np.float32)
    means[:, 0] = (points[:, 0] + 1.0) * 0.5 * (w - 1)
    means[:, 1] = (points[:, 1] + 1.0) * 0.5 * (h - 1)
    scales = np.maximum(points[:, 2:4].astype(np.float32), 0.5)
    rotations = (points[:, 4].astype(np.float32) * (2.0 * np.pi)) % np.pi
    colors = bilinear_sample(img, means)
    field = GaussianField.from_numpy(means, scales, rotations, colors, None, device=device)
    return field, time.time() - start


def build_comparison_analogue(
    method: str,
    img: np.ndarray,
    image_path: str | Path,
    final_budget: int,
    seed: int,
    device: str,
    base_fit: FitConfig,
    scfg_default: StructureTensorConfig | None = None,
) -> tuple[GaussianField, FitConfig, float, int]:
    from structsplat.config import InitConfig, StructureTensorConfig
    from structsplat.init import build_field

    init_start = time.time()
    scfg_default = scfg_default or StructureTensorConfig()

    if method == "gaussianimage":
        field = build_field(
            img,
            InitConfig(strategy="random", num_gaussians=final_budget, seed=seed),
            scfg_default,
            device=device,
        )
        return field, base_fit, time.time() - init_start, final_budget

    if method == "gaussianimage_plus":
        start_budget = max(16, final_budget // 2)
        field = build_field(
            img,
            InitConfig(strategy="random", num_gaussians=start_budget, seed=seed),
            scfg_default,
            device=device,
        )
        fcfg = residual_fit_cfg(base_fit, final_budget, start_budget, "residual_add", add_times=4)
        return field, fcfg, time.time() - init_start, start_budget

    if method == "image_gs":
        start_budget = max(16, final_budget // 2)
        field = build_field(
            img,
            InitConfig(
                strategy="iso_blue_noise",
                num_gaussians=start_budget,
                density_mode="gradient",
                sampling_mode="density_random",
                scale_mode="spacing",
                seed=seed,
            ),
            scfg_default,
            device=device,
        )
        fcfg = residual_fit_cfg(base_fit, final_budget, start_budget, "residual_add", add_times=4)
        return field, fcfg, time.time() - init_start, start_budget

    if method == "instant_gi_quadtree":
        field, init_seconds = build_instant_gi_field(img, image_path, final_budget, seed, device)
        return field, base_fit, init_seconds, field.n

    raise ValueError(f"unknown comparison analogue {method!r}")
