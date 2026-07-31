#!/usr/bin/env python3
"""Run ABL-002's normalized-vs-additive screen with a portable visual report.

Owning task: ``tasks/ABL-002-stage-search.md``.

Diagnostic reproduction command (the output directory must be new):

    env PYTHONPATH=src:. python scripts/experiments/abl002_renderer_report.py \
      tests/test_images \
      results/abl002_additive_visual_coco4_m512_i750_b2k5k_s012_20260731_diagnostic \
      --budgets 2000 5000 --seeds 0 1 2 --iters 750 --max-side 512 \
      --log-every 25 --renderers cuda cuda_additive --lpips --device cuda

This driver exists because the historical ``benchmarks.stage_search`` ABL-002 harness emits
scalar rows but not the field/image/telemetry bundle required for visual experiment handoff.  It
binds one renderer-only protocol and deliberately reuses the maintained workflow report schema.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import html
import json
import math
from pathlib import Path
import shutil
import statistics
import sys
import time
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from structsplat import metrics as M  # noqa: E402
from structsplat.config import FitConfig, InitConfig, StructureTensorConfig  # noqa: E402
from structsplat.fit import fit  # noqa: E402
from structsplat.gaussians import GaussianField  # noqa: E402
from structsplat.init import build_field  # noqa: E402
from structsplat.render import render_field as render_gaussian_field  # noqa: E402
from structsplat.workflows import (  # noqa: E402
    _atomic_json,
    _atomic_text,
    _discover_images,
    _load_rgb,
    _portable_metric_value,
    _relative_link,
    _repository_state,
    _run_card,
    _save_error,
    _save_rgb,
    _sha256_file,
    _svg_curve,
    _write_metrics,
)


PROFILE = "abl002_renderer_screen_v1"
SCOPE = (
    "Diagnostic renderer-only proxy: tracked development images, fixed Gaussian count, raw "
    "fit-space metrics, and an instrumented 512px/750-step trajectory. It cannot promote a "
    "default or support a held-out/public claim."
)
TARGETS = (28.0, 30.0, 32.0)
RENDERERS = ("normalized", "additive", "cuda", "cuda_additive")
FAMILY = {
    "normalized": "normalized",
    "cuda": "normalized",
    "additive": "additive",
    "cuda_additive": "additive",
}
COLORS = {"normalized": "#147d72", "additive": "#e65f2b"}
CURVE_SPECS = (
    ("psnr", "PSNR", "higher is better"),
    ("loss", "optimization objective", "lower is better"),
    ("ssim", "SSIM", "higher is better"),
    ("ms_ssim", "MS-SSIM", "higher is better"),
    ("lpips", "LPIPS", "lower is better"),
    ("mse", "MSE", "lower is better"),
    ("mae", "MAE", "lower is better"),
    ("cvar99_mse", "CVaR99 MSE", "lower is better"),
    ("p99_mse", "p99 MSE", "lower is better"),
    ("interior_hole_fraction", "coverage-hole fraction", "lower is better"),
    ("render_out_of_range_fraction", "out-of-range channel fraction", "lower is better"),
    ("elapsed_seconds", "cumulative fit seconds", "lower is faster"),
)


def _pixel_sha256(array: np.ndarray) -> str:
    pixels = np.rint(np.clip(array, 0.0, 1.0) * 255.0).astype(np.uint8)
    digest = hashlib.sha256()
    digest.update(np.asarray(pixels.shape, dtype="<i8").tobytes())
    digest.update(pixels.tobytes(order="C"))
    return digest.hexdigest()


def _variant(renderer: str, budget: int) -> str:
    return f"{FAMILY[renderer]}_n{int(budget)}"


def _renderer_label(renderer: str) -> str:
    implementation = "exact CUDA" if renderer.startswith("cuda") else "PyTorch reference"
    return f"{FAMILY[renderer].capitalize()} · {implementation}"


def _render(field: GaussianField, cfg: FitConfig, height: int, width: int, *,
            mode: str | None = None, colors=None):
    selected_mode = cfg.renderer if mode is None else mode
    return render_gaussian_field(
        field.means,
        field.conics(cfg.aa_dilation),
        field.colors if colors is None else colors,
        field.radii(cfg.sigma_cutoff, cfg.aa_dilation),
        height,
        width,
        cfg.render_chunk,
        selected_mode,
        field.opacity_values(),
        scales=field.effective_scales(0.0),
        rotations=field.rotations,
        support_fade=cfg.support_fade,
        sigma_cutoff=cfg.sigma_cutoff,
        support_fade_alpha=0.0,
    )


def _coverage_mode(renderer: str) -> str:
    return "cuda_additive" if renderer.startswith("cuda") else "additive"


def _curve_point(field: GaussianField, target, cfg: FitConfig, *, step: int,
                 elapsed_seconds: float, lpips: bool, observer_loss: float | None) -> tuple[dict, Any]:
    import torch

    height, width = target.shape[:2]
    with torch.no_grad():
        prediction = _render(field, cfg, height, width)
        residual = prediction - target
        pixel_mse = residual.square().mean(dim=2).reshape(-1)
        mse = pixel_mse.mean()
        p99 = torch.quantile(pixel_mse, 0.99)
        tail = pixel_mse[pixel_mse >= p99]
        ssim_value = M.ssim(prediction, target, backend=cfg.ssim_backend)
        objective = (
            (1.0 - float(cfg.ssim_weight)) * residual.abs().mean()
            + float(cfg.ssim_weight) * (1.0 - ssim_value)
        )
        coverage = _render(
            field,
            cfg,
            height,
            width,
            mode=_coverage_mode(cfg.renderer),
            colors=torch.ones_like(field.colors),
        )[..., 0]
        colors = field.colors
        point = {
            "phase": "fit",
            "event": "initial" if step == 0 else "checkpoint",
            "accepted": True,
            "attempted_steps": int(step),
            "accepted_steps": int(step),
            "elapsed_seconds": float(elapsed_seconds),
            "n_gaussians": int(field.n),
            "psnr": float(M.psnr_from_mse(mse)),
            "ssim": float(ssim_value),
            "ms_ssim": float(M.ms_ssim(prediction, target)),
            "lpips": M.LPIPS.distance(prediction, target) if lpips else None,
            "mse": float(mse),
            "mae": float(residual.abs().mean()),
            "max_abs": float(residual.abs().max()),
            "cvar99_mse": float(tail.mean()),
            "p99_mse": float(p99),
            "interior_hole_fraction": float((coverage < 0.1).float().mean()),
            "boundary_hole_fraction": None,
            "loss": float(objective),
            "observer_loss": None if observer_loss is None else float(observer_loss),
            "render_min": float(prediction.min()),
            "render_max": float(prediction.max()),
            "render_out_of_range_fraction": float(
                ((prediction < 0.0) | (prediction > 1.0)).float().mean()
            ),
            "color_min": float(colors.min()),
            "color_max": float(colors.max()),
            "color_out_of_range_fraction": float(
                ((colors < 0.0) | (colors > 1.0)).float().mean()
            ),
        }
    return point, prediction


def _capture_steps(iters: int, log_every: int, requested: Iterable[int] | None) -> set[int]:
    if requested is not None:
        result = {max(0, min(int(iters), int(step))) for step in requested}
        result.update({0, int(iters)})
        return result
    candidates = {0, int(log_every), int(iters)}
    for fraction in (0.25, 0.5, 0.75):
        raw = fraction * int(iters)
        candidates.add(int(round(raw / log_every) * log_every))
    return {max(0, min(int(iters), step)) for step in candidates}


def _auc(curves: list[dict[str, Any]], key: str = "psnr") -> float | None:
    values = sorted(
        (int(point["attempted_steps"]), float(point[key]))
        for point in curves
        if point.get(key) is not None and math.isfinite(float(point[key]))
    )
    if not values:
        return None
    if len(values) == 1:
        return values[0][1]
    area = sum(
        (x1 - x0) * 0.5 * (y0 + y1)
        for (x0, y0), (x1, y1) in zip(values, values[1:])
    )
    return float(area / max(1, values[-1][0] - values[0][0]))


def _job_key(*, source_sha256: str, renderer: str, budget: int, seed: int,
             args: argparse.Namespace) -> str:
    payload = {
        "profile": PROFILE,
        "source_sha256": source_sha256,
        "renderer": renderer,
        "budget": int(budget),
        "seed": int(seed),
        "iters": int(args.iters),
        "max_side": int(args.max_side),
        "log_every": int(args.log_every),
        "lpips": bool(args.lpips),
        "device": str(args.device),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _configs(renderer: str, budget: int, seed: int, args: argparse.Namespace):
    tensor = StructureTensorConfig(
        grad_sigma=1.0,
        tensor_sigma=2.0,
        gradient_operator="central",
        color_space="luma",
        flat_frac=0.02,
        corner_frac=0.15,
    )
    initialization = InitConfig(
        strategy="quadtree_wse",
        num_gaussians=int(budget),
        density_base=0.05,
        density_power=1.0,
        density_mode="structure",
        sampling_mode="wse",
        max_axis_ratio=6.0,
        coherence_power=1.0,
        orientation_mode="tensor",
        scale_mode="spacing",
        scale_cap_mode="none",
        background_fraction=0.0,
        background_grid=0,
        init_scale_mult=1.0,
        color_mode="bilinear",
        color_radius=1.5,
        opacity_mode="none",
        seed=int(seed),
    )
    fitting = FitConfig(
        iters=int(args.iters),
        render_chunk=512,
        ssim_weight=0.3,
        ssim_backend="builtin",
        compute_lpips=False,
        pixel_loss="l1",
        optimizer="adam",
        lr_schedule="none",
        renderer=renderer,
        aa_dilation=0.0,
        color_basis="constant",
        color_solve_every=None,
        color_solve_schedule="none",
        max_gaussians=int(budget),
        target_psnr=30.0,
        target_psnrs=list(TARGETS),
        log_every=int(args.log_every),
    )
    return tensor, initialization, fitting


def _run_cell(image_path: Path, relative: Path, image: np.ndarray, original_size,
              *, renderer: str, budget: int, seed: int, outdir: Path,
              repository: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    import torch

    source_sha256 = _sha256_file(image_path)
    key = _job_key(
        source_sha256=source_sha256,
        renderer=renderer,
        budget=budget,
        seed=seed,
        args=args,
    )
    variant = _variant(renderer, budget)
    job_out = outdir / "runs" / variant / relative.with_suffix("") / f"seed_{seed}"
    result_path = job_out / "result.json"
    if args.resume and result_path.is_file():
        existing = json.loads(result_path.read_text(encoding="utf-8"))
        if existing.get("job_key") != key:
            raise RuntimeError(f"resume contract mismatch for {result_path}")
        return existing
    if job_out.exists() and any(job_out.iterdir()):
        raise RuntimeError(f"non-empty cell output without --resume: {job_out}")
    job_out.mkdir(parents=True, exist_ok=True)

    target = torch.as_tensor(image, device=args.device, dtype=torch.float32)
    tensor_cfg, init_cfg, fit_cfg = _configs(renderer, budget, seed, args)
    started = time.perf_counter()
    init_started = time.perf_counter()
    field = build_field(image, init_cfg, tensor_cfg, device=target.device)
    init_seconds = time.perf_counter() - init_started

    stored: list[dict[str, Any]] = [
        {
            "step": 0,
            "field": field.detached(),
            "elapsed_seconds": 0.0,
            "observer_loss": None,
        }
    ]
    fit_started = time.perf_counter()

    def observer(current: GaussianField, step: int, observer_loss: float) -> None:
        stored.append(
            {
                "step": int(step),
                "field": current.detached(),
                "elapsed_seconds": time.perf_counter() - fit_started,
                "observer_loss": float(observer_loss),
            }
        )

    output = fit(
        field,
        target,
        fit_cfg,
        verbose=not args.quiet,
        iteration_observer=observer,
        observer_every=int(args.log_every),
    )
    fit_seconds = float(output["fit_seconds"])
    telemetry_started = time.perf_counter()
    capture_steps = _capture_steps(args.iters, args.log_every, args.snapshot_steps)
    curves: list[dict[str, Any]] = []
    snapshots: list[dict[str, Any]] = []
    terminal_field = stored[-1]["field"]
    terminal_render = None
    for record in stored:
        point, prediction = _curve_point(
            record["field"],
            target,
            fit_cfg,
            step=record["step"],
            elapsed_seconds=record["elapsed_seconds"],
            lpips=args.lpips,
            observer_loss=record["observer_loss"],
        )
        curves.append(point)
        if record["step"] == int(args.iters):
            terminal_field = record["field"]
            terminal_render = prediction
        if record["step"] in capture_steps:
            stem = f"step_{int(record['step']):04d}"
            reconstruction = job_out / "intermediate" / f"{stem}.png"
            error = job_out / "intermediate" / f"{stem}_error_x4.png"
            prediction_np = prediction.detach().cpu().numpy()
            _save_rgb(reconstruction, prediction_np)
            _save_error(error, prediction_np, image)
            snapshots.append(
                {
                    **point,
                    "reconstruction": str(reconstruction),
                    "error_x4": str(error),
                }
            )
    if terminal_render is None:
        terminal_render = _render(terminal_field, fit_cfg, *target.shape[:2])
    terminal = curves[-1]
    telemetry_seconds = time.perf_counter() - telemetry_started

    target_path = job_out / "target.png"
    reconstruction_path = job_out / "reconstruction.png"
    error_path = job_out / "absolute_error_x4.png"
    field_path = job_out / "field.npz"
    history_path = job_out / "history.json"
    config_path = job_out / "config.json"
    _save_rgb(target_path, image)
    terminal_np = terminal_render.detach().cpu().numpy()
    _save_rgb(reconstruction_path, terminal_np)
    _save_error(error_path, terminal_np, image)
    terminal_field.save(str(field_path))
    _atomic_json(
        history_path,
        {
            "metric_convention": "raw fit-space tensors; PNGs are display-clamped",
            "coverage_hole_definition": "unnormalized weight sum < 0.1",
            "fit_history": output["history"],
            "curves": _portable_metric_value(outdir, curves),
            "snapshots": _portable_metric_value(outdir, snapshots),
        },
    )
    config = {
        "schema": "structsplat.current_pipeline.run.v1",
        "job_key": key,
        "profile": PROFILE,
        "scope": SCOPE,
        "source": {
            "path": str(image_path),
            "relative": relative.as_posix(),
            "sha256": source_sha256,
            "target_pixel_sha256": _pixel_sha256(image),
            "original_size": list(original_size),
            "fit_size": [int(image.shape[1]), int(image.shape[0])],
        },
        "method": "structsplat_renderer_screen",
        "method_label": _renderer_label(renderer),
        "variant": variant,
        "renderer": renderer,
        "renderer_equation": FAMILY[renderer],
        "budget": int(budget),
        "seed": int(seed),
        "structure_tensor": asdict(tensor_cfg),
        "initialization": asdict(init_cfg),
        "fit_config": asdict(fit_cfg),
        "metric_convention": "raw fit-space tensors; PNGs are display-clamped",
        "capture_steps": sorted(capture_steps),
        "repository": repository,
        "environment": {
            "device": str(args.device),
            "gpu": torch.cuda.get_device_name(torch.device(args.device))
            if torch.device(args.device).type == "cuda" else None,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "numpy": np.__version__,
        },
    }
    _atomic_json(config_path, config)
    row = {
        "schema": "structsplat.current_pipeline.metric.v1",
        "job_key": key,
        "status": "ok",
        "error": "",
        "profile": PROFILE,
        "profile_scope": SCOPE,
        "method": "structsplat_renderer_screen",
        "method_label": _renderer_label(renderer),
        "variant": variant,
        "renderer": renderer,
        "renderer_equation": FAMILY[renderer],
        "source_id": relative.as_posix(),
        "source_path": str(image_path),
        "masked": False,
        "target_pixel_sha256": _pixel_sha256(image),
        "max_side": int(args.max_side),
        "seed": int(seed),
        "budget": int(budget),
        "start_budget": int(budget),
        "start_gaussians": int(budget),
        "final_budget": int(terminal_field.n),
        "n_gaussians": int(terminal_field.n),
        "iters": int(args.iters),
        "attempted_steps": int(args.iters),
        "accepted_steps": int(args.iters),
        **{key: terminal[key] for key in (
            "psnr", "ssim", "ms_ssim", "lpips", "mse", "mae", "max_abs",
            "cvar99_mse", "p99_mse", "interior_hole_fraction", "loss", "render_min",
            "render_max", "render_out_of_range_fraction", "color_min", "color_max",
            "color_out_of_range_fraction",
        )},
        "auc_psnr": _auc(curves),
        "iters_to_targets": output["iters_to_targets"],
        "init_seconds": float(init_seconds),
        "fit_seconds": fit_seconds,
        "telemetry_seconds": float(telemetry_seconds),
        "total_seconds": float(time.perf_counter() - started),
        "phase_seconds": {"fit": fit_seconds, "post_fit_telemetry": telemetry_seconds},
        "target_png": str(target_path),
        "reconstruction_png": str(reconstruction_path),
        "error_png": str(error_path),
        "field_npz": str(field_path),
        "field_sha256": _sha256_file(field_path),
        "history_json": str(history_path),
        "config_json": str(config_path),
        "curves": curves,
        "snapshots": snapshots,
    }
    _atomic_json(result_path, row)
    return row


def _error_row(relative: Path, *, renderer: str, budget: int, seed: int,
               error: Exception) -> dict[str, Any]:
    return {
        "schema": "structsplat.current_pipeline.metric.v1",
        "status": "error",
        "error": f"{type(error).__name__}: {error}",
        "profile": PROFILE,
        "profile_scope": SCOPE,
        "method": "structsplat_renderer_screen",
        "method_label": _renderer_label(renderer),
        "variant": _variant(renderer, budget),
        "renderer": renderer,
        "renderer_equation": FAMILY[renderer],
        "source_id": relative.as_posix(),
        "seed": int(seed),
        "budget": int(budget),
    }


def _mean_curves(rows: list[dict[str, Any]], budget: int, family: str) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("status") != "ok" or row.get("budget") != budget:
            continue
        if row.get("renderer_equation") != family:
            continue
        for point in row.get("curves") or []:
            grouped.setdefault(int(point["attempted_steps"]), []).append(point)
    result = []
    for step, points in sorted(grouped.items()):
        aggregate: dict[str, Any] = {"attempted_steps": step}
        for key, _label, _direction in CURVE_SPECS:
            values = [float(point[key]) for point in points if point.get(key) is not None]
            aggregate[key] = statistics.mean(values) if values else None
        result.append(aggregate)
    return result


def _comparison_curve(series: dict[str, list[dict[str, Any]]], key: str, title: str) -> str:
    available: dict[str, list[tuple[float, float]]] = {}
    for label, points in series.items():
        values = [
            (float(point["attempted_steps"]), float(point[key]))
            for point in points
            if point.get(key) is not None and math.isfinite(float(point[key]))
        ]
        if values:
            available[label] = values
    if not available:
        return f"<div class='chart empty'><strong>{html.escape(title)}</strong><span>not available</span></div>"
    width, height, pad = 430, 180, 30
    all_values = [value for values in available.values() for value in values]
    xs = [value[0] for value in all_values]
    ys = [value[1] for value in all_values]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    if xmax <= xmin:
        xmax = xmin + 1.0
    if ymax <= ymin:
        ymax = ymin + 1.0
    polylines = []
    legends = []
    for index, (label, values) in enumerate(available.items()):
        coords = []
        for x, y in values:
            px = pad + (x - xmin) / (xmax - xmin) * (width - 2 * pad)
            py = height - pad - (y - ymin) / (ymax - ymin) * (height - 2 * pad)
            coords.append(f"{px:.1f},{py:.1f}")
        color = COLORS[label]
        polylines.append(
            f"<polyline points='{' '.join(coords)}' style='stroke:{color};fill:none;stroke-width:3'/>")
        legends.append(
            f"<span><i style='background:{color}'></i>{html.escape(label)}</span>")
    return (
        "<div class='chart'><strong>" + html.escape(title) + "</strong>"
        + "<div class='legend'>" + "".join(legends) + "</div>"
        + f"<svg viewBox='0 0 {width} {height}' role='img' aria-label='{html.escape(title)}'>"
        + f"<path d='M{pad},{height-pad}H{width-pad}M{pad},{pad}V{height-pad}'/>"
        + "".join(polylines)
        + f"<text x='{pad}' y='{height-7}'>{xmin:.0f}</text>"
        + f"<text x='{width-pad}' y='{height-7}' text-anchor='end'>{xmax:.0f} steps</text>"
        + f"<text x='{pad+3}' y='{pad+11}'>{ymax:.4g}</text>"
        + f"<text x='{pad+3}' y='{height-pad-4}'>{ymin:.4g}</text></svg></div>"
    )


def _paired_summary(rows: list[dict[str, Any]]) -> str:
    ok = [row for row in rows if row.get("status") == "ok"]
    cells: dict[tuple[str, int, int], dict[str, dict[str, Any]]] = {}
    for row in ok:
        key = (str(row["source_id"]), int(row["budget"]), int(row["seed"]))
        cells.setdefault(key, {})[str(row["renderer_equation"])] = row
    parts = []
    for budget in sorted({int(row["budget"]) for row in ok}):
        pairs = [value for key, value in cells.items() if key[1] == budget and set(value) == {"normalized", "additive"}]
        if not pairs:
            continue
        def delta(metric: str) -> float | None:
            values = [
                float(pair["additive"][metric]) - float(pair["normalized"][metric])
                for pair in pairs
                if pair["additive"].get(metric) is not None
                and pair["normalized"].get(metric) is not None
            ]
            return statistics.mean(values) if values else None

        def formatted(metric: str, digits: int) -> str:
            value = delta(metric)
            return "-" if value is None else f"{value:+.{digits}f}"
        parts.append(
            "<tr>"
            f"<td>{budget:,}</td><td>{len(pairs)}</td>"
            f"<td>{formatted('psnr', 4)}</td><td>{formatted('ssim', 5)}</td>"
            f"<td>{formatted('ms_ssim', 5)}</td><td>{formatted('lpips', 5)}</td>"
            f"<td>{formatted('auc_psnr', 4)}</td><td>{formatted('loss', 5)}</td>"
            "</tr>"
        )
    return "".join(parts)


def _visual_panels(outdir: Path, rows: list[dict[str, Any]]) -> str:
    ok = [row for row in rows if row.get("status") == "ok"]
    if not ok:
        return ""
    preferred_seed = min(int(row["seed"]) for row in ok)
    panels = []
    keys = sorted({(str(row["source_id"]), int(row["budget"])) for row in ok})
    for source_id, budget in keys:
        group = {
            str(row["renderer_equation"]): row
            for row in ok
            if row["source_id"] == source_id
            and int(row["budget"]) == budget
            and int(row["seed"]) == preferred_seed
        }
        if set(group) != {"normalized", "additive"}:
            continue
        normalized = group["normalized"]
        additive = group["additive"]
        figures = []
        for label, row, key in (
            ("target", normalized, "target_png"),
            ("normalized reconstruction", normalized, "reconstruction_png"),
            ("normalized error ×4", normalized, "error_png"),
            ("additive reconstruction", additive, "reconstruction_png"),
            ("additive error ×4", additive, "error_png"),
        ):
            link = _relative_link(outdir, row[key])
            figures.append(
                f"<figure><a href='{html.escape(link)}'><img src='{html.escape(link)}' "
                f"loading='lazy' alt='{html.escape(label)}'></a><figcaption>{html.escape(label)}</figcaption></figure>"
            )
        panels.append(
            f"<article class='comparison'><h3>{html.escape(source_id)} · N={budget:,} · seed {preferred_seed}</h3>"
            f"<div class='comparison-images'>{''.join(figures)}</div></article>"
        )
    return "".join(panels)


def _run_card_with_diagnostics(outdir: Path, row: dict[str, Any]) -> str:
    card = _run_card(outdir, row)
    extra = (
        _svg_curve(row.get("curves") or [], "loss", "Optimization objective over attempted steps", "#9c2f2f")
        + _svg_curve(
            row.get("curves") or [],
            "render_out_of_range_fraction",
            "Out-of-range channel fraction over attempted steps",
            "#4f378b",
        )
    )
    card = card.replace("<div class='charts'>", "<div class='charts'>" + extra, 1)
    range_summary = (
        "<div class='metrics'>"
        f"<span><b>[{float(row['render_min']):.3f}, {float(row['render_max']):.3f}]</b> raw render range</span>"
        f"<span><b>{float(row['render_out_of_range_fraction']):.2%}</b> out-of-range channels</span>"
        f"<span><b>[{float(row['color_min']):.3f}, {float(row['color_max']):.3f}]</b> component-color range</span>"
        "</div>"
    )
    return card.replace("<div class='links'>", range_summary + "<div class='links'>", 1)


def _write_index(outdir: Path, rows: list[dict[str, Any]], command: str) -> None:
    ok = [row for row in rows if row.get("status") == "ok"]
    errors = [row for row in rows if row.get("status") != "ok"]
    table_rows = []
    for row in ok:
        lpips = "-" if row.get("lpips") is None else f"{float(row['lpips']):.5f}"
        table_rows.append(
            "<tr>"
            f"<td>{html.escape(str(row['renderer_equation']))}</td>"
            f"<td>{html.escape(str(row['source_id']))}</td>"
            f"<td>{int(row['budget'])}</td><td>{int(row['seed'])}</td>"
            f"<td>{float(row['psnr']):.4f}</td><td>{float(row['ms_ssim']):.5f}</td>"
            f"<td>{lpips}</td><td>{float(row['loss']):.5f}</td>"
            f"<td>{float(row['fit_seconds']):.3f}</td>"
            "</tr>"
        )
    aggregate = []
    for budget in sorted({int(row["budget"]) for row in ok}):
        series = {
            family: _mean_curves(ok, budget, family)
            for family in ("normalized", "additive")
        }
        for key, label, direction in CURVE_SPECS:
            aggregate.append(
                _comparison_curve(series, key, f"N={budget:,} · {label} ({direction})")
            )
    error_html = "".join(
        f"<li><code>{html.escape(str(row.get('source_id')))}</code>: {html.escape(str(row.get('error')))}</li>"
        for row in errors
    )
    cards = "".join(_run_card_with_diagnostics(outdir, row) for row in ok)
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ABL-002 normalized versus additive renderer screen</title>
<style>
:root{{--paper:#f3efe5;--ink:#1d2528;--muted:#64706f;--line:#c8c0ae;--accent:#e65f2b;--teal:#147d72;--panel:#fffdf7}}
*{{box-sizing:border-box}}body{{margin:0;background:linear-gradient(135deg,#eee7d8,#f7f4eb 55%,#e5eee9);color:var(--ink);font-family:Georgia,"Times New Roman",serif;line-height:1.45}}
header,main{{width:min(1600px,calc(100% - 32px));margin:auto}}header{{padding:48px 0 26px;border-bottom:3px double var(--ink)}}
h1{{font-size:clamp(2rem,5vw,4.5rem);line-height:.96;margin:0 0 18px;max-width:1100px}}h2{{margin-top:38px}}h3{{margin:0 0 10px}}
code,.identity,.metrics,table,.links{{font-family:"Liberation Mono",monospace}}.scope{{max-width:1100px;color:var(--muted)}}
.links{{display:flex;gap:16px;flex-wrap:wrap;margin:18px 0}}a{{color:#095c57}}main{{padding:24px 0 70px}}
table{{width:100%;border-collapse:collapse;background:var(--panel);font-size:.82rem}}th,td{{border:1px solid var(--line);padding:7px;text-align:right}}th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){{text-align:left}}
.comparison,.run{{margin:24px 0;padding:18px;background:rgba(255,253,247,.92);border:1px solid var(--line);box-shadow:7px 7px 0 rgba(29,37,40,.08)}}
.comparison-images,.hero-images,.intermediate{{display:flex;gap:10px;overflow:auto;padding:8px 0}}figure{{margin:0;min-width:220px;flex:1}}figure.small{{min-width:180px;max-width:300px}}img{{width:100%;height:auto;display:block;background:#111;border:1px solid #777}}figcaption{{font-size:.82rem;color:var(--muted);margin-top:5px}}
.charts,.aggregate{{display:grid;grid-template-columns:repeat(auto-fit,minmax(310px,1fr));gap:12px;margin-top:16px}}.chart{{background:#fff;border:1px solid var(--line);padding:9px}}.chart strong{{display:block}}.chart svg{{width:100%;height:auto}}.chart path{{fill:none;stroke:#c7c1b5;stroke-width:1}}.chart polyline{{fill:none;stroke-width:3}}.chart text{{font:10px "Liberation Mono",monospace;fill:#6b6b66}}.empty{{min-height:110px;color:var(--muted)}}
.legend{{display:flex;gap:12px;font:11px "Liberation Mono",monospace;margin:4px 0}}.legend i{{display:inline-block;width:12px;height:3px;margin-right:4px;vertical-align:middle}}.metrics{{display:flex;gap:10px;flex-wrap:wrap;margin:14px 0}}.metrics span{{border:1px solid var(--line);padding:7px 10px;background:#fff}}.identity{{color:var(--muted)}}details{{margin-top:15px}}summary{{cursor:pointer;font-weight:bold}}.errors{{color:#8a251a}}
@media(max-width:700px){{header,main{{width:min(100% - 18px,1600px)}}.comparison,.run{{padding:10px;box-shadow:4px 4px 0 rgba(29,37,40,.08)}}table{{display:block;overflow:auto}}}}
</style></head><body><header><p class="identity">STRUCTSPLAT / ABL-002 / DIAGNOSTIC</p>
<h1>Normalized versus additive Gaussian-field fitting</h1><p class="scope">{html.escape(SCOPE)}</p>
<p><code>{html.escape(command)}</code></p><div class="links"><a href="manifest.json">manifest.json</a><a href="metrics.json">metrics.json</a><a href="metrics.jsonl">metrics.jsonl</a><a href="metrics.csv">metrics.csv</a></div></header><main>
<h2>Paired outcome</h2><p>Additive minus normalized. Positive quality deltas favor additive except LPIPS/loss, where lower is better.</p>
<table><thead><tr><th>N</th><th>pairs</th><th>ΔPSNR</th><th>ΔSSIM</th><th>ΔMS-SSIM</th><th>ΔLPIPS</th><th>ΔPSNR AUC</th><th>Δloss</th></tr></thead><tbody>{_paired_summary(ok)}</tbody></table>
<h2>Visual comparisons</h2><p>Seed 0; click every image for its native fit resolution.</p>{_visual_panels(outdir, ok)}
<h2>Aggregate trajectories</h2><p>Means over all image × seed cells at each attempted step.</p><div class="aggregate">{''.join(aggregate)}</div>
<h2>Run matrix</h2><table><thead><tr><th>renderer</th><th>image</th><th>N</th><th>seed</th><th>PSNR</th><th>MS-SSIM</th><th>LPIPS</th><th>loss</th><th>fit s</th></tr></thead><tbody>{''.join(table_rows)}</tbody></table>
{f'<section class="errors"><h2>Errors</h2><ul>{error_html}</ul></section>' if errors else ''}
<h2>Per-run artifacts and trajectories</h2>{cards}</main></body></html>"""
    _atomic_text(outdir / "index.html", document)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("outdir", type=Path)
    parser.add_argument("--budgets", type=int, nargs="+", default=[2000, 5000])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--iters", type=int, default=750)
    parser.add_argument("--max-side", type=int, default=512)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--snapshot-steps", type=int, nargs="+")
    parser.add_argument("--renderers", nargs="+", choices=RENDERERS, default=["cuda", "cuda_additive"])
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--lpips", action="store_true")
    parser.add_argument("--device", default="cuda")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--resume", action="store_true")
    mode.add_argument("--overwrite", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser


def _validate(args: argparse.Namespace) -> None:
    if any(value <= 0 for value in args.budgets):
        raise ValueError("budgets must be positive")
    if args.iters <= 0 or args.max_side <= 0 or args.log_every <= 0:
        raise ValueError("iters, max-side, and log-every must be positive")
    if args.log_every > args.iters:
        raise ValueError("log-every must not exceed iters")
    families = [FAMILY[value] for value in args.renderers]
    if len(set(families)) != len(families):
        raise ValueError("choose one implementation per renderer equation")
    if set(families) != {"normalized", "additive"}:
        raise ValueError("the renderer screen requires one normalized and one additive arm")
    if len(set(args.budgets)) != len(args.budgets) or len(set(args.seeds)) != len(args.seeds):
        raise ValueError("budgets and seeds must be unique")


def _warm_runtime(args: argparse.Namespace, discovered: list[tuple[Path, Path]]) -> None:
    """Move one-time extension/model setup outside every measured cell."""
    import torch

    device = torch.device(args.device)
    if device.type == "cuda" and any(value.startswith("cuda") for value in args.renderers):
        from structsplat.cuda_render import _load_extension

        _load_extension()
        torch.cuda.synchronize(device)
    if args.lpips:
        image, _ = _load_rgb(discovered[0][0], min(int(args.max_side), 96))
        target = torch.as_tensor(image, device=device, dtype=torch.float32)
        M.LPIPS.distance(target, target)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _validate(args)
    args.source = args.source.expanduser().resolve()
    args.outdir = args.outdir.expanduser().resolve()
    if args.outdir.exists() and any(args.outdir.iterdir()):
        if args.overwrite:
            shutil.rmtree(args.outdir)
        elif not args.resume:
            raise RuntimeError(f"output is non-empty: {args.outdir}; use --resume or --overwrite")
    args.outdir.mkdir(parents=True, exist_ok=True)
    discovered = _discover_images(args.source)
    if args.max_images is not None:
        discovered = discovered[: int(args.max_images)]
    _warm_runtime(args, discovered)
    repository = _repository_state()
    variants = [_variant(renderer, budget) for budget in args.budgets for renderer in args.renderers]
    command = " ".join(sys.argv if argv is None else [str(Path(__file__)), *argv])
    manifest = {
        "schema": "structsplat.current_pipeline.workflow.v1",
        "title": "ABL-002 normalized versus additive renderer screen",
        "profile": {"name": PROFILE, "scope": SCOPE},
        "command": command,
        "source": str(args.source),
        "variants": variants,
        "renderers": list(args.renderers),
        "budgets": list(args.budgets),
        "seeds": list(args.seeds),
        "iters": int(args.iters),
        "max_side": int(args.max_side),
        "log_every": int(args.log_every),
        "lpips": bool(args.lpips),
        "runtime_warmup": "CUDA extension and LPIPS model initialized before measured cells",
        "metric_convention": "raw fit-space tensors; PNGs are display-clamped",
        "images": [
            {"path": str(path), "relative": relative.as_posix(), "sha256": _sha256_file(path)}
            for path, relative in discovered
        ],
        "repository": repository,
    }
    _atomic_json(args.outdir / "manifest.json", manifest)

    rows: list[dict[str, Any]] = []
    for image_path, relative in discovered:
        image, original_size = _load_rgb(image_path, args.max_side)
        for budget in args.budgets:
            for seed in args.seeds:
                for renderer in args.renderers:
                    print(
                        f"[{relative.as_posix()}] N={budget} seed={seed} renderer={renderer}",
                        flush=True,
                    )
                    try:
                        row = _run_cell(
                            image_path,
                            relative,
                            image,
                            original_size,
                            renderer=renderer,
                            budget=budget,
                            seed=seed,
                            outdir=args.outdir,
                            repository=repository,
                            args=args,
                        )
                    except Exception as error:  # retain failed cells in the bundle
                        print(f"  ERROR: {type(error).__name__}: {error}", flush=True)
                        row = _error_row(
                            relative,
                            renderer=renderer,
                            budget=budget,
                            seed=seed,
                            error=error,
                        )
                    rows.append(row)
                    _write_metrics(args.outdir, rows)
    _write_index(args.outdir, rows, command)
    return 1 if any(row.get("status") != "ok" for row in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
