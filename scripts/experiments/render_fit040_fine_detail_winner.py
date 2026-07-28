#!/usr/bin/env python3
"""Replay and render the FIT-040 fine-detail winner on the prior Janelle data."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import matplotlib
import numpy as np
from PIL import __version__ as PILLOW_VERSION
from PIL import Image, ImageDraw, ImageFont
import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
for _root in (REPOSITORY_ROOT, REPOSITORY_ROOT / "src"):
    _text = str(_root)
    if _text not in sys.path:
        sys.path.insert(0, _text)

from scripts.experiments.fit032_janelle_dipole_screen import (  # noqa: E402
    _base_config,
    _prepare_current_job,
    _scaled_field,
)
from scripts.experiments.fit033_janelle_highpass_solve import (  # noqa: E402
    _evaluate_all,
    gaussian_blur,
)
from scripts.experiments.fit040_janelle_production_pursuit import (  # noqa: E402
    _disabled_phase,
)
from structsplat.fit import _MaskConstraint  # noqa: E402
from structsplat.safe_schedule import (  # noqa: E402
    SafeScheduleConfig,
    run_safe_schedule,
)


DEFAULT_BASE_JOB = (
    REPOSITORY_ROOT
    / "runs/fit031_new_methods_comparison_20260728"
    / "base_exact/runs/current/C0001/seed_0"
)
DEFAULT_SELECTION = (
    REPOSITORY_ROOT
    / "runs/fit031_new_methods_comparison_20260728/fit033/result.json"
)
DEFAULT_OUT = (
    REPOSITORY_ROOT
    / "ara/evidence/fit031-new-method-stages-janelle-2026-07-28/visuals"
)
FONT_REGULAR = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
FONT_BOLD = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
BACKGROUND = (10, 15, 27)
CARD = (19, 27, 44)
TEXT = (235, 240, 248)
MUTED = (160, 174, 196)
ACCENT = (61, 214, 198)
WARNING = (255, 190, 77)
RECTANGLE = (255, 103, 103)
WAVE_COLORS = (
    (60, 205, 255),
    (65, 230, 155),
    (245, 218, 75),
    (255, 152, 73),
    (239, 96, 154),
    (164, 113, 255),
    (190, 205, 220),
    (255, 255, 255),
)
SOURCE_FILES = (
    "scripts/experiments/render_fit040_fine_detail_winner.py",
    "scripts/experiments/fit040_janelle_production_pursuit.py",
    "scripts/experiments/fit033_janelle_highpass_solve.py",
    "benchmarks/highpass_births.py",
    "benchmarks/residual_birth_color_solve.py",
    "src/structsplat/safe_schedule.py",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _git(*arguments: str) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPOSITORY_ROOT), *arguments],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    path = FONT_BOLD if bold else FONT_REGULAR
    return ImageFont.truetype(str(path), size=size)


def _tensor_rgb(image: torch.Tensor) -> Image.Image:
    array = (
        image.detach()
        .cpu()
        .clamp(0.0, 1.0)
        .mul(255.0)
        .round()
        .to(torch.uint8)
        .numpy()
    )
    return Image.fromarray(array, mode="RGB")


def _save_rgb(path: Path, image: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _tensor_rgb(image).save(path)


def _heatmap(value: torch.Tensor, scale: float) -> Image.Image:
    normalized = (
        value.detach().cpu().numpy() / max(float(scale), 1e-12)
    ).clip(0.0, 1.0)
    mapped = matplotlib.colormaps["magma"](normalized)[..., :3]
    return Image.fromarray(np.round(mapped * 255.0).astype(np.uint8), mode="RGB")


def _signed_heatmap(value: torch.Tensor, scale: float) -> Image.Image:
    normalized = (
        value.detach().cpu().numpy() / max(float(scale), 1e-12)
    ).clip(-1.0, 1.0)
    positive = np.maximum(normalized, 0.0)
    negative = np.maximum(-normalized, 0.0)
    intensity = np.sqrt(np.abs(normalized))
    rgb = np.stack(
        (
            0.92 * negative + 0.08 * positive,
            0.88 * positive + 0.06 * negative,
            0.72 * negative + 0.60 * positive,
        ),
        axis=-1,
    )
    rgb *= intensity[..., None]
    return Image.fromarray(np.round(rgb * 255.0).astype(np.uint8), mode="RGB")


def _crop_bounds(
    shape: tuple[int, int],
    center_yx: tuple[int, int],
    half_size: int,
) -> tuple[int, int, int, int]:
    height, width = shape
    center_y, center_x = center_yx
    y0 = max(0, min(height - 2 * half_size, center_y - half_size))
    x0 = max(0, min(width - 2 * half_size, center_x - half_size))
    return x0, y0, min(width, x0 + 2 * half_size), min(
        height, y0 + 2 * half_size
    )


def _mask_view_bounds(
    mask: torch.Tensor,
    detail_bounds: tuple[int, int, int, int],
    *,
    padding: int = 28,
) -> tuple[int, int, int, int]:
    y, x = torch.where(mask)
    x0 = min(int(x.min()), detail_bounds[0]) - padding
    y0 = min(int(y.min()), detail_bounds[1]) - padding
    x1 = max(int(x.max()) + 1, detail_bounds[2]) + padding
    y1 = max(int(y.max()) + 1, detail_bounds[3]) + padding
    height, width = mask.shape
    return (
        max(0, x0),
        max(0, y0),
        min(width, x1),
        min(height, y1),
    )


def _draw_heading(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    title: str,
    subtitle: str,
) -> None:
    draw.text(xy, title, font=_font(25, bold=True), fill=TEXT)
    draw.text(
        (xy[0], xy[1] + 34),
        subtitle,
        font=_font(16),
        fill=MUTED,
    )


def _resize_contain(
    image: Image.Image,
    size: tuple[int, int],
    *,
    resample: Image.Resampling = Image.Resampling.LANCZOS,
) -> Image.Image:
    copy = image.copy()
    copy.thumbnail(size, resample)
    canvas = Image.new("RGB", size, CARD)
    x = (size[0] - copy.width) // 2
    y = (size[1] - copy.height) // 2
    canvas.paste(copy, (x, y))
    return canvas


def _full_frame_montage(
    target: Image.Image,
    base: Image.Image,
    winner: Image.Image,
    *,
    view_bounds: tuple[int, int, int, int],
    detail_bounds: tuple[int, int, int, int],
    base_rows: int,
    added_rows: int,
    highpass_reduction: float,
    laplacian_reduction: float,
) -> Image.Image:
    panel_width = 720
    panel_height = 330
    gap = 24
    top = 112
    canvas = Image.new(
        "RGB",
        (3 * panel_width + 4 * gap, top + panel_height + 94),
        BACKGROUND,
    )
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (gap, 20),
        "Fine-detail winner — same masked Janelle target",
        font=_font(34, bold=True),
        fill=TEXT,
    )
    draw.text(
        (gap, 66),
        (
            "Mask-bounded subject view; the red rectangle is the fixed "
            "pre-treatment detail crop."
        ),
        font=_font(18),
        fill=MUTED,
    )
    titles = (
        ("Target", "Reference pixels"),
        ("Current base", f"{base_rows:,} rows"),
        (
            "Orthogonal pursuit",
            (
                f"+{added_rows:,} rows  •  "
                f"HP −{100 * highpass_reduction:.2f}%  •  "
                f"Lap −{100 * laplacian_reduction:.2f}%"
            ),
        ),
    )
    for index, (source, heading) in enumerate(zip((target, base, winner), titles)):
        x = gap + index * (panel_width + gap)
        panel = source.crop(view_bounds)
        panel_draw = ImageDraw.Draw(panel)
        rectangle = (
            detail_bounds[0] - view_bounds[0],
            detail_bounds[1] - view_bounds[1],
            detail_bounds[2] - view_bounds[0] - 1,
            detail_bounds[3] - view_bounds[1] - 1,
        )
        panel_draw.rectangle(rectangle, outline=RECTANGLE, width=3)
        panel = _resize_contain(panel, (panel_width, panel_height))
        canvas.paste(panel, (x, top))
        _draw_heading(draw, (x, top + panel_height + 12), *heading)
    return canvas


def _detail_montage(
    target: Image.Image,
    base: Image.Image,
    winner: Image.Image,
    *,
    detail_bounds: tuple[int, int, int, int],
    center_yx: tuple[int, int],
    added_rows: int,
    local_highpass_reduction: float,
) -> Image.Image:
    scale = 3
    crop_size = detail_bounds[2] - detail_bounds[0]
    panel_size = crop_size * scale
    gap = 24
    top = 116
    canvas = Image.new(
        "RGB",
        (3 * panel_size + 4 * gap, top + panel_size + 100),
        BACKGROUND,
    )
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (gap, 18),
        "Fixed fine-detail crop — exact pixels at 3×",
        font=_font(34, bold=True),
        fill=TEXT,
    )
    draw.text(
        (gap, 64),
        (
            f"Center y={center_yx[0]}, x={center_yx[1]}; selected from the "
            "base residual before the winner was run."
        ),
        font=_font(18),
        fill=MUTED,
    )
    titles = (
        ("Target", "Reference pixels"),
        ("Current base", "Before pursuit"),
        (
            "Orthogonal pursuit",
            (
                f"+{added_rows:,} rows  •  fixed-crop HP "
                f"−{100 * local_highpass_reduction:.2f}%"
            ),
        ),
    )
    for index, (source, heading) in enumerate(zip((target, base, winner), titles)):
        x = gap + index * (panel_size + gap)
        panel = source.crop(detail_bounds).resize(
            (panel_size, panel_size),
            Image.Resampling.NEAREST,
        )
        canvas.paste(panel, (x, top))
        _draw_heading(draw, (x, top + panel_size + 12), *heading)
    return canvas


def _diagnostic_montage(
    base_error: torch.Tensor,
    winner_error: torch.Tensor,
    base_highpass: torch.Tensor,
    winner_highpass: torch.Tensor,
    correction: torch.Tensor,
    highpass_delta: torch.Tensor,
    *,
    detail_bounds: tuple[int, int, int, int],
    error_scale: float,
    highpass_scale: float,
    highpass_delta_scale: float,
) -> Image.Image:
    scale = 3
    crop_size = detail_bounds[2] - detail_bounds[0]
    panel_size = crop_size * scale
    gap = 24
    header = 128
    row_heading = 70
    row_gap = 118
    width = 3 * panel_size + 4 * gap
    height = header + 2 * (panel_size + row_heading) + row_gap
    canvas = Image.new("RGB", (width, height), BACKGROUND)
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (gap, 18),
        "What changed in the fixed crop",
        font=_font(34, bold=True),
        fill=TEXT,
    )
    draw.text(
        (gap, 64),
        (
            "Base and winner error maps share the earlier base-derived "
            "scales; no per-panel normalization."
        ),
        font=_font(18),
        fill=MUTED,
    )
    x0, y0, x1, y1 = detail_bounds
    correction_rgb = (
        correction[y0:y1, x0:x1]
        .detach()
        .cpu()
        .mul(8.0)
        .add(0.5)
        .clamp(0.0, 1.0)
    )
    panels = (
        (
            _heatmap(base_error[y0:y1, x0:x1], error_scale),
            "Base absolute RGB error",
            f"Common scale: {error_scale:.5f}",
        ),
        (
            _heatmap(winner_error[y0:y1, x0:x1], error_scale),
            "Winner absolute RGB error",
            f"Common scale: {error_scale:.5f}",
        ),
        (
            _tensor_rgb(correction_rgb),
            "Winner − base RGB ×8",
            "Neutral gray means no correction",
        ),
        (
            _heatmap(base_highpass[y0:y1, x0:x1], highpass_scale),
            "Base high-pass residual",
            f"Common scale: {highpass_scale:.5f}",
        ),
        (
            _heatmap(winner_highpass[y0:y1, x0:x1], highpass_scale),
            "Winner high-pass residual",
            f"Common scale: {highpass_scale:.5f}",
        ),
        (
            _signed_heatmap(
                highpass_delta[y0:y1, x0:x1],
                highpass_delta_scale,
            ),
            "High-pass MSE change",
            "Green improves • magenta worsens",
        ),
    )
    for index, (panel, title, subtitle) in enumerate(panels):
        row = index // 3
        column = index % 3
        x = gap + column * (panel_size + gap)
        y = header + row * (panel_size + row_heading + row_gap)
        panel = panel.resize(
            (panel_size, panel_size),
            Image.Resampling.NEAREST,
        )
        canvas.paste(panel, (x, y))
        _draw_heading(draw, (x, y + panel_size + 12), title, subtitle)
    return canvas


def _site_montage(
    target: Image.Image,
    winner_means: torch.Tensor,
    *,
    base_rows: int,
    batch_rows: int,
    view_bounds: tuple[int, int, int, int],
    detail_bounds: tuple[int, int, int, int],
) -> Image.Image:
    view = target.crop(view_bounds).copy()
    overlay = ImageDraw.Draw(view)
    tail = winner_means[base_rows:].detach().cpu().numpy()
    for index, (x, y) in enumerate(tail):
        color = WAVE_COLORS[min(index // batch_rows, len(WAVE_COLORS) - 1)]
        local_x = float(x) - view_bounds[0]
        local_y = float(y) - view_bounds[1]
        radius = 1.7
        overlay.ellipse(
            (
                local_x - radius,
                local_y - radius,
                local_x + radius,
                local_y + radius,
            ),
            fill=color,
            outline=(0, 0, 0),
            width=1,
        )
    overlay.rectangle(
        (
            detail_bounds[0] - view_bounds[0],
            detail_bounds[1] - view_bounds[1],
            detail_bounds[2] - view_bounds[0] - 1,
            detail_bounds[3] - view_bounds[1] - 1,
        ),
        outline=RECTANGLE,
        width=3,
    )
    panel_width = 1500
    panel_height = round(panel_width * view.height / view.width)
    view = view.resize((panel_width, panel_height), Image.Resampling.NEAREST)
    canvas = Image.new(
        "RGB",
        (panel_width + 64, panel_height + 176),
        BACKGROUND,
    )
    canvas.paste(view, (32, 92))
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (32, 18),
        "Where the winner spent its rows",
        font=_font(34, bold=True),
        fill=TEXT,
    )
    draw.text(
        (32, 62),
        (
            "Each dot is one 0.35 px Gaussian center; colors indicate "
            f"{batch_rows}-row pursuit waves."
        ),
        font=_font(18),
        fill=MUTED,
    )
    waves = max(1, (len(tail) + batch_rows - 1) // batch_rows)
    legend_y = panel_height + 112
    for wave in range(waves):
        x = 32 + wave * 180
        color = WAVE_COLORS[min(wave, len(WAVE_COLORS) - 1)]
        draw.ellipse((x, legend_y, x + 15, legend_y + 15), fill=color)
        draw.text(
            (x + 24, legend_y - 4),
            f"wave {wave + 1}",
            font=_font(16),
            fill=TEXT,
        )
    draw.rectangle(
        (
            32 + waves * 180,
            legend_y,
            32 + waves * 180 + 18,
            legend_y + 15,
        ),
        outline=RECTANGLE,
        width=2,
    )
    draw.text(
        (60 + waves * 180, legend_y - 4),
        "fixed crop",
        font=_font(16),
        fill=TEXT,
    )
    return canvas


def _load_visual_protocol(
    path: Path,
) -> tuple[tuple[int, int], int, float, float, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    center = tuple(int(value) for value in payload["selection"]["crop_center_yx"])
    scales = payload["visualization_scales"]
    return (
        (center[0], center[1]),
        96,
        float(scales["error_p99"]),
        float(scales["highpass_error_p99"]),
        payload,
    )


def _saved_base_render_parity(
    base_job: Path,
    rendered: Image.Image,
) -> dict[str, Any] | None:
    saved_path = base_job / "reconstruction.png"
    if not saved_path.is_file():
        return None
    saved = np.asarray(Image.open(saved_path).convert("RGB"), dtype=np.int16)
    current = np.asarray(rendered, dtype=np.int16)
    if saved.shape != current.shape:
        return {
            "saved_path": str(saved_path.resolve()),
            "saved_sha256": _sha256(saved_path),
            "shape_exact": False,
            "saved_shape": list(saved.shape),
            "current_shape": list(current.shape),
        }
    difference = np.abs(saved - current)
    return {
        "saved_path": str(saved_path.resolve()),
        "saved_sha256": _sha256(saved_path),
        "shape_exact": True,
        "max_abs_u8": int(difference.max()),
        "mean_abs_u8": float(difference.mean()),
        "nonzero_channels": int(np.count_nonzero(difference)),
        "total_channels": int(difference.size),
        "interpretation": (
            "display-space parity check; a one-level difference is compatible "
            "with documented CUDA atomic rendering nondeterminism"
        ),
    }


def run(args: argparse.Namespace) -> None:
    if args.out.exists() and any(args.out.iterdir()):
        raise RuntimeError(f"output directory is not empty: {args.out}")
    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA but torch.cuda.is_available() is false")
    torch.manual_seed(0)

    prepared = _prepare_current_job(args.base_job)
    target = torch.as_tensor(
        prepared["target"],
        device=device,
        dtype=torch.float32,
    ).contiguous()
    mask = torch.as_tensor(
        prepared["mask"],
        device=device,
        dtype=torch.bool,
    )
    base = _scaled_field(prepared["field_path"], device, 1.0, 1.0)
    cfg = replace(
        _base_config(args),
        color_solve_maxiter=1,
        color_solve_lambda=1e30,
    )
    defaults = SafeScheduleConfig()
    schedule = SafeScheduleConfig(
        capacity=base.n,
        storage_policy="dynamic",
        boundary_enabled=True,
        coverage_target_gaussians=base.n,
        detail_target_gaussians=base.n,
        coverage_tau=float(args.coverage_tau),
        boundary_band=float(args.boundary_band),
        pursuit_tail_enabled=True,
        bootstrap=_disabled_phase(defaults.bootstrap, base.n),
        coverage=_disabled_phase(defaults.coverage, base.n),
        detail=_disabled_phase(defaults.detail, base.n),
        boundary=_disabled_phase(defaults.boundary, base.n),
        redistribution=_disabled_phase(defaults.redistribution, base.n),
        polish=_disabled_phase(defaults.polish, base.n),
    )

    started = time.perf_counter()
    replay = run_safe_schedule(
        base,
        target,
        mask,
        cfg,
        schedule,
        verbose=not args.quiet,
    )
    elapsed = time.perf_counter() - started
    winner = replay["field"]
    pursuit = replay["pursuit_tail"]
    if not bool(pursuit["target_reached"]):
        raise RuntimeError(
            "visual replay did not reach the frozen fine-detail target; "
            f"termination={pursuit['termination_reason']}"
        )

    constraint = _MaskConstraint.from_mask(
        prepared["mask"],
        device,
        target.dtype,
        cfg.sigma_cutoff,
        cfg.mask_margin,
        aa_dilation=cfg.aa_dilation,
        min_scale=0.35,
        cap_mode=cfg.mask_cap_mode,
        undercoverage_band=cfg.mask_undercoverage_band,
    )
    baseline_metrics, base_render, _ = _evaluate_all(
        base,
        target,
        mask,
        cfg,
        constraint,
        args.coverage_tau,
    )
    winner_metrics, winner_render, _ = _evaluate_all(
        winner,
        target,
        mask,
        cfg,
        constraint,
        args.coverage_tau,
    )

    center_yx, half_size, error_scale, highpass_scale, selection = (
        _load_visual_protocol(args.selection)
    )
    detail_bounds = _crop_bounds(mask.shape, center_yx, half_size)
    view_bounds = _mask_view_bounds(mask, detail_bounds)
    deep = (
        constraint.sdf_flat.reshape(mask.shape)
        > float(constraint.margin) + 6.0
    )
    base_residual = base_render - target
    winner_residual = winner_render - target
    base_highpass_rgb = base_residual - gaussian_blur(base_residual, 1.5)
    winner_highpass_rgb = winner_residual - gaussian_blur(
        winner_residual,
        1.5,
    )
    base_error = base_residual.square().mean(dim=2).sqrt()
    winner_error = winner_residual.square().mean(dim=2).sqrt()
    base_highpass = base_highpass_rgb.square().mean(dim=2).sqrt()
    winner_highpass = winner_highpass_rgb.square().mean(dim=2).sqrt()
    highpass_delta = base_highpass_rgb.square().mean(
        dim=2
    ) - winner_highpass_rgb.square().mean(dim=2)
    highpass_delta_scale = float(
        torch.quantile(highpass_delta[deep].abs(), 0.99)
    )
    x0, y0, x1, y1 = detail_bounds
    crop_deep = deep[y0:y1, x0:x1]
    if not bool(crop_deep.any()):
        raise RuntimeError("fixed detail crop has no deep-mask pixels")
    local_before = float(
        base_highpass_rgb[y0:y1, x0:x1][crop_deep].square().mean()
    )
    local_after = float(
        winner_highpass_rgb[y0:y1, x0:x1][crop_deep].square().mean()
    )
    local_reduction = 1.0 - local_after / local_before

    field_path = args.out / "winner_field.npz"
    winner.save(str(field_path))
    image_dir = args.out / "images"
    _save_rgb(image_dir / "target.png", target)
    _save_rgb(image_dir / "base.png", base_render)
    _save_rgb(image_dir / "winner.png", winner_render)

    target_image = _tensor_rgb(target)
    base_image = _tensor_rgb(base_render)
    winner_image = _tensor_rgb(winner_render)
    full_frame = _full_frame_montage(
        target_image,
        base_image,
        winner_image,
        view_bounds=view_bounds,
        detail_bounds=detail_bounds,
        base_rows=base.n,
        added_rows=int(pursuit["activated_rows"]),
        highpass_reduction=float(pursuit["highpass_reduction"]),
        laplacian_reduction=float(pursuit["laplacian_reduction"]),
    )
    detail = _detail_montage(
        target_image,
        base_image,
        winner_image,
        detail_bounds=detail_bounds,
        center_yx=center_yx,
        added_rows=int(pursuit["activated_rows"]),
        local_highpass_reduction=local_reduction,
    )
    diagnostics = _diagnostic_montage(
        base_error,
        winner_error,
        base_highpass,
        winner_highpass,
        winner_render - base_render,
        highpass_delta,
        detail_bounds=detail_bounds,
        error_scale=error_scale,
        highpass_scale=highpass_scale,
        highpass_delta_scale=highpass_delta_scale,
    )
    allocation = _site_montage(
        target_image,
        winner.means,
        base_rows=base.n,
        batch_rows=int(pursuit["batch_rows"]),
        view_bounds=view_bounds,
        detail_bounds=detail_bounds,
    )
    montage_paths = {
        "full_frame": args.out / "full_frame_comparison.png",
        "detail_crop": args.out / "detail_crop_comparison.png",
        "diagnostics": args.out / "detail_diagnostics.png",
        "site_allocation": args.out / "site_allocation.png",
    }
    full_frame.save(montage_paths["full_frame"])
    detail.save(montage_paths["detail_crop"])
    diagnostics.save(montage_paths["diagnostics"])
    allocation.save(montage_paths["site_allocation"])

    raw_image_paths = {
        "target": image_dir / "target.png",
        "base": image_dir / "base.png",
        "winner": image_dir / "winner.png",
    }
    source_target = args.base_job / "target.png"
    source_mask = Path(prepared["mask_path"])
    visual_hashes = {
        name: {
            "path": str(path.relative_to(args.out)),
            "sha256": _sha256(path),
            "bytes": path.stat().st_size,
        }
        for name, path in {**montage_paths, **raw_image_paths}.items()
    }
    payload = {
        "schema": "structsplat.fit040.fine-detail-visual-replay.v1",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "scope": {
            "purpose": (
                "descriptive visual replay of the FIT-039/040 winner on the "
                "same exposed Janelle target used by the stage experiments"
            ),
            "fit042_independent_confirmation": False,
            "reason": (
                "FIT-042 excludes frame_00008/C0001 and all FIT-023--041 "
                "development sources"
            ),
            "claim_change_authorized": False,
            "published_fit040_endpoint_reproduced": False,
            "comparability": (
                "This replay uses the exact target, mask, and RTX-4090 "
                "replication base from the preceding stage audit. The original "
                "RTX-3050 FIT-040 field is no longer on disk, so published "
                "FIT-040 endpoint metrics remain evidence-only."
            ),
        },
        "source": {
            "base_job": str(args.base_job.resolve()),
            "base_field": str(prepared["field_path"]),
            "base_field_sha256": _sha256(prepared["field_path"]),
            "base_rows": base.n,
            "target": str(source_target.resolve()),
            "target_sha256": _sha256(source_target),
            "mask": str(source_mask.resolve()),
            "mask_sha256": _sha256(source_mask),
            "selection_artifact": str(args.selection.resolve()),
            "selection_artifact_sha256": _sha256(args.selection),
        },
        "environment": {
            "device": str(device),
            "gpu": (
                torch.cuda.get_device_name(device)
                if device.type == "cuda"
                else None
            ),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "numpy": np.__version__,
            "pillow": PILLOW_VERSION,
            "matplotlib": matplotlib.__version__,
            "git_commit": _git("rev-parse", "HEAD"),
            "git_dirty": bool(_git("status", "--porcelain")),
        },
        "protocol": {
            "method": "FIT-040 production orthogonal fine-detail pursuit",
            "fit_config": asdict(cfg),
            "schedule": asdict(schedule),
            "seed": 0,
            "visual_selection": {
                "rule": (
                    "reuse FIT-033's first base high-pass site; selected before "
                    "the winning treatment"
                ),
                "center_yx": list(center_yx),
                "half_size": half_size,
                "detail_bounds_xyxy": list(detail_bounds),
                "subject_view_bounds_xyxy": list(view_bounds),
                "error_scale": error_scale,
                "highpass_error_scale": highpass_scale,
                "highpass_delta_abs_p99": highpass_delta_scale,
                "selection_record": selection["selection"]["highpass"],
            },
            "executed_sources": [
                {
                    "path": relative,
                    "sha256": _sha256(REPOSITORY_ROOT / relative),
                }
                for relative in SOURCE_FILES
            ],
        },
        "result": {
            "elapsed_seconds": elapsed,
            "replay_metrics": replay["metrics"],
            "baseline_metrics": baseline_metrics,
            "winner_metrics": winner_metrics,
            "pursuit_tail": pursuit,
            "fixed_crop": {
                "deep_pixels": int(crop_deep.sum()),
                "highpass_mse_before": local_before,
                "highpass_mse_after": local_after,
                "highpass_reduction": local_reduction,
            },
            "saved_base_render_parity": _saved_base_render_parity(
                args.base_job,
                base_image,
            ),
            "field": {
                "path": str(field_path.relative_to(args.out)),
                "sha256": _sha256(field_path),
                "rows": winner.n,
            },
        },
        "visuals": visual_hashes,
    }
    result_path = args.out / "result.json"
    _atomic_json(result_path, payload)
    print(
        json.dumps(
            {
                "out": str(args.out),
                "target_reached": pursuit["target_reached"],
                "added_rows": pursuit["activated_rows"],
                "highpass_reduction": pursuit["highpass_reduction"],
                "laplacian_reduction": pursuit["laplacian_reduction"],
                "fixed_crop_highpass_reduction": local_reduction,
                "result_sha256": _sha256(result_path),
            },
            indent=2,
            sort_keys=True,
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-job", type=Path, default=DEFAULT_BASE_JOB)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--renderer", default="cuda")
    parser.add_argument("--mask-margin", type=float, default=0.75)
    parser.add_argument("--boundary-band", type=float, default=4.0)
    parser.add_argument("--coverage-tau", type=float, default=0.05)
    parser.add_argument("--quiet", action="store_true")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
