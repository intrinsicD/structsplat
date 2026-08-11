#!/usr/bin/env python3
"""Run HIER-022's frozen normalized-to-additive pure-Gaussian diagnostic.

The controller first calibrates the coverage weight on five programmatic fixtures, records that
decision, and only then opens the four bound COCO sources.  The resulting bundle is explicitly
development evidence: it is source-bound, produced from a dirty implementation revision, and
cannot promote a renderer default or publication claim.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import csv
from html import escape
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
for candidate in (ROOT, ROOT / "src"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from scripts.experiments import hier005_pixel_contraction as report_utils  # noqa: E402
from scripts.experiments import hier010_residual_anchor_projection as viz_utils  # noqa: E402
from structsplat.additive_continuation import (  # noqa: E402
    AdditiveContinuationConfig,
    AdditiveContinuationResult,
    fit_additive_continuation,
)
from structsplat.cli import save_error_heatmap, save_image  # noqa: E402
from structsplat.config import FitConfig, InitConfig, StructureTensorConfig  # noqa: E402
from structsplat.fit import fit  # noqa: E402
from structsplat.gaussians import GaussianField  # noqa: E402
from structsplat.init import build_field  # noqa: E402
from structsplat.render import render_field  # noqa: E402


REPORT_SCHEMA = "structsplat.hier022_additive_continuation.diagnostic.v1"
ARMS = (
    "normalized_plain",
    "additive_plain",
    "continuation_no_coverage",
    "continuation_coverage",
)
COVERAGE_WEIGHTS = (0.01, 0.05, 0.2)
SOURCE_BINDINGS = {
    "COCO_train2014_000000000009.jpg": (
        "35cdfe8259aca40d564baf33ee749d82ce852446bd9574f0c47551d8bfffda99"
    ),
    "COCO_train2014_000000000025.jpg": (
        "d8f12a26d8803701cabac80494b080f998e5ed9bafaf61a2825ce6212c85487a"
    ),
    "COCO_train2014_000000000030.jpg": (
        "0444b10826d376ad9075805061405f6071a62b80eda29c5f284ed77b093d5b1d"
    ),
    "COCO_train2014_000000000034.jpg": (
        "2c46871034fa901ae795a8bb916ba7f2f728507cab9e511cced0986bd083d193"
    ),
}
CALIBRATION_SIZE = 48
CALIBRATION_BUDGET = 128
CALIBRATION_SEED = 0
CALIBRATION_STEPS = 160
CHECKPOINT_EVERY = 25
FEATURE_CAP_PX = 12.0
COEFFICIENT_LIMIT = 16.0
PARITY_LIMIT = 2e-5


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--max-side", type=int, default=160)
    parser.add_argument("--budgets", type=int, nargs="+", default=[640])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--iters", type=int, default=500)
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    parser.add_argument(
        "--coverage-weights", type=float, nargs="+", default=list(COVERAGE_WEIGHTS)
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--lpips", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--render-chunk", type=int, default=256)
    parser.add_argument("--error-scale", type=float, default=4.0)
    parser.add_argument("--resume", action="store_true")
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    frozen = {
        "max_side": 160,
        "budgets": [640],
        "seeds": [0, 1],
        "iters": 500,
        "arms": list(ARMS),
        "coverage_weights": list(COVERAGE_WEIGHTS),
        "device": "cuda",
        "lpips": True,
        "render_chunk": 256,
    }
    for name, expected in frozen.items():
        if getattr(args, name) != expected:
            raise SystemExit(
                f"frozen HIER-022 protocol requires {name}={expected!r}, "
                f"got {getattr(args, name)!r}"
            )
    if not math.isfinite(args.error_scale) or args.error_scale <= 0.0:
        raise SystemExit("--error-scale must be finite and positive")
    if not args.images.is_dir():
        raise SystemExit(f"image directory does not exist: {args.images}")


def _command() -> str:
    return " ".join(shlex.quote(value) for value in sys.argv)


def _write_json(path: Path, value: object) -> None:
    report_utils._write_json(path, value)


def _git_record() -> dict[str, object]:
    return report_utils._git_record()


def _discover_sources(root: Path) -> list[Path]:
    paths: list[Path] = []
    actual: dict[str, str] = {}
    for name in SOURCE_BINDINGS:
        path = root / name
        if path.is_file():
            actual[name] = report_utils._sha256(path)
            paths.append(path.resolve())
    if actual != SOURCE_BINDINGS:
        raise SystemExit(
            "HIER-022 source bank is missing or hash-mismatched: "
            f"expected {SOURCE_BINDINGS}, got {actual}"
        )
    return paths


def _snapshot_sources(output_root: Path) -> list[dict[str, object]]:
    paths = (
        Path(__file__).resolve(),
        ROOT / "src" / "structsplat" / "additive_continuation.py",
        ROOT / "tests" / "test_additive_continuation.py",
        ROOT / "tasks" / "HIER-022-normalized-to-additive-continuation.md",
        ROOT / "docs" / "research" / "2026-08-11-pure-gaussian-additive-continuation.md",
        ROOT / "scripts" / "check_report_bundle.py",
    )
    records: list[dict[str, object]] = []
    for source in paths:
        relative = source.relative_to(ROOT)
        destination = output_root / "source_snapshot" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        records.append(
            {
                "repository_path": str(relative),
                "snapshot_path": str(destination.relative_to(output_root)),
                "bytes": destination.stat().st_size,
                "sha256": report_utils._sha256(destination),
            }
        )
    return records


def _installed_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _environment(torch) -> dict[str, object]:
    gpu: dict[str, object] | None = None
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        gpu = {
            "name": props.name,
            "total_memory_bytes": int(props.total_memory),
            "capability": list(torch.cuda.get_device_capability(0)),
            "cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
        }
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "executable": sys.executable,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "pillow": _installed_version("pillow"),
        "lpips": _installed_version("lpips"),
        "pytorch_msssim": _installed_version("pytorch-msssim"),
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu": gpu,
        "pid": os.getpid(),
    }


def _init_config(count: int, seed: int) -> InitConfig:
    return InitConfig(
        strategy="aniso_onedge",
        num_gaussians=count,
        seed=seed,
        sampling_mode="wse",
        flank_offset_frac=0.0,
        scale_cap_mode="feature",
        scale_cap_max=FEATURE_CAP_PX,
        color_mode="bilinear",
    )


def _fit_config(args: argparse.Namespace, arm: str) -> FitConfig:
    renderer = "cuda" if arm == "normalized_plain" else "cuda_additive"
    return FitConfig(
        iters=args.iters,
        lr_means=5e-2,
        lr_scales=3e-2,
        lr_rot=1e-2,
        lr_color=3e-2,
        lr_opacity=1e-2,
        optimizer="adam",
        pixel_loss="l1",
        ssim_weight=0.3,
        log_every=CHECKPOINT_EVERY,
        checkpoint_policy="best_psnr_final_count",
        sigma_cutoff=3.0,
        support_fade=False,
        aa_dilation=0.0,
        render_chunk=args.render_chunk,
        renderer=renderer,
        color_basis="constant",
        compute_lpips=False,
        max_gaussians=args.budgets[0],
    )


def _continuation_config(
    args: argparse.Namespace,
    coverage_weight: float,
    *,
    steps: int | None = None,
) -> AdditiveContinuationConfig:
    return AdditiveContinuationConfig(
        steps=args.iters if steps is None else steps,
        checkpoint_every=CHECKPOINT_EVERY,
        lr_means=5e-2,
        lr_scales=3e-2,
        lr_rotations=1e-2,
        lr_coefficients=3e-2,
        lr_masses=1e-2,
        pixel_loss="l1",
        ssim_weight=0.3,
        coverage_weight=coverage_weight,
        normalization_eps=1e-8,
        coefficient_abs_limit=COEFFICIENT_LIMIT,
        log_mass_abs_limit=8.0,
        min_scale_px=0.35,
        sigma_cutoff=3.0,
        aa_dilation=0.0,
        render_chunk=args.render_chunk,
        renderer="cuda_additive",
        support_fade=False,
    )


def _synthetic_fixtures() -> dict[str, np.ndarray]:
    size = CALIBRATION_SIZE
    yy, xx = np.mgrid[:size, :size]
    x = xx.astype(np.float32) / float(size - 1)
    y = yy.astype(np.float32) / float(size - 1)
    constant = np.empty((size, size, 3), dtype=np.float32)
    constant[...] = np.asarray([0.2, 0.5, 0.8], dtype=np.float32)
    ramp = np.stack([x, y, 0.5 * (x + y)], axis=2).astype(np.float32)
    edge = np.empty_like(constant)
    edge[:, : size // 2] = np.asarray([0.1, 0.2, 0.8], dtype=np.float32)
    edge[:, size // 2 :] = np.asarray([0.9, 0.8, 0.1], dtype=np.float32)
    checker_value = ((xx // 4 + yy // 4) % 2).astype(np.float32)
    checker = np.repeat(checker_value[:, :, None], 3, axis=2)
    isolated = np.full((size, size, 3), 0.35, dtype=np.float32)
    isolated[12, 12] = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    isolated[13, 35] = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
    isolated[35, 24] = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
    return {
        "constant": constant,
        "ramp": ramp,
        "edge": edge,
        "checker": checker,
        "isolated_detail": isolated,
    }


def _run_calibration(args: argparse.Namespace, output_root: Path, torch) -> float:
    calibration_root = output_root / "calibration"
    calibration_root.mkdir(parents=True, exist_ok=True)
    fixtures = _synthetic_fixtures()
    rows: list[dict[str, object]] = []
    tensor_config = StructureTensorConfig()
    for fixture_name, target in fixtures.items():
        fixture_root = calibration_root / fixture_name
        fixture_root.mkdir(parents=True, exist_ok=True)
        save_image(str(fixture_root / "source.png"), target)
        np.random.seed(CALIBRATION_SEED)
        torch.manual_seed(CALIBRATION_SEED)
        torch.cuda.manual_seed_all(CALIBRATION_SEED)
        initial = build_field(
            target,
            _init_config(CALIBRATION_BUDGET, CALIBRATION_SEED),
            tensor_config,
            device=args.device,
        )
        for weight in args.coverage_weights:
            cell_root = fixture_root / f"coverage_{weight:g}"
            cell_root.mkdir(parents=True, exist_ok=True)
            config = _continuation_config(args, weight, steps=CALIBRATION_STEPS)
            torch.cuda.reset_peak_memory_stats()
            result = fit_additive_continuation(
                initial.detached(), target, config=config, verbose=False
            )
            torch.cuda.synchronize()
            coefficient_abs_max = float(result.field.colors.detach().abs().max().cpu())
            mse = float(np.mean(
                (result.reconstruction_raw.astype(np.float64) - target.astype(np.float64)) ** 2
            ))
            result.field.save(str(cell_root / "field.gaussian.npz"))
            save_image(str(cell_root / "reconstruction.png"), result.reconstruction_raw)
            _write_json(cell_root / "history.json", result.checkpoint_records())
            row = {
                "fixture": fixture_name,
                "coverage_weight": float(weight),
                "terminal_mse": mse,
                "terminal_psnr_db": -10.0 * math.log10(max(mse, 1e-12)),
                "coefficient_abs_max": coefficient_abs_max,
                "finite": bool(np.isfinite(result.reconstruction_raw).all()),
                "completed": result.completed,
                "selected_step": result.selected_step,
                "endpoint_parity_max_abs": result.endpoint_parity_max_abs,
                "selected_coverage_loss": result.selected_coverage_loss,
                "renderer_calls": result.renderer_calls,
                "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                "field_path": str((cell_root / "field.gaussian.npz").relative_to(output_root)),
                "history_path": str((cell_root / "history.json").relative_to(output_root)),
            }
            _write_json(cell_root / "row.json", row)
            rows.append(row)
    _write_json(calibration_root / "rows.json", rows)

    aggregates: dict[str, dict[str, object]] = {}
    eligible_weights: list[float] = []
    for weight in args.coverage_weights:
        selected = [row for row in rows if row["coverage_weight"] == float(weight)]
        eligible = bool(
            len(selected) == len(fixtures)
            and all(
                row["completed"]
                and row["finite"]
                and float(row["coefficient_abs_max"]) <= COEFFICIENT_LIMIT
                for row in selected
            )
        )
        aggregates[f"{weight:g}"] = {
            "coverage_weight": float(weight),
            "eligible": eligible,
            "mean_terminal_mse": float(
                np.mean([float(row["terminal_mse"]) for row in selected])
            ),
            "max_coefficient_abs": max(
                float(row["coefficient_abs_max"]) for row in selected
            ),
        }
        if eligible:
            eligible_weights.append(float(weight))
    if not eligible_weights:
        raise RuntimeError("no calibration coverage weight satisfies the frozen finite/bounded gate")
    selected_weight = min(
        eligible_weights,
        key=lambda value: (float(aggregates[f"{value:g}"]["mean_terminal_mse"]), value),
    )
    decision = {
        "schema": REPORT_SCHEMA,
        "status": "diagnostic",
        "phase": "synthetic_calibration",
        "fixtures": list(fixtures),
        "size": CALIBRATION_SIZE,
        "budget": CALIBRATION_BUDGET,
        "seed": CALIBRATION_SEED,
        "steps": CALIBRATION_STEPS,
        "candidate_weights": list(args.coverage_weights),
        "selection_rule": (
            "lowest mean exact-additive terminal MSE among finite completed cells with "
            "coefficient_abs_max <= 16; ties choose smaller weight"
        ),
        "aggregates": aggregates,
        "selected_coverage_weight": selected_weight,
        "recorded_before_natural_execution": True,
        "recorded_unix_seconds": time.time(),
    }
    _write_json(calibration_root / "decision.json", decision)
    (calibration_root / "FROZEN").write_text(
        f"selected_coverage_weight={selected_weight:g}\n", encoding="utf-8"
    )
    return selected_weight


def _field_render(field: GaussianField, height: int, width: int, renderer: str,
                  render_chunk: int):
    return render_field(
        field.means,
        field.conics(),
        field.colors,
        field.radii(3.0),
        height,
        width,
        chunk=render_chunk,
        mode=renderer,
        scales=field.scales(),
        rotations=field.rotations,
        support_fade=False,
        sigma_cutoff=3.0,
    )


def _unit_coverage(field: GaussianField, height: int, width: int, renderer: str,
                   render_chunk: int):
    import torch

    return render_field(
        field.means,
        field.conics(),
        torch.ones_like(field.colors),
        field.radii(3.0),
        height,
        width,
        chunk=render_chunk,
        mode=renderer,
        scales=field.scales(),
        rotations=field.rotations,
        support_fade=False,
        sigma_cutoff=3.0,
    )[..., :1]


def _coverage_record(denominator) -> dict[str, float]:
    import torch

    quantiles = torch.as_tensor(
        [0.0, 0.01, 0.05, 0.5, 0.95, 0.99, 1.0],
        device=denominator.device,
        dtype=denominator.dtype,
    )
    values = torch.quantile(denominator.reshape(-1), quantiles).detach().cpu().numpy()
    return {
        "coverage_loss": float((denominator - 1.0).square().mean().detach().cpu()),
        "denominator_min": float(values[0]),
        "denominator_q01": float(values[1]),
        "denominator_q05": float(values[2]),
        "denominator_q50": float(values[3]),
        "denominator_q95": float(values[4]),
        "denominator_q99": float(values[5]),
        "denominator_max": float(values[6]),
    }


def _selected_continuation_coverage(result: AdditiveContinuationResult) -> dict[str, float]:
    selected = next(checkpoint for checkpoint in result.checkpoints if checkpoint.selected)
    return {
        "coverage_loss": selected.coverage_loss,
        "denominator_min": selected.denominator_min,
        "denominator_q01": selected.denominator_q01,
        "denominator_q05": selected.denominator_q05,
        "denominator_q50": selected.denominator_q50,
        "denominator_q95": selected.denominator_q95,
        "denominator_q99": selected.denominator_q99,
        "denominator_max": selected.denominator_max,
    }


def _coefficient_record(field: GaussianField) -> dict[str, float]:
    colors = field.colors.detach().cpu().numpy().astype(np.float64)
    absolute = np.abs(colors)
    positive = np.maximum(colors, 0.0).sum(axis=0)
    negative = np.maximum(-colors, 0.0).sum(axis=0)
    cancellation = (positive + negative) / np.maximum(np.abs(positive - negative), 1e-12)
    return {
        "coefficient_min": float(colors.min()),
        "coefficient_max": float(colors.max()),
        "coefficient_abs_max": float(absolute.max()),
        "coefficient_abs_median": float(np.median(absolute)),
        "coefficient_abs_q99": float(np.quantile(absolute, 0.99)),
        "coefficient_negative_fraction": float(np.mean(colors < 0.0)),
        "coefficient_cancellation_ratio_mean": float(cancellation.mean()),
        "coefficient_cancellation_ratio_max": float(cancellation.max()),
    }


def _display_metrics(reconstruction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    shown = np.rint(np.clip(reconstruction, 0.0, 1.0) * 255.0) / 255.0
    reference = np.rint(np.clip(target, 0.0, 1.0) * 255.0) / 255.0
    mse = float(np.mean((shown.astype(np.float64) - reference.astype(np.float64)) ** 2))
    return {
        "display_mse": mse,
        "display_psnr_db": -10.0 * math.log10(max(mse, 1e-12)),
    }


def _trajectory_baseline(result: dict[str, object], steps: int) -> list[dict[str, float]]:
    history = result["history"]
    records = [{"step": 0.0, "psnr_db": float(history["psnr"][0])}]
    checkpoints = result["checkpoint_history"]
    for step, value in zip(checkpoints["iter"], checkpoints["psnr"], strict=True):
        records.append({"step": float(step), "psnr_db": float(value)})
    return _normalize_trajectory(records, steps)


def _trajectory_continuation(
    result: AdditiveContinuationResult, steps: int
) -> list[dict[str, float]]:
    records = [
        {"step": float(checkpoint.step), "psnr_db": checkpoint.raw_psnr_db}
        for checkpoint in result.checkpoints
    ]
    return _normalize_trajectory(records, steps)


def _normalize_trajectory(
    records: list[dict[str, float]], steps: int
) -> list[dict[str, float]]:
    by_step = {float(record["step"]): float(record["psnr_db"]) for record in records}
    ordered = [{"step": step, "psnr_db": by_step[step]} for step in sorted(by_step)]
    if ordered[0]["step"] != 0.0:
        ordered.insert(0, {"step": 0.0, "psnr_db": ordered[0]["psnr_db"]})
    if ordered[-1]["step"] < float(steps):
        ordered.append({"step": float(steps), "psnr_db": ordered[-1]["psnr_db"]})
    return ordered


def _psnr_auc(records: list[dict[str, float]], steps: int) -> float:
    x = np.asarray([record["step"] for record in records], dtype=np.float64)
    y = np.asarray([record["psnr_db"] for record in records], dtype=np.float64)
    return float(np.trapezoid(y, x) / float(steps))


def _write_curve(path: Path, trajectory: list[dict[str, float]], title: str) -> None:
    width, height = 720, 300
    margin = 42
    xs = np.asarray([row["step"] for row in trajectory], dtype=np.float64)
    ys = np.asarray([row["psnr_db"] for row in trajectory], dtype=np.float64)
    xmin, xmax = float(xs.min()), max(float(xs.max()), 1.0)
    ymin, ymax = float(ys.min()), float(ys.max())
    if ymax <= ymin:
        ymax = ymin + 1.0
    px = margin + (xs - xmin) / (xmax - xmin) * (width - 2 * margin)
    py = height - margin - (ys - ymin) / (ymax - ymin) * (height - 2 * margin)
    points = " ".join(f"{x:.2f},{y:.2f}" for x, y in zip(px, py, strict=True))
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">
<rect width="100%" height="100%" fill="white"/><text x="{margin}" y="22" font-size="14">{escape(title)}</text>
<line x1="{margin}" y1="{height-margin}" x2="{width-margin}" y2="{height-margin}" stroke="#555"/>
<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height-margin}" stroke="#555"/>
<polyline fill="none" stroke="#1769aa" stroke-width="2" points="{points}"/>
<text x="{margin}" y="{height-8}" font-size="11">0</text><text x="{width-margin-30}" y="{height-8}" font-size="11">{int(xmax)} steps</text>
<text x="4" y="{height-margin}" font-size="11">{ymin:.1f}</text><text x="4" y="{margin+4}" font-size="11">{ymax:.1f} dB</text>
</svg>"""
    path.write_text(svg, encoding="utf-8")


def _save_visuals(
    artifact_dir: Path,
    target: np.ndarray,
    reconstruction: np.ndarray,
    error_scale: float,
) -> tuple[int, int, int, int]:
    mask = np.ones(target.shape[:2], dtype=bool)
    save_image(str(artifact_dir / "source.png"), target)
    save_image(str(artifact_dir / "reconstruction.png"), reconstruction)
    save_error_heatmap(
        str(artifact_dir / "error.png"), reconstruction - target, scale=error_scale
    )
    bounds = viz_utils._worst_crop_bounds(reconstruction, target, mask)
    viz_utils._save_crop(artifact_dir / "source_crop.png", target, bounds)
    viz_utils._save_crop(artifact_dir / "reconstruction_crop.png", reconstruction, bounds)
    shown_error = np.repeat(
        np.clip(
            np.mean(np.abs(reconstruction.astype(np.float64) - target), axis=2)
            * error_scale,
            0.0,
            1.0,
        )[:, :, None],
        3,
        axis=2,
    )
    viz_utils._save_crop(artifact_dir / "error_crop.png", shown_error, bounds)
    return bounds


def _run_method(
    initial: GaussianField,
    target: np.ndarray,
    arm: str,
    selected_weight: float,
    args: argparse.Namespace,
    torch,
) -> dict[str, object]:
    target_tensor = torch.as_tensor(target, device=args.device, dtype=torch.float32)
    torch.cuda.reset_peak_memory_stats()
    fit_started = time.perf_counter()
    if arm in ("normalized_plain", "additive_plain"):
        fit_config = _fit_config(args, arm)
        result = fit(initial.detached(), target_tensor, fit_config, verbose=False)
        torch.cuda.synchronize()
        field = result["field"]
        expected = result["render"].detach().cpu().numpy().astype(np.float32, copy=False)
        trajectory = _trajectory_baseline(result, args.iters)
        renderer_calls = (
            args.iters
            + len(result["checkpoint_history"]["iter"])
            + int(result["selected_from_checkpoint"])
        )
        selected_step = int(result["selected_iter"])
        completed = bool(result["iterations_run"] == args.iters)
        method_status = "completed" if completed else "incomplete"
        history: object = result["history"]
        history_extra = {"checkpoint_history": result["checkpoint_history"]}
        coverage_weight = 0.0
        initial_mass = None
        endpoint_parity = 0.0
        semantic_family = (
            "normalized_weighted_sum_v1"
            if arm == "normalized_plain"
            else "additive_rgb_peak_one_v1"
        )
        renderer = fit_config.renderer
        fit_seconds = float(result["fit_seconds"])
        with torch.no_grad():
            coverage = _coverage_record(
                _unit_coverage(field, target.shape[0], target.shape[1],
                               "cuda_additive", args.render_chunk)
            )
        coverage_scope = "unit_mass_basis_diagnostic"
        renderer_calls_coverage = 1
    else:
        coverage_weight = 0.0 if arm == "continuation_no_coverage" else selected_weight
        continuation_config = _continuation_config(args, coverage_weight)
        result = fit_additive_continuation(
            initial.detached(), target, config=continuation_config, verbose=False
        )
        torch.cuda.synchronize()
        field = result.field
        expected = result.reconstruction_raw
        trajectory = _trajectory_continuation(result, args.iters)
        renderer_calls = result.renderer_calls
        selected_step = result.selected_step
        completed = result.completed
        method_status = result.status
        history = result.checkpoint_records()
        history_extra = {}
        initial_mass = result.initial_mass
        endpoint_parity = result.endpoint_parity_max_abs
        semantic_family = "additive_rgb_peak_one_v1"
        renderer = continuation_config.renderer
        fit_seconds = result.elapsed_seconds
        coverage = _selected_continuation_coverage(result)
        coverage_scope = "selected_training_mass_diagnostic_discarded_at_endpoint"
        renderer_calls_coverage = 0
    peak = int(torch.cuda.max_memory_allocated())
    return {
        "field": field,
        "expected": expected,
        "trajectory": trajectory,
        "renderer_calls": renderer_calls,
        "renderer_calls_coverage": renderer_calls_coverage,
        "selected_step": selected_step,
        "completed": completed,
        "method_status": method_status,
        "history": history,
        "history_extra": history_extra,
        "coverage_weight": coverage_weight,
        "initial_mass": initial_mass,
        "endpoint_parity": endpoint_parity,
        "semantic_family": semantic_family,
        "renderer": renderer,
        "fit_seconds": fit_seconds,
        "wall_fit_seconds": time.perf_counter() - fit_started,
        "peak_cuda_allocated_bytes": peak,
        "coverage": coverage,
        "coverage_scope": coverage_scope,
    }


def _write_cell(
    *,
    output_root: Path,
    image_path: Path,
    target: np.ndarray,
    raster: dict[str, object],
    seed: int,
    budget: int,
    arm: str,
    initial_field_sha256: str,
    init_seconds: float,
    method: dict[str, object],
    args: argparse.Namespace,
    torch,
) -> dict[str, object]:
    artifact_dir = output_root / "artifacts" / f"{image_path.stem}__s{seed}__n{budget}__{arm}"
    artifact_dir.mkdir(parents=True, exist_ok=False)
    field: GaussianField = method["field"]
    field_path = artifact_dir / "field.gaussian.npz"
    field.save(str(field_path))
    with np.load(field_path) as payload:
        field_keys = sorted(payload.files)
    decode_started = time.perf_counter()
    cold_field = GaussianField.load(str(field_path), device=args.device)
    decode_seconds = time.perf_counter() - decode_started
    render_started = time.perf_counter()
    with torch.no_grad():
        cold_tensor = _field_render(
            cold_field, target.shape[0], target.shape[1], method["renderer"], args.render_chunk
        )
        repeated_tensor = _field_render(
            cold_field, target.shape[0], target.shape[1], method["renderer"], args.render_chunk
        )
    torch.cuda.synchronize()
    render_seconds = time.perf_counter() - render_started
    cold = cold_tensor.detach().cpu().numpy().astype(np.float32, copy=False)
    repeated = repeated_tensor.detach().cpu().numpy().astype(np.float32, copy=False)
    expected = np.asarray(method["expected"], dtype=np.float32)
    metric_started = time.perf_counter()
    metrics = report_utils._metric_values(
        cold,
        target,
        np.ones(target.shape[:2], dtype=bool),
        device=args.device,
        compute_lpips=args.lpips,
    )
    if metrics["lpips"] is None:
        raise RuntimeError(f"LPIPS is required but unavailable: {metrics['lpips_error']}")
    metric_seconds = time.perf_counter() - metric_started
    bounds = _save_visuals(artifact_dir, target, cold, args.error_scale)
    trajectory = method["trajectory"]
    _write_curve(artifact_dir / "learning_curve.svg", trajectory, f"{image_path.stem} {arm}")
    _write_json(artifact_dir / "fit_history.json", method["history"])
    _write_json(artifact_dir / "projection_history.json", [])
    _write_json(artifact_dir / "geometry_history.json", [])
    cell_config = {
        "schema": REPORT_SCHEMA,
        "arm": arm,
        "seed": seed,
        "budget": budget,
        "init": asdict(_init_config(budget, seed)),
        "fit": (
            asdict(_fit_config(args, arm))
            if arm in ("normalized_plain", "additive_plain")
            else asdict(_continuation_config(args, float(method["coverage_weight"])))
        ),
    }
    _write_json(artifact_dir / "config.json", cell_config)
    np.savez_compressed(
        artifact_dir / "analysis.npz",
        crop_bounds=np.asarray(bounds, dtype=np.int32),
        reconstruction_raw=cold,
        error_raw=cold.astype(np.float32) - target.astype(np.float32),
        trajectory_step=np.asarray([row["step"] for row in trajectory], dtype=np.float32),
        trajectory_psnr_db=np.asarray(
            [row["psnr_db"] for row in trajectory], dtype=np.float32
        ),
    )
    auxiliary_rgb = any(key in field_keys for key in ("color_grads", "opacities"))
    mass_payload = any("mass" in key.lower() for key in field_keys)
    cold_parity = float(np.max(np.abs(cold.astype(np.float64) - expected.astype(np.float64))))
    repeated_parity = float(
        np.max(np.abs(repeated.astype(np.float64) - cold.astype(np.float64)))
    )
    row: dict[str, object] = {
        "schema": REPORT_SCHEMA,
        "status": "diagnostic",
        "phase": "development",
        "image": image_path.stem,
        "arm": arm,
        "seed": seed,
        "semantic_family": method["semantic_family"],
        "artifact_dir": str(artifact_dir.relative_to(output_root)),
        "source_path": str(image_path),
        "source_sha256": report_utils._sha256(image_path),
        "source_file_bytes": image_path.stat().st_size,
        "original_width": raster["original_width"],
        "original_height": raster["original_height"],
        "width": target.shape[1],
        "height": target.shape[0],
        "active_pixels": int(target.shape[0] * target.shape[1]),
        "target_gaussians": budget,
        "n_gaussians": field.n,
        "initial_field_sha256": initial_field_sha256,
        "field_file_sha256": report_utils._sha256(field_path),
        "field_file_bytes": field_path.stat().st_size,
        "field_npz_keys": field_keys,
        "mass_payload_present": mass_payload,
        "auxiliary_rgb_payload_present": auxiliary_rgb,
        "method_status": method["method_status"],
        "completed": method["completed"],
        "selected_step": method["selected_step"],
        "selected_lambda": 0.0 if arm.startswith("continuation") else None,
        "coverage_weight": method["coverage_weight"],
        "coverage_scope": method["coverage_scope"],
        "initial_mass": method["initial_mass"],
        "endpoint_internal_parity_max_abs": method["endpoint_parity"],
        "attempted_steps": args.iters,
        "psnr_auc_attempted_step": _psnr_auc(trajectory, args.iters),
        "renderer_calls_fit": method["renderer_calls"],
        "renderer_calls_coverage_diagnostic": method["renderer_calls_coverage"],
        "init_seconds": init_seconds,
        "fit_seconds": method["fit_seconds"],
        "wall_fit_seconds": method["wall_fit_seconds"],
        "cold_decode_seconds": decode_seconds,
        "render_seconds": render_seconds,
        "metric_seconds": metric_seconds,
        "pipeline_algorithm_seconds": init_seconds + float(method["fit_seconds"]),
        "peak_cuda_allocated_bytes": method["peak_cuda_allocated_bytes"],
        "maintained_render_parity_max_abs": cold_parity,
        "repeated_render_parity_max_abs": repeated_parity,
        "finite_reconstruction": bool(np.isfinite(cold).all()),
        "raw_mse": metrics["masked_mse"],
        **_display_metrics(cold, target),
        **_coefficient_record(field),
        **method["coverage"],
        **metrics,
    }
    _write_json(artifact_dir / "row.json", row)
    return row


def _write_tables(output_root: Path, rows: list[dict[str, object]]) -> None:
    _write_json(
        output_root / "metrics.json",
        {"schema": REPORT_SCHEMA, "status": "diagnostic", "rows": rows},
    )
    with (output_root / "metrics.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    columns = sorted({key for row in rows for key in row})
    with (output_root / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _mean(rows: list[dict[str, object]], key: str) -> float:
    return float(np.mean([float(row[key]) for row in rows]))


def _decision(rows: list[dict[str, object]], selected_weight: float) -> dict[str, object]:
    expected_count = len(SOURCE_BINDINGS) * 2
    by_arm = {arm: [row for row in rows if row["arm"] == arm] for arm in ARMS}
    complete = all(len(by_arm[arm]) == expected_count for arm in ARMS)
    aggregates = {
        arm: {
            "cell_count": len(by_arm[arm]),
            "mean_psnr_db": _mean(by_arm[arm], "psnr_db") if by_arm[arm] else None,
            "mean_ms_ssim": _mean(by_arm[arm], "ms_ssim") if by_arm[arm] else None,
            "mean_lpips": _mean(by_arm[arm], "lpips") if by_arm[arm] else None,
            "mean_pixel_max": _mean(by_arm[arm], "artifact_pixel_rmse_max")
            if by_arm[arm] else None,
            "mean_patch7_max": _mean(by_arm[arm], "artifact_patch_rmse_max_7")
            if by_arm[arm] else None,
            "mean_coverage_loss": _mean(by_arm[arm], "coverage_loss")
            if by_arm[arm] else None,
            "mean_psnr_auc": _mean(by_arm[arm], "psnr_auc_attempted_step")
            if by_arm[arm] else None,
            "mean_fit_seconds": _mean(by_arm[arm], "fit_seconds")
            if by_arm[arm] else None,
        }
        for arm in ARMS
    }
    gates: dict[str, bool] = {"all_cells_present": complete}
    if complete:
        normalized = by_arm["normalized_plain"]
        additive = by_arm["additive_plain"]
        no_coverage = by_arm["continuation_no_coverage"]
        continuation = by_arm["continuation_coverage"]
        keys = lambda row: (row["image"], row["seed"])
        additive_by_key = {keys(row): row for row in additive}
        gates.update(
            {
                "all_exact_additive_endpoint": all(
                    row["completed"]
                    and row["method_status"] == "completed"
                    and row["selected_lambda"] == 0.0
                    and row["n_gaussians"] == row["target_gaussians"] == 640
                    and row["finite_reconstruction"]
                    and float(row["coefficient_abs_max"]) <= COEFFICIENT_LIMIT
                    and float(row["maintained_render_parity_max_abs"]) <= PARITY_LIMIT
                    and not row["mass_payload_present"]
                    and not row["auxiliary_rgb_payload_present"]
                    for row in continuation
                ),
                "mean_psnr_within_0p25_db_normalized": (
                    _mean(continuation, "psnr_db") >= _mean(normalized, "psnr_db") - 0.25
                ),
                "closes_half_positive_normalized_additive_gap": (
                    _mean(continuation, "psnr_db") - _mean(additive, "psnr_db")
                    >= 0.5
                    * max(0.0, _mean(normalized, "psnr_db") - _mean(additive, "psnr_db"))
                ),
                "mean_lpips_no_worse_than_additive": (
                    _mean(continuation, "lpips") <= _mean(additive, "lpips")
                ),
                "all_lpips_no_more_than_0p01_worse_than_additive": all(
                    float(row["lpips"])
                    <= float(additive_by_key[keys(row)]["lpips"]) + 0.01
                    for row in continuation
                ),
                "mean_pixel_max_no_worse_than_additive": (
                    _mean(continuation, "artifact_pixel_rmse_max")
                    <= _mean(additive, "artifact_pixel_rmse_max")
                ),
                "mean_patch7_max_no_worse_than_additive": (
                    _mean(continuation, "artifact_patch_rmse_max_7")
                    <= _mean(additive, "artifact_patch_rmse_max_7")
                ),
                "coverage_mse_reduced_at_least_25_percent": (
                    _mean(continuation, "coverage_loss")
                    <= 0.75 * _mean(no_coverage, "coverage_loss")
                ),
            }
        )
    numeric_pass = bool(gates and all(gates.values()))
    if not complete:
        failure_class = "incomplete_execution"
    elif not gates.get("coverage_mse_reduced_at_least_25_percent", False):
        failure_class = "coverage_target"
    elif not gates.get("all_exact_additive_endpoint", False):
        failure_class = "basis_conditioning"
    elif not numeric_pass:
        failure_class = "continuation_path"
    else:
        failure_class = None
    return {
        "schema": REPORT_SCHEMA,
        "status": "diagnostic",
        "phase": "development",
        "selected_coverage_weight": selected_weight,
        "aggregates": aggregates,
        "gates": gates,
        "numeric_pass": numeric_pass,
        "visual_review": "pending",
        "overall_pass": False,
        "failure_class_if_numeric": failure_class,
        "formal_claim_ready": False,
        "interpretation": (
            "Numeric gates pass; full-frame and worst-crop visual review is still required."
            if numeric_pass
            else "The frozen mechanism gate failed; retain the bank and do not tune it in place."
        ),
    }


def _write_report(
    output_root: Path,
    rows: list[dict[str, object]],
    decision: dict[str, object],
) -> None:
    table_rows = []
    cards = []
    for row in rows:
        artifact = escape(str(row["artifact_dir"]))
        table_rows.append(
            "<tr>"
            f"<td>{escape(str(row['image']))}</td><td>{int(row['seed'])}</td>"
            f"<td>{escape(str(row['arm']))}</td><td>{float(row['psnr_db']):.3f}</td>"
            f"<td>{float(row['ms_ssim']):.5f}</td><td>{float(row['lpips']):.5f}</td>"
            f"<td>{float(row['artifact_pixel_rmse_max']):.4f}</td>"
            f"<td>{float(row['artifact_patch_rmse_max_7']):.4f}</td>"
            f"<td>{float(row['coverage_loss']):.4g}</td>"
            f"<td>{float(row['coefficient_abs_max']):.3f}</td>"
            f"<td><a href='{artifact}/reconstruction.png'>full</a> · "
            f"<a href='{artifact}/reconstruction_crop.png'>crop</a> · "
            f"<a href='{artifact}/error.png'>error</a> · "
            f"<a href='{artifact}/learning_curve.svg'>curve</a></td></tr>"
        )
        cards.append(
            f"<section><h3>{escape(str(row['image']))} · seed {int(row['seed'])} · "
            f"{escape(str(row['arm']))}</h3>"
            f"<a href='{artifact}/source.png'><img src='{artifact}/source.png'></a>"
            f"<a href='{artifact}/reconstruction.png'><img src='{artifact}/reconstruction.png'></a>"
            f"<a href='{artifact}/error.png'><img src='{artifact}/error.png'></a>"
            f"<a href='{artifact}/reconstruction_crop.png'>"
            f"<img src='{artifact}/reconstruction_crop.png'></a></section>"
        )
    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>HIER-022 additive continuation</title><style>
body{{font-family:system-ui;margin:2rem;max-width:1700px}}table{{border-collapse:collapse}}
th,td{{border:1px solid #bbb;padding:.35rem}}img{{max-width:300px;max-height:240px;margin:.25rem}}
section{{border-top:1px solid #aaa;margin-top:1rem}}pre,code{{white-space:pre-wrap}}
</style></head><body><h1>HIER-022 normalized-to-additive continuation</h1>
<p><strong>Consumed development diagnostic.</strong> This dirty, source-bound report is not a
formal semantic selection, codec result, default change, novelty claim, or publication claim.</p>
<p><code>{escape(_command())}</code></p>
<p><a href="config.json">config</a> · <a href="decision.json">decision</a> ·
<a href="metrics.json">JSON</a> · <a href="metrics.jsonl">JSONL</a> ·
<a href="metrics.csv">CSV</a> · <a href="attempts.json">attempts</a> ·
<a href="calibration/decision.json">calibration</a> · <a href="manifest.json">manifest</a></p>
<h2>Decision</h2><pre>{escape(json.dumps(decision, indent=2, sort_keys=True))}</pre>
<h2>Cells</h2><table><tr><th>image</th><th>seed</th><th>arm</th><th>PSNR</th>
<th>MS-SSIM</th><th>LPIPS</th><th>pixel max</th><th>7x7 max</th><th>coverage MSE</th>
<th>|p|max</th><th>artifacts</th></tr>{''.join(table_rows)}</table>
<h2>Visual audit</h2>{''.join(cards)}</body></html>"""
    (output_root / "index.html").write_text(html, encoding="utf-8")


def _write_manifest(output_root: Path) -> None:
    files = []
    for path in sorted(output_root.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            files.append(
                {
                    "path": str(path.relative_to(output_root)),
                    "bytes": path.stat().st_size,
                    "sha256": report_utils._sha256(path),
                }
            )
    _write_json(
        output_root / "manifest.json",
        {"schema": REPORT_SCHEMA, "status": "diagnostic", "files": files},
    )


def main() -> None:
    args = _parser().parse_args()
    _validate_args(args)
    if (args.out / "COMPLETED").is_file():
        raise SystemExit(f"completed HIER-022 bundle is immutable: {args.out}")
    if args.out.exists() and any(args.out.iterdir()) and not args.resume:
        raise SystemExit(f"output directory is non-empty; use --resume after interruption: {args.out}")
    args.out.mkdir(parents=True, exist_ok=True)

    import torch

    if not torch.cuda.is_available():
        raise SystemExit("frozen HIER-022 protocol requires CUDA")
    sources = _discover_sources(args.images)
    git = _git_record()
    snapshots = _snapshot_sources(args.out)
    environment = _environment(torch)
    _write_json(args.out / "environment.json", environment)
    config = {
        "schema": REPORT_SCHEMA,
        "status": "diagnostic",
        "command": _command(),
        "git": git,
        "source_snapshots": snapshots,
        "source_bindings": SOURCE_BINDINGS,
        "arguments": vars(args),
        "synthetic_calibration": {
            "size": CALIBRATION_SIZE,
            "budget": CALIBRATION_BUDGET,
            "seed": CALIBRATION_SEED,
            "steps": CALIBRATION_STEPS,
            "coverage_weights": list(COVERAGE_WEIGHTS),
        },
        "init": asdict(_init_config(args.budgets[0], args.seeds[0])),
        "structure_tensor": asdict(StructureTensorConfig()),
        "continuation_schedule": {"hold_fraction": 0.35, "anneal_fraction": 0.50,
                                  "endpoint_fraction": 0.15},
    }
    _write_json(args.out / "config.json", config)
    subprocess.run(
        ["git", "diff", "--binary", "HEAD"],
        cwd=ROOT,
        check=False,
        stdout=(args.out / "git.diff").open("wb"),
    )

    calibration_decision = args.out / "calibration" / "decision.json"
    if args.resume and calibration_decision.is_file():
        selected_weight = float(
            json.loads(calibration_decision.read_text(encoding="utf-8"))[
                "selected_coverage_weight"
            ]
        )
    else:
        selected_weight = _run_calibration(args, args.out, torch)
    (args.out / "NATURAL_STARTED").write_text(
        f"selected_coverage_weight={selected_weight:g}\n", encoding="utf-8"
    )

    rows: list[dict[str, object]] = []
    attempts: list[dict[str, object]] = []
    metrics_path = args.out / "metrics.json"
    if args.resume and metrics_path.is_file():
        rows = json.loads(metrics_path.read_text(encoding="utf-8")).get("rows", [])
    row_keys = {(row["image"], row["seed"], row["arm"]) for row in rows}
    tensor_config = StructureTensorConfig()
    for image_path in sources:
        target, mask, raster = report_utils._load_evaluation_raster(
            image_path, None, max_side=args.max_side, mask_threshold=0.5
        )
        if mask is not None:
            raise RuntimeError("HIER-022 requires an unmasked full-frame source")
        for seed in args.seeds:
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            init_started = time.perf_counter()
            initial = build_field(
                target,
                _init_config(args.budgets[0], seed),
                tensor_config,
                device=args.device,
            )
            init_seconds = time.perf_counter() - init_started
            initial_path = args.out / "initial_fields" / f"{image_path.stem}__s{seed}__n640.npz"
            initial_path.parent.mkdir(parents=True, exist_ok=True)
            if not initial_path.exists():
                initial.save(str(initial_path))
            initial_sha = report_utils._sha256(initial_path)
            for arm in args.arms:
                stable_key = (image_path.stem, seed, arm)
                if stable_key in row_keys:
                    continue
                cell_started = time.perf_counter()
                try:
                    method = _run_method(
                        initial, target, arm, selected_weight, args, torch
                    )
                    row = _write_cell(
                        output_root=args.out,
                        image_path=image_path,
                        target=target,
                        raster=raster,
                        seed=seed,
                        budget=args.budgets[0],
                        arm=arm,
                        initial_field_sha256=initial_sha,
                        init_seconds=init_seconds,
                        method=method,
                        args=args,
                        torch=torch,
                    )
                    rows.append(row)
                    row_keys.add(stable_key)
                    attempts.append(
                        {
                            "image": image_path.stem,
                            "seed": seed,
                            "arm": arm,
                            "status": "ok",
                            "elapsed_seconds": time.perf_counter() - cell_started,
                        }
                    )
                except Exception as exc:
                    attempts.append(
                        {
                            "image": image_path.stem,
                            "seed": seed,
                            "arm": arm,
                            "status": "error",
                            "error": f"{type(exc).__name__}: {exc}"[:1000],
                            "elapsed_seconds": time.perf_counter() - cell_started,
                        }
                    )
                finally:
                    _write_tables(args.out, rows)
                    _write_json(
                        args.out / "attempts.json",
                        {"schema": REPORT_SCHEMA, "status": "diagnostic", "attempts": attempts},
                    )
                    torch.cuda.empty_cache()

    decision = _decision(rows, selected_weight)
    _write_json(args.out / "decision.json", decision)
    _write_report(args.out, rows, decision)
    (args.out / "COMPLETED").write_text(
        "HIER-022 consumed development diagnostic; do not overwrite.\n", encoding="utf-8"
    )
    _write_manifest(args.out)
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
