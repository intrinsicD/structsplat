"""Clear, report-producing entry points for the current operational pipeline.

The public files in ``scripts/`` are deliberately thin. This module owns image
discovery, deterministic mask pairing, the frozen pipeline matrix, resumable
per-cell artifacts, portable HTML reports, ablation/stage registries, and
optional official GaussianImage/Image-GS subprocesses.
"""
from __future__ import annotations

import argparse
import copy
import csv
from dataclasses import replace
import hashlib
import html
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from typing import Any, Callable, Iterable

import numpy as np
from PIL import Image

from .pipeline import (
    CURRENT_PROFILE_EVIDENCE_SCOPE,
    CURRENT_PROFILE_NAME,
    MIN_MASK_MARGIN,
    PipelineConfig,
    profile_manifest,
    render_field,
    run_current_pipeline,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
IMAGE_SUFFIXES = frozenset(
    {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
)
HEADLINE_TARGETS = (28.0, 30.0, 32.0)
ABLATION_ARMS = (
    "full",
    "no_bootstrap",
    "no_coverage_growth",
    "no_detail_growth",
    "no_closure",
    "no_redistribution",
    "no_polish",
    "no_pareto_checkpoints",
    "no_boundary_specialization",
)
STAGE_VARIANTS: dict[str, tuple[str, ...]] = {
    "initialization": (
        "quadtree_wse",
        "aniso_onedge",
        "iso_blue_noise",
        "grid",
        "random",
    ),
    "storage": ("dynamic", "fixed_capacity", "geometric"),
    "checkpoint": ("pareto50", "terminal"),
    "bootstrap": ("current", "full_resolution", "disabled"),
    "coverage": ("current", "births256", "births1024", "disabled"),
    "detail": ("current", "birth_only", "disabled"),
    "closure": ("current", "generic_no_boundary", "disabled"),
    "redistribution": ("current", "events32", "events128", "disabled"),
    "polish": ("current", "steps1000", "steps4000", "disabled"),
    # BENCH-018: commit-gate granularity. `current` is the schedule's inherited 250.
    "commit_gate": ("current", "block25", "block50", "block100", "block500"),
    # FIT-028: ADR-0026 interior coverage trade-off budget. `current` is the strict 0.0 gate.
    "hole_budget": ("current", "budget1e4", "budget5e4", "budget2e3"),
}
# BENCH-018 arm values, keyed by variant name.
COMMIT_GATE_BLOCK_STEPS: dict[str, int] = {
    "block25": 25,
    "block50": 50,
    "block100": 100,
    "block500": 500,
}
# FIT-028 arm values, keyed by variant name.
HOLE_REGRESSION_BUDGETS: dict[str, float] = {
    "budget1e4": 1e-4,
    "budget5e4": 5e-4,
    "budget2e3": 2e-3,
}

ScheduleTransform = Callable[[Any], Any]
# The transactional phases whose block is the unit of discarded work (BENCH-018).
_GATED_PHASES = (
    "bootstrap",
    "coverage",
    "detail",
    "boundary",
    "redistribution",
    "polish",
)


def _atomic_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)


def _atomic_json(path: Path, payload: Any) -> None:
    _atomic_text(
        path,
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pixel_sha256(array: np.ndarray) -> str:
    value = np.rint(np.clip(array, 0.0, 1.0) * 255.0).astype(np.uint8)
    digest = hashlib.sha256()
    digest.update(np.asarray(value.shape, dtype="<i8").tobytes())
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _git(*arguments: str) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPOSITORY_ROOT), *arguments],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _repository_state() -> dict[str, Any]:
    status = _git("status", "--short")
    return {
        "commit": _git("rev-parse", "HEAD"),
        "branch": _git("branch", "--show-current"),
        "dirty": bool(status),
        "status_sha256": hashlib.sha256(status.encode()).hexdigest(),
    }


def _slug(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return cleaned or "item"


def _save_rgb(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pixels = np.clip(np.asarray(array) * 255.0 + 0.5, 0, 255).astype(np.uint8)
    Image.fromarray(pixels, mode="RGB").save(path)


def _save_error(path: Path, reconstruction: np.ndarray, target: np.ndarray) -> None:
    error = np.clip(
        4.0 * np.mean(np.abs(reconstruction - target), axis=2), 0.0, 1.0
    )
    heat = np.stack(
        (
            error,
            np.power(error, 1.5) * 0.55,
            np.power(error, 3.0) * 0.18,
        ),
        axis=2,
    )
    _save_rgb(path, heat)


def _load_rgb(path: Path, max_side: int | None) -> tuple[np.ndarray, tuple[int, int]]:
    with Image.open(path) as source:
        image = source.convert("RGB")
        original_size = image.size
        if max_side is not None and max(image.size) > int(max_side):
            scale = float(max_side) / max(image.size)
            image = image.resize(
                (
                    max(1, round(image.size[0] * scale)),
                    max(1, round(image.size[1] * scale)),
                ),
                Image.Resampling.LANCZOS,
            )
        return np.asarray(image, dtype=np.float32) / 255.0, original_size


def _load_mask(path: Path, size: tuple[int, int]) -> np.ndarray:
    with Image.open(path) as source:
        if source.mode in {"RGBA", "LA"} or "transparency" in source.info:
            mask = source.convert("RGBA").getchannel("A")
        else:
            mask = source.convert("L")
        if mask.size != size:
            mask = mask.resize(size, Image.Resampling.NEAREST)
        return np.asarray(mask, dtype=np.uint8) > 127


def _discover_images(source: Path) -> list[tuple[Path, Path]]:
    source = source.expanduser().resolve()
    if source.is_file():
        if source.suffix.lower() not in IMAGE_SUFFIXES:
            raise ValueError(f"unsupported image extension: {source}")
        return [(source, Path(source.name))]
    if not source.is_dir():
        raise FileNotFoundError(f"image source does not exist: {source}")
    images = sorted(
        path
        for path in source.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not images:
        raise ValueError(f"no supported images found below {source}")
    return [(path, path.relative_to(source)) for path in images]


def _resolve_mask(mask_root: Path | None, relative: Path) -> Path | None:
    if mask_root is None:
        return None
    root = mask_root.expanduser().resolve()
    exact = root / relative
    if exact.is_file():
        return exact
    parent = root / relative.parent
    candidates = [
        parent / f"{relative.stem}{suffix}" for suffix in sorted(IMAGE_SUFFIXES)
    ]
    matches = [path for path in candidates if path.is_file()]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(
            f"no mask for {relative}; expected {exact} or the same relative stem"
        )
    raise RuntimeError(
        f"ambiguous masks for {relative}: {', '.join(str(path) for path in matches)}"
    )


def _prepare_source(
    image_path: Path,
    relative: Path,
    *,
    mask_root: Path | None,
    direct_mask: Path | None,
    mask_invert: bool,
    max_side: int | None,
) -> dict[str, Any]:
    image, original_size = _load_rgb(image_path, max_side)
    mask_path = (
        None
        if direct_mask is None
        else direct_mask.expanduser().resolve()
    )
    if mask_path is not None and not mask_path.is_file():
        raise FileNotFoundError(f"mask image does not exist: {mask_path}")
    if mask_path is None:
        mask_path = _resolve_mask(mask_root, relative)
    mask = None
    if mask_path is not None:
        mask = _load_mask(mask_path, (image.shape[1], image.shape[0]))
        if mask_invert:
            mask = ~mask
        if not mask.any():
            raise ValueError(f"mask contains no foreground: {mask_path}")
    target = image if mask is None else image * mask[..., None].astype(np.float32)
    return {
        "image_path": image_path,
        "relative": relative,
        "mask_path": mask_path,
        "image": image,
        "mask": mask,
        "target": target,
        "original_size": original_size,
    }


def _metric_bundle(prediction, target, mask, *, lpips: bool) -> dict[str, Any]:
    import torch

    from .metrics import LPIPS, ms_ssim, psnr_from_mse, ssim

    pred = prediction.clamp(0.0, 1.0)
    truth = target.clamp(0.0, 1.0)
    pixel_mse = (pred - truth).square().mean(dim=2)
    if mask is None:
        selected = pixel_mse.reshape(-1)
    else:
        selected = pixel_mse[
            torch.as_tensor(mask, device=pred.device, dtype=torch.bool)
        ]
    mse = selected.mean()
    return {
        "psnr": float(psnr_from_mse(mse)),
        "ssim": float(ssim(pred, truth)),
        "ms_ssim": float(ms_ssim(pred, truth)),
        "lpips": LPIPS.distance(pred, truth) if lpips else None,
        "mse": float(mse),
        "mae": float((pred - truth).abs().mean()),
        "max_abs": float((pred - truth).abs().max()),
    }


def _curve_auc(points: Iterable[dict[str, Any]]) -> float | None:
    by_step: dict[int, float] = {}
    for point in points:
        value = point.get("psnr")
        if value is not None:
            by_step[int(point.get("attempted_steps", 0))] = float(value)
    ordered = sorted(by_step.items())
    if not ordered:
        return None
    if len(ordered) == 1:
        return ordered[0][1]
    area = 0.0
    for (x0, y0), (x1, y1) in zip(ordered, ordered[1:]):
        area += (x1 - x0) * 0.5 * (y0 + y1)
    return area / max(1, ordered[-1][0] - ordered[0][0])


def _target_hits(points: Iterable[dict[str, Any]]) -> dict[str, int | None]:
    ordered = sorted(points, key=lambda point: int(point.get("attempted_steps", 0)))
    result: dict[str, int | None] = {}
    for target in HEADLINE_TARGETS:
        hit = next(
            (
                int(point.get("attempted_steps", 0))
                for point in ordered
                if point.get("psnr") is not None
                and float(point["psnr"]) >= target
            ),
            None,
        )
        result[str(target)] = hit
    return result


def _phase_timings(history: list[dict[str, Any]]) -> dict[str, float]:
    timings: dict[str, float] = {}
    previous = 0.0
    for record in history:
        current = float(record.get("elapsed_seconds", previous))
        phase = str(record.get("phase", "unknown"))
        timings[phase] = timings.get(phase, 0.0) + max(0.0, current - previous)
        previous = current
    return timings


def _gate_telemetry(history: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-phase commit-gate accounting for FIT-028/FIT-029/BENCH-018.

    The transactional gate trials a block of steps and rolls it back when the metric vector
    regresses, so ``attempted`` counts work spent and ``accepted`` counts work kept. Rejection
    reasons are the schedule's own strings (``interior_holes_regressed`` and friends); a rejected
    block may cite several at once, so the histogram counts reason occurrences, not blocks.
    """

    phases: dict[str, dict[str, Any]] = {}
    reasons_total: dict[str, int] = {}
    for record in history:
        # Only gated trial blocks carry a decision; phase_end/initialization markers do not.
        if not record.get("reasons") and record.get("event") in {
            "phase_end",
            "initialization",
        }:
            continue
        phase = str(record.get("phase", "unknown"))
        bucket = phases.setdefault(
            phase,
            {
                "attempted_steps": 0,
                "accepted_steps": 0,
                "blocks": 0,
                "accepted_blocks": 0,
                "rejection_reasons": {},
            },
        )
        attempted = int(record.get("attempted_steps", 0) or 0)
        accepted = int(record.get("accepted_steps", 0) or 0)
        bucket["attempted_steps"] += attempted
        bucket["accepted_steps"] += accepted
        bucket["blocks"] += 1
        if record.get("accepted"):
            bucket["accepted_blocks"] += 1
        for reason in record.get("reasons") or ():
            name = str(reason)
            bucket["rejection_reasons"][name] = (
                bucket["rejection_reasons"].get(name, 0) + 1
            )
            reasons_total[name] = reasons_total.get(name, 0) + 1
    for bucket in phases.values():
        attempted = bucket["attempted_steps"]
        bucket["step_acceptance"] = (
            None if attempted <= 0 else bucket["accepted_steps"] / attempted
        )
    attempted_total = sum(bucket["attempted_steps"] for bucket in phases.values())
    accepted_total = sum(bucket["accepted_steps"] for bucket in phases.values())
    return {
        "phases": phases,
        "attempted_steps": attempted_total,
        "accepted_steps": accepted_total,
        "step_acceptance": (
            None if attempted_total <= 0 else accepted_total / attempted_total
        ),
        "rejection_reasons": reasons_total,
    }


def _capture_name(ordinal: int, record: dict[str, Any]) -> str:
    return (
        f"{ordinal:03d}_"
        f"{_slug(str(record.get('phase', 'phase')))}_"
        f"{_slug(str(record.get('event', 'event')))}"
    )


def _should_capture(record: dict[str, Any]) -> bool:
    if not record.get("accepted"):
        return False
    event = str(record.get("event", ""))
    phase = str(record.get("phase", ""))
    return (
        phase == "initialization"
        or event == "phase_end"
        or event not in {"global_fit", "local_residual_fit"}
    )


def _job_key(
    prepared: dict[str, Any],
    *,
    method: str,
    variant: str,
    strategy: str,
    seed: int,
    device: str,
    mask_margin: float,
    fine_detail: bool,
    fine_detail_pursuit: bool,
    lpips: bool,
) -> str:
    payload = {
        "profile": CURRENT_PROFILE_NAME,
        "source_sha256": _sha256_file(prepared["image_path"]),
        "mask_sha256": (
            None
            if prepared["mask_path"] is None
            else _sha256_file(prepared["mask_path"])
        ),
        "relative": prepared["relative"].as_posix(),
        "method": method,
        "variant": variant,
        "strategy": strategy,
        "seed": int(seed),
        "device": str(device),
        "mask_margin": float(mask_margin),
        "fine_detail": bool(fine_detail),
        "fine_detail_pursuit": bool(fine_detail_pursuit),
        "lpips": bool(lpips),
        "target_pixel_sha256": _pixel_sha256(prepared["target"]),
        "implementation_sha256": {
            name: _sha256_file(REPOSITORY_ROOT / path)
            for name, path in {
                "pipeline": "src/structsplat/pipeline.py",
                "safe_schedule": "src/structsplat/safe_schedule.py",
                "detail_pursuit": "src/structsplat/detail_pursuit.py",
                "workflows": "src/structsplat/workflows.py",
            }.items()
        },
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _run_job(
    prepared: dict[str, Any],
    *,
    job_out: Path,
    comparison_source: Path,
    method: str,
    method_label: str,
    variant: str,
    strategy: str,
    schedule_transform: ScheduleTransform | None,
    seed: int,
    device: str,
    mask_margin: float,
    fine_detail: bool,
    fine_detail_pursuit: bool,
    lpips: bool,
    resume: bool,
    overwrite: bool,
    verbose: bool,
    requested_max_side: int | None,
) -> dict[str, Any]:
    key = _job_key(
        prepared,
        method=method,
        variant=variant,
        strategy=strategy,
        seed=seed,
        device=device,
        mask_margin=mask_margin,
        fine_detail=fine_detail,
        fine_detail_pursuit=fine_detail_pursuit,
        lpips=lpips,
    )
    result_path = job_out / "result.json"
    if result_path.is_file() and resume:
        existing = json.loads(result_path.read_text(encoding="utf-8"))
        if existing.get("job_key") != key:
            raise RuntimeError(
                f"resume contract mismatch for existing result: {result_path}"
            )
        return existing
    if job_out.exists() and any(job_out.iterdir()):
        if overwrite:
            shutil.rmtree(job_out)
        else:
            raise RuntimeError(
                f"output is non-empty: {job_out}; use --resume or --overwrite"
            )
    job_out.mkdir(parents=True, exist_ok=True)
    target_path = job_out / "target.png"
    _save_rgb(target_path, prepared["target"])
    curves: list[dict[str, Any]] = []
    snapshots: list[dict[str, Any]] = []
    capture_ordinal = 0

    def observer(field, record, target, base_cfg) -> None:
        nonlocal capture_ordinal
        selected = record.get("selected") or {}
        point = {
            "phase": record.get("phase"),
            "event": record.get("event"),
            "accepted": bool(record.get("accepted")),
            "attempted_steps": int(record.get("global_attempted_steps", 0)),
            "accepted_steps": int(record.get("global_accepted_steps", 0)),
            "elapsed_seconds": float(record.get("elapsed_seconds", 0.0)),
            "n_gaussians": selected.get("n_gaussians"),
            "psnr": selected.get("foreground_psnr_db"),
            "boundary_psnr": (
                selected.get("boundary_psnr_db")
                if prepared["mask"] is not None
                else None
            ),
            "cvar99_mse": selected.get("cvar99_mse"),
            "p99_mse": selected.get("p99_mse"),
            "interior_hole_fraction": selected.get("interior_hole_fraction"),
            "boundary_hole_fraction": (
                selected.get("boundary_hole_fraction")
                if prepared["mask"] is not None
                else None
            ),
            "ms_ssim": None,
            "ssim": None,
            "lpips": None,
            "mse": None,
            "mae": None,
        }
        curves.append(point)
        if not _should_capture(record):
            return
        import torch

        with torch.no_grad():
            rendered = render_field(field, target, base_cfg).clamp(0.0, 1.0)
            metrics = _metric_bundle(
                rendered, target, prepared["mask"], lpips=lpips
            )
        point.update(
            {
                "psnr": metrics["psnr"],
                "ssim": metrics["ssim"],
                "ms_ssim": metrics["ms_ssim"],
                "lpips": metrics["lpips"],
                "mse": metrics["mse"],
                "mae": metrics["mae"],
            }
        )
        stem = _capture_name(capture_ordinal, record)
        reconstruction_path = job_out / "intermediate" / f"{stem}.png"
        error_path = job_out / "intermediate" / f"{stem}_error_x4.png"
        render_np = rendered.detach().cpu().numpy()
        _save_rgb(reconstruction_path, render_np)
        _save_error(error_path, render_np, prepared["target"])
        snapshots.append(
            {
                **point,
                "reconstruction": str(reconstruction_path),
                "error_x4": str(error_path),
            }
        )
        capture_ordinal += 1

    started = time.perf_counter()
    output = run_current_pipeline(
        prepared["image"],
        mask=prepared["mask"],
        device=device,
        seed=seed,
        strategy=strategy,
        mask_margin=mask_margin,
        fine_detail=fine_detail,
        fine_detail_pursuit=fine_detail_pursuit,
        schedule_transform=schedule_transform,
        observer=observer,
        verbose=verbose,
    )
    import torch

    field = output["field"]
    final_render = output["render"].clamp(0.0, 1.0)
    final_metrics = _metric_bundle(
        final_render, output["target"], prepared["mask"], lpips=lpips
    )
    field_path = job_out / "field.npz"
    reconstruction_path = job_out / "reconstruction.png"
    error_path = job_out / "absolute_error_x4.png"
    field.save(str(field_path))
    final_np = final_render.detach().cpu().numpy()
    _save_rgb(reconstruction_path, final_np)
    _save_error(error_path, final_np, prepared["target"])
    schedule_result = output["schedule_result"]
    history = list(schedule_result["history"])
    history_path = job_out / "history.json"
    _atomic_json(
        history_path,
        {
            "schedule_history": history,
            "curves": curves,
            "snapshots": snapshots,
        },
    )
    config = {
        "schema": "structsplat.current_pipeline.run.v1",
        "job_key": key,
        "profile": output["profile"],
        "source": {
            "path": str(prepared["image_path"]),
            "relative": prepared["relative"].as_posix(),
            "sha256": _sha256_file(prepared["image_path"]),
            "mask_path": (
                None
                if prepared["mask_path"] is None
                else str(prepared["mask_path"])
            ),
            "mask_sha256": (
                None
                if prepared["mask_path"] is None
                else _sha256_file(prepared["mask_path"])
            ),
            "target_pixel_sha256": _pixel_sha256(prepared["target"]),
            "original_size": list(prepared["original_size"]),
            "fit_size": [
                int(prepared["image"].shape[1]),
                int(prepared["image"].shape[0]),
            ],
        },
        "method": method,
        "method_label": method_label,
        "variant": variant,
        "seed": int(seed),
        "initialization": output["initialization"],
        "fit_config": output["fit_config"],
        "schedule": output["schedule"],
        "repository": _repository_state(),
        "environment": {
            "device": device,
            "gpu": torch.cuda.get_device_name(torch.device(device)),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "numpy": np.__version__,
        },
    }
    _atomic_json(job_out / "config.json", config)
    requested_iters = sum(
        int(phase["max_steps"])
        for name, phase in output["schedule"].items()
        if isinstance(phase, dict)
        and "max_steps" in phase
        and (
            name != "error_tail"
            or float(output["schedule"].get("error_tail_fraction", 0.0)) > 0.0
        )
    )
    row = {
        "schema": "structsplat.current_pipeline.metric.v1",
        "job_key": key,
        "status": "ok",
        "error": "",
        "profile": CURRENT_PROFILE_NAME,
        "profile_scope": CURRENT_PROFILE_EVIDENCE_SCOPE,
        "method": method,
        "method_label": method_label,
        "variant": variant,
        "source_id": prepared["relative"].as_posix(),
        "source_path": str(comparison_source.resolve()),
        "original_source_path": str(prepared["image_path"]),
        "mask_path": (
            None
            if prepared["mask_path"] is None
            else str(prepared["mask_path"])
        ),
        "masked": prepared["mask"] is not None,
        "target_pixel_sha256": _pixel_sha256(prepared["target"]),
        "max_side": 0,
        "requested_max_side": requested_max_side,
        "seed": int(seed),
        "start_budget": int(output["initialization"]["config"]["num_gaussians"]),
        "start_gaussians": int(output["initialization"]["config"]["num_gaussians"]),
        "final_budget": int(field.n),
        "n_gaussians": int(field.n),
        "physical_capacity": int(
            schedule_result["storage"]["final_physical_rows"]
        ),
        "active_limit": int(
            schedule_result["storage"]["base_active_limit"]
        ),
        "iters": requested_iters,
        "attempted_steps": int(schedule_result["attempted_steps"]),
        "accepted_steps": int(schedule_result["accepted_steps"]),
        "psnr": final_metrics["psnr"],
        "ssim": final_metrics["ssim"],
        "ms_ssim": final_metrics["ms_ssim"],
        "lpips": final_metrics["lpips"],
        "mse": final_metrics["mse"],
        "mae": final_metrics["mae"],
        "max_abs": final_metrics["max_abs"],
        "auc_psnr": _curve_auc(curves),
        "iters_to_targets": _target_hits(curves),
        "init_seconds": output["timing"]["initialization_seconds"],
        "fit_seconds": output["timing"]["schedule_seconds"],
        "render_seconds": output["timing"]["final_render_seconds"],
        "total_seconds": time.perf_counter() - started,
        "phase_seconds": _phase_timings(history),
        "gate_telemetry": _gate_telemetry(history),
        "error_tail": schedule_result.get("error_tail"),
        "pursuit_tail": schedule_result.get("pursuit_tail"),
        "render_fps_median": (
            1.0 / output["timing"]["final_render_seconds"]
            if output["timing"]["final_render_seconds"] > 0.0
            else None
        ),
        "target_png": str(target_path),
        "reconstruction_png": str(reconstruction_path),
        "error_png": str(error_path),
        "field_npz": str(field_path),
        "field_sha256": _sha256_file(field_path),
        "history_json": str(history_path),
        "config_json": str(job_out / "config.json"),
        "curves": curves,
        "snapshots": snapshots,
    }
    _atomic_json(result_path, row)
    return row


def _error_row(
    prepared: dict[str, Any],
    *,
    method: str,
    method_label: str,
    variant: str,
    seed: int,
    error: Exception,
) -> dict[str, Any]:
    return {
        "schema": "structsplat.current_pipeline.metric.v1",
        "status": "error",
        "error": f"{type(error).__name__}: {error}",
        "profile": CURRENT_PROFILE_NAME,
        "method": method,
        "method_label": method_label,
        "variant": variant,
        "source_id": prepared["relative"].as_posix(),
        "source_path": str(prepared["image_path"]),
        "mask_path": (
            None
            if prepared["mask_path"] is None
            else str(prepared["mask_path"])
        ),
        "seed": int(seed),
    }


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, default=str)
    return value


def _portable_metric_value(outdir: Path, value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _portable_metric_value(outdir, item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_portable_metric_value(outdir, item) for item in value]
    if isinstance(value, (str, Path)):
        raw = str(value)
        path = Path(raw)
        if path.is_absolute():
            try:
                return path.resolve().relative_to(outdir.resolve()).as_posix()
            except ValueError:
                pass
        return raw
    return value


def _write_metrics(outdir: Path, rows: list[dict[str, Any]]) -> None:
    portable_rows = [_portable_metric_value(outdir, row) for row in rows]
    _atomic_json(outdir / "metrics.json", portable_rows)
    _atomic_text(
        outdir / "metrics.jsonl",
        "".join(json.dumps(row, default=str) + "\n" for row in portable_rows),
    )
    if not portable_rows:
        return
    fields = sorted(
        {
            key
            for row in portable_rows
            for key in row
            if key not in {"curves", "snapshots"}
        }
    )
    temporary = outdir / "metrics.csv.tmp"
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in portable_rows:
            writer.writerow(
                {key: _csv_value(row.get(key)) for key in fields}
            )
    temporary.replace(outdir / "metrics.csv")


def _relative_link(outdir: Path, value: str | Path) -> str:
    path = Path(value)
    try:
        return Path(os.path.relpath(path, outdir)).as_posix()
    except ValueError:
        return path.as_posix()


def _svg_curve(
    points: list[dict[str, Any]],
    key: str,
    title: str,
    color: str,
) -> str:
    values = [
        (float(point["attempted_steps"]), float(point[key]))
        for point in points
        if point.get(key) is not None
        and math.isfinite(float(point[key]))
    ]
    if not values:
        return (
            f"<div class='chart empty'><strong>{html.escape(title)}</strong>"
            "<span>not available</span></div>"
        )
    width, height, pad = 390, 150, 28
    xs = [value[0] for value in values]
    ys = [value[1] for value in values]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    if xmax <= xmin:
        xmax = xmin + 1.0
    if ymax <= ymin:
        ymax = ymin + 1.0
    coords = []
    for x, y in values:
        px = pad + (x - xmin) / (xmax - xmin) * (width - 2 * pad)
        py = height - pad - (y - ymin) / (ymax - ymin) * (height - 2 * pad)
        coords.append(f"{px:.1f},{py:.1f}")
    return (
        "<div class='chart'>"
        f"<strong>{html.escape(title)}</strong>"
        f"<svg viewBox='0 0 {width} {height}' role='img' "
        f"aria-label='{html.escape(title)}'>"
        f"<path d='M{pad},{height-pad}H{width-pad}M{pad},{pad}V{height-pad}'/>"
        f"<polyline points='{' '.join(coords)}' "
        f"style='stroke:{color}'/>"
        f"<text x='{pad}' y='{height-7}'>{xmin:.0f}</text>"
        f"<text x='{width-pad}' y='{height-7}' text-anchor='end'>{xmax:.0f} steps</text>"
        f"<text x='{pad+3}' y='{pad+11}'>{ymax:.4g}</text>"
        f"<text x='{pad+3}' y='{height-pad-4}'>{ymin:.4g}</text>"
        "</svg></div>"
    )


def _run_card(outdir: Path, row: dict[str, Any]) -> str:
    curves = list(row.get("curves") or [])
    snapshots = list(row.get("snapshots") or [])
    artifacts = []
    for label, key in (
        ("field.npz", "field_npz"),
        ("history.json", "history_json"),
        ("config.json", "config_json"),
    ):
        if row.get(key):
            link = _relative_link(outdir, row[key])
            artifacts.append(
                f"<a href='{html.escape(link)}'>{html.escape(label)}</a>"
            )
    images = []
    for label, key in (
        ("target", "target_png"),
        ("reconstruction", "reconstruction_png"),
        ("absolute error x4", "error_png"),
    ):
        if row.get(key):
            link = _relative_link(outdir, row[key])
            images.append(
                f"<figure><a href='{html.escape(link)}'><img src='{html.escape(link)}' "
                f"loading='lazy' alt='{html.escape(label)}'></a>"
                f"<figcaption>{html.escape(label)}</figcaption></figure>"
            )
    intermediate = []
    for snapshot in snapshots:
        link = _relative_link(outdir, snapshot["reconstruction"])
        error_link = _relative_link(outdir, snapshot["error_x4"])
        label = (
            f"{snapshot.get('phase')} / {snapshot.get('event')} / "
            f"{snapshot.get('attempted_steps')} steps"
        )
        intermediate.append(
            "<figure class='small'>"
            f"<a href='{html.escape(link)}'><img src='{html.escape(link)}' "
            f"loading='lazy' alt='{html.escape(label)}'></a>"
            f"<a href='{html.escape(error_link)}'>error</a>"
            f"<figcaption>{html.escape(label)}</figcaption></figure>"
        )
    lpips_text = (
        "n/a" if row.get("lpips") is None else f"{float(row['lpips']):.5f}"
    )
    phase_timings = "".join(
        f"<span><b>{float(seconds):.3f}s</b> {html.escape(str(phase))}</span>"
        for phase, seconds in (row.get("phase_seconds") or {}).items()
    )
    gate = row.get("gate_telemetry") or {}
    gate_html = ""
    if gate.get("phases"):
        overall = gate.get("step_acceptance")
        overall_text = "n/a" if overall is None else f"{float(overall):.1%}"
        phase_rows = "".join(
            "<tr>"
            f"<td>{html.escape(str(phase))}</td>"
            f"<td class='n'>{int(stats['attempted_steps']):,}</td>"
            f"<td class='n'>{int(stats['accepted_steps']):,}</td>"
            f"<td class='n'>"
            + (
                "n/a"
                if stats.get("step_acceptance") is None
                else f"{float(stats['step_acceptance']):.1%}"
            )
            + "</td>"
            f"<td class='n'>{int(stats['accepted_blocks']):,}/{int(stats['blocks']):,}</td>"
            "<td>"
            + html.escape(
                ", ".join(
                    f"{name} x{count}"
                    for name, count in sorted(
                        (stats.get("rejection_reasons") or {}).items(),
                        key=lambda item: -item[1],
                    )
                )
                or "—"
            )
            + "</td>"
            "</tr>"
            for phase, stats in gate["phases"].items()
        )
        gate_html = (
            "<details open><summary>commit-gate accounting "
            f"({overall_text} of attempted steps kept)</summary>"
            "<table class='gate'><thead><tr><th>phase</th><th>attempted</th>"
            "<th>accepted</th><th>step acceptance</th><th>blocks</th>"
            "<th>rejection reasons</th></tr></thead>"
            f"<tbody>{phase_rows}</tbody></table></details>"
        )
    error_tail = row.get("error_tail") or {}
    error_tail_html = ""
    if error_tail.get("enabled"):
        before = error_tail.get("before") or {}
        after = error_tail.get("after") or {}
        before_psnr = float(before.get("foreground_psnr_db", float("nan")))
        after_psnr = float(after.get("foreground_psnr_db", float("nan")))
        error_tail_html = (
            "<details open><summary>error-only fine-detail stage</summary>"
            "<div class='metrics'>"
            f"<span><b>{int(error_tail['estimated_complete_rows']):,}</b> "
            "estimated complete rows</span>"
            f"<span><b>{float(error_tail['fraction']):.0%}</b> allocation</span>"
            f"<span><b>{int(error_tail['requested_rows']):,}</b> requested</span>"
            f"<span><b>{int(error_tail['activated_rows']):,}</b> activated</span>"
            f"<span><b>{before_psnr:.4f} → {after_psnr:.4f}</b> foreground PSNR</span>"
            f"<span><b>{float(error_tail['foreground_psnr_gain_db']):+.4f} dB</b> "
            "stage gain</span>"
            "</div>"
            f"<p><code>{html.escape(str(error_tail.get('formula')))}</code>. "
            f"Allocation: {html.escape(str(error_tail.get('allocation_termination_reason')))}; "
            "convergence: "
            f"{html.escape(str(error_tail.get('convergence_termination_reason')))}.</p>"
            "</details>"
        )
    pursuit_tail = row.get("pursuit_tail") or {}
    pursuit_tail_html = ""
    if pursuit_tail.get("enabled"):
        pursuit_tail_html = (
            "<details open><summary>orthogonal fine-detail pursuit</summary>"
            "<div class='metrics'>"
            f"<span><b>{int(pursuit_tail['activated_rows']):,}</b> "
            "added Gaussians</span>"
            f"<span><b>{float(pursuit_tail['highpass_reduction']):.2%}</b> "
            "high-pass reduction</span>"
            f"<span><b>{float(pursuit_tail['laplacian_reduction']):.2%}</b> "
            "Laplacian reduction</span>"
            f"<span><b>{int(pursuit_tail['waves_accepted'])}</b> accepted waves</span>"
            "</div>"
            f"<p>Termination: {html.escape(str(pursuit_tail['termination_reason']))}; "
            f"target reached: {bool(pursuit_tail['target_reached'])}.</p>"
            "</details>"
        )
    return (
        "<article class='run'>"
        f"<h2>{html.escape(str(row.get('method_label', row.get('method'))))}</h2>"
        f"<p class='identity'>{html.escape(str(row.get('source_id')))} · "
        f"seed {row.get('seed')} · {row.get('n_gaussians')} Gaussians</p>"
        "<div class='metrics'>"
        f"<span><b>{float(row['psnr']):.4f}</b> PSNR</span>"
        f"<span><b>{float(row['ms_ssim']):.5f}</b> MS-SSIM</span>"
        f"<span><b>{lpips_text}</b> LPIPS</span>"
        f"<span><b>{float(row['fit_seconds']):.3f}s</b> fit</span>"
        f"<span><b>{float(row['total_seconds']):.3f}s</b> total</span>"
        "</div>"
        f"<details><summary>phase timings</summary><div class='metrics'>{phase_timings}</div>"
        "</details>"
        f"{gate_html}"
        f"{error_tail_html}"
        f"{pursuit_tail_html}"
        f"<div class='links'>{' '.join(artifacts)}</div>"
        f"<div class='hero-images'>{''.join(images)}</div>"
        "<div class='charts'>"
        f"{_svg_curve(curves, 'psnr', 'PSNR over attempted steps', '#e65f2b')}"
        f"{_svg_curve(curves, 'ssim', 'SSIM over attempted steps', '#247ba0')}"
        f"{_svg_curve(curves, 'ms_ssim', 'MS-SSIM over attempted steps', '#147d72')}"
        f"{_svg_curve(curves, 'lpips', 'LPIPS over attempted steps', '#b58416')}"
        f"{_svg_curve(curves, 'mse', 'MSE over attempted steps', '#8f2d56')}"
        f"{_svg_curve(curves, 'mae', 'MAE over attempted steps', '#5c4d7d')}"
        f"{_svg_curve(curves, 'cvar99_mse', 'CVaR99 MSE over attempted steps', '#b33c1f')}"
        f"{_svg_curve(curves, 'p99_mse', 'p99 MSE over attempted steps', '#7a5195')}"
        f"{_svg_curve(curves, 'interior_hole_fraction', 'Interior holes over attempted steps', '#2f6b3c')}"
        f"{_svg_curve(curves, 'boundary_hole_fraction', 'Boundary holes over attempted steps', '#99582a')}"
        f"{_svg_curve(curves, 'elapsed_seconds', 'Cumulative time over steps', '#345995')}"
        "</div>"
        "<details><summary>target and intermediate states</summary>"
        f"<div class='intermediate'>{''.join(intermediate)}</div></details>"
        "</article>"
    )


def _native_card(outdir: Path, row: dict[str, Any]) -> str:
    reconstruction = row.get("reconstruction_png")
    error = row.get("workflow_error_png")
    images = ""
    if reconstruction:
        link = _relative_link(outdir, reconstruction)
        images += (
            f"<figure><a href='{html.escape(link)}'><img src='{html.escape(link)}' "
            "loading='lazy' alt='native reconstruction'></a>"
            "<figcaption>native reconstruction</figcaption></figure>"
        )
    if error:
        link = _relative_link(outdir, error)
        images += (
            f"<figure><a href='{html.escape(link)}'><img src='{html.escape(link)}' "
            "loading='lazy' alt='native absolute error x4'></a>"
            "<figcaption>absolute error x4</figcaption></figure>"
        )
    return (
        "<article class='run native'>"
        f"<h2>{html.escape(str(row.get('method_label', row.get('method'))))}</h2>"
        f"<p class='identity'>{html.escape(str(row.get('source_id', row.get('image'))))} · "
        f"seed {row.get('seed')} · {row.get('budget_cap')} Gaussians</p>"
        "<div class='metrics'>"
        f"<span><b>{float(row['psnr']):.4f}</b> PSNR</span>"
        f"<span><b>{float(row['ms_ssim']):.5f}</b> MS-SSIM</span>"
        f"<span><b>{float(row['fit_seconds']):.3f}s</b> fit</span>"
        "</div>"
        f"<div class='hero-images'>{images}</div>"
        "</article>"
    )


def _write_index(
    outdir: Path,
    *,
    title: str,
    rows: list[dict[str, Any]],
    native_rows: list[dict[str, Any]],
    command: str,
) -> None:
    ok = [row for row in rows if row.get("status") == "ok"]
    errors = [row for row in rows if row.get("status") != "ok"]
    summary_rows = []
    for row in ok:
        lpips = "-" if row.get("lpips") is None else f"{float(row['lpips']):.5f}"
        summary_rows.append(
            "<tr>"
            f"<td>{html.escape(str(row['method_label']))}</td>"
            f"<td>{html.escape(str(row['source_id']))}</td>"
            f"<td>{row['seed']}</td><td>{row['n_gaussians']}</td>"
            f"<td>{float(row['psnr']):.4f}</td>"
            f"<td>{float(row['ms_ssim']):.5f}</td><td>{lpips}</td>"
            f"<td>{float(row['fit_seconds']):.3f}</td>"
            f"<td>{float(row['total_seconds']):.3f}</td>"
            "</tr>"
        )
    error_rows = "".join(
        f"<li><code>{html.escape(str(row.get('source_id')))}</code>: "
        f"{html.escape(str(row.get('error')))}</li>"
        for row in errors
    )
    native_links = []
    for name in ("gaussianimage", "image_gs"):
        path = outdir / "baselines" / name / "index.html"
        if path.is_file():
            native_links.append(
                f"<a href='{_relative_link(outdir, path)}'>{html.escape(name)} report</a>"
            )
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title>
<style>
:root{{--paper:#f3efe5;--ink:#1d2528;--muted:#64706f;--line:#c8c0ae;--accent:#e65f2b;
--teal:#147d72;--panel:#fffdf7;--blue:#345995}}
*{{box-sizing:border-box}}body{{margin:0;background:
radial-gradient(circle at 12% 8%,#fff9e9 0,transparent 28rem),
linear-gradient(135deg,#eee7d8,#f7f4eb 55%,#e5eee9);color:var(--ink);
font-family:Georgia,"Times New Roman",serif;line-height:1.45}}
header,main{{width:min(1500px,calc(100% - 32px));margin:auto}}
header{{padding:54px 0 28px;border-bottom:3px double var(--ink)}}h1{{font-size:clamp(2rem,5vw,4.7rem);
line-height:.95;margin:0 0 18px;max-width:1050px;letter-spacing:-.035em}}
h2{{margin:.1em 0;font-size:1.7rem}}.scope{{max-width:1000px;color:var(--muted);font-size:1.05rem}}
code,.identity,.metrics,table,.links{{font-family:"Liberation Mono",monospace}}
.links{{display:flex;gap:16px;flex-wrap:wrap;margin:18px 0}}a{{color:#095c57}}
main{{padding:28px 0 70px}}table{{width:100%;border-collapse:collapse;background:var(--panel);
font-size:.82rem}}th,td{{border:1px solid var(--line);padding:8px;text-align:right}}
th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){{text-align:left}}
.run{{margin:34px 0;padding:22px;background:rgba(255,253,247,.9);border:1px solid var(--line);
box-shadow:8px 8px 0 rgba(29,37,40,.08)}}.native{{border-left:6px solid var(--blue)}}
.identity{{color:var(--muted);margin-top:4px}}.metrics{{display:flex;gap:10px;flex-wrap:wrap;margin:14px 0}}
.metrics span{{border:1px solid var(--line);padding:7px 10px;background:#fff}}
.hero-images,.intermediate{{display:flex;gap:12px;overflow:auto;padding:8px 0}}
figure{{margin:0;min-width:250px;flex:1}}figure.small{{min-width:180px;max-width:300px}}
img{{width:100%;height:auto;display:block;background:#111;border:1px solid #777}}
figcaption{{font-size:.82rem;color:var(--muted);margin-top:5px}}.charts{{display:grid;
grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:12px;margin-top:16px}}
.chart{{background:#fff;border:1px solid var(--line);padding:9px}}.chart strong{{display:block}}
.chart svg{{width:100%;height:auto}}.chart path{{fill:none;stroke:#c7c1b5;stroke-width:1}}
.chart polyline{{fill:none;stroke-width:3}}.chart text{{font:10px "Liberation Mono",monospace;fill:#6b6b66}}
.empty{{display:flex;min-height:110px;flex-direction:column;justify-content:center;color:var(--muted)}}
details{{margin-top:15px}}summary{{cursor:pointer;font-weight:bold}}.errors{{color:#8a251a}}
table.gate td.n,table.gate th:nth-child(2){{text-align:right}}
table.gate td:last-child,table.gate th:last-child{{text-align:left;font-size:.78rem}}
@media(max-width:700px){{header,main{{width:min(100% - 18px,1500px)}}header{{padding-top:32px}}
.run{{padding:12px;box-shadow:4px 4px 0 rgba(29,37,40,.08)}}table{{display:block;overflow:auto}}}}
</style></head><body>
<header><p class="identity">STRUCTSPLAT / {html.escape(CURRENT_PROFILE_NAME)}</p>
<h1>{html.escape(title)}</h1>
<p class="scope">{html.escape(CURRENT_PROFILE_EVIDENCE_SCOPE)}. Masked and unmasked runs share
the same count, stage, optimizer, and checkpoint profile; unmasked runs omit only boundary
initialization, containment, losses, metrics, and proposals.</p>
<p><code>{html.escape(command)}</code></p>
<div class="links"><a href="metrics.json">metrics.json</a><a href="metrics.csv">metrics.csv</a>
<a href="metrics.jsonl">metrics.jsonl</a><a href="manifest.json">manifest.json</a>
{' · '.join(native_links)}</div></header><main>
<h2>Run matrix</h2><table><thead><tr><th>method</th><th>image</th><th>seed</th><th>N</th>
<th>PSNR</th><th>MS-SSIM</th><th>LPIPS</th><th>fit s</th><th>total s</th></tr></thead>
<tbody>{''.join(summary_rows)}</tbody></table>
{f'<section class="errors"><h2>Errors</h2><ul>{error_rows}</ul></section>' if errors else ''}
<section>{''.join(_run_card(outdir, row) for row in ok)}</section>
{f'<section><h2>Official native baselines</h2>{"".join(_native_card(outdir, row) for row in native_rows if row.get("status") == "ok")}</section>' if native_rows else ''}
</main></body></html>"""
    _atomic_text(outdir / "index.html", document)


def _baseline_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--baselines",
        nargs="*",
        choices=("gaussianimage", "image_gs"),
        default=[],
        help="optionally execute pinned official native baselines",
    )
    parser.add_argument("--gaussianimage-repo", type=Path)
    parser.add_argument("--gaussianimage-python", type=Path)
    parser.add_argument("--image-gs-repo", type=Path)
    parser.add_argument("--image-gs-python", type=Path)
    parser.add_argument("--libstdcxx-preload", type=Path)


def _common_arguments(
    parser: argparse.ArgumentParser,
    *,
    multiple_seeds: bool,
    direct_mask: bool = False,
) -> None:
    parser.add_argument("source", type=Path, help="image file or recursively scanned folder")
    parser.add_argument("outdir", type=Path, help="destination/result folder")
    masks = parser.add_mutually_exclusive_group()
    if direct_mask:
        masks.add_argument(
            "--mask",
            type=Path,
            help="mask image for a single source image",
        )
    masks.add_argument(
        "--mask-dir",
        type=Path,
        help="optional parallel mask tree with matching relative stems",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-side", type=int)
    parser.add_argument("--max-images", type=int)
    parser.add_argument(
        "--mask-margin",
        type=float,
        default=PipelineConfig.mask_margin,
    )
    if direct_mask:
        parser.add_argument(
            "--mask-invert",
            action="store_true",
            help="treat the dark side of the mask as foreground",
        )
    if multiple_seeds:
        parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    else:
        parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lpips", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--resume", action="store_true")
    mode.add_argument("--overwrite", action="store_true")
    parser.add_argument("--quiet", action="store_true")


def _require_positive(args: argparse.Namespace) -> None:
    if args.max_side is not None and args.max_side <= 0:
        raise ValueError("--max-side must be positive")
    if args.max_images is not None and args.max_images <= 0:
        raise ValueError("--max-images must be positive")
    if (
        not math.isfinite(args.mask_margin)
        or args.mask_margin < MIN_MASK_MARGIN
    ):
        raise ValueError(
            f"--mask-margin must be finite and preserve the >={MIN_MASK_MARGIN} "
            "containment floor"
        )


def _materialize_input(outdir: Path, prepared: dict[str, Any]) -> Path:
    relative = prepared["relative"].with_suffix(".png")
    path = outdir / "inputs" / relative
    _save_rgb(path, prepared["target"])
    return path


def _phase_disabled(phase):
    return replace(phase, max_steps=0)


def _ablation_transform(arm: str) -> ScheduleTransform | None:
    if arm == "full":
        return None

    def transform(schedule):
        schedule = copy.deepcopy(schedule)
        if arm == "no_bootstrap":
            schedule.bootstrap = _phase_disabled(schedule.bootstrap)
        elif arm == "no_coverage_growth":
            schedule.coverage = _phase_disabled(schedule.coverage)
        elif arm == "no_detail_growth":
            schedule.detail = _phase_disabled(schedule.detail)
        elif arm == "no_closure":
            schedule.boundary = _phase_disabled(schedule.boundary)
        elif arm == "no_redistribution":
            schedule.redistribution = _phase_disabled(schedule.redistribution)
        elif arm == "no_polish":
            schedule.polish = _phase_disabled(schedule.polish)
        elif arm == "no_pareto_checkpoints":
            schedule.pareto_safe_checkpoints = False
        elif arm == "no_boundary_specialization":
            schedule.boundary_enabled = False
            schedule.boundary = replace(
                schedule.boundary, name="general_closure"
            )
        else:
            raise ValueError(f"unknown ablation arm: {arm}")
        return schedule

    return transform


def _stage_transform(stage: str, variant: str) -> tuple[str, ScheduleTransform | None]:
    strategy = "quadtree_wse"
    if stage == "initialization":
        return variant, None
    if variant in {"current", "dynamic", "pareto50"}:
        return strategy, None

    def transform(schedule):
        schedule = copy.deepcopy(schedule)
        if stage == "storage":
            schedule.storage_policy = variant
        elif stage == "checkpoint":
            schedule.pareto_safe_checkpoints = variant == "pareto50"
        elif stage == "bootstrap":
            if variant == "full_resolution":
                schedule.bootstrap = replace(
                    schedule.bootstrap, lowpass_downsample=1
                )
            elif variant == "disabled":
                schedule.bootstrap = _phase_disabled(schedule.bootstrap)
        elif stage == "coverage":
            if variant == "births256":
                schedule.coverage_birth_count = 256
            elif variant == "births1024":
                schedule.coverage_birth_count = 1_024
            elif variant == "disabled":
                schedule.coverage = _phase_disabled(schedule.coverage)
        elif stage == "detail":
            if variant == "birth_only":
                schedule.detail_split_count = 0
            elif variant == "disabled":
                schedule.detail = _phase_disabled(schedule.detail)
        elif stage == "closure":
            if variant == "generic_no_boundary":
                schedule.boundary_enabled = False
                schedule.boundary = replace(
                    schedule.boundary, name="general_closure"
                )
            elif variant == "disabled":
                schedule.boundary = _phase_disabled(schedule.boundary)
        elif stage == "redistribution":
            if variant == "events32":
                schedule.redistribution_count = 32
            elif variant == "events128":
                schedule.redistribution_count = 128
            elif variant == "disabled":
                schedule.redistribution = _phase_disabled(
                    schedule.redistribution
                )
        elif stage == "polish":
            if variant == "steps1000":
                schedule.polish = replace(schedule.polish, max_steps=1_000)
            elif variant == "steps4000":
                schedule.polish = replace(schedule.polish, max_steps=4_000)
            elif variant == "disabled":
                schedule.polish = _phase_disabled(schedule.polish)
        elif stage == "commit_gate":
            # BENCH-018 sets one granularity for every gated phase, exactly as
            # `PipelineConfig.block_steps` does, and clamps it to each phase ceiling.
            block = COMMIT_GATE_BLOCK_STEPS[variant]
            for name in _GATED_PHASES:
                phase = getattr(schedule, name)
                setattr(
                    schedule,
                    name,
                    replace(
                        phase,
                        block_steps=max(1, min(int(block), int(phase.max_steps))),
                    ),
                )
        elif stage == "hole_budget":
            schedule.hole_regression_budget = HOLE_REGRESSION_BUDGETS[variant]
        else:
            raise ValueError(f"unknown stage: {stage}")
        return schedule

    return strategy, transform


def _native_error_images(
    outdir: Path, native_rows: list[dict[str, Any]]
) -> None:
    for index, row in enumerate(native_rows):
        if row.get("status") != "ok" or not row.get("reconstruction_png"):
            continue
        source = Path(str(row["source_path"]))
        reconstruction = Path(str(row["reconstruction_png"]))
        if not source.is_file() or not reconstruction.is_file():
            continue
        target, _ = _load_rgb(source, None)
        prediction, _ = _load_rgb(reconstruction, None)
        if prediction.shape != target.shape:
            continue
        error_path = outdir / "baseline_errors" / (
            f"{_slug(str(row.get('method')))}_{index:04d}_error_x4.png"
        )
        _save_error(error_path, prediction, target)
        row["workflow_error_png"] = str(error_path)


def _load_json_rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"expected JSON row list: {path}")
    return payload


def _run_baselines(
    args: argparse.Namespace,
    *,
    outdir: Path,
    inputs: list[Path],
    seeds: list[int],
) -> list[dict[str, Any]]:
    requested = list(dict.fromkeys(args.baselines))
    if not requested:
        return []
    structsplat_rows = _load_json_rows(outdir / "metrics.json")
    full_rows = [
        row
        for row in structsplat_rows
        if row.get("status") == "ok"
        and row.get("method") == "structsplat_best_default"
    ]
    budgets = sorted({int(row["final_budget"]) for row in full_rows})
    horizons = sorted({int(row["iters"]) for row in full_rows})
    if not budgets or not horizons:
        raise RuntimeError(
            "native baselines require at least one successful current-profile row"
        )
    if len(horizons) != 1:
        raise RuntimeError(
            f"native baseline comparison requires one requested horizon, got {horizons}"
        )
    common = [
        "--images",
        *(str(path) for path in inputs),
        "--max-sides",
        "0",
        "--budgets",
        *(str(budget) for budget in budgets),
        "--iters",
        str(horizons[0]),
        "--seeds",
        *(str(seed) for seed in seeds),
        "--structsplat-metrics",
        str(outdir / "metrics.json"),
        "--structsplat-methods",
        "structsplat_best_default",
        "--device",
        args.device,
    ]
    if args.lpips:
        common.append("--lpips")
    if args.resume:
        common.append("--resume")
    if args.libstdcxx_preload is not None:
        common.extend(
            ["--libstdcxx-preload", str(args.libstdcxx_preload.resolve())]
        )
    environment = os.environ.copy()
    existing_pythonpath = environment.get("PYTHONPATH")
    prefix = f"{REPOSITORY_ROOT / 'src'}:{REPOSITORY_ROOT}"
    environment["PYTHONPATH"] = (
        prefix if not existing_pythonpath else f"{prefix}:{existing_pythonpath}"
    )
    collected: list[dict[str, Any]] = []
    if "gaussianimage" in requested:
        if args.gaussianimage_repo is None or args.gaussianimage_python is None:
            raise ValueError(
                "GaussianImage requires --gaussianimage-repo and "
                "--gaussianimage-python"
            )
        baseline_out = outdir / "baselines" / "gaussianimage"
        command = [
            sys.executable,
            "-m",
            "benchmarks.native_gaussianimage_compare",
            *common,
            "--outdir",
            str(baseline_out),
            "--gaussianimage-repo",
            str(args.gaussianimage_repo.resolve()),
            "--native-python",
            str(args.gaussianimage_python.resolve()),
            "--profile",
            "matched_steps_fixed_n",
        ]
        subprocess.run(
            command,
            cwd=REPOSITORY_ROOT,
            env=environment,
            check=True,
        )
        collected.extend(_load_json_rows(baseline_out / "metrics.json"))
    if "image_gs" in requested:
        if args.image_gs_repo is None or args.image_gs_python is None:
            raise ValueError(
                "Image-GS requires --image-gs-repo and --image-gs-python"
            )
        baseline_out = outdir / "baselines" / "image_gs"
        command = [
            sys.executable,
            "-m",
            "benchmarks.native_image_gs_compare",
            *common,
            "--outdir",
            str(baseline_out),
            "--image-gs-repo",
            str(args.image_gs_repo.resolve()),
            "--image-gs-python",
            str(args.image_gs_python.resolve()),
            "--native-device",
            args.device,
            "--profile",
            "matched_steps_fixed_n",
        ]
        subprocess.run(
            command,
            cwd=REPOSITORY_ROOT,
            env=environment,
            check=True,
        )
        collected.extend(_load_json_rows(baseline_out / "metrics.json"))
    _native_error_images(outdir, collected)
    return collected


def _execute(
    args: argparse.Namespace,
    *,
    title: str,
    variants: list[tuple[str, str, ScheduleTransform | None, str, str]],
    single_image: bool = False,
    with_baselines: bool = False,
) -> int:
    _require_positive(args)
    fine_detail = bool(getattr(args, "fine_detail", False))
    fine_detail_pursuit = bool(
        getattr(args, "fine_detail_pursuit", False)
    )
    outdir = args.outdir.expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    discovered = _discover_images(args.source)
    if args.max_images is not None:
        discovered = discovered[: args.max_images]
    direct_mask = getattr(args, "mask", None)
    if direct_mask is not None and len(discovered) != 1:
        raise ValueError(
            f"--mask requires exactly one source image, found {len(discovered)}"
        )
    if single_image and len(discovered) != 1:
        raise ValueError(
            f"stage search requires exactly one image, found {len(discovered)}"
        )
    seeds = list(args.seeds) if hasattr(args, "seeds") else [int(args.seed)]
    rows: list[dict[str, Any]] = []
    comparison_inputs: list[Path] = []
    prepared_items = []
    for image_path, relative in discovered:
        prepared = _prepare_source(
            image_path,
            relative,
            mask_root=args.mask_dir,
            direct_mask=direct_mask,
            mask_invert=bool(getattr(args, "mask_invert", False)),
            max_side=args.max_side,
        )
        prepared_items.append(prepared)
        comparison_inputs.append(_materialize_input(outdir, prepared))
    if fine_detail_pursuit and any(
        item["mask"] is None for item in prepared_items
    ):
        raise ValueError("--fine-detail-pursuit requires a mask for every image")
    for prepared, comparison_source in zip(prepared_items, comparison_inputs):
        for variant, strategy, transform, method, method_label in variants:
            if (
                variant == "no_boundary_specialization"
                and prepared["mask"] is None
            ):
                continue
            for seed in seeds:
                relative_stem = prepared["relative"].with_suffix("")
                job_out = (
                    outdir
                    / "runs"
                    / _slug(variant)
                    / relative_stem
                    / f"seed_{seed}"
                )
                try:
                    row = _run_job(
                        prepared,
                        job_out=job_out,
                        comparison_source=comparison_source,
                        method=method,
                        method_label=method_label,
                        variant=variant,
                        strategy=strategy,
                        schedule_transform=transform,
                        seed=seed,
                        device=args.device,
                        mask_margin=args.mask_margin,
                        fine_detail=fine_detail,
                        fine_detail_pursuit=fine_detail_pursuit,
                        lpips=args.lpips,
                        resume=args.resume,
                        overwrite=args.overwrite,
                        verbose=not args.quiet,
                        requested_max_side=args.max_side,
                    )
                except Exception as error:
                    row = _error_row(
                        prepared,
                        method=method,
                        method_label=method_label,
                        variant=variant,
                        seed=seed,
                        error=error,
                    )
                rows.append(row)
                _write_metrics(outdir, rows)
    manifest = {
        "schema": "structsplat.current_pipeline.workflow.v1",
        "title": title,
        "profile": profile_manifest(
            masked=any(item["mask"] is not None for item in prepared_items),
            mask_margin=float(args.mask_margin),
            fine_detail=fine_detail,
            fine_detail_pursuit=fine_detail_pursuit,
        ),
        "command": " ".join(sys.argv),
        "source": str(args.source.expanduser().resolve()),
        "mask_dir": (
            None
            if args.mask_dir is None
            else str(args.mask_dir.expanduser().resolve())
        ),
        "mask": (
            None
            if direct_mask is None
            else str(direct_mask.expanduser().resolve())
        ),
        "mask_invert": bool(getattr(args, "mask_invert", False)),
        "fine_detail": fine_detail,
        "fine_detail_pursuit": fine_detail_pursuit,
        "variants": [variant for variant, *_ in variants],
        "seeds": seeds,
        "images": [
            {
                "path": str(item["image_path"]),
                "relative": item["relative"].as_posix(),
                "mask": (
                    None
                    if item["mask_path"] is None
                    else str(item["mask_path"])
                ),
            }
            for item in prepared_items
        ],
        "repository": _repository_state(),
    }
    _atomic_json(outdir / "manifest.json", manifest)
    native_rows: list[dict[str, Any]] = []
    if with_baselines:
        native_rows = _run_baselines(
            args,
            outdir=outdir,
            inputs=comparison_inputs,
            seeds=seeds,
        )
        _atomic_json(outdir / "native_metrics.json", native_rows)
    _write_index(
        outdir,
        title=title,
        rows=rows,
        native_rows=native_rows,
        command=" ".join(sys.argv),
    )
    return 1 if any(row.get("status") != "ok" for row in rows) else 0


def build_convert_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Convert an image or image tree with the current masked/unmasked "
            "safe-schedule profile"
        )
    )
    _common_arguments(parser, multiple_seeds=False, direct_mask=True)
    fine_detail_group = parser.add_mutually_exclusive_group()
    fine_detail_group.add_argument(
        "--fine-detail",
        action="store_true",
        help=(
            "append the optional terminal error-only stage: estimate effective "
            "residual sites, request half as small Gaussians, and converge safely"
        ),
    )
    fine_detail_group.add_argument(
        "--fine-detail-pursuit",
        action="store_true",
        help=(
            "append sparse high-pass pursuit waves until explicit detail targets "
            "are reached under the protected-metric gate"
        ),
    )
    return parser


def build_benchmark_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark the current profile and write curves, timings, images, raw rows, "
            "and a portable index.html"
        )
    )
    _common_arguments(parser, multiple_seeds=True)
    _baseline_arguments(parser)
    return parser


def build_ablation_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the fixed current-profile stage ablation matrix"
    )
    _common_arguments(parser, multiple_seeds=True)
    parser.add_argument(
        "--arms",
        nargs="+",
        choices=ABLATION_ARMS,
        default=list(ABLATION_ARMS),
    )
    _baseline_arguments(parser)
    return parser


def build_stage_search_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Test every registered variant of one current-pipeline stage on one image"
        )
    )
    _common_arguments(parser, multiple_seeds=True)
    parser.add_argument("--stage", required=True, choices=tuple(STAGE_VARIANTS))
    parser.add_argument(
        "--variants",
        nargs="+",
        help="optional subset; defaults to every registered variant for --stage",
    )
    _baseline_arguments(parser)
    return parser


def main_convert(argv: list[str] | None = None) -> int:
    args = build_convert_parser().parse_args(argv)
    fine_detail = bool(args.fine_detail)
    fine_detail_pursuit = bool(args.fine_detail_pursuit)
    if fine_detail_pursuit:
        variant = "fine_detail_pursuit"
        method = "structsplat_best_default_detail_pursuit"
        method_label = "Current profile + orthogonal fine-detail pursuit"
        title = "Current Pipeline + Orthogonal Fine-Detail Pursuit"
    elif fine_detail:
        variant = "fine_detail"
        method = "structsplat_best_default_error_tail50"
        method_label = "Current profile + error-only fine detail"
        title = "Current Pipeline + Error-Only Fine Detail"
    else:
        variant = "current"
        method = "structsplat_best_default"
        method_label = "Current profile"
        title = "Current Pipeline Conversion"
    variants = [
        (
            variant,
            "quadtree_wse",
            None,
            method,
            method_label,
        )
    ]
    return _execute(
        args,
        title=title,
        variants=variants,
    )


def main_benchmark(argv: list[str] | None = None) -> int:
    args = build_benchmark_parser().parse_args(argv)
    variants = [
        (
            "current",
            "quadtree_wse",
            None,
            "structsplat_best_default",
            "Current profile",
        )
    ]
    return _execute(
        args,
        title="Current Pipeline Benchmark",
        variants=variants,
        with_baselines=True,
    )


def main_ablation(argv: list[str] | None = None) -> int:
    args = build_ablation_parser().parse_args(argv)
    variants = []
    for arm in args.arms:
        method = (
            "structsplat_best_default"
            if arm == "full"
            else f"ablation/{arm}"
        )
        variants.append(
            (
                arm,
                "quadtree_wse",
                _ablation_transform(arm),
                method,
                "Current profile" if arm == "full" else arm.replace("_", " "),
            )
        )
    return _execute(
        args,
        title="Current Pipeline Ablation",
        variants=variants,
        with_baselines=True,
    )


def main_stage_search(argv: list[str] | None = None) -> int:
    args = build_stage_search_parser().parse_args(argv)
    available = STAGE_VARIANTS[args.stage]
    selected = list(available) if args.variants is None else list(args.variants)
    unknown = [variant for variant in selected if variant not in available]
    if unknown:
        raise ValueError(
            f"unknown {args.stage} variants {unknown}; available: {list(available)}"
        )
    variants = []
    for variant in selected:
        strategy, transform = _stage_transform(args.stage, variant)
        method = (
            "structsplat_best_default"
            if variant == available[0]
            else f"stage/{args.stage}/{variant}"
        )
        variants.append(
            (
                variant,
                strategy,
                transform,
                method,
                f"{args.stage}: {variant}",
            )
        )
    return _execute(
        args,
        title=f"Stage Search: {args.stage}",
        variants=variants,
        single_image=True,
        with_baselines=True,
    )
