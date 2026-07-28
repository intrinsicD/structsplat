#!/usr/bin/env python3
"""Run the FIT-040 and FIT-031 tails on every other persisted Janelle view.

This is a correlated cross-view diagnostic, not FIT-042 independent-scene
confirmation.  Each arm starts from the same integrity-checked, resolution-
adapted ``.rtgsv`` field for one view.  The production orthogonal-pursuit tail
and the natural FIT-031 error-only tail then run with their shipped stopping
policies under one harmonized renderer.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import html
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import traceback
from typing import Any

import numpy as np
from PIL import __version__ as PILLOW_VERSION
from PIL import Image
import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
for _root in (
    REPOSITORY_ROOT,
    REPOSITORY_ROOT / "src",
    REPOSITORY_ROOT / "deprecated_scripts",
):
    _text = str(_root)
    if _text not in sys.path:
        sys.path.insert(0, _text)

from scripts.experiments.fit032_janelle_dipole_screen import (  # noqa: E402
    _base_config,
    _prepare_janelle,
)
from scripts.experiments.fit033_janelle_highpass_solve import (  # noqa: E402
    _evaluate_all,
    gaussian_blur,
)
from scripts.experiments.fit040_janelle_production_pursuit import (  # noqa: E402
    _disabled_phase,
)
from structsplat.fit import _MaskConstraint  # noqa: E402
from structsplat.gaussians import GaussianField  # noqa: E402
from structsplat.safe_schedule import (  # noqa: E402
    SafeScheduleConfig,
    run_safe_schedule,
    safe_commit_decision,
)


SCHEMA = "structsplat.janelle_cross_view_tail_diagnostic.v1"
DEFAULT_CAPTURE_ROOT = Path("/home/alex/Dropbox/Work/Janelle/2025_03_07_stage_with_fabric")
DEFAULT_REALTIME_ROOT = Path("/home/alex/Documents/realtime-gs")
DEFAULT_OUT = REPOSITORY_ROOT / "runs/janelle_cross_view_tail_diagnostic_20260728"
DEFAULT_REPORT_OUT = REPOSITORY_ROOT / "ara/evidence/janelle-cross-view-tail-diagnostic-2026-07-28"
REFERENCE_CELL = ("frame_00008", "C0001")
ARMS = ("pursuit", "error_only")
DETAIL_KEYS = (
    "detail_highpass_sigma_0_75_mse",
    "detail_highpass_sigma_1_5_mse",
    "detail_highpass_sigma_3_mse",
    "detail_laplacian_mse",
    "detail_residual_mse",
    "detail_sobel_mse",
)
PREFIX_TENSORS = (
    "means",
    "log_scales",
    "rotations",
    "colors",
    "opacities",
    "scale_max",
    "color_grads",
    "background_mask",
    "filter_variance",
)
SOURCE_FILES = (
    "scripts/experiments/run_janelle_cross_view_tail_diagnostic.py",
    "scripts/experiments/fit032_janelle_dipole_screen.py",
    "scripts/experiments/fit033_janelle_highpass_solve.py",
    "scripts/experiments/fit040_janelle_production_pursuit.py",
    "scripts/experiments/fit041_janelle_equal_base_error_tail.py",
    "benchmarks/highpass_births.py",
    "benchmarks/residual_birth_color_solve.py",
    "src/structsplat/detail_pursuit.py",
    "src/structsplat/safe_schedule.py",
    "tasks/FIT-031-error-only-fine-detail-tail.md",
    "tasks/FIT-040-opt-in-orthogonal-detail-pursuit-tail.md",
    "tasks/FIT-041-equal-base-error-tail-control.md",
    "tasks/FIT-042-independent-fine-detail-pursuit-confirmation.md",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def _atomic_json(path: Path, payload: Any) -> None:
    _atomic_text(
        path,
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
    )


def _git(*arguments: str) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPOSITORY_ROOT), *arguments],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _save_rgb(path: Path, value: torch.Tensor | np.ndarray) -> None:
    tensor = torch.as_tensor(value).detach().cpu().clamp(0.0, 1.0)
    array = tensor.mul(255.0).round().to(torch.uint8).numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array, mode="RGB").save(path)


def _save_mask(path: Path, value: torch.Tensor | np.ndarray) -> None:
    array = torch.as_tensor(value).detach().cpu().to(torch.uint8).mul(255).numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array, mode="L").save(path)


def _crop_bounds(
    shape: tuple[int, int],
    center_yx: tuple[int, int],
    half_size: int,
) -> tuple[int, int, int, int]:
    height, width = shape
    crop_width = min(2 * half_size, width)
    crop_height = min(2 * half_size, height)
    center_y, center_x = center_yx
    x0 = max(0, min(width - crop_width, center_x - crop_width // 2))
    y0 = max(0, min(height - crop_height, center_y - crop_height // 2))
    return x0, y0, x0 + crop_width, y0 + crop_height


def _save_crop(
    path: Path,
    value: torch.Tensor,
    bounds: tuple[int, int, int, int],
) -> None:
    x0, y0, x1, y1 = bounds
    _save_rgb(path, value[y0:y1, x0:x1])


def _save_residual_crop(
    path: Path,
    rendered: torch.Tensor,
    target: torch.Tensor,
    bounds: tuple[int, int, int, int],
    scale: float,
) -> None:
    x0, y0, x1, y1 = bounds
    error = (
        (rendered - target).abs().mean(dim=2)[y0:y1, x0:x1].detach().cpu()
        / max(float(scale), 1e-12)
    ).clamp(0.0, 1.0)
    heat = torch.stack(
        (
            error,
            torch.sqrt(error) * 0.72,
            (1.0 - error) * 0.10,
        ),
        dim=2,
    )
    _save_rgb(path, heat)


def _snapshot_sources(out: Path) -> list[dict[str, Any]]:
    records = []
    for relative in SOURCE_FILES:
        source = REPOSITORY_ROOT / relative
        destination = out / "source_snapshot" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        records.append(
            {
                "path": relative,
                "sha256": _sha256(source),
                "bytes": source.stat().st_size,
            }
        )
    return records


def _discover_cells(args: argparse.Namespace) -> list[tuple[str, str]]:
    requested_frames = set(args.frames or [])
    requested_views = set(args.views or [])
    cells = []
    for frame_path in sorted(args.capture_root.glob("frame_*")):
        if not frame_path.is_dir():
            continue
        frame = frame_path.name
        if requested_frames and frame not in requested_frames:
            continue
        for archive in sorted((frame_path / "gaussians2d").glob("*.rtgsv")):
            view = archive.stem
            if requested_views and view not in requested_views:
                continue
            if (frame, view) == REFERENCE_CELL:
                continue
            cells.append((frame, view))
    if args.limit is not None:
        cells = cells[: int(args.limit)]
    if not cells:
        raise RuntimeError("no eligible Janelle view/frame cells were discovered")
    return cells


def _load_compact_view(path: Path, realtime_root: Path):
    source_root = str((realtime_root / "src").resolve())
    if source_root not in sys.path:
        sys.path.insert(0, source_root)
    from rtgs.data.compact_views import CompactView

    return CompactView.load(path, device="cpu")


def _field_from_observation(
    observation,
    device: torch.device,
    scale_x: float,
    scale_y: float,
) -> GaussianField:
    amplitudes = observation.amplitudes.to(device)
    opacities = None
    if not bool(torch.equal(amplitudes, torch.ones_like(amplitudes))):
        probabilities = amplitudes.clamp(1e-6, 1.0 - 1e-6)
        opacities = torch.logit(probabilities)
    field = GaussianField(
        observation.local_means().to(device),
        observation.log_scales.to(device),
        observation.rotations.to(device),
        observation.colors.to(device),
        opacities,
        None,
        (None if observation.color_grads is None else observation.color_grads.to(device)),
        None,
        (None if observation.filter_variance is None else observation.filter_variance.to(device)),
    )
    scale = torch.tensor(
        [scale_x, scale_y],
        device=device,
        dtype=field.means.dtype,
    )
    field.means = (field.means + 0.5) * scale - 0.5
    field.log_scales = field.log_scales + torch.log(scale)
    if field.filter_variance is not None:
        field.filter_variance = field.filter_variance * float(scale_x * scale_y)
    # run_safe_schedule promotes legacy opaque fields to this exact finite logit
    # before applying any tail.  Materialize it now so the persisted base and the
    # separately evaluated baseline are the actual shared schedule-entry state.
    if field.opacities is None:
        field.opacities = torch.full(
            (field.n,),
            10.0,
            device=device,
            dtype=field.means.dtype,
        )
    return field


def _constraint(
    mask_cpu: np.ndarray,
    target: torch.Tensor,
    cfg,
    boundary_band: float,
) -> _MaskConstraint:
    return _MaskConstraint.from_mask(
        mask_cpu,
        target.device,
        target.dtype,
        cfg.sigma_cutoff,
        cfg.mask_margin,
        aa_dilation=cfg.aa_dilation,
        min_scale=0.35,
        cap_mode=cfg.mask_cap_mode,
        undercoverage_band=boundary_band,
    )


def _canonical_base(
    args: argparse.Namespace,
    frame: str,
    view_id: str,
    cell_dir: Path,
    device: torch.device,
) -> dict[str, Any]:
    archive = (args.capture_root / frame / "gaussians2d" / f"{view_id}.rtgsv").resolve()
    compact = _load_compact_view(archive, args.realtime_root)
    prepare_args = argparse.Namespace(
        realtime_root=args.realtime_root,
        capture_root=args.capture_root,
        frame=frame,
        view_id=view_id,
        max_side=int(args.max_side),
        field=archive,
    )
    prepared = _prepare_janelle(prepare_args)
    if compact.alpha is None:
        raise RuntimeError("compact field is missing its authoritative alpha")
    packed_alpha = compact.alpha.crop_mask()
    prepared_alpha = prepared["prepared"].alpha_crop.detach().cpu().bool()
    binding_checks = {
        "view_id": compact.view_id == view_id,
        "fit_window": (tuple(compact.observation.fit_window) == tuple(prepared["fit_window"])),
        "packed_alpha_exact": torch.equal(packed_alpha, prepared_alpha),
        "rgb_source_sha256": (
            _sha256(Path(prepared["image_path"])) == compact.source["rgb"]["sha256"]
        ),
        "mask_source_sha256": (
            compact.source["mask"] is not None
            and _sha256(Path(prepared["mask_path"])) == compact.source["mask"]["sha256"]
        ),
        "normalized_blend": compact.observation.blend_mode == "normalized",
        "structsplat_provider": compact.observation.provider == "structsplat",
    }
    if not all(binding_checks.values()):
        raise RuntimeError(f"source binding failed for {frame}/{view_id}: {binding_checks}")

    target_cpu = np.asarray(prepared["target"], dtype=np.float32)
    mask_cpu = np.asarray(prepared["mask"], dtype=bool)
    target = torch.as_tensor(
        target_cpu,
        device=device,
        dtype=torch.float32,
    ).contiguous()
    mask = torch.as_tensor(mask_cpu, device=device, dtype=torch.bool)
    target_path = cell_dir / "source/target.png"
    mask_path = cell_dir / "source/mask.png"
    _save_rgb(target_path, target)
    _save_mask(mask_path, mask)

    scale_x, scale_y = (float(value) for value in prepared["scale"])
    base = _field_from_observation(
        compact.observation,
        device,
        scale_x,
        scale_y,
    )
    cfg = replace(
        _base_config(args),
        color_solve_maxiter=1,
        color_solve_lambda=1e30,
    )
    constraint = _constraint(
        mask_cpu,
        target,
        cfg,
        float(args.boundary_band),
    )
    before_means = base.means.detach().clone()
    before_scales = base.log_scales.detach().clone()
    constraint.apply(base, cfg, refresh=True)
    first_constraint_delta = {
        "mean_max_abs": float((base.means - before_means).abs().max()),
        "log_scale_max_abs": float((base.log_scales - before_scales).abs().max()),
    }
    # Anisotropic certification can tighten scale_max once after the first
    # projection because its active-axis test sees the newly capped field.
    # Stabilize that auxiliary tensor before persisting the shared base; the
    # next refresh (including schedule entry) must then be exactly idempotent.
    before_means = base.means.detach().clone()
    before_scales = base.log_scales.detach().clone()
    before_caps = base.scale_max.detach().clone()
    constraint.apply(base, cfg, refresh=True)
    stabilization_delta = {
        "mean_max_abs": float((base.means - before_means).abs().max()),
        "log_scale_max_abs": float((base.log_scales - before_scales).abs().max()),
        "scale_max_max_abs": float((base.scale_max - before_caps).abs().max()),
    }
    before_means = base.means.detach().clone()
    before_scales = base.log_scales.detach().clone()
    before_caps = base.scale_max.detach().clone()
    constraint.apply(base, cfg, refresh=True)
    idempotence_delta = {
        "mean_max_abs": float((base.means - before_means).abs().max()),
        "log_scale_max_abs": float((base.log_scales - before_scales).abs().max()),
        "scale_max_max_abs": float((base.scale_max - before_caps).abs().max()),
    }
    if any(value != 0.0 for value in idempotence_delta.values()):
        raise RuntimeError(
            "shared base constraint adapter did not reach an exact fixed point: "
            f"{idempotence_delta}"
        )
    base_path = cell_dir / "base/field.npz"
    base_path.parent.mkdir(parents=True, exist_ok=True)
    base.save(str(base_path))
    baseline, base_render, base_quality = _evaluate_all(
        base,
        target,
        mask,
        cfg,
        constraint,
        float(args.coverage_tau),
    )
    _save_rgb(cell_dir / "images/full/target.png", target)
    _save_rgb(cell_dir / "images/full/base.png", base_render)

    residual = base_render - target
    highpass = residual - gaussian_blur(residual, 1.5)
    deep = constraint.sdf_flat.reshape(mask.shape) > float(constraint.margin) + 6.0
    if not bool(deep.any()):
        deep = mask
    score = highpass.abs().mean(dim=2).masked_fill(~deep, -float("inf"))
    peak = int(torch.argmax(score))
    width = int(target.shape[1])
    center = (peak // width, peak % width)
    bounds = _crop_bounds(
        (int(target.shape[0]), width),
        center,
        int(args.crop_half_size),
    )
    baseline_abs = residual.abs().mean(dim=2)[deep]
    residual_scale = float(torch.quantile(baseline_abs, 0.99).clamp_min(1e-6))
    _save_crop(cell_dir / "images/detail/target.png", target, bounds)
    _save_crop(cell_dir / "images/detail/base.png", base_render, bounds)
    _save_residual_crop(
        cell_dir / "images/detail/base_error.png",
        base_render,
        target,
        bounds,
        residual_scale,
    )

    source = {
        "frame": frame,
        "view_id": view_id,
        "archive": str(archive),
        "archive_sha256": compact.sha256,
        "archive_bytes": compact.bytes,
        "image": str(Path(prepared["image_path"]).resolve()),
        "image_sha256": _sha256(Path(prepared["image_path"])),
        "mask": str(Path(prepared["mask_path"]).resolve()),
        "mask_sha256": _sha256(Path(prepared["mask_path"])),
        "target": str(target_path.resolve()),
        "target_sha256": _sha256(target_path),
        "materialized_mask": str(mask_path.resolve()),
        "materialized_mask_sha256": _sha256(mask_path),
        "binding_checks": binding_checks,
        "compact_calibration_sha256": compact.calibration_sha256,
        "compact_fit_config_digest": compact.observation.fit_config_digest,
        "compact_producer_version": compact.observation.producer_version,
        "compact_n_init": compact.observation.n_init,
        "native_size": list(prepared["native_size"]),
        "fit_size": list(prepared["fit_size"]),
        "fit_window": list(prepared["fit_window"]),
        "scale": list(prepared["scale"]),
        "pillow": PILLOW_VERSION,
    }
    base_record = {
        "path": str(base_path.resolve()),
        "sha256": _sha256(base_path),
        "rows": int(base.n),
        "constraint_resolution_adapter": {
            "first_projection": first_constraint_delta,
            "anisotropic_cap_stabilization": stabilization_delta,
            "fixed_point_check": idempotence_delta,
        },
        "baseline": baseline,
        "detail_crop_bounds_xyxy": list(bounds),
        "detail_crop_center_yx": list(center),
        "detail_crop_rule": ("maximum pre-treatment deep sigma-1.5 RGB high-pass residual"),
        "residual_heatmap_p99_scale": residual_scale,
        "explicit_opaque_logit": 10.0,
    }
    eligible = int(baseline["detail_deep_pixels"]) >= int(args.minimum_deep_pixels)
    return {
        "source": source,
        "base_record": base_record,
        "base": base,
        "base_quality": base_quality,
        "target": target,
        "mask": mask,
        "mask_cpu": mask_cpu,
        "cfg": cfg,
        "constraint": constraint,
        "crop_bounds": bounds,
        "residual_scale": residual_scale,
        "eligible": eligible,
    }


def _schedule(
    arm: str,
    base_rows: int,
    args: argparse.Namespace,
) -> SafeScheduleConfig:
    defaults = SafeScheduleConfig()
    values: dict[str, Any] = {
        "capacity": base_rows,
        "storage_policy": "dynamic",
        "boundary_enabled": True,
        "coverage_target_gaussians": base_rows,
        "detail_target_gaussians": base_rows,
        "coverage_tau": float(args.coverage_tau),
        "boundary_band": float(args.boundary_band),
        "bootstrap": _disabled_phase(defaults.bootstrap, base_rows),
        "coverage": _disabled_phase(defaults.coverage, base_rows),
        "detail": _disabled_phase(defaults.detail, base_rows),
        "boundary": _disabled_phase(defaults.boundary, base_rows),
        "redistribution": _disabled_phase(defaults.redistribution, base_rows),
        "polish": _disabled_phase(defaults.polish, base_rows),
    }
    if arm == "pursuit":
        values["pursuit_tail_enabled"] = True
    elif arm == "error_only":
        values["error_tail_fraction"] = float(args.error_fraction)
    else:
        raise ValueError(f"unknown arm {arm!r}")
    return SafeScheduleConfig(**values)


def _reduction(before: float, after: float) -> float:
    return 0.0 if before <= 0.0 else 1.0 - after / before


def _prefix_exact(
    base: GaussianField,
    candidate: GaussianField,
) -> tuple[bool, dict[str, bool]]:
    checks = {}
    for name in PREFIX_TENSORS:
        before = getattr(base, name)
        after = getattr(candidate, name)
        if before is None:
            checks[name] = after is None
        else:
            checks[name] = bool(
                after is not None
                and after.shape[0] >= base.n
                and torch.equal(before, after[: base.n])
            )
    return all(checks.values()), checks


def _load_completed_arm(
    result_path: Path,
    expected_base_sha256: str,
    expected_target_sha256: str,
    expected_mask_sha256: str,
) -> dict[str, Any] | None:
    if not result_path.is_file():
        return None
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    binding = payload.get("source", {})
    expected = (
        payload.get("schema") == f"{SCHEMA}.arm"
        and binding.get("base_sha256") == expected_base_sha256
        and binding.get("target_sha256") == expected_target_sha256
        and binding.get("mask_sha256") == expected_mask_sha256
    )
    if not expected:
        raise RuntimeError(f"stale or mismatched arm result: {result_path}")
    return payload


def _run_arm(
    arm: str,
    prepared: dict[str, Any],
    cell_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    arm_dir = cell_dir / "arms" / arm
    result_path = arm_dir / "result.json"
    source = prepared["source"]
    base_record = prepared["base_record"]
    cached = _load_completed_arm(
        result_path,
        base_record["sha256"],
        source["target_sha256"],
        source["materialized_mask_sha256"],
    )
    if cached is not None:
        return cached

    torch.manual_seed(int(args.seed))
    base = GaussianField.load(
        base_record["path"],
        device=prepared["target"].device,
    )
    schedule = _schedule(arm, base.n, args)
    started = time.perf_counter()
    result = run_safe_schedule(
        base,
        prepared["target"],
        prepared["mask"],
        prepared["cfg"],
        schedule,
        verbose=not args.quiet,
    )
    elapsed = time.perf_counter() - started
    final, rendered, final_quality = _evaluate_all(
        result["field"],
        prepared["target"],
        prepared["mask"],
        prepared["cfg"],
        prepared["constraint"],
        float(args.coverage_tau),
    )
    reductions = {
        key: _reduction(
            float(base_record["baseline"][key]),
            float(final[key]),
        )
        for key in DETAIL_KEYS
    }
    safe, protected_reasons = safe_commit_decision(
        prepared["base_quality"],
        final_quality,
        schedule.tolerances,
        schedule.hole_regression_budget,
    )
    outside_exact = (
        float(final["outside_max_abs"]) == 0.0 and float(final["outside_coverage_max"]) == 0.0
    )
    protected_safe = bool(safe and outside_exact)
    target_reached_common = bool(
        protected_safe
        and reductions["detail_highpass_sigma_1_5_mse"]
        >= float(schedule.pursuit_tail_highpass_target)
        and reductions["detail_laplacian_mse"] >= float(schedule.pursuit_tail_laplacian_target)
    )
    tail = result["pursuit_tail"] if arm == "pursuit" else result["error_tail"]
    activated_rows = int(tail["activated_rows"])
    prefix_exact = None
    prefix_checks = None
    if arm == "pursuit":
        prefix_exact, prefix_checks = _prefix_exact(base, result["field"])
        protected_safe = bool(protected_safe and prefix_exact)
        target_reached_common = bool(target_reached_common and prefix_exact)

    arm_dir.mkdir(parents=True, exist_ok=True)
    field_path = arm_dir / "field.npz"
    result["field"].save(str(field_path))
    full_path = cell_dir / "images/full" / f"{arm}.png"
    detail_path = cell_dir / "images/detail" / f"{arm}.png"
    error_path = cell_dir / "images/detail" / f"{arm}_error.png"
    _save_rgb(full_path, rendered)
    _save_crop(detail_path, rendered, prepared["crop_bounds"])
    _save_residual_crop(
        error_path,
        rendered,
        prepared["target"],
        prepared["crop_bounds"],
        prepared["residual_scale"],
    )
    _atomic_json(arm_dir / "history.json", result["history"])
    payload = {
        "schema": f"{SCHEMA}.arm",
        "arm": arm,
        "source": {
            "base": base_record["path"],
            "base_sha256": base_record["sha256"],
            "target": source["target"],
            "target_sha256": source["target_sha256"],
            "mask": source["materialized_mask"],
            "mask_sha256": source["materialized_mask_sha256"],
        },
        "fit_config": asdict(prepared["cfg"]),
        "schedule": asdict(schedule),
        "baseline": base_record["baseline"],
        "final": final,
        "relative_reductions": reductions,
        "tail": tail,
        "activated_rows": activated_rows,
        "target_reached_common_25hp_20lap": target_reached_common,
        "protected_safe": protected_safe,
        "protected_reasons": list(protected_reasons),
        "outside_exact_zero": outside_exact,
        "inherited_prefix_exact": prefix_exact,
        "inherited_prefix_checks": prefix_checks,
        "foreground_psnr_gain_db": (
            float(final["foreground_psnr_db"])
            - float(base_record["baseline"]["foreground_psnr_db"])
        ),
        "highpass_reduction_per_1000_rows": (
            1000.0 * reductions["detail_highpass_sigma_1_5_mse"] / max(activated_rows, 1)
        ),
        "laplacian_reduction_per_1000_rows": (
            1000.0 * reductions["detail_laplacian_mse"] / max(activated_rows, 1)
        ),
        "seconds": elapsed,
        "schedule_seconds": float(result["seconds"]),
        "attempted_steps": int(result["attempted_steps"]),
        "accepted_steps": int(result["accepted_steps"]),
        "converged": bool(result["converged"]),
        "storage": result["storage"],
        "field": {
            "path": str(field_path.resolve()),
            "sha256": _sha256(field_path),
            "rows": int(result["field"].n),
        },
        "images": {
            "full": str(full_path.resolve()),
            "detail": str(detail_path.resolve()),
            "detail_error": str(error_path.resolve()),
        },
    }
    _atomic_json(result_path, payload)
    return payload


def _winner(arms: dict[str, dict[str, Any]]) -> tuple[str, str]:
    pursuit = arms["pursuit"]
    error = arms["error_only"]
    pursuit_pass = bool(pursuit["target_reached_common_25hp_20lap"])
    error_pass = bool(error["target_reached_common_25hp_20lap"])
    if pursuit_pass and not error_pass:
        return "pursuit", "only pursuit reached the shared detail target"
    if error_pass and not pursuit_pass:
        return "error_only", "only error-only reached the shared detail target"
    if pursuit_pass and error_pass:
        pursuit_rows = int(pursuit["activated_rows"])
        error_rows = int(error["activated_rows"])
        if pursuit_rows < error_rows:
            return "pursuit", "both reached target; pursuit used fewer rows"
        if error_rows < pursuit_rows:
            return "error_only", "both reached target; error-only used fewer rows"
        return "tie", "both reached target with equal added rows"
    return "neither", "neither arm reached the shared detail target"


def _cell_payload(
    prepared: dict[str, Any],
    arms: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    winner, reason = _winner(arms)
    pursuit = arms["pursuit"]
    error = arms["error_only"]
    comparison = {
        "winner": winner,
        "winner_rule": (
            "reach >=25% deep sigma-1.5 high-pass and >=20% deep Laplacian "
            "reduction while protected-safe; if both pass, fewer added rows wins"
        ),
        "reason": reason,
        "pursuit_fewer_rows": (int(pursuit["activated_rows"]) < int(error["activated_rows"])),
        "pursuit_higher_highpass_reduction": (
            float(pursuit["relative_reductions"]["detail_highpass_sigma_1_5_mse"])
            > float(error["relative_reductions"]["detail_highpass_sigma_1_5_mse"])
        ),
        "pursuit_higher_laplacian_reduction": (
            float(pursuit["relative_reductions"]["detail_laplacian_mse"])
            > float(error["relative_reductions"]["detail_laplacian_mse"])
        ),
        "error_higher_foreground_psnr_gain": (
            float(error["foreground_psnr_gain_db"]) > float(pursuit["foreground_psnr_gain_db"])
        ),
    }
    return {
        "schema": f"{SCHEMA}.cell",
        "source": prepared["source"],
        "eligible": bool(prepared["eligible"]),
        "eligibility_rule": "at least 4096 deep-interior pixels",
        "base": prepared["base_record"],
        "arms": arms,
        "comparison": comparison,
    }


def _percentiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"p25": None, "median": None, "p75": None, "mean": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "p25": float(np.quantile(array, 0.25)),
        "median": float(np.quantile(array, 0.50)),
        "p75": float(np.quantile(array, 0.75)),
        "mean": float(array.mean()),
    }


def _arm_aggregate(
    cells: list[dict[str, Any]],
    arm: str,
) -> dict[str, Any]:
    records = [cell["arms"][arm] for cell in cells]
    return {
        "cells": len(records),
        "target_reached": sum(bool(row["target_reached_common_25hp_20lap"]) for row in records),
        "protected_safe": sum(bool(row["protected_safe"]) for row in records),
        "added_rows": _percentiles([float(row["activated_rows"]) for row in records]),
        "highpass_reduction": _percentiles(
            [float(row["relative_reductions"]["detail_highpass_sigma_1_5_mse"]) for row in records]
        ),
        "laplacian_reduction": _percentiles(
            [float(row["relative_reductions"]["detail_laplacian_mse"]) for row in records]
        ),
        "foreground_psnr_gain_db": _percentiles(
            [float(row["foreground_psnr_gain_db"]) for row in records]
        ),
        "seconds": _percentiles([float(row["seconds"]) for row in records]),
        "total_seconds": sum(float(row["seconds"]) for row in records),
        "total_added_rows": sum(int(row["activated_rows"]) for row in records),
    }


def _aggregate(
    requested: list[tuple[str, str]],
    cells: list[dict[str, Any]],
    failures: list[dict[str, Any]],
) -> dict[str, Any]:
    eligible = [cell for cell in cells if cell["eligible"]]
    winners = {
        name: sum(cell["comparison"]["winner"] == name for cell in eligible)
        for name in ("pursuit", "error_only", "tie", "neither")
    }
    by_frame = {}
    for frame in sorted({cell["source"]["frame"] for cell in eligible}):
        subset = [cell for cell in eligible if cell["source"]["frame"] == frame]
        by_frame[frame] = {
            "cells": len(subset),
            "winners": {
                name: sum(cell["comparison"]["winner"] == name for cell in subset)
                for name in ("pursuit", "error_only", "tie", "neither")
            },
            "pursuit": _arm_aggregate(subset, "pursuit"),
            "error_only": _arm_aggregate(subset, "error_only"),
        }
    paired = {
        "pursuit_fewer_rows": sum(cell["comparison"]["pursuit_fewer_rows"] for cell in eligible),
        "pursuit_higher_highpass_reduction": sum(
            cell["comparison"]["pursuit_higher_highpass_reduction"] for cell in eligible
        ),
        "pursuit_higher_laplacian_reduction": sum(
            cell["comparison"]["pursuit_higher_laplacian_reduction"] for cell in eligible
        ),
        "error_higher_foreground_psnr_gain": sum(
            cell["comparison"]["error_higher_foreground_psnr_gain"] for cell in eligible
        ),
    }
    return {
        "schema": f"{SCHEMA}.summary",
        "scope": {
            "requested_cells": len(requested),
            "completed_cells": len(cells),
            "eligible_cells": len(eligible),
            "failed_cells": len(failures),
            "excluded_reference_cell": list(REFERENCE_CELL),
            "frames": sorted({frame for frame, _ in requested}),
            "unique_views": sorted({view for _, view in requested}),
            "independent_scene_confirmation": False,
            "interpretation": (
                "paired cross-view transfer diagnostic within one Janelle "
                "capture/session; views and adjacent frames are correlated"
            ),
        },
        "primary_rule": (
            "reach >=25% deep sigma-1.5 high-pass and >=20% deep Laplacian "
            "reduction while protected-safe; if both pass, fewer rows wins"
        ),
        "arms": {
            "pursuit": _arm_aggregate(eligible, "pursuit"),
            "error_only": _arm_aggregate(eligible, "error_only"),
        },
        "winners": winners,
        "paired_counts": paired,
        "by_frame": by_frame,
        "failures": failures,
    }


def _csv_rows(cells: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for cell in cells:
        pursuit = cell["arms"]["pursuit"]
        error = cell["arms"]["error_only"]
        row: dict[str, Any] = {
            "frame": cell["source"]["frame"],
            "view_id": cell["source"]["view_id"],
            "eligible": cell["eligible"],
            "base_rows": cell["base"]["rows"],
            "deep_pixels": cell["base"]["baseline"]["detail_deep_pixels"],
            "winner": cell["comparison"]["winner"],
            "winner_reason": cell["comparison"]["reason"],
        }
        for prefix, arm in (("pursuit", pursuit), ("error_only", error)):
            row.update(
                {
                    f"{prefix}_target_reached": arm["target_reached_common_25hp_20lap"],
                    f"{prefix}_protected_safe": arm["protected_safe"],
                    f"{prefix}_added_rows": arm["activated_rows"],
                    f"{prefix}_highpass_reduction": arm["relative_reductions"][
                        "detail_highpass_sigma_1_5_mse"
                    ],
                    f"{prefix}_laplacian_reduction": arm["relative_reductions"][
                        "detail_laplacian_mse"
                    ],
                    f"{prefix}_foreground_psnr_gain_db": arm["foreground_psnr_gain_db"],
                    f"{prefix}_seconds": arm["seconds"],
                }
            )
        rows.append(row)
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        _atomic_text(path, "")
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _format_pct(value: float) -> str:
    return f"{100.0 * float(value):.1f}%"


def _format_float(value: float, digits: int = 3) -> str:
    return f"{float(value):.{digits}f}"


def _copy_report_images(
    cells: list[dict[str, Any]],
    report_out: Path,
) -> dict[tuple[str, str], dict[str, str]]:
    records = {}
    for cell in cells:
        frame = cell["source"]["frame"]
        view = cell["source"]["view_id"]
        destination = report_out / "images" / frame / view
        destination.mkdir(parents=True, exist_ok=True)
        cell_dir = Path(cell["base"]["path"]).parents[1]
        sources = {
            "target": cell_dir / "images/detail/target.png",
            "base": cell_dir / "images/detail/base.png",
            "base_error": cell_dir / "images/detail/base_error.png",
            "pursuit": cell_dir / "images/detail/pursuit.png",
            "pursuit_error": cell_dir / "images/detail/pursuit_error.png",
            "error_only": cell_dir / "images/detail/error_only.png",
            "error_only_error": cell_dir / "images/detail/error_only_error.png",
        }
        relative = {}
        for name, source in sources.items():
            target = destination / f"{name}.png"
            shutil.copy2(source, target)
            relative[name] = target.relative_to(report_out).as_posix()
        relative["full_dir"] = os.path.relpath(
            cell_dir / "images/full",
            report_out,
        )
        records[(frame, view)] = relative
    return records


def _report_html(
    summary: dict[str, Any],
    cells: list[dict[str, Any]],
    images: dict[tuple[str, str], dict[str, str]],
) -> str:
    pursuit = summary["arms"]["pursuit"]
    error = summary["arms"]["error_only"]
    eligible = int(summary["scope"]["eligible_cells"])
    cards = []
    for cell in cells:
        frame = cell["source"]["frame"]
        view = cell["source"]["view_id"]
        p = cell["arms"]["pursuit"]
        e = cell["arms"]["error_only"]
        comparison = cell["comparison"]
        media = images[(frame, view)]
        winner = comparison["winner"]
        winner_label = {
            "pursuit": "Pursuit",
            "error_only": "Error-only",
            "tie": "Tie",
            "neither": "No target hit",
        }[winner]
        image_panels = "".join(
            (
                '<figure><a href="{src}"><img loading="lazy" src="{src}" '
                'alt="{label} crop for {frame} {view}"></a>'
                "<figcaption>{label}</figcaption></figure>"
            ).format(
                src=html.escape(media[name], quote=True),
                label=html.escape(label),
                frame=html.escape(frame),
                view=html.escape(view),
            )
            for name, label in (
                ("target", "Target"),
                ("base", "Shared base"),
                ("pursuit", "FIT-040 pursuit"),
                ("error_only", "FIT-031 error-only"),
            )
        )
        error_panels = "".join(
            (
                '<figure><a href="{src}"><img loading="lazy" src="{src}" '
                'alt="{label} residual crop for {frame} {view}"></a>'
                "<figcaption>{label}</figcaption></figure>"
            ).format(
                src=html.escape(media[name], quote=True),
                label=html.escape(label),
                frame=html.escape(frame),
                view=html.escape(view),
            )
            for name, label in (
                ("base_error", "Base |error|"),
                ("pursuit_error", "Pursuit |error|"),
                ("error_only_error", "Error-only |error|"),
            )
        )
        full_dir = html.escape(media["full_dir"], quote=True)
        cards.append(
            f"""
            <article class="view-card" data-winner="{winner}" data-frame="{frame}">
              <div class="view-head">
                <div><span class="eyebrow">{frame}</span><h3>{view}</h3></div>
                <span class="winner winner-{winner}">{winner_label}</span>
              </div>
              <p class="reason">{html.escape(comparison["reason"])}</p>
              <div class="image-grid">{image_panels}</div>
              <details>
                <summary>Same-scale residual heatmaps</summary>
                <div class="error-grid">{error_panels}</div>
              </details>
              <table class="mini">
                <thead><tr><th>Arm</th><th>Target</th><th>Rows</th><th>HP</th><th>Lap</th><th>FG PSNR</th></tr></thead>
                <tbody>
                  <tr>
                    <td>Pursuit</td>
                    <td>{"✓" if p["target_reached_common_25hp_20lap"] else "—"}</td>
                    <td>{int(p["activated_rows"]):,}</td>
                    <td>{_format_pct(p["relative_reductions"]["detail_highpass_sigma_1_5_mse"])}</td>
                    <td>{_format_pct(p["relative_reductions"]["detail_laplacian_mse"])}</td>
                    <td>{float(p["foreground_psnr_gain_db"]):+.3f} dB</td>
                  </tr>
                  <tr>
                    <td>Error-only</td>
                    <td>{"✓" if e["target_reached_common_25hp_20lap"] else "—"}</td>
                    <td>{int(e["activated_rows"]):,}</td>
                    <td>{_format_pct(e["relative_reductions"]["detail_highpass_sigma_1_5_mse"])}</td>
                    <td>{_format_pct(e["relative_reductions"]["detail_laplacian_mse"])}</td>
                    <td>{float(e["foreground_psnr_gain_db"]):+.3f} dB</td>
                  </tr>
                </tbody>
              </table>
              <p class="links">
                Full frames:
                <a href="{full_dir}/target.png">target</a> ·
                <a href="{full_dir}/base.png">base</a> ·
                <a href="{full_dir}/pursuit.png">pursuit</a> ·
                <a href="{full_dir}/error_only.png">error-only</a>
              </p>
            </article>
            """
        )

    frame_rows = []
    for frame, record in summary["by_frame"].items():
        frame_rows.append(
            f"""
            <tr>
              <td>{html.escape(frame)}</td>
              <td>{record["cells"]}</td>
              <td>{record["pursuit"]["target_reached"]}/{record["cells"]}</td>
              <td>{record["error_only"]["target_reached"]}/{record["cells"]}</td>
              <td>{record["winners"]["pursuit"]}</td>
              <td>{record["winners"]["error_only"]}</td>
              <td>{record["winners"]["neither"]}</td>
            </tr>
            """
        )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Janelle cross-view fine-detail tails</title>
  <style>
    :root {{
      color-scheme: dark;
      --bg: #09101d; --card: #121c2d; --card2: #17243a;
      --text: #edf3fb; --muted: #9eacc0; --line: #2b3c55;
      --cyan: #3dd6c6; --amber: #ffbe4d; --red: #ff7070; --blue: #65a9ff;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; background: var(--bg); color: var(--text);
      font: 15px/1.55 system-ui, sans-serif; }}
    main {{ max-width: 1500px; margin: auto; padding: 36px 24px 80px; }}
    h1 {{ font-size: clamp(30px, 4vw, 54px); line-height: 1.05; margin: 8px 0 14px; }}
    h2 {{ margin-top: 42px; font-size: 25px; }}
    h3 {{ margin: 2px 0 0; font-size: 23px; }}
    a {{ color: #8ac7ff; }}
    .lede {{ max-width: 980px; color: var(--muted); font-size: 18px; }}
    .caveat {{ border: 1px solid #765b25; background: #2d2415; padding: 14px 18px;
      border-radius: 12px; max-width: 1100px; }}
    .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 12px; margin: 28px 0; }}
    .metric {{ background: var(--card); border: 1px solid var(--line); border-radius: 14px;
      padding: 16px; }}
    .metric strong {{ display: block; font-size: 30px; }}
    .metric span, .eyebrow {{ color: var(--muted); font-size: 12px; text-transform: uppercase;
      letter-spacing: .08em; }}
    .table-wrap {{ overflow-x: auto; }}
    table {{ border-collapse: collapse; width: 100%; background: var(--card);
      border: 1px solid var(--line); }}
    th, td {{ padding: 10px 12px; border-bottom: 1px solid var(--line); text-align: right; }}
    th:first-child, td:first-child {{ text-align: left; }}
    .controls {{ display: flex; flex-wrap: wrap; gap: 8px; margin: 18px 0; }}
    button {{ border: 1px solid var(--line); background: var(--card2); color: var(--text);
      border-radius: 999px; padding: 8px 13px; cursor: pointer; }}
    button.active {{ border-color: var(--cyan); color: var(--cyan); }}
    .gallery {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(570px, 1fr));
      gap: 18px; }}
    .view-card {{ background: var(--card); border: 1px solid var(--line); border-radius: 16px;
      padding: 16px; min-width: 0; }}
    .view-card[hidden] {{ display: none; }}
    .view-head {{ display: flex; align-items: center; justify-content: space-between; gap: 12px; }}
    .winner {{ border-radius: 999px; padding: 6px 10px; font-weight: 700; }}
    .winner-pursuit {{ background: #133d39; color: var(--cyan); }}
    .winner-error_only {{ background: #342b15; color: var(--amber); }}
    .winner-neither {{ background: #392026; color: #ff9a9a; }}
    .winner-tie {{ background: #203653; color: #9bc5ff; }}
    .reason {{ color: var(--muted); margin: 8px 0 14px; }}
    .image-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 7px; }}
    .error-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 7px;
      margin-top: 10px; }}
    figure {{ margin: 0; min-width: 0; }}
    img {{ width: 100%; height: auto; display: block; border-radius: 7px;
      image-rendering: auto; background: #000; }}
    figcaption {{ color: var(--muted); font-size: 12px; padding-top: 4px; }}
    details {{ margin: 11px 0; color: var(--muted); }}
    summary {{ cursor: pointer; }}
    .mini {{ font-size: 13px; margin-top: 10px; }}
    .mini th, .mini td {{ padding: 7px 8px; }}
    .links {{ color: var(--muted); font-size: 13px; margin-bottom: 0; }}
    .method-note {{ color: var(--muted); max-width: 1050px; }}
    @media (max-width: 720px) {{
      main {{ padding: 24px 12px 60px; }}
      .gallery {{ grid-template-columns: 1fr; }}
      .image-grid {{ grid-template-columns: repeat(2, 1fr); }}
    }}
  </style>
</head>
<body><main>
  <span class="eyebrow">StructSplat · 2026-07-28</span>
  <h1>Janelle cross-view fine-detail tails</h1>
  <p class="lede">A paired comparison of the production FIT-040 orthogonal-pursuit tail and
  FIT-031 error-only tail on every remaining persisted Janelle view. Each card uses a
  deterministic pre-treatment detail crop, so the allocation behavior can be judged visually.</p>
  <p class="caveat"><strong>Scope boundary:</strong> this is a correlated transfer diagnostic
  within one capture/session, using archived mask-contained bases. It is not independent-scene
  FIT-042 confirmation and does not establish a population-level success rate.</p>

  <div class="metrics">
    <div class="metric"><strong>{eligible}</strong><span>eligible / completed views</span></div>
    <div class="metric"><strong>{pursuit["target_reached"]}/{eligible}</strong><span>pursuit target hits</span></div>
    <div class="metric"><strong>{error["target_reached"]}/{eligible}</strong><span>error-only target hits</span></div>
    <div class="metric"><strong>{summary["winners"]["pursuit"]}</strong><span>pursuit wins</span></div>
    <div class="metric"><strong>{pursuit["added_rows"]["median"]:,.0f}</strong><span>pursuit median added rows</span></div>
    <div class="metric"><strong>{error["added_rows"]["median"]:,.0f}</strong><span>error-only median added rows</span></div>
  </div>

  <h2>Aggregate comparison</h2>
  <div class="table-wrap"><table>
    <thead><tr><th>Method</th><th>Target hits</th><th>Protected-safe</th>
      <th>Median rows</th><th>Median HP reduction</th><th>Median Lap reduction</th>
      <th>Median FG PSNR gain</th></tr></thead>
    <tbody>
      <tr><td>FIT-040 pursuit</td><td>{pursuit["target_reached"]}/{eligible}</td>
        <td>{pursuit["protected_safe"]}/{eligible}</td>
        <td>{pursuit["added_rows"]["median"]:,.0f}</td>
        <td>{_format_pct(pursuit["highpass_reduction"]["median"])}</td>
        <td>{_format_pct(pursuit["laplacian_reduction"]["median"])}</td>
        <td>{pursuit["foreground_psnr_gain_db"]["median"]:+.3f} dB</td></tr>
      <tr><td>FIT-031 error-only</td><td>{error["target_reached"]}/{eligible}</td>
        <td>{error["protected_safe"]}/{eligible}</td>
        <td>{error["added_rows"]["median"]:,.0f}</td>
        <td>{_format_pct(error["highpass_reduction"]["median"])}</td>
        <td>{_format_pct(error["laplacian_reduction"]["median"])}</td>
        <td>{error["foreground_psnr_gain_db"]["median"]:+.3f} dB</td></tr>
    </tbody>
  </table></div>
  <p class="method-note">Primary target: at least 25% deep sigma-1.5 high-pass reduction and
  20% deep Laplacian reduction, with protected metrics safe. Natural stopping policies are used:
  pursuit adds 128-row waves up to 2,048 rows; error-only estimates a 50%-residual-support tail
  and optimizes it to its deterministic stopping condition. Row totals are therefore an outcome,
  not an equal-row treatment.</p>

  <h2>Frame breakdown</h2>
  <div class="table-wrap"><table>
    <thead><tr><th>Frame</th><th>Views</th><th>Pursuit hits</th><th>Error hits</th>
      <th>Pursuit wins</th><th>Error wins</th><th>Neither</th></tr></thead>
    <tbody>{"".join(frame_rows)}</tbody>
  </table></div>

  <h2>Per-view visual judgment</h2>
  <div class="controls">
    <button class="active" data-filter="all">All</button>
    <button data-filter="pursuit">Pursuit wins</button>
    <button data-filter="error_only">Error-only wins</button>
    <button data-filter="neither">No target hit</button>
    <button data-filter="frame_00008">Frame 8</button>
    <button data-filter="frame_00009">Frame 9</button>
  </div>
  <section class="gallery">{"".join(cards)}</section>

  <h2>Files and audit trail</h2>
  <p><a href="comparison.csv">Tidy per-view CSV</a> ·
  <a href="summary.json">Machine-readable summary</a> ·
  <a href="run.md">Protocol and limitations</a></p>
</main>
<script>
  const buttons = [...document.querySelectorAll("button[data-filter]")];
  const cards = [...document.querySelectorAll(".view-card")];
  buttons.forEach(button => button.addEventListener("click", () => {{
    buttons.forEach(item => item.classList.remove("active"));
    button.classList.add("active");
    const filter = button.dataset.filter;
    cards.forEach(card => {{
      card.hidden = !(filter === "all" || card.dataset.winner === filter ||
        card.dataset.frame === filter);
    }});
  }}));
</script>
</body></html>
"""


def _run_markdown(
    args: argparse.Namespace,
    manifest: dict[str, Any],
    summary: dict[str, Any],
) -> str:
    pursuit = summary["arms"]["pursuit"]
    error = summary["arms"]["error_only"]
    eligible = summary["scope"]["eligible_cells"]
    command = " ".join(manifest["command"])
    return f"""# Janelle cross-view fine-detail tail diagnostic

## Scope

This user-requested diagnostic compares FIT-040 orthogonal pursuit with FIT-031's
error-only tail on all persisted Janelle image/view cells except the previously used
`frame_00008/C0001` reference. It is descriptive evidence within one capture session,
not FIT-042 independent-scene confirmation.

## Frozen paired protocol

- Cells: {summary["scope"]["requested_cells"]} automatically discovered remaining
  frame/view pairs across {", ".join(summary["scope"]["frames"])}.
- Shared base: the integrity-checked `.rtgsv` field paired with each calibrated RGB and
  authoritative packed mask, adapted to max-side {args.max_side}.
- Renderer: `{args.renderer}` for both arms.
- FIT-040 arm: shipped 128-row pursuit waves, 2,048-row ceiling, 25% high-pass and
  20% Laplacian stopping targets.
- FIT-031 arm: shipped residual-support estimate at fraction {args.error_fraction},
  512-row allocation waves, and the natural 4,000-step convergence ceiling.
- Shared target rule: protected-safe plus at least 25% deep sigma-1.5 high-pass and
  20% deep Laplacian reduction. If both pass, fewer added rows wins.
- Seed: {args.seed}. No repeated-seed inference is claimed because the methods are
  deterministic under this setup.

## Execution

```bash
{command}
```

- Git commit: `{manifest["environment"]["git_commit"]}`
- Dirty worktree recorded: `{manifest["environment"]["git_dirty"]}`
- GPU: `{manifest["environment"]["gpu"]}`
- Torch / CUDA: `{manifest["environment"]["torch"]}` /
  `{manifest["environment"]["torch_cuda"]}`
- Pillow: `{manifest["environment"]["pillow"]}`
- Source snapshot: `{manifest["source_snapshot"]}`

## Descriptive outcome

- Completed / eligible: {summary["scope"]["completed_cells"]} / {eligible}
- Pursuit target hits: {pursuit["target_reached"]}/{eligible}; median added rows
  {pursuit["added_rows"]["median"]:,.0f}; median high-pass / Laplacian reductions
  {_format_pct(pursuit["highpass_reduction"]["median"])} /
  {_format_pct(pursuit["laplacian_reduction"]["median"])}.
- Error-only target hits: {error["target_reached"]}/{eligible}; median added rows
  {error["added_rows"]["median"]:,.0f}; median high-pass / Laplacian reductions
  {_format_pct(error["highpass_reduction"]["median"])} /
  {_format_pct(error["laplacian_reduction"]["median"])}.
- Winner counts: pursuit {summary["winners"]["pursuit"]}, error-only
  {summary["winners"]["error_only"]}, tie {summary["winners"]["tie"]}, neither
  {summary["winners"]["neither"]}.
- Failures: {summary["scope"]["failed_cells"]}.

## Limitations

1. Views from the same calibrated capture and two adjacent frames are strongly correlated.
2. The bases are archived byte-capped, mask-contained fields (roughly 5k rows), not fresh
   current-pipeline 10–11k fits. The comparison is internally paired but does not reproduce
   the exact prior C0001 base distribution.
3. The arm budgets are natural method budgets, not equal-row budgets. FIT-031 is expected to
   favor broad foreground PSNR; FIT-040 is explicitly optimized for the fine-detail target.
4. Pillow {PILLOW_VERSION} materialized these targets. Exact target and mask PNG hashes are
   recorded per cell.
5. This evidence must not be counted as FIT-042's sealed independent-scene screen or
   confirmation set.
"""


def _build_report(
    args: argparse.Namespace,
    manifest: dict[str, Any],
    summary: dict[str, Any],
    cells: list[dict[str, Any]],
) -> None:
    report_out = args.report_out.resolve()
    report_out.mkdir(parents=True, exist_ok=True)
    rows = _csv_rows(cells)
    _write_csv(report_out / "comparison.csv", rows)
    _atomic_json(report_out / "summary.json", summary)
    images = _copy_report_images(cells, report_out)
    _atomic_text(
        report_out / "index.html",
        _report_html(summary, cells, images),
    )
    _atomic_text(
        report_out / "run.md",
        _run_markdown(args, manifest, summary),
    )
    report_manifest = {
        "schema": f"{SCHEMA}.report",
        "title": "Janelle cross-view fine-detail tails",
        "summary_sha256": _sha256(report_out / "summary.json"),
        "comparison_csv_sha256": _sha256(report_out / "comparison.csv"),
        "index_sha256": _sha256(report_out / "index.html"),
        "run_markdown_sha256": _sha256(report_out / "run.md"),
        "source_run": str(args.out.resolve()),
        "source_run_manifest_sha256": _sha256(args.out / "manifest.json"),
        "cells": len(cells),
    }
    _atomic_json(report_out / "artifact.json", report_manifest)


def _manifest(
    args: argparse.Namespace,
    cells: list[tuple[str, str]],
    source_records: list[dict[str, Any]],
) -> dict[str, Any]:
    device = torch.device(args.device)
    return {
        "schema": f"{SCHEMA}.manifest",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "in_progress",
        "protocol_status": "post_hoc_user_requested_cross_view_diagnostic",
        "independent_scene_confirmation": False,
        "command": [sys.executable, *sys.argv],
        "capture_root": str(args.capture_root.resolve()),
        "realtime_root": str(args.realtime_root.resolve()),
        "out": str(args.out.resolve()),
        "report_out": str(args.report_out.resolve()),
        "excluded_reference_cell": list(REFERENCE_CELL),
        "requested_cells": [{"frame": frame, "view_id": view} for frame, view in cells],
        "methods": {
            "pursuit": "FIT-040 shipped natural stopping policy",
            "error_only": ("FIT-031 shipped 0.5 residual-support natural stopping policy"),
        },
        "shared_renderer": args.renderer,
        "max_side": int(args.max_side),
        "seed": int(args.seed),
        "source_snapshot": source_records,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "device": str(device),
            "gpu": (torch.cuda.get_device_name(device) if device.type == "cuda" else None),
            "pillow": PILLOW_VERSION,
            "git_commit": _git("rev-parse", "HEAD"),
            "git_dirty": bool(_git("status", "--porcelain")),
        },
    }


def run(args: argparse.Namespace) -> None:
    args.capture_root = args.capture_root.resolve()
    args.realtime_root = args.realtime_root.resolve()
    args.out = args.out.resolve()
    args.report_out = args.report_out.resolve()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA but torch.cuda.is_available() is false")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    cells_requested = _discover_cells(args)
    args.out.mkdir(parents=True, exist_ok=True)
    snapshot_path = args.out / "source_snapshot.json"
    if snapshot_path.is_file():
        source_records = json.loads(snapshot_path.read_text(encoding="utf-8"))
    else:
        source_records = _snapshot_sources(args.out)
        _atomic_json(snapshot_path, source_records)
    manifest_path = args.out / "manifest.json"
    manifest = _manifest(args, cells_requested, source_records)
    if manifest_path.is_file():
        prior = json.loads(manifest_path.read_text(encoding="utf-8"))
        if prior.get("requested_cells") != manifest["requested_cells"]:
            raise RuntimeError("existing run requested-cell manifest differs; use a new --out")
        manifest["created_at_utc"] = prior["created_at_utc"]
    _atomic_json(manifest_path, manifest)

    completed: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    total = len(cells_requested)
    for index, (frame, view_id) in enumerate(cells_requested, start=1):
        cell_dir = args.out / "cells" / frame / view_id
        cell_result = cell_dir / "result.json"
        if cell_result.is_file():
            payload = json.loads(cell_result.read_text(encoding="utf-8"))
            if payload.get("schema") != f"{SCHEMA}.cell":
                raise RuntimeError(f"unexpected completed-cell schema: {cell_result}")
            completed.append(payload)
            print(
                f"[{index}/{total}] {frame}/{view_id}: cached ({payload['comparison']['winner']})",
                flush=True,
            )
            continue
        print(f"[{index}/{total}] {frame}/{view_id}: preparing", flush=True)
        try:
            prepared = _canonical_base(
                args,
                frame,
                view_id,
                cell_dir,
                device,
            )
            if not prepared["eligible"]:
                payload = {
                    "schema": f"{SCHEMA}.cell",
                    "source": prepared["source"],
                    "eligible": False,
                    "eligibility_rule": "at least 4096 deep-interior pixels",
                    "base": prepared["base_record"],
                    "arms": {},
                    "comparison": {
                        "winner": "ineligible",
                        "reason": "too few deep-interior pixels",
                    },
                }
            else:
                arms = {}
                for arm in ARMS:
                    print(
                        f"[{index}/{total}] {frame}/{view_id}: {arm}",
                        flush=True,
                    )
                    arms[arm] = _run_arm(
                        arm,
                        prepared,
                        cell_dir,
                        args,
                    )
                payload = _cell_payload(prepared, arms)
            _atomic_json(cell_result, payload)
            completed.append(payload)
            print(
                f"[{index}/{total}] {frame}/{view_id}: {payload['comparison']['winner']}",
                flush=True,
            )
            del prepared
            if device.type == "cuda":
                torch.cuda.empty_cache()
        except Exception as error:  # noqa: BLE001 - isolate result-bearing cells
            failure = {
                "frame": frame,
                "view_id": view_id,
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            }
            failures.append(failure)
            _atomic_json(cell_dir / "failure.json", failure)
            print(
                f"[{index}/{total}] {frame}/{view_id}: FAILED: {error}",
                file=sys.stderr,
                flush=True,
            )
            if args.fail_fast:
                raise
            if device.type == "cuda":
                torch.cuda.empty_cache()
        progress = {
            "requested": total,
            "completed": len(completed),
            "failures": failures,
            "last_cell": {"frame": frame, "view_id": view_id},
        }
        _atomic_json(args.out / "progress.json", progress)

    summary = _aggregate(cells_requested, completed, failures)
    _atomic_json(args.out / "summary.json", summary)
    _write_csv(args.out / "comparison.csv", _csv_rows(completed))
    manifest["status"] = "complete" if not failures and len(completed) == total else "partial"
    manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["summary_sha256"] = _sha256(args.out / "summary.json")
    manifest["completed_cells"] = len(completed)
    manifest["failed_cells"] = len(failures)
    _atomic_json(manifest_path, manifest)
    _build_report(args, manifest, summary, completed)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-root", type=Path, default=DEFAULT_CAPTURE_ROOT)
    parser.add_argument("--realtime-root", type=Path, default=DEFAULT_REALTIME_ROOT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--report-out", type=Path, default=DEFAULT_REPORT_OUT)
    parser.add_argument("--frames", nargs="*", default=None)
    parser.add_argument("--views", nargs="*", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-side", type=int, default=1200)
    parser.add_argument("--crop-half-size", type=int, default=96)
    parser.add_argument("--minimum-deep-pixels", type=int, default=4096)
    parser.add_argument("--error-fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--renderer", default="cuda")
    parser.add_argument("--mask-margin", type=float, default=0.75)
    parser.add_argument("--boundary-band", type=float, default=4.0)
    parser.add_argument("--coverage-tau", type=float, default=0.05)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
