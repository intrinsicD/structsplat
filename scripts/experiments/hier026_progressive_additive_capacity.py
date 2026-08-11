#!/usr/bin/env python3
"""Run HIER-026's frozen untouched-DIV2K pure-additive capacity confirmation."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import csv
from html import escape
import json
import math
from pathlib import Path
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

from scripts.experiments import hier022_additive_continuation as h22  # noqa: E402
from scripts.experiments import hier023_unit_gauge_continuation as h23  # noqa: E402
from scripts.experiments import hier024_gauge_geometry_projection as h24  # noqa: E402
from structsplat.config import FitConfig, StructureTensorConfig  # noqa: E402
from structsplat.fit import fit  # noqa: E402
from structsplat.gaussians import GaussianField  # noqa: E402
from structsplat.init import build_field  # noqa: E402
from structsplat.progressive_additive_capacity import (  # noqa: E402
    ProgressiveAdditiveCapacityConfig,
    ProgressiveAdditiveCapacityResult,
    fit_progressive_additive_capacity,
)


REPORT_SCHEMA = "structsplat.hier026_progressive_additive_capacity.confirmation.v1"
ARMS = (
    "normalized_plain_n640",
    "additive_plain_n640",
    "additive_projected_n640",
    "cold_additive_projected_n896",
    "progressive_residual_n896",
    "progressive_residual_projected_n896",
    "cold_additive_projected_n960",
)
PROJECTED_ARMS = frozenset(
    (
        "additive_projected_n640",
        "cold_additive_projected_n896",
        "progressive_residual_projected_n896",
        "cold_additive_projected_n960",
    )
)
PURE_ADDITIVE_ARMS = frozenset(set(ARMS) - {"normalized_plain_n640"})
PROGRESSIVE_ARMS = frozenset(
    ("progressive_residual_n896", "progressive_residual_projected_n896")
)
COUNT_BY_ARM = {
    "normalized_plain_n640": 640,
    "additive_plain_n640": 640,
    "additive_projected_n640": 640,
    "cold_additive_projected_n896": 896,
    "progressive_residual_n896": 896,
    "progressive_residual_projected_n896": 896,
    "cold_additive_projected_n960": 960,
}
SELECTION_SALT = "HIER-025-confirm-v1:"
SELECTION_ORDER = ("0895.png", "0860.png", "0898.png", "0847.png")
SELECTION_BINDINGS = {
    "0895.png": "0644b064658788ac2695cfa2d57d4c2704d3d5e3173f310daf06262914deb703",
    "0860.png": "082cd6a3d95e3b16ec770c3502325c1fcb6cc890e9791a7a27b61614e028ef4e",
    "0898.png": "0b554a43bfb78b6ebda36539d5d3f2cdd1568a394ac430981ee0ac5d96aaab7c",
    "0847.png": "10494b910838e73fad90d013d95d07dfc4ffd618f6416f819d372d9788c6d096",
}
SOURCE_BINDINGS = {
    "0895.png": "a1c0888648fed4eb909c6e7f5f5db220ae98861294ebfdfa14b2c72567e96b2b",
    "0860.png": "eac29d623ecfab9e2299c04b49e5da3f282a576eb7f9107d0b88076c972ac3ef",
    "0898.png": "4cd6696b8e59615ceacff729181dd9b0cc5ea936ea9a57e089bb1fe4fe87c347",
    "0847.png": "ce39eab49b45fc08177556f7c9ae0d0e928e283fb3cd471bddf0fbf17db8ca73",
}
ARCHIVE_SHA256 = "20dd31fd84d777bc1cf5d6b7654a3f569c0aec74458ae094122ad1d0489900fc"
ARCHIVE_BYTES = 448_993_893
COEFFICIENT_LIMIT = 16.0
PARITY_LIMIT = 2e-5
FOUR_ARRAY_KEYS = frozenset(("means", "log_scales", "rotations", "colors"))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--max-side", type=int, default=160)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--lpips", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--render-chunk", type=int, default=256)
    parser.add_argument("--error-scale", type=float, default=4.0)
    parser.add_argument("--resume", action="store_true")
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    frozen = {
        "max_side": 160,
        "seeds": [0, 1],
        "device": "cuda",
        "lpips": True,
        "render_chunk": 256,
    }
    for name, expected in frozen.items():
        if getattr(args, name) != expected:
            raise SystemExit(
                f"frozen HIER-026 protocol requires {name}={expected!r}, "
                f"got {getattr(args, name)!r}"
            )
    if not math.isfinite(args.error_scale) or args.error_scale <= 0.0:
        raise SystemExit("--error-scale must be finite and positive")
    if not args.images.is_dir():
        raise SystemExit(f"image directory does not exist: {args.images}")
    args.iters = 500
    args.budgets = [640]


def _command() -> str:
    return " ".join(shlex.quote(value) for value in sys.argv)


def _write_json(path: Path, value: object) -> None:
    h22._write_json(path, value)


def _discover_sources(root: Path) -> list[Path]:
    actual_names = sorted(path.name for path in root.iterdir() if path.is_file())
    if actual_names != sorted(SELECTION_ORDER):
        raise SystemExit(
            "HIER-026 extraction root must contain exactly the four bound members: "
            f"got {actual_names!r}"
        )
    paths = [root / name for name in SELECTION_ORDER]
    hashes = {path.name: h22.report_utils._sha256(path) for path in paths}
    if hashes != SOURCE_BINDINGS:
        raise SystemExit(f"HIER-026 source hash binding differs: {hashes!r}")
    return [path.resolve() for path in paths]


def _snapshot_sources(output_root: Path) -> list[dict[str, object]]:
    paths = (
        Path(__file__).resolve(),
        ROOT / "scripts" / "experiments" / "hier022_additive_continuation.py",
        ROOT / "scripts" / "experiments" / "hier023_unit_gauge_continuation.py",
        ROOT / "scripts" / "experiments" / "hier024_gauge_geometry_projection.py",
        ROOT / "src" / "structsplat" / "progressive_additive_capacity.py",
        ROOT / "src" / "structsplat" / "endpoint_appearance_projection.py",
        ROOT / "tests" / "test_progressive_additive_capacity.py",
        ROOT / "tests" / "test_endpoint_appearance_projection.py",
        ROOT / "tasks" / "HIER-026-progressive-additive-capacity-parity.md",
        ROOT / "scripts" / "check_report_bundle.py",
    )
    records = []
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
                "sha256": h22.report_utils._sha256(destination),
            }
        )
    return records


def _progressive_config(args: argparse.Namespace) -> ProgressiveAdditiveCapacityConfig:
    return ProgressiveAdditiveCapacityConfig(
        base_gaussians=640,
        residual_gaussians=256,
        base_steps=500,
        joint_steps=200,
        checkpoint_every=25,
        lr_means=5e-2,
        lr_scales=3e-2,
        lr_rotations=1e-2,
        lr_coefficients=3e-2,
        feature_cap_px=12.0,
        coefficient_abs_limit=COEFFICIENT_LIMIT,
        sigma_cutoff=3.0,
        render_chunk=args.render_chunk,
        renderer="cuda_additive",
    )


def _cold_fit_config(args: argparse.Namespace, count: int) -> FitConfig:
    return FitConfig(
        iters=500,
        lr_means=5e-2,
        lr_scales=3e-2,
        lr_rot=1e-2,
        lr_color=3e-2,
        optimizer="adam",
        pixel_loss="l1",
        ssim_weight=0.3,
        log_every=25,
        checkpoint_policy="best_psnr_final_count",
        sigma_cutoff=3.0,
        support_fade=False,
        aa_dilation=0.0,
        render_chunk=args.render_chunk,
        renderer="cuda_additive",
        color_basis="constant",
        compute_lpips=False,
        max_gaussians=count,
    )


def _pure_endpoint(field: GaussianField) -> GaussianField:
    if field.opacities is not None or field.color_grads is not None:
        raise RuntimeError("HIER-026 cold additive control is not a constant-color pure field")
    return GaussianField(
        field.means.detach().clone(),
        field.log_scales.detach().clone(),
        field.rotations.detach().clone(),
        field.colors.detach().clone(),
    )


def _coverage(field: GaussianField, target: np.ndarray, args, torch) -> dict[str, float]:
    with torch.no_grad():
        return h22._coverage_record(
            h22._unit_coverage(
                field,
                target.shape[0],
                target.shape[1],
                "cuda_additive",
                args.render_chunk,
            )
        )


def _run_cold_additive(
    target: np.ndarray, seed: int, count: int, args: argparse.Namespace, torch
) -> dict[str, object]:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    init_started = time.perf_counter()
    initial = build_field(
        target,
        h22._init_config(count, seed),
        StructureTensorConfig(),
        device=args.device,
    )
    init_seconds = time.perf_counter() - init_started
    initial_audit = initial.detached()
    target_tensor = torch.as_tensor(target, device=args.device, dtype=torch.float32)
    torch.cuda.reset_peak_memory_stats()
    wall_started = time.perf_counter()
    result = fit(
        initial,
        target_tensor,
        _cold_fit_config(args, count),
        verbose=False,
    )
    training_field = result["field"].detached()
    endpoint = _pure_endpoint(training_field)
    with torch.no_grad():
        rendered = h22._field_render(
            endpoint,
            target.shape[0],
            target.shape[1],
            "cuda_additive",
            args.render_chunk,
        )
    torch.cuda.synchronize()
    wall_seconds = time.perf_counter() - wall_started
    expected = rendered.detach().cpu().numpy().astype(np.float32, copy=False)
    fit_expected = result["render"].detach().cpu().numpy().astype(np.float32, copy=False)
    endpoint_parity = float(
        np.max(np.abs(expected.astype(np.float64) - fit_expected.astype(np.float64)))
    )
    renderer_calls = (
        int(result["iterations_run"])
        + len(result["checkpoint_history"]["iter"])
        + int(bool(result["selected_from_checkpoint"]))
    )
    return {
        "field": endpoint,
        "expected": expected,
        "trajectory": h22._trajectory_baseline(result, 500),
        "renderer_calls": renderer_calls,
        "normalized_calls": 0,
        "additive_numerator_calls": renderer_calls,
        "additive_denominator_calls": 0,
        "renderer_calls_coverage_diagnostic": 1,
        "selected_step": int(result["selected_iter"]),
        "completed": int(result["iterations_run"]) == 500,
        "method_status": (
            "completed" if int(result["iterations_run"]) == 500 else "incomplete"
        ),
        "history": {
            "history": result["history"],
            "checkpoint_history": result["checkpoint_history"],
        },
        "endpoint_parity": endpoint_parity,
        "semantic_family": "additive_rgb_peak_one_v1",
        "renderer": "cuda_additive",
        "fit_seconds": float(result["fit_seconds"]),
        "wall_fit_seconds": wall_seconds,
        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "coverage": _coverage(endpoint, target, args, torch),
        "attempted_steps": 500,
        "gaussian_row_updates": count * 500,
        "base_count": 0,
        "residual_count": count,
        "diagnostic_renderer_calls": 1,
        "init_seconds": init_seconds,
        "initial_field_digest": h24._field_digest(initial_audit),
        "base_shared_digest": None,
        "residual_sha256": None,
        "birth_field_digest": None,
        "appended_initial_digest": None,
        "preprojection_endpoint_digest": h24._field_digest(endpoint),
        "progressive_result": None,
        "audit_initial_field": initial_audit,
        "audit_training_field": training_field,
        "stage_kind": "cold_full_target",
    }


def _progressive_trajectory(
    result: ProgressiveAdditiveCapacityResult,
    *,
    stage: str | None,
    steps: int,
) -> list[dict[str, float]]:
    records = [
        {"step": float(point.step), "psnr_db": point.psnr_db}
        for point in result.trajectory
        if stage is None or point.stage == stage
    ]
    return h22._normalize_trajectory(records, steps)


def _run_progressive(
    target: np.ndarray, seed: int, args: argparse.Namespace, torch
) -> tuple[ProgressiveAdditiveCapacityResult, dict[str, object], dict[str, object]]:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.reset_peak_memory_stats()
    wall_started = time.perf_counter()
    result = fit_progressive_additive_capacity(
        target,
        seed=seed,
        config=_progressive_config(args),
        device=args.device,
        verbose=False,
    )
    torch.cuda.synchronize()
    wall_seconds = time.perf_counter() - wall_started
    peak = int(torch.cuda.max_memory_allocated())
    base_observers = sum(point.stage == "base" for point in result.trajectory)
    joint_observers = sum(point.stage == "joint" for point in result.trajectory)
    base_history = result.stage_histories["base"]
    base_completed = int(base_history["iterations_run"]) == 500
    base_method = {
        "field": result.base_field,
        "expected": result.base_reconstruction_raw,
        "trajectory": _progressive_trajectory(result, stage="base", steps=500),
        "renderer_calls": result.base_fit_renderer_calls + base_observers,
        "normalized_calls": 0,
        "additive_numerator_calls": result.base_fit_renderer_calls + base_observers,
        "additive_denominator_calls": 0,
        "renderer_calls_coverage_diagnostic": 1,
        "selected_step": result.selected_steps["base"],
        "completed": base_completed,
        "method_status": "completed" if base_completed else "incomplete",
        "history": {"base": base_history},
        "endpoint_parity": result.base_endpoint_parity_max_abs,
        "semantic_family": "additive_rgb_peak_one_v1",
        "renderer": "cuda_additive",
        "fit_seconds": float(base_history["fit_seconds"]),
        "wall_fit_seconds": float(base_history["fit_seconds"]),
        "peak_cuda_allocated_bytes": peak,
        "coverage": _coverage(result.base_field, target, args, torch),
        "attempted_steps": 500,
        "gaussian_row_updates": 640 * 500,
        "base_count": 640,
        "residual_count": 0,
        "diagnostic_renderer_calls": 1,
        "init_seconds": 0.0,
        "initial_field_digest": result.initial_field_digest,
        "base_shared_digest": result.base_endpoint_field_digest,
        "residual_sha256": result.residual_sha256,
        "birth_field_digest": result.birth_field_digest,
        "appended_initial_digest": result.appended_initial_digest,
        "preprojection_endpoint_digest": result.base_endpoint_field_digest,
        "progressive_result": result,
        "audit_initial_field": result.initial_field,
        "audit_training_field": result.base_training_field,
        "stage_kind": "shared_full_target_base",
        "shared_execution_wall_seconds": wall_seconds,
    }
    progressive_method = {
        "field": result.field,
        "expected": result.reconstruction_raw,
        "trajectory": _progressive_trajectory(result, stage=None, steps=700),
        "renderer_calls": result.fit_renderer_calls + base_observers + joint_observers,
        "normalized_calls": 0,
        "additive_numerator_calls": (
            result.fit_renderer_calls + base_observers + joint_observers
        ),
        "additive_denominator_calls": 0,
        "renderer_calls_coverage_diagnostic": 1,
        "selected_step": result.selected_steps["joint"],
        "completed": result.completed,
        "method_status": result.status,
        "history": {
            "stages": result.stage_histories,
            "trajectory": result.trajectory_records(),
            "selected_steps": result.selected_steps,
        },
        "endpoint_parity": max(
            result.base_endpoint_parity_max_abs,
            result.append_parity_max_abs,
            result.endpoint_parity_max_abs,
        ),
        "semantic_family": "additive_rgb_peak_one_v1",
        "renderer": "cuda_additive",
        "fit_seconds": result.fit_seconds,
        "wall_fit_seconds": wall_seconds,
        "peak_cuda_allocated_bytes": peak,
        "coverage": _coverage(result.field, target, args, torch),
        "attempted_steps": result.attempted_steps,
        "gaussian_row_updates": result.gaussian_row_updates,
        "base_count": result.base_count,
        "residual_count": result.residual_count,
        "diagnostic_renderer_calls": result.diagnostic_renderer_calls,
        "init_seconds": 0.0,
        "initial_field_digest": result.initial_field_digest,
        "base_shared_digest": result.base_endpoint_field_digest,
        "residual_sha256": result.residual_sha256,
        "birth_field_digest": result.birth_field_digest,
        "appended_initial_digest": result.appended_initial_digest,
        "preprojection_endpoint_digest": result.endpoint_field_digest,
        "progressive_result": result,
        "audit_initial_field": result.initial_field,
        "audit_training_field": result.appended_initial_field,
        "stage_kind": "progressive_residual_joint",
        "shared_execution_wall_seconds": wall_seconds,
    }
    return result, base_method, progressive_method


def _projection_record(method: dict[str, object]) -> dict[str, object]:
    projection = method["projection_result"]
    if projection is None:
        return {
            "selected_iteration": None,
            "initial_sse": None,
            "final_sse": None,
            "forward_applications": 0,
            "transpose_applications": 0,
            "relative_normal_residual_max": None,
            "adjoint_relative_error": None,
            "initial_operator_parity_max_abs": None,
            "maintained_render_parity_max_abs": None,
            "normal_diagonal_min": None,
            "normal_diagonal_max": None,
            "geometry_exact": True,
            "checkpoints": [],
        }
    receipt = projection.projection
    return {
        "selected_iteration": receipt.selected_iteration,
        "initial_sse": receipt.initial_sse,
        "final_sse": receipt.final_sse,
        "forward_applications": receipt.forward_applications,
        "transpose_applications": receipt.transpose_applications,
        "relative_normal_residual_max": receipt.relative_normal_residual_max,
        "adjoint_relative_error": receipt.adjoint_relative_error,
        "initial_operator_parity_max_abs": receipt.initial_operator_parity_max_abs,
        "maintained_render_parity_max_abs": receipt.maintained_render_parity_max_abs,
        "normal_diagonal_min": receipt.normal_diagonal_min,
        "normal_diagonal_max": receipt.normal_diagonal_max,
        "geometry_exact": projection.geometry_exact,
        "checkpoints": receipt.checkpoint_records(),
    }


def _save_field(path: Path, field: GaussianField) -> dict[str, object]:
    field.save(str(path))
    with np.load(path) as payload:
        keys = sorted(payload.files)
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": h22.report_utils._sha256(path),
        "keys": keys,
        "field_digest": h24._field_digest(field),
        "count": field.n,
    }


def _save_shared_audit(
    output_root: Path,
    image_stem: str,
    seed: int,
    progressive: ProgressiveAdditiveCapacityResult,
    cold896: dict[str, object],
    cold960: dict[str, object],
) -> dict[str, object]:
    directory = output_root / "shared" / f"{image_stem}__s{seed}"
    directory.mkdir(parents=True, exist_ok=True)
    fields = {
        "base_initial": progressive.initial_field,
        "base_training": progressive.base_training_field,
        "base_endpoint": progressive.base_field,
        "residual_birth": progressive.birth_field,
        "appended_initial": progressive.appended_initial_field,
        "progressive_endpoint": progressive.field,
        "cold896_initial": cold896["audit_initial_field"],
        "cold896_training": cold896["audit_training_field"],
        "cold896_endpoint": cold896["field"],
        "cold960_initial": cold960["audit_initial_field"],
        "cold960_training": cold960["audit_training_field"],
        "cold960_endpoint": cold960["field"],
    }
    records = {}
    for name, field in fields.items():
        record = _save_field(directory / f"{name}.field.gaussian.npz", field)
        record["path"] = str(Path(record["path"]).relative_to(output_root))
        records[name] = record
    receipt = {
        "schema": REPORT_SCHEMA,
        "image": image_stem,
        "seed": seed,
        "fields": records,
        "base_shared_digest": progressive.base_endpoint_field_digest,
        "residual_sha256": progressive.residual_sha256,
        "birth_field_digest": progressive.birth_field_digest,
        "appended_initial_digest": progressive.appended_initial_digest,
        "progressive_endpoint_digest": progressive.endpoint_field_digest,
        "counts": {
            "base": progressive.base_count,
            "residual": progressive.residual_count,
            "progressive": progressive.total_count,
            "cold896": int(cold896["field"].n),
            "cold960": int(cold960["field"].n),
        },
        "steps": {"base": 500, "joint": 200, "cold": 500},
        "gaussian_row_updates": {
            "base": 640 * 500,
            "progressive": progressive.gaussian_row_updates,
            "cold896": int(cold896["gaussian_row_updates"]),
            "cold960": int(cold960["gaussian_row_updates"]),
        },
    }
    _write_json(directory / "receipt.json", receipt)
    return {
        "dir": str(directory.relative_to(output_root)),
        "receipt_path": str((directory / "receipt.json").relative_to(output_root)),
        "receipt_sha256": h22.report_utils._sha256(directory / "receipt.json"),
        "fields": records,
    }


def _write_cell(
    *,
    output_root: Path,
    image_path: Path,
    target: np.ndarray,
    raster: dict[str, object],
    seed: int,
    arm: str,
    method: dict[str, object],
    shared_audit: dict[str, object],
    args: argparse.Namespace,
    torch,
) -> dict[str, object]:
    count = COUNT_BY_ARM[arm]
    artifact_dir = output_root / "artifacts" / f"{image_path.stem}__s{seed}__n{count}__{arm}"
    artifact_dir.mkdir(parents=True, exist_ok=False)
    field: GaussianField = method["field"]
    field_path = artifact_dir / "field.gaussian.npz"
    field_record = _save_field(field_path, field)
    field_keys = field_record["keys"]
    incoming_path = artifact_dir / "incoming.field.gaussian.npz"
    proposal_path = artifact_dir / "proposal.field.gaussian.npz"
    method["incoming_field"].save(str(incoming_path))
    method["proposal_field"].save(str(proposal_path))
    projection = _projection_record(method)
    _write_json(artifact_dir / "projection_history.json", projection)

    decode_started = time.perf_counter()
    cold_field = GaussianField.load(str(field_path), device=args.device)
    decode_seconds = time.perf_counter() - decode_started
    render_started = time.perf_counter()
    with torch.no_grad():
        cold_tensor = h22._field_render(
            cold_field,
            target.shape[0],
            target.shape[1],
            method["renderer"],
            args.render_chunk,
        )
        repeated_tensor = h22._field_render(
            cold_field,
            target.shape[0],
            target.shape[1],
            method["renderer"],
            args.render_chunk,
        )
    torch.cuda.synchronize()
    render_seconds = time.perf_counter() - render_started
    cold = cold_tensor.detach().cpu().numpy().astype(np.float32, copy=False)
    repeated = repeated_tensor.detach().cpu().numpy().astype(np.float32, copy=False)
    expected = np.asarray(method["expected"], dtype=np.float32)
    metric_started = time.perf_counter()
    metrics = h22.report_utils._metric_values(
        cold,
        target,
        np.ones(target.shape[:2], dtype=bool),
        device=args.device,
        compute_lpips=args.lpips,
    )
    if metrics["lpips"] is None:
        raise RuntimeError(f"LPIPS is required but unavailable: {metrics['lpips_error']}")
    metric_seconds = time.perf_counter() - metric_started
    bounds = h22._save_visuals(artifact_dir, target, cold, args.error_scale)
    trajectory = method["trajectory"]
    h22._write_curve(
        artifact_dir / "learning_curve.svg", trajectory, f"{image_path.stem} {arm}"
    )
    _write_json(artifact_dir / "fit_history.json", method["history"])
    progressive: ProgressiveAdditiveCapacityResult | None = method.get(
        "progressive_result"
    )
    geometry_record = (
        {
            "base_count": progressive.base_count,
            "residual_count": progressive.residual_count,
            "total_count": progressive.total_count,
            "base_endpoint_unchanged": progressive.base_endpoint_unchanged,
            "joint_training_mask_absent": progressive.joint_training_mask_absent,
            "training_payload_removed": progressive.training_payload_removed,
            "base_endpoint_parity_max_abs": progressive.base_endpoint_parity_max_abs,
            "append_parity_max_abs": progressive.append_parity_max_abs,
            "endpoint_parity_max_abs": progressive.endpoint_parity_max_abs,
            "negative_birth_coefficients": progressive.negative_birth_coefficients,
            "initial_field_digest": progressive.initial_field_digest,
            "base_training_field_digest": progressive.base_training_field_digest,
            "base_endpoint_field_digest": progressive.base_endpoint_field_digest,
            "residual_sha256": progressive.residual_sha256,
            "birth_field_digest": progressive.birth_field_digest,
            "appended_initial_digest": progressive.appended_initial_digest,
            "endpoint_field_digest": progressive.endpoint_field_digest,
        }
        if progressive is not None
        else {}
    )
    _write_json(artifact_dir / "geometry_history.json", geometry_record)
    fit_config: dict[str, object] | None
    if arm == "normalized_plain_n640":
        fit_config = asdict(h23._fit_config(args, "normalized_plain"))
    elif arm.startswith("cold_additive"):
        fit_config = asdict(_cold_fit_config(args, count))
    else:
        fit_config = asdict(_progressive_config(args))
    _write_json(
        artifact_dir / "config.json",
        {
            "schema": REPORT_SCHEMA,
            "arm": arm,
            "seed": seed,
            "count": count,
            "init": asdict(h22._init_config(count if arm.startswith("cold") else 640, seed)),
            "fit": fit_config,
            "projection": (
                asdict(h24._projection_config(args)) if arm in PROJECTED_ARMS else None
            ),
            "safety": asdict(h24._safety_config()) if arm in PROJECTED_ARMS else None,
            "shared_audit_receipt": shared_audit["receipt_path"],
        },
    )
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
    cold_parity = float(
        np.max(np.abs(cold.astype(np.float64) - expected.astype(np.float64)))
    )
    repeated_parity = float(
        np.max(np.abs(repeated.astype(np.float64) - cold.astype(np.float64)))
    )
    incoming_metrics = method["incoming_selection_metrics"]
    proposal_metrics = method["proposal_selection_metrics"]
    total_seconds = (
        float(method["init_seconds"])
        + float(method["wall_fit_seconds"])
        + float(method["projection_seconds"])
        + float(method["projection_metric_seconds"])
        + decode_seconds
        + render_seconds
        + metric_seconds
    )
    pure = arm in PURE_ADDITIVE_ARMS
    row: dict[str, object] = {
        "schema": REPORT_SCHEMA,
        "status": "diagnostic",
        "phase": "untouched_confirmation",
        "image": image_path.stem,
        "arm": arm,
        "seed": seed,
        "semantic_family": method["semantic_family"],
        "renderer": method["renderer"],
        "artifact_dir": str(artifact_dir.relative_to(output_root)),
        "source_path": str(image_path),
        "source_rank": SELECTION_ORDER.index(image_path.name) + 1,
        "source_sha256": SOURCE_BINDINGS[image_path.name],
        "source_file_bytes": image_path.stat().st_size,
        "selection_salt": SELECTION_SALT,
        "selection_sha256": SELECTION_BINDINGS[image_path.name],
        "archive_sha256": ARCHIVE_SHA256,
        "archive_bytes": ARCHIVE_BYTES,
        "original_width": raster["original_width"],
        "original_height": raster["original_height"],
        "width": target.shape[1],
        "height": target.shape[0],
        "active_pixels": int(target.shape[0] * target.shape[1]),
        "target_gaussians": count,
        "n_gaussians": field.n,
        "count_ratio_vs_normalized_n640": count / 640.0,
        "field_file_sha256": field_record["sha256"],
        "field_file_bytes": field_record["bytes"],
        "field_npz_keys": field_keys,
        "pure_additive_endpoint": pure,
        "four_array_endpoint_exact": not pure or set(field_keys) == FOUR_ARRAY_KEYS,
        "mass_payload_present": any("mass" in key.lower() for key in field_keys),
        "denominator_payload_present": any("denom" in key.lower() for key in field_keys),
        "optimizer_payload_present": any("optimizer" in key.lower() for key in field_keys),
        "auxiliary_rgb_payload_present": any(
            key in field_keys for key in ("color_grads", "opacities")
        ),
        "training_payload_present": pure and set(field_keys) != FOUR_ARRAY_KEYS,
        "method_status": method["method_status"],
        "completed": method["completed"],
        "selected_step": method["selected_step"],
        "selected_lambda": 0.0 if pure else None,
        "attempted_steps": method["attempted_steps"],
        "gaussian_row_updates": method["gaussian_row_updates"],
        "base_count": method["base_count"],
        "residual_count": method["residual_count"],
        "stage_kind": method["stage_kind"],
        "endpoint_internal_parity_max_abs": method["endpoint_parity"],
        "renderer_calls_fit": method["renderer_calls"],
        "normalized_calls_fit": method["normalized_calls"],
        "additive_numerator_calls_fit": method["additive_numerator_calls"],
        "additive_denominator_calls_fit": method["additive_denominator_calls"],
        "renderer_calls_coverage_diagnostic": method[
            "renderer_calls_coverage_diagnostic"
        ],
        "diagnostic_renderer_calls_fit": method["diagnostic_renderer_calls"],
        "projection_applied": method["projection_applied"],
        "projection_selected": method["projection_selected"],
        "projection_reason": method["projection_reason"],
        "projection_clauses": method["projection_clauses"],
        "projection_seconds": method["projection_seconds"],
        "projection_metric_seconds": method["projection_metric_seconds"],
        "projection_selected_iteration": projection["selected_iteration"],
        "projection_initial_sse": projection["initial_sse"],
        "projection_final_sse": projection["final_sse"],
        "projection_forward_applications": projection["forward_applications"],
        "projection_transpose_applications": projection["transpose_applications"],
        "projection_relative_normal_residual_max": projection[
            "relative_normal_residual_max"
        ],
        "projection_adjoint_relative_error": projection["adjoint_relative_error"],
        "projection_initial_operator_parity_max_abs": projection[
            "initial_operator_parity_max_abs"
        ],
        "projection_maintained_render_parity_max_abs": projection[
            "maintained_render_parity_max_abs"
        ],
        "projection_geometry_exact": projection["geometry_exact"],
        "incoming_field_digest": method["incoming_field_digest"],
        "proposal_field_digest": method["proposal_field_digest"],
        "final_field_digest": method["final_field_digest"],
        "incoming_field_file_sha256": h22.report_utils._sha256(incoming_path),
        "proposal_field_file_sha256": h22.report_utils._sha256(proposal_path),
        "initial_field_digest": method["initial_field_digest"],
        "base_shared_digest": method["base_shared_digest"],
        "residual_sha256": method["residual_sha256"],
        "birth_field_digest": method["birth_field_digest"],
        "appended_initial_digest": method["appended_initial_digest"],
        "preprojection_endpoint_digest": method["preprojection_endpoint_digest"],
        "shared_audit_dir": shared_audit["dir"],
        "shared_audit_receipt": shared_audit["receipt_path"],
        "shared_audit_receipt_sha256": shared_audit["receipt_sha256"],
        "init_seconds": method["init_seconds"],
        "fit_seconds": method["fit_seconds"],
        "wall_fit_seconds": method["wall_fit_seconds"],
        "cold_decode_seconds": decode_seconds,
        "render_seconds": render_seconds,
        "metric_seconds": metric_seconds,
        "pipeline_algorithm_seconds": (
            float(method["init_seconds"])
            + float(method["fit_seconds"])
            + float(method["projection_seconds"])
        ),
        "total_seconds": total_seconds,
        "peak_cuda_allocated_bytes": method["peak_cuda_allocated_bytes"],
        "maintained_render_parity_max_abs": cold_parity,
        "repeated_render_parity_max_abs": repeated_parity,
        "finite_reconstruction": bool(np.isfinite(cold).all()),
        "psnr_auc_attempted_step": h22._psnr_auc(
            trajectory, int(method["attempted_steps"])
        ),
        "raw_mse": metrics["masked_mse"],
        **h22._display_metrics(cold, target),
        **h22._coefficient_record(field),
        **method["coverage"],
        **metrics,
    }
    for prefix, values in (
        ("incoming", incoming_metrics),
        ("proposal", proposal_metrics),
    ):
        for key in ("raw_mse", "ms_ssim", "lpips", "pixel_max", "patch7_max"):
            row[f"{prefix}_{key}"] = None if values is None else values[key]
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


def _integrity(rows: list[dict[str, object]], arm: str) -> bool:
    expected_count = COUNT_BY_ARM[arm]
    return bool(
        len(rows) == len(SOURCE_BINDINGS) * 2
        and all(
            row["completed"]
            and row["method_status"] == "completed"
            and row["n_gaussians"] == row["target_gaussians"] == expected_count
            and row["finite_reconstruction"]
            and float(row["coefficient_abs_max"]) <= COEFFICIENT_LIMIT
            and float(row["maintained_render_parity_max_abs"]) <= PARITY_LIMIT
            and float(row["repeated_render_parity_max_abs"]) <= PARITY_LIMIT
            and float(row["endpoint_internal_parity_max_abs"]) <= PARITY_LIMIT
            and (
                arm not in PURE_ADDITIVE_ARMS
                or (
                    row["selected_lambda"] == 0.0
                    and row["semantic_family"] == "additive_rgb_peak_one_v1"
                    and row["renderer"] == "cuda_additive"
                    and row["four_array_endpoint_exact"]
                    and not row["mass_payload_present"]
                    and not row["denominator_payload_present"]
                    and not row["optimizer_payload_present"]
                    and not row["auxiliary_rgb_payload_present"]
                    and not row["training_payload_present"]
                )
            )
            for row in rows
        )
    )


def _quality_gate(
    candidate: list[dict[str, object]], normalized: list[dict[str, object]]
) -> dict[str, object]:
    key = lambda row: (row["image"], row["seed"])
    normalized_by_key = {key(row): row for row in normalized}
    clauses = {
        "mean_psnr_at_least_normalized": (
            _mean(candidate, "psnr_db") >= _mean(normalized, "psnr_db")
        ),
        "all_psnr_within_normalized_minus_0p10_db": all(
            float(row["psnr_db"])
            >= float(normalized_by_key[key(row)]["psnr_db"]) - 0.10
            for row in candidate
        ),
        "mean_ms_ssim_within_normalized_minus_1e_4": (
            _mean(candidate, "ms_ssim") >= _mean(normalized, "ms_ssim") - 1e-4
        ),
        "mean_lpips_within_normalized_plus_0p002": (
            _mean(candidate, "lpips") <= _mean(normalized, "lpips") + 0.002
        ),
        "all_lpips_within_normalized_plus_0p01": all(
            float(row["lpips"])
            <= float(normalized_by_key[key(row)]["lpips"]) + 0.01
            for row in candidate
        ),
        "mean_pixel_max_within_normalized_plus_0p005": (
            _mean(candidate, "artifact_pixel_rmse_max")
            <= _mean(normalized, "artifact_pixel_rmse_max") + 0.005
        ),
        "mean_patch7_max_within_normalized_plus_0p005": (
            _mean(candidate, "artifact_patch_rmse_max_7")
            <= _mean(normalized, "artifact_patch_rmse_max_7") + 0.005
        ),
        "all_local_max_within_normalized_plus_0p02": all(
            float(row["artifact_pixel_rmse_max"])
            <= float(normalized_by_key[key(row)]["artifact_pixel_rmse_max"]) + 0.02
            and float(row["artifact_patch_rmse_max_7"])
            <= float(normalized_by_key[key(row)]["artifact_patch_rmse_max_7"]) + 0.02
            for row in candidate
        ),
    }
    paired_psnr_deltas = [
        float(row["psnr_db"]) - float(normalized_by_key[key(row)]["psnr_db"])
        for row in candidate
    ]
    return {
        "clauses": clauses,
        "numeric_pass": all(clauses.values()),
        "mean_psnr_delta_db": float(np.mean(paired_psnr_deltas)),
        "minimum_psnr_delta_db": float(np.min(paired_psnr_deltas)),
        "maximum_psnr_delta_db": float(np.max(paired_psnr_deltas)),
    }


def _decision(rows: list[dict[str, object]]) -> dict[str, object]:
    expected_count = len(SOURCE_BINDINGS) * 2
    by_arm = {arm: [row for row in rows if row["arm"] == arm] for arm in ARMS}
    complete = all(len(by_arm[arm]) == expected_count for arm in ARMS)
    aggregates = {
        arm: {
            "cell_count": len(by_arm[arm]),
            "count": COUNT_BY_ARM[arm],
            "mean_psnr_db": _mean(by_arm[arm], "psnr_db") if by_arm[arm] else None,
            "mean_ms_ssim": _mean(by_arm[arm], "ms_ssim") if by_arm[arm] else None,
            "mean_lpips": _mean(by_arm[arm], "lpips") if by_arm[arm] else None,
            "mean_pixel_max": _mean(by_arm[arm], "artifact_pixel_rmse_max")
            if by_arm[arm]
            else None,
            "mean_patch7_max": _mean(by_arm[arm], "artifact_patch_rmse_max_7")
            if by_arm[arm]
            else None,
            "mean_fit_seconds": _mean(by_arm[arm], "fit_seconds")
            if by_arm[arm]
            else None,
            "mean_gaussian_row_updates": _mean(by_arm[arm], "gaussian_row_updates")
            if by_arm[arm]
            else None,
            "mean_renderer_calls": _mean(by_arm[arm], "renderer_calls_fit")
            if by_arm[arm]
            else None,
            "projection_selected_count": sum(
                bool(row["projection_selected"]) for row in by_arm[arm]
            ),
        }
        for arm in ARMS
    }
    integrity = {
        arm: _integrity(by_arm[arm], arm) if complete else False for arm in ARMS
    }
    projection_fail_closed = complete and all(
        (
            row["projection_selected"] and all(row["projection_clauses"].values())
        )
        or (
            not row["projection_selected"]
            and row["final_field_digest"] == row["incoming_field_digest"]
        )
        for arm in PROJECTED_ARMS
        for row in by_arm[arm]
    )
    shared_base_exact = complete and all(
        row["base_shared_digest"] is not None
        for arm in (
            "additive_plain_n640",
            "additive_projected_n640",
            "progressive_residual_n896",
            "progressive_residual_projected_n896",
        )
        for row in by_arm[arm]
    )
    if shared_base_exact:
        grouped: dict[tuple[object, object], set[object]] = {}
        for arm in (
            "additive_plain_n640",
            "additive_projected_n640",
            "progressive_residual_n896",
            "progressive_residual_projected_n896",
        ):
            for row in by_arm[arm]:
                grouped.setdefault((row["image"], row["seed"]), set()).add(
                    row["base_shared_digest"]
                )
        shared_base_exact = all(len(values) == 1 for values in grouped.values())
    progressive_accounting = complete and all(
        row["base_count"] == 640
        and row["residual_count"] == 256
        and row["n_gaussians"] == 896
        and row["attempted_steps"] == 700
        and row["gaussian_row_updates"] == 499_200
        for arm in PROGRESSIVE_ARMS
        for row in by_arm[arm]
    )
    cold_accounting = complete and all(
        row["attempted_steps"] == 500
        and row["gaussian_row_updates"] == COUNT_BY_ARM[arm] * 500
        for arm in ("cold_additive_projected_n896", "cold_additive_projected_n960")
        for row in by_arm[arm]
    )

    quality: dict[str, object] = {}
    if complete:
        normalized = by_arm["normalized_plain_n640"]
        for arm in (
            "additive_plain_n640",
            "additive_projected_n640",
            "cold_additive_projected_n896",
            "progressive_residual_projected_n896",
            "cold_additive_projected_n960",
        ):
            quality[arm] = _quality_gate(by_arm[arm], normalized)
            quality[arm]["integrity_pass"] = integrity[arm]
            quality[arm]["numeric_quality_capable"] = bool(
                integrity[arm]
                and projection_fail_closed
                and quality[arm]["numeric_pass"]
            )
    numeric_capable = {
        arm: bool(quality.get(arm, {}).get("numeric_quality_capable", False))
        for arm in quality
    }
    progressive_mechanism_supported = bool(
        numeric_capable.get("progressive_residual_projected_n896", False)
        and aggregates["progressive_residual_projected_n896"]["mean_psnr_db"]
        >= aggregates["cold_additive_projected_n896"]["mean_psnr_db"]
    )
    robust_n960 = bool(
        numeric_capable.get("cold_additive_projected_n960", False)
        and quality["cold_additive_projected_n960"]["mean_psnr_delta_db"] >= 0.10
        and quality["cold_additive_projected_n960"]["minimum_psnr_delta_db"] >= 0.0
    )
    selected_arm = None
    selector_reason = "no pure-additive rung passes the frozen numeric gate"
    if numeric_capable.get("additive_projected_n640", False):
        selected_arm = "additive_projected_n640"
        selector_reason = "same-count projected additive is the smallest passing rung"
    else:
        n896 = [
            arm
            for arm in (
                "cold_additive_projected_n896",
                "progressive_residual_projected_n896",
            )
            if numeric_capable.get(arm, False)
        ]
        if n896:
            if progressive_mechanism_supported:
                selected_arm = "progressive_residual_projected_n896"
                selector_reason = "progressive N=896 passes and is noninferior to cold N=896"
            else:
                selected_arm = max(
                    n896,
                    key=lambda arm: float(aggregates[arm]["mean_psnr_db"]),
                )
                selector_reason = "selected the higher-PSNR passing N=896 capacity rung"
        elif robust_n960:
            selected_arm = "cold_additive_projected_n960"
            selector_reason = "no N=896 rung passes; the frozen robust N=960 rung passes"
    gates = {
        "all_cells_present": complete,
        "all_arm_integrity": complete and all(integrity.values()),
        "projection_transactions_fail_closed": projection_fail_closed,
        "shared_base_digest_exact": shared_base_exact,
        "progressive_accounting_exact": progressive_accounting,
        "cold_accounting_exact": cold_accounting,
    }
    numeric_solution = selected_arm is not None and all(gates.values())
    return {
        "schema": REPORT_SCHEMA,
        "status": "diagnostic",
        "phase": "untouched_confirmation",
        "aggregates": aggregates,
        "integrity": integrity,
        "quality": quality,
        "gates": gates,
        "progressive_mechanism_supported_numeric": progressive_mechanism_supported,
        "robust_n960_numeric": robust_n960,
        "same_count_additive_better_numeric": numeric_capable.get(
            "additive_plain_n640", False
        ),
        "normalization_not_required_for_fidelity_numeric": numeric_solution,
        "numeric_selected_arm": selected_arm,
        "selector_reason": selector_reason,
        "numeric_pass": numeric_solution,
        "visual_review": "pending_native_audit",
        "overall_pass": False,
        "formal_claim_ready": False,
        "interpretation": (
            "A pure-additive capacity rung passes numerically; native visual audit is required."
            if numeric_solution
            else "No frozen pure-additive capacity rung passes; retain without tuning."
        ),
    }


def _write_report(
    output_root: Path, rows: list[dict[str, object]], decision: dict[str, object]
) -> None:
    table_rows = []
    cards = []
    for row in rows:
        artifact = escape(str(row["artifact_dir"]))
        table_rows.append(
            "<tr>"
            f"<td>{escape(str(row['image']))}</td><td>{int(row['seed'])}</td>"
            f"<td>{escape(str(row['arm']))}</td><td>{int(row['n_gaussians'])}</td>"
            f"<td>{float(row['psnr_db']):.3f}</td>"
            f"<td>{float(row['ms_ssim']):.5f}</td><td>{float(row['lpips']):.5f}</td>"
            f"<td>{float(row['artifact_pixel_rmse_max']):.4f}</td>"
            f"<td>{float(row['artifact_patch_rmse_max_7']):.4f}</td>"
            f"<td>{'yes' if row['projection_selected'] else 'no'}</td>"
            f"<td><a href='{artifact}/source.png'>source</a> · "
            f"<a href='{artifact}/reconstruction.png'>full</a> · "
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
<title>HIER-026 pure-additive capacity parity</title><style>
body{{font-family:system-ui;margin:2rem;max-width:1900px}}table{{border-collapse:collapse}}
th,td{{border:1px solid #bbb;padding:.35rem}}img{{max-width:300px;max-height:240px;margin:.25rem}}
section{{border-top:1px solid #aaa;margin-top:1rem}}pre,code{{white-space:pre-wrap}}
</style></head><body><h1>HIER-026 progressive pure-additive capacity parity</h1>
<p><strong>Untouched-data producer confirmation.</strong> The protocol and source-name binding
preceded pixel decode, but dirty source and producer review keep this provisional.</p>
<p><code>{escape(_command())}</code></p>
<p><a href="config.json">config</a> · <a href="decision.json">decision</a> ·
<a href="metrics.json">JSON</a> · <a href="metrics.jsonl">JSONL</a> ·
<a href="metrics.csv">CSV</a> · <a href="attempts.json">attempts</a> ·
<a href="manifest.json">manifest</a></p>
<h2>Decision</h2><pre>{escape(json.dumps(decision, indent=2, sort_keys=True))}</pre>
<h2>Cells</h2><table><tr><th>image</th><th>seed</th><th>arm</th><th>N</th><th>PSNR</th>
<th>MS-SSIM</th><th>LPIPS</th><th>pixel max</th><th>7x7 max</th><th>projection</th>
<th>artifacts</th></tr>{''.join(table_rows)}</table>
<h2>Native visual audit</h2>{''.join(cards)}</body></html>"""
    (output_root / "index.html").write_text(html, encoding="utf-8")


def _write_manifest(output_root: Path) -> None:
    files = []
    for path in sorted(output_root.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            files.append(
                {
                    "path": str(path.relative_to(output_root)),
                    "bytes": path.stat().st_size,
                    "sha256": h22.report_utils._sha256(path),
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
        raise SystemExit(f"completed HIER-026 bundle is immutable: {args.out}")
    if args.out.exists() and any(args.out.iterdir()) and not args.resume:
        raise SystemExit(f"output directory is non-empty; use --resume: {args.out}")
    args.out.mkdir(parents=True, exist_ok=True)

    import torch

    if not torch.cuda.is_available():
        raise SystemExit("frozen HIER-026 protocol requires CUDA")
    sources = _discover_sources(args.images)
    _write_json(args.out / "environment.json", h22._environment(torch))
    snapshots = _snapshot_sources(args.out)
    _write_json(
        args.out / "config.json",
        {
            "schema": REPORT_SCHEMA,
            "status": "diagnostic",
            "command": _command(),
            "git": h22._git_record(),
            "source_snapshots": snapshots,
            "archive": {
                "url": "https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip",
                "bytes": ARCHIVE_BYTES,
                "sha256": ARCHIVE_SHA256,
            },
            "source_selection": {
                "salt": SELECTION_SALT,
                "order": list(SELECTION_ORDER),
                "selection_bindings": SELECTION_BINDINGS,
                "source_bindings": SOURCE_BINDINGS,
                "decoded_before_protocol_freeze": False,
                "inherited_from_sealed_hier025_phase_c": True,
            },
            "arguments": vars(args),
            "arms": list(ARMS),
            "counts": COUNT_BY_ARM,
            "structure_tensor": asdict(StructureTensorConfig()),
            "progressive": asdict(_progressive_config(args)),
            "cold_n896_fit": asdict(_cold_fit_config(args, 896)),
            "cold_n960_fit": asdict(_cold_fit_config(args, 960)),
            "projection": asdict(h24._projection_config(args)),
            "safety": asdict(h24._safety_config()),
            "shared_fit_reuse": (
                "additive N=640 and both progressive arms share the exact base execution; "
                "projected/unprojected progressive share one pre-projection endpoint"
            ),
            "claim_limits": [
                "max-side-160 only",
                "count/work exchange, not same-count or equal-byte superiority",
                "dirty-source producer confirmation",
                "no codec, production, default, or novelty claim",
            ],
        },
    )
    with (args.out / "git.diff").open("wb") as handle:
        subprocess.run(
            ["git", "diff", "--binary", "HEAD"],
            cwd=ROOT,
            check=False,
            stdout=handle,
        )
    (args.out / "NATURAL_STARTED").write_text(
        "HIER-026 untouched source pixels decoded; no in-place tuning or replay.\n",
        encoding="utf-8",
    )

    rows: list[dict[str, object]] = []
    attempts: list[dict[str, object]] = []
    metrics_path = args.out / "metrics.json"
    attempts_path = args.out / "attempts.json"
    if args.resume and metrics_path.is_file():
        rows = json.loads(metrics_path.read_text(encoding="utf-8")).get("rows", [])
    if args.resume and attempts_path.is_file():
        attempts = json.loads(attempts_path.read_text(encoding="utf-8")).get(
            "attempts", []
        )
    row_keys = {(row["image"], row["seed"], row["arm"]) for row in rows}
    for image_path in sources:
        target, mask, raster = h22.report_utils._load_evaluation_raster(
            image_path, None, max_side=args.max_side, mask_threshold=0.5
        )
        if mask is not None:
            raise RuntimeError("HIER-026 requires an unmasked full-frame source")
        for seed in args.seeds:
            expected_keys = {(image_path.stem, seed, arm) for arm in ARMS}
            if expected_keys <= row_keys:
                continue
            methods: dict[str, dict[str, object]] = {}
            shared_audit = None
            fit_error = None
            try:
                progressive, additive_base, progressive_method = _run_progressive(
                    target, seed, args, torch
                )
                np.random.seed(seed)
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                normalized = h23._run_method(
                    progressive.initial_field,
                    target,
                    "normalized_plain",
                    args,
                    torch,
                )
                normalized.update(
                    {
                        "attempted_steps": 500,
                        "gaussian_row_updates": 640 * 500,
                        "base_count": 0,
                        "residual_count": 640,
                        "diagnostic_renderer_calls": 0,
                        "init_seconds": 0.0,
                        "initial_field_digest": progressive.initial_field_digest,
                        "base_shared_digest": None,
                        "residual_sha256": None,
                        "birth_field_digest": None,
                        "appended_initial_digest": None,
                        "preprojection_endpoint_digest": h24._field_digest(
                            normalized["field"]
                        ),
                        "progressive_result": None,
                        "audit_initial_field": progressive.initial_field,
                        "audit_training_field": normalized["field"].detached(),
                        "stage_kind": "normalized_full_target",
                    }
                )
                cold896 = _run_cold_additive(target, seed, 896, args, torch)
                cold960 = _run_cold_additive(target, seed, 960, args, torch)
                shared_audit = _save_shared_audit(
                    args.out,
                    image_path.stem,
                    seed,
                    progressive,
                    cold896,
                    cold960,
                )
                methods = {
                    "normalized_plain_n640": h24._base_method(normalized),
                    "additive_plain_n640": h24._base_method(additive_base),
                    "additive_projected_n640": h24._project_method(
                        additive_base, target, args
                    ),
                    "cold_additive_projected_n896": h24._project_method(
                        cold896, target, args
                    ),
                    "progressive_residual_n896": h24._base_method(progressive_method),
                    "progressive_residual_projected_n896": h24._project_method(
                        progressive_method, target, args
                    ),
                    "cold_additive_projected_n960": h24._project_method(
                        cold960, target, args
                    ),
                }
            except Exception as exc:
                fit_error = exc
            for arm in ARMS:
                stable_key = (image_path.stem, seed, arm)
                if stable_key in row_keys:
                    continue
                cell_started = time.perf_counter()
                try:
                    if fit_error is not None:
                        raise RuntimeError(f"paired execution failed: {fit_error}")
                    if shared_audit is None:
                        raise RuntimeError("shared audit receipt was not created")
                    row = _write_cell(
                        output_root=args.out,
                        image_path=image_path,
                        target=target,
                        raster=raster,
                        seed=seed,
                        arm=arm,
                        method=methods[arm],
                        shared_audit=shared_audit,
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
                        attempts_path,
                        {
                            "schema": REPORT_SCHEMA,
                            "status": "diagnostic",
                            "attempts": attempts,
                        },
                    )
                    torch.cuda.empty_cache()

    decision = _decision(rows)
    _write_json(args.out / "decision.json", decision)
    _write_report(args.out, rows, decision)
    (args.out / "COMPLETED").write_text(
        "HIER-026 untouched producer confirmation complete; do not overwrite.\n",
        encoding="utf-8",
    )
    _write_manifest(args.out)
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
