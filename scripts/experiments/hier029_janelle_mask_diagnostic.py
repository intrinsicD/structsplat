#!/usr/bin/env python3
"""Run HIER-029's full-resolution Janelle mask-factorial diagnostic."""

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
from scripts.experiments import hier024_gauge_geometry_projection as h24  # noqa: E402
from scripts.experiments import hier026_progressive_additive_capacity as h26  # noqa: E402
from structsplat.cli import save_error_heatmap, save_image  # noqa: E402
from structsplat.config import FitConfig, StructureTensorConfig  # noqa: E402
from structsplat.endpoint_appearance_projection import (  # noqa: E402
    project_additive_endpoint,
    select_safe_projection,
)
from structsplat.fit import fit  # noqa: E402
from structsplat.gaussians import GaussianField  # noqa: E402
from structsplat.init import build_field, build_masked_field  # noqa: E402
from structsplat.residual_pursuit_additive import (  # noqa: E402
    ResidualPursuitAdditiveConfig,
    append_residual_pursuit_gaussians,
)


REPORT_SCHEMA = "structsplat.hier029_janelle_mask_diagnostic.diagnostic.v1"
SOURCE_SHA256 = "ae24fe99d3f8edbd04cd2c85ebc4fe9bfd95abe878c22abb7691cadcfc5c411b"
MASK_SHA256 = "94dcbf7005dbeb1d183e259a569d783aa5df900255e763385bed91f02d3b80c3"
NATIVE_SHAPE = (4608, 5328)
EVALUATION_SHAPE = (1038, 1200)
MODES = ("full_frame", "masked_foreground")
ARMS = (
    "normalized_plain_n640",
    "cold_additive_projected_n960",
    "residual_pursuit_additive_n1024",
    "cold_additive_projected_n1024",
)
PURE_ADDITIVE_ARMS = frozenset(set(ARMS) - {"normalized_plain_n640"})
COUNT_BY_ARM = {
    "normalized_plain_n640": 640,
    "cold_additive_projected_n960": 960,
    "residual_pursuit_additive_n1024": 1024,
    "cold_additive_projected_n1024": 1024,
}
GAUSSIAN_ROW_UPDATES_BY_ARM = {
    "normalized_plain_n640": 640 * 500,
    "cold_additive_projected_n960": 960 * 500,
    "residual_pursuit_additive_n1024": 960 * 500,
    "cold_additive_projected_n1024": 1024 * 500,
}
FOUR_ARRAY_KEYS = frozenset(("means", "log_scales", "rotations", "colors"))
COEFFICIENT_LIMIT = 16.0
PARITY_LIMIT = 2e-5


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("mask", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--max-side", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--lpips", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--render-chunk", type=int, default=256)
    parser.add_argument("--error-scale", type=float, default=4.0)
    parser.add_argument("--resume", action="store_true")
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    frozen = {
        "max_side": 1200,
        "seed": 0,
        "device": "cuda",
        "lpips": True,
        "render_chunk": 256,
    }
    for name, expected in frozen.items():
        if getattr(args, name) != expected:
            raise SystemExit(
                f"frozen HIER-029 diagnostic requires {name}={expected!r}, "
                f"got {getattr(args, name)!r}"
            )
    if not math.isfinite(args.error_scale) or args.error_scale <= 0.0:
        raise SystemExit("--error-scale must be finite and positive")
    for name in ("image", "mask"):
        if not getattr(args, name).is_file():
            raise SystemExit(f"{name} does not exist: {getattr(args, name)}")
    args.iters = 500
    args.budgets = [640]


def _command() -> str:
    return " ".join(shlex.quote(value) for value in sys.argv)


def _write_json(path: Path, value: object) -> None:
    h22._write_json(path, value)


def _snapshot_sources(output_root: Path) -> list[dict[str, object]]:
    paths = (
        Path(__file__).resolve(),
        ROOT / "scripts" / "experiments" / "hier028_residual_pursuit_additive.py",
        ROOT / "src" / "structsplat" / "residual_pursuit_additive.py",
        ROOT / "src" / "structsplat" / "endpoint_appearance_projection.py",
        ROOT / "src" / "structsplat" / "contraction_refinement.py",
        ROOT / "tests" / "test_residual_pursuit_additive.py",
        ROOT / "tests" / "test_endpoint_appearance_projection.py",
        ROOT / "tests" / "test_hier029_janelle_mask_diagnostic.py",
        ROOT / "tasks" / "HIER-029-janelle-full-resolution-mask-diagnostic.md",
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


def _fit_config(args: argparse.Namespace, renderer: str, count: int, *, masked: bool) -> FitConfig:
    return FitConfig(
        iters=500,
        lr_means=5e-2,
        lr_scales=3e-2,
        lr_rot=1e-2,
        lr_color=3e-2,
        optimizer="adam",
        pixel_loss="l1",
        loss_weighting="mask" if masked else "none",
        ssim_weight=0.3,
        log_every=25,
        checkpoint_policy="best_psnr_final_count",
        sigma_cutoff=3.0,
        support_fade=False,
        aa_dilation=0.0,
        render_chunk=args.render_chunk,
        renderer=renderer,
        color_basis="constant",
        compute_lpips=False,
        max_gaussians=count,
    )


def _tail_config(args: argparse.Namespace) -> ResidualPursuitAdditiveConfig:
    return ResidualPursuitAdditiveConfig(
        tail_gaussians=64,
        scale_px=0.35,
        coefficient_abs_limit=COEFFICIENT_LIMIT,
        sigma_cutoff=3.0,
        render_chunk=args.render_chunk,
        renderer="cuda_additive",
    )


def _objective(source: np.ndarray, mask: np.ndarray, mode: str) -> np.ndarray:
    if mode == "full_frame":
        return np.array(source, dtype=np.float32, order="C", copy=True)
    if mode == "masked_foreground":
        return np.ascontiguousarray(source * mask[:, :, None])
    raise ValueError(f"unknown objective mode {mode!r}")


def _initial_field(
    source: np.ndarray,
    mask: np.ndarray,
    mode: str,
    count: int,
    seed: int,
    args: argparse.Namespace,
) -> GaussianField:
    config = h22._init_config(count, seed)
    tensor_config = StructureTensorConfig()
    if mode == "masked_foreground":
        return build_masked_field(
            _objective(source, mask, mode),
            mask,
            config,
            tensor_config,
            device=args.device,
            contain=False,
        )
    return build_field(source, config, tensor_config, device=args.device)


def _run_fit(
    source: np.ndarray,
    mask: np.ndarray,
    mode: str,
    count: int,
    renderer: str,
    args: argparse.Namespace,
    torch,
) -> dict[str, object]:
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    objective = _objective(source, mask, mode)
    init_started = time.perf_counter()
    initial = _initial_field(source, mask, mode, count, args.seed, args)
    init_seconds = time.perf_counter() - init_started
    initial_audit = initial.detached()
    target_tensor = torch.as_tensor(objective, device=args.device, dtype=torch.float32)
    # ``fit`` owns construction of its device-side mask constraint and therefore accepts the
    # canonical NumPy mask here. Passing an already-CUDA tensor would cross the NumPy geometry
    # boundary inside ``MaskGeometry.build``.
    fit_mask = mask if mode == "masked_foreground" else None
    config = _fit_config(args, renderer, count, masked=mode == "masked_foreground")
    torch.cuda.reset_peak_memory_stats()
    wall_started = time.perf_counter()
    print(f"[{mode}] fit {renderer} N={count} / 500", flush=True)
    result = fit(
        initial,
        target_tensor,
        config,
        mask=fit_mask,
        verbose=False,
    )
    training_field = result["field"].detached()
    endpoint = h26._pure_endpoint(training_field) if renderer == "cuda_additive" else training_field
    with torch.no_grad():
        rendered = h22._field_render(
            endpoint,
            source.shape[0],
            source.shape[1],
            renderer,
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
    with torch.no_grad():
        coverage = h26._coverage(endpoint, source, args, torch)
    return {
        "field": endpoint,
        "expected": expected,
        "trajectory": h22._trajectory_baseline(result, 500),
        "renderer_calls": renderer_calls,
        "normalized_calls": renderer_calls if renderer == "cuda" else 0,
        "additive_numerator_calls": renderer_calls if renderer == "cuda_additive" else 0,
        "additive_denominator_calls": 0,
        "renderer_calls_coverage_diagnostic": 1,
        "selected_step": int(result["selected_iter"]),
        "completed": int(result["iterations_run"]) == 500,
        "method_status": ("completed" if int(result["iterations_run"]) == 500 else "incomplete"),
        "history": {
            "history": result["history"],
            "checkpoint_history": result["checkpoint_history"],
        },
        "hold_psnr_db": None,
        "optimizer_reset_count": 0,
        "optimizer_reset_step": None,
        "hold_optimizer_reset_count": 0,
        "endpoint_parity": endpoint_parity,
        "semantic_family": (
            "normalized_weighted_sum_v1" if renderer == "cuda" else "additive_rgb_peak_one_v1"
        ),
        "renderer": renderer,
        "fit_seconds": float(result["fit_seconds"]),
        "wall_fit_seconds": wall_seconds,
        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "coverage": coverage,
        "attempted_steps": 500,
        "gaussian_row_updates": count * 500,
        "diagnostic_renderer_calls": 1,
        "init_seconds": init_seconds,
        "initial_field_digest": h24._field_digest(initial_audit),
        "preprojection_endpoint_digest": h24._field_digest(endpoint),
        "audit_initial_field": initial_audit,
        "audit_training_field": training_field,
        "fit_config": config,
        "objective_target": objective,
    }


def _foreground_bounds(mask: np.ndarray, padding: int = 16) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    if not len(xs):
        raise ValueError("foreground mask is empty")
    return (
        max(0, int(xs.min()) - padding),
        max(0, int(ys.min()) - padding),
        min(mask.shape[1], int(xs.max()) + padding + 1),
        min(mask.shape[0], int(ys.max()) + padding + 1),
    )


def _metric_domains(
    reconstruction: np.ndarray,
    source: np.ndarray,
    mask: np.ndarray,
    args: argparse.Namespace,
) -> tuple[dict[str, object], dict[str, object]]:
    full = h22.report_utils._metric_values(
        np.ascontiguousarray(reconstruction),
        source,
        np.ones(source.shape[:2], dtype=bool),
        device=args.device,
        compute_lpips=args.lpips,
    )
    x0, y0, x1, y1 = _foreground_bounds(mask)
    crop_mask = np.ascontiguousarray(mask[y0:y1, x0:x1])
    foreground_source = np.ascontiguousarray(source[y0:y1, x0:x1] * crop_mask[:, :, None])
    foreground_reconstruction = np.ascontiguousarray(
        reconstruction[y0:y1, x0:x1] * crop_mask[:, :, None]
    )
    foreground = h22.report_utils._metric_values(
        foreground_reconstruction,
        foreground_source,
        crop_mask,
        device=args.device,
        compute_lpips=args.lpips,
    )
    for name, values in (("full", full), ("foreground", foreground)):
        if values["lpips"] is None:
            raise RuntimeError(
                f"LPIPS is required for {name} metrics but unavailable: {values['lpips_error']}"
            )
    return full, foreground


def _selection_metrics(
    reconstruction: np.ndarray,
    source: np.ndarray,
    mask: np.ndarray,
    mode: str,
    args: argparse.Namespace,
) -> dict[str, float]:
    if mode == "full_frame":
        values = h22.report_utils._metric_values(
            np.ascontiguousarray(reconstruction),
            source,
            np.ones(source.shape[:2], dtype=bool),
            device=args.device,
            compute_lpips=args.lpips,
        )
    else:
        x0, y0, x1, y1 = _foreground_bounds(mask)
        crop_mask = np.ascontiguousarray(mask[y0:y1, x0:x1])
        values = h22.report_utils._metric_values(
            np.ascontiguousarray(reconstruction[y0:y1, x0:x1] * crop_mask[:, :, None]),
            np.ascontiguousarray(source[y0:y1, x0:x1] * crop_mask[:, :, None]),
            crop_mask,
            device=args.device,
            compute_lpips=args.lpips,
        )
    if values["lpips"] is None:
        raise RuntimeError(
            f"LPIPS is required for {mode} projection safety but unavailable: "
            f"{values['lpips_error']}"
        )
    return {
        "raw_mse": float(values["masked_mse"]),
        "ms_ssim": float(values["ms_ssim"]),
        "lpips": float(values["lpips"]),
        "pixel_max": float(values["artifact_pixel_rmse_max"]),
        "patch7_max": float(values["artifact_patch_rmse_max_7"]),
    }


def _project_method(
    incoming: dict[str, object],
    source: np.ndarray,
    mask: np.ndarray,
    mode: str,
    args: argparse.Namespace,
    torch,
) -> dict[str, object]:
    print(f"[{mode}] project N={incoming['field'].n}", flush=True)
    started = time.perf_counter()
    result = project_additive_endpoint(
        incoming["field"],
        incoming["objective_target"],
        config=h24._projection_config(args),
        device=args.device,
        mask=mask if mode == "masked_foreground" else None,
    )
    projection_seconds = time.perf_counter() - started
    metric_started = time.perf_counter()
    incoming_metrics = _selection_metrics(
        np.asarray(incoming["expected"], dtype=np.float32), source, mask, mode, args
    )
    proposal_metrics = _selection_metrics(result.reconstruction_raw, source, mask, mode, args)
    projection_metric_seconds = time.perf_counter() - metric_started
    coefficient_abs_max = float(result.field.colors.detach().abs().max().cpu())
    decision = select_safe_projection(
        incoming_metrics,
        proposal_metrics,
        proposal_finite=bool(np.isfinite(result.reconstruction_raw).all()),
        coefficient_abs_max=coefficient_abs_max,
        config=h24._safety_config(),
    )
    method = dict(incoming)
    method.update(
        {
            "field": result.field if decision.selected else incoming["field"],
            "expected": (result.reconstruction_raw if decision.selected else incoming["expected"]),
            "projection_applied": True,
            "projection_selected": decision.selected,
            "projection_reason": decision.reason,
            "projection_clauses": dict(decision.clauses),
            "projection_seconds": projection_seconds,
            "projection_metric_seconds": projection_metric_seconds,
            "projection_result": result,
            "incoming_field": incoming["field"],
            "proposal_field": result.field,
            "incoming_selection_metrics": incoming_metrics,
            "proposal_selection_metrics": proposal_metrics,
            "incoming_field_digest": h24._field_digest(incoming["field"]),
            "proposal_field_digest": h24._field_digest(result.field),
            "endpoint_parity": max(
                float(incoming["endpoint_parity"]),
                float(result.projection.maintained_render_parity_max_abs),
            ),
        }
    )
    method["final_field_digest"] = h24._field_digest(method["field"])
    with torch.no_grad():
        method["coverage"] = h26._coverage(method["field"], source, args, torch)
    return method


def _pursuit_method(
    base: dict[str, object],
    source: np.ndarray,
    mask: np.ndarray,
    mode: str,
    args: argparse.Namespace,
    torch,
) -> dict[str, object]:
    print(f"[{mode}] append 64 residual-pursuit rows", flush=True)
    result = append_residual_pursuit_gaussians(
        base["field"],
        base["objective_target"],
        config=_tail_config(args),
        selection_mask=mask if mode == "masked_foreground" else None,
    )
    method = dict(base)
    method.update(
        {
            "field": result.field,
            "expected": result.reconstruction_raw,
            "completed": result.completed,
            "method_status": result.status,
            "pursuit_result": result,
            "pursuit_seconds": result.elapsed_seconds,
            "base_projection_final_digest": base["final_field_digest"],
            "final_field_digest": result.endpoint_field_digest,
            "preprojection_endpoint_digest": result.endpoint_field_digest,
            "endpoint_parity": max(
                float(base["endpoint_parity"]),
                result.analytic_render_parity_max_abs,
            ),
        }
    )
    with torch.no_grad():
        method["coverage"] = h26._coverage(result.field, source, args, torch)
    return method


def _base_method(method: dict[str, object]) -> dict[str, object]:
    result = h24._base_method(method)
    result["pursuit_result"] = None
    result["pursuit_seconds"] = 0.0
    result["base_projection_final_digest"] = result["final_field_digest"]
    return result


def _save_shared_audit(
    output_root: Path,
    mode: str,
    methods: dict[str, dict[str, object]],
) -> dict[str, object]:
    directory = output_root / "shared" / mode
    directory.mkdir(parents=True, exist_ok=True)
    normalized = methods["normalized_plain_n640"]
    base960 = methods["cold_additive_projected_n960"]
    pursuit = methods["residual_pursuit_additive_n1024"]
    cold1024 = methods["cold_additive_projected_n1024"]
    pursuit_result = pursuit["pursuit_result"]
    fields = {
        "n640_initial": normalized["audit_initial_field"],
        "n640_training": normalized["audit_training_field"],
        "n640_endpoint": normalized["field"],
        "n960_initial": base960["audit_initial_field"],
        "n960_training": base960["audit_training_field"],
        "n960_projected": base960["field"],
        "pursuit_tail64": pursuit_result.tail_field,
        "pursuit_n1024_endpoint": pursuit["field"],
        "n1024_initial": cold1024["audit_initial_field"],
        "n1024_training": cold1024["audit_training_field"],
        "n1024_projected": cold1024["field"],
    }
    records = {}
    for name, field in fields.items():
        record = h26._save_field(directory / f"{name}.field.gaussian.npz", field)
        record["path"] = str(Path(record["path"]).relative_to(output_root))
        records[name] = record
    receipt = {
        "schema": REPORT_SCHEMA,
        "mode": mode,
        "seed": 0,
        "fields": records,
        "counts": {"normalized": 640, "base": 960, "tail": 64, "total": 1024},
        "steps": {"normalized": 500, "base": 500, "cold_control": 500, "tail": 0},
        "gaussian_row_updates": GAUSSIAN_ROW_UPDATES_BY_ARM,
        "base_prefix_bit_exact": pursuit_result.base_prefix_bit_exact,
        "selection_mask_applied": pursuit_result.selection_mask_applied,
        "selection_active_pixels": pursuit_result.selection_active_pixels,
    }
    _write_json(directory / "receipt.json", receipt)
    return {
        "dir": str(directory.relative_to(output_root)),
        "receipt_path": str((directory / "receipt.json").relative_to(output_root)),
        "receipt_sha256": h22.report_utils._sha256(directory / "receipt.json"),
        "fields": records,
    }


def _metric_prefix(prefix: str, metrics: dict[str, object]) -> dict[str, object]:
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def _save_visuals(
    artifact_dir: Path,
    source: np.ndarray,
    reconstruction: np.ndarray,
    mask: np.ndarray,
    mode: str,
    error_scale: float,
) -> tuple[int, int, int, int]:
    objective_source = _objective(source, mask, mode)
    objective_reconstruction = (
        reconstruction if mode == "full_frame" else reconstruction * mask[:, :, None]
    )
    objective_mask = np.ones(mask.shape, dtype=bool) if mode == "full_frame" else mask
    save_image(str(artifact_dir / "source.png"), source)
    save_image(str(artifact_dir / "mask.png"), mask.astype(np.float32))
    save_image(str(artifact_dir / "reconstruction.png"), reconstruction)
    save_image(str(artifact_dir / "objective_source.png"), objective_source)
    save_image(str(artifact_dir / "objective_reconstruction.png"), objective_reconstruction)
    save_image(str(artifact_dir / "foreground_source.png"), source * mask[:, :, None])
    save_image(
        str(artifact_dir / "foreground_reconstruction.png"),
        reconstruction * mask[:, :, None],
    )
    save_error_heatmap(str(artifact_dir / "error.png"), reconstruction - source, scale=error_scale)
    save_error_heatmap(
        str(artifact_dir / "foreground_error.png"),
        (reconstruction - source) * mask[:, :, None],
        scale=error_scale,
    )
    save_error_heatmap(
        str(artifact_dir / "objective_error.png"),
        objective_reconstruction - objective_source,
        scale=error_scale,
    )
    bounds = h22.viz_utils._worst_crop_bounds(
        objective_reconstruction, objective_source, objective_mask
    )
    h22.viz_utils._save_crop(artifact_dir / "source_crop.png", source, bounds)
    h22.viz_utils._save_crop(artifact_dir / "reconstruction_crop.png", reconstruction, bounds)
    shown_error = np.repeat(
        np.clip(
            np.mean(np.abs(reconstruction.astype(np.float64) - source), axis=2) * error_scale,
            0.0,
            1.0,
        )[:, :, None],
        3,
        axis=2,
    )
    h22.viz_utils._save_crop(artifact_dir / "error_crop.png", shown_error, bounds)
    return bounds


def _write_cell(
    *,
    output_root: Path,
    image_path: Path,
    mask_path: Path,
    source: np.ndarray,
    mask: np.ndarray,
    raster: dict[str, object],
    mode: str,
    arm: str,
    method: dict[str, object],
    shared_audit: dict[str, object],
    args: argparse.Namespace,
    torch,
) -> dict[str, object]:
    count = COUNT_BY_ARM[arm]
    artifact_dir = output_root / "artifacts" / f"C0001__{mode}__s0__{arm}"
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
        cold_tensor = h22._field_render(
            cold_field,
            source.shape[0],
            source.shape[1],
            method["renderer"],
            args.render_chunk,
        )
        repeated_tensor = h22._field_render(
            cold_field,
            source.shape[0],
            source.shape[1],
            method["renderer"],
            args.render_chunk,
        )
    torch.cuda.synchronize()
    render_seconds = time.perf_counter() - render_started
    cold = cold_tensor.detach().cpu().numpy().astype(np.float32, copy=False)
    repeated = repeated_tensor.detach().cpu().numpy().astype(np.float32, copy=False)
    expected = np.asarray(method["expected"], dtype=np.float32)
    maintained_parity = float(np.max(np.abs(cold.astype(np.float64) - expected.astype(np.float64))))
    repeated_parity = float(np.max(np.abs(repeated.astype(np.float64) - cold.astype(np.float64))))
    metric_started = time.perf_counter()
    full_metrics, foreground_metrics = _metric_domains(cold, source, mask, args)
    metric_seconds = time.perf_counter() - metric_started
    primary = full_metrics if mode == "full_frame" else foreground_metrics
    bounds = _save_visuals(artifact_dir, source, cold, mask, mode, args.error_scale)
    h22._write_curve(
        artifact_dir / "learning_curve.svg",
        method["trajectory"],
        f"C0001 {mode} {arm}",
    )
    _write_json(artifact_dir / "fit_history.json", method["history"])
    incoming_path = artifact_dir / "incoming.field.gaussian.npz"
    proposal_path = artifact_dir / "proposal.field.gaussian.npz"
    method["incoming_field"].save(str(incoming_path))
    method["proposal_field"].save(str(proposal_path))
    projection = h26._projection_record(method)
    projection.update(
        {
            "schema": REPORT_SCHEMA,
            "mask_applied": mode == "masked_foreground",
            "objective_mode": mode,
            "selected": method["projection_selected"],
            "reason": method["projection_reason"],
            "clauses": method["projection_clauses"],
        }
    )
    _write_json(artifact_dir / "projection_history.json", projection)
    pursuit_result = method.get("pursuit_result")
    pursuit_path = artifact_dir / "pursuit_history.json"
    if pursuit_result is None:
        pursuit_payload = {
            "schema": REPORT_SCHEMA,
            "applied": False,
            "selection_mask_applied": False,
            "trajectory": [],
        }
    else:
        tail_path = artifact_dir / "tail.field.gaussian.npz"
        pursuit_result.tail_field.save(str(tail_path))
        pursuit_payload = {
            "schema": REPORT_SCHEMA,
            "applied": True,
            "config": asdict(_tail_config(args)),
            "base_count": pursuit_result.base_count,
            "tail_count": pursuit_result.tail_count,
            "total_count": pursuit_result.total_count,
            "base_field_digest": pursuit_result.base_field_digest,
            "tail_field_digest": pursuit_result.tail_field_digest,
            "endpoint_field_digest": pursuit_result.endpoint_field_digest,
            "base_prefix_bit_exact": pursuit_result.base_prefix_bit_exact,
            "fixed_tail_geometry": pursuit_result.fixed_tail_geometry,
            "training_payload_removed": pursuit_result.training_payload_removed,
            "selection_mask_applied": pursuit_result.selection_mask_applied,
            "selection_active_pixels": pursuit_result.selection_active_pixels,
            "residual_scan_pixel_evaluations": (pursuit_result.residual_scan_pixel_evaluations),
            "tail_kernel_pixel_updates": pursuit_result.tail_kernel_pixel_updates,
            "analytic_render_parity_max_abs": (pursuit_result.analytic_render_parity_max_abs),
            "initial_pixel_rmse_max": pursuit_result.initial_pixel_rmse_max,
            "final_pixel_rmse_max": pursuit_result.final_pixel_rmse_max,
            "coefficient_abs_max": pursuit_result.coefficient_abs_max,
            "trajectory": pursuit_result.trajectory_records(),
            "tail_file": "tail.field.gaussian.npz",
            "tail_file_sha256": h22.report_utils._sha256(tail_path),
        }
    _write_json(pursuit_path, pursuit_payload)
    _write_json(
        artifact_dir / "geometry_history.json",
        {
            "schema": REPORT_SCHEMA,
            "initial_field_digest": method["initial_field_digest"],
            "preprojection_endpoint_digest": method["preprojection_endpoint_digest"],
            "final_field_digest": method["final_field_digest"],
            "base_projection_final_digest": method["base_projection_final_digest"],
            "training_payload_stripped": arm in PURE_ADDITIVE_ARMS,
            "mask_encoder_only": mode == "masked_foreground",
        },
    )
    fit_count = 960 if arm == "residual_pursuit_additive_n1024" else count
    renderer = "cuda" if arm == "normalized_plain_n640" else "cuda_additive"
    _write_json(
        artifact_dir / "config.json",
        {
            "schema": REPORT_SCHEMA,
            "mode": mode,
            "arm": arm,
            "seed": args.seed,
            "count": count,
            "init_count": fit_count,
            "init": asdict(h22._init_config(fit_count, args.seed)),
            "fit": asdict(
                _fit_config(
                    args,
                    renderer,
                    fit_count,
                    masked=mode == "masked_foreground",
                )
            ),
            "projection": (
                asdict(h24._projection_config(args)) if arm != "normalized_plain_n640" else None
            ),
            "safety": (asdict(h24._safety_config()) if arm != "normalized_plain_n640" else None),
            "pursuit": (
                asdict(_tail_config(args)) if arm == "residual_pursuit_additive_n1024" else None
            ),
            "shared_audit_receipt": shared_audit["receipt_path"],
        },
    )
    np.savez_compressed(
        artifact_dir / "analysis.npz",
        crop_bounds=np.asarray(bounds, dtype=np.int32),
        foreground_bounds=np.asarray(_foreground_bounds(mask), dtype=np.int32),
        mask=mask,
        reconstruction_raw=cold,
        full_error_raw=cold.astype(np.float32) - source.astype(np.float32),
        objective_error_raw=(cold.astype(np.float32) - source.astype(np.float32))
        * (
            np.ones((*mask.shape, 1), dtype=np.float32)
            if mode == "full_frame"
            else mask[:, :, None]
        ),
        trajectory_step=np.asarray([row["step"] for row in method["trajectory"]], dtype=np.float32),
        trajectory_psnr_db=np.asarray(
            [row["psnr_db"] for row in method["trajectory"]], dtype=np.float32
        ),
    )
    pure = arm in PURE_ADDITIVE_ARMS
    base_projection_final = method.get("base_projection_final_digest", method["final_field_digest"])
    row: dict[str, object] = {
        "schema": REPORT_SCHEMA,
        "status": "diagnostic",
        "phase": "exposed_single_image_full_resolution_diagnostic",
        "image": "C0001",
        "mode": mode,
        "objective_domain": (
            "full_frame" if mode == "full_frame" else "black_matted_foreground_crop"
        ),
        "arm": arm,
        "seed": args.seed,
        "semantic_family": method["semantic_family"],
        "renderer": method["renderer"],
        "artifact_dir": str(artifact_dir.relative_to(output_root)),
        "source_path": str(image_path),
        "source_sha256": SOURCE_SHA256,
        "source_file_bytes": image_path.stat().st_size,
        "mask_path": str(mask_path),
        "mask_sha256": MASK_SHA256,
        "mask_file_bytes": mask_path.stat().st_size,
        "mask_threshold": 0.5,
        "original_width": raster["original_width"],
        "original_height": raster["original_height"],
        "width": source.shape[1],
        "height": source.shape[0],
        "active_pixels": (int(mask.sum()) if mode == "masked_foreground" else int(mask.size)),
        "mask_active_pixels": int(mask.sum()),
        "mask_active_fraction": float(mask.mean()),
        "target_gaussians": count,
        "n_gaussians": field.n,
        "field_file_sha256": h22.report_utils._sha256(field_path),
        "field_file_bytes": field_path.stat().st_size,
        "field_npz_keys": field_keys,
        "mass_payload_present": any("mass" in key.lower() for key in field_keys),
        "denominator_payload_present": any("denom" in key.lower() for key in field_keys),
        "optimizer_payload_present": any("optimizer" in key.lower() for key in field_keys),
        "auxiliary_rgb_payload_present": any(
            key in field_keys for key in ("color_grads", "opacities")
        ),
        "pure_additive_endpoint": pure,
        "four_array_endpoint_exact": not pure or set(field_keys) == FOUR_ARRAY_KEYS,
        "training_payload_present": pure and set(field_keys) != FOUR_ARRAY_KEYS,
        "mask_encoder_only": mode == "masked_foreground",
        "method_status": method["method_status"],
        "completed": method["completed"],
        "selected_step": method["selected_step"],
        "selected_lambda": 0.0 if pure else None,
        "endpoint_internal_parity_max_abs": method["endpoint_parity"],
        "attempted_steps": 500,
        "gaussian_row_updates": GAUSSIAN_ROW_UPDATES_BY_ARM[arm],
        "renderer_calls_fit": method["renderer_calls"],
        "normalized_calls_fit": method["normalized_calls"],
        "additive_numerator_calls_fit": method["additive_numerator_calls"],
        "additive_denominator_calls_fit": method["additive_denominator_calls"],
        "diagnostic_renderer_calls_fit": method["diagnostic_renderer_calls"],
        "init_seconds": method["init_seconds"],
        "fit_seconds": method["fit_seconds"],
        "wall_fit_seconds": method["wall_fit_seconds"],
        "cold_decode_seconds": decode_seconds,
        "render_seconds": render_seconds,
        "metric_seconds": metric_seconds,
        "projection_seconds": method["projection_seconds"],
        "projection_metric_seconds": method["projection_metric_seconds"],
        "pursuit_seconds": method.get("pursuit_seconds", 0.0),
        "pipeline_algorithm_seconds": (
            float(method["init_seconds"])
            + float(method["fit_seconds"])
            + float(method["projection_seconds"])
            + float(method.get("pursuit_seconds", 0.0))
        ),
        "total_seconds": (
            float(method["init_seconds"])
            + float(method["wall_fit_seconds"])
            + decode_seconds
            + render_seconds
            + metric_seconds
            + float(method["projection_seconds"])
            + float(method["projection_metric_seconds"])
            + float(method.get("pursuit_seconds", 0.0))
        ),
        "peak_cuda_allocated_bytes": method["peak_cuda_allocated_bytes"],
        "maintained_render_parity_max_abs": maintained_parity,
        "repeated_render_parity_max_abs": repeated_parity,
        "finite_reconstruction": bool(np.isfinite(cold).all()),
        "masked_mse": primary["masked_mse"],
        "raw_mse": primary["masked_mse"],
        "psnr_db": primary["psnr_db"],
        "ssim": primary["ssim"],
        "ms_ssim": primary["ms_ssim"],
        "ssim_window": primary["ssim_window"],
        "lpips": primary["lpips"],
        "lpips_error": primary["lpips_error"],
        "artifact_metric_domain": primary["artifact_metric_domain"],
        "artifact_pixel_rmse_q99": primary["artifact_pixel_rmse_q99"],
        "artifact_pixel_rmse_q999": primary["artifact_pixel_rmse_q999"],
        "artifact_pixel_rmse_max": primary["artifact_pixel_rmse_max"],
        "artifact_patch_rmse_max_7": primary["artifact_patch_rmse_max_7"],
        **_metric_prefix("full", full_metrics),
        **_metric_prefix("foreground", foreground_metrics),
        **h22._display_metrics(cold, method["objective_target"]),
        **h22._coefficient_record(field),
        **method["coverage"],
        "projection_applied": method["projection_applied"],
        "projection_mask_applied": (mode == "masked_foreground" and method["projection_applied"]),
        "projection_scope": (
            "base_n960"
            if arm == "residual_pursuit_additive_n1024"
            else "endpoint"
            if method["projection_applied"]
            else "none"
        ),
        "projection_selected": method["projection_selected"],
        "projection_reason": method["projection_reason"],
        "projection_clauses": method["projection_clauses"],
        "projection_selected_iteration": projection["selected_iteration"],
        "projection_initial_sse": projection["initial_sse"],
        "projection_final_sse": projection["final_sse"],
        "projection_forward_applications": projection["forward_applications"],
        "projection_transpose_applications": projection["transpose_applications"],
        "projection_relative_normal_residual_max": projection["relative_normal_residual_max"],
        "projection_adjoint_relative_error": projection["adjoint_relative_error"],
        "projection_initial_operator_parity_max_abs": projection["initial_operator_parity_max_abs"],
        "projection_maintained_render_parity_max_abs": projection[
            "maintained_render_parity_max_abs"
        ],
        "projection_geometry_exact": projection["geometry_exact"],
        "incoming_field_digest": method["incoming_field_digest"],
        "proposal_field_digest": method["proposal_field_digest"],
        "base_projection_final_digest": base_projection_final,
        "final_field_digest": method["final_field_digest"],
        "initial_field_digest": method["initial_field_digest"],
        "preprojection_endpoint_digest": method["preprojection_endpoint_digest"],
        "incoming_field_file_sha256": h22.report_utils._sha256(incoming_path),
        "proposal_field_file_sha256": h22.report_utils._sha256(proposal_path),
        "shared_audit_dir": shared_audit["dir"],
        "shared_audit_receipt": shared_audit["receipt_path"],
        "shared_audit_receipt_sha256": shared_audit["receipt_sha256"],
        "pursuit_applied": pursuit_result is not None,
        "pursuit_base_count": (None if pursuit_result is None else pursuit_result.base_count),
        "pursuit_tail_count": (None if pursuit_result is None else pursuit_result.tail_count),
        "pursuit_base_field_digest": (
            None if pursuit_result is None else pursuit_result.base_field_digest
        ),
        "pursuit_tail_field_digest": (
            None if pursuit_result is None else pursuit_result.tail_field_digest
        ),
        "pursuit_base_prefix_bit_exact": (
            None if pursuit_result is None else pursuit_result.base_prefix_bit_exact
        ),
        "pursuit_fixed_tail_geometry": (
            None if pursuit_result is None else pursuit_result.fixed_tail_geometry
        ),
        "pursuit_selection_mask_applied": (
            False if pursuit_result is None else pursuit_result.selection_mask_applied
        ),
        "pursuit_selection_active_pixels": (
            0 if pursuit_result is None else pursuit_result.selection_active_pixels
        ),
        "pursuit_analytic_render_parity_max_abs": (
            None if pursuit_result is None else pursuit_result.analytic_render_parity_max_abs
        ),
        "pursuit_residual_scan_pixel_evaluations": (
            0 if pursuit_result is None else pursuit_result.residual_scan_pixel_evaluations
        ),
        "pursuit_tail_kernel_pixel_updates": (
            0 if pursuit_result is None else pursuit_result.tail_kernel_pixel_updates
        ),
        "pursuit_renderer_calls": (0 if pursuit_result is None else pursuit_result.renderer_calls),
        "pursuit_history_path": str(pursuit_path.relative_to(output_root)),
    }
    row["pursuit_history_sha256"] = h22.report_utils._sha256(pursuit_path)
    for prefix, values in (
        ("incoming", method["incoming_selection_metrics"]),
        ("proposal", method["proposal_selection_metrics"]),
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
        if columns:
            writer.writeheader()
            writer.writerows(rows)


def _delta(
    indexed: dict[tuple[str, str], dict[str, object]],
    mode: str,
    left: str,
    right: str,
    metric: str,
) -> float | None:
    if (mode, left) not in indexed or (mode, right) not in indexed:
        return None
    return float(indexed[(mode, left)][metric]) - float(indexed[(mode, right)][metric])


def _decision(rows: list[dict[str, object]]) -> dict[str, object]:
    indexed = {(str(row["mode"]), str(row["arm"])): row for row in rows}
    expected = {(mode, arm) for mode in MODES for arm in ARMS}
    all_cells = set(indexed) == expected and len(rows) == len(expected)
    integrity = bool(
        all_cells
        and all(
            row["completed"]
            and row["n_gaussians"] == COUNT_BY_ARM[str(row["arm"])]
            and row["maintained_render_parity_max_abs"] <= PARITY_LIMIT
            and row["repeated_render_parity_max_abs"] <= PARITY_LIMIT
            and row["endpoint_internal_parity_max_abs"] <= PARITY_LIMIT
            for row in rows
        )
    )
    comparisons = {}
    for mode in MODES:
        comparisons[mode] = {
            "pursuit_vs_normalized_primary_psnr_db": _delta(
                indexed,
                mode,
                "residual_pursuit_additive_n1024",
                "normalized_plain_n640",
                "psnr_db",
            ),
            "pursuit_vs_cold_n1024_primary_psnr_db": _delta(
                indexed,
                mode,
                "residual_pursuit_additive_n1024",
                "cold_additive_projected_n1024",
                "psnr_db",
            ),
            "pursuit_vs_base_n960_primary_psnr_db": _delta(
                indexed,
                mode,
                "residual_pursuit_additive_n1024",
                "cold_additive_projected_n960",
                "psnr_db",
            ),
            "pursuit_vs_normalized_foreground_psnr_db": _delta(
                indexed,
                mode,
                "residual_pursuit_additive_n1024",
                "normalized_plain_n640",
                "foreground_psnr_db",
            ),
            "pursuit_vs_normalized_full_psnr_db": _delta(
                indexed,
                mode,
                "residual_pursuit_additive_n1024",
                "normalized_plain_n640",
                "full_psnr_db",
            ),
        }
    return {
        "schema": REPORT_SCHEMA,
        "status": "diagnostic",
        "all_cells_present": all_cells,
        "integrity_pass": integrity,
        "comparisons": comparisons,
        "visual_review": "pending_producer_audit",
        "overall_pass": False,
        "formal_claim_ready": False,
        "interpretation": (
            "Complete exposed Janelle scaling diagnostic; inspect metrics and native visuals."
            if all_cells and integrity
            else "Diagnostic is incomplete or failed integrity; inspect attempts and errors."
        ),
        "claim_limits": [
            "one exposed image and one seed",
            "max-side-1200 project full-resolution regime, not native 5328x4608",
            "dirty-source producer diagnostic",
            "no default, held-out, codec-rate, or publication claim",
        ],
    }


def _fmt(value: object, digits: int = 4) -> str:
    return "—" if value is None else f"{float(value):.{digits}f}"


def _write_report(
    output_root: Path,
    rows: list[dict[str, object]],
    attempts: list[dict[str, object]],
    decision: dict[str, object],
) -> None:
    table_rows = []
    cards = []
    for row in rows:
        artifact = escape(str(row["artifact_dir"]))
        table_rows.append(
            "<tr>"
            f"<td>{escape(str(row['mode']))}</td>"
            f"<td>{escape(str(row['arm']))}</td><td>{int(row['n_gaussians'])}</td>"
            f"<td>{_fmt(row['psnr_db'], 3)}</td><td>{_fmt(row['ms_ssim'], 5)}</td>"
            f"<td>{_fmt(row['lpips'], 5)}</td>"
            f"<td>{_fmt(row['full_psnr_db'], 3)}</td>"
            f"<td>{_fmt(row['full_ms_ssim'], 5)}</td><td>{_fmt(row['full_lpips'], 5)}</td>"
            f"<td>{_fmt(row['foreground_psnr_db'], 3)}</td>"
            f"<td>{_fmt(row['foreground_ms_ssim'], 5)}</td>"
            f"<td>{_fmt(row['foreground_lpips'], 5)}</td>"
            f"<td>{_fmt(row['artifact_pixel_rmse_max'], 4)}</td>"
            f"<td>{_fmt(row['artifact_patch_rmse_max_7'], 4)}</td>"
            f"<td>{float(row['fit_seconds']):.1f}s</td>"
            f"<td><a href='{artifact}/row.json'>row</a> · "
            f"<a href='{artifact}/field.gaussian.npz'>field</a> · "
            f"<a href='{artifact}/fit_history.json'>fit</a> · "
            f"<a href='{artifact}/projection_history.json'>projection</a> · "
            f"<a href='{artifact}/pursuit_history.json'>pursuit</a></td></tr>"
        )
        cards.append(
            f"<article class='card'><h3>{escape(str(row['mode']))} · "
            f"{escape(str(row['arm']))}</h3>"
            f"<p>Objective PSNR {_fmt(row['psnr_db'], 3)} dB · MS-SSIM "
            f"{_fmt(row['ms_ssim'], 5)} · LPIPS {_fmt(row['lpips'], 5)}</p>"
            "<div class='images'>"
            f"<figure><a href='{artifact}/source.png'><img src='{artifact}/source.png'></a>"
            "<figcaption>source</figcaption></figure>"
            f"<figure><a href='{artifact}/reconstruction.png'><img "
            f"src='{artifact}/reconstruction.png'></a><figcaption>raw reconstruction</figcaption>"
            "</figure>"
            f"<figure><a href='{artifact}/error.png'><img src='{artifact}/error.png'></a>"
            "<figcaption>full-frame error ×4</figcaption></figure>"
            f"<figure><a href='{artifact}/foreground_reconstruction.png'><img "
            f"src='{artifact}/foreground_reconstruction.png'></a>"
            "<figcaption>black-matted foreground</figcaption></figure>"
            f"<figure><a href='{artifact}/foreground_error.png'><img "
            f"src='{artifact}/foreground_error.png'></a>"
            "<figcaption>foreground error ×4</figcaption></figure>"
            f"<figure><a href='{artifact}/reconstruction_crop.png'><img "
            f"src='{artifact}/reconstruction_crop.png'></a>"
            "<figcaption>worst objective crop</figcaption></figure>"
            f"<a class='hidden-link' href='{artifact}/source_crop.png'>source crop</a>"
            f"<a class='hidden-link' href='{artifact}/error_crop.png'>error crop</a>"
            "</div></article>"
        )
    errors = [attempt for attempt in attempts if attempt.get("status") != "ok"]
    error_html = (
        "<p>No execution errors were recorded.</p>"
        if not errors
        else f"<pre>{escape(json.dumps(errors, indent=2, sort_keys=True))}</pre>"
    )
    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>HIER-029 Janelle full-resolution mask diagnostic</title><style>
:root{{--ink:#18202a;--muted:#66717f;--line:#d8dee6;--panel:#f6f8fa}}
body{{font-family:system-ui,sans-serif;color:var(--ink);margin:2rem;max-width:2200px}}
h1,h2{{line-height:1.15}}p{{max-width:1100px}}code,pre{{white-space:pre-wrap}}
table{{border-collapse:collapse;font-size:.88rem}}th,td{{border:1px solid var(--line);padding:.4rem}}
th{{background:var(--panel);position:sticky;top:0}}.cards{{display:grid;gap:1.5rem}}
.card{{border-top:2px solid var(--line)}}.images{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:.7rem}}
figure{{margin:0}}img{{width:100%;height:auto;display:block;background:#111}}figcaption{{color:var(--muted);font-size:.84rem}}
.hidden-link{{font-size:.8rem}}@media(max-width:900px){{.images{{grid-template-columns:1fr}}}}
</style></head><body><h1>HIER-029 · Janelle C0001 mask diagnostic</h1>
<p><strong>Diagnostic only.</strong> “Full resolution” means StructSplat’s established Janelle
max-side-1200 regime (1200×1038), derived from the hash-bound native 5328×4608 camera image.
This is one exposed image and one seed, produced from a dirty worktree; it cannot confirm a
general claim.</p>
<p>The full-frame mode trains every pixel. The masked-foreground mode uses masked initialization,
mask-weighted L1/SSIM, mask-restricted projection, and mask-restricted residual selection. The
mask is encoder-only and no containment/zero-outside rule is applied. Foreground perceptual
metrics use a black-matted, mask-bounding-box crop; full metrics always use the complete RGB
canvas.</p>
<p><code>{escape(_command())}</code></p>
<p><a href="input/source.png">evaluation source</a> · <a href="input/mask.png">mask</a> ·
<a href="config.json">config</a> · <a href="decision.json">decision</a> ·
<a href="metrics.json">JSON</a> · <a href="metrics.jsonl">JSONL</a> ·
<a href="metrics.csv">CSV</a> · <a href="attempts.json">attempts</a> ·
<a href="manifest.json">manifest</a></p>
<h2>Diagnostic comparison</h2><pre>{escape(json.dumps(decision, indent=2, sort_keys=True))}</pre>
<h2>Metrics</h2><div style="overflow:auto"><table><tr><th>mode</th><th>arm</th><th>N</th>
<th>objective PSNR</th><th>objective MS-SSIM</th><th>objective LPIPS</th>
<th>full PSNR</th><th>full MS-SSIM</th><th>full LPIPS</th>
<th>foreground PSNR</th><th>foreground MS-SSIM</th><th>foreground LPIPS</th>
<th>pixel max</th><th>7×7 max</th><th>fit</th><th>artifacts</th></tr>
{"".join(table_rows)}</table></div><h2>Execution errors</h2>{error_html}
<h2>Native-size visual comparisons and error maps</h2><div class="cards">{"".join(cards)}</div>
</body></html>"""
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


def _run_mode(
    mode: str,
    source: np.ndarray,
    mask: np.ndarray,
    args: argparse.Namespace,
    torch,
) -> dict[str, dict[str, object]]:
    normalized_raw = _run_fit(source, mask, mode, 640, "cuda", args, torch)
    cold960_raw = _run_fit(source, mask, mode, 960, "cuda_additive", args, torch)
    cold1024_raw = _run_fit(source, mask, mode, 1024, "cuda_additive", args, torch)
    normalized = _base_method(normalized_raw)
    projected960 = _project_method(cold960_raw, source, mask, mode, args, torch)
    projected960["pursuit_result"] = None
    projected960["pursuit_seconds"] = 0.0
    projected960["base_projection_final_digest"] = projected960["final_field_digest"]
    projected1024 = _project_method(cold1024_raw, source, mask, mode, args, torch)
    projected1024["pursuit_result"] = None
    projected1024["pursuit_seconds"] = 0.0
    projected1024["base_projection_final_digest"] = projected1024["final_field_digest"]
    pursuit = _pursuit_method(projected960, source, mask, mode, args, torch)
    return {
        "normalized_plain_n640": normalized,
        "cold_additive_projected_n960": projected960,
        "residual_pursuit_additive_n1024": pursuit,
        "cold_additive_projected_n1024": projected1024,
    }


def main() -> None:
    args = _parser().parse_args()
    _validate_args(args)
    if (args.out / "COMPLETED").is_file():
        raise SystemExit(f"completed HIER-029 bundle is immutable: {args.out}")
    if args.out.exists() and any(args.out.iterdir()) and not args.resume:
        raise SystemExit(f"output directory is non-empty; use --resume: {args.out}")
    args.out.mkdir(parents=True, exist_ok=True)

    if h22.report_utils._sha256(args.image) != SOURCE_SHA256:
        raise SystemExit("Janelle source SHA-256 differs from the frozen HIER-029 binding")
    if h22.report_utils._sha256(args.mask) != MASK_SHA256:
        raise SystemExit("Janelle mask SHA-256 differs from the frozen HIER-029 binding")

    import torch

    if not torch.cuda.is_available():
        raise SystemExit("frozen HIER-029 diagnostic requires CUDA")
    source, mask, raster = h22.report_utils._load_evaluation_raster(
        args.image, args.mask, max_side=args.max_side, mask_threshold=0.5
    )
    if mask is None:
        raise RuntimeError("HIER-029 mask was not loaded")
    if (raster["original_height"], raster["original_width"]) != NATIVE_SHAPE:
        raise RuntimeError(f"native Janelle shape differs: {raster!r}")
    if source.shape[:2] != EVALUATION_SHAPE:
        raise RuntimeError(f"evaluation raster must be {EVALUATION_SHAPE}, got {source.shape}")

    input_dir = args.out / "input"
    input_dir.mkdir(exist_ok=True)
    save_image(str(input_dir / "source.png"), source)
    save_image(str(input_dir / "mask.png"), mask.astype(np.float32))
    save_image(str(input_dir / "foreground_black_matted.png"), source * mask[:, :, None])
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
            "arguments": vars(args),
            "source": {
                "path": str(args.image.resolve()),
                "sha256": SOURCE_SHA256,
                "native_shape": list(NATIVE_SHAPE),
            },
            "mask": {
                "path": str(args.mask.resolve()),
                "sha256": MASK_SHA256,
                "threshold": 0.5,
                "active_pixels": int(mask.sum()),
                "active_fraction": float(mask.mean()),
            },
            "raster": raster,
            "modes": list(MODES),
            "arms": list(ARMS),
            "counts": COUNT_BY_ARM,
            "gaussian_row_updates": GAUSSIAN_ROW_UPDATES_BY_ARM,
            "structure_tensor": asdict(StructureTensorConfig()),
            "fit_normalized_full": asdict(_fit_config(args, "cuda", 640, masked=False)),
            "fit_normalized_masked": asdict(_fit_config(args, "cuda", 640, masked=True)),
            "fit_additive_n960_full": asdict(_fit_config(args, "cuda_additive", 960, masked=False)),
            "fit_additive_n960_masked": asdict(
                _fit_config(args, "cuda_additive", 960, masked=True)
            ),
            "projection": asdict(h24._projection_config(args)),
            "safety": asdict(h24._safety_config()),
            "pursuit": asdict(_tail_config(args)),
            "claim_limits": [
                "one exposed Janelle image and seed zero",
                "max-side-1200 project regime, not native-5328",
                "dirty-source producer diagnostic",
                "no default or publication claim",
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
        "HIER-029 hash-bound Janelle source and mask decoded after protocol freeze.\n",
        encoding="utf-8",
    )

    rows: list[dict[str, object]] = []
    attempts: list[dict[str, object]] = []
    metrics_path = args.out / "metrics.json"
    attempts_path = args.out / "attempts.json"
    if args.resume and metrics_path.is_file():
        rows = json.loads(metrics_path.read_text(encoding="utf-8")).get("rows", [])
    if args.resume and attempts_path.is_file():
        attempts = json.loads(attempts_path.read_text(encoding="utf-8")).get("attempts", [])
    row_keys = {(row["mode"], row["arm"]) for row in rows}
    for mode in MODES:
        expected_keys = {(mode, arm) for arm in ARMS}
        if expected_keys <= row_keys:
            continue
        mode_started = time.perf_counter()
        methods: dict[str, dict[str, object]] = {}
        shared_audit = None
        fit_error = None
        try:
            methods = _run_mode(mode, source, mask, args, torch)
            shared_audit = _save_shared_audit(args.out, mode, methods)
        except Exception as exc:
            fit_error = exc
        for arm in ARMS:
            key = (mode, arm)
            if key in row_keys:
                continue
            cell_started = time.perf_counter()
            try:
                if fit_error is not None:
                    raise RuntimeError(f"paired mode execution failed: {fit_error}")
                if shared_audit is None:
                    raise RuntimeError("shared audit receipt was not created")
                row = _write_cell(
                    output_root=args.out,
                    image_path=args.image.resolve(),
                    mask_path=args.mask.resolve(),
                    source=source,
                    mask=mask,
                    raster=raster,
                    mode=mode,
                    arm=arm,
                    method=methods[arm],
                    shared_audit=shared_audit,
                    args=args,
                    torch=torch,
                )
                rows.append(row)
                row_keys.add(key)
                attempts.append(
                    {
                        "image": "C0001",
                        "mode": mode,
                        "seed": args.seed,
                        "arm": arm,
                        "status": "ok",
                        "elapsed_seconds": time.perf_counter() - cell_started,
                        "mode_elapsed_seconds": time.perf_counter() - mode_started,
                    }
                )
            except Exception as exc:
                attempts.append(
                    {
                        "image": "C0001",
                        "mode": mode,
                        "seed": args.seed,
                        "arm": arm,
                        "status": "error",
                        "error": f"{type(exc).__name__}: {exc}"[:2000],
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
    _write_report(args.out, rows, attempts, decision)
    (args.out / "COMPLETED").write_text(
        "HIER-029 exposed Janelle diagnostic complete; do not overwrite.\n",
        encoding="utf-8",
    )
    _write_manifest(args.out)
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
