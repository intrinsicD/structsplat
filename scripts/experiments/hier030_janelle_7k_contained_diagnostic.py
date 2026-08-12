#!/usr/bin/env python3
"""Run HIER-030's 7k Janelle capacity and hard-containment diagnostic."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
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
from scripts.experiments import hier029_janelle_mask_diagnostic as h29  # noqa: E402
from structsplat import mask as mask_geometry  # noqa: E402
from structsplat.cli import save_error_heatmap, save_image  # noqa: E402
from structsplat.config import FitConfig, StructureTensorConfig  # noqa: E402
from structsplat.endpoint_appearance_projection import (  # noqa: E402
    project_additive_endpoint,
    select_safe_projection,
)
from structsplat.fit import fit  # noqa: E402
from structsplat.gaussians import GaussianField  # noqa: E402
from structsplat.init import build_field, build_masked_field  # noqa: E402
from structsplat.render import render_field  # noqa: E402
from structsplat.residual_pursuit_additive import (  # noqa: E402
    ResidualPursuitAdditiveConfig,
    append_residual_pursuit_gaussians,
)


REPORT_SCHEMA = "structsplat.hier030_janelle_7k_contained.diagnostic.v1"
SOURCE_SHA256 = h29.SOURCE_SHA256
MASK_SHA256 = h29.MASK_SHA256
NATIVE_SHAPE = h29.NATIVE_SHAPE
EVALUATION_SHAPE = h29.EVALUATION_SHAPE
MODES = ("full_frame", "masked_contained")
ARMS = (
    "normalized_plain_n4375",
    "cold_additive_projected_n6562",
    "residual_pursuit_additive_n7000",
    "cold_additive_projected_n7000",
)
NORMALIZED_ARM = ARMS[0]
BASE_ARM = ARMS[1]
PURSUIT_ARM = ARMS[2]
COLD_ARM = ARMS[3]
PURE_ADDITIVE_ARMS = frozenset((BASE_ARM, PURSUIT_ARM, COLD_ARM))
PROJECTED_ARMS = frozenset((BASE_ARM, COLD_ARM))
COUNT_BY_ARM = {
    NORMALIZED_ARM: 4375,
    BASE_ARM: 6562,
    PURSUIT_ARM: 7000,
    COLD_ARM: 7000,
}
FIT_COUNT_BY_ARM = {
    NORMALIZED_ARM: 4375,
    BASE_ARM: 6562,
    PURSUIT_ARM: 6562,
    COLD_ARM: 7000,
}
TAIL_COUNT = 438
ITERS = 500
GAUSSIAN_ROW_UPDATES_BY_ARM = {
    arm: FIT_COUNT_BY_ARM[arm] * ITERS for arm in ARMS
}
FOUR_ARRAY_KEYS = frozenset(("means", "log_scales", "rotations", "colors"))
COEFFICIENT_LIMIT = 16.0
PARITY_LIMIT = 2e-5
CONTAINMENT_LIMIT = 1e-7
MASK_MARGIN = 0.75
SIGMA_CUTOFF = 3.0
TAIL_SCALE = 0.35
TAIL_EROSION_RADIUS = MASK_MARGIN + SIGMA_CUTOFF * TAIL_SCALE
DEFAULT_HIER029_REPORT = (
    ROOT
    / "results"
    / "hier029_janelle_c0001_s1200_mask_factorial_s0_diagnostic_2026-08-11"
)


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
    parser.add_argument("--hier029-report", type=Path, default=DEFAULT_HIER029_REPORT)
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
                f"frozen HIER-030 diagnostic requires {name}={expected!r}, "
                f"got {getattr(args, name)!r}"
            )
    if not math.isfinite(args.error_scale) or args.error_scale <= 0.0:
        raise SystemExit("--error-scale must be finite and positive")
    for name in ("image", "mask"):
        if not getattr(args, name).is_file():
            raise SystemExit(f"{name} does not exist: {getattr(args, name)}")
    if not (args.hier029_report / "metrics.json").is_file():
        raise SystemExit(f"HIER-029 metrics are missing: {args.hier029_report}")
    args.iters = ITERS
    args.budgets = [4375]


def _command() -> str:
    return " ".join(shlex.quote(value) for value in sys.argv)


def _write_json(path: Path, value: object) -> None:
    h22._write_json(path, value)


def _snapshot_sources(output_root: Path) -> list[dict[str, object]]:
    paths = (
        Path(__file__).resolve(),
        ROOT / "scripts" / "experiments" / "hier029_janelle_mask_diagnostic.py",
        ROOT / "src" / "structsplat" / "residual_pursuit_additive.py",
        ROOT / "src" / "structsplat" / "endpoint_appearance_projection.py",
        ROOT / "src" / "structsplat" / "fit.py",
        ROOT / "src" / "structsplat" / "init.py",
        ROOT / "tests" / "test_residual_pursuit_additive.py",
        ROOT / "tests" / "test_hier030_janelle_7k_contained_diagnostic.py",
        ROOT / "tasks" / "HIER-030-janelle-7k-contained-mask-diagnostic.md",
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


def _objective(source: np.ndarray, mask: np.ndarray, mode: str) -> np.ndarray:
    if mode == "full_frame":
        return np.array(source, dtype=np.float32, order="C", copy=True)
    if mode == "masked_contained":
        return np.ascontiguousarray(source * mask[:, :, None])
    raise ValueError(f"unknown mode {mode!r}")


def _init_config(count: int, seed: int, *, contained: bool):
    config = h22._init_config(count, seed)
    if contained:
        config = replace(config, scale_cap_mode="none", scale_cap_max=None)
    return config


def _fit_config(
    args: argparse.Namespace,
    renderer: str,
    count: int,
    *,
    contained: bool,
) -> FitConfig:
    return FitConfig(
        iters=ITERS,
        lr_means=5e-2,
        lr_scales=3e-2,
        lr_rot=1e-2,
        lr_color=3e-2,
        optimizer="adam",
        pixel_loss="l1",
        loss_weighting="mask" if contained else "none",
        mask_contain=contained,
        mask_margin=MASK_MARGIN,
        mask_cap_mode="anisotropic",
        mask_cap_refresh_every=10,
        ssim_weight=0.3,
        log_every=25,
        checkpoint_policy="best_psnr_final_count",
        sigma_cutoff=SIGMA_CUTOFF,
        support_fade=contained,
        aa_dilation=0.0,
        render_chunk=args.render_chunk,
        renderer=renderer,
        color_basis="constant",
        compute_lpips=False,
        max_gaussians=count,
    )


def _projection_config(args: argparse.Namespace, *, contained: bool):
    return replace(
        h24._projection_config(args),
        support_fade_alpha=1.0 if contained else 0.0,
    )


def _tail_config(args: argparse.Namespace, *, contained: bool):
    return ResidualPursuitAdditiveConfig(
        tail_gaussians=TAIL_COUNT,
        scale_px=TAIL_SCALE,
        coefficient_abs_limit=COEFFICIENT_LIMIT,
        sigma_cutoff=SIGMA_CUTOFF,
        support_fade=contained,
        render_chunk=args.render_chunk,
        renderer="cuda_additive",
    )


def _render(
    field: GaussianField,
    height: int,
    width: int,
    renderer: str,
    render_chunk: int,
    *,
    support_fade: bool,
):
    return render_field(
        field.means,
        field.conics(),
        field.colors,
        field.radii(SIGMA_CUTOFF),
        height,
        width,
        chunk=render_chunk,
        mode=renderer,
        scales=field.scales(),
        rotations=field.rotations,
        support_fade=support_fade,
        sigma_cutoff=SIGMA_CUTOFF,
    )


def _pure_endpoint(field: GaussianField) -> GaussianField:
    if field.opacities is not None or field.color_grads is not None:
        raise RuntimeError("HIER-030 requires constant-color opacity-free fields")
    return GaussianField(
        field.means.detach().clone(),
        field.log_scales.detach().clone(),
        field.rotations.detach().clone(),
        field.colors.detach().clone(),
    )


def _initial_field(
    source: np.ndarray,
    mask: np.ndarray,
    mode: str,
    count: int,
    seed: int,
    args: argparse.Namespace,
) -> GaussianField:
    contained = mode == "masked_contained"
    config = _init_config(count, seed, contained=contained)
    tensor_config = StructureTensorConfig()
    if contained:
        return build_masked_field(
            _objective(source, mask, mode),
            mask,
            config,
            tensor_config,
            device=args.device,
            sigma_cutoff=SIGMA_CUTOFF,
            mask_margin=MASK_MARGIN,
            contain=True,
            cap_mode="anisotropic",
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
    contained = mode == "masked_contained"
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    objective = _objective(source, mask, mode)
    init_started = time.perf_counter()
    initial = _initial_field(source, mask, mode, count, args.seed, args)
    init_seconds = time.perf_counter() - init_started
    initial_audit = initial.detached()
    target_tensor = torch.as_tensor(objective, device=args.device, dtype=torch.float32)
    config = _fit_config(args, renderer, count, contained=contained)
    torch.cuda.reset_peak_memory_stats()
    wall_started = time.perf_counter()
    print(f"[{mode}] fit {renderer} N={count} / {ITERS}", flush=True)
    result = fit(
        initial,
        target_tensor,
        config,
        mask=mask if contained else None,
        verbose=False,
    )
    training_field = result["field"].detached()
    endpoint = _pure_endpoint(training_field)
    with torch.no_grad():
        rendered = _render(
            endpoint,
            source.shape[0],
            source.shape[1],
            renderer,
            args.render_chunk,
            support_fade=contained,
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
        "trajectory": h22._trajectory_baseline(result, ITERS),
        "renderer_calls": renderer_calls,
        "normalized_calls": renderer_calls if renderer == "cuda" else 0,
        "additive_numerator_calls": renderer_calls if renderer == "cuda_additive" else 0,
        "additive_denominator_calls": 0,
        "selected_step": int(result["selected_iter"]),
        "completed": int(result["iterations_run"]) == ITERS,
        "method_status": (
            "completed" if int(result["iterations_run"]) == ITERS else "incomplete"
        ),
        "history": {
            "history": result["history"],
            "checkpoint_history": result["checkpoint_history"],
        },
        "endpoint_parity": endpoint_parity,
        "semantic_family": (
            "normalized_weighted_sum_v1" if renderer == "cuda" else "additive_rgb_peak_one_v1"
        ),
        "renderer": renderer,
        "support_fade": contained,
        "fit_seconds": float(result["fit_seconds"]),
        "wall_fit_seconds": wall_seconds,
        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "attempted_steps": ITERS,
        "gaussian_row_updates": count * ITERS,
        "init_seconds": init_seconds,
        "initial_field_digest": h24._field_digest(initial_audit),
        "preprojection_endpoint_digest": h24._field_digest(endpoint),
        "audit_initial_field": initial_audit,
        "audit_training_field": training_field,
        "fit_config": config,
        "objective_target": objective,
    }


def _base_method(method: dict[str, object]) -> dict[str, object]:
    result = h24._base_method(method)
    result["pursuit_result"] = None
    result["pursuit_seconds"] = 0.0
    result["base_projection_final_digest"] = result["final_field_digest"]
    return result


def _project_method(
    incoming: dict[str, object],
    source: np.ndarray,
    mask: np.ndarray,
    mode: str,
    args: argparse.Namespace,
) -> dict[str, object]:
    contained = mode == "masked_contained"
    print(f"[{mode}] project N={incoming['field'].n}", flush=True)
    started = time.perf_counter()
    result = project_additive_endpoint(
        incoming["field"],
        incoming["objective_target"],
        config=_projection_config(args, contained=contained),
        device=args.device,
        mask=mask if contained else None,
    )
    projection_seconds = time.perf_counter() - started
    metric_started = time.perf_counter()
    incoming_metrics = h29._selection_metrics(
        np.asarray(incoming["expected"], dtype=np.float32), source, mask, mode, args
    )
    proposal_metrics = h29._selection_metrics(
        result.reconstruction_raw, source, mask, mode, args
    )
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
            "expected": result.reconstruction_raw if decision.selected else incoming["expected"],
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
    return method


def _pursuit_method(
    base: dict[str, object],
    source: np.ndarray,
    mask: np.ndarray,
    mode: str,
    args: argparse.Namespace,
) -> dict[str, object]:
    contained = mode == "masked_contained"
    selection = (
        mask_geometry.erode(mask, TAIL_EROSION_RADIUS) if contained else None
    )
    if selection is not None and not selection.any():
        raise RuntimeError("the frozen eroded pursuit mask is empty")
    print(f"[{mode}] append {TAIL_COUNT} residual-pursuit rows", flush=True)
    result = append_residual_pursuit_gaussians(
        base["field"],
        base["objective_target"],
        config=_tail_config(args, contained=contained),
        selection_mask=selection,
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
                float(base["endpoint_parity"]), result.analytic_render_parity_max_abs
            ),
        }
    )
    return method


def _run_mode(
    mode: str,
    source: np.ndarray,
    mask: np.ndarray,
    args: argparse.Namespace,
    torch,
) -> dict[str, dict[str, object]]:
    normalized_raw = _run_fit(source, mask, mode, 4375, "cuda", args, torch)
    base_raw = _run_fit(source, mask, mode, 6562, "cuda_additive", args, torch)
    cold_raw = _run_fit(source, mask, mode, 7000, "cuda_additive", args, torch)
    normalized = _base_method(normalized_raw)
    projected_base = _project_method(base_raw, source, mask, mode, args)
    projected_base["pursuit_result"] = None
    projected_base["pursuit_seconds"] = 0.0
    projected_base["base_projection_final_digest"] = projected_base["final_field_digest"]
    projected_cold = _project_method(cold_raw, source, mask, mode, args)
    projected_cold["pursuit_result"] = None
    projected_cold["pursuit_seconds"] = 0.0
    projected_cold["base_projection_final_digest"] = projected_cold["final_field_digest"]
    pursuit = _pursuit_method(projected_base, source, mask, mode, args)
    return {
        NORMALIZED_ARM: normalized,
        BASE_ARM: projected_base,
        PURSUIT_ARM: pursuit,
        COLD_ARM: projected_cold,
    }


def _unit_coverage(
    field: GaussianField,
    height: int,
    width: int,
    args: argparse.Namespace,
    torch,
    *,
    support_fade: bool,
) -> np.ndarray:
    unit = GaussianField(
        field.means,
        field.log_scales,
        field.rotations,
        torch.ones_like(field.colors),
    )
    with torch.no_grad():
        rendered = _render(
            unit,
            height,
            width,
            "cuda_additive",
            args.render_chunk,
            support_fade=support_fade,
        )
    return rendered[..., 0].detach().cpu().numpy().astype(np.float32, copy=False)


def _containment_record(
    field: GaussianField,
    reconstruction: np.ndarray,
    mask: np.ndarray,
    mode: str,
    args: argparse.Namespace,
    torch,
) -> tuple[dict[str, object], np.ndarray]:
    coverage = _unit_coverage(
        field,
        mask.shape[0],
        mask.shape[1],
        args,
        torch,
        support_fade=mode == "masked_contained",
    )
    means = field.means.detach().cpu().numpy()
    ix = np.rint(means[:, 0]).astype(np.int64)
    iy = np.rint(means[:, 1]).astype(np.int64)
    in_canvas = (ix >= 0) & (ix < mask.shape[1]) & (iy >= 0) & (iy < mask.shape[0])
    centre_inside = np.zeros(field.n, dtype=bool)
    centre_inside[in_canvas] = mask[iy[in_canvas], ix[in_canvas]]
    outside_coverage = np.abs(coverage[~mask])
    outside_reconstruction = np.abs(reconstruction[~mask])
    record = {
        "schema": REPORT_SCHEMA,
        "applied": mode == "masked_contained",
        "mask_margin_px": MASK_MARGIN if mode == "masked_contained" else None,
        "mask_cap_mode": "anisotropic" if mode == "masked_contained" else None,
        "support_fade": mode == "masked_contained",
        "centres_total": field.n,
        "centres_inside_mask": int(centre_inside.sum()),
        "centres_outside_mask": int((~centre_inside).sum()),
        "centre_inside_fraction": float(centre_inside.mean()),
        "unit_coverage_outside_abs_max": float(outside_coverage.max(initial=0.0)),
        "unit_coverage_outside_nonzero_gt_1e7": int(
            np.count_nonzero(outside_coverage > CONTAINMENT_LIMIT)
        ),
        "reconstruction_outside_abs_max": float(outside_reconstruction.max(initial=0.0)),
        "reconstruction_outside_nonzero_gt_1e7": int(
            np.count_nonzero(outside_reconstruction > CONTAINMENT_LIMIT)
        ),
    }
    record["pass"] = bool(
        mode != "masked_contained"
        or (
            record["centres_outside_mask"] == 0
            and record["unit_coverage_outside_abs_max"] <= CONTAINMENT_LIMIT
            and record["reconstruction_outside_abs_max"] <= CONTAINMENT_LIMIT
        )
    )
    return record, coverage


def _placement_image(
    source: np.ndarray,
    mask: np.ndarray,
    field: GaussianField,
    mode: str,
) -> np.ndarray:
    if mode == "masked_contained":
        canvas = np.repeat((0.16 + 0.34 * mask.astype(np.float32))[..., None], 3, axis=2)
    else:
        canvas = np.clip(source * 0.35, 0.0, 1.0)
    means = field.means.detach().cpu().numpy()
    ix = np.rint(means[:, 0]).astype(np.int64)
    iy = np.rint(means[:, 1]).astype(np.int64)
    valid = (ix >= 0) & (ix < source.shape[1]) & (iy >= 0) & (iy < source.shape[0])
    for dx, dy in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)):
        x = np.clip(ix[valid] + dx, 0, source.shape[1] - 1)
        y = np.clip(iy[valid] + dy, 0, source.shape[0] - 1)
        canvas[y, x] = np.asarray([1.0, 0.12, 0.04], dtype=np.float32)
    return canvas


def _metric_prefix(prefix: str, metrics: dict[str, object]) -> dict[str, object]:
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


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
    args: argparse.Namespace,
    torch,
) -> dict[str, object]:
    count = COUNT_BY_ARM[arm]
    contained = mode == "masked_contained"
    artifact_dir = output_root / "artifacts" / f"C0001__{mode}__s0__{arm}"
    artifact_dir.mkdir(parents=True, exist_ok=False)
    field: GaussianField = method["field"]
    field_path = artifact_dir / "field.gaussian.npz"
    field.save(str(field_path))
    with np.load(field_path) as payload:
        field_keys = sorted(payload.files)
    method["audit_initial_field"].save(str(artifact_dir / "initial.field.gaussian.npz"))
    method["audit_training_field"].save(str(artifact_dir / "training.field.gaussian.npz"))
    decode_started = time.perf_counter()
    cold_field = GaussianField.load(str(field_path), device=args.device)
    decode_seconds = time.perf_counter() - decode_started
    render_started = time.perf_counter()
    with torch.no_grad():
        cold_tensor = _render(
            cold_field,
            source.shape[0],
            source.shape[1],
            method["renderer"],
            args.render_chunk,
            support_fade=contained,
        )
        repeated_tensor = _render(
            cold_field,
            source.shape[0],
            source.shape[1],
            method["renderer"],
            args.render_chunk,
            support_fade=contained,
        )
    torch.cuda.synchronize()
    render_seconds = time.perf_counter() - render_started
    cold = cold_tensor.detach().cpu().numpy().astype(np.float32, copy=False)
    repeated = repeated_tensor.detach().cpu().numpy().astype(np.float32, copy=False)
    expected = np.asarray(method["expected"], dtype=np.float32)
    maintained_parity = float(
        np.max(np.abs(cold.astype(np.float64) - expected.astype(np.float64)))
    )
    repeated_parity = float(
        np.max(np.abs(repeated.astype(np.float64) - cold.astype(np.float64)))
    )
    metric_started = time.perf_counter()
    full_metrics, foreground_metrics = h29._metric_domains(cold, source, mask, args)
    metric_seconds = time.perf_counter() - metric_started
    primary = full_metrics if mode == "full_frame" else foreground_metrics
    bounds = h29._save_visuals(
        artifact_dir,
        source,
        cold,
        mask,
        "full_frame" if mode == "full_frame" else "masked_foreground",
        args.error_scale,
    )
    containment, coverage = _containment_record(
        cold_field, cold, mask, mode, args, torch
    )
    _write_json(artifact_dir / "containment.json", containment)
    save_image(str(artifact_dir / "placement.png"), _placement_image(source, mask, field, mode))
    coverage_scale = max(float(coverage.max(initial=0.0)), 1e-8)
    save_image(
        str(artifact_dir / "unit_coverage.png"),
        np.clip(coverage / coverage_scale, 0.0, 1.0),
    )
    outside_support = np.repeat(
        (np.abs(coverage) * (~mask)).astype(np.float32)[..., None], 3, axis=2
    )
    save_error_heatmap(
        str(artifact_dir / "outside_support.png"),
        outside_support,
        scale=max(1.0, args.error_scale),
    )
    h22._write_curve(
        artifact_dir / "learning_curve.svg", method["trajectory"], f"C0001 {mode} {arm}"
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
            "mask_applied": contained,
            "objective_mode": mode,
            "support_fade_alpha": 1.0 if contained else 0.0,
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
            "config": asdict(_tail_config(args, contained=contained)),
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
            "selection_erosion_radius_px": TAIL_EROSION_RADIUS if contained else None,
            "residual_scan_pixel_evaluations": (
                pursuit_result.residual_scan_pixel_evaluations
            ),
            "tail_kernel_pixel_updates": pursuit_result.tail_kernel_pixel_updates,
            "analytic_render_parity_max_abs": pursuit_result.analytic_render_parity_max_abs,
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
            "training_payload_stripped": True,
            "mask_contained": contained,
            "containment_path": "CORE-010/011" if contained else None,
        },
    )
    _write_json(
        artifact_dir / "config.json",
        {
            "schema": REPORT_SCHEMA,
            "mode": mode,
            "arm": arm,
            "seed": args.seed,
            "count": count,
            "init_count": FIT_COUNT_BY_ARM[arm],
            "init": asdict(
                _init_config(FIT_COUNT_BY_ARM[arm], args.seed, contained=contained)
            ),
            "fit": asdict(
                _fit_config(
                    args,
                    "cuda" if arm == NORMALIZED_ARM else "cuda_additive",
                    FIT_COUNT_BY_ARM[arm],
                    contained=contained,
                )
            ),
            "projection": (
                asdict(_projection_config(args, contained=contained))
                if arm != NORMALIZED_ARM
                else None
            ),
            "safety": asdict(h24._safety_config()) if arm != NORMALIZED_ARM else None,
            "pursuit": (
                asdict(_tail_config(args, contained=contained)) if arm == PURSUIT_ARM else None
            ),
        },
    )
    np.savez_compressed(
        artifact_dir / "analysis.npz",
        crop_bounds=np.asarray(bounds, dtype=np.int32),
        foreground_bounds=np.asarray(h29._foreground_bounds(mask), dtype=np.int32),
        mask=mask,
        reconstruction_raw=cold,
        unit_coverage=coverage,
        full_error_raw=cold.astype(np.float32) - source.astype(np.float32),
        objective_error_raw=(cold.astype(np.float32) - source.astype(np.float32))
        * (np.ones((*mask.shape, 1), dtype=np.float32) if not contained else mask[:, :, None]),
    )
    pure_additive = arm in PURE_ADDITIVE_ARMS
    row: dict[str, object] = {
        "schema": REPORT_SCHEMA,
        "status": "diagnostic",
        "phase": "exposed_single_image_7k_containment_diagnostic",
        "image": "C0001",
        "mode": mode,
        "objective_domain": (
            "full_frame" if mode == "full_frame" else "black_matted_foreground_crop"
        ),
        "arm": arm,
        "seed": args.seed,
        "semantic_family": method["semantic_family"],
        "renderer": method["renderer"],
        "support_fade": contained,
        "sigma_cutoff": SIGMA_CUTOFF,
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
        "active_pixels": int(mask.sum()) if contained else int(mask.size),
        "mask_active_pixels": int(mask.sum()),
        "mask_active_fraction": float(mask.mean()),
        "target_gaussians": count,
        "fit_gaussians": FIT_COUNT_BY_ARM[arm],
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
        "mask_payload_present": any("mask" in key.lower() for key in field_keys),
        "scale_cap_payload_present": "scale_max" in field_keys,
        "pure_additive_endpoint": pure_additive,
        "four_array_endpoint_exact": set(field_keys) == FOUR_ARRAY_KEYS,
        "training_payload_present": set(field_keys) != FOUR_ARRAY_KEYS,
        "mask_contained": contained,
        "method_status": method["method_status"],
        "completed": method["completed"],
        "selected_step": method["selected_step"],
        "endpoint_internal_parity_max_abs": method["endpoint_parity"],
        "attempted_steps": ITERS,
        "gaussian_row_updates": GAUSSIAN_ROW_UPDATES_BY_ARM[arm],
        "renderer_calls_fit": method["renderer_calls"],
        "normalized_calls_fit": method["normalized_calls"],
        "additive_numerator_calls_fit": method["additive_numerator_calls"],
        "additive_denominator_calls_fit": method["additive_denominator_calls"],
        "init_seconds": method["init_seconds"],
        "fit_seconds": method["fit_seconds"],
        "wall_fit_seconds": method["wall_fit_seconds"],
        "cold_decode_seconds": decode_seconds,
        "render_seconds": render_seconds,
        "metric_seconds": metric_seconds,
        "projection_seconds": method["projection_seconds"],
        "projection_metric_seconds": method["projection_metric_seconds"],
        "pursuit_seconds": method.get("pursuit_seconds", 0.0),
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
        "containment_pass": containment["pass"],
        "centres_inside_mask": containment["centres_inside_mask"],
        "centres_outside_mask": containment["centres_outside_mask"],
        "unit_coverage_outside_abs_max": containment["unit_coverage_outside_abs_max"],
        "reconstruction_outside_abs_max": containment["reconstruction_outside_abs_max"],
        "containment_history_path": str(
            (artifact_dir / "containment.json").relative_to(output_root)
        ),
        "projection_applied": method["projection_applied"],
        "projection_mask_applied": contained and method["projection_applied"],
        "projection_selected": method["projection_selected"],
        "projection_reason": method["projection_reason"],
        "projection_clauses": method["projection_clauses"],
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
        "base_projection_final_digest": method["base_projection_final_digest"],
        "final_field_digest": method["final_field_digest"],
        "initial_field_digest": method["initial_field_digest"],
        "preprojection_endpoint_digest": method["preprojection_endpoint_digest"],
        "pursuit_applied": pursuit_result is not None,
        "pursuit_base_count": None if pursuit_result is None else pursuit_result.base_count,
        "pursuit_tail_count": None if pursuit_result is None else pursuit_result.tail_count,
        "pursuit_base_field_digest": (
            None if pursuit_result is None else pursuit_result.base_field_digest
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
        "pursuit_history_path": str(pursuit_path.relative_to(output_root)),
        "pursuit_history_sha256": h22.report_utils._sha256(pursuit_path),
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
        if columns:
            writer.writeheader()
            writer.writerows(rows)


def _historical_rows(args: argparse.Namespace, output_root: Path) -> list[dict[str, object]]:
    metrics_path = args.hier029_report / "metrics.json"
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    rows = payload.get("rows", [])
    selected = [
        {
            key: row.get(key)
            for key in (
                "mode",
                "arm",
                "n_gaussians",
                "psnr_db",
                "ms_ssim",
                "lpips",
                "full_psnr_db",
                "foreground_psnr_db",
                "artifact_pixel_rmse_max",
                "artifact_patch_rmse_max_7",
            )
        }
        for row in rows
    ]
    record = {
        "schema": REPORT_SCHEMA,
        "role": "historical_literal_count_context",
        "source_metrics_path": str(metrics_path.resolve()),
        "source_metrics_sha256": h22.report_utils._sha256(metrics_path),
        "rows": selected,
    }
    _write_json(output_root / "historical_hier029.json", record)
    return selected


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


def _decision(
    rows: list[dict[str, object]], historical: list[dict[str, object]]
) -> dict[str, object]:
    indexed = {(str(row["mode"]), str(row["arm"])): row for row in rows}
    expected = {(mode, arm) for mode in MODES for arm in ARMS}
    all_cells = set(indexed) == expected and len(rows) == len(expected)
    containment = bool(
        all_cells
        and all(
            row["containment_pass"]
            and row["centres_outside_mask"] == 0
            and row["unit_coverage_outside_abs_max"] <= CONTAINMENT_LIMIT
            and row["reconstruction_outside_abs_max"] <= CONTAINMENT_LIMIT
            for row in rows
            if row["mode"] == "masked_contained"
        )
    )
    integrity = bool(
        all_cells
        and containment
        and all(
            row["completed"]
            and row["n_gaussians"] == COUNT_BY_ARM[str(row["arm"])]
            and row["four_array_endpoint_exact"]
            and row["maintained_render_parity_max_abs"] <= PARITY_LIMIT
            and row["repeated_render_parity_max_abs"] <= PARITY_LIMIT
            and row["endpoint_internal_parity_max_abs"] <= PARITY_LIMIT
            for row in rows
        )
    )
    historical_index = {
        (str(row["mode"]), str(row["arm"])): row for row in historical
    }
    historical_pursuit = {
        "full_frame": historical_index.get(
            ("full_frame", "residual_pursuit_additive_n1024")
        ),
        "masked_contained": historical_index.get(
            ("masked_foreground", "residual_pursuit_additive_n1024")
        ),
    }
    comparisons = {}
    for mode in MODES:
        row = indexed.get((mode, PURSUIT_ARM))
        old = historical_pursuit[mode]
        comparisons[mode] = {
            "pursuit_vs_normalized_primary_psnr_db": _delta(
                indexed, mode, PURSUIT_ARM, NORMALIZED_ARM, "psnr_db"
            ),
            "pursuit_vs_cold_n7000_primary_psnr_db": _delta(
                indexed, mode, PURSUIT_ARM, COLD_ARM, "psnr_db"
            ),
            "pursuit_vs_base_n6562_primary_psnr_db": _delta(
                indexed, mode, PURSUIT_ARM, BASE_ARM, "psnr_db"
            ),
            "pursuit_7k_vs_hier029_literal_primary_psnr_db": (
                None if row is None or old is None else float(row["psnr_db"]) - float(old["psnr_db"])
            ),
        }
    return {
        "schema": REPORT_SCHEMA,
        "status": "diagnostic",
        "all_cells_present": all_cells,
        "integrity_pass": integrity,
        "containment_pass": containment,
        "comparisons": comparisons,
        "visual_review": "pending_producer_audit",
        "overall_pass": integrity,
        "formal_claim_ready": False,
        "interpretation": (
            "Complete exposed Janelle 7k capacity/containment diagnostic; inspect metrics and visuals."
            if integrity
            else "Diagnostic is incomplete or failed integrity; inspect attempts and containment."
        ),
        "density_context": {
            "evaluation_pixels": EVALUATION_SHAPE[0] * EVALUATION_SHAPE[1],
            "literal_hier028_final_count": 1024,
            "scaled_final_count": 7000,
            "max_side_density_equivalent_count_approx": 57600,
        },
        "claim_limits": [
            "one exposed image and one seed",
            "1200x1038 evaluation raster, not native 5328x4608",
            "7k is not density-equivalent to HIER-028 at max-side 160",
            "dirty-source producer diagnostic",
            "no default, rate, or publication claim",
        ],
    }


def _fmt(value: object, digits: int = 4) -> str:
    return "—" if value is None else f"{float(value):.{digits}f}"


def _write_report(
    output_root: Path,
    rows: list[dict[str, object]],
    attempts: list[dict[str, object]],
    historical: list[dict[str, object]],
    decision: dict[str, object],
) -> None:
    table_rows = []
    cards = []
    for row in rows:
        artifact = escape(str(row["artifact_dir"]))
        table_rows.append(
            "<tr>"
            f"<td>{escape(str(row['mode']))}</td><td>{escape(str(row['arm']))}</td>"
            f"<td>{int(row['n_gaussians'])}</td><td>{_fmt(row['psnr_db'], 3)}</td>"
            f"<td>{_fmt(row['ms_ssim'], 5)}</td><td>{_fmt(row['lpips'], 5)}</td>"
            f"<td>{_fmt(row['full_psnr_db'], 3)}</td>"
            f"<td>{_fmt(row['foreground_psnr_db'], 3)}</td>"
            f"<td>{_fmt(row['artifact_pixel_rmse_max'], 4)}</td>"
            f"<td>{_fmt(row['artifact_patch_rmse_max_7'], 4)}</td>"
            f"<td>{int(row['centres_outside_mask'])}</td>"
            f"<td>{float(row['unit_coverage_outside_abs_max']):.2e}</td>"
            f"<td>{float(row['reconstruction_outside_abs_max']):.2e}</td>"
            f"<td>{float(row['fit_seconds']):.1f}s</td>"
            f"<td><a href='{artifact}/row.json'>row</a> · "
            f"<a href='{artifact}/field.gaussian.npz'>field</a> · "
            f"<a href='{artifact}/containment.json'>containment</a> · "
            f"<a href='{artifact}/fit_history.json'>fit</a> · "
            f"<a href='{artifact}/projection_history.json'>projection</a> · "
            f"<a href='{artifact}/pursuit_history.json'>pursuit</a></td></tr>"
        )
        cards.append(
            f"<article class='card'><h3>{escape(str(row['mode']))} · "
            f"{escape(str(row['arm']))}</h3>"
            f"<p>Objective PSNR {_fmt(row['psnr_db'], 3)} dB · MS-SSIM "
            f"{_fmt(row['ms_ssim'], 5)} · LPIPS {_fmt(row['lpips'], 5)} · "
            f"outside centres {int(row['centres_outside_mask'])} · outside support "
            f"{float(row['unit_coverage_outside_abs_max']):.2e}</p>"
            "<div class='images'>"
            f"<figure><a href='{artifact}/source.png'><img src='{artifact}/source.png'></a>"
            "<figcaption>source</figcaption></figure>"
            f"<figure><a href='{artifact}/reconstruction.png'><img src='{artifact}/reconstruction.png'></a>"
            "<figcaption>raw reconstruction</figcaption></figure>"
            f"<figure><a href='{artifact}/error.png'><img src='{artifact}/error.png'></a>"
            "<figcaption>full-frame error ×4</figcaption></figure>"
            f"<figure><a href='{artifact}/foreground_reconstruction.png'><img "
            f"src='{artifact}/foreground_reconstruction.png'></a>"
            "<figcaption>black-matted foreground</figcaption></figure>"
            f"<figure><a href='{artifact}/foreground_error.png'><img "
            f"src='{artifact}/foreground_error.png'></a>"
            "<figcaption>foreground error ×4</figcaption></figure>"
            f"<figure><a href='{artifact}/placement.png'><img src='{artifact}/placement.png'></a>"
            "<figcaption>Gaussian centres (red)</figcaption></figure>"
            f"<figure><a href='{artifact}/unit_coverage.png'><img src='{artifact}/unit_coverage.png'></a>"
            "<figcaption>unit-coefficient support coverage</figcaption></figure>"
            f"<figure><a href='{artifact}/outside_support.png'><img src='{artifact}/outside_support.png'></a>"
            "<figcaption>support outside mask</figcaption></figure>"
            f"<figure><a href='{artifact}/reconstruction_crop.png'><img "
            f"src='{artifact}/reconstruction_crop.png'></a>"
            "<figcaption>worst objective crop</figcaption></figure>"
            f"<a class='hidden-link' href='{artifact}/source_crop.png'>source crop</a>"
            f"<a class='hidden-link' href='{artifact}/error_crop.png'>error crop</a>"
            "</div></article>"
        )
    historical_rows = "".join(
        "<tr>"
        f"<td>{escape(str(row['mode']))}</td><td>{escape(str(row['arm']))}</td>"
        f"<td>{int(row['n_gaussians'])}</td><td>{_fmt(row['psnr_db'], 3)}</td>"
        f"<td>{_fmt(row['ms_ssim'], 5)}</td><td>{_fmt(row['lpips'], 5)}</td></tr>"
        for row in historical
    )
    errors = [attempt for attempt in attempts if attempt.get("status") != "ok"]
    error_html = (
        "<p>No execution errors were recorded.</p>"
        if not errors
        else f"<pre>{escape(json.dumps(errors, indent=2, sort_keys=True))}</pre>"
    )
    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>HIER-030 Janelle 7k contained-mask diagnostic</title><style>
:root{{--ink:#18202a;--muted:#66717f;--line:#d8dee6;--panel:#f6f8fa}}
body{{font-family:system-ui,sans-serif;color:var(--ink);margin:2rem;max-width:2200px}}
h1,h2{{line-height:1.15}}p{{max-width:1200px}}code,pre{{white-space:pre-wrap}}
table{{border-collapse:collapse;font-size:.86rem}}th,td{{border:1px solid var(--line);padding:.4rem}}
th{{background:var(--panel);position:sticky;top:0}}.cards{{display:grid;gap:1.5rem}}
.card{{border-top:2px solid var(--line)}}.images{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:.7rem}}
figure{{margin:0}}img{{width:100%;height:auto;display:block;background:#111}}figcaption{{color:var(--muted);font-size:.84rem}}
.hidden-link{{font-size:.8rem}}@media(max-width:900px){{.images{{grid-template-columns:1fr}}}}
</style></head><body><h1>HIER-030 · Janelle C0001 7k + contained mask</h1>
<p><strong>Corrected diagnostic.</strong> HIER-029 used HIER-028's literal 640/960/1024 counts,
not approximately 7k, and its mask affected the loss without constraining the fitted geometry.
This report scales the same count ratios to 4,375 / 6,562+438 / 7,000 and requires every masked
Gaussian centre and its C0 three-sigma support to stay inside the mask.</p>
<p>The raster is 1200×1038, the repository's Janelle evaluation regime; the hash-bound camera
source is 5328×4608. Seven thousand rows are still far below the roughly 57,600 rows needed to
preserve HIER-028's max-side scaling density. The 500-step schedule is intentionally unchanged.</p>
<p><code>{escape(_command())}</code></p>
<p><a href="input/source.png">evaluation source</a> · <a href="input/mask.png">mask</a> ·
<a href="config.json">config</a> · <a href="decision.json">decision</a> ·
<a href="metrics.json">JSON</a> · <a href="metrics.jsonl">JSONL</a> ·
<a href="metrics.csv">CSV</a> · <a href="attempts.json">attempts</a> ·
<a href="historical_hier029.json">HIER-029 context</a> · <a href="manifest.json">manifest</a></p>
<h2>Diagnostic decision</h2><pre>{escape(json.dumps(decision, indent=2, sort_keys=True))}</pre>
<h2>Corrected 7k matrix</h2><div style="overflow:auto"><table><tr><th>mode</th><th>arm</th><th>N</th>
<th>objective PSNR</th><th>objective MS-SSIM</th><th>objective LPIPS</th><th>full PSNR</th>
<th>foreground PSNR</th><th>pixel max</th><th>7×7 max</th><th>centres outside</th>
<th>support outside max</th><th>reconstruction outside max</th><th>fit</th><th>artifacts</th></tr>
{"".join(table_rows)}</table></div>
<h2>Historical literal-count HIER-029 context</h2><p>These rows are copied from the immutable
HIER-029 metrics ledger and are not rerun cells.</p><table><tr><th>mode</th><th>arm</th><th>N</th>
<th>PSNR</th><th>MS-SSIM</th><th>LPIPS</th></tr>{historical_rows}</table>
<h2>Execution errors</h2>{error_html}
<h2>Evaluation-size visual comparisons, placement, support, and errors</h2>
<div class="cards">{"".join(cards)}</div></body></html>"""
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
        raise SystemExit(f"completed HIER-030 bundle is immutable: {args.out}")
    if args.out.exists() and any(args.out.iterdir()) and not args.resume:
        raise SystemExit(f"output directory is non-empty; use --resume: {args.out}")
    args.out.mkdir(parents=True, exist_ok=True)
    if h22.report_utils._sha256(args.image) != SOURCE_SHA256:
        raise SystemExit("Janelle source SHA-256 differs from the frozen HIER-030 binding")
    if h22.report_utils._sha256(args.mask) != MASK_SHA256:
        raise SystemExit("Janelle mask SHA-256 differs from the frozen HIER-030 binding")

    import torch

    if not torch.cuda.is_available():
        raise SystemExit("frozen HIER-030 diagnostic requires CUDA")
    source, mask, raster = h22.report_utils._load_evaluation_raster(
        args.image, args.mask, max_side=args.max_side, mask_threshold=0.5
    )
    if mask is None:
        raise RuntimeError("HIER-030 mask was not loaded")
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
    historical = _historical_rows(args, args.out)
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
                "margin_px": MASK_MARGIN,
                "cap_mode": "anisotropic",
                "tail_selection_erosion_radius_px": TAIL_EROSION_RADIUS,
            },
            "raster": raster,
            "modes": list(MODES),
            "arms": list(ARMS),
            "counts": COUNT_BY_ARM,
            "fit_counts": FIT_COUNT_BY_ARM,
            "tail_count": TAIL_COUNT,
            "gaussian_row_updates": GAUSSIAN_ROW_UPDATES_BY_ARM,
            "structure_tensor": asdict(StructureTensorConfig()),
            "historical_hier029_metrics_sha256": h22.report_utils._sha256(
                args.hier029_report / "metrics.json"
            ),
            "claim_limits": [
                "one exposed Janelle image and seed zero",
                "max-side-1200 evaluation regime, not native-5328",
                "7k is not density-equivalent to max-side-160 HIER-028",
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
        "HIER-030 hash-bound Janelle source and mask decoded after protocol freeze.\n",
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
        fit_error = None
        try:
            methods = _run_mode(mode, source, mask, args, torch)
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
                    {"schema": REPORT_SCHEMA, "status": "diagnostic", "attempts": attempts},
                )
                torch.cuda.empty_cache()

    decision = _decision(rows, historical)
    _write_json(args.out / "decision.json", decision)
    _write_report(args.out, rows, attempts, historical, decision)
    (args.out / "COMPLETED").write_text(
        "HIER-030 exposed Janelle 7k containment diagnostic complete; do not overwrite.\n",
        encoding="utf-8",
    )
    _write_manifest(args.out)
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
