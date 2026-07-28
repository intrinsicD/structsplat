#!/usr/bin/env python3
"""Equal-row FIT-032 screen on the saved exposed Janelle C0001 field.

This assay is intentionally downstream of the maintained pipeline. It rescales the saved native
field and authoritative calibrated crop to a bounded development resolution, then compares:

* gauge-lifted residual color dipoles,
* FIT-031's error-ranked isotropic births, and
* the repository's responsibility-ranked moment-preserving split.

Every arm starts from the same field, adds the same net row count, uses the same renderer and
protected metric vector, and receives the same fixed-topology recovery work.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
from typing import Any

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
for _root in (
    REPOSITORY_ROOT,
    REPOSITORY_ROOT / "src",
    REPOSITORY_ROOT / "deprecated_scripts",
):
    _text = str(_root)
    if _text not in sys.path:
        sys.path.insert(0, _text)

from benchmarks.gauge_lifted_dipole import (  # noqa: E402
    DipoleSelection,
    apply_dipole_split,
    select_residual_dipoles,
)
from structsplat.config import FitConfig  # noqa: E402
from structsplat.fit import (  # noqa: E402
    _MaskConstraint,
    _moment_preserving_duplicate_indices,
    _responsibility_error_density_scores,
)
from structsplat.gaussians import GaussianField  # noqa: E402
from structsplat.safe_schedule import (  # noqa: E402
    SafeScheduleConfig,
    _birth_components,
    _phase_fit_config,
    _safe_fit_block,
    evaluate_quality,
)


DEFAULT_CAPTURE_ROOT = Path(
    "/home/alex/Dropbox/Work/Janelle/2025_03_07_stage_with_fabric"
)
DEFAULT_REALTIME_ROOT = Path("/home/alex/Documents/realtime-gs")
DEFAULT_FIELD = (
    REPOSITORY_ROOT
    / "runs/janelle_C0001_transactional_candidates_factorial_20260723"
    / "pareto_checkpoint/C0001_safe_commit_full.npz"
)
DEFAULT_OUT = REPOSITORY_ROOT / "runs/fit032_janelle_dipole_20260728"
SOURCE_FILES = (
    "benchmarks/gauge_lifted_dipole.py",
    "scripts/experiments/fit032_janelle_dipole_screen.py",
    "tasks/FIT-032-gauge-lifted-residual-dipoles.md",
)
ARMS = ("dipole", "error_birth", "moment_split")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
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


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _save_rgb(path: Path, image: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    array = (
        image.detach().cpu().clamp(0.0, 1.0).mul(255.0).round()
        .to(torch.uint8).numpy()
    )
    Image.fromarray(array, mode="RGB").save(path)


def _prepare_janelle(args: argparse.Namespace) -> dict[str, Any]:
    import fit_janelle_complete_refinement as legacy

    bridge = legacy._load_bridge(args.realtime_root.resolve())
    source = legacy._find_source(
        bridge,
        args.capture_root.resolve(),
        args.frame,
        args.view_id,
    )
    prepared = bridge.CONVERTER.prepare_source_view(source)
    if prepared.alpha_crop is None:
        raise RuntimeError("prepared Janelle source lost its authoritative mask")

    rgb_native = prepared.rgb.detach().cpu().clamp(0.0, 1.0)
    mask_native = prepared.alpha_crop.detach().cpu().bool()
    native_height, native_width = mask_native.shape
    scale = min(1.0, float(args.max_side) / max(native_height, native_width))
    width = max(1, round(native_width * scale))
    height = max(1, round(native_height * scale))

    rgb_u8 = rgb_native.mul(255.0).round().to(torch.uint8).numpy()
    mask_u8 = mask_native.to(torch.uint8).mul(255).numpy()
    rgb_resized = np.asarray(
        Image.fromarray(rgb_u8, mode="RGB").resize(
            (width, height), Image.Resampling.LANCZOS
        ),
        dtype=np.float32,
    ) / 255.0
    mask_resized = np.asarray(
        Image.fromarray(mask_u8, mode="L").resize(
            (width, height), Image.Resampling.NEAREST
        ),
        dtype=np.uint8,
    ) > 127
    target = rgb_resized * mask_resized[..., None].astype(np.float32)
    return {
        "source": source,
        "prepared": prepared,
        "target": target,
        "mask": mask_resized,
        "native_size": [native_width, native_height],
        "fit_size": [width, height],
        "scale": [width / native_width, height / native_height],
        "image_path": source.rgb_path,
        "mask_path": source.mask_path,
        "fit_window": list(prepared.fit_window),
        "field_path": args.field.resolve(),
        "source_kind": "calibrated_native_field_replay",
    }


def _prepare_current_job(path: Path) -> dict[str, Any]:
    job = path.resolve()
    config = json.loads((job / "config.json").read_text(encoding="utf-8"))
    field_path = job / "field.npz"
    target_path = job / "target.png"
    if not field_path.is_file() or not target_path.is_file():
        raise FileNotFoundError(f"current-pipeline job is incomplete: {job}")
    target = np.asarray(
        Image.open(target_path).convert("RGB"),
        dtype=np.float32,
    ) / 255.0
    height, width = target.shape[:2]
    mask_path = Path(config["source"]["mask_path"])
    mask = np.asarray(
        Image.open(mask_path).convert("L").resize(
            (width, height),
            Image.Resampling.NEAREST,
        ),
        dtype=np.uint8,
    ) > 127
    return {
        "source": None,
        "prepared": None,
        "target": target,
        "mask": mask,
        "native_size": [width, height],
        "fit_size": [width, height],
        "scale": [1.0, 1.0],
        "image_path": Path(config["source"]["path"]),
        "mask_path": mask_path,
        "fit_window": None,
        "field_path": field_path,
        "source_kind": "current_pipeline_job",
        "base_job": job,
        "base_config": job / "config.json",
    }


def _scaled_field(
    path: Path,
    device: torch.device,
    scale_x: float,
    scale_y: float,
) -> GaussianField:
    field = GaussianField.load(str(path), device=device)
    scale = torch.tensor(
        [scale_x, scale_y],
        device=device,
        dtype=field.means.dtype,
    )
    means = (field.means.detach() + 0.5) * scale - 0.5
    log_scales = field.log_scales.detach() + torch.log(scale)
    scale_max = (
        None
        if field.scale_max is None
        else field.scale_max.detach() * scale
    )
    filter_variance = (
        None
        if field.filter_variance is None
        else field.filter_variance.detach() * float(scale_x * scale_y)
    )
    return GaussianField(
        means=means,
        log_scales=log_scales,
        rotations=field.rotations.detach().clone(),
        colors=field.colors.detach().clone(),
        opacities=(
            None if field.opacities is None else field.opacities.detach().clone()
        ),
        scale_max=scale_max,
        color_grads=(
            None if field.color_grads is None else field.color_grads.detach().clone()
        ),
        background_mask=(
            None
            if field.background_mask is None
            else field.background_mask.detach().clone()
        ),
        filter_variance=filter_variance,
    )


def _base_config(args: argparse.Namespace) -> FitConfig:
    return FitConfig(
        iters=1,
        renderer=args.renderer,
        render_chunk=512,
        pixel_loss="l2",
        ssim_weight=0.0,
        loss_weighting="mask",
        mask_contain=True,
        mask_margin=float(args.mask_margin),
        mask_cap_mode="anisotropic",
        mask_cap_refresh_every=100,
        mask_undercoverage_band=float(args.boundary_band),
        mask_undercoverage_tau=float(args.coverage_tau),
        mask_undercoverage_every=8,
        support_fade=True,
        checkpoint_policy="terminal",
        split_scale=0.35,
        split_oversample=8.0,
        split_min_spacing=1.0,
        densify_max_axis_ratio=6.0,
        densify_coherence_power=1.0,
        color_solve_lambda=1e-4,
        color_solve_maxiter=32,
        compute_lpips=False,
        log_every=1,
    )


def _gaussian_blur(image: torch.Tensor, sigma: float = 1.5) -> torch.Tensor:
    radius = int(math.ceil(3.0 * sigma))
    coordinate = torch.arange(
        -radius,
        radius + 1,
        device=image.device,
        dtype=image.dtype,
    )
    kernel = torch.exp(-0.5 * (coordinate / sigma).square())
    kernel = kernel / kernel.sum()
    value = image.permute(2, 0, 1).unsqueeze(0)
    horizontal = kernel.view(1, 1, 1, -1).expand(3, 1, 1, -1)
    vertical = kernel.view(1, 1, -1, 1).expand(3, 1, -1, 1)
    value = F.conv2d(
        F.pad(value, (radius, radius, 0, 0), mode="reflect"),
        horizontal,
        groups=3,
    )
    value = F.conv2d(
        F.pad(value, (0, 0, radius, radius), mode="reflect"),
        vertical,
        groups=3,
    )
    return value[0].permute(1, 2, 0)


@torch.no_grad()
def _evaluate(
    field: GaussianField,
    target: torch.Tensor,
    mask: torch.Tensor,
    cfg: FitConfig,
    constraint: _MaskConstraint,
    coverage_tau: float,
) -> tuple[dict[str, Any], torch.Tensor, Any]:
    quality, rendered = evaluate_quality(
        field,
        target,
        mask,
        cfg,
        constraint,
        coverage_tau,
    )
    residual = rendered - target
    highpass = residual - _gaussian_blur(residual)
    deep = (
        constraint.sdf_flat.reshape(mask.shape)
        > float(constraint.margin) + 6.0
    )
    if not bool(deep.any()):
        deep = mask
    record = quality.to_dict()
    record.update(
        {
            "fine_detail_highpass_mse": float(
                highpass[deep].square().mean()
            ),
            "fine_detail_pixels": int(deep.sum()),
            "render_min": float(rendered.min()),
            "render_max": float(rendered.max()),
        }
    )
    return record, rendered, quality


def _selection_prefix(
    selection: DipoleSelection,
    count: int,
) -> DipoleSelection:
    return replace(
        selection,
        parents=selection.parents[:count],
        displacement=selection.displacement[:count],
        contrast=selection.contrast[:count],
        score=selection.score[:count],
        unclipped_score=selection.unclipped_score[:count],
        color_clip=selection.color_clip[:count],
        support_scale=selection.support_scale[:count],
    )


@torch.no_grad()
def _dipole_proposal(
    base: GaussianField,
    selection: DipoleSelection,
    target: torch.Tensor,
    mask: torch.Tensor,
    cfg: FitConfig,
    constraint: _MaskConstraint,
    coverage_tau: float,
) -> tuple[GaussianField, torch.Tensor, dict[str, Any], torch.Tensor, Any]:
    trials = []
    best = None
    for lift_scale in (0.25, 0.5, 1.0):
        trial = apply_dipole_split(
            base,
            selection,
            lift_scale=lift_scale,
        )
        before_means = trial.means.detach().clone()
        before_scales = trial.log_scales.detach().clone()
        constraint.apply(trial, cfg, refresh=True)
        metrics, rendered, quality = _evaluate(
            trial,
            target,
            mask,
            cfg,
            constraint,
            coverage_tau,
        )
        line = {
            "lift_scale": lift_scale,
            "metrics": metrics,
            "constraint_mean_max_abs": float(
                (trial.means - before_means).abs().max()
            ),
            "constraint_log_scale_max_abs": float(
                (trial.log_scales - before_scales).abs().max()
            ),
        }
        trials.append(line)
        candidate = (
            float(metrics["foreground_mse"]),
            float(metrics["fine_detail_highpass_mse"]),
            lift_scale,
            trial,
            rendered,
            quality,
        )
        if best is None or candidate[:3] < best[:3]:
            best = candidate
    assert best is not None
    _, _, chosen_scale, field, rendered, quality = best
    touched = torch.cat(
        [
            selection.parents,
            torch.arange(
                base.n,
                field.n,
                device=base.means.device,
                dtype=torch.long,
            ),
        ]
    )
    metadata = {
        "chosen_lift_scale": chosen_scale,
        "line_search": trials,
        "parents": int(selection.parents.numel()),
        "score_sum": float(selection.score.sum()),
        "score_mean": float(selection.score.mean()),
        "support_scale_min": float(selection.support_scale.min()),
        "support_scale_median": float(selection.support_scale.median()),
        "support_scale_max": float(selection.support_scale.max()),
        "color_clip_fraction": float((selection.color_clip < 1.0).float().mean()),
    }
    chosen_metrics, _, _ = _evaluate(
        field,
        target,
        mask,
        cfg,
        constraint,
        coverage_tau,
    )
    return field, touched, {**metadata, "metrics": chosen_metrics}, rendered, quality


@torch.no_grad()
def _birth_proposal(
    base: GaussianField,
    target: torch.Tensor,
    rendered: torch.Tensor,
    cfg: FitConfig,
    constraint: _MaskConstraint,
    schedule: SafeScheduleConfig,
    count: int,
) -> tuple[GaussianField, torch.Tensor, dict[str, Any]]:
    components, metadata = _birth_components(
        base,
        target,
        rendered,
        cfg,
        constraint,
        schedule,
        count,
        "error_tail",
    )
    if components is None or components.n != count:
        selected = 0 if components is None else components.n
        raise RuntimeError(
            f"error birth produced {selected} rows for requested budget {count}"
        )
    trial = base.append(components)
    constraint.apply(trial, cfg, refresh=True)
    touched = torch.arange(
        base.n,
        trial.n,
        device=base.means.device,
        dtype=torch.long,
    )
    return trial, touched, metadata


@torch.no_grad()
def _moment_proposal(
    base: GaussianField,
    target: torch.Tensor,
    rendered: torch.Tensor,
    cfg: FitConfig,
    constraint: _MaskConstraint,
    count: int,
) -> tuple[GaussianField, torch.Tensor, dict[str, Any]]:
    scores, components = _responsibility_error_density_scores(
        base,
        target,
        rendered,
        cfg,
        support_fade_alpha=1.0,
    )
    footprint = base.scales().detach().prod(dim=1)
    score = scores * torch.sqrt(
        footprint / torch.quantile(footprint, 0.75).clamp_min(1e-8)
    )
    if base.background_mask is not None:
        score = score.masked_fill(
            base.background_mask,
            -float("inf"),
        )
    finite = int(torch.isfinite(score).sum())
    if finite < count:
        raise RuntimeError(
            f"moment control has only {finite} finite parents for budget {count}"
        )
    parents = torch.topk(score, k=count).indices
    split_cfg = replace(
        cfg,
        split_scale=0.60,
        refine_primitive="moment_preserving",
    )
    trial, added = _moment_preserving_duplicate_indices(
        base,
        parents,
        split_cfg,
        target.shape[0],
        target.shape[1],
    )
    if added != count:
        raise RuntimeError(
            f"moment split produced {added} rows for requested budget {count}"
        )
    constraint.apply(trial, cfg, refresh=True)
    touched = torch.cat(
        [
            parents,
            torch.arange(
                base.n,
                trial.n,
                device=base.means.device,
                dtype=torch.long,
            ),
        ]
    )
    return trial, touched, {
        "parents": count,
        "score_mean": float(score[parents].mean()),
        "responsibility_mass_mean": float(
            components["mass"][parents].mean()
        ),
        "split_shrink": 0.60,
    }


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


def _decision(
    baseline_mse: float,
    rows: list[dict[str, Any]],
    budgets: tuple[int, ...],
) -> dict[str, Any]:
    by_key = {
        (row["arm"], int(row["budget"])): row
        for row in rows
    }
    budget_records = []
    passes = 0
    for budget in budgets:
        dipole = by_key[("dipole", budget)]
        controls = [
            by_key[("error_birth", budget)],
            by_key[("moment_split", budget)],
        ]
        dipole_immediate = baseline_mse - float(
            dipole["immediate"]["foreground_mse"]
        )
        control_immediate = max(
            baseline_mse - float(row["immediate"]["foreground_mse"])
            for row in controls
        )
        dipole_recovered = baseline_mse - float(
            dipole["selected"]["foreground_mse"]
        )
        control_recovered = max(
            baseline_mse - float(row["selected"]["foreground_mse"])
            for row in controls
        )
        passed = (
            dipole_immediate > 0.0
            and dipole_immediate >= 2.0 * max(control_immediate, 0.0)
            and dipole_recovered > 0.0
            and dipole_recovered >= 1.5 * max(control_recovered, 0.0)
            and bool(dipole["recovery_accepted"])
        )
        passes += int(passed)
        budget_records.append(
            {
                "budget": budget,
                "dipole_immediate_reduction": dipole_immediate,
                "strongest_control_immediate_reduction": control_immediate,
                "dipole_recovered_reduction": dipole_recovered,
                "strongest_control_recovered_reduction": control_recovered,
                "passed": passed,
            }
        )
    return {
        "rule": (
            "dipole >=2x strongest immediate and >=1.5x strongest recovered "
            "foreground-MSE reduction at >=2/3 budgets, with protected recovery accepted"
        ),
        "budget_records": budget_records,
        "budgets_passed": passes,
        "promote": passes >= 2,
    }


def run(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise RuntimeError("requested CUDA but torch.cuda.is_available() is false")
    budgets = tuple(sorted(set(int(value) for value in args.budgets)))
    if not budgets or budgets[0] <= 0:
        raise ValueError("budgets must contain positive integers")
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        raise RuntimeError(f"output directory is not empty: {out}")
    out.mkdir(parents=True, exist_ok=True)
    source_records = _snapshot_sources(out)

    torch.manual_seed(0)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)

    prepared = (
        _prepare_current_job(args.base_job)
        if args.base_job is not None
        else _prepare_janelle(args)
    )
    width, height = prepared["fit_size"]
    scale_x, scale_y = prepared["scale"]
    target = torch.as_tensor(
        prepared["target"],
        device=device,
        dtype=torch.float32,
    ).contiguous()
    mask_cpu = torch.as_tensor(prepared["mask"], dtype=torch.bool)
    mask = mask_cpu.to(device=device)
    cfg = _base_config(args)
    geometry_scale = min(scale_x, scale_y)
    cfg = replace(
        cfg,
        mask_margin=float(args.mask_margin) * geometry_scale,
        mask_undercoverage_band=float(args.boundary_band) * geometry_scale,
    )
    base = _scaled_field(
        prepared["field_path"],
        device,
        scale_x,
        scale_y,
    )
    if base.opacities is None:
        raise RuntimeError("saved Janelle field has no explicit opacity logits")
    if base.color_grads is not None:
        raise RuntimeError("FIT-032 currently requires constant-color saved fields")
    constraint = _MaskConstraint.from_mask(
        prepared["mask"],
        device,
        target.dtype,
        cfg.sigma_cutoff,
        cfg.mask_margin,
        aa_dilation=cfg.aa_dilation,
        min_scale=0.35 * geometry_scale,
        cap_mode=cfg.mask_cap_mode,
        undercoverage_band=cfg.mask_undercoverage_band,
    )
    constraint.apply(base, cfg, refresh=True)
    baseline, base_render, base_quality = _evaluate(
        base,
        target,
        mask,
        cfg,
        constraint,
        args.coverage_tau,
    )
    _save_rgb(out / "images/target.png", target)
    _save_rgb(out / "images/baseline.png", base_render)

    schedule = SafeScheduleConfig(
        capacity=base.n + max(budgets),
        coverage_tau=float(args.coverage_tau),
        boundary_band=float(args.boundary_band) * geometry_scale,
        pareto_safe_checkpoints=True,
        pareto_checkpoint_every=max(
            1, min(int(args.checkpoint_every), int(args.recovery_steps))
        ),
    )
    recovery_cfg = _phase_fit_config(
        cfg,
        schedule.error_tail,
        int(args.recovery_steps),
    )

    selection_started = time.perf_counter()
    full_selection = select_residual_dipoles(
        base,
        target,
        base_render,
        cfg,
        mask,
        max(budgets),
        trust_radius=float(args.trust_radius),
        max_color_contrast=float(args.max_color_contrast),
        minimum_spacing=float(args.minimum_spacing),
        spacing_scale=float(args.spacing_scale),
    )
    selection_seconds = time.perf_counter() - selection_started
    if int(full_selection.parents.numel()) < max(budgets):
        raise RuntimeError(
            f"dipole selector returned {full_selection.parents.numel()} "
            f"parents for maximum budget {max(budgets)}"
        )

    rows: list[dict[str, Any]] = []
    for budget in budgets:
        for arm in ARMS:
            started = time.perf_counter()
            if arm == "dipole":
                selection = _selection_prefix(full_selection, budget)
                (
                    proposal,
                    touched,
                    proposal_metadata,
                    immediate_render,
                    immediate_quality,
                ) = _dipole_proposal(
                    base,
                    selection,
                    target,
                    mask,
                    cfg,
                    constraint,
                    args.coverage_tau,
                )
                immediate = proposal_metadata.pop("metrics")
            elif arm == "error_birth":
                proposal, touched, proposal_metadata = _birth_proposal(
                    base,
                    target,
                    base_render,
                    cfg,
                    constraint,
                    schedule,
                    budget,
                )
                immediate, immediate_render, immediate_quality = _evaluate(
                    proposal,
                    target,
                    mask,
                    cfg,
                    constraint,
                    args.coverage_tau,
                )
            else:
                proposal, touched, proposal_metadata = _moment_proposal(
                    base,
                    target,
                    base_render,
                    cfg,
                    constraint,
                    budget,
                )
                immediate, immediate_render, immediate_quality = _evaluate(
                    proposal,
                    target,
                    mask,
                    cfg,
                    constraint,
                    args.coverage_tau,
                )
            proposal_seconds = time.perf_counter() - started
            if proposal.n != base.n + budget:
                raise RuntimeError(
                    f"{arm}/{budget}: expected {base.n + budget} rows, got {proposal.n}"
                )
            _save_rgb(
                out / f"images/{arm}_{budget:03d}_immediate.png",
                immediate_render,
            )

            recovery_started = time.perf_counter()
            (
                recovered_field,
                _,
                selected_quality,
                recovery_record,
                recovery_output,
            ) = _safe_fit_block(
                proposal,
                None,
                base_quality,
                target,
                mask_cpu,
                mask,
                recovery_cfg,
                constraint,
                schedule,
                trainable_rows=None,
                verbose=False,
            )
            recovery_seconds = time.perf_counter() - recovery_started
            terminal_field = recovery_output["field"].detached()
            constraint.apply(terminal_field, cfg, refresh=True)
            terminal, terminal_render, _ = _evaluate(
                terminal_field,
                target,
                mask,
                cfg,
                constraint,
                args.coverage_tau,
            )
            accepted = bool(recovery_record["accepted"])
            if accepted:
                selected, selected_render, _ = _evaluate(
                    recovered_field,
                    target,
                    mask,
                    cfg,
                    constraint,
                    args.coverage_tau,
                )
            else:
                selected = dict(baseline)
                selected_render = base_render
            _save_rgb(
                out / f"images/{arm}_{budget:03d}_terminal.png",
                terminal_render,
            )
            _save_rgb(
                out / f"images/{arm}_{budget:03d}_selected.png",
                selected_render,
            )
            row = {
                "arm": arm,
                "budget": budget,
                "n_before": base.n,
                "n_after": proposal.n,
                "touched_rows": int(torch.unique(touched).numel()),
                "proposal_seconds": proposal_seconds,
                "recovery_seconds": recovery_seconds,
                "immediate": immediate,
                "terminal": terminal,
                "selected": selected,
                "recovery_accepted": accepted,
                "recovery_reasons": recovery_record["reasons"],
                "recovery_selected_steps": int(
                    recovery_record["accepted_steps"]
                ),
                "recovery_attempted_steps": int(
                    recovery_record["attempted_steps"]
                ),
                "recovery_metadata": recovery_record["metadata"],
                "proposal_metadata": proposal_metadata,
            }
            rows.append(row)
            _atomic_json(out / "partial_results.json", rows)
            print(
                f"{arm:12s} K={budget:3d} "
                f"immediate={immediate['foreground_psnr_db']:.6f} "
                f"terminal={terminal['foreground_psnr_db']:.6f} "
                f"accepted={accepted}"
            )

    decision = _decision(
        float(baseline["foreground_mse"]),
        rows,
        budgets,
    )
    repository_status = _git("status", "--short")
    result = {
        "schema": "structsplat.fit032.janelle-dipole-screen.v1",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": {
            "capture_root": str(args.capture_root.resolve()),
            "frame": args.frame,
            "view_id": args.view_id.upper(),
            "source_kind": prepared["source_kind"],
            "image_path": str(prepared["image_path"]),
            "mask_path": str(prepared["mask_path"]),
            "image_sha256": _sha256(prepared["image_path"]),
            "mask_sha256": _sha256(prepared["mask_path"]),
            "fit_window": prepared["fit_window"],
            "native_size": prepared["native_size"],
            "fit_size": prepared["fit_size"],
            "scale": prepared["scale"],
            "geometry_scale": geometry_scale,
            "foreground_pixels": int(mask.sum()),
        },
        "base_field": {
            "path": str(prepared["field_path"]),
            "sha256": _sha256(prepared["field_path"]),
            "n_gaussians": base.n,
        },
        "protocol": {
            "budgets": budgets,
            "arms": ARMS,
            "recovery_steps": int(args.recovery_steps),
            "checkpoint_every": int(args.checkpoint_every),
            "trust_radius": float(args.trust_radius),
            "max_color_contrast": float(args.max_color_contrast),
            "minimum_spacing": float(args.minimum_spacing),
            "spacing_scale": float(args.spacing_scale),
            "lift_scale_grid": [0.25, 0.5, 1.0],
            "fine_detail_metric": (
                "MSE of sigma-1.5 high-pass RGB residual on SDF > margin+6 pixels"
            ),
            "fit_config": asdict(cfg),
            "recovery_fit_config": asdict(recovery_cfg),
        },
        "baseline": baseline,
        "selection": {
            "seconds": selection_seconds,
            "candidate_count": full_selection.candidate_count,
            "rejected_background": full_selection.rejected_background,
            "rejected_mask": full_selection.rejected_mask,
            "rejected_degenerate": full_selection.rejected_degenerate,
            "selected": int(full_selection.parents.numel()),
        },
        "rows": rows,
        "decision": decision,
        "repository": {
            "commit": _git("rev-parse", "HEAD"),
            "branch": _git("branch", "--show-current"),
            "status": repository_status,
            "status_sha256": hashlib.sha256(
                repository_status.encode()
            ).hexdigest(),
            "source_snapshot": source_records,
        },
        "environment": {
            "platform": platform.platform(),
            "python": sys.version,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "device": str(device),
            "gpu": (
                torch.cuda.get_device_name(device)
                if device.type == "cuda"
                else None
            ),
            "peak_cuda_bytes": (
                int(torch.cuda.max_memory_allocated(device))
                if device.type == "cuda"
                else None
            ),
        },
    }
    _atomic_json(out / "result.json", result)
    print(json.dumps(decision, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-root", type=Path, default=DEFAULT_CAPTURE_ROOT)
    parser.add_argument("--realtime-root", type=Path, default=DEFAULT_REALTIME_ROOT)
    parser.add_argument("--field", type=Path, default=DEFAULT_FIELD)
    parser.add_argument(
        "--base-job",
        type=Path,
        default=None,
        help="bind directly to a completed current-pipeline seed job",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--frame", default="frame_00008")
    parser.add_argument("--view-id", default="C0001")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--renderer", default="cuda_tiled")
    parser.add_argument("--max-side", type=int, default=1200)
    parser.add_argument("--budgets", type=int, nargs="+", default=[32, 64, 128])
    parser.add_argument("--recovery-steps", type=int, default=80)
    parser.add_argument("--checkpoint-every", type=int, default=20)
    parser.add_argument("--coverage-tau", type=float, default=0.05)
    parser.add_argument("--boundary-band", type=float, default=4.0)
    parser.add_argument("--mask-margin", type=float, default=0.75)
    parser.add_argument("--trust-radius", type=float, default=0.35)
    parser.add_argument("--max-color-contrast", type=float, default=0.5)
    parser.add_argument("--minimum-spacing", type=float, default=3.0)
    parser.add_argument("--spacing-scale", type=float, default=0.75)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
