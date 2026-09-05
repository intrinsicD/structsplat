"""Bounded parent and transaction controls for the FIT-050 development experiment."""
from __future__ import annotations

import copy
from dataclasses import asdict, replace
import time

import numpy as np
import torch

from structsplat.color_ray import refine_color_ray
from structsplat.config import FitConfig, InitConfig, StructureTensorConfig
from structsplat.fit import _MaskConstraint, _solve_colors_normalized, fit
from structsplat.init import build_field
from structsplat.safe_schedule import SafeScheduleConfig, evaluate_quality, safe_commit_decision

ARMS = ["noop", "legacy_cg32", "cg_ray", "gradient_ray", "jacobi_ray", "adam32"]
IMAGE_IDS = [9, 25, 30, 34]
SEEDS = [0, 1]


def parent_configs(seed, *, smoke=False, device="cuda"):
    init = InitConfig(strategy="quadtree_wse", num_gaussians=16 if smoke else 2000,
                      seed=int(seed), opacity_mode="none")
    cfg = FitConfig(
        iters=3 if smoke else 750, renderer="normalized" if device == "cpu" else "cuda",
        pixel_loss="l2", ssim_weight=0.0, support_fade=True,
        quality_coverage_backend="reference", quality_tail_backend="reference",
        lr_means=0.05, lr_scales=0.03, lr_rot=0.01, lr_color=0.03, lr_opacity=0.01,
        optimizer="adam", lr_schedule="none", color_solve_schedule="none",
        color_solve_every=None, color_solve_lambda=1e-4, color_solve_maxiter=32,
        checkpoint_policy="terminal", log_every=1 if smoke else 25,
        compute_lpips=False, render_chunk=512, split_every=None, prune_every=None,
        relocate_every=None, adaptive_count=False,
    )
    return init, cfg


def sync(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)


def fit_parent(target_np, seed, *, smoke=False, device="cuda"):
    """Fit one exact-count parent; callers persist initial/terminal fields and optimizer state."""
    torch.manual_seed(seed)
    init, cfg = parent_configs(seed, smoke=smoke, device=device)
    source = np.asarray(target_np, dtype=np.float32)
    sync(device)
    started = time.perf_counter()
    field = build_field(source, init, StructureTensorConfig(), device=device)
    if field.n != init.num_gaussians:
        raise RuntimeError("initializer violated exact parent count")
    initial = field.detached()
    sync(device)
    init_seconds = time.perf_counter() - started
    target = torch.as_tensor(source, device=device)
    output = fit(field, target, cfg, verbose=False, return_optimizer_state=True)
    sync(device)
    if output["iterations_run"] != cfg.iters or output["field"].n != init.num_gaussians:
        raise RuntimeError("parent violated frozen count/horizon")
    output.update({"initial_field": initial, "init_config": asdict(init), "fit_config": asdict(cfg),
                   "tensor_config": asdict(StructureTensorConfig()), "init_seconds": init_seconds,
                   "parent_total_seconds": time.perf_counter() - started})
    return output


def full_frame_context(target, cfg, count):
    mask = torch.ones(target.shape[:2], dtype=torch.bool, device=target.device)
    constraint = _MaskConstraint.from_mask(
        mask.cpu().numpy(), target.device, target.dtype, cfg.sigma_cutoff, cfg.mask_margin,
        cap_mode=cfg.mask_cap_mode, undercoverage_band=cfg.mask_undercoverage_band)
    schedule = SafeScheduleConfig(capacity=count, coverage_target_gaussians=count,
                                  detail_target_gaussians=count)
    return mask, constraint, schedule


def _changed_noncolors(before, after):
    changed = []
    for name in ("means", "log_scales", "rotations", "opacities", "scale_max",
                 "color_grads", "background_mask", "filter_variance"):
        left, right = getattr(before, name), getattr(after, name)
        if (left is None) != (right is None) or (left is not None and not torch.equal(left, right)):
            changed.append(name)
    return changed


def run_arm(parent, target, cfg, mask, constraint, schedule, arm, optimizer_state, *, smoke=False):
    """One independently charged transaction; Adam is the full-parameter practical control."""
    if arm not in ARMS:
        raise ValueError("unknown FIT-050 arm")
    cfg = replace(cfg, quality_coverage_backend="reference", quality_tail_backend="reference")
    sync(target.device)
    started = time.perf_counter()
    if arm.endswith("_ray"):
        field, quality, metadata = refine_color_ray(parent, target, cfg, mask, constraint, schedule,
            direction=arm.removesuffix("_ray"), max_trials=6, cg_maxiter=32)
        metadata["arm"] = arm
        metadata["noncolor_changed_fields"] = _changed_noncolors(parent, field)
        metadata["transaction_seconds"] = time.perf_counter() - started
        return field, quality, metadata, {}
    before, _ = evaluate_quality(parent, target, mask, cfg, constraint, schedule.coverage_tau)
    field = parent.detached()
    counts = {"quality_evaluations": 1, "gaussian_renders": 1, "raw_coverage_passes": 1,
              "basis_denominator_passes": 0, "basis_apply_calls": 0,
              "basis_transpose_calls": 0, "gradient_evaluations": 0}
    metadata = {"arm": arm, "parent_metrics": before.to_dict(), "counts": counts,
                "quality_coverage_backend": "reference", "quality_tail_backend": "reference",
                "trials": [], "selected_fraction": 0.0, "selected_alpha": 0.0,
                "accepted": False, "coefficients_changed": False,
                "foreground_mse_improved": False, "rollback_reason": "noop"}
    history = {}
    if arm == "legacy_cg32":
        stats = _solve_colors_normalized(field, target, replace(cfg, color_solve_maxiter=32),
                                         *target.shape[:2], support_fade_alpha=1.0)
        metadata["legacy_cg"] = stats
        for key in ("basis_apply_calls", "basis_transpose_calls"):
            counts[key] += stats[key]
        counts["basis_denominator_passes"] += stats["denominator_calls"]
    elif arm == "adam32":
        steps = 2 if smoke else 32
        continuation_cfg = replace(cfg, iters=steps, log_every=1)
        output = fit(field, target, continuation_cfg, verbose=False,
                     optimizer_state=copy.deepcopy(optimizer_state), return_optimizer_state=True)
        field, history = output["field"], output["history"]
        metadata["continuation_config"] = asdict(continuation_cfg)
        metadata["iterations_run"] = output["iterations_run"]
        metadata["candidate_noncolor_changed_fields"] = _changed_noncolors(parent, field)
        # Terminal, topology-free fit: one render/backward per step, one terminal render;
        # no observer, checkpoint selection, color solve, or auxiliary raw-weight pass.
        counts["gaussian_renders"] += steps + 1
        counts["gradient_evaluations"] += steps
    selected, quality = field, before
    if arm != "noop":
        candidate, _ = evaluate_quality(field, target, mask, cfg, constraint, schedule.coverage_tau)
        counts["quality_evaluations"] += 1
        counts["gaussian_renders"] += 1
        counts["raw_coverage_passes"] += 1
        accepted, reasons = safe_commit_decision(before, candidate, schedule.tolerances,
                                                schedule.hole_regression_budget)
        metadata["candidate_metrics"] = candidate.to_dict()
        metadata["candidate_reasons"] = reasons
        if accepted:
            quality = candidate
            metadata.update({"accepted": True, "selected_fraction": 1.0, "selected_alpha": 1.0,
                             "rollback_reason": None})
        else:
            selected = parent.detached()
            metadata["rollback_reason"] = "candidate_rejected"
    metadata["noncolor_changed_fields"] = _changed_noncolors(parent, selected)
    if arm != "adam32" and metadata["noncolor_changed_fields"]:
        raise RuntimeError("RGB-only transaction changed non-color state")
    metadata["coefficients_changed"] = not torch.equal(parent.colors, selected.colors)
    metadata["foreground_mse_improved"] = quality.foreground_mse < before.foreground_mse
    metadata["selected_metrics"] = quality.to_dict()
    sync(target.device)
    metadata["transaction_seconds"] = time.perf_counter() - started
    return selected, quality, metadata, history
