"""FIT-050: one guarded, full-frame normalized RGB-only transaction.

This experimental helper does not change maintained fitting defaults. Geometry, opacity and
support remain fixed, enabling image-space line trials without another Gaussian rasterization.
Every selected trial is checked again through the unchanged reference-coverage quality path.
"""
from __future__ import annotations

from dataclasses import replace
import math
import time

import torch

from .config import FitConfig
from .fit import (
    _MaskConstraint,
    _color_solve_renderer_supported,
    _normalized_color_basis_apply,
    _normalized_color_basis_transpose,
    _normalized_color_denominator,
    _raw_weight_map_field,
    _solve_colors_normalized,
)
from .gaussians import GaussianField
from .render import (
    _element_budget, _flat_tile_slices, _support_weight, _tile_bounds, _tile_coords,
)
from .safe_schedule import (
    QualityMetrics, SafeScheduleConfig, _quality_from_render, evaluate_quality,
    safe_commit_decision,
)


@torch.no_grad()
def _normalized_color_basis_diagonal(
    field: GaussianField, cfg: FitConfig, height: int, width: int, denominator: torch.Tensor,
) -> torch.Tensor:
    """Exact diagonal of A-transpose A, streamed with the reference support equations."""
    device, dtype = field.means.device, field.means.dtype
    means = field.means.detach()
    conics = field.conics(cfg.aa_dilation).detach()
    radii = field.radii(cfg.sigma_cutoff, cfg.aa_dilation)
    opacities = field.opacity_values()
    x0, y0, tx, count = _tile_bounds(means, radii, height, width)
    diagonal = torch.zeros(field.n, device=device, dtype=dtype)
    for start, end in _flat_tile_slices(count, _element_budget(cfg.render_chunk)):
        row, px, py = _tile_coords(x0, y0, tx, count, start, end, device)
        dx, dy = px.to(dtype) - means[row, 0], py.to(dtype) - means[row, 1]
        a, b, c = conics[row, 0], conics[row, 1], conics[row, 2]
        distance = a * dx.square() + 2.0 * b * dx * dy + c * dy.square()
        weight = _support_weight(distance, cfg.sigma_cutoff, cfg.support_fade, 1.0)
        if opacities is not None:
            weight = weight * opacities[row]
        basis = weight / (denominator.reshape(-1)[py * width + px] + cfg.normalization_eps)
        diagonal.index_add_(0, row, basis.square())
    return diagonal


def _ray_alpha(
    gradient: torch.Tensor, direction: torch.Tensor, image_direction: torch.Tensor, ridge: float,
) -> tuple[float, float, float | None]:
    """Scalar minimizer of ||r-alpha Av||² + ridge ||alpha v||² across all RGB."""
    numerator = float((gradient * direction).sum())
    denominator = float(image_direction.square().sum() + ridge * direction.square().sum())
    alpha = None
    if math.isfinite(numerator) and math.isfinite(denominator) and numerator > 0 and denominator > 0:
        candidate = numerator / denominator
        if math.isfinite(candidate) and candidate > 0:
            alpha = candidate
    return numerator, denominator, alpha


@torch.no_grad()
def refine_color_ray(
    field: GaussianField,
    target: torch.Tensor,
    cfg: FitConfig,
    mask: torch.Tensor,
    constraint: _MaskConstraint,
    schedule: SafeScheduleConfig,
    *,
    direction: str = "jacobi",
    max_trials: int = 6,
    cg_maxiter: int = 32,
) -> tuple[GaussianField, QualityMetrics, dict]:
    """Return an owned accepted field or exact owned rollback, quality, and JSON metadata.

    Only all-true masks are supported by this first experiment. RGB is signed and never clamped.
    Trials stop at the first safe nonzero coefficient change; replay failure rolls back without
    trying another fraction. Timings include synchronization and every rejected trial/check.
    """
    if target.is_cuda:
        torch.cuda.synchronize(target.device)
    started = time.perf_counter()
    if direction not in {"gradient", "jacobi", "cg"}:
        raise ValueError("direction must be gradient, jacobi, or cg")
    if isinstance(max_trials, bool) or not isinstance(max_trials, int) or not 1 <= max_trials <= 6:
        raise ValueError("max_trials must be an integer in [1, 6]")
    if isinstance(cg_maxiter, bool) or not isinstance(cg_maxiter, int) or cg_maxiter <= 0:
        raise ValueError("cg_maxiter must be a positive integer")
    if not _color_solve_renderer_supported(cfg.renderer) or cfg.color_basis != "constant":
        raise ValueError("color ray requires a normalized renderer and constant colors")
    if field.color_grads is not None or field.n < 1:
        raise ValueError("color ray requires at least one constant-color Gaussian")
    if cfg.pixel_loss != "l2" or cfg.ssim_weight != 0 or cfg.loss_target_downsample != 1:
        raise ValueError("color ray requires full-resolution L2 without SSIM")
    if target.ndim != 3 or target.shape[2] != 3 or target.numel() == 0:
        raise ValueError("target must have nonempty shape (H, W, 3)")
    if target.device != field.colors.device or target.dtype != field.colors.dtype:
        raise ValueError("target must share the field device and dtype")
    if mask.dtype != torch.bool or mask.shape != target.shape[:2] or mask.device != target.device:
        raise ValueError("mask must be a device-matched bool (H, W) tensor")
    if not bool(mask.all()):
        raise ValueError("this color-ray experiment supports full-frame all-true masks only")
    if not bool(torch.isfinite(target).all()) or bool((target < 0).any() or (target > 1).any()):
        raise ValueError("target must be finite in [0, 1]")
    for name in ("means", "log_scales", "rotations", "colors", "opacities", "filter_variance"):
        value = getattr(field, name)
        if value is not None and not bool(torch.isfinite(value).all()):
            raise ValueError(f"field.{name} must be finite")
    if (constraint.H, constraint.W) != target.shape[:2] or constraint.inside.device != target.device:
        raise ValueError("constraint must match the target shape and device")
    if not torch.equal(constraint.inside, mask):
        raise ValueError("constraint and mask must describe the same foreground")
    if not math.isfinite(cfg.color_solve_lambda) or cfg.color_solve_lambda < 0:
        raise ValueError("color_solve_lambda must be finite and nonnegative")

    def sync() -> None:
        if target.is_cuda:
            torch.cuda.synchronize(target.device)

    sync()
    reference_cfg = replace(cfg, quality_coverage_backend="reference", quality_tail_backend="reference")
    parent = field.detached()
    height, width = target.shape[:2]
    counts = {name: 0 for name in (
        "quality_evaluations", "gaussian_renders", "raw_coverage_passes",
        "basis_denominator_passes", "basis_apply_calls", "basis_transpose_calls",
        "basis_diagonal_passes", "interpolated_quality_evaluations", "legacy_cg_iterations",
    )}
    phases = {}
    metadata = {
        "direction": direction, "max_trials": max_trials, "ridge": float(cfg.color_solve_lambda),
        "cg_maxiter": cg_maxiter, "quality_coverage_backend": "reference",
        "quality_tail_backend": "reference", "trials": [],
        "counts": counts, "phase_seconds": phases, "selected_fraction": 0.0,
        "selected_alpha": 0.0, "selected_trial_index": None, "accepted": False,
        "coefficients_changed": False, "foreground_mse_improved": False,
        "rollback_reason": None, "basis_parent_max_abs_error": None,
        "replay_max_abs_error": None, "replay_reasons": [], "replay_metrics": None,
        "numerator": None, "denominator": None, "alpha_star": None,
    }

    def measured(name, function):
        sync()
        begin = time.perf_counter()
        result = function()
        sync()
        phases[name] = phases.get(name, 0.0) + time.perf_counter() - begin
        return result

    def actual_quality(candidate):
        counts["quality_evaluations"] += 1
        counts["gaussian_renders"] += 1
        counts["raw_coverage_passes"] += 1
        return evaluate_quality(candidate, target, mask, reference_cfg, constraint, schedule.coverage_tau)

    before, parent_image = measured("initial_reference_quality", lambda: actual_quality(parent))
    metadata["parent_metrics"] = before.to_dict()

    def finish(selected, quality, reason=None):
        metadata["rollback_reason"] = reason
        metadata["selected_metrics"] = quality.to_dict()
        sync()
        metadata["elapsed_seconds"] = time.perf_counter() - started
        def json_value(value):
            if isinstance(value, float) and not math.isfinite(value):
                return None
            if isinstance(value, dict):
                return {key: json_value(item) for key, item in value.items()}
            if isinstance(value, list):
                return [json_value(item) for item in value]
            return value
        return selected, quality, json_value(metadata)

    if not before.finite:
        return finish(parent, before, "nonfinite_parent_render")
    counts["basis_denominator_passes"] += 1
    den = measured("basis_denominator", lambda: _normalized_color_denominator(
        parent, reference_cfg, height, width, support_fade_alpha=1.0))
    counts["basis_apply_calls"] += 1
    basis_parent = measured("basis_parent", lambda: _normalized_color_basis_apply(
        parent, parent.colors, reference_cfg, height, width, den, support_fade_alpha=1.0))
    parity = float((basis_parent - parent_image).abs().max())
    metadata["basis_parent_max_abs_error"] = parity
    if not math.isfinite(parity) or parity > 2e-5:
        return finish(parent, before, "basis_parent_parity_failed")
    counts["raw_coverage_passes"] += 1
    coverage = measured("trial_reference_coverage", lambda: _raw_weight_map_field(
        parent, reference_cfg, height, width, support_fade_alpha=1.0).reshape(height, width))

    if direction == "cg":
        endpoint = parent.detached()
        stats = measured("legacy_cg", lambda: _solve_colors_normalized(
            endpoint, target, replace(reference_cfg, color_solve_maxiter=cg_maxiter),
            height, width, support_fade_alpha=1.0))
        metadata["legacy_cg"] = stats
        counts["legacy_cg_iterations"] += stats["iterations"]
        for key in ("basis_apply_calls", "basis_transpose_calls"):
            counts[key] += stats[key]
        counts["basis_denominator_passes"] += stats["denominator_calls"]
        vector = endpoint.colors - parent.colors
        alpha_star = 1.0
    else:
        residual = target - basis_parent
        counts["basis_transpose_calls"] += 1
        gradient = measured("residual_transpose", lambda: _normalized_color_basis_transpose(
            parent, residual, reference_cfg, height, width, den, support_fade_alpha=1.0))
        vector = gradient
        if direction == "jacobi":
            counts["basis_diagonal_passes"] += 1
            diagonal = measured("basis_diagonal", lambda: _normalized_color_basis_diagonal(
                parent, reference_cfg, height, width, den)) + float(cfg.color_solve_lambda)
            valid = torch.isfinite(diagonal) & (diagonal > 0)
            divisor = torch.where(valid, diagonal, torch.ones_like(diagonal))
            vector = torch.where(valid[:, None], gradient / divisor[:, None], torch.zeros_like(gradient))

    if not bool(torch.isfinite(vector).all()) or not bool((vector != 0).any()):
        return finish(parent, before, "invalid_or_zero_direction")
    counts["basis_apply_calls"] += 1
    image_direction = measured("direction_render", lambda: _normalized_color_basis_apply(
        parent, vector, reference_cfg, height, width, den, support_fade_alpha=1.0))
    if direction != "cg":
        numerator, denominator, alpha_star = _ray_alpha(
            gradient, vector, image_direction, float(cfg.color_solve_lambda))
        metadata["numerator"], metadata["denominator"] = numerator, denominator
    metadata["alpha_star"] = alpha_star
    if alpha_star is None or not bool(torch.isfinite(image_direction).all()):
        return finish(parent, before, "invalid_line_minimizer")

    for index in range(max_trials):
        trial_started = time.perf_counter()
        fraction = 2.0 ** (-index)
        alpha = alpha_star * fraction
        colors = parent.colors + alpha * vector
        surrogate = parent_image + alpha * image_direction
        counts["interpolated_quality_evaluations"] += 1
        trial_quality = measured("trial_metrics", lambda: _quality_from_render(
            surrogate, target, coverage, mask, constraint, schedule.coverage_tau, parent.n))
        accepted, reasons = safe_commit_decision(
            before, trial_quality, schedule.tolerances, schedule.hole_regression_budget)
        changed = bool((colors != parent.colors).any())
        finite = bool(torch.isfinite(colors).all() and torch.isfinite(surrogate).all())
        if not finite:
            accepted, reasons = False, reasons + ["nonfinite_trial"]
        if not changed:
            accepted, reasons = False, reasons + ["unchanged_coefficients"]
        sync()
        trial_record = {
            "index": index, "fraction": fraction, "alpha": alpha,
            "surrogate_metrics": trial_quality.to_dict(), "surrogate_accepted": accepted,
            "reasons": reasons, "coefficients_changed": changed, "finite": finite,
            "coefficient_max_abs_change": float((colors - parent.colors).abs().max()),
            "image_max_abs_change": float((surrogate - parent_image).abs().max()),
            "raw_sse": float((surrogate - target).square().sum()),
            "ridge_penalty": float(cfg.color_solve_lambda * (colors - parent.colors).square().sum()),
            "elapsed_seconds": time.perf_counter() - trial_started,
        }
        metadata["trials"].append(trial_record)
        if not accepted:
            continue
        candidate = parent.detached()
        candidate.colors.copy_(colors)
        replay_quality, replay = measured("selected_reference_quality", lambda: actual_quality(candidate))
        replay_error = float((replay - surrogate).abs().max())
        valid, replay_reasons = safe_commit_decision(
            before, replay_quality, schedule.tolerances, schedule.hole_regression_budget)
        if not math.isfinite(replay_error) or replay_error > 2e-5:
            valid = False
            replay_reasons.append("replay_parity_failed")
        metadata["replay_max_abs_error"] = replay_error
        metadata["replay_reasons"] = replay_reasons
        metadata["replay_metrics"] = replay_quality.to_dict()
        trial_record["replay_metrics"] = replay_quality.to_dict()
        trial_record["replay_max_abs_error"] = replay_error
        trial_record["replay_accepted"] = valid
        if not valid:
            return finish(parent, before, "selected_replay_failed")
        metadata.update({
            "selected_fraction": fraction, "selected_alpha": alpha, "selected_trial_index": index,
            "accepted": True, "coefficients_changed": True,
            "foreground_mse_improved": replay_quality.foreground_mse < before.foreground_mse,
        })
        return finish(candidate, replay_quality)
    return finish(parent, before, "all_trials_rejected")


__all__ = ["refine_color_ray"]
