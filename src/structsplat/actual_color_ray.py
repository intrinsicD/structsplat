"""FIT-051: actual-render RGB transactions, separate from FIT-050 interpolation.

Streaming directions are cross-backend proposals, not exact native-renderer Jacobians. Every
trial and selected replay uses the maintained renderer and unchanged reference quality gates.
"""
from __future__ import annotations

from dataclasses import replace
import math
import time

import torch

from .color_ray import _normalized_color_basis_diagonal
from .config import FitConfig
from .fit import (
    _MaskConstraint, _color_solve_renderer_supported, _normalized_color_basis_transpose,
    _normalized_color_denominator, _render, _solve_colors_normalized,
)
from .gaussians import GaussianField
from .safe_schedule import (
    QualityMetrics, SafeScheduleConfig, _quality_from_render, _quality_render_inputs,
    safe_commit_decision,
)


def _native_color_gradient(field, residual, cfg):
    """VJP of the actual renderer, requesting only owned cloned RGB coefficients.

    The existing CUDA backward may internally compute unused gradients; this is not a new
    specialized color-only kernel and its entire backward invocation is charged.
    """
    candidate = field.detached()
    with torch.enable_grad():
        candidate.colors.requires_grad_(True)
        native = _render(candidate, cfg, *residual.shape[:2], support_fade_alpha=1.0)
        gradient, = torch.autograd.grad(native, candidate.colors, grad_outputs=residual.detach(),
                                        create_graph=False, retain_graph=False)
    return gradient.detach(), native.detach()


def _json_value(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    return value


def _finite_quality(quality):
    # Finite RGB can still overflow squared-error reductions in low precision. Preserve the
    # maintained metric vector, but fail this transaction closed on any nonfinite reduction.
    return quality.finite and all(math.isfinite(value) for value in quality.to_dict().values())


@torch.no_grad()
def refine_actual_color_ray(
    field: GaussianField,
    target: torch.Tensor,
    cfg: FitConfig,
    mask: torch.Tensor,
    constraint: _MaskConstraint,
    schedule: SafeScheduleConfig,
    *,
    direction: str = "native_gradient",
    max_trials: int = 6,
    cg_maxiter: int = 32,
) -> tuple[GaussianField, QualityMetrics, dict, dict]:
    """One actual-render transaction returning owned field, quality, JSON and raw tensors.

    The tensor dictionary contains detached same-device parent/direction/trial/replay RGB and
    raw coverage, plus gradient, diagonal and CG endpoint when applicable. These artifacts are
    retained within measured work. CG alpha1 copies its solved endpoint exactly; other fractions
    use parent RGB plus alpha times the saved direction. No image interpolation is performed.
    """
    if target.is_cuda:
        torch.cuda.synchronize(target.device)
    started = time.perf_counter()
    directions = {"cg", "streaming_gradient", "streaming_jacobi", "native_gradient"}
    if direction not in directions:
        raise ValueError(f"direction must be one of {sorted(directions)}")
    if isinstance(max_trials, bool) or not isinstance(max_trials, int) or not 1 <= max_trials <= 6:
        raise ValueError("max_trials must be an integer in [1, 6]")
    if isinstance(cg_maxiter, bool) or not isinstance(cg_maxiter, int) or cg_maxiter < 1:
        raise ValueError("cg_maxiter must be a positive integer")
    if not _color_solve_renderer_supported(cfg.renderer) or cfg.color_basis != "constant" or field.color_grads is not None:
        raise ValueError("actual color rays require normalized constant-color rendering")
    if cfg.pixel_loss != "l2" or cfg.ssim_weight != 0 or cfg.loss_target_downsample != 1:
        raise ValueError("actual color rays require full-resolution L2 without SSIM")
    if target.ndim != 3 or target.shape[-1] != 3 or target.numel() == 0 or field.n < 1:
        raise ValueError("nonempty HWC RGB target and at least one Gaussian are required")
    if target.device != field.colors.device or target.dtype != field.colors.dtype:
        raise ValueError("target must share field dtype and device")
    if mask.dtype != torch.bool or mask.shape != target.shape[:2] or mask.device != target.device or not bool(mask.all()):
        raise ValueError("actual color rays currently require a full-frame all-true bool mask")
    if (constraint.H, constraint.W) != target.shape[:2] or constraint.inside.device != target.device or not torch.equal(constraint.inside, mask):
        raise ValueError("constraint must match the full-frame target and mask")
    if not bool(torch.isfinite(target).all()) or bool(((target < 0) | (target > 1)).any()):
        raise ValueError("target must be finite in [0, 1]")
    for name in ("means", "log_scales", "rotations", "colors", "opacities", "filter_variance"):
        value = getattr(field, name)
        if value is not None and not bool(torch.isfinite(value).all()):
            raise ValueError(f"field.{name} must be finite")
    ridge = float(cfg.color_solve_lambda)
    if not math.isfinite(ridge) or ridge < 0:
        raise ValueError("ridge must be finite and nonnegative")

    reference_cfg = replace(cfg, quality_coverage_backend="reference", quality_tail_backend="reference")
    parent = field.detached()
    height, width = target.shape[:2]
    counts = {name: 0 for name in (
        "quality_evaluations", "gaussian_renders", "raw_coverage_passes",
        "actual_direction_render_calls", "native_gradient_forward_calls", "native_color_vjp_calls",
        "basis_denominator_passes", "basis_transpose_calls", "basis_diagonal_passes",
        "basis_apply_calls", "legacy_cg_iterations",
    )}
    phases = {}
    metadata = {
        "direction": direction, "max_trials": max_trials, "cg_maxiter": cg_maxiter, "ridge": ridge,
        "proposal_semantics": ("renderer-native color VJP" if direction == "native_gradient"
                                else "streaming cross-backend proposal"),
        "quality_coverage_backend": "reference", "quality_tail_backend": "reference",
        "image_interpolation": False, "cg_endpoint_exact_alpha1": direction == "cg",
        "counts": counts, "phase_seconds": phases, "trials": [], "numerator": None,
        "denominator": None, "alpha_star": None, "selected_fraction": 0.0,
        "selected_alpha": 0.0, "selected_trial_index": None, "accepted": False,
        "coefficients_changed": False, "foreground_mse_improved": False,
        "replay_metrics": None, "replay_max_abs_error": None, "replay_reasons": [],
        "rollback_reason": None,
    }
    tensors = {name: None for name in (
        "parent_render", "parent_denominator", "gradient", "diagonal", "direction",
        "direction_render", "native_gradient_render", "cg_endpoint_colors",
        "replay_render", "replay_denominator",
    )}
    tensors["trial_renders"], tensors["trial_denominators"] = [], []

    def sync():
        if target.is_cuda:
            torch.cuda.synchronize(target.device)

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
        image, denominator = _quality_render_inputs(candidate, reference_cfg, height, width,
                                                     mask, schedule.coverage_tau)
        quality = _quality_from_render(image, target, denominator, mask, constraint,
                                       schedule.coverage_tau, candidate.n, tail_backend="reference")
        return quality, image.detach(), denominator.detach()

    before, parent_image, parent_den = measured("initial_actual_quality", lambda: actual_quality(parent))
    tensors["parent_render"], tensors["parent_denominator"] = parent_image, parent_den
    metadata["parent_metrics"] = before.to_dict()

    def finish(selected, quality, reason=None):
        metadata["rollback_reason"] = reason
        metadata["selected_metrics"] = quality.to_dict()
        sync()
        metadata["elapsed_seconds"] = time.perf_counter() - started
        return selected, quality, _json_value(metadata), tensors

    if not _finite_quality(before):
        return finish(parent, before, "nonfinite_parent_render")
    residual = (target - parent_image).detach()
    if direction == "cg":
        endpoint = parent.detached()
        stats = measured("legacy_cg", lambda: _solve_colors_normalized(
            endpoint, target, replace(reference_cfg, color_solve_maxiter=cg_maxiter), height, width,
            support_fade_alpha=1.0))
        metadata["legacy_cg"] = stats
        counts["legacy_cg_iterations"] = stats["iterations"]
        counts["basis_denominator_passes"] += stats["denominator_calls"]
        counts["basis_apply_calls"] += stats["basis_apply_calls"]
        counts["basis_transpose_calls"] += stats["basis_transpose_calls"]
        tensors["cg_endpoint_colors"] = endpoint.colors.detach()
        vector = endpoint.colors - parent.colors
    elif direction == "native_gradient":
        counts["native_gradient_forward_calls"] += 1
        counts["gaussian_renders"] += 1
        counts["native_color_vjp_calls"] += 1
        vector, native_image = measured("native_color_vjp", lambda: _native_color_gradient(parent, residual, reference_cfg))
        tensors["gradient"], tensors["native_gradient_render"] = vector, native_image
    else:
        counts["basis_denominator_passes"] += 1
        den = measured("streaming_denominator", lambda: _normalized_color_denominator(
            parent, reference_cfg, height, width, support_fade_alpha=1.0))
        counts["basis_transpose_calls"] += 1
        gradient = measured("streaming_transpose", lambda: _normalized_color_basis_transpose(
            parent, residual, reference_cfg, height, width, den, support_fade_alpha=1.0))
        tensors["gradient"] = gradient.detach()
        vector = gradient
        if direction == "streaming_jacobi":
            counts["basis_diagonal_passes"] += 1
            diagonal = measured("streaming_diagonal", lambda: _normalized_color_basis_diagonal(
                parent, reference_cfg, height, width, den))
            tensors["diagonal"] = diagonal.detach()
            divisor = diagonal + ridge
            valid = torch.isfinite(divisor) & (divisor > 0)
            safe = torch.where(valid, divisor, torch.ones_like(divisor))
            vector = torch.where(valid[:, None], gradient / safe[:, None], torch.zeros_like(gradient))
    tensors["direction"] = vector.detach()
    if not bool(torch.isfinite(vector).all()) or not bool((vector != 0).any()):
        return finish(parent, before, "invalid_or_zero_direction")
    direction_field = parent.detached()
    direction_field.colors.copy_(vector)
    counts["actual_direction_render_calls"] += 1
    counts["gaussian_renders"] += 1
    image_direction = measured("actual_direction_render", lambda: _render(
        direction_field, reference_cfg, height, width, support_fade_alpha=1.0))
    tensors["direction_render"] = image_direction.detach()
    if not bool(torch.isfinite(image_direction).all()):
        return finish(parent, before, "nonfinite_direction_render")
    alpha_star = 1.0
    if direction != "cg":
        numerator = float((residual * image_direction).sum())
        denominator = float(image_direction.square().sum() + ridge * vector.square().sum())
        metadata["numerator"], metadata["denominator"] = numerator, denominator
        if not math.isfinite(numerator) or not math.isfinite(denominator) or numerator <= 0 or denominator <= 0:
            return finish(parent, before, "invalid_line_minimizer")
        alpha_star = numerator / denominator
        if not math.isfinite(alpha_star) or alpha_star <= 0:
            return finish(parent, before, "invalid_line_minimizer")
    metadata["alpha_star"] = alpha_star

    for index in range(max_trials):
        trial_started = time.perf_counter()
        fraction, alpha = 2.0 ** (-index), alpha_star * 2.0 ** (-index)
        candidate = parent.detached()
        colors = tensors["cg_endpoint_colors"] if direction == "cg" and index == 0 else parent.colors + alpha * vector
        candidate.colors.copy_(colors)
        actual, raw, denominator = measured("actual_trial_quality", lambda: actual_quality(candidate))
        tensors["trial_renders"].append(raw)
        tensors["trial_denominators"].append(denominator)
        accepted, reasons = safe_commit_decision(before, actual, schedule.tolerances,
                                                schedule.hole_regression_budget)
        changed = not torch.equal(candidate.colors, parent.colors)
        finite = bool(torch.isfinite(candidate.colors).all()) and _finite_quality(actual)
        if not finite:
            accepted, reasons = False, reasons + ["nonfinite_trial"]
        if not changed:
            accepted, reasons = False, reasons + ["unchanged_coefficients"]
        record = {
            "index": index, "fraction": fraction, "alpha": alpha, "actual_metrics": actual.to_dict(),
            "accepted": accepted, "reasons": reasons, "finite": finite,
            "coefficients_changed": changed, "coefficient_max_abs_change": float((candidate.colors - parent.colors).abs().max()),
            "image_max_abs_change": float((raw - parent_image).abs().max()),
            "raw_sse": float((raw - target).square().sum()),
            "ridge_penalty": float(ridge * (candidate.colors - parent.colors).square().sum()),
            "elapsed_seconds": time.perf_counter() - trial_started,
            "transaction_elapsed_seconds": time.perf_counter() - started,
        }
        metadata["trials"].append(record)
        if not accepted:
            continue
        replay_quality, replay, replay_den = measured("selected_actual_replay", lambda: actual_quality(candidate))
        tensors["replay_render"], tensors["replay_denominator"] = replay, replay_den
        error = float((replay - raw).abs().max())
        valid, replay_reasons = safe_commit_decision(before, replay_quality, schedule.tolerances,
                                                    schedule.hole_regression_budget)
        if not _finite_quality(replay_quality):
            valid = False
            replay_reasons.append("nonfinite_replay_quality")
        if not math.isfinite(error) or error > 2e-5:
            valid = False
            replay_reasons.append("replay_parity_failed")
        metadata.update({"replay_metrics": replay_quality.to_dict(), "replay_max_abs_error": error,
                         "replay_reasons": replay_reasons})
        record.update({"replay_metrics": replay_quality.to_dict(), "replay_max_abs_error": error,
                       "replay_accepted": valid})
        if not valid:
            return finish(parent, before, "selected_replay_failed")
        metadata.update({"accepted": True, "coefficients_changed": True, "selected_fraction": fraction,
            "selected_alpha": alpha, "selected_trial_index": index,
            "foreground_mse_improved": replay_quality.foreground_mse < before.foreground_mse})
        return finish(candidate, replay_quality)
    return finish(parent, before, "all_trials_rejected")


__all__ = ["refine_actual_color_ray"]
