"""Implicit pixel-field contraction for direct additive Observation Field V2 (HIER-005).

This is a deterministic, default-off research reference.  Pixel leaves are procedural CPU data
during topology construction rather than persistent trainable rows.  A quadtree frontier proposes
local contractions with a closed-form continuous Gaussian inner product; every accepted action is
then re-fitted and scored on the actual discrete finite-support renderer.  Optional recovery can
materialize touched rows, touched rows plus a direct-neighbor halo, or all active rows temporarily
for differentiable optimization.
The result is a direct-additive ``ObservationField2D``.

Estimated row bytes are useful only for proposal ordering.  They are not a compressed stream and
must not be reported as actual rate; COMP-013 and FIT-030 own complete coded-byte decisions.
"""
from __future__ import annotations

from dataclasses import dataclass
import heapq
import itertools
import math
import time
from typing import Literal

import numpy as np

from .observation_field import (
    AlphaSemantics,
    CanvasCropTransform,
    FieldSemantics,
    ObservationField2D,
    SupportSemantics,
    adapt_direct_additive,
    pack_alpha,
)


CoefficientDomain = Literal["signed", "nonnegative"]
PairPolicy = Literal["exact_count", "always"]
RecoveryRenderer = Literal["additive", "cuda_additive", "cuda_tiled_additive"]
RecoverySchedule = Literal["actions", "progress"]
RecoveryScope = Literal["touched", "touched_neighborhood", "all_error_weighted"]


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be > 0, got {result}")
    return result


def _positive_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise TypeError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and > 0, got {result}")
    return result


def _nonnegative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be >= 0, got {result}")
    return result


def _nonnegative_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise TypeError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and >= 0, got {result}")
    return result


def _mask_aware_smoothed_error(
    residual_energy: np.ndarray,
    mask: np.ndarray,
    sigma_px: float,
) -> np.ndarray:
    """Smooth scalar error without diluting foreground values across a mask boundary."""

    if residual_energy.ndim != 2 or mask.shape != residual_energy.shape:
        raise ValueError("residual_energy and mask must have the same 2D shape")
    if mask.dtype != np.bool_:
        raise ValueError("mask must be boolean")
    sigma = _nonnegative_float(sigma_px, "sigma_px")
    values = np.asarray(residual_energy, dtype=np.float32)
    if not np.isfinite(values).all() or (values < 0.0).any():
        raise ValueError("residual_energy must be finite and nonnegative")
    if sigma == 0.0:
        return np.where(mask, values, 0.0).astype(np.float32, copy=False)

    from .structure_tensor import gaussian_blur

    mask_float = mask.astype(np.float32)
    numerator = gaussian_blur(values * mask_float, sigma)
    denominator = gaussian_blur(mask_float, sigma)
    result = np.zeros_like(values, dtype=np.float32)
    np.divide(numerator, denominator, out=result, where=denominator > 1e-8)
    result[~mask] = 0.0
    return result


def _normalized_error_update_weights(
    scores: np.ndarray,
    *,
    power: float,
    floor: float,
    ceiling: float,
) -> np.ndarray:
    """Map nonnegative per-row error exposure to mean-one Adam update multipliers."""

    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all() or (values < 0.0).any():
        raise ValueError("scores must be a finite nonnegative vector")
    exponent = _positive_float(power, "power")
    lower = _positive_float(floor, "floor")
    upper = _positive_float(ceiling, "ceiling")
    if lower > 1.0 or upper < 1.0 or lower > upper:
        raise ValueError("error-weight bounds must satisfy 0 < floor <= 1 <= ceiling")
    if values.size == 0:
        return values.astype(np.float32)
    mean = float(values.mean())
    if mean <= np.finfo(np.float64).tiny:
        return np.ones(values.shape, dtype=np.float32)
    weights = np.power(values / mean, exponent)
    weights = np.clip(weights, lower, upper)
    weights /= max(float(weights.mean()), np.finfo(np.float64).tiny)
    weights = np.clip(weights, lower, upper)
    return weights.astype(np.float32)


@dataclass(frozen=True)
class PixelContractionConfig:
    """Configuration for :func:`contract_image`.

    ``estimated_row_bytes`` prices proposals only.  The default 32 bytes is the uncompressed
    float32 payload of two means, two log-scales, one rotation, and three RGB coefficients.
    """

    target_gaussians: int
    leaf_scale_px: float = 0.18
    sigma_cutoff: float = 3.0
    support_fade_alpha: float = 0.0
    coefficient_domain: CoefficientDomain = "signed"
    estimated_row_bytes: int = 32
    proposal_batch_size: int = 64
    merge_batch_size: int = 8
    pair_shortlist: int = 3
    exact_option_shortlist: int = 2
    pair_policy: PairPolicy = "exact_count"
    max_exact_distortion_per_estimated_byte: float | None = None
    max_actions: int | None = None
    recovery_steps: int = 0
    recovery_scope: RecoveryScope = "touched"
    recovery_schedule: RecoverySchedule = "progress"
    recovery_progress_checkpoints: int = 16
    recovery_every_actions: int = 128
    recovery_neighborhood_radius_px: int = 1
    recovery_device: str = "cpu"
    recovery_renderer: RecoveryRenderer = "additive"
    recovery_render_chunk: int = 256
    recovery_lr_means: float = 0.005
    recovery_lr_scales: float = 0.003
    recovery_lr_rotations: float = 0.001
    recovery_lr_coefficients: float = 0.003
    recovery_max_mean_shift_px: float = 1.5
    recovery_max_log_scale_shift: float = 0.35
    recovery_max_rotation_shift_rad: float = 0.35
    recovery_error_smoothing_sigma_px: float = 1.5
    recovery_error_weight_power: float = 0.5
    recovery_error_weight_floor: float = 0.05
    recovery_error_weight_ceiling: float = 4.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "target_gaussians", _positive_int(self.target_gaussians, "target_gaussians")
        )
        object.__setattr__(
            self, "leaf_scale_px", _positive_float(self.leaf_scale_px, "leaf_scale_px")
        )
        object.__setattr__(
            self, "sigma_cutoff", _positive_float(self.sigma_cutoff, "sigma_cutoff")
        )
        fade = float(self.support_fade_alpha)
        if not math.isfinite(fade) or not 0.0 <= fade <= 1.0:
            raise ValueError("support_fade_alpha must be finite and in [0, 1]")
        object.__setattr__(self, "support_fade_alpha", fade)
        if self.coefficient_domain not in ("signed", "nonnegative"):
            raise ValueError("coefficient_domain must be 'signed' or 'nonnegative'")
        if self.pair_policy not in ("exact_count", "always"):
            raise ValueError("pair_policy must be 'exact_count' or 'always'")
        for name in (
            "estimated_row_bytes",
            "proposal_batch_size",
            "merge_batch_size",
            "pair_shortlist",
            "exact_option_shortlist",
        ):
            object.__setattr__(self, name, _positive_int(getattr(self, name), name))
        limit = self.max_exact_distortion_per_estimated_byte
        if limit is not None:
            limit_value = float(limit)
            if not math.isfinite(limit_value):
                raise ValueError(
                    "max_exact_distortion_per_estimated_byte must be finite when supplied"
                )
            object.__setattr__(
                self, "max_exact_distortion_per_estimated_byte", limit_value
            )
        if self.max_actions is not None:
            object.__setattr__(self, "max_actions", _positive_int(self.max_actions, "max_actions"))
        object.__setattr__(
            self, "recovery_steps", _nonnegative_int(self.recovery_steps, "recovery_steps")
        )
        if self.recovery_scope not in (
            "touched",
            "touched_neighborhood",
            "all_error_weighted",
        ):
            raise ValueError(
                "recovery_scope must be 'touched', 'touched_neighborhood', or "
                "'all_error_weighted'"
            )
        if self.recovery_schedule not in ("actions", "progress"):
            raise ValueError("recovery_schedule must be 'actions' or 'progress'")
        for name in (
            "recovery_progress_checkpoints",
            "recovery_every_actions",
            "recovery_render_chunk",
        ):
            object.__setattr__(self, name, _positive_int(getattr(self, name), name))
        object.__setattr__(
            self,
            "recovery_neighborhood_radius_px",
            _nonnegative_int(
                self.recovery_neighborhood_radius_px,
                "recovery_neighborhood_radius_px",
            ),
        )
        if not isinstance(self.recovery_device, str) or not self.recovery_device.strip():
            raise ValueError("recovery_device must be a nonempty torch device string")
        if self.recovery_renderer not in (
            "additive",
            "cuda_additive",
            "cuda_tiled_additive",
        ):
            raise ValueError("unsupported recovery_renderer")
        if self.recovery_renderer.startswith("cuda") and not self.recovery_device.startswith(
            "cuda"
        ):
            raise ValueError("a CUDA recovery renderer requires a CUDA recovery device")
        for name in (
            "recovery_lr_means",
            "recovery_lr_scales",
            "recovery_lr_rotations",
            "recovery_lr_coefficients",
            "recovery_max_mean_shift_px",
            "recovery_max_log_scale_shift",
            "recovery_max_rotation_shift_rad",
            "recovery_error_weight_power",
            "recovery_error_weight_floor",
            "recovery_error_weight_ceiling",
        ):
            object.__setattr__(self, name, _positive_float(getattr(self, name), name))
        object.__setattr__(
            self,
            "recovery_error_smoothing_sigma_px",
            _nonnegative_float(
                self.recovery_error_smoothing_sigma_px,
                "recovery_error_smoothing_sigma_px",
            ),
        )
        if not (
            self.recovery_error_weight_floor
            <= 1.0
            <= self.recovery_error_weight_ceiling
        ):
            raise ValueError(
                "recovery error-weight bounds must satisfy 0 < floor <= 1 <= ceiling"
            )


@dataclass(frozen=True)
class LocalRescueConfig:
    """Bounded terminal residual repair over a frozen direct-additive base field."""

    max_rows: int
    scale_px: float = 0.75
    nms_radius_px: int = 1
    steps: int = 400
    learning_rate: float = 0.05
    tail_fraction: float = 0.01
    tail_weight: float = 4.0
    pixel_rmse_threshold: float = 0.02
    patch7_rmse_threshold: float = 0.01
    device: str = "cpu"
    renderer: RecoveryRenderer = "additive"
    render_chunk: int = 256

    def __post_init__(self) -> None:
        object.__setattr__(self, "max_rows", _positive_int(self.max_rows, "max_rows"))
        object.__setattr__(self, "scale_px", _positive_float(self.scale_px, "scale_px"))
        object.__setattr__(
            self,
            "nms_radius_px",
            _nonnegative_int(self.nms_radius_px, "nms_radius_px"),
        )
        object.__setattr__(self, "steps", _positive_int(self.steps, "steps"))
        for name in (
            "learning_rate",
            "tail_fraction",
            "tail_weight",
            "pixel_rmse_threshold",
            "patch7_rmse_threshold",
        ):
            object.__setattr__(self, name, _positive_float(getattr(self, name), name))
        if self.tail_fraction > 1.0:
            raise ValueError("tail_fraction must lie in (0, 1]")
        if not isinstance(self.device, str) or not self.device.strip():
            raise ValueError("device must be a nonempty torch device string")
        if self.renderer not in ("additive", "cuda_additive", "cuda_tiled_additive"):
            raise ValueError("unsupported rescue renderer")
        if self.renderer.startswith("cuda") and not self.device.startswith("cuda"):
            raise ValueError("a CUDA rescue renderer requires a CUDA device")
        object.__setattr__(
            self, "render_chunk", _positive_int(self.render_chunk, "render_chunk")
        )


@dataclass(frozen=True)
class ContractionEvent:
    """One accepted, exactly validated topology action."""

    action_index: int
    batch_index: int
    level: int
    region_x: int
    region_y: int
    kind: str
    count_before: int
    count_after: int
    rows_removed: int
    rows_added: int
    analytic_continuous_sse: float
    exact_discrete_sse_delta: float
    exact_distortion_per_estimated_byte: float
    patch_xyxy: tuple[int, int, int, int]

    def to_record(self) -> dict[str, object]:
        return {
            "action_index": self.action_index,
            "batch_index": self.batch_index,
            "level": self.level,
            "region_x": self.region_x,
            "region_y": self.region_y,
            "kind": self.kind,
            "count_before": self.count_before,
            "count_after": self.count_after,
            "rows_removed": self.rows_removed,
            "rows_added": self.rows_added,
            "rows_saved": self.rows_removed - self.rows_added,
            "analytic_continuous_sse": self.analytic_continuous_sse,
            "exact_discrete_sse_delta": self.exact_discrete_sse_delta,
            "exact_distortion_per_estimated_byte": (
                self.exact_distortion_per_estimated_byte
            ),
            "patch_xyxy": list(self.patch_xyxy),
        }


@dataclass(frozen=True)
class RecoveryEvent:
    """One optimizer checkpoint over a declared active-row scope."""

    checkpoint_index: int
    action_count: int
    active_count: int
    touched_count: int
    newly_touched_count: int
    optimized_count: int
    neighborhood_count: int
    new_neighborhood_count: int
    accepted_new_neighborhood_count: int
    protected_optimized_count: int
    recovery_scope: RecoveryScope
    attempted_steps: int
    selected_step: int
    sse_before: float
    sse_after: float
    attribution_seconds: float
    error_smoothing_sigma_px: float
    error_weight_min: float
    error_weight_mean: float
    error_weight_p50: float
    error_weight_p90: float
    error_weight_max: float
    error_weight_effective_rows: float
    raw_error_score_mean: float
    raw_error_score_max: float
    elapsed_seconds: float
    accepted: bool

    def to_record(self) -> dict[str, object]:
        return {
            "checkpoint_index": self.checkpoint_index,
            "action_count": self.action_count,
            "active_count": self.active_count,
            "touched_count": self.touched_count,
            "newly_touched_count": self.newly_touched_count,
            "optimized_count": self.optimized_count,
            "neighborhood_count": self.neighborhood_count,
            "new_neighborhood_count": self.new_neighborhood_count,
            "accepted_new_neighborhood_count": self.accepted_new_neighborhood_count,
            "protected_optimized_count": self.protected_optimized_count,
            "recovery_scope": self.recovery_scope,
            "attempted_steps": self.attempted_steps,
            "selected_step": self.selected_step,
            "sse_before": self.sse_before,
            "sse_after": self.sse_after,
            "sse_gain": self.sse_before - self.sse_after,
            "attribution_seconds": self.attribution_seconds,
            "error_smoothing_sigma_px": self.error_smoothing_sigma_px,
            "error_weight_min": self.error_weight_min,
            "error_weight_mean": self.error_weight_mean,
            "error_weight_p50": self.error_weight_p50,
            "error_weight_p90": self.error_weight_p90,
            "error_weight_max": self.error_weight_max,
            "error_weight_effective_rows": self.error_weight_effective_rows,
            "raw_error_score_mean": self.raw_error_score_mean,
            "raw_error_score_max": self.raw_error_score_max,
            "elapsed_seconds": self.elapsed_seconds,
            "accepted": self.accepted,
        }


@dataclass(frozen=True)
class PixelContractionResult:
    """Contracted field and diagnostic telemetry.

    ``estimated_field_bytes`` follows the configured proposal price.  ``canonical_raw_bytes`` is
    the exact byte count of the field's raw NumPy arrays, still without a container or entropy
    coder.  Neither value is an actual compressed-file rate.
    """

    field: ObservationField2D
    reconstruction_raw: np.ndarray
    reconstruction: np.ndarray
    history: tuple[ContractionEvent, ...]
    recovery_history: tuple[RecoveryEvent, ...]
    initial_count: int
    target_count: int
    final_count: int
    initial_sse: float
    final_sse: float
    elapsed_seconds: float
    stop_reason: str
    estimated_field_bytes: int
    canonical_raw_bytes: int
    touched_active_rows: int
    untouched_active_rows: int
    recovery_neighbor_active_rows: int
    protected_initial_rows: int
    protected_active_rows: int
    blocked_regions: int

    def history_records(self) -> list[dict[str, object]]:
        return [event.to_record() for event in self.history]

    def recovery_records(self) -> list[dict[str, object]]:
        return [event.to_record() for event in self.recovery_history]


@dataclass(frozen=True)
class LocalRescueResult:
    """A frozen-base rescue field and exact optimization telemetry."""

    field: ObservationField2D
    reconstruction_raw: np.ndarray
    reconstruction: np.ndarray
    selected_xy: np.ndarray
    rows_added: int
    selected_step: int
    initial_sse: float
    final_sse: float
    violation_before: float
    violation_after: float
    pixel_rmse_max_before: float
    pixel_rmse_max_after: float
    patch7_rmse_max_before: float
    patch7_rmse_max_after: float
    elapsed_seconds: float

    def to_record(self) -> dict[str, object]:
        return {
            "rows_added": self.rows_added,
            "selected_step": self.selected_step,
            "initial_sse": self.initial_sse,
            "final_sse": self.final_sse,
            "sse_gain": self.initial_sse - self.final_sse,
            "violation_before": self.violation_before,
            "violation_after": self.violation_after,
            "pixel_rmse_max_before": self.pixel_rmse_max_before,
            "pixel_rmse_max_after": self.pixel_rmse_max_after,
            "patch7_rmse_max_before": self.patch7_rmse_max_before,
            "patch7_rmse_max_after": self.patch7_rmse_max_after,
            "elapsed_seconds": self.elapsed_seconds,
            "selected_xy": self.selected_xy.tolist(),
        }


@dataclass(frozen=True)
class _Proposal:
    kind: str
    remove_ids: tuple[int, ...]
    means: np.ndarray
    covariances: np.ndarray
    analytic_sse: float
    rows_saved: int
    resolves_region: bool
    protected_outputs: tuple[bool, ...] = ()


@dataclass(frozen=True)
class _EvaluatedProposal:
    proposal: _Proposal
    level: int
    region_y: int
    region_x: int
    version: int
    means: np.ndarray
    covariances: np.ndarray
    coefficients: np.ndarray
    patch_xyxy: tuple[int, int, int, int]
    patch_render: np.ndarray
    exact_sse_delta: float
    exact_slope: float


def _as_mean(value: object, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (2,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite array with shape (2,)")
    return array


def _as_covariance(value: object, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (2, 2) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite array with shape (2, 2)")
    if not np.allclose(array, array.T, rtol=0.0, atol=1e-12):
        raise ValueError(f"{name} must be symmetric")
    eigenvalues = np.linalg.eigvalsh(array)
    if not np.isfinite(eigenvalues).all() or float(eigenvalues[0]) <= 0.0:
        raise ValueError(f"{name} must be positive definite")
    return array


def gaussian_inner_product(
    mean_a: object,
    covariance_a: object,
    mean_b: object,
    covariance_b: object,
) -> float:
    """Continuous inner product of two peak-one 2D Gaussian kernels.

    For ``g_i(x)=exp(-0.5 (x-mu_i)^T Sigma_i^-1 (x-mu_i))`` this returns
    ``integral g_a(x) g_b(x) dx`` over the plane.
    """

    mu_a = _as_mean(mean_a, "mean_a")
    mu_b = _as_mean(mean_b, "mean_b")
    cov_a = _as_covariance(covariance_a, "covariance_a")
    cov_b = _as_covariance(covariance_b, "covariance_b")
    return _gaussian_inner_product_unchecked(mu_a, cov_a, mu_b, cov_b)


def _gaussian_inner_product_unchecked(
    mu_a: np.ndarray, cov_a: np.ndarray, mu_b: np.ndarray, cov_b: np.ndarray
) -> float:
    # Expanded symmetric-2x2 algebra avoids three tiny ``np.linalg`` dispatches per pair.  This
    # inner loop dominates candidate construction on image-sized frontiers.
    determinant_a = float(cov_a[0, 0] * cov_a[1, 1] - cov_a[0, 1] * cov_a[0, 1])
    determinant_b = float(cov_b[0, 0] * cov_b[1, 1] - cov_b[0, 1] * cov_b[0, 1])
    summed_00 = float(cov_a[0, 0] + cov_b[0, 0])
    summed_01 = float(cov_a[0, 1] + cov_b[0, 1])
    summed_11 = float(cov_a[1, 1] + cov_b[1, 1])
    determinant_sum = summed_00 * summed_11 - summed_01 * summed_01
    delta_x = float(mu_a[0] - mu_b[0])
    delta_y = float(mu_a[1] - mu_b[1])
    quadratic = (
        summed_11 * delta_x * delta_x
        - 2.0 * summed_01 * delta_x * delta_y
        + summed_00 * delta_y * delta_y
    ) / determinant_sum
    exponent = -0.5 * quadratic
    determinant_ratio = math.sqrt(determinant_a * determinant_b / determinant_sum)
    return float(2.0 * math.pi * determinant_ratio * math.exp(exponent))


def _covariances_from_compact(compact: np.ndarray) -> np.ndarray:
    result = np.empty((compact.shape[0], 2, 2), dtype=np.float64)
    result[:, 0, 0] = compact[:, 0]
    result[:, 0, 1] = compact[:, 1]
    result[:, 1, 0] = compact[:, 1]
    result[:, 1, 1] = compact[:, 2]
    return result


def _covariances_to_compact(covariances: np.ndarray, dtype: np.dtype) -> np.ndarray:
    return np.stack(
        [covariances[:, 0, 0], covariances[:, 0, 1], covariances[:, 1, 1]], axis=1
    ).astype(dtype, copy=False)


def _kernel_gram(means_a: np.ndarray, covs_a: np.ndarray, means_b: np.ndarray,
                 covs_b: np.ndarray) -> np.ndarray:
    result = np.empty((means_a.shape[0], means_b.shape[0]), dtype=np.float64)
    for row in range(means_a.shape[0]):
        for column in range(means_b.shape[0]):
            result[row, column] = _gaussian_inner_product_unchecked(
                means_a[row], covs_a[row], means_b[column], covs_b[column]
            )
    return result


def _solve_gram(
    gram: np.ndarray, rhs: np.ndarray, coefficient_domain: CoefficientDomain
) -> np.ndarray:
    """Solve one- or two-basis least squares, including an exact tiny NNLS enumeration."""

    basis_count = gram.shape[0]
    if gram.shape != (basis_count, basis_count) or rhs.shape != (basis_count, 3):
        raise ValueError("incompatible Gram-system shapes")
    if basis_count not in (1, 2):
        raise ValueError("local coefficient solve supports one or two basis functions")
    if coefficient_domain == "signed":
        return np.linalg.lstsq(gram, rhs, rcond=1e-10)[0]

    result = np.zeros((basis_count, 3), dtype=np.float64)
    for channel in range(3):
        channel_rhs = rhs[:, channel]
        candidates = [np.zeros(basis_count, dtype=np.float64)]
        for basis in range(basis_count):
            if gram[basis, basis] > 1e-14:
                candidate = np.zeros(basis_count, dtype=np.float64)
                candidate[basis] = max(channel_rhs[basis] / gram[basis, basis], 0.0)
                candidates.append(candidate)
        if basis_count == 2:
            unconstrained = np.linalg.lstsq(gram, channel_rhs, rcond=1e-10)[0]
            if np.all(unconstrained >= -1e-12):
                candidates.append(np.maximum(unconstrained, 0.0))
        best = candidates[0]
        best_value = float(best @ gram @ best - 2.0 * best @ channel_rhs)
        for candidate in candidates[1:]:
            value = float(candidate @ gram @ candidate - 2.0 * candidate @ channel_rhs)
            if value < best_value - 1e-15:
                best = candidate
                best_value = value
        result[:, channel] = best
    return result


def _moment_parent(
    means: np.ndarray, covariances: np.ndarray, coefficients: np.ndarray, minimum_variance: float
) -> tuple[np.ndarray, np.ndarray]:
    self_inner = np.asarray(
        [
            _gaussian_inner_product_unchecked(mean, covariance, mean, covariance)
            for mean, covariance in zip(means, covariances)
        ],
        dtype=np.float64,
    )
    weights = self_inner * np.sum(coefficients * coefficients, axis=1)
    if float(weights.sum()) <= 1e-20:
        weights = np.ones(means.shape[0], dtype=np.float64)
    weights = weights / weights.sum()
    parent_mean = np.sum(weights[:, None] * means, axis=0)
    centered = means - parent_mean[None, :]
    parent_covariance = np.sum(
        weights[:, None, None]
        * (covariances + centered[:, :, None] * centered[:, None, :]),
        axis=0,
    )
    eigenvalues, eigenvectors = np.linalg.eigh(parent_covariance)
    eigenvalues = np.maximum(eigenvalues, minimum_variance)
    parent_covariance = (eigenvectors * eigenvalues[None, :]) @ eigenvectors.T
    return parent_mean, parent_covariance


def _continuous_fit_error(
    old_means: np.ndarray,
    old_covariances: np.ndarray,
    old_coefficients: np.ndarray,
    basis_means: np.ndarray,
    basis_covariances: np.ndarray,
    coefficient_domain: CoefficientDomain,
) -> tuple[float, np.ndarray]:
    old_gram = _kernel_gram(old_means, old_covariances, old_means, old_covariances)
    basis_gram = _kernel_gram(
        basis_means, basis_covariances, basis_means, basis_covariances
    )
    cross = _kernel_gram(basis_means, basis_covariances, old_means, old_covariances)
    rhs = cross @ old_coefficients
    coefficients = _solve_gram(basis_gram, rhs, coefficient_domain)
    old_energy = float(np.sum(old_coefficients * (old_gram @ old_coefficients)))
    residual = old_energy
    residual -= 2.0 * float(np.sum(coefficients * rhs))
    residual += float(np.sum(coefficients * (basis_gram @ coefficients)))
    tolerance = 1e-10 * max(old_energy, 1.0)
    if residual < -tolerance:
        raise FloatingPointError(f"continuous contraction energy became negative: {residual}")
    return max(residual, 0.0), coefficients


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


class _ContractionEngine:
    _UNRESOLVED = np.uint8(0)
    _RESOLVED = np.uint8(1)
    _READY = np.uint8(2)
    _BLOCKED = np.uint8(3)

    def __init__(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        config: PixelContractionConfig,
        initial_coefficients: np.ndarray | None = None,
        protected_leaf_mask: np.ndarray | None = None,
    ) -> None:
        self.config = config
        self.image = image
        self.mask = mask
        self.height, self.width = image.shape[:2]
        self.initial_count = int(mask.sum())
        if config.target_gaussians > self.initial_count:
            raise ValueError(
                "target_gaussians cannot exceed the number of active source pixels: "
                f"{config.target_gaussians} > {self.initial_count}"
            )

        capacity = self.initial_count
        self.means = np.empty((capacity, 2), dtype=np.float32)
        self.compact_covariances = np.empty((capacity, 3), dtype=np.float32)
        self.coefficients = np.empty((capacity, 3), dtype=np.float32)
        self.active = np.ones(capacity, dtype=bool)
        self.protected = np.zeros(capacity, dtype=bool)
        yy, xx = np.nonzero(mask)
        self.means[:, 0] = xx.astype(np.float32)
        self.means[:, 1] = yy.astype(np.float32)
        leaf_variance = np.float32(config.leaf_scale_px**2)
        self.compact_covariances[:, 0] = leaf_variance
        self.compact_covariances[:, 1] = 0.0
        self.compact_covariances[:, 2] = leaf_variance
        if initial_coefficients is None:
            self.coefficients[:] = image[mask]
        else:
            coefficients = np.asarray(initial_coefficients, dtype=np.float32)
            if coefficients.shape != (capacity, 3):
                raise ValueError(
                    "initial_coefficients must have shape "
                    f"({capacity}, 3), got {coefficients.shape}"
                )
            if not np.isfinite(coefficients).all():
                raise ValueError("initial_coefficients must be finite")
            if config.coefficient_domain == "nonnegative" and (coefficients < 0.0).any():
                raise ValueError(
                    "initial_coefficients must be nonnegative for the nonnegative domain"
                )
            self.coefficients[:] = coefficients
        if protected_leaf_mask is not None:
            self.protected[:] = protected_leaf_mask[mask]
        self.protected_initial_count = int(self.protected.sum())

        self.present: list[np.ndarray] = [mask.copy()]
        while self.present[-1].shape != (1, 1):
            child = self.present[-1]
            child_height, child_width = child.shape
            parent = np.zeros(((child_height + 1) // 2, (child_width + 1) // 2), dtype=bool)
            for offset_y in (0, 1):
                for offset_x in (0, 1):
                    sampled = child[offset_y::2, offset_x::2]
                    parent[: sampled.shape[0], : sampled.shape[1]] |= sampled
            self.present.append(parent)

        self.states = [np.zeros(shape.shape, dtype=np.uint8) for shape in self.present]
        self.atom_a = [np.full(shape.shape, -1, dtype=np.int32) for shape in self.present]
        self.atom_b = [np.full(shape.shape, -1, dtype=np.int32) for shape in self.present]
        self.versions = [np.zeros(shape.shape, dtype=np.uint32) for shape in self.present]
        leaf_ids = np.full(mask.shape, -1, dtype=np.int32)
        leaf_ids[mask] = np.arange(capacity, dtype=np.int32)
        self.states[0][mask] = self._RESOLVED
        self.atom_a[0][:] = leaf_ids

        self.partial_atoms: dict[tuple[int, int, int], tuple[int, ...]] = {}
        self.proposal_cache: dict[tuple[int, int, int, int], tuple[_Proposal, ...]] = {}
        self.evaluation_cache: dict[
            tuple[int, int, int, int], dict[int, tuple[int, _EvaluatedProposal]]
        ] = {}
        self.dirty_boxes: list[tuple[int, int, int, int]] = []
        self.heap: list[tuple[float, int, int, int, int, int]] = []
        self.serial = 0
        self.current_count = self.initial_count
        self.current_render = self._initial_render()
        self.initial_sse = self._masked_sse(self.current_render)
        self.history: list[ContractionEvent] = []
        self.recovery_history: list[RecoveryEvent] = []
        self.recovery_elapsed_seconds = 0.0
        self.ever_touched = np.zeros(capacity, dtype=bool)
        self.pending_touched = np.zeros(capacity, dtype=bool)
        self.recovery_neighbors = np.zeros(capacity, dtype=bool)
        self.last_recovery_action_count = 0
        self.next_recovery_progress_index = 1
        self.batch_index = 0
        self.stop_reason = "no_ready_candidate"
        if config.target_gaussians < self.initial_count:
            self._initialize_frontier()

    def _initial_render(self) -> np.ndarray:
        result = np.zeros((self.height, self.width, 3), dtype=np.float64)
        source = np.zeros((self.height, self.width, 3), dtype=np.float64)
        source[self.mask] = self.coefficients.astype(np.float64)
        variance = float(self.compact_covariances[0, 0])
        radius = max(int(math.ceil(self.config.sigma_cutoff * math.sqrt(variance))), 1)
        tail = self.config.support_fade_alpha * math.exp(-0.5 * self.config.sigma_cutoff**2)
        for delta_y in range(-radius, radius + 1):
            source_y0 = max(0, -delta_y)
            source_y1 = min(self.height, self.height - delta_y)
            destination_y0 = source_y0 + delta_y
            destination_y1 = source_y1 + delta_y
            for delta_x in range(-radius, radius + 1):
                weight = max(
                    math.exp(-0.5 * (delta_x * delta_x + delta_y * delta_y) / variance)
                    - tail,
                    0.0,
                )
                if weight == 0.0:
                    continue
                source_x0 = max(0, -delta_x)
                source_x1 = min(self.width, self.width - delta_x)
                destination_x0 = source_x0 + delta_x
                destination_x1 = source_x1 + delta_x
                result[
                    destination_y0:destination_y1, destination_x0:destination_x1
                ] += weight * source[source_y0:source_y1, source_x0:source_x1]
        return result

    def _masked_sse(self, render: np.ndarray) -> float:
        residual = render[self.mask] - self.image[self.mask].astype(np.float64)
        return float(np.sum(residual * residual))

    def _gaussian_field_for_ids(self, atom_ids: np.ndarray, device: str):
        """Materialize selected engine slots without importing torch at module import time."""
        from .gaussians import GaussianField

        covariances = _covariances_from_compact(self.compact_covariances[atom_ids])
        eigenvalues, eigenvectors = np.linalg.eigh(covariances)
        eigenvalues = np.maximum(eigenvalues, 1e-12)
        first_axes = eigenvectors[:, :, 0]
        rotations = np.arctan2(first_axes[:, 1], first_axes[:, 0])
        scales = np.sqrt(eigenvalues)
        return GaussianField.from_numpy(
            self.means[atom_ids],
            scales.astype(np.float32),
            rotations.astype(np.float32),
            self.coefficients[atom_ids],
            device=device,
        )

    def _render_recovery_field(self, field):
        from .render import render_field

        return render_field(
            field.means,
            field.conics(),
            field.colors,
            field.radii(self.config.sigma_cutoff),
            self.height,
            self.width,
            chunk=self.config.recovery_render_chunk,
            mode=self.config.recovery_renderer,
            opacities=None,
            scales=field.scales(),
            rotations=field.rotations,
            support_fade=self.config.support_fade_alpha > 0.0,
            sigma_cutoff=self.config.sigma_cutoff,
            support_fade_alpha=self.config.support_fade_alpha,
        )

    def _rebuild_frontier(self) -> None:
        """Invalidate geometry-dependent proposal state after a recovery checkpoint."""
        self.heap.clear()
        self.proposal_cache.clear()
        self.evaluation_cache.clear()
        self.dirty_boxes.clear()
        self.serial = 0
        for level in range(1, len(self.states)):
            ready_regions = np.argwhere(self.states[level] == self._READY)
            for region_y, region_x in ready_regions:
                atoms = self._derived_region_atoms(level, int(region_y), int(region_x))
                self._enqueue_region(level, int(region_y), int(region_x), atoms)

    def _recovery_update_weights(self, field):
        """Return per-row Adam update multipliers and attribution telemetry."""

        import torch

        if self.config.recovery_scope != "all_error_weighted":
            weights = torch.ones(
                field.n,
                device=field.means.device,
                dtype=field.means.dtype,
            )
            return weights, {
                "attribution_seconds": 0.0,
                "error_weight_min": 1.0,
                "error_weight_mean": 1.0,
                "error_weight_p50": 1.0,
                "error_weight_p90": 1.0,
                "error_weight_max": 1.0,
                "error_weight_effective_rows": float(field.n),
                "raw_error_score_mean": 0.0,
                "raw_error_score_max": 0.0,
            }

        from .render import render_field

        started = time.perf_counter()
        residual = self.current_render - self.image.astype(np.float64)
        residual_energy = np.mean(residual * residual, axis=2)
        smoothed_error = _mask_aware_smoothed_error(
            residual_energy,
            self.mask,
            self.config.recovery_error_smoothing_sigma_px,
        )
        error_tensor = torch.as_tensor(
            smoothed_error,
            device=field.means.device,
            dtype=field.means.dtype,
        )
        mask_tensor = torch.as_tensor(
            self.mask,
            device=field.means.device,
            dtype=field.means.dtype,
        )
        probe_colors = torch.zeros(
            (field.n, 3),
            device=field.means.device,
            dtype=field.means.dtype,
            requires_grad=True,
        )
        probe_render = render_field(
            field.means.detach(),
            field.conics().detach(),
            probe_colors,
            field.radii(self.config.sigma_cutoff),
            self.height,
            self.width,
            chunk=self.config.recovery_render_chunk,
            mode=self.config.recovery_renderer,
            opacities=None,
            scales=field.scales().detach(),
            rotations=field.rotations.detach(),
            support_fade=self.config.support_fade_alpha > 0.0,
            sigma_cutoff=self.config.sigma_cutoff,
            support_fade_alpha=self.config.support_fade_alpha,
        )
        probe_objective = torch.sum(probe_render[..., 0] * error_tensor)
        probe_objective = probe_objective + torch.sum(
            probe_render[..., 1] * mask_tensor
        )
        probe_gradient = torch.autograd.grad(
            probe_objective,
            probe_colors,
            create_graph=False,
            retain_graph=False,
        )[0]
        numerator = probe_gradient[:, 0].clamp_min(0.0)
        denominator = probe_gradient[:, 1]
        raw_scores = torch.where(
            denominator > 1e-8,
            numerator / denominator.clamp_min(1e-8),
            torch.zeros_like(numerator),
        )
        raw_scores_numpy = raw_scores.detach().cpu().numpy().astype(np.float64)
        weights_numpy = _normalized_error_update_weights(
            raw_scores_numpy,
            power=self.config.recovery_error_weight_power,
            floor=self.config.recovery_error_weight_floor,
            ceiling=self.config.recovery_error_weight_ceiling,
        )
        weights = torch.as_tensor(
            weights_numpy,
            device=field.means.device,
            dtype=field.means.dtype,
        )
        weight_sum = float(np.sum(weights_numpy, dtype=np.float64))
        squared_weight_sum = float(
            np.sum(weights_numpy.astype(np.float64) ** 2, dtype=np.float64)
        )
        effective_rows = (
            weight_sum * weight_sum / squared_weight_sum
            if squared_weight_sum > 0.0
            else 0.0
        )
        return weights, {
            "attribution_seconds": time.perf_counter() - started,
            "error_weight_min": float(np.min(weights_numpy)),
            "error_weight_mean": float(np.mean(weights_numpy)),
            "error_weight_p50": float(np.quantile(weights_numpy, 0.5)),
            "error_weight_p90": float(np.quantile(weights_numpy, 0.9)),
            "error_weight_max": float(np.max(weights_numpy)),
            "error_weight_effective_rows": effective_rows,
            "raw_error_score_mean": float(np.mean(raw_scores_numpy)),
            "raw_error_score_max": float(np.max(raw_scores_numpy)),
        }

    @staticmethod
    def _scale_adam_row_updates(field, previous, row_weights) -> None:
        """Apply per-row learning-rate multipliers after Adam preconditioning."""

        parameters = (
            field.means,
            field.log_scales,
            field.rotations,
            field.colors,
        )
        for parameter, old_value in zip(parameters, previous, strict=True):
            shape = (row_weights.shape[0],) + (1,) * (parameter.ndim - 1)
            multiplier = row_weights.reshape(shape)
            parameter.copy_(old_value + (parameter - old_value) * multiplier)

    def _direct_neighborhood_active_ids(self, seed_ids: np.ndarray) -> np.ndarray:
        """Return active rows in the rounded-center Chebyshev halo of ``seed_ids``."""

        if seed_ids.size == 0:
            return np.empty(0, dtype=np.int64)
        radius = self.config.recovery_neighborhood_radius_px
        seed_xy = np.rint(self.means[seed_ids]).astype(np.int64)
        seed_xy[:, 0] = np.clip(seed_xy[:, 0], 0, self.width - 1)
        seed_xy[:, 1] = np.clip(seed_xy[:, 1], 0, self.height - 1)
        halo = np.zeros((self.height, self.width), dtype=bool)
        for x, y in seed_xy:
            halo[
                max(0, int(y) - radius) : min(self.height, int(y) + radius + 1),
                max(0, int(x) - radius) : min(self.width, int(x) + radius + 1),
            ] = True
        active_ids = np.flatnonzero(self.active)
        active_xy = np.rint(self.means[active_ids]).astype(np.int64)
        active_xy[:, 0] = np.clip(active_xy[:, 0], 0, self.width - 1)
        active_xy[:, 1] = np.clip(active_xy[:, 1], 0, self.height - 1)
        return active_ids[halo[active_xy[:, 1], active_xy[:, 0]]]

    def _recover_pending_rows(self) -> None:
        if self.config.recovery_steps == 0:
            return
        newly_touched_ids = np.flatnonzero(self.active & self.pending_touched)
        if newly_touched_ids.size == 0:
            self.last_recovery_action_count = len(self.history)
            return
        touched_ids = np.flatnonzero(self.active & self.ever_touched)
        candidate_neighbor_ids = np.empty(0, dtype=np.int64)
        new_neighbor_ids = np.empty(0, dtype=np.int64)
        if self.config.recovery_scope == "all_error_weighted":
            atom_ids = np.flatnonzero(self.active)
        elif self.config.recovery_scope == "touched_neighborhood":
            candidate_neighbor_ids = self._direct_neighborhood_active_ids(newly_touched_ids)
            candidate_mask = np.zeros(self.active.shape, dtype=bool)
            candidate_mask[candidate_neighbor_ids] = True
            candidate_mask &= self.active & ~self.ever_touched
            new_mask = candidate_mask & ~self.recovery_neighbors
            new_neighbor_ids = np.flatnonzero(new_mask)
            scope = self.active & (
                self.ever_touched | self.recovery_neighbors | candidate_mask
            )
            atom_ids = np.flatnonzero(scope)
        else:
            atom_ids = touched_ids

        import torch

        started = time.perf_counter()
        before_sse = self._masked_sse(self.current_render)
        field = self._gaussian_field_for_ids(atom_ids, self.config.recovery_device).trainable()
        row_weights, weight_telemetry = self._recovery_update_weights(field)
        anchor_means = field.means.detach().clone()
        anchor_log_scales = field.log_scales.detach().clone()
        anchor_rotations = field.rotations.detach().clone()
        anchor_coefficients = field.colors.detach().clone()
        protected_rows = torch.as_tensor(
            self.protected[atom_ids],
            device=field.means.device,
            dtype=torch.bool,
        )
        mean_limit = self.config.recovery_max_mean_shift_px
        mean_lower = anchor_means - mean_limit
        mean_upper = anchor_means + mean_limit
        mean_lower[:, 0].clamp_(min=0.0)
        mean_lower[:, 1].clamp_(min=0.0)
        mean_upper[:, 0].clamp_(max=float(self.width - 1))
        mean_upper[:, 1].clamp_(max=float(self.height - 1))
        log_scale_limit = self.config.recovery_max_log_scale_shift
        global_log_min = math.log(1e-3)
        global_log_max = math.log(float(max(self.width, self.height)))
        log_scale_lower = (anchor_log_scales - log_scale_limit).clamp(
            min=global_log_min
        )
        log_scale_upper = (anchor_log_scales + log_scale_limit).clamp(
            max=global_log_max
        )
        rotation_limit = self.config.recovery_max_rotation_shift_rad
        rotation_lower = anchor_rotations - rotation_limit
        rotation_upper = anchor_rotations + rotation_limit

        target = torch.as_tensor(
            self.image, device=field.means.device, dtype=field.means.dtype
        )
        objective = torch.as_tensor(self.mask, device=field.means.device, dtype=torch.bool)
        with torch.no_grad():
            initial_optimized_render = self._render_recovery_field(field)
            current = torch.as_tensor(
                self.current_render, device=field.means.device, dtype=field.means.dtype
            )
            frozen_base = current - initial_optimized_render

        optimizer = torch.optim.Adam(
            field.parameter_groups(
                self.config.recovery_lr_means,
                self.config.recovery_lr_scales,
                self.config.recovery_lr_rotations,
                self.config.recovery_lr_coefficients,
            )
        )
        best_sse = before_sse
        best_step = 0
        best_state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None
        best_render: torch.Tensor | None = None
        for step in range(1, self.config.recovery_steps + 1):
            optimizer.zero_grad(set_to_none=True)
            rendered = frozen_base + self._render_recovery_field(field)
            residual = rendered[objective] - target[objective]
            loss = torch.mean(residual.square())
            if not bool(torch.isfinite(loss)):
                break
            loss.backward()
            previous_step = None
            if self.config.recovery_scope == "all_error_weighted":
                previous_step = (
                    field.means.detach().clone(),
                    field.log_scales.detach().clone(),
                    field.rotations.detach().clone(),
                    field.colors.detach().clone(),
                )
            optimizer.step()
            with torch.no_grad():
                if previous_step is not None:
                    self._scale_adam_row_updates(field, previous_step, row_weights)
                if bool(torch.any(protected_rows)):
                    field.means[protected_rows] = anchor_means[protected_rows]
                    field.log_scales[protected_rows] = anchor_log_scales[protected_rows]
                    field.rotations[protected_rows] = anchor_rotations[protected_rows]
                field.means.clamp_(mean_lower, mean_upper)
                field.log_scales.clamp_(log_scale_lower, log_scale_upper)
                field.rotations.clamp_(rotation_lower, rotation_upper)
                if self.config.coefficient_domain == "nonnegative":
                    field.colors.clamp_(min=0.0)
                candidate_render = frozen_base + self._render_recovery_field(field)
                candidate_residual = candidate_render[objective] - target[objective]
                candidate_sse = float(torch.sum(candidate_residual.square()).item())
                if math.isfinite(candidate_sse) and candidate_sse < best_sse:
                    best_sse = candidate_sse
                    best_step = step
                    best_state = (
                        field.means.detach().clone(),
                        field.log_scales.detach().clone(),
                        field.rotations.detach().clone(),
                        field.colors.detach().clone(),
                    )
                    best_render = candidate_render.detach().clone()

        accepted = False
        accepted_new_neighbor_count = 0
        after_sse = before_sse
        tolerance = 1e-10 * max(before_sse, 1.0)
        if best_state is not None and best_render is not None and best_sse < before_sse - tolerance:
            best_means, best_log_scales, best_rotations, best_coefficients = best_state
            means = best_means.cpu().numpy().astype(np.float32)
            scales_squared = np.exp(
                2.0 * best_log_scales.cpu().numpy().astype(np.float64)
            )
            rotations = best_rotations.cpu().numpy().astype(np.float64)
            cosines = np.cos(rotations)
            sines = np.sin(rotations)
            covariance_00 = (
                cosines * cosines * scales_squared[:, 0]
                + sines * sines * scales_squared[:, 1]
            )
            covariance_01 = cosines * sines * (
                scales_squared[:, 0] - scales_squared[:, 1]
            )
            covariance_11 = (
                sines * sines * scales_squared[:, 0]
                + cosines * cosines * scales_squared[:, 1]
            )
            compact = np.stack(
                [covariance_00, covariance_01, covariance_11], axis=1
            ).astype(np.float32)
            coefficients = best_coefficients.cpu().numpy().astype(np.float32)
            candidate_render = best_render.cpu().numpy().astype(np.float64)
            candidate_sse = self._masked_sse(candidate_render)
            if candidate_sse < before_sse - tolerance:
                changed_rows = (
                    torch.any(best_means != anchor_means, dim=1)
                    | torch.any(best_log_scales != anchor_log_scales, dim=1)
                    | (best_rotations != anchor_rotations)
                    | torch.any(best_coefficients != anchor_coefficients, dim=1)
                ).cpu().numpy()
                protected_numpy = self.protected[atom_ids]
                if np.any(protected_numpy):
                    means[protected_numpy] = self.means[atom_ids[protected_numpy]]
                    compact[protected_numpy] = self.compact_covariances[
                        atom_ids[protected_numpy]
                    ]
                self.means[atom_ids] = means
                self.compact_covariances[atom_ids] = compact
                self.coefficients[atom_ids] = coefficients
                self.current_render = candidate_render
                if new_neighbor_ids.size:
                    new_neighbor_rows = np.isin(atom_ids, new_neighbor_ids)
                    accepted_new_ids = atom_ids[changed_rows & new_neighbor_rows]
                    self.recovery_neighbors[accepted_new_ids] = True
                    accepted_new_neighbor_count = int(accepted_new_ids.size)
                after_sse = candidate_sse
                accepted = True
                self._rebuild_frontier()

        self.pending_touched[:] = False
        self.last_recovery_action_count = len(self.history)
        elapsed = time.perf_counter() - started
        self.recovery_elapsed_seconds += elapsed
        self.recovery_history.append(
            RecoveryEvent(
                checkpoint_index=len(self.recovery_history),
                action_count=len(self.history),
                active_count=self.current_count,
                touched_count=int(touched_ids.size),
                newly_touched_count=int(newly_touched_ids.size),
                optimized_count=int(atom_ids.size),
                neighborhood_count=int(
                    np.sum(~self.ever_touched[atom_ids])
                    if self.config.recovery_scope == "touched_neighborhood"
                    else 0
                ),
                new_neighborhood_count=int(new_neighbor_ids.size),
                accepted_new_neighborhood_count=accepted_new_neighbor_count,
                protected_optimized_count=int(np.sum(self.protected[atom_ids])),
                recovery_scope=self.config.recovery_scope,
                attempted_steps=self.config.recovery_steps,
                selected_step=best_step if accepted else 0,
                sse_before=before_sse,
                sse_after=after_sse,
                attribution_seconds=float(weight_telemetry["attribution_seconds"]),
                error_smoothing_sigma_px=(
                    self.config.recovery_error_smoothing_sigma_px
                ),
                error_weight_min=float(weight_telemetry["error_weight_min"]),
                error_weight_mean=float(weight_telemetry["error_weight_mean"]),
                error_weight_p50=float(weight_telemetry["error_weight_p50"]),
                error_weight_p90=float(weight_telemetry["error_weight_p90"]),
                error_weight_max=float(weight_telemetry["error_weight_max"]),
                error_weight_effective_rows=float(
                    weight_telemetry["error_weight_effective_rows"]
                ),
                raw_error_score_mean=float(weight_telemetry["raw_error_score_mean"]),
                raw_error_score_max=float(weight_telemetry["raw_error_score_max"]),
                elapsed_seconds=elapsed,
                accepted=accepted,
            )
        )

    def _recovery_due(self) -> bool:
        if self.config.recovery_steps == 0:
            return False
        if self.config.recovery_schedule == "actions":
            return (
                len(self.history) - self.last_recovery_action_count
                >= self.config.recovery_every_actions
            )
        total_reduction = self.initial_count - self.config.target_gaussians
        completed_reduction = self.initial_count - self.current_count
        return (
            total_reduction > 0
            and self.next_recovery_progress_index
            <= self.config.recovery_progress_checkpoints
            and completed_reduction * self.config.recovery_progress_checkpoints
            >= self.next_recovery_progress_index * total_reduction
        )

    def _recover_when_due(self) -> None:
        if not self._recovery_due():
            return
        self._recover_pending_rows()
        if self.config.recovery_schedule == "progress":
            total_reduction = self.initial_count - self.config.target_gaussians
            completed_reduction = self.initial_count - self.current_count
            crossed = (
                completed_reduction * self.config.recovery_progress_checkpoints
                // total_reduction
            )
            self.next_recovery_progress_index = min(
                int(crossed) + 1,
                self.config.recovery_progress_checkpoints + 1,
            )

    def _initialize_frontier(self) -> None:
        if len(self.states) == 1:
            return
        height, width = self.states[1].shape
        for region_y in range(height):
            for region_x in range(width):
                if self.present[1][region_y, region_x]:
                    self._try_resolve_or_queue(1, region_y, region_x)

    def _child_cells(self, level: int, region_y: int, region_x: int):
        child_level = level - 1
        child_height, child_width = self.states[child_level].shape
        for offset_y in (0, 1):
            child_y = 2 * region_y + offset_y
            if child_y >= child_height:
                continue
            for offset_x in (0, 1):
                child_x = 2 * region_x + offset_x
                if child_x < child_width and self.present[child_level][child_y, child_x]:
                    yield child_y, child_x

    def _derived_region_atoms(self, level: int, region_y: int, region_x: int) -> tuple[int, ...]:
        key = (level, region_y, region_x)
        if key in self.partial_atoms:
            return self.partial_atoms[key]
        atoms: list[int] = []
        for child_y, child_x in self._child_cells(level, region_y, region_x):
            if self.states[level - 1][child_y, child_x] != self._RESOLVED:
                raise RuntimeError("attempted to gather an unresolved quadtree child")
            first = int(self.atom_a[level - 1][child_y, child_x])
            second = int(self.atom_b[level - 1][child_y, child_x])
            if first >= 0:
                atoms.append(first)
            if second >= 0:
                atoms.append(second)
        if len(atoms) != len(set(atoms)) or any(not self.active[atom] for atom in atoms):
            raise RuntimeError("quadtree region contains stale or duplicate atom ids")
        return tuple(atoms)

    def _try_resolve_or_queue(self, level: int, region_y: int, region_x: int) -> None:
        while level < len(self.states):
            if not self.present[level][region_y, region_x]:
                return
            if self.states[level][region_y, region_x] != self._UNRESOLVED:
                return
            children = list(self._child_cells(level, region_y, region_x))
            if any(
                self.states[level - 1][child_y, child_x] != self._RESOLVED
                for child_y, child_x in children
            ):
                return
            atoms = self._derived_region_atoms(level, region_y, region_x)
            if not atoms:
                raise RuntimeError("present quadtree region has no active atoms")
            if len(atoms) == 1:
                self.states[level][region_y, region_x] = self._RESOLVED
                self.atom_a[level][region_y, region_x] = atoms[0]
                if level + 1 == len(self.states):
                    return
                region_y //= 2
                region_x //= 2
                level += 1
                continue
            self.states[level][region_y, region_x] = self._READY
            self._enqueue_region(level, region_y, region_x, atoms)
            return

    def _atom_arrays(self, atom_ids: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        indices = np.asarray(atom_ids, dtype=np.int64)
        means = self.means[indices].astype(np.float64)
        covariances = _covariances_from_compact(self.compact_covariances[indices])
        coefficients = self.coefficients[indices].astype(np.float64)
        return means, covariances, coefficients

    def _replacement_proposal(
        self, kind: str, atom_ids: tuple[int, ...], resolves_region: bool
    ) -> _Proposal:
        means, covariances, coefficients = self._atom_arrays(atom_ids)
        parent_mean, parent_covariance = _moment_parent(
            means, covariances, coefficients, minimum_variance=1e-6
        )
        basis_means = parent_mean[None, :]
        basis_covariances = parent_covariance[None, :, :]
        analytic_sse, _ = _continuous_fit_error(
            means,
            covariances,
            coefficients,
            basis_means,
            basis_covariances,
            self.config.coefficient_domain,
        )
        return _Proposal(
            kind=kind,
            remove_ids=atom_ids,
            means=basis_means,
            covariances=basis_covariances,
            analytic_sse=analytic_sse,
            rows_saved=len(atom_ids) - 1,
            resolves_region=resolves_region,
        )

    def _detail_proposal(self, atom_ids: tuple[int, ...]) -> _Proposal:
        means, covariances, coefficients = self._atom_arrays(atom_ids)
        parent_mean, parent_covariance = _moment_parent(
            means, covariances, coefficients, minimum_variance=1e-6
        )
        best_error = float("inf")
        best_index = -1
        best_means: np.ndarray | None = None
        best_covariances: np.ndarray | None = None
        for retained_index in range(len(atom_ids)):
            basis_means = np.stack([parent_mean, means[retained_index]], axis=0)
            basis_covariances = np.stack(
                [parent_covariance, covariances[retained_index]], axis=0
            )
            error, _ = _continuous_fit_error(
                means,
                covariances,
                coefficients,
                basis_means,
                basis_covariances,
                self.config.coefficient_domain,
            )
            if error < best_error - 1e-15:
                best_error = error
                best_index = retained_index
                best_means = basis_means
                best_covariances = basis_covariances
        if best_index < 0 or best_means is None or best_covariances is None:
            raise RuntimeError("failed to select a parent-plus-detail basis")
        return _Proposal(
            kind=f"parent_detail:{best_index}",
            remove_ids=atom_ids,
            means=best_means,
            covariances=best_covariances,
            analytic_sse=best_error,
            rows_saved=len(atom_ids) - 2,
            resolves_region=True,
        )

    def _protected_detail_proposal(
        self, atom_ids: tuple[int, ...]
    ) -> _Proposal | None:
        """Carry up to two protected leaf geometries exactly while reducing other rows."""

        protected_ids = tuple(atom_id for atom_id in atom_ids if self.protected[atom_id])
        if not protected_ids or len(protected_ids) > 2:
            return None
        unprotected_ids = tuple(
            atom_id for atom_id in atom_ids if not self.protected[atom_id]
        )
        if len(protected_ids) == 1:
            if len(unprotected_ids) < 2:
                return None
            protected_means, protected_covariances, _ = self._atom_arrays(protected_ids)
            means, covariances, coefficients = self._atom_arrays(unprotected_ids)
            parent_mean, parent_covariance = _moment_parent(
                means,
                covariances,
                coefficients,
                minimum_variance=1e-6,
            )
            basis_means = np.concatenate(
                [protected_means, parent_mean[None, :]], axis=0
            )
            basis_covariances = np.concatenate(
                [protected_covariances, parent_covariance[None, :, :]], axis=0
            )
            output_protection = (True, False)
        else:
            if not unprotected_ids:
                return None
            basis_means, basis_covariances, _ = self._atom_arrays(protected_ids)
            output_protection = (True, True)
        old_means, old_covariances, old_coefficients = self._atom_arrays(atom_ids)
        analytic_sse, _ = _continuous_fit_error(
            old_means,
            old_covariances,
            old_coefficients,
            basis_means,
            basis_covariances,
            self.config.coefficient_domain,
        )
        return _Proposal(
            kind=f"protected_detail:{len(protected_ids)}",
            remove_ids=atom_ids,
            means=basis_means,
            covariances=basis_covariances,
            analytic_sse=analytic_sse,
            rows_saved=len(atom_ids) - len(output_protection),
            resolves_region=True,
            protected_outputs=output_protection,
        )

    def _region_contractible(self, atom_ids: tuple[int, ...]) -> bool:
        protected_count = sum(self.protected[atom_id] for atom_id in atom_ids)
        if protected_count == 0:
            return len(atom_ids) >= 2
        unprotected_count = len(atom_ids) - protected_count
        protected_resolve = (
            (protected_count == 1 and unprotected_count >= 2)
            or (protected_count == 2 and unprotected_count >= 1)
        )
        return protected_resolve or unprotected_count >= 2

    def _region_proposals(self, atom_ids: tuple[int, ...]) -> list[_Proposal]:
        atom_count = len(atom_ids)
        if atom_count < 2:
            raise RuntimeError("ready region must contain at least two atoms")
        proposals: list[_Proposal] = []
        protected_ids = tuple(atom_id for atom_id in atom_ids if self.protected[atom_id])
        if protected_ids:
            protected_proposal = self._protected_detail_proposal(atom_ids)
            if protected_proposal is not None:
                proposals.append(protected_proposal)
        else:
            if atom_count <= 4:
                proposals.append(self._replacement_proposal("hard", atom_ids, True))
            if atom_count >= 3:
                proposals.append(self._detail_proposal(atom_ids))
        if atom_count > 2:
            pair_ids = tuple(
                atom_id for atom_id in atom_ids if not self.protected[atom_id]
            )
            pair_proposals = [
                self._replacement_proposal("pair", pair, False)
                for pair in itertools.combinations(pair_ids, 2)
            ]
            pair_proposals.sort(
                key=lambda proposal: (
                    proposal.analytic_sse,
                    proposal.remove_ids,
                )
            )
            proposals.extend(pair_proposals[: self.config.pair_shortlist])
        return proposals

    def _enqueue_region(
        self, level: int, region_y: int, region_x: int, atom_ids: tuple[int, ...]
    ) -> None:
        if not self._region_contractible(atom_ids):
            self.states[level][region_y, region_x] = self._BLOCKED
            return
        # Keep the image-sized frontier cheap.  This proxy schedules *regions* only; once a region
        # is popped, all of its contraction options are ranked by the closed-form Gaussian
        # product and then by exact discrete distortion.  Caching those expensive options only
        # for popped regions avoids an O(pixels) forest of small NumPy objects.
        coefficients = self.coefficients[np.asarray(atom_ids, dtype=np.int64)].astype(np.float64)
        centered = coefficients - coefficients.mean(axis=0, keepdims=True)
        priority = float(np.sum(centered * centered)) / (
            max(len(atom_ids) - 1, 1) * self.config.estimated_row_bytes
        )
        self.serial += 1
        heapq.heappush(
            self.heap,
            (
                priority,
                self.serial,
                level,
                region_y,
                region_x,
                int(self.versions[level][region_y, region_x]),
            ),
        )

    @staticmethod
    def _cache_key(
        level: int, region_y: int, region_x: int, version: int
    ) -> tuple[int, int, int, int]:
        return level, region_y, region_x, version

    def _cached_proposals(
        self,
        level: int,
        region_y: int,
        region_x: int,
        version: int,
        atom_ids: tuple[int, ...],
    ) -> tuple[_Proposal, ...]:
        key = self._cache_key(level, region_y, region_x, version)
        cached = self.proposal_cache.get(key)
        if cached is None:
            cached = tuple(self._region_proposals(atom_ids))
            self.proposal_cache[key] = cached
        return cached

    def _cached_evaluations(
        self,
        level: int,
        region_y: int,
        region_x: int,
        version: int,
        atom_ids: tuple[int, ...],
        gap: int,
    ) -> tuple[_EvaluatedProposal, ...]:
        key = self._cache_key(level, region_y, region_x, version)
        proposals = self._cached_proposals(
            level, region_y, region_x, version, atom_ids
        )
        eligible = [
            index
            for index, proposal in enumerate(proposals)
            if proposal.rows_saved <= gap
            and not (
                level + 1 == len(self.states)
                and proposal.resolves_region
                and proposal.means.shape[0] > self.config.target_gaussians
            )
        ]
        if self.config.pair_policy == "exact_count":
            non_pair = [index for index in eligible if proposals[index].kind != "pair"]
            if non_pair:
                eligible = non_pair
        eligible.sort(
            key=lambda index: (
                proposals[index].analytic_sse
                / (proposals[index].rows_saved * self.config.estimated_row_bytes),
                proposals[index].kind,
                proposals[index].remove_ids,
            )
        )
        selected_indices = eligible[: self.config.exact_option_shortlist]
        if eligible:
            smallest_action = min(
                eligible,
                key=lambda index: (
                    proposals[index].rows_saved,
                    proposals[index].analytic_sse,
                    proposals[index].kind,
                ),
            )
            if smallest_action not in selected_indices:
                selected_indices.append(smallest_action)

        cache = self.evaluation_cache.setdefault(key, {})
        values: list[_EvaluatedProposal] = []
        for index in selected_indices:
            proposal = proposals[index]
            cached = cache.get(index)
            if cached is not None:
                evaluated_generation, previous = cached
                dirty = self.dirty_boxes[evaluated_generation:]
                if not any(_boxes_overlap(previous.patch_xyxy, box) for box in dirty):
                    cache[index] = (len(self.dirty_boxes), previous)
                    values.append(previous)
                    continue
            evaluated = self._evaluate(
                proposal, level, region_y, region_x, version
            )
            cache[index] = (len(self.dirty_boxes), evaluated)
            values.append(evaluated)
        return tuple(values)

    def _geometry_bounds(
        self, means: np.ndarray, covariances: np.ndarray
    ) -> tuple[int, int, int, int]:
        minimum_x, minimum_y = self.width, self.height
        maximum_x, maximum_y = -1, -1
        for mean, covariance in zip(means, covariances):
            radius_x = max(
                int(math.ceil(self.config.sigma_cutoff * math.sqrt(covariance[0, 0]))), 1
            )
            radius_y = max(
                int(math.ceil(self.config.sigma_cutoff * math.sqrt(covariance[1, 1]))), 1
            )
            center_x, center_y = np.rint(mean).astype(np.int64)
            left = max(int(center_x) - radius_x, 0)
            right = min(int(center_x) + radius_x, self.width - 1)
            top = max(int(center_y) - radius_y, 0)
            bottom = min(int(center_y) + radius_y, self.height - 1)
            if left <= right and top <= bottom:
                minimum_x = min(minimum_x, left)
                maximum_x = max(maximum_x, right)
                minimum_y = min(minimum_y, top)
                maximum_y = max(maximum_y, bottom)
        if maximum_x < minimum_x or maximum_y < minimum_y:
            raise RuntimeError("candidate support does not intersect the image")
        return minimum_x, minimum_y, maximum_x, maximum_y

    def _kernel_matrix(
        self,
        means: np.ndarray,
        covariances: np.ndarray,
        bounds: tuple[int, int, int, int],
    ) -> np.ndarray:
        left, top, right, bottom = bounds
        yy, xx = np.mgrid[top : bottom + 1, left : right + 1]
        flat_x = xx.reshape(-1).astype(np.float64)
        flat_y = yy.reshape(-1).astype(np.float64)
        matrix = np.zeros((flat_x.shape[0], means.shape[0]), dtype=np.float64)
        tail = self.config.support_fade_alpha * math.exp(-0.5 * self.config.sigma_cutoff**2)
        for column, (mean, covariance) in enumerate(zip(means, covariances)):
            radius_x = max(
                int(math.ceil(self.config.sigma_cutoff * math.sqrt(covariance[0, 0]))), 1
            )
            radius_y = max(
                int(math.ceil(self.config.sigma_cutoff * math.sqrt(covariance[1, 1]))), 1
            )
            center_x, center_y = np.rint(mean).astype(np.int64)
            active = (
                (flat_x >= center_x - radius_x)
                & (flat_x <= center_x + radius_x)
                & (flat_y >= center_y - radius_y)
                & (flat_y <= center_y + radius_y)
            )
            determinant = covariance[0, 0] * covariance[1, 1] - covariance[0, 1] ** 2
            inverse_00 = covariance[1, 1] / determinant
            inverse_01 = -covariance[0, 1] / determinant
            inverse_11 = covariance[0, 0] / determinant
            delta_x = flat_x - mean[0]
            delta_y = flat_y - mean[1]
            quadratic = (
                inverse_00 * delta_x * delta_x
                + 2.0 * inverse_01 * delta_x * delta_y
                + inverse_11 * delta_y * delta_y
            )
            weights = np.maximum(np.exp(-0.5 * quadratic) - tail, 0.0)
            matrix[:, column] = np.where(active, weights, 0.0)
        return matrix

    def _evaluate(
        self,
        proposal: _Proposal,
        level: int,
        region_y: int,
        region_x: int,
        version: int,
    ) -> _EvaluatedProposal:
        old_means, old_covariances, old_coefficients = self._atom_arrays(proposal.remove_ids)
        # Acceptance is evaluated after the same float32 materialization used by the field.
        new_means = proposal.means.astype(np.float32).astype(np.float64)
        new_compact = _covariances_to_compact(
            proposal.covariances, np.dtype(np.float32)
        )
        new_covariances = _covariances_from_compact(new_compact)
        old_bounds = self._geometry_bounds(old_means, old_covariances)
        new_bounds = self._geometry_bounds(new_means, new_covariances)
        bounds = (
            min(old_bounds[0], new_bounds[0]),
            min(old_bounds[1], new_bounds[1]),
            max(old_bounds[2], new_bounds[2]),
            max(old_bounds[3], new_bounds[3]),
        )
        left, top, right, bottom = bounds
        old_weights = self._kernel_matrix(old_means, old_covariances, bounds)
        new_weights = self._kernel_matrix(new_means, new_covariances, bounds)
        current_patch = self.current_render[top : bottom + 1, left : right + 1]
        old_contribution = (old_weights @ old_coefficients).reshape(current_patch.shape)
        base = current_patch - old_contribution
        target_patch = self.image[top : bottom + 1, left : right + 1].astype(np.float64)
        objective = self.mask[top : bottom + 1, left : right + 1].reshape(-1)
        residual_target = (target_patch - base).reshape(-1, 3)
        objective_weights = new_weights[objective]
        gram = objective_weights.T @ objective_weights
        rhs = objective_weights.T @ residual_target[objective]
        coefficients = _solve_gram(gram, rhs, self.config.coefficient_domain)
        coefficients = coefficients.astype(np.float32).astype(np.float64)
        patch_render = base + (new_weights @ coefficients).reshape(base.shape)
        old_residual = (current_patch - target_patch).reshape(-1, 3)[objective]
        new_residual = (patch_render - target_patch).reshape(-1, 3)[objective]
        exact_delta = float(
            np.sum(new_residual * new_residual) - np.sum(old_residual * old_residual)
        )
        slope = exact_delta / (
            proposal.rows_saved * self.config.estimated_row_bytes
        )
        return _EvaluatedProposal(
            proposal=proposal,
            level=level,
            region_y=region_y,
            region_x=region_x,
            version=version,
            means=new_means.astype(np.float32),
            covariances=new_covariances.astype(np.float32),
            coefficients=coefficients.astype(np.float32),
            patch_xyxy=bounds,
            patch_render=patch_render,
            exact_sse_delta=exact_delta,
            exact_slope=slope,
        )

    def _entry_valid(self, entry: tuple[float, int, int, int, int, int]) -> bool:
        _, _, level, region_y, region_x, version = entry
        return (
            self.states[level][region_y, region_x] == self._READY
            and int(self.versions[level][region_y, region_x]) == version
        )

    def _apply(self, evaluated: _EvaluatedProposal) -> None:
        proposal = evaluated.proposal
        key = (evaluated.level, evaluated.region_y, evaluated.region_x)
        if (
            self.states[evaluated.level][evaluated.region_y, evaluated.region_x] != self._READY
            or int(self.versions[evaluated.level][evaluated.region_y, evaluated.region_x])
            != evaluated.version
        ):
            raise RuntimeError("attempted to apply a stale contraction proposal")
        region_atoms = self._derived_region_atoms(*key)
        if not set(proposal.remove_ids).issubset(region_atoms):
            raise RuntimeError("contraction proposal no longer belongs to its quadtree region")

        left, top, right, bottom = evaluated.patch_xyxy
        self.current_render[top : bottom + 1, left : right + 1] = evaluated.patch_render
        self.dirty_boxes.append(evaluated.patch_xyxy)
        output_count = evaluated.means.shape[0]
        protected_outputs = (
            proposal.protected_outputs
            if proposal.protected_outputs
            else (False,) * output_count
        )
        if len(protected_outputs) != output_count:
            raise RuntimeError("proposal protection flags do not match its output rows")
        protected_inputs = sum(self.protected[atom_id] for atom_id in proposal.remove_ids)
        if sum(protected_outputs) != protected_inputs:
            raise RuntimeError("contraction proposal would lose or invent a protected leaf")
        output_ids = proposal.remove_ids[:output_count]
        for atom_id in proposal.remove_ids:
            self.active[atom_id] = False
            self.protected[atom_id] = False
        output_indices = np.asarray(output_ids, dtype=np.int64)
        self.means[output_indices] = evaluated.means
        self.compact_covariances[output_indices] = _covariances_to_compact(
            evaluated.covariances.astype(np.float64), np.dtype(np.float32)
        )
        self.coefficients[output_indices] = evaluated.coefficients
        self.active[output_indices] = True
        self.protected[output_indices] = np.asarray(protected_outputs, dtype=bool)
        self.ever_touched[output_indices] = True
        self.pending_touched[output_indices] = True

        count_before = self.current_count
        self.current_count -= proposal.rows_saved
        if proposal.resolves_region:
            self.partial_atoms.pop(key, None)
            self.states[evaluated.level][evaluated.region_y, evaluated.region_x] = self._RESOLVED
            self.atom_a[evaluated.level][evaluated.region_y, evaluated.region_x] = output_ids[0]
            self.atom_b[evaluated.level][evaluated.region_y, evaluated.region_x] = (
                output_ids[1] if output_count == 2 else -1
            )
        else:
            remove_set = set(proposal.remove_ids)
            updated_atoms: list[int] = []
            inserted = False
            for atom_id in region_atoms:
                if atom_id in remove_set:
                    if not inserted:
                        updated_atoms.extend(output_ids)
                        inserted = True
                else:
                    updated_atoms.append(atom_id)
            if not inserted or len(updated_atoms) != len(region_atoms) - proposal.rows_saved:
                raise RuntimeError("failed to update a partially contracted region")
            self.partial_atoms[key] = tuple(updated_atoms)

        self.versions[evaluated.level][evaluated.region_y, evaluated.region_x] += 1
        cache_key = self._cache_key(
            evaluated.level, evaluated.region_y, evaluated.region_x, evaluated.version
        )
        self.proposal_cache.pop(cache_key, None)
        self.evaluation_cache.pop(cache_key, None)
        self.history.append(
            ContractionEvent(
                action_index=len(self.history),
                batch_index=self.batch_index,
                level=evaluated.level,
                region_x=evaluated.region_x,
                region_y=evaluated.region_y,
                kind=proposal.kind,
                count_before=count_before,
                count_after=self.current_count,
                rows_removed=len(proposal.remove_ids),
                rows_added=output_count,
                analytic_continuous_sse=proposal.analytic_sse,
                exact_discrete_sse_delta=evaluated.exact_sse_delta,
                exact_distortion_per_estimated_byte=evaluated.exact_slope,
                patch_xyxy=evaluated.patch_xyxy,
            )
        )

        if proposal.resolves_region:
            if evaluated.level + 1 < len(self.states):
                self._try_resolve_or_queue(
                    evaluated.level + 1,
                    evaluated.region_y // 2,
                    evaluated.region_x // 2,
                )
        else:
            self._enqueue_region(
                evaluated.level,
                evaluated.region_y,
                evaluated.region_x,
                self.partial_atoms[key],
            )

    def run(self) -> None:
        while self.current_count > self.config.target_gaussians:
            if self.config.max_actions is not None and len(self.history) >= self.config.max_actions:
                self._recover_pending_rows()
                self.stop_reason = "action_limit"
                return
            gap = self.current_count - self.config.target_gaussians
            entries: list[tuple[float, int, int, int, int, int]] = []
            evaluated: list[_EvaluatedProposal] = []
            # A hard distortion ceiling is a stopping rule, so a proxy-shortlisted page cannot
            # prove it.  In that opt-in mode scan the complete current frontier before declaring
            # the ceiling exhausted; the ordinary unconstrained path keeps its bounded page.
            entry_limit = (
                math.inf
                if self.config.max_exact_distortion_per_estimated_byte is not None
                else self.config.proposal_batch_size
            )
            while self.heap and len(entries) < entry_limit:
                entry = heapq.heappop(self.heap)
                if not self._entry_valid(entry):
                    continue
                entries.append(entry)
                _, _, level, region_y, region_x, version = entry
                atom_ids = self._derived_region_atoms(level, region_y, region_x)
                for item in self._cached_evaluations(
                    level, region_y, region_x, version, atom_ids, gap
                ):
                    evaluated.append(item)
            if not entries:
                self._recover_pending_rows()
                self.stop_reason = (
                    "protected_topology_limit"
                    if self.protected_initial_count > 0
                    and any(np.any(state == self._BLOCKED) for state in self.states)
                    else "no_ready_candidate"
                )
                return
            evaluated.sort(
                key=lambda item: (
                    item.exact_slope,
                    item.proposal.analytic_sse,
                    item.level,
                    item.region_y,
                    item.region_x,
                    item.proposal.kind,
                    item.proposal.remove_ids,
                )
            )
            selected: list[_EvaluatedProposal] = []
            selected_regions: set[tuple[int, int, int]] = set()
            selected_boxes: list[tuple[int, int, int, int]] = []
            saved = 0
            action_slots = self.config.merge_batch_size
            if self.config.max_actions is not None:
                action_slots = min(action_slots, self.config.max_actions - len(self.history))
            for item in evaluated:
                region_key = (item.level, item.region_y, item.region_x)
                if region_key in selected_regions:
                    continue
                if saved + item.proposal.rows_saved > gap:
                    continue
                if any(_boxes_overlap(item.patch_xyxy, box) for box in selected_boxes):
                    continue
                limit = self.config.max_exact_distortion_per_estimated_byte
                if limit is not None and item.exact_slope > limit:
                    continue
                selected.append(item)
                selected_regions.add(region_key)
                selected_boxes.append(item.patch_xyxy)
                saved += item.proposal.rows_saved
                if len(selected) >= action_slots or saved == gap:
                    break

            if not selected:
                for entry in entries:
                    if self._entry_valid(entry):
                        heapq.heappush(self.heap, entry)
                self._recover_pending_rows()
                self.stop_reason = (
                    "distortion_limit"
                    if self.config.max_exact_distortion_per_estimated_byte is not None
                    else "no_exact_count_action"
                )
                return

            for item in selected:
                self._apply(item)
            for entry in entries:
                region_key = (entry[2], entry[3], entry[4])
                if region_key not in selected_regions and self._entry_valid(entry):
                    heapq.heappush(self.heap, entry)
            self.batch_index += 1
            self._recover_when_due()

        self._recover_pending_rows()
        self.stop_reason = "target_reached"

    def observation_field(self) -> ObservationField2D:
        active_ids = np.flatnonzero(self.active)
        means = self.means[active_ids].astype(np.float32, copy=True)
        covariances = _covariances_from_compact(self.compact_covariances[active_ids])
        eigenvalues, eigenvectors = np.linalg.eigh(covariances)
        eigenvalues = np.maximum(eigenvalues, 1e-12)
        first_axes = eigenvectors[:, :, 0]
        rotations = np.arctan2(first_axes[:, 1], first_axes[:, 0])
        log_scales = 0.5 * np.log(eigenvalues)

        packed_alpha = None
        alpha_semantics = AlphaSemantics()
        if not self.mask.all():
            packed_alpha = pack_alpha(self.mask)
            alpha_semantics = AlphaSemantics(
                payload_encoding="binary_exact_packbits_little",
                matting_mode="multiply_alpha",
                boundary_policy="unconstrained",
            )
        semantics = FieldSemantics(
            coefficient_domain=self.config.coefficient_domain,
            support=SupportSemantics(
                mode="axis_aligned_bbox",
                sigma_cutoff=self.config.sigma_cutoff,
                fade_alpha=self.config.support_fade_alpha,
                minimum_radius_px=1,
            ),
            alpha=alpha_semantics,
        )
        adaptation = adapt_direct_additive(
            means_xy=means,
            log_scales_xy=log_scales.astype(np.float32),
            rotations_rad=rotations.astype(np.float32),
            rgb_coeff=self.coefficients[active_ids].astype(np.float32, copy=True),
            canvas_crop=CanvasCropTransform(
                canvas_width=self.width,
                canvas_height=self.height,
                crop_x=0,
                crop_y=0,
                crop_width=self.width,
                crop_height=self.height,
            ),
            semantics=semantics,
            packed_alpha=packed_alpha,
        )
        return adaptation.require_pixel_exact()


def _validated_image(image: object) -> np.ndarray:
    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a NumPy array")
    if image.ndim != 3 or image.shape[2] != 3 or image.shape[0] < 1 or image.shape[1] < 1:
        raise ValueError("image must have nonempty HWC shape (H, W, 3)")
    if image.dtype.kind not in "fiu" or not np.isfinite(image).all():
        raise ValueError("image must contain finite numeric values")
    result = np.asarray(image, dtype=np.float32)
    if (result < 0.0).any() or (result > 1.0).any():
        raise ValueError("image values must lie in [0, 1]")
    return np.array(result, dtype=np.float32, order="C", copy=True)


def _validated_mask(mask: object | None, shape: tuple[int, int]) -> np.ndarray:
    if mask is None:
        return np.ones(shape, dtype=bool)
    if not isinstance(mask, np.ndarray) or mask.shape != shape or mask.dtype != np.bool_:
        raise ValueError(f"mask must be a bool NumPy array with shape {shape}")
    result = np.array(mask, dtype=bool, order="C", copy=True)
    if not result.any():
        raise ValueError("mask must contain at least one active pixel")
    return result


def _validated_protected_leaf_mask(
    protected_leaf_mask: object | None,
    active_mask: np.ndarray,
    target_gaussians: int,
) -> np.ndarray:
    if protected_leaf_mask is None:
        return np.zeros(active_mask.shape, dtype=bool)
    if (
        not isinstance(protected_leaf_mask, np.ndarray)
        or protected_leaf_mask.shape != active_mask.shape
        or protected_leaf_mask.dtype != np.bool_
    ):
        raise ValueError(
            "protected_leaf_mask must be a bool NumPy array with shape "
            f"{active_mask.shape}"
        )
    result = np.array(protected_leaf_mask, dtype=bool, order="C", copy=True)
    if np.any(result & ~active_mask):
        raise ValueError("protected_leaf_mask must be a subset of the active mask")
    protected_count = int(result.sum())
    if protected_count > target_gaussians:
        raise ValueError(
            "protected_leaf_mask count cannot exceed target_gaussians: "
            f"{protected_count} > {target_gaussians}"
        )
    return result


def contract_image(
    image: np.ndarray,
    config: PixelContractionConfig,
    *,
    mask: np.ndarray | None = None,
    initial_coefficients: np.ndarray | None = None,
    protected_leaf_mask: np.ndarray | None = None,
) -> PixelContractionResult:
    """Contract an image's implicit pixel field into a direct additive Gaussian field.

    ``initial_coefficients`` optionally supplies one signed RGB row per active mask pixel in
    NumPy's boolean-index order.  It is useful when overlapping pixel leaves first require an
    appearance prefilter; the default remains the historical source-RGB initialization.
    ``protected_leaf_mask`` reserves selected pixel means/covariances from topology removal and
    geometry optimization while allowing their RGB coefficients to be locally refitted.
    """

    if not isinstance(config, PixelContractionConfig):
        raise TypeError("config must be PixelContractionConfig")
    source = _validated_image(image)
    active_mask = _validated_mask(mask, source.shape[:2])
    protected_mask = _validated_protected_leaf_mask(
        protected_leaf_mask,
        active_mask,
        config.target_gaussians,
    )
    started = time.perf_counter()
    engine = _ContractionEngine(
        source,
        active_mask,
        config,
        initial_coefficients=initial_coefficients,
        protected_leaf_mask=protected_mask,
    )
    engine.run()
    field = engine.observation_field()
    elapsed = time.perf_counter() - started
    reconstruction_raw = np.asarray(engine.current_render, dtype=np.float32)
    reconstruction = (
        reconstruction_raw * active_mask[:, :, None]
        if field.semantics.alpha.matting_mode == "multiply_alpha"
        else reconstruction_raw.copy()
    )
    final_sse = engine._masked_sse(engine.current_render)
    touched_active_rows = int(np.sum(engine.active & engine.ever_touched))
    untouched_active_rows = int(np.sum(engine.active & ~engine.ever_touched))
    protected_active_rows = int(np.sum(engine.active & engine.protected))
    if protected_active_rows != engine.protected_initial_count:
        raise RuntimeError(
            "protected leaf count changed during contraction: "
            f"{engine.protected_initial_count} -> {protected_active_rows}"
        )
    if protected_active_rows:
        expected_y, expected_x = np.nonzero(protected_mask)
        expected_means = np.stack([expected_x, expected_y], axis=1).astype(np.float32)
        observed_ids = np.flatnonzero(engine.active & engine.protected)
        observed_means = engine.means[observed_ids]
        expected_order = np.lexsort((expected_means[:, 0], expected_means[:, 1]))
        observed_order = np.lexsort((observed_means[:, 0], observed_means[:, 1]))
        if not np.array_equal(
            observed_means[observed_order], expected_means[expected_order]
        ):
            raise RuntimeError("protected leaf means changed during contraction or recovery")
        leaf_variance = np.float32(config.leaf_scale_px**2)
        expected_compact = np.tile(
            np.asarray([leaf_variance, 0.0, leaf_variance], dtype=np.float32),
            (protected_active_rows, 1),
        )
        if not np.array_equal(
            engine.compact_covariances[observed_ids], expected_compact
        ):
            raise RuntimeError("protected leaf covariances changed during contraction or recovery")
    canonical_raw_bytes = sum(array.nbytes for array in field._array_items().values())
    alpha_bytes = 0 if field.packed_alpha is None else int(field.packed_alpha.nbytes)
    return PixelContractionResult(
        field=field,
        reconstruction_raw=reconstruction_raw,
        reconstruction=reconstruction,
        history=tuple(engine.history),
        recovery_history=tuple(engine.recovery_history),
        initial_count=engine.initial_count,
        target_count=config.target_gaussians,
        final_count=field.n,
        initial_sse=engine.initial_sse,
        final_sse=final_sse,
        elapsed_seconds=elapsed,
        stop_reason=engine.stop_reason,
        estimated_field_bytes=field.n * config.estimated_row_bytes + alpha_bytes,
        canonical_raw_bytes=int(canonical_raw_bytes),
        touched_active_rows=touched_active_rows,
        untouched_active_rows=untouched_active_rows,
        recovery_neighbor_active_rows=int(
            np.sum(engine.active & engine.recovery_neighbors & ~engine.ever_touched)
        ),
        protected_initial_rows=engine.protected_initial_count,
        protected_active_rows=protected_active_rows,
        blocked_regions=int(
            sum(np.sum(state == engine._BLOCKED) for state in engine.states)
        ),
    )


def observation_to_gaussian_field(
    field: ObservationField2D, *, device: str = "cpu"
):
    """Materialize a direct Observation Field V2 as a torch ``GaussianField``."""

    if not isinstance(field, ObservationField2D):
        raise TypeError("field must be ObservationField2D")
    if field.semantics.renderer_equation != "additive_rgb_peak_one_v1":
        raise ValueError("only direct additive peak-one fields can be materialized")
    from .gaussians import GaussianField

    return GaussianField.from_numpy(
        np.array(field.means_xy, copy=True),
        np.exp(field.log_scales_xy).copy(),
        np.array(field.rotations_rad, copy=True),
        np.array(field.rgb_coeff, copy=True),
        device=device,
        filter_variance=(
            None
            if field.filter_variance_px2 is None
            else np.array(field.filter_variance_px2, copy=True)
        ),
    )


def render_observation_field(
    field: ObservationField2D,
    *,
    device: str = "cpu",
    renderer: str = "additive",
    render_chunk: int = 4096,
    apply_declared_alpha: bool = True,
) -> np.ndarray:
    """Render a direct field through the maintained torch additive renderer."""

    if renderer not in ("additive", "cuda_additive", "cuda_tiled_additive"):
        raise ValueError("renderer must be additive, cuda_additive, or cuda_tiled_additive")
    gaussian_field = observation_to_gaussian_field(field, device=device)
    from .render import render_field

    dilation = field.semantics.filtering.aa_dilation_px2
    import torch

    with torch.no_grad():
        rendered = render_field(
            gaussian_field.means,
            gaussian_field.conics(dilation),
            gaussian_field.colors,
            gaussian_field.radii(field.semantics.support.sigma_cutoff, dilation),
            field.canvas_crop.crop_height,
            field.canvas_crop.crop_width,
            chunk=render_chunk,
            mode=renderer,
            opacities=None,
            scales=gaussian_field.effective_scales(dilation),
            rotations=gaussian_field.rotations,
            support_fade=field.semantics.support.fade_alpha > 0.0,
            sigma_cutoff=field.semantics.support.sigma_cutoff,
            support_fade_alpha=field.semantics.support.fade_alpha,
        )
        if field.background_rgb is not None:
            background = torch.as_tensor(
                field.background_rgb, device=rendered.device, dtype=rendered.dtype
            )
            rendered = rendered + background
        if (
            apply_declared_alpha
            and field.packed_alpha is not None
            and field.semantics.alpha.matting_mode == "multiply_alpha"
        ):
            alpha = torch.as_tensor(
                field.alpha_mask(), device=rendered.device, dtype=rendered.dtype
            )
            rendered = rendered * alpha[:, :, None]
    return rendered.detach().cpu().numpy()


def _rescue_seed_locations(
    residual: np.ndarray,
    mask: np.ndarray,
    *,
    max_rows: int,
    nms_radius_px: int,
) -> np.ndarray:
    """Choose deterministic high-error pixel centers with Chebyshev-radius NMS."""

    scores = np.mean(np.asarray(residual, dtype=np.float64) ** 2, axis=2)
    scores[~mask] = -np.inf
    order = np.argsort(-scores.reshape(-1), kind="stable")
    blocked = np.zeros(mask.shape, dtype=bool)
    height, width = mask.shape
    selected: list[tuple[int, int]] = []
    for flat_index in order:
        y, x = divmod(int(flat_index), width)
        if not mask[y, x] or blocked[y, x]:
            continue
        selected.append((x, y))
        if len(selected) >= max_rows:
            break
        x0 = max(0, x - nms_radius_px)
        x1 = min(width, x + nms_radius_px + 1)
        y0 = max(0, y - nms_radius_px)
        y1 = min(height, y + nms_radius_px + 1)
        blocked[y0:y1, x0:x1] = True
    return np.asarray(selected, dtype=np.int64).reshape(-1, 2)


def rescue_observation_field(
    field: ObservationField2D,
    target: np.ndarray,
    config: LocalRescueConfig,
    *,
    mask: np.ndarray | None = None,
) -> LocalRescueResult:
    """Repair a frozen direct-additive field with bounded signed residual rows.

    The base field is never materialized as trainable state. Candidate centers are selected once
    from its raw residual with stable descending RGB-MSE order and Chebyshev-radius NMS. Rescue
    geometry is fixed; only the new RGB coefficients are optimized. The accepted checkpoint is
    the lexicographic minimum of raw-domain normalized worst-pixel/7x7-patch violation and SSE,
    with the unchanged base field included as checkpoint ``-1``.
    """

    if not isinstance(field, ObservationField2D):
        raise TypeError("field must be ObservationField2D")
    if not isinstance(config, LocalRescueConfig):
        raise TypeError("config must be LocalRescueConfig")
    if field.semantics.renderer_equation != "additive_rgb_peak_one_v1":
        raise ValueError("local rescue requires a direct additive peak-one field")
    if field.semantics.coefficient_domain != "signed":
        raise ValueError("local rescue requires signed RGB coefficients")
    if field.structural_mass is not None:
        raise ValueError("local rescue does not yet define structural-mass rows")
    if field.filter_variance_px2 is not None:
        raise ValueError("local rescue does not yet define per-row filter variance")

    source = _validated_image(target)
    if source.shape[:2] != field.crop_shape:
        raise ValueError(
            f"target shape {source.shape[:2]} does not match field crop {field.crop_shape}"
        )
    if mask is None and field.packed_alpha is not None:
        active_mask = field.alpha_mask()
    else:
        active_mask = _validated_mask(mask, source.shape[:2])

    started = time.perf_counter()
    base_raw = render_observation_field(
        field,
        device=config.device,
        renderer=config.renderer,
        render_chunk=config.render_chunk,
        apply_declared_alpha=False,
    ).astype(np.float32, copy=False)
    residual = source - base_raw
    selected_xy = _rescue_seed_locations(
        residual,
        active_mask,
        max_rows=config.max_rows,
        nms_radius_px=config.nms_radius_px,
    )
    if selected_xy.shape[0] == 0:  # Defensive: validated masks always contain foreground.
        raise RuntimeError("local rescue could not select a foreground seed")

    import torch
    import torch.nn.functional as torch_functional

    from .gaussians import GaussianField
    from .render import render_field

    device = torch.device(config.device)
    height, width = source.shape[:2]
    seed_x = selected_xy[:, 0]
    seed_y = selected_xy[:, 1]
    seed_colors = residual[seed_y, seed_x]
    rescue_field = GaussianField.from_numpy(
        selected_xy.astype(np.float32),
        np.full((selected_xy.shape[0], 2), config.scale_px, dtype=np.float32),
        np.zeros(selected_xy.shape[0], dtype=np.float32),
        seed_colors.astype(np.float32, copy=False),
        device=config.device,
    )
    rescue_field.colors.requires_grad_(True)
    optimizer = torch.optim.Adam([rescue_field.colors], lr=config.learning_rate)
    base_tensor = torch.as_tensor(
        np.ascontiguousarray(base_raw), device=device, dtype=torch.float32
    )
    target_tensor = torch.as_tensor(
        np.ascontiguousarray(source), device=device, dtype=torch.float32
    )
    mask_tensor = torch.as_tensor(active_mask, device=device, dtype=torch.bool)
    mask_float = mask_tensor.to(dtype=torch.float32)
    dilation = field.semantics.filtering.aa_dilation_px2
    conics = rescue_field.conics(dilation)
    scales = rescue_field.effective_scales(dilation)
    radii = rescue_field.radii(field.semantics.support.sigma_cutoff, dilation)
    patch_side = min(7, height, width)
    if patch_side % 2 == 0:
        patch_side -= 1
    patch_side = max(patch_side, 1)

    def render_rescue():
        return render_field(
            rescue_field.means,
            conics,
            rescue_field.colors,
            radii,
            height,
            width,
            chunk=config.render_chunk,
            mode=config.renderer,
            opacities=None,
            scales=scales,
            rotations=rescue_field.rotations,
            support_fade=field.semantics.support.fade_alpha > 0.0,
            sigma_cutoff=field.semantics.support.sigma_cutoff,
            support_fade_alpha=field.semantics.support.fade_alpha,
        )

    def raw_metrics(reconstruction):
        residual_tensor = reconstruction - target_tensor
        pixel_mse = torch.mean(residual_tensor.square(), dim=2)
        foreground_pixel_mse = pixel_mse[mask_tensor]
        sse = torch.sum(residual_tensor[mask_tensor].square())
        pixel_rmse_max = torch.sqrt(torch.max(foreground_pixel_mse))
        black_matted_pixel_mse = pixel_mse * mask_float
        patch_mse = torch_functional.avg_pool2d(
            black_matted_pixel_mse[None, None],
            kernel_size=patch_side,
            stride=1,
            padding=0,
        )
        patch_rmse_max = torch.sqrt(torch.max(patch_mse))
        violation = torch.maximum(
            pixel_rmse_max / config.pixel_rmse_threshold,
            patch_rmse_max / config.patch7_rmse_threshold,
        )
        return sse, pixel_rmse_max, patch_rmse_max, violation, foreground_pixel_mse

    with torch.no_grad():
        (
            initial_sse_tensor,
            initial_pixel_tensor,
            initial_patch_tensor,
            initial_violation_tensor,
            _,
        ) = raw_metrics(base_tensor)
    initial_sse = float(initial_sse_tensor.item())
    initial_pixel = float(initial_pixel_tensor.item())
    initial_patch = float(initial_patch_tensor.item())
    initial_violation = float(initial_violation_tensor.item())
    best_key = (initial_violation, initial_sse)
    best_step = -1
    best_colors: np.ndarray | None = None
    best_rescue_raw: np.ndarray | None = None
    best_metrics = (initial_sse, initial_pixel, initial_patch, initial_violation)

    for step in range(config.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        rescue_raw_tensor = render_rescue()
        reconstruction_tensor = base_tensor + rescue_raw_tensor
        sse, pixel_max, patch_max, violation, foreground_pixel_mse = raw_metrics(
            reconstruction_tensor
        )
        candidate_key = (float(violation.detach().item()), float(sse.detach().item()))
        if candidate_key < best_key:
            best_key = candidate_key
            best_step = step
            best_colors = rescue_field.colors.detach().cpu().numpy().copy()
            best_rescue_raw = rescue_raw_tensor.detach().cpu().numpy().copy()
            best_metrics = (
                candidate_key[1],
                float(pixel_max.detach().item()),
                float(patch_max.detach().item()),
                candidate_key[0],
            )
        if step == config.steps:
            break
        tail_count = max(
            1,
            int(math.ceil(config.tail_fraction * foreground_pixel_mse.numel())),
        )
        tail_loss = torch.topk(
            foreground_pixel_mse,
            k=tail_count,
            largest=True,
            sorted=False,
        ).values.mean()
        loss = foreground_pixel_mse.mean() + config.tail_weight * tail_loss
        loss.backward()
        optimizer.step()

    if best_colors is None or best_rescue_raw is None:
        result_field = field
        result_raw = np.array(base_raw, copy=True)
        accepted_xy = np.empty((0, 2), dtype=np.int64)
        rows_added = 0
    else:
        result_field = ObservationField2D(
            means_xy=np.concatenate(
                [field.means_xy, selected_xy.astype(field.means_xy.dtype)], axis=0
            ),
            log_scales_xy=np.concatenate(
                [
                    field.log_scales_xy,
                    np.full(
                        (selected_xy.shape[0], 2),
                        math.log(config.scale_px),
                        dtype=field.log_scales_xy.dtype,
                    ),
                ],
                axis=0,
            ),
            rotations_rad=np.concatenate(
                [
                    field.rotations_rad,
                    np.zeros(selected_xy.shape[0], dtype=field.rotations_rad.dtype),
                ],
                axis=0,
            ),
            rgb_coeff=np.concatenate(
                [field.rgb_coeff, best_colors.astype(field.rgb_coeff.dtype)], axis=0
            ),
            canvas_crop=field.canvas_crop,
            semantics=field.semantics,
            background_rgb=field.background_rgb,
            packed_alpha=field.packed_alpha,
            camera=field.camera,
            schema_version=field.schema_version,
        )
        result_raw = base_raw + best_rescue_raw
        accepted_xy = selected_xy
        rows_added = int(selected_xy.shape[0])

    reconstruction = np.asarray(result_raw, dtype=np.float32)
    if (
        result_field.packed_alpha is not None
        and result_field.semantics.alpha.matting_mode == "multiply_alpha"
    ):
        reconstruction = reconstruction * result_field.alpha_mask()[:, :, None]
    final_sse, final_pixel, final_patch, final_violation = best_metrics
    return LocalRescueResult(
        field=result_field,
        reconstruction_raw=np.asarray(result_raw, dtype=np.float32),
        reconstruction=np.asarray(reconstruction, dtype=np.float32),
        selected_xy=accepted_xy,
        rows_added=rows_added,
        selected_step=best_step,
        initial_sse=initial_sse,
        final_sse=final_sse,
        violation_before=initial_violation,
        violation_after=final_violation,
        pixel_rmse_max_before=initial_pixel,
        pixel_rmse_max_after=final_pixel,
        patch7_rmse_max_before=initial_patch,
        patch7_rmse_max_after=final_patch,
        elapsed_seconds=time.perf_counter() - started,
    )
