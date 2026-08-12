"""Safeguarded all-row RGB projection for a pure additive Gaussian endpoint (HIER-024).

This default-off wrapper adapts an opacity-free ``GaussianField`` exactly to Observation Field V2,
invokes the existing matrix-free coefficient projector, and converts the result back without
changing geometry or renderer semantics.  Torch remains a lazy dependency.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field
import math
from typing import TYPE_CHECKING, Mapping

import numpy as np

from .contraction_refinement import (
    CoefficientProjectionConfig,
    CoefficientProjectionResult,
    project_contracted_coefficients,
)
from .observation_field import (
    CanvasCropTransform,
    adapt_factorized_additive_gaussian_field,
)


if TYPE_CHECKING:
    from .gaussians import GaussianField


_ADDITIVE_RENDERERS = frozenset(("additive", "cuda_additive", "cuda_tiled_additive"))
_SAFETY_KEYS = frozenset(("raw_mse", "ms_ssim", "lpips", "pixel_max", "patch7_max"))


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be >= 1, got {result}")
    return result


def _finite_nonnegative(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise TypeError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and >= 0, got {result}")
    return result


def _projection_mask(value: object, shape: tuple[int, int]) -> np.ndarray:
    if not isinstance(value, np.ndarray) or value.dtype != np.bool_ or value.shape != shape:
        raise ValueError(f"mask must be a bool NumPy array with shape {shape}")
    result = np.array(value, dtype=bool, order="C", copy=True)
    if not result.any():
        raise ValueError("mask must contain at least one active pixel")
    return result


@dataclass(frozen=True)
class EndpointAppearanceProjectionConfig:
    """Exact adapter and frozen HIER-024 projection controls."""

    solver: CoefficientProjectionConfig = dataclass_field(
        default_factory=lambda: CoefficientProjectionConfig(
            tolerance=1e-6,
            max_iterations=48,
            ridge=1e-8,
            coefficient_abs_limit=16.0,
            regularization_center="input",
            solver_start="input",
            frozen_base_mode="explicit",
            allow_unsafe_stage_zero_reconditioning=False,
            selection_mode="transaction",
        )
    )
    renderer: str = "additive"
    render_chunk: int = 256
    sigma_cutoff: float = 3.0
    support_fade_alpha: float = 0.0
    aa_dilation_px2: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.solver, CoefficientProjectionConfig):
            raise TypeError("solver must be CoefficientProjectionConfig")
        if self.renderer not in _ADDITIVE_RENDERERS:
            expected = ", ".join(sorted(_ADDITIVE_RENDERERS))
            raise ValueError(f"renderer must be one of {expected}")
        object.__setattr__(
            self, "render_chunk", _positive_integer(self.render_chunk, "render_chunk")
        )
        for name in ("sigma_cutoff", "support_fade_alpha", "aa_dilation_px2"):
            object.__setattr__(self, name, _finite_nonnegative(getattr(self, name), name))
        if self.sigma_cutoff <= 0.0:
            raise ValueError("sigma_cutoff must be > 0")
        if self.support_fade_alpha > 1.0:
            raise ValueError("support_fade_alpha must be <= 1")


@dataclass(frozen=True)
class EndpointAppearanceProjectionResult:
    """Projected pure-additive endpoint plus the existing solver receipt."""

    field: "GaussianField"
    reconstruction_raw: np.ndarray
    projection: CoefficientProjectionResult
    geometry_exact: bool

    def __post_init__(self) -> None:
        reconstruction = np.array(self.reconstruction_raw, dtype=np.float32, order="C", copy=True)
        if reconstruction.ndim != 3 or reconstruction.shape[2] != 3:
            raise ValueError("reconstruction_raw must have HWC RGB shape")
        if not np.isfinite(reconstruction).all():
            raise ValueError("reconstruction_raw must be finite")
        reconstruction.flags.writeable = False
        object.__setattr__(self, "reconstruction_raw", reconstruction)
        if not isinstance(self.projection, CoefficientProjectionResult):
            raise TypeError("projection must be CoefficientProjectionResult")
        if not isinstance(self.geometry_exact, bool):
            raise TypeError("geometry_exact must be bool")
        if not self.geometry_exact:
            raise ValueError("coefficient projection changed endpoint geometry")
        if self.field.opacities is not None or self.field.color_grads is not None:
            raise ValueError("projected endpoint must be opacity-free constant-color additive")


@dataclass(frozen=True)
class ProjectionSafetyConfig:
    """Frozen target-known rollback tolerances for the projected proposal."""

    coefficient_abs_limit: float = 16.0
    ms_ssim_tolerance: float = 1e-5
    lpips_tolerance: float = 0.0
    local_tolerance: float = 1e-6

    def __post_init__(self) -> None:
        for name in (
            "coefficient_abs_limit",
            "ms_ssim_tolerance",
            "lpips_tolerance",
            "local_tolerance",
        ):
            object.__setattr__(self, name, _finite_nonnegative(getattr(self, name), name))
        if self.coefficient_abs_limit <= 0.0:
            raise ValueError("coefficient_abs_limit must be > 0")


@dataclass(frozen=True)
class ProjectionSafetyDecision:
    selected: bool
    reason: str
    clauses: Mapping[str, bool]

    def __post_init__(self) -> None:
        if not isinstance(self.selected, bool):
            raise TypeError("selected must be bool")
        if not isinstance(self.reason, str) or not self.reason:
            raise ValueError("reason must be a non-empty string")
        clauses = dict(self.clauses)
        if not clauses or not all(isinstance(value, bool) for value in clauses.values()):
            raise TypeError("clauses must map names to bool values")
        object.__setattr__(self, "clauses", clauses)
        if self.selected != all(clauses.values()):
            raise ValueError("selected must equal the conjunction of safety clauses")


def _metric_record(value: object, name: str) -> dict[str, float]:
    if not isinstance(value, Mapping) or set(value) != _SAFETY_KEYS:
        raise ValueError(f"{name} must contain exactly {sorted(_SAFETY_KEYS)}")
    record = {}
    for key in sorted(_SAFETY_KEYS):
        item = value[key]
        if isinstance(item, bool) or not isinstance(item, (int, float, np.integer, np.floating)):
            raise TypeError(f"{name}.{key} must be numeric")
        record[key] = float(item)
    return record


def select_safe_projection(
    incoming_metrics: Mapping[str, float],
    proposal_metrics: Mapping[str, float],
    *,
    proposal_finite: bool,
    coefficient_abs_max: float,
    config: ProjectionSafetyConfig | None = None,
) -> ProjectionSafetyDecision:
    """Select only a strict-MSE, perceptual/local-safe bounded projection proposal."""

    if not isinstance(proposal_finite, bool):
        raise TypeError("proposal_finite must be bool")
    incoming = _metric_record(incoming_metrics, "incoming_metrics")
    proposal = _metric_record(proposal_metrics, "proposal_metrics")
    cfg = config or ProjectionSafetyConfig()
    coefficient = float(coefficient_abs_max)
    all_metrics_finite = all(
        math.isfinite(value) for value in (*incoming.values(), *proposal.values())
    )
    clauses = {
        "finite": proposal_finite and all_metrics_finite and math.isfinite(coefficient),
        "bounded": math.isfinite(coefficient) and coefficient <= cfg.coefficient_abs_limit,
        "strict_lower_raw_mse": (all_metrics_finite and proposal["raw_mse"] < incoming["raw_mse"]),
        "ms_ssim_safe": (
            all_metrics_finite
            and proposal["ms_ssim"] >= incoming["ms_ssim"] - cfg.ms_ssim_tolerance
        ),
        "lpips_safe": (
            all_metrics_finite and proposal["lpips"] <= incoming["lpips"] + cfg.lpips_tolerance
        ),
        "pixel_max_safe": (
            all_metrics_finite
            and proposal["pixel_max"] <= incoming["pixel_max"] + cfg.local_tolerance
        ),
        "patch7_max_safe": (
            all_metrics_finite
            and proposal["patch7_max"] <= incoming["patch7_max"] + cfg.local_tolerance
        ),
    }
    selected = all(clauses.values())
    failed = [name for name, passed in clauses.items() if not passed]
    reason = "selected" if selected else "rollback:" + ",".join(failed)
    return ProjectionSafetyDecision(selected, reason, clauses)


def project_additive_endpoint(
    field: "GaussianField",
    target: np.ndarray,
    *,
    config: EndpointAppearanceProjectionConfig | None = None,
    device: str = "cpu",
    mask: np.ndarray | None = None,
) -> EndpointAppearanceProjectionResult:
    """Run the existing all-row coefficient solve without changing Gaussian geometry.

    ``mask`` limits the encoder-side least-squares objective. It is not persisted in the pure
    additive endpoint. Omitting it preserves the historical full-canvas solve.
    """

    from .gaussians import GaussianField
    import torch

    if not isinstance(field, GaussianField):
        raise TypeError("field must be GaussianField")
    if field.n < 1:
        raise ValueError("field must contain at least one Gaussian")
    if field.opacities is not None or field.color_grads is not None:
        raise ValueError("projection requires opacity-free constant RGB coefficients")
    if not isinstance(target, np.ndarray) or target.ndim != 3 or target.shape[2] != 3:
        raise ValueError("target must have HWC RGB shape")
    if target.dtype.kind not in "fiu" or not np.isfinite(target).all():
        raise ValueError("target must contain finite numeric values")
    source = np.asarray(target, dtype=np.float32)
    height, width = source.shape[:2]
    active = (
        np.ones((height, width), dtype=bool)
        if mask is None
        else _projection_mask(mask, (height, width))
    )
    cfg = config or EndpointAppearanceProjectionConfig()
    adaptation = adapt_factorized_additive_gaussian_field(
        field,
        canvas_crop=CanvasCropTransform(width, height, 0, 0, width, height),
        coefficient_domain="signed",
        sigma_cutoff=cfg.sigma_cutoff,
        support_fade_alpha=cfg.support_fade_alpha,
        aa_dilation_px2=cfg.aa_dilation_px2,
    )
    observation = adaptation.require_pixel_exact()
    all_rows = np.ones(observation.n, dtype=bool)
    no_rows = np.zeros(observation.n, dtype=bool)
    projection = project_contracted_coefficients(
        observation,
        source,
        active,
        all_rows,
        no_rows,
        config=cfg.solver,
        device=device,
        renderer=cfg.renderer,
        render_chunk=cfg.render_chunk,
    )
    observation_geometry_exact = bool(
        np.array_equal(projection.field.means_xy, observation.means_xy)
        and np.array_equal(projection.field.log_scales_xy, observation.log_scales_xy)
        and np.array_equal(projection.field.rotations_rad, observation.rotations_rad)
        and np.array_equal(projection.field.filter_variance_px2, observation.filter_variance_px2)
    )
    projected = GaussianField(
        field.means.detach().clone(),
        field.log_scales.detach().clone(),
        field.rotations.detach().clone(),
        torch.as_tensor(
            np.array(projection.field.rgb_coeff, copy=True),
            device=field.colors.device,
            dtype=field.colors.dtype,
        ).clone(),
        opacities=None,
        scale_max=(None if field.scale_max is None else field.scale_max.detach().clone()),
        color_grads=None,
        background_mask=(
            None if field.background_mask is None else field.background_mask.detach().clone()
        ),
        filter_variance=(
            None if field.filter_variance is None else field.filter_variance.detach().clone()
        ),
    )
    geometry_exact = bool(
        observation_geometry_exact
        and torch.equal(projected.means, field.means)
        and torch.equal(projected.log_scales, field.log_scales)
        and torch.equal(projected.rotations, field.rotations)
    )
    return EndpointAppearanceProjectionResult(
        field=projected,
        reconstruction_raw=projection.reconstruction_raw,
        projection=projection,
        geometry_exact=geometry_exact,
    )
