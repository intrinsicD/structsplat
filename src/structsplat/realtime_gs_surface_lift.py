"""Optional visibility-ordered alpha-shell lift for codec-native realtime-gs views.

The module stays importable without torch or realtime-gs.  The optional dependencies are imported
inside :func:`initialize_visibility_ordered_surface` only when a caller explicitly executes the
cross-repository lift.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math
import time
from typing import Any, Callable, Sequence

from .realtime_gs_adapter import (
    RealtimeGSCodecNativeView,
    make_alpha_support_backend,
)


@dataclass(frozen=True)
class VisibilityOrderedSurfaceLiftConfig:
    """Controls that are independent of realtime-gs's existing CompactCarve configuration."""

    support_soft_coverage: float = 0.95
    coverage_scale: float = 1.0
    neutral_color_std_sigma: float = 1e12
    apply_surface_cover: bool = True
    surface_cover_isotropic: bool = False
    surface_cover_min_planarity: float = 0.0

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.support_soft_coverage)
            or not 0.0 < self.support_soft_coverage < 1.0
        ):
            raise ValueError("support_soft_coverage must be finite and lie in (0,1)")
        if not math.isfinite(self.coverage_scale) or self.coverage_scale <= 0.0:
            raise ValueError("coverage_scale must be finite and positive")
        if (
            not math.isfinite(self.neutral_color_std_sigma)
            or self.neutral_color_std_sigma <= 0.0
        ):
            raise ValueError("neutral_color_std_sigma must be finite and positive")
        if not isinstance(self.apply_surface_cover, bool):
            raise TypeError("apply_surface_cover must be bool")
        if not isinstance(self.surface_cover_isotropic, bool):
            raise TypeError("surface_cover_isotropic must be bool")
        if (
            not math.isfinite(self.surface_cover_min_planarity)
            or not 0.0 <= self.surface_cover_min_planarity < 1.0
        ):
            raise ValueError("surface_cover_min_planarity must lie in [0,1)")


@dataclass(frozen=True)
class VisibilityOrderedSurfaceLiftResult:
    """Raw alpha-shell placement, optional covered result, and complete mechanism diagnostics."""

    initialization: Any
    raw_initialization: Any
    support_backends: tuple[Any, ...]
    diagnostics: dict[str, Any]


def _summary(values: Any) -> dict[str, float]:
    values = values.detach().double()
    return {
        "min": float(values.min()),
        "p10": float(values.quantile(0.10)),
        "median": float(values.median()),
        "p90": float(values.quantile(0.90)),
        "max": float(values.max()),
    }


def initialize_visibility_ordered_surface(
    inputs: Any,
    views: Sequence[RealtimeGSCodecNativeView],
    carve_config: Any,
    config: VisibilityOrderedSurfaceLiftConfig | None = None,
    *,
    progress_callback: Callable[[Any], None] | None = None,
    candidate_audit_callback: Callable[[Any], None] | None = None,
) -> VisibilityOrderedSurfaceLiftResult:
    """Lift the first maximal alpha-support point on every selected source ray.

    Sparse structural mass remains the unchanged CompactCarve anchor proposal distribution.  Each
    alpha backend reconstructs a constant declared soft coverage inside its packet mask, while a
    very large color standard deviation makes selection depend only on support.  ``torch.argmax``
    returns the first index on a maximal support plateau, providing the visibility order.  The
    ordinary appearance query still supplies the selected color.

    When requested, realtime-gs's existing cover reconciliation replaces only covariance and
    opacity after placement; means, colors/SH, lineage, count, and packet bytes stay unchanged.
    """
    config = config or VisibilityOrderedSurfaceLiftConfig()
    if len(views) == 0:
        raise ValueError("views must be non-empty")
    if len(views) != int(inputs.n_views):
        raise ValueError("views must contain one paired codec-native view per input view")
    for index, (view, field) in enumerate(zip(views, inputs.observations, strict=True)):
        if not isinstance(view, RealtimeGSCodecNativeView):
            raise TypeError(f"views[{index}] must be RealtimeGSCodecNativeView")
        if view.structural_field is not field:
            raise ValueError(f"views[{index}] does not own inputs.observations[{index}]")

    try:
        import torch
        from rtgs.lift.compact_carve import CompactCarveConfig, CompactCarveInitializer
        from rtgs.lift.surfel_init import SurfelInitConfig, reconcile_covariances
    except ImportError as exc:  # pragma: no cover - isolated import test covers the boundary
        raise RuntimeError(
            "visibility-ordered surface lifting requires torch and the optional realtime-gs package"
        ) from exc
    if not isinstance(carve_config, CompactCarveConfig):
        raise TypeError("carve_config must be rtgs.lift.compact_carve.CompactCarveConfig")

    support_backends = tuple(
        make_alpha_support_backend(
            view,
            coverage_scale=config.coverage_scale,
            soft_coverage=config.support_soft_coverage,
        )
        for view in views
    )
    support_config = replace(
        carve_config,
        coverage_scale=config.coverage_scale,
        color_std_sigma=config.neutral_color_std_sigma,
    )
    structural_pairs_before = tuple(
        int(getattr(view.query_backend.structural_backend, "total_pairs_evaluated", 0))
        for view in views
    )
    captured: list[Any] = []

    def capture(audit: Any) -> None:
        captured.append(audit)
        if candidate_audit_callback is not None:
            candidate_audit_callback(audit)

    placement_started = time.perf_counter()
    raw = CompactCarveInitializer(support_config).initialize(
        inputs,
        backends=list(support_backends),
        candidate_audit_callback=capture,
        progress_callback=progress_callback,
    )
    if torch.cuda.is_available() and any(
        getattr(view.query_backend._appearance, "is_cuda", False) for view in views
    ):
        torch.cuda.synchronize()
    placement_seconds = time.perf_counter() - placement_started
    structural_pairs_after = tuple(
        int(getattr(view.query_backend.structural_backend, "total_pairs_evaluated", 0))
        for view in views
    )
    if structural_pairs_after != structural_pairs_before:
        raise RuntimeError("alpha-support placement unexpectedly queried the sparse structural index")
    if len(captured) != 1:
        raise RuntimeError("CompactCarve did not emit exactly one candidate audit")
    audit = captured[0]
    selected = audit.selected_candidate_indices
    selected_depth_indices = audit.candidate_best_depth_indices[selected]
    selected_depth_sigmas = audit.candidate_depth_sigmas[selected]

    cover_diagnostics = None
    cover_seconds = 0.0
    initialization = raw
    if config.apply_surface_cover:
        cover_config = SurfelInitConfig(
            isotropic=config.surface_cover_isotropic,
            use_resolution_floor=False,
            min_planarity=config.surface_cover_min_planarity,
        )
        cover_started = time.perf_counter()
        covered = reconcile_covariances(raw.gaussians, cover_config)
        cover_seconds = time.perf_counter() - cover_started
        cover_diagnostics = covered.diagnostics
        initialization = replace(raw, gaussians=covered.gaussians)
        if not torch.equal(initialization.gaussians.means, raw.gaussians.means):
            raise RuntimeError("surface-cover reconciliation changed placement means")
        if not torch.equal(initialization.gaussians.sh, raw.gaussians.sh):
            raise RuntimeError("surface-cover reconciliation changed appearance coefficients")

    diagnostics: dict[str, Any] = {
        "schema": "structsplat.visibility_ordered_surface_lift.v1",
        "ownership": {
            "ray_proposals": "sparse_structural_measure",
            "depth_support": "exact_packet_alpha",
            "selected_radiance": "codec_native_appearance_query",
            "render_extent": (
                "local_surface_cover"
                if config.apply_surface_cover
                else "compact_carve_localization_covariance"
            ),
        },
        "depth_rule": "first_sample_attaining_maximum_multiview_alpha_support",
        "support_soft_coverage": config.support_soft_coverage,
        "reconstructed_soft_coverage": [
            backend.reconstructed_soft_coverage() for backend in support_backends
        ],
        "support_weight_inside": [backend.weight_inside for backend in support_backends],
        "support_query_count": sum(backend.total_queries for backend in support_backends),
        "support_query_points": sum(backend.total_points for backend in support_backends),
        "support_index_entries": sum(backend.n_entries for backend in support_backends),
        "support_new_payload_bytes": sum(backend.payload_bytes for backend in support_backends),
        "structural_pairs_before": list(structural_pairs_before),
        "structural_pairs_after": list(structural_pairs_after),
        "selected_depth_index": _summary(selected_depth_indices),
        "selected_depth_sigma": _summary(selected_depth_sigmas),
        "placement_seconds": placement_seconds,
        "surface_cover_seconds": cover_seconds,
        "surface_cover": cover_diagnostics,
        "raw_compact_carve": raw.diagnostics,
    }
    initialization = replace(
        initialization,
        diagnostics={**raw.diagnostics, "visibility_ordered_surface_lift": diagnostics},
    )
    return VisibilityOrderedSurfaceLiftResult(
        initialization=initialization,
        raw_initialization=raw,
        support_backends=support_backends,
        diagnostics=diagnostics,
    )


__all__ = [
    "VisibilityOrderedSurfaceLiftConfig",
    "VisibilityOrderedSurfaceLiftResult",
    "initialize_visibility_ordered_surface",
]
