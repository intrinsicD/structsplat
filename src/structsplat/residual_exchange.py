"""Guarded exact-count residual column exchange for direct additive fields.

The method is a deterministic, default-off research reference.  It interprets a fitted
``ObservationField2D`` as an active set: a residual atom can enter only by replacing an existing
row, and the maintained renderer must approve every transaction.  Pricing uses a sparse list of
finite-support samples and never materializes a dense pixel-by-Gaussian matrix.

Torch remains a lazy dependency so importing this module preserves the NumPy-only analysis
boundary.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math
import time

import numpy as np

from .observation_field import ObservationField2D
from .progressive_residual_quadtree import progressive_artifact_metrics


def _integer(value: object, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {result}")
    return result


def _finite(value: object, name: str, *, minimum: float = 0.0, strict: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise TypeError(f"{name} must be a finite number")
    result = float(value)
    invalid = result <= minimum if strict else result < minimum
    if not math.isfinite(result) or invalid:
        relation = ">" if strict else ">="
        raise ValueError(f"{name} must be finite and {relation} {minimum}, got {result}")
    return result


def _image(value: object, shape: tuple[int, int] | None, name: str) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a NumPy array")
    if value.ndim != 3 or value.shape[2] != 3 or min(value.shape[:2]) < 1:
        raise ValueError(f"{name} must have non-empty HWC RGB shape")
    if shape is not None and value.shape[:2] != shape:
        raise ValueError(f"{name} must have spatial shape {shape}")
    if value.dtype.kind not in "fiu" or not np.isfinite(value).all():
        raise ValueError(f"{name} must contain finite numeric values")
    return np.array(value, dtype=np.float32, order="C", copy=True)


def _mask(value: object, shape: tuple[int, int]) -> np.ndarray:
    if not isinstance(value, np.ndarray) or value.dtype != np.bool_ or value.shape != shape:
        raise ValueError(f"mask must be a bool NumPy array with shape {shape}")
    result = np.array(value, dtype=bool, order="C", copy=True)
    if not result.any():
        raise ValueError("mask must contain at least one active pixel")
    return result


def _readonly(array: np.ndarray) -> np.ndarray:
    result = np.array(array, order="C", copy=True)
    result.flags.writeable = False
    return result


@dataclass(frozen=True)
class ResidualExchangeConfig:
    """Frozen search and transaction settings for residual column exchange."""

    candidate_shapes: tuple[tuple[float, float, float], ...] = (
        (0.18, 0.18, 0.0),
        (0.30, 0.30, 0.0),
        (0.45, 0.45, 0.0),
    )
    max_exchanges: int = 128
    site_count: int = 96
    site_nms_radius_px: int = 1
    donor_count: int = 64
    proposal_frontier: int = 24
    coefficient_abs_limit: float = 16.0
    minimum_sse_gain: float = 1e-10
    pricing_absolute_tolerance: float = 2e-6
    local_absolute_tolerance: float = 1e-12
    renderer_parity_tolerance: float = 2e-6
    pixel_rmse_threshold: float = 0.02
    patch7_rmse_threshold: float = 0.01

    def __post_init__(self) -> None:
        try:
            shapes = tuple(tuple(shape) for shape in self.candidate_shapes)
        except TypeError as exc:
            raise TypeError("candidate_shapes must be an iterable of (sx, sy, rotation) triples") from exc
        if not shapes:
            raise ValueError("candidate_shapes must not be empty")
        normalized: list[tuple[float, float, float]] = []
        for index, shape in enumerate(shapes):
            if len(shape) != 3:
                raise ValueError(f"candidate_shapes[{index}] must have three values")
            sx = _finite(shape[0], f"candidate_shapes[{index}].sx", strict=True)
            sy = _finite(shape[1], f"candidate_shapes[{index}].sy", strict=True)
            rotation = _finite(shape[2], f"candidate_shapes[{index}].rotation")
            normalized.append((sx, sy, rotation))
        object.__setattr__(self, "candidate_shapes", tuple(normalized))
        for name in (
            "max_exchanges",
            "site_count",
            "donor_count",
            "proposal_frontier",
        ):
            object.__setattr__(self, name, _integer(getattr(self, name), name, minimum=1))
        object.__setattr__(
            self,
            "site_nms_radius_px",
            _integer(self.site_nms_radius_px, "site_nms_radius_px"),
        )
        for name in (
            "coefficient_abs_limit",
            "pricing_absolute_tolerance",
            "renderer_parity_tolerance",
            "pixel_rmse_threshold",
            "patch7_rmse_threshold",
        ):
            object.__setattr__(self, name, _finite(getattr(self, name), name, strict=True))
        for name in ("minimum_sse_gain", "local_absolute_tolerance"):
            object.__setattr__(self, name, _finite(getattr(self, name), name))


@dataclass(frozen=True)
class ResidualExchangeCheckpoint:
    """One committed exchange, including its analytical and cold-render audit."""

    accepted_count: int
    row_index: int
    site_x: int
    site_y: int
    scale_x: float
    scale_y: float
    rotation_rad: float
    coefficient_r: float
    coefficient_g: float
    coefficient_b: float
    site_rank: int
    shape_rank: int
    donor_rank: int
    proposal_rank: int
    proposals_tested: int
    predicted_sse_gain: float
    actual_sse_gain: float
    pricing_error_abs: float
    raw_sse: float
    psnr_db: float
    display_pixel_rmse_max: float
    display_patch7_rmse_max: float
    display_gate_pass: bool
    renderer_parity_max_abs: float
    elapsed_seconds: float

    def to_record(self) -> dict[str, object]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class ResidualExchangeResult:
    """Last safe exact-count field and the complete accepted-pivot trajectory."""

    field: ObservationField2D
    reconstruction: np.ndarray
    checkpoints: tuple[ResidualExchangeCheckpoint, ...]
    replaced_row_mask: np.ndarray
    initial_sse: float
    final_sse: float
    initial_pixel_rmse_max: float
    final_pixel_rmse_max: float
    initial_patch7_rmse_max: float
    final_patch7_rmse_max: float
    stop_reason: str
    proposed_pairs: int
    cold_rendered_pairs: int
    maximum_pricing_error_abs: float
    maintained_render_parity_max_abs: float
    repeated_render_parity_max_abs: float
    elapsed_seconds: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "reconstruction", _readonly(self.reconstruction))
        object.__setattr__(self, "replaced_row_mask", _readonly(self.replaced_row_mask))
        if self.replaced_row_mask.dtype != np.bool_ or self.replaced_row_mask.shape != (
            self.field.n,
        ):
            raise ValueError("replaced_row_mask must align with the result field")
        if int(self.replaced_row_mask.sum()) != len(self.checkpoints):
            raise ValueError("every committed exchange must lock exactly one distinct row")

    @property
    def accepted_exchanges(self) -> int:
        return len(self.checkpoints)

    def checkpoint_records(self) -> list[dict[str, object]]:
        return [checkpoint.to_record() for checkpoint in self.checkpoints]


@dataclass(frozen=True)
class _RowSupport:
    pixels: np.ndarray
    weights: np.ndarray
    box: tuple[int, int, int, int]
    diagonal: float


@dataclass(frozen=True)
class _Proposal:
    row_index: int
    donor_rank: int
    site_x: int
    site_y: int
    site_rank: int
    scale_x: float
    scale_y: float
    rotation_rad: float
    shape_rank: int
    coefficient: np.ndarray
    support: _RowSupport
    predicted_sse_gain: float


def _support_for_geometry(
    field: ObservationField2D,
    mask: np.ndarray,
    mean_xy: tuple[float, float],
    scale_xy: tuple[float, float],
    rotation_rad: float,
) -> _RowSupport:
    """Evaluate one exact maintained AABB basis column on active pixels."""

    height, width = field.crop_shape
    mean_x, mean_y = mean_xy
    scale_x, scale_y = scale_xy
    cosine = math.cos(rotation_rad)
    sine = math.sin(rotation_rad)
    variance_x = cosine * cosine * scale_x * scale_x + sine * sine * scale_y * scale_y
    variance_y = sine * sine * scale_x * scale_x + cosine * cosine * scale_y * scale_y
    support = field.semantics.support
    radius_x = max(
        int(math.ceil(support.sigma_cutoff * math.sqrt(variance_x))),
        support.minimum_radius_px,
    )
    radius_y = max(
        int(math.ceil(support.sigma_cutoff * math.sqrt(variance_y))),
        support.minimum_radius_px,
    )
    center_x = int(np.rint(np.float64(mean_x)))
    center_y = int(np.rint(np.float64(mean_y)))
    x0 = max(0, center_x - radius_x)
    x1 = min(width - 1, center_x + radius_x)
    y0 = max(0, center_y - radius_y)
    y1 = min(height - 1, center_y + radius_y)
    if x1 < x0 or y1 < y0:
        return _RowSupport(
            pixels=np.empty(0, dtype=np.int64),
            weights=np.empty(0, dtype=np.float64),
            box=(x0, y0, x1, y1),
            diagonal=0.0,
        )
    yy, xx = np.mgrid[y0 : y1 + 1, x0 : x1 + 1]
    active = mask[y0 : y1 + 1, x0 : x1 + 1]
    pixels = (yy[active] * width + xx[active]).astype(np.int64, copy=False)
    dx = xx[active].astype(np.float64) - float(mean_x)
    dy = yy[active].astype(np.float64) - float(mean_y)
    inv_x = 1.0 / (float(scale_x) * float(scale_x))
    inv_y = 1.0 / (float(scale_y) * float(scale_y))
    conic_a = cosine * cosine * inv_x + sine * sine * inv_y
    conic_b = cosine * sine * (inv_x - inv_y)
    conic_c = sine * sine * inv_x + cosine * cosine * inv_y
    quadratic = conic_a * dx * dx + 2.0 * conic_b * dx * dy + conic_c * dy * dy
    weights = np.exp(-0.5 * quadratic)
    if support.fade_alpha > 0.0:
        tail = support.fade_alpha * math.exp(-0.5 * support.sigma_cutoff**2)
        weights = np.maximum(weights - tail, 0.0)
    weights = weights.astype(np.float64, copy=False)
    return _RowSupport(
        pixels=pixels,
        weights=weights,
        box=(x0, y0, x1, y1),
        diagonal=float(weights @ weights),
    )


def _field_supports(field: ObservationField2D, mask: np.ndarray) -> list[_RowSupport]:
    scales = np.exp(field.log_scales_xy.astype(np.float64))
    return [
        _support_for_geometry(
            field,
            mask,
            (float(mean[0]), float(mean[1])),
            (float(scale[0]), float(scale[1])),
            float(rotation),
        )
        for mean, scale, rotation in zip(
            field.means_xy, scales, field.rotations_rad, strict=True
        )
    ]


def _boxes_disjoint(
    first: tuple[int, int, int, int], second: tuple[int, int, int, int]
) -> bool:
    return bool(
        first[2] < second[0]
        or second[2] < first[0]
        or first[3] < second[1]
        or second[3] < first[1]
    )


def _row_prices(
    supports: list[_RowSupport], coefficients: np.ndarray, residual: np.ndarray
) -> np.ndarray:
    """Return exact SSE increase from deleting every row for ``residual=target-render``."""

    counts = np.fromiter((support.pixels.size for support in supports), dtype=np.int64)
    if int(counts.sum()) == 0:
        return np.zeros(len(supports), dtype=np.float64)
    row_ids = np.repeat(np.arange(len(supports), dtype=np.int64), counts)
    pixels = np.concatenate([support.pixels for support in supports if support.pixels.size])
    weights = np.concatenate([support.weights for support in supports if support.weights.size])
    residual_flat = residual.reshape(-1, 3).astype(np.float64, copy=False)
    inner = np.stack(
        [
            np.bincount(
                row_ids,
                weights=weights * residual_flat[pixels, channel],
                minlength=len(supports),
            )
            for channel in range(3)
        ],
        axis=1,
    )
    diagonal = np.bincount(row_ids, weights=weights * weights, minlength=len(supports))
    colors = coefficients.astype(np.float64, copy=False)
    return diagonal * np.sum(colors * colors, axis=1) + 2.0 * np.sum(colors * inner, axis=1)


def _residual_sites(
    residual: np.ndarray, mask: np.ndarray, count: int, radius: int
) -> list[tuple[int, int]]:
    score = np.sum(residual.astype(np.float64) ** 2, axis=2)
    score[~mask] = -np.inf
    order = np.argsort(-score.reshape(-1), kind="stable")
    height, width = mask.shape
    blocked = np.zeros(mask.shape, dtype=bool)
    sites: list[tuple[int, int]] = []
    for flat_index in order:
        y, x = divmod(int(flat_index), width)
        if not mask[y, x] or blocked[y, x]:
            continue
        sites.append((x, y))
        if len(sites) >= count:
            break
        blocked[
            max(0, y - radius) : min(height, y + radius + 1),
            max(0, x - radius) : min(width, x + radius + 1),
        ] = True
    return sites


def _replace_row(
    field: ObservationField2D,
    row_index: int,
    mean_xy: tuple[float, float],
    scale_xy: tuple[float, float],
    rotation_rad: float,
    coefficient: np.ndarray,
) -> ObservationField2D:
    means = np.array(field.means_xy, copy=True)
    log_scales = np.array(field.log_scales_xy, copy=True)
    rotations = np.array(field.rotations_rad, copy=True)
    coefficients = np.array(field.rgb_coeff, copy=True)
    means[row_index] = mean_xy
    log_scales[row_index] = np.log(np.asarray(scale_xy, dtype=np.float64))
    rotations[row_index] = rotation_rad
    coefficients[row_index] = coefficient
    return replace(
        field,
        means_xy=means,
        log_scales_xy=log_scales,
        rotations_rad=rotations,
        rgb_coeff=coefficients,
    )


def _psnr_from_sse(sse: float, active_values: int) -> float:
    mse = sse / active_values
    return float("inf") if mse == 0.0 else float(-10.0 * math.log10(mse))


def exchange_residual_columns(
    field: ObservationField2D,
    target: np.ndarray,
    mask: np.ndarray,
    *,
    config: ResidualExchangeConfig | None = None,
    device: str = "cpu",
    renderer: str = "additive",
    render_chunk: int = 256,
) -> ResidualExchangeResult:
    """Replace low-value rows with exact residual atoms under a cold-render transaction.

    Each candidate coefficient is the exact one-column least-squares solution on the current
    residual.  Its leaving row must have a disjoint support box, so candidate gain minus deletion
    price is the exact predicted pair gain.  The maintained renderer remains authoritative: a
    proposal commits only if cold SSE strictly improves and both displayed local maxima are
    individually nonworse.
    """

    started = time.perf_counter()
    if not isinstance(field, ObservationField2D):
        raise TypeError("field must be ObservationField2D")
    if field.n < 1:
        raise ValueError("field must contain at least one row")
    if field.semantics.renderer_equation != "additive_rgb_peak_one_v1":
        raise ValueError("column exchange requires direct additive peak-one semantics")
    if field.semantics.support.mode != "axis_aligned_bbox":
        raise ValueError("column exchange requires axis-aligned AABB support")
    if field.background_rgb is not None:
        raise ValueError("column exchange requires a zero-DC field")
    if field.structural_mass is not None or field.filter_variance_px2 is not None:
        raise ValueError("column exchange currently requires appearance-only unfiltered rows")
    if (
        field.semantics.filtering.mode != "none"
        or field.semantics.filtering.aa_dilation_px2 != 0.0
    ):
        raise ValueError("column exchange currently requires filtering.mode='none'")
    if renderer not in ("additive", "cuda_additive", "cuda_tiled_additive"):
        raise ValueError("renderer must be additive, cuda_additive, or cuda_tiled_additive")
    chunk = _integer(render_chunk, "render_chunk", minimum=1)
    cfg = config or ResidualExchangeConfig()
    source = _image(target, field.crop_shape, "target")
    active = _mask(mask, field.crop_shape)
    if field.packed_alpha is not None and not np.array_equal(field.alpha_mask(), active):
        raise ValueError("provided mask must exactly match the field's declared alpha")

    from .pixel_contraction import render_observation_field

    current_field = field
    current = render_observation_field(
        current_field, device=device, renderer=renderer, render_chunk=chunk
    )
    current = np.where(active[:, :, None], current, 0.0).astype(np.float32, copy=False)
    raw_metrics = progressive_artifact_metrics(
        current,
        source,
        active,
        pixel_threshold=cfg.pixel_rmse_threshold,
        patch7_threshold=cfg.patch7_rmse_threshold,
        displayed=False,
    )
    display_metrics = progressive_artifact_metrics(
        current,
        source,
        active,
        pixel_threshold=cfg.pixel_rmse_threshold,
        patch7_threshold=cfg.patch7_rmse_threshold,
        displayed=True,
    )
    initial_sse = float(raw_metrics["sse"])
    initial_pixel = float(display_metrics["pixel_rmse_max"])
    initial_patch = float(display_metrics["patch7_rmse_max"])
    supports = _field_supports(current_field, active)
    locked = np.zeros(field.n, dtype=bool)
    checkpoints: list[ResidualExchangeCheckpoint] = []
    proposed_pairs = 0
    cold_rendered_pairs = 0
    maximum_pricing_error = 0.0
    maximum_parity = 0.0
    stop_reason = "maximum_exchanges"
    active_values = int(active.sum()) * 3

    for accepted_index in range(1, cfg.max_exchanges + 1):
        residual = source.astype(np.float64) - current.astype(np.float64)
        prices = _row_prices(supports, current_field.rgb_coeff, residual)
        eligible = np.flatnonzero(~locked & np.isfinite(prices))
        donor_order = eligible[
            np.argsort(prices[eligible], kind="stable")[: cfg.donor_count]
        ]
        if donor_order.size == 0:
            stop_reason = "no_eligible_donor"
            break
        donor_ranks = {int(row): rank for rank, row in enumerate(donor_order)}
        sites = _residual_sites(
            residual, active, cfg.site_count, cfg.site_nms_radius_px
        )
        proposals: list[_Proposal] = []
        residual_flat = residual.reshape(-1, 3)
        for site_rank, (site_x, site_y) in enumerate(sites):
            for shape_rank, (scale_x, scale_y, rotation) in enumerate(
                cfg.candidate_shapes
            ):
                support = _support_for_geometry(
                    current_field,
                    active,
                    (float(site_x), float(site_y)),
                    (scale_x, scale_y),
                    rotation,
                )
                if support.diagonal <= np.finfo(np.float64).tiny:
                    continue
                inner = support.weights @ residual_flat[support.pixels]
                coefficient = inner / support.diagonal
                if (
                    not np.isfinite(coefficient).all()
                    or float(np.max(np.abs(coefficient))) > cfg.coefficient_abs_limit
                ):
                    continue
                candidate_gain = float(inner @ inner / support.diagonal)
                for row_index in donor_order:
                    row = int(row_index)
                    if not _boxes_disjoint(support.box, supports[row].box):
                        continue
                    predicted_gain = candidate_gain - float(prices[row])
                    if predicted_gain <= cfg.minimum_sse_gain:
                        continue
                    proposals.append(
                        _Proposal(
                            row_index=row,
                            donor_rank=donor_ranks[row],
                            site_x=site_x,
                            site_y=site_y,
                            site_rank=site_rank,
                            scale_x=scale_x,
                            scale_y=scale_y,
                            rotation_rad=rotation,
                            shape_rank=shape_rank,
                            coefficient=coefficient,
                            support=support,
                            predicted_sse_gain=predicted_gain,
                        )
                    )
        proposed_pairs += len(proposals)
        if not proposals:
            stop_reason = "no_improving_pair"
            break
        proposals.sort(
            key=lambda proposal: (
                -proposal.predicted_sse_gain,
                proposal.site_rank,
                proposal.shape_rank,
                proposal.donor_rank,
                proposal.row_index,
            )
        )

        accepted: tuple[
            _Proposal,
            ObservationField2D,
            np.ndarray,
            dict[str, object],
            dict[str, object],
            float,
            float,
        ] | None = None
        for proposal_rank, proposal in enumerate(proposals[: cfg.proposal_frontier]):
            candidate_field = _replace_row(
                current_field,
                proposal.row_index,
                (float(proposal.site_x), float(proposal.site_y)),
                (proposal.scale_x, proposal.scale_y),
                proposal.rotation_rad,
                proposal.coefficient,
            )
            candidate = render_observation_field(
                candidate_field, device=device, renderer=renderer, render_chunk=chunk
            )
            candidate = np.where(active[:, :, None], candidate, 0.0).astype(
                np.float32, copy=False
            )
            cold_rendered_pairs += 1
            candidate_raw = progressive_artifact_metrics(
                candidate,
                source,
                active,
                pixel_threshold=cfg.pixel_rmse_threshold,
                patch7_threshold=cfg.patch7_rmse_threshold,
                displayed=False,
            )
            candidate_display = progressive_artifact_metrics(
                candidate,
                source,
                active,
                pixel_threshold=cfg.pixel_rmse_threshold,
                patch7_threshold=cfg.patch7_rmse_threshold,
                displayed=True,
            )
            actual_gain = float(raw_metrics["sse"]) - float(candidate_raw["sse"])
            pricing_error = abs(proposal.predicted_sse_gain - actual_gain)
            maximum_pricing_error = max(maximum_pricing_error, pricing_error)

            donor = supports[proposal.row_index]
            analytical = current.astype(np.float64, copy=True).reshape(-1, 3)
            analytical[donor.pixels] -= (
                donor.weights[:, None]
                * current_field.rgb_coeff[proposal.row_index].astype(np.float64)[None, :]
            )
            analytical[proposal.support.pixels] += (
                proposal.support.weights[:, None] * proposal.coefficient[None, :]
            )
            analytical = analytical.reshape(current.shape)
            parity = float(np.max(np.abs(analytical - candidate.astype(np.float64))))
            maximum_parity = max(maximum_parity, parity)
            safe = bool(
                actual_gain > cfg.minimum_sse_gain
                and pricing_error <= cfg.pricing_absolute_tolerance
                and parity <= cfg.renderer_parity_tolerance
                and float(candidate_display["pixel_rmse_max"])
                <= float(display_metrics["pixel_rmse_max"]) + cfg.local_absolute_tolerance
                and float(candidate_display["patch7_rmse_max"])
                <= float(display_metrics["patch7_rmse_max"]) + cfg.local_absolute_tolerance
            )
            if safe:
                accepted = (
                    proposal,
                    candidate_field,
                    candidate,
                    candidate_raw,
                    candidate_display,
                    actual_gain,
                    parity,
                )
                accepted_proposal_rank = proposal_rank
                break
        if accepted is None:
            stop_reason = "no_cold_safe_pair"
            break

        (
            proposal,
            current_field,
            current,
            raw_metrics,
            display_metrics,
            actual_gain,
            parity,
        ) = accepted
        supports[proposal.row_index] = proposal.support
        locked[proposal.row_index] = True
        coefficient = proposal.coefficient
        checkpoints.append(
            ResidualExchangeCheckpoint(
                accepted_count=accepted_index,
                row_index=proposal.row_index,
                site_x=proposal.site_x,
                site_y=proposal.site_y,
                scale_x=proposal.scale_x,
                scale_y=proposal.scale_y,
                rotation_rad=proposal.rotation_rad,
                coefficient_r=float(coefficient[0]),
                coefficient_g=float(coefficient[1]),
                coefficient_b=float(coefficient[2]),
                site_rank=proposal.site_rank,
                shape_rank=proposal.shape_rank,
                donor_rank=proposal.donor_rank,
                proposal_rank=accepted_proposal_rank,
                proposals_tested=accepted_proposal_rank + 1,
                predicted_sse_gain=proposal.predicted_sse_gain,
                actual_sse_gain=actual_gain,
                pricing_error_abs=abs(proposal.predicted_sse_gain - actual_gain),
                raw_sse=float(raw_metrics["sse"]),
                psnr_db=_psnr_from_sse(float(raw_metrics["sse"]), active_values),
                display_pixel_rmse_max=float(display_metrics["pixel_rmse_max"]),
                display_patch7_rmse_max=float(display_metrics["patch7_rmse_max"]),
                display_gate_pass=bool(display_metrics["gate_pass"]),
                renderer_parity_max_abs=parity,
                elapsed_seconds=time.perf_counter() - started,
            )
        )

    repeated = render_observation_field(
        current_field, device=device, renderer=renderer, render_chunk=chunk
    )
    repeated = np.where(active[:, :, None], repeated, 0.0).astype(np.float32, copy=False)
    repeated_parity = float(np.max(np.abs(repeated - current)))
    return ResidualExchangeResult(
        field=current_field,
        reconstruction=current,
        checkpoints=tuple(checkpoints),
        replaced_row_mask=locked,
        initial_sse=initial_sse,
        final_sse=float(raw_metrics["sse"]),
        initial_pixel_rmse_max=initial_pixel,
        final_pixel_rmse_max=float(display_metrics["pixel_rmse_max"]),
        initial_patch7_rmse_max=initial_patch,
        final_patch7_rmse_max=float(display_metrics["patch7_rmse_max"]),
        stop_reason=stop_reason,
        proposed_pairs=proposed_pairs,
        cold_rendered_pairs=cold_rendered_pairs,
        maximum_pricing_error_abs=maximum_pricing_error,
        maintained_render_parity_max_abs=maximum_parity,
        repeated_render_parity_max_abs=repeated_parity,
        elapsed_seconds=time.perf_counter() - started,
    )


__all__ = [
    "ResidualExchangeCheckpoint",
    "ResidualExchangeConfig",
    "ResidualExchangeResult",
    "exchange_residual_columns",
]
