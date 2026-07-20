"""Benchmark-only local-linear reproducing compositor core.

This module deliberately does not alter :mod:`structsplat.render`.  It reuses that renderer's
exact rectangular support convention and Gaussian/opacity weights, then replaces the normalized
constant fit at each pixel by the intercept of a weighted affine fit.  The scientific question is
whether this simple moving-least-squares primitive is numerically coherent before any renderer,
training, codec, or production experiment is authorized.

The solve is intentionally narrow: reduced QR followed by a triangular solve.  There is no ridge,
pseudoinverse, clipping, fallback, ghost sample, or support modification.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Final

import torch

from structsplat.render import _support_weight, _tile_bounds, _tile_coords


NORMALIZED_EPS: Final[float] = 1e-8
SIGMA_CUTOFF: Final[float] = 3.0


@dataclass(frozen=True)
class GaussianGeometry:
    """Physical Gaussian geometry needed by the analytic compositor.

    ``means`` are ``(x, y)`` pixel coordinates, ``conics`` store ``(a, b, c)`` for
    ``a*dx^2 + 2*b*dx*dy + c*dy^2``, and ``radii`` are the current renderer's integer AABB
    half-widths.  ``opacities`` are already-decoded multiplicative values, not logits.
    """

    means: torch.Tensor
    conics: torch.Tensor
    radii: torch.Tensor
    opacities: torch.Tensor | None = None

    def validated(self) -> "GaussianGeometry":
        if self.means.ndim != 2 or self.means.shape[1] != 2:
            raise ValueError("means must have shape (N, 2)")
        count = int(self.means.shape[0])
        if self.conics.shape != (count, 3):
            raise ValueError("conics must have shape (N, 3)")
        if self.radii.shape != (count, 2):
            raise ValueError("radii must have shape (N, 2)")
        if self.opacities is not None and self.opacities.shape != (count,):
            raise ValueError("opacities must have shape (N,)")
        if self.means.dtype not in (torch.float32, torch.float64):
            raise ValueError("means must be float32 or float64")
        if self.conics.dtype != self.means.dtype:
            raise ValueError("conics must have the same dtype as means")
        if self.radii.dtype not in (
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        ):
            raise ValueError("radii must have an integer dtype")
        if self.opacities is not None and self.opacities.dtype != self.means.dtype:
            raise ValueError("opacities must have the same dtype as means")
        tensors = (self.means, self.conics, self.radii)
        if self.opacities is not None:
            tensors += (self.opacities,)
        if any(value.device.type != "cpu" for value in tensors):
            raise ValueError("BENCH-013 Stage 0 is a deterministic CPU assay")
        if not bool(torch.isfinite(self.means).all()):
            raise ValueError("means must be finite")
        if not bool(torch.isfinite(self.conics).all()):
            raise ValueError("conics must be finite")
        if bool((self.radii < 1).any()):
            raise ValueError("radii must be positive")
        if self.opacities is not None:
            if not bool(torch.isfinite(self.opacities).all()):
                raise ValueError("opacities must be finite")
            if bool((self.opacities < 0.0).any()):
                raise ValueError("opacities must be non-negative")
        return self


@dataclass(frozen=True)
class SupportWeights:
    """Dense Stage-0 view of the current renderer's sparse rectangular support."""

    weights: torch.Tensor
    support: torch.Tensor
    pixel_xy: torch.Tensor
    active_counts: torch.Tensor


@dataclass(frozen=True)
class GeometryDiagnostics:
    """Frozen per-pixel early-screen diagnostics for one persisted field."""

    mass: torch.Tensor
    active_counts: torch.Tensor
    design_singular_values: torch.Tensor
    rank_thresholds: torch.Tensor
    ranks: torch.Tensor
    covariance_eigenvalues: torch.Tensor
    condition_numbers: torch.Tensor
    mass_pass: torch.Tensor
    contributor_pass: torch.Tensor
    rank_pass: torch.Tensor
    condition_pass: torch.Tensor
    early_pass: torch.Tensor
    support_weights: SupportWeights


@dataclass(frozen=True)
class LocalLinearOperator:
    """Per-pixel weighted-affine coefficient operator and numerical diagnostics."""

    coefficient_weights: torch.Tensor
    support_weights: SupportWeights
    ranks: torch.Tensor
    conditions: torch.Tensor
    rank_thresholds: torch.Tensor
    height: int
    width: int
    coordinate_scale: float

    @property
    def dtype(self) -> torch.dtype:
        return self.coefficient_weights.dtype

    @property
    def sample_count(self) -> int:
        return int(self.coefficient_weights.shape[2])

    @property
    def effective_weights(self) -> torch.Tensor:
        """Linear weights mapping sample values to the fitted intercept."""

        return self.coefficient_weights[:, 0, :]

    def coefficients(self, sample_values: torch.Tensor) -> torch.Tensor:
        """Return ``(H, W, 3, C)`` intercept/x-slope/y-slope coefficients."""

        values = _validate_sample_values(sample_values, self.sample_count, self.dtype)
        coefficients = torch.einsum("pkn,nc->pkc", self.coefficient_weights, values)
        return coefficients.reshape(self.height, self.width, 3, values.shape[1])

    def render(self, sample_values: torch.Tensor) -> torch.Tensor:
        """Return the unclipped fitted intercept as ``(H, W, C)``."""

        return self.coefficients(sample_values)[:, :, 0, :]

    def normalized_constant_render(
        self,
        sample_values: torch.Tensor,
        *,
        epsilon: float = NORMALIZED_EPS,
    ) -> torch.Tensor:
        """Current normalized constant-fit equation using the identical support weights."""

        if not math.isfinite(epsilon) or epsilon <= 0.0:
            raise ValueError("epsilon must be finite and positive")
        values = _validate_sample_values(sample_values, self.sample_count, self.dtype)
        weights = self.support_weights.weights
        numerator = weights @ values
        denominator = weights.sum(dim=1, keepdim=True) + float(epsilon)
        return (numerator / denominator).reshape(
            self.height, self.width, values.shape[1]
        )


class RankDeficientPixelError(RuntimeError):
    """Raised before solving when any scored pixel lacks a full affine design."""

    def __init__(
        self,
        ranks: torch.Tensor,
        conditions: torch.Tensor,
        active_counts: torch.Tensor,
        width: int,
    ) -> None:
        deficient = torch.nonzero(ranks < 3, as_tuple=False).reshape(-1)
        first = int(deficient[0]) if deficient.numel() else -1
        x = -1 if first < 0 else first % int(width)
        y = -1 if first < 0 else first // int(width)
        self.ranks = ranks
        self.conditions = conditions
        self.active_counts = active_counts
        self.deficient_indices = deficient
        super().__init__(
            f"{int(deficient.numel())} scored pixels are rank deficient; "
            f"first pixel is (x={x}, y={y})"
        )


def _validate_dimensions(height: int, width: int) -> tuple[int, int]:
    height = int(height)
    width = int(width)
    if height <= 0 or width <= 0:
        raise ValueError("height and width must be positive")
    return height, width


def _validate_sample_values(
    sample_values: torch.Tensor,
    sample_count: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    if sample_values.ndim == 1:
        sample_values = sample_values[:, None]
    if sample_values.ndim != 2 or sample_values.shape[0] != sample_count:
        raise ValueError("sample_values must have shape (N,) or (N, C)")
    if sample_values.device.type != "cpu" or sample_values.dtype != dtype:
        raise ValueError("sample_values must match the operator's CPU dtype")
    if not bool(torch.isfinite(sample_values).all()):
        raise ValueError("sample_values must be finite")
    return sample_values


def renderer_support_weights(
    geometry: GaussianGeometry,
    height: int,
    width: int,
) -> SupportWeights:
    """Evaluate exactly the current normalized compositing support and raw weights.

    The current renderer uses the clipped integer AABB of each Gaussian's cutoff ellipse.  It
    evaluates the Gaussian everywhere in that rectangle; it does not add a second ellipse mask.
    This dense formulation is Stage-0-only and mirrors those semantics directly.
    """

    geometry = geometry.validated()
    height, width = _validate_dimensions(height, width)
    dtype = geometry.means.dtype
    yy, xx = torch.meshgrid(
        torch.arange(height, dtype=dtype),
        torch.arange(width, dtype=dtype),
        indexing="ij",
    )
    pixel_xy = torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=1)
    if geometry.opacities is not None:
        raise ValueError("BENCH-013 Stage 0 freezes opacity=None")
    x0, y0, tile_width, tile_sizes = _tile_bounds(
        geometry.means,
        geometry.radii,
        height,
        width,
    )
    gid, px, py = _tile_coords(
        x0,
        y0,
        tile_width,
        tile_sizes,
        0,
        int(geometry.means.shape[0]),
        geometry.means.device,
    )
    dx = px.to(dtype) - geometry.means[gid, 0]
    dy = py.to(dtype) - geometry.means[gid, 1]
    a = geometry.conics[gid, 0]
    b = geometry.conics[gid, 1]
    c = geometry.conics[gid, 2]
    quadratic = a * dx.square() + 2.0 * b * dx * dy + c * dy.square()
    contribution = _support_weight(
        quadratic,
        sigma_cutoff=SIGMA_CUTOFF,
        support_fade=False,
    )
    flat = py * width + px
    support = torch.zeros(
        (height * width, int(geometry.means.shape[0])),
        dtype=torch.bool,
    )
    weights = torch.zeros(
        (height * width, int(geometry.means.shape[0])),
        dtype=dtype,
    )
    support[flat, gid] = True
    weights[flat, gid] = contribution
    return SupportWeights(
        weights=weights,
        support=support,
        pixel_xy=pixel_xy,
        active_counts=support.sum(dim=1),
    )


def geometry_diagnostics(
    geometry: GaussianGeometry,
    height: int,
    width: int,
    *,
    minimum_mass: float = 1e-5,
    maximum_condition: float = 8192.0,
) -> GeometryDiagnostics:
    """Compute the complete vectorized BENCH-013 early conditioning screen.

    The geometry must be the mathematical float64 promotion of a persisted production float32
    field.  Contributor counts come from the exact ``_tile_coords`` enumeration even if a weight
    underflows; weights, normalized design, and covariance use their numerical float64 values.
    """

    geometry = geometry.validated()
    if geometry.means.dtype != torch.float64:
        raise ValueError("early-screen rank/conditioning diagnostics require float64 geometry")
    if not math.isfinite(minimum_mass) or minimum_mass <= 0.0:
        raise ValueError("minimum_mass must be finite and positive")
    if not math.isfinite(maximum_condition) or maximum_condition <= 0.0:
        raise ValueError("maximum_condition must be finite and positive")
    support_weights = renderer_support_weights(geometry, height, width)
    weights = support_weights.weights
    mass = weights.sum(dim=1)
    safe_mass = torch.where(mass > 0.0, mass, torch.ones_like(mass))
    probability = weights / safe_mass[:, None]
    coordinate_scale = float(max(int(height) - 1, int(width) - 1, 1))
    offsets = (
        geometry.means[None, :, :] - support_weights.pixel_xy[:, None, :]
    ) / coordinate_scale
    design = torch.cat(
        (
            torch.ones(
                (height * width, geometry.means.shape[0], 1),
                dtype=torch.float64,
            ),
            offsets,
        ),
        dim=2,
    )
    weighted_design = torch.sqrt(probability)[:, :, None] * design
    singular_values = torch.linalg.svdvals(weighted_design)
    if singular_values.shape[1] < 3:
        singular_values = torch.cat(
            (
                singular_values,
                torch.zeros(
                    (height * width, 3 - singular_values.shape[1]),
                    dtype=torch.float64,
                ),
            ),
            dim=1,
        )
    thresholds = (
        support_weights.active_counts.clamp_min(3).to(torch.float64)
        * torch.finfo(torch.float64).eps
        * singular_values[:, 0]
    )
    ranks = (singular_values > thresholds[:, None]).sum(dim=1)
    mean = torch.einsum("pn,pnd->pd", probability, offsets)
    centered = offsets - mean[:, None, :]
    covariance = torch.einsum(
        "pn,pni,pnj->pij",
        probability,
        centered,
        centered,
    )
    covariance = 0.5 * (covariance + covariance.transpose(1, 2))
    covariance_eigenvalues = torch.linalg.eigvalsh(covariance)
    smallest = covariance_eigenvalues[:, 0]
    largest = covariance_eigenvalues[:, 1]
    condition = torch.where(
        smallest > 0.0,
        largest / smallest,
        torch.full_like(smallest, float("inf")),
    )
    mass_pass = mass >= float(minimum_mass)
    contributor_pass = support_weights.active_counts >= 3
    rank_pass = ranks == 3
    condition_pass = torch.isfinite(condition) & (condition <= float(maximum_condition))
    early_pass = mass_pass & contributor_pass & rank_pass & condition_pass
    return GeometryDiagnostics(
        mass=mass,
        active_counts=support_weights.active_counts,
        design_singular_values=singular_values,
        rank_thresholds=thresholds,
        ranks=ranks,
        covariance_eigenvalues=covariance_eigenvalues,
        condition_numbers=condition,
        mass_pass=mass_pass,
        contributor_pass=contributor_pass,
        rank_pass=rank_pass,
        condition_pass=condition_pass,
        early_pass=early_pass,
        support_weights=support_weights,
    )


def build_local_linear_operator(
    geometry: GaussianGeometry,
    height: int,
    width: int,
) -> LocalLinearOperator:
    """Build the full-rank weighted-affine QR operator, failing before any fallback.

    This is the small-control mathematical QR oracle/prototype, not the frozen float32
    implementation described by BENCH-013.  It deliberately accepts only float64 inputs.  SVD is
    used only to report rank and condition; coefficients are obtained solely by reduced QR and a
    triangular solve.
    """

    geometry = geometry.validated()
    if geometry.means.dtype != torch.float64:
        raise ValueError("the local-linear QR oracle requires float64 geometry")
    height, width = _validate_dimensions(height, width)
    support_weights = renderer_support_weights(geometry, height, width)
    pixel_count, sample_count = support_weights.weights.shape
    dtype = geometry.means.dtype
    coefficients = torch.zeros((pixel_count, 3, sample_count), dtype=dtype)
    ranks = torch.zeros(pixel_count, dtype=torch.int64)
    conditions = torch.full((pixel_count,), float("inf"), dtype=dtype)
    thresholds = torch.zeros(pixel_count, dtype=dtype)
    designs: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    epsilon = torch.finfo(dtype).eps

    for pixel_index in range(pixel_count):
        active = torch.nonzero(
            support_weights.weights[pixel_index] > 0.0,
            as_tuple=False,
        ).reshape(-1)
        weight = support_weights.weights[pixel_index, active]
        coordinate_scale = float(max(height - 1, width - 1, 1))
        offsets = (
            geometry.means[active] - support_weights.pixel_xy[pixel_index]
        ) / coordinate_scale
        design = torch.cat(
            (torch.ones((active.numel(), 1), dtype=dtype), offsets),
            dim=1,
        )
        weighted_design = torch.sqrt(weight)[:, None] * design
        singular = torch.linalg.svdvals(weighted_design)
        maximum = float(singular[0]) if singular.numel() else 0.0
        threshold = float(epsilon) * max(int(active.numel()), 3) * maximum
        rank = int((singular > threshold).sum())
        ranks[pixel_index] = rank
        thresholds[pixel_index] = threshold
        if rank == 3:
            conditions[pixel_index] = singular[0] / singular[-1]
        designs.append((active, weight, weighted_design))

    if bool((ranks < 3).any()):
        raise RankDeficientPixelError(
            ranks,
            conditions,
            support_weights.active_counts,
            width,
        )

    for pixel_index, (active, weight, weighted_design) in enumerate(designs):
        q, r = torch.linalg.qr(weighted_design, mode="reduced")
        right = q.T * torch.sqrt(weight)[None, :]
        local_operator = torch.linalg.solve_triangular(r, right, upper=True)
        if not bool(torch.isfinite(local_operator).all()):
            raise RuntimeError("local-linear QR solve produced non-finite coefficients")
        coefficients[pixel_index, :, active] = local_operator

    return LocalLinearOperator(
        coefficient_weights=coefficients,
        support_weights=support_weights,
        ranks=ranks,
        conditions=conditions,
        rank_thresholds=thresholds,
        height=height,
        width=width,
        coordinate_scale=float(max(height - 1, width - 1, 1)),
    )


def effective_weight_l1(operator: LocalLinearOperator) -> torch.Tensor:
    """Per-pixel L1 amplification of the fitted-intercept weights."""

    return operator.effective_weights.abs().sum(dim=1)


def range_overshoot(image: torch.Tensor, low: float = 0.0, high: float = 1.0) -> float:
    """Maximum unclipped excursion outside a frozen scalar range."""

    if not math.isfinite(low) or not math.isfinite(high) or low >= high:
        raise ValueError("low/high must be finite and ordered")
    below = torch.clamp(float(low) - image, min=0.0)
    above = torch.clamp(image - float(high), min=0.0)
    return float(torch.maximum(below.max(), above.max()))


__all__ = [
    "GaussianGeometry",
    "GeometryDiagnostics",
    "LocalLinearOperator",
    "NORMALIZED_EPS",
    "RankDeficientPixelError",
    "SupportWeights",
    "build_local_linear_operator",
    "effective_weight_l1",
    "geometry_diagnostics",
    "range_overshoot",
    "renderer_support_weights",
]
