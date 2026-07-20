"""Numerical core for the parent-bound BENCH-013 Stage-0 downstream assay.

The mathematical path is a float64, active-row Householder-QR oracle.  The candidate path is the
frozen float32 15-channel ``index_add`` accumulator followed by a centered 2-by-2 Cholesky solve.
Neither path changes production support membership, adds regularization, clips an output, or uses
a rank-adaptive fallback.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Final

import torch

from structsplat.render import _support_weight, _tile_bounds, _tile_coords


SIGMA_CUTOFF: Final[float] = 3.0
CHANNEL_COUNT: Final[int] = 15


@dataclass(frozen=True)
class FrozenContributors:
    """Exact detached production AABB contributor list."""

    gid: torch.Tensor
    px: torch.Tensor
    py: torch.Tensor
    height: int
    width: int
    sample_count: int

    @property
    def flat(self) -> torch.Tensor:
        return self.py * int(self.width) + self.px

    @property
    def active_counts(self) -> torch.Tensor:
        return torch.bincount(self.flat, minlength=int(self.height) * int(self.width))


@dataclass(frozen=True)
class ReferenceResult:
    """Float64 QR output and independent effective-weight diagnostics."""

    output: torch.Tensor
    mass: torch.Tensor
    active_counts: torch.Tensor
    singular_values: torch.Tensor
    rank_thresholds: torch.Tensor
    ranks: torch.Tensor
    covariance_eigenvalues: torch.Tensor
    condition_numbers: torch.Tensor
    partition: torch.Tensor
    first_moment: torch.Tensor
    amplification_l1: torch.Tensor
    effective_weights: torch.Tensor
    finite_pass: torch.Tensor
    contributors: FrozenContributors


@dataclass(frozen=True)
class CandidateResult:
    """Float32 15-channel candidate output and solve diagnostics."""

    output: torch.Tensor
    accumulators: torch.Tensor
    mass: torch.Tensor
    active_counts: torch.Tensor
    mass_pass: torch.Tensor
    contributor_pass: torch.Tensor
    offset_mean: torch.Tensor
    color_mean: torch.Tensor
    centered_covariance: torch.Tensor
    covariance_eigenvalues: torch.Tensor
    condition_numbers: torch.Tensor
    centered_cross_covariance: torch.Tensor
    cholesky: torch.Tensor
    solved_slopes: torch.Tensor
    inverse_mean: torch.Tensor
    cholesky_info: torch.Tensor
    cholesky_info_pass: torch.Tensor
    cholesky_diagonal_pass: torch.Tensor
    finite_pass: torch.Tensor
    solve_pass: torch.Tensor
    partition: torch.Tensor
    first_moment: torch.Tensor
    amplification_l1: torch.Tensor
    contributors: FrozenContributors


def conics_from_parameters(log_scales: torch.Tensor, rotations: torch.Tensor) -> torch.Tensor:
    """Recompute zero-dilation RS conics without detaching differentiable parameters."""

    if log_scales.ndim != 2 or log_scales.shape[1] != 2:
        raise ValueError("log_scales must have shape (N,2)")
    if rotations.shape != (log_scales.shape[0],):
        raise ValueError("rotations must have shape (N,)")
    scales = torch.exp(log_scales)
    inv_x = scales[:, 0].square().reciprocal()
    inv_y = scales[:, 1].square().reciprocal()
    cosine = torch.cos(rotations)
    sine = torch.sin(rotations)
    a = cosine.square() * inv_x + sine.square() * inv_y
    b = cosine * sine * (inv_x - inv_y)
    c = sine.square() * inv_x + cosine.square() * inv_y
    return torch.stack((a, b, c), dim=1)


def enumerate_contributors(
    means: torch.Tensor,
    radii: torch.Tensor,
    height: int,
    width: int,
) -> FrozenContributors:
    """Call the production clipped-AABB enumeration directly, including ``torch.round``."""

    if means.ndim != 2 or means.shape[1] != 2:
        raise ValueError("means must have shape (N,2)")
    if radii.shape != means.shape or radii.dtype not in (
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    ):
        raise ValueError("radii must be an integer tensor with shape (N,2)")
    if means.device.type != "cpu" or radii.device.type != "cpu":
        raise ValueError("BENCH-013 Stage 0 is a deterministic CPU assay")
    height = int(height)
    width = int(width)
    if height <= 0 or width <= 0:
        raise ValueError("height and width must be positive")
    x0, y0, tile_width, tile_sizes = _tile_bounds(means, radii, height, width)
    gid, px, py = _tile_coords(
        x0,
        y0,
        tile_width,
        tile_sizes,
        0,
        int(means.shape[0]),
        means.device,
    )
    return FrozenContributors(
        gid=gid,
        px=px,
        py=py,
        height=height,
        width=width,
        sample_count=int(means.shape[0]),
    )


def _validate_inputs(
    means: torch.Tensor,
    conics: torch.Tensor,
    colors: torch.Tensor,
    radii: torch.Tensor,
    *,
    dtype: torch.dtype,
) -> None:
    count = int(means.shape[0])
    if means.shape != (count, 2) or conics.shape != (count, 3):
        raise ValueError("means/conics have invalid shapes")
    if colors.ndim != 2 or colors.shape[0] != count:
        raise ValueError("colors must have shape (N,C)")
    if radii.shape != (count, 2):
        raise ValueError("radii must have shape (N,2)")
    for name, value in (("means", means), ("conics", conics), ("colors", colors)):
        if value.dtype != dtype or value.device.type != "cpu":
            raise ValueError(f"{name} must be CPU {dtype}")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"{name} must be finite")


def _contribution_weights(
    means: torch.Tensor,
    conics: torch.Tensor,
    contributors: FrozenContributors,
) -> tuple[torch.Tensor, torch.Tensor]:
    gid = contributors.gid
    dx = contributors.px.to(means.dtype) - means[gid, 0]
    dy = contributors.py.to(means.dtype) - means[gid, 1]
    a = conics[gid, 0]
    b = conics[gid, 1]
    c = conics[gid, 2]
    quadratic = a * dx.square() + 2.0 * b * dx * dy + c * dy.square()
    return _support_weight(quadratic, SIGMA_CUTOFF, False), torch.stack((dx, dy), dim=1)


def _active_groups(
    contributors: FrozenContributors,
) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    """Return ``(pixels,gids[pixel,k])`` groups in current row order."""

    flat = contributors.flat
    count = int(contributors.sample_count)
    key = flat * max(count, 1) + contributors.gid
    order = torch.argsort(key, stable=True)
    sorted_gid = contributors.gid[order]
    active = torch.bincount(
        flat, minlength=int(contributors.height) * int(contributors.width)
    )
    starts = torch.cumsum(active, dim=0) - active
    groups: list[tuple[torch.Tensor, torch.Tensor]] = []
    for width in torch.unique(active, sorted=True).tolist():
        width = int(width)
        if width == 0:
            continue
        pixels = torch.nonzero(active == width, as_tuple=False).reshape(-1)
        positions = starts[pixels, None] + torch.arange(width)[None, :]
        groups.append((pixels, sorted_gid[positions]))
    return tuple(groups)


def reference_qr(
    means: torch.Tensor,
    conics: torch.Tensor,
    colors: torch.Tensor,
    radii: torch.Tensor,
    height: int,
    width: int,
    *,
    contributors: FrozenContributors | None = None,
    retain_effective_weights: bool = True,
) -> ReferenceResult:
    """Run the frozen float64 active-row Householder-QR oracle."""

    _validate_inputs(means, conics, colors, radii, dtype=torch.float64)
    if contributors is None:
        contributors = enumerate_contributors(means, radii, height, width)
    pixel_count = int(height) * int(width)
    count = int(means.shape[0])
    channel_count = int(colors.shape[1])
    weights, _ = _contribution_weights(means, conics, contributors)
    dense_weights = torch.zeros((pixel_count, count), dtype=torch.float64)
    dense_weights[contributors.flat, contributors.gid] = weights
    mass = dense_weights.sum(dim=1)
    output = torch.full((pixel_count, channel_count), float("nan"), dtype=torch.float64)
    singular_values = torch.zeros((pixel_count, 3), dtype=torch.float64)
    thresholds = torch.zeros(pixel_count, dtype=torch.float64)
    ranks = torch.zeros(pixel_count, dtype=torch.int64)
    covariance_eigenvalues = torch.zeros((pixel_count, 2), dtype=torch.float64)
    condition = torch.full((pixel_count,), float("inf"), dtype=torch.float64)
    partition = torch.full((pixel_count,), float("nan"), dtype=torch.float64)
    first_moment = torch.full((pixel_count, 2), float("nan"), dtype=torch.float64)
    amplification = torch.full((pixel_count,), float("nan"), dtype=torch.float64)
    effective = torch.zeros((pixel_count, count), dtype=torch.float64)
    coordinate_scale = float(max(int(height) - 1, int(width) - 1, 1))
    epsilon = torch.finfo(torch.float64).eps

    for pixels, gids in _active_groups(contributors):
        probability = dense_weights[pixels[:, None], gids]
        probability = probability / mass[pixels, None]
        pixel_xy = torch.stack(
            ((pixels % int(width)).to(torch.float64), (pixels // int(width)).to(torch.float64)),
            dim=1,
        )
        offsets = (means[gids] - pixel_xy[:, None, :]) / coordinate_scale
        design = torch.cat(
            (torch.ones((*offsets.shape[:2], 1), dtype=torch.float64), offsets), dim=2
        )
        root_probability = torch.sqrt(probability)
        weighted_design = root_probability[:, :, None] * design
        weighted_colors = root_probability[:, :, None] * colors[gids]
        singular = torch.linalg.svdvals(weighted_design)
        if singular.shape[1] < 3:
            singular = torch.cat(
                (
                    singular,
                    torch.zeros(
                        (singular.shape[0], 3 - singular.shape[1]), dtype=torch.float64
                    ),
                ),
                dim=1,
            )
        singular_values[pixels] = singular
        threshold = (
            max(int(gids.shape[1]), 3) * epsilon * singular[:, 0]
        )
        thresholds[pixels] = threshold
        local_ranks = (singular > threshold[:, None]).sum(dim=1)
        ranks[pixels] = local_ranks

        mean = torch.einsum("pk,pkd->pd", probability, offsets)
        centered = offsets - mean[:, None, :]
        covariance = torch.einsum("pk,pki,pkj->pij", probability, centered, centered)
        covariance = 0.5 * (covariance + covariance.transpose(1, 2))
        eigenvalues = torch.linalg.eigvalsh(covariance)
        covariance_eigenvalues[pixels] = eigenvalues
        condition[pixels] = torch.where(
            eigenvalues[:, 0] > 0.0,
            eigenvalues[:, 1] / eigenvalues[:, 0],
            torch.full_like(eigenvalues[:, 0], float("inf")),
        )

        accepted = torch.nonzero(local_ranks == 3, as_tuple=False).reshape(-1)
        if accepted.numel():
            q, r = torch.linalg.qr(weighted_design, mode="reduced")
            q_accepted = q[accepted]
            r_accepted = r[accepted]
            coefficients = torch.linalg.solve_triangular(
                r_accepted,
                q_accepted.transpose(1, 2) @ weighted_colors[accepted],
                upper=True,
            )
            accepted_pixels = pixels[accepted]
            output[accepted_pixels] = coefficients[:, 0, :]
            identity = torch.zeros((accepted.numel(), 3, 1), dtype=torch.float64)
            identity[:, 0, 0] = 1.0
            intermediate = torch.linalg.solve_triangular(
                r_accepted.transpose(1, 2), identity, upper=False
            )
            h = torch.linalg.solve_triangular(
                r_accepted, intermediate, upper=True
            ).squeeze(2)
            phi = probability[accepted] * torch.einsum(
                "pkd,pd->pk", design[accepted], h
            )
            partition[accepted_pixels] = phi.sum(dim=1)
            first_moment[accepted_pixels] = torch.einsum(
                "pk,pkd->pd", phi, offsets[accepted]
            )
            amplification[accepted_pixels] = phi.abs().sum(dim=1)
            if retain_effective_weights:
                effective[accepted_pixels[:, None], gids[accepted]] = phi

    finite = (
        torch.isfinite(output).all(dim=1)
        & torch.isfinite(mass)
        & torch.isfinite(singular_values).all(dim=1)
        & torch.isfinite(covariance_eigenvalues).all(dim=1)
        & torch.isfinite(partition)
        & torch.isfinite(first_moment).all(dim=1)
        & torch.isfinite(amplification)
    )
    return ReferenceResult(
        output=output.reshape(int(height), int(width), channel_count),
        mass=mass,
        active_counts=contributors.active_counts,
        singular_values=singular_values,
        rank_thresholds=thresholds,
        ranks=ranks,
        covariance_eigenvalues=covariance_eigenvalues,
        condition_numbers=condition,
        partition=partition,
        first_moment=first_moment,
        amplification_l1=amplification,
        effective_weights=effective,
        finite_pass=finite,
        contributors=contributors,
    )


def candidate_cholesky(
    means: torch.Tensor,
    conics: torch.Tensor,
    colors: torch.Tensor,
    radii: torch.Tensor,
    height: int,
    width: int,
    *,
    contributors: FrozenContributors | None = None,
) -> CandidateResult:
    """Run the exact frozen float32 15-channel accumulator and centered Cholesky solve."""

    _validate_inputs(means, conics, colors, radii, dtype=torch.float32)
    if colors.shape[1] != 3:
        raise ValueError("the frozen accumulator requires RGB colors")
    if contributors is None:
        contributors = enumerate_contributors(means, radii, height, width)
    weights, pixel_minus_mean = _contribution_weights(means, conics, contributors)
    coordinate_scale = float(max(int(height) - 1, int(width) - 1, 1))
    offsets = -pixel_minus_mean / coordinate_scale
    gid = contributors.gid
    color = colors[gid]
    dx = offsets[:, 0]
    dy = offsets[:, 1]
    contributions = torch.cat(
        (
            weights[:, None],
            weights[:, None] * offsets,
            weights[:, None] * torch.stack((dx.square(), dx * dy, dy.square()), dim=1),
            weights[:, None] * color,
            weights[:, None] * dx[:, None] * color,
            weights[:, None] * dy[:, None] * color,
        ),
        dim=1,
    )
    if contributions.shape[1] != CHANNEL_COUNT:
        raise RuntimeError("internal 15-channel layout mismatch")
    pixel_count = int(height) * int(width)
    accumulators = torch.zeros((pixel_count, CHANNEL_COUNT), dtype=torch.float32)
    accumulators = accumulators.index_add(0, contributors.flat, contributions)
    mass = accumulators[:, 0]
    mean = accumulators[:, 1:3] / mass[:, None]
    second = torch.stack(
        (
            torch.stack((accumulators[:, 3], accumulators[:, 4]), dim=1),
            torch.stack((accumulators[:, 4], accumulators[:, 5]), dim=1),
        ),
        dim=1,
    ) / mass[:, None, None]
    covariance = second - mean[:, :, None] * mean[:, None, :]
    covariance = 0.5 * (covariance + covariance.transpose(1, 2))
    covariance_eigenvalues = torch.linalg.eigvalsh(covariance)
    condition_numbers = torch.where(
        covariance_eigenvalues[:, 0] > 0.0,
        covariance_eigenvalues[:, 1] / covariance_eigenvalues[:, 0],
        torch.full_like(covariance_eigenvalues[:, 0], float("inf")),
    )
    color_mean = accumulators[:, 6:9] / mass[:, None]
    cross = torch.stack((accumulators[:, 9:12], accumulators[:, 12:15]), dim=1)
    cross = cross / mass[:, None, None] - mean[:, :, None] * color_mean[:, None, :]
    cholesky, info = torch.linalg.cholesky_ex(covariance, check_errors=False)
    info_pass = info == 0
    diagonal_pass = (torch.diagonal(cholesky, dim1=1, dim2=2) > 0.0).all(dim=1)

    # There is deliberately no alternate system for a failed factorization.  Solve only the
    # accepted subset and leave all derived values unavailable (NaN) elsewhere so the complete
    # matrix can still be recorded without manufacturing a fallback result.
    accepted = info_pass & diagonal_pass
    accepted_indices = torch.nonzero(accepted, as_tuple=False).reshape(-1)
    slopes = torch.full_like(cross, float("nan"))
    inverse_mean = torch.full_like(mean, float("nan"))
    if accepted_indices.numel():
        solved = torch.cholesky_solve(
            cross[accepted_indices], cholesky[accepted_indices]
        )
        solved_mean = torch.cholesky_solve(
            mean[accepted_indices, :, None], cholesky[accepted_indices]
        ).squeeze(2)
        slopes[accepted_indices] = solved
        inverse_mean[accepted_indices] = solved_mean
    output = color_mean - torch.einsum("pi,pic->pc", mean, slopes)

    probability_element = weights / mass[contributors.flat]
    centered_element = offsets - mean[contributors.flat]
    phi = probability_element * (
        1.0 - torch.einsum("ed,ed->e", centered_element, inverse_mean[contributors.flat])
    )
    partition = torch.zeros(pixel_count, dtype=torch.float32).index_add(
        0, contributors.flat, phi
    )
    first_moment = torch.zeros((pixel_count, 2), dtype=torch.float32).index_add(
        0, contributors.flat, phi[:, None] * offsets
    )
    amplification = torch.zeros(pixel_count, dtype=torch.float32).index_add(
        0, contributors.flat, phi.abs()
    )
    finite = (
        torch.isfinite(accumulators).all(dim=1)
        & torch.isfinite(mean).all(dim=1)
        & torch.isfinite(covariance).all(dim=(1, 2))
        & torch.isfinite(color_mean).all(dim=1)
        & torch.isfinite(cross).all(dim=(1, 2))
        & torch.isfinite(cholesky).all(dim=(1, 2))
        & torch.isfinite(slopes).all(dim=(1, 2))
        & torch.isfinite(inverse_mean).all(dim=1)
        & torch.isfinite(output).all(dim=1)
        & torch.isfinite(partition)
        & torch.isfinite(first_moment).all(dim=1)
        & torch.isfinite(amplification)
    )
    solve_pass = info_pass & diagonal_pass & finite
    unavailable = torch.full_like(output, float("nan"))
    output = torch.where(solve_pass[:, None], output, unavailable)
    partition = torch.where(
        solve_pass, partition, torch.full_like(partition, float("nan"))
    )
    first_moment = torch.where(
        solve_pass[:, None],
        first_moment,
        torch.full_like(first_moment, float("nan")),
    )
    amplification = torch.where(
        solve_pass, amplification, torch.full_like(amplification, float("nan"))
    )
    return CandidateResult(
        output=output.reshape(int(height), int(width), 3),
        accumulators=accumulators,
        mass=mass,
        active_counts=contributors.active_counts,
        mass_pass=mass >= 1e-5,
        contributor_pass=contributors.active_counts >= 3,
        offset_mean=mean,
        color_mean=color_mean,
        centered_covariance=covariance,
        covariance_eigenvalues=covariance_eigenvalues,
        condition_numbers=condition_numbers,
        centered_cross_covariance=cross,
        cholesky=cholesky,
        solved_slopes=slopes,
        inverse_mean=inverse_mean,
        cholesky_info=info,
        cholesky_info_pass=info_pass,
        cholesky_diagonal_pass=diagonal_pass,
        finite_pass=finite,
        solve_pass=solve_pass,
        partition=partition,
        first_moment=first_moment,
        amplification_l1=amplification,
        contributors=contributors,
    )


def reference_render_frozen(
    means: torch.Tensor,
    log_scales: torch.Tensor,
    rotations: torch.Tensor,
    colors: torch.Tensor,
    radii: torch.Tensor,
    contributors: FrozenContributors,
) -> torch.Tensor:
    """Differentiable float64 QR render with a frozen contributor list."""

    return reference_qr(
        means,
        conics_from_parameters(log_scales, rotations),
        colors,
        radii,
        contributors.height,
        contributors.width,
        contributors=contributors,
        retain_effective_weights=False,
    ).output


def candidate_render_frozen(
    means: torch.Tensor,
    log_scales: torch.Tensor,
    rotations: torch.Tensor,
    colors: torch.Tensor,
    radii: torch.Tensor,
    contributors: FrozenContributors,
) -> torch.Tensor:
    """Differentiable float32 candidate render with a frozen contributor list."""

    return candidate_cholesky(
        means,
        conics_from_parameters(log_scales, rotations),
        colors,
        radii,
        contributors.height,
        contributors.width,
        contributors=contributors,
    ).output


def range_excursion(output: torch.Tensor, sample_colors: torch.Tensor) -> float:
    """Maximum unclipped output excursion beyond per-channel sample extrema."""

    if output.ndim != 3 or sample_colors.ndim != 2:
        raise ValueError("output/sample_colors have invalid shapes")
    low = sample_colors.min(dim=0).values
    high = sample_colors.max(dim=0).values
    below = torch.clamp(low[None, None, :] - output, min=0.0)
    above = torch.clamp(output - high[None, None, :], min=0.0)
    return float(torch.maximum(below.max(), above.max()))


def quantile_linear(values: torch.Tensor, quantile: float) -> float:
    """Frozen NumPy linear quantile, deliberately not ``torch.quantile``."""

    if not math.isfinite(quantile) or not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be within [0,1]")
    import numpy as np

    return float(np.quantile(values.detach().cpu().numpy(), quantile, method="linear"))


__all__ = [
    "CHANNEL_COUNT",
    "CandidateResult",
    "FrozenContributors",
    "ReferenceResult",
    "candidate_cholesky",
    "candidate_render_frozen",
    "conics_from_parameters",
    "enumerate_contributors",
    "quantile_linear",
    "range_excursion",
    "reference_qr",
    "reference_render_frozen",
]
