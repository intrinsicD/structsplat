"""Synthetic-only bridge from StructSplat's normalized renderer to ``Y = A C``.

The bridge snapshots explicit float32 CPU geometry, reproduces the reference renderer's detached
rounded-center support rectangles and integer pixel coordinates, and constructs normalized sparse
color columns.  It is benchmark-local and is not a codec, lifecycle runner, or improvement claim.
Only constant per-Gaussian RGB is represented; affine color gradients are intentionally outside
this proof.

CPU forward and color-adjoint parity are gating mechanics checks.  Current CUDA comparison is
reported separately and is explicitly nongating because extension/toolchain availability and
parallel accumulation order are environment dependent.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
import torch

from benchmarks.renderer_gram_aq_proof import (
    ClampDiagnostics,
    RendererLinearMap,
    clamp_diagnostics,
)
from structsplat.render import (
    _EPS,
    _resolve_support_fade_alpha,
    _support_weight,
    _tile_bounds,
    _tile_coords,
    render_field,
)


FloatArray = NDArray[np.float64]

# Frozen after 768 deterministic synthetic float32 cases spanning N=4..32, H=7..32, W=8..32,
# clipped/off-image support, opacities, no/full/partial fade, and arbitrary RGB/cotangents.  The
# observed maxima were 4.216825e-7 forward and 1.822311e-5 transpose.  These margins cover CPU
# accumulation-order variation while remaining far below a codec quality tolerance.
CPU_FORWARD_ATOL = 7.5e-7
CPU_FORWARD_RTOL = 2.0e-6
CPU_TRANSPOSE_ATOL = 2.5e-5
CPU_TRANSPOSE_RTOL = 2.0e-6
CUDA_REPORT_ATOL = 2.0e-5
CUDA_REPORT_RTOL = 2.0e-5
DEFAULT_RENDER_EPS = float(_EPS)


def _float32_cpu_tensor(
    value: ArrayLike, *, name: str, shape_tail: tuple[int, ...]
) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float32, device="cpu").detach().clone()
    if tensor.ndim != len(shape_tail) + 1 or tuple(tensor.shape[1:]) != shape_tail:
        suffix = ", ".join(str(item) for item in shape_tail)
        raise ValueError(f"{name} must have shape (N, {suffix})")
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{name} must be finite")
    return tensor


def _radii_cpu_tensor(value: ArrayLike, *, rows: int) -> torch.Tensor:
    raw = torch.as_tensor(value, device="cpu")
    if raw.ndim != 2 or tuple(raw.shape) != (int(rows), 2):
        raise ValueError("radii must have shape (N, 2)")
    if raw.dtype.is_floating_point:
        if not bool(torch.isfinite(raw).all()) or not bool(torch.equal(raw, torch.round(raw))):
            raise ValueError("floating radii must be finite integer values")
    radii = raw.to(dtype=torch.long).detach().clone()
    if bool((radii < 0).any()):
        raise ValueError("radii must be nonnegative")
    return radii


@dataclass(frozen=True)
class ParityReport:
    max_absolute_error: float
    max_relative_error: float
    root_mean_square_error: float
    atol: float
    rtol: float
    passed: bool
    value_count: int


def _parity_report(
    candidate: ArrayLike,
    reference: ArrayLike,
    *,
    atol: float,
    rtol: float,
) -> ParityReport:
    actual = np.asarray(candidate, dtype=np.float64)
    expected = np.asarray(reference, dtype=np.float64)
    if actual.shape != expected.shape:
        raise ValueError("parity arrays differ in shape")
    if actual.size == 0:
        raise ValueError("parity arrays must be non-empty")
    if not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise ValueError("parity arrays must be finite")
    difference = actual - expected
    relative = np.abs(difference) / np.maximum(np.abs(expected), 1e-8)
    return ParityReport(
        max_absolute_error=float(np.max(np.abs(difference))),
        max_relative_error=float(np.max(relative)),
        root_mean_square_error=float(np.sqrt(np.mean(difference * difference))),
        atol=float(atol),
        rtol=float(rtol),
        passed=bool(np.allclose(actual, expected, atol=float(atol), rtol=float(rtol))),
        value_count=int(actual.size),
    )


@dataclass(frozen=True)
class NormalizedRendererBridge:
    """Frozen synthetic geometry and its normalized sparse color operator."""

    linear_map: RendererLinearMap
    means: torch.Tensor
    conics: torch.Tensor
    radii: torch.Tensor
    opacities: torch.Tensor | None
    height: int
    width: int
    eps: float
    sigma_cutoff: float
    support_fade: bool
    support_fade_alpha: float
    denominator: FloatArray
    raw_support_elements: int
    nonzero_elements: int

    def render_colors(self, colors: ArrayLike) -> FloatArray:
        values = np.asarray(colors, dtype=np.float64)
        if values.ndim != 2 or int(values.shape[0]) != self.linear_map.column_count:
            raise ValueError("colors must have one row per Gaussian")
        if not np.isfinite(values).all():
            raise ValueError("colors must be finite")
        return self.linear_map.apply(values).reshape(self.height, self.width, int(values.shape[1]))

    def clamp_accounting(
        self,
        colors: ArrayLike,
        *,
        lower: float = 0.0,
        upper: float = 1.0,
    ) -> ClampDiagnostics:
        return clamp_diagnostics(self.render_colors(colors), lower=lower, upper=upper)


def build_normalized_renderer_bridge(
    means: ArrayLike,
    conics: ArrayLike,
    radii: ArrayLike,
    opacities: ArrayLike | None,
    height: int,
    width: int,
    *,
    eps: float = DEFAULT_RENDER_EPS,
    support_fade: bool = False,
    sigma_cutoff: float = 3.0,
    support_fade_alpha: float | None = None,
) -> NormalizedRendererBridge:
    """Construct normalized ``A`` using reference support and pixel-coordinate semantics."""

    if int(height) <= 0 or int(width) <= 0:
        raise ValueError("height and width must be positive")
    if not math.isfinite(eps) or float(eps) <= 0.0:
        raise ValueError("eps must be finite and positive")
    if not math.isfinite(sigma_cutoff) or float(sigma_cutoff) <= 0.0:
        raise ValueError("sigma_cutoff must be finite and positive")
    means_tensor = _float32_cpu_tensor(means, name="means", shape_tail=(2,))
    conics_tensor = _float32_cpu_tensor(conics, name="conics", shape_tail=(3,))
    if int(means_tensor.shape[0]) == 0:
        raise ValueError("the synthetic bridge requires at least one Gaussian")
    if int(conics_tensor.shape[0]) != int(means_tensor.shape[0]):
        raise ValueError("means and conics must have the same row count")
    radii_tensor = _radii_cpu_tensor(radii, rows=int(means_tensor.shape[0]))
    opacity_tensor = None
    if opacities is not None:
        opacity_tensor = (
            torch.as_tensor(
                opacities,
                dtype=torch.float32,
                device="cpu",
            )
            .reshape(-1)
            .detach()
            .clone()
        )
        if int(opacity_tensor.numel()) != int(means_tensor.shape[0]):
            raise ValueError("opacities must have one value per Gaussian")
        if not bool(torch.isfinite(opacity_tensor).all()):
            raise ValueError("opacities must be finite")
    fade_alpha = _resolve_support_fade_alpha(support_fade, support_fade_alpha)

    with torch.no_grad():
        x0, y0, tile_width, counts = _tile_bounds(
            means_tensor,
            radii_tensor,
            int(height),
            int(width),
        )
        gaussian_ids, pixel_x, pixel_y = _tile_coords(
            x0,
            y0,
            tile_width,
            counts,
            0,
            int(means_tensor.shape[0]),
            means_tensor.device,
        )
        dx = pixel_x.to(torch.float32) - means_tensor[gaussian_ids, 0]
        dy = pixel_y.to(torch.float32) - means_tensor[gaussian_ids, 1]
        selected_conics = conics_tensor[gaussian_ids]
        quadratic = (
            selected_conics[:, 0] * dx * dx
            + 2.0 * selected_conics[:, 1] * dx * dy
            + selected_conics[:, 2] * dy * dy
        )
        raw_weights = _support_weight(
            quadratic,
            float(sigma_cutoff),
            support_fade,
            fade_alpha,
        )
        if opacity_tensor is not None:
            raw_weights = raw_weights * opacity_tensor[gaussian_ids]
        flat_pixels = pixel_y * int(width) + pixel_x
        denominator = torch.zeros(int(height) * int(width), dtype=torch.float32)
        denominator.index_add_(0, flat_pixels, raw_weights)
        normalized_weights = raw_weights / (denominator[flat_pixels] + float(eps))

        columns: list[tuple[NDArray[np.int64], FloatArray]] = []
        offset = 0
        nonzero_elements = 0
        for count_tensor in counts:
            count = int(count_tensor)
            column_pixels = flat_pixels[offset : offset + count]
            column_weights = normalized_weights[offset : offset + count]
            keep = column_weights != 0.0
            kept_pixels = column_pixels[keep].cpu().numpy().astype(np.int64, copy=True)
            kept_weights = column_weights[keep].cpu().numpy().astype(np.float64, copy=True)
            columns.append((kept_pixels, kept_weights))
            nonzero_elements += int(keep.sum())
            offset += count

    linear_map = RendererLinearMap.from_column_footprints(
        int(height) * int(width),
        columns,
    )
    return NormalizedRendererBridge(
        linear_map=linear_map,
        means=means_tensor,
        conics=conics_tensor,
        radii=radii_tensor,
        opacities=opacity_tensor,
        height=int(height),
        width=int(width),
        eps=float(eps),
        sigma_cutoff=float(sigma_cutoff),
        support_fade=bool(support_fade),
        support_fade_alpha=float(fade_alpha),
        denominator=denominator.cpu().numpy().astype(np.float64, copy=True),
        raw_support_elements=int(counts.sum()),
        nonzero_elements=nonzero_elements,
    )


def _colors_tensor(bridge: NormalizedRendererBridge, colors: ArrayLike) -> torch.Tensor:
    values = torch.as_tensor(colors, dtype=torch.float32, device="cpu").detach().clone()
    if values.shape != (bridge.linear_map.column_count, 3):
        raise ValueError("StructSplat parity colors must have shape (N, 3)")
    if not bool(torch.isfinite(values).all()):
        raise ValueError("colors must be finite")
    return values


def validate_cpu_forward(
    bridge: NormalizedRendererBridge,
    colors: ArrayLike,
    *,
    atol: float = CPU_FORWARD_ATOL,
    rtol: float = CPU_FORWARD_RTOL,
) -> ParityReport:
    """Gate the sparse map against the repository CPU normalized renderer."""

    if bridge.eps != DEFAULT_RENDER_EPS:
        raise ValueError("repository CPU parity requires its fixed eps=1e-8")
    colors_tensor = _colors_tensor(bridge, colors)
    with torch.no_grad():
        reference = render_field(
            bridge.means,
            bridge.conics,
            colors_tensor,
            bridge.radii,
            bridge.height,
            bridge.width,
            mode="normalized",
            opacities=bridge.opacities,
            support_fade=bridge.support_fade,
            sigma_cutoff=bridge.sigma_cutoff,
            support_fade_alpha=bridge.support_fade_alpha,
        )
    candidate = bridge.render_colors(colors_tensor.numpy())
    return _parity_report(
        candidate,
        reference.cpu().numpy(),
        atol=float(atol),
        rtol=float(rtol),
    )


def validate_cpu_color_transpose(
    bridge: NormalizedRendererBridge,
    colors: ArrayLike,
    pixel_cotangent: ArrayLike,
    *,
    atol: float = CPU_TRANSPOSE_ATOL,
    rtol: float = CPU_TRANSPOSE_RTOL,
) -> ParityReport:
    """Gate ``A.T`` against autograd of the repository renderer with respect to RGB."""

    if bridge.eps != DEFAULT_RENDER_EPS:
        raise ValueError("repository CPU transpose parity requires its fixed eps=1e-8")
    colors_tensor = _colors_tensor(bridge, colors).requires_grad_(True)
    cotangent = (
        torch.as_tensor(
            pixel_cotangent,
            dtype=torch.float32,
            device="cpu",
        )
        .detach()
        .clone()
    )
    if cotangent.shape != (bridge.height, bridge.width, 3):
        raise ValueError("pixel_cotangent must have shape (H, W, 3)")
    if not bool(torch.isfinite(cotangent).all()):
        raise ValueError("pixel_cotangent must be finite")
    rendered = render_field(
        bridge.means,
        bridge.conics,
        colors_tensor,
        bridge.radii,
        bridge.height,
        bridge.width,
        mode="normalized",
        opacities=bridge.opacities,
        support_fade=bridge.support_fade,
        sigma_cutoff=bridge.sigma_cutoff,
        support_fade_alpha=bridge.support_fade_alpha,
    )
    (rendered * cotangent).sum().backward()
    assert colors_tensor.grad is not None
    candidate = bridge.linear_map.transpose(cotangent.reshape(-1, 3).numpy())
    return _parity_report(
        candidate,
        colors_tensor.grad.detach().numpy(),
        atol=float(atol),
        rtol=float(rtol),
    )


@dataclass(frozen=True)
class OverlapDiagnostics:
    column_count: int
    nonempty_columns: int
    nonzero_elements: int
    overlap_edges: tuple[tuple[int, int], ...]
    nonzero_gram_edges: int
    degrees: tuple[int, ...]
    max_degree: int
    mean_degree: float
    graph_density: float
    diagonal_frobenius_sq: float
    off_diagonal_frobenius_sq: float
    off_diagonal_energy_fraction: float
    max_absolute_off_diagonal: float
    max_absolute_correlation: float


def overlap_diagnostics(bridge: NormalizedRendererBridge) -> OverlapDiagnostics:
    """Emit support graph and off-diagonal ``A.T A`` energy without forming the Gram matrix."""

    operator = bridge.linear_map
    per_pixel: list[list[tuple[int, float]]] = [[] for _ in range(operator.pixel_count)]
    nonempty_columns = 0
    nonzero_elements = 0
    for column in range(operator.column_count):
        footprint = operator.footprint(column)
        if footprint.pixel_indices.size:
            nonempty_columns += 1
        nonzero_elements += int(footprint.pixel_indices.size)
        for pixel, weight in zip(footprint.pixel_indices, footprint.weights, strict=True):
            per_pixel[int(pixel)].append((column, float(weight)))

    pair_products: dict[tuple[int, int], float] = {}
    support_edges: set[tuple[int, int]] = set()
    for contributions in per_pixel:
        for left in range(len(contributions)):
            left_column, left_weight = contributions[left]
            for right in range(left + 1, len(contributions)):
                right_column, right_weight = contributions[right]
                pair = (left_column, right_column)
                support_edges.add(pair)
                pair_products[pair] = pair_products.get(pair, 0.0) + left_weight * right_weight

    edges = tuple(sorted(support_edges))
    degrees = [0] * operator.column_count
    for left, right in edges:
        degrees[left] += 1
        degrees[right] += 1
    diagonal = operator.column_norm_sq
    diagonal_frobenius_sq = float(np.dot(diagonal, diagonal))
    off_diagonal_frobenius_sq = 2.0 * math.fsum(value * value for value in pair_products.values())
    total_frobenius_sq = diagonal_frobenius_sq + off_diagonal_frobenius_sq
    correlations = []
    for (left, right), value in pair_products.items():
        denominator = math.sqrt(float(diagonal[left] * diagonal[right]))
        correlations.append(0.0 if denominator == 0.0 else abs(value) / denominator)
    possible_edges = operator.column_count * (operator.column_count - 1) // 2
    return OverlapDiagnostics(
        column_count=operator.column_count,
        nonempty_columns=nonempty_columns,
        nonzero_elements=nonzero_elements,
        overlap_edges=edges,
        nonzero_gram_edges=sum(value != 0.0 for value in pair_products.values()),
        degrees=tuple(degrees),
        max_degree=max(degrees, default=0),
        mean_degree=float(np.mean(degrees)) if degrees else 0.0,
        graph_density=0.0 if possible_edges == 0 else float(len(edges) / possible_edges),
        diagonal_frobenius_sq=diagonal_frobenius_sq,
        off_diagonal_frobenius_sq=float(off_diagonal_frobenius_sq),
        off_diagonal_energy_fraction=(
            0.0
            if total_frobenius_sq == 0.0
            else float(off_diagonal_frobenius_sq / total_frobenius_sq)
        ),
        max_absolute_off_diagonal=max(
            (abs(value) for value in pair_products.values()),
            default=0.0,
        ),
        max_absolute_correlation=max(correlations, default=0.0),
    )


@dataclass(frozen=True)
class CUDANongatingReport:
    mode: str
    available: bool
    reason: str | None
    parity: ParityReport | None
    gating: bool = False


def current_cuda_nongating_diagnostics(
    bridge: NormalizedRendererBridge,
    colors: ArrayLike,
    *,
    modes: Sequence[str] = ("cuda",),
) -> tuple[CUDANongatingReport, ...]:
    """Compare current CUDA renderers when available; never turn the result into a gate."""

    colors_cpu = _colors_tensor(bridge, colors)
    reference = bridge.render_colors(colors_cpu.numpy())
    if not torch.cuda.is_available():
        return tuple(
            CUDANongatingReport(mode=mode, available=False, reason="CUDA unavailable", parity=None)
            for mode in modes
        )
    if bridge.eps != DEFAULT_RENDER_EPS:
        return tuple(
            CUDANongatingReport(
                mode=mode,
                available=False,
                reason="current render_field CUDA modes expose only eps=1e-8",
                parity=None,
            )
            for mode in modes
        )

    means = bridge.means.to("cuda")
    conics = bridge.conics.to("cuda")
    radii = bridge.radii.to("cuda")
    colors_cuda = colors_cpu.to("cuda")
    opacities = None if bridge.opacities is None else bridge.opacities.to("cuda")
    reports: list[CUDANongatingReport] = []
    for mode in modes:
        try:
            with torch.no_grad():
                rendered = render_field(
                    means,
                    conics,
                    colors_cuda,
                    radii,
                    bridge.height,
                    bridge.width,
                    mode=mode,
                    opacities=opacities,
                    support_fade=bridge.support_fade,
                    sigma_cutoff=bridge.sigma_cutoff,
                    support_fade_alpha=bridge.support_fade_alpha,
                )
            torch.cuda.synchronize()
        except RuntimeError as error:
            reports.append(
                CUDANongatingReport(
                    mode=mode,
                    available=False,
                    reason=str(error),
                    parity=None,
                )
            )
            continue
        reports.append(
            CUDANongatingReport(
                mode=mode,
                available=True,
                reason=None,
                parity=_parity_report(
                    rendered.detach().cpu().numpy(),
                    reference,
                    atol=CUDA_REPORT_ATOL,
                    rtol=CUDA_REPORT_RTOL,
                ),
            )
        )
    return tuple(reports)
