"""Exact additive pixel contributions for HIER-033/035 (ADR-0006).

Diagnostic reference only: constant RGB, no opacity/affine/filter variance, C0 fade at three
sigma. Returns detached statistics, not an autograd replacement. Tile membership is discrete;
split curvature is valid only while active support is unchanged. Finite trials are authoritative.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from .gaussians import GaussianField
from .render import _flat_tile_slices, _tile_bounds, _tile_coords


PARAMETER_ORDER = ("mean_x", "mean_y", "log_sx", "log_sy", "rotation", "red", "green", "blue")


@dataclass(frozen=True)
class PixelGradientPacket:
    signed: torch.Tensor
    absolute: torch.Tensor
    contribution_square: torch.Tensor
    gram: torch.Tensor
    split_matrix: torch.Tensor
    support_count: torch.Tensor

    @property
    def coherence(self):
        return self.signed.abs() / self.absolute.clamp_min(torch.finfo(self.signed.dtype).tiny)


def _validate_field(field: GaussianField):
    if any(x is not None for x in (field.opacities, field.color_grads, field.filter_variance)):
        raise ValueError("pixel-gradient reference requires constant RGB without opacity/filter")
    shapes = ((field.n, 2), (field.n, 2), (field.n,), (field.n, 3))
    for value, shape in zip((field.means, field.log_scales, field.rotations, field.colors), shapes):
        if value.shape != shape or not bool(torch.isfinite(value).all()):
            raise ValueError("field parameters must have valid shapes and finite values")
        if value.dtype != field.means.dtype or value.device != field.means.device:
            raise ValueError("field parameters must share dtype and device")
    if not bool(torch.isfinite(field.scales()).all()) or not bool((field.scales() > 0).all()):
        raise ValueError("scales must be finite and positive")


@torch.no_grad()
def additive_pixel_jacobians(field: GaussianField, height: int, width: int, *,
                             max_pairs: int = 65536):
    """Yield (row, pixel, weight, RGB-Jacobian, weight-mean-Hessian) bounded batches.

    J has shape (pairs,3,8), weight Hessian (pairs,2,2). The coordinate traversal is shared
    with the maintained renderer, while derivatives are evaluated independently in the RS frame.
    """
    _validate_field(field)
    if any(isinstance(x, bool) or not isinstance(x, int) or x <= 0
           for x in (height, width, max_pairs)):
        raise ValueError("image dimensions and max_pairs must be positive integers")
    means, scales, rotations, colors = (x.detach() for x in (
        field.means, field.scales(), field.rotations, field.colors))
    conics = field.conics().detach()
    bounds = _tile_bounds(means, field.radii(3.0), height, width)
    x0, y0, tile_width, tile_elements = bounds
    cutoff = math.exp(-4.5)
    for start, end in _flat_tile_slices(tile_elements, max_pairs):
        gid_all, px_all, py_all = _tile_coords(*bounds, start, end, means.device)
        for offset in range(0, gid_all.numel(), max_pairs):
            gid, px, py = (x[offset:offset + max_pairs] for x in (gid_all, px_all, py_all))
            dx, dy = px.to(means.dtype) - means[gid, 0], py.to(means.dtype) - means[gid, 1]
            a, b, c = conics[gid].unbind(1)
            raw = torch.exp(-0.5 * (a * dx.square() + 2 * b * dx * dy + c * dy.square()))
            weight = (raw - cutoff).clamp_min(0)
            # clamp_min has derivative one at exactly zero; this agrees with PyTorch's oracle.
            active_raw = torch.where(raw >= cutoff, raw, torch.zeros_like(raw))
            cosine, sine = rotations[gid].cos(), rotations[gid].sin()
            xi, eta = cosine * dx + sine * dy, -sine * dx + cosine * dy
            inv_x, inv_y = scales[gid, 0].square().reciprocal(), scales[gid, 1].square().reciprocal()
            ax, ay = a * dx + b * dy, b * dx + c * dy
            derivative = active_raw[:, None] * torch.stack((
                ax, ay, xi.square() * inv_x, eta.square() * inv_y,
                xi * eta * (inv_y - inv_x),
            ), dim=1)
            jacobian = means.new_zeros((gid.numel(), 3, 8))
            jacobian[:, :, :5] = colors[gid, :, None] * derivative[:, None, :]
            jacobian[:, :, 5:] = weight[:, None, None] * torch.eye(3, device=means.device, dtype=means.dtype)
            hessian = active_raw[:, None, None] * torch.stack((
                ax.square() - a, ax * ay - b, ax * ay - b, ay.square() - c,
            ), dim=1).reshape(-1, 2, 2)
            yield gid, py * width + px, weight, jacobian, hessian


@torch.no_grad()
def pixel_gradient_packet(field: GaussianField, image_adjoint: torch.Tensor, *,
                          max_pairs: int = 65536) -> PixelGradientPacket:
    """Accumulate per-row signed/absolute contributions, GN blocks, and split curvature.

    image_adjoint is dL/dC, including the caller's loss normalization and mask. gram is the
    unweighted image Jacobian Gram; L2 callers multiply it by their squared-loss normalization.
    Absolute activity contracts RGB first, then takes absolute values before pixel reduction.
    """
    if image_adjoint.ndim != 3 or image_adjoint.shape[-1] != 3:
        raise ValueError("image_adjoint must have HWC RGB shape")
    if image_adjoint.dtype != field.means.dtype or image_adjoint.device != field.means.device:
        raise ValueError("adjoint must match field dtype and device")
    if not bool(torch.isfinite(image_adjoint).all()):
        raise ValueError("adjoint must be finite")
    signed = field.means.new_zeros((field.n, 8))
    absolute, square = torch.zeros_like(signed), torch.zeros_like(signed)
    gram = field.means.new_zeros((field.n, 8, 8))
    split = field.means.new_zeros((field.n, 2, 2))
    support = torch.zeros(field.n, device=field.means.device, dtype=torch.int64)
    adjoint = image_adjoint.detach().reshape(-1, 3)
    for gid, pixel, weight, jacobian, hessian in additive_pixel_jacobians(
        field, *image_adjoint.shape[:2], max_pairs=max_pairs,
    ):
        contribution = (jacobian * adjoint[pixel, :, None]).sum(1)
        signed.index_add_(0, gid, contribution)
        absolute.index_add_(0, gid, contribution.abs())
        square.index_add_(0, gid, contribution.square())
        gram.index_add_(0, gid, jacobian.transpose(1, 2) @ jacobian)
        pressure = (adjoint[pixel] * field.colors.detach()[gid]).sum(1)
        split.index_add_(0, gid, pressure[:, None, None] * hessian)
        support.index_add_(0, gid, (weight > 0).to(torch.int64))
    return PixelGradientPacket(signed, absolute, square, gram, split, support)
