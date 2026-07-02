"""Differentiable reference rasterizer (ADR-0003 normalized default; additive via ADR-0006).

Normalized: I_hat(p) = sum_i c_i o_i G_i(p) / (sum_i o_i G_i(p) + eps)
Additive  : I_hat(p) = sum_i c_i o_i G_i(p)              (opt-in, ADR-0006)
with G_i(p) = exp(-1/2 (p-mu_i)^T Sigma_i^-1 (p-mu_i)) and optional per-Gaussian opacity o_i.

Each Gaussian is evaluated exactly on the pixels of its own support rectangle — the axis-aligned
bounding box of its sigma_cutoff ellipse (radii = (rx, ry) per Gaussian) — laid out as one flat
1-D tensor (ragged tiles via repeat_interleave), so nothing is padded to a shared chunk tile
size. Slices bound peak memory. Fully differentiable w.r.t. means, conics, colors, opacities.
This is the *reference*; the CUDA/Vulkan tile rasterizer (PORT-001) is the performance path and
the piece that ports into IntrinsicEngine as an RHI pass.

Requires torch.
"""
from __future__ import annotations
import torch

_EPS = 1e-8


def _flat_tile_slices(rx, ry, budget: int):
    """Split Gaussians into index ranges whose total flat tile size fits the element budget."""
    n = (2 * rx + 1) * (2 * ry + 1)
    csum = torch.cumsum(n, dim=0)
    N = n.shape[0]
    start = 0
    base = 0
    while start < N:
        end = int(torch.searchsorted(csum, base + budget, right=True))
        end = max(end, start + 1)
        yield start, min(end, N)
        start = min(end, N)
        base = int(csum[start - 1])


def _tile_coords(rx, ry, ix, iy, s, e, dev):
    """Flat per-pixel Gaussian ids and integer pixel coords for Gaussians [s, e)."""
    rxs, rys = rx[s:e], ry[s:e]
    Tx = 2 * rxs + 1
    n = Tx * (2 * rys + 1)
    ends = torch.cumsum(n, dim=0)
    total = int(ends[-1])
    gid = torch.repeat_interleave(torch.arange(s, e, device=dev), n)
    t = torch.arange(total, device=dev) - (ends - n)[gid - s]
    Txg = Tx[gid - s]
    px = ix[gid] + t % Txg - rxs[gid - s]
    py = iy[gid] + t // Txg - rys[gid - s]
    return gid, px, py


def _accumulate(means, conics, colors, radii, H, W, chunk, opacities, normalize: bool):
    dev, dt = means.device, means.dtype
    num = torch.zeros(H * W, 3, device=dev, dtype=dt)
    den = torch.zeros(H * W, 1, device=dev, dtype=dt) if normalize else None

    rx = radii[:, 0].clamp(max=W)
    ry = radii[:, 1].clamp(max=H)
    ix = torch.round(means[:, 0].detach()).long()
    iy = torch.round(means[:, 1].detach()).long()
    budget = max(chunk, 64) * 4096
    for s, e in _flat_tile_slices(rx, ry, budget):
        gid, px, py = _tile_coords(rx, ry, ix, iy, s, e, dev)
        dx = px.to(dt) - means[gid, 0]
        dy = py.to(dt) - means[gid, 1]
        a, b, c = conics[gid, 0], conics[gid, 1], conics[gid, 2]
        q = a * dx * dx + 2.0 * b * dx * dy + c * dy * dy
        w = torch.exp(-0.5 * q)
        if opacities is not None:
            w = w * opacities[gid]
        valid = (px >= 0) & (px < W) & (py >= 0) & (py < H)
        w = w * valid
        flat = py.clamp(0, H - 1) * W + px.clamp(0, W - 1)
        num = num.index_add(0, flat, w[:, None] * colors[gid])
        if normalize:
            den = den.index_add(0, flat, w[:, None])

    if normalize:
        return (num / (den + _EPS)).view(H, W, 3)
    return num.view(H, W, 3)


def render(means, conics, colors, radii, H: int, W: int, chunk: int = 4096, opacities=None):
    """Normalized weighted-sum rasterizer (ADR-0003 default)."""
    return _accumulate(means, conics, colors, radii, H, W, chunk, opacities, normalize=True)


def render_additive(means, conics, colors, radii, H: int, W: int, chunk: int = 4096, opacities=None):
    """Additive / unnormalized accumulation (ADR-0006, opt-in)."""
    return _accumulate(means, conics, colors, radii, H, W, chunk, opacities, normalize=False)


def render_field(means, conics, colors, radii, H: int, W: int,
                 chunk: int = 4096, mode: str = "normalized", opacities=None):
    if mode == "normalized":
        return render(means, conics, colors, radii, H, W, chunk, opacities)
    if mode == "additive":
        return render_additive(means, conics, colors, radii, H, W, chunk, opacities)
    raise ValueError(f"unknown renderer {mode!r}; expected normalized or additive")


@torch.no_grad()
def gaussian_activity(means, conics, radii, H: int, W: int, chunk: int = 4096):
    """Return each Gaussian's summed unnormalized weight over the image.

    This is a diagnostic/pruning helper for the reference fitter. It intentionally mirrors the
    renderer support window so a Gaussian that contributes no weight inside the image can be
    removed without changing the mathematical renderer. Opacity-free by design: the prune
    threshold is expressed in raw weight-sum units.
    """
    dev, dt = means.device, means.dtype
    N = means.shape[0]
    activity = torch.zeros(N, device=dev, dtype=dt)
    rx = radii[:, 0].clamp(max=W)
    ry = radii[:, 1].clamp(max=H)
    ix = torch.round(means[:, 0]).long()
    iy = torch.round(means[:, 1]).long()
    budget = max(chunk, 64) * 4096
    for s, e in _flat_tile_slices(rx, ry, budget):
        gid, px, py = _tile_coords(rx, ry, ix, iy, s, e, dev)
        dx = px.to(dt) - means[gid, 0]
        dy = py.to(dt) - means[gid, 1]
        a, b, c = conics[gid, 0], conics[gid, 1], conics[gid, 2]
        q = a * dx * dx + 2.0 * b * dx * dy + c * dy * dy
        w = torch.exp(-0.5 * q)
        valid = (px >= 0) & (px < W) & (py >= 0) & (py < H)
        activity.index_add_(0, gid, w * valid)
    return activity
