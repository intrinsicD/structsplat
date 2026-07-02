"""Differentiable reference rasterizer: normalized weighted summation, no depth sort (ADR-0003).

I_hat(p) = sum_i c_i G_i(p) / (sum_i G_i(p) + eps),  G_i(p)=exp(-1/2 (p-mu_i)^T Sigma_i^-1 (p-mu_i))

Chunked + vectorized per Gaussian bounding box. Correct and fully differentiable w.r.t. means,
conics, and colors. This is the *reference*; the CUDA/Vulkan tile rasterizer (PORT-001) is the
performance path and the piece that ports into IntrinsicEngine as an RHI pass.

Requires torch.
"""
from __future__ import annotations
import torch

_EPS = 1e-8


def render(means, conics, colors, radii, H: int, W: int, chunk: int = 4096):
    dev, dt = means.device, means.dtype
    N = means.shape[0]
    num = torch.zeros(H * W, 3, device=dev, dtype=dt)
    den = torch.zeros(H * W, 1, device=dev, dtype=dt)

    # sort by radius so each chunk shares a tile size (bounds memory, avoids clipping)
    order = torch.argsort(radii)
    for start in range(0, N, chunk):
        idx = order[start:start + chunk]
        r = int(torch.clamp(radii[idx].max(), min=1).item())
        T = 2 * r + 1
        mu = means[idx]                                   # (M,2)
        cx, cy = mu[:, 0], mu[:, 1]
        ix = torch.round(cx).long()
        iy = torch.round(cy).long()
        off = torch.arange(-r, r + 1, device=dev)
        oy, ox = torch.meshgrid(off, off, indexing="ij")  # (T,T)
        px = ix[:, None, None] + ox[None]                 # (M,T,T)
        py = iy[:, None, None] + oy[None]
        dx = px.to(dt) - cx[:, None, None]
        dy = py.to(dt) - cy[:, None, None]
        a = conics[idx, 0][:, None, None]
        b = conics[idx, 1][:, None, None]
        c = conics[idx, 2][:, None, None]
        q = a * dx * dx + 2.0 * b * dx * dy + c * dy * dy
        w = torch.exp(-0.5 * q)                           # (M,T,T)
        valid = (px >= 0) & (px < W) & (py >= 0) & (py < H)
        w = w * valid
        flat = (py.clamp(0, H - 1) * W + px.clamp(0, W - 1)).reshape(-1)
        col = colors[idx][:, None, None, :].expand(-1, T, T, 3)
        wc = (w[..., None] * col).reshape(-1, 3)
        num = num.index_add(0, flat, wc)
        den = den.index_add(0, flat, w.reshape(-1, 1))

    img = num / (den + _EPS)
    return img.view(H, W, 3)


@torch.no_grad()
def gaussian_activity(means, conics, radii, H: int, W: int, chunk: int = 4096):
    """Return each Gaussian's summed unnormalized weight over the image.

    This is a diagnostic/pruning helper for the reference fitter. It intentionally mirrors the
    renderer support window so a Gaussian that contributes no weight inside the image can be
    removed without changing the mathematical renderer.
    """
    dev, dt = means.device, means.dtype
    activity = torch.zeros(means.shape[0], device=dev, dtype=dt)
    order = torch.argsort(radii)
    for start in range(0, means.shape[0], chunk):
        idx = order[start:start + chunk]
        r = int(torch.clamp(radii[idx].max(), min=1).item())
        mu = means[idx]
        cx, cy = mu[:, 0], mu[:, 1]
        ix = torch.round(cx).long()
        iy = torch.round(cy).long()
        off = torch.arange(-r, r + 1, device=dev)
        oy, ox = torch.meshgrid(off, off, indexing="ij")
        px = ix[:, None, None] + ox[None]
        py = iy[:, None, None] + oy[None]
        dx = px.to(dt) - cx[:, None, None]
        dy = py.to(dt) - cy[:, None, None]
        a = conics[idx, 0][:, None, None]
        b = conics[idx, 1][:, None, None]
        c = conics[idx, 2][:, None, None]
        q = a * dx * dx + 2.0 * b * dx * dy + c * dy * dy
        w = torch.exp(-0.5 * q)
        valid = (px >= 0) & (px < W) & (py >= 0) & (py < H)
        activity[idx] = (w * valid).flatten(1).sum(dim=1)
    return activity
