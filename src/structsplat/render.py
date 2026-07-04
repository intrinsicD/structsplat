"""Differentiable rasterizers (ADR-0003 normalized default; additive via ADR-0006).

Normalized: I_hat(p) = sum_i c_i o_i G_i(p) / (sum_i o_i G_i(p) + eps)
Additive  : I_hat(p) = sum_i c_i o_i G_i(p)              (opt-in, ADR-0006)
CUDA      : exact normalized/additive CUDA extension      (opt-in)
cuda_tiled: exact tiled/culling CUDA extension             (opt-in)
gsplat    : optional GaussianImage++ alpha/sum renderer   (experimental)
with G_i(p) = exp(-1/2 (p-mu_i)^T Sigma_i^-1 (p-mu_i)) and optional per-Gaussian opacity o_i.

Each Gaussian is evaluated exactly on the pixels of its own support rectangle — the axis-aligned
bounding box of its sigma_cutoff ellipse (radii = (rx, ry) per Gaussian) intersected with the
image (CORE-003) — laid out as one flat 1-D tensor (ragged tiles via repeat_interleave), so
nothing is padded to a shared chunk tile size and no element is spent on off-image pixels.
Slices bound peak memory. Fully differentiable w.r.t. means, conics, colors, opacities.
This is the *reference*; the CUDA/Vulkan tile rasterizer (PORT-001) is the performance path and
the piece that ports into IntrinsicEngine as an RHI pass.

Requires torch. The CUDA path additionally requires a local CUDA toolchain for the extension.
"""
from __future__ import annotations
import torch

_EPS = 1e-8
_ELEMENTS_PER_RENDER_CHUNK = 4096


def _element_budget(chunk: int) -> int:
    """Flat tile-element budget for reference-renderer slices.

    `chunk` is expressed in 4096-element units, with a 64-unit floor for small test configs.
    """
    return max(int(chunk), 64) * _ELEMENTS_PER_RENDER_CHUNK


def _flat_tile_slices(n, budget: int):
    """Split Gaussians into index ranges whose total flat tile size fits the element budget."""
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


def _tile_bounds(means, radii, H: int, W: int):
    """Per-Gaussian support rectangle CLIPPED to the image (CORE-003).

    Clipping (instead of clamping the radius symmetrically) keeps the in-image part of the
    support of Gaussians whose center sits outside the image or whose extent exceeds it, and
    spends no tile elements on off-image pixels — so no validity mask is needed downstream.
    Fully-outside Gaussians get an empty (zero-area) tile.
    """
    ix = torch.round(means[:, 0].detach()).long()
    iy = torch.round(means[:, 1].detach()).long()
    x0 = (ix - radii[:, 0]).clamp(min=0)
    x1 = (ix + radii[:, 0]).clamp(max=W - 1)
    y0 = (iy - radii[:, 1]).clamp(min=0)
    y1 = (iy + radii[:, 1]).clamp(max=H - 1)
    Tx = (x1 - x0 + 1).clamp(min=0)
    n = Tx * (y1 - y0 + 1).clamp(min=0)
    return x0, y0, Tx, n


def _tile_coords(x0, y0, Tx, n, s, e, dev):
    """Flat per-pixel Gaussian ids and integer pixel coords for Gaussians [s, e)."""
    ns = n[s:e]
    ends = torch.cumsum(ns, dim=0)
    total = int(ends[-1]) if ns.numel() else 0
    gid = torch.repeat_interleave(torch.arange(s, e, device=dev), ns)
    t = torch.arange(total, device=dev) - (ends - ns)[gid - s]
    Txg = Tx[gid]
    px = x0[gid] + t % Txg
    py = y0[gid] + t // Txg
    return gid, px, py


def _accumulate(means, conics, colors, radii, H, W, chunk, opacities, normalize: bool):
    dev, dt = means.device, means.dtype
    num = torch.zeros(H * W, 3, device=dev, dtype=dt)
    den = torch.zeros(H * W, 1, device=dev, dtype=dt) if normalize else None

    x0, y0, Tx, n = _tile_bounds(means, radii, H, W)
    budget = _element_budget(chunk)
    for s, e in _flat_tile_slices(n, budget):
        gid, px, py = _tile_coords(x0, y0, Tx, n, s, e, dev)
        dx = px.to(dt) - means[gid, 0]
        dy = py.to(dt) - means[gid, 1]
        a, b, c = conics[gid, 0], conics[gid, 1], conics[gid, 2]
        q = a * dx * dx + 2.0 * b * dx * dy + c * dy * dy
        w = torch.exp(-0.5 * q)
        if opacities is not None:
            w = w * opacities[gid]
        flat = py * W + px
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


def _gsplat_import_error(exc: BaseException) -> RuntimeError:
    return RuntimeError(
        "renderer='gsplat' requires a working local gsplat CUDA extension. "
        "The Python package may import successfully while the compiled extension still fails "
        "to load; check CUDA/PyTorch/libstdc++ compatibility. "
        f"Original error: {exc}"
    )


def render_cuda_sum(means, scales, rotations, colors, H: int, W: int,
                    opacities=None, block: int = 16, radius_clip: float = 1.0):
    """CUDA additive/sum rasterizer backed by GaussianImage/GaussianImage++ gsplat.

    The local gsplat kernel uses pixel-coordinate 2D means/scales/rotations, matching
    StructSplat's field parameterization. It is intentionally exposed as a separate additive
    renderer stage because it does not implement the normalized weighted-sum reference.
    """
    if not means.is_cuda:
        raise RuntimeError("renderer='cuda' requires CUDA tensors; pass device='cuda'.")
    try:
        from gsplat import project_gaussians_2d_scale_rot, rasterize_gaussians_plus
    except Exception as exc:  # pragma: no cover - environment dependent
        raise _gsplat_import_error(exc) from exc

    if means.numel() == 0:
        return torch.zeros(H, W, 3, device=means.device, dtype=colors.dtype)
    scales = scales.to(device=means.device, dtype=means.dtype)
    rotations = rotations.reshape(-1, 1).to(device=means.device, dtype=means.dtype)
    opacity = (
        torch.ones(means.shape[0], 1, device=means.device, dtype=means.dtype)
        if opacities is None
        else opacities.reshape(-1, 1).to(device=means.device, dtype=means.dtype)
    )
    colors = colors.to(device=means.device, dtype=means.dtype)
    tile_bounds = ((W + block - 1) // block, (H + block - 1) // block, 1)
    background = torch.zeros(3, device=means.device, dtype=means.dtype)
    try:
        xys, depths, cuda_radii, cuda_conics, num_tiles_hit = project_gaussians_2d_scale_rot(
            means, scales, rotations, H, W, tile_bounds, coords_norm=False,
            radius_clip=radius_clip,
        )
        out = rasterize_gaussians_plus(
            xys, depths, cuda_radii, cuda_conics, num_tiles_hit, colors, opacity,
            H, W, block, block, background=background, return_alpha=False,
            radius_clip=radius_clip,
        )
    except Exception as exc:  # pragma: no cover - environment dependent
        raise _gsplat_import_error(exc) from exc
    if isinstance(out, tuple):
        out = out[0]
    return out.reshape(H, W, -1)[..., :3]


def render_field(means, conics, colors, radii, H: int, W: int,
                 chunk: int = 4096, mode: str = "normalized", opacities=None,
                 scales=None, rotations=None):
    if mode == "normalized":
        return render(means, conics, colors, radii, H, W, chunk, opacities)
    if mode == "additive":
        return render_additive(means, conics, colors, radii, H, W, chunk, opacities)
    if mode in ("cuda", "cuda_normalized"):
        from .cuda_render import render_cuda_exact
        return render_cuda_exact(means, conics, colors, radii, H, W, opacities=opacities,
                                 normalize=True, eps=_EPS)
    if mode == "cuda_additive":
        from .cuda_render import render_cuda_exact
        return render_cuda_exact(means, conics, colors, radii, H, W, opacities=opacities,
                                 normalize=False, eps=_EPS)
    if mode in ("cuda_tiled", "cuda_tiled_normalized"):
        from .cuda_render import render_cuda_exact
        return render_cuda_exact(means, conics, colors, radii, H, W, opacities=opacities,
                                 normalize=True, eps=_EPS, tiled=True)
    if mode == "cuda_tiled_additive":
        from .cuda_render import render_cuda_exact
        return render_cuda_exact(means, conics, colors, radii, H, W, opacities=opacities,
                                 normalize=False, eps=_EPS, tiled=True)
    if mode in ("gsplat", "cuda_gsplat"):
        if scales is None or rotations is None:
            raise ValueError("renderer='gsplat' requires scales and rotations")
        return render_cuda_sum(means, scales, rotations, colors, H, W, opacities=opacities)
    raise ValueError(
        f"unknown renderer {mode!r}; expected normalized, additive, cuda, "
        "cuda_additive, cuda_tiled, cuda_tiled_additive, or gsplat"
    )


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
    x0, y0, Tx, n = _tile_bounds(means, radii, H, W)
    budget = _element_budget(chunk)
    for s, e in _flat_tile_slices(n, budget):
        gid, px, py = _tile_coords(x0, y0, Tx, n, s, e, dev)
        dx = px.to(dt) - means[gid, 0]
        dy = py.to(dt) - means[gid, 1]
        a, b, c = conics[gid, 0], conics[gid, 1], conics[gid, 2]
        q = a * dx * dx + 2.0 * b * dx * dy + c * dy * dy
        activity.index_add_(0, gid, torch.exp(-0.5 * q))
    return activity
