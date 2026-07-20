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
import math
import torch

_EPS = 1e-8
_ELEMENTS_PER_RENDER_CHUNK = 4096
_NORMALIZED_CUDA_MODES = ("cuda", "cuda_normalized", "cuda_tiled", "cuda_tiled_normalized")
_ADDITIVE_CUDA_MODES = ("cuda_additive", "cuda_tiled_additive")


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


def _resolve_support_fade_alpha(support_fade: bool,
                                support_fade_alpha: float | None = None) -> float:
    if support_fade_alpha is None:
        return 1.0 if support_fade else 0.0
    return max(0.0, min(1.0, float(support_fade_alpha)))


def _support_weight(q: torch.Tensor, sigma_cutoff: float, support_fade: bool,
                    support_fade_alpha: float | None = None) -> torch.Tensor:
    w = torch.exp(-0.5 * q)
    alpha = _resolve_support_fade_alpha(support_fade, support_fade_alpha)
    if alpha > 0.0:
        w = torch.clamp(w - alpha * math.exp(-0.5 * float(sigma_cutoff) ** 2), min=0.0)
    return w


def _pixel_colors(colors, color_grads, gid, dx, dy, scales, rotations):
    base = colors[gid]
    if color_grads is None:
        return base
    if scales is None or rotations is None:
        raise ValueError("affine color rendering requires scales and rotations")
    theta = rotations[gid]
    c, s = torch.cos(theta), torch.sin(theta)
    sx = scales[gid, 0].clamp_min(1e-6)
    sy = scales[gid, 1].clamp_min(1e-6)
    local_x = (c * dx + s * dy) / sx
    local_y = (-s * dx + c * dy) / sy
    grads = color_grads[gid]
    return base + grads[:, 0, :] * local_x[:, None] + grads[:, 1, :] * local_y[:, None]


def _slice_contribution(means, conics, colors, x0, y0, Tx, n, H, W, opacities, normalize: bool,
                        s: int, e: int, support_fade: bool, sigma_cutoff: float,
                        color_grads=None, scales=None, rotations=None,
                        support_fade_alpha: float | None = None):
    dev, dt = means.device, means.dtype
    gid, px, py = _tile_coords(x0, y0, Tx, n, s, e, dev)
    dx = px.to(dt) - means[gid, 0]
    dy = py.to(dt) - means[gid, 1]
    a, b, c = conics[gid, 0], conics[gid, 1], conics[gid, 2]
    q = a * dx * dx + 2.0 * b * dx * dy + c * dy * dy
    w = _support_weight(q, sigma_cutoff, support_fade, support_fade_alpha)
    if opacities is not None:
        w = w * opacities[gid]
    flat = py * W + px
    pix_colors = _pixel_colors(colors, color_grads, gid, dx, dy, scales, rotations)
    num = torch.zeros(H * W, 3, device=dev, dtype=dt)
    num = num.index_add(0, flat, w[:, None] * pix_colors)
    if not normalize:
        return num
    den = torch.zeros(H * W, 1, device=dev, dtype=dt)
    den = den.index_add(0, flat, w[:, None])
    return num, den


def _checkpoint_inputs(means, conics, colors, opacities, color_grads, scales, rotations):
    inputs = [means, conics, colors]
    has = []
    for value in (opacities, color_grads, scales, rotations):
        has.append(value is not None)
        if value is not None:
            inputs.append(value)
    return inputs, tuple(has)


def _unpack_checkpoint_extras(extras, has):
    idx = 0
    out = []
    for present in has:
        if present:
            out.append(extras[idx])
            idx += 1
        else:
            out.append(None)
    return out


def _should_checkpoint(checkpoint_chunks: bool, inputs: list[torch.Tensor]) -> bool:
    return (
        checkpoint_chunks
        and torch.is_grad_enabled()
        and any(t.requires_grad for t in inputs)
    )


def _accumulate(means, conics, colors, radii, H, W, chunk, opacities, normalize: bool,
                support_fade: bool = False, sigma_cutoff: float = 3.0,
                color_grads=None, scales=None, rotations=None,
                support_fade_alpha: float | None = None,
                checkpoint_chunks: bool = False):
    dev, dt = means.device, means.dtype
    num = torch.zeros(H * W, 3, device=dev, dtype=dt)
    den = torch.zeros(H * W, 1, device=dev, dtype=dt) if normalize else None

    x0, y0, Tx, n = _tile_bounds(means, radii, H, W)
    budget = _element_budget(chunk)
    checkpoint_inputs, checkpoint_has = _checkpoint_inputs(
        means, conics, colors, opacities, color_grads, scales, rotations
    )
    use_checkpoint = _should_checkpoint(checkpoint_chunks, checkpoint_inputs)
    if use_checkpoint:
        from torch.utils.checkpoint import checkpoint
    for s, e in _flat_tile_slices(n, budget):
        if use_checkpoint:
            def chunk_fn(means_, conics_, colors_, *extras, _s=s, _e=e):
                opacities_, color_grads_, scales_, rotations_ = _unpack_checkpoint_extras(
                    extras, checkpoint_has
                )
                return _slice_contribution(
                    means_, conics_, colors_, x0, y0, Tx, n, H, W, opacities_, normalize,
                    _s, _e, support_fade, sigma_cutoff, color_grads=color_grads_,
                    scales=scales_, rotations=rotations_,
                    support_fade_alpha=support_fade_alpha,
                )

            contrib = checkpoint(chunk_fn, *checkpoint_inputs, use_reentrant=False)
            if normalize:
                n_part, d_part = contrib
                num = num + n_part
                den = den + d_part
            else:
                num = num + contrib
        else:
            gid, px, py = _tile_coords(x0, y0, Tx, n, s, e, dev)
            dx = px.to(dt) - means[gid, 0]
            dy = py.to(dt) - means[gid, 1]
            a, b, c = conics[gid, 0], conics[gid, 1], conics[gid, 2]
            q = a * dx * dx + 2.0 * b * dx * dy + c * dy * dy
            w = _support_weight(q, sigma_cutoff, support_fade, support_fade_alpha)
            if opacities is not None:
                w = w * opacities[gid]
            flat = py * W + px
            pix_colors = _pixel_colors(colors, color_grads, gid, dx, dy, scales, rotations)
            num = num.index_add(0, flat, w[:, None] * pix_colors)
            if normalize:
                den = den.index_add(0, flat, w[:, None])

    if normalize:
        return (num / (den + _EPS)).view(H, W, 3)
    return num.view(H, W, 3)


def render(means, conics, colors, radii, H: int, W: int, chunk: int = 4096, opacities=None,
           support_fade: bool = False, sigma_cutoff: float = 3.0, color_grads=None,
           scales=None, rotations=None, support_fade_alpha: float | None = None,
           checkpoint_chunks: bool = False):
    """Normalized weighted-sum rasterizer (ADR-0003 default)."""
    return _accumulate(
        means, conics, colors, radii, H, W, chunk, opacities, normalize=True,
        support_fade=support_fade, sigma_cutoff=sigma_cutoff, color_grads=color_grads,
        scales=scales, rotations=rotations, support_fade_alpha=support_fade_alpha,
        checkpoint_chunks=checkpoint_chunks,
    )


def render_additive(means, conics, colors, radii, H: int, W: int, chunk: int = 4096,
                    opacities=None, support_fade: bool = False, sigma_cutoff: float = 3.0,
                    color_grads=None, scales=None, rotations=None,
                    support_fade_alpha: float | None = None,
                    checkpoint_chunks: bool = False):
    """Additive / unnormalized accumulation (ADR-0006, opt-in)."""
    return _accumulate(
        means, conics, colors, radii, H, W, chunk, opacities, normalize=False,
        support_fade=support_fade, sigma_cutoff=sigma_cutoff, color_grads=color_grads,
        scales=scales, rotations=rotations, support_fade_alpha=support_fade_alpha,
        checkpoint_chunks=checkpoint_chunks,
    )


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
                 scales=None, rotations=None, support_fade: bool = False,
                 sigma_cutoff: float = 3.0, color_grads=None,
                 support_fade_alpha: float | None = None,
                 checkpoint_chunks: bool = False):
    fade_alpha = _resolve_support_fade_alpha(support_fade, support_fade_alpha)
    if mode == "normalized":
        return render(
            means, conics, colors, radii, H, W, chunk, opacities,
            support_fade=support_fade, sigma_cutoff=sigma_cutoff, color_grads=color_grads,
            scales=scales, rotations=rotations, support_fade_alpha=fade_alpha,
            checkpoint_chunks=checkpoint_chunks,
        )
    if mode == "additive":
        return render_additive(
            means, conics, colors, radii, H, W, chunk, opacities,
            support_fade=support_fade, sigma_cutoff=sigma_cutoff, color_grads=color_grads,
            scales=scales, rotations=rotations, support_fade_alpha=fade_alpha,
            checkpoint_chunks=checkpoint_chunks,
        )
    if fade_alpha not in (0.0, 1.0) and mode in _NORMALIZED_CUDA_MODES:
        if not means.is_cuda:
            raise RuntimeError("StructSplat CUDA renderer requires CUDA tensors; pass device='cuda'.")
        return render(
            means, conics, colors, radii, H, W, chunk, opacities,
            support_fade=support_fade, sigma_cutoff=sigma_cutoff, color_grads=color_grads,
            scales=scales, rotations=rotations, support_fade_alpha=fade_alpha,
            checkpoint_chunks=checkpoint_chunks,
        )
    if fade_alpha not in (0.0, 1.0) and mode in _ADDITIVE_CUDA_MODES:
        if not means.is_cuda:
            raise RuntimeError("StructSplat CUDA renderer requires CUDA tensors; pass device='cuda'.")
        return render_additive(
            means, conics, colors, radii, H, W, chunk, opacities,
            support_fade=support_fade, sigma_cutoff=sigma_cutoff, color_grads=color_grads,
            scales=scales, rotations=rotations, support_fade_alpha=fade_alpha,
            checkpoint_chunks=checkpoint_chunks,
        )
    if color_grads is not None and mode in _NORMALIZED_CUDA_MODES:
        if not means.is_cuda:
            raise RuntimeError("StructSplat CUDA renderer requires CUDA tensors; pass device='cuda'.")
        # The custom CUDA extension has no affine-color backward kernel yet. Use the exact
        # normalized reference equation on CUDA so ABL/fair-regime quality cells can run without
        # changing renderer semantics; timing for this arm must be interpreted as a fallback.
        return render(
            means, conics, colors, radii, H, W, chunk, opacities,
            support_fade=support_fade, sigma_cutoff=sigma_cutoff, color_grads=color_grads,
            scales=scales, rotations=rotations, support_fade_alpha=fade_alpha,
            checkpoint_chunks=checkpoint_chunks,
        )
    if color_grads is not None:
        raise ValueError(
            "affine color rendering is currently supported by the reference normalized/additive "
            f"renderers only; got renderer={mode!r}"
        )
    if mode in ("cuda", "cuda_normalized"):
        from .cuda_render import render_cuda_exact
        return render_cuda_exact(means, conics, colors, radii, H, W, opacities=opacities,
                                 normalize=True, eps=_EPS, support_fade=fade_alpha > 0.0,
                                 sigma_cutoff=sigma_cutoff)
    if mode == "cuda_additive":
        from .cuda_render import render_cuda_exact
        return render_cuda_exact(means, conics, colors, radii, H, W, opacities=opacities,
                                 normalize=False, eps=_EPS, support_fade=fade_alpha > 0.0,
                                 sigma_cutoff=sigma_cutoff)
    if mode in ("cuda_tiled", "cuda_tiled_normalized"):
        from .cuda_render import render_cuda_exact
        return render_cuda_exact(means, conics, colors, radii, H, W, opacities=opacities,
                                 normalize=True, eps=_EPS, tiled=True,
                                 support_fade=fade_alpha > 0.0, sigma_cutoff=sigma_cutoff)
    if mode == "cuda_tiled_additive":
        from .cuda_render import render_cuda_exact
        return render_cuda_exact(means, conics, colors, radii, H, W, opacities=opacities,
                                 normalize=False, eps=_EPS, tiled=True,
                                 support_fade=fade_alpha > 0.0, sigma_cutoff=sigma_cutoff)
    if mode in ("gsplat", "cuda_gsplat"):
        if scales is None or rotations is None:
            raise ValueError("renderer='gsplat' requires scales and rotations")
        return render_cuda_sum(means, scales, rotations, colors, H, W, opacities=opacities)
    raise ValueError(
        f"unknown renderer {mode!r}; expected normalized, additive, cuda, "
        "cuda_additive, cuda_tiled, cuda_tiled_additive, or gsplat"
    )


@torch.no_grad()
def gaussian_activity(means, conics, radii, H: int, W: int, chunk: int = 4096,
                      support_fade: bool = False, sigma_cutoff: float = 3.0,
                      support_fade_alpha: float | None = None):
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
        activity.index_add_(
            0, gid, _support_weight(q, sigma_cutoff, support_fade, support_fade_alpha)
        )
    return activity
