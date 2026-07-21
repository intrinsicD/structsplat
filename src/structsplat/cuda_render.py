"""Quality-preserving CUDA renderer for StructSplat's reference math.

This module owns a small PyTorch CUDA extension that matches ``render.py``'s clipped-support
normalized and additive equations. It is separate from the GaussianImage++ ``gsplat`` wrapper,
whose alpha-compositing semantics are useful for comparison but are not renderer-equivalent.
"""
from __future__ import annotations

import os
from pathlib import Path

import torch
from torch.autograd import Function
from torch.autograd.function import once_differentiable

from .render import _tile_bounds

_EXT = None


def _load_extension():
    global _EXT
    if _EXT is not None:
        return _EXT
    if not torch.cuda.is_available():
        raise RuntimeError("StructSplat CUDA renderer requires torch.cuda.is_available().")
    try:
        from torch.utils.cpp_extension import load
    except Exception as exc:  # pragma: no cover - depends on local torch install
        raise RuntimeError("StructSplat CUDA renderer requires torch.utils.cpp_extension.") from exc

    root = Path(__file__).resolve().parent / "cuda"
    sources = [str(root / "render_ext.cpp"), str(root / "render_ext.cu")]
    try:
        _EXT = load(
            name="structsplat_render_ext",
            sources=sources,
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3"],
            with_cuda=True,
            verbose=os.environ.get("STRUCTSPLAT_CUDA_VERBOSE", "0") == "1",
        )
    except Exception as exc:  # pragma: no cover - toolchain/environment dependent
        raise RuntimeError(
            "StructSplat CUDA renderer extension failed to build or load. "
            "Check CUDA_HOME, nvcc, PyTorch CUDA version, and libstdc++ compatibility. "
            f"Original error: {exc}"
        ) from exc
    return _EXT


class _ExactRenderCuda(Function):
    @staticmethod
    def forward(
        ctx,
        means,
        conics,
        colors,
        radii,
        opacities,
        height,
        width,
        normalize,
        eps,
        tiled,
        tile_size,
        support_fade,
        sigma_cutoff,
        backward_variant,
        tile_index_backend,
        tile_ellipse_cull,
    ):
        ext = _load_extension()
        means = means.contiguous()
        conics = conics.contiguous()
        colors = colors.contiguous()
        radii = radii.contiguous()
        opacities = opacities.contiguous()
        if tiled:
            # Exact ellipse culling drops (tile, Gaussian) pairs whose weight is provably zero
            # everywhere in the tile. That zero is exact only under support fade, where the
            # visible weight is max(exp(-q/2) - tail, 0); without fade every in-rectangle pixel
            # keeps a positive weight, so culling would change the rendered result and is
            # disabled regardless of the requested flag.
            cull = bool(tile_ellipse_cull) and bool(support_fade)
            if tile_index_backend == "cuda":
                tile_gids, tile_offsets = ext.build_tile_index(
                    means, conics.detach(), radii, int(height), int(width), int(tile_size),
                    cull, float(sigma_cutoff),
                )
            else:
                tile_gids, tile_offsets = _build_tile_index(means, radii, int(height),
                                                            int(width), int(tile_size))
                if tile_gids.numel() >= 2 ** 31 or tile_offsets.numel() >= 2 ** 31:
                    raise RuntimeError("tile index exceeds int32; reduce N or image size")
                tile_gids = tile_gids.to(torch.int32)
                tile_offsets = tile_offsets.to(torch.int32)
            out, den = ext.forward_tiled(
                means, conics, colors, radii, opacities, tile_gids, tile_offsets,
                int(height), int(width), bool(normalize), float(eps), int(tile_size),
                bool(support_fade), float(sigma_cutoff),
            )
        else:
            tile_gids = means.new_empty(0, dtype=torch.int32)
            tile_offsets = means.new_empty(0, dtype=torch.int32)
            out, den = ext.forward(
                means, conics, colors, radii, opacities,
                int(height), int(width), bool(normalize), float(eps),
                bool(support_fade), float(sigma_cutoff),
            )
        ctx.save_for_backward(
            means, conics, colors, radii, opacities, den, out, tile_gids, tile_offsets
        )
        ctx.normalize = bool(normalize)
        ctx.eps = float(eps)
        ctx.tiled = bool(tiled)
        ctx.tile_size = int(tile_size)
        ctx.support_fade = bool(support_fade)
        ctx.sigma_cutoff = float(sigma_cutoff)
        ctx.backward_variant = str(backward_variant)
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_out):
        means, conics, colors, radii, opacities, den, out, tile_gids, tile_offsets = (
            ctx.saved_tensors
        )
        # forward signature:
        # means, conics, colors, radii, opacities, height, width, normalize, eps, tiled, tile_size,
        # support_fade, sigma_cutoff, backward_variant, tile_index_backend, tile_ellipse_cull
        need_means, need_conics, need_colors, _, need_opac = ctx.needs_input_grad[:5]
        if not (need_means or need_conics or need_colors or need_opac):
            return (None,) * 16
        ext = _load_extension()
        if ctx.tiled:
            grad_means, grad_conics, grad_colors, grad_opacities = ext.backward_tiled(
                grad_out.contiguous(),
                means, conics, colors, radii, opacities, den.contiguous(), out.contiguous(),
                tile_gids, tile_offsets, ctx.normalize, ctx.eps, ctx.tile_size,
                ctx.support_fade, ctx.sigma_cutoff,
            )
        elif ctx.backward_variant == "baseline":
            grad_means, grad_conics, grad_colors, grad_opacities = ext.backward(
                grad_out.contiguous(),
                means, conics, colors, radii, opacities, den.contiguous(), out.contiguous(),
                ctx.normalize, ctx.eps, ctx.support_fade, ctx.sigma_cutoff,
            )
        else:
            grad_means, grad_conics, grad_colors, grad_opacities = ext.backward_block_reduce(
                grad_out.contiguous(),
                means, conics, colors, radii, opacities, den.contiguous(), out.contiguous(),
                ctx.normalize, ctx.eps, ctx.support_fade, ctx.sigma_cutoff,
            )
        if opacities.numel() == 0 or not need_opac:
            grad_opacities = None
        return (
            grad_means if need_means else None,
            grad_conics if need_conics else None,
            grad_colors if need_colors else None,
            None,
            grad_opacities,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


@torch.no_grad()
def _build_tile_index(means: torch.Tensor, radii: torch.Tensor, H: int, W: int,
                      tile_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    if tile_size <= 0 or tile_size * tile_size > 1024:
        raise ValueError("tile_size must be positive with tile_size^2 <= 1024")
    tiles_x = (W + tile_size - 1) // tile_size
    tiles_y = (H + tile_size - 1) // tile_size
    num_tiles = tiles_x * tiles_y
    if means.shape[0] == 0 or num_tiles == 0:
        offsets = torch.zeros(num_tiles + 1, device=means.device, dtype=torch.long)
        gids = torch.empty(0, device=means.device, dtype=torch.long)
        return gids, offsets

    x0, y0, tx, n = _tile_bounds(means, radii, H, W)
    valid = n > 0
    if not bool(valid.any()):
        offsets = torch.zeros(num_tiles + 1, device=means.device, dtype=torch.long)
        gids = torch.empty(0, device=means.device, dtype=torch.long)
        return gids, offsets

    gid_base = torch.arange(means.shape[0], device=means.device, dtype=torch.long)[valid]
    x0 = x0[valid]
    y0 = y0[valid]
    tx = tx[valid]
    ty = (n[valid] // tx).long()
    tile_x0 = torch.div(x0, tile_size, rounding_mode="floor")
    tile_y0 = torch.div(y0, tile_size, rounding_mode="floor")
    tile_x1 = torch.div(x0 + tx - 1, tile_size, rounding_mode="floor")
    tile_y1 = torch.div(y0 + ty - 1, tile_size, rounding_mode="floor")
    tile_w = tile_x1 - tile_x0 + 1
    tile_h = tile_y1 - tile_y0 + 1
    counts = tile_w * tile_h
    gids = torch.repeat_interleave(gid_base, counts)
    if gids.numel() == 0:
        offsets = torch.zeros(num_tiles + 1, device=means.device, dtype=torch.long)
        return gids, offsets
    starts = torch.cumsum(counts, dim=0) - counts
    local = torch.arange(gids.numel(), device=means.device, dtype=torch.long)
    local = local - torch.repeat_interleave(starts, counts)
    tile_x = torch.repeat_interleave(tile_x0, counts) + local % torch.repeat_interleave(tile_w, counts)
    tile_y = torch.repeat_interleave(tile_y0, counts) + local // torch.repeat_interleave(tile_w, counts)
    tile_ids = (tile_y * tiles_x + tile_x).long()
    order = torch.argsort(tile_ids)
    tile_ids = tile_ids[order]
    gids = gids[order].contiguous()
    per_tile = torch.bincount(tile_ids, minlength=num_tiles)
    offsets = torch.empty(num_tiles + 1, device=means.device, dtype=torch.long)
    offsets[0] = 0
    offsets[1:] = torch.cumsum(per_tile, dim=0)
    return gids, offsets.contiguous()


def render_cuda_exact(means, conics, colors, radii, H: int, W: int,
                      opacities=None, normalize: bool = True, eps: float = 1e-8,
                      tiled: bool = False, tile_size: int = 16,
                      support_fade: bool = False, sigma_cutoff: float = 3.0,
                      backward_variant: str = "baseline",
                      tile_index_backend: str = "cuda",
                      tile_ellipse_cull: bool = True):
    """Render with StructSplat's exact normalized/additive math on CUDA.

    Args mirror ``render_field``. Only float32 CUDA tensors are supported; CPU or non-float32
    callers should use the reference renderer.

    ``tile_index_backend`` selects how the tiled path builds its tile-to-Gaussian index:
    ``"cuda"`` (default) bins, CUB-sorts, and ranges entirely inside the extension;
    ``"torch"`` keeps the original torch-op builder for parity testing. ``tile_ellipse_cull``
    additionally drops (tile, Gaussian) pairs whose support-fade weight is exactly zero across
    the whole tile; it is a no-op unless ``support_fade`` is on.
    """
    if not means.is_cuda:
        raise RuntimeError("StructSplat CUDA renderer requires CUDA tensors; pass device='cuda'.")
    if means.dtype != torch.float32 or conics.dtype != torch.float32 or colors.dtype != torch.float32:
        raise RuntimeError("StructSplat CUDA renderer currently supports float32 tensors only.")
    if backward_variant not in ("baseline", "block_reduce"):
        raise ValueError(
            "backward_variant must be baseline or block_reduce, "
            f"got {backward_variant!r}"
        )
    if tiled and backward_variant != "baseline":
        raise ValueError("block_reduce backward is experimental and untiled-only")
    if tile_index_backend not in ("cuda", "torch"):
        raise ValueError(
            f"tile_index_backend must be cuda or torch, got {tile_index_backend!r}"
        )
    if opacities is None:
        opacities = means.new_empty(0)
    return _ExactRenderCuda.apply(
        means, conics, colors, radii, opacities, H, W, normalize, eps, tiled, tile_size,
        support_fade, sigma_cutoff, backward_variant, tile_index_backend, tile_ellipse_cull
    )
