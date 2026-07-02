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
    def forward(ctx, means, conics, colors, radii, opacities, height, width, normalize, eps):
        ext = _load_extension()
        means = means.contiguous()
        conics = conics.contiguous()
        colors = colors.contiguous()
        radii = radii.contiguous()
        opacities = opacities.contiguous()
        out, den = ext.forward(
            means, conics, colors, radii, opacities,
            int(height), int(width), bool(normalize), float(eps),
        )
        ctx.save_for_backward(means, conics, colors, radii, opacities, den, out)
        ctx.normalize = bool(normalize)
        ctx.eps = float(eps)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        means, conics, colors, radii, opacities, den, out = ctx.saved_tensors
        ext = _load_extension()
        grad_means, grad_conics, grad_colors, grad_opacities = ext.backward(
            grad_out.contiguous(),
            means, conics, colors, radii, opacities, den.contiguous(), out.contiguous(),
            ctx.normalize, ctx.eps,
        )
        if opacities.numel() == 0:
            grad_opacities = None
        return (
            grad_means,
            grad_conics,
            grad_colors,
            None,
            grad_opacities,
            None,
            None,
            None,
            None,
        )


def render_cuda_exact(means, conics, colors, radii, H: int, W: int,
                      opacities=None, normalize: bool = True, eps: float = 1e-8):
    """Render with StructSplat's exact normalized/additive math on CUDA.

    Args mirror ``render_field``. Only float32 CUDA tensors are supported; CPU or non-float32
    callers should use the reference renderer.
    """
    if not means.is_cuda:
        raise RuntimeError("StructSplat CUDA renderer requires CUDA tensors; pass device='cuda'.")
    if means.dtype != torch.float32 or conics.dtype != torch.float32 or colors.dtype != torch.float32:
        raise RuntimeError("StructSplat CUDA renderer currently supports float32 tensors only.")
    if opacities is None:
        opacities = means.new_empty(0)
    return _ExactRenderCuda.apply(means, conics, colors, radii, opacities, H, W, normalize, eps)

