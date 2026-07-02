"""Structure tensor J = G_rho * (grad I  grad I^T) and its eigen-analysis.

This single operator drives everything downstream (ADR-0004):
  * energy  = lam1 + lam2      -> the density field for sampling
  * coherence, classification  -> flat / edge / corner decision
  * across_edge_angle          -> orientation for anisotropic covariance init

Pure NumPy: no autograd needed at init time, and it stays importable without torch.
All maps are float32, shape (H, W). Image input is (H, W, 3) or (H, W) in [0, 1].
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass

from .config import StructureTensorConfig

_LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)  # Rec.709


def to_luma(img: np.ndarray) -> np.ndarray:
    img = np.asarray(img, dtype=np.float32)
    if img.ndim == 2:
        return img
    return img[..., :3] @ _LUMA


def _gaussian_kernel(sigma: float) -> np.ndarray:
    radius = max(1, int(round(3.0 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    return (k / k.sum()).astype(np.float32)


def _conv1d(a: np.ndarray, k: np.ndarray, axis: int) -> np.ndarray:
    """Separable 1D convolution along `axis` with reflect padding."""
    r = len(k) // 2
    pad = [(r, r) if ax == axis else (0, 0) for ax in range(a.ndim)]
    ap = np.pad(a, pad, mode="reflect")
    out = np.zeros_like(a)
    for i, w in enumerate(k):
        sl = [slice(None)] * a.ndim
        sl[axis] = slice(i, i + a.shape[axis])
        out += w * ap[tuple(sl)]
    return out


def _conv2d(a: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Small reflect-padded 2D convolution for fixed gradient kernels."""
    kh, kw = k.shape
    rh, rw = kh // 2, kw // 2
    ap = np.pad(a, ((rh, rh), (rw, rw)), mode="reflect")
    out = np.zeros_like(a, dtype=np.float32)
    for y in range(kh):
        for x in range(kw):
            out += np.float32(k[y, x]) * ap[y:y + a.shape[0], x:x + a.shape[1]]
    return out


def gaussian_blur(a: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return a
    k = _gaussian_kernel(sigma)
    return _conv1d(_conv1d(a, k, 0), k, 1)


def gradients(g: np.ndarray, operator: str = "central") -> tuple[np.ndarray, np.ndarray]:
    """Return `(iy, ix)` gradients using a selectable local operator."""
    if operator == "central":
        iy, ix = np.gradient(g)
        return iy.astype(np.float32), ix.astype(np.float32)
    # _conv2d is a cross-correlation, so the +1 column must sit at +x for d/dx to match
    # np.gradient's sign convention (the tensor J is sign-invariant, but callers of this
    # function are not necessarily)
    if operator == "sobel":
        kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32) / 8.0
        ky = kx.T
    elif operator == "scharr":
        kx = np.array([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]], dtype=np.float32) / 32.0
        ky = kx.T
    else:
        raise ValueError(f"unknown gradient_operator {operator!r}; expected central, sobel, or scharr")
    ix = _conv2d(g, kx)
    iy = _conv2d(g, ky)
    return iy, ix


@dataclass
class StructureTensor:
    lam1: np.ndarray            # larger eigenvalue (across-edge energy)
    lam2: np.ndarray            # smaller eigenvalue (along-edge energy)
    across_edge_angle: np.ndarray  # radians; direction of the gradient (major eigvec of J)
    coherence: np.ndarray       # ((lam1-lam2)/(lam1+lam2))^2 in [0,1]
    energy: np.ndarray          # lam1 + lam2
    label: np.ndarray           # 0=flat, 1=edge, 2=corner (uint8)

    @property
    def along_edge_angle(self) -> np.ndarray:
        """Tangent direction; the axis an edge Gaussian should be elongated along."""
        return self.across_edge_angle + np.pi / 2.0


def compute(img: np.ndarray, cfg: StructureTensorConfig | None = None) -> StructureTensor:
    cfg = cfg or StructureTensorConfig()
    img = np.asarray(img, dtype=np.float32)
    if cfg.color_space == "rgb" and img.ndim == 3:
        # Di Zenzo multi-channel tensor: J = sum_c grad I_c grad I_c^T. Sees iso-luminant
        # (chroma-only) edges that the luma tensor misses.
        channels = [gaussian_blur(img[..., c], cfg.grad_sigma) for c in range(min(3, img.shape[-1]))]
    elif cfg.color_space in ("luma", "rgb"):
        channels = [gaussian_blur(to_luma(img), cfg.grad_sigma)]
    else:
        raise ValueError(f"unknown color_space {cfg.color_space!r}; expected luma or rgb")
    g = channels[0]

    Jxx = np.zeros(g.shape, dtype=np.float32)
    Jxy = np.zeros(g.shape, dtype=np.float32)
    Jyy = np.zeros(g.shape, dtype=np.float32)
    for ch in channels:
        iy, ix = gradients(ch, cfg.gradient_operator)
        Jxx += ix * ix
        Jxy += ix * iy
        Jyy += iy * iy
    Jxx = gaussian_blur(Jxx, cfg.tensor_sigma)
    Jxy = gaussian_blur(Jxy, cfg.tensor_sigma)
    Jyy = gaussian_blur(Jyy, cfg.tensor_sigma)

    half = 0.5 * (Jxx + Jyy)
    diff = 0.5 * (Jxx - Jyy)
    r = np.sqrt(diff * diff + Jxy * Jxy)
    lam1 = half + r
    lam2 = np.clip(half - r, 0.0, None)

    # orientation of the major eigenvector of J (the gradient / across-edge direction)
    angle = 0.5 * np.arctan2(2.0 * Jxy, Jxx - Jyy)

    energy = lam1 + lam2
    coherence = ((lam1 - lam2) / (energy + 1e-12)) ** 2

    ref = np.percentile(energy, 99.0) + 1e-12
    label = np.zeros(g.shape, dtype=np.uint8)
    is_flat = energy < cfg.flat_frac * ref
    is_corner = (lam2 > cfg.corner_frac * ref) & (~is_flat)
    label[~is_flat] = 1          # edge (default for structured, non-flat pixels)
    label[is_corner] = 2         # corner overrides
    label[is_flat] = 0

    return StructureTensor(
        lam1=lam1.astype(np.float32), lam2=lam2.astype(np.float32),
        across_edge_angle=angle.astype(np.float32), coherence=coherence.astype(np.float32),
        energy=energy.astype(np.float32), label=label,
    )
