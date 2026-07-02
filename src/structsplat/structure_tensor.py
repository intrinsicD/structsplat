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


def gaussian_blur(a: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return a
    k = _gaussian_kernel(sigma)
    return _conv1d(_conv1d(a, k, 0), k, 1)


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
    g = to_luma(img)
    g = gaussian_blur(g, cfg.grad_sigma)

    # central-difference gradients (axis 0 = y, axis 1 = x)
    iy, ix = np.gradient(g)
    Jxx = gaussian_blur(ix * ix, cfg.tensor_sigma)
    Jxy = gaussian_blur(ix * iy, cfg.tensor_sigma)
    Jyy = gaussian_blur(iy * iy, cfg.tensor_sigma)

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
