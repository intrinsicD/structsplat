"""Density field and candidate generation for feature-aware sampling.

Density is derived from the structure-tensor energy (optionally the *residual* energy for
finer pyramid levels). It controls where Gaussians concentrate: high near edges/corners, a
non-zero floor everywhere so flat regions are still covered (ADR-0004, INIT-002).

Pure NumPy.
"""
from __future__ import annotations
import numpy as np

from . import structure_tensor as st
from .config import InitConfig, StructureTensorConfig


def density_from_energy(energy: np.ndarray, base: float, power: float) -> np.ndarray:
    e = np.maximum(energy.astype(np.float64), 0.0)
    ref = np.percentile(e, 99.0) + 1e-12
    d = np.clip(e / ref, 0.0, 1.0) ** power
    d = base + (1.0 - base) * d
    return (d / d.sum()).astype(np.float64)  # normalized pmf over pixels


def local_variance(img: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    g = st.to_luma(img)
    mu = st.gaussian_blur(g, sigma)
    mu2 = st.gaussian_blur(g * g, sigma)
    return np.maximum(mu2 - mu * mu, 0.0).astype(np.float32)


def feature_for_mode(img: np.ndarray, tensor: st.StructureTensor,
                     mode: str, scfg: StructureTensorConfig | None = None) -> np.ndarray:
    scfg = scfg or StructureTensorConfig()
    if mode == "structure":
        return tensor.energy
    if mode == "gradient":
        return np.sqrt(np.maximum(tensor.energy, 0.0)).astype(np.float32)
    if mode == "variance":
        return local_variance(img, scfg.tensor_sigma)
    if mode == "hybrid":
        a = density_from_energy(tensor.energy, 0.0, 1.0)
        b = density_from_energy(local_variance(img, scfg.tensor_sigma), 0.0, 1.0)
        return (0.5 * a + 0.5 * b).astype(np.float32)
    if mode == "uniform":
        return np.ones(tensor.energy.shape, dtype=np.float32)
    raise ValueError(
        f"unknown density_mode {mode!r}; expected structure, gradient, variance, hybrid, or uniform"
    )


def density_from_tensor_and_image(img: np.ndarray, tensor: st.StructureTensor,
                                  icfg: InitConfig,
                                  scfg: StructureTensorConfig | None = None) -> np.ndarray:
    feature = feature_for_mode(img, tensor, icfg.density_mode, scfg)
    return density_from_energy(feature, icfg.density_base, icfg.density_power)


def density_from_image(img: np.ndarray, icfg: InitConfig,
                       scfg: StructureTensorConfig | None = None) -> tuple[np.ndarray, st.StructureTensor]:
    tensor = st.compute(img, scfg)
    d = density_from_tensor_and_image(img, tensor, icfg, scfg)
    return d, tensor


def density_from_residual(residual: np.ndarray, base: float, power: float,
                          grad_sigma: float, mode: str = "structure") -> np.ndarray:
    """Residual-driven density for pyramid level ell: chase leftover error (HIER-001)."""
    scfg = StructureTensorConfig(grad_sigma=grad_sigma)
    tensor = st.compute(np.abs(residual), scfg)
    feature = feature_for_mode(np.abs(residual), tensor, mode, scfg)
    return density_from_energy(feature, base, power)


def sample_candidates(density: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Draw `n` sub-pixel candidate positions ~ density. Returns (n, 2) as (x, y) float."""
    H, W = density.shape
    flat = density.ravel()
    flat = flat / flat.sum()
    idx = rng.choice(flat.size, size=n, replace=True, p=flat)
    ys, xs = np.divmod(idx, W)
    jitter = rng.random((n, 2))
    xs = xs + jitter[:, 0]
    ys = ys + jitter[:, 1]
    return np.stack([xs, ys], axis=1).astype(np.float64)
