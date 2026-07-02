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
    e = energy.astype(np.float64)
    ref = np.percentile(e, 99.0) + 1e-12
    d = np.clip(e / ref, 0.0, 1.0) ** power
    d = base + (1.0 - base) * d
    return (d / d.sum()).astype(np.float64)  # normalized pmf over pixels


def density_from_image(img: np.ndarray, icfg: InitConfig,
                       scfg: StructureTensorConfig | None = None) -> tuple[np.ndarray, st.StructureTensor]:
    tensor = st.compute(img, scfg)
    d = density_from_energy(tensor.energy, icfg.density_base, icfg.density_power)
    return d, tensor


def density_from_residual(residual: np.ndarray, base: float, power: float,
                          grad_sigma: float) -> np.ndarray:
    """Residual-driven density for pyramid level ell: chase leftover error (HIER-001)."""
    scfg = StructureTensorConfig(grad_sigma=grad_sigma)
    tensor = st.compute(np.abs(residual), scfg)
    return density_from_energy(tensor.energy, base, power)


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
