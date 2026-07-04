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


def density_from_energy(
    energy: np.ndarray,
    base: float,
    power: float,
    ref: float | None = None,
) -> np.ndarray:
    e = np.maximum(energy.astype(np.float64), 0.0)
    ref = st.energy_reference(e) if ref is None else float(ref)
    d = np.clip(e / ref, 0.0, 1.0) ** power
    d = base + (1.0 - base) * d
    s = d.sum()
    if not s > 0.0:  # identically-zero feature (flat image / zero residual) with base=0
        return np.full(d.shape, 1.0 / d.size, dtype=np.float64)
    return (d / s).astype(np.float64)  # normalized pmf over pixels


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
        # Return a RAW combined feature so the single base/power/normalize in
        # density_from_energy applies exactly once. Previously this mixed two already-normalized
        # pmfs and re-passed the mixture through density_from_energy, double-normalizing so
        # density_power/density_base meant something different for this mode (INIT-005). Each
        # feature is scaled to a comparable [0,1] range by its own reference before averaging.
        e = np.maximum(tensor.energy.astype(np.float64), 0.0)
        v = local_variance(img, scfg.tensor_sigma).astype(np.float64)
        eref = getattr(tensor, "energy_ref", None)
        eref = st.energy_reference(e) if eref is None else float(eref)
        a = np.clip(e / eref, 0.0, 1.0)
        b = np.clip(v / st.energy_reference(v), 0.0, 1.0)
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
    ref = getattr(tensor, "energy_ref", None) if icfg.density_mode == "structure" else None
    return density_from_energy(feature, icfg.density_base, icfg.density_power, ref)


def density_from_image(img: np.ndarray, icfg: InitConfig,
                       scfg: StructureTensorConfig | None = None) -> tuple[np.ndarray, st.StructureTensor]:
    tensor = st.compute(img, scfg)
    d = density_from_tensor_and_image(img, tensor, icfg, scfg)
    return d, tensor


def density_from_residual(residual: np.ndarray, base: float, power: float,
                          scfg: StructureTensorConfig | None = None,
                          mode: str = "structure") -> np.ndarray:
    """Residual-driven density for pyramid level ell: chase leftover error (HIER-001).

    Takes the full StructureTensorConfig (not just grad_sigma) so the residual density honors
    the same tensor operator / sigma / thresholds as the orientation tensor. Prefer computing
    the tensor once and calling density_from_tensor_and_image when you also need orientations.
    """
    scfg = scfg or StructureTensorConfig()
    resid = np.abs(residual)
    tensor = st.compute(resid, scfg)
    feature = feature_for_mode(resid, tensor, mode, scfg)
    ref = getattr(tensor, "energy_ref", None) if mode == "structure" else None
    return density_from_energy(feature, base, power, ref)


def warp_unit_points(u: np.ndarray, density: np.ndarray, chunk: int = 8192) -> np.ndarray:
    """Map unit-square points through the separable inverse CDF of `density`.

    y comes from the row-marginal CDF, x from the conditional CDF of the selected row, and the
    remainder places the point uniformly inside the pixel cell (centers at integer coords).
    Preserves the low-discrepancy structure of `u` (e.g. Halton) while matching the density,
    which i.i.d. `sample_candidates` cannot do. Returns (n, 2) as (x, y) float.
    """
    H, W = density.shape
    d = density.astype(np.float64) + 1e-15  # strictly positive => invertible CDF
    row_mass = d.sum(axis=1)
    cy = np.cumsum(row_mass)
    ty = np.clip(u[:, 1], 0.0, 1.0 - 1e-12) * cy[-1]
    iy = np.clip(np.searchsorted(cy, ty, side="right"), 0, H - 1)
    fy = (ty - np.where(iy > 0, cy[np.maximum(iy - 1, 0)], 0.0)) / row_mass[iy]

    n = u.shape[0]
    ix = np.empty(n, dtype=np.int64)
    fx = np.empty(n, dtype=np.float64)
    row_cum = np.cumsum(d, axis=1)
    tx_all = np.clip(u[:, 0], 0.0, 1.0 - 1e-12)
    for s in range(0, n, chunk):  # (chunk, W) slices bound memory
        rows = row_cum[iy[s:s + chunk]]
        tx = tx_all[s:s + chunk] * rows[:, -1]
        j = np.clip((rows < tx[:, None]).sum(axis=1), 0, W - 1)
        prev = np.where(j > 0, np.take_along_axis(rows, np.maximum(j - 1, 0)[:, None], 1)[:, 0], 0.0)
        ix[s:s + chunk] = j
        fx[s:s + chunk] = (tx - prev) / d[iy[s:s + chunk], j]
    xs = np.clip(ix - 0.5 + np.clip(fx, 0.0, 1.0), 0.0, W - 1.0)
    ys = np.clip(iy - 0.5 + np.clip(fy, 0.0, 1.0), 0.0, H - 1.0)
    return np.stack([xs, ys], axis=1)


def sample_candidates(density: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Draw `n` sub-pixel candidate positions ~ density. Returns (n, 2) as (x, y) float."""
    H, W = density.shape
    flat = density.ravel()
    flat = flat / flat.sum()
    idx = rng.choice(flat.size, size=n, replace=True, p=flat)
    ys, xs = np.divmod(idx, W)
    # integer coords are pixel centers (renderer convention): jitter within the footprint.
    # Fold (reflect) out-of-range half-pixel jitter back into [0, W-1] instead of clipping it:
    # clipping piled probability atoms at x=0 / x=W-1 (and coincident candidates there); folding
    # keeps the sub-pixel density continuous at the border (INIT-005).
    jitter = rng.random((n, 2)) - 0.5
    xs = _reflect(xs + jitter[:, 0], W - 1.0)
    ys = _reflect(ys + jitter[:, 1], H - 1.0)
    return np.stack([xs, ys], axis=1).astype(np.float64)


def _reflect(v: np.ndarray, hi: float) -> np.ndarray:
    """Reflect values into [0, hi] at both boundaries (no probability atoms at the edges)."""
    v = np.abs(v)               # reflect at 0
    return hi - np.abs(hi - v)  # reflect at hi
