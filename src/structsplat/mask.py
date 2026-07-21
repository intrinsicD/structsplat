"""Binary-mask geometry for mask-contained fitting (CORE-010).

Exact Euclidean distance transform, signed distance, erosion, nearest-inside feature transform,
and boundary color dilation for fitting alpha-masked images so the Gaussian field stays inside the
mask. Pure NumPy and importable without torch (core invariant 1): the torch-side containment loop
in ``fit.py`` consumes the :class:`MaskGeometry` this module precomputes.

The EDT is the separable two-pass lower-envelope algorithm (Felzenszwalb & Huttenlocher 2012),
extended to carry the argmin so the same two passes yield an exact feature transform (the nearest
seed pixel of every pixel). scipy is intentionally not a dependency.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

# Effective lower bound on RS scales in the fitter (fit.py clamps log-scales to log(0.35)). The
# containment cap is never asked to go below this, and the projection erosion reserves room for it.
MIN_SCALE = 0.35


def _edt_1d(f: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """1-D squared distance transform of sampled function ``f`` with argmin.

    Returns ``(d, arg)`` where ``d[i] = min_j (i - j)**2 + f[j]`` and ``arg[i]`` is a ``j``
    achieving it. ``f`` may contain ``+inf`` (a location that is not a seed); a line with no finite
    entry returns all-``inf`` distances and ``-1`` args.
    """
    n = f.shape[0]
    d = np.full(n, np.inf, dtype=np.float64)
    arg = np.full(n, -1, dtype=np.int64)
    finite = np.nonzero(np.isfinite(f))[0]
    if finite.size == 0:
        return d, arg
    v = np.empty(finite.size, dtype=np.int64)   # locations of the lower-envelope parabolas
    z = np.empty(finite.size + 1, dtype=np.float64)  # boundaries between them
    v[0] = finite[0]
    z[0] = -np.inf
    z[1] = np.inf
    k = 0
    for q in finite[1:].tolist():
        fq = f[q] + q * q
        s = (fq - (f[v[k]] + v[k] * v[k])) / (2.0 * q - 2.0 * v[k])
        # z[0] = -inf guarantees this terminates at k = 0 (s is finite).
        while s <= z[k]:
            k -= 1
            s = (fq - (f[v[k]] + v[k] * v[k])) / (2.0 * q - 2.0 * v[k])
        k += 1
        v[k] = q
        z[k] = s
        z[k + 1] = np.inf
    idx = np.arange(n)
    # Region of parabola m is [z[m], z[m+1]); pick the largest m with z[m] <= i.
    m = np.searchsorted(z[: k + 1], idx, side="right") - 1
    chosen = v[m]
    d = (idx - chosen) ** 2 + f[chosen]
    arg = chosen.astype(np.int64)
    return d, arg


@dataclass
class _EDT:
    d2: np.ndarray          # (H, W) squared distance to the nearest seed
    fy: np.ndarray          # (H, W) row of the nearest seed
    fx: np.ndarray          # (H, W) column of the nearest seed


def _edt_2d(seed: np.ndarray) -> _EDT:
    """Exact squared EDT + feature transform. ``seed`` True marks distance-zero sites."""
    seed = np.asarray(seed, dtype=bool)
    H, W = seed.shape
    f = np.where(seed, 0.0, np.inf)
    g = np.empty((H, W), dtype=np.float64)
    ny = np.empty((H, W), dtype=np.int64)
    for x in range(W):  # vertical pass (per column)
        col_d, col_arg = _edt_1d(f[:, x])
        g[:, x] = col_d
        ny[:, x] = col_arg
    D = np.empty((H, W), dtype=np.float64)
    nx = np.empty((H, W), dtype=np.int64)
    for y in range(H):  # horizontal pass (per row) over the column result
        row_d, row_arg = _edt_1d(g[y, :])
        D[y, :] = row_d
        nx[y, :] = row_arg
    rows = np.arange(H)[:, None]
    fx = nx
    fy = ny[rows, nx]  # nearest-seed row lives in the column selected by the horizontal pass
    return _EDT(D, fy, fx)


def as_bool_mask(mask: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Coerce an array to a boolean (H, W) mask. RGBA uses alpha; RGB uses the channel mean."""
    a = np.asarray(mask)
    if a.ndim == 3:
        a = a[..., 3] if a.shape[2] == 4 else a.mean(axis=2)
    elif a.ndim != 2:
        raise ValueError(f"mask must be 2-D or 3-D, got shape {a.shape}")
    a = a.astype(np.float64)
    if a.size and a.max() > 1.0:  # accept 0..255 masks too
        a = a / 255.0
    return a > threshold


def squared_edt(seed: np.ndarray) -> np.ndarray:
    """Squared Euclidean distance to the nearest True pixel of ``seed`` (0 at seeds)."""
    return _edt_2d(seed).d2


def signed_distance(inside: np.ndarray) -> np.ndarray:
    """Signed distance in pixels: +distance-to-nearest-outside inside, -distance-to-inside out.

    Empty complements are clipped to the image diagonal so an all-inside or all-outside mask yields
    a large finite magnitude rather than an infinity.
    """
    inside = np.asarray(inside, dtype=bool)
    H, W = inside.shape
    maxd = float(np.hypot(H, W))
    d_out = np.sqrt(squared_edt(~inside))
    d_in = np.sqrt(squared_edt(inside))
    d_out = np.minimum(d_out, maxd)
    d_in = np.minimum(d_in, maxd)
    return np.where(inside, d_out, -d_in)


def erode(inside: np.ndarray, radius: float) -> np.ndarray:
    """Morphological erosion: keep pixels at least ``radius`` px from the nearest outside pixel."""
    inside = np.asarray(inside, dtype=bool)
    if radius <= 0.0:
        return inside.copy()
    d_out = np.sqrt(squared_edt(~inside))
    return d_out >= float(radius)


def nearest_inside_index(region: np.ndarray) -> np.ndarray:
    """(H, W) flat index of the nearest True pixel of ``region`` for every pixel (-1 if empty)."""
    region = np.asarray(region, dtype=bool)
    H, W = region.shape
    if not region.any():
        return np.full(H * W, -1, dtype=np.int64)
    edt = _edt_2d(region)
    return (edt.fy * W + edt.fx).reshape(-1).astype(np.int64)


def color_dilate(img: np.ndarray, inside: np.ndarray) -> np.ndarray:
    """Push boundary colors outward: fill outside pixels with the nearest inside pixel's color.

    Removes matte contamination from init colors sampled near the boundary while leaving the inside
    image untouched. Returns ``img`` unchanged if the mask is empty.
    """
    img = np.asarray(img, dtype=np.float32)
    inside = np.asarray(inside, dtype=bool)
    if not inside.any():
        return img.copy()
    edt = _edt_2d(inside)
    out = img.copy()
    out[~inside] = img[edt.fy[~inside], edt.fx[~inside]]
    return out


@dataclass
class MaskGeometry:
    """Precomputed mask geometry consumed by the containment loop.

    All arrays share the mask's (H, W). ``sdf`` is the signed distance used for the scale cap;
    ``eroded`` is the projection-admissible interior; ``nearest_inside_flat`` maps every pixel to
    the flat index of the nearest ``eroded`` pixel (for projecting drifted means); ``outside_flat``
    indexes the pixels outside the mask (for the coverage penalty and loss weighting).
    """

    inside: np.ndarray            # (H, W) bool
    sdf: np.ndarray               # (H, W) float, signed distance (+inside)
    eroded: np.ndarray            # (H, W) bool, projection interior
    nearest_inside_flat: np.ndarray   # (H*W,) int, nearest eroded-pixel flat index
    outside_flat: np.ndarray      # (K,) int, flat indices outside the mask
    erode_radius: float

    @property
    def shape(self) -> tuple[int, int]:
        return self.inside.shape

    @classmethod
    def build(cls, mask: np.ndarray, erode_radius: float,
              threshold: float = 0.5) -> "MaskGeometry":
        inside = as_bool_mask(mask, threshold)
        if not inside.any():
            raise ValueError("mask is empty (no pixel inside); cannot build mask geometry")
        H, W = inside.shape
        sdf = signed_distance(inside)
        eroded = erode(inside, erode_radius)
        region = eroded if eroded.any() else inside  # mask thinner than the erosion radius
        nearest = nearest_inside_index(region)
        outside_flat = np.nonzero(~inside.reshape(-1))[0].astype(np.int64)
        return cls(inside, sdf.astype(np.float32), eroded, nearest, outside_flat,
                   float(erode_radius))
