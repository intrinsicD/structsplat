"""Anisotropic, density-adaptive blue-noise sampling via Weighted Sample Elimination.

WSE (Yuksel, EGSR 2015) turns a dense candidate set into a blue-noise subset of exactly N
points. We extend it two ways (ADR-0005):
  * per-point target radius r_i  -> density adaptivity (dense near features, sparse in flats)
  * a per-point metric tensor M_i -> anisotropy: distance is measured Mahalanobis-style in a
    frame aligned to the local structure, so points pack tightly across edges and spread along
    them while retaining blue-noise spectra in the warped metric.

Grid-accelerated + lazy-deletion max-heap. Pure NumPy; runs once per image at init time.
"""
from __future__ import annotations
import heapq
import numpy as np


def anisotropy_metric(angle: np.ndarray, ratio: np.ndarray) -> np.ndarray:
    """Unit-area metric tensors M (K,2,2) whose unit ball is an ellipse.

    `angle` is the across-edge (gradient) direction. `ratio` >= 1 is major/minor axis ratio.
    We make across-edge spacing small and along-edge spacing large (dense across, sparse along)
    with equal ellipse area so counts stay comparable to the isotropic case.
    """
    ratio = np.maximum(ratio, 1.0)
    s_across = 1.0 / np.sqrt(ratio)   # small spacing across the edge
    s_along = np.sqrt(ratio)          # large spacing along the edge
    c, s = np.cos(angle), np.sin(angle)
    # eigenvector e_across = (c, s) (gradient dir); e_along = (-s, c)
    # M = sum_k (1/spacing_k^2) e_k e_k^T
    a_across = 1.0 / s_across ** 2
    a_along = 1.0 / s_along ** 2
    K = angle.shape[0]
    M = np.empty((K, 2, 2), dtype=np.float64)
    M[:, 0, 0] = a_across * c * c + a_along * s * s
    M[:, 0, 1] = a_across * c * s - a_along * s * c
    M[:, 1, 0] = M[:, 0, 1]
    M[:, 1, 1] = a_across * s * s + a_along * c * c
    return M


def _build_grid(points: np.ndarray, cell: float):
    mn = points.min(axis=0)
    gi = np.floor((points - mn) / cell).astype(np.int64)
    grid: dict[tuple[int, int], list[int]] = {}
    for idx, (gx, gy) in enumerate(gi):
        grid.setdefault((gx, gy), []).append(idx)
    return grid, gi, mn


def eliminate(points: np.ndarray, n: int, r_i: np.ndarray,
              metric: np.ndarray | None = None, alpha: float = 8.0) -> np.ndarray:
    """Return indices (n,) of a blue-noise subset of `points`.

    points : (M,2) candidate positions
    n      : target count (<= M)
    r_i    : (M,) per-point target radius (local desired spacing)
    metric : (M,2,2) unit-area metric per point, or None for Euclidean
    """
    M = points.shape[0]
    if n >= M:
        return np.arange(M)
    r_i = np.asarray(r_i, dtype=np.float64)
    cell = 2.0 * float(r_i.max())
    grid, gi, mn = _build_grid(points, cell)

    def neighbors(i: int):
        gx, gy = gi[i]
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                cellpts = grid.get((gx + dx, gy + dy))
                if cellpts:
                    yield from cellpts

    def pair_dist(i: int, j: int) -> float:
        dv = points[i] - points[j]
        if metric is None:
            return float(np.hypot(dv[0], dv[1]))
        Mm = 0.5 * (metric[i] + metric[j])
        d2 = dv @ Mm @ dv
        return float(np.sqrt(max(d2, 0.0)))

    def contrib(i: int, j: int) -> float:
        # weight that j contributes to i, using i's target radius (Yuksel adaptive form)
        two_r = 2.0 * r_i[i]
        d = pair_dist(i, j)
        if d >= two_r:
            return 0.0
        return (1.0 - d / two_r) ** alpha

    weights = np.zeros(M, dtype=np.float64)
    for i in range(M):
        w = 0.0
        for j in neighbors(i):
            if j != i:
                w += contrib(i, j)
        weights[i] = w

    version = np.zeros(M, dtype=np.int64)
    alive = np.ones(M, dtype=bool)
    heap = [(-weights[i], i, 0) for i in range(M)]
    heapq.heapify(heap)

    remaining = M
    while remaining > n:
        negw, i, ver = heapq.heappop(heap)
        if not alive[i] or ver != version[i]:
            continue  # stale entry
        # remove the most crowded surviving sample
        alive[i] = False
        remaining -= 1
        for j in neighbors(i):
            if j == i or not alive[j]:
                continue
            weights[j] -= contrib(j, i)
            version[j] += 1
            heapq.heappush(heap, (-weights[j], j, version[j]))

    return np.nonzero(alive)[0]
