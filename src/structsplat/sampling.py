"""Anisotropic, density-adaptive blue-noise sampling via Weighted Sample Elimination.

WSE (Yuksel, EGSR 2015) turns a dense candidate set into a blue-noise subset of exactly N
points. We extend it two ways (ADR-0005):
  * per-point target radius r_i  -> density adaptivity (dense near features, sparse in flats)
  * a per-point metric tensor M_i -> anisotropy: distance is measured Mahalanobis-style in a
    frame aligned to the local structure, so points pack tightly across edges and spread along
    them while retaining blue-noise spectra in the warped metric.

Pair discovery and the initial crowding weights are fully vectorized over grid-cell offsets;
only the greedy heap loop stays in Python (it is inherently sequential but touches just each
removed point's neighbor list). Pure NumPy; runs once per image at init time.
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


def _metric_min_eigenvalue(metric: np.ndarray) -> np.ndarray:
    """Smaller eigenvalue of each symmetric 2x2 metric (vectorized)."""
    half = 0.5 * (metric[:, 0, 0] + metric[:, 1, 1])
    diff = 0.5 * (metric[:, 0, 0] - metric[:, 1, 1])
    r = np.sqrt(diff * diff + metric[:, 0, 1] ** 2)
    return half - r


def _neighbor_pairs(points: np.ndarray, r_i: np.ndarray, metric: np.ndarray | None,
                    alpha: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """All ordered pairs (recv, ctrb) with nonzero WSE weight, and their contributions.

    A pair contributes iff pair_dist(recv, ctrb) < 2*r_i[recv], where pair_dist is Euclidean
    or Mahalanobis in the averaged metric. The Mahalanobis distance can undershoot the
    Euclidean one by up to sqrt(lam_min) of the averaged pair metric, and
    lam_min((M_i+M_j)/2) >= lam_min(M_i)/2, so a per-RECEIVER Euclidean reach of
    2*r_i*sqrt(2/lam_min_i) is a valid bound for any contributor. Far grid-offset rings are
    then searched only by the (few, high-anisotropy) receivers whose reach extends that far.
    """
    M = points.shape[0]
    if metric is None:
        reach = 2.0 * r_i
    else:
        lam_min = np.maximum(_metric_min_eigenvalue(metric), 1e-12)
        # two valid bounds; take the tighter per point
        reach = 2.0 * r_i * np.minimum(np.sqrt(2.0 / lam_min), 1.0 / np.sqrt(lam_min.min()))
    reach_max = float(reach.max())

    # grid: cells no coarser than needed; bounded offset range keeps each pass vectorized
    cell = max(float(np.median(2.0 * r_i)), reach_max / 16.0)
    mn = points.min(axis=0)
    gxy = np.floor((points - mn) / cell).astype(np.int64)
    ncx = int(gxy[:, 0].max()) + 1
    ncy = int(gxy[:, 1].max()) + 1
    cid = gxy[:, 1] * ncx + gxy[:, 0]
    order = np.argsort(cid, kind="stable")
    cid_sorted = cid[order]
    starts = np.searchsorted(cid_sorted, np.arange(ncx * ncy, dtype=np.int64))
    counts = np.diff(np.append(starts, M))
    k = int(np.ceil(reach_max / cell))

    idx = np.arange(M, dtype=np.int64)
    recv_all, ctrb_all, w_all = [], [], []
    for dy in range(-k, k + 1):
        for dx in range(-k, k + 1):
            # closest possible distance between points in cells this offset apart
            ring = cell * np.hypot(max(abs(dx) - 1, 0), max(abs(dy) - 1, 0))
            near = reach >= ring
            if not near.any():
                continue
            src0 = idx[near]
            gx = gxy[near, 0] + dx
            gy = gxy[near, 1] + dy
            ok = (gx >= 0) & (gx < ncx) & (gy >= 0) & (gy < ncy)
            if not ok.any():
                continue
            src = src0[ok]
            pc = gy[ok] * ncx + gx[ok]
            cnt = counts[pc]
            nz = cnt > 0
            if not nz.any():
                continue
            src, ptr, cnt = src[nz], starts[pc[nz]], cnt[nz]
            recv = np.repeat(src, cnt)
            ends = np.cumsum(cnt)
            within = np.arange(int(ends[-1])) - np.repeat(ends - cnt, cnt)
            ctrb = order[np.repeat(ptr, cnt) + within]

            dv = points[recv] - points[ctrb]
            if metric is None:
                d2 = dv[:, 0] ** 2 + dv[:, 1] ** 2
            else:
                Mm = 0.5 * (metric[recv] + metric[ctrb])
                d2 = (Mm[:, 0, 0] * dv[:, 0] ** 2
                      + 2.0 * Mm[:, 0, 1] * dv[:, 0] * dv[:, 1]
                      + Mm[:, 1, 1] * dv[:, 1] ** 2)
            two_r = 2.0 * r_i[recv]
            keep = (d2 < two_r * two_r) & (recv != ctrb)
            if not keep.any():
                continue
            d = np.sqrt(np.maximum(d2[keep], 0.0))
            recv_all.append(recv[keep])
            ctrb_all.append(ctrb[keep])
            w_all.append((1.0 - d / two_r[keep]) ** alpha)

    if not recv_all:
        e = np.empty(0, dtype=np.int64)
        return e, e.copy(), np.empty(0, dtype=np.float64)
    return np.concatenate(recv_all), np.concatenate(ctrb_all), np.concatenate(w_all)


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
    points = np.asarray(points, dtype=np.float64)
    r_i = np.asarray(r_i, dtype=np.float64)

    recv, ctrb, w = _neighbor_pairs(points, r_i, metric, alpha)
    weights = np.bincount(recv, weights=w, minlength=M).tolist()

    # CSR keyed by CONTRIBUTOR: removing x must decrement every receiver that x crowds.
    # Plain Python lists: the greedy loop below iterates element-wise and NumPy scalar
    # boxing would dominate it.
    by_ctrb = np.argsort(ctrb, kind="stable")
    recv_of = recv[by_ctrb].tolist()
    w_of = w[by_ctrb].tolist()
    indptr = np.searchsorted(ctrb[by_ctrb], np.arange(M + 1, dtype=np.int64)).tolist()

    version = [0] * M
    alive = [True] * M
    heap = [(-weights[i], i, 0) for i in range(M)]
    heapq.heapify(heap)
    push, pop = heapq.heappush, heapq.heappop

    remaining = M
    while remaining > n:
        negw, i, ver = pop(heap)
        if not alive[i] or ver != version[i]:
            continue  # stale entry
        # remove the most crowded surviving sample
        alive[i] = False
        remaining -= 1
        for e in range(indptr[i], indptr[i + 1]):
            j = recv_of[e]
            if not alive[j]:
                continue
            weights[j] -= w_of[e]
            version[j] += 1
            push(heap, (-weights[j], j, version[j]))

    return np.nonzero(alive)[0]
