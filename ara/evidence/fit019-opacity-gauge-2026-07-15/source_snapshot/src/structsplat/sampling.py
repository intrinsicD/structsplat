"""Anisotropic, density-adaptive point sampling: WSE blue noise + alternative samplers.

WSE (Yuksel, EGSR 2015) turns a dense candidate set into a blue-noise subset of exactly N
points. We extend it two ways (ADR-0005):
  * per-point target radius r_i  -> density adaptivity (dense near features, sparse in flats)
  * a per-point metric tensor M_i -> anisotropy: distance is measured Mahalanobis-style in a
    frame aligned to the local structure, so points pack tightly across edges and spread along
    them while retaining blue-noise spectra in the warped metric.

Pair discovery and the initial crowding weights are fully vectorized over grid-cell offsets;
only the greedy heap loop stays in Python (it is inherently sequential but touches just each
removed point's neighbor list). Pure NumPy; runs once per image at init time.

The alternative samplers (`dart_throwing`, `farthest_point`, `cvt`, `halton_unit`) exist so the
stage ablation (ABL-002 / ADR-0010) can measure how much of the result is owed to WSE blue noise
specifically versus any reasonably even, density-adaptive point set. They share WSE's
(candidates, n, r_i, metric) interface where the algorithm permits.
"""
from __future__ import annotations
import heapq
import math
import warnings
import numpy as np


def _warn_if_not_reducing(n: int, M: int) -> None:
    """Warn when a sampler cannot enforce its exact-N contract because n > M.

    All samplers short-circuit to `arange(M)` when `n >= M`, returning fewer than the requested
    `n` points. A misconfigured `candidate_oversample` (or tiny image) hits this silently; the
    warning surfaces it instead of a mysterious under-count downstream (INIT-005).
    """
    if n > M:
        warnings.warn(
            f"requested n={n} exceeds candidate count M={M}; returning all {M} candidates, "
            "so the exact-N contract is not met (increase candidate_oversample or the budget).",
            stacklevel=3,
        )


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
    occupied, starts = np.unique(cid_sorted, return_index=True)
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
            pos = np.searchsorted(occupied, pc)
            in_range = pos < len(occupied)
            if not in_range.any():
                continue
            pc_in = pc[in_range]
            pos_in = pos[in_range]
            matched = occupied[pos_in] == pc_in
            if not matched.any():
                continue
            src = src[in_range][matched]
            ptr = starts[pos_in[matched]]
            cnt = counts[pos_in[matched]]
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


def _eliminate_partition(points: np.ndarray, n: int, r_i: np.ndarray,
                         metric: np.ndarray | None = None,
                         alpha: float = 8.0) -> tuple[np.ndarray, np.ndarray]:
    """One unchanged WSE pass: sorted survivors and chronological removals."""
    M = points.shape[0]
    target_n = min(max(int(n), 0), M)
    if M == 0:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty.copy()
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

    removed = []
    remaining = M
    while remaining > target_n:
        _negw, i, ver = pop(heap)
        if not alive[i] or ver != version[i]:
            continue  # stale entry
        # remove the most crowded surviving sample
        removed.append(i)
        alive[i] = False
        remaining -= 1
        for e in range(indptr[i], indptr[i + 1]):
            j = recv_of[e]
            if not alive[j]:
                continue
            weights[j] -= w_of[e]
            version[j] += 1
            push(heap, (-weights[j], j, version[j]))
    return np.nonzero(alive)[0], np.asarray(removed, dtype=np.int64)


def progressive_order(points: np.ndarray, r_i: np.ndarray,
                      metric: np.ndarray | None = None, alpha: float = 8.0) -> np.ndarray:
    """Return a progressive-WSE permutation of a fixed terminal sample set.

    This mirrors Yuksel's secondary recursive halving: the first half is re-eliminated with
    radii enlarged by sqrt(2) in 2D, recursively ordered, then followed by the eliminated shell
    in reverse removal order. The input set and every point attribute remain unchanged.
    """
    points = np.asarray(points, dtype=np.float64)
    r_i = np.asarray(r_i, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"points must have shape (N, 2), got {points.shape}")
    if r_i.shape != (len(points),):
        raise ValueError(f"r_i must have shape ({len(points)},), got {r_i.shape}")
    if metric is not None:
        metric = np.asarray(metric, dtype=np.float64)
        if metric.shape != (len(points), 2, 2):
            raise ValueError(
                f"metric must have shape ({len(points)}, 2, 2), got {metric.shape}")

    def recurse(active: np.ndarray, scale: float) -> np.ndarray:
        if len(active) < 3:
            return active.copy()
        local_metric = None if metric is None else metric[active]
        keep, removed = _eliminate_partition(
            points[active],
            len(active) // 2,
            r_i[active] * scale,
            metric=local_metric,
            alpha=alpha,
        )
        core = recurse(active[keep], scale * math.sqrt(2.0))
        shell = active[removed[::-1]]
        return np.concatenate([core, shell])

    active = np.arange(len(points), dtype=np.int64)
    return recurse(active, math.sqrt(2.0))


def eliminate(points: np.ndarray, n: int, r_i: np.ndarray,
              metric: np.ndarray | None = None, alpha: float = 8.0, *,
              progressive: bool = False) -> np.ndarray:
    """Return indices (n,) of a blue-noise subset of `points`.

    ``progressive=True`` only permutes the fixed terminal survivor set. False preserves the
    historical candidate-index ordering and remains the compatibility default.
    """
    M = points.shape[0]
    if n >= M:
        _warn_if_not_reducing(n, M)
        keep = np.arange(M, dtype=np.int64)
    else:
        keep, _removed = _eliminate_partition(points, n, r_i, metric=metric, alpha=alpha)
    if not progressive or len(keep) < 3:
        return keep
    order = progressive_order(
        np.asarray(points)[keep],
        np.asarray(r_i)[keep],
        None if metric is None else np.asarray(metric)[keep],
        alpha=alpha,
    )
    return keep[order]


def floyd_steinberg(density: np.ndarray, n: int) -> np.ndarray:
    """Density-map Floyd-Steinberg dithering -> exactly `n` pixel-center positions.

    The normalized density is interpreted as expected sample mass per pixel (`n * density`).
    Error diffusion turns that scalar field into a one-bit placement mask while a quota guard
    preserves the exact-N contract for all `n <= H*W`. This is the Instant-GI-style O(HW)
    placement control for ABL-004; it is deliberately NumPy-only and independent of torch.
    """
    H, W = density.shape
    n = int(n)
    if n <= 0:
        return np.empty((0, 2), dtype=np.float64)
    M = H * W
    if n >= M:
        _warn_if_not_reducing(n, M)
        yy, xx = np.mgrid[0:H, 0:W]
        return np.stack([xx.ravel(), yy.ravel()], axis=1).astype(np.float64)

    d = np.nan_to_num(np.maximum(density.astype(np.float64), 0.0), copy=False)
    total = float(d.sum())
    if total <= 0.0:
        d = np.full((H, W), 1.0 / M, dtype=np.float64)
    else:
        d = d / total
    work = d * n
    keep = np.zeros((H, W), dtype=bool)

    remaining = n
    visited = 0
    for y in range(H):
        left_to_right = (y % 2) == 0
        xs = range(W) if left_to_right else range(W - 1, -1, -1)
        for x in xs:
            slots_left = M - visited
            if remaining <= 0:
                q = 0
            elif remaining >= slots_left:
                q = 1
            else:
                q = 1 if work[y, x] >= 0.5 else 0
            keep[y, x] = bool(q)
            remaining -= q
            err = work[y, x] - q

            if left_to_right:
                if x + 1 < W:
                    work[y, x + 1] += err * 7.0 / 16.0
                if y + 1 < H:
                    if x > 0:
                        work[y + 1, x - 1] += err * 3.0 / 16.0
                    work[y + 1, x] += err * 5.0 / 16.0
                    if x + 1 < W:
                        work[y + 1, x + 1] += err * 1.0 / 16.0
            else:
                if x > 0:
                    work[y, x - 1] += err * 7.0 / 16.0
                if y + 1 < H:
                    if x + 1 < W:
                        work[y + 1, x + 1] += err * 3.0 / 16.0
                    work[y + 1, x] += err * 5.0 / 16.0
                    if x > 0:
                        work[y + 1, x - 1] += err * 1.0 / 16.0
            visited += 1

    ys, xs = np.nonzero(keep)
    return np.stack([xs, ys], axis=1).astype(np.float64)


def _metric_components(metric: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return metric[:, 0, 0], metric[:, 0, 1], metric[:, 1, 1]


def _pair_d2(
    points: np.ndarray,
    j: int,
    metric: np.ndarray | None,
    metric_components: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> np.ndarray:
    """Squared distance from every point to point j (Euclidean or averaged-metric Mahalanobis)."""
    dv = points - points[j]
    if metric is None:
        return dv[:, 0] ** 2 + dv[:, 1] ** 2
    m00, m01, m11 = metric_components or _metric_components(metric)
    dx = dv[:, 0]
    dy = dv[:, 1]
    return (0.5 * (m00 + m00[j]) * dx ** 2
            + (m01 + m01[j]) * dx * dy
            + 0.5 * (m11 + m11[j]) * dy ** 2)


def dart_throwing(points: np.ndarray, n: int, r_i: np.ndarray,
                  metric: np.ndarray | None = None, rng: np.random.Generator | None = None,
                  beta: float = 0.75) -> np.ndarray:
    """Variable-radius Poisson-disk (sequential dart throwing) subset of exactly `n` points.

    Candidates are visited in a random order; one is accepted iff its (metric) distance to
    every already-accepted point is at least beta * (r_cand + r_accepted). If the disks are
    too greedy to reach `n`, the shortfall is filled farthest-first from the rejects, so the
    exact-N contract of `eliminate` is preserved. Classic Poisson disk, unlike WSE, never
    revisits a decision — the ablation uses it to price WSE's global elimination.
    """
    M = points.shape[0]
    if n >= M:
        _warn_if_not_reducing(n, M)
        return np.arange(M)
    rng = rng or np.random.default_rng(0)
    points = np.asarray(points, dtype=np.float64)
    r_i = np.asarray(r_i, dtype=np.float64)
    metric_components = _metric_components(metric) if metric is not None else None

    # Euclidean inflation bound, same reasoning as _neighbor_pairs: d_E <= d_M / sqrt(lam_min).
    infl = 1.0
    if metric is not None:
        infl = 1.0 / np.sqrt(max(float(_metric_min_eigenvalue(metric).min()), 1e-12))
    cell = max(float(np.median(2.0 * beta * r_i)), 1e-6)
    mn = points.min(axis=0)
    gxy = np.floor((points - mn) / cell).astype(np.int64)

    order = rng.permutation(M)
    grid: dict[tuple[int, int], list[int]] = {}
    occupied_cells: list[tuple[int, int]] = []
    cell_r_max: dict[tuple[int, int], float] = {}
    accepted: list[int] = []
    taken = np.zeros(M, dtype=bool)
    r_acc_max = 0.0
    for i in order:
        reach = beta * (r_i[i] + r_acc_max) * infl
        # floor+1, not ceil: cell indices are floors, so a distance of exactly k*cell can
        # still straddle k+1 cell boundaries
        k = int(np.floor(reach / cell)) + 1
        gx, gy = int(gxy[i, 0]), int(gxy[i, 1])
        neigh = []
        if (2 * k + 1) ** 2 > 4 * len(occupied_cells):
            candidate_cells = occupied_cells
        else:
            candidate_cells = (
                (gx + dx, gy + dy)
                for dy in range(-k, k + 1)
                for dx in range(-k, k + 1)
            )
        for key in candidate_cells:
            bucket = grid.get(key)
            if not bucket:
                continue
            dx = key[0] - gx
            dy = key[1] - gy
            ring = cell * np.hypot(max(abs(dx) - 1, 0), max(abs(dy) - 1, 0))
            if ring > beta * (float(r_i[i]) + cell_r_max[key]) * infl:
                continue
            neigh += bucket
        if neigh:
            nb = np.asarray(neigh, dtype=np.int64)
            dv = points[nb] - points[i]
            if metric is None:
                d2 = dv[:, 0] ** 2 + dv[:, 1] ** 2
            else:
                Mm = 0.5 * (metric[nb] + metric[i])
                d2 = (Mm[:, 0, 0] * dv[:, 0] ** 2 + 2.0 * Mm[:, 0, 1] * dv[:, 0] * dv[:, 1]
                      + Mm[:, 1, 1] * dv[:, 1] ** 2)
            lim = beta * (r_i[nb] + r_i[i])
            if bool((d2 < lim * lim).any()):
                continue
        accepted.append(int(i))
        taken[i] = True
        key = (gx, gy)
        bucket = grid.setdefault(key, [])
        if not bucket:
            occupied_cells.append(key)
            cell_r_max[key] = float(r_i[i])
        else:
            cell_r_max[key] = max(cell_r_max[key], float(r_i[i]))
        bucket.append(int(i))
        r_acc_max = max(r_acc_max, float(r_i[i]))
        if len(accepted) == n:
            break

    if len(accepted) < n:  # disks too greedy: fill farthest-first from the rejects
        rest = np.nonzero(~taken)[0]

        def norm_d2_to(j: int) -> np.ndarray:
            # metric (Mahalanobis) squared distance from every reject to point j, radius-normalized
            # — same distance semantics the acceptance test uses, so the fill does not silently mix
            # Euclidean and metric distances inside one ablation arm (INIT-005).
            d2 = _pair_d2(points, j, metric, metric_components)[rest]
            return d2 / np.maximum(r_i[rest] + r_i[j], 1e-12) ** 2

        dmin = np.full(rest.shape[0], np.inf)
        for j in accepted:
            np.minimum(dmin, norm_d2_to(j), out=dmin)
        while len(accepted) < n:
            b = int(np.argmax(dmin))
            j = int(rest[b])
            accepted.append(j)
            dmin[b] = -np.inf
            np.minimum(dmin, norm_d2_to(j), out=dmin)
    return np.asarray(sorted(accepted), dtype=np.int64)


def farthest_point(points: np.ndarray, n: int, r_i: np.ndarray | None = None,
                   metric: np.ndarray | None = None,
                   rng: np.random.Generator | None = None) -> np.ndarray:
    """Greedy maximin (farthest-point) subset of exactly `n` points. O(M*n).

    Distance is radius-normalized (d / (r_i + r_j)) when `r_i` is given, which makes the
    maximin objective density-adaptive, and Mahalanobis in the averaged metric when `metric`
    is given. Deterministic given the candidate set and rng. Strong uniformity, no blue-noise
    spectrum — the ablation's 'even but not blue' probe.
    """
    M = points.shape[0]
    if n >= M:
        _warn_if_not_reducing(n, M)
        return np.arange(M)
    rng = rng or np.random.default_rng(0)
    points = np.asarray(points, dtype=np.float64)
    metric_components = _metric_components(metric) if metric is not None else None

    def dist2_to(j: int) -> np.ndarray:
        d2 = _pair_d2(points, j, metric, metric_components)
        if r_i is not None:
            d2 = d2 / np.maximum(r_i + r_i[j], 1e-12) ** 2
        return d2

    sel = np.empty(n, dtype=np.int64)
    sel[0] = int(rng.integers(M))
    dmin = dist2_to(int(sel[0]))
    for t in range(1, n):
        j = int(np.argmax(dmin))
        sel[t] = j
        np.minimum(dmin, dist2_to(j), out=dmin)
    return np.sort(sel)


def cvt(points: np.ndarray, n: int, iters: int = 8,
        rng: np.random.Generator | None = None) -> np.ndarray:
    """Lloyd/k-means relaxation of a density-drawn candidate set -> ~centroidal Voronoi points.

    Because the candidates are distributed like the density, the k-means centroids approximate
    a density-adaptive CVT. Returns (n,2) *positions* (centroids move off the candidate set).
    Brute-force chunked assignment: fine at screening budgets (n <= ~5k), slow beyond — the
    reference-code caveat in CLAUDE.md applies.
    """
    M = points.shape[0]
    points = np.asarray(points, dtype=np.float64)
    if n >= M:
        _warn_if_not_reducing(n, M)
        return points.copy()
    rng = rng or np.random.default_rng(0)
    centers = points[rng.choice(M, size=n, replace=False)].copy()
    chunk = max(1, int(4e6) // max(n, 1))
    for _ in range(iters):
        assign = np.empty(M, dtype=np.int64)
        for s in range(0, M, chunk):
            block = points[s:s + chunk]
            d2 = ((block[:, None, :] - centers[None, :, :]) ** 2).sum(-1)
            assign[s:s + chunk] = np.argmin(d2, axis=1)
        sums = np.zeros((n, 2))
        np.add.at(sums, assign, points)
        counts = np.bincount(assign, minlength=n).astype(np.float64)
        empty = counts == 0
        counts[empty] = 1.0
        centers = sums / counts[:, None]
        if empty.any():  # reseed dead centers so the count contract holds
            centers[empty] = points[rng.choice(M, size=int(empty.sum()), replace=False)]
    return centers


def _van_der_corput(i: np.ndarray, base: int) -> np.ndarray:
    v = np.zeros(i.shape, dtype=np.float64)
    denom = np.ones(i.shape, dtype=np.float64)
    ii = i.astype(np.int64).copy()
    while np.any(ii > 0):
        denom *= base
        v += (ii % base) / denom
        ii //= base
    return v


def halton_unit(n: int, rng: np.random.Generator | None = None) -> np.ndarray:
    """Halton (2,3) low-discrepancy points in the unit square with a Cranley-Patterson rotation.

    The random toroidal shift keeps the low-discrepancy structure while making the set
    seed-dependent (reproducibility invariant: the shift comes from the caller's rng).
    """
    i = np.arange(n, dtype=np.int64)
    u = np.stack([_van_der_corput(i, 2), _van_der_corput(i, 3)], axis=1)
    shift = (rng or np.random.default_rng(0)).random(2)
    return (u + shift) % 1.0
