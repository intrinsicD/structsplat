import numpy as np
import pytest
from structsplat import sampling as sa


def test_wse_exact_count_and_spacing():
    rng = np.random.default_rng(0)
    pts = rng.random((3000, 2)) * 100.0
    n = 500
    r_i = np.full(len(pts), np.sqrt(1.0 / (np.pi * n / (100 * 100))))
    keep = sa.eliminate(pts, n, r_i, metric=None)
    assert len(keep) == n
    kept = pts[keep]
    d = np.sqrt(((kept[:, None] - kept[None]) ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    assert d.min() > 0.0  # blue-noise: no coincident points


def test_wse_isotropic_spacing_beats_random_subset():
    # non-vacuous spacing bound (INIT-005): the old assertion `d.min() > 0` passed for ANY
    # non-coincident set; a real blue-noise sampler must separate points far better than a
    # random subset of the same size.
    rng = np.random.default_rng(7)
    M, n = 4000, 500
    pts = rng.random((M, 2)) * 100.0
    r_i = np.full(M, np.sqrt(1.0 / (np.pi * n / (100 * 100))))
    kept = pts[sa.eliminate(pts, n, r_i, metric=None)]
    d = np.sqrt(((kept[:, None] - kept[None]) ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    rand = pts[rng.choice(M, n, replace=False)]
    dr = np.sqrt(((rand[:, None] - rand[None]) ** 2).sum(-1))
    np.fill_diagonal(dr, np.inf)
    assert d.min() > 3.0 * dr.min()   # WSE min-spacing an order above the random subset


def _nn_axis_ratio(kept):
    d = np.sqrt(((kept[:, None] - kept[None]) ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    disp = kept - kept[d.argmin(1)]
    return np.abs(disp[:, 0]).mean() / np.abs(disp[:, 1]).mean()


def _brute_neighbor_pairs(points, r_i, metric, alpha):
    recv_all, ctrb_all, w_all = [], [], []
    for recv in range(len(points)):
        for ctrb in range(len(points)):
            if recv == ctrb:
                continue
            dv = points[recv] - points[ctrb]
            if metric is None:
                d2 = dv[0] ** 2 + dv[1] ** 2
            else:
                Mm = 0.5 * (metric[recv] + metric[ctrb])
                d2 = (Mm[0, 0] * dv[0] ** 2 + 2.0 * Mm[0, 1] * dv[0] * dv[1]
                      + Mm[1, 1] * dv[1] ** 2)
            two_r = 2.0 * r_i[recv]
            if d2 < two_r * two_r:
                d = np.sqrt(max(d2, 0.0))
                recv_all.append(recv)
                ctrb_all.append(ctrb)
                w_all.append((1.0 - d / two_r) ** alpha)
    return (
        np.asarray(recv_all, dtype=np.int64),
        np.asarray(ctrb_all, dtype=np.int64),
        np.asarray(w_all, dtype=np.float64),
    )


def _sort_pairs(recv, ctrb, w):
    order = np.lexsort((ctrb, recv))
    return recv[order], ctrb[order], w[order]


def test_neighbor_pairs_match_bruteforce_on_sparse_occupied_cells():
    points = np.array([
        [0.0, 0.0],
        [6.0, 0.0],
        [0.0, 7.0],
        [1_000_000.0, 1_000_000.0],
        [1_000_006.0, 1_000_000.0],
        [2_000_000.0, 4.0],
    ])
    r_i = np.array([5.0, 5.5, 4.0, 5.0, 5.5, 3.0])
    actual = _sort_pairs(*sa._neighbor_pairs(points, r_i, metric=None, alpha=3.0))
    expected = _sort_pairs(*_brute_neighbor_pairs(points, r_i, metric=None, alpha=3.0))

    assert np.array_equal(actual[0], expected[0])
    assert np.array_equal(actual[1], expected[1])
    assert np.allclose(actual[2], expected[2])


def test_neighbor_pairs_match_bruteforce_with_metric():
    points = np.array([
        [0.0, 0.0],
        [3.0, 0.2],
        [0.3, 4.0],
        [7.0, 7.0],
        [10.0, 7.5],
    ])
    r_i = np.array([2.6, 2.8, 2.5, 3.0, 2.7])
    metric = sa.anisotropy_metric(
        np.array([0.0, 0.2, 1.1, 0.5, 1.4]),
        np.array([3.0, 2.0, 4.0, 1.5, 2.5]),
    )
    actual = _sort_pairs(*sa._neighbor_pairs(points, r_i, metric=metric, alpha=4.0))
    expected = _sort_pairs(*_brute_neighbor_pairs(points, r_i, metric=metric, alpha=4.0))

    assert np.array_equal(actual[0], expected[0])
    assert np.array_equal(actual[1], expected[1])
    assert np.allclose(actual[2], expected[2])


def test_wse_anisotropic_metric_shapes_nn_displacements():
    # the central contribution: the per-point metric must steer nearest-neighbor spacing.
    # An across=x metric, an across=y metric, and isotropic must give distinct, direction-
    # consistent NN-displacement anisotropy (INIT-005 — previously untested).
    rng = np.random.default_rng(0)
    M, n = 4000, 500
    pts = rng.random((M, 2)) * 100.0
    r_i = np.full(M, np.sqrt(1.0 / (np.pi * n / (100 * 100))))
    iso = _nn_axis_ratio(pts[sa.eliminate(pts, n, r_i, metric=None)])
    mx = _nn_axis_ratio(pts[sa.eliminate(pts, n, r_i,
                          metric=sa.anisotropy_metric(np.zeros(M), np.full(M, 4.0)))])
    my = _nn_axis_ratio(pts[sa.eliminate(pts, n, r_i,
                          metric=sa.anisotropy_metric(np.full(M, np.pi / 2), np.full(M, 4.0)))])
    assert 0.85 < iso < 1.18            # isotropic: no preferred axis
    assert mx > 1.3                     # across=x metric elongates NN displacement in x
    assert my < 0.77                    # rotating the metric 90 deg swaps the axis
    assert mx > my


def test_sampler_warns_when_n_exceeds_candidates():
    pts = np.random.default_rng(0).random((10, 2))
    r_i = np.ones(10)
    with pytest.warns(UserWarning, match="exact-N contract"):
        keep = sa.eliminate(pts, 20, r_i, metric=None)
    assert len(keep) == 10
    with pytest.warns(UserWarning, match="exact-N contract"):
        sa.dart_throwing(pts, 20, r_i, rng=np.random.default_rng(0))


def test_dart_throwing_fill_uses_metric():
    # radii large enough that the accept phase stalls and the shortfall fill dominates; the fill
    # must honor the metric (INIT-005), so a strong anisotropic metric changes which points fill.
    rng = np.random.default_rng(0)
    pts = rng.random((400, 2)) * 10.0
    r_i = np.full(400, 50.0)                       # disks too greedy -> mostly fill
    metric = sa.anisotropy_metric(np.zeros(400), np.full(400, 8.0))
    a = sa.dart_throwing(pts, 120, r_i, metric=None, rng=np.random.default_rng(0))
    b = sa.dart_throwing(pts, 120, r_i, metric=metric, rng=np.random.default_rng(0))
    assert len(a) == len(b) == 120
    assert not np.array_equal(a, b)                # the metric influences the filled selection


def test_wse_density_adaptivity():
    # Adaptivity comes from the PER-POINT radius: small r_i (dense target) on the left,
    # large r_i (sparse target) on the right -> WSE keeps the left denser. (A uniform r_i
    # would instead produce uniform output regardless of candidate density.)
    rng = np.random.default_rng(1)
    left = rng.random((2000, 2)) * np.array([40, 100])
    right = rng.random((1500, 2)) * np.array([60, 100]) + np.array([40, 0])
    pts = np.concatenate([left, right])
    r_i = np.concatenate([np.full(len(left), 2.0), np.full(len(right), 6.0)])
    kept = pts[sa.eliminate(pts, 400, r_i, metric=None)]
    frac_left = np.mean(kept[:, 0] < 40)
    assert frac_left > 0.6  # dense-target region retains more points


def test_anisotropy_metric_unit_area():
    M = sa.anisotropy_metric(np.array([0.3, 1.1, 2.0]), np.array([1.0, 3.0, 6.0]))
    dets = np.linalg.det(M)
    assert np.allclose(dets, 1.0, atol=1e-6)


def test_dart_throwing_exact_count_and_separation():
    rng = np.random.default_rng(2)
    pts = rng.random((3000, 2)) * 100.0
    n = 400
    r_i = np.full(len(pts), np.sqrt(1.0 / (np.pi * n / (100 * 100))))
    keep = sa.dart_throwing(pts, n, r_i, rng=np.random.default_rng(0))
    assert len(keep) == n
    assert len(np.unique(keep)) == n
    kept = pts[keep]
    d = np.sqrt(((kept[:, None] - kept[None]) ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    # random subsets of this size have expected min distance ~0.1; Poisson disk far above
    assert d.min() > 0.5


def test_dart_throwing_fill_preserves_exact_n():
    # radii so large the disks accept only a handful -> the fill path must top up to n
    rng = np.random.default_rng(3)
    pts = rng.random((500, 2)) * 10.0
    keep = sa.dart_throwing(pts, 200, np.full(500, 50.0), rng=np.random.default_rng(0))
    assert len(keep) == 200
    assert len(np.unique(keep)) == 200


def test_farthest_point_exact_count_and_spread():
    rng = np.random.default_rng(4)
    pts = rng.random((2000, 2)) * 50.0
    keep = sa.farthest_point(pts, 100, rng=np.random.default_rng(0))
    assert len(keep) == 100
    kept = pts[keep]
    d = np.sqrt(((kept[:, None] - kept[None]) ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    rand = pts[rng.choice(2000, 100, replace=False)]
    dr = np.sqrt(((rand[:, None] - rand[None]) ** 2).sum(-1))
    np.fill_diagonal(dr, np.inf)
    assert d.min() > 2.0 * dr.min()  # maximin objective => far better separated than random


def test_cvt_count_bounds_and_adaptivity():
    rng = np.random.default_rng(5)
    left = rng.random((4000, 2)) * np.array([20, 100])           # dense candidate half
    right = rng.random((400, 2)) * np.array([80, 100]) + np.array([20, 0])
    pts = np.concatenate([left, right])
    centers = sa.cvt(pts, 300, rng=np.random.default_rng(0))
    assert centers.shape == (300, 2)
    assert np.isfinite(centers).all()
    assert np.mean(centers[:, 0] < 20) > 0.5  # centroids follow the candidate density


def test_halton_unit_low_discrepancy_and_seeded():
    u = sa.halton_unit(512, np.random.default_rng(0))
    assert u.shape == (512, 2) and (u >= 0).all() and (u < 1).all()
    # every cell of a 8x8 stratification is hit (i.i.d. random misses ~1/3 of them at 512)
    cells = (u * 8).astype(int)
    assert len({(a, b) for a, b in cells}) == 64
    v = sa.halton_unit(512, np.random.default_rng(1))
    assert not np.allclose(u, v)  # Cranley-Patterson shift depends on the rng
