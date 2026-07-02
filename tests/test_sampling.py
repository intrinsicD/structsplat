import numpy as np
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
