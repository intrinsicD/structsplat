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
