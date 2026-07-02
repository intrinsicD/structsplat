import numpy as np
from structsplat import density as de
from structsplat import sampling as sa


def test_warp_unit_points_matches_density():
    # density concentrated in the left half -> most warped points land there
    d = np.ones((40, 60), np.float64)
    d[:, :30] = 9.0
    d /= d.sum()
    u = sa.halton_unit(2000, np.random.default_rng(0))
    pts = de.warp_unit_points(u, d)
    assert pts.shape == (2000, 2)
    assert (pts[:, 0] >= 0).all() and (pts[:, 0] <= 59).all()
    assert (pts[:, 1] >= 0).all() and (pts[:, 1] <= 39).all()
    frac_left = np.mean(pts[:, 0] < 29.5)
    assert 0.85 < frac_left < 0.95  # 9:1 density => 90% of mass on the left


def test_warp_unit_points_uniform_is_stratified():
    d = np.full((32, 32), 1.0 / (32 * 32))
    u = sa.halton_unit(256, np.random.default_rng(0))
    pts = de.warp_unit_points(u, d)
    # uniform density: warp is affine, so Halton stratification survives
    cells = ((pts + 0.5) / 32 * 4).astype(int).clip(0, 3)
    assert len({(a, b) for a, b in cells}) == 16
