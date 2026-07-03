import numpy as np
from structsplat import density as de
from structsplat import sampling as sa
from structsplat import structure_tensor as st
from structsplat.config import InitConfig


def test_hybrid_density_power_changes_pmf():
    # hybrid must apply base/power/normalize exactly once (INIT-005): previously it mixed two
    # already-normalized pmfs and re-normalized the mixture, so density_power meant something
    # different for this mode. density_power must still visibly reshape the hybrid pmf.
    img = np.zeros((48, 48, 3), np.float32)
    img[:, 24:] = 1.0
    img[10:20, 10:20] = 0.5
    tensor = st.compute(img)
    p1 = de.density_from_tensor_and_image(img, tensor, InitConfig(density_mode="hybrid",
                                                                  density_power=1.0))
    p3 = de.density_from_tensor_and_image(img, tensor, InitConfig(density_mode="hybrid",
                                                                  density_power=3.0))
    assert np.abs(p1 - p3).max() > 1e-5
    assert np.isclose(p1.sum(), 1.0) and np.isclose(p3.sum(), 1.0)


def test_sample_candidates_no_border_atoms():
    # folding (not clipping) the half-pixel jitter must not pile probability atoms at the
    # image borders (INIT-005).
    H, W = 32, 40
    density = np.ones((H, W), np.float64)
    density /= density.sum()
    pts = de.sample_candidates(density, 20000, np.random.default_rng(0))
    xs, ys = pts[:, 0], pts[:, 1]
    assert np.mean(xs == 0.0) < 1e-3 and np.mean(xs == W - 1.0) < 1e-3
    assert np.mean(ys == 0.0) < 1e-3 and np.mean(ys == H - 1.0) < 1e-3
    edge_band = np.mean(xs < 0.5) + np.mean(xs > W - 1.5)
    interior_band = np.mean(np.abs(xs - W / 2) < 0.5)
    assert edge_band < 4.0 * interior_band


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
