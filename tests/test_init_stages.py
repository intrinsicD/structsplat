# ruff: noqa: E402
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from structsplat import init as I
from structsplat.config import InitConfig


def _toy(H=48, W=64):
    img = np.zeros((H, W, 3), np.float32)
    img[:, W // 2:] = [0.9, 0.85, 0.2]
    img[10:25, 8:24] = [0.1, 0.3, 0.8]
    return img


def _assert_field_ok(f, n):
    assert f.n == n
    for t in (f.means, f.log_scales, f.rotations, f.colors):
        assert torch.isfinite(t).all()
    assert (f.means[:, 0] >= -0.5).all() and (f.means[:, 0] <= 63.5).all()
    assert (f.means[:, 1] >= -0.5).all() and (f.means[:, 1] <= 47.5).all()


@pytest.mark.parametrize("mode", I.SAMPLING_MODES)
def test_every_sampling_mode_builds_exact_n(mode):
    img = _toy()
    for strat in ("iso_blue_noise", "aniso_flanking"):
        f = I.build_field(img, InitConfig(strategy=strat, num_gaussians=200,
                                          sampling_mode=mode, seed=0))
        _assert_field_ok(f, 200)


@pytest.mark.parametrize("mode", ["dart_throwing", "halton", "farthest_point", "cvt"])
def test_new_sampling_modes_are_seed_deterministic(mode):
    img = _toy()
    cfg = InitConfig(num_gaussians=120, sampling_mode=mode, seed=7)
    a = I.build_field(img, cfg)
    b = I.build_field(img, cfg)
    assert torch.equal(a.means, b.means)
    c = I.build_field(img, InitConfig(num_gaussians=120, sampling_mode=mode, seed=8))
    assert not torch.equal(a.means, c.means)


def test_orientation_modes():
    img = _toy()
    base = I.build_field(img, InitConfig(num_gaussians=150, seed=0))
    zero = I.build_field(img, InitConfig(num_gaussians=150, orientation_mode="zero", seed=0))
    rand = I.build_field(img, InitConfig(num_gaussians=150, orientation_mode="random", seed=0))
    assert torch.equal(zero.rotations, torch.zeros(150))
    assert torch.equal(base.means, zero.means)  # orientation must not move the points
    assert not torch.equal(base.rotations, rand.rotations)
    with pytest.raises(ValueError):
        I.build_field(img, InitConfig(num_gaussians=10, orientation_mode="bogus"))


def test_scale_mode_knn_tracks_local_spacing():
    img = _toy()
    f = I.build_field(img, InitConfig(num_gaussians=150, scale_mode="knn", seed=0))
    _assert_field_ok(f, 150)
    s = torch.exp(f.log_scales)
    assert (s > 0).all() and float(s.max()) < 64.0


def test_nn_spacing_matches_broadcast_reference_across_chunks():
    rng = np.random.default_rng(0)
    pts = rng.random((37, 2)) * np.array([64.0, 48.0])
    d2 = ((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(d2, np.inf)
    expected = np.clip(np.sqrt(d2.min(axis=1)), 0.75, 12.0)

    got = I._nn_spacing(pts, r_min=0.75, r_max=12.0, chunk=5)

    assert np.allclose(got, expected, rtol=1e-12, atol=1e-12)


def test_nn_spacing_rejects_nonpositive_chunk():
    with pytest.raises(ValueError, match="chunk must be > 0"):
        I._nn_spacing(np.zeros((2, 2)), chunk=0)


def test_candidate_oversample_below_one_is_rejected():
    with pytest.raises(ValueError, match="candidate_oversample must be >= 1"):
        InitConfig(candidate_oversample=0.5)
    InitConfig(candidate_oversample=1.0)  # valid


def test_flank_offset_clears_blur_width():
    # the offset floor is applied after the fraction, so flanked edge centers clear the blur
    # width (INIT-005) instead of degenerating toward on-edge placement.
    img = np.full((64, 64, 3), 0.1, np.float32)
    img[:, 32:] = 0.9
    onedge = I.build_field(img, InitConfig(strategy="aniso_onedge", num_gaussians=400, seed=0))
    flanked = I.build_field(img, InitConfig(strategy="aniso_flanking", num_gaussians=400, seed=0))
    # near the ridge, flanked centers sit measurably farther from x=31.5 than on-edge centers
    on_x = onedge.means[:, 0].numpy()
    fl_x = flanked.means[:, 0].numpy()
    on_near = np.abs(on_x - 31.5) < 6
    fl_near = np.abs(fl_x - 31.5) < 6
    assert np.abs(fl_x[fl_near] - 31.5).mean() > np.abs(on_x[on_near] - 31.5).mean()


def test_two_sided_colors_stay_on_the_center_side():
    # step edge: a two_sided color must come from the side of the edge its center is on
    # (the parity sign used for flanking is arbitrary for off-ridge starts)
    img = np.full((64, 64, 3), 0.1, np.float32)
    img[:, 32:] = 0.9
    f = I.build_field(img, InitConfig(strategy="aniso_flanking", num_gaussians=400,
                                      color_mode="two_sided", seed=0))
    x = f.means[:, 0].numpy()
    c = f.colors[:, 0].numpy()
    near = np.abs(x - 31.5) < 4
    agree = np.mean((x[near] > 31.5) == (c[near] > 0.5))
    assert agree > 0.98


def test_jittered_grid_drops_evenly():
    # a count that forces truncation: with bottom-row dropping the max y-gap doubles
    img = _toy(H=60, W=60)
    f = I.build_field(img, InitConfig(strategy="iso_blue_noise", num_gaussians=91,
                                      sampling_mode="jittered_grid", seed=0))
    assert f.n == 91
    ys = np.sort(np.unique(np.round(f.means[:, 1].numpy() / 6.0)))
    assert ys.max() >= 8  # points still reach the bottom rows of the grid


@pytest.mark.parametrize("strategy,color", [
    ("quadtree_aggregate", "aggregate"),
    ("quadtree_hybrid", "aggregate"),
    ("quadtree_wse", "bilinear"),
])
def test_quadtree_variants_are_exact_and_deterministic(strategy, color):
    img = _toy()
    cfg = InitConfig(
        strategy=strategy,
        num_gaussians=128,
        density_mode="variance",
        color_mode=color,
        seed=0,
    )
    a = I.build_field(img, cfg)
    b = I.build_field(img, cfg)
    _assert_field_ok(a, 128)
    assert torch.equal(a.means, b.means)
    assert torch.equal(a.colors, b.colors)
    assert (a.colors >= 0).all() and (a.colors <= 1).all()


def test_hard_scale_cap_is_stored_and_applied():
    img = _toy()
    f = I.build_field(
        img,
        InitConfig(
            strategy="aniso_flanking",
            num_gaussians=128,
            density_mode="variance",
            scale_cap_mode="hard",
            scale_cap_max=3.0,
            seed=0,
        ),
    )
    assert f.scale_max is not None
    assert torch.isfinite(f.scale_max).all()
    assert float(torch.exp(f.log_scales).max()) <= 3.0 + 1e-6
