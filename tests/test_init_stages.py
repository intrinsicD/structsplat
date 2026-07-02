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


def test_jittered_grid_drops_evenly():
    # a count that forces truncation: with bottom-row dropping the max y-gap doubles
    img = _toy(H=60, W=60)
    f = I.build_field(img, InitConfig(strategy="iso_blue_noise", num_gaussians=91,
                                      sampling_mode="jittered_grid", seed=0))
    assert f.n == 91
    ys = np.sort(np.unique(np.round(f.means[:, 1].numpy() / 6.0)))
    assert ys.max() >= 8  # points still reach the bottom rows of the grid
