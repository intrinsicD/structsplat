import numpy as np
import pytest

torch = pytest.importorskip("torch")
from structsplat.config import InitConfig, FitConfig, PyramidConfig
from structsplat import init as _init
from structsplat.init import STRATEGIES
from structsplat.fit import fit
from structsplat.pyramid import fit_pyramid


def _toy(H=32, W=32):
    img = np.zeros((H, W, 3), np.float32)
    img[:, W // 2:, :] = 1.0
    img[8:12, 8:12] = [1.0, 0.5, 0.0]
    return img


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_build_field_all_strategies(strategy):
    img = _toy()
    icfg = InitConfig(strategy=strategy, num_gaussians=64, seed=0)
    field = _init.build_field(img, icfg)
    assert 32 <= field.n <= 90
    for p in (field.means, field.log_scales, field.rotations, field.colors):
        assert torch.isfinite(p).all()


def test_fit_runs_and_improves():
    img = _toy()
    target = torch.as_tensor(img)
    icfg = InitConfig(strategy="aniso_flanking", num_gaussians=64)
    field = _init.build_field(img, icfg)
    out = fit(field, target, FitConfig(iters=15, log_every=5))
    assert np.isfinite(out["psnr"]) and out["psnr"] > 0


def test_pyramid_runs():
    img = _toy()
    target = torch.as_tensor(img)
    icfg = InitConfig(strategy="aniso_flanking", num_gaussians=48)
    out = fit_pyramid(img, target, icfg, FitConfig(iters=10, log_every=5),
                      PyramidConfig(levels=2, level_fractions=[0.5, 0.5], iters_per_level=8),
                      verbose=False)
    assert np.isfinite(out["psnr"])
