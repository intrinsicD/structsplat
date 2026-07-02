# ruff: noqa: E402
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from structsplat import init as _init  # noqa: F401  (torch import guard ordering)
from structsplat.config import FitConfig, InitConfig, PyramidConfig, StructureTensorConfig
from structsplat.pyramid import fit_pyramid


def _toy():
    img = np.zeros((24, 32, 3), np.float32)
    img[:, 16:] = 1.0
    img[5:12, 4:12] = [1.0, 0.5, 0.0]
    return img


def test_pyramid_aggregates_history_and_targets_across_levels():
    img = _toy()
    target = torch.as_tensor(img)
    icfg = InitConfig(strategy="iso_blue_noise", num_gaussians=64, seed=0)
    fcfg = FitConfig(iters=999, log_every=2, target_psnr=5.0)  # iters overridden per level
    pcfg = PyramidConfig(levels=2, level_fractions=[0.5, 0.5], iters_per_level=6)
    out = fit_pyramid(img, target, icfg, fcfg, pcfg, verbose=False)
    h = out["history"]
    assert max(h["iter"]) >= 6  # second level's iterations continue the global count
    assert h["iter"] == sorted(h["iter"])
    assert h["elapsed"] == sorted(h["elapsed"])
    assert out["iters_to_target"] is not None  # trivial target: reached in level 0
    assert out["fit_seconds"] > 0
    assert out["prefix_metrics"] is not None and len(out["prefix_metrics"]) == 2


def test_pyramid_rejects_missing_fractions():
    img = _toy()
    target = torch.as_tensor(img)
    icfg = InitConfig(num_gaussians=16, seed=0)
    fcfg = FitConfig(iters=2)
    with pytest.raises(ValueError):
        fit_pyramid(img, target, icfg, fcfg,
                    PyramidConfig(levels=3, level_fractions=[0.5, 0.5], iters_per_level=1),
                    verbose=False)


def test_pyramid_prefix_metrics_skipped_under_restructuring():
    img = _toy()
    target = torch.as_tensor(img)
    icfg = InitConfig(strategy="iso_blue_noise", num_gaussians=32, seed=0)
    fcfg = FitConfig(iters=999, split_every=3, split_count=4, split_mode="residual_add")
    pcfg = PyramidConfig(levels=2, level_fractions=[0.5, 0.5], iters_per_level=6)
    out = fit_pyramid(img, target, icfg, fcfg, pcfg, verbose=False)
    assert out["prefix_metrics"] is None  # prefixes are meaningless once the field reshuffles


def test_pyramid_level_tensor_cfg_keeps_color_space():
    from structsplat.pyramid import _level_tensor_cfg
    base = StructureTensorConfig(color_space="rgb", gradient_operator="scharr")
    for lvl in (0, 1, 3):
        cfg = _level_tensor_cfg(base, PyramidConfig(), lvl)
        assert cfg.color_space == "rgb"
        assert cfg.gradient_operator == "scharr"
