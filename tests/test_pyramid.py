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


def test_pyramid_aggregates_iterations_run_across_levels():
    img = _toy()
    target = torch.as_tensor(img)
    icfg = InitConfig(strategy="iso_blue_noise", num_gaussians=32, seed=0)
    fcfg = FitConfig(iters=999, log_every=2)
    pcfg = PyramidConfig(levels=3, level_fractions=[0.2, 0.3, 0.5], iters_per_level=5)
    out = fit_pyramid(img, target, icfg, fcfg, pcfg, verbose=False)
    assert out["iterations_run"] == 3 * 5   # summed across levels, not the last level only
    assert out["stopped_early"] is False


def test_pyramid_accepts_explicit_level_iteration_schedule():
    img = _toy()
    target = torch.as_tensor(img)
    icfg = InitConfig(strategy="iso_blue_noise", num_gaussians=32, seed=0)
    fcfg = FitConfig(iters=999, log_every=1)
    pcfg = PyramidConfig(
        levels=2,
        level_fractions=[0.5, 0.5],
        iters_per_level=99,
        level_iters=[2, 4],
    )
    out = fit_pyramid(img, target, icfg, fcfg, pcfg, verbose=False)
    assert out["iterations_run"] == 6
    assert out["level_iters"] == [2, 4]
    assert [s["iters"] for s in out["level_summaries"]] == [2, 4]
    assert max(out["history"]["iter"]) == 5


def test_pyramid_runs_with_additive_renderer():
    img = _toy()
    target = torch.as_tensor(img)
    icfg = InitConfig(strategy="iso_blue_noise", num_gaussians=24, seed=0)
    fcfg = FitConfig(iters=999, log_every=1, renderer="additive", ssim_weight=0.0)
    pcfg = PyramidConfig(
        levels=2,
        level_fractions=[0.35, 0.65],
        level_iters=[2, 3],
    )

    out = fit_pyramid(img, target, icfg, fcfg, pcfg, verbose=False)

    assert out["iterations_run"] == 5
    assert out["level_budgets"] == [8, 16]
    assert out["prefix_metrics"] is not None
    assert np.isfinite(out["psnr"])


def test_pyramid_rejects_incomplete_level_iteration_schedule():
    img = _toy()
    target = torch.as_tensor(img)
    icfg = InitConfig(strategy="iso_blue_noise", num_gaussians=32, seed=0)
    fcfg = FitConfig(iters=999, log_every=1)
    pcfg = PyramidConfig(levels=3, level_fractions=[0.2, 0.3, 0.5], level_iters=[2, 4])
    with pytest.raises(ValueError, match="level_iters"):
        fit_pyramid(img, target, icfg, fcfg, pcfg, verbose=False)


def test_pyramid_early_stop_leaves_no_phantom_iteration_gap():
    img = _toy()
    target = torch.as_tensor(img)
    icfg = InitConfig(strategy="iso_blue_noise", num_gaussians=24, seed=0)
    # min_delta huge => every level early-stops at the second logged eval (it=1 -> 2 iters run)
    fcfg = FitConfig(iters=999, log_every=1, early_stop_patience=1, early_stop_min_delta=1e6)
    pcfg = PyramidConfig(levels=2, level_fractions=[0.5, 0.5], iters_per_level=6)
    out = fit_pyramid(img, target, icfg, fcfg, pcfg, verbose=False)
    assert out["stopped_early"] is True
    assert out["iterations_run"] == 2 * 2                      # each level stopped after 2 iters
    assert out["stopped_at"] == 1
    h = out["history"]
    # the combined axis advances by actual iters, so no jump to the nominal per-level budget (6)
    assert max(h["iter"]) == out["iterations_run"] - 1
    assert h["iter"] == sorted(h["iter"])
    assert all(i < out["iterations_run"] for i in h["iter"])


def test_allocate_budget_sums_to_total():
    from structsplat.pyramid import _allocate_budget
    cases = [
        (64, [0.1, 0.2, 0.3, 0.4]),
        (10, [0.1, 0.2, 0.3, 0.4]),
        (7, [0.25, 0.25, 0.25, 0.25]),
        (5, [0.2, 0.3, 0.5]),
        (1, [1.0]),
        (20000, [0.1, 0.2, 0.3, 0.4]),
        (3, [0.5, 0.5]),
    ]
    for total, fracs in cases:
        b = _allocate_budget(total, fracs)
        assert sum(b) == total, (total, fracs, b)
        assert len(b) == len(fracs)
        assert all(x >= 0 for x in b)


def test_pyramid_places_exactly_num_gaussians():
    img = _toy()
    target = torch.as_tensor(img)
    icfg = InitConfig(strategy="iso_blue_noise", num_gaussians=50, seed=0)
    fcfg = FitConfig(iters=999)  # no prune/split -> field.n stays the placed budget
    pcfg = PyramidConfig(levels=4, level_fractions=[0.1, 0.2, 0.3, 0.4], iters_per_level=3)
    out = fit_pyramid(img, target, icfg, fcfg, pcfg, verbose=False)
    assert sum(out["level_budgets"]) == 50
    assert out["level_counts"][-1] == 50


def test_pyramid_level_tensor_cfg_keeps_color_space():
    from structsplat.pyramid import _level_tensor_cfg
    base = StructureTensorConfig(color_space="rgb", gradient_operator="scharr")
    for lvl in (0, 1, 3):
        cfg = _level_tensor_cfg(base, PyramidConfig(), lvl)
        assert cfg.color_space == "rgb"
        assert cfg.gradient_operator == "scharr"
