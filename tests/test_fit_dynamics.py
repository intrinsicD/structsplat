# ruff: noqa: E402
import numpy as np
import pytest

torch = pytest.importorskip("torch")
from structsplat.config import FitConfig, InitConfig
from structsplat import init as _init
from structsplat.fit import _carry_adam_state, _lr_factor, _make_optimizer, fit


def test_lr_factor_is_global_step_decay():
    cfg = FitConfig(lr_decay_every=100, lr_decay_gamma=0.5)
    assert _lr_factor(cfg, 0) == 1.0
    assert _lr_factor(cfg, 99) == 1.0
    assert _lr_factor(cfg, 100) == 0.5
    assert _lr_factor(cfg, 250) == 0.25
    assert _lr_factor(FitConfig(lr_decay_every=None), 1000) == 1.0


def test_carry_adam_state_preserves_moments_across_prune_and_split():
    img = np.random.default_rng(0).random((16, 16, 3)).astype(np.float32)
    field = _init.build_field(img, InitConfig(strategy="random", num_gaussians=8, seed=0))
    field.trainable()
    cfg = FitConfig()
    opt = _make_optimizer(field, cfg)
    target = torch.as_tensor(img)
    from structsplat.render import render
    im = render(field.means, field.conics(), field.colors, field.radii(3.0), 16, 16)
    (im - target).abs().mean().backward()
    opt.step()
    old_exp_avg = opt.state[field.means]["exp_avg"].clone()

    keep = torch.zeros(8, dtype=torch.bool)
    keep[:5] = True
    pruned = field.subset(keep)
    n_new = 3
    grown = pruned.append(pruned.subset(slice(0, n_new)))
    grown.trainable()
    opt2 = _carry_adam_state(opt, grown, cfg, keep, n_new)
    st = opt2.state[grown.means]
    assert torch.allclose(st["exp_avg"][:5], old_exp_avg[:5])   # survivors keep moments
    assert torch.all(st["exp_avg"][5:] == 0)                    # new rows start clean
    assert st["exp_avg"].shape[0] == 8


def test_fit_with_decay_prune_split_still_improves():
    img = np.zeros((24, 24, 3), np.float32)
    img[:, 12:] = 1.0
    target = torch.as_tensor(img)
    field = _init.build_field(img, InitConfig(strategy="iso_blue_noise", num_gaussians=32, seed=0))
    out = fit(field, target, FitConfig(
        iters=30, log_every=5, lr_decay_every=10, lr_decay_gamma=0.5,
        prune_every=8, prune_min_activity=1e-6, prune_keep_min=8,
        split_every=8, split_count=4, max_gaussians=64,
    ), verbose=False)
    assert np.isfinite(out["psnr"])
    assert out["history"]["psnr"][-1] >= out["history"]["psnr"][0] - 0.5
