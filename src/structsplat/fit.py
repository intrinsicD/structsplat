"""Per-image optimization (the baseline fitter every strategy is scored through).

loss = (1 - w) * L1 + w * (1 - SSIM), Adam with per-parameter-group learning rates. Records
PSNR history and iterations-to-target so the ablation can compare *quality at fixed budget*
and *convergence speed* (turn-3 prediction: flanking wins most at low budgets / early iters).
Requires torch.
"""
from __future__ import annotations
import math
import torch

from .config import FitConfig
from .gaussians import GaussianField
from .render import render
from . import metrics as M


def fit(field: GaussianField, target: torch.Tensor, cfg: FitConfig, verbose: bool = True) -> dict:
    H, W = target.shape[0], target.shape[1]
    field.trainable()
    opt = torch.optim.Adam(field.parameter_groups(cfg.lr_means, cfg.lr_scales,
                                                  cfg.lr_rot, cfg.lr_color))
    lo, hi = math.log(0.35), math.log(max(H, W))
    hist = {"iter": [], "psnr": []}
    iters_to_target = None

    for it in range(cfg.iters):
        conics = field.conics()
        radii = field.radii(cfg.sigma_cutoff)
        img = render(field.means, conics, field.colors, radii, H, W, cfg.render_chunk)
        l1 = (img - target).abs().mean()
        s = M.ssim(img, target)
        loss = (1 - cfg.ssim_weight) * l1 + cfg.ssim_weight * (1 - s)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        with torch.no_grad():
            field.log_scales.clamp_(lo, hi)

        if it % cfg.log_every == 0 or it == cfg.iters - 1:
            with torch.no_grad():
                p = M.psnr(img, target)
            hist["iter"].append(it)
            hist["psnr"].append(p)
            if cfg.target_psnr and iters_to_target is None and p >= cfg.target_psnr:
                iters_to_target = it
            if verbose:
                print(f"  iter {it:5d}  psnr {p:6.2f}  loss {float(loss):.5f}")

    with torch.no_grad():
        conics = field.conics()
        radii = field.radii(cfg.sigma_cutoff)
        img = render(field.means, conics, field.colors, radii, H, W, cfg.render_chunk)
        out = {
            "field": field, "history": hist, "render": img,
            "psnr": M.psnr(img, target), "ssim": float(M.ssim(img, target)),
            "ms_ssim": M.ms_ssim(img, target), "lpips": M.LPIPS.distance(img, target),
            "iters_to_target": iters_to_target, "n_gaussians": field.n,
        }
    return out
