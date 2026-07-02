"""Progressive hierarchical fitting = error-driven densification organized as a blue-noise stack.

Level 0 places a fraction of the budget from the image density and fits. Each finer level
recomputes density from the *residual* structure tensor (INIT-002/HIER-001), samples the next
tranche of Gaussians there, appends them (append order == coarse->fine == a natural LOD prefix),
and re-fits the whole field. Because the reference renderer is normalized (not additive, ADR-0003),
this is densification rather than residual summation.
Requires torch.
"""
from __future__ import annotations
import numpy as np
import torch

from .config import InitConfig, FitConfig, PyramidConfig, StructureTensorConfig
from . import init as _init
from . import density as de
from . import structure_tensor as st
from .gaussians import GaussianField
from .render import render
from .fit import fit
from . import metrics as M


def _concat(a: GaussianField, b: GaussianField) -> GaussianField:
    return a.append(b)


def _level_value(values, lvl: int, default):
    if values is None or len(values) == 0:
        return default
    return values[min(lvl, len(values) - 1)]


def _level_tensor_cfg(base: StructureTensorConfig | None, pcfg: PyramidConfig,
                      lvl: int) -> StructureTensorConfig:
    base = base or StructureTensorConfig()
    default_grad_sigma = base.grad_sigma if lvl == 0 else pcfg.residual_grad_sigma
    return StructureTensorConfig(
        grad_sigma=_level_value(pcfg.level_grad_sigmas, lvl, default_grad_sigma),
        tensor_sigma=_level_value(pcfg.level_tensor_sigmas, lvl, base.tensor_sigma),
        flat_frac=base.flat_frac,
        corner_frac=base.corner_frac,
    )


@torch.no_grad()
def prefix_metrics(field: GaussianField, counts: list[int], target: torch.Tensor,
                   cfg: FitConfig) -> list[dict]:
    H, W = target.shape[:2]
    rows = []
    for lvl, n in enumerate(counts):
        sub = field.subset(slice(0, n))
        img = render(sub.means, sub.conics(cfg.aa_dilation), sub.colors,
                     sub.radii(cfg.sigma_cutoff, cfg.aa_dilation), H, W, cfg.render_chunk)
        rows.append({
            "level": lvl,
            "n_gaussians": n,
            "psnr": M.psnr(img, target),
            "ssim": float(M.ssim(img, target)),
            "ms_ssim": M.ms_ssim(img, target),
        })
    return rows


def fit_pyramid(img: np.ndarray, target: torch.Tensor, icfg: InitConfig,
                fcfg: FitConfig, pcfg: PyramidConfig,
                scfg: StructureTensorConfig | None = None, verbose: bool = True) -> dict:
    H, W = img.shape[:2]
    device = target.device
    total = icfg.num_gaussians
    fracs = pcfg.level_fractions[:pcfg.levels]
    fracs = [f / sum(fracs) for f in fracs]

    field = None
    counts = []
    level_summaries = []
    level_cfg = FitConfig(**{**fcfg.__dict__, "iters": pcfg.iters_per_level})
    for lvl, frac in enumerate(fracs):
        n_lvl = max(1, int(round(total * frac)))
        icfg_lvl = InitConfig(**{**icfg.__dict__, "num_gaussians": n_lvl, "seed": icfg.seed + lvl})
        scfg_lvl = _level_tensor_cfg(scfg, pcfg, lvl)
        if lvl == 0:
            new = _init.build_field(img, icfg_lvl, scfg_lvl, device=device)
        else:
            with torch.no_grad():
                cur = render(field.means, field.conics(fcfg.aa_dilation), field.colors,
                             field.radii(fcfg.sigma_cutoff, fcfg.aa_dilation), H, W,
                             fcfg.render_chunk)
                residual = (target - cur).abs().cpu().numpy()
            dens = de.density_from_residual(residual, icfg.density_base, icfg.density_power,
                                            scfg_lvl.grad_sigma)
            tensor = st.compute(residual, scfg_lvl)
            new = _init.build_field(img, icfg_lvl, scfg_lvl, density=dens, tensor=tensor,
                                    device=device)
        field = new if field is None else _concat(field, new)
        counts.append(field.n)
        if verbose:
            print(f"[pyramid] level {lvl}: +{new.n} -> {field.n} gaussians")
        out = fit(field, target, level_cfg, verbose=verbose)
        field = out["field"]
        counts[-1] = field.n
        level_summaries.append({
            "level": lvl,
            "added": new.n,
            "n_gaussians": field.n,
            "psnr": out["psnr"],
            "ms_ssim": out["ms_ssim"],
        })

    out["level_summaries"] = level_summaries
    out["level_counts"] = counts
    if pcfg.evaluate_prefixes:
        out["prefix_metrics"] = prefix_metrics(field, counts, target, fcfg)
    return out
