"""Progressive hierarchical fitting = error-driven densification organized as a blue-noise stack.

Level 0 places a fraction of the budget from the image density and fits. Each finer level
recomputes density from the *residual* structure tensor (INIT-002/HIER-001), samples the next
tranche of Gaussians there, appends them (append order == coarse->fine == a natural LOD prefix),
and re-fits the whole field. Because the reference renderer is normalized (not additive, ADR-0003),
this is densification rather than residual summation.
Requires torch.
"""
from __future__ import annotations
import copy
import numpy as np
import torch

from .config import InitConfig, FitConfig, PyramidConfig, StructureTensorConfig
from . import init as _init
from . import density as de
from . import structure_tensor as st
from .gaussians import GaussianField
from .render import render
from .fit import fit


def _concat(a: GaussianField, b: GaussianField) -> GaussianField:
    cat = lambda x, y: torch.cat([x.detach(), y.detach()], 0)
    return GaussianField(cat(a.means, b.means), cat(a.log_scales, b.log_scales),
                         cat(a.rotations, b.rotations), cat(a.colors, b.colors))


def fit_pyramid(img: np.ndarray, target: torch.Tensor, icfg: InitConfig,
                fcfg: FitConfig, pcfg: PyramidConfig,
                scfg: StructureTensorConfig | None = None, verbose: bool = True) -> dict:
    H, W = img.shape[:2]
    total = icfg.num_gaussians
    fracs = pcfg.level_fractions[:pcfg.levels]
    fracs = [f / sum(fracs) for f in fracs]

    field = None
    level_cfg = FitConfig(**{**fcfg.__dict__, "iters": pcfg.iters_per_level})
    for lvl, frac in enumerate(fracs):
        n_lvl = max(1, int(round(total * frac)))
        icfg_lvl = InitConfig(**{**icfg.__dict__, "num_gaussians": n_lvl, "seed": icfg.seed + lvl})
        if lvl == 0:
            new = _init.build_field(img, icfg_lvl, scfg)
        else:
            with torch.no_grad():
                cur = render(field.means, field.conics(), field.colors,
                             field.radii(fcfg.sigma_cutoff), H, W, fcfg.render_chunk)
                residual = (target - cur).abs().cpu().numpy()
            dens = de.density_from_residual(residual, icfg.density_base, icfg.density_power,
                                            pcfg.residual_grad_sigma)
            tensor = st.compute(residual, StructureTensorConfig(grad_sigma=pcfg.residual_grad_sigma))
            new = _init.build_field(img, icfg_lvl, scfg, density=dens, tensor=tensor)
        field = new if field is None else _concat(field, new)
        if verbose:
            print(f"[pyramid] level {lvl}: +{new.n} -> {field.n} gaussians")
        out = fit(field, target, level_cfg, verbose=verbose)
        field = out["field"]

    return out
