"""Progressive hierarchical fitting = error-driven densification organized as a blue-noise stack.

Level 0 places a fraction of the budget from the image density and fits. Each finer level
recomputes density from the *residual* structure tensor (INIT-002/HIER-001), samples the next
tranche of Gaussians there, appends them (append order == coarse->fine == a natural LOD prefix),
and re-fits the whole field. Because the reference renderer is normalized (not additive, ADR-0003),
this is densification rather than residual summation.
Requires torch.
"""
from __future__ import annotations
import math
import numpy as np
import torch

from .config import InitConfig, FitConfig, PyramidConfig, StructureTensorConfig
from . import init as _init
from . import density as de
from . import structure_tensor as st
from .gaussians import GaussianField
from .render import render_field
from .fit import fit
from . import metrics as M


def _concat(a: GaussianField, b: GaussianField) -> GaussianField:
    return a.append(b)


def _allocate_budget(total: int, fracs: list[float]) -> list[int]:
    """Split `total` Gaussians across levels so the level budgets sum exactly to `total`.

    Largest-remainder (Hamilton) allocation: floor each level's share, then hand the leftover
    units to the levels with the largest fractional parts. Per-level rounding (`round(total*f)`)
    made pyramid runs place a slightly different budget than the nominal N they were compared at
    (HIER-002); this guarantees `sum(budgets) == total`.
    """
    total = int(total)
    L = len(fracs)
    s = float(sum(fracs))
    if total <= 0 or s <= 0:
        return [0] * L
    quotas = [total * f / s for f in fracs]
    base = [int(math.floor(q)) for q in quotas]
    rem = total - sum(base)
    # break ties by the (larger) fraction so the ordering is deterministic
    order = sorted(range(L), key=lambda i: (quotas[i] - base[i], fracs[i]), reverse=True)
    for i in order[:rem]:
        base[i] += 1
    return base


def _level_value(values, lvl: int, default):
    if values is None or len(values) == 0:
        return default
    return values[min(lvl, len(values) - 1)]


def _level_iters(pcfg: PyramidConfig) -> list[int]:
    if pcfg.level_iters is None:
        iters = [int(pcfg.iters_per_level)] * int(pcfg.levels)
    else:
        iters = [int(v) for v in pcfg.level_iters[:pcfg.levels]]
        if len(iters) < pcfg.levels:
            raise ValueError(
                f"PyramidConfig.levels={pcfg.levels} but only "
                f"{len(pcfg.level_iters)} level_iters given")
    if any(v <= 0 for v in iters):
        raise ValueError(f"pyramid level iterations must be > 0, got {iters}")
    return iters


def _level_tensor_cfg(base: StructureTensorConfig | None, pcfg: PyramidConfig,
                      lvl: int) -> StructureTensorConfig:
    base = base or StructureTensorConfig()
    default_grad_sigma = base.grad_sigma if lvl == 0 else pcfg.residual_grad_sigma
    return StructureTensorConfig(
        grad_sigma=_level_value(pcfg.level_grad_sigmas, lvl, default_grad_sigma),
        tensor_sigma=_level_value(pcfg.level_tensor_sigmas, lvl, base.tensor_sigma),
        gradient_operator=base.gradient_operator,
        color_space=base.color_space,
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
        img = render_field(sub.means, sub.conics(cfg.aa_dilation), sub.colors,
                           sub.radii(cfg.sigma_cutoff, cfg.aa_dilation),
                           H, W, cfg.render_chunk, cfg.renderer, sub.opacity_values(),
                           scales=sub.effective_scales(
                               cfg.aa_dilation
                               if cfg.renderer in ("gsplat", "cuda_gsplat") else 0.0
                           ), rotations=sub.rotations,
                           support_fade=cfg.support_fade, sigma_cutoff=cfg.sigma_cutoff,
                           checkpoint_chunks=cfg.render_checkpoint)
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
                scfg: StructureTensorConfig | None = None, verbose: bool = True,
                loss_weight_map=None, iteration_observer=None,
                observer_every: int = 0) -> dict:
    H, W = img.shape[:2]
    device = target.device
    total = icfg.num_gaussians
    fracs = pcfg.level_fractions[:pcfg.levels]
    if len(fracs) < pcfg.levels:
        raise ValueError(
            f"PyramidConfig.levels={pcfg.levels} but only {len(fracs)} level_fractions given; "
            "silently running fewer levels would misreport the compute budget")
    if not fracs or sum(fracs) <= 0:
        raise ValueError("level_fractions must contain positive values")
    fracs = [f / sum(fracs) for f in fracs]

    field = None
    counts = []
    level_summaries = []
    # global (across-level) trajectory so convergence metrics cover the WHOLE pyramid run,
    # not just the final level's re-fit
    combined = {
        "iter": [],
        "psnr": [],
        "loss": [],
        "n_gaussians": [],
        "elapsed": [],
        "loss_target_full_weight": [],
    }
    combined_itt: dict[str, int | None] = {}
    fit_seconds_total = 0.0
    iter_offset = 0            # actual iterations run so far (history axis)
    elapsed_offset = 0.0
    iterations_run_total = 0
    stopped_early_any = False
    stopped_at_total = None
    # nominal planned span, so a cosine LR schedule covers the whole pyramid rather than
    # restarting each level (HIER-002); early stops do not change the schedule shape.
    level_iters = _level_iters(pcfg)
    sched_total = sum(level_iters)
    budgets = _allocate_budget(total, fracs)   # sums exactly to `total`
    if fcfg.loss_target_downsample > 1:
        full_target_at = float(fcfg.loss_target_full_frac) * sched_total
        for lvl in range(1, len(budgets)):
            boundary = sum(level_iters[:lvl])
            if budgets[lvl] > 0 and boundary < full_target_at:
                raise ValueError(
                    f"pyramid level {lvl} adds {budgets[lvl]} Gaussians at global iteration "
                    f"{boundary} before the loss-target curriculum reaches the full target "
                    f"at {full_target_at:g}"
                )
    for lvl, frac in enumerate(fracs):
        level_it = level_iters[lvl]
        level_offset = sum(level_iters[:lvl])
        level_cfg = FitConfig(**{**fcfg.__dict__, "iters": level_it})
        n_lvl = budgets[lvl]
        if lvl == 0 and n_lvl <= 0:
            raise ValueError(
                f"level 0 received a 0-Gaussian budget for num_gaussians={total} across "
                f"{pcfg.levels} levels; increase num_gaussians or the coarse fraction")
        icfg_lvl = InitConfig(**{**icfg.__dict__, "num_gaussians": n_lvl, "seed": icfg.seed + lvl})
        scfg_lvl = _level_tensor_cfg(scfg, pcfg, lvl)
        if lvl == 0:
            new = _init.build_field(img, icfg_lvl, scfg_lvl, device=device)
        else:
            with torch.no_grad():
                cur = render_field(field.means, field.conics(fcfg.aa_dilation), field.colors,
                                   field.radii(fcfg.sigma_cutoff, fcfg.aa_dilation), H, W,
                                   fcfg.render_chunk, fcfg.renderer, field.opacity_values(),
                                   scales=field.effective_scales(
                                       fcfg.aa_dilation
                                       if fcfg.renderer in ("gsplat", "cuda_gsplat") else 0.0
                                   ), rotations=field.rotations,
                                   support_fade=fcfg.support_fade,
                                   sigma_cutoff=fcfg.sigma_cutoff,
                                   checkpoint_chunks=fcfg.render_checkpoint)
                residual = (target - cur).abs().cpu().numpy()
            # one tensor drives both density and orientation, under the full level config
            # (previously the density tensor silently used default operator/sigma/thresholds)
            tensor = st.compute(residual, scfg_lvl)
            dens = de.density_from_tensor_and_image(residual, tensor, icfg_lvl, scfg_lvl)
            new = _init.build_field(img, icfg_lvl, scfg_lvl, density=dens, tensor=tensor,
                                    device=device)
        field = new if field is None else _concat(field, new)
        counts.append(field.n)
        if verbose:
            print(f"[pyramid] level {lvl}: +{new.n} -> {field.n} gaussians")
        out = fit(field, target, level_cfg, verbose=verbose,
                  sched_offset=level_offset, sched_total=sched_total,
                  loss_weight_map=loss_weight_map,
                  iteration_observer=iteration_observer, observer_every=observer_every)
        field = out["field"]
        counts[-1] = field.n
        h = out["history"]
        combined["iter"] += [iter_offset + i for i in h["iter"]]
        combined["elapsed"] += [elapsed_offset + e for e in h["elapsed"]]
        for k in ("psnr", "loss", "n_gaussians", "loss_target_full_weight"):
            combined[k] += h[k]
        for key, v in out.get("iters_to_targets", {}).items():
            combined_itt.setdefault(key, None)
            if combined_itt[key] is None and v is not None:
                combined_itt[key] = iter_offset + v
        fit_seconds_total += float(out.get("fit_seconds", 0.0))
        level_iters_run = int(out.get("iterations_run", level_cfg.iters))
        iterations_run_total += level_iters_run
        if bool(out.get("stopped_early", False)):
            stopped_early_any = True
            if stopped_at_total is None and out.get("stopped_at") is not None:
                stopped_at_total = iter_offset + int(out["stopped_at"])
        # advance by iterations actually run, not the full level budget: an early-stopped level
        # would otherwise insert phantom iterations into the combined axis / iters-to-target.
        iter_offset += level_iters_run
        if combined["elapsed"]:
            elapsed_offset = combined["elapsed"][-1]
        level_summaries.append({
            "level": lvl,
            "added": new.n,
            "n_gaussians": field.n,
            "psnr": out["psnr"],
            "ms_ssim": out["ms_ssim"],
            "iters": level_it,
        })

    out["history"] = combined
    out["iters_to_targets"] = combined_itt
    out["iters_to_target"] = (combined_itt.get(str(float(fcfg.target_psnr)))
                              if fcfg.target_psnr is not None else None)
    out["fit_seconds"] = fit_seconds_total
    # aggregate across levels: the final level's fit output only knew about its own re-fit, so
    # pyramid rows under-reported iterations by ~levels x (HIER-002).
    out["iterations_run"] = iterations_run_total
    out["stopped_early"] = stopped_early_any
    out["stopped_at"] = stopped_at_total
    out["level_summaries"] = level_summaries
    out["level_counts"] = counts
    out["level_budgets"] = budgets   # placed per level; sums to num_gaussians
    out["level_iters"] = level_iters
    if pcfg.evaluate_prefixes:
        restructures = (((level_cfg.prune_every or 0) > 0 and level_cfg.prune_min_activity > 0)
                        or ((level_cfg.split_every or 0) > 0 and level_cfg.split_count > 0))
        # pruning/splitting reorders and reshapes the field, so "the first n_l Gaussians"
        # no longer corresponds to levels 0..l — a prefix render would be a fabrication
        out["prefix_metrics"] = None if restructures else prefix_metrics(field, counts, target, fcfg)
    return out
