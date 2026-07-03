"""Per-image optimization (the baseline fitter every strategy is scored through).

loss = (1 - w) * pixel + w * (1 - SSIM), Adam/AdamW with per-parameter-group learning rates.
Records PSNR history and iterations-to-target so the ablation can compare *quality at fixed
budget* and *convergence speed*. Supports selectable pixel loss (l1/l2/charbonnier, with an
optional warmup), optimizer, LR schedule, renderer mode, opacity, and residual densification.
Requires torch.
"""
from __future__ import annotations
import math
import time
import torch
import torch.nn.functional as F

from .config import FitConfig
from .gaussians import GaussianField
from .render import (
    _flat_tile_slices,
    _tile_bounds,
    _tile_coords,
    gaussian_activity,
    render_field,
)
from . import metrics as M

# Renderer modes that accumulate additively (ADR-0006): a new/split Gaussian stacks on the
# existing accumulation, so it must carry the residual color, not the full target color.
_ADDITIVE_RENDERERS = ("additive", "cuda_additive", "gsplat", "cuda_gsplat")
# Shared lower bound on densified Gaussian scales (px); was duplicated as bare 0.35 literals.
_MIN_DENSIFY_SCALE = 0.35


def _make_optimizer(field: GaussianField, cfg: FitConfig):
    groups = field.parameter_groups(cfg.lr_means, cfg.lr_scales, cfg.lr_rot, cfg.lr_color,
                                    cfg.lr_opacity)
    if cfg.optimizer == "adam":
        return torch.optim.Adam(groups)
    if cfg.optimizer == "adamw":
        return torch.optim.AdamW(groups)
    raise ValueError(f"unknown optimizer {cfg.optimizer!r}; expected adam or adamw")


def _lr_factor(cfg: FitConfig, it: int, sched_offset: int = 0,
               sched_total: int | None = None) -> float:
    """LR multiplier as a pure function of the global iteration.

    Because it depends only on `it` (not optimizer internal state), the schedule survives the
    optimizer rebuild after a prune/split — a freshly constructed StepLR/CosineLR would silently
    reset. `lr_schedule` selects the shape; `lr_schedule="none"` with `lr_decay_every` set falls
    back to step decay for backward compatibility.

    `sched_offset`/`sched_total` let a caller place this fit inside a larger run so the cosine
    phase spans the whole run rather than restarting each call: the pyramid passes the level's
    global start and the pyramid-wide iteration count so `lr_schedule='cosine'` measures the same
    schedule in pyramid and single-stage cells (HIER-002). Defaults reproduce the single-stage
    behavior (offset 0, total = cfg.iters).
    """
    schedule = cfg.lr_schedule
    if schedule == "none" and cfg.lr_decay_every is not None and cfg.lr_decay_every > 0:
        schedule = "step"
    if schedule == "none":
        return 1.0
    if schedule == "step":
        step = cfg.lr_decay_every if cfg.lr_decay_every and cfg.lr_decay_every > 0 else 500
        return cfg.lr_decay_gamma ** ((sched_offset + it) // step)
    if schedule == "cosine":
        total = sched_total if sched_total is not None else cfg.iters
        return 0.5 * (1.0 + math.cos(math.pi * (sched_offset + it) / max(1, total)))
    raise ValueError(f"unknown lr_schedule {cfg.lr_schedule!r}; expected none, step, or cosine")


def _carry_adam_state(opt_old, field: GaussianField, cfg: FitConfig,
                      keep: torch.Tensor | None, n_new: int):
    """Fresh optimizer for a restructured field, carrying Adam moments for survivors.

    Without this, every prune/split resets exp_avg/exp_avg_sq for ALL Gaussians and the
    loss spikes while Adam re-estimates curvature. Surviving rows keep their moments; new
    rows start at zero and share the tensor's inherited `step` (Adam keeps one step per
    parameter tensor), so their first updates behave like a normal warm Adam step. Handles
    any number of parameter groups (4, or 5 with opacity) and both Adam/AdamW.
    """
    opt_new = _make_optimizer(field, cfg)
    for g_old, g_new in zip(opt_old.param_groups, opt_new.param_groups):
        p_old, p_new = g_old["params"][0], g_new["params"][0]
        st = opt_old.state.get(p_old)
        if not st:
            continue
        exp_avg, exp_sq = st["exp_avg"], st["exp_avg_sq"]
        if keep is not None:
            exp_avg, exp_sq = exp_avg[keep], exp_sq[keep]
        if n_new > 0:
            pad = exp_avg.new_zeros((n_new,) + exp_avg.shape[1:])
            exp_avg = torch.cat([exp_avg, pad], dim=0)
            exp_sq = torch.cat([exp_sq, pad.clone()], dim=0)
        step = st["step"]
        opt_new.state[p_new] = {
            "step": step.clone() if torch.is_tensor(step) else step,
            "exp_avg": exp_avg.contiguous(),
            "exp_avg_sq": exp_sq.contiguous(),
        }
    return opt_new


def _pixel_loss(pred: torch.Tensor, target: torch.Tensor, kind: str,
                charbonnier_eps: float = 1e-3) -> torch.Tensor:
    if kind == "l1":
        return (pred - target).abs().mean()
    if kind == "l2":
        return F.mse_loss(pred, target)
    if kind == "charbonnier":
        return torch.sqrt((pred - target) ** 2 + charbonnier_eps ** 2).mean()
    raise ValueError(f"unknown pixel_loss {kind!r}; expected l1, l2, or charbonnier")


def _loss_kind(cfg: FitConfig, it: int) -> str:
    if cfg.loss_warmup_iters > 0 and it < cfg.loss_warmup_iters:
        return cfg.loss_warmup_pixel_loss
    return cfg.pixel_loss


def _render(field: GaussianField, cfg: FitConfig, H: int, W: int) -> torch.Tensor:
    return render_field(field.means, field.conics(cfg.aa_dilation), field.colors,
                        field.radii(cfg.sigma_cutoff, cfg.aa_dilation),
                        H, W, cfg.render_chunk, cfg.renderer, field.opacity_values(),
                        scales=field.scales(), rotations=field.rotations)


def _target_list(cfg: FitConfig) -> list[float]:
    # normalize to float so the history keys are always str(float); an int in target_psnrs
    # would otherwise produce a "35" key that str(float(target_psnr)) == "35.0" never finds
    vals = [float(v) for v in cfg.target_psnrs]
    if cfg.target_psnr is not None:
        vals.append(float(cfg.target_psnr))
    return sorted(set(vals))


@torch.no_grad()
def _nearest_image_colors(target: torch.Tensor, means: torch.Tensor) -> torch.Tensor:
    H, W = target.shape[:2]
    xy = means.detach()
    x = torch.clamp(torch.round(xy[:, 0]).long(), 0, W - 1)
    y = torch.clamp(torch.round(xy[:, 1]).long(), 0, H - 1)
    return target[y, x]


@torch.no_grad()
def _nearest_scale_caps(field: GaussianField, means: torch.Tensor) -> torch.Tensor | None:
    if field.scale_max is None:
        return None
    if field.n == 0 or means.numel() == 0:
        return None
    idx = torch.cdist(means.detach(), field.means.detach()).argmin(dim=1)
    return field.scale_max.detach()[idx].clone()


@torch.no_grad()
def _clamp_new_log_scales(log_scales: torch.Tensor, scale_max: torch.Tensor | None
                          ) -> torch.Tensor:
    if scale_max is None:
        return log_scales
    scales = torch.minimum(torch.exp(log_scales), torch.clamp(scale_max, min=1e-3))
    return torch.log(torch.clamp(scales, min=1e-3))


@torch.no_grad()
def _support_residual_scores(field: GaussianField, residual: torch.Tensor,
                             cfg: FitConfig) -> torch.Tensor:
    H, W = residual.shape
    dev, dt = field.means.device, field.means.dtype
    means = field.means.detach()
    conics = field.conics(cfg.aa_dilation).detach()
    radii = field.radii(cfg.sigma_cutoff, cfg.aa_dilation)
    score = torch.zeros(field.n, device=dev, dtype=dt)
    weight = torch.zeros(field.n, device=dev, dtype=dt)
    x0, y0, Tx, n = _tile_bounds(means, radii, H, W)
    budget = max(cfg.render_chunk, 64) * 4096
    for s, e in _flat_tile_slices(n, budget):
        gid, px, py = _tile_coords(x0, y0, Tx, n, s, e, dev)
        dx = px.to(dt) - means[gid, 0]
        dy = py.to(dt) - means[gid, 1]
        a, b, c = conics[gid, 0], conics[gid, 1], conics[gid, 2]
        q = a * dx * dx + 2.0 * b * dx * dy + c * dy * dy
        w = torch.exp(-0.5 * q)
        score.index_add_(0, gid, w * residual[py, px].to(dt))
        weight.index_add_(0, gid, w)
    return score / (weight + 1e-8)


def _luma(img: torch.Tensor) -> torch.Tensor:
    return 0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]


@torch.no_grad()
def _image_gradients(img: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    gray = _luma(img)
    gx = torch.zeros_like(gray)
    gy = torch.zeros_like(gray)
    gx[:, 1:-1] = 0.5 * (gray[:, 2:] - gray[:, :-2])
    gx[:, 0] = gray[:, 1] - gray[:, 0] if gray.shape[1] > 1 else 0.0
    gx[:, -1] = gray[:, -1] - gray[:, -2] if gray.shape[1] > 1 else 0.0
    gy[1:-1, :] = 0.5 * (gray[2:, :] - gray[:-2, :])
    gy[0, :] = gray[1, :] - gray[0, :] if gray.shape[0] > 1 else 0.0
    gy[-1, :] = gray[-1, :] - gray[-2, :] if gray.shape[0] > 1 else 0.0
    return gx, gy


@torch.no_grad()
def _maybe_prune(field: GaussianField, cfg: FitConfig, H: int,
                 W: int) -> tuple[GaussianField, torch.Tensor | None]:
    if cfg.prune_min_activity <= 0.0:
        return field, None
    if field.n <= max(1, cfg.prune_keep_min):
        return field, None
    activity = gaussian_activity(field.means, field.conics(cfg.aa_dilation),
                                 field.radii(cfg.sigma_cutoff, cfg.aa_dilation),
                                 H, W, cfg.render_chunk)
    # gaussian_activity is opacity-free; without this a Gaussian the optimizer has driven fully
    # transparent keeps its geometric weight-sum and is never pruned at fixed N (FIT-002).
    opac = field.opacity_values()
    if opac is not None:
        activity = activity * opac
    keep = activity > cfg.prune_min_activity
    if int(keep.sum()) < cfg.prune_keep_min:
        topk = torch.topk(activity, k=min(cfg.prune_keep_min, field.n)).indices
        keep = torch.zeros_like(activity, dtype=torch.bool)
        keep[topk] = True
    if int(keep.sum()) >= field.n:
        return field, None
    return field.subset(keep), keep


@torch.no_grad()
def _split_from_residual(field: GaussianField, target: torch.Tensor, render_img: torch.Tensor,
                         cfg: FitConfig) -> tuple[GaussianField, int]:
    if cfg.split_count <= 0:
        return field, 0
    if cfg.max_gaussians is not None and field.n >= cfg.max_gaussians:
        return field, 0

    H, W = target.shape[:2]
    residual = (render_img - target).abs().mean(dim=2)
    x = torch.clamp(torch.round(field.means[:, 0]).long(), 0, W - 1)
    y = torch.clamp(torch.round(field.means[:, 1]).long(), 0, H - 1)
    if cfg.split_mode == "support_duplicate":
        scores = _support_residual_scores(field, residual, cfg)
    else:
        scores = residual[y, x]
    room = field.n if cfg.max_gaussians is None else max(0, cfg.max_gaussians - field.n)
    k = min(cfg.split_count, field.n, room)
    if k <= 0:
        return field, 0

    idx = torch.topk(scores, k=k).indices
    means = field.means.detach()[idx].clone()
    scales = field.scales().detach()[idx]
    theta = field.rotations.detach()[idx]
    direction = torch.stack([torch.cos(theta), torch.sin(theta)], dim=1)
    sign = torch.where(torch.arange(k, device=means.device) % 2 == 0, 1.0, -1.0).unsqueeze(1)
    offset = direction * scales[:, :1] * 0.35 * sign
    means = means + offset
    means[:, 0].clamp_(0, W - 1)
    means[:, 1].clamp_(0, H - 1)
    log_scales = field.log_scales.detach()[idx].clone() + math.log(cfg.split_scale)
    rotations = theta.clone()
    if cfg.renderer in _ADDITIVE_RENDERERS:
        # additive stacks on the existing accumulation: full target colors double-count
        # brightness at the child positions; the residual is what the child should add.
        colors = _nearest_image_colors((target - render_img).detach(), means)
    else:
        colors = _nearest_image_colors(target, means)
    opacities = None if field.opacities is None else field.opacities.detach()[idx].clone()
    scale_max = None if field.scale_max is None else field.scale_max.detach()[idx].clone()
    log_scales = _clamp_new_log_scales(log_scales, scale_max)
    new = GaussianField(means, log_scales, rotations, colors, opacities, scale_max)
    return field.append(new), k


@torch.no_grad()
def _add_from_residual(field: GaussianField, target: torch.Tensor, render_img: torch.Tensor,
                       cfg: FitConfig, tensor_aligned: bool = False) -> tuple[GaussianField, int]:
    if cfg.split_count <= 0:
        return field, 0
    if cfg.max_gaussians is not None and field.n >= cfg.max_gaussians:
        return field, 0

    H, W = target.shape[:2]
    residual_map = (render_img - target).abs().mean(dim=2)
    if tensor_aligned:
        score_map = F.avg_pool2d(residual_map[None, None], 3, stride=1, padding=1)[0, 0]
        residual = (0.7 * residual_map + 0.3 * score_map).reshape(-1)
    else:
        residual = residual_map.reshape(-1)
    room = field.n if cfg.max_gaussians is None else max(0, cfg.max_gaussians - field.n)
    k = min(cfg.split_count, int(residual.numel()), room)
    if k <= 0:
        return field, 0

    idx = torch.topk(residual, k=k).indices
    y = torch.div(idx, W, rounding_mode="floor")
    x = idx - y * W
    offsets = torch.stack([
        ((torch.arange(k, device=target.device) % 3).to(target.dtype) - 1.0) * 0.25,
        (((torch.arange(k, device=target.device) // 3) % 3).to(target.dtype) - 1.0) * 0.25,
    ], dim=1)
    means = torch.stack([x.to(target.dtype), y.to(target.dtype)], dim=1) + offsets
    means[:, 0].clamp_(0, W - 1)
    means[:, 1].clamp_(0, H - 1)
    base_scale = math.sqrt((H * W) / max(field.n + k, 1)) * cfg.split_scale
    if tensor_aligned:
        gx, gy = _image_gradients(target)
        grad = torch.sqrt(gx[y, x] ** 2 + gy[y, x] ** 2)
        ref = torch.quantile(torch.sqrt(gx ** 2 + gy ** 2).reshape(-1), 0.95)
        coherence = torch.clamp(grad / torch.clamp(ref, min=1e-6), 0.0, 1.0)
        # mirror InitConfig anisotropy: ratio = 1 + (max_axis_ratio-1)*coherence**power
        ratio = 1.0 + (cfg.densify_max_axis_ratio - 1.0) * coherence ** cfg.densify_coherence_power
        base = max(base_scale, _MIN_DENSIFY_SCALE)
        s_along = base * torch.sqrt(ratio)
        s_across = base / torch.sqrt(ratio)
        log_scales = torch.log(torch.stack([s_along, s_across], dim=1).clamp(min=_MIN_DENSIFY_SCALE))
        rotations = torch.atan2(gy[y, x], gx[y, x]) + math.pi * 0.5
    else:
        log_scales = torch.full((k, 2), math.log(max(base_scale, _MIN_DENSIFY_SCALE)),
                                device=target.device, dtype=target.dtype)
        rotations = torch.zeros(k, device=target.device, dtype=target.dtype)
    scale_max = _nearest_scale_caps(field, means)
    log_scales = _clamp_new_log_scales(log_scales, scale_max)
    if cfg.renderer in _ADDITIVE_RENDERERS:
        # additive stacks on the existing accumulation: injecting full target colors
        # overshoots; the residual color is what the new Gaussian should contribute
        colors = (target - render_img)[y, x]
    else:
        colors = target[y, x]
    return field.append(GaussianField(means, log_scales, rotations, colors,
                                      scale_max=scale_max)), k


def fit(field: GaussianField, target: torch.Tensor, cfg: FitConfig, verbose: bool = True,
        sched_offset: int = 0, sched_total: int | None = None) -> dict:
    H, W = target.shape[0], target.shape[1]
    field.trainable()
    opt = _make_optimizer(field, cfg)
    base_lrs = [g["lr"] for g in opt.param_groups]
    lo, hi = math.log(0.35), math.log(max(H, W))
    hist = {"iter": [], "psnr": [], "loss": [], "n_gaussians": [], "elapsed": []}
    targets = _target_list(cfg)
    iters_to_targets = {str(t): None for t in targets}
    start_time = time.time()
    best_logged_psnr = -math.inf
    stale_logs = 0
    stopped_early = False
    last_iter = -1

    for it in range(cfg.iters):
        last_iter = it
        factor = _lr_factor(cfg, it, sched_offset, sched_total)
        for g, base in zip(opt.param_groups, base_lrs):
            g["lr"] = base * factor
        img = _render(field, cfg, H, W)
        # count that produced this render/PSNR, before any prune/split restructures the field;
        # logging the post-restructure field.n would pair a pre-step PSNR with a post-step N.
        n_at_render = field.n
        pix = _pixel_loss(img, target, _loss_kind(cfg, it), cfg.charbonnier_eps)
        s = M.ssim(img, target)
        loss = (1 - cfg.ssim_weight) * pix + cfg.ssim_weight * (1 - s)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        log_now = it % cfg.log_every == 0 or it == cfg.iters - 1
        with torch.no_grad():
            field.log_scales.clamp_(lo, hi)
            if getattr(field, "scale_max", None) is not None:
                cap = torch.log(torch.clamp(field.scale_max, min=1e-3))
                torch.minimum(field.log_scales, cap, out=field.log_scales)
            if targets or log_now:
                p_now = M.psnr(img, target)
            for t in targets:
                key = str(t)
                if iters_to_targets[key] is None and p_now >= t:
                    iters_to_targets[key] = it

        keep = None
        added = 0
        # never restructure on the final iteration: the returned field/metrics would include
        # Gaussians that no optimizer step ever touched
        last_it = it == cfg.iters - 1
        if (not last_it and cfg.prune_every is not None and cfg.prune_every > 0
                and (it + 1) % cfg.prune_every == 0):
            field, keep = _maybe_prune(field, cfg, H, W)
            if verbose and keep is not None:
                print(f"  prune {int((~keep).sum())} -> {field.n} gaussians")
        if (not last_it and cfg.split_every is not None and cfg.split_every > 0
                and (it + 1) % cfg.split_every == 0):
            if cfg.split_mode in ("duplicate", "support_duplicate"):
                field, added = _split_from_residual(field, target, img, cfg)
            elif cfg.split_mode in ("residual_add", "residual_tensor_add"):
                field, added = _add_from_residual(
                    field, target, img, cfg, tensor_aligned=cfg.split_mode == "residual_tensor_add"
                )
            else:
                raise ValueError(
                    f"unknown split_mode {cfg.split_mode!r}; expected duplicate, "
                    "support_duplicate, residual_add, or residual_tensor_add")
            if verbose and added > 0:
                print(f"  split +{added} -> {field.n} gaussians")
        if keep is not None or added > 0:
            field.trainable()
            opt = _carry_adam_state(opt, field, cfg, keep, added)
            base_lrs = [g["lr"] for g in opt.param_groups]

        if log_now:
            hist["iter"].append(it)
            hist["psnr"].append(p_now)
            hist["loss"].append(loss.item())
            hist["n_gaussians"].append(n_at_render)
            hist["elapsed"].append(time.time() - start_time)
            if verbose:
                print(f"  iter {it:5d}  psnr {p_now:6.2f}  loss {loss.item():.5f}")
            if cfg.early_stop_patience is not None and it >= cfg.early_stop_min_iters:
                if p_now > best_logged_psnr + cfg.early_stop_min_delta:
                    best_logged_psnr = p_now
                    stale_logs = 0
                else:
                    stale_logs += 1
                    if stale_logs >= cfg.early_stop_patience:
                        stopped_early = True
                        if verbose:
                            print(f"  early stop at iter {it}  best_psnr {best_logged_psnr:6.2f}")
                        break

    fit_seconds = time.time() - start_time  # before the final eval: metrics (esp. LPIPS
    with torch.no_grad():                   # model construction) must not pollute the timing
        img = _render(field, cfg, H, W)
        out = {
            "field": field, "history": hist, "render": img,
            "psnr": M.psnr(img, target), "ssim": float(M.ssim(img, target)),
            "ms_ssim": M.ms_ssim(img, target),
            "lpips": M.LPIPS.distance(img, target) if cfg.compute_lpips else None,
            "iters_to_target": (
                iters_to_targets.get(str(float(cfg.target_psnr)))
                if cfg.target_psnr is not None else None
            ),
            "iters_to_targets": iters_to_targets,
            "n_gaussians": field.n,
            "fit_seconds": fit_seconds,
            "iterations_run": last_iter + 1,
            "stopped_early": stopped_early,
        }
    return out
