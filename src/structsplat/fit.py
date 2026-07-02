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
from .render import gaussian_activity, render_field
from . import metrics as M


def _make_optimizer(field: GaussianField, cfg: FitConfig):
    groups = field.parameter_groups(cfg.lr_means, cfg.lr_scales, cfg.lr_rot, cfg.lr_color,
                                    cfg.lr_opacity)
    if cfg.optimizer == "adam":
        return torch.optim.Adam(groups)
    if cfg.optimizer == "adamw":
        return torch.optim.AdamW(groups)
    raise ValueError(f"unknown optimizer {cfg.optimizer!r}; expected adam or adamw")


def _lr_factor(cfg: FitConfig, it: int) -> float:
    """LR multiplier as a pure function of the global iteration.

    Because it depends only on `it` (not optimizer internal state), the schedule survives the
    optimizer rebuild after a prune/split — a freshly constructed StepLR/CosineLR would silently
    reset. `lr_schedule` selects the shape; `lr_schedule="none"` with `lr_decay_every` set falls
    back to step decay for backward compatibility.
    """
    schedule = cfg.lr_schedule
    if schedule == "none" and cfg.lr_decay_every is not None and cfg.lr_decay_every > 0:
        schedule = "step"
    if schedule == "none":
        return 1.0
    if schedule == "step":
        step = cfg.lr_decay_every if cfg.lr_decay_every and cfg.lr_decay_every > 0 else 500
        return cfg.lr_decay_gamma ** (it // step)
    if schedule == "cosine":
        return 0.5 * (1.0 + math.cos(math.pi * it / max(1, cfg.iters)))
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
                        H, W, cfg.render_chunk, cfg.renderer, field.opacity_values())


def _target_list(cfg: FitConfig) -> list[float]:
    vals = list(cfg.target_psnrs)
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
def _maybe_prune(field: GaussianField, cfg: FitConfig, H: int,
                 W: int) -> tuple[GaussianField, torch.Tensor | None]:
    if cfg.prune_min_activity <= 0.0:
        return field, None
    if field.n <= max(1, cfg.prune_keep_min):
        return field, None
    activity = gaussian_activity(field.means, field.conics(cfg.aa_dilation),
                                 field.radii(cfg.sigma_cutoff, cfg.aa_dilation),
                                 H, W, cfg.render_chunk)
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
    colors = _nearest_image_colors(target, means)
    new = GaussianField(means, log_scales, rotations, colors)
    return field.append(new), k


@torch.no_grad()
def _add_from_residual(field: GaussianField, target: torch.Tensor, render_img: torch.Tensor,
                       cfg: FitConfig) -> tuple[GaussianField, int]:
    if cfg.split_count <= 0:
        return field, 0
    if cfg.max_gaussians is not None and field.n >= cfg.max_gaussians:
        return field, 0

    H, W = target.shape[:2]
    residual = (render_img - target).abs().mean(dim=2).reshape(-1)
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
    log_scales = torch.full((k, 2), math.log(max(base_scale, 0.35)),
                            device=target.device, dtype=target.dtype)
    rotations = torch.zeros(k, device=target.device, dtype=target.dtype)
    colors = target[y, x]
    return field.append(GaussianField(means, log_scales, rotations, colors)), k


def fit(field: GaussianField, target: torch.Tensor, cfg: FitConfig, verbose: bool = True) -> dict:
    H, W = target.shape[0], target.shape[1]
    field.trainable()
    opt = _make_optimizer(field, cfg)
    base_lrs = [g["lr"] for g in opt.param_groups]
    lo, hi = math.log(0.35), math.log(max(H, W))
    hist = {"iter": [], "psnr": [], "loss": [], "n_gaussians": [], "elapsed": []}
    targets = _target_list(cfg)
    iters_to_targets = {str(t): None for t in targets}
    start_time = time.time()

    for it in range(cfg.iters):
        factor = _lr_factor(cfg, it)
        for g, base in zip(opt.param_groups, base_lrs):
            g["lr"] = base * factor
        img = _render(field, cfg, H, W)
        pix = _pixel_loss(img, target, _loss_kind(cfg, it), cfg.charbonnier_eps)
        s = M.ssim(img, target)
        loss = (1 - cfg.ssim_weight) * pix + cfg.ssim_weight * (1 - s)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        log_now = it % cfg.log_every == 0 or it == cfg.iters - 1
        with torch.no_grad():
            field.log_scales.clamp_(lo, hi)
            if targets or log_now:
                p_now = M.psnr(img, target)
            for t in targets:
                key = str(t)
                if iters_to_targets[key] is None and p_now >= t:
                    iters_to_targets[key] = it

        keep = None
        added = 0
        if cfg.prune_every is not None and cfg.prune_every > 0 and (it + 1) % cfg.prune_every == 0:
            field, keep = _maybe_prune(field, cfg, H, W)
            if verbose and keep is not None:
                print(f"  prune {int((~keep).sum())} -> {field.n} gaussians")
        if cfg.split_every is not None and cfg.split_every > 0 and (it + 1) % cfg.split_every == 0:
            if cfg.split_mode == "duplicate":
                field, added = _split_from_residual(field, target, img, cfg)
            elif cfg.split_mode == "residual_add":
                field, added = _add_from_residual(field, target, img, cfg)
            else:
                raise ValueError(
                    f"unknown split_mode {cfg.split_mode!r}; expected duplicate or residual_add")
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
            hist["n_gaussians"].append(field.n)
            hist["elapsed"].append(time.time() - start_time)
            if verbose:
                print(f"  iter {it:5d}  psnr {p_now:6.2f}  loss {loss.item():.5f}")

    with torch.no_grad():
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
            "fit_seconds": time.time() - start_time,
        }
    return out
