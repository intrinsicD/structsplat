"""Per-image optimization (the baseline fitter every strategy is scored through).

loss = (1 - w) * L1 + w * (1 - SSIM), Adam with per-parameter-group learning rates. Records
PSNR history and iterations-to-target so the ablation can compare *quality at fixed budget*
and *convergence speed* (turn-3 prediction: flanking wins most at low budgets / early iters).
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


def _make_scheduler(opt, cfg: FitConfig):
    schedule = cfg.lr_schedule
    if schedule == "none" and cfg.lr_decay_every is not None and cfg.lr_decay_every > 0:
        schedule = "step"
    if schedule == "none":
        return None
    if schedule == "step":
        step = cfg.lr_decay_every if cfg.lr_decay_every is not None and cfg.lr_decay_every > 0 else 500
        return torch.optim.lr_scheduler.StepLR(opt, step_size=step, gamma=cfg.lr_decay_gamma)
    if schedule == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, cfg.iters))
    raise ValueError(f"unknown lr_schedule {cfg.lr_schedule!r}; expected none, step, or cosine")


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
    return render_field(field.means, field.conics(), field.colors, field.radii(cfg.sigma_cutoff),
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
def _maybe_prune(field: GaussianField, cfg: FitConfig, H: int, W: int) -> tuple[GaussianField, int]:
    if cfg.prune_min_activity <= 0.0:
        return field, 0
    if field.n <= max(1, cfg.prune_keep_min):
        return field, 0
    activity = gaussian_activity(field.means, field.conics(), field.radii(cfg.sigma_cutoff),
                                 H, W, cfg.render_chunk)
    keep = activity > cfg.prune_min_activity
    if int(keep.sum()) < cfg.prune_keep_min:
        topk = torch.topk(activity, k=min(cfg.prune_keep_min, field.n)).indices
        keep = torch.zeros_like(activity, dtype=torch.bool)
        keep[topk] = True
    removed = field.n - int(keep.sum())
    if removed <= 0:
        return field, 0
    return field.subset(keep), removed


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
    means = torch.stack([x.to(target.dtype) + 0.5, y.to(target.dtype) + 0.5], dim=1) + offsets
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
    scheduler = _make_scheduler(opt, cfg)
    lo, hi = math.log(0.35), math.log(max(H, W))
    hist = {"iter": [], "psnr": [], "loss": [], "n_gaussians": [], "elapsed": []}
    targets = _target_list(cfg)
    iters_to_targets = {str(t): None for t in targets}
    start_time = time.time()

    for it in range(cfg.iters):
        img = _render(field, cfg, H, W)
        pix = _pixel_loss(img, target, _loss_kind(cfg, it), cfg.charbonnier_eps)
        s = M.ssim(img, target)
        loss = (1 - cfg.ssim_weight) * pix + cfg.ssim_weight * (1 - s)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if scheduler is not None:
            scheduler.step()
        with torch.no_grad():
            field.log_scales.clamp_(lo, hi)
            p_now = M.psnr(img, target)
            for t in targets:
                key = str(t)
                if iters_to_targets[key] is None and p_now >= t:
                    iters_to_targets[key] = it

        changed = False
        if cfg.prune_every is not None and cfg.prune_every > 0 and (it + 1) % cfg.prune_every == 0:
            field, removed = _maybe_prune(field, cfg, H, W)
            changed = changed or removed > 0
            if verbose and removed > 0:
                print(f"  prune {removed} -> {field.n} gaussians")
        if cfg.split_every is not None and cfg.split_every > 0 and (it + 1) % cfg.split_every == 0:
            if cfg.split_mode == "duplicate":
                field, added = _split_from_residual(field, target, img, cfg)
            elif cfg.split_mode == "residual_add":
                field, added = _add_from_residual(field, target, img, cfg)
            else:
                raise ValueError(f"unknown split_mode {cfg.split_mode!r}; expected duplicate or residual_add")
            changed = changed or added > 0
            if verbose and added > 0:
                print(f"  split +{added} -> {field.n} gaussians")
        if changed:
            field.trainable()
            opt = _make_optimizer(field, cfg)
            scheduler = _make_scheduler(opt, cfg)

        if it % cfg.log_every == 0 or it == cfg.iters - 1:
            with torch.no_grad():
                p = M.psnr(img, target)
            hist["iter"].append(it)
            hist["psnr"].append(p)
            hist["loss"].append(float(loss))
            hist["n_gaussians"].append(field.n)
            hist["elapsed"].append(time.time() - start_time)
            if verbose:
                print(f"  iter {it:5d}  psnr {p:6.2f}  loss {float(loss):.5f}")

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
