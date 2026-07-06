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
    _EPS,
    _element_budget,
    _flat_tile_slices,
    _support_weight,
    _tile_bounds,
    _tile_coords,
    gaussian_activity,
    render_field,
)
from . import metrics as M

# Renderer modes that accumulate additively (ADR-0006): a new/split Gaussian stacks on the
# existing accumulation, so it must carry the residual color, not the full target color.
_ADDITIVE_RENDERERS = (
    "additive", "cuda_additive", "cuda_tiled_additive", "gsplat", "cuda_gsplat",
)
# Shared lower bound on densified Gaussian scales (px); was duplicated as bare 0.35 literals.
_MIN_DENSIFY_SCALE = 0.35


class _Adan(torch.optim.Optimizer):
    """Small local Adan implementation with per-parameter tensor state."""

    def __init__(self, params, betas=(0.98, 0.92, 0.99), eps: float = 1e-8):
        defaults = {"betas": betas, "eps": eps}
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            beta1, beta2, beta3 = group["betas"]
            eps = group["eps"]
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.detach()
                if grad.is_sparse:
                    raise RuntimeError("Adan does not support sparse gradients")
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_diff"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                    state["prev_grad"] = grad.clone()
                    grad_diff = torch.zeros_like(grad)
                else:
                    grad_diff = grad - state["prev_grad"]
                state["step"] += 1

                exp_avg = state["exp_avg"]
                exp_avg_diff = state["exp_avg_diff"]
                exp_avg_sq = state["exp_avg_sq"]
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_diff.mul_(beta2).add_(grad_diff, alpha=1.0 - beta2)
                update_grad = grad + beta2 * grad_diff
                exp_avg_sq.mul_(beta3).addcmul_(update_grad, update_grad, value=1.0 - beta3)

                step = int(state["step"])
                m_hat = exp_avg / (1.0 - beta1 ** step)
                v_hat = exp_avg_diff / (1.0 - beta2 ** step)
                n_hat = exp_avg_sq / (1.0 - beta3 ** step)
                p.addcdiv_(m_hat + beta2 * v_hat, n_hat.sqrt().add_(eps), value=-lr)
                state["prev_grad"].copy_(grad)
        return loss


def _make_optimizer(field: GaussianField, cfg: FitConfig):
    groups = field.parameter_groups(cfg.lr_means, cfg.lr_scales, cfg.lr_rot, cfg.lr_color,
                                    cfg.lr_opacity)
    if cfg.optimizer == "adam":
        return torch.optim.Adam(groups)
    if cfg.optimizer == "adamw":
        return torch.optim.AdamW(groups)
    if cfg.optimizer == "adan":
        return _Adan(groups)
    raise ValueError(f"unknown optimizer {cfg.optimizer!r}; expected adam, adamw, or adan")


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
                      keep: torch.Tensor | None, n_new: int,
                      reset_idx: torch.Tensor | None = None):
    """Fresh optimizer for a restructured field, carrying per-Gaussian optimizer buffers.

    Without this, every prune/split resets exp_avg/exp_avg_sq for ALL Gaussians and the
    loss spikes while Adam re-estimates curvature. Surviving rows keep their moments; new
    rows start at zero and share the tensor's inherited `step` (Adam keeps one step per
    parameter tensor), so their first updates behave like a normal warm Adam step. Handles
    any number of parameter groups (4, or 5 with opacity), Adam/AdamW, and Adan.
    """
    opt_new = _make_optimizer(field, cfg)
    for g_old, g_new in zip(opt_old.param_groups, opt_new.param_groups):
        p_old, p_new = g_old["params"][0], g_new["params"][0]
        st = opt_old.state.get(p_old)
        if not st:
            continue
        carried = {}
        for key, value in st.items():
            if torch.is_tensor(value):
                out = value.detach().clone()
                if out.shape == p_old.shape:
                    if keep is not None:
                        out = out[keep]
                    if n_new > 0:
                        pad = out.new_zeros((n_new,) + out.shape[1:])
                        out = torch.cat([out, pad], dim=0)
                    if reset_idx is not None and reset_idx.numel() > 0:
                        ridx = reset_idx.to(device=out.device, dtype=torch.long)
                        out[ridx] = 0
                    out = out.contiguous()
                carried[key] = out
            else:
                carried[key] = value
        opt_new.state[p_new] = carried
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
                        scales=field.scales(), rotations=field.rotations,
                        support_fade=cfg.support_fade, sigma_cutoff=cfg.sigma_cutoff,
                        color_grads=field.color_grads)


def _color_solve_enabled(cfg: FitConfig) -> bool:
    return cfg.color_solve_every is not None and cfg.color_solve_every > 0


def _ensure_color_basis(field: GaussianField, cfg: FitConfig) -> GaussianField:
    if cfg.color_basis == "affine":
        return field.with_affine_colors()
    return field


@torch.no_grad()
def _normalized_color_denominator(field: GaussianField, cfg: FitConfig,
                                  H: int, W: int) -> torch.Tensor:
    dev, dt = field.means.device, field.means.dtype
    den = torch.zeros(H * W, 1, device=dev, dtype=dt)
    means = field.means.detach()
    conics = field.conics(cfg.aa_dilation).detach()
    radii = field.radii(cfg.sigma_cutoff, cfg.aa_dilation)
    opacities = field.opacity_values()
    if opacities is not None:
        opacities = opacities.detach()
    x0, y0, Tx, n = _tile_bounds(means, radii, H, W)
    budget = _element_budget(cfg.render_chunk)
    for s, e in _flat_tile_slices(n, budget):
        gid, px, py = _tile_coords(x0, y0, Tx, n, s, e, dev)
        dx = px.to(dt) - means[gid, 0]
        dy = py.to(dt) - means[gid, 1]
        a, b, c = conics[gid, 0], conics[gid, 1], conics[gid, 2]
        q = a * dx * dx + 2.0 * b * dx * dy + c * dy * dy
        w = _support_weight(q, cfg.sigma_cutoff, cfg.support_fade)
        if opacities is not None:
            w = w * opacities[gid]
        den.index_add_(0, py * W + px, w[:, None])
    return den


@torch.no_grad()
def _normalized_color_basis_apply(field: GaussianField, colors: torch.Tensor, cfg: FitConfig,
                                  H: int, W: int, den: torch.Tensor) -> torch.Tensor:
    dev, dt = field.means.device, field.means.dtype
    out = torch.zeros(H * W, colors.shape[1], device=dev, dtype=dt)
    means = field.means.detach()
    conics = field.conics(cfg.aa_dilation).detach()
    radii = field.radii(cfg.sigma_cutoff, cfg.aa_dilation)
    opacities = field.opacity_values()
    if opacities is not None:
        opacities = opacities.detach()
    x0, y0, Tx, n = _tile_bounds(means, radii, H, W)
    budget = _element_budget(cfg.render_chunk)
    for s, e in _flat_tile_slices(n, budget):
        gid, px, py = _tile_coords(x0, y0, Tx, n, s, e, dev)
        dx = px.to(dt) - means[gid, 0]
        dy = py.to(dt) - means[gid, 1]
        a, b, c = conics[gid, 0], conics[gid, 1], conics[gid, 2]
        q = a * dx * dx + 2.0 * b * dx * dy + c * dy * dy
        w = _support_weight(q, cfg.sigma_cutoff, cfg.support_fade)
        if opacities is not None:
            w = w * opacities[gid]
        flat = py * W + px
        basis = w[:, None] / (den[flat] + _EPS)
        out.index_add_(0, flat, basis * colors[gid])
    return out.view(H, W, colors.shape[1])


@torch.no_grad()
def _normalized_color_basis_transpose(field: GaussianField, image: torch.Tensor,
                                      cfg: FitConfig, H: int, W: int,
                                      den: torch.Tensor) -> torch.Tensor:
    dev, dt = field.means.device, field.means.dtype
    out = torch.zeros(field.n, image.shape[-1], device=dev, dtype=dt)
    means = field.means.detach()
    conics = field.conics(cfg.aa_dilation).detach()
    radii = field.radii(cfg.sigma_cutoff, cfg.aa_dilation)
    opacities = field.opacity_values()
    if opacities is not None:
        opacities = opacities.detach()
    x0, y0, Tx, n = _tile_bounds(means, radii, H, W)
    budget = _element_budget(cfg.render_chunk)
    for s, e in _flat_tile_slices(n, budget):
        gid, px, py = _tile_coords(x0, y0, Tx, n, s, e, dev)
        dx = px.to(dt) - means[gid, 0]
        dy = py.to(dt) - means[gid, 1]
        a, b, c = conics[gid, 0], conics[gid, 1], conics[gid, 2]
        q = a * dx * dx + 2.0 * b * dx * dy + c * dy * dy
        w = _support_weight(q, cfg.sigma_cutoff, cfg.support_fade)
        if opacities is not None:
            w = w * opacities[gid]
        flat = py * W + px
        basis = w[:, None] / (den[flat] + _EPS)
        out.index_add_(0, gid, basis * image[py, px].to(dt))
    return out


@torch.no_grad()
def _solve_colors_normalized(field: GaussianField, target: torch.Tensor, cfg: FitConfig,
                             H: int, W: int) -> dict[str, float | int]:
    """Solve fixed-geometry normalized-renderer RGB colors with implicit CG.

    The normalized renderer is linear in colors once means/scales/rotations/opacities are fixed:
    render = A c. We solve `(A.T A + lambda I)c = A.T target + lambda c_prev` for the three RGB
    channels simultaneously, without materializing the dense pixel-by-Gaussian matrix.
    """
    if cfg.renderer != "normalized":
        raise ValueError(
            "color_solve_every currently supports renderer='normalized' only; "
            f"got {cfg.renderer!r}"
        )
    if field.color_grads is not None:
        raise ValueError(
            "color_solve_every currently supports color_basis='constant' only; "
            "affine color coefficients are optimized with Adam"
        )
    if field.n == 0:
        return {"iterations": 0, "relative_residual": 0.0}

    den = _normalized_color_denominator(field, cfg, H, W)
    lam = float(cfg.color_solve_lambda)
    x0 = field.colors.detach().clone()
    b = _normalized_color_basis_transpose(field, target, cfg, H, W, den)
    if lam > 0.0:
        b = b + lam * x0

    def normal_matvec(x: torch.Tensor) -> torch.Tensor:
        ax = _normalized_color_basis_apply(field, x, cfg, H, W, den)
        atax = _normalized_color_basis_transpose(field, ax, cfg, H, W, den)
        if lam > 0.0:
            atax = atax + lam * x
        return atax

    x = x0.clone()
    r = b - normal_matvec(x)
    p = r.clone()
    rs = (r * r).sum(dim=0)
    b_norm = torch.sqrt((b * b).sum(dim=0)).clamp_min(1e-12)
    rel = torch.sqrt(rs) / b_norm
    iterations = 0
    eps = torch.finfo(field.colors.dtype).eps
    for _ in range(int(cfg.color_solve_maxiter)):
        if bool(torch.all(rel <= 1e-5)):
            break
        ap = normal_matvec(p)
        denom = (p * ap).sum(dim=0)
        valid = denom.abs() > eps
        if not bool(torch.any(valid)):
            break
        safe_denom = torch.where(valid, denom, torch.ones_like(denom))
        alpha = torch.where(valid, rs / safe_denom, torch.zeros_like(rs))
        x = x + p * alpha
        r = r - ap * alpha
        rs_next = (r * r).sum(dim=0)
        safe_rs = torch.where(rs > eps, rs, torch.ones_like(rs))
        beta = torch.where(valid, rs_next / safe_rs, torch.zeros_like(rs))
        p = r + p * beta
        rs = rs_next
        rel = torch.sqrt(rs) / b_norm
        iterations += 1

    field.colors.copy_(x)
    return {
        "iterations": iterations,
        "relative_residual": float(rel.max().detach().cpu()),
    }


@torch.no_grad()
def _reset_optimizer_state_for_param(opt: torch.optim.Optimizer, param: torch.Tensor) -> None:
    state = opt.state.get(param)
    if not state:
        return
    for value in state.values():
        if torch.is_tensor(value) and value.shape == param.shape:
            value.zero_()


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
def _residual_add_colors(target: torch.Tensor, render_img: torch.Tensor, y: torch.Tensor,
                         x: torch.Tensor, cfg: FitConfig) -> torch.Tensor:
    if cfg.renderer in _ADDITIVE_RENDERERS:
        # Additive renderers stack raw contributions, so children should carry only the missing
        # residual. This preserves the FIT-002 additive semantics regardless of split_color_init.
        return (target - render_img)[y, x]
    if cfg.split_color_init == "residual":
        # Normalized rendering averages colors by weight. A corrective child must sit on the
        # other side of the current render, not at the current target color, to pull the weighted
        # average toward the target in sparse high-error regions.
        return target[y, x] + (target - render_img)[y, x]
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
    budget = _element_budget(cfg.render_chunk)
    for s, e in _flat_tile_slices(n, budget):
        gid, px, py = _tile_coords(x0, y0, Tx, n, s, e, dev)
        dx = px.to(dt) - means[gid, 0]
        dy = py.to(dt) - means[gid, 1]
        a, b, c = conics[gid, 0], conics[gid, 1], conics[gid, 2]
        q = a * dx * dx + 2.0 * b * dx * dy + c * dy * dy
        w = torch.exp(-0.5 * q)
        if cfg.support_fade:
            w = torch.clamp(w - math.exp(-0.5 * cfg.sigma_cutoff ** 2), min=0.0)
        score.index_add_(0, gid, w * residual[py, px].to(dt))
        weight.index_add_(0, gid, w)
    return score / (weight + 1e-8)


def _luma(img: torch.Tensor) -> torch.Tensor:
    return 0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]


@torch.no_grad()
def _spaced_topk_pixels(scores: torch.Tensor, W: int, k: int, min_spacing: float,
                        oversample: float) -> torch.Tensor:
    if k <= 0:
        return scores.new_empty((0,), dtype=torch.long)
    if min_spacing <= 0.0:
        return torch.topk(scores, k=k).indices

    total = int(scores.numel())
    selected: list[int] = []
    blocked: set[int] = set()
    candidate_count = min(total, max(k, int(math.ceil(k * oversample))))
    min_d2 = float(min_spacing * min_spacing)
    while True:
        cand = torch.topk(scores, k=candidate_count).indices.detach().cpu().tolist()
        for idx in cand:
            if idx in blocked:
                continue
            y = idx // W
            x = idx - y * W
            ok = True
            for prev in selected:
                py = prev // W
                px = prev - py * W
                dx = x - px
                dy = y - py
                if dx * dx + dy * dy < min_d2:
                    ok = False
                    break
            blocked.add(idx)
            if ok:
                selected.append(idx)
                if len(selected) == k:
                    return torch.as_tensor(selected, device=scores.device, dtype=torch.long)
        if len(selected) >= k or candidate_count >= total:
            break
        candidate_count = min(total, max(candidate_count + 1, candidate_count * 2))

    # If the requested radius cannot provide K separated pixels, fill the remaining capacity with
    # the best unused pixels. This preserves exact growth while spacing whenever possible.
    if len(selected) < k:
        for idx in torch.topk(scores, k=total).indices.detach().cpu().tolist():
            if idx not in selected:
                selected.append(idx)
                if len(selected) == k:
                    break
    return torch.as_tensor(selected[:k], device=scores.device, dtype=torch.long)


@torch.no_grad()
def _residual_candidate_pixels(score_map: torch.Tensor, k: int, min_spacing: float,
                               oversample: float, downsample: int = 1) -> torch.Tensor:
    """Select high-error pixels, optionally through a coarse max-pooled residual pyramid.

    The coarse path keeps relocation candidate search cheap for large images. It first chooses
    high-error coarse cells, then snaps each cell to its best full-resolution pixel. Spacing-NMS
    remains exact full-resolution behavior because the current spacing selector already has the
    intended fallback semantics and is normally disabled for relocation benchmarks.
    """
    H, W = score_map.shape
    scores = score_map.reshape(-1)
    if k <= 0:
        return scores.new_empty((0,), dtype=torch.long)
    k = min(int(k), int(scores.numel()))
    if k <= 0:
        return scores.new_empty((0,), dtype=torch.long)
    if downsample <= 1 or min_spacing > 0.0:
        return _spaced_topk_pixels(scores, W, k, min_spacing, oversample)

    ds = max(1, min(int(downsample), H, W))
    pooled = F.max_pool2d(score_map[None, None], kernel_size=ds, stride=ds, ceil_mode=True)[0, 0]
    coarse_w = pooled.shape[1]
    coarse_scores = pooled.reshape(-1)
    candidate_count = min(
        int(coarse_scores.numel()),
        max(k, int(math.ceil(k * max(1.0, oversample)))),
    )
    coarse_idx = torch.topk(coarse_scores, k=candidate_count).indices.detach().cpu().tolist()

    selected: list[int] = []
    seen: set[int] = set()
    for cidx in coarse_idx:
        cy = cidx // coarse_w
        cx = cidx - cy * coarse_w
        y0 = cy * ds
        x0 = cx * ds
        y1 = min(y0 + ds, H)
        x1 = min(x0 + ds, W)
        patch = score_map[y0:y1, x0:x1]
        if patch.numel() == 0:
            continue
        local = int(torch.argmax(patch.reshape(-1)).detach().cpu())
        yy = y0 + local // (x1 - x0)
        xx = x0 + local % (x1 - x0)
        idx = yy * W + xx
        if idx not in seen:
            selected.append(idx)
            seen.add(idx)
            if len(selected) == k:
                return torch.as_tensor(selected, device=scores.device, dtype=torch.long)

    # Degenerate case: coarse cells were fewer than requested. Fill by exact full-res order while
    # preserving the pixels already selected through the coarse pass.
    for idx in torch.topk(scores, k=int(scores.numel())).indices.detach().cpu().tolist():
        if idx not in seen:
            selected.append(idx)
            seen.add(idx)
            if len(selected) == k:
                break
    return torch.as_tensor(selected[:k], device=scores.device, dtype=torch.long)


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
def _abs_laplacian(img: torch.Tensor) -> torch.Tensor:
    gray = _luma(img)
    padded = F.pad(gray[None, None], (1, 1, 1, 1), mode="replicate")[0, 0]
    center = padded[1:-1, 1:-1]
    lap = padded[1:-1, :-2] + padded[1:-1, 2:] + padded[:-2, 1:-1] + padded[2:, 1:-1] \
        - 4.0 * center
    return lap.abs()


@torch.no_grad()
def _freq_violation_scores(field: GaussianField, target: torch.Tensor,
                           render_img: torch.Tensor, cfg: FitConfig
                           ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    H, W = target.shape[:2]
    dev, dt = field.means.device, field.means.dtype
    residual = (render_img - target).abs().mean(dim=2)
    gx, gy = _image_gradients(target)
    grad_mag = torch.sqrt(gx * gx + gy * gy)
    freq = grad_mag + 0.5 * _abs_laplacian(target)
    freq_ref = torch.quantile(freq.reshape(-1), 0.95)
    if float(freq_ref.detach().cpu()) <= 1e-8:
        freq_n = torch.zeros_like(freq)
    else:
        freq_n = torch.clamp(freq / freq_ref.clamp_min(1e-8), 0.0, 1.0)

    x = torch.clamp(torch.round(field.means[:, 0]).long(), 0, W - 1)
    y = torch.clamp(torch.round(field.means[:, 1]).long(), 0, H - 1)
    f = freq_n[y, x].to(dt)
    residual_sample = residual[y, x].to(dt)
    residual_ref = torch.quantile(residual.reshape(-1), 0.9).to(dt).clamp_min(1e-8)
    residual_n = torch.clamp(residual_sample / residual_ref, 0.0, 4.0)

    gvec = torch.stack([gx[y, x].to(dt), gy[y, x].to(dt)], dim=1)
    gnorm = torch.linalg.norm(gvec, dim=1, keepdim=True)
    gdir = torch.where(gnorm > 1e-8, gvec / gnorm.clamp_min(1e-8), torch.zeros_like(gvec))

    theta = field.rotations.detach()
    c, s = torch.cos(theta), torch.sin(theta)
    axis0 = torch.stack([c, s], dim=1)
    axis1 = torch.stack([-s, c], dim=1)
    align0 = torch.abs((axis0 * gdir).sum(dim=1))
    align1 = torch.abs((axis1 * gdir).sum(dim=1))

    base_limit = max(
        math.sqrt((H * W) / max(field.n, 1)) * cfg.split_scale,
        _MIN_DENSIFY_SCALE,
    )
    edge_limit = _MIN_DENSIFY_SCALE + (base_limit - _MIN_DENSIFY_SCALE) * (1.0 - f)
    tangent_limit = edge_limit * 3.0
    limit0 = edge_limit * align0 + tangent_limit * (1.0 - align0)
    limit1 = edge_limit * align1 + tangent_limit * (1.0 - align1)

    scales = field.scales().detach()
    violation0 = scales[:, 0] / limit0.clamp_min(_MIN_DENSIFY_SCALE) - 1.0
    violation1 = scales[:, 1] / limit1.clamp_min(_MIN_DENSIFY_SCALE) - 1.0
    violation = torch.maximum(violation0, violation1)
    split_axis = torch.where(violation0 >= violation1, 0, 1).to(device=dev, dtype=torch.long)
    score = torch.clamp(violation, min=0.0) * f * (0.25 + residual_n)
    return score, split_axis, {
        "violation0": violation0,
        "violation1": violation1,
        "freq": f,
        "residual": residual_sample,
    }


@torch.no_grad()
def _maybe_prune(field: GaussianField, cfg: FitConfig, H: int,
                 W: int) -> tuple[GaussianField, torch.Tensor | None]:
    if cfg.prune_min_activity <= 0.0:
        return field, None
    if field.n <= max(1, cfg.prune_keep_min):
        return field, None
    activity = gaussian_activity(field.means, field.conics(cfg.aa_dilation),
                                 field.radii(cfg.sigma_cutoff, cfg.aa_dilation),
                                 H, W, cfg.render_chunk, cfg.support_fade, cfg.sigma_cutoff)
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


def _logit(p: torch.Tensor) -> torch.Tensor:
    p = torch.clamp(p, 1e-4, 1.0 - 1e-4)
    return torch.log(p / (1.0 - p))


@torch.no_grad()
def _major_axis(scales: torch.Tensor, theta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    c, s = torch.cos(theta), torch.sin(theta)
    x_axis = torch.stack([c, s], dim=1)
    y_axis = torch.stack([-s, c], dim=1)
    use_x = scales[:, 0] >= scales[:, 1]
    direction = torch.where(use_x[:, None], x_axis, y_axis)
    major = torch.where(use_x, scales[:, 0], scales[:, 1])
    return direction, major


@torch.no_grad()
def _fp_duplicate_indices(field: GaussianField, idx: torch.Tensor, cfg: FitConfig,
                          H: int, W: int, split_axis: torch.Tensor | None = None
                          ) -> tuple[GaussianField, int]:
    k = int(idx.numel())
    if k <= 0:
        return field, 0

    means = field.means.detach().clone()
    log_scales = field.log_scales.detach().clone()
    rotations = field.rotations.detach().clone()
    colors = field.colors.detach().clone()
    color_grads = None if field.color_grads is None else field.color_grads.detach().clone()
    scales = torch.exp(log_scales[idx])
    theta = rotations[idx]
    if split_axis is None:
        direction, offset_scale = _major_axis(scales, theta)
    else:
        axis = split_axis.to(device=scales.device, dtype=torch.long)
        c, s = torch.cos(theta), torch.sin(theta)
        x_axis = torch.stack([c, s], dim=1)
        y_axis = torch.stack([-s, c], dim=1)
        use_x = axis == 0
        direction = torch.where(use_x[:, None], x_axis, y_axis)
        offset_scale = torch.where(use_x, scales[:, 0], scales[:, 1])
    offset = direction * offset_scale[:, None]

    child_means = means[idx].clone() + offset
    means[idx] = means[idx] - offset
    means[idx, 0].clamp_(0, W - 1)
    means[idx, 1].clamp_(0, H - 1)
    child_means[:, 0].clamp_(0, W - 1)
    child_means[:, 1].clamp_(0, H - 1)

    child_log_scales = log_scales[idx].clone() + math.log(cfg.split_scale)
    log_scales[idx] = _clamp_new_log_scales(log_scales[idx] + math.log(cfg.split_scale),
                                            None if field.scale_max is None
                                            else field.scale_max.detach()[idx])
    child_log_scales = _clamp_new_log_scales(
        child_log_scales,
        None if field.scale_max is None else field.scale_max.detach()[idx].clone(),
    )
    child_rotations = theta.clone()
    child_colors = colors[idx].clone()

    if field.opacities is None:
        opacities = torch.full((field.n,), 10.0, device=means.device, dtype=means.dtype)
        parent_opacity = torch.ones(k, device=means.device, dtype=means.dtype)
    else:
        opacities = field.opacities.detach().clone()
        parent_opacity = torch.sigmoid(opacities[idx])
    split_opacity = parent_opacity * 0.5
    opacities[idx] = _logit(split_opacity)
    child_opacities = _logit(split_opacity)

    scale_max = None if field.scale_max is None else field.scale_max.detach().clone()
    child_scale_max = None if field.scale_max is None else field.scale_max.detach()[idx].clone()
    child_color_grads = None if color_grads is None else color_grads[idx].clone()
    child = GaussianField(child_means, child_log_scales, child_rotations, child_colors,
                          child_opacities, child_scale_max, child_color_grads)
    return GaussianField(
        means, log_scales, rotations, colors, opacities, scale_max, color_grads
    ).append(child), k


@torch.no_grad()
def _moment_preserving_duplicate_indices(field: GaussianField, idx: torch.Tensor,
                                         cfg: FitConfig, H: int, W: int,
                                         split_axis: torch.Tensor | None = None
                                         ) -> tuple[GaussianField, int]:
    k = int(idx.numel())
    if k <= 0:
        return field, 0

    means = field.means.detach().clone()
    log_scales = field.log_scales.detach().clone()
    rotations = field.rotations.detach().clone()
    colors = field.colors.detach().clone()
    color_grads = None if field.color_grads is None else field.color_grads.detach().clone()
    scales = torch.exp(log_scales[idx])
    theta = rotations[idx]
    if split_axis is None:
        use_x = scales[:, 0] >= scales[:, 1]
    else:
        use_x = split_axis.to(device=scales.device, dtype=torch.long) == 0
    c, s = torch.cos(theta), torch.sin(theta)
    x_axis = torch.stack([c, s], dim=1)
    y_axis = torch.stack([-s, c], dim=1)
    direction = torch.where(use_x[:, None], x_axis, y_axis)
    axis_scale = torch.where(use_x, scales[:, 0], scales[:, 1])
    shrink = min(max(float(cfg.split_scale), 1e-4), 1.0 - 1e-4)
    offset_scale = axis_scale * math.sqrt(max(1.0 - shrink * shrink, 0.0))
    offset = direction * offset_scale[:, None]

    child_means = means[idx].clone() + offset
    means[idx] = means[idx] - offset
    means[idx, 0].clamp_(0, W - 1)
    means[idx, 1].clamp_(0, H - 1)
    child_means[:, 0].clamp_(0, W - 1)
    child_means[:, 1].clamp_(0, H - 1)

    child_log_scales = log_scales[idx].clone()
    parent_log_scales = log_scales[idx].clone()
    axis_idx = torch.where(use_x, 0, 1)
    parent_log_scales[torch.arange(k, device=axis_idx.device), axis_idx] += math.log(shrink)
    child_log_scales[torch.arange(k, device=axis_idx.device), axis_idx] += math.log(shrink)
    log_scales[idx] = _clamp_new_log_scales(
        parent_log_scales,
        None if field.scale_max is None else field.scale_max.detach()[idx],
    )
    child_log_scales = _clamp_new_log_scales(
        child_log_scales,
        None if field.scale_max is None else field.scale_max.detach()[idx].clone(),
    )
    child_rotations = theta.clone()
    child_colors = colors[idx].clone()

    if field.opacities is None:
        opacities = torch.full((field.n,), 10.0, device=means.device, dtype=means.dtype)
        parent_opacity = torch.ones(k, device=means.device, dtype=means.dtype)
    else:
        opacities = field.opacities.detach().clone()
        parent_opacity = torch.sigmoid(opacities[idx])
    split_opacity = parent_opacity * 0.5
    opacities[idx] = _logit(split_opacity)
    child_opacities = _logit(split_opacity)

    scale_max = None if field.scale_max is None else field.scale_max.detach().clone()
    child_scale_max = None if field.scale_max is None else field.scale_max.detach()[idx].clone()
    child_color_grads = None if color_grads is None else color_grads[idx].clone()
    child = GaussianField(child_means, child_log_scales, child_rotations, child_colors,
                          child_opacities, child_scale_max, child_color_grads)
    return GaussianField(
        means, log_scales, rotations, colors, opacities, scale_max, color_grads
    ).append(child), k


@torch.no_grad()
def _ranked_wave_scores(field: GaussianField, residual: torch.Tensor,
                        cfg: FitConfig) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    H, W = residual.shape
    support = _support_residual_scores(field, residual, cfg)
    activity = gaussian_activity(
        field.means, field.conics(cfg.aa_dilation),
        field.radii(cfg.sigma_cutoff, cfg.aa_dilation), H, W, cfg.render_chunk,
        cfg.support_fade, cfg.sigma_cutoff,
    )
    opac = field.opacity_values()
    if opac is not None:
        activity = activity * opac
    footprint = torch.exp(field.log_scales.detach()).prod(dim=1)

    def norm(x: torch.Tensor) -> torch.Tensor:
        return x / torch.clamp(torch.quantile(x.detach(), 0.9), min=1e-8)

    support_n = norm(support)
    activity_n = norm(activity)
    footprint_n = norm(footprint)
    score = support_n * torch.sqrt(torch.clamp(activity_n, min=0.0)) \
        * torch.sqrt(torch.clamp(footprint_n, min=0.0))
    return score, {
        "residual_support": support,
        "activity": activity,
        "footprint": footprint,
    }


@torch.no_grad()
def _ranked_wave_from_residual(field: GaussianField, target: torch.Tensor,
                               render_img: torch.Tensor,
                               cfg: FitConfig) -> tuple[GaussianField, int, dict[str, float]]:
    if cfg.split_count <= 0:
        return field, 0, {}
    if cfg.max_gaussians is not None and field.n >= cfg.max_gaussians:
        return field, 0, {}
    H, W = target.shape[:2]
    residual = (render_img - target).abs().mean(dim=2)
    room = field.n if cfg.max_gaussians is None else max(0, cfg.max_gaussians - field.n)
    k = min(cfg.split_count, field.n, room)
    if k <= 0:
        return field, 0, {}
    score, components = _ranked_wave_scores(field, residual, cfg)
    idx = torch.topk(score, k=k).indices
    grown, added = _fp_duplicate_indices(field, idx, cfg, H, W)
    stats = {
        "score_mean": float(score[idx].mean().detach().cpu()),
        "residual_support_mean": float(components["residual_support"][idx].mean().detach().cpu()),
        "activity_mean": float(components["activity"][idx].mean().detach().cpu()),
        "footprint_mean": float(components["footprint"][idx].mean().detach().cpu()),
    }
    return grown, added, stats


@torch.no_grad()
def _freq_violation_from_residual(field: GaussianField, target: torch.Tensor,
                                  render_img: torch.Tensor,
                                  cfg: FitConfig) -> tuple[GaussianField, int, dict[str, float]]:
    if cfg.split_count <= 0:
        return field, 0, {}
    if cfg.max_gaussians is not None and field.n >= cfg.max_gaussians:
        return field, 0, {}
    H, W = target.shape[:2]
    room = field.n if cfg.max_gaussians is None else max(0, cfg.max_gaussians - field.n)
    k = min(cfg.split_count, field.n, room)
    if k <= 0:
        return field, 0, {}
    score, split_axis, components = _freq_violation_scores(field, target, render_img, cfg)
    idx = torch.topk(score, k=k).indices
    grown, added = _fp_duplicate_indices(field, idx, cfg, H, W, split_axis=split_axis[idx])
    axis_sel = split_axis[idx]
    stats = {
        "freq_violation_score_mean": float(score[idx].mean().detach().cpu()),
        "freq_violation_score_max": float(score[idx].max().detach().cpu()),
        "freq_violation_axis0_count": int((axis_sel == 0).sum().detach().cpu()),
        "freq_violation_axis1_count": int((axis_sel == 1).sum().detach().cpu()),
        "freq_violation_freq_mean": float(components["freq"][idx].mean().detach().cpu()),
    }
    return grown, added, stats


@torch.no_grad()
def _absgrad_wave_from_scores(field: GaussianField, grad_scores: torch.Tensor,
                              cfg: FitConfig, H: int, W: int
                              ) -> tuple[GaussianField, int, dict[str, float], torch.Tensor]:
    if cfg.split_count <= 0:
        return field, 0, {}, grad_scores
    if cfg.max_gaussians is not None and field.n >= cfg.max_gaussians:
        return field, 0, {}, grad_scores
    room = field.n if cfg.max_gaussians is None else max(0, cfg.max_gaussians - field.n)
    k = min(cfg.split_count, field.n, room)
    if k <= 0:
        return field, 0, {}, grad_scores
    scores = grad_scores.detach()
    idx = torch.topk(scores, k=k).indices
    grown, added = _fp_duplicate_indices(field, idx, cfg, H, W)
    stats = {
        "absgrad_score_mean": float(scores[idx].mean().detach().cpu()),
        "absgrad_score_max": float(scores[idx].max().detach().cpu()),
    }
    # Do not immediately re-split the same parent only because its pre-split score was high.
    carried = scores.clone()
    carried[idx] = 0
    if added > 0:
        carried = torch.cat([carried, carried.new_zeros(added)], dim=0)
    return grown, added, stats, carried


@torch.no_grad()
def _relocate_from_residual(field: GaussianField, target: torch.Tensor,
                            render_img: torch.Tensor,
                            cfg: FitConfig
                            ) -> tuple[GaussianField, int, torch.Tensor | None,
                                       dict[str, float | str]]:
    if cfg.relocate_count <= 0 or field.n <= 0:
        return field, 0, None, {}
    H, W = target.shape[:2]
    k = min(cfg.relocate_count, field.n, int(H * W))
    activity = gaussian_activity(
        field.means, field.conics(cfg.aa_dilation),
        field.radii(cfg.sigma_cutoff, cfg.aa_dilation), H, W, cfg.render_chunk,
        cfg.support_fade, cfg.sigma_cutoff,
    )
    opac = field.opacity_values()
    if opac is not None:
        activity = activity * opac
    low_idx = torch.topk(-activity, k=k).indices

    residual_map = (render_img - target).abs().mean(dim=2)
    base_scale = math.sqrt((H * W) / max(field.n, 1)) * cfg.split_scale
    min_spacing = cfg.split_min_spacing * max(base_scale, _MIN_DENSIFY_SCALE)
    pix = _residual_candidate_pixels(
        residual_map, k, min_spacing, cfg.split_oversample, cfg.relocate_residual_downsample
    )
    residual = residual_map.reshape(-1)
    y = torch.div(pix, W, rounding_mode="floor")
    x = pix - y * W
    means = field.means.detach().clone()
    log_scales = field.log_scales.detach().clone()
    rotations = field.rotations.detach().clone()
    colors = field.colors.detach().clone()
    color_grads = None if field.color_grads is None else field.color_grads.detach().clone()
    new_means = torch.stack([x.to(target.dtype), y.to(target.dtype)], dim=1)
    means[low_idx] = new_means
    base = max(base_scale, _MIN_DENSIFY_SCALE)
    log_scales[low_idx] = math.log(base)
    rotations[low_idx] = 0.0
    if cfg.renderer in _ADDITIVE_RENDERERS:
        colors[low_idx] = (target - render_img)[y, x]
    else:
        # Low-opacity children colored like the current normalized render add almost no visible
        # jump, then the optimizer can move them toward the target residual.
        colors[low_idx] = render_img[y, x]
    if color_grads is not None:
        color_grads[low_idx] = 0.0

    if field.opacities is None:
        opacities = torch.full((field.n,), 10.0, device=means.device, dtype=means.dtype)
    else:
        opacities = field.opacities.detach().clone()
    opacities[low_idx] = _logit(torch.full((k,), cfg.relocate_init_opacity,
                                           device=means.device, dtype=means.dtype))
    scale_max = None if field.scale_max is None else field.scale_max.detach().clone()
    if scale_max is not None:
        nearest = _nearest_scale_caps(field, new_means)
        if nearest is not None:
            scale_max[low_idx] = nearest
        log_scales[low_idx] = _clamp_new_log_scales(log_scales[low_idx], scale_max[low_idx])
    stats = {
        "activity_mean": float(activity[low_idx].mean().detach().cpu()),
        "residual_mean": float(residual[pix].mean().detach().cpu()),
        "residual_downsample": float(cfg.relocate_residual_downsample),
        "activity_source": "gaussian_activity",
    }
    return (
        GaussianField(means, log_scales, rotations, colors, opacities, scale_max, color_grads),
        k, low_idx, stats,
    )


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
    if cfg.split_mode == "fp_duplicate":
        return _fp_duplicate_indices(field, idx, cfg, H, W)
    if cfg.split_mode == "moment_preserving":
        return _moment_preserving_duplicate_indices(field, idx, cfg, H, W)
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

    base_scale = math.sqrt((H * W) / max(field.n + k, 1)) * cfg.split_scale
    min_spacing = cfg.split_min_spacing * max(base_scale, _MIN_DENSIFY_SCALE)
    idx = _spaced_topk_pixels(residual, W, k, min_spacing, cfg.split_oversample)
    y = torch.div(idx, W, rounding_mode="floor")
    x = idx - y * W
    offsets = torch.stack([
        ((torch.arange(k, device=target.device) % 3).to(target.dtype) - 1.0) * 0.25,
        (((torch.arange(k, device=target.device) // 3) % 3).to(target.dtype) - 1.0) * 0.25,
    ], dim=1)
    means = torch.stack([x.to(target.dtype), y.to(target.dtype)], dim=1) + offsets
    means[:, 0].clamp_(0, W - 1)
    means[:, 1].clamp_(0, H - 1)
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
    colors = _residual_add_colors(target, render_img, y, x, cfg)
    return field.append(GaussianField(means, log_scales, rotations, colors,
                                      scale_max=scale_max)), k


def fit(field: GaussianField, target: torch.Tensor, cfg: FitConfig, verbose: bool = True,
        sched_offset: int = 0, sched_total: int | None = None) -> dict:
    H, W = target.shape[0], target.shape[1]
    field = _ensure_color_basis(field, cfg)
    if _color_solve_enabled(cfg) and cfg.renderer != "normalized":
        raise ValueError(
            "color_solve_every currently supports renderer='normalized' only; "
            f"got {cfg.renderer!r}"
        )
    if _color_solve_enabled(cfg) and field.color_grads is not None:
        raise ValueError(
            "color_solve_every currently supports color_basis='constant' only; "
            "affine color coefficients are optimized with Adam"
        )
    field.trainable()
    opt = _make_optimizer(field, cfg)
    base_lrs = [g["lr"] for g in opt.param_groups]
    lo, hi = math.log(0.35), math.log(max(H, W))
    hist = {
        "iter": [], "psnr": [], "loss": [], "n_gaussians": [], "elapsed": [],
        "split_events": [], "relocate_events": [], "color_solve_events": [],
    }
    absgrad_scores = torch.zeros(field.n, device=target.device, dtype=target.dtype)
    targets = _target_list(cfg)
    iters_to_targets = {str(t): None for t in targets}
    target_thresholds = None
    target_iters_device = None
    if targets:
        target_thresholds = torch.as_tensor(
            [M.psnr_mse_threshold(t) for t in targets],
            device=target.device,
            dtype=target.dtype,
        )
        target_iters_device = torch.full(
            (len(targets),), -1, device=target.device, dtype=torch.long
        )
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
        if cfg.ssim_weight == 0.0:
            loss = pix
        else:
            s = M.ssim(img, target, backend=cfg.ssim_backend)
            loss = (1 - cfg.ssim_weight) * pix + cfg.ssim_weight * (1 - s)
        if field.color_grads is not None and cfg.color_grad_l2 > 0.0:
            loss = loss + cfg.color_grad_l2 * (field.color_grads * field.color_grads).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.split_mode == "absgrad_wave" and field.means.grad is not None:
            with torch.no_grad():
                absgrad_scores.mul_(cfg.absgrad_decay).add_(
                    field.means.grad.detach().abs().sum(dim=1)
                )
        opt.step()
        log_now = it % cfg.log_every == 0 or it == cfg.iters - 1
        with torch.no_grad():
            field.log_scales.clamp_(lo, hi)
            if getattr(field, "scale_max", None) is not None:
                cap = torch.log(torch.clamp(field.scale_max, min=1e-3))
                torch.minimum(field.log_scales, cap, out=field.log_scales)
            if _color_solve_enabled(cfg) and (it + 1) % int(cfg.color_solve_every) == 0:
                stats = _solve_colors_normalized(field, target, cfg, H, W)
                _reset_optimizer_state_for_param(opt, field.colors)
                event = {"iter": it, **stats}
                hist["color_solve_events"].append(event)
                img = _render(field, cfg, H, W)
                if verbose:
                    print(
                        "  color solve "
                        f"{int(stats['iterations'])} cg iters  "
                        f"rel {float(stats['relative_residual']):.3e}"
                    )
            mse_now = None
            if target_thresholds is not None and target_iters_device is not None:
                mse_now = M.mse(img, target).detach().clamp_min(1e-12)
                newly_reached = (target_iters_device < 0) & (mse_now <= target_thresholds)
                target_iters_device = torch.where(
                    newly_reached,
                    torch.full_like(target_iters_device, it),
                    target_iters_device,
                )
            if log_now:
                if mse_now is None:
                    mse_now = M.mse(img, target).detach().clamp_min(1e-12)
                p_now = float(M.psnr_from_mse(mse_now))

        keep = None
        added = 0
        relocated = 0
        reset_idx = None
        # never restructure on the final iteration: the returned field/metrics would include
        # Gaussians that no optimizer step ever touched
        last_it = it == cfg.iters - 1
        split_due = (not last_it and cfg.split_every is not None and cfg.split_every > 0
                     and cfg.split_count > 0 and (it + 1) % cfg.split_every == 0)
        relocate_periodic_due = (
            not last_it and cfg.relocate_every is not None and cfg.relocate_every > 0
            and (it + 1) % cfg.relocate_every == 0
        )
        relocate_split_due = bool(cfg.relocate_at_split and split_due)
        if (not last_it and cfg.prune_every is not None and cfg.prune_every > 0
                and (it + 1) % cfg.prune_every == 0):
            field, keep = _maybe_prune(field, cfg, H, W)
            if verbose and keep is not None:
                print(f"  prune {int((~keep).sum())} -> {field.n} gaussians")
            if keep is not None and cfg.split_mode == "absgrad_wave":
                absgrad_scores = absgrad_scores[keep]
        if cfg.relocate_count > 0 and (relocate_periodic_due or relocate_split_due):
            field, relocated, reset_idx, relocate_stats = _relocate_from_residual(
                field, target, img, cfg
            )
            if relocated > 0:
                if relocate_periodic_due and relocate_split_due:
                    trigger = "periodic+split"
                elif relocate_split_due:
                    trigger = "split"
                else:
                    trigger = "periodic"
                event = {
                    "iter": it,
                    "mode": "relocate",
                    "trigger": trigger,
                    "count": relocated,
                    **relocate_stats,
                }
                hist["relocate_events"].append(event)
                if verbose:
                    print(f"  relocate {relocated} gaussians")
                if cfg.split_mode == "absgrad_wave" and reset_idx is not None:
                    absgrad_scores[reset_idx] = 0
        if split_due:
            split_event = None
            absgrad_scores_carried = False
            if cfg.split_mode in (
                "duplicate", "fp_duplicate", "moment_preserving", "support_duplicate"
            ):
                field, added = _split_from_residual(field, target, img, cfg)
            elif cfg.split_mode == "ranked_wave":
                field, added, ranked_stats = _ranked_wave_from_residual(field, target, img, cfg)
                split_event = {
                    "iter": it,
                    "mode": cfg.split_mode,
                    "added": added,
                    **ranked_stats,
                }
            elif cfg.split_mode == "absgrad_wave":
                field, added, absgrad_stats, absgrad_scores = _absgrad_wave_from_scores(
                    field, absgrad_scores, cfg, H, W
                )
                absgrad_scores_carried = True
                split_event = {
                    "iter": it,
                    "mode": cfg.split_mode,
                    "added": added,
                    **absgrad_stats,
                }
            elif cfg.split_mode == "freq_violation":
                field, added, freq_stats = _freq_violation_from_residual(
                    field, target, img, cfg
                )
                split_event = {
                    "iter": it,
                    "mode": cfg.split_mode,
                    "added": added,
                    **freq_stats,
                }
            elif cfg.split_mode in ("residual_add", "residual_tensor_add"):
                field, added = _add_from_residual(
                    field, target, img, cfg, tensor_aligned=cfg.split_mode == "residual_tensor_add"
                )
            else:
                raise ValueError(
                    f"unknown split_mode {cfg.split_mode!r}; expected duplicate, "
                    "fp_duplicate, moment_preserving, support_duplicate, residual_add, "
                    "residual_tensor_add, ranked_wave, absgrad_wave, or freq_violation")
            if verbose and added > 0:
                print(f"  split +{added} -> {field.n} gaussians")
            if added > 0:
                hist["split_events"].append(
                    split_event or {"iter": it, "mode": cfg.split_mode, "added": added}
                )
                if cfg.split_mode == "absgrad_wave" and not absgrad_scores_carried:
                    absgrad_scores = torch.cat(
                        [absgrad_scores, absgrad_scores.new_zeros(added)], dim=0
                    )
        if keep is not None or added > 0 or relocated > 0:
            field.trainable()
            opt = _carry_adam_state(opt, field, cfg, keep, added, reset_idx=reset_idx)
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
        if target_iters_device is not None:
            reached = target_iters_device.detach().cpu().tolist()
            iters_to_targets = {
                str(t): (None if int(v) < 0 else int(v)) for t, v in zip(targets, reached)
            }
        img = _render(field, cfg, H, W)
        out = {
            "field": field, "history": hist, "render": img,
            "psnr": M.psnr(img, target),
            "ssim": float(M.ssim(img, target, backend=cfg.ssim_backend)),
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
