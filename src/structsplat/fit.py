"""Per-image optimization (the baseline fitter every strategy is scored through).

loss = (1 - w) * pixel + w * (1 - SSIM), Adam/AdamW with per-parameter-group learning rates.
Records PSNR history and iterations-to-target so the ablation can compare *quality at fixed
budget* and *convergence speed*. Supports selectable pixel loss (l1/l2/charbonnier, with an
optional warmup), optimizer, LR schedule, renderer mode, opacity, and residual densification.
Requires torch.
"""
from __future__ import annotations
from dataclasses import replace
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
    _resolve_support_fade_alpha,
    _support_weight,
    _tile_bounds,
    _tile_coords,
    gaussian_activity,
    render_field,
)
from . import mask as _mask
from . import metrics as M
from . import structure_tensor as st

# Renderer modes that accumulate additively (ADR-0006): a new/split Gaussian stacks on the
# existing accumulation, so it must carry the residual color, not the full target color.
_ADDITIVE_RENDERERS = (
    "additive", "cuda_additive", "cuda_tiled_additive", "gsplat", "cuda_gsplat",
)
_NORMALIZED_COLOR_SOLVE_RENDERERS = (
    "normalized", "cuda", "cuda_normalized", "cuda_tiled", "cuda_tiled_normalized",
)
# Shared lower bound on densified Gaussian scales (px); was duplicated as bare 0.35 literals.
_MIN_DENSIFY_SCALE = 0.35


class _MaskConstraint:
    """Torch-side mask containment built from a NumPy :class:`mask.MaskGeometry` (CORE-010).

    ``apply`` projects means into the eroded mask interior and overwrites ``field.scale_max`` with a
    fresh per-row cap derived from the signed distance, so the sigma_cutoff support ellipse of the
    *effective* covariance (base scales + filter variance + AA dilation) stays inside the mask. With
    ``support_fade=True`` (fade_alpha=1) the renderer weight is exactly zero outside that ellipse, so
    a contained field renders exactly zero outside the mask. ``coverage`` is a differentiable soft
    penalty on the raw out-of-mask weight sum. Because ``apply`` overwrites ``scale_max`` each call,
    ``mask_contain`` owns the cap and does not compose with ``scale_cap_mode``.
    """

    def __init__(self, geom: _mask.MaskGeometry, device, dtype,
                 sigma_cutoff: float, margin: float, min_scale: float = _MIN_DENSIFY_SCALE):
        H, W = geom.shape
        self.H, self.W = int(H), int(W)
        self.sigma_cutoff = float(sigma_cutoff)
        self.margin = float(margin)
        self.min_scale = float(min_scale)
        self.inside = torch.as_tensor(geom.inside, device=device, dtype=torch.bool)
        self.eroded_flat = torch.as_tensor(
            geom.eroded.reshape(-1), device=device, dtype=torch.bool)
        self.sdf_flat = torch.as_tensor(
            geom.sdf.reshape(-1), device=device, dtype=dtype)
        self.nearest_inside_flat = torch.as_tensor(
            geom.nearest_inside_flat, device=device, dtype=torch.long)
        self.outside_flat = torch.as_tensor(geom.outside_flat, device=device, dtype=torch.long)
        # (H, W) 1.0 inside / 0.0 outside, for mask loss weighting and SSIM matting.
        self.inside_weight = self.inside.to(dtype=dtype)

    @classmethod
    def from_mask(cls, mask_arr, device, dtype, sigma_cutoff: float, margin: float,
                  aa_dilation: float = 0.0, min_scale: float = _MIN_DENSIFY_SCALE
                  ) -> "_MaskConstraint":
        # Reserve room for the min scale (and any global AA dilation) so a projected mean always
        # admits a cap >= min_scale without violating containment.
        erode_radius = float(margin) + float(sigma_cutoff) * math.sqrt(
            float(min_scale) ** 2 + max(float(aa_dilation), 0.0)
        )
        geom = _mask.MaskGeometry.build(mask_arr, erode_radius)
        return cls(geom, device, dtype, sigma_cutoff, margin, min_scale)

    def _extra_variance(self, field: GaussianField, aa_dilation: float):
        """Isotropic variance added to base scales at render time (AA dilation + filter)."""
        extra = float(aa_dilation)
        if field.filter_variance is not None:
            fv = field.filter_variance.to(
                device=field.means.device, dtype=field.means.dtype
            ).reshape(-1)
            return fv + extra
        return extra

    @torch.no_grad()
    def apply(self, field: GaussianField, cfg: FitConfig | None = None,
              aa_dilation: float | None = None) -> None:
        if aa_dilation is None:
            aa_dilation = 0.0 if cfg is None else float(cfg.aa_dilation)
        H, W = self.H, self.W
        dt = field.means.dtype
        means = field.means
        ix = torch.round(means[:, 0]).clamp_(0, W - 1).long()
        iy = torch.round(means[:, 1]).clamp_(0, H - 1).long()
        flat = iy * W + ix
        inside_here = self.eroded_flat[flat]
        tgt = self.nearest_inside_flat[flat].clamp_min(0)
        tx = (tgt % W).to(dt)
        ty = torch.div(tgt, W, rounding_mode="floor").to(dt)
        newx = torch.where(inside_here, means[:, 0], tx).clamp_(0, W - 1)
        newy = torch.where(inside_here, means[:, 1], ty).clamp_(0, H - 1)
        field.means.copy_(torch.stack([newx, newy], dim=1))
        ix2 = torch.round(field.means[:, 0]).clamp_(0, W - 1).long()
        iy2 = torch.round(field.means[:, 1]).clamp_(0, H - 1).long()
        d = (self.sdf_flat[iy2 * W + ix2] - self.margin).clamp_min(1e-3)
        cap_sq = (d / self.sigma_cutoff) ** 2 - self._extra_variance(field, aa_dilation)
        cap = torch.sqrt(cap_sq.clamp_min(self.min_scale ** 2))
        scale_max = torch.stack([cap, cap], dim=1)
        field.scale_max = scale_max
        log_cap = torch.log(scale_max.clamp_min(1e-3))
        field.log_scales.copy_(torch.minimum(field.log_scales, log_cap))

    def coverage(self, field: GaussianField, cfg: FitConfig,
                 support_fade_alpha: float | None = None) -> torch.Tensor:
        """Differentiable mean out-of-mask unnormalized weight sum sum_i o_i G_i(p), p not in M."""
        H, W = self.H, self.W
        dev, dt = field.means.device, field.means.dtype
        if self.outside_flat.numel() == 0:
            return field.means.sum() * 0.0
        means = field.means
        conics = field.conics(cfg.aa_dilation)
        radii = field.radii(cfg.sigma_cutoff, cfg.aa_dilation)
        opacities = field.opacity_values()
        den = torch.zeros(H * W, device=dev, dtype=dt)
        x0, y0, Tx, n = _tile_bounds(means, radii, H, W)
        budget = _element_budget(cfg.render_chunk)
        for s, e in _flat_tile_slices(n, budget):
            gid, px, py = _tile_coords(x0, y0, Tx, n, s, e, dev)
            dx = px.to(dt) - means[gid, 0]
            dy = py.to(dt) - means[gid, 1]
            a, b, c = conics[gid, 0], conics[gid, 1], conics[gid, 2]
            q = a * dx * dx + 2.0 * b * dx * dy + c * dy * dy
            w = _support_weight(q, cfg.sigma_cutoff, cfg.support_fade, support_fade_alpha)
            if opacities is not None:
                w = w * opacities[gid]
            den = den.index_add(0, py * W + px, w)
        return den[self.outside_flat].sum() / float(self.outside_flat.numel())


def _generation_density_filter_variance(cfg: FitConfig, H: int, W: int,
                                        n_after: int) -> float:
    """Fixed isotropic variance assigned to one birth cohort.

    The ``9*pi`` default is the average area of a three-sigma disk. ``n_after`` is the active
    Gaussian count immediately after inserting the cohort, matching GaussianImage++.
    """
    if n_after <= 0:
        raise ValueError(f"n_after must be > 0, got {n_after}")
    return min(
        float(cfg.covariance_filter_max_variance),
        float(H * W) / (float(cfg.covariance_filter_alpha) * float(n_after)),
    )


@torch.no_grad()
def _ensure_generation_density_filter(field: GaussianField, cfg: FitConfig,
                                      H: int, W: int) -> GaussianField:
    """Assign the initial cohort and fill rows appended without cohort metadata."""
    if cfg.covariance_filter_mode == "none":
        return field
    value = _generation_density_filter_variance(cfg, H, W, field.n)
    if field.filter_variance is None:
        field.filter_variance = torch.full(
            (field.n,), value, device=field.means.device, dtype=field.means.dtype
        )
        return field
    variance = field.filter_variance.detach().to(
        device=field.means.device, dtype=field.means.dtype
    ).reshape(-1).clone()
    if variance.numel() != field.n:
        raise ValueError(
            "filter_variance must have one value per Gaussian, "
            f"got {variance.numel()} for {field.n} rows"
        )
    if bool((variance < 0.0).any()):
        raise ValueError("filter_variance must be non-negative")
    # append() represents an unassigned filtered cohort with exact zero variance. Zero is also
    # the valid result when max_variance=0, in which case this fill is intentionally a no-op.
    variance[variance == 0.0] = value
    field.filter_variance = variance
    return field


def _new_cohort_filter_variance(field: GaussianField, cfg: FitConfig, H: int, W: int,
                                count: int) -> torch.Tensor | None:
    if cfg.covariance_filter_mode == "none" or count <= 0:
        return None
    value = _generation_density_filter_variance(cfg, H, W, field.n + count)
    return torch.full(
        (count,), value, device=field.means.device, dtype=field.means.dtype
    )


def _mark_new_parent_idx(field: GaussianField, idx: torch.Tensor) -> GaussianField:
    field._new_parent_idx = idx.detach().clone()  # type: ignore[attr-defined]
    return field


def _background_mask(field: GaussianField) -> torch.Tensor | None:
    mask = getattr(field, "background_mask", None)
    if mask is None or int(mask.numel()) != int(field.n):
        return None
    if not bool(mask.any()):
        return None
    return mask.to(device=field.means.device, dtype=torch.bool)


def _detail_count(field: GaussianField) -> int:
    mask = _background_mask(field)
    if mask is None:
        return int(field.n)
    return int((~mask).sum().item())


def _mask_background_scores(scores: torch.Tensor, field: GaussianField,
                            fill: float = -float("inf")) -> torch.Tensor:
    mask = _background_mask(field)
    if mask is None:
        return scores
    out = scores.clone()
    out[mask.to(device=out.device)] = fill
    return out


def _zero_background_geometry_state(field: GaussianField, opt: torch.optim.Optimizer) -> None:
    mask = _background_mask(field)
    if mask is None:
        return
    for param in (field.means, field.log_scales, field.rotations):
        if param.grad is not None:
            param.grad[mask] = 0
        state = opt.state.get(param)
        if not state:
            continue
        for value in state.values():
            if torch.is_tensor(value) and value.shape == param.shape:
                value[mask] = 0


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
        total = max(1, sched_total if sched_total is not None else cfg.iters)
        progress = (sched_offset + it) / total
        start = cfg.lr_cosine_start_frac
        if progress <= start:
            return 1.0
        tail_progress = (progress - start) / (1.0 - start)
        return 0.5 * (1.0 + math.cos(math.pi * tail_progress))
    raise ValueError(f"unknown lr_schedule {cfg.lr_schedule!r}; expected none, step, or cosine")


def _carry_adam_state(opt_old, field: GaussianField, cfg: FitConfig,
                      keep: torch.Tensor | None, n_new: int,
                      reset_idx: torch.Tensor | None = None,
                      new_parent_idx: torch.Tensor | None = None):
    """Fresh optimizer for a restructured field, carrying per-Gaussian optimizer buffers.

    Without this, every prune/split resets exp_avg/exp_avg_sq for ALL Gaussians and the
    loss spikes while Adam re-estimates curvature. Surviving rows keep their moments; new
    rows start at zero by default and share the tensor's inherited `step` (Adam keeps one step
    per parameter tensor), so their first updates behave like a normal warm Adam step. With
    seed_new_row_optimizer_state enabled, split children inherit parent rows and sampled-add
    children use the carried-row median. Handles any number of parameter groups (4, or 5 with
    opacity), Adam/AdamW, and Adan.
    """
    opt_new = _make_optimizer(field, cfg)
    if new_parent_idx is not None:
        new_parent_idx = new_parent_idx.detach().to(dtype=torch.long)
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
                        if cfg.seed_new_row_optimizer_state and out.shape[0] > 0:
                            median = out.median(dim=0).values
                            pad[:] = median
                            if new_parent_idx is not None and new_parent_idx.numel() == n_new:
                                src = new_parent_idx.to(device=out.device)
                                valid = (src >= 0) & (src < out.shape[0])
                                if bool(valid.any()):
                                    pad[valid] = out[src[valid]]
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


def _pixel_loss_map(pred: torch.Tensor, target: torch.Tensor, kind: str,
                    charbonnier_eps: float = 1e-3) -> torch.Tensor:
    diff = pred - target
    if kind == "l1":
        return diff.abs().mean(dim=2)
    if kind == "l2":
        return (diff * diff).mean(dim=2)
    if kind == "charbonnier":
        return torch.sqrt(diff * diff + charbonnier_eps ** 2).mean(dim=2)
    raise ValueError(f"unknown pixel_loss {kind!r}; expected l1, l2, or charbonnier")


def _pixel_loss(pred: torch.Tensor, target: torch.Tensor, kind: str,
                charbonnier_eps: float = 1e-3) -> torch.Tensor:
    return _pixel_loss_map(pred, target, kind, charbonnier_eps).mean()


def _weighted_pixel_loss(pred: torch.Tensor, target: torch.Tensor, kind: str,
                         charbonnier_eps: float, weight: torch.Tensor | None) -> torch.Tensor:
    loss_map = _pixel_loss_map(pred, target, kind, charbonnier_eps)
    if weight is None:
        return loss_map.mean()
    w = weight.to(device=loss_map.device, dtype=loss_map.dtype)
    return (loss_map * w).sum() / w.sum().clamp_min(1e-12)


def _prepare_loss_target(target: torch.Tensor, downsample: int) -> torch.Tensor:
    """Build one area-lowpass target at the original shape for loss supervision."""
    if downsample == 1:
        return target
    height, width = target.shape[:2]
    low_size = (max(1, height // downsample), max(1, width // downsample))
    with torch.no_grad():
        bchw = target.permute(2, 0, 1).unsqueeze(0)
        low = F.interpolate(bchw, size=low_size, mode="area")
        restored = F.interpolate(
            low,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )
    return restored.squeeze(0).permute(1, 2, 0)


def _loss_target_full_weight(
    cfg: FitConfig,
    it: int,
    sched_offset: int = 0,
    sched_total: int | None = None,
) -> float:
    """Cosine blend weight for the full target on the global fit schedule."""
    if cfg.loss_target_downsample == 1:
        return 1.0
    total = max(1, sched_total if sched_total is not None else cfg.iters)
    transition = float(cfg.loss_target_full_frac) * total
    progress = min(1.0, max(0.0, (sched_offset + it) / transition))
    return 0.5 * (1.0 - math.cos(math.pi * progress))


def _validate_loss_target_schedule(
    cfg: FitConfig,
    *,
    sched_offset: int,
    sched_total: int | None,
) -> None:
    """Reject state-changing/stop events that can precede full-target supervision."""
    if cfg.loss_target_downsample == 1:
        return
    total = max(1, sched_total if sched_total is not None else cfg.iters)
    full_at = float(cfg.loss_target_full_frac) * total

    early_events: list[tuple[str, int]] = []
    if (
        cfg.prune_every is not None
        and 0 < cfg.prune_every <= cfg.iters
        and cfg.prune_min_activity > 0.0
    ):
        early_events.append(("prune", int(cfg.prune_every)))
    if (
        cfg.split_every is not None
        and 0 < cfg.split_every <= cfg.iters
        and cfg.split_count > 0
    ):
        early_events.append(("split", int(cfg.split_every)))
    if (
        cfg.relocate_every is not None
        and 0 < cfg.relocate_every <= cfg.iters
        and cfg.relocate_count > 0
    ):
        early_events.append(("relocate", int(cfg.relocate_every)))
    if cfg.adaptive_count and cfg.adaptive_growth_every <= cfg.iters:
        early_events.append(("adaptive growth", int(cfg.adaptive_growth_every)))
    for name, period in early_events:
        first_event_iter = sched_offset + period - 1
        if first_event_iter < full_at:
            raise ValueError(
                f"{name} event at global iteration {first_event_iter} occurs before the "
                f"loss-target curriculum reaches the full target at {full_at:g}"
            )
    if cfg.early_stop_patience is not None:
        first_permitted = sched_offset + int(cfg.early_stop_min_iters)
        if first_permitted < full_at:
            raise ValueError(
                f"early stopping is permitted at global iteration {first_permitted} before "
                f"the loss-target curriculum reaches the full target at {full_at:g}"
            )


_SOBEL_KERNEL_CACHE: dict[tuple[str, int, str, int], tuple[torch.Tensor, torch.Tensor]] = {}


def _sobel_gradients(image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Channel-wise Sobel gradients for an HWC image using reflected boundaries."""
    if image.ndim != 3:
        raise ValueError(f"expected an HWC image, got shape {tuple(image.shape)}")
    height, width, channels = image.shape
    if channels <= 0:
        raise ValueError("expected at least one image channel")
    device = image.device
    key = (
        device.type,
        -1 if device.index is None else int(device.index),
        str(image.dtype),
        int(channels),
    )
    kernels = _SOBEL_KERNEL_CACHE.get(key)
    if kernels is None:
        horizontal = image.new_tensor([
            [-1.0, 0.0, 1.0],
            [-2.0, 0.0, 2.0],
            [-1.0, 0.0, 1.0],
        ])
        vertical = horizontal.t().contiguous()
        kernels = (
            horizontal.view(1, 1, 3, 3).expand(channels, 1, 3, 3).contiguous(),
            vertical.view(1, 1, 3, 3).expand(channels, 1, 3, 3).contiguous(),
        )
        _SOBEL_KERNEL_CACHE[key] = kernels
    bchw = image.permute(2, 0, 1).unsqueeze(0)
    pad_mode = "reflect" if height > 1 and width > 1 else "replicate"
    padded = F.pad(bchw, (1, 1, 1, 1), mode=pad_mode)
    grad_x = F.conv2d(padded, kernels[0], groups=channels)
    grad_y = F.conv2d(padded, kernels[1], groups=channels)
    return (
        grad_x.squeeze(0).permute(1, 2, 0),
        grad_y.squeeze(0).permute(1, 2, 0),
    )


def _geometry_consistent_loss(
    pred: torch.Tensor,
    target_gradients: tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    """Ground-truth-gradient-weighted Sobel discrepancy from arXiv:2512.24018.

    The paper divides the channel-summed discrepancy by H*W, rather than H*W*C.
    Positive target-gradient magnitudes suppress flat-region contributions while retaining
    directional horizontal/vertical supervision.
    """
    pred_x, pred_y = _sobel_gradients(pred)
    target_x, target_y = target_gradients
    discrepancy = (
        target_x.abs() * (pred_x - target_x).square()
        + target_y.abs() * (pred_y - target_y).square()
    )
    return discrepancy.sum(dim=2).mean()


def _coerce_weight_map(weight, H: int, W: int, device, dtype) -> torch.Tensor:
    """Coerce an (H,W)/(H,W,1) map to an (H,W) tensor at the target resolution."""
    weight = torch.as_tensor(weight, device=device, dtype=dtype)
    if weight.ndim == 3 and weight.shape[-1] == 1:
        weight = weight[..., 0]
    if weight.ndim != 2:
        raise ValueError(f"loss_weight_map must have shape (H,W), got {tuple(weight.shape)}")
    if tuple(weight.shape) != (H, W):
        weight = F.interpolate(
            weight[None, None], size=(H, W), mode="bilinear", align_corners=False
        )[0, 0]
    return weight


def _prepare_loss_weight_map(target: torch.Tensor, cfg: FitConfig,
                             loss_weight_map=None) -> torch.Tensor | None:
    if cfg.loss_weighting == "none":
        return None
    H, W = target.shape[:2]
    if cfg.loss_weighting == "mask":
        # Mask mode uses the raw indicator (zeros allowed) so out-of-mask pixels drop out of the
        # pixel loss entirely, unlike the tensor mode's [1, 1+beta] emphasis map.
        if loss_weight_map is None:
            raise ValueError(
                "loss_weighting='mask' requires a mask; pass mask= to fit()")
        weight = _coerce_weight_map(loss_weight_map, H, W, target.device, target.dtype)
        weight = torch.nan_to_num(weight.detach(), nan=0.0, posinf=1.0, neginf=0.0)
        return weight.clamp(0.0, 1.0)
    if cfg.loss_weighting != "tensor":
        raise ValueError(
            f"loss_weighting must be none, tensor, or mask, got {cfg.loss_weighting!r}")
    if loss_weight_map is None:
        energy = st.compute(target.detach().cpu().numpy()).energy
        weight = torch.as_tensor(energy, device=target.device, dtype=target.dtype)
    else:
        weight = torch.as_tensor(loss_weight_map, device=target.device, dtype=target.dtype)
    if weight.ndim == 3 and weight.shape[-1] == 1:
        weight = weight[..., 0]
    if weight.ndim != 2:
        raise ValueError(f"loss_weight_map must have shape (H,W), got {tuple(weight.shape)}")
    H, W = target.shape[:2]
    if tuple(weight.shape) != (H, W):
        weight = F.interpolate(
            weight[None, None], size=(H, W), mode="bilinear", align_corners=False
        )[0, 0]
    weight = torch.nan_to_num(weight.detach(), nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    ref = weight.max().clamp_min(1e-12)
    norm = torch.where(ref > 1e-12, (weight / ref).clamp(0.0, 1.0), torch.zeros_like(weight))
    return (1.0 + float(cfg.loss_weight_beta) * norm).detach()


def _loss_kind(cfg: FitConfig, it: int) -> str:
    if cfg.loss_warmup_iters > 0 and it < cfg.loss_warmup_iters:
        return cfg.loss_warmup_pixel_loss
    return cfg.pixel_loss


def _fit_support_fade_alpha(cfg: FitConfig, it: int | None = None) -> float:
    if cfg.support_fade_until_frac is None:
        return _resolve_support_fade_alpha(cfg.support_fade)
    if it is None:
        it = cfg.iters
    total = max(1, int(cfg.iters))
    until = float(cfg.support_fade_until_frac) * float(total)
    if float(it) <= until:
        return 1.0
    cross = int(cfg.support_fade_crossfade_iters)
    if cross <= 0:
        return 0.0
    progress = float(it) - until
    if progress >= float(cross):
        return 0.0
    return max(0.0, 1.0 - progress / float(cross))


def _render(field: GaussianField, cfg: FitConfig, H: int, W: int,
            support_fade_alpha: float | None = None) -> torch.Tensor:
    scales = field.effective_scales(
        cfg.aa_dilation if cfg.renderer in ("gsplat", "cuda_gsplat") else 0.0
    )
    return render_field(field.means, field.conics(cfg.aa_dilation), field.colors,
                        field.radii(cfg.sigma_cutoff, cfg.aa_dilation),
                        H, W, cfg.render_chunk, cfg.renderer, field.opacity_values(),
                        scales=scales, rotations=field.rotations,
                        support_fade=cfg.support_fade, sigma_cutoff=cfg.sigma_cutoff,
                        color_grads=field.color_grads,
                        support_fade_alpha=support_fade_alpha,
                        checkpoint_chunks=cfg.render_checkpoint)


def _qat_enabled(cfg: FitConfig) -> bool:
    return cfg.qat_mode != "off"


def _make_qat_codec_config(field: GaussianField, cfg: FitConfig):
    """Freeze codec quantizer assumptions for fit-time QAT and later encoding."""
    from .codec import CodecConfig, color_ranges, scale_ranges

    color_lo, color_hi = color_ranges(field)
    scale_lo, scale_hi = scale_ranges(field)
    return CodecConfig(
        bits_means=cfg.qat_bits_means,
        bits_scales=cfg.qat_bits_scales,
        bits_rot=cfg.qat_bits_rot,
        bits_colors=cfg.qat_bits_colors,
        bits_opacity=cfg.qat_bits_opacity,
        color_lo=color_lo,
        color_hi=color_hi,
        scale_lo=scale_lo,
        scale_hi=scale_hi,
    )


def _noise_quantize(x: torch.Tensor, lo, hi, bits: int) -> torch.Tensor:
    levels = (1 << bits) - 1
    lo = torch.as_tensor(lo, device=x.device, dtype=x.dtype)
    hi = torch.as_tensor(hi, device=x.device, dtype=x.dtype)
    step = (hi - lo).clamp_min(1e-12) / max(levels, 1)
    return (x + (torch.rand_like(x) - 0.5) * step).clamp(lo, hi)


def _noise_circular(x: torch.Tensor, period: float, bits: int) -> torch.Tensor:
    levels = 1 << bits
    period_t = torch.as_tensor(period, device=x.device, dtype=x.dtype)
    step = period_t / max(levels, 1)
    return torch.remainder(torch.remainder(x, period_t) + (torch.rand_like(x) - 0.5) * step,
                           period_t)


def _noise_quantized_view(field: GaussianField, H: int, W: int, ccfg) -> GaussianField:
    if field.color_grads is not None:
        raise ValueError(
            "fit-time QAT currently supports color_basis='constant' only; "
            "codec v1 cannot encode affine color fields"
        )
    slo = torch.as_tensor(ccfg.scale_lo, dtype=field.log_scales.dtype,
                          device=field.log_scales.device)
    shi = torch.as_tensor(ccfg.scale_hi, dtype=field.log_scales.dtype,
                          device=field.log_scales.device)
    clo = torch.as_tensor(ccfg.color_lo, dtype=field.colors.dtype, device=field.colors.device)
    chi = torch.as_tensor(ccfg.color_hi, dtype=field.colors.dtype, device=field.colors.device)
    means = torch.stack([
        _noise_quantize(field.means[:, 0], 0.0, W - 1.0, ccfg.bits_means),
        _noise_quantize(field.means[:, 1], 0.0, H - 1.0, ccfg.bits_means),
    ], dim=1)
    # Codec v1 materializes covariance filtering into RS scales, so QAT must quantize the same
    # effective representation rather than a base scale that the bitstream cannot reconstruct.
    log_scales = _noise_quantize(field.effective_log_scales(), slo, shi, ccfg.bits_scales)
    theta = (
        _noise_circular(field.rotations, float(torch.pi), ccfg.bits_rot)
        if ccfg.rot_circular
        else _noise_quantize(torch.remainder(field.rotations, torch.pi), 0.0, float(torch.pi),
                             ccfg.bits_rot)
    )
    colors = _noise_quantize(field.colors, clo, chi, ccfg.bits_colors)
    opacities = None
    if field.opacities is not None:
        p = _noise_quantize(torch.sigmoid(field.opacities), 0.0, 1.0, ccfg.bits_opacity)
        p = p.clamp(1e-4, 1.0 - 1e-4)
        opacities = torch.log(p / (1.0 - p))
    return GaussianField(means, log_scales, theta, colors, opacities)


def _qat_field_view(field: GaussianField, cfg: FitConfig, H: int, W: int, ccfg):
    if not _qat_enabled(cfg):
        return field
    if field.color_grads is not None:
        raise ValueError(
            "fit-time QAT currently supports color_basis='constant' only; "
            "codec v1 cannot encode affine color fields"
        )
    if cfg.qat_mode == "ste":
        from .codec import quantized_view

        return quantized_view(field, H, W, ccfg)
    if cfg.qat_mode == "noise":
        return _noise_quantized_view(field, H, W, ccfg)
    raise ValueError(f"unknown qat_mode {cfg.qat_mode!r}")


def _render_loss_view(field: GaussianField, cfg: FitConfig, H: int, W: int, qat_ccfg=None,
                      support_fade_alpha: float | None = None):
    qf = _qat_field_view(field, cfg, H, W, qat_ccfg) if _qat_enabled(cfg) else field
    return _render(qf, cfg, H, W, support_fade_alpha=support_fade_alpha)


def differentiable_rate_bpp(field: GaussianField, H: int, W: int, cfg: FitConfig) -> torch.Tensor:
    """Smooth in-loop rate proxy used by COMP-004.

    This is not the final entropy model. It covers every encoded attribute with a Laplace-like
    residual code-length proxy in bits/pixel so rate pressure can affect gradients during fitting.
    Actual RD numbers still come from `codec.encode`/`rd_point`.
    """

    area = max(float(H * W), 1.0)
    bits = field.means.new_tensor(0.0)

    def add_proxy(x: torch.Tensor, step, center=None) -> torch.Tensor:
        step_t = torch.as_tensor(step, device=x.device, dtype=x.dtype).clamp_min(1e-12)
        if center is None:
            center_t = x.detach().median(dim=0).values if x.ndim > 1 else x.detach().median()
        else:
            center_t = torch.as_tensor(center, device=x.device, dtype=x.dtype)
        return torch.log1p(((x - center_t) / step_t).abs()).sum() / math.log(2.0)

    mean_steps = field.means.new_tensor([
        max(float(W - 1), 1.0) / max((1 << cfg.qat_bits_means) - 1, 1),
        max(float(H - 1), 1.0) / max((1 << cfg.qat_bits_means) - 1, 1),
    ])
    bits = bits + add_proxy(field.means, mean_steps)
    # Use a conservative unit-range step for log-scales/colors in the proxy; exact frozen codec
    # ranges are returned separately as `qat_codec_config` when QAT is enabled.
    bits = bits + add_proxy(
        field.effective_log_scales(),
        1.0 / max((1 << cfg.qat_bits_scales) - 1, 1),
    )
    bits = bits + add_proxy(
        torch.remainder(field.rotations, torch.pi),
        float(torch.pi) / max(1 << cfg.qat_bits_rot, 1),
        center=float(torch.pi) * 0.5,
    )
    bits = bits + add_proxy(field.colors, 1.0 / max((1 << cfg.qat_bits_colors) - 1, 1))
    if field.opacities is not None:
        bits = bits + add_proxy(
            torch.sigmoid(field.opacities),
            1.0 / max((1 << cfg.qat_bits_opacity) - 1, 1),
            center=0.5,
        )
    return bits / area


def _color_solve_schedule_tokens(cfg: FitConfig) -> set[str]:
    tokens = set(str(cfg.color_solve_schedule).split("+"))
    if tokens == {"none"}:
        return set()
    if cfg.color_solve_every is not None and cfg.color_solve_every > 0:
        tokens.add("every")
    else:
        tokens.discard("every")
    return tokens


def _color_solve_enabled(cfg: FitConfig) -> bool:
    return bool(_color_solve_schedule_tokens(cfg))


def _color_solve_renderer_supported(renderer: str) -> bool:
    return renderer in _NORMALIZED_COLOR_SOLVE_RENDERERS


def _ensure_color_basis(field: GaussianField, cfg: FitConfig) -> GaussianField:
    if cfg.color_basis == "affine":
        return field.with_affine_colors()
    return field


@torch.no_grad()
def _normalized_color_denominator(field: GaussianField, cfg: FitConfig,
                                  H: int, W: int,
                                  support_fade_alpha: float | None = None) -> torch.Tensor:
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
        w = _support_weight(q, cfg.sigma_cutoff, cfg.support_fade, support_fade_alpha)
        if opacities is not None:
            w = w * opacities[gid]
        den.index_add_(0, py * W + px, w[:, None])
    return den


@torch.no_grad()
def _normalized_color_basis_apply(field: GaussianField, colors: torch.Tensor, cfg: FitConfig,
                                  H: int, W: int, den: torch.Tensor,
                                  support_fade_alpha: float | None = None) -> torch.Tensor:
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
        w = _support_weight(q, cfg.sigma_cutoff, cfg.support_fade, support_fade_alpha)
        if opacities is not None:
            w = w * opacities[gid]
        flat = py * W + px
        basis = w[:, None] / (den[flat] + _EPS)
        out.index_add_(0, flat, basis * colors[gid])
    return out.view(H, W, colors.shape[1])


@torch.no_grad()
def _normalized_color_basis_transpose(field: GaussianField, image: torch.Tensor,
                                      cfg: FitConfig, H: int, W: int,
                                      den: torch.Tensor,
                                      support_fade_alpha: float | None = None) -> torch.Tensor:
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
        w = _support_weight(q, cfg.sigma_cutoff, cfg.support_fade, support_fade_alpha)
        if opacities is not None:
            w = w * opacities[gid]
        flat = py * W + px
        basis = w[:, None] / (den[flat] + _EPS)
        out.index_add_(0, gid, basis * image[py, px].to(dt))
    return out


@torch.no_grad()
def _solve_colors_normalized(field: GaussianField, target: torch.Tensor, cfg: FitConfig,
                             H: int, W: int,
                             support_fade_alpha: float | None = None) -> dict[str, float | int]:
    """Solve fixed-geometry normalized-renderer RGB colors with implicit CG.

    The normalized renderer is linear in colors once means/scales/rotations/opacities are fixed:
    render = A c. We solve `(A.T A + lambda I)c = A.T target + lambda c_prev` for the three RGB
    channels simultaneously, without materializing the dense pixel-by-Gaussian matrix.
    """
    if not _color_solve_renderer_supported(cfg.renderer):
        raise ValueError(
            "color_solve_every currently supports normalized renderers only; "
            f"got {cfg.renderer!r}"
        )
    if field.color_grads is not None:
        raise ValueError(
            "color_solve_every currently supports color_basis='constant' only; "
            "affine color coefficients are optimized with Adam"
        )
    if field.n == 0:
        return {"iterations": 0, "relative_residual": 0.0}

    den = _normalized_color_denominator(
        field, cfg, H, W, support_fade_alpha=support_fade_alpha
    )
    lam = float(cfg.color_solve_lambda)
    x0 = field.colors.detach().clone()
    b = _normalized_color_basis_transpose(
        field, target, cfg, H, W, den, support_fade_alpha=support_fade_alpha
    )
    if lam > 0.0:
        b = b + lam * x0

    def normal_matvec(x: torch.Tensor) -> torch.Tensor:
        ax = _normalized_color_basis_apply(
            field, x, cfg, H, W, den, support_fade_alpha=support_fade_alpha
        )
        atax = _normalized_color_basis_transpose(
            field, ax, cfg, H, W, den, support_fade_alpha=support_fade_alpha
        )
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


def _row_parameter_tensors(field: GaussianField) -> list[torch.Tensor]:
    tensors = [field.means, field.log_scales, field.rotations, field.colors]
    if field.opacities is not None:
        tensors.append(field.opacities)
    if field.color_grads is not None:
        tensors.append(field.color_grads)
    return tensors


def _capture_young_row_temper_snapshots(
    field: GaussianField,
    row_birth_iter: torch.Tensor,
    it: int,
    cfg: FitConfig,
) -> tuple[torch.Tensor, torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]] | None:
    if cfg.new_row_temper_iters <= 0 or row_birth_iter.numel() == 0:
        return None
    age = it - row_birth_iter
    active = (row_birth_iter >= 0) & (age >= 0) & (age < int(cfg.new_row_temper_iters))
    if not bool(active.any()):
        return None
    idx = torch.nonzero(active, as_tuple=False).reshape(-1)
    if cfg.new_row_temper_iters <= 1:
        ramp = torch.ones(idx.shape[0], device=idx.device, dtype=field.means.dtype)
    else:
        max_age = max(1, int(cfg.new_row_temper_iters) - 1)
        ramp = (
            float(cfg.new_row_temper_start)
            + (1.0 - float(cfg.new_row_temper_start)) * age[idx].to(torch.float32)
            / float(max_age)
        ).to(device=field.means.device, dtype=field.means.dtype)
    if bool(torch.all(ramp >= 1.0)):
        return None
    snapshots = [(p, p.detach()[idx].clone()) for p in _row_parameter_tensors(field)]
    return idx, ramp, snapshots


@torch.no_grad()
def _apply_young_row_temper_snapshot(
    snapshot: tuple[torch.Tensor, torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]] | None,
) -> int:
    if snapshot is None:
        return 0
    idx, ramp, snapshots = snapshot
    for param, before in snapshots:
        shape = (ramp.shape[0],) + (1,) * (param[idx].ndim - 1)
        param[idx] = before + (param[idx] - before) * ramp.reshape(shape)
    return int(idx.numel())


def _take_new_parent_idx(field: GaussianField, count: int, base_n: int,
                         device: torch.device) -> torch.Tensor:
    fallback = torch.full((count,), -1, device=device, dtype=torch.long)
    parent = getattr(field, "_new_parent_idx", None)
    if hasattr(field, "_new_parent_idx"):
        delattr(field, "_new_parent_idx")
    if parent is None or int(parent.numel()) != int(count):
        return fallback
    parent = parent.to(device=device, dtype=torch.long)
    return torch.where((parent >= 0) & (parent < int(base_n)), parent, fallback)


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
                             cfg: FitConfig,
                             support_fade_alpha: float | None = None) -> torch.Tensor:
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
        w = _support_weight(q, cfg.sigma_cutoff, cfg.support_fade, support_fade_alpha)
        score.index_add_(0, gid, w * residual[py, px].to(dt))
        weight.index_add_(0, gid, w)
    return score / (weight + 1e-8)


@torch.no_grad()
def _responsibility_error_density_scores(
    field: GaussianField,
    target: torch.Tensor,
    render_img: torch.Tensor,
    cfg: FitConfig,
    support_fade_alpha: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Score normalized-renderer owners by responsibility-weighted squared error density.

    This intentionally mirrors the reference renderer's clipped support rectangles, Gaussian
    tail fade, opacity, AA/filter covariance, and denominator epsilon. The first pass forms the
    global per-pixel denominator; the second scatters normalized ownership mass and error back to
    Gaussian rows. Keeping the two passes explicit avoids treating raw support overlap as
    ownership when several Gaussians cover the same pixel.
    """
    if cfg.renderer not in _NORMALIZED_COLOR_SOLVE_RENDERERS:
        raise ValueError(
            "responsibility error density requires normalized renderer semantics, "
            f"got renderer={cfg.renderer!r}"
        )
    if target.shape != render_img.shape or target.ndim != 3 or target.shape[-1] != 3:
        raise ValueError(
            "target and render_img must have matching HxWx3 shapes, "
            f"got {tuple(target.shape)} and {tuple(render_img.shape)}"
        )

    H, W = target.shape[:2]
    dev, dt = field.means.device, field.means.dtype
    means = field.means.detach()
    conics = field.conics(cfg.aa_dilation).detach()
    radii = field.radii(cfg.sigma_cutoff, cfg.aa_dilation)
    opacities = field.opacity_values()
    if opacities is not None:
        opacities = opacities.detach().to(device=dev, dtype=dt)
    x0, y0, Tx, n = _tile_bounds(means, radii, H, W)
    budget = _element_budget(cfg.render_chunk)

    denominator = torch.zeros(H * W, device=dev, dtype=dt)
    for s, e in _flat_tile_slices(n, budget):
        gid, px, py = _tile_coords(x0, y0, Tx, n, s, e, dev)
        dx = px.to(dt) - means[gid, 0]
        dy = py.to(dt) - means[gid, 1]
        a, b, c = conics[gid, 0], conics[gid, 1], conics[gid, 2]
        q = a * dx * dx + 2.0 * b * dx * dy + c * dy * dy
        weight = _support_weight(
            q, cfg.sigma_cutoff, cfg.support_fade, support_fade_alpha
        )
        if opacities is not None:
            weight = weight * opacities[gid]
        denominator.index_add_(0, py * W + px, weight)

    pixel_error = (render_img.detach() - target.detach()).square().mean(dim=2).reshape(-1)
    pixel_error = pixel_error.to(device=dev, dtype=dt)
    error = torch.zeros(field.n, device=dev, dtype=dt)
    mass = torch.zeros(field.n, device=dev, dtype=dt)
    for s, e in _flat_tile_slices(n, budget):
        gid, px, py = _tile_coords(x0, y0, Tx, n, s, e, dev)
        dx = px.to(dt) - means[gid, 0]
        dy = py.to(dt) - means[gid, 1]
        a, b, c = conics[gid, 0], conics[gid, 1], conics[gid, 2]
        q = a * dx * dx + 2.0 * b * dx * dy + c * dy * dy
        weight = _support_weight(
            q, cfg.sigma_cutoff, cfg.support_fade, support_fade_alpha
        )
        if opacities is not None:
            weight = weight * opacities[gid]
        flat = py * W + px
        responsibility = weight / (denominator[flat] + _EPS)
        mass.index_add_(0, gid, responsibility)
        error.index_add_(0, gid, responsibility * pixel_error[flat])

    score = error / mass.clamp_min(_EPS).pow(float(cfg.responsibility_mass_alpha))
    return score, {
        "mass": mass,
        "error": error,
        "denominator": denominator.view(H, W),
    }


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
                 W: int, support_fade_alpha: float | None = None
                 ) -> tuple[GaussianField, torch.Tensor | None]:
    if cfg.prune_min_activity <= 0.0:
        return field, None
    if field.n <= max(1, cfg.prune_keep_min):
        return field, None
    activity = gaussian_activity(field.means, field.conics(cfg.aa_dilation),
                                 field.radii(cfg.sigma_cutoff, cfg.aa_dilation),
                                 H, W, cfg.render_chunk, cfg.support_fade, cfg.sigma_cutoff,
                                 support_fade_alpha=support_fade_alpha)
    # gaussian_activity is opacity-free; without this a Gaussian the optimizer has driven fully
    # transparent keeps its geometric weight-sum and is never pruned at fixed N (FIT-002).
    opac = field.opacity_values()
    if opac is not None:
        activity = activity * opac
    keep = activity > cfg.prune_min_activity
    bg = _background_mask(field)
    if bg is not None:
        keep = keep | bg.to(device=keep.device)
    if int(keep.sum()) < cfg.prune_keep_min:
        need = min(cfg.prune_keep_min, field.n) - int(keep.sum())
        activity_fill = activity.clone()
        activity_fill[keep] = -float("inf")
        if need > 0:
            topk = torch.topk(activity_fill, k=need).indices
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
    child_filter_variance = (
        None if field.filter_variance is None
        else field.filter_variance.detach()[idx].clone()
    )
    child = GaussianField(child_means, child_log_scales, child_rotations, child_colors,
                          child_opacities, child_scale_max, child_color_grads,
                          filter_variance=child_filter_variance)
    background_mask = (
        None if field.background_mask is None else field.background_mask.detach().clone()
    )
    grown = GaussianField(
        means, log_scales, rotations, colors, opacities, scale_max, color_grads,
        background_mask,
        None if field.filter_variance is None else field.filter_variance.detach().clone(),
    ).append(child)
    return _mark_new_parent_idx(grown, idx), k


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
    child_filter_variance = (
        None if field.filter_variance is None
        else field.filter_variance.detach()[idx].clone()
    )
    child = GaussianField(child_means, child_log_scales, child_rotations, child_colors,
                          child_opacities, child_scale_max, child_color_grads,
                          filter_variance=child_filter_variance)
    background_mask = (
        None if field.background_mask is None else field.background_mask.detach().clone()
    )
    grown = GaussianField(
        means, log_scales, rotations, colors, opacities, scale_max, color_grads,
        background_mask,
        None if field.filter_variance is None else field.filter_variance.detach().clone(),
    ).append(child)
    return _mark_new_parent_idx(grown, idx), k


@torch.no_grad()
def _simple_duplicate_indices(field: GaussianField, idx: torch.Tensor, target: torch.Tensor,
                              render_img: torch.Tensor, cfg: FitConfig, H: int, W: int
                              ) -> tuple[GaussianField, int]:
    k = int(idx.numel())
    if k <= 0:
        return field, 0

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
    filter_variance = (
        None if field.filter_variance is None
        else field.filter_variance.detach()[idx].clone()
    )
    new = GaussianField(
        means, log_scales, rotations, colors, opacities, scale_max,
        filter_variance=filter_variance,
    )
    return _mark_new_parent_idx(field.append(new), idx), k


@torch.no_grad()
def _duplicate_primitive_indices(field: GaussianField, idx: torch.Tensor, target: torch.Tensor,
                                 render_img: torch.Tensor, cfg: FitConfig, H: int, W: int,
                                 split_axis: torch.Tensor | None = None
                                 ) -> tuple[GaussianField, int]:
    if cfg.refine_primitive == "duplicate":
        return _simple_duplicate_indices(field, idx, target, render_img, cfg, H, W)
    if cfg.refine_primitive == "fp":
        return _fp_duplicate_indices(field, idx, cfg, H, W, split_axis=split_axis)
    if cfg.refine_primitive == "moment_preserving":
        return _moment_preserving_duplicate_indices(field, idx, cfg, H, W,
                                                   split_axis=split_axis)
    raise ValueError(
        "refine_primitive='sampled_add' is only valid for residual/residual_tensor "
        "sampled-add growth")


@torch.no_grad()
def _ranked_wave_scores(field: GaussianField, residual: torch.Tensor,
                        cfg: FitConfig,
                        support_fade_alpha: float | None = None
                        ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    H, W = residual.shape
    support = _support_residual_scores(
        field, residual, cfg, support_fade_alpha=support_fade_alpha
    )
    activity = gaussian_activity(
        field.means, field.conics(cfg.aa_dilation),
        field.radii(cfg.sigma_cutoff, cfg.aa_dilation), H, W, cfg.render_chunk,
        cfg.support_fade, cfg.sigma_cutoff, support_fade_alpha=support_fade_alpha,
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
                               cfg: FitConfig,
                               support_fade_alpha: float | None = None
                               ) -> tuple[GaussianField, int, dict[str, float]]:
    if cfg.split_count <= 0:
        return field, 0, {}
    if cfg.max_gaussians is not None and field.n >= cfg.max_gaussians:
        return field, 0, {}
    H, W = target.shape[:2]
    residual = (render_img - target).abs().mean(dim=2)
    room = field.n if cfg.max_gaussians is None else max(0, cfg.max_gaussians - field.n)
    k = min(cfg.split_count, _detail_count(field), room)
    if k <= 0:
        return field, 0, {}
    score, components = _ranked_wave_scores(
        field, residual, cfg, support_fade_alpha=support_fade_alpha
    )
    score = _mask_background_scores(score, field)
    idx = torch.topk(score, k=k).indices
    grown, added = _duplicate_primitive_indices(field, idx, target, render_img, cfg, H, W)
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
    k = min(cfg.split_count, _detail_count(field), room)
    if k <= 0:
        return field, 0, {}
    score, split_axis, components = _freq_violation_scores(field, target, render_img, cfg)
    score = _mask_background_scores(score, field)
    idx = torch.topk(score, k=k).indices
    grown, added = _duplicate_primitive_indices(
        field, idx, target, render_img, cfg, H, W, split_axis=split_axis[idx]
    )
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
def _absgrad_wave_from_scores(field: GaussianField, target: torch.Tensor,
                              render_img: torch.Tensor, grad_scores: torch.Tensor,
                              cfg: FitConfig, H: int, W: int
                              ) -> tuple[GaussianField, int, dict[str, float], torch.Tensor]:
    if cfg.split_count <= 0:
        return field, 0, {}, grad_scores
    if cfg.max_gaussians is not None and field.n >= cfg.max_gaussians:
        return field, 0, {}, grad_scores
    room = field.n if cfg.max_gaussians is None else max(0, cfg.max_gaussians - field.n)
    k = min(cfg.split_count, _detail_count(field), room)
    if k <= 0:
        return field, 0, {}, grad_scores
    scores = _mask_background_scores(grad_scores.detach(), field)
    idx = torch.topk(scores, k=k).indices
    grown, added = _duplicate_primitive_indices(
        field, idx, target, render_img, cfg, H, W
    )
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
                            cfg: FitConfig,
                            support_fade_alpha: float | None = None
                            ) -> tuple[GaussianField, int, torch.Tensor | None,
                                       dict[str, float | str]]:
    if cfg.relocate_count <= 0 or field.n <= 0:
        return field, 0, None, {}
    H, W = target.shape[:2]
    field = _ensure_generation_density_filter(field, cfg, H, W)
    k = min(cfg.relocate_count, _detail_count(field), int(H * W))
    if k <= 0:
        return field, 0, None, {}
    activity = gaussian_activity(
        field.means, field.conics(cfg.aa_dilation),
        field.radii(cfg.sigma_cutoff, cfg.aa_dilation), H, W, cfg.render_chunk,
        cfg.support_fade, cfg.sigma_cutoff, support_fade_alpha=support_fade_alpha,
    )
    opac = field.opacity_values()
    if opac is not None:
        activity = activity * opac
    activity = _mask_background_scores(activity, field, fill=float("inf"))
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
    filter_variance = (
        None if field.filter_variance is None
        else field.filter_variance.detach().clone()
    )
    if filter_variance is not None and cfg.covariance_filter_mode == "generation_density":
        filter_variance[low_idx] = _generation_density_filter_variance(cfg, H, W, field.n)
    stats = {
        "activity_mean": float(activity[low_idx].mean().detach().cpu()),
        "residual_mean": float(residual[pix].mean().detach().cpu()),
        "residual_downsample": float(cfg.relocate_residual_downsample),
        "activity_source": "gaussian_activity",
    }
    return (
        GaussianField(
            means,
            log_scales,
            rotations,
            colors,
            opacities,
            scale_max,
            color_grads,
            None if field.background_mask is None else field.background_mask.detach().clone(),
            filter_variance,
        ),
        k, low_idx, stats,
    )


@torch.no_grad()
def _split_from_residual_with_stats(
    field: GaussianField,
    target: torch.Tensor,
    render_img: torch.Tensor,
    cfg: FitConfig,
    support_fade_alpha: float | None = None,
    *,
    collect_stats: bool = True,
) -> tuple[GaussianField, int, dict[str, float | bool]]:
    if cfg.split_count <= 0:
        return field, 0, {}
    if cfg.max_gaussians is not None and field.n >= cfg.max_gaussians:
        return field, 0, {}
    if cfg.refine_site == "none":
        return field, 0, {}
    if cfg.refine_primitive == "sampled_add":
        if cfg.refine_site not in ("residual", "residual_tensor"):
            raise ValueError(
                "refine_primitive='sampled_add' requires refine_site residual or "
                f"residual_tensor, got {cfg.refine_site!r}")
        grown, added = _add_from_residual(
            field,
            target,
            render_img,
            cfg,
            tensor_aligned=cfg.refine_site == "residual_tensor",
        )
        return grown, added, {}

    score_start = time.perf_counter()
    H, W = target.shape[:2]
    residual = (render_img - target).abs().mean(dim=2)
    x = torch.clamp(torch.round(field.means[:, 0]).long(), 0, W - 1)
    y = torch.clamp(torch.round(field.means[:, 1]).long(), 0, H - 1)
    if cfg.refine_site == "support":
        scores = _support_residual_scores(
            field, residual, cfg, support_fade_alpha=support_fade_alpha
        )
        components = None
    elif cfg.refine_site == "responsibility":
        scores, components = _responsibility_error_density_scores(
            field, target, render_img, cfg, support_fade_alpha=support_fade_alpha
        )
    elif cfg.refine_site == "residual_tensor":
        smoothed = F.avg_pool2d(residual[None, None], 3, stride=1, padding=1)[0, 0]
        scores = (0.7 * residual + 0.3 * smoothed)[y, x]
        components = None
    elif cfg.refine_site == "residual":
        scores = residual[y, x]
        components = None
    else:
        raise ValueError(
            "refine_site must be residual, residual_tensor, support, or responsibility in "
            "_split_from_residual, "
            f"got {cfg.refine_site!r}")
    site_score_seconds = time.perf_counter() - score_start
    room = field.n if cfg.max_gaussians is None else max(0, cfg.max_gaussians - field.n)
    k = min(cfg.split_count, _detail_count(field), room)
    if k <= 0:
        return field, 0, {}

    scores_finite = (
        bool(torch.isfinite(scores).all().detach().cpu()) if collect_stats else False
    )
    scores = _mask_background_scores(scores, field)
    idx = torch.topk(scores, k=k).indices
    grown, added = _duplicate_primitive_indices(field, idx, target, render_img, cfg, H, W)
    if not collect_stats:
        return grown, added, {}
    stats: dict[str, float | bool] = {
        "site_score_mean": float(scores[idx].mean().detach().cpu()),
        "site_score_max": float(scores[idx].max().detach().cpu()),
        "site_scores_finite": scores_finite,
        "site_score_seconds": float(site_score_seconds),
    }
    if components is not None:
        selected_mass = components["mass"][idx]
        selected_error = components["error"][idx]
        stats.update({
            "responsibility_mass_alpha": float(cfg.responsibility_mass_alpha),
            "responsibility_mass_mean": float(selected_mass.mean().detach().cpu()),
            "responsibility_mass_min": float(selected_mass.min().detach().cpu()),
            "responsibility_mass_max": float(selected_mass.max().detach().cpu()),
            "responsibility_error_mean": float(selected_error.mean().detach().cpu()),
        })
    return grown, added, stats


@torch.no_grad()
def _split_from_residual(field: GaussianField, target: torch.Tensor, render_img: torch.Tensor,
                         cfg: FitConfig,
                         support_fade_alpha: float | None = None
                         ) -> tuple[GaussianField, int]:
    grown, added, _ = _split_from_residual_with_stats(
        field,
        target,
        render_img,
        cfg,
        support_fade_alpha=support_fade_alpha,
        collect_stats=False,
    )
    return grown, added


@torch.no_grad()
def _add_from_residual(field: GaussianField, target: torch.Tensor, render_img: torch.Tensor,
                       cfg: FitConfig, tensor_aligned: bool = False) -> tuple[GaussianField, int]:
    if cfg.split_count <= 0:
        return field, 0
    if cfg.max_gaussians is not None and field.n >= cfg.max_gaussians:
        return field, 0

    H, W = target.shape[:2]
    field = _ensure_generation_density_filter(field, cfg, H, W)
    room = field.n if cfg.max_gaussians is None else max(0, cfg.max_gaussians - field.n)
    k = min(cfg.split_count, int(H * W), room)
    if k <= 0:
        return field, 0

    # The score and the inserted cohort share this scale. Keeping the two coupled is the central
    # FIT-017 control: only site scoring changes relative to residual_tensor sampled-add growth.
    base_scale = math.sqrt((H * W) / max(field.n + k, 1)) * cfg.split_scale
    residual_map = (render_img - target).abs().mean(dim=2)
    if cfg.sampled_add_score == "signed_gaussian":
        residual = _matched_residual_score_map(target, render_img, base_scale).reshape(-1)
    elif cfg.sampled_add_score == "gaussian_abs":
        residual = _magnitude_residual_score_map(target, render_img, base_scale).reshape(-1)
    elif tensor_aligned:
        score_map = F.avg_pool2d(residual_map[None, None], 3, stride=1, padding=1)[0, 0]
        residual = (0.7 * residual_map + 0.3 * score_map).reshape(-1)
    else:
        residual = residual_map.reshape(-1)
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
    filter_variance = _new_cohort_filter_variance(field, cfg, H, W, k)
    return field.append(GaussianField(
        means,
        log_scales,
        rotations,
        colors,
        scale_max=scale_max,
        filter_variance=filter_variance,
    )), k


@torch.no_grad()
def _gaussian_residual_score_map(target: torch.Tensor, render_img: torch.Tensor,
                                 sigma_px: float, *, preserve_sign: bool) -> torch.Tensor:
    """Filter RGB residual with one planned isotropic Gaussian footprint.

    With ``preserve_sign=True``, per-channel convolution happens before the RGB norm, so adjacent
    errors of opposite sign cancel instead of looking like useful coherent support. False is the
    same-width magnitude-blur control. Kernels are truncated only where an offset cannot overlap
    the finite image; their common normalization does not affect ranking.
    """
    if target.shape != render_img.shape or target.ndim != 3 or target.shape[2] != 3:
        raise ValueError(
            "target and render_img must have matching (H, W, 3) shapes, got "
            f"{tuple(target.shape)} and {tuple(render_img.shape)}"
        )
    if not math.isfinite(sigma_px) or sigma_px <= 0.0:
        raise ValueError(f"sigma_px must be finite and > 0, got {sigma_px}")

    H, W = target.shape[:2]
    sigma = max(float(sigma_px), _MIN_DENSIFY_SCALE)
    residual = target - render_img
    if not preserve_sign:
        residual = residual.abs()
    residual = residual.permute(2, 0, 1).unsqueeze(0)

    def kernel(radius: int) -> torch.Tensor:
        if radius <= 0:
            return residual.new_ones(1)
        offsets = torch.arange(-radius, radius + 1, device=residual.device,
                               dtype=residual.dtype)
        weights = torch.exp(-0.5 * (offsets / sigma) ** 2)
        return weights / weights.sum().clamp_min(torch.finfo(weights.dtype).eps)

    radius = int(math.ceil(3.0 * sigma))
    radius_x = min(radius, max(W - 1, 0))
    radius_y = min(radius, max(H - 1, 0))
    if radius_x > 0:
        weights_x = kernel(radius_x).view(1, 1, 1, -1).expand(3, 1, 1, -1)
        residual = F.conv2d(residual, weights_x, padding=(0, radius_x), groups=3)
    if radius_y > 0:
        weights_y = kernel(radius_y).view(1, 1, -1, 1).expand(3, 1, -1, 1)
        residual = F.conv2d(residual, weights_y, padding=(radius_y, 0), groups=3)
    return torch.linalg.vector_norm(residual[0], dim=0)


@torch.no_grad()
def _matched_residual_score_map(target: torch.Tensor, render_img: torch.Tensor,
                                sigma_px: float) -> torch.Tensor:
    """Signed coherent-error score used by FIT-017."""
    return _gaussian_residual_score_map(
        target, render_img, sigma_px, preserve_sign=True
    )


@torch.no_grad()
def _magnitude_residual_score_map(target: torch.Tensor, render_img: torch.Tensor,
                                  sigma_px: float) -> torch.Tensor:
    """Same-width blurred-magnitude control for FIT-017."""
    return _gaussian_residual_score_map(
        target, render_img, sigma_px, preserve_sign=False
    )


def _raw_attribute_bits_per_gaussian(field: GaussianField) -> int:
    attrs = 2 + 2 + 1 + 3  # xy, log-scales, rotation, RGB
    if field.opacities is not None:
        attrs += 1
    if field.color_grads is not None:
        attrs += 6  # cx/cy RGB affine coefficients
    return attrs * 32


def estimated_raw_bpp(field: GaussianField, H: int, W: int) -> float:
    """Raw float32 attribute rate used as FIT-008's cheap in-loop rate proxy."""
    pixels = max(1, int(H) * int(W))
    return float(field.n * _raw_attribute_bits_per_gaussian(field)) / float(pixels)


def _adaptive_effective_max_gaussians(field: GaussianField, cfg: FitConfig,
                                      H: int, W: int) -> int | None:
    cap = cfg.max_gaussians
    if cfg.target_bpp is not None:
        bits = max(1, _raw_attribute_bits_per_gaussian(field))
        rate_cap = max(1, int(math.floor(float(cfg.target_bpp) * H * W / bits)))
        cap = rate_cap if cap is None else min(int(cap), rate_cap)
    return cap


def _adaptive_stop_reason(cfg: FitConfig, field: GaussianField, H: int, W: int,
                          psnr_now: float, ms_ssim_now: float | None,
                          stale_waves: int) -> str | None:
    if cfg.target_psnr is not None and psnr_now >= float(cfg.target_psnr):
        return "target_psnr_reached"
    if cfg.target_ms_ssim is not None and ms_ssim_now is not None \
            and ms_ssim_now >= float(cfg.target_ms_ssim):
        return "target_ms_ssim_reached"
    if cfg.target_bpp is not None and estimated_raw_bpp(field, H, W) >= float(cfg.target_bpp):
        return "target_bpp_reached"
    cap = _adaptive_effective_max_gaussians(field, cfg, H, W)
    if cap is not None and field.n >= cap:
        return "max_gaussians_reached"
    if stale_waves >= cfg.adaptive_patience:
        return "stalled"
    return None


@torch.no_grad()
def _adaptive_growth_from_residual(field: GaussianField, target: torch.Tensor,
                                   render_img: torch.Tensor, cfg: FitConfig,
                                   H: int, W: int,
                                   support_fade_alpha: float | None = None
                                   ) -> tuple[GaussianField, int, dict[str, float]]:
    cap = _adaptive_effective_max_gaussians(field, cfg, H, W)
    room = field.n if cap is None else max(0, cap - field.n)
    k = min(int(cfg.adaptive_growth_count), int(room))
    if k <= 0:
        return field, 0, {}
    grow_cfg = replace(
        cfg,
        split_count=k,
        split_mode=cfg.adaptive_split_mode,
        refine_site=None,
        refine_primitive=None,
        refine_nms=None,
        max_gaussians=field.n + k,
    )
    if cfg.adaptive_split_mode == "ranked_wave":
        return _ranked_wave_from_residual(
            field, target, render_img, grow_cfg, support_fade_alpha=support_fade_alpha
        )
    if cfg.adaptive_split_mode == "freq_violation":
        return _freq_violation_from_residual(field, target, render_img, grow_cfg)
    if cfg.adaptive_split_mode in ("residual_add", "residual_tensor_add"):
        grown, added = _add_from_residual(
            field,
            target,
            render_img,
            grow_cfg,
            tensor_aligned=cfg.adaptive_split_mode == "residual_tensor_add",
        )
        return grown, added, {}
    return (*_split_from_residual(
        field, target, render_img, grow_cfg, support_fade_alpha=support_fade_alpha
    ), {})


def fit(field: GaussianField, target: torch.Tensor, cfg: FitConfig, verbose: bool = True,
        sched_offset: int = 0, sched_total: int | None = None, loss_weight_map=None,
        mask=None, iteration_observer=None, observer_every: int = 0) -> dict:
    checkpoint_enabled = cfg.checkpoint_policy == "best_psnr_final_count"
    if checkpoint_enabled and (
        sched_offset != 0 or (sched_total is not None and sched_total != cfg.iters)
    ):
        raise ValueError(
            "checkpoint_policy='best_psnr_final_count' currently supports single-stage fits "
            "only; global schedule offsets would make selected-iteration metadata ambiguous"
        )
    _validate_loss_target_schedule(
        cfg,
        sched_offset=sched_offset,
        sched_total=sched_total,
    )
    H, W = target.shape[0], target.shape[1]
    field = _ensure_generation_density_filter(field, cfg, H, W)
    # CORE-010 mask containment / coverage penalty / mask loss weighting. Built once; the EDT is
    # the only meaningful cost and it is opt-in.
    mask_constraint = None
    if cfg.mask_contain or cfg.mask_coverage_weight > 0.0 or cfg.loss_weighting == "mask":
        if mask is None:
            raise ValueError(
                "mask_contain, mask_coverage_weight>0, and loss_weighting='mask' require a mask; "
                "pass mask= to fit()")
        mask_constraint = _MaskConstraint.from_mask(
            mask, target.device, target.dtype, cfg.sigma_cutoff, cfg.mask_margin,
            aa_dilation=cfg.aa_dilation,
        )
        if tuple(mask_constraint.inside.shape) != (H, W):
            raise ValueError(
                f"mask shape {tuple(mask_constraint.inside.shape)} does not match target "
                f"{(H, W)}")
    weight_map_arg = loss_weight_map
    if cfg.loss_weighting == "mask":
        weight_map_arg = mask_constraint.inside_weight
    pixel_weight = _prepare_loss_weight_map(target, cfg, weight_map_arg)
    ssim_mask = (
        mask_constraint.inside_weight.reshape(H, W, 1)
        if (mask_constraint is not None and cfg.loss_weighting == "mask")
        else None
    )
    lowpass_loss_target = _prepare_loss_target(target, cfg.loss_target_downsample)
    target_geometry_gradients = (
        tuple(gradient.detach() for gradient in _sobel_gradients(target))
        if cfg.geometry_loss_weight > 0.0
        else None
    )
    field = _ensure_color_basis(field, cfg)
    if _qat_enabled(cfg) and field.color_grads is not None:
        raise ValueError(
            "fit-time QAT currently supports color_basis='constant' only; "
            "codec v1 cannot encode affine color fields"
        )
    qat_ccfg = _make_qat_codec_config(field, cfg) if _qat_enabled(cfg) else None
    if _color_solve_enabled(cfg) and not _color_solve_renderer_supported(cfg.renderer):
        raise ValueError(
            "color_solve_every currently supports normalized renderers only; "
            f"got {cfg.renderer!r}"
        )
    if _color_solve_enabled(cfg) and field.color_grads is not None:
        raise ValueError(
            "color_solve_every currently supports color_basis='constant' only; "
            "affine color coefficients are optimized with Adam"
        )
    field.trainable()
    if mask_constraint is not None and cfg.mask_contain:
        mask_constraint.apply(field, cfg)
    opt = _make_optimizer(field, cfg)
    base_lrs = [g["lr"] for g in opt.param_groups]
    lo, hi = math.log(0.35), math.log(max(H, W))
    hist = {
        "iter": [], "psnr": [], "loss": [], "n_gaussians": [], "elapsed": [],
        "split_events": [], "relocate_events": [], "color_solve_events": [],
        "adaptive_events": [], "qat_rate_bpp": [], "rate_loss": [],
        "support_fade_alpha": [], "tempered_new_rows": [], "geometry_loss": [],
        "loss_target_full_weight": [],
    }
    checkpoint_history = (
        {"iter": [], "psnr": [], "n_gaussians": [], "terminal": []}
        if checkpoint_enabled
        else None
    )
    best_checkpoints_by_count: dict[int, dict] = {}
    color_solve_tokens = _color_solve_schedule_tokens(cfg)
    start_time = time.time()
    row_birth_iter = torch.full((field.n,), -1, device=target.device, dtype=torch.long)

    def run_color_solve_event(iter_idx: int, trigger: str,
                              support_fade_alpha: float | None = None) -> None:
        stats = _solve_colors_normalized(
            field, target, cfg, H, W, support_fade_alpha=support_fade_alpha
        )
        _reset_optimizer_state_for_param(opt, field.colors)
        hist["color_solve_events"].append({
            "iter": iter_idx,
            "trigger": trigger,
            "support_fade_alpha": (
                None if support_fade_alpha is None else float(support_fade_alpha)
            ),
            **stats,
        })
        if verbose:
            print(
                "  color solve "
                f"{trigger} {int(stats['iterations'])} cg iters  "
                f"rel {float(stats['relative_residual']):.3e}"
            )

    def record_checkpoint(
        iteration: int,
        support_fade_alpha: float,
        *,
        render_img: torch.Tensor | None = None,
        psnr: float | None = None,
        terminal: bool = False,
    ) -> tuple[torch.Tensor, float]:
        """Record one post-transition state and retain the best snapshot for its count."""
        if checkpoint_history is None:
            raise RuntimeError("checkpoint recording called while checkpoint selection is off")
        with torch.no_grad():
            if render_img is None:
                render_img = _render(
                    field, cfg, H, W, support_fade_alpha=support_fade_alpha
                )
            if psnr is None:
                psnr = float(M.psnr(render_img, target))
            else:
                psnr = float(psnr)
        count = int(field.n)
        same_state_as_last = bool(
            checkpoint_history["iter"]
            and checkpoint_history["iter"][-1] == int(iteration)
            and checkpoint_history["n_gaussians"][-1] == count
        )
        if same_state_as_last:
            checkpoint_history["psnr"][-1] = psnr
            checkpoint_history["terminal"][-1] = bool(terminal)
        else:
            checkpoint_history["iter"].append(int(iteration))
            checkpoint_history["psnr"].append(psnr)
            checkpoint_history["n_gaussians"].append(count)
            checkpoint_history["terminal"].append(bool(terminal))

        previous = best_checkpoints_by_count.get(count)
        # The final evaluation is authoritative for an already-recorded identical state. In a
        # tie it also wins, avoiding a needless restore of terminal parameters from a clone.
        same_state_as_best = bool(
            previous is not None and int(previous["iter"]) == int(iteration)
        )
        if (
            previous is None
            or psnr > float(previous["psnr"])
            or (terminal and (same_state_as_best or psnr == float(previous["psnr"])))
        ):
            best_checkpoints_by_count[count] = {
                "iter": int(iteration),
                "psnr": psnr,
                "n_gaussians": count,
                # A terminal winner can reuse the live field. Earlier winners need an exact
                # deep tensor snapshot so later optimizer steps cannot mutate the selected state.
                "field": None if terminal else field.detached(),
                "terminal": bool(terminal),
            }
        return render_img, psnr

    if "init" in color_solve_tokens:
        run_color_solve_event(0, "init", _fit_support_fade_alpha(cfg, 0))
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
    best_logged_psnr = -math.inf
    stale_logs = 0
    stopped_early = False
    adaptive_prev_psnr: float | None = None
    adaptive_stale_waves = 0
    adaptive_stop = None
    adaptive_finished = False
    last_iter = -1

    for it in range(cfg.iters):
        last_iter = it
        fade_alpha = _fit_support_fade_alpha(cfg, it)
        factor = _lr_factor(cfg, it, sched_offset, sched_total)
        for g, base in zip(opt.param_groups, base_lrs):
            g["lr"] = base * factor
        img = _render_loss_view(
            field, cfg, H, W, qat_ccfg, support_fade_alpha=fade_alpha
        )
        # count that produced this render/PSNR, before any prune/split restructures the field;
        # logging the post-restructure field.n would pair a pre-step PSNR with a post-step N.
        n_at_render = field.n
        loss_target_full_weight = _loss_target_full_weight(
            cfg,
            it,
            sched_offset=sched_offset,
            sched_total=sched_total,
        )
        if loss_target_full_weight <= 0.0:
            objective_target = lowpass_loss_target
        elif loss_target_full_weight >= 1.0:
            objective_target = target
        else:
            objective_target = torch.lerp(
                lowpass_loss_target,
                target,
                loss_target_full_weight,
            )
        pix = _weighted_pixel_loss(
            img,
            objective_target,
            _loss_kind(cfg, it),
            cfg.charbonnier_eps,
            pixel_weight,
        )
        if cfg.ssim_weight == 0.0:
            loss = pix
        else:
            if ssim_mask is not None:
                # Matte both sides so boundary SSIM windows compare consistent zeros; with hard
                # containment the render is already ~0 outside, so this only zeroes the target.
                s = M.ssim(img * ssim_mask, objective_target * ssim_mask,
                           backend=cfg.ssim_backend)
            else:
                s = M.ssim(img, objective_target, backend=cfg.ssim_backend)
            loss = (1 - cfg.ssim_weight) * pix + cfg.ssim_weight * (1 - s)
        geometry_loss = None
        if (
            target_geometry_gradients is not None
            and (sched_offset + it) % int(cfg.geometry_loss_every) == 0
        ):
            geometry_loss = _geometry_consistent_loss(img, target_geometry_gradients)
            # Preserve the requested average regularization strength when evaluating the
            # expensive gradient term intermittently.
            active_weight = float(cfg.geometry_loss_weight) * int(cfg.geometry_loss_every)
            loss = loss + active_weight * geometry_loss
        if field.color_grads is not None and cfg.color_grad_l2 > 0.0:
            loss = loss + cfg.color_grad_l2 * (field.color_grads * field.color_grads).mean()
        if mask_constraint is not None and cfg.mask_coverage_weight > 0.0:
            loss = loss + float(cfg.mask_coverage_weight) * mask_constraint.coverage(
                field, cfg, support_fade_alpha=fade_alpha)
        rate_bpp = None
        if cfg.lambda_rate > 0.0:
            rate_bpp = differentiable_rate_bpp(field, H, W, cfg)
            loss = loss + float(cfg.lambda_rate) * rate_bpp
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.refine_site == "absgrad" and field.means.grad is not None:
            with torch.no_grad():
                absgrad_scores.mul_(cfg.absgrad_decay).add_(
                    field.means.grad.detach().abs().sum(dim=1)
                )
        temper_snapshot = _capture_young_row_temper_snapshots(field, row_birth_iter, it, cfg)
        _zero_background_geometry_state(field, opt)
        opt.step()
        tempered_count = _apply_young_row_temper_snapshot(temper_snapshot)
        last_it = it == cfg.iters - 1
        log_now = it % cfg.log_every == 0 or it == cfg.iters - 1
        adaptive_due = (
            cfg.adaptive_count and not adaptive_finished and not last_it
            and (it + 1) % int(cfg.adaptive_growth_every) == 0
        )
        with torch.no_grad():
            field.log_scales.clamp_(lo, hi)
            if getattr(field, "scale_max", None) is not None:
                cap = torch.log(torch.clamp(field.scale_max, min=1e-3))
                torch.minimum(field.log_scales, cap, out=field.log_scales)
            if mask_constraint is not None and cfg.mask_contain:
                # Re-contain after the optimizer moved means/scales: project drift back inside and
                # refresh the per-row cap from the current signed distance.
                mask_constraint.apply(field, cfg)
            if (
                "every" in color_solve_tokens
                and cfg.color_solve_every is not None
                and cfg.color_solve_every > 0
                and (it + 1) % int(cfg.color_solve_every) == 0
            ):
                run_color_solve_event(it, "every", fade_alpha)
                img = _render(field, cfg, H, W, support_fade_alpha=fade_alpha)
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
            elif adaptive_due:
                if mse_now is None:
                    mse_now = M.mse(img, target).detach().clamp_min(1e-12)
                p_now = float(M.psnr_from_mse(mse_now))

        keep = None
        added = 0
        relocated = 0
        reset_idx = None
        state_seed_base_n = field.n
        new_parent_idx_parts: list[torch.Tensor] = []
        # never restructure on the final iteration: the returned field/metrics would include
        # Gaussians that no optimizer step ever touched
        split_due = (not last_it and cfg.split_every is not None and cfg.split_every > 0
                     and cfg.split_count > 0 and (it + 1) % cfg.split_every == 0
                     and not (
                         cfg.split_schedule_stops_at_max
                         and cfg.max_gaussians is not None
                         and field.n >= cfg.max_gaussians
                     ))
        relocate_periodic_due = (
            not last_it and cfg.relocate_every is not None and cfg.relocate_every > 0
            and (it + 1) % cfg.relocate_every == 0
        )
        relocate_split_due = bool(cfg.relocate_at_split and split_due)
        if (not last_it and cfg.prune_every is not None and cfg.prune_every > 0
                and (it + 1) % cfg.prune_every == 0):
            field, keep = _maybe_prune(field, cfg, H, W, support_fade_alpha=fade_alpha)
            if verbose and keep is not None:
                print(f"  prune {int((~keep).sum())} -> {field.n} gaussians")
            if keep is not None and cfg.refine_site == "absgrad":
                absgrad_scores = absgrad_scores[keep]
            if keep is not None:
                row_birth_iter = row_birth_iter[keep]
                state_seed_base_n = field.n
        if cfg.relocate_count > 0 and (relocate_periodic_due or relocate_split_due):
            field, relocated, reset_idx, relocate_stats = _relocate_from_residual(
                field, target, img, cfg, support_fade_alpha=fade_alpha
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
                if cfg.refine_site == "absgrad" and reset_idx is not None:
                    absgrad_scores[reset_idx] = 0
                if reset_idx is not None and cfg.new_row_temper_iters > 0:
                    row_birth_iter[reset_idx.to(device=row_birth_iter.device)] = it + 1
        if split_due:
            split_event = None
            absgrad_scores_carried = False
            mode_label = cfg.split_mode
            if cfg.refine_site in ("none", "residual", "residual_tensor", "support"):
                field, added = _split_from_residual(
                    field, target, img, cfg, support_fade_alpha=fade_alpha
                )
                if added > 0:
                    new_parent_idx_parts.append(
                        _take_new_parent_idx(field, added, state_seed_base_n, target.device)
                    )
            elif cfg.refine_site == "responsibility":
                field, added, responsibility_stats = _split_from_residual_with_stats(
                    field, target, img, cfg, support_fade_alpha=fade_alpha
                )
                if added > 0:
                    new_parent_idx_parts.append(
                        _take_new_parent_idx(field, added, state_seed_base_n, target.device)
                    )
                split_event = {
                    "iter": it,
                    "mode": mode_label,
                    "refine_site": cfg.refine_site,
                    "refine_primitive": cfg.refine_primitive,
                    "added": added,
                    **responsibility_stats,
                }
            elif cfg.refine_site == "ranked":
                field, added, ranked_stats = _ranked_wave_from_residual(
                    field, target, img, cfg, support_fade_alpha=fade_alpha
                )
                if added > 0:
                    new_parent_idx_parts.append(
                        _take_new_parent_idx(field, added, state_seed_base_n, target.device)
                    )
                split_event = {
                    "iter": it,
                    "mode": mode_label,
                    "refine_site": cfg.refine_site,
                    "refine_primitive": cfg.refine_primitive,
                    "added": added,
                    **ranked_stats,
                }
            elif cfg.refine_site == "absgrad":
                field, added, absgrad_stats, absgrad_scores = _absgrad_wave_from_scores(
                    field, target, img, absgrad_scores, cfg, H, W
                )
                if added > 0:
                    new_parent_idx_parts.append(
                        _take_new_parent_idx(field, added, state_seed_base_n, target.device)
                    )
                absgrad_scores_carried = True
                split_event = {
                    "iter": it,
                    "mode": mode_label,
                    "refine_site": cfg.refine_site,
                    "refine_primitive": cfg.refine_primitive,
                    "added": added,
                    **absgrad_stats,
                }
            elif cfg.refine_site == "freq_violation":
                field, added, freq_stats = _freq_violation_from_residual(
                    field, target, img, cfg
                )
                if added > 0:
                    new_parent_idx_parts.append(
                        _take_new_parent_idx(field, added, state_seed_base_n, target.device)
                    )
                split_event = {
                    "iter": it,
                    "mode": mode_label,
                    "refine_site": cfg.refine_site,
                    "refine_primitive": cfg.refine_primitive,
                    "added": added,
                    **freq_stats,
                }
            else:
                raise ValueError(
                    f"unknown refine_site {cfg.refine_site!r}; expected one of "
                    "none, residual, residual_tensor, support, responsibility, ranked, "
                    "absgrad, or freq_violation")
            if verbose and added > 0:
                print(f"  split +{added} -> {field.n} gaussians")
            if added > 0:
                hist["split_events"].append(
                    split_event or {
                        "iter": it,
                        "mode": mode_label,
                        "refine_site": cfg.refine_site,
                        "refine_primitive": cfg.refine_primitive,
                        "added": added,
                    }
                )
                if cfg.refine_site == "absgrad" and not absgrad_scores_carried:
                    absgrad_scores = torch.cat(
                        [absgrad_scores, absgrad_scores.new_zeros(added)], dim=0
                    )
        adaptive_should_stop = False
        if adaptive_due:
            ms_ssim_now = None
            if cfg.target_ms_ssim is not None:
                with torch.no_grad():
                    ms_ssim_now = float(M.ms_ssim(img, target))
            marginal = None if adaptive_prev_psnr is None else p_now - adaptive_prev_psnr
            if marginal is not None:
                if marginal < cfg.adaptive_min_delta_psnr:
                    adaptive_stale_waves += 1
                else:
                    adaptive_stale_waves = 0
            reason = _adaptive_stop_reason(
                cfg, field, H, W, p_now, ms_ssim_now, adaptive_stale_waves
            )
            event = {
                "iter": it,
                "psnr": p_now,
                "n_gaussians": field.n,
                "estimated_bpp": estimated_raw_bpp(field, H, W),
                "marginal_psnr": marginal,
                "stale_waves": adaptive_stale_waves,
            }
            if ms_ssim_now is not None:
                event["ms_ssim"] = ms_ssim_now
            if reason is not None:
                adaptive_stop = reason
                hist["adaptive_events"].append({**event, "action": "stop", "reason": reason})
                adaptive_finished = True
                adaptive_should_stop = not cfg.adaptive_continue_after_stop
                if verbose:
                    print(f"  adaptive stop {reason} at {field.n} gaussians")
            else:
                field, adaptive_added, adaptive_stats = _adaptive_growth_from_residual(
                    field, target, img, cfg, H, W, support_fade_alpha=fade_alpha
                )
                if adaptive_added <= 0:
                    adaptive_stop = "no_growth"
                    hist["adaptive_events"].append(
                        {**event, "action": "stop", "reason": adaptive_stop}
                    )
                    adaptive_finished = True
                    adaptive_should_stop = not cfg.adaptive_continue_after_stop
                else:
                    if adaptive_added > 0:
                        new_parent_idx_parts.append(
                            _take_new_parent_idx(
                                field, adaptive_added, state_seed_base_n, target.device
                            )
                        )
                    growth_event = {
                        **event,
                        "action": "grow",
                        "mode": cfg.adaptive_split_mode,
                        "added": adaptive_added,
                        "n_after": field.n,
                        **adaptive_stats,
                    }
                    hist["adaptive_events"].append(growth_event)
                    hist["split_events"].append(
                        {
                            "iter": it,
                            "mode": cfg.adaptive_split_mode,
                            "trigger": "adaptive",
                            "added": adaptive_added,
                            **adaptive_stats,
                        }
                    )
                    if verbose:
                        print(f"  adaptive grow +{adaptive_added} -> {field.n} gaussians")
                    if cfg.refine_site == "absgrad":
                        absgrad_scores = torch.cat(
                            [absgrad_scores, absgrad_scores.new_zeros(adaptive_added)], dim=0
                        )
                    added += adaptive_added
            adaptive_prev_psnr = p_now
        if keep is not None or added > 0 or relocated > 0:
            if added > 0:
                row_birth_iter = torch.cat([
                    row_birth_iter,
                    torch.full((added,), it + 1, device=row_birth_iter.device,
                               dtype=row_birth_iter.dtype),
                ])
            new_parent_idx = None
            if new_parent_idx_parts:
                new_parent_idx = torch.cat(new_parent_idx_parts, dim=0)
                if new_parent_idx.numel() != added:
                    new_parent_idx = None
            field.trainable()
            opt = _carry_adam_state(
                opt, field, cfg, keep, added, reset_idx=reset_idx,
                new_parent_idx=new_parent_idx,
            )
            base_lrs = [g["lr"] for g in opt.param_groups]
            if mask_constraint is not None and cfg.mask_contain:
                # New/relocated rows inherit caps from neighbours; re-derive them from the mask so
                # split/relocation/growth children are contained before the next render.
                mask_constraint.apply(field, cfg)
            if "on_split" in color_solve_tokens and (added > 0 or relocated > 0):
                run_color_solve_event(it, "on_split", fade_alpha)

        # This is deliberately separate from the legacy trajectory history below. That history
        # is a pre-step render paired with n_at_render; checkpoint selection evaluates the actual
        # field after the optimizer step and every prune/split/relocation/color-solve transition.
        if checkpoint_enabled and log_now and not last_it:
            record_checkpoint(
                it + 1,
                _fit_support_fade_alpha(cfg, it + 1),
            )

        if (
            iteration_observer is not None
            and observer_every > 0
            and ((it + 1) % observer_every == 0 or it + 1 == cfg.iters)
        ):
            # Read-only observer of the post-step, post-restructure field (e.g. the igsv
            # live-viewer bridge, ADR-0018). no_grad so an observer cannot extend the
            # autograd graph; it must not mutate the field or touch RNG state.
            with torch.no_grad():
                iteration_observer(field, sched_offset + it + 1, float(loss.item()))

        if log_now:
            hist["iter"].append(it)
            hist["psnr"].append(p_now)
            hist["loss"].append(loss.item())
            hist["n_gaussians"].append(n_at_render)
            hist["elapsed"].append(time.time() - start_time)
            if rate_bpp is None:
                hist["qat_rate_bpp"].append(None)
                hist["rate_loss"].append(None)
            else:
                rb = float(rate_bpp.detach().cpu())
                hist["qat_rate_bpp"].append(rb)
                hist["rate_loss"].append(float(cfg.lambda_rate) * rb)
            hist["support_fade_alpha"].append(float(fade_alpha))
            hist["tempered_new_rows"].append(tempered_count)
            hist["geometry_loss"].append(
                None if geometry_loss is None else float(geometry_loss.detach().cpu())
            )
            hist["loss_target_full_weight"].append(float(loss_target_full_weight))
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
            if adaptive_should_stop:
                break

    final_fade_alpha = _fit_support_fade_alpha(cfg, last_iter + 1)
    if "final" in color_solve_tokens:
        run_color_solve_event(last_iter, "final", final_fade_alpha)

    fit_seconds = time.time() - start_time  # before the final eval: metrics (esp. LPIPS
    with torch.no_grad():                   # model construction) must not pollute the timing
        if target_iters_device is not None:
            reached = target_iters_device.detach().cpu().tolist()
            iters_to_targets = {
                str(t): (None if int(v) < 0 else int(v)) for t, v in zip(targets, reached)
            }
        terminal_field = field
        if mask_constraint is not None and cfg.mask_contain:
            mask_constraint.apply(field, cfg)
        terminal_img = _render(field, cfg, H, W, support_fade_alpha=final_fade_alpha)
        terminal_psnr = M.psnr(terminal_img, target)
        terminal_ssim = float(M.ssim(terminal_img, target, backend=cfg.ssim_backend))
        terminal_ms_ssim = M.ms_ssim(terminal_img, target)
        terminal_lpips = (
            M.LPIPS.distance(terminal_img, target) if cfg.compute_lpips else None
        )
        terminal_iter = last_iter + 1
        terminal_n = int(field.n)
        selected_record = None
        if checkpoint_enabled:
            record_checkpoint(
                terminal_iter,
                final_fade_alpha,
                render_img=terminal_img,
                psnr=terminal_psnr,
                terminal=True,
            )
            selected_record = best_checkpoints_by_count[terminal_n]
            if int(selected_record["n_gaussians"]) != terminal_n:
                raise RuntimeError("checkpoint selection violated the terminal-count invariant")
            if selected_record["field"] is not None:
                field = selected_record["field"].trainable()
                if mask_constraint is not None and cfg.mask_contain:
                    mask_constraint.apply(field, cfg)
                img = _render(field, cfg, H, W, support_fade_alpha=final_fade_alpha)
                final_psnr = M.psnr(img, target)
                final_ssim = float(M.ssim(img, target, backend=cfg.ssim_backend))
                final_ms_ssim = M.ms_ssim(img, target)
                final_lpips = M.LPIPS.distance(img, target) if cfg.compute_lpips else None
            else:
                img = terminal_img
                final_psnr = terminal_psnr
                final_ssim = terminal_ssim
                final_ms_ssim = terminal_ms_ssim
                final_lpips = terminal_lpips
        else:
            img = terminal_img
            final_psnr = terminal_psnr
            final_ssim = terminal_ssim
            final_ms_ssim = terminal_ms_ssim
            final_lpips = terminal_lpips
            selected_record = {
                "iter": terminal_iter,
                "psnr": float(terminal_psnr),
                "n_gaussians": terminal_n,
                "field": None,
                "terminal": True,
            }
        if cfg.adaptive_count and adaptive_stop is None:
            adaptive_stop = _adaptive_stop_reason(
                cfg,
                terminal_field,
                H,
                W,
                terminal_psnr,
                terminal_ms_ssim,
                adaptive_stale_waves,
            ) or "iteration_limit"
        if cfg.adaptive_count:
            hist["adaptive_stop_reason"] = adaptive_stop
            hist["adaptive_selected_n"] = terminal_n
        out = {
            "field": field, "history": hist, "render": img,
            "psnr": final_psnr,
            "ssim": final_ssim,
            "ms_ssim": final_ms_ssim,
            "lpips": final_lpips,
            "iters_to_target": (
                iters_to_targets.get(str(float(cfg.target_psnr)))
                if cfg.target_psnr is not None else None
            ),
            "iters_to_targets": iters_to_targets,
            "n_gaussians": field.n,
            "background_count": field.background_count,
            "detail_count": field.n - field.background_count,
            "fit_seconds": fit_seconds,
            "iterations_run": last_iter + 1,
            "stopped_early": stopped_early,
            "stopped_at": last_iter if stopped_early else None,
            "adaptive_stop_reason": adaptive_stop if cfg.adaptive_count else None,
            "adaptive_selected_n": field.n if cfg.adaptive_count else None,
            "estimated_bpp": estimated_raw_bpp(field, H, W),
            "qat_mode": cfg.qat_mode,
            "lambda_rate": float(cfg.lambda_rate),
            "qat_rate_bpp": float(differentiable_rate_bpp(field, H, W, cfg).detach().cpu())
            if cfg.lambda_rate > 0.0 else None,
            "qat_codec_config": qat_ccfg,
            "loss_weighting": cfg.loss_weighting,
            "loss_weight_beta": float(cfg.loss_weight_beta),
            "loss_target_downsample": int(cfg.loss_target_downsample),
            "loss_target_full_frac": float(cfg.loss_target_full_frac),
            "geometry_loss_weight": float(cfg.geometry_loss_weight),
            "geometry_loss_every": int(cfg.geometry_loss_every),
            "lr_schedule": cfg.lr_schedule,
            "lr_cosine_start_frac": float(cfg.lr_cosine_start_frac),
            "checkpoint_policy": cfg.checkpoint_policy,
            "checkpoint_history": checkpoint_history,
            "trajectory_terminal_iter": terminal_iter,
            "trajectory_terminal_n_gaussians": terminal_n,
            "trajectory_terminal_psnr": float(terminal_psnr),
            "trajectory_terminal_ssim": terminal_ssim,
            "trajectory_terminal_ms_ssim": float(terminal_ms_ssim),
            "trajectory_terminal_lpips": terminal_lpips,
            "selected_iter": int(selected_record["iter"]),
            "selected_n_gaussians": int(selected_record["n_gaussians"]),
            "selected_checkpoint_psnr": float(selected_record["psnr"]),
            "selected_psnr": float(final_psnr),
            "selected_from_checkpoint": selected_record["field"] is not None,
            "covariance_filter_mode": cfg.covariance_filter_mode,
            "covariance_filter_active": field.filter_variance is not None,
            "covariance_filter_alpha": float(cfg.covariance_filter_alpha),
            "covariance_filter_max_variance": float(cfg.covariance_filter_max_variance),
            "covariance_filter_cohorts": (
                0 if field.filter_variance is None
                else int(torch.unique(field.filter_variance.detach()).numel())
            ),
            "covariance_filter_variance_min": (
                None if field.filter_variance is None
                else float(field.filter_variance.min().detach().cpu())
            ),
            "covariance_filter_variance_max": (
                None if field.filter_variance is None
                else float(field.filter_variance.max().detach().cpu())
            ),
            "loss_weight_mean": (
                None if pixel_weight is None else float(pixel_weight.mean().detach().cpu())
            ),
            "loss_weight_max": (
                None if pixel_weight is None else float(pixel_weight.max().detach().cpu())
            ),
        }
    return out
