"""Reconstruction metrics: PSNR and SSIM/MS-SSIM built-in (torch, no external deps).

LPIPS is optional (pip install lpips); import is guarded so the repo runs without it.
Images are (H,W,3) or (B,3,H,W) in [0,1]. BENCH-001 fixes the exact protocol.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F


def _to_bchw(x):
    if x.dim() == 3:
        x = x.permute(2, 0, 1).unsqueeze(0)
    return x


def psnr(pred, target, max_val: float = 1.0) -> float:
    mse = F.mse_loss(pred, target).clamp_min(1e-12)
    return float(10.0 * torch.log10(max_val * max_val / mse))


def _gaussian_window(win: int, sigma: float, device, dtype):
    x = torch.arange(win, device=device, dtype=dtype) - (win - 1) / 2.0
    g = torch.exp(-(x ** 2) / (2 * sigma ** 2))
    g = (g / g.sum()).unsqueeze(0)
    w2 = (g.t() @ g)
    return w2.expand(3, 1, win, win).contiguous()


def ssim(pred, target, win: int = 11, sigma: float = 1.5) -> torch.Tensor:
    p, t = _to_bchw(pred), _to_bchw(target)
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    w = _gaussian_window(win, sigma, p.device, p.dtype)
    pad = win // 2
    mu_p = F.conv2d(p, w, padding=pad, groups=3)
    mu_t = F.conv2d(t, w, padding=pad, groups=3)
    mu_p2, mu_t2, mu_pt = mu_p * mu_p, mu_t * mu_t, mu_p * mu_t
    sig_p = F.conv2d(p * p, w, padding=pad, groups=3) - mu_p2
    sig_t = F.conv2d(t * t, w, padding=pad, groups=3) - mu_t2
    sig_pt = F.conv2d(p * t, w, padding=pad, groups=3) - mu_pt
    s = ((2 * mu_pt + C1) * (2 * sig_pt + C2)) / ((mu_p2 + mu_t2 + C1) * (sig_p + sig_t + C2))
    return s.mean()


def ms_ssim(pred, target, weights=(0.0448, 0.2856, 0.3001, 0.2363, 0.1333)) -> float:
    p, t = _to_bchw(pred), _to_bchw(target)
    vals = []
    for i, _ in enumerate(weights):
        vals.append(ssim(p.squeeze(0).permute(1, 2, 0), t.squeeze(0).permute(1, 2, 0)))
        if i < len(weights) - 1:
            p = F.avg_pool2d(p, 2)
            t = F.avg_pool2d(t, 2)
    ms = torch.stack([v.clamp_min(1e-6) ** w for v, w in zip(vals, weights)]).prod()
    return float(ms)


class LPIPS:
    """Lazy optional wrapper. Returns None if the `lpips` package is unavailable."""
    _net = None

    @classmethod
    def distance(cls, pred, target):
        try:
            import lpips  # type: ignore
        except Exception:
            return None
        if cls._net is None:
            cls._net = lpips.LPIPS(net="alex")
        p = _to_bchw(pred) * 2 - 1
        t = _to_bchw(target) * 2 - 1
        return float(cls._net(p, t).mean())
