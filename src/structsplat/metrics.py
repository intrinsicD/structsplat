"""Reconstruction metrics: PSNR and SSIM/MS-SSIM built-in (torch, no external deps).

LPIPS is optional (pip install lpips); import is guarded so the repo runs without it.
Images are (H,W,3) or (B,3,H,W) in [0,1]. BENCH-001 fixes the exact protocol.

Protocol notes (deliberate deviations from the reference implementations — consistent across
every cell of a comparison, but do not quote these numbers against papers without checking):
  * ssim() uses zero-padded convolutions and averages over all pixels, which biases border
    windows relative to the valid-window formulation (Wang et al. use valid windows).
  * ms_ssim() applies the full SSIM (including luminance) at every scale instead of
    contrast*structure at the fine scales; negatives are clamped before the power. It drops any
    scale whose smaller side is below the SSIM window (those scales are dominated by zero
    padding) and renormalizes the remaining weights. For paper-comparable numbers install
    pytorch-msssim (the `[metrics]` extra) and report that.
  * Clamping convention (COMP-002): metrics here operate on whatever tensor they are given.
    Fit-time / training metrics are computed on the *raw* (possibly out-of-[0,1]) render so the
    reported number tracks the loss the optimizer sees. Display-referred rate-distortion metrics
    (codec.rd_point) clamp the render to [0,1] first, so bpp-vs-quality numbers are comparable to
    display-referred baselines (JPEG / GaussianImage) that cannot represent out-of-range values.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F

_WINDOW_CACHE: dict[tuple[int, float, str, int, str], torch.Tensor] = {}
_FUSED_SSIM = None
_FUSED_SSIM_ATTEMPTED = False


def _to_bchw(x):
    if x.dim() == 3:
        x = x.permute(2, 0, 1).unsqueeze(0)
    return x


def mse(pred, target) -> torch.Tensor:
    return F.mse_loss(pred, target)


def psnr_from_mse(mse_value: torch.Tensor, max_val: float = 1.0) -> torch.Tensor:
    mse_value = mse_value.clamp_min(1e-12)
    return 10.0 * torch.log10((max_val * max_val) / mse_value)


def psnr_tensor(pred, target, max_val: float = 1.0) -> torch.Tensor:
    return psnr_from_mse(mse(pred, target), max_val)


def psnr(pred, target, max_val: float = 1.0) -> float:
    return float(psnr_tensor(pred, target, max_val))


def psnr_mse_threshold(target_psnr: float, max_val: float = 1.0) -> float:
    return (max_val * max_val) * (10.0 ** (-float(target_psnr) / 10.0))


def clear_ssim_window_cache() -> None:
    _WINDOW_CACHE.clear()


def ssim_window_cache_info() -> dict:
    return {"size": len(_WINDOW_CACHE), "keys": list(_WINDOW_CACHE.keys())}


def _gaussian_window(win: int, sigma: float, device, dtype):
    """Cached separable Gaussian window as a (horizontal, vertical) 1D convolution pair.

    The window is a separable outer product, so the 11x11 blur it used to build costs 121
    multiply-adds per output pixel where two 1x11 passes cost 22. Both forms are the same
    operator up to float associativity (measured agreement ~1e-8 on the SSIM value and ~2e-6
    relative on its gradient), and one cache entry per (win, sigma, device, dtype) key is
    retained so `ssim_window_cache_info()` semantics are unchanged.
    """
    dev = torch.device(device)
    key = (int(win), float(sigma), dev.type, -1 if dev.index is None else int(dev.index), str(dtype))
    cached = _WINDOW_CACHE.get(key)
    if cached is not None:
        return cached
    x = torch.arange(win, device=device, dtype=dtype) - (win - 1) / 2.0
    g = torch.exp(-(x ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    horizontal = g.view(1, 1, 1, win).expand(3, 1, 1, win).contiguous()
    vertical = g.view(1, 1, win, 1).expand(3, 1, win, 1).contiguous()
    _WINDOW_CACHE[key] = (horizontal, vertical)
    return _WINDOW_CACHE[key]


def _ssim_blur(x, wh, wv, pad: int):
    x = F.conv2d(x, wh, padding=(0, pad), groups=3)
    return F.conv2d(x, wv, padding=(pad, 0), groups=3)


class SSIMTargetStats:
    """Target-side SSIM statistics, computed once and reused across a fit (FIT-027).

    Within one fit the target ``t`` is constant, so ``blur(t)``, ``mu_t2`` and ``sig_t`` are
    recomputed every iteration to produce the identical result and require no gradient. This
    object holds them under **explicit ownership**: the fitter constructs one, hands it to
    :func:`ssim`, and drops it whenever it swaps the target. ``metrics.ssim()`` stays stateless
    when no cache is passed, so reporting and ablation values remain comparable to prior runs.

    :meth:`matches` is the guard against silently serving stale statistics. It checks object
    identity *and* the autograd version counter, so a target mutated in place — not just
    replaced — invalidates the cache rather than poisoning the loss.
    """

    __slots__ = ("mu_t", "mu_t2", "sig_t", "_src", "_src_version", "_key")

    def __init__(self, target, win: int = 11, sigma: float = 1.5):
        t = _to_bchw(target)
        wh, wv = _gaussian_window(win, sigma, t.device, t.dtype)
        pad = win // 2
        with torch.no_grad():
            self.mu_t = _ssim_blur(t, wh, wv, pad)
            self.mu_t2 = self.mu_t * self.mu_t
            self.sig_t = _ssim_blur(t * t, wh, wv, pad) - self.mu_t2
        self._src = target
        self._src_version = int(target._version)
        self._key = (int(win), float(sigma), tuple(t.shape), str(t.device), str(t.dtype))

    def matches(self, target, win: int, sigma: float) -> bool:
        """True only for the exact, unmutated tensor these statistics were built from."""
        if target is not self._src or int(target._version) != self._src_version:
            return False
        t = _to_bchw(target)
        return self._key == (
            int(win), float(sigma), tuple(t.shape), str(t.device), str(t.dtype)
        )


def _ssim_builtin_bchw(p, t, win: int, sigma: float, stats: "SSIMTargetStats | None" = None
                       ) -> torch.Tensor:
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    wh, wv = _gaussian_window(win, sigma, p.device, p.dtype)
    pad = win // 2

    def blur(x):
        return _ssim_blur(x, wh, wv, pad)

    mu_p = blur(p)
    if stats is None:
        mu_t = blur(t)
        mu_t2 = mu_t * mu_t
        sig_t = blur(t * t) - mu_t2
    else:
        mu_t, mu_t2, sig_t = stats.mu_t, stats.mu_t2, stats.sig_t
    mu_p2, mu_pt = mu_p * mu_p, mu_p * mu_t
    sig_p = blur(p * p) - mu_p2
    sig_pt = blur(p * t) - mu_pt
    s = ((2 * mu_pt + C1) * (2 * sig_pt + C2)) / ((mu_p2 + mu_t2 + C1) * (sig_p + sig_t + C2))
    return s.mean()


def _fused_ssim_fn(device: torch.device):
    global _FUSED_SSIM, _FUSED_SSIM_ATTEMPTED
    if device.type not in ("cuda", "mps"):
        return None
    if not _FUSED_SSIM_ATTEMPTED:
        _FUSED_SSIM_ATTEMPTED = True
        try:
            from fused_ssim import fused_ssim as _impl  # type: ignore
            _FUSED_SSIM = _impl
        except Exception:
            _FUSED_SSIM = None
    return _FUSED_SSIM


def fused_ssim_available(device: torch.device | str | None = None) -> bool:
    dev = torch.device(device) if device is not None else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    return _fused_ssim_fn(dev) is not None


def ssim(pred, target, win: int = 11, sigma: float = 1.5,
         backend: str = "builtin",
         target_stats: "SSIMTargetStats | None" = None) -> torch.Tensor:
    """SSIM. Stateless unless the caller owns and passes ``target_stats`` (FIT-027).

    A stale or mismatched cache is ignored rather than trusted, so passing one can change the
    cost of this call but never its value.
    """
    p, t = _to_bchw(pred), _to_bchw(target)
    if backend not in ("builtin", "fused", "auto"):
        raise ValueError(f"unknown ssim backend {backend!r}; expected builtin, fused, or auto")
    if backend in ("fused", "auto") and win == 11 and abs(float(sigma) - 1.5) < 1e-12:
        fn = _fused_ssim_fn(p.device)
        if fn is not None:
            try:
                return fn(p.contiguous(), t.contiguous(), padding="same",
                          train=bool(p.requires_grad), reduction="mean")
            except Exception:
                pass
    stats = target_stats if (
        target_stats is not None and target_stats.matches(target, win, sigma)
    ) else None
    return _ssim_builtin_bchw(p, t, win, sigma, stats)


def ms_ssim(pred, target, weights=(0.0448, 0.2856, 0.3001, 0.2363, 0.1333),
            win: int = 11, backend: str = "builtin") -> float:
    # Accept HWC or BCHW; ssim() consumes BCHW directly (the old squeeze(0).permute crashed on
    # a true (B,3,H,W) batch and was a no-op round-trip for B=1). COMP-002.
    p, t = _to_bchw(pred), _to_bchw(target)
    vals, used_w = [], []
    for i, w in enumerate(weights):
        if min(p.shape[-2], p.shape[-1]) < win:
            break  # a scale smaller than the SSIM window is dominated by zero padding: drop it
        vals.append(ssim(p, t, win=win, backend=backend))
        used_w.append(w)
        if i < len(weights) - 1:
            p = F.avg_pool2d(p, 2)
            t = F.avg_pool2d(t, 2)
    if not vals:
        raise ValueError(
            f"ms_ssim: image {tuple(p.shape[-2:])} smaller than the SSIM window {win}; "
            "use ssim() with a smaller window for images below one window.")
    tot = sum(used_w)
    used_w = [w / tot for w in used_w]  # renormalize over the scales actually used
    ms = torch.stack([v.clamp_min(1e-6) ** w for v, w in zip(vals, used_w)]).prod()
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
        cls._net = cls._net.to(device=p.device, dtype=p.dtype).eval()
        with torch.no_grad():
            return float(cls._net(p, t).mean())
