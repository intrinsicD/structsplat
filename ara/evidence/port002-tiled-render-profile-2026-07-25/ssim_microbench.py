"""Isolated SSIM loss-term profile backing the PORT-002 separable-window change and FIT-027.

Measures, with CUDA events on the current stream, the forward+backward cost of the SSIM half of
StructSplat's default `0.7 L1 + 0.3 SSIM` objective, against L1 and Adam, and against two
optimizations: the separable Gaussian window (shipped with PORT-002) and additionally caching the
fixed target's statistics (prototype only; FIT-027).

Run: LD_PRELOAD=/lib/x86_64-linux-gnu/libstdc++.so.6 PYTHONPATH=src:. \
       python ara/evidence/port002-tiled-render-profile-2026-07-25/ssim_microbench.py
"""

from __future__ import annotations

import json

import torch
import torch.nn.functional as F

from structsplat import metrics as M

C1, C2 = 0.01 ** 2, 0.03 ** 2


_DENSE_CACHE: dict = {}


def dense_window(win, sigma, device, dtype):
    """The pre-PORT-002 dense 11x11 outer-product window.

    Cached exactly as the shipped `_gaussian_window` was before this change. Rebuilding it per
    call would charge the baseline for work the real baseline never did and inflate the measured
    speedup, so the cache is part of the fair comparison, not an optimization of it.
    """
    key = (int(win), float(sigma), str(device), str(dtype))
    if key not in _DENSE_CACHE:
        x = torch.arange(win, device=device, dtype=dtype) - (win - 1) / 2.0
        g = torch.exp(-(x ** 2) / (2 * sigma ** 2))
        g = (g / g.sum()).unsqueeze(0)
        _DENSE_CACHE[key] = (g.t() @ g).expand(3, 1, win, win).contiguous()
    return _DENSE_CACHE[key]


def dense_ssim(p, t, win=11, sigma=1.5):
    w, pad = dense_window(win, sigma, p.device, p.dtype), win // 2
    mu_p = F.conv2d(p, w, padding=pad, groups=3)
    mu_t = F.conv2d(t, w, padding=pad, groups=3)
    mu_p2, mu_t2, mu_pt = mu_p * mu_p, mu_t * mu_t, mu_p * mu_t
    sig_p = F.conv2d(p * p, w, padding=pad, groups=3) - mu_p2
    sig_t = F.conv2d(t * t, w, padding=pad, groups=3) - mu_t2
    sig_pt = F.conv2d(p * t, w, padding=pad, groups=3) - mu_pt
    return (((2 * mu_pt + C1) * (2 * sig_pt + C2))
            / ((mu_p2 + mu_t2 + C1) * (sig_p + sig_t + C2))).mean()


class CachedTargetSSIM:
    """FIT-027 prototype: separable blur plus target statistics cached across iterations."""

    def __init__(self, t, win=11, sigma=1.5):
        self.wh, self.wv = M._gaussian_window(win, sigma, t.device, t.dtype)
        self.pad = win // 2
        with torch.no_grad():
            self.t = t
            self.mu_t = self._blur(t)
            self.mu_t2 = self.mu_t * self.mu_t
            self.sig_t = self._blur(t * t) - self.mu_t2

    def _blur(self, x):
        x = F.conv2d(x, self.wh, padding=(0, self.pad), groups=3)
        return F.conv2d(x, self.wv, padding=(self.pad, 0), groups=3)

    def __call__(self, p):
        mu_p = self._blur(p)
        mu_p2, mu_pt = mu_p * mu_p, mu_p * self.mu_t
        sig_p = self._blur(p * p) - mu_p2
        sig_pt = self._blur(p * self.t) - mu_pt
        return (((2 * mu_pt + C1) * (2 * sig_pt + C2))
                / ((mu_p2 + self.mu_t2 + C1) * (sig_p + self.sig_t + C2))).mean()


def median_ms(fn, warmup=10, repeats=25, inner=1):
    """Median per-call time, batching `inner` calls into one timed region.

    At small image sizes a single SSIM forward+backward is well under a millisecond and the
    measurement is dominated by launch and timer overhead — repeated single-call timings on this
    machine spread by more than 50%, enough to invent or hide an effect. Batching pushes the timed
    region above that floor. The reported spread makes the remaining noise visible rather than
    hiding it behind a median.
    """
    for _ in range(warmup):
        for _ in range(inner):
            fn()
    samples = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(inner):
            fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) / inner)
    samples.sort()
    return samples[len(samples) // 2], samples[0], samples[-1]


def main() -> int:
    if not torch.cuda.is_available():
        raise SystemExit("ssim_microbench requires a CUDA device")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    out = {"device": torch.cuda.get_device_name(0), "torch": torch.__version__, "cells": []}

    # Ramp the device clocks before the first measured cell. Without this the first cell is timed
    # against a partly idle GPU and reads slow, which biases whichever comparison happens to run
    # first rather than any particular arm.
    ramp = torch.rand(1024, 1024, 3, device=device)
    for _ in range(200):
        ramp = ramp * 1.0001 + 1e-6
    torch.cuda.synchronize()
    del ramp

    for size in (128, 192, 256, 384, 512, 1024):
        gen = torch.Generator(device=device).manual_seed(0)
        pred = torch.rand(size, size, 3, device=device, generator=gen).requires_grad_(True)
        target = torch.rand(size, size, 3, device=device, generator=gen)
        p, t = M._to_bchw(pred), M._to_bchw(target)
        cached = CachedTargetSSIM(t)

        def grad_of(loss_fn):
            def run():
                torch.autograd.grad(1.0 - loss_fn(M._to_bchw(pred), t), pred)
            return run

        inner = max(20, int(2_000_000 / (size * size)))
        dense_fb, d_lo, d_hi = median_ms(grad_of(lambda a, b: dense_ssim(a, b)), inner=inner)
        sep_fb, s_lo, s_hi = median_ms(
            grad_of(lambda a, b: M._ssim_builtin_bchw(a, b, 11, 1.5)), inner=inner)
        cached_fb, _, _ = median_ms(grad_of(lambda a, _b: cached(a)), inner=inner)
        l1_fb, _, _ = median_ms(
            lambda: torch.autograd.grad((pred - target).abs().mean(), pred), inner=inner)

        (g_dense,) = torch.autograd.grad(1.0 - dense_ssim(p, t), pred, retain_graph=True)
        (g_sep,) = torch.autograd.grad(1.0 - M._ssim_builtin_bchw(p, t, 11, 1.5), pred,
                                       retain_graph=True)
        (g_cached,) = torch.autograd.grad(1.0 - cached(p), pred)
        scale = float(g_dense.abs().max())

        cell = {
            "size": size,
            "dense_fwd_bwd_ms": dense_fb,
            "separable_fwd_bwd_ms": sep_fb,
            "cached_target_fwd_bwd_ms": cached_fb,
            "l1_fwd_bwd_ms": l1_fb,
            "inner_calls_per_timed_region": inner,
            "dense_spread_ms": [d_lo, d_hi],
            "separable_spread_ms": [s_lo, s_hi],
            "dense_spread_pct_of_median": (d_hi - d_lo) / dense_fb * 100.0,
            "separable_spread_pct_of_median": (s_hi - s_lo) / sep_fb * 100.0,
            "separable_speedup": dense_fb / sep_fb,
            "cached_target_speedup": dense_fb / cached_fb,
            "value_absdiff_separable": abs(float(dense_ssim(p, t).detach())
                                           - float(M._ssim_builtin_bchw(p, t, 11, 1.5).detach())),
            "grad_relmax_separable": float((g_sep - g_dense).abs().max()) / scale,
            "grad_relmax_cached_target": float((g_cached - g_dense).abs().max()) / scale,
        }
        out["cells"].append(cell)
        print(f"{size}^2: dense {dense_fb:.3f} ms | separable {sep_fb:.3f} ms "
              f"({cell['separable_speedup']:.2f}x) | +cached target {cached_fb:.3f} ms "
              f"({cell['cached_target_speedup']:.2f}x) | L1 {l1_fb:.3f} ms")
        print(f"       spread: dense {cell['dense_spread_pct_of_median']:.0f}%, "
              f"separable {cell['separable_spread_pct_of_median']:.0f}% "
              f"| grad relmax sep {cell['grad_relmax_separable']:.2e}")

    for n in (2048, 8192):
        params = [torch.randn(n, k, device=device, requires_grad=True) for k in (2, 2, 1, 3, 1)]
        for prm in params:
            prm.grad = torch.randn_like(prm)
        opt = torch.optim.Adam(params, lr=1e-2)

        def adam_step():
            opt.step()
            opt.zero_grad(set_to_none=False)

        ms, _, _ = median_ms(adam_step, inner=20)
        out.setdefault("adam_ms", {})[str(n)] = ms
        print(f"Adam step N={n}: {ms:.3f} ms")

    path = __file__.replace("ssim_microbench.py", "ssim_microbench.json")
    with open(path, "w") as handle:
        json.dump(out, handle, indent=2)
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
