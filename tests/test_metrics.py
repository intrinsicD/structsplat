# ruff: noqa: E402
import numpy as np
import pytest

torch = pytest.importorskip("torch")
from structsplat import metrics as M


def test_ms_ssim_accepts_bchw_and_hwc_batch():
    # documented contract is HWC or BCHW; the old squeeze(0).permute crashed on a real batch
    rng = np.random.default_rng(0)
    a = torch.as_tensor(rng.random((32, 32, 3)).astype(np.float32))
    b = torch.as_tensor(rng.random((32, 32, 3)).astype(np.float32))
    hwc = M.ms_ssim(a, b)
    bchw = M.ms_ssim(a.permute(2, 0, 1)[None], b.permute(2, 0, 1)[None])
    assert np.isfinite(hwc) and np.isfinite(bchw)
    assert abs(hwc - bchw) < 1e-5           # same content, both layouts agree


def test_ms_ssim_identical_images_is_one():
    a = torch.as_tensor(np.random.default_rng(1).random((40, 40, 3)).astype(np.float32))
    assert abs(M.ms_ssim(a, a) - 1.0) < 1e-4


def test_ms_ssim_drops_scales_below_window():
    # at 48x48 the 6x6 and 3x3 scales are smaller than the 11x11 window and must be dropped
    # (they were dominated by zero padding); the result stays a valid, finite [0,1] score.
    a = torch.as_tensor(np.random.default_rng(2).random((48, 48, 3)).astype(np.float32))
    b = torch.as_tensor(np.random.default_rng(3).random((48, 48, 3)).astype(np.float32))
    v = M.ms_ssim(a, b, win=11)
    assert 0.0 <= v <= 1.0 and np.isfinite(v)


def test_ms_ssim_raises_below_one_window():
    a = torch.zeros(8, 8, 3)
    with pytest.raises(ValueError, match="smaller than the SSIM window"):
        M.ms_ssim(a, a, win=11)


def test_ssim_window_is_cached_by_shape_device_dtype():
    M.clear_ssim_window_cache()
    a = torch.as_tensor(np.random.default_rng(4).random((24, 24, 3)).astype(np.float32))
    b = torch.as_tensor(np.random.default_rng(5).random((24, 24, 3)).astype(np.float32))

    M.ssim(a, b)
    assert M.ssim_window_cache_info()["size"] == 1
    M.ssim(a, b)
    assert M.ssim_window_cache_info()["size"] == 1
    M.ssim(a.double(), b.double())
    assert M.ssim_window_cache_info()["size"] == 2


def test_fused_ssim_backend_falls_back_or_matches_builtin():
    a = torch.as_tensor(np.random.default_rng(6).random((32, 32, 3)).astype(np.float32))
    b = torch.as_tensor(np.random.default_rng(7).random((32, 32, 3)).astype(np.float32))
    assert torch.allclose(M.ssim(a, b, backend="fused"), M.ssim(a, b), atol=1e-6)

    if torch.cuda.is_available() and M.fused_ssim_available("cuda"):
        ac = a.permute(2, 0, 1).unsqueeze(0).cuda()
        bc = b.permute(2, 0, 1).unsqueeze(0).cuda()
        fused = M.ssim(ac, bc, backend="fused")
        builtin = M.ssim(ac, bc, backend="builtin")
        torch.cuda.synchronize()
        assert torch.allclose(fused, builtin, atol=2e-5, rtol=2e-5)


def test_ssim_separable_window_matches_dense_outer_product_reference():
    """The separable blur must reproduce the dense 2D window it replaced (value and gradient).

    Guards the PORT-002 follow-on optimization in `_gaussian_window`: a Gaussian window is an
    outer product, so two 1D passes and one 2D pass are the same operator up to float
    associativity. Any future change that breaks separability shows up here.
    """
    rng = np.random.default_rng(11)
    pred = torch.as_tensor(rng.random((1, 3, 40, 44)).astype(np.float32)).requires_grad_(True)
    target = torch.as_tensor(rng.random((1, 3, 40, 44)).astype(np.float32))
    win, sigma, pad = 11, 1.5, 5

    x = torch.arange(win, dtype=torch.float32) - (win - 1) / 2.0
    g = torch.exp(-(x ** 2) / (2 * sigma ** 2))
    g = (g / g.sum()).unsqueeze(0)
    dense = (g.t() @ g).expand(3, 1, win, win).contiguous()

    def dense_ssim(p, t):
        c1, c2 = 0.01 ** 2, 0.03 ** 2
        mu_p = torch.nn.functional.conv2d(p, dense, padding=pad, groups=3)
        mu_t = torch.nn.functional.conv2d(t, dense, padding=pad, groups=3)
        mu_p2, mu_t2, mu_pt = mu_p * mu_p, mu_t * mu_t, mu_p * mu_t
        sig_p = torch.nn.functional.conv2d(p * p, dense, padding=pad, groups=3) - mu_p2
        sig_t = torch.nn.functional.conv2d(t * t, dense, padding=pad, groups=3) - mu_t2
        sig_pt = torch.nn.functional.conv2d(p * t, dense, padding=pad, groups=3) - mu_pt
        return (((2 * mu_pt + c1) * (2 * sig_pt + c2))
                / ((mu_p2 + mu_t2 + c1) * (sig_p + sig_t + c2))).mean()

    want = dense_ssim(pred, target)
    got = M._ssim_builtin_bchw(pred, target, win, sigma)
    assert torch.allclose(got, want, atol=1e-6, rtol=1e-6)

    (gw,) = torch.autograd.grad(want, pred, retain_graph=True)
    (gg,) = torch.autograd.grad(got, pred)
    assert torch.allclose(gg, gw, atol=1e-7, rtol=1e-5)


def test_ssim_window_cache_returns_separable_pair():
    M.clear_ssim_window_cache()
    a = torch.as_tensor(np.random.default_rng(12).random((24, 24, 3)).astype(np.float32))
    M.ssim(a, a)
    horizontal, vertical = M._gaussian_window(11, 1.5, torch.device("cpu"), torch.float32)
    assert horizontal.shape == (3, 1, 1, 11)
    assert vertical.shape == (3, 1, 11, 1)
    assert M.ssim_window_cache_info()["size"] == 1


class TestSSIMTargetStats:
    """FIT-027: caching the target half of SSIM must change cost, never value."""

    @staticmethod
    def _pair(seed: int = 0, h: int = 48, w: int = 64):
        g = torch.Generator().manual_seed(seed)
        pred = torch.rand(h, w, 3, generator=g)
        target = torch.rand(h, w, 3, generator=g)
        return pred, target

    def test_cached_value_matches_the_stateless_form(self):
        pred, target = self._pair()
        stats = M.SSIMTargetStats(target)
        plain = M.ssim(pred, target)
        cached = M.ssim(pred, target, target_stats=stats)
        assert torch.allclose(plain, cached, atol=1e-12, rtol=0)

    def test_cached_gradient_matches_the_stateless_form(self):
        pred, target = self._pair(seed=3)
        a = pred.clone().requires_grad_(True)
        b = pred.clone().requires_grad_(True)
        M.ssim(a, target).backward()
        M.ssim(b, target, target_stats=M.SSIMTargetStats(target)).backward()
        assert torch.allclose(a.grad, b.grad, atol=1e-12, rtol=0)

    def test_cache_holds_no_graph_and_does_not_backprop_into_the_target(self):
        pred, target = self._pair(seed=5)
        target = target.requires_grad_(True)
        stats = M.SSIMTargetStats(target)
        assert not stats.mu_t.requires_grad
        assert not stats.sig_t.requires_grad

    def test_a_different_target_is_rejected_not_silently_reused(self):
        pred, target = self._pair(seed=7)
        other = torch.rand_like(target)
        stats = M.SSIMTargetStats(target)
        assert not stats.matches(other, 11, 1.5)
        # The wrong cache must be ignored, so the value still tracks the target passed in.
        assert torch.allclose(
            M.ssim(pred, other, target_stats=stats), M.ssim(pred, other),
            atol=1e-12, rtol=0,
        )

    def test_in_place_mutation_of_the_target_invalidates_the_cache(self):
        """The curriculum and pyramid paths rewrite targets; stale stats must not survive."""
        pred, target = self._pair(seed=11)
        stats = M.SSIMTargetStats(target)
        assert stats.matches(target, 11, 1.5)
        with torch.no_grad():
            target.mul_(0.5)
        assert not stats.matches(target, 11, 1.5)
        assert torch.allclose(
            M.ssim(pred, target, target_stats=stats), M.ssim(pred, target),
            atol=1e-12, rtol=0,
        )

    def test_window_parameters_are_part_of_the_cache_key(self):
        pred, target = self._pair(seed=13)
        stats = M.SSIMTargetStats(target, win=11, sigma=1.5)
        assert not stats.matches(target, 7, 1.5)
        assert not stats.matches(target, 11, 2.0)
        assert torch.allclose(
            M.ssim(pred, target, win=7, target_stats=stats), M.ssim(pred, target, win=7),
            atol=1e-12, rtol=0,
        )
