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
