# ruff: noqa: E402
import numpy as np
import pytest

torch = pytest.importorskip("torch")
from structsplat import codec
from structsplat.config import FitConfig, InitConfig
from structsplat import init as _init
from structsplat.render import render


def _toy(H=48, W=48):
    img = np.zeros((H, W, 3), np.float32)
    img[:, W // 2:, :] = 1.0
    img[10:20, 10:20] = [1.0, 0.5, 0.0]
    return img


def _fitted_field(img, n=128):
    field = _init.build_field(img, InitConfig(strategy="aniso_flanking", num_gaussians=n, seed=0))
    return field


def _render_of(field, target, fcfg):
    return render(field.means, field.conics(), field.colors,
                  field.radii(fcfg.sigma_cutoff), *target.shape[:2], fcfg.render_chunk)


def test_roundtrip_precision_and_bpp():
    img = _toy()
    target = torch.as_tensor(img)
    field = _fitted_field(img)
    fcfg = FitConfig()
    r = codec.rd_point(field, target, fcfg, codec.CodecConfig())
    assert r["bytes"] > 0 and r["bpp"] <= r["raw_bpp"] * 1.2  # zlib should not inflate much
    # 16-bit means / 8-bit attributes: decoded render stays close to the uncompressed one
    blob = codec.encode(field, *img.shape[:2], codec.CodecConfig())
    dec = codec.decode(blob)
    d = (_render_of(field, target, fcfg) - _render_of(dec, target, fcfg)).abs()
    assert float(d.max()) < 0.15 and float(d.mean()) < 0.01
    # means quantization error bounded by one lattice step (no reorder: rows must align)
    plain = codec.decode(codec.encode(field, *img.shape[:2],
                                      codec.CodecConfig(morton_reorder=False)))
    err = (plain.means - field.means).abs().max()
    assert float(err) <= (img.shape[1] - 1) / (2 ** 16 - 1) + 1e-4


def test_morton_reorder_changes_order_not_content():
    img = _toy()
    target = torch.as_tensor(img)
    field = _fitted_field(img)
    fcfg = FitConfig()
    a = codec.decode(codec.encode(field, 48, 48, codec.CodecConfig(morton_reorder=True)))
    b = codec.decode(codec.encode(field, 48, 48, codec.CodecConfig(morton_reorder=False)))
    da = _render_of(a, target, fcfg)
    db = _render_of(b, target, fcfg)
    assert float((da - db).abs().max()) < 1e-2  # order-independent renderer


def test_rotation_period_invariance():
    img = _toy()
    field = _fitted_field(img)
    with torch.no_grad():
        field.rotations += np.pi  # same Gaussians, shifted parameterization
    blob = codec.encode(field, 48, 48, codec.CodecConfig())
    dec = codec.decode(blob)
    assert torch.isfinite(dec.rotations).all()
    assert float(dec.rotations.max()) <= np.pi + 1e-5


def test_qat_improves_coarse_quantization():
    img = _toy()
    target = torch.as_tensor(img)
    field = _fitted_field(img)
    fcfg = FitConfig()
    coarse = codec.CodecConfig(bits_means=10, bits_scales=4, bits_rot=4, bits_colors=4)
    before = codec.rd_point(field, target, fcfg, coarse)["psnr"]
    ccfg = codec.qat_finetune(field, target, fcfg, coarse, iters=60)
    after = codec.rd_point(field, target, fcfg, ccfg)["psnr"]
    assert after > before
