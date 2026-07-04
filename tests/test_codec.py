# ruff: noqa: E402
import numpy as np
import pytest

torch = pytest.importorskip("torch")
from structsplat import codec
from structsplat.config import FitConfig, InitConfig
from structsplat import init as _init
from structsplat.render import render, render_field


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


def test_opacity_roundtrip_and_render_parity():
    img = _toy()
    target = torch.as_tensor(img)
    field = _init.build_field(img, InitConfig(strategy="aniso_flanking", num_gaussians=96,
                                              opacity_mode="constant", init_opacity=0.7, seed=0))
    fcfg = FitConfig()
    blob = codec.encode(field, 48, 48, codec.CodecConfig())
    dec = codec.decode(blob)
    assert dec.opacities is not None
    p_orig = field.opacity_values()
    p_dec = dec.opacity_values()
    assert float((p_orig.sort().values - p_dec.sort().values).abs().max()) < 1.0 / 255 + 1e-3
    # decoded render (with opacities applied) must track the original opacity render
    a = render(field.means, field.conics(), field.colors, field.radii(3.0), 48, 48,
               opacities=field.opacity_values())
    b = render(dec.means, dec.conics(), dec.colors, dec.radii(3.0), 48, 48,
               opacities=dec.opacity_values())
    assert float((a - b).abs().mean()) < 0.02
    # and rd_point must run the opacity path end to end
    r = codec.rd_point(field, target, fcfg, codec.CodecConfig())
    assert np.isfinite(r["psnr"]) and r["raw_bpp"] > 0


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


def test_off_image_means_bounded_by_lattice_step():
    # fit() never clamps means: a Gaussian can sit off-image. Quantizing over the padded extent
    # stored in the header bounds the round-trip error by one lattice step (COMP-002).
    img = _toy()
    field = _fitted_field(img, n=64)
    with torch.no_grad():
        field.means[0] = torch.tensor([-8.0, 53.0])   # off the left / bottom edges
        field.means[1] = torch.tensor([60.0, -5.0])
    ccfg = codec.CodecConfig(morton_reorder=False)
    blob = codec.encode(field, *img.shape[:2], ccfg)
    dec = codec.decode(blob)
    h = codec.blob_header(blob)
    span_x = h["means_hi"][0] - h["means_lo"][0]
    span_y = h["means_hi"][1] - h["means_lo"][1]
    step = max(span_x, span_y) / (2 ** ccfg.bits_means - 1)
    err = (dec.means - field.means).abs().max().item()
    assert err <= step + 1e-4
    assert h["means_lo"][0] <= -8.0 and h["means_hi"][0] >= 60.0


def test_blob_is_self_describing_render():
    # decode_and_render must reproduce the fitted renderer's output with no out-of-band FitConfig
    img = _toy()
    field = _fitted_field(img)
    fcfg = FitConfig(renderer="additive", aa_dilation=0.3, sigma_cutoff=2.5,
                     support_fade=True)
    blob = codec.encode(field, *img.shape[:2], codec.CodecConfig(), fcfg)
    h = codec.blob_header(blob)
    assert h["renderer"] == "additive" and h["aa_dilation"] == 0.3 and h["sigma_cutoff"] == 2.5
    assert h["support_fade"] is True
    auto = codec.decode_and_render(blob)                       # semantics from the header
    dec = codec.decode(blob)
    manual = render_field(dec.means, dec.conics(0.3), dec.colors,
                          dec.radii(2.5, 0.3), *img.shape[:2], mode="additive",
                          support_fade=True, sigma_cutoff=2.5)
    assert torch.allclose(auto, manual, atol=1e-5)


def test_additive_field_roundtrips_through_qat_and_rd():
    # an additive-fit field must fine-tune and RD-score through additive compositing (COMP-002)
    img = _toy()
    target = torch.as_tensor(img)
    field = _init.build_field(img, InitConfig(strategy="aniso_flanking", num_gaussians=96,
                                              opacity_mode="constant", init_opacity=0.7, seed=0))
    fcfg = FitConfig(renderer="additive")
    coarse = codec.CodecConfig(bits_means=12, bits_scales=6, bits_rot=6, bits_colors=6)
    before = codec.rd_point(field, target, fcfg, coarse)["psnr"]
    ccfg = codec.qat_finetune(field, target, fcfg, coarse, iters=40)
    after = codec.rd_point(field, target, fcfg, ccfg)
    assert np.isfinite(after["psnr"]) and after["psnr"] > before - 1.0


def test_rd_point_clamps_render_before_metrics():
    # a field that overshoots [0,1] must be scored on the clamped render (display-referred)
    img = _toy()
    target = torch.as_tensor(img)
    field = _fitted_field(img)
    with torch.no_grad():
        field.colors += 3.0        # drive the render well above 1.0
    r = codec.rd_point(field, target, FitConfig(), codec.CodecConfig())
    img_clamped = codec.decode_and_render(codec.encode(field, *img.shape[:2],
                                          codec.CodecConfig(), FitConfig())).clamp(0, 1)
    from structsplat import metrics as M
    assert abs(r["psnr"] - M.psnr(img_clamped, target)) < 1e-3
