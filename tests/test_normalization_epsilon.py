# ruff: noqa: E402
import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from structsplat import codec
from structsplat.config import FitConfig
from structsplat.fit import (
    _normalized_color_basis_apply,
    _normalized_color_basis_transpose,
    _normalized_color_denominator,
    _render,
    fit,
)
from structsplat.gaussians import GaussianField
from structsplat.render import render, render_field


def _weak_support_field(device="cpu", *, trainable=False):
    field = GaussianField.from_numpy(
        means=np.asarray([[0.0, 0.0]], dtype=np.float32),
        scales=np.asarray([[1.0, 1.0]], dtype=np.float32),
        angles=np.zeros(1, dtype=np.float32),
        colors=np.asarray([[0.8, 0.4, 0.2]], dtype=np.float32),
        device=device,
    )
    return field.trainable() if trainable else field


@pytest.mark.parametrize("value", [0.0, -1e-12, float("inf"), float("nan")])
def test_fit_config_rejects_nonpositive_or_nonfinite_normalization_eps(value):
    with pytest.raises(ValueError, match="normalization_eps must be finite and > 0"):
        FitConfig(normalization_eps=value)


def test_weak_support_matches_analytic_epsilon_attenuation_and_default():
    field = _weak_support_field()
    common = (field.means, field.conics(), field.colors, field.radii(8.0), 1, 8)

    implicit = render(*common, sigma_cutoff=8.0)
    explicit = render(*common, sigma_cutoff=8.0, normalization_eps=1e-8)
    repaired = render(*common, sigma_cutoff=8.0, normalization_eps=1e-12)

    weight = math.exp(-0.5 * 7.0**2)
    expected_default = field.colors[0] * weight / (weight + 1e-8)
    expected_repaired = field.colors[0] * weight / (weight + 1e-12)
    assert torch.equal(implicit, explicit)
    assert torch.allclose(explicit[0, 7], expected_default, atol=1e-7, rtol=1e-6)
    assert torch.allclose(repaired[0, 7], expected_repaired, atol=1e-6, rtol=1e-6)
    assert repaired[0, 7, 0] > explicit[0, 7, 0] + 0.7


def test_zero_denominator_stays_finite_black_and_invalid_direct_epsilon_fails():
    means = torch.zeros(0, 2)
    conics = torch.zeros(0, 3)
    colors = torch.zeros(0, 3)
    radii = torch.zeros(0, 2, dtype=torch.long)
    image = render(
        means, conics, colors, radii, 3, 4, normalization_eps=1e-12
    )
    assert torch.isfinite(image).all()
    assert torch.count_nonzero(image) == 0
    with pytest.raises(ValueError, match="normalization_eps must be finite and > 0"):
        render(means, conics, colors, radii, 3, 4, normalization_eps=0.0)


def test_additive_render_is_independent_of_normalization_eps():
    field = _weak_support_field()
    common = (
        field.means,
        field.conics(),
        field.colors,
        field.radii(8.0),
        1,
        8,
    )
    default = render_field(*common, mode="additive", sigma_cutoff=8.0)
    custom = render_field(
        *common, mode="additive", sigma_cutoff=8.0, normalization_eps=1e-12
    )
    assert torch.equal(custom, default)


def test_fit_render_and_fixed_geometry_operators_share_custom_epsilon():
    field = _weak_support_field()
    cfg = FitConfig(
        iters=1,
        renderer="normalized",
        sigma_cutoff=8.0,
        render_chunk=1,
        normalization_eps=1e-12,
        ssim_weight=0.0,
    )
    denominator = _normalized_color_denominator(field, cfg, 1, 8)
    applied = _normalized_color_basis_apply(
        field, field.colors, cfg, 1, 8, denominator
    )
    rendered = _render(field, cfg, 1, 8)
    assert torch.allclose(applied, rendered, atol=1e-7, rtol=1e-6)

    probe = torch.linspace(0.1, 0.8, 8).reshape(1, 8, 1).expand(-1, -1, 3)
    transposed = _normalized_color_basis_transpose(
        field, probe, cfg, 1, 8, denominator
    )
    lhs = (applied * probe).sum()
    rhs = (field.colors * transposed).sum()
    assert torch.allclose(lhs, rhs, atol=1e-6, rtol=1e-6)

    fit_target = _render(field, cfg, 12, 12).detach()
    result = fit(field.detached(), fit_target, cfg, verbose=False)
    assert result["normalization_eps"] == 1e-12


def test_codec_persists_nondefault_epsilon_and_legacy_default_is_implicit():
    field = _weak_support_field()
    custom_cfg = FitConfig(sigma_cutoff=8.0, normalization_eps=1e-12)
    custom_blob = codec.encode(field, 1, 8, codec.CodecConfig(), custom_cfg)
    custom_header = codec.blob_header(custom_blob)
    assert custom_header["normalization_eps"] == 1e-12

    decoded = codec.decode(custom_blob)
    manual = render_field(
        decoded.means,
        decoded.conics(),
        decoded.colors,
        decoded.radii(8.0),
        1,
        8,
        normalization_eps=1e-12,
        sigma_cutoff=8.0,
    )
    assert torch.allclose(codec.decode_and_render(custom_blob), manual, atol=1e-7, rtol=1e-6)

    default_blob = codec.encode(
        field, 1, 8, codec.CodecConfig(), FitConfig(sigma_cutoff=8.0)
    )
    assert "normalization_eps" not in codec.blob_header(default_blob)


@pytest.mark.parametrize("mode", ["cuda", "cuda_tiled"])
def test_cuda_custom_epsilon_forward_and_gradient_parity_when_available(mode):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    rng = np.random.default_rng(17017)
    height, width, count = 12, 14, 5
    means = np.stack(
        [rng.uniform(1.0, width - 2.0, count), rng.uniform(1.0, height - 2.0, count)],
        axis=1,
    )
    scales = rng.uniform(0.8, 1.2, (count, 2))
    angles = rng.uniform(0.0, np.pi, count)
    colors = rng.uniform(0.1, 0.9, (count, 3))
    target = torch.rand(
        height, width, 3, device="cuda", generator=torch.Generator(device="cuda").manual_seed(17)
    )

    def run(renderer):
        field = GaussianField.from_numpy(
            means, scales, angles, colors, device="cuda"
        ).trainable()
        image = render_field(
            field.means,
            field.conics(),
            field.colors,
            field.radii(8.0),
            height,
            width,
            mode=renderer,
            sigma_cutoff=8.0,
            normalization_eps=1e-12,
        )
        (image - target).square().mean().backward()
        torch.cuda.synchronize()
        return (
            image.detach(),
            field.means.grad.detach(),
            field.log_scales.grad.detach(),
            field.rotations.grad.detach(),
            field.colors.grad.detach(),
        )

    try:
        reference = run("normalized")
        candidate = run(mode)
    except RuntimeError as exc:
        pytest.skip(str(exc))
    for got, expected in zip(candidate, reference, strict=True):
        assert torch.allclose(got, expected, atol=4e-4, rtol=4e-4)
