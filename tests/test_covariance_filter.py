# ruff: noqa: E402
import math

import pytest

torch = pytest.importorskip("torch")

from structsplat import codec
from structsplat.config import FitConfig
from structsplat.fit import (
    _add_from_residual,
    _ensure_generation_density_filter,
    _generation_density_filter_variance,
    _relocate_from_residual,
    _render,
    fit,
)
from structsplat.gaussians import GaussianField
from structsplat.render import render_field


def _field(n: int = 4) -> GaussianField:
    x = torch.linspace(2.0, 9.0, n)
    means = torch.stack([x, torch.flip(x, dims=[0])], dim=1)
    scales = torch.stack([
        torch.linspace(0.8, 1.4, n),
        torch.linspace(1.6, 0.9, n),
    ], dim=1)
    colors = torch.stack([
        torch.linspace(0.1, 0.8, n),
        torch.linspace(0.8, 0.2, n),
        torch.full((n,), 0.4),
    ], dim=1)
    return GaussianField(means, torch.log(scales), torch.linspace(-0.3, 0.4, n), colors)


def test_generation_density_formula_and_config_validation():
    cfg = FitConfig(covariance_filter_mode="generation_density")
    expected = 20 * 30 / (9.0 * math.pi * 10)
    assert _generation_density_filter_variance(cfg, 20, 30, 10) == pytest.approx(expected)

    capped = FitConfig(
        covariance_filter_mode="generation_density",
        covariance_filter_max_variance=0.25,
    )
    assert _generation_density_filter_variance(capped, 200, 300, 1) == 0.25
    with pytest.raises(ValueError, match="covariance_filter_mode"):
        FitConfig(covariance_filter_mode="elapsed_age")
    with pytest.raises(ValueError, match="covariance_filter_alpha"):
        FitConfig(covariance_filter_alpha=0.0)
    with pytest.raises(ValueError, match="covariance_filter_max_variance"):
        FitConfig(covariance_filter_max_variance=-1.0)


def test_default_mode_does_not_mutate_field_or_create_filter_state():
    field = _field()
    before = field.conics().clone()
    returned = _ensure_generation_density_filter(field, FitConfig(), 12, 12)

    assert returned is field
    assert field.filter_variance is None
    assert torch.equal(field.conics(), before)

    target = torch.zeros(12, 12, 3)
    out = fit(
        field,
        target,
        FitConfig(iters=1, log_every=1, ssim_weight=0.0),
        verbose=False,
    )
    assert out["field"].filter_variance is None
    assert out["covariance_filter_mode"] == "none"
    assert out["covariance_filter_variance_min"] is None


def test_filtered_covariance_matches_materialized_scales_and_normalized_render():
    field = _field(3)
    field.filter_variance = torch.tensor([0.2, 1.3, 2.7])
    materialized = field.materialize_covariance_filter()

    assert materialized.filter_variance is None
    assert torch.allclose(materialized.scales(), field.effective_scales(), atol=1e-7)
    assert torch.allclose(materialized.conics(), field.conics(), atol=1e-6)
    assert torch.equal(materialized.radii(3.0), field.radii(3.0))

    args = (12, 12, 64, "normalized")
    filtered_image = render_field(
        field.means, field.conics(), field.colors, field.radii(3.0), *args,
        scales=field.effective_scales(), rotations=field.rotations,
    )
    materialized_image = render_field(
        materialized.means, materialized.conics(), materialized.colors,
        materialized.radii(3.0), *args,
        scales=materialized.scales(), rotations=materialized.rotations,
    )
    assert torch.allclose(filtered_image, materialized_image, atol=2e-6)


def test_initial_and_sampled_add_rows_keep_distinct_birth_cohorts():
    H, W = 12, 10
    field = _field(4)
    target = torch.ones(H, W, 3)
    current = torch.zeros_like(target)
    cfg = FitConfig(
        covariance_filter_mode="generation_density",
        split_mode="residual_add",
        split_count=2,
        max_gaussians=6,
    )

    grown, added = _add_from_residual(field, target, current, cfg)

    assert added == 2
    assert grown.filter_variance is not None
    initial = H * W / (cfg.covariance_filter_alpha * 4)
    new = H * W / (cfg.covariance_filter_alpha * 6)
    assert torch.allclose(grown.filter_variance[:4], torch.full((4,), initial))
    assert torch.allclose(grown.filter_variance[4:], torch.full((2,), new))
    assert initial > new


def test_filter_lifecycle_subset_append_npz_and_relocation(tmp_path):
    field = _field(4)
    field.filter_variance = torch.tensor([1.0, 2.0, 3.0, 4.0])
    subset = field.subset(torch.tensor([True, False, True, False]))
    joined = subset.append(_field(1))
    assert torch.equal(joined.filter_variance, torch.tensor([1.0, 3.0, 0.0]))

    path = tmp_path / "filtered.npz"
    field.save(str(path))
    loaded = GaussianField.load(str(path))
    assert torch.equal(loaded.filter_variance, field.filter_variance)

    H = W = 12
    target = torch.ones(H, W, 3)
    current = torch.zeros_like(target)
    cfg = FitConfig(
        covariance_filter_mode="generation_density",
        relocate_count=1,
        relocate_init_opacity=0.1,
    )
    relocated, count, reset_idx, _ = _relocate_from_residual(
        loaded, target, current, cfg
    )
    assert count == 1 and reset_idx is not None
    expected = H * W / (cfg.covariance_filter_alpha * field.n)
    assert float(relocated.filter_variance[reset_idx[0]]) == pytest.approx(expected)


def test_gsplat_path_receives_effective_scales(monkeypatch):
    from structsplat import render as render_module

    field = _field(2)
    field.filter_variance = torch.tensor([0.5, 2.0])
    captured = {}

    def fake_cuda_sum(means, scales, rotations, colors, H, W, opacities=None, **kwargs):
        captured["scales"] = scales
        return torch.zeros(H, W, 3, dtype=colors.dtype, device=colors.device)

    monkeypatch.setattr(render_module, "render_cuda_sum", fake_cuda_sum)
    cfg = FitConfig(renderer="gsplat", aa_dilation=0.3)
    _render(field, cfg, 12, 12)

    assert torch.allclose(captured["scales"], field.effective_scales(0.3))


def test_exact_cuda_path_receives_filtered_conics_and_radii(monkeypatch):
    from structsplat import cuda_render

    field = _field(2)
    field.filter_variance = torch.tensor([0.5, 2.0])
    captured = {}

    def fake_cuda_exact(means, conics, colors, radii, H, W, **kwargs):
        captured["conics"] = conics
        captured["radii"] = radii
        return torch.zeros(H, W, 3, dtype=colors.dtype, device=colors.device)

    monkeypatch.setattr(cuda_render, "render_cuda_exact", fake_cuda_exact)
    cfg = FitConfig(renderer="cuda", aa_dilation=0.3)
    _render(field, cfg, 12, 12)

    assert torch.allclose(captured["conics"], field.conics(0.3))
    assert torch.equal(captured["radii"], field.radii(cfg.sigma_cutoff, 0.3))


def test_codec_materializes_filter_variance_without_losing_covariance():
    field = _field(4)
    field.filter_variance = torch.tensor([0.2, 0.7, 1.4, 2.2])
    ccfg = codec.CodecConfig(bits_scales=16, morton_reorder=False)

    blob = codec.encode(field, 12, 12, ccfg)
    decoded = codec.decode(blob)
    header = codec.blob_header(blob)

    assert header["covariance_filter_materialized"] is True
    assert decoded.filter_variance is None
    span = max(
        header["scale_hi"][0] - header["scale_lo"][0],
        header["scale_hi"][1] - header["scale_lo"][1],
    )
    error = (decoded.log_scales - field.effective_log_scales()).abs().max().item()
    assert error <= span / (2 ** ccfg.bits_scales - 1) + 1e-5
