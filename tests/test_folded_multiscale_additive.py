# ruff: noqa: E402
from dataclasses import replace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from structsplat.folded_multiscale_additive import (
    FoldedMultiscaleAdditiveConfig,
    area_bilinear_lowpass,
    fit_folded_multiscale_additive,
)
from structsplat.gaussians import GaussianField
from structsplat.render import render_field


def _fixtures(size: int = 24) -> dict[str, np.ndarray]:
    yy, xx = np.mgrid[:size, :size]
    x = xx.astype(np.float32) / float(size - 1)
    y = yy.astype(np.float32) / float(size - 1)
    constant = np.full((size, size, 3), 0.375, dtype=np.float32)
    ramp = np.stack((x, y, 0.2 + 0.6 * x), axis=2).astype(np.float32)
    edge = np.full((size, size, 3), 0.1, dtype=np.float32)
    edge[:, size // 2 :] = np.asarray([0.9, 0.7, 0.2], dtype=np.float32)
    radius2 = (x - 0.55) ** 2 + (y - 0.4) ** 2
    blob_value = 0.15 + 0.75 * np.exp(-radius2 / 0.035)
    blob = np.stack((blob_value, 0.8 * blob_value, 0.4 + 0.3 * blob_value), axis=2)
    texture_value = ((xx // 3 + yy // 3) % 2).astype(np.float32)
    texture = np.stack((texture_value, 1.0 - texture_value, 0.5 * texture_value), axis=2)
    return {
        "constant": constant,
        "ramp": ramp,
        "edge": edge,
        "blob": blob.astype(np.float32),
        "texture": texture.astype(np.float32),
    }


def _config(renderer: str = "additive") -> FoldedMultiscaleAdditiveConfig:
    return FoldedMultiscaleAdditiveConfig(
        total_gaussians=12,
        coarse_gaussians=4,
        coarse_steps=1,
        detail_steps=1,
        joint_steps=1,
        checkpoint_every=1,
        detail_feature_cap_px=6.0,
        renderer=renderer,
        render_chunk=8,
    )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"total_gaussians": 1, "coarse_gaussians": 1}, "smaller"),
        ({"coarse_gaussians": 0}, "coarse_gaussians"),
        ({"coarse_steps": 0}, "coarse_steps"),
        ({"coefficient_abs_limit": 0.0}, "coefficient_abs_limit"),
        ({"renderer": "normalized"}, "renderer must be"),
    ],
)
def test_config_rejects_invalid_contracts(kwargs, message):
    with pytest.raises((TypeError, ValueError), match=message):
        FoldedMultiscaleAdditiveConfig(**kwargs)


def test_area_bilinear_lowpass_is_deterministic_shape_preserving_and_smoothing():
    target = _fixtures(25)["texture"]
    first = area_bilinear_lowpass(target, 2)
    second = area_bilinear_lowpass(target, 2)

    assert first.shape == target.shape
    assert first.dtype == np.float32
    assert not first.flags.writeable
    assert np.array_equal(first, second)
    assert float(first.std()) < float(target.std())
    constant = _fixtures(25)["constant"]
    assert np.max(np.abs(area_bilinear_lowpass(constant, 2) - constant)) <= 1e-7
    assert np.array_equal(area_bilinear_lowpass(target, 1), target)


def test_folded_fit_has_exact_count_geometry_fold_and_pure_endpoint(tmp_path):
    target = _fixtures()["ramp"]
    result = fit_folded_multiscale_additive(target, config=_config(), seed=3)

    assert result.completed
    assert result.status == "completed"
    assert result.field.n == 12
    assert result.coarse_count == 4
    assert result.detail_count == 8
    assert result.attempted_steps == result.completed_steps == 3
    assert [point.step for point in result.trajectory] == [1, 2, 3]
    assert [point.stage for point in result.trajectory] == ["coarse", "detail", "joint"]
    assert result.observer_renderer_calls == 3
    assert result.coarse_geometry_exact
    assert result.training_mask_removed
    assert result.fold_parity_max_abs <= 2e-5
    assert result.endpoint_parity_max_abs <= 2e-5
    assert result.field.opacities is None
    assert result.field.color_grads is None
    assert result.field.background_mask is None
    assert result.reconstruction_raw.shape == target.shape
    assert np.isfinite(result.reconstruction_raw).all()

    path = tmp_path / "field.npz"
    result.field.save(str(path))
    with np.load(path) as payload:
        assert "background_mask" not in payload.files
        assert "opacities" not in payload.files
        assert "color_grads" not in payload.files
    cold = GaussianField.load(str(path))
    rendered = render_field(
        cold.means,
        cold.conics(),
        cold.colors,
        cold.radii(3.0),
        target.shape[0],
        target.shape[1],
        mode="additive",
        scales=cold.scales(),
        rotations=cold.rotations,
        sigma_cutoff=3.0,
    ).detach().numpy()
    assert np.max(np.abs(rendered - result.reconstruction_raw)) <= 2e-5


def test_folded_fit_is_deterministic_on_cpu():
    target = _fixtures()["blob"]
    first = fit_folded_multiscale_additive(target, config=_config(), seed=7)
    second = fit_folded_multiscale_additive(target, config=_config(), seed=7)

    assert np.array_equal(first.reconstruction_raw, second.reconstruction_raw)
    for name in ("means", "log_scales", "rotations", "colors"):
        assert torch.equal(getattr(first.field, name), getattr(second.field, name))
    assert first.trajectory_records() == second.trajectory_records()


@pytest.mark.parametrize("fixture", ["constant", "ramp", "edge", "blob", "texture"])
def test_procedural_killing_fixtures_complete_without_count_or_finite_failure(fixture):
    result = fit_folded_multiscale_additive(
        _fixtures()[fixture], config=_config(), seed=0
    )
    assert result.completed
    assert result.field.n == 12
    assert result.coefficient_abs_max <= 16.0
    assert np.isfinite(result.reconstruction_raw).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_owned_cuda_folded_fit_and_cold_additive_parity(tmp_path):
    target = _fixtures()["edge"]
    config = replace(_config(), renderer="cuda_additive")
    result = fit_folded_multiscale_additive(
        target, config=config, seed=2, device="cuda"
    )
    assert result.completed
    assert result.fold_parity_max_abs <= 2e-5
    assert result.endpoint_parity_max_abs <= 2e-5

    path = tmp_path / "field.npz"
    result.field.save(str(path))
    cold = GaussianField.load(str(path), device="cuda")
    rendered = render_field(
        cold.means,
        cold.conics(),
        cold.colors,
        cold.radii(3.0),
        target.shape[0],
        target.shape[1],
        mode="cuda_additive",
        scales=cold.scales(),
        rotations=cold.rotations,
        sigma_cutoff=3.0,
    ).detach().cpu().numpy()
    assert np.max(np.abs(rendered - result.reconstruction_raw)) <= 2e-5
