from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from structsplat.gaussians import GaussianField
from structsplat.render import render_field
from structsplat.residual_pursuit_additive import (
    ResidualPursuitAdditiveConfig,
    append_residual_pursuit_gaussians,
)


def _base(device: str = "cpu") -> GaussianField:
    return GaussianField.from_numpy(
        np.asarray([[2.0, 2.0]], dtype=np.float32),
        np.asarray([[1.5, 1.5]], dtype=np.float32),
        np.zeros(1, dtype=np.float32),
        np.zeros((1, 3), dtype=np.float32),
        device=device,
    )


def _render(field: GaussianField, shape: tuple[int, int], mode: str = "additive"):
    return render_field(
        field.means,
        field.conics(),
        field.colors,
        field.radii(3.0),
        shape[0],
        shape[1],
        mode=mode,
        chunk=8,
        scales=field.scales(),
        rotations=field.rotations,
        sigma_cutoff=3.0,
    )


def _fixtures(size: int = 6) -> dict[str, np.ndarray]:
    yy, xx = np.mgrid[:size, :size]
    x = xx.astype(np.float32) / float(size - 1)
    y = yy.astype(np.float32) / float(size - 1)
    constant = np.full((size, size, 3), 0.375, dtype=np.float32)
    ramp = np.stack((x, y, 0.2 + 0.6 * x), axis=2).astype(np.float32)
    edge = np.full((size, size, 3), 0.05, dtype=np.float32)
    edge[:, size // 2 :] = np.asarray([0.95, 0.7, 0.2], dtype=np.float32)
    radius2 = (x - 0.55) ** 2 + (y - 0.4) ** 2
    blob_value = 0.1 + 0.85 * np.exp(-radius2 / 0.035)
    blob = np.stack((blob_value, 0.8 * blob_value, 0.35 + 0.3 * blob_value), axis=2)
    texture_value = ((xx + yy) % 2).astype(np.float32)
    texture = np.stack((texture_value, 1.0 - texture_value, 0.5 * texture_value), axis=2)
    return {
        "constant": constant,
        "ramp": ramp,
        "edge": edge,
        "blob": blob.astype(np.float32),
        "texture": texture.astype(np.float32),
    }


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"tail_gaussians": 0}, "tail_gaussians"),
        ({"scale_px": 0.0}, "scale_px"),
        ({"coefficient_abs_limit": 0.0}, "coefficient_abs_limit"),
        ({"sigma_cutoff": 0.0}, "sigma_cutoff"),
        ({"render_chunk": 0}, "render_chunk"),
        ({"renderer": "normalized"}, "additive semantics"),
    ],
)
def test_config_rejects_invalid_contracts(kwargs, message):
    with pytest.raises((TypeError, ValueError), match=message):
        ResidualPursuitAdditiveConfig(**kwargs)


def test_row_major_tie_break_and_signed_residual_coefficients():
    target = np.zeros((5, 5, 3), dtype=np.float32)
    target[1, 3] = np.asarray([0.6, -0.4, 0.2], dtype=np.float32)
    target[3, 1] = np.asarray([0.6, -0.4, 0.2], dtype=np.float32)

    result = append_residual_pursuit_gaussians(
        _base(),
        target,
        config=ResidualPursuitAdditiveConfig(tail_gaussians=1, render_chunk=8),
    )

    assert (result.trajectory[0].x, result.trajectory[0].y) == (3, 1)
    assert result.trajectory[0].coefficient_rgb[1] < 0.0
    assert result.final_pixel_rmse_max <= result.initial_pixel_rmse_max


def test_pursuit_is_deterministic_pure_and_preserves_the_base_prefix(tmp_path):
    target = _fixtures()["blob"]
    field = _base()
    original = field.detached()
    config = ResidualPursuitAdditiveConfig(tail_gaussians=8, render_chunk=8)

    first = append_residual_pursuit_gaussians(field, target, config=config)
    second = append_residual_pursuit_gaussians(field, target, config=config)

    assert first.completed
    assert first.base_count == 1
    assert first.tail_count == 8
    assert first.total_count == 9
    assert first.residual_scan_pixel_evaluations == 8 * 6 * 6
    assert first.renderer_calls == 2
    assert first.base_prefix_bit_exact
    assert first.fixed_tail_geometry
    assert first.training_payload_removed
    assert first.analytic_render_parity_max_abs <= 2e-5
    assert first.coefficient_abs_max <= config.coefficient_abs_limit
    assert first.endpoint_field_digest == second.endpoint_field_digest
    assert first.trajectory_records() == second.trajectory_records()
    assert np.array_equal(first.reconstruction_raw, second.reconstruction_raw)
    assert all(
        torch.equal(getattr(first.field, name)[: field.n], getattr(field, name))
        for name in ("means", "log_scales", "rotations", "colors")
    )
    assert all(
        torch.equal(getattr(field, name), getattr(original, name))
        for name in ("means", "log_scales", "rotations", "colors")
    )

    path = tmp_path / "endpoint.npz"
    first.field.save(str(path))
    with np.load(path) as payload:
        assert set(payload.files) == {"means", "log_scales", "rotations", "colors"}
    cold = GaussianField.load(str(path))
    replay = _render(cold, target.shape[:2]).detach().numpy()
    assert np.max(np.abs(replay - first.reconstruction_raw)) <= 2e-5


@pytest.mark.parametrize("fixture", ["constant", "ramp", "edge", "blob", "texture"])
def test_pursuit_reduces_procedural_residual_tail(fixture):
    target = _fixtures()[fixture]
    result = append_residual_pursuit_gaussians(
        _base(),
        target,
        config=ResidualPursuitAdditiveConfig(
            tail_gaussians=target.shape[0] * target.shape[1],
            render_chunk=8,
        ),
    )

    initial_mse = np.mean((result.base_reconstruction_raw - target) ** 2)
    final_mse = np.mean((result.reconstruction_raw - target) ** 2)
    assert final_mse < initial_mse
    assert result.final_pixel_rmse_max < result.initial_pixel_rmse_max
    assert result.field.n == 1 + target.shape[0] * target.shape[1]


def test_returned_endpoint_has_finite_gradients_for_every_parameter_group():
    target = _fixtures()["edge"]
    result = append_residual_pursuit_gaussians(
        _base(),
        target,
        config=ResidualPursuitAdditiveConfig(tail_gaussians=8, render_chunk=8),
    )
    trainable = result.field.detached().trainable()
    rendered = _render(trainable, target.shape[:2])
    rendered.square().mean().backward()

    for name in ("means", "log_scales", "rotations", "colors"):
        gradient = getattr(trainable, name).grad
        assert gradient is not None
        assert torch.isfinite(gradient).all()


def test_pursuit_rejects_impure_input_and_out_of_bounds_coefficients():
    pure = _base()
    impure = GaussianField(
        pure.means,
        pure.log_scales,
        pure.rotations,
        pure.colors,
        opacities=torch.ones(pure.n),
    )
    with pytest.raises(ValueError, match="pure four-array"):
        append_residual_pursuit_gaussians(impure, _fixtures()["ramp"])

    with pytest.raises(RuntimeError, match="coefficient_abs_limit"):
        append_residual_pursuit_gaussians(
            pure,
            np.full((5, 5, 3), 20.0, dtype=np.float32),
            config=ResidualPursuitAdditiveConfig(
                tail_gaussians=1,
                coefficient_abs_limit=1.0,
                render_chunk=8,
            ),
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_owned_cuda_construction_cold_replay_and_cpu_renderer_parity(tmp_path):
    target = _fixtures()["texture"]
    config = replace(
        ResidualPursuitAdditiveConfig(tail_gaussians=8, render_chunk=8),
        renderer="cuda_additive",
    )
    result = append_residual_pursuit_gaussians(_base("cuda"), target, config=config)
    assert result.completed
    assert result.analytic_render_parity_max_abs <= 2e-5

    path = tmp_path / "field.npz"
    result.field.save(str(path))
    cold_cuda = GaussianField.load(str(path), device="cuda")
    cuda_render = _render(cold_cuda, target.shape[:2], "cuda_additive").detach().cpu()
    cold_cpu = GaussianField.load(str(path), device="cpu")
    cpu_render = _render(cold_cpu, target.shape[:2], "additive").detach().cpu()

    assert torch.max(torch.abs(cuda_render - cpu_render)).item() <= 2e-4
    assert np.max(np.abs(cuda_render.numpy() - result.reconstruction_raw)) <= 2e-5
