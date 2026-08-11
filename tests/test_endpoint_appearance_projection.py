# ruff: noqa: E402
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from structsplat.endpoint_appearance_projection import (
    EndpointAppearanceProjectionConfig,
    ProjectionSafetyConfig,
    project_additive_endpoint,
    select_safe_projection,
)
from structsplat.gaussians import GaussianField
from structsplat.render import render_field


def _field(device="cpu", *, colors=None):
    if colors is None:
        colors = np.asarray(
            [[0.4, 0.1, 0.2], [0.2, 0.5, 0.1], [0.1, 0.2, 0.4]],
            dtype=np.float32,
        )
    return GaussianField.from_numpy(
        means=np.asarray([[2.0, 2.0], [6.0, 3.0], [4.0, 6.0]], dtype=np.float32),
        scales=np.asarray([[1.5, 1.2], [1.4, 1.8], [1.7, 1.1]], dtype=np.float32),
        angles=np.asarray([0.1, 0.7, 1.2], dtype=np.float32),
        colors=colors,
        device=device,
    )


def _target(device="cpu"):
    target_field = _field(
        device,
        colors=np.asarray(
            [[0.8, 0.2, 0.1], [0.1, 0.7, 0.3], [0.2, 0.3, 0.9]],
            dtype=np.float32,
        ),
    )
    return render_field(
        target_field.means,
        target_field.conics(),
        target_field.colors,
        target_field.radii(3.0),
        8,
        9,
        mode="additive" if device == "cpu" else "cuda_additive",
    ).detach().cpu().numpy()


def _metrics(**updates):
    result = {
        "raw_mse": 0.02,
        "ms_ssim": 0.95,
        "lpips": 0.10,
        "pixel_max": 0.3,
        "patch7_max": 0.15,
    }
    result.update(updates)
    return result


def test_config_rejects_nonadditive_renderer_and_bad_safety_limits():
    with pytest.raises(ValueError, match="renderer must be"):
        EndpointAppearanceProjectionConfig(renderer="normalized")
    with pytest.raises(ValueError, match="coefficient_abs_limit must be > 0"):
        ProjectionSafetyConfig(coefficient_abs_limit=0.0)


def test_safety_selector_accepts_only_strict_finite_multimetric_improvement():
    incoming = _metrics()
    proposal = _metrics(
        raw_mse=0.019,
        ms_ssim=0.951,
        lpips=0.099,
        pixel_max=0.29,
        patch7_max=0.14,
    )
    accepted = select_safe_projection(
        incoming, proposal, proposal_finite=True, coefficient_abs_max=2.0
    )
    assert accepted.selected
    assert accepted.reason == "selected"
    assert all(accepted.clauses.values())

    lpips_rollback = select_safe_projection(
        incoming,
        _metrics(raw_mse=0.019, lpips=0.1001),
        proposal_finite=True,
        coefficient_abs_max=2.0,
    )
    assert not lpips_rollback.selected
    assert not lpips_rollback.clauses["lpips_safe"]

    nonfinite = select_safe_projection(
        incoming,
        _metrics(raw_mse=float("nan")),
        proposal_finite=False,
        coefficient_abs_max=float("inf"),
    )
    assert not nonfinite.selected
    assert not nonfinite.clauses["finite"]
    assert not nonfinite.clauses["bounded"]


def test_cpu_projection_is_deterministic_geometry_exact_and_pure_additive(tmp_path):
    source = _field()
    source_copy = source.detached()
    target = _target()
    config = EndpointAppearanceProjectionConfig(render_chunk=1)
    first = project_additive_endpoint(source, target, config=config)
    second = project_additive_endpoint(source, target, config=config)

    assert first.geometry_exact
    assert first.field.n == source.n
    assert first.field.opacities is None
    assert first.field.color_grads is None
    assert first.projection.final_sse <= first.projection.initial_sse
    assert first.projection.selected_iteration >= 0
    assert first.projection.initial_operator_parity_max_abs < 1e-6
    assert first.projection.maintained_render_parity_max_abs < 1e-6
    assert np.array_equal(first.reconstruction_raw, second.reconstruction_raw)
    assert torch.equal(first.field.means, source.means)
    assert torch.equal(first.field.log_scales, source.log_scales)
    assert torch.equal(first.field.rotations, source.rotations)
    for left, right in (
        (source.means, source_copy.means),
        (source.log_scales, source_copy.log_scales),
        (source.rotations, source_copy.rotations),
        (source.colors, source_copy.colors),
    ):
        assert torch.equal(left, right)

    expected = render_field(
        first.field.means,
        first.field.conics(),
        first.field.colors,
        first.field.radii(3.0),
        8,
        9,
        mode="additive",
    ).detach().numpy()
    assert np.allclose(first.reconstruction_raw, expected, atol=1e-6, rtol=1e-6)
    path = tmp_path / "projected.npz"
    first.field.save(str(path))
    with np.load(path) as payload:
        assert "opacities" not in payload.files
        assert all("mass" not in key and "denom" not in key for key in payload.files)


def test_cuda_projection_cold_parity_when_available():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    source = _field("cuda")
    target = _target("cuda")
    try:
        result = project_additive_endpoint(
            source,
            target,
            config=EndpointAppearanceProjectionConfig(
                renderer="cuda_additive", render_chunk=1
            ),
            device="cuda",
        )
    except RuntimeError as exc:
        pytest.skip(str(exc))
    assert result.geometry_exact
    assert result.projection.maintained_render_parity_max_abs <= 2e-5
    assert result.projection.adjoint_relative_error <= 5e-5
    assert float(result.field.colors.abs().max()) <= 16.0
