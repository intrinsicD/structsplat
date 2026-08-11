# ruff: noqa: E402
import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from structsplat.additive_continuation import (
    AdditiveContinuationConfig,
    continuation_phase_lengths,
    continuation_schedule,
    fit_additive_continuation,
    render_additive_continuation,
)
from structsplat.gaussians import GaussianField
from structsplat.render import render_field


def _field(device="cpu", *, trainable=False):
    result = GaussianField.from_numpy(
        means=np.asarray([[2.0, 2.0], [5.0, 4.0], [3.0, 6.0]], dtype=np.float32),
        scales=np.asarray([[1.5, 1.1], [1.2, 1.7], [1.4, 1.0]], dtype=np.float32),
        angles=np.asarray([0.1, 0.8, 1.3], dtype=np.float32),
        colors=np.asarray(
            [[0.8, 0.2, 0.1], [0.1, 0.7, 0.3], [0.2, 0.3, 0.9]],
            dtype=np.float32,
        ),
        device=device,
    )
    return result.trainable() if trainable else result


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"steps": 3}, "steps must be >= 4"),
        ({"coverage_weight": -1.0}, "coverage_weight must be >= 0.0"),
        ({"ssim_weight": 1.1}, "ssim_weight must be <= 1.0"),
        ({"renderer": "normalized"}, "renderer must use additive semantics"),
        ({"pixel_loss": "charbonnier"}, "pixel_loss must be 'l1' or 'l2'"),
    ],
)
def test_config_rejects_invalid_values(kwargs, message):
    with pytest.raises(ValueError, match=message):
        AdditiveContinuationConfig(**kwargs)


def test_frozen_schedule_has_exact_35_50_15_phases():
    assert continuation_phase_lengths(500) == (175, 250, 75)
    assert continuation_schedule(175, 500).lambda_value == 1.0
    first_anneal = continuation_schedule(176, 500)
    assert first_anneal.phase == "anneal"
    assert 0.0 < first_anneal.lambda_value < 1.0
    anneal_end = continuation_schedule(425, 500)
    assert anneal_end.phase == "anneal"
    assert anneal_end.lambda_value == 0.0
    assert not anneal_end.endpoint_eligible
    endpoint = continuation_schedule(426, 500)
    assert endpoint.phase == "endpoint"
    assert endpoint.lambda_value == 0.0
    assert endpoint.endpoint_eligible


def test_render_matches_both_closed_form_endpoints_and_zero_support():
    field = _field()
    config = AdditiveContinuationConfig(steps=4, ssim_weight=0.0)
    log_masses = torch.log(torch.tensor([0.4, 1.2, 0.8]))
    got_additive = render_additive_continuation(
        field, log_masses, 8, 9, 0.0, config=config
    )
    expected_additive = render_field(
        field.means,
        field.conics(),
        field.colors,
        field.radii(3.0),
        8,
        9,
        mode="additive",
    )
    assert got_additive.image is got_additive.numerator
    assert torch.equal(got_additive.image, expected_additive)

    masses = torch.exp(log_masses)
    expected_denominator = render_field(
        field.means,
        field.conics(),
        torch.ones_like(field.colors),
        field.radii(3.0),
        8,
        9,
        mode="additive",
        opacities=masses,
    )[..., :1]
    got_normalized = render_additive_continuation(
        field, log_masses, 8, 9, 1.0, config=config
    )
    expected_normalized = expected_additive / (
        expected_denominator + config.normalization_eps
    )
    assert torch.allclose(got_normalized.denominator, expected_denominator)
    assert torch.allclose(got_normalized.image, expected_normalized)

    outside = GaussianField.from_numpy(
        np.asarray([[100.0, 100.0]]),
        np.asarray([[1.0, 1.0]]),
        np.zeros(1),
        np.ones((1, 3)),
    )
    weak = render_additive_continuation(
        outside, torch.zeros(1), 4, 5, 1.0, config=config
    )
    assert torch.isfinite(weak.image).all()
    assert torch.count_nonzero(weak.image) == 0


def test_intermediate_render_backpropagates_to_geometry_coefficients_and_positive_mass():
    field = _field(trainable=True)
    log_masses = torch.tensor([-0.4, 0.0, 0.5], requires_grad=True)
    rendered = render_additive_continuation(
        field,
        log_masses,
        8,
        9,
        0.45,
        config=AdditiveContinuationConfig(steps=4, ssim_weight=0.0),
    )
    assert bool((torch.exp(log_masses) > 0.0).all())
    rendered.image.square().mean().backward()
    for gradient in (
        field.means.grad,
        field.log_scales.grad,
        field.rotations.grad,
        field.colors.grad,
        log_masses.grad,
    ):
        assert gradient is not None
        assert torch.isfinite(gradient).all()
        assert bool((gradient != 0.0).any())


def test_fit_is_deterministic_selects_only_endpoint_and_discards_mass(tmp_path):
    yy, xx = np.mgrid[:8, :9]
    target = np.stack(
        [xx / 8.0, yy / 7.0, 0.5 + 0.2 * np.sin(xx)],
        axis=2,
    ).astype(np.float32)
    source = _field()
    source_copy = source.detached()
    config = AdditiveContinuationConfig(
        steps=8,
        checkpoint_every=2,
        pixel_loss="l2",
        ssim_weight=0.0,
        coverage_weight=0.05,
        renderer="additive",
        render_chunk=1,
    )
    first = fit_additive_continuation(source, target, config=config)
    second = fit_additive_continuation(source, target, config=config)

    assert first.completed
    assert first.selected_step == 8
    selected = [checkpoint for checkpoint in first.checkpoints if checkpoint.selected]
    assert len(selected) == 1
    assert selected[0].phase == "endpoint"
    assert selected[0].lambda_value == 0.0
    assert first.field.opacities is None
    assert first.field.scale_max is None
    assert first.field.color_grads is None
    assert first.endpoint_parity_max_abs == 0.0
    assert float(first.field.colors.abs().max()) <= config.coefficient_abs_limit
    assert np.array_equal(first.reconstruction_raw, second.reconstruction_raw)
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
    assert np.array_equal(first.reconstruction_raw, expected)
    path = tmp_path / "endpoint.npz"
    first.field.save(str(path))
    with np.load(path) as payload:
        assert "opacities" not in payload.files
        assert all("mass" not in name for name in payload.files)


def test_cuda_forward_and_backward_match_reference_when_available():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    rng = np.random.default_rng(22022)
    count, height, width = 7, 10, 12
    means = np.stack(
        [rng.uniform(1.0, width - 2.0, count), rng.uniform(1.0, height - 2.0, count)],
        axis=1,
    )
    scales = rng.uniform(0.8, 2.0, (count, 2))
    angles = rng.uniform(0.0, math.pi, count)
    colors = rng.uniform(-0.2, 1.0, (count, 3))
    masses = rng.uniform(0.2, 1.5, count)
    target = torch.rand(
        height,
        width,
        3,
        device="cuda",
        generator=torch.Generator(device="cuda").manual_seed(22022),
    )

    def run(renderer):
        field = GaussianField.from_numpy(
            means, scales, angles, colors, device="cuda"
        ).trainable()
        log_masses = torch.tensor(
            np.log(masses), device="cuda", dtype=torch.float32, requires_grad=True
        )
        rendered = render_additive_continuation(
            field,
            log_masses,
            height,
            width,
            0.37,
            config=AdditiveContinuationConfig(
                steps=4, renderer=renderer, ssim_weight=0.0, render_chunk=1
            ),
        )
        (rendered.image - target).square().mean().backward()
        torch.cuda.synchronize()
        return (
            rendered.image.detach(),
            rendered.denominator.detach(),
            field.means.grad.detach(),
            field.log_scales.grad.detach(),
            field.rotations.grad.detach(),
            field.colors.grad.detach(),
            log_masses.grad.detach(),
        )

    try:
        reference = run("additive")
        candidate = run("cuda_additive")
    except RuntimeError as exc:
        pytest.skip(str(exc))
    for got, expected in zip(candidate, reference, strict=True):
        assert torch.allclose(got, expected, atol=5e-4, rtol=5e-4)
