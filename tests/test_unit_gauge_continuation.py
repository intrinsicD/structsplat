# ruff: noqa: E402
import math
from dataclasses import replace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from structsplat.gaussians import GaussianField
from structsplat.render import render_field
from structsplat.unit_gauge_continuation import (
    UnitGaugeContinuationConfig,
    fit_unit_gauge_continuation,
    render_unit_gauge,
    unit_gauge_phase_lengths,
    unit_gauge_schedule,
)


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
        ({"ssim_weight": 1.1}, "ssim_weight must be <= 1.0"),
        ({"normalized_renderer": "additive"}, "unknown normalized renderer"),
        ({"additive_renderer": "normalized"}, "unknown additive renderer"),
        ({"pixel_loss": "charbonnier"}, "pixel_loss must be 'l1' or 'l2'"),
    ],
)
def test_config_rejects_invalid_values(kwargs, message):
    with pytest.raises(ValueError, match=message):
        UnitGaugeContinuationConfig(**kwargs)


def test_frozen_schedule_has_exact_35_15_50_phases_and_clean_boundary():
    assert unit_gauge_phase_lengths(500) == (175, 75, 250)
    assert unit_gauge_schedule(175, 500).lambda_value == 1.0
    first_anneal = unit_gauge_schedule(176, 500)
    assert first_anneal.phase == "anneal"
    assert 0.0 < first_anneal.lambda_value < 1.0
    anneal_end = unit_gauge_schedule(250, 500)
    assert anneal_end.phase == "anneal"
    assert 0.0 < anneal_end.lambda_value < 1.0
    assert not anneal_end.endpoint_eligible
    endpoint = unit_gauge_schedule(251, 500)
    assert endpoint.phase == "endpoint"
    assert endpoint.lambda_value == 0.0
    assert endpoint.endpoint_eligible


def test_render_dispatches_exact_endpoints_and_is_finite_without_support():
    field = _field()
    config = UnitGaugeContinuationConfig(steps=4, ssim_weight=0.0)
    normalized = render_unit_gauge(field, 8, 9, 1.0, config=config)
    expected_normalized = render_field(
        field.means,
        field.conics(),
        field.colors,
        field.radii(3.0),
        8,
        9,
        mode="normalized",
    )
    assert normalized.numerator is None
    assert normalized.denominator is None
    assert normalized.normalized_calls == 1
    assert torch.equal(normalized.image, expected_normalized)

    additive = render_unit_gauge(field, 8, 9, 0.0, config=config)
    expected_additive = render_field(
        field.means,
        field.conics(),
        field.colors,
        field.radii(3.0),
        8,
        9,
        mode="additive",
    )
    assert additive.image is additive.numerator
    assert additive.denominator is None
    assert additive.additive_numerator_calls == 1
    assert torch.equal(additive.image, expected_additive)

    outside = GaussianField.from_numpy(
        np.asarray([[100.0, 100.0]]),
        np.asarray([[1.0, 1.0]]),
        np.zeros(1),
        np.ones((1, 3)),
    )
    weak = render_unit_gauge(outside, 4, 5, 0.4, config=config)
    assert torch.isfinite(weak.image).all()
    assert torch.count_nonzero(weak.image) == 0


def test_intermediate_render_backpropagates_to_geometry_and_coefficients():
    field = _field(trainable=True)
    rendered = render_unit_gauge(
        field,
        8,
        9,
        0.45,
        config=UnitGaugeContinuationConfig(steps=4, ssim_weight=0.0),
    )
    assert rendered.denominator is not None
    assert rendered.additive_numerator_calls == 1
    assert rendered.additive_denominator_calls == 1
    rendered.image.square().mean().backward()
    for gradient in (
        field.means.grad,
        field.log_scales.grad,
        field.rotations.grad,
        field.colors.grad,
    ):
        assert gradient is not None
        assert torch.isfinite(gradient).all()
        assert bool((gradient != 0.0).any())


def test_fit_is_deterministic_resets_once_and_persists_only_additive_field(tmp_path):
    yy, xx = np.mgrid[:8, :9]
    target = np.stack(
        [xx / 8.0, yy / 7.0, 0.5 + 0.2 * np.sin(xx)], axis=2
    ).astype(np.float32)
    source = _field()
    source_copy = source.detached()
    config = UnitGaugeContinuationConfig(
        steps=8,
        checkpoint_every=2,
        pixel_loss="l2",
        ssim_weight=0.0,
        render_chunk=1,
        reset_optimizer_at_endpoint=True,
    )
    first = fit_unit_gauge_continuation(source, target, config=config)
    second = fit_unit_gauge_continuation(source, target, config=config)

    assert first.completed
    assert first.optimizer_reset_count == 1
    assert first.optimizer_reset_step == 5
    assert first.selected_step is not None
    selected = [checkpoint for checkpoint in first.checkpoints if checkpoint.selected]
    assert len(selected) == 1
    assert selected[0].phase == "endpoint"
    assert selected[0].lambda_value == 0.0
    hold = next(checkpoint for checkpoint in first.checkpoints if checkpoint.step == 3)
    assert hold.phase == "hold"
    assert hold.optimizer_reset_count == 0
    assert first.field.opacities is None
    assert first.field.scale_max is None
    assert first.field.color_grads is None
    assert first.endpoint_parity_max_abs == 0.0
    assert first.renderer_calls == (
        first.normalized_calls
        + first.additive_numerator_calls
        + first.additive_denominator_calls
    )
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
        assert all("mass" not in name and "optimizer" not in name for name in payload.files)

    no_reset = fit_unit_gauge_continuation(
        source, target, config=replace(config, reset_optimizer_at_endpoint=False)
    )
    assert no_reset.completed
    assert no_reset.optimizer_reset_count == 0
    assert no_reset.optimizer_reset_step is None


@pytest.mark.parametrize("lambda_value", [1.0, 0.37, 0.0])
def test_cuda_forward_and_backward_match_reference_when_available(lambda_value):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    rng = np.random.default_rng(23023)
    count, height, width = 7, 10, 12
    means = np.stack(
        [rng.uniform(1.0, width - 2.0, count), rng.uniform(1.0, height - 2.0, count)],
        axis=1,
    )
    scales = rng.uniform(0.8, 2.0, (count, 2))
    angles = rng.uniform(0.0, math.pi, count)
    colors = rng.uniform(-0.2, 1.0, (count, 3))
    target = torch.rand(
        height,
        width,
        3,
        device="cuda",
        generator=torch.Generator(device="cuda").manual_seed(23023),
    )

    def run(normalized_renderer, additive_renderer):
        field = GaussianField.from_numpy(
            means, scales, angles, colors, device="cuda"
        ).trainable()
        rendered = render_unit_gauge(
            field,
            height,
            width,
            lambda_value,
            config=UnitGaugeContinuationConfig(
                steps=4,
                normalized_renderer=normalized_renderer,
                additive_renderer=additive_renderer,
                ssim_weight=0.0,
                render_chunk=1,
            ),
        )
        (rendered.image - target).square().mean().backward()
        torch.cuda.synchronize()
        return (
            rendered.image.detach(),
            field.means.grad.detach(),
            field.log_scales.grad.detach(),
            field.rotations.grad.detach(),
            field.colors.grad.detach(),
        )

    try:
        reference = run("normalized", "additive")
        candidate = run("cuda", "cuda_additive")
    except RuntimeError as exc:
        pytest.skip(str(exc))
    for got, expected in zip(candidate, reference, strict=True):
        assert torch.allclose(got, expected, atol=5e-4, rtol=5e-4)
