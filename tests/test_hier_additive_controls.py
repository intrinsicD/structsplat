from dataclasses import replace

import pytest
import torch

from benchmarks.hier_additive_controls import (
    ControlConfig, additive_render, curvature_direction, fit_control, pack,
)
from structsplat.gaussians import GaussianField


def fixture():
    field = GaussianField(torch.tensor([[5.0, 6.0], [10.0, 8.0]], dtype=torch.float64),
        torch.tensor([[1.8, 1.2], [1.5, 2.0]], dtype=torch.float64).log(),
        torch.tensor([0.3, -0.4], dtype=torch.float64),
        torch.tensor([[0.4, 0.2, 0.3], [0.2, 0.5, 0.3]], dtype=torch.float64))
    target = additive_render(field, 15, 17).detach()
    initial = field.detached()
    initial.means += 0.4
    initial.colors *= 0.8
    return initial, target


@pytest.mark.parametrize("arm", ["adam", "diagonal", "block"])
def test_control_counts_terminal_steps_reduces_fixture_loss_and_preserves_input(arm):
    initial, target = fixture()
    before = pack(initial).clone()
    field, raw, history, elapsed = fit_control(initial, target, ControlConfig(arm=arm, steps=8))
    assert len(history) == 9 and history[-1]["iteration"] == 8
    assert history[-1]["gradient_evaluations"] == 8
    assert history[-1]["objective"] < history[0]["objective"]
    assert elapsed >= history[-1]["elapsed_seconds"]
    assert all(a["elapsed_seconds"] <= b["elapsed_seconds"] for a, b in zip(history, history[1:]))
    torch.testing.assert_close(pack(initial), before, rtol=0, atol=0)
    torch.testing.assert_close(additive_render(field, 15, 17), raw)
    assert field.n == initial.n
    if arm != "adam":
        assert all(a["objective"] >= b["objective"] for a, b in zip(history, history[1:]))
        assert history[-1]["forward_evaluations"] == 1 + sum(h["line_search_trials"] for h in history)


@pytest.mark.parametrize("arm", ["diagonal", "block"])
def test_curvature_direction_is_bounded_and_handles_singular_blocks(arm):
    cfg = ControlConfig(arm=arm)
    gradient = torch.ones(3, 8, dtype=torch.float64)
    gram = torch.eye(8, dtype=torch.float64).repeat(3, 1, 1)
    gram[0].zero_()
    delta = curvature_direction(gradient, gram, cfg)
    assert bool(torch.isfinite(delta).all())
    assert bool((delta.abs() <= delta.new_tensor(cfg.trust)).all())
    assert bool((delta * gradient).sum(1).lt(0).all())


def test_invalid_controls_and_nonfinite_inputs_fail_closed():
    initial, target = fixture()
    for kwargs in ({"steps": 0}, {"arm": "other"}, {"damping": 0}, {"trust": (1,)}):
        with pytest.raises(ValueError):
            ControlConfig(**kwargs)
    with pytest.raises(ValueError, match="target"):
        fit_control(initial, target * float("nan"), ControlConfig(steps=1))
    initial.opacities = torch.ones(2)
    with pytest.raises(ValueError, match="semantics"):
        fit_control(initial, target, ControlConfig(steps=1))


def test_block_control_is_reproducible_on_cpu():
    initial, target = fixture()
    cfg = replace(ControlConfig(), arm="block", steps=4)
    first = fit_control(initial, target, cfg)
    second = fit_control(initial, target, cfg)
    torch.testing.assert_close(pack(first[0]), pack(second[0]), rtol=0, atol=0)
    assert [h["objective"] for h in first[2]] == [h["objective"] for h in second[2]]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_pixel_packet_matches_owned_cuda_additive_gradient():
    from benchmarks.hier_additive_controls import unpack
    from structsplat.pixel_gradient import pixel_gradient_packet

    initial, target = fixture()
    field = unpack(pack(initial).float().cuda()).trainable()
    target = target.float().cuda()
    raw = additive_render(field, 15, 17, renderer="cuda_additive")
    (0.5 * (raw - target).square().mean()).backward()
    expected = torch.cat((field.means.grad, field.log_scales.grad,
                          field.rotations.grad[:, None], field.colors.grad), 1)
    packet = pixel_gradient_packet(field, (raw - target).detach() / target.numel())
    torch.testing.assert_close(packet.signed, expected, atol=2e-7, rtol=2e-4)
