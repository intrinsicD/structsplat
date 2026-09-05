from dataclasses import replace
import numpy as np
import pytest
import torch

from benchmarks.hier_additive_controls import (
    ControlConfig, additive_render, curvature_direction, pack, unpack,
)
from benchmarks.hier_coupling_oracle import MODES, dense_direction, dense_system, fit_coupling
from structsplat.gaussians import GaussianField
from structsplat.pixel_gradient import pixel_gradient_packet


def small_case(dtype=torch.float64):
    field = GaussianField.from_numpy(
        np.array([[5.2, 7.1], [8.7, 6.8]], np.float32),
        np.array([[2.3, 1.4], [2.1, 1.3]], np.float32),
        np.array([.31, -.27], np.float32),
        np.array([[.3, .2, .1], [.1, .3, .2]], np.float32))
    field = unpack(pack(field).to(dtype))
    target = additive_render(field, 16, 16).detach()
    initial = field.detached()
    initial.colors *= .8
    initial.means += .15
    return initial, target


@pytest.mark.parametrize("device,dtype", [("cpu", torch.float64), ("cuda", torch.float32)])
def test_dense_system_bridges_to_autograd_packet_and_old_step(device, dtype):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    from scripts.experiments.hier035_convergence import fixture
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    initial, target = fixture("translated", 77)
    field = unpack(pack(initial).to(device=device, dtype=dtype)).trainable()
    target = target.to(device=device, dtype=dtype)
    raw = additive_render(field, 64, 64)
    loss = .5 * (raw - target).square().mean()
    gradients = torch.autograd.grad(loss, (field.means, field.log_scales, field.rotations, field.colors))
    automatic = torch.cat((gradients[0], gradients[1], gradients[2][:, None], gradients[3]), 1)
    gradient, gram, size = dense_system(field, raw.detach() - target)
    packet = pixel_gradient_packet(field, (raw.detach() - target) / target.numel())
    tolerance = dict(rtol=1e-6, atol=1e-8) if device == "cpu" else dict(rtol=1e-5, atol=1e-7)
    torch.testing.assert_close(gradient, automatic, **tolerance)
    torch.testing.assert_close(gradient, packet.signed, **tolerance)
    blocks = torch.stack([gram[i*8:(i+1)*8, i*8:(i+1)*8] for i in range(field.n)])
    torch.testing.assert_close(blocks, packet.gram / target.numel(), **tolerance)
    cfg = ControlConfig(arm="block")
    dense_step, _ = dense_direction(gradient, gram, cfg, "block_row")
    old_step = curvature_direction(packet.signed, packet.gram / target.numel(), cfg)
    step_tolerance = dict(rtol=1e-6, atol=1e-8) if device == "cpu" else dict(rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(dense_step, old_step, **step_tolerance)
    assert size == target.numel() * field.n * 8 * target.element_size()


def test_dense_gram_matches_finite_jvp_and_has_cross_row_entries():
    field, target = small_case()
    raw = additive_render(field, 16, 16)
    gradient, gram, _ = dense_system(field, raw - target)
    delta = torch.linspace(-.1, .1, 16, dtype=torch.float64).reshape(2, 8)
    epsilon = 1e-5
    plus = additive_render(unpack(pack(field) + epsilon * delta), 16, 16)
    minus = additive_render(unpack(pack(field) - epsilon * delta), 16, 16)
    jvp = (plus - minus) / (2 * epsilon)
    torch.testing.assert_close((gradient * delta).sum(), ((raw - target) * jvp).mean(), rtol=1e-6, atol=1e-8)
    torch.testing.assert_close(delta.flatten() @ gram @ delta.flatten(), jvp.square().mean(), rtol=1e-6, atol=1e-8)
    assert float(gram[:8, 8:].abs().sum()) > 0


def test_factorial_only_changes_coupling_and_cap():
    field, target = small_case()
    gradient, gram, _ = dense_system(field, additive_render(field, 16, 16) - target)
    cfg = ControlConfig(arm="block", trust=(.01,) * 8)
    deltas = {mode: dense_direction(gradient, gram, cfg, mode)[0] for mode in MODES}
    assert not torch.allclose(deltas["full_shared"], deltas["block_shared"])
    for mode, delta in deltas.items():
        assert float((delta / delta.new_tensor(cfg.trust)).abs().max()) <= 1 + 1e-12
        assert float((gradient * delta).sum()) < 0
    uncoupled = gram.clone()
    uncoupled[:8, 8:] = 0
    uncoupled[8:, :8] = 0
    for cap in ("row", "shared"):
        full, _ = dense_direction(gradient, uncoupled, cfg, f"full_{cap}")
        block, _ = dense_direction(gradient, uncoupled, cfg, f"block_{cap}")
        torch.testing.assert_close(full, block, rtol=0, atol=0)


@pytest.mark.parametrize("limit", [{"max_jacobian_bytes": 1}, {"max_parameters": 15}])
def test_dense_limits_fail_before_matrix_allocation(monkeypatch, limit):
    field, target = small_case()
    def forbidden(*args, **kwargs):
        raise AssertionError("allocated after failed guard")
    monkeypatch.setattr(torch.Tensor, "new_zeros", forbidden)
    with pytest.raises(MemoryError):
        dense_system(field, target, **limit)


@pytest.mark.parametrize("mode", MODES)
def test_fitter_preserves_owned_count_bounds_terminal_and_work(mode):
    field, target = small_case()
    before = pack(field).clone()
    cfg = ControlConfig(arm="block", steps=3)
    first, raw, history, elapsed = fit_coupling(field, target, cfg, mode=mode)
    second, again, second_history, _ = fit_coupling(field, target, cfg, mode=mode)
    torch.testing.assert_close(pack(field), before, rtol=0, atol=0)
    torch.testing.assert_close(pack(first), pack(second), rtol=0, atol=0)
    torch.testing.assert_close(raw, again, rtol=0, atol=0)
    assert first.n == field.n and len(history) == 4 and history[-1]["iteration"] == 3
    assert history[-1]["linear_solves"] == history[-1]["jacobian_constructions"] == 3
    assert history[-1]["forward_evaluations"] == 1 + sum(h["line_search_trials"] for h in history)
    assert elapsed >= history[-1]["elapsed_seconds"] > 0
    assert [h["objective"] for h in history] == [h["objective"] for h in second_history]
    assert all(b["objective"] <= a["objective"] for a, b in zip(history, history[1:]))
    first.colors.zero_()
    torch.testing.assert_close(pack(field), before, rtol=0, atol=0)


def test_all_trial_rejections_retain_exact_original_state_and_charge_work(monkeypatch):
    import benchmarks.hier_coupling_oracle as module
    field, target = small_case()
    target.zero_()
    before = pack(field).clone()
    count = 0
    def increasing_render(*args, **kwargs):
        nonlocal count
        count += 1
        return torch.ones_like(target) * count
    monkeypatch.setattr(module, "additive_render", increasing_render)
    result, _, trace, _ = fit_coupling(field, target, ControlConfig(arm="block", steps=2))
    torch.testing.assert_close(pack(result), before, rtol=0, atol=0)
    assert count == 13 and trace[-1]["forward_evaluations"] == 13
    assert all(not h["accepted"] and h["line_search_trials"] == 6 for h in trace[1:])


def test_invalid_and_failed_solves_are_not_silent_fallbacks(monkeypatch):
    field, target = small_case()
    with pytest.raises(ValueError):
        fit_coupling(field, target, ControlConfig(arm="adam"))
    with pytest.raises(ValueError):
        fit_coupling(field, target, mode="unknown")
    with pytest.raises(ValueError):
        dense_system(field, target[..., 0])
    with pytest.raises(ValueError):
        dense_system(field, target, max_parameters=True)
    outside = field.detached()
    outside.means[0, 0] = -1
    with pytest.raises(ValueError):
        fit_coupling(outside, target)
    def broken(*args, **kwargs):
        raise torch.linalg.LinAlgError("deliberate singular solve")
    monkeypatch.setattr(torch.linalg, "solve", broken)
    with pytest.raises(torch.linalg.LinAlgError):
        fit_coupling(field, target, replace(ControlConfig(arm="block"), steps=1))
