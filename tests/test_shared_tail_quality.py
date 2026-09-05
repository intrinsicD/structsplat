"""Same-device parity for the opt-in shared quality order statistics."""
from dataclasses import replace

import pytest
import torch

from structsplat import safe_schedule as safe
from structsplat.config import FitConfig
from structsplat.pipeline import PipelineConfig, build_fit_config


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("count", [1, 2, 99, 100, 101, 199, 200, 201, 1003, 65537])
def test_shared_tail_exact_same_device(device, dtype, count):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    generator = torch.Generator(device=device).manual_seed(193)
    values = torch.rand(count * 2, generator=generator, device=device, dtype=dtype)[::2]
    for sample in (values, values.round(), torch.zeros_like(values), values * 1e15):
        expected = safe._tail_error_stats(sample)
        actual = safe._tail_error_stats(sample, "shared")
        for first, second in zip(expected, actual):
            torch.testing.assert_close(first, second, rtol=0, atol=0)


@pytest.mark.parametrize("exceptional", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_preserves_reference(exceptional):
    values = torch.tensor([0., 1., exceptional])
    for first, second in zip(safe._tail_error_stats(values),
                             safe._tail_error_stats(values, "shared")):
        torch.testing.assert_close(first, second, rtol=0, atol=0, equal_nan=True)


def test_large_input_contract_fallback(monkeypatch):
    monkeypatch.setattr(safe, "_QUANTILE_MAX_ELEMENTS", 100)
    for n in (99, 100, 101):
        values = torch.linspace(0, 1, n)
        for first, second in zip(safe._tail_error_stats(values),
                                 safe._tail_error_stats(values, "shared")):
            torch.testing.assert_close(first, second, rtol=0, atol=0)


def test_shared_uses_one_selection_and_no_quantile(monkeypatch):
    original = torch.topk
    calls = []
    def topk(*args, **kwargs):
        calls.append(kwargs["k"])
        return original(*args, **kwargs)
    monkeypatch.setattr(torch, "topk", topk)
    def forbidden(*args, **kwargs):
        raise AssertionError("shared finite path must not sort full quantile")
    monkeypatch.setattr(torch, "quantile", forbidden)
    safe._tail_error_stats(torch.arange(101, dtype=torch.float32), "shared")
    assert calls == [4]


def test_tail_config_and_invalid_inputs():
    assert FitConfig().quality_tail_backend == "reference"
    assert build_fit_config(PipelineConfig(quality_tail_backend="shared"), "cpu").quality_tail_backend == "shared"
    with pytest.raises(ValueError):
        replace(FitConfig(), quality_tail_backend="unknown")
    with pytest.raises(ValueError):
        safe._tail_error_stats(torch.empty(0), "shared")


@pytest.mark.parametrize("guard_p99", [False, True])
def test_shared_complete_gate_reasons(guard_p99):
    from structsplat.fit import _MaskConstraint
    target = torch.rand(29, 31, 3, generator=torch.Generator().manual_seed(71))
    mask = torch.ones(29, 31, dtype=torch.bool)
    cfg = FitConfig()
    constraint = _MaskConstraint.from_mask(mask.numpy(), "cpu", target.dtype,
                                           cfg.sigma_cutoff, cfg.mask_margin)
    den = torch.ones_like(mask, dtype=torch.float32)
    tolerance = safe.CommitTolerances(guard_p99=guard_p99)
    before = safe._quality_from_render(target * 0.9, target, den, mask, constraint, .05, 5)
    for render in (target * .9, target * .95, target * .8):
        reference = safe._quality_from_render(render, target, den, mask, constraint, .05, 5)
        shared = safe._quality_from_render(render, target, den, mask, constraint, .05, 5,
                                          tail_backend="shared")
        assert reference == shared
        assert safe.safe_commit_decision(before, reference, tolerance) == safe.safe_commit_decision(
            before, shared, tolerance)
