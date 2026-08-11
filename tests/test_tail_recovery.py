from __future__ import annotations

from dataclasses import replace
import struct
import subprocess
import sys

import numpy as np
import pytest

from structsplat.config import FitConfig
from structsplat.gaussians import GaussianField
from structsplat.tail_recovery import (
    ConfidenceTailConfig,
    ConfidenceTailResult,
    SparseTailPayload,
    apply_sparse_tail_payload,
    render_confidence_gated_self_prior,
    render_sparse_tail_payload,
    select_pixel_safe_tail,
)


torch = pytest.importorskip("torch")


def _field(device: str = "cpu") -> GaussianField:
    return GaussianField.from_numpy(
        means=np.asarray([[0.0, 0.0]], dtype=np.float32),
        scales=np.asarray([[1.0, 1.0]], dtype=np.float32),
        angles=np.zeros(1, dtype=np.float32),
        colors=np.asarray([[0.8, 0.4, 0.2]], dtype=np.float32),
        device=device,
    )


@pytest.mark.parametrize("value", [1.0, 0.5, 0.0, float("inf"), float("nan")])
def test_config_rejects_nonexpanding_scale(value: float) -> None:
    with pytest.raises(ValueError):
        ConfidenceTailConfig(scale_multiplier=value)


@pytest.mark.parametrize("value", [0.0, -1e-8, float("inf"), float("nan")])
def test_config_rejects_invalid_explicit_threshold(value: float) -> None:
    with pytest.raises(ValueError, match="coverage_threshold"):
        ConfidenceTailConfig(coverage_threshold=value)


def test_module_import_keeps_torch_lazy() -> None:
    command = (
        "import sys; sys.modules['torch'] = None; "
        "import structsplat.tail_recovery; print('ok')"
    )
    result = subprocess.run(
        [sys.executable, "-c", command],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


def test_zero_support_is_filled_and_supported_pixels_are_bit_exact() -> None:
    field = _field()
    cfg = FitConfig(
        renderer="normalized",
        render_chunk=1,
        sigma_cutoff=3.0,
        normalization_eps=1e-8,
    )
    result = render_confidence_gated_self_prior(field, cfg, 1, 5)

    assert result.activation_count == 1
    assert result.denominator[0, 4] == 0.0
    assert torch.count_nonzero(result.baseline[0, 4]) == 0
    assert torch.all(result.prior[0, 4] > 0.0)
    assert torch.equal(result.candidate[0, 4], result.prior[0, 4])
    assert torch.equal(
        result.candidate[~result.activation_mask],
        result.baseline[~result.activation_mask],
    )
    assert result.outside_identity_max_abs == 0.0
    assert torch.isfinite(result.candidate).all()


def test_candidate_matches_missing_mass_formula_without_mutating_field() -> None:
    field = _field()
    before = field.detached()
    cfg = FitConfig(
        renderer="normalized",
        render_chunk=1,
        sigma_cutoff=4.0,
        normalization_eps=1e-3,
    )
    result = render_confidence_gated_self_prior(
        field,
        cfg,
        1,
        5,
        ConfidenceTailConfig(scale_multiplier=2.0, coverage_threshold=0.1),
    )

    recovered = result.baseline + result.missing_mass.unsqueeze(-1) * result.prior
    expected = torch.where(
        result.activation_mask.unsqueeze(-1), recovered, result.baseline
    )
    assert torch.equal(result.candidate, expected)
    assert torch.equal(result.recovered, recovered)
    assert result.coverage_threshold == 0.1
    assert result.normalization_eps == 1e-3
    assert result.scale_multiplier == 2.0
    assert field.n == before.n == 1
    for current, original in (
        (field.means, before.means),
        (field.log_scales, before.log_scales),
        (field.rotations, before.rotations),
        (field.colors, before.colors),
    ):
        assert torch.equal(current, original)


def test_additive_renderer_and_invalid_shapes_fail_closed() -> None:
    field = _field()
    with pytest.raises(ValueError, match="requires a normalized renderer"):
        render_confidence_gated_self_prior(
            field, FitConfig(renderer="additive"), 3, 3
        )
    with pytest.raises(ValueError, match="height"):
        render_confidence_gated_self_prior(field, FitConfig(), 0, 3)
    with pytest.raises(ValueError, match="width"):
        render_confidence_gated_self_prior(field, FitConfig(), 3, True)
    with pytest.raises(TypeError, match="ConfidenceTailConfig"):
        render_confidence_gated_self_prior(  # type: ignore[arg-type]
            field, FitConfig(), 3, 3, object()
        )


def test_repeated_cpu_render_is_exact() -> None:
    field = _field()
    cfg = FitConfig(renderer="normalized", render_chunk=1, sigma_cutoff=3.0)
    first = render_confidence_gated_self_prior(field, cfg, 2, 5)
    second = render_confidence_gated_self_prior(field, cfg, 2, 5)

    assert torch.equal(second.denominator, first.denominator)
    assert torch.equal(second.activation_mask, first.activation_mask)
    assert torch.equal(second.prior, first.prior)
    assert torch.equal(second.candidate, first.candidate)


def test_cuda_candidate_matches_reference_when_available() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    rng = np.random.default_rng(19019)
    means = np.stack([rng.uniform(1, 7, 4), rng.uniform(1, 6, 4)], axis=1)
    scales = rng.uniform(0.7, 1.4, (4, 2))
    angles = rng.uniform(0.0, np.pi, 4)
    colors = rng.uniform(0.1, 0.9, (4, 3))
    cpu_field = GaussianField.from_numpy(means, scales, angles, colors)
    cuda_field = GaussianField.from_numpy(means, scales, angles, colors, device="cuda")
    reference_cfg = FitConfig(
        renderer="normalized", render_chunk=2, sigma_cutoff=4.0
    )
    cuda_cfg = replace(reference_cfg, renderer="cuda")

    reference = render_confidence_gated_self_prior(cpu_field, reference_cfg, 8, 9)
    try:
        candidate = render_confidence_gated_self_prior(cuda_field, cuda_cfg, 8, 9)
        torch.cuda.synchronize()
    except RuntimeError as exc:
        pytest.skip(str(exc))

    assert torch.equal(candidate.activation_mask.cpu(), reference.activation_mask)
    assert torch.allclose(
        candidate.denominator.cpu(), reference.denominator, atol=2e-5, rtol=2e-5
    )
    assert torch.allclose(
        candidate.candidate.cpu(), reference.candidate, atol=2e-5, rtol=2e-5
    )


def _synthetic_tail(device: str = "cpu") -> tuple[ConfidenceTailResult, torch.Tensor]:
    baseline = torch.tensor(
        [[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.1000] * 3, [0.0] * 3]],
        dtype=torch.float32,
        device=device,
    )
    recovered = torch.tensor(
        [[[0.2, 0.2, 0.2], [0.8, 0.8, 0.8], [0.1001] * 3, [0.3] * 3]],
        dtype=torch.float32,
        device=device,
    )
    target = torch.tensor(
        [[[0.2, 0.2, 0.2], [0.2, 0.2, 0.2], [0.1001] * 3, [0.3] * 3]],
        dtype=torch.float32,
        device=device,
    )
    active = torch.tensor([[True, True, True, False]], device=device)
    result = ConfidenceTailResult(
        baseline=baseline,
        prior=torch.zeros_like(baseline),
        recovered=recovered,
        candidate=torch.where(active.unsqueeze(-1), recovered, baseline),
        denominator=torch.zeros((1, 4), dtype=torch.float32, device=device),
        missing_mass=torch.ones((1, 4), dtype=torch.float32, device=device),
        activation_mask=active,
        scale_multiplier=2.0,
        coverage_threshold=1e-8,
        normalization_eps=1e-8,
    )
    return result, target


def test_sst1_round_trip_is_canonical_and_exactly_accounted() -> None:
    payload = SparseTailPayload(height=3, width=4, flat_indices=(0, 5, 11))
    encoded = payload.to_bytes()

    assert encoded[:4] == b"SST1"
    assert len(encoded) == payload.encoded_size == 16 + 4 * payload.count
    assert SparseTailPayload.from_bytes(encoded) == payload
    assert SparseTailPayload.from_bytes(memoryview(encoded)).to_bytes() == encoded


@pytest.mark.parametrize(
    "payload,match",
    [
        (b"", "shorter"),
        (struct.pack("<4sIII", b"BAD!", 1, 1, 0), "magic"),
        (struct.pack("<4sIII", b"SST1", 0, 1, 0), "height"),
        (struct.pack("<4sIII", b"SST1", 1, 1, 1), "length"),
        (struct.pack("<4sIII", b"SST1", 1, 1, 0) + b"x", "length"),
        (struct.pack("<4sIIIII", b"SST1", 2, 2, 2, 1, 1), "increasing"),
        (struct.pack("<4sIIIII", b"SST1", 2, 2, 2, 2, 1), "increasing"),
        (struct.pack("<4sIIII", b"SST1", 2, 2, 1, 4), "outside"),
    ],
)
def test_sst1_parser_rejects_hostile_payloads(payload: bytes, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        SparseTailPayload.from_bytes(payload)


def test_sst1_constructor_rejects_bad_types_and_oversubscribed_raster() -> None:
    with pytest.raises(TypeError, match="height"):
        SparseTailPayload(True, 2)
    with pytest.raises(TypeError, match="tuple"):
        SparseTailPayload(2, 2, [1])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="raster size"):
        SparseTailPayload(1, 1, (0, 0))


def test_pixel_safe_selection_certifies_each_site_and_exact_payload_replay() -> None:
    tail, target = _synthetic_tail()
    baseline_before = tail.baseline.clone()
    recovered_before = tail.recovered.clone()

    selected = select_pixel_safe_tail(tail, target)
    decoded_payload = SparseTailPayload.from_bytes(selected.payload.to_bytes())
    replay = apply_sparse_tail_payload(tail, decoded_payload)

    assert selected.payload.flat_indices == (0, 2)
    assert selected.selected_count == 2
    assert torch.equal(replay, selected.candidate)
    assert selected.outside_identity_max_abs(tail.baseline) == 0.0
    assert torch.equal(selected.candidate[~selected.selected_mask], tail.baseline[~selected.selected_mask])
    assert torch.all(selected.raw_improvement_mask[selected.selected_mask])
    assert torch.all(selected.display_nonregression_mask[selected.selected_mask])
    baseline_sse = torch.square(tail.baseline - target).sum(dim=-1)
    candidate_sse = torch.square(selected.candidate - target).sum(dim=-1)
    assert torch.all(candidate_sse[selected.selected_mask] < baseline_sse[selected.selected_mask])
    quantize = lambda value: torch.round(torch.clamp(value, 0.0, 1.0) * 255.0)
    baseline_display_sse = torch.square(quantize(tail.baseline) - quantize(target)).sum(dim=-1)
    candidate_display_sse = torch.square(quantize(selected.candidate) - quantize(target)).sum(dim=-1)
    assert torch.all(
        candidate_display_sse[selected.selected_mask]
        <= baseline_display_sse[selected.selected_mask]
    )
    assert torch.equal(tail.baseline, baseline_before)
    assert torch.equal(tail.recovered, recovered_before)


def test_payload_indices_are_authoritative_and_dimensions_fail_closed() -> None:
    tail, _ = _synthetic_tail()
    payload = SparseTailPayload(1, 4, (3,))
    replay = apply_sparse_tail_payload(tail, payload)

    assert not bool(tail.activation_mask[0, 3])
    assert torch.equal(replay[0, 3], tail.recovered[0, 3])
    assert torch.equal(replay[0, :3], tail.baseline[0, :3])
    with pytest.raises(ValueError, match="dimensions"):
        apply_sparse_tail_payload(tail, SparseTailPayload(2, 2, ()))


def test_sparse_decoder_matches_full_tail_and_empty_payload_is_ordinary() -> None:
    field = _field()
    cfg = FitConfig(
        renderer="normalized",
        render_chunk=1,
        sigma_cutoff=3.0,
        normalization_eps=1e-8,
    )
    full = render_confidence_gated_self_prior(field, cfg, 1, 5)
    payload = SparseTailPayload(1, 5, (4,))
    expected = apply_sparse_tail_payload(full, payload)
    before = field.detached()

    sparse = render_sparse_tail_payload(field, cfg, 1, 5, payload)
    ordinary = render_sparse_tail_payload(field, cfg, 1, 5, SparseTailPayload(1, 5))

    assert torch.allclose(sparse, expected, atol=2e-5, rtol=2e-5)
    assert torch.equal(ordinary, full.baseline)
    assert torch.equal(sparse[0, :4], ordinary[0, :4])
    assert field.n == before.n == 1
    for current, original in (
        (field.means, before.means),
        (field.log_scales, before.log_scales),
        (field.rotations, before.rotations),
        (field.colors, before.colors),
    ):
        assert torch.equal(current, original)


def test_sparse_decoder_rejects_wrong_renderer_and_dimensions() -> None:
    field = _field()
    with pytest.raises(ValueError, match="normalized renderer"):
        render_sparse_tail_payload(
            field, FitConfig(renderer="additive"), 1, 5, SparseTailPayload(1, 5)
        )
    with pytest.raises(ValueError, match="dimensions"):
        render_sparse_tail_payload(field, FitConfig(), 1, 5, SparseTailPayload(5, 1))


def test_pixel_safe_selection_rejects_invalid_target() -> None:
    tail, target = _synthetic_tail()
    with pytest.raises(ValueError, match="shape"):
        select_pixel_safe_tail(tail, target[:, :2])
    invalid = target.clone()
    invalid[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        select_pixel_safe_tail(tail, invalid)


def test_sparse_selection_cpu_cuda_parity_when_available() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    cpu_tail, cpu_target = _synthetic_tail()
    cuda_tail, cuda_target = _synthetic_tail("cuda")

    cpu = select_pixel_safe_tail(cpu_tail, cpu_target)
    cuda = select_pixel_safe_tail(cuda_tail, cuda_target)
    replay = apply_sparse_tail_payload(cuda_tail, cpu.payload)
    torch.cuda.synchronize()

    assert cuda.payload.to_bytes() == cpu.payload.to_bytes()
    assert torch.equal(cuda.selected_mask.cpu(), cpu.selected_mask)
    assert torch.equal(cuda.candidate.cpu(), cpu.candidate)
    assert torch.equal(replay.cpu(), cpu.candidate)


def test_sparse_decoder_matches_full_cuda_tail_when_available() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    rng = np.random.default_rng(20020)
    means = np.stack([rng.uniform(1, 7, 12), rng.uniform(1, 6, 12)], axis=1)
    scales = rng.uniform(0.7, 1.4, (12, 2))
    angles = rng.uniform(0.0, np.pi, 12)
    colors = rng.uniform(0.1, 0.9, (12, 3))
    field = GaussianField.from_numpy(means, scales, angles, colors, device="cuda")
    cfg = FitConfig(renderer="cuda", render_chunk=2, sigma_cutoff=4.0)
    payload = SparseTailPayload(8, 9, (0, 7, 17, 40, 71))

    try:
        full = render_confidence_gated_self_prior(field, cfg, 8, 9)
        expected = apply_sparse_tail_payload(full, payload)
        sparse = render_sparse_tail_payload(field, cfg, 8, 9, payload)
        torch.cuda.synchronize()
    except RuntimeError as exc:
        pytest.skip(str(exc))

    assert torch.allclose(sparse, expected, atol=2e-5, rtol=2e-5)
    outside = torch.ones(72, dtype=torch.bool)
    outside[list(payload.flat_indices)] = False
    assert torch.allclose(
        sparse.reshape(-1, 3).cpu()[outside],
        full.baseline.reshape(-1, 3).cpu()[outside],
        atol=2e-5,
        rtol=2e-5,
    )
