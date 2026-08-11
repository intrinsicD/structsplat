from __future__ import annotations

import struct
import subprocess
import sys

import numpy as np
import pytest

from structsplat.config import FitConfig
from structsplat.gaussians import GaussianField
from structsplat.source_patch_tail import (
    SourcePatchConfig,
    SourcePatchPayload,
    apply_source_patch_payload,
    render_source_patch_payload,
    select_source_patch_tail,
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


def test_module_import_keeps_torch_lazy() -> None:
    command = (
        "import sys; sys.modules['torch'] = None; "
        "import structsplat.source_patch_tail; print('ok')"
    )
    result = subprocess.run(
        [sys.executable, "-c", command], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


@pytest.mark.parametrize("radius", [True, 1.5, -1])
def test_config_rejects_invalid_radius(radius: object) -> None:
    with pytest.raises((TypeError, ValueError), match="radius"):
        SourcePatchConfig(radius=radius)  # type: ignore[arg-type]


def test_spt1_round_trip_and_exact_size() -> None:
    payload = SourcePatchPayload(
        3, 4, flat_indices=(0, 5, 11), rgb8=((1, 2, 3), (4, 5, 6), (253, 254, 255))
    )
    encoded = payload.to_bytes()

    assert encoded[:4] == b"SPT1"
    assert len(encoded) == payload.encoded_size == 16 + 7 * 3
    assert SourcePatchPayload.from_bytes(encoded) == payload
    assert SourcePatchPayload.from_bytes(memoryview(encoded)).to_bytes() == encoded


@pytest.mark.parametrize(
    "payload,match",
    [
        (b"", "shorter"),
        (struct.pack("<4sIII", b"BAD!", 1, 1, 0), "magic"),
        (struct.pack("<4sIII", b"SPT1", 0, 1, 0), "height"),
        (struct.pack("<4sIII", b"SPT1", 1, 1, 1), "length"),
        (struct.pack("<4sIII", b"SPT1", 1, 1, 0) + b"x", "length"),
        (
            struct.pack("<4sIIIIBBBIBBB", b"SPT1", 2, 2, 2, 1, 1, 2, 3, 1, 4, 5, 6),
            "increasing",
        ),
        (struct.pack("<4sIIIIBBB", b"SPT1", 2, 2, 1, 4, 1, 2, 3), "outside"),
    ],
)
def test_spt1_rejects_hostile_input(payload: bytes, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        SourcePatchPayload.from_bytes(payload)


def test_spt1_constructor_rejects_mismatched_or_invalid_colors() -> None:
    with pytest.raises(ValueError, match="equal length"):
        SourcePatchPayload(2, 2, (0,), ())
    with pytest.raises(TypeError, match="RGB tuple"):
        SourcePatchPayload(2, 2, (0,), ((1, 2),))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=r"\[0, 255\]"):
        SourcePatchPayload(2, 2, (0,), ((1, 2, 256),))


def test_selection_expands_square_filters_equal_display_and_is_pointwise_safe() -> None:
    baseline = torch.zeros((5, 5, 3), dtype=torch.float32)
    target = torch.full((5, 5, 3), 128.0 / 255.0, dtype=torch.float32)
    target[1, 1] = 0.0  # In the expanded patch but already display-identical.
    denominator = torch.ones((5, 5), dtype=torch.float32)
    denominator[2, 2] = 0.0

    result = select_source_patch_tail(
        baseline,
        denominator,
        target,
        1e-8,
        SourcePatchConfig(radius=1),
    )
    decoded = apply_source_patch_payload(
        baseline, SourcePatchPayload.from_bytes(result.payload.to_bytes())
    )

    assert result.seed_count == 1
    assert result.expanded_count == 9
    assert result.selected_count == 8
    assert not bool(result.selected_mask[1, 1])
    assert result.payload.encoded_size == 16 + 7 * 8
    assert torch.equal(decoded, result.candidate)
    assert torch.equal(result.candidate[result.selected_mask], target[result.selected_mask])
    assert torch.equal(result.candidate[~result.selected_mask], baseline[~result.selected_mask])
    assert result.outside_identity_max_abs(baseline) == 0.0
    assert result.pointwise_raw_sse_delta_max < 0.0
    assert result.pointwise_display_sse_delta_max < 0.0


def test_expansion_clips_at_raster_boundary() -> None:
    baseline = torch.zeros((4, 4, 3))
    target = torch.ones((4, 4, 3))
    denominator = torch.ones((4, 4))
    denominator[0, 0] = 0.0
    result = select_source_patch_tail(
        baseline, denominator, target, 1e-8, SourcePatchConfig(radius=1)
    )

    assert result.expanded_count == result.selected_count == 4
    assert result.payload.flat_indices == (0, 1, 4, 5)


def test_selection_rejects_invalid_inputs() -> None:
    baseline = torch.zeros((2, 2, 3))
    denominator = torch.ones((2, 2))
    target = torch.ones((2, 2, 3))
    with pytest.raises(ValueError, match="target shape"):
        select_source_patch_tail(baseline, denominator, target[:1], 1e-8)
    invalid = denominator.clone()
    invalid[0, 0] = -1.0
    with pytest.raises(ValueError, match="nonnegative"):
        select_source_patch_tail(baseline, invalid, target, 1e-8)
    target[0, 0, 0] = 2.0
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        select_source_patch_tail(baseline, denominator, target, 1e-8)


def test_render_payload_matches_ordinary_plus_apply_without_mutating_field() -> None:
    field = _field()
    before = field.detached()
    cfg = FitConfig(renderer="normalized", render_chunk=1, sigma_cutoff=3.0)
    payload = SourcePatchPayload(1, 5, (4,), ((10, 20, 30),))
    empty = SourcePatchPayload(1, 5)

    ordinary = render_source_patch_payload(field, cfg, 1, 5, empty)
    candidate = render_source_patch_payload(field, cfg, 1, 5, payload)
    expected = apply_source_patch_payload(ordinary, payload)

    assert torch.equal(candidate, expected)
    assert torch.equal(candidate[0, 4], torch.tensor([10, 20, 30]) / 255.0)
    assert field.n == before.n == 1
    for current, original in (
        (field.means, before.means),
        (field.log_scales, before.log_scales),
        (field.rotations, before.rotations),
        (field.colors, before.colors),
    ):
        assert torch.equal(current, original)


def test_render_payload_requires_normalized_renderer_and_matching_dimensions() -> None:
    field = _field()
    with pytest.raises(ValueError, match="normalized renderer"):
        render_source_patch_payload(
            field, FitConfig(renderer="additive"), 1, 5, SourcePatchPayload(1, 5)
        )
    with pytest.raises(ValueError, match="dimensions"):
        render_source_patch_payload(field, FitConfig(), 1, 5, SourcePatchPayload(5, 1))


def test_selection_and_decode_cpu_cuda_parity_when_available() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    baseline = torch.zeros((5, 5, 3), dtype=torch.float32)
    target = torch.full((5, 5, 3), 128.0 / 255.0, dtype=torch.float32)
    denominator = torch.ones((5, 5), dtype=torch.float32)
    denominator[2, 2] = 0.0
    cpu = select_source_patch_tail(
        baseline, denominator, target, 1e-8, SourcePatchConfig(radius=1)
    )
    cuda = select_source_patch_tail(
        baseline.cuda(),
        denominator.cuda(),
        target.cuda(),
        1e-8,
        SourcePatchConfig(radius=1),
    )
    replay = apply_source_patch_payload(baseline.cuda(), cpu.payload)
    torch.cuda.synchronize()

    assert cuda.payload.to_bytes() == cpu.payload.to_bytes()
    assert torch.equal(cuda.selected_mask.cpu(), cpu.selected_mask)
    assert torch.equal(cuda.candidate.cpu(), cpu.candidate)
    assert torch.equal(replay.cpu(), cpu.candidate)
