from __future__ import annotations

import json
import struct
import zlib
from pathlib import Path

import numpy as np
import pytest

from benchmarks import ssp2e_arithmetic as arithmetic
from benchmarks import ssp2e_container as C
from benchmarks import ssp2f_size_oracle as O


def _pack(values: np.ndarray, bits: int) -> bytes:
    dtype = np.dtype("u1" if bits <= 8 else "<u2" if bits <= 16 else "<u4")
    return np.asarray(values, dtype=dtype).tobytes(order="C")


def _planar(values: np.ndarray, bits: int) -> bytes:
    dtype = np.dtype("u1" if bits <= 8 else "<u2" if bits <= 16 else "<u4")
    raw = np.asarray(values, dtype=dtype).tobytes(order="C")
    if dtype.itemsize == 1:
        return raw
    return np.frombuffer(raw, dtype=np.uint8).reshape(-1, dtype.itemsize).T.ravel().tobytes()


def _deltas(values: np.ndarray, bits: int) -> np.ndarray:
    preceding = np.vstack((np.zeros((1, values.shape[1]), dtype=np.uint32), values[:-1]))
    return (values - preceding) & ((1 << bits) - 1)


def _sspl1(
    rgb: np.ndarray,
    *,
    bits: tuple[int, int, int, int] = (12, 6, 6, 8),
) -> bytes:
    colors = np.asarray(rgb, dtype=np.uint32)
    if colors.ndim != 2 or colors.shape[1] != 3 or np.any(colors >= 256):
        raise ValueError("test RGB must be uint8-representable [N,3]")
    n = int(colors.shape[0])
    bm, bs, br, bc = bits
    row = np.arange(n, dtype=np.uint32)
    means = np.column_stack(
        (
            (29 * row + 3) % (1 << bm),
            (47 * row[::-1] + 5) % (1 << bm),
        )
    ).astype(np.uint32)
    scales = np.column_stack((row % (1 << bs), (3 * row + 2) % (1 << bs))).astype(np.uint32)
    rotation = ((5 * row + 1) % (1 << br)).astype(np.uint32)
    raws = (
        _planar(_deltas(means, bm).T.ravel(), bm),
        _pack(scales.T.ravel(), bs),
        _pack(rotation, br),
        _pack(_deltas(colors, bc).T.ravel(), bc),
    )
    header = {
        "n": n,
        "H": 64,
        "W": 80,
        "bits": list(bits),
        "morton": True,
        "color_delta": True,
        "means_byte_planar": True,
        "rot_circular": True,
        "stream_codec": "zlib",
        "stream_raw_lengths": [len(raw) for raw in raws],
        "color_lo": [-1.0, -0.5, 0.0],
        "color_hi": [1.0, 1.5, 2.0],
        "scale_lo": [-3.0, -2.0],
        "scale_hi": [2.0, 3.0],
        "means_lo": [0.0, 0.0],
        "means_hi": [79.0, 63.0],
        "has_opacity": False,
        "bits_opacity": 8,
        "renderer": "cuda",
        "aa_dilation": 0.0,
        "sigma_cutoff": 3.0,
        "support_fade": False,
        "render_chunk": 512,
    }
    header_raw = json.dumps(header).encode("utf-8")
    blob = b"SSPL1" + struct.pack("<I", len(header_raw)) + header_raw
    for raw in raws:
        payload = zlib.compress(raw, level=9)
        blob += struct.pack("<I", len(payload)) + payload
    return blob


@pytest.fixture(scope="module")
def native_coder(tmp_path_factory: pytest.TempPathFactory) -> arithmetic.NativeArithmeticCoder:
    path = Path(tmp_path_factory.mktemp("ssp2f-size-native")) / "libssp2e_range.so"
    return arithmetic.NativeArithmeticCoder(arithmetic.build_native(path))


@pytest.mark.parametrize("bits", ((12, 6, 6, 8), (16, 8, 8, 8)))
def test_oracle_matches_full_override_for_baseline_random_and_sequential_commits(
    native_coder: arithmetic.NativeArithmeticCoder,
    bits: tuple[int, int, int, int],
) -> None:
    rng = np.random.default_rng(sum(bits))
    source_rgb = rng.integers(0, 256, size=(96, 3), dtype=np.uint32)
    source = _sspl1(source_rgb, bits=bits)
    oracle = O.SSP2FSizeOracle(source, native_coder)

    baseline_blob = oracle.encode_and_verify()
    assert oracle.complete_bytes == len(baseline_blob)
    assert oracle.byte_breakdown["prefix"] == 280
    assert sum(oracle.byte_breakdown.values()) == len(baseline_blob)

    for step in range(12):
        row = int(rng.integers(0, len(source_rgb)))
        channel_index = step % 3
        channel = O.RGB_CHANNELS[channel_index]
        current_rgb = oracle.rgb_symbols
        new_symbol = (int(current_rgb[row, channel_index]) + step + 1) % 256
        old_epoch = oracle.epoch
        old_digest = oracle.state_sha256
        proposal = oracle.propose(row, channel, new_symbol)
        assert proposal.epoch == old_epoch
        assert proposal.state_sha256 == old_digest
        assert oracle.epoch == old_epoch
        assert oracle.state_sha256 == old_digest

        candidate_rgb = current_rgb.copy()
        candidate_rgb[row, channel_index] = new_symbol
        full_blob = C.encode_ssp2f_rgb_override(
            source,
            candidate_rgb,
            arithmetic_encode=native_coder.encode,
            arithmetic_decode=native_coder.decode,
        )
        assert proposal.complete_bytes == len(full_blob)
        assert oracle.commit(proposal) == len(full_blob)
        assert oracle.epoch == old_epoch + 1
        assert oracle.state_sha256 != old_digest
        assert oracle.encode_and_verify() == full_blob

    final_blob = oracle.encode_and_verify()
    parsed = C.parse_container(final_blob, reference_sspl1=source)
    decoded = C.decode_container(
        final_blob,
        arithmetic_decode=native_coder.decode,
        arithmetic_encode=native_coder.encode,
    )
    source_parts = C.extract_sspl1_parts(source)
    assert parsed.header == source_parts.header
    for name in O.NON_RGB_CHANNELS:
        np.testing.assert_array_equal(decoded.symbols[name], source_parts.symbols[name])
    for index, name in enumerate(O.RGB_CHANNELS):
        np.testing.assert_array_equal(decoded.symbols[name], oracle.rgb_symbols[:, index])
    assert (
        C.encode_ssp2f_rgb_override(
            source,
            oracle.rgb_symbols,
            arithmetic_encode=native_coder.encode,
            arithmetic_decode=native_coder.decode,
        )
        == final_blob
    )


def test_histogram_preserving_swap_keeps_model_but_changes_finite_arithmetic_size(
    native_coder: arithmetic.NativeArithmeticCoder,
) -> None:
    rng = np.random.default_rng(18)
    red = rng.integers(0, 256, size=64, dtype=np.uint32)
    assert (int(red[0]), int(red[34])) == (228, 38)
    rgb = np.column_stack(
        (
            red,
            (17 * np.arange(64, dtype=np.uint32) + 3) % 256,
            (29 * np.arange(64, dtype=np.uint32) + 5) % 256,
        )
    )
    source = _sspl1(rgb)
    baseline = O.SSP2FSizeOracle(source, native_coder)
    swapped_rgb = rgb.copy()
    swapped_rgb[[0, 34], 0] = swapped_rgb[[34, 0], 0]
    swapped = O.SSP2FSizeOracle(source, native_coder, rgb_symbols=swapped_rgb)

    np.testing.assert_array_equal(baseline.frequencies["red"], swapped.frequencies["red"])
    assert baseline.model_payload_sha256 == swapped.model_payload_sha256
    assert baseline.arithmetic_bytes["red"] == 47
    assert swapped.arithmetic_bytes["red"] == 46
    for name in ("scale_x", "scale_y", "green", "blue"):
        assert baseline.arithmetic_bytes[name] == swapped.arithmetic_bytes[name]
    assert baseline.complete_bytes == swapped.complete_bytes + 1
    assert len(baseline.encode_and_verify()) == baseline.complete_bytes
    assert len(swapped.encode_and_verify()) == swapped.complete_bytes


def test_stale_proposal_fails_after_an_intervening_commit(
    native_coder: arithmetic.NativeArithmeticCoder,
) -> None:
    rgb = np.column_stack(
        (
            np.arange(32, dtype=np.uint32),
            2 * np.arange(32, dtype=np.uint32),
            3 * np.arange(32, dtype=np.uint32),
        )
    )
    oracle = O.SSP2FSizeOracle(_sspl1(rgb), native_coder)
    first = oracle.propose(0, "red", 7)
    stale = oracle.propose(1, "green", 9)
    oracle.commit(first)
    with pytest.raises(O.SSP2FSizeOracleError, match="stale"):
        oracle.commit(stale)
    assert len(oracle.encode_and_verify()) == oracle.complete_bytes
