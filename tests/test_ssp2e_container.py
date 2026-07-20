from __future__ import annotations

import hashlib
import json
import struct
import zlib

import numpy as np
import pytest

from benchmarks import ssp2e_arithmetic as arithmetic
from benchmarks import ssp2e_container as C


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
    *,
    n: int = 19,
    bits: tuple[int, int, int, int] = (12, 6, 6, 8),
    mean_offset: int = 0,
) -> tuple[bytes, dict[str, np.ndarray]]:
    bm, bs, br, bc = bits
    row = np.arange(n, dtype=np.uint32)
    means = np.column_stack(
        (
            (29 * row + mean_offset) % (1 << bm),
            (47 * row[::-1] + mean_offset) % (1 << bm),
        )
    ).astype(np.uint32)
    scales = np.column_stack((row % (1 << bs), (3 * row + 2) % (1 << bs))).astype(np.uint32)
    rotation = ((5 * row + 1) % (1 << br)).astype(np.uint32)
    colors = np.column_stack(((7 * row) % 256, (11 * row + 1) % 256, (13 * row + 2) % 256))
    colors = colors.astype(np.uint32)
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
    expected = {
        "mean_x": means[:, 0],
        "mean_y": means[:, 1],
        "scale_x": scales[:, 0],
        "scale_y": scales[:, 1],
        "rotation": rotation,
        "red": colors[:, 0],
        "green": colors[:, 1],
        "blue": colors[:, 2],
    }
    return blob, expected


def _frequencies(alphabet: int, location: int, scale: int) -> list[int]:
    assert 0 <= location < 64 and 0 <= scale < 16
    base, remainder = divmod(C.TOTAL_FREQUENCY, alphabet)
    return [base + (symbol < remainder) for symbol in range(alphabet)]


def _arith_encode(symbols: np.ndarray, cdfs: np.ndarray) -> bytes:
    assert symbols.dtype == np.uint32
    assert cdfs.dtype == np.uint32
    assert cdfs.shape == (symbols.size, int(cdfs.shape[1]))
    assert np.all(cdfs[:, 0] == 0)
    assert np.all(cdfs[:, -1] == C.TOTAL_FREQUENCY)
    return symbols.astype(np.uint8).tobytes(order="C")


def _arith_decode(payload: bytes, cdfs: np.ndarray, n: int) -> np.ndarray:
    if len(payload) != n or cdfs.shape[0] != n:
        raise ValueError("raw proof coder length mismatch")
    return np.frombuffer(payload, dtype=np.uint8).astype(np.uint32)


def _head() -> bytes:
    zero = C.HeadChannel((0,) * 8, 0, (0,) * 8, 0)
    return C.serialize_head((zero,) * 5)


def _models() -> dict[str, tuple[int, int]]:
    return {name: (index, index % 16) for index, name in enumerate(C.MODELED_CHANNELS)}


def _repair_outer(blob: bytes) -> bytes:
    changed = bytearray(blob)
    struct.pack_into("<Q", changed, 8, len(changed))
    struct.pack_into("<I", changed, 16, zlib.crc32(changed[C.OUTER.size :]) & 0xFFFFFFFF)
    return bytes(changed)


def _replace_payload(blob: bytes, name: str, payload: bytes) -> bytes:
    parsed = C.parse_container(blob)
    selected = next(entry for entry in parsed.entries if entry.name == name)
    delta = len(payload) - selected.length
    changed = bytearray(
        blob[: selected.offset] + payload + blob[selected.offset + selected.length :]
    )
    for index, entry in enumerate(parsed.entries):
        directory = C.OUTER.size + C.HEADER.size + index * C.DIRECTORY.size
        if entry.name == name:
            struct.pack_into("<Q", changed, directory + 12, len(payload))
        elif entry.offset > selected.offset:
            struct.pack_into("<Q", changed, directory + 4, entry.offset + delta)
    return _repair_outer(bytes(changed))


def test_strict_sspl1_extraction_and_fixed_header_semantic_roundtrip() -> None:
    blob, expected = _sspl1()
    parts = C.extract_sspl1_parts(blob)

    assert parts.total_bytes == len(blob)
    assert parts.header.to_sspl1_header() == parts.original_header
    assert len(parts.header.to_bytes()) == 100
    assert C.ContainerHeader.from_bytes(parts.header.to_bytes()) == parts.header
    assert parts.blob_sha256 == hashlib.sha256(blob).hexdigest()
    for name, values in expected.items():
        np.testing.assert_array_equal(parts.symbols[name], values)

    # Replacing a source member with a valid but noncanonical compression level is rejected by
    # the frozen oracle parser before this module extracts any bytes.
    (header_length,) = struct.unpack_from("<I", blob, 5)
    offset = 9 + header_length
    (length,) = struct.unpack_from("<I", blob, offset)
    raw = zlib.decompress(blob[offset + 4 : offset + 4 + length])
    alternate = zlib.compress(raw, level=1)
    malformed = (
        blob[:offset] + struct.pack("<I", len(alternate)) + alternate + blob[offset + 4 + length :]
    )
    with pytest.raises(C.ContainerError, match="canonical"):
        C.extract_sspl1_parts(malformed)


def test_head_serialization_lsb_features_and_signed_int32_clamping() -> None:
    first = C.HeadChannel((1, 2, 3, 4, 5, 6, 7, 8), -3, (-1,) * 8, 9)
    zero = C.HeadChannel((0,) * 8, 0, (0,) * 8, 0)
    head = C.serialize_head((first, zero, zero, zero, zero))
    assert C.parse_head(head)[0] == first

    grid = bytes(range(256))
    cells = np.asarray([0, 1, 2, 255], dtype=np.uint16)
    rows = C.infer_head_rows(grid, head, cells)
    expected_location = []
    expected_scale = []
    for cell in cells.tolist():
        phi = [2 * ((cell >> bit) & 1) - 1 for bit in range(8)]
        expected_location.append(
            min(63, max(0, -3 + sum(w * p for w, p in zip(first.location_weights, phi))))
        )
        expected_scale.append(
            min(15, max(0, 9 + sum(w * p for w, p in zip(first.scale_weights, phi))))
        )
    np.testing.assert_array_equal(rows["scale_x"][0], expected_location)
    np.testing.assert_array_equal(rows["scale_x"][1], expected_scale)


def test_static_cdf_rows_and_exact_empirical_positive_normalization() -> None:
    calls: list[tuple[int, int, int]] = []

    def provider(alphabet: int, location: int, scale: int) -> list[int]:
        calls.append((alphabet, location, scale))
        return _frequencies(alphabet, location, scale)

    cdfs = C.cdf_rows_from_models(
        64,
        np.asarray([3, 3, 4], dtype=np.uint8),
        np.asarray([2, 2, 1], dtype=np.uint8),
        provider,
    )
    assert cdfs.shape == (3, 65)
    assert cdfs.dtype == np.uint32
    assert calls == [(64, 3, 2), (64, 4, 1)]
    assert np.all(np.diff(cdfs, axis=1) > 0)
    assert np.all(cdfs[:, -1] == 32768)

    symbols = np.asarray([0, 0, 1], dtype=np.uint32)
    frequencies = C.empirical_frequencies(symbols, 64)
    assert np.all(frequencies > 0)
    assert int(frequencies.sum()) == 32768
    assert frequencies[0] > frequencies[1] > frequencies[2]
    # Integer remainder ties favor the ascending symbol.
    equal = C.empirical_frequencies(np.asarray([1, 0], dtype=np.uint32), 64)
    assert equal[0] == equal[1]

    original = np.asarray([8, 8, 1], dtype=np.uint32)
    edited = np.asarray([9, 8, 1], dtype=np.uint32)
    original_counts = np.bincount(original, minlength=64)
    edited_counts = np.bincount(edited, minlength=64)
    original_frequencies = C.empirical_frequencies_from_counts(original_counts, 64)
    edited_frequencies = C.empirical_frequencies_from_counts(edited_counts, 64)
    np.testing.assert_array_equal(original_frequencies, C.empirical_frequencies(original, 64))
    np.testing.assert_array_equal(edited_frequencies, C.empirical_frequencies(edited, 64))
    assert original_frequencies[1] == 10902
    assert edited_frequencies[1] == 10903
    assert original_frequencies[8] == 21804
    assert edited_frequencies[8] == edited_frequencies[9] == 10902

    with pytest.raises(C.ContainerError, match="length 64"):
        C.empirical_frequencies_from_counts(np.ones(63, dtype=np.uint32), 64)
    with pytest.raises(C.ContainerError, match="positive"):
        C.empirical_frequencies_from_counts(np.zeros(64, dtype=np.uint32), 64)
    invalid_counts = np.zeros(64, dtype=np.int64)
    invalid_counts[0] = -1
    with pytest.raises(C.ContainerError, match="negative"):
        C.empirical_frequencies_from_counts(invalid_counts, 64)


def test_vectorized_static_cdf_materialization_matches_all_key_reference_rows() -> None:
    locations = np.repeat(np.arange(64, dtype=np.uint8), 16)
    scales = np.tile(np.arange(16, dtype=np.uint8), 64)
    locations = np.concatenate((locations, locations[::-1], locations[:257]))
    scales = np.concatenate((scales, scales[::-1], scales[:257]))
    calls: list[tuple[int, int]] = []

    def provider(alphabet: int, location: int, scale: int) -> list[int]:
        assert alphabet == 64
        calls.append((location, scale))
        return _frequencies(alphabet, location, scale)

    observed = C.cdf_rows_from_models(64, locations, scales, provider)
    reference = np.asarray(
        [
            np.cumsum((0, *_frequencies(64, int(location), int(scale))), dtype=np.uint32)
            for location, scale in zip(locations, scales, strict=True)
        ],
        dtype=np.uint32,
    )
    np.testing.assert_array_equal(observed, reference)
    assert calls == [(location, scale) for location in range(64) for scale in range(16)]


@pytest.mark.parametrize("magic", [b"SSP2E", b"SSP2S"])
def test_contextual_roundtrip_exact_bytes_symbols_and_accounting(magic: bytes) -> None:
    source, expected = _sspl1()
    grid = bytes(range(256))
    head = _head()
    blob = C.encode_contextual(
        source,
        grid,
        head,
        arithmetic_encode=_arith_encode,
        frequency_provider=_frequencies,
        magic=magic,
    )
    parsed = C.parse_container(blob, reference_sspl1=source)
    decoded = C.decode_container(
        blob,
        arithmetic_decode=_arith_decode,
        arithmetic_encode=_arith_encode,
        frequency_provider=_frequencies,
        reference_sspl1=source,
    )
    source_parts = C.extract_sspl1_parts(source)

    assert parsed.magic == magic
    assert [entry.stream_id for entry in parsed.entries] == list(range(1, 10))
    assert parsed.entries[0].offset == 300
    assert parsed.payloads["grid"] == grid
    assert parsed.payloads["head"] == head
    assert parsed.payloads["means"] == source_parts.payloads["means"]
    assert parsed.payloads["rotation"] == source_parts.payloads["rotation"]
    arithmetic_bytes = sum(len(parsed.payloads[name]) for name in C.MODELED_CHANNELS)
    assert len(blob) == (
        C.CONTEXTUAL_FIXED_BYTES
        + len(source_parts.payloads["means"])
        + len(source_parts.payloads["rotation"])
        + arithmetic_bytes
    )
    assert parsed.prefix_bytes + parsed.payload_bytes == len(blob)
    assert decoded.symbol_sha256 == source_parts.symbol_sha256
    for name, values in expected.items():
        np.testing.assert_array_equal(decoded.symbols[name], values)


def test_factorized_roundtrips_global_and_empirical_models() -> None:
    source, expected = _sspl1(bits=(16, 8, 8, 8))
    global_blob = C.encode_ssp2l(
        source,
        _models(),
        arithmetic_encode=_arith_encode,
        frequency_provider=_frequencies,
    )
    empirical_blob = C.encode_ssp2f(source, arithmetic_encode=_arith_encode)

    for blob, magic in ((global_blob, b"SSP2L"), (empirical_blob, b"SSP2F")):
        parsed = C.parse_container(blob, reference_sspl1=source)
        decoded = C.decode_container(
            blob,
            arithmetic_decode=_arith_decode,
            arithmetic_encode=_arith_encode,
            frequency_provider=_frequencies if magic == b"SSP2L" else None,
            reference_sspl1=source,
        )
        assert parsed.magic == magic
        assert parsed.entries[0].offset == 280
        assert [entry.stream_id for entry in parsed.entries] == [1, 2, 3, 4, 5, 6, 7, 10]
        assert parsed.prefix_bytes + parsed.payload_bytes == len(blob)
        for name, values in expected.items():
            np.testing.assert_array_equal(decoded.symbols[name], values)

    global_parsed = C.parse_container(global_blob)
    empirical_parsed = C.parse_container(empirical_blob)
    assert len(global_parsed.payloads["model"]) == 10
    assert C.parse_global_model(global_parsed.payloads["model"]) == _models()
    frequencies = C.parse_empirical_model(
        empirical_parsed.payloads["model"], empirical_parsed.header
    )
    assert set(frequencies) == set(C.MODELED_CHANNELS)
    assert all(int(values.sum()) == 32768 and np.all(values > 0) for values in frequencies.values())


def test_empirical_frequency_serialization_delegates_byte_exactly() -> None:
    source, _ = _sspl1(bits=(16, 8, 8, 8))
    parts = C.extract_sspl1_parts(source)
    frequencies = {
        name: C.empirical_frequencies(
            parts.symbols[name],
            1 << (parts.header.bits[1] if name.startswith("scale_") else parts.header.bits[3]),
        )
        for name in C.MODELED_CHANNELS
    }
    raw = b"".join(
        frequencies[name].astype("<u2", copy=False).tobytes(order="C")
        for name in C.MODELED_CHANNELS
    )
    expected = zlib.compress(raw, level=9)
    assert C.serialize_empirical_frequencies(frequencies, parts.header) == expected
    assert C.serialize_empirical_model(parts.symbols, parts.header) == expected

    missing = dict(frequencies)
    missing.pop("blue")
    with pytest.raises(C.ContainerError, match="exactly"):
        C.serialize_empirical_frequencies(missing, parts.header)
    wrong_sum = dict(frequencies)
    wrong_sum["red"] = wrong_sum["red"].copy()
    wrong_sum["red"][0] += 1
    with pytest.raises(C.ContainerError, match="sum"):
        C.serialize_empirical_frequencies(wrong_sum, parts.header)


def test_ssp2f_absolute_rgb_override_is_strict_and_byte_canonical() -> None:
    source, expected = _sspl1(bits=(16, 8, 8, 8))
    parts = C.extract_sspl1_parts(source)
    rgb = np.column_stack((expected["red"], expected["green"], expected["blue"])).astype(np.uint32)
    rgb[0] = np.asarray((255, 0, 127), dtype=np.uint32)
    rgb[-1] = np.asarray((3, 5, 7), dtype=np.uint32)

    blob = C.encode_ssp2f_rgb_override(
        source,
        rgb,
        arithmetic_encode=arithmetic.encode,
        arithmetic_decode=arithmetic.decode,
    )
    parsed = C.parse_container(blob, reference_sspl1=source)
    decoded = C.decode_container(
        blob,
        arithmetic_decode=arithmetic.decode,
        arithmetic_encode=arithmetic.encode,
    )
    assert parsed.magic == b"SSP2F"
    assert parsed.header == parts.header
    for name in ("mean_x", "mean_y", "scale_x", "scale_y", "rotation"):
        np.testing.assert_array_equal(decoded.symbols[name], parts.symbols[name])
    for index, name in enumerate(("red", "green", "blue")):
        np.testing.assert_array_equal(decoded.symbols[name], rgb[:, index])
    assert (
        C.encode_ssp2f_rgb_override(
            source,
            rgb,
            arithmetic_encode=arithmetic.encode,
            arithmetic_decode=arithmetic.decode,
        )
        == blob
    )

    with pytest.raises(C.ContainerError, match="RGB override"):
        C.encode_ssp2f_rgb_override(
            source,
            rgb[:, :2],
            arithmetic_encode=arithmetic.encode,
            arithmetic_decode=arithmetic.decode,
        )
    invalid = rgb.copy()
    invalid[0, 0] = 256
    with pytest.raises(C.ContainerError, match="inside"):
        C.encode_ssp2f_rgb_override(
            source,
            invalid,
            arithmetic_encode=arithmetic.encode,
            arithmetic_decode=arithmetic.decode,
        )


@pytest.mark.parametrize("bits", [(12, 6, 6, 8), (16, 8, 8, 8)])
def test_real_static_asset_and_arithmetic_adapter_boundary(
    bits: tuple[int, int, int, int],
) -> None:
    from benchmarks import ssp2e_arithmetic as arithmetic
    from benchmarks import ssp2e_assets as assets

    source, _ = _sspl1(bits=bits)
    tables = assets.load_tables()

    def provider(alphabet: int, location: int, scale: int) -> tuple[int, ...]:
        return tables[alphabet].frequency_row(location, scale)

    blob = C.encode_contextual(
        source,
        bytes(range(256)),
        _head(),
        arithmetic_encode=arithmetic.encode,
        frequency_provider=provider,
    )
    decoded = C.decode_container(
        blob,
        arithmetic_decode=arithmetic.decode,
        arithmetic_encode=arithmetic.encode,
        frequency_provider=provider,
        reference_sspl1=source,
    )
    assert decoded.symbol_sha256 == C.extract_sspl1_parts(source).symbol_sha256


def test_outer_crc_total_directory_suffix_and_fixed_lengths_fail_closed() -> None:
    source, _ = _sspl1()
    valid = C.encode_contextual(
        source,
        bytes(range(256)),
        _head(),
        arithmetic_encode=_arith_encode,
        frequency_provider=_frequencies,
    )

    corrupted = bytearray(valid)
    corrupted[-1] ^= 1
    with pytest.raises(C.ContainerError, match="CRC"):
        C.parse_container(bytes(corrupted))
    with pytest.raises(C.ContainerError, match="total length"):
        C.parse_container(valid + b"x")
    with pytest.raises(C.ContainerError, match="suffix"):
        C.parse_container(_repair_outer(valid + b"x"))

    wrong_id = bytearray(valid)
    struct.pack_into("<I", wrong_id, C.OUTER.size + C.HEADER.size, 2)
    with pytest.raises(C.ContainerError, match="ID/order"):
        C.parse_container(_repair_outer(bytes(wrong_id)))

    wrong_offset = bytearray(valid)
    struct.pack_into("<Q", wrong_offset, C.OUTER.size + C.HEADER.size + 4, 301)
    with pytest.raises(C.ContainerError, match="gap, overlap, alias"):
        C.parse_container(_repair_outer(bytes(wrong_offset)))

    empty_arithmetic = bytearray(valid)
    scale_x_directory = C.OUTER.size + C.HEADER.size + C.DIRECTORY.size
    struct.pack_into("<Q", empty_arithmetic, scale_x_directory + 12, 0)
    with pytest.raises(C.ContainerError, match="empty scale_x"):
        C.parse_container(_repair_outer(bytes(empty_arithmetic)))

    short_grid = _replace_payload(valid, "grid", C.parse_container(valid).payloads["grid"][:-1])
    with pytest.raises(C.ContainerError, match="grid"):
        C.parse_container(short_grid)
    short_head = _replace_payload(valid, "head", C.parse_container(valid).payloads["head"][:-1])
    with pytest.raises(C.ContainerError, match="head"):
        C.parse_container(short_head)


def test_header_source_identity_and_canonical_zlib_models_fail_closed() -> None:
    source, _ = _sspl1()
    contextual = C.encode_contextual(
        source,
        bytes(256),
        _head(),
        arithmetic_encode=_arith_encode,
        frequency_provider=_frequencies,
    )

    wrong_header = bytearray(contextual)
    values = list(C.HEADER.unpack_from(wrong_header, C.OUTER.size))
    values[26] = 2  # cdf_table_id
    wrong_header[C.OUTER.size : C.OUTER.size + C.HEADER.size] = C.HEADER.pack(*values)
    with pytest.raises(C.ContainerError, match="cdf_table_id"):
        C.parse_container(_repair_outer(bytes(wrong_header)))

    other_source, _ = _sspl1(mean_offset=1)
    with pytest.raises(C.ContainerError, match="means payload"):
        C.parse_container(contextual, reference_sspl1=other_source)

    empirical = C.encode_ssp2f(source, arithmetic_encode=_arith_encode)
    model = C.parse_container(empirical).payloads["model"]
    raw = zlib.decompress(model)
    noncanonical = _replace_payload(empirical, "model", zlib.compress(raw, level=1))
    with pytest.raises(C.ContainerError, match="canonical"):
        C.parse_container(noncanonical)

    changed_raw = bytearray(raw)
    first = struct.unpack_from("<H", changed_raw)[0]
    struct.pack_into("<H", changed_raw, 0, first + 1)
    wrong_sum = _replace_payload(empirical, "model", zlib.compress(bytes(changed_raw), level=9))
    with pytest.raises(C.ContainerError, match="sum"):
        C.parse_container(wrong_sum)

    # A positive, valid-sum, canonical zlib model can still be inconsistent with the symbols.
    # Parsing accepts its syntax; cold decoding must reject the semantic model substitution.
    changed_raw = bytearray(raw)
    frequencies = np.frombuffer(changed_raw, dtype="<u2")
    donor = int(np.flatnonzero(frequencies > 1)[0])
    recipient = 0 if donor != 0 else 1
    donor_value = struct.unpack_from("<H", changed_raw, 2 * donor)[0]
    recipient_value = struct.unpack_from("<H", changed_raw, 2 * recipient)[0]
    struct.pack_into("<H", changed_raw, 2 * donor, donor_value - 1)
    struct.pack_into("<H", changed_raw, 2 * recipient, recipient_value + 1)
    substituted = _replace_payload(empirical, "model", zlib.compress(bytes(changed_raw), level=9))
    C.parse_container(substituted)
    with pytest.raises(C.ContainerError, match="canonically describe decoded symbols"):
        C.decode_container(
            substituted,
            arithmetic_decode=_arith_decode,
            arithmetic_encode=_arith_encode,
        )

    global_blob = C.encode_ssp2l(
        source,
        _models(),
        arithmetic_encode=_arith_encode,
        frequency_provider=_frequencies,
    )
    bad_global = bytearray(C.parse_container(global_blob).payloads["model"])
    bad_global[0] = 64
    with pytest.raises(C.ContainerError, match="out of range"):
        C.parse_container(_replace_payload(global_blob, "model", bytes(bad_global)))


def test_decode_rejects_noncanonical_arithmetic_and_reference_symbol_drift() -> None:
    source, _ = _sspl1()
    valid = C.encode_contextual(
        source,
        bytes(256),
        _head(),
        arithmetic_encode=_arith_encode,
        frequency_provider=_frequencies,
    )

    def alternate_encoder(symbols: np.ndarray, cdfs: np.ndarray) -> bytes:
        return _arith_encode(symbols, cdfs) + b"\0"

    with pytest.raises(C.ContainerError, match="noncanonical"):
        C.decode_container(
            valid,
            arithmetic_decode=_arith_decode,
            arithmetic_encode=alternate_encoder,
            frequency_provider=_frequencies,
        )

    scale_x = bytearray(C.parse_container(valid).payloads["scale_x"])
    scale_x[0] ^= 1
    changed = _replace_payload(valid, "scale_x", bytes(scale_x))
    with pytest.raises(C.ContainerError, match="reference SSPL1"):
        C.decode_container(
            changed,
            arithmetic_decode=_arith_decode,
            arithmetic_encode=_arith_encode,
            frequency_provider=_frequencies,
            reference_sspl1=source,
        )


def test_frozen_shuffle_permutation_hash() -> None:
    permutation = C.shuffled_permutation(8192)
    assert np.array_equal(np.sort(permutation), np.arange(8192, dtype=np.uint32))
    serialized = permutation.astype("<u2").tobytes(order="C")
    assert hashlib.sha256(serialized).hexdigest() == (
        "63691f3fdda3cbf83d1bc7b954785dc5467b11433bc4fbbe84fd5ec2d94c05f2"
    )
