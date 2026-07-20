from __future__ import annotations

import hashlib
import json
import struct
import zlib
from pathlib import Path

import numpy as np
import pytest

from benchmarks import ssp2e_arithmetic as arithmetic
from benchmarks import ssp2e_container as comp009_container
from benchmarks import ssp2v_container as C


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
    n: int = C.ROW_COUNT,
    bits: tuple[int, int, int, int] = (12, 6, 6, 8),
    mean_offset: int = 0,
    color_offset: int = 0,
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
    colors = np.column_stack(
        (
            (7 * row + color_offset) % (1 << bc),
            (11 * row + 1 + color_offset) % (1 << bc),
            (13 * row + 2 + color_offset) % (1 << bc),
        )
    ).astype(np.uint32)
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
    symbols = {
        "mean_x": means[:, 0],
        "mean_y": means[:, 1],
        "scale_x": scales[:, 0],
        "scale_y": scales[:, 1],
        "rotation": rotation,
        "red": colors[:, 0],
        "green": colors[:, 1],
        "blue": colors[:, 2],
    }
    return blob, symbols


def _descriptor(family: int, matched: int, base_k: int) -> C.SSP2VDescriptor:
    actual = 1 if family == 0 else matched
    entries = (2 * matched - 1) * base_k if family == 0 else matched * base_k
    return C.SSP2VDescriptor(
        family_id=family,
        actual_stage_count=actual,
        matched_rvq_stage_count=matched,
        base_k=base_k,
        codebook_entries=entries,
    )


def _flat_inputs(
    descriptor: C.SSP2VDescriptor,
) -> tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...]]:
    assert descriptor.family_id == 0
    row = np.arange(descriptor.codebook_entries, dtype=np.int64)
    codebook = np.column_stack((row % 256, (3 * row + 17) % 256, (7 * row + 29) % 256))
    indices = ((37 * np.arange(C.ROW_COUNT, dtype=np.uint32) + 11) % len(row)).astype(np.uint32)
    return (codebook.astype(np.uint8),), (indices,)


def _rvq_inputs(
    descriptor: C.SSP2VDescriptor,
) -> tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...]]:
    assert descriptor.family_id == 1
    row = np.arange(descriptor.base_k, dtype=np.int64)
    stage0 = np.column_stack((row % 256, (5 * row + 7) % 256, (9 * row + 3) % 256))
    codebooks: list[np.ndarray] = [stage0.astype(np.uint8)]
    indices: list[np.ndarray] = []
    for stage in range(descriptor.actual_stage_count):
        indices.append(
            (
                (17 + (stage + 2) * 19 * np.arange(C.ROW_COUNT, dtype=np.uint32))
                % descriptor.base_k
            ).astype(np.uint32)
        )
        if stage:
            residual = np.column_stack(
                (
                    ((stage + 1) * row) % 41 - 20,
                    ((stage + 3) * row) % 53 - 26,
                    ((stage + 5) * row) % 61 - 30,
                )
            )
            codebooks.append(residual.astype(np.int16))
    return tuple(codebooks), tuple(indices)


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


@pytest.fixture(scope="module")
def source() -> tuple[bytes, dict[str, np.ndarray]]:
    return _sspl1()


@pytest.fixture(scope="module")
def flat_blob(source: tuple[bytes, dict[str, np.ndarray]]) -> bytes:
    descriptor = _descriptor(0, 2, 64)
    codebooks, indices = _flat_inputs(descriptor)
    return C.encode_ssp2v(source[0], descriptor, codebooks, indices)


@pytest.fixture(scope="module")
def rvq_blob(source: tuple[bytes, dict[str, np.ndarray]]) -> bytes:
    descriptor = _descriptor(1, 3, 64)
    codebooks, indices = _rvq_inputs(descriptor)
    return C.encode_ssp2v(source[0], descriptor, codebooks, indices)


@pytest.fixture(scope="module")
def native_arithmetic(tmp_path_factory: pytest.TempPathFactory) -> arithmetic.NativeArithmeticCoder:
    path = Path(tmp_path_factory.mktemp("ssp2v-native")) / "ssp2e_range_ext.so"
    arithmetic.build_native(path)
    return arithmetic.NativeArithmeticCoder(path)


def test_struct_sizes_and_distinct_header_semantics(
    source: tuple[bytes, dict[str, np.ndarray]],
) -> None:
    assert (C.OUTER.size, C.HEADER.size, C.DIRECTORY.size, C.DESCRIPTOR.size) == (20, 100, 20, 16)
    parts = C.extract_sspl1_parts(source[0])
    legacy = comp009_container.extract_sspl1_parts(source[0]).header

    assert parts.header.aux_model_id == 0
    assert legacy.cdf_table_id == 1
    assert parts.header.to_bytes() != legacy.to_bytes()
    assert C.HEADER.unpack(parts.header.to_bytes())[26] == 0
    assert comp009_container.HEADER.unpack(legacy.to_bytes())[26] == 1
    assert (
        parts.header.to_sspl1_header()
        == comp009_container.extract_sspl1_parts(source[0]).original_header
    )
    assert C.SSP2VHeader.from_bytes(parts.header.to_bytes()) == parts.header


def test_source_n_is_rejected_before_empirical_work() -> None:
    short, _ = _sspl1(n=19)
    with pytest.raises(C.ContainerError, match="requires n=8192"):
        C.extract_sspl1_parts(short)
    with pytest.raises(C.ContainerError, match="length 8192"):
        C.empirical_frequencies(np.arange(19, dtype=np.uint32), 64)


@pytest.mark.parametrize(
    ("family", "matched", "base_k", "actual", "entries", "codebook_bytes", "label"),
    (
        (0, 2, 64, 1, 192, 576, "flat_s2_k64_l192"),
        (0, 2, 128, 1, 384, 1152, "flat_s2_k128_l384"),
        (0, 3, 64, 1, 320, 960, "flat_s3_k64_l320"),
        (0, 3, 128, 1, 640, 1920, "flat_s3_k128_l640"),
        (1, 2, 64, 2, 128, 576, "rvq_s2_k64"),
        (1, 2, 128, 2, 256, 1152, "rvq_s2_k128"),
        (1, 3, 64, 3, 192, 960, "rvq_s3_k64"),
        (1, 3, 128, 3, 384, 1920, "rvq_s3_k128"),
    ),
)
def test_exact_eight_descriptor_menu_and_matched_codebook_bytes(
    family: int,
    matched: int,
    base_k: int,
    actual: int,
    entries: int,
    codebook_bytes: int,
    label: str,
) -> None:
    descriptor = _descriptor(family, matched, base_k)
    assert descriptor.actual_stage_count == actual
    assert descriptor.codebook_entries == entries
    assert descriptor.codebook_bytes == codebook_bytes
    assert descriptor.label == label
    assert C.SSP2VDescriptor.from_bytes(descriptor.to_bytes()) == descriptor
    if family == 0:
        assert descriptor.stage_alphabets == (entries,)
    else:
        assert descriptor.stage_alphabets == (base_k,) * actual


@pytest.mark.parametrize(
    "descriptor",
    (
        C.SSP2VDescriptor(2, 1, 2, 64, 192),
        C.SSP2VDescriptor(0, 2, 2, 64, 192),
        C.SSP2VDescriptor(0, 1, 4, 64, 448),
        C.SSP2VDescriptor(0, 1, 2, 63, 189),
        C.SSP2VDescriptor(0, 1, 2, 64, 191),
        C.SSP2VDescriptor(0, 1, 2, 64, 192, reserved=1),
        C.SSP2VDescriptor(0, 1, 2, 64, 192, descriptor_version=2),
        C.SSP2VDescriptor(0, 1, 2, 64, 192, tag=b"BAD!"),
    ),
)
def test_descriptor_rejects_every_nonmenu_or_reserved_form(
    descriptor: C.SSP2VDescriptor,
) -> None:
    with pytest.raises(C.ContainerError):
        descriptor.validated()


@pytest.mark.parametrize("alphabet", C.LEGAL_INDEX_ALPHABETS)
def test_empirical_formula_python_native_arithmetic_parity_and_certificate(
    alphabet: int,
    native_arithmetic: arithmetic.NativeArithmeticCoder,
) -> None:
    indices = (
        (np.arange(C.ROW_COUNT, dtype=np.uint64) ** 2 + 37 * np.arange(C.ROW_COUNT)) % alphabet
    ).astype(np.uint32)
    observed = C.empirical_frequencies(indices, alphabet)

    counts = [int(np.count_nonzero(indices == value)) for value in range(alphabet)]
    residual = C.TOTAL_FREQUENCY - alphabet
    allocation = [(residual * count) // C.ROW_COUNT for count in counts]
    remainders = [(residual * count) % C.ROW_COUNT for count in counts]
    remaining = residual - sum(allocation)
    order = sorted(range(alphabet), key=lambda value: (-remainders[value], value))
    for value in order[:remaining]:
        allocation[value] += 1
    expected = np.asarray([1 + value for value in allocation], dtype=np.uint32)
    np.testing.assert_array_equal(observed, expected)
    assert np.all(observed > 0)
    assert int(observed.sum()) == C.TOTAL_FREQUENCY

    rows = C.cdf_rows(observed)
    python_payload = arithmetic.encode(indices, rows)
    native_payload = native_arithmetic.encode(indices, rows)
    assert native_payload == python_payload
    np.testing.assert_array_equal(arithmetic.decode(python_payload, rows, C.ROW_COUNT), indices)
    np.testing.assert_array_equal(
        native_arithmetic.decode(python_payload, rows, C.ROW_COUNT), indices
    )
    certificate = arithmetic.coder_certificate(indices, rows, python_payload)
    assert certificate["inequality_holds"] is True
    assert certificate["payload_sha256"] == hashlib.sha256(python_payload).hexdigest()


def test_empirical_remainder_ties_and_invalid_inputs() -> None:
    indices = np.zeros(C.ROW_COUNT, dtype=np.uint32)
    indices[:2] = (1, 2)
    frequencies = C.empirical_frequencies(indices, 64)
    assert frequencies[1] == frequencies[2]
    with pytest.raises(C.ContainerError, match="outside the frozen menu"):
        C.empirical_frequencies(indices, 256)
    invalid = indices.copy()
    invalid[-1] = 64
    with pytest.raises(C.ContainerError, match="exceeds"):
        C.empirical_frequencies(invalid, 64)
    with pytest.raises(C.ContainerError, match="negative"):
        C.empirical_frequencies(np.full(C.ROW_COUNT, -1, dtype=np.int32), 64)


def test_ssp2z_exact_roundtrip_header_payloads_and_accounting(
    source: tuple[bytes, dict[str, np.ndarray]],
) -> None:
    blob = C.encode_ssp2z(source[0])
    parsed = C.parse_container(blob, reference_sspl1=source[0])
    decoded = C.decode_ssp2z(blob, reference_sspl1=source[0])
    source_parts = C.extract_sspl1_parts(source[0])

    assert parsed.magic == b"SSP2Z"
    assert parsed.prefix_bytes == 200
    assert parsed.entries[0].offset == 200
    assert [entry.stream_id for entry in parsed.entries] == [1, 2, 3, 4]
    assert parsed.component_bytes()["total"] == len(blob)
    for name in ("means", "scales", "rotation", "colors"):
        assert parsed.payloads[f"{name}_zlib"] == source_parts.payloads[name]
    assert decoded.symbol_sha256 == source_parts.symbol_sha256
    assert decoded.canonical_blob_sha256 == hashlib.sha256(blob).hexdigest()
    assert decoded.rgb_accounting is not None
    assert decoded.rgb_accounting.rgb_sse == 0
    assert decoded.rgb_accounting.clip_components == 0
    for name, values in source[1].items():
        np.testing.assert_array_equal(decoded.symbols[name], values)


@pytest.mark.parametrize("bits", ((12, 6, 6, 8), (16, 8, 8, 8)))
def test_ssp2z_supports_exactly_both_frozen_bit_tuples(
    bits: tuple[int, int, int, int],
) -> None:
    source, symbols = _sspl1(bits=bits)
    blob = C.encode_ssp2z(source)
    decoded = C.decode_ssp2z(blob, reference_sspl1=source)
    assert decoded.parsed.header.bits == bits
    for name, values in symbols.items():
        np.testing.assert_array_equal(decoded.symbols[name], values)


def test_flat_ssp2v_roundtrip_canonical_model_bytes_and_lossy_accounting(
    source: tuple[bytes, dict[str, np.ndarray]], flat_blob: bytes
) -> None:
    decoded = C.decode_ssp2v(flat_blob, reference_sspl1=source[0])
    parsed = decoded.parsed
    descriptor = _descriptor(0, 2, 64)
    codebooks, indices = _flat_inputs(descriptor)

    assert (
        parsed.header.to_bytes() == C.parse_container(C.encode_ssp2z(source[0])).header.to_bytes()
    )
    assert parsed.descriptor == descriptor
    assert parsed.prefix_bytes == 260
    assert parsed.entries[0].offset == 260
    assert [entry.stream_id for entry in parsed.entries] == list(range(1, 8))
    assert parsed.payloads["descriptor_raw"] == descriptor.to_bytes()
    assert parsed.payloads["codebook_raw"] == C.serialize_codebooks(codebooks, descriptor)
    assert parsed.payloads["index_model_zlib"] == C.serialize_index_model(indices, descriptor)
    assert decoded.canonical_blob_sha256 == hashlib.sha256(flat_blob).hexdigest()
    assert decoded.used_entries_per_stage == (192,)
    assert decoded.unique_index_tuples == 192
    assert len(decoded.coder_certificates) == 1
    assert decoded.coder_certificates[0]["inequality_holds"] is True
    assert decoded.rgb_accounting is not None
    assert decoded.rgb_accounting.clip_components == 0
    assert decoded.rgb_accounting.rgb_sse > 0
    for name in ("mean_x", "mean_y", "scale_x", "scale_y", "rotation"):
        np.testing.assert_array_equal(decoded.symbols[name], source[1][name])

    components = parsed.component_bytes()
    expected = (
        260
        + len(parsed.payloads["means_zlib"])
        + len(parsed.payloads["scales_zlib"])
        + len(parsed.payloads["rotation_zlib"])
        + 16
        + 3 * descriptor.codebook_entries
        + len(parsed.payloads["index_model_zlib"])
        + len(parsed.payloads["index_0_arith"])
    )
    assert components["total"] == expected == len(flat_blob)


def test_three_stage_rvq_roundtrip_prefix_clipping_and_components(
    source: tuple[bytes, dict[str, np.ndarray]], rvq_blob: bytes
) -> None:
    decoded = C.decode_ssp2v(rvq_blob, reference_sspl1=source[0])
    parsed = decoded.parsed
    descriptor = _descriptor(1, 3, 64)
    codebooks, indices = _rvq_inputs(descriptor)

    assert parsed.descriptor == descriptor
    assert parsed.prefix_bytes == 300
    assert parsed.entries[0].offset == 300
    assert [entry.stream_id for entry in parsed.entries] == list(range(1, 10))
    assert len(parsed.payloads["codebook_raw"]) == 15 * descriptor.base_k
    assert decoded.canonical_blob_sha256 == hashlib.sha256(rvq_blob).hexdigest()
    assert len(decoded.indices) == len(decoded.codebooks) == len(decoded.coder_certificates) == 3
    raw, clipped = C.reconstruct_rgb(descriptor, codebooks, indices)
    np.testing.assert_array_equal(decoded.raw_rgb, raw)
    np.testing.assert_array_equal(
        np.column_stack(
            (decoded.symbols["red"], decoded.symbols["green"], decoded.symbols["blue"])
        ),
        clipped,
    )
    assert decoded.rgb_accounting is not None
    assert decoded.rgb_accounting.clip_components == (
        decoded.rgb_accounting.clip_low_components + decoded.rgb_accounting.clip_high_components
    )
    assert parsed.component_bytes()["total"] == len(rvq_blob)


def test_codebook_grammar_int16_bounds_and_explicit_clip_behavior() -> None:
    descriptor = _descriptor(1, 3, 64)
    stage0 = np.zeros((64, 3), dtype=np.uint8)
    stage0[0] = (250, 5, 100)
    stage1 = np.zeros((64, 3), dtype=np.int16)
    stage1[0] = (10, -10, 200)
    stage2 = np.zeros((64, 3), dtype=np.int16)
    stage2[0] = (0, 0, 0)
    indices = (np.zeros(C.ROW_COUNT, dtype=np.uint32),) * 3

    raw_payload = C.serialize_codebooks((stage0, stage1, stage2), descriptor)
    recovered = C.parse_codebooks(raw_payload, descriptor)
    assert recovered[0].dtype == np.uint8
    assert recovered[1].dtype == np.int16
    assert recovered[1].tobytes() == stage1.astype("<i2").tobytes()
    raw, clipped = C.reconstruct_rgb(descriptor, recovered, indices)
    np.testing.assert_array_equal(raw[0], (260, -5, 300))
    np.testing.assert_array_equal(clipped[0], (255, 0, 255))

    out_of_range = stage1.astype(np.int64)
    out_of_range[0, 0] = 32768
    with pytest.raises(C.ContainerError, match="outside"):
        C.serialize_codebooks((stage0, out_of_range, stage2), descriptor)
    with pytest.raises(C.ContainerError, match="length mismatch"):
        C.parse_codebooks(raw_payload[:-1], descriptor)


def test_exact_rgb_accounting_boundaries_repeats_and_distinct_collisions() -> None:
    source = np.asarray(
        (
            (0, 0, 0),
            (255, 255, 255),
            (10, 20, 30),
            (10, 20, 30),
            (40, 50, 60),
        ),
        dtype=np.uint8,
    )
    raw = np.asarray(
        (
            (0, 255, -1),
            (255, 256, 255),
            (9, 21, 30),
            (9, 21, 30),
            (9, 21, 30),
        ),
        dtype=np.int32,
    )
    accounting = C.rgb_accounting(source, raw)
    assert accounting.clip_low_components == 1
    assert accounting.clip_high_components == 1
    assert accounting.clip_components == 2
    assert accounting.clip_rows == 2
    assert accounting.distinct_input_rgb == 4
    assert accounting.distinct_reconstructed_rgb == 3
    assert accounting.merged_input_colors == 1
    assert accounting.collision_pairs == 1
    clipped = np.clip(raw.astype(np.int64), 0, 255)
    errors = clipped - source.astype(np.int64)
    assert accounting.rgb_sse == int(np.square(errors).sum())
    assert accounting.sum_abs_error == int(np.abs(errors).sum())
    assert accounting.sum_signed_error == int(errors.sum())
    assert accounting.max_abs_error == int(np.abs(errors).max())
    with pytest.raises(C.ContainerError, match="signed-integer"):
        C.rgb_accounting(source, raw.astype(np.uint32))
    too_large = raw.astype(np.int64)
    too_large[0, 0] = 1 << 31
    with pytest.raises(C.ContainerError, match="signed-int32"):
        C.rgb_accounting(source, too_large)


def test_outer_crc_total_directory_gap_alias_suffix_and_truncation_fail_closed(
    source: tuple[bytes, dict[str, np.ndarray]],
) -> None:
    valid = C.encode_ssp2z(source[0])
    with pytest.raises(C.ContainerError, match="total length"):
        C.parse_container(valid[:-1])
    with pytest.raises(C.ContainerError, match="total length"):
        C.parse_container(valid + b"x")
    with pytest.raises(C.ContainerError, match="suffix"):
        C.parse_container(_repair_outer(valid + b"x"))

    corrupted = bytearray(valid)
    corrupted[-1] ^= 1
    with pytest.raises(C.ContainerError, match="CRC"):
        C.parse_container(bytes(corrupted))

    wrong_id = bytearray(valid)
    struct.pack_into("<I", wrong_id, C.OUTER.size + C.HEADER.size, 2)
    with pytest.raises(C.ContainerError, match="ID/order"):
        C.parse_container(_repair_outer(bytes(wrong_id)))

    wrong_offset = bytearray(valid)
    struct.pack_into("<Q", wrong_offset, C.OUTER.size + C.HEADER.size + 4, 201)
    with pytest.raises(C.ContainerError, match="gap, overlap, alias"):
        C.parse_container(_repair_outer(bytes(wrong_offset)))

    empty = bytearray(valid)
    struct.pack_into("<Q", empty, C.OUTER.size + C.HEADER.size + 12, 0)
    with pytest.raises(C.ContainerError, match="empty means"):
        C.parse_container(_repair_outer(bytes(empty)))

    wrong_version = bytearray(valid)
    wrong_version[5] = 2
    with pytest.raises(C.ContainerError, match="version"):
        C.parse_container(bytes(wrong_version))
    wrong_count = bytearray(valid)
    wrong_count[7] = 5
    with pytest.raises(C.ContainerError, match="stream count"):
        C.parse_container(bytes(wrong_count))


def test_header_flags_ranges_and_reference_payload_identity_fail_closed(
    source: tuple[bytes, dict[str, np.ndarray]],
) -> None:
    valid = C.encode_ssp2z(source[0])
    for index, value, message in (
        (0, C.ROW_COUNT - 1, "requires n=8192"),
        (24, 2, "renderer_id"),
        (25, 1, "support_fade"),
        (26, 1, "aux_model_id"),
        (27, 8, "zlib_level"),
    ):
        changed = bytearray(valid)
        values = list(C.HEADER.unpack_from(changed, C.OUTER.size))
        values[index] = value
        changed[C.OUTER.size : C.OUTER.size + C.HEADER.size] = C.HEADER.pack(*values)
        with pytest.raises(C.ContainerError, match=message):
            C.parse_container(_repair_outer(bytes(changed)))

    values = list(C.HEADER.unpack_from(valid, C.OUTER.size))
    values[3:7] = (12, 7, 6, 8)
    changed = bytearray(valid)
    changed[C.OUTER.size : C.OUTER.size + C.HEADER.size] = C.HEADER.pack(*values)
    with pytest.raises(C.ContainerError, match="bit tuple"):
        C.parse_container(_repair_outer(bytes(changed)))

    other_source, _ = _sspl1(mean_offset=1)
    with pytest.raises(C.ContainerError, match="means_zlib"):
        C.parse_container(valid, reference_sspl1=other_source)


def test_canonical_zlib_payloads_and_single_bit_with_repaired_crc_are_rejected(
    source: tuple[bytes, dict[str, np.ndarray]], flat_blob: bytes
) -> None:
    z_blob = C.encode_ssp2z(source[0])
    means = C.parse_container(z_blob).payloads["means_zlib"]
    noncanonical_means = zlib.compress(zlib.decompress(means), level=1)
    with pytest.raises(C.ContainerError, match="canonical"):
        C.parse_container(_replace_payload(z_blob, "means_zlib", noncanonical_means))

    model = C.parse_container(flat_blob).payloads["index_model_zlib"]
    noncanonical_model = zlib.compress(zlib.decompress(model), level=1)
    with pytest.raises(C.ContainerError, match="canonical"):
        C.parse_container(_replace_payload(flat_blob, "index_model_zlib", noncanonical_model))

    corrupted = bytearray(flat_blob)
    parsed = C.parse_container(flat_blob)
    codebook = next(entry for entry in parsed.entries if entry.name == "codebook_raw")
    corrupted[codebook.offset + codebook.length // 2] ^= 1
    # Raw codebook bytes have no internal checksum; after an attacker repairs the complete CRC,
    # the mutation is a different syntactically valid stream.  Reference-free parsing accepts it,
    # while the original complete-container CRC rejects the un-repaired single-bit corruption.
    with pytest.raises(C.ContainerError, match="CRC"):
        C.parse_container(bytes(corrupted))
    repaired = _repair_outer(bytes(corrupted))
    assert C.parse_container(repaired).payloads["codebook_raw"] != parsed.payloads["codebook_raw"]


def test_descriptor_codebook_and_outer_stage_mismatch_fail_closed(flat_blob: bytes) -> None:
    parsed = C.parse_container(flat_blob)
    descriptor = bytearray(parsed.payloads["descriptor_raw"])
    descriptor[-1] = 1  # reserved uint32 least-significant byte
    with pytest.raises(C.ContainerError, match="reserved"):
        C.parse_container(_replace_payload(flat_blob, "descriptor_raw", bytes(descriptor)))

    values = list(C.DESCRIPTOR.unpack(parsed.payloads["descriptor_raw"]))
    values[6] -= 1  # codebook_entries
    with pytest.raises(C.ContainerError, match="entry count"):
        C.parse_container(_replace_payload(flat_blob, "descriptor_raw", C.DESCRIPTOR.pack(*values)))

    with pytest.raises(C.ContainerError, match="codebook payload length"):
        C.parse_container(
            _replace_payload(flat_blob, "codebook_raw", parsed.payloads["codebook_raw"][:-1])
        )

    wrong_count = bytearray(flat_blob)
    wrong_count[7] = 8
    with pytest.raises(C.ContainerError):
        C.parse_container(bytes(wrong_count))


def test_index_model_sum_table_substitution_and_arithmetic_canonicality_fail_closed(
    flat_blob: bytes,
) -> None:
    parsed = C.parse_container(flat_blob)
    descriptor = parsed.descriptor
    assert descriptor is not None
    model_raw = bytearray(zlib.decompress(parsed.payloads["index_model_zlib"]))

    wrong_sum = bytearray(model_raw)
    first = struct.unpack_from("<H", wrong_sum)[0]
    struct.pack_into("<H", wrong_sum, 0, first + 1)
    with pytest.raises(C.ContainerError, match="sum to 32768"):
        C.parse_container(
            _replace_payload(
                flat_blob,
                "index_model_zlib",
                zlib.compress(bytes(wrong_sum), level=9),
            )
        )

    frequencies = np.frombuffer(model_raw, dtype="<u2").astype(np.uint32)
    donor = int(np.flatnonzero(frequencies > 1)[0])
    recipient = 0 if donor != 0 else 1
    frequencies[donor] -= 1
    frequencies[recipient] += 1
    substituted_model = zlib.compress(frequencies.astype("<u2").tobytes(), level=9)
    substituted = _replace_payload(flat_blob, "index_model_zlib", substituted_model)
    with pytest.raises(C.ContainerError, match="arithmetic|model"):
        C.decode_ssp2v(substituted)

    decoded = C.decode_ssp2v(flat_blob)
    rows = C.cdf_rows(frequencies)
    matching_payload = arithmetic.encode(decoded.indices[0], rows)
    fully_substituted = _replace_payload(substituted, "index_0_arith", matching_payload)
    C.parse_container(fully_substituted)
    with pytest.raises(C.ContainerError, match="does not canonically describe"):
        C.decode_ssp2v(fully_substituted)

    noncanonical = _replace_payload(
        flat_blob,
        "index_0_arith",
        parsed.payloads["index_0_arith"] + b"\0",
    )
    with pytest.raises(C.ContainerError, match="arithmetic"):
        C.decode_ssp2v(noncanonical)


def test_reference_common_payload_and_nonrgb_identity_are_mandatory(
    source: tuple[bytes, dict[str, np.ndarray]], flat_blob: bytes
) -> None:
    other_means, _ = _sspl1(mean_offset=1)
    with pytest.raises(C.ContainerError, match="means_zlib"):
        C.parse_container(flat_blob, reference_sspl1=other_means)

    other_colors, _ = _sspl1(color_offset=1)
    # RGB is intentionally lossy and the V stream has no source-color payload.  The shared exact
    # payload/header authority is still valid, while accounting changes against the new reference.
    parsed = C.parse_container(flat_blob, reference_sspl1=other_colors)
    assert parsed.magic == b"SSP2V"
    original = C.decode_ssp2v(flat_blob, reference_sspl1=source[0]).rgb_accounting
    shifted = C.decode_ssp2v(flat_blob, reference_sspl1=other_colors).rgb_accounting
    assert original is not None and shifted is not None
    assert original.rgb_sse != shifted.rgb_sse


def test_encoder_rejects_noncanonical_adapter_and_index_or_codebook_contracts(
    source: tuple[bytes, dict[str, np.ndarray]],
) -> None:
    descriptor = _descriptor(0, 2, 64)
    codebooks, indices = _flat_inputs(descriptor)

    def noncanonical_encoder(symbols: np.ndarray, rows: np.ndarray) -> bytes:
        return arithmetic.encode(symbols, rows) + b"\0"

    with pytest.raises(C.ContainerError, match="byte parity"):
        C.encode_ssp2v(
            source[0],
            descriptor,
            codebooks,
            indices,
            arithmetic_encode=noncanonical_encoder,
        )
    bad_index = indices[0].copy()
    bad_index[-1] = descriptor.codebook_entries
    with pytest.raises(C.ContainerError, match="exceeds"):
        C.encode_ssp2v(source[0], descriptor, codebooks, (bad_index,))
    with pytest.raises(C.ContainerError, match="stage count"):
        C.encode_ssp2v(source[0], descriptor, codebooks, ())
    with pytest.raises(C.ContainerError, match="shape"):
        C.serialize_codebooks((codebooks[0][:-1],), descriptor)


def test_complete_component_accounting_never_conflates_prefix_or_payloads(
    flat_blob: bytes, rvq_blob: bytes
) -> None:
    for blob, expected_prefix in ((flat_blob, 260), (rvq_blob, 300)):
        parsed = C.parse_container(blob)
        components = parsed.component_bytes()
        assert components["outer"] == 20
        assert components["header"] == 100
        assert components["directory"] == expected_prefix - 120
        assert sum(value for name, value in components.items() if name != "total") == len(blob)
        assert components["total"] == len(blob)
