from __future__ import annotations

import copy
import hashlib
import json
import struct
import zlib

import numpy as np
import pytest

from benchmarks import sgi_entropy_oracle as O


def _pack(values: np.ndarray, bits: int) -> bytes:
    dtype = np.dtype("u1" if bits <= 8 else "<u2" if bits <= 16 else "<u4")
    return np.asarray(values, dtype=dtype).tobytes()


def _planar(values: np.ndarray, bits: int) -> bytes:
    dtype = np.dtype("u1" if bits <= 8 else "<u2" if bits <= 16 else "<u4")
    raw = np.asarray(values, dtype=dtype).tobytes()
    if dtype.itemsize == 1:
        return raw
    return np.frombuffer(raw, dtype=np.uint8).reshape(-1, dtype.itemsize).T.ravel().tobytes()


def _deltas(values: np.ndarray, bits: int) -> np.ndarray:
    preceding = np.vstack([np.zeros((1, values.shape[1]), dtype=np.uint32), values[:-1]])
    return (values.astype(np.uint32) - preceding) & ((1 << bits) - 1)


def _blob(
    *,
    n: int = 8,
    bits: tuple[int, int, int, int] = (12, 6, 6, 8),
    means: np.ndarray | None = None,
    scales: np.ndarray | None = None,
    rotation: np.ndarray | None = None,
    colors: np.ndarray | None = None,
) -> tuple[bytes, dict[str, np.ndarray], dict[str, int]]:
    bm, bs, br, bc = bits
    means = np.asarray(
        means
        if means is not None
        else np.column_stack(
            [
                np.arange(n, dtype=np.uint32) * 31,
                np.arange(n, dtype=np.uint32)[::-1] * 47,
            ]
        ),
        dtype=np.uint32,
    )
    scales = np.asarray(
        scales
        if scales is not None
        else np.column_stack(
            [
                np.arange(n, dtype=np.uint32) % (1 << bs),
                (3 * np.arange(n, dtype=np.uint32)) % (1 << bs),
            ]
        ),
        dtype=np.uint32,
    )
    rotation = np.asarray(
        rotation if rotation is not None else np.arange(n, dtype=np.uint32) % (1 << br),
        dtype=np.uint32,
    )
    colors = np.asarray(
        colors
        if colors is not None
        else np.column_stack(
            [
                np.arange(n, dtype=np.uint32),
                2 * np.arange(n, dtype=np.uint32),
                3 * np.arange(n, dtype=np.uint32),
            ]
        ),
        dtype=np.uint32,
    )
    raw_streams = [
        _planar(_deltas(means, bm).T.ravel(), bm),
        _pack(scales.T.ravel(), bs),
        _pack(rotation, br),
        _pack(_deltas(colors, bc).T.ravel(), bc),
    ]
    header = {
        "n": n,
        "H": 64,
        "W": 64,
        "bits": list(bits),
        "morton": True,
        "color_delta": True,
        "means_byte_planar": True,
        "rot_circular": True,
        "stream_codec": "zlib",
        "stream_raw_lengths": [len(raw) for raw in raw_streams],
        "color_lo": [-1.0, -1.0, -1.0],
        "color_hi": [1.0, 1.0, 1.0],
        "scale_lo": [-1.0, -1.0],
        "scale_hi": [1.0, 1.0],
        "means_lo": [0.0, 0.0],
        "means_hi": [63.0, 63.0],
        "has_opacity": False,
        "bits_opacity": 8,
        "renderer": "cuda",
        "aa_dilation": 0.0,
        "sigma_cutoff": 3.0,
        "support_fade": False,
        "render_chunk": 512,
    }
    header_bytes = json.dumps(header).encode()
    blob = O.MAGIC + struct.pack("<I", len(header_bytes)) + header_bytes
    payload_lengths = {}
    for name, raw in zip(("means", "scales", "rotation", "colors"), raw_streams, strict=True):
        payload = zlib.compress(raw, level=9)
        payload_lengths[name] = len(payload)
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
    return blob, expected, payload_lengths


def _row(image: int, bits: tuple[int, int, int, int], numerator: int, denominator: int) -> dict:
    return O.seal_record(
        {
            "schema": O.ROW_SCHEMA,
            "image_id": O.FROZEN_IMAGE_IDS[image],
            "bit_tuple": list(bits),
            "candidate_lower_bound_bytes": numerator,
            "baseline_complete_sspl1_bytes": denominator,
            "ratio_numerator": numerator,
            "ratio_denominator": denominator,
            "strict_byte_win": numerator < denominator,
        }
    )


def _grid(first: tuple[int, int] = (89, 100), second: tuple[int, int] = (89, 100)):
    return [
        _row(image, bits, *(first if tuple_index == 0 else second))
        for tuple_index, bits in enumerate(O.FROZEN_BIT_TUPLES)
        for image in range(8)
    ]


def test_parse_recovers_all_absolute_production_symbols_and_exact_bytes():
    blob, expected, payload_lengths = _blob(n=11)
    parsed = O.parse_sspl1(blob)

    assert parsed.total_bytes == len(blob)
    assert parsed.payload_bytes == payload_lengths
    assert parsed.blob_sha256 == hashlib.sha256(blob).hexdigest()
    assert set(parsed.symbols) == set(expected)
    for name, values in expected.items():
        np.testing.assert_array_equal(parsed.symbols[name], values)
    copies = O.extract_quantized_arrays(blob)
    copies["mean_x"][0] += 1
    assert parsed.symbols["mean_x"][0] == expected["mean_x"][0]


def test_extraction_matches_the_production_encoder_quantizers():
    torch = pytest.importorskip("torch")
    from structsplat import codec
    from structsplat.gaussians import GaussianField

    generator = torch.Generator().manual_seed(17)
    n, height, width = 31, 47, 53
    field = GaussianField(
        torch.rand((n, 2), generator=generator) * torch.tensor([width - 1, height - 1]),
        torch.randn((n, 2), generator=generator) * 0.25,
        torch.randn(n, generator=generator) * 4.0,
        torch.randn((n, 3), generator=generator),
    )
    config = codec.CodecConfig(bits_means=12, bits_scales=6, bits_rot=6, bits_colors=8)
    blob = codec.encode(field, height, width, config)
    parsed = O.parse_sspl1(blob)
    header = codec.blob_header(blob)

    means, scales, rotation, colors, _ = codec._params(field)  # noqa: SLF001
    order = codec._morton_order(means[:, 0], means[:, 1], height, width)  # noqa: SLF001
    means, scales, rotation, colors = (value[order] for value in (means, scales, rotation, colors))
    expected = {
        "mean_x": codec._quant(  # noqa: SLF001
            means, header["means_lo"], header["means_hi"], 12
        )[:, 0],
        "mean_y": codec._quant(  # noqa: SLF001
            means, header["means_lo"], header["means_hi"], 12
        )[:, 1],
        "scale_x": codec._quant(  # noqa: SLF001
            scales, header["scale_lo"], header["scale_hi"], 6
        )[:, 0],
        "scale_y": codec._quant(  # noqa: SLF001
            scales, header["scale_lo"], header["scale_hi"], 6
        )[:, 1],
        "rotation": codec._quant_circular(rotation, np.pi, 6),  # noqa: SLF001
        "red": codec._quant(  # noqa: SLF001
            colors, header["color_lo"], header["color_hi"], 8
        )[:, 0],
        "green": codec._quant(  # noqa: SLF001
            colors, header["color_lo"], header["color_hi"], 8
        )[:, 1],
        "blue": codec._quant(  # noqa: SLF001
            colors, header["color_lo"], header["color_hi"], 8
        )[:, 2],
    }
    for name, values in expected.items():
        np.testing.assert_array_equal(parsed.symbols[name], values)


def test_parser_rejects_trailing_bytes_noncanonical_zlib_and_symbol_overflow():
    blob, _, _ = _blob()
    with pytest.raises(O.OracleError, match="trailing"):
        O.parse_sspl1(blob + b"x")

    # Locate the first payload and replace it by a valid level-1 encoding of the same raw bytes.
    (hlen,) = struct.unpack_from("<I", blob, 5)
    offset = 9 + hlen
    (length,) = struct.unpack_from("<I", blob, offset)
    payload = blob[offset + 4 : offset + 4 + length]
    raw = zlib.decompress(payload)
    alternate = zlib.compress(raw, level=1)
    changed = (
        blob[:offset] + struct.pack("<I", len(alternate)) + alternate + blob[offset + 4 + length :]
    )
    with pytest.raises(O.OracleError, match="canonical"):
        O.parse_sspl1(changed)
    assert O.parse_sspl1(changed, canonical_zlib9=False).total_bytes == len(changed)

    overflowing, _, _ = _blob(scales=np.full((8, 2), 127, dtype=np.uint32))
    with pytest.raises(O.OracleError, match="bit depth"):
        O.parse_sspl1(overflowing)


def test_parser_rejects_nonproduction_header_keys_order_and_bytes():
    blob, _, _ = _blob()
    (hlen,) = struct.unpack_from("<I", blob, 5)
    header = json.loads(blob[9 : 9 + hlen])
    suffix = blob[9 + hlen :]

    header["unused_padding"] = "inflate-baseline"
    extra = json.dumps(header).encode()
    changed = blob[:5] + struct.pack("<I", len(extra)) + extra + suffix
    with pytest.raises(O.OracleError, match="key set/order"):
        O.parse_sspl1(changed)

    header.pop("unused_padding")
    compact = json.dumps(header, separators=(",", ":")).encode()
    changed = blob[:5] + struct.pack("<I", len(compact)) + compact + suffix
    with pytest.raises(O.OracleError, match="exact production JSON"):
        O.parse_sspl1(changed)


def test_position_cell_formula_has_exact_edges_and_uses_wide_multiply():
    q = np.array([0, 255, 256, 4095, (1 << 32) - 1], dtype=np.uint64)
    cells = O.position_cells(q, q, 32 if q[-1] > 4095 else 12)
    assert cells[0] == 0
    assert cells[-1] == 255

    q12 = np.array([0, 255, 256, 4095], dtype=np.uint32)
    assert O.position_cells(q12, np.zeros_like(q12), 12).tolist() == [0, 0, 1, 15]


def test_entropy_byte_floor_is_integer_certified_not_float_rounded():
    n = 8
    symbols = {
        "mean_x": np.zeros(n, dtype=np.uint32),
        "mean_y": np.zeros(n, dtype=np.uint32),
    }
    # Each of five channels contains four zeros and four ones in one cell: exactly 8 bits per
    # channel, 40 total bits, hence exactly five certified bytes.
    for name in ("scale_x", "scale_y", "red", "green", "blue"):
        symbols[name] = np.array([0, 1] * 4, dtype=np.uint32)
    certificate = O.conditional_entropy_certificate(symbols, 12)
    assert certificate["entropy_floor_bytes_exact"] == 5
    assert certificate["total_bits_diagnostic"] == pytest.approx(40.0)

    # Making every symbol cell-deterministic collapses the exact ratio to one.
    for name in ("scale_x", "scale_y", "red", "green", "blue"):
        symbols[name] = np.zeros(n, dtype=np.uint32)
    assert O.conditional_entropy_certificate(symbols, 12)["entropy_floor_bytes_exact"] == 0


def test_row_uses_only_compulsory_overhead_inherited_payloads_and_entropy_floor():
    blob, _, payloads = _blob(n=8)
    row = O.make_oracle_row(
        blob,
        "synthetic",
        expected_n=None,
        require_prepared_extent=False,
    )
    expected = (
        O.SSP2E_COMPULSORY_BYTES
        + payloads["means"]
        + payloads["rotation"]
        + row["entropy"]["entropy_floor_bytes_exact"]
    )
    assert row["candidate_lower_bound_bytes"] == expected
    assert row["ratio_denominator"] == len(blob)
    O.verify_record(row)
    assert O.replay_row(blob, row, expected_n=None, require_prepared_extent=False)["row_exact"]


def test_ssp2e_layout_is_exact_and_unrepresentable_metadata_fails():
    assert O.SSP2E_OUTER_BYTES == 20
    assert O.SSP2E_HEADER_BYTES == 100
    assert O.SSP2E_DIRECTORY_ENTRY_BYTES == 20
    assert O.SSP2E_COMPULSORY_BYTES == 656

    blob, _, _ = _blob()
    (hlen,) = struct.unpack_from("<I", blob, 5)
    header = json.loads(blob[9 : 9 + hlen])
    header["scale_lo"][0] = 0.1
    raw_header = json.dumps(header).encode()
    changed = blob[:5] + struct.pack("<I", len(raw_header)) + raw_header + blob[9 + hlen :]
    with pytest.raises(O.OracleError, match="float32"):
        O.make_oracle_row(
            changed,
            "synthetic",
            expected_n=None,
            require_prepared_extent=False,
        )


def test_bootstrap_matrix_is_frozen_c_order_uint8():
    matrix = O.bootstrap_index_matrix()
    assert matrix.shape == (100_000, 8)
    assert matrix.dtype == np.uint8
    assert not matrix.flags.writeable
    assert matrix[:3].tolist() == [
        [1, 4, 4, 0, 1, 4, 7, 5],
        [7, 4, 2, 0, 6, 5, 2, 4],
        [7, 1, 6, 6, 1, 6, 2, 1],
    ]
    assert hashlib.sha256(matrix.tobytes(order="C")).hexdigest() == O.BOOTSTRAP_MATRIX_SHA256


def test_all_exact_gate_boundaries_survive_and_replay():
    # Seven 0.80 ratios plus one exact 1.02 ratio exercise the permitted worst equality while
    # satisfying the GM, strict-win, and bootstrap gates.
    rows = []
    for bits in O.FROZEN_BIT_TUPLES:
        for image in range(8):
            rows.append(_row(image, bits, 102 if image == 7 else 80, 100))
    result = O._analyze_unbacked_test_rows(rows)  # noqa: SLF001
    assert result["decision"] == "TEST_ONLY_NO_SCIENTIFIC_DECISION"
    assert result["would_be_decision"] == "ORACLE_INCONCLUSIVE_IMPLEMENT_CODER"
    assert result["coder_authorized"] is False
    for item in result["tuple_results"]:
        assert all(item["gates"].values())
        assert item["worst_ratio_diagnostic"] == pytest.approx(1.02)
        assert item["strict_image_wins"] == 7


def test_any_failed_tuple_kills_the_whole_fixed_grid_and_tampering_is_rejected():
    result = O._analyze_unbacked_test_rows(_grid(second=(91, 100)))  # noqa: SLF001
    assert result["would_be_decision"] == ("ABANDON_JOINTLY_REQUIRED_TWO_TUPLE_COMP008_CANDIDATE")
    assert result["coder_authorized"] is False
    assert result["tuple_results"][0]["all_gates_pass"] is True
    assert result["tuple_results"][1]["gates"]["geometric_mean_lte_0_90_exact"] is False
    assert result["surviving_bit_tuples"] == [list(O.FROZEN_BIT_TUPLES[0])]
    assert result["rejected_bit_tuples"] == [list(O.FROZEN_BIT_TUPLES[1])]

    rows = _grid()
    tampered = copy.deepcopy(rows)
    tampered[0]["ratio_numerator"] += 1
    with pytest.raises(O.OracleError, match="mismatch"):
        O._analyze_unbacked_test_rows(tampered)  # noqa: SLF001


def test_scientific_analysis_requires_and_reparses_all_persisted_blobs():
    rows = []
    blobs = {}
    for bits in O.FROZEN_BIT_TUPLES:
        blob, _, _ = _blob(n=O.FROZEN_GAUSSIAN_COUNT, bits=bits)
        (hlen,) = struct.unpack_from("<I", blob, 5)
        header = json.loads(blob[9 : 9 + hlen])
        header["H"] = header["W"] = 768
        header["means_hi"] = [767.0, 767.0]
        raw_header = json.dumps(header).encode()
        blob = blob[:5] + struct.pack("<I", len(raw_header)) + raw_header + blob[9 + hlen :]
        for image_id in O.FROZEN_IMAGE_IDS:
            row = O.make_oracle_row(blob, image_id)
            rows.append(row)
            blobs[(image_id, bits)] = blob

    with pytest.raises(O.OracleError, match="requires all 16 persisted"):
        O.analyze_rows(rows)
    result = O.analyze_rows(rows, blobs=blobs)
    assert result["row_count"] == 16
    assert result["tuple_results"][0]["image_ids"] == list(O.FROZEN_IMAGE_IDS)
    replay = O.replay_analysis(rows, result, blobs=blobs)
    assert replay["ok"] and replay["analysis_exact"]

    forged = copy.deepcopy(rows)
    forged[0]["ratio_numerator"] += 1
    forged[0]["candidate_lower_bound_bytes"] += 1
    forged[0] = O.seal_record(forged[0])
    with pytest.raises(O.OracleError, match="artifact-backed oracle row differs"):
        O.analyze_rows(forged, blobs=blobs)


def test_geometric_mean_equality_at_point_nine_is_allowed_exactly():
    result = O._analyze_unbacked_test_rows(  # noqa: SLF001
        _grid(first=(9, 10), second=(9, 10))
    )
    assert all(item["gates"]["geometric_mean_lte_0_90_exact"] for item in result["tuple_results"])


def test_bootstrap_equality_at_point_nine_five_fails_strictly():
    summary = O._bootstrap_summary([19] * 8, [20] * 8)  # noqa: SLF001
    assert summary["samples_equal_0_95"] == O.BOOTSTRAP_REPLICATES
    assert summary["upper_strictly_below_0_95_exact"] is False


def test_cli_analyze_emits_canonical_json(tmp_path, capsys):
    path = tmp_path / "rows.jsonl"
    path.write_bytes(b"".join(O.canonical_json(row) for row in _grid()))
    with pytest.raises(O.OracleError, match="requires all 16 persisted SSPL1 blobs"):
        O.main(["analyze", "--rows", str(path)])
    assert capsys.readouterr().out == ""
