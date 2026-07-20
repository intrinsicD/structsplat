from __future__ import annotations

import ctypes
import hashlib
import itertools
from pathlib import Path

import numpy as np
import pytest

from benchmarks.ssp2e_arithmetic import (
    TOTAL_FREQUENCY,
    ArithmeticCodingError,
    NativeArithmeticCoder,
    _Parameters,
    _decode_generic,
    _encode_detailed_generic,
    _zero_padded_terminator_inside,
    build_native,
    coder_certificate,
    decode,
    encode,
    encode_detailed,
)
from benchmarks.ssp2e_assets import (
    CDF_ASSET_NAME,
    CDF_ASSET_SHA256,
    CDF_ASSET_SIZE,
    COST_ASSET_NAME,
    COST_ASSET_SHA256,
    COST_ASSET_SIZE,
    AssetError,
    build_cdf_asset_bytes,
    build_cost_asset_bytes,
    load_costs,
    load_tables,
    verify_assets,
)


def _cdf(frequencies: tuple[int, ...]) -> np.ndarray:
    return np.asarray((0, *itertools.accumulate(frequencies)), dtype=np.uint32)


def _rows(row: np.ndarray, count: int) -> np.ndarray:
    return np.tile(row, (count, 1))


def _positive_compositions(total: int, parts: int):
    for cuts in itertools.combinations(range(1, total), parts - 1):
        endpoints = (0, *cuts, total)
        yield tuple(endpoints[index + 1] - endpoints[index] for index in range(parts))


def test_decimal_generators_reproduce_frozen_asset_hashes() -> None:
    cdf_data = build_cdf_asset_bytes()
    cost_data = build_cost_asset_bytes()
    assert len(cdf_data) == CDF_ASSET_SIZE
    assert hashlib.sha256(cdf_data).hexdigest() == CDF_ASSET_SHA256
    assert len(cost_data) == COST_ASSET_SIZE
    assert hashlib.sha256(cost_data).hexdigest() == COST_ASSET_SHA256


def test_frozen_assets_load_strictly() -> None:
    identity = verify_assets()
    assert identity["cdf_sha256"] == CDF_ASSET_SHA256
    assert identity["cost_sha256"] == COST_ASSET_SHA256
    tables = load_tables()
    assert tuple(tables) == (64, 256)
    for alphabet, table in tables.items():
        assert len(table.cdfs) == 64 * 16
        assert len(table.frequencies) == 64 * 16 * alphabet
        for location, scale in ((0, 0), (31, 8), (63, 15)):
            row = table.row(location, scale)
            assert row[0] == 0
            assert row[-1] == TOTAL_FREQUENCY
            assert all(left < right for left, right in itertools.pairwise(row))
    costs = load_costs()
    assert len(costs) == TOTAL_FREQUENCY + 1
    assert costs[1] == 15 * (1 << 24)
    assert costs[TOTAL_FREQUENCY // 2] == 1 << 24
    assert costs[TOTAL_FREQUENCY] == 0


@pytest.mark.parametrize("asset_name", (CDF_ASSET_NAME, COST_ASSET_NAME))
def test_frozen_asset_loader_rejects_mutation(tmp_path: Path, asset_name: str) -> None:
    source = Path(__file__).parents[1] / "benchmarks" / "assets" / asset_name
    mutated = bytearray(source.read_bytes())
    mutated[len(mutated) // 2] ^= 1
    path = tmp_path / asset_name
    path.write_bytes(mutated)
    with pytest.raises(AssetError, match="SHA-256"):
        if asset_name == CDF_ASSET_NAME:
            load_tables(path)
        else:
            load_costs(path)


def test_reduced_width_exhaustive_compositions_sequences_and_terminators() -> None:
    parameters = _Parameters(width=8, total=4)
    cases = 0
    for alphabet in (2, 3):
        for frequencies in _positive_compositions(parameters.total, alphabet):
            row = _cdf(frequencies)
            for count in range(7):
                cdf_rows = _rows(row, count)
                for symbols in itertools.product(range(alphabet), repeat=count):
                    details = _encode_detailed_generic(symbols, cdf_rows, parameters)
                    recovered = _decode_generic(
                        details.payload,
                        cdf_rows,
                        count,
                        parameters,
                    )
                    assert tuple(int(value) for value in recovered) == symbols
                    assert _zero_padded_terminator_inside(details, parameters.width)
                    cases += 1
    assert cases == 3660


def test_reduced_width_exhaustive_varying_cdf_rows() -> None:
    """Exhaust every per-position CDF choice, not only one repeated composition."""
    parameters = _Parameters(width=8, total=4)
    cases = 0
    for alphabet in (2, 3):
        rows = tuple(
            _cdf(frequencies) for frequencies in _positive_compositions(parameters.total, alphabet)
        )
        for count in range(7):
            for row_indices in itertools.product(range(len(rows)), repeat=count):
                cdf_rows = np.asarray([rows[index] for index in row_indices], dtype=np.uint32)
                if count == 0:
                    cdf_rows = np.empty((0, alphabet + 1), dtype=np.uint32)
                for symbols in itertools.product(range(alphabet), repeat=count):
                    details = _encode_detailed_generic(symbols, cdf_rows, parameters)
                    recovered = _decode_generic(
                        details.payload,
                        cdf_rows,
                        count,
                        parameters,
                    )
                    assert tuple(int(value) for value in recovered) == symbols
                    assert _zero_padded_terminator_inside(details, parameters.width)
                    cases += 1
    assert cases == 653_858


@pytest.mark.parametrize(
    ("frequencies", "symbols"),
    (
        ((512,) * 64, np.arange(8192, dtype=np.uint32) % 64),
        (
            (32705, *(1 for _ in range(63))),
            np.full(8192, 63, dtype=np.uint32),
        ),
        (
            (128,) * 256,
            np.arange(8192, dtype=np.uint32) % 256,
        ),
    ),
    ids=("uniform-64", "minimum-frequency-64", "uniform-256"),
)
def test_production_width_representative_roundtrip_and_certificate(
    frequencies: tuple[int, ...],
    symbols: np.ndarray,
) -> None:
    row = _cdf(frequencies)
    cdf_rows = _rows(row, len(symbols))
    payload = encode(symbols, cdf_rows)
    assert np.array_equal(decode(payload, cdf_rows, len(symbols)), symbols)
    certificate = coder_certificate(symbols, cdf_rows, payload)
    assert certificate["inequality_holds"] is True
    assert certificate["payload_bytes"] == len(payload)
    assert certificate["payload_sha256"] == hashlib.sha256(payload).hexdigest()
    assert certificate["physical_bit_length"] <= 8 * len(payload)
    assert certificate["physical_bit_length"] > 8 * (len(payload) - 1)


def test_static_table_varying_rows_roundtrip() -> None:
    table = load_tables()[256]
    count = 1024
    cdf_rows = np.asarray(
        [table.row(index % 64, (index * 11) % 16) for index in range(count)],
        dtype=np.uint32,
    )
    symbols = (np.arange(count, dtype=np.uint32) * 37 + 19) % 256
    payload = encode(symbols, cdf_rows)
    assert np.array_equal(decode(payload, cdf_rows, count), symbols)


def test_empty_stream_has_one_canonical_encoding() -> None:
    cdf_rows = np.empty((0, 3), dtype=np.uint32)
    payload = encode([], cdf_rows)
    assert payload == b"\x40"
    assert decode(payload, cdf_rows, 0).tolist() == []
    with pytest.raises(ArithmeticCodingError, match="malformed"):
        decode(b"\x40\x00", cdf_rows, 0)


def test_malformed_noncanonical_wrong_count_and_model_are_rejected() -> None:
    row = _cdf((512,) * 64)
    symbols = np.arange(129, dtype=np.uint32) % 64
    cdf_rows = _rows(row, len(symbols))
    details = encode_detailed(symbols, cdf_rows)
    payload = details.payload

    malformed = {
        "truncation": payload[:-1],
        "zero suffix": payload + b"\x00",
        "nonzero suffix": payload + b"\x01",
        "termination mutation": payload[:-1] + bytes((payload[-1] ^ 1,)),
    }
    for candidate in malformed.values():
        with pytest.raises(ArithmeticCodingError, match="malformed"):
            decode(candidate, cdf_rows, len(symbols))

    with pytest.raises(ArithmeticCodingError, match="malformed"):
        decode(payload, cdf_rows[:-1], len(symbols) - 1)
    extra_rows = np.concatenate((cdf_rows, cdf_rows[-1:]), axis=0)
    with pytest.raises(ArithmeticCodingError, match="malformed"):
        decode(payload, extra_rows, len(symbols) + 1)

    wrong_model = cdf_rows.copy()
    wrong_model[0, 1] += 1
    with pytest.raises(ArithmeticCodingError, match="malformed"):
        decode(payload, wrong_model, len(symbols))


def test_nonzero_padding_and_representative_bit_flip_are_detected() -> None:
    row = _cdf((512,) * 64)
    symbols = np.asarray((0, 1), dtype=np.uint32)
    cdf_rows = _rows(row, len(symbols))
    details = encode_detailed(symbols, cdf_rows)
    assert details.bit_length % 8 == 6
    nonzero_padding = details.payload[:-1] + bytes((details.payload[-1] | 1,))
    with pytest.raises(ArithmeticCodingError, match="malformed"):
        decode(nonzero_padding, cdf_rows, len(symbols))

    longer_symbols = np.arange(129, dtype=np.uint32) % 64
    longer_rows = _rows(row, len(longer_symbols))
    payload = bytearray(encode(longer_symbols, longer_rows))
    payload[0] ^= 0x80
    # A raw arithmetic payload has no checksum: a canonical bit flip can decode
    # to a different sequence, which the enclosing container's CRC/symbol hash
    # is required to reject.
    try:
        recovered = decode(payload, longer_rows, len(longer_symbols))
    except ArithmeticCodingError:
        return
    assert not np.array_equal(recovered, longer_symbols)


@pytest.mark.parametrize(
    ("symbols", "cdf_rows", "message"),
    (
        (np.asarray((2,), dtype=np.uint32), np.asarray(((0, 1, 32768),)), "outside"),
        (np.asarray((0,), dtype=np.int32), np.asarray(((1, 2, 32768),)), "start"),
        (np.asarray((0,), dtype=np.int32), np.asarray(((0, 0, 32768),)), "increasing"),
        (np.asarray((-1,), dtype=np.int32), np.asarray(((0, 1, 32768),)), "negative"),
        (np.asarray((0.0,)), np.asarray(((0, 1, 32768),)), "integers"),
    ),
)
def test_illegal_inputs_fail_closed(
    symbols: np.ndarray,
    cdf_rows: np.ndarray,
    message: str,
) -> None:
    with pytest.raises(ArithmeticCodingError, match=message):
        encode(symbols, cdf_rows)


def test_decoder_rejects_empty_physical_payload() -> None:
    with pytest.raises(ArithmeticCodingError, match="empty"):
        decode(b"", np.empty((0, 3), dtype=np.uint32), 0)


def test_python_native_differential(tmp_path: Path) -> None:
    library = build_native(tmp_path / "libssp2e_range.so")
    native = NativeArithmeticCoder(library)
    table64 = load_tables()[64]
    table256 = load_tables()[256]
    cases: list[tuple[np.ndarray, np.ndarray]] = []

    symbols64 = (np.arange(777, dtype=np.uint32) * 29 + 7) % 64
    rows64 = np.asarray(
        [table64.row(index % 64, (index * 5) % 16) for index in range(len(symbols64))],
        dtype=np.uint32,
    )
    cases.append((symbols64, rows64))

    symbols256 = (np.arange(513, dtype=np.uint32) * 101 + 31) % 256
    rows256 = np.asarray(
        [table256.row(index % 64, (index * 13) % 16) for index in range(len(symbols256))],
        dtype=np.uint32,
    )
    cases.append((symbols256, rows256))

    rare_row = _cdf((32705, *(1 for _ in range(63))))
    rare_symbols = np.full(1024, 63, dtype=np.uint32)
    cases.append((rare_symbols, _rows(rare_row, len(rare_symbols))))

    for symbols, cdf_rows in cases:
        python_payload = encode(symbols, cdf_rows)
        native_payload = native.encode(symbols, cdf_rows)
        assert native_payload == python_payload
        assert np.array_equal(native.decode(python_payload, cdf_rows, len(symbols)), symbols)
        assert np.array_equal(decode(native_payload, cdf_rows, len(symbols)), symbols)

        with pytest.raises(ArithmeticCodingError, match="canonical"):
            native.decode(python_payload + b"\x00", cdf_rows, len(symbols))


def test_native_repeated_size_matches_exact_python_and_native_payloads(tmp_path: Path) -> None:
    native = NativeArithmeticCoder(build_native(tmp_path / "libssp2e_repeated.so"))
    tables = load_tables()
    rng = np.random.default_rng(20260718)
    e3_symbols = np.asarray(
        (
            2,
            0,
            2,
            2,
            2,
            0,
            0,
            2,
            0,
            0,
            1,
            0,
            0,
            2,
            2,
            2,
            1,
            2,
            2,
            2,
            2,
            2,
            1,
            2,
            0,
            0,
            2,
            0,
            1,
            2,
            2,
            1,
            1,
            0,
            0,
            1,
            0,
        ),
        dtype=np.uint32,
    )
    cases = (
        (np.empty(0, dtype=np.uint32), _cdf((16384, 16384))),
        (np.full(1024, 63, dtype=np.uint32), _cdf((32705, *(1 for _ in range(63))))),
        (
            rng.integers(0, 64, size=8192, dtype=np.uint32),
            np.asarray(tables[64].row(23, 9), dtype=np.uint32),
        ),
        (
            rng.integers(0, 256, size=8192, dtype=np.uint32),
            np.asarray(tables[256].row(41, 6), dtype=np.uint32),
        ),
        (e3_symbols, np.asarray((0, 11415, 30790, 32768), dtype=np.uint32)),
    )

    for symbols, row in cases:
        rows = _rows(row, len(symbols))
        python_payload = encode(symbols, rows)
        native_payload = native.encode(symbols, rows)
        assert native_payload == python_payload
        assert native.size_repeated(symbols, row) == len(python_payload)

    assert native.size_repeated([], np.asarray((0, 1, 32768), dtype=np.uint32)) == 1
    e3_details = encode_detailed(e3_symbols, _rows(cases[-1][1], len(e3_symbols)))
    assert len(e3_details.termination_bits) == 12


def test_native_repeated_size_rejects_malformed_python_inputs(tmp_path: Path) -> None:
    native = NativeArithmeticCoder(build_native(tmp_path / "libssp2e_repeated_negative.so"))
    cases = (
        (np.asarray((2,), dtype=np.uint32), np.asarray((0, 1, 32768)), "outside"),
        (np.asarray((0,), dtype=np.uint32), np.asarray((1, 2, 32768)), "start"),
        (np.asarray((0,), dtype=np.uint32), np.asarray((0, 0, 32768)), "increasing"),
        (np.asarray((-1,), dtype=np.int32), np.asarray((0, 1, 32768)), "negative"),
        (np.asarray((0.0,)), np.asarray((0, 1, 32768)), "integers"),
        (np.asarray((0,), dtype=np.uint32), np.asarray(((0, 1, 32768),)), "1-dimensional"),
    )
    for symbols, row, message in cases:
        with pytest.raises(ArithmeticCodingError, match=message):
            native.size_repeated(symbols, row)


def test_repeated_size_direct_abi_statuses_and_overflow_guards(tmp_path: Path) -> None:
    library = ctypes.CDLL(str(build_native(tmp_path / "libssp2e_repeated_abi.so")))
    uint32_pointer = ctypes.POINTER(ctypes.c_uint32)
    function = library.ssp2e_size_repeated
    function.argtypes = [
        uint32_pointer,
        uint32_pointer,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_size_t),
    ]
    function.restype = ctypes.c_int

    row = np.asarray((0, 16384, 32768), dtype=np.uint32)
    row_pointer = row.ctypes.data_as(uint32_pointer)
    symbol = np.asarray((0,), dtype=np.uint32)
    symbol_pointer = symbol.ctypes.data_as(uint32_pointer)

    assert function(None, row_pointer, 0, 2, None) == 1

    output_size = ctypes.c_size_t(99)
    assert function(None, row_pointer, 0, 2, ctypes.byref(output_size)) == 0
    assert output_size.value == 1

    output_size.value = 99
    assert function(None, None, 0, 2, ctypes.byref(output_size)) == 1
    assert output_size.value == 0

    output_size.value = 99
    assert function(None, row_pointer, 1, 2, ctypes.byref(output_size)) == 1
    assert output_size.value == 0

    malformed = np.asarray((0, 0, 32768), dtype=np.uint32)
    output_size.value = 99
    assert (
        function(
            None,
            malformed.ctypes.data_as(uint32_pointer),
            0,
            2,
            ctypes.byref(output_size),
        )
        == 3
    )
    assert output_size.value == 0

    outside = np.asarray((2,), dtype=np.uint32)
    output_size.value = 99
    assert (
        function(
            outside.ctypes.data_as(uint32_pointer),
            row_pointer,
            1,
            2,
            ctypes.byref(output_size),
        )
        == 3
    )
    assert output_size.value == 0

    output_size.value = 99
    assert function(None, row_pointer, 0, 32769, ctypes.byref(output_size)) == 3
    assert output_size.value == 0

    output_size.value = 99
    assert (
        function(
            symbol_pointer,
            row_pointer,
            ctypes.c_size_t(-1).value,
            2,
            ctypes.byref(output_size),
        )
        == 1
    )
    assert output_size.value == 0


def test_legacy_native_library_loads_until_repeated_size_is_requested(tmp_path: Path) -> None:
    legacy_path = build_native(
        tmp_path / "libssp2e_legacy.so",
        extra_flags=("-Dssp2e_size_repeated=ssp2e_size_repeated_legacy_hidden",),
    )
    native = NativeArithmeticCoder(legacy_path)
    row = _cdf((16384, 16384))
    symbols = np.asarray((0, 1, 1, 0), dtype=np.uint32)
    rows = _rows(row, len(symbols))
    payload = native.encode(symbols, rows)
    assert payload == encode(symbols, rows)
    assert np.array_equal(native.decode(payload, rows, len(symbols)), symbols)
    with pytest.raises(ArithmeticCodingError, match="rebuild the native arithmetic library"):
        native.size_repeated(symbols, row)


def test_native_decoder_rejects_each_frozen_negative_class(tmp_path: Path) -> None:
    native = NativeArithmeticCoder(build_native(tmp_path / "libssp2e_range_negative.so"))
    row = _cdf((512,) * 64)
    symbols = np.arange(129, dtype=np.uint32) % 64
    rows = _rows(row, len(symbols))
    payload = encode(symbols, rows)

    malformed = (
        payload[:-1],
        payload + b"\x00",
        payload + b"\x01",
        payload[:-1] + bytes((payload[-1] ^ 1,)),
    )
    for candidate in malformed:
        with pytest.raises(ArithmeticCodingError, match="native decode failed"):
            native.decode(candidate, rows, len(symbols))

    padding_symbols = np.asarray((0, 1), dtype=np.uint32)
    padding_rows = _rows(row, len(padding_symbols))
    detailed = encode_detailed(padding_symbols, padding_rows)
    assert detailed.bit_length % 8 == 6
    nonzero_padding = detailed.payload[:-1] + bytes((detailed.payload[-1] | 1,))
    with pytest.raises(ArithmeticCodingError, match="native decode failed"):
        native.decode(nonzero_padding, padding_rows, len(padding_symbols))

    with pytest.raises(ArithmeticCodingError, match="native decode failed"):
        native.decode(payload, rows[:-1], len(symbols) - 1)
    wrong_model = rows.copy()
    wrong_model[0, 1] += 1
    with pytest.raises(ArithmeticCodingError, match="native decode failed"):
        native.decode(payload, wrong_model, len(symbols))
