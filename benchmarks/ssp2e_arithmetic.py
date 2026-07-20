"""Exact canonical SSP2E v1 arithmetic coding primitives.

The public Python reference interface deliberately accepts the explicit
``uint32[N, A+1]`` CDF matrix used at the container boundary.  It does not
silently infer, normalize, or repair probabilities.
"""

from __future__ import annotations

import ctypes
import hashlib
import operator
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

FULL = 1 << 32
MASK = FULL - 1
HALF = 1 << 31
Q1 = 1 << 30
Q3 = 3 << 30
TOTAL_FREQUENCY = 32768


class ArithmeticCodingError(ValueError):
    """An arithmetic input or payload violates the frozen canonical format."""


@dataclass(frozen=True)
class EncodedPayload:
    """Reference-encoder details needed by proof tests and certificates."""

    payload: bytes
    bit_length: int
    actual_frequencies: tuple[int, ...]
    final_low: int
    final_high: int
    termination_bits: tuple[int, ...]


class _BitWriter:
    def __init__(self) -> None:
        self._bytes = bytearray()
        self._byte = 0
        self._used = 0
        self._bits: list[int] = []

    @property
    def bit_length(self) -> int:
        return len(self._bits)

    @property
    def bits(self) -> tuple[int, ...]:
        return tuple(self._bits)

    def write(self, bit: int) -> None:
        if bit not in (0, 1):
            raise AssertionError("bit writer received a non-bit")
        self._bits.append(bit)
        self._byte = (self._byte << 1) | bit
        self._used += 1
        if self._used == 8:
            self._bytes.append(self._byte)
            self._byte = 0
            self._used = 0

    def finish(self) -> bytes:
        if self._used:
            self._bytes.append(self._byte << (8 - self._used))
            self._byte = 0
            self._used = 0
        return bytes(self._bytes)


class _BitReader:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload
        self.position = 0

    def read(self) -> int:
        position = self.position
        self.position += 1
        if position >= 8 * len(self._payload):
            return 0
        byte = self._payload[position // 8]
        return (byte >> (7 - position % 8)) & 1


@dataclass(frozen=True)
class _Parameters:
    width: int
    total: int

    @property
    def full(self) -> int:
        return 1 << self.width

    @property
    def mask(self) -> int:
        return self.full - 1

    @property
    def half(self) -> int:
        return 1 << (self.width - 1)

    @property
    def q1(self) -> int:
        return 1 << (self.width - 2)

    @property
    def q3(self) -> int:
        return 3 << (self.width - 2)

    def validate(self) -> None:
        if self.width < 3:
            raise ArithmeticCodingError("arithmetic state width must be at least three")
        if not 2 <= self.total <= self.q1:
            raise ArithmeticCodingError(f"frequency total {self.total} is outside [2, 2^(width-2)]")


_PRODUCTION_PARAMETERS = _Parameters(width=32, total=TOTAL_FREQUENCY)


def _integer_array(value: ArrayLike, *, dimensions: int, name: str) -> NDArray[np.uint64]:
    array = np.asarray(value)
    if array.ndim != dimensions:
        raise ArithmeticCodingError(f"{name} must be a {dimensions}-dimensional array")
    if array.size == 0:
        return array.astype(np.uint64)
    if array.dtype.kind not in "iu":
        raise ArithmeticCodingError(f"{name} must contain integers")
    if array.dtype.kind == "i" and np.any(array < 0):
        raise ArithmeticCodingError(f"{name} contains a negative value")
    try:
        converted = array.astype(np.uint64, copy=False)
    except (OverflowError, TypeError, ValueError) as error:
        raise ArithmeticCodingError(f"{name} cannot be represented as unsigned integers") from error
    return converted


def _prepare_rows(
    cdf_rows: ArrayLike,
    count: int,
    parameters: _Parameters,
) -> NDArray[np.uint64]:
    rows = _integer_array(cdf_rows, dimensions=2, name="cdf_rows")
    if rows.shape[0] != count:
        raise ArithmeticCodingError(
            f"cdf_rows has {rows.shape[0]} rows for a declared symbol count of {count}"
        )
    if rows.shape[1] < 3:
        raise ArithmeticCodingError("every CDF row must describe at least two symbols")
    if np.any(rows[:, 0] != 0) or np.any(rows[:, -1] != parameters.total):
        raise ArithmeticCodingError("every CDF row must start at zero and end at the exact total")
    if np.any(rows[:, 1:] <= rows[:, :-1]):
        raise ArithmeticCodingError("every CDF row must be strictly increasing")
    return rows


def _prepare_encode_inputs(
    symbols: ArrayLike,
    cdf_rows: ArrayLike,
    parameters: _Parameters,
) -> tuple[NDArray[np.uint64], NDArray[np.uint64]]:
    symbol_array = _integer_array(symbols, dimensions=1, name="symbols")
    rows = _prepare_rows(cdf_rows, len(symbol_array), parameters)
    alphabet = rows.shape[1] - 1
    if np.any(symbol_array >= alphabet):
        raise ArithmeticCodingError(f"symbol is outside the CDF alphabet [0, {alphabet})")
    return symbol_array, rows


def _prepare_repeated_inputs(
    symbols: ArrayLike,
    cdf_row: ArrayLike,
    parameters: _Parameters,
) -> tuple[NDArray[np.uint64], NDArray[np.uint64]]:
    symbol_array = _integer_array(symbols, dimensions=1, name="symbols")
    row = _integer_array(cdf_row, dimensions=1, name="cdf_row")
    if row.size < 3:
        raise ArithmeticCodingError("cdf_row must describe at least two symbols")
    alphabet = int(row.size) - 1
    if alphabet > parameters.total:
        raise ArithmeticCodingError(
            f"cdf_row alphabet {alphabet} exceeds the exact frequency total"
        )
    if int(row[0]) != 0 or int(row[-1]) != parameters.total:
        raise ArithmeticCodingError("cdf_row must start at zero and end at the exact total")
    if np.any(row[1:] <= row[:-1]):
        raise ArithmeticCodingError("cdf_row must be strictly increasing")
    if np.any(symbol_array >= alphabet):
        raise ArithmeticCodingError(f"symbol is outside the CDF alphabet [0, {alphabet})")
    return symbol_array, row


def _emit_with_pending(writer: _BitWriter, bit: int, pending: int) -> int:
    writer.write(bit)
    for _ in range(pending):
        writer.write(1 - bit)
    return 0


def _encode_detailed_generic(
    symbols: ArrayLike,
    cdf_rows: ArrayLike,
    parameters: _Parameters,
) -> EncodedPayload:
    parameters.validate()
    symbol_array, rows = _prepare_encode_inputs(symbols, cdf_rows, parameters)
    low = 0
    high = parameters.mask
    pending = 0
    writer = _BitWriter()
    frequencies: list[int] = []

    for index, symbol_value in enumerate(symbol_array):
        symbol = int(symbol_value)
        cumulative = int(rows[index, symbol])
        frequency = int(rows[index, symbol + 1]) - cumulative
        frequencies.append(frequency)
        interval_range = high - low + 1
        unit = interval_range // parameters.total
        if unit < 1:
            raise ArithmeticCodingError("arithmetic interval cannot represent the frequency total")
        child_low = low + unit * cumulative
        child_high = child_low + unit * frequency - 1
        if not (0 <= child_low <= child_high <= parameters.mask):
            raise ArithmeticCodingError("arithmetic child interval overflow")
        low, high = child_low, child_high

        while True:
            if high < parameters.half:
                pending = _emit_with_pending(writer, 0, pending)
            elif low >= parameters.half:
                pending = _emit_with_pending(writer, 1, pending)
                low -= parameters.half
                high -= parameters.half
            elif low >= parameters.q1 and high < parameters.q3:
                pending += 1
                low -= parameters.q1
                high -= parameters.q1
            else:
                break
            low = (low << 1) & parameters.mask
            high = ((high << 1) & parameters.mask) | 1

    final_low, final_high = low, high
    termination_start = writer.bit_length
    pending += 1
    if low < parameters.q1:
        _emit_with_pending(writer, 0, pending)
    else:
        _emit_with_pending(writer, 1, pending)
    bit_length = writer.bit_length
    termination_bits = writer.bits[termination_start:]
    payload = writer.finish()
    return EncodedPayload(
        payload=payload,
        bit_length=bit_length,
        actual_frequencies=tuple(frequencies),
        final_low=final_low,
        final_high=final_high,
        termination_bits=termination_bits,
    )


def encode(symbols: ArrayLike, cdf_rows: ArrayLike) -> bytes:
    """Encode one symbol per explicit CDF row using exact SSP2E v1 arithmetic."""

    return _encode_detailed_generic(symbols, cdf_rows, _PRODUCTION_PARAMETERS).payload


def encode_detailed(symbols: ArrayLike, cdf_rows: ArrayLike) -> EncodedPayload:
    """Encode and expose the canonical physical bit length and selected frequencies."""

    return _encode_detailed_generic(symbols, cdf_rows, _PRODUCTION_PARAMETERS)


def _locate_symbol(row: NDArray[np.uint64], scaled: int) -> int:
    lower = 0
    upper = len(row) - 1
    while lower + 1 < upper:
        middle = (lower + upper) // 2
        if int(row[middle]) <= scaled:
            lower = middle
        else:
            upper = middle
    return lower


def _decode_generic(
    payload: bytes | bytearray | memoryview,
    cdf_rows: ArrayLike,
    count: int,
    parameters: _Parameters,
) -> NDArray[np.uint32]:
    parameters.validate()
    if isinstance(count, bool):
        raise ArithmeticCodingError("symbol count must be a nonnegative integer")
    try:
        count = operator.index(count)
    except TypeError as error:
        raise ArithmeticCodingError("symbol count must be a nonnegative integer") from error
    if count < 0:
        raise ArithmeticCodingError("symbol count must be a nonnegative integer")
    try:
        physical_payload = bytes(payload)
    except (TypeError, ValueError) as error:
        raise ArithmeticCodingError("payload must be bytes-like") from error
    if not physical_payload:
        raise ArithmeticCodingError("canonical arithmetic payload cannot be empty")
    rows = _prepare_rows(cdf_rows, count, parameters)
    reader = _BitReader(physical_payload)
    low = 0
    high = parameters.mask
    code = 0
    for _ in range(parameters.width):
        code = ((code << 1) & parameters.mask) | reader.read()

    decoded = np.empty(count, dtype=np.uint32)
    for index in range(count):
        if not low <= code <= high:
            raise ArithmeticCodingError("arithmetic code lies outside the current interval")
        interval_range = high - low + 1
        unit = interval_range // parameters.total
        if unit < 1:
            raise ArithmeticCodingError("arithmetic interval cannot represent the frequency total")
        scaled = (code - low) // unit
        if scaled >= parameters.total:
            raise ArithmeticCodingError("arithmetic code entered the unassigned interval tail")
        row = rows[index]
        symbol = _locate_symbol(row, scaled)
        if not (int(row[symbol]) <= scaled < int(row[symbol + 1])):
            raise ArithmeticCodingError("arithmetic code does not select a positive CDF interval")
        decoded[index] = symbol
        child_low = low + unit * int(row[symbol])
        child_high = low + unit * int(row[symbol + 1]) - 1
        low, high = child_low, child_high

        while True:
            if high < parameters.half:
                pass
            elif low >= parameters.half:
                low -= parameters.half
                high -= parameters.half
                code -= parameters.half
            elif low >= parameters.q1 and high < parameters.q3:
                low -= parameters.q1
                high -= parameters.q1
                code -= parameters.q1
            else:
                break
            low = (low << 1) & parameters.mask
            high = ((high << 1) & parameters.mask) | 1
            code = ((code << 1) & parameters.mask) | reader.read()

    canonical = _encode_detailed_generic(decoded, rows, parameters).payload
    if canonical != physical_payload:
        raise ArithmeticCodingError(
            "payload has malformed termination, padding, suffix, count, or model"
        )
    return decoded


def decode(
    payload: bytes | bytearray | memoryview,
    cdf_rows: ArrayLike,
    n: int,
) -> NDArray[np.uint32]:
    """Decode exactly ``n`` symbols and require canonical byte-identical re-encoding."""

    return _decode_generic(payload, cdf_rows, n, _PRODUCTION_PARAMETERS)


def _canonical_unsigned_bytes(value: int) -> bytes:
    if value < 0:
        raise ValueError("canonical unsigned integer cannot be negative")
    size = max(1, (value.bit_length() + 7) // 8)
    return value.to_bytes(size, byteorder="big", signed=False)


def _integer_sha256(value: int) -> str:
    return hashlib.sha256(_canonical_unsigned_bytes(value)).hexdigest()


def coder_certificate(
    symbols: ArrayLike,
    cdf_rows: ArrayLike,
    payload: bytes | bytearray | memoryview | None = None,
) -> dict[str, int | str | bool]:
    """Return the exact frozen frequency-product certificate for one stream.

    Integer hashes are SHA-256 over the minimal unsigned big-endian magnitude,
    with zero represented by one zero byte.
    """

    detailed = encode_detailed(symbols, cdf_rows)
    if payload is not None and bytes(payload) != detailed.payload:
        raise ArithmeticCodingError("certificate payload is not the canonical encoding")
    frequency_product = 1
    for frequency in detailed.actual_frequencies:
        frequency_product *= frequency
    left = frequency_product << (8 * len(detailed.payload))
    right = TOTAL_FREQUENCY ** len(detailed.actual_frequencies)
    if left < right:
        raise ArithmeticCodingError("exact frequency-product certificate failed")
    return {
        "certificate_version": 1,
        "symbol_count": len(detailed.actual_frequencies),
        "payload_bytes": len(detailed.payload),
        "physical_bit_length": detailed.bit_length,
        "frequency_product_bit_length": frequency_product.bit_length(),
        "frequency_product_byte_length": len(_canonical_unsigned_bytes(frequency_product)),
        "frequency_product_sha256": _integer_sha256(frequency_product),
        "left_bit_length": left.bit_length(),
        "left_byte_length": len(_canonical_unsigned_bytes(left)),
        "left_sha256": _integer_sha256(left),
        "right_bit_length": right.bit_length(),
        "right_byte_length": len(_canonical_unsigned_bytes(right)),
        "right_sha256": _integer_sha256(right),
        "inequality_holds": left >= right,
        "payload_sha256": hashlib.sha256(detailed.payload).hexdigest(),
        "integer_encoding": "minimal-unsigned-big-endian-zero-is-00",
    }


def _zero_padded_terminator_inside(details: EncodedPayload, width: int) -> bool:
    """Check the reduced-width zero-padded terminator tag against its final interval."""

    if width <= 0:
        raise ValueError("width must be positive")
    bits = list(details.termination_bits[:width])
    bits.extend([0] * (width - len(bits)))
    tag = 0
    for bit in bits:
        tag = (tag << 1) | bit
    return details.final_low <= tag <= details.final_high


NATIVE_SOURCE_PATH = Path(__file__).resolve().with_name("ssp2e_range_ext.cpp")


def build_native(
    output_path: Path,
    *,
    compiler: str | None = None,
    extra_flags: Sequence[str] = (),
) -> Path:
    """Build the frozen C++17 C ABI as a shared library at ``output_path``."""

    selected_compiler = compiler or os.environ.get("CXX") or shutil.which("c++")
    if not selected_compiler:
        raise RuntimeError("no C++ compiler was found")
    destination = Path(output_path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = [
        selected_compiler,
        "-std=c++17",
        "-O3",
        "-DNDEBUG",
        "-fPIC",
        "-shared",
        str(NATIVE_SOURCE_PATH),
        "-o",
        str(destination),
        *extra_flags,
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)
    return destination


class NativeArithmeticCoder:
    """Thin strict wrapper over the frozen C++17 reference-compatible C ABI."""

    _SUCCESS = 0
    _BUFFER_TOO_SMALL = 2

    def __init__(self, library_path: Path) -> None:
        self._library = ctypes.CDLL(str(Path(library_path).resolve()))
        self._size_repeated_abi = None
        self._size_repeated_abi_checked = False
        uint8_pointer = ctypes.POINTER(ctypes.c_uint8)
        uint32_pointer = ctypes.POINTER(ctypes.c_uint32)
        self._library.ssp2e_encode.argtypes = [
            uint32_pointer,
            uint32_pointer,
            ctypes.c_size_t,
            ctypes.c_size_t,
            uint8_pointer,
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_size_t),
        ]
        self._library.ssp2e_encode.restype = ctypes.c_int
        self._library.ssp2e_decode.argtypes = [
            uint8_pointer,
            ctypes.c_size_t,
            uint32_pointer,
            ctypes.c_size_t,
            ctypes.c_size_t,
            uint32_pointer,
        ]
        self._library.ssp2e_decode.restype = ctypes.c_int
        self._library.ssp2e_last_error.argtypes = []
        self._library.ssp2e_last_error.restype = ctypes.c_char_p

    def _resolve_size_repeated_abi(self):
        if not self._size_repeated_abi_checked:
            try:
                function = self._library.ssp2e_size_repeated
            except AttributeError:
                function = None
            if function is not None:
                uint32_pointer = ctypes.POINTER(ctypes.c_uint32)
                function.argtypes = [
                    uint32_pointer,
                    uint32_pointer,
                    ctypes.c_size_t,
                    ctypes.c_size_t,
                    ctypes.POINTER(ctypes.c_size_t),
                ]
                function.restype = ctypes.c_int
            self._size_repeated_abi = function
            self._size_repeated_abi_checked = True
        if self._size_repeated_abi is None:
            raise ArithmeticCodingError(
                "native library does not export ssp2e_size_repeated; "
                "rebuild the native arithmetic library"
            )
        return self._size_repeated_abi

    def _error(self, operation: str, status: int) -> ArithmeticCodingError:
        raw_message = self._library.ssp2e_last_error()
        message = raw_message.decode("utf-8", errors="replace") if raw_message else "unknown error"
        return ArithmeticCodingError(f"native {operation} failed ({status}): {message}")

    @staticmethod
    def _native_inputs(
        symbols: ArrayLike,
        cdf_rows: ArrayLike,
    ) -> tuple[NDArray[np.uint32], NDArray[np.uint32]]:
        symbol_array, rows = _prepare_encode_inputs(
            symbols,
            cdf_rows,
            _PRODUCTION_PARAMETERS,
        )
        return (
            np.ascontiguousarray(symbol_array, dtype=np.uint32),
            np.ascontiguousarray(rows, dtype=np.uint32),
        )

    def encode(self, symbols: ArrayLike, cdf_rows: ArrayLike) -> bytes:
        symbol_array, rows = self._native_inputs(symbols, cdf_rows)
        alphabet = rows.shape[1] - 1
        output_size = ctypes.c_size_t()
        status = self._library.ssp2e_encode(
            symbol_array.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            rows.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            len(symbol_array),
            alphabet,
            None,
            0,
            ctypes.byref(output_size),
        )
        if status != self._BUFFER_TOO_SMALL:
            raise self._error("encode sizing", status)
        output = np.empty(output_size.value, dtype=np.uint8)
        status = self._library.ssp2e_encode(
            symbol_array.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            rows.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            len(symbol_array),
            alphabet,
            output.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            len(output),
            ctypes.byref(output_size),
        )
        if status != self._SUCCESS or output_size.value != len(output):
            raise self._error("encode", status)
        return output.tobytes()

    def size_repeated(self, symbols: ArrayLike, cdf_row: ArrayLike) -> int:
        """Return exact encoded bytes for one CDF row repeated over all symbols."""

        symbol_array, row = _prepare_repeated_inputs(
            symbols,
            cdf_row,
            _PRODUCTION_PARAMETERS,
        )
        native_symbols = np.ascontiguousarray(symbol_array, dtype=np.uint32)
        native_row = np.ascontiguousarray(row, dtype=np.uint32)
        function = self._resolve_size_repeated_abi()
        output_size = ctypes.c_size_t()
        symbol_pointer = (
            native_symbols.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32))
            if len(native_symbols)
            else None
        )
        status = function(
            symbol_pointer,
            native_row.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            len(native_symbols),
            len(native_row) - 1,
            ctypes.byref(output_size),
        )
        if status != self._SUCCESS:
            raise self._error("repeated-row sizing", status)
        if output_size.value == 0:
            raise ArithmeticCodingError("native repeated-row sizing returned zero bytes")
        return int(output_size.value)

    def decode(
        self,
        payload: bytes | bytearray | memoryview,
        cdf_rows: ArrayLike,
        n: int,
    ) -> NDArray[np.uint32]:
        rows = _prepare_rows(cdf_rows, n, _PRODUCTION_PARAMETERS)
        native_rows = np.ascontiguousarray(rows, dtype=np.uint32)
        physical_payload = np.frombuffer(bytes(payload), dtype=np.uint8).copy()
        if not len(physical_payload):
            raise ArithmeticCodingError("canonical arithmetic payload cannot be empty")
        output = np.empty(n, dtype=np.uint32)
        status = self._library.ssp2e_decode(
            physical_payload.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            len(physical_payload),
            native_rows.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            n,
            native_rows.shape[1] - 1,
            output.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        )
        if status != self._SUCCESS:
            raise self._error("decode", status)
        return output
