"""Exact COMP-009 SSP2E-family containers, independent of the arithmetic coder.

The module owns framing, fixed-header semantics, model serialization, and CDF-row construction.
Arithmetic coding is deliberately injected through two tiny adapters::

    encode(symbols: uint32[N], cdfs: uint32[N, A + 1]) -> bytes
    decode(payload: bytes, cdfs: uint32[N, A + 1], n: int) -> uint32[N]

Only synthetic SSPL1 inputs should reach this module before the COMP-009 source-bound preflight.
The SSPL1 authority is the frozen :mod:`benchmarks.sgi_entropy_oracle` parser.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
import zlib
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import numpy as np

from benchmarks import sgi_entropy_oracle as oracle


OUTER = struct.Struct("<5sBBBQI")
HEADER = struct.Struct("<III4B4i12f5I")
DIRECTORY = struct.Struct("<IQQ")
HEAD_CHANNEL = struct.Struct("<8bh8bh")

VERSION = 1
FLAGS = 0
CDF_TABLE_ID = 1
ZLIB_LEVEL = 9
RENDERER_CUDA = 1
TOTAL_FREQUENCY = 32768

CONTEXTUAL_MAGICS = (b"SSP2E", b"SSP2S")
FACTORIZED_MAGICS = (b"SSP2L", b"SSP2F")
ALL_MAGICS = CONTEXTUAL_MAGICS + FACTORIZED_MAGICS
MODELED_CHANNELS = ("scale_x", "scale_y", "red", "green", "blue")
SYMBOL_CHANNELS = (
    "mean_x",
    "mean_y",
    "scale_x",
    "scale_y",
    "rotation",
    "red",
    "green",
    "blue",
)

CONTEXTUAL_STREAMS = (
    "means",
    "scale_x",
    "scale_y",
    "rotation",
    "red",
    "green",
    "blue",
    "grid",
    "head",
)
FACTORIZED_STREAMS = (
    "means",
    "scale_x",
    "scale_y",
    "rotation",
    "red",
    "green",
    "blue",
    "model",
)
CONTEXTUAL_IDS = tuple(range(1, 10))
FACTORIZED_IDS = (1, 2, 3, 4, 5, 6, 7, 10)
CONTEXTUAL_PREFIX_BYTES = OUTER.size + HEADER.size + 9 * DIRECTORY.size
FACTORIZED_PREFIX_BYTES = OUTER.size + HEADER.size + 8 * DIRECTORY.size
CONTEXTUAL_FIXED_BYTES = CONTEXTUAL_PREFIX_BYTES + 256 + 100

if OUTER.size != 20 or HEADER.size != 100 or DIRECTORY.size != 20:
    raise RuntimeError("COMP-009 struct layout drift")
if CONTEXTUAL_PREFIX_BYTES != 300 or FACTORIZED_PREFIX_BYTES != 280:
    raise RuntimeError("COMP-009 pre-payload offset drift")
if CONTEXTUAL_FIXED_BYTES != 656:
    raise RuntimeError("COMP-009 fixed contextual overhead drift")


class ContainerError(ValueError):
    """A COMP-009 container or injected-adapter contract was violated."""


ArithmeticEncode = Callable[[np.ndarray, np.ndarray], bytes]
ArithmeticDecode = Callable[[bytes, np.ndarray, int], np.ndarray]
FrequencyProvider = Callable[[int, int, int], Sequence[int]]


def _uint(value: object, name: str, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ContainerError(f"{name} must be an integer")
    result = int(value)
    if not 0 <= result <= maximum:
        raise ContainerError(f"{name} is outside [0, {maximum}]")
    return result


def _positive_uint(value: object, name: str, maximum: int) -> int:
    result = _uint(value, name, maximum)
    if result == 0:
        raise ContainerError(f"{name} must be positive")
    return result


def _float32(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ContainerError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or float(np.float32(result)) != result:
        raise ContainerError(f"{name} must be finite and exactly float32-representable")
    return result


def _tuple_values(value: object, size: int, name: str) -> tuple[object, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != size:
        raise ContainerError(f"{name} must contain exactly {size} values")
    return tuple(value)


def _packed_itemsize(bits: int) -> int:
    if not 1 <= bits <= 32:
        raise ContainerError("bit depth must lie in [1, 32]")
    return 1 if bits <= 8 else 2 if bits <= 16 else 4


@dataclass(frozen=True)
class ContainerHeader:
    """The semantic view of the frozen 100-byte SSP2 header."""

    n: int
    height: int
    width: int
    bits: tuple[int, int, int, int]
    means_lo: tuple[int, int]
    means_hi: tuple[int, int]
    scale_lo: tuple[float, float]
    scale_hi: tuple[float, float]
    color_lo: tuple[float, float, float]
    color_hi: tuple[float, float, float]
    aa_dilation: float
    sigma_cutoff: float
    render_chunk: int
    renderer_id: int = RENDERER_CUDA
    support_fade: int = 0
    cdf_table_id: int = CDF_TABLE_ID
    zlib_level: int = ZLIB_LEVEL

    def validated(self) -> ContainerHeader:
        """Return ``self`` after enforcing every frozen header invariant."""
        _positive_uint(self.n, "n", (1 << 32) - 1)
        _positive_uint(self.height, "height", (1 << 32) - 1)
        _positive_uint(self.width, "width", (1 << 32) - 1)
        bits = tuple(_positive_uint(value, f"bits[{i}]", 255) for i, value in enumerate(self.bits))
        if bits not in oracle.FROZEN_BIT_TUPLES:
            raise ContainerError(f"unsupported COMP-009 bit tuple: {bits}")
        for name, endpoints in (("means_lo", self.means_lo), ("means_hi", self.means_hi)):
            if len(endpoints) != 2:
                raise ContainerError(f"{name} must contain two values")
            for index, value in enumerate(endpoints):
                integer = int(value)
                if (
                    isinstance(value, bool)
                    or integer != value
                    or not -(1 << 31) <= integer < 1 << 31
                ):
                    raise ContainerError(f"{name}[{index}] must be an int32")
        if (
            self.means_lo[0] > 0
            or self.means_lo[1] > 0
            or self.means_hi[0] < self.width - 1
            or self.means_hi[1] < self.height - 1
            or self.means_lo[0] > self.means_hi[0]
            or self.means_lo[1] > self.means_hi[1]
        ):
            raise ContainerError("mean extent does not contain the image box")
        float_groups = (
            ("scale_lo", self.scale_lo, 2),
            ("scale_hi", self.scale_hi, 2),
            ("color_lo", self.color_lo, 3),
            ("color_hi", self.color_hi, 3),
        )
        for name, values, expected in float_groups:
            if len(values) != expected:
                raise ContainerError(f"{name} must contain {expected} values")
            for index, value in enumerate(values):
                _float32(value, f"{name}[{index}]")
        _float32(self.aa_dilation, "aa_dilation")
        _float32(self.sigma_cutoff, "sigma_cutoff")
        _positive_uint(self.render_chunk, "render_chunk", (1 << 32) - 1)
        if self.renderer_id != RENDERER_CUDA:
            raise ContainerError("renderer_id must identify the frozen CUDA renderer")
        if self.support_fade != 0:
            raise ContainerError("support_fade must be zero")
        if self.cdf_table_id != CDF_TABLE_ID:
            raise ContainerError("cdf_table_id must be one")
        if self.zlib_level != ZLIB_LEVEL:
            raise ContainerError("zlib_level must be nine")
        return self

    @classmethod
    def from_sspl1(cls, header: Mapping[str, object]) -> ContainerHeader:
        """Convert frozen-parser SSPL1 semantics without silently dropping a flag."""
        try:
            oracle._validate_ssp2e_header_representability(header)  # noqa: SLF001
        except (oracle.OracleError, TypeError, ValueError) as exc:
            raise ContainerError(f"SSPL1 header is not SSP2-representable: {exc}") from exc
        fixed_flags = {
            "morton": True,
            "color_delta": True,
            "means_byte_planar": True,
            "rot_circular": True,
            "stream_codec": "zlib",
            "has_opacity": False,
            "bits_opacity": 8,
            "renderer": "cuda",
            "support_fade": False,
        }
        for name, expected in fixed_flags.items():
            if header.get(name) != expected:
                raise ContainerError(f"SSPL1 {name!r} must equal {expected!r}")
        bits_raw = _tuple_values(header.get("bits"), 4, "bits")
        means_lo_raw = _tuple_values(header.get("means_lo"), 2, "means_lo")
        means_hi_raw = _tuple_values(header.get("means_hi"), 2, "means_hi")

        def floats(name: str, size: int) -> tuple[float, ...]:
            return tuple(
                _float32(value, f"{name}[{index}]")
                for index, value in enumerate(_tuple_values(header.get(name), size, name))
            )

        result = cls(
            n=_positive_uint(header.get("n"), "n", (1 << 32) - 1),
            height=_positive_uint(header.get("H"), "H", (1 << 32) - 1),
            width=_positive_uint(header.get("W"), "W", (1 << 32) - 1),
            bits=tuple(_positive_uint(value, "bits", 255) for value in bits_raw),  # type: ignore[arg-type]
            means_lo=tuple(int(value) for value in means_lo_raw),  # type: ignore[arg-type]
            means_hi=tuple(int(value) for value in means_hi_raw),  # type: ignore[arg-type]
            scale_lo=floats("scale_lo", 2),  # type: ignore[arg-type]
            scale_hi=floats("scale_hi", 2),  # type: ignore[arg-type]
            color_lo=floats("color_lo", 3),  # type: ignore[arg-type]
            color_hi=floats("color_hi", 3),  # type: ignore[arg-type]
            aa_dilation=_float32(header.get("aa_dilation"), "aa_dilation"),
            sigma_cutoff=_float32(header.get("sigma_cutoff"), "sigma_cutoff"),
            render_chunk=_positive_uint(header.get("render_chunk"), "render_chunk", (1 << 32) - 1),
        ).validated()
        expected_lengths = result.raw_stream_lengths()
        if header.get("stream_raw_lengths") != list(expected_lengths):
            raise ContainerError("SSPL1 raw-stream lengths disagree with fixed header semantics")
        return result

    def raw_stream_lengths(self) -> tuple[int, int, int, int]:
        bm, bs, br, bc = self.bits
        return (
            2 * self.n * _packed_itemsize(bm),
            2 * self.n * _packed_itemsize(bs),
            self.n * _packed_itemsize(br),
            3 * self.n * _packed_itemsize(bc),
        )

    def to_sspl1_header(self) -> dict[str, object]:
        """Reconstruct the exact production SSPL1 semantic header in canonical key order."""
        self.validated()
        return {
            "n": self.n,
            "H": self.height,
            "W": self.width,
            "bits": list(self.bits),
            "morton": True,
            "color_delta": True,
            "means_byte_planar": True,
            "rot_circular": True,
            "stream_codec": "zlib",
            "stream_raw_lengths": list(self.raw_stream_lengths()),
            "color_lo": list(self.color_lo),
            "color_hi": list(self.color_hi),
            "scale_lo": list(self.scale_lo),
            "scale_hi": list(self.scale_hi),
            "means_lo": [float(value) for value in self.means_lo],
            "means_hi": [float(value) for value in self.means_hi],
            "has_opacity": False,
            "bits_opacity": 8,
            "renderer": "cuda",
            "aa_dilation": self.aa_dilation,
            "sigma_cutoff": self.sigma_cutoff,
            "support_fade": False,
            "render_chunk": self.render_chunk,
        }

    def to_bytes(self) -> bytes:
        self.validated()
        values = (
            self.n,
            self.height,
            self.width,
            *self.bits,
            *self.means_lo,
            *self.means_hi,
            *self.scale_lo,
            *self.scale_hi,
            *self.color_lo,
            *self.color_hi,
            self.aa_dilation,
            self.sigma_cutoff,
            self.render_chunk,
            self.renderer_id,
            self.support_fade,
            self.cdf_table_id,
            self.zlib_level,
        )
        try:
            return HEADER.pack(*values)
        except struct.error as exc:  # pragma: no cover - guarded field by field above
            raise ContainerError(f"cannot serialize fixed header: {exc}") from exc

    @classmethod
    def from_bytes(cls, payload: bytes) -> ContainerHeader:
        if len(payload) != HEADER.size:
            raise ContainerError("fixed header must contain exactly 100 bytes")
        values = HEADER.unpack(payload)
        result = cls(
            n=values[0],
            height=values[1],
            width=values[2],
            bits=tuple(values[3:7]),  # type: ignore[arg-type]
            means_lo=tuple(values[7:9]),  # type: ignore[arg-type]
            means_hi=tuple(values[9:11]),  # type: ignore[arg-type]
            scale_lo=tuple(values[11:13]),  # type: ignore[arg-type]
            scale_hi=tuple(values[13:15]),  # type: ignore[arg-type]
            color_lo=tuple(values[15:18]),  # type: ignore[arg-type]
            color_hi=tuple(values[18:21]),  # type: ignore[arg-type]
            aa_dilation=values[21],
            sigma_cutoff=values[22],
            render_chunk=values[23],
            renderer_id=values[24],
            support_fade=values[25],
            cdf_table_id=values[26],
            zlib_level=values[27],
        )
        return result.validated()


@dataclass(frozen=True)
class SSPL1Parts:
    """Strictly extracted SSPL1 source state needed by the SSP2 family."""

    header: ContainerHeader
    original_header: dict[str, object]
    symbols: dict[str, np.ndarray]
    payloads: dict[str, bytes]
    blob_sha256: str
    symbol_sha256: str
    total_bytes: int


@dataclass(frozen=True)
class DirectoryEntry:
    stream_id: int
    name: str
    offset: int
    length: int


@dataclass(frozen=True)
class ParsedContainer:
    magic: bytes
    header: ContainerHeader
    entries: tuple[DirectoryEntry, ...]
    payloads: dict[str, bytes]
    total_bytes: int
    crc32: int

    @property
    def prefix_bytes(self) -> int:
        return (
            CONTEXTUAL_PREFIX_BYTES if self.magic in CONTEXTUAL_MAGICS else FACTORIZED_PREFIX_BYTES
        )

    @property
    def payload_bytes(self) -> int:
        return sum(entry.length for entry in self.entries)


@dataclass(frozen=True)
class DecodedContainer:
    parsed: ParsedContainer
    symbols: dict[str, np.ndarray]
    symbol_sha256: str


@dataclass(frozen=True)
class HeadChannel:
    location_weights: tuple[int, ...]
    location_bias: int
    scale_weights: tuple[int, ...]
    scale_bias: int


def _symbol_sha256(symbols: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in SYMBOL_CHANNELS:
        values = np.ascontiguousarray(symbols[name], dtype="<u4")
        digest.update(name.encode("ascii") + b"\0")
        digest.update(struct.pack("<Q", int(values.size)))
        digest.update(values.tobytes(order="C"))
    return digest.hexdigest()


def _strict_zlib(payload: bytes, expected_bytes: int, name: str) -> bytes:
    decoder = zlib.decompressobj()
    try:
        raw = decoder.decompress(payload, expected_bytes + 1)
        if len(raw) > expected_bytes or decoder.unconsumed_tail:
            raise ContainerError(f"{name} expands beyond its declared raw length")
        raw += decoder.flush()
    except zlib.error as exc:
        raise ContainerError(f"invalid {name} zlib member") from exc
    if len(raw) != expected_bytes:
        raise ContainerError(f"{name} raw length mismatch: {len(raw)} != {expected_bytes}")
    if not decoder.eof or decoder.unused_data or decoder.unconsumed_tail:
        raise ContainerError(f"{name} is not exactly one zlib member")
    if zlib.compress(raw, level=ZLIB_LEVEL) != payload:
        raise ContainerError(f"{name} is not canonical local zlib-9")
    return raw


def extract_sspl1_parts(blob: bytes) -> SSPL1Parts:
    """Extract exact payload bytes only after the frozen oracle parser accepts the source."""
    try:
        parsed = oracle.parse_sspl1(blob, canonical_zlib9=True)
    except oracle.OracleError as exc:
        raise ContainerError(f"invalid SSPL1 source: {exc}") from exc
    header = ContainerHeader.from_sspl1(parsed.header)
    reconstructed = header.to_sspl1_header()
    if json.dumps(reconstructed).encode("utf-8") != json.dumps(parsed.header).encode("utf-8"):
        raise ContainerError("fixed header cannot reproduce exact SSPL1 header semantics")

    (header_length,) = struct.unpack_from("<I", blob, 5)
    offset = 9 + int(header_length)
    payloads: dict[str, bytes] = {}
    for name in ("means", "scales", "rotation", "colors"):
        (length,) = struct.unpack_from("<I", blob, offset)
        offset += 4
        payloads[name] = blob[offset : offset + int(length)]
        offset += int(length)
        if len(payloads[name]) != parsed.payload_bytes[name]:
            raise ContainerError(f"frozen parser accounting drift for {name}")
    if offset != len(blob):  # pragma: no cover - frozen parser already proves this
        raise ContainerError("frozen parser and exact payload walk disagree")
    symbols = {
        name: np.ascontiguousarray(values, dtype=np.uint32)
        for name, values in parsed.symbols.items()
    }
    return SSPL1Parts(
        header=header,
        original_header=dict(parsed.header),
        symbols=symbols,
        payloads=payloads,
        blob_sha256=parsed.blob_sha256,
        symbol_sha256=parsed.symbol_sha256,
        total_bytes=parsed.total_bytes,
    )


def parse_head(payload: bytes) -> tuple[HeadChannel, ...]:
    if not isinstance(payload, bytes) or len(payload) != 100:
        raise ContainerError("head must contain exactly 100 bytes")
    channels: list[HeadChannel] = []
    for index in range(5):
        values = HEAD_CHANNEL.unpack_from(payload, index * HEAD_CHANNEL.size)
        channels.append(
            HeadChannel(
                location_weights=tuple(values[:8]),
                location_bias=values[8],
                scale_weights=tuple(values[9:17]),
                scale_bias=values[17],
            )
        )
    return tuple(channels)


def serialize_head(channels: Sequence[HeadChannel]) -> bytes:
    if len(channels) != 5:
        raise ContainerError("head must contain exactly five channel records")
    payload = bytearray()
    for channel in channels:
        if len(channel.location_weights) != 8 or len(channel.scale_weights) != 8:
            raise ContainerError("each head weight vector must contain eight values")
        try:
            payload.extend(
                HEAD_CHANNEL.pack(
                    *channel.location_weights,
                    channel.location_bias,
                    *channel.scale_weights,
                    channel.scale_bias,
                )
            )
        except struct.error as exc:
            raise ContainerError(
                f"head parameter is outside its signed storage range: {exc}"
            ) from exc
    result = bytes(payload)
    if parse_head(result) != tuple(channels):
        raise ContainerError("head serialization did not round-trip")
    return result


def infer_head_rows(
    grid: bytes, head: bytes, cells: np.ndarray
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Infer exact signed-int32 ``(loc, scale)`` rows in frozen channel order."""
    if not isinstance(grid, bytes) or len(grid) != 256:
        raise ContainerError("grid must contain exactly 256 bytes")
    cell_values = np.asarray(cells)
    if cell_values.ndim != 1 or cell_values.dtype.kind not in "ui":
        raise ContainerError("cells must be a one-dimensional integer array")
    if cell_values.size and (int(cell_values.min()) < 0 or int(cell_values.max()) >= 256):
        raise ContainerError("cell index exceeds the 16x16 grid")
    selected = np.frombuffer(grid, dtype=np.uint8)[cell_values.astype(np.int64)]
    shifts = np.arange(8, dtype=np.uint8)
    phi = (2 * ((selected[:, None] >> shifts[None, :]) & 1).astype(np.int32) - 1).astype(np.int32)
    result: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, parameters in zip(MODELED_CHANNELS, parse_head(head), strict=True):
        location = phi @ np.asarray(parameters.location_weights, dtype=np.int32)
        location += np.int32(parameters.location_bias)
        scale = phi @ np.asarray(parameters.scale_weights, dtype=np.int32)
        scale += np.int32(parameters.scale_bias)
        result[name] = (
            np.clip(location, 0, 63).astype(np.uint8),
            np.clip(scale, 0, 15).astype(np.uint8),
        )
    return result


def _coerce_frequencies(values: Sequence[int], alphabet: int, name: str) -> np.ndarray:
    sequence = list(values)
    if len(sequence) != alphabet:
        raise ContainerError(f"{name} has {len(sequence)} frequencies, expected {alphabet}")
    result = np.empty(alphabet, dtype=np.uint32)
    total = 0
    for symbol, value in enumerate(sequence):
        frequency = _positive_uint(value, f"{name}[{symbol}]", 65535)
        result[symbol] = frequency
        total += frequency
    if total != TOTAL_FREQUENCY:
        raise ContainerError(f"{name} frequencies sum to {total}, expected {TOTAL_FREQUENCY}")
    return result


def cdf_rows_from_models(
    alphabet: int,
    locations: np.ndarray,
    scales: np.ndarray,
    frequency_provider: FrequencyProvider,
) -> np.ndarray:
    """Materialize exact uint32 CDF rows through the injected static-table provider."""
    if alphabet not in (64, 256):
        raise ContainerError(f"unsupported static-table alphabet: {alphabet}")
    loc = np.asarray(locations)
    scale = np.asarray(scales)
    if (
        loc.ndim != 1
        or scale.shape != loc.shape
        or loc.dtype.kind not in "ui"
        or scale.dtype.kind not in "ui"
    ):
        raise ContainerError("location and scale rows must be same-shaped integer vectors")
    if loc.size and (
        int(loc.min()) < 0 or int(scale.min()) < 0 or int(loc.max()) > 63 or int(scale.max()) > 15
    ):
        raise ContainerError("static-table row index is out of range")
    # At most 64*16 distinct table rows exist.  Construct each occupied row once and gather the
    # N rows with NumPy indexing; a Python loop over all 8192 symbols made the reference decoder
    # measure interpreter overhead rather than the native arithmetic core.
    flat_keys = loc.astype(np.uint16, copy=False) * 16 + scale.astype(np.uint16, copy=False)
    occupied, inverse = np.unique(flat_keys, return_inverse=True)
    unique_rows = np.empty((occupied.size, alphabet + 1), dtype=np.uint32)
    unique_rows[:, 0] = 0
    for index, flat_key in enumerate(occupied.tolist()):
        location, scale_index = divmod(int(flat_key), 16)
        frequencies = _coerce_frequencies(
            frequency_provider(alphabet, location, scale_index),
            alphabet,
            f"static CDF A={alphabet},loc={location},scale={scale_index}",
        )
        np.cumsum(frequencies, dtype=np.uint32, out=unique_rows[index, 1:])
    return np.ascontiguousarray(unique_rows[inverse])


def empirical_frequencies(symbols: np.ndarray, alphabet: int) -> np.ndarray:
    """Frozen positive-frequency normalization for one SSP2F channel."""
    values = np.asarray(symbols)
    if values.ndim != 1 or values.dtype.kind not in "ui" or values.size == 0:
        raise ContainerError("empirical symbols must be a nonempty integer vector")
    if alphabet not in (64, 256) or int(values.min()) < 0 or int(values.max()) >= alphabet:
        raise ContainerError("empirical symbol exceeds its frozen alphabet")
    counts = np.bincount(values.astype(np.int64), minlength=alphabet)
    return empirical_frequencies_from_counts(counts, alphabet)


def empirical_frequencies_from_counts(counts: np.ndarray, alphabet: int) -> np.ndarray:
    """Normalize one exact nonempty histogram with the frozen largest-remainder rule."""
    if alphabet not in (64, 256):
        raise ContainerError(f"unsupported empirical alphabet: {alphabet}")
    raw = np.asarray(counts)
    if raw.shape != (alphabet,) or raw.dtype.kind not in "iu":
        raise ContainerError(f"empirical counts must be an integer vector of length {alphabet}")
    if raw.dtype.kind == "i" and np.any(raw < 0):
        raise ContainerError("empirical counts contain a negative value")
    counts64 = np.ascontiguousarray(raw, dtype=np.uint64)
    sample_count = sum(int(value) for value in counts64)
    if not 1 <= sample_count < 1 << 32:
        raise ContainerError("empirical counts must sum to a positive uint32 row count")
    residual_total = TOTAL_FREQUENCY - alphabet
    products = np.uint64(residual_total) * counts64
    allocation = products // np.uint64(sample_count)
    remainders = products % np.uint64(sample_count)
    remaining = residual_total - int(allocation.sum())
    order = sorted(range(alphabet), key=lambda symbol: (-int(remainders[symbol]), symbol))
    if remaining:
        allocation[np.asarray(order[:remaining], dtype=np.int64)] += 1
    result = np.ascontiguousarray(allocation + 1, dtype=np.uint32)
    if np.any(result == 0) or int(result.sum()) != TOTAL_FREQUENCY:
        raise ContainerError("empirical normalization invariant failed")
    return result


def serialize_empirical_frequencies(
    frequencies: Mapping[str, Sequence[int] | np.ndarray],
    header: ContainerHeader,
) -> bytes:
    """Serialize validated five-channel SSP2F frequencies in canonical channel order."""
    header.validated()
    if set(frequencies) != set(MODELED_CHANNELS):
        raise ContainerError("SSP2F model must define exactly the five modeled channels")
    raw = bytearray()
    for name in MODELED_CHANNELS:
        alphabet = _channel_alphabet(header, name)
        normalized = _coerce_frequencies(frequencies[name], alphabet, f"SSP2F {name}")
        raw.extend(normalized.astype("<u2", copy=False).tobytes(order="C"))
    return zlib.compress(bytes(raw), level=ZLIB_LEVEL)


def serialize_empirical_model(symbols: Mapping[str, np.ndarray], header: ContainerHeader) -> bytes:
    frequencies: dict[str, np.ndarray] = {}
    for name in MODELED_CHANNELS:
        alphabet = _channel_alphabet(header, name)
        frequencies[name] = empirical_frequencies(symbols[name], alphabet)
    return serialize_empirical_frequencies(frequencies, header)


def parse_empirical_model(payload: bytes, header: ContainerHeader) -> dict[str, np.ndarray]:
    expected = sum(_channel_alphabet(header, name) * 2 for name in MODELED_CHANNELS)
    raw = _strict_zlib(payload, expected, "SSP2F model")
    result: dict[str, np.ndarray] = {}
    offset = 0
    for name in MODELED_CHANNELS:
        alphabet = _channel_alphabet(header, name)
        end = offset + 2 * alphabet
        values = np.frombuffer(raw[offset:end], dtype="<u2").astype(np.uint32)
        result[name] = _coerce_frequencies(values.tolist(), alphabet, f"SSP2F {name}")
        offset = end
    if offset != len(raw):  # pragma: no cover - expected length and loop are fixed together
        raise ContainerError("SSP2F model accounting drift")
    return result


def serialize_global_model(models: Mapping[str, tuple[int, int]]) -> bytes:
    payload = bytearray()
    if set(models) != set(MODELED_CHANNELS):
        raise ContainerError("SSP2L model must define exactly the five modeled channels")
    for name in MODELED_CHANNELS:
        location, scale = models[name]
        payload.extend(
            bytes(
                (
                    _uint(location, f"{name}.location", 63),
                    _uint(scale, f"{name}.scale", 15),
                )
            )
        )
    return bytes(payload)


def parse_global_model(payload: bytes) -> dict[str, tuple[int, int]]:
    if not isinstance(payload, bytes) or len(payload) != 10:
        raise ContainerError("SSP2L model must contain exactly ten bytes")
    result: dict[str, tuple[int, int]] = {}
    for index, name in enumerate(MODELED_CHANNELS):
        location, scale = payload[2 * index : 2 * index + 2]
        if location > 63 or scale > 15:
            raise ContainerError(f"SSP2L {name} model index is out of range")
        result[name] = (location, scale)
    return result


def shuffled_permutation(n: int) -> np.ndarray:
    """Derive the frozen SHA-256 row permutation (production uses exactly ``n=8192``)."""
    _positive_uint(n, "shuffle row count", (1 << 32) - 1)
    domain = b"structsplat-comp009-shuffle-v1\0"
    keyed = [
        (hashlib.sha256(domain + struct.pack("<I", index)).digest(), index) for index in range(n)
    ]
    return np.asarray([index for _, index in sorted(keyed)], dtype=np.uint32)


def _channel_alphabet(header: ContainerHeader, name: str) -> int:
    if name in ("scale_x", "scale_y"):
        return 1 << header.bits[1]
    if name in ("red", "green", "blue"):
        return 1 << header.bits[3]
    raise ContainerError(f"{name!r} is not an arithmetic channel")


def _symbol_vector(values: object, n: int, alphabet: int, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.shape != (n,) or array.dtype.kind not in "ui":
        raise ContainerError(f"{name} decoder output must be an integer vector of length {n}")
    if array.size and (int(array.min()) < 0 or int(array.max()) >= alphabet):
        raise ContainerError(f"{name} decoder output exceeds alphabet {alphabet}")
    return np.ascontiguousarray(array, dtype=np.uint32)


def _assemble(magic: bytes, header: ContainerHeader, payloads: Mapping[str, bytes]) -> bytes:
    if magic in CONTEXTUAL_MAGICS:
        names, ids, prefix = CONTEXTUAL_STREAMS, CONTEXTUAL_IDS, CONTEXTUAL_PREFIX_BYTES
    elif magic in FACTORIZED_MAGICS:
        names, ids, prefix = FACTORIZED_STREAMS, FACTORIZED_IDS, FACTORIZED_PREFIX_BYTES
    else:
        raise ContainerError(f"unknown SSP2 magic: {magic!r}")
    if set(payloads) != set(names):
        raise ContainerError("payload mapping differs from the frozen stream set")
    ordered: list[bytes] = []
    for name in names:
        payload = payloads[name]
        if not isinstance(payload, bytes) or not payload:
            raise ContainerError(f"{name} payload must be nonempty immutable bytes")
        ordered.append(payload)
    directory = bytearray()
    offset = prefix
    for stream_id, payload in zip(ids, ordered, strict=True):
        directory.extend(DIRECTORY.pack(stream_id, offset, len(payload)))
        offset += len(payload)
    body = header.to_bytes() + bytes(directory) + b"".join(ordered)
    total_bytes = OUTER.size + len(body)
    crc = zlib.crc32(body) & 0xFFFFFFFF
    blob = OUTER.pack(magic, VERSION, FLAGS, len(names), total_bytes, crc) + body
    if len(blob) != total_bytes:  # pragma: no cover - arithmetic identity
        raise ContainerError("container byte accounting drift")
    return blob


def parse_container(blob: bytes, *, reference_sspl1: bytes | None = None) -> ParsedContainer:
    """Strictly parse one complete SSP2E/SSP2S/SSP2L/SSP2F byte string."""
    if not isinstance(blob, bytes):
        raise ContainerError("container input must be immutable bytes")
    if len(blob) < OUTER.size + HEADER.size:
        raise ContainerError("truncated SSP2 container")
    magic, version, flags, stream_count, total_bytes, stored_crc = OUTER.unpack_from(blob)
    if magic not in ALL_MAGICS:
        raise ContainerError("unrecognized SSP2 magic")
    if version != VERSION or flags != FLAGS:
        raise ContainerError("unsupported SSP2 version or flags")
    names = CONTEXTUAL_STREAMS if magic in CONTEXTUAL_MAGICS else FACTORIZED_STREAMS
    ids = CONTEXTUAL_IDS if magic in CONTEXTUAL_MAGICS else FACTORIZED_IDS
    prefix = CONTEXTUAL_PREFIX_BYTES if magic in CONTEXTUAL_MAGICS else FACTORIZED_PREFIX_BYTES
    if stream_count != len(names):
        raise ContainerError("stream count disagrees with container magic")
    if total_bytes != len(blob):
        raise ContainerError("outer total length disagrees with physical bytes")
    observed_crc = zlib.crc32(blob[OUTER.size :]) & 0xFFFFFFFF
    if stored_crc != observed_crc:
        raise ContainerError("outer CRC-32 mismatch")
    header = ContainerHeader.from_bytes(blob[OUTER.size : OUTER.size + HEADER.size])

    entries: list[DirectoryEntry] = []
    payloads: dict[str, bytes] = {}
    expected_offset = prefix
    directory_offset = OUTER.size + HEADER.size
    for index, (expected_id, name) in enumerate(zip(ids, names, strict=True)):
        stream_id, offset, length = DIRECTORY.unpack_from(
            blob, directory_offset + index * DIRECTORY.size
        )
        if stream_id != expected_id:
            raise ContainerError(f"directory ID/order mismatch for {name}")
        if offset != expected_offset:
            raise ContainerError(f"directory gap, overlap, alias, or offset mismatch for {name}")
        if length == 0:
            raise ContainerError(f"empty {name} payload is forbidden")
        end = offset + length
        if end > len(blob):
            raise ContainerError(f"truncated {name} payload")
        payload = blob[offset:end]
        entries.append(DirectoryEntry(stream_id, name, offset, length))
        payloads[name] = payload
        expected_offset = end
    if expected_offset != len(blob):
        raise ContainerError("container has a suffix or unclaimed trailing bytes")

    means_bytes, _, rotation_bytes, _ = header.raw_stream_lengths()
    _strict_zlib(payloads["means"], means_bytes, "means")
    _strict_zlib(payloads["rotation"], rotation_bytes, "rotation")
    if magic in CONTEXTUAL_MAGICS:
        if len(payloads["grid"]) != 256:
            raise ContainerError("contextual grid must contain exactly 256 bytes")
        parse_head(payloads["head"])
    elif magic == b"SSP2L":
        parse_global_model(payloads["model"])
    else:
        parse_empirical_model(payloads["model"], header)

    result = ParsedContainer(
        magic=magic,
        header=header,
        entries=tuple(entries),
        payloads=payloads,
        total_bytes=len(blob),
        crc32=stored_crc,
    )
    if result.prefix_bytes + result.payload_bytes != result.total_bytes:
        raise ContainerError("complete container accounting mismatch")
    if reference_sspl1 is not None:
        source = extract_sspl1_parts(reference_sspl1)
        if header != source.header:
            raise ContainerError("fixed header differs from the reference SSPL1 semantics")
        if payloads["means"] != source.payloads["means"]:
            raise ContainerError("means payload is not byte-identical to reference SSPL1")
        if payloads["rotation"] != source.payloads["rotation"]:
            raise ContainerError("rotation payload is not byte-identical to reference SSPL1")
    return result


def _context_cells(parts: SSPL1Parts, magic: bytes) -> np.ndarray:
    cells = oracle.position_cells(
        parts.symbols["mean_x"], parts.symbols["mean_y"], parts.header.bits[0]
    )
    if magic == b"SSP2S":
        cells = cells[shuffled_permutation(parts.header.n)]
    return cells


def _encode_arithmetic(
    symbols: np.ndarray, cdfs: np.ndarray, encoder: ArithmeticEncode, name: str
) -> bytes:
    payload = encoder(np.ascontiguousarray(symbols, dtype=np.uint32), cdfs)
    if not isinstance(payload, bytes) or not payload:
        raise ContainerError(f"arithmetic encoder returned invalid {name} payload")
    return payload


def encode_contextual(
    sspl1_blob: bytes,
    grid: bytes,
    head: bytes,
    *,
    arithmetic_encode: ArithmeticEncode,
    frequency_provider: FrequencyProvider,
    magic: bytes = b"SSP2E",
) -> bytes:
    """Encode exact SSP2E or shuffled-control SSP2S framing from one canonical SSPL1."""
    if magic not in CONTEXTUAL_MAGICS:
        raise ContainerError("contextual encoder magic must be SSP2E or SSP2S")
    if not isinstance(grid, bytes) or len(grid) != 256:
        raise ContainerError("grid must contain exactly 256 bytes")
    parse_head(head)
    parts = extract_sspl1_parts(sspl1_blob)
    cells = _context_cells(parts, magic)
    model_rows = infer_head_rows(grid, head, cells)
    payloads: dict[str, bytes] = {
        "means": parts.payloads["means"],
        "rotation": parts.payloads["rotation"],
        "grid": grid,
        "head": head,
    }
    for name in MODELED_CHANNELS:
        alphabet = _channel_alphabet(parts.header, name)
        location, scale = model_rows[name]
        cdfs = cdf_rows_from_models(alphabet, location, scale, frequency_provider)
        payloads[name] = _encode_arithmetic(parts.symbols[name], cdfs, arithmetic_encode, name)
    blob = _assemble(magic, parts.header, payloads)
    parse_container(blob, reference_sspl1=sspl1_blob)
    return blob


def encode_ssp2l(
    sspl1_blob: bytes,
    models: Mapping[str, tuple[int, int]],
    *,
    arithmetic_encode: ArithmeticEncode,
    frequency_provider: FrequencyProvider,
) -> bytes:
    """Encode the complete static-table factorized comparator."""
    parts = extract_sspl1_parts(sspl1_blob)
    model_payload = serialize_global_model(models)
    normalized_models = parse_global_model(model_payload)
    payloads: dict[str, bytes] = {
        "means": parts.payloads["means"],
        "rotation": parts.payloads["rotation"],
        "model": model_payload,
    }
    for name in MODELED_CHANNELS:
        alphabet = _channel_alphabet(parts.header, name)
        location, scale = normalized_models[name]
        locations = np.full(parts.header.n, location, dtype=np.uint8)
        scales = np.full(parts.header.n, scale, dtype=np.uint8)
        cdfs = cdf_rows_from_models(alphabet, locations, scales, frequency_provider)
        payloads[name] = _encode_arithmetic(parts.symbols[name], cdfs, arithmetic_encode, name)
    blob = _assemble(b"SSP2L", parts.header, payloads)
    parse_container(blob, reference_sspl1=sspl1_blob)
    return blob


def _encode_ssp2f_parts(
    parts: SSPL1Parts,
    symbols: Mapping[str, np.ndarray],
    arithmetic_encode: ArithmeticEncode,
) -> bytes:
    model_payload = serialize_empirical_model(symbols, parts.header)
    frequencies = parse_empirical_model(model_payload, parts.header)
    payloads: dict[str, bytes] = {
        "means": parts.payloads["means"],
        "rotation": parts.payloads["rotation"],
        "model": model_payload,
    }
    for name in MODELED_CHANNELS:
        cdf = np.concatenate(
            (np.zeros(1, dtype=np.uint32), np.cumsum(frequencies[name], dtype=np.uint32))
        )
        cdfs = np.repeat(cdf[None, :], parts.header.n, axis=0)
        values = _symbol_vector(
            symbols[name],
            parts.header.n,
            _channel_alphabet(parts.header, name),
            name,
        )
        payloads[name] = _encode_arithmetic(values, cdfs, arithmetic_encode, name)
    return _assemble(b"SSP2F", parts.header, payloads)


def encode_ssp2f(sspl1_blob: bytes, *, arithmetic_encode: ArithmeticEncode) -> bytes:
    """Encode the complete empirical-frequency factorized comparator."""
    parts = extract_sspl1_parts(sspl1_blob)
    blob = _encode_ssp2f_parts(parts, parts.symbols, arithmetic_encode)
    parse_container(blob, reference_sspl1=sspl1_blob)
    return blob


def encode_ssp2f_rgb_override(
    sspl1_blob: bytes,
    rgb_symbols: np.ndarray,
    *,
    arithmetic_encode: ArithmeticEncode,
    arithmetic_decode: ArithmeticDecode,
) -> bytes:
    """Encode absolute RGB overrides while proving every non-RGB SSP2F invariant."""
    parts = extract_sspl1_parts(sspl1_blob)
    rgb = np.asarray(rgb_symbols)
    color_alphabet = 1 << parts.header.bits[3]
    if (
        rgb.shape != (parts.header.n, 3)
        or rgb.dtype.kind not in "ui"
        or (rgb.dtype.kind == "i" and np.any(rgb < 0))
        or (rgb.size and int(rgb.max()) >= color_alphabet)
    ):
        raise ContainerError(
            f"SSP2F RGB override must be an integer [{parts.header.n},3] array "
            f"inside [0,{color_alphabet})"
        )
    normalized_rgb = np.ascontiguousarray(rgb, dtype=np.uint32)
    symbols = {
        name: np.ascontiguousarray(values, dtype=np.uint32)
        for name, values in parts.symbols.items()
    }
    for index, name in enumerate(("red", "green", "blue")):
        symbols[name] = np.ascontiguousarray(normalized_rgb[:, index])

    blob = _encode_ssp2f_parts(parts, symbols, arithmetic_encode)
    parsed = parse_container(blob, reference_sspl1=sspl1_blob)
    if parsed.header != parts.header:
        raise ContainerError("SSP2F RGB override changed the frozen header")
    decoded = decode_container(
        blob,
        arithmetic_decode=arithmetic_decode,
        arithmetic_encode=arithmetic_encode,
    )
    for name in ("mean_x", "mean_y", "scale_x", "scale_y", "rotation"):
        if not np.array_equal(decoded.symbols[name], parts.symbols[name]):
            raise ContainerError(f"SSP2F RGB override changed non-RGB channel {name}")
    for index, name in enumerate(("red", "green", "blue")):
        if not np.array_equal(decoded.symbols[name], normalized_rgb[:, index]):
            raise ContainerError(f"SSP2F RGB override cold-decode differs in {name}")
    reencoded = _encode_ssp2f_parts(parts, decoded.symbols, arithmetic_encode)
    if reencoded != blob:
        raise ContainerError("SSP2F RGB override is not byte-canonical after cold decode")
    return blob


def _decode_means_rotation(parsed: ParsedContainer) -> dict[str, np.ndarray]:
    header = parsed.header
    means_bytes, _, rotation_bytes, _ = header.raw_stream_lengths()
    means_raw = _strict_zlib(parsed.payloads["means"], means_bytes, "means")
    itemsize = _packed_itemsize(header.bits[0])
    if itemsize == 1:
        means_flat = np.frombuffer(means_raw, dtype=np.uint8).astype(np.uint32)
    else:
        planes = np.frombuffer(means_raw, dtype=np.uint8).reshape(itemsize, 2 * header.n)
        interleaved = np.ascontiguousarray(planes.T)
        dtype = np.dtype("<u2" if itemsize == 2 else "<u4")
        means_flat = np.frombuffer(interleaved.tobytes(), dtype=dtype).astype(np.uint32)
    if means_flat.size and int(means_flat.max()) >= 1 << header.bits[0]:
        raise ContainerError("mean symbol exceeds declared bit depth")
    means = means_flat.reshape(2, header.n).T
    means = np.cumsum(means, axis=0, dtype=np.uint64)
    means &= np.uint64((1 << header.bits[0]) - 1)

    rotation_raw = _strict_zlib(parsed.payloads["rotation"], rotation_bytes, "rotation")
    rotation_dtype = np.dtype(
        "u1" if header.bits[2] <= 8 else "<u2" if header.bits[2] <= 16 else "<u4"
    )
    rotation = np.frombuffer(rotation_raw, dtype=rotation_dtype).astype(np.uint32)
    if rotation.size and int(rotation.max()) >= 1 << header.bits[2]:
        raise ContainerError("rotation symbol exceeds declared bit depth")
    return {
        "mean_x": np.ascontiguousarray(means[:, 0], dtype=np.uint32),
        "mean_y": np.ascontiguousarray(means[:, 1], dtype=np.uint32),
        "rotation": np.ascontiguousarray(rotation, dtype=np.uint32),
    }


def decode_container(
    blob: bytes,
    *,
    arithmetic_decode: ArithmeticDecode,
    arithmetic_encode: ArithmeticEncode,
    frequency_provider: FrequencyProvider | None = None,
    reference_sspl1: bytes | None = None,
) -> DecodedContainer:
    """Cold-decode all symbols and require canonical arithmetic re-encoding."""
    parsed = parse_container(blob, reference_sspl1=reference_sspl1)
    symbols = _decode_means_rotation(parsed)
    header = parsed.header

    cdf_by_channel: dict[str, np.ndarray] = {}
    if parsed.magic in CONTEXTUAL_MAGICS:
        if frequency_provider is None:
            raise ContainerError("contextual decoding requires the static frequency provider")
        cells = oracle.position_cells(symbols["mean_x"], symbols["mean_y"], header.bits[0])
        if parsed.magic == b"SSP2S":
            cells = cells[shuffled_permutation(header.n)]
        rows = infer_head_rows(parsed.payloads["grid"], parsed.payloads["head"], cells)
        for name in MODELED_CHANNELS:
            alphabet = _channel_alphabet(header, name)
            cdf_by_channel[name] = cdf_rows_from_models(
                alphabet, rows[name][0], rows[name][1], frequency_provider
            )
    elif parsed.magic == b"SSP2L":
        if frequency_provider is None:
            raise ContainerError("SSP2L decoding requires the static frequency provider")
        models = parse_global_model(parsed.payloads["model"])
        for name in MODELED_CHANNELS:
            alphabet = _channel_alphabet(header, name)
            location, scale = models[name]
            cdf_by_channel[name] = cdf_rows_from_models(
                alphabet,
                np.full(header.n, location, dtype=np.uint8),
                np.full(header.n, scale, dtype=np.uint8),
                frequency_provider,
            )
    else:
        frequencies = parse_empirical_model(parsed.payloads["model"], header)
        for name in MODELED_CHANNELS:
            cdf = np.concatenate(
                (np.zeros(1, dtype=np.uint32), np.cumsum(frequencies[name], dtype=np.uint32))
            )
            cdf_by_channel[name] = np.repeat(cdf[None, :], header.n, axis=0)

    for name in MODELED_CHANNELS:
        alphabet = _channel_alphabet(header, name)
        cdfs = cdf_by_channel[name]
        try:
            decoded = arithmetic_decode(parsed.payloads[name], cdfs, header.n)
        except Exception as exc:
            raise ContainerError(f"arithmetic decoder rejected {name}: {exc}") from exc
        values = _symbol_vector(decoded, header.n, alphabet, name)
        canonical = _encode_arithmetic(values, cdfs, arithmetic_encode, name)
        if canonical != parsed.payloads[name]:
            raise ContainerError(f"noncanonical {name} arithmetic payload")
        symbols[name] = values

    ordered = {name: symbols[name] for name in SYMBOL_CHANNELS}
    if parsed.magic == b"SSP2F":
        canonical_model = serialize_empirical_model(ordered, header)
        if canonical_model != parsed.payloads["model"]:
            raise ContainerError("SSP2F model does not canonically describe decoded symbols")
    digest = _symbol_sha256(ordered)
    if reference_sspl1 is not None:
        source = extract_sspl1_parts(reference_sspl1)
        if digest != source.symbol_sha256:
            raise ContainerError("decoded symbol hash differs from reference SSPL1")
        for name in SYMBOL_CHANNELS:
            if not np.array_equal(ordered[name], source.symbols[name]):
                raise ContainerError(f"decoded {name} differs from reference SSPL1")
    return DecodedContainer(parsed=parsed, symbols=ordered, symbol_sha256=digest)


__all__ = [
    "ALL_MAGICS",
    "CONTEXTUAL_FIXED_BYTES",
    "CONTEXTUAL_PREFIX_BYTES",
    "ContainerError",
    "ContainerHeader",
    "DecodedContainer",
    "DirectoryEntry",
    "FACTORIZED_PREFIX_BYTES",
    "HeadChannel",
    "MODELED_CHANNELS",
    "ParsedContainer",
    "SSPL1Parts",
    "cdf_rows_from_models",
    "decode_container",
    "empirical_frequencies",
    "empirical_frequencies_from_counts",
    "encode_contextual",
    "encode_ssp2f",
    "encode_ssp2f_rgb_override",
    "encode_ssp2l",
    "extract_sspl1_parts",
    "infer_head_rows",
    "parse_container",
    "parse_empirical_model",
    "parse_global_model",
    "parse_head",
    "serialize_empirical_frequencies",
    "serialize_empirical_model",
    "serialize_global_model",
    "serialize_head",
    "shuffled_permutation",
]
