"""Benchmark-only fixed-header codec for gauge-free covariance charts.

The container deliberately keeps every non-geometry decision identical across ``current_rs``,
``canonical_rs``, and ``log_spd``.  It is source-decodable (the chart, predictor, coder, ranges,
and bit depths are all in the blob), uses one deterministic Morton order, and accounts for every
byte.  Nothing in this module changes or extends the production SSPL1 codec.
"""
from __future__ import annotations

import math
import struct
import zlib
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch

from benchmarks import gauge_free_covariance_core as core
from structsplat.gaussians import GaussianField

Chart = Literal["current_rs", "canonical_rs", "log_spd"]
Predictor = Literal["absolute", "modular_delta"]
Coder = Literal["zlib9", "zstd9"]

_MAGIC = b"GFCOV01\0"
_VERSION = 1
_CHART_TO_ID: dict[str, int] = {"current_rs": 0, "canonical_rs": 1, "log_spd": 2}
_PREDICTOR_TO_ID: dict[str, int] = {"absolute": 0, "modular_delta": 1}
_CODER_TO_ID: dict[str, int] = {"zlib9": 0, "zstd9": 1}
_ID_TO_CHART = {value: key for key, value in _CHART_TO_ID.items()}
_ID_TO_PREDICTOR = {value: key for key, value in _PREDICTOR_TO_ID.items()}
_ID_TO_CODER = {value: key for key, value in _CODER_TO_ID.items()}

# version, chart, predictor, coder, N, H, W, means bits, three geometry bits, colors bits.
_METADATA = struct.Struct("<BBBBIII5B3x")
# means lo/hi (2+2), geometry lo/hi (3+3), colors lo/hi (3+3), all float32.
_RANGES = struct.Struct("<16f")
_CRC = struct.Struct("<I")
_FRAME = struct.Struct("<QQI")  # raw bytes, compressed bytes, CRC32 of raw bytes
HEADER_BYTES = len(_MAGIC) + _METADATA.size + _RANGES.size + _CRC.size
RANGE_BYTES = _RANGES.size
FRAME_BYTES = _FRAME.size
STREAM_NAMES = ("means", "geometry", "colors")
MAX_GEOMETRY_PROJECTIONS = 1 << 20


@dataclass(frozen=True)
class CodecConfig:
    """Frozen assay decisions for one container arm.

    Geometry bits are coordinate-specific.  The frozen search exhausts all tuples with entries
    in ``[3, 10]`` and total budget 12, 18, or 24 bits per Gaussian.
    """

    chart: Chart = "current_rs"
    geometry_bits: tuple[int, int, int] = (6, 6, 6)
    predictor: Predictor = "absolute"  # geometry only; means/colors always use Morton deltas
    coder: Coder = "zlib9"
    bits_means: int = 12
    bits_colors: int = 8

    def __post_init__(self) -> None:
        _validate_choice("chart", self.chart, _CHART_TO_ID)
        _validate_choice("predictor", self.predictor, _PREDICTOR_TO_ID)
        _validate_choice("coder", self.coder, _CODER_TO_ID)
        bits = _geometry_bits(self.geometry_bits)
        object.__setattr__(self, "geometry_bits", bits)
        _validate_bits("bits_means", self.bits_means, 1, 24)
        _validate_bits("bits_colors", self.bits_colors, 1, 24)
        if self.bits_means != 12 or self.bits_colors != 8:
            raise ValueError("the frozen assay requires 12-bit means and 8-bit colors")

    @property
    def geometry_total_bits(self) -> int:
        return sum(self.geometry_bits)


# Descriptive alias for callers that want to distinguish this benchmark from production SSPL1.
GaugeFreeCodecConfig = CodecConfig


@dataclass(frozen=True)
class _Header:
    chart: Chart
    predictor: Predictor
    coder: Coder
    n: int
    height: int
    width: int
    bits_means: int
    geometry_bits: tuple[int, int, int]
    bits_colors: int
    means_lo: np.ndarray
    means_hi: np.ndarray
    geometry_lo: np.ndarray
    geometry_hi: np.ndarray
    colors_lo: np.ndarray
    colors_hi: np.ndarray


def _validate_choice(name: str, value: str, choices: dict[str, int]) -> None:
    if value not in choices:
        expected = ", ".join(sorted(choices))
        raise ValueError(f"{name} must be one of {expected}; got {value!r}")


def _validate_bits(name: str, value: int, lower: int, upper: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    if not lower <= result <= upper:
        raise ValueError(f"{name} must be in [{lower},{upper}]")
    return result


def _geometry_bits(values: tuple[int, int, int]) -> tuple[int, int, int]:
    if not isinstance(values, (tuple, list)) or len(values) != 3:
        raise ValueError("geometry_bits must contain exactly three bit depths")
    result = tuple(_validate_bits("geometry bit depth", value, 3, 10) for value in values)
    if sum(result) not in {12, 18, 24}:
        raise ValueError("geometry_bits must total 12, 18, or 24")
    return result  # type: ignore[return-value]


def _crc32(payload: bytes) -> int:
    return zlib.crc32(payload) & 0xFFFFFFFF


def morton_order(means: np.ndarray, height: int, width: int) -> np.ndarray:
    """Return the one stable 10-bit Morton order used by every chart arm."""
    points = np.asarray(means, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] == 0:
        raise ValueError("means must be a non-empty (N,2) array")
    if not np.isfinite(points).all():
        raise ValueError("means must be finite")
    if isinstance(height, bool) or isinstance(width, bool) or height <= 0 or width <= 0:
        raise ValueError("height and width must be positive integers")
    x = np.clip(np.rint(points[:, 0] / max(int(width) - 1, 1) * 1023), 0, 1023).astype(
        np.uint64
    )
    y = np.clip(np.rint(points[:, 1] / max(int(height) - 1, 1) * 1023), 0, 1023).astype(
        np.uint64
    )
    code = np.zeros(points.shape[0], dtype=np.uint64)
    for bit in range(10):
        shift = np.uint64(bit)
        code |= ((x >> shift) & np.uint64(1)) << np.uint64(2 * bit)
        code |= ((y >> shift) & np.uint64(1)) << np.uint64(2 * bit + 1)
    return np.argsort(code, kind="stable")


def decoder_synchronized_morton_order(
    means: np.ndarray, height: int, width: int, *, bits: int = 12
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Quantize means first and return their stable decoder-visible Morton order."""
    symbols, lower, upper = _linear_quantize_columns(means, (bits, bits))
    reconstructed = _linear_dequantize_columns(symbols, lower, upper, (bits, bits))
    return morton_order(reconstructed, height, width), symbols, lower, upper


def _linear_quantize_columns(
    values: np.ndarray, bits: tuple[int, ...]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != len(bits):
        raise ValueError("values and bit-depth tuple disagree")
    symbols: list[np.ndarray] = []
    lower: list[float] = []
    upper: list[float] = []
    for column, depth in enumerate(bits):
        quantized, lo, hi = core.quantize_linear(x[:, column : column + 1], depth)
        symbols.append(quantized[:, 0])
        lower.append(float(lo[0]))
        upper.append(float(hi[0]))
    return np.stack(symbols, axis=1), np.asarray(lower), np.asarray(upper)


def _linear_dequantize_columns(
    symbols: np.ndarray, lower: np.ndarray, upper: np.ndarray, bits: tuple[int, ...]
) -> np.ndarray:
    q = np.asarray(symbols, dtype=np.uint32)
    columns = [
        core.dequantize_linear(
            q[:, column : column + 1], lower[column : column + 1], upper[column : column + 1], depth
        )[:, 0]
        for column, depth in enumerate(bits)
    ]
    return np.stack(columns, axis=1)


def _geometry_coordinates(
    log_scales: np.ndarray, rotations: np.ndarray, chart: Chart
) -> tuple[np.ndarray, np.ndarray]:
    if chart == "current_rs":
        return np.asarray(log_scales, dtype=np.float64), np.mod(rotations, math.pi)
    if chart == "canonical_rs":
        return core.canonicalize_rs(log_scales, rotations)
    if chart == "log_spd":
        return core.rs_to_log_spd(log_scales, rotations), np.empty(0, dtype=np.float64)
    raise AssertionError(f"unreachable chart {chart!r}")


def _quantize_geometry(
    log_scales: np.ndarray,
    rotations: np.ndarray,
    chart: Chart,
    bits: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coordinates, angles = _geometry_coordinates(log_scales, rotations, chart)
    if chart == "log_spd":
        return _linear_quantize_columns(coordinates, bits)
    scale_symbols, scale_lo, scale_hi = _linear_quantize_columns(coordinates, bits[:2])
    angle_symbols = core.quantize_circular(angles, bits[2]).reshape(-1, 1)
    symbols = np.concatenate([scale_symbols, angle_symbols], axis=1)
    lower = np.concatenate([scale_lo, np.asarray([0.0])])
    upper = np.concatenate([scale_hi, np.asarray([math.pi])])
    return symbols, lower, upper


def _dequantize_geometry_grid(
    chart: Chart,
    symbols: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    bits: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray]:
    if chart == "log_spd":
        coordinates = _linear_dequantize_columns(
            symbols, lower, upper, bits
        )
        return core.log_spd_to_rs(coordinates)
    scales = _linear_dequantize_columns(
        symbols[:, :2], lower[:2], upper[:2], bits[:2]
    )
    angles = core.dequantize_circular(symbols[:, 2], bits[2])
    return scales, angles


def _dequantize_geometry(header: _Header, symbols: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return _dequantize_geometry_grid(
        header.chart,
        symbols,
        header.geometry_lo,
        header.geometry_hi,
        header.geometry_bits,
    )


def _stabilize_geometry(
    log_scales: np.ndarray,
    rotations: np.ndarray,
    chart: Chart,
    bits: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project geometry onto an encoder/decoder fixed point.

    The cold decoder returns float32 RS parameters.  Coarse quantization can create exact
    isotropy, and the log-SPD inverse/forward maps can move an endpoint by one float32 ULP.  Emit
    the lexicographically canonical state of the resulting fixed point or finite cycle.  Choosing
    a cycle representative makes the result independent of which phase the input field occupies.
    """
    def project(
        scales: np.ndarray, angles: np.ndarray
    ) -> tuple[
        tuple[bytes, bytes, bytes],
        tuple[np.ndarray, np.ndarray, np.ndarray],
        np.ndarray,
        np.ndarray,
    ]:
        symbols, lower, upper = _quantize_geometry(scales, angles, chart, bits)
        key = (
            symbols.tobytes(),
            np.asarray(lower, dtype="<f4").tobytes(),
            np.asarray(upper, dtype="<f4").tobytes(),
        )
        decoded_scales, decoded_angles = _dequantize_geometry_grid(
            chart, symbols, lower, upper, bits
        )
        projected_scales = np.asarray(decoded_scales, dtype=np.float32).astype(np.float64)
        projected_angles = np.asarray(decoded_angles, dtype=np.float32).astype(np.float64)
        return (
            key,
            (symbols.copy(), lower.copy(), upper.copy()),
            projected_scales,
            projected_angles,
        )

    scales = np.asarray(log_scales, dtype=np.float32).astype(np.float64)
    angles = np.asarray(rotations, dtype=np.float32).reshape(-1).astype(np.float64)
    initial = project(scales, angles)
    tortoise_key = initial[0]
    hare = project(initial[2], initial[3])
    projections = 2
    power = cycle_length = 1

    # Brent's algorithm finds the eventual syntax-state cycle with constant memory.  Float32
    # decoder state is finite; the high operational limit prevents a malformed field from hanging
    # the benchmark while remaining far above the longest adversarial preflight transient.
    while tortoise_key != hare[0]:
        if projections >= MAX_GEOMETRY_PROJECTIONS:
            raise RuntimeError(
                "geometry quantizer exceeded the decoded-field cycle-search budget"
            )
        if power == cycle_length:
            tortoise_key = hare[0]
            power *= 2
            cycle_length = 0
        hare = project(hare[2], hare[3])
        projections += 1
        cycle_length += 1

    best_key = hare[0]
    best_state = hare[1]
    cursor = hare
    for _ in range(1, cycle_length):
        cursor = project(cursor[2], cursor[3])
        if cursor[0] < best_key:
            best_key = cursor[0]
            best_state = cursor[1]
    return best_state


def _predict(symbols: np.ndarray, bits: tuple[int, ...], predictor: Predictor) -> np.ndarray:
    if predictor == "absolute":
        return np.asarray(symbols, dtype=np.uint32)
    columns = [
        core.modular_delta(symbols[:, index : index + 1], depth)[:, 0]
        for index, depth in enumerate(bits)
    ]
    return np.stack(columns, axis=1)


def _unpredict(symbols: np.ndarray, bits: tuple[int, ...], predictor: Predictor) -> np.ndarray:
    if predictor == "absolute":
        return np.asarray(symbols, dtype=np.uint32)
    columns = [
        core.modular_undelta(symbols[:, index : index + 1], depth)[:, 0]
        for index, depth in enumerate(bits)
    ]
    return np.stack(columns, axis=1)


def _packed_bytes(n: int, bits: tuple[int, ...]) -> int:
    return sum((n * depth + 7) // 8 for depth in bits)


def _pack_planes(symbols: np.ndarray, bits: tuple[int, ...]) -> bytes:
    q = np.asarray(symbols, dtype=np.uint32)
    if q.ndim != 2 or q.shape[1] != len(bits):
        raise ValueError("symbol array and bit-depth tuple disagree")
    return b"".join(core.bitpack(q[:, column], depth) for column, depth in enumerate(bits))


def _unpack_planes(payload: bytes, n: int, bits: tuple[int, ...]) -> np.ndarray:
    expected = _packed_bytes(n, bits)
    if len(payload) != expected:
        raise ValueError(f"packed stream has {len(payload)} bytes, expected {expected}")
    columns: list[np.ndarray] = []
    offset = 0
    for depth in bits:
        length = (n * depth + 7) // 8
        columns.append(core.bitunpack(payload[offset : offset + length], depth, n))
        offset += length
    return np.stack(columns, axis=1)


def _compress(raw: bytes, coder: Coder) -> bytes:
    if coder == "zlib9":
        return zlib.compress(raw, level=9)
    if coder == "zstd9":
        try:
            import zstandard as zstd
        except ImportError as exc:  # pragma: no cover - depends on optional benchmark environment
            raise RuntimeError("zstd9 requires the optional 'zstandard' package") from exc
        return zstd.ZstdCompressor(
            level=9, write_checksum=True, write_content_size=True, write_dict_id=False
        ).compress(raw)
    raise AssertionError(f"unreachable coder {coder!r}")


def _decompress(payload: bytes, coder: Coder, expected_bytes: int) -> bytes:
    try:
        if coder == "zlib9":
            decoder = zlib.decompressobj()
            raw = decoder.decompress(payload, expected_bytes + 1) + decoder.flush()
            if not decoder.eof or decoder.unused_data or decoder.unconsumed_tail:
                raise ValueError("invalid or concatenated zlib stream")
        elif coder == "zstd9":
            try:
                import zstandard as zstd
            except ImportError as exc:  # pragma: no cover - optional benchmark environment
                raise RuntimeError("zstd9 requires the optional 'zstandard' package") from exc
            raw = zstd.ZstdDecompressor().decompress(
                payload, max_output_size=expected_bytes, allow_extra_data=False
            )
        else:  # pragma: no cover - parser prevents this
            raise AssertionError(f"unreachable coder {coder!r}")
    except RuntimeError:
        raise
    except Exception as exc:
        raise ValueError(f"invalid {coder} stream") from exc
    if len(raw) != expected_bytes:
        raise ValueError(f"decoded stream has {len(raw)} bytes, expected {expected_bytes}")
    return raw


def _field_arrays(field: GaussianField) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if field.n <= 0:
        raise ValueError("cannot encode an empty GaussianField")
    if field.opacities is not None:
        raise ValueError("benchmark container does not encode opacity")
    if field.color_grads is not None:
        raise ValueError("benchmark container does not encode affine color gradients")
    means = field.means.detach().cpu().numpy().astype(np.float64)
    # A covariance filter has no separate stream in the assay.  Folding it into RS preserves the
    # represented covariance and prevents an out-of-band decoder dependency.
    log_scales = field.effective_log_scales().detach().cpu().numpy().astype(np.float64)
    rotations = field.rotations.detach().cpu().numpy().astype(np.float64).reshape(-1)
    colors = field.colors.detach().cpu().numpy().astype(np.float64)
    n = field.n
    if means.shape != (n, 2) or log_scales.shape != (n, 2) or colors.shape != (n, 3):
        raise ValueError("GaussianField parameter shapes are inconsistent")
    if rotations.shape != (n,):
        raise ValueError("GaussianField rotations must have shape (N,)")
    return means, log_scales, rotations, colors


def _encode_header(header: _Header) -> bytes:
    metadata = _METADATA.pack(
        _VERSION,
        _CHART_TO_ID[header.chart],
        _PREDICTOR_TO_ID[header.predictor],
        _CODER_TO_ID[header.coder],
        header.n,
        header.height,
        header.width,
        header.bits_means,
        *header.geometry_bits,
        header.bits_colors,
    )
    range_values = np.concatenate(
        [
            header.means_lo,
            header.means_hi,
            header.geometry_lo,
            header.geometry_hi,
            header.colors_lo,
            header.colors_hi,
        ]
    )
    ranges = _RANGES.pack(*range_values.tolist())
    body = metadata + ranges
    return _MAGIC + body + _CRC.pack(_crc32(body))


def _enum_value(mapping: dict[int, str], value: int, name: str) -> str:
    try:
        return mapping[value]
    except KeyError as exc:
        raise ValueError(f"unknown {name} id {value}") from exc


def _decode_header(blob: bytes) -> _Header:
    if len(blob) < HEADER_BYTES:
        raise ValueError("truncated gauge-free covariance header")
    if blob[: len(_MAGIC)] != _MAGIC:
        raise ValueError("not a gauge-free covariance container")
    body_start = len(_MAGIC)
    body_end = body_start + _METADATA.size + _RANGES.size
    body = blob[body_start:body_end]
    (stored_crc,) = _CRC.unpack_from(blob, body_end)
    if _crc32(body) != stored_crc:
        raise ValueError("gauge-free covariance header CRC mismatch")
    values = _METADATA.unpack_from(body)
    version, chart_id, predictor_id, coder_id = values[:4]
    if version != _VERSION:
        raise ValueError(f"unsupported gauge-free covariance version {version}")
    n, height, width = (int(value) for value in values[4:7])
    if n <= 0 or height <= 0 or width <= 0:
        raise ValueError("container dimensions must be positive")
    bits_means = _validate_bits("bits_means", values[7], 1, 24)
    geometry_bits = _geometry_bits(tuple(int(value) for value in values[8:11]))
    bits_colors = _validate_bits("bits_colors", values[11], 1, 24)
    ranges = np.asarray(_RANGES.unpack_from(body, _METADATA.size), dtype=np.float64)
    if not np.isfinite(ranges).all():
        raise ValueError("container ranges must be finite")
    means_lo, means_hi = ranges[0:2], ranges[2:4]
    geometry_lo, geometry_hi = ranges[4:7], ranges[7:10]
    colors_lo, colors_hi = ranges[10:13], ranges[13:16]
    if np.any(means_hi < means_lo) or np.any(geometry_hi < geometry_lo) or np.any(
        colors_hi < colors_lo
    ):
        raise ValueError("container range upper endpoint is below its lower endpoint")
    return _Header(
        chart=_enum_value(_ID_TO_CHART, chart_id, "chart"),  # type: ignore[arg-type]
        predictor=_enum_value(_ID_TO_PREDICTOR, predictor_id, "predictor"),  # type: ignore[arg-type]
        coder=_enum_value(_ID_TO_CODER, coder_id, "coder"),  # type: ignore[arg-type]
        n=n,
        height=height,
        width=width,
        bits_means=bits_means,
        geometry_bits=geometry_bits,
        bits_colors=bits_colors,
        means_lo=means_lo,
        means_hi=means_hi,
        geometry_lo=geometry_lo,
        geometry_hi=geometry_hi,
        colors_lo=colors_lo,
        colors_hi=colors_hi,
    )


def _stream_bits(header: _Header) -> tuple[tuple[int, ...], ...]:
    return (
        (header.bits_means, header.bits_means),
        header.geometry_bits,
        (header.bits_colors, header.bits_colors, header.bits_colors),
    )


def _read_frames(blob: bytes, header: _Header, *, decompress: bool) -> tuple[list[bytes], dict]:
    offset = HEADER_BYTES
    raws: list[bytes] = []
    stream_records: dict[str, dict[str, int]] = {}
    for name, bits in zip(STREAM_NAMES, _stream_bits(header), strict=True):
        if offset + FRAME_BYTES > len(blob):
            raise ValueError(f"truncated {name} stream frame")
        raw_bytes, payload_bytes, raw_crc = _FRAME.unpack_from(blob, offset)
        offset += FRAME_BYTES
        expected_raw = _packed_bytes(header.n, bits)
        if raw_bytes != expected_raw:
            raise ValueError(
                f"{name} frame declares {raw_bytes} raw bytes, expected {expected_raw}"
            )
        payload_end = offset + int(payload_bytes)
        if payload_end > len(blob):
            raise ValueError(f"truncated {name} stream payload")
        payload = blob[offset:payload_end]
        offset = payload_end
        if decompress:
            raw = _decompress(payload, header.coder, expected_raw)
            if _crc32(raw) != raw_crc:
                raise ValueError(f"{name} stream CRC mismatch")
            raws.append(raw)
        stream_records[name] = {
            "raw_bytes": int(raw_bytes),
            "payload_bytes": int(payload_bytes),
            "frame_bytes": FRAME_BYTES,
            "framed_bytes": FRAME_BYTES + int(payload_bytes),
        }
    if offset != len(blob):
        raise ValueError(f"container has {len(blob) - offset} trailing bytes")
    return raws, stream_records


def encode(
    field: GaussianField,
    height: int,
    width: int,
    config: CodecConfig | None = None,
) -> bytes:
    """Encode one field into a benchmark-only, self-describing container."""
    cfg = config or CodecConfig()
    if not isinstance(height, int) or isinstance(height, bool) or height <= 0:
        raise ValueError("height must be a positive integer")
    if not isinstance(width, int) or isinstance(width, bool) or width <= 0:
        raise ValueError("width must be a positive integer")
    means, log_scales, rotations, colors = _field_arrays(field)
    order, mean_symbols, means_lo, means_hi = decoder_synchronized_morton_order(
        means, height, width, bits=cfg.bits_means
    )
    # Row order must be a function of decoder-visible state.  Ordering the unquantized means can
    # swap rows after a cold decode when a point crosses a Morton-cell boundary.  Sort by the
    # reconstructed 12-bit means and then carry that one order through every stream.
    mean_symbols = mean_symbols[order]
    log_scales, rotations, colors = (
        values[order] for values in (log_scales, rotations, colors)
    )
    geometry_symbols, geometry_lo, geometry_hi = _stabilize_geometry(
        log_scales, rotations, cfg.chart, cfg.geometry_bits
    )
    color_symbols, colors_lo, colors_hi = _linear_quantize_columns(
        colors, (cfg.bits_colors, cfg.bits_colors, cfg.bits_colors)
    )
    bits_by_stream = (
        (cfg.bits_means, cfg.bits_means),
        cfg.geometry_bits,
        (cfg.bits_colors, cfg.bits_colors, cfg.bits_colors),
    )
    stream_predictors: tuple[Predictor, Predictor, Predictor] = (
        "modular_delta",
        cfg.predictor,
        "modular_delta",
    )
    raw_streams = [
        _pack_planes(_predict(symbols, bits, predictor), bits)
        for symbols, bits, predictor in zip(
            (mean_symbols, geometry_symbols, color_symbols),
            bits_by_stream,
            stream_predictors,
            strict=True,
        )
    ]

    header = _Header(
        chart=cfg.chart,
        predictor=cfg.predictor,
        coder=cfg.coder,
        n=field.n,
        height=height,
        width=width,
        bits_means=cfg.bits_means,
        geometry_bits=cfg.geometry_bits,
        bits_colors=cfg.bits_colors,
        means_lo=means_lo,
        means_hi=means_hi,
        geometry_lo=geometry_lo,
        geometry_hi=geometry_hi,
        colors_lo=colors_lo,
        colors_hi=colors_hi,
    )
    blob = bytearray(_encode_header(header))
    for raw in raw_streams:
        payload = _compress(raw, cfg.coder)
        blob.extend(_FRAME.pack(len(raw), len(payload), _crc32(raw)))
        blob.extend(payload)
    return bytes(blob)


def decode(blob: bytes, device: str | torch.device = "cpu") -> GaussianField:
    """Cold-decode a blob without any out-of-band codec configuration."""
    if not isinstance(blob, (bytes, bytearray, memoryview)):
        raise TypeError("blob must be bytes-like")
    payload = bytes(blob)
    header = _decode_header(payload)
    raws, _ = _read_frames(payload, header, decompress=True)
    predicted = [
        _unpack_planes(raw, header.n, bits)
        for raw, bits in zip(raws, _stream_bits(header), strict=True)
    ]
    stream_predictors: tuple[Predictor, Predictor, Predictor] = (
        "modular_delta",
        header.predictor,
        "modular_delta",
    )
    mean_symbols, geometry_symbols, color_symbols = [
        _unpredict(symbols, bits, predictor)
        for symbols, bits, predictor in zip(
            predicted, _stream_bits(header), stream_predictors, strict=True
        )
    ]
    means = _linear_dequantize_columns(
        mean_symbols,
        header.means_lo,
        header.means_hi,
        (header.bits_means, header.bits_means),
    )
    log_scales, rotations = _dequantize_geometry(header, geometry_symbols)
    colors = _linear_dequantize_columns(
        color_symbols,
        header.colors_lo,
        header.colors_hi,
        (header.bits_colors, header.bits_colors, header.bits_colors),
    )

    def tensor(values: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(np.ascontiguousarray(values), dtype=torch.float32, device=device)

    return GaussianField(tensor(means), tensor(log_scales), tensor(rotations), tensor(colors))


def blob_header(blob: bytes) -> dict[str, object]:
    """Return the validated fixed-header fields in a JSON-friendly form."""
    header = _decode_header(bytes(blob))
    return {
        "format": "GFCOV01",
        "version": _VERSION,
        "chart": header.chart,
        "predictor": header.predictor,
        "coder": header.coder,
        "n": header.n,
        "height": header.height,
        "width": header.width,
        "bits_means": header.bits_means,
        "geometry_bits": list(header.geometry_bits),
        "geometry_total_bits": sum(header.geometry_bits),
        "bits_colors": header.bits_colors,
        "means_lo": header.means_lo.tolist(),
        "means_hi": header.means_hi.tolist(),
        "geometry_lo": header.geometry_lo.tolist(),
        "geometry_hi": header.geometry_hi.tolist(),
        "colors_lo": header.colors_lo.tolist(),
        "colors_hi": header.colors_hi.tolist(),
        "header_bytes": HEADER_BYTES,
        "range_bytes": RANGE_BYTES,
        "range_scalar_bytes": 4,
    }


def blob_components(blob: bytes) -> dict[str, object]:
    """Return exact byte accounting, validating all frame boundaries along the way."""
    payload = bytes(blob)
    header = _decode_header(payload)
    _, streams = _read_frames(payload, header, decompress=False)
    framed_bytes = sum(record["framed_bytes"] for record in streams.values())
    result: dict[str, object] = {
        "header_bytes": HEADER_BYTES,
        "range_bytes": RANGE_BYTES,
        "range_scalar_bytes": 4,
        "metadata_and_crc_bytes": HEADER_BYTES - RANGE_BYTES,
        "framing_bytes": len(STREAM_NAMES) * FRAME_BYTES,
        "stream_payload_bytes": sum(record["payload_bytes"] for record in streams.values()),
        "framed_stream_bytes": framed_bytes,
        "total_bytes": len(payload),
        "streams": streams,
    }
    if HEADER_BYTES + framed_bytes != len(payload):  # construction/parsing invariant
        raise AssertionError("component accounting does not sum to the complete container")
    return result


def canonical_reencode(blob: bytes) -> bytes:
    """Validate, decompress, and deterministically reconstruct the exact container bytes.

    This is the strict cold-replay path for the benchmark.  It avoids float round trips entirely:
    the decoded symbols and fixed binary metadata are canonical, so both zlib9 and zstd9 reproduce
    the source bytes byte-for-byte.
    """
    payload = bytes(blob)
    header = _decode_header(payload)
    raw_streams, _ = _read_frames(payload, header, decompress=True)
    replay = bytearray(_encode_header(header))
    for raw in raw_streams:
        compressed = _compress(raw, header.coder)
        replay.extend(_FRAME.pack(len(raw), len(compressed), _crc32(raw)))
        replay.extend(compressed)
    result = bytes(replay)
    if result != payload:
        raise ValueError("container is valid but its entropy stream is not in canonical form")
    return result


def decoded_reencode(blob: bytes) -> bytes:
    """Cold-decode to a GaussianField and pass it through the ordinary encoder.

    Unlike :func:`canonical_reencode`, this exercises the transmitted float32 reconstruction,
    chart inverse, Morton order, range derivation, quantizer, predictor, and entropy coder.
    """
    payload = bytes(blob)
    header = _decode_header(payload)
    config = CodecConfig(
        chart=header.chart,
        geometry_bits=header.geometry_bits,
        predictor=header.predictor,
        coder=header.coder,
        bits_means=header.bits_means,
        bits_colors=header.bits_colors,
    )
    return encode(decode(payload, device="cpu"), header.height, header.width, config)


# Explicit aliases make the benchmark API readable without shadowing the production codec API.
encode_field = encode
decode_field = decode
container_components = blob_components
