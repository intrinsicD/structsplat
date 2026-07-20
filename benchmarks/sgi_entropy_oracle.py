"""COMP-008 benchmark-only conditional-entropy lower-bound oracle.

This module deliberately does *not* implement the proposed SSP2E coder or its learned head.  It
parses complete production SSPL1 byte strings, recovers the absolute quantizer symbols, and asks a
strictly easier oracle question: even if five attribute streams were coded at their empirical
entropy independently in each 16x16 position cell, could the compulsory future container beat the
complete SSPL1 baseline by the frozen margin?

The scientific lower bound is integer-certified.  For counts ``n_c`` and ``n_cs``, conditional
entropy is ``log2(prod(n_c**n_c) / prod(n_cs**n_cs))``.  Python integers therefore determine
``floor(H / 8)`` without trusting floating-point logarithms.  Floating-point entropy and bootstrap
values in output records are explicitly diagnostics; every decision comparison is exact integer
arithmetic.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import zlib
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


MAGIC = b"SSPL1"
ROW_SCHEMA = "structsplat.comp008.sgi-entropy-oracle-row.v1"
ANALYSIS_SCHEMA = "structsplat.comp008.sgi-entropy-oracle-analysis.v1"
REPLAY_SCHEMA = "structsplat.comp008.sgi-entropy-oracle-replay.v1"

FROZEN_BIT_TUPLES: tuple[tuple[int, int, int, int], ...] = (
    (12, 6, 6, 8),
    (16, 8, 8, 8),
)
FROZEN_IMAGE_COUNT = 8
FROZEN_GAUSSIAN_COUNT = 8192
FROZEN_IMAGE_IDS = (
    "nomao-saeki-33553",
    "martyn-seddon-220",
    "zugr-108",
    "jason-briscoe-149782",
    "martin-wessely-211",
    "stefan-kunze-26931",
    "vita-vilcina-3055",
    "philippe-wuyts-45997",
)
POSITION_CELLS_PER_AXIS = 16
POSITION_CELL_COUNT = POSITION_CELLS_PER_AXIS**2
SSP2E_OUTER_BYTES = struct.calcsize("<5sBBBQI")
SSP2E_HEADER_BYTES = struct.calcsize("<III4B4i12f5I")
SSP2E_DIRECTORY_ENTRY_BYTES = struct.calcsize("<IQQ")
SSP2E_DIRECTORY_ENTRIES = 9
SSP2E_GRID_BYTES = 256
SSP2E_HEAD_BYTES = 100
SSP2E_COMPULSORY_BYTES = (
    SSP2E_OUTER_BYTES
    + SSP2E_HEADER_BYTES
    + SSP2E_DIRECTORY_ENTRIES * SSP2E_DIRECTORY_ENTRY_BYTES
    + SSP2E_GRID_BYTES
    + SSP2E_HEAD_BYTES
)
if SSP2E_COMPULSORY_BYTES != 656:  # pragma: no cover - struct formats are frozen literals
    raise RuntimeError("SSP2E compulsory byte layout drift")

BOOTSTRAP_DOMAIN = b"structsplat-comp008-bootstrap-v1\0"
BOOTSTRAP_REPLICATES = 100_000
BOOTSTRAP_UPPER_INDEX = 97_499
BOOTSTRAP_MATRIX_SHA256 = "cca3d7403c0c41ea5b5cd8604e566114939ffe90177f9a2ec2a71490ba7232b9"

_STREAM_NAMES = ("means", "scales", "rotation", "colors")
_SYMBOL_NAMES = (
    "mean_x",
    "mean_y",
    "scale_x",
    "scale_y",
    "rotation",
    "red",
    "green",
    "blue",
)
_MODELED_NAMES = ("scale_x", "scale_y", "red", "green", "blue")
_PRODUCTION_HEADER_KEYS = (
    "n",
    "H",
    "W",
    "bits",
    "morton",
    "color_delta",
    "means_byte_planar",
    "rot_circular",
    "stream_codec",
    "stream_raw_lengths",
    "color_lo",
    "color_hi",
    "scale_lo",
    "scale_hi",
    "means_lo",
    "means_hi",
    "has_opacity",
    "bits_opacity",
    "renderer",
    "aa_dilation",
    "sigma_cutoff",
    "support_fade",
    "render_chunk",
)


class OracleError(ValueError):
    """A frozen parsing, integrity, or analysis condition was violated."""


def canonical_json(value: Any) -> bytes:
    """Serialize one finite JSON value canonically, including a terminal newline."""
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def seal_record(record: Mapping[str, Any], hash_field: str = "record_sha256") -> dict[str, Any]:
    """Return a copy carrying a hash over all fields except ``hash_field``."""
    payload = dict(record)
    payload.pop(hash_field, None)
    payload[hash_field] = sha256_bytes(canonical_json(payload))
    return payload


def verify_record(record: Mapping[str, Any], hash_field: str = "record_sha256") -> None:
    observed = record.get(hash_field)
    if not isinstance(observed, str) or len(observed) != 64:
        raise OracleError(f"missing or invalid {hash_field}")
    core = dict(record)
    core.pop(hash_field, None)
    expected = sha256_bytes(canonical_json(core))
    if observed != expected:
        raise OracleError(f"{hash_field} mismatch: {observed} != {expected}")


def _json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise OracleError(f"duplicate SSPL1 header key: {key!r}")
        result[key] = value
    return result


def _invalid_json_constant(value: str) -> None:
    raise OracleError(f"non-finite JSON constant in SSPL1 header: {value}")


def _load_header(payload: bytes) -> dict[str, Any]:
    try:
        text = payload.decode("utf-8")
        value = json.loads(
            text,
            object_pairs_hook=_json_object,
            parse_constant=_invalid_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise OracleError("invalid SSPL1 JSON header") from exc
    if not isinstance(value, dict):
        raise OracleError("SSPL1 JSON header is not an object")
    return value


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise OracleError(f"{name} must be a positive integer")
    return value


def _flag(header: Mapping[str, Any], name: str, default: bool = False) -> bool:
    value = header.get(name, default)
    if not isinstance(value, bool):
        raise OracleError(f"SSPL1 header field {name!r} must be boolean")
    return value


def _packed_dtype(bits: int) -> np.dtype:
    if not 1 <= bits <= 32:
        raise OracleError(f"unsupported SSPL1 bit depth: {bits}")
    if bits <= 8:
        return np.dtype("u1")
    if bits <= 16:
        return np.dtype("<u2")
    return np.dtype("<u4")


def _strict_zlib_decode(payload: bytes, expected_bytes: int, *, canonical_zlib9: bool) -> bytes:
    """Decode exactly one bounded zlib member and reject suffixes or non-level-9 encodings."""
    decoder = zlib.decompressobj()
    try:
        raw = decoder.decompress(payload, expected_bytes + 1)
        raw += decoder.flush()
    except zlib.error as exc:
        raise OracleError("invalid SSPL1 zlib stream") from exc
    if len(raw) != expected_bytes:
        raise OracleError(f"SSPL1 zlib raw length mismatch: {len(raw)} != {expected_bytes}")
    if not decoder.eof or decoder.unused_data or decoder.unconsumed_tail:
        raise OracleError("SSPL1 payload is not exactly one complete zlib stream")
    if canonical_zlib9 and zlib.compress(raw, level=9) != payload:
        raise OracleError("SSPL1 stream is not the canonical local zlib level-9 encoding")
    return raw


def _unpack(raw: bytes, bits: int, count: int) -> np.ndarray:
    dtype = _packed_dtype(bits)
    expected = count * dtype.itemsize
    if len(raw) != expected:
        raise OracleError(f"packed stream has {len(raw)} bytes, expected {expected}")
    values = np.frombuffer(raw, dtype=dtype, count=count).astype(np.uint32)
    if values.size and int(values.max()) >= 1 << bits:
        raise OracleError("packed symbol exceeds its declared bit depth")
    return values


def _unpack_byte_planar(raw: bytes, bits: int, count: int) -> np.ndarray:
    dtype = _packed_dtype(bits)
    expected = count * dtype.itemsize
    if len(raw) != expected:
        raise OracleError(f"planar stream has {len(raw)} bytes, expected {expected}")
    if dtype.itemsize == 1:
        return _unpack(raw, bits, count)
    planes = np.frombuffer(raw, dtype=np.uint8).reshape(dtype.itemsize, count)
    interleaved = np.ascontiguousarray(planes.T)
    values = np.frombuffer(interleaved.tobytes(), dtype=dtype, count=count).astype(np.uint32)
    if values.size and int(values.max()) >= 1 << bits:
        raise OracleError("planar symbol exceeds its declared bit depth")
    return values


def _absolute_modular(values: np.ndarray, bits: int) -> np.ndarray:
    absolute = np.cumsum(values, axis=0, dtype=np.uint64)
    absolute &= np.uint64((1 << bits) - 1)
    return absolute.astype(np.uint32)


def _array_digest(symbols: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in _SYMBOL_NAMES:
        array = np.ascontiguousarray(symbols[name], dtype="<u4")
        digest.update(name.encode("ascii") + b"\0")
        digest.update(struct.pack("<Q", int(array.size)))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True)
class ParsedSSPL1:
    """Complete SSPL1 accounting and absolute quantizer symbols."""

    header: dict[str, Any]
    symbols: dict[str, np.ndarray]
    payload_bytes: dict[str, int]
    raw_bytes: dict[str, int]
    total_bytes: int
    header_bytes: int
    blob_sha256: str
    symbol_sha256: str


def parse_sspl1(blob: bytes, *, canonical_zlib9: bool = True) -> ParsedSSPL1:
    """Strictly parse a complete, opacity-free SSPL1 zlib container.

    Stored Morton deltas are inverted so every returned array is an absolute production quantizer
    symbol in encoded row order.  No float field reconstruction and no pixel decode occurs.
    """
    if not isinstance(blob, bytes):
        raise OracleError("SSPL1 input must be immutable bytes")
    if len(blob) < 9 or blob[:5] != MAGIC:
        raise OracleError("not an SSPL1 byte string")
    (header_length,) = struct.unpack_from("<I", blob, 5)
    header_end = 9 + int(header_length)
    if header_end > len(blob):
        raise OracleError("truncated SSPL1 header")
    header = _load_header(blob[9:header_end])
    if tuple(header) != _PRODUCTION_HEADER_KEYS:
        raise OracleError("SSPL1 header key set/order differs from current production codec")
    if json.dumps(header).encode("utf-8") != blob[9:header_end]:
        raise OracleError("SSPL1 header bytes are not the exact production JSON serialization")
    if header.get("bits_opacity") != 8:
        raise OracleError("opacity-free COMP-008 requires production bits_opacity=8 metadata")

    n = _positive_int(header.get("n"), "n")
    _positive_int(header.get("H"), "H")
    _positive_int(header.get("W"), "W")
    raw_bits = header.get("bits")
    if not isinstance(raw_bits, list) or len(raw_bits) != 4:
        raise OracleError("SSPL1 bits must be a four-element list")
    bits: list[int] = []
    for index, value in enumerate(raw_bits):
        if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 32:
            raise OracleError(f"invalid SSPL1 bits[{index}]")
        bits.append(value)

    if _flag(header, "has_opacity"):
        raise OracleError("COMP-008 excludes opacity-bearing SSPL1 streams")
    if header.get("stream_codec", "zlib") != "zlib":
        raise OracleError("COMP-008 requires SSPL1 zlib streams")
    raw_lengths_value = header.get("stream_raw_lengths")
    if not isinstance(raw_lengths_value, list) or len(raw_lengths_value) != 4:
        raise OracleError("SSPL1 stream_raw_lengths must contain exactly four lengths")
    raw_lengths: list[int] = []
    for index, value in enumerate(raw_lengths_value):
        raw_lengths.append(_positive_int(value, f"stream_raw_lengths[{index}]"))

    b_means, b_scales, b_rotation, b_colors = bits
    expected_raw = [
        2 * n * _packed_dtype(b_means).itemsize,
        2 * n * _packed_dtype(b_scales).itemsize,
        n * _packed_dtype(b_rotation).itemsize,
        3 * n * _packed_dtype(b_colors).itemsize,
    ]
    if raw_lengths != expected_raw:
        raise OracleError(f"SSPL1 raw stream lengths {raw_lengths} != {expected_raw}")

    payloads: dict[str, bytes] = {}
    raws: dict[str, bytes] = {}
    offset = header_end
    for name, expected_length in zip(_STREAM_NAMES, expected_raw, strict=True):
        if offset + 4 > len(blob):
            raise OracleError(f"truncated {name} stream framing")
        (payload_length,) = struct.unpack_from("<I", blob, offset)
        offset += 4
        end = offset + int(payload_length)
        if end > len(blob):
            raise OracleError(f"truncated {name} stream payload")
        payload = blob[offset:end]
        payloads[name] = payload
        raws[name] = _strict_zlib_decode(payload, expected_length, canonical_zlib9=canonical_zlib9)
        offset = end
    if offset != len(blob):
        raise OracleError(f"unexpected {len(blob) - offset} trailing SSPL1 bytes")

    if _flag(header, "means_byte_planar"):
        means_flat = _unpack_byte_planar(raws["means"], b_means, 2 * n)
    else:
        means_flat = _unpack(raws["means"], b_means, 2 * n)
    means = means_flat.reshape(2, n).T
    if _flag(header, "morton"):
        means = _absolute_modular(means, b_means)

    scales = _unpack(raws["scales"], b_scales, 2 * n).reshape(2, n).T
    rotation = _unpack(raws["rotation"], b_rotation, n)
    colors = _unpack(raws["colors"], b_colors, 3 * n).reshape(3, n).T
    if _flag(header, "color_delta"):
        colors = _absolute_modular(colors, b_colors)

    symbols = {
        "mean_x": np.ascontiguousarray(means[:, 0]),
        "mean_y": np.ascontiguousarray(means[:, 1]),
        "scale_x": np.ascontiguousarray(scales[:, 0]),
        "scale_y": np.ascontiguousarray(scales[:, 1]),
        "rotation": np.ascontiguousarray(rotation),
        "red": np.ascontiguousarray(colors[:, 0]),
        "green": np.ascontiguousarray(colors[:, 1]),
        "blue": np.ascontiguousarray(colors[:, 2]),
    }
    return ParsedSSPL1(
        header=header,
        symbols=symbols,
        payload_bytes={name: len(payloads[name]) for name in _STREAM_NAMES},
        raw_bytes={name: len(raws[name]) for name in _STREAM_NAMES},
        total_bytes=len(blob),
        header_bytes=header_end,
        blob_sha256=sha256_bytes(blob),
        symbol_sha256=_array_digest(symbols),
    )


def extract_quantized_arrays(blob: bytes, *, canonical_zlib9: bool = True) -> dict[str, np.ndarray]:
    """Convenience wrapper returning copies of all eight absolute symbol arrays."""
    parsed = parse_sspl1(blob, canonical_zlib9=canonical_zlib9)
    return {name: value.copy() for name, value in parsed.symbols.items()}


def position_cells(mean_x: np.ndarray, mean_y: np.ndarray, bits_means: int) -> np.ndarray:
    """Return frozen 16x16 row-major cells from absolute mean symbols."""
    if not 1 <= bits_means <= 32:
        raise OracleError("bits_means must lie in [1, 32]")
    x = np.asarray(mean_x, dtype=np.uint64).reshape(-1)
    y = np.asarray(mean_y, dtype=np.uint64).reshape(-1)
    if x.shape != y.shape:
        raise OracleError("mean_x and mean_y shapes disagree")
    limit = 1 << bits_means
    if (x.size and int(x.max()) >= limit) or (y.size and int(y.max()) >= limit):
        raise OracleError("mean symbol exceeds bits_means")
    cell_x = np.minimum(15, (x * 16) >> bits_means)
    cell_y = np.minimum(15, (y * 16) >> bits_means)
    return (16 * cell_y + cell_x).astype(np.uint16)


def _integer_bytes(value: int) -> bytes:
    return value.to_bytes(max(1, (value.bit_length() + 7) // 8), "big")


def _floor_log256_ratio(numerator: int, denominator: int) -> int:
    """Compute floor(log_256(numerator / denominator)) using only exact comparisons."""
    if numerator <= 0 or denominator <= 0:
        raise OracleError("entropy certificate ratio must be positive")
    if numerator < denominator:
        raise OracleError("conditional empirical entropy cannot be negative")
    estimate = max(0, (numerator.bit_length() - denominator.bit_length()) // 8)
    while numerator < denominator << (8 * estimate):
        estimate -= 1
    while numerator >= denominator << (8 * (estimate + 1)):
        estimate += 1
    return estimate


def conditional_entropy_certificate(
    symbols: Mapping[str, np.ndarray], bits_means: int
) -> dict[str, Any]:
    """Certify the byte floor of empirical H(attributes | 16x16 position cell)."""
    missing = [name for name in ("mean_x", "mean_y", *_MODELED_NAMES) if name not in symbols]
    if missing:
        raise OracleError(f"missing symbol arrays: {missing}")
    cells = position_cells(symbols["mean_x"], symbols["mean_y"], bits_means)
    n = int(cells.size)
    numerator = 1
    denominator = 1
    channel_diagnostics: dict[str, float] = {}
    occupied = np.bincount(cells, minlength=POSITION_CELL_COUNT)

    for name in _MODELED_NAMES:
        values = np.asarray(symbols[name], dtype=np.uint32).reshape(-1)
        if values.size != n:
            raise OracleError(f"{name} has {values.size} rows, expected {n}")
        # Modeled channels are frozen at at most eight bits, but infer the finite alphabet from
        # observed symbols so the exact count calculation has no empty-alphabet dependency.
        alphabet = max(1, int(values.max()) + 1 if values.size else 1)
        joint_index = cells.astype(np.uint64) * np.uint64(alphabet) + values.astype(np.uint64)
        joint = np.bincount(
            joint_index.astype(np.int64), minlength=POSITION_CELL_COUNT * alphabet
        ).reshape(POSITION_CELL_COUNT, alphabet)

        diagnostic = 0.0
        for cell_count, row in zip(occupied.tolist(), joint, strict=True):
            count = int(cell_count)
            if count:
                numerator *= pow(count, count)
                diagnostic += count * math.log2(count)
            for symbol_count in row[row > 0].tolist():
                frequency = int(symbol_count)
                denominator *= pow(frequency, frequency)
                diagnostic -= frequency * math.log2(frequency)
        channel_diagnostics[name] = diagnostic

    floor_bytes = _floor_log256_ratio(numerator, denominator)
    return {
        "method": "exact_integer_power_ratio_floor_log256",
        "modeled_channels": list(_MODELED_NAMES),
        "position_grid": [16, 16],
        "rows": n,
        "occupied_cells": int(np.count_nonzero(occupied)),
        "per_channel_bits_diagnostic": channel_diagnostics,
        "total_bits_diagnostic": float(sum(channel_diagnostics.values())),
        "entropy_floor_bytes_exact": floor_bytes,
        "entropy_down_bits_exact": 8 * floor_bytes,
        "entropy_minus_down_bits_strict_upper": 8,
        "ratio_numerator_bits": numerator.bit_length(),
        "ratio_denominator_bits": denominator.bit_length(),
        "ratio_numerator_sha256": sha256_bytes(_integer_bytes(numerator)),
        "ratio_denominator_sha256": sha256_bytes(_integer_bytes(denominator)),
    }


def _finite_float_list(header: Mapping[str, Any], name: str, length: int) -> list[float]:
    raw = header.get(name)
    if not isinstance(raw, list) or len(raw) != length:
        raise OracleError(f"SSPL1 {name} must contain {length} values")
    result: list[float] = []
    for value in raw:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise OracleError(f"SSPL1 {name} contains a non-numeric value")
        converted = float(value)
        if not math.isfinite(converted):
            raise OracleError(f"SSPL1 {name} contains a non-finite value")
        result.append(converted)
    return result


def _validate_ssp2e_header_representability(header: Mapping[str, Any]) -> None:
    """Prove that the frozen 100-byte SSP2E header can reproduce SSPL1 metadata."""
    means_lo = _finite_float_list(header, "means_lo", 2)
    means_hi = _finite_float_list(header, "means_hi", 2)
    for value in (*means_lo, *means_hi):
        if value != math.trunc(value) or not -(1 << 31) <= value < (1 << 31):
            raise OracleError("SSPL1 mean extent is not exactly representable by SSP2E int32")
    height = _positive_int(header.get("H"), "H")
    width = _positive_int(header.get("W"), "W")
    if (
        means_lo[0] > 0
        or means_lo[1] > 0
        or means_hi[0] < width - 1
        or means_hi[1] < height - 1
        or means_lo[0] > means_hi[0]
        or means_lo[1] > means_hi[1]
    ):
        raise OracleError("SSPL1 mean extent does not contain the image box")
    for name, length in (("scale_lo", 2), ("scale_hi", 2), ("color_lo", 3), ("color_hi", 3)):
        for value in _finite_float_list(header, name, length):
            if float(np.float32(value)) != value:
                raise OracleError(f"SSPL1 {name} is not exactly representable by SSP2E float32")
    for name in ("aa_dilation", "sigma_cutoff"):
        value = header.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise OracleError(f"SSPL1 {name} must be numeric")
        converted = float(value)
        if not math.isfinite(converted) or float(np.float32(converted)) != converted:
            raise OracleError(f"SSPL1 {name} is not exactly representable by SSP2E float32")
    render_chunk = header.get("render_chunk")
    if (
        isinstance(render_chunk, bool)
        or not isinstance(render_chunk, int)
        or not 1 <= render_chunk < (1 << 32)
    ):
        raise OracleError("SSPL1 render_chunk is not representable by SSP2E uint32")
    if header.get("renderer") != "cuda":
        raise OracleError("COMP-008 requires the owned exact renderer='cuda'")
    if _flag(header, "support_fade"):
        raise OracleError("COMP-008 freezes support_fade=False")
    if "covariance_filter_materialized" in header:
        raise OracleError("COMP-008 excludes covariance-filtered fields")


def make_oracle_row(
    blob: bytes,
    image_id: str,
    *,
    expected_n: int | None = FROZEN_GAUSSIAN_COUNT,
    require_frozen_bits: bool = True,
    require_prepared_extent: bool = True,
    canonical_zlib9: bool = True,
) -> dict[str, Any]:
    """Create one self-hashed oracle row from a complete SSPL1 baseline."""
    if not image_id or not isinstance(image_id, str):
        raise OracleError("image_id must be a non-empty string")
    if expected_n is not None and image_id not in FROZEN_IMAGE_IDS:
        raise OracleError(f"image_id {image_id!r} is outside the frozen development set")
    parsed = parse_sspl1(blob, canonical_zlib9=canonical_zlib9)
    header = parsed.header
    bit_tuple = tuple(int(value) for value in header["bits"])
    if require_frozen_bits and bit_tuple not in FROZEN_BIT_TUPLES:
        raise OracleError(f"bit tuple {bit_tuple} is outside the frozen COMP-008 grid")
    n = int(header["n"])
    if expected_n is not None and n != expected_n:
        raise OracleError(f"SSPL1 n={n}, expected {expected_n}")
    if require_prepared_extent and max(int(header["H"]), int(header["W"])) != 768:
        raise OracleError("prepared image must have exact max-side 768")
    required_flags = {
        "morton": True,
        "color_delta": True,
        "means_byte_planar": True,
        "rot_circular": True,
    }
    for name, expected in required_flags.items():
        if _flag(header, name) is not expected:
            raise OracleError(f"SSPL1 {name} must be {expected} for COMP-008")
    _validate_ssp2e_header_representability(header)

    entropy = conditional_entropy_certificate(parsed.symbols, bit_tuple[0])
    entropy_bytes = int(entropy["entropy_floor_bytes_exact"])
    lower_bound = (
        SSP2E_COMPULSORY_BYTES
        + parsed.payload_bytes["means"]
        + parsed.payload_bytes["rotation"]
        + entropy_bytes
    )
    row = {
        "schema": ROW_SCHEMA,
        "image_id": image_id,
        "bit_tuple": list(bit_tuple),
        "n": n,
        "height": int(header["H"]),
        "width": int(header["W"]),
        "sspl1_sha256": parsed.blob_sha256,
        "symbol_sha256": parsed.symbol_sha256,
        "baseline_complete_sspl1_bytes": parsed.total_bytes,
        "sspl1_header_bytes": parsed.header_bytes,
        "sspl1_payload_bytes": dict(parsed.payload_bytes),
        "sspl1_raw_bytes": dict(parsed.raw_bytes),
        "ssp2e_compulsory_bytes": SSP2E_COMPULSORY_BYTES,
        "candidate_context_scope": (
            "five independent static streams; CDF depends only on bit tuple, channel, "
            "absolute-mean cell, transmitted grid/head, and frozen table"
        ),
        "entropy": entropy,
        "candidate_lower_bound_bytes": lower_bound,
        "ratio_numerator": lower_bound,
        "ratio_denominator": parsed.total_bytes,
        "strict_byte_win": lower_bound < parsed.total_bytes,
    }
    return seal_record(row)


@lru_cache(maxsize=1)
def bootstrap_index_matrix() -> np.ndarray:
    """Return the frozen, read-only 100000x8 uint8 bootstrap index matrix."""
    matrix = np.empty((BOOTSTRAP_REPLICATES, FROZEN_IMAGE_COUNT), dtype=np.uint8)
    for replicate in range(BOOTSTRAP_REPLICATES):
        digest = hashlib.sha256(BOOTSTRAP_DOMAIN + struct.pack("<Q", replicate)).digest()
        for column in range(FROZEN_IMAGE_COUNT):
            matrix[replicate, column] = digest[column] & 7
    observed = sha256_bytes(matrix.tobytes(order="C"))
    if observed != BOOTSTRAP_MATRIX_SHA256:
        raise RuntimeError(f"frozen bootstrap matrix hash drifted: {observed}")
    matrix.setflags(write=False)
    return matrix


def _compare_ratio_product(
    numerators: Sequence[int],
    denominators: Sequence[int],
    threshold_numerator: int,
    threshold_denominator: int,
) -> int:
    """Compare a product of ratios with threshold**len, returning -1/0/+1."""
    if len(numerators) != len(denominators) or not numerators:
        raise OracleError("ratio product requires equally sized non-empty vectors")
    left = math.prod(int(value) for value in numerators) * threshold_denominator ** len(numerators)
    right = math.prod(int(value) for value in denominators) * threshold_numerator ** len(
        denominators
    )
    return (left > right) - (left < right)


def _bootstrap_summary(numerators: Sequence[int], denominators: Sequence[int]) -> dict[str, Any]:
    matrix = bootstrap_index_matrix()
    log_ratios = np.log(np.asarray(numerators, dtype=np.float64)) - np.log(
        np.asarray(denominators, dtype=np.float64)
    )
    sample_gm = np.exp(np.mean(log_ratios[matrix], axis=1))
    upper_diagnostic = float(np.sort(sample_gm)[BOOTSTRAP_UPPER_INDEX])

    # Gate the order statistic exactly.  Repeated index multisets share one exact product, so at
    # most C(15, 7)=6435 big-integer comparisons are needed rather than 100000.
    comparison_cache: dict[tuple[int, ...], int] = {}
    below = equal = above = 0
    for sample in matrix:
        key = tuple(sorted(int(index) for index in sample))
        comparison = comparison_cache.get(key)
        if comparison is None:
            comparison = _compare_ratio_product(
                [numerators[index] for index in key],
                [denominators[index] for index in key],
                19,
                20,
            )
            comparison_cache[key] = comparison
        if comparison < 0:
            below += 1
        elif comparison == 0:
            equal += 1
        else:
            above += 1
    strict_pass = below >= BOOTSTRAP_UPPER_INDEX + 1
    return {
        "domain_hex": BOOTSTRAP_DOMAIN.hex(),
        "replicates": BOOTSTRAP_REPLICATES,
        "sample_size": FROZEN_IMAGE_COUNT,
        "matrix_dtype": "uint8",
        "matrix_shape": [BOOTSTRAP_REPLICATES, FROZEN_IMAGE_COUNT],
        "matrix_sha256": BOOTSTRAP_MATRIX_SHA256,
        "upper_order_index_zero_based": BOOTSTRAP_UPPER_INDEX,
        "upper_geometric_mean_ratio_diagnostic": upper_diagnostic,
        "samples_strictly_below_0_95": below,
        "samples_equal_0_95": equal,
        "samples_strictly_above_0_95": above,
        "upper_strictly_below_0_95_exact": strict_pass,
        "unique_index_multisets": len(comparison_cache),
    }


def _validate_analysis_rows(
    rows: Iterable[Mapping[str, Any]],
) -> dict[tuple[int, ...], list[dict[str, Any]]]:
    groups: dict[tuple[int, ...], list[dict[str, Any]]] = {}
    seen_cells: set[tuple[str, tuple[int, ...]]] = set()
    for raw in rows:
        row = dict(raw)
        verify_record(row)
        if row.get("schema") != ROW_SCHEMA:
            raise OracleError(f"unexpected row schema: {row.get('schema')!r}")
        bits_value = row.get("bit_tuple")
        if (
            not isinstance(bits_value, list)
            or len(bits_value) != 4
            or any(isinstance(value, bool) or not isinstance(value, int) for value in bits_value)
        ):
            raise OracleError("row bit_tuple must contain four integers")
        bit_tuple = tuple(int(value) for value in bits_value)
        if bit_tuple not in FROZEN_BIT_TUPLES:
            raise OracleError(f"row bit tuple {bit_tuple} is outside the frozen grid")
        image_id = row.get("image_id")
        if not isinstance(image_id, str) or not image_id:
            raise OracleError("row image_id must be a non-empty string")
        cell = (image_id, bit_tuple)
        if cell in seen_cells:
            raise OracleError(f"duplicate oracle cell: {cell}")
        seen_cells.add(cell)
        numerator = row.get("ratio_numerator")
        denominator = row.get("ratio_denominator")
        if (
            isinstance(numerator, bool)
            or not isinstance(numerator, int)
            or numerator <= 0
            or isinstance(denominator, bool)
            or not isinstance(denominator, int)
            or denominator <= 0
        ):
            raise OracleError("row ratio must contain positive integer numerator/denominator")
        if row.get("candidate_lower_bound_bytes") != numerator:
            raise OracleError("row ratio numerator is not the candidate lower bound")
        if row.get("baseline_complete_sspl1_bytes") != denominator:
            raise OracleError("row ratio denominator is not the complete SSPL1 size")
        if row.get("strict_byte_win") is not (numerator < denominator):
            raise OracleError("row strict_byte_win disagrees with exact bytes")
        groups.setdefault(bit_tuple, []).append(row)

    if set(groups) != set(FROZEN_BIT_TUPLES):
        raise OracleError("analysis requires both frozen bit tuples")
    image_sets = []
    for bit_tuple in FROZEN_BIT_TUPLES:
        group = groups[bit_tuple]
        if len(group) != FROZEN_IMAGE_COUNT:
            raise OracleError(f"bit tuple {bit_tuple} has {len(group)} rows, expected 8")
        image_order = {image_id: index for index, image_id in enumerate(FROZEN_IMAGE_IDS)}
        group.sort(key=lambda row: image_order.get(str(row["image_id"]), FROZEN_IMAGE_COUNT))
        image_sets.append(tuple(str(row["image_id"]) for row in group))
    expected_images = FROZEN_IMAGE_IDS
    if image_sets[0] != expected_images or image_sets[1] != expected_images:
        raise OracleError("rows do not cover the exact frozen development image IDs")
    return groups


def _validate_rows_against_blobs(
    groups: Mapping[tuple[int, ...], Sequence[Mapping[str, Any]]],
    blobs: Mapping[tuple[str, tuple[int, int, int, int]], bytes],
) -> None:
    expected_cells = {
        (image_id, bit_tuple) for bit_tuple in FROZEN_BIT_TUPLES for image_id in FROZEN_IMAGE_IDS
    }
    if set(blobs) != expected_cells:
        missing = sorted(expected_cells - set(blobs))
        extra = sorted(set(blobs) - expected_cells)
        raise OracleError(f"SSPL1 blob grid mismatch: missing={missing}, extra={extra}")
    for bit_tuple in FROZEN_BIT_TUPLES:
        for row in groups[bit_tuple]:
            image_id = str(row["image_id"])
            blob = blobs[(image_id, bit_tuple)]
            if not isinstance(blob, bytes):
                raise OracleError(f"SSPL1 blob for {(image_id, bit_tuple)} is not bytes")
            recomputed = make_oracle_row(blob, image_id)
            if canonical_json(recomputed) != canonical_json(dict(row)):
                raise OracleError(f"artifact-backed oracle row differs for {(image_id, bit_tuple)}")


def _analyze_validated_groups(
    groups: Mapping[tuple[int, ...], list[dict[str, Any]]],
) -> dict[str, Any]:
    """Reduce already artifact-validated groups; public callers must use :func:`analyze_rows`."""
    tuple_results: list[dict[str, Any]] = []
    for bit_tuple in FROZEN_BIT_TUPLES:
        group = groups[bit_tuple]
        numerators = [int(row["ratio_numerator"]) for row in group]
        denominators = [int(row["ratio_denominator"]) for row in group]
        logs = [
            math.log(numerator) - math.log(denominator)
            for numerator, denominator in zip(numerators, denominators, strict=True)
        ]
        geometric_mean_diagnostic = math.exp(sum(logs) / len(logs))
        gm_pass = _compare_ratio_product(numerators, denominators, 9, 10) <= 0
        wins = sum(
            numerator < denominator
            for numerator, denominator in zip(numerators, denominators, strict=True)
        )
        worst_index = max(
            range(len(group)),
            key=lambda index: Fraction(numerators[index], denominators[index]),
        )
        worst_pass = 50 * numerators[worst_index] <= 51 * denominators[worst_index]
        bootstrap = _bootstrap_summary(numerators, denominators)
        gates = {
            "geometric_mean_lte_0_90_exact": gm_pass,
            "strict_image_wins_gte_7": wins >= 7,
            "worst_ratio_lte_1_02_exact": worst_pass,
            "bootstrap_upper_lt_0_95_exact": bool(bootstrap["upper_strictly_below_0_95_exact"]),
        }
        passed = all(gates.values())
        tuple_results.append(
            {
                "bit_tuple": list(bit_tuple),
                "image_ids": [str(row["image_id"]) for row in group],
                "ratio_numerators": numerators,
                "ratio_denominators": denominators,
                "geometric_mean_ratio_diagnostic": geometric_mean_diagnostic,
                "strict_image_wins": wins,
                "worst_image_id": str(group[worst_index]["image_id"]),
                "worst_ratio_diagnostic": numerators[worst_index] / denominators[worst_index],
                "bootstrap": bootstrap,
                "gates": gates,
                "all_gates_pass": passed,
                "decision": (
                    "SURVIVE_NECESSARY_CONDITION_SCREEN"
                    if passed
                    else "REJECT_TUPLE_UNDER_FROZEN_MARGIN"
                ),
            }
        )

    both_survive = all(result["all_gates_pass"] for result in tuple_results)
    rejected = [result["bit_tuple"] for result in tuple_results if not result["all_gates_pass"]]
    survived = [result["bit_tuple"] for result in tuple_results if result["all_gates_pass"]]
    analysis = {
        "schema": ANALYSIS_SCHEMA,
        "scientific_scope": "necessary-rate-lower-bound-only",
        "row_count": 2 * FROZEN_IMAGE_COUNT,
        "frozen_bit_tuples": [list(value) for value in FROZEN_BIT_TUPLES],
        "tuple_results": tuple_results,
        "decision": (
            "ORACLE_INCONCLUSIVE_IMPLEMENT_CODER"
            if both_survive
            else "ABANDON_JOINTLY_REQUIRED_TWO_TUPLE_COMP008_CANDIDATE"
        ),
        "coder_authorized": both_survive,
        "rejected_bit_tuples": rejected,
        "surviving_bit_tuples": survived,
        "limitations": [
            "No SSP2E range coder or learned mean/scale head is implemented.",
            "Entropy and bootstrap floating-point values are diagnostics; gates are exact.",
            "A surviving lower bound is necessary but not sufficient for an actual-rate win.",
            "A rejected tuple is scoped to its fixed QAT field, ordered symbols, 16x16 cells, "
            "five independent marginal streams, and 656-byte SSP2E candidate.",
            "The joint policy rejection does not call any surviving tuple impossible.",
        ],
    }
    return seal_record(analysis, "analysis_sha256")


def _analyze_unbacked_test_rows(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Exercise exact gate math without emitting an actionable scientific artifact."""
    scientific = _analyze_validated_groups(_validate_analysis_rows(rows))
    result = dict(scientific)
    result.pop("analysis_sha256", None)
    result["schema"] = "structsplat.comp008.sgi-entropy-oracle-analysis.test-only.v1"
    result["scientific_scope"] = "unbacked-test-only-no-decision"
    result["would_be_decision"] = result["decision"]
    result["decision"] = "TEST_ONLY_NO_SCIENTIFIC_DECISION"
    result["coder_authorized"] = False
    return seal_record(result, "analysis_sha256")


def analyze_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    blobs: Mapping[tuple[str, tuple[int, int, int, int]], bytes] | None = None,
) -> dict[str, Any]:
    """Reparse every persisted blob, then apply all four frozen KILL gates."""
    groups = _validate_analysis_rows(rows)
    if blobs is None:
        raise OracleError("scientific analysis requires all 16 persisted SSPL1 blobs")
    _validate_rows_against_blobs(groups, blobs)
    return _analyze_validated_groups(groups)


def replay_analysis(
    rows: Iterable[Mapping[str, Any]],
    recorded: Mapping[str, Any],
    *,
    blobs: Mapping[tuple[str, tuple[int, int, int, int]], bytes] | None = None,
) -> dict[str, Any]:
    """Recompute an analysis and report byte-exact canonical replay."""
    verify_record(recorded, "analysis_sha256")
    recomputed = analyze_rows(rows, blobs=blobs)
    exact = canonical_json(recomputed) == canonical_json(dict(recorded))
    replay = {
        "schema": REPLAY_SCHEMA,
        "analysis_exact": exact,
        "recorded_analysis_sha256": recorded["analysis_sha256"],
        "recomputed_analysis_sha256": recomputed["analysis_sha256"],
        "bootstrap_matrix_sha256": sha256_bytes(bootstrap_index_matrix().tobytes(order="C")),
        "ok": exact,
    }
    return seal_record(replay, "replay_sha256")


def replay_row(
    blob: bytes,
    recorded: Mapping[str, Any],
    *,
    expected_n: int | None = FROZEN_GAUSSIAN_COUNT,
    require_prepared_extent: bool = True,
) -> dict[str, Any]:
    """Recreate one row from its SSPL1 bytes and compare canonical records."""
    verify_record(recorded)
    recomputed = make_oracle_row(
        blob,
        str(recorded.get("image_id", "")),
        expected_n=expected_n,
        require_prepared_extent=require_prepared_extent,
    )
    exact = canonical_json(recomputed) == canonical_json(dict(recorded))
    return {
        "row_exact": exact,
        "recorded_sha256": recorded["record_sha256"],
        "recomputed_sha256": recomputed["record_sha256"],
    }


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise OracleError(f"{path} must contain one JSON object")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise OracleError(f"{path}:{number} is not a JSON object")
        rows.append(value)
    return rows


def _emit(value: Mapping[str, Any], output: Path | None) -> None:
    payload = canonical_json(dict(value))
    if output is None:
        print(payload.decode("utf-8"), end="")
    else:
        output.write_bytes(payload)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    row = subparsers.add_parser("row", help="create one oracle row from a complete SSPL1 blob")
    row.add_argument("--blob", type=Path, required=True)
    row.add_argument("--image-id", required=True)
    row.add_argument(
        "--expected-n",
        type=int,
        default=FROZEN_GAUSSIAN_COUNT,
        help="set to 0 only for synthetic diagnostics",
    )
    row.add_argument("--output", type=Path)

    analyze = subparsers.add_parser("analyze", help="analyze the frozen 16-row JSONL grid")
    analyze.add_argument("--rows", type=Path, required=True)
    analyze.add_argument("--output", type=Path)

    replay = subparsers.add_parser("replay", help="replay one recorded aggregate analysis")
    replay.add_argument("--rows", type=Path, required=True)
    replay.add_argument("--analysis", type=Path, required=True)
    replay.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "row":
        expected_n = None if args.expected_n == 0 else args.expected_n
        value = make_oracle_row(args.blob.read_bytes(), args.image_id, expected_n=expected_n)
    elif args.command == "analyze":
        value = analyze_rows(_read_jsonl(args.rows))
    elif args.command == "replay":
        value = replay_analysis(_read_jsonl(args.rows), _read_json(args.analysis))
    else:  # pragma: no cover - argparse prevents this branch
        raise AssertionError(args.command)
    _emit(value, args.output)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
