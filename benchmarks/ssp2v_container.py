"""Exact COMP-011 SSP2Z/SSP2V v1 containers and RGB reconstruction.

This module is deliberately limited to byte grammar, empirical index models, arithmetic-index
coding, and exact symbol reconstruction.  It does not fit a codebook, render an image, choose a
candidate, or own lifecycle policy.  The only source format accepted by the encoders is a strict
canonical SSPL1 byte string; callers must enforce the COMP-011 stage guards before supplying one.

The 100-byte wire header intentionally has its own :class:`SSP2VHeader` semantic type.  Its fourth
auxiliary integer is ``aux_model_id=0`` and must never be confused with COMP-009's
``cdf_table_id=1`` even though the physical struct is shared.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
import zlib
from dataclasses import asdict, dataclass
from typing import Callable, Mapping, Sequence

import numpy as np

from benchmarks import sgi_entropy_oracle as oracle
from benchmarks import ssp2e_arithmetic as arithmetic
from benchmarks import ssp2e_container as comp009_container


OUTER = struct.Struct("<5sBBBQI")
HEADER = struct.Struct("<III4B4i12f5I")
DIRECTORY = struct.Struct("<IQQ")
DESCRIPTOR = struct.Struct("<4sBBBBHHI")

VERSION = 1
FLAGS = 0
RENDERER_CUDA = 1
SUPPORT_FADE = 0
AUX_MODEL_ID = 0
ZLIB_LEVEL = 9
TOTAL_FREQUENCY = arithmetic.TOTAL_FREQUENCY
ROW_COUNT = 8192

SSP2Z_MAGIC = b"SSP2Z"
SSP2V_MAGIC = b"SSP2V"
ALL_MAGICS = (SSP2Z_MAGIC, SSP2V_MAGIC)

SSP2Z_STREAMS = ("means_zlib", "scales_zlib", "rotation_zlib", "colors_zlib")
SSP2Z_IDS = (1, 2, 3, 4)
SSP2Z_PREFIX_BYTES = OUTER.size + HEADER.size + len(SSP2Z_STREAMS) * DIRECTORY.size

SSP2V_FIXED_STREAMS = (
    "means_zlib",
    "scales_zlib",
    "rotation_zlib",
    "descriptor_raw",
    "codebook_raw",
    "index_model_zlib",
)
LEGAL_INDEX_ALPHABETS = (64, 128, 192, 320, 384, 640)
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

if OUTER.size != 20 or HEADER.size != 100 or DIRECTORY.size != 20 or DESCRIPTOR.size != 16:
    raise RuntimeError("COMP-011 struct layout drift")
if SSP2Z_PREFIX_BYTES != 200:
    raise RuntimeError("COMP-011 SSP2Z prefix drift")


class ContainerError(ValueError):
    """A COMP-011 container, model, codebook, or adapter contract was violated."""


ArithmeticEncode = Callable[[np.ndarray, np.ndarray], bytes]
ArithmeticDecode = Callable[[bytes, np.ndarray, int], np.ndarray]


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


def _tuple(value: object, size: int, name: str) -> tuple[object, ...]:
    if not isinstance(value, (tuple, list)) or len(value) != size:
        raise ContainerError(f"{name} must contain exactly {size} values")
    return tuple(value)


def _packed_dtype(bits: int) -> np.dtype:
    if not 1 <= bits <= 32:
        raise ContainerError("bit depth must lie in [1, 32]")
    return np.dtype("u1" if bits <= 8 else "<u2" if bits <= 16 else "<u4")


@dataclass(frozen=True)
class SSP2VHeader:
    """Semantic view of the COMP-011 common 100-byte header."""

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
    support_fade: int = SUPPORT_FADE
    aux_model_id: int = AUX_MODEL_ID
    zlib_level: int = ZLIB_LEVEL

    def validated(self) -> SSP2VHeader:
        """Return ``self`` after enforcing every frozen header invariant."""

        if _positive_uint(self.n, "n", (1 << 32) - 1) != ROW_COUNT:
            raise ContainerError(f"COMP-011 requires n={ROW_COUNT}")
        _positive_uint(self.height, "height", (1 << 32) - 1)
        _positive_uint(self.width, "width", (1 << 32) - 1)
        bits_raw = _tuple(self.bits, 4, "bits")
        bits = tuple(_positive_uint(value, f"bits[{i}]", 255) for i, value in enumerate(bits_raw))
        if bits not in oracle.FROZEN_BIT_TUPLES:
            raise ContainerError(f"unsupported COMP-011 bit tuple: {bits}")

        endpoints: dict[str, tuple[int, int]] = {}
        for name, values in (("means_lo", self.means_lo), ("means_hi", self.means_hi)):
            pair = _tuple(values, 2, name)
            parsed: list[int] = []
            for index, value in enumerate(pair):
                if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                    raise ContainerError(f"{name}[{index}] must be an int32")
                integer = int(value)
                if not -(1 << 31) <= integer < 1 << 31:
                    raise ContainerError(f"{name}[{index}] must be an int32")
                parsed.append(integer)
            endpoints[name] = (parsed[0], parsed[1])
        means_lo = endpoints["means_lo"]
        means_hi = endpoints["means_hi"]
        if (
            means_lo[0] > means_hi[0]
            or means_lo[1] > means_hi[1]
            or means_lo[0] > 0
            or means_lo[1] > 0
            or means_hi[0] < self.width - 1
            or means_hi[1] < self.height - 1
        ):
            raise ContainerError("mean extent does not contain the image box")

        float_groups = (
            ("scale_lo", self.scale_lo, 2),
            ("scale_hi", self.scale_hi, 2),
            ("color_lo", self.color_lo, 3),
            ("color_hi", self.color_hi, 3),
        )
        normalized: dict[str, tuple[float, ...]] = {}
        for name, values, size in float_groups:
            normalized[name] = tuple(
                _float32(value, f"{name}[{index}]")
                for index, value in enumerate(_tuple(values, size, name))
            )
        if any(
            lower > upper
            for lower, upper in zip(normalized["scale_lo"], normalized["scale_hi"], strict=True)
        ):
            raise ContainerError("scale range is reversed")
        if any(
            lower > upper
            for lower, upper in zip(normalized["color_lo"], normalized["color_hi"], strict=True)
        ):
            raise ContainerError("color range is reversed")
        aa_dilation = _float32(self.aa_dilation, "aa_dilation")
        sigma_cutoff = _float32(self.sigma_cutoff, "sigma_cutoff")
        if aa_dilation < 0:
            raise ContainerError("aa_dilation must be nonnegative")
        if sigma_cutoff <= 0:
            raise ContainerError("sigma_cutoff must be positive")
        _positive_uint(self.render_chunk, "render_chunk", (1 << 32) - 1)
        if self.renderer_id != RENDERER_CUDA:
            raise ContainerError("renderer_id must identify the frozen CUDA renderer")
        if self.support_fade != SUPPORT_FADE:
            raise ContainerError("support_fade must be zero")
        if self.aux_model_id != AUX_MODEL_ID:
            raise ContainerError("aux_model_id must be zero")
        if self.zlib_level != ZLIB_LEVEL:
            raise ContainerError("zlib_level must be nine")
        return self

    @classmethod
    def from_sspl1(cls, header: Mapping[str, object]) -> SSP2VHeader:
        """Convert strict production SSPL1 semantics into the distinct COMP-011 type."""

        try:
            legacy = comp009_container.ContainerHeader.from_sspl1(header)
        except (comp009_container.ContainerError, TypeError, ValueError) as exc:
            raise ContainerError(f"SSPL1 header is not SSP2V-representable: {exc}") from exc
        result = cls(
            n=legacy.n,
            height=legacy.height,
            width=legacy.width,
            bits=legacy.bits,
            means_lo=legacy.means_lo,
            means_hi=legacy.means_hi,
            scale_lo=legacy.scale_lo,
            scale_hi=legacy.scale_hi,
            color_lo=legacy.color_lo,
            color_hi=legacy.color_hi,
            aa_dilation=legacy.aa_dilation,
            sigma_cutoff=legacy.sigma_cutoff,
            render_chunk=legacy.render_chunk,
        ).validated()
        reconstructed = result.to_sspl1_header()
        if (
            tuple(header) != tuple(reconstructed)
            or json.dumps(header).encode() != json.dumps(reconstructed).encode()
        ):
            raise ContainerError("COMP-011 header cannot reproduce exact SSPL1 semantics")
        return result

    def raw_stream_lengths(self) -> tuple[int, int, int, int]:
        self.validated()
        bm, bs, br, bc = self.bits
        return (
            2 * self.n * _packed_dtype(bm).itemsize,
            2 * self.n * _packed_dtype(bs).itemsize,
            self.n * _packed_dtype(br).itemsize,
            3 * self.n * _packed_dtype(bc).itemsize,
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
            self.aux_model_id,
            self.zlib_level,
        )
        try:
            return HEADER.pack(*values)
        except struct.error as exc:  # pragma: no cover - validated above
            raise ContainerError(f"cannot serialize COMP-011 header: {exc}") from exc

    @classmethod
    def from_bytes(cls, payload: bytes) -> SSP2VHeader:
        if not isinstance(payload, bytes) or len(payload) != HEADER.size:
            raise ContainerError("common header must contain exactly 100 immutable bytes")
        values = HEADER.unpack(payload)
        return cls(
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
            aux_model_id=values[26],
            zlib_level=values[27],
        ).validated()


@dataclass(frozen=True)
class SSP2VDescriptor:
    """The exact 16-byte descriptor for one of the eight frozen VQ variants."""

    family_id: int
    actual_stage_count: int
    matched_rvq_stage_count: int
    base_k: int
    codebook_entries: int
    descriptor_version: int = VERSION
    reserved: int = 0
    tag: bytes = b"VQ1\0"

    def validated(self) -> SSP2VDescriptor:
        if self.tag != b"VQ1\0":
            raise ContainerError("SSP2V descriptor tag mismatch")
        if self.descriptor_version != VERSION:
            raise ContainerError("unsupported SSP2V descriptor version")
        family = _uint(self.family_id, "family_id", 255)
        stages = _positive_uint(self.actual_stage_count, "actual_stage_count", 255)
        matched = _positive_uint(self.matched_rvq_stage_count, "matched_rvq_stage_count", 255)
        base_k = _positive_uint(self.base_k, "base_k", (1 << 16) - 1)
        entries = _positive_uint(self.codebook_entries, "codebook_entries", (1 << 16) - 1)
        if self.reserved != 0:
            raise ContainerError("SSP2V descriptor reserved field must be zero")
        if family not in (0, 1) or matched not in (2, 3) or base_k not in (64, 128):
            raise ContainerError("descriptor is outside the frozen eight-variant menu")
        expected_stages = 1 if family == 0 else matched
        expected_entries = (2 * matched - 1) * base_k if family == 0 else matched * base_k
        if stages != expected_stages or entries != expected_entries:
            raise ContainerError("descriptor stage or codebook-entry count disagrees with menu")
        return self

    @property
    def family(self) -> str:
        self.validated()
        return "flat" if self.family_id == 0 else "rvq"

    @property
    def stage_alphabets(self) -> tuple[int, ...]:
        self.validated()
        if self.family_id == 0:
            return (self.codebook_entries,)
        return (self.base_k,) * self.actual_stage_count

    @property
    def codebook_bytes(self) -> int:
        self.validated()
        if self.family_id == 0:
            return 3 * self.codebook_entries
        return 3 * self.base_k + 6 * self.base_k * (self.actual_stage_count - 1)

    @property
    def label(self) -> str:
        self.validated()
        if self.family_id == 0:
            return f"flat_s{self.matched_rvq_stage_count}_k{self.base_k}_l{self.codebook_entries}"
        return f"rvq_s{self.matched_rvq_stage_count}_k{self.base_k}"

    def to_bytes(self) -> bytes:
        self.validated()
        return DESCRIPTOR.pack(
            self.tag,
            self.descriptor_version,
            self.family_id,
            self.actual_stage_count,
            self.matched_rvq_stage_count,
            self.base_k,
            self.codebook_entries,
            self.reserved,
        )

    @classmethod
    def from_bytes(cls, payload: bytes) -> SSP2VDescriptor:
        if not isinstance(payload, bytes) or len(payload) != DESCRIPTOR.size:
            raise ContainerError("SSP2V descriptor must contain exactly 16 immutable bytes")
        tag, version, family, stages, matched, base_k, entries, reserved = DESCRIPTOR.unpack(
            payload
        )
        return cls(
            family_id=family,
            actual_stage_count=stages,
            matched_rvq_stage_count=matched,
            base_k=base_k,
            codebook_entries=entries,
            descriptor_version=version,
            reserved=reserved,
            tag=tag,
        ).validated()


@dataclass(frozen=True)
class SSPL1Parts:
    """Strict source state needed to create either COMP-011 container."""

    header: SSP2VHeader
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
    header: SSP2VHeader
    entries: tuple[DirectoryEntry, ...]
    payloads: dict[str, bytes]
    total_bytes: int
    crc32: int
    descriptor: SSP2VDescriptor | None = None

    @property
    def prefix_bytes(self) -> int:
        return OUTER.size + HEADER.size + DIRECTORY.size * len(self.entries)

    @property
    def payload_bytes(self) -> int:
        return sum(entry.length for entry in self.entries)

    def component_bytes(self) -> dict[str, int]:
        components = {
            "outer": OUTER.size,
            "header": HEADER.size,
            "directory": DIRECTORY.size * len(self.entries),
        }
        components.update({entry.name: entry.length for entry in self.entries})
        components["total"] = self.total_bytes
        if sum(value for key, value in components.items() if key != "total") != self.total_bytes:
            raise ContainerError("component accounting does not sum to complete bytes")
        return components


@dataclass(frozen=True)
class RGBAccounting:
    """Exact final-clipped RGB error, clipping, and collision diagnostics."""

    rgb_sse: int
    sum_abs_error: int
    sum_signed_error: int
    max_abs_error: int
    clip_low_components: int
    clip_high_components: int
    clip_components: int
    clip_rows: int
    distinct_input_rgb: int
    distinct_reconstructed_rgb: int
    merged_input_colors: int
    collision_pairs: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class DecodedContainer:
    parsed: ParsedContainer
    symbols: dict[str, np.ndarray]
    symbol_sha256: str
    canonical_blob_sha256: str
    codebooks: tuple[np.ndarray, ...] = ()
    indices: tuple[np.ndarray, ...] = ()
    raw_rgb: np.ndarray | None = None
    rgb_accounting: RGBAccounting | None = None
    coder_certificates: tuple[dict[str, int | str | bool], ...] = ()
    used_entries_per_stage: tuple[int, ...] = ()
    unique_index_tuples: int = 0


def _symbol_sha256(symbols: Mapping[str, np.ndarray]) -> str:
    if set(symbols) != set(SYMBOL_CHANNELS):
        raise ContainerError("symbol digest requires all eight absolute arrays")
    digest = hashlib.sha256()
    for name in SYMBOL_CHANNELS:
        values = np.ascontiguousarray(symbols[name], dtype="<u4")
        if values.shape != (ROW_COUNT,):
            raise ContainerError(f"{name} must contain exactly {ROW_COUNT} symbols")
        digest.update(name.encode("ascii") + b"\0")
        digest.update(struct.pack("<Q", int(values.size)))
        digest.update(values.tobytes(order="C"))
    return digest.hexdigest()


def _strict_zlib(payload: bytes, expected_bytes: int, name: str) -> bytes:
    if not isinstance(payload, bytes) or not payload:
        raise ContainerError(f"{name} must be nonempty immutable bytes")
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
    """Strictly extract canonical SSPL1 symbols/payloads and enforce ``N=8192``."""

    try:
        legacy = comp009_container.extract_sspl1_parts(blob)
    except comp009_container.ContainerError as exc:
        raise ContainerError(f"invalid SSPL1 source: {exc}") from exc
    header = SSP2VHeader.from_sspl1(legacy.original_header)
    return SSPL1Parts(
        header=header,
        symbols={
            name: np.ascontiguousarray(values, dtype=np.uint32)
            for name, values in legacy.symbols.items()
        },
        payloads=dict(legacy.payloads),
        blob_sha256=legacy.blob_sha256,
        symbol_sha256=legacy.symbol_sha256,
        total_bytes=legacy.total_bytes,
    )


def empirical_frequencies(indices: np.ndarray, alphabet: int) -> np.ndarray:
    """Return the frozen positive empirical frequencies for one 8192-index stream."""

    alphabet = _positive_uint(alphabet, "index alphabet", (1 << 16) - 1)
    if alphabet not in LEGAL_INDEX_ALPHABETS:
        raise ContainerError(f"index alphabet {alphabet} is outside the frozen menu")
    values = np.asarray(indices)
    if values.shape != (ROW_COUNT,) or values.dtype.kind not in "iu":
        raise ContainerError(f"empirical indices must be an integer vector of length {ROW_COUNT}")
    if values.dtype.kind == "i" and np.any(values < 0):
        raise ContainerError("empirical index is negative")
    if values.size and int(values.max()) >= alphabet:
        raise ContainerError("empirical index exceeds its alphabet")
    counts = np.bincount(values.astype(np.int64), minlength=alphabet)
    residual_total = TOTAL_FREQUENCY - alphabet
    products = residual_total * counts
    allocation = products // ROW_COUNT
    remainders = products % ROW_COUNT
    remaining = residual_total - int(allocation.sum())
    order = sorted(range(alphabet), key=lambda index: (-int(remainders[index]), index))
    if remaining:
        allocation[np.asarray(order[:remaining], dtype=np.int64)] += 1
    frequencies = np.ascontiguousarray(allocation + 1, dtype=np.uint32)
    if np.any(frequencies == 0) or int(frequencies.sum()) != TOTAL_FREQUENCY:
        raise ContainerError("empirical frequency normalization invariant failed")
    return frequencies


def cdf_rows(frequencies: np.ndarray) -> np.ndarray:
    """Materialize the exact constant CDF matrix consumed by the COMP-009 coder."""

    values = np.asarray(frequencies)
    if (
        values.ndim != 1
        or values.size not in LEGAL_INDEX_ALPHABETS
        or values.dtype.kind not in "iu"
    ):
        raise ContainerError("frequency vector has an unsupported alphabet")
    if values.dtype.kind == "i" and np.any(values <= 0):
        raise ContainerError("every frequency must be positive")
    if np.any(values == 0) or int(values.astype(object).sum()) != TOTAL_FREQUENCY:
        raise ContainerError("frequency vector must be positive and sum to 32768")
    cdf = np.concatenate((np.zeros(1, dtype=np.uint32), np.cumsum(values, dtype=np.uint32)))
    return np.repeat(cdf[None, :], ROW_COUNT, axis=0)


def _index_vector(values: object, alphabet: int, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.shape != (ROW_COUNT,) or array.dtype.kind not in "iu":
        raise ContainerError(f"{name} must be an integer vector of length {ROW_COUNT}")
    if array.dtype.kind == "i" and np.any(array < 0):
        raise ContainerError(f"{name} contains a negative index")
    if array.size and int(array.max()) >= alphabet:
        raise ContainerError(f"{name} exceeds alphabet {alphabet}")
    return np.ascontiguousarray(array, dtype=np.uint32)


def serialize_index_model(indices: Sequence[np.ndarray], descriptor: SSP2VDescriptor) -> bytes:
    descriptor.validated()
    if not isinstance(indices, Sequence) or isinstance(indices, (bytes, bytearray, str)):
        raise ContainerError("indices must be a stage sequence")
    if len(indices) != descriptor.actual_stage_count:
        raise ContainerError("index stage count disagrees with descriptor")
    raw = bytearray()
    for stage, (values, alphabet) in enumerate(
        zip(indices, descriptor.stage_alphabets, strict=True)
    ):
        normalized = _index_vector(values, alphabet, f"index_{stage}")
        frequencies = empirical_frequencies(normalized, alphabet)
        raw.extend(frequencies.astype("<u2", copy=False).tobytes(order="C"))
    payload = zlib.compress(bytes(raw), level=ZLIB_LEVEL)
    _strict_zlib(payload, 2 * sum(descriptor.stage_alphabets), "index model")
    return payload


def parse_index_model(payload: bytes, descriptor: SSP2VDescriptor) -> tuple[np.ndarray, ...]:
    descriptor.validated()
    expected = 2 * sum(descriptor.stage_alphabets)
    raw = _strict_zlib(payload, expected, "index model")
    result: list[np.ndarray] = []
    offset = 0
    for stage, alphabet in enumerate(descriptor.stage_alphabets):
        end = offset + 2 * alphabet
        frequencies = np.frombuffer(raw[offset:end], dtype="<u2").astype(np.uint32)
        if np.any(frequencies == 0) or int(frequencies.sum()) != TOTAL_FREQUENCY:
            raise ContainerError(
                f"index model stage {stage} must be positive and sum to {TOTAL_FREQUENCY}"
            )
        result.append(np.ascontiguousarray(frequencies))
        offset = end
    if offset != len(raw):  # pragma: no cover - exact expected length and loop agree
        raise ContainerError("index model byte accounting drift")
    return tuple(result)


def _integer_matrix(
    value: object,
    shape: tuple[int, int],
    lower: int,
    upper: int,
    name: str,
) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != shape or array.dtype.kind not in "iu":
        raise ContainerError(f"{name} must be an integer array with shape {shape}")
    if array.size and (int(array.min()) < lower or int(array.max()) > upper):
        raise ContainerError(f"{name} is outside [{lower}, {upper}]")
    return array


def serialize_codebooks(codebooks: Sequence[np.ndarray], descriptor: SSP2VDescriptor) -> bytes:
    descriptor.validated()
    if not isinstance(codebooks, Sequence) or isinstance(codebooks, (bytes, bytearray, str)):
        raise ContainerError("codebooks must be a stage sequence")
    if len(codebooks) != descriptor.actual_stage_count:
        raise ContainerError("codebook stage count disagrees with descriptor")
    raw = bytearray()
    if descriptor.family_id == 0:
        values = _integer_matrix(
            codebooks[0], (descriptor.codebook_entries, 3), 0, 255, "flat codebook"
        )
        raw.extend(np.ascontiguousarray(values, dtype=np.uint8).tobytes(order="C"))
    else:
        stage0 = _integer_matrix(
            codebooks[0], (descriptor.base_k, 3), 0, 255, "RVQ stage-0 codebook"
        )
        raw.extend(np.ascontiguousarray(stage0, dtype=np.uint8).tobytes(order="C"))
        for stage in range(1, descriptor.actual_stage_count):
            residual = _integer_matrix(
                codebooks[stage],
                (descriptor.base_k, 3),
                -(1 << 15),
                (1 << 15) - 1,
                f"RVQ stage-{stage} codebook",
            )
            raw.extend(np.ascontiguousarray(residual, dtype="<i2").tobytes(order="C"))
    if len(raw) != descriptor.codebook_bytes:
        raise ContainerError("raw codebook byte accounting drift")
    return bytes(raw)


def parse_codebooks(payload: bytes, descriptor: SSP2VDescriptor) -> tuple[np.ndarray, ...]:
    descriptor.validated()
    if not isinstance(payload, bytes) or len(payload) != descriptor.codebook_bytes:
        raise ContainerError(
            f"codebook payload length mismatch: {len(payload) if isinstance(payload, bytes) else -1}"
            f" != {descriptor.codebook_bytes}"
        )
    result: list[np.ndarray] = []
    offset = 0
    if descriptor.family_id == 0:
        values = np.frombuffer(payload, dtype=np.uint8).reshape(descriptor.codebook_entries, 3)
        result.append(np.ascontiguousarray(values, dtype=np.uint8))
    else:
        first_end = 3 * descriptor.base_k
        stage0 = np.frombuffer(payload[:first_end], dtype=np.uint8).reshape(descriptor.base_k, 3)
        result.append(np.ascontiguousarray(stage0, dtype=np.uint8))
        offset = first_end
        for _stage in range(1, descriptor.actual_stage_count):
            end = offset + 6 * descriptor.base_k
            residual = np.frombuffer(payload[offset:end], dtype="<i2").reshape(descriptor.base_k, 3)
            result.append(np.ascontiguousarray(residual, dtype=np.int16))
            offset = end
    if serialize_codebooks(result, descriptor) != payload:
        raise ContainerError("codebook payload is not canonical entry-major little-endian data")
    return tuple(result)


def reconstruct_rgb(
    descriptor: SSP2VDescriptor,
    codebooks: Sequence[np.ndarray],
    indices: Sequence[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Return signed-int32 raw accumulation and final clipped ``uint32[N,3]`` RGB."""

    normalized_codebooks = parse_codebooks(serialize_codebooks(codebooks, descriptor), descriptor)
    if len(indices) != descriptor.actual_stage_count:
        raise ContainerError("index stage count disagrees with descriptor")
    normalized_indices = tuple(
        _index_vector(values, alphabet, f"index_{stage}")
        for stage, (values, alphabet) in enumerate(
            zip(indices, descriptor.stage_alphabets, strict=True)
        )
    )
    raw = np.ascontiguousarray(
        normalized_codebooks[0][normalized_indices[0]].astype(np.int32), dtype=np.int32
    )
    if descriptor.family_id == 1:
        for stage in range(1, descriptor.actual_stage_count):
            raw = np.add(
                raw,
                normalized_codebooks[stage][normalized_indices[stage]].astype(np.int32),
                dtype=np.int32,
            )
    clipped = np.ascontiguousarray(np.clip(raw, 0, 255), dtype=np.uint32)
    if descriptor.family_id == 0 and (np.any(raw < 0) or np.any(raw > 255)):
        raise ContainerError("flat VQ reconstruction unexpectedly clipped")
    return raw, clipped


def rgb_accounting(source_rgb: np.ndarray, raw_rgb: np.ndarray) -> RGBAccounting:
    """Compute exact clipped error, clipping, and distinct-input collision definitions."""

    source = np.asarray(source_rgb)
    raw = np.asarray(raw_rgb)
    if source.ndim != 2 or source.shape[1:] != (3,) or source.dtype.kind not in "iu":
        raise ContainerError("source_rgb must be an integer [N,3] array")
    if raw.shape != source.shape or raw.dtype.kind != "i":
        raise ContainerError("raw_rgb must be a signed-integer array matching source_rgb")
    if source.size and (int(source.min()) < 0 or int(source.max()) > 255):
        raise ContainerError("source_rgb must lie in [0,255]")
    if raw.size and (int(raw.min()) < -(1 << 31) or int(raw.max()) >= 1 << 31):
        raise ContainerError("raw_rgb must be exactly signed-int32-representable")
    source_i64 = np.ascontiguousarray(source, dtype=np.int64)
    raw_i64 = np.ascontiguousarray(raw, dtype=np.int64)
    clipped = np.clip(raw_i64, 0, 255)
    error = clipped - source_i64
    low = raw_i64 < 0
    high = raw_i64 > 255

    rgb_sse = int(np.square(error, dtype=np.int64).sum(dtype=np.int64))
    sum_abs_error = int(np.abs(error).sum(dtype=np.int64))
    sum_signed_error = int(error.sum(dtype=np.int64))
    max_abs_error = int(np.abs(error).max()) if error.size else 0
    clip_low_components = int(low.sum())
    clip_high_components = int(high.sum())
    clip_rows = int(np.logical_or(low, high).any(axis=1).sum())

    python_errors = [
        int(min(255, max(0, int(raw_value)))) - int(source_value)
        for source_value, raw_value in zip(source_i64.flat, raw_i64.flat, strict=True)
    ]
    if (
        rgb_sse != sum(value * value for value in python_errors)
        or sum_abs_error != sum(abs(value) for value in python_errors)
        or sum_signed_error != sum(python_errors)
        or max_abs_error != max((abs(value) for value in python_errors), default=0)
    ):
        raise ContainerError("int64 and arbitrary-precision RGB accounting disagree")

    source_tuples = [tuple(int(value) for value in row) for row in source_i64.tolist()]
    reconstructed_tuples = [tuple(int(value) for value in row) for row in clipped.tolist()]
    groups: dict[tuple[int, int, int], set[tuple[int, int, int]]] = {}
    for source_value, reconstructed in zip(source_tuples, reconstructed_tuples, strict=True):
        groups.setdefault(reconstructed, set()).add(source_value)
    group_sizes = [len(values) for values in groups.values()]
    return RGBAccounting(
        rgb_sse=rgb_sse,
        sum_abs_error=sum_abs_error,
        sum_signed_error=sum_signed_error,
        max_abs_error=max_abs_error,
        clip_low_components=clip_low_components,
        clip_high_components=clip_high_components,
        clip_components=clip_low_components + clip_high_components,
        clip_rows=clip_rows,
        distinct_input_rgb=len(set(source_tuples)),
        distinct_reconstructed_rgb=len(set(reconstructed_tuples)),
        merged_input_colors=sum(max(0, size - 1) for size in group_sizes),
        collision_pairs=sum(size * (size - 1) // 2 for size in group_sizes),
    )


def _unpack(raw: bytes, bits: int, count: int, name: str) -> np.ndarray:
    dtype = _packed_dtype(bits)
    if len(raw) != count * dtype.itemsize:
        raise ContainerError(f"{name} packed length mismatch")
    values = np.frombuffer(raw, dtype=dtype, count=count).astype(np.uint32)
    if values.size and int(values.max()) >= 1 << bits:
        raise ContainerError(f"{name} symbol exceeds declared bit depth")
    return values


def _unpack_byte_planar(raw: bytes, bits: int, count: int) -> np.ndarray:
    dtype = _packed_dtype(bits)
    if len(raw) != count * dtype.itemsize:
        raise ContainerError("means planar length mismatch")
    if dtype.itemsize == 1:
        return _unpack(raw, bits, count, "means")
    planes = np.frombuffer(raw, dtype=np.uint8).reshape(dtype.itemsize, count)
    interleaved = np.ascontiguousarray(planes.T)
    values = np.frombuffer(interleaved.tobytes(), dtype=dtype, count=count).astype(np.uint32)
    if values.size and int(values.max()) >= 1 << bits:
        raise ContainerError("means symbol exceeds declared bit depth")
    return values


def _absolute_modular(values: np.ndarray, bits: int) -> np.ndarray:
    result = np.cumsum(values, axis=0, dtype=np.uint64)
    result &= np.uint64((1 << bits) - 1)
    return np.ascontiguousarray(result, dtype=np.uint32)


def _decode_common(parsed: ParsedContainer, *, include_colors: bool) -> dict[str, np.ndarray]:
    header = parsed.header
    means_bytes, scales_bytes, rotation_bytes, colors_bytes = header.raw_stream_lengths()
    means_raw = _strict_zlib(parsed.payloads["means_zlib"], means_bytes, "means")
    scales_raw = _strict_zlib(parsed.payloads["scales_zlib"], scales_bytes, "scales")
    rotation_raw = _strict_zlib(parsed.payloads["rotation_zlib"], rotation_bytes, "rotation")

    means = _unpack_byte_planar(means_raw, header.bits[0], 2 * header.n).reshape(2, header.n).T
    means = _absolute_modular(means, header.bits[0])
    scales = _unpack(scales_raw, header.bits[1], 2 * header.n, "scales").reshape(2, header.n).T
    rotation = _unpack(rotation_raw, header.bits[2], header.n, "rotation")
    result = {
        "mean_x": np.ascontiguousarray(means[:, 0], dtype=np.uint32),
        "mean_y": np.ascontiguousarray(means[:, 1], dtype=np.uint32),
        "scale_x": np.ascontiguousarray(scales[:, 0], dtype=np.uint32),
        "scale_y": np.ascontiguousarray(scales[:, 1], dtype=np.uint32),
        "rotation": np.ascontiguousarray(rotation, dtype=np.uint32),
    }
    if include_colors:
        colors_raw = _strict_zlib(parsed.payloads["colors_zlib"], colors_bytes, "colors")
        colors = _unpack(colors_raw, header.bits[3], 3 * header.n, "colors").reshape(3, header.n).T
        colors = _absolute_modular(colors, header.bits[3])
        result.update(
            {
                "red": np.ascontiguousarray(colors[:, 0], dtype=np.uint32),
                "green": np.ascontiguousarray(colors[:, 1], dtype=np.uint32),
                "blue": np.ascontiguousarray(colors[:, 2], dtype=np.uint32),
            }
        )
    return result


def _ssp2v_names(stage_count: int) -> tuple[str, ...]:
    if stage_count not in (1, 2, 3):
        raise ContainerError("SSP2V stage count must be one, two, or three")
    return SSP2V_FIXED_STREAMS + tuple(f"index_{stage}_arith" for stage in range(stage_count))


def _assemble(magic: bytes, header: SSP2VHeader, payloads: Mapping[str, bytes]) -> bytes:
    header.validated()
    if magic == SSP2Z_MAGIC:
        names = SSP2Z_STREAMS
    elif magic == SSP2V_MAGIC:
        descriptor_payload = payloads.get("descriptor_raw")
        if not isinstance(descriptor_payload, bytes):
            raise ContainerError("SSP2V assembly requires descriptor bytes")
        descriptor = SSP2VDescriptor.from_bytes(descriptor_payload)
        names = _ssp2v_names(descriptor.actual_stage_count)
    else:
        raise ContainerError("unknown COMP-011 magic")
    if set(payloads) != set(names):
        raise ContainerError("payload mapping differs from the frozen stream set")
    ordered: list[bytes] = []
    for name in names:
        payload = payloads[name]
        if not isinstance(payload, bytes) or not payload:
            raise ContainerError(f"{name} payload must be nonempty immutable bytes")
        ordered.append(payload)
    prefix = OUTER.size + HEADER.size + DIRECTORY.size * len(names)
    directory = bytearray()
    offset = prefix
    for stream_id, payload in enumerate(ordered, start=1):
        directory.extend(DIRECTORY.pack(stream_id, offset, len(payload)))
        offset += len(payload)
    body = header.to_bytes() + bytes(directory) + b"".join(ordered)
    total_bytes = OUTER.size + len(body)
    crc = zlib.crc32(body) & 0xFFFFFFFF
    blob = OUTER.pack(magic, VERSION, FLAGS, len(names), total_bytes, crc) + body
    if len(blob) != total_bytes:
        raise ContainerError("complete container byte accounting drift")
    return blob


def parse_container(blob: bytes, *, reference_sspl1: bytes | None = None) -> ParsedContainer:
    """Strictly parse one complete SSP2Z or SSP2V byte string."""

    if not isinstance(blob, bytes):
        raise ContainerError("container input must be immutable bytes")
    if len(blob) < OUTER.size + HEADER.size:
        raise ContainerError("truncated COMP-011 container")
    magic, version, flags, stream_count, total_bytes, stored_crc = OUTER.unpack_from(blob)
    if magic not in ALL_MAGICS:
        raise ContainerError("unrecognized COMP-011 magic")
    if version != VERSION or flags != FLAGS:
        raise ContainerError("unsupported COMP-011 version or flags")
    if magic == SSP2Z_MAGIC:
        if stream_count != len(SSP2Z_STREAMS):
            raise ContainerError("stream count disagrees with SSP2Z magic")
        names = SSP2Z_STREAMS
    else:
        stage_count = int(stream_count) - len(SSP2V_FIXED_STREAMS)
        names = _ssp2v_names(stage_count)
    prefix = OUTER.size + HEADER.size + DIRECTORY.size * len(names)
    if len(blob) < prefix:
        raise ContainerError("truncated COMP-011 directory")
    if total_bytes != len(blob):
        raise ContainerError("outer total length disagrees with physical bytes")
    observed_crc = zlib.crc32(blob[OUTER.size :]) & 0xFFFFFFFF
    if stored_crc != observed_crc:
        raise ContainerError("outer CRC-32 mismatch")
    header = SSP2VHeader.from_bytes(blob[OUTER.size : OUTER.size + HEADER.size])

    entries: list[DirectoryEntry] = []
    payloads: dict[str, bytes] = {}
    expected_offset = prefix
    directory_offset = OUTER.size + HEADER.size
    for index, name in enumerate(names):
        stream_id, offset, length = DIRECTORY.unpack_from(
            blob, directory_offset + index * DIRECTORY.size
        )
        expected_id = index + 1
        if stream_id != expected_id:
            raise ContainerError(f"directory ID/order mismatch for {name}")
        if offset != expected_offset:
            raise ContainerError(f"directory gap, overlap, alias, or offset mismatch for {name}")
        if length == 0:
            raise ContainerError(f"empty {name} payload is forbidden")
        end = int(offset) + int(length)
        if end > len(blob):
            raise ContainerError(f"truncated {name} payload")
        payload = blob[int(offset) : end]
        entries.append(DirectoryEntry(stream_id, name, int(offset), int(length)))
        payloads[name] = payload
        expected_offset = end
    if expected_offset != len(blob):
        raise ContainerError("container has a suffix or unclaimed trailing bytes")

    means_bytes, scales_bytes, rotation_bytes, colors_bytes = header.raw_stream_lengths()
    _strict_zlib(payloads["means_zlib"], means_bytes, "means")
    _strict_zlib(payloads["scales_zlib"], scales_bytes, "scales")
    _strict_zlib(payloads["rotation_zlib"], rotation_bytes, "rotation")
    descriptor: SSP2VDescriptor | None = None
    if magic == SSP2Z_MAGIC:
        _strict_zlib(payloads["colors_zlib"], colors_bytes, "colors")
    else:
        descriptor = SSP2VDescriptor.from_bytes(payloads["descriptor_raw"])
        if descriptor.actual_stage_count != len(names) - len(SSP2V_FIXED_STREAMS):
            raise ContainerError("descriptor stage count disagrees with outer stream count")
        parse_codebooks(payloads["codebook_raw"], descriptor)
        parse_index_model(payloads["index_model_zlib"], descriptor)

    parsed = ParsedContainer(
        magic=magic,
        header=header,
        entries=tuple(entries),
        payloads=payloads,
        total_bytes=len(blob),
        crc32=stored_crc,
        descriptor=descriptor,
    )
    parsed.component_bytes()
    if reference_sspl1 is not None:
        source = extract_sspl1_parts(reference_sspl1)
        if header != source.header:
            raise ContainerError("common header differs from reference SSPL1 semantics")
        required = (
            ("means", "scales", "rotation", "colors")
            if magic == SSP2Z_MAGIC
            else (
                "means",
                "scales",
                "rotation",
            )
        )
        for source_name in required:
            payload_name = f"{source_name}_zlib"
            if payloads[payload_name] != source.payloads[source_name]:
                raise ContainerError(f"{payload_name} is not byte-identical to reference SSPL1")
    return parsed


def encode_ssp2z(sspl1_blob: bytes) -> bytes:
    """Create the exact binary-framing-only control from canonical SSPL1 bytes."""

    source = extract_sspl1_parts(sspl1_blob)
    payloads = {
        f"{name}_zlib": source.payloads[name] for name in ("means", "scales", "rotation", "colors")
    }
    blob = _assemble(SSP2Z_MAGIC, source.header, payloads)
    decode_ssp2z(blob, reference_sspl1=sspl1_blob)
    return blob


def encode_ssp2v(
    sspl1_blob: bytes,
    descriptor: SSP2VDescriptor,
    codebooks: Sequence[np.ndarray],
    indices: Sequence[np.ndarray],
    *,
    arithmetic_encode: ArithmeticEncode = arithmetic.encode,
) -> bytes:
    """Encode one complete VQ variant while requiring Python/caller arithmetic byte parity."""

    descriptor.validated()
    source = extract_sspl1_parts(sspl1_blob)
    if len(indices) != descriptor.actual_stage_count:
        raise ContainerError("index stage count disagrees with descriptor")
    normalized_indices = tuple(
        _index_vector(values, alphabet, f"index_{stage}")
        for stage, (values, alphabet) in enumerate(
            zip(indices, descriptor.stage_alphabets, strict=True)
        )
    )
    model_payload = serialize_index_model(normalized_indices, descriptor)
    frequencies = parse_index_model(model_payload, descriptor)
    payloads: dict[str, bytes] = {
        "means_zlib": source.payloads["means"],
        "scales_zlib": source.payloads["scales"],
        "rotation_zlib": source.payloads["rotation"],
        "descriptor_raw": descriptor.to_bytes(),
        "codebook_raw": serialize_codebooks(codebooks, descriptor),
        "index_model_zlib": model_payload,
    }
    for stage, (values, stage_frequencies) in enumerate(
        zip(normalized_indices, frequencies, strict=True)
    ):
        rows = cdf_rows(stage_frequencies)
        try:
            candidate = arithmetic_encode(values, rows)
        except Exception as exc:
            raise ContainerError(f"arithmetic encoder rejected index_{stage}: {exc}") from exc
        if not isinstance(candidate, bytes) or not candidate:
            raise ContainerError(f"arithmetic encoder returned invalid index_{stage} payload")
        python_payload = arithmetic.encode(values, rows)
        if candidate != python_payload:
            raise ContainerError(f"Python/caller arithmetic byte parity failed for index_{stage}")
        payloads[f"index_{stage}_arith"] = candidate
    blob = _assemble(SSP2V_MAGIC, source.header, payloads)
    decode_ssp2v(blob, reference_sspl1=sspl1_blob)
    return blob


def decode_ssp2z(blob: bytes, *, reference_sspl1: bytes | None = None) -> DecodedContainer:
    """Cold-decode SSP2Z and require exact canonical complete-container re-encoding."""

    parsed = parse_container(blob, reference_sspl1=reference_sspl1)
    if parsed.magic != SSP2Z_MAGIC:
        raise ContainerError("decode_ssp2z requires SSP2Z magic")
    symbols = _decode_common(parsed, include_colors=True)
    ordered = {name: symbols[name] for name in SYMBOL_CHANNELS}
    digest = _symbol_sha256(ordered)
    accounting: RGBAccounting | None = None
    if reference_sspl1 is not None:
        source = extract_sspl1_parts(reference_sspl1)
        if digest != source.symbol_sha256:
            raise ContainerError("SSP2Z symbol hash differs from reference SSPL1")
        for name in SYMBOL_CHANNELS:
            if not np.array_equal(ordered[name], source.symbols[name]):
                raise ContainerError(f"SSP2Z decoded {name} differs from reference SSPL1")
        source_rgb = np.column_stack(
            (source.symbols["red"], source.symbols["green"], source.symbols["blue"])
        )
        accounting = rgb_accounting(source_rgb, source_rgb.astype(np.int32))
    reencoded = _assemble(parsed.magic, parsed.header, parsed.payloads)
    if reencoded != blob:
        raise ContainerError("SSP2Z decode-to-reencode is not byte canonical")
    raw_rgb = np.column_stack((ordered["red"], ordered["green"], ordered["blue"])).astype(np.int32)
    return DecodedContainer(
        parsed=parsed,
        symbols=ordered,
        symbol_sha256=digest,
        canonical_blob_sha256=hashlib.sha256(reencoded).hexdigest(),
        raw_rgb=np.ascontiguousarray(raw_rgb),
        rgb_accounting=accounting,
    )


def decode_ssp2v(
    blob: bytes,
    *,
    arithmetic_decode: ArithmeticDecode = arithmetic.decode,
    arithmetic_encode: ArithmeticEncode = arithmetic.encode,
    reference_sspl1: bytes | None = None,
) -> DecodedContainer:
    """Cold-decode all indices, canonically re-encode, and reconstruct clipped RGB symbols."""

    parsed = parse_container(blob, reference_sspl1=reference_sspl1)
    if parsed.magic != SSP2V_MAGIC or parsed.descriptor is None:
        raise ContainerError("decode_ssp2v requires SSP2V magic and descriptor")
    descriptor = parsed.descriptor
    codebooks = parse_codebooks(parsed.payloads["codebook_raw"], descriptor)
    frequencies = parse_index_model(parsed.payloads["index_model_zlib"], descriptor)
    decoded_indices: list[np.ndarray] = []
    canonical_payloads: dict[str, bytes] = dict(parsed.payloads)
    certificates: list[dict[str, int | str | bool]] = []
    for stage, (alphabet, stage_frequencies) in enumerate(
        zip(descriptor.stage_alphabets, frequencies, strict=True)
    ):
        name = f"index_{stage}_arith"
        rows = cdf_rows(stage_frequencies)
        try:
            decoded = arithmetic_decode(parsed.payloads[name], rows, ROW_COUNT)
        except Exception as exc:
            raise ContainerError(f"arithmetic decoder rejected index_{stage}: {exc}") from exc
        values = _index_vector(decoded, alphabet, f"decoded index_{stage}")
        try:
            canonical = arithmetic_encode(values, rows)
        except Exception as exc:
            raise ContainerError(f"arithmetic encoder rejected index_{stage}: {exc}") from exc
        if not isinstance(canonical, bytes) or canonical != parsed.payloads[name]:
            raise ContainerError(f"noncanonical index_{stage} arithmetic payload")
        python_canonical = arithmetic.encode(values, rows)
        if python_canonical != parsed.payloads[name]:
            raise ContainerError(f"Python/caller arithmetic parity failed for index_{stage}")
        certificates.append(arithmetic.coder_certificate(values, rows, parsed.payloads[name]))
        canonical_payloads[name] = canonical
        decoded_indices.append(values)

    canonical_model = serialize_index_model(decoded_indices, descriptor)
    if canonical_model != parsed.payloads["index_model_zlib"]:
        raise ContainerError("index model does not canonically describe decoded indices")
    canonical_payloads["index_model_zlib"] = canonical_model
    canonical_payloads["descriptor_raw"] = descriptor.to_bytes()
    canonical_payloads["codebook_raw"] = serialize_codebooks(codebooks, descriptor)

    raw_rgb, clipped_rgb = reconstruct_rgb(descriptor, codebooks, decoded_indices)
    common = _decode_common(parsed, include_colors=False)
    common.update(
        {
            "red": np.ascontiguousarray(clipped_rgb[:, 0], dtype=np.uint32),
            "green": np.ascontiguousarray(clipped_rgb[:, 1], dtype=np.uint32),
            "blue": np.ascontiguousarray(clipped_rgb[:, 2], dtype=np.uint32),
        }
    )
    ordered = {name: common[name] for name in SYMBOL_CHANNELS}
    digest = _symbol_sha256(ordered)
    accounting: RGBAccounting | None = None
    if reference_sspl1 is not None:
        source = extract_sspl1_parts(reference_sspl1)
        for name in ("mean_x", "mean_y", "scale_x", "scale_y", "rotation"):
            if not np.array_equal(ordered[name], source.symbols[name]):
                raise ContainerError(f"SSP2V decoded non-RGB {name} differs from reference SSPL1")
        source_rgb = np.column_stack(
            (source.symbols["red"], source.symbols["green"], source.symbols["blue"])
        )
        accounting = rgb_accounting(source_rgb, raw_rgb)
        if descriptor.family_id == 0 and accounting.clip_components != 0:
            raise ContainerError("flat VQ accounting contains clipping")

    reencoded = _assemble(parsed.magic, parsed.header, canonical_payloads)
    if reencoded != blob:
        raise ContainerError("SSP2V decode-to-reencode is not byte canonical")
    index_matrix = np.column_stack(decoded_indices)
    return DecodedContainer(
        parsed=parsed,
        symbols=ordered,
        symbol_sha256=digest,
        canonical_blob_sha256=hashlib.sha256(reencoded).hexdigest(),
        codebooks=codebooks,
        indices=tuple(decoded_indices),
        raw_rgb=np.ascontiguousarray(raw_rgb, dtype=np.int32),
        rgb_accounting=accounting,
        coder_certificates=tuple(certificates),
        used_entries_per_stage=tuple(int(np.unique(values).size) for values in decoded_indices),
        unique_index_tuples=int(np.unique(index_matrix, axis=0).shape[0]),
    )


__all__ = [
    "ALL_MAGICS",
    "AUX_MODEL_ID",
    "ContainerError",
    "DecodedContainer",
    "DirectoryEntry",
    "LEGAL_INDEX_ALPHABETS",
    "ParsedContainer",
    "RGBAccounting",
    "ROW_COUNT",
    "SSP2VDescriptor",
    "SSP2VHeader",
    "SSP2Z_PREFIX_BYTES",
    "TOTAL_FREQUENCY",
    "cdf_rows",
    "decode_ssp2v",
    "decode_ssp2z",
    "empirical_frequencies",
    "encode_ssp2v",
    "encode_ssp2z",
    "extract_sspl1_parts",
    "parse_codebooks",
    "parse_container",
    "parse_index_model",
    "reconstruct_rgb",
    "rgb_accounting",
    "serialize_codebooks",
    "serialize_index_model",
]
