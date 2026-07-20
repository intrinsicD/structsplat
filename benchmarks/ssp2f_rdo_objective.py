"""Deterministic clamped float64 RGB objective for COMP-012 coordinate RDO.

This module owns only the fixed CPU objective

``D(C) = ||clip(A @ C, 0, 1) - reference||_F^2``.

``A`` is frozen as canonical sparse column footprints.  Candidate SSP2 RGB
symbols are cold-dequantized through explicit per-channel bounds: arithmetic
is performed in float64, cast once to float32, then promoted to float64 for
the renderer.  A target reference follows the equally explicit
``uint8 -> float32 / 255 -> float64`` path.

Scalar proposals update only the affected renderer column and deterministically
re-reduce every touched squared-residual block.  This is efficient incremental
scoring, not a claim of bitwise identity with an independently rebuilt render.
Commits are immediate and deterministic.  ``reconcile`` independently rebuilds
every render, residual, block, and scalar cache in frozen column/pixel order,
then either validates and rebases within fixed tolerances or fails closed.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Literal, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]
UInt8Array = NDArray[np.uint8]
ReferenceKind = Literal["source", "target"]

CHANNEL_NAMES = ("red", "green", "blue")
NORMALIZED_ROW_SUM_ATOL = 1e-5
OBJECTIVE_BLOCK_SIZE = 4096
OBJECTIVE_ARITHMETIC_SCHEMA = "float64-squared-residual-block-reduction-v1"
RECONCILIATION_RENDER_ATOL = 1e-12
RECONCILIATION_RENDER_RTOL = 1e-13
# Objective coefficients are one tenth of the frozen 1e-12 search-acceptance
# coefficient; their sum is therefore at most one fifth of its threshold.
RECONCILIATION_OBJECTIVE_ATOL = 1e-13
RECONCILIATION_OBJECTIVE_RTOL = 1e-13

_OPERATOR_DOMAIN = b"structsplat-comp012-normalized-operator-v1\0"
_CONFIG_DOMAIN = b"structsplat-comp012-rgb-objective-config-v1\0"
# Deliberately identical to ``ssp2f_coordinate_rdo.state_digest`` so proposal
# identities cross the objective/search adapter boundary without translation.
_STATE_DOMAIN = b"structsplat.comp012.rgb-state.v1\x00"
_PROPOSAL_DOMAIN = b"structsplat-comp012-rgb-objective-proposal-v1\0"


class SSP2FRDOObjectiveError(ValueError):
    """A static objective input or state transition is invalid."""


class SSP2FRDOReconciliationError(RuntimeError):
    """An incremental objective cache differs materially from a full render."""

    def __init__(self, report: ObjectiveReconciliation) -> None:
        super().__init__(
            "incremental RGB objective failed full recomputation reconciliation: "
            f"render_error={report.render_max_abs_error:.17g} "
            f"(limit={report.render_tolerance:.17g}), "
            f"clamped_error={report.clamped_max_abs_error:.17g} "
            f"(limit={report.clamped_tolerance:.17g}), "
            f"residual_error={report.residual_max_abs_error:.17g} "
            f"(limit={report.residual_tolerance:.17g}), "
            f"block_error={report.objective_blocks_max_abs_error:.17g} "
            f"(limit={report.objective_blocks_tolerance:.17g}), "
            f"objective_error={report.objective_abs_error:.17g} "
            f"(limit={report.objective_tolerance:.17g})"
        )
        self.report = report


def _canonical_json(value: Any) -> bytes:
    try:
        return (
            json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise SSP2FRDOObjectiveError(f"value is not finite canonical JSON: {exc}") from exc


def _readonly(value: np.ndarray, *, dtype: np.dtype[Any] | type[Any]) -> np.ndarray:
    result = np.ascontiguousarray(value, dtype=dtype).copy()
    result.setflags(write=False)
    return result


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise SSP2FRDOObjectiveError(f"{name} must be an integer")
    result = int(value)
    if result <= 0:
        raise SSP2FRDOObjectiveError(f"{name} must be positive")
    return result


def _finite_vector(value: ArrayLike, name: str, length: int) -> FloatArray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (length,) or not np.isfinite(result).all():
        raise SSP2FRDOObjectiveError(f"{name} must be a finite length-{length} vector")
    return np.ascontiguousarray(result)


def _validate_rgb(value: Any, *, rows: int, levels: int, name: str) -> UInt8Array:
    if not isinstance(value, np.ndarray):
        raise SSP2FRDOObjectiveError(f"{name} must be a NumPy array")
    if value.dtype != np.uint8 or value.shape != (rows, 3):
        raise SSP2FRDOObjectiveError(f"{name} must have dtype uint8 and shape [{rows},3]")
    if value.size and int(value.max()) > levels:
        raise SSP2FRDOObjectiveError(f"{name} contains a symbol above the declared level {levels}")
    return np.ascontiguousarray(value).copy()


def _float_tolerance(
    first: float,
    second: float,
    *,
    atol: float,
    rtol: float,
) -> float:
    return float(atol + rtol * max(abs(first), abs(second)))


def _require_finite_nonnegative(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise SSP2FRDOObjectiveError(f"{name} must be finite and nonnegative")
    return result


def _objective_block_sums(squared_residual: FloatArray) -> FloatArray:
    if (
        not isinstance(squared_residual, np.ndarray)
        or squared_residual.dtype != np.float64
        or squared_residual.ndim != 2
        or squared_residual.shape[1] != 3
        or not np.isfinite(squared_residual).all()
        or np.any(squared_residual < 0.0)
    ):
        raise SSP2FRDOObjectiveError(
            "squared residual cache must be finite, nonnegative, and float64 [pixels,3]"
        )
    flattened = squared_residual.reshape(-1)
    blocks = np.empty(
        (flattened.size + OBJECTIVE_BLOCK_SIZE - 1) // OBJECTIVE_BLOCK_SIZE,
        dtype=np.float64,
    )
    for block_index in range(blocks.size):
        start = block_index * OBJECTIVE_BLOCK_SIZE
        stop = min(start + OBJECTIVE_BLOCK_SIZE, flattened.size)
        blocks[block_index] = np.sum(flattened[start:stop], dtype=np.float64)
    if not np.isfinite(blocks).all() or np.any(blocks < 0.0):
        raise SSP2FRDOObjectiveError("objective block reduction is non-finite or negative")
    return blocks


def _array_reconciliation(
    cached: FloatArray,
    full: FloatArray,
    *,
    atol: float,
    rtol: float,
) -> tuple[float, float]:
    if cached.shape != full.shape:
        return math.inf, 0.0
    if not np.isfinite(cached).all() or not np.isfinite(full).all():
        return math.inf, 0.0
    difference = np.abs(cached - full)
    error = float(np.max(difference, initial=0.0))
    scale = float(
        max(
            np.max(np.abs(cached), initial=0.0),
            np.max(np.abs(full), initial=0.0),
        )
    )
    return error, float(atol + rtol * scale)


@dataclass(frozen=True)
class FrozenColumn:
    """One immutable sparse renderer column in ascending pixel order."""

    pixel_indices: NDArray[np.int64]
    weights: FloatArray


class FrozenNormalizedOperator:
    """Canonical immutable nonnegative normalized ``A`` in sparse-column form."""

    def __init__(
        self,
        pixel_count: int,
        columns: Sequence[FrozenColumn | tuple[ArrayLike, ArrayLike]],
    ) -> None:
        pixels = _positive_int(pixel_count, "pixel_count")
        if isinstance(columns, (str, bytes)) or not isinstance(columns, Sequence) or not columns:
            raise SSP2FRDOObjectiveError("columns must be a nonempty sequence")

        frozen_columns: list[FrozenColumn] = []
        row_sums = np.zeros(pixels, dtype=np.float64)
        for column_index, column in enumerate(columns):
            if isinstance(column, FrozenColumn):
                raw_indices = column.pixel_indices
                raw_weights = column.weights
            else:
                if not isinstance(column, Sequence) or len(column) != 2:
                    raise SSP2FRDOObjectiveError(
                        f"column {column_index} must be (pixel_indices, weights)"
                    )
                raw_indices, raw_weights = column
            indices_array = np.asarray(raw_indices)
            if indices_array.ndim != 1 or not np.issubdtype(indices_array.dtype, np.integer):
                raise SSP2FRDOObjectiveError(
                    f"column {column_index} indices must be an integer vector"
                )
            indices = np.ascontiguousarray(indices_array, dtype=np.int64)
            weights = np.asarray(raw_weights, dtype=np.float64)
            if weights.ndim != 1 or weights.shape != indices.shape:
                raise SSP2FRDOObjectiveError(
                    f"column {column_index} indices and weights must be equal-length vectors"
                )
            if not np.isfinite(weights).all() or np.any(weights < 0.0):
                raise SSP2FRDOObjectiveError(
                    f"column {column_index} weights must be finite and nonnegative"
                )
            if np.any(indices < 0) or np.any(indices >= pixels):
                raise SSP2FRDOObjectiveError(
                    f"column {column_index} contains an out-of-range pixel"
                )
            order = np.argsort(indices, kind="stable")
            indices = indices[order]
            weights = np.ascontiguousarray(weights[order])
            if indices.size and np.any(indices[1:] == indices[:-1]):
                raise SSP2FRDOObjectiveError(f"column {column_index} contains duplicate pixels")
            retained = weights != 0.0
            indices = indices[retained]
            weights = weights[retained]
            np.add.at(row_sums, indices, weights)
            frozen_columns.append(
                FrozenColumn(
                    pixel_indices=_readonly(indices, dtype=np.int64),
                    weights=_readonly(weights, dtype=np.float64),
                )
            )

        maximum_sum = float(row_sums.max(initial=0.0))
        if maximum_sum > 1.0 + NORMALIZED_ROW_SUM_ATOL:
            raise SSP2FRDOObjectiveError(
                "normalized operator row sum exceeds one: "
                f"{maximum_sum:.17g} > {1.0 + NORMALIZED_ROW_SUM_ATOL:.17g}"
            )

        self._pixel_count = pixels
        self._columns = tuple(frozen_columns)
        self._row_sums = _readonly(row_sums, dtype=np.float64)
        self._sha256 = self._compute_sha256()

    @classmethod
    def from_dense(cls, matrix: ArrayLike) -> FrozenNormalizedOperator:
        """Freeze a finite nonempty dense ``[pixels, fields]`` matrix."""

        dense = np.asarray(matrix, dtype=np.float64)
        if dense.ndim != 2 or dense.shape[0] == 0 or dense.shape[1] == 0:
            raise SSP2FRDOObjectiveError(
                "normalized weight matrix must be nonempty and two-dimensional"
            )
        if not np.isfinite(dense).all() or np.any(dense < 0.0):
            raise SSP2FRDOObjectiveError("normalized weight matrix must be finite and nonnegative")
        columns: list[tuple[NDArray[np.int64], FloatArray]] = []
        for index in range(int(dense.shape[1])):
            pixels = np.flatnonzero(dense[:, index] != 0.0).astype(np.int64)
            columns.append((pixels, dense[pixels, index]))
        return cls(int(dense.shape[0]), columns)

    @classmethod
    def from_operator(cls, operator: Any) -> FrozenNormalizedOperator:
        """Freeze a footprint operator such as ``RendererLinearMap``.

        The source operator is never retained.  Every public footprint is copied,
        sorted, validated, and hashed, so later mutation of the source object
        cannot change this objective.
        """

        try:
            pixel_count = operator.pixel_count
            column_count = operator.column_count
            footprint = operator.footprint
        except AttributeError as exc:
            raise SSP2FRDOObjectiveError(
                "operator must expose pixel_count, column_count, and footprint(column)"
            ) from exc
        fields = _positive_int(column_count, "operator.column_count")
        columns: list[tuple[ArrayLike, ArrayLike]] = []
        for index in range(fields):
            try:
                current = footprint(index)
            except Exception as exc:
                raise SSP2FRDOObjectiveError(f"operator footprint({index}) failed") from exc
            if hasattr(current, "pixel_indices") and hasattr(current, "weights"):
                columns.append((current.pixel_indices, current.weights))
            elif isinstance(current, Sequence) and len(current) == 2:
                columns.append((current[0], current[1]))
            else:
                raise SSP2FRDOObjectiveError(f"operator footprint({index}) has no indices/weights")
        return cls(_positive_int(pixel_count, "operator.pixel_count"), columns)

    def _compute_sha256(self) -> str:
        digest = hashlib.sha256(_OPERATOR_DOMAIN)
        digest.update(self._pixel_count.to_bytes(8, "little", signed=False))
        digest.update(len(self._columns).to_bytes(8, "little", signed=False))
        for index, column in enumerate(self._columns):
            digest.update(index.to_bytes(8, "little", signed=False))
            digest.update(column.pixel_indices.size.to_bytes(8, "little", signed=False))
            digest.update(column.pixel_indices.astype("<i8", copy=False).tobytes(order="C"))
            digest.update(column.weights.astype("<f8", copy=False).tobytes(order="C"))
        return digest.hexdigest()

    @property
    def pixel_count(self) -> int:
        return self._pixel_count

    @property
    def column_count(self) -> int:
        return len(self._columns)

    @property
    def sha256(self) -> str:
        return self._sha256

    @property
    def row_sums(self) -> FloatArray:
        return self._row_sums.copy()

    def footprint(self, column: int) -> FrozenColumn:
        if isinstance(column, (bool, np.bool_)) or not isinstance(column, (int, np.integer)):
            raise SSP2FRDOObjectiveError("renderer column must be an integer")
        index = int(column)
        if not 0 <= index < self.column_count:
            raise SSP2FRDOObjectiveError("renderer column is out of range")
        current = self._columns[index]
        return FrozenColumn(current.pixel_indices.copy(), current.weights.copy())

    def apply(self, colors: ArrayLike) -> FloatArray:
        """Apply ``A`` in fixed column order with fixed per-column pixel order."""

        values = np.asarray(colors)
        if (
            values.dtype != np.float64
            or values.shape != (self.column_count, 3)
            or not np.isfinite(values).all()
        ):
            raise SSP2FRDOObjectiveError("colors must be a finite float64 [fields,3] array")
        output = np.zeros((self.pixel_count, 3), dtype=np.float64)
        for index, column in enumerate(self._columns):
            if column.pixel_indices.size:
                output[column.pixel_indices] += column.weights[:, None] * values[index][None, :]
        return output

    def add_scaled_scalar_column(
        self,
        image: FloatArray,
        column: int,
        channel: int,
        scalar: float,
    ) -> None:
        """Apply one scalar edit to a writable cached ``A @ C`` image."""

        if (
            not isinstance(image, np.ndarray)
            or image.dtype != np.float64
            or image.shape != (self.pixel_count, 3)
            or not image.flags.writeable
        ):
            raise SSP2FRDOObjectiveError("image must be a writable float64 [pixels,3] array")
        index = int(column)
        component = int(channel)
        if not 0 <= index < self.column_count or not 0 <= component < 3:
            raise SSP2FRDOObjectiveError("renderer edit index is out of range")
        if not math.isfinite(float(scalar)):
            raise SSP2FRDOObjectiveError("renderer edit scalar must be finite")
        current = self._columns[index]
        if current.pixel_indices.size:
            image[current.pixel_indices, component] += current.weights * float(scalar)


def freeze_normalized_operator(value: Any) -> FrozenNormalizedOperator:
    """Copy a dense matrix or public-footprint operator into canonical storage."""

    if isinstance(value, FrozenNormalizedOperator):
        return FrozenNormalizedOperator(
            value.pixel_count,
            [
                (value._columns[index].pixel_indices, value._columns[index].weights)
                for index in range(value.column_count)
            ],
        )
    if isinstance(value, np.ndarray):
        return FrozenNormalizedOperator.from_dense(value)
    return FrozenNormalizedOperator.from_operator(value)


def affine_color_table(
    color_lo: ArrayLike,
    color_hi: ArrayLike,
    *,
    bits_color: int = 8,
) -> FloatArray:
    """Build the exact SSP2 cold-decode lookup table, promoted to float64."""

    bits = _positive_int(bits_color, "bits_color")
    if bits > 8:
        raise SSP2FRDOObjectiveError("uint8 RGB state cannot represent bits_color > 8")
    lower = _finite_vector(color_lo, "color_lo", 3)
    upper = _finite_vector(color_hi, "color_hi", 3)
    if np.any(upper <= lower):
        raise SSP2FRDOObjectiveError("every color_hi must be greater than color_lo")
    levels = (1 << bits) - 1
    symbols = np.arange(levels + 1, dtype=np.float64)[:, None]
    decoded64 = lower[None, :] + symbols / float(levels) * (upper - lower)[None, :]
    return np.ascontiguousarray(decoded64.astype(np.float32).astype(np.float64))


def target_reference(target_rgb: np.ndarray, *, pixel_count: int) -> FloatArray:
    """Apply the frozen strict-RGB ``uint8 -> float32 / 255 -> float64`` path."""

    if not isinstance(target_rgb, np.ndarray) or target_rgb.dtype != np.uint8:
        raise SSP2FRDOObjectiveError("target reference must have dtype uint8")
    if target_rgb.ndim == 3 and target_rgb.shape[2] == 3:
        flattened = target_rgb.reshape(-1, 3)
    elif target_rgb.ndim == 2 and target_rgb.shape[1] == 3:
        flattened = target_rgb
    else:
        raise SSP2FRDOObjectiveError("target reference must have shape [H,W,3] or [pixels,3]")
    if int(flattened.shape[0]) != pixel_count:
        raise SSP2FRDOObjectiveError("target reference pixel count differs from A")
    normalized = flattened.astype(np.float32) / np.float32(255.0)
    return np.ascontiguousarray(normalized.astype(np.float64))


def source_reference(source_image: ArrayLike, *, pixel_count: int) -> FloatArray:
    """Validate an already clamped deterministic source render."""

    image = np.asarray(source_image)
    if image.ndim == 3 and image.shape[2] == 3:
        image = image.reshape(-1, 3)
    if image.shape != (pixel_count, 3):
        raise SSP2FRDOObjectiveError(
            "source reference must have shape [H,W,3] or [pixels,3] matching A"
        )
    if image.dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise SSP2FRDOObjectiveError("source reference must have dtype float32 or float64")
    result = np.ascontiguousarray(image, dtype=np.float64)
    if not np.isfinite(result).all() or np.any(result < 0.0) or np.any(result > 1.0):
        raise SSP2FRDOObjectiveError(
            "source reference must be finite and inside the clamped [0,1] range"
        )
    return result


@dataclass(frozen=True)
class RGBObjectiveProposal:
    """One scalar edit scored against an exact engine epoch and origin state."""

    epoch: int
    config_sha256: str
    origin_state_sha256: str
    candidate_state_sha256: str
    row: int
    channel: int
    old_symbol: int
    new_symbol: int
    old_color: float
    new_color: float
    objective_before: float
    objective: float
    objective_delta: float
    affected_pixels: int
    proposal_sha256: str


@dataclass(frozen=True)
class ObjectiveReconciliation:
    """Independent full-render parity diagnostics for one current state."""

    epoch: int
    state_sha256: str
    commits_since_reconciliation: int
    cached_objective: float
    full_objective: float
    objective_abs_error: float
    objective_tolerance: float
    render_max_abs_error: float
    render_tolerance: float
    clamped_max_abs_error: float
    clamped_tolerance: float
    residual_max_abs_error: float
    residual_tolerance: float
    squared_residual_max_abs_error: float
    squared_residual_tolerance: float
    objective_blocks_max_abs_error: float
    objective_blocks_tolerance: float
    residual_reduced_objective: float
    residual_reduced_objective_abs_error: float
    residual_reduced_objective_tolerance: float
    passed: bool
    rebased: bool


class SSP2FRGBObjective:
    """Stateful deterministic clamped normalized-render SSE engine."""

    @classmethod
    def from_source_symbols(
        cls,
        normalized_weights: Any,
        rgb_symbols: np.ndarray,
        source_symbols: np.ndarray,
        *,
        color_lo: ArrayLike,
        color_hi: ArrayLike,
        bits_color: int = 8,
    ) -> SSP2FRGBObjective:
        """Construct ``D_source`` directly from the immutable source RGB state."""

        operator = freeze_normalized_operator(normalized_weights)
        bits = _positive_int(bits_color, "bits_color")
        if bits > 8:
            raise SSP2FRDOObjectiveError("uint8 RGB state cannot represent bits_color > 8")
        levels = (1 << bits) - 1
        source = _validate_rgb(
            source_symbols,
            rows=operator.column_count,
            levels=levels,
            name="source_symbols",
        )
        table = affine_color_table(color_lo, color_hi, bits_color=bits)
        colors = np.ascontiguousarray(
            table[source, np.arange(3)[None, :]],
            dtype=np.float64,
        )
        reference = np.clip(operator.apply(colors), 0.0, 1.0)
        return cls(
            operator,
            rgb_symbols,
            reference,
            reference_kind="source",
            color_lo=color_lo,
            color_hi=color_hi,
            bits_color=bits,
        )

    def __init__(
        self,
        normalized_weights: Any,
        rgb_symbols: np.ndarray,
        reference_image: ArrayLike,
        *,
        reference_kind: ReferenceKind,
        color_lo: ArrayLike,
        color_hi: ArrayLike,
        bits_color: int = 8,
    ) -> None:
        if reference_kind not in ("source", "target"):
            raise SSP2FRDOObjectiveError("reference_kind must be 'source' or 'target'")
        self._operator = freeze_normalized_operator(normalized_weights)
        bits = _positive_int(bits_color, "bits_color")
        if bits > 8:
            raise SSP2FRDOObjectiveError("uint8 RGB state cannot represent bits_color > 8")
        self._bits_color = bits
        self._levels = (1 << bits) - 1
        self._color_table = _readonly(
            affine_color_table(color_lo, color_hi, bits_color=bits),
            dtype=np.float64,
        )
        self._symbols = _validate_rgb(
            rgb_symbols,
            rows=self._operator.column_count,
            levels=self._levels,
            name="rgb_symbols",
        )
        if reference_kind == "target":
            reference = target_reference(
                np.asarray(reference_image),
                pixel_count=self._operator.pixel_count,
            )
        else:
            reference = source_reference(
                reference_image,
                pixel_count=self._operator.pixel_count,
            )
        self._reference_kind = reference_kind
        self._reference = _readonly(reference, dtype=np.float64)
        self._config_sha256 = self._compute_config_sha256()
        self._epoch = 0
        self._state_sha256 = self._compute_state_sha256(self._symbols)
        (
            self._unclamped,
            self._clamped,
            self._residual,
            self._squared_residual,
            self._objective_blocks,
            self._objective,
        ) = self._full_components(self._symbols)
        self._commits_since_reconciliation = 0

    def _compute_config_sha256(self) -> str:
        digest = hashlib.sha256(_CONFIG_DOMAIN)
        digest.update(self._operator.sha256.encode("ascii"))
        digest.update(self._reference_kind.encode("ascii") + b"\0")
        digest.update(self._bits_color.to_bytes(1, "little", signed=False))
        digest.update(OBJECTIVE_ARITHMETIC_SCHEMA.encode("ascii") + b"\0")
        digest.update(OBJECTIVE_BLOCK_SIZE.to_bytes(8, "little", signed=False))
        for value in (
            NORMALIZED_ROW_SUM_ATOL,
            RECONCILIATION_RENDER_ATOL,
            RECONCILIATION_RENDER_RTOL,
            RECONCILIATION_OBJECTIVE_ATOL,
            RECONCILIATION_OBJECTIVE_RTOL,
        ):
            digest.update(float(value).hex().encode("ascii") + b"\0")
        digest.update(self._color_table.astype("<f8", copy=False).tobytes(order="C"))
        digest.update(self._reference.astype("<f8", copy=False).tobytes(order="C"))
        return digest.hexdigest()

    def _compute_state_sha256(self, symbols: UInt8Array) -> str:
        digest = hashlib.sha256(_STATE_DOMAIN)
        digest.update(int(symbols.shape[0]).to_bytes(8, "big", signed=False))
        digest.update(symbols.tobytes(order="C"))
        return digest.hexdigest()

    def _decoded_colors(self, symbols: UInt8Array) -> FloatArray:
        channels = np.arange(3)[None, :]
        return np.ascontiguousarray(
            self._color_table[symbols, channels],
            dtype=np.float64,
        )

    def _full_components(
        self,
        symbols: UInt8Array,
    ) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray, float]:
        colors = self._decoded_colors(symbols)
        unclamped = self._operator.apply(colors)
        if not np.isfinite(unclamped).all():
            raise SSP2FRDOObjectiveError("unclamped render is non-finite")
        clamped = np.clip(unclamped, 0.0, 1.0)
        residual = clamped - self._reference
        squared_residual = residual * residual
        blocks = _objective_block_sums(squared_residual)
        objective = _require_finite_nonnegative(
            np.sum(blocks, dtype=np.float64),
            "full objective",
        )
        return unclamped, clamped, residual, squared_residual, blocks, objective

    @staticmethod
    def _channel_index(channel: Any) -> int:
        if isinstance(channel, str):
            try:
                return CHANNEL_NAMES.index(channel)
            except ValueError as exc:
                raise SSP2FRDOObjectiveError(
                    "channel must be red, green, blue, or an integer in [0,3)"
                ) from exc
        if isinstance(channel, (bool, np.bool_)) or not isinstance(channel, (int, np.integer)):
            raise SSP2FRDOObjectiveError("channel must be red, green, blue, or an integer in [0,3)")
        result = int(channel)
        if not 0 <= result < 3:
            raise SSP2FRDOObjectiveError("channel integer is out of range")
        return result

    @staticmethod
    def _proposal_digest(record: dict[str, Any]) -> str:
        return hashlib.sha256(_PROPOSAL_DOMAIN + _canonical_json(record)).hexdigest()

    @property
    def epoch(self) -> int:
        return self._epoch

    @property
    def state_sha256(self) -> str:
        return self._state_sha256

    @property
    def config_sha256(self) -> str:
        return self._config_sha256

    @property
    def operator_sha256(self) -> str:
        return self._operator.sha256

    @property
    def objective(self) -> float:
        return self._objective

    @property
    def rgb_symbols(self) -> UInt8Array:
        return self._symbols.copy()

    @property
    def rendered_image(self) -> FloatArray:
        return self._clamped.copy()

    @property
    def reference_image(self) -> FloatArray:
        return self._reference.copy()

    @property
    def commits_since_reconciliation(self) -> int:
        return self._commits_since_reconciliation

    def render(self, rgb_symbols: np.ndarray) -> FloatArray:
        """Return a fresh full clamped render without mutating engine state."""

        symbols = _validate_rgb(
            rgb_symbols,
            rows=self._operator.column_count,
            levels=self._levels,
            name="rgb_symbols",
        )
        return self._full_components(symbols)[1]

    def evaluate(self, rgb_symbols: np.ndarray) -> float:
        """Return a fresh authoritative block-reduced full-render objective."""

        symbols = _validate_rgb(
            rgb_symbols,
            rows=self._operator.column_count,
            levels=self._levels,
            name="rgb_symbols",
        )
        return self._full_components(symbols)[5]

    def __call__(self, rgb_symbols: np.ndarray) -> float:
        return self.evaluate(rgb_symbols)

    def state_digest_for(self, rgb_symbols: np.ndarray) -> str:
        symbols = _validate_rgb(
            rgb_symbols,
            rows=self._operator.column_count,
            levels=self._levels,
            name="rgb_symbols",
        )
        return self._compute_state_sha256(symbols)

    def propose(self, row: int, channel: int | str, new_symbol: int) -> RGBObjectiveProposal:
        if isinstance(row, (bool, np.bool_)) or not isinstance(row, (int, np.integer)):
            raise SSP2FRDOObjectiveError("proposal row must be an integer")
        normalized_row = int(row)
        if not 0 <= normalized_row < self._operator.column_count:
            raise SSP2FRDOObjectiveError("proposal row is out of range")
        normalized_channel = self._channel_index(channel)
        if isinstance(new_symbol, (bool, np.bool_)) or not isinstance(
            new_symbol, (int, np.integer)
        ):
            raise SSP2FRDOObjectiveError("proposal symbol must be an integer")
        normalized_symbol = int(new_symbol)
        if not 0 <= normalized_symbol <= self._levels:
            raise SSP2FRDOObjectiveError("proposal symbol is outside the RGB alphabet")
        old_symbol = int(self._symbols[normalized_row, normalized_channel])
        if old_symbol == normalized_symbol:
            raise SSP2FRDOObjectiveError("proposal must change the selected symbol")

        old_color = float(self._color_table[old_symbol, normalized_channel])
        new_color = float(self._color_table[normalized_symbol, normalized_channel])
        delta = new_color - old_color
        footprint = self._operator._columns[normalized_row]
        indices = footprint.pixel_indices
        _require_finite_nonnegative(self._objective, "cached objective")
        if (
            self._objective_blocks.dtype != np.float64
            or self._objective_blocks.ndim != 1
            or self._objective_blocks.size
            != (self._squared_residual.size + OBJECTIVE_BLOCK_SIZE - 1) // OBJECTIVE_BLOCK_SIZE
            or not np.isfinite(self._objective_blocks).all()
            or np.any(self._objective_blocks < 0.0)
        ):
            raise SSP2FRDOObjectiveError("cached objective blocks must be finite and nonnegative")
        if (
            self._squared_residual.dtype != np.float64
            or self._squared_residual.shape != self._residual.shape
            or not np.isfinite(self._squared_residual).all()
            or np.any(self._squared_residual < 0.0)
        ):
            raise SSP2FRDOObjectiveError("cached squared residuals must be finite and nonnegative")
        candidate_unclamped = (
            self._unclamped[indices, normalized_channel] + footprint.weights * delta
        )
        candidate_residual = (
            np.clip(candidate_unclamped, 0.0, 1.0) - self._reference[indices, normalized_channel]
        )
        candidate_squared = candidate_residual * candidate_residual
        flat_indices = indices * 3 + normalized_channel
        touched_blocks = np.unique(flat_indices // OBJECTIVE_BLOCK_SIZE)
        candidate_blocks = self._objective_blocks.copy()
        flattened_squared = self._squared_residual.reshape(-1)
        for block_index_value in touched_blocks:
            block_index = int(block_index_value)
            start = block_index * OBJECTIVE_BLOCK_SIZE
            stop = min(start + OBJECTIVE_BLOCK_SIZE, flattened_squared.size)
            block_values = flattened_squared[start:stop].copy()
            in_block = flat_indices // OBJECTIVE_BLOCK_SIZE == block_index
            block_values[flat_indices[in_block] - start] = candidate_squared[in_block]
            candidate_blocks[block_index] = np.sum(block_values, dtype=np.float64)
        if not np.isfinite(candidate_blocks).all() or np.any(candidate_blocks < 0.0):
            raise SSP2FRDOObjectiveError("candidate objective blocks are non-finite or negative")
        objective = _require_finite_nonnegative(
            np.sum(candidate_blocks, dtype=np.float64),
            "proposal objective",
        )
        objective_delta = float(objective - self._objective)

        candidate = self._symbols.copy()
        candidate[normalized_row, normalized_channel] = normalized_symbol
        candidate_digest = self._compute_state_sha256(candidate)
        record = {
            "epoch": self._epoch,
            "config_sha256": self._config_sha256,
            "origin_state_sha256": self._state_sha256,
            "candidate_state_sha256": candidate_digest,
            "row": normalized_row,
            "channel": normalized_channel,
            "old_symbol": old_symbol,
            "new_symbol": normalized_symbol,
            "old_color_hex": old_color.hex(),
            "new_color_hex": new_color.hex(),
            "objective_before_hex": self._objective.hex(),
            "objective_hex": objective.hex(),
            "objective_delta_hex": objective_delta.hex(),
            "affected_pixels": int(indices.size),
        }
        return RGBObjectiveProposal(
            epoch=self._epoch,
            config_sha256=self._config_sha256,
            origin_state_sha256=self._state_sha256,
            candidate_state_sha256=candidate_digest,
            row=normalized_row,
            channel=normalized_channel,
            old_symbol=old_symbol,
            new_symbol=normalized_symbol,
            old_color=old_color,
            new_color=new_color,
            objective_before=self._objective,
            objective=objective,
            objective_delta=objective_delta,
            affected_pixels=int(indices.size),
            proposal_sha256=self._proposal_digest(record),
        )

    def commit(self, proposal: RGBObjectiveProposal) -> float:
        """Commit one verified current-epoch proposal and return cached objective."""

        if not isinstance(proposal, RGBObjectiveProposal):
            raise SSP2FRDOObjectiveError("commit requires an RGBObjectiveProposal")
        if (
            proposal.epoch != self._epoch
            or proposal.config_sha256 != self._config_sha256
            or proposal.origin_state_sha256 != self._state_sha256
        ):
            raise SSP2FRDOObjectiveError("stale RGB objective proposal")
        expected = self.propose(proposal.row, proposal.channel, proposal.new_symbol)
        if expected != proposal:
            raise SSP2FRDOObjectiveError("RGB objective proposal payload or digest differs")

        self._operator.add_scaled_scalar_column(
            self._unclamped,
            proposal.row,
            proposal.channel,
            proposal.new_color - proposal.old_color,
        )
        footprint = self._operator._columns[proposal.row]
        indices = footprint.pixel_indices
        self._clamped[indices, proposal.channel] = np.clip(
            self._unclamped[indices, proposal.channel],
            0.0,
            1.0,
        )
        self._residual[indices, proposal.channel] = (
            self._clamped[indices, proposal.channel] - self._reference[indices, proposal.channel]
        )
        flat_indices = indices * 3 + proposal.channel
        flattened_squared = self._squared_residual.reshape(-1)
        flattened_squared[flat_indices] = (
            self._residual[indices, proposal.channel] * self._residual[indices, proposal.channel]
        )
        for block_index_value in np.unique(flat_indices // OBJECTIVE_BLOCK_SIZE):
            block_index = int(block_index_value)
            start = block_index * OBJECTIVE_BLOCK_SIZE
            stop = min(start + OBJECTIVE_BLOCK_SIZE, flattened_squared.size)
            self._objective_blocks[block_index] = np.sum(
                flattened_squared[start:stop],
                dtype=np.float64,
            )
        committed_objective = _require_finite_nonnegative(
            np.sum(self._objective_blocks, dtype=np.float64),
            "committed objective",
        )
        if committed_objective.hex() != proposal.objective.hex():
            raise SSP2FRDOObjectiveError("committed objective blocks differ from verified proposal")
        self._symbols[proposal.row, proposal.channel] = proposal.new_symbol
        self._objective = committed_objective
        self._epoch += 1
        self._state_sha256 = proposal.candidate_state_sha256
        self._commits_since_reconciliation += 1
        return self._objective

    def reconcile(self) -> ObjectiveReconciliation:
        """Full-recompute, validate drift, and rebase all objective caches."""

        expected_state = self._compute_state_sha256(self._symbols)
        if expected_state != self._state_sha256:
            raise SSP2FRDOObjectiveError("RGB objective state digest drift")
        (
            full_unclamped,
            full_clamped,
            full_residual,
            full_squared_residual,
            full_objective_blocks,
            full_objective,
        ) = self._full_components(self._symbols)
        render_error, render_tolerance = _array_reconciliation(
            self._unclamped,
            full_unclamped,
            atol=RECONCILIATION_RENDER_ATOL,
            rtol=RECONCILIATION_RENDER_RTOL,
        )
        clamped_error, clamped_tolerance = _array_reconciliation(
            self._clamped,
            full_clamped,
            atol=RECONCILIATION_RENDER_ATOL,
            rtol=RECONCILIATION_RENDER_RTOL,
        )
        residual_error, residual_tolerance = _array_reconciliation(
            self._residual,
            full_residual,
            atol=RECONCILIATION_RENDER_ATOL,
            rtol=RECONCILIATION_RENDER_RTOL,
        )
        squared_residual_error, squared_residual_tolerance = _array_reconciliation(
            self._squared_residual,
            full_squared_residual,
            atol=RECONCILIATION_OBJECTIVE_ATOL,
            rtol=RECONCILIATION_OBJECTIVE_RTOL,
        )
        objective_blocks_error, objective_blocks_tolerance = _array_reconciliation(
            self._objective_blocks,
            full_objective_blocks,
            atol=RECONCILIATION_OBJECTIVE_ATOL,
            rtol=RECONCILIATION_OBJECTIVE_RTOL,
        )
        try:
            if (
                not isinstance(self._residual, np.ndarray)
                or self._residual.dtype != np.float64
                or self._residual.shape != full_residual.shape
                or not np.isfinite(self._residual).all()
            ):
                raise SSP2FRDOObjectiveError("cached residual is invalid")
            residual_reduced_blocks = _objective_block_sums(self._residual * self._residual)
            residual_reduced_objective = _require_finite_nonnegative(
                np.sum(residual_reduced_blocks, dtype=np.float64),
                "residual-reduced objective",
            )
        except SSP2FRDOObjectiveError:
            residual_reduced_objective = math.nan
        if math.isfinite(residual_reduced_objective):
            residual_reduced_objective_error = abs(residual_reduced_objective - full_objective)
            residual_reduced_objective_tolerance = _float_tolerance(
                residual_reduced_objective,
                full_objective,
                atol=RECONCILIATION_OBJECTIVE_ATOL,
                rtol=RECONCILIATION_OBJECTIVE_RTOL,
            )
        else:
            residual_reduced_objective_error = math.inf
            residual_reduced_objective_tolerance = 0.0
        objective_error = abs(self._objective - full_objective)
        objective_tolerance = _float_tolerance(
            self._objective,
            full_objective,
            atol=RECONCILIATION_OBJECTIVE_ATOL,
            rtol=RECONCILIATION_OBJECTIVE_RTOL,
        )
        objective_valid = math.isfinite(self._objective) and self._objective >= 0.0
        passed = (
            objective_valid
            and render_error <= render_tolerance
            and clamped_error <= clamped_tolerance
            and residual_error <= residual_tolerance
            and squared_residual_error <= squared_residual_tolerance
            and objective_blocks_error <= objective_blocks_tolerance
            and residual_reduced_objective_error <= residual_reduced_objective_tolerance
            and objective_error <= objective_tolerance
        )
        report = ObjectiveReconciliation(
            epoch=self._epoch,
            state_sha256=self._state_sha256,
            commits_since_reconciliation=self._commits_since_reconciliation,
            cached_objective=self._objective,
            full_objective=full_objective,
            objective_abs_error=objective_error,
            objective_tolerance=objective_tolerance,
            render_max_abs_error=render_error,
            render_tolerance=render_tolerance,
            clamped_max_abs_error=clamped_error,
            clamped_tolerance=clamped_tolerance,
            residual_max_abs_error=residual_error,
            residual_tolerance=residual_tolerance,
            squared_residual_max_abs_error=squared_residual_error,
            squared_residual_tolerance=squared_residual_tolerance,
            objective_blocks_max_abs_error=objective_blocks_error,
            objective_blocks_tolerance=objective_blocks_tolerance,
            residual_reduced_objective=residual_reduced_objective,
            residual_reduced_objective_abs_error=residual_reduced_objective_error,
            residual_reduced_objective_tolerance=residual_reduced_objective_tolerance,
            passed=passed,
            rebased=passed,
        )
        if not passed:
            raise SSP2FRDOReconciliationError(report)

        self._unclamped = full_unclamped
        self._clamped = full_clamped
        self._residual = full_residual
        self._squared_residual = full_squared_residual
        self._objective_blocks = full_objective_blocks
        self._objective = full_objective
        self._commits_since_reconciliation = 0
        return report


__all__ = [
    "CHANNEL_NAMES",
    "NORMALIZED_ROW_SUM_ATOL",
    "OBJECTIVE_ARITHMETIC_SCHEMA",
    "OBJECTIVE_BLOCK_SIZE",
    "ObjectiveReconciliation",
    "RECONCILIATION_OBJECTIVE_ATOL",
    "RECONCILIATION_OBJECTIVE_RTOL",
    "RECONCILIATION_RENDER_ATOL",
    "RECONCILIATION_RENDER_RTOL",
    "RGBObjectiveProposal",
    "SSP2FRDOObjectiveError",
    "SSP2FRDOReconciliationError",
    "SSP2FRGBObjective",
    "FrozenColumn",
    "FrozenNormalizedOperator",
    "affine_color_table",
    "freeze_normalized_operator",
    "source_reference",
    "target_reference",
]
