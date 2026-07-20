"""Standalone mechanics proof for renderer-Gram RGB additive quantization.

This module is deliberately benchmark-local.  It proves algebra and optimization mechanics for
an explicit fixed linear renderer ``Y = A C``; it does not import a StructSplat codec lifecycle,
read an experiment payload, define a bitstream, or provide evidence of a quality/rate gain.

The implementation keeps ``A`` as sparse column footprints.  Full-Gram move scores therefore use
the exact identity

``Delta D = 2 delta.T (A.T Z)_i + ||A[:, i]||^2 ||delta||^2``

without materializing ``A.T @ A``.  A separate diagonal-Gram control intentionally drops column
interactions.  The zero-order empirical index-rate delta is computed directly from the two counts
changed by a move.  Fixed-assignment codebooks are solved by matrix-free conjugate gradients on
``S.T A.T A S``.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Literal, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
ObjectiveKind = Literal["source", "target"]
GramKind = Literal["full", "diagonal"]


def _finite_float64(value: ArrayLike, *, name: str, ndim: int | None = None) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    return array


def _assignments(value: ArrayLike, *, rows: int, codes: int) -> IntArray:
    raw = np.asarray(value)
    if raw.ndim != 1 or int(raw.size) != int(rows):
        raise ValueError(f"assignments must be a vector of length {rows}")
    if not np.issubdtype(raw.dtype, np.integer):
        raise ValueError("assignments must have an integer dtype")
    result = raw.astype(np.int64, copy=True)
    if np.any(result < 0) or np.any(result >= int(codes)):
        raise ValueError("assignment is outside the codebook")
    return result


@dataclass(frozen=True)
class ColumnFootprint:
    """One sparse column of a fixed renderer map."""

    pixel_indices: IntArray
    weights: FloatArray


class RendererLinearMap:
    """Float64 ``A``/``A.T`` operator stored as validated sparse column footprints."""

    def __init__(
        self,
        n_pixels: int,
        columns: Sequence[ColumnFootprint | tuple[ArrayLike, ArrayLike]],
    ) -> None:
        if int(n_pixels) <= 0:
            raise ValueError("n_pixels must be positive")
        if not columns:
            raise ValueError("at least one renderer column is required")

        indices: list[IntArray] = []
        weights: list[FloatArray] = []
        for column_index, column in enumerate(columns):
            if isinstance(column, ColumnFootprint):
                raw_indices = np.asarray(column.pixel_indices)
                raw_weights = column.weights
            else:
                if len(column) != 2:
                    raise ValueError("each column footprint must be (pixel_indices, weights)")
                raw_indices, raw_weights = column
                raw_indices = np.asarray(raw_indices)
            if raw_indices.ndim != 1 or not np.issubdtype(raw_indices.dtype, np.integer):
                raise ValueError(f"column {column_index} indices must be an integer vector")
            current_indices = raw_indices.astype(np.int64, copy=True)
            current_weights = _finite_float64(
                raw_weights,
                name=f"column {column_index} weights",
                ndim=1,
            ).copy()
            if current_indices.shape != current_weights.shape:
                raise ValueError(f"column {column_index} indices/weights differ in length")
            if np.any(current_indices < 0) or np.any(current_indices >= int(n_pixels)):
                raise ValueError(f"column {column_index} contains an out-of-range pixel")
            if np.unique(current_indices).size != current_indices.size:
                raise ValueError(f"column {column_index} contains duplicate pixels")
            order = np.argsort(current_indices, kind="stable")
            indices.append(current_indices[order])
            weights.append(current_weights[order])

        self._n_pixels = int(n_pixels)
        self._indices = tuple(indices)
        self._weights = tuple(weights)
        self._column_norm_sq = np.asarray(
            [float(np.dot(column, column)) for column in self._weights],
            dtype=np.float64,
        )

    @classmethod
    def from_dense(cls, matrix: ArrayLike) -> RendererLinearMap:
        """Build footprints from a finite dense ``(pixels, fields)`` matrix."""

        dense = _finite_float64(matrix, name="renderer matrix", ndim=2)
        if dense.shape[0] == 0 or dense.shape[1] == 0:
            raise ValueError("renderer matrix must be non-empty")
        columns: list[ColumnFootprint] = []
        for column_index in range(int(dense.shape[1])):
            pixel_indices = np.flatnonzero(dense[:, column_index] != 0.0).astype(np.int64)
            columns.append(
                ColumnFootprint(
                    pixel_indices=pixel_indices,
                    weights=dense[pixel_indices, column_index].astype(np.float64, copy=True),
                )
            )
        return cls(int(dense.shape[0]), columns)

    @classmethod
    def from_column_footprints(
        cls,
        n_pixels: int,
        columns: Sequence[ColumnFootprint | tuple[ArrayLike, ArrayLike]],
    ) -> RendererLinearMap:
        return cls(n_pixels, columns)

    @property
    def pixel_count(self) -> int:
        return self._n_pixels

    @property
    def column_count(self) -> int:
        return len(self._indices)

    @property
    def column_norm_sq(self) -> FloatArray:
        return self._column_norm_sq.copy()

    def column_squared_norm(self, column: int) -> float:
        """Return ``||A[:, column]||^2`` without copying the full diagonal."""

        self._validate_column(column)
        return float(self._column_norm_sq[column])

    def footprint(self, column: int) -> ColumnFootprint:
        self._validate_column(column)
        return ColumnFootprint(
            self._indices[column].copy(),
            self._weights[column].copy(),
        )

    def _validate_column(self, column: int) -> None:
        if int(column) < 0 or int(column) >= self.column_count:
            raise IndexError("renderer column is out of range")

    def apply(self, field_values: ArrayLike) -> FloatArray:
        """Apply ``A`` to one vector or a row-major channel matrix."""

        values = _finite_float64(field_values, name="field values")
        vector = values.ndim == 1
        if vector:
            values = values[:, None]
        if values.ndim != 2 or int(values.shape[0]) != self.column_count:
            raise ValueError("field values must have one row per renderer column")
        output = np.zeros((self.pixel_count, int(values.shape[1])), dtype=np.float64)
        for column, (pixel_indices, weights) in enumerate(zip(self._indices, self._weights)):
            if pixel_indices.size:
                output[pixel_indices] += weights[:, None] * values[column][None, :]
        return output[:, 0] if vector else output

    def transpose(self, pixel_values: ArrayLike) -> FloatArray:
        """Apply ``A.T`` to one vector or a row-major channel matrix."""

        values = _finite_float64(pixel_values, name="pixel values")
        vector = values.ndim == 1
        if vector:
            values = values[:, None]
        if values.ndim != 2 or int(values.shape[0]) != self.pixel_count:
            raise ValueError("pixel values must have one row per renderer pixel")
        output = np.zeros((self.column_count, int(values.shape[1])), dtype=np.float64)
        for column, (pixel_indices, weights) in enumerate(zip(self._indices, self._weights)):
            if pixel_indices.size:
                output[column] = np.sum(weights[:, None] * values[pixel_indices], axis=0)
        return output[:, 0] if vector else output

    def column_dot(self, column: int, pixel_values: ArrayLike) -> FloatArray:
        """Return one row of ``A.T @ pixel_values`` without a global transpose pass."""

        self._validate_column(column)
        values = _finite_float64(pixel_values, name="pixel values")
        vector = values.ndim == 1
        if vector:
            values = values[:, None]
        if values.ndim != 2 or int(values.shape[0]) != self.pixel_count:
            raise ValueError("pixel values must have one row per renderer pixel")
        pixel_indices = self._indices[column]
        if not pixel_indices.size:
            result = np.zeros((int(values.shape[1]),), dtype=np.float64)
        else:
            result = np.sum(
                self._weights[column][:, None] * values[pixel_indices],
                axis=0,
            )
        return result[0] if vector else result

    def add_scaled_column(
        self,
        pixel_values: FloatArray,
        column: int,
        coefficient: ArrayLike,
    ) -> None:
        """In-place ``pixel_values += A[:, column] * coefficient``."""

        self._validate_column(column)
        if pixel_values.dtype != np.float64 or pixel_values.ndim != 2:
            raise ValueError("pixel_values must be a writable float64 matrix")
        if int(pixel_values.shape[0]) != self.pixel_count:
            raise ValueError("pixel_values must have one row per renderer pixel")
        delta = _finite_float64(coefficient, name="column coefficient", ndim=1)
        if int(delta.size) != int(pixel_values.shape[1]):
            raise ValueError("column coefficient channel count differs from pixel_values")
        pixel_indices = self._indices[column]
        if pixel_indices.size:
            pixel_values[pixel_indices] += self._weights[column][:, None] * delta[None, :]


@dataclass(frozen=True)
class MoveDeltaCheck:
    kind: ObjectiveKind
    predicted_delta: float
    direct_delta: float
    absolute_error: float
    relative_error: float

    def passes(self, *, atol: float = 1e-12, rtol: float = 1e-10) -> bool:
        return self.absolute_error <= float(atol) + float(rtol) * max(
            abs(self.predicted_delta),
            abs(self.direct_delta),
        )


def exact_residual_move_delta(
    renderer: RendererLinearMap,
    residual_pixels: ArrayLike,
    field_index: int,
    color_delta: ArrayLike,
) -> float:
    """Exact full-Gram distortion delta for a residual ``Z`` and one field move."""

    residual = _finite_float64(residual_pixels, name="residual pixels", ndim=2)
    if int(residual.shape[0]) != renderer.pixel_count:
        raise ValueError("residual row count differs from renderer pixels")
    delta = _finite_float64(color_delta, name="color delta", ndim=1)
    if int(delta.size) != int(residual.shape[1]):
        raise ValueError("color delta channel count differs from residual")
    correlation = renderer.column_dot(field_index, residual)
    return float(
        2.0 * np.dot(delta, correlation)
        + renderer.column_squared_norm(field_index) * np.dot(delta, delta)
    )


def _validated_quantizer_inputs(
    renderer: RendererLinearMap,
    codebook: ArrayLike,
    assignments: ArrayLike,
) -> tuple[FloatArray, IntArray, FloatArray]:
    codes = _finite_float64(codebook, name="codebook", ndim=2)
    if codes.shape[0] == 0 or codes.shape[1] == 0:
        raise ValueError("codebook must be non-empty")
    selected = _assignments(
        assignments,
        rows=renderer.column_count,
        codes=int(codes.shape[0]),
    )
    quantized = codes[selected]
    return codes, selected, quantized


def _delta_check(predicted: float, direct: float, kind: ObjectiveKind) -> MoveDeltaCheck:
    absolute = abs(float(predicted) - float(direct))
    scale = max(abs(float(predicted)), abs(float(direct)), 1e-30)
    return MoveDeltaCheck(kind, float(predicted), float(direct), absolute, absolute / scale)


def check_source_move_delta(
    renderer: RendererLinearMap,
    source_colors: ArrayLike,
    codebook: ArrayLike,
    assignments: ArrayLike,
    field_index: int,
    new_code: int,
) -> MoveDeltaCheck:
    """Compare the exact formula with direct ``||A(quantized-source)||^2`` recomputation."""

    codes, selected, quantized = _validated_quantizer_inputs(renderer, codebook, assignments)
    source = _finite_float64(source_colors, name="source colors", ndim=2)
    if source.shape != quantized.shape:
        raise ValueError("source colors must match the quantized field shape")
    if int(new_code) < 0 or int(new_code) >= int(codes.shape[0]):
        raise IndexError("new_code is outside the codebook")
    residual = renderer.apply(quantized - source)
    delta = codes[int(new_code)] - quantized[int(field_index)]
    predicted = exact_residual_move_delta(renderer, residual, field_index, delta)
    before = float(np.sum(residual * residual))
    after_colors = quantized.copy()
    after_colors[int(field_index)] = codes[int(new_code)]
    after_residual = renderer.apply(after_colors - source)
    direct = float(np.sum(after_residual * after_residual)) - before
    return _delta_check(predicted, direct, "source")


def check_target_move_delta(
    renderer: RendererLinearMap,
    target_pixels: ArrayLike,
    codebook: ArrayLike,
    assignments: ArrayLike,
    field_index: int,
    new_code: int,
) -> MoveDeltaCheck:
    """Compare the exact formula with direct ``||A quantized-target||^2`` recomputation."""

    codes, selected, quantized = _validated_quantizer_inputs(renderer, codebook, assignments)
    target = _finite_float64(target_pixels, name="target pixels", ndim=2)
    if int(target.shape[0]) != renderer.pixel_count or int(target.shape[1]) != int(codes.shape[1]):
        raise ValueError("target pixels must match renderer rows and codebook channels")
    if int(new_code) < 0 or int(new_code) >= int(codes.shape[0]):
        raise IndexError("new_code is outside the codebook")
    residual = renderer.apply(quantized) - target
    delta = codes[int(new_code)] - quantized[int(field_index)]
    predicted = exact_residual_move_delta(renderer, residual, field_index, delta)
    before = float(np.sum(residual * residual))
    after_colors = quantized.copy()
    after_colors[int(field_index)] = codes[int(new_code)]
    after_residual = renderer.apply(after_colors) - target
    direct = float(np.sum(after_residual * after_residual)) - before
    return _delta_check(predicted, direct, "target")


def _count_log_count(count: int) -> float:
    return 0.0 if int(count) == 0 else float(count) * math.log2(float(count))


def empirical_index_rate_bits(
    counts: ArrayLike,
    *,
    active_code_penalty_bits: float = 0.0,
) -> float:
    """Zero-order plug-in index entropy plus an optional exact active-code penalty."""

    raw = np.asarray(counts)
    if raw.ndim != 1 or not np.issubdtype(raw.dtype, np.integer):
        raise ValueError("counts must be an integer vector")
    values = raw.astype(np.int64, copy=False)
    if np.any(values < 0):
        raise ValueError("counts must be nonnegative")
    if float(active_code_penalty_bits) < 0.0 or not math.isfinite(active_code_penalty_bits):
        raise ValueError("active_code_penalty_bits must be finite and nonnegative")
    total = int(values.sum())
    if total == 0:
        entropy_bits = 0.0
    else:
        entropy_bits = _count_log_count(total) - math.fsum(
            _count_log_count(int(count)) for count in values
        )
    return float(entropy_bits + active_code_penalty_bits * int(np.count_nonzero(values)))


def empirical_index_rate_delta_bits(
    counts: ArrayLike,
    old_code: int,
    new_code: int,
    *,
    active_code_penalty_bits: float = 0.0,
) -> float:
    """O(1) exact count update for one zero-order empirical-entropy move."""

    raw = np.asarray(counts)
    if raw.ndim != 1 or not np.issubdtype(raw.dtype, np.integer):
        raise ValueError("counts must be an integer vector")
    values = raw.astype(np.int64, copy=False)
    if np.any(values < 0):
        raise ValueError("counts must be nonnegative")
    if not (0 <= int(old_code) < int(values.size)) or not (0 <= int(new_code) < int(values.size)):
        raise IndexError("rate-delta code is out of range")
    if int(old_code) == int(new_code):
        return 0.0
    old_count = int(values[int(old_code)])
    new_count = int(values[int(new_code)])
    if old_count <= 0:
        raise ValueError("old_code has no assigned item")
    if float(active_code_penalty_bits) < 0.0 or not math.isfinite(active_code_penalty_bits):
        raise ValueError("active_code_penalty_bits must be finite and nonnegative")

    entropy_delta = (
        _count_log_count(old_count)
        + _count_log_count(new_count)
        - _count_log_count(old_count - 1)
        - _count_log_count(new_count + 1)
    )
    active_before = int(old_count > 0) + int(new_count > 0)
    active_after = int(old_count - 1 > 0) + int(new_count + 1 > 0)
    return float(entropy_delta + float(active_code_penalty_bits) * (active_after - active_before))


@dataclass(frozen=True)
class ClampDiagnostics:
    lower_count: int
    upper_count: int
    active_count: int
    total_count: int
    active_fraction: float
    max_underflow: float
    max_overflow: float
    squared_clip_change: float
    warning: bool


def clamp_diagnostics(
    rendered: ArrayLike,
    *,
    lower: float = 0.0,
    upper: float = 1.0,
) -> ClampDiagnostics:
    """Account for values where a production clamp invalidates global linearity."""

    values = _finite_float64(rendered, name="rendered values")
    if values.size == 0:
        raise ValueError("rendered values must be non-empty")
    if not math.isfinite(lower) or not math.isfinite(upper) or not lower < upper:
        raise ValueError("clamp bounds must be finite and strictly ordered")
    below = values < float(lower)
    above = values > float(upper)
    active = below | above
    clipped = np.clip(values, float(lower), float(upper))
    return ClampDiagnostics(
        lower_count=int(np.count_nonzero(below)),
        upper_count=int(np.count_nonzero(above)),
        active_count=int(np.count_nonzero(active)),
        total_count=int(values.size),
        active_fraction=float(np.count_nonzero(active) / values.size),
        max_underflow=float(np.max(float(lower) - values[below])) if np.any(below) else 0.0,
        max_overflow=float(np.max(values[above] - float(upper))) if np.any(above) else 0.0,
        squared_clip_change=float(np.sum((clipped - values) ** 2)),
        warning=bool(np.any(active)),
    )


@dataclass(frozen=True)
class MoveProposal:
    field_index: int | None
    old_code: int | None
    new_code: int | None
    distortion_delta: float
    rate_delta_bits: float
    objective_delta: float
    candidate_count: int


@dataclass(frozen=True)
class ICMResult:
    assignments: IntArray
    quantized_colors: FloatArray
    objective_kind: ObjectiveKind
    gram_kind: GramKind
    optimized_distortion: float
    actual_full_distortion: float
    rate_bits: float
    objective: float
    objective_history: tuple[float, ...]
    distortion_history: tuple[float, ...]
    rate_history_bits: tuple[float, ...]
    accepted_move_deltas: tuple[float, ...]
    moves_per_sweep: tuple[int, ...]
    terminal: bool
    terminal_certificate: MoveProposal
    clamp: ClampDiagnostics


class _PreparedObjective:
    def __init__(
        self,
        renderer: RendererLinearMap,
        quantized: FloatArray,
        *,
        objective_kind: ObjectiveKind,
        gram_kind: GramKind,
        reference_colors: FloatArray | None,
        target_pixels: FloatArray | None,
    ) -> None:
        self.renderer = renderer
        self.kind = objective_kind
        self.gram = gram_kind
        self.reference = reference_colors
        self.target = target_pixels
        self.diagonal = renderer.column_norm_sq
        self.linear: FloatArray | None = None
        self.base = 0.0
        self.residual: FloatArray | None = None

        if gram_kind == "full":
            if objective_kind == "source":
                assert reference_colors is not None
                self.residual = renderer.apply(quantized - reference_colors)
            else:
                assert target_pixels is not None
                self.residual = renderer.apply(quantized) - target_pixels
        elif objective_kind == "target":
            assert reference_colors is not None and target_pixels is not None
            base_residual = renderer.apply(reference_colors) - target_pixels
            self.base = float(np.sum(base_residual * base_residual))
            self.linear = renderer.transpose(base_residual)

    def distortion(self, quantized: FloatArray) -> float:
        if self.gram == "full":
            assert self.residual is not None
            return float(np.sum(self.residual * self.residual))
        assert self.reference is not None
        error = quantized - self.reference
        value = self.base + float(np.sum(self.diagonal[:, None] * error * error))
        if self.kind == "target":
            assert self.linear is not None
            value += 2.0 * float(np.sum(self.linear * error))
        return float(value)

    def actual_full_distortion(self, quantized: FloatArray) -> float:
        if self.kind == "source":
            assert self.reference is not None
            residual = self.renderer.apply(quantized - self.reference)
        else:
            assert self.target is not None
            residual = self.renderer.apply(quantized) - self.target
        return float(np.sum(residual * residual))

    def move_delta(self, field_index: int, old_color: FloatArray, new_color: FloatArray) -> float:
        delta = new_color - old_color
        if self.gram == "full":
            assert self.residual is not None
            return exact_residual_move_delta(
                self.renderer,
                self.residual,
                field_index,
                delta,
            )
        assert self.reference is not None
        reference = self.reference[field_index]
        value = self.diagonal[field_index] * (
            float(np.dot(new_color - reference, new_color - reference))
            - float(np.dot(old_color - reference, old_color - reference))
        )
        if self.kind == "target":
            assert self.linear is not None
            value += 2.0 * float(np.dot(self.linear[field_index], delta))
        return float(value)

    def apply_move(self, field_index: int, delta: FloatArray) -> None:
        if self.gram == "full":
            assert self.residual is not None
            self.renderer.add_scaled_column(self.residual, field_index, delta)


def _prepare_objective(
    renderer: RendererLinearMap,
    quantized: FloatArray,
    *,
    objective_kind: ObjectiveKind,
    gram_kind: GramKind,
    reference_colors: ArrayLike | None,
    target_pixels: ArrayLike | None,
) -> _PreparedObjective:
    if objective_kind not in ("source", "target"):
        raise ValueError("objective_kind must be 'source' or 'target'")
    if gram_kind not in ("full", "diagonal"):
        raise ValueError("gram_kind must be 'full' or 'diagonal'")
    channels = int(quantized.shape[1])
    reference = None
    if reference_colors is not None:
        reference = _finite_float64(reference_colors, name="reference colors", ndim=2)
        if reference.shape != quantized.shape:
            raise ValueError("reference colors must match quantized colors")
    target = None
    if target_pixels is not None:
        target = _finite_float64(target_pixels, name="target pixels", ndim=2)
        if target.shape != (renderer.pixel_count, channels):
            raise ValueError("target pixels must match renderer pixels and color channels")
    if objective_kind == "source" and reference is None:
        raise ValueError("source objective requires reference_colors")
    if objective_kind == "target" and target is None:
        raise ValueError("target objective requires target_pixels")
    if gram_kind == "diagonal" and reference is None:
        raise ValueError("diagonal-Gram control requires reference_colors")
    return _PreparedObjective(
        renderer,
        quantized,
        objective_kind=objective_kind,
        gram_kind=gram_kind,
        reference_colors=reference,
        target_pixels=target,
    )


def _best_global_move(
    objective: _PreparedObjective,
    codebook: FloatArray,
    assignments: IntArray,
    counts: IntArray,
    *,
    rate_weight: float,
    active_code_penalty_bits: float,
) -> MoveProposal:
    best: tuple[float, int, int, float, float] | None = None
    candidate_count = 0
    for field_index, old_code_raw in enumerate(assignments):
        old_code = int(old_code_raw)
        old_color = codebook[old_code]
        for new_code in range(int(codebook.shape[0])):
            if new_code == old_code:
                continue
            candidate_count += 1
            distortion_delta = objective.move_delta(
                field_index,
                old_color,
                codebook[new_code],
            )
            rate_delta = empirical_index_rate_delta_bits(
                counts,
                old_code,
                new_code,
                active_code_penalty_bits=active_code_penalty_bits,
            )
            total_delta = distortion_delta + rate_weight * rate_delta
            candidate = (
                float(total_delta),
                int(field_index),
                int(new_code),
                float(distortion_delta),
                float(rate_delta),
            )
            if best is None or candidate[:3] < best[:3]:
                best = candidate
    if best is None:
        return MoveProposal(None, None, None, 0.0, 0.0, 0.0, candidate_count)
    total_delta, field_index, new_code, distortion_delta, rate_delta = best
    return MoveProposal(
        field_index=field_index,
        old_code=int(assignments[field_index]),
        new_code=new_code,
        distortion_delta=distortion_delta,
        rate_delta_bits=rate_delta,
        objective_delta=total_delta,
        candidate_count=candidate_count,
    )


def run_deterministic_icm(
    renderer: RendererLinearMap,
    codebook: ArrayLike,
    assignments: ArrayLike,
    *,
    objective_kind: ObjectiveKind,
    gram_kind: GramKind = "full",
    reference_colors: ArrayLike | None = None,
    target_pixels: ArrayLike | None = None,
    rate_weight: float = 0.0,
    active_code_penalty_bits: float = 0.0,
    max_sweeps: int = 100,
    improvement_tolerance: float = 1e-12,
    clamp_bounds: tuple[float, float] = (0.0, 1.0),
) -> ICMResult:
    """Run deterministic sequential ICM and return a no-improving-move certificate.

    Candidate codes and fields are visited in ascending order.  A move is accepted only when its
    complete distortion-plus-rate delta is below ``-improvement_tolerance``.  The codebook is
    fixed; use :func:`solve_fixed_assignment_codebook` for the alternating codebook step.
    """

    codes, selected, quantized = _validated_quantizer_inputs(renderer, codebook, assignments)
    if not math.isfinite(rate_weight) or float(rate_weight) < 0.0:
        raise ValueError("rate_weight must be finite and nonnegative")
    if int(max_sweeps) <= 0:
        raise ValueError("max_sweeps must be positive")
    if not math.isfinite(improvement_tolerance) or float(improvement_tolerance) < 0.0:
        raise ValueError("improvement_tolerance must be finite and nonnegative")
    counts = np.bincount(selected, minlength=int(codes.shape[0])).astype(np.int64)
    prepared = _prepare_objective(
        renderer,
        quantized,
        objective_kind=objective_kind,
        gram_kind=gram_kind,
        reference_colors=reference_colors,
        target_pixels=target_pixels,
    )

    rate = empirical_index_rate_bits(
        counts,
        active_code_penalty_bits=active_code_penalty_bits,
    )
    distortion = prepared.distortion(quantized)
    objective_value = distortion + float(rate_weight) * rate
    objective_history = [float(objective_value)]
    distortion_history = [float(distortion)]
    rate_history = [float(rate)]
    accepted_deltas: list[float] = []
    moves_per_sweep: list[int] = []

    for _sweep in range(int(max_sweeps)):
        moves = 0
        for field_index in range(renderer.column_count):
            old_code = int(selected[field_index])
            old_color = codes[old_code]
            best_code = old_code
            best_total_delta = 0.0
            for new_code in range(int(codes.shape[0])):
                if new_code == old_code:
                    continue
                distortion_delta = prepared.move_delta(
                    field_index,
                    old_color,
                    codes[new_code],
                )
                rate_delta = empirical_index_rate_delta_bits(
                    counts,
                    old_code,
                    new_code,
                    active_code_penalty_bits=active_code_penalty_bits,
                )
                total_delta = distortion_delta + float(rate_weight) * rate_delta
                if total_delta < best_total_delta or (
                    total_delta == best_total_delta and new_code < best_code
                ):
                    best_code = new_code
                    best_total_delta = float(total_delta)
            if best_code == old_code or best_total_delta >= -float(improvement_tolerance):
                continue

            delta = codes[best_code] - old_color
            prepared.apply_move(field_index, delta)
            selected[field_index] = best_code
            quantized[field_index] = codes[best_code]
            counts[old_code] -= 1
            counts[best_code] += 1
            accepted_deltas.append(best_total_delta)
            moves += 1

        moves_per_sweep.append(moves)
        rate = empirical_index_rate_bits(
            counts,
            active_code_penalty_bits=active_code_penalty_bits,
        )
        distortion = prepared.distortion(quantized)
        objective_value = distortion + float(rate_weight) * rate
        scale = max(1.0, abs(objective_history[-1]), abs(objective_value))
        if objective_value > objective_history[-1] + 5e-13 * scale:
            raise RuntimeError("ICM sweep increased the exact optimized objective")
        objective_history.append(float(objective_value))
        distortion_history.append(float(distortion))
        rate_history.append(float(rate))
        if moves == 0:
            break

    certificate = _best_global_move(
        prepared,
        codes,
        selected,
        counts,
        rate_weight=float(rate_weight),
        active_code_penalty_bits=float(active_code_penalty_bits),
    )
    terminal = certificate.objective_delta >= -float(improvement_tolerance)
    rendered = renderer.apply(quantized)
    return ICMResult(
        assignments=selected.copy(),
        quantized_colors=quantized.copy(),
        objective_kind=objective_kind,
        gram_kind=gram_kind,
        optimized_distortion=float(distortion_history[-1]),
        actual_full_distortion=prepared.actual_full_distortion(quantized),
        rate_bits=float(rate_history[-1]),
        objective=float(objective_history[-1]),
        objective_history=tuple(objective_history),
        distortion_history=tuple(distortion_history),
        rate_history_bits=tuple(rate_history),
        accepted_move_deltas=tuple(accepted_deltas),
        moves_per_sweep=tuple(moves_per_sweep),
        terminal=bool(terminal),
        terminal_certificate=certificate,
        clamp=clamp_diagnostics(
            rendered,
            lower=float(clamp_bounds[0]),
            upper=float(clamp_bounds[1]),
        ),
    )


@dataclass(frozen=True)
class CodebookSolveResult:
    codebook: FloatArray
    rendered: FloatArray
    target_pixels: FloatArray
    squared_error: float
    converged: bool
    channel_iterations: tuple[int, ...]
    channel_normal_residuals: tuple[float, ...]
    normal_apply_calls: int


def _group_sum(values: FloatArray, assignments: IntArray, n_codes: int) -> FloatArray:
    result = np.zeros((int(n_codes),), dtype=np.float64)
    np.add.at(result, assignments, values)
    return result


def _conjugate_gradient_psd(
    operator: Callable[[FloatArray], FloatArray],
    rhs: FloatArray,
    initial: FloatArray,
    *,
    rtol: float,
    atol: float,
    max_iterations: int,
) -> tuple[FloatArray, int, float, bool, int]:
    value = initial.astype(np.float64, copy=True)
    calls = 0

    def apply(vector: FloatArray) -> FloatArray:
        nonlocal calls
        calls += 1
        result = _finite_float64(operator(vector), name="normal-operator result", ndim=1)
        if result.shape != vector.shape:
            raise ValueError("normal operator changed vector shape")
        return result

    residual = rhs - apply(value)
    rhs_norm = float(np.linalg.norm(rhs))
    tolerance = max(float(atol), float(rtol) * rhs_norm)
    residual_norm = float(np.linalg.norm(residual))
    if residual_norm <= tolerance:
        return value, 0, residual_norm, True, calls
    direction = residual.copy()
    residual_sq = float(np.dot(residual, residual))
    iterations = 0
    for iterations in range(1, int(max_iterations) + 1):
        normal_direction = apply(direction)
        curvature = float(np.dot(direction, normal_direction))
        curvature_floor = (
            np.finfo(np.float64).eps
            * float(np.linalg.norm(direction))
            * float(np.linalg.norm(normal_direction))
        )
        if curvature <= curvature_floor:
            break
        step = residual_sq / curvature
        value += step * direction
        residual -= step * normal_direction
        next_residual_sq = float(np.dot(residual, residual))
        residual_norm = math.sqrt(max(0.0, next_residual_sq))
        if residual_norm <= tolerance:
            return value, iterations, residual_norm, True, calls
        direction = residual + (next_residual_sq / residual_sq) * direction
        residual_sq = next_residual_sq
    residual = rhs - apply(value)
    residual_norm = float(np.linalg.norm(residual))
    return value, iterations, residual_norm, residual_norm <= tolerance, calls


def solve_fixed_assignment_codebook(
    renderer: RendererLinearMap,
    assignments: ArrayLike,
    n_codes: int,
    *,
    target_pixels: ArrayLike | None = None,
    source_colors: ArrayLike | None = None,
    initial_codebook: ArrayLike | None = None,
    rtol: float = 1e-12,
    atol: float = 1e-14,
    max_iterations: int | None = None,
) -> CodebookSolveResult:
    """Solve fixed-assignment codebooks via matrix-free ``S.T A.T A S`` products.

    Exactly one of ``target_pixels`` or ``source_colors`` is required.  The source-preserving case
    uses ``A @ source_colors`` as its pixel target.  No renderer Gram matrix or ``A @ S`` design is
    materialized.
    """

    if int(n_codes) <= 0:
        raise ValueError("n_codes must be positive")
    if (target_pixels is None) == (source_colors is None):
        raise ValueError("provide exactly one of target_pixels or source_colors")
    selected = _assignments(assignments, rows=renderer.column_count, codes=int(n_codes))
    if not math.isfinite(rtol) or not math.isfinite(atol) or rtol < 0.0 or atol < 0.0:
        raise ValueError("CG tolerances must be finite and nonnegative")
    if max_iterations is None:
        max_iterations = max(32, 4 * int(n_codes))
    if int(max_iterations) <= 0:
        raise ValueError("max_iterations must be positive")

    if source_colors is not None:
        source = _finite_float64(source_colors, name="source colors", ndim=2)
        if int(source.shape[0]) != renderer.column_count or int(source.shape[1]) == 0:
            raise ValueError("source colors must have one non-empty row per renderer column")
        target = renderer.apply(source)
    else:
        target = _finite_float64(target_pixels, name="target pixels", ndim=2)
        if int(target.shape[0]) != renderer.pixel_count or int(target.shape[1]) == 0:
            raise ValueError("target pixels must have one non-empty row per renderer pixel")
    channels = int(target.shape[1])
    if initial_codebook is None:
        initial = np.zeros((int(n_codes), channels), dtype=np.float64)
    else:
        initial = _finite_float64(initial_codebook, name="initial codebook", ndim=2)
        if initial.shape != (int(n_codes), channels):
            raise ValueError("initial codebook shape differs from n_codes/target channels")

    target_transpose = renderer.transpose(target)
    solved = np.zeros_like(initial)
    iterations: list[int] = []
    residuals: list[float] = []
    converged: list[bool] = []
    apply_calls = 0

    def normal_apply(code_values: FloatArray) -> FloatArray:
        field_values = code_values[selected]
        rendered_values = renderer.apply(field_values)
        field_gradient = renderer.transpose(rendered_values)
        return _group_sum(field_gradient, selected, int(n_codes))

    for channel in range(channels):
        rhs = _group_sum(target_transpose[:, channel], selected, int(n_codes))
        solution, used, residual, channel_converged, calls = _conjugate_gradient_psd(
            normal_apply,
            rhs,
            initial[:, channel],
            rtol=float(rtol),
            atol=float(atol),
            max_iterations=int(max_iterations),
        )
        solved[:, channel] = solution
        iterations.append(int(used))
        residuals.append(float(residual))
        converged.append(bool(channel_converged))
        apply_calls += int(calls)

    rendered = renderer.apply(solved[selected])
    residual = rendered - target
    return CodebookSolveResult(
        codebook=solved,
        rendered=rendered,
        target_pixels=target.copy(),
        squared_error=float(np.sum(residual * residual)),
        converged=all(converged),
        channel_iterations=tuple(iterations),
        channel_normal_residuals=tuple(residuals),
        normal_apply_calls=apply_calls,
    )
