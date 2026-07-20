"""Exact deterministic model fitting for the frozen COMP-009 SSP2E head.

This module deliberately knows nothing about the arithmetic-coder implementation or development
artifacts.  The only bridge to the coder is an injected CDF-frequency callable plus the frozen
integer ``cost_by_frequency`` table; model selection similarly receives an injected payload-length
callback.  That separation keeps every pre-data fitting proof synthetic and replayable.
"""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass
from functools import lru_cache
from typing import Mapping, Protocol, Sequence

import numpy as np


CHANNELS = ("scale_x", "scale_y", "red", "green", "blue")
CHANNEL_COUNT = len(CHANNELS)
CELL_COUNT = 256
FEATURE_COUNT = 8
LOCATION_COUNT = 64
SCALE_COUNT = 16
ARITHMETIC_TOTAL = 32768
GAUSSIAN_COUNT = 8192
MAX_OUTER_SWEEPS = 64

HEAD_RECORD = struct.Struct("<8bh8bh")
HEAD_BYTES = CHANNEL_COUNT * HEAD_RECORD.size
GRID_BYTES = CELL_COUNT

GRID_START_DOMAIN = b"structsplat-comp009-grid-start-v1\0"
SHUFFLE_DOMAIN = b"structsplat-comp009-shuffle-v1\0"
SHUFFLE_PERMUTATION_SHA256 = "63691f3fdda3cbf83d1bc7b954785dc5467b11433bc4fbbe84fd5ec2d94c05f2"


class ModelError(ValueError):
    """A model, objective, or deterministic fitting input is invalid."""


class ConvergenceError(RuntimeError):
    """A start did not reach a fixed point inside the frozen sweep cap."""

    def __init__(
        self,
        start_id: int,
        trajectory: tuple[SweepRecord, ...],
        terminal_model: SSP2EModel,
    ) -> None:
        super().__init__(
            f"grid start {start_id} did not reach a fixed point within "
            f"{len(trajectory) - 1} outer sweeps"
        )
        self.start_id = start_id
        self.trajectory = trajectory
        self.terminal_model = terminal_model
        self.terminal_objective_q24 = trajectory[-1].objective_q24


class CDFFrequencies(Protocol):
    """Minimal arithmetic-table interface consumed by the fitter."""

    def __call__(self, channel: str, location: int, scale: int) -> Sequence[int]: ...


class EncodeLengths(Protocol):
    """Return the five arithmetic payload byte lengths for a complete fitted model."""

    def __call__(self, model: SSP2EModel) -> Sequence[int] | Mapping[str, int]: ...


def _bounded_integer(value: object, name: str, lower: int, upper: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ModelError(f"{name} must be an integer")
    result = int(value)
    if not lower <= result <= upper:
        raise ModelError(f"{name} must lie in [{lower}, {upper}]")
    return result


def _integer_array(value: object, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.kind not in "iu" or array.dtype.kind == "b":
        raise ModelError(f"{name} must contain integers")
    return array


@dataclass(frozen=True)
class ChannelHead:
    """One exact ``<8bh8bh>`` SSP2E channel record."""

    location_weights: tuple[int, ...]
    location_bias: int
    scale_weights: tuple[int, ...]
    scale_bias: int

    def __post_init__(self) -> None:
        location_weights = tuple(
            _bounded_integer(value, "location weight", -128, 127) for value in self.location_weights
        )
        scale_weights = tuple(
            _bounded_integer(value, "scale weight", -128, 127) for value in self.scale_weights
        )
        if len(location_weights) != FEATURE_COUNT or len(scale_weights) != FEATURE_COUNT:
            raise ModelError("each head axis must have exactly eight weights")
        object.__setattr__(self, "location_weights", location_weights)
        object.__setattr__(self, "scale_weights", scale_weights)
        object.__setattr__(
            self,
            "location_bias",
            _bounded_integer(self.location_bias, "location bias", -32768, 32767),
        )
        object.__setattr__(
            self,
            "scale_bias",
            _bounded_integer(self.scale_bias, "scale bias", -32768, 32767),
        )

    def to_bytes(self) -> bytes:
        """Serialize in the frozen signed little-endian field order."""
        return HEAD_RECORD.pack(
            *self.location_weights,
            self.location_bias,
            *self.scale_weights,
            self.scale_bias,
        )

    @classmethod
    def from_bytes(cls, payload: bytes) -> ChannelHead:
        if len(payload) != HEAD_RECORD.size:
            raise ModelError(f"channel head has {len(payload)} bytes, expected {HEAD_RECORD.size}")
        fields = HEAD_RECORD.unpack(payload)
        return cls(tuple(fields[:8]), fields[8], tuple(fields[9:17]), fields[17])


@dataclass(frozen=True)
class SSP2EModel:
    """The exact 256-byte grid and five-record, 100-byte head."""

    grid: bytes
    heads: tuple[ChannelHead, ...]

    def __post_init__(self) -> None:
        try:
            grid = bytes(self.grid)
        except (TypeError, ValueError) as exc:
            raise ModelError("grid must be byte-like") from exc
        heads = tuple(self.heads)
        if len(grid) != GRID_BYTES:
            raise ModelError(f"grid has {len(grid)} bytes, expected {GRID_BYTES}")
        if len(heads) != CHANNEL_COUNT or not all(isinstance(head, ChannelHead) for head in heads):
            raise ModelError("model must contain five ChannelHead records")
        object.__setattr__(self, "grid", grid)
        object.__setattr__(self, "heads", heads)

    @property
    def grid_raw(self) -> bytes:
        return self.grid

    @property
    def head_raw(self) -> bytes:
        payload = b"".join(head.to_bytes() for head in self.heads)
        if len(payload) != HEAD_BYTES:  # pragma: no cover - protected by fixed records
            raise ModelError("serialized head length drift")
        return payload

    @property
    def lexicographic_bytes(self) -> bytes:
        return self.grid_raw + self.head_raw

    @classmethod
    def from_raw(cls, grid_raw: bytes, head_raw: bytes) -> SSP2EModel:
        if len(head_raw) != HEAD_BYTES:
            raise ModelError(f"head has {len(head_raw)} bytes, expected {HEAD_BYTES}")
        heads = tuple(
            ChannelHead.from_bytes(head_raw[offset : offset + HEAD_RECORD.size])
            for offset in range(0, HEAD_BYTES, HEAD_RECORD.size)
        )
        return cls(grid_raw, heads)


def _feature_table() -> np.ndarray:
    values = np.arange(256, dtype=np.uint16)[:, None]
    shifts = np.arange(FEATURE_COUNT, dtype=np.uint16)[None, :]
    return (2 * ((values >> shifts) & 1).astype(np.int32) - 1).astype(np.int32)


FEATURE_TABLE = _feature_table()


def grid_features(grid: bytes | Sequence[int] | np.ndarray) -> np.ndarray:
    """Return exact least-significant-bit-first features for all 256 context cells."""
    values = (
        _integer_array(grid, "grid") if not isinstance(grid, bytes) else np.frombuffer(grid, "u1")
    )
    values = values.reshape(-1)
    if values.size != CELL_COUNT or (
        values.size and (int(values.min()) < 0 or int(values.max()) > 255)
    ):
        raise ModelError("grid must contain exactly 256 uint8 values")
    return FEATURE_TABLE[np.asarray(values, dtype=np.uint8)]


def infer_cell_parameters(model: SSP2EModel) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Infer signed-int32 ``(location, scale)`` arrays for the 256 cells."""
    phi = grid_features(model.grid)
    result: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, head in zip(CHANNELS, model.heads, strict=True):
        location = np.clip(
            np.int32(head.location_bias) + phi @ np.asarray(head.location_weights, dtype=np.int32),
            0,
            LOCATION_COUNT - 1,
        ).astype(np.uint8)
        scale = np.clip(
            np.int32(head.scale_bias) + phi @ np.asarray(head.scale_weights, dtype=np.int32),
            0,
            SCALE_COUNT - 1,
        ).astype(np.uint8)
        result[name] = (location, scale)
    return result


def infer_row_parameters(
    model: SSP2EModel, context_cells: Sequence[int] | np.ndarray
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Infer table rows for ordered symbols without introducing row-index context."""
    cells = _integer_array(context_cells, "context_cells").reshape(-1)
    if cells.size and (int(cells.min()) < 0 or int(cells.max()) >= CELL_COUNT):
        raise ModelError("context cell lies outside [0, 255]")
    by_cell = infer_cell_parameters(model)
    indices = cells.astype(np.uint16)
    return {name: (loc[indices], scale[indices]) for name, (loc, scale) in by_cell.items()}


def position_cells(
    mean_x: Sequence[int] | np.ndarray,
    mean_y: Sequence[int] | np.ndarray,
    bits_means: int,
) -> np.ndarray:
    """Map exact mean symbols to frozen row-major 16x16 position contexts."""
    bits = _bounded_integer(bits_means, "bits_means", 1, 32)
    x = _integer_array(mean_x, "mean_x").astype(np.uint64).reshape(-1)
    y = _integer_array(mean_y, "mean_y").astype(np.uint64).reshape(-1)
    if x.shape != y.shape:
        raise ModelError("mean_x and mean_y shapes disagree")
    limit = 1 << bits
    if (x.size and int(x.max()) >= limit) or (y.size and int(y.max()) >= limit):
        raise ModelError("mean symbol exceeds bits_means")
    cell_x = np.minimum(15, (x * 16) >> bits)
    cell_y = np.minimum(15, (y * 16) >> bits)
    return (16 * cell_y + cell_x).astype(np.uint16)


def _bitreverse8(value: int) -> int:
    result = 0
    for bit in range(8):
        result |= ((value >> bit) & 1) << (7 - bit)
    return result


def initial_grid(start_id: int) -> bytes:
    """Generate one of the eight frozen deterministic grid starts (IDs 0 through 7)."""
    start = _bounded_integer(start_id, "start_id", 0, 7)
    if start == 0:
        return bytes(range(CELL_COUNT))
    if start == 1:
        return bytes(cell ^ (cell >> 1) for cell in range(CELL_COUNT))
    if start == 2:
        return bytes(_bitreverse8(cell) for cell in range(CELL_COUNT))
    return bytes(
        hashlib.sha256(GRID_START_DOMAIN + struct.pack("<BH", start, cell)).digest()[0]
        for cell in range(CELL_COUNT)
    )


def all_initial_grids() -> tuple[bytes, ...]:
    return tuple(initial_grid(start) for start in range(8))


@lru_cache(maxsize=1)
def _shuffle_permutation_raw() -> bytes:
    keyed = [
        (hashlib.sha256(SHUFFLE_DOMAIN + struct.pack("<I", index)).digest(), index)
        for index in range(GAUSSIAN_COUNT)
    ]
    permutation = np.asarray(
        [index for _, index in sorted(keyed)],
        dtype="<u2",
    )
    payload = permutation.tobytes()
    observed = hashlib.sha256(payload).hexdigest()
    if observed != SHUFFLE_PERMUTATION_SHA256:
        raise ModelError(
            f"static shuffle permutation hash mismatch: {observed} != {SHUFFLE_PERMUTATION_SHA256}"
        )
    return payload


def shuffle_permutation() -> np.ndarray:
    """Return a fresh copy of the frozen uint16 permutation."""
    return np.frombuffer(_shuffle_permutation_raw(), dtype="<u2").copy()


def shuffled_position_contexts(context_cells: Sequence[int] | np.ndarray) -> np.ndarray:
    """Apply the static SSP2S negative-control permutation to true position cells."""
    cells = _integer_array(context_cells, "context_cells").reshape(-1)
    if cells.size != GAUSSIAN_COUNT:
        raise ModelError(f"SSP2S requires exactly {GAUSSIAN_COUNT} contexts")
    if cells.size and (int(cells.min()) < 0 or int(cells.max()) >= CELL_COUNT):
        raise ModelError("context cell lies outside [0, 255]")
    return cells.astype(np.uint16)[shuffle_permutation()]


@dataclass(frozen=True)
class ContextHistograms:
    """Five ordered ``[cell, symbol]`` count matrices."""

    counts: tuple[np.ndarray, ...]
    alphabets: tuple[int, ...]
    rows: int

    def __post_init__(self) -> None:
        counts = tuple(self.counts)
        alphabets = tuple(_bounded_integer(value, "alphabet", 1, 256) for value in self.alphabets)
        if len(counts) != CHANNEL_COUNT or len(alphabets) != CHANNEL_COUNT:
            raise ModelError("histograms must contain the five frozen channels")
        rows = _bounded_integer(self.rows, "rows", 0, GAUSSIAN_COUNT)
        normalized = []
        row_totals = None
        for name, count, alphabet in zip(CHANNELS, counts, alphabets, strict=True):
            if alphabet not in (64, 256):
                raise ModelError(f"{name} alphabet must be 64 or 256")
            array = _integer_array(count, f"{name} histogram")
            if (
                array.shape != (CELL_COUNT, alphabet)
                or np.any(array < 0)
                or (array.size and int(array.max()) > rows)
            ):
                raise ModelError(f"{name} histogram has invalid shape or negative count")
            array = np.ascontiguousarray(array, dtype=np.int64)
            totals = array.sum(axis=1, dtype=np.int64)
            if row_totals is None:
                row_totals = totals
            elif not np.array_equal(row_totals, totals):
                raise ModelError("channel histograms disagree on per-cell row counts")
            array.setflags(write=False)
            normalized.append(array)
        if row_totals is None or int(row_totals.sum(dtype=np.int64)) != rows:
            raise ModelError("histogram row total mismatch")
        object.__setattr__(self, "counts", tuple(normalized))
        object.__setattr__(self, "alphabets", alphabets)
        object.__setattr__(self, "rows", rows)


def aggregate_context_histograms(
    context_cells: Sequence[int] | np.ndarray,
    symbols: Mapping[str, Sequence[int] | np.ndarray],
    alphabets: Mapping[str, int],
) -> ContextHistograms:
    """Aggregate ordered symbols into exact per-context integer histograms."""
    if set(symbols) != set(CHANNELS) or set(alphabets) != set(CHANNELS):
        raise ModelError("symbols and alphabets must have exactly the five frozen channels")
    cells = _integer_array(context_cells, "context_cells").reshape(-1)
    if cells.size and (int(cells.min()) < 0 or int(cells.max()) >= CELL_COUNT):
        raise ModelError("context cell lies outside [0, 255]")
    cells64 = cells.astype(np.int64)
    matrices = []
    alphabet_order = []
    for name in CHANNELS:
        alphabet = _bounded_integer(alphabets[name], f"{name} alphabet", 1, 256)
        if alphabet not in (64, 256):
            raise ModelError(f"{name} alphabet must be 64 or 256")
        values = _integer_array(symbols[name], name).reshape(-1)
        if values.size != cells.size:
            raise ModelError(f"{name} has {values.size} rows, expected {cells.size}")
        if values.size and (int(values.min()) < 0 or int(values.max()) >= alphabet):
            raise ModelError(f"{name} symbol lies outside its alphabet")
        joint = cells64 * alphabet + values.astype(np.int64)
        counts = np.bincount(joint, minlength=CELL_COUNT * alphabet).reshape(CELL_COUNT, alphabet)
        matrices.append(counts)
        alphabet_order.append(alphabet)
    return ContextHistograms(tuple(matrices), tuple(alphabet_order), int(cells.size))


@dataclass(frozen=True)
class ModelObjective:
    """Precomputed exact Q24 cost for every ``[channel, cell, loc, scale]`` state."""

    cell_costs: tuple[np.ndarray, ...]

    def __post_init__(self) -> None:
        costs = tuple(self.cell_costs)
        if len(costs) != CHANNEL_COUNT:
            raise ModelError("objective must contain five channel cost tensors")
        normalized = []
        safe_cell_max = np.iinfo(np.int64).max // (CELL_COUNT * CHANNEL_COUNT)
        for name, cost in zip(CHANNELS, costs, strict=True):
            array = _integer_array(cost, f"{name} cell costs")
            if array.shape != (CELL_COUNT, LOCATION_COUNT, SCALE_COUNT) or np.any(array < 0):
                raise ModelError(f"{name} objective has invalid shape or negative cost")
            array = np.ascontiguousarray(array, dtype=np.int64)
            if array.size and int(array.max()) > safe_cell_max:
                raise ModelError(f"{name} objective could overflow signed int64 totals")
            array.setflags(write=False)
            normalized.append(array)
        object.__setattr__(self, "cell_costs", tuple(normalized))

    def channel_cost(self, channel_index: int, model: SSP2EModel) -> int:
        locations, scales = infer_cell_parameters(model)[CHANNELS[channel_index]]
        cells = np.arange(CELL_COUNT)
        return int(self.cell_costs[channel_index][cells, locations, scales].sum(dtype=np.int64))

    def objective_q24(self, model: SSP2EModel) -> int:
        inferred = infer_cell_parameters(model)
        cells = np.arange(CELL_COUNT)
        return sum(
            int(cost[cells, inferred[name][0], inferred[name][1]].sum(dtype=np.int64))
            for name, cost in zip(CHANNELS, self.cell_costs, strict=True)
        )


def build_model_objective(
    histograms: ContextHistograms,
    cdf_frequencies: CDFFrequencies,
    cost_by_frequency: Sequence[int] | np.ndarray,
) -> ModelObjective:
    """Consume only exact CDF frequencies and the exact integer Q24 cost asset."""
    costs = _integer_array(cost_by_frequency, "cost_by_frequency").reshape(-1)
    if costs.shape != (ARITHMETIC_TOTAL + 1,) or np.any(costs < 0):
        raise ModelError("cost_by_frequency must be a nonnegative integer array indexed 0..32768")
    if costs.size and int(costs.max()) > np.iinfo(np.uint32).max:
        raise ModelError("Q24 cost exceeds uint32 asset range")
    costs64 = costs.astype(np.int64)
    result = []
    for name, counts, alphabet in zip(
        CHANNELS, histograms.counts, histograms.alphabets, strict=True
    ):
        frequencies = np.empty((LOCATION_COUNT, SCALE_COUNT, alphabet), dtype=np.int64)
        for location in range(LOCATION_COUNT):
            for scale in range(SCALE_COUNT):
                row = _integer_array(
                    cdf_frequencies(name, location, scale),
                    f"CDF frequencies for {name}/{location}/{scale}",
                ).reshape(-1)
                if (
                    row.shape != (alphabet,)
                    or np.any(row <= 0)
                    or (row.size and int(row.max()) > ARITHMETIC_TOTAL)
                ):
                    raise ModelError("CDF frequency row has invalid shape or nonpositive entry")
                row64 = row.astype(np.int64)
                if int(row64.sum(dtype=np.int64)) != ARITHMETIC_TOTAL:
                    raise ModelError("CDF frequency row does not sum to 32768")
                frequencies[location, scale] = row64
        state_symbol_cost = costs64[frequencies].reshape(LOCATION_COUNT * SCALE_COUNT, alphabet)
        # Integer matrix multiplication is exact here: the frozen 8192-row/Q24 domain is well
        # inside signed int64.  ModelObjective independently checks the resulting bound.
        by_cell = counts @ state_symbol_cost.T
        result.append(by_cell.reshape(CELL_COUNT, LOCATION_COUNT, SCALE_COUNT))
    return ModelObjective(tuple(result))


def best_global_pairs(objective: ModelObjective) -> tuple[tuple[int, int], ...]:
    """Return exact SSP2L/global initialization pairs; ties are ascending ``(loc, scale)``."""
    result = []
    for costs in objective.cell_costs:
        totals = costs.sum(axis=0, dtype=np.int64)
        flat = int(np.argmin(totals.reshape(-1)))
        result.append(divmod(flat, SCALE_COUNT))
    return tuple(result)


@dataclass(frozen=True)
class SweepRecord:
    sweep: int
    objective_q24: int
    strict_updates: int
    head_updates: int
    grid_updates: int


@dataclass(frozen=True)
class FitResult:
    start_id: int
    model: SSP2EModel
    objective_q24: int
    sweeps: int
    trajectory: tuple[SweepRecord, ...]


@dataclass
class _MutableModel:
    grid: np.ndarray
    location_weights: np.ndarray
    location_biases: np.ndarray
    scale_weights: np.ndarray
    scale_biases: np.ndarray

    def freeze(self) -> SSP2EModel:
        heads = tuple(
            ChannelHead(
                tuple(int(value) for value in self.location_weights[channel]),
                int(self.location_biases[channel]),
                tuple(int(value) for value in self.scale_weights[channel]),
                int(self.scale_biases[channel]),
            )
            for channel in range(CHANNEL_COUNT)
        )
        return SSP2EModel(bytes(self.grid.astype(np.uint8)), heads)


def _mutable_from_model(model: SSP2EModel) -> _MutableModel:
    return _MutableModel(
        np.frombuffer(model.grid, dtype=np.uint8).copy(),
        np.asarray([head.location_weights for head in model.heads], dtype=np.int32),
        np.asarray([head.location_bias for head in model.heads], dtype=np.int32),
        np.asarray([head.scale_weights for head in model.heads], dtype=np.int32),
        np.asarray([head.scale_bias for head in model.heads], dtype=np.int32),
    )


def _initial_model(objective: ModelObjective, start_id: int) -> SSP2EModel:
    pairs = best_global_pairs(objective)
    zero = (0,) * FEATURE_COUNT
    heads = tuple(ChannelHead(zero, location, zero, scale) for location, scale in pairs)
    return SSP2EModel(initial_grid(start_id), heads)


def _axis_candidate_costs(
    cell_cost: np.ndarray,
    base: np.ndarray,
    candidates: np.ndarray,
    other_axis: np.ndarray,
    *,
    location_axis: bool,
) -> np.ndarray:
    """Evaluate one exhaustive coordinate in bounded chunks with exact int64 sums."""
    result = np.empty(candidates.size, dtype=np.int64)
    cell_indices = np.arange(CELL_COUNT, dtype=np.intp)[:, None]
    other = other_axis.astype(np.intp)[:, None]
    limit = LOCATION_COUNT - 1 if location_axis else SCALE_COUNT - 1
    for offset in range(0, candidates.size, 2048):
        chunk = candidates[offset : offset + 2048]
        predicted = np.clip(base[:, None] + chunk[None, :], 0, limit).astype(np.intp)
        if location_axis:
            selected = cell_cost[cell_indices, predicted, other]
        else:
            selected = cell_cost[cell_indices, other, predicted]
        result[offset : offset + chunk.size] = selected.sum(axis=0, dtype=np.int64)
    return result


def _strict_choice(
    candidates: np.ndarray, candidate_costs: np.ndarray, current_cost: int
) -> tuple[int | None, int]:
    minimum = int(candidate_costs.min())
    if minimum >= current_cost:
        return None, current_cost
    # Candidates are ascending and np.flatnonzero is ordered, so this is the required first
    # (smallest) improving minimizer.  A current-value tie never moves because minimum then cannot
    # be strictly below current_cost.
    index = int(np.flatnonzero(candidate_costs == minimum)[0])
    return int(candidates[index]), minimum


def _channel_predictions(
    state: _MutableModel, channel: int, phi: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    locations = np.clip(
        state.location_biases[channel] + phi @ state.location_weights[channel],
        0,
        LOCATION_COUNT - 1,
    ).astype(np.intp)
    scales = np.clip(
        state.scale_biases[channel] + phi @ state.scale_weights[channel],
        0,
        SCALE_COUNT - 1,
    ).astype(np.intp)
    return locations, scales


def _channel_cost(cell_cost: np.ndarray, locations: np.ndarray, scales: np.ndarray) -> int:
    return int(cell_cost[np.arange(CELL_COUNT), locations, scales].sum(dtype=np.int64))


def _bias_candidates(dot_products: np.ndarray, limit: int, current: int) -> np.ndarray:
    thresholds = (
        np.arange(limit + 1, dtype=np.int64)[:, None] - dot_products.astype(np.int64)[None, :]
    ).reshape(-1)
    values = np.concatenate([thresholds, np.asarray([-32768, 32767, current], dtype=np.int64)])
    return np.unique(np.clip(values, -32768, 32767)).astype(np.int32)


def _head_sweep(objective: ModelObjective, state: _MutableModel) -> int:
    updates = 0
    phi = FEATURE_TABLE[state.grid]
    weight_candidates = np.arange(-128, 128, dtype=np.int32)
    for channel, cell_cost in enumerate(objective.cell_costs):
        for feature in range(FEATURE_COUNT):
            locations, scales = _channel_predictions(state, channel, phi)
            current_cost = _channel_cost(cell_cost, locations, scales)
            dot = phi @ state.location_weights[channel]
            base = (
                state.location_biases[channel]
                + dot
                - phi[:, feature] * state.location_weights[channel, feature]
            )
            # _axis_candidate_costs adds one scalar candidate to every cell.  Weight candidates
            # instead have a +/- sign per cell, so evaluate that exact matrix directly.
            predicted = np.clip(
                base[:, None] + phi[:, feature, None] * weight_candidates[None, :],
                0,
                LOCATION_COUNT - 1,
            ).astype(np.intp)
            candidate_costs = cell_cost[
                np.arange(CELL_COUNT)[:, None], predicted, scales[:, None]
            ].sum(axis=0, dtype=np.int64)
            chosen, _ = _strict_choice(weight_candidates, candidate_costs, current_cost)
            if chosen is not None:
                state.location_weights[channel, feature] = chosen
                updates += 1

        locations, scales = _channel_predictions(state, channel, phi)
        current_cost = _channel_cost(cell_cost, locations, scales)
        dot = phi @ state.location_weights[channel]
        candidates = _bias_candidates(dot, LOCATION_COUNT - 1, state.location_biases[channel])
        candidate_costs = _axis_candidate_costs(
            cell_cost, dot, candidates, scales, location_axis=True
        )
        chosen, _ = _strict_choice(candidates, candidate_costs, current_cost)
        if chosen is not None:
            state.location_biases[channel] = chosen
            updates += 1

        for feature in range(FEATURE_COUNT):
            locations, scales = _channel_predictions(state, channel, phi)
            current_cost = _channel_cost(cell_cost, locations, scales)
            dot = phi @ state.scale_weights[channel]
            base = (
                state.scale_biases[channel]
                + dot
                - phi[:, feature] * state.scale_weights[channel, feature]
            )
            predicted = np.clip(
                base[:, None] + phi[:, feature, None] * weight_candidates[None, :],
                0,
                SCALE_COUNT - 1,
            ).astype(np.intp)
            candidate_costs = cell_cost[
                np.arange(CELL_COUNT)[:, None], locations[:, None], predicted
            ].sum(axis=0, dtype=np.int64)
            chosen, _ = _strict_choice(weight_candidates, candidate_costs, current_cost)
            if chosen is not None:
                state.scale_weights[channel, feature] = chosen
                updates += 1

        locations, scales = _channel_predictions(state, channel, phi)
        current_cost = _channel_cost(cell_cost, locations, scales)
        dot = phi @ state.scale_weights[channel]
        candidates = _bias_candidates(dot, SCALE_COUNT - 1, state.scale_biases[channel])
        candidate_costs = _axis_candidate_costs(
            cell_cost, dot, candidates, locations, location_axis=False
        )
        chosen, _ = _strict_choice(candidates, candidate_costs, current_cost)
        if chosen is not None:
            state.scale_biases[channel] = chosen
            updates += 1
    return updates


def _grid_sweep(objective: ModelObjective, state: _MutableModel) -> int:
    updates = 0
    candidates = np.arange(256, dtype=np.int32)
    candidate_parameters = []
    for channel in range(CHANNEL_COUNT):
        locations = np.clip(
            state.location_biases[channel] + FEATURE_TABLE @ state.location_weights[channel],
            0,
            LOCATION_COUNT - 1,
        ).astype(np.intp)
        scales = np.clip(
            state.scale_biases[channel] + FEATURE_TABLE @ state.scale_weights[channel],
            0,
            SCALE_COUNT - 1,
        ).astype(np.intp)
        candidate_parameters.append((locations, scales))
    for cell in range(CELL_COUNT):
        candidate_costs = np.zeros(256, dtype=np.int64)
        for cell_cost, (locations, scales) in zip(
            objective.cell_costs, candidate_parameters, strict=True
        ):
            candidate_costs += cell_cost[cell, locations, scales]
        current = int(candidate_costs[int(state.grid[cell])])
        chosen, _ = _strict_choice(candidates, candidate_costs, current)
        if chosen is not None:
            state.grid[cell] = chosen
            updates += 1
    return updates


def _outer_sweep(objective: ModelObjective, state: _MutableModel) -> tuple[int, int, int]:
    before = objective.objective_q24(state.freeze())
    head_updates = _head_sweep(objective, state)
    grid_updates = _grid_sweep(objective, state)
    after = objective.objective_q24(state.freeze())
    updates = head_updates + grid_updates
    if (updates == 0 and after != before) or (updates > 0 and after >= before):
        raise ModelError("coordinate sweep violated strict objective monotonicity")
    return after, head_updates, grid_updates


def fit_one_start(
    objective: ModelObjective,
    start_id: int,
    *,
    max_sweeps: int = MAX_OUTER_SWEEPS,
) -> FitResult:
    """Fit one start; ``max_sweeps`` exists only for bounded synthetic cap proofs."""
    start = _bounded_integer(start_id, "start_id", 0, 7)
    cap = _bounded_integer(max_sweeps, "max_sweeps", 1, MAX_OUTER_SWEEPS)
    state = _mutable_from_model(_initial_model(objective, start))
    initial_objective = objective.objective_q24(state.freeze())
    trajectory = [SweepRecord(0, initial_objective, 0, 0, 0)]
    for sweep in range(1, cap + 1):
        terminal, head_updates, grid_updates = _outer_sweep(objective, state)
        strict_updates = head_updates + grid_updates
        trajectory.append(SweepRecord(sweep, terminal, strict_updates, head_updates, grid_updates))
        if strict_updates == 0:
            model = state.freeze()
            return FitResult(start, model, terminal, sweep, tuple(trajectory))
    raise ConvergenceError(start, tuple(trajectory), state.freeze())


def fit_eight_starts(objective: ModelObjective) -> tuple[FitResult, ...]:
    """Fit all eight required starts with the immutable 64-sweep production cap."""
    return tuple(fit_one_start(objective, start, max_sweeps=MAX_OUTER_SWEEPS) for start in range(8))


def verify_fixed_point(objective: ModelObjective, model: SSP2EModel) -> bool:
    """Re-run one complete sweep on a copy and require byte/objective identity."""
    state = _mutable_from_model(model)
    before = objective.objective_q24(model)
    after, head_updates, grid_updates = _outer_sweep(objective, state)
    return (
        head_updates == 0
        and grid_updates == 0
        and after == before
        and state.freeze().lexicographic_bytes == model.lexicographic_bytes
    )


@dataclass(frozen=True)
class EncodedStart:
    fit: FitResult
    payload_lengths: tuple[int, ...]

    @property
    def total_payload_bytes(self) -> int:
        return sum(self.payload_lengths)

    @property
    def normative_key(self) -> tuple[int, int, bytes]:
        return (
            self.total_payload_bytes,
            self.fit.objective_q24,
            self.fit.model.lexicographic_bytes,
        )


@dataclass(frozen=True)
class ModelSelection:
    selected: EncodedStart
    encoded_starts: tuple[EncodedStart, ...]
    duplicate_winner_start_ids: tuple[int, ...]

    @property
    def selection_reason(self) -> dict[str, object]:
        return {
            "criteria": [
                "sum_five_arithmetic_payload_lengths",
                "exact_q24_objective",
                "lexicographic_grid_raw_head_raw",
            ],
            "selected_start_id": self.selected.fit.start_id,
            "selected_payload_lengths": list(self.selected.payload_lengths),
            "selected_payload_bytes": self.selected.total_payload_bytes,
            "selected_objective_q24": self.selected.fit.objective_q24,
            "duplicate_normative_winner_start_ids": list(self.duplicate_winner_start_ids),
        }


def _normalize_payload_lengths(value: Sequence[int] | Mapping[str, int]) -> tuple[int, ...]:
    if isinstance(value, Mapping):
        if set(value) != set(CHANNELS):
            raise ModelError("payload-length mapping must have exactly the five frozen channels")
        values = tuple(value[name] for name in CHANNELS)
    else:
        values = tuple(value)
    if len(values) != CHANNEL_COUNT:
        raise ModelError("encode_lengths must return five payload lengths")
    return tuple(
        _bounded_integer(length, "arithmetic payload length", 1, np.iinfo(np.int64).max)
        for length in values
    )


def select_fitted_model(fits: Sequence[FitResult], encode_lengths: EncodeLengths) -> ModelSelection:
    """Encode every converged start and apply the frozen three-level selection rule."""
    fit_tuple = tuple(fits)
    if len(fit_tuple) != 8 or tuple(fit.start_id for fit in fit_tuple) != tuple(range(8)):
        raise ModelError("selection requires exactly the ordered converged starts 0..7")
    encoded = tuple(
        EncodedStart(fit, _normalize_payload_lengths(encode_lengths(fit.model)))
        for fit in fit_tuple
    )
    best_key = min(candidate.normative_key for candidate in encoded)
    winners = tuple(candidate for candidate in encoded if candidate.normative_key == best_key)
    # Equal normative keys imply byte-identical models.  Lowest start ID only names the persisted
    # record; it does not add a fourth model-selection criterion.
    selected = min(winners, key=lambda candidate: candidate.fit.start_id)
    return ModelSelection(
        selected,
        encoded,
        tuple(candidate.fit.start_id for candidate in winners),
    )


def fit_and_select(objective: ModelObjective, encode_lengths: EncodeLengths) -> ModelSelection:
    """Convenience wrapper that still encodes all eight independently."""
    return select_fitted_model(fit_eight_starts(objective), encode_lengths)
