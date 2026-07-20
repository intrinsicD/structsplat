"""Exact deterministic integer Lloyd and sequential RVQ for COMP-011.

The module has two implementations of the same frozen kernel:

* :func:`fit_start_reference` is the small, auditable NumPy/Python proof; and
* :class:`NativeLloydFitter` wraps the independently implemented C++17 kernel.

Scientific fitting and the forced-sweep feasibility benchmark are intentionally
different entry points.  A forced sweep can therefore never leak into, extend,
or replace a scientific convergence trajectory.
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import shutil
import struct
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Protocol, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

PROTOCOL_ROWS = 8192
START_IDS = (0, 1, 2, 3)
MAX_UPDATE_SWEEPS = 100
INT16_MIN = -(1 << 15)
INT16_MAX = (1 << 15) - 1
INT32_MIN = -(1 << 31)
INT32_MAX = (1 << 31) - 1
INT64_MAX = (1 << 63) - 1
UINT64_MAX = (1 << 64) - 1

START_DOMAIN = b"structsplat-comp011-vq-start-v1\0"
STATE_DOMAIN = b"structsplat-comp011-vq-state-v1\0"
COMPLEXITY_DOMAIN = b"structsplat-comp011-complexity-v1\0"
NATIVE_SOURCE_PATH = Path(__file__).with_name("ssp2v_lloyd_ext.cpp")

ConvergenceStatus = Literal["fixed", "cycle", "cap"]


class LloydError(ValueError):
    """The requested exact Lloyd operation is invalid."""


class LloydConvergenceError(LloydError):
    """At least one prescribed start did not reach a fixed point."""

    def __init__(self, starts: tuple[StartResult, ...]) -> None:
        self.starts = starts
        failures = ", ".join(
            f"start {result.start_id}: {result.status}" for result in starts if not result.converged
        )
        super().__init__(f"all four prescribed starts must converge ({failures})")


@dataclass(frozen=True)
class LloydSpec:
    """Metadata that fixes one flat or one RVQ-stage Lloyd problem."""

    family_id: int
    matched_stage_count: int
    base_k: int
    actual_stage_index: int
    codebook_size: int

    def __post_init__(self) -> None:
        if self.family_id not in (0, 1):
            raise LloydError("family_id must be 0 (flat) or 1 (RVQ)")
        if self.matched_stage_count not in (2, 3):
            raise LloydError("matched_stage_count must be two or three")
        if not 1 <= self.base_k <= 0xFFFF:
            raise LloydError("base_k must fit the frozen uint16 start-key field")
        if self.codebook_size <= 0:
            raise LloydError("codebook_size must be positive")
        if self.family_id == 0:
            expected = (2 * self.matched_stage_count - 1) * self.base_k
            if self.actual_stage_index != 0:
                raise LloydError("a flat fit has only actual stage zero")
            if self.codebook_size != expected:
                raise LloydError(f"flat codebook_size must equal {expected}")
        else:
            if not 0 <= self.actual_stage_index < self.matched_stage_count:
                raise LloydError("RVQ actual_stage_index is outside its matched stage count")
            if self.codebook_size != self.base_k:
                raise LloydError("an RVQ stage codebook_size must equal base_k")

    @property
    def stores_unsigned_bytes(self) -> bool:
        return self.actual_stage_index == 0


@dataclass(frozen=True)
class WeightedVectors:
    """Lexicographically sorted unique signed-int32 triples and exact counts."""

    vectors: NDArray[np.int32] = field(repr=False, compare=False)
    counts: tuple[int, ...]
    original_count: int


@dataclass(frozen=True)
class TrajectoryRecord:
    """One canonical codebook state, including the repeated terminal fixed state."""

    state_index: int
    update_sweep: int
    codebook_sha256: str
    objective_sse: int


@dataclass(frozen=True)
class StartResult:
    """Complete scientific result for one frozen start."""

    spec: LloydSpec
    start_id: int
    status: ConvergenceStatus
    update_sweeps: int
    terminal_sse: int
    codebook: NDArray[np.int32] = field(repr=False, compare=False)
    indices: NDArray[np.uint32] = field(repr=False, compare=False)
    trajectory: tuple[TrajectoryRecord, ...]

    @property
    def converged(self) -> bool:
        return self.status == "fixed"

    @property
    def serialized_codebook(self) -> bytes:
        return serialize_codebook(self.codebook, self.spec)


@dataclass(frozen=True)
class StageFitResult:
    """All four converged starts and their frozen winner for one stage."""

    spec: LloydSpec
    starts: tuple[StartResult, ...]
    selected_start_id: int
    codebook: NDArray[np.int32] = field(repr=False, compare=False)
    indices: NDArray[np.uint32] = field(repr=False, compare=False)
    terminal_sse: int

    @property
    def winner(self) -> StartResult:
        return self.starts[self.selected_start_id]


@dataclass(frozen=True)
class VariantFitResult:
    """A complete flat or strictly sequential RVQ symbol fit."""

    family_id: int
    matched_stage_count: int
    base_k: int
    stages: tuple[StageFitResult, ...]
    raw_accumulator: NDArray[np.int32] = field(repr=False, compare=False)
    reconstructed_rgb: NDArray[np.uint8] = field(repr=False, compare=False)
    stage_indices: tuple[NDArray[np.uint32], ...] = field(repr=False, compare=False)


@dataclass(frozen=True)
class ForcedSweepResult:
    """Timing-only output; deliberately contains no convergence status or trajectory."""

    spec: LloydSpec
    start_id: int
    forced_sweeps: int
    final_codebook: NDArray[np.int32] = field(repr=False, compare=False)
    final_codebook_sha256: str


class StartFitter(Protocol):
    def fit_start(
        self,
        rows: ArrayLike,
        spec: LloydSpec,
        start_id: int,
        *,
        max_update_sweeps: int = MAX_UPDATE_SWEEPS,
    ) -> StartResult: ...


def _readonly(array: NDArray) -> NDArray:
    result = np.ascontiguousarray(array).copy()
    result.flags.writeable = False
    return result


def _prepare_rows(rows: ArrayLike, *, name: str = "vectors") -> NDArray[np.int32]:
    array = np.asarray(rows)
    if array.ndim != 2 or array.shape[1:] != (3,):
        raise LloydError(f"{name} must have shape [N,3]")
    if len(array) == 0:
        raise LloydError(f"{name} must be nonempty")
    if array.dtype.kind not in "iu":
        raise LloydError(f"{name} must contain integers")
    if array.dtype.kind == "u":
        maximum = int(array.max())
        minimum = 0
    else:
        maximum = int(array.max())
        minimum = int(array.min())
    if minimum < INT32_MIN or maximum > INT32_MAX:
        raise LloydError(f"{name} must be exactly representable as signed int32")
    return np.ascontiguousarray(array, dtype=np.int32)


def _validate_storage_domain(rows: NDArray[np.int32], spec: LloydSpec) -> None:
    minimum = int(rows.min())
    maximum = int(rows.max())
    if spec.stores_unsigned_bytes:
        if minimum < 0 or maximum > 255:
            raise LloydError("flat/RVQ-stage-0 inputs and centroids must lie in [0,255]")


def _validate_centroid_storage(codebook: NDArray[np.int32], spec: LloydSpec) -> None:
    minimum = int(codebook.min())
    maximum = int(codebook.max())
    if spec.stores_unsigned_bytes:
        if minimum < 0 or maximum > 255:
            raise LloydError("stage-zero centroid is not uint8-representable")
    elif minimum < INT16_MIN or maximum > INT16_MAX:
        raise LloydError("later RVQ centroid is not int16-representable")


def _check_distance_domain(rows: NDArray[np.int32]) -> None:
    spans = [int(rows[:, coordinate].max()) - int(rows[:, coordinate].min()) for coordinate in range(3)]
    if sum(span * span for span in spans) > INT64_MAX:
        raise LloydError("an exact squared distance can exceed signed int64")


def collapse_weighted(rows: ArrayLike) -> WeightedVectors:
    """Collapse rows into the protocol's lexicographically sorted weighted state."""

    prepared = _prepare_rows(rows)
    _check_distance_domain(prepared)
    vectors, counts = np.unique(prepared, axis=0, return_counts=True)
    frozen_vectors = _readonly(vectors.astype(np.int32, copy=False))
    exact_counts = tuple(int(value) for value in counts)
    if sum(exact_counts) != len(prepared) or any(value <= 0 for value in exact_counts):
        raise AssertionError("weighted unique-vector accounting failed")
    return WeightedVectors(frozen_vectors, exact_counts, len(prepared))


def start_key(vector: Sequence[int], spec: LloydSpec, start_id: int) -> bytes:
    """Return the exact 32-byte SHA-256 key used by frozen farthest-first starts."""

    if start_id not in START_IDS:
        raise LloydError("start_id must be one of 0,1,2,3")
    if len(vector) != 3:
        raise LloydError("a start-key vector must have three components")
    components = tuple(int(value) for value in vector)
    if any(value < INT32_MIN or value > INT32_MAX for value in components):
        raise LloydError("start-key vector component is outside signed int32")
    suffix = struct.pack(
        "<BBHBBiii",
        spec.family_id,
        spec.matched_stage_count,
        spec.base_k,
        spec.actual_stage_index,
        start_id,
        *components,
    )
    return hashlib.sha256(START_DOMAIN + suffix).digest()


def _codebook_state_bytes(codebook: ArrayLike) -> bytes:
    prepared = _prepare_rows(codebook, name="codebook")
    tuples = [tuple(int(value) for value in row) for row in prepared]
    if tuples != sorted(tuples):
        raise LloydError("codebook labels must be lexicographically canonical")
    header = struct.pack("<I", len(prepared))
    body = prepared.astype("<i4", copy=False).tobytes(order="C")
    return header + body


def codebook_state_sha256(codebook: ArrayLike) -> str:
    """Hash a canonical internal int32 codebook state with domain separation."""

    return hashlib.sha256(STATE_DOMAIN + _codebook_state_bytes(codebook)).hexdigest()


def serialize_codebook(codebook: ArrayLike, spec: LloydSpec) -> bytes:
    """Serialize a terminal codebook exactly as SSP2V charges it."""

    prepared = _prepare_rows(codebook, name="codebook")
    if len(prepared) != spec.codebook_size:
        raise LloydError("codebook row count disagrees with codebook_size")
    tuples = [tuple(int(value) for value in row) for row in prepared]
    if tuples != sorted(tuples):
        raise LloydError("codebook labels must be lexicographically canonical")
    _validate_centroid_storage(prepared, spec)
    if spec.stores_unsigned_bytes:
        return prepared.astype(np.uint8, copy=False).tobytes(order="C")
    return prepared.astype("<i2", copy=False).tobytes(order="C")


def round_weighted_mean(sum_value: int, count: int) -> int:
    """Exact floor-divmod nearest integer; a half tie chooses the smaller integer."""

    if count <= 0:
        raise LloydError("weighted-mean count must be positive")
    quotient, remainder = divmod(int(sum_value), int(count))
    return quotient if 2 * remainder <= count else quotient + 1


def _distance_to_one(vectors: NDArray[np.int32], centroid: NDArray[np.int32]) -> NDArray[np.int64]:
    difference = vectors.astype(np.int64) - centroid.astype(np.int64)
    return np.sum(difference * difference, axis=1, dtype=np.int64)


def initialize_codebook(weighted: WeightedVectors, spec: LloydSpec, start_id: int) -> NDArray[np.int32]:
    """Build one exact SHA-keyed farthest-first codebook and canonicalize labels."""

    vectors = weighted.vectors
    _validate_storage_domain(vectors, spec)
    keys = [start_key(vector, spec, start_id) for vector in vectors]
    vector_tuples = [tuple(int(value) for value in vector) for vector in vectors]
    first = min(range(len(vectors)), key=lambda index: (keys[index], vector_tuples[index]))
    selected = [first]
    selected_mask = np.zeros(len(vectors), dtype=np.bool_)
    selected_mask[first] = True
    nearest = _distance_to_one(vectors, vectors[first])

    while len(selected) < min(len(vectors), spec.codebook_size):
        candidates = np.flatnonzero(~selected_mask)
        maximum_distance = int(nearest[candidates].max())
        tied = [int(index) for index in candidates if int(nearest[index]) == maximum_distance]
        chosen = min(tied, key=lambda index: (keys[index], vector_tuples[index]))
        selected.append(chosen)
        selected_mask[chosen] = True
        nearest = np.minimum(nearest, _distance_to_one(vectors, vectors[chosen]))

    centroids = [vector_tuples[index] for index in selected]
    if len(vectors) < spec.codebook_size:
        cyclic_order = sorted(
            range(len(vectors)), key=lambda index: (keys[index], vector_tuples[index])
        )
        missing = spec.codebook_size - len(centroids)
        centroids.extend(vector_tuples[cyclic_order[index % len(cyclic_order)]] for index in range(missing))
    codebook = np.asarray(sorted(centroids), dtype=np.int32)
    _validate_centroid_storage(codebook, spec)
    return np.ascontiguousarray(codebook)


def _nearest_indices_and_distances(
    vectors: NDArray[np.int32], codebook: NDArray[np.int32]
) -> tuple[NDArray[np.uint32], NDArray[np.int64]]:
    best_indices = np.zeros(len(vectors), dtype=np.uint32)
    best_distances = np.full(len(vectors), INT64_MAX, dtype=np.int64)
    vectors64 = vectors.astype(np.int64, copy=False)
    # Chunking bounds temporary memory while preserving global lowest-label ties.
    for begin in range(0, len(codebook), 128):
        block = codebook[begin : begin + 128].astype(np.int64, copy=False)
        difference = vectors64[:, None, :] - block[None, :, :]
        distances = np.sum(difference * difference, axis=2, dtype=np.int64)
        local_indices = np.argmin(distances, axis=1)
        local_distances = distances[np.arange(len(vectors)), local_indices]
        better = local_distances < best_distances
        best_distances[better] = local_distances[better]
        best_indices[better] = (local_indices[better] + begin).astype(np.uint32)
    return best_indices, best_distances


def _objective(
    weighted: WeightedVectors, codebook: NDArray[np.int32]
) -> tuple[int, NDArray[np.uint32]]:
    assignments, distances = _nearest_indices_and_distances(weighted.vectors, codebook)
    objective = sum(int(distance) * count for distance, count in zip(distances, weighted.counts))
    if objective < 0 or objective > UINT64_MAX:
        raise LloydError("weighted Lloyd objective exceeds checked uint64")
    return objective, assignments


def _updated_codebook(
    weighted: WeightedVectors,
    codebook: NDArray[np.int32],
    spec: LloydSpec,
) -> NDArray[np.int32]:
    assignments, _ = _nearest_indices_and_distances(weighted.vectors, codebook)
    counts = [0] * len(codebook)
    sums = [[0, 0, 0] for _ in range(len(codebook))]
    for vector, weight, raw_index in zip(weighted.vectors, weighted.counts, assignments):
        index = int(raw_index)
        counts[index] += weight
        for coordinate in range(3):
            sums[index][coordinate] += int(vector[coordinate]) * weight

    updated: list[tuple[int, int, int]] = []
    for index, old in enumerate(codebook):
        if counts[index] == 0:
            updated.append(tuple(int(value) for value in old))
        else:
            updated.append(
                tuple(round_weighted_mean(sums[index][coordinate], counts[index]) for coordinate in range(3))
            )
    result = np.asarray(sorted(updated), dtype=np.int32)
    _validate_centroid_storage(result, spec)
    return np.ascontiguousarray(result)


def _materialize_indices(rows: NDArray[np.int32], codebook: NDArray[np.int32]) -> NDArray[np.uint32]:
    indices, _ = _nearest_indices_and_distances(rows, codebook)
    return np.ascontiguousarray(indices, dtype=np.uint32)


def _transition_status(
    previous_state: bytes,
    next_state: bytes,
    earlier_states: set[bytes],
) -> Literal["fixed", "cycle", "continue"]:
    """Classify exact state bytes; fixed-point priority is part of the protocol."""

    if next_state == previous_state:
        return "fixed"
    if next_state in earlier_states:
        return "cycle"
    return "continue"


def fit_start_reference(
    rows: ArrayLike,
    spec: LloydSpec,
    start_id: int,
    *,
    max_update_sweeps: int = MAX_UPDATE_SWEEPS,
) -> StartResult:
    """Run one exact scientific start using the auditable Python proof."""

    if not 1 <= max_update_sweeps <= MAX_UPDATE_SWEEPS:
        raise LloydError("max_update_sweeps must lie in [1,100]")
    prepared = _prepare_rows(rows)
    _validate_storage_domain(prepared, spec)
    _check_distance_domain(prepared)
    weighted = collapse_weighted(prepared)
    codebook = initialize_codebook(weighted, spec, start_id)
    initial_objective, _ = _objective(weighted, codebook)
    trajectory = [
        TrajectoryRecord(0, 0, codebook_state_sha256(codebook), initial_objective)
    ]
    state_bytes = _codebook_state_bytes(codebook)
    seen_states = {state_bytes}
    status: ConvergenceStatus = "cap"

    for sweep in range(1, max_update_sweeps + 1):
        next_codebook = _updated_codebook(weighted, codebook, spec)
        next_objective, _ = _objective(weighted, next_codebook)
        if next_objective > trajectory[-1].objective_sse:
            raise LloydError("an exact Lloyd update increased its weighted objective")
        trajectory.append(
            TrajectoryRecord(
                len(trajectory),
                sweep,
                codebook_state_sha256(next_codebook),
                next_objective,
            )
        )
        next_bytes = _codebook_state_bytes(next_codebook)
        transition = _transition_status(state_bytes, next_bytes, seen_states)
        codebook = next_codebook
        state_bytes = next_bytes
        if transition == "fixed":
            status = "fixed"
            break
        if transition == "cycle":
            status = "cycle"
            break
        seen_states.add(next_bytes)

    terminal_sse, _ = _objective(weighted, codebook)
    indices = _materialize_indices(prepared, codebook)
    return StartResult(
        spec=spec,
        start_id=start_id,
        status=status,
        update_sweeps=len(trajectory) - 1,
        terminal_sse=terminal_sse,
        codebook=_readonly(codebook),
        indices=_readonly(indices),
        trajectory=tuple(trajectory),
    )


def force_sweeps_reference(
    rows: ArrayLike,
    spec: LloydSpec,
    start_id: int,
    *,
    forced_sweeps: int = 10,
) -> ForcedSweepResult:
    """Run an exact number of updates without producing scientific convergence evidence."""

    if forced_sweeps <= 0:
        raise LloydError("forced_sweeps must be positive")
    prepared = _prepare_rows(rows)
    _validate_storage_domain(prepared, spec)
    _check_distance_domain(prepared)
    weighted = collapse_weighted(prepared)
    codebook = initialize_codebook(weighted, spec, start_id)
    for _ in range(forced_sweeps):
        codebook = _updated_codebook(weighted, codebook, spec)
    return ForcedSweepResult(
        spec,
        start_id,
        forced_sweeps,
        _readonly(codebook),
        codebook_state_sha256(codebook),
    )


def _winner_key(result: StartResult) -> tuple[int, bytes, tuple[int, ...], int]:
    return (
        result.terminal_sse,
        result.serialized_codebook,
        tuple(int(value) for value in result.indices),
        result.start_id,
    )


def fit_stage(
    rows: ArrayLike,
    spec: LloydSpec,
    *,
    fitter: StartFitter | None = None,
) -> StageFitResult:
    """Run all four starts, reject any failed start, and choose the exact winner."""

    engine: StartFitter = fitter if fitter is not None else _ReferenceFitter()
    starts = tuple(engine.fit_start(rows, spec, start_id) for start_id in START_IDS)
    if any(not result.converged for result in starts):
        raise LloydConvergenceError(starts)
    winner = min(starts, key=_winner_key)
    return StageFitResult(
        spec=spec,
        starts=starts,
        selected_start_id=winner.start_id,
        codebook=_readonly(winner.codebook),
        indices=_readonly(winner.indices),
        terminal_sse=winner.terminal_sse,
    )


class _ReferenceFitter:
    def fit_start(
        self,
        rows: ArrayLike,
        spec: LloydSpec,
        start_id: int,
        *,
        max_update_sweeps: int = MAX_UPDATE_SWEEPS,
    ) -> StartResult:
        return fit_start_reference(
            rows, spec, start_id, max_update_sweeps=max_update_sweeps
        )


def _prepare_protocol_rgb(rows: ArrayLike, require_protocol_rows: bool) -> NDArray[np.int32]:
    prepared = _prepare_rows(rows, name="rgb")
    if require_protocol_rows and len(prepared) != PROTOCOL_ROWS:
        raise LloydError(f"COMP-011 RGB must contain exactly {PROTOCOL_ROWS} rows")
    if int(prepared.min()) < 0 or int(prepared.max()) > 255:
        raise LloydError("absolute RGB symbols must lie in [0,255]")
    return prepared


def _validate_protocol_menu(base_k: int, require_protocol_rows: bool) -> None:
    if require_protocol_rows and base_k not in (64, 128):
        raise LloydError("COMP-011 protocol base_k must be 64 or 128")


def fit_flat(
    rgb: ArrayLike,
    matched_stage_count: int,
    base_k: int,
    *,
    fitter: StartFitter | None = None,
    require_protocol_rows: bool = True,
) -> VariantFitResult:
    """Fit one matched-capacity flat arm; no target or rate enters this function."""

    source = _prepare_protocol_rgb(rgb, require_protocol_rows)
    _validate_protocol_menu(base_k, require_protocol_rows)
    spec = LloydSpec(
        family_id=0,
        matched_stage_count=matched_stage_count,
        base_k=base_k,
        actual_stage_index=0,
        codebook_size=(2 * matched_stage_count - 1) * base_k,
    )
    stage = fit_stage(source, spec, fitter=fitter)
    accumulator = stage.codebook[stage.indices].astype(np.int32, copy=False)
    reconstructed = np.clip(accumulator, 0, 255).astype(np.uint8)
    return VariantFitResult(
        0,
        matched_stage_count,
        base_k,
        (stage,),
        _readonly(accumulator),
        _readonly(reconstructed),
        (_readonly(stage.indices),),
    )


def fit_rvq(
    rgb: ArrayLike,
    matched_stage_count: int,
    base_k: int,
    *,
    fitter: StartFitter | None = None,
    require_protocol_rows: bool = True,
) -> VariantFitResult:
    """Fit strict sequential RVQ, subtracting un-clipped integer reconstructions."""

    source = _prepare_protocol_rgb(rgb, require_protocol_rows)
    _validate_protocol_menu(base_k, require_protocol_rows)
    accumulator = np.zeros_like(source, dtype=np.int32)
    stages: list[StageFitResult] = []
    indices: list[NDArray[np.uint32]] = []
    for stage_index in range(matched_stage_count):
        residual = source.astype(np.int64) - accumulator.astype(np.int64)
        if np.any(residual < INT32_MIN) or np.any(residual > INT32_MAX):
            raise LloydError("sequential RVQ residual is outside signed int32")
        stage_input = np.ascontiguousarray(residual, dtype=np.int32)
        spec = LloydSpec(1, matched_stage_count, base_k, stage_index, base_k)
        stage = fit_stage(stage_input, spec, fitter=fitter)
        contribution = stage.codebook[stage.indices].astype(np.int64, copy=False)
        updated = accumulator.astype(np.int64) + contribution
        if np.any(updated < INT32_MIN) or np.any(updated > INT32_MAX):
            raise LloydError("sequential RVQ accumulator is outside signed int32")
        accumulator = np.ascontiguousarray(updated, dtype=np.int32)
        stages.append(stage)
        indices.append(_readonly(stage.indices))
    reconstructed = np.clip(accumulator, 0, 255).astype(np.uint8)
    return VariantFitResult(
        1,
        matched_stage_count,
        base_k,
        tuple(stages),
        _readonly(accumulator),
        _readonly(reconstructed),
        tuple(indices),
    )


def complexity_fixture(n: int = PROTOCOL_ROWS) -> NDArray[np.uint8]:
    """Construct only the target-free protocol synthetic RGB complexity fixture."""

    if n <= 0 or n > 0x1_0000_0000:
        raise LloydError("synthetic fixture row count is outside uint32")
    result = np.empty((n, 3), dtype=np.uint8)
    for index in range(n):
        digest = hashlib.sha256(COMPLEXITY_DOMAIN + struct.pack("<I", index)).digest()
        result[index] = (digest[0], digest[1], digest[2])
    return result


def build_native(
    output_path: Path,
    *,
    compiler: str | None = None,
    extra_flags: Sequence[str] = (),
) -> Path:
    """Build the independent C++17 exact fitter as a shared library."""

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


class NativeLloydFitter:
    """Strict ctypes wrapper over the independently implemented native kernel."""

    _SUCCESS = 0

    def __init__(self, library_path: Path) -> None:
        self._library = ctypes.CDLL(str(Path(library_path).resolve()))
        i32p = ctypes.POINTER(ctypes.c_int32)
        u32p = ctypes.POINTER(ctypes.c_uint32)
        u64p = ctypes.POINTER(ctypes.c_uint64)
        u8p = ctypes.POINTER(ctypes.c_uint8)
        sizep = ctypes.POINTER(ctypes.c_size_t)
        self._library.ssp2v_lloyd_fit.argtypes = [
            i32p,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_uint8,
            ctypes.c_uint8,
            ctypes.c_uint16,
            ctypes.c_uint8,
            ctypes.c_uint8,
            ctypes.c_uint32,
            i32p,
            u32p,
            u8p,
            u64p,
            ctypes.c_size_t,
            sizep,
            u32p,
            u32p,
            u64p,
        ]
        self._library.ssp2v_lloyd_fit.restype = ctypes.c_int
        self._library.ssp2v_lloyd_force_sweeps.argtypes = [
            i32p,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_uint8,
            ctypes.c_uint8,
            ctypes.c_uint16,
            ctypes.c_uint8,
            ctypes.c_uint8,
            ctypes.c_uint32,
            i32p,
            u8p,
        ]
        self._library.ssp2v_lloyd_force_sweeps.restype = ctypes.c_int
        self._library.ssp2v_lloyd_last_error.argtypes = []
        self._library.ssp2v_lloyd_last_error.restype = ctypes.c_char_p

    def _error(self, operation: str, status: int) -> LloydError:
        raw = self._library.ssp2v_lloyd_last_error()
        message = raw.decode("utf-8", errors="replace") if raw else "unknown native error"
        return LloydError(f"native {operation} failed ({status}): {message}")

    @staticmethod
    def _inputs(rows: ArrayLike, spec: LloydSpec, start_id: int) -> NDArray[np.int32]:
        prepared = _prepare_rows(rows)
        _validate_storage_domain(prepared, spec)
        _check_distance_domain(prepared)
        if start_id not in START_IDS:
            raise LloydError("start_id must be one of 0,1,2,3")
        return prepared

    def fit_start(
        self,
        rows: ArrayLike,
        spec: LloydSpec,
        start_id: int,
        *,
        max_update_sweeps: int = MAX_UPDATE_SWEEPS,
    ) -> StartResult:
        if not 1 <= max_update_sweeps <= MAX_UPDATE_SWEEPS:
            raise LloydError("max_update_sweeps must lie in [1,100]")
        prepared = self._inputs(rows, spec, start_id)
        codebook = np.empty((spec.codebook_size, 3), dtype=np.int32)
        indices = np.empty(len(prepared), dtype=np.uint32)
        capacity = max_update_sweeps + 1
        hashes = np.empty((capacity, 32), dtype=np.uint8)
        objectives = np.empty(capacity, dtype=np.uint64)
        trajectory_count = ctypes.c_size_t()
        update_sweeps = ctypes.c_uint32()
        convergence_status = ctypes.c_uint32()
        terminal_sse = ctypes.c_uint64()
        status = self._library.ssp2v_lloyd_fit(
            prepared.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            len(prepared),
            spec.codebook_size,
            spec.family_id,
            spec.matched_stage_count,
            spec.base_k,
            spec.actual_stage_index,
            start_id,
            max_update_sweeps,
            codebook.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            indices.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            hashes.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            objectives.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
            capacity,
            ctypes.byref(trajectory_count),
            ctypes.byref(update_sweeps),
            ctypes.byref(convergence_status),
            ctypes.byref(terminal_sse),
        )
        if status != self._SUCCESS:
            raise self._error("fit", status)
        status_names: tuple[ConvergenceStatus, ...] = ("fixed", "cycle", "cap")
        if convergence_status.value >= len(status_names):
            raise LloydError("native fitter returned an unknown convergence status")
        count = trajectory_count.value
        if count != update_sweeps.value + 1 or not 2 <= count <= capacity:
            raise LloydError("native fitter returned an inconsistent trajectory length")
        trajectory = tuple(
            TrajectoryRecord(
                state_index=index,
                update_sweep=index,
                codebook_sha256=bytes(hashes[index]).hex(),
                objective_sse=int(objectives[index]),
            )
            for index in range(count)
        )
        if trajectory[-1].objective_sse != terminal_sse.value:
            raise LloydError("native terminal objective disagrees with its trajectory")
        if codebook_state_sha256(codebook) != trajectory[-1].codebook_sha256:
            raise LloydError("native terminal codebook hash disagrees with Python proof")
        return StartResult(
            spec,
            start_id,
            status_names[convergence_status.value],
            update_sweeps.value,
            terminal_sse.value,
            _readonly(codebook),
            _readonly(indices),
            trajectory,
        )

    def force_sweeps(
        self,
        rows: ArrayLike,
        spec: LloydSpec,
        start_id: int,
        *,
        forced_sweeps: int = 10,
    ) -> ForcedSweepResult:
        if not 1 <= forced_sweeps <= 0xFFFF_FFFF:
            raise LloydError("forced_sweeps must be a positive uint32")
        prepared = self._inputs(rows, spec, start_id)
        codebook = np.empty((spec.codebook_size, 3), dtype=np.int32)
        digest = np.empty(32, dtype=np.uint8)
        status = self._library.ssp2v_lloyd_force_sweeps(
            prepared.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            len(prepared),
            spec.codebook_size,
            spec.family_id,
            spec.matched_stage_count,
            spec.base_k,
            spec.actual_stage_index,
            start_id,
            forced_sweeps,
            codebook.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            digest.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        )
        if status != self._SUCCESS:
            raise self._error("forced sweeps", status)
        python_digest = codebook_state_sha256(codebook)
        if python_digest != bytes(digest).hex():
            raise LloydError("native forced-sweep hash disagrees with Python proof")
        return ForcedSweepResult(
            spec,
            start_id,
            forced_sweeps,
            _readonly(codebook),
            python_digest,
        )
