"""Data-free coordinate-search core for COMP-012 exact-byte RGB RDO.

The module intentionally knows nothing about SSP2F's renderer, target data, or filesystem
lifecycle. Callers inject one coordinated adapter with incremental objective proposals, exact
candidate-rate proposals, checkpoint-only full materialization, and atomic paired commits.

The search owns the frozen traversal, common-epsilon objective classification, objective-first
lazy rate schedule, cap enforcement, stale-evidence checks, immediate Gauss-Seidel commits,
cycle/resource failure handling, and epsilon-terminal full-label certificate.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

import numpy as np


LEDGER_SCHEMA = "structsplat.comp012.ssp2f-coordinate-rdo-ledger.v3"
STATE_DIGEST_DOMAIN = b"structsplat.comp012.rgb-state.v1\x00"
PROPOSAL_TRANSCRIPT_DOMAIN = b"structsplat.comp012.proposal-transcript.v3\x00"
VISITED_STATE_TRANSCRIPT_DOMAIN = b"structsplat.comp012.visited-state-transcript.v1\x00"
OBJECTIVE_RELATIVE_TOLERANCE = 1e-12
PROPOSAL_TRANSCRIPT_CHUNK_SIZE = 1024
CHANNEL_NAMES = ("R", "G", "B")
L1 = tuple(range(256))
L3 = tuple(range(0, 256, 3))
FROZEN_UNIFORM_STEPS = (1, 2, 3, 4, 5, 6, 8, 10, 12, 16)
SSP2F_COMPONENT_KEYS = (
    "prefix",
    "means",
    "scale_x_arithmetic",
    "scale_y_arithmetic",
    "rotation",
    "red_arithmetic",
    "green_arithmetic",
    "blue_arithmetic",
    "model_zlib9",
    "total",
)


class CoordinateRDOError(ValueError):
    """The caller supplied an invalid static search configuration."""


class _RuntimeFailure(RuntimeError):
    def __init__(self, reason: str, detail: str) -> None:
        super().__init__(detail)
        self.reason = reason
        self.detail = detail


class _StaleEvidence(_RuntimeFailure):
    def __init__(self, detail: str) -> None:
        super().__init__("stale_proposal", detail)


class _ResourceLimit(_RuntimeFailure):
    def __init__(self, name: str) -> None:
        super().__init__("resource_limit", name)


class _ValidMethodFailure(_RuntimeFailure):
    pass


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize a finite JSON value canonically for later source-bound sealing."""

    try:
        return (
            json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise CoordinateRDOError(f"value is not finite canonical JSON: {exc}") from exc


def _seal_record(record: Mapping[str, Any], field_name: str) -> dict[str, Any]:
    result = dict(record)
    result.pop(field_name, None)
    result[field_name] = hashlib.sha256(canonical_json_bytes(result)).hexdigest()
    return result


def _validate_digest(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise CoordinateRDOError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _validate_rgb_state(value: Any, name: str) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise CoordinateRDOError(f"{name} must be a NumPy array")
    if value.dtype != np.uint8 or value.ndim != 2 or value.shape[0] <= 0 or value.shape[1] != 3:
        raise CoordinateRDOError(f"{name} must have dtype uint8 and shape [N,3] with N>0")
    return np.ascontiguousarray(value).copy()


def _readonly_copy(value: np.ndarray) -> np.ndarray:
    result = np.ascontiguousarray(value).copy()
    result.setflags(write=False)
    return result


def state_digest(state: np.ndarray) -> str:
    """Hash an unambiguous uint8[N,3] state without trusting caller metadata."""

    checked = _validate_rgb_state(state, "state")
    digest = hashlib.sha256()
    digest.update(STATE_DIGEST_DOMAIN)
    digest.update(int(checked.shape[0]).to_bytes(8, "big", signed=False))
    digest.update(checked.tobytes(order="C"))
    return digest.hexdigest()


def _finite_objective(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise CoordinateRDOError(f"{name} must be a real scalar")
    result = float(value)
    if not math.isfinite(result):
        raise CoordinateRDOError(f"{name} must be finite")
    if result < 0.0:
        raise CoordinateRDOError(f"{name} must be nonnegative")
    # Canonicalize negative zero so an SSE cannot acquire a path-dependent sign bit.
    return 0.0 if result == 0.0 else result


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise CoordinateRDOError(f"{name} must be a positive integer")
    return value


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CoordinateRDOError(f"{name} must be a nonnegative integer")
    return value


def _normalize_components(value: Any, complete_bytes: int, name: str) -> dict[str, int]:
    if not isinstance(value, Mapping):
        raise CoordinateRDOError(f"{name} must be an SSP2F component-byte mapping")
    if set(value) != set(SSP2F_COMPONENT_KEYS):
        raise CoordinateRDOError(f"{name} must contain exactly the frozen SSP2F component keys")
    if tuple(value) != SSP2F_COMPONENT_KEYS:
        raise CoordinateRDOError(f"{name} keys must use frozen SSP2F component order")
    result: dict[str, int] = {}
    for key in SSP2F_COMPONENT_KEYS:
        component_bytes = value[key]
        result[key] = (
            _nonnegative_int(component_bytes, f"{name}.{key}")
            if key == "total"
            else _positive_int(component_bytes, f"{name}.{key}")
        )
    if result["prefix"] != 280:
        raise CoordinateRDOError(f"{name}.prefix must equal the frozen 280-byte prefix")
    if result["total"] != complete_bytes:
        raise CoordinateRDOError(f"{name}.total must equal complete_bytes")
    if sum(result[key] for key in SSP2F_COMPONENT_KEYS if key != "total") != complete_bytes:
        raise CoordinateRDOError(f"{name} non-total components must sum to complete_bytes")
    return result


def _objective_record(value: float) -> dict[str, Any]:
    return {"value": value, "float_hex": value.hex()}


def strictly_improves(candidate: float, current: float) -> bool:
    """Apply the exact COMP-012 numerical-hygiene acceptance predicate."""

    candidate_value = _finite_objective(candidate, "candidate objective")
    current_value = _finite_objective(current, "current objective")
    return candidate_value < current_value - OBJECTIVE_RELATIVE_TOLERANCE * max(
        1.0, abs(current_value)
    )


def uniform_lattice(step: int) -> tuple[int, ...]:
    """Construct ``sorted(unique({0, step, ...} union {255}))`` exactly."""

    frozen_step = _positive_int(step, "uniform lattice step")
    if frozen_step > 255:
        raise CoordinateRDOError("uniform lattice step cannot exceed 255")
    return tuple(sorted({*range(0, 256, frozen_step), 255}))


def _validate_lattice(lattice: Sequence[int]) -> tuple[int, ...]:
    if isinstance(lattice, (str, bytes)) or not isinstance(lattice, Sequence):
        raise CoordinateRDOError("lattice must be a sequence")
    values = tuple(lattice)
    if not values or values[0] != 0 or values[-1] != 255:
        raise CoordinateRDOError("lattice must contain endpoints 0 and 255")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0 or value > 255
        for value in values
    ):
        raise CoordinateRDOError("lattice values must be uint8-domain integers")
    if tuple(sorted(set(values))) != values:
        raise CoordinateRDOError("lattice values must be strictly increasing and unique")
    return values


def map_to_lattice(source_rgb: np.ndarray, lattice: Sequence[int]) -> np.ndarray:
    """Map each scalar to its nearest lattice value, breaking exact ties downward."""

    source = _validate_rgb_state(source_rgb, "source_rgb")
    values = _validate_lattice(lattice)
    lattice_array = np.asarray(values, dtype=np.int16)
    flat = source.reshape(-1).astype(np.int16)
    upper_index = np.searchsorted(lattice_array, flat, side="left")
    upper_index = np.minimum(upper_index, len(lattice_array) - 1)
    lower_index = np.maximum(upper_index - 1, 0)
    lower = lattice_array[lower_index]
    upper = lattice_array[upper_index]
    choose_lower = flat - lower <= upper - flat
    mapped = np.where(choose_lower, lower, upper).astype(np.uint8)
    return mapped.reshape(source.shape)


def independent_step3_map(source_rgb: np.ndarray) -> np.ndarray:
    """Construct the independent COMP-012 step-3 start from source symbols."""

    return map_to_lattice(source_rgb, L3)


ArmStartMode = Literal["coarse", "direct_fine", "nested_fine"]


def make_arm_start(
    source_rgb: np.ndarray,
    *,
    mode: ArmStartMode,
    coarse_terminal: np.ndarray | None = None,
) -> np.ndarray:
    """Make direct/coarse independent starts or a detached nested-fine start."""

    source = _validate_rgb_state(source_rgb, "source_rgb")
    if mode in {"coarse", "direct_fine"}:
        if coarse_terminal is not None:
            raise CoordinateRDOError(f"{mode} start cannot consume a coarse terminal")
        return independent_step3_map(source)
    if mode != "nested_fine":
        raise CoordinateRDOError(f"unknown arm start mode {mode!r}")
    if coarse_terminal is None:
        raise CoordinateRDOError("nested_fine start requires a coarse terminal")
    terminal = _validate_rgb_state(coarse_terminal, "coarse_terminal")
    if terminal.shape != source.shape:
        raise CoordinateRDOError("coarse terminal shape differs from source")
    if not np.isin(terminal, np.asarray(L3, dtype=np.uint8)).all():
        raise CoordinateRDOError("coarse terminal contains a value outside L3")
    return terminal


@dataclass(frozen=True)
class UniformLatticeState:
    step: int
    state: np.ndarray = field(repr=False, compare=False)
    state_digest: str


@dataclass(frozen=True)
class UniformLatticeEvaluation:
    step: int
    state_digest: str
    objective: float
    complete_bytes: int
    blob_sha256: str


def build_uniform_lattice_states(
    source_rgb: np.ndarray,
    *,
    steps: Sequence[int] = FROZEN_UNIFORM_STEPS,
) -> tuple[UniformLatticeState, ...]:
    """Construct detached target-blind uniform-lattice states in declared step order."""

    source = _validate_rgb_state(source_rgb, "source_rgb")
    frozen_steps = tuple(steps)
    if frozen_steps != FROZEN_UNIFORM_STEPS:
        raise CoordinateRDOError("uniform lattice construction requires the frozen step menu")
    result: list[UniformLatticeState] = []
    for step in frozen_steps:
        mapped = map_to_lattice(source, uniform_lattice(step))
        mapped.setflags(write=False)
        result.append(
            UniformLatticeState(step=step, state=mapped, state_digest=state_digest(mapped))
        )
    return tuple(result)


def select_uniform_lattice(
    evaluations: Sequence[UniformLatticeEvaluation],
    *,
    cap_bytes: int,
    steps: Sequence[int] = FROZEN_UNIFORM_STEPS,
) -> UniformLatticeEvaluation | None:
    """Select by source objective, bytes, numeric step, then ordinary blob digest."""

    cap = _positive_int(cap_bytes, "cap_bytes")
    required_steps = tuple(steps)
    if required_steps != FROZEN_UNIFORM_STEPS:
        raise CoordinateRDOError("uniform lattice selection requires the frozen step menu")
    if not isinstance(evaluations, Sequence) or isinstance(evaluations, (str, bytes)):
        raise CoordinateRDOError("uniform lattice evaluations must be a sequence")
    by_step: dict[int, UniformLatticeEvaluation] = {}
    normalized: list[UniformLatticeEvaluation] = []
    for index, evaluation in enumerate(evaluations):
        if not isinstance(evaluation, UniformLatticeEvaluation):
            raise CoordinateRDOError(f"uniform evaluation {index} has the wrong type")
        if evaluation.step not in required_steps or evaluation.step in by_step:
            raise CoordinateRDOError("uniform evaluations must contain every frozen step once")
        normalized_evaluation = UniformLatticeEvaluation(
            step=evaluation.step,
            state_digest=_validate_digest(
                evaluation.state_digest, f"uniform evaluation {evaluation.step} state digest"
            ),
            objective=_finite_objective(
                evaluation.objective, f"uniform evaluation {evaluation.step} objective"
            ),
            complete_bytes=_positive_int(
                evaluation.complete_bytes,
                f"uniform evaluation {evaluation.step} complete_bytes",
            ),
            blob_sha256=_validate_digest(
                evaluation.blob_sha256, f"uniform evaluation {evaluation.step} blob digest"
            ),
        )
        by_step[evaluation.step] = normalized_evaluation
        normalized.append(normalized_evaluation)
    if set(by_step) != set(required_steps) or len(normalized) != len(required_steps):
        raise CoordinateRDOError("uniform evaluations must contain the exact frozen step menu")
    feasible = [evaluation for evaluation in normalized if evaluation.complete_bytes <= cap]
    if not feasible:
        return None
    return min(
        feasible,
        key=lambda evaluation: (
            evaluation.objective,
            evaluation.complete_bytes,
            evaluation.step,
            evaluation.blob_sha256,
        ),
    )


@dataclass(frozen=True)
class ObjectiveProposalRequest:
    epoch: int
    origin_state_digest: str
    row: int
    channel: int
    current_value: int
    replacement_value: int


@dataclass(frozen=True)
class ObjectiveProposal:
    epoch: int
    origin_state_digest: str
    candidate_state_digest: str
    row: int
    channel: int
    current_value: int
    replacement_value: int
    objective: float
    margin_error: float
    commit_token: Any = field(repr=False, compare=False)


@dataclass(frozen=True)
class RateProposalRequest:
    epoch: int
    origin_state_digest: str
    candidate_state_digest: str
    row: int
    channel: int
    current_value: int
    replacement_value: int
    objective: float
    objective_commit_token: Any = field(repr=False, compare=False)


@dataclass(frozen=True)
class RateProposal:
    epoch: int
    origin_state_digest: str
    candidate_state_digest: str
    row: int
    channel: int
    current_value: int
    replacement_value: int
    complete_bytes: int
    component_bytes: Mapping[str, int]
    commit_token: Any = field(repr=False, compare=False)


MaterializationPurpose = Literal["initial", "sweep_end", "audit_jump", "terminal"]


@dataclass(frozen=True)
class MaterializationRequest:
    purpose: MaterializationPurpose
    epoch: int
    origin_state_digest: str | None
    state_digest: str
    expected_objective: float | None
    expected_complete_bytes: int | None
    state: np.ndarray = field(repr=False, compare=False)


@dataclass(frozen=True)
class MaterializedState:
    epoch: int
    state_digest: str
    objective: float
    complete_bytes: int
    component_bytes: Mapping[str, int]
    blob_sha256: str
    objective_abs_error: float
    objective_tolerance: float
    objective_reconciliation_passed: bool


@dataclass(frozen=True)
class WinnerCommitRequest:
    """One objective-plus-rate winner delivered once to the paired coordinator."""

    origin_epoch: int
    accepted_epoch: int
    origin_state_digest: str
    candidate_state_digest: str
    row: int
    channel: int
    current_value: int
    replacement_value: int
    objective: float
    complete_bytes: int
    component_bytes: Mapping[str, int]
    objective_commit_token: Any = field(repr=False, compare=False)
    rate_commit_token: Any = field(repr=False, compare=False)
    candidate_state: np.ndarray = field(repr=False, compare=False)


@dataclass(frozen=True)
class WinnerCommitReceipt:
    """Fail-closed epoch/digest acknowledgement from the commit coordinator."""

    origin_epoch: int
    accepted_epoch: int
    origin_state_digest: str
    candidate_state_digest: str
    objective: float
    complete_bytes: int
    component_bytes: Mapping[str, int]


class ExactRDOAdapter(Protocol):
    """Incremental-objective, exact-rate, and paired-commit coordinator.

    ``score`` and ``price`` must not mutate accepted state. ``price`` is called only for
    objectively improving candidates selected by the core's best-first objective schedule.
    ``commit`` must consume the objective and rate proposal tokens atomically and synchronize both
    incremental engines. A failed or partial commit poisons the run; the core never resumes.
    """

    def score(self, request: ObjectiveProposalRequest) -> ObjectiveProposal:
        """Score one scalar edit against the current accepted objective state."""

    def price(self, request: RateProposalRequest) -> RateProposal:
        """Return exact complete SSP2F bytes for one improving scalar edit."""

    def materialize(self, request: MaterializationRequest) -> MaterializedState:
        """Full-render and full-encode a checkpoint state without changing accepted state."""

    def commit(self, request: WinnerCommitRequest) -> WinnerCommitReceipt:
        """Atomically commit exactly one paired objective-plus-rate winner."""


@dataclass(frozen=True)
class SearchLimits:
    """Explicit caller-frozen resource ceilings; this module supplies no scientific defaults."""

    max_sweeps: int
    max_proposals: int
    max_rate_prices: int
    max_accepted_edits: int
    max_materializations: int

    def __post_init__(self) -> None:
        for name in (
            "max_sweeps",
            "max_proposals",
            "max_rate_prices",
            "max_accepted_edits",
            "max_materializations",
        ):
            _nonnegative_int(getattr(self, name), name)

    def as_record(self) -> dict[str, int]:
        return {
            "max_sweeps": self.max_sweeps,
            "max_proposals": self.max_proposals,
            "max_rate_prices": self.max_rate_prices,
            "max_accepted_edits": self.max_accepted_edits,
            "max_materializations": self.max_materializations,
        }


@dataclass(frozen=True)
class SearchResult:
    status: Literal[
        "core_terminal", "valid_method_failure", "valid_nonconvergence", "invalid_no_decision"
    ]
    failure_reason: str | None
    state: np.ndarray = field(repr=False, compare=False)
    ledger: Mapping[str, Any]

    @property
    def promotable(self) -> bool:
        """The search core never has lifecycle authority to promote an experiment."""

        return False

    @property
    def core_terminal(self) -> bool:
        return self.status == "core_terminal"


def _alphabet_record(name: Literal["L1", "L3"]) -> tuple[tuple[int, ...], dict[str, Any]]:
    if name == "L1":
        values = L1
    elif name == "L3":
        values = L3
    else:
        raise CoordinateRDOError("alphabet must be exactly 'L1' or 'L3'")
    payload = canonical_json_bytes(list(values))
    return values, {
        "name": name,
        "values": list(values),
        "values_sha256": hashlib.sha256(payload).hexdigest(),
    }


def _local_neighbor_values(
    value: int,
    *,
    alphabet_name: Literal["L1", "L3"],
    alphabet: tuple[int, ...],
) -> tuple[int, ...]:
    if value not in alphabet:
        raise CoordinateRDOError(f"state value {value} is outside the active alphabet")
    if alphabet_name == "L1":
        return tuple(
            sorted(
                candidate
                for candidate in {value - 3, value - 1, value + 1, value + 3}
                if 0 <= candidate <= 255
            )
        )
    index = alphabet.index(value)
    result = []
    if index > 0:
        result.append(alphabet[index - 1])
    if index + 1 < len(alphabet):
        result.append(alphabet[index + 1])
    return tuple(result)


def _full_label_values(value: int, alphabet: tuple[int, ...]) -> tuple[int, ...]:
    if value not in alphabet:
        raise CoordinateRDOError(f"state value {value} is outside the active alphabet")
    return tuple(candidate for candidate in alphabet if candidate != value)


def _validate_objective_proposal(
    response: Any,
    *,
    request: ObjectiveProposalRequest,
) -> tuple[str, float, float, Any]:
    if not isinstance(response, ObjectiveProposal):
        raise _RuntimeFailure("invalid_proposal", "score callback returned the wrong type")
    if (
        response.epoch != request.epoch
        or response.origin_state_digest != request.origin_state_digest
        or response.row != request.row
        or response.channel != request.channel
        or response.current_value != request.current_value
        or response.replacement_value != request.replacement_value
    ):
        raise _StaleEvidence("score callback returned stale proposal coordinates or origin state")
    try:
        candidate_digest = _validate_digest(
            response.candidate_state_digest, "objective candidate state digest"
        )
        objective = _finite_objective(response.objective, "objective proposal")
        margin_error = _finite_objective(response.margin_error, "objective proposal margin_error")
    except CoordinateRDOError as exc:
        raise _RuntimeFailure("invalid_proposal", str(exc)) from exc
    return candidate_digest, objective, margin_error, response.commit_token


def _validate_rate_proposal(
    response: Any,
    *,
    request: RateProposalRequest,
) -> tuple[int, dict[str, int], Any]:
    if not isinstance(response, RateProposal):
        raise _RuntimeFailure("invalid_proposal", "price callback returned the wrong type")
    if (
        response.epoch != request.epoch
        or response.origin_state_digest != request.origin_state_digest
        or response.candidate_state_digest != request.candidate_state_digest
        or response.row != request.row
        or response.channel != request.channel
        or response.current_value != request.current_value
        or response.replacement_value != request.replacement_value
    ):
        raise _StaleEvidence("price callback returned stale proposal coordinates or state")
    try:
        complete_bytes = _positive_int(response.complete_bytes, "rate proposal complete_bytes")
        components = _normalize_components(
            response.component_bytes, complete_bytes, "rate proposal component_bytes"
        )
    except CoordinateRDOError as exc:
        raise _RuntimeFailure("invalid_proposal", str(exc)) from exc
    return complete_bytes, components, response.commit_token


def _validate_materialization(
    response: Any,
    *,
    request: MaterializationRequest,
) -> tuple[float, int, dict[str, int], str, dict[str, Any]]:
    if not isinstance(response, MaterializedState):
        raise _RuntimeFailure("invalid_proposal", "materialize callback returned the wrong type")
    if response.epoch != request.epoch or response.state_digest != request.state_digest:
        raise _StaleEvidence("materialize callback returned a stale epoch or state digest")
    try:
        objective = _finite_objective(response.objective, "materialized objective")
        complete_bytes = _positive_int(response.complete_bytes, "materialized complete_bytes")
        components = _normalize_components(
            response.component_bytes, complete_bytes, "materialized component_bytes"
        )
        blob_digest = _validate_digest(response.blob_sha256, "materialized blob digest")
        objective_error = _finite_objective(
            response.objective_abs_error, "materialized objective_abs_error"
        )
        objective_tolerance = _finite_objective(
            response.objective_tolerance, "materialized objective_tolerance"
        )
    except CoordinateRDOError as exc:
        raise _RuntimeFailure("invalid_proposal", str(exc)) from exc
    if not isinstance(response.objective_reconciliation_passed, bool):
        raise _RuntimeFailure(
            "invalid_proposal", "checkpoint reconciliation pass flag must be boolean"
        )
    computed_error = (
        0.0 if request.expected_objective is None else abs(objective - request.expected_objective)
    )
    if computed_error.hex() != objective_error.hex():
        raise _RuntimeFailure(
            "invalid_proposal", "checkpoint objective error differs from recomputation"
        )
    if not response.objective_reconciliation_passed or objective_error > objective_tolerance:
        raise _RuntimeFailure(
            "objective_reconciliation_failed",
            "checkpoint objective reconciliation did not pass its certified tolerance",
        )
    if (
        request.expected_complete_bytes is not None
        and complete_bytes != request.expected_complete_bytes
    ):
        raise _RuntimeFailure(
            "invalid_proposal", "materialized bytes differ from exact proposal price"
        )
    reconciliation = {
        "tracked_objective": (
            _objective_record(request.expected_objective)
            if request.expected_objective is not None
            else None
        ),
        "full_objective": _objective_record(objective),
        "objective_abs_error": _objective_record(objective_error),
        "objective_tolerance": _objective_record(objective_tolerance),
        "passed": True,
        "rebased": request.expected_objective is not None,
    }
    return objective, complete_bytes, components, blob_digest, reconciliation


def _validate_commit(
    response: Any,
    *,
    request: WinnerCommitRequest,
) -> None:
    if not isinstance(response, WinnerCommitReceipt):
        raise _RuntimeFailure("invalid_proposal", "commit callback returned the wrong type")
    if (
        response.origin_epoch != request.origin_epoch
        or response.accepted_epoch != request.accepted_epoch
        or response.origin_state_digest != request.origin_state_digest
        or response.candidate_state_digest != request.candidate_state_digest
    ):
        raise _StaleEvidence("commit callback returned a stale epoch or state digest")
    try:
        objective = _finite_objective(response.objective, "commit objective")
        complete_bytes = _positive_int(response.complete_bytes, "commit complete_bytes")
        components = _normalize_components(
            response.component_bytes, complete_bytes, "commit component_bytes"
        )
    except CoordinateRDOError as exc:
        raise _RuntimeFailure("invalid_proposal", str(exc)) from exc
    if objective.hex() != request.objective.hex():
        raise _RuntimeFailure("invalid_proposal", "commit objective differs from paired winner")
    if complete_bytes != request.complete_bytes or components != dict(request.component_bytes):
        raise _RuntimeFailure(
            "invalid_proposal", "commit bytes/components differ from the paired winner"
        )


def _checkpoint_record(
    *,
    purpose: MaterializationPurpose,
    epoch: int,
    state_digest_value: str,
    objective: float,
    complete_bytes: int,
    component_bytes: Mapping[str, int],
    blob_sha256: str,
    objective_reconciliation: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "purpose": purpose,
        "epoch": epoch,
        "state_digest": state_digest_value,
        "objective": _objective_record(objective),
        "complete_bytes": complete_bytes,
        "component_bytes": dict(component_bytes),
        "blob_sha256": blob_sha256,
        "objective_reconciliation": dict(objective_reconciliation),
    }


_PASS_KIND_CODES = {"local_sweep": 0, "full_label_audit": 1}
_PROPOSAL_DISPOSITION_CODES = {
    "not_strictly_improving_unpriced": 0,
    "infeasible_cap": 1,
    "dominated_by_feasible_better_objective_unpriced": 2,
    "dominated_by_rate_or_value": 3,
    "ambiguous_feasible_noncommittable": 4,
    "accepted": 5,
    "aborted_incomplete_coordinate": 6,
}
_TRANSCRIPT_REQUIRED_FIELDS = {
    "pass_index",
    "pass_kind",
    "row",
    "channel",
    "origin_epoch",
    "origin_state_digest",
    "current_value",
    "replacement_value",
    "objective_float_hex",
    "objective_margin_float_hex",
    "margin_error_float_hex",
    "common_coordinate_epsilon_float_hex",
    "objective_class",
    "rate_priced",
    "rate_query_index",
    "complete_bytes",
    "component_record_sha256",
    "byte_margin",
    "rate_query_class",
    "feasibility_class",
    "candidate_state_digest",
    "disposition",
}


class ProposalTranscript:
    """Bounded rolling evidence for every accepted and rejected scalar proposal.

    Raw proposal rows and completed chunk digests are hashed and immediately discarded. Memory is
    constant in proposal count: two rolling hash states plus disposition and closest-margin
    summaries.
    """

    def __init__(self, *, chunk_size: int = PROPOSAL_TRANSCRIPT_CHUNK_SIZE) -> None:
        self._chunk_size = _positive_int(chunk_size, "proposal transcript chunk_size")
        self._aggregate = hashlib.sha256(PROPOSAL_TRANSCRIPT_DOMAIN)
        self._chunk_hasher = self._new_chunk_hasher(0)
        self._chunk_count = 0
        self._chunk_aggregate = hashlib.sha256(PROPOSAL_TRANSCRIPT_DOMAIN + b"chunk-aggregate\x00")
        self._completed_chunks = 0
        self._count = 0
        self._disposition_counts = {name: 0 for name in _PROPOSAL_DISPOSITION_CODES}
        self._closest_objective: tuple[float, int, dict[str, Any]] | None = None
        self._closest_rate: tuple[int, int, dict[str, Any]] | None = None
        self._finished: dict[str, Any] | None = None

    @staticmethod
    def _framed_update(hasher: Any, payload: bytes) -> None:
        hasher.update(len(payload).to_bytes(8, "big", signed=False))
        hasher.update(payload)

    @staticmethod
    def _new_chunk_hasher(index: int) -> Any:
        digest = hashlib.sha256(PROPOSAL_TRANSCRIPT_DOMAIN + b"chunk\x00")
        digest.update(index.to_bytes(8, "big", signed=False))
        return digest

    @staticmethod
    def _margin_identity(record: Mapping[str, Any], sequence: int) -> dict[str, Any]:
        return {
            "sequence": sequence,
            "pass_index": record["pass_index"],
            "pass_kind": record["pass_kind"],
            "row": record["row"],
            "channel": record["channel"],
            "origin_epoch": record["origin_epoch"],
            "replacement_value": record["replacement_value"],
        }

    def add(self, record: Mapping[str, Any]) -> None:
        if self._finished is not None:
            raise RuntimeError("cannot append to a finished proposal transcript")
        if set(record) != _TRANSCRIPT_REQUIRED_FIELDS:
            raise RuntimeError("internal proposal transcript field-set drift")
        disposition = record["disposition"]
        if disposition not in _PROPOSAL_DISPOSITION_CODES:
            raise RuntimeError("internal proposal disposition drift")
        candidate_digest = _validate_digest(
            record["candidate_state_digest"], "transcript candidate state digest"
        )
        origin_digest = _validate_digest(
            record["origin_state_digest"], "transcript origin state digest"
        )
        objective_margin = float.fromhex(record["objective_margin_float_hex"])
        if not math.isfinite(objective_margin):
            raise RuntimeError("internal transcript objective margin is non-finite")
        margin_error = float.fromhex(record["margin_error_float_hex"])
        common_epsilon = float.fromhex(record["common_coordinate_epsilon_float_hex"])
        if (
            not math.isfinite(margin_error)
            or margin_error < 0.0
            or not math.isfinite(common_epsilon)
            or common_epsilon < margin_error
            or record["objective_class"]
            not in ("definitely_improving", "ambiguous", "nonimproving")
        ):
            raise RuntimeError("internal transcript objective classification is invalid")
        rate_priced = record["rate_priced"]
        if not isinstance(rate_priced, bool):
            raise RuntimeError("internal transcript rate_priced must be boolean")
        if rate_priced:
            if (
                not isinstance(record["complete_bytes"], int)
                or not isinstance(record["byte_margin"], int)
                or not isinstance(record["rate_query_index"], int)
            ):
                raise RuntimeError("priced transcript row lacks exact integer rate evidence")
            component_digest = _validate_digest(
                record["component_record_sha256"],
                "transcript component record digest",
            )
            if record["rate_query_class"] != "exact_priced" or record["feasibility_class"] not in (
                "feasible",
                "infeasible",
            ):
                raise RuntimeError("priced transcript row has invalid rate classes")
        elif (
            record["complete_bytes"] is not None
            or record["byte_margin"] is not None
            or record["rate_query_index"] is not None
            or record["component_record_sha256"] is not None
        ):
            raise RuntimeError("unpriced transcript row contains fabricated rate evidence")
        else:
            component_digest = None
            if (
                record["rate_query_class"]
                not in (
                    "definitely_nonimproving_unpriced",
                    "dominance_pruned_unpriced",
                    "incomplete_unpriced",
                )
                or record["feasibility_class"] != "not_queried"
            ):
                raise RuntimeError("unpriced transcript row has invalid rate classes")

        sequence = self._count
        normalized = dict(record)
        normalized["candidate_state_digest"] = candidate_digest
        normalized["origin_state_digest"] = origin_digest
        normalized["component_record_sha256"] = component_digest
        normalized["disposition_code"] = _PROPOSAL_DISPOSITION_CODES[disposition]
        normalized["pass_kind_code"] = _PASS_KIND_CODES[record["pass_kind"]]
        normalized["sequence"] = sequence
        payload = canonical_json_bytes(normalized)
        self._framed_update(self._aggregate, payload)
        self._framed_update(self._chunk_hasher, payload)
        self._count += 1
        self._chunk_count += 1
        self._disposition_counts[disposition] += 1

        identity = self._margin_identity(record, sequence)
        objective_key = (abs(objective_margin), sequence)
        objective_summary = {
            **identity,
            "objective_margin_float_hex": objective_margin.hex(),
            "margin_error_float_hex": margin_error.hex(),
            "common_coordinate_epsilon_float_hex": common_epsilon.hex(),
            "objective_class": record["objective_class"],
        }
        if self._closest_objective is None or objective_key < self._closest_objective[:2]:
            self._closest_objective = (*objective_key, objective_summary)
        if rate_priced:
            byte_margin = int(record["byte_margin"])
            rate_key = (abs(byte_margin), sequence)
            rate_summary = {**identity, "byte_margin": byte_margin}
            if self._closest_rate is None or rate_key < self._closest_rate[:2]:
                self._closest_rate = (*rate_key, rate_summary)

        if self._chunk_count == self._chunk_size:
            self._flush()

    def _flush(self) -> None:
        if self._chunk_count == 0:
            return
        chunk_record = {
            "chunk_index": self._completed_chunks,
            "first_sequence": self._count - self._chunk_count,
            "record_count": self._chunk_count,
            "records_sha256": self._chunk_hasher.hexdigest(),
        }
        self._framed_update(self._chunk_aggregate, canonical_json_bytes(chunk_record))
        self._completed_chunks += 1
        self._chunk_count = 0
        self._chunk_hasher = self._new_chunk_hasher(self._completed_chunks)

    @property
    def count(self) -> int:
        return self._count

    def finish(self) -> dict[str, Any]:
        if self._finished is not None:
            return dict(self._finished)
        self._flush()
        record = {
            "schema": "structsplat.comp012.proposal-transcript.v3",
            "ordered_record_encoding": "length-prefixed canonical JSON",
            "chunk_capacity": self._chunk_size,
            "proposal_count": self._count,
            "ordered_proposals_sha256": self._aggregate.hexdigest(),
            "chunk_count": self._completed_chunks,
            "aggregate_chunks_sha256": self._chunk_aggregate.hexdigest(),
            "disposition_codes": dict(_PROPOSAL_DISPOSITION_CODES),
            "disposition_counts": dict(self._disposition_counts),
            "closest_objective_margin": (
                self._closest_objective[2] if self._closest_objective is not None else None
            ),
            "closest_rate_margin": (
                self._closest_rate[2] if self._closest_rate is not None else None
            ),
        }
        self._finished = _seal_record(record, "proposal_transcript_sha256")
        return dict(self._finished)


def _visited_state_summary(ledger: Mapping[str, Any]) -> tuple[int, str]:
    """Seal the bounded accepted-state sequence without persisting a duplicate list."""

    hasher = hashlib.sha256(VISITED_STATE_TRANSCRIPT_DOMAIN)
    count = 0
    for digest_value in (
        ledger["initial_state_digest"],
        *(edit["state_digest"] for edit in ledger["accepted_edits"]),
    ):
        digest = _validate_digest(digest_value, "visited state digest")
        payload = digest.encode("ascii")
        hasher.update(len(payload).to_bytes(8, "big", signed=False))
        hasher.update(payload)
        count += 1
    return count, hasher.hexdigest()


def _finalize_outcome(
    *,
    ledger: dict[str, Any],
    state: np.ndarray,
    status: Literal["valid_method_failure", "valid_nonconvergence", "invalid_no_decision"],
    reason: str,
    detail: str,
    counters: Mapping[str, int],
    active_pass: dict[str, Any] | None,
    active_query: Mapping[str, Any] | None,
    transcript: ProposalTranscript,
) -> SearchResult:
    if active_pass is not None and "pass_sha256" not in active_pass:
        active_pass["status"] = status
        active_pass["complete_full_traversal"] = False
        active_pass["failure_reason"] = reason
        active_pass["failure_detail"] = detail
        sealed_pass = _seal_record(active_pass, "pass_sha256")
        active_pass.clear()
        active_pass.update(sealed_pass)
    ledger["status"] = status
    ledger["failure_reason"] = reason
    ledger["failure_detail"] = detail
    ledger["failure_query"] = dict(active_query) if active_query is not None else None
    ledger["promotable"] = False
    ledger["promotion_authority"] = "lifecycle_reducer_only"
    ledger["terminal_certificate"] = None
    ledger["proposal_transcript"] = transcript.finish()
    (
        ledger["visited_state_count"],
        ledger["visited_state_digests_sha256"],
    ) = _visited_state_summary(ledger)
    ledger["counters"] = dict(counters)
    sealed = _seal_record(ledger, "ledger_sha256")
    return SearchResult(
        status=status,
        failure_reason=reason,
        state=state.copy(),
        ledger=sealed,
    )


def run_coordinate_search(
    initial_state: np.ndarray,
    *,
    alphabet: Literal["L1", "L3"],
    cap_bytes: int,
    adapter: ExactRDOAdapter,
    limits: SearchLimits,
) -> SearchResult:
    """Run objective-first Gauss-Seidel search plus exhaustive full-label audits."""

    state = _validate_rgb_state(initial_state, "initial_state")
    alphabet_values, alphabet_metadata = _alphabet_record(alphabet)
    cap = _positive_int(cap_bytes, "cap_bytes")
    if any(
        not callable(getattr(adapter, name, None))
        for name in ("score", "price", "materialize", "commit")
    ):
        raise CoordinateRDOError(
            "adapter must implement callable score/price/materialize/commit methods"
        )
    if not isinstance(limits, SearchLimits):
        raise CoordinateRDOError("limits must be a SearchLimits record")
    if not np.isin(state, np.asarray(alphabet_values, dtype=np.uint8)).all():
        raise CoordinateRDOError("initial_state contains a value outside the active alphabet")

    current_digest = state_digest(state)
    epoch = 0
    counters = {
        "sweeps": 0,
        "full_label_audits": 0,
        "passes": 0,
        "coordinate_visits": 0,
        "proposals": 0,
        "rate_prices": 0,
        "accepted_edits": 0,
        "objective_definitely_improving": 0,
        "objective_ambiguous": 0,
        "objective_nonimproving": 0,
        "dominance_pruned": 0,
        "feasible_ambiguous_levels": 0,
        "materializations": 0,
        "commit_attempts": 0,
        "commits": 0,
    }
    ledger: dict[str, Any] = {
        "schema": LEDGER_SCHEMA,
        "shape": [int(state.shape[0]), 3],
        "dtype": "uint8",
        "traversal": "local_sweeps_then_full_label_audit_row_ascending_then_R_G_B",
        "local_neighborhood": {
            "L3": "immediate_predecessor_successor",
            "L1": "sorted_union_of_in_range_v_minus_3_v_minus_1_v_plus_1_v_plus_3",
        },
        "terminal_neighborhood": "every_other_active_alphabet_value_ascending",
        "alphabet": alphabet_metadata,
        "cap_bytes": cap,
        "objective_acceptance": {
            "relative_tolerance": OBJECTIVE_RELATIVE_TOLERANCE,
            "formula": "candidate < current - 1e-12 * max(1, abs(current))",
            "objective_domain": "finite_nonnegative",
        },
        "proposal_schedule": (
            "score_all_labels; common_coordinate_epsilon=max_margin_error; "
            "price_definite_and_ambiguous_levels_by_exact_objective"
        ),
        "rate_pruning": (
            "nonimprovers_unpriced; after_first_feasible_definite_or_ambiguous_level "
            "later_levels_soundly_unpriced"
        ),
        "ambiguous_policy": "exact_rate_price_but_never_commit",
        "tie_order": [
            "objective",
            "complete_bytes",
            "replacement_value",
        ],
        "accepted_winner_commit": (
            "exactly_once_paired_objective_and_rate_tokens_before_core_state_mutation"
        ),
        "checkpoint_schedule": ["initial", "sweep_end", "audit_jump", "terminal"],
        "limits": limits.as_record(),
        "initial": None,
        "passes": [],
        "accepted_edits": [],
        "proposal_transcript": None,
        "initial_state_digest": current_digest,
        "visited_state_count": None,
        "visited_state_digests_sha256": None,
        "status": "running",
        "failure_reason": None,
        "failure_detail": None,
        "failure_query": None,
        "promotable": False,
        "promotion_authority": "lifecycle_reducer_only",
        "terminal_certificate": None,
    }
    transcript = ProposalTranscript()
    active_pass: dict[str, Any] | None = None
    active_query: dict[str, Any] | None = None
    pending_coordinate: list[dict[str, Any]] = []

    def consume(counter: str, maximum: int) -> None:
        if counters[counter] >= maximum:
            raise _ResourceLimit(f"max_{counter}")
        counters[counter] += 1

    def full_checkpoint(
        *,
        purpose: MaterializationPurpose,
        expected_objective: float | None,
        expected_bytes: int | None,
        expected_components: Mapping[str, int] | None,
        origin_digest: str | None,
    ) -> tuple[float, int, dict[str, int], str, dict[str, Any]]:
        nonlocal active_query
        consume("materializations", limits.max_materializations)
        active_query = {
            "phase": "checkpoint",
            "purpose": purpose,
            "epoch": epoch,
            "origin_state_digest": origin_digest,
            "state_digest": current_digest,
        }
        request = MaterializationRequest(
            purpose=purpose,
            epoch=epoch,
            origin_state_digest=origin_digest,
            state_digest=current_digest,
            expected_objective=expected_objective,
            expected_complete_bytes=expected_bytes,
            state=_readonly_copy(state),
        )
        try:
            response = adapter.materialize(request)
        except _RuntimeFailure:
            raise
        except MemoryError as exc:
            raise _RuntimeFailure("oom", "materialize callback raised MemoryError") from exc
        except Exception as exc:
            raise _RuntimeFailure(
                "invalid_proposal", f"materialize callback raised {type(exc).__name__}"
            ) from exc
        (
            objective_value,
            bytes_value,
            components,
            blob_digest,
            reconciliation,
        ) = _validate_materialization(response, request=request)
        if expected_components is not None and components != dict(expected_components):
            raise _RuntimeFailure(
                "invalid_proposal", f"{purpose} checkpoint components differ from tracked state"
            )
        active_query = None
        checkpoint_value = _checkpoint_record(
            purpose=purpose,
            epoch=epoch,
            state_digest_value=current_digest,
            objective=objective_value,
            complete_bytes=bytes_value,
            component_bytes=components,
            blob_sha256=blob_digest,
            objective_reconciliation=reconciliation,
        )
        return (
            objective_value,
            bytes_value,
            components,
            blob_digest,
            checkpoint_value,
        )

    def record_pending() -> None:
        nonlocal pending_coordinate
        if active_pass is None:
            raise RuntimeError("internal active pass is absent while recording proposals")
        for proposal in pending_coordinate:
            disposition = proposal.get("disposition")
            if disposition not in _PROPOSAL_DISPOSITION_CODES:
                raise RuntimeError("internal proposal lacks a terminal disposition")
            priced = proposal["rate_priced"]
            if priced:
                rate_query_class = "exact_priced"
                feasibility_class = "feasible" if proposal["bytes"] <= cap else "infeasible"
                component_record_sha256 = hashlib.sha256(
                    canonical_json_bytes(proposal["components"])
                ).hexdigest()
            else:
                feasibility_class = "not_queried"
                component_record_sha256 = None
                if proposal["objective_class"] == "nonimproving":
                    rate_query_class = "definitely_nonimproving_unpriced"
                elif disposition == "aborted_incomplete_coordinate":
                    rate_query_class = "incomplete_unpriced"
                else:
                    rate_query_class = "dominance_pruned_unpriced"
            transcript.add(
                {
                    "pass_index": proposal["pass_index"],
                    "pass_kind": proposal["pass_kind"],
                    "row": proposal["row"],
                    "channel": proposal["channel"],
                    "origin_epoch": proposal["origin_epoch"],
                    "origin_state_digest": proposal["origin_digest"],
                    "current_value": proposal["current_value"],
                    "replacement_value": proposal["replacement_value"],
                    "objective_float_hex": proposal["objective"].hex(),
                    "objective_margin_float_hex": proposal["objective_margin"].hex(),
                    "margin_error_float_hex": proposal["margin_error"].hex(),
                    "common_coordinate_epsilon_float_hex": proposal[
                        "common_coordinate_epsilon"
                    ].hex(),
                    "objective_class": proposal["objective_class"],
                    "rate_priced": priced,
                    "rate_query_index": proposal["rate_query_index"] if priced else None,
                    "complete_bytes": proposal["bytes"] if priced else None,
                    "component_record_sha256": component_record_sha256,
                    "byte_margin": cap - proposal["bytes"] if priced else None,
                    "rate_query_class": rate_query_class,
                    "feasibility_class": feasibility_class,
                    "candidate_state_digest": proposal["digest"],
                    "disposition": disposition,
                }
            )
            active_pass["disposition_counts"][disposition] += 1
        pending_coordinate = []

    def abort_pending() -> None:
        if pending_coordinate and any(
            proposal["objective_class"] is None for proposal in pending_coordinate
        ):
            common_epsilon = max(proposal["margin_error"] for proposal in pending_coordinate)
            for proposal in pending_coordinate:
                proposal["common_coordinate_epsilon"] = common_epsilon
                margin = proposal["objective_margin"]
                if margin > common_epsilon:
                    proposal["objective_class"] = "definitely_improving"
                elif margin < -common_epsilon:
                    proposal["objective_class"] = "nonimproving"
                else:
                    proposal["objective_class"] = "ambiguous"
                if active_pass is not None:
                    objective_class = proposal["objective_class"]
                    active_pass["objective_class_counts"][objective_class] += 1
                    counters[f"objective_{objective_class}"] += 1
        for proposal in pending_coordinate:
            proposal["disposition"] = "aborted_incomplete_coordinate"
        if pending_coordinate and active_pass is not None:
            record_pending()

    def seal_pass(
        *,
        status: str,
        complete_full_traversal: bool,
        checkpoint_value: Mapping[str, Any] | None,
        local_zero_certificate: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        nonlocal active_pass
        if active_pass is None:
            raise RuntimeError("internal pass state is absent")
        active_pass["status"] = status
        active_pass["end_epoch"] = epoch
        active_pass["end_state_digest"] = current_digest
        active_pass["end_objective"] = _objective_record(current_objective)
        active_pass["end_complete_bytes"] = current_bytes
        active_pass["complete_full_traversal"] = complete_full_traversal
        active_pass["checkpoint"] = dict(checkpoint_value) if checkpoint_value is not None else None
        active_pass["local_zero_certificate"] = (
            dict(local_zero_certificate) if local_zero_certificate is not None else None
        )
        sealed = _seal_record(active_pass, "pass_sha256")
        active_pass.clear()
        active_pass.update(sealed)
        result = active_pass
        active_pass = None
        return result

    def score_candidate(
        *,
        pass_index: int,
        pass_kind: Literal["local_sweep", "full_label_audit"],
        row: int,
        channel: int,
        origin_epoch: int,
        origin_digest: str,
        current_value: int,
        replacement_value: int,
        origin_objective: float,
    ) -> dict[str, Any]:
        nonlocal active_query
        active_query = {
            "phase": "objective_score",
            "pass_index": pass_index,
            "pass_kind": pass_kind,
            "row": row,
            "channel": channel,
            "epoch": origin_epoch,
            "origin_state_digest": origin_digest,
            "current_value": current_value,
            "replacement_value": replacement_value,
        }
        consume("proposals", limits.max_proposals)
        if active_pass is None:
            raise RuntimeError("internal active pass disappeared")
        active_pass["proposal_count"] += 1
        active_pass["objective_score_count"] += 1
        request = ObjectiveProposalRequest(
            epoch=origin_epoch,
            origin_state_digest=origin_digest,
            row=row,
            channel=channel,
            current_value=current_value,
            replacement_value=replacement_value,
        )
        try:
            response = adapter.score(request)
        except _RuntimeFailure:
            raise
        except MemoryError as exc:
            raise _RuntimeFailure("oom", "score callback raised MemoryError") from exc
        except Exception as exc:
            raise _RuntimeFailure(
                "invalid_proposal", f"score callback raised {type(exc).__name__}"
            ) from exc
        candidate_digest, proposal_objective, margin_error, objective_token = (
            _validate_objective_proposal(response, request=request)
        )
        boundary = origin_objective - OBJECTIVE_RELATIVE_TOLERANCE * max(1.0, abs(origin_objective))
        active_query = None
        return {
            "pass_index": pass_index,
            "pass_kind": pass_kind,
            "row": row,
            "channel": channel,
            "origin_epoch": origin_epoch,
            "origin_digest": origin_digest,
            "current_value": current_value,
            "replacement_value": replacement_value,
            "digest": candidate_digest,
            "objective": proposal_objective,
            "objective_margin": boundary - proposal_objective,
            "margin_error": margin_error,
            "common_coordinate_epsilon": None,
            "objective_class": None,
            "objective_token": objective_token,
            "rate_priced": False,
            "rate_query_index": None,
            "bytes": None,
            "components": None,
            "rate_token": None,
            "disposition": None,
        }

    def price_candidate(proposal: dict[str, Any]) -> None:
        nonlocal active_query
        active_query = {
            "phase": "exact_rate_price",
            "pass_index": proposal["pass_index"],
            "pass_kind": proposal["pass_kind"],
            "row": proposal["row"],
            "channel": proposal["channel"],
            "epoch": proposal["origin_epoch"],
            "origin_state_digest": proposal["origin_digest"],
            "candidate_state_digest": proposal["digest"],
            "current_value": proposal["current_value"],
            "replacement_value": proposal["replacement_value"],
        }
        consume("rate_prices", limits.max_rate_prices)
        query_index = counters["rate_prices"] - 1
        if active_pass is None:
            raise RuntimeError("internal active pass disappeared")
        active_pass["rate_price_count"] += 1
        request = RateProposalRequest(
            epoch=proposal["origin_epoch"],
            origin_state_digest=proposal["origin_digest"],
            candidate_state_digest=proposal["digest"],
            row=proposal["row"],
            channel=proposal["channel"],
            current_value=proposal["current_value"],
            replacement_value=proposal["replacement_value"],
            objective=proposal["objective"],
            objective_commit_token=proposal["objective_token"],
        )
        try:
            response = adapter.price(request)
        except _RuntimeFailure:
            raise
        except MemoryError as exc:
            raise _RuntimeFailure("oom", "price callback raised MemoryError") from exc
        except Exception as exc:
            raise _RuntimeFailure(
                "invalid_proposal", f"price callback raised {type(exc).__name__}"
            ) from exc
        complete_bytes, components, rate_token = _validate_rate_proposal(response, request=request)
        proposal["rate_priced"] = True
        proposal["rate_query_index"] = query_index
        proposal["bytes"] = complete_bytes
        proposal["components"] = components
        proposal["rate_token"] = rate_token
        active_query = None

    def visit_coordinate(
        *,
        pass_index: int,
        pass_kind: Literal["local_sweep", "full_label_audit"],
        row: int,
        channel: int,
        replacements: Sequence[int],
    ) -> bool:
        nonlocal active_query
        nonlocal current_bytes
        nonlocal current_components
        nonlocal current_digest
        nonlocal current_objective
        nonlocal epoch
        nonlocal pending_coordinate
        nonlocal state

        origin_digest = current_digest
        origin_epoch = epoch
        origin_objective = current_objective
        current_value = int(state[row, channel])
        pending_coordinate = []
        for replacement_value in replacements:
            pending_coordinate.append(
                score_candidate(
                    pass_index=pass_index,
                    pass_kind=pass_kind,
                    row=row,
                    channel=channel,
                    origin_epoch=origin_epoch,
                    origin_digest=origin_digest,
                    current_value=current_value,
                    replacement_value=replacement_value,
                    origin_objective=origin_objective,
                )
            )

        common_epsilon = max(proposal["margin_error"] for proposal in pending_coordinate)
        for proposal in pending_coordinate:
            proposal["common_coordinate_epsilon"] = common_epsilon
            margin = proposal["objective_margin"]
            if margin > common_epsilon:
                objective_class = "definitely_improving"
            elif margin < -common_epsilon:
                objective_class = "nonimproving"
            else:
                objective_class = "ambiguous"
            proposal["objective_class"] = objective_class

        rate_relevant = [
            proposal
            for proposal in pending_coordinate
            if proposal["objective_class"] != "nonimproving"
        ]
        if active_pass is None:
            raise RuntimeError("internal active pass disappeared")
        for objective_class in ("definitely_improving", "ambiguous", "nonimproving"):
            count = sum(
                proposal["objective_class"] == objective_class for proposal in pending_coordinate
            )
            active_pass["objective_class_counts"][objective_class] += count
            counters[f"objective_{objective_class}"] += count
        if not rate_relevant:
            for proposal in pending_coordinate:
                proposal["disposition"] = "not_strictly_improving_unpriced"
            record_pending()
            return False

        winner: dict[str, Any] | None = None
        feasible_ambiguous_level = False
        for objective_value in sorted({proposal["objective"] for proposal in rate_relevant}):
            level = sorted(
                (
                    proposal
                    for proposal in rate_relevant
                    if proposal["objective"] == objective_value
                ),
                key=lambda proposal: proposal["replacement_value"],
            )
            for proposal in level:
                price_candidate(proposal)
            feasible = [proposal for proposal in level if proposal["bytes"] <= cap]
            if feasible:
                winner = min(
                    feasible,
                    key=lambda proposal: (
                        proposal["bytes"],
                        proposal["replacement_value"],
                    ),
                )
                if level[0]["objective_class"] == "ambiguous":
                    feasible_ambiguous_level = True
                    break
                break

        if feasible_ambiguous_level:
            winner = None
            active_pass["feasible_ambiguous_levels"] += 1
            counters["feasible_ambiguous_levels"] += 1

        if winner is None:
            for proposal in pending_coordinate:
                if proposal["objective_class"] == "nonimproving":
                    disposition = "not_strictly_improving_unpriced"
                elif not proposal["rate_priced"]:
                    disposition = "dominated_by_feasible_better_objective_unpriced"
                    active_pass["dominance_pruned_count"] += 1
                    counters["dominance_pruned"] += 1
                elif proposal["bytes"] > cap:
                    disposition = "infeasible_cap"
                elif proposal["objective_class"] == "ambiguous":
                    disposition = "ambiguous_feasible_noncommittable"
                else:
                    raise RuntimeError("definitely improving feasible proposal was not selected")
                proposal["disposition"] = disposition
            record_pending()
            return False

        if counters["accepted_edits"] >= limits.max_accepted_edits:
            raise _ResourceLimit("max_accepted_edits")

        candidate = state.copy()
        candidate[row, channel] = winner["replacement_value"]
        candidate_digest = state_digest(candidate)
        if candidate_digest != winner["digest"]:
            active_query = {
                "phase": "candidate_digest_check",
                "pass_index": pass_index,
                "pass_kind": pass_kind,
                "row": row,
                "channel": channel,
                "origin_state_digest": origin_digest,
                "reported_candidate_state_digest": winner["digest"],
                "computed_candidate_state_digest": candidate_digest,
            }
            raise _RuntimeFailure(
                "invalid_proposal", "objective proposal candidate digest differs from core state"
            )
        if candidate_digest in visited:
            active_query = {
                "phase": "cycle_check",
                "pass_index": pass_index,
                "pass_kind": pass_kind,
                "row": row,
                "channel": channel,
                "epoch": origin_epoch,
                "origin_state_digest": origin_digest,
                "candidate_state_digest": winner["digest"],
                "current_value": current_value,
                "replacement_value": winner["replacement_value"],
            }
            raise _RuntimeFailure(
                "cycle_detected",
                "strictly improving winner revisits a prior state digest",
            )

        commit_request = WinnerCommitRequest(
            origin_epoch=origin_epoch,
            accepted_epoch=origin_epoch + 1,
            origin_state_digest=origin_digest,
            candidate_state_digest=winner["digest"],
            row=row,
            channel=channel,
            current_value=current_value,
            replacement_value=winner["replacement_value"],
            objective=winner["objective"],
            complete_bytes=winner["bytes"],
            component_bytes=dict(winner["components"]),
            objective_commit_token=winner["objective_token"],
            rate_commit_token=winner["rate_token"],
            candidate_state=_readonly_copy(candidate),
        )
        active_query = {
            "phase": "commit_paired_winner",
            "pass_index": pass_index,
            "pass_kind": pass_kind,
            "row": row,
            "channel": channel,
            "epoch": origin_epoch,
            "origin_state_digest": origin_digest,
            "candidate_state_digest": winner["digest"],
            "current_value": current_value,
            "replacement_value": winner["replacement_value"],
        }
        counters["commit_attempts"] += 1
        try:
            commit_response = adapter.commit(commit_request)
        except _RuntimeFailure:
            raise
        except MemoryError as exc:
            raise _RuntimeFailure("oom", "commit callback raised MemoryError") from exc
        except Exception as exc:
            raise _RuntimeFailure(
                "invalid_proposal", f"commit callback raised {type(exc).__name__}"
            ) from exc
        _validate_commit(commit_response, request=commit_request)
        active_query = None

        for proposal in pending_coordinate:
            if proposal["objective_class"] == "nonimproving":
                disposition = "not_strictly_improving_unpriced"
            elif not proposal["rate_priced"]:
                disposition = "dominated_by_feasible_better_objective_unpriced"
                active_pass["dominance_pruned_count"] += 1
                counters["dominance_pruned"] += 1
            elif proposal["bytes"] > cap:
                disposition = "infeasible_cap"
            elif proposal is winner:
                disposition = "accepted"
            elif proposal["objective_class"] == "ambiguous":
                disposition = "ambiguous_feasible_noncommittable"
            else:
                disposition = "dominated_by_rate_or_value"
            proposal["disposition"] = disposition

        state = candidate
        current_digest = candidate_digest
        current_objective = winner["objective"]
        current_bytes = winner["bytes"]
        current_components = winner["components"]
        epoch += 1
        counters["accepted_edits"] += 1
        counters["commits"] += 1
        if active_pass is None:
            raise RuntimeError("internal active pass disappeared")
        active_pass["accepted_edits"] += 1
        visited.add(current_digest)
        ledger["accepted_edits"].append(
            {
                "pass_index": pass_index,
                "pass_kind": pass_kind,
                "row": row,
                "channel": channel,
                "channel_name": CHANNEL_NAMES[channel],
                "origin_epoch": origin_epoch,
                "accepted_epoch": epoch,
                "origin_state_digest": origin_digest,
                "state_digest": current_digest,
                "current_value": current_value,
                "replacement_value": winner["replacement_value"],
                "objective": _objective_record(current_objective),
                "complete_bytes": current_bytes,
                "component_bytes": dict(current_components),
                "paired_commit": True,
            }
        )
        record_pending()
        return True

    try:
        (
            current_objective,
            current_bytes,
            current_components,
            current_blob,
            initial_checkpoint,
        ) = full_checkpoint(
            purpose="initial",
            expected_objective=None,
            expected_bytes=None,
            expected_components=None,
            origin_digest=None,
        )
        ledger["initial"] = initial_checkpoint
        if current_bytes > cap:
            raise _ValidMethodFailure(
                "initial_state_over_cap",
                f"initial complete bytes {current_bytes} exceed cap {cap}",
            )

        visited = {current_digest}
        while True:
            consume("sweeps", limits.max_sweeps)
            sweep_index = counters["sweeps"] - 1
            pass_index = counters["passes"]
            counters["passes"] += 1
            active_pass = {
                "pass_index": pass_index,
                "pass_kind": "local_sweep",
                "local_sweep_index": sweep_index,
                "start_epoch": epoch,
                "start_state_digest": current_digest,
                "start_objective": _objective_record(current_objective),
                "start_complete_bytes": current_bytes,
                "coordinate_visits": 0,
                "proposal_count": 0,
                "objective_score_count": 0,
                "rate_price_count": 0,
                "objective_class_counts": {
                    "definitely_improving": 0,
                    "ambiguous": 0,
                    "nonimproving": 0,
                },
                "dominance_pruned_count": 0,
                "feasible_ambiguous_levels": 0,
                "accepted_edits": 0,
                "disposition_counts": {name: 0 for name in _PROPOSAL_DISPOSITION_CODES},
                "checkpoint": None,
                "local_zero_certificate": None,
                "status": "running",
                "complete_full_traversal": False,
                "failure_reason": None,
                "failure_detail": None,
            }
            ledger["passes"].append(active_pass)

            for row in range(state.shape[0]):
                for channel in range(3):
                    active_pass["coordinate_visits"] += 1
                    counters["coordinate_visits"] += 1
                    visit_coordinate(
                        pass_index=pass_index,
                        pass_kind="local_sweep",
                        row=row,
                        channel=channel,
                        replacements=_local_neighbor_values(
                            int(state[row, channel]),
                            alphabet_name=alphabet,
                            alphabet=alphabet_values,
                        ),
                    )

            (
                current_objective,
                current_bytes,
                current_components,
                current_blob,
                checkpoint_value,
            ) = full_checkpoint(
                purpose="sweep_end",
                expected_objective=current_objective,
                expected_bytes=current_bytes,
                expected_components=current_components,
                origin_digest=current_digest,
            )
            if active_pass["accepted_edits"] > 0:
                seal_pass(
                    status="accepted_edits_continue_local",
                    complete_full_traversal=True,
                    checkpoint_value=checkpoint_value,
                )
                continue

            local_zero_certificate = _seal_record(
                {
                    "schema": "structsplat.comp012.local-zero-certificate.v1",
                    "terminal": False,
                    "local_sweep_index": sweep_index,
                    "epoch": epoch,
                    "state_digest": current_digest,
                    "objective": _objective_record(current_objective),
                    "complete_bytes": current_bytes,
                    "coordinate_visits": active_pass["coordinate_visits"],
                    "proposal_count": active_pass["proposal_count"],
                    "objective_score_count": active_pass["objective_score_count"],
                    "rate_price_count": active_pass["rate_price_count"],
                    "objective_class_counts": dict(active_pass["objective_class_counts"]),
                    "dominance_pruned_count": active_pass["dominance_pruned_count"],
                    "feasible_ambiguous_levels": active_pass["feasible_ambiguous_levels"],
                    "accepted_edits": 0,
                    "requires_full_label_audit": True,
                },
                "local_zero_certificate_sha256",
            )
            seal_pass(
                status="local_zero_requires_full_label_audit",
                complete_full_traversal=True,
                checkpoint_value=checkpoint_value,
                local_zero_certificate=local_zero_certificate,
            )

            counters["full_label_audits"] += 1
            audit_index = counters["full_label_audits"] - 1
            pass_index = counters["passes"]
            counters["passes"] += 1
            active_pass = {
                "pass_index": pass_index,
                "pass_kind": "full_label_audit",
                "full_label_audit_index": audit_index,
                "trigger_local_zero_certificate_sha256": local_zero_certificate[
                    "local_zero_certificate_sha256"
                ],
                "start_epoch": epoch,
                "start_state_digest": current_digest,
                "start_objective": _objective_record(current_objective),
                "start_complete_bytes": current_bytes,
                "coordinate_visits": 0,
                "proposal_count": 0,
                "objective_score_count": 0,
                "rate_price_count": 0,
                "objective_class_counts": {
                    "definitely_improving": 0,
                    "ambiguous": 0,
                    "nonimproving": 0,
                },
                "dominance_pruned_count": 0,
                "feasible_ambiguous_levels": 0,
                "accepted_edits": 0,
                "disposition_counts": {name: 0 for name in _PROPOSAL_DISPOSITION_CODES},
                "checkpoint": None,
                "local_zero_certificate": None,
                "status": "running",
                "complete_full_traversal": False,
                "failure_reason": None,
                "failure_detail": None,
            }
            ledger["passes"].append(active_pass)
            accepted_jump = False
            for row in range(state.shape[0]):
                if accepted_jump:
                    break
                for channel in range(3):
                    active_pass["coordinate_visits"] += 1
                    counters["coordinate_visits"] += 1
                    if visit_coordinate(
                        pass_index=pass_index,
                        pass_kind="full_label_audit",
                        row=row,
                        channel=channel,
                        replacements=_full_label_values(int(state[row, channel]), alphabet_values),
                    ):
                        accepted_jump = True
                        break

            if accepted_jump:
                (
                    current_objective,
                    current_bytes,
                    current_components,
                    current_blob,
                    checkpoint_value,
                ) = full_checkpoint(
                    purpose="audit_jump",
                    expected_objective=current_objective,
                    expected_bytes=current_bytes,
                    expected_components=current_components,
                    origin_digest=current_digest,
                )
                seal_pass(
                    status="accepted_jump_resume_local",
                    complete_full_traversal=False,
                    checkpoint_value=checkpoint_value,
                )
                continue

            expected_coordinate_visits = int(state.shape[0]) * 3
            expected_proposals = expected_coordinate_visits * (len(alphabet_values) - 1)
            if (
                active_pass["coordinate_visits"] != expected_coordinate_visits
                or active_pass["proposal_count"] != expected_proposals
                or active_pass["objective_score_count"] != expected_proposals
                or active_pass["rate_price_count"] + active_pass["dominance_pruned_count"]
                != active_pass["objective_class_counts"]["definitely_improving"]
                + active_pass["objective_class_counts"]["ambiguous"]
                or active_pass["accepted_edits"] != 0
            ):
                raise _RuntimeFailure(
                    "invalid_proposal",
                    "full-label terminal audit accounting is incomplete",
                )
            (
                current_objective,
                current_bytes,
                current_components,
                current_blob,
                terminal_checkpoint,
            ) = full_checkpoint(
                purpose="terminal",
                expected_objective=current_objective,
                expected_bytes=current_bytes,
                expected_components=current_components,
                origin_digest=current_digest,
            )
            terminal_audit_pass = seal_pass(
                status="full_label_zero_terminal",
                complete_full_traversal=True,
                checkpoint_value=terminal_checkpoint,
            )
            terminal_certificate = _seal_record(
                {
                    "schema": "structsplat.comp012.full-label-terminal-certificate.v1",
                    "alphabet_name": alphabet,
                    "alphabet_values_sha256": alphabet_metadata["values_sha256"],
                    "cap_bytes": cap,
                    "epoch": epoch,
                    "state_digest": current_digest,
                    "objective": _objective_record(current_objective),
                    "complete_bytes": current_bytes,
                    "component_bytes": current_components,
                    "blob_sha256": current_blob,
                    "trigger_local_zero_certificate_sha256": local_zero_certificate[
                        "local_zero_certificate_sha256"
                    ],
                    "full_label_audit_index": audit_index,
                    "full_label_audit_pass_sha256": terminal_audit_pass["pass_sha256"],
                    "coordinate_visits": terminal_audit_pass["coordinate_visits"],
                    "proposal_count": terminal_audit_pass["proposal_count"],
                    "expected_coordinate_visits": expected_coordinate_visits,
                    "expected_proposal_count": expected_proposals,
                    "accepted_edits": 0,
                    "complete_row_channel_traversal": (
                        terminal_audit_pass["coordinate_visits"] == expected_coordinate_visits
                    ),
                    "every_other_active_label_objective_scored": (
                        terminal_audit_pass["objective_score_count"] == expected_proposals
                    ),
                    "objective_class_counts": dict(terminal_audit_pass["objective_class_counts"]),
                    "rate_price_count": terminal_audit_pass["rate_price_count"],
                    "dominance_pruned_count": terminal_audit_pass["dominance_pruned_count"],
                    "feasible_ambiguous_levels": terminal_audit_pass["feasible_ambiguous_levels"],
                    "every_decision_relevant_label_priced_or_soundly_pruned": (
                        terminal_audit_pass["rate_price_count"]
                        + terminal_audit_pass["dominance_pruned_count"]
                        == terminal_audit_pass["objective_class_counts"]["definitely_improving"]
                        + terminal_audit_pass["objective_class_counts"]["ambiguous"]
                    ),
                    "terminal_depends_on_full_label_zero": True,
                    "terminal_kind": "epsilon_core_terminal",
                    "ambiguous_candidates_are_never_committed": True,
                    "dominance_pruning_requires_common_coordinate_epsilon": True,
                    "stationarity_scope": (
                        "frozen_cpu_objective_epsilon_scalar_alphabet_order_cap_tolerance"
                    ),
                },
                "terminal_certificate_sha256",
            )
            ledger["status"] = "core_terminal"
            ledger["failure_reason"] = None
            ledger["failure_detail"] = None
            ledger["failure_query"] = None
            ledger["promotable"] = False
            ledger["promotion_authority"] = "lifecycle_reducer_only"
            ledger["terminal_certificate"] = terminal_certificate
            ledger["proposal_transcript"] = transcript.finish()
            (
                ledger["visited_state_count"],
                ledger["visited_state_digests_sha256"],
            ) = _visited_state_summary(ledger)
            ledger["counters"] = dict(counters)
            sealed_ledger = _seal_record(ledger, "ledger_sha256")
            return SearchResult(
                status="core_terminal",
                failure_reason=None,
                state=state.copy(),
                ledger=sealed_ledger,
            )

    except _RuntimeFailure as exc:
        abort_pending()
        if isinstance(exc, _ResourceLimit):
            outcome_status = "valid_nonconvergence"
        elif isinstance(exc, _ValidMethodFailure):
            outcome_status = "valid_method_failure"
        else:
            outcome_status = "invalid_no_decision"
        return _finalize_outcome(
            ledger=ledger,
            state=state,
            status=outcome_status,
            reason=exc.reason,
            detail=exc.detail,
            counters=counters,
            active_pass=active_pass,
            active_query=active_query,
            transcript=transcript,
        )
    except MemoryError:
        abort_pending()
        return _finalize_outcome(
            ledger=ledger,
            state=state,
            status="invalid_no_decision",
            reason="oom",
            detail="coordinate-search core raised MemoryError",
            counters=counters,
            active_pass=active_pass,
            active_query=active_query,
            transcript=transcript,
        )


__all__ = [
    "CHANNEL_NAMES",
    "CoordinateRDOError",
    "ExactRDOAdapter",
    "FROZEN_UNIFORM_STEPS",
    "L1",
    "L3",
    "LEDGER_SCHEMA",
    "MaterializationRequest",
    "MaterializedState",
    "OBJECTIVE_RELATIVE_TOLERANCE",
    "ObjectiveProposal",
    "ObjectiveProposalRequest",
    "PROPOSAL_TRANSCRIPT_CHUNK_SIZE",
    "ProposalTranscript",
    "RateProposal",
    "RateProposalRequest",
    "SSP2F_COMPONENT_KEYS",
    "SearchLimits",
    "SearchResult",
    "UniformLatticeEvaluation",
    "UniformLatticeState",
    "WinnerCommitReceipt",
    "WinnerCommitRequest",
    "build_uniform_lattice_states",
    "canonical_json_bytes",
    "independent_step3_map",
    "make_arm_start",
    "map_to_lattice",
    "run_coordinate_search",
    "select_uniform_lattice",
    "state_digest",
    "strictly_improves",
    "uniform_lattice",
]
