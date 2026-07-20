"""Fail-closed paired adapter for COMP-012 exact-byte RGB coordinate RDO.

The objective engine and exact-size oracle are independently stateful.  This
adapter gives the coordinate-search core one transactional boundary across
them:

* objective proposals are nonmutating and do not invoke the byte oracle;
* exact rate is evaluated lazily and binds both engine proposals into one
  deterministic, single-use token;
* a durable commit-start record is appended and fsynced before either engine
  is changed;
* both commits and every postcondition are verified exactly; and
* any malformed, stale, partial, or otherwise unverifiable operation
  permanently poisons the adapter.  A poisoned adapter cannot be resumed.

The journal is intentionally a crash/partial-commit detector, not a recovery
log.  Existing journal paths are rejected and no rollback path exists.
"""

from __future__ import annotations

import hashlib
import math
import os
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks import ssp2f_coordinate_rdo as coordinate
from benchmarks.ssp2f_rdo_objective import (
    RECONCILIATION_OBJECTIVE_ATOL,
    RECONCILIATION_OBJECTIVE_RTOL,
    ObjectiveReconciliation,
    RGBObjectiveProposal,
    SSP2FRGBObjective,
)
from benchmarks.ssp2f_size_oracle import (
    RGB_CHANNELS,
    SSP2FSizeOracle,
    SSP2FSizeProposal,
)


JOURNAL_SCHEMA = "structsplat.comp012.ssp2f-paired-commit-journal.v1"
_OBJECTIVE_TOKEN_DOMAIN = b"structsplat.comp012.objective-token.v1\x00"
_PAIRED_TOKEN_DOMAIN = b"structsplat.comp012.paired-token.v1\x00"
_JOURNAL_COMMIT_DOMAIN = b"structsplat.comp012.journal-commit.v1\x00"
_DIGEST_CHARS = frozenset("0123456789abcdef")


class SSP2FPairedAdapterError(RuntimeError):
    """The paired adapter could not establish or preserve its invariants."""


class SSP2FPairedAdapterPoisoned(SSP2FPairedAdapterError):
    """The paired adapter is permanently unusable after a runtime failure."""


@dataclass(frozen=True)
class _EngineSnapshot:
    epoch: int
    state_digest: str
    state: np.ndarray
    objective: float
    complete_bytes: int
    component_bytes: dict[str, int]
    size_internal_digest: str
    objective_config_digest: str
    objective_operator_digest: str


class _ObjectiveCommitToken:
    """Adapter-owned, bounded-lifetime handle to one objective proposal."""

    __slots__ = (
        "commit_used",
        "generation",
        "objective_proposal",
        "owner",
        "price_used",
        "sequence",
        "token_sha256",
    )

    def __init__(
        self,
        *,
        owner: object,
        generation: int,
        sequence: int,
        objective_proposal: RGBObjectiveProposal,
        token_sha256: str,
    ) -> None:
        self.owner = owner
        self.generation = generation
        self.sequence = sequence
        self.objective_proposal = objective_proposal
        self.token_sha256 = token_sha256
        self.price_used = False
        self.commit_used = False


class _PairedCommitToken:
    """Single-use binding of one objective proposal and one rate proposal."""

    __slots__ = (
        "commit_used",
        "generation",
        "objective_token",
        "owner",
        "sequence",
        "size_proposal",
        "token_sha256",
    )

    def __init__(
        self,
        *,
        owner: object,
        generation: int,
        sequence: int,
        objective_token: _ObjectiveCommitToken,
        size_proposal: SSP2FSizeProposal,
        token_sha256: str,
    ) -> None:
        self.owner = owner
        self.generation = generation
        self.sequence = sequence
        self.objective_token = objective_token
        self.size_proposal = size_proposal
        self.token_sha256 = token_sha256
        self.commit_used = False


def _sha256(value: Mapping[str, Any], domain: bytes) -> str:
    return hashlib.sha256(domain + coordinate.canonical_json_bytes(value)).hexdigest()


def _valid_digest(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _DIGEST_CHARS for character in value)
    ):
        raise SSP2FPairedAdapterError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _finite_nonnegative(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (int, float, np.integer, np.floating),
    ):
        raise SSP2FPairedAdapterError(f"{name} must be a real scalar")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise SSP2FPairedAdapterError(f"{name} must be finite and nonnegative")
    return 0.0 if result == 0.0 else result


def _objective_error_bound(value: float) -> float:
    """Bound one incremental scalar by the frozen COMP-012 reconciliation model."""

    checked = _finite_nonnegative(value, "objective error-bound input")
    return float(
        (
            RECONCILIATION_OBJECTIVE_ATOL
            + RECONCILIATION_OBJECTIVE_RTOL * abs(checked)
        )
        / (1.0 - RECONCILIATION_OBJECTIVE_RTOL)
    )


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise SSP2FPairedAdapterError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise SSP2FPairedAdapterError(f"{name} must be at least {minimum}")
    return result


def _normalize_components(
    value: Any,
    complete_bytes: int,
    *,
    name: str,
) -> dict[str, int]:
    if not isinstance(value, Mapping):
        raise SSP2FPairedAdapterError(f"{name} must be a component-byte mapping")
    expected_keys = set(coordinate.SSP2F_COMPONENT_KEYS)
    if set(value) != expected_keys:
        raise SSP2FPairedAdapterError(
            f"{name} must contain exactly the frozen SSP2F component keys"
        )
    if tuple(value.keys()) != coordinate.SSP2F_COMPONENT_KEYS:
        raise SSP2FPairedAdapterError(
            f"{name} keys must follow the frozen SSP2F component order"
        )
    result: dict[str, int] = {}
    for key in coordinate.SSP2F_COMPONENT_KEYS:
        result[key] = _integer(
            value[key],
            f"{name}.{key}",
            minimum=0 if key == "total" else 1,
        )
    if result["prefix"] != 280:
        raise SSP2FPairedAdapterError(f"{name}.prefix must equal 280")
    if result["total"] != complete_bytes:
        raise SSP2FPairedAdapterError(f"{name}.total must equal complete_bytes")
    subtotal = sum(
        result[key] for key in coordinate.SSP2F_COMPONENT_KEYS if key != "total"
    )
    if subtotal != complete_bytes:
        raise SSP2FPairedAdapterError(
            f"{name} non-total components must sum to complete_bytes"
        )
    return result


def _readonly_state(value: np.ndarray) -> np.ndarray:
    result = np.ascontiguousarray(value, dtype=np.uint8).copy()
    result.setflags(write=False)
    return result


class SSP2FPairedRDOAdapter:
    """Concrete exact-objective/exact-rate adapter with no recovery path."""

    def __init__(
        self,
        objective: SSP2FRGBObjective,
        size_oracle: SSP2FSizeOracle,
        *,
        journal_path: str | os.PathLike[str],
    ) -> None:
        if not isinstance(objective, SSP2FRGBObjective):
            raise SSP2FPairedAdapterError("objective must be an SSP2FRGBObjective")
        if not isinstance(size_oracle, SSP2FSizeOracle):
            raise SSP2FPairedAdapterError("size_oracle must be an SSP2FSizeOracle")
        self._objective = objective
        self._size_oracle = size_oracle
        self._lock = threading.RLock()
        self._owner = object()
        self._generation = 0
        self._score_sequence = 0
        self._pair_sequence = 0
        self._poison_reason: str | None = None
        self._last_reconciliation: ObjectiveReconciliation | None = None
        self._journal_path = Path(journal_path)
        self._journal_fd = -1

        initial = self._observe_engines()
        if initial.epoch != 0:
            raise SSP2FPairedAdapterError(
                "paired engines must both begin at epoch zero for coordinate search"
            )
        self._epoch = initial.epoch
        self._state_digest = initial.state_digest
        self._objective_value = initial.objective
        self._complete_bytes = initial.complete_bytes
        self._component_bytes = initial.component_bytes
        self._size_internal_digest = initial.size_internal_digest
        self._objective_config_digest = initial.objective_config_digest
        self._objective_operator_digest = initial.objective_operator_digest

        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_APPEND
        flags |= getattr(os, "O_CLOEXEC", 0)
        try:
            self._journal_fd = os.open(self._journal_path, flags, 0o600)
            self._append_journal_record(
                {
                    "schema": JOURNAL_SCHEMA,
                    "record": "journal_start",
                    "epoch": self._epoch,
                    "state_digest": self._state_digest,
                    "objective_config_sha256": self._objective_config_digest,
                    "objective_operator_sha256": self._objective_operator_digest,
                    "size_internal_sha256": self._size_internal_digest,
                    "complete_bytes": self._complete_bytes,
                    "component_bytes": dict(self._component_bytes),
                }
            )
            directory_fd = os.open(
                self._journal_path.parent,
                os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
            )
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except Exception as exc:
            if self._journal_fd >= 0:
                os.close(self._journal_fd)
                self._journal_fd = -1
            raise SSP2FPairedAdapterError(
                f"could not create a fresh durable paired journal: {type(exc).__name__}: {exc}"
            ) from exc

    def __del__(self) -> None:
        descriptor = getattr(self, "_journal_fd", -1)
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError:
                pass
            self._journal_fd = -1

    @property
    def poisoned(self) -> bool:
        return self._poison_reason is not None

    @property
    def poison_reason(self) -> str | None:
        return self._poison_reason

    @property
    def journal_path(self) -> Path:
        return self._journal_path

    @property
    def last_reconciliation(self) -> ObjectiveReconciliation | None:
        return self._last_reconciliation

    @property
    def epoch(self) -> int:
        return self._epoch

    @property
    def state_digest(self) -> str:
        return self._state_digest

    def _append_journal_record(self, record: Mapping[str, Any]) -> None:
        if self._journal_fd < 0:
            raise SSP2FPairedAdapterError("paired journal descriptor is not open")
        payload = coordinate.canonical_json_bytes(dict(record))
        written = 0
        while written < len(payload):
            count = os.write(self._journal_fd, payload[written:])
            if count <= 0:
                raise OSError("paired journal write made no progress")
            written += count
        os.fsync(self._journal_fd)

    def _require_live(self) -> None:
        if self._poison_reason is not None:
            raise SSP2FPairedAdapterPoisoned(
                f"paired adapter is permanently poisoned: {self._poison_reason}"
            )

    def _poison(self, operation: str, exc: BaseException) -> SSP2FPairedAdapterPoisoned:
        if self._poison_reason is None:
            self._poison_reason = f"{operation}: {type(exc).__name__}: {exc}"
            try:
                self._append_journal_record(
                    {
                        "schema": JOURNAL_SCHEMA,
                        "record": "poison",
                        "operation": operation,
                        "cached_epoch": self._epoch,
                        "cached_state_digest": self._state_digest,
                        "reason": self._poison_reason,
                    }
                )
            except BaseException:
                # Poison persistence is best-effort because the original
                # failure may itself be a broken write/fsync.  Never replace
                # or weaken the causal error.
                pass
        return SSP2FPairedAdapterPoisoned(
            f"paired adapter is permanently poisoned: {self._poison_reason}"
        )

    def _observe_engines(self) -> _EngineSnapshot:
        objective_epoch = _integer(self._objective.epoch, "objective epoch")
        size_epoch = _integer(self._size_oracle.epoch, "size epoch")
        if objective_epoch != size_epoch:
            raise SSP2FPairedAdapterError("objective and size epochs differ")

        objective_state = self._objective.rgb_symbols
        if (
            not isinstance(objective_state, np.ndarray)
            or objective_state.dtype != np.uint8
            or objective_state.ndim != 2
            or objective_state.shape[0] <= 0
            or objective_state.shape[1] != 3
        ):
            raise SSP2FPairedAdapterError("objective RGB state is not uint8 [N,3]")
        size_state_raw = self._size_oracle.rgb_symbols
        if (
            not isinstance(size_state_raw, np.ndarray)
            or size_state_raw.ndim != 2
            or size_state_raw.shape != objective_state.shape
            or size_state_raw.dtype.kind not in "iu"
            or (size_state_raw.dtype.kind == "i" and np.any(size_state_raw < 0))
            or (size_state_raw.size and int(size_state_raw.max()) > 255)
        ):
            raise SSP2FPairedAdapterError(
                "size-oracle RGB state cannot be represented as uint8 [N,3]"
            )
        size_state = np.ascontiguousarray(size_state_raw, dtype=np.uint8)
        if not np.array_equal(objective_state, size_state):
            raise SSP2FPairedAdapterError("objective and size RGB states differ")
        common_digest = coordinate.state_digest(objective_state)
        if _valid_digest(self._objective.state_sha256, "objective state digest") != common_digest:
            raise SSP2FPairedAdapterError(
                "objective state digest differs from the common coordinate digest"
            )

        objective_value = _finite_nonnegative(
            self._objective.objective,
            "objective value",
        )
        complete_bytes = _integer(
            self._size_oracle.complete_bytes,
            "complete bytes",
            minimum=1,
        )
        components_without_total = self._size_oracle.byte_breakdown
        components = dict(components_without_total)
        components["total"] = complete_bytes
        normalized_components = _normalize_components(
            components,
            complete_bytes,
            name="size-oracle component bytes",
        )
        size_internal_digest = _valid_digest(
            self._size_oracle.state_sha256,
            "size internal state digest",
        )
        objective_config_digest = _valid_digest(
            self._objective.config_sha256,
            "objective config digest",
        )
        objective_operator_digest = _valid_digest(
            self._objective.operator_sha256,
            "objective operator digest",
        )
        return _EngineSnapshot(
            epoch=objective_epoch,
            state_digest=common_digest,
            state=np.ascontiguousarray(objective_state).copy(),
            objective=objective_value,
            complete_bytes=complete_bytes,
            component_bytes=normalized_components,
            size_internal_digest=size_internal_digest,
            objective_config_digest=objective_config_digest,
            objective_operator_digest=objective_operator_digest,
        )

    def _validate_live_engines(self) -> _EngineSnapshot:
        observed = self._observe_engines()
        if (
            observed.epoch != self._epoch
            or observed.state_digest != self._state_digest
            or observed.objective.hex() != self._objective_value.hex()
            or observed.complete_bytes != self._complete_bytes
            or observed.component_bytes != self._component_bytes
            or observed.size_internal_digest != self._size_internal_digest
            or observed.objective_config_digest != self._objective_config_digest
            or observed.objective_operator_digest != self._objective_operator_digest
        ):
            raise SSP2FPairedAdapterError(
                "an engine changed outside the paired adapter or its cached state drifted"
            )
        return observed

    def _validate_coordinates(
        self,
        *,
        epoch: Any,
        origin_digest: Any,
        row: Any,
        channel: Any,
        current_value: Any,
        replacement_value: Any,
        state: np.ndarray,
    ) -> tuple[int, int, int]:
        normalized_epoch = _integer(epoch, "request epoch")
        if normalized_epoch != self._epoch:
            raise SSP2FPairedAdapterError("request epoch is stale")
        if _valid_digest(origin_digest, "request origin digest") != self._state_digest:
            raise SSP2FPairedAdapterError("request origin state digest is stale")
        normalized_row = _integer(row, "request row")
        normalized_channel = _integer(channel, "request channel")
        if normalized_row >= state.shape[0] or normalized_channel >= 3:
            raise SSP2FPairedAdapterError("request coordinate is out of range")
        normalized_current = _integer(current_value, "request current value")
        normalized_replacement = _integer(
            replacement_value,
            "request replacement value",
        )
        if normalized_current > 255 or normalized_replacement > 255:
            raise SSP2FPairedAdapterError("request RGB value is outside [0,255]")
        if int(state[normalized_row, normalized_channel]) != normalized_current:
            raise SSP2FPairedAdapterError("request current value differs from accepted state")
        if normalized_current == normalized_replacement:
            raise SSP2FPairedAdapterError("request replacement must change the selected value")
        return normalized_row, normalized_channel, normalized_replacement

    def score(
        self,
        request: coordinate.ObjectiveProposalRequest,
    ) -> coordinate.ObjectiveProposal:
        """Score an edit without invoking or mutating the exact-size oracle."""

        with self._lock:
            self._require_live()
            try:
                if not isinstance(request, coordinate.ObjectiveProposalRequest):
                    raise SSP2FPairedAdapterError(
                        "score requires an ObjectiveProposalRequest"
                    )
                before = self._validate_live_engines()
                row, channel, replacement = self._validate_coordinates(
                    epoch=request.epoch,
                    origin_digest=request.origin_state_digest,
                    row=request.row,
                    channel=request.channel,
                    current_value=request.current_value,
                    replacement_value=request.replacement_value,
                    state=before.state,
                )
                proposal = self._objective.propose(row, channel, replacement)
                after = self._validate_live_engines()
                if (
                    proposal.epoch != self._epoch
                    or proposal.config_sha256 != self._objective_config_digest
                    or proposal.origin_state_sha256 != self._state_digest
                    or proposal.row != row
                    or proposal.channel != channel
                    or proposal.old_symbol != request.current_value
                    or proposal.new_symbol != replacement
                    or proposal.objective_before.hex() != self._objective_value.hex()
                ):
                    raise SSP2FPairedAdapterError(
                        "objective engine returned a mismatched proposal"
                    )
                candidate = before.state.copy()
                candidate[row, channel] = replacement
                candidate_digest = coordinate.state_digest(candidate)
                if proposal.candidate_state_sha256 != candidate_digest:
                    raise SSP2FPairedAdapterError(
                        "objective candidate digest differs from the edited RGB state"
                    )
                objective_value = _finite_nonnegative(
                    proposal.objective,
                    "proposal objective",
                )
                margin_error = float(
                    _objective_error_bound(objective_value)
                    + (1.0 + coordinate.OBJECTIVE_RELATIVE_TOLERANCE)
                    * _objective_error_bound(self._objective_value)
                )
                sequence = self._score_sequence
                self._score_sequence += 1
                token_record = {
                    "generation": self._generation,
                    "sequence": sequence,
                    "epoch": self._epoch,
                    "origin_state_digest": self._state_digest,
                    "candidate_state_digest": candidate_digest,
                    "row": row,
                    "channel": channel,
                    "current_value": request.current_value,
                    "replacement_value": replacement,
                    "objective_float_hex": objective_value.hex(),
                    "objective_proposal_sha256": proposal.proposal_sha256,
                }
                token = _ObjectiveCommitToken(
                    owner=self._owner,
                    generation=self._generation,
                    sequence=sequence,
                    objective_proposal=proposal,
                    token_sha256=_sha256(token_record, _OBJECTIVE_TOKEN_DOMAIN),
                )
                if after.state_digest != before.state_digest:
                    raise SSP2FPairedAdapterError("score mutated accepted engine state")
                return coordinate.ObjectiveProposal(
                    epoch=self._epoch,
                    origin_state_digest=self._state_digest,
                    candidate_state_digest=candidate_digest,
                    row=row,
                    channel=channel,
                    current_value=request.current_value,
                    replacement_value=replacement,
                    objective=objective_value,
                    margin_error=margin_error,
                    commit_token=token,
                )
            except SSP2FPairedAdapterPoisoned:
                raise
            except Exception as exc:
                raise self._poison("score", exc) from exc

    def _candidate_components(
        self,
        proposal: SSP2FSizeProposal,
    ) -> dict[str, int]:
        components = dict(self._component_bytes)
        components[f"{proposal.channel}_arithmetic"] = proposal.arithmetic_bytes
        components["model_zlib9"] = len(proposal.model_payload)
        components["total"] = proposal.complete_bytes
        return _normalize_components(
            components,
            proposal.complete_bytes,
            name="rate proposal component bytes",
        )

    def price(
        self,
        request: coordinate.RateProposalRequest,
    ) -> coordinate.RateProposal:
        """Evaluate exact complete bytes only after a valid objective score."""

        with self._lock:
            self._require_live()
            try:
                if not isinstance(request, coordinate.RateProposalRequest):
                    raise SSP2FPairedAdapterError("price requires a RateProposalRequest")
                before = self._validate_live_engines()
                row, channel, replacement = self._validate_coordinates(
                    epoch=request.epoch,
                    origin_digest=request.origin_state_digest,
                    row=request.row,
                    channel=request.channel,
                    current_value=request.current_value,
                    replacement_value=request.replacement_value,
                    state=before.state,
                )
                objective_token = request.objective_commit_token
                if (
                    not isinstance(objective_token, _ObjectiveCommitToken)
                    or objective_token.owner is not self._owner
                    or objective_token.generation != self._generation
                ):
                    raise SSP2FPairedAdapterError(
                        "price received an unknown or stale objective token"
                    )
                if objective_token.price_used or objective_token.commit_used:
                    raise SSP2FPairedAdapterError("objective token was reused")
                objective_proposal = objective_token.objective_proposal
                if (
                    objective_proposal.epoch != self._epoch
                    or objective_proposal.origin_state_sha256 != self._state_digest
                    or objective_proposal.candidate_state_sha256
                    != request.candidate_state_digest
                    or objective_proposal.row != row
                    or objective_proposal.channel != channel
                    or objective_proposal.old_symbol != request.current_value
                    or objective_proposal.new_symbol != replacement
                    or objective_proposal.objective.hex()
                    != _finite_nonnegative(
                        request.objective,
                        "priced objective",
                    ).hex()
                ):
                    raise SSP2FPairedAdapterError(
                        "price request differs from its objective token"
                    )
                objective_token.price_used = True
                regenerated_objective = self._objective.propose(row, channel, replacement)
                if regenerated_objective != objective_proposal:
                    raise SSP2FPairedAdapterError(
                        "objective token no longer matches a fresh proposal"
                    )
                size_proposal = self._size_oracle.propose(
                    row,
                    RGB_CHANNELS[channel],
                    replacement,
                )
                after = self._validate_live_engines()
                if (
                    size_proposal.epoch != self._epoch
                    or size_proposal.state_sha256 != self._size_internal_digest
                    or size_proposal.row != row
                    or size_proposal.channel != RGB_CHANNELS[channel]
                    or size_proposal.old_symbol != request.current_value
                    or size_proposal.new_symbol != replacement
                ):
                    raise SSP2FPairedAdapterError(
                        "size oracle returned a mismatched proposal"
                    )
                components = self._candidate_components(size_proposal)
                pair_sequence = self._pair_sequence
                self._pair_sequence += 1
                pair_record = {
                    "generation": self._generation,
                    "sequence": pair_sequence,
                    "epoch": self._epoch,
                    "origin_state_digest": self._state_digest,
                    "candidate_state_digest": request.candidate_state_digest,
                    "objective_token_sha256": objective_token.token_sha256,
                    "objective_proposal_sha256": objective_proposal.proposal_sha256,
                    "size_proposal_sha256": size_proposal.proposal_sha256,
                    "objective_float_hex": objective_proposal.objective.hex(),
                    "complete_bytes": size_proposal.complete_bytes,
                    "component_bytes": components,
                }
                pair_token = _PairedCommitToken(
                    owner=self._owner,
                    generation=self._generation,
                    sequence=pair_sequence,
                    objective_token=objective_token,
                    size_proposal=size_proposal,
                    token_sha256=_sha256(pair_record, _PAIRED_TOKEN_DOMAIN),
                )
                if after.state_digest != before.state_digest:
                    raise SSP2FPairedAdapterError("price mutated accepted engine state")
                return coordinate.RateProposal(
                    epoch=self._epoch,
                    origin_state_digest=self._state_digest,
                    candidate_state_digest=request.candidate_state_digest,
                    row=row,
                    channel=channel,
                    current_value=request.current_value,
                    replacement_value=replacement,
                    complete_bytes=size_proposal.complete_bytes,
                    component_bytes=components,
                    commit_token=pair_token,
                )
            except SSP2FPairedAdapterPoisoned:
                raise
            except Exception as exc:
                raise self._poison("price", exc) from exc

    def _validate_winner(
        self,
        request: coordinate.WinnerCommitRequest,
        snapshot: _EngineSnapshot,
    ) -> tuple[_ObjectiveCommitToken, _PairedCommitToken]:
        if not isinstance(request, coordinate.WinnerCommitRequest):
            raise SSP2FPairedAdapterError("commit requires a WinnerCommitRequest")
        row, channel, replacement = self._validate_coordinates(
            epoch=request.origin_epoch,
            origin_digest=request.origin_state_digest,
            row=request.row,
            channel=request.channel,
            current_value=request.current_value,
            replacement_value=request.replacement_value,
            state=snapshot.state,
        )
        if request.accepted_epoch != self._epoch + 1:
            raise SSP2FPairedAdapterError("accepted epoch is not the next paired epoch")
        objective_token = request.objective_commit_token
        pair_token = request.rate_commit_token
        if (
            not isinstance(objective_token, _ObjectiveCommitToken)
            or not isinstance(pair_token, _PairedCommitToken)
            or objective_token.owner is not self._owner
            or pair_token.owner is not self._owner
            or objective_token.generation != self._generation
            or pair_token.generation != self._generation
            or pair_token.objective_token is not objective_token
        ):
            raise SSP2FPairedAdapterError("winner does not carry one known paired token")
        if (
            not objective_token.price_used
            or objective_token.commit_used
            or pair_token.commit_used
        ):
            raise SSP2FPairedAdapterError("paired winner token is unused incorrectly or reused")

        objective_proposal = objective_token.objective_proposal
        size_proposal = pair_token.size_proposal
        components = _normalize_components(
            request.component_bytes,
            _integer(request.complete_bytes, "winner complete bytes", minimum=1),
            name="winner component bytes",
        )
        requested_objective = _finite_nonnegative(
            request.objective,
            "winner objective",
        )
        expected_components = self._candidate_components(size_proposal)
        expected_candidate = snapshot.state.copy()
        expected_candidate[row, channel] = replacement
        if (
            not isinstance(request.candidate_state, np.ndarray)
            or request.candidate_state.dtype != np.uint8
            or request.candidate_state.shape != snapshot.state.shape
            or not np.array_equal(request.candidate_state, expected_candidate)
        ):
            raise SSP2FPairedAdapterError(
                "winner candidate state is not the requested single edit"
            )
        candidate_digest = coordinate.state_digest(expected_candidate)
        if (
            _valid_digest(request.candidate_state_digest, "winner candidate digest")
            != candidate_digest
            or objective_proposal.candidate_state_sha256 != candidate_digest
            or objective_proposal.row != row
            or objective_proposal.channel != channel
            or objective_proposal.old_symbol != request.current_value
            or objective_proposal.new_symbol != replacement
            or objective_proposal.objective.hex() != requested_objective.hex()
            or size_proposal.row != row
            or size_proposal.channel != RGB_CHANNELS[channel]
            or size_proposal.old_symbol != request.current_value
            or size_proposal.new_symbol != replacement
            or size_proposal.complete_bytes != request.complete_bytes
            or components != expected_components
        ):
            raise SSP2FPairedAdapterError(
                "winner fields differ from the bound objective/rate proposals"
            )
        return objective_token, pair_token

    def commit(
        self,
        request: coordinate.WinnerCommitRequest,
    ) -> coordinate.WinnerCommitReceipt:
        """Durably announce and then commit one winner to both engines once."""

        with self._lock:
            self._require_live()
            try:
                before = self._validate_live_engines()
                objective_token, pair_token = self._validate_winner(request, before)
                objective_proposal = objective_token.objective_proposal
                size_proposal = pair_token.size_proposal

                if self._objective.propose(
                    request.row,
                    request.channel,
                    request.replacement_value,
                ) != objective_proposal:
                    raise SSP2FPairedAdapterError(
                        "objective proposal failed commit prevalidation"
                    )
                if self._size_oracle.propose(
                    request.row,
                    RGB_CHANNELS[request.channel],
                    request.replacement_value,
                ) != size_proposal:
                    raise SSP2FPairedAdapterError(
                        "size proposal failed commit prevalidation"
                    )
                self._validate_live_engines()

                objective_token.commit_used = True
                pair_token.commit_used = True
                commit_identity = {
                    "origin_epoch": request.origin_epoch,
                    "accepted_epoch": request.accepted_epoch,
                    "origin_state_digest": request.origin_state_digest,
                    "candidate_state_digest": request.candidate_state_digest,
                    "row": request.row,
                    "channel": request.channel,
                    "current_value": request.current_value,
                    "replacement_value": request.replacement_value,
                    "objective_float_hex": _finite_nonnegative(
                        request.objective,
                        "winner objective",
                    ).hex(),
                    "complete_bytes": request.complete_bytes,
                    "component_bytes": dict(request.component_bytes),
                    "objective_token_sha256": objective_token.token_sha256,
                    "paired_token_sha256": pair_token.token_sha256,
                    "objective_proposal_sha256": objective_proposal.proposal_sha256,
                    "size_proposal_sha256": size_proposal.proposal_sha256,
                }
                commit_sha256 = _sha256(commit_identity, _JOURNAL_COMMIT_DOMAIN)
                self._append_journal_record(
                    {
                        "schema": JOURNAL_SCHEMA,
                        "record": "commit_start",
                        "commit_sha256": commit_sha256,
                        **commit_identity,
                    }
                )

                committed_objective = self._objective.commit(objective_proposal)
                if (
                    _finite_nonnegative(
                        committed_objective,
                        "committed objective",
                    ).hex()
                    != _finite_nonnegative(
                        request.objective,
                        "winner objective",
                    ).hex()
                    or self._objective.epoch != request.accepted_epoch
                    or self._objective.state_sha256 != request.candidate_state_digest
                ):
                    raise SSP2FPairedAdapterError(
                        "objective engine failed its post-commit verification"
                    )
                committed_bytes = self._size_oracle.commit(size_proposal)
                if committed_bytes != request.complete_bytes:
                    raise SSP2FPairedAdapterError(
                        "size oracle failed its post-commit byte verification"
                    )

                after = self._observe_engines()
                expected_components = _normalize_components(
                    request.component_bytes,
                    request.complete_bytes,
                    name="committed component bytes",
                )
                if (
                    after.epoch != request.accepted_epoch
                    or after.state_digest != request.candidate_state_digest
                    or after.objective.hex()
                    != _finite_nonnegative(
                        request.objective,
                        "winner objective",
                    ).hex()
                    or after.complete_bytes != request.complete_bytes
                    or after.component_bytes != expected_components
                    or not np.array_equal(after.state, request.candidate_state)
                    or after.objective_config_digest != self._objective_config_digest
                    or after.objective_operator_digest != self._objective_operator_digest
                ):
                    raise SSP2FPairedAdapterError(
                        "paired engines disagree after the two commits"
                    )

                self._append_journal_record(
                    {
                        "schema": JOURNAL_SCHEMA,
                        "record": "commit_complete",
                        "commit_sha256": commit_sha256,
                        "accepted_epoch": after.epoch,
                        "state_digest": after.state_digest,
                        "objective_float_hex": after.objective.hex(),
                        "complete_bytes": after.complete_bytes,
                        "component_bytes": dict(after.component_bytes),
                        "objective_token_sha256": objective_token.token_sha256,
                        "paired_token_sha256": pair_token.token_sha256,
                        "size_internal_sha256": after.size_internal_digest,
                    }
                )
                self._epoch = after.epoch
                self._state_digest = after.state_digest
                self._objective_value = after.objective
                self._complete_bytes = after.complete_bytes
                self._component_bytes = after.component_bytes
                self._size_internal_digest = after.size_internal_digest
                return coordinate.WinnerCommitReceipt(
                    origin_epoch=request.origin_epoch,
                    accepted_epoch=request.accepted_epoch,
                    origin_state_digest=request.origin_state_digest,
                    candidate_state_digest=request.candidate_state_digest,
                    objective=after.objective,
                    complete_bytes=after.complete_bytes,
                    component_bytes=dict(after.component_bytes),
                )
            except SSP2FPairedAdapterPoisoned:
                raise
            except Exception as exc:
                raise self._poison("commit", exc) from exc

    def materialize(
        self,
        request: coordinate.MaterializationRequest,
    ) -> coordinate.MaterializedState:
        """Full-reconcile the objective and ordinary-encode/verify exact bytes."""

        with self._lock:
            self._require_live()
            try:
                if not isinstance(request, coordinate.MaterializationRequest):
                    raise SSP2FPairedAdapterError(
                        "materialize requires a MaterializationRequest"
                    )
                before = self._validate_live_engines()
                if request.purpose not in {"initial", "sweep_end", "audit_jump", "terminal"}:
                    raise SSP2FPairedAdapterError(
                        "materialization purpose is outside the frozen checkpoint schedule"
                    )
                if (
                    _integer(request.epoch, "materialization epoch") != self._epoch
                    or _valid_digest(
                        request.state_digest,
                        "materialization state digest",
                    )
                    != self._state_digest
                    or not isinstance(request.state, np.ndarray)
                    or request.state.dtype != np.uint8
                    or request.state.shape != before.state.shape
                    or not np.array_equal(request.state, before.state)
                ):
                    raise SSP2FPairedAdapterError(
                        "materialization request state or epoch is stale"
                    )
                if request.purpose == "initial":
                    if (
                        request.origin_state_digest is not None
                        or request.expected_objective is not None
                        or request.expected_complete_bytes is not None
                    ):
                        raise SSP2FPairedAdapterError(
                            "initial materialization must not carry tracked expectations"
                        )
                else:
                    if request.origin_state_digest != self._state_digest:
                        raise SSP2FPairedAdapterError(
                            "checkpoint origin digest must equal accepted state"
                        )
                    if request.expected_objective is None:
                        raise SSP2FPairedAdapterError(
                            "noninitial checkpoint lacks its tracked objective"
                        )
                    expected = _finite_nonnegative(
                        request.expected_objective,
                        "expected objective",
                    )
                    if expected.hex() != self._objective_value.hex():
                        raise SSP2FPairedAdapterError(
                            "checkpoint objective differs from paired tracked state"
                        )
                    if (
                        _integer(
                            request.expected_complete_bytes,
                            "expected complete bytes",
                            minimum=1,
                        )
                        != self._complete_bytes
                    ):
                        raise SSP2FPairedAdapterError(
                            "checkpoint bytes differ from paired tracked state"
                        )

                report = self._objective.reconcile()
                self._last_reconciliation = report
                if (
                    not report.passed
                    or not report.rebased
                    or report.epoch != self._epoch
                    or report.state_sha256 != self._state_digest
                ):
                    raise SSP2FPairedAdapterError(
                        "objective reconciliation did not certify the accepted state"
                    )
                blob = self._size_oracle.encode_and_verify()
                after = self._observe_engines()
                blob_sha256 = hashlib.sha256(blob).hexdigest()
                if (
                    after.epoch != before.epoch
                    or after.state_digest != before.state_digest
                    or not np.array_equal(after.state, before.state)
                    or after.complete_bytes != len(blob)
                    or after.component_bytes != before.component_bytes
                    or after.size_internal_digest != before.size_internal_digest
                    or after.objective_config_digest != self._objective_config_digest
                    or after.objective_operator_digest != self._objective_operator_digest
                    or report.full_objective.hex() != after.objective.hex()
                ):
                    raise SSP2FPairedAdapterError(
                        "full reconciliation/encoding changed or disagreed with accepted state"
                    )
                expected_objective = request.expected_objective
                objective_error = (
                    0.0
                    if expected_objective is None
                    else abs(after.objective - float(expected_objective))
                )
                if objective_error > report.objective_tolerance:
                    raise SSP2FPairedAdapterError(
                        "full objective differs from tracked objective beyond tolerance"
                    )
                if (
                    request.expected_complete_bytes is not None
                    and after.complete_bytes != request.expected_complete_bytes
                ):
                    raise SSP2FPairedAdapterError(
                        "ordinary encoded bytes differ from tracked exact bytes"
                    )

                self._objective_value = after.objective
                self._generation += 1
                return coordinate.MaterializedState(
                    epoch=after.epoch,
                    state_digest=after.state_digest,
                    objective=after.objective,
                    complete_bytes=after.complete_bytes,
                    component_bytes=dict(after.component_bytes),
                    blob_sha256=blob_sha256,
                    objective_abs_error=objective_error,
                    objective_tolerance=report.objective_tolerance,
                    objective_reconciliation_passed=True,
                )
            except SSP2FPairedAdapterPoisoned:
                raise
            except Exception as exc:
                raise self._poison("materialize", exc) from exc


__all__ = [
    "JOURNAL_SCHEMA",
    "SSP2FPairedAdapterError",
    "SSP2FPairedAdapterPoisoned",
    "SSP2FPairedRDOAdapter",
]
