from __future__ import annotations

import hashlib
import math
import sys
from collections.abc import Callable, Mapping
from dataclasses import replace
from typing import Any

import numpy as np
import pytest

from benchmarks import ssp2f_coordinate_rdo as R


Objective = Callable[[np.ndarray], float]
ByteFunction = Callable[[np.ndarray], int]
MarginErrorFunction = Callable[[R.ObjectiveProposalRequest, np.ndarray, float], float]


def _blob_digest(state: np.ndarray) -> str:
    return hashlib.sha256(b"fake-ordinary-blob\x00" + state.tobytes()).hexdigest()


def _deep_size(value: Any, seen: set[int] | None = None) -> int:
    """Conservative recursive size of an acyclic JSON-like result."""

    identities = seen if seen is not None else set()
    identity = id(value)
    if identity in identities:
        return 0
    identities.add(identity)
    size = sys.getsizeof(value)
    if isinstance(value, Mapping):
        return size + sum(
            _deep_size(key, identities) + _deep_size(item, identities)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple, set, frozenset)):
        return size + sum(_deep_size(item, identities) for item in value)
    return size


class FakeAdapter:
    """Stateful two-engine adapter that makes protocol violations observable."""

    def __init__(
        self,
        initial_state: np.ndarray,
        *,
        objective: Objective,
        byte_function: ByteFunction | None = None,
        margin_error: MarginErrorFunction | None = None,
        score_fault: str | None = None,
        price_fault: str | None = None,
        materialize_fault: str | None = None,
        commit_fault: str | None = None,
        checkpoint_drift: Mapping[str, float] | None = None,
        checkpoint_tolerance: Mapping[str, float] | None = None,
        checkpoint_pass: Mapping[str, bool] | None = None,
    ) -> None:
        self.objective_function = objective
        self.byte_function = byte_function or (lambda _state: 300)
        self.margin_error_function = margin_error or (lambda _request, _candidate, _objective: 0.0)
        self.score_fault = score_fault
        self.price_fault = price_fault
        self.materialize_fault = materialize_fault
        self.commit_fault = commit_fault
        self.checkpoint_drift = dict(checkpoint_drift or {})
        self.checkpoint_tolerance = dict(checkpoint_tolerance or {})
        self.checkpoint_pass = dict(checkpoint_pass or {})
        self.epoch = 0
        self.digest = R.state_digest(initial_state)
        self.state = initial_state.copy()
        self.score_requests: list[R.ObjectiveProposalRequest] = []
        self.price_requests: list[R.RateProposalRequest] = []
        self.materialization_requests: list[R.MaterializationRequest] = []
        self.commit_requests: list[R.WinnerCommitRequest] = []

    @staticmethod
    def _components(complete_bytes: int) -> dict[str, int]:
        assert complete_bytes >= 288
        result = {key: 1 for key in R.SSP2F_COMPONENT_KEYS}
        result["prefix"] = 280
        result["model_zlib9"] = complete_bytes - 287
        result["total"] = complete_bytes
        return result

    @staticmethod
    def _objective_token(
        request: R.ObjectiveProposalRequest,
        candidate_digest: str,
        objective: float,
    ) -> tuple[Any, ...]:
        return (
            "objective",
            request.epoch,
            request.origin_state_digest,
            request.row,
            request.channel,
            request.current_value,
            request.replacement_value,
            candidate_digest,
            float(objective).hex(),
        )

    def _assert_origin(
        self,
        epoch: int,
        origin_state_digest: str,
        row: int,
        channel: int,
        current_value: int,
    ) -> None:
        assert epoch == self.epoch
        assert origin_state_digest == self.digest
        assert int(self.state[row, channel]) == current_value

    def score(self, request: R.ObjectiveProposalRequest) -> R.ObjectiveProposal:
        self.score_requests.append(request)
        self._assert_origin(
            request.epoch,
            request.origin_state_digest,
            request.row,
            request.channel,
            request.current_value,
        )
        candidate = self.state.copy()
        candidate[request.row, request.channel] = request.replacement_value
        candidate_digest = R.state_digest(candidate)
        objective = float(self.objective_function(candidate))
        margin_error = float(self.margin_error_function(request, candidate, objective))
        response = R.ObjectiveProposal(
            epoch=request.epoch,
            origin_state_digest=request.origin_state_digest,
            candidate_state_digest=candidate_digest,
            row=request.row,
            channel=request.channel,
            current_value=request.current_value,
            replacement_value=request.replacement_value,
            objective=objective,
            margin_error=margin_error,
            commit_token=self._objective_token(request, candidate_digest, objective),
        )
        if self.score_fault == "stale":
            return replace(response, epoch=response.epoch + 1)
        if self.score_fault == "negative":
            return replace(response, objective=-1.0)
        if self.score_fault == "bad_digest":
            return replace(response, candidate_state_digest="f" * 64)
        return response

    def price(self, request: R.RateProposalRequest) -> R.RateProposal:
        self.price_requests.append(request)
        self._assert_origin(
            request.epoch,
            request.origin_state_digest,
            request.row,
            request.channel,
            request.current_value,
        )
        candidate = self.state.copy()
        candidate[request.row, request.channel] = request.replacement_value
        assert R.state_digest(candidate) == request.candidate_state_digest
        objective_request = R.ObjectiveProposalRequest(
            epoch=request.epoch,
            origin_state_digest=request.origin_state_digest,
            row=request.row,
            channel=request.channel,
            current_value=request.current_value,
            replacement_value=request.replacement_value,
        )
        assert request.objective_commit_token == self._objective_token(
            objective_request,
            request.candidate_state_digest,
            request.objective,
        )
        complete_bytes = int(self.byte_function(candidate))
        components = self._components(complete_bytes)
        response = R.RateProposal(
            epoch=request.epoch,
            origin_state_digest=request.origin_state_digest,
            candidate_state_digest=request.candidate_state_digest,
            row=request.row,
            channel=request.channel,
            current_value=request.current_value,
            replacement_value=request.replacement_value,
            complete_bytes=complete_bytes,
            component_bytes=components,
            commit_token=(
                "rate",
                request.epoch,
                request.origin_state_digest,
                request.candidate_state_digest,
                request.replacement_value,
                complete_bytes,
            ),
        )
        if self.price_fault == "stale":
            return replace(response, epoch=response.epoch + 1)
        if self.price_fault == "bad_components":
            return replace(
                response,
                component_bytes={"prefix": complete_bytes, "total": complete_bytes},
            )
        if self.price_fault == "bad_sum":
            bad = dict(components)
            bad["means"] += 1
            return replace(response, component_bytes=bad)
        if self.price_fault == "bad_prefix":
            bad = dict(components)
            bad["prefix"] += 1
            bad["model_zlib9"] -= 1
            return replace(response, component_bytes=bad)
        if self.price_fault == "zero_payload":
            bad = dict(components)
            bad["means"] = 0
            bad["model_zlib9"] += 1
            return replace(response, component_bytes=bad)
        if self.price_fault == "bad_order":
            return replace(
                response,
                component_bytes=dict(reversed(tuple(components.items()))),
            )
        return response

    def materialize(self, request: R.MaterializationRequest) -> R.MaterializedState:
        self.materialization_requests.append(request)
        assert request.epoch == self.epoch
        assert request.state_digest == self.digest
        np.testing.assert_array_equal(request.state, self.state)
        complete_bytes = int(self.byte_function(request.state))
        objective = float(self.objective_function(request.state))
        objective += float(self.checkpoint_drift.get(request.purpose, 0.0))
        objective_error = (
            0.0
            if request.expected_objective is None
            else abs(objective - request.expected_objective)
        )
        tolerance = float(self.checkpoint_tolerance.get(request.purpose, objective_error))
        passed = self.checkpoint_pass.get(request.purpose, objective_error <= tolerance)
        components = self._components(complete_bytes)
        response = R.MaterializedState(
            epoch=request.epoch,
            state_digest=request.state_digest,
            objective=objective,
            complete_bytes=complete_bytes,
            component_bytes=components,
            blob_sha256=_blob_digest(request.state),
            objective_abs_error=objective_error,
            objective_tolerance=tolerance,
            objective_reconciliation_passed=passed,
        )
        if self.materialize_fault == "stale":
            return replace(response, epoch=response.epoch + 1)
        if self.materialize_fault == "bad_components":
            return replace(
                response,
                component_bytes={"prefix": complete_bytes, "total": complete_bytes},
            )
        if self.materialize_fault == "bad_error":
            return replace(response, objective_abs_error=objective_error + 1.0)
        return response

    def commit(self, request: R.WinnerCommitRequest) -> R.WinnerCommitReceipt:
        self.commit_requests.append(request)
        self._assert_origin(
            request.origin_epoch,
            request.origin_state_digest,
            request.row,
            request.channel,
            request.current_value,
        )
        assert request.accepted_epoch == self.epoch + 1
        assert request.candidate_state_digest == R.state_digest(request.candidate_state)
        assert (
            int(request.candidate_state[request.row, request.channel]) == request.replacement_value
        )
        objective_request = R.ObjectiveProposalRequest(
            epoch=request.origin_epoch,
            origin_state_digest=request.origin_state_digest,
            row=request.row,
            channel=request.channel,
            current_value=request.current_value,
            replacement_value=request.replacement_value,
        )
        assert request.objective_commit_token == self._objective_token(
            objective_request,
            request.candidate_state_digest,
            request.objective,
        )
        assert request.rate_commit_token == (
            "rate",
            request.origin_epoch,
            request.origin_state_digest,
            request.candidate_state_digest,
            request.replacement_value,
            request.complete_bytes,
        )
        assert request.complete_bytes == self.byte_function(request.candidate_state)
        assert dict(request.component_bytes) == self._components(request.complete_bytes)
        receipt = R.WinnerCommitReceipt(
            origin_epoch=request.origin_epoch,
            accepted_epoch=request.accepted_epoch,
            origin_state_digest=request.origin_state_digest,
            candidate_state_digest=request.candidate_state_digest,
            objective=request.objective,
            complete_bytes=request.complete_bytes,
            component_bytes=dict(request.component_bytes),
        )
        if self.commit_fault == "stale":
            return replace(receipt, accepted_epoch=receipt.accepted_epoch + 1)
        if self.commit_fault == "bad_objective":
            return replace(receipt, objective=receipt.objective + 1.0)
        if self.commit_fault == "bad_components":
            bad = dict(receipt.component_bytes)
            bad["means"] += 1
            return replace(receipt, component_bytes=bad)
        if self.commit_fault == "partial_raise":
            self.epoch = request.accepted_epoch
            self.digest = request.candidate_state_digest
            self.state = request.candidate_state.copy()
            raise RuntimeError("simulated partial cross-engine commit")

        self.epoch = request.accepted_epoch
        self.digest = request.candidate_state_digest
        self.state = request.candidate_state.copy()
        return receipt


def _limits(**overrides: int) -> R.SearchLimits:
    values = {
        "max_sweeps": 100,
        "max_proposals": 100_000,
        "max_rate_prices": 100_000,
        "max_accepted_edits": 1_000,
        "max_materializations": 5_000,
    }
    values.update(overrides)
    return R.SearchLimits(**values)


def _first_channel_objective(values: Mapping[int, float], default: float) -> Objective:
    def objective(state: np.ndarray) -> float:
        return float(values.get(int(state[0, 0]), default))

    return objective


def _squared_objective(target: np.ndarray) -> Objective:
    target_i64 = target.astype(np.int64)

    def objective(state: np.ndarray) -> float:
        delta = state.astype(np.int64) - target_i64
        return float(np.sum(delta * delta))

    return objective


def _run(
    initial: np.ndarray,
    objective: Objective,
    *,
    alphabet: str = "L1",
    cap_bytes: int = 300,
    byte_function: ByteFunction | None = None,
    margin_error: MarginErrorFunction | None = None,
    limits: R.SearchLimits | None = None,
    score_fault: str | None = None,
    price_fault: str | None = None,
    materialize_fault: str | None = None,
    commit_fault: str | None = None,
    checkpoint_drift: Mapping[str, float] | None = None,
    checkpoint_tolerance: Mapping[str, float] | None = None,
    checkpoint_pass: Mapping[str, bool] | None = None,
) -> tuple[R.SearchResult, FakeAdapter]:
    adapter = FakeAdapter(
        initial,
        objective=objective,
        byte_function=byte_function,
        margin_error=margin_error,
        score_fault=score_fault,
        price_fault=price_fault,
        materialize_fault=materialize_fault,
        commit_fault=commit_fault,
        checkpoint_drift=checkpoint_drift,
        checkpoint_tolerance=checkpoint_tolerance,
        checkpoint_pass=checkpoint_pass,
    )
    result = R.run_coordinate_search(
        initial,
        alphabet=alphabet,  # type: ignore[arg-type]
        cap_bytes=cap_bytes,
        adapter=adapter,
        limits=limits or _limits(),
    )
    return result, adapter


def test_lattice_helpers_use_endpoints_lower_ties_and_detached_arm_starts() -> None:
    assert R.uniform_lattice(4)[:3] == (0, 4, 8)
    assert R.uniform_lattice(4)[-2:] == (252, 255)
    source = np.array([[1, 2, 254], [255, 0, 128]], dtype=np.uint8)
    mapped = R.map_to_lattice(source, (0, 2, 4, 255))
    np.testing.assert_array_equal(
        mapped,
        np.array([[0, 2, 255], [255, 0, 4]], dtype=np.uint8),
    )
    np.testing.assert_array_equal(
        R.independent_step3_map(np.array([[1, 2, 254]], dtype=np.uint8)),
        np.array([[0, 3, 255]], dtype=np.uint8),
    )

    coarse = R.make_arm_start(source, mode="coarse")
    direct = R.make_arm_start(source, mode="direct_fine")
    nested = R.make_arm_start(source, mode="nested_fine", coarse_terminal=coarse)
    np.testing.assert_array_equal(coarse, direct)
    np.testing.assert_array_equal(nested, coarse)
    nested[0, 0] = 255
    assert coarse[0, 0] != 255

    with pytest.raises(R.CoordinateRDOError, match="requires a coarse terminal"):
        R.make_arm_start(source, mode="nested_fine")
    with pytest.raises(R.CoordinateRDOError, match="outside L3"):
        R.make_arm_start(
            source,
            mode="nested_fine",
            coarse_terminal=np.array([[1, 3, 255], [255, 0, 129]], dtype=np.uint8),
        )


def test_uniform_lattice_construction_and_frozen_selection_ties() -> None:
    source = np.array([[1, 2, 254]], dtype=np.uint8)
    states = R.build_uniform_lattice_states(source)
    assert tuple(item.step for item in states) == R.FROZEN_UNIFORM_STEPS
    assert len({item.state_digest for item in states}) < len(states)
    assert all(not item.state.flags.writeable for item in states)

    evaluations = [
        R.UniformLatticeEvaluation(
            step=step,
            state_digest=hashlib.sha256(f"state-{step}".encode()).hexdigest(),
            objective=0.0 if step in {2, 3, 4} else 1.0,
            complete_bytes=9 if step in {2, 3} else 10,
            blob_sha256=hashlib.sha256(f"blob-{step}".encode()).hexdigest(),
        )
        for step in R.FROZEN_UNIFORM_STEPS
    ]
    assert R.select_uniform_lattice(evaluations, cap_bytes=10).step == 2
    assert R.select_uniform_lattice(evaluations, cap_bytes=8) is None
    with pytest.raises(R.CoordinateRDOError, match="exact frozen step menu"):
        R.select_uniform_lattice(evaluations[:-1], cap_bytes=10)


def test_strict_improvement_uses_tolerance_and_rejects_negative_sse() -> None:
    assert not R.strictly_improves(1.0 - 1e-12, 1.0)
    assert R.strictly_improves(1.0 - 2e-12, 1.0)
    assert not R.strictly_improves(1_000_000.0 - 1e-6, 1_000_000.0)
    assert R.strictly_improves(1_000_000.0 - 2e-6, 1_000_000.0)
    with pytest.raises(R.CoordinateRDOError, match="nonnegative"):
        R.strictly_improves(-1.0, 1.0)


def test_l3_scores_exact_local_and_full_labels_but_never_prices_nonimprovers() -> None:
    initial = np.array([[0, 255, 3]], dtype=np.uint8)
    result, adapter = _run(initial, lambda _state: 0.0, alphabet="L3")

    assert result.status == "core_terminal"
    assert result.core_terminal
    assert not result.promotable
    local, audit = result.ledger["passes"]
    assert local["pass_kind"] == "local_sweep"
    assert local["proposal_count"] == 4
    assert audit["pass_kind"] == "full_label_audit"
    assert audit["proposal_count"] == 3 * (len(R.L3) - 1)
    assert [request.replacement_value for request in adapter.score_requests[:4]] == [3, 252, 0, 6]
    assert not adapter.price_requests
    assert [request.purpose for request in adapter.materialization_requests] == [
        "initial",
        "sweep_end",
        "terminal",
    ]
    certificate = result.ledger["terminal_certificate"]
    assert certificate["complete_row_channel_traversal"]
    assert certificate["every_other_active_label_objective_scored"]
    assert certificate["every_decision_relevant_label_priced_or_soundly_pruned"]
    assert certificate["terminal_kind"] == "epsilon_core_terminal"


def test_l1_local_union_and_full_audit_score_every_label() -> None:
    initial = np.array([[0, 2, 255]], dtype=np.uint8)
    result, adapter = _run(initial, lambda _state: 0.0)

    assert result.status == "core_terminal"
    assert result.ledger["passes"][0]["proposal_count"] == 7
    assert [request.replacement_value for request in adapter.score_requests[:7]] == [
        1,
        3,
        1,
        3,
        5,
        252,
        254,
    ]
    assert result.ledger["passes"][1]["proposal_count"] == 3 * 255
    assert len(adapter.score_requests) == 7 + 3 * 255
    assert result.ledger["counters"]["rate_prices"] == 0


def test_exact_rate_pricing_is_lazy_and_prices_an_entire_objective_tie() -> None:
    initial = np.array([[1, 0, 0]], dtype=np.uint8)
    objective = _first_channel_objective({0: 1.0, 1: 10.0, 2: 1.0, 4: 2.0}, default=20.0)

    def bytes_for_state(state: np.ndarray) -> int:
        return {0: 301, 2: 299}.get(int(state[0, 0]), 300)

    result, adapter = _run(initial, objective, cap_bytes=300, byte_function=bytes_for_state)

    assert result.status == "core_terminal"
    assert result.ledger["accepted_edits"][0]["replacement_value"] == 2
    origin_prices = [
        request.replacement_value
        for request in adapter.price_requests
        if request.epoch == 0 and request.row == 0 and request.channel == 0
    ]
    assert origin_prices == [0, 2]
    assert 4 not in origin_prices
    transcript = result.ledger["proposal_transcript"]
    assert transcript["disposition_counts"]["dominated_by_feasible_better_objective_unpriced"] > 0


def test_objective_levels_are_priced_best_first_until_first_feasible_level() -> None:
    initial = np.array([[1, 0, 0]], dtype=np.uint8)
    objective = _first_channel_objective({0: 0.0, 1: 10.0, 2: 1.0, 4: 2.0}, default=20.0)

    def bytes_for_state(state: np.ndarray) -> int:
        return {0: 301, 2: 299}.get(int(state[0, 0]), 300)

    result, adapter = _run(initial, objective, cap_bytes=300, byte_function=bytes_for_state)

    assert result.status == "core_terminal"
    assert result.ledger["accepted_edits"][0]["replacement_value"] == 2
    assert [
        request.replacement_value
        for request in adapter.price_requests
        if request.epoch == 0 and request.row == 0 and request.channel == 0
    ] == [0, 2]


def test_common_epsilon_ambiguous_levels_are_priced_never_committed_and_prune() -> None:
    initial = np.array([[1, 0, 0]], dtype=np.uint8)
    objective = _first_channel_objective({0: 9.0, 1: 10.0, 2: 9.5, 4: 9.8}, default=12.0)

    def error(
        request: R.ObjectiveProposalRequest,
        _candidate: np.ndarray,
        _objective: float,
    ) -> float:
        if request.row == 0 and request.channel == 0:
            return 2.0 if request.replacement_value in {0, 2, 4} else 0.0
        return 0.0

    result, adapter = _run(initial, objective, margin_error=error)

    assert result.status == "core_terminal"
    assert not adapter.commit_requests
    assert {
        request.replacement_value
        for request in adapter.price_requests
        if request.row == 0 and request.channel == 0
    } == {0}
    transcript = result.ledger["proposal_transcript"]
    assert transcript["disposition_counts"]["ambiguous_feasible_noncommittable"] > 0
    assert transcript["disposition_counts"]["dominated_by_feasible_better_objective_unpriced"] > 0
    assert result.ledger["counters"]["feasible_ambiguous_levels"] > 0
    assert result.ledger["terminal_certificate"]["ambiguous_candidates_are_never_committed"]


def test_both_common_epsilon_boundaries_are_ambiguous() -> None:
    initial = np.array([[1, 0, 0]], dtype=np.uint8)
    boundary = 10.0 - R.OBJECTIVE_RELATIVE_TOLERANCE * 10.0
    objective = _first_channel_objective(
        {0: boundary - 2.0, 1: 10.0, 2: boundary + 2.0},
        default=20.0,
    )

    def error(
        request: R.ObjectiveProposalRequest,
        _candidate: np.ndarray,
        _objective: float,
    ) -> float:
        return (
            2.0
            if request.row == 0 and request.channel == 0 and request.replacement_value in {0, 2}
            else 0.0
        )

    result, adapter = _run(initial, objective, margin_error=error)

    assert result.status == "core_terminal"
    assert result.ledger["passes"][0]["objective_class_counts"]["ambiguous"] == 2
    assert not adapter.commit_requests
    assert 0 in [request.replacement_value for request in adapter.price_requests]
    assert 2 not in [
        request.replacement_value
        for request in adapter.price_requests
        if request.row == 0 and request.channel == 0
    ]


def test_cap_blocked_improvements_receive_full_label_certificate() -> None:
    initial = np.zeros((1, 3), dtype=np.uint8)

    def bytes_for_state(state: np.ndarray) -> int:
        return 300 if np.array_equal(state, initial) else 301

    result, adapter = _run(
        initial,
        _squared_objective(np.array([[10, 0, 0]], dtype=np.uint8)),
        cap_bytes=300,
        byte_function=bytes_for_state,
    )

    assert result.status == "core_terminal"
    np.testing.assert_array_equal(result.state, initial)
    assert not adapter.commit_requests
    assert result.ledger["terminal_certificate"]["proposal_count"] == 3 * 255
    assert result.ledger["proposal_transcript"]["disposition_counts"]["infeasible_cap"] > 0


def test_gauss_seidel_reaches_target_without_per_winner_materialization() -> None:
    initial = np.zeros((1, 3), dtype=np.uint8)
    target = np.array([[7, 0, 0]], dtype=np.uint8)
    result, adapter = _run(initial, _squared_objective(target))

    assert result.status == "core_terminal"
    np.testing.assert_array_equal(result.state, target)
    assert [edit["replacement_value"] for edit in result.ledger["accepted_edits"]] == [3, 6, 7]
    assert result.ledger["counters"]["sweeps"] == 4
    assert result.ledger["counters"]["commits"] == 3
    assert len(adapter.commit_requests) == 3
    assert adapter.epoch == 3
    np.testing.assert_array_equal(adapter.state, result.state)
    assert [request.purpose for request in adapter.materialization_requests] == [
        "initial",
        "sweep_end",
        "sweep_end",
        "sweep_end",
        "sweep_end",
        "terminal",
    ]


def test_local_zero_is_not_terminal_and_full_label_audit_can_escape() -> None:
    initial = np.zeros((1, 3), dtype=np.uint8)
    objective = _first_channel_objective({0: 1.0, 1: 2.0, 3: 2.0, 10: 0.0}, default=4.0)
    result, adapter = _run(initial, objective)

    assert result.status == "core_terminal"
    assert int(result.state[0, 0]) == 10
    assert [item["pass_kind"] for item in result.ledger["passes"]] == [
        "local_sweep",
        "full_label_audit",
        "local_sweep",
        "full_label_audit",
    ]
    first_local, first_audit = result.ledger["passes"][:2]
    assert first_local["status"] == "local_zero_requires_full_label_audit"
    assert first_local["local_zero_certificate"]["terminal"] is False
    assert first_audit["status"] == "accepted_jump_resume_local"
    assert first_audit["complete_full_traversal"] is False
    assert len(adapter.commit_requests) == 1
    assert [request.purpose for request in adapter.materialization_requests] == [
        "initial",
        "sweep_end",
        "audit_jump",
        "sweep_end",
        "terminal",
    ]


def test_every_commit_has_one_accepted_transcript_row_and_paired_tokens() -> None:
    initial = np.zeros((1, 3), dtype=np.uint8)
    result, adapter = _run(initial, _squared_objective(np.array([[7, 0, 0]], dtype=np.uint8)))

    accepted = result.ledger["accepted_edits"]
    transcript = result.ledger["proposal_transcript"]
    assert len(adapter.commit_requests) == len(accepted)
    assert result.ledger["counters"]["commit_attempts"] == len(accepted)
    assert result.ledger["counters"]["commits"] == len(accepted)
    assert transcript["disposition_counts"]["accepted"] == len(accepted)
    assert "chunks" not in transcript
    assert all(edit["paired_commit"] for edit in accepted)
    assert "visited_state_digests" not in result.ledger
    assert result.ledger["visited_state_count"] == len(accepted) + 1
    assert all(
        "accepted_state_digests" not in search_pass for search_pass in result.ledger["passes"]
    )


def test_transcript_is_deterministic_and_contains_no_raw_proposal_rows() -> None:
    initial = np.zeros((2, 3), dtype=np.uint8)
    first, _ = _run(initial, lambda _state: 0.0)
    second, _ = _run(initial, lambda _state: 0.0)
    first_transcript = first.ledger["proposal_transcript"]
    second_transcript = second.ledger["proposal_transcript"]

    assert first_transcript["proposal_count"] == 2 * 3 * 2 + 2 * 3 * 255
    assert (
        first_transcript["ordered_proposals_sha256"]
        == second_transcript["ordered_proposals_sha256"]
    )
    assert (
        first_transcript["aggregate_chunks_sha256"] == second_transcript["aggregate_chunks_sha256"]
    )
    assert first_transcript["chunk_count"] == second_transcript["chunk_count"]
    assert "chunks" not in first_transcript
    unsealed = dict(first_transcript)
    digest = unsealed.pop("proposal_transcript_sha256")
    assert hashlib.sha256(R.canonical_json_bytes(unsealed)).hexdigest() == digest


def test_million_proposal_transcript_has_bounded_summary_memory() -> None:
    transcript = R.ProposalTranscript(chunk_size=R.PROPOSAL_TRANSCRIPT_CHUNK_SIZE)
    record = {
        "pass_index": 0,
        "pass_kind": "local_sweep",
        "row": 0,
        "channel": 0,
        "origin_epoch": 0,
        "origin_state_digest": "1" * 64,
        "current_value": 0,
        "replacement_value": 1,
        "objective_float_hex": (1.0).hex(),
        "objective_margin_float_hex": (-1e-12).hex(),
        "margin_error_float_hex": (0.0).hex(),
        "common_coordinate_epsilon_float_hex": (0.0).hex(),
        "objective_class": "nonimproving",
        "rate_priced": False,
        "rate_query_index": None,
        "complete_bytes": None,
        "component_record_sha256": None,
        "byte_margin": None,
        "rate_query_class": "definitely_nonimproving_unpriced",
        "feasibility_class": "not_queried",
        "candidate_state_digest": "0" * 64,
        "disposition": "not_strictly_improving_unpriced",
    }
    for _ in range(1_000_000):
        transcript.add(record)
    summary = transcript.finish()

    assert summary["proposal_count"] == 1_000_000
    assert summary["chunk_count"] == math.ceil(1_000_000 / R.PROPOSAL_TRANSCRIPT_CHUNK_SIZE)
    assert _deep_size(summary) < 20_000
    assert "chunks" not in summary


@pytest.mark.parametrize(
    ("fault_location", "fault", "expected_reason"),
    [
        ("score", "stale", "stale_proposal"),
        ("score", "negative", "invalid_proposal"),
        ("score", "bad_digest", "invalid_proposal"),
        ("price", "stale", "stale_proposal"),
        ("price", "bad_components", "invalid_proposal"),
        ("price", "bad_sum", "invalid_proposal"),
        ("price", "bad_prefix", "invalid_proposal"),
        ("price", "zero_payload", "invalid_proposal"),
        ("price", "bad_order", "invalid_proposal"),
        ("commit", "stale", "stale_proposal"),
        ("commit", "bad_objective", "invalid_proposal"),
        ("commit", "bad_components", "invalid_proposal"),
    ],
)
def test_stale_negative_or_tampered_proposals_fail_closed(
    fault_location: str,
    fault: str,
    expected_reason: str,
) -> None:
    initial = np.zeros((1, 3), dtype=np.uint8)
    result, _adapter = _run(
        initial,
        _squared_objective(np.array([[3, 0, 0]], dtype=np.uint8)),
        score_fault=fault if fault_location == "score" else None,
        price_fault=fault if fault_location == "price" else None,
        commit_fault=fault if fault_location == "commit" else None,
    )

    assert result.status == "invalid_no_decision"
    assert result.failure_reason == expected_reason
    assert not result.promotable
    assert result.ledger["terminal_certificate"] is None


@pytest.mark.parametrize("fault", ["stale", "bad_components", "bad_error"])
def test_tampered_checkpoint_evidence_fails_closed(fault: str) -> None:
    initial = np.zeros((1, 3), dtype=np.uint8)
    result, adapter = _run(
        initial,
        lambda _state: 0.0,
        materialize_fault=fault,
    )

    assert result.status == "invalid_no_decision"
    assert result.failure_reason in {"stale_proposal", "invalid_proposal"}
    assert len(adapter.materialization_requests) == 1
    assert not adapter.score_requests


def test_partial_external_commit_poisoning_never_mutates_core_state() -> None:
    initial = np.zeros((1, 3), dtype=np.uint8)
    result, adapter = _run(
        initial,
        _squared_objective(np.array([[3, 0, 0]], dtype=np.uint8)),
        commit_fault="partial_raise",
    )

    assert result.status == "invalid_no_decision"
    assert result.failure_reason == "invalid_proposal"
    np.testing.assert_array_equal(result.state, initial)
    assert adapter.epoch == 1
    assert not np.array_equal(adapter.state, initial)
    assert result.ledger["counters"]["commit_attempts"] == 1
    assert result.ledger["counters"]["commits"] == 0
    assert result.ledger["failure_query"]["phase"] == "commit_paired_winner"


def test_checkpoint_reconciliation_accepts_bounded_drift_and_rebases_terminal() -> None:
    initial = np.zeros((1, 3), dtype=np.uint8)
    result, _adapter = _run(
        initial,
        lambda _state: 0.0,
        checkpoint_drift={"terminal": 1e-13},
        checkpoint_tolerance={"terminal": 1e-12},
    )

    assert result.status == "core_terminal"
    checkpoint = result.ledger["passes"][-1]["checkpoint"]
    reconciliation = checkpoint["objective_reconciliation"]
    assert reconciliation["passed"]
    assert reconciliation["rebased"]
    assert reconciliation["objective_abs_error"]["value"] == 1e-13
    assert result.ledger["terminal_certificate"]["objective"]["value"] == 1e-13


def test_checkpoint_reconciliation_outside_bound_is_invalid() -> None:
    initial = np.zeros((1, 3), dtype=np.uint8)
    result, _adapter = _run(
        initial,
        lambda _state: 0.0,
        checkpoint_drift={"terminal": 1e-3},
        checkpoint_tolerance={"terminal": 1e-4},
    )

    assert result.status == "invalid_no_decision"
    assert result.failure_reason == "objective_reconciliation_failed"


def test_negative_initial_full_objective_is_invalid() -> None:
    initial = np.zeros((1, 3), dtype=np.uint8)
    result, adapter = _run(initial, lambda _state: -1.0)

    assert result.status == "invalid_no_decision"
    assert result.failure_reason == "invalid_proposal"
    assert len(adapter.materialization_requests) == 1
    assert not adapter.score_requests


def test_cycle_is_detected_before_external_commit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    constant_digest = "0" * 64
    monkeypatch.setattr(R, "state_digest", lambda _state: constant_digest)
    initial = np.zeros((1, 3), dtype=np.uint8)
    result, adapter = _run(
        initial,
        _squared_objective(np.array([[3, 0, 0]], dtype=np.uint8)),
    )

    assert result.status == "invalid_no_decision"
    assert result.failure_reason == "cycle_detected"
    assert not adapter.commit_requests
    np.testing.assert_array_equal(result.state, initial)


def test_deterministic_resource_cap_is_valid_nonconvergence() -> None:
    initial = np.zeros((1, 3), dtype=np.uint8)
    result, adapter = _run(
        initial,
        _squared_objective(np.array([[3, 0, 0]], dtype=np.uint8)),
        limits=_limits(max_proposals=0),
    )

    assert result.status == "valid_nonconvergence"
    assert result.failure_reason == "resource_limit"
    assert result.ledger["failure_detail"] == "max_proposals"
    assert not result.promotable
    assert not adapter.commit_requests


def test_initial_infeasibility_is_valid_fixed_method_failure() -> None:
    initial = np.zeros((1, 3), dtype=np.uint8)
    result, adapter = _run(
        initial,
        lambda _state: 0.0,
        cap_bytes=299,
        byte_function=lambda _state: 300,
    )

    assert result.status == "valid_method_failure"
    assert result.failure_reason == "initial_state_over_cap"
    assert result.ledger["passes"] == []
    assert not adapter.score_requests
    assert not adapter.price_requests


def test_terminal_matches_independent_exhaustive_stationarity_check() -> None:
    initial = np.array([[0, 255, 0]], dtype=np.uint8)
    target = np.array([[12, 240, 30]], dtype=np.uint8)

    def byte_function(state: np.ndarray) -> int:
        return 300 + int(np.count_nonzero(state > 200))

    objective = _squared_objective(target)
    result, _adapter = _run(
        initial,
        objective,
        alphabet="L3",
        cap_bytes=302,
        byte_function=byte_function,
    )

    assert result.status == "core_terminal"
    terminal_objective = objective(result.state)
    for row in range(result.state.shape[0]):
        for channel in range(3):
            for replacement in R.L3:
                if replacement == int(result.state[row, channel]):
                    continue
                candidate = result.state.copy()
                candidate[row, channel] = replacement
                assert byte_function(candidate) > 302 or not R.strictly_improves(
                    objective(candidate), terminal_objective
                )


def test_static_validation_rejects_wrong_state_and_incomplete_adapter() -> None:
    valid = np.zeros((1, 3), dtype=np.uint8)
    adapter = FakeAdapter(valid, objective=lambda _state: 0.0)
    with pytest.raises(R.CoordinateRDOError, match=r"dtype uint8.*\[N,3\]"):
        R.run_coordinate_search(
            valid.astype(np.int64),
            alphabet="L1",
            cap_bytes=300,
            adapter=adapter,
            limits=_limits(),
        )

    class IncompleteAdapter:
        score = adapter.score
        price = adapter.price
        materialize = adapter.materialize

    with pytest.raises(R.CoordinateRDOError, match="score/price/materialize/commit"):
        R.run_coordinate_search(
            valid,
            alphabet="L1",
            cap_bytes=300,
            adapter=IncompleteAdapter(),  # type: ignore[arg-type]
            limits=_limits(),
        )
