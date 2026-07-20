from __future__ import annotations

import json
import struct
import zlib
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from benchmarks import ssp2e_arithmetic as arithmetic
from benchmarks import ssp2f_coordinate_rdo as coordinate
from benchmarks import ssp2f_rdo_adapter as adapter_module
from benchmarks import ssp2f_rdo_objective as objective_module
from benchmarks import ssp2f_size_oracle as size_module


def _pack(values: np.ndarray, bits: int) -> bytes:
    dtype = np.dtype("u1" if bits <= 8 else "<u2" if bits <= 16 else "<u4")
    return np.asarray(values, dtype=dtype).tobytes(order="C")


def _planar(values: np.ndarray, bits: int) -> bytes:
    dtype = np.dtype("u1" if bits <= 8 else "<u2" if bits <= 16 else "<u4")
    raw = np.asarray(values, dtype=dtype).tobytes(order="C")
    if dtype.itemsize == 1:
        return raw
    return np.frombuffer(raw, dtype=np.uint8).reshape(-1, dtype.itemsize).T.ravel().tobytes()


def _deltas(values: np.ndarray, bits: int) -> np.ndarray:
    preceding = np.vstack((np.zeros((1, values.shape[1]), dtype=np.uint32), values[:-1]))
    return (values - preceding) & ((1 << bits) - 1)


def _sspl1(rgb: np.ndarray) -> bytes:
    colors = np.asarray(rgb, dtype=np.uint32)
    if colors.ndim != 2 or colors.shape[1] != 3 or np.any(colors >= 256):
        raise ValueError("test RGB must be uint8-representable [N,3]")
    n = int(colors.shape[0])
    bits = (12, 6, 6, 8)
    bm, bs, br, bc = bits
    row = np.arange(n, dtype=np.uint32)
    means = np.column_stack(
        (
            (29 * row + 3) % (1 << bm),
            (47 * row[::-1] + 5) % (1 << bm),
        )
    ).astype(np.uint32)
    scales = np.column_stack((row % (1 << bs), (3 * row + 2) % (1 << bs))).astype(
        np.uint32
    )
    rotation = ((5 * row + 1) % (1 << br)).astype(np.uint32)
    raw_streams = (
        _planar(_deltas(means, bm).T.ravel(), bm),
        _pack(scales.T.ravel(), bs),
        _pack(rotation, br),
        _pack(_deltas(colors, bc).T.ravel(), bc),
    )
    header = {
        "n": n,
        "H": n,
        "W": 1,
        "bits": list(bits),
        "morton": True,
        "color_delta": True,
        "means_byte_planar": True,
        "rot_circular": True,
        "stream_codec": "zlib",
        "stream_raw_lengths": [len(raw) for raw in raw_streams],
        "color_lo": [0.0, 0.0, 0.0],
        "color_hi": [1.0, 1.0, 1.0],
        "scale_lo": [-3.0, -2.0],
        "scale_hi": [2.0, 3.0],
        "means_lo": [0.0, 0.0],
        "means_hi": [1.0, float(max(1, n - 1))],
        "has_opacity": False,
        "bits_opacity": 8,
        "renderer": "cuda",
        "aa_dilation": 0.0,
        "sigma_cutoff": 3.0,
        "support_fade": False,
        "render_chunk": 512,
    }
    header_raw = json.dumps(header).encode("utf-8")
    blob = b"SSPL1" + struct.pack("<I", len(header_raw)) + header_raw
    for raw in raw_streams:
        payload = zlib.compress(raw, level=9)
        blob += struct.pack("<I", len(payload)) + payload
    return blob


@pytest.fixture(scope="module")
def native_coder(tmp_path_factory: pytest.TempPathFactory) -> arithmetic.NativeArithmeticCoder:
    path = Path(tmp_path_factory.mktemp("ssp2f-adapter-native")) / "libssp2e_range.so"
    return arithmetic.NativeArithmeticCoder(arithmetic.build_native(path))


def _make_engines(
    native_coder: arithmetic.NativeArithmeticCoder,
    *,
    rgb: np.ndarray | None = None,
) -> tuple[objective_module.SSP2FRGBObjective, size_module.SSP2FSizeOracle]:
    symbols = (
        np.asarray(
            [
                [30, 60, 90],
                [45, 75, 105],
                [15, 120, 180],
            ],
            dtype=np.uint8,
        )
        if rgb is None
        else np.ascontiguousarray(rgb, dtype=np.uint8)
    )
    objective = objective_module.SSP2FRGBObjective(
        np.eye(len(symbols), dtype=np.float64),
        symbols,
        np.zeros((len(symbols), 3), dtype=np.uint8),
        reference_kind="target",
        color_lo=[0.0, 0.0, 0.0],
        color_hi=[1.0, 1.0, 1.0],
    )
    size_oracle = size_module.SSP2FSizeOracle(_sspl1(symbols), native_coder)
    return objective, size_oracle


def _make_adapter(
    native_coder: arithmetic.NativeArithmeticCoder,
    tmp_path: Path,
    *,
    rgb: np.ndarray | None = None,
    name: str = "paired.journal",
) -> tuple[
    adapter_module.SSP2FPairedRDOAdapter,
    objective_module.SSP2FRGBObjective,
    size_module.SSP2FSizeOracle,
]:
    objective, size_oracle = _make_engines(native_coder, rgb=rgb)
    adapter = adapter_module.SSP2FPairedRDOAdapter(
        objective,
        size_oracle,
        journal_path=tmp_path / name,
    )
    return adapter, objective, size_oracle


def _score_request(
    adapter: adapter_module.SSP2FPairedRDOAdapter,
    state: np.ndarray,
    *,
    row: int = 0,
    channel: int = 0,
    replacement: int = 0,
) -> coordinate.ObjectiveProposalRequest:
    return coordinate.ObjectiveProposalRequest(
        epoch=adapter.epoch,
        origin_state_digest=adapter.state_digest,
        row=row,
        channel=channel,
        current_value=int(state[row, channel]),
        replacement_value=replacement,
    )


def _score_and_price(
    adapter: adapter_module.SSP2FPairedRDOAdapter,
    state: np.ndarray,
    *,
    row: int = 0,
    channel: int = 0,
    replacement: int = 0,
) -> tuple[coordinate.ObjectiveProposal, coordinate.RateProposal]:
    scored = adapter.score(
        _score_request(
            adapter,
            state,
            row=row,
            channel=channel,
            replacement=replacement,
        )
    )
    priced = adapter.price(
        coordinate.RateProposalRequest(
            epoch=scored.epoch,
            origin_state_digest=scored.origin_state_digest,
            candidate_state_digest=scored.candidate_state_digest,
            row=scored.row,
            channel=scored.channel,
            current_value=scored.current_value,
            replacement_value=scored.replacement_value,
            objective=scored.objective,
            objective_commit_token=scored.commit_token,
        )
    )
    return scored, priced


def _winner(
    scored: coordinate.ObjectiveProposal,
    priced: coordinate.RateProposal,
    state: np.ndarray,
) -> coordinate.WinnerCommitRequest:
    candidate = state.copy()
    candidate[scored.row, scored.channel] = scored.replacement_value
    return coordinate.WinnerCommitRequest(
        origin_epoch=scored.epoch,
        accepted_epoch=scored.epoch + 1,
        origin_state_digest=scored.origin_state_digest,
        candidate_state_digest=scored.candidate_state_digest,
        row=scored.row,
        channel=scored.channel,
        current_value=scored.current_value,
        replacement_value=scored.replacement_value,
        objective=scored.objective,
        complete_bytes=priced.complete_bytes,
        component_bytes=dict(priced.component_bytes),
        objective_commit_token=scored.commit_token,
        rate_commit_token=priced.commit_token,
        candidate_state=candidate,
    )


def _initial_materialization(
    adapter: adapter_module.SSP2FPairedRDOAdapter,
    state: np.ndarray,
) -> coordinate.MaterializedState:
    return adapter.materialize(
        coordinate.MaterializationRequest(
            purpose="initial",
            epoch=0,
            origin_state_digest=None,
            state_digest=coordinate.state_digest(state),
            expected_objective=None,
            expected_complete_bytes=None,
            state=state,
        )
    )


def _assert_poisoned_methods(
    adapter: adapter_module.SSP2FPairedRDOAdapter,
    score_request: coordinate.ObjectiveProposalRequest,
) -> None:
    assert adapter.poisoned
    with pytest.raises(adapter_module.SSP2FPairedAdapterPoisoned):
        adapter.score(score_request)
    with pytest.raises(adapter_module.SSP2FPairedAdapterPoisoned):
        adapter.price(None)  # type: ignore[arg-type]
    with pytest.raises(adapter_module.SSP2FPairedAdapterPoisoned):
        adapter.commit(None)  # type: ignore[arg-type]
    with pytest.raises(adapter_module.SSP2FPairedAdapterPoisoned):
        adapter.materialize(None)  # type: ignore[arg-type]


def test_real_objective_rate_commit_and_materialization_are_exact_and_journaled(
    native_coder: arithmetic.NativeArithmeticCoder,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter, objective, size_oracle = _make_adapter(native_coder, tmp_path)
    state = objective.rgb_symbols
    initial = _initial_materialization(adapter, state)
    assert initial.state_digest == coordinate.state_digest(state)
    assert initial.objective_reconciliation_passed
    assert initial.objective_abs_error == 0.0
    assert initial.complete_bytes == len(size_oracle.encode_and_verify())
    assert set(initial.component_bytes) == set(coordinate.SSP2F_COMPONENT_KEYS)
    assert initial.component_bytes["prefix"] == 280
    assert initial.component_bytes["total"] == initial.complete_bytes
    assert sum(
        initial.component_bytes[key]
        for key in coordinate.SSP2F_COMPONENT_KEYS
        if key != "total"
    ) == initial.complete_bytes
    assert adapter.last_reconciliation is not None
    assert adapter.last_reconciliation.passed

    scored, priced = _score_and_price(adapter, state)
    assert objective.epoch == size_oracle.epoch == 0
    assert scored.objective < initial.objective
    assert set(priced.component_bytes) == set(coordinate.SSP2F_COMPONENT_KEYS)
    assert priced.component_bytes["prefix"] == 280
    request = _winner(scored, priced, state)
    commit_calls = {"objective": 0, "size": 0}
    objective_commit = objective.commit
    size_commit = size_oracle.commit

    def counted_objective_commit(proposal: Any) -> float:
        commit_calls["objective"] += 1
        return objective_commit(proposal)

    def counted_size_commit(proposal: Any) -> int:
        commit_calls["size"] += 1
        return size_commit(proposal)

    monkeypatch.setattr(objective, "commit", counted_objective_commit)
    monkeypatch.setattr(size_oracle, "commit", counted_size_commit)
    receipt = adapter.commit(request)
    assert commit_calls == {"objective": 1, "size": 1}
    assert receipt.accepted_epoch == 1
    assert receipt.candidate_state_digest == coordinate.state_digest(request.candidate_state)
    assert receipt.objective.hex() == scored.objective.hex()
    assert receipt.complete_bytes == priced.complete_bytes
    assert receipt.component_bytes == priced.component_bytes
    assert objective.epoch == size_oracle.epoch == adapter.epoch == 1
    np.testing.assert_array_equal(objective.rgb_symbols, request.candidate_state)
    np.testing.assert_array_equal(
        size_oracle.rgb_symbols.astype(np.uint8),
        request.candidate_state,
    )

    checkpoint = adapter.materialize(
        coordinate.MaterializationRequest(
            purpose="sweep_end",
            epoch=1,
            origin_state_digest=receipt.candidate_state_digest,
            state_digest=receipt.candidate_state_digest,
            expected_objective=receipt.objective,
            expected_complete_bytes=receipt.complete_bytes,
            state=request.candidate_state,
        )
    )
    assert checkpoint.objective_abs_error <= checkpoint.objective_tolerance
    assert checkpoint.complete_bytes == len(size_oracle.encode_and_verify())
    records = [
        json.loads(line)
        for line in adapter.journal_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [record["record"] for record in records] == [
        "journal_start",
        "commit_start",
        "commit_complete",
    ]
    assert records[1]["candidate_state_digest"] == receipt.candidate_state_digest
    assert records[1]["complete_bytes"] == receipt.complete_bytes
    assert records[1]["commit_sha256"] == records[2]["commit_sha256"]
    assert records[2]["state_digest"] == receipt.candidate_state_digest
    assert records[2]["complete_bytes"] == receipt.complete_bytes


def test_price_is_lazy_nonmutating_and_objective_token_is_single_use(
    native_coder: arithmetic.NativeArithmeticCoder,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter, objective, size_oracle = _make_adapter(native_coder, tmp_path)
    state = objective.rgb_symbols
    score_request = _score_request(adapter, state)
    current_objective = objective.objective
    with monkeypatch.context() as patch:
        patch.setattr(
            size_oracle,
            "propose",
            lambda *args, **kwargs: pytest.fail("score invoked the exact-rate oracle"),
        )
        scored = adapter.score(score_request)
    candidate_bound = (
        objective_module.RECONCILIATION_OBJECTIVE_ATOL
        + objective_module.RECONCILIATION_OBJECTIVE_RTOL * abs(scored.objective)
    ) / (1.0 - objective_module.RECONCILIATION_OBJECTIVE_RTOL)
    current_bound = (
        objective_module.RECONCILIATION_OBJECTIVE_ATOL
        + objective_module.RECONCILIATION_OBJECTIVE_RTOL * abs(current_objective)
    ) / (1.0 - objective_module.RECONCILIATION_OBJECTIVE_RTOL)
    exact_margin_error = candidate_bound + (
        1.0 + coordinate.OBJECTIVE_RELATIVE_TOLERANCE
    ) * current_bound
    assert scored.margin_error.hex() == exact_margin_error.hex()
    candidate = state.copy()
    candidate[scored.row, scored.channel] = scored.replacement_value
    assert abs(scored.objective - objective.evaluate(candidate)) <= candidate_bound
    assert objective.epoch == size_oracle.epoch == 0
    rate_request = coordinate.RateProposalRequest(
        epoch=scored.epoch,
        origin_state_digest=scored.origin_state_digest,
        candidate_state_digest=scored.candidate_state_digest,
        row=scored.row,
        channel=scored.channel,
        current_value=scored.current_value,
        replacement_value=scored.replacement_value,
        objective=scored.objective,
        objective_commit_token=scored.commit_token,
    )
    adapter.price(rate_request)
    assert objective.epoch == size_oracle.epoch == 0
    with pytest.raises(adapter_module.SSP2FPairedAdapterPoisoned, match="reused"):
        adapter.price(rate_request)
    _assert_poisoned_methods(adapter, score_request)


def test_journal_failure_poisoning_happens_before_either_engine_commit(
    native_coder: arithmetic.NativeArithmeticCoder,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter, objective, size_oracle = _make_adapter(native_coder, tmp_path)
    state = objective.rgb_symbols
    scored, priced = _score_and_price(adapter, state)
    score_request = _score_request(adapter, state)

    def fail_journal(record: Any) -> None:
        raise OSError("injected fsync failure")

    monkeypatch.setattr(adapter, "_append_journal_record", fail_journal)
    with pytest.raises(adapter_module.SSP2FPairedAdapterPoisoned, match="fsync"):
        adapter.commit(_winner(scored, priced, state))
    assert objective.epoch == size_oracle.epoch == 0
    _assert_poisoned_methods(adapter, score_request)


def test_commit_completion_journal_failure_poisoning_keeps_both_commits_unusable(
    native_coder: arithmetic.NativeArithmeticCoder,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter, objective, size_oracle = _make_adapter(native_coder, tmp_path)
    state = objective.rgb_symbols
    scored, priced = _score_and_price(adapter, state)
    score_request = _score_request(adapter, state)
    append_record = adapter._append_journal_record

    def fail_completion(record: Any) -> None:
        if record["record"] == "commit_complete":
            raise OSError("injected completion fsync failure")
        append_record(record)

    monkeypatch.setattr(adapter, "_append_journal_record", fail_completion)
    with pytest.raises(adapter_module.SSP2FPairedAdapterPoisoned, match="completion fsync"):
        adapter.commit(_winner(scored, priced, state))
    assert objective.epoch == size_oracle.epoch == 1
    records = [
        json.loads(line)
        for line in adapter.journal_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [record["record"] for record in records] == [
        "journal_start",
        "commit_start",
        "poison",
    ]
    assert records[2]["operation"] == "commit"
    _assert_poisoned_methods(adapter, score_request)


def test_one_sided_engine_failure_is_permanent_and_never_rolled_back(
    native_coder: arithmetic.NativeArithmeticCoder,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter, objective, size_oracle = _make_adapter(native_coder, tmp_path)
    state = objective.rgb_symbols
    scored, priced = _score_and_price(adapter, state)
    score_request = _score_request(adapter, state)

    def fail_size_commit(proposal: Any) -> int:
        raise RuntimeError("injected size commit failure")

    monkeypatch.setattr(size_oracle, "commit", fail_size_commit)
    with pytest.raises(adapter_module.SSP2FPairedAdapterPoisoned, match="size commit"):
        adapter.commit(_winner(scored, priced, state))
    assert objective.epoch == 1
    assert size_oracle.epoch == 0
    _assert_poisoned_methods(adapter, score_request)


def test_unknown_external_engine_mutation_is_detected_and_poisoned(
    native_coder: arithmetic.NativeArithmeticCoder,
    tmp_path: Path,
) -> None:
    adapter, objective, size_oracle = _make_adapter(native_coder, tmp_path)
    original = objective.rgb_symbols
    objective.commit(objective.propose(1, 2, 0))
    assert objective.epoch == 1
    assert size_oracle.epoch == 0
    request = _score_request(adapter, original)
    with pytest.raises(adapter_module.SSP2FPairedAdapterPoisoned, match="epochs differ"):
        adapter.score(request)
    _assert_poisoned_methods(adapter, request)


def test_materialization_encode_disagreement_poisoning(
    native_coder: arithmetic.NativeArithmeticCoder,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter, objective, size_oracle = _make_adapter(native_coder, tmp_path)
    state = objective.rgb_symbols
    request = _score_request(adapter, state)
    monkeypatch.setattr(size_oracle, "encode_and_verify", lambda: b"not-an-ssp2f-blob")
    with pytest.raises(adapter_module.SSP2FPairedAdapterPoisoned, match="disagreed"):
        _initial_materialization(adapter, state)
    _assert_poisoned_methods(adapter, request)


def test_malformed_component_schema_and_commit_token_reuse_poison(
    native_coder: arithmetic.NativeArithmeticCoder,
    tmp_path: Path,
) -> None:
    adapter, objective, _ = _make_adapter(native_coder, tmp_path, name="malformed.journal")
    state = objective.rgb_symbols
    scored, priced = _score_and_price(adapter, state)
    malformed = _winner(scored, priced, state)
    bad_components = dict(malformed.component_bytes)
    bad_components.pop("prefix")
    malformed = replace(malformed, component_bytes=bad_components)
    with pytest.raises(adapter_module.SSP2FPairedAdapterPoisoned, match="component keys"):
        adapter.commit(malformed)

    adapter2, objective2, _ = _make_adapter(native_coder, tmp_path, name="reuse.journal")
    state2 = objective2.rgb_symbols
    scored2, priced2 = _score_and_price(adapter2, state2)
    winner = _winner(scored2, priced2, state2)
    adapter2.commit(winner)
    with pytest.raises(adapter_module.SSP2FPairedAdapterPoisoned):
        adapter2.commit(winner)

    adapter3, objective3, _ = _make_adapter(native_coder, tmp_path, name="order.journal")
    state3 = objective3.rgb_symbols
    scored3, priced3 = _score_and_price(adapter3, state3)
    wrong_order = {
        key: priced3.component_bytes[key]
        for key in reversed(coordinate.SSP2F_COMPONENT_KEYS)
    }
    unordered_winner = replace(
        _winner(scored3, priced3, state3),
        component_bytes=wrong_order,
    )
    with pytest.raises(adapter_module.SSP2FPairedAdapterPoisoned, match="component order"):
        adapter3.commit(unordered_winner)


def test_constructor_rejects_state_epoch_mismatch_and_existing_journal(
    native_coder: arithmetic.NativeArithmeticCoder,
    tmp_path: Path,
) -> None:
    objective, size_oracle = _make_engines(native_coder)
    existing = tmp_path / "existing.journal"
    existing.write_text("stale\n", encoding="utf-8")
    with pytest.raises(adapter_module.SSP2FPairedAdapterError, match="fresh durable"):
        adapter_module.SSP2FPairedRDOAdapter(
            objective,
            size_oracle,
            journal_path=existing,
        )

    mismatched_objective, mismatched_size = _make_engines(native_coder)
    mismatched_objective.commit(mismatched_objective.propose(0, 0, 0))
    with pytest.raises(adapter_module.SSP2FPairedAdapterError, match="epochs differ"):
        adapter_module.SSP2FPairedRDOAdapter(
            mismatched_objective,
            mismatched_size,
            journal_path=tmp_path / "epoch-mismatch.journal",
        )

    state_objective, state_size = _make_engines(native_coder)
    state_size._symbols["red"][0] = 0
    with pytest.raises(adapter_module.SSP2FPairedAdapterError, match="RGB states differ"):
        adapter_module.SSP2FPairedRDOAdapter(
            state_objective,
            state_size,
            journal_path=tmp_path / "state-mismatch.journal",
        )


def test_coordinate_search_end_to_end_zero_objective_smoke(
    native_coder: arithmetic.NativeArithmeticCoder,
    tmp_path: Path,
) -> None:
    state = np.zeros((2, 3), dtype=np.uint8)
    adapter, _, size_oracle = _make_adapter(
        native_coder,
        tmp_path,
        rgb=state,
        name="search.journal",
    )
    result = coordinate.run_coordinate_search(
        state,
        alphabet="L3",
        cap_bytes=size_oracle.complete_bytes,
        adapter=adapter,
        limits=coordinate.SearchLimits(
            max_sweeps=1,
            max_proposals=600,
            max_rate_prices=0,
            max_accepted_edits=0,
            max_materializations=3,
        ),
    )
    assert result.status == "core_terminal"
    assert result.core_terminal
    assert result.ledger["counters"]["rate_prices"] == 0
    assert result.ledger["terminal_certificate"]["component_bytes"]["prefix"] == 280
    assert not adapter.poisoned
