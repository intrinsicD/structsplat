from __future__ import annotations

import hashlib
import itertools
import struct
from pathlib import Path

import numpy as np
import pytest

import benchmarks.ssp2v_lloyd as L


@pytest.fixture(scope="session")
def native_fitter(tmp_path_factory: pytest.TempPathFactory) -> L.NativeLloydFitter:
    root = tmp_path_factory.mktemp("ssp2v_lloyd_native")
    return L.NativeLloydFitter(L.build_native(root / "libssp2v_lloyd.so"))


def _assert_start_equal(left: L.StartResult, right: L.StartResult) -> None:
    assert left.spec == right.spec
    assert left.start_id == right.start_id
    assert left.status == right.status
    assert left.update_sweeps == right.update_sweeps
    assert left.terminal_sse == right.terminal_sse
    assert left.trajectory == right.trajectory
    assert np.array_equal(left.codebook, right.codebook)
    assert np.array_equal(left.indices, right.indices)


def test_start_key_and_state_hash_match_frozen_byte_grammars() -> None:
    spec = L.LloydSpec(1, 3, 2, 1, 2)
    key = L.start_key((-7, 8, 9), spec, 3)
    manual = hashlib.sha256(
        b"structsplat-comp011-vq-start-v1\0"
        + struct.pack("<BBHBBiii", 1, 3, 2, 1, 3, -7, 8, 9)
    ).digest()
    assert key == manual
    assert key.hex() == "b38f2b8edad5691027c3e7679b5005b9eadf938539ac6f7cd012acff263459ee"

    codebook = np.asarray(((-1, 2, -3), (4, 5, 6)), dtype=np.int32)
    manual_state = hashlib.sha256(
        b"structsplat-comp011-vq-state-v1\0"
        + struct.pack("<I", 2)
        + codebook.astype("<i4").tobytes()
    ).hexdigest()
    assert L.codebook_state_sha256(codebook) == manual_state
    assert manual_state == "7cb0beffe6c9eaa168d745fed2cd9ee1fa397f5a6da63584483184226aec4ab6"


def test_weighted_mean_is_exact_and_chooses_smaller_half_tie() -> None:
    assert L.round_weighted_mean(1, 2) == 0
    assert L.round_weighted_mean(-1, 2) == -1
    assert L.round_weighted_mean(-3, 2) == -2
    for count in range(1, 9):
        for total in range(-24, 25):
            chosen = L.round_weighted_mean(total, count)
            candidates = range(chosen - 3, chosen + 4)
            # The constant sum(x^2) term can be omitted from the scalar SSE.
            scores = {value: count * value * value - 2 * total * value for value in candidates}
            best_score = min(scores.values())
            assert chosen == min(value for value, score in scores.items() if score == best_score)


def test_unique_histogram_is_lexicographic_exact_and_permutation_invariant() -> None:
    rows = np.asarray(((2, 0, -1), (-1, 4, 3), (2, 0, -1), (0, 0, 0)), dtype=np.int32)
    weighted = L.collapse_weighted(rows)
    assert weighted.vectors.tolist() == [[-1, 4, 3], [0, 0, 0], [2, 0, -1]]
    assert weighted.counts == (1, 1, 2)
    permuted = L.collapse_weighted(rows[[2, 0, 3, 1]])
    assert np.array_equal(weighted.vectors, permuted.vectors)
    assert weighted.counts == permuted.counts


def test_assignment_tie_uses_lowest_canonical_label_and_update_canonicalizes() -> None:
    vectors = np.asarray(((0, 0, 0),), dtype=np.int32)
    codebook = np.asarray(((-1, 0, 0), (1, 0, 0)), dtype=np.int32)
    indices, distances = L._nearest_indices_and_distances(vectors, codebook)
    assert indices.tolist() == [0]
    assert distances.tolist() == [1]

    spec = L.LloydSpec(1, 2, 2, 1, 2)
    weighted = L.collapse_weighted(((-2, 0, 0), (-1, 0, 0), (8, 0, 0)))
    reversed_labels = np.asarray(((9, 0, 0), (0, 0, 0)), dtype=np.int32)
    updated = L._updated_codebook(weighted, reversed_labels, spec)
    assert updated.tolist() == [[-2, 0, 0], [8, 0, 0]]


def test_initialization_u_less_than_k_fills_cyclically_and_dead_centroids_stay() -> None:
    rows = np.asarray(((2, 2, 2), (9, 9, 9), (2, 2, 2)), dtype=np.int32)
    spec = L.LloydSpec(0, 2, 1, 0, 3)
    weighted = L.collapse_weighted(rows)
    initialized = L.initialize_codebook(weighted, spec, 0)
    assert initialized.shape == (3, 3)
    assert {tuple(row) for row in initialized.tolist()} == {(2, 2, 2), (9, 9, 9)}
    result = L.fit_start_reference(rows, spec, 0)
    assert result.converged
    assert result.update_sweeps == 1
    assert np.array_equal(result.codebook, initialized)
    assert len(set(result.indices.tolist())) == 2


def test_fixed_trajectory_repeats_terminal_hash_and_cap_preserves_real_state() -> None:
    rows = np.asarray(((0, 0, 0), (1, 0, 0), (2, 0, 0), (9, 0, 0), (10, 0, 0)))
    spec = L.LloydSpec(1, 2, 2, 0, 2)
    fixed = L.fit_start_reference(rows, spec, 0)
    assert fixed.status == "fixed"
    assert fixed.update_sweeps == 2
    assert fixed.trajectory[-1].codebook_sha256 == fixed.trajectory[-2].codebook_sha256
    assert fixed.trajectory[-1].objective_sse == fixed.terminal_sse == 3

    capped = L.fit_start_reference(rows, spec, 0, max_update_sweeps=1)
    assert capped.status == "cap"
    assert capped.update_sweeps == 1
    assert capped.trajectory[-1].codebook_sha256 == L.codebook_state_sha256(capped.codebook)
    assert capped.trajectory[-1].objective_sse == capped.terminal_sse


def test_cycle_state_machine_fixture_does_not_misclassify_fixed_point() -> None:
    state_a = b"A"
    state_b = b"B"
    assert L._transition_status(state_a, state_a, {state_a}) == "fixed"
    assert L._transition_status(state_b, state_a, {state_a, state_b}) == "cycle"
    assert L._transition_status(state_b, b"C", {state_a, state_b}) == "continue"


def test_cycle_fixture_seals_the_repeated_nonfixed_terminal_state(monkeypatch) -> None:
    state_a = np.asarray(((0, 0, 0),), dtype=np.int32)
    state_b = np.asarray(((1, 0, 0),), dtype=np.int32)
    spec = L.LloydSpec(1, 2, 1, 1, 1)

    monkeypatch.setattr(L, "initialize_codebook", lambda weighted, fit_spec, start_id: state_a)

    def toggle(weighted, codebook, fit_spec):
        del weighted, fit_spec
        return state_b if np.array_equal(codebook, state_a) else state_a

    monkeypatch.setattr(L, "_updated_codebook", toggle)
    result = L.fit_start_reference(((0, 0, 0), (1, 0, 0)), spec, 0)
    assert result.status == "cycle"
    assert result.update_sweeps == 2
    assert result.trajectory[-1].codebook_sha256 == result.trajectory[0].codebook_sha256
    assert np.array_equal(result.codebook, state_a)


def test_every_start_must_converge_before_winner_selection() -> None:
    rows = np.asarray(((0, 0, 0), (1, 0, 0), (2, 0, 0), (9, 0, 0), (10, 0, 0)))
    spec = L.LloydSpec(1, 2, 2, 0, 2)

    class OneSweepFitter:
        def fit_start(self, values, fit_spec, start_id, *, max_update_sweeps=100):
            del max_update_sweeps
            return L.fit_start_reference(values, fit_spec, start_id, max_update_sweeps=1)

    with pytest.raises(L.LloydConvergenceError) as caught:
        L.fit_stage(rows, spec, fitter=OneSweepFitter())
    assert len(caught.value.starts) == 4
    assert all(result.status == "cap" for result in caught.value.starts)


def test_serialization_is_exact_and_rejects_noncanonical_or_unrepresentable() -> None:
    flat = L.LloydSpec(0, 2, 1, 0, 3)
    assert L.serialize_codebook(((0, 1, 2), (3, 4, 5), (255, 255, 255)), flat) == bytes(
        (0, 1, 2, 3, 4, 5, 255, 255, 255)
    )
    residual = L.LloydSpec(1, 2, 1, 1, 1)
    assert L.serialize_codebook(((-1, 256, -32768),), residual) == bytes.fromhex(
        "ffff00010080"
    )
    with pytest.raises(L.LloydError, match="canonical"):
        L.serialize_codebook(((2, 0, 0), (1, 0, 0), (3, 0, 0)), flat)
    with pytest.raises(L.LloydError, match="int16"):
        L.serialize_codebook(((0, 0, 32768),), residual)


def test_winner_key_uses_serialized_bytes_then_numeric_indices_then_start() -> None:
    spec = L.LloydSpec(1, 2, 1, 1, 1)

    def result(value: int, indices: tuple[int, ...], start_id: int) -> L.StartResult:
        codebook = np.asarray(((value, 0, 0),), dtype=np.int32)
        index_array = np.asarray(indices, dtype=np.uint32)
        record = L.TrajectoryRecord(0, 0, L.codebook_state_sha256(codebook), 7)
        return L.StartResult(spec, start_id, "fixed", 1, 7, codebook, index_array, (record,))

    # int16 little-endian byte order is authoritative: 0x0000 sorts before 0xffff.
    assert L._winner_key(result(0, (1,), 3)) < L._winner_key(result(-1, (0,), 0))
    assert L._winner_key(result(0, (0, 2), 3)) < L._winner_key(result(0, (1, 0), 0))
    assert L._winner_key(result(0, (0,), 0)) < L._winner_key(result(0, (0,), 1))


def test_each_start_is_invariant_to_row_permutation_up_to_index_permutation() -> None:
    rows = np.asarray(
        ((-5, 1, 0), (4, 0, 2), (-5, 1, 0), (9, -3, 1), (0, 0, 0), (4, 0, 2)),
        dtype=np.int32,
    )
    permutation = np.asarray((5, 2, 4, 0, 3, 1))
    spec = L.LloydSpec(1, 2, 2, 1, 2)
    for start_id in L.START_IDS:
        original = L.fit_start_reference(rows, spec, start_id)
        permuted = L.fit_start_reference(rows[permutation], spec, start_id)
        assert original.terminal_sse == permuted.terminal_sse
        assert original.trajectory == permuted.trajectory
        assert np.array_equal(original.codebook, permuted.codebook)
        assert np.array_equal(permuted.indices, original.indices[permutation])


def test_native_matches_reference_on_exhaustive_small_signed_multisets(
    native_fitter: L.NativeLloydFitter,
) -> None:
    universe = ((-2, 0, 1), (-1, 1, 0), (0, -1, 2), (2, 0, -2))
    spec = L.LloydSpec(1, 2, 2, 1, 2)
    cases = 0
    for count in range(1, 5):
        for row_indices in itertools.combinations_with_replacement(range(len(universe)), count):
            rows = np.asarray([universe[index] for index in row_indices], dtype=np.int32)
            for start_id in L.START_IDS:
                _assert_start_equal(
                    L.fit_start_reference(rows, spec, start_id),
                    native_fitter.fit_start(rows, spec, start_id),
                )
                cases += 1
    assert cases == 276


def test_native_matches_reference_on_stage_zero_ties_duplicates_and_cap(
    native_fitter: L.NativeLloydFitter,
) -> None:
    rows = np.asarray(
        ((0, 0, 0), (0, 0, 0), (1, 1, 1), (2, 2, 2), (127, 128, 129), (255, 255, 255)),
        dtype=np.int32,
    )
    spec = L.LloydSpec(0, 2, 1, 0, 3)
    for start_id in L.START_IDS:
        _assert_start_equal(
            L.fit_start_reference(rows, spec, start_id),
            native_fitter.fit_start(rows, spec, start_id),
        )

    cap_rows = np.asarray(((0, 0, 0), (1, 0, 0), (2, 0, 0), (9, 0, 0), (10, 0, 0)))
    cap_spec = L.LloydSpec(1, 2, 2, 0, 2)
    _assert_start_equal(
        L.fit_start_reference(cap_rows, cap_spec, 0, max_update_sweeps=1),
        native_fitter.fit_start(cap_rows, cap_spec, 0, max_update_sweeps=1),
    )


def test_checked_domains_reject_distance_overflow_and_later_centroid_overflow(
    native_fitter: L.NativeLloydFitter,
) -> None:
    huge = np.asarray(
        ((L.INT32_MIN, L.INT32_MIN, L.INT32_MIN), (L.INT32_MAX,) * 3), dtype=np.int64
    )
    spec = L.LloydSpec(1, 2, 2, 1, 2)
    with pytest.raises(L.LloydError, match="distance"):
        L.fit_start_reference(huge, spec, 0)
    with pytest.raises(L.LloydError, match="distance"):
        native_fitter.fit_start(huge, spec, 0)

    outside_int16 = np.asarray(((-40000, 0, 0), (40000, 0, 0)), dtype=np.int32)
    with pytest.raises(L.LloydError, match="int16"):
        L.fit_start_reference(outside_int16, spec, 0)
    with pytest.raises(L.LloydError, match="int16"):
        native_fitter.fit_start(outside_int16, spec, 0)


def test_forced_ten_sweeps_is_separate_and_matches_native_without_mutating_science(
    native_fitter: L.NativeLloydFitter,
) -> None:
    rows = np.asarray(((0, 0, 0), (1, 0, 0), (2, 0, 0), (9, 0, 0), (10, 0, 0)))
    spec = L.LloydSpec(1, 2, 2, 0, 2)
    scientific_before = native_fitter.fit_start(rows, spec, 0)
    reference = L.force_sweeps_reference(rows, spec, 0, forced_sweeps=10)
    native = native_fitter.force_sweeps(rows, spec, 0, forced_sweeps=10)
    assert reference.forced_sweeps == native.forced_sweeps == 10
    assert reference.final_codebook_sha256 == native.final_codebook_sha256
    assert np.array_equal(reference.final_codebook, native.final_codebook)
    scientific_after = native_fitter.fit_start(rows, spec, 0)
    _assert_start_equal(scientific_before, scientific_after)
    assert scientific_after.update_sweeps == 2


def test_flat_and_rvq_orchestration_materialize_expected_shapes() -> None:
    rgb = np.asarray(
        ((0, 0, 0), (5, 7, 9), (30, 20, 10), (100, 110, 120), (255, 250, 245)),
        dtype=np.uint8,
    )
    flat = L.fit_flat(rgb, 2, 1, require_protocol_rows=False)
    assert flat.family_id == 0
    assert len(flat.stages) == 1
    assert flat.stages[0].codebook.shape == (3, 3)
    assert flat.reconstructed_rgb.dtype == np.uint8
    assert np.array_equal(
        flat.raw_accumulator, flat.stages[0].codebook[flat.stage_indices[0]]
    )

    rvq = L.fit_rvq(rgb, 2, 1, require_protocol_rows=False)
    assert rvq.family_id == 1
    assert len(rvq.stages) == 2
    expected = sum(
        (stage.codebook[indices].astype(np.int32) for stage, indices in zip(rvq.stages, rvq.stage_indices)),
        start=np.zeros_like(rgb, dtype=np.int32),
    )
    assert np.array_equal(rvq.raw_accumulator, expected)
    assert np.array_equal(rvq.reconstructed_rgb, np.clip(expected, 0, 255).astype(np.uint8))


def test_three_stage_rvq_subtracts_unclipped_accumulator() -> None:
    seen_inputs: dict[int, list[np.ndarray]] = {0: [], 1: [], 2: []}
    centroids = {0: 255, 1: 100, 2: -355}

    class RecordingFitter:
        def fit_start(self, rows, spec, start_id, *, max_update_sweeps=100):
            del max_update_sweeps
            prepared = np.asarray(rows, dtype=np.int32)
            seen_inputs[spec.actual_stage_index].append(prepared.copy())
            codebook = np.asarray(((centroids[spec.actual_stage_index], 0, 0),), dtype=np.int32)
            indices = np.zeros(len(prepared), dtype=np.uint32)
            sse = int(np.square(prepared.astype(np.int64) - codebook[0]).sum())
            digest = L.codebook_state_sha256(codebook)
            trajectory = (
                L.TrajectoryRecord(0, 0, digest, sse),
                L.TrajectoryRecord(1, 1, digest, sse),
            )
            return L.StartResult(spec, start_id, "fixed", 1, sse, codebook, indices, trajectory)

    rgb = np.asarray(((0, 0, 0),), dtype=np.uint8)
    result = L.fit_rvq(
        rgb, 3, 1, fitter=RecordingFitter(), require_protocol_rows=False
    )
    assert all(len(entries) == 4 for entries in seen_inputs.values())
    assert seen_inputs[0][0].tolist() == [[0, 0, 0]]
    assert seen_inputs[1][0].tolist() == [[-255, 0, 0]]
    # Stage one makes the raw accumulator 355.  Clipping it before subtraction
    # would incorrectly produce -255 instead of the frozen -355 residual.
    assert seen_inputs[2][0].tolist() == [[-355, 0, 0]]
    assert result.raw_accumulator.tolist() == [[0, 0, 0]]
    assert result.reconstructed_rgb.tolist() == [[0, 0, 0]]


def test_protocol_row_gate_precedes_high_level_fitting() -> None:
    with pytest.raises(L.LloydError, match="8192"):
        L.fit_flat(np.zeros((8, 3), dtype=np.uint8), 2, 1)
    with pytest.raises(L.LloydError, match="8192"):
        L.fit_rvq(np.zeros((8, 3), dtype=np.uint8), 2, 1)


def test_protocol_menu_rejects_nonfrozen_k_after_row_validation() -> None:
    rgb = np.zeros((L.PROTOCOL_ROWS, 3), dtype=np.uint8)
    with pytest.raises(L.LloydError, match="64 or 128"):
        L.fit_flat(rgb, 2, 1)
    with pytest.raises(L.LloydError, match="64 or 128"):
        L.fit_rvq(rgb, 2, 1)


def test_complexity_fixture_is_target_free_and_byte_exact() -> None:
    fixture = L.complexity_fixture(4)
    assert fixture.dtype == np.uint8
    assert fixture.tolist() == [[183, 43, 45], [152, 95, 31], [21, 24, 4], [23, 130, 56]]
    manual = []
    for index in range(4):
        digest = hashlib.sha256(
            b"structsplat-comp011-complexity-v1\0" + struct.pack("<I", index)
        ).digest()
        manual.append((digest[0], digest[1], digest[2]))
    assert fixture.tolist() == [list(row) for row in manual]


def test_native_source_is_the_bound_standalone_file() -> None:
    assert L.NATIVE_SOURCE_PATH.name == "ssp2v_lloyd_ext.cpp"
    assert L.NATIVE_SOURCE_PATH.is_file()
    assert Path(L.__file__).with_name("ssp2v_lloyd_ext.cpp") == L.NATIVE_SOURCE_PATH
