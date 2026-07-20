from __future__ import annotations

import hashlib
import struct

import numpy as np
import pytest

from benchmarks import ssp2e_model as M


GRID_HASHES = (
    "40aff2e9d2d8922e47afd4648e6967497158785fbd1da870e7110266bf944880",
    "6ccee3ab08882a58e0debe15a25ada10de0891da82a9ac5fcf4ee591617b0c39",
    "459cb7f92764cf14cedc73ac8441f9632c2f3c921d6548a7f0672d182b2f13f6",
    "ed4673ad0a62e83d4f37e2c89840887244efc0a8b93207189d85117cc0805ee7",
    "6cf719adb055c87bbedac8815eb6996b9829c473cd71cf27dfbf95ee24291bce",
    "a4eb419ceb590ffb584a72a1be6751681ba8195d95450226a3afb753caaf231c",
    "43d6a533847cc9e2aac616036f2efc612ddee33523df01905fd011ce5c021c02",
    "f037588291db1791e08d4bd8d132c17c4a438fb6b2eafe25699ba4def5e4a541",
)


def _zero_objective() -> M.ModelObjective:
    shape = (M.CELL_COUNT, M.LOCATION_COUNT, M.SCALE_COUNT)
    return M.ModelObjective(tuple(np.zeros(shape, dtype=np.int64) for _ in M.CHANNELS))


def _parity_objective() -> M.ModelObjective:
    shape = (M.CELL_COUNT, M.LOCATION_COUNT, M.SCALE_COUNT)
    costs = [np.zeros(shape, dtype=np.int64) for _ in M.CHANNELS]
    for cell in range(M.CELL_COUNT):
        target = 2 if cell & 1 else 0
        costs[0][cell] = np.abs(np.arange(64, dtype=np.int64)[:, None] - target)
    return M.ModelObjective(tuple(costs))


def _head(
    *,
    location_weights: tuple[int, ...] = (0,) * 8,
    location_bias: int = 0,
    scale_weights: tuple[int, ...] = (0,) * 8,
    scale_bias: int = 0,
) -> M.ChannelHead:
    return M.ChannelHead(location_weights, location_bias, scale_weights, scale_bias)


def _fit_result(start_id: int, model: M.SSP2EModel, objective: int) -> M.FitResult:
    record = M.SweepRecord(1, objective, 0, 0, 0)
    return M.FitResult(start_id, model, objective, 1, (record,))


def test_grid_and_head_round_trip_exact_signed_layout_and_int32_inference() -> None:
    first = _head(
        location_weights=(-128, -7, -1, 0, 1, 7, 63, 127),
        location_bias=-1234,
        scale_weights=(127, 63, 7, 1, 0, -1, -7, -128),
        scale_bias=32767,
    )
    heads = (first, _head(location_weights=(1,) * 8, location_bias=7)) + tuple(
        _head() for _ in range(3)
    )
    model = M.SSP2EModel(bytes(range(256)), heads)

    expected_first = struct.pack(
        "<8bh8bh",
        -128,
        -7,
        -1,
        0,
        1,
        7,
        63,
        127,
        -1234,
        127,
        63,
        7,
        1,
        0,
        -1,
        -7,
        -128,
        32767,
    )
    assert model.head_raw[:20] == expected_first
    assert len(model.grid_raw) == 256
    assert len(model.head_raw) == 100
    assert M.SSP2EModel.from_raw(model.grid_raw, model.head_raw) == model

    inferred = M.infer_cell_parameters(model)
    phi0 = np.asarray([-1] * 8, dtype=np.int32)
    expected_location = int(np.clip(-1234 + np.dot(first.location_weights, phi0), 0, 63))
    expected_scale = int(np.clip(32767 + np.dot(first.scale_weights, phi0), 0, 15))
    assert inferred["scale_x"][0].dtype == np.uint8
    assert inferred["scale_x"][0][0] == expected_location
    assert inferred["scale_x"][1][0] == expected_scale

    rows = M.infer_row_parameters(model, np.asarray([255, 0, 255], dtype=np.uint16))
    assert rows["scale_x"][0].tolist() == [
        int(inferred["scale_x"][0][255]),
        int(inferred["scale_x"][0][0]),
        int(inferred["scale_x"][0][255]),
    ]


@pytest.mark.parametrize(
    ("constructor", "match"),
    [
        (lambda: _head(location_weights=(0,) * 7), "eight weights"),
        (lambda: _head(location_weights=(0,) * 7 + (128,)), "location weight"),
        (lambda: _head(scale_bias=-32769), "scale bias"),
        (lambda: M.SSP2EModel(b"x", (_head(),) * 5), "grid has"),
        (lambda: M.SSP2EModel.from_raw(bytes(256), bytes(99)), "head has"),
    ],
)
def test_model_serialization_rejects_nonrepresentable_state(constructor, match: str) -> None:
    with pytest.raises(M.ModelError, match=match):
        constructor()


def test_position_cells_use_wide_multiply_and_exact_edge_formula() -> None:
    q32 = np.asarray([0, (1 << 32) // 16 - 1, (1 << 32) // 16, (1 << 32) - 1])
    cells = M.position_cells(q32, np.zeros_like(q32), 32)
    assert cells.tolist() == [0, 0, 1, 15]
    y = M.position_cells(np.zeros(3, dtype=np.uint16), [0, 2048, 4095], 12)
    assert y.tolist() == [0, 128, 240]


def test_all_eight_grid_starts_have_frozen_ids_domains_and_hashes() -> None:
    grids = M.all_initial_grids()
    assert len(grids) == 8
    assert grids[0] == bytes(range(256))
    assert grids[1] == bytes(cell ^ (cell >> 1) for cell in range(256))
    assert grids[2][1] == 128
    assert tuple(hashlib.sha256(grid).hexdigest() for grid in grids) == GRID_HASHES
    for start in range(3, 8):
        expected = bytes(
            hashlib.sha256(M.GRID_START_DOMAIN + struct.pack("<BH", start, cell)).digest()[0]
            for cell in range(256)
        )
        assert grids[start] == expected
    assert len(set(grids)) == 8


def test_static_shuffle_permutation_is_exact_unique_and_returns_copies() -> None:
    first = M.shuffle_permutation()
    second = M.shuffle_permutation()
    assert first.dtype == np.dtype("<u2")
    assert first[:12].tolist() == [
        1580,
        2634,
        1817,
        2825,
        6741,
        592,
        1371,
        1678,
        6939,
        7283,
        3368,
        5993,
    ]
    assert np.array_equal(np.sort(first), np.arange(8192))
    assert hashlib.sha256(first.astype("<u2").tobytes()).hexdigest() == (
        M.SHUFFLE_PERMUTATION_SHA256
    )
    first[0] = 0
    assert second[0] == 1580

    contexts = np.arange(8192, dtype=np.uint16) % 256
    assert np.array_equal(M.shuffled_position_contexts(contexts), contexts[second])


def test_histograms_and_injected_frequency_table_consume_exact_integer_q24_costs() -> None:
    contexts = np.asarray([0, 0, 0, 1], dtype=np.uint16)
    symbols = {
        "scale_x": np.asarray([3, 3, 4, 2]),
        "scale_y": np.asarray([1, 2, 1, 2]),
        "red": np.asarray([3, 3, 4, 2]),
        "green": np.asarray([1, 2, 1, 2]),
        "blue": np.asarray([3, 3, 4, 2]),
    }
    alphabets = {name: (64 if name.startswith("scale") else 256) for name in M.CHANNELS}
    histograms = M.aggregate_context_histograms(contexts, symbols, alphabets)
    assert histograms.rows == 4
    assert histograms.counts[0][0, 3] == 2
    assert histograms.counts[0][1, 2] == 1

    calls = []

    def frequencies(channel: str, location: int, scale: int) -> np.ndarray:
        calls.append((channel, location, scale))
        alphabet = alphabets[channel]
        row = np.ones(alphabet, dtype=np.int64)
        row[location % alphabet] += M.ARITHMETIC_TOTAL - alphabet
        return row

    cost = np.arange(M.ARITHMETIC_TOTAL + 1, dtype=np.uint32)[::-1].copy()
    objective = M.build_model_objective(histograms, frequencies, cost)
    assert len(calls) == len(M.CHANNELS) * 64 * 16
    frequent = M.ARITHMETIC_TOTAL - 64 + 1
    expected = 2 * int(cost[frequent]) + int(cost[1])
    assert objective.cell_costs[0][0, 3, 7] == expected
    assert objective.cell_costs[0][0, 4, 7] == 2 * int(cost[1]) + int(cost[frequent])
    assert objective.cell_costs[0][0, 3, 0] == objective.cell_costs[0][0, 3, 15]

    with pytest.raises(M.ModelError, match="integers"):
        M.build_model_objective(histograms, frequencies, np.ones(32769, dtype=np.float64))


def test_flat_ties_keep_current_grid_and_ascending_global_bias_pair() -> None:
    objective = _zero_objective()
    result = M.fit_one_start(objective, 3)
    assert result.sweeps == 1
    assert result.objective_q24 == 0
    assert result.model.grid == M.initial_grid(3)
    assert all(head == _head() for head in result.model.heads)
    assert result.trajectory == (M.SweepRecord(0, 0, 0, 0, 0), M.SweepRecord(1, 0, 0, 0, 0))
    assert M.best_global_pairs(objective) == ((0, 0),) * 5
    assert M.verify_fixed_point(objective, result.model)


def test_coordinate_descent_is_strictly_monotonic_and_reaches_a_verified_fixed_point() -> None:
    objective = _parity_objective()

    result = M.fit_one_start(objective, 0)
    trajectory = [record.objective_q24 for record in result.trajectory]
    assert trajectory == [256, 0, 0]
    assert result.trajectory[1].strict_updates > 0
    assert result.trajectory[-1].strict_updates == 0
    assert all(later <= earlier for earlier, later in zip(trajectory, trajectory[1:]))
    assert result.model.heads[0].location_weights[0] == 2
    assert M.verify_fixed_point(objective, result.model)


def test_actual_eight_start_fit_preserves_each_grid_at_a_flat_fixed_point() -> None:
    fits = M.fit_eight_starts(_zero_objective())
    assert tuple(fit.start_id for fit in fits) == tuple(range(8))
    assert all(fit.sweeps == 1 and fit.objective_q24 == 0 for fit in fits)
    assert tuple(fit.model.grid for fit in fits) == M.all_initial_grids()


def test_every_start_is_fitted_and_production_wrapper_forces_64_sweeps(monkeypatch) -> None:
    observed = []
    objective = _zero_objective()

    def fake_fit(obj, start_id, *, max_sweeps):
        assert obj is objective
        observed.append((start_id, max_sweeps))
        model = M.SSP2EModel(M.initial_grid(start_id), (_head(),) * 5)
        return _fit_result(start_id, model, 0)

    monkeypatch.setattr(M, "fit_one_start", fake_fit)
    fits = M.fit_eight_starts(objective)
    assert [fit.start_id for fit in fits] == list(range(8))
    assert observed == [(start, 64) for start in range(8)]


def test_sweep_cap_failure_carries_real_strict_terminal_state_and_trajectory() -> None:
    objective = _parity_objective()
    with pytest.raises(M.ConvergenceError) as caught:
        M.fit_one_start(objective, 0, max_sweeps=1)
    assert caught.value.start_id == 0
    assert caught.value.terminal_objective_q24 == 0
    assert [record.objective_q24 for record in caught.value.trajectory] == [256, 0]
    assert caught.value.trajectory[-1].strict_updates > 0
    # The terminal state happens to be optimal, but the fitter correctly refuses to call it a
    # fixed point until a complete following sweep observes no strict improvement.
    assert objective.objective_q24(caught.value.terminal_model) == 0


def test_selection_encodes_all_starts_then_uses_length_objective_and_model_bytes() -> None:
    heads = (_head(),) * 5
    models = [M.SSP2EModel(bytes([start]) + bytes(255), heads) for start in range(8)]
    fits = tuple(
        _fit_result(start, models[start], 10 if start in (2, 3) else 100 + start)
        for start in range(8)
    )
    called = []

    def lengths(model: M.SSP2EModel):
        start = model.grid[0]
        called.append(start)
        if start in (2, 3):
            return {name: 1 for name in M.CHANNELS}
        if start == 4:  # Better Q24 than most, but loses the primary byte criterion.
            return [2, 1, 1, 1, 1]
        return [2, 2, 2, 2, 2]

    selection = M.select_fitted_model(fits, lengths)
    assert called == list(range(8))
    assert selection.selected.fit.start_id == 2
    assert selection.selected.total_payload_bytes == 5
    assert selection.selected.fit.objective_q24 == 10
    assert selection.duplicate_winner_start_ids == (2,)
    assert selection.selection_reason["criteria"] == [
        "sum_five_arithmetic_payload_lengths",
        "exact_q24_objective",
        "lexicographic_grid_raw_head_raw",
    ]


def test_selection_rejects_missing_starts_and_nonpositive_or_partial_lengths() -> None:
    model = M.SSP2EModel(bytes(256), (_head(),) * 5)
    fits = tuple(_fit_result(start, model, 0) for start in range(8))
    with pytest.raises(M.ModelError, match="five payload"):
        M.select_fitted_model(fits, lambda _: [1, 2])
    with pytest.raises(M.ModelError, match=r"\[1"):
        M.select_fitted_model(fits, lambda _: [1, 1, 1, 1, 0])
    with pytest.raises(M.ModelError, match="ordered converged"):
        M.select_fitted_model(fits[:-1], lambda _: [1] * 5)
