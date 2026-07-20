from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from benchmarks import renderer_gram_aq_proof as gram
from benchmarks import ssp2f_coordinate_rdo as coordinate
from benchmarks import ssp2f_rdo_objective as O


def _dense_weights() -> np.ndarray:
    return np.asarray(
        [
            [0.75, 0.25, 0.00, 0.00],
            [0.00, 0.60, 0.40, 0.00],
            [0.20, 0.00, 0.30, 0.50],
            [0.00, 0.00, 0.00, 0.00],
            [0.10, 0.20, 0.30, 0.40],
            [0.55, 0.00, 0.00, 0.45],
        ],
        dtype=np.float64,
    )


def _symbols() -> np.ndarray:
    return np.asarray(
        [
            [0, 255, 128],
            [255, 0, 64],
            [200, 180, 255],
            [32, 220, 0],
        ],
        dtype=np.uint8,
    )


def _direct_color_table(
    low: np.ndarray,
    high: np.ndarray,
    *,
    bits: int = 8,
) -> np.ndarray:
    levels = (1 << bits) - 1
    decoded = (
        low.astype(np.float64)[None, :]
        + np.arange(levels + 1, dtype=np.float64)[:, None]
        / float(levels)
        * (high.astype(np.float64) - low.astype(np.float64))[None, :]
    )
    return decoded.astype(np.float32).astype(np.float64)


def _direct_colors(
    symbols: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
) -> np.ndarray:
    table = _direct_color_table(low, high)
    return table[symbols, np.arange(3)[None, :]]


def _target() -> np.ndarray:
    return np.asarray(
        [
            [[0, 64, 255], [255, 128, 0], [32, 32, 32]],
            [[16, 240, 120], [80, 160, 240], [220, 40, 180]],
        ],
        dtype=np.uint8,
    )


def test_dense_target_objective_matches_declared_float_paths_and_clamp() -> None:
    weights = _dense_weights()
    symbols = _symbols()
    low = np.asarray([-0.75, -0.25, 0.1], dtype=np.float64)
    high = np.asarray([1.75, 1.5, 2.0], dtype=np.float64)
    target = _target()
    engine = O.SSP2FRGBObjective(
        weights,
        symbols,
        target,
        reference_kind="target",
        color_lo=low,
        color_hi=high,
    )

    colors = _direct_colors(symbols, low, high)
    expected_render = np.clip(weights @ colors, 0.0, 1.0)
    expected_reference = (target.reshape(-1, 3).astype(np.float32) / np.float32(255.0)).astype(
        np.float64
    )
    expected_objective = float(
        np.sum((expected_render - expected_reference) ** 2, dtype=np.float64)
    )

    np.testing.assert_allclose(engine.rendered_image, expected_render, rtol=0.0, atol=2e-16)
    np.testing.assert_array_equal(engine.reference_image, expected_reference)
    assert engine.objective == pytest.approx(expected_objective, rel=2e-16, abs=2e-16)
    assert engine.evaluate(symbols) == pytest.approx(expected_objective, rel=2e-16, abs=2e-16)
    assert engine(symbols) == engine.evaluate(symbols)

    decoded64_without_codec_cast = (
        low[None, :] + np.arange(256, dtype=np.float64)[:, None] / 255.0 * (high - low)[None, :]
    )
    assert np.any(O.affine_color_table(low, high) != decoded64_without_codec_cast)


def test_source_reference_and_sparse_operator_have_dense_parity() -> None:
    weights = _dense_weights()
    symbols = _symbols()
    low = np.asarray([-0.4, -0.2, 0.0])
    high = np.asarray([1.2, 1.4, 1.1])
    source_colors = _direct_colors(symbols, low, high)
    source_image = np.clip(weights @ source_colors, 0.0, 1.0)
    candidate = symbols.copy()
    candidate[[0, 1, 3], [0, 2, 1]] = [50, 190, 80]

    dense = O.SSP2FRGBObjective(
        weights,
        candidate,
        source_image,
        reference_kind="source",
        color_lo=low,
        color_hi=high,
    )
    sparse_map = gram.RendererLinearMap.from_dense(weights)
    sparse = O.SSP2FRGBObjective(
        sparse_map,
        candidate,
        source_image,
        reference_kind="source",
        color_lo=low,
        color_hi=high,
    )

    assert dense.operator_sha256 == sparse.operator_sha256
    np.testing.assert_array_equal(dense.rendered_image, sparse.rendered_image)
    assert dense.objective.hex() == sparse.objective.hex()
    assert dense.config_sha256 == sparse.config_sha256


def test_source_objective_constructor_uses_clamped_cold_source_render() -> None:
    weights = _dense_weights()
    source = _symbols()
    low = np.asarray([-0.5, -0.25, 0.0])
    high = np.asarray([1.25, 1.5, 2.0])
    identity = O.SSP2FRGBObjective.from_source_symbols(
        weights,
        source,
        source,
        color_lo=low,
        color_hi=high,
    )
    assert identity.objective == 0.0

    candidate = source.copy()
    candidate[2, 2] = 0
    changed = O.SSP2FRGBObjective.from_source_symbols(
        weights,
        candidate,
        source,
        color_lo=low,
        color_hi=high,
    )
    colors = _direct_colors(source, low, high)
    expected_reference = np.clip(weights @ colors, 0.0, 1.0)
    np.testing.assert_allclose(
        changed.reference_image,
        expected_reference,
        rtol=0.0,
        atol=2e-16,
    )
    assert changed.objective > 0.0


def test_scalar_proposals_match_full_recomputation_across_clamp_crossings() -> None:
    weights = np.asarray(
        [
            [0.7, 0.3, 0.0],
            [0.1, 0.9, 0.0],
            [0.0, 0.25, 0.75],
            [0.4, 0.0, 0.6],
        ],
        dtype=np.float64,
    )
    symbols = np.asarray(
        [[0, 250, 20], [255, 5, 240], [120, 128, 130]],
        dtype=np.uint8,
    )
    target = np.asarray(
        [[0, 255, 64], [255, 0, 200], [30, 100, 250], [220, 80, 10]],
        dtype=np.uint8,
    )
    engine = O.SSP2FRGBObjective(
        weights,
        symbols,
        target,
        reference_kind="target",
        color_lo=[-1.0, -0.5, -0.25],
        color_hi=[2.0, 1.5, 2.5],
    )

    edits = ((0, 0, 255), (1, "green", 255), (2, "blue", 0), (1, 0, 10))
    stale = engine.propose(2, 1, 60)
    for row, channel, replacement in edits:
        proposal = engine.propose(row, channel, replacement)
        candidate = engine.rgb_symbols
        channel_index = O.CHANNEL_NAMES.index(channel) if isinstance(channel, str) else channel
        candidate[row, channel_index] = replacement
        full_objective = engine.evaluate(candidate)
        tolerance = O.RECONCILIATION_OBJECTIVE_ATOL + O.RECONCILIATION_OBJECTIVE_RTOL * max(
            abs(proposal.objective), abs(full_objective)
        )
        assert abs(proposal.objective - full_objective) <= tolerance
        assert proposal.candidate_state_sha256 == engine.state_digest_for(candidate)
        assert proposal.origin_state_sha256 == engine.state_sha256
        assert proposal.objective_delta == pytest.approx(
            proposal.objective - proposal.objective_before,
            rel=2e-15,
            abs=2e-15,
        )
        assert engine.commit(proposal) == proposal.objective
        assert engine.state_sha256 == proposal.candidate_state_sha256

    with pytest.raises(O.SSP2FRDOObjectiveError, match="stale"):
        engine.commit(stale)
    report = engine.reconcile()
    assert report.passed
    assert report.rebased
    assert report.commits_since_reconciliation == len(edits)
    assert engine.commits_since_reconciliation == 0
    assert engine.objective.hex() == report.full_objective.hex()
    np.testing.assert_array_equal(engine.rgb_symbols, candidate)


def test_random_sequential_commits_reconcile_with_full_objective() -> None:
    rng = np.random.default_rng(20260718)
    raw = rng.random((37, 19), dtype=np.float64)
    weights = raw / np.maximum(raw.sum(axis=1, keepdims=True), 1.0)
    symbols = rng.integers(0, 256, size=(19, 3), dtype=np.uint8)
    target = rng.integers(0, 256, size=(37, 3), dtype=np.uint8)
    engine = O.SSP2FRGBObjective(
        weights,
        symbols,
        target,
        reference_kind="target",
        color_lo=[-0.75, -0.5, -0.25],
        color_hi=[1.75, 1.5, 1.25],
    )

    for step in range(400):
        row = int(rng.integers(0, symbols.shape[0]))
        channel = int(rng.integers(0, 3))
        old = int(engine.rgb_symbols[row, channel])
        replacement = int(rng.integers(0, 255))
        if replacement == old:
            replacement = 255
        if replacement == old:
            replacement = 0
        proposal = engine.propose(row, channel, replacement)
        candidate = engine.rgb_symbols
        candidate[row, channel] = replacement
        full = engine.evaluate(candidate)
        assert abs(proposal.objective - full) <= (
            O.RECONCILIATION_OBJECTIVE_ATOL
            + O.RECONCILIATION_OBJECTIVE_RTOL * max(abs(proposal.objective), abs(full))
        )
        engine.commit(proposal)
        if (step + 1) % 73 == 0:
            report = engine.reconcile()
            assert report.passed

    report = engine.reconcile()
    assert report.passed
    assert engine.objective.hex() == engine.evaluate(engine.rgb_symbols).hex()
    np.testing.assert_array_equal(engine.rendered_image, engine.render(engine.rgb_symbols))
    acceptance_tolerance = 1e-12 * max(1.0, abs(report.full_objective))
    assert report.objective_tolerance <= 0.2 * acceptance_tolerance


def test_proposal_is_bound_to_complete_payload_and_noop_is_rejected() -> None:
    engine = O.SSP2FRGBObjective(
        _dense_weights(),
        _symbols(),
        _target(),
        reference_kind="target",
        color_lo=[0.0, 0.0, 0.0],
        color_hi=[1.0, 1.0, 1.0],
    )
    proposal = engine.propose(0, "red", 7)
    assert proposal.origin_state_sha256 == coordinate.state_digest(engine.rgb_symbols)
    candidate = engine.rgb_symbols
    candidate[0, 0] = 7
    assert proposal.candidate_state_sha256 == coordinate.state_digest(candidate)
    other_target = _target()
    other_target[0, 0, 0] = 1
    other_engine = O.SSP2FRGBObjective(
        _dense_weights(),
        _symbols(),
        other_target,
        reference_kind="target",
        color_lo=[0.0, 0.0, 0.0],
        color_hi=[1.0, 1.0, 1.0],
    )
    with pytest.raises(O.SSP2FRDOObjectiveError, match="stale"):
        other_engine.commit(proposal)
    with pytest.raises(O.SSP2FRDOObjectiveError, match="payload or digest"):
        engine.commit(replace(proposal, objective=proposal.objective + 1.0))
    assert engine.epoch == 0
    assert engine.rgb_symbols[0, 0] == 0
    with pytest.raises(O.SSP2FRDOObjectiveError, match="must change"):
        engine.propose(0, 0, 0)
    with pytest.raises(O.SSP2FRDOObjectiveError, match="out of range"):
        engine.propose(4, 0, 1)
    with pytest.raises(O.SSP2FRDOObjectiveError, match="out of range"):
        engine.propose(0, 3, 1)
    with pytest.raises(O.SSP2FRDOObjectiveError, match="outside"):
        engine.propose(0, 0, 256)


def test_inputs_are_frozen_and_public_arrays_are_detached() -> None:
    weights = _dense_weights()
    symbols = _symbols()
    engine = O.SSP2FRGBObjective(
        weights,
        symbols,
        _target(),
        reference_kind="target",
        color_lo=[0.0, 0.0, 0.0],
        color_hi=[1.0, 1.0, 1.0],
    )
    baseline_config = engine.config_sha256
    baseline_objective = engine.objective
    weights[:] = 0.0
    symbols[:] = 0
    detached_symbols = engine.rgb_symbols
    detached_render = engine.rendered_image
    detached_reference = engine.reference_image
    detached_symbols[:] = 5
    detached_render[:] = 0.0
    detached_reference[:] = 1.0

    assert engine.config_sha256 == baseline_config
    assert engine.objective == baseline_objective
    np.testing.assert_array_equal(engine.rgb_symbols, _symbols())
    assert np.any(engine.rendered_image != 0.0)
    assert np.any(engine.reference_image != 1.0)


@pytest.mark.parametrize(
    ("cache_name", "error_field", "tolerance_field"),
    (
        ("unclamped", "render_max_abs_error", "render_tolerance"),
        ("clamped", "clamped_max_abs_error", "clamped_tolerance"),
        ("residual", "residual_max_abs_error", "residual_tolerance"),
        (
            "squared_residual",
            "squared_residual_max_abs_error",
            "squared_residual_tolerance",
        ),
        (
            "objective_blocks",
            "objective_blocks_max_abs_error",
            "objective_blocks_tolerance",
        ),
        ("objective", "objective_abs_error", "objective_tolerance"),
    ),
)
def test_reconciliation_fails_closed_on_every_cache_corruption(
    cache_name: str,
    error_field: str,
    tolerance_field: str,
) -> None:
    engine = O.SSP2FRGBObjective(
        _dense_weights(),
        _symbols(),
        _target(),
        reference_kind="target",
        color_lo=[0.0, 0.0, 0.0],
        color_hi=[1.0, 1.0, 1.0],
    )
    if cache_name == "unclamped":
        engine._unclamped[0, 0] += 1e-4
    elif cache_name == "clamped":
        engine._clamped[0, 0] += 1e-4
    elif cache_name == "residual":
        engine._residual[0, 0] += 1000.0
    elif cache_name == "squared_residual":
        engine._squared_residual[0, 0] += 1e-4
    elif cache_name == "objective_blocks":
        engine._objective_blocks[0] += 1e-4
    elif cache_name == "objective":
        engine._objective += 1e-4
    else:
        raise AssertionError(cache_name)
    with pytest.raises(O.SSP2FRDOReconciliationError) as caught:
        engine.reconcile()
    assert not caught.value.report.passed
    assert not caught.value.report.rebased
    assert getattr(caught.value.report, error_field) > getattr(caught.value.report, tolerance_field)
    if cache_name == "residual":
        assert (
            caught.value.report.residual_reduced_objective_abs_error
            > caught.value.report.residual_reduced_objective_tolerance
        )


def test_negative_objective_cache_and_blocks_are_poison() -> None:
    engine = O.SSP2FRGBObjective.from_source_symbols(
        _dense_weights(),
        _symbols(),
        _symbols(),
        color_lo=[0.0, 0.0, 0.0],
        color_hi=[1.0, 1.0, 1.0],
    )
    assert engine.objective == 0.0
    engine._objective = -np.finfo(np.float64).eps
    with pytest.raises(O.SSP2FRDOReconciliationError) as caught:
        engine.reconcile()
    assert not caught.value.report.passed
    assert caught.value.report.cached_objective < 0.0

    blocks_engine = O.SSP2FRGBObjective.from_source_symbols(
        _dense_weights(),
        _symbols(),
        _symbols(),
        color_lo=[0.0, 0.0, 0.0],
        color_hi=[1.0, 1.0, 1.0],
    )
    blocks_engine._objective_blocks[0] = -np.finfo(np.float64).eps
    with pytest.raises(O.SSP2FRDOObjectiveError, match="finite and nonnegative"):
        blocks_engine.propose(0, 0, 1)


def test_reversible_incremental_path_never_produces_negative_sse() -> None:
    rng = np.random.default_rng(3)
    raw = rng.random((2000, 25), dtype=np.float64)
    weights = raw / raw.sum(axis=1, keepdims=True)
    source = rng.integers(0, 256, size=(25, 3), dtype=np.uint8)
    engine = O.SSP2FRGBObjective.from_source_symbols(
        weights,
        source,
        source,
        color_lo=[-0.5, -0.25, 0.0],
        color_hi=[1.5, 1.25, 2.0],
    )
    history: list[tuple[int, int, int]] = []
    for step in range(50):
        row = int(rng.integers(0, source.shape[0]))
        channel = int(rng.integers(0, 3))
        old_symbol = int(engine.rgb_symbols[row, channel])
        new_symbol = (old_symbol + step + 1) % 256
        if new_symbol == old_symbol:
            new_symbol = (new_symbol + 1) % 256
        proposal = engine.propose(row, channel, new_symbol)
        assert proposal.objective >= 0.0
        assert engine.commit(proposal) >= 0.0
        history.append((row, channel, old_symbol))

    for row, channel, old_symbol in reversed(history):
        proposal = engine.propose(row, channel, old_symbol)
        assert proposal.objective >= 0.0
        assert engine.commit(proposal) >= 0.0

    np.testing.assert_array_equal(engine.rgb_symbols, source)
    assert engine.objective >= 0.0
    report = engine.reconcile()
    assert report.passed
    assert report.full_objective == 0.0
    assert engine.objective == 0.0


@pytest.mark.parametrize(
    ("weights", "match"),
    (
        (np.asarray([[-0.1, 0.2]], dtype=np.float64), "nonnegative"),
        (np.asarray([[0.8, 0.3]], dtype=np.float64), "row sum"),
        (np.asarray([[np.nan]], dtype=np.float64), "finite"),
    ),
)
def test_invalid_normalized_operators_fail(
    weights: np.ndarray,
    match: str,
) -> None:
    with pytest.raises(O.SSP2FRDOObjectiveError, match=match):
        O.FrozenNormalizedOperator.from_dense(weights)


def test_invalid_rgb_reference_and_dequantization_inputs_fail() -> None:
    weights = _dense_weights()
    symbols = _symbols()
    with pytest.raises(O.SSP2FRDOObjectiveError, match="dtype uint8"):
        O.SSP2FRGBObjective(
            weights,
            symbols.astype(np.uint32),
            _target(),
            reference_kind="target",
            color_lo=[0.0] * 3,
            color_hi=[1.0] * 3,
        )
    with pytest.raises(O.SSP2FRDOObjectiveError, match="target reference"):
        O.SSP2FRGBObjective(
            weights,
            symbols,
            _target().astype(np.float32),
            reference_kind="target",
            color_lo=[0.0] * 3,
            color_hi=[1.0] * 3,
        )
    with pytest.raises(O.SSP2FRDOObjectiveError, match="source reference"):
        O.SSP2FRGBObjective(
            weights,
            symbols,
            np.zeros((5, 3), dtype=np.float64),
            reference_kind="source",
            color_lo=[0.0] * 3,
            color_hi=[1.0] * 3,
        )
    with pytest.raises(O.SSP2FRDOObjectiveError, match="greater"):
        O.affine_color_table([0.0, 1.0, 0.0], [1.0, 1.0, 2.0])
    with pytest.raises(O.SSP2FRDOObjectiveError, match="bits_color"):
        O.affine_color_table([0.0] * 3, [1.0] * 3, bits_color=9)


def test_lower_bit_alphabet_and_empty_footprint_are_deterministic() -> None:
    weights = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    symbols = np.asarray([[0, 15, 7], [15, 0, 8]], dtype=np.uint8)
    source = np.zeros((2, 3), dtype=np.float64)
    engine = O.SSP2FRGBObjective(
        weights,
        symbols,
        source,
        reference_kind="source",
        color_lo=[0.0] * 3,
        color_hi=[1.0] * 3,
        bits_color=4,
    )
    proposal = engine.propose(0, 0, 15)
    assert proposal.old_color == 0.0
    assert proposal.new_color == 1.0
    assert proposal.affected_pixels == 1
    engine.commit(proposal)
    with pytest.raises(O.SSP2FRDOObjectiveError, match="above"):
        engine.state_digest_for(np.asarray([[16, 0, 0], [0, 0, 0]], dtype=np.uint8))

    sparse = O.FrozenNormalizedOperator(
        2,
        (
            (np.asarray([], dtype=np.int64), np.asarray([], dtype=np.float64)),
            (np.asarray([0], dtype=np.int64), np.asarray([1.0])),
        ),
    )
    empty_engine = O.SSP2FRGBObjective(
        sparse,
        np.asarray([[0, 0, 0], [0, 0, 0]], dtype=np.uint8),
        source,
        reference_kind="source",
        color_lo=[0.0] * 3,
        color_hi=[1.0] * 3,
    )
    empty = empty_engine.propose(0, 2, 255)
    assert empty.affected_pixels == 0
    assert empty.objective_delta == 0.0
    assert empty.objective.hex() == empty.objective_before.hex()
