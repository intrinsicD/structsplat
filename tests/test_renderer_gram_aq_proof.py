import numpy as np
import pytest

from benchmarks import renderer_gram_aq_proof as proof


def _dense_operator(matrix):
    return proof.RendererLinearMap.from_dense(np.asarray(matrix, dtype=np.float64))


def test_dense_and_column_footprint_operators_match_and_are_adjoint():
    dense = np.asarray(
        [
            [0.7, 0.0, -0.2],
            [0.1, 0.4, 0.0],
            [0.0, 0.6, 0.3],
            [0.2, 0.0, 0.9],
        ],
        dtype=np.float64,
    )
    dense_operator = proof.RendererLinearMap.from_dense(dense)
    sparse_operator = proof.RendererLinearMap.from_column_footprints(
        4,
        [
            ([3, 0, 1], [0.2, 0.7, 0.1]),
            ([2, 1], [0.6, 0.4]),
            ([0, 2, 3], [-0.2, 0.3, 0.9]),
        ],
    )
    fields = np.asarray([[0.3, -0.2], [0.7, 0.4], [-0.5, 0.8]], dtype=np.float64)
    pixels = np.asarray(
        [[0.1, -0.4], [0.9, 0.2], [-0.3, 0.5], [0.7, -0.8]],
        dtype=np.float64,
    )

    expected_render = dense @ fields
    expected_transpose = dense.T @ pixels
    assert np.allclose(dense_operator.apply(fields), expected_render, atol=1e-16, rtol=1e-16)
    assert np.allclose(sparse_operator.apply(fields), expected_render, atol=1e-16, rtol=1e-16)
    assert np.allclose(dense_operator.transpose(pixels), expected_transpose, atol=1e-16, rtol=1e-16)
    assert np.allclose(
        sparse_operator.transpose(pixels), expected_transpose, atol=1e-16, rtol=1e-16
    )
    assert np.dot(dense_operator.apply(fields[:, 0]), pixels[:, 0]) == pytest.approx(
        np.dot(fields[:, 0], dense_operator.transpose(pixels[:, 0])),
        abs=2e-16,
    )
    assert np.array_equal(dense_operator.column_norm_sq, np.sum(dense * dense, axis=0))


def test_column_footprints_reject_duplicate_pixels_instead_of_hiding_cross_terms():
    with pytest.raises(ValueError, match="duplicate pixels"):
        proof.RendererLinearMap.from_column_footprints(
            3,
            [([0, 0], [0.2, 0.3])],
        )


def test_float64_source_and_target_single_move_deltas_match_direct_recomputation():
    matrix = np.asarray(
        [
            [0.65, 0.25, 0.0, 0.10],
            [0.15, 0.70, 0.20, 0.0],
            [0.0, 0.10, 0.75, 0.30],
            [0.05, 0.0, 0.25, 0.80],
            [0.20, 0.15, 0.0, 0.05],
        ],
        dtype=np.float64,
    )
    renderer = _dense_operator(matrix)
    source = np.asarray(
        [[0.12, 0.80, 0.31], [0.90, 0.18, 0.42], [0.44, 0.57, 0.76], [0.65, 0.11, 0.93]],
        dtype=np.float64,
    )
    codebook = np.asarray(
        [[0.05, 0.85, 0.25], [0.52, 0.48, 0.70], [0.95, 0.10, 0.40]],
        dtype=np.float64,
    )
    assignments = np.asarray([0, 2, 1, 1], dtype=np.int64)
    target = renderer.apply(source) + np.asarray(
        [
            [0.03, -0.01, 0.02],
            [-0.02, 0.04, -0.01],
            [0.01, 0.02, -0.03],
            [-0.04, -0.02, 0.01],
            [0.02, 0.01, 0.03],
        ],
        dtype=np.float64,
    )

    source_check = proof.check_source_move_delta(
        renderer,
        source,
        codebook,
        assignments,
        field_index=1,
        new_code=1,
    )
    target_check = proof.check_target_move_delta(
        renderer,
        target,
        codebook,
        assignments,
        field_index=1,
        new_code=1,
    )

    assert source_check.passes(atol=2e-15, rtol=2e-13)
    assert target_check.passes(atol=2e-15, rtol=2e-13)
    assert source_check.relative_error < 2e-13
    assert target_check.relative_error < 2e-13
    assert source_check.direct_delta != pytest.approx(target_check.direct_delta, abs=1e-8)


def test_diagonal_gram_control_is_exact_without_overlap_and_intentionally_wrong_with_overlap():
    source = np.zeros((2, 1), dtype=np.float64)
    codebook = np.ones((1, 1), dtype=np.float64)
    assignments = np.zeros((2,), dtype=np.int64)
    nonoverlap = _dense_operator([[1.0, 0.0], [0.0, 2.0]])
    overlap = _dense_operator([[1.0, 1.0], [0.0, 1.0]])

    nonoverlap_control = proof.run_deterministic_icm(
        nonoverlap,
        codebook,
        assignments,
        objective_kind="source",
        gram_kind="diagonal",
        reference_colors=source,
        max_sweeps=1,
    )
    overlap_control = proof.run_deterministic_icm(
        overlap,
        codebook,
        assignments,
        objective_kind="source",
        gram_kind="diagonal",
        reference_colors=source,
        max_sweeps=1,
    )

    assert nonoverlap_control.optimized_distortion == pytest.approx(
        nonoverlap_control.actual_full_distortion,
        abs=1e-15,
    )
    assert overlap_control.optimized_distortion == pytest.approx(3.0, abs=1e-15)
    assert overlap_control.actual_full_distortion == pytest.approx(5.0, abs=1e-15)


@pytest.mark.parametrize(
    "old_code,new_code",
    [(0, 2), (1, 0), (0, 1)],
)
def test_exact_empirical_rate_delta_matches_full_count_recomputation(old_code, new_code):
    counts = np.asarray([3, 1, 0], dtype=np.int64)
    penalty = 7.25
    before = proof.empirical_index_rate_bits(
        counts,
        active_code_penalty_bits=penalty,
    )
    after_counts = counts.copy()
    after_counts[old_code] -= 1
    after_counts[new_code] += 1
    expected = (
        proof.empirical_index_rate_bits(
            after_counts,
            active_code_penalty_bits=penalty,
        )
        - before
    )
    actual = proof.empirical_index_rate_delta_bits(
        counts,
        old_code,
        new_code,
        active_code_penalty_bits=penalty,
    )
    assert actual == pytest.approx(expected, abs=2e-15)


def _explicit_complete_objective(renderer, colors, source, assignments, rate_weight, penalty):
    residual = renderer.apply(colors - source)
    distortion = float(np.sum(residual * residual))
    counts = np.bincount(assignments, minlength=3).astype(np.int64)
    rate = proof.empirical_index_rate_bits(counts, active_code_penalty_bits=penalty)
    return distortion + rate_weight * rate


def test_rate_aware_icm_is_deterministic_monotone_and_terminal_under_brute_force():
    renderer = _dense_operator(
        [
            [1.0, 0.35, 0.0, 0.0],
            [0.2, 0.9, 0.25, 0.0],
            [0.0, 0.2, 1.0, 0.4],
            [0.0, 0.0, 0.3, 0.85],
            [0.1, 0.15, 0.0, 0.2],
        ]
    )
    source = np.asarray(
        [[0.05, 0.20], [0.38, 0.62], [0.82, 0.45], [1.12, 0.95]],
        dtype=np.float64,
    )
    codebook = np.asarray([[-0.15, 0.10], [0.42, 0.58], [1.20, 1.05]], dtype=np.float64)
    initial = np.asarray([0, 0, 2, 2], dtype=np.int64)
    rate_weight = 0.025
    penalty = 0.4

    first = proof.run_deterministic_icm(
        renderer,
        codebook,
        initial,
        objective_kind="source",
        gram_kind="full",
        reference_colors=source,
        rate_weight=rate_weight,
        active_code_penalty_bits=penalty,
        improvement_tolerance=1e-13,
        max_sweeps=30,
    )
    second = proof.run_deterministic_icm(
        renderer,
        codebook,
        initial,
        objective_kind="source",
        gram_kind="full",
        reference_colors=source,
        rate_weight=rate_weight,
        active_code_penalty_bits=penalty,
        improvement_tolerance=1e-13,
        max_sweeps=30,
    )

    assert np.array_equal(first.assignments, second.assignments)
    assert first.objective_history == second.objective_history
    assert first.accepted_move_deltas == second.accepted_move_deltas
    assert len(first.accepted_move_deltas) > 0
    assert all(delta < -1e-13 for delta in first.accepted_move_deltas)
    assert all(
        after <= before + 2e-13
        for before, after in zip(first.objective_history, first.objective_history[1:])
    )
    assert first.terminal
    assert first.terminal_certificate.objective_delta >= -1e-13

    terminal_objective = _explicit_complete_objective(
        renderer,
        first.quantized_colors,
        source,
        first.assignments,
        rate_weight,
        penalty,
    )
    brute_force_deltas = []
    for field_index, old_code in enumerate(first.assignments):
        for new_code in range(codebook.shape[0]):
            if new_code == old_code:
                continue
            moved_assignments = first.assignments.copy()
            moved_assignments[field_index] = new_code
            moved_colors = codebook[moved_assignments]
            brute_force_deltas.append(
                _explicit_complete_objective(
                    renderer,
                    moved_colors,
                    source,
                    moved_assignments,
                    rate_weight,
                    penalty,
                )
                - terminal_objective
            )
    assert min(brute_force_deltas) == pytest.approx(
        first.terminal_certificate.objective_delta,
        abs=2e-14,
    )
    assert min(brute_force_deltas) >= -1e-13


def test_target_aware_diagonal_icm_is_deterministic_and_reports_actual_full_error():
    renderer = _dense_operator([[0.8, 0.3, 0.0], [0.1, 0.7, 0.4], [0.0, 0.2, 0.9]])
    reference = np.asarray([[0.2], [0.6], [0.9]], dtype=np.float64)
    target = renderer.apply(reference) + np.asarray([[0.05], [-0.04], [0.02]])
    codebook = np.asarray([[0.1], [0.55], [1.1]], dtype=np.float64)
    initial = np.asarray([2, 0, 0], dtype=np.int64)

    result = proof.run_deterministic_icm(
        renderer,
        codebook,
        initial,
        objective_kind="target",
        gram_kind="diagonal",
        reference_colors=reference,
        target_pixels=target,
        max_sweeps=20,
    )
    direct = renderer.apply(result.quantized_colors) - target
    assert result.terminal
    assert result.actual_full_distortion == pytest.approx(float(np.sum(direct * direct)))
    assert np.isfinite(result.optimized_distortion)


@pytest.mark.parametrize("solve_kind", ["source", "target"])
def test_fixed_assignment_matrix_free_codebook_solve_matches_explicit_least_squares(solve_kind):
    matrix = np.asarray(
        [
            [0.8, 0.1, 0.0, 0.2, 0.0],
            [0.2, 0.7, 0.1, 0.0, 0.3],
            [0.0, 0.2, 0.9, 0.1, 0.0],
            [0.1, 0.0, 0.3, 0.8, 0.2],
            [0.0, 0.1, 0.0, 0.2, 0.9],
            [0.3, 0.0, 0.2, 0.0, 0.1],
        ],
        dtype=np.float64,
    )
    renderer = _dense_operator(matrix)
    assignments = np.asarray([0, 1, 0, 2, 1], dtype=np.int64)
    source = np.asarray(
        [[0.1, 0.7], [0.5, 0.2], [0.9, 0.4], [0.3, 1.0], [0.8, 0.6]],
        dtype=np.float64,
    )
    if solve_kind == "source":
        target = renderer.apply(source)
        result = proof.solve_fixed_assignment_codebook(
            renderer,
            assignments,
            4,
            source_colors=source,
            rtol=1e-13,
            atol=1e-15,
        )
    else:
        target = renderer.apply(source) + np.asarray(
            [[0.02, -0.03], [0.01, 0.01], [-0.04, 0.02], [0.03, 0.0], [0.0, -0.02], [0.01, 0.04]],
            dtype=np.float64,
        )
        result = proof.solve_fixed_assignment_codebook(
            renderer,
            assignments,
            4,
            target_pixels=target,
            rtol=1e-13,
            atol=1e-15,
        )

    selector = np.zeros((assignments.size, 4), dtype=np.float64)
    selector[np.arange(assignments.size), assignments] = 1.0
    explicit_design = matrix @ selector
    explicit_codebook, _residuals, _rank, _singular = np.linalg.lstsq(
        explicit_design,
        target,
        rcond=None,
    )
    explicit_render = explicit_design @ explicit_codebook
    explicit_error = float(np.sum((explicit_render - target) ** 2))

    assert result.converged
    assert result.normal_apply_calls > 0
    assert np.allclose(result.rendered, explicit_render, atol=2e-12, rtol=2e-12)
    assert result.squared_error == pytest.approx(explicit_error, abs=3e-14)
    # Code 3 is unused: zero-start CG leaves its unidentifiable row at the min-norm value zero.
    assert np.array_equal(result.codebook[3], np.zeros((2,), dtype=np.float64))


def test_clamp_warning_accounts_for_both_tails_and_exact_clip_change():
    diagnostics = proof.clamp_diagnostics(
        np.asarray([[-0.2, 0.0], [1.0, 1.3]], dtype=np.float64),
    )
    assert diagnostics.lower_count == 1
    assert diagnostics.upper_count == 1
    assert diagnostics.active_count == 2
    assert diagnostics.total_count == 4
    assert diagnostics.active_fraction == pytest.approx(0.5)
    assert diagnostics.max_underflow == pytest.approx(0.2)
    assert diagnostics.max_overflow == pytest.approx(0.3)
    assert diagnostics.squared_clip_change == pytest.approx(0.13)
    assert diagnostics.warning
