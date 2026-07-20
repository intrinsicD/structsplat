# ruff: noqa: E402
import math

import pytest

torch = pytest.importorskip("torch")

from benchmarks.residual_tangent_auction_matrixfree import (
    DEFAULT_BLOCK_SIZE,
    DAMPING_COLUMN_NORMALIZED,
    DAMPING_DIMENSIONLESS_PARENT_CHART,
    FROZEN_RCONDS,
    MatrixFreeConvergenceError,
    build_masked_jacobian_operators,
    factorize_joint_tangent_six,
    factorize_masked_tangent,
    literal_projector_energy,
    randomized_range_finder,
    residualize_six_columns,
    select_tangent_six_all_pixel_oracle,
    select_tangent_six_crossfit,
    solve_damped_normal,
    solve_joint_damped_normal,
    uniform_linf_scale_joint_coefficients,
)


DEVICE_DTYPE_CASES = [
    pytest.param("cpu", torch.float32, id="cpu-float32"),
    pytest.param("cpu", torch.float64, id="cpu-float64"),
]
if torch.cuda.is_available():
    DEVICE_DTYPE_CASES.append(pytest.param("cuda", torch.float32, id="cuda-float32"))


def _linear_operators(matrix, *, mask=None, bias=None):
    if bias is None:
        bias = matrix.new_zeros((matrix.shape[0],))

    def closure(step):
        return bias + matrix @ step

    return build_masked_jacobian_operators(
        closure,
        matrix.new_zeros((matrix.shape[1],)),
        mask,
    )


def _rank_deficient_matrix():
    generator = torch.Generator().manual_seed(27)
    rotation, _ = torch.linalg.qr(
        torch.randn(12, 12, generator=generator, dtype=torch.float64), mode="reduced"
    )
    axes = torch.eye(12, dtype=torch.float64)
    delta = 1e-4
    # Columns 0/4 are dependent, column 5 is null, and columns 2/3 are nearly dependent.
    # Nonzero columns have arbitrary raw scales but unit norms after Stage-0-style scaling.
    columns = torch.column_stack(
        (
            axes[:, 0],
            axes[:, 1],
            axes[:, 2],
            (axes[:, 2] + delta * axes[:, 3]) / math.sqrt(1.0 + delta**2),
            2.0 * axes[:, 0],
            torch.zeros(12, dtype=torch.float64),
        )
    )
    return rotation @ columns


def _explicit_scaled_svd(design, rcond):
    scales = torch.linalg.vector_norm(design, dim=0).clamp_min(1e-12)
    left, singular, vh = torch.linalg.svd(design / scales, full_matrices=False)
    keep = singular > float(rcond) * singular[0]
    return scales, left[:, keep], singular[keep], vh[keep], singular


def _explicit_damped_action(design, residual, *, rcond, damping, finite_offset=None):
    offset = torch.zeros_like(residual) if finite_offset is None else finite_offset
    scales, left, singular, vh, _all = _explicit_scaled_svd(design, rcond)
    coordinates = (
        singular
        / (singular.square() + float(damping) ** 2)
        * (left.T @ (residual - offset))
    )
    coefficients = (vh.T @ coordinates) / scales
    coefficients = torch.where(
        torch.linalg.vector_norm(design, dim=0) > 1e-12,
        coefficients,
        torch.zeros_like(coefficients),
    )
    return coefficients, offset + design @ coefficients


class _RecordingOperators:
    def __init__(self, operators):
        self._operators = operators
        self.jvp_directions = []
        self.vjp_cotangents = []

    def __getattr__(self, name):
        return getattr(self._operators, name)

    def jvp(self, direction):
        self.jvp_directions.append(direction.detach().clone())
        return self._operators.jvp(direction)

    def vjp(self, cotangent):
        self.vjp_cotangents.append(cotangent.detach().clone())
        return self._operators.vjp(cotangent)


def _paired_linear_operators(matrix, discovery_mask):
    template = matrix.new_zeros((matrix.shape[1],))

    def closure(step):
        return matrix @ step

    discovery = build_masked_jacobian_operators(
        closure,
        template,
        discovery_mask,
    )
    heldout = build_masked_jacobian_operators(
        closure,
        template,
        ~discovery_mask,
    )
    return discovery, heldout


@pytest.mark.parametrize("device,dtype", DEVICE_DTYPE_CASES)
def test_masked_jvp_vjp_match_explicit_matrix_and_adjoint(device, dtype):
    matrix = torch.arange(45, device=device, dtype=dtype).reshape(9, 5) / 17.0 - 0.6
    bias = torch.linspace(-0.2, 0.3, 9, device=device, dtype=dtype)
    mask = torch.tensor(
        [True, False, True, True, False, False, True, False, True],
        device=device,
    )

    def nonlinear_closure(step):
        linear = matrix @ step
        return bias + linear + 0.05 * linear.square()

    operators = build_masked_jacobian_operators(
        nonlinear_closure,
        torch.ones(5, device=device, dtype=dtype),
        mask,
    )
    direction = torch.tensor([0.3, -0.2, 0.5, 0.1, -0.4], device=device, dtype=dtype)
    cotangent = torch.linspace(-0.4, 0.7, int(mask.sum()), device=device, dtype=dtype)
    explicit = matrix[mask]
    tolerance = 2e-5 if dtype == torch.float32 else 2e-12

    jvp = operators.jvp(direction)
    vjp = operators.vjp(cotangent)
    assert torch.allclose(jvp, explicit @ direction, atol=tolerance, rtol=tolerance)
    assert torch.allclose(vjp, explicit.T @ cotangent, atol=tolerance, rtol=tolerance)
    assert torch.allclose(jvp @ cotangent, direction @ vjp, atol=tolerance, rtol=tolerance)
    assert torch.equal(operators.origin, torch.zeros_like(operators.origin))
    assert torch.equal(operators.masked_output_at_zero, bias[mask])
    assert torch.equal(operators.select_output(matrix), explicit)
    assert operators.parameter_count == 5
    assert operators.full_output_count == 9
    assert operators.output_count == 5


def test_blocked_range_and_reduced_svd_match_rank_deficient_explicit_oracle():
    matrix = _rank_deficient_matrix()
    operators = _linear_operators(matrix)
    factorizations = factorize_masked_tangent(
        operators,
        seed=91,
        probe_count=5,
        probe_tolerance=1e-11,
    )
    expected_ranks = {1e-5: 4, 1e-4: 3}

    assert set(factorizations) == set(FROZEN_RCONDS)
    for rcond, factorization in factorizations.items():
        rank = expected_ranks[rcond]
        scales, explicit_left, explicit_singular, _explicit_vh, _all = (
            _explicit_scaled_svd(matrix, rcond)
        )
        assert factorization.rank == rank
        assert torch.allclose(factorization.column_scales, scales, atol=2e-12, rtol=2e-12)
        assert torch.allclose(
            factorization.singular_values,
            explicit_singular,
            atol=2e-12,
            rtol=2e-12,
        )
        expected_projector = explicit_left @ explicit_left.T
        actual_projector = factorization.left_basis @ factorization.left_basis.T
        # The retained fourth direction is only 5e-5 of the leading scaled singular value.
        assert torch.allclose(actual_projector, expected_projector, atol=1e-9, rtol=1e-9)
        assert torch.allclose(
            factorization.reduced_matrix,
            factorization.range_result.basis.T @ matrix,
            atol=2e-12,
            rtol=2e-12,
        )
        identity = torch.eye(rank, dtype=matrix.dtype)
        assert torch.allclose(
            factorization.left_basis.T @ factorization.left_basis,
            identity,
            atol=2e-12,
            rtol=2e-12,
        )

    range_result = factorizations[1e-5].range_result
    assert range_result.converged is True
    assert range_result.block_size == DEFAULT_BLOCK_SIZE == 32
    assert range_result.sample_cap == matrix.shape[1]
    assert range_result.sample_count == matrix.shape[1]
    assert range_result.rank == 4
    assert range_result.reconstruction_error <= 1e-11
    assert range_result.basis_dimension_history == (0, 4)
    assert len(range_result.reconstruction_error_history) == 2
    assert range_result.reconstruction_error_history[-1] <= range_result.reconstruction_error_history[0]
    assert torch.allclose(
        range_result.basis.T @ range_result.basis,
        torch.eye(4, dtype=matrix.dtype),
        atol=2e-12,
        rtol=2e-12,
    )


def test_range_and_factorization_hashes_are_seed_deterministic():
    matrix = _rank_deficient_matrix()
    operators = _linear_operators(matrix)
    first = randomized_range_finder(
        operators,
        seed=314,
        probe_count=4,
        probe_tolerance=1e-11,
    )
    second = randomized_range_finder(
        operators,
        seed=314,
        probe_count=4,
        probe_tolerance=1e-11,
    )
    changed = randomized_range_finder(
        operators,
        seed=315,
        probe_count=4,
        probe_tolerance=1e-11,
    )

    assert torch.equal(first.basis, second.basis)
    assert first.basis_sha256 == second.basis_sha256
    assert first.probe_directions_sha256 == second.probe_directions_sha256
    assert first.probe_images_sha256 == second.probe_images_sha256
    assert first.sample_directions_sha256 == second.sample_directions_sha256
    assert first.sample_directions_sha256 != changed.sample_directions_sha256
    assert first.probe_directions_sha256 != changed.probe_directions_sha256

    first_factor = factorize_masked_tangent(
        operators, seed=314, probe_count=4, probe_tolerance=1e-11
    )
    second_factor = factorize_masked_tangent(
        operators, seed=314, probe_count=4, probe_tolerance=1e-11
    )
    for rcond in FROZEN_RCONDS:
        assert (
            first_factor[rcond].left_basis_sha256
            == second_factor[rcond].left_basis_sha256
        )
        assert (
            first_factor[rcond].singular_values_sha256
            == second_factor[rcond].singular_values_sha256
        )


def test_damped_normal_action_matches_truncated_explicit_svd():
    matrix = _rank_deficient_matrix()
    operators = _linear_operators(matrix)
    factorization = factorize_masked_tangent(
        operators,
        rconds=(1e-4,),
        seed=8,
        probe_count=5,
        probe_tolerance=1e-11,
    )[1e-4]
    generator = torch.Generator().manual_seed(44)
    residual = torch.randn(matrix.shape[0], generator=generator, dtype=torch.float64)
    damping = 1e-2
    action = solve_damped_normal(
        factorization,
        residual,
        damping=damping,
        damping_coordinate_system=DAMPING_COLUMN_NORMALIZED,
    )

    scales, left, singular, vh, _all = _explicit_scaled_svd(matrix, 1e-4)
    expected_step = (
        vh.T
        @ (
            singular
            / (singular.square() + damping**2)
            * (left.T @ residual)
        )
    ) / scales
    expected_action = matrix @ expected_step
    expected_post = float((residual - expected_action).square().sum())
    assert torch.allclose(action.parameter_step, expected_step, atol=3e-10, rtol=3e-10)
    assert torch.allclose(action.predicted_action, expected_action, atol=3e-11, rtol=3e-11)
    assert torch.allclose(
        operators.jvp(action.parameter_step),
        action.predicted_action,
        atol=3e-11,
        rtol=3e-11,
    )
    assert action.predicted_post_energy == pytest.approx(expected_post, abs=2e-12)
    assert action.predicted_energy_reduction == pytest.approx(
        action.residual_energy - expected_post,
        abs=2e-12,
    )


def test_damped_normal_default_uses_raw_dimensionless_parent_chart():
    matrix = _rank_deficient_matrix()
    factorization = factorize_masked_tangent(
        _linear_operators(matrix),
        rconds=(1e-4,),
        seed=8,
        probe_count=5,
        probe_tolerance=1e-11,
    )[1e-4]
    generator = torch.Generator().manual_seed(45)
    residual = torch.randn(matrix.shape[0], generator=generator, dtype=torch.float64)
    damping = 1e-2
    action = solve_damped_normal(factorization, residual, damping=damping)

    retained = factorization.left_basis.T @ matrix
    rhs = factorization.left_basis.T @ residual
    expected_step = torch.linalg.solve(
        retained.T @ retained
        + damping**2 * torch.eye(matrix.shape[1], dtype=matrix.dtype),
        retained.T @ rhs,
    )
    expected_action = factorization.left_basis @ (retained @ expected_step)
    assert action.damping_coordinate_system == DAMPING_DIMENSIONLESS_PARENT_CHART
    assert torch.allclose(action.parameter_step, expected_step, atol=3e-10, rtol=3e-10)
    assert torch.allclose(action.predicted_action, expected_action, atol=3e-10, rtol=3e-10)


def test_compact_joint_six_matches_explicit_svd_at_both_rconds_and_dampings():
    matrix = _rank_deficient_matrix()
    generator = torch.Generator().manual_seed(818)
    raw_extension, _ = torch.linalg.qr(
        torch.randn(12, 6, generator=generator, dtype=torch.float64),
        mode="reduced",
    )
    extension = torch.column_stack(
        (
            raw_extension[:, 0],
            raw_extension[:, 1],
            raw_extension[:, 2],
            raw_extension[:, 2] + 1e-4 * raw_extension[:, 3],
            raw_extension[:, 4],
            raw_extension[:, 5],
        )
    )
    operators = _linear_operators(matrix)
    base = factorize_masked_tangent(
        operators,
        seed=118,
        probe_count=6,
        probe_tolerance=1e-11,
    )[1e-5]
    factors = factorize_joint_tangent_six(
        base,
        extension,
        rconds=FROZEN_RCONDS,
    )
    repeated = factorize_joint_tangent_six(
        base,
        extension,
        rconds=FROZEN_RCONDS,
    )
    design = torch.cat((matrix, extension), dim=1)
    residual = torch.randn(12, generator=generator, dtype=torch.float64)
    finite_offset = 0.03 * torch.randn(12, generator=generator, dtype=torch.float64)

    for rcond in FROZEN_RCONDS:
        factor = factors[rcond]
        scales, explicit_left, explicit_singular, _vh, _all = _explicit_scaled_svd(
            design,
            rcond,
        )
        assert factor.rank == explicit_singular.numel()
        assert torch.allclose(factor.column_scales, scales, atol=3e-12, rtol=3e-12)
        assert torch.allclose(
            factor.singular_values,
            explicit_singular,
            atol=5e-11,
            rtol=5e-11,
        )
        assert torch.allclose(
            factor.left_basis @ factor.left_basis.T,
            explicit_left @ explicit_left.T,
            atol=2e-9,
            rtol=2e-9,
        )
        assert factor.additional_jvp_calls == factor.additional_vjp_calls == 0
        assert factor.reused_range_jvp_calls == base.range_result.jvp_calls
        assert factor.reused_reduced_vjp_calls == base.vjp_calls
        assert factor.left_basis_sha256 == repeated[rcond].left_basis_sha256
        assert factor.singular_values_sha256 == repeated[rcond].singular_values_sha256

        for damping in (1e-3, 1e-2):
            action = solve_joint_damped_normal(
                factor,
                residual,
                damping=damping,
                finite_offset=finite_offset,
                damping_coordinate_system=DAMPING_COLUMN_NORMALIZED,
            )
            expected_coefficients, expected_prediction = _explicit_damped_action(
                design,
                residual,
                rcond=rcond,
                damping=damping,
                finite_offset=finite_offset,
            )
            assert action.finite_offset_present is True
            assert torch.allclose(
                action.concatenated_coefficients,
                expected_coefficients,
                atol=3e-8,
                rtol=3e-8,
            )
            assert torch.allclose(
                action.base_parameter_step,
                expected_coefficients[: matrix.shape[1]],
                atol=3e-8,
                rtol=3e-8,
            )
            assert torch.allclose(
                action.extension_coefficients,
                expected_coefficients[matrix.shape[1] :],
                atol=3e-8,
                rtol=3e-8,
            )
            assert torch.allclose(
                action.predicted_action,
                expected_prediction,
                atol=2e-10,
                rtol=2e-10,
            )
            assert torch.allclose(
                action.predicted_action,
                finite_offset
                + operators.jvp(action.base_parameter_step)
                + extension @ action.extension_coefficients,
                atol=2e-10,
                rtol=2e-10,
            )

    rescaling = torch.tensor(
        [0.01, -100.0, 3.0, -0.2, 7.0, 0.5],
        dtype=torch.float64,
    )
    rescaled_factor = factorize_joint_tangent_six(
        base,
        extension * rescaling,
        rconds=(1e-5,),
    )[1e-5]
    original_action = solve_joint_damped_normal(
        factors[1e-5],
        residual,
        damping=1e-2,
        finite_offset=finite_offset,
        damping_coordinate_system=DAMPING_COLUMN_NORMALIZED,
    )
    rescaled_action = solve_joint_damped_normal(
        rescaled_factor,
        residual,
        damping=1e-2,
        finite_offset=finite_offset,
        damping_coordinate_system=DAMPING_COLUMN_NORMALIZED,
    )
    assert torch.allclose(
        rescaled_action.predicted_action,
        original_action.predicted_action,
        atol=2e-10,
        rtol=2e-10,
    )
    assert torch.allclose(
        rescaled_action.base_parameter_step,
        original_action.base_parameter_step,
        atol=3e-8,
        rtol=3e-8,
    )
    assert torch.allclose(
        rescaled_action.extension_coefficients * rescaling,
        original_action.extension_coefficients,
        atol=3e-8,
        rtol=3e-8,
    )


def test_tangent_six_crossfit_is_leak_safe_deterministic_and_selected_column_only():
    generator = torch.Generator().manual_seed(909)
    discovery_matrix = torch.eye(10, dtype=torch.float64)
    heldout_matrix = torch.randn(7, 10, generator=generator, dtype=torch.float64) / 4.0
    full_matrix = torch.cat((discovery_matrix, heldout_matrix), dim=0)
    discovery_mask = torch.zeros(17, dtype=torch.bool)
    discovery_mask[:10] = True
    raw_discovery, raw_heldout = _paired_linear_operators(full_matrix, discovery_mask)
    discovery = _RecordingOperators(raw_discovery)
    heldout = _RecordingOperators(raw_heldout)
    coordinate_ids = (
        "k",
        "j",
        "i",
        "h",
        "z_tie",
        "a_tie",
        "g",
        "f",
        "e",
        "d",
    )
    discovery_residual = torch.tensor(
        [0.9, -0.8, 0.7, -0.6, 0.5, -0.5, 0.4, 0.3, 0.2, 0.1],
        dtype=torch.float64,
    )
    heldout_residual = torch.randn(7, generator=generator, dtype=torch.float64)
    result = select_tangent_six_crossfit(
        discovery,
        heldout,
        discovery_residual,
        heldout_residual,
        coordinate_ids=coordinate_ids,
        rcond=1e-5,
        damping=0.0,
    )

    expected_indices = (0, 1, 2, 3, 5, 4)
    expected_ids = tuple(coordinate_ids[index] for index in expected_indices)
    assert result.coordinate_indices == expected_indices
    assert result.coordinate_ids == expected_ids
    assert len(set(result.coordinate_indices)) == 6
    assert result.rank == 6
    assert result.discovery_vjp_calls == len(discovery.vjp_cotangents) == 6
    assert result.discovery_jvp_calls == len(discovery.jvp_directions) == 6
    assert result.heldout_vjp_calls == len(heldout.vjp_cotangents) == 0
    assert result.heldout_jvp_calls == len(heldout.jvp_directions) == 6
    for directions in (discovery.jvp_directions, heldout.jvp_directions):
        observed = tuple(int(direction.abs().argmax()) for direction in directions)
        assert observed == expected_indices
        assert all(float(direction.abs().sum()) == 1.0 for direction in directions)
    assert result.omp_qr_rank_history == tuple(range(7))
    assert all(
        later <= earlier + 1e-15
        for earlier, later in zip(
            result.omp_residual_energy_history,
            result.omp_residual_energy_history[1:],
        )
    )
    assert torch.allclose(
        result.discovery_design,
        discovery_matrix[:, list(expected_indices)],
        atol=0.0,
        rtol=0.0,
    )
    assert torch.allclose(
        result.heldout_design,
        heldout_matrix[:, list(expected_indices)],
        atol=0.0,
        rtol=0.0,
    )
    assert torch.allclose(
        result.heldout_predicted_action,
        result.heldout_design @ result.coefficients,
        atol=2e-15,
        rtol=2e-15,
    )
    assert torch.allclose(
        raw_discovery.jvp(result.sparse_parameter_step),
        result.discovery_predicted_action,
        atol=2e-15,
        rtol=2e-15,
    )

    perturbed_heldout = heldout_residual + 1000.0 * torch.arange(
        7,
        dtype=torch.float64,
    )
    perturbed = select_tangent_six_crossfit(
        raw_discovery,
        raw_heldout,
        discovery_residual,
        perturbed_heldout,
        coordinate_ids=coordinate_ids,
        rcond=1e-5,
        damping=0.0,
    )
    assert perturbed.coordinate_indices == result.coordinate_indices
    assert perturbed.coordinate_ids == result.coordinate_ids
    assert torch.equal(perturbed.coefficients, result.coefficients)
    assert torch.equal(perturbed.sparse_parameter_step, result.sparse_parameter_step)
    assert torch.equal(
        perturbed.discovery_predicted_action,
        result.discovery_predicted_action,
    )
    assert torch.equal(
        perturbed.heldout_predicted_action,
        result.heldout_predicted_action,
    )
    assert perturbed.coordinate_identity_sha256 == result.coordinate_identity_sha256
    assert perturbed.coefficients_sha256 == result.coefficients_sha256
    assert perturbed.heldout_post_energy != result.heldout_post_energy


def test_tangent_six_all_pixel_oracle_is_separate_and_matches_explicit_fit():
    generator = torch.Generator().manual_seed(910)
    full_matrix = torch.randn(19, 9, generator=generator, dtype=torch.float64)
    discovery_mask = torch.zeros(19, dtype=torch.bool)
    discovery_mask[::2] = True
    raw_discovery, raw_heldout = _paired_linear_operators(full_matrix, discovery_mask)
    discovery = _RecordingOperators(raw_discovery)
    heldout = _RecordingOperators(raw_heldout)
    discovery_residual = torch.randn(
        raw_discovery.output_count,
        generator=generator,
        dtype=torch.float64,
    )
    heldout_residual = torch.randn(
        raw_heldout.output_count,
        generator=generator,
        dtype=torch.float64,
    )
    result = select_tangent_six_all_pixel_oracle(
        discovery,
        heldout,
        discovery_residual,
        heldout_residual,
        coordinate_ids=tuple(f"coordinate-{index}" for index in range(9)),
        rcond=1e-4,
        damping=1e-2,
        damping_coordinate_system=DAMPING_COLUMN_NORMALIZED,
    )
    combined_design = torch.cat((result.discovery_design, result.heldout_design))
    combined_residual = torch.cat((discovery_residual, heldout_residual))
    expected_coefficients, expected_prediction = _explicit_damped_action(
        combined_design,
        combined_residual,
        rcond=1e-4,
        damping=1e-2,
    )
    assert result.fit_policy == "all_pixel_oracle_identity_and_coefficients"
    assert result.damping_coordinate_system == DAMPING_COLUMN_NORMALIZED
    assert len(set(result.coordinate_indices)) == 6
    assert result.discovery_vjp_calls == len(discovery.vjp_cotangents) == 6
    assert result.heldout_vjp_calls == len(heldout.vjp_cotangents) == 6
    assert result.discovery_jvp_calls == len(discovery.jvp_directions) == 6
    assert result.heldout_jvp_calls == len(heldout.jvp_directions) == 6
    assert torch.allclose(
        result.coefficients,
        expected_coefficients,
        atol=3e-12,
        rtol=3e-12,
    )
    assert torch.allclose(
        torch.cat(
            (result.discovery_predicted_action, result.heldout_predicted_action)
        ),
        expected_prediction,
        atol=3e-12,
        rtol=3e-12,
    )
    assert torch.allclose(
        torch.cat(
            (
                full_matrix[discovery_mask] @ result.sparse_parameter_step,
                full_matrix[~discovery_mask] @ result.sparse_parameter_step,
            )
        ),
        expected_prediction,
        atol=3e-12,
        rtol=3e-12,
    )


def test_uniform_joint_linf_scale_uses_one_factor_for_every_coordinate():
    base = torch.tensor([2.0, -4.0, 0.5], dtype=torch.float64)
    extension = torch.tensor([1.0, -3.0, 0.0, 0.2, -0.4, 2.5], dtype=torch.float64)
    result = uniform_linf_scale_joint_coefficients(base, extension, 0.75)
    expected_factor = 0.75 / 4.0
    expected = torch.cat((base, extension)) * expected_factor
    assert result.factor == pytest.approx(expected_factor, abs=0.0)
    assert result.unscaled_max_abs == 4.0
    assert result.scaled_max_abs == pytest.approx(0.75, abs=0.0)
    assert torch.equal(result.concatenated_coefficients, expected)
    assert torch.equal(result.base_parameter_step, expected[:3])
    assert torch.equal(result.extension_coefficients, expected[3:])
    nonzero = torch.cat((base, extension)) != 0.0
    assert torch.allclose(
        result.concatenated_coefficients[nonzero]
        / torch.cat((base, extension))[nonzero],
        torch.full((8,), expected_factor, dtype=torch.float64),
        atol=3e-17,
        rtol=3e-17,
    )

    unchanged = uniform_linf_scale_joint_coefficients(base / 10.0, extension / 10.0, 0.75)
    assert unchanged.factor == 1.0
    assert torch.equal(
        unchanged.concatenated_coefficients,
        torch.cat((base / 10.0, extension / 10.0)),
    )


def test_six_column_residualization_and_literal_projector_energy_match_joint_oracle():
    generator = torch.Generator().manual_seed(77)
    tangent_left, _ = torch.linalg.qr(
        torch.randn(16, 3, generator=generator, dtype=torch.float64), mode="reduced"
    )
    tangent_right, _ = torch.linalg.qr(
        torch.randn(7, 3, generator=generator, dtype=torch.float64), mode="reduced"
    )
    tangent = (
        tangent_left
        @ torch.diag(torch.tensor([2.0, 0.7, 0.1], dtype=torch.float64))
        @ tangent_right.T
    )
    operators = _linear_operators(tangent)
    factorization = factorize_masked_tangent(
        operators,
        rconds=(1e-5,),
        seed=12,
        probe_count=5,
        probe_tolerance=1e-11,
    )[1e-5]

    raw_new = torch.randn(16, 3, generator=generator, dtype=torch.float64)
    raw_new = raw_new - tangent_left @ (tangent_left.T @ raw_new)
    new_basis, _ = torch.linalg.qr(raw_new, mode="reduced")
    tangent_coefficients = torch.randn(7, 2, generator=generator, dtype=torch.float64)
    columns = torch.column_stack(
        (
            tangent @ tangent_coefficients[:, 0],
            tangent @ tangent_coefficients[:, 1],
            new_basis[:, 0],
            2.0 * new_basis[:, 0],
            new_basis[:, 1],
            new_basis[:, 2],
        )
    )
    residual = (
        0.8 * tangent_left[:, 0]
        - 0.3 * tangent_left[:, 2]
        + 0.9 * new_basis[:, 0]
        - 0.5 * new_basis[:, 2]
        + 0.05 * torch.randn(16, generator=generator, dtype=torch.float64)
    )

    residualized = residualize_six_columns(factorization, columns)
    result = literal_projector_energy(
        factorization,
        residual,
        columns,
        rcond=1e-10,
    )
    assert result.base_rank == 3
    assert result.joint_rank == 6
    assert torch.equal(result.residualized_columns, residualized)
    assert result.tangent_orthogonality_max_abs <= 2e-14
    assert float((factorization.left_basis.T @ residualized).abs().max()) <= 2e-14

    _base_scales, base_basis, _base_singular, _base_vh, _ = _explicit_scaled_svd(
        tangent, 1e-5
    )
    _joint_scales, joint_basis, _joint_singular, _joint_vh, _ = _explicit_scaled_svd(
        torch.cat((tangent, columns), dim=1), 1e-10
    )
    expected_base = float((base_basis.T @ residual).square().sum())
    expected_joint = float((joint_basis.T @ residual).square().sum())
    assert result.tangent_projector_energy == pytest.approx(expected_base, abs=3e-12)
    assert result.joint_projector_energy == pytest.approx(expected_joint, abs=3e-12)
    assert result.incremental_projector_energy == pytest.approx(
        expected_joint - expected_base,
        abs=3e-12,
    )
    assert result.unexplained_energy == pytest.approx(
        float(residual.square().sum()) - expected_joint,
        abs=3e-12,
    )

    # Literal projection is invariant to arbitrary nonzero column units because both the
    # matrix-free and explicit joint systems normalize every [J,A] column before truncation.
    rescaled_columns = columns * torch.tensor(
        [0.01, 100.0, 3.0, 0.2, 7.0, 0.5], dtype=torch.float64
    )
    rescaled = literal_projector_energy(
        factorization,
        residual,
        rescaled_columns,
        rcond=1e-10,
    )
    assert rescaled.joint_rank == result.joint_rank
    assert rescaled.joint_projector_energy == pytest.approx(
        result.joint_projector_energy,
        abs=3e-12,
    )


def test_range_nonconvergence_is_reported_and_factorization_fails_closed():
    matrix = torch.diag(torch.tensor([3.0, 1.0, 0.2, 0.0], dtype=torch.float64))
    operators = _linear_operators(matrix)
    result = randomized_range_finder(
        operators,
        seed=5,
        block_size=1,
        sample_cap=1,
        probe_count=3,
        probe_tolerance=1e-12,
    )
    assert result.converged is False
    assert result.sample_count == result.sample_cap == 1
    assert result.rank == 1
    assert result.reconstruction_error > result.probe_tolerance

    with pytest.raises(MatrixFreeConvergenceError) as captured:
        factorize_masked_tangent(
            operators,
            seed=5,
            block_size=1,
            sample_cap=1,
            probe_count=3,
            probe_tolerance=1e-12,
        )
    assert captured.value.result.converged is False
    assert "did not converge" in str(captured.value)


def test_float32_full_cap_uses_exact_coordinate_jvp_fallback_without_relaxing_tolerance():
    # The weak third image direction is below the float32 block-SVD cutoff.  A randomized block
    # that spans all three parameter coordinates therefore still misses its fixed-probe content,
    # while the coordinate Jacobian's float64 rank construction retains it exactly.
    matrix = torch.diag(torch.tensor([1.0, 0.1, 5e-6], dtype=torch.float32))
    recording = _RecordingOperators(_linear_operators(matrix))
    tolerance = 1e-6
    result = randomized_range_finder(
        recording,
        seed=19,
        block_size=3,
        sample_cap=3,
        probe_count=3,
        probe_tolerance=tolerance,
    )

    assert result.randomized_reconstruction_error > tolerance
    assert result.exact_fallback_attempted is True
    assert result.exact_fallback_converged is True
    assert result.exact_fallback_reconstruction_error == 0.0
    assert result.reconstruction_error == 0.0
    assert result.converged is True
    assert result.basis_construction == "exact_coordinate_jvp_float64_svd"
    assert result.sample_count == result.sample_cap == matrix.shape[1]
    assert result.exact_fallback_jvp_calls == matrix.shape[1]
    assert result.jvp_calls == 3 + 3 + 3
    assert result.exact_fallback_rank == 3
    assert len(result.exact_fallback_coordinate_images_sha256) == 64
    assert len(result.exact_fallback_basis_float64_sha256) == 64
    assert torch.equal(
        torch.stack(recording.jvp_directions[-3:], dim=1),
        torch.eye(3, dtype=torch.float32),
    )
    assert torch.equal(
        result.basis @ (result.basis.T @ result.probe_images),
        result.probe_images,
    )

    factorization = factorize_masked_tangent(
        _linear_operators(matrix),
        rconds=(1e-5,),
        seed=19,
        block_size=3,
        sample_cap=3,
        probe_count=3,
        probe_tolerance=tolerance,
    )[1e-5]
    assert factorization.range_result.exact_fallback_converged is True
    assert factorization.range_result.rank == 3


def test_exact_coordinate_fallback_still_fails_closed_when_cast_basis_misses_tolerance():
    generator = torch.Generator().manual_seed(1)
    matrix = torch.randn(7, 4, generator=generator, dtype=torch.float32)
    operators = _linear_operators(matrix)

    with pytest.raises(MatrixFreeConvergenceError) as captured:
        factorize_masked_tangent(
            operators,
            rconds=(1e-5,),
            seed=8,
            block_size=4,
            sample_cap=4,
            probe_count=4,
            probe_tolerance=1e-12,
        )
    result = captured.value.result
    assert result.exact_fallback_attempted is True
    assert result.exact_fallback_converged is False
    assert result.exact_fallback_reconstruction_error > result.probe_tolerance
    assert result.converged is False
    assert "exact coordinate-JVP fallback also failed" in str(captured.value)


def test_zero_jacobian_converges_to_rank_zero_without_sampling():
    template = torch.zeros(6, dtype=torch.float64)
    bias = torch.linspace(-0.1, 0.1, 9, dtype=torch.float64)

    def zero_jacobian_closure(step):
        return bias + 0.0 * step.sum()

    operators = build_masked_jacobian_operators(zero_jacobian_closure, template)
    factorizations = factorize_masked_tangent(
        operators,
        seed=3,
        probe_count=4,
        probe_tolerance=1e-12,
    )
    for factorization in factorizations.values():
        assert factorization.rank == 0
        assert factorization.range_result.converged is True
        assert factorization.range_result.sample_count == 0
        residual = torch.linspace(-0.5, 0.5, 9, dtype=torch.float64)
        action = solve_damped_normal(factorization, residual, damping=1e-2)
        assert torch.equal(action.parameter_step, torch.zeros_like(template))
        assert torch.equal(action.predicted_action, torch.zeros_like(residual))
        assert action.predicted_energy_reduction == 0.0


def test_masks_and_shapes_fail_closed_before_derivative_work():
    matrix = torch.eye(4, dtype=torch.float64)

    with pytest.raises(ValueError, match="selects no rows"):
        _linear_operators(matrix, mask=torch.zeros(4, dtype=torch.bool))
    with pytest.raises(ValueError, match="mask length"):
        _linear_operators(matrix, mask=torch.ones(3, dtype=torch.bool))

    operators = _linear_operators(matrix)
    with pytest.raises(ValueError, match="JVP direction"):
        operators.jvp(torch.zeros(3, dtype=torch.float64))
    factorization = factorize_masked_tangent(
        operators,
        rconds=(1e-5,),
        probe_tolerance=1e-12,
    )[1e-5]
    with pytest.raises(ValueError, match="exactly six|shape"):
        residualize_six_columns(factorization, torch.zeros(4, 5, dtype=torch.float64))
    assert math.isfinite(factorization.cutoff)
