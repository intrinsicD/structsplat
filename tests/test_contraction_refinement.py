from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from structsplat.contraction_refinement import (
    AlternatingGeometryConfig,
    CoefficientProjectionConfig,
    GeometryRelaxationConfig,
    alternate_projected_geometry,
    project_contracted_coefficients,
    relax_contracted_geometry,
    select_residual_anchor_leaves,
)
from structsplat.overlap_elimination import lattice_observation_field
from structsplat.pixel_contraction import render_observation_field


def test_residual_anchor_selection_is_exact_deterministic_and_spread():
    target = np.zeros((9, 11, 3), dtype=np.float32)
    reconstruction = target.copy()
    reconstruction[2, 3] = 1.0
    reconstruction[6, 8] = 0.9
    reconstruction[2, 4] = 0.8
    mask = np.ones(target.shape[:2], dtype=bool)

    first = select_residual_anchor_leaves(
        reconstruction,
        target,
        mask,
        5,
        patch_side=7,
        nms_radius_px=1,
    )
    second = select_residual_anchor_leaves(
        reconstruction,
        target,
        mask,
        5,
        patch_side=7,
        nms_radius_px=1,
    )

    assert first.selected_count == first.requested_count == 5
    assert first.nms_selected_count == 5
    assert first.protected_mask[2, 3]
    assert first.protected_mask[6, 8]
    assert not first.protected_mask[2, 4]
    assert np.array_equal(first.protected_mask, second.protected_mask)
    assert np.array_equal(first.score, second.score)
    assert not first.protected_mask.flags.writeable
    assert not first.score.flags.writeable


def _projection_fixture():
    mask = np.ones((9, 9), dtype=bool)
    basis = np.zeros(mask.shape, dtype=bool)
    basis[2, 2] = True
    basis[6, 6] = True
    true_coefficients = np.asarray(
        [[0.25, 0.70, 0.15], [0.85, 0.10, 0.45]], dtype=np.float32
    )
    true_field = lattice_observation_field(
        mask,
        basis,
        true_coefficients,
        scale_px=0.55,
        sigma_cutoff=3.0,
    )
    target = render_observation_field(true_field)
    initial_field = replace(true_field, rgb_coeff=np.zeros_like(true_coefficients))
    return initial_field, true_coefficients, target, mask


def test_projection_recovers_fixed_geometry_and_matches_maintained_renderer():
    field, true_coefficients, target, mask = _projection_fixture()
    means_before = field.means_xy.copy()
    scales_before = field.log_scales_xy.copy()
    rotations_before = field.rotations_rad.copy()
    touched = np.ones(field.n, dtype=bool)

    result = project_contracted_coefficients(
        field,
        target,
        mask,
        touched,
        config=CoefficientProjectionConfig(
            tolerance=1e-7,
            max_iterations=12,
            ridge=0.0,
            coefficient_abs_limit=2.0,
        ),
        render_chunk=1,
    )

    assert result.selected_iteration > 0
    assert result.final_sse < 1e-10
    assert result.final_sse < result.initial_sse
    assert result.adjoint_relative_error < 1e-6
    assert result.maintained_render_parity_max_abs < 2e-6
    assert np.allclose(result.field.rgb_coeff, true_coefficients, atol=2e-5)
    assert np.array_equal(result.field.means_xy, means_before)
    assert np.array_equal(result.field.log_scales_xy, scales_before)
    assert np.array_equal(result.field.rotations_rad, rotations_before)
    assert sum(checkpoint.selected for checkpoint in result.checkpoints) == 1
    assert all(
        checkpoint.raw_sse <= result.initial_sse + 1e-8
        for checkpoint in result.checkpoints
        if checkpoint.selectable
    )


def test_projection_freezes_protected_rows_and_fails_closed_on_coefficient_limit():
    field, _true_coefficients, target, mask = _projection_fixture()
    touched = np.ones(field.n, dtype=bool)
    protected = np.asarray([True, False])

    partial = project_contracted_coefficients(
        field,
        target,
        mask,
        touched,
        protected,
        config=CoefficientProjectionConfig(
            tolerance=1e-7,
            max_iterations=12,
            ridge=0.0,
            coefficient_abs_limit=2.0,
        ),
        render_chunk=1,
    )
    assert np.array_equal(partial.field.rgb_coeff[0], field.rgb_coeff[0])
    assert partial.trainable_rows == 1
    assert partial.protected_rows == 1

    rejected = project_contracted_coefficients(
        field,
        target,
        mask,
        touched,
        config=CoefficientProjectionConfig(
            tolerance=1e-7,
            max_iterations=12,
            ridge=0.0,
            coefficient_abs_limit=0.10,
        ),
        render_chunk=1,
    )
    assert rejected.selected_iteration == 0
    assert rejected.field.canonical_hash() == field.canonical_hash()
    assert rejected.final_sse == pytest.approx(rejected.initial_sse)
    assert any(not checkpoint.selectable for checkpoint in rejected.checkpoints[1:])

    over_limit_field = replace(
        field,
        rgb_coeff=np.full_like(field.rgb_coeff, 0.20),
    )
    stage_zero = project_contracted_coefficients(
        over_limit_field,
        target,
        mask,
        touched,
        config=CoefficientProjectionConfig(
            tolerance=1e-7,
            max_iterations=12,
            ridge=0.0,
            coefficient_abs_limit=0.10,
        ),
        render_chunk=1,
    )
    assert stage_zero.selected_iteration == 0
    assert len(stage_zero.checkpoints) == 1
    assert stage_zero.checkpoints[0].selected
    assert not stage_zero.checkpoints[0].selectable
    assert stage_zero.field.canonical_hash() == over_limit_field.canonical_hash()


def test_origin_reconditioning_discards_duplicate_column_nullspace_component():
    field, _true_coefficients, _target, mask = _projection_fixture()
    duplicate_geometry = replace(
        field,
        means_xy=np.repeat(field.means_xy[:1], 2, axis=0),
        log_scales_xy=np.repeat(field.log_scales_xy[:1], 2, axis=0),
        rotations_rad=np.repeat(field.rotations_rad[:1], 2, axis=0),
    )
    target_field = replace(
        duplicate_geometry,
        rgb_coeff=np.full((2, 3), 0.5, dtype=np.float32),
    )
    target = render_observation_field(target_field)
    unsafe = replace(
        duplicate_geometry,
        rgb_coeff=np.asarray([[20.0, 20.0, 20.0], [-19.0, -19.0, -19.0]], dtype=np.float32),
    )

    result = project_contracted_coefficients(
        unsafe,
        target,
        mask,
        np.ones(unsafe.n, dtype=bool),
        config=CoefficientProjectionConfig(
            tolerance=1e-7,
            max_iterations=12,
            ridge=0.0,
            coefficient_abs_limit=2.0,
            regularization_center="zero",
            solver_start="zero",
            frozen_base_mode="explicit",
            allow_unsafe_stage_zero_reconditioning=True,
        ),
        render_chunk=1,
    )

    assert not result.checkpoints[0].selectable
    assert result.selected_iteration > 0
    assert result.final_sse < 1e-10
    assert np.allclose(result.field.rgb_coeff, 0.5, atol=2e-5)
    assert np.max(np.abs(result.field.rgb_coeff)) <= 2.0
    assert result.initial_operator_parity_max_abs < 2e-6
    assert result.maintained_render_parity_max_abs < 2e-6


def test_explicit_frozen_base_preserves_partial_rows_and_recovers_remainder():
    field, true_coefficients, target, mask = _projection_fixture()
    partial_coefficients = np.zeros_like(true_coefficients)
    partial_coefficients[0] = true_coefficients[0]
    partial_field = replace(field, rgb_coeff=partial_coefficients)
    touched = np.asarray([False, True])

    result = project_contracted_coefficients(
        partial_field,
        target,
        mask,
        touched,
        config=CoefficientProjectionConfig(
            tolerance=1e-7,
            max_iterations=12,
            ridge=0.0,
            coefficient_abs_limit=2.0,
            regularization_center="zero",
            solver_start="zero",
            frozen_base_mode="explicit",
        ),
        render_chunk=1,
    )

    assert result.selected_iteration > 0
    assert result.frozen_rows == 1
    assert np.array_equal(result.field.rgb_coeff[0], partial_field.rgb_coeff[0])
    assert np.allclose(result.field.rgb_coeff[1], true_coefficients[1], atol=2e-5)
    assert result.final_sse < 1e-10
    assert result.initial_operator_parity_max_abs < 2e-6


def test_projection_legacy_modes_are_the_backward_compatible_default():
    field, _true_coefficients, target, mask = _projection_fixture()
    touched = np.ones(field.n, dtype=bool)
    common = dict(
        tolerance=1e-7,
        max_iterations=12,
        ridge=1e-8,
        coefficient_abs_limit=2.0,
    )

    default = project_contracted_coefficients(
        field,
        target,
        mask,
        touched,
        config=CoefficientProjectionConfig(**common),
        render_chunk=1,
    )
    explicit_legacy = project_contracted_coefficients(
        field,
        target,
        mask,
        touched,
        config=CoefficientProjectionConfig(
            **common,
            regularization_center="input",
            solver_start="input",
            frozen_base_mode="subtract",
            allow_unsafe_stage_zero_reconditioning=False,
        ),
        render_chunk=1,
    )

    assert default.field.canonical_hash() == explicit_legacy.field.canonical_hash()
    default_records = default.checkpoint_records()
    legacy_records = explicit_legacy.checkpoint_records()
    assert len(default_records) == len(legacy_records)
    for default_record, legacy_record in zip(default_records, legacy_records, strict=True):
        default_record.pop("elapsed_seconds")
        legacy_record.pop("elapsed_seconds")
        assert default_record == legacy_record
    assert np.array_equal(default.reconstruction_raw, explicit_legacy.reconstruction_raw)


def test_bounded_intermediate_can_cross_the_display_transaction_only_internally():
    mask = np.ones((9, 9), dtype=bool)
    basis = np.zeros(mask.shape, dtype=bool)
    basis[2, 2] = True
    basis[4, 4] = True
    basis[6, 6] = True
    geometry = lattice_observation_field(
        mask,
        basis,
        np.zeros((3, 3), dtype=np.float32),
        scale_px=1.2,
        sigma_cutoff=3.0,
    )
    rng = np.random.default_rng(4)
    field = replace(
        geometry,
        rgb_coeff=rng.uniform(-2.0, 2.0, (3, 3)).astype(np.float32),
    )
    target = rng.uniform(0.0, 1.0, (9, 9, 3)).astype(np.float32)
    common = dict(
        tolerance=1e-7,
        max_iterations=20,
        ridge=1e-8,
        coefficient_abs_limit=3.0,
        regularization_center="zero",
        solver_start="zero",
        frozen_base_mode="explicit",
        allow_unsafe_stage_zero_reconditioning=True,
        pixel_rmse_threshold=0.1,
        patch7_rmse_threshold=0.1,
    )

    guarded = project_contracted_coefficients(
        field,
        target,
        mask,
        np.ones(field.n, dtype=bool),
        config=CoefficientProjectionConfig(**common),
        render_chunk=2,
    )
    intermediate = project_contracted_coefficients(
        field,
        target,
        mask,
        np.ones(field.n, dtype=bool),
        config=CoefficientProjectionConfig(
            **common,
            selection_mode="bounded_intermediate",
        ),
        render_chunk=2,
    )

    selected = next(checkpoint for checkpoint in intermediate.checkpoints if checkpoint.selected)
    assert guarded.selected_iteration == 0
    assert guarded.field.canonical_hash() == field.canonical_hash()
    assert intermediate.selected_iteration > 0
    assert intermediate.final_sse < guarded.final_sse
    assert selected.bounded
    assert selected.selectable
    assert not selected.transaction_safe


def _geometry_escape_fixture():
    mask = np.ones((11, 11), dtype=bool)
    basis = np.zeros(mask.shape, dtype=bool)
    basis[2, 2] = True
    field = lattice_observation_field(
        mask,
        basis,
        np.asarray([[0.8, 0.3, 0.5]], dtype=np.float32),
        scale_px=0.8,
        sigma_cutoff=3.0,
    )
    target_field = replace(
        field,
        means_xy=np.asarray([[5.0, 5.0]], dtype=np.float32),
    )
    return field, render_observation_field(target_field), mask


def test_geometry_relaxation_freezes_rgb_and_obeys_global_anchor_trust_region():
    field, target, mask = _geometry_escape_fixture()
    config = GeometryRelaxationConfig(
        steps=80,
        checkpoint_every=10,
        lr_means=0.08,
        lr_scales=0.01,
        lr_rotations=0.005,
        max_mean_shift_px=0.6,
        max_log_scale_shift=0.25,
        max_rotation_shift_rad=0.2,
    )

    result = relax_contracted_geometry(
        field,
        target,
        mask,
        config=config,
        render_chunk=1,
    )

    assert result.selected_step > 0
    assert result.final_sse < result.initial_sse
    assert np.array_equal(result.field.rgb_coeff, field.rgb_coeff)
    assert result.mean_shift_max_px <= config.max_mean_shift_px + 1e-6
    assert result.log_scale_shift_max <= config.max_log_scale_shift + 1e-6
    assert result.rotation_shift_max_rad <= config.max_rotation_shift_rad + 1e-6
    assert result.maintained_render_parity_max_abs < 2e-6
    for name, value in field._array_items().items():
        if name not in {"means_xy", "log_scales_xy", "rotations_rad"}:
            assert np.array_equal(value, result.field._array_items()[name])


def test_alternating_geometry_escapes_and_final_transaction_is_deterministic():
    field, target, mask = _geometry_escape_fixture()
    projection = CoefficientProjectionConfig(
        tolerance=1e-7,
        max_iterations=20,
        ridge=1e-8,
        coefficient_abs_limit=2.0,
        regularization_center="zero",
        solver_start="zero",
        frozen_base_mode="explicit",
        allow_unsafe_stage_zero_reconditioning=True,
        selection_mode="bounded_intermediate",
    )
    geometry = GeometryRelaxationConfig(
        steps=100,
        checkpoint_every=10,
        lr_means=0.08,
        lr_scales=0.01,
        lr_rotations=0.005,
        max_mean_shift_px=4.0,
        max_log_scale_shift=0.5,
        max_rotation_shift_rad=0.5,
    )
    config = AlternatingGeometryConfig(
        rounds=1,
        geometry=geometry,
        projection=projection,
    )

    first = alternate_projected_geometry(
        field,
        target,
        mask,
        config=config,
        render_chunk=1,
    )
    second = alternate_projected_geometry(
        field,
        target,
        mask,
        config=config,
        render_chunk=1,
    )

    selected = next(checkpoint for checkpoint in first.checkpoints if checkpoint.selected)
    assert first.field.n == field.n
    assert first.selected_stage != "input"
    assert first.final_sse < 0.25 * first.initial_sse
    assert selected.transaction_safe
    assert np.max(np.abs(first.field.rgb_coeff)) <= projection.coefficient_abs_limit
    assert first.maintained_render_parity_max_abs < 2e-6
    assert np.linalg.norm(first.field.means_xy[0] - field.means_xy[0]) <= 4.0 + 1e-6
    assert first.field.canonical_hash() == second.field.canonical_hash()
    assert np.array_equal(first.reconstruction_raw, second.reconstruction_raw)
    assert first.total_geometry_steps == geometry.steps


def test_alternating_geometry_rejects_transactional_inner_projection():
    with pytest.raises(ValueError, match="bounded_intermediate"):
        AlternatingGeometryConfig(
            projection=CoefficientProjectionConfig(selection_mode="transaction")
        )


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"tolerance": 0.0}, "tolerance"),
        ({"max_iterations": 0}, "max_iterations"),
        ({"ridge": -1.0}, "ridge"),
        ({"coefficient_abs_limit": 0.0}, "coefficient_abs_limit"),
        ({"regularization_center": "old"}, "regularization_center"),
        ({"solver_start": "old"}, "solver_start"),
        ({"frozen_base_mode": "old"}, "frozen_base_mode"),
        ({"selection_mode": "old"}, "selection_mode"),
        (
            {"allow_unsafe_stage_zero_reconditioning": 1},
            "allow_unsafe_stage_zero_reconditioning",
        ),
    ],
)
def test_projection_config_fails_closed(kwargs, match):
    with pytest.raises((TypeError, ValueError), match=match):
        CoefficientProjectionConfig(**kwargs)
