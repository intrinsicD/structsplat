from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from structsplat.contraction_refinement import (
    CoefficientProjectionConfig,
    project_contracted_coefficients,
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


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"tolerance": 0.0}, "tolerance"),
        ({"max_iterations": 0}, "max_iterations"),
        ({"ridge": -1.0}, "ridge"),
        ({"coefficient_abs_limit": 0.0}, "coefficient_abs_limit"),
    ],
)
def test_projection_config_fails_closed(kwargs, match):
    with pytest.raises((TypeError, ValueError), match=match):
        CoefficientProjectionConfig(**kwargs)
