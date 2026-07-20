from dataclasses import asdict
from pathlib import Path

import numpy as np

from benchmarks import local_linear_reproducing_assay as assay


def test_frozen_target_formulas_at_decisive_points() -> None:
    points = np.array(((0, 0), (63, 0), (0, 63), (63, 63), (31.5, 31.5)))
    constant = assay.target_values("constant", points)
    np.testing.assert_array_equal(constant, np.broadcast_to((0.25, 0.50, 0.75), (5, 3)))

    affine = assay.target_values("affine_xy", points)
    np.testing.assert_allclose(affine[0], (0.10, 0.80, 0.20), rtol=0.0, atol=0.0)
    np.testing.assert_allclose(affine[3], (0.70, 0.30, 0.80), rtol=0.0, atol=2e-16)
    np.testing.assert_allclose(affine[4], (0.40, 0.55, 0.50), rtol=0.0, atol=2e-16)

    quadratic = assay.target_values("quadratic", points)
    np.testing.assert_allclose(quadratic[0], (0.15, 0.75, 0.20), rtol=0.0, atol=0.0)
    np.testing.assert_allclose(quadratic[3], (0.75, 0.30, 0.60), rtol=0.0, atol=2e-16)
    np.testing.assert_array_equal(
        assay.target_values("vertical_step", points)[[0, 1]],
        np.array(((0.15, 0.30, 0.75), (0.85, 0.70, 0.20))),
    )
    np.testing.assert_array_equal(
        assay.target_values("diagonal_step", points)[[0, 3]],
        np.array(((0.10, 0.65, 0.25), (0.90, 0.20, 0.75))),
    )


def test_checker_uses_continuous_coordinates_and_frozen_parity() -> None:
    points = np.array(((0.0, 0.0), (7.999, 0.0), (8.0, 0.0), (8.0, 8.0)))
    actual = assay.target_values("checker8", points)
    low = np.array((0.15, 0.30, 0.75))
    high = np.array((0.85, 0.70, 0.20))
    np.testing.assert_array_equal(actual, np.stack((low, low, high, low)))


def test_float32_targets_are_a_single_contiguous_cast_of_float64() -> None:
    for target in assay.TARGETS:
        reference = np.ascontiguousarray(assay.target_image(target, dtype=np.float64)).astype(
            np.float32
        )
        candidate = assay.target_image(target, dtype=np.float32)
        assert candidate.flags.c_contiguous
        np.testing.assert_array_equal(candidate, reference)


def test_frozen_field_and_cell_grid_is_exact() -> None:
    fields = assay.field_specs()
    cells = assay.cell_specs()
    assert len(fields) == assay.EXPECTED_FIELDS == 63
    assert len(cells) == assay.EXPECTED_CELLS == 108
    assert len({row["field_id"] for row in fields}) == 63
    assert len({row["cell_id"] for row in cells}) == 108
    assert sum(row["cohort"] == "target_conditioned" for row in fields) == 54
    assert sum(row["cohort"] == "shared_constant" for row in fields) == 9
    assert sum(row["cohort"] == "target_conditioned" for row in cells) == 54
    assert sum(row["cohort"] == "shared_constant" for row in cells) == 54
    shared = [row for row in fields if row["cohort"] == "shared_constant"]
    assert all(row["evaluation_targets"] == list(assay.TARGETS) for row in shared)


def test_explicit_initializer_and_structure_tensor_configuration() -> None:
    init, structure = assay.frozen_configs(56, 2)
    values = asdict(init)
    assert values["strategy"] == "quadtree_wse"
    assert values["num_gaussians"] == 56
    assert values["seed"] == 2
    assert values["candidate_oversample"] == 6.0
    assert values["wse_progressive_order"] is False
    assert values["flank_offset_frac"] == 0.0
    assert values["opacity_mode"] == "none"
    assert asdict(structure) == {
        "grad_sigma": 1.0,
        "tensor_sigma": 2.0,
        "gradient_operator": "central",
        "color_space": "luma",
        "flat_frac": 0.02,
        "corner_frac": 0.15,
    }


def test_plumbing_controls_validate_without_relaxation() -> None:
    controls = assay.plumbing_controls()
    assert controls["valid"] is True
    assert controls["positive_affine"]["accepted"] is True
    assert controls["positive_affine"]["affine_max_abs_error"] <= 1e-12
    assert controls["negative_collinear"]["rejected_by_rank"] is True
    assert controls["negative_one_node"]["rejected_by_contributor_and_rank"] is True


def test_decision_helper_distinguishes_kill_pending_and_unavailable() -> None:
    kill = assay.derive_decision(
        controls_valid=True,
        artifact_audit_passed=True,
        complete=True,
        early_failure_count=1,
    )
    assert kill == {
        "decision": "kill",
        "classification": "stage0_kill",
        "downstream_status": assay.SHORT_CIRCUIT,
    }
    pending = assay.derive_decision(
        controls_valid=True,
        artifact_audit_passed=True,
        complete=True,
        early_failure_count=0,
    )
    assert pending["classification"] == "early_screen_pass_downstream_required"
    assert pending["downstream_status"] == assay.PENDING
    unavailable = assay.derive_decision(
        controls_valid=False,
        artifact_audit_passed=True,
        complete=True,
        early_failure_count=0,
    )
    assert unavailable["decision"] == "unavailable"


def test_deterministic_npz_has_stable_bytes_and_raw_infinity(tmp_path: Path) -> None:
    arrays = {
        "finite": np.arange(6, dtype=np.float64).reshape(2, 3),
        "condition": np.array((1.0, np.inf), dtype=np.float64),
    }
    first = tmp_path / "first.npz"
    second = tmp_path / "second.npz"
    assay._deterministic_npz(first, arrays)
    assay._deterministic_npz(second, arrays)
    assert first.read_bytes() == second.read_bytes()
    with np.load(first, allow_pickle=False) as archive:
        np.testing.assert_array_equal(archive["finite"], arrays["finite"])
        assert np.isinf(archive["condition"][1])
