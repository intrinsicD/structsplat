# ruff: noqa: E402
import json

import numpy as np
import pytest

pytest.importorskip("skimage")
torch = pytest.importorskip("torch")

from benchmarks import residual_tangent_auction_stage1 as S


@pytest.fixture(scope="module")
def manifest_and_preflight():
    torch.set_num_threads(1)
    return S.run_preflight()


def test_frozen_target_suite_has_exact_hashes_and_is_disjoint_from_prior_screens():
    records = S.target_suite()
    assert tuple(records) == S.TARGET_FAMILIES
    assert {family: item["pixel_sha256"] for family, item in records.items()} == (
        S.EXPECTED_TARGET_HASHES
    )
    assert len({item["pixel_sha256"] for item in records.values()}) == 6
    for item in records.values():
        image = item["image"]
        assert image.shape == (64, 64, 3)
        assert image.dtype == np.float32
        assert np.isfinite(image).all()
        assert float(image.min()) >= 0.0
        assert float(image.max()) <= 1.0
        assert item["split"] == "development"

    natural = records["natural_texture"]["metadata"]
    assert natural["provider"] == "skimage.data.astronaut"
    assert natural["crop_yx"] == [176, 240, 208, 272]
    assert natural["resize"] is None
    assert len(natural["source_sha256"]) == 64

    prior = S._prior_target_hashes()
    current = {item["pixel_sha256"] for item in records.values()}
    assert current.isdisjoint(prior["FIT-020"])
    assert current.isdisjoint(prior["COMP-006"])


def test_checkerboard_folds_are_balanced_disjoint_and_exhaustive():
    masks = S.fold_masks()
    assert set(masks) == {0, 1}
    assert all(mask.shape == (64, 64) and mask.dtype == np.bool_ for mask in masks.values())
    assert int(masks[0].sum()) == 2048
    assert int(masks[1].sum()) == 2048
    assert not np.logical_and(masks[0], masks[1]).any()
    assert np.logical_or(masks[0], masks[1]).all()
    for tile_y in range(8):
        for tile_x in range(8):
            tile = masks[(tile_y + tile_x) % 2][
                tile_y * 8 : (tile_y + 1) * 8,
                tile_x * 8 : (tile_x + 1) * 8,
            ]
            assert tile.all()


def test_manifest_enumerates_named_component_counts_without_duplicate_keys(
    manifest_and_preflight,
):
    manifest, _preflight = manifest_and_preflight
    counts = manifest["expected_counts"]
    assert counts["crossfit_immediate_action_cells"] == 3072
    assert counts["all_pixel_oracle_action_cells"] == 1536
    assert counts["recovery_trajectories"] == 816
    assert counts["primary_component_cells"] == 3072 + 1536 + 816 == 5424
    assert counts["new_recovery_checkpoint_renders"] == 1632
    assert counts["logical_recovery_checkpoint_records"] == 2448
    assert counts["unregularized_nonbirth_linear_score_rows"] == 864
    assert counts["finite_birth_affine_action_score_rows"] == 288
    assert counts["total_diagnostic_score_rows"] == 1152
    assert counts["parent_checkpoints"] == 24

    action_expected = sum(
        counts[key]
        for key in (
            "crossfit_immediate_action_cells",
            "all_pixel_oracle_action_cells",
            "crossfit_no_action_baselines",
            "all_pixel_no_action_baselines",
            "carrier_bank_sensitivity_crossfit",
            "carrier_bank_sensitivity_oracle",
        )
    )
    assert len(manifest["action_cells"]) == action_expected
    assert len(manifest["recovery_plan"]) == 816
    assert len(manifest["score_plan"]) == 1152
    for key in ("action_cells", "selection_plan", "score_plan", "recovery_plan"):
        ids = [row["cell_id"] for row in manifest[key]]
        assert len(ids) == len(set(ids))

    statistics = manifest["statistics"]
    assert statistics["crossfit_fit_policy"].startswith(
        "select_identity_and_fit_six_coefficients_on_discovery"
    )
    assert statistics["crossfit_heldout"] == "transported_gain_without_heldout_refit"
    assert statistics["post_selection_all_pixel_refit"] == "excluded_oracle_or_recovery_vector"


def test_score_plan_keeps_birth_offset_and_oracle_labels_separate(
    manifest_and_preflight,
):
    manifest, _preflight = manifest_and_preflight
    assert manifest["axes"]["rank_thresholds"] == list(S.RCONDS)
    assert manifest["gates"]["stability_gate_axes"] == [
        "rank_threshold",
        "trust_radius",
    ]
    assert manifest["axes"]["damping_coordinate_system"] == (
        "dimensionless_parent_chart"
    )
    assert manifest["statistics"]["column_normalized_sensitivity_ridge"] == (
        "diagnostic_only"
    )
    for row in manifest["score_plan"]:
        if "birth" in row["action"]:
            assert row["statistic"].startswith("finite_affine_action_gain_")
        elif row["selection_scope"] == "crossfit":
            assert row["statistic"] == "unregularized_projector_energy_discovery"
        else:
            assert row["statistic"] == "unregularized_projector_energy_all_pixel_oracle"
        if row["selection_scope"] != "crossfit":
            assert row["selection_scope"] == (
                "all_pixel_identity_and_coefficient_oracle"
            )
        assert row["rank_threshold"] == row["rcond"]
        assert row["damping"] is None
        assert row["trust_radius"] is None


def test_no_opacity_n64_parent_and_centered_jvp_path_pass():
    target = S.target_suite()["natural_texture"]["image"]
    field = S.build_no_opacity_parent(target)
    assert field.n == 64
    assert field.opacities is None
    vector = S._pack_no_opacity(field)
    assert vector.numel() == 512
    chart = S.coordinate_chart_preflight(field)
    assert chart["mean_test_is_rotated_anisotropic"] is True
    assert chart["log_scale_unit_delta"] == pytest.approx(1.0)
    assert chart["rotation_unit_delta_radians"] == pytest.approx(1.0)
    assert chart["display_linear_rgb_unit_delta"] == pytest.approx(1.0)
    assert chart["independent_core_frozen_render_max_abs"] <= 1e-10
    assert chart["pass"] is True
    check = S.no_opacity_jvp_preflight(field)
    assert check["opacity_present"] is False
    assert check["parameter_columns"] == 512
    assert check["relative_l2"] <= 1e-3
    assert check["pass"] is True


def test_joint_two_birth_solve_uses_updated_base_and_common_denominator():
    target = S.target_suite()["ramp_shading"]["image"]
    field = S.build_no_opacity_parent(target)
    check = S.joint_two_birth_preflight(field)
    assert check["birth_count"] == 2
    assert check["optimized_rgb_coefficients"] == 6
    assert check["common_denominator"] is True
    assert check["birth_geometry"] == "fixed_before_rgb_solve"
    assert check["finite_realization_support"] == "native_recentered_recomputed_aabb"
    assert check["parent_support_membership_changes"] > 0
    assert check["parent_support_change"]["xor_count"] == (
        check["parent_support_change"]["added_count"]
        + check["parent_support_change"]["removed_count"]
    )
    assert check["birth_support"]["row_count"] == 2
    assert check["birth_support"]["fixed_geometry"] is True
    assert len(check["birth_support"]["membership_sha256"]) == 64
    assert check["parent_opacity_present"] is False
    assert check["nonzero_base_update_trust_radius"] == 0.75
    assert check["formula_direct_render_max_abs"] <= 1e-10
    assert check["solved_render_max_abs"] <= 1e-10
    assert check["solved_color_max_abs"] <= 1e-10
    assert check["pass"] is True


def test_joint_affine_and_carrier_realizations_pass_at_both_linf_radii():
    target = S.target_suite()["stationary_sinusoid"]["image"]
    field = S.build_no_opacity_parent(target)
    check = S.joint_realization_preflight(field)
    assert check["trust_norm"] == "linf"
    assert check["dimensionless_action_scaling"] == "one_uniform_linf_scale"
    assert check["finite_realization_support"] == "native_recentered_recomputed_aabb"
    assert set(check["radii"]) == {"0.25", "0.75"}
    for values in check["radii"].values():
        assert values["updated_opacity_present"] is False
        assert values["affine_joint_max_abs"] <= 1e-10
        assert values["carrier_joint_max_abs"] <= 1e-10
        assert values["support_membership_changes"] > 0
        assert values["support_change"]["xor_count"] == (
            values["support_change"]["added_count"]
            + values["support_change"]["removed_count"]
        )
        assert len(values["support_change"]["parent_membership_sha256"]) == 64
        assert len(values["support_change"]["native_membership_sha256"]) == 64
        assert values["pass"] is True
    assert check["pass"] is True


def test_crossfit_fits_identity_and_coefficients_on_discovery_without_heldout_refit():
    check = S.crossfit_leakage_preflight()
    for values in check["folds"].values():
        assert values["fit_policy"] == "identity_and_coefficients_fit_on_discovery_only"
        assert values["heldout_refit"] is False
        assert values["selected"] == values["selected_after_heldout_perturbation"]
        assert values["discovery_residual_exactly_equal"] is True
        assert values["discovery_input_hash_exactly_equal"] is True
        assert values["discovery_input_sha256"] == (
            values["discovery_input_sha256_after_heldout_perturbation"]
        )
        assert values["numerical_repeatability_absolute_tolerance"] == 1e-12
        assert values["coefficient_max_abs_change_after_heldout_perturbation"] == 0.0
        assert values["prediction_max_abs_change_after_heldout_perturbation"] == 0.0
        assert values["heldout_score_changed"] is True
        assert values["unchanged"] is True
    assert check["pass"] is True


def test_numerical_preflight_passes_but_development_remains_fail_closed(
    manifest_and_preflight,
):
    manifest, preflight = manifest_and_preflight
    assert preflight["preflight_pass"] is True, {
        name: check for name, check in preflight["checks"].items() if not check["pass"]
    }
    assert all(item["pass"] for item in preflight["checks"].values())
    assert preflight["scientific_cells_present"] is False
    assert preflight["development_runner_present"] is False
    assert preflight["recovery_runner_present"] is False
    assert preflight["development_run_authorized"] is False
    assert preflight["method_promotion_authorized"] is False
    assert preflight["compression_claim_authorized"] is False
    assert preflight["damping_coordinate_system"] == "dimensionless_parent_chart"
    assert preflight["column_normalized_sensitivity_ridge"] == "diagnostic_only"
    assert manifest["development_run_authorized"] is False
    assert manifest["scientific_cells_present"] is False


def test_cli_writes_strict_source_bound_artifacts(tmp_path):
    outdir = tmp_path / "stage1_preflight"
    assert S.main(["--outdir", str(outdir)]) == 0
    assert {path.name for path in outdir.iterdir()} == {
        "config.json",
        "manifest.json",
        "preflight.json",
        "summary.md",
    }

    def reject_constant(value):
        raise AssertionError(f"non-strict JSON constant {value}")

    loaded = {}
    for name in ("config.json", "manifest.json", "preflight.json"):
        text = (outdir / name).read_text()
        assert "Infinity" not in text
        assert "NaN" not in text
        loaded[name] = json.loads(text, parse_constant=reject_constant)
    config = loaded["config.json"]
    preflight = loaded["preflight.json"]
    critical = config["provenance"]["critical_sources"]
    assert "benchmarks/residual_tangent_auction_stage1.py" in critical
    assert set(S.CRITICAL_SOURCE_PATHS) <= set(critical)
    assert len(critical["benchmarks/residual_tangent_auction_stage1.py"]["sha256"]) == 64
    assert config["expected_counts"]["primary_component_cells"] == 5424
    assert config["constants"]["damping_coordinate_system"] == (
        "dimensionless_parent_chart"
    )
    assert preflight["preflight_pass"] is True
    assert preflight["development_run_authorized"] is False
    summary = (outdir / "summary.md").read_text()
    assert "Development run authorized:** no" in summary
    assert "no quality, convergence, speed" in summary
