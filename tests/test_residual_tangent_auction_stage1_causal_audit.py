from __future__ import annotations

from benchmarks import residual_tangent_auction_stage1 as manifest_module
from benchmarks import residual_tangent_auction_stage1_causal_audit as audit


def _calibration_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    counter = 0
    for family in manifest_module.TARGET_FAMILIES:
        for seed in manifest_module.SEEDS:
            for horizon, iterations in manifest_module.PARENT_HORIZONS:
                for fold in manifest_module.FOLDS:
                    for damping in manifest_module.DAMPINGS:
                        for rcond in manifest_module.RCONDS:
                            for radius in manifest_module.TRUST_RADII:
                                for action in audit.ALL_ACTIONS:
                                    counter += 1
                                    predicted = 1e-4 + counter * 1e-9
                                    rows.append(
                                        {
                                            "cell_id": f"cell-{counter}",
                                            "selection_scope": "crossfit",
                                            "family": family,
                                            "seed": seed,
                                            "parent_horizon": horizon,
                                            "parent_iters": iterations,
                                            "heldout_fold": fold,
                                            "damping": damping,
                                            "rcond": rcond,
                                            "trust_radius": radius,
                                            "action": action,
                                            "base_mse": 0.1,
                                            "predicted_mse_gain": predicted,
                                            "realized_mse_gain": 1.1 * predicted,
                                        }
                                    )
    return rows


def test_audit_is_outside_every_experiment_source_binding() -> None:
    isolation = audit.source_binding_isolation()
    assert isolation["pass"] is True
    assert isolation["included_in_any_experiment_source_binding"] is False


def test_self_test_rejects_duplicate_keys_and_nonfinite_numbers() -> None:
    result = audit.run_self_test()
    assert result["status"] == "pass"
    assert result["checks"]["duplicate_json_keys_rejected"] is True
    assert result["checks"]["nonfinite_numbers_rejected"] is True
    assert result["checks"]["bootstrap_indices_deterministic"] is True


def test_all_eight_action_calibration_requires_and_accepts_all_16_strata() -> None:
    result = audit.all_eight_action_calibration(_calibration_rows())
    assert result["status"] == "available"
    assert result["raw_row_count"] == 3_072
    assert len(result["strata"]) == 16
    assert all(cell["raw_row_count"] == 192 for cell in result["strata"].values())
    assert result["gate_pass"] is True


def test_all_eight_action_calibration_fails_one_bad_causal_stratum() -> None:
    rows = _calibration_rows()
    selected = [row for row in rows if row["action"] == "full_j" and row["parent_iters"] == 160]
    for index, row in enumerate(selected):
        row["realized_mse_gain"] = float(selected[-index - 1]["predicted_mse_gain"])
    result = audit.all_eight_action_calibration(rows)
    assert result["gate_pass"] is False
    assert result["strata"]["action=full_j|parent_horizon=near_plateau"]["gate_pass"] is False


def test_strict_projector_condition_rejects_any_negative_unit() -> None:
    bootstrap = {"status": "available", "bootstrap_95_lower": 0.01}
    cell = {
        "rank_gain_compatible_with_added_truncated_direction": True,
        "negative_incremental_energy_unit_count": 0,
    }
    assert audit._strict_projector_condition_pass(cell, bootstrap) is True

    cell["negative_incremental_energy_unit_count"] = 1
    assert audit._strict_projector_condition_pass(cell, bootstrap) is False
