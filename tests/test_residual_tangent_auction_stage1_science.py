# ruff: noqa: E402
import json

import pytest

torch = pytest.importorskip("torch")

from benchmarks import residual_tangent_auction_stage1_analysis as analysis
from benchmarks.residual_tangent_auction_stage1_science import (
    CELLS_PER_SCOPE_UNIT,
    EXPECTED_CROSSFIT_CELLS,
    EXPECTED_ORACLE_CELLS,
    EXPECTED_SMALL_CARRIER_SENSITIVITY_CELLS,
    _evidence_projection,
    _load_jsonl,
    _manifest_core_cells,
    _manifest_sensitivity_cells,
    _solve_fixed_six,
    _unit_groups,
    run_toy_preflight,
    validate_toy_preflight,
)


@pytest.fixture(scope="module")
def toy_result():
    return run_toy_preflight()


def test_toy_preflight_exercises_full_path_without_development_targets(toy_result):
    assert toy_result["preflight_pass"] is True
    assert toy_result["development_targets_accessed"] is False
    assert toy_result["scientific_cells_present"] is False
    assert toy_result["scope_action_record_count"] == 24
    assert all(toy_result["checks"].values())


def test_toy_preflight_validates_jvp_birth_parity_support_and_recovery(toy_result):
    assert toy_result["jvp"]["relative_l2"] <= 1e-3
    assert toy_result["birth_direct_parity_max_abs"] <= 1e-10
    assert toy_result["support_crossing"]["xor_memberships"] > 0
    assert toy_result["recovery_one_step"]["optimizer"] == "fresh_adam"
    assert toy_result["recovery_one_step"]["native_dynamic_support"] is True


def test_manifest_partition_is_exactly_72_resumable_scope_units():
    cells = _manifest_core_cells()
    crossfit = [item for item in cells if item["selection_scope"] == "crossfit"]
    oracle = [item for item in cells if item["selection_scope"] != "crossfit"]
    groups = _unit_groups(cells)
    assert len(crossfit) == EXPECTED_CROSSFIT_CELLS
    assert len(oracle) == EXPECTED_ORACLE_CELLS
    assert len(groups) == 72
    assert {len(value) for value in groups.values()} == {CELLS_PER_SCOPE_UNIT}
    sensitivity = _manifest_sensitivity_cells()
    assert len(sensitivity) == EXPECTED_SMALL_CARRIER_SENSITIVITY_CELLS == 288


def test_persisted_toy_preflight_is_live_source_bound(tmp_path, toy_result):
    path = tmp_path / "toy.json"
    path.write_text(
        json.dumps(toy_result, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    validated = validate_toy_preflight(path)
    assert validated["binding_sha256"] == toy_result["binding_sha256"]


def test_fixed_six_zero_damping_uses_truncated_minimum_norm_solve():
    generator = torch.Generator().manual_seed(93)
    design = torch.randn(40, 6, generator=generator, dtype=torch.float64)
    design[:, 5] = design[:, 4]
    truth = torch.tensor([0.3, -0.2, 0.1, 0.05, 0.4, 0.4], dtype=torch.float64)
    residual = design @ truth
    coefficients, rank, condition = _solve_fixed_six(
        design, residual, damping=0.0, rcond=1e-5
    )
    assert rank == 5
    assert condition is not None
    assert torch.allclose(design @ coefficients, residual, atol=1e-11, rtol=1e-11)
    assert coefficients[4] == pytest.approx(coefficients[5], abs=1e-12)


def test_analyzer_projection_uses_predicted_energy_and_realized_psnr():
    rich = {
        "selection_scope": "crossfit",
        "family": "ramp_shading",
        "seed": 101,
        "parent_iters": 160,
        "heldout_fold": 0,
        "action": "affine6",
        "damping": 1e-3,
        "rcond": 1e-5,
        "trust_radius": 0.25,
        "status": "ok",
        "error": None,
        "binding_sha256": "a" * 64,
        "base_mse": 0.01,
        "predicted_mse_gain": 0.002,
        "realized_mse_gain": 0.001,
        "realized_psnr": 21.0,
        "cell_id": "rich-cell",
        "identity_sha256": "b" * 64,
        "support": {},
    }
    mask = torch.tensor([True, False, True, True])
    projected = _evidence_projection(rich, mask)
    assert projected["residual_energy_reduction"] == pytest.approx(0.006)
    assert projected["realized_gain"] == pytest.approx(0.001)
    assert projected["psnr_db"] == pytest.approx(21.0)
    assert projected["cell_id"] == analysis.stable_cell_id(projected)


def test_jsonl_loader_rejects_duplicate_terminal_ids(tmp_path):
    path = tmp_path / "rows.jsonl"
    row = {"cell_id": "same", "status": "ok"}
    path.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate cell_id"):
        _load_jsonl(path, "cell_id")
