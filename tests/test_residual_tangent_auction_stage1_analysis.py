from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from benchmarks import residual_tangent_auction_stage1_analysis as A


BINDING = "b" * 64


def _hash(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _record(
    *,
    family: str = A.TARGET_FAMILIES[0],
    seed: int = 101,
    parent_checkpoint: int = 160,
    fold: int = 0,
    action: str = "tangent6",
    damping: float = 1e-3,
    rcond: float = 1e-5,
    trust_radius: float = 0.25,
    checkpoint: int = 0,
    counter: int = 1,
    component: str | None = None,
    scope: str = A.CROSS_FIT_SCOPE,
) -> dict[str, Any]:
    if component is None:
        component = "crossfit_immediate" if checkpoint == 0 else "crossfit_recovery"
    parent_mse = 0.1
    predicted = 1e-3 + counter * 1e-8
    record: dict[str, Any] = {
        "schema": A.SCHEMA_ID,
        "record_kind": A.RECORD_KIND,
        "component": component,
        "status": "ok",
        "error": None,
        "binding_sha256": BINDING,
        "selection_scope": scope,
        "ledger": A.MATCHED_LEDGER,
        "family": family,
        "seed": seed,
        "parent_checkpoint": parent_checkpoint,
        "fold": fold,
        "action": action,
        "damping": damping,
        "rcond": rcond,
        "trust_radius": trust_radius,
        "checkpoint": checkpoint,
        "scoring_mask_sha256": _hash(
            f"mask/{scope}/{family}/{seed}/{parent_checkpoint}/{fold}/{checkpoint}"
        ),
        "parent_mse": parent_mse,
        "predicted_gain": predicted,
        "realized_gain": 1.1 * predicted,
        "residual_energy_reduction": 1.0,
        "psnr_db": 30.0,
        "metadata": {"synthetic": True},
    }
    record["cell_id"] = A.stable_cell_id(record)
    return record


def _passing_records() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    counter = 0
    action_psnr = {
        "tangent6": 0.00,
        "affine6": 0.30,
        "carrier6": 0.02,
        "birth2_rgb6": 0.05,
    }
    action_energy = {
        "tangent6": 1.00,
        "affine6": 1.40,
        "carrier6": 1.00,
        "birth2_rgb6": 1.05,
    }
    for family_index, family in enumerate(A.TARGET_FAMILIES):
        for seed in A.SEEDS:
            for parent_checkpoint in A.PARENT_CHECKPOINTS:
                for fold in A.FOLDS:
                    base = (
                        30.0
                        + 0.01 * family_index
                        + 0.0001 * (seed - 100)
                        + 0.001 * fold
                        + 0.00001 * parent_checkpoint
                    )
                    for damping in A.DAMPINGS:
                        for rcond in A.RCONDS:
                            for radius in A.TRUST_RADII:
                                for action in A.ACTIONS:
                                    counter += 1
                                    record = _record(
                                        family=family,
                                        seed=seed,
                                        parent_checkpoint=parent_checkpoint,
                                        fold=fold,
                                        action=action,
                                        damping=damping,
                                        rcond=rcond,
                                        trust_radius=radius,
                                        checkpoint=0,
                                        counter=counter,
                                    )
                                    record["psnr_db"] = base + action_psnr[action]
                                    record["residual_energy_reduction"] = action_energy[action]
                                    records.append(record)
    recovery_psnr = {
        "tangent6": 0.00,
        "affine6": 0.15,
        "carrier6": 0.02,
        "birth2_rgb6": 0.05,
    }
    for family_index, family in enumerate(A.TARGET_FAMILIES):
        for seed in A.SEEDS:
            for fold in A.FOLDS:
                base = 31.0 + 0.01 * family_index + 0.001 * fold
                for radius in A.TRUST_RADII:
                    for checkpoint in A.RECOVERY_CHECKPOINTS:
                        for action in A.ACTIONS:
                            counter += 1
                            record = _record(
                                family=family,
                                seed=seed,
                                parent_checkpoint=A.NEAR_PLATEAU_CHECKPOINT,
                                fold=fold,
                                action=action,
                                damping=A.PRIMARY_DAMPING,
                                rcond=A.PRIMARY_RCOND,
                                trust_radius=radius,
                                checkpoint=checkpoint,
                                counter=counter,
                            )
                            # Keep a positive 0.10 dB affine advantage over the stronger birth
                            # after recovery.  Carrier remains below that native control.
                            record["psnr_db"] = base + recovery_psnr[action]
                            record["residual_energy_reduction"] = action_energy[action]
                            records.append(record)
    return records


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False)
            + "\n"
            for record in records
        ),
        encoding="utf-8",
    )


@pytest.fixture(scope="module")
def passing_records() -> list[dict[str, Any]]:
    return _passing_records()


@pytest.fixture(scope="module")
def passing_analysis(tmp_path_factory, passing_records):
    path = tmp_path_factory.mktemp("bench009-analysis") / "evidence.jsonl"
    _write_jsonl(path, passing_records)
    expected = {record["cell_id"] for record in passing_records}
    loaded = A.load_jsonl(
        path, expected_ids=expected, expected_binding_sha256=BINDING
    )
    return loaded, A.analyze_evidence(loaded)


def test_frozen_schema_count_and_synthetic_passing_evidence(passing_analysis):
    loaded, analysis = passing_analysis
    immediate_count = 6 * 2 * 2 * 2 * 4 * 2 * 2 * 2
    recovery_count = 6 * 2 * 2 * 4 * 2 * 2
    assert immediate_count == 1536
    assert recovery_count == 384
    assert loaded.available
    assert loaded.observed_count == immediate_count + recovery_count == 1920
    assert analysis["status"] == "available"
    assert analysis["gates"] == {
        "validity_pass": True,
        "representation_survival_pass": True,
        "matched_six_dof_utility_screen_pass": True,
        "causal_outside_j_audit_present": False,
        "objective_alignment_audit_present": False,
        "scientific_interpretation_authorized": False,
    }
    assert analysis["interpretation_boundary"][
        "matched_survival_is_not_outside_j_evidence"
    ] is True
    assert analysis["calibration"]["raw_row_count"] == 1536
    assert analysis["calibration"]["at_or_below_floor_count"] == 0
    assert analysis["calibration"]["spearman"] == pytest.approx(1.0)
    assert analysis["calibration"]["median_realized_over_predicted"] == pytest.approx(
        1.1
    )
    assert analysis["calibration"]["all_action_horizon_strata_pass"] is True
    assert len(analysis["calibration"]["strata"]) == 8
    assert analysis["winner_stability"]["changed_family_count"] == 0
    survival = analysis["matched_survival"]["action_survival"]
    assert survival["affine6"]["survives"] is True
    assert survival["carrier6"]["survives"] is False
    affine_cell = analysis["matched_survival"]["cells"]["affine6"][
        "radius=0.25|checkpoint=0"
    ]
    assert affine_cell["positive_family_count"] == 6
    assert affine_cell["bootstrap_replicates"] == 100_000
    assert affine_cell["bootstrap_rng"] == "numpy.PCG64"
    assert affine_cell["bootstrap_seed"] == 20_260_715
    assert affine_cell["residual_energy_ratio_vs_stronger_control"] == pytest.approx(
        1.4 / 1.05
    )


def test_gain_floor_retains_rows_but_excludes_only_calibration_statistics():
    rows = [_record(action=action, counter=index) for index, action in enumerate(A.ACTIONS)]
    floor = rows[0]["parent_mse"] * 1e-5
    rows[0]["predicted_gain"] = floor
    rows[1]["predicted_gain"] = floor / 2.0
    rows[2]["predicted_gain"] = 2.0 * floor
    rows[3]["predicted_gain"] = 3.0 * floor
    for index, row in enumerate(rows):
        row["realized_gain"] = row["predicted_gain"] * (1.0 + index * 0.01)
    result = A.calibration_statistics(rows)
    assert result["status"] == "available"
    assert result["raw_row_count"] == 4
    assert result["at_or_below_floor_count"] == 2
    assert result["eligible_row_count"] == 2
    assert result["enough_non_tied_eligible_rows"] is False
    assert result["failure_reason"] == "fewer than three eligible non-tied rows"
    assert result["gate_pass"] is False


def test_winner_changes_in_two_families_fail_frozen_stability(passing_records):
    records = copy.deepcopy(passing_records)
    changed = set(A.TARGET_FAMILIES[:2])
    for record in records:
        if (
            record["component"] == "crossfit_immediate"
            and record["family"] in changed
            and record["damping"] == A.DAMPINGS[1]
            and record["action"] == "affine6"
        ):
            record["psnr_db"] -= 1.0
    result = A.winner_stability(records)
    assert result["status"] == "available"
    assert set(result["changed_families"]) == changed
    assert result["changed_family_count"] == 2
    assert result["gate_pass"] is False


def test_opposite_horizon_instabilities_cannot_cancel_when_pooled(passing_records):
    records = copy.deepcopy(passing_records)
    changed = set(A.TARGET_FAMILIES[:2])
    for record in records:
        if (
            record["component"] == "crossfit_immediate"
            and record["family"] in changed
            and record["action"] == "affine6"
            and (
                (
                    record["parent_checkpoint"] == 24
                    and record["damping"] == A.DAMPINGS[1]
                )
                or (
                    record["parent_checkpoint"] == 160
                    and record["damping"] == A.DAMPINGS[0]
                )
            )
        ):
            # Affine is normally +0.30 dB and birth +0.05 dB.  Moving affine to
            # -0.10 dB makes birth win in one opposite damping cell per horizon.  If the
            # horizons were pooled, affine would still average +0.10 dB and appear stable.
            record["psnr_db"] -= 0.40
    result = A.winner_stability(records)
    assert result["status"] == "available"
    assert set(result["changed_families"]) == changed
    assert result["changed_family_count"] == 2
    assert result["gate_pass"] is False
    for family in changed:
        by_horizon = result["unique_winners_by_family_and_parent_checkpoint"][family]
        assert set(by_horizon["24"]) == {"affine6", "birth2_rgb6"}
        assert set(by_horizon["160"]) == {"affine6", "birth2_rgb6"}


def test_negative_long_recovery_closes_affine_even_when_step_zero_passes(
    passing_records,
):
    records = copy.deepcopy(passing_records)
    for record in records:
        if (
            record["component"] == "crossfit_recovery"
            and record["checkpoint"] == 100
            and record["action"] == "affine6"
        ):
            record["psnr_db"] -= 0.2
    result = A.matched_survival_statistics(records)
    assert result["status"] == "available"
    assert result["action_survival"]["affine6"]["survives"] is False
    for radius_result in result["action_survival"]["affine6"]["radii"].values():
        assert radius_result["checks"]["energy_ratio_at_least_1.25"] is True
        assert (
            radius_result["checks"]["immediate_psnr_delta_at_least_0.15_db"]
            is True
        )
        assert radius_result["checks"]["recovery_mean_positive"]["100"] is False


def test_oracle_rows_cannot_change_any_gate(passing_records, passing_analysis):
    _loaded, reference = passing_analysis
    records = copy.deepcopy(passing_records)
    oracle = _record(
        family=A.TARGET_FAMILIES[0],
        fold=-1,
        action="carrier6",
        component="all_pixel_oracle",
        scope=A.ORACLE_SCOPE,
    )
    oracle.update(
        predicted_gain=1e9,
        realized_gain=1e9,
        residual_energy_reduction=1e9,
        psnr_db=1e9,
    )
    records.append(oracle)
    evidence = A.EvidenceLoad(
        status="available",
        records=tuple(records),
        errors=(),
        expected_count=len(records),
        observed_count=len(records),
    )
    result = A.analyze_evidence(evidence)
    assert result["status"] == "available"
    assert result["oracle_rows_excluded"] == 1
    assert result["gates"] == reference["gates"]
    assert result["calibration"] == reference["calibration"]
    assert result["winner_stability"] == reference["winner_stability"]
    assert result["matched_survival"] == reference["matched_survival"]


@pytest.mark.parametrize("failure", ["duplicate", "missing", "stale", "error", "nonfinite"])
def test_jsonl_integrity_failures_are_unavailable(tmp_path, failure):
    first = _record(action="tangent6")
    second = _record(action="affine6")
    records = [first, second]
    expected = {first["cell_id"], second["cell_id"]}
    path = tmp_path / "evidence.jsonl"
    expected_binding = BINDING
    if failure == "duplicate":
        records = [first, first]
    elif failure == "missing":
        records = [first]
    elif failure == "stale":
        expected_binding = "c" * 64
    elif failure == "error":
        second["status"] = "error"
        second["error"] = "synthetic failure"
    if failure == "nonfinite":
        _write_jsonl(path, records)
        payload = path.read_text(encoding="utf-8").replace(
            '"predicted_gain":0.00100001', '"predicted_gain":1e309', 1
        )
        path.write_text(payload, encoding="utf-8")
    else:
        _write_jsonl(path, records)
    loaded = A.load_jsonl(
        path,
        expected_ids=expected,
        expected_binding_sha256=expected_binding,
    )
    assert loaded.status == "unavailable"
    assert loaded.errors
    analyzed = A.analyze_evidence(loaded)
    assert analyzed["status"] == "unavailable"
    assert analyzed["gates"] is None


def test_duplicate_json_keys_and_stable_key_hash_mismatch_fail_closed(tmp_path):
    record = _record()
    path = tmp_path / "evidence.jsonl"
    payload = json.dumps(record, sort_keys=True, separators=(",", ":"))
    payload = payload[:-1] + ',"seed":999}'
    path.write_text(payload + "\n", encoding="utf-8")
    duplicate = A.load_jsonl(
        path,
        expected_ids={record["cell_id"]},
        expected_binding_sha256=BINDING,
    )
    assert duplicate.status == "unavailable"
    assert any("duplicate JSON key" in error for error in duplicate.errors)

    bad = _record()
    bad["seed"] = 211
    _write_jsonl(path, [bad])
    mismatched = A.load_jsonl(
        path,
        expected_ids={record["cell_id"]},
        expected_binding_sha256=BINDING,
    )
    assert mismatched.status == "unavailable"
    assert any("cell_id does not hash" in error for error in mismatched.errors)


def test_same_scoring_mask_and_parent_mse_are_required(tmp_path):
    records = [_record(action=action) for action in A.ACTIONS]
    records[1]["scoring_mask_sha256"] = "d" * 64
    path = tmp_path / "evidence.jsonl"
    _write_jsonl(path, records)
    loaded = A.load_jsonl(
        path,
        expected_ids={record["cell_id"] for record in records},
        expected_binding_sha256=BINDING,
    )
    assert loaded.available
    result = A.analyze_evidence(loaded)
    assert result["status"] == "unavailable"
    assert any("scoring-mask mismatch" in error for error in result["errors"])


def test_expected_key_helper_rejects_duplicate_stable_keys():
    record = _record()
    key = {field: record[field] for field in A.STABLE_KEY_FIELDS}
    assert A.expected_cell_ids([key]) == {record["cell_id"]}
    with pytest.raises(ValueError, match="duplicates"):
        A.expected_cell_ids([key, key])


def test_bootstrap_family_samples_are_deterministic_and_shared(passing_analysis):
    _loaded, analysis = passing_analysis
    cells = analysis["matched_survival"]["cells"]
    affine_025 = cells["affine6"]["radius=0.25|checkpoint=0"]
    affine_075 = cells["affine6"]["radius=0.75|checkpoint=0"]
    assert affine_025["bootstrap_95_lower_db"] == affine_075[
        "bootstrap_95_lower_db"
    ]
    family_means = np.asarray(
        [
            affine_025["family_means"][family]["mean_psnr_delta_db"]
            for family in A.TARGET_FAMILIES
        ]
    )
    rng = np.random.Generator(np.random.PCG64(20_260_715))
    indices = rng.integers(0, 6, size=(100_000, 6), endpoint=False)
    expected_lower = np.percentile(np.mean(family_means[indices], axis=1), 2.5)
    assert affine_025["bootstrap_95_lower_db"] == pytest.approx(expected_lower)
