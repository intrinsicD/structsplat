from __future__ import annotations

import copy

import pytest

from benchmarks import ssp2e_actual_coder as actual


def _rows(*, candidate: int = 890, shuffled_modeled: int = 500) -> list[dict]:
    rows: list[dict] = []
    for bits in actual.FROZEN_TUPLES:
        for image in actual.FROZEN_IMAGES:
            row = {
                "schema": actual.ROW_SCHEMA,
                "image_id": image,
                "bit_tuple": list(bits),
                "sspl1_bytes": 1000,
                "ssp2e_bytes": candidate,
                "ssp2s_bytes": 1000,
                "ssp2l_bytes": 1010,
                "ssp2f_bytes": 1020,
                "primary_bytes": 1000,
                "primary_name": "sspl1",
                "ssp2e_modeled_bytes": 400,
                "ssp2s_modeled_bytes": shuffled_modeled,
                "ssp2e_decode_ns": 120,
                "primary_decode_ns": 100,
                "ssp2e_peak_rss_bytes": 1200,
                "primary_peak_rss_bytes": 1000,
                "integrity": {name: True for name in actual.REQUIRED_INTEGRITY_KEYS},
            }
            rows.append(actual.seal_record(row))
    return rows


def test_all_exact_gates_pass_and_authorize_only_separate_confirmation():
    analysis = actual.analyze_rows(_rows())
    actual.verify_record(analysis, "analysis_sha256")
    assert analysis["decision"] == "AUTHORIZE_SEPARATELY_BOUND_SSP2E_CONFIRMATION"
    assert analysis["confirmation_access_authorized_in_this_task"] is False
    assert all(result["all_gates_pass"] for result in analysis["tuple_results"])


def test_actual_rate_tie_at_point_nine_passes_but_bootstrap_is_strict():
    analysis = actual.analyze_rows(_rows(candidate=900))
    result = analysis["tuple_results"][0]["actual_rate"]
    assert result["gates"]["geometric_mean_lte_0_90_exact"] is True
    assert result["gates"]["bootstrap_upper_lt_0_95_exact"] is True


def test_one_tuple_resource_failure_kills_joint_candidate():
    rows = _rows()
    for index, row in enumerate(rows):
        if tuple(row["bit_tuple"]) == actual.FROZEN_TUPLES[1]:
            core = dict(row)
            core.pop("record_sha256")
            core["ssp2e_decode_ns"] = 201
            rows[index] = actual.seal_record(core)
    analysis = actual.analyze_rows(rows)
    assert analysis["decision"] == "ABANDON_FIXED_SSP2E_V1"
    assert analysis["tuple_results"][0]["all_gates_pass"] is True
    assert analysis["tuple_results"][1]["resources"]["pass"] is False


def test_factorized_minimum_and_tie_order_are_recomputed():
    rows = _rows()
    row = copy.deepcopy(rows[0])
    row.pop("record_sha256")
    row["ssp2l_bytes"] = 800
    row["primary_bytes"] = 1000
    with pytest.raises(actual.ActualCoderError, match="primary_bytes"):
        actual.analyze_rows([actual.seal_record(row), *rows[1:]])


def test_duplicate_missing_and_broken_seal_fail_closed():
    rows = _rows()
    with pytest.raises(actual.ActualCoderError, match="16 rows"):
        actual.analyze_rows(rows[:-1])
    rows[0]["ssp2e_bytes"] += 1
    with pytest.raises(actual.ActualCoderError, match="record_sha256 mismatch"):
        actual.analyze_rows(rows)


def test_integrity_dictionary_cannot_pass_with_vacuous_all_true_key():
    rows = _rows()
    core = dict(rows[0])
    core.pop("record_sha256")
    core["integrity"] = {"trust_me": True}
    rows[0] = actual.seal_record(core)
    with pytest.raises(actual.ActualCoderError, match="integrity keys"):
        actual.analyze_rows(rows)
