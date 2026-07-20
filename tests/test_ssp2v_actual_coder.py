from __future__ import annotations

import copy
import math

import pytest

from benchmarks import ssp2v_actual_coder as A


_UNSET = object()


def _metrics(
    *,
    psnr: float = 30.0,
    ms_ssim: float = 0.9,
    lpips: float = 0.1,
    psnr_values: list[float] | None = None,
) -> list[dict[str, float]]:
    psnrs = [psnr, psnr, psnr] if psnr_values is None else psnr_values
    return [
        {"psnr": value, "ms_ssim": ms_ssim, "lpips": lpips} for value in psnrs
    ]


def _quality_metrics(state: str) -> list[dict[str, float]]:
    if state == "qualified":
        return _metrics()
    if state == "failed":
        return _metrics(psnr=29.94)
    if state == "ambiguous":
        return _metrics(psnr=29.95)
    if state == "median_ambiguous":
        return _metrics(psnr=29.98)
    if state == "spread_invalid":
        return _metrics(psnr_values=[30.0, 30.002, 30.0])
    raise AssertionError(f"unknown test quality state {state}")


def _variant(
    label: str,
    complete_bytes: int,
    quality: str,
    *,
    exact: bool = True,
    fixed: bool = True,
) -> dict[str, object]:
    family, stages, base_k = A.VARIANT_SPECS[label]
    return {
        "label": label,
        "family_rank": family,
        "matched_stage_count": stages,
        "base_k": base_k,
        "complete_bytes": complete_bytes,
        "exact": exact,
        "all_lloyd_starts_fixed": fixed,
        "metrics": _quality_metrics(quality),
    }


def _variants(
    *,
    selected_quality: str = "qualified",
    all_failed: bool = False,
    ambiguous_before: bool = False,
    ambiguous_after: bool = False,
) -> list[dict[str, object]]:
    result = []
    for index, label in enumerate(A.VARIANT_ORDER):
        quality = "failed"
        if not all_failed and index == 0:
            quality = selected_quality
        if ambiguous_before and index == 0:
            quality = "ambiguous"
        if ambiguous_before and index == 1:
            quality = "qualified"
        if ambiguous_after and index == 1:
            quality = "ambiguous"
        result.append(_variant(label, 900 + 10 * index, quality))
    return result


def _formats() -> dict[str, dict[str, object]]:
    sizes = {
        "sspl1": 1000,
        "ssp2z": 1100,
        "ssp2e": 1200,
        "ssp2s": 1300,
        "ssp2l": 970,
        "ssp2f": 1250,
    }
    return {
        name: {"complete_bytes": sizes[name], "exact": True} for name in A.PRIMARY_ORDER
    }


def _primary_resource(
    *,
    candidate_label: str = A.VARIANT_ORDER[0],
    q_name: str = "ssp2l",
    candidate_ns: int = 900,
    q_ns: int = 1000,
    candidate_rss: int = 100_000,
    q_rss: int = 100_000,
    fresh: bool = True,
    schedule: bool = True,
) -> dict[str, object]:
    return {
        "candidate_label": candidate_label,
        "q_name": q_name,
        "candidate_median_ns": candidate_ns,
        "q_median_ns": q_ns,
        "candidate_peak_rss_bytes": candidate_rss,
        "q_peak_rss_bytes": q_rss,
        "fresh_comp011": fresh,
        "schedule_exact": schedule,
    }


def _secondary_resource(
    *,
    l_ns: int = 1000,
    b0_ns: int = 1200,
    fresh: bool = True,
    schedule: bool = True,
) -> dict[str, object]:
    return {
        "ssp2l_median_ns": l_ns,
        "sspl1_median_ns": b0_ns,
        "fresh_comp011": fresh,
        "schedule_exact": schedule,
    }


def _complete_row(
    image_id: str,
    bits: tuple[int, int, int, int],
    *,
    variants: list[dict[str, object]] | None = None,
    primary_resource: dict[str, object] | None | object = _UNSET,
    secondary_resource: dict[str, object] | None | object = _UNSET,
) -> dict[str, object]:
    record = {
        "schema": A.ROW_SCHEMA,
        "image_id": image_id,
        "bit_tuple": list(bits),
        "validity": "valid",
        "invalid_reason": None,
        "method_state": "complete",
        "lloyd_all_fixed": True,
        "formats": _formats(),
        "primary_metrics": _metrics(),
        "variants": _variants() if variants is None else variants,
        "primary_resource": _primary_resource()
        if primary_resource is _UNSET
        else primary_resource,
        "secondary_resource": _secondary_resource()
        if secondary_resource is _UNSET
        else secondary_resource,
    }
    return A.seal_record(record)


def _grid(
    *,
    variant_factory=None,
    primary_resource_factory=None,
    secondary_resource_factory=None,
) -> list[dict[str, object]]:
    rows = []
    for image_index, image in enumerate(A.FROZEN_IMAGES):
        for tuple_index, bits in enumerate(A.FROZEN_TUPLES):
            variants = (
                _variants()
                if variant_factory is None
                else variant_factory(image_index, tuple_index)
            )
            primary = (
                _primary_resource()
                if primary_resource_factory is None
                else primary_resource_factory(image_index, tuple_index)
            )
            secondary = (
                _secondary_resource()
                if secondary_resource_factory is None
                else secondary_resource_factory(image_index, tuple_index)
            )
            rows.append(
                _complete_row(
                    image,
                    bits,
                    variants=variants,
                    primary_resource=primary,
                    secondary_resource=secondary,
                )
            )
    return rows


def _early_failure_row() -> dict[str, object]:
    return A.seal_record(
        {
            "schema": A.ROW_SCHEMA,
            "image_id": A.FROZEN_IMAGES[0],
            "bit_tuple": list(A.FROZEN_TUPLES[0]),
            "validity": "valid",
            "invalid_reason": None,
            "method_state": "lloyd_failure",
            "lloyd_all_fixed": False,
            "formats": None,
            "primary_metrics": None,
            "variants": None,
            "primary_resource": None,
            "secondary_resource": None,
        }
    )


def _invalid_row(reason: str = "synthetic_protocol_invalidity") -> dict[str, object]:
    return A.seal_record(
        {
            "schema": A.ROW_SCHEMA,
            "image_id": A.FROZEN_IMAGES[0],
            "bit_tuple": list(A.FROZEN_TUPLES[0]),
            "validity": "invalid",
            "invalid_reason": reason,
            "method_state": "invalid",
            "lloyd_all_fixed": None,
            "formats": None,
            "primary_metrics": None,
            "variants": None,
            "primary_resource": None,
            "secondary_resource": None,
        }
    )


def test_canonical_seal_roundtrip_and_hostile_mutation() -> None:
    row = _early_failure_row()
    A.verify_record(row)
    assert A.canonical_json({"b": 2, "a": 1}) == b'{"a":1,"b":2}\n'

    changed = copy.deepcopy(row)
    changed["image_id"] = A.FROZEN_IMAGES[1]
    with pytest.raises(A.ActualCoderError, match="record_sha256 mismatch"):
        A.verify_record(changed)

    malformed = dict(row)
    malformed["record_sha256"] = "A" * 64
    with pytest.raises(A.ActualCoderError, match="invalid record_sha256"):
        A.verify_record(malformed)

    with pytest.raises(A.ActualCoderError, match="finite canonical JSON"):
        A.canonical_json({"value": float("nan")})


def test_primary_selection_uses_all_six_and_frozen_tie_order() -> None:
    formats = {
        name: {"complete_bytes": 100, "exact": True} for name in A.PRIMARY_ORDER
    }
    selected = A.select_primary(formats)
    assert selected["selected_name"] == "sspl1"

    formats["sspl1"]["complete_bytes"] = 101
    assert A.select_primary(formats)["selected_name"] == "ssp2z"

    formats["ssp2e"]["exact"] = False
    invalid = A.select_primary(formats)
    assert invalid["state"] == "invalid"
    assert invalid["invalid_formats"] == ["ssp2e"]

    del formats["ssp2f"]
    with pytest.raises(A.ActualCoderError, match="exactly the six"):
        A.select_primary(formats)


def test_banded_metric_boundaries_are_asymmetric_and_exact() -> None:
    threshold = 0.05
    band = 0.002
    pass_edge = threshold - band
    fail_edge = threshold + band
    assert A.classify_banded_metric(pass_edge, threshold, band) == "definitely_passes"
    assert A.classify_banded_metric(fail_edge, threshold, band) == "ambiguous"
    assert (
        A.classify_banded_metric(math.nextafter(fail_edge, math.inf), threshold, band)
        == "definitely_fails"
    )


def test_quality_envelope_classification_and_spread_invalidity() -> None:
    primary = [
        {"psnr": 30.0, "ms_ssim": 0.90, "lpips": 0.10},
        {"psnr": 30.0004, "ms_ssim": 0.900004, "lpips": 0.100004},
        {"psnr": 29.9996, "ms_ssim": 0.899996, "lpips": 0.099996},
    ]
    candidate = [
        {"psnr": 29.99, "ms_ssim": 0.90, "lpips": 0.10},
        {"psnr": 29.9904, "ms_ssim": 0.900004, "lpips": 0.100004},
        {"psnr": 29.9896, "ms_ssim": 0.899996, "lpips": 0.099996},
    ]
    result = A.evaluate_quality(primary, candidate)
    assert result["state"] == "definitely_qualified"
    assert result["conservative"]["psnr_loss"] == pytest.approx(0.0108)
    assert set(result["metric_classes"].values()) == {"definitely_passes"}

    invalid = A.evaluate_quality(_metrics(), _quality_metrics("spread_invalid"))
    assert invalid["state"] == "invalid"
    assert invalid["invalid_reason"] == "atomic_repeat_spread_cap_exceeded"


def test_variant_selection_obeys_complete_bytes_then_frozen_fields() -> None:
    variants = [_variant(label, 1000, "qualified") for label in reversed(A.VARIANT_ORDER)]
    selected = A.select_variant(_metrics(), variants)
    assert selected["state"] == "selected"
    assert selected["selected_label"] == "flat_s2_k64_l192"
    assert selected["ordered_labels"] == list(A.VARIANT_ORDER)
    assert selected["complete_blob_tiebreak_reachable"] is False

    smaller = copy.deepcopy(variants)
    for record in smaller:
        if record["label"] == "rvq_s3_k128":
            record["complete_bytes"] = 999
    selected = A.select_variant(_metrics(), smaller)
    assert selected["selected_label"] == "rvq_s3_k128"


def test_variant_ambiguity_before_selection_invalidates_but_later_does_not() -> None:
    before = A.select_variant(_metrics(), _variants(ambiguous_before=True))
    assert before["state"] == "invalid"
    assert before["invalid_reason"] == "ambiguous_variant_precedes_first_definite_qualifier"

    after = A.select_variant(_metrics(), _variants(ambiguous_after=True))
    assert after["state"] == "selected"
    assert after["selected_label"] == A.VARIANT_ORDER[0]
    assert after["quality"][A.VARIANT_ORDER[1]]["state"] == "ambiguous"

    no_candidate = A.select_variant(_metrics(), _variants(all_failed=True))
    assert no_candidate["state"] == "method_failure"
    assert no_candidate["selected_label"] is None


def test_variant_exactness_spread_and_lloyd_fail_closed() -> None:
    inexact = _variants()
    inexact[3]["exact"] = False
    assert A.select_variant(_metrics(), inexact)["state"] == "invalid"

    spread = _variants()
    spread[4]["metrics"] = _quality_metrics("spread_invalid")
    assert A.select_variant(_metrics(), spread)["state"] == "invalid"

    nonconverged = _variants()
    nonconverged[7]["all_lloyd_starts_fixed"] = False
    result = A.select_variant(_metrics(), nonconverged)
    assert result["state"] == "method_failure"
    assert result["invalid_reason"] == "lloyd_nonconvergence_in_complete_candidate_record"


def test_variant_descriptor_and_menu_are_strict() -> None:
    variants = _variants()
    variants[0]["family_rank"] = False
    with pytest.raises(A.ActualCoderError, match="nonnegative integer"):
        A.select_variant(_metrics(), variants)

    variants = _variants()
    variants[0]["family_rank"] = "0"
    with pytest.raises(A.ActualCoderError, match="nonnegative integer"):
        A.select_variant(_metrics(), variants)

    variants = _variants()
    variants[0]["base_k"] = 128
    with pytest.raises(A.ActualCoderError, match="descriptor fields disagree"):
        A.select_variant(_metrics(), variants)

    variants = _variants()
    variants[0]["label"] = variants[1]["label"]
    with pytest.raises(A.ActualCoderError, match="descriptor fields disagree|every frozen label"):
        A.select_variant(_metrics(), variants)

    variants = _variants()[:-1]
    with pytest.raises(A.ActualCoderError, match="exactly eight"):
        A.select_variant(_metrics(), variants)


def test_exact_rational_order_uses_index_for_equal_ratios() -> None:
    assert A.rational_order_indices([2, 1, 6, 1], [2, 1, 3, 2]) == [3, 0, 1, 2]
    with pytest.raises(A.ActualCoderError, match="equal nonempty"):
        A.rational_order_indices([1], [])


def test_bootstrap_exact_order_statistic_threshold_and_tie_index() -> None:
    below = A.bootstrap_order_statistic(
        [97] * 8,
        [100] * 8,
        threshold_numerator_power=49**8,
        threshold_denominator_power=50**8,
    )
    assert below["upper_replicate_index"] == 97_499
    assert below["upper_unsimplified_numerator"] == 97**8
    assert below["upper_unsimplified_denominator"] == 100**8
    assert below["upper_strictly_below_threshold_exact"] is True
    assert below["strictly_below_threshold"] == 100_000

    equal = A.bootstrap_order_statistic(
        [98] * 8,
        [100] * 8,
        threshold_numerator_power=49**8,
        threshold_denominator_power=50**8,
    )
    assert equal["upper_strictly_below_threshold_exact"] is False
    assert equal["strictly_below_threshold"] == 0
    assert equal["upper_replicate_index"] == 97_499


def test_conservative_median_band_edges_and_image_ties() -> None:
    passing = A.conservative_median_psnr([0.018] * 8)
    assert passing["state"] == "definitely_passes"
    assert passing["sorted_image_indices"] == list(range(8))

    ambiguous = A.conservative_median_psnr([0.022] * 8)
    assert ambiguous["state"] == "ambiguous"
    failing = A.conservative_median_psnr([math.nextafter(0.022, math.inf)] * 8)
    assert failing["state"] == "definitely_fails"


def test_full_passing_grid_authorizes_both_separately_bound_confirmations() -> None:
    rows = _grid()
    analysis = A.analyze_rows(rows)
    assert A.analyze_rows(list(reversed(rows))) == analysis
    A.verify_record(analysis, "analysis_sha256")
    assert analysis["status"] == "valid_complete"
    assert analysis["decision"] == A.MAIN_DECISION_AUTHORIZE
    assert analysis["ssp2l_decision"] == A.SECONDARY_DECISION_AUTHORIZE
    assert analysis["confirmation_access_authorized_in_this_task"] is False
    assert len(analysis["tuple_results"]) == 2
    assert len(analysis["secondary_tuple_results"]) == 2
    for result in analysis["tuple_results"]:
        assert result["all_gates_pass"] is True
        assert set(result["actual_rate"]["gates"].values()) == {True}
        assert set(result["operational_bundle"]["gates"].values()) == {True}
        assert result["resources"]["pass"] is True
        assert result["actual_rate"]["bootstrap"]["upper_replicate_index"] == 97_499
    for result in analysis["secondary_tuple_results"]:
        assert result["pass"] is True
        assert set(result["gates"].values()) == {True}


def test_valid_primary_resource_gate_miss_abandons_without_invalidating_secondary() -> None:
    def resources(image_index: int, tuple_index: int):
        del tuple_index
        return _primary_resource(candidate_ns=2500 if image_index == 7 else 900)

    analysis = A.analyze_rows(_grid(primary_resource_factory=resources))
    assert analysis["status"] == "valid_method_failure"
    assert analysis["decision"] == A.MAIN_DECISION_ABANDON
    assert analysis["ssp2l_decision"] == A.SECONDARY_DECISION_AUTHORIZE
    assert any(not result["resources"]["pass"] for result in analysis["tuple_results"])


def test_no_quality_qualified_candidate_is_valid_failure_and_skips_primary_resources() -> None:
    rows = _grid(
        variant_factory=lambda _i, _t: _variants(all_failed=True),
        primary_resource_factory=lambda _i, _t: None,
    )
    analysis = A.analyze_rows(rows)
    assert analysis["status"] == "valid_method_failure"
    assert analysis["decision"] == A.MAIN_DECISION_ABANDON
    assert analysis["ssp2l_decision"] == A.SECONDARY_DECISION_AUTHORIZE
    assert analysis["tuple_results"] == []


def test_ambiguous_variant_and_median_produce_no_decision() -> None:
    ambiguous_rows = _grid(
        variant_factory=lambda _i, _t: _variants(ambiguous_before=True),
        primary_resource_factory=lambda _i, _t: None,
    )
    analysis = A.analyze_rows(ambiguous_rows)
    assert analysis["status"] == "invalid_no_decision"
    assert analysis["decision"] == A.INVALID_DECISION
    assert analysis["ssp2l_decision"] == A.INVALID_DECISION

    median_rows = _grid(
        variant_factory=lambda _i, _t: _variants(selected_quality="median_ambiguous")
    )
    analysis = A.analyze_rows(median_rows)
    assert analysis["status"] == "invalid_no_decision"
    assert analysis["invalid_reason"] == "median_psnr_loss_in_exclusion_band"


def test_schedule_identity_errors_are_invalid_not_negative_gate_evidence() -> None:
    wrong_identity = _grid(
        primary_resource_factory=lambda _i, _t: _primary_resource(q_name="sspl1")
    )
    analysis = A.analyze_rows(wrong_identity)
    assert analysis["status"] == "invalid_no_decision"
    assert analysis["invalid_reason"] == "primary_resource_identity_or_schedule_mismatch"

    stale_secondary = _grid(
        secondary_resource_factory=lambda _i, _t: _secondary_resource(fresh=False)
    )
    analysis = A.analyze_rows(stale_secondary)
    assert analysis["status"] == "invalid_no_decision"
    assert analysis["invalid_reason"] == "secondary_resource_schedule_mismatch"


def test_secondary_gate_failure_never_changes_a_passing_main_decision() -> None:
    def secondary(image_index: int, tuple_index: int):
        del image_index, tuple_index
        return _secondary_resource(l_ns=1501, b0_ns=1200)

    analysis = A.analyze_rows(_grid(secondary_resource_factory=secondary))
    assert analysis["decision"] == A.MAIN_DECISION_AUTHORIZE
    assert analysis["ssp2l_decision"] == A.SECONDARY_DECISION_NONE
    assert all(result["pass"] is False for result in analysis["secondary_tuple_results"])


def test_early_lloyd_failure_closes_main_and_makes_secondary_inapplicable() -> None:
    analysis = A.analyze_rows([_early_failure_row()])
    assert analysis["status"] == "valid_method_failure"
    assert analysis["decision"] == A.MAIN_DECISION_ABANDON
    assert analysis["ssp2l_decision"] == A.SECONDARY_DECISION_INAPPLICABLE
    assert analysis["early_failure"]["kind"] == "required_lloyd_start_nonconvergence"

    invalid = A.analyze_rows([_early_failure_row(), _complete_row(
        A.FROZEN_IMAGES[1], A.FROZEN_TUPLES[0]
    )])
    assert invalid["status"] == "invalid_no_decision"


def test_explicit_invalid_row_produces_fail_closed_no_decision() -> None:
    analysis = A.analyze_rows([_invalid_row("renderer_binary_drift")])
    assert analysis["status"] == "invalid_no_decision"
    assert analysis["invalid_reason"] == "invalid_input_row:renderer_binary_drift"
    assert analysis["decision"] == A.INVALID_DECISION


def test_full_grid_structure_seals_and_exact_keys_are_hostile() -> None:
    rows = _grid()
    with pytest.raises(A.ActualCoderError, match="duplicate"):
        A.analyze_rows([*rows[:-1], rows[0]])
    with pytest.raises(A.ActualCoderError, match="sixteen-cell grid"):
        A.analyze_rows(rows[:-1])

    extra = copy.deepcopy(rows[0])
    extra["unexpected"] = True
    extra = A.seal_record(extra)
    with pytest.raises(A.ActualCoderError, match="row key mismatch"):
        A.analyze_rows([extra])

    contradictory = copy.deepcopy(_early_failure_row())
    contradictory["primary_resource"] = _primary_resource()
    contradictory = A.seal_record(contradictory)
    with pytest.raises(A.ActualCoderError, match="forbidden later-stage"):
        A.analyze_rows([contradictory])


def test_primary_resources_are_forbidden_when_any_cell_has_no_selection() -> None:
    rows = _grid()
    first = copy.deepcopy(rows[0])
    core = dict(first)
    core.pop("record_sha256")
    core["variants"] = _variants(all_failed=True)
    core["primary_resource"] = None
    rows[0] = A.seal_record(core)
    analysis = A.analyze_rows(rows)
    assert analysis["status"] == "invalid_no_decision"
    assert (
        analysis["invalid_reason"]
        == "primary_resource_schedule_ran_without_all_sixteen_selections"
    )


def test_row_structural_types_reject_bool_as_integer() -> None:
    row = _complete_row(A.FROZEN_IMAGES[0], A.FROZEN_TUPLES[0])
    core = dict(row)
    core.pop("record_sha256")
    core["formats"]["sspl1"]["complete_bytes"] = True
    with pytest.raises(A.ActualCoderError, match="positive integer"):
        A.analyze_rows([A.seal_record(core)])
