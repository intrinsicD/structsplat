"""COMP-009 SSP2E actual-coder experiment and exact decision logic.

The codec/model/container implementations live in separate modules so their synthetic proof suite
can be sealed before this runner imports any COMP-008 development stream.  This module currently
owns the result-row contract and exact aggregate gates; lifecycle commands are added only after the
pre-data core passes its proof tests.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from fractions import Fraction
from typing import Any

import numpy as np

from benchmarks import sgi_entropy_oracle as oracle


ROW_SCHEMA = "structsplat.comp009.ssp2e-actual-row.v1"
ANALYSIS_SCHEMA = "structsplat.comp009.ssp2e-actual-analysis.v1"
FROZEN_IMAGES = oracle.FROZEN_IMAGE_IDS
FROZEN_TUPLES = oracle.FROZEN_BIT_TUPLES
REQUIRED_INTEGRITY_KEYS = frozenset(
    {
        "all_eight_true_starts_encoded",
        "all_eight_shuffled_starts_encoded",
        "all_fits_fixed_points",
        "all_start_python_native_payloads_equal",
        "all_python_framed_blobs_with_python_vs_native_arithmetic_equal",
        "all_python_container_decodes_with_python_vs_native_arithmetic_equal",
        "all_symbols_exact",
        "all_boundary_arrays_exact",
        "all_arithmetic_certificates_hold",
        "complete_byte_accounting_exact",
        "input_sspl1_hash_exact",
        "boundary_hash_matches_comp008",
        "persisted_artifacts_reverified",
    }
)


class ActualCoderError(ValueError):
    """A frozen COMP-009 row or decision invariant was violated."""


def canonical_json(value: Any) -> bytes:
    """Return finite canonical JSON with one terminal newline."""
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode("utf-8")


def seal_record(record: Mapping[str, Any], field: str = "record_sha256") -> dict[str, Any]:
    """Seal a JSON record over all fields except its seal field."""
    result = dict(record)
    result.pop(field, None)
    result[field] = hashlib.sha256(canonical_json(result)).hexdigest()
    return result


def verify_record(record: Mapping[str, Any], field: str = "record_sha256") -> None:
    """Verify a record produced by :func:`seal_record`."""
    observed = record.get(field)
    if not isinstance(observed, str) or len(observed) != 64:
        raise ActualCoderError(f"missing or invalid {field}")
    core = dict(record)
    core.pop(field, None)
    expected = hashlib.sha256(canonical_json(core)).hexdigest()
    if observed != expected:
        raise ActualCoderError(f"{field} mismatch: {observed} != {expected}")


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ActualCoderError(f"{name} must be a positive integer")
    return value


def _ratio_gm(numerators: Sequence[int], denominators: Sequence[int]) -> float:
    return float(
        np.exp(
            np.mean(
                np.log(np.asarray(numerators, dtype=np.float64))
                - np.log(np.asarray(denominators, dtype=np.float64))
            )
        )
    )


def _bootstrap_gate(numerators: Sequence[int], denominators: Sequence[int]) -> dict[str, Any]:
    """Evaluate the frozen 97.5% upper-GM gate with exact threshold comparisons."""
    if len(numerators) != 8 or len(denominators) != 8:
        raise ActualCoderError("bootstrap requires exactly eight paired image ratios")
    matrix = oracle.bootstrap_index_matrix()
    below = 0
    equal = 0
    for indices in matrix:
        selected_num = math.prod(numerators[int(index)] for index in indices)
        selected_den = math.prod(denominators[int(index)] for index in indices)
        left = selected_num * 20**8
        right = selected_den * 19**8
        if left < right:
            below += 1
        elif left == right:
            equal += 1
    above = int(matrix.shape[0]) - below - equal
    log_ratio = np.log(np.asarray(numerators, dtype=np.float64)) - np.log(
        np.asarray(denominators, dtype=np.float64)
    )
    diagnostics = np.exp(np.mean(log_ratio[matrix], axis=1))
    upper = float(np.sort(diagnostics)[oracle.BOOTSTRAP_UPPER_INDEX])
    return {
        "matrix_sha256": oracle.BOOTSTRAP_MATRIX_SHA256,
        "replicates": int(matrix.shape[0]),
        "upper_index_zero_based": oracle.BOOTSTRAP_UPPER_INDEX,
        "strictly_below_0_95": below,
        "equal_0_95": equal,
        "strictly_above_0_95": above,
        "upper_strictly_below_0_95_exact": below > oracle.BOOTSTRAP_UPPER_INDEX,
        "upper_geometric_mean_ratio_diagnostic": upper,
    }


def _validate_row(row: Mapping[str, Any]) -> dict[str, Any]:
    verify_record(row)
    if row.get("schema") != ROW_SCHEMA:
        raise ActualCoderError("unexpected result-row schema")
    image = row.get("image_id")
    if image not in FROZEN_IMAGES:
        raise ActualCoderError(f"unexpected development image {image!r}")
    raw_tuple = row.get("bit_tuple")
    if not isinstance(raw_tuple, list) or tuple(raw_tuple) not in FROZEN_TUPLES:
        raise ActualCoderError("unexpected bit tuple")
    for name in (
        "sspl1_bytes",
        "ssp2e_bytes",
        "ssp2s_bytes",
        "ssp2l_bytes",
        "ssp2f_bytes",
        "ssp2e_modeled_bytes",
        "ssp2s_modeled_bytes",
        "ssp2e_decode_ns",
        "primary_decode_ns",
        "ssp2e_peak_rss_bytes",
        "primary_peak_rss_bytes",
    ):
        _positive_int(row.get(name), name)
    integrity = row.get("integrity")
    if (
        not isinstance(integrity, dict)
        or set(integrity) != REQUIRED_INTEGRITY_KEYS
        or any(value is not True for value in integrity.values())
    ):
        raise ActualCoderError("row integrity keys/values differ from the frozen exact contract")
    primary = min(int(row["sspl1_bytes"]), int(row["ssp2l_bytes"]), int(row["ssp2f_bytes"]))
    if row.get("primary_bytes") != primary:
        raise ActualCoderError("primary_bytes is not min(SSPL1,SSP2L,SSP2F)")
    expected_primary_name = min(
        ("sspl1", "ssp2l", "ssp2f"),
        key=lambda name: (
            int(row[f"{name}_bytes"]),
            ("sspl1", "ssp2l", "ssp2f").index(name),
        ),
    )
    if row.get("primary_name") != expected_primary_name:
        raise ActualCoderError("primary_name does not follow frozen byte/tie selection")
    return dict(row)


def _tuple_result(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    ordered = sorted(rows, key=lambda row: FROZEN_IMAGES.index(str(row["image_id"])))
    if [row["image_id"] for row in ordered] != list(FROZEN_IMAGES):
        raise ActualCoderError("tuple rows do not contain each frozen image exactly once")

    candidate = [int(row["ssp2e_bytes"]) for row in ordered]
    primary = [int(row["primary_bytes"]) for row in ordered]
    modeled = [int(row["ssp2e_modeled_bytes"]) for row in ordered]
    shuffled = [int(row["ssp2s_modeled_bytes"]) for row in ordered]

    rate_gates = {
        "geometric_mean_lte_0_90_exact": math.prod(candidate) * 10**8 <= math.prod(primary) * 9**8,
        "strict_image_wins_gte_7": sum(a < b for a, b in zip(candidate, primary, strict=True)) >= 7,
        "worst_ratio_lte_1_02_exact": all(
            50 * a <= 51 * b for a, b in zip(candidate, primary, strict=True)
        ),
    }
    bootstrap = _bootstrap_gate(candidate, primary)
    rate_gates["bootstrap_upper_lt_0_95_exact"] = bool(bootstrap["upper_strictly_below_0_95_exact"])

    attribution_gates = {
        "aggregate_modeled_bytes_lte_0_90_exact": 10 * sum(modeled) <= 9 * sum(shuffled),
        "strict_modeled_wins_gte_6": sum(a < b for a, b in zip(modeled, shuffled, strict=True))
        >= 6,
    }

    time_num = [int(row["ssp2e_decode_ns"]) for row in ordered]
    time_den = [int(row["primary_decode_ns"]) for row in ordered]
    sorted_indices = sorted(
        range(8), key=lambda index: (Fraction(time_num[index], time_den[index]), index)
    )
    upper_median_index = sorted_indices[4]
    resource_gates = {
        "upper_median_time_ratio_lte_1_5_exact": 2 * time_num[upper_median_index]
        <= 3 * time_den[upper_median_index],
        "worst_time_ratio_lte_2_exact": all(
            a <= 2 * b for a, b in zip(time_num, time_den, strict=True)
        ),
        "all_peak_rss_ratios_lte_1_5_exact": all(
            2 * int(row["ssp2e_peak_rss_bytes"]) <= 3 * int(row["primary_peak_rss_bytes"])
            for row in ordered
        ),
    }

    rate_pass = all(rate_gates.values())
    attribution_pass = all(attribution_gates.values())
    resource_pass = all(resource_gates.values())
    return {
        "bit_tuple": list(ordered[0]["bit_tuple"]),
        "image_ids": list(FROZEN_IMAGES),
        "candidate_bytes": candidate,
        "primary_bytes": primary,
        "primary_names": [row["primary_name"] for row in ordered],
        "modeled_candidate_bytes": modeled,
        "modeled_shuffled_bytes": shuffled,
        "actual_rate": {
            "geometric_mean_ratio_diagnostic": _ratio_gm(candidate, primary),
            "strict_image_wins": sum(a < b for a, b in zip(candidate, primary, strict=True)),
            "worst_ratio_diagnostic": max(a / b for a, b in zip(candidate, primary, strict=True)),
            "gates": rate_gates,
            "bootstrap": bootstrap,
            "pass": rate_pass,
        },
        "spatial_attribution": {
            "aggregate_ratio_diagnostic": sum(modeled) / sum(shuffled),
            "strict_image_wins": sum(a < b for a, b in zip(modeled, shuffled, strict=True)),
            "gates": attribution_gates,
            "pass": attribution_pass,
        },
        "resources": {
            "time_ratios": [a / b for a, b in zip(time_num, time_den, strict=True)],
            "upper_median_time_ratio_diagnostic": time_num[upper_median_index]
            / time_den[upper_median_index],
            "worst_time_ratio_diagnostic": max(
                a / b for a, b in zip(time_num, time_den, strict=True)
            ),
            "gates": resource_gates,
            "pass": resource_pass,
        },
        "all_gates_pass": rate_pass and attribution_pass and resource_pass,
    }


def analyze_rows(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Validate 16 sealed rows and apply every frozen COMP-009 development gate."""
    validated = [_validate_row(row) for row in rows]
    if len(validated) != 16:
        raise ActualCoderError(f"expected 16 rows, found {len(validated)}")
    identities = [(row["image_id"], tuple(row["bit_tuple"])) for row in validated]
    if len(set(identities)) != 16:
        raise ActualCoderError("duplicate development image/tuple row")
    expected = {(image, bits) for image in FROZEN_IMAGES for bits in FROZEN_TUPLES}
    if set(identities) != expected:
        raise ActualCoderError("development row grid is incomplete or unexpected")

    tuple_results = []
    for bits in FROZEN_TUPLES:
        group = [row for row in validated if tuple(row["bit_tuple"]) == bits]
        tuple_results.append(_tuple_result(group))
    all_pass = all(result["all_gates_pass"] for result in tuple_results)
    analysis = {
        "schema": ANALYSIS_SCHEMA,
        "row_count": len(validated),
        "tuple_results": tuple_results,
        "decision": (
            "AUTHORIZE_SEPARATELY_BOUND_SSP2E_CONFIRMATION"
            if all_pass
            else "ABANDON_FIXED_SSP2E_V1"
        ),
        "confirmation_access_authorized_in_this_task": False,
        "claim_scope": "actual-rate-spatial-attribution-and-resource-development-evidence-only",
        "non_claims": [
            "Decoded image quality and representation expressiveness are exactly unchanged.",
            "Ordinary field-fitting convergence and renderer-only speed are not tested.",
            "No comparison with conventional or learned image codecs is made.",
        ],
    }
    return seal_record(analysis, "analysis_sha256")


__all__ = [
    "ANALYSIS_SCHEMA",
    "ActualCoderError",
    "REQUIRED_INTEGRITY_KEYS",
    "ROW_SCHEMA",
    "analyze_rows",
    "canonical_json",
    "seal_record",
    "verify_record",
]
