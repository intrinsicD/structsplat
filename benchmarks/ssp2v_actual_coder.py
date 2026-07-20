"""Exact, payload-free COMP-011 analysis and decision logic.

This module deliberately has no filesystem API.  It consumes small sealed records produced by
the COMP-011 lifecycle, recomputes every decision-relevant selection and aggregate predicate, and
returns one sealed analysis object.  Stream bytes, target pixels, renderer tensors, and timing
worker artifacts remain under the lifecycle runner's authority.

The row-level ``exact`` flags are validity attestations: the lifecycle must set them only after
the detailed reproduction, canonical cold-decode, symbol, and boundary checks required by the
frozen task.  They never stand in for rate, quality, or resource gates, all of which are recomputed
here from their scalar inputs.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from functools import cmp_to_key, lru_cache
from typing import Any

from benchmarks import sgi_entropy_oracle as oracle


PROTOCOL = "COMP-011-complete-stream-rgb-vq-v2"
TASK_SHA256 = "e73b69321de353c572292e17688019e6c1a62b55ebe1d1e3e66bac4120bd9ea5"
ROW_SCHEMA = "structsplat.comp011.ssp2v-actual-row.v2"
ANALYSIS_SCHEMA = "structsplat.comp011.ssp2v-actual-analysis.v2"

FROZEN_IMAGES = oracle.FROZEN_IMAGE_IDS
FROZEN_TUPLES = oracle.FROZEN_BIT_TUPLES
PRIMARY_ORDER = ("sspl1", "ssp2z", "ssp2e", "ssp2s", "ssp2l", "ssp2f")
VARIANT_ORDER = (
    "flat_s2_k64_l192",
    "flat_s2_k128_l384",
    "flat_s3_k64_l320",
    "flat_s3_k128_l640",
    "rvq_s2_k64",
    "rvq_s2_k128",
    "rvq_s3_k64",
    "rvq_s3_k128",
)
VARIANT_SPECS: dict[str, tuple[int, int, int]] = {
    "flat_s2_k64_l192": (0, 2, 64),
    "flat_s2_k128_l384": (0, 2, 128),
    "flat_s3_k64_l320": (0, 3, 64),
    "flat_s3_k128_l640": (0, 3, 128),
    "rvq_s2_k64": (1, 2, 64),
    "rvq_s2_k128": (1, 2, 128),
    "rvq_s3_k64": (1, 3, 64),
    "rvq_s3_k128": (1, 3, 128),
}

QUALITY_THRESHOLDS = {
    "psnr_loss": 0.05,
    "ms_ssim_loss": 0.0005,
    "lpips_increase": 0.001,
}
QUALITY_BANDS = {
    "psnr_loss": 0.002,
    "ms_ssim_loss": 0.00002,
    "lpips_increase": 0.00002,
}
SPREAD_CAPS = {
    "psnr": 0.001,
    "ms_ssim": 0.00001,
    "lpips": 0.00001,
}
MEDIAN_PSNR_PASS_MAX = 0.018
MEDIAN_PSNR_FAIL_ABOVE = 0.022

MAIN_DECISION_AUTHORIZE = "AUTHORIZE_SEPARATELY_BOUND_SSP2V_CONFIRMATION"
MAIN_DECISION_ABANDON = "ABANDON_FIXED_SSP2V_V1_UNDER_COMP011_V2"
INVALID_DECISION = "INVALID_NO_DECISION"
SECONDARY_DECISION_AUTHORIZE = "AUTHORIZE_SEPARATELY_BOUND_SSP2L_CONFIRMATION"
SECONDARY_DECISION_NONE = "NO_SSP2L_CONFIRMATION_AUTHORIZATION"
SECONDARY_DECISION_INAPPLICABLE = "INAPPLICABLE_EARLY_METHOD_FAILURE"

_ROW_KEYS = frozenset(
    {
        "schema",
        "image_id",
        "bit_tuple",
        "validity",
        "invalid_reason",
        "method_state",
        "lloyd_all_fixed",
        "formats",
        "primary_metrics",
        "variants",
        "primary_resource",
        "secondary_resource",
        "record_sha256",
    }
)
_FORMAT_KEYS = frozenset({"complete_bytes", "exact"})
_METRIC_KEYS = frozenset({"psnr", "ms_ssim", "lpips"})
_VARIANT_KEYS = frozenset(
    {
        "label",
        "family_rank",
        "matched_stage_count",
        "base_k",
        "complete_bytes",
        "exact",
        "all_lloyd_starts_fixed",
        "metrics",
    }
)
_PRIMARY_RESOURCE_KEYS = frozenset(
    {
        "candidate_label",
        "q_name",
        "candidate_median_ns",
        "q_median_ns",
        "candidate_peak_rss_bytes",
        "q_peak_rss_bytes",
        "fresh_comp011",
        "schedule_exact",
    }
)
_SECONDARY_RESOURCE_KEYS = frozenset(
    {
        "ssp2l_median_ns",
        "sspl1_median_ns",
        "fresh_comp011",
        "schedule_exact",
    }
)


class ActualCoderError(ValueError):
    """A sealed analysis input is structurally malformed or internally contradictory."""


def canonical_json(value: Any) -> bytes:
    """Serialize one finite canonical JSON value with a terminal newline."""

    try:
        return (
            json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ActualCoderError(f"value is not finite canonical JSON: {exc}") from exc


def seal_record(record: Mapping[str, Any], field: str = "record_sha256") -> dict[str, Any]:
    """Return a copy sealed over all fields except ``field``."""

    result = dict(record)
    result.pop(field, None)
    result[field] = hashlib.sha256(canonical_json(result)).hexdigest()
    return result


def verify_record(record: Mapping[str, Any], field: str = "record_sha256") -> None:
    """Verify a canonical record seal without accepting upper-case or shortened digests."""

    observed = record.get(field)
    if (
        not isinstance(observed, str)
        or len(observed) != 64
        or any(character not in "0123456789abcdef" for character in observed)
    ):
        raise ActualCoderError(f"missing or invalid {field}")
    core = dict(record)
    core.pop(field, None)
    expected = hashlib.sha256(canonical_json(core)).hexdigest()
    if observed != expected:
        raise ActualCoderError(f"{field} mismatch: {observed} != {expected}")


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ActualCoderError(f"{name} must be a mapping")
    return value


def _exact_keys(value: Mapping[str, Any], expected: frozenset[str], name: str) -> None:
    observed = frozenset(value)
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise ActualCoderError(f"{name} key mismatch; missing={missing}, extra={extra}")


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ActualCoderError(f"{name} must be a positive integer")
    return value


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ActualCoderError(f"{name} must be a nonnegative integer")
    return value


def _finite_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ActualCoderError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ActualCoderError(f"{name} must be finite")
    return result


def _required_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ActualCoderError(f"{name} must be exactly boolean")
    return value


def _ratio_diagnostic(numerators: Sequence[int], denominators: Sequence[int]) -> float:
    return math.exp(
        sum(math.log(numerator) - math.log(denominator) for numerator, denominator in zip(
            numerators, denominators, strict=True
        ))
        / len(numerators)
    )


def classify_banded_metric(value: float, threshold: float, band: float) -> str:
    """Apply the frozen pass/fail/ambiguity inequalities exactly as written."""

    observed = _finite_float(value, "banded metric")
    frozen_threshold = _finite_float(threshold, "metric threshold")
    frozen_band = _finite_float(band, "metric exclusion band")
    if frozen_band < 0.0:
        raise ActualCoderError("metric exclusion band must be nonnegative")
    if observed <= frozen_threshold - frozen_band:
        return "definitely_passes"
    if observed > frozen_threshold + frozen_band:
        return "definitely_fails"
    return "ambiguous"


def _metric_repetitions(value: Any, name: str) -> list[dict[str, float]]:
    if not isinstance(value, list) or len(value) != 3:
        raise ActualCoderError(f"{name} must contain exactly three repetitions")
    result: list[dict[str, float]] = []
    for index, raw in enumerate(value):
        record = _mapping(raw, f"{name}[{index}]")
        _exact_keys(record, _METRIC_KEYS, f"{name}[{index}]")
        result.append(
            {
                metric: _finite_float(record[metric], f"{name}[{index}].{metric}")
                for metric in ("psnr", "ms_ssim", "lpips")
            }
        )
    return result


def evaluate_quality(primary_metrics: Any, candidate_metrics: Any) -> dict[str, Any]:
    """Recompute spread caps, conservative envelopes, and quality classification."""

    baseline = _metric_repetitions(primary_metrics, "primary_metrics")
    candidate = _metric_repetitions(candidate_metrics, "candidate_metrics")
    baseline_values = {
        metric: [record[metric] for record in baseline]
        for metric in ("psnr", "ms_ssim", "lpips")
    }
    candidate_values = {
        metric: [record[metric] for record in candidate]
        for metric in ("psnr", "ms_ssim", "lpips")
    }
    spreads = {
        "primary": {
            metric: max(values) - min(values) for metric, values in baseline_values.items()
        },
        "candidate": {
            metric: max(values) - min(values) for metric, values in candidate_values.items()
        },
    }
    spread_gates = {
        side: {
            metric: value <= SPREAD_CAPS[metric] for metric, value in side_values.items()
        }
        for side, side_values in spreads.items()
    }
    if not all(gate for side in spread_gates.values() for gate in side.values()):
        return {
            "state": "invalid",
            "invalid_reason": "atomic_repeat_spread_cap_exceeded",
            "spreads": spreads,
            "spread_gates": spread_gates,
            "conservative": None,
            "metric_classes": None,
        }

    conservative = {
        "psnr_loss": max(baseline_values["psnr"]) - min(candidate_values["psnr"]),
        "ms_ssim_loss": max(baseline_values["ms_ssim"])
        - min(candidate_values["ms_ssim"]),
        "lpips_increase": max(candidate_values["lpips"]) - min(baseline_values["lpips"]),
    }
    classes = {
        name: classify_banded_metric(value, QUALITY_THRESHOLDS[name], QUALITY_BANDS[name])
        for name, value in conservative.items()
    }
    if any(value == "definitely_fails" for value in classes.values()):
        state = "definitely_failed"
    elif all(value == "definitely_passes" for value in classes.values()):
        state = "definitely_qualified"
    else:
        state = "ambiguous"
    return {
        "state": state,
        "invalid_reason": None,
        "spreads": spreads,
        "spread_gates": spread_gates,
        "conservative": conservative,
        "metric_classes": classes,
    }


def select_primary(formats: Any) -> dict[str, Any]:
    """Select ``Q`` by complete bytes and the frozen six-format tie order."""

    records = _mapping(formats, "formats")
    if set(records) != set(PRIMARY_ORDER):
        raise ActualCoderError("formats must contain exactly the six frozen primary formats")
    normalized: dict[str, dict[str, Any]] = {}
    invalid = []
    for rank, name in enumerate(PRIMARY_ORDER):
        record = _mapping(records[name], f"formats.{name}")
        _exact_keys(record, _FORMAT_KEYS, f"formats.{name}")
        exact = _required_bool(record["exact"], f"formats.{name}.exact")
        normalized[name] = {
            "complete_bytes": _positive_int(
                record["complete_bytes"], f"formats.{name}.complete_bytes"
            ),
            "exact": exact,
            "tie_rank": rank,
        }
        if not exact:
            invalid.append(name)
    if invalid:
        return {
            "state": "invalid",
            "invalid_reason": "inexact_or_noncanonical_primary_format",
            "invalid_formats": invalid,
            "selected_name": None,
            "selected_bytes": None,
            "formats": normalized,
        }
    selected_name = min(
        PRIMARY_ORDER,
        key=lambda name: (normalized[name]["complete_bytes"], normalized[name]["tie_rank"]),
    )
    return {
        "state": "selected",
        "invalid_reason": None,
        "invalid_formats": [],
        "selected_name": selected_name,
        "selected_bytes": normalized[selected_name]["complete_bytes"],
        "formats": normalized,
    }


def _normalize_variant(raw: Any, index: int) -> dict[str, Any]:
    record = _mapping(raw, f"variants[{index}]")
    _exact_keys(record, _VARIANT_KEYS, f"variants[{index}]")
    label = record["label"]
    if not isinstance(label, str) or label not in VARIANT_SPECS:
        raise ActualCoderError(f"variants[{index}].label is not a frozen variant")
    try:
        label_bytes = label.encode("ascii")
    except UnicodeEncodeError as exc:  # pragma: no cover - membership already limits this
        raise ActualCoderError("variant label is not ASCII") from exc
    expected_family, expected_stages, expected_k = VARIANT_SPECS[label]
    family = _nonnegative_int(record["family_rank"], f"variants[{index}].family_rank")
    stages = _positive_int(record["matched_stage_count"], f"variants[{index}].matched_stage_count")
    base_k = _positive_int(record["base_k"], f"variants[{index}].base_k")
    if (family, stages, base_k) != (expected_family, expected_stages, expected_k):
        raise ActualCoderError(f"variants[{index}] descriptor fields disagree with {label}")
    return {
        "label": label,
        "label_bytes": label_bytes,
        "family_rank": family,
        "matched_stage_count": stages,
        "base_k": base_k,
        "complete_bytes": _positive_int(
            record["complete_bytes"], f"variants[{index}].complete_bytes"
        ),
        "exact": _required_bool(record["exact"], f"variants[{index}].exact"),
        "all_lloyd_starts_fixed": _required_bool(
            record["all_lloyd_starts_fixed"],
            f"variants[{index}].all_lloyd_starts_fixed",
        ),
        "metrics": _metric_repetitions(record["metrics"], f"variants[{index}].metrics"),
    }


def select_variant(primary_metrics: Any, variants: Any) -> dict[str, Any]:
    """Apply the frozen byte order and fail/ambiguity/qualification traversal."""

    baseline = _metric_repetitions(primary_metrics, "primary_metrics")
    if not isinstance(variants, list) or len(variants) != len(VARIANT_ORDER):
        raise ActualCoderError("variants must contain exactly eight records")
    normalized = [_normalize_variant(raw, index) for index, raw in enumerate(variants)]
    labels = [record["label"] for record in normalized]
    if len(set(labels)) != len(VARIANT_ORDER) or set(labels) != set(VARIANT_ORDER):
        raise ActualCoderError("variants must contain every frozen label exactly once")

    # For the legal menu, (family_rank, matched_stage_count, base_k, label) is unique.  Therefore
    # the final complete-blob byte comparator in the frozen tuple is provably unreachable.  The
    # lifecycle still binds every complete blob; omitting it from this small scalar row cannot
    # change any legal ordering.
    descriptor_prefixes = {
        (
            record["family_rank"],
            record["matched_stage_count"],
            record["base_k"],
            record["label_bytes"],
        )
        for record in normalized
    }
    if len(descriptor_prefixes) != len(VARIANT_ORDER):  # pragma: no cover - frozen table proof
        raise RuntimeError("frozen legal variant descriptors do not make the blob tie unreachable")

    invalid_exact = [record["label"] for record in normalized if not record["exact"]]
    if invalid_exact:
        return {
            "state": "invalid",
            "invalid_reason": "inexact_or_noncanonical_ssp2v_variant",
            "invalid_variants": invalid_exact,
            "ordered_labels": [],
            "selected_label": None,
            "selected_bytes": None,
            "quality": {},
            "all_lloyd_starts_fixed": False,
            "complete_blob_tiebreak_reachable": False,
        }
    failed_fits = [
        record["label"] for record in normalized if not record["all_lloyd_starts_fixed"]
    ]
    if failed_fits:
        return {
            "state": "method_failure",
            "invalid_reason": "lloyd_nonconvergence_in_complete_candidate_record",
            "invalid_variants": failed_fits,
            "ordered_labels": [],
            "selected_label": None,
            "selected_bytes": None,
            "quality": {},
            "all_lloyd_starts_fixed": False,
            "complete_blob_tiebreak_reachable": False,
        }

    quality = {
        record["label"]: evaluate_quality(baseline, record["metrics"])
        for record in normalized
    }
    invalid_quality = [label for label, record in quality.items() if record["state"] == "invalid"]
    if invalid_quality:
        return {
            "state": "invalid",
            "invalid_reason": "quality_repeat_spread_invalid",
            "invalid_variants": invalid_quality,
            "ordered_labels": [],
            "selected_label": None,
            "selected_bytes": None,
            "quality": quality,
            "all_lloyd_starts_fixed": True,
            "complete_blob_tiebreak_reachable": False,
        }

    ordered = sorted(
        normalized,
        key=lambda record: (
            record["complete_bytes"],
            record["family_rank"],
            record["matched_stage_count"],
            record["base_k"],
            record["label_bytes"],
        ),
    )
    selected: dict[str, Any] | None = None
    ambiguous_before_selection: str | None = None
    for record in ordered:
        state = quality[record["label"]]["state"]
        if state == "definitely_failed":
            continue
        if state == "ambiguous":
            ambiguous_before_selection = record["label"]
            break
        if state == "definitely_qualified":
            selected = record
            break
        raise RuntimeError(f"unhandled quality state {state!r}")  # pragma: no cover

    common = {
        "ordered_labels": [record["label"] for record in ordered],
        "quality": quality,
        "all_lloyd_starts_fixed": True,
        "complete_blob_tiebreak_reachable": False,
    }
    if ambiguous_before_selection is not None:
        return {
            **common,
            "state": "invalid",
            "invalid_reason": "ambiguous_variant_precedes_first_definite_qualifier",
            "invalid_variants": [ambiguous_before_selection],
            "selected_label": None,
            "selected_bytes": None,
        }
    if selected is None:
        ambiguous = [label for label, record in quality.items() if record["state"] == "ambiguous"]
        if ambiguous:
            return {
                **common,
                "state": "invalid",
                "invalid_reason": "no_definite_qualifier_and_at_least_one_ambiguous_variant",
                "invalid_variants": ambiguous,
                "selected_label": None,
                "selected_bytes": None,
            }
        return {
            **common,
            "state": "method_failure",
            "invalid_reason": None,
            "invalid_variants": [],
            "selected_label": None,
            "selected_bytes": None,
        }
    return {
        **common,
        "state": "selected",
        "invalid_reason": None,
        "invalid_variants": [],
        "selected_label": selected["label"],
        "selected_bytes": selected["complete_bytes"],
    }


def rational_order_indices(numerators: Sequence[int], denominators: Sequence[int]) -> list[int]:
    """Order exact positive rationals by cross-product, then original index."""

    if len(numerators) != len(denominators) or not numerators:
        raise ActualCoderError("rational ordering requires equal nonempty sequences")
    nums = tuple(_positive_int(value, "rational numerator") for value in numerators)
    dens = tuple(_positive_int(value, "rational denominator") for value in denominators)

    def compare(left: int, right: int) -> int:
        lhs = nums[left] * dens[right]
        rhs = nums[right] * dens[left]
        if lhs < rhs:
            return -1
        if lhs > rhs:
            return 1
        return (left > right) - (left < right)

    return sorted(range(len(nums)), key=cmp_to_key(compare))


@lru_cache(maxsize=32)
def _bootstrap_statistic_cached(
    numerators: tuple[int, ...],
    denominators: tuple[int, ...],
    threshold_num: int,
    threshold_den: int,
) -> tuple[int, int, int, int, int]:
    matrix = oracle.bootstrap_index_matrix()
    pairs = [
        (
            math.prod(numerators[int(index)] for index in indices),
            math.prod(denominators[int(index)] for index in indices),
            replicate,
        )
        for replicate, indices in enumerate(matrix)
    ]

    def compare(left: tuple[int, int, int], right: tuple[int, int, int]) -> int:
        lhs = left[0] * right[1]
        rhs = right[0] * left[1]
        if lhs < rhs:
            return -1
        if lhs > rhs:
            return 1
        return (left[2] > right[2]) - (left[2] < right[2])

    pairs.sort(key=cmp_to_key(compare))
    upper_num, upper_den, upper_replicate = pairs[oracle.BOOTSTRAP_UPPER_INDEX]
    strict_count = sum(
        numerator * threshold_den < denominator * threshold_num
        for numerator, denominator, _ in pairs
    )
    return upper_num, upper_den, upper_replicate, strict_count, len(pairs)


def bootstrap_order_statistic(
    numerators: Sequence[int],
    denominators: Sequence[int],
    *,
    threshold_numerator_power: int,
    threshold_denominator_power: int,
) -> dict[str, Any]:
    """Compute the exact frozen 97,499 order statistic and strict threshold gate."""

    if len(numerators) != 8 or len(denominators) != 8:
        raise ActualCoderError("bootstrap requires exactly eight paired image ratios")
    nums = tuple(_positive_int(value, "bootstrap numerator") for value in numerators)
    dens = tuple(_positive_int(value, "bootstrap denominator") for value in denominators)
    threshold_num = _positive_int(threshold_numerator_power, "bootstrap threshold numerator")
    threshold_den = _positive_int(
        threshold_denominator_power, "bootstrap threshold denominator"
    )
    upper_num, upper_den, upper_replicate, strict_count, replicate_count = (
        _bootstrap_statistic_cached(nums, dens, threshold_num, threshold_den)
    )
    upper_strict = upper_num * threshold_den < upper_den * threshold_num
    if upper_strict != (strict_count >= oracle.BOOTSTRAP_UPPER_INDEX + 1):
        raise RuntimeError("bootstrap order statistic/count equivalence failed")
    return {
        "matrix_sha256": oracle.BOOTSTRAP_MATRIX_SHA256,
        "replicates": replicate_count,
        "upper_index_zero_based": oracle.BOOTSTRAP_UPPER_INDEX,
        "upper_unsimplified_numerator": upper_num,
        "upper_unsimplified_denominator": upper_den,
        "upper_replicate_index": upper_replicate,
        "threshold_numerator_power": threshold_num,
        "threshold_denominator_power": threshold_den,
        "strictly_below_threshold": strict_count,
        "upper_strictly_below_threshold_exact": upper_strict,
        "upper_geometric_mean_ratio_diagnostic": math.exp(
            (math.log(upper_num) - math.log(upper_den)) / 8.0
        ),
        "ordering": "exact_cross_product_then_ascending_replicate_index",
    }


def conservative_median_psnr(losses: Sequence[float]) -> dict[str, Any]:
    """Apply the frozen eight-image conventional median and its exclusion band."""

    if len(losses) != 8:
        raise ActualCoderError("tuple median requires exactly eight PSNR losses")
    values = [_finite_float(value, "conservative PSNR loss") for value in losses]
    order = sorted(range(8), key=lambda index: (values[index], index))
    median = (values[order[3]] + values[order[4]]) / 2.0
    if median <= MEDIAN_PSNR_PASS_MAX:
        state = "definitely_passes"
    elif median > MEDIAN_PSNR_FAIL_ABOVE:
        state = "definitely_fails"
    else:
        state = "ambiguous"
    return {
        "sorted_image_indices": order,
        "sorted_losses": [values[index] for index in order],
        "middle_image_indices": [order[3], order[4]],
        "median": median,
        "median_float_hex": median.hex(),
        "state": state,
        "pass_max": MEDIAN_PSNR_PASS_MAX,
        "fail_above": MEDIAN_PSNR_FAIL_ABOVE,
    }


def _validate_primary_resource(value: Any) -> dict[str, Any]:
    record = _mapping(value, "primary_resource")
    _exact_keys(record, _PRIMARY_RESOURCE_KEYS, "primary_resource")
    candidate_label = record["candidate_label"]
    q_name = record["q_name"]
    if candidate_label not in VARIANT_ORDER or q_name not in PRIMARY_ORDER:
        raise ActualCoderError("primary resource references an unknown candidate or Q format")
    return {
        "candidate_label": candidate_label,
        "q_name": q_name,
        "candidate_median_ns": _positive_int(
            record["candidate_median_ns"], "primary_resource.candidate_median_ns"
        ),
        "q_median_ns": _positive_int(record["q_median_ns"], "primary_resource.q_median_ns"),
        "candidate_peak_rss_bytes": _positive_int(
            record["candidate_peak_rss_bytes"],
            "primary_resource.candidate_peak_rss_bytes",
        ),
        "q_peak_rss_bytes": _positive_int(
            record["q_peak_rss_bytes"], "primary_resource.q_peak_rss_bytes"
        ),
        "fresh_comp011": _required_bool(
            record["fresh_comp011"], "primary_resource.fresh_comp011"
        ),
        "schedule_exact": _required_bool(
            record["schedule_exact"], "primary_resource.schedule_exact"
        ),
    }


def _validate_secondary_resource(value: Any) -> dict[str, Any]:
    record = _mapping(value, "secondary_resource")
    _exact_keys(record, _SECONDARY_RESOURCE_KEYS, "secondary_resource")
    return {
        "ssp2l_median_ns": _positive_int(
            record["ssp2l_median_ns"], "secondary_resource.ssp2l_median_ns"
        ),
        "sspl1_median_ns": _positive_int(
            record["sspl1_median_ns"], "secondary_resource.sspl1_median_ns"
        ),
        "fresh_comp011": _required_bool(
            record["fresh_comp011"], "secondary_resource.fresh_comp011"
        ),
        "schedule_exact": _required_bool(
            record["schedule_exact"], "secondary_resource.schedule_exact"
        ),
    }


def _validate_row(row: Mapping[str, Any]) -> dict[str, Any]:
    verify_record(row)
    _exact_keys(row, _ROW_KEYS, "row")
    if row.get("schema") != ROW_SCHEMA:
        raise ActualCoderError("unexpected COMP-011 row schema")
    image = row.get("image_id")
    if image not in FROZEN_IMAGES:
        raise ActualCoderError(f"unexpected development image {image!r}")
    bits = row.get("bit_tuple")
    if not isinstance(bits, list) or tuple(bits) not in FROZEN_TUPLES:
        raise ActualCoderError("unexpected COMP-011 bit tuple")
    validity = row.get("validity")
    method_state = row.get("method_state")
    if validity not in ("valid", "invalid"):
        raise ActualCoderError("row validity must be valid or invalid")

    common = {
        "image_id": image,
        "bit_tuple": tuple(bits),
        "record_sha256": row["record_sha256"],
    }
    if validity == "invalid":
        reason = row.get("invalid_reason")
        if not isinstance(reason, str) or not reason:
            raise ActualCoderError("invalid row requires a nonempty invalid_reason")
        if (
            method_state != "invalid"
            or row.get("lloyd_all_fixed") is not None
            or any(
                row.get(name) is not None
                for name in (
                    "formats",
                    "primary_metrics",
                    "variants",
                    "primary_resource",
                    "secondary_resource",
                )
            )
        ):
            raise ActualCoderError("invalid row must not carry decision inputs")
        return {**common, "state": "invalid", "invalid_reason": reason}

    if row.get("invalid_reason") is not None:
        raise ActualCoderError("valid row must have invalid_reason=null")
    if method_state == "lloyd_failure":
        if (
            row.get("lloyd_all_fixed") is not False
            or any(
                row.get(name) is not None
                for name in (
                    "formats",
                    "primary_metrics",
                    "variants",
                    "primary_resource",
                    "secondary_resource",
                )
            )
        ):
            raise ActualCoderError("Lloyd terminal row carries forbidden later-stage inputs")
        return {**common, "state": "lloyd_failure", "invalid_reason": None}
    if method_state != "complete" or row.get("lloyd_all_fixed") is not True:
        raise ActualCoderError("valid complete row must attest all Lloyd starts fixed")

    primary = select_primary(row.get("formats"))
    baseline_metrics = _metric_repetitions(row.get("primary_metrics"), "primary_metrics")
    candidate = select_variant(baseline_metrics, row.get("variants"))
    primary_resource = (
        None
        if row.get("primary_resource") is None
        else _validate_primary_resource(row.get("primary_resource"))
    )
    secondary_resource = (
        None
        if row.get("secondary_resource") is None
        else _validate_secondary_resource(row.get("secondary_resource"))
    )
    return {
        **common,
        "state": "complete",
        "invalid_reason": None,
        "primary": primary,
        "candidate": candidate,
        "primary_resource": primary_resource,
        "secondary_resource": secondary_resource,
    }


def _ordered_tuple_rows(rows: Sequence[Mapping[str, Any]], bits: tuple[int, ...]) -> list[dict[str, Any]]:
    selected = [row for row in rows if tuple(row["bit_tuple"]) == bits]
    if len(selected) != 8:
        raise ActualCoderError(f"tuple {bits} must contain exactly eight rows")
    by_image = {str(row["image_id"]): row for row in selected}
    if set(by_image) != set(FROZEN_IMAGES):
        raise ActualCoderError(f"tuple {bits} does not contain each frozen image exactly once")
    return [dict(by_image[image]) for image in FROZEN_IMAGES]


def _primary_resource_result(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    resources = []
    for row in rows:
        resource = row.get("primary_resource")
        if not isinstance(resource, Mapping):
            return {"state": "invalid", "invalid_reason": "missing_primary_resource_record"}
        if (
            resource["candidate_label"] != row["candidate"]["selected_label"]
            or resource["q_name"] != row["primary"]["selected_name"]
            or resource["fresh_comp011"] is not True
            or resource["schedule_exact"] is not True
        ):
            return {
                "state": "invalid",
                "invalid_reason": "primary_resource_identity_or_schedule_mismatch",
            }
        resources.append(resource)
    time_num = [int(resource["candidate_median_ns"]) for resource in resources]
    time_den = [int(resource["q_median_ns"]) for resource in resources]
    order = rational_order_indices(time_num, time_den)
    upper_index = order[4]
    gates = {
        "upper_median_time_ratio_lte_1_5_exact": 2 * time_num[upper_index]
        <= 3 * time_den[upper_index],
        "worst_time_ratio_lte_2_exact": all(
            numerator <= 2 * denominator
            for numerator, denominator in zip(time_num, time_den, strict=True)
        ),
        "all_peak_rss_ratios_lte_1_5_exact": all(
            2 * int(resource["candidate_peak_rss_bytes"])
            <= 3 * int(resource["q_peak_rss_bytes"])
            for resource in resources
        ),
    }
    return {
        "state": "valid",
        "invalid_reason": None,
        "candidate_median_ns": time_num,
        "q_median_ns": time_den,
        "ratio_order_image_indices": order,
        "upper_median_image_index": upper_index,
        "time_ratios_diagnostic": [
            numerator / denominator
            for numerator, denominator in zip(time_num, time_den, strict=True)
        ],
        "gates": gates,
        "pass": all(gates.values()),
    }


def _secondary_resource_result(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    resources = []
    for row in rows:
        resource = row.get("secondary_resource")
        if not isinstance(resource, Mapping):
            return {"state": "invalid", "invalid_reason": "missing_secondary_resource_record"}
        if resource["fresh_comp011"] is not True or resource["schedule_exact"] is not True:
            return {
                "state": "invalid",
                "invalid_reason": "secondary_resource_schedule_mismatch",
            }
        resources.append(resource)
    time_num = [int(resource["ssp2l_median_ns"]) for resource in resources]
    time_den = [int(resource["sspl1_median_ns"]) for resource in resources]
    order = rational_order_indices(time_num, time_den)
    upper_index = order[4]
    gate = 4 * time_num[upper_index] <= 5 * time_den[upper_index]
    return {
        "state": "valid",
        "invalid_reason": None,
        "ssp2l_median_ns": time_num,
        "sspl1_median_ns": time_den,
        "ratio_order_image_indices": order,
        "upper_median_image_index": upper_index,
        "upper_median_time_ratio_lte_1_25_exact": gate,
        "time_ratios_diagnostic": [
            numerator / denominator
            for numerator, denominator in zip(time_num, time_den, strict=True)
        ],
        "pass": gate,
    }


def _main_tuple_result(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    candidate_bytes = [int(row["candidate"]["selected_bytes"]) for row in rows]
    primary_bytes = [int(row["primary"]["selected_bytes"]) for row in rows]
    z_bytes = [int(row["primary"]["formats"]["ssp2z"]["complete_bytes"]) for row in rows]
    b0_bytes = [int(row["primary"]["formats"]["sspl1"]["complete_bytes"]) for row in rows]
    rate_gates = {
        "geometric_mean_lte_0_95_exact": math.prod(candidate_bytes) * 20**8
        <= math.prod(primary_bytes) * 19**8,
        "strict_image_wins_gte_7": sum(
            candidate < primary
            for candidate, primary in zip(candidate_bytes, primary_bytes, strict=True)
        )
        >= 7,
        "worst_ratio_lte_1_02_exact": all(
            100 * candidate <= 102 * primary
            for candidate, primary in zip(candidate_bytes, primary_bytes, strict=True)
        ),
    }
    bootstrap = bootstrap_order_statistic(
        candidate_bytes,
        primary_bytes,
        threshold_numerator_power=49**8,
        threshold_denominator_power=50**8,
    )
    rate_gates["bootstrap_upper_lt_0_98_exact"] = bool(
        bootstrap["upper_strictly_below_threshold_exact"]
    )
    operational_gates = {
        "geometric_mean_v_over_z_lt_1_exact": math.prod(candidate_bytes) < math.prod(z_bytes),
        "strict_v_over_z_wins_gte_7": sum(
            candidate < control
            for candidate, control in zip(candidate_bytes, z_bytes, strict=True)
        )
        >= 7,
    }
    losses = [
        float(row["candidate"]["quality"][row["candidate"]["selected_label"]]["conservative"][
            "psnr_loss"
        ])
        for row in rows
    ]
    median = conservative_median_psnr(losses)
    resources = _primary_resource_result(rows)
    if resources["state"] != "valid":
        return {
            "state": "invalid",
            "invalid_reason": resources["invalid_reason"],
            "bit_tuple": list(rows[0]["bit_tuple"]),
        }
    quality_gates = {
        "all_selected_candidates_definitely_qualified": all(
            row["candidate"]["quality"][row["candidate"]["selected_label"]]["state"]
            == "definitely_qualified"
            for row in rows
        ),
        "median_psnr_loss_definitely_passes": median["state"] == "definitely_passes",
    }
    convergence_gates = {
        "all_prescribed_lloyd_starts_fixed": all(
            row["candidate"]["all_lloyd_starts_fixed"] is True for row in rows
        )
    }
    invalid_reason = (
        "median_psnr_loss_in_exclusion_band" if median["state"] == "ambiguous" else None
    )
    if invalid_reason is not None:
        state = "invalid"
        passed = False
    else:
        state = "valid"
        passed = (
            all(rate_gates.values())
            and all(operational_gates.values())
            and all(quality_gates.values())
            and all(convergence_gates.values())
            and resources["pass"]
        )
    return {
        "state": state,
        "invalid_reason": invalid_reason,
        "bit_tuple": list(rows[0]["bit_tuple"]),
        "image_ids": list(FROZEN_IMAGES),
        "candidate_bytes": candidate_bytes,
        "primary_bytes": primary_bytes,
        "primary_names": [row["primary"]["selected_name"] for row in rows],
        "ssp2z_bytes": z_bytes,
        "sspl1_bytes": b0_bytes,
        "actual_rate": {
            "geometric_mean_ratio_diagnostic": _ratio_diagnostic(
                candidate_bytes, primary_bytes
            ),
            "strict_image_wins": sum(
                candidate < primary
                for candidate, primary in zip(candidate_bytes, primary_bytes, strict=True)
            ),
            "worst_ratio_diagnostic": max(
                candidate / primary
                for candidate, primary in zip(candidate_bytes, primary_bytes, strict=True)
            ),
            "gates": rate_gates,
            "bootstrap": bootstrap,
            "pass": all(rate_gates.values()),
        },
        "operational_bundle": {
            "v_over_z_geometric_mean_ratio_diagnostic": _ratio_diagnostic(
                candidate_bytes, z_bytes
            ),
            "z_over_b0_geometric_mean_ratio_diagnostic": _ratio_diagnostic(z_bytes, b0_bytes),
            "strict_v_over_z_wins": sum(
                candidate < control
                for candidate, control in zip(candidate_bytes, z_bytes, strict=True)
            ),
            "gates": operational_gates,
            "pass": all(operational_gates.values()),
        },
        "quality": {"median_psnr": median, "gates": quality_gates},
        "convergence": {"gates": convergence_gates},
        "resources": resources,
        "all_gates_pass": passed,
    }


def _secondary_tuple_result(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    l_bytes = [int(row["primary"]["formats"]["ssp2l"]["complete_bytes"]) for row in rows]
    b0_bytes = [int(row["primary"]["formats"]["sspl1"]["complete_bytes"]) for row in rows]
    resource = _secondary_resource_result(rows)
    if resource["state"] != "valid":
        return {
            "state": "invalid",
            "invalid_reason": resource["invalid_reason"],
            "bit_tuple": list(rows[0]["bit_tuple"]),
        }
    bootstrap = bootstrap_order_statistic(
        l_bytes,
        b0_bytes,
        threshold_numerator_power=1,
        threshold_denominator_power=1,
    )
    gates = {
        "geometric_mean_lte_0_98_exact": math.prod(l_bytes) * 50**8
        <= math.prod(b0_bytes) * 49**8,
        "strict_image_wins_gte_7": sum(
            candidate < baseline
            for candidate, baseline in zip(l_bytes, b0_bytes, strict=True)
        )
        >= 7,
        "bootstrap_upper_lt_1_exact": bool(bootstrap["upper_strictly_below_threshold_exact"]),
        "exact_symbol_and_boundary_reconstruction": all(
            row["primary"]["formats"]["ssp2l"]["exact"] is True
            and row["primary"]["formats"]["sspl1"]["exact"] is True
            for row in rows
        ),
        "decode_upper_median_ratio_lte_1_25_exact": bool(resource["pass"]),
    }
    return {
        "state": "valid",
        "invalid_reason": None,
        "bit_tuple": list(rows[0]["bit_tuple"]),
        "image_ids": list(FROZEN_IMAGES),
        "ssp2l_bytes": l_bytes,
        "sspl1_bytes": b0_bytes,
        "geometric_mean_ratio_diagnostic": _ratio_diagnostic(l_bytes, b0_bytes),
        "strict_image_wins": sum(
            candidate < baseline
            for candidate, baseline in zip(l_bytes, b0_bytes, strict=True)
        ),
        "bootstrap": bootstrap,
        "resources": resource,
        "gates": gates,
        "pass": all(gates.values()),
    }


def _invalid_analysis(
    rows: Sequence[Mapping[str, Any]], reason: str, *, row_seals: Sequence[str]
) -> dict[str, Any]:
    return seal_record(
        {
            "schema": ANALYSIS_SCHEMA,
            "protocol": PROTOCOL,
            "task_sha256": TASK_SHA256,
            "row_count": len(rows),
            "row_record_sha256": list(row_seals),
            "status": "invalid_no_decision",
            "invalid_reason": reason,
            "decision": INVALID_DECISION,
            "ssp2l_decision": INVALID_DECISION,
            "tuple_results": [],
            "secondary_tuple_results": [],
            "early_failure": None,
            "confirmation_access_authorized_in_this_task": False,
            "claim_scope": "invalid-no-decision",
        },
        "analysis_sha256",
    )


def analyze_rows(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Validate sealed rows and recompute both frozen COMP-011 decisions.

    A source-valid early Lloyd failure is represented by one terminal row and produces the frozen
    abandonment decision without authorizing the secondary screen.  The complete path requires
    all sixteen identities.  Structural corruption raises :class:`ActualCoderError`; scientific
    invalidity returns a sealed ``INVALID_NO_DECISION`` analysis.
    """

    materialized = list(rows)
    if not materialized:
        raise ActualCoderError("analysis requires at least one sealed row")
    validated = [_validate_row(_mapping(row, "row")) for row in materialized]
    identities = [(row["image_id"], tuple(row["bit_tuple"])) for row in validated]
    if len(set(identities)) != len(identities):
        raise ActualCoderError("duplicate COMP-011 row identity")
    validated.sort(
        key=lambda row: (
            FROZEN_TUPLES.index(tuple(row["bit_tuple"])),
            FROZEN_IMAGES.index(str(row["image_id"])),
        )
    )
    row_seals = [str(row["record_sha256"]) for row in validated]

    invalid_rows = [row for row in validated if row["state"] == "invalid"]
    if invalid_rows:
        return _invalid_analysis(
            validated,
            f"invalid_input_row:{invalid_rows[0]['invalid_reason']}",
            row_seals=row_seals,
        )

    lloyd_failures = [row for row in validated if row["state"] == "lloyd_failure"]
    if lloyd_failures:
        if len(validated) != 1 or len(lloyd_failures) != 1:
            return _invalid_analysis(
                validated,
                "early_lloyd_failure_must_be_the_only_analysis_row",
                row_seals=row_seals,
            )
        failure = lloyd_failures[0]
        return seal_record(
            {
                "schema": ANALYSIS_SCHEMA,
                "protocol": PROTOCOL,
                "task_sha256": TASK_SHA256,
                "row_count": 1,
                "row_record_sha256": row_seals,
                "status": "valid_method_failure",
                "invalid_reason": None,
                "decision": MAIN_DECISION_ABANDON,
                "ssp2l_decision": SECONDARY_DECISION_INAPPLICABLE,
                "tuple_results": [],
                "secondary_tuple_results": [],
                "early_failure": {
                    "kind": "required_lloyd_start_nonconvergence",
                    "image_id": failure["image_id"],
                    "bit_tuple": list(failure["bit_tuple"]),
                },
                "confirmation_access_authorized_in_this_task": False,
                "claim_scope": "valid-preregistered-lloyd-method-failure-development-only",
            },
            "analysis_sha256",
        )

    expected = {(image, bits) for image in FROZEN_IMAGES for bits in FROZEN_TUPLES}
    if len(validated) != 16 or set(identities) != expected:
        raise ActualCoderError("complete analysis requires the exact sixteen-cell grid")
    if any(row["state"] != "complete" for row in validated):  # pragma: no cover - exhaustive
        raise RuntimeError("unhandled validated row state")

    protocol_invalid = []
    for row in validated:
        if row["primary"]["state"] != "selected":
            protocol_invalid.append(row["primary"]["invalid_reason"])
        if row["candidate"]["state"] == "invalid":
            protocol_invalid.append(row["candidate"]["invalid_reason"])
        if row["candidate"]["state"] == "method_failure" and row["candidate"].get(
            "invalid_reason"
        ):
            protocol_invalid.append(row["candidate"]["invalid_reason"])
        secondary = row.get("secondary_resource")
        if secondary is None:
            protocol_invalid.append("missing_secondary_resource_record")
        elif secondary["fresh_comp011"] is not True or secondary["schedule_exact"] is not True:
            protocol_invalid.append("secondary_resource_schedule_mismatch")
    if protocol_invalid:
        return _invalid_analysis(
            validated, str(protocol_invalid[0]), row_seals=row_seals
        )

    all_selected = all(row["candidate"]["state"] == "selected" for row in validated)
    any_selection_failure = any(
        row["candidate"]["state"] == "method_failure" for row in validated
    )
    if not all_selected and not any_selection_failure:
        return _invalid_analysis(
            validated, "candidate_selection_state_grid_is_inconsistent", row_seals=row_seals
        )
    if all_selected:
        if any(row.get("primary_resource") is None for row in validated):
            return _invalid_analysis(
                validated, "missing_primary_resource_record", row_seals=row_seals
            )
    elif any(row.get("primary_resource") is not None for row in validated):
        return _invalid_analysis(
            validated,
            "primary_resource_schedule_ran_without_all_sixteen_selections",
            row_seals=row_seals,
        )

    secondary_results = [
        _secondary_tuple_result(_ordered_tuple_rows(validated, bits)) for bits in FROZEN_TUPLES
    ]
    invalid_secondary = next(
        (result for result in secondary_results if result["state"] == "invalid"), None
    )
    if invalid_secondary is not None:
        return _invalid_analysis(
            validated,
            str(invalid_secondary["invalid_reason"]),
            row_seals=row_seals,
        )

    tuple_results: list[dict[str, Any]] = []
    if all_selected:
        tuple_results = [
            _main_tuple_result(_ordered_tuple_rows(validated, bits)) for bits in FROZEN_TUPLES
        ]
        invalid_main = next(
            (result for result in tuple_results if result["state"] == "invalid"), None
        )
        if invalid_main is not None:
            return _invalid_analysis(
                validated, str(invalid_main["invalid_reason"]), row_seals=row_seals
            )

    main_pass = all_selected and all(result["all_gates_pass"] for result in tuple_results)
    secondary_pass = all(result["pass"] for result in secondary_results)
    decision = MAIN_DECISION_AUTHORIZE if main_pass else MAIN_DECISION_ABANDON
    secondary_decision = (
        SECONDARY_DECISION_AUTHORIZE if secondary_pass else SECONDARY_DECISION_NONE
    )
    status = "valid_complete" if main_pass else "valid_method_failure"
    return seal_record(
        {
            "schema": ANALYSIS_SCHEMA,
            "protocol": PROTOCOL,
            "task_sha256": TASK_SHA256,
            "row_count": len(validated),
            "row_record_sha256": row_seals,
            "status": status,
            "invalid_reason": None,
            "decision": decision,
            "ssp2l_decision": secondary_decision,
            "tuple_results": tuple_results,
            "secondary_tuple_results": secondary_results,
            "early_failure": None,
            "confirmation_access_authorized_in_this_task": False,
            "claim_scope": (
                "complete-stream-rate-conservative-quality-convergence-and-resource-"
                "development-evidence-only"
            ),
        },
        "analysis_sha256",
    )


__all__ = [
    "ANALYSIS_SCHEMA",
    "ActualCoderError",
    "FROZEN_IMAGES",
    "FROZEN_TUPLES",
    "INVALID_DECISION",
    "MAIN_DECISION_ABANDON",
    "MAIN_DECISION_AUTHORIZE",
    "PRIMARY_ORDER",
    "PROTOCOL",
    "QUALITY_BANDS",
    "QUALITY_THRESHOLDS",
    "ROW_SCHEMA",
    "SECONDARY_DECISION_AUTHORIZE",
    "SECONDARY_DECISION_INAPPLICABLE",
    "SECONDARY_DECISION_NONE",
    "SPREAD_CAPS",
    "TASK_SHA256",
    "VARIANT_ORDER",
    "VARIANT_SPECS",
    "analyze_rows",
    "bootstrap_order_statistic",
    "canonical_json",
    "classify_banded_metric",
    "conservative_median_psnr",
    "evaluate_quality",
    "rational_order_indices",
    "seal_record",
    "select_primary",
    "select_variant",
    "verify_record",
]
