"""Pure analysis helpers for the COMP-007 gauge-free covariance codec assay.

The benchmark runner owns fitting, encoding, cold decoding, and artifact I/O.  This module keeps
the scientific reduction separate: it constructs stable rate--distortion envelopes, compares
them only on legal common support, clusters the inference at source-image level, and evaluates
the frozen development gates from ``tasks/COMP-007-gauge-free-covariance-codec.md``.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
import json
import math
from typing import Any

import numpy as np


CHARTS = ("current_rs", "canonical_rs", "log_spd")
CODERS = ("zlib", "zstd")
COUNTS = (640, 2560)
TOTAL_BITS = (12, 18, 24)
BITS_MIN = 3
BITS_MAX = 10
PSNR_GRID_POINTS = 25
MINIMUM_OVERLAP_DB = 0.10
BOOTSTRAP_REPLICATES = 20_000
BOOTSTRAP_SEED = 20_260_716

_ENCODE_ALIASES = (
    ("encode_seconds", 1.0),
    ("encode_time_seconds", 1.0),
    ("encode_time_s", 1.0),
    ("encode_s", 1.0),
    ("encode_ms", 1e-3),
    ("encode_time_ms", 1e-3),
)
_DECODE_ALIASES = (
    ("cold_decode_seconds", 1.0),
    ("cold_decode_time_seconds", 1.0),
    ("cold_decode_time_s", 1.0),
    ("decode_seconds", 1.0),
    ("decode_time_seconds", 1.0),
    ("decode_time_s", 1.0),
    ("decode_s", 1.0),
    ("cold_decode_ms", 1e-3),
    ("decode_ms", 1e-3),
    ("decode_time_ms", 1e-3),
)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("analysis outputs must be finite")
        return float(value)
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return str(value)


def _positive_float(row: Mapping[str, Any], key: str) -> float:
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"row has no numeric {key!r}") from exc
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{key} must be finite and positive")
    return value


def _finite_float(row: Mapping[str, Any], key: str) -> float:
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"row has no numeric {key!r}") from exc
    if not math.isfinite(value):
        raise ValueError(f"{key} must be finite")
    return value


def _timing_seconds(row: Mapping[str, Any], kind: str) -> float:
    aliases = _ENCODE_ALIASES if kind == "encode" else _DECODE_ALIASES
    for key, scale in aliases:
        if key in row:
            value = _positive_float(row, key) * scale
            if not math.isfinite(value) or value <= 0.0:
                break
            return value
    raise ValueError(f"row has no finite positive {kind} timing")


def _allocation_value(row: Mapping[str, Any]) -> Any:
    if "allocation" not in row:
        raise ValueError("candidate row is missing allocation")
    return _json_safe(row["allocation"])


def _allocation_token(row: Mapping[str, Any]) -> str:
    return json.dumps(_allocation_value(row), sort_keys=True, separators=(",", ":"))


def _operating_point_key(row: Mapping[str, Any]) -> tuple[int, str]:
    try:
        total_bits = int(row["total_bits"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("candidate row is missing integer total_bits") from exc
    return total_bits, _allocation_token(row)


def expected_allocations() -> tuple[tuple[int, int, int], ...]:
    """Return the frozen exhaustive ordered covariance-bit allocations."""
    return tuple(
        (first, second, third)
        for total in TOTAL_BITS
        for first in range(BITS_MIN, BITS_MAX + 1)
        for second in range(BITS_MIN, BITS_MAX + 1)
        for third in range(BITS_MIN, BITS_MAX + 1)
        if first + second + third == total
    )


def pareto_envelope(
    rows: Sequence[Mapping[str, Any]],
    *,
    rate_key: str = "total_bytes",
    quality_key: str = "psnr",
) -> list[dict[str, Any]]:
    """Return a stable lower-rate/higher-quality Pareto envelope.

    Exact rate/quality duplicates retain the first input row.  Equal-rate lower-quality and
    equal-quality higher-rate rows are removed.  The returned points therefore have strictly
    increasing rate and quality and are safe for interpolation in either direction.
    """
    indexed: list[tuple[float, float, int, Mapping[str, Any]]] = []
    for index, row in enumerate(rows):
        rate = _positive_float(row, rate_key)
        quality = _finite_float(row, quality_key)
        indexed.append((rate, quality, index, row))
    indexed.sort(key=lambda item: (item[0], -item[1], item[2]))

    envelope: list[dict[str, Any]] = []
    best_quality = -math.inf
    for _, quality, _, row in indexed:
        if quality <= best_quality:
            continue
        envelope.append(_json_safe(dict(row)))
        best_quality = quality
    return envelope


def _common_quality_grid(
    envelopes: Sequence[Sequence[Mapping[str, Any]]],
    *,
    quality_key: str,
    minimum_overlap_db: float,
    grid_points: int,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    if grid_points < 2:
        raise ValueError("grid_points must be at least two")
    if minimum_overlap_db < 0.0:
        raise ValueError("minimum_overlap_db must be non-negative")
    if not envelopes or any(not envelope for envelope in envelopes):
        return None, {
            "status": "unavailable",
            "reason": "empty_pareto_envelope",
            "minimum_required_db": float(minimum_overlap_db),
        }
    minima = [min(_finite_float(row, quality_key) for row in envelope) for envelope in envelopes]
    maxima = [max(_finite_float(row, quality_key) for row in envelope) for envelope in envelopes]
    lower = max(minima)
    upper = min(maxima)
    width = upper - lower
    support = {
        "lower_psnr_db": float(lower),
        "upper_psnr_db": float(upper),
        "width_db": float(width),
        "minimum_required_db": float(minimum_overlap_db),
        "grid_points": int(grid_points),
    }
    if width + 1e-12 < minimum_overlap_db:
        return None, {
            "status": "unavailable",
            "reason": "insufficient_common_psnr_overlap",
            **support,
        }
    return np.linspace(lower, upper, grid_points, dtype=np.float64), {
        "status": "available",
        **support,
    }


def _interpolate_log_values(
    envelope: Sequence[Mapping[str, Any]],
    quality_grid: np.ndarray,
    *,
    value_key: str,
    quality_key: str,
) -> np.ndarray:
    qualities = np.asarray([_finite_float(row, quality_key) for row in envelope], dtype=np.float64)
    values = np.asarray([_positive_float(row, value_key) for row in envelope], dtype=np.float64)
    if np.any(np.diff(qualities) <= 0.0):
        raise ValueError("Pareto quality coordinates must be strictly increasing")
    if quality_grid[0] < qualities[0] - 1e-12 or quality_grid[-1] > qualities[-1] + 1e-12:
        raise ValueError("interpolation would extrapolate outside the Pareto envelope")
    return np.interp(quality_grid, qualities, np.log(values))


def _selected_operating_points(
    rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[int, str], Mapping[str, Any]]:
    """Select the stable minimum-byte row for each frozen allocation."""
    selected: dict[tuple[int, str], tuple[float, float, int, Mapping[str, Any]]] = {}
    for index, row in enumerate(rows):
        key = _operating_point_key(row)
        candidate = (
            _positive_float(row, "total_bytes"),
            -_finite_float(row, "psnr"),
            index,
            row,
        )
        if key not in selected or candidate[:3] < selected[key][:3]:
            selected[key] = candidate
    return {key: value[3] for key, value in selected.items()}


def worst_matched_psnr_shortfall(
    challenger_rows: Sequence[Mapping[str, Any]],
    reference_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Compare reconstruction quality at identical bit-allocation operating points.

    Positive shortfall means the challenger has lower PSNR.  Mismatched allocation grids are
    unavailable rather than silently intersected.
    """
    challenger = _selected_operating_points(challenger_rows)
    reference = _selected_operating_points(reference_rows)
    challenger_keys = set(challenger)
    reference_keys = set(reference)
    if challenger_keys != reference_keys or not challenger_keys:
        return {
            "status": "unavailable",
            "reason": "mismatched_operating_point_grid",
            "missing_from_challenger": [
                [bits, json.loads(allocation)]
                for bits, allocation in sorted(reference_keys - challenger_keys)
            ],
            "missing_from_reference": [
                [bits, json.loads(allocation)]
                for bits, allocation in sorted(challenger_keys - reference_keys)
            ],
        }

    records: list[dict[str, Any]] = []
    for bits, allocation in sorted(challenger_keys):
        challenger_row = challenger[(bits, allocation)]
        reference_row = reference[(bits, allocation)]
        records.append(
            {
                "total_bits": int(bits),
                "allocation": json.loads(allocation),
                "challenger_psnr_db": _finite_float(challenger_row, "psnr"),
                "reference_psnr_db": _finite_float(reference_row, "psnr"),
                "shortfall_db": _finite_float(reference_row, "psnr")
                - _finite_float(challenger_row, "psnr"),
            }
        )
    worst = max(records, key=lambda row: (row["shortfall_db"], -row["total_bits"]))
    return {
        "status": "available",
        "operating_points": len(records),
        "worst_shortfall_db": float(worst["shortfall_db"]),
        "worst_operating_point": worst,
        "records": records,
    }


def worst_envelope_psnr_shortfall(
    challenger_rows: Sequence[Mapping[str, Any]],
    reference_rows: Sequence[Mapping[str, Any]],
    *,
    rate_key: str = "total_bytes",
    quality_key: str = "psnr",
) -> dict[str, Any]:
    """Return the worst challenger quality loss at matched complete-stream rate.

    Both Pareto envelopes are linearly interpolated in log-rate over their shared rate support.
    The union of their knots plus both support endpoints is sufficient to locate the maximum
    difference between the two piecewise-linear curves.  No extrapolation is permitted.
    """
    challenger = pareto_envelope(
        challenger_rows, rate_key=rate_key, quality_key=quality_key
    )
    reference = pareto_envelope(reference_rows, rate_key=rate_key, quality_key=quality_key)
    if not challenger or not reference:
        return {"status": "unavailable", "reason": "empty_pareto_envelope"}
    challenger_log_rate = np.log(
        np.asarray([_positive_float(row, rate_key) for row in challenger], dtype=np.float64)
    )
    reference_log_rate = np.log(
        np.asarray([_positive_float(row, rate_key) for row in reference], dtype=np.float64)
    )
    lower = max(float(challenger_log_rate[0]), float(reference_log_rate[0]))
    upper = min(float(challenger_log_rate[-1]), float(reference_log_rate[-1]))
    if upper <= lower:
        return {
            "status": "unavailable",
            "reason": "no_common_rate_support",
            "lower_bytes": float(math.exp(lower)),
            "upper_bytes": float(math.exp(upper)),
        }
    knots = np.unique(
        np.concatenate(
            [
                np.asarray([lower, upper]),
                challenger_log_rate[
                    (challenger_log_rate >= lower) & (challenger_log_rate <= upper)
                ],
                reference_log_rate[
                    (reference_log_rate >= lower) & (reference_log_rate <= upper)
                ],
            ]
        )
    )
    challenger_quality = np.interp(
        knots,
        challenger_log_rate,
        np.asarray([_finite_float(row, quality_key) for row in challenger]),
    )
    reference_quality = np.interp(
        knots,
        reference_log_rate,
        np.asarray([_finite_float(row, quality_key) for row in reference]),
    )
    shortfalls = reference_quality - challenger_quality
    worst_index = int(np.argmax(shortfalls))
    return {
        "status": "available",
        "common_rate_support": {
            "lower_bytes": float(math.exp(lower)),
            "upper_bytes": float(math.exp(upper)),
            "knot_count": int(knots.size),
        },
        "worst_shortfall_db": float(shortfalls[worst_index]),
        "worst_operating_point": {
            "matched_bytes": float(math.exp(float(knots[worst_index]))),
            "challenger_psnr_db": float(challenger_quality[worst_index]),
            "reference_psnr_db": float(reference_quality[worst_index]),
        },
        "records": [
            {
                "matched_bytes": float(math.exp(float(rate))),
                "challenger_psnr_db": float(left),
                "reference_psnr_db": float(right),
                "shortfall_db": float(shortfall),
            }
            for rate, left, right, shortfall in zip(
                knots,
                challenger_quality,
                reference_quality,
                shortfalls,
                strict=True,
            )
        ],
    }


def _matched_ratio_diagnostics(
    challenger_rows: Sequence[Mapping[str, Any]],
    reference_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    challenger = _selected_operating_points(challenger_rows)
    reference = _selected_operating_points(reference_rows)
    if set(challenger) != set(reference) or not challenger:
        return {
            "status": "unavailable",
            "reason": "mismatched_operating_point_grid",
        }
    geometry: list[float] = []
    encode: list[float] = []
    decode: list[float] = []
    try:
        for key in sorted(challenger):
            left = challenger[key]
            right = reference[key]
            geometry.append(
                _positive_float(left, "geometry_attributed_bytes")
                / _positive_float(right, "geometry_attributed_bytes")
            )
            encode.append(_timing_seconds(left, "encode") / _timing_seconds(right, "encode"))
            decode.append(_timing_seconds(left, "decode") / _timing_seconds(right, "decode"))
    except ValueError as exc:
        return {"status": "unavailable", "reason": str(exc)}
    return {
        "status": "available",
        "operating_points": len(geometry),
        "median_geometry_ratio": float(np.median(geometry)),
        "median_encode_time_ratio": float(np.median(encode)),
        "median_cold_decode_time_ratio": float(np.median(decode)),
        "geometry_ratios": [float(value) for value in geometry],
        "encode_time_ratios": [float(value) for value in encode],
        "cold_decode_time_ratios": [float(value) for value in decode],
    }


def compare_envelopes(
    challenger_rows: Sequence[Mapping[str, Any]],
    reference_rows: Sequence[Mapping[str, Any]],
    *,
    rate_key: str = "total_bytes",
    quality_key: str = "psnr",
    component_rate_key: str = "geometry_attributed_bytes",
    grid_points: int = PSNR_GRID_POINTS,
    minimum_overlap_db: float = MINIMUM_OVERLAP_DB,
) -> dict[str, Any]:
    """Compare two Pareto envelopes by geometric mean rate over common PSNR support."""
    challenger = pareto_envelope(challenger_rows, rate_key=rate_key, quality_key=quality_key)
    reference = pareto_envelope(reference_rows, rate_key=rate_key, quality_key=quality_key)
    quality_grid, overlap = _common_quality_grid(
        [challenger, reference],
        quality_key=quality_key,
        minimum_overlap_db=minimum_overlap_db,
        grid_points=grid_points,
    )
    result: dict[str, Any] = {
        "status": overlap["status"],
        "challenger_envelope": challenger,
        "reference_envelope": reference,
        "common_psnr_overlap": overlap,
        "matched_psnr_shortfall": worst_matched_psnr_shortfall(challenger_rows, reference_rows),
        "envelope_psnr_shortfall": worst_envelope_psnr_shortfall(
            challenger_rows,
            reference_rows,
            rate_key=rate_key,
            quality_key=quality_key,
        ),
        "matched_operating_points": _matched_ratio_diagnostics(challenger_rows, reference_rows),
    }
    if quality_grid is None:
        result["reason"] = overlap["reason"]
        return result

    challenger_log_rate = _interpolate_log_values(
        challenger, quality_grid, value_key=rate_key, quality_key=quality_key
    )
    reference_log_rate = _interpolate_log_values(
        reference, quality_grid, value_key=rate_key, quality_key=quality_key
    )
    area_ratio = math.exp(float(np.mean(challenger_log_rate - reference_log_rate)))
    result.update(
        {
            "quality_grid_psnr_db": [float(value) for value in quality_grid],
            "area_rate_ratio": float(area_ratio),
            "area_rate_reduction": float(1.0 - area_ratio),
            "challenger_geometric_mean_bytes": float(math.exp(float(np.mean(challenger_log_rate)))),
            "reference_geometric_mean_bytes": float(math.exp(float(np.mean(reference_log_rate)))),
        }
    )
    try:
        challenger_component = _interpolate_log_values(
            challenger,
            quality_grid,
            value_key=component_rate_key,
            quality_key=quality_key,
        )
        reference_component = _interpolate_log_values(
            reference,
            quality_grid,
            value_key=component_rate_key,
            quality_key=quality_key,
        )
    except ValueError as exc:
        result["geometry_area"] = {"status": "unavailable", "reason": str(exc)}
    else:
        component_ratio = math.exp(float(np.mean(challenger_component - reference_component)))
        result["geometry_area"] = {
            "status": "available",
            "rate_ratio": float(component_ratio),
            "rate_reduction": float(1.0 - component_ratio),
        }
    return result


def _three_chart_comparison(
    rows_by_chart: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    envelopes = {chart: pareto_envelope(rows_by_chart[chart]) for chart in CHARTS}
    quality_grid, overlap = _common_quality_grid(
        [envelopes[chart] for chart in CHARTS],
        quality_key="psnr",
        minimum_overlap_db=MINIMUM_OVERLAP_DB,
        grid_points=PSNR_GRID_POINTS,
    )
    result: dict[str, Any] = {
        "status": overlap["status"],
        "common_psnr_overlap": overlap,
    }
    if quality_grid is None:
        result["reason"] = overlap["reason"]
        return result
    rates = {
        chart: math.exp(
            float(
                np.mean(
                    _interpolate_log_values(
                        envelopes[chart],
                        quality_grid,
                        value_key="total_bytes",
                        quality_key="psnr",
                    )
                )
            )
        )
        for chart in CHARTS
    }
    current = rates["current_rs"]
    canonical = rates["canonical_rs"]
    log_spd = rates["log_spd"]
    denominator = current - log_spd
    mechanism_fraction = None if denominator <= 0.0 else (current - canonical) / denominator
    result.update(
        {
            "geometric_mean_bytes": {key: float(value) for key, value in rates.items()},
            "canonical_to_current_ratio": float(canonical / current),
            "log_spd_to_current_ratio": float(log_spd / current),
            "log_spd_to_canonical_ratio": float(log_spd / canonical),
            "canonical_fraction_of_log_spd_gain": (
                None if mechanism_fraction is None else float(mechanism_fraction)
            ),
        }
    )
    return result


def paired_bootstrap(
    ratios: Sequence[float],
    *,
    bootstrap_seed: int = BOOTSTRAP_SEED,
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
    indices: np.ndarray | None = None,
) -> dict[str, Any]:
    """Bootstrap image-paired geometric mean rate ratios deterministically."""
    values = np.asarray(ratios, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("ratios must be a non-empty finite vector")
    if np.any(values <= 0.0):
        raise ValueError("rate ratios must be positive")
    if bootstrap_replicates <= 0:
        raise ValueError("bootstrap_replicates must be positive")
    if indices is None:
        rng = np.random.default_rng(bootstrap_seed)
        indices = rng.integers(
            0, values.size, size=(bootstrap_replicates, values.size), dtype=np.int64
        )
    else:
        indices = np.asarray(indices, dtype=np.int64)
        if indices.shape != (bootstrap_replicates, values.size):
            raise ValueError("shared bootstrap indices have the wrong shape")
        if np.any(indices < 0) or np.any(indices >= values.size):
            raise ValueError("shared bootstrap index is out of range")
    log_values = np.log(values)
    samples = np.exp(np.mean(log_values[indices], axis=1))
    lower, upper = np.quantile(samples, [0.025, 0.975])
    return {
        "geometric_mean_ratio": float(math.exp(float(np.mean(log_values)))),
        "ci95": [float(lower), float(upper)],
        "bootstrap_seed": int(bootstrap_seed),
        "bootstrap_replicates": int(bootstrap_replicates),
        "bootstrap_rng": "numpy.PCG64",
        "samples": [float(value) for value in samples],
    }


def _candidate_grid_errors(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    expected = set(expected_allocations())
    errors: list[str] = []
    for chart in CHARTS:
        chart_rows = [row for row in rows if str(row.get("chart")) == chart]
        allocations: set[tuple[int, int, int]] = set()
        for row in chart_rows:
            value = _allocation_value(row)
            if not isinstance(value, list) or len(value) != 3:
                errors.append(f"{chart}: allocation is not an ordered triple")
                continue
            try:
                allocation = tuple(int(item) for item in value)
            except (TypeError, ValueError):
                errors.append(f"{chart}: allocation contains a non-integer")
                continue
            if list(allocation) != value:
                errors.append(f"{chart}: allocation contains a non-integer")
                continue
            if int(row["total_bits"]) != sum(allocation):
                errors.append(f"{chart}: total_bits disagrees with allocation {allocation}")
            allocations.add(allocation)
        if allocations != expected:
            errors.append(
                f"{chart}: exhaustive allocation grid differs "
                f"(missing={len(expected - allocations)}, extra={len(allocations - expected)})"
            )
    return errors


def _gate(value: Any, threshold: Any, comparator: str, passes: bool) -> dict[str, Any]:
    return {
        "value": _json_safe(value),
        "threshold": _json_safe(threshold),
        "comparator": comparator,
        "passes": bool(passes),
    }


def analyze_candidates(
    rows: Sequence[Mapping[str, Any]],
    *,
    bootstrap_seed: int = BOOTSTRAP_SEED,
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
) -> dict[str, Any]:
    """Analyze a complete frozen COMP-007 development candidate matrix.

    The result is JSON-safe and contains per-cell, per-image, and per-coder reductions, all 20,000
    deterministic paired bootstrap samples, a literal gate ledger, and a fail-closed decision.
    """
    if not rows:
        return {
            "status": "unavailable",
            "errors": ["no candidate rows"],
            "per_cell": [],
            "per_image": [],
            "per_coder": {},
            "gates": {},
            "decision": "unavailable",
        }
    if bootstrap_replicates <= 0:
        raise ValueError("bootstrap_replicates must be positive")

    grouped: dict[tuple[str, int, str], list[Mapping[str, Any]]] = defaultdict(list)
    errors: list[str] = []
    cell_ids: dict[tuple[str, int, str], set[str]] = defaultdict(set)
    for row in rows:
        try:
            image_id = str(row["image_id"])
            count = int(row["n"])
            coder = str(row["coder"])
            chart = str(row["chart"])
            cell_id = str(row["cell_id"])
            _positive_float(row, "total_bytes")
            _positive_float(row, "geometry_attributed_bytes")
            _finite_float(row, "psnr")
            _timing_seconds(row, "encode")
            _timing_seconds(row, "decode")
            _operating_point_key(row)
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"invalid candidate row: {exc}")
            continue
        if coder not in CODERS:
            errors.append(f"unexpected coder {coder!r}")
        if chart not in CHARTS:
            errors.append(f"unexpected chart {chart!r}")
        if count not in COUNTS:
            errors.append(f"unexpected count {count}")
        grouped[(image_id, count, coder)].append(row)
        cell_ids[(image_id, count, coder)].add(cell_id)

    image_ids = sorted({key[0] for key in grouped})
    if len(image_ids) != 12:
        errors.append(f"development requires exactly 12 images, found {len(image_ids)}")
    expected_cells = {
        (image_id, count, coder) for image_id in image_ids for count in COUNTS for coder in CODERS
    }
    missing_cells = sorted(expected_cells - set(grouped))
    if missing_cells:
        errors.append(f"missing {len(missing_cells)} image/count/coder cells")

    per_cell: list[dict[str, Any]] = []
    cell_lookup: dict[tuple[str, int, str], dict[str, Any]] = {}
    for key in sorted(grouped):
        image_id, count, coder = key
        cell_rows = grouped[key]
        ids = cell_ids[key]
        if len(ids) != 1:
            errors.append(f"{key}: expected one cell_id, found {sorted(ids)}")
        grid_errors = _candidate_grid_errors(cell_rows)
        errors.extend(f"{key}: {message}" for message in grid_errors)
        rows_by_chart = {
            chart: [row for row in cell_rows if str(row.get("chart")) == chart] for chart in CHARTS
        }
        if any(not rows_by_chart[chart] for chart in CHARTS):
            errors.append(f"{key}: missing one or more charts")
            continue
        try:
            log_vs_canonical = compare_envelopes(
                rows_by_chart["log_spd"], rows_by_chart["canonical_rs"]
            )
            canonical_vs_current = compare_envelopes(
                rows_by_chart["canonical_rs"], rows_by_chart["current_rs"]
            )
            log_vs_current = compare_envelopes(
                rows_by_chart["log_spd"], rows_by_chart["current_rs"]
            )
            mechanism = _three_chart_comparison(rows_by_chart)
        except ValueError as exc:
            errors.append(f"{key}: analysis failed: {exc}")
            continue
        comparisons = {
            "log_spd_vs_canonical_rs": log_vs_canonical,
            "canonical_rs_vs_current_rs": canonical_vs_current,
            "log_spd_vs_current_rs": log_vs_current,
            "current_canonical_log_spd_mechanism": mechanism,
        }
        required_statuses = [
            log_vs_canonical.get("status"),
            canonical_vs_current.get("status"),
            log_vs_current.get("status"),
            mechanism.get("status"),
            log_vs_canonical.get("geometry_area", {}).get("status"),
            log_vs_canonical.get("matched_psnr_shortfall", {}).get("status"),
            log_vs_canonical.get("envelope_psnr_shortfall", {}).get("status"),
            log_vs_canonical.get("matched_operating_points", {}).get("status"),
        ]
        available = all(status == "available" for status in required_statuses)
        if not available:
            errors.append(f"{key}: one or more required comparisons are unavailable")
        cell = {
            "cell_id": sorted(ids)[0] if len(ids) == 1 else sorted(ids),
            "image_id": image_id,
            "n": int(count),
            "coder": coder,
            "status": "available" if available else "unavailable",
            "comparisons": comparisons,
        }
        per_cell.append(cell)
        cell_lookup[key] = cell

    per_image: list[dict[str, Any]] = []
    image_lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for coder in CODERS:
        for image_id in image_ids:
            cells = [cell_lookup.get((image_id, count, coder)) for count in COUNTS]
            if any(cell is None or cell["status"] != "available" for cell in cells):
                errors.append(f"{image_id}/{coder}: unavailable count aggregation")
                continue
            complete_cells = [cell for cell in cells if cell is not None]
            primary = [cell["comparisons"]["log_spd_vs_canonical_rs"] for cell in complete_cells]
            mechanism = [
                cell["comparisons"]["current_canonical_log_spd_mechanism"]
                for cell in complete_cells
            ]
            area_ratios = [float(item["area_rate_ratio"]) for item in primary]
            geometry_ratios = [float(item["geometry_area"]["rate_ratio"]) for item in primary]
            encode_ratios = [
                float(value)
                for item in primary
                for value in item["matched_operating_points"]["encode_time_ratios"]
            ]
            decode_ratios = [
                float(value)
                for item in primary
                for value in item["matched_operating_points"]["cold_decode_time_ratios"]
            ]
            canonical_current = [float(item["canonical_to_current_ratio"]) for item in mechanism]
            log_current = [float(item["log_spd_to_current_ratio"]) for item in mechanism]
            canonical_current_ratio = math.exp(float(np.mean(np.log(canonical_current))))
            log_current_ratio = math.exp(float(np.mean(np.log(log_current))))
            denominator = 1.0 - log_current_ratio
            mechanism_fraction = (
                None if denominator <= 0.0 else (1.0 - canonical_current_ratio) / denominator
            )
            image = {
                "image_id": image_id,
                "coder": coder,
                "count_strata": list(COUNTS),
                "area_rate_ratio": float(math.exp(float(np.mean(np.log(area_ratios))))),
                "geometry_area_rate_ratio": float(
                    math.exp(float(np.mean(np.log(geometry_ratios))))
                ),
                "median_encode_time_ratio": float(np.median(encode_ratios)),
                "median_cold_decode_time_ratio": float(np.median(decode_ratios)),
                "worst_envelope_psnr_shortfall_db": float(
                    max(item["envelope_psnr_shortfall"]["worst_shortfall_db"] for item in primary)
                ),
                "per_n_area_rate_ratio": {
                    str(count): float(item["area_rate_ratio"])
                    for count, item in zip(COUNTS, primary, strict=True)
                },
                "canonical_to_current_ratio": float(canonical_current_ratio),
                "log_spd_to_current_ratio": float(log_current_ratio),
                "canonical_fraction_of_log_spd_gain": (
                    None if mechanism_fraction is None else float(mechanism_fraction)
                ),
            }
            per_image.append(image)
            image_lookup[(image_id, coder)] = image

    per_coder: dict[str, Any] = {}
    if len(image_ids) == 12:
        rng = np.random.default_rng(bootstrap_seed)
        shared_indices = rng.integers(
            0,
            len(image_ids),
            size=(bootstrap_replicates, len(image_ids)),
            dtype=np.int64,
        )
    else:
        shared_indices = None
    for coder in CODERS:
        images = [image_lookup.get((image_id, coder)) for image_id in image_ids]
        if any(image is None for image in images) or shared_indices is None:
            continue
        complete_images = [image for image in images if image is not None]
        ratios = [float(image["area_rate_ratio"]) for image in complete_images]
        bootstrap = paired_bootstrap(
            ratios,
            bootstrap_seed=bootstrap_seed,
            bootstrap_replicates=bootstrap_replicates,
            indices=shared_indices,
        )
        geometry = [float(image["geometry_area_rate_ratio"]) for image in complete_images]
        encode = [float(image["median_encode_time_ratio"]) for image in complete_images]
        decode = [float(image["median_cold_decode_time_ratio"]) for image in complete_images]
        shortfalls = [float(image["worst_envelope_psnr_shortfall_db"]) for image in complete_images]
        canonical_current = [
            float(image["canonical_to_current_ratio"]) for image in complete_images
        ]
        log_current = [float(image["log_spd_to_current_ratio"]) for image in complete_images]
        aggregate_canonical_current = math.exp(float(np.mean(np.log(canonical_current))))
        aggregate_log_current = math.exp(float(np.mean(np.log(log_current))))
        mechanism_denominator = 1.0 - aggregate_log_current
        mechanism_fraction = (
            None
            if mechanism_denominator <= 0.0
            else (1.0 - aggregate_canonical_current) / mechanism_denominator
        )
        per_n = {
            str(count): {
                "median_ratio": float(
                    np.median(
                        [image["per_n_area_rate_ratio"][str(count)] for image in complete_images]
                    )
                ),
                "geometric_mean_ratio": float(
                    math.exp(
                        float(
                            np.mean(
                                np.log(
                                    [
                                        image["per_n_area_rate_ratio"][str(count)]
                                        for image in complete_images
                                    ]
                                )
                            )
                        )
                    )
                ),
            }
            for count in COUNTS
        }
        per_coder[coder] = {
            "image_count": len(complete_images),
            "image_area_rate_ratios": ratios,
            "median_area_rate_ratio": float(np.median(ratios)),
            "median_area_rate_reduction": float(np.median([1.0 - ratio for ratio in ratios])),
            "paired_image_bootstrap": bootstrap,
            "wins_below_one": int(sum(ratio < 1.0 for ratio in ratios)),
            "median_geometry_area_rate_ratio": float(np.median(geometry)),
            "median_geometry_stream_reduction": float(
                np.median([1.0 - ratio for ratio in geometry])
            ),
            "worst_envelope_psnr_shortfall_db": float(max(shortfalls)),
            "median_encode_time_ratio": float(np.median(encode)),
            "median_cold_decode_time_ratio": float(np.median(decode)),
            "per_n": per_n,
            "canonical_to_current_geometric_mean_ratio": float(aggregate_canonical_current),
            "log_spd_to_current_geometric_mean_ratio": float(aggregate_log_current),
            "canonical_fraction_of_log_spd_gain": (
                None if mechanism_fraction is None else float(mechanism_fraction)
            ),
        }

    gates: dict[str, Any] = {}
    if not errors and set(per_coder) == set(CODERS):
        gate_values: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
        for coder in CODERS:
            summary = per_coder[coder]
            gate_values["gate_1_median_whole_container_reduction"][coder] = _gate(
                summary["median_area_rate_reduction"],
                0.01,
                ">=",
                summary["median_area_rate_reduction"] >= 0.01,
            )
            upper = summary["paired_image_bootstrap"]["ci95"][1]
            gate_values["gate_2_bootstrap_upper_rate_ratio"][coder] = _gate(
                upper, 0.995, "<", upper < 0.995
            )
            gate_values["gate_3_image_wins"][coder] = _gate(
                summary["wins_below_one"], 9, ">=", summary["wins_below_one"] >= 9
            )
            gate_values["gate_4_median_covariance_stream_reduction"][coder] = _gate(
                summary["median_geometry_stream_reduction"],
                0.03,
                ">=",
                summary["median_geometry_stream_reduction"] >= 0.03,
            )
            gate_values["gate_5_worst_envelope_psnr_shortfall_db"][coder] = _gate(
                summary["worst_envelope_psnr_shortfall_db"],
                0.05,
                "<=",
                summary["worst_envelope_psnr_shortfall_db"] <= 0.05,
            )
            timing_pass = (
                summary["median_encode_time_ratio"] <= 1.25
                and summary["median_cold_decode_time_ratio"] <= 1.25
            )
            gate_values["gate_6_timing_ratios"][coder] = _gate(
                {
                    "encode": summary["median_encode_time_ratio"],
                    "cold_decode": summary["median_cold_decode_time_ratio"],
                },
                {"encode": 1.25, "cold_decode": 1.25},
                "each <=",
                timing_pass,
            )
            direction_pass = all(
                summary["per_n"][str(count)]["median_ratio"] < 1.0 for count in COUNTS
            )
            gate_values["gate_7_direction_stable_across_n"][coder] = _gate(
                {str(count): summary["per_n"][str(count)]["median_ratio"] for count in COUNTS},
                {str(count): 1.0 for count in COUNTS},
                "each <",
                direction_pass,
            )
            fraction = summary["canonical_fraction_of_log_spd_gain"]
            gate_values["gate_8_canonical_mechanism_fraction"][coder] = _gate(
                fraction,
                0.75,
                "<",
                fraction is not None and fraction < 0.75,
            )
        gates = {
            name: {
                "per_coder": by_coder,
                "passes": all(item["passes"] for item in by_coder.values()),
            }
            for name, by_coder in gate_values.items()
        }

    availability = not errors and len(per_cell) == 12 * len(COUNTS) * len(CODERS)
    if not availability:
        decision = "unavailable"
    elif gates and all(gate["passes"] for gate in gates.values()):
        decision = "pass_authorize_confirmation"
    else:
        decision = "kill"
    return _json_safe(
        {
            "status": "available" if availability else "unavailable",
            "protocol": {
                "charts": list(CHARTS),
                "coders": list(CODERS),
                "counts": list(COUNTS),
                "psnr_grid_points": PSNR_GRID_POINTS,
                "minimum_overlap_db": MINIMUM_OVERLAP_DB,
                "bootstrap_seed": int(bootstrap_seed),
                "bootstrap_replicates": int(bootstrap_replicates),
                "independent_unit": "image",
            },
            "errors": sorted(set(errors)),
            "per_cell": per_cell,
            "per_image": per_image,
            "per_coder": per_coder,
            "gates": gates,
            "decision": decision,
            "confirmation_authorized": decision == "pass_authorize_confirmation",
        }
    )


# Short alias for callers that treat analysis modules uniformly.
analyze = analyze_candidates
