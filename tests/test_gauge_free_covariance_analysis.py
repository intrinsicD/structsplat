import json
import math

import numpy as np
import pytest

from benchmarks import gauge_free_covariance_analysis as A


def _point(
    *,
    total_bytes: float,
    psnr: float,
    geometry_bytes: float | None = None,
    total_bits: int = 12,
    allocation: tuple[int, int, int] = (4, 4, 4),
) -> dict[str, object]:
    return {
        "total_bytes": total_bytes,
        "geometry_attributed_bytes": geometry_bytes or 0.5 * total_bytes,
        "psnr": psnr,
        "total_bits": total_bits,
        "allocation": allocation,
        "encode_seconds": 1.0,
        "cold_decode_seconds": 1.0,
    }


def test_pareto_envelope_is_stable_and_strictly_nondominated() -> None:
    rows = [
        {"name": "first", "total_bytes": 100, "psnr": 20.0},
        {"name": "duplicate", "total_bytes": 100, "psnr": 20.0},
        {"name": "same-rate-better", "total_bytes": 100, "psnr": 20.5},
        {"name": "dominated", "total_bytes": 140, "psnr": 20.4},
        {"name": "next", "total_bytes": 180, "psnr": 21.0},
        {"name": "same-quality-worse-rate", "total_bytes": 200, "psnr": 21.0},
    ]
    envelope = A.pareto_envelope(rows)
    assert [row["name"] for row in envelope] == ["same-rate-better", "next"]
    assert [row["total_bytes"] for row in envelope] == [100, 180]


def test_common_support_area_ratio_and_exact_operating_point_diagnostic() -> None:
    reference = [
        _point(total_bytes=100, geometry_bytes=50, psnr=20.0, total_bits=12),
        _point(
            total_bytes=200,
            geometry_bytes=100,
            psnr=21.0,
            total_bits=18,
            allocation=(6, 6, 6),
        ),
        _point(
            total_bytes=400,
            geometry_bytes=200,
            psnr=22.0,
            total_bits=24,
            allocation=(8, 8, 8),
        ),
    ]
    challenger = [
        {
            **row,
            "total_bytes": 0.9 * float(row["total_bytes"]),
            "geometry_attributed_bytes": 0.8 * float(row["geometry_attributed_bytes"]),
        }
        for row in reference
    ]
    result = A.compare_envelopes(challenger, reference)
    assert result["status"] == "available"
    assert result["area_rate_ratio"] == pytest.approx(0.9)
    assert result["geometry_area"]["rate_ratio"] == pytest.approx(0.8)
    assert len(result["quality_grid_psnr_db"]) == 25
    assert result["matched_psnr_shortfall"]["worst_shortfall_db"] == pytest.approx(0.0)
    assert result["envelope_psnr_shortfall"]["worst_shortfall_db"] < 0.0


def test_comparison_refuses_short_common_psnr_support() -> None:
    left = [_point(total_bytes=100, psnr=20.00), _point(total_bytes=110, psnr=20.05)]
    right = [_point(total_bytes=120, psnr=20.02), _point(total_bytes=130, psnr=20.09)]
    result = A.compare_envelopes(left, right)
    assert result["status"] == "unavailable"
    assert result["reason"] == "insufficient_common_psnr_overlap"
    assert result["common_psnr_overlap"]["width_db"] == pytest.approx(0.03)


def test_matched_shortfall_requires_the_same_allocation_grid() -> None:
    left = [_point(total_bytes=90, psnr=20.0)]
    right = [_point(total_bytes=100, psnr=20.0, allocation=(3, 4, 5))]
    result = A.worst_matched_psnr_shortfall(left, right)
    assert result["status"] == "unavailable"
    assert result["reason"] == "mismatched_operating_point_grid"


def test_envelope_shortfall_is_measured_at_matched_complete_rate() -> None:
    reference = [
        _point(total_bytes=100, psnr=20.0),
        _point(total_bytes=200, psnr=22.0, total_bits=18, allocation=(6, 6, 6)),
    ]
    challenger = [
        _point(total_bytes=100, psnr=20.0),
        _point(total_bytes=200, psnr=21.8, total_bits=18, allocation=(3, 7, 8)),
    ]
    result = A.worst_envelope_psnr_shortfall(challenger, reference)
    assert result["status"] == "available"
    assert result["worst_shortfall_db"] == pytest.approx(0.2)
    assert result["worst_operating_point"]["matched_bytes"] == pytest.approx(200.0)


def test_paired_bootstrap_is_deterministic_and_geometric() -> None:
    first = A.paired_bootstrap([0.8, 1.0, 1.2], bootstrap_seed=9, bootstrap_replicates=100)
    second = A.paired_bootstrap([0.8, 1.0, 1.2], bootstrap_seed=9, bootstrap_replicates=100)
    assert first == second
    assert first["geometric_mean_ratio"] == pytest.approx((0.8 * 1.0 * 1.2) ** (1 / 3))
    assert len(first["samples"]) == 100


def _development_rows(
    *,
    canonical_fraction_failure: bool = False,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    chart_factors = {
        "current_rs": 1.0,
        "canonical_rs": 0.90 if canonical_fraction_failure else 0.95,
        "log_spd": 0.88 if canonical_fraction_failure else 0.85,
    }
    geometry_factors = {"current_rs": 1.0, "canonical_rs": 0.95, "log_spd": 0.80}
    for coder_index, coder in enumerate(A.CODERS):
        for image_index in range(12):
            image_id = f"{2 * image_index + 2:02d}"
            for n in A.COUNTS:
                base_scale = (1.0 + 0.03 * coder_index) * (n / 640)
                for chart in A.CHARTS:
                    for allocation in A.expected_allocations():
                        total_bits = sum(allocation)
                        # The allocation-specific term creates a large strict envelope while all
                        # chart rate ratios remain analytically constant.
                        progress = (total_bits - 12) / 12 + 0.001 * (
                            allocation[0] + 2 * allocation[1] + 3 * allocation[2]
                        )
                        base_bytes = base_scale * (1000.0 * math.exp(0.6 * progress))
                        psnr = 20.0 + progress
                        rows.append(
                            {
                                "cell_id": f"kodak-{image_id}-n{n}",
                                "image_id": image_id,
                                "n": n,
                                "coder": coder,
                                "chart": chart,
                                "total_bits": total_bits,
                                "allocation": allocation,
                                "total_bytes": base_bytes * chart_factors[chart],
                                "geometry_attributed_bytes": (
                                    0.4 * base_bytes * geometry_factors[chart]
                                ),
                                "psnr": psnr,
                                "encode_seconds": 1.1 if chart == "log_spd" else 1.0,
                                "cold_decode_seconds": 1.1 if chart == "log_spd" else 1.0,
                            }
                        )
    return rows


def test_complete_development_analysis_passes_all_frozen_gates_and_is_json_safe() -> None:
    result = A.analyze_candidates(_development_rows(), bootstrap_seed=7, bootstrap_replicates=200)
    assert result["status"] == "available"
    assert result["decision"] == "pass_authorize_confirmation"
    assert result["confirmation_authorized"] is True
    assert len(result["per_cell"]) == 48
    assert len(result["per_image"]) == 24
    assert all(gate["passes"] for gate in result["gates"].values())
    for coder in A.CODERS:
        summary = result["per_coder"][coder]
        assert summary["wins_below_one"] == 12
        assert summary["median_area_rate_ratio"] == pytest.approx(0.85 / 0.95)
        assert summary["canonical_fraction_of_log_spd_gain"] == pytest.approx(1 / 3)
        assert len(summary["paired_image_bootstrap"]["samples"]) == 200
    json.dumps(result, allow_nan=False)


def test_mechanism_gate_rejects_when_canonical_explains_too_much_gain() -> None:
    result = A.analyze_candidates(
        _development_rows(canonical_fraction_failure=True),
        bootstrap_seed=11,
        bootstrap_replicates=100,
    )
    assert result["status"] == "available"
    assert result["decision"] == "kill"
    gate = result["gates"]["gate_8_canonical_mechanism_fraction"]
    assert gate["passes"] is False
    assert all(item["value"] > 0.75 for item in gate["per_coder"].values())


def test_analysis_fails_closed_on_incomplete_primary_matrix() -> None:
    rows = _development_rows()
    result = A.analyze_candidates(rows[:-1], bootstrap_replicates=20)
    assert result["status"] == "unavailable"
    assert result["decision"] == "unavailable"
    assert result["confirmation_authorized"] is False
    assert result["errors"]


def test_expected_grid_has_only_legal_exhaustive_allocations() -> None:
    allocations = A.expected_allocations()
    assert len(allocations) == 84
    assert all(sum(allocation) in A.TOTAL_BITS for allocation in allocations)
    assert all(A.BITS_MIN <= bit <= A.BITS_MAX for allocation in allocations for bit in allocation)
    assert len(set(allocations)) == len(allocations)


def test_invalid_rate_and_bootstrap_inputs_are_rejected() -> None:
    with pytest.raises(ValueError):
        A.pareto_envelope([{"total_bytes": 0, "psnr": 20.0}])
    with pytest.raises(ValueError):
        A.paired_bootstrap([1.0, np.nan], bootstrap_replicates=10)
