import pytest

from benchmarks.wse_prefix_audit import _aggregate


def test_wse_prefix_audit_aggregate_applies_joint_quality_and_runtime_gates():
    rows = []
    for seed in (0, 1):
        for prefix in (4, 8):
            rows.extend([
                {
                    "seed": seed,
                    "prefix_count": prefix,
                    "order": "candidate_index",
                    "normalized_min_spacing": 0.5,
                    "inverse_normalized_coverage_hole": 0.5,
                },
                {
                    "seed": seed,
                    "prefix_count": prefix,
                    "order": "progressive",
                    "normalized_min_spacing": 0.7,
                    "inverse_normalized_coverage_hole": 0.8,
                },
            ])
    timings = [
        {
            "selection_seconds": 1.0,
            "order_seconds": 0.2,
            "terminal_sets_identical": True,
        },
        {
            "selection_seconds": 1.0,
            "order_seconds": 0.2,
            "terminal_sets_identical": True,
        },
    ]

    summary, decision = _aggregate(rows, timings)

    assert summary["4"]["progressive"]["delta_normalized_min_spacing"] \
        == pytest.approx(0.2)
    assert decision["joint_pair_win_fraction"] == 1.0
    assert decision["ordering_overhead_fraction"] == pytest.approx(0.2)
    assert decision["accepted"] is True
