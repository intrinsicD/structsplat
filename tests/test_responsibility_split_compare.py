# ruff: noqa: E402
from pathlib import Path

import pytest

pytest.importorskip("torch")

from benchmarks.responsibility_split_compare import (
    ARMS,
    SOURCE_PATHS,
    _aggregate,
    _decision,
    _source_provenance,
)


def _rows():
    rows = []
    values = {
        "residual": (20.0, 21.0, 22.0, 10.0),
        "support": (20.1, 21.1, 22.1, 10.0),
        "responsibility_alpha1": (20.15, 21.15, 22.1, 10.4),
        "responsibility_alpha0.7": (20.3, 21.25, 22.08, 11.0),
    }
    for image_idx in range(4):
        for seed in (0, 1):
            for arm in ARMS:
                immediate, post20, post100, elapsed = values[arm]
                rows.append({
                    "image": f"image{image_idx}",
                    "seed": seed,
                    "arm": arm,
                    "immediate_psnr": immediate,
                    "post20_psnr": post20,
                    "post100_psnr": post100,
                    "score_seconds": 0.1,
                    "total100_seconds": elapsed,
                    "parent_jaccard_vs_residual": 0.5,
                    "parent_jaccard_vs_support": 0.6,
                    "final_count": 80,
                    "scores_finite": True,
                    "renders_finite": True,
                })
    return rows


def test_fit018_guard_uses_stronger_existing_arm_and_exact_preregistered_gates():
    rows = _rows()
    aggregate = _aggregate(rows)
    decision = _decision(rows, aggregate)

    comparison = aggregate["comparison"]
    assert comparison["stronger_existing_arm"] == "support"
    assert comparison["delta_post20_psnr"] == pytest.approx(0.15)
    assert comparison["post20_positive_count"] == 8
    assert comparison["delta_post100_psnr"] == pytest.approx(-0.02)
    assert comparison["total100_overhead_fraction"] == pytest.approx(0.1)
    assert decision["guard_survives"] is True


def test_fit018_guard_fails_nonfinite_or_wrong_count_rows():
    rows = _rows()
    rows[0]["scores_finite"] = False
    rows[1]["final_count"] = 79

    decision = _decision(rows, _aggregate(rows))

    assert decision["guard_survives"] is False
    assert decision["checks"]["all_scores_finite"] is False
    assert decision["checks"]["all_final_counts_eq_80"] is False


def test_fit018_source_provenance_hashes_untracked_benchmark_content():
    root = Path(__file__).resolve().parents[1]
    source = _source_provenance(root)

    assert set(source["files"]) == set(SOURCE_PATHS)
    assert all(len(value) == 64 for value in source["files"].values())
    assert len(source["combined_sha256"]) == 64
