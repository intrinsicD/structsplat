# ruff: noqa: E402
import hashlib
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from benchmarks.exact_backward_profile import (
    PARITY_ATOL,
    PREREGISTERED_GATE,
    CellSpec,
    _strict_json,
    actual_support_statistics,
    attach_pair_ratios,
    collect_provenance,
    evaluate_preregistered_gate,
    inspect_exact_backward_source,
    make_field_arrays,
    profile_grid,
    requested_scale,
    write_outputs,
)


ROOT = Path(__file__).resolve().parents[1]


def test_preregistered_gate_is_frozen_and_requires_all_registered_checks():
    assert PREREGISTERED_GATE["registered_before_optimized_measurement"] is True
    assert PREREGISTERED_GATE["minimum_exact_backward_reduction_fraction"] == 0.10
    assert PREREGISTERED_GATE["minimum_representative_step_reduction_fraction"] == 0.03
    assert PREREGISTERED_GATE["minimum_grid_direction_reduction_fraction"] == 0.0
    assert (
        PREREGISTERED_GATE[
            "maximum_peak_incremental_allocator_memory_regression_fraction"
        ]
        == 0.05
    )
    assert PREREGISTERED_GATE["maximum_representative_timing_cv"] == 0.05
    assert PREREGISTERED_GATE["default_flip_authorized"] is False
    assert PREREGISTERED_GATE["quality_or_compression_claim_authorized"] is False
    cell = PREREGISTERED_GATE["representative_cell"]
    rows = []
    for resolution in (128, 256):
        for count in (512, 2048):
            for overlap in (4.0, 16.0):
                common = {
                    "height": resolution,
                    "width": resolution,
                    "n_gaussians": count,
                    "requested_support_overlap": overlap,
                    "status": "ok",
                    "exact_backward_cv": 0.02,
                    "representative_fit_step_cv": 0.02,
                }
                rows.extend([
                    {
                        **common,
                        "mode": "cuda",
                        "exact_backward_median_ms": 10.0,
                        "representative_fit_step_median_ms": 20.0,
                        "peak_incremental_allocated_bytes": 1000,
                    },
                    {
                        **common,
                        "mode": "cuda_block_reduce",
                        "exact_backward_median_ms": 8.9,
                        "representative_fit_step_median_ms": 19.2,
                        "peak_incremental_allocated_bytes": 1040,
                    },
                ])
    candidate = next(
        row for row in rows
        if row["mode"] == "cuda_block_reduce"
        and all(row[key] == value for key, value in cell.items())
    )
    structure = {"comparison_ready": True}
    parity = [{"passed": True}]
    calibration = {
        "reference_vs_cuda": {"passed": True},
        "reference_vs_cuda_block_reduce": {"passed": True},
    }

    passed = evaluate_preregistered_gate(
        rows, structure, parity, calibration
    )
    assert passed["pass"] is True
    assert passed["microprofile_speedup_claim_authorized"] is True
    assert passed["default_flip_authorized"] is False

    failing_cases = (
        (
            [
                {**row, "exact_backward_median_ms": 9.01}
                if row is candidate else row
                for row in rows
            ],
            structure,
            parity,
        ),
        (
            [
                {**row, "representative_fit_step_median_ms": 19.5}
                if row is candidate else row
                for row in rows
            ],
            structure,
            parity,
        ),
        (
            [
                {**row, "peak_incremental_allocated_bytes": 1060}
                if row is candidate else row
                for row in rows
            ],
            structure,
            parity,
        ),
        (
            [
                {**row, "exact_backward_cv": 0.051}
                if row is candidate else row
                for row in rows
            ],
            structure,
            parity,
        ),
        (rows, {"comparison_ready": False}, parity),
        (rows, structure, [{"passed": False}]),
        (
            [
                {
                    **row,
                    "exact_backward_median_ms": 10.1,
                }
                if (
                    row["mode"] == "cuda_block_reduce"
                    and row["height"] == 128
                    and row["n_gaussians"] == 512
                    and row["requested_support_overlap"] == 4.0
                )
                else row
                for row in rows
            ],
            structure,
            parity,
        ),
    )
    for rows, candidate_structure, candidate_parity in failing_cases:
        failed = evaluate_preregistered_gate(
            rows, candidate_structure, candidate_parity, calibration
        )
        assert failed["pass"] is False
        assert failed["microprofile_speedup_claim_authorized"] is False
        assert failed["default_flip_authorized"] is False


def test_live_cuda_source_has_baseline_atomic_and_candidate_reduction_patterns():
    source = (ROOT / "src/structsplat/cuda/render_ext.cu").read_text()
    evidence = inspect_exact_backward_source(source)
    baseline = evidence["baseline"]
    candidate = evidence["block_reduce"]

    assert baseline["one_block_per_gaussian"] is True
    assert baseline["threads_per_block"] == 256
    assert baseline["thread_local_pixel_loop"] is True
    assert baseline["atomic_callsite_counts_by_gradient"] == {
        "colors": 3,
        "conics": 3,
        "means": 2,
        "opacities": 1,
    }
    assert baseline["same_gaussian_gradient_addresses_across_threads"] is True
    assert baseline["actionable_within_block_reduction_pattern"] is True
    assert candidate["one_block_per_gaussian"] is True
    assert candidate["threads_per_block"] == 256
    assert candidate["thread_local_pixel_loop"] is True
    assert candidate["uses_warp_shuffle_reduction"] is True
    assert candidate["uses_shared_warp_sums"] is True
    assert candidate["atomic_callsite_count"] == 0
    assert candidate["direct_gradient_component_writes"] is True
    assert candidate["implemented_pattern_matches_experiment"] is True
    assert evidence["comparison_ready"] is True
    assert evidence["evidence_kind"].endswith("not_hardware_counter")


def test_requested_overlap_controls_deterministic_support_work():
    low = CellSpec(128, 128, 256, 4.0, 9)
    high = CellSpec(128, 128, 256, 16.0, 9)
    assert requested_scale(high) > requested_scale(low)

    low_arrays = make_field_arrays(low)
    low_again = make_field_arrays(low)
    high_arrays = make_field_arrays(high)
    assert all((low_arrays[key] == low_again[key]).all() for key in low_arrays)
    low_support = actual_support_statistics(low, low_arrays)
    high_support = actual_support_statistics(high, high_arrays)
    assert high_support["actual_support_overlap"] > low_support["actual_support_overlap"]
    assert low_support["support_aabb_visits"] > 0
    assert "proxy" in low_support["evidence_kind"]


def test_block_reduce_ratios_are_emitted_only_after_semantic_parity():
    base = {
        "height": 64,
        "width": 64,
        "n_gaussians": 32,
        "requested_support_overlap": 4.0,
        "status": "ok",
    }
    rows = [
        {
            **base,
            "mode": "cuda",
            "exact_backward_median_ms": 2.0,
            "representative_fit_step_median_ms": 5.0,
            "peak_incremental_allocated_bytes": 1000,
        },
        {
            **base,
            "mode": "cuda_block_reduce",
            "exact_backward_median_ms": 1.5,
            "representative_fit_step_median_ms": 4.5,
            "peak_incremental_allocated_bytes": 1040,
        },
    ]
    key = (64, 64, 32, 4.0)
    attach_pair_ratios(rows, {key: {"passed": True}})
    assert rows[0]["candidate_over_baseline_exact_backward_time_ratio"] == 0.75
    assert rows[0]["exact_backward_reduction_fraction"] == 0.25
    assert rows[1]["candidate_over_baseline_representative_step_time_ratio"] == 0.9
    assert rows[1]["representative_step_reduction_fraction"] == pytest.approx(0.1)
    assert rows[1]["peak_incremental_allocator_memory_regression_fraction"] == pytest.approx(
        0.04
    )
    assert rows[1]["microprofile_speedup_claim_authorized"] is False

    failed_rows = [dict(row) for row in rows]
    attach_pair_ratios(failed_rows, {key: {"passed": False}})
    assert all(
        row["candidate_over_baseline_exact_backward_time_ratio"] is None
        for row in failed_rows
    )


def test_provenance_hashes_exact_worktree_sources_even_when_untracked():
    provenance = collect_provenance(ROOT)
    benchmark = ROOT / "benchmarks/exact_backward_profile.py"
    expected = hashlib.sha256(benchmark.read_bytes()).hexdigest()

    assert len(provenance["git_commit"]) == 40
    assert isinstance(provenance["git_dirty"], bool)
    assert len(provenance["tracked_diff_sha256"]) == 64
    assert provenance["critical_source_hash_mode"].endswith("independent_of_git_index")
    assert provenance["critical_sources"][str(benchmark.relative_to(ROOT))]["sha256"] == expected
    assert "src/structsplat/cuda/render_ext.cu" in provenance["critical_sources"]


def test_output_writer_is_strict_json_and_preserves_scope(tmp_path):
    row = {
        "height": 64,
        "width": 64,
        "n_gaussians": 32,
        "requested_support_overlap": 4.0,
        "actual_support_overlap": 4.2,
        "mode": "cuda",
        "exact_forward_median_ms": 1.0,
        "exact_backward_median_ms": 2.0,
        "fit_backward_median_ms": 2.5,
        "optimizer_update_median_ms": 0.5,
        "representative_fit_step_median_ms": 5.0,
        "exact_backward_share_of_representative_step": 0.4,
        "exact_backward_cv": 0.02,
        "representative_fit_step_cv": 0.03,
        "peak_incremental_allocated_bytes": 1024,
    }
    result = {
        "protocol": "test",
        "work_scope": "cuda_event_only",
        "timing_method": "events",
        "memory_method": "allocator",
        "loss_scope": "test",
        "optimizer_scope": "test",
        "grid": {},
        "preregistered_gate": PREREGISTERED_GATE,
        "environment": {"device_name": "test GPU"},
        "provenance": {},
        "ncu_evidence": {"status": "unavailable_test", "collected": False},
        "decision": {
            "status": "keep_benchmark_only_optimization_gate_failed",
            "pass": False,
            "microprofile_speedup_claim_authorized": False,
        },
        "rows": [row],
        "raw_samples": [],
        "limitations": ["bounded"],
    }
    write_outputs(result, tmp_path)

    assert {path.name for path in tmp_path.iterdir()} == {
        "aggregate.json",
        "config.json",
        "ncu.json",
        "rows.csv",
        "rows.json",
        "samples.csv",
        "summary.md",
    }
    for name in ("aggregate.json", "config.json", "ncu.json", "rows.json"):
        text = (tmp_path / name).read_text()
        assert "Infinity" not in text and "NaN" not in text
        json.loads(text)
    with pytest.raises(ValueError):
        _strict_json({"invalid": float("inf")})
    assert (
        "Source/device-bound microprofile speedup claim authorized: **no**"
        in (tmp_path / "summary.md").read_text()
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_tiny_gpu_profile_records_separated_phases_parity_and_raw_samples():
    result = profile_grid(
        resolutions=(32,),
        counts=(32,),
        overlaps=(2.0,),
        warmup=1,
        repeats=3,
        seed=31,
        device_index=0,
    )

    assert len(result["rows"]) == 2
    assert len(result["raw_samples"]) == 2 * 5 * 3
    assert result["reference_parity_calibration"]["reference_vs_cuda"]["passed"] is True
    assert (
        result["reference_parity_calibration"]["reference_vs_cuda_block_reduce"][
            "passed"
        ]
        is True
    )
    assert result["cell_parity"][0]["passed"] is True
    assert result["cell_parity"][0]["gradient_max_abs"] <= PARITY_ATOL
    assert {row["mode"] for row in result["rows"]} == {"cuda", "cuda_block_reduce"}
    for row in result["rows"]:
        assert row["work_scope"].endswith("device_timeline_only")
        assert row["exact_forward_median_ms"] > 0.0
        assert row["exact_backward_median_ms"] > 0.0
        assert row["exact_backward_cv"] >= 0.0
        assert row["optimizer_update_median_ms"] > 0.0
        assert row["representative_fit_step_median_ms"] > 0.0
        assert row["representative_fit_step_cv"] >= 0.0
        assert row["peak_allocated_bytes"] >= row["baseline_allocated_bytes"]
        assert row["microprofile_speedup_claim_authorized"] is False
