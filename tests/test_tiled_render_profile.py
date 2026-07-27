# ruff: noqa: E402
import math

import numpy as np
import pytest

pytest.importorskip("torch")

import torch

from benchmarks.tiled_render_profile import (
    PREREGISTERED_GATE,
    CellSpec,
    build_cells,
    classify_arm_parity,
    evaluate_gate,
    make_field_arrays,
    requested_scale,
    summarize_rows,
)


def test_build_cells_covers_full_grid_including_representative_cell():
    cells = build_cells((256, 512), (2048, 8192), (4.0, 16.0), (1.0, 6.0), seed=0)
    assert len(cells) == 16
    rep = PREREGISTERED_GATE["representative_cell"]
    assert any(
        c.height == rep["height"]
        and c.n_gaussians == rep["n_gaussians"]
        and c.requested_support_overlap == rep["requested_support_overlap"]
        and c.axis_ratio == rep["axis_ratio"]
        for c in cells
    )


def test_make_field_arrays_is_deterministic_and_ratio_preserves_area():
    iso = CellSpec(128, 128, 256, 8.0, 1.0, seed=3)
    aniso = CellSpec(128, 128, 256, 8.0, 9.0, seed=3)
    a = make_field_arrays(iso)
    b = make_field_arrays(iso)
    for key in a:
        np.testing.assert_array_equal(a[key], b[key])
    stretched = make_field_arrays(aniso)
    # sqrt(ratio) stretch on sx and shrink on sy keeps the scale product (support area) fixed.
    np.testing.assert_allclose(
        stretched["scales"][:, 0] * stretched["scales"][:, 1],
        a["scales"][:, 0] * a["scales"][:, 1],
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        stretched["scales"][:, 0] / stretched["scales"][:, 1],
        9.0 * (a["scales"][:, 0] / a["scales"][:, 1]),
        rtol=1e-5,
    )
    assert requested_scale(iso) > 0.0


def _row(arm, height, n, overlap, ratio, step_ms, index_ms=None, step_cv=0.01):
    return {
        "height": height,
        "width": height,
        "n_gaussians": n,
        "requested_support_overlap": overlap,
        "axis_ratio": ratio,
        "seed": 0,
        "arm": arm,
        "index_ms": index_ms,
        "forward_ms": 1.0,
        "backward_ms": 2.0,
        "step_ms": step_ms,
        "step_cv": step_cv,
        "backward_cv": 0.01,
    }


def _paired_rows(candidate_step, base_step=4.0, index_ms=0.1):
    rep = PREREGISTERED_GATE["representative_cell"]
    rows = []
    for ratio in (1.0, rep["axis_ratio"]):
        rows.append(_row("cuda", rep["height"], rep["n_gaussians"],
                         rep["requested_support_overlap"], ratio, base_step))
        rows.append(_row("cuda_tiled_gpu_index_cull", rep["height"], rep["n_gaussians"],
                         rep["requested_support_overlap"], ratio, candidate_step,
                         index_ms=index_ms))
    return rows


def test_evaluate_gate_passes_when_candidate_beats_exact_everywhere():
    result = evaluate_gate(_paired_rows(candidate_step=3.0), parity_failures=[])
    assert result["pass"] is True
    assert result["checks"]["representative_step_ratio"] == pytest.approx(0.75)


def test_evaluate_gate_fails_on_parity_slow_step_or_index_share():
    assert evaluate_gate(_paired_rows(3.0), parity_failures=[{"arm": "x"}])["pass"] is False
    assert evaluate_gate(_paired_rows(5.0), parity_failures=[])["pass"] is False
    heavy_index = _paired_rows(3.0, index_ms=1.0)  # > 15% of the 3.0 ms step
    assert evaluate_gate(heavy_index, parity_failures=[])["pass"] is False


def test_summarize_rows_renders_missing_index_as_dash():
    text = summarize_rows([_row("cuda", 256, 2048, 4.0, 1.0, 2.5)])
    assert "| - |" in text
    assert "| cuda |" in text
    assert math.isfinite(2.5)


def _flat(values):
    return torch.tensor(values, dtype=torch.float32)


def test_adr0024_candidate_diverging_from_baseline_gates():
    """A tiled arm that disagrees with exact `cuda` fails, regardless of the reference."""
    ref = _flat([0.5, 0.5, 0.5])
    baseline = _flat([0.5, 0.5, 0.5])
    got = _flat([0.5, 0.5, 0.9])  # candidate's own divergence
    rec = classify_arm_parity("cuda_tiled_gpu_index_cull", got, ref, baseline, True)
    assert rec["gating"] == "candidate_vs_baseline"
    assert rec["reference_diagnostic"] is False


def test_adr0024_reference_mismatch_gates_when_baseline_is_clean():
    """If the baseline matches the reference and the candidate does not, that is a regression."""
    ref = _flat([0.5, 0.5, 0.5])
    baseline = _flat([0.5, 0.5, 0.5])
    # Within candidate-vs-baseline tolerance is impossible here by construction, so drive the
    # baseline-clean branch through the baseline arm itself.
    rec = classify_arm_parity("cuda", _flat([0.5, 0.5, 0.9]), ref, baseline, True)
    assert rec["gating"] == "reference_regression_not_in_baseline"


def test_adr0024_baseline_attributable_mismatch_is_reported_not_gated():
    """The 2026-07-24 case: every arm inherits one bad pixel from the unmodified baseline."""
    ref = _flat([0.5, 0.5, 0.5])
    baseline = _flat([0.5, 0.5, 0.5 + 8.9e-4])  # baseline itself misses the reference
    got = _flat([0.5, 0.5, 0.5 + 8.9e-4 + 1.8e-7])  # candidate tracks the baseline closely
    rec = classify_arm_parity("cuda_tiled_gpu_index_cull", got, ref, baseline, False)
    assert rec["gating"] is None
    assert rec["reference_diagnostic"] is True
    assert rec["values_over_reference_tol"] == 1
    assert rec["max_abs_vs_baseline"] < 1e-6


def test_adr0024_clean_arm_is_neither_gated_nor_reported():
    ref = _flat([0.25, 0.75])
    rec = classify_arm_parity("cuda_tiled_gpu_index_cull", ref.clone(), ref, ref.clone(), True)
    assert rec["gating"] is None
    assert rec["reference_diagnostic"] is False


def test_evaluate_gate_reports_baseline_attributable_diagnostics():
    diag = [{"cell": {"height": 512, "n_gaussians": 8192,
                      "requested_support_overlap": 4.0, "axis_ratio": 1.0},
             "arm": "cuda_tiled_gpu_index_cull", "max_abs_vs_reference": 8.877516e-4,
             "values_over_reference_tol": 1, "values_total": 786432,
             "max_abs_vs_baseline": 1.788139e-7}]
    result = evaluate_gate(_paired_rows(candidate_step=3.0), parity_failures=[],
                           reference_diagnostics=diag)
    assert result["pass"] is True
    assert result["checks"]["baseline_attributable_reference_cells"] == 1
    assert result["checks"]["baseline_attributable_reference_detail"][0]["arm"] == (
        "cuda_tiled_gpu_index_cull")
