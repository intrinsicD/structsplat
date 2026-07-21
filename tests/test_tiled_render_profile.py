# ruff: noqa: E402
import math

import numpy as np
import pytest

pytest.importorskip("torch")

from benchmarks.tiled_render_profile import (
    PREREGISTERED_GATE,
    CellSpec,
    build_cells,
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
