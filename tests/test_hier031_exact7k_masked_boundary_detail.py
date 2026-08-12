from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from scripts.experiments import hier031_exact7k_masked_boundary_detail as h31
from scripts.experiments import hier031_finalize_objective_views as finalize_views


def test_feasibility_separates_count_independent_and_micro_repairable_holes():
    inside = np.zeros((14, 14), dtype=bool)
    inside[2:9, 2:9] = True
    inside[12, 12] = True

    record, geometry = h31._feasibility_audit(inside)

    assert record["components"] == 2
    assert record["centreless_components"] == 1
    assert record["centreless_component_pixels"] == 1
    assert record["ordinary_admissible_centres"] > 0
    assert record["isotropic_unreachable_pixels"] >= 1
    assert record["micro_all_mask_centres_certified"] is True
    assert geometry["isotropic_unreachable"][12, 12]


def test_coverage_metrics_report_true_positive_minimum_without_raw_holes():
    inside = np.zeros((7, 7), dtype=bool)
    inside[1:6, 1:6] = True
    sdf = np.where(inside, 2.0, -1.0)
    ridge = np.zeros_like(inside)
    ridge[3, 3] = True
    unreachable = np.zeros_like(inside)
    coverage = np.zeros((7, 7), dtype=np.float32)
    coverage[inside] = 0.125
    coverage[3, 3] = 0.5

    metrics = h31._coverage_metrics(coverage, inside, sdf, ridge, unreachable)

    assert metrics["raw_hole_pixels"] == 0
    assert metrics["coverage_lt_005_pixels"] == 0
    assert metrics["coverage_inside_min"] == pytest.approx(0.125)
    assert metrics["coverage_inside_max"] == pytest.approx(0.5)


def test_final_arm_matrix_keeps_exact_count_pipeline_controls_last():
    assert h31.CAPACITY == 7_000
    assert h31.ARMS[-3:] == (
        "deep_only_terminal_closure_n7000",
        "pipeline_fixed_n7000",
        "pipeline_boundary_recycle_n7000",
    )
    assert h31.ADDITIVE_ARMS == frozenset(h31.ARMS[:-2])


def test_presentation_finalizer_preserves_measurements_and_fields(tmp_path, monkeypatch):
    root = tmp_path / "bundle"
    root.mkdir()
    (root / "COMPLETED").write_text("complete\n", encoding="utf-8")
    (root / "index.html").write_text("old index", encoding="utf-8")
    (root / "manifest.json").write_text("{}", encoding="utf-8")
    rows = []
    target = np.zeros((4, 4, 3), dtype=np.uint8)
    target[1:3, 1:3] = 160
    reconstruction = target.copy()
    error = np.zeros_like(target)
    for arm in h31.ARMS:
        artifact = root / "artifacts" / arm
        artifact.mkdir(parents=True)
        Image.fromarray(target).save(artifact / "objective_source.png")
        Image.fromarray(reconstruction).save(artifact / "objective_reconstruction.png")
        Image.fromarray(error).save(artifact / "objective_error.png")
        np.savez_compressed(
            artifact / "analysis.npz",
            worst_crop_bounds=np.asarray([0, 0, 4, 4]),
            hair_crop_bounds=np.asarray([1, 1, 3, 3]),
        )
        (artifact / "field.gaussian.npz").write_bytes(f"field:{arm}".encode())
        rows.append({"arm": arm, "artifact_dir": f"artifacts/{arm}"})
    protected = {
        "metrics.json": {"schema": h31.REPORT_SCHEMA, "rows": rows},
        "decision.json": {},
        "attempts.json": {"attempts": []},
        "feasibility.json": {},
    }
    for name, value in protected.items():
        (root / name).write_text(h31.json.dumps(value), encoding="utf-8")
    (root / "metrics.jsonl").write_text("rows\n", encoding="utf-8")
    (root / "metrics.csv").write_text("rows\n", encoding="utf-8")
    np.savez_compressed(root / "feasibility.npz", mask=np.ones((1, 1)))

    monkeypatch.setattr(
        h31,
        "_write_report",
        lambda output_root, *_args: (output_root / "index.html").write_text(
            "objective index", encoding="utf-8"
        ),
    )
    monkeypatch.setattr(
        h31,
        "_write_manifest",
        lambda output_root: (output_root / "manifest.json").write_text(
            "final manifest", encoding="utf-8"
        ),
    )

    record = finalize_views.finalize(root)

    assert record["protected_unchanged"] is True
    assert record["measurement_recomputed"] is False
    assert record["protected_hashes_before"] == record["protected_hashes_after"]
    finalized = np.asarray(
        Image.open(root / "artifacts" / h31.ARMS[0] / "source_crop.png")
    )
    assert np.array_equal(finalized, target)
