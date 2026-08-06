from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image
import pytest

from scripts.check_report_bundle import check_bundle
from structsplat.overlap_elimination import (
    AppearanceSolveConfig,
    select_protected_feature_leaves,
    solve_fixed_lattice_appearance,
)
from structsplat.pixel_contraction import (
    _ContractionEngine,
    PixelContractionConfig,
    contract_image,
)


ROOT = Path(__file__).resolve().parents[1]
REPORT_SCHEMA = "structsplat.hier009_dynamic_overlap_recovery.diagnostic.v1"
ARMS = (
    "delta_touched",
    "overlap_touched",
    "overlap_halo",
    "overlap_halo_protected",
)


def _gradient(height: int = 16, width: int = 16) -> np.ndarray:
    yy, xx = np.mgrid[:height, :width]
    return np.stack(
        [
            xx / max(width - 1, 1),
            yy / max(height - 1, 1),
            (xx + yy) / max(width + height - 2, 1),
        ],
        axis=2,
    ).astype(np.float32)


def test_protected_feature_selection_is_exact_deterministic_and_thin_aware() -> None:
    image = np.zeros((15, 17, 3), dtype=np.float32)
    image[:, 8, :] = 1.0
    mask = np.ones(image.shape[:2], dtype=bool)
    first = select_protected_feature_leaves(image, mask, 12)
    second = select_protected_feature_leaves(image, mask, 12)
    assert first.selected_count == first.requested_count == 12
    assert first.nms_selected_count == 12
    assert np.array_equal(first.protected_mask, second.protected_mask)
    assert np.array_equal(first.priority, second.priority)
    protected_x = np.nonzero(first.protected_mask)[1]
    assert np.mean(np.abs(protected_x - 8) <= 2) == 1.0
    assert np.max(first.priority[mask]) == pytest.approx(1.0)


def test_protected_feature_selection_validates_count_mask_and_parameters() -> None:
    image = _gradient(4, 5)
    mask = np.ones((4, 5), dtype=bool)
    with pytest.raises(ValueError, match="cannot exceed"):
        select_protected_feature_leaves(image, mask, 21)
    with pytest.raises(ValueError, match="highpass_sigma_px"):
        select_protected_feature_leaves(image, mask, 2, highpass_sigma_px=0.0)
    with pytest.raises(ValueError, match="nms_radius_px"):
        select_protected_feature_leaves(image, mask, 2, nms_radius_px=-1)
    with pytest.raises(ValueError, match="bool"):
        select_protected_feature_leaves(image, mask.astype(np.uint8), 2)


def test_direct_neighborhood_scope_is_the_exact_3x3_center_halo() -> None:
    image = _gradient(5, 5)
    mask = np.ones((5, 5), dtype=bool)
    config = PixelContractionConfig(
        target_gaussians=20,
        recovery_scope="touched_neighborhood",
        recovery_neighborhood_radius_px=1,
    )
    engine = _ContractionEngine(image, mask, config)
    ids = engine._direct_neighborhood_active_ids(np.asarray([12], dtype=np.int64))
    coordinates = {tuple(engine.means[index].astype(int)) for index in ids}
    assert coordinates == {
        (x, y) for y in range(1, 4) for x in range(1, 4)
    }


def test_zero_protection_preserves_default_contraction_exactly() -> None:
    image = np.random.default_rng(9).random((8, 9, 3), dtype=np.float32)
    mask = np.ones(image.shape[:2], dtype=bool)
    config = PixelContractionConfig(target_gaussians=36)
    baseline = contract_image(image, config, mask=mask)
    explicit_zero = contract_image(
        image,
        config,
        mask=mask,
        protected_leaf_mask=np.zeros(mask.shape, dtype=bool),
    )
    for name in baseline.field._array_items():
        assert np.array_equal(
            baseline.field._array_items()[name],
            explicit_zero.field._array_items()[name],
        )
    assert baseline.history_records() == explicit_zero.history_records()
    assert baseline.reconstruction_raw.tobytes() == explicit_zero.reconstruction_raw.tobytes()


def test_dynamic_overlap_halo_keeps_protected_geometry_and_changes_neighbors() -> None:
    image = _gradient()
    mask = np.ones(image.shape[:2], dtype=bool)
    selection = select_protected_feature_leaves(image, mask, 8)
    coefficients, _, _ = solve_fixed_lattice_appearance(
        image,
        mask,
        mask,
        scale_px=0.5,
        config=AppearanceSolveConfig(),
    )
    config = PixelContractionConfig(
        target_gaussians=128,
        leaf_scale_px=0.5,
        recovery_steps=2,
        recovery_scope="touched_neighborhood",
        recovery_progress_checkpoints=2,
        recovery_neighborhood_radius_px=1,
        recovery_device="cpu",
        recovery_renderer="additive",
        recovery_render_chunk=64,
    )
    result = contract_image(
        image,
        config,
        mask=mask,
        initial_coefficients=coefficients,
        protected_leaf_mask=selection.protected_mask,
    )
    assert result.final_count == 128
    assert result.stop_reason == "target_reached"
    assert result.protected_initial_rows == result.protected_active_rows == 8
    assert any(event.neighborhood_count > 0 for event in result.recovery_history)
    assert any(
        event.accepted_new_neighborhood_count > 0
        for event in result.recovery_history
    )
    protected_xy = np.stack(np.nonzero(selection.protected_mask)[::-1], axis=1)
    field_xy = {tuple(value) for value in result.field.means_xy.tolist()}
    assert all(tuple(value.astype(np.float32)) in field_xy for value in protected_xy)


def test_protected_mask_validation_fails_closed() -> None:
    image = _gradient(6, 7)
    mask = np.ones(image.shape[:2], dtype=bool)
    mask[0, 0] = False
    config = PixelContractionConfig(target_gaussians=20)
    outside = np.zeros(mask.shape, dtype=bool)
    outside[0, 0] = True
    with pytest.raises(ValueError, match="subset"):
        contract_image(image, config, mask=mask, protected_leaf_mask=outside)
    too_many = mask.copy()
    with pytest.raises(ValueError, match="cannot exceed"):
        contract_image(image, config, mask=mask, protected_leaf_mask=too_many)


def test_hier009_driver_writes_complete_eight_cell_report(tmp_path: Path) -> None:
    pixels = np.random.default_rng(41).integers(0, 256, size=(18, 24, 3), dtype=np.uint8)
    source = tmp_path / "source.png"
    Image.fromarray(pixels, mode="RGB").save(source)
    mask_pixels = np.zeros((18, 24), dtype=np.uint8)
    mask_pixels[:, :16] = 255
    mask = tmp_path / "mask.png"
    Image.fromarray(mask_pixels, mode="L").save(mask)
    output = tmp_path / "report"
    script = ROOT / "scripts" / "experiments" / "hier009_dynamic_overlap_recovery.py"
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--images",
            str(source),
            "--mask",
            str(mask),
            "--out",
            str(output),
            "--max-side",
            "12",
            "--target-gaussians",
            "24",
            "48",
            "--recovery-checkpoints",
            "1",
            "--recovery-steps",
            "1",
            "--device",
            "cpu",
            "--renderer",
            "additive",
            "--render-chunk",
            "64",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    assert payload["schema"] == REPORT_SCHEMA
    assert payload["row_count"] == 8
    assert {
        (row["arm"], row["target_gaussians"]) for row in payload["rows"]
    } == {(arm, count) for arm in ARMS for count in (24, 48)}
    assert all(row["maintained_render_parity_max_abs"] < 2e-6 for row in payload["rows"])
    assert (output / "curves" / "snapshot__psnr_db.svg").is_file()
    assert (output / "curves" / "recovery__psnr_after_db.svg").is_file()
    artifact = output / "artifacts" / "source__overlap_halo_protected__n48"
    assert (artifact / "protected.png").is_file()
    assert (artifact / "analysis.npz").is_file()
    assert check_bundle(output) == []
