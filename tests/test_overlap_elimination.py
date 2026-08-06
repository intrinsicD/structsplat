from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from PIL import Image

from scripts.check_report_bundle import check_bundle

from structsplat.observation_field import ObservationField2D
from structsplat.overlap_elimination import (
    AppearanceSolveConfig,
    FeatureEliminationConfig,
    FieldOptimizerConfig,
    feature_wse_schur_eliminate,
    gaussian_stencil,
    lattice_observation_field,
    optimize_observation_field,
    render_fixed_lattice,
    solve_fixed_lattice_appearance,
)
from structsplat.pixel_contraction import (
    PixelContractionConfig,
    contract_image,
    render_observation_field,
)


ROOT = Path(__file__).resolve().parents[1]


def _pattern(name: str, height: int = 18, width: int = 20) -> np.ndarray:
    yy, xx = np.mgrid[:height, :width]
    if name == "flat":
        result = np.full((height, width, 3), (0.2, 0.45, 0.7), dtype=np.float32)
    elif name == "step":
        value = (xx >= width // 2).astype(np.float32)
        result = np.stack([value, 0.2 + 0.6 * value, 1.0 - value], axis=2)
    elif name == "diagonal":
        value = (xx >= yy + 1).astype(np.float32)
        result = np.stack([value, 0.25 + 0.5 * value, 1.0 - value], axis=2)
    elif name == "gradient":
        result = np.stack(
            [
                xx / max(width - 1, 1),
                yy / max(height - 1, 1),
                (xx + yy) / max(width + height - 2, 1),
            ],
            axis=2,
        )
    elif name == "checkerboard":
        value = ((xx + yy) % 2).astype(np.float32)
        result = np.stack([value, 1.0 - value, 0.2 + 0.6 * value], axis=2)
    else:  # pragma: no cover - test helper guard
        raise ValueError(name)
    return np.asarray(result, dtype=np.float32)


def test_neighbourhood_overlap_is_material_at_half_pixel_scale() -> None:
    _, delta_weights = gaussian_stencil(0.18)
    overlap_offsets, overlap_weights = gaussian_stencil(0.50)
    overlap = {
        tuple(offset): float(weight)
        for offset, weight in zip(overlap_offsets.tolist(), overlap_weights.tolist())
    }

    assert float(np.partition(delta_weights, -2)[-2]) < 3e-7
    assert overlap[(0, 1)] == pytest.approx(np.exp(-2.0), rel=1e-7)
    assert overlap[(1, 1)] == pytest.approx(np.exp(-4.0), rel=1e-7)
    assert len(overlap) == 25  # smooth 5x5 AABB enumeration, with effective 3x3 mass


@pytest.mark.parametrize("pattern", ["flat", "step", "diagonal", "gradient", "checkerboard"])
@pytest.mark.parametrize("scale", [0.18, 0.50])
def test_full_lattice_prefit_is_exact_on_synthetic_patterns(pattern: str, scale: float) -> None:
    image = _pattern(pattern)
    mask = np.ones(image.shape[:2], dtype=bool)

    coefficients, reconstruction, diagnostics = solve_fixed_lattice_appearance(
        image,
        mask,
        mask,
        scale_px=scale,
        config=AppearanceSolveConfig(tolerance=1e-8, max_iterations=200, ridge=1e-8),
    )

    assert diagnostics.converged
    assert diagnostics.data_pixel_rmse_max < 2e-5
    assert diagnostics.coefficient_abs_max < 4.0
    assert np.max(np.abs(reconstruction - image)) < 4e-5
    np.testing.assert_allclose(
        render_fixed_lattice(coefficients, mask, scale_px=scale),
        reconstruction,
        atol=2e-7,
        rtol=0.0,
    )


def test_masked_prefit_matches_the_declared_renderer_and_lossless_field(tmp_path) -> None:
    image = _pattern("gradient", 16, 19)
    mask = np.zeros(image.shape[:2], dtype=bool)
    mask[2:-2, 3:-3] = True
    mask[6:9, 8:11] = False
    coefficients, reconstruction, diagnostics = solve_fixed_lattice_appearance(
        image,
        mask,
        mask,
        scale_px=0.5,
    )
    field = lattice_observation_field(mask, mask, coefficients, scale_px=0.5)
    path = tmp_path / "field.npz"
    field.save_lossless(path)
    loaded = ObservationField2D.load_lossless(path)
    maintained = render_observation_field(
        loaded,
        device="cpu",
        renderer="additive",
        render_chunk=64,
        apply_declared_alpha=False,
    )

    assert diagnostics.data_pixel_rmse_max < 2e-5
    np.testing.assert_allclose(maintained, reconstruction, atol=4e-7, rtol=0.0)
    assert np.count_nonzero(loaded.alpha_mask()) == int(mask.sum())


def test_feature_wse_schur_is_exact_nested_and_deterministic() -> None:
    image = _pattern("diagonal", 22, 24)
    mask = np.ones(image.shape[:2], dtype=bool)
    coefficients, _, _ = solve_fixed_lattice_appearance(image, mask, mask, scale_px=0.5)
    config = FeatureEliminationConfig(target_count=132)

    first = feature_wse_schur_eliminate(
        image,
        mask,
        coefficients,
        (264, 132),
        scale_px=0.5,
        config=config,
    )
    second = feature_wse_schur_eliminate(
        image,
        mask,
        coefficients,
        (264, 132),
        scale_px=0.5,
        config=config,
    )

    assert int(first.survivors_by_count[264].sum()) == 264
    assert int(first.survivors_by_count[132].sum()) == 132
    assert np.all(first.survivors_by_count[132] <= first.survivors_by_count[264])
    np.testing.assert_array_equal(first.removal_order, second.removal_order)
    for count in (264, 132):
        np.testing.assert_array_equal(first.survivors_by_count[count], second.survivors_by_count[count])


def test_feature_wse_protects_a_strong_step_edge() -> None:
    image = _pattern("step", 28, 30)
    mask = np.ones(image.shape[:2], dtype=bool)
    coefficients, _, _ = solve_fixed_lattice_appearance(image, mask, mask, scale_px=0.5)
    result = feature_wse_schur_eliminate(
        image,
        mask,
        coefficients,
        (210,),
        scale_px=0.5,
        config=FeatureEliminationConfig(target_count=210),
    )
    survivors = result.survivors_by_count[210]
    edge_band = np.zeros_like(mask)
    edge_band[:, 13:17] = True
    flat_band = np.zeros_like(mask)
    flat_band[:, :4] = True

    edge_retention = float(np.mean(survivors[edge_band]))
    flat_retention = float(np.mean(survivors[flat_band]))
    assert edge_retention > flat_retention
    assert float(np.mean(result.target_radius[result.feature_normalized[mask] > 0.5])) < float(
        np.mean(result.target_radius)
    )


def test_overlap_reduces_the_local_schur_residual_fraction() -> None:
    image = _pattern("flat", 18, 20)
    mask = np.ones(image.shape[:2], dtype=bool)
    results = {}
    for scale in (0.18, 0.50):
        coefficients, _, _ = solve_fixed_lattice_appearance(image, mask, mask, scale_px=scale)
        results[scale] = feature_wse_schur_eliminate(
            image,
            mask,
            coefficients,
            (180,),
            scale_px=scale,
            config=FeatureEliminationConfig(target_count=180),
        )

    assert float(np.mean(results[0.50].schur_residual_fraction)) < 0.85 * float(
        np.mean(results[0.18].schur_residual_fraction)
    )


def test_quadtree_accepts_prefiltered_overlap_coefficients() -> None:
    image = _pattern("gradient", 18, 20)
    mask = np.ones(image.shape[:2], dtype=bool)
    coefficients, _, solve = solve_fixed_lattice_appearance(image, mask, mask, scale_px=0.5)
    result = contract_image(
        image,
        PixelContractionConfig(target_gaussians=180, leaf_scale_px=0.5),
        mask=mask,
        initial_coefficients=coefficients,
    )

    assert solve.data_pixel_rmse_max < 2e-5
    assert result.final_count == 180
    assert result.initial_sse < 1e-7
    assert result.stop_reason == "target_reached"


def test_common_optimizer_has_safe_measurable_impact_and_respects_trust_regions() -> None:
    image = _pattern("gradient", 14, 16)
    mask = np.ones(image.shape[:2], dtype=bool)
    basis = np.zeros_like(mask)
    basis[::2, ::2] = True
    coefficients, _, _ = solve_fixed_lattice_appearance(image, mask, basis, scale_px=0.5)
    field = lattice_observation_field(mask, basis, coefficients, scale_px=0.5)
    config = FieldOptimizerConfig(
        steps=4,
        checkpoint_every=1,
        device="cpu",
        renderer="additive",
        render_chunk=64,
        max_mean_shift=0.04,
        max_log_scale_shift=0.03,
    )

    result = optimize_observation_field(field, image, mask, config=config)

    assert result.selected_step > 0
    assert result.optimizer_sse_gain > 0.0
    assert result.optimizer_psnr_gain_db > 0.0
    assert result.mean_shift_max <= config.max_mean_shift + 1e-6
    assert result.log_scale_shift_max <= config.max_log_scale_shift + 1e-6
    step_zero = result.checkpoints[0]
    selected = next(checkpoint for checkpoint in result.checkpoints if checkpoint.selected)
    assert selected.raw_pixel_rmse_max <= step_zero.raw_pixel_rmse_max + 1e-6
    assert selected.raw_patch7_rmse_max <= step_zero.raw_patch7_rmse_max + 1e-6
    maintained = render_observation_field(
        result.field,
        device="cpu",
        renderer="additive",
        render_chunk=64,
        apply_declared_alpha=False,
    )
    np.testing.assert_allclose(maintained, result.reconstruction_raw, atol=4e-7, rtol=0.0)


def test_invalid_initial_coefficients_and_elimination_contract_fail_closed() -> None:
    image = _pattern("flat", 8, 9)
    mask = np.ones(image.shape[:2], dtype=bool)
    with pytest.raises(ValueError, match="initial_coefficients"):
        contract_image(
            image,
            PixelContractionConfig(target_gaussians=36),
            mask=mask,
            initial_coefficients=np.zeros((3, 3), dtype=np.float32),
        )
    coefficients, _, _ = solve_fixed_lattice_appearance(image, mask, mask, scale_px=0.5)
    with pytest.raises(ValueError, match="smallest"):
        feature_wse_schur_eliminate(
            image,
            mask,
            coefficients,
            (36,),
            scale_px=0.5,
            config=FeatureEliminationConfig(target_count=24),
        )


def test_hier008_driver_writes_complete_eight_cell_report(tmp_path: Path) -> None:
    pixels = np.random.default_rng(31).integers(0, 256, size=(18, 24, 3), dtype=np.uint8)
    source = tmp_path / "source.png"
    Image.fromarray(pixels, mode="RGB").save(source)
    mask_pixels = np.zeros((18, 24), dtype=np.uint8)
    mask_pixels[:, :16] = 255
    mask = tmp_path / "mask.png"
    Image.fromarray(mask_pixels, mode="L").save(mask)
    output = tmp_path / "report"
    script = ROOT / "scripts" / "experiments" / "hier008_overlap_elimination.py"
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
            "--support-arms",
            "near_delta=0.18",
            "overlap=0.50",
            "--schedulers",
            "quadtree",
            "feature_wse_schur",
            "--optimizer-steps",
            "1",
            "--checkpoint-every",
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
    assert payload["schema"] == "structsplat.hier008_overlap_elimination.diagnostic.v1"
    assert payload["claim_ready"] is False
    assert payload["row_count"] == 8
    assert {
        (row["support_arm"], row["scheduler"], row["n_gaussians"])
        for row in payload["rows"]
    } == {
        (support, scheduler, count)
        for support in ("near_delta", "overlap")
        for scheduler in ("quadtree", "feature_wse_schur")
        for count in (24, 48)
    }
    assert all(row["maintained_render_parity_max_abs"] < 2e-6 for row in payload["rows"])
    assert (output / "curves" / "snapshot__psnr_db.svg").is_file()
    assert (output / "curves" / "optimizer__raw_sse.svg").is_file()
    artifact = output / "artifacts" / "source__overlap__feature_wse_schur__n48"
    assert (artifact / "analysis.npz").is_file()
    assert (artifact / "optimizer_history.json").is_file()
    assert check_bundle(output) == []
