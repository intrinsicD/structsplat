from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image
import pytest

from scripts.check_report_bundle import check_bundle
from structsplat.observation_field import ObservationField2D
from structsplat.pixel_contraction import (
    _ContractionEngine,
    _mask_aware_smoothed_error,
    _normalized_error_update_weights,
    LocalRescueConfig,
    PixelContractionConfig,
    contract_image,
    gaussian_inner_product,
    render_observation_field,
    rescue_observation_field,
)


ROOT = Path(__file__).resolve().parents[1]


def _image(height: int = 7, width: int = 9, seed: int = 4) -> np.ndarray:
    return np.random.default_rng(seed).random((height, width, 3), dtype=np.float32)


def test_gaussian_inner_product_matches_closed_form_and_numerical_quadrature():
    covariance = np.array([[2.0, 0.4], [0.4, 0.8]], dtype=np.float64)
    mean = np.array([0.5, -0.25], dtype=np.float64)
    self_inner = gaussian_inner_product(mean, covariance, mean, covariance)
    assert self_inner == pytest.approx(np.pi * np.sqrt(np.linalg.det(covariance)), rel=1e-12)

    mean_b = np.array([-0.7, 0.9], dtype=np.float64)
    covariance_b = np.array([[0.6, -0.15], [-0.15, 1.4]], dtype=np.float64)
    analytic = gaussian_inner_product(mean, covariance, mean_b, covariance_b)
    axis = np.linspace(-8.0, 8.0, 501)
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    points = np.stack([xx, yy], axis=-1)

    def kernel(center: np.ndarray, cov: np.ndarray) -> np.ndarray:
        delta = points - center
        inverse = np.linalg.inv(cov)
        quadratic = np.einsum("...i,ij,...j->...", delta, inverse, delta)
        return np.exp(-0.5 * quadratic)

    spacing = axis[1] - axis[0]
    numerical = float(np.sum(kernel(mean, covariance) * kernel(mean_b, covariance_b)) * spacing**2)
    assert numerical == pytest.approx(analytic, rel=4e-4)


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"target_gaussians": 0}, ValueError),
        ({"target_gaussians": 1, "leaf_scale_px": 0.0}, ValueError),
        ({"target_gaussians": 1, "support_fade_alpha": 1.1}, ValueError),
        ({"target_gaussians": 1, "coefficient_domain": "invalid"}, ValueError),
        ({"target_gaussians": 1, "pair_policy": "invalid"}, ValueError),
        ({"target_gaussians": 1, "recovery_steps": -1}, ValueError),
        ({"target_gaussians": 1, "recovery_every_actions": 0}, ValueError),
        ({"target_gaussians": 1, "recovery_schedule": "invalid"}, ValueError),
        ({"target_gaussians": 1, "recovery_scope": "invalid"}, ValueError),
        (
            {
                "target_gaussians": 1,
                "recovery_error_smoothing_sigma_px": -0.1,
            },
            ValueError,
        ),
        (
            {"target_gaussians": 1, "recovery_error_weight_floor": 1.1},
            ValueError,
        ),
        (
            {"target_gaussians": 1, "recovery_error_weight_ceiling": 0.9},
            ValueError,
        ),
        (
            {
                "target_gaussians": 1,
                "recovery_renderer": "cuda_additive",
                "recovery_device": "cpu",
            },
            ValueError,
        ),
        ({"target_gaussians": 1, "proposal_batch_size": True}, TypeError),
    ],
)
def test_config_fails_closed(kwargs, error):
    with pytest.raises(error):
        PixelContractionConfig(**kwargs)


def test_pixel_endpoint_is_near_exact_and_uses_no_topology_actions():
    image = _image(5, 7)
    result = contract_image(image, PixelContractionConfig(target_gaussians=35))
    assert result.final_count == 35
    assert result.stop_reason == "target_reached"
    assert result.history == ()
    assert result.initial_sse == pytest.approx(result.final_sse, abs=1e-18)
    assert result.final_sse < 1e-8
    assert result.field.semantics.renderer_equation == "additive_rgb_peak_one_v1"
    assert result.field.semantics.coefficient_domain == "signed"
    maintained = render_observation_field(result.field)
    assert np.max(np.abs(maintained - result.reconstruction)) < 2e-6


def test_exact_counts_are_reached_on_odd_shapes_and_one_row_endpoint():
    image = _image(5, 7)
    for target in (34, 23, 11, 2, 1):
        result = contract_image(
            image,
            PixelContractionConfig(
                target_gaussians=target,
                proposal_batch_size=12,
                merge_batch_size=4,
            ),
        )
        assert result.final_count == target
        assert result.stop_reason == "target_reached"
        counts = [event.count_before for event in result.history]
        counts.extend(event.count_after for event in result.history[-1:])
        assert counts == sorted(counts, reverse=True)


def test_exact_discrete_action_deltas_telescope_to_final_sse():
    image = _image(6, 8, seed=11)
    result = contract_image(
        image,
        PixelContractionConfig(
            target_gaussians=13,
            proposal_batch_size=16,
            merge_batch_size=4,
            pair_policy="always",
        ),
    )
    assert result.history
    accumulated = result.initial_sse + sum(
        event.exact_discrete_sse_delta for event in result.history
    )
    assert accumulated == pytest.approx(result.final_sse, abs=2e-6)
    previous = result.initial_count
    for event in result.history:
        assert event.count_before == previous
        assert event.count_after == event.count_before - event.rows_removed + event.rows_added
        assert event.rows_removed > event.rows_added
        assert np.isfinite(event.analytic_continuous_sse)
        assert np.isfinite(event.exact_discrete_sse_delta)
        left, top, right, bottom = event.patch_xyxy
        assert 0 <= left <= right < image.shape[1]
        assert 0 <= top <= bottom < image.shape[0]
        previous = event.count_after
    assert previous == result.final_count


def test_masked_nonnegative_contraction_packs_alpha_and_matches_renderer(tmp_path: Path):
    image = _image(7, 9, seed=13)
    yy, xx = np.mgrid[:7, :9]
    mask = ((xx - 4) ** 2 + (yy - 3) ** 2 <= 10) & ~((xx == 4) & (yy == 3))
    result = contract_image(
        image,
        PixelContractionConfig(
            target_gaussians=7,
            coefficient_domain="nonnegative",
            proposal_batch_size=12,
            merge_batch_size=4,
        ),
        mask=mask,
    )
    assert result.final_count == 7
    assert result.field.packed_alpha is not None
    assert np.array_equal(result.field.alpha_mask(), mask)
    assert result.field.semantics.alpha.matting_mode == "multiply_alpha"
    assert float(result.field.rgb_coeff.min()) >= 0.0
    assert np.array_equal(result.reconstruction[~mask], np.zeros((int((~mask).sum()), 3)))

    field_path = tmp_path / "field.npz"
    result.field.save_lossless(field_path)
    cold = ObservationField2D.load_lossless(field_path)
    assert cold.canonical_hash() == result.field.canonical_hash()
    maintained = render_observation_field(cold)
    assert np.max(np.abs(maintained - result.reconstruction)) < 2e-6


def test_contraction_is_bit_deterministic_for_same_input_and_config():
    image = _image(8, 8, seed=17)
    config = PixelContractionConfig(
        target_gaussians=15,
        proposal_batch_size=16,
        merge_batch_size=4,
        pair_policy="always",
    )
    first = contract_image(image, config)
    second = contract_image(image, config)
    assert first.field.canonical_hash() == second.field.canonical_hash()
    assert first.history_records() == second.history_records()
    assert np.array_equal(first.reconstruction, second.reconstruction)
    assert np.array_equal(first.touched_row_mask, second.touched_row_mask)
    assert np.array_equal(first.protected_row_mask, second.protected_row_mask)
    assert not first.touched_row_mask.flags.writeable
    assert not first.protected_row_mask.flags.writeable
    assert int(first.touched_row_mask.sum()) == first.touched_active_rows
    assert int((~first.touched_row_mask).sum()) == first.untouched_active_rows
    assert int(first.protected_row_mask.sum()) == first.protected_active_rows


def test_selective_recovery_freezes_untouched_rows_and_telescopes_sse():
    image = _image(8, 8, seed=77)
    mask = np.ones(image.shape[:2], dtype=bool)
    config = PixelContractionConfig(
        target_gaussians=32,
        recovery_steps=4,
        recovery_schedule="actions",
        recovery_every_actions=2,
        recovery_render_chunk=16,
        recovery_lr_means=0.03,
        recovery_lr_scales=0.02,
        recovery_lr_rotations=0.01,
        recovery_lr_coefficients=0.02,
    )
    engine = _ContractionEngine(image, mask, config)
    initial_means = engine.means.copy()
    initial_covariances = engine.compact_covariances.copy()
    initial_coefficients = engine.coefficients.copy()
    engine.run()
    untouched = engine.active & ~engine.ever_touched
    assert untouched.any()
    assert np.array_equal(engine.means[untouched], initial_means[untouched])
    assert np.array_equal(
        engine.compact_covariances[untouched], initial_covariances[untouched]
    )
    assert np.array_equal(engine.coefficients[untouched], initial_coefficients[untouched])
    assert engine.recovery_history
    assert any(event.accepted for event in engine.recovery_history)
    assert all(event.sse_after <= event.sse_before for event in engine.recovery_history)
    assert all(
        event.touched_count >= event.newly_touched_count
        for event in engine.recovery_history
    )
    assert any(
        event.touched_count > event.newly_touched_count
        for event in engine.recovery_history[1:]
    )
    telescoped = engine.initial_sse
    telescoped += sum(event.exact_discrete_sse_delta for event in engine.history)
    telescoped += sum(
        event.sse_after - event.sse_before for event in engine.recovery_history
    )
    assert telescoped == pytest.approx(engine._masked_sse(engine.current_render), abs=2e-6)

    first = contract_image(image, config)
    second = contract_image(image, config)
    assert first.field.canonical_hash() == second.field.canonical_hash()
    assert first.final_count == config.target_gaussians
    assert first.touched_active_rows + first.untouched_active_rows == first.final_count
    assert first.untouched_active_rows > 0
    first_records = first.recovery_records()
    second_records = second.recovery_records()
    for record in first_records + second_records:
        record.pop("elapsed_seconds")
    assert first_records == second_records


def test_mask_aware_error_smoothing_and_row_weight_normalization():
    mask = np.zeros((9, 11), dtype=bool)
    mask[2:8, 3:9] = True
    constant = np.zeros(mask.shape, dtype=np.float32)
    constant[mask] = 3.0
    smoothed_constant = _mask_aware_smoothed_error(constant, mask, 1.5)
    assert np.allclose(smoothed_constant[mask], 3.0, rtol=1e-6, atol=1e-6)
    assert np.array_equal(smoothed_constant[~mask], np.zeros(int((~mask).sum())))

    impulse = np.zeros(mask.shape, dtype=np.float32)
    impulse[5, 6] = 1.0
    smoothed_impulse = _mask_aware_smoothed_error(impulse, mask, 1.0)
    assert 0.0 < smoothed_impulse[5, 6] < 1.0
    assert smoothed_impulse[5, 5] > 0.0
    assert np.array_equal(smoothed_impulse[~mask], np.zeros(int((~mask).sum())))

    uniform = _normalized_error_update_weights(
        np.ones(8), power=0.5, floor=0.05, ceiling=4.0
    )
    assert np.array_equal(uniform, np.ones(8, dtype=np.float32))
    weighted = _normalized_error_update_weights(
        np.array([0.0, 0.1, 0.5, 2.0, 8.0]),
        power=0.5,
        floor=0.05,
        ceiling=4.0,
    )
    assert np.isfinite(weighted).all()
    assert float(weighted.min()) >= 0.05
    assert float(weighted.max()) <= 4.0
    assert weighted.tolist() == sorted(weighted.tolist())
    assert float(weighted.mean()) == pytest.approx(1.0, abs=1e-6)


def test_error_weights_scale_post_adam_row_updates_exactly():
    from types import SimpleNamespace

    import torch

    field = SimpleNamespace(
        means=torch.zeros((2, 2)),
        log_scales=torch.zeros((2, 2)),
        rotations=torch.zeros(2),
        colors=torch.zeros((2, 3)),
    )
    parameters = (field.means, field.log_scales, field.rotations, field.colors)
    previous = tuple(parameter.clone() for parameter in parameters)
    for parameter in parameters:
        parameter.add_(1.0)
    with torch.no_grad():
        _ContractionEngine._scale_adam_row_updates(
            field,
            previous,
            torch.tensor([0.25, 2.0]),
        )
    for parameter in parameters:
        assert torch.equal(parameter[0], torch.full_like(parameter[0], 0.25))
        assert torch.equal(parameter[1], torch.full_like(parameter[1], 2.0))


def test_all_active_error_weighted_recovery_moves_untouched_rows_and_is_deterministic():
    image = _image(8, 8, seed=77)
    mask = np.ones(image.shape[:2], dtype=bool)
    config = PixelContractionConfig(
        target_gaussians=32,
        recovery_steps=4,
        recovery_scope="all_error_weighted",
        recovery_schedule="actions",
        recovery_every_actions=2,
        recovery_render_chunk=16,
        recovery_lr_means=0.03,
        recovery_lr_scales=0.02,
        recovery_lr_rotations=0.01,
        recovery_lr_coefficients=0.02,
    )
    engine = _ContractionEngine(image, mask, config)
    initial_means = engine.means.copy()
    initial_coefficients = engine.coefficients.copy()
    engine.run()
    untouched = engine.active & ~engine.ever_touched
    assert untouched.any()
    assert np.any(engine.means[untouched] != initial_means[untouched])
    assert np.any(engine.coefficients[untouched] != initial_coefficients[untouched])
    assert engine.recovery_history
    assert all(
        event.optimized_count == event.active_count
        for event in engine.recovery_history
    )
    assert all(
        event.recovery_scope == "all_error_weighted"
        for event in engine.recovery_history
    )
    assert all(event.sse_after <= event.sse_before for event in engine.recovery_history)
    assert any(
        event.error_weight_max > event.error_weight_min
        for event in engine.recovery_history
    )
    assert all(
        0.0 < event.error_weight_effective_rows <= event.optimized_count
        for event in engine.recovery_history
    )

    first = contract_image(image, config)
    second = contract_image(image, config)
    assert first.field.canonical_hash() == second.field.canonical_hash()
    first_records = first.recovery_records()
    second_records = second.recovery_records()
    for record in first_records + second_records:
        record.pop("elapsed_seconds")
        record.pop("attribution_seconds")
    assert first_records == second_records


def test_progress_recovery_has_count_normalized_checkpoint_work():
    image = _image(16, 16, seed=99)
    config = PixelContractionConfig(
        target_gaussians=128,
        recovery_steps=1,
        recovery_schedule="progress",
        recovery_progress_checkpoints=4,
        recovery_render_chunk=32,
        proposal_batch_size=16,
        merge_batch_size=4,
    )
    result = contract_image(image, config)
    assert result.final_count == config.target_gaussians
    assert len(result.recovery_history) == config.recovery_progress_checkpoints
    assert sum(event.attempted_steps for event in result.recovery_history) == 4
    assert [event.active_count for event in result.recovery_history] == sorted(
        (event.active_count for event in result.recovery_history), reverse=True
    )


def test_one_progress_checkpoint_is_terminal_all_active_recovery():
    image = _image(12, 12, seed=101)
    config = PixelContractionConfig(
        target_gaussians=72,
        recovery_steps=2,
        recovery_scope="all_error_weighted",
        recovery_schedule="progress",
        recovery_progress_checkpoints=1,
        recovery_render_chunk=32,
        proposal_batch_size=16,
        merge_batch_size=4,
    )
    result = contract_image(image, config)
    assert result.final_count == config.target_gaussians
    assert len(result.recovery_history) == 1
    event = result.recovery_history[0]
    assert event.action_count == len(result.history)
    assert event.active_count == config.target_gaussians
    assert event.optimized_count == config.target_gaussians
    assert event.attempted_steps == config.recovery_steps


def test_faded_large_support_round_trips_and_matches_maintained_renderer():
    image = _image(7, 9, seed=103)
    config = PixelContractionConfig(
        target_gaussians=21,
        sigma_cutoff=4.5,
        support_fade_alpha=1.0,
        proposal_batch_size=16,
        merge_batch_size=4,
    )
    result = contract_image(image, config)
    assert result.field.semantics.support.sigma_cutoff == 4.5
    assert result.field.semantics.support.fade_alpha == 1.0
    maintained = render_observation_field(result.field)
    assert np.max(np.abs(maintained - result.reconstruction)) < 2e-6


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_rows": 0},
        {"max_rows": 1, "scale_px": 0.0},
        {"max_rows": 1, "nms_radius_px": -1},
        {"max_rows": 1, "steps": 0},
        {"max_rows": 1, "tail_fraction": 1.1},
        {"max_rows": 1, "renderer": "cuda_additive", "device": "cpu"},
    ],
)
def test_local_rescue_config_fails_closed(kwargs):
    with pytest.raises((TypeError, ValueError)):
        LocalRescueConfig(**kwargs)


def test_local_rescue_freezes_base_rows_improves_worst_error_and_round_trips():
    image = _image(9, 11, seed=123)
    base = contract_image(
        image,
        PixelContractionConfig(
            target_gaussians=20,
            proposal_batch_size=16,
            merge_batch_size=4,
        ),
    )
    config = LocalRescueConfig(
        max_rows=8,
        steps=30,
        learning_rate=0.03,
        render_chunk=64,
    )
    first = rescue_observation_field(base.field, image, config)
    second = rescue_observation_field(base.field, image, config)

    assert first.rows_added == config.max_rows
    assert first.field.n == base.field.n + config.max_rows
    assert first.selected_step >= 0
    assert first.violation_after < first.violation_before
    assert first.final_sse < first.initial_sse
    assert np.array_equal(first.field.means_xy[: base.field.n], base.field.means_xy)
    assert np.array_equal(
        first.field.log_scales_xy[: base.field.n], base.field.log_scales_xy
    )
    assert np.array_equal(
        first.field.rotations_rad[: base.field.n], base.field.rotations_rad
    )
    assert np.array_equal(first.field.rgb_coeff[: base.field.n], base.field.rgb_coeff)
    maintained = render_observation_field(first.field, render_chunk=64)
    assert np.max(np.abs(maintained - first.reconstruction)) < 2e-6
    assert first.field.canonical_hash() == second.field.canonical_hash()
    first_record = first.to_record()
    second_record = second.to_record()
    first_record.pop("elapsed_seconds")
    second_record.pop("elapsed_seconds")
    assert first_record == second_record


def test_local_rescue_applies_declared_alpha_when_objective_mask_is_narrower():
    image = _image(7, 9, seed=124)
    field_mask = np.ones(image.shape[:2], dtype=bool)
    field_mask[:, -1] = False
    objective_mask = field_mask.copy()
    objective_mask[:, 5:] = False
    base = contract_image(
        image,
        PixelContractionConfig(
            target_gaussians=16,
            proposal_batch_size=16,
            merge_batch_size=4,
        ),
        mask=field_mask,
    )
    result = rescue_observation_field(
        base.field,
        image,
        LocalRescueConfig(max_rows=4, steps=5, render_chunk=32),
        mask=objective_mask,
    )
    maintained = render_observation_field(result.field, render_chunk=32)
    assert np.max(np.abs(maintained - result.reconstruction)) < 2e-6
    assert np.array_equal(result.reconstruction[~field_mask], np.zeros((7, 3)))


def test_local_rescue_driver_writes_cold_visual_curve_report(tmp_path: Path):
    pixels = np.random.default_rng(125).integers(
        0, 256, size=(9, 11, 3), dtype=np.uint8
    )
    source = tmp_path / "source.png"
    Image.fromarray(pixels, mode="RGB").save(source)
    image = pixels.astype(np.float32) / 255.0
    base = contract_image(
        image,
        PixelContractionConfig(
            target_gaussians=20,
            proposal_batch_size=16,
            merge_batch_size=4,
        ),
    )
    base_artifact = tmp_path / "base" / "artifacts" / "source__n20"
    base_artifact.mkdir(parents=True)
    base.field.save_lossless(base_artifact / "field.observation.npz")
    Image.fromarray(pixels, mode="RGB").save(base_artifact / "source.png")
    (base_artifact / "row.json").write_text(
        json.dumps(
            {
                "image": source.name,
                "source_path": str(source),
                "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "source_file_bytes": source.stat().st_size,
                "original_width": 11,
                "original_height": 9,
                "mask_source_sha256": None,
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "repair"
    script = ROOT / "scripts" / "experiments" / "hier005_artifact_repair.py"
    command = [
        sys.executable,
        str(script),
        "--base-artifact",
        str(base_artifact),
        "--out",
        str(output),
        "--rescue-rows",
        "4",
        "--steps",
        "3",
        "--device",
        "cpu",
        "--renderer",
        "additive",
        "--render-chunk",
        "32",
    ]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["schema"].endswith("diagnostic.v1")
    assert metrics["claim_ready"] is False
    assert len(metrics["rows"]) == 2
    base_row, rescue_row = metrics["rows"]
    assert base_row["rescue_limit"] == 0
    assert base_row["n_gaussians"] == 20
    assert rescue_row["rescue_limit"] == 4
    assert rescue_row["rescue_rows_added"] in (0, 4)
    assert rescue_row["base_prefix_bit_exact"] is True
    assert rescue_row["maintained_render_parity_max_abs"] < 2e-6
    assert (output / "index.html").is_file()
    assert (output / "metrics.csv").is_file()
    assert (output / "curves" / "psnr_db.svg").is_file()
    assert (output / "curves" / "artifact_pixel_rmse_max.svg").is_file()
    verification = json.loads((output / "verification.json").read_text(encoding="utf-8"))
    assert verification["status"] == "verified_diagnostic"
    assert verification["metric_rows_checked"] == 2
    assert (output / "artifacts" / "rescue_0004" / "error.png").is_file()
    assert (output / "artifacts" / "rescue_0004" / "error_crop.png").is_file()
    assert (output / "artifacts" / "rescue_0004" / "rescue_centers.png").is_file()
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    manifest_paths = {record["path"] for record in manifest["files"]}
    assert "metrics.json" in manifest_paths
    assert "curves/artifact_patch_rmse_max_7.svg" in manifest_paths
    assert "artifacts/rescue_0004/field.observation.npz" in manifest_paths
    assert check_bundle(output) == []

    rejected = subprocess.run(
        command,
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert rejected.returncode != 0
    assert "refusing to overwrite" in rejected.stderr


def test_numpy_contraction_module_imports_without_torch():
    command = (
        "import sys; sys.modules['torch'] = None; "
        "import structsplat.pixel_contraction; print('ok')"
    )
    completed = subprocess.run(
        [sys.executable, "-c", command],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "ok"


def test_diagnostic_driver_cold_renders_and_writes_comparison_rows(tmp_path: Path):
    pixels = np.random.default_rng(21).integers(0, 256, size=(18, 24, 3), dtype=np.uint8)
    source = tmp_path / "source.png"
    Image.fromarray(pixels, mode="RGB").save(source)
    mask_pixels = np.zeros((18, 24), dtype=np.uint8)
    mask_pixels[:, :16] = 255
    mask = tmp_path / "mask.png"
    Image.fromarray(mask_pixels, mode="L").save(mask)
    output = tmp_path / "report"
    script = ROOT / "scripts" / "experiments" / "hier005_pixel_contraction.py"
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--images",
            str(source),
            "--out",
            str(output),
            "--target-gaussians",
            "48",
            "--mask",
            str(mask),
            "--max-side",
            "12",
            "--device",
            "cpu",
            "--renderer",
            "additive",
            "--proposal-batch-size",
            "16",
            "--merge-batch-size",
            "4",
            "--recovery-steps",
            "1",
            "--recovery-scope",
            "all_error_weighted",
            "--recovery-progress-checkpoints",
            "2",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["schema"].endswith("diagnostic.v1")
    assert metrics["claim_ready"] is False
    assert metrics["metric_domains"]["psnr_db"] == "thresholded foreground mask only"
    assert "exact displayed 8-bit PNG values" in metrics["metric_domains"][
        "artifact_pixel_rmse_*"
    ]
    row = metrics["rows"][0]
    assert row["status"] == "diagnostic"
    assert row["method"] == "implicit_pixel_contraction_all_error_weighted_recovery"
    assert row["n_gaussians"] == 48
    assert row["stop_reason"] == "target_reached"
    assert row["source_file_bytes"] == source.stat().st_size
    assert row["source_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert (row["original_width"], row["original_height"]) == (24, 18)
    assert (row["width"], row["height"]) == (12, 9)
    assert row["active_pixels"] == 72
    assert row["rgb_resampling"] == "pillow_lanczos"
    assert row["mask_resampling"] == "pillow_nearest"
    assert row["mask_source_sha256"] == hashlib.sha256(mask.read_bytes()).hexdigest()
    assert row["maintained_render_parity_max_abs"] < 2e-6
    assert row["repeated_render_parity_max_abs"] == 0.0
    assert row["recovery_determinism"] == "cpu_bit_deterministic_or_recovery_disabled"
    assert row["recovery_scope"] == "all_error_weighted"
    assert row["recovery_checkpoints"] == 2
    assert row["recovery_optimized_rows_max"] > row["touched_active_rows"]
    assert row["recovery_attribution_seconds"] > 0.0
    assert row["recovery_error_weight_effective_rows_mean"] > 0.0
    assert row["first_render_seconds"] > 0.0
    assert row["render_seconds"] > 0.0
    assert row["metric_seconds"] > 0.0
    assert row["artifact_metric_domain"] == "display_png_8bit_black_matted_rgb"
    assert 0.0 <= row["artifact_pixel_rmse_q99"] <= row["artifact_pixel_rmse_q999"]
    assert row["artifact_pixel_rmse_q999"] <= row["artifact_pixel_rmse_max"]
    assert row["artifact_patch_rmse_max_3"] >= 0.0
    assert row["artifact_patch_rmse_max_7"] >= 0.0
    assert row["artifact_patch_rmse_max_15"] >= 0.0
    assert row["artifact_patch_rmse_max_31"] >= 0.0
    assert row["artifact_gate_pass"] == (
        row["artifact_pixel_rmse_max"] <= row["artifact_gate_pixel_max_threshold"]
        and row["artifact_patch_rmse_max_7"]
        <= row["artifact_gate_patch7_max_threshold"]
    )
    assert row["lossless_reference_bytes"] > row["canonical_raw_bytes"]
    evaluation_source = output / "artifacts" / "source__n48" / "source.png"
    assert row["evaluation_source_png_bytes"] == evaluation_source.stat().st_size
    assert row["evaluation_png_over_estimated_ratio"] > 0.0
    assert row["estimated_bits_per_active_pixel"] > row["estimated_bits_per_pixel"]
    assert (output / "index.html").stat().st_size > 0
    report = (output / "index.html").read_text(encoding="utf-8")
    assert "All outcome metrics versus Gaussian count" in report
    assert "same-raster file-size comparison" in report
    assert (output / "curves" / "psnr_db.svg").is_file()
    assert (output / "curves" / "contraction_seconds.svg").is_file()
    assert (output / "curves" / "recovery_seconds.svg").is_file()
    assert (output / "curves" / "recovery_attribution_seconds.svg").is_file()
    assert (output / "curves" / "recovery_error_weight_p90_mean.svg").is_file()
    assert (output / "curves" / "artifact_pixel_rmse_max.svg").is_file()
    assert (output / "curves" / "artifact_patch_rmse_max_7.svg").is_file()
    assert (output / "curves" / "source_over_estimated_ratio.svg").is_file()
    curve_catalog = json.loads((output / "curves" / "catalog.json").read_text())
    assert {curve["metric"] for curve in curve_catalog["curves"]} >= {
        "psnr_db",
        "ms_ssim",
        "evaluation_png_over_estimated_ratio",
        "artifact_pixel_rmse_q999",
        "artifact_patch_rmse_max_7",
        "total_seconds",
    }
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    manifest_paths = {entry["path"] for entry in manifest["files"]}
    assert "metrics.json" in manifest_paths
    assert "artifacts/source__n48/field.observation.npz" in manifest_paths
    assert "artifacts/source__n48/recovery_history.json" in manifest_paths
    assert "curves/psnr_db.svg" in manifest_paths
    assert "curves/estimated_bits_per_active_pixel.svg" in manifest_paths
    assert (
        "source_snapshot/scripts/experiments/hier005_pixel_contraction.py"
        in manifest_paths
    )
    assert "source_snapshot/src/structsplat/pixel_contraction.py" in manifest_paths
    assert check_bundle(output) == []

    rejected = subprocess.run(
        [
            sys.executable,
            str(script),
            "--images",
            str(source),
            "--out",
            str(output),
            "--target-gaussians",
            "48",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert rejected.returncode != 0
    assert "refusing to overwrite" in rejected.stderr
