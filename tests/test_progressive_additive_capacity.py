# ruff: noqa: E402
from dataclasses import replace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from scripts.experiments import hier026_progressive_additive_capacity as h26
from structsplat.gaussians import GaussianField
from structsplat.progressive_additive_capacity import (
    ProgressiveAdditiveCapacityConfig,
    fit_progressive_additive_capacity,
)
from structsplat.render import render_field


def _fixtures(size: int = 20) -> dict[str, np.ndarray]:
    yy, xx = np.mgrid[:size, :size]
    x = xx.astype(np.float32) / float(size - 1)
    y = yy.astype(np.float32) / float(size - 1)
    constant = np.full((size, size, 3), 0.375, dtype=np.float32)
    ramp = np.stack((x, y, 0.2 + 0.6 * x), axis=2).astype(np.float32)
    edge = np.full((size, size, 3), 0.05, dtype=np.float32)
    edge[:, size // 2 :] = np.asarray([0.95, 0.7, 0.2], dtype=np.float32)
    radius2 = (x - 0.55) ** 2 + (y - 0.4) ** 2
    blob_value = 0.1 + 0.85 * np.exp(-radius2 / 0.035)
    blob = np.stack((blob_value, 0.8 * blob_value, 0.35 + 0.3 * blob_value), axis=2)
    texture_value = ((xx // 2 + yy // 2) % 2).astype(np.float32)
    texture = np.stack((texture_value, 1.0 - texture_value, 0.5 * texture_value), axis=2)
    return {
        "constant": constant,
        "ramp": ramp,
        "edge": edge,
        "blob": blob.astype(np.float32),
        "texture": texture.astype(np.float32),
    }


def _config(renderer: str = "additive") -> ProgressiveAdditiveCapacityConfig:
    return ProgressiveAdditiveCapacityConfig(
        base_gaussians=4,
        residual_gaussians=2,
        base_steps=1,
        joint_steps=1,
        checkpoint_every=1,
        feature_cap_px=6.0,
        renderer=renderer,
        render_chunk=8,
    )


def _render(field: GaussianField, shape: tuple[int, int], mode: str = "additive"):
    return render_field(
        field.means,
        field.conics(),
        field.colors,
        field.radii(3.0),
        shape[0],
        shape[1],
        mode=mode,
        chunk=8,
        scales=field.scales(),
        rotations=field.rotations,
        sigma_cutoff=3.0,
    )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"base_gaussians": 0}, "base_gaussians"),
        ({"residual_gaussians": 0}, "residual_gaussians"),
        ({"base_steps": 0}, "base_steps"),
        ({"joint_steps": 0}, "joint_steps"),
        ({"feature_cap_px": 0.0}, "feature_cap_px"),
        ({"coefficient_abs_limit": 0.0}, "coefficient_abs_limit"),
        ({"renderer": "normalized"}, "renderer must be"),
    ],
)
def test_config_rejects_invalid_contracts(kwargs, message):
    with pytest.raises((TypeError, ValueError), match=message):
        ProgressiveAdditiveCapacityConfig(**kwargs)


def test_progressive_fit_has_exact_work_counts_and_four_array_endpoints(tmp_path):
    target = _fixtures()["ramp"]
    result = fit_progressive_additive_capacity(target, config=_config(), seed=3)

    assert result.completed
    assert result.status == "completed"
    assert result.base_field.n == result.base_count == 4
    assert result.field.n == result.total_count == 6
    assert result.residual_count == 2
    assert result.attempted_steps == result.completed_steps == 2
    assert result.gaussian_row_updates == 4 * 1 + 6 * 1
    assert [point.step for point in result.trajectory] == [1, 2]
    assert [point.stage for point in result.trajectory] == ["base", "joint"]
    assert result.observer_renderer_calls == 2
    assert result.diagnostic_renderer_calls == 4
    assert result.base_endpoint_unchanged
    assert result.joint_training_mask_absent
    assert result.training_payload_removed
    assert result.base_endpoint_parity_max_abs <= 2e-5
    assert result.append_parity_max_abs <= 2e-5
    assert result.endpoint_parity_max_abs <= 2e-5

    for name, field, expected in (
        ("base", result.base_field, result.base_reconstruction_raw),
        ("progressive", result.field, result.reconstruction_raw),
    ):
        path = tmp_path / f"{name}.npz"
        field.save(str(path))
        with np.load(path) as payload:
            assert set(payload.files) == {
                "means",
                "log_scales",
                "rotations",
                "colors",
            }
        cold = GaussianField.load(str(path))
        rendered = _render(cold, target.shape[:2]).detach().numpy()
        assert np.max(np.abs(rendered - expected)) <= 2e-5


def test_progressive_fit_is_deterministic_and_records_stable_digests():
    target = _fixtures()["blob"]
    first = fit_progressive_additive_capacity(target, config=_config(), seed=7)
    second = fit_progressive_additive_capacity(target, config=_config(), seed=7)

    assert np.array_equal(first.base_reconstruction_raw, second.base_reconstruction_raw)
    assert np.array_equal(first.reconstruction_raw, second.reconstruction_raw)
    for name in (
        "initial_field_digest",
        "base_training_field_digest",
        "base_endpoint_field_digest",
        "residual_sha256",
        "birth_field_digest",
        "appended_initial_digest",
        "endpoint_field_digest",
    ):
        assert getattr(first, name) == getattr(second, name)
    assert first.trajectory_records() == second.trajectory_records()


def test_signed_residual_births_and_joint_fit_can_update_the_base_prefix():
    result = fit_progressive_additive_capacity(
        _fixtures()["edge"], config=replace(_config(), base_steps=2, joint_steps=2), seed=0
    )

    assert result.negative_birth_coefficients > 0
    prefix_changed = any(
        not torch.equal(
            getattr(result.base_field, name), getattr(result.field, name)[: result.base_count]
        )
        for name in ("means", "log_scales", "rotations", "colors")
    )
    assert prefix_changed
    assert result.base_endpoint_unchanged
    assert result.field.background_mask is None


def test_returned_endpoint_has_finite_gradients_for_all_parameter_groups():
    result = fit_progressive_additive_capacity(
        _fixtures()["texture"], config=_config(), seed=1
    )
    field = result.field.detached().trainable()
    rendered = _render(field, result.reconstruction_raw.shape[:2])
    rendered.square().mean().backward()

    for name in ("means", "log_scales", "rotations", "colors"):
        gradient = getattr(field, name).grad
        assert gradient is not None
        assert torch.isfinite(gradient).all()


@pytest.mark.parametrize("fixture", ["constant", "ramp", "edge", "blob", "texture"])
def test_procedural_killing_fixtures_complete_without_count_or_finite_failure(fixture):
    result = fit_progressive_additive_capacity(
        _fixtures()[fixture], config=_config(), seed=0
    )
    assert result.completed
    assert result.field.n == 6
    assert result.coefficient_abs_max <= 16.0
    assert np.isfinite(result.base_reconstruction_raw).all()
    assert np.isfinite(result.reconstruction_raw).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_owned_cuda_fit_cold_replay_and_cpu_renderer_parity(tmp_path):
    target = _fixtures()["edge"]
    config = replace(_config(), renderer="cuda_additive")
    result = fit_progressive_additive_capacity(
        target, config=config, seed=2, device="cuda"
    )
    assert result.completed

    path = tmp_path / "field.npz"
    result.field.save(str(path))
    cold_cuda = GaussianField.load(str(path), device="cuda")
    cuda_render = _render(cold_cuda, target.shape[:2], "cuda_additive").detach().cpu()
    cold_cpu = GaussianField.load(str(path), device="cpu")
    cpu_render = _render(cold_cpu, target.shape[:2], "additive").detach().cpu()

    assert torch.max(torch.abs(cuda_render - cpu_render)).item() <= 2e-4
    assert np.max(np.abs(cuda_render.numpy() - result.reconstruction_raw)) <= 2e-5


def _decision_rows(psnr_by_arm: dict[str, float]) -> list[dict[str, object]]:
    rows = []
    projected = h26.PROJECTED_ARMS
    progressive = h26.PROGRESSIVE_ARMS
    for image in ("0895", "0860", "0898", "0847"):
        for seed in (0, 1):
            shared_digest = f"shared-{image}-{seed}"
            for arm in h26.ARMS:
                pure = arm in h26.PURE_ADDITIVE_ARMS
                count = h26.COUNT_BY_ARM[arm]
                is_progressive = arm in progressive
                is_projected = arm in projected
                incoming_digest = f"incoming-{image}-{seed}-{arm}"
                rows.append(
                    {
                        "image": image,
                        "seed": seed,
                        "arm": arm,
                        "completed": True,
                        "method_status": "completed",
                        "n_gaussians": count,
                        "target_gaussians": count,
                        "finite_reconstruction": True,
                        "coefficient_abs_max": 1.0,
                        "maintained_render_parity_max_abs": 0.0,
                        "repeated_render_parity_max_abs": 0.0,
                        "endpoint_internal_parity_max_abs": 0.0,
                        "selected_lambda": 0.0 if pure else None,
                        "semantic_family": (
                            "additive_rgb_peak_one_v1"
                            if pure
                            else "normalized_weighted_sum_v1"
                        ),
                        "renderer": "cuda_additive" if pure else "cuda",
                        "four_array_endpoint_exact": True,
                        "mass_payload_present": False,
                        "denominator_payload_present": False,
                        "optimizer_payload_present": False,
                        "auxiliary_rgb_payload_present": False,
                        "training_payload_present": False,
                        "projection_selected": is_projected,
                        "projection_clauses": {"safe": True} if is_projected else {},
                        "final_field_digest": incoming_digest,
                        "incoming_field_digest": incoming_digest,
                        "base_shared_digest": (
                            shared_digest
                            if arm
                            in {
                                "additive_plain_n640",
                                "additive_projected_n640",
                                "progressive_residual_n896",
                                "progressive_residual_projected_n896",
                            }
                            else None
                        ),
                        "base_count": 640 if is_progressive else 0,
                        "residual_count": 256 if is_progressive else count,
                        "attempted_steps": 700 if is_progressive else 500,
                        "gaussian_row_updates": (
                            499_200 if is_progressive else count * 500
                        ),
                        "psnr_db": psnr_by_arm[arm],
                        "ms_ssim": 0.95,
                        "lpips": 0.1,
                        "artifact_pixel_rmse_max": 0.3,
                        "artifact_patch_rmse_max_7": 0.2,
                        "fit_seconds": 1.0,
                        "renderer_calls_fit": 1,
                    }
                )
    return rows


def test_frozen_decision_prefers_supported_progressive_n896_over_larger_rung():
    psnr = {arm: 30.0 for arm in h26.ARMS}
    psnr.update(
        {
            "additive_plain_n640": 29.5,
            "additive_projected_n640": 29.6,
            "cold_additive_projected_n896": 30.2,
            "progressive_residual_n896": 30.1,
            "progressive_residual_projected_n896": 30.3,
            "cold_additive_projected_n960": 30.6,
        }
    )
    decision = h26._decision(_decision_rows(psnr))

    assert decision["numeric_pass"]
    assert decision["progressive_mechanism_supported_numeric"]
    assert decision["numeric_selected_arm"] == "progressive_residual_projected_n896"
    assert not decision["same_count_additive_better_numeric"]


def test_frozen_decision_uses_robust_n960_only_when_both_n896_rungs_fail():
    psnr = {arm: 30.0 for arm in h26.ARMS}
    psnr.update(
        {
            "additive_plain_n640": 29.0,
            "additive_projected_n640": 29.1,
            "cold_additive_projected_n896": 29.5,
            "progressive_residual_n896": 29.4,
            "progressive_residual_projected_n896": 29.6,
            "cold_additive_projected_n960": 30.2,
        }
    )
    decision = h26._decision(_decision_rows(psnr))

    assert decision["numeric_pass"]
    assert decision["robust_n960_numeric"]
    assert decision["numeric_selected_arm"] == "cold_additive_projected_n960"
