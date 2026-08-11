from __future__ import annotations

from argparse import Namespace
import math

from scripts.experiments import hier014_conditioned_projection as hier014


def _args(phase: str = "kodak") -> Namespace:
    return Namespace(
        phase=phase,
        projection_ridge=1e-8,
        projection_tolerance=1e-6,
        projection_max_iterations=96,
        coefficient_limit=16.0,
    )


def _synthetic_rows(*, local_regression: bool = False):
    rows = []
    ratios = {
        "h005_control": 1.0,
        "legacy_input_subtract": 1.0,
        "origin_subtract": 0.85,
        "origin_explicit": 0.80,
    }
    for image_index, image in enumerate(("kodim01", "kodim07", "kodim13", "kodim19")):
        for arm in hier014.KODAK_ARMS:
            ratio = ratios[arm]
            pixel = 0.10 if arm == "h005_control" else 0.08
            if local_regression and arm == "origin_explicit" and image_index == 0:
                pixel = 0.101
            rows.append(
                {
                    "image": image,
                    "arm": arm,
                    "masked_mse": ratio,
                    "psnr_db": 30.0 - 10.0 * math.log10(ratio),
                    "ms_ssim": 0.90 + 0.01 * (1.0 - ratio),
                    "lpips": 0.10 - 0.01 * (1.0 - ratio),
                    "artifact_pixel_rmse_max": pixel,
                    "artifact_patch_rmse_max_7": 0.05 if arm == "h005_control" else 0.04,
                    "projection_selected_iteration": 0 if arm == "h005_control" else 8,
                    "coefficient_abs_max": 1.0,
                    "maintained_render_parity_max_abs": 1e-7,
                    "repeated_render_parity_max_abs": 1e-7,
                    "projection_overhead_ratio": 0.01,
                    "non_rgb_arrays_bit_exact": True,
                    "n_gaussians": 7000,
                }
            )
    return rows


def test_projection_configs_isolate_restart_and_explicit_base() -> None:
    legacy = hier014._projection_config(_args(), "legacy_input_subtract")
    subtract = hier014._projection_config(_args(), "origin_subtract")
    explicit = hier014._projection_config(_args(), "origin_explicit")

    assert legacy.regularization_center == legacy.solver_start == "input"
    assert legacy.frozen_base_mode == "subtract"
    assert not legacy.allow_unsafe_stage_zero_reconditioning
    assert subtract.regularization_center == subtract.solver_start == "zero"
    assert subtract.frozen_base_mode == "subtract"
    assert subtract.allow_unsafe_stage_zero_reconditioning
    assert explicit.regularization_center == explicit.solver_start == "zero"
    assert explicit.frozen_base_mode == "explicit"
    assert explicit.allow_unsafe_stage_zero_reconditioning


def test_kodak_decision_accepts_a_safe_material_effect() -> None:
    decision = hier014._aggregate(_synthetic_rows(), _args())

    assert decision["decision"] == "pass"
    assert all(decision["gate_predicates"].values())
    assert decision["arm_aggregates"]["origin_explicit"][
        "geometric_mean_mse_ratio"
    ] == 0.8
    assert decision["arm_aggregates"]["origin_explicit"]["nonzero_solves"] == 4


def test_kodak_decision_fails_one_local_regression() -> None:
    decision = hier014._aggregate(_synthetic_rows(local_regression=True), _args())

    assert decision["decision"] == "fail"
    assert not decision["gate_predicates"]["all_pixel_max_noninferior"]


def test_kodak_source_bank_is_exactly_four_hash_bound_images() -> None:
    assert tuple(hier014.KODAK_BINDINGS) == (
        "kodim01.png",
        "kodim07.png",
        "kodim13.png",
        "kodim19.png",
    )
    assert all(len(digest) == 64 for digest in hier014.KODAK_BINDINGS.values())
