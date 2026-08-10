from __future__ import annotations

import math

from scripts.experiments import hier013_global_projection_development as hier013


def _synthetic_rows(*, local_regression: bool = False):
    rows = []
    attempts = []
    for relative in sorted(hier013.EXPECTED_SOURCES):
        family = "DIV2K" if "DIV2K_train_HR" in relative else "COCO"
        stem = relative.rsplit("/", 1)[-1].rsplit(".", 1)[0]
        source_image = f"{family}/{stem}"
        for seed in hier013.SEEDS:
            for arm in hier013.ARMS:
                ratio = {
                    "h005_control": 1.0,
                    "touched_projection": 0.98,
                    "global_projection": 0.70,
                    "exchange_global_projection": 0.72,
                }[arm]
                psnr_delta = -10.0 * math.log10(ratio)
                pixel = {
                    "h005_control": 0.10,
                    "touched_projection": 0.095,
                    "global_projection": 0.080,
                    "exchange_global_projection": 0.085,
                }[arm]
                if local_regression and arm == "global_projection" and seed == 0:
                    pixel = 0.101
                patch = {
                    "h005_control": 0.050,
                    "touched_projection": 0.049,
                    "global_projection": 0.040,
                    "exchange_global_projection": 0.045,
                }[arm]
                lpips = {
                    "h005_control": 0.10,
                    "touched_projection": 0.098,
                    "global_projection": 0.070,
                    "exchange_global_projection": 0.075,
                }[arm]
                rows.append(
                    {
                        "source_image": source_image,
                        "family": family,
                        "seed": seed,
                        "arm": arm,
                        "masked_mse": ratio,
                        "psnr_db": 30.0 + psnr_delta,
                        "ms_ssim": 0.90 + 0.01 * (1.0 - ratio),
                        "lpips": lpips,
                        "artifact_pixel_rmse_max": pixel,
                        "artifact_patch_rmse_max_7": patch,
                        "pipeline_algorithm_seconds": 1.0 + (0.1 if arm != "h005_control" else 0.0),
                        "projection_seconds": 0.0 if arm == "h005_control" else 0.1,
                        "complete_reference_stream_bytes": 1000,
                        "n_gaussians": 7000,
                        "non_rgb_arrays_bit_exact": True,
                        "maintained_render_parity_max_abs": 0.0,
                        "repeated_render_parity_max_abs": 0.0,
                        "projection_internal_render_parity_max_abs": 0.0,
                        "projection_adjoint_relative_error": 0.0,
                        "projection_transaction_pass": True,
                        "projection_coefficient_abs_max": 1.0,
                        "exchange_internal_render_parity_max_abs": 0.0,
                        "exchange_pricing_error_max_abs": 0.0,
                    }
                )
                attempts.append(
                    {
                        "status": "ok",
                        "image": source_image,
                        "family": family,
                        "seed": seed,
                        "arm": arm,
                    }
                )
    return rows, attempts


def test_source_set_digest_is_frozen() -> None:
    assert len(hier013.EXPECTED_SOURCES) == 16
    assert hier013._source_set_sha256(hier013.EXPECTED_SOURCES) == (
        hier013.EXPECTED_SOURCE_SET_SHA256
    )


def test_decision_accepts_large_safe_effect_and_prefers_simpler_global() -> None:
    rows, attempts = _synthetic_rows()
    aggregates, decision = hier013._decision_from_rows(rows, attempts)

    assert decision["development_gate_pass"] is True
    assert decision["global_vs_exchange_selection"] == "global_projection"
    assert decision["successful_cells"] == 192
    assert decision["bootstrap"]["mse_ratio_high_95"] < 0.71
    assert len(aggregates["global_projection_image_means"]) == 16


def test_decision_fails_a_single_local_max_regression() -> None:
    rows, attempts = _synthetic_rows(local_regression=True)
    _aggregates, decision = hier013._decision_from_rows(rows, attempts)

    assert decision["development_gate_pass"] is False
    assert decision["clauses"]["no_local_max_regression"] is False
