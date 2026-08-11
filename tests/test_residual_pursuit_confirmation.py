# ruff: noqa: E402
import hashlib

import pytest

pytest.importorskip("torch")

from scripts.experiments import hier028_residual_pursuit_additive as h28


def _decision_rows(
    psnr_by_arm: dict[str, float],
    *,
    pursuit_pixel: float = 0.20,
    base_pixel: float = 0.30,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for filename in h28.SELECTION_ORDER:
        image = filename.removesuffix(".png")
        for seed in (0, 1):
            shared_base = f"projected-n960-{image}-{seed}"
            for arm in h28.ARMS:
                count = h28.COUNT_BY_ARM[arm]
                pure = arm in h28.PURE_ADDITIVE_ARMS
                projected = pure
                incoming = f"incoming-{image}-{seed}-{arm}"
                is_pursuit = arm == "residual_pursuit_additive_n1024"
                pixel = (
                    pursuit_pixel
                    if is_pursuit
                    else base_pixel
                    if arm == "cold_additive_projected_n960"
                    else 0.25
                )
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
                        "endpoint_internal_parity_max_abs": 0.0,
                        "maintained_render_parity_max_abs": 0.0,
                        "repeated_render_parity_max_abs": 0.0,
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
                        "projection_selected": projected,
                        "projection_clauses": {"safe": True} if projected else {},
                        "incoming_field_digest": incoming,
                        "base_projection_final_digest": (
                            shared_base if is_pursuit else incoming
                        ),
                        "final_field_digest": (
                            shared_base
                            if arm == "cold_additive_projected_n960"
                            else f"final-{image}-{seed}-{arm}"
                        ),
                        "pursuit_applied": is_pursuit,
                        "pursuit_base_count": 960 if is_pursuit else None,
                        "pursuit_tail_count": 64 if is_pursuit else None,
                        "pursuit_base_prefix_bit_exact": is_pursuit or None,
                        "pursuit_fixed_tail_geometry": is_pursuit or None,
                        "pursuit_analytic_render_parity_max_abs": (
                            0.0 if is_pursuit else None
                        ),
                        "pursuit_base_field_digest": (
                            shared_base if is_pursuit else None
                        ),
                        "attempted_steps": 500,
                        "gaussian_row_updates": h28.GAUSSIAN_ROW_UPDATES_BY_ARM[arm],
                        "psnr_db": psnr_by_arm[arm],
                        "ms_ssim": 0.95,
                        "lpips": 0.10,
                        "artifact_pixel_rmse_max": pixel,
                        "artifact_patch_rmse_max_7": 0.15,
                        "fit_seconds": 1.0,
                        "pursuit_seconds": 0.1 if is_pursuit else 0.0,
                    }
                )
    return rows


def _passing_psnr() -> dict[str, float]:
    return {
        "normalized_plain_n640": 30.0,
        "cold_additive_projected_n960": 30.2,
        "residual_pursuit_additive_n1024": 30.8,
        "cold_additive_projected_n1024": 29.0,
    }


def test_frozen_selection_bindings_counts_and_work_are_exact():
    assert len(h28.SELECTION_ORDER) == len(h28.SOURCE_BINDINGS) == 8
    assert set(h28.SELECTION_ORDER) == set(h28.SOURCE_BINDINGS)
    assert set(h28.SELECTION_ORDER) == set(h28.SELECTION_BINDINGS)
    assert tuple(
        sorted(
            h28.SELECTION_ORDER,
            key=lambda name: hashlib.sha256(
                f"{h28.SELECTION_SALT}{name}".encode()
            ).hexdigest(),
        )
    ) == h28.SELECTION_ORDER
    assert all(
        hashlib.sha256(f"{h28.SELECTION_SALT}{name}".encode()).hexdigest()
        == h28.SELECTION_BINDINGS[name]
        for name in h28.SELECTION_ORDER
    )

    rows = _decision_rows(_passing_psnr())
    assert len(rows) == 8 * 2 * 4 == 64
    assert all(
        row["gaussian_row_updates"]
        == h28.GAUSSIAN_ROW_UPDATES_BY_ARM[str(row["arm"])]
        for row in rows
    )


def test_decision_selects_robust_pursuit_with_exact_shared_base():
    decision = h28._decision(_decision_rows(_passing_psnr()))

    assert decision["numeric_pass"]
    assert decision["pursuit_robust_numeric"]
    assert decision["numeric_selected_arm"] == "residual_pursuit_additive_n1024"
    assert decision["gates"]["pursuit_contract_exact"]
    assert decision["gates"]["shared_projected_n960_base_exact"]
    assert decision["gates"]["pursuit_local_nonregression_vs_base"]
    assert not decision["same_count_cold_numeric"]


def test_decision_rejects_pursuit_local_regression_without_relaxation():
    rows = _decision_rows(_passing_psnr(), pursuit_pixel=0.31, base_pixel=0.30)
    decision = h28._decision(rows)

    assert not decision["numeric_pass"]
    assert not decision["pursuit_robust_numeric"]
    assert not decision["gates"]["pursuit_local_nonregression_vs_base"]
    assert decision["numeric_selected_arm"] is None


def test_same_count_cold_control_cannot_substitute_for_pursuit():
    psnr = _passing_psnr()
    psnr["residual_pursuit_additive_n1024"] = 29.0
    psnr["cold_additive_projected_n1024"] = 30.8
    decision = h28._decision(_decision_rows(psnr))

    assert decision["same_count_cold_numeric"]
    assert not decision["numeric_pass"]
    assert decision["numeric_selected_arm"] is None
