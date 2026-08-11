# ruff: noqa: E402
import hashlib

import pytest

pytest.importorskip("torch")

from scripts.experiments import hier027_cold_additive_capacity as h27


def _decision_rows(psnr_by_arm: dict[str, float]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for filename in h27.SELECTION_ORDER:
        image = filename.removesuffix(".png")
        for seed in (0, 1):
            shared_n640 = f"n640-{image}-{seed}"
            shared_n1088 = f"n1088-{image}-{seed}"
            for arm in h27.ARMS:
                count = h27.COUNT_BY_ARM[arm]
                pure = arm in h27.PURE_ADDITIVE_ARMS
                projected = arm in h27.PROJECTED_ARMS
                incoming = f"incoming-{image}-{seed}-{arm}"
                if arm in {"additive_plain_n640", "additive_projected_n640"}:
                    endpoint = shared_n640
                elif arm in {
                    "cold_additive_plain_n1088",
                    "cold_additive_projected_n1088",
                }:
                    endpoint = shared_n1088
                else:
                    endpoint = f"endpoint-{image}-{seed}-{arm}"
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
                        "final_field_digest": incoming,
                        "preprojection_endpoint_digest": endpoint,
                        "attempted_steps": 500,
                        "gaussian_row_updates": count * 500,
                        "psnr_db": psnr_by_arm[arm],
                        "ms_ssim": 0.95,
                        "lpips": 0.10,
                        "artifact_pixel_rmse_max": 0.30,
                        "artifact_patch_rmse_max_7": 0.20,
                        "fit_seconds": 1.0,
                    }
                )
    return rows


def _psnr_values() -> dict[str, float]:
    return {arm: 30.0 for arm in h27.ARMS}


def test_frozen_matrix_bindings_counts_and_work_are_exact():
    assert len(h27.SELECTION_ORDER) == len(h27.SOURCE_BINDINGS) == 8
    assert set(h27.SELECTION_ORDER) == set(h27.SOURCE_BINDINGS)
    assert set(h27.SELECTION_ORDER) == set(h27.SELECTION_BINDINGS)
    assert tuple(
        sorted(
            h27.SELECTION_ORDER,
            key=lambda name: hashlib.sha256(
                f"{h27.SELECTION_SALT}{name}".encode()
            ).hexdigest(),
        )
    ) == h27.SELECTION_ORDER
    assert all(
        hashlib.sha256(f"{h27.SELECTION_SALT}{name}".encode()).hexdigest()
        == h27.SELECTION_BINDINGS[name]
        for name in h27.SELECTION_ORDER
    )

    rows = _decision_rows(_psnr_values())
    assert len(rows) == 8 * 2 * 7 == 112
    assert {
        arm: {row["n_gaussians"] for row in rows if row["arm"] == arm}
        for arm in h27.ARMS
    } == {arm: {count} for arm, count in h27.COUNT_BY_ARM.items()}
    assert all(
        row["attempted_steps"] == 500
        and row["gaussian_row_updates"]
        == h27.COUNT_BY_ARM[str(row["arm"])] * 500
        for row in rows
    )


def test_decision_selects_primary_n1088_and_requires_shared_endpoints():
    psnr = _psnr_values()
    psnr.update(
        {
            "additive_plain_n640": 29.0,
            "additive_projected_n640": 29.1,
            "cold_additive_projected_n1024": 30.1,
            "cold_additive_plain_n1088": 30.1,
            "cold_additive_projected_n1088": 30.2,
            "cold_additive_projected_n1152": 29.0,
        }
    )
    rows = _decision_rows(psnr)
    decision = h27._decision(rows)

    assert decision["numeric_pass"]
    assert decision["primary_n1088_numeric"]
    assert decision["numeric_selected_arm"] == "cold_additive_projected_n1088"
    assert decision["gates"]["shared_preprojection_endpoints_exact"]

    broken = [dict(row) for row in rows]
    broken[broken.index(next(row for row in broken if row["arm"] == "additive_projected_n640"))][
        "preprojection_endpoint_digest"
    ] = "different"
    rejected = h27._decision(broken)
    assert not rejected["numeric_pass"]
    assert not rejected["gates"]["shared_preprojection_endpoints_exact"]


def test_decision_uses_n1152_only_as_robust_fallback():
    psnr = _psnr_values()
    psnr.update(
        {
            "additive_plain_n640": 29.0,
            "additive_projected_n640": 29.1,
            "cold_additive_projected_n1024": 30.2,
            "cold_additive_plain_n1088": 29.0,
            "cold_additive_projected_n1088": 29.0,
            "cold_additive_projected_n1152": 30.6,
        }
    )
    decision = h27._decision(_decision_rows(psnr))

    assert decision["numeric_pass"]
    assert not decision["primary_n1088_numeric"]
    assert decision["fallback_n1152_numeric"]
    assert decision["numeric_selected_arm"] == "cold_additive_projected_n1152"


def test_n1024_boundary_cannot_be_selected():
    psnr = _psnr_values()
    psnr.update(
        {
            "additive_plain_n640": 29.0,
            "additive_projected_n640": 29.1,
            "cold_additive_projected_n1024": 30.2,
            "cold_additive_plain_n1088": 29.0,
            "cold_additive_projected_n1088": 29.0,
            "cold_additive_projected_n1152": 29.0,
        }
    )
    decision = h27._decision(_decision_rows(psnr))

    assert decision["boundary_n1024_numeric_only"]
    assert not decision["numeric_pass"]
    assert decision["numeric_selected_arm"] is None
