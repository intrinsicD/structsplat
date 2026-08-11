from __future__ import annotations

from scripts.experiments import hier021_replay_persisted as replay
from scripts.experiments import hier021_source_patch_tail as h21


def _metrics(
    *,
    mse: float,
    psnr: float,
    ms_ssim: float,
    lpips: float,
    pixel: float,
    patch: float,
) -> dict[str, float]:
    return {
        "masked_mse": mse,
        "psnr_db": psnr,
        "ssim": ms_ssim - 0.01,
        "ms_ssim": ms_ssim,
        "lpips": lpips,
        "artifact_pixel_rmse_max": pixel,
        "artifact_patch_rmse_max_7": patch,
    }


def _rows() -> list[dict[str, object]]:
    groups = ["consumed_h15_h19"] * 20 + ["hier020_fresh"] * 4 + [
        "tests_test_images"
    ] * 16
    rows: list[dict[str, object]] = []
    for index, group in enumerate(groups):
        image = f"persisted_{index:02d}"
        lineage = f"lineage_{index:02d}"
        baseline_pixel = 0.81 if index in (2, 25) else 0.70
        baseline_patch = 0.41 if index == 25 else 0.30
        rows.append(
            {
                "source_replay_group": group,
                "source_lineage": lineage,
                "image": image,
                "arm": h21.CONTROL_ARM,
                "n_gaussians": 7000,
                **_metrics(
                    mse=0.10,
                    psnr=20.0,
                    ms_ssim=0.80,
                    lpips=0.20,
                    pixel=0.80,
                    patch=0.40,
                ),
            }
        )
        rows.append(
            {
                "source_replay_group": group,
                "source_lineage": lineage,
                "image": image,
                "arm": h21.DIRECT_ARM,
                "n_gaussians": 7000,
                "persisted_field_reused_without_refit": True,
                **_metrics(
                    mse=0.010,
                    psnr=30.0,
                    ms_ssim=0.90,
                    lpips=0.10,
                    pixel=baseline_pixel,
                    patch=baseline_patch,
                ),
                "spt1_field_file_sha256_before": image,
                "spt1_field_file_sha256_after": image,
                "spt1_payload_roundtrip_exact": True,
                "spt1_candidate_finite": True,
                "spt1_outside_identity_max_abs": 0.0,
                "spt1_baseline_cold_parity_max_abs": 1e-6,
                "spt1_candidate_repeated_parity_max_abs": 1e-6,
                "spt1_selected_mode": h21.CANDIDATE_MODE,
                "spt1_selected_count": 49,
                "spt1_candidate_payload_bytes": 359,
                "spt1_selected_payload_bytes": 359,
                "spt1_pipeline_time_ratio": 1.01,
                "spt1_decode_time_ratio": 1.10,
                "spt1_selection_clauses": {"safe": True},
                "spt1_metric_deltas_vs_baseline": {
                    "mse_ratio": 0.90,
                    "ms_ssim_delta": 0.01,
                    "lpips_delta": -0.01,
                    "pixel_max_delta": 0.70 - baseline_pixel,
                    "patch7_max_delta": 0.30 - baseline_patch,
                },
                "spt1_selected_metrics": _metrics(
                    mse=0.009,
                    psnr=30.45,
                    ms_ssim=0.91,
                    lpips=0.09,
                    pixel=0.70,
                    patch=0.30,
                ),
            }
        )
    return rows


def test_source_specs_freeze_twenty_plus_four_plus_sixteen_fields() -> None:
    assert [spec["direct_count"] for spec in replay.SOURCE_SPECS] == [20, 4, 16]
    assert sum(int(spec["direct_count"]) for spec in replay.SOURCE_SPECS) == 40


def test_decision_requires_all_fields_and_repairs_recorded_local_failures() -> None:
    rows = _rows()
    attempts = [{"status": "ok"}] * 80

    pending = replay._decision(rows, attempts, visual_disposition="pending")
    reviewed = replay._decision(rows, attempts, visual_disposition="pass")

    assert all(pending["gates"].values())
    assert pending["baseline_local_failure_count"] == 2
    assert pending["numeric_bank_pass"]
    assert not pending["bounded_bank_pass"]
    assert reviewed["bounded_bank_pass"]


def test_test_image_h005_regression_rejects_the_replay() -> None:
    rows = _rows()
    test_control = next(
        row
        for row in rows
        if row["source_replay_group"] == "tests_test_images"
        and row["arm"] == h21.CONTROL_ARM
    )
    test_control["lpips"] = 0.05

    decision = replay._decision(rows, [{"status": "ok"}] * 80, visual_disposition="pass")

    assert not decision["gates"]["all_tests_lpips_noninferior_vs_h005"]
    assert not decision["numeric_bank_pass"]
    assert not decision["bounded_bank_pass"]
