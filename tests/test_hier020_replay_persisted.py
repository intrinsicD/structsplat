from __future__ import annotations

from scripts.experiments import hier020_replay_persisted as replay
from scripts.experiments import hier020_sparse_pixel_safe_tail as h20


def _metrics(
    mse: float, ms_ssim: float, lpips: float, pixel: float, patch: float
) -> dict[str, float]:
    return {
        "masked_mse": mse,
        "psnr_db": 30.0,
        "ssim": ms_ssim - 0.01,
        "ms_ssim": ms_ssim,
        "lpips": lpips,
        "artifact_pixel_rmse_max": pixel,
        "artifact_patch_rmse_max_7": patch,
    }


def _rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for lineage, _, _ in replay.LINEAGES:
        for index in range(4):
            image = f"{lineage}_{index}"
            rows.append(
                {
                    "image": image,
                    "arm": h20.CONTROL_ARM,
                    "n_gaussians": 7000,
                    **_metrics(0.1, 0.8, 0.2, 0.8, 0.4),
                }
            )
            rows.append(
                {
                    "image": image,
                    "arm": h20.DIRECT_ARM,
                    "source_lineage": lineage,
                    "n_gaussians": 7000,
                    "persisted_field_reused_without_refit": True,
                    **_metrics(0.01, 0.9, 0.1, 0.81, 0.3),
                    "sst1_field_file_sha256_before": image,
                    "sst1_field_file_sha256_after": image,
                    "sst1_payload_roundtrip_exact": True,
                    "sst1_candidate_finite": True,
                    "sst1_outside_identity_max_abs": 0.0,
                    "sst1_baseline_cold_parity_max_abs": 1e-6,
                    "sst1_candidate_repeated_parity_max_abs": 1e-6,
                    "sst1_optimized_decode_parity_max_abs": 1e-6,
                    "sst1_selected_mode": h20.CANDIDATE_MODE,
                    "sst1_selected_count": 2,
                    "sst1_candidate_payload_bytes": 24,
                    "sst1_selected_payload_bytes": 24,
                    "sst1_selected_metrics": _metrics(0.009, 0.91, 0.09, 0.7, 0.3),
                    "sst1_selection_clauses": {"safe": True},
                    "sst1_metric_deltas_vs_baseline": {
                        "mse_ratio": 0.9,
                        "ms_ssim_delta": 0.01,
                        "lpips_delta": -0.01,
                        "pixel_max_delta": -0.11,
                        "patch7_max_delta": 0.0,
                    },
                }
            )
    return rows


def test_lineage_contract_is_five_disjoint_four_image_banks() -> None:
    assert len(replay.LINEAGES) == 5
    assert len({lineage for lineage, _, _ in replay.LINEAGES}) == 5
    assert len({schema for _, schema, _ in replay.LINEAGES}) == 5


def test_consumed_decision_requires_all_twenty_and_repairs_local_failures() -> None:
    decision = replay._decision(
        _rows(), [{"status": "ok"}] * 40, visual_disposition="pass"
    )

    assert all(decision["gates"].values())
    assert decision["baseline_local_failure_count"] == 20
    assert decision["selected_pixel_count"] == 40
    assert decision["bounded_bank_pass"]


def test_visual_pending_keeps_numeric_pass_but_not_bounded_pass() -> None:
    decision = replay._decision(
        _rows(), [{"status": "ok"}] * 40, visual_disposition="pending"
    )

    assert decision["numeric_bank_pass"]
    assert not decision["bounded_bank_pass"]
