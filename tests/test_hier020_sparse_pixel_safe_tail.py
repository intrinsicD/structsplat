from __future__ import annotations

from argparse import Namespace
from dataclasses import asdict
import hashlib

from scripts.experiments import hier020_sparse_pixel_safe_tail as hier020


def _args(*, phase: str = "development") -> Namespace:
    return Namespace(
        phase=phase,
        target_gaussians=7000,
        max_side=512,
        seed=0,
        direct_fit_steps=750,
        device="cuda",
        additive_renderer="cuda_additive",
        direct_renderer="cuda",
        render_chunk=256,
        lpips=True,
        tail_scale_multiplier=2.0,
        tail_coverage_threshold=1e-8,
        error_scale=4.0,
        development_decision=None,
        recover_from=None,
        visual_disposition="pass",
    )


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


def _control(image: str) -> dict[str, object]:
    return {
        "image": image,
        "arm": hier020.CONTROL_ARM,
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


def _direct(image: str, *, selected_pixel: float = 0.70) -> dict[str, object]:
    selected = _metrics(
        mse=0.009,
        psnr=30.45,
        ms_ssim=0.91,
        lpips=0.09,
        pixel=selected_pixel,
        patch=0.30,
    )
    return {
        "image": image,
        "arm": hier020.DIRECT_ARM,
        "n_gaussians": 7000,
        **_metrics(
            mse=0.010,
            psnr=30.0,
            ms_ssim=0.90,
            lpips=0.10,
            pixel=0.81,
            patch=0.30,
        ),
        "sst1_field_file_sha256_before": f"field-{image}",
        "sst1_field_file_sha256_after": f"field-{image}",
        "sst1_payload_roundtrip_exact": True,
        "sst1_candidate_finite": True,
        "sst1_outside_identity_max_abs": 0.0,
        "sst1_baseline_cold_parity_max_abs": 1e-6,
        "sst1_candidate_repeated_parity_max_abs": 1e-6,
        "sst1_selected_mode": hier020.CANDIDATE_MODE,
        "sst1_selected_count": 2,
        "sst1_selected_metrics": selected,
        "sst1_candidate_payload_bytes": 24,
        "sst1_selected_payload_bytes": 24,
        "sst1_pipeline_time_ratio": 1.05,
        "sst1_decode_time_ratio": 2.5,
        "sst1_selection_clauses": {
            "candidate_finite": True,
            "outside_payload_bit_exact": True,
            "baseline_cold_parity_le_2e_5": True,
            "candidate_repeated_parity_le_2e_5": True,
            "payload_roundtrip_exact": True,
            "pointwise_raw_strict": True,
            "pointwise_display_noninferior": True,
            "mse_noninferior": True,
            "pixel_max_noninferior": True,
            "patch7_max_noninferior": True,
            "ms_ssim_noninferior": True,
            "lpips_noninferior": True,
        },
        "sst1_metric_deltas_vs_baseline": {
            "mse_ratio": 0.9,
            "psnr_delta_db": 0.45,
            "ms_ssim_delta": 0.01,
            "lpips_delta": -0.01,
            "pixel_max_delta": selected_pixel - 0.81,
            "patch7_max_delta": 0.0,
        },
    }


def test_prospective_bindings_match_frozen_salted_filename_hashes() -> None:
    assert len(hier020.DEVELOPMENT_BINDINGS) == 4
    assert tuple(hier020.DEVELOPMENT_BINDINGS) == tuple(hier020.SELECTION_DIGESTS)
    for name, digest in hier020.SELECTION_DIGESTS.items():
        assert hashlib.sha256(("HIER-020-v1:" + name).encode()).hexdigest() == digest


def test_configs_freeze_direct_fit_and_tail_geometry() -> None:
    init, fit, tail = hier020._configs(_args())

    assert init.num_gaussians == 7000
    assert init.background_fraction == 0.0
    assert fit.normalization_eps == 1e-8
    assert fit.renderer == "cuda"
    assert asdict(tail) == {
        "scale_multiplier": 2.0,
        "coverage_threshold": 1e-8,
    }


def test_selection_requires_pointwise_and_global_safety() -> None:
    baseline = _metrics(
        mse=0.1,
        psnr=30.0,
        ms_ssim=0.95,
        lpips=0.1,
        pixel=0.5,
        patch=0.2,
    )
    candidate = _metrics(
        mse=0.09,
        psnr=30.45,
        ms_ssim=0.951,
        lpips=0.09,
        pixel=0.45,
        patch=0.18,
    )
    kwargs = {
        "selected_count": 2,
        "finite": True,
        "outside_identity_max_abs": 0.0,
        "baseline_parity_max_abs": 1e-6,
        "repeated_parity_max_abs": 1e-6,
        "payload_roundtrip_exact": True,
        "pointwise_raw_delta_max": -1e-6,
        "pointwise_display_delta_max": 0.0,
    }

    selected, clauses, _ = hier020._selection(baseline, candidate, **kwargs)
    assert selected == hier020.CANDIDATE_MODE
    assert all(clauses.values())

    tied, _, _ = hier020._selection(baseline, baseline, **kwargs)
    assert tied == hier020.BASELINE_MODE

    rejected, clauses, _ = hier020._selection(
        baseline, candidate, **{**kwargs, "pointwise_display_delta_max": 1.0}
    )
    assert rejected == hier020.BASELINE_MODE
    assert not clauses["pointwise_display_noninferior"]


def test_development_gate_accepts_complete_safe_portfolio_and_repairs_local_case() -> None:
    rows: list[dict[str, object]] = []
    for index in range(4):
        image = f"coco_{index}"
        rows.extend((_control(image), _direct(image)))
    decision = hier020._decision(rows, [{"status": "ok"}] * 8, _args())

    assert all(decision["gates"].values())
    assert decision["baseline_local_failure_count"] == 4
    assert decision["numeric_candidates"] == [hier020.CANDIDATE_MODE]


def test_one_unrepaired_local_case_rejects_portfolio() -> None:
    rows: list[dict[str, object]] = []
    for index in range(4):
        image = f"coco_{index}"
        rows.extend(
            (_control(image), _direct(image, selected_pixel=0.82 if index == 0 else 0.70))
        )
    decision = hier020._decision(rows, [{"status": "ok"}] * 8, _args())

    assert not decision["gates"]["all_pixel_max_noninferior_vs_h005"]
    assert not decision["gates"]["all_baseline_local_failures_repaired"]
    assert decision["numeric_candidates"] == []


def test_tests_replay_requires_all_sixteen_pairs() -> None:
    rows: list[dict[str, object]] = []
    for index in range(16):
        image = f"test_{index}"
        rows.extend((_control(image), _direct(image)))
    decision = hier020._decision(
        rows, [{"status": "ok"}] * 32, _args(phase="replay_tests")
    )

    assert decision["gates"]["complete_h005_pairs"]
    assert decision["bounded_bank_pass"]
