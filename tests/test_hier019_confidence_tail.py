from __future__ import annotations

from argparse import Namespace
from dataclasses import asdict
import hashlib

from scripts.experiments import hier019_confidence_tail as hier019


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
        control_metrics=None,
        recover_from=None,
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
        "arm": hier019.CONTROL_ARM,
        "n_gaussians": 7000,
        **_metrics(
            mse=1.0,
            psnr=20.0,
            ms_ssim=0.70,
            lpips=0.60,
            pixel=0.80,
            patch=0.40,
        ),
    }


def _direct(image: str, *, h005_regression: bool = False) -> dict[str, object]:
    selected = _metrics(
        mse=0.09,
        psnr=30.45,
        ms_ssim=0.951,
        lpips=0.09,
        pixel=0.45,
        patch=0.18,
    )
    if h005_regression:
        selected["artifact_pixel_rmse_max"] = 0.81
    return {
        "image": image,
        "arm": hier019.DIRECT_ARM,
        "n_gaussians": 7000,
        **_metrics(
            mse=0.10,
            psnr=30.0,
            ms_ssim=0.95,
            lpips=0.10,
            pixel=0.50,
            patch=0.20,
        ),
        "tail_field_file_sha256_before": f"field-{image}",
        "tail_field_file_sha256_after": f"field-{image}",
        "tail_selected_mode": hier019.CANDIDATE_MODE,
        "tail_selected_metrics": selected,
        "tail_activation_count": 3,
        "tail_candidate_finite": True,
        "tail_outside_identity_max_abs": 0.0,
        "tail_baseline_cold_parity_max_abs": 1e-6,
        "tail_candidate_repeated_parity_max_abs": 1e-6,
        "tail_pipeline_time_ratio": 1.05,
        "tail_render_time_ratio": 3.0,
        "tail_selection_clauses": {
            "candidate_finite": True,
            "outside_activation_bit_exact": True,
            "baseline_cold_parity_le_2e_5": True,
            "candidate_repeated_parity_le_2e_5": True,
            "mse_noninferior": True,
            "pixel_max_noninferior": True,
            "patch7_max_noninferior": True,
            "ms_ssim_noninferior": True,
            "lpips_noninferior": True,
        },
        "tail_metric_deltas_vs_baseline": {
            "mse_ratio": 0.9,
            "psnr_delta_db": 0.45,
            "ms_ssim_delta": 0.001,
            "lpips_delta": -0.01,
            "pixel_max_delta": -0.05,
            "patch7_max_delta": -0.02,
        },
    }


def test_prospective_bindings_match_frozen_salted_filename_hashes() -> None:
    assert len(hier019.DEVELOPMENT_BINDINGS) == 4
    assert tuple(hier019.DEVELOPMENT_BINDINGS) == tuple(hier019.SELECTION_DIGESTS)
    for name, digest in hier019.SELECTION_DIGESTS.items():
        assert hashlib.sha256(("HIER-019-v1:" + name).encode()).hexdigest() == digest


def test_configs_freeze_ordinary_direct_fit_and_one_octave_tail() -> None:
    init, fit, tail = hier019._configs(_args())

    assert init.num_gaussians == 7000
    assert init.background_fraction == 0.0
    assert fit.normalization_eps == 1e-8
    assert fit.renderer == "cuda"
    assert asdict(tail) == {
        "scale_multiplier": 2.0,
        "coverage_threshold": 1e-8,
    }


def test_selection_requires_nonregression_and_a_material_improvement() -> None:
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
    selected, clauses, _ = hier019._selection(
        baseline,
        candidate,
        activation_count=2,
        finite=True,
        outside_identity_max_abs=0.0,
        baseline_parity_max_abs=1e-6,
        repeated_parity_max_abs=1e-6,
    )
    assert all(clauses.values())
    assert selected == hier019.CANDIDATE_MODE

    tied, _, _ = hier019._selection(
        baseline,
        baseline,
        activation_count=2,
        finite=True,
        outside_identity_max_abs=0.0,
        baseline_parity_max_abs=1e-6,
        repeated_parity_max_abs=1e-6,
    )
    assert tied == hier019.BASELINE_MODE

    candidate["lpips"] = 0.11
    rejected, clauses, _ = hier019._selection(
        baseline,
        candidate,
        activation_count=2,
        finite=True,
        outside_identity_max_abs=0.0,
        baseline_parity_max_abs=1e-6,
        repeated_parity_max_abs=1e-6,
    )
    assert not clauses["lpips_noninferior"]
    assert rejected == hier019.BASELINE_MODE


def test_development_gate_accepts_complete_safe_portfolio() -> None:
    rows: list[dict[str, object]] = []
    for index in range(4):
        image = f"coco_{index}"
        rows.extend((_control(image), _direct(image)))
    decision = hier019._decision(rows, [{"status": "ok"}] * 8, _args())

    assert all(decision["gates"].values())
    assert decision["numeric_candidates"] == [hier019.CANDIDATE_MODE]
    assert decision["tail_candidate_selected_count"] == 4


def test_one_selected_local_regression_vs_h005_rejects_portfolio() -> None:
    rows: list[dict[str, object]] = []
    for index in range(4):
        image = f"coco_{index}"
        rows.extend((_control(image), _direct(image, h005_regression=index == 0)))
    decision = hier019._decision(rows, [{"status": "ok"}] * 8, _args())

    assert not decision["gates"]["all_pixel_max_noninferior_vs_h005"]
    assert decision["numeric_candidates"] == []


def test_consumed_test_replay_requires_all_sixteen_pairs() -> None:
    rows: list[dict[str, object]] = []
    for index in range(16):
        image = f"test_{index}"
        rows.extend((_control(image), _direct(image)))
    decision = hier019._decision(
        rows,
        [{"status": "ok"}] * 32,
        _args(phase="replay_tests"),
    )

    assert decision["gates"]["complete_h005_pairs"]
    assert decision["gates"]["all_ms_ssim_noninferior_vs_h005"]
    assert decision["gates"]["all_lpips_noninferior_vs_h005"]
    assert decision["bounded_bank_pass"]
