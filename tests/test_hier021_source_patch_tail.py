from __future__ import annotations

from argparse import Namespace
from dataclasses import asdict
import hashlib

from scripts.experiments import hier021_source_patch_tail as hier021


def _args() -> Namespace:
    return Namespace(
        target_gaussians=7000,
        max_side=512,
        seed=0,
        direct_fit_steps=750,
        device="cuda",
        additive_renderer="cuda_additive",
        direct_renderer="cuda",
        render_chunk=256,
        lpips=True,
        patch_radius=3,
        coverage_threshold=1e-8,
        error_scale=4.0,
        images=None,
        review_from=None,
        visual_disposition="pending",
        phase="development",
    )


def _metrics(
    mse: float,
    ms: float,
    lpips: float,
    pixel: float,
    patch: float,
    *,
    psnr: float = 30.0,
):
    return {
        "masked_mse": mse,
        "psnr_db": psnr,
        "ssim": ms - 0.01,
        "ms_ssim": ms,
        "lpips": lpips,
        "artifact_pixel_rmse_max": pixel,
        "artifact_patch_rmse_max_7": patch,
    }


def _rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index in range(4):
        image = f"fresh_{index}"
        rows.append(
            {
                "image": image,
                "arm": hier021.CONTROL_ARM,
                "n_gaussians": 7000,
                **_metrics(0.1, 0.8, 0.2, 0.8, 0.4, psnr=20.0),
            }
        )
        rows.append(
            {
                "image": image,
                "arm": hier021.DIRECT_ARM,
                "n_gaussians": 7000,
                **_metrics(0.01, 0.9, 0.1, 0.81, 0.3),
                "spt1_field_file_sha256_before": image,
                "spt1_field_file_sha256_after": image,
                "spt1_payload_roundtrip_exact": True,
                "spt1_candidate_finite": True,
                "spt1_outside_identity_max_abs": 0.0,
                "spt1_baseline_cold_parity_max_abs": 1e-6,
                "spt1_candidate_repeated_parity_max_abs": 1e-6,
                "spt1_selected_mode": hier021.CANDIDATE_MODE,
                "spt1_selected_count": 49,
                "spt1_selected_payload_bytes": 359,
                "spt1_candidate_payload_bytes": 359,
                "spt1_pipeline_time_ratio": 1.01,
                "spt1_decode_time_ratio": 1.1,
                "spt1_selection_clauses": {"safe": True},
                "spt1_selected_metrics": _metrics(0.009, 0.91, 0.09, 0.7, 0.3),
            }
        )
    return rows


def test_bindings_match_frozen_salted_filename_hashes() -> None:
    assert len(hier021.DEVELOPMENT_BINDINGS) == 4
    assert tuple(hier021.DEVELOPMENT_BINDINGS) == tuple(hier021.SELECTION_DIGESTS)
    for name, digest in hier021.SELECTION_DIGESTS.items():
        assert hashlib.sha256(("HIER-021-v1:" + name).encode()).hexdigest() == digest


def test_configs_freeze_direct_fit_and_patch_geometry() -> None:
    init, fit, patch = hier021._configs(_args())

    assert init.num_gaussians == 7000
    assert fit.normalization_eps == 1e-8
    assert asdict(patch) == {"radius": 3, "coverage_threshold": 1e-8}


def test_selection_requires_pointwise_and_perceptual_safety() -> None:
    baseline = _metrics(0.1, 0.95, 0.1, 0.5, 0.2)
    candidate = _metrics(0.09, 0.951, 0.09, 0.45, 0.18)
    kwargs = {
        "payload_count": 4,
        "finite": True,
        "outside_identity_max_abs": 0.0,
        "baseline_parity_max_abs": 1e-6,
        "repeated_parity_max_abs": 1e-6,
        "payload_roundtrip_exact": True,
        "pointwise_raw_delta_max": -1e-6,
        "pointwise_display_delta_max": -1.0,
    }
    selected, clauses, _ = hier021._selection(baseline, candidate, **kwargs)
    assert selected == hier021.CANDIDATE_MODE
    assert all(clauses.values())

    candidate["lpips"] = 0.101
    rejected, clauses, _ = hier021._selection(baseline, candidate, **kwargs)
    assert rejected == hier021.BASELINE_MODE
    assert not clauses["lpips_noninferior"]


def test_decision_separates_numeric_and_visual_pass() -> None:
    pending = hier021._decision(_rows(), [{"status": "ok"}] * 8, visual_disposition="pending")
    reviewed = hier021._decision(_rows(), [{"status": "ok"}] * 8, visual_disposition="pass")

    assert all(pending["gates"].values())
    assert pending["baseline_local_failure_count"] == 4
    assert pending["numeric_bank_pass"]
    assert not pending["bounded_bank_pass"]
    assert reviewed["bounded_bank_pass"]
