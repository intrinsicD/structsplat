from __future__ import annotations

from argparse import Namespace
from dataclasses import asdict
import hashlib
import json

import numpy as np
import pytest

from scripts.experiments import hier018_counted_background as hier018
from structsplat.config import FitConfig
from structsplat.gaussians import GaussianField
from structsplat.init import background_count


torch = pytest.importorskip("torch")


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
        error_scale=4.0,
        development_decision=None,
        control_metrics=None,
        recover_from=None,
    )


def _row(
    image: str,
    arm: str,
    *,
    mse: float,
    psnr: float,
    ms_ssim: float,
    lpips: float,
    pixel_max: float,
    patch_max: float,
    seconds: float,
) -> dict[str, object]:
    row: dict[str, object] = {
        "image": image,
        "arm": arm,
        "masked_mse": mse,
        "psnr_db": psnr,
        "ms_ssim": ms_ssim,
        "lpips": lpips,
        "artifact_pixel_rmse_max": pixel_max,
        "artifact_patch_rmse_max_7": patch_max,
        "pipeline_algorithm_seconds": seconds,
        "n_gaussians": 7000,
        "maintained_render_parity_max_abs": 1e-6,
        "repeated_render_parity_max_abs": 1e-6,
    }
    if arm == hier018.BACKGROUND_ARM:
        row.update(
            {
                "background_count": 64,
                "detail_count": 6936,
                "background_geometry_bit_exact": True,
                "background_geometry_persistence_bit_exact": True,
                "denominator_min": 1e-5,
                "baseline_low_coverage_pixel_count": 1,
                "low_coverage_display_error_improvement_max": 0.01,
            }
        )
    return row


def _development_rows(*, regress: bool = False) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index in range(4):
        image = f"coco_{index}"
        rows.append(
            _row(
                image,
                hier018.CONTROL_ARM,
                mse=1.0,
                psnr=20.0,
                ms_ssim=0.70,
                lpips=0.60,
                pixel_max=0.80,
                patch_max=0.40,
                seconds=100.0,
            )
        )
        rows.append(
            _row(
                image,
                hier018.BASELINE_ARM,
                mse=0.10,
                psnr=30.0,
                ms_ssim=0.950,
                lpips=0.100,
                pixel_max=0.50,
                patch_max=0.20,
                seconds=10.0,
            )
        )
        rows.append(
            _row(
                image,
                hier018.BACKGROUND_ARM,
                mse=0.11 if regress and index == 0 else 0.09,
                psnr=30.45,
                ms_ssim=0.951,
                lpips=0.090,
                pixel_max=0.45,
                patch_max=0.18,
                seconds=10.5,
            )
        )
    return rows


def test_prospective_bindings_match_frozen_salted_filename_hashes() -> None:
    assert len(hier018.DEVELOPMENT_BINDINGS) == 4
    assert tuple(hier018.DEVELOPMENT_BINDINGS) == tuple(hier018.SELECTION_DIGESTS)
    for name, digest in hier018.SELECTION_DIGESTS.items():
        assert hashlib.sha256(("HIER-018-v1:" + name).encode()).hexdigest() == digest


def test_configs_reserve_exactly_64_rows_and_change_only_background_init() -> None:
    baseline, candidate, fit = hier018._configs(_args())
    baseline_record = asdict(baseline)
    candidate_record = asdict(candidate)

    assert baseline_record.pop("background_fraction") == 0.0
    assert baseline_record.pop("background_grid") == 0
    assert candidate_record.pop("background_fraction") == 0.05
    assert candidate_record.pop("background_grid") == 8
    assert candidate_record == baseline_record
    assert background_count(candidate) == 64
    assert candidate.num_gaussians - background_count(candidate) == 6936
    assert fit.normalization_eps == 1e-8


def _toy_field(*, background_mean_x: float = 0.0) -> GaussianField:
    return GaussianField.from_numpy(
        means=np.asarray([[background_mean_x, 0.0], [2.0, 2.0]], dtype=np.float32),
        scales=np.asarray([[2.0, 2.0], [1.0, 1.0]], dtype=np.float32),
        angles=np.zeros(2, dtype=np.float32),
        colors=np.asarray([[0.2, 0.3, 0.4], [0.7, 0.6, 0.5]], dtype=np.float32),
        background_mask=np.asarray([True, False]),
    )


def test_background_geometry_certificate_detects_any_background_shift(tmp_path) -> None:
    initial = _toy_field()
    color_only = initial.detached()
    color_only.colors[0, 0] += 0.1
    exact = hier018._background_geometry_delta(initial, color_only)
    assert exact["background_geometry_bit_exact"]
    assert exact["background_mean_shift_max"] == 0.0

    shifted = _toy_field(background_mean_x=0.25)
    changed = hier018._background_geometry_delta(initial, shifted)
    assert not changed["background_geometry_bit_exact"]
    assert changed["background_mean_shift_max"] == pytest.approx(0.25)

    path = tmp_path / "field.npz"
    color_only.save(str(path))
    persisted = GaussianField.load(str(path))
    assert hier018._background_geometry_delta(initial, persisted)[
        "background_geometry_bit_exact"
    ]
    assert hier018._background_geometry_hash(initial) == hier018._background_geometry_hash(
        persisted
    )


def test_background_coverage_records_denominator_stratified_error(tmp_path) -> None:
    field = GaussianField.from_numpy(
        means=np.asarray([[1.0, 1.0]], dtype=np.float32),
        scales=np.asarray([[2.0, 2.0]], dtype=np.float32),
        angles=np.zeros(1, dtype=np.float32),
        colors=np.asarray([[0.2, 0.3, 0.4]], dtype=np.float32),
        background_mask=np.asarray([True]),
    )
    image = np.zeros((3, 3, 3), dtype=np.float32)
    reconstruction = np.full_like(image, 0.1)
    record = hier018._background_coverage(
        field,
        FitConfig(renderer="normalized", render_chunk=1),
        image,
        reconstruction,
        tmp_path,
    )

    assert record["background_denominator_min"] > 0.0
    bands = record["error_by_background_denominator_band"]
    assert sum(int(value["count"]) for value in bands.values()) == 9
    assert (tmp_path / "background_denominator.npz").is_file()
    persisted = json.loads((tmp_path / "background_denominator.json").read_text())
    assert persisted == record


def test_development_gate_accepts_only_a_complete_noninferior_certificate() -> None:
    attempts = [{"status": "ok"} for _ in range(12)]
    decision = hier018._development_decision(_development_rows(), attempts)

    assert all(decision["gates"].values())
    assert decision["numeric_candidates"] == [hier018.BACKGROUND_ARM]


def test_one_raw_mse_regression_rejects_background_candidate() -> None:
    decision = hier018._development_decision(_development_rows(regress=True), [])

    assert not decision["gates"]["all_mse_noninferior_vs_baseline"]
    assert decision["numeric_candidates"] == []


def test_consumed_tests_replay_requires_every_pair_noninferior() -> None:
    rows: list[dict[str, object]] = []
    for bank_index in range(16):
        image_rows = _development_rows()[0:3]
        for row in image_rows[1:]:
            rows.append({**row, "image": f"test_{bank_index}"})
    decision = hier018._replay_decision(rows, [], _args(phase="replay_tests"))

    assert decision["clauses"]["complete_pairs"]
    assert decision["bounded_bank_pass"]
