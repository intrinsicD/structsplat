from __future__ import annotations

import argparse

import numpy as np
import pytest

from scripts.experiments import hier029_janelle_mask_diagnostic as h29


def _args(tmp_path, **updates):
    image = tmp_path / "image.jpg"
    mask = tmp_path / "mask.png"
    image.write_bytes(b"image")
    mask.write_bytes(b"mask")
    values = {
        "image": image,
        "mask": mask,
        "out": tmp_path / "out",
        "max_side": 1200,
        "seed": 0,
        "device": "cuda",
        "lpips": True,
        "render_chunk": 256,
        "error_scale": 4.0,
        "resume": False,
    }
    values.update(updates)
    return argparse.Namespace(**values)


def test_frozen_protocol_validation_and_fit_controls(tmp_path):
    args = _args(tmp_path)
    h29._validate_args(args)

    assert args.iters == 500
    assert args.budgets == [640]
    full = h29._fit_config(args, "cuda_additive", 960, masked=False)
    masked = h29._fit_config(args, "cuda_additive", 960, masked=True)
    assert full.loss_weighting == "none"
    assert masked.loss_weighting == "mask"
    assert full.checkpoint_policy == masked.checkpoint_policy == "best_psnr_final_count"
    assert full.max_gaussians == masked.max_gaussians == 960
    assert not full.support_fade and not masked.support_fade


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"max_side": 1199}, "max_side"),
        ({"seed": 1}, "seed"),
        ({"device": "cpu"}, "device"),
        ({"lpips": False}, "lpips"),
        ({"render_chunk": 128}, "render_chunk"),
    ],
)
def test_frozen_protocol_rejects_drift(tmp_path, updates, message):
    with pytest.raises(SystemExit, match=message):
        h29._validate_args(_args(tmp_path, **updates))


def test_mask_objective_and_foreground_bounds_are_deterministic():
    source = np.arange(8 * 10 * 3, dtype=np.float32).reshape(8, 10, 3) / 240.0
    mask = np.zeros((8, 10), dtype=bool)
    mask[2:6, 3:8] = True

    assert np.array_equal(h29._objective(source, mask, "full_frame"), source)
    masked = h29._objective(source, mask, "masked_foreground")
    assert np.array_equal(masked[mask], source[mask])
    assert np.count_nonzero(masked[~mask]) == 0
    assert h29._foreground_bounds(mask, padding=1) == (2, 1, 9, 7)


def test_decision_remains_diagnostic_even_for_complete_integrity_matrix():
    rows = []
    for mode in h29.MODES:
        for arm in h29.ARMS:
            rows.append(
                {
                    "mode": mode,
                    "arm": arm,
                    "completed": True,
                    "n_gaussians": h29.COUNT_BY_ARM[arm],
                    "maintained_render_parity_max_abs": 0.0,
                    "repeated_render_parity_max_abs": 0.0,
                    "endpoint_internal_parity_max_abs": 0.0,
                    "psnr_db": 30.0,
                    "foreground_psnr_db": 31.0,
                    "full_psnr_db": 29.0,
                }
            )

    decision = h29._decision(rows)

    assert decision["all_cells_present"]
    assert decision["integrity_pass"]
    assert decision["overall_pass"] is False
    assert decision["formal_claim_ready"] is False
