from __future__ import annotations

from argparse import Namespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from scripts.experiments import hier030_janelle_7k_contained_diagnostic as h30
from structsplat.gaussians import GaussianField


def _args() -> Namespace:
    return Namespace(render_chunk=256, iters=500, budgets=[4375])


def test_scaled_ladder_is_exact_and_preserves_hier028_ratios():
    assert h30.COUNT_BY_ARM == {
        h30.NORMALIZED_ARM: 4375,
        h30.BASE_ARM: 6562,
        h30.PURSUIT_ARM: 7000,
        h30.COLD_ARM: 7000,
    }
    assert h30.FIT_COUNT_BY_ARM[h30.PURSUIT_ARM] + h30.TAIL_COUNT == 7000
    assert h30.TAIL_COUNT == 438
    assert h30.GAUSSIAN_ROW_UPDATES_BY_ARM[h30.PURSUIT_ARM] == 6562 * 500
    assert h30.TAIL_EROSION_RADIUS == pytest.approx(1.8)


def test_contained_configs_own_caps_fade_support_and_keep_mask_state_encoder_only():
    init = h30._init_config(4375, 0, contained=True)
    fit = h30._fit_config(_args(), "cuda", 4375, contained=True)
    projection = h30._projection_config(_args(), contained=True)
    pursuit = h30._tail_config(_args(), contained=True)

    assert init.scale_cap_mode == "none"
    assert init.scale_cap_max is None
    assert fit.loss_weighting == "mask"
    assert fit.mask_contain
    assert fit.mask_cap_mode == "anisotropic"
    assert fit.mask_margin == pytest.approx(0.75)
    assert fit.support_fade
    assert projection.support_fade_alpha == 1.0
    assert pursuit.support_fade
    assert pursuit.tail_gaussians == 438


def test_full_frame_configs_preserve_hier028_support_and_feature_cap():
    init = h30._init_config(4375, 0, contained=False)
    fit = h30._fit_config(_args(), "cuda", 4375, contained=False)
    projection = h30._projection_config(_args(), contained=False)
    pursuit = h30._tail_config(_args(), contained=False)

    assert init.scale_cap_mode == "feature"
    assert not fit.mask_contain
    assert fit.loss_weighting == "none"
    assert not fit.support_fade
    assert projection.support_fade_alpha == 0.0
    assert not pursuit.support_fade


def test_pure_endpoint_materializes_containment_without_side_payload():
    field = GaussianField.from_numpy(
        np.asarray([[2.0, 3.0]], dtype=np.float32),
        np.asarray([[0.5, 0.75]], dtype=np.float32),
        np.zeros(1, dtype=np.float32),
        np.ones((1, 3), dtype=np.float32),
        scale_max=np.asarray([[0.5, 0.75]], dtype=np.float32),
    )

    endpoint = h30._pure_endpoint(field)

    assert endpoint.scale_max is None
    assert endpoint.opacities is None
    assert endpoint.color_grads is None
    assert torch.equal(endpoint.means, field.means)
    assert torch.equal(endpoint.log_scales, field.log_scales)


def test_decision_requires_all_masked_containment_receipts():
    rows = []
    for mode in h30.MODES:
        for arm in h30.ARMS:
            rows.append(
                {
                    "mode": mode,
                    "arm": arm,
                    "completed": True,
                    "n_gaussians": h30.COUNT_BY_ARM[arm],
                    "four_array_endpoint_exact": True,
                    "maintained_render_parity_max_abs": 0.0,
                    "repeated_render_parity_max_abs": 0.0,
                    "endpoint_internal_parity_max_abs": 0.0,
                    "containment_pass": True,
                    "centres_outside_mask": 0,
                    "unit_coverage_outside_abs_max": 0.0,
                    "reconstruction_outside_abs_max": 0.0,
                    "psnr_db": 30.0,
                }
            )

    decision = h30._decision(rows, [])
    assert decision["integrity_pass"]
    assert decision["containment_pass"]
    assert not decision["formal_claim_ready"]

    rows[-1]["centres_outside_mask"] = 1
    failed = h30._decision(rows, [])
    assert not failed["integrity_pass"]
    assert not failed["containment_pass"]
