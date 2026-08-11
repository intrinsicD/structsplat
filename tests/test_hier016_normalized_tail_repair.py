from __future__ import annotations

from argparse import Namespace
import hashlib

from scripts.experiments import hier016_normalized_tail_repair as hier016


def _args() -> Namespace:
    return Namespace(tail_steps=100)


def _rows(*, local_regression: bool = False) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index in range(4):
        image = f"coco_{index}"
        rows.append(
            {
                "image": image,
                "arm": "h005_control",
                "masked_mse": 1.0,
                "psnr_db": 20.0,
                "ms_ssim": 0.70,
                "lpips": 0.60,
                "artifact_pixel_rmse_max": 0.80,
                "artifact_patch_rmse_max_7": 0.40,
                "n_gaussians": 7000,
                "coefficient_abs_max": 4.0,
                "maintained_render_parity_max_abs": 1e-6,
                "repeated_render_parity_max_abs": 1e-6,
            }
        )
        rows.append(
            {
                "image": image,
                "arm": hier016.DIRECT_ARM,
                "masked_mse": 0.10,
                "psnr_db": 30.0,
                "ms_ssim": 0.95,
                "lpips": 0.10,
                "artifact_pixel_rmse_max": 0.50,
                "artifact_patch_rmse_max_7": 0.20,
                "n_gaussians": 7000,
                "coefficient_abs_max": 4.0,
                "maintained_render_parity_max_abs": 1e-6,
                "repeated_render_parity_max_abs": 1e-6,
            }
        )
        for arm in hier016.TAIL_ARMS:
            pixel = 0.45 if arm == "tail_top0_1pct" else 0.46
            if local_regression and index == 0 and arm == "tail_top0_1pct":
                pixel = 0.81
            rows.append(
                {
                    "image": image,
                    "arm": arm,
                    "masked_mse": 0.09,
                    "psnr_db": 30.45,
                    "ms_ssim": 0.951,
                    "lpips": 0.09,
                    "artifact_pixel_rmse_max": pixel,
                    "artifact_patch_rmse_max_7": 0.18,
                    "n_gaussians": 7000,
                    "coefficient_abs_max": 4.2,
                    "color_shift_max": 0.2,
                    "selected_tail_step": 10,
                    "tail_transaction_safe": True,
                    "non_color_arrays_bit_exact": True,
                    "tail_internal_render_parity_max_abs": 1e-6,
                    "maintained_render_parity_max_abs": 1e-6,
                    "repeated_render_parity_max_abs": 1e-6,
                }
            )
    return rows


def test_prospective_bindings_match_frozen_salted_filename_hashes() -> None:
    assert len(hier016.DEVELOPMENT_BINDINGS) == 4
    assert tuple(hier016.DEVELOPMENT_BINDINGS) == tuple(hier016.SELECTION_DIGESTS)
    for name, digest in hier016.SELECTION_DIGESTS.items():
        assert hashlib.sha256(("HIER-016-v1:" + name).encode()).hexdigest() == digest


def test_tail_configs_change_only_the_preregistered_fraction() -> None:
    one = hier016._tail_config(_args(), "tail_top1pct")
    point_one = hier016._tail_config(_args(), "tail_top0_1pct")

    assert one.tail_fraction == 0.01
    assert point_one.tail_fraction == 0.001
    one_record = vars(one).copy()
    point_one_record = vars(point_one).copy()
    one_record.pop("tail_fraction")
    point_one_record.pop("tail_fraction")
    assert one_record == point_one_record
    assert one.steps == 100 and one.checkpoint_every == 5
    assert one.learning_rate == 0.01 and one.tail_weight == 4.0
    assert one.max_color_shift == 1.0 and one.color_abs_limit == 8.0


def test_development_gate_selects_smaller_worst_pixel_ratio() -> None:
    attempts = [{"status": "ok"} for _ in range(16)]
    decision = hier016._development_decision(_rows(), attempts)

    assert all(decision["arms"]["tail_top0_1pct"]["gate"].values())
    assert all(decision["arms"]["tail_top1pct"]["gate"].values())
    assert decision["numeric_candidates"] == ["tail_top0_1pct", "tail_top1pct"]
    assert decision["numeric_disposition"] == "tail_top0_1pct"


def test_one_h005_local_regression_rejects_only_that_tail_arm() -> None:
    decision = hier016._development_decision(_rows(local_regression=True), [])

    assert not decision["arms"]["tail_top0_1pct"]["gate"][
        "all_pixel_max_noninferior_vs_h005"
    ]
    assert decision["numeric_candidates"] == ["tail_top1pct"]
