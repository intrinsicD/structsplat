from __future__ import annotations

from argparse import Namespace
import math

from scripts.experiments import hier015_geometry_escape as hier015


def _args(*, phase: str = "development", disposition: str | None = None) -> Namespace:
    return Namespace(
        phase=phase,
        disposition=disposition,
        target_gaussians=7000,
        max_side=512,
        seed=0,
        projection_max_iterations=96,
        coefficient_limit=16.0,
        geometry_steps=400,
        direct_fit_steps=750,
        device="cuda",
        additive_renderer="cuda_additive",
        direct_renderer="cuda",
        render_chunk=256,
        lpips=True,
    )


def _synthetic_rows(
    *,
    hierarchy_ratio: float = 0.70,
    direct_psnr_delta: float = 2.5,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for image_index in range(4):
        image = f"coco_{image_index}"
        for arm in hier015.DEVELOPMENT_ARMS:
            if arm == "h005_control":
                ratio = 1.0
                psnr_delta = 0.0
                pixel_max = 0.20
                patch_max = 0.10
                overhead = 0.0
                geometry_changed = False
            elif arm in hier015.HIERARCHY_ARMS:
                ratio = hierarchy_ratio
                psnr_delta = -10.0 * math.log10(ratio)
                pixel_max = 0.15
                patch_max = 0.08
                overhead = 0.25
                geometry_changed = True
            elif arm == hier015.DIRECT_ARM:
                ratio = 10.0 ** (-direct_psnr_delta / 10.0)
                psnr_delta = direct_psnr_delta
                pixel_max = 0.12
                patch_max = 0.07
                overhead = None
                geometry_changed = True
            else:
                ratio = 0.95
                psnr_delta = -10.0 * math.log10(ratio)
                pixel_max = 0.18
                patch_max = 0.09
                overhead = 0.1
                geometry_changed = False
            rows.append(
                {
                    "image": image,
                    "arm": arm,
                    "masked_mse": ratio,
                    "psnr_db": 20.0 + psnr_delta,
                    "ms_ssim": 0.80 + 0.01 * (1.0 - ratio),
                    "lpips": 0.20 - 0.01 * (1.0 - ratio),
                    "artifact_pixel_rmse_max": pixel_max,
                    "artifact_patch_rmse_max_7": patch_max,
                    "n_gaussians": 7000,
                    "coefficient_abs_max": 8.0,
                    "geometry_changed": geometry_changed,
                    "selected_transaction_safe": True,
                    "maintained_render_parity_max_abs": 1e-6,
                    "repeated_render_parity_max_abs": 1e-6,
                    "method_overhead_ratio": overhead,
                }
            )
    return rows


def test_development_bank_has_four_frozen_hash_bindings() -> None:
    assert tuple(hier015.DEVELOPMENT_BINDINGS) == (
        "COCO_train2014_000000371955.jpg",
        "COCO_train2014_000000012379.jpg",
        "COCO_train2014_000000090218.jpg",
        "COCO_train2014_000000237851.jpg",
    )
    assert all(len(digest) == 64 for digest in hier015.DEVELOPMENT_BINDINGS.values())


def test_frozen_configs_isolate_geometry_alternation_and_direct_control() -> None:
    args = _args()
    one = hier015._alternating_config(args, "relax_1x400")
    two = hier015._alternating_config(args, "relax_2x200")
    init, fit = hier015._direct_configs(args)

    assert one.rounds == 1 and one.geometry.steps == 400
    assert two.rounds == 2 and two.geometry.steps == 200
    assert one.rounds * one.geometry.steps == two.rounds * two.geometry.steps == 400
    assert one.projection.selection_mode == two.projection.selection_mode
    assert one.projection.selection_mode == "bounded_intermediate"
    assert one.projection.regularization_center == one.projection.solver_start == "zero"
    assert one.projection.frozen_base_mode == "explicit"
    assert init.strategy == "aniso_onedge"
    assert init.num_gaussians == 7000
    assert init.sampling_mode == "wse"
    assert init.scale_cap_mode == "feature" and init.scale_cap_max == 38.4
    assert fit.iters == 750 and fit.renderer == "cuda"
    assert fit.pixel_loss == "l1" and fit.ssim_weight == 0.3
    assert fit.checkpoint_policy == "best_psnr_final_count"
    assert fit.max_gaussians == 7000
    assert fit.prune_every is None and fit.split_every is None


def test_decision_prefers_a_passing_hierarchy_arm_before_direct() -> None:
    decision = hier015._aggregate(_synthetic_rows(), _args())

    assert all(decision["hierarchy_gates"]["relax_1x400"].values())
    assert all(decision["direct_gate"].values())
    assert decision["numeric_candidates"] == [
        "relax_1x400",
        hier015.DIRECT_ARM,
    ]
    assert decision["numeric_disposition"] == "relax_1x400"


def test_decision_falls_back_to_direct_only_after_hierarchy_fails() -> None:
    decision = hier015._aggregate(
        _synthetic_rows(hierarchy_ratio=0.95),
        _args(),
    )

    assert not decision["hierarchy_gates"]["relax_1x400"][
        "geometric_mean_mse_ratio_le_0_80"
    ]
    assert all(decision["direct_gate"].values())
    assert decision["numeric_candidates"] == [hier015.DIRECT_ARM]
    assert decision["numeric_disposition"] == hier015.DIRECT_ARM


def test_replay_completeness_accounts_for_hierarchy_control_rows() -> None:
    direct_rows = [
        {
            "n_gaussians": 7000,
            "masked_mse": 0.1,
            "psnr_db": 10.0,
            "ms_ssim": 0.8,
            "lpips": 0.2,
        }
        for _ in range(16)
    ]
    hierarchy_rows = direct_rows * 2

    direct = hier015._aggregate(
        direct_rows,
        _args(phase="replay", disposition=hier015.DIRECT_ARM),
    )
    hierarchy = hier015._aggregate(
        hierarchy_rows,
        _args(phase="replay", disposition="relax_1x400"),
    )

    assert direct["complete"]
    assert hierarchy["complete"]
