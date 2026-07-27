from dataclasses import asdict

import numpy as np
from PIL import Image

from structsplat.pipeline import (
    ACTIVE_LIMIT,
    CURRENT_PROFILE_NAME,
    INITIAL_GAUSSIANS,
    PHYSICAL_CAPACITY,
    PipelineConfig,
    RECIPE,
    build_current_fit_config,
    build_current_schedule,
    build_initialization_config,
    profile_manifest,
)
from structsplat.workflows import (
    ABLATION_ARMS,
    STAGE_VARIANTS,
    build_ablation_parser,
    build_benchmark_parser,
    build_convert_parser,
    build_stage_search_parser,
    _prepare_source,
)


def test_current_profile_is_the_single_measured_recipe():
    schedule = build_current_schedule(boundary_enabled=True)

    assert CURRENT_PROFILE_NAME == f"{RECIPE['name']}_{RECIPE['version']}"
    assert schedule.storage_policy == "dynamic"
    assert schedule.capacity == PHYSICAL_CAPACITY == 11_000
    assert schedule.resolved_base_active_limit() == ACTIVE_LIMIT == 11_000
    assert schedule.detail_tail_max_rows == 0
    assert schedule.pareto_safe_checkpoints is True
    assert schedule.pareto_checkpoint_every == 50
    assert schedule.event_color_solve is False
    assert schedule.refinement_policy == "global"


def test_unmasked_schedule_is_identical_except_boundary_specialization():
    masked = asdict(build_current_schedule(boundary_enabled=True))
    unmasked = asdict(build_current_schedule(boundary_enabled=False))

    assert masked.pop("boundary_enabled") is True
    assert unmasked.pop("boundary_enabled") is False
    assert masked["boundary"].pop("name") == "boundary_closure"
    assert unmasked["boundary"].pop("name") == "general_closure"
    assert masked == unmasked

    masked_fit = asdict(build_current_fit_config(masked=True))
    unmasked_fit = asdict(build_current_fit_config(masked=False))
    assert masked_fit.pop("mask_contain") is True
    assert unmasked_fit.pop("mask_contain") is False
    assert masked_fit == unmasked_fit


def test_initialization_preserves_count_and_parameters_across_paths():
    config = build_initialization_config(seed=7)
    manifest_masked = profile_manifest(masked=True)
    manifest_unmasked = profile_manifest(masked=False)

    assert config.num_gaussians == INITIAL_GAUSSIANS == 5_000
    assert config.strategy == "quadtree_wse"
    assert config.wse_progressive_order is True
    assert manifest_masked["initial_gaussians"] == INITIAL_GAUSSIANS
    assert manifest_unmasked["initial_gaussians"] == INITIAL_GAUSSIANS
    assert manifest_masked["physical_capacity"] == manifest_unmasked["physical_capacity"]
    assert manifest_masked["active_limit"] == manifest_unmasked["active_limit"]
    assert manifest_masked["mask_margin"] == manifest_unmasked["mask_margin"] == 0.75
    assert (
        manifest_masked["requested_optimizer_steps"]
        == manifest_unmasked["requested_optimizer_steps"]
    )


def test_public_parsers_expose_only_the_four_clear_workflows(tmp_path):
    source = tmp_path / "images"
    out = tmp_path / "out"

    convert = build_convert_parser().parse_args([str(source), str(out)])
    benchmark = build_benchmark_parser().parse_args([str(source), str(out)])
    ablation = build_ablation_parser().parse_args([str(source), str(out)])
    stage = build_stage_search_parser().parse_args(
        [str(source), str(out), "--stage", "coverage"]
    )

    assert convert.seed == 0
    assert convert.mask_margin == PipelineConfig.mask_margin == 0.75
    direct_mask = build_convert_parser().parse_args(
        [str(source / "one.png"), str(out), "--mask", str(source / "one_mask.png")]
    )
    assert direct_mask.mask == source / "one_mask.png"
    assert direct_mask.mask_dir is None
    assert benchmark.seeds == [0]
    assert ablation.arms == list(ABLATION_ARMS)
    assert stage.stage == "coverage"
    assert STAGE_VARIANTS["coverage"][0] == "current"
    assert all(
        variants[0] in {
            "current",
            "quadtree_wse",
            "dynamic",
            "pareto50",
        }
        for variants in STAGE_VARIANTS.values()
    )


def test_convert_direct_mask_is_loaded_and_can_be_inverted(tmp_path):
    image_path = tmp_path / "one.png"
    mask_path = tmp_path / "one_mask.png"
    Image.fromarray(np.full((4, 4, 3), 255, dtype=np.uint8)).save(image_path)
    alpha = np.zeros((4, 4), dtype=np.uint8)
    alpha[:, 2:] = 255
    rgba = np.full((4, 4, 4), 255, dtype=np.uint8)
    rgba[..., 3] = alpha
    Image.fromarray(rgba, mode="RGBA").save(mask_path)

    prepared = _prepare_source(
        image_path,
        image_path.relative_to(tmp_path),
        mask_root=None,
        direct_mask=mask_path,
        mask_invert=False,
        max_side=None,
    )
    inverted = _prepare_source(
        image_path,
        image_path.relative_to(tmp_path),
        mask_root=None,
        direct_mask=mask_path,
        mask_invert=True,
        max_side=None,
    )

    assert prepared["mask_path"] == mask_path.resolve()
    assert prepared["mask"][:, 2:].all()
    assert not prepared["mask"][:, :2].any()
    assert np.array_equal(inverted["mask"], ~prepared["mask"])
