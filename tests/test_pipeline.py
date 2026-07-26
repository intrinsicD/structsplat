"""CORE-012: the maintained best-pipeline entrypoint (ADR-0025).

The fast tests cover the recipe surface and the config -> schedule translation, which is where a
future recipe change lands. The two end-to-end arm tests are ``slow``: the schedule is a real fit,
not a unit of work, so the portable gate checks its wiring and ``-m slow`` checks its behavior.
"""
from __future__ import annotations

import numpy as np
import pytest

from structsplat.pipeline import (
    RECIPE,
    PipelineConfig,
    build_fit_config,
    build_init_config,
    build_schedule,
    run_pipeline,
)


def _image(h: int = 64, w: int = 64) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w]
    img = np.stack([
        0.5 + 0.4 * np.sin(xx / 7.0),
        0.5 + 0.4 * np.cos(yy / 9.0),
        ((xx + yy) % 32) / 32.0,
    ], axis=-1)
    return np.clip(img, 0.0, 1.0).astype(np.float32)


def _mask(h: int = 64, w: int = 64) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w]
    return ((xx - w / 2.0) ** 2 + (yy - h / 2.0) ** 2) < (0.34 * w) ** 2


def _tiny_config(**overrides) -> PipelineConfig:
    base = dict(capacity=220, seed=0, device="cpu", step_scale=0.004, block_steps=2)
    base.update(overrides)
    return PipelineConfig(**base)


class TestRecipe:
    def test_recipe_declares_its_evidence_and_scope(self):
        """A recipe entry without a claim behind it is the failure mode this module exists to stop."""
        assert RECIPE["name"] and RECIPE["version"]
        assert RECIPE["evidence"], "recipe must cite the claims that authorize it"
        for claim in RECIPE["evidence"]:
            assert claim.startswith("C") and claim[1:].isdigit()
        assert RECIPE["evidence_scope"], "recipe must state how far its evidence reaches"
        assert RECIPE["choices"], "recipe must name the choices it pins"

    def test_defaults_match_the_measured_recipe(self):
        """The shipped defaults are the FIT-023/025 development winner, not the library defaults."""
        cfg = PipelineConfig()
        assert cfg.pareto_safe_checkpoints is True      # C50
        assert cfg.pareto_checkpoint_every == 50        # C50
        assert cfg.event_color_solve is False           # C50 refuted it
        assert cfg.storage_policy == "dynamic"          # C51 kept the default dynamic
        assert build_init_config(cfg, 100).strategy == "quadtree_wse"      # ADR-0013
        assert build_init_config(cfg, 100).wse_progressive_order is True   # C25

    def test_recipe_does_not_leak_into_library_defaults(self):
        """FitConfig/InitConfig stay conservative; only this module ships the measured recipe."""
        from structsplat.config import FitConfig, InitConfig

        assert InitConfig().wse_progressive_order is False
        assert FitConfig().checkpoint_policy == "terminal"


class TestConfigDerivation:
    def test_capacity_alone_rescales_the_schedule(self):
        cfg = PipelineConfig(capacity=11_000)
        assert cfg.resolved_initial() == 5_000
        assert cfg.resolved_boundary() == 500
        assert cfg.resolved_coverage_target() == 8_000
        assert cfg.resolved_detail_target() == 10_000

    def test_halved_capacity_halves_every_phase_target(self):
        cfg = PipelineConfig(capacity=5_500)
        assert cfg.resolved_initial() == 2_500
        assert cfg.resolved_coverage_target() == 4_000
        assert cfg.resolved_detail_target() == 5_000

    @pytest.mark.parametrize("kwargs", [
        dict(capacity=0),
        dict(capacity=100, initial_gaussians=200),
        dict(capacity=100, initial_gaussians=50, boundary_gaussians=50),
        dict(step_scale=0.0),
        dict(block_steps=0),
        dict(capacity=100, initial_gaussians=90, coverage_target=80),
    ])
    def test_invalid_configs_are_rejected(self, kwargs):
        with pytest.raises(ValueError):
            PipelineConfig(**kwargs).validate()

    def test_schedule_translation_preserves_ordering_and_pins(self):
        cfg = PipelineConfig(capacity=11_000, step_scale=0.5, block_steps=25)
        schedule = build_schedule(cfg)
        assert schedule.capacity == 11_000
        assert schedule.coverage_target_gaussians == 8_000
        assert schedule.detail_target_gaussians == 10_000
        assert schedule.pareto_safe_checkpoints is True
        assert schedule.pareto_checkpoint_every == 50
        assert schedule.event_color_solve is False
        assert schedule.detail_tail_max_rows == 0  # C52 rejected the specialized tail
        assert schedule.refinement_policy == "global"
        for phase in schedule.phases:
            assert phase.max_steps >= 1
            assert 1 <= phase.block_steps <= phase.max_steps
        schedule.validate(cfg.resolved_initial())

    def test_step_scale_never_produces_an_empty_phase(self):
        schedule = build_schedule(PipelineConfig(step_scale=1e-6))
        for phase in schedule.phases:
            assert phase.max_steps >= 1
            assert phase.block_steps >= 1

    def test_schedule_overrides_reject_unknown_fields(self):
        cfg = PipelineConfig(schedule_overrides={"not_a_field": 1})
        with pytest.raises(ValueError, match="unknown schedule override"):
            build_schedule(cfg)

    def test_schedule_overrides_apply(self):
        schedule = build_schedule(
            PipelineConfig(schedule_overrides={"refinement_policy": "local_neighborhood"})
        )
        assert schedule.refinement_policy == "local_neighborhood"

    def test_renderer_default_follows_the_device(self):
        """ADR-0011 ships `cuda`; C53 keeps the tiled path an explicit opt-in."""
        assert build_fit_config(PipelineConfig(), "cpu").renderer == "normalized"
        assert build_fit_config(PipelineConfig(), "cuda:0").renderer == "cuda"
        assert build_fit_config(PipelineConfig(renderer="cuda_tiled"), "cpu").renderer \
            == "cuda_tiled"

    def test_fit_config_keeps_the_compact_support_the_schedule_requires(self):
        cfg = build_fit_config(PipelineConfig(), "cpu")
        assert cfg.support_fade is True   # exact zero outside the mask needs the C0 cutoff
        assert cfg.ssim_weight == 0.0     # the schedule gates on an unweighted metric vector


class TestArmSelection:
    def test_mask_shape_is_validated(self):
        with pytest.raises(ValueError, match="does not match"):
            run_pipeline(_image(), np.ones((32, 32), dtype=bool), _tiny_config())

    def test_empty_mask_is_rejected_with_the_full_frame_hint(self):
        with pytest.raises(ValueError, match="full-frame"):
            run_pipeline(_image(), np.zeros((64, 64), dtype=bool), _tiny_config())

    def test_image_shape_is_validated(self):
        with pytest.raises(ValueError, match=r"\(H, W, 3\)"):
            run_pipeline(np.zeros((64, 64), dtype=np.float32), None, _tiny_config())


@pytest.mark.slow
class TestEndToEnd:
    def test_full_frame_arm_skips_boundary_closure(self):
        result = run_pipeline(_image(), None, _tiny_config(), verbose=False)
        assert result["arm"] == "full_frame"
        assert result["init"]["boundary_rows"] == 0
        skipped = [
            record for record in result["history"]
            if record["event"] == "phase_skipped"
        ]
        assert len(skipped) == 1
        assert skipped[0]["metadata"]["reason"] == "full_frame_arm"
        # Nothing is outside a full frame, so the containment metrics are trivially satisfied.
        assert result["metrics"]["outside_max_abs"] == 0.0
        assert result["field"].n >= result["init"]["n"]
        assert result["recipe"]["version"] == RECIPE["version"]

    def test_masked_arm_runs_boundary_closure_and_renders_zero_outside(self):
        result = run_pipeline(_image(), _mask(), _tiny_config(), verbose=False)
        assert result["arm"] == "masked"
        assert result["init"]["boundary_rows"] > 0
        phases = {record["phase"] for record in result["history"]}
        assert "boundary_closure" in phases
        assert not [r for r in result["history"] if r["event"] == "phase_skipped"]
        # The containment guarantee (ADR-0017/0019 + support_fade) must survive the schedule.
        assert result["metrics"]["outside_max_abs"] == 0.0
        assert result["metrics"]["outside_coverage_max"] == 0.0

    def test_both_arms_share_the_schedule_and_the_commit_gate(self):
        """The arms must differ in the mask/boundary stages only."""
        masked = run_pipeline(_image(), _mask(), _tiny_config(), verbose=False)
        full = run_pipeline(_image(), None, _tiny_config(), verbose=False)
        assert masked["schedule"] == full["schedule"]
        shared_phases = {"bootstrap", "coverage_growth", "detail_growth",
                         "redistribution", "safe_polish"}
        assert shared_phases <= {r["phase"] for r in masked["history"]}
        assert shared_phases <= {r["phase"] for r in full["history"]}
        # Same gate: no accepted record may regress the protected foreground metric.
        for result in (masked, full):
            for record in result["history"]:
                if record["event"].startswith("phase") or not record["accepted"]:
                    continue
                before = record["before"]["foreground_mse"]
                selected = record["selected"]["foreground_mse"]
                assert selected <= before * (1.0 + 1e-5) + 1e-10

    def test_run_is_reproducible_from_its_own_provenance(self):
        result = run_pipeline(_image(), None, _tiny_config(), verbose=False)
        assert result["pipeline_config"]["seed"] == 0
        assert result["device"] == "cpu"
        assert result["recipe"]["name"] == RECIPE["name"]
        assert result["fit_config"]["renderer"] == "normalized"
