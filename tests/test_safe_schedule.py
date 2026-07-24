import math
from dataclasses import replace

import pytest
import torch

import structsplat.fit as fit_module
import structsplat.safe_schedule as safe_schedule_module
from structsplat.config import FitConfig
from structsplat.fit import _MaskConstraint, _render, fit
from structsplat.gaussians import GaussianField
from structsplat.pool import POOL_PARK_COORD, prepare_transactional_pool
from structsplat.safe_schedule import (
    CommitTolerances,
    PhaseBudget,
    QualityMetrics,
    SafeScheduleConfig,
    _detail_tail_score,
    _expand_spatial_neighborhood,
    _safe_fit_block,
    adapt_optimizer_state,
    evaluate_quality,
    propose_birth,
    propose_merge_rebirth,
    propose_split,
    run_safe_schedule,
    safe_commit_decision,
)


def _metrics(**updates):
    values = {
        "n_gaussians": 10,
        "foreground_mse": 0.10,
        "boundary_mse": 0.20,
        "cvar99_mse": 0.30,
        "p99_mse": 0.25,
        "interior_hole_fraction": 0.10,
        "boundary_hole_fraction": 0.20,
        "outside_max_abs": 0.0,
        "outside_coverage_max": 0.0,
        "finite": True,
    }
    values.update(updates)
    return QualityMetrics(**values)


def _field(n: int, height: int, width: int) -> GaussianField:
    generator = torch.Generator().manual_seed(4)
    means = torch.stack(
        [
            torch.randint(5, width - 5, (n,), generator=generator),
            torch.randint(5, height - 5, (n,), generator=generator),
        ],
        dim=1,
    ).float()
    scales = torch.full((n, 2), 1.5)
    colors = torch.rand(n, 3, generator=generator)
    return GaussianField(
        means,
        torch.log(scales),
        torch.zeros(n),
        colors,
        torch.full((n,), torch.logit(torch.tensor(0.8))),
    )


def _mask(height: int, width: int) -> torch.Tensor:
    mask = torch.zeros(height, width, dtype=torch.bool)
    mask[3:-3, 3:-3] = True
    return mask


def _base_cfg(**updates) -> FitConfig:
    values = {
        "iters": 2,
        "renderer": "normalized",
        "render_chunk": 64,
        "pixel_loss": "l2",
        "ssim_weight": 0.0,
        "support_fade": True,
        "mask_contain": True,
        "mask_margin": 0.75,
        "mask_cap_mode": "anisotropic",
        "mask_cap_refresh_every": 1,
        "loss_weighting": "mask",
        "checkpoint_policy": "terminal",
        "log_every": 1,
    }
    values.update(updates)
    return FitConfig(**values)


def test_safe_commit_is_pareto_not_scalar_compensation():
    before = _metrics()
    # A large foreground gain is not allowed to pay for a damaged boundary.
    candidate = _metrics(foreground_mse=0.05, boundary_mse=0.21)
    accepted, reasons = safe_commit_decision(before, candidate, CommitTolerances())
    assert not accepted
    assert "boundary_mse_regressed" in reasons

    safe = _metrics(
        foreground_mse=0.09,
        boundary_mse=0.19,
        cvar99_mse=0.29,
        interior_hole_fraction=0.09,
        boundary_hole_fraction=0.19,
    )
    accepted, reasons = safe_commit_decision(before, safe, CommitTolerances())
    assert accepted
    assert reasons == []


def test_p99_guard_is_explicit_and_pareto_safe():
    before = _metrics()
    candidate = _metrics(
        foreground_mse=0.09,
        boundary_mse=0.19,
        cvar99_mse=0.29,
        p99_mse=0.26,
        interior_hole_fraction=0.09,
        boundary_hole_fraction=0.19,
    )
    accepted, reasons = safe_commit_decision(
        before, candidate, CommitTolerances()
    )
    assert accepted
    assert reasons == []

    accepted, reasons = safe_commit_decision(
        before,
        candidate,
        CommitTolerances(guard_p99=True),
    )
    assert not accepted
    assert "p99_mse_regressed" in reasons


def test_detail_tail_reserve_validation_separates_physical_and_active_limits():
    valid = SafeScheduleConfig(
        capacity=12,
        storage_policy="fixed_capacity",
        base_active_limit=10,
        detail_tail_max_rows=2,
        detail_tail_batch_rows=1,
        coverage_target_gaussians=8,
        detail_target_gaussians=10,
        event_min_count=1,
    )
    valid.validate(6)
    assert valid.resolved_base_active_limit() == 10

    with pytest.raises(ValueError, match="requires storage_policy"):
        replace(valid, storage_policy="dynamic").validate(6)
    with pytest.raises(ValueError, match="cannot exceed capacity"):
        replace(valid, detail_tail_max_rows=3).validate(6)
    with pytest.raises(ValueError, match="cannot exceed base_active_limit"):
        replace(valid, detail_target_gaussians=11).validate(6)


def test_spatial_neighborhood_expands_seed_rows_in_one_mask():
    field = GaussianField.from_numpy(
        means=[[1.0, 1.0], [2.0, 1.0], [4.0, 1.0], [20.0, 20.0]],
        scales=[[1.0, 1.0]] * 4,
        angles=[0.0] * 4,
        colors=[[0.0, 0.0, 0.0]] * 4,
    )
    trainable, metadata = _expand_spatial_neighborhood(
        field, torch.tensor([0]), neighbor_count=2
    )
    assert trainable.tolist() == [True, True, True, False]
    assert metadata["seed_rows"] == 1
    assert metadata["trainable_rows"] == 3


def test_optimizer_state_resizes_and_zeros_only_touched_rows():
    state = {
        "state": {
            0: {
                "step": torch.tensor(3.0),
                "exp_avg": torch.ones(3, 2),
                "exp_avg_sq": torch.full((3, 2), 2.0),
            }
        },
        "param_groups": [{"params": [0]}],
    }
    resized = adapt_optimizer_state(state, 3, 5, torch.tensor([1, 4]))
    assert resized["state"][0]["exp_avg"].shape == (5, 2)
    torch.testing.assert_close(resized["state"][0]["exp_avg"][0], torch.ones(2))
    torch.testing.assert_close(resized["state"][0]["exp_avg"][1], torch.zeros(2))
    torch.testing.assert_close(resized["state"][0]["exp_avg"][3], torch.zeros(2))
    torch.testing.assert_close(resized["state"][0]["exp_avg"][4], torch.zeros(2))
    # Input state is a rollback-safe independent snapshot.
    assert state["state"][0]["exp_avg"].shape == (3, 2)
    assert torch.all(state["state"][0]["exp_avg"] == 1)


def test_fit_optimizer_state_and_local_row_freeze_round_trip():
    height = width = 24
    mask = _mask(height, width)
    target = torch.zeros(height, width, 3)
    target[mask] = torch.tensor([0.2, 0.5, 0.8])
    field = _field(4, height, width)
    cfg = _base_cfg(mask_interior_undercoverage_weight=0.01)

    first = fit(
        field.detached(),
        target,
        cfg,
        mask=mask.numpy(),
        verbose=False,
        return_optimizer_state=True,
    )
    assert first["optimizer_state"] is not None

    incoming = first["field"].detached()
    frozen_before = incoming.subset(torch.tensor([0, 1, 2]))
    trainable = torch.tensor([False, False, False, True])
    second = fit(
        incoming,
        target,
        cfg,
        mask=mask.numpy(),
        verbose=False,
        optimizer_state=first["optimizer_state"],
        trainable_row_mask=trainable,
        return_optimizer_state=True,
    )
    frozen_after = second["field"].subset(torch.tensor([0, 1, 2]))
    for name in ("means", "log_scales", "rotations", "colors", "opacities"):
        torch.testing.assert_close(
            getattr(frozen_after, name),
            getattr(frozen_before, name),
            rtol=0,
            atol=0,
        )


def test_fit_active_prefix_matches_compact_fit_within_float_tolerance():
    height = width = 24
    mask = _mask(height, width)
    target = torch.zeros(height, width, 3)
    target[mask] = torch.tensor([0.2, 0.5, 0.8])
    field = _field(4, height, width)
    pooled, state = prepare_transactional_pool(field.detached(), capacity=7)
    cfg = _base_cfg(iters=3, mask_interior_undercoverage_weight=0.01)

    compact = fit(
        field.detached(),
        target,
        cfg,
        mask=mask.numpy(),
        verbose=False,
        return_optimizer_state=True,
    )
    fixed = fit(
        pooled,
        target,
        cfg,
        mask=mask.numpy(),
        verbose=False,
        return_optimizer_state=True,
        active_row_count=state.active_n,
    )

    assert fixed["field"].n == 7
    assert fixed["n_gaussians"] == compact["n_gaussians"] == 4
    assert fixed["active_row_count"] == 4
    assert torch.all(fixed["field"].means[4:] == POOL_PARK_COORD)
    fixed_active = fixed["field"].prefix_view(4)
    for name in (
        "means",
        "log_scales",
        "rotations",
        "colors",
        "opacities",
        "scale_max",
    ):
        torch.testing.assert_close(
            getattr(fixed_active, name),
            getattr(compact["field"], name),
            rtol=1e-6,
            atol=1e-8,
        )
    torch.testing.assert_close(
        fixed["render"], compact["render"], rtol=1e-6, atol=1e-8
    )
    for key, compact_param in compact["optimizer_state"]["state"].items():
        fixed_param = fixed["optimizer_state"]["state"][key]
        for name, compact_value in compact_param.items():
            fixed_value = fixed_param[name]
            if torch.is_tensor(compact_value) and compact_value.ndim > 0:
                assert fixed_value.shape[0] == pooled.n
                assert torch.count_nonzero(fixed_value[4:]) == 0
                fixed_value = fixed_value[:compact_value.shape[0]]
            torch.testing.assert_close(
                fixed_value, compact_value, rtol=1e-6, atol=1e-8
            )


def test_fit_active_prefix_rejects_optimizer_without_active_shape_updates():
    height = width = 24
    mask = _mask(height, width)
    target = torch.zeros(height, width, 3)
    field, state = prepare_transactional_pool(
        _field(4, height, width),
        capacity=7,
    )
    with pytest.raises(ValueError, match="requires optimizer='adam'"):
        fit(
            field,
            target,
            replace(_base_cfg(iters=1), optimizer="adamw"),
            mask=mask.numpy(),
            verbose=False,
            active_row_count=state.active_n,
        )


def test_fit_returns_state_matched_intermediate_checkpoints():
    height = width = 24
    mask = _mask(height, width)
    target = torch.zeros(height, width, 3)
    target[mask] = torch.tensor([0.2, 0.5, 0.8])
    cfg = _base_cfg(iters=5)
    output = fit(
        _field(4, height, width),
        target,
        cfg,
        mask=mask.numpy(),
        verbose=False,
        return_optimizer_state=True,
        state_checkpoint_every=2,
    )
    checkpoints = output["state_checkpoints"]
    assert [checkpoint["iter"] for checkpoint in checkpoints] == [2, 4, 5]
    assert [checkpoint["terminal"] for checkpoint in checkpoints] == [
        False,
        False,
        True,
    ]
    assert all(checkpoint["optimizer_state"] is not None for checkpoint in checkpoints)
    for name in ("means", "log_scales", "rotations", "colors", "opacities"):
        torch.testing.assert_close(
            getattr(checkpoints[-1]["field"], name),
            getattr(output["field"], name),
            rtol=0,
            atol=0,
        )

    resumed = fit(
        checkpoints[0]["field"],
        target,
        _base_cfg(iters=1),
        mask=mask.numpy(),
        verbose=False,
        optimizer_state=checkpoints[0]["optimizer_state"],
        return_optimizer_state=True,
    )
    assert resumed["iterations_run"] == 1
    assert resumed["state_checkpoints"] is None


def test_state_checkpoint_observation_does_not_change_fit_trajectory():
    height = width = 24
    mask = _mask(height, width)
    target = torch.zeros(height, width, 3)
    target[mask] = torch.tensor([0.2, 0.5, 0.8])
    initial = _field(4, height, width)
    cfg = _base_cfg(iters=3)
    control = fit(
        initial.detached(),
        target,
        cfg,
        mask=mask.numpy(),
        verbose=False,
        return_optimizer_state=True,
    )
    observed = fit(
        initial.detached(),
        target,
        cfg,
        mask=mask.numpy(),
        verbose=False,
        return_optimizer_state=True,
        state_checkpoint_every=1,
    )
    assert control["state_checkpoints"] is None
    assert len(observed["state_checkpoints"]) == 3
    for name in ("means", "log_scales", "rotations", "colors", "opacities"):
        torch.testing.assert_close(
            getattr(control["field"], name),
            getattr(observed["field"], name),
            rtol=0,
            atol=0,
        )
    assert control["psnr"] == observed["psnr"]


def test_safe_fit_block_competes_raw_and_color_solved_checkpoints():
    height = width = 24
    mask = _mask(height, width)
    target = torch.zeros(height, width, 3)
    target[mask] = torch.tensor([0.2, 0.5, 0.8])
    field = _field(4, height, width)
    cfg = _base_cfg(iters=2)
    constraint = _MaskConstraint.from_mask(
        mask.numpy(),
        target.device,
        target.dtype,
        cfg.sigma_cutoff,
        cfg.mask_margin,
        cap_mode=cfg.mask_cap_mode,
        undercoverage_band=cfg.mask_undercoverage_band,
    )
    constraint.apply(field, cfg, refresh=True)
    metrics, _ = evaluate_quality(
        field, target, mask, cfg, constraint, coverage_tau=0.05
    )
    schedule = SafeScheduleConfig(
        capacity=4,
        coverage_target_gaussians=4,
        detail_target_gaussians=4,
        pareto_safe_checkpoints=True,
        pareto_checkpoint_every=1,
        event_color_solve=True,
    )
    _, _, _, record, output = _safe_fit_block(
        field,
        None,
        metrics,
        target,
        mask.numpy(),
        mask,
        cfg,
        constraint,
        schedule,
        post_color_solve=True,
        verbose=False,
    )
    candidates = record["metadata"]["checkpoint_candidates"]
    assert output["iterations_run"] == 2
    assert [(item["iter"], item["color_solved"]) for item in candidates] == [
        (1, False),
        (1, True),
        (2, False),
        (2, True),
    ]
    assert record["metadata"]["pareto_safe_checkpoints"] is True
    assert record["metadata"]["event_color_solve_enabled"] is True
    assert record["accepted_steps"] <= record["attempted_steps"]


def test_safe_fit_block_can_select_an_earlier_safe_checkpoint(monkeypatch):
    height = width = 24
    mask = _mask(height, width)
    target = torch.zeros(height, width, 3)
    cfg = _base_cfg(iters=2)
    field = _field(4, height, width)
    constraint = _MaskConstraint.from_mask(
        mask.numpy(),
        target.device,
        target.dtype,
        cfg.sigma_cutoff,
        cfg.mask_margin,
        cap_mode=cfg.mask_cap_mode,
        undercoverage_band=cfg.mask_undercoverage_band,
    )
    early = field.detached()
    early.colors.fill_(1.0)
    terminal = field.detached()
    terminal.colors.fill_(2.0)

    def fake_fit(*args, **kwargs):
        return {
            "field": terminal,
            "optimizer_state": None,
            "state_checkpoints": [
                {
                    "iter": 1,
                    "field": early,
                    "optimizer_state": None,
                    "terminal": False,
                },
                {
                    "iter": 2,
                    "field": terminal,
                    "optimizer_state": None,
                    "terminal": True,
                },
            ],
            "iterations_run": 2,
            "psnr": 9.0,
        }

    def fake_evaluate(candidate_field, *args, **kwargs):
        marker = float(candidate_field.colors[0, 0])
        if marker == 1.0:
            candidate = _metrics(
                foreground_mse=0.09,
                boundary_mse=0.19,
                cvar99_mse=0.29,
                p99_mse=0.24,
                interior_hole_fraction=0.09,
                boundary_hole_fraction=0.19,
            )
        else:
            candidate = _metrics(foreground_mse=0.11)
        return candidate, target

    monkeypatch.setattr(safe_schedule_module, "fit", fake_fit)
    monkeypatch.setattr(safe_schedule_module, "evaluate_quality", fake_evaluate)
    selected, _, selected_metrics, record, _ = _safe_fit_block(
        field,
        None,
        _metrics(),
        target,
        mask.numpy(),
        mask,
        cfg,
        constraint,
        SafeScheduleConfig(
            capacity=4,
            coverage_target_gaussians=4,
            detail_target_gaussians=4,
            pareto_safe_checkpoints=True,
            pareto_checkpoint_every=1,
        ),
    )
    assert record["accepted"] is True
    assert record["accepted_steps"] == 1
    assert record["metadata"]["selected_from_pareto_checkpoint"] is True
    assert selected_metrics.foreground_mse == 0.09
    assert float(selected.colors[0, 0]) == 1.0


def test_event_color_solve_zeros_selected_color_moments(monkeypatch):
    height = width = 24
    mask = _mask(height, width)
    target = torch.zeros(height, width, 3)
    cfg = _base_cfg(iters=1)
    field = _field(4, height, width)
    constraint = _MaskConstraint.from_mask(
        mask.numpy(),
        target.device,
        target.dtype,
        cfg.sigma_cutoff,
        cfg.mask_margin,
        cap_mode=cfg.mask_cap_mode,
        undercoverage_band=cfg.mask_undercoverage_band,
    )
    terminal = field.detached()
    terminal.colors.fill_(1.0)
    state = {
        "state": {
            3: {
                "step": torch.tensor(1.0),
                "exp_avg": torch.ones_like(terminal.colors),
                "exp_avg_sq": torch.ones_like(terminal.colors),
            },
        },
        "param_groups": [
            {"params": [0]},
            {"params": [1]},
            {"params": [2]},
            {"params": [3]},
        ],
    }

    def fake_fit(*args, **kwargs):
        return {
            "field": terminal,
            "optimizer_state": state,
            "state_checkpoints": None,
            "iterations_run": 1,
            "psnr": 10.0,
        }

    def fake_solve(candidate_field, *args, **kwargs):
        candidate_field.colors.fill_(2.0)
        return {"iterations": 1, "relative_residual": 0.0}

    def fake_evaluate(candidate_field, *args, **kwargs):
        marker = float(candidate_field.colors[0, 0])
        candidate = _metrics(
            foreground_mse=0.09 if marker == 1.0 else 0.08,
            boundary_mse=0.19 if marker == 1.0 else 0.18,
            cvar99_mse=0.29 if marker == 1.0 else 0.28,
            p99_mse=0.24 if marker == 1.0 else 0.23,
            interior_hole_fraction=0.09,
            boundary_hole_fraction=0.19,
        )
        return candidate, target

    monkeypatch.setattr(safe_schedule_module, "fit", fake_fit)
    monkeypatch.setattr(
        safe_schedule_module, "_solve_colors_normalized", fake_solve
    )
    monkeypatch.setattr(safe_schedule_module, "evaluate_quality", fake_evaluate)
    selected, selected_state, _, record, _ = _safe_fit_block(
        field,
        None,
        _metrics(),
        target,
        mask.numpy(),
        mask,
        cfg,
        constraint,
        SafeScheduleConfig(
            capacity=4,
            coverage_target_gaussians=4,
            detail_target_gaussians=4,
            event_color_solve=True,
        ),
        post_color_solve=True,
    )
    assert float(selected.colors[0, 0]) == 2.0
    assert record["metadata"]["event_color_solve_applied"] is True
    assert torch.count_nonzero(selected_state["state"][3]["exp_avg"]) == 0
    assert torch.count_nonzero(selected_state["state"][3]["exp_avg_sq"]) == 0


def test_fit_reuses_precomputed_mask_constraint(monkeypatch):
    height = width = 24
    mask = _mask(height, width)
    target = torch.zeros(height, width, 3)
    field = _field(4, height, width)
    cfg = _base_cfg()
    constraint = _MaskConstraint.from_mask(
        mask.numpy(),
        target.device,
        target.dtype,
        cfg.sigma_cutoff,
        cfg.mask_margin,
        cap_mode=cfg.mask_cap_mode,
        undercoverage_band=cfg.mask_undercoverage_band,
    )

    def fail_rebuild(*args, **kwargs):
        raise AssertionError("mask EDT was rebuilt")

    monkeypatch.setattr(fit_module._mask.MaskGeometry, "build", fail_rebuild)
    output = fit(
        field,
        target,
        cfg,
        mask=mask.numpy(),
        verbose=False,
        mask_constraint_override=constraint,
    )
    assert output["field"].n == field.n


def test_undercoverage_floor_respects_intermittent_cadence(monkeypatch):
    height = width = 24
    mask = _mask(height, width)
    target = torch.zeros(height, width, 3)
    field = _field(4, height, width)
    cfg = _base_cfg(
        iters=5,
        mask_interior_undercoverage_weight=0.01,
        mask_undercoverage_every=3,
    )
    constraint = _MaskConstraint.from_mask(
        mask.numpy(),
        target.device,
        target.dtype,
        cfg.sigma_cutoff,
        cfg.mask_margin,
        cap_mode=cfg.mask_cap_mode,
        undercoverage_band=cfg.mask_undercoverage_band,
    )
    original = constraint.raw_weight_map
    calls = 0

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(constraint, "raw_weight_map", counted)
    fit(
        field,
        target,
        cfg,
        mask=mask.numpy(),
        verbose=False,
        mask_constraint_override=constraint,
    )
    assert calls == 2  # steps 0 and 3


def test_coverage_birth_is_mask_contained_and_uses_local_covariance():
    height = width = 36
    mask = _mask(height, width)
    target = torch.zeros(height, width, 3)
    target[8:14, 8:14] = 1.0
    target[22:31, 20:30, 1] = 0.8
    field = _field(6, height, width)
    cfg = _base_cfg(iters=1)
    constraint = _MaskConstraint.from_mask(
        mask.numpy(),
        target.device,
        target.dtype,
        cfg.sigma_cutoff,
        cfg.mask_margin,
        cap_mode="anisotropic",
        undercoverage_band=4.0,
    )
    constraint.apply(field, cfg, refresh=True)
    render = _render(field, cfg, height, width, support_fade_alpha=1.0)
    schedule = SafeScheduleConfig(
        capacity=12,
        coverage_target_gaussians=8,
        detail_target_gaussians=10,
        event_min_count=1,
        event_spacing_px=3.0,
    )
    proposal = propose_birth(
        field, target, render, cfg, constraint, schedule, 4, "coverage"
    )
    assert proposal is not None
    assert proposal.field.n == field.n + proposal.count
    assert proposal.metadata["covariance_rule"].startswith("local residual support")
    new_means = proposal.field.means[field.n:]
    ix = new_means[:, 0].round().long().clamp(0, width - 1)
    iy = new_means[:, 1].round().long().clamp(0, height - 1)
    assert torch.all(constraint.eroded_flat[iy * width + ix])
    assert torch.unique(proposal.field.log_scales[field.n:], dim=0).shape[0] > 1

    pooled, state = prepare_transactional_pool(field.detached(), capacity=12)
    fixed_proposal = propose_birth(
        pooled,
        target,
        render,
        cfg,
        constraint,
        replace(schedule, storage_policy="fixed_capacity"),
        4,
        "coverage",
        active_n=state.active_n,
    )
    assert fixed_proposal is not None
    assert fixed_proposal.field.n == 12
    assert fixed_proposal.active_n == proposal.field.n
    assert fixed_proposal.touched.tolist() == list(
        range(field.n, proposal.field.n)
    )
    for name in (
        "means",
        "log_scales",
        "rotations",
        "colors",
        "opacities",
        "scale_max",
    ):
        torch.testing.assert_close(
            getattr(
                fixed_proposal.field.prefix_view(fixed_proposal.active_n),
                name,
            ),
            getattr(proposal.field, name),
            rtol=0,
            atol=0,
        )


def test_detail_tail_score_excludes_holes_boundary_and_low_frequency_error():
    height = width = 40
    mask = torch.zeros(height, width, dtype=torch.bool)
    mask[2:-2, 2:-2] = True
    target = torch.zeros(height, width, 3)
    target[16:24, 16:24] = (
        torch.arange(8)[:, None] + torch.arange(8)[None, :]
    ).remainder(2)[..., None]
    render = torch.zeros_like(target)
    field = _field(6, height, width)
    cfg = _base_cfg(iters=1)
    constraint = _MaskConstraint.from_mask(
        mask.numpy(),
        target.device,
        target.dtype,
        cfg.sigma_cutoff,
        cfg.mask_margin,
        cap_mode="anisotropic",
        undercoverage_band=4.0,
    )
    schedule = SafeScheduleConfig(
        capacity=10,
        storage_policy="fixed_capacity",
        base_active_limit=8,
        detail_tail_max_rows=2,
        detail_tail_batch_rows=1,
        coverage_target_gaussians=7,
        detail_target_gaussians=8,
        event_min_count=1,
    )
    denominator = torch.ones(height, width)
    denominator[19, 19] = 0.0
    score, allowed, persistent, metadata = _detail_tail_score(
        field,
        target,
        render,
        cfg,
        constraint,
        schedule,
        denominator=denominator,
        coherence=torch.zeros(height, width),
    )

    assert metadata["eligible_pixels"] > 0
    assert persistent[18:22, 18:22].max() > 0
    assert not allowed[19, 19]
    assert not torch.isfinite(score[19, 19])
    assert not torch.isfinite(score[3, 20])
    assert torch.isfinite(score[18, 18])
    # A spatially constant color mismatch has no persistent high-frequency tail signal.
    flat_render = torch.full_like(target, 0.25)
    flat_target = torch.zeros_like(target)
    flat_score, _, flat_persistent, flat_metadata = _detail_tail_score(
        field,
        flat_target,
        flat_render,
        cfg,
        constraint,
        schedule,
        denominator=torch.ones_like(denominator),
        coherence=torch.zeros(height, width),
    )
    assert flat_metadata["eligible_pixels"] == 0
    assert flat_persistent.max() == 0
    assert not torch.isfinite(flat_score).any()


def test_moment_split_preserves_integrated_raw_mass_before_caps():
    height = width = 36
    mask = torch.ones(height, width, dtype=torch.bool)
    target = torch.zeros(height, width, 3)
    target[10:24, 10:24] = torch.tensor([0.7, 0.2, 0.9])
    field = _field(6, height, width)
    cfg = _base_cfg(iters=1)
    constraint = _MaskConstraint.from_mask(
        mask.numpy(),
        target.device,
        target.dtype,
        cfg.sigma_cutoff,
        cfg.mask_margin,
        cap_mode="anisotropic",
        undercoverage_band=4.0,
    )
    constraint.apply(field, cfg, refresh=True)
    render = _render(field, cfg, height, width, support_fade_alpha=1.0)
    schedule = SafeScheduleConfig(
        capacity=7,
        coverage_target_gaussians=6,
        detail_target_gaussians=7,
        event_min_count=1,
    )
    before_mass = (
        torch.sigmoid(field.opacities) * field.scales().prod(dim=1)
    ).sum()
    proposal = propose_split(
        field, target, render, cfg, constraint, schedule, 1
    )
    assert proposal is not None
    after_mass = (
        torch.sigmoid(proposal.field.opacities)
        * proposal.field.scales().prod(dim=1)
    ).sum()
    torch.testing.assert_close(after_mass, before_mass, rtol=2e-4, atol=2e-4)
    assert "integrated raw mass" in proposal.metadata["mass_rule"]


def test_merge_rebirth_is_batched_count_neutral_and_reuses_absorbed_partner():
    height = width = 32
    mask = torch.ones(height, width, dtype=torch.bool)
    target = torch.zeros(height, width, 3)
    target[25, 25] = 1.0
    field = GaussianField.from_numpy(
        means=[
            [8.0, 8.0], [8.5, 8.0],
            [15.0, 15.0], [15.5, 15.0],
            [22.0, 8.0], [22.5, 8.0],
        ],
        scales=[[2.0, 2.0]] * 6,
        angles=[0.0] * 6,
        colors=[[0.1, 0.1, 0.1]] * 6,
        opacities=[0.4] * 6,
    )
    cfg = _base_cfg(iters=1, split_scale=0.35)
    constraint = _MaskConstraint.from_mask(
        mask.numpy(),
        target.device,
        target.dtype,
        cfg.sigma_cutoff,
        cfg.mask_margin,
        cap_mode="anisotropic",
        undercoverage_band=4.0,
    )
    constraint.apply(field, cfg, refresh=True)
    render = _render(field, cfg, height, width, support_fade_alpha=1.0)
    schedule = SafeScheduleConfig(
        capacity=6,
        coverage_target_gaussians=6,
        detail_target_gaussians=6,
        event_min_count=1,
        event_spacing_px=3.0,
        redistribution_min_responsibility=1.0,
    )
    proposal = propose_merge_rebirth(
        field, target, render, cfg, constraint, schedule, 1
    )
    assert proposal is not None
    assert proposal.field.n == field.n
    assert proposal.metadata["count_neutral"] is True
    assert proposal.metadata["merged"] == 1
    keep, absorbed = proposal.touched.tolist()
    midpoint = 0.5 * (field.means[keep] + field.means[absorbed])
    torch.testing.assert_close(proposal.field.means[keep], midpoint)
    assert torch.all(proposal.field.scales()[keep] > field.scales()[keep])
    # The absorbed identity is the birth row itself; it never passes through a shared donor list.
    torch.testing.assert_close(
        proposal.field.means[absorbed],
        torch.tensor([25.0, 25.0]),
    )
    assert "direct partner rebirth" in proposal.metadata["merge_rule"]


def test_tiny_safe_schedule_emits_monotone_selected_sequence():
    height = width = 28
    mask = _mask(height, width)
    yy, xx = torch.meshgrid(
        torch.linspace(0, 1, height),
        torch.linspace(0, 1, width),
        indexing="ij",
    )
    target = torch.stack([xx, yy, 0.5 * (xx + yy)], dim=2)
    target = target * mask[..., None]
    field = _field(6, height, width)
    one = (1e-2, 8e-3, 3e-3, 1e-2, 3e-3)

    def phase(name: str, target_n: int | None, lowpass: int = 1):
        return PhaseBudget(name, 1, 1, target_n, one, 0.001, 0.002, lowpass)

    schedule = SafeScheduleConfig(
        capacity=9,
        coverage_target_gaussians=7,
        detail_target_gaussians=8,
        event_min_count=1,
        recovery_steps=1,
        event_spacing_px=3.0,
        coverage_birth_count=1,
        detail_birth_count=1,
        detail_split_count=1,
        boundary_birth_count=1,
        redistribution_count=1,
        stale_patience=1,
        bootstrap=phase("bootstrap", 6, 2),
        coverage=phase("coverage_growth", 7),
        detail=phase("detail_growth", 8),
        boundary=phase("boundary_closure", 9),
        redistribution=phase("redistribution", 9),
        polish=phase("safe_polish", 9),
    )
    result = run_safe_schedule(
        field,
        target,
        mask,
        _base_cfg(iters=1),
        schedule,
        verbose=False,
    )
    assert result["field"].n <= schedule.capacity
    assert result["history"]
    selected = [entry["selected"] for entry in result["history"]]
    for before, after in zip(selected, selected[1:]):
        assert after["foreground_mse"] <= before["foreground_mse"] * (1 + 3e-6) + 1e-10
        assert after["boundary_mse"] <= before["boundary_mse"] * (1 + 3e-6) + 1e-10
        assert after["interior_hole_fraction"] <= before["interior_hole_fraction"]
        assert after["boundary_hole_fraction"] <= before["boundary_hole_fraction"]
    assert math.isfinite(result["metrics"]["foreground_psnr_db"])

    fixed = run_safe_schedule(
        field,
        target,
        mask,
        _base_cfg(iters=1),
        replace(
            schedule,
            capacity=11,
            storage_policy="fixed_capacity",
            base_active_limit=9,
        ),
        verbose=False,
    )
    assert fixed["storage"] == {
        "policy": "fixed_capacity",
        "capacity": 11,
        "base_active_limit": 9,
        "detail_tail_max_rows": 0,
        "detail_tail_activated_rows": 0,
        "initial_active_n": 6,
        "final_active_n": result["field"].n,
        "initial_physical_rows": 11,
        "final_physical_rows": 11,
        "fixed_shape_during_fit": True,
        "output_compacted": True,
    }
    assert fixed["metrics"] == result["metrics"]
    assert [
        (entry["phase"], entry["event"], entry["accepted"], entry["selected"])
        for entry in fixed["history"]
    ] == [
        (entry["phase"], entry["event"], entry["accepted"], entry["selected"])
        for entry in result["history"]
    ]
    for name in (
        "means",
        "log_scales",
        "rotations",
        "colors",
        "opacities",
        "scale_max",
    ):
        torch.testing.assert_close(
            getattr(fixed["field"], name),
            getattr(result["field"], name),
            rtol=0,
            atol=0,
        )


def test_geometric_storage_grows_and_matches_fixed_capacity():
    height = width = 28
    mask = _mask(height, width)
    yy, xx = torch.meshgrid(
        torch.linspace(0, 1, height),
        torch.linspace(0, 1, width),
        indexing="ij",
    )
    target = torch.stack([xx, yy, 0.5 * (xx + yy)], dim=2)
    target = target * mask[..., None]
    field = _field(6, height, width)
    one = (1e-2, 8e-3, 3e-3, 1e-2, 3e-3)

    def phase(name: str, target_n: int | None, lowpass: int = 1):
        return PhaseBudget(name, 1, 1, target_n, one, 0.001, 0.002, lowpass)

    base = SafeScheduleConfig(
        capacity=11,
        base_active_limit=9,
        coverage_target_gaussians=7,
        detail_target_gaussians=8,
        event_min_count=1,
        recovery_steps=1,
        event_spacing_px=3.0,
        coverage_birth_count=1,
        detail_birth_count=1,
        detail_split_count=1,
        boundary_birth_count=1,
        redistribution_count=1,
        stale_patience=1,
        bootstrap=phase("bootstrap", 6, 2),
        coverage=phase("coverage_growth", 7),
        detail=phase("detail_growth", 8),
        boundary=phase("boundary_closure", 9),
        redistribution=phase("redistribution", 9),
        polish=phase("safe_polish", 9),
    )

    fixed = run_safe_schedule(
        field, target, mask, _base_cfg(iters=1),
        replace(base, storage_policy="fixed_capacity"), verbose=False,
    )
    geometric = run_safe_schedule(
        field, target, mask, _base_cfg(iters=1),
        replace(base, storage_policy="geometric", initial_capacity=6), verbose=False,
    )

    # The physical pool starts at the initial field and grows geometrically to the ceiling.
    assert geometric["storage"]["policy"] == "geometric"
    assert geometric["storage"]["initial_physical_rows"] == 6
    assert geometric["storage"]["final_physical_rows"] == 9  # min(capacity=11, base_active_limit=9)
    assert geometric["storage"]["geometric_migrations"] == 1
    assert geometric["storage"]["growth_factor"] == 2.0
    assert geometric["storage"]["fixed_shape_during_fit"] is False
    assert geometric["storage"]["output_compacted"] is True

    # Growth only adds inactive storage, so the fitted outcome is identical to fixed capacity.
    assert geometric["metrics"] == fixed["metrics"]
    assert geometric["field"].n == fixed["field"].n
    for name in ("means", "log_scales", "rotations", "colors", "opacities", "scale_max"):
        torch.testing.assert_close(
            getattr(geometric["field"], name),
            getattr(fixed["field"], name),
            rtol=0,
            atol=0,
        )


def test_geometric_storage_policy_validation():
    valid = SafeScheduleConfig(
        capacity=32,
        storage_policy="geometric",
        base_active_limit=16,
        growth_factor=1.5,
        initial_capacity=8,
        coverage_target_gaussians=12,
        detail_target_gaussians=14,
    )
    valid.validate(6)
    with pytest.raises(ValueError, match="growth_factor"):
        replace(valid, growth_factor=1.0).validate(6)
    with pytest.raises(ValueError, match="initial_capacity"):
        replace(valid, initial_capacity=64).validate(6)  # above capacity
    with pytest.raises(ValueError, match="initial_capacity"):
        replace(valid, initial_capacity=2).validate(6)  # below the initial field
    with pytest.raises(ValueError, match="storage_policy"):
        replace(valid, storage_policy="bogus").validate(6)
    # A geometric fit cannot also reserve a detail tail (that stays fixed_capacity per ADR-0022).
    with pytest.raises(ValueError, match="requires storage_policy"):
        replace(valid, detail_tail_max_rows=4).validate(6)


def test_fixed_reserve_is_activated_only_by_the_late_detail_tail():
    height = width = 32
    mask = _mask(height, width)
    target = torch.zeros(height, width, 3)
    checker = (
        torch.arange(12)[:, None] + torch.arange(12)[None, :]
    ).remainder(2).float()
    target[10:22, 10:22] = checker[..., None]
    target *= mask[..., None]
    field = _field(6, height, width)
    one = (1e-2, 8e-3, 3e-3, 1e-2, 3e-3)

    def phase(name: str, target_n: int):
        return PhaseBudget(name, 1, 1, target_n, one, 0.001, 0.002)

    schedule = SafeScheduleConfig(
        capacity=10,
        storage_policy="fixed_capacity",
        base_active_limit=8,
        detail_tail_max_rows=2,
        detail_tail_batch_rows=1,
        coverage_target_gaussians=7,
        detail_target_gaussians=8,
        event_min_count=1,
        recovery_steps=2,
        event_spacing_px=3.0,
        coverage_birth_count=1,
        detail_birth_count=1,
        detail_split_count=1,
        boundary_birth_count=1,
        redistribution_count=1,
        stale_patience=1,
        bootstrap=phase("bootstrap", 6),
        coverage=phase("coverage_growth", 7),
        detail=phase("detail_growth", 8),
        boundary=phase("boundary_closure", 8),
        redistribution=phase("redistribution", 8),
        polish=phase("safe_polish", 10),
    )
    result = run_safe_schedule(
        field,
        target,
        mask,
        _base_cfg(iters=1),
        schedule,
        verbose=False,
    )

    tail_records = [
        record
        for record in result["history"]
        if record["phase"] == "detail_tail"
    ]
    assert tail_records[0]["event"] == "detail_tail_birth"
    assert tail_records[0]["accepted"] is True
    tail_end = tail_records[-1]["metadata"]
    assert tail_end == {
        "termination_reason": "no_safe_effective_winner",
        "start_n": 8,
        "end_n": 9,
        "activated_rows": 1,
        "configured_max_rows": 2,
        "batch_rows": 1,
        "wave_budget": 2,
        "waves_attempted": 2,
        "waves_accepted": 1,
        "minimum_gain_per_row": 0.0,
    }
    assert result["storage"]["capacity"] == 10
    assert result["storage"]["base_active_limit"] == 8
    assert result["storage"]["detail_tail_activated_rows"] == 1
    assert result["field"].n == 9


def test_tiny_local_neighborhood_schedule_records_local_selection_and_boundary_defects():
    height = width = 28
    mask = _mask(height, width)
    target = torch.zeros(height, width, 3)
    target[mask] = torch.tensor([0.25, 0.55, 0.85])
    field = _field(6, height, width)
    one = (5e-3, 4e-3, 2e-3, 5e-3, 2e-3)

    def phase(name: str, target_n: int | None):
        return PhaseBudget(name, 1, 1, target_n, one, 0.001, 0.002)

    schedule = SafeScheduleConfig(
        capacity=9,
        coverage_target_gaussians=7,
        detail_target_gaussians=8,
        event_min_count=1,
        recovery_steps=1,
        event_spacing_px=3.0,
        coverage_birth_count=1,
        detail_birth_count=1,
        detail_split_count=1,
        boundary_birth_count=1,
        redistribution_count=1,
        refinement_policy="local_neighborhood",
        local_seed_count=2,
        local_neighbor_count=1,
        topology_neighbor_count=1,
        boundary_recycle_at_capacity=True,
        boundary_residual_mse_threshold=1e-3,
        boundary_residual_min_pixels=1,
        bootstrap=phase("bootstrap", 6),
        coverage=phase("coverage_growth", 7),
        detail=phase("detail_growth", 8),
        boundary=phase("boundary_closure", 9),
        redistribution=phase("redistribution", 9),
        polish=phase("safe_polish", 9),
    )
    result = run_safe_schedule(
        field,
        target,
        mask,
        _base_cfg(iters=1),
        schedule,
        verbose=False,
    )
    local_records = [
        record
        for record in result["history"]
        if record["event"] == "local_residual_fit"
    ]
    assert local_records
    assert all(
        record["metadata"]["selection"]["trainable_rows"] < result["field"].n
        for record in local_records
    )
    boundary_end = next(
        record
        for record in result["history"]
        if record["phase"] == "boundary_closure"
        and record["event"] == "phase_end"
    )
    diagnostics = boundary_end["metadata"]["diagnostics"]
    assert diagnostics["band_pixels"] > 0
    assert diagnostics["residual_components"] >= 0
    assert boundary_end["metadata"]["termination_reason"] in {
        "step_budget",
        "metric_target",
        "deterministic_fixed_point",
    }


def test_interior_undercoverage_weight_validation():
    with pytest.raises(ValueError, match="mask_interior_undercoverage_weight"):
        FitConfig(mask_interior_undercoverage_weight=-1.0)
    with pytest.raises(ValueError, match="mask_undercoverage_every"):
        FitConfig(mask_undercoverage_every=0)
