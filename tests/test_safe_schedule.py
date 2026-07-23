import math

import pytest
import torch

import structsplat.fit as fit_module
from structsplat.config import FitConfig
from structsplat.fit import _MaskConstraint, _render, fit
from structsplat.gaussians import GaussianField
from structsplat.safe_schedule import (
    CommitTolerances,
    PhaseBudget,
    QualityMetrics,
    SafeScheduleConfig,
    _expand_spatial_neighborhood,
    adapt_optimizer_state,
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
