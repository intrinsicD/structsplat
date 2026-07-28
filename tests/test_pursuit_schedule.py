import math

import pytest
import torch
import structsplat.safe_schedule as safe_schedule_module

from structsplat.config import FitConfig
from structsplat.gaussians import GaussianField
from structsplat.safe_schedule import (
    CommitTolerances,
    PhaseBudget,
    SafeScheduleConfig,
    run_safe_schedule,
)


def _field(count: int, height: int, width: int) -> GaussianField:
    generator = torch.Generator().manual_seed(4)
    means = torch.stack(
        [
            torch.randint(5, width - 5, (count,), generator=generator),
            torch.randint(5, height - 5, (count,), generator=generator),
        ],
        dim=1,
    ).float()
    return GaussianField(
        means=means,
        log_scales=torch.full((count, 2), math.log(1.5)),
        rotations=torch.zeros(count),
        colors=torch.zeros(count, 3),
        opacities=torch.full((count,), torch.logit(torch.tensor(0.8))),
    )


def _mask(height: int, width: int) -> torch.Tensor:
    mask = torch.zeros(height, width, dtype=torch.bool)
    mask[3:-3, 3:-3] = True
    return mask


def _cfg() -> FitConfig:
    return FitConfig(
        iters=1,
        renderer="normalized",
        render_chunk=64,
        pixel_loss="l2",
        ssim_weight=0.0,
        support_fade=True,
        mask_contain=True,
        mask_margin=0.75,
        mask_cap_mode="anisotropic",
        mask_cap_refresh_every=1,
        loss_weighting="mask",
        checkpoint_policy="terminal",
        log_every=1,
    )


@pytest.mark.parametrize("force_rejection", [False, True])
def test_orthogonal_pursuit_commits_or_rolls_back_transactionally(
    force_rejection,
    monkeypatch,
):
    height = width = 40
    mask = _mask(height, width)
    target = torch.zeros(height, width, 3)
    for y, x, value in (
        (10, 10, 1.0),
        (14, 23, 0.9),
        (24, 12, 0.8),
        (27, 27, 0.7),
    ):
        target[y, x] = value
    target *= mask[..., None]
    field = _field(6, height, width)
    learning_rates = (2e-3, 1.5e-3, 5e-4, 5e-3, 5e-4)

    def disabled(name: str) -> PhaseBudget:
        return PhaseBudget(
            name,
            0,
            1,
            6,
            learning_rates,
            0.0,
            0.0,
        )

    schedule = SafeScheduleConfig(
        capacity=6,
        storage_policy="dynamic",
        coverage_target_gaussians=6,
        detail_target_gaussians=6,
        pursuit_tail_enabled=True,
        pursuit_tail_batch_rows=4,
        pursuit_tail_max_rows=4,
        pursuit_tail_blur_sigma=1.0,
        pursuit_tail_nms_radius=1,
        pursuit_tail_deep_offset=2.0,
        pursuit_tail_highpass_target=0.0,
        pursuit_tail_laplacian_target=0.0,
        pursuit_tail_color_maxiter=32,
        event_min_count=1,
        tolerances=CommitTolerances(
            minimum_relative_gain=2.0 if force_rejection else 1e-8
        ),
        bootstrap=disabled("bootstrap"),
        coverage=disabled("coverage_growth"),
        detail=disabled("detail_growth"),
        boundary=disabled("boundary_closure"),
        redistribution=disabled("redistribution"),
        polish=disabled("safe_polish"),
    )
    if not force_rejection:
        monkeypatch.setattr(
            safe_schedule_module,
            "relative_detail_reductions",
            lambda before, after: {
                "highpass": 0.30,
                "laplacian": 0.25,
            },
        )
    result = run_safe_schedule(
        field,
        target,
        mask,
        _cfg(),
        schedule,
        verbose=False,
    )
    pursuit = result["pursuit_tail"]

    assert pursuit["enabled"] is True
    assert pursuit["waves_attempted"] == 1
    assert pursuit["unique_sites"] in (0, 4)
    assert len(pursuit["site_sha256"]) == 64
    assert len(pursuit["site_set_sha256"]) == 64
    assert len(pursuit["waves"][0]["batch_site_sha256"]) == 64
    assert len(pursuit["waves"][0]["all_sites_sha256"]) == 64
    assert len(pursuit["waves"][0]["batch_site_set_sha256"]) == 64
    assert len(pursuit["waves"][0]["all_site_set_sha256"]) == 64
    assert result["metrics"]["outside_max_abs"] == 0.0
    assert result["metrics"]["outside_coverage_max"] == 0.0
    if force_rejection:
        assert pursuit["termination_reason"] == "protected_rejection"
        assert pursuit["activated_rows"] == 0
        assert pursuit["waves_accepted"] == 0
        assert result["field"].n == 6
        assert "no_material_gain" in pursuit["waves"][0]["protected_reasons"]
    else:
        assert pursuit["termination_reason"] == "targets_reached"
        assert pursuit["target_reached"] is True
        assert pursuit["activated_rows"] == 4
        assert pursuit["waves_accepted"] == 1
        assert pursuit["waves"][0]["inherited_rows_frozen"] is True
        assert (
            pursuit["site_sha256"]
            == pursuit["waves"][0]["batch_site_sha256"]
            == pursuit["waves"][0]["all_sites_sha256"]
        )
        assert (
            pursuit["site_set_sha256"]
            == pursuit["waves"][0]["batch_site_set_sha256"]
            == pursuit["waves"][0]["all_site_set_sha256"]
        )
        assert result["field"].n == 10
        assert result["storage"]["pursuit_tail_activated_rows"] == 4
