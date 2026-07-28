import math

import numpy as np
import torch

from structsplat.config import FitConfig
from structsplat.detail_pursuit import (
    fine_detail_metrics,
    select_pursuit_births,
    solve_partial_colors_normalized,
)
from structsplat.fit import _MaskConstraint, _render
from structsplat.gaussians import GaussianField


def _constraint(height: int, width: int) -> _MaskConstraint:
    mask = np.zeros((height, width), dtype=bool)
    mask[1:-1, 1:-1] = True
    return _MaskConstraint.from_mask(
        mask,
        "cpu",
        torch.float32,
        sigma_cutoff=3.0,
        margin=0.75,
        cap_mode="anisotropic",
        undercoverage_band=4.0,
    )


def _field(
    means: torch.Tensor,
    colors: torch.Tensor,
    *,
    scale: float = 1.0,
) -> GaussianField:
    count = int(means.shape[0])
    return GaussianField(
        means=means.clone().float(),
        log_scales=torch.full((count, 2), math.log(scale)),
        rotations=torch.zeros(count),
        colors=colors.clone().float(),
        opacities=torch.full((count,), torch.logit(torch.tensor(0.8))),
    )


def _cfg() -> FitConfig:
    return FitConfig(
        iters=1,
        renderer="normalized",
        render_chunk=64,
        support_fade=True,
        mask_contain=True,
        mask_margin=0.75,
        mask_cap_mode="anisotropic",
        loss_weighting="mask",
        checkpoint_policy="terminal",
    )


def test_pursuit_selection_is_deterministic_spaced_and_exact_site_forbidden():
    height = width = 32
    target = torch.zeros(height, width, 3)
    rendered = torch.zeros_like(target)
    for y, x, value in (
        (9, 9, 1.0),
        (11, 10, 0.9),
        (20, 20, 0.8),
        (23, 9, 0.7),
    ):
        rendered[y, x] = value
    field = _field(
        torch.tensor([[16.0, 16.0]]),
        torch.zeros(1, 3),
    )
    constraint = _constraint(height, width)

    first = select_pursuit_births(
        field,
        target,
        rendered,
        constraint,
        3,
        blur_sigma=1.5,
        nms_radius=2,
        deep_offset=2.0,
        scale=0.35,
        opacity=0.8,
    )
    repeated = select_pursuit_births(
        field,
        target,
        rendered,
        constraint,
        3,
        blur_sigma=1.5,
        nms_radius=2,
        deep_offset=2.0,
        scale=0.35,
        opacity=0.8,
    )

    torch.testing.assert_close(first.sites, repeated.sites, rtol=0, atol=0)
    assert first.components is not None
    assert first.components.n == 3
    coordinates = [
        (int(site) // width, int(site) % width)
        for site in first.sites
    ]
    for index, (y0, x0) in enumerate(coordinates):
        for y1, x1 in coordinates[index + 1 :]:
            assert max(abs(y0 - y1), abs(x0 - x1)) > 2

    forbidden = torch.zeros(height, width, dtype=torch.bool)
    forbidden.reshape(-1)[first.sites[0]] = True
    next_wave = select_pursuit_births(
        field,
        target,
        rendered,
        constraint,
        2,
        blur_sigma=1.5,
        nms_radius=2,
        deep_offset=2.0,
        scale=0.35,
        opacity=0.8,
        forbidden_mask=forbidden,
    )
    assert int(first.sites[0]) not in set(int(site) for site in next_wave.sites)


def test_partial_color_solve_freezes_inherited_rows_and_reduces_error():
    height = width = 24
    base = _field(
        torch.tensor([[7.0, 8.0], [16.0, 15.0]]),
        torch.tensor([[0.2, 0.4, 0.1], [0.7, 0.1, 0.3]]),
        scale=1.4,
    )
    added = _field(
        torch.tensor([[12.0, 11.0]]),
        torch.zeros(1, 3),
        scale=0.8,
    )
    candidate = base.append(added)
    truth = candidate.detached()
    truth.colors[-1] = torch.tensor([0.9, 0.25, 0.6])
    cfg = _cfg()
    target = _render(truth, cfg, height, width, support_fade_alpha=1.0)
    before = _render(candidate, cfg, height, width, support_fade_alpha=1.0)

    result = solve_partial_colors_normalized(
        candidate,
        target,
        cfg,
        torch.tensor([2]),
        ridge=1e-6,
        max_iterations=64,
        tolerance=1e-7,
    )
    after = _render(result.field, cfg, height, width, support_fade_alpha=1.0)

    for name in ("means", "log_scales", "rotations", "colors", "opacities"):
        torch.testing.assert_close(
            getattr(result.field, name)[: base.n],
            getattr(base, name),
            rtol=0,
            atol=0,
        )
    assert float((after - target).square().mean()) < (
        0.01 * float((before - target).square().mean())
    )
    assert result.relative_residual < 1e-5


def test_fine_detail_metrics_are_zero_for_an_exact_reconstruction():
    height = width = 24
    target = torch.rand(height, width, 3)
    metrics = fine_detail_metrics(
        target,
        target,
        _constraint(height, width),
        blur_sigma=1.5,
        deep_offset=2.0,
    )

    assert metrics.pixels > 0
    assert metrics.highpass_mse == 0.0
    assert metrics.laplacian_mse == 0.0
