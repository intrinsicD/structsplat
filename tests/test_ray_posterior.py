from __future__ import annotations

import math

import pytest
import torch

from structsplat.realtime_gs_ray_posterior import (
    RayPosteriorConfig,
    robust_depth_posterior,
)


def _posterior(costs: torch.Tensor, valid: torch.Tensor | None = None):
    return robust_depth_posterior(
        costs,
        torch.ones_like(costs, dtype=torch.bool) if valid is None else valid,
        dustbin_cost=0.65,
        best_view_count=2,
        min_evidence_views=2,
        temperature=0.08,
        view_dispersion_weight=0.25,
    )


def test_robust_posterior_uses_consistent_views_and_ignores_one_occluded_outlier() -> None:
    # Depth 0 is mediocre everywhere.  Depth 1 has three mutually consistent views and one
    # occlusion.  Depth 2 has one deceptively perfect repeated-texture observation but disagrees
    # with the remaining views.  Best-two dispersion makes the coherent mode win.
    costs = torch.tensor(
        [
            [
                [0.35, 0.36, 0.34, 0.37],
                [0.10, 0.12, 0.11, 0.95],
                [0.01, 0.25, 0.27, 0.28],
            ]
        ],
        dtype=torch.float32,
    )
    result = _posterior(costs)

    assert result.best_index.tolist() == [1]
    assert result.evidence_views.tolist() == [[2, 2, 2]]
    assert result.posterior[0, 1] > result.posterior[0, 2]
    assert 0.0 <= float(result.normalized_entropy[0]) <= 1.0
    assert float(result.margin[0]) > 0.0


def test_dustbin_rejects_depth_without_two_real_observations() -> None:
    costs = torch.tensor(
        [[[0.05, 0.8, 0.9], [0.20, 0.21, 0.95]]], dtype=torch.float32
    )
    valid = torch.tensor([[[True, False, False], [True, True, False]]])
    result = _posterior(costs, valid)

    assert result.eligible_depth.tolist() == [[False, True]]
    assert result.best_index.tolist() == [1]
    assert result.posterior[0, 0] == 0.0
    assert result.posterior[0, 1] == 1.0


def test_depth_tie_selects_first_sample_deterministically() -> None:
    costs = torch.full((2, 4, 3), 0.2)
    first = _posterior(costs)
    second = _posterior(costs.clone())

    assert first.best_index.tolist() == [0, 0]
    assert torch.equal(first.best_index, second.best_index)
    assert torch.equal(first.posterior, second.posterior)
    assert torch.allclose(first.posterior, torch.full_like(first.posterior, 0.25))


def test_no_eligible_depth_is_explicit_not_nan() -> None:
    costs = torch.zeros((3, 5, 2), dtype=torch.float32)
    valid = torch.zeros_like(costs, dtype=torch.bool)
    result = _posterior(costs, valid)

    assert not result.eligible_depth.any()
    assert torch.equal(result.posterior, torch.zeros_like(result.posterior))
    assert result.best_index.tolist() == [0, 0, 0]
    assert torch.isinf(result.best_cost).all()
    assert torch.equal(result.normalized_entropy, torch.ones(3))


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"target_views": 1, "best_view_count": 2}, "best_view_count"),
        ({"best_view_count": 1, "min_evidence_views": 2}, "min_evidence_views"),
        ({"fine_samples": 4}, "fine_samples"),
        ({"dustbin_cost": 0.0}, "dustbin_cost"),
        ({"dino_weight": 0.0, "detail_weight": 0.0}, "descriptor weight"),
        ({"min_primary_fraction": 1.1}, "min_primary_fraction"),
        ({"feature_storage_dtype": "float64"}, "feature_storage_dtype"),
        ({"view_dispersion_weight": -0.1}, "view_dispersion_weight"),
    ],
)
def test_config_rejects_invalid_values(changes: dict[str, object], message: str) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        RayPosteriorConfig(**changes)


def test_config_defaults_are_finite_and_closed_for_fallback() -> None:
    config = RayPosteriorConfig()

    assert config.apply_reciprocal is True
    assert config.min_reciprocal_views >= 1
    assert config.min_primary_fraction >= 0.75
    assert config.fine_samples % 2 == 1
    assert math.isfinite(config.posterior_temperature)
