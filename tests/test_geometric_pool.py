"""Unit coverage for the geometric-growth storage primitive (pool.py).

The schedule and the physical-growth migration are tested in isolation here; the wiring into the
transactional safe schedule (``storage_policy="geometric"``) is exercised in
``tests/test_safe_schedule.py``.
"""
from __future__ import annotations

import pytest
import torch

from structsplat.gaussians import GaussianField
from structsplat.pool import (
    POOL_PARK_COORD,
    POOL_PARK_LOG_SCALE,
    POOL_PARK_OPACITY_LOGIT,
    geometric_capacity_schedule,
    grow_transactional_capacity,
)


def _field(n: int) -> GaussianField:
    return GaussianField(
        torch.randn(n, 2),
        torch.randn(n, 2),
        torch.randn(n),
        torch.rand(n, 3),
        torch.randn(n),
    )


def test_geometric_schedule_grows_to_max_of_required_and_one_step():
    # Like realtime-gs's arena: target = min(cap, max(required, ceil(current * factor))).
    # A geometric step that overshoots the requirement leaves headroom...
    assert geometric_capacity_schedule(4, 5, growth_factor=2.0, max_capacity=100) == 8
    assert geometric_capacity_schedule(8, 9, growth_factor=2.0, max_capacity=100) == 16
    # ...but a requirement beyond one step jumps straight to it (no over-allocation).
    assert geometric_capacity_schedule(4, 30, growth_factor=2.0, max_capacity=100) == 30
    # The cap is a hard ceiling.
    assert geometric_capacity_schedule(64, 100, growth_factor=2.0, max_capacity=100) == 100
    # Already large enough: no growth.
    assert geometric_capacity_schedule(50, 40, growth_factor=2.0, max_capacity=100) == 50
    # A gentler factor still advances by at least one row per step.
    assert geometric_capacity_schedule(10, 11, growth_factor=1.05, max_capacity=100) == 11


def test_geometric_schedule_validates_factor_and_ceiling():
    with pytest.raises(ValueError, match="growth_factor"):
        geometric_capacity_schedule(4, 5, growth_factor=1.0, max_capacity=100)
    with pytest.raises(ValueError, match="exceeds max_capacity"):
        geometric_capacity_schedule(4, 200, growth_factor=2.0, max_capacity=100)


def test_grow_preserves_active_prefix_and_parks_new_tail():
    torch.manual_seed(0)
    field = _field(6)
    before = {
        "means": field.means.clone(),
        "log_scales": field.log_scales.clone(),
        "rotations": field.rotations.clone(),
        "colors": field.colors.clone(),
        "opacities": field.opacities.clone(),
    }

    grown = grow_transactional_capacity(field, 16)
    assert grown.n == 16

    # The existing rows keep their exact values and order.
    assert torch.equal(grown.means[:6], before["means"])
    assert torch.equal(grown.log_scales[:6], before["log_scales"])
    assert torch.equal(grown.rotations[:6], before["rotations"])
    assert torch.equal(grown.colors[:6], before["colors"])
    assert torch.equal(grown.opacities[:6], before["opacities"])

    # The appended rows are parked off-image (exact-zero render contribution).
    assert torch.all(grown.means[6:] == POOL_PARK_COORD)
    assert torch.allclose(grown.log_scales[6:], torch.full((10, 2), POOL_PARK_LOG_SCALE))
    assert torch.all(grown.rotations[6:] == 0.0)
    assert torch.all(grown.colors[6:] == 0.0)
    assert torch.all(grown.opacities[6:] == POOL_PARK_OPACITY_LOGIT)


def test_grow_is_identity_when_target_matches_and_rejects_shrink():
    field = _field(8)
    assert grow_transactional_capacity(field, 8) is field
    with pytest.raises(ValueError, match="below the current row count"):
        grow_transactional_capacity(field, 4)


def test_grow_rejects_color_gradients():
    field = _field(4)
    field.color_grads = torch.zeros(4, 3, 2)
    with pytest.raises(ValueError, match="constant"):
        grow_transactional_capacity(field, 8)


def test_incremental_growth_doubles_and_stays_amortized():
    # Growing one row at a time past a target uses O(log N) migrations: 4 -> 8 -> 16 -> 32 -> 64.
    field = _field(4)
    cap = 4
    steps = [cap]
    while cap < 50:
        cap = geometric_capacity_schedule(cap, cap + 1, growth_factor=2.0, max_capacity=256)
        field = grow_transactional_capacity(field, cap)
        steps.append(cap)
    assert steps == [4, 8, 16, 32, 64]
    assert field.n == 64
    # A single reserve straight to 50 instead jumps exactly there (no wasted rows).
    assert geometric_capacity_schedule(4, 50, growth_factor=2.0, max_capacity=256) == 50
