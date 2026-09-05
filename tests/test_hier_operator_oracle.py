import pytest
import torch

from benchmarks.hier_additive_controls import additive_render, pack
from benchmarks.hier_operator_oracle import FAMILIES, SHAPE, action_names, finite_actions, fixture


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_oracle_fixtures_and_bank_are_reproducible_count_funded_and_owned(family, seed):
    field, target = fixture(family, seed)
    original = pack(field).clone()
    actions, packet, base = finite_actions(field, target)
    assert len(actions) == 15 and [a.name for a in actions] == action_names()
    assert all(a.field.n == 3 for a in actions)
    assert bool(torch.isfinite(packet.signed).all())
    assert bool(((target >= 0) & (target <= 1)).all())
    second, again = fixture(family, seed)
    torch.testing.assert_close(pack(field), pack(second), rtol=0, atol=0)
    torch.testing.assert_close(target, again, rtol=0, atol=0)
    for action in actions:
        assert action.donor in (1, 2) if action.family in ("birth", "split") else action.donor is None
    actions[-1].field.colors.zero_()
    torch.testing.assert_close(pack(field), original, rtol=0, atol=0)
    torch.testing.assert_close(additive_render(field, *SHAPE), base, rtol=0, atol=0)


def test_color_quadratic_predictor_is_exact_for_its_finite_linear_edit():
    field, target = fixture("color", 1)
    actions, _packet, base = finite_actions(field, target)
    objective = 0.5 * (base - target).double().square().mean()
    for action in actions:
        if action.family == "color":
            gain = objective - 0.5 * (additive_render(action.field, *SHAPE) - target).double().square().mean()
            assert action.predicted_gain == pytest.approx(float(gain), rel=2e-5, abs=1e-10)


def test_support_gap_has_residual_without_existing_pixel_gradient():
    field, target = fixture("support_gap", 0)
    actions, packet, base = finite_actions(field, target)
    assert float((base - target).square().sum()) > 0.1
    assert float(packet.absolute.max()) < 1e-10
    assert any(action.family == "birth" and action.predicted_gain > 0 for action in actions)


def test_split_halves_parent_colors_and_really_removes_paid_donor():
    field, target = fixture("two_lobes", 2)
    actions, _packet, _base = finite_actions(field, target)
    for action in actions:
        if action.family == "split":
            torch.testing.assert_close(action.field.colors[:2].sum(0), field.colors[0])
            survivor = 2 if action.donor == 1 else 1
            torch.testing.assert_close(pack(action.field)[2], pack(field)[survivor])

