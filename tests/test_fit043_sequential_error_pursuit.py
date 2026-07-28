import math

import pytest
import torch

from scripts.experiments.fit043_sequential_error_pursuit import (
    _already_satisfied,
    _prefix_exact,
    _sha256,
    _verify_sha256,
    adjusted_stage_target,
)
from structsplat.gaussians import GaussianField


def _field(count: int) -> GaussianField:
    return GaussianField(
        means=torch.arange(2 * count, dtype=torch.float32).reshape(count, 2),
        log_scales=torch.full((count, 2), math.log(0.5)),
        rotations=torch.zeros(count),
        colors=torch.linspace(0.1, 0.9, 3 * count).reshape(count, 3),
        opacities=torch.full((count,), torch.logit(torch.tensor(0.8))),
    )


def test_cumulative_target_transform_is_exact_for_remaining_deficit():
    stage_target = adjusted_stage_target(100.0, 80.0, 0.25)

    assert stage_target == pytest.approx(0.0625)
    assert 80.0 * (1.0 - stage_target) == pytest.approx(75.0)
    assert adjusted_stage_target(100.0, 70.0, 0.25) == 0.0


@pytest.mark.parametrize(
    ("highpass", "laplacian", "protected", "expected"),
    [
        (0.25, 0.20, True, True),
        (0.30, 0.19, True, False),
        (0.24, 0.30, True, False),
        (0.30, 0.30, False, False),
    ],
)
def test_already_satisfied_skip_requires_both_targets_and_protection(
    highpass,
    laplacian,
    protected,
    expected,
):
    assert _already_satisfied(highpass, laplacian, protected) is expected


def test_stale_source_binding_is_rejected(tmp_path):
    source = tmp_path / "field.bin"
    source.write_bytes(b"frozen")
    expected = _sha256(source)
    _verify_sha256(source, expected, "fixture")

    source.write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        _verify_sha256(source, expected, "fixture")


def test_error_prefix_equality_covers_all_inherited_tensors():
    inherited = _field(3)
    candidate = inherited.append(_field(2))

    exact, checks = _prefix_exact(inherited, candidate)
    assert exact is True
    assert all(checks.values())

    candidate.colors[1, 0] += 0.125
    exact, checks = _prefix_exact(inherited, candidate)
    assert exact is False
    assert checks["colors"] is False
