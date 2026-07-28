from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from benchmarks.highpass_births import select_highpass_births
from benchmarks.highpass_ridge_births import select_highpass_ridge_births
from structsplat.gaussians import GaussianField


def _field():
    return GaussianField.from_numpy(
        means=np.asarray([[12.0, 12.0]], dtype=np.float32),
        scales=np.asarray([[2.0, 2.0]], dtype=np.float32),
        angles=np.asarray([0.0], dtype=np.float32),
        colors=np.asarray([[0.5, 0.5, 0.5]], dtype=np.float32),
        opacities=np.asarray([1.0], dtype=np.float32),
    )


def test_ridge_births_preserve_sites_and_bound_anisotropy():
    height = width = 28
    target = torch.zeros(height, width, 3)
    rendered = target.clone()
    rendered[5, 4:11] = 1.0
    rendered[14, 12:20] = 0.8
    rendered[22, 5:12] = 0.6
    constraint = SimpleNamespace(
        sdf_flat=torch.full((height * width,), 20.0),
        margin=0.75,
    )
    isotropic = select_highpass_births(
        _field(), target, rendered, constraint, 3
    )
    ridge = select_highpass_ridge_births(
        _field(),
        target,
        rendered,
        constraint,
        3,
        max_long_scale=1.5,
        coherence_power=1.0,
    )

    assert torch.equal(ridge.sites, isotropic.sites)
    scales = ridge.components.scales()
    assert torch.all(scales[:, 0] >= 0.35)
    assert torch.all(scales[:, 0] <= 1.5)
    assert torch.allclose(scales[:, 1], torch.full((3,), 0.35))
    assert torch.all(torch.isfinite(ridge.components.rotations))
    assert ridge.metadata["long_scale_max"] <= 1.5


def test_ridge_births_validate_shape_policy():
    target = torch.zeros(16, 16, 3)
    rendered = target.clone()
    rendered[8, 8] = 1.0
    constraint = SimpleNamespace(
        sdf_flat=torch.full((16 * 16,), 20.0),
        margin=0.75,
    )
    with pytest.raises(ValueError, match="at least short_scale"):
        select_highpass_ridge_births(
            _field(),
            target,
            rendered,
            constraint,
            1,
            max_long_scale=0.2,
        )
    with pytest.raises(ValueError, match="positive"):
        select_highpass_ridge_births(
            _field(),
            target,
            rendered,
            constraint,
            1,
            coherence_power=0.0,
        )
