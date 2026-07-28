from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from benchmarks.highpass_births import select_highpass_births
from structsplat.gaussians import GaussianField


def _field():
    return GaussianField.from_numpy(
        means=np.asarray([[8.0, 8.0]], dtype=np.float32),
        scales=np.asarray([[2.0, 2.0]], dtype=np.float32),
        angles=np.asarray([0.0], dtype=np.float32),
        colors=np.asarray([[0.5, 0.5, 0.5]], dtype=np.float32),
    )


def test_forbidden_mask_removes_prior_detail_site():
    height = width = 20
    target = torch.zeros(height, width, 3)
    rendered = target.clone()
    rendered[5, 5] = 1.0
    rendered[14, 14] = 0.8
    constraint = SimpleNamespace(
        sdf_flat=torch.full((height * width,), 20.0),
        margin=0.75,
    )
    initial = select_highpass_births(
        _field(), target, rendered, constraint, 1
    )
    forbidden = torch.zeros(height, width, dtype=torch.bool)
    y = int(initial.sites[0] // width)
    x = int(initial.sites[0] % width)
    forbidden[y, x] = True
    replacement = select_highpass_births(
        _field(),
        target,
        rendered,
        constraint,
        1,
        forbidden_mask=forbidden,
    )

    assert int(replacement.sites[0]) != int(initial.sites[0])
    assert replacement.metadata["forbidden_eligible_pixels"] == 1
    with pytest.raises(ValueError, match="spatial shape"):
        select_highpass_births(
            _field(),
            target,
            rendered,
            constraint,
            1,
            forbidden_mask=torch.zeros(3, 3, dtype=torch.bool),
        )
