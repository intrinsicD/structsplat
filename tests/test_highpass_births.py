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
        opacities=np.asarray([1.0], dtype=np.float32),
    )


def test_highpass_births_are_nested_deterministic_and_spatially_separated():
    height = width = 24
    target = torch.zeros(height, width, 3)
    rendered = target.clone()
    peaks = ((4, 4), (4, 16), (12, 9), (18, 18), (19, 4))
    for rank, (y, x) in enumerate(peaks):
        rendered[y, x] = 1.0 - 0.1 * rank
    constraint = SimpleNamespace(
        sdf_flat=torch.full((height * width,), 20.0),
        margin=0.75,
    )

    full = select_highpass_births(
        _field(),
        target,
        rendered,
        constraint,
        4,
        nms_radius=2,
    )
    prefix = select_highpass_births(
        _field(),
        target,
        rendered,
        constraint,
        2,
        nms_radius=2,
    )
    repeated = select_highpass_births(
        _field(),
        target,
        rendered,
        constraint,
        4,
        nms_radius=2,
    )

    assert torch.equal(prefix.sites, full.sites[:2])
    assert torch.equal(repeated.sites, full.sites)
    assert torch.equal(repeated.scores, full.scores)
    assert torch.all(full.scores[:-1] >= full.scores[1:])
    xy = full.components.means
    chebyshev = (xy[:, None] - xy[None, :]).abs().amax(dim=2)
    off_diagonal = ~torch.eye(4, dtype=torch.bool)
    assert torch.all(chebyshev[off_diagonal] > 2)
    assert torch.allclose(full.components.scales(), torch.full((4, 2), 0.35))
    assert torch.allclose(
        torch.sigmoid(full.components.opacities),
        torch.full((4,), 0.8),
    )


def test_highpass_births_respect_deep_interior_and_validate_parameters():
    height = width = 16
    target = torch.zeros(height, width, 3)
    rendered = target.clone()
    rendered[3, 3] = 1.0
    rendered[12, 12] = 0.8
    sdf = torch.full((height, width), 20.0)
    sdf[3, 3] = 1.0
    constraint = SimpleNamespace(sdf_flat=sdf.reshape(-1), margin=0.75)

    selected = select_highpass_births(
        _field(),
        target,
        rendered,
        constraint,
        1,
        nms_radius=1,
    )
    assert selected.components.means.tolist() == [[12.0, 12.0]]

    with pytest.raises(ValueError, match="positive"):
        select_highpass_births(
            _field(), target, rendered, constraint, 0
        )
    with pytest.raises(ValueError, match="at least 0.35"):
        select_highpass_births(
            _field(), target, rendered, constraint, 1, scale=0.2
        )
