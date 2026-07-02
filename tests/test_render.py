import numpy as np
import pytest

torch = pytest.importorskip("torch")
from structsplat.gaussians import GaussianField
from structsplat.render import render


def test_two_gaussian_blend_and_grad():
    means = np.array([[3.0, 3.0], [6.0, 3.0]])
    scales = np.full((2, 2), 1.5)
    colors = np.array([[1.0, 0, 0], [0, 0, 1.0]])
    g = GaussianField.from_numpy(means, scales, np.zeros(2), colors).trainable()
    img = render(g.means, g.conics(), g.colors, g.radii(3.0), 10, 10)
    left, right = img[3, 3], img[3, 6]
    assert left[0] > left[2] and right[2] > right[0]      # red vs blue dominance
    loss = (img - 0.5).abs().mean()
    loss.backward()
    assert g.means.grad is not None and g.colors.grad is not None
