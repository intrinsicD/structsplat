# ruff: noqa: E402
import numpy as np
import pytest

torch = pytest.importorskip("torch")
from structsplat.gaussians import GaussianField
from structsplat.render import gaussian_activity, render, render_field


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


def test_gaussian_activity_marks_visible_support():
    means = np.array([[3.0, 3.0], [50.0, 50.0]])
    scales = np.full((2, 2), 1.5)
    colors = np.ones((2, 3))
    g = GaussianField.from_numpy(means, scales, np.zeros(2), colors)
    activity = gaussian_activity(g.means, g.conics(), g.radii(3.0), 10, 10, chunk=1)
    assert activity.shape == (2,)
    assert torch.isfinite(activity).all()
    assert activity[0] > 0
    assert activity[1] == 0


def test_additive_renderer_runs_and_differs():
    means = np.array([[3.0, 3.0], [3.5, 3.0]])
    scales = np.full((2, 2), 1.5)
    colors = np.array([[0.5, 0, 0], [0.5, 0, 0]])
    g = GaussianField.from_numpy(means, scales, np.zeros(2), colors)
    norm = render_field(g.means, g.conics(), g.colors, g.radii(3.0), 10, 10, mode="normalized")
    add = render_field(g.means, g.conics(), g.colors, g.radii(3.0), 10, 10, mode="additive")
    assert torch.isfinite(add).all()
    assert add[3, 3, 0] > norm[3, 3, 0]


def test_opacity_changes_normalized_blend():
    means = np.array([[3.0, 3.0], [3.0, 3.0]])
    scales = np.full((2, 2), 1.5)
    colors = np.array([[1.0, 0, 0], [0, 0, 1.0]])
    g = GaussianField.from_numpy(means, scales, np.zeros(2), colors)
    equal = render_field(g.means, g.conics(), g.colors, g.radii(3.0), 10, 10)
    biased = render_field(
        g.means, g.conics(), g.colors, g.radii(3.0), 10, 10,
        opacities=torch.tensor([0.9, 0.1]),
    )
    assert equal[3, 3, 0] < biased[3, 3, 0]
