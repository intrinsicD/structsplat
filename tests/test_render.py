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


def _naive_render(g, H, W, opacities=None):
    """Dense O(N*H*W) reference: every Gaussian on every pixel, no support window."""
    ys, xs = torch.meshgrid(torch.arange(H, dtype=torch.float32),
                            torch.arange(W, dtype=torch.float32), indexing="ij")
    conics = g.conics()
    num = torch.zeros(H, W, 3)
    den = torch.zeros(H, W, 1)
    for i in range(g.n):
        dx = xs - g.means[i, 0]
        dy = ys - g.means[i, 1]
        a, b, c = conics[i]
        w = torch.exp(-0.5 * (a * dx * dx + 2 * b * dx * dy + c * dy * dy))
        if opacities is not None:
            w = w * opacities[i]
        num += w[..., None] * g.colors[i]
        den += w[..., None]
    return num / (den + 1e-8)


def test_render_matches_naive_reference():
    rng = np.random.default_rng(0)
    N, H, W = 40, 24, 32
    means = np.stack([rng.uniform(-4, W + 4, N), rng.uniform(-4, H + 4, N)], 1)
    scales = np.exp(rng.uniform(np.log(0.6), np.log(6.0), (N, 2)))
    angles = rng.uniform(0, np.pi, N)
    colors = rng.random((N, 3))
    g = GaussianField.from_numpy(means, scales, angles, colors)
    fast = render(g.means, g.conics(), g.colors, g.radii(6.0), H, W, chunk=1)
    naive = _naive_render(g, H, W)
    assert torch.allclose(fast, naive, atol=2e-3)  # sigma_cutoff=6 leaves ~exp(-18) tails


def test_offimage_gaussian_keeps_inimage_support():
    # center right of the image, extent covering it: the old symmetric radius clamp
    # (rx = min(rx, W)) dropped its entire in-image support (CORE-003)
    means = np.array([[25.0, 5.0]])
    scales = np.array([[10.0, 10.0]])
    colors = np.array([[1.0, 0.0, 0.0]])
    g = GaussianField.from_numpy(means, scales, np.zeros(1), colors).trainable()
    H, W = 10, 10
    activity = gaussian_activity(g.means, g.conics(), g.radii(3.0), H, W)
    assert activity[0] > 0
    img = render(g.means, g.conics(), g.colors, g.radii(3.0), H, W)
    assert img[5, 9, 0] > 0.5
    img.sum().backward()
    assert g.means.grad is not None and g.means.grad.abs().sum() > 0


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
