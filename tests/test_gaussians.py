# ruff: noqa: E402
import numpy as np
import pytest

torch = pytest.importorskip("torch")
from structsplat.gaussians import GaussianField


def _explicit_inv_cov(scale, theta):
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    return np.linalg.inv(R @ np.diag(scale ** 2) @ R.T)


def test_conics_match_explicit_inverse():
    means = np.zeros((3, 2))
    scales = np.array([[3.0, 1.0], [2.0, 5.0], [4.0, 4.0]])
    angles = np.array([0.7, -1.2, 0.3])
    g = GaussianField.from_numpy(means, scales, angles, np.zeros((3, 3)))
    conics = g.conics().numpy()
    for i in range(3):
        got = np.array([[conics[i, 0], conics[i, 1]], [conics[i, 1], conics[i, 2]]])
        assert np.allclose(got, _explicit_inv_cov(scales[i], angles[i]), atol=1e-5)


def test_radii_positive_and_save_load(tmp_path):
    g = GaussianField.from_numpy(np.zeros((5, 2)), np.full((5, 2), 2.0),
                                 np.zeros(5), np.zeros((5, 3)))
    assert (g.radii(3.0) >= 1).all()
    p = tmp_path / "g.npz"
    g.save(str(p))
    h = GaussianField.load(str(p))
    assert torch.allclose(g.means, h.means)


def test_append_preserves_none_opacity_appearance():
    # None opacity renders as 1.0; after appending an opacity-carrying field, the padded
    # logits of the None side must stay ~1.0, not silently become sigmoid(0)=0.5
    a = GaussianField.from_numpy(np.zeros((2, 2)), np.full((2, 2), 2.0),
                                 np.zeros(2), np.zeros((2, 3)))
    b = GaussianField.from_numpy(np.ones((3, 2)), np.full((3, 2), 2.0),
                                 np.zeros(3), np.zeros((3, 3)), opacities=np.zeros(3))
    ab = a.append(b)
    vals = ab.opacity_values()
    assert torch.allclose(vals[:2], torch.ones(2), atol=1e-4)
    ba = b.append(a)
    assert torch.allclose(ba.opacity_values()[3:], torch.ones(2), atol=1e-4)


def test_opacity_save_load(tmp_path):
    g = GaussianField.from_numpy(np.zeros((3, 2)), np.full((3, 2), 2.0),
                                 np.zeros(3), np.zeros((3, 3)), opacities=np.ones(3))
    assert g.opacity_values().shape == (3,)
    p = tmp_path / "g_opacity.npz"
    g.save(str(p))
    h = GaussianField.load(str(p))
    assert h.opacities is not None
    assert torch.allclose(g.opacity_values(), h.opacity_values())
