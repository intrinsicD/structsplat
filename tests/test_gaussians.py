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


def test_affine_color_save_load_and_append_padding(tmp_path):
    grads = np.zeros((2, 2, 3), dtype=np.float32)
    grads[:, 0, 0] = [0.1, 0.2]
    a = GaussianField.from_numpy(
        np.zeros((2, 2)), np.full((2, 2), 2.0), np.zeros(2), np.zeros((2, 3)),
        color_grads=grads,
    )
    p = tmp_path / "affine.npz"
    a.save(str(p))
    loaded = GaussianField.load(str(p))
    assert loaded.color_grads is not None
    assert torch.allclose(loaded.color_grads, a.color_grads)

    b = GaussianField.from_numpy(np.ones((1, 2)), np.full((1, 2), 2.0),
                                 np.zeros(1), np.ones((1, 3)))
    ab = a.append(b)
    assert ab.color_grads is not None
    assert torch.allclose(ab.color_grads[:2], a.color_grads)
    assert torch.allclose(ab.color_grads[2], torch.zeros(2, 3))


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


def test_from_numpy_copies_and_does_not_alias_caller():
    # torch.as_tensor is zero-copy for float32 CPU ndarrays; from_numpy must clone so an
    # optimizer step never mutates the caller's init arrays in place (CORE-004).
    means = np.zeros((4, 2), dtype=np.float32)
    scales = np.full((4, 2), 2.0, dtype=np.float32)
    angles = np.zeros(4, dtype=np.float32)
    colors = np.zeros((4, 3), dtype=np.float32)
    opacities = np.ones(4, dtype=np.float32)
    scale_max = np.full((4, 2), 5.0, dtype=np.float32)
    color_grads = np.zeros((4, 2, 3), dtype=np.float32)
    g = GaussianField.from_numpy(means, scales, angles, colors,
                                 opacities=opacities, scale_max=scale_max,
                                 color_grads=color_grads)
    assert not np.shares_memory(g.means.numpy(), means)
    assert not np.shares_memory(g.rotations.numpy(), angles)
    assert not np.shares_memory(g.colors.numpy(), colors)
    assert not np.shares_memory(g.opacities.numpy(), opacities)
    assert not np.shares_memory(g.scale_max.numpy(), scale_max)
    assert not np.shares_memory(g.color_grads.numpy(), color_grads)
    # mutating the field must leave the caller arrays untouched
    with torch.no_grad():
        g.means.add_(1.0)
        g.colors.add_(1.0)
    assert np.allclose(means, 0.0) and np.allclose(colors, 0.0)


def test_fitconfig_rejects_negative_dilation():
    from structsplat.config import FitConfig
    with pytest.raises(ValueError, match="aa_dilation must be >= 0"):
        FitConfig(aa_dilation=-0.1)
    FitConfig(aa_dilation=0.0)  # valid
    with pytest.raises(ValueError, match="color_basis must be constant or affine"):
        FitConfig(color_basis="quadratic")
    with pytest.raises(ValueError, match="color_grad_l2 must be >= 0"):
        FitConfig(color_grad_l2=-1.0)
    with pytest.raises(ValueError, match="adaptive_count requires"):
        FitConfig(adaptive_count=True)
    with pytest.raises(ValueError, match="target_ms_ssim"):
        FitConfig(target_ms_ssim=1.5)
    with pytest.raises(ValueError, match="target_bpp"):
        FitConfig(target_bpp=0.0)
    with pytest.raises(ValueError, match="adaptive_split_mode"):
        FitConfig(adaptive_split_mode="bogus")


def test_fitconfig_rejects_invalid_relocation_downsample():
    from structsplat.config import FitConfig
    with pytest.raises(ValueError, match="relocate_residual_downsample must be >= 1"):
        FitConfig(relocate_residual_downsample=0)
    FitConfig(relocate_residual_downsample=1)


def test_negative_dilation_raises():
    g = GaussianField.from_numpy(np.zeros((3, 2)), np.full((3, 2), 2.0),
                                 np.zeros(3), np.zeros((3, 3)))
    with pytest.raises(ValueError, match="dilation must be >= 0"):
        g.conics(dilation=-0.5)
    with pytest.raises(ValueError, match="dilation must be >= 0"):
        g.radii(3.0, dilation=-0.5)


def test_opacity_save_load(tmp_path):
    g = GaussianField.from_numpy(np.zeros((3, 2)), np.full((3, 2), 2.0),
                                 np.zeros(3), np.zeros((3, 3)), opacities=np.ones(3))
    assert g.opacity_values().shape == (3,)
    p = tmp_path / "g_opacity.npz"
    g.save(str(p))
    h = GaussianField.load(str(p))
    assert h.opacities is not None
    assert torch.allclose(g.opacity_values(), h.opacity_values())
