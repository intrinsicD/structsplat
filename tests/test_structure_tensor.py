import numpy as np
from structsplat import structure_tensor as st
from structsplat.config import InitConfig
from structsplat import density as de


def _vertical_edge(H=64, W=64):
    img = np.zeros((H, W, 3), np.float32)
    img[:, W // 2:, :] = 1.0
    return img


def test_near_blank_noisy_image_labels_not_saturated():
    # sparse-content guard (INIT-005): on a near-blank image with only sensor noise, the old
    # percentile(energy, 99) reference was itself noise-scaled, so labels saturated to non-flat
    # and the density binarized. The floored reference must keep such an image mostly flat.
    rng = np.random.default_rng(0)
    img = np.clip(0.5 + 0.01 * rng.standard_normal((64, 64, 3)), 0, 1).astype(np.float32)
    t = st.compute(img)
    assert (t.label == 0).mean() > 0.7         # not saturated to edge/corner
    assert (t.label == 2).mean() < 0.05        # essentially no spurious corners
    # density must not binarize: density_power still moves the pmf and mass is not all at the max
    d1 = de.density_from_tensor_and_image(img, t, InitConfig(density_power=1.0))
    d3 = de.density_from_tensor_and_image(img, t, InitConfig(density_power=3.0))
    assert np.abs(d1 - d3).max() > 1e-6
    assert (d1 >= d1.max() * 0.999).mean() < 0.05


def test_structured_image_flat_fraction_preserved():
    # the floored reference must be inert on a clean structured image (median energy ~ 0)
    assert (st.compute(_vertical_edge()).label == 0).mean() > 0.75


def test_edge_orientation_and_labels():
    t = st.compute(_vertical_edge())
    col = t.across_edge_angle[:, 32]
    ang = np.mod(col, np.pi)
    # vertical edge -> gradient horizontal -> angle ~ 0 (mod pi)
    assert np.mean((ang < 0.25) | (ang > np.pi - 0.25)) > 0.8
    # flat and edge labels both present
    labels = set(np.unique(t.label).tolist())
    assert 0 in labels and 1 in labels


def test_energy_concentrates_on_edge():
    t = st.compute(_vertical_edge())
    edge = t.energy[:, 30:34].mean()
    flat = t.energy[:, 5:15].mean()
    assert edge > 10 * (flat + 1e-9)


def test_along_edge_is_orthogonal():
    t = st.compute(_vertical_edge())
    d = np.mod(t.along_edge_angle - t.across_edge_angle, np.pi)
    assert np.allclose(d, np.pi / 2, atol=1e-4)


def test_gradient_operator_variants_find_edge():
    from structsplat.config import StructureTensorConfig

    for operator in ("central", "sobel", "scharr"):
        t = st.compute(_vertical_edge(), StructureTensorConfig(gradient_operator=operator))
        edge = t.energy[:, 30:34].mean()
        flat = t.energy[:, 5:15].mean()
        assert edge > 5 * (flat + 1e-9)


def test_rgb_tensor_sees_isoluminant_edge():
    from structsplat.config import StructureTensorConfig

    # red|green edge with identical Rec.709 luma: invisible to the luma tensor
    img = np.zeros((32, 32, 3), np.float32)
    img[:, :16] = [1.0, 0.0, 0.0]
    img[:, 16:] = [0.0, 0.2126 / 0.7152, 0.0]
    t_luma = st.compute(img, StructureTensorConfig(color_space="luma"))
    t_rgb = st.compute(img, StructureTensorConfig(color_space="rgb"))
    assert t_rgb.energy.max() > 50 * (t_luma.energy.max() + 1e-9)
    # and on a luma edge both agree on the orientation
    t2 = st.compute(_vertical_edge(), StructureTensorConfig(color_space="rgb"))
    ang = np.mod(t2.across_edge_angle[:, 32], np.pi)
    assert np.mean((ang < 0.25) | (ang > np.pi - 0.25)) > 0.8
