import numpy as np
from structsplat import structure_tensor as st


def _vertical_edge(H=64, W=64):
    img = np.zeros((H, W, 3), np.float32)
    img[:, W // 2:, :] = 1.0
    return img


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
