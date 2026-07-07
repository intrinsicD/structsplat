# ruff: noqa: E402
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from structsplat import init as I
from structsplat import structure_tensor as ST
from structsplat.config import InitConfig


def _toy(H=48, W=64):
    img = np.zeros((H, W, 3), np.float32)
    img[:, W // 2:] = [0.9, 0.85, 0.2]
    img[10:25, 8:24] = [0.1, 0.3, 0.8]
    return img


def _assert_field_ok(f, n):
    assert f.n == n
    for t in (f.means, f.log_scales, f.rotations, f.colors):
        assert torch.isfinite(t).all()
    assert (f.means[:, 0] >= -0.5).all() and (f.means[:, 0] <= 63.5).all()
    assert (f.means[:, 1] >= -0.5).all() and (f.means[:, 1] <= 47.5).all()


def _quadtree_leaves_reference(density, n):
    H, W = density.shape
    mass_sat = I._sat(density)

    def priority(cell):
        x0, y0, x1, y1 = cell
        area = (x1 - x0) * (y1 - y0)
        return float(I._rect_sum(mass_sat, x0, y0, x1, y1)), area

    leaves = [(0, 0, W, H)]
    while len(leaves) < n:
        remaining = n - len(leaves)
        choices = []
        for i, cell in enumerate(leaves):
            children = I._split_cell(cell)
            if children and len(children) - 1 <= remaining:
                choices.append((priority(cell), i, children))
        if not choices:
            break
        _, idx, children = max(choices, key=lambda x: x[0])
        leaves.pop(idx)
        leaves.extend(children)

    if len(leaves) < n:
        child_pool = []
        for cell in leaves:
            child_pool.extend(I._split_cell(cell))
        child_pool.sort(key=priority, reverse=True)
        leaves.extend(child_pool[:n - len(leaves)])

    if len(leaves) > n:
        leaves = sorted(leaves, key=priority, reverse=True)[:n]
    return leaves


def test_quadtree_heap_matches_reference_leaves():
    rng = np.random.default_rng(12)
    gradient = np.arange(63, dtype=np.float64).reshape(7, 9)
    densities = [
        np.ones((7, 9), dtype=np.float64),
        rng.random((7, 9)),
        (gradient % 5) / 5.0,
    ]
    for density in densities:
        for n in (0, 1, 2, 3, 4, 5, 8, 13, 20, 37, 64, 80):
            assert I._quadtree_leaves(density, n) == _quadtree_leaves_reference(density, n)


def _feature_run_lengths_reference(tensor, pts, angles, scfg, icfg):
    H, W = tensor.energy.shape
    scfg = scfg or I.StructureTensorConfig()
    energy = np.maximum(tensor.energy, 0.0)
    ref = ST.energy_reference(energy)
    floor = ST.flat_threshold(energy, scfg.flat_frac, ref)
    local_energy = I._nearest(energy, pts)
    max_steps = int(np.ceil(max(H, W)))
    if icfg.scale_cap_max is not None:
        max_steps = max(1, min(max_steps, int(np.ceil(icfg.scale_cap_max
                                                      * icfg.scale_feature_sigma))))
    lengths = np.full(len(pts), np.inf, dtype=np.float64)
    for i, (p, angle, e0) in enumerate(zip(pts, angles, local_energy)):
        if e0 < floor:
            continue
        threshold = max(floor, float(e0) * icfg.scale_feature_energy_frac)
        direction = np.array([np.cos(angle), np.sin(angle)], dtype=np.float64)

        def walk(sign):
            last = 0
            for step in range(1, max_steps + 1):
                q = p + sign * step * direction
                x = int(round(q[0]))
                y = int(round(q[1]))
                if x < 0 or x >= W or y < 0 or y >= H:
                    break
                if tensor.label[y, x] == 2:
                    break
                if energy[y, x] < threshold:
                    break
                last = step
            return last

        lengths[i] = walk(1.0) + walk(-1.0) + 1.0
    return lengths


def test_feature_run_lengths_matches_reference():
    rng = np.random.default_rng(4)
    H, W = 15, 17
    energy = rng.random((H, W), dtype=np.float32) * 0.4 + 0.2
    energy[:, :2] = 0.0
    energy[6:9, 8:11] = 1.5
    label = np.ones((H, W), dtype=np.uint8)
    label[energy < 0.05] = 0
    label[3, 4] = 2
    label[11, 12] = 2
    zeros = np.zeros((H, W), dtype=np.float32)
    tensor = ST.StructureTensor(
        lam1=zeros, lam2=zeros, across_edge_angle=zeros, coherence=zeros,
        energy=energy, label=label, energy_ref=ST.energy_reference(energy),
    )
    pts = np.array([
        [1.2, 1.1],
        [8.0, 7.0],
        [15.4, 13.7],
        [4.1, 3.0],
        [12.0, 11.2],
        [0.1, 8.0],
    ], dtype=np.float64)
    angles = np.array([0.0, np.pi / 4.0, -np.pi / 3.0, np.pi / 2.0, 0.2, -0.7])
    cfg = InitConfig(
        scale_cap_mode="feature",
        scale_cap_max=4.0,
        scale_feature_sigma=2.5,
        scale_feature_energy_frac=0.5,
    )

    got = I._feature_run_lengths(tensor, pts, angles, None, cfg)
    expected = _feature_run_lengths_reference(tensor, pts, angles, None, cfg)

    assert np.array_equal(got, expected)


def test_feature_run_lengths_uses_cached_energy_reference(monkeypatch):
    H, W = 9, 11
    energy = np.zeros((H, W), dtype=np.float32)
    energy[2:7, 5] = 1.0
    label = np.where(energy > 0, 1, 0).astype(np.uint8)
    zeros = np.zeros((H, W), dtype=np.float32)
    tensor = ST.StructureTensor(
        lam1=zeros, lam2=zeros, across_edge_angle=zeros, coherence=zeros,
        energy=energy, label=label, energy_ref=ST.energy_reference(energy),
    )
    pts = np.array([[5.0, 4.0]], dtype=np.float64)
    angles = np.array([np.pi / 2.0])
    cfg = InitConfig(scale_cap_mode="feature", scale_cap_max=4.0)
    expected = I._feature_run_lengths(tensor, pts, angles, None, cfg)

    def fail_energy_reference(_energy):
        raise AssertionError("energy_reference should not be recomputed")

    monkeypatch.setattr(ST, "energy_reference", fail_energy_reference)
    got = I._feature_run_lengths(tensor, pts, angles, None, cfg)

    assert np.array_equal(got, expected)


@pytest.mark.parametrize("mode", I.SAMPLING_MODES)
def test_every_sampling_mode_builds_exact_n(mode):
    img = _toy()
    for strat in ("iso_blue_noise", "aniso_flanking"):
        f = I.build_field(img, InitConfig(strategy=strat, num_gaussians=200,
                                          sampling_mode=mode, seed=0))
        _assert_field_ok(f, 200)


@pytest.mark.parametrize("mode", ["dart_throwing", "halton", "farthest_point", "cvt"])
def test_new_sampling_modes_are_seed_deterministic(mode):
    img = _toy()
    cfg = InitConfig(num_gaussians=120, sampling_mode=mode, seed=7)
    a = I.build_field(img, cfg)
    b = I.build_field(img, cfg)
    assert torch.equal(a.means, b.means)
    c = I.build_field(img, InitConfig(num_gaussians=120, sampling_mode=mode, seed=8))
    assert not torch.equal(a.means, c.means)


def test_orientation_modes():
    img = _toy()
    base = I.build_field(img, InitConfig(num_gaussians=150, seed=0))
    zero = I.build_field(img, InitConfig(num_gaussians=150, orientation_mode="zero", seed=0))
    rand = I.build_field(img, InitConfig(num_gaussians=150, orientation_mode="random", seed=0))
    assert torch.equal(zero.rotations, torch.zeros(150))
    assert torch.equal(base.means, zero.means)  # orientation must not move the points
    assert not torch.equal(base.rotations, rand.rotations)
    with pytest.raises(ValueError):
        I.build_field(img, InitConfig(num_gaussians=10, orientation_mode="bogus"))


def test_scale_mode_knn_tracks_local_spacing():
    img = _toy()
    f = I.build_field(img, InitConfig(num_gaussians=150, scale_mode="knn", seed=0))
    _assert_field_ok(f, 150)
    s = torch.exp(f.log_scales)
    assert (s > 0).all() and float(s.max()) < 64.0


def test_nn_spacing_matches_broadcast_reference_across_chunks():
    rng = np.random.default_rng(0)
    pts = rng.random((37, 2)) * np.array([64.0, 48.0])
    d2 = ((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(d2, np.inf)
    expected = np.clip(np.sqrt(d2.min(axis=1)), 0.75, 12.0)

    got = I._nn_spacing(pts, r_min=0.75, r_max=12.0, chunk=5)

    assert np.allclose(got, expected, rtol=1e-12, atol=1e-12)


def test_nn_spacing_rejects_nonpositive_chunk():
    with pytest.raises(ValueError, match="chunk must be > 0"):
        I._nn_spacing(np.zeros((2, 2)), chunk=0)


def test_candidate_oversample_below_one_is_rejected():
    with pytest.raises(ValueError, match="candidate_oversample must be >= 1"):
        InitConfig(candidate_oversample=0.5)
    InitConfig(candidate_oversample=1.0)  # valid


def test_background_layer_reserves_budget_and_marks_rows():
    img = _toy()
    cfg = InitConfig(
        strategy="random",
        num_gaussians=20,
        background_fraction=0.25,
        background_grid=4,
        seed=0,
    )
    f = I.build_field(img, cfg)

    _assert_field_ok(f, 20)
    assert I.background_count(cfg) == 5
    assert f.background_count == 5
    assert torch.equal(f.background_mask[:5], torch.ones(5, dtype=torch.bool))
    assert torch.equal(f.background_mask[5:], torch.zeros(15, dtype=torch.bool))
    bg_scales = torch.exp(f.log_scales[:5])
    assert torch.allclose(bg_scales[:, 0], bg_scales[:, 1])
    assert torch.isfinite(f.colors[:5]).all()


def test_background_fraction_validation():
    with pytest.raises(ValueError, match="background_fraction"):
        InitConfig(background_fraction=1.0, background_grid=4)
    with pytest.raises(ValueError, match="background_grid"):
        InitConfig(background_fraction=0.1, background_grid=0)


def test_flank_offset_clears_blur_width():
    # the offset floor is applied after the fraction, so flanked edge centers clear the blur
    # width (INIT-005) instead of degenerating toward on-edge placement.
    img = np.full((64, 64, 3), 0.1, np.float32)
    img[:, 32:] = 0.9
    onedge = I.build_field(img, InitConfig(strategy="aniso_onedge", num_gaussians=400, seed=0))
    flanked = I.build_field(img, InitConfig(strategy="aniso_flanking", num_gaussians=400, seed=0))
    # near the ridge, flanked centers sit measurably farther from x=31.5 than on-edge centers
    on_x = onedge.means[:, 0].numpy()
    fl_x = flanked.means[:, 0].numpy()
    on_near = np.abs(on_x - 31.5) < 6
    fl_near = np.abs(fl_x - 31.5) < 6
    assert np.abs(fl_x[fl_near] - 31.5).mean() > np.abs(on_x[on_near] - 31.5).mean()


def test_two_sided_colors_stay_on_the_center_side():
    # step edge: a two_sided color must come from the side of the edge its center is on
    # (the parity sign used for flanking is arbitrary for off-ridge starts)
    img = np.full((64, 64, 3), 0.1, np.float32)
    img[:, 32:] = 0.9
    f = I.build_field(img, InitConfig(strategy="aniso_flanking", num_gaussians=400,
                                      color_mode="two_sided", seed=0))
    x = f.means[:, 0].numpy()
    c = f.colors[:, 0].numpy()
    near = np.abs(x - 31.5) < 4
    agree = np.mean((x[near] > 31.5) == (c[near] > 0.5))
    assert agree > 0.98


def test_jittered_grid_drops_evenly():
    # a count that forces truncation: with bottom-row dropping the max y-gap doubles
    img = _toy(H=60, W=60)
    f = I.build_field(img, InitConfig(strategy="iso_blue_noise", num_gaussians=91,
                                      sampling_mode="jittered_grid", seed=0))
    assert f.n == 91
    ys = np.sort(np.unique(np.round(f.means[:, 1].numpy() / 6.0)))
    assert ys.max() >= 8  # points still reach the bottom rows of the grid


@pytest.mark.parametrize("strategy,color", [
    ("quadtree_aggregate", "aggregate"),
    ("quadtree_hybrid", "aggregate"),
    ("quadtree_wse", "bilinear"),
])
def test_quadtree_variants_are_exact_and_deterministic(strategy, color):
    img = _toy()
    cfg = InitConfig(
        strategy=strategy,
        num_gaussians=128,
        density_mode="variance",
        color_mode=color,
        seed=0,
    )
    a = I.build_field(img, cfg)
    b = I.build_field(img, cfg)
    _assert_field_ok(a, 128)
    assert torch.equal(a.means, b.means)
    assert torch.equal(a.colors, b.colors)
    assert (a.colors >= 0).all() and (a.colors <= 1).all()


def test_hard_scale_cap_is_stored_and_applied():
    img = _toy()
    f = I.build_field(
        img,
        InitConfig(
            strategy="aniso_flanking",
            num_gaussians=128,
            density_mode="variance",
            scale_cap_mode="hard",
            scale_cap_max=3.0,
            seed=0,
        ),
    )
    assert f.scale_max is not None
    assert torch.isfinite(f.scale_max).all()
    assert float(torch.exp(f.log_scales).max()) <= 3.0 + 1e-6


def test_feature_relative_scale_caps_use_local_feature_scale():
    H, W = 6, 8
    zeros = np.zeros((H, W), dtype=np.float32)
    label = np.zeros((H, W), dtype=np.uint8)
    label[:, 3:5] = 1
    tensor = ST.StructureTensor(
        lam1=zeros, lam2=zeros, across_edge_angle=zeros, coherence=zeros,
        energy=label.astype(np.float32), label=label, energy_ref=1.0,
    )
    pts = np.array([[3.0, 2.0], [1.0, 2.0]], dtype=np.float64)
    scales = np.array([[12.0, 4.0], [12.0, 4.0]], dtype=np.float64)
    feature_scale = np.array([2.0, 2.0], dtype=np.float64)
    cfg = InitConfig(
        scale_cap_mode="feature_rel",
        scale_feature_rel_along=4.0,
        scale_feature_rel_across=1.0,
    )

    caps = I._scale_caps(tensor, pts, np.zeros(2), scales, feature_scale, cfg, None)

    assert caps is not None
    np.testing.assert_allclose(caps[0], [8.0, 2.0])
    assert np.isinf(caps[1]).all()


def test_feature_relative_scale_caps_bind_resolution_invariant_fraction():
    def continuous_edge(h, w):
        yy, xx = np.mgrid[0:h, 0:w]
        x = xx / max(w - 1, 1)
        y = yy / max(h - 1, 1)
        img = np.zeros((h, w, 3), dtype=np.float32)
        img[..., 0] = (x > 0.48).astype(np.float32)
        img[..., 1] = (y > 0.55).astype(np.float32) * 0.75
        img[..., 2] = ((x + y) > 1.05).astype(np.float32) * 0.5
        return img

    def bind_fraction(side):
        img = continuous_edge(int(round(side * 0.75)), side)
        field = I.build_field(
            img,
            InitConfig(
                strategy="aniso_onedge",
                num_gaussians=256,
                sampling_mode="density_random",
                scale_cap_mode="feature_rel",
                scale_feature_rel_along=3.0,
                scale_feature_rel_across=1.0,
                seed=3,
            ),
        )
        assert field.scale_max is not None
        caps = field.scale_max
        scales = torch.exp(field.log_scales)
        finite = torch.isfinite(caps)
        bound = finite & torch.isclose(scales, caps, rtol=1e-4, atol=1e-4)
        return float(bound.any(dim=1).float().mean())

    frac160 = bind_fraction(160)
    frac768 = bind_fraction(768)

    assert frac160 > 0.0
    assert abs(frac160 - frac768) <= 0.10
