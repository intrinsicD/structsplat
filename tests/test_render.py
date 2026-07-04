# ruff: noqa: E402
import numpy as np
import pytest

torch = pytest.importorskip("torch")
from structsplat.cuda_render import _build_tile_index
from structsplat.gaussians import GaussianField
from structsplat.render import _element_budget, gaussian_activity, render, render_field


def test_element_budget_preserves_chunk_floor_and_unit():
    assert _element_budget(1) == 64 * 4096
    assert _element_budget(64) == 64 * 4096
    assert _element_budget(512) == 512 * 4096


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


def test_support_fade_zeroes_tail_at_cutoff():
    means = np.array([[5.0, 5.0]])
    scales = np.full((1, 2), 1.0)
    colors = np.array([[1.0, 0.0, 0.0]])
    g = GaussianField.from_numpy(means, scales, np.zeros(1), colors)
    hard = render_field(g.means, g.conics(), g.colors, g.radii(3.0), 11, 11, mode="additive")
    faded = render_field(
        g.means, g.conics(), g.colors, g.radii(3.0), 11, 11, mode="additive",
        support_fade=True, sigma_cutoff=3.0,
    )
    assert hard[5, 8, 0] > 0
    assert faded[5, 8, 0] == 0
    assert 0 < faded[5, 7, 0] < hard[5, 7, 0]


def test_support_fade_reduces_activity():
    means = np.array([[5.0, 5.0]])
    scales = np.full((1, 2), 1.0)
    colors = np.ones((1, 3))
    g = GaussianField.from_numpy(means, scales, np.zeros(1), colors)
    hard = gaussian_activity(g.means, g.conics(), g.radii(3.0), 11, 11)
    faded = gaussian_activity(
        g.means, g.conics(), g.radii(3.0), 11, 11,
        support_fade=True, sigma_cutoff=3.0,
    )
    assert 0 < faded[0] < hard[0]


def test_cuda_renderer_is_explicit_about_requirements():
    means = np.array([[3.0, 3.0]])
    scales = np.full((1, 2), 1.5)
    colors = np.array([[1.0, 0, 0]])
    g = GaussianField.from_numpy(means, scales, np.zeros(1), colors)
    with pytest.raises(RuntimeError, match="requires CUDA tensors"):
        render_field(g.means, g.conics(), g.colors, g.radii(3.0), 10, 10, mode="cuda")
    with pytest.raises(RuntimeError, match="requires CUDA tensors"):
        render_field(g.means, g.conics(), g.colors, g.radii(3.0), 10, 10, mode="cuda_tiled")
    with pytest.raises(ValueError, match="requires scales and rotations"):
        render_field(g.means, g.conics(), g.colors, g.radii(3.0), 10, 10, mode="gsplat")
    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError, match="requires CUDA tensors"):
            render_field(
                g.means, g.conics(), g.colors, g.radii(3.0), 10, 10,
                mode="cuda_additive",
            )


def test_cuda_tile_index_culls_to_overlapping_tiles():
    means = torch.tensor([[2.0, 2.0], [20.0, 20.0]])
    radii = torch.tensor([[1, 1], [1, 1]], dtype=torch.long)
    gids, offsets = _build_tile_index(means, radii, H=8, W=8, tile_size=4)
    assert offsets.tolist() == [0, 1, 1, 1, 1]
    assert gids.tolist() == [0]


def test_cuda_exact_matches_reference_when_available():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    rng = np.random.default_rng(1)
    N, H, W = 24, 18, 20
    means = np.stack([rng.uniform(-2, W + 2, N), rng.uniform(-2, H + 2, N)], 1)
    scales = np.exp(rng.uniform(np.log(0.7), np.log(4.0), (N, 2)))
    angles = rng.uniform(0, np.pi, N)
    colors = rng.random((N, 3))
    g = GaussianField.from_numpy(means, scales, angles, colors, device="cuda").trainable()
    try:
        ref = render_field(g.means, g.conics(), g.colors, g.radii(3.0), H, W,
                           mode="normalized")
        cu = render_field(g.means, g.conics(), g.colors, g.radii(3.0), H, W,
                          mode="cuda")
        add_ref = render_field(g.means, g.conics(), g.colors, g.radii(3.0), H, W,
                               mode="additive")
        add_cu = render_field(g.means, g.conics(), g.colors, g.radii(3.0), H, W,
                              mode="cuda_additive")
        torch.cuda.synchronize()
    except RuntimeError as exc:
        pytest.skip(str(exc))
    assert torch.allclose(cu, ref, atol=2e-5, rtol=2e-5)
    assert torch.allclose(add_cu, add_ref, atol=2e-5, rtol=2e-5)


def test_cuda_tiled_exact_matches_reference_when_available():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    rng = np.random.default_rng(11)
    N, H, W = 32, 21, 23
    means = np.stack([rng.uniform(-3, W + 3, N), rng.uniform(-3, H + 3, N)], 1)
    scales = np.exp(rng.uniform(np.log(0.7), np.log(4.0), (N, 2)))
    angles = rng.uniform(0, np.pi, N)
    colors = rng.random((N, 3))
    opac = rng.uniform(0.2, 1.0, N)
    g = GaussianField.from_numpy(means, scales, angles, colors, opacities=opac,
                                  device="cuda").trainable()
    opacity = g.opacity_values()
    try:
        ref = render_field(g.means, g.conics(), g.colors, g.radii(3.0), H, W,
                           mode="normalized", opacities=opacity,
                           support_fade=True, sigma_cutoff=3.0)
        tiled = render_field(g.means, g.conics(), g.colors, g.radii(3.0), H, W,
                             mode="cuda_tiled", opacities=opacity,
                             support_fade=True, sigma_cutoff=3.0)
        add_ref = render_field(g.means, g.conics(), g.colors, g.radii(3.0), H, W,
                               mode="additive", opacities=opacity,
                               support_fade=True, sigma_cutoff=3.0)
        add_tiled = render_field(g.means, g.conics(), g.colors, g.radii(3.0), H, W,
                                 mode="cuda_tiled_additive", opacities=opacity,
                                 support_fade=True, sigma_cutoff=3.0)
        torch.cuda.synchronize()
    except RuntimeError as exc:
        pytest.skip(str(exc))
    assert torch.allclose(tiled, ref, atol=2e-5, rtol=2e-5)
    assert torch.allclose(add_tiled, add_ref, atol=2e-5, rtol=2e-5)


def test_cuda_exact_backward_matches_reference_when_available():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    rng = np.random.default_rng(2)
    N, H, W = 18, 14, 16
    means = np.stack([rng.uniform(0, W - 1, N), rng.uniform(0, H - 1, N)], 1)
    scales = np.exp(rng.uniform(np.log(0.8), np.log(3.0), (N, 2)))
    angles = rng.uniform(0, np.pi, N)
    colors = rng.random((N, 3))
    target = torch.rand(H, W, 3, device="cuda")

    def run(mode):
        f = GaussianField.from_numpy(means, scales, angles, colors, device="cuda").trainable()
        img = render_field(
            f.means, f.conics(), f.colors, f.radii(3.0), H, W, mode=mode,
            support_fade=True, sigma_cutoff=3.0,
        )
        loss = (img - target).square().mean()
        loss.backward()
        torch.cuda.synchronize()
        return img.detach(), f.means.grad.detach(), f.log_scales.grad.detach(), f.colors.grad.detach()

    try:
        ref = run("normalized")
        cu = run("cuda")
    except RuntimeError as exc:
        pytest.skip(str(exc))
    for got, exp in zip(cu, ref, strict=True):
        assert torch.allclose(got, exp, atol=2e-4, rtol=2e-4)


def test_cuda_tiled_backward_matches_reference_when_available():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    rng = np.random.default_rng(12)
    N, H, W = 18, 17, 19
    means = np.stack([rng.uniform(0, W - 1, N), rng.uniform(0, H - 1, N)], 1)
    scales = np.exp(rng.uniform(np.log(0.8), np.log(3.0), (N, 2)))
    angles = rng.uniform(0, np.pi, N)
    colors = rng.random((N, 3))
    target = torch.rand(H, W, 3, device="cuda")

    def run(mode):
        f = GaussianField.from_numpy(means, scales, angles, colors, device="cuda").trainable()
        img = render_field(
            f.means, f.conics(), f.colors, f.radii(3.0), H, W, mode=mode,
            support_fade=True, sigma_cutoff=3.0,
        )
        loss = (img - target).square().mean()
        loss.backward()
        torch.cuda.synchronize()
        return img.detach(), f.means.grad.detach(), f.log_scales.grad.detach(), f.colors.grad.detach()

    try:
        ref = run("normalized")
        tiled = run("cuda_tiled")
    except RuntimeError as exc:
        pytest.skip(str(exc))
    for got, exp in zip(tiled, ref, strict=True):
        assert torch.allclose(got, exp, atol=3e-4, rtol=3e-4)


@pytest.mark.parametrize("normalize", [True, False])
@pytest.mark.parametrize("use_opacities", [False, True])
@pytest.mark.parametrize("dilation", [0.0, 0.3])
@pytest.mark.parametrize("support_fade", [False, True])
def test_cuda_exact_parity_matrix_when_available(normalize, use_opacities, dilation, support_fade):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    rng = np.random.default_rng(7)
    N, H, W = 20, 16, 18
    means = np.stack([rng.uniform(-2, W + 2, N), rng.uniform(-2, H + 2, N)], 1)
    scales = np.exp(rng.uniform(np.log(0.7), np.log(3.5), (N, 2)))
    angles = rng.uniform(0, np.pi, N)
    colors = rng.random((N, 3))
    opac = rng.uniform(0.1, 1.0, N) if use_opacities else None
    target = torch.rand(H, W, 3, device="cuda")
    ref_mode = "normalized" if normalize else "additive"
    cuda_mode = "cuda" if normalize else "cuda_additive"

    def run(mode):
        f = GaussianField.from_numpy(means, scales, angles, colors, opacities=opac,
                                     device="cuda").trainable()
        o = None if opac is None else f.opacity_values()
        img = render_field(f.means, f.conics(dilation), f.colors,
                           f.radii(3.0, dilation), H, W, mode=mode, opacities=o,
                           support_fade=support_fade, sigma_cutoff=3.0)
        loss = (img - target).square().mean()
        loss.backward()
        torch.cuda.synchronize()
        grads = [f.means.grad, f.log_scales.grad, f.colors.grad]
        if opac is not None:
            grads.append(f.opacities.grad)
        return img.detach(), [g.detach() for g in grads]

    try:
        ref_img, ref_g = run(ref_mode)
        cu_img, cu_g = run(cuda_mode)
    except RuntimeError as exc:
        pytest.skip(str(exc))
    assert torch.allclose(cu_img, ref_img, atol=2e-4, rtol=2e-4)
    for got, exp in zip(cu_g, ref_g, strict=True):
        assert torch.allclose(got, exp, atol=3e-4, rtol=3e-4)


def test_cuda_empty_field_matches_reference_zeros_when_available():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    H, W = 8, 9
    empty = torch.zeros(0, 2, device="cuda")
    conics = torch.zeros(0, 3, device="cuda")
    colors = torch.zeros(0, 3, device="cuda")
    radii = torch.zeros(0, 2, dtype=torch.long, device="cuda")
    try:
        ref = render_field(empty, conics, colors, radii, H, W, mode="normalized")
        cu = render_field(empty, conics, colors, radii, H, W, mode="cuda")
    except RuntimeError as exc:
        pytest.skip(str(exc))
    assert torch.count_nonzero(cu) == 0
    assert torch.allclose(cu, ref)


def test_cuda_nonfinite_mean_no_illegal_access_when_available():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    # A diverged fit can emit NaN / huge means; the CUDA path must drop them (zero support)
    # instead of overflowing the tile bounds, matching the reference's off-image behavior.
    H, W = 12, 12
    means = np.array([[5.0, 5.0], [np.nan, 3.0], [1e12, 1e12]])
    scales = np.full((3, 2), 1.5)
    colors = np.ones((3, 3))
    g = GaussianField.from_numpy(means, scales, np.zeros(3), colors, device="cuda")
    try:
        cu = render_field(g.means, g.conics(), g.colors, g.radii(3.0), H, W, mode="cuda")
        torch.cuda.synchronize()
    except RuntimeError as exc:
        pytest.skip(str(exc))
    assert torch.isfinite(cu).all()


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
