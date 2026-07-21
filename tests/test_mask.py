# ruff: noqa: E402
"""Mask-contained fitting (CORE-010): geometry exactness, containment, penalty, loss, regression."""
import subprocess
import sys

import numpy as np
import pytest

from structsplat import mask as m


# --------------------------------------------------------------------------- geometry (NumPy)

def _brute_squared_edt(seed: np.ndarray) -> np.ndarray:
    H, W = seed.shape
    ys, xs = np.nonzero(seed)
    out = np.full((H, W), np.inf)
    for y in range(H):
        for x in range(W):
            d = (ys - y) ** 2 + (xs - x) ** 2
            out[y, x] = d.min() if d.size else np.inf
    return out


def test_squared_edt_and_feature_transform_match_brute_force():
    rng = np.random.default_rng(0)
    for _ in range(50):
        H, W = int(rng.integers(3, 14)), int(rng.integers(3, 14))
        seed = rng.random((H, W)) < 0.35
        if not seed.any():
            seed[0, 0] = True
        exp = _brute_squared_edt(seed)
        assert np.allclose(m.squared_edt(seed), exp)
        edt = m._edt_2d(seed)
        for y in range(H):
            for x in range(W):
                fy, fx = int(edt.fy[y, x]), int(edt.fx[y, x])
                assert seed[fy, fx]  # reported feature is an actual seed
                assert (fy - y) ** 2 + (fx - x) ** 2 == exp[y, x]  # at the exact distance


def test_edt_handles_concave_masks_and_holes():
    seed = np.ones((12, 12), bool)
    seed[0:6, 6:12] = False   # concave corner
    seed[9, 2] = False        # hole
    assert np.allclose(m.squared_edt(seed), _brute_squared_edt(seed))


def test_signed_distance_sign_and_values():
    inside = np.zeros((20, 20), bool)
    inside[5:15, 5:15] = True
    sdf = m.signed_distance(inside)
    assert (sdf[inside] > 0).all() and (sdf[~inside] < 0).all()
    assert abs(sdf[5, 5] - 1.0) < 1e-9      # interior corner: nearest outside is 1 px
    assert abs(sdf[9, 9] - 5.0) < 1e-9      # deepest interior pixel


def test_signed_distance_all_inside_is_finite():
    inside = np.ones((8, 8), bool)
    sdf = m.signed_distance(inside)
    assert np.isfinite(sdf).all() and (sdf > 0).all()


def test_erode_is_a_monotone_subset():
    inside = np.zeros((20, 20), bool)
    inside[4:16, 4:16] = True
    e1 = m.erode(inside, 2.0)
    e2 = m.erode(inside, 4.0)
    assert e2.sum() < e1.sum() < inside.sum()
    assert bool((inside | e1 == inside).all()) and bool((e1 | e2 == e1).all())


def test_nearest_inside_index_lands_inside():
    inside = np.zeros((16, 16), bool)
    inside[6:10, 6:10] = True
    nn = m.nearest_inside_index(inside)
    for flat in (0, 16 * 16 - 1, 15):  # far corners
        ty, tx = divmod(int(nn[flat]), 16)
        assert inside[ty, tx]
    # already-inside pixels map to themselves
    flat = 6 * 16 + 6
    assert int(nn[flat]) == flat


def test_nearest_inside_index_empty_region_is_minus_one():
    empty = np.zeros((5, 5), bool)
    assert (m.nearest_inside_index(empty) == -1).all()


def test_color_dilate_preserves_inside_and_fills_outside():
    rng = np.random.default_rng(2)
    inside = np.zeros((16, 16), bool)
    inside[4:12, 4:12] = True
    img = rng.random((16, 16, 3)).astype(np.float32)
    out = m.color_dilate(img, inside)
    assert np.array_equal(out[inside], img[inside])
    assert out.shape == img.shape
    # every filled pixel copies some inside pixel's color
    for y in range(16):
        for x in range(16):
            if not inside[y, x]:
                assert np.any(np.all(np.isclose(img[inside], out[y, x]), axis=1))


def test_as_bool_mask_variants():
    assert m.as_bool_mask(np.array([[0, 255], [255, 0]], np.uint8)).tolist() == [
        [False, True], [True, False]]
    rgba = np.zeros((2, 2, 4), np.float32)
    rgba[0, 0, 3] = 1.0
    assert m.as_bool_mask(rgba)[0, 0] and not m.as_bool_mask(rgba)[1, 1]


def test_empty_mask_geometry_rejected():
    with pytest.raises(ValueError):
        m.MaskGeometry.build(np.zeros((6, 6), bool), erode_radius=1.0)


def test_mask_module_imports_without_torch():
    # Init-time math must not pull torch (core invariant 1).
    code = (
        "import sys; sys.modules['torch'] = None; "
        "import numpy as np; import structsplat.mask as mm; "
        "assert mm.signed_distance(np.ones((4,4), bool)).shape == (4,4)"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


# --------------------------------------------------------------------------- fit / init (torch)

torch = pytest.importorskip("torch")
from structsplat.config import FitConfig, InitConfig
from structsplat.fit import _MaskConstraint, fit
from structsplat.gaussians import GaussianField
from structsplat.init import build_field, build_masked_field


def _disk(H=48, W=48, r=15):
    yy, xx = np.mgrid[0:H, 0:W]
    inside = ((yy - H // 2) ** 2 + (xx - W // 2) ** 2) <= r * r
    img = np.stack([xx / W, yy / H, 0.4 + 0.0 * xx], -1).astype(np.float32)
    return (img * inside[..., None]).astype(np.float32), inside


def _means_inside(field, inside, cutoff, margin):
    mx = field.means.detach().cpu().numpy()
    H, W = inside.shape
    eroded = m.erode(inside, margin + cutoff * m.MIN_SCALE)
    ix = np.clip(np.round(mx[:, 0]).astype(int), 0, W - 1)
    iy = np.clip(np.round(mx[:, 1]).astype(int), 0, H - 1)
    return bool(eroded[iy, ix].all())


def _ellipses_inside(field, inside, cutoff):
    """Sufficient containment check: sigma_cutoff * max effective axis <= signed distance."""
    mx = field.means.detach().cpu().numpy()
    H, W = inside.shape
    sdf = m.signed_distance(inside)
    eff = field.effective_scales().detach().cpu().numpy()
    ix = np.clip(np.round(mx[:, 0]).astype(int), 0, W - 1)
    iy = np.clip(np.round(mx[:, 1]).astype(int), 0, H - 1)
    reach = cutoff * eff.max(axis=1)
    return bool((reach <= sdf[iy, ix] + 1e-4).all())


def test_masked_init_contains_every_seed_and_ellipse():
    img, inside = _disk()
    icfg = InitConfig(strategy="quadtree_wse", num_gaussians=150, seed=0)
    field = build_masked_field(img, inside, icfg, sigma_cutoff=3.0, mask_margin=1.5)
    assert _means_inside(field, inside, 3.0, 1.5)
    assert _ellipses_inside(field, inside, 3.0)


def test_hard_containment_exact_zero_outside_with_support_fade():
    img, inside = _disk()
    target = torch.as_tensor(img)
    icfg = InitConfig(strategy="quadtree_wse", num_gaussians=150, seed=0)
    fcfg = FitConfig(iters=60, mask_contain=True, support_fade=True,
                     loss_weighting="mask", log_every=100)
    field = build_masked_field(img, inside, icfg, sigma_cutoff=fcfg.sigma_cutoff,
                               mask_margin=fcfg.mask_margin)
    out = fit(field, target, fcfg, mask=inside, verbose=False)
    render = out["render"].detach().cpu().numpy()
    assert float(np.abs(render[~inside]).max()) == 0.0
    assert _ellipses_inside(out["field"], inside, fcfg.sigma_cutoff)


def test_containment_survives_split_prune_relocate():
    img, inside = _disk(H=40, W=40, r=14)
    target = torch.as_tensor(img)
    icfg = InitConfig(strategy="quadtree_wse", num_gaussians=80, seed=0)
    fcfg = FitConfig(iters=45, mask_contain=True, support_fade=True, loss_weighting="mask",
                     split_every=12, split_count=10, prune_every=18, prune_min_activity=1e-6,
                     relocate_every=15, relocate_count=5, log_every=100)
    field = build_masked_field(img, inside, icfg, sigma_cutoff=fcfg.sigma_cutoff,
                               mask_margin=fcfg.mask_margin)
    out = fit(field, target, fcfg, mask=inside, verbose=False)
    render = out["render"].detach().cpu().numpy()
    assert float(np.abs(render[~inside]).max()) == 0.0


def test_coverage_penalty_has_outward_gradients():
    H = W = 32
    inside = np.zeros((H, W), bool)
    inside[:, :16] = True
    mc = _MaskConstraint.from_mask(inside, torch.device("cpu"), torch.float32, 3.0, 1.5)
    means = torch.tensor([[15.0, 16.0]], requires_grad=True)
    log_scales = torch.log(torch.tensor([[4.0, 4.0]])).requires_grad_(True)
    field = GaussianField(means, log_scales, torch.zeros(1, requires_grad=True),
                          torch.ones(1, 3, requires_grad=True))
    cfg = FitConfig(iters=1, sigma_cutoff=3.0)
    cov = mc.coverage(field, cfg, support_fade_alpha=0.0)
    cov_value = float(cov.detach())
    cov.backward()
    # gradient descent (-grad) must pull the mean deeper inside and shrink the scale
    assert cov_value > 0.0
    assert float(means.grad[0, 0]) > 0.0
    assert float(log_scales.grad[0, 0]) > 0.0


def test_coverage_penalty_reduces_outside_weight_over_fit():
    img, inside = _disk()
    target = torch.as_tensor(img)
    icfg = InitConfig(strategy="quadtree_wse", num_gaussians=120, seed=0)

    def outside_weight(coverage_weight):
        fcfg = FitConfig(iters=50, mask_coverage_weight=coverage_weight, support_fade=False,
                         loss_weighting="mask", log_every=100)
        field = build_masked_field(img, inside, icfg, contain=False,
                                   sigma_cutoff=fcfg.sigma_cutoff, mask_margin=fcfg.mask_margin)
        torch.manual_seed(0)
        out = fit(field, target, fcfg, mask=inside, verbose=False)
        render = out["render"].detach().cpu().numpy()
        return float(np.abs(render[~inside]).mean())

    assert outside_weight(2.0) < outside_weight(0.0)


def test_mask_loss_weighting_zeroes_outside_pixels():
    from structsplat.fit import _prepare_loss_weight_map
    inside = np.zeros((10, 10), bool)
    inside[2:8, 2:8] = True
    target = torch.zeros(10, 10, 3)
    cfg = FitConfig(iters=1, loss_weighting="mask")
    w = _prepare_loss_weight_map(target, cfg, torch.as_tensor(inside.astype(np.float32)))
    assert w[inside].eq(1.0).all() and w[~inside].eq(0.0).all()
    with pytest.raises(ValueError):
        _prepare_loss_weight_map(target, cfg, None)


def test_no_mask_path_is_unchanged():
    rng = np.random.default_rng(1)
    img = rng.random((24, 24, 3)).astype(np.float32)
    target = torch.as_tensor(img)
    icfg = InitConfig(strategy="iso_blue_noise", num_gaussians=40, seed=3)
    torch.manual_seed(7)
    r1 = fit(build_field(img, icfg), target, FitConfig(iters=25, log_every=100), verbose=False)
    torch.manual_seed(7)
    r2 = fit(build_field(img, icfg), target, FitConfig(iters=25, log_every=100), verbose=False,
             mask=None)
    assert abs(r1["psnr"] - r2["psnr"]) < 1e-9


def test_fit_requires_mask_when_a_mask_feature_is_on():
    target = torch.zeros(12, 12, 3)
    field = build_field(np.zeros((12, 12, 3), np.float32), InitConfig(num_gaussians=10, seed=0))
    with pytest.raises(ValueError):
        fit(field, target, FitConfig(iters=1, mask_contain=True), verbose=False)


def test_build_masked_field_rejects_scale_cap_mode_when_containing():
    img, inside = _disk(H=24, W=24, r=8)
    icfg = InitConfig(strategy="iso_blue_noise", num_gaussians=20, seed=0,
                      scale_cap_mode="hard", scale_cap_max=5.0)
    with pytest.raises(ValueError):
        build_masked_field(img, inside, icfg, contain=True)


def test_config_validation_mask_fields():
    with pytest.raises(ValueError):
        FitConfig(iters=1, mask_margin=0.0)
    with pytest.raises(ValueError):
        FitConfig(iters=1, mask_coverage_weight=-1.0)
    with pytest.raises(ValueError):
        FitConfig(iters=1, loss_weighting="bogus")
