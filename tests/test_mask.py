# ruff: noqa: E402
"""Mask-contained fitting (CORE-010) and boundary coverage (CORE-011): geometry exactness,
containment, anisotropic caps, penalties, boundary spawning, loss, regression."""
import math
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


# ------------------------------------------------------------------ boundary coverage (CORE-011)

def _halfplane(H=40, W=40, edge=20):
    inside = np.zeros((H, W), bool)
    inside[:, :edge] = True
    return inside


def test_gaussian_smooth_preserves_constants_and_reduces_variation():
    rng = np.random.default_rng(3)
    const = np.full((12, 17), 0.7)
    assert np.allclose(m.gaussian_smooth(const, 2.0), const)
    noisy = rng.random((24, 24))
    smoothed = m.gaussian_smooth(noisy, 2.0)
    assert smoothed.std() < noisy.std()


def test_boundary_normals_point_inward_and_vanish_when_unreliable():
    inside = _halfplane()
    n = m.boundary_normals(m.signed_distance(inside))
    # straight vertical edge: inward is -x everywhere near the boundary
    assert np.allclose(n[20, 15], [-1.0, 0.0], atol=1e-3)
    assert np.allclose(np.hypot(n[..., 0], n[..., 1])[5:35, 10:18], 1.0, atol=1e-3)
    # disk centre sits on the medial axis: the smoothed gradient collapses to ~zero
    _, disk = _disk()
    nd = m.boundary_normals(m.signed_distance(disk))
    assert float(np.hypot(nd[24, 24, 0], nd[24, 24, 1])) < 1e-6


def test_anisotropic_cap_certifies_tangent_elongation_on_straight_edge():
    inside = _halfplane()
    mc = _MaskConstraint.from_mask(inside, torch.device("cpu"), torch.float32, 3.0, 1.5,
                                   cap_mode="anisotropic")
    # tangent-aligned Gaussian (long axis vertical, theta=pi/2) close to the straight edge,
    # pressing at its isotropic reach so certification triggers
    means = torch.tensor([[15.0, 20.0]])
    d = 20.0 - 15.0 - 1.5  # SDF - margin at x=15 is 5 - 1.5 = 3.5
    iso_cap = d / 3.0
    log_scales = torch.log(torch.tensor([[2.0 * iso_cap, 0.4]]))
    field = GaussianField(means, log_scales, torch.tensor([math.pi / 2]), torch.ones(1, 3))
    mc.apply(field)
    caps = field.scale_max
    # short (across, x-ish after rotation) axis keeps the isotropic cap; long axis is certified
    # well beyond it on a straight edge
    assert float(caps[0, 1]) <= iso_cap + 1e-4
    assert float(caps[0, 0]) > 2.0 * iso_cap
    # and never below the isotropic cap anywhere
    assert float(caps.min()) >= 0.35 - 1e-6


def test_anisotropic_cap_binds_at_corners():
    H = W = 40
    inside = np.zeros((H, W), bool)
    inside[:20, :20] = True  # square corner at (20, 20)
    mc = _MaskConstraint.from_mask(inside, torch.device("cpu"), torch.float32, 3.0, 1.5,
                                   cap_mode="anisotropic")

    def certified_long(x, y):
        means = torch.tensor([[float(x), float(y)]])
        log_scales = torch.log(torch.tensor([[3.0, 0.4]]))
        field = GaussianField(means, log_scales, torch.tensor([math.pi / 2]),
                              torch.ones(1, 3))
        mc.apply(field)
        return float(field.scale_max[0, 0])

    # same depth from the left edge: mid-edge row can elongate along the tangent much further
    # than a row whose tangent runs into the corner
    assert certified_long(15, 10) > certified_long(15, 18) + 0.5


def test_anisotropic_containment_property_random_fields():
    from structsplat.fit import _render

    rng = np.random.default_rng(5)
    for trial in range(3):
        H = W = 40
        yy, xx = np.mgrid[0:H, 0:W]
        cx, cy = rng.uniform(14, 26, size=2)
        r1, r2 = rng.uniform(8, 13, size=2)
        inside = (((xx - cx) / r1) ** 2 + ((yy - cy) / r2) ** 2) <= 1.0
        inside[int(cy), int(cx)] = True
        mc = _MaskConstraint.from_mask(inside, torch.device("cpu"), torch.float32, 3.0, 1.5,
                                       cap_mode="anisotropic")
        n = 40
        torch.manual_seed(trial)
        means = torch.rand(n, 2) * torch.tensor([[float(W - 1), float(H - 1)]])
        log_scales = torch.log(torch.rand(n, 2) * 6.0 + 0.35)
        field = GaussianField(means, log_scales, torch.rand(n) * 6.28,
                              torch.rand(n, 3))
        mc.apply(field)
        cfg = FitConfig(iters=1, support_fade=True, log_every=100)
        img = _render(field, cfg, H, W, support_fade_alpha=1.0)
        outside = img.detach().cpu().numpy()[~inside]
        assert float(np.abs(outside).max()) == 0.0


def test_anisotropic_fit_keeps_exact_zero_outside_and_unlocks_tangent_scales():
    img, inside = _disk()
    target = torch.as_tensor(img)
    icfg = InitConfig(strategy="quadtree_wse", num_gaussians=150, seed=0)
    fcfg = FitConfig(iters=60, mask_contain=True, mask_cap_mode="anisotropic",
                     mask_cap_refresh_every=5, support_fade=True, loss_weighting="mask",
                     log_every=100)
    field = build_masked_field(img, inside, icfg, sigma_cutoff=fcfg.sigma_cutoff,
                               mask_margin=fcfg.mask_margin, cap_mode="anisotropic")
    out = fit(field, target, fcfg, mask=inside, verbose=False)
    render = out["render"].detach().cpu().numpy()
    assert float(np.abs(render[~inside]).max()) == 0.0
    # somewhere a certified cap exceeds every possible isotropic cap component
    caps = out["field"].scale_max
    assert float((caps.max(dim=1).values / caps.min(dim=1).values).max()) > 1.2


def test_undercoverage_penalty_pulls_gaussians_toward_uncovered_band():
    inside = _halfplane(H=32, W=32, edge=16)
    mc = _MaskConstraint.from_mask(inside, torch.device("cpu"), torch.float32, 3.0, 1.5,
                                   cap_mode="isotropic", undercoverage_band=3.0)
    assert mc.band_flat.numel() > 0
    # a small Gaussian whose support reaches the band but whose tail weight there is under tau
    # (the penalty acts through the same support tiles as the renderer, so a Gaussian with no
    # overlap gets no pull — the loss is support-limited by design)
    means = torch.tensor([[10.0, 16.0]], requires_grad=True)
    log_scales = torch.log(torch.tensor([[1.5, 1.5]])).requires_grad_(True)
    field = GaussianField(means, log_scales, torch.zeros(1, requires_grad=True),
                          torch.ones(1, 3, requires_grad=True))
    cfg = FitConfig(iters=1, sigma_cutoff=3.0)
    pen = mc.undercoverage(field, cfg, support_fade_alpha=0.0)
    value = float(pen.detach())
    pen.backward()
    assert value > 0.0
    # gradient descent (-grad) must move the mean toward the boundary band (+x)
    assert float(means.grad[0, 0]) < 0.0


def test_undercoverage_penalty_is_zero_when_band_is_covered():
    inside = _halfplane(H=32, W=32, edge=16)
    mc = _MaskConstraint.from_mask(inside, torch.device("cpu"), torch.float32, 3.0, 1.5,
                                   undercoverage_band=3.0)
    means = torch.tensor([[13.0, 16.0]])
    log_scales = torch.log(torch.tensor([[30.0, 30.0]]))
    field = GaussianField(means, log_scales, torch.zeros(1), torch.ones(1, 3))
    cfg = FitConfig(iters=1, sigma_cutoff=3.0)
    assert float(mc.undercoverage(field, cfg, support_fade_alpha=0.0)) == 0.0


def test_undercoverage_weight_improves_boundary_band_coverage():
    img, inside = _disk()
    target = torch.as_tensor(img)
    icfg = InitConfig(strategy="quadtree_wse", num_gaussians=120, seed=0)
    sdf = m.signed_distance(inside)
    band = (sdf > 1.5) & (sdf <= 4.5)

    def band_deficit(weight):
        fcfg = FitConfig(iters=50, mask_contain=True, support_fade=True, loss_weighting="mask",
                         mask_undercoverage_weight=weight, mask_undercoverage_band=3.0,
                         log_every=100)
        field = build_masked_field(img, inside, icfg, sigma_cutoff=fcfg.sigma_cutoff,
                                   mask_margin=fcfg.mask_margin)
        torch.manual_seed(0)
        out = fit(field, target, fcfg, mask=inside, verbose=False)
        render = out["render"].detach().cpu().numpy()
        uncovered = np.abs(render).sum(axis=2) == 0.0
        return float((uncovered & band).sum())

    assert band_deficit(2.0) <= band_deficit(0.0)


def test_boundary_tangent_add_spawns_contained_tangent_children():
    img, inside = _disk()
    target = torch.as_tensor(img)
    icfg = InitConfig(strategy="quadtree_wse", num_gaussians=100, seed=0)
    fcfg = FitConfig(iters=40, mask_contain=True, mask_cap_mode="anisotropic",
                     support_fade=True, loss_weighting="mask",
                     mask_boundary_add_every=10, mask_boundary_add_count=12,
                     mask_boundary_add_band=4.0, mask_boundary_add_spacing=4.0,
                     log_every=100)
    field = build_masked_field(img, inside, icfg, sigma_cutoff=fcfg.sigma_cutoff,
                               mask_margin=fcfg.mask_margin, cap_mode="anisotropic")
    n0 = field.n
    out = fit(field, target, fcfg, mask=inside, verbose=False)
    events = out["history"]["boundary_add_events"]
    assert events and sum(e["added"] for e in events) > 0
    assert out["n_gaussians"] > n0
    render = out["render"].detach().cpu().numpy()
    assert float(np.abs(render[~inside]).max()) == 0.0
    # all means still inside the eroded interior after spawning + containment
    # (the isotropic ellipse check does not apply: anisotropic caps intentionally exceed it,
    # and the exact-zero render above is the containment statement that matters)
    assert _means_inside(out["field"], inside, fcfg.sigma_cutoff, fcfg.mask_margin)


def test_boundary_tangent_add_unit_orientation_and_projection():
    from structsplat.fit import _boundary_tangent_add

    inside = _halfplane(H=40, W=40, edge=20)
    H = W = 40
    target = torch.zeros(H, W, 3)
    target[:, :20] = 0.8  # constant bright object on the inside
    render = torch.zeros(H, W, 3)  # nothing rendered yet: residual peaks in the band
    mc = _MaskConstraint.from_mask(inside, torch.device("cpu"), torch.float32, 3.0, 1.5)
    icfg = InitConfig(strategy="iso_blue_noise", num_gaussians=12, seed=0)
    img = target.numpy()
    field = build_masked_field(img, inside, icfg, contain=True)
    cfg = FitConfig(iters=1, mask_boundary_add_count=6, mask_boundary_add_band=4.0,
                    mask_boundary_add_spacing=4.0)
    n0 = field.n
    grown, added = _boundary_tangent_add(field, target, render, cfg, mc)
    assert added > 0 and grown.n == n0 + added
    new_theta = grown.rotations[n0:]
    # vertical edge -> tangent is vertical -> theta = +-pi/2
    assert torch.allclose(torch.abs(torch.sin(new_theta)), torch.ones_like(new_theta),
                          atol=1e-3)
    new_means = grown.means[n0:]
    eroded = m.erode(inside, 1.5 + 3.0 * m.MIN_SCALE)
    ix = np.clip(np.round(new_means[:, 0].numpy()).astype(int), 0, W - 1)
    iy = np.clip(np.round(new_means[:, 1].numpy()).astype(int), 0, H - 1)
    assert bool(eroded[iy, ix].all())
    # sx is the along-tangent axis and starts no shorter than the across axis (invariant 3)
    new_scales = torch.exp(grown.log_scales[n0:])
    assert bool((new_scales[:, 0] >= new_scales[:, 1] - 1e-6).all())


def test_boundary_add_requires_mask():
    target = torch.zeros(12, 12, 3)
    field = build_field(np.zeros((12, 12, 3), np.float32), InitConfig(num_gaussians=10, seed=0))
    with pytest.raises(ValueError):
        fit(field, target, FitConfig(iters=1, mask_boundary_add_every=5,
                                     mask_boundary_add_count=4), verbose=False)
    with pytest.raises(ValueError):
        fit(field, target, FitConfig(iters=1, mask_undercoverage_weight=1.0), verbose=False)


def test_config_validation_boundary_fields():
    with pytest.raises(ValueError):
        FitConfig(iters=1, mask_cap_mode="bogus")
    with pytest.raises(ValueError):
        FitConfig(iters=1, mask_cap_refresh_every=0)
    with pytest.raises(ValueError):
        FitConfig(iters=1, mask_undercoverage_weight=-0.1)
    with pytest.raises(ValueError):
        FitConfig(iters=1, mask_undercoverage_band=0.0)
    with pytest.raises(ValueError):
        FitConfig(iters=1, mask_undercoverage_tau=0.0)
    with pytest.raises(ValueError):
        FitConfig(iters=1, mask_boundary_add_every=0)
    with pytest.raises(ValueError):
        FitConfig(iters=1, mask_boundary_add_count=-1)
    with pytest.raises(ValueError):
        FitConfig(iters=1, mask_boundary_add_spacing=0.0)
