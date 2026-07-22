"""FIT-021 pooled lifecycle, byte-budgeted capacity, and error-triage events (ADR-0020)."""
import numpy as np
import pytest
import torch

from structsplat import codec
from structsplat.config import FitConfig
from structsplat.fit import fit, _MaskConstraint
from structsplat.gaussians import GaussianField
from structsplat.pool import (
    POOL_PARK_COORD,
    derive_capacity,
    gaussian_row_bytes,
    prepare_pooled_field,
)
from structsplat.render import render
from structsplat import triage as triage_mod


def _test_image(H=48, W=48):
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    img = np.stack([
        0.2 + 0.6 * (xx / W),
        0.3 + 0.4 * (yy / H),
        0.5 + 0.3 * np.sin(xx / 6.0) * np.cos(yy / 6.0),
    ], axis=2).clip(0.0, 1.0).astype(np.float32)
    mask = (((yy - H / 2) ** 2 + (xx - W / 2) ** 2) <= (0.42 * H) ** 2).astype(np.float32)
    return img * mask[..., None], mask


def _random_field(n, H, W, img, seed=0, opacities=True):
    rng = np.random.default_rng(seed)
    pos = rng.uniform(6, min(H, W) - 6, size=(n, 2)).astype(np.float32)
    colors = img[pos[:, 1].astype(int), pos[:, 0].astype(int)]
    return GaussianField.from_numpy(
        pos, np.full((n, 2), 2.0, np.float32), np.zeros(n, np.float32), colors,
        opacities=np.full(n, 2.0, np.float32) if opacities else None,
    )


def _pooled_cfg(**kw):
    base = dict(
        iters=60, log_every=30, triage_every=20,
        triage_spawn_count=16, triage_split_count=4, triage_merge_count=4,
        triage_park_count=4, pool_capacity=200,
        ssim_weight=0.0, render_chunk=64,
    )
    base.update(kw)
    return FitConfig(**base)


def test_gaussian_row_bytes_matches_sspl1_defaults():
    # ADR-0007 defaults: means 2x16 bit, scales 2x8, rot 8, colors 3x8, opacity 8.
    assert gaussian_row_bytes(has_opacity=True) == 11
    assert gaussian_row_bytes(has_opacity=False) == 10


def test_derive_capacity_breakdown_and_budget():
    img, mask = _test_image()
    cfg = FitConfig(triage_every=10, target_file_bytes=6000, pool_header_allowance=1024)
    capacity, breakdown = derive_capacity(cfg, has_opacity=True, mask_arr=mask)
    assert capacity == breakdown["capacity"] > 0
    spent = (breakdown["header_allowance"] + breakdown["alpha_payload_bytes"]
             + breakdown["stream_frame_bytes"] + capacity * breakdown["gaussian_row_bytes"])
    assert spent <= 6000
    # one more row would exceed the budget
    assert spent + breakdown["gaussian_row_bytes"] > 6000


def test_parked_rows_render_exact_zero_and_no_gradient():
    H = W = 32
    img, _ = _test_image(H, W)
    field = _random_field(24, H, W, img)
    field, pool = prepare_pooled_field(field, _pooled_cfg(pool_capacity=40), H, W)
    assert field.n == 40 and pool.live_count == 24
    field.trainable()

    def _render(f, rows=None):
        means = f.means if rows is None else f.means[rows]
        conics = f.conics(0.0) if rows is None else f.conics(0.0)[rows]
        colors = f.colors if rows is None else f.colors[rows]
        radii = f.radii(3.0) if rows is None else f.radii(3.0)[rows]
        opac = f.opacity_values()
        if rows is not None:
            opac = opac[rows]
        return render(means, conics, colors, radii, H, W, 64, opacities=opac)

    full = _render(field)
    live_only = _render(field, pool.live)
    assert torch.equal(full, live_only)

    loss = _render(field).square().mean()
    loss.backward()
    parked = ~pool.live
    assert field.means.grad is not None
    assert float(field.means.grad[parked].abs().max()) == 0.0
    assert float(field.colors.grad[parked].abs().max()) == 0.0


def test_prepare_creates_opacities_and_rejects_small_capacity():
    H = W = 32
    img, _ = _test_image(H, W)
    field = _random_field(24, H, W, img, opacities=False)
    field, pool = prepare_pooled_field(field, _pooled_cfg(pool_capacity=30), H, W)
    assert field.opacities is not None and field.opacities.shape[0] == 30
    with pytest.raises(ValueError, match="smaller than the initial field"):
        prepare_pooled_field(_random_field(24, H, W, img), _pooled_cfg(pool_capacity=8), H, W)


def test_pooled_config_validation():
    with pytest.raises(ValueError, match="replaces the independent topology timers"):
        FitConfig(triage_every=10, pool_capacity=100, split_every=50, split_count=8)
    with pytest.raises(ValueError, match="needs a capacity"):
        FitConfig(triage_every=10)
    with pytest.raises(ValueError, match="require pooled triage mode"):
        FitConfig(target_file_bytes=1000)
    with pytest.raises(ValueError, match="color_basis='constant'"):
        FitConfig(triage_every=10, pool_capacity=100, color_basis="affine")
    with pytest.raises(ValueError, match="post-fit QAT"):
        FitConfig(triage_every=10, pool_capacity=100, qat_mode="ste")
    with pytest.raises(ValueError, match="normalized renderer semantics"):
        FitConfig(triage_every=10, pool_capacity=100, renderer="additive")
    with pytest.raises(ValueError, match="checkpoint_policy='terminal'"):
        FitConfig(triage_every=10, pool_capacity=100,
                  checkpoint_policy="best_psnr_final_count")


def test_alpha_stream_roundtrip_and_backward_compat():
    H = W = 32
    img, mask = _test_image(H, W)
    field = _random_field(20, H, W, img)
    plain = codec.encode(field, H, W)
    with_alpha = codec.encode(field, H, W, alpha=mask)
    assert codec.decode_alpha(plain) is None
    alpha = codec.decode_alpha(with_alpha)
    assert alpha is not None and alpha.shape == (H, W)
    assert np.abs(alpha - mask).max() < 1e-6
    # soft alpha survives up to 8-bit quantization
    soft = (mask * 0.7 + 0.1).astype(np.float32)
    soft_alpha = codec.decode_alpha(codec.encode(field, H, W, alpha=soft))
    assert np.abs(soft_alpha - soft).max() <= (0.5 / 255.0) + 1e-6
    # the Gaussian payload decodes identically whether or not alpha rides along
    a = codec.decode(plain)
    b = codec.decode(with_alpha)
    for name in ("means", "log_scales", "rotations", "colors", "opacities"):
        va, vb = getattr(a, name), getattr(b, name)
        assert torch.equal(va, vb)
    # byte accounting still exact with the extra stream
    comp = codec.blob_components(with_alpha)
    assert "alpha" in comp["streams"]
    total = comp["header_bytes"] + sum(
        s["framed_bytes"] for s in comp["streams"].values()
    )
    assert total == len(with_alpha)


def test_envelope_merge_contains_both_inputs_and_parks_partner():
    H = W = 32
    img, _ = _test_image(H, W)
    means = np.array([[14.0, 16.0], [17.0, 16.0], [8.0, 8.0], [24.0, 24.0],
                      [8.0, 24.0], [24.0, 8.0]], np.float32)
    colors = np.full((6, 3), 0.5, np.float32)
    field = GaussianField.from_numpy(
        means, np.full((6, 2), 2.0, np.float32), np.zeros(6, np.float32), colors,
        opacities=np.full(6, 2.0, np.float32),
    )
    cfg = _pooled_cfg(pool_capacity=8, triage_merge_count=1)
    field, pool = prepare_pooled_field(field, cfg, H, W)
    touched, stats = triage_mod._merge_pairs(field, pool, cfg, None)
    assert stats["merged"] == 1
    assert pool.live_count == 5  # 6 live at entry, one partner parked into the free list
    keep = touched[0].item()
    part = touched[1].item()
    assert not bool(pool.live[part])
    assert float(field.means[part, 0]) == POOL_PARK_COORD
    # merged covariance envelope contains both translated inputs (generalized-eigen ratio <= 1)
    merged_cov = triage_mod._row_covariances(field, torch.tensor([keep]))[0]
    mid = torch.tensor([15.5, 16.0])
    for orig_mean in (torch.tensor([14.0, 16.0]), torch.tensor([17.0, 16.0])):
        d = orig_mean - mid
        spatial = torch.diag(torch.tensor([4.0, 4.0])) + torch.outer(d, d)
        vals, vecs = torch.linalg.eigh(merged_cov)
        inv_sqrt = vecs @ torch.diag(vals.clamp_min(1e-8).rsqrt()) @ vecs.T
        ratio = torch.linalg.eigvalsh(inv_sqrt @ spatial @ inv_sqrt)[-1]
        assert float(ratio) <= 1.0 + 1e-4


def test_park_gate_spares_sole_owner():
    H = W = 32
    img, _ = _test_image(H, W)
    # two nearly co-located rows (each ~0.5 responsibility everywhere), one isolated sole
    # owner whose max responsibility is ~1 despite its tiny summed activity
    means = np.array([[8.0, 8.0], [8.2, 8.0], [24.0, 24.0]], np.float32)
    field = GaussianField.from_numpy(
        means, np.full((3, 2), 2.0, np.float32), np.zeros(3, np.float32),
        np.full((3, 3), 0.5, np.float32), opacities=np.full(3, 2.0, np.float32),
    )
    cfg = _pooled_cfg(pool_capacity=4, triage_park_count=3, prune_keep_min=1,
                      triage_park_max_responsibility=0.6)
    field, pool = prepare_pooled_field(field, cfg, H, W)
    target = torch.as_tensor(img)
    render_img = torch.zeros_like(target)
    stats = triage_mod.attribution_pass(field, target, render_img, cfg)
    parked = triage_mod._park_donors(field, pool, stats, cfg)
    assert 2 not in parked.tolist()  # the sole owner survives regardless of its low activity


def test_pooled_fit_end_to_end_budget_and_determinism():
    H = W = 48
    img, mask = _test_image(H, W)
    target = torch.as_tensor(img)
    cfg = FitConfig(
        iters=60, log_every=30, triage_every=20,
        triage_spawn_count=16, triage_split_count=4, triage_merge_count=4,
        triage_park_count=4, target_file_bytes=6000, pool_header_allowance=1024,
        mask_contain=True, support_fade=True, loss_weighting="mask",
        ssim_weight=0.0, render_chunk=64,
    )

    def run():
        torch.manual_seed(0)
        field = _random_field(40, H, W, img)
        return fit(field, target, cfg, verbose=False, mask=mask)

    # Bitwise reproducibility is asserted under single-thread CPU execution (the BENCH-012
    # determinism convention): the synthetic target has exact symmetry ties, and dynamic
    # MKL/OMP thread teams under load can reorder float reductions enough to flip them.
    n_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        out = run()
        out2 = run()
    finally:
        torch.set_num_threads(n_threads)
    pool_info = out["pool"]
    assert pool_info is not None
    assert pool_info["live_n"] == out["n_gaussians"] == out["field"].n
    assert pool_info["live_n"] <= pool_info["capacity"]
    assert pool_info["budget"]["target_file_bytes"] == 6000
    assert out["history"]["triage_events"], "triage events must fire"
    for event in out["history"]["triage_events"]:
        assert event["live"] <= event["capacity"]
    # the trajectory history reports live rows, never the constant capacity
    assert max(out["history"]["n_gaussians"]) <= pool_info["capacity"]
    # grew beyond the start and improved
    assert pool_info["live_n"] > pool_info["initial_live"]
    assert out["psnr"] > 10.0

    blob = codec.encode(out["field"], H, W, fcfg=cfg, alpha=mask)
    assert len(blob) <= cfg.target_file_bytes
    assert codec.decode(blob).n == out["n_gaussians"]

    # exact-zero outside the mask (CORE-010 guarantee preserved by the pooled path)
    outside = torch.as_tensor(1.0 - mask)[..., None]
    assert float((out["render"] * outside).abs().max()) == 0.0

    # bitwise reproducibility of the full pooled path
    for name in ("means", "log_scales", "rotations", "colors", "opacities"):
        va = getattr(out["field"], name)
        vb = getattr(out2["field"], name)
        assert torch.equal(va, vb), f"{name} not reproducible"
    assert out["history"]["triage_events"] == out2["history"]["triage_events"]


def test_pooled_fit_optimizer_never_rebuilt(monkeypatch):
    H = W = 32
    img, _ = _test_image(H, W)
    target = torch.as_tensor(img)
    import structsplat.fit as fitmod

    def _boom(*args, **kwargs):  # pragma: no cover - failure path
        raise AssertionError("pooled mode must not rebuild the optimizer")

    monkeypatch.setattr(fitmod, "_carry_adam_state", _boom)
    torch.manual_seed(0)
    field = _random_field(24, H, W, img)
    cfg = _pooled_cfg(iters=45, triage_every=15, pool_capacity=64)
    out = fit(field, target, cfg, verbose=False)
    assert out["history"]["triage_events"]
    assert out["pool"]["capacity"] == 64


def test_mask_apply_exempt_keeps_parked_rows_off_image():
    H = W = 32
    img, mask = _test_image(H, W)
    field = _random_field(12, H, W, img)
    cfg = _pooled_cfg(pool_capacity=20, mask_contain=True, support_fade=True)
    field, pool = prepare_pooled_field(field, cfg, H, W, mask_arr=mask)
    mc = _MaskConstraint.from_mask(mask, field.means.device, field.means.dtype,
                                   cfg.sigma_cutoff, cfg.mask_margin)
    mc.apply(field, cfg, exempt=~pool.live)
    parked = (~pool.live).nonzero().reshape(-1)
    assert float(field.means[parked].max()) == POOL_PARK_COORD
    # live rows were projected inside
    live_rows = pool.live.nonzero().reshape(-1)
    assert float(field.means[live_rows, 0].min()) >= 0.0
