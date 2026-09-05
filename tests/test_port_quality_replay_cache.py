"""Procedural CPU checks for PORT checker geometry reuse, not method timing evidence."""
from dataclasses import replace
import gc
from types import SimpleNamespace
import weakref

import numpy as np
import pytest
import torch

from benchmarks.hier_research_report import _port_quality_replayer
from benchmarks.port007_controls import ellipse_mask
from structsplat.config import FitConfig
from structsplat.fit import _MaskConstraint
import structsplat.safe_schedule as safe


def _fixture(kind="full"):
    mask = np.ones((19, 23), dtype=bool) if kind == "full" else ellipse_mask(19, 23)
    cfg = FitConfig(renderer="normalized", sigma_cutoff=3., mask_margin=.75,
                    aa_dilation=.1, mask_cap_mode="isotropic", mask_undercoverage_band=1.5)
    rgb = np.full((*mask.shape, 3), .2, dtype=np.float32)
    den = np.full(mask.shape, .2, dtype=np.float32)
    target = np.zeros_like(rgb)
    return rgb, den, target, mask, cfg, 7


def _uncached(rgb, den, target, mask, cfg, count):
    constraint = _MaskConstraint.from_mask(
        mask, "cpu", torch.float32, cfg.sigma_cutoff, cfg.mask_margin,
        aa_dilation=cfg.aa_dilation, cap_mode=cfg.mask_cap_mode,
        undercoverage_band=cfg.mask_undercoverage_band)
    return safe._quality_from_render(
        torch.from_numpy(rgb), torch.from_numpy(target), torch.from_numpy(den),
        torch.from_numpy(mask), constraint, .05, count).to_dict()


def _spy_construction(monkeypatch):
    original, built = _MaskConstraint.from_mask, []
    def construct(*args, **kwargs):
        result = original(*args, **kwargs)
        built.append(result)
        return result
    monkeypatch.setattr(_MaskConstraint, "from_mask", construct)
    return built


def test_identical_geometry_reused_across_array_copies_and_storage_layouts(monkeypatch):
    args = _fixture("ellipse")
    built = _spy_construction(monkeypatch)
    original, used = safe._quality_from_render, []
    def quality(*a, **k):
        used.append(a[4])
        return original(*a, **k)
    monkeypatch.setattr(safe, "_quality_from_render", quality)
    replay = _port_quality_replayer()
    expected = replay(*args)
    for mask in (args[3].copy(), np.asfortranarray(args[3])):
        changed = (*args[:3], mask, *args[4:])
        assert replay(*changed) == expected
    assert len(built) == 1 and len(used) == 3
    assert all(constraint is built[0] for constraint in used)


@pytest.mark.parametrize("change", ["shape", "dtype", "bytes", "sigma_cutoff", "mask_margin",
                                     "aa_dilation", "mask_cap_mode", "mask_undercoverage_band"])
def test_every_mask_and_constructor_config_component_invalidates(monkeypatch, change):
    args = list(_fixture())
    built = _spy_construction(monkeypatch)
    # This key-only test includes uint8 masks, which are not the quality evaluator's bool
    # indexing interface. Keep real geometry construction but do not invoke raw quality here.
    monkeypatch.setattr(safe, "_quality_from_render",
                        lambda *a, **k: SimpleNamespace(to_dict=lambda: {"key_fixture": True}))
    replay = _port_quality_replayer()
    replay(*args)
    changed = list(args)
    if change == "shape":
        changed[3] = args[3].reshape(23, 19)
        changed[0] = args[0].reshape(23, 19, 3)
        changed[1] = args[1].reshape(23, 19)
        changed[2] = args[2].reshape(23, 19, 3)
        assert changed[3].tobytes() == args[3].tobytes()
    elif change == "dtype":
        changed[3] = args[3].astype(np.uint8)
        assert changed[3].tobytes() == args[3].tobytes()
    elif change == "bytes":
        changed[3] = args[3].copy()
        changed[3][9, 11] = False
    else:
        value = {"sigma_cutoff": 2.5, "mask_margin": 1., "aa_dilation": .2,
                 "mask_cap_mode": "anisotropic", "mask_undercoverage_band": 2.}[change]
        changed[4] = replace(args[4], **{change: value})
    replay(*changed)
    replay(*args)
    assert len(built) == 2 and built[0] is not built[1]


@pytest.mark.parametrize("kind", ["full", "ellipse"])
def test_cached_raw_quality_is_exact_uncached_quality_for_changing_inputs(kind, monkeypatch):
    args = list(_fixture(kind))
    expected = _uncached(*args)
    changed_rgb = list(args)
    changed_rgb[0] = np.full_like(args[0], .3)
    changed_den = list(args)
    changed_den[1] = np.zeros_like(args[1])
    changed_target = list(args)
    changed_target[2] = np.full_like(args[2], .15)
    changed_count = list(args)
    changed_count[5] = 11
    cases = [args, changed_rgb, changed_den, changed_target, changed_count]
    reference = [_uncached(*case) for case in cases]
    built = _spy_construction(monkeypatch)
    replay = _port_quality_replayer()
    actual = [replay(*case) for case in cases]
    assert actual == reference and actual[0] == expected
    assert actual[1]["foreground_mse"] != expected["foreground_mse"]
    assert actual[2]["interior_hole_fraction"] != expected["interior_hole_fraction"]
    if kind == "ellipse":
        assert actual[2]["boundary_hole_fraction"] != expected["boundary_hole_fraction"]
    else:
        # A full mask has no finite boundary band; its preserved convention is zero.
        assert actual[2]["boundary_hole_fraction"] == expected["boundary_hole_fraction"] == 0.
    assert actual[3]["foreground_mse"] != expected["foreground_mse"]
    assert actual[4]["n_gaussians"] == 11 and expected["n_gaussians"] == 7
    assert len(built) == 1
    # The returned scalar dict is not shared or cached either.
    actual[0]["foreground_mse"] = -123.
    assert replay(*args) == expected


@pytest.mark.parametrize("bad", ["rgb", "den"])
def test_nonfinite_new_arrays_never_reuse_prior_finite_result(monkeypatch, bad):
    args = list(_fixture())
    built = _spy_construction(monkeypatch)
    replay = _port_quality_replayer()
    assert replay(*args)["finite"]
    args[0 if bad == "rgb" else 1].flat[0] = float("nan")
    assert replay(*args)["finite"] is False
    assert len(built) == 1


def test_caller_mask_mutation_cannot_corrupt_a_cached_geometry(monkeypatch):
    args = list(_fixture("ellipse"))
    original_mask = args[3].copy()
    expected = _uncached(*args)
    built = _spy_construction(monkeypatch)
    replay = _port_quality_replayer()
    assert replay(*args) == expected
    args[3][9, 11] = False
    replay(*args)
    np.testing.assert_array_equal(built[0].inside.numpy(), original_mask)
    args[3][:] = original_mask
    assert replay(*args) == expected
    assert len(built) == 2


def test_separate_validation_replayers_never_share_geometry(monkeypatch):
    args = _fixture()
    built = _spy_construction(monkeypatch)
    first, second = _port_quality_replayer(), _port_quality_replayer()
    assert first(*args) == second(*args) == first(*args)
    assert len(built) == 2 and built[0] is not built[1]
    assert built[0].inside.data_ptr() != built[1].inside.data_ptr()


def test_replayer_retains_no_input_arrays_or_config_objects():
    args = _fixture("ellipse")
    references = [weakref.ref(value) for value in args[:-1]]
    replay = _port_quality_replayer()
    replay(*args)
    del args
    gc.collect()
    assert all(reference() is None for reference in references)


@pytest.mark.parametrize("index", [0, 1, 3])
def test_shape_guard_still_fails_before_geometry_construction(monkeypatch, index):
    args = list(_fixture())
    built = _spy_construction(monkeypatch)
    args[index] = args[index][:-1]
    with pytest.raises(ValueError, match="quality array shape mismatch"):
        _port_quality_replayer()(*args)
    assert not built
