# ruff: noqa: E402
from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

torch = pytest.importorskip("torch")

from benchmarks import coco_fit_compare as coco
from benchmarks import common
from benchmarks import cross_repo_matrix_compare as cross
from benchmarks import optimization_followup as opt
from benchmarks import quadtree_init_compare as qt
from benchmarks import rate_distortion as rd
from benchmarks import stage_search


def _write_toy(path, size=8):
    img = np.zeros((size, size, 3), np.float32)
    img[:, size // 2:, :] = 1.0
    img[1:3, 1:3] = [1.0, 0.5, 0.0]
    Image.fromarray((img * 255).astype(np.uint8)).save(path)


def test_common_image_auc_and_row_writers(tmp_path):
    img_path = tmp_path / "toy.png"
    _write_toy(img_path, size=4)

    img = common.load_image(img_path, max_side=2)
    assert img.shape[:2] == (2, 2)

    sampled = common.bilinear_sample(img, np.array([[0.0, 0.0], [0.5, 0.5]], dtype=np.float32))
    assert sampled.shape == (2, 3)
    assert np.allclose(sampled[0], img[0, 0])
    assert common.psnr_auc({"iter": [0, 2], "psnr": [10.0, 14.0]}) == pytest.approx(12.0)

    rows = [{"b": 2, "a": 1}]
    common.write_rows(tmp_path / "rows", rows)
    assert json.loads((tmp_path / "rows.json").read_text()) == rows
    assert (tmp_path / "rows.csv").read_text().splitlines()[0] == "b,a"


def test_rate_distortion_run_rd_tiny_smoke(tmp_path):
    img_path = tmp_path / "toy.png"
    outdir = tmp_path / "rd"
    _write_toy(img_path, size=16)

    rows = rd.run_rd(
        [str(img_path)],
        budgets=(8,),
        strategy="random",
        seeds=(0,),
        iters=1,
        bit_mixes=((8, 8, 8, 8),),
        qat_iters=0,
        render_chunk=8,
        outdir=str(outdir),
        device="cpu",
    )

    assert len(rows) == 1
    assert (outdir / "rate_distortion.json").exists()
    assert (outdir / "rate_distortion.csv").exists()
    assert (outdir / "rate_distortion.md").exists()


def test_coco_summary_keeps_zero_ok_method_cells(tmp_path):
    rows = [
        {
            "image": "toy",
            "method": "structsplat",
            "seed": 0,
            "psnr": 10.0,
            "ms_ssim": 0.9,
            "fit_seconds": 1.0,
        }
    ]

    coco._write_summary(rows, [{"image": "toy"}], tmp_path, 8, 1, 8, [0])

    text = (tmp_path / "summary.md").read_text()
    assert "| Instant-GI qt | 0 | - | - | - | - | - | - |" in text
    assert "| toy | 10.00 (0.00) | - | - | - | - |" in text


def test_cross_repo_summary_keeps_zero_ok_method_cells_and_errors(tmp_path):
    rows = [
        _cross_row("structsplat_current", "StructSplat current", status="ok"),
        {
            **_cross_row("instant_gi_quadtree", "Instant-GI qt", status="error"),
            "error": "RuntimeError: missing checkpoint",
        },
    ]
    selected = [{"image": "toy", "difficulty": "regular", "source_path": "/tmp/toy.jpg"}]

    cross._write_summary(rows, selected, tmp_path)

    text = (tmp_path / "summary.md").read_text()
    assert "| GaussianImage | 0 | - | - | - | - | - | - | - | - | - | - | - |" in text
    assert "## Errors" in text
    assert "missing checkpoint" in text


def test_cross_repo_grid_writes_with_missing_method_rows(tmp_path):
    target = tmp_path / "target.png"
    recon = tmp_path / "recon.png"
    _write_toy(target, size=16)
    _write_toy(recon, size=16)
    selected = [{
        "image": "toy",
        "difficulty": "regular",
        "source_path": str(target),
        "selected_16": str(target),
    }]
    rows = [
        {
            **_cross_row("structsplat_current", "StructSplat current", status="ok"),
            "max_side": 16,
            "iters": 2,
            "seed": 0,
            "reconstruction_path": str(recon),
            "n_gaussians": 8,
        },
        {
            **_cross_row("instant_gi_quadtree", "Instant-GI qt", status="error"),
            "max_side": 16,
            "iters": 2,
            "seed": 0,
            "error": "RuntimeError: missing external dependency",
        },
    ]

    cross._make_grid(rows, selected, tmp_path, 16, 2, 0, include_seed_in_name=False)

    assert (tmp_path / "grids" / "comparison_16px_2it.png").exists()


def test_cross_repo_shipped_defaults_builds_without_searched_growth(tmp_path):
    image = tmp_path / "toy.png"
    _write_toy(image, size=16)
    img = common.load_image(image)
    base_fit = cross._base_fit(iters=1, renderer="normalized", lpips=False)

    field, fcfg, _init_seconds, start_budget = cross._build_method(
        "structsplat_shipped_defaults",
        img,
        image,
        final_budget=16,
        seed=0,
        device="cpu",
        base_fit=base_fit,
    )

    assert field.n == 16
    assert start_budget == 16
    assert fcfg.iters == 1
    assert fcfg.renderer == "normalized"
    assert fcfg.pixel_loss == "l1"
    assert fcfg.split_every is None
    assert fcfg.max_gaussians is None


def test_stage_search_writer_preserves_summarize_bytes(tmp_path):
    row = {
        **{k: "baseline" for k in stage_search.STAGE_KEYS},
        "strategy": "aniso_flanking",
        "budget": 8,
        "seed": 0,
        "image": "toy",
        "config_label": "baseline",
        "status": "ok",
        "psnr": 10.0,
        "ms_ssim": 0.9,
        "auc_psnr": 9.5,
        "iters_to_target": None,
        "fit_seconds": 1.0,
        "init_seconds": 0.1,
        "history": {"iter": [0], "psnr": [10.0]},
        "prefix_metrics": None,
    }

    expected = stage_search.summarize([row])
    stage_search._write([row], tmp_path)

    assert (tmp_path / "summary.md").read_text() == expected
    assert (tmp_path / "stage_search.json").exists()
    assert (tmp_path / "stage_search.csv").exists()


def test_followup_summary_writers_accept_stub_rows(tmp_path):
    image = tmp_path / "toy.png"
    _write_toy(image)

    opt_args = SimpleNamespace(
        budget=8,
        iters=1,
        seeds=[0],
        max_side=8,
        render_chunk=8,
        device="cpu",
    )
    opt_out = tmp_path / "opt"
    opt_out.mkdir()
    opt.write_summary(
        opt_out,
        {"validation": [_opt_row("candidate_a")]},
        [image],
        opt_args,
    )
    assert "candidate_a" in ((opt_out / "summary.md").read_text())

    qt_args = SimpleNamespace(budget=8, iters=1, seeds=[0], max_side=8, device="cpu")
    qt_out = tmp_path / "qt"
    qt_out.mkdir()
    qt.write_summary(qt_out, [_qt_row("candidate_a")], [image], qt_args)
    assert "candidate_a" in ((qt_out / "summary.md").read_text())


def _cross_row(method: str, label: str, status: str) -> dict:
    row = {
        "image": "toy",
        "difficulty": "regular",
        "source_path": "/tmp/toy.jpg",
        "max_side": 8,
        "iters": 1,
        "budget_cap": 8,
        "seed": 0,
        "method": method,
        "method_label": label,
        "method_note": "stub",
        "status": status,
        "error": "",
    }
    if status == "ok":
        row.update({
            "psnr": 10.0,
            "ssim": 0.8,
            "ms_ssim": 0.9,
            "lpips": None,
            "auc_psnr": 9.5,
            "mae": 0.1,
            "edge_mae": 0.2,
            "fit_seconds": 1.0,
            "total_seconds": 1.1,
        })
    return row


def _opt_row(label: str) -> dict:
    return {
        "image": "toy",
        "seed": 0,
        "candidate": label,
        "label": label,
        "config_label": label,
        "psnr": 10.0,
        "ms_ssim": 0.9,
        "auc_psnr": 9.5,
        "fit_seconds": 1.0,
        "total_seconds": 1.1,
        "iterations_run": 1,
        "n_gaussians": 8,
        "stopped_early": False,
    }


def _qt_row(candidate: str) -> dict:
    return {
        "image": "toy",
        "seed": 0,
        "candidate": candidate,
        "init_psnr": 8.0,
        "psnr": 10.0,
        "ms_ssim": 0.9,
        "auc_psnr": 9.5,
        "final_max_scale": 2.0,
        "final_p95_scale": 1.5,
        "init_seconds": 0.1,
        "fit_seconds": 1.0,
        "total_seconds": 1.1,
    }
