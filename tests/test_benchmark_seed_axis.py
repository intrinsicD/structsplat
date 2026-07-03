# ruff: noqa: E402
import argparse

import pytest

torch = pytest.importorskip("torch")

from benchmarks.common import add_seed_args, resolve_seeds
from benchmarks import coco_fit_compare as coco
from benchmarks import cross_repo_matrix_compare as cross
from benchmarks import optimization_followup as opt
from benchmarks import quadtree_init_compare as qt


def test_seed_args_keep_legacy_alias_and_add_axis():
    parser = argparse.ArgumentParser()
    add_seed_args(parser)

    assert resolve_seeds(**vars(parser.parse_args([]))) == [0]
    assert resolve_seeds(**vars(parser.parse_args(["--seed", "3"]))) == [3]
    assert resolve_seeds(**vars(parser.parse_args(["--seeds", "2", "4"]))) == [2, 4]
    assert resolve_seeds(**vars(parser.parse_args(["--seed", "3", "--seeds", "1", "2"]))) == [1, 2]


def test_coco_summary_reports_image_seed_variance(tmp_path):
    rows = [
        {"image": "toy", "method": "structsplat", "seed": 0, "psnr": 10.0,
         "ms_ssim": 0.90, "fit_seconds": 1.0},
        {"image": "toy", "method": "structsplat", "seed": 1, "psnr": 12.0,
         "ms_ssim": 0.94, "fit_seconds": 1.4},
    ]
    coco._write_summary(rows, [{"image": "toy"}], tmp_path, 16, 2, 32, [0, 1])

    text = (tmp_path / "summary.md").read_text()
    assert "Seeds: 0, 1" in text
    assert "PSNR Std" in text
    assert "1.000" in text


def test_cross_repo_summary_reports_image_seed_variance(tmp_path):
    rows = [
        _cross_row(seed=0, psnr=10.0, ms_ssim=0.90),
        _cross_row(seed=1, psnr=12.0, ms_ssim=0.94),
    ]
    selected = [{"image": "toy", "difficulty": "regular", "source_path": "/tmp/toy.jpg"}]
    cross._write_summary(rows, selected, tmp_path)

    text = (tmp_path / "summary.md").read_text()
    assert "Seeds: 0, 1" in text
    assert "PSNR Std" in text
    assert "1.0000" in text


def test_optimization_aggregate_counts_image_seed_runs():
    rows = [
        _opt_row("candidate_a", seed=0, image="a", psnr=10.0),
        _opt_row("candidate_a", seed=1, image="a", psnr=12.0),
        _opt_row("candidate_a", seed=0, image="b", psnr=14.0),
        _opt_row("candidate_a", seed=1, image="b", psnr=16.0),
    ]

    agg = opt._aggregate(rows)
    assert agg[0]["runs"] == 4
    assert agg[0]["mean_psnr"] == 13.0
    assert agg[0]["std_psnr"] > 0.0


def test_quadtree_aggregate_counts_image_seed_runs():
    rows = [
        _quadtree_row("candidate_a", seed=0, image="a", psnr=10.0),
        _quadtree_row("candidate_a", seed=1, image="a", psnr=12.0),
        _quadtree_row("candidate_a", seed=0, image="b", psnr=14.0),
        _quadtree_row("candidate_a", seed=1, image="b", psnr=16.0),
    ]

    agg = qt._aggregate(rows)
    assert agg[0]["runs"] == 4
    assert agg[0]["mean_psnr"] == 13.0
    assert agg[0]["std_psnr"] > 0.0


def _cross_row(seed: int, psnr: float, ms_ssim: float) -> dict:
    return {
        "image": "toy",
        "difficulty": "regular",
        "source_path": "/tmp/toy.jpg",
        "max_side": 32,
        "iters": 2,
        "budget_cap": 16,
        "seed": seed,
        "method": "gaussianimage",
        "method_label": cross.METHOD_LABELS["gaussianimage"],
        "psnr": psnr,
        "ssim": 0.8,
        "ms_ssim": ms_ssim,
        "lpips": None,
        "auc_psnr": psnr - 1.0,
        "mae": 0.1,
        "edge_mae": 0.2,
        "fit_seconds": 1.0,
        "total_seconds": 1.2,
        "status": "ok",
        "error": "",
    }


def _opt_row(candidate: str, seed: int, image: str, psnr: float) -> dict:
    return {
        "candidate": candidate,
        "seed": seed,
        "image": image,
        "psnr": psnr,
        "ms_ssim": 0.9,
        "auc_psnr": psnr - 1.0,
        "fit_seconds": 1.0,
        "total_seconds": 1.2,
        "iterations_run": 2,
        "n_gaussians": 16,
        "stopped_early": False,
    }


def _quadtree_row(candidate: str, seed: int, image: str, psnr: float) -> dict:
    return {
        "candidate": candidate,
        "seed": seed,
        "image": image,
        "init_psnr": psnr - 1.0,
        "psnr": psnr,
        "ms_ssim": 0.9,
        "auc_psnr": psnr - 0.5,
        "final_max_scale": 2.0,
        "final_p95_scale": 1.5,
        "init_seconds": 0.1,
        "fit_seconds": 1.0,
        "total_seconds": 1.1,
    }
