# ruff: noqa: E402
import json
import os

import numpy as np
import pytest

pytest.importorskip("torch")

from structsplat.batch import expand_images, json_safe_args, unique_bases
from structsplat.cli import main, save_image


def test_expand_images_handles_dirs_globs_and_dedup(tmp_path):
    d = tmp_path / "imgs"
    d.mkdir()
    for name in ("b.png", "a.jpg", "notes.txt"):
        (d / name).write_bytes(b"x")
    loose = tmp_path / "c.png"
    loose.write_bytes(b"x")

    got = expand_images([str(d), str(loose), str(tmp_path / "*.png"), str(loose)])
    assert got == sorted([str(d / "a.jpg"), str(d / "b.png"), str(loose)])


def test_unique_bases_disambiguates_same_named_files():
    paths = ["/x/kodim01.png", "/y/kodim01.png", "/z/other.png"]
    assert unique_bases(paths) == ["kodim01", "kodim01_1", "other"]


def test_json_safe_args_keeps_scalars_and_drops_func():
    import argparse

    ns = argparse.Namespace(seed=3, strategy="quadtree_wse", func=print,
                            level_fractions=[0.1, 0.9], mask=None)
    safe = json_safe_args(ns)
    assert safe == {"level_fractions": [0.1, 0.9], "mask": None, "seed": 3,
                    "strategy": "quadtree_wse"}


def _write_toy_images(root, names):
    paths = []
    for k, name in enumerate(names):
        img = np.zeros((12, 12, 3), np.float32)
        img[:, 6:] = (k + 1) / (len(names) + 1)
        path = root / name
        save_image(str(path), img)
        paths.append(path)
    return paths


def _run_cli(monkeypatch, argv):
    monkeypatch.setattr("sys.argv", ["structsplat", *argv])
    main()


def test_batch_fit_two_workers_resume_and_metrics(tmp_path, monkeypatch, capsys):
    imgs = tmp_path / "imgs"
    imgs.mkdir()
    _write_toy_images(imgs, ["one.png", "two.png"])
    outdir = tmp_path / "runs"
    argv = [
        "batch-fit", str(imgs),
        "--num-gaussians", "8", "--iters", "1",
        "--workers", "2", "--devices", "cpu", "--torch-threads", "1",
        "--outdir", str(outdir),
    ]

    _run_cli(monkeypatch, argv)
    metrics = outdir / "metrics.jsonl"
    rows = [json.loads(line) for line in metrics.read_text().splitlines()]
    assert len(rows) == 2
    assert {row["base"] for row in rows} == {"one", "two"}
    for row in rows:
        assert row["device"] == "cpu"
        assert np.isfinite(row["psnr"])
        assert row["config"]["num_gaussians"] == 8
        assert os.path.exists(row["out_npz"])
        assert os.path.exists(row["out_png"])

    # Resume: nothing new is fitted and no rows are appended.
    _run_cli(monkeypatch, argv)
    assert "2 skipped (resume)" in capsys.readouterr().out
    assert len(metrics.read_text().splitlines()) == 2


def test_batch_fit_isolates_per_image_failures(tmp_path, monkeypatch):
    imgs = tmp_path / "imgs"
    imgs.mkdir()
    _write_toy_images(imgs, ["good.png"])
    (imgs / "broken.png").write_bytes(b"this is not an image")
    outdir = tmp_path / "runs"

    with pytest.raises(SystemExit) as excinfo:
        _run_cli(monkeypatch, [
            "batch-fit", str(imgs),
            "--num-gaussians", "8", "--iters", "1",
            "--workers", "1", "--devices", "cpu", "--torch-threads", "1",
            "--outdir", str(outdir),
        ])
    assert excinfo.value.code == 1
    rows = [json.loads(line) for line in (outdir / "metrics.jsonl").read_text().splitlines()]
    by_base = {row["base"]: row for row in rows}
    assert "error" in by_base["broken"]
    assert np.isfinite(by_base["good"]["psnr"])


def test_batch_fit_rejects_mask_options(tmp_path, monkeypatch):
    imgs = tmp_path / "imgs"
    imgs.mkdir()
    _write_toy_images(imgs, ["one.png"])
    with pytest.raises(SystemExit, match="mask"):
        _run_cli(monkeypatch, [
            "batch-fit", str(imgs), "--mask-contain",
            "--outdir", str(tmp_path / "runs"),
        ])
