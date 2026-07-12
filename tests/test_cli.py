import numpy as np
import pytest

from structsplat.cli import _benchmark_symbol, load_image, main, save_image


def test_save_image_rounds_not_truncates(tmp_path):
    # astype(uint8) floored every pixel down by up to 1/255; np.rint round-trips exact 8-bit
    # values and halves the quantization bias (COMP-002).
    pytest.importorskip("PIL")
    vals = (np.arange(256) / 255.0).astype(np.float32)
    img = np.repeat(vals[None, :, None], 3, axis=2).reshape(1, 256, 3)
    p = tmp_path / "grad.png"
    save_image(str(p), img)
    back = load_image(str(p))
    # every exact k/255 level must survive the round-trip; truncation shifted them down
    assert np.abs(back - img).max() <= 1e-6


def test_benchmark_import_helper_adds_repo_root(monkeypatch):
    import sys
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[1]
    cleaned = [
        p for p in sys.path
        if p and Path(p).resolve() != repo_root
    ]
    monkeypatch.setattr(sys, "path", cleaned)
    for name in list(sys.modules):
        if name == "benchmarks" or name.startswith("benchmarks."):
            monkeypatch.delitem(sys.modules, name, raising=False)

    run_stage_search = _benchmark_symbol("benchmarks.stage_search", "run_stage_search")

    assert callable(run_stage_search)
    assert str(repo_root) in sys.path


def test_fit_cli_accepts_feedforward_short_refinement(tmp_path, monkeypatch, capsys):
    pytest.importorskip("torch")
    img = np.zeros((12, 12, 3), np.float32)
    img[:, 6:] = 1.0
    path = tmp_path / "toy.png"
    save_image(str(path), img)
    outdir = tmp_path / "runs"
    monkeypatch.setattr(
        "sys.argv",
        [
            "structsplat",
            "fit",
            str(path),
            "--strategy",
            "feedforward",
            "--predictor-fallback-strategy",
            "grid",
            "--num-gaussians",
            "8",
            "--iters",
            "1",
            "--loss-target-downsample",
            "2",
            "--loss-target-full-frac",
            "1.0",
            "--chunk",
            "8",
            "--outdir",
            str(outdir),
            "--device",
            "cpu",
        ],
    )

    main()

    assert (outdir / "toy_feedforward.npz").exists()
    assert "8 gaussians" in capsys.readouterr().out


def test_fit_cli_accepts_qat_rate_flags(tmp_path, monkeypatch, capsys):
    pytest.importorskip("torch")
    img = np.zeros((12, 12, 3), np.float32)
    img[:, 6:] = 1.0
    path = tmp_path / "toy.png"
    save_image(str(path), img)
    outdir = tmp_path / "runs"
    monkeypatch.setattr(
        "sys.argv",
        [
            "structsplat",
            "fit",
            str(path),
            "--strategy",
            "random",
            "--num-gaussians",
            "8",
            "--iters",
            "1",
            "--chunk",
            "8",
            "--render-checkpoint",
            "--qat-mode",
            "ste",
            "--lambda-rate",
            "0.01",
            "--qat-bits-colors",
            "5",
            "--outdir",
            str(outdir),
            "--device",
            "cpu",
        ],
    )

    main()

    assert (outdir / "toy_random.npz").exists()
    assert "8 gaussians" in capsys.readouterr().out


def test_stage_search_cli_forwards_sharding_and_factored_axes(monkeypatch):
    import benchmarks.stage_search as stage_search

    captured = {}

    def fake_run_stage_search(images, **kwargs):
        captured["images"] = images
        captured.update(kwargs)
        return []

    monkeypatch.setattr(stage_search, "run_stage_search", fake_run_stage_search)
    monkeypatch.setattr(
        "sys.argv",
        [
            "structsplat",
            "stage-search",
            "images",
            "--mode",
            "influence",
            "--refine-sites",
            "none",
            "residual_tensor",
            "--refine-primitives",
            "duplicate",
            "moment_preserving",
            "--refine-nms-modes",
            "off",
            "on",
            "--refine-color-inits",
            "target",
            "residual",
            "--refine-score-modes",
            "legacy_abs",
            "gaussian_abs",
            "signed_gaussian",
            "--refine-prune-modes",
            "off",
            "on",
            "--refine-relocate-modes",
            "off",
            "on",
            "--early-exit",
            "--early-exit-window",
            "25",
            "--early-exit-min-delta",
            "0.125",
            "--early-exit-min-iters",
            "50",
            "--resume",
            "--max-new-cells",
            "7",
        ],
    )

    main()

    assert captured["images"] == ["images"]
    assert captured["mode"] == "influence"
    assert captured["refine_sites"] == ["none", "residual_tensor"]
    assert captured["refine_primitives"] == ["duplicate", "moment_preserving"]
    assert captured["refine_nms_modes"] == ["off", "on"]
    assert captured["refine_color_inits"] == ["target", "residual"]
    assert captured["refine_score_modes"] == [
        "legacy_abs", "gaussian_abs", "signed_gaussian",
    ]
    assert captured["refine_prune_modes"] == ["off", "on"]
    assert captured["refine_relocate_modes"] == ["off", "on"]
    assert captured["early_exit"] is True
    assert captured["early_exit_window"] == 25
    assert captured["early_exit_min_delta"] == 0.125
    assert captured["early_exit_min_iters"] == 50
    assert captured["resume"] is True
    assert captured["max_new_cells"] == 7
