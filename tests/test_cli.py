import json

import numpy as np
import pytest

from structsplat.cli import _benchmark_symbol, load_image, main, save_image


def test_best_default_conversion_is_not_a_structsplat_subcommand(
    monkeypatch, capsys
):
    """ADR-0028 leaves scripts/convert.py as the only best-default conversion CLI."""
    monkeypatch.setattr("sys.argv", ["structsplat", "convert"])

    with pytest.raises(SystemExit):
        main()

    assert "invalid choice: 'convert'" in capsys.readouterr().err


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
            "image-to-gaussians2d",
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


def test_fit_cli_accepts_progressive_wse_order(tmp_path, monkeypatch, capsys):
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
            "iso_blue_noise",
            "--wse-progressive-order",
            "--num-gaussians",
            "8",
            "--iters",
            "1",
            "--chunk",
            "8",
            "--outdir",
            str(outdir),
            "--device",
            "cpu",
        ],
    )

    main()

    assert (outdir / "toy_iso_blue_noise.npz").exists()
    assert "8 gaussians" in capsys.readouterr().out


def test_gaussians2d_to_image_outputs_reconstruction_error_metrics_and_overlay(
    tmp_path, monkeypatch, capsys
):
    pytest.importorskip("torch")
    from structsplat.gaussians import GaussianField

    height, width = 16, 18
    field = GaussianField.from_numpy(
        means=np.asarray([[5.0, 7.0], [12.0, 8.0]], dtype=np.float32),
        scales=np.asarray([[6.0, 5.0], [6.0, 5.0]], dtype=np.float32),
        angles=np.asarray([0.0, 0.3], dtype=np.float32),
        colors=np.asarray([[0.8, 0.2, 0.1], [0.1, 0.4, 0.9]], dtype=np.float32),
    )
    field_path = tmp_path / "field.npz"
    field.save(str(field_path))
    reference = np.full((height, width, 3), 0.25, dtype=np.float32)
    reference_path = tmp_path / "reference.png"
    save_image(str(reference_path), reference)
    reconstruction_path = tmp_path / "reconstruction.png"
    error_path = tmp_path / "absolute_error.png"
    raw_error_path = tmp_path / "signed_error.npy"
    metrics_path = tmp_path / "metrics.json"
    overlay_path = tmp_path / "gaussians.png"
    monkeypatch.setattr(
        "sys.argv",
        [
            "structsplat",
            "gaussians2d-to-image",
            str(field_path),
            "--reference",
            str(reference_path),
            "--out",
            str(reconstruction_path),
            "--error-out",
            str(error_path),
            "--raw-error-out",
            str(raw_error_path),
            "--metrics-out",
            str(metrics_path),
            "--gaussians-out",
            str(overlay_path),
            "--normalization-eps",
            "1e-12",
            "--device",
            "cpu",
        ],
    )

    main()

    for path in (reconstruction_path, error_path, raw_error_path, metrics_path, overlay_path):
        assert path.is_file()
    assert load_image(str(reconstruction_path)).shape == (height, width, 3)
    assert np.load(raw_error_path).shape == (height, width, 3)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metrics["format"] == "GaussianField NPZ"
    assert metrics["n_gaussians"] == 2
    assert metrics["width"] == width
    assert metrics["height"] == height
    assert metrics["normalization_eps"] == 1e-12
    assert metrics["mse"] >= 0.0
    assert np.isfinite(metrics["psnr_db"])
    assert "rendered 2 Gaussians" in capsys.readouterr().out


def test_render_npz_without_reference_requires_canvas_dimensions(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    from structsplat.gaussians import GaussianField

    field = GaussianField.from_numpy(
        means=np.asarray([[2.0, 2.0]], dtype=np.float32),
        scales=np.asarray([[2.0, 2.0]], dtype=np.float32),
        angles=np.asarray([0.0], dtype=np.float32),
        colors=np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32),
    )
    field_path = tmp_path / "field.npz"
    field.save(str(field_path))
    monkeypatch.setattr(
        "sys.argv",
        ["structsplat", "render", str(field_path), "--out", str(tmp_path / "out.png")],
    )

    with pytest.raises(SystemExit, match="requires --reference"):
        main()


def test_render_self_describing_sspl1_without_reference(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    from structsplat import codec
    from structsplat.gaussians import GaussianField

    field = GaussianField.from_numpy(
        means=np.asarray([[3.0, 3.0]], dtype=np.float32),
        scales=np.asarray([[2.0, 2.0]], dtype=np.float32),
        angles=np.asarray([0.0], dtype=np.float32),
        colors=np.asarray([[0.25, 0.5, 0.75]], dtype=np.float32),
    )
    stream_path = tmp_path / "field.sspl1"
    stream_path.write_bytes(codec.encode(field, 8, 9))
    output_path = tmp_path / "decoded.png"
    monkeypatch.setattr(
        "sys.argv",
        [
            "structsplat",
            "render",
            str(stream_path),
            "--out",
            str(output_path),
            "--device",
            "cpu",
        ],
    )

    main()

    assert load_image(str(output_path)).shape == (8, 9, 3)


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
            "responsibility",
            "--responsibility-mass-alpha",
            "0.65",
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
    assert captured["refine_sites"] == ["none", "responsibility"]
    assert captured["responsibility_mass_alpha"] == pytest.approx(0.65)
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
