import numpy as np
import pytest

from structsplat.cli import load_image, main, save_image


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
