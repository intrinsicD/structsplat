# ruff: noqa: E402
import json
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from PIL import Image

from benchmarks.feedforward_teacher_export import export_teacher_fields
from structsplat import init as _init
from structsplat.config import InitConfig
from structsplat.gaussians import GaussianField


def _toy(H=24, W=24):
    img = np.zeros((H, W, 3), np.float32)
    img[:, W // 2:] = [0.9, 0.8, 0.2]
    img[4:10, 4:12] = [0.1, 0.3, 0.8]
    return img


def _write_toy(path):
    img = _toy()
    Image.fromarray(np.rint(img * 255).astype(np.uint8)).save(path)


def test_feedforward_fallback_matches_tensor_prior_strategy():
    img = _toy()
    ff = _init.build_field(
        img,
        InitConfig(
            strategy="feedforward",
            predictor_fallback_strategy="aniso_flanking",
            num_gaussians=16,
            seed=4,
            sampling_mode="density_random",
        ),
    )
    prior = _init.build_field(
        img,
        InitConfig(
            strategy="aniso_flanking",
            num_gaussians=16,
            seed=4,
            sampling_mode="density_random",
        ),
    )

    assert ff.n == 16
    assert torch.equal(ff.means, prior.means)
    assert torch.equal(ff.colors, prior.colors)


def test_feedforward_checkpoint_truncates_and_pads(tmp_path):
    img = _toy()
    teacher = _init.build_field(
        img, InitConfig(strategy="grid", num_gaussians=6, seed=0)
    )
    ckpt = tmp_path / "teacher.npz"
    teacher.save(str(ckpt))

    trunc = _init.build_field(
        img,
        InitConfig(strategy="feedforward", num_gaussians=4, predictor_checkpoint=str(ckpt)),
    )
    assert trunc.n == 4
    assert torch.equal(trunc.means, teacher.means[:4])

    padded = _init.build_field(
        img,
        InitConfig(
            strategy="feedforward",
            num_gaussians=8,
            predictor_checkpoint=str(ckpt),
            predictor_fallback_strategy="grid",
        ),
    )
    assert padded.n == 8
    assert torch.equal(padded.means[:6], teacher.means)
    assert torch.isfinite(padded.colors).all()


def test_feedforward_rejects_recursive_fallback():
    with pytest.raises(ValueError, match="predictor_fallback_strategy"):
        InitConfig(strategy="feedforward", predictor_fallback_strategy="feedforward")


def test_teacher_export_writes_manifest_and_fields(tmp_path):
    img_path = tmp_path / "toy.png"
    outdir = tmp_path / "teachers"
    _write_toy(img_path)

    rows = export_teacher_fields(
        [str(img_path)],
        outdir=str(outdir),
        budget=8,
        strategy="grid",
        seed=0,
        iters=1,
        max_side=None,
        render_chunk=8,
        device="cpu",
    )

    assert len(rows) == 1
    manifest = json.loads((outdir / "teacher_manifest.json").read_text(encoding="utf-8"))
    assert manifest["rows"][0]["budget"] == 8
    field_path = outdir / "fields" / "toy_n8_seed0.npz"
    assert field_path.exists()
    field = GaussianField.load(str(field_path))
    assert field.n == 8
    assert (outdir / "config.json").exists()
    assert (outdir / "summary.md").exists()
