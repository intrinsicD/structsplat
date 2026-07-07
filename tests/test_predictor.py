# ruff: noqa: E402
import json
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from PIL import Image

from benchmarks.feedforward_eval import evaluate_feedforward_predictor
from benchmarks.feedforward_teacher_export import export_teacher_fields
from benchmarks.feedforward_train import train_feedforward_predictor
from structsplat import init as _init
from structsplat.config import InitConfig
from structsplat.gaussians import GaussianField
from structsplat.predictor import load_learned_checkpoint, predictor_input_tensor


def _toy(H=24, W=24):
    img = np.zeros((H, W, 3), np.float32)
    img[:, W // 2:] = [0.9, 0.8, 0.2]
    img[4:10, 4:12] = [0.1, 0.3, 0.8]
    return img


def _write_toy(path):
    img = _toy()
    Image.fromarray(np.rint(img * 255).astype(np.uint8)).save(path)


def test_predictor_input_tensor_supports_tensor_prior_channels():
    img = _toy()
    image_only = predictor_input_tensor(img, 16, input_mode="image", device="cpu")
    with_tensor = predictor_input_tensor(img, 16, input_mode="image_tensor", device="cpu")

    assert image_only.shape == (1, 3, 16, 16)
    assert with_tensor.shape == (1, 7, 16, 16)
    assert torch.isfinite(with_tensor).all()


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


def test_tiny_learned_predictor_trains_and_loads(tmp_path):
    img_path = tmp_path / "toy.png"
    teacher_dir = tmp_path / "teachers"
    train_dir = tmp_path / "train"
    _write_toy(img_path)

    export_teacher_fields(
        [str(img_path)],
        outdir=str(teacher_dir),
        budget=8,
        strategy="grid",
        seed=0,
        iters=1,
        max_side=None,
        render_chunk=8,
        device="cpu",
    )
    result = train_feedforward_predictor(
        teacher_dir / "teacher_manifest.json",
        outdir=train_dir,
        image_size=16,
        hidden=8,
        epochs=2,
        lr=1e-3,
        seed=0,
        device="cpu",
    )

    ckpt = train_dir / "predictor.pt"
    assert result["checkpoint_path"] == str(ckpt)
    assert ckpt.exists()
    loaded = load_learned_checkpoint(ckpt, device="cpu")
    assert loaded["input_mode"] == "image"
    assert loaded["in_channels"] == 3
    assert (train_dir / "loss_history.csv").exists()
    assert (train_dir / "summary.md").exists()

    img = _toy()
    field = _init.build_field(
        img,
        InitConfig(
            strategy="feedforward",
            num_gaussians=8,
            predictor_checkpoint=str(ckpt),
            predictor_fallback_strategy="grid",
        ),
    )
    assert field.n == 8
    assert torch.isfinite(field.means).all()
    assert torch.isfinite(field.log_scales).all()
    assert torch.isfinite(field.colors).all()
    assert torch.isfinite(field.opacities).all()
    assert float(field.means[:, 0].min()) >= 0.0
    assert float(field.means[:, 0].max()) <= img.shape[1] - 1
    assert float(field.means[:, 1].min()) >= 0.0
    assert float(field.means[:, 1].max()) <= img.shape[0] - 1

    trunc = _init.build_field(
        img,
        InitConfig(strategy="feedforward", num_gaussians=4, predictor_checkpoint=str(ckpt)),
    )
    assert trunc.n == 4

    padded = _init.build_field(
        img,
        InitConfig(
            strategy="feedforward",
            num_gaussians=10,
            predictor_checkpoint=str(ckpt),
            predictor_fallback_strategy="grid",
        ),
    )
    assert padded.n == 10
    assert torch.isfinite(padded.colors).all()


def test_feedforward_eval_compares_equal_n_methods(tmp_path):
    img_path = tmp_path / "toy.png"
    teacher_dir = tmp_path / "teachers"
    train_dir = tmp_path / "train"
    train_tensor_dir = tmp_path / "train_tensor"
    eval_dir = tmp_path / "eval"
    _write_toy(img_path)

    export_teacher_fields(
        [str(img_path)],
        outdir=str(teacher_dir),
        budget=8,
        strategy="grid",
        seed=0,
        iters=1,
        max_side=None,
        render_chunk=8,
        device="cpu",
    )
    train = train_feedforward_predictor(
        teacher_dir / "teacher_manifest.json",
        outdir=train_dir,
        image_size=16,
        hidden=8,
        epochs=2,
        lr=1e-3,
        seed=0,
        device="cpu",
    )
    train_tensor = train_feedforward_predictor(
        teacher_dir / "teacher_manifest.json",
        outdir=train_tensor_dir,
        image_size=16,
        hidden=8,
        input_mode="image_tensor",
        epochs=2,
        lr=1e-3,
        seed=0,
        device="cpu",
    )
    rows = evaluate_feedforward_predictor(
        [str(img_path)],
        checkpoint=train["checkpoint_path"],
        tensor_checkpoint=train_tensor["checkpoint_path"],
        outdir=eval_dir,
        budget=8,
        iters=1,
        max_side=None,
        render_chunk=8,
        seed=0,
        device="cpu",
        prior_strategy="grid",
        scratch_strategy="random",
        methods=("learned", "learned_tensor", "tensor_prior", "scratch"),
    )

    assert len(rows) == 4
    assert {r["method"] for r in rows} == {
        "learned", "learned_tensor", "tensor_prior", "scratch",
    }
    assert all(r["budget"] == 8 for r in rows)
    assert all(r["n_gaussians"] == 8 for r in rows)
    assert all(r["iterations_run"] == 1 for r in rows)
    assert all(r["total_seconds"] >= 0.0 for r in rows)
    assert (eval_dir / "feedforward_eval.json").exists()
    assert (eval_dir / "feedforward_eval.csv").exists()
    summary = (eval_dir / "summary.md").read_text(encoding="utf-8")
    assert "learned" in summary
    assert "tensor_prior" in summary
    assert "scratch" in summary
