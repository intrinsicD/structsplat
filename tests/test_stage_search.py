# ruff: noqa: E402
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from PIL import Image

from benchmarks.stage_search import run_stage_search


def _write_toy(path):
    img = np.zeros((16, 16, 3), np.float32)
    img[:, 8:, :] = 1.0
    img[3:6, 3:6] = [1.0, 0.5, 0.0]
    Image.fromarray((img * 255).astype(np.uint8)).save(path)


def test_stage_search_writes_ranked_outputs(tmp_path):
    img_path = tmp_path / "toy.png"
    outdir = tmp_path / "stage_search"
    _write_toy(img_path)

    rows = run_stage_search(
        [str(img_path)],
        budgets=[16],
        seeds=[0],
        iters=2,
        max_side=None,
        strategies=["aniso_flanking"],
        tensor_operators=["central", "scharr"],
        density_modes=["structure"],
        sampling_modes=["density_random"],
        color_modes=["bilinear"],
        scale_modes=["spacing"],
        opacity_modes=["none", "constant"],
        renderers=["normalized"],
        pixel_losses=["l1"],
        optimizers=["adam"],
        lr_schedules=["none"],
        refine_modes=["none"],
        pyramid_modes=["single"],
        render_chunk=8,
        outdir=str(outdir),
        device="cpu",
    )

    assert len(rows) == 4
    assert {r["tensor"] for r in rows} == {"central", "scharr"}
    assert {r["opacity"] for r in rows} == {"none", "constant"}
    assert (outdir / "stage_search.json").exists()
    assert (outdir / "stage_search.csv").exists()
    assert (outdir / "summary.md").exists()
