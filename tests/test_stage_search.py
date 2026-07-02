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
        scale_cap_modes=["feature12"],
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


def test_stage_search_dedupes_equivalent_configs(tmp_path):
    img_path = tmp_path / "toy.png"
    _write_toy(img_path)
    # strategy=random ignores the tensor stage entirely: 2 tensors x 2 strategies -> 3 runs
    rows = run_stage_search(
        [str(img_path)], budgets=[16], seeds=[0], iters=2, max_side=None,
        strategies=["aniso_flanking", "random"],
        tensor_operators=["central", "scharr"],
        density_modes=["structure"], sampling_modes=["density_random"],
        color_modes=["bilinear"], scale_modes=["spacing"], scale_cap_modes=["none"],
        opacity_modes=["none"],
        renderers=["normalized"], pixel_losses=["l1"], optimizers=["adam"],
        lr_schedules=["none"], refine_modes=["none"], pyramid_modes=["single"],
        render_chunk=8, outdir=str(tmp_path / "dedupe"), device="cpu",
    )
    assert len(rows) == 3
    assert sum(r["strategy"] == "random" for r in rows) == 1


def test_stage_influence_writes_paired_deltas(tmp_path):
    img_path = tmp_path / "toy.png"
    outdir = tmp_path / "influence"
    _write_toy(img_path)

    rows = run_stage_search(
        [str(img_path)], budgets=[16], seeds=[0, 1], iters=4, max_side=None,
        mode="influence",
        strategies=["aniso_flanking"],
        tensor_operators=["central"], tensor_colors=["luma", "rgb"],
        density_modes=["structure"], sampling_modes=["wse", "density_random"],
        orientation_modes=["tensor"], color_modes=["bilinear"], scale_modes=["spacing"],
        scale_cap_modes=["feature12"], opacity_modes=["none"], renderers=["normalized"],
        pixel_losses=["l1"],
        optimizers=["adam"], lr_schedules=["none"], refine_modes=["none"],
        pyramid_modes=["single"], target_psnr=5.0,
        render_chunk=8, outdir=str(outdir), device="cpu",
    )
    # baseline + tensor_color=rgb + sampling=density_random, x 2 seeds
    assert len(rows) == 6
    assert sum(r["is_baseline"] for r in rows) == 2
    assert all("auc_psnr" in r and "iters_to_target" in r for r in rows)
    assert (outdir / "influence.md").exists()
    text = (outdir / "influence.md").read_text()
    assert "tensor_color=rgb" in text and "sampling=density_random" in text
    assert "ΔPSNR" in text
