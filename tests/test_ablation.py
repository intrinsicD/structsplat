# ruff: noqa: E402
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from PIL import Image

from benchmarks.ablation import run_ablation, fitness, _config_key


def _write_toy(path):
    img = np.zeros((16, 16, 3), np.float32)
    img[:, 8:, :] = 1.0
    img[3:6, 3:6] = [1.0, 0.5, 0.0]
    Image.fromarray((img * 255).astype(np.uint8)).save(path)


def test_ablation_sweeps_flanking_and_writes_outputs(tmp_path):
    img_path = tmp_path / "toy.png"
    outdir = tmp_path / "ablation"
    _write_toy(img_path)

    rows = run_ablation(
        [str(img_path)],
        budgets=[16],
        strategies=["aniso_flanking"],
        seeds=[0],
        iters=2,
        target_psnr=None,
        target_psnrs=[0.0],
        flank_offsets=[0.0, 0.5],
        flat_fracs=[0.02],
        corner_fracs=[0.15],
        max_axis_ratios=[4.0],
        coherence_powers=[1.0],
        render_chunk=8,
        outdir=str(outdir),
        device="cpu",
        write_plots=False,
    )

    assert len(rows) == 2
    assert {r["flank_offset_frac"] for r in rows} == {0.0, 0.5}
    assert all("0.0" in r["iters_to_targets"] for r in rows)
    assert (outdir / "ablation.json").exists()
    assert (outdir / "ablation.csv").exists()
    assert (outdir / "summary.md").exists()
    # BENCH-002: run reproducible from its own artifacts (resolved config + versions logged)
    import json
    cfg = json.loads((outdir / "config.json").read_text())
    assert cfg["device"] == "cpu" and cfg["versions"]["torch"] is not None
    assert cfg["resolved"]["flank_offsets"] == [0.0, 0.5]
    assert cfg["resolved"]["renderer"] == "normalized"
    assert {r["renderer"] for r in rows} == {"normalized"}


def test_ablation_control_arms_thread_sampler_and_relocation(tmp_path):
    img_path = tmp_path / "toy.png"
    outdir = tmp_path / "controls"
    _write_toy(img_path)

    rows = run_ablation(
        [str(img_path)],
        budgets=[16],
        strategies=["floyd_steinberg", "density_random", "random_relocate"],
        seeds=[0],
        iters=2,
        target_psnr=None,
        target_psnrs=[],
        flank_offsets=[0.5],
        render_chunk=8,
        relocate_every=1,
        relocate_count=4,
        outdir=str(outdir),
        device="cpu",
        write_plots=False,
    )

    by_strategy = {r["strategy"]: r for r in rows}
    assert by_strategy["floyd_steinberg"]["init_strategy"] == "aniso_flanking"
    assert by_strategy["floyd_steinberg"]["sampling_mode"] == "floyd_steinberg"
    assert by_strategy["density_random"]["init_strategy"] == "iso_blue_noise"
    assert by_strategy["density_random"]["sampling_mode"] == "density_random"
    assert by_strategy["random_relocate"]["init_strategy"] == "random"
    assert by_strategy["random_relocate"]["fit_control"] == "relocate"
    assert by_strategy["random_relocate"]["relocate_every"] == 1
    assert by_strategy["random_relocate"]["relocate_count"] == 4


def test_ablation_max_side_and_resume_jsonl(tmp_path):
    img_path = tmp_path / "toy.png"
    outdir = tmp_path / "resume"
    _write_toy(img_path)

    kwargs = dict(
        images=[str(img_path)],
        budgets=[16],
        strategies=["density_random"],
        seeds=[0],
        iters=2,
        target_psnr=None,
        target_psnrs=[],
        render_chunk=8,
        max_side=12,
        outdir=str(outdir),
        device="cpu",
        write_plots=False,
    )
    rows = run_ablation(**kwargs)
    resumed = run_ablation(**kwargs, resume=True)
    additive = run_ablation(**kwargs, resume=True, renderer="additive")

    assert len(rows) == len(resumed) == 1
    assert len(additive) == 2
    assert rows[0]["height"] == rows[0]["width"] == 12
    assert rows[0]["max_side"] == 12
    assert sum(1 for _ in (outdir / "ablation.jsonl").open()) == 2
    assert {r["renderer"] for r in additive} == {"normalized", "additive"}
    import json
    cfg = json.loads((outdir / "config.json").read_text())
    assert cfg["resolved"]["max_side"] == 12
    assert cfg["resolved"]["renderer"] == "additive"


def test_ablation_max_new_cells_advances_resumed_chunks(tmp_path):
    img_path = tmp_path / "toy.png"
    outdir = tmp_path / "chunks"
    _write_toy(img_path)

    kwargs = dict(
        images=[str(img_path)],
        budgets=[16],
        strategies=["density_random", "random_relocate"],
        seeds=[0],
        iters=2,
        target_psnr=None,
        target_psnrs=[],
        render_chunk=8,
        outdir=str(outdir),
        device="cpu",
        write_plots=False,
        max_new_cells=1,
    )
    first = run_ablation(**kwargs)
    second = run_ablation(**kwargs, resume=True)

    assert len(first) == 1
    assert len(second) == 2
    assert {r["strategy"] for r in second} == {"density_random", "random_relocate"}
    assert sum(1 for _ in (outdir / "ablation.jsonl").open()) == 2


def test_fitness_uses_best_config_not_pooled_mean():
    # two configs of one strategy at one budget: a good one and a bad one. fitness must return
    # the best config's mean, not the pooled mean that a wide hyperparameter sweep would dilute.
    rows = [
        {"strategy": "aniso_flanking", "budget": 16, "flank_offset_frac": 0.0, "flat_frac": 0.02,
         "corner_frac": 0.15, "max_axis_ratio": 4.0, "coherence_power": 1.0, "psnr": 30.0},
        {"strategy": "aniso_flanking", "budget": 16, "flank_offset_frac": 0.5, "flat_frac": 0.02,
         "corner_frac": 0.15, "max_axis_ratio": 4.0, "coherence_power": 1.0, "psnr": 10.0},
    ]
    assert len({_config_key(r) for r in rows}) == 2
    assert fitness(rows, "aniso_flanking", 16) == 30.0   # best config, not the pooled mean (20)
