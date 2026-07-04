import csv
import json

from benchmarks.abl004_confirmation import (
    ConfirmationSpec,
    bootstrap_mean_ci,
    expected_cells,
    write_confirmation_analysis,
    write_plan,
)


def _write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _row(image, budget, seed, strategy, psnr, ms_ssim):
    return {
        "image": image,
        "source_path": f"{image}.png",
        "height": 16,
        "width": 16,
        "max_side": 16,
        "strategy": strategy,
        "init_strategy": strategy,
        "sampling_mode": "wse",
        "fit_control": "none",
        "relocate_every": None,
        "relocate_count": None,
        "iters": 2,
        "target_psnr": 35.0,
        "target_psnrs": [35.0],
        "render_chunk": 8,
        "renderer": "normalized",
        "pixel_loss": "l1",
        "ssim_weight": 0.3,
        "compute_lpips": False,
        "budget": budget,
        "seed": seed,
        "flank_offset_frac": 0.0 if strategy == "aniso_onedge" else 0.5,
        "flat_frac": 0.02,
        "corner_frac": 0.15,
        "max_axis_ratio": 6.0,
        "coherence_power": 1.0,
        "psnr": psnr,
        "ssim": 0.9,
        "ms_ssim": ms_ssim,
        "lpips": None,
        "iters_to_target": None,
        "iters_to_targets": {"35.0": None},
        "n_gaussians": budget,
        "init_seconds": 0.1,
        "fit_seconds": 0.2,
    }


def _read_csv(path):
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_confirmation_plan_writes_expected_cells(tmp_path):
    spec = ConfirmationSpec(
        images=("a.png", "b.png"),
        budgets=(16, 32),
        seeds=(0, 1),
        strategies=("aniso_onedge", "aniso_flanking"),
        max_side=16,
        iters=2,
        renderer="normalized",
    )

    cells = write_plan(spec, tmp_path)

    assert len(cells) == 16
    assert len(expected_cells(spec)) == 16
    assert (tmp_path / "confirmation_plan.csv").exists()
    cfg = json.loads((tmp_path / "confirmation_config.json").read_text())
    assert cfg["resolved"]["expected_cells"] == 16
    assert cfg["resolved"]["strategies"] == ["aniso_onedge", "aniso_flanking"]


def test_confirmation_analysis_reports_pairs_missing_and_ranks(tmp_path):
    spec = ConfirmationSpec(
        images=("a.png", "b.png"),
        budgets=(16,),
        seeds=(0, 1),
        strategies=("aniso_onedge", "aniso_flanking", "floyd_steinberg"),
        max_side=16,
        iters=2,
        renderer="normalized",
        baseline="aniso_onedge",
    )
    rows = [
        _row("a", 16, 0, "aniso_onedge", 30.0, 0.91),
        _row("a", 16, 0, "aniso_flanking", 29.0, 0.90),
        _row("a", 16, 0, "floyd_steinberg", 28.0, 0.89),
        _row("a", 16, 1, "aniso_onedge", 31.0, 0.92),
        _row("a", 16, 1, "aniso_flanking", 31.5, 0.925),
        _row("a", 16, 1, "floyd_steinberg", 29.0, 0.90),
        _row("b", 16, 0, "aniso_onedge", 20.0, 0.80),
        _row("b", 16, 0, "aniso_flanking", 19.5, 0.79),
        _row("b", 16, 0, "floyd_steinberg", 18.0, 0.78),
        # Leave b/seed1/floyd_steinberg missing to exercise partial accounting.
        _row("b", 16, 1, "aniso_onedge", 21.0, 0.81),
        _row("b", 16, 1, "aniso_flanking", 20.0, 0.80),
    ]
    _write_jsonl(tmp_path / "ablation.jsonl", rows)

    summary = write_confirmation_analysis(
        tmp_path,
        spec,
        bootstrap_samples=50,
        bootstrap_seed=0,
    )

    assert summary["expected_cells"] == 12
    assert summary["completed_cells"] == 11
    assert summary["missing_cells"] == 1
    missing = _read_csv(tmp_path / "missing_cells.csv")
    assert missing[0]["image"] == "b"
    assert missing[0]["seed"] == "1"
    assert missing[0]["strategy"] == "floyd_steinberg"

    leaderboard = _read_csv(tmp_path / "leaderboard.csv")
    onedge = [r for r in leaderboard if r["strategy"] == "aniso_onedge"][0]
    assert onedge["runs"] == "4"
    assert onedge["rank_psnr"] == "1"

    paired = _read_csv(tmp_path / "paired_deltas_vs_baseline.csv")
    flank = [r for r in paired if r["left"] == "aniso_flanking"][0]
    assert flank["paired_units"] == "4"
    assert float(flank["mean_delta_psnr"]) == -0.5
    assert flank["wins_psnr"] == "1"
    assert flank["losses_psnr"] == "3"

    pairwise = _read_csv(tmp_path / "pairwise_deltas.csv")
    assert {r["left"] for r in pairwise} == {"aniso_onedge", "aniso_flanking"}
    ranks = _read_csv(tmp_path / "rank_stability.csv")
    assert [r for r in ranks if r["strategy"] == "aniso_onedge"][0]["wins"] == "2"
    assert (tmp_path / "confirmation_analysis.md").exists()


def test_bootstrap_mean_ci_handles_empty_and_singleton():
    assert bootstrap_mean_ci([], samples=10) == (None, None, None)
    assert bootstrap_mean_ci([2.0], samples=10) == (2.0, 2.0, 2.0)
