# ruff: noqa: E402
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from PIL import Image

from benchmarks.stage_search import (
    INFLUENCE_DEFAULTS,
    LEGACY_REFINE_MODES,
    run_stage_search,
    _background_kwargs,
    _canonicalize,
    _color_solve_kwargs,
    _legacy_refine_config,
    _loss_weight_kwargs,
    _refine_kwargs,
    _refine_kwargs_from_config,
    _row_temper_kwargs,
    _scale_cap_kwargs,
    _state_seed_kwargs,
    _support_fade_kwargs,
    summarize,
)


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
    assert all(r["loss_weighting"] == "none" for r in rows)
    assert all(r["edge_mae"] >= 0.0 for r in rows)
    assert (outdir / "stage_search.json").exists()
    assert (outdir / "stage_search.csv").exists()
    assert (outdir / "summary.md").exists()
    index = (outdir / "index.html").read_text(encoding="utf-8")
    assert "stage_search.csv" in index
    assert "summary.md" in index
    assert "Per-Stage Marginals" in index


def test_stage_search_summary_uses_headline_target_columns():
    base = {
        "status": "ok",
        "image": "toy",
        "budget": 16,
        "seed": 0,
        "psnr": 30.0,
        "ms_ssim": 0.9,
        "auc_psnr": 29.0,
        "fit_seconds": 0.1,
        "iters_to_target": None,
        "iters_to_targets": {"28.0": 2, "30.0": 5, "32.0": None, "35.0": None},
    }
    base.update({
        "strategy": "aniso_flanking",
        "tensor": "central",
        "tensor_color": "luma",
        "density": "structure",
        "sampling": "wse",
        "orientation": "tensor",
        "color": "bilinear",
        "scale": "spacing",
        "scale_cap": "none",
        "opacity": "none",
        "renderer": "normalized",
        "aa": 0.0,
        "color_basis": "constant",
        "color_solve": "none",
        "loss": "l1",
        "optimizer": "adam",
        "lr_schedule": "none",
        "refine": "none",
        "pyramid": "single",
    })
    base["config_label"] = "baseline"

    text = summarize([base])

    assert "Hit 28" in text
    assert "Iter 30" in text
    assert "Iters→target" not in text
    assert "Hit 35" not in text


def test_stage_search_early_exit_is_opt_in_and_records_nominal_auc(tmp_path):
    img_path = tmp_path / "toy.png"
    _write_toy(img_path)
    rows = run_stage_search(
        [str(img_path)],
        budgets=[16],
        seeds=[0],
        iters=20,
        max_side=None,
        strategies=["aniso_flanking"],
        tensor_operators=["central"],
        density_modes=["structure"],
        sampling_modes=["density_random"],
        color_modes=["bilinear"],
        scale_modes=["spacing"],
        scale_cap_modes=["none"],
        opacity_modes=["none"],
        renderers=["normalized"],
        pixel_losses=["l1"],
        optimizers=["adam"],
        lr_schedules=["none"],
        refine_modes=["none"],
        pyramid_modes=["single"],
        early_exit=True,
        early_exit_window=1,
        early_exit_min_delta=1e6,
        log_every=1,
        render_chunk=8,
        outdir=str(tmp_path / "early"),
        device="cpu",
    )

    assert len(rows) == 1
    row = rows[0]
    assert row["stopped_early"] is True
    assert row["stopped_at"] is not None
    assert row["iterations_run"] < 20
    assert row["auc_psnr_horizon"] == "nominal_hold_last"


def test_stage_search_resume_skips_completed_cells_and_limits_new_cells(tmp_path):
    img_path = tmp_path / "toy.png"
    outdir = tmp_path / "resume"
    _write_toy(img_path)
    common = {
        "images": [str(img_path)],
        "budgets": [16],
        "seeds": [0],
        "iters": 2,
        "max_side": None,
        "mode": "influence",
        "strategies": ["aniso_flanking"],
        "tensor_operators": ["central"],
        "tensor_colors": ["luma"],
        "density_modes": ["structure"],
        "sampling_modes": ["density_random"],
        "orientation_modes": ["tensor"],
        "color_modes": ["bilinear"],
        "scale_modes": ["spacing"],
        "scale_cap_modes": ["none"],
        "background_modes": ["off"],
        "opacity_modes": ["none"],
        "renderers": ["normalized"],
        "aa_dilations": [0.0],
        "color_basis_modes": ["constant"],
        "color_solve_modes": ["none"],
        "pixel_losses": ["l1", "charbonnier"],
        "loss_weight_modes": ["none"],
        "optimizers": ["adam"],
        "lr_schedules": ["none"],
        "refine_modes": ["none"],
        "pyramid_modes": ["single"],
        "render_chunk": 8,
        "outdir": str(outdir),
        "device": "cpu",
    }

    first = run_stage_search(**common, max_new_cells=1)
    assert len(first) == 1
    assert len((outdir / "stage_search.jsonl").read_text(encoding="utf-8").splitlines()) == 1

    second = run_stage_search(**common, resume=True, max_new_cells=10)
    assert len(second) == 2
    assert {row["loss"] for row in second} == {"l1", "charbonnier"}
    assert len((outdir / "stage_search.jsonl").read_text(encoding="utf-8").splitlines()) == 2

    third = run_stage_search(**common, resume=True, max_new_cells=10)
    assert len(third) == 2
    assert len((outdir / "stage_search.jsonl").read_text(encoding="utf-8").splitlines()) == 2


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


def test_refine_arms_do_not_exceed_cell_budget(tmp_path):
    # BENCH-002 #1: an adding-refine arm must end at (not above) the cell budget, so it is
    # compared at equal capacity with the baseline it is ranked against.
    img_path = tmp_path / "toy.png"
    _write_toy(img_path)
    rows = run_stage_search(
        [str(img_path)], budgets=[32], seeds=[0], iters=4, max_side=None,
        strategies=["aniso_flanking"], tensor_operators=["central"],
        density_modes=["structure"], sampling_modes=["density_random"],
        color_modes=["bilinear"], scale_modes=["spacing"], scale_cap_modes=["none"],
        opacity_modes=["none"], renderers=["normalized"], pixel_losses=["l1"],
        optimizers=["adam"], lr_schedules=["none"],
        refine_modes=[
            "none", "residual_add", "residual_add_nms_residual_color", "duplicate",
            "fp_duplicate", "ranked_wave", "absgrad_wave", "relocate",
        ],
        pyramid_modes=["single"],
        split_every=2, split_count=8, render_chunk=8,
        outdir=str(tmp_path / "budget"), device="cpu",
    )
    assert rows and all(r["n_gaussians"] <= r["budget"] for r in rows)
    # adding arms start below budget and climb back to it (equal final capacity)
    adds = [r for r in rows if r["refine"] not in ("none", "relocate")]
    assert adds and all(r["init_budget"] < r["budget"] for r in adds)
    relocate = [r for r in rows if r["refine"] == "relocate"]
    assert relocate and all(r["init_budget"] == r["budget"] for r in relocate)


def test_refine_kwargs_threads_fit004_modes():
    nms = _refine_kwargs("residual_tensor_add_nms", 5, 7, None, 0.0)
    assert nms == {
        "split_every": 5,
        "split_count": 7,
        "split_mode": "residual_tensor_add",
        "split_min_spacing": 1.0,
        "split_oversample": 8.0,
    }

    combined = _refine_kwargs("residual_add_nms_residual_color", 5, 7, None, 0.0)
    assert combined["split_mode"] == "residual_add"
    assert combined["split_min_spacing"] == 1.0
    assert combined["split_oversample"] == 8.0
    assert combined["split_color_init"] == "residual"

    assert _refine_kwargs("fp_duplicate", 5, 7, None, 0.0)["split_mode"] == "fp_duplicate"
    assert _refine_kwargs("moment_preserving", 5, 7, None, 0.0)["split_mode"] == "moment_preserving"
    assert _refine_kwargs("ranked_wave", 5, 7, None, 0.0)["split_mode"] == "ranked_wave"
    assert _refine_kwargs("absgrad_wave", 5, 7, None, 0.0)["split_mode"] == "absgrad_wave"
    assert _refine_kwargs("freq_violation", 5, 7, None, 0.0)["split_mode"] == "freq_violation"
    assert _refine_kwargs("relocate", 5, 7, None, 0.0) == {
        "relocate_every": 5,
        "relocate_count": 7,
    }


def test_legacy_refine_alias_table_covers_all_influence_modes():
    for mode in LEGACY_REFINE_MODES:
        cfg = _legacy_refine_config(mode)
        kwargs = _refine_kwargs_from_config(cfg, 5, 7, 3, 1e-2)
        assert cfg["refine"] == mode
        assert cfg["refine_site"] in {
            "none", "residual", "residual_tensor", "support", "ranked", "absgrad",
            "freq_violation",
        }
        assert cfg["refine_primitive"] in {
            "duplicate", "fp", "moment_preserving", "sampled_add",
        }
        assert cfg["refine_nms"] in {"off", "on"}
        assert cfg["refine_color"] in {"target", "residual"}
        if cfg["refine_site"] != "none":
            assert kwargs["split_every"] == 5
            assert kwargs["split_count"] == 7
            assert kwargs["refine_site"] == cfg["refine_site"]
            assert kwargs["refine_primitive"] == cfg["refine_primitive"]
            assert kwargs["refine_nms"] == cfg["refine_nms"]


def test_aa_dilation_is_stage_axis(tmp_path):
    img_path = tmp_path / "toy.png"
    _write_toy(img_path)
    rows = run_stage_search(
        [str(img_path)], budgets=[16], seeds=[0], iters=2, max_side=None,
        strategies=["aniso_flanking"], tensor_operators=["central"],
        density_modes=["structure"], sampling_modes=["density_random"],
        color_modes=["bilinear"], scale_modes=["spacing"], scale_cap_modes=["none"],
        opacity_modes=["none"], renderers=["normalized"], aa_dilations=[0.0, 0.3],
        pixel_losses=["l1"], optimizers=["adam"], lr_schedules=["none"],
        refine_modes=["none"], pyramid_modes=["single"], render_chunk=8,
        outdir=str(tmp_path / "aa"), device="cpu",
    )
    assert len(rows) == 2
    assert {r["aa"] for r in rows} == {0.0, 0.3}
    assert all("aa=" in r["config_label"] for r in rows)


def test_color_solve_is_stage_axis(tmp_path):
    img_path = tmp_path / "toy.png"
    _write_toy(img_path)
    rows = run_stage_search(
        [str(img_path)], budgets=[16], seeds=[0], iters=10, max_side=None,
        strategies=["aniso_flanking"], tensor_operators=["central"],
        density_modes=["structure"], sampling_modes=["density_random"],
        color_modes=["bilinear"], scale_modes=["spacing"], scale_cap_modes=["none"],
        opacity_modes=["none"], renderers=["normalized"], aa_dilations=[0.0],
        color_solve_modes=["none", "every10"],
        pixel_losses=["l1"], optimizers=["adam"], lr_schedules=["none"],
        refine_modes=["none"], pyramid_modes=["single"], render_chunk=8,
        outdir=str(tmp_path / "color_solve"), device="cpu",
    )
    assert len(rows) == 2
    by_mode = {r["color_solve"]: r for r in rows}
    assert by_mode["none"]["color_solve_event_count"] == 0
    assert by_mode["every10"]["color_solve_every"] == 10
    assert by_mode["every10"]["color_solve_event_count"] == 1
    assert all("color_solve=" in r["config_label"] for r in rows)


def test_split_recovery_stage_mode_parsers():
    assert _state_seed_kwargs("off") == {"seed_new_row_optimizer_state": False}
    assert _state_seed_kwargs("on") == {"seed_new_row_optimizer_state": True}
    assert _row_temper_kwargs("off") == {"new_row_temper_iters": 0}
    assert _row_temper_kwargs("warmup5") == {"new_row_temper_iters": 5}
    assert _support_fade_kwargs("off") == {
        "support_fade": False,
        "support_fade_until_frac": None,
    }
    assert _support_fade_kwargs("on") == {
        "support_fade": True,
        "support_fade_until_frac": None,
    }
    assert _support_fade_kwargs("until0.5") == {
        "support_fade": False,
        "support_fade_until_frac": 0.5,
    }


def test_feature_relative_scale_cap_is_stage_axis():
    assert _scale_cap_kwargs("feature_rel") == {
        "scale_cap_mode": "feature_rel",
        "scale_cap_max": None,
    }
    assert _scale_cap_kwargs("feature_relative") == {
        "scale_cap_mode": "feature_rel",
        "scale_cap_max": None,
    }
    assert "feature_rel" in INFLUENCE_DEFAULTS["scale_cap_modes"]


def test_loss_weighting_is_stage_axis():
    assert _loss_weight_kwargs("none") == {"loss_weighting": "none"}
    assert _loss_weight_kwargs("tensor") == {
        "loss_weighting": "tensor",
        "loss_weight_beta": 1.0,
    }
    assert _loss_weight_kwargs("tensor_2.5") == {
        "loss_weighting": "tensor",
        "loss_weight_beta": 2.5,
    }
    assert "tensor" in INFLUENCE_DEFAULTS["loss_weight_modes"]


def test_background_layer_is_stage_axis():
    assert _background_kwargs("off") == {"background_fraction": 0.0, "background_grid": 0}
    assert _background_kwargs("frac0.25_grid4") == {
        "background_fraction": 0.25,
        "background_grid": 4,
    }
    assert "frac0.10_grid16" in INFLUENCE_DEFAULTS["background_modes"]


def test_stage_search_runs_background_layer_and_logs_accounting(tmp_path):
    img_path = tmp_path / "toy.png"
    _write_toy(img_path)
    rows = run_stage_search(
        [str(img_path)], budgets=[16], seeds=[0], iters=3, max_side=None,
        strategies=["random"], tensor_operators=["central"], tensor_colors=["luma"],
        density_modes=["structure"], sampling_modes=["wse"],
        orientation_modes=["tensor"], color_modes=["bilinear"], scale_modes=["spacing"],
        scale_cap_modes=["none"], background_modes=["off", "frac0.25_grid2"],
        opacity_modes=["none"], renderers=["normalized"], aa_dilations=[0.0],
        color_basis_modes=["constant"], color_solve_modes=["none"],
        pixel_losses=["l1"], loss_weight_modes=["none"],
        optimizers=["adam"], lr_schedules=["none"], refine_modes=["none"],
        pyramid_modes=["single"], render_chunk=8, outdir=str(tmp_path / "background"),
        device="cpu",
    )

    assert len(rows) == 2
    by_bg = {r["background"]: r for r in rows}
    assert by_bg["off"]["background_count"] == 0
    assert by_bg["frac0.25_grid2"]["background_count"] == 4
    assert by_bg["frac0.25_grid2"]["detail_count"] == 12
    assert by_bg["frac0.25_grid2"]["background_actual_fraction"] == pytest.approx(0.25)
    assert by_bg["frac0.25_grid2"]["large_support_count"] >= 0
    assert "background=frac0.25_grid2" in by_bg["frac0.25_grid2"]["config_label"]


def test_stage_search_runs_tensor_loss_weighting(tmp_path):
    img_path = tmp_path / "toy.png"
    _write_toy(img_path)
    rows = run_stage_search(
        [str(img_path)], budgets=[16], seeds=[0], iters=3, max_side=None,
        strategies=["aniso_onedge"], tensor_operators=["central"], tensor_colors=["luma"],
        density_modes=["structure"], sampling_modes=["density_random"],
        orientation_modes=["tensor"], color_modes=["bilinear"], scale_modes=["spacing"],
        scale_cap_modes=["none"], opacity_modes=["none"], renderers=["normalized"],
        aa_dilations=[0.0], color_basis_modes=["constant"], color_solve_modes=["none"],
        pixel_losses=["l1"], loss_weight_modes=["none", "tensor"],
        optimizers=["adam"], lr_schedules=["none"], refine_modes=["none"],
        pyramid_modes=["single"], render_chunk=8, outdir=str(tmp_path / "loss_weight"),
        device="cpu",
    )

    assert len(rows) == 2
    by_weight = {r["loss_weight"]: r for r in rows}
    assert by_weight["none"]["loss_weighting"] == "none"
    assert by_weight["none"]["loss_weight_mean"] is None
    assert by_weight["tensor"]["loss_weighting"] == "tensor"
    assert by_weight["tensor"]["loss_weight_mean"] >= 1.0
    assert by_weight["tensor"]["loss_weight_max"] >= by_weight["tensor"]["loss_weight_mean"]


def test_split_recovery_levers_are_stage_axes(tmp_path):
    img_path = tmp_path / "toy.png"
    _write_toy(img_path)
    rows = run_stage_search(
        [str(img_path)], budgets=[24], seeds=[0], iters=5, max_side=None,
        strategies=["aniso_flanking"], tensor_operators=["central"], tensor_colors=["luma"],
        density_modes=["structure"], sampling_modes=["density_random"],
        orientation_modes=["tensor"], color_modes=["bilinear"], scale_modes=["spacing"],
        scale_cap_modes=["none"], opacity_modes=["none"], renderers=["normalized"],
        aa_dilations=[0.0], color_basis_modes=["constant"], color_solve_modes=["none"],
        pixel_losses=["l1"], optimizers=["adam"], lr_schedules=["none"],
        refine_modes=["duplicate"], state_seed_modes=["off", "on"],
        row_temper_modes=["off", "warmup3"], support_fade_modes=["off", "until0.5"],
        pyramid_modes=["single"], split_every=2, split_count=4, render_chunk=8,
        outdir=str(tmp_path / "fit011_axes"), device="cpu",
    )

    assert len(rows) == 8
    assert all(r["status"] == "ok" for r in rows)
    assert {r["state_seed"] for r in rows} == {"off", "on"}
    assert {r["row_temper"] for r in rows} == {"off", "warmup3"}
    assert {r["support_fade"] for r in rows} == {"off", "until0.5"}
    assert all("state_seed=" in r["config_label"] for r in rows)
    assert all("row_temper=" in r["config_label"] for r in rows)
    assert all("support_fade=" in r["config_label"] for r in rows)
    warmup = [r for r in rows if r["row_temper"] == "warmup3"]
    assert warmup and all(r["new_row_temper_iters"] == 3 for r in warmup)
    assert any((r["post_split_delta_mean"] is not None) for r in rows)


def test_new_stage_smoke_matrix_records_expected_events_and_html(tmp_path):
    img_path = tmp_path / "toy.png"
    outdir = tmp_path / "new_stages"
    _write_toy(img_path)

    rows = run_stage_search(
        [str(img_path)], budgets=[32], seeds=[0], iters=10, max_side=None,
        strategies=["aniso_flanking"], tensor_operators=["central"], tensor_colors=["luma"],
        density_modes=["structure"], sampling_modes=["density_random"],
        orientation_modes=["tensor"], color_modes=["bilinear"], scale_modes=["spacing"],
        scale_cap_modes=["none"], opacity_modes=["none"], renderers=["normalized"],
        aa_dilations=[0.0], color_basis_modes=["constant"],
        color_solve_modes=["none", "every10"], pixel_losses=["l1"], optimizers=["adam"],
        lr_schedules=["none"], refine_modes=["none", "freq_violation", "moment_preserving"],
        pyramid_modes=["single"], split_every=5, split_count=8, render_chunk=8,
        outdir=str(outdir), device="cpu",
    )

    assert len(rows) == 6
    assert all(r["status"] == "ok" for r in rows)
    assert {(r["color_solve"], r["refine"]) for r in rows} == {
        ("none", "none"),
        ("none", "freq_violation"),
        ("none", "moment_preserving"),
        ("every10", "none"),
        ("every10", "freq_violation"),
        ("every10", "moment_preserving"),
    }

    for row in rows:
        if row["color_solve"] == "every10":
            assert row["color_solve_every"] == 10
            assert row["color_solve_event_count"] == 1
        else:
            assert row["color_solve_every"] is None
            assert row["color_solve_event_count"] == 0

        if row["refine"] == "none":
            assert row["split_event_count"] == 0
            assert row["n_gaussians"] == row["budget"]
            assert row["init_budget"] == row["budget"]
        else:
            assert row["split_event_count"] == 1
            assert row["n_gaussians"] == row["budget"]
            assert row["init_budget"] < row["budget"]

        if row["refine"] == "freq_violation":
            assert row["freq_violation_score_mean"] is not None
            assert (
                row["freq_violation_axis0_count"] + row["freq_violation_axis1_count"]
                == row["split_event_count"] * 8
            )

    html = (outdir / "index.html").read_text(encoding="utf-8")
    assert "Cells</div><div class=\"value\">6</div>" in html
    assert "OK</div><div class=\"value\">6</div>" in html
    assert "<code>every10</code>" in html
    assert "<code>freq_violation</code>" in html
    assert "<code>moment_preserving</code>" in html
    assert "stage_search.json" in html


def test_factored_refine_axes_smoke_residual_tensor_moment(tmp_path):
    img_path = tmp_path / "toy.png"
    outdir = tmp_path / "factored_refine"
    _write_toy(img_path)

    rows = run_stage_search(
        [str(img_path)], budgets=[32], seeds=[0], iters=10, max_side=None,
        strategies=["aniso_flanking"], tensor_operators=["central"], tensor_colors=["luma"],
        density_modes=["structure"], sampling_modes=["density_random"],
        orientation_modes=["tensor"], color_modes=["bilinear"], scale_modes=["spacing"],
        scale_cap_modes=["none"], opacity_modes=["none"], renderers=["normalized"],
        aa_dilations=[0.0], color_basis_modes=["constant"], color_solve_modes=["none"],
        pixel_losses=["l1"], optimizers=["adam"], lr_schedules=["none"],
        refine_sites=["none", "residual", "residual_tensor"],
        refine_primitives=["sampled_add", "moment_preserving"],
        refine_nms_modes=["off"], refine_color_inits=["target"],
        refine_prune_modes=["off"], refine_relocate_modes=["off"],
        pyramid_modes=["single"], split_every=5, split_count=8, render_chunk=8,
        outdir=str(outdir), device="cpu",
    )

    assert len(rows) == 5  # site=none makes primitive inert and dedupes one cell
    assert all(r["status"] == "ok" for r in rows)
    combos = {(r["refine_site"], r["refine_primitive"]) for r in rows}
    assert ("residual_tensor", "moment_preserving") in combos
    target = [
        r for r in rows
        if r["refine_site"] == "residual_tensor"
        and r["refine_primitive"] == "moment_preserving"
    ][0]
    assert target["split_event_count"] == 1
    assert target["n_gaussians"] == target["budget"]
    assert target["init_budget"] < target["budget"]
    assert "refine_site=residual_tensor" in target["config_label"]


def test_color_basis_is_stage_axis(tmp_path):
    img_path = tmp_path / "toy.png"
    _write_toy(img_path)
    rows = run_stage_search(
        [str(img_path)], budgets=[16], seeds=[0], iters=3, max_side=None,
        strategies=["aniso_flanking"], tensor_operators=["central"],
        density_modes=["structure"], sampling_modes=["density_random"],
        color_modes=["bilinear"], scale_modes=["spacing"], scale_cap_modes=["none"],
        opacity_modes=["none"], renderers=["normalized"], aa_dilations=[0.0],
        color_basis_modes=["constant", "affine"], color_solve_modes=["none"],
        pixel_losses=["l1"], optimizers=["adam"], lr_schedules=["none"],
        refine_modes=["none"], pyramid_modes=["single"], render_chunk=8,
        outdir=str(tmp_path / "color_basis"), device="cpu",
    )
    assert len(rows) == 2
    assert {r["color_basis"] for r in rows} == {"constant", "affine"}
    assert all(r["status"] == "ok" for r in rows)
    assert all("color_basis=" in r["config_label"] for r in rows)


def test_stage_search_records_explicit_pyramid_level_iters(tmp_path):
    img_path = tmp_path / "toy.png"
    _write_toy(img_path)
    rows = run_stage_search(
        [str(img_path)], budgets=[16], seeds=[0], iters=6, max_side=None,
        strategies=["aniso_flanking"], tensor_operators=["central"], tensor_colors=["luma"],
        density_modes=["structure"], sampling_modes=["density_random"],
        orientation_modes=["tensor"], color_modes=["bilinear"], scale_modes=["spacing"],
        scale_cap_modes=["none"], opacity_modes=["none"], renderers=["normalized"],
        aa_dilations=[0.0], color_basis_modes=["constant"], color_solve_modes=["none"],
        pixel_losses=["l1"], optimizers=["adam"], lr_schedules=["none"],
        refine_modes=["none"], pyramid_modes=["pyramid"],
        pyramid_levels=2, pyramid_fractions=[0.5, 0.5], pyramid_level_iters=[2, 4],
        render_chunk=8, outdir=str(tmp_path / "pyramid_level_iters"), device="cpu",
    )

    assert len(rows) == 1
    assert rows[0]["status"] == "ok"
    assert rows[0]["level_iters"] == [2, 4]
    assert rows[0]["iterations_run"] == 6


def test_adaptive_count_metadata_is_recorded(tmp_path):
    img_path = tmp_path / "toy.png"
    _write_toy(img_path)
    rows = run_stage_search(
        [str(img_path)], budgets=[8], seeds=[0], iters=4, max_side=None,
        strategies=["aniso_flanking"], tensor_operators=["central"],
        density_modes=["structure"], sampling_modes=["density_random"],
        color_modes=["bilinear"], scale_modes=["spacing"], scale_cap_modes=["none"],
        opacity_modes=["none"], renderers=["normalized"], aa_dilations=[0.0],
        color_basis_modes=["constant"], color_solve_modes=["none"],
        pixel_losses=["l1"], optimizers=["adam"], lr_schedules=["none"],
        refine_modes=["none"], pyramid_modes=["single"], render_chunk=8,
        adaptive_count=True, max_gaussians=12, target_psnr=99.0,
        adaptive_growth_every=1, adaptive_growth_count=2,
        adaptive_split_mode="residual_add", adaptive_min_delta_psnr=1e6,
        adaptive_patience=1, outdir=str(tmp_path / "adaptive"), device="cpu",
    )

    assert len(rows) == 1 and rows[0]["status"] == "ok"
    row = rows[0]
    assert row["adaptive_count"] is True
    assert row["adaptive_event_count"] >= 1
    assert row["adaptive_stop_reason"] in {"stalled", "max_gaussians_reached"}
    assert row["adaptive_selected_n"] == row["n_gaussians"]
    assert row["estimated_bpp"] > 0.0


def test_color_solve_kwargs_parse_stage_modes():
    assert _color_solve_kwargs("none") == {
        "color_solve_every": None,
        "color_solve_schedule": "none",
    }
    assert _color_solve_kwargs("every10") == {
        "color_solve_every": 10,
        "color_solve_schedule": "every",
    }
    assert _color_solve_kwargs("every25") == {
        "color_solve_every": 25,
        "color_solve_schedule": "every",
    }
    assert _color_solve_kwargs("init+on_split") == {
        "color_solve_every": None,
        "color_solve_schedule": "init+on_split",
    }
    assert _color_solve_kwargs("every10+on_split") == {
        "color_solve_every": 10,
        "color_solve_schedule": "every+on_split",
    }
    with pytest.raises(ValueError):
        _color_solve_kwargs("every0")


def test_config_json_is_written(tmp_path):
    img_path = tmp_path / "toy.png"
    outdir = tmp_path / "cfg"
    _write_toy(img_path)
    run_stage_search(
        [str(img_path)], budgets=[16], seeds=[0], iters=2, max_side=None,
        strategies=["aniso_flanking"], tensor_operators=["central"],
        density_modes=["structure"], sampling_modes=["density_random"],
        color_modes=["bilinear"], scale_modes=["spacing"], scale_cap_modes=["none"],
        opacity_modes=["none"], renderers=["normalized"], pixel_losses=["l1"],
        optimizers=["adam"], lr_schedules=["none"], refine_modes=["none"],
        pyramid_modes=["single"], render_chunk=8, outdir=str(outdir), device="cpu",
    )
    import json
    cfg = json.loads((outdir / "config.json").read_text())
    assert cfg["device"] == "cpu" and cfg["versions"]["structsplat"] is not None


def test_broken_cell_does_not_void_the_sweep(tmp_path):
    # BENCH-002 #4: one failing arm (renderer=cuda on a CPU box) must leave an error row and let
    # the rest of the sweep complete and write its outputs.
    img_path = tmp_path / "toy.png"
    outdir = tmp_path / "iso"
    _write_toy(img_path)
    rows = run_stage_search(
        [str(img_path)], budgets=[16], seeds=[0], iters=2, max_side=None,
        strategies=["aniso_flanking"], tensor_operators=["central"],
        density_modes=["structure"], sampling_modes=["density_random"],
        color_modes=["bilinear"], scale_modes=["spacing"], scale_cap_modes=["none"],
        opacity_modes=["none"], renderers=["cuda", "normalized"], pixel_losses=["l1"],
        optimizers=["adam"], lr_schedules=["none"], refine_modes=["none"],
        pyramid_modes=["single"], render_chunk=8, outdir=str(outdir), device="cpu",
    )
    statuses = {r["renderer"]: r["status"] for r in rows}
    assert statuses["cuda"] == "error" and statuses["normalized"] == "ok"
    assert (outdir / "summary.md").exists()
    assert (outdir / "stage_search.jsonl").exists()


def test_canonicalize_keeps_distinct_quadtree_density_cells():
    # BENCH-002 #6: the jittered_grid density pin must NOT apply to quadtree strategies (they
    # read density), so two quadtree cells that differ only in density stay distinct.
    canon = {k: "central" for k in ("tensor", "tensor_color")}
    canon.update({"density": "structure", "sampling": "wse", "orientation": "tensor"})
    base = {"strategy": "quadtree_hybrid", "tensor": "central", "tensor_color": "luma",
            "density": "hybrid", "sampling": "jittered_grid", "orientation": "tensor",
            "color": "bilinear", "scale": "spacing", "scale_cap": "none", "opacity": "none",
            "renderer": "normalized", "loss": "l1", "optimizer": "adam", "lr_schedule": "none",
            "refine": "none", "pyramid": "single"}
    a = _canonicalize({**base, "density": "structure"}, canon)
    b = _canonicalize({**base, "density": "hybrid"}, canon)
    assert a["density"] != b["density"]           # not collapsed
    # but a blue-noise strategy with jittered_grid DOES pin density (placement ignores it)
    bn = _canonicalize({**base, "strategy": "aniso_flanking", "density": "hybrid"}, canon)
    assert bn["density"] == "structure"


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
        scale_cap_modes=["feature12"], background_modes=["off"],
        opacity_modes=["none"], renderers=["normalized"],
        aa_dilations=[0.0], color_basis_modes=["constant"], color_solve_modes=["none"],
        pixel_losses=["l1"], loss_weight_modes=["none"],
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
