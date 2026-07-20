# ruff: noqa: E402
import copy
import csv
import hashlib
import json
import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from benchmarks.gauge_equivalence_audit import (
    _action_hash,
    _counterexample_field,
    _target_suite as fit019_target_suite,
    _tensor_sha256,
)
from structsplat import metrics as M
from structsplat.config import FitConfig
from structsplat.fit import _render, fit
from benchmarks.perturb_recover_spectroscopy import (
    COVERAGES,
    HELDOUT_VARIANTS,
    RECOVERY_STEPS,
    TRAIN_VARIANTS,
    _aggregate_seed_pairs,
    _coverage_actions,
    _extract_trajectory,
    _fit_ridge_model,
    _grouped_cv_ridge,
    _predict_ridge,
    _target_suite,
    _terminal_auc,
    _write_paired_outputs,
    aggregate_and_decide,
    run,
)


FAMILIES = (
    "sinusoid",
    "chirp",
    "oriented_ramp",
    "opposing_ramps",
    "soft_ridge",
    "lattice",
)


def test_frozen_constants_and_coverage_chain_replace_one_ticket_at_a_time():
    assert RECOVERY_STEPS == (0, 1, 2, 5, 10, 20, 40, 60, 100, 200)
    assert COVERAGES == (5, 6, 7, 8)
    assert TRAIN_VARIANTS == (0, 2, 3, 5)
    assert HELDOUT_VARIANTS == (1, 4)

    ranking = [17, 4, 29, 8, 2, 31, 11, 6]
    actions = _coverage_actions(ranking)
    assert actions == {
        5: [17, 4, 29, 8, 2, 29, 4, 17],
        6: [17, 4, 29, 8, 2, 31, 4, 17],
        7: [17, 4, 29, 8, 2, 31, 11, 17],
        8: [17, 4, 29, 8, 2, 31, 11, 6],
    }
    for coverage in COVERAGES:
        assert len(actions[coverage]) == 8
        assert len(set(actions[coverage])) == coverage
        assert set(actions[coverage]) <= set(ranking)

    for before_coverage, after_coverage in zip(COVERAGES[:-1], COVERAGES[1:], strict=True):
        before = actions[before_coverage]
        after = actions[after_coverage]
        changed = [idx for idx, pair in enumerate(zip(before, after, strict=True)) if pair[0] != pair[1]]
        assert len(changed) == 1
        idx = changed[0]
        assert before.count(before[idx]) > 1
        assert after[idx] == ranking[after_coverage - 1]
        assert after[idx] not in before


@pytest.mark.parametrize("ranking", [[0, 1, 2, 3, 4, 5, 6], [0, 1, 2, 3, 4, 5, 6, 6]])
def test_coverage_chain_rejects_short_or_nonunique_rankings(ranking):
    with pytest.raises(ValueError, match="eight|unique"):
        _coverage_actions(ranking)


def test_target_suite_is_deterministic_split_complete_and_disjoint_from_fit019():
    left = _target_suite(48)
    right = _target_suite(48)
    assert tuple(left) == tuple(right)
    assert len(left) == 36

    observed_families = []
    observed_hashes = set()
    split_counts = {"train": 0, "heldout": 0}
    for target_id, entry in left.items():
        assert set(entry) == {"family", "variant", "split", "params", "image"}
        assert entry["family"] in FAMILIES
        assert entry["variant"] in range(6)
        assert entry["split"] == (
            "train" if entry["variant"] in TRAIN_VARIANTS else "heldout"
        )
        assert target_id == f"{entry['family']}_v{entry['variant']}"
        assert isinstance(entry["params"], tuple)
        image = entry["image"]
        assert image.shape == (48, 48, 3)
        assert image.dtype == np.float32
        assert np.isfinite(image).all()
        assert float(image.min()) >= 0.0
        assert float(image.max()) <= 1.0
        assert np.array_equal(image, right[target_id]["image"])
        assert entry["params"] == right[target_id]["params"]
        observed_families.append(entry["family"])
        observed_hashes.add(_tensor_sha256(image))
        split_counts[entry["split"]] += 1

    assert tuple(dict.fromkeys(observed_families)) == FAMILIES
    assert len(observed_hashes) == 36
    assert split_counts == {"train": 24, "heldout": 12}
    old_hashes = {_tensor_sha256(image) for image in fit019_target_suite(48).values()}
    assert observed_hashes.isdisjoint(old_hashes)
    frozen_payload = "\n".join(
        f"{target_id}:{_tensor_sha256(entry['image'])}" for target_id, entry in left.items()
    )
    assert hashlib.sha256(frozen_payload.encode()).hexdigest() == (
        "4e51518b9c2b1816311f1b2856c19ea77acaea913b534987fa776cf160740ebe"
    )


def test_extract_trajectory_maps_preupdate_history_and_appends_terminal_state():
    result = {
        "history": {
            "iter": list(range(200)),
            "psnr": [10.0 + 0.05 * step for step in range(200)],
            "loss": [2.0 - 0.005 * step for step in range(200)],
        },
        "iterations_run": 200,
        "trajectory_terminal_iter": 200,
        "trajectory_terminal_psnr": 21.25,
        "trajectory_terminal_loss": 0.125,
    }
    trajectory = _extract_trajectory(result)

    assert trajectory["full_steps"] == list(range(201))
    assert trajectory["full_psnr"][:-1] == result["history"]["psnr"]
    assert trajectory["full_psnr"][-1] == pytest.approx(21.25)
    assert tuple(trajectory["psnr_by_step"]) == RECOVERY_STEPS
    assert tuple(trajectory["loss_by_step"]) == RECOVERY_STEPS
    assert trajectory["psnr_by_step"][0] == pytest.approx(10.0)
    assert trajectory["psnr_by_step"][20] == pytest.approx(11.0)
    assert trajectory["psnr_by_step"][200] == pytest.approx(21.25)
    assert trajectory["loss_by_step"][100] == pytest.approx(1.5)
    assert trajectory["loss_by_step"][200] == pytest.approx(0.125)


def test_real_fit_history_step_zero_and_prefix_semantics_are_exact():
    field = _counterexample_field()
    target = torch.rand(12, 12, 3, generator=torch.Generator().manual_seed(27))
    common = {
        "renderer": "normalized",
        "render_chunk": 512,
        "log_every": 1,
        "lr_schedule": "none",
        "checkpoint_policy": "terminal",
    }
    immediate = _render(field, FitConfig(**common), 12, 12)
    long = fit(field.detached(), target, FitConfig(iters=200, **common), verbose=False)
    short = fit(field.detached(), target, FitConfig(iters=20, **common), verbose=False)
    assert long["history"]["psnr"][0] == float(M.psnr(immediate, target))
    assert long["history"]["psnr"][20] == short["trajectory_terminal_psnr"]


@pytest.mark.parametrize(
    "mutation,match",
    [
        ({"iterations_run": 199}, "200"),
        ({"trajectory_terminal_iter": 199}, "terminal"),
        ({"history": {"iter": list(range(199)), "psnr": [1.0] * 199, "loss": [1.0] * 199}}, "history"),
    ],
)
def test_extract_trajectory_rejects_incomplete_or_misaligned_runs(mutation, match):
    result = {
        "history": {
            "iter": list(range(200)),
            "psnr": [1.0] * 200,
            "loss": [1.0] * 200,
        },
        "iterations_run": 200,
        "trajectory_terminal_iter": 200,
        "trajectory_terminal_psnr": 1.0,
        "trajectory_terminal_loss": 1.0,
    }
    result.update(mutation)
    with pytest.raises(ValueError, match=match):
        _extract_trajectory(result)


def test_terminal_auc_includes_the_terminal_endpoint_and_validates_input():
    assert _terminal_auc([0, 1, 2], [0.0, 2.0, 2.0]) == pytest.approx(1.5)
    assert _terminal_auc([0, 200], [10.0, 14.0]) == pytest.approx(12.0)
    with pytest.raises(ValueError, match="length"):
        _terminal_auc([0, 1], [1.0])
    with pytest.raises(ValueError, match="increasing"):
        _terminal_auc([0, 2, 1], [1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="finite"):
        _terminal_auc([0, 1], [1.0, math.nan])


def _raw_rows_for_target(
    target: str,
    *,
    family: str = "sinusoid",
    variant: int = 0,
    split: str = "train",
    deltas: dict[int, tuple[float, float, float, float]] | None = None,
):
    deltas = deltas or {
        5: (-0.30, -0.10, 0.20, 0.40),
        6: (-0.20, -0.05, 0.10, 0.25),
        7: (-0.10, 0.00, 0.05, 0.10),
    }
    rows = []
    for seed in (0, 1, 2):
        reference = {
            0: 20.0 + 0.10 * seed,
            2: 20.3 + 0.10 * seed,
            10: 21.0 + 0.10 * seed,
            200: 22.0 + 0.10 * seed,
        }
        for coverage in COVERAGES:
            if coverage == 8:
                d0 = d2 = d10 = dy = 0.0
            else:
                base_d0, base_d2, base_d10, base_dy = deltas[coverage]
                d0 = base_d0 + 0.01 * seed
                d2 = base_d2 + 0.02 * seed
                d10 = base_d10 + 0.03 * seed
                dy = base_dy + 0.04 * seed
            rows.append(
                {
                    "target": target,
                    "family": family,
                    "variant": variant,
                    "split": split,
                    "seed": seed,
                    "coverage": coverage,
                    "pre_psnr": 19.0 + 0.20 * seed,
                    "residual_entropy": 0.70 + 0.01 * seed,
                    "residual_top_decile_mass": 0.35 + 0.01 * seed,
                    "intervention_rms": 0.02 * (8 - coverage) + 0.001 * seed,
                    "psnr_step0": reference[0] + d0,
                    "psnr_step2": reference[2] + d2,
                    "psnr_step10": reference[10] + d10,
                    "psnr_step200": reference[200] + dy,
                }
            )
    return rows


def test_seed_pairs_are_paired_then_averaged_before_model_fitting():
    aggregated = _aggregate_seed_pairs(_raw_rows_for_target("sinusoid_v0"))
    assert len(aggregated) == 3
    assert [row["coverage"] for row in aggregated] == [5, 6, 7]

    c5 = aggregated[0]
    assert c5["target"] == "sinusoid_v0"
    assert c5["family"] == "sinusoid"
    assert c5["variant"] == 0
    assert c5["split"] == "train"
    assert c5["seed_count"] == 3
    assert c5["coverage_deficit"] == 3
    assert c5["coverage_deficit_sq"] == 9
    assert c5["pre_psnr"] == pytest.approx(19.2)
    assert c5["residual_entropy"] == pytest.approx(0.71)
    assert c5["residual_top_decile_mass"] == pytest.approx(0.36)
    assert c5["intervention_rms"] == pytest.approx(0.061)
    assert c5["d0"] == pytest.approx(-0.29)
    assert c5["d2"] == pytest.approx(-0.08)
    assert c5["d10"] == pytest.approx(0.23)
    assert c5["y"] == pytest.approx(0.44)
    assert c5["c8_gain_0_10"] == pytest.approx(1.0)
    assert c5["bend"] == pytest.approx(((0.23 + 0.08) / 8.0) - ((-0.08 + 0.29) / 2.0))


def test_seed_pair_aggregation_rejects_missing_duplicate_or_mixed_cells():
    rows = _raw_rows_for_target("sinusoid_v0")
    with pytest.raises(ValueError, match="exactly one"):
        _aggregate_seed_pairs(rows[:-1])
    with pytest.raises(ValueError, match="exactly one"):
        _aggregate_seed_pairs([*rows, copy.deepcopy(rows[0])])

    mixed = copy.deepcopy(rows)
    mixed[0]["family"] = "chirp"
    with pytest.raises(ValueError, match="identity|family"):
        _aggregate_seed_pairs(mixed)


def test_ridge_standardizes_from_fit_rows_and_handles_zero_variance_features():
    rows = [
        {
            "target": f"target_{idx}",
            "x": float(idx - 4),
            "constant": 7.0,
            "y": 1.5 * float(idx - 4) - 0.2,
        }
        for idx in range(9)
    ]
    model = _fit_ridge_model(rows, ("x", "constant"), 1e-4)
    prediction = _predict_ridge(model, rows)
    assert prediction.shape == (len(rows),)
    assert np.isfinite(prediction).all()
    assert float(np.max(np.abs(prediction - np.asarray([row["y"] for row in rows])))) < 1e-3

    before = prediction.copy()
    extreme_heldout = [{"x": 1e9, "constant": -1e9, "y": 0.0}]
    assert np.isfinite(_predict_ridge(model, extreme_heldout)).all()
    # Prediction must not update the train-only scaler/model in place.
    assert np.array_equal(_predict_ridge(model, rows), before)


def test_paired_output_uses_union_schema_for_heldout_predictions(tmp_path):
    rows = [
        {"target": "train", "y": 0.1},
        {"target": "heldout", "y": -0.2, "prediction_response": -0.1},
    ]
    _write_paired_outputs(tmp_path, rows)
    with (tmp_path / "paired_rows.csv").open(newline="", encoding="utf-8") as handle:
        written = list(csv.DictReader(handle))
    assert "prediction_response" in written[0]
    assert written[0]["prediction_response"] == ""
    assert written[1]["prediction_response"] == "-0.1"
    assert json.loads((tmp_path / "paired_rows.json").read_text()) == rows


def test_grouped_cv_keeps_all_coverages_of_a_target_in_one_fold(monkeypatch):
    import benchmarks.perturb_recover_spectroscopy as spectroscopy

    rows = []
    target_names = {f"sinusoid_v{variant}" for variant in TRAIN_VARIANTS}
    for variant in TRAIN_VARIANTS:
        for coverage in (5, 6, 7):
            x = float(variant) + 0.25 * coverage
            rows.append(
                {
                    "target": f"sinusoid_v{variant}",
                    "coverage": coverage,
                    "x": x,
                    "constant": 2.0,
                    "y": 0.75 * x + 0.1,
                }
            )

    original = spectroscopy._fit_ridge_model
    fitted_target_sets = []

    def spy(fit_rows, feature_names, alpha):
        fitted_target_sets.append({row["target"] for row in fit_rows})
        return original(fit_rows, feature_names, alpha)

    monkeypatch.setattr(spectroscopy, "_fit_ridge_model", spy)
    result = _grouped_cv_ridge(rows, ("x", "constant"))

    assert result["alpha"] in {1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0}
    assert np.isfinite(_predict_ridge(result["model"], rows)).all()
    omitted = {
        tuple(sorted(target_names - fitted))
        for fitted in fitted_target_sets
        if len(fitted) == len(target_names) - 1
    }
    assert omitted == {(target,) for target in target_names}
    assert all(len(fitted) in (len(target_names) - 1, len(target_names)) for fitted in fitted_target_sets)


def _decision_rows():
    rows = []
    for family_idx, family in enumerate(FAMILIES):
        for variant in range(6):
            split = "train" if variant in TRAIN_VARIANTS else "heldout"
            target = f"{family}_v{variant}"
            for seed in (0, 1, 2):
                reference_curve = np.asarray(
                    [
                        20.0
                        + 2.0 * math.log1p(step) / math.log(201.0)
                        + 0.05 * seed
                        for step in range(201)
                    ],
                    dtype=np.float64,
                )
                ranking = list(range(8))
                actions = _coverage_actions(ranking)
                for coverage in COVERAGES:
                    if coverage == 8:
                        d2 = y = 0.0
                    else:
                        sign = 1.0 if (family_idx + variant + coverage) % 2 == 0 else -1.0
                        y = sign * (0.20 + 0.02 * (coverage - 5))
                        # With d0=d10=0, this makes the frozen bend exactly y.
                        d2 = -1.6 * y
                    delta_curve = np.interp(
                        np.arange(201, dtype=np.float64),
                        np.asarray([0.0, 2.0, 10.0, 200.0]),
                        np.asarray([0.0, d2, 0.0, y]),
                    )
                    curve = reference_curve + delta_curve
                    checkpoint_psnr = {
                        step: float(curve[step]) for step in RECOVERY_STEPS
                    }
                    checkpoint_loss = {
                        step: float(2.0 - 0.005 * step + 0.0001 * coverage)
                        for step in RECOVERY_STEPS
                    }
                    action = actions[coverage]
                    rows.append(
                        {
                            "target": target,
                            "family": family,
                            "variant": variant,
                            "split": split,
                            "seed": seed,
                            "coverage": coverage,
                            "target_sha256": hashlib.sha256(target.encode()).hexdigest(),
                            "source_combined_sha256": hashlib.sha256(b"source").hexdigest(),
                            "canonical_count": 32,
                            "canonical_field_sha256": hashlib.sha256(
                                f"{target}:{seed}".encode()
                            ).hexdigest(),
                            "ranking_groups": json.dumps(ranking, separators=(",", ":")),
                            "ranking_hash": _action_hash(ranking),
                            "action_groups": json.dumps(action, separators=(",", ":")),
                            "action_hash": _action_hash(action),
                            "unique_action_groups": coverage,
                            "repeated_action_count": 8 - coverage,
                            "pre_psnr": 19.0 + 0.02 * family_idx,
                            "pre_loss": 0.5,
                            "residual_entropy": 0.70,
                            "residual_top_decile_mass": 0.35,
                            "intervention_rms": 0.01 * (8 - coverage),
                            "trajectory_auc": _terminal_auc(
                                list(range(201)), curve.tolist()
                            ),
                            "peak_gain": float(curve.max() - curve[0]),
                            "peak_step": int(np.argmax(curve)),
                            "late_asymptote": float(curve[160:201].mean()),
                            "psnr_curve": json.dumps(
                                curve.tolist(), separators=(",", ":")
                            ),
                            "checkpoint_psnr": json.dumps(
                                checkpoint_psnr, separators=(",", ":")
                            ),
                            "checkpoint_loss": json.dumps(
                                checkpoint_loss, separators=(",", ":")
                            ),
                            "history_complete": True,
                            "final_count": 40,
                            "renders_finite": True,
                            "action_seconds": 0.01,
                            "fit_seconds": 1.0,
                            "total_seconds": 1.01,
                            **{
                                f"psnr_step{step}": checkpoint_psnr[step]
                                for step in RECOVERY_STEPS
                            },
                            **{
                                f"loss_step{step}": checkpoint_loss[step]
                                for step in RECOVERY_STEPS
                            },
                        }
                    )
    return rows


def test_decision_passes_only_when_response_shape_predicts_and_improves_selection():
    result = aggregate_and_decide(_decision_rows())
    assert result["integrity"]["passes"] is True
    assert result["signal"]["passes"] is True
    assert result["prediction"]["passes"] is True
    assert result["screening"]["passes"] is True
    assert result["decision"] == "pass"
    assert result["prediction"]["response_rmse"] <= 0.85 * result["prediction"]["early_rmse"]
    assert result["prediction"]["eligible_sign_cells"] == 36
    assert result["prediction"]["families_response_better"] == 6
    assert result["screening"]["response_regret"] < result["screening"]["early_regret"]
    assert result["screening"]["response_regret"] < result["screening"]["step10_regret"]
    assert result["screening"]["response_regret"] < result["screening"]["c8_regret"]


def _replace_curve_point(row, step, value):
    curve = json.loads(row["psnr_curve"])
    curve[step] = float(value)
    checkpoints = json.loads(row["checkpoint_psnr"])
    checkpoints[str(step)] = float(value)
    row[f"psnr_step{step}"] = float(value)
    row["psnr_curve"] = json.dumps(curve, separators=(",", ":"))
    row["checkpoint_psnr"] = json.dumps(checkpoints, separators=(",", ":"))
    row["trajectory_auc"] = _terminal_auc(list(range(201)), curve)
    row["peak_gain"] = float(max(curve) - curve[0])
    row["peak_step"] = int(np.argmax(curve))
    row["late_asymptote"] = float(np.mean(curve[160:201]))


def test_decision_stops_on_weak_signal_or_nonpredictive_bend_without_retuning():
    weak = _decision_rows()
    reference_200 = {
        (row["target"], row["seed"]): row["psnr_step200"]
        for row in weak
        if row["coverage"] == 8
    }
    for row in weak:
        if row["coverage"] != 8:
            _replace_curve_point(
                row, 200, reference_200[(row["target"], row["seed"])]
            )
    weak_result = aggregate_and_decide(weak)
    assert weak_result["integrity"]["passes"] is True
    assert weak_result["signal"]["passes"] is False
    assert weak_result["prediction"]["passes"] is False
    assert weak_result["screening"]["passes"] is False
    assert weak_result["decision"] == "stop"

    nonpredictive = _decision_rows()
    reference_2 = {
        (row["target"], row["seed"]): row["psnr_step2"]
        for row in nonpredictive
        if row["coverage"] == 8
    }
    for row in nonpredictive:
        if row["coverage"] != 8:
            _replace_curve_point(row, 2, reference_2[(row["target"], row["seed"])])
    failed = aggregate_and_decide(nonpredictive)
    assert failed["signal"]["passes"] is True
    assert failed["prediction"]["passes"] is False
    assert failed["screening"]["passes"] is False
    assert failed["decision"] == "stop"


def test_decision_rejects_missing_or_duplicate_frozen_cells():
    rows = _decision_rows()
    with pytest.raises(ValueError, match="exactly one"):
        aggregate_and_decide(rows[:-1])
    with pytest.raises(ValueError, match="exactly one"):
        aggregate_and_decide([*rows, copy.deepcopy(rows[0])])


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda row: row.pop("history_complete"), "integrity.*missing"),
        (lambda row: row.__setitem__("final_count", 39), "32->40"),
        (lambda row: row.__setitem__("renders_finite", False), "finite renders"),
        (lambda row: row.__setitem__("unique_action_groups", 8), "unique action"),
        (lambda row: row.__setitem__("action_hash", "bad"), "action hash"),
        (lambda row: row.__setitem__("psnr_step20", -999.0), "PSNR checkpoint"),
    ],
)
def test_decision_integrity_is_fail_closed(mutate, match):
    rows = _decision_rows()
    mutate(rows[0])
    with pytest.raises(ValueError, match=match):
        aggregate_and_decide(rows)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"seeds": (0, 1)}, "seeds"),
        ({"size": 32}, "size|48"),
        ({"device": "cuda"}, "device|cpu"),
        ({"pre_iters": 39}, "pre|40"),
    ],
)
def test_frozen_run_rejects_protocol_mutation_before_writing(tmp_path, kwargs, match):
    outdir = tmp_path / "result"
    with pytest.raises(ValueError, match=match):
        run(outdir=str(outdir), **kwargs)
    assert not outdir.exists()
