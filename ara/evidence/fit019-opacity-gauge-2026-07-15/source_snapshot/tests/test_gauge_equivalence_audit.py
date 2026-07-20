# ruff: noqa: E402
import copy
import json
import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from benchmarks.gauge_equivalence_audit import (
    ARMS,
    TARGET_NAMES,
    _arm_specs,
    _counterexample_field,
    _evaluate_action,
    _multiset_jaccard,
    _row_action,
    _sequential_moment_action,
    _target_suite,
    aggregate_and_decide,
    exact_opacity_refinement,
    group_responsibility_scores,
    run,
    run_algebra_audit,
)
from structsplat.config import FitConfig
from structsplat.fit import _render, _responsibility_error_density_scores
from structsplat.gaussians import GaussianField


def _logits(values):
    values = np.asarray(values, dtype=np.float32)
    return np.log(values / (1.0 - values))


def _rich_field():
    return GaussianField.from_numpy(
        means=np.asarray([[2.0, 2.0], [4.0, 3.0], [3.0, 5.0]], dtype=np.float32),
        scales=np.asarray([[1.4, 1.0], [1.0, 1.8], [1.2, 1.1]], dtype=np.float32),
        angles=np.asarray([0.2, -0.4, 0.8], dtype=np.float32),
        colors=np.asarray([[0.8, 0.1, 0.3], [0.1, 0.9, 0.2], [0.2, 0.3, 0.9]], dtype=np.float32),
        opacities=_logits([0.8, 0.4, 0.65]),
        scale_max=np.asarray([[3.0, 2.0], [2.0, 3.0], [2.5, 2.5]], dtype=np.float32),
        color_grads=np.arange(18, dtype=np.float32).reshape(3, 2, 3) / 100.0,
        background_mask=np.asarray([False, True, False]),
        filter_variance=np.asarray([0.1, 0.2, 0.3], dtype=np.float32),
    )


@pytest.mark.parametrize("fractions", [(0.5, 0.5), (0.2, 0.8), (0.1, 0.2, 0.7)])
@pytest.mark.parametrize("fade", [False, True])
def test_exact_opacity_refinement_preserves_render_denominator_and_group_scores(fractions, fade):
    field = _rich_field()
    gauge, groups, row_fractions = exact_opacity_refinement(
        field, split_groups={0, 2}, fractions=fractions
    )
    cfg = FitConfig(
        renderer="normalized",
        responsibility_mass_alpha=0.7,
        support_fade=fade,
        sigma_cutoff=2.5,
        render_chunk=1,
    )
    target = torch.rand(8, 8, 3, generator=torch.Generator().manual_seed(4))
    fade_alpha = 0.4 if fade else None
    base_render = _render(field, cfg, 8, 8, support_fade_alpha=fade_alpha)
    gauge_render = _render(gauge, cfg, 8, 8, support_fade_alpha=fade_alpha)
    base_score, base_components = _responsibility_error_density_scores(
        field, target, base_render, cfg, support_fade_alpha=fade_alpha
    )
    gauge_score, gauge_components = _responsibility_error_density_scores(
        gauge, target, gauge_render, cfg, support_fade_alpha=fade_alpha
    )
    labels, group_score, group_mass, group_error = group_responsibility_scores(
        gauge, groups, gauge_components, alpha=0.7
    )

    expected = base_score[groups] * row_fractions.pow(0.3)
    assert float((base_render - gauge_render).abs().max()) <= 2e-6
    assert torch.allclose(
        base_components["denominator"], gauge_components["denominator"], atol=2e-6, rtol=2e-6
    )
    assert torch.allclose(gauge_score, expected, atol=2e-6, rtol=2e-6)
    assert labels.tolist() == [0, 1, 2]
    assert torch.allclose(group_score, base_score, atol=2e-6, rtol=2e-6)
    assert torch.allclose(group_mass, base_components["mass"], atol=2e-5, rtol=2e-6)
    assert torch.allclose(group_error, base_components["error"], atol=2e-6, rtol=2e-6)


def test_alpha1_values_are_invariant_but_duplicate_rows_consume_topk_slots():
    field = _counterexample_field()
    target = torch.rand(12, 12, 3, generator=torch.Generator().manual_seed(9))
    cfg = FitConfig(renderer="normalized", responsibility_mass_alpha=1.0, render_chunk=1)
    base_render = _render(field, cfg, 12, 12)
    base_score, _ = _responsibility_error_density_scores(field, target, base_render, cfg)
    top_group = int(torch.argmax(base_score))
    gauge, groups, _ = exact_opacity_refinement(field, split_groups={top_group})
    gauge_render = _render(gauge, cfg, 12, 12)
    gauge_score, components = _responsibility_error_density_scores(gauge, target, gauge_render, cfg)
    labels, quotient, _, _ = group_responsibility_scores(gauge, groups, components, alpha=1.0)

    base_action = _row_action(base_score, torch.arange(3), 2)
    raw_action = _row_action(gauge_score, groups, 2)
    quotient_action = [int(labels[idx]) for idx in torch.argsort(quotient, descending=True)[:2]]
    assert torch.allclose(gauge_score, base_score[groups], atol=1e-6, rtol=1e-6)
    assert raw_action == [top_group, top_group]
    assert quotient_action == base_action


def test_projected_topk_agreement_records_raw_ticket_change_and_quotient_parity():
    field = _counterexample_field()
    target = torch.rand(12, 12, 3, generator=torch.Generator().manual_seed(9))
    cfg = FitConfig(renderer="normalized", render_chunk=1)
    base_render = _render(field, cfg, 12, 12)
    groups = torch.arange(field.n, dtype=torch.long)
    alpha1_cfg = FitConfig(renderer="normalized", render_chunk=1, responsibility_mass_alpha=1.0)
    alpha1_score, _ = _responsibility_error_density_scores(field, target, base_render, alpha1_cfg)
    top_group = int(torch.argmax(alpha1_score))
    gauge, gauge_groups, _ = exact_opacity_refinement(field, split_groups={top_group})
    gauge_render = _render(gauge, cfg, 12, 12)
    specs, diagnostics = _arm_specs(
        field,
        gauge,
        groups,
        gauge_groups,
        target,
        base_render,
        gauge_render,
        k=2,
        render_chunk=1,
    )
    by_arm = {spec["arm"]: spec for spec in specs}
    assert by_arm["responsibility_alpha1_gauge_row"]["projected_topk_exact"] is False
    assert by_arm["responsibility_alpha1_gauge_row"]["projected_topk_multiset_jaccard"] < 1.0
    assert by_arm["quotient_alpha1_gauge"]["projected_topk_exact"] is True
    assert diagnostics["quotient_alpha0.7_action_match"] is True
    assert diagnostics["quotient_alpha1_action_match"] is True


def test_algebra_audit_contains_both_deterministic_counterexamples():
    result = run_algebra_audit()
    assert result["alpha0.7"]["render_max_abs"] <= 2e-6
    assert result["alpha0.7"]["child_law_max_abs"] <= 2e-6
    assert result["alpha0.7"]["gauge_row_changed"] is True
    assert result["alpha1"]["gauge_unique_groups"] == 1
    assert result["positive_control"] == {
        "alpha0.7_changed": True,
        "alpha1_duplicate_ticket": True,
        "quotient_restored_both": True,
    }


def test_exact_refinement_preserves_optional_row_attributes_and_roles():
    field = _rich_field()
    gauge, groups, _ = exact_opacity_refinement(field, split_groups={0, 1})
    assert groups.tolist() == [0, 0, 1, 1, 2]
    for name in (
        "means",
        "log_scales",
        "rotations",
        "colors",
        "scale_max",
        "color_grads",
        "background_mask",
        "filter_variance",
    ):
        original = getattr(field, name)
        refined = getattr(gauge, name)
        assert original is not None and refined is not None
        assert torch.equal(refined[0], original[0])
        assert torch.equal(refined[1], original[0])
        assert torch.equal(refined[2], original[1])
        assert torch.equal(refined[3], original[1])
        assert torch.equal(refined[4], original[2])


def test_group_scores_accept_sparse_permuted_labels_and_zero_mass():
    field = GaussianField.from_numpy(
        means=np.asarray([[50.0, 50.0], [50.0, 50.0], [60.0, 60.0]], dtype=np.float32),
        scales=np.ones((3, 2), dtype=np.float32),
        angles=np.zeros(3, dtype=np.float32),
        colors=np.ones((3, 3), dtype=np.float32),
        opacities=_logits([0.4, 0.2, 0.5]),
    )
    groups = torch.tensor([9, 9, 3], dtype=torch.long)
    components = {"mass": torch.zeros(3), "error": torch.zeros(3)}
    labels, score, mass, error = group_responsibility_scores(field, groups, components, alpha=0.7)
    assert labels.tolist() == [3, 9]
    assert score.tolist() == [0.0, 0.0]
    assert mass.tolist() == [0.0, 0.0]
    assert error.tolist() == [0.0, 0.0]
    assert torch.isfinite(score).all()


@pytest.mark.parametrize(
    "fractions,match",
    [((0.4, 0.4), "sum to one"), ((0.0, 1.0), "positive"), ((math.nan,), "positive")],
)
def test_exact_refinement_rejects_invalid_fractions(fractions, match):
    with pytest.raises(ValueError, match=match):
        exact_opacity_refinement(_rich_field(), split_groups={0}, fractions=fractions)


def test_exact_refinement_requires_explicit_opacity_and_known_groups():
    field = _rich_field()
    field.opacities = None
    with pytest.raises(ValueError, match="explicit opacity"):
        exact_opacity_refinement(field, split_groups={0})
    with pytest.raises(ValueError, match="unknown canonical"):
        exact_opacity_refinement(_rich_field(), split_groups={99})


def test_group_validation_rejects_nonidentical_members_and_mixed_roles():
    field = _rich_field().subset(torch.tensor([0, 1]))
    components = {"mass": torch.ones(2), "error": torch.ones(2)}
    with pytest.raises(ValueError, match="non-identical means"):
        group_responsibility_scores(field, torch.tensor([0, 0]), components, alpha=1.0)

    identical = _rich_field().subset(torch.tensor([0, 0]))
    identical.background_mask = torch.tensor([False, True])
    with pytest.raises(ValueError, match="mixes background"):
        group_responsibility_scores(identical, torch.tensor([4, 4]), components, alpha=1.0)


def _decision_rows():
    rows = []
    for target in TARGET_NAMES:
        for seed in (0, 1):
            for arm in ARMS:
                is_quotient = arm == "quotient_alpha1_gauge"
                is_raw_gauge = arm == "responsibility_alpha1_gauge_row"
                if is_quotient:
                    post20, post100, total = 20.10, 22.00, 1.05
                elif is_raw_gauge:
                    post20, post100, total = 20.00, 22.00, 1.00
                elif arm == "support_canonical":
                    post20, post100, total = 20.00, 22.00, 1.00
                else:
                    post20, post100, total = 20.02, 22.00, 1.00
                if arm in ("responsibility_alpha1_gauge_row", "support_gauge_row"):
                    action = [0, 0, 2, 2, 4, 4, 6, 6]
                else:
                    action = list(range(8))
                rows.append(
                    {
                        "target": target,
                        "seed": seed,
                        "arm": arm,
                        "immediate_psnr": 19.0,
                        "post20_psnr": post20,
                        "post100_psnr": post100,
                        "post20_ms_ssim": 0.9,
                        "post100_ms_ssim": 0.95,
                        "post100_auc": 20.0,
                        "score_seconds": 0.01,
                        "total100_seconds": total,
                        "unique_action_groups": len(set(action)),
                        "action_groups": json.dumps(action),
                        "action_hash": __import__("hashlib")
                        .sha256(json.dumps(action, separators=(",", ":")).encode("ascii"))
                        .hexdigest(),
                        "render_max_abs": 1e-7,
                        "quotient_alpha1_max_abs": 1e-8,
                        "quotient_alpha1_max_rel": 1e-8,
                        "quotient_alpha0.7_max_abs": 1e-8,
                        "quotient_alpha0.7_max_rel": 1e-8,
                        "quotient_alpha1_action_match": True,
                        "quotient_alpha0.7_action_match": True,
                        "alpha1_child_law_max_abs": 1e-8,
                        "alpha0.7_child_law_max_abs": 1e-8,
                        "scores_finite": True,
                        "renders_finite": True,
                        "final_count": 40,
                        "post20_count": 40,
                        "post100_count": 40,
                    }
                )
    return rows


def test_decision_separates_commutation_and_recovery_utility():
    algebra = run_algebra_audit()
    aggregate = aggregate_and_decide(_decision_rows(), algebra)
    assert aggregate["commutation"]["confirmed"] is True
    assert aggregate["commutation"]["raw_alpha1_changed_target_count"] == 8
    assert aggregate["utility"]["survives"] is True

    failed_rows = copy.deepcopy(_decision_rows())
    for row in failed_rows:
        if row["arm"] == "quotient_alpha1_gauge":
            row["post20_psnr"] = 19.9
    failed = aggregate_and_decide(failed_rows, algebra)
    assert failed["commutation"]["confirmed"] is True
    assert failed["utility"]["survives"] is False


def test_decision_gates_both_alpha_actions_relative_error_and_stage_a_numbers():
    algebra = run_algebra_audit()

    bad_relative = copy.deepcopy(_decision_rows())
    bad_relative[0]["quotient_alpha0.7_max_rel"] = 3e-5
    result = aggregate_and_decide(bad_relative, algebra)
    assert result["commutation"]["checks"]["quotient_scores_both_alphas_within_2e-5"] is False

    bad_action = copy.deepcopy(_decision_rows())
    bad_action[0]["quotient_alpha0.7_action_match"] = False
    result = aggregate_and_decide(bad_action, algebra)
    assert result["commutation"]["checks"]["quotient_actions_both_alphas_match_16_of_16"] is False

    bad_algebra = copy.deepcopy(algebra)
    bad_algebra["alpha1"]["denominator_max_abs"] = 3e-6
    result = aggregate_and_decide(_decision_rows(), bad_algebra)
    assert result["commutation"]["checks"]["algebra_numeric_invariants"] is False


def test_raw_change_gate_uses_multisets_and_requires_both_seeds():
    rows = _decision_rows()
    canonical = [0, 0, 1, 2, 3, 4, 5, 6]
    gauge = [0, 1, 1, 2, 3, 4, 5, 6]
    assert set(canonical) == set(gauge)
    assert _multiset_jaccard(canonical, gauge) < 1.0
    for row in rows:
        if row["arm"] == "responsibility_alpha1_canonical":
            row["action_groups"] = json.dumps(canonical)
        elif row["arm"] == "responsibility_alpha1_gauge_row":
            row["action_groups"] = json.dumps(gauge)
    result = aggregate_and_decide(rows, run_algebra_audit())
    assert result["commutation"]["raw_alpha1_changed_target_count"] == 8
    assert all(
        value == 1.0 for value in result["commutation"]["raw_alpha1_set_jaccard_by_target"].values()
    )

    one_seed_only = _decision_rows()
    canonical_by_key = {
        (row["target"], row["seed"]): row["action_groups"]
        for row in one_seed_only
        if row["arm"] == "responsibility_alpha1_canonical"
    }
    for row in one_seed_only:
        if row["arm"] == "responsibility_alpha1_gauge_row" and row["seed"] == 0:
            row["action_groups"] = canonical_by_key[(row["target"], row["seed"])]
    result = aggregate_and_decide(one_seed_only, run_algebra_audit())
    assert result["commutation"]["raw_alpha1_changed_target_count"] == 0
    assert result["commutation"]["confirmed"] is False


def test_decision_checks_recovered_counts_and_rejects_malformed_cell_tables():
    rows = _decision_rows()
    rows[0]["post100_count"] = 39
    result = aggregate_and_decide(rows, run_algebra_audit())
    assert result["utility"]["checks"]["all_immediate_and_recovered_counts_eq_40"] is False
    assert result["utility"]["survives"] is False

    rows = _decision_rows()
    with pytest.raises(ValueError, match="exactly one row"):
        aggregate_and_decide(rows[:-1], run_algebra_audit())
    with pytest.raises(ValueError, match="exactly one row"):
        aggregate_and_decide([*rows, copy.deepcopy(rows[0])], run_algebra_audit())


def test_sequential_repeated_actions_choose_highest_opacity_with_stable_ties(monkeypatch):
    import benchmarks.gauge_equivalence_audit as audit

    field = _counterexample_field().subset(torch.tensor([0]))
    calls = []
    original = audit._moment_preserving_duplicate_indices

    def spy(current, idx, cfg, height, width):
        calls.append(int(idx.item()))
        return original(current, idx, cfg, height, width)

    monkeypatch.setattr(audit, "_moment_preserving_duplicate_indices", spy)
    target = torch.zeros(12, 12, 3)
    result = _sequential_moment_action(
        field,
        [0, 0, 0],
        target=target,
        render_img=target,
        cfg=FitConfig(renderer="normalized", split_scale=0.7),
    )
    assert calls == [0, 0, 1]
    assert result.n == 4


def test_recovery_records_immediate_and_both_horizon_counts():
    field = _counterexample_field()
    target = torch.rand(12, 12, 3, generator=torch.Generator().manual_seed(12))
    before = _render(field, FitConfig(renderer="normalized", render_chunk=1), 12, 12)
    result = _evaluate_action(field, target, before, [0], seed=0, render_chunk=1)
    assert result["final_count"] == 4
    assert result["post20_count"] == 4
    assert result["post100_count"] == 4


def test_target_suite_is_complete_finite_and_deterministic():
    left = _target_suite(48)
    right = _target_suite(48)
    assert tuple(left) == TARGET_NAMES
    for name in TARGET_NAMES:
        assert left[name].shape == (48, 48, 3)
        assert left[name].dtype == np.float32
        assert np.isfinite(left[name]).all()
        assert float(left[name].min()) >= 0.0
        assert float(left[name].max()) <= 1.0
        assert np.array_equal(left[name], right[name])


def test_frozen_run_rejects_protocol_mutation_before_writing(tmp_path):
    with pytest.raises(ValueError, match="requires seeds"):
        run(outdir=str(tmp_path / "seed"), seeds=(0,))
    with pytest.raises(ValueError, match="size/N/add/pre"):
        run(outdir=str(tmp_path / "shape"), size=32)
    with pytest.raises(ValueError, match="requires device"):
        run(outdir=str(tmp_path / "device"), device="cuda")
