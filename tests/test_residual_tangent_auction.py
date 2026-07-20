# ruff: noqa: E402
import json

import pytest

torch = pytest.importorskip("torch")

from benchmarks.residual_tangent_auction import (
    DIAGNOSTIC_PRICE_SCOPE,
    FAMILIES,
    _residualize,
    _scaled_svd,
    _solve_svd,
    build_reference,
    finite_difference_jvp_error,
    identifiability_targets,
    main,
    run_identifiability_suite,
)


def test_scaled_svd_is_column_scale_invariant_and_removes_gauge_nullity():
    generator = torch.Generator().manual_seed(7)
    first = torch.randn(40, generator=generator, dtype=torch.float64)
    second = torch.randn(40, generator=generator, dtype=torch.float64)
    design = torch.stack((first, second, first + 2.0 * second, 3.0 * first), dim=1)
    rescaled = design * torch.tensor([0.01, 100.0, 4.0, 0.2], dtype=torch.float64)
    system = _scaled_svd(design, rcond=1e-10)
    rescaled_system = _scaled_svd(rescaled, rcond=1e-10)

    assert system.rank == 2
    assert rescaled_system.rank == 2
    assert torch.allclose(
        system.u @ system.u.T,
        rescaled_system.u @ rescaled_system.u.T,
        atol=2e-12,
        rtol=2e-12,
    )

    rhs = torch.randn(40, generator=generator, dtype=torch.float64)
    _coefficients, fitted = _solve_svd(system, rhs)
    _scaled_coefficients, rescaled_fitted = _solve_svd(rescaled_system, rhs)
    assert torch.allclose(fitted, rescaled_fitted, atol=2e-12, rtol=2e-12)
    residual = _residualize(system, design)
    assert float((system.u.T @ residual).abs().max()) <= 2e-12


def test_current_tangent_jvp_matches_finite_difference_and_uses_image_quotient():
    reference = build_reference()
    error = finite_difference_jvp_error(reference, seed=3)

    assert error["relative_l2"] <= 1e-6
    assert error["max_abs"] <= 1e-8
    assert 0 < reference.base_system.rank < reference.base_system.raw_columns
    for packet in reference.packets.values():
        residualized = _residualize(reference.base_system, packet.columns)
        assert float((reference.base_system.u.T @ residualized).abs().max()) <= 2e-12


def test_affine_carrier_and_finite_birth_packets_are_exact_on_frozen_geometry():
    reference = build_reference()
    coefficients = {
        "affine": torch.tensor(
            [0.07, -0.03, 0.02, -0.05, 0.04, 0.06], dtype=torch.float64
        ),
        "carrier": torch.tensor(
            [0.09, -0.06, 0.03, -0.04, 0.08, 0.05], dtype=torch.float64
        ),
        "finite_birth": torch.tensor([0.82, 0.16, 0.61], dtype=torch.float64),
    }
    for name, packet in reference.packets.items():
        predicted = (
            reference.base_render.reshape(-1)
            + packet.offset
            + packet.columns @ coefficients[name]
        ).reshape_as(reference.base_render)
        realized = packet.realize(coefficients[name])
        assert torch.allclose(predicted, realized, atol=3e-13, rtol=3e-13)

    birth = reference.packets["finite_birth"]
    assert float(birth.offset.abs().max()) > 1e-4
    assert birth.optimized_dofs == 3
    assert birth.fixed_dofs == 6
    assert birth.packet_dofs == 9
    assert "finite-opacity" in birth.description


def test_identifiability_controls_separate_tangent_and_extension_families():
    result = run_identifiability_suite()
    targets = result["by_target"]

    assert targets["null"]["extension_winner"] == "none"
    assert targets["current_tangent"]["extension_winner"] == "none"
    for target_kind in ("affine", "carrier", "finite_birth"):
        assert targets[target_kind]["extension_winner"] == target_kind

    current = targets["current_tangent"]
    assert current["base_predicted_mse"] <= current["initial_mse"] * 1e-3
    assert current["base_realized_mse"] <= current["initial_mse"] * 1e-3
    assert current["base_linearization_max_abs"] <= 2e-4

    for target_kind in ("affine", "carrier", "finite_birth"):
        target = targets[target_kind]
        winner = target["families"][target_kind]
        competitors = [
            target["families"][name]["incremental_mse_reduction_vs_base"]
            for name in FAMILIES
            if name != target_kind
        ]
        assert winner["incremental_mse_reduction_vs_base"] > max(competitors)
        assert winner["predicted_mse"] <= target["initial_mse"] * 1e-6
        assert winner["packet_realization_max_abs"] <= 3e-13


def test_stage0_is_deterministic_and_carries_no_rate_claim():
    first = run_identifiability_suite(seed=11)
    second = run_identifiability_suite(seed=11)

    assert first["base_field_sha256"] == second["base_field_sha256"]
    assert first["target_sha256"] == second["target_sha256"]
    assert first["rows"] == second["rows"]
    assert first["price_scope"] == DIAGNOSTIC_PRICE_SCOPE
    assert first["compression_claim_authorized"] is False
    assert first["method_promotion_authorized"] is False
    assert first["stage0_gate"]["pass"] is True
    assert first["stage1_instrument_implementation_authorized"] is True
    assert first["stage1_development_run_authorized"] is False
    assert all(check["pass"] for check in first["stage0_gate"]["checks"].values())
    assert first["base_tangent"]["nullity"] > 0
    assert "image-space" in first["base_tangent"]["gauge_treatment"]
    for row in first["rows"]:
        assert row["price_scope"] == DIAGNOSTIC_PRICE_SCOPE
        assert row["packet_dofs"] >= row["optimized_dofs"]


def test_cli_writes_bounded_stage0_artifacts(tmp_path):
    outdir = tmp_path / "auction"
    assert main(["--outdir", str(outdir), "--seed", "5"]) == 0

    assert {path.name for path in outdir.iterdir()} == {
        "config.json",
        "identifiability.json",
        "rows.csv",
        "summary.md",
    }
    config_text = (outdir / "config.json").read_text()
    aggregate_text = (outdir / "identifiability.json").read_text()
    assert "Infinity" not in config_text + aggregate_text
    assert "NaN" not in config_text + aggregate_text
    config = json.loads(config_text)
    aggregate = json.loads(aggregate_text)
    summary = (outdir / "summary.md").read_text()
    assert config["compression_claim_authorized"] is False
    assert aggregate["price_scope"] == DIAGNOSTIC_PRICE_SCOPE
    assert aggregate["stage0_gate"]["pass"] is True
    assert aggregate["method_promotion_authorized"] is False
    assert "not bytes, BPP, or compression evidence" in summary
    assert "self-consistency gate only" in summary
    assert "self-generated" in summary
    assert "dirty-tree" in summary
    assert "finite_birth" in summary
    provenance = config["provenance"]
    assert len(provenance["git_commit"]) == 40
    assert isinstance(provenance["git_dirty"], bool)
    assert len(provenance["tracked_diff_sha256"]) == 64
    assert provenance["critical_source_hash_mode"].endswith("independent_of_git_index")
    assert set(provenance["critical_sources"]) == {
        "benchmarks/residual_tangent_auction.py",
        "src/structsplat/gaussians.py",
        "src/structsplat/render.py",
    }
    assert all(
        len(entry["sha256"]) == 64
        for entry in provenance["critical_sources"].values()
    )
    assert provenance["untracked_file_count"] == len(provenance["untracked_files"])


def test_identifiability_target_hashes_cover_every_declared_control():
    reference = build_reference()
    targets = identifiability_targets(reference)
    assert set(targets) == {
        "null",
        "current_tangent",
        "affine",
        "carrier",
        "finite_birth",
    }
    for target in targets.values():
        assert tuple(target.shape) == (reference.height, reference.width, 3)
        assert bool(torch.isfinite(target).all())
        assert float(target.min()) >= 0.0
        assert float(target.max()) <= 1.0
