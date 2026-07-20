from __future__ import annotations

import math

import numpy as np
import torch

from benchmarks import nested_residual_extension_diagnostic as diagnostic


def _orthonormal(rows: int, columns: int, seed: int = 7) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    value = torch.randn((rows, columns), generator=generator, dtype=torch.float64)
    return torch.linalg.qr(value, mode="reduced").Q


def test_algebra_controls_pass() -> None:
    result = diagnostic.run_algebra_controls()
    assert result["status"] == "pass"
    assert all(result["checks"].values())


def test_factorization_seed_uses_exact_bench009_crossfit_unit_axes() -> None:
    for spec in diagnostic._unit_specs():
        expected = diagnostic._stable_id(
            {
                "target_id": spec["target_id"],
                "seed": spec["seed"],
                "parent_horizon": "near_plateau",
                "selection_scope": "crossfit",
                "heldout_fold": spec["heldout_fold"],
            }
        )
        assert spec["bench009_unit_id"] == expected
        assert spec["bench009_unit_id"] != spec["unit_id"]


def test_nested_delta_is_sign_and_permutation_invariant() -> None:
    generator = torch.Generator(device="cpu").manual_seed(11)
    q_j = _orthonormal(31, 5)
    columns = torch.randn((31, 6), generator=generator, dtype=torch.float64)
    residual = torch.randn(31, generator=generator, dtype=torch.float64)
    first = diagnostic.nested_extension_basis(q_j, columns, residual, parameter_count=19)
    permutation = torch.tensor([4, 0, 5, 2, 1, 3])
    signs = torch.tensor([-1.0, 1.0, -1.0, 1.0, 1.0, -1.0], dtype=torch.float64)
    second = diagnostic.nested_extension_basis(
        q_j,
        columns[:, permutation] * signs,
        residual,
        parameter_count=19,
    )
    assert first.extension_rank == second.extension_rank
    assert math.isclose(first.nested_delta, second.nested_delta, rel_tol=1e-12, abs_tol=1e-12)
    assert first.nested_delta >= 0.0


def test_extension_inside_fixed_base_has_zero_rank_and_delta() -> None:
    generator = torch.Generator(device="cpu").manual_seed(13)
    q_j = _orthonormal(27, 4)
    coefficients = torch.randn((4, 6), generator=generator, dtype=torch.float64)
    residual = torch.randn(27, generator=generator, dtype=torch.float64)
    result = diagnostic.nested_extension_basis(
        q_j, q_j @ coefficients, residual, parameter_count=17
    )
    assert result.extension_rank == 0
    assert result.nested_delta <= 32.0 * result.tolerance


def test_threshold_is_relative_to_original_unscaled_columns() -> None:
    columns = torch.zeros((12, 6), dtype=torch.float64)
    columns[0, 0] = 1.0
    columns[1, 1] = 1.1e-5
    columns[2, 2] = 0.9e-5
    result = diagnostic.nested_extension_basis(
        torch.zeros((12, 0), dtype=torch.float64),
        columns,
        torch.arange(12, dtype=torch.float64),
        parameter_count=6,
    )
    assert result.cutoff == 1e-5
    assert result.extension_rank == 2


def test_nested_physical_solve_and_joint_trust_are_finite() -> None:
    generator = torch.Generator(device="cpu").manual_seed(17)
    q_j = _orthonormal(33, 5)
    columns = torch.randn((33, 6), generator=generator, dtype=torch.float64)
    residual = torch.randn(33, generator=generator, dtype=torch.float64)
    nested = diagnostic.nested_extension_basis(q_j, columns, residual, parameter_count=21)
    base_design = q_j.T @ torch.randn((33, 21), generator=generator, dtype=torch.float64)
    solved = diagnostic.solve_nested_physical_system(
        nested, base_design, columns, residual, damping=1e-3
    )
    base, extension, factor = diagnostic.uniform_joint_linf_scale(
        solved.base_step, solved.extension_coefficients, 0.25
    )
    assert torch.isfinite(base).all()
    assert torch.isfinite(extension).all()
    assert 0.0 < factor <= 1.0
    assert float(torch.cat((base, extension)).abs().max()) <= 0.25 + 1e-12


def _analysis_rows(binding: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for unit_index, spec in enumerate(diagnostic._unit_specs()):
        for candidate in diagnostic.CANDIDATES:
            for radius in diagnostic.TRUST_RADII:
                predicted = 1e-4 + unit_index * 1e-6
                row_id = f"{candidate}-{radius}-{unit_index}"
                rows.append(
                    {
                        "row_id": row_id,
                        **spec,
                        "binding_sha256": binding,
                        "status": "ok",
                        "candidate": candidate,
                        "trust_radius": radius,
                        "base_mse": 0.1,
                        "predicted_mse_gain": predicted,
                        "realized_mse_gain": 1.1 * predicted,
                        "nested_delta": 0.01 + unit_index * 1e-4,
                        "extension_rank": 3,
                        "numerical_tolerance": float(np.finfo(np.float64).eps * 4096),
                        "base_orthogonality_max_abs": 1e-15,
                        "extension_orthogonality_max_abs": 1e-15,
                        "cross_orthogonality_max_abs": 1e-15,
                    }
                )
    return rows


def test_analysis_requires_and_accepts_all_four_calibration_strata() -> None:
    rows = _analysis_rows("binding")
    result = diagnostic.analyze_rows(rows, "binding")
    assert result["status"] == "available"
    assert result["row_count"] == 96
    assert len(result["calibration_strata"]) == 4
    assert result["all_frozen_calibration_strata_pass"] is True
    assert result["classification"] == "diagnostic_pass"
    assert diagnostic.analyze_rows(list(reversed(rows)), "binding") == result


def test_analysis_closes_without_retuning_when_one_stratum_reverses() -> None:
    rows = _analysis_rows("binding")
    selected = [
        row
        for row in rows
        if row["candidate"] == "carrier6" and row["trust_radius"] == 0.75
    ]
    realized = [float(row["realized_mse_gain"]) for row in reversed(selected)]
    for row, value in zip(selected, realized, strict=True):
        row["realized_mse_gain"] = value
    result = diagnostic.analyze_rows(rows, "binding")
    assert result["status"] == "available"
    assert result["all_frozen_calibration_strata_pass"] is False
    assert result["classification"] == "diagnostic_fail"
    assert result["no_rescue_rule"] == "close_without_retuning"
