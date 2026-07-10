from __future__ import annotations

import pytest

from benchmarks import storage_budget as S


def test_fixed_168_kib_float32_rs_budget_is_exact() -> None:
    assert S.FIXED_STORAGE_BUDGET_BYTES == 172_032
    assert S.CONSTANT_RGB_RS_SCALARS_PER_GAUSSIAN == 8
    assert S.CONSTANT_RGB_RS_BYTES_PER_GAUSSIAN == 32
    assert S.FIXED_STORAGE_GAUSSIAN_CAP == 5_376
    assert S.gaussian_cap_for_budget(S.FIXED_STORAGE_BUDGET_BYTES) == 5_376
    assert S.exact_gaussian_cap_for_budget(S.FIXED_STORAGE_BUDGET_BYTES) == 5_376
    assert S.representation_payload_bytes(5_376) == 172_032


@pytest.mark.parametrize("budget", [0, -1, 31])
def test_budget_must_fit_at_least_one_gaussian(budget: int) -> None:
    with pytest.raises(ValueError):
        S.gaussian_cap_for_budget(budget)


@pytest.mark.parametrize("budget", [True, 172_032.0, "172032"])
def test_budget_rejects_ambiguous_coercions(budget: object) -> None:
    with pytest.raises(TypeError):
        S.gaussian_cap_for_budget(budget)  # type: ignore[arg-type]


def test_layout_validation_rejects_invalid_or_inexact_budgets() -> None:
    with pytest.raises(ValueError, match="at least one Gaussian"):
        S.gaussian_cap_for_budget(32, fixed_overhead_bytes=32)
    with pytest.raises(ValueError, match="not exactly divisible"):
        S.exact_gaussian_cap_for_budget(33)
    with pytest.raises(ValueError):
        S.gaussian_cap_for_budget(172_032, bytes_per_gaussian=0)
    with pytest.raises(ValueError):
        S.gaussian_cap_for_budget(172_032, fixed_overhead_bytes=-1)


def test_row_accounting_records_exact_file_sizes_and_safe_names(tmp_path) -> None:
    source = tmp_path / "source.jpg"
    target = tmp_path / "target.png"
    reconstruction = tmp_path / "reconstruction.png"
    source.write_bytes(b"source")
    target.write_bytes(b"target-data")
    reconstruction.write_bytes(b"reconstruction-data")

    row = S.row_storage_accounting(
        5_376,
        source_path=source,
        target_path=target,
        reconstruction_path=reconstruction,
    )

    assert row["source_file_bytes"] == 6
    assert row["target_file_bytes"] == 11
    assert row["reconstruction_file_bytes"] == 19
    assert row["representation_payload_bytes"] == 172_032
    assert row["storage_budget_status"] == "exact"
    assert row["storage_exact_budget"] is True
    assert row["storage_unused_budget_bytes"] == 0
    assert row["storage_over_budget_bytes"] == 0
    assert "bytes" not in row
    assert "image_size_bytes" not in row


@pytest.mark.parametrize(
    ("count", "status", "exact", "unused", "over"),
    [
        (5_375, "underfilled", False, 32, 0),
        (5_376, "exact", True, 0, 0),
        (5_377, "overfilled", False, 0, 32),
    ],
)
def test_row_accounting_exposes_underfill_and_overfill(
    count: int,
    status: str,
    exact: bool,
    unused: int,
    over: int,
) -> None:
    row = S.row_storage_accounting(count)

    assert row["storage_actual_gaussians"] == count
    assert row["storage_budget_status"] == status
    assert row["storage_exact_budget"] is exact
    assert row["storage_unused_budget_bytes"] == unused
    assert row["storage_over_budget_bytes"] == over
    assert row["source_file_bytes"] is None
    assert row["target_file_bytes"] is None
    assert row["reconstruction_file_bytes"] is None


def test_row_accounting_fails_closed_for_requested_missing_artifact(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="source_path does not exist"):
        S.row_storage_accounting(5_376, source_path=tmp_path / "missing.png")


def test_representation_count_rejects_negative_and_boolean_values() -> None:
    with pytest.raises(ValueError):
        S.row_storage_accounting(-1)
    with pytest.raises(TypeError):
        S.row_storage_accounting(True)
