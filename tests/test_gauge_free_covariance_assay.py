import json
from pathlib import Path

import numpy as np
import pytest

from benchmarks import gauge_free_covariance_assay as A
from benchmarks import gauge_free_covariance_codec as C
from structsplat.gaussians import GaussianField


def _field() -> GaussianField:
    return GaussianField.from_numpy(
        means=np.array([[1.0, 2.0], [7.0, 4.0], [3.0, 9.0], [11.0, 6.0]]),
        scales=np.array([[2.0, 0.7], [1.0, 1.0], [0.8, 2.5], [3.0, 1.2]]),
        angles=np.array([-0.2, 7.0, 1.1, -4.2]),
        colors=np.array(
            [[0.1, 0.2, 0.3], [0.7, 0.4, 0.2], [0.3, 0.8, 0.5], [1.1, -0.1, 0.6]]
        ),
    )


def test_frozen_allocation_and_split_grid() -> None:
    allocations = A.allocations()
    assert len(allocations) == 84
    assert allocations == tuple(sorted(allocations, key=lambda value: (sum(value), *value)))
    assert A._split_ids("development") == tuple(range(2, 25, 2))
    assert A._split_ids("confirmation") == tuple(range(1, 24, 2))


def test_source_manifest_covers_the_complete_structsplat_source_tree() -> None:
    root = Path(__file__).resolve().parents[1]
    package_sources = {
        path.relative_to(root).as_posix()
        for path in (root / "src" / "structsplat").rglob("*")
        if path.suffix in {".py", ".cpp", ".cu"}
    }
    assert package_sources <= set(A.SOURCE_PATHS)
    assert {"benchmarks/__init__.py", "benchmarks/_util.py", "benchmarks/common.py"} <= set(
        A.SOURCE_PATHS
    )


def test_covariance_error_is_zero_for_an_exact_canonical_round_trip() -> None:
    field = _field()
    blob = C.encode(
        field,
        16,
        16,
        C.CodecConfig(chart="canonical_rs", geometry_bits=(10, 10, 4)),
    )
    decoded = C.decode(blob)
    error = A._covariance_error(field, decoded, 16, 16)
    assert error["relative_frobenius_rms"] >= 0.0
    assert error["relative_frobenius_max_row"] >= error["relative_frobenius_rms"] * 0.0


def test_unquantized_chart_controls_pass_on_cpu() -> None:
    controls = A._unquantized_controls(_field(), 16, 16, "cpu")
    assert controls["covariance_roundtrip_max_relative"] <= 1e-12
    assert controls["render_current_vs_canonical_max_abs"] <= 2e-5
    assert controls["render_current_vs_log_spd_max_abs"] <= 2e-5


def test_timed_codec_paths_are_deterministic() -> None:
    config = C.CodecConfig(
        chart="log_spd",
        geometry_bits=(4, 4, 4),
        predictor="modular_delta",
        coder="zlib9",
    )
    blob, encode_seconds, encode_samples = A._timed_encode(_field(), 16, 16, config)
    decoded, decode_seconds, decode_samples = A._timed_decode(blob)
    assert encode_seconds > 0.0
    assert decode_seconds > 0.0
    assert len(encode_samples) == A.TIMING_REPEATS
    assert len(decode_samples) == A.TIMING_REPEATS
    assert decoded.n == 4
    assert C.canonical_reencode(blob) == blob


def test_confirmation_authorization_rejects_a_flag_only_forgery(tmp_path: Path) -> None:
    assert A._confirmation_authorized(None) is False
    path = tmp_path / "analysis.json"
    path.write_text("{}", encoding="utf-8")
    assert A._confirmation_authorized(path) is False
    path.write_text(
        json.dumps(
            {
                "schema": "comp007-analysis-v2",
                "split": "development",
                "analysis": {
                    "decision": "pass_authorize_confirmation",
                    "confirmation_authorized": True,
                },
                "checks": {"artifact_audit_passed": True},
            }
        ),
        encoding="utf-8",
    )
    assert A._confirmation_authorized(path) is False


def test_field_hash_changes_with_covariance() -> None:
    first = _field()
    second = _field().detached()
    second.log_scales[0, 0] += 0.01
    assert A._field_sha256(first) != A._field_sha256(second)


def test_run_rejects_cpu_before_creating_artifacts(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="requires a CUDA"):
        A.run_split(tmp_path / "never-created", tmp_path, "development", "cpu", None)
    assert not (tmp_path / "never-created").exists()
