from __future__ import annotations

from argparse import Namespace
import csv
import hashlib
import json

import numpy as np
import pytest

from scripts.experiments import hier017_normalization_epsilon as hier017
from scripts.experiments import hier019_confidence_tail as hier019
from scripts.experiments import hier020_sparse_pixel_safe_tail as hier020
from scripts.experiments import hier021_source_patch_tail as hier021
from scripts.check_report_bundle import check_bundle


def _args(*, phase: str = "development") -> Namespace:
    return Namespace(
        phase=phase,
        target_gaussians=7000,
        max_side=512,
        seed=0,
        direct_fit_steps=750,
        device="cuda",
        additive_renderer="cuda_additive",
        direct_renderer="cuda",
        render_chunk=256,
        lpips=True,
        error_scale=4.0,
        development_decision=None,
        control_metrics=None,
    )


def _row(
    image: str,
    arm: str,
    *,
    mse: float,
    psnr: float,
    ms_ssim: float,
    lpips: float,
    pixel_max: float,
    patch_max: float,
    epsilon: float | None = None,
    improvement: float = 0.0,
) -> dict[str, object]:
    row: dict[str, object] = {
        "image": image,
        "arm": arm,
        "masked_mse": mse,
        "psnr_db": psnr,
        "ms_ssim": ms_ssim,
        "lpips": lpips,
        "artifact_pixel_rmse_max": pixel_max,
        "artifact_patch_rmse_max_7": patch_max,
        "n_gaussians": 7000,
        "maintained_render_parity_max_abs": 1e-6,
        "repeated_render_parity_max_abs": 1e-6,
    }
    if epsilon is not None:
        row.update(
            {
                "normalization_eps": epsilon,
                "fit_normalization_eps": epsilon,
                "render_normalization_eps": epsilon,
                "initial_field_content_sha256": f"init-{image}",
                "epsilon_sensitive_display_error_improvement_max": improvement,
            }
        )
    return row


def _development_rows(*, regress: bool = False) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index in range(4):
        image = f"coco_{index}"
        rows.append(
            _row(
                image,
                hier017.CONTROL_ARM,
                mse=1.0,
                psnr=20.0,
                ms_ssim=0.70,
                lpips=0.60,
                pixel_max=0.80,
                patch_max=0.40,
            )
        )
        rows.append(
            _row(
                image,
                hier017.EPS8_ARM,
                mse=0.10,
                psnr=30.0,
                ms_ssim=0.950,
                lpips=0.100,
                pixel_max=0.50,
                patch_max=0.20,
                epsilon=hier017.EPS_BASELINE,
            )
        )
        rows.append(
            _row(
                image,
                hier017.DECODE_ARM,
                mse=0.099,
                psnr=30.04,
                ms_ssim=0.950,
                lpips=0.100,
                pixel_max=0.48,
                patch_max=0.19,
                epsilon=hier017.EPS_CANDIDATE,
            )
        )
        candidate_mse = 0.11 if regress and index == 0 else 0.09
        rows.append(
            _row(
                image,
                hier017.EPS12_ARM,
                mse=candidate_mse,
                psnr=30.45,
                ms_ssim=0.951,
                lpips=0.090,
                pixel_max=0.45,
                patch_max=0.18,
                epsilon=hier017.EPS_CANDIDATE,
                improvement=0.01 if index == 0 else 0.0,
            )
        )
    return rows


def test_prospective_bindings_match_frozen_salted_filename_hashes() -> None:
    assert len(hier017.DEVELOPMENT_BINDINGS) == 4
    assert tuple(hier017.DEVELOPMENT_BINDINGS) == tuple(hier017.SELECTION_DIGESTS)
    for name, digest in hier017.SELECTION_DIGESTS.items():
        assert hashlib.sha256(("HIER-017-v1:" + name).encode()).hexdigest() == digest


def test_direct_configs_change_only_normalization_epsilon() -> None:
    init8, fit8 = hier017._direct_configs(_args(), hier017.EPS_BASELINE)
    init12, fit12 = hier017._direct_configs(_args(), hier017.EPS_CANDIDATE)

    assert init8 == init12
    fit8_record = vars(fit8).copy()
    fit12_record = vars(fit12).copy()
    assert fit8_record.pop("normalization_eps") == hier017.EPS_BASELINE
    assert fit12_record.pop("normalization_eps") == hier017.EPS_CANDIDATE
    assert fit8_record == fit12_record

    attribution = hier017._pseudo_decode_result(np.zeros((2, 2, 3), dtype=np.float32))
    assert attribution["normalization_eps"] == hier017.EPS_BASELINE
    assert attribution["history"]["render_epsilon"] == hier017.EPS_CANDIDATE


def test_development_gate_advances_only_the_consistently_fitted_candidate() -> None:
    attempts = [{"status": "ok"} for _ in range(16)]
    decision = hier017._development_decision(_development_rows(), attempts)

    assert all(decision["gates"].values())
    assert decision["numeric_candidates"] == [hier017.EPS12_ARM]
    assert hier017.DECODE_ARM not in decision["numeric_candidates"]


def test_one_raw_mse_regression_rejects_candidate() -> None:
    decision = hier017._development_decision(_development_rows(regress=True), [])

    assert not decision["gates"]["all_mse_noninferior_vs_eps8"]
    assert decision["numeric_candidates"] == []


def test_consumed_tests_replay_requires_every_pair_noninferior() -> None:
    rows = [
        row
        for row in _development_rows()
        if row["arm"] in (hier017.EPS8_ARM, hier017.EPS12_ARM)
    ] * 4
    decision = hier017._replay_decision(rows, [], _args(phase="replay_tests"))

    assert decision["clauses"]["complete_pairs"]
    assert decision["bounded_bank_pass"]


@pytest.mark.parametrize(
    "schema",
    [
        hier017.REPORT_SCHEMA,
        hier019.REPORT_SCHEMA,
        hier020.REPORT_SCHEMA,
        hier021.REPORT_SCHEMA,
    ],
)
def test_report_checker_accepts_hier015_plus_diagnostic_contract(tmp_path, schema) -> None:
    artifact_dir = tmp_path / "artifacts" / "toy__direct_eps1e8__n7000"
    artifact_dir.mkdir(parents=True)
    snapshot = tmp_path / "source_snapshot" / "driver.py"
    snapshot.parent.mkdir(parents=True)
    snapshot.write_text("# frozen\n", encoding="utf-8")
    for name in (
        "source.png",
        "reconstruction.png",
        "error.png",
        "source_crop.png",
        "reconstruction_crop.png",
        "error_crop.png",
        "analysis.npz",
        "field.gaussian.npz",
    ):
        (artifact_dir / name).write_bytes(b"artifact")
    for name in (
        "projection_history.json",
        "geometry_history.json",
        "fit_history.json",
    ):
        (artifact_dir / name).write_text("{}\n", encoding="utf-8")

    field_path = artifact_dir / "field.gaussian.npz"
    row = {
        "schema": schema,
        "status": "diagnostic",
        "image": "toy",
        "arm": hier017.EPS8_ARM,
        "target_gaussians": 7000,
        "n_gaussians": 7000,
        "artifact_dir": str(artifact_dir.relative_to(tmp_path)),
        "field_file_sha256": hashlib.sha256(field_path.read_bytes()).hexdigest(),
        "masked_mse": 0.01,
        "psnr_db": 20.0,
        "ms_ssim": 0.8,
        "lpips": 0.2,
        "maintained_render_parity_max_abs": 1e-6,
        "repeated_render_parity_max_abs": 1e-6,
    }
    (artifact_dir / "row.json").write_text(
        json.dumps(row, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    config = {
        "schema": schema,
        "status": "diagnostic",
        "command": "python driver.py",
        "git": {"revision": "a" * 40, "branch": "main", "dirty": True},
        "source_snapshots": [
            {
                "snapshot_path": str(snapshot.relative_to(tmp_path)),
                "sha256": hashlib.sha256(snapshot.read_bytes()).hexdigest(),
                "bytes": snapshot.stat().st_size,
            }
        ],
    }
    (tmp_path / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    metrics = {"schema": schema, "status": "diagnostic", "rows": [row]}
    (tmp_path / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (tmp_path / "metrics.jsonl").write_text(
        json.dumps(row, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (tmp_path / "metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(row))
        writer.writeheader()
        writer.writerow(row)
    (tmp_path / "attempts.json").write_text(
        json.dumps(
            {"schema": schema, "status": "diagnostic", "attempts": [{"status": "ok"}]},
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "decision.json").write_text(
        json.dumps({"schema": schema, "status": "diagnostic"}, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    links = "".join(
        f"<a href='{path}'>x</a>"
        for path in (
            "manifest.json",
            "metrics.json",
            "metrics.jsonl",
            "metrics.csv",
            str((artifact_dir / "source.png").relative_to(tmp_path)),
            str((artifact_dir / "reconstruction.png").relative_to(tmp_path)),
            str((artifact_dir / "error.png").relative_to(tmp_path)),
            str((artifact_dir / "reconstruction_crop.png").relative_to(tmp_path)),
        )
    )
    (tmp_path / "index.html").write_text(f"<html><body>1{links}</body></html>", encoding="utf-8")
    files = []
    for path in sorted(tmp_path.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            files.append(
                {
                    "path": str(path.relative_to(tmp_path)),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "bytes": path.stat().st_size,
                }
            )
    (tmp_path / "manifest.json").write_text(
        json.dumps({"schema": schema, "status": "diagnostic", "files": files}, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )

    assert check_bundle(tmp_path) == []
