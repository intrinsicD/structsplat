from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest


ARMS = (
    "interior",
    "posterior_no_reciprocal",
    "vggt_raw_known_ray",
    "vggt_coherent_wse",
)


def _artifact(root: Path, relative: str, payload: bytes | None = None) -> dict[str, object]:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload if payload is not None else relative.encode("utf-8"))
    return {
        "path": relative,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "bytes": path.stat().st_size,
    }


def _write_core019_report(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    command = "python core019_coherent_depth_downstream.py --out report"
    plan = {
        "schema": "core019.coherent_depth.plan.v1",
        "command": command,
        "repositories": {
            "structsplat": {
                "head": "a" * 40,
                "branch": "core/019-calibrated-coherent-depth-fusion",
                "dirty": True,
            }
        },
        "train_camera_ids": ["C0001", "C0002"],
        "report_camera_ids": ["C0003"],
        "vggt": {"checkpoint_bytes": 5_026_367_224, "checkpoint_sha256": "b" * 64},
    }
    plan_payload = (json.dumps(plan, sort_keys=True) + "\n").encode()
    plan_artifact = _artifact(root, "plan.json", plan_payload)
    shared = _artifact(root, "shared_packets.json", b"{}\n")
    feature = _artifact(root, "feature_receipt.json", b"{}\n")
    field = {
        "arrays": _artifact(root, "coherent_depth_field/field.npz"),
        "receipt": _artifact(root, "coherent_depth_field/receipt.json", b"{}\n"),
        "contact_sheet": _artifact(root, "coherent_depth_field/contact_sheet.png"),
    }

    rows = []
    records = []
    required_links = ["plan.json", field["contact_sheet"]["path"]]
    for arm in ARMS:
        rows.append(
            {
                "arm": arm,
                "status": "ok",
                "initial_reporting_psnr": 10.0,
                "reporting_psnr": 20.0,
                "reporting_ms_ssim": 0.9,
                "reporting_lpips": 0.2,
                "reporting_gradient_mae": 0.03,
                "pretraining_seconds": 1.0,
                "training_native_seconds": 2.0,
                "original_over_packets_plus_model": 1.1,
                "initial_n_gaussians": 10_000,
                "final_n_gaussians": 10_000,
                "final_model_bytes": 100,
            }
        )
        prefix = f"arms/{arm}"
        models = {
            key: _artifact(root, f"{prefix}/{key}.npz" if "npz" in key else f"{prefix}/{key}.ply")
            for key in ("initial_npz", "initial_ply", "final_npz", "final_ply")
        }
        views = {}
        for view_index in range(4):
            views[f"view_{view_index}"] = {
                name: _artifact(root, f"{prefix}/visuals/view_{view_index}_{name}.png")
                for name in ("target", "initial", "final", "error_x4", "alpha", "depth_support")
            }
        curves = _artifact(root, f"{prefix}/checkpoint_metrics.json", b"[]\n")
        history = _artifact(root, f"{prefix}/training_history.json", b"{}\n")
        contact = _artifact(root, f"{prefix}/visuals/reporting_contact_sheet.png")
        records.append(
            {
                "arm": arm,
                "status": "ok",
                "input": {"packet_hashes": ["c" * 64, "d" * 64]},
                "curves": curves,
                "history": history,
                "models": models,
                "visuals": {"contact_sheet": contact, "views": views},
            }
        )
        required_links.extend([curves["path"], models["final_npz"]["path"], contact["path"]])

    metrics_payload = (json.dumps(rows, indent=2, sort_keys=True) + "\n").encode()
    metrics_json = _artifact(root, "metrics.json", metrics_payload)
    metrics_jsonl = _artifact(
        root,
        "metrics.jsonl",
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows).encode(),
    )
    csv_path = root / "metrics.csv"
    fields = sorted({key for row in rows for key in row})
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    metrics_csv = {
        "path": "metrics.csv",
        "sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
        "bytes": csv_path.stat().st_size,
    }
    curves = _artifact(root, "all_metric_curves.png")
    _artifact(root, "decision.json", b"{}\n")
    links = [
        "manifest.json",
        "metrics.json",
        "metrics.jsonl",
        "metrics.csv",
        "decision.json",
        curves["path"],
        *required_links,
    ]
    index_payload = (
        "<!doctype html><title>CORE-019 fixture</title>"
        + "".join(f'<a href="{link}">{link}</a>' for link in links)
    ).encode()
    report = _artifact(root, "index.html", index_payload)
    manifest = {
        "schema": "core019.coherent_depth.manifest.v1",
        "status": "ok",
        "claim_ready": False,
        "command": command,
        "plan": plan_artifact,
        "shared_packets": shared,
        "feature_receipt": feature,
        "coherent_depth_field": field,
        "records": records,
        "metrics": {
            "metrics.json": metrics_json,
            "metrics.jsonl": metrics_jsonl,
            "metrics.csv": metrics_csv,
        },
        "plots": {"all_metric_curves": curves},
        "report": report,
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _load_driver():
    pytest.importorskip("rtgs")
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "experiments"
        / "core019_coherent_depth_downstream.py"
    )
    spec = importlib.util.spec_from_file_location("_core019_coherent_depth_driver", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _passing_rows(driver):
    common = {
        "status": "ok",
        "input_bytes": 100,
        "original_source_bytes": 1_000,
        "final_model_bytes": 200,
        "original_over_packets": 10.0,
        "original_over_packets_plus_model": 1_000 / 300,
        "initial_n_gaussians": driver.INITIAL_GAUSSIANS,
        "final_n_gaussians": 20_000,
        "reporting_ssim": 0.90,
        "reporting_p99_abs": 0.10,
        "lift_seconds": 2.0,
        "training_native_seconds": 10.0,
        "peak_vram_gb": 1.0,
    }
    return [
        {
            **common,
            "arm": "interior",
            "initial_reporting_psnr": 10.0,
            "initial_reporting_lpips": 0.40,
            "initial_reporting_gradient_mae": 0.04,
            "reporting_psnr": 20.0,
            "reporting_ms_ssim": 0.90,
            "reporting_lpips": 0.20,
            "reporting_gradient_mae": 0.03,
        },
        {
            **common,
            "arm": "posterior_no_reciprocal",
            "initial_reporting_psnr": 11.0,
            "initial_reporting_lpips": 0.39,
            "initial_reporting_gradient_mae": 0.039,
            "reporting_psnr": 19.8,
            "reporting_ms_ssim": 0.89,
            "reporting_lpips": 0.21,
            "reporting_gradient_mae": 0.031,
        },
        {
            **common,
            "arm": "vggt_raw_known_ray",
            "initial_reporting_psnr": 11.0,
            "initial_reporting_lpips": 0.38,
            "initial_reporting_gradient_mae": 0.038,
            "reporting_psnr": 19.9,
            "reporting_ms_ssim": 0.89,
            "reporting_lpips": 0.21,
            "reporting_gradient_mae": 0.031,
        },
        {
            **common,
            "arm": "vggt_coherent_wse",
            "initial_reporting_psnr": 12.1,
            "initial_reporting_lpips": 0.35,
            "initial_reporting_gradient_mae": 0.035,
            "reporting_psnr": 20.1,
            "reporting_ms_ssim": 0.91,
            "reporting_lpips": 0.19,
            "reporting_gradient_mae": 0.029,
        },
    ]


def _records(rows):
    values = {
        "interior": (10.0, 20.0, 20.0),
        "posterior_no_reciprocal": (11.0, 19.7, 19.8),
        "vggt_raw_known_ray": (11.0, 19.8, 19.9),
        "vggt_coherent_wse": (12.1, 20.1, 20.1),
    }
    records = []
    for row in rows:
        arm = row["arm"]
        curve_rows = []
        for step, psnr in zip((0, 500, 1_500), values[arm], strict=True):
            curve_rows.append(
                {
                    "step": step,
                    "optimization_elapsed_seconds": step / 100.0,
                    "metrics": {
                        "reporting": {
                            "aggregate": {
                                "psnr": psnr,
                                "ms_ssim": 0.91 if arm == "vggt_coherent_wse" else 0.90,
                                "lpips": 0.19 if arm == "vggt_coherent_wse" else 0.20,
                                "gradient_mae": 0.029 if arm == "vggt_coherent_wse" else 0.03,
                            }
                        }
                    },
                }
            )
        records.append(
            {
                "arm": arm,
                "status": "ok",
                "curve_rows": curve_rows,
                "input": {"packet_hashes": ["a" * 64]},
                "lift_diagnostics": {
                    "surface_cover": {
                        "spacing": {
                            "median": 1.0,
                            "p90": 1.5 if arm == "vggt_coherent_wse" else 2.0,
                        }
                    }
                },
            }
        )
    return records


def test_core019_protocol_is_frozen_disjoint_and_capacity_bounded() -> None:
    driver = _load_driver()

    assert driver.REPORT_CAMERA_IDS == ("C0024", "C0010", "C1004", "C0022")
    assert len(driver.TRAIN_CAMERA_IDS) == 26
    assert set(driver.TRAIN_CAMERA_IDS).isdisjoint(driver.REPORT_CAMERA_IDS)
    assert set(driver.TRAIN_CAMERA_IDS) | set(driver.REPORT_CAMERA_IDS) == set(
        driver.ALL_CAMERA_IDS
    )
    assert driver.ARMS == (
        "interior",
        "posterior_no_reciprocal",
        "vggt_raw_known_ray",
        "vggt_coherent_wse",
    )
    assert driver.INITIAL_GAUSSIANS == 10_000
    assert driver.MAX_GAUSSIANS == 30_000
    assert driver.ITERATIONS == 1_500
    assert driver.FIXED_PREFIX_STEPS == 500
    assert driver._interior_carve_config().candidate_multiplier == 4
    assert driver.BASE_CARVE_CONFIG().candidate_multiplier == 2

    coherent = driver._coherent_config()
    assert coherent.max_anchor_rounds == 12
    assert coherent.wse_anchor_fraction == 0.15
    assert coherent.contraction_max_cluster_size == 2
    assert coherent.surface_cover_max_pixel_sigma == 2.0
    train = driver.base._train_config()
    assert train.density.start_iter == 600
    assert train.density.max_gaussians == driver.MAX_GAUSSIANS


def test_core019_scalar_pass_still_requires_native_visual_review() -> None:
    driver = _load_driver()
    rows = _passing_rows(driver)

    decision = driver._decision(_records(rows), rows)

    assert decision["scalar_pass"] is True
    assert all(decision["gates"].values())
    assert decision["manual_visual_review_required"] is True
    assert decision["advance"] is False


def test_core019_decision_rejects_missing_step_zero_gain() -> None:
    driver = _load_driver()
    rows = _passing_rows(driver)
    candidate = next(row for row in rows if row["arm"] == "vggt_coherent_wse")
    candidate["initial_reporting_psnr"] = 11.9

    decision = driver._decision(_records(rows), rows)

    assert decision["scalar_pass"] is False
    assert decision["gates"]["step0_psnr_plus_2db_over_interior"] is False
    assert decision["advance"] is False


def test_core019_full_vs_raw_gate_counts_every_prespecified_quality_metric() -> None:
    driver = _load_driver()
    rows = _passing_rows(driver)
    raw = next(row for row in rows if row["arm"] == "vggt_raw_known_ray")
    candidate = next(row for row in rows if row["arm"] == "vggt_coherent_wse")
    candidate.update(
        {
            "reporting_psnr": raw["reporting_psnr"] - 0.1,
            "reporting_ssim": raw["reporting_ssim"] - 0.01,
            "reporting_lpips": raw["reporting_lpips"] + 0.01,
            "reporting_gradient_mae": raw["reporting_gradient_mae"] + 0.001,
            "reporting_p99_abs": raw["reporting_p99_abs"] + 0.01,
            "training_native_seconds": raw["training_native_seconds"] + 1.0,
        }
    )

    decision = driver._decision(_records(rows), rows)

    assert candidate["reporting_ms_ssim"] > raw["reporting_ms_ssim"]
    assert decision["gates"]["full_beats_raw_quality_or_convergence"] is True
    improvements = decision["full_vs_raw"]["quality_or_convergence_improvements"]
    assert improvements["reporting_ms_ssim"] is True
    assert sum(improvements.values()) == 1


def test_core019_report_checker_accepts_only_explicit_nonclaim_bundle(tmp_path: Path) -> None:
    from scripts.check_report_bundle import check_bundle

    _write_core019_report(tmp_path)
    assert check_bundle(tmp_path, allow_dirty=True) == []
    assert any("repository was dirty" in problem for problem in check_bundle(tmp_path))

    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["claim_ready"] = True
    manifest["records"][0]["input"]["packet_hashes"] = ["e" * 64]
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    problems = check_bundle(tmp_path, allow_dirty=True)
    assert "manifest.json CORE-019 claim_ready must remain false" in problems
    assert "CORE-019 arms did not reuse one identical packet set" in problems
