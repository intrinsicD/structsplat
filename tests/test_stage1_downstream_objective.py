from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

from benchmarks import stage1_downstream_objective as B


ROOT = Path(__file__).resolve().parent.parent


def _artifact(root: Path, name: str, payload: bytes | None = None) -> dict:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload if payload is not None else name.encode("utf-8"))
    return {
        "path": str(path.resolve()),
        "sha256": B.sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _semantics(family: str) -> dict:
    additive = family == "additive"
    return {
        "provider": "native" if additive else "structsplat",
        "equation": "additive_sum" if additive else "normalized_weighted_sum",
        "blend_mode": "additive" if additive else "normalized",
        "alpha_policy": "packed_alpha",
        "coordinate_convention": "top-left pixel center is (0.5,0.5)",
        "semantic_digest": hashlib.sha256(family.encode()).hexdigest(),
    }


def _frozen_protocol(tmp_path: Path) -> dict:
    shared = _artifact(tmp_path, "bindings/shared.json", b"{}\n")
    environments = [
        _artifact(tmp_path, "bindings/structsplat-env.json", b"{}\n"),
        _artifact(tmp_path, "bindings/realtime-env.json", b"{}\n"),
    ]
    families = ("additive", "normalized", "normalized_no_boundary")
    captures = []
    for capture_index in range(3):
        frame_id = f"frame_{capture_index}"
        frame_families = []
        for family in families:
            frame_families.append(
                {
                    "id": family,
                    "field_manifest": _artifact(
                        tmp_path,
                        f"bindings/{frame_id}_{family}_manifest.json",
                        json.dumps({"frame": frame_id, "family": family}).encode(),
                    ),
                    "stage1_metrics": _artifact(
                        tmp_path,
                        f"bindings/{frame_id}_{family}_stage1.json",
                        b"{}\n",
                    ),
                    "semantics": _semantics(family),
                }
            )
        captures.append(
            {
                "id": f"capture_{capture_index}",
                "frames": [
                    {
                        "id": frame_id,
                        "pixels": shared,
                        "masks": shared,
                        "cameras": shared,
                        "split": {
                            "train": ["C0001", "C0002"],
                            "heldout": ["C0003"],
                        },
                        "families": frame_families,
                    }
                ],
            }
        )
    protocol = {
        "schema": B.PROTOCOL_SCHEMA,
        "task_id": "BENCH-019",
        "state": "review",
        "driver": "driver-a",
        "claim_scope": "general",
        "repositories": [
            {
                "name": "structsplat",
                "root": "/source/structsplat",
                "commit": "a" * 40,
                "branch": "bench/019",
                "dirty": False,
                "status_sha256": hashlib.sha256(b"").hexdigest(),
                "environment": environments[0],
            },
            {
                "name": "realtime-gs",
                "root": "/source/realtime-gs",
                "commit": "b" * 40,
                "branch": "experiment/three-provider",
                "dirty": False,
                "status_sha256": hashlib.sha256(b"").hexdigest(),
                "environment": environments[1],
            },
        ],
        "captures": captures,
        "downstream": {
            "task_manifest": shared,
            "dataset_manifest": shared,
            "environment": environments[1],
            "schedule_config": shared,
            "command": ["python", "driver.py", "run"],
            "working_directory": str(tmp_path.resolve()),
            "outcome_root": str(tmp_path / "outcomes"),
            "seeds": [11, 12, 13],
            "initializers": ["fixed_initializer"],
            "result_schema": B.RESULT_SCHEMA,
        },
        "predictors": [
            {"name": "image_quality", "direction": "higher"},
            {"name": "query_error", "direction": "lower"},
        ],
        "responses": [
            {"name": "heldout_psnr", "direction": "higher", "primary": True},
            {"name": "fit_seconds", "direction": "lower", "primary": False},
        ],
        "analysis": {
            "bootstrap_replicates": 200,
            "bootstrap_seed": 19019,
            "minimum_capture_groups": 3,
            "minimum_frames": 2,
            "minimum_family_count": 3,
            "minimum_spearman": 0.8,
            "minimum_bootstrap_lower": 0.5,
            "minimum_lofo_top1_agreement": 1.0,
            "selection_priority": ["image_quality", "query_error"],
            "missing_policy": "fail_closed",
        },
        "aa_replay": {
            "frame_id": "frame_0",
            "family_id": "additive",
            "seed": 11,
            "initializer": "fixed_initializer",
            "primary_replicate": "primary",
            "replay_replicate": "aa",
            "metric_abs_tolerance": {
                "image_quality": 0.0,
                "query_error": 0.0,
                "heldout_psnr": 0.0,
            },
        },
    }
    protocol["design_sha256"] = B.design_digest(protocol)
    review = _artifact(tmp_path, "bindings/review.json", b"approved\n")
    protocol["review"] = {
        "driver": "driver-a",
        "reviewer": "reviewer-b",
        "verdict": "approved",
        "design_sha256": protocol["design_sha256"],
        "artifact": review,
    }
    protocol["state"] = "frozen"
    protocol["protocol_sha256"] = B.protocol_digest(protocol)
    B.validate_protocol(protocol, require_frozen=True)
    return protocol


def _rows(protocol: dict, tmp_path: Path, *, with_artifacts: bool = False) -> list[dict]:
    family_score = {
        "additive": 3.0,
        "normalized": 2.0,
        "normalized_no_boundary": 1.0,
    }
    frame_index = {
        frame["id"]: (capture["id"], frame)
        for capture in protocol["captures"]
        for frame in capture["frames"]
    }
    rows = []
    primary = protocol["aa_replay"]["primary_replicate"]
    for cell_index, key in enumerate(B.expected_cell_keys(protocol)):
        frame_id, family_id, seed, initializer, replicate = key
        capture_id, frame = frame_index[frame_id]
        family = next(value for value in frame["families"] if value["id"] == family_id)
        score = family_score[family_id]
        # The A/A replay receives exactly the primary cell's factor digest.
        factor_key = f"{frame_id}:{seed}:{initializer}:{primary}"
        factor_digest = hashlib.sha256(factor_key.encode()).hexdigest()
        artifacts = {}
        if with_artifacts:
            shared_artifact = _artifact(
                tmp_path,
                f"cell_sources/{cell_index}.bin",
                f"cell-{cell_index}".encode(),
            )
            artifacts = {name: shared_artifact for name in B.REQUIRED_CELL_ARTIFACTS}
        rows.append(
            {
                "schema": B.ROW_SCHEMA,
                "status": "ok",
                "error": "",
                "capture_id": capture_id,
                "frame_id": frame_id,
                "family_id": family_id,
                "seed": seed,
                "initializer": initializer,
                "replicate_id": replicate,
                "field_manifest_sha256": family["field_manifest"]["sha256"],
                "field_semantic_digest": family["semantics"]["semantic_digest"],
                "downstream_factor_digest": factor_digest,
                "stage1": {
                    "image_quality": 30.0 + score,
                    "query_error": 5.0,
                },
                "downstream": {
                    "heldout_psnr": 20.0 + score + 0.001 * (seed - 11),
                    "fit_seconds": 100.0 - score,
                },
                "artifacts": artifacts,
            }
        )
    # Exact A/A means even the seed-dependent metric matches the primary row byte-for-value.
    aa_key = B.expected_cell_keys(protocol)[-1]
    primary_key = (*aa_key[:4], primary)
    primary_row = next(row for row in rows if B._row_key(row) == primary_key)  # noqa: SLF001
    aa_row = next(row for row in rows if B._row_key(row) == aa_key)  # noqa: SLF001
    aa_row["stage1"] = copy.deepcopy(primary_row["stage1"])
    aa_row["downstream"] = copy.deepcopy(primary_row["downstream"])
    return rows


def _load_report_checker():
    path = ROOT / "scripts/check_report_bundle.py"
    spec = importlib.util.spec_from_file_location("_bench019_report_checker", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _draft_from_frozen(protocol: dict, repositories: list[Path]) -> dict:
    draft = copy.deepcopy(protocol)
    draft["state"] = "draft"
    draft.pop("design_sha256")
    draft.pop("protocol_sha256")
    draft.pop("review")
    draft["repositories"] = [
        {
            "name": record["name"],
            "root": str(root),
            "environment": {"path": record["environment"]["path"]},
        }
        for record, root in zip(protocol["repositories"], repositories)
    ]
    for capture in draft["captures"]:
        for frame in capture["frames"]:
            for name in ("pixels", "masks", "cameras"):
                frame[name] = {"path": frame[name]["path"]}
            for family in frame["families"]:
                for name in ("field_manifest", "stage1_metrics"):
                    family[name] = {"path": family[name]["path"]}
    for name in ("task_manifest", "dataset_manifest", "environment", "schedule_config"):
        draft["downstream"][name] = {"path": draft["downstream"][name]["path"]}
    return draft


def _clean_repository(path: Path) -> None:
    path.mkdir()
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    subprocess.run(["git", "-C", str(path), "config", "user.name", "Fixture"], check=True)
    subprocess.run(
        ["git", "-C", str(path), "config", "user.email", "fixture@example.invalid"],
        check=True,
    )
    (path / "tracked.txt").write_text("clean\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(path), "add", "tracked.txt"], check=True)
    subprocess.run(["git", "-C", str(path), "commit", "-qm", "fixture"], check=True)


def test_protocol_rejects_semantic_relabelling_and_split_overlap(tmp_path: Path) -> None:
    protocol = _frozen_protocol(tmp_path)
    relabelled = copy.deepcopy(protocol)
    relabelled["captures"][0]["frames"][0]["families"][1]["semantics"][
        "equation"
    ] = "additive_sum"
    relabelled["design_sha256"] = B.design_digest(relabelled)
    relabelled["review"]["design_sha256"] = relabelled["design_sha256"]
    relabelled["protocol_sha256"] = B.protocol_digest(relabelled)
    with pytest.raises(B.ProtocolError, match="equation and blend_mode disagree"):
        B.validate_protocol(relabelled, require_frozen=True)

    overlapping = copy.deepcopy(protocol)
    overlapping["captures"][0]["frames"][0]["split"]["heldout"] = ["C0001"]
    overlapping["design_sha256"] = B.design_digest(overlapping)
    overlapping["review"]["design_sha256"] = overlapping["design_sha256"]
    overlapping["protocol_sha256"] = B.protocol_digest(overlapping)
    with pytest.raises(B.ProtocolError, match="disjoint non-empty splits"):
        B.validate_protocol(overlapping, require_frozen=True)

    underscoped = copy.deepcopy(protocol)
    underscoped["analysis"]["minimum_capture_groups"] = 1
    underscoped["design_sha256"] = B.design_digest(underscoped)
    underscoped["review"]["design_sha256"] = underscoped["design_sha256"]
    underscoped["protocol_sha256"] = B.protocol_digest(underscoped)
    with pytest.raises(B.ProtocolError, match="general claim requires at least three"):
        B.validate_protocol(underscoped, require_frozen=True)


def test_protocol_lifecycle_requires_clean_sources_and_distinct_review(tmp_path: Path) -> None:
    source_protocol = _frozen_protocol(tmp_path)
    repositories = [tmp_path / "repo-a", tmp_path / "repo-b"]
    for repository in repositories:
        _clean_repository(repository)
    draft = _draft_from_frozen(source_protocol, repositories)
    reviewed = B.prepare_review(draft, base=tmp_path)
    assert reviewed["state"] == "review"
    B.validate_protocol(reviewed)

    review = B.review_template(reviewed)
    review.update(
        {
            "reviewer": "reviewer-b",
            "verdict": "approved",
            "outcome_accessed": False,
        }
    )
    review_path = tmp_path / "prospective-review.json"
    review_path.write_text(json.dumps(review), encoding="utf-8")
    frozen = B.finalize_protocol(reviewed, review_path)
    B.validate_protocol(frozen, require_frozen=True)
    assert frozen["review"]["reviewer"] == "reviewer-b"

    (repositories[0] / "untracked.txt").write_text("dirty\n", encoding="utf-8")
    with pytest.raises(B.ProtocolError, match="is dirty"):
        B.prepare_review(draft, base=tmp_path)


def test_analysis_selects_only_the_preregistered_valid_surrogate(tmp_path: Path) -> None:
    protocol = _frozen_protocol(tmp_path)
    rows = _rows(protocol, tmp_path)
    assert B.validate_result_rows(protocol, rows) == []
    result = B.analyze(protocol, rows)
    assert result["aa_replay"]["passed"] is True
    assert result["scope"]["scope_gate_passed"] is True
    assert result["decision"]["state"] == "select_stage1_surrogate"
    assert result["decision"]["selected_predictor"] == "image_quality"
    image = next(
        row for row in result["correlations"] if row["predictor"] == "image_quality"
    )
    assert image["pooled_within_frame_spearman"] == pytest.approx(1.0)
    assert image["cluster_bootstrap"]["low"] == pytest.approx(1.0)
    assert image["lofo_top1_agreement"] == pytest.approx(1.0)


def test_missing_factor_parity_and_aa_drift_fail_closed(tmp_path: Path) -> None:
    protocol = _frozen_protocol(tmp_path)
    rows = _rows(protocol, tmp_path)
    rows[1]["downstream_factor_digest"] = "f" * 64
    problems = B.validate_result_rows(protocol, rows)
    assert any("downstream config changes across field families" in problem for problem in problems)

    rows = _rows(protocol, tmp_path)
    rows[-1]["downstream"]["heldout_psnr"] += 0.01
    result = B.analyze(protocol, rows)
    assert result["aa_replay"]["passed"] is False
    assert result["decision"]["state"] == "question_unavailable"


def test_portable_report_passes_gate_and_detects_artifact_tamper(tmp_path: Path) -> None:
    protocol = _frozen_protocol(tmp_path)
    rows = _rows(protocol, tmp_path, with_artifacts=True)
    report = tmp_path / "report"
    manifest = B.write_report(
        protocol,
        rows,
        report,
        command="python -m benchmarks.stage1_downstream_objective analyze",
    )
    assert manifest["claim_ready"] is True
    checker = _load_report_checker()
    assert checker.check_bundle(report) == []

    artifact = next((report / "artifacts").rglob("field.bin"))
    artifact.write_bytes(b"tampered")
    problems = checker.check_bundle(report)
    assert any("artifact byte count differs" in problem for problem in problems)
    assert any("artifact SHA-256 differs" in problem for problem in problems)
