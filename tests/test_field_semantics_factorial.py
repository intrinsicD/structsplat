from __future__ import annotations

import copy
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from benchmarks import field_semantics_factorial as bench


def _git(root: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=root, check=True, capture_output=True, text=True)


def _write(path: Path, value: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")
    return path


def _artifact(path: Path) -> dict[str, str]:
    return {"path": str(path)}


def _make_draft(tmp_path: Path) -> tuple[dict, Path]:
    repository = tmp_path / "repository"
    repository.mkdir(parents=True)
    _git(repository, "init")
    _git(repository, "config", "user.email", "bench020@example.invalid")
    _git(repository, "config", "user.name", "BENCH-020 test")
    draft = bench.protocol_template()
    draft["driver"] = "driver-agent"
    draft["repositories"][0]["root"] = str(repository)

    counter = 0

    def bound(label: str) -> dict[str, str]:
        nonlocal counter
        counter += 1
        safe = "".join(ch if ch.isalnum() else "_" for ch in label)
        return _artifact(
            _write(repository / "inputs" / f"{counter:03d}_{safe}.json", f"{label}:{counter}\n")
        )

    draft["repositories"][0]["environment"] = bound("repository environment")
    for split in ("development", "confirmation"):
        for unit in draft["datasets"][split]:
            for name in ("pixels", "mask", "camera", "prepared_target"):
                unit[name] = bound(f"{split} {unit['id']} {name}")
    for entry in draft["initial_geometry"]:
        entry["bank"] = bound(f"geometry {entry['unit_id']} {entry['seed']}")
    for arm in draft["arms"]:
        for name in ("profile", "loss_contract", "gate_contract"):
            arm[name] = bound(f"{arm['id']} {name}")
    for phase in bench.PHASES:
        draft["phases"][phase]["work_contract"] = bound(f"{phase} work")
    draft["execution"]["environment"] = bound("execution environment")
    draft["execution"]["downstream_protocol"] = bound("downstream protocol")
    draft["execution"]["working_directory"] = str(repository)
    for phase in bench.PHASES:
        draft["execution"]["outcome_roots"][phase] = str(tmp_path / "outcomes" / phase)

    _git(repository, "add", ".")
    _git(repository, "commit", "-m", "fixture")
    return draft, repository


def _freeze(tmp_path: Path) -> tuple[dict, Path]:
    draft, repository = _make_draft(tmp_path)
    review_ready = bench.prepare_review(draft, base=tmp_path)
    review = bench.review_template(review_ready, base=tmp_path)
    review.update(
        {
            "reviewer": "distinct-protocol-reviewer",
            "verdict": "approved",
            "outcomes_accessed": False,
        }
    )
    review_path = tmp_path / "protocol-review.json"
    review_path.write_text(json.dumps(review), encoding="utf-8")
    return bench.finalize_protocol(review_ready, review_path, base=tmp_path), repository


def _sealed(path: Path) -> dict[str, object]:
    return {
        "path": str(path),
        "sha256": bench.sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _metric_values(protocol: dict, cell: dict) -> dict[str, float]:
    names = {
        spec["name"]
        for spec in protocol["metrics"]
        if spec["availability"] == "required" and cell["phase"] in spec["phases"]
    }
    values = {
        "foreground_psnr": 35.0,
        "boundary_psnr": 34.0,
        "ms_ssim": 0.95,
        "lpips": 0.05,
        "alpha_mae": 0.001,
        "outside_rgb_mae": 0.001,
        "downstream_response": 0.7,
    }
    if cell["phase"] == "coefficient_screen":
        if cell["coefficient_variant"] == "counted_dc_signed_bounded":
            values["foreground_psnr"] = 34.0
            values["lpips"] = 0.06
        else:
            values["foreground_psnr"] = 35.0
            values["lpips"] = 0.05
    else:
        family = cell["family"]
        alpha = cell["alpha_policy"]
        if family == "direct_additive" and alpha == "alpha_gated":
            values.update(foreground_psnr=36.0, lpips=0.04, downstream_response=0.80)
        elif family == "direct_additive":
            values.update(foreground_psnr=35.8, lpips=0.05, downstream_response=0.70)
        elif family == "incumbent_factorized_additive" and alpha == "alpha_gated":
            values.update(foreground_psnr=35.8, lpips=0.045, downstream_response=0.70)
        elif family == "incumbent_factorized_additive":
            values.update(foreground_psnr=35.7, lpips=0.055, downstream_response=0.65)
        elif family == "normalized_plain" and alpha == "alpha_gated":
            values.update(foreground_psnr=35.7, lpips=0.05, downstream_response=0.68)
        elif family == "normalized_plain":
            values.update(foreground_psnr=35.65, lpips=0.058, downstream_response=0.64)
        else:
            values.update(foreground_psnr=36.2, lpips=0.039, downstream_response=0.81)
    return {name: float(values[name]) for name in names}


def _make_rows(
    protocol: dict,
    phase: str,
    *,
    domain_lock: dict | None = None,
    confirmation_lock: dict | None = None,
) -> list[dict]:
    cells = bench.expected_cells(
        protocol,
        phase,
        domain_lock=domain_lock,
        confirmation_lock=confirmation_lock,
    )
    rows = []
    for cell in cells:
        directory = Path(protocol["execution"]["outcome_roots"][phase]) / cell["cell_id"]
        field_payload = _write(directory / "field.npz", cell["identity_sha256"])
        raw = _write(directory / "raw.bin", cell["identity_sha256"])
        evaluated = _write(directory / "evaluated.bin", cell["identity_sha256"])
        metrics = _metric_values(protocol, cell)
        telemetry = {
            "row_count": cell["row_count"],
            "canonical_raw_bytes": cell["canonical_raw_bytes"],
            "iterations_requested": cell["requested_work"]["iterations"],
            "iterations_executed": cell["requested_work"]["iterations"],
            "renderer_calls_requested": cell["requested_work"]["renderer_call_cap"],
            "renderer_calls_executed": cell["requested_work"]["renderer_call_cap"] - 1,
            "wall_seconds": 10.0,
            "peak_memory_mb": 100.0,
            "checkpoint_id": "terminal",
            "authoritative_preclamp_sha256": bench.sha256_file(raw),
            "evaluation_clip_policy": "clip_0_1_for_metrics_only",
        }
        history_points = [
            {"iteration": 0, "wall_seconds": 0.0, "value": 20.0},
            {
                "iteration": telemetry["iterations_executed"],
                "wall_seconds": telemetry["wall_seconds"],
                "value": metrics[protocol["convergence"]["metric"]],
            },
        ]
        telemetry.update(bench.summarize_convergence(protocol, phase, history_points))
        arm = next(arm for arm in protocol["arms"] if arm["id"] == cell["arm_id"])
        field_manifest_value = {
            "schema": bench.FIELD_MANIFEST_SCHEMA,
            "cell_id": cell["cell_id"],
            "identity_sha256": cell["identity_sha256"],
            "semantic_sha256": cell["bindings"]["semantic_sha256"],
            "renderer_equation": arm["semantics"]["renderer_equation"],
            "coefficient_variant": cell["coefficient_variant"],
            "alpha_policy": cell["alpha_policy"],
            "row_count": cell["row_count"],
            "canonical_raw_bytes": cell["canonical_raw_bytes"],
            "authoritative_preclamp_sha256": telemetry["authoritative_preclamp_sha256"],
            "payload_format": arm["semantics"]["payload_format"],
            "payload_sha256": bench.sha256_file(field_payload),
            "payload_bytes": field_payload.stat().st_size,
        }
        field_manifest = _write(
            directory / "field_manifest.json",
            json.dumps(field_manifest_value, sort_keys=True),
        )
        metrics_json = _write(
            directory / "metrics.json",
            json.dumps(
                {
                    "schema": bench.METRICS_SCHEMA,
                    "cell_id": cell["cell_id"],
                    "metrics": metrics,
                    "telemetry_sha256": hashlib.sha256(bench.canonical_json(telemetry)).hexdigest(),
                },
                sort_keys=True,
            ),
        )
        history_json = _write(
            directory / "history.json",
            json.dumps(
                {
                    "schema": bench.HISTORY_SCHEMA,
                    "cell_id": cell["cell_id"],
                    "metric": protocol["convergence"]["metric"],
                    "points": history_points,
                },
                sort_keys=True,
            ),
        )
        rows.append(
            {
                "schema": bench.ROW_SCHEMA,
                "cell": cell,
                "status": "ok",
                "error": None,
                "telemetry": telemetry,
                "metrics": metrics,
                "artifacts": {
                    "field_manifest": _sealed(field_manifest),
                    "field_payload": _sealed(field_payload),
                    "metrics_json": _sealed(metrics_json),
                    "history_json": _sealed(history_json),
                    "raw_render": _sealed(raw),
                    "evaluated_render": _sealed(evaluated),
                },
            }
        )
    return rows


def _development_run(tmp_path: Path):
    protocol, _repository = _freeze(tmp_path)
    result_root = tmp_path
    coefficient_rows = _make_rows(protocol, "coefficient_screen")
    domain_lock = bench.analyze_coefficient_screen(protocol, coefficient_rows, base=result_root)
    development_rows = _make_rows(protocol, "development", domain_lock=domain_lock)
    development = bench.analyze_factorial(
        protocol,
        development_rows,
        "development",
        base=result_root,
        domain_lock=domain_lock,
    )
    return {
        "protocol": protocol,
        "root": result_root,
        "coefficient_rows": coefficient_rows,
        "domain_lock": domain_lock,
        "development_rows": development_rows,
        "development": development,
    }


def _complete_run(tmp_path: Path):
    run = _development_run(tmp_path)
    protocol = run["protocol"]
    result_root = run["root"]
    domain_lock = run["domain_lock"]
    development = run["development"]
    review = bench.development_review_template(protocol, development)
    review.update(reviewer="distinct-results-reviewer", verdict="approved")
    review_path = tmp_path / "development-review.json"
    review_path.write_text(json.dumps(review), encoding="utf-8")
    confirmation_lock = bench.lock_confirmation(protocol, development, review_path, base=tmp_path)
    confirmation_rows = _make_rows(
        protocol,
        "confirmation",
        domain_lock=domain_lock,
        confirmation_lock=confirmation_lock,
    )
    confirmation = bench.analyze_factorial(
        protocol,
        confirmation_rows,
        "confirmation",
        base=result_root,
        domain_lock=domain_lock,
        confirmation_lock=confirmation_lock,
    )
    return {
        **run,
        "confirmation_lock": confirmation_lock,
        "confirmation_rows": confirmation_rows,
        "confirmation": confirmation,
    }


def test_protocol_lifecycle_requires_clean_distinct_outcome_unseen_review(tmp_path):
    draft, repository = _make_draft(tmp_path)
    review_ready = bench.prepare_review(draft, base=tmp_path)
    assert review_ready["state"] == "review"
    assert len(review_ready["design_sha256"]) == 64
    review = bench.review_template(review_ready, base=tmp_path)
    review.update(reviewer="driver-agent", verdict="approved", outcomes_accessed=False)
    review_path = tmp_path / "self-review.json"
    review_path.write_text(json.dumps(review), encoding="utf-8")
    with pytest.raises(bench.ProtocolError, match="distinct approval"):
        bench.finalize_protocol(review_ready, review_path, base=tmp_path)

    _write(repository / "dirty.txt", "dirty")
    with pytest.raises(bench.ProtocolError, match="dirty"):
        bench.prepare_review(draft, base=tmp_path)


def test_protocol_rejects_split_overlap_semantic_relabel_and_mixed_policy(tmp_path):
    draft, _repository = _make_draft(tmp_path)
    draft["datasets"]["confirmation"][0]["pixels"] = copy.deepcopy(
        draft["datasets"]["development"][0]["pixels"]
    )
    with pytest.raises(bench.ProtocolError, match="hash-disjoint"):
        bench.prepare_review(draft, base=tmp_path)

    draft, _repository = _make_draft(tmp_path / "semantic")
    arm = next(arm for arm in draft["arms"] if arm["family"] == "normalized_plain")
    arm["semantics"]["renderer_equation"] = "additive_rgb_peak_one_v1"
    arm["semantic_sha256"] = hashlib.sha256(bench.canonical_json(arm["semantics"])).hexdigest()
    with pytest.raises(bench.ProtocolError, match="semantic relabelling"):
        bench.prepare_review(draft, base=tmp_path / "semantic")

    draft, _repository = _make_draft(tmp_path / "scope")
    arm = next(arm for arm in draft["arms"] if arm["id"] == "direct_additive")
    arm["policy_contracts"]["alpha_gated"]["gate_scope"] = "full_frame"
    with pytest.raises(bench.ProtocolError, match="loss/gate/profile scopes"):
        bench.prepare_review(draft, base=tmp_path / "scope")


def test_dual_arm_requires_validated_independent_structural_target(tmp_path):
    draft, _repository = _make_draft(tmp_path)
    direct = next(arm for arm in draft["arms"] if arm["family"] == "direct_additive")
    dual = copy.deepcopy(direct)
    dual["id"] = "dual_additive"
    dual["family"] = "dual_additive"
    dual["semantics"]["structural_mass"] = "independently_supervised"
    dual["semantic_sha256"] = hashlib.sha256(bench.canonical_json(dual["semantics"])).hexdigest()
    for policy_ledgers in dual["raw_byte_ledgers"].values():
        for ledger in policy_ledgers.values():
            mass = next(
                component
                for component in ledger["components"]
                if component["name"] == "structural_mass"
            )
            mass["bytes_per_row"] = 4
            ledger["bytes_per_row"] += 4
    draft["arms"].append(dual)
    draft["gates"]["killing"]["candidate_families"].append("dual_additive")
    with pytest.raises(bench.ProtocolError, match="validated structural target"):
        bench.prepare_review(draft, base=tmp_path)


def test_raw_byte_ledger_counts_dc_alpha_and_equal_lane_exactly(tmp_path):
    protocol, _repository = _freeze(tmp_path)
    arm = next(arm for arm in protocol["arms"] if arm["id"] == "direct_additive")
    alpha = arm["raw_byte_ledgers"]["counted_dc_signed_bounded"]["alpha_gated"]
    contained = arm["raw_byte_ledgers"]["zero_dc_nonnegative"]["hard_contained"]
    unit = protocol["datasets"]["development"][0]["id"]
    assert alpha["fixed_bytes_by_unit"][unit] - contained["fixed_bytes_by_unit"][unit] == 524

    plans = bench.expected_cells(protocol, "coefficient_screen")
    counted = next(
        cell for cell in plans if cell["coefficient_variant"] == "counted_dc_signed_bounded"
    )
    assert counted["canonical_raw_bytes"] == (
        alpha["fixed_bytes_by_unit"][unit] + counted["row_count"] * alpha["bytes_per_row"]
    )


def test_protocol_rejects_incomplete_geometry_and_semantic_byte_ledger(tmp_path):
    draft, _repository = _make_draft(tmp_path / "prefix")
    entry = draft["initial_geometry"][0]
    entry["prefixes"] = [prefix for prefix in entry["prefixes"] if prefix["row_count"] != 1024]
    with pytest.raises(bench.ProtocolError, match="lacks planned prefix"):
        bench.prepare_review(draft, base=tmp_path / "prefix")

    draft, _repository = _make_draft(tmp_path / "ledger")
    direct = next(arm for arm in draft["arms"] if arm["family"] == "direct_additive")
    ledger = direct["raw_byte_ledgers"]["zero_dc_nonnegative"]["alpha_gated"]
    mass = next(
        component for component in ledger["components"] if component["name"] == "structural_mass"
    )
    mass["bytes_per_row"] = 4
    ledger["bytes_per_row"] += 4
    with pytest.raises(bench.ProtocolError, match="non-dual arm"):
        bench.prepare_review(draft, base=tmp_path / "ledger")


def test_general_claim_requires_independent_confirmation_capture_groups(tmp_path):
    draft, _repository = _make_draft(tmp_path)
    for unit in draft["datasets"]["confirmation"]:
        unit["capture_group"] = "one_confirmation_capture"
    with pytest.raises(bench.ProtocolError, match="minimum capture groups"):
        bench.prepare_review(draft, base=tmp_path)

    draft, _repository = _make_draft(tmp_path / "convergence")
    draft["convergence"]["auc_horizon"] = "observed_only"
    with pytest.raises(bench.ProtocolError, match="auc_horizon"):
        bench.prepare_review(draft, base=tmp_path / "convergence")


def test_coefficient_screen_aa_and_frozen_advancement(tmp_path):
    protocol, _repository = _freeze(tmp_path)
    root = tmp_path
    rows = _make_rows(protocol, "coefficient_screen")
    status = bench.validate_result_rows(protocol, rows, "coefficient_screen", base=root)
    assert status["complete"]
    assert bench.aa_replay_result(protocol, rows)["pass"]
    lock = bench.analyze_coefficient_screen(protocol, rows, base=root)
    assert lock["decision"] == "advance"
    assert lock["finalists"] == ["zero_dc_nonnegative"]

    tampered = copy.deepcopy(rows)
    replay = next(row for row in tampered if row["cell"]["replicate"] == "aa")
    replay["metrics"]["foreground_psnr"] += 0.01
    assert not bench.aa_replay_result(protocol, tampered)["pass"]


def test_rows_fail_closed_on_missing_error_factor_and_manifest_drift(tmp_path):
    protocol, _repository = _freeze(tmp_path)
    root = tmp_path
    rows = _make_rows(protocol, "coefficient_screen")
    with pytest.raises(bench.ProtocolError, match="missing"):
        bench.validate_result_rows(protocol, rows[:-1], "coefficient_screen", base=root)

    error_rows = copy.deepcopy(rows)
    error_rows[0].update(
        status="error", error="intentional", telemetry={}, metrics={}, artifacts={}
    )
    status = bench.validate_result_rows(
        protocol,
        error_rows,
        "coefficient_screen",
        base=root,
        require_complete=False,
    )
    assert status["errors"]

    drift = copy.deepcopy(rows)
    drift[0]["telemetry"]["canonical_raw_bytes"] += 1
    with pytest.raises(bench.ProtocolError, match="canonical_raw_bytes"):
        bench.validate_result_rows(protocol, drift, "coefficient_screen", base=root)

    history_drift = copy.deepcopy(rows)
    history_path = Path(history_drift[0]["artifacts"]["history_json"]["path"])
    original_history = history_path.read_text()
    history_value = json.loads(original_history)
    history_value["points"][-1]["value"] += 1.0
    history_path.write_text(json.dumps(history_value), encoding="utf-8")
    history_drift[0]["artifacts"]["history_json"] = _sealed(history_path)
    with pytest.raises(bench.ProtocolError, match="history replay"):
        bench.validate_result_rows(
            protocol,
            history_drift,
            "coefficient_screen",
            base=root,
        )
    history_path.write_text(original_history, encoding="utf-8")

    manifest_drift = copy.deepcopy(rows)
    path = Path(manifest_drift[0]["artifacts"]["field_manifest"]["path"])
    value = json.loads(path.read_text())
    value["renderer_equation"] = "normalized_weighted_sum_v1"
    path.write_text(json.dumps(value), encoding="utf-8")
    manifest_drift[0]["artifacts"]["field_manifest"] = _sealed(path)
    with pytest.raises(bench.ProtocolError, match="field_manifest"):
        bench.validate_result_rows(protocol, manifest_drift, "coefficient_screen", base=root)


def test_rows_and_analysis_enforce_frozen_phase_outcome_roots(tmp_path):
    protocol, _repository = _freeze(tmp_path)
    rows = _make_rows(protocol, "coefficient_screen")
    escaped = copy.deepcopy(rows)
    raw = _write(tmp_path / "outside-phase.bin", escaped[0]["cell"]["identity_sha256"])
    escaped[0]["artifacts"]["raw_render"] = _sealed(raw)
    with pytest.raises(bench.ProtocolError, match="frozen coefficient_screen outcome root"):
        bench.validate_result_rows(
            protocol,
            escaped,
            "coefficient_screen",
            base=tmp_path,
        )

    development_root = Path(protocol["execution"]["outcome_roots"]["development"])
    _write(development_root / "premature.txt", "development result leaked early")
    with pytest.raises(bench.ProtocolError, match="non-empty before coefficient_screen"):
        bench.analyze_coefficient_screen(protocol, rows, base=tmp_path)


def test_development_gate_selects_one_nondominated_semantic_without_score(tmp_path):
    run = _complete_run(tmp_path)
    development = run["development"]
    assert development["status"]["complete"]
    assert development["matched_input_invariants"]["pass"]
    assert development["decision"] == "advance_one"
    assert development["selected"] == {
        "arm_id": "direct_additive",
        "coefficient_variant": "zero_dc_nonnegative",
        "alpha_policy": "alpha_gated",
    }
    assert len(development["nondominated"]) == 1


def test_heterogeneous_tradeoff_does_not_pick_with_hidden_tiebreak(tmp_path):
    protocol, _repository = _freeze(tmp_path)
    root = tmp_path
    coefficient = _make_rows(protocol, "coefficient_screen")
    lock = bench.analyze_coefficient_screen(protocol, coefficient, base=root)
    rows = _make_rows(protocol, "development", domain_lock=lock)
    for row in rows:
        if (
            row["cell"]["family"] == "direct_additive"
            and row["cell"]["alpha_policy"] == "hard_contained"
        ):
            row["metrics"]["foreground_psnr"] = 36.1
            row["metrics"]["lpips"] = 0.06
            metrics_path = Path(row["artifacts"]["metrics_json"]["path"])
            value = json.loads(metrics_path.read_text())
            value["metrics"] = row["metrics"]
            metrics_path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
            row["artifacts"]["metrics_json"] = _sealed(metrics_path)
    analysis = bench.analyze_factorial(protocol, rows, "development", base=root, domain_lock=lock)
    assert analysis["decision"] == "heterogeneous_tradeoff"
    assert analysis["selected"] is None


def test_confirmation_lock_requires_distinct_review_and_empty_sealed_root(tmp_path):
    run = _development_run(tmp_path)
    protocol = run["protocol"]
    development = run["development"]
    review = bench.development_review_template(protocol, development)
    review.update(reviewer=protocol["driver"], verdict="approved")
    review_path = tmp_path / "bad-review.json"
    review_path.write_text(json.dumps(review), encoding="utf-8")
    with pytest.raises(bench.ProtocolError, match="distinct approval"):
        bench.lock_confirmation(protocol, development, review_path, base=tmp_path)

    review["reviewer"] = "different-reviewer"
    review_path.write_text(json.dumps(review), encoding="utf-8")
    confirmation_root = Path(protocol["execution"]["outcome_roots"]["confirmation"])
    _write(confirmation_root / "leaked.txt", "outcome already visible")
    with pytest.raises(bench.ProtocolError, match="locking is too late"):
        bench.lock_confirmation(protocol, development, review_path, base=tmp_path)


def test_sealed_confirmation_and_audited_portable_report(tmp_path):
    run = _complete_run(tmp_path)
    assert run["confirmation"]["decision"] == "confirm_one"
    audit = bench.results_audit_template(run["protocol"], run["confirmation"])
    audit.update(reviewer="distinct-final-auditor", verdict="approved")
    audit_path = tmp_path / "results-audit.json"
    audit_path.write_text(json.dumps(audit), encoding="utf-8")
    report = tmp_path / "report"
    manifest = bench.write_report(
        run["protocol"],
        run["coefficient_rows"],
        run["domain_lock"],
        run["development_rows"],
        run["development"],
        report,
        row_base=run["root"],
        confirmation_lock=run["confirmation_lock"],
        confirmation_rows=run["confirmation_rows"],
        confirmation_analysis=run["confirmation"],
        results_audit_path=audit_path,
        command="bench020 surrogate",
    )
    assert manifest["claim_ready"]
    assert manifest["decision"]["outcome"] == "direct_additive"
    assert bench.validate_report_bundle(report) == []
    checker = subprocess.run(
        [sys.executable, "scripts/check_report_bundle.py", str(report)],
        cwd=Path(__file__).resolve().parents[1],
        check=False,
        capture_output=True,
        text=True,
    )
    assert checker.returncode == 0, checker.stdout + checker.stderr

    rows_path = report / manifest["files"]["development_json"]["path"]
    rows = json.loads(rows_path.read_text())
    rows[0]["metrics"]["foreground_psnr"] += 1.0
    rows_path.write_text(json.dumps(rows), encoding="utf-8")
    problems = bench.validate_report_bundle(report)
    assert any("hash/size mismatch" in problem or "row digest" in problem for problem in problems)


def test_report_csv_projects_every_row(tmp_path):
    run = _complete_run(tmp_path)
    report = tmp_path / "diagnostic-report"
    manifest = bench.write_report(
        run["protocol"],
        run["coefficient_rows"],
        run["domain_lock"],
        run["development_rows"],
        run["development"],
        report,
        row_base=run["root"],
        command="diagnostic",
    )
    assert not manifest["claim_ready"]
    csv_path = report / manifest["files"]["development_csv"]["path"]
    with csv_path.open(newline="", encoding="utf-8") as handle:
        projected = list(csv.DictReader(handle))
    assert len(projected) == len(run["development_rows"])
    assert {row["status"] for row in projected} == {"ok"}

    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames
        tampered = list(reader)
    tampered[0]["status"] = "error"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(tampered)
    manifest_path = report / "manifest.json"
    manifest_value = json.loads(manifest_path.read_text())
    manifest_value["files"]["development_csv"] = _sealed(csv_path)
    manifest_value["files"]["development_csv"]["path"] = csv_path.relative_to(report).as_posix()
    manifest_path.write_text(json.dumps(manifest_value), encoding="utf-8")
    assert any("development CSV" in problem for problem in bench.validate_report_bundle(report))
