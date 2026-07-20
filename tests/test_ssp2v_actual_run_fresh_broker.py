"""Focused, payload-free checks for captured replay fresh-root brokering."""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from typing import Any

import pytest

from benchmarks import ssp2v_actual_coder as actual
from benchmarks import ssp2v_actual_run as run


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _stream_record(name: str) -> dict[str, Any]:
    return {
        "blob_bytes": len(name.encode("utf-8")),
        "blob_sha256": _digest(f"{name}:blob"),
        "symbol_sha256": _digest(f"{name}:symbols"),
        "boundary_state_sha256": _digest(f"{name}:boundary"),
    }


def _quality_plan_authorities() -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    q_name = actual.PRIMARY_ORDER[0]
    variant = actual.VARIANT_ORDER[0]
    candidate_cells: list[dict[str, Any]] = []
    original_cells: list[dict[str, Any]] = []
    for image_id in actual.FROZEN_IMAGES:
        for bits in actual.FROZEN_TUPLES:
            identity = {
                "image_id": image_id,
                "bit_tuple": list(bits),
            }
            candidate_cells.append(
                {
                    **identity,
                    "relative_path": f"candidates/{image_id}/{'-'.join(map(str, bits))}",
                    "execution": {"selected_q": {"name": q_name}},
                }
            )
            original_cells.append(
                {
                    **identity,
                    "q_name": q_name,
                    "selection": {
                        "state": "selected",
                        "selected_label": variant,
                        "ordered_labels": [variant],
                    },
                }
            )
    phase_a_sha256 = "1" * 64
    context = {
        "phase_a": {"replay_phase_a_sha256": phase_a_sha256},
        "candidate_replay": {"candidate_set_sha256": "2" * 64},
        "replayed_candidate": {
            "state": "complete_candidate_set",
            "cells": candidate_cells,
        },
    }
    targets = {
        "target_copy_sha256": "3" * 64,
        "targets": [
            {
                "image_id": image_id,
                "rgb_copy": {
                    "relative_path": f"targets/{image_id}/target.rgb",
                    "sha256": _digest(f"{image_id}:target"),
                },
                "rgb_dimensions_hw": [8, 9],
            }
            for image_id in actual.FROZEN_IMAGES
        ],
    }
    original_quality = {
        "quality_stage_sha256": "4" * 64,
        "cells": original_cells,
    }
    preflight = {
        "native": {
            "arithmetic": {
                "path": "arithmetic.so",
                "bytes": 7,
                "sha256": "5" * 64,
            }
        },
        "renderer": {
            "relative_path": "runtime/renderer/renderer.so",
            "bytes": 11,
            "sha256": "6" * 64,
        },
        "renderer_proof": {
            "same_worker_metric_repeat": {"lpips_state_sha256": "7" * 64}
        },
    }
    return context, targets, original_quality, preflight


def test_quality_plan_body_is_deterministic_and_performs_no_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context, targets, original_quality, preflight = _quality_plan_authorities()
    inputs_before = copy.deepcopy((context, targets, original_quality, preflight))
    variant = actual.VARIANT_ORDER[0]

    monkeypatch.setattr(
        run,
        "source_manifest",
        lambda: {"source_manifest_sha256": "8" * 64},
    )
    monkeypatch.setattr(
        run,
        "_decision_relevant_variant_prefix",
        lambda _original, _candidate: [variant],
    )
    monkeypatch.setattr(
        run,
        "_decoded_stream_record",
        lambda _execution, name: _stream_record(name),
    )

    def forbidden_execution(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("pure quality planning attempted worker execution")

    monkeypatch.setattr(run, "_run_render_metric_task", forbidden_execution)
    monkeypatch.setattr(run, "_run_json_worker", forbidden_execution)

    first = run._replay_quality_plan_body(  # noqa: SLF001
        context=context,
        targets=targets,
        original_quality=original_quality,
        preflight_record=preflight,
    )
    second = run._replay_quality_plan_body(  # noqa: SLF001
        context=context,
        targets=targets,
        original_quality=original_quality,
        preflight_record=preflight,
    )

    assert first == second
    assert (context, targets, original_quality, preflight) == inputs_before
    assert first["scientific_reduction_deferred"] is True
    assert first["broker_executes_only_declared_tasks"] is True
    assert len(first["cells"]) == 16
    assert first["task_count"] == 16 * 3 * (1 + 1)
    task_ids = [
        task["task_id"] for cell in first["cells"] for task in cell["tasks"]
    ]
    assert len(task_ids) == len(set(task_ids)) == first["task_count"]
    assert first["cells"][0]["schedule"] == [
        ["primary", variant],
        [variant, "primary"],
        ["primary", variant],
    ]


def _resource_row(state: str) -> dict[str, Any]:
    image_id = actual.FROZEN_IMAGES[0]
    bits = actual.FROZEN_TUPLES[0]
    selected = actual.VARIANT_ORDER[0] if state == "selected" else None
    return {
        "quality_cell": {
            "image_id": image_id,
            "bit_tuple": list(bits),
            "q_name": actual.PRIMARY_ORDER[0],
            "selection": {
                "state": state,
                "selected_label": selected,
                "invalid_reason": "synthetic-invalid" if state == "invalid" else None,
            },
        },
        "candidate_cell": {
            "image_id": image_id,
            "bit_tuple": list(bits),
            "relative_path": f"candidates/{image_id}/synthetic",
            "execution": {},
        },
    }


@pytest.mark.parametrize(
    (
        "quality_state",
        "expected_plan_state",
        "expected_cells",
        "expected_primary",
        "expected_task_count",
    ),
    [
        ("invalid", "inapplicable_invalid_quality", 0, False, 0),
        ("method_failure", "complete_resource_plan", 1, False, 20),
        ("selected", "complete_resource_plan", 1, True, 40),
    ],
)
def test_resource_plan_branches_only_authorize_applicable_schedules(
    monkeypatch: pytest.MonkeyPatch,
    quality_state: str,
    expected_plan_state: str,
    expected_cells: int,
    expected_primary: bool,
    expected_task_count: int,
) -> None:
    monkeypatch.setattr(
        run,
        "source_manifest",
        lambda: {"source_manifest_sha256": "9" * 64},
    )
    monkeypatch.setattr(
        run,
        "_decoded_stream_record",
        lambda _execution, name: _stream_record(name),
    )
    context = {"phase_a": {"replay_phase_a_sha256": "a" * 64}}
    original_quality = {"quality_stage_sha256": "b" * 64}

    plan = run._replay_resource_plan_body(  # noqa: SLF001
        context=context,
        original_quality=original_quality,
        row_inputs=[_resource_row(quality_state)],
        quality_plan_sha256="c" * 64,
        quality_results_sha256="d" * 64,
    )

    actual.verify_record(plan, "resource_plan_sha256")
    assert plan["state"] == expected_plan_state
    assert plan["cell_count"] == expected_cells
    assert plan["primary_schedules_applicable"] is expected_primary
    assert plan["task_count"] == expected_task_count
    assert plan["scientific_summary_deferred"] is True
    if not plan["cells"]:
        assert plan["task_count"] == 0
        return

    cell = plan["cells"][0]
    assert cell["secondary"]["kind"] == "secondary"
    assert len(cell["secondary"]["tasks"]) == 20
    if expected_primary:
        assert cell["primary"]["kind"] == "primary"
        assert len(cell["primary"]["tasks"]) == 20
    else:
        assert cell["primary"] is None
    task_ids = [
        task["task_id"]
        for schedule in (cell["secondary"], cell["primary"])
        if schedule is not None
        for task in schedule["tasks"]
    ]
    assert len(task_ids) == len(set(task_ids)) == expected_task_count


def test_fresh_broker_ancestry_requires_fresh_root_with_null_parent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(run.sandbox, "verify_attestation", lambda _value: None)
    fresh = {
        "restriction_mode": run.sandbox.RESTRICTION_MODE_FRESH,
        "parent_sandbox_attestation_sha256": None,
    }
    run._verify_sandbox_ancestry(  # noqa: SLF001
        fresh,
        expected_parent_sandbox_attestation_sha256=None,
    )

    inherited = {
        "restriction_mode": run.sandbox.RESTRICTION_MODE_INHERITED,
        "parent_sandbox_attestation_sha256": "e" * 64,
    }
    with pytest.raises(run.LifecycleError, match="fresh root with null parent"):
        run._verify_sandbox_ancestry(  # noqa: SLF001
            inherited,
            expected_parent_sandbox_attestation_sha256=None,
        )

    malformed_fresh = {
        "restriction_mode": run.sandbox.RESTRICTION_MODE_FRESH,
        "parent_sandbox_attestation_sha256": "f" * 64,
    }
    with pytest.raises(run.LifecycleError, match="fresh root with null parent"):
        run._verify_sandbox_ancestry(  # noqa: SLF001
            malformed_fresh,
            expected_parent_sandbox_attestation_sha256=None,
        )


def test_final_record_is_byte_exact_host_reduction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    quality_replay = [{"image_id": "synthetic-quality"}]
    row_inputs = [{"quality_cell": {"selection": {"state": "selected"}}}]
    resource_replay = [{"image_id": "synthetic-resource"}]
    replay_rows = [{"row_sha256": "1" * 64}]
    replay_analysis = {"signature": "same", "decision": "synthetic"}
    context = {
        "candidate_replay": {"candidate_set_sha256": "2" * 64},
        "phase_a": {"replay_phase_a_sha256": "3" * 64},
    }
    plan = {"quality_plan_sha256": "4" * 64}
    quality_results = {"quality_results_sha256": "5" * 64}
    authorization = {
        "quality_authorization_sha256": "6" * 64,
        "resource_plan": {
            "state": "complete_resource_plan",
            "resource_plan_sha256": "7" * 64,
        },
    }
    resource_results = {"resource_results_sha256": "8" * 64}
    original_benchmark = {"benchmark_stage_sha256": "9" * 64}
    original_analysis_record = {"analysis_record_sha256": "a" * 64}
    original_analysis = {"signature": "same"}

    monkeypatch.setattr(
        run,
        "source_manifest",
        lambda: {"source_manifest_sha256": "b" * 64},
    )
    monkeypatch.setattr(
        run,
        "_verify_replay_quality_authorization",
        lambda *_args, **_kwargs: (quality_replay, row_inputs),
    )
    monkeypatch.setattr(
        run,
        "_reduce_replay_resource_results",
        lambda *_args, **_kwargs: (resource_replay, replay_rows),
    )
    monkeypatch.setattr(run.actual, "analyze_rows", lambda _rows: replay_analysis)
    monkeypatch.setattr(
        run,
        "_analysis_gate_signature",
        lambda value: value["signature"],
    )
    reduction_inputs = {
        "context": context,
        "plan": plan,
        "quality_results": quality_results,
        "authorization": authorization,
        "resource_results": resource_results,
        "targets": {},
        "original_quality": {},
        "original_benchmark": original_benchmark,
        "original_analysis_record": original_analysis_record,
        "original_analysis": original_analysis,
        "preflight_record": {},
        "captured_source_root": tmp_path,
        "replay_source_authority": {},
    }

    expected = run._replay_phase_b_final_record(  # noqa: SLF001
        **reduction_inputs
    )
    run._verify_replay_phase_b_final_record(  # noqa: SLF001
        expected,
        **reduction_inputs,
    )
    assert expected["quality_replay"] == quality_replay
    assert expected["resource_replay"] == resource_replay
    assert expected["replay_rows"] == replay_rows
    assert expected["replay_analysis"] == replay_analysis

    for field in (
        "quality_replay",
        "resource_replay",
        "replay_rows",
        "replay_analysis",
    ):
        tampered = copy.deepcopy(expected)
        tampered[field] = []
        tampered = actual.seal_record(
            {
                key: value
                for key, value in tampered.items()
                if key != "captured_replay_worker_sha256"
            },
            "captured_replay_worker_sha256",
        )
        with pytest.raises(
            run.LifecycleError,
            match="deterministic intermediate reduction",
        ):
            run._verify_replay_phase_b_final_record(  # noqa: SLF001
                tampered,
                **reduction_inputs,
            )


@pytest.mark.parametrize("tamper", ["extra_field", "source_manifest"])
def test_invalid_quality_requires_exact_zero_resource_result_schema(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tamper: str,
) -> None:
    source_sha256 = "1" * 64
    authorization = {
        "source_manifest_sha256": source_sha256,
        "quality_authorization_sha256": "2" * 64,
        "resource_plan": {
            "state": "inapplicable_invalid_quality",
            "resource_plan_sha256": "3" * 64,
        },
    }
    resource_results = actual.seal_record(
        {
            "schema": run.REPLAY_RESOURCE_RESULTS_SCHEMA,
            "protocol": run.PROTOCOL,
            "task_sha256": run.TASK_SHA256,
            "source_manifest_sha256": source_sha256,
            "quality_authorization_sha256": "2" * 64,
            "resource_plan_sha256": "3" * 64,
            "cells": [],
            "cell_count": 0,
            "task_count": 0,
            "all_workers_fresh_root": True,
            "scientific_summary_performed": False,
            "confirmation_payloads_accessed": False,
        },
        "resource_results_sha256",
    )
    malformed = {
        key: value
        for key, value in resource_results.items()
        if key != "resource_results_sha256"
    }
    if tamper == "extra_field":
        malformed["unexpected"] = True
    else:
        malformed["source_manifest_sha256"] = "4" * 64
    malformed = actual.seal_record(
        malformed,
        "resource_results_sha256",
    )
    monkeypatch.setattr(
        run,
        "_verify_replay_quality_authorization",
        lambda *_args, **_kwargs: (
            [],
            [
                {
                    "quality_cell": {
                        "image_id": actual.FROZEN_IMAGES[0],
                        "bit_tuple": list(actual.FROZEN_TUPLES[0]),
                        "selection": {
                            "state": "invalid",
                            "invalid_reason": "synthetic",
                        },
                    }
                }
            ],
        ),
    )

    with pytest.raises(
        run.LifecycleError,
        match="forbidden resources",
    ):
        run._replay_phase_b_final_record(  # noqa: SLF001
            context={
                "candidate_replay": {},
                "phase_a": {"replay_phase_a_sha256": "5" * 64},
            },
            plan={},
            quality_results={},
            authorization=authorization,
            resource_results=malformed,
            targets={},
            original_quality={},
            original_benchmark={},
            original_analysis_record={},
            original_analysis={},
            preflight_record={},
            captured_source_root=tmp_path,
            replay_source_authority={},
        )


@pytest.mark.parametrize("broker", ["quality", "resource"])
def test_host_brokers_fail_closed_inside_an_existing_sandbox(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    broker: str,
) -> None:
    monkeypatch.setenv(run.sandbox.SANDBOX_ATTESTATION_ENV, "0" * 64)
    if broker == "quality":
        call = lambda: run._execute_replay_quality_plan(  # noqa: E731, SLF001
            {},
            outdir=tmp_path,
            context={},
            targets={},
            original_quality={},
            preflight_record={},
            captured_source_root=tmp_path,
            system_read_only=[],
            gpu_read_write=[],
            scratch_root=tmp_path,
        )
    else:
        call = lambda: run._execute_replay_resource_plan(  # noqa: E731, SLF001
            {},
            {},
            {},
            outdir=tmp_path,
            context={},
            targets={},
            original_quality={},
            preflight_record={},
            captured_source_root=tmp_path,
            replay_source_authority={},
            system_read_only=[],
        )

    with pytest.raises(run.LifecycleError, match="outside every sandbox"):
        call()
