import ast
import inspect
import json
from pathlib import Path
import tarfile
import textwrap

import numpy as np
import torch

from benchmarks import local_linear_reproducing_assay as early
from benchmarks import local_linear_reproducing_full_assay as assay


def _synthetic_field() -> dict[str, np.ndarray]:
    means = np.array(((0, 0), (63, 0), (0, 63), (63, 63)), dtype=np.float32)
    scales = np.full((4, 2), 63.0, dtype=np.float32)
    logs = np.log(scales).astype(np.float32)
    rotations = np.zeros(4, dtype=np.float32)
    inverse = np.float32(1.0 / (63.0**2))
    conics = np.broadcast_to(np.array((inverse, 0.0, inverse), dtype=np.float32), (4, 3)).copy()
    return {
        "means": means,
        "log_scales": logs,
        "rotations": rotations,
        "colors": np.broadcast_to(np.array((0.25, 0.5, 0.75), dtype=np.float32), (4, 3)).copy(),
        "conics_float32": conics,
        "radii": np.full((4, 2), 63, dtype=np.int64),
    }


def test_frozen_parent_and_clarified_task_hashes_are_distinct_and_current() -> None:
    root = Path(__file__).resolve().parents[1]
    assert assay.PROTOCOL.endswith("-full-v3")
    assert assay.SCHEMA.endswith("-full-config-v3")
    assert assay.PARENT_TASK_SHA256 == (
        "cc5506f1536771d77a8ade9e8c076520d772d984287124db70d0c5c3ae8aeb11"
    )
    assert assay.PREVIOUS_CLARIFIED_TASK_SHA256 == (
        "9d74511c95786592b5071b5f9b61434983b95bb0a28a97377903b45327cebb09"
    )
    assert assay.TASK_SHA256 == (
        "b40b9075262c8c2dc07212490a1178b0168b6da9e433cb4ca23a10957bb1d0ad"
    )
    assert assay._sha256_file(
        root / "tasks/BENCH-013-local-linear-reproducing-compositor.md"
    ) == assay.TASK_SHA256
    assert assay.RECOVERY_PREDECESSOR == {
        "path": "results/bench013_local_linear_stage0_full_2026-07-16",
        "binding_sha256": (
            "6817380d2599ff2d5d8a0a22d244b36fb5e9cb76ae4b220777d40d87b6a5e7d4"
        ),
        "inventory_sha256": (
            "111c0680df375e753ef6578847e1b80769e4e5f1a9bed09ebb9e5c9f05231ab9"
        ),
        "inventory_algorithm": (
            "python_path_component_order_relative_posix_tab_size_tab_sha256_lf_no_trailing_lf"
        ),
    }


def test_recovery_inventory_uses_frozen_python_path_component_order(
    tmp_path: Path,
) -> None:
    nested = tmp_path / "input_fields" / "a.npz"
    root_file = tmp_path / "input_fields.jsonl"
    nested.parent.mkdir()
    nested.write_bytes(b"nested")
    root_file.write_bytes(b"root")

    record = assay._directory_inventory(tmp_path)
    path_order = (nested, root_file)
    expected_lines = [
        (
            f"{path.relative_to(tmp_path).as_posix()}\t{path.stat().st_size}\t"
            f"{assay._sha256_file(path)}"
        )
        for path in path_order
    ]
    expected_payload = "\n".join(expected_lines).encode("utf-8")
    assert record == {
        "algorithm": assay.RECOVERY_PREDECESSOR["inventory_algorithm"],
        "file_count": 2,
        "total_bytes": 10,
        "payload_bytes": len(expected_payload),
        "sha256": assay._sha256_bytes(expected_payload),
    }

    string_order = sorted(
        path_order, key=lambda path: path.relative_to(tmp_path).as_posix()
    )
    assert string_order == [root_file, nested]
    string_payload = "\n".join(
        f"{path.relative_to(tmp_path).as_posix()}\t{path.stat().st_size}\t"
        f"{assay._sha256_file(path)}"
        for path in string_order
    ).encode("utf-8")
    assert assay._sha256_bytes(string_payload) != record["sha256"]


def test_real_recovery_predecessor_inventory_and_invalid_lifecycle_are_frozen() -> None:
    root = Path(__file__).resolve().parents[1]
    record = assay.verify_recovery_predecessor(root)
    predecessor = root / assay.RECOVERY_PREDECESSOR["path"]
    assert record["verified"] is True
    assert record["expected_binding_sha256"] == assay.RECOVERY_PREDECESSOR[
        "binding_sha256"
    ]
    assert record["inventory"] == {
        "algorithm": assay.RECOVERY_PREDECESSOR["inventory_algorithm"],
        "file_count": 358,
        "total_bytes": 1_141_773_631,
        "payload_bytes": 45_140,
        "sha256": assay.RECOVERY_PREDECESSOR["inventory_sha256"],
    }
    assert record["lifecycle"]["replay_artifact_audit_passed"] is False
    assert record["lifecycle"]["config_recomputed_binding_sha256"] == (
        assay.RECOVERY_PREDECESSOR["binding_sha256"]
    )
    assert record["lifecycle"]["replay_check_count"] == 14
    assert record["lifecycle"]["replay_pass_count"] == 13
    assert record["lifecycle"]["replay_failed_checks"] == ["forward_raw_replay"]
    assert record["lifecycle"]["analysis_absent"] is True
    assert record["lifecycle"]["completion_absent"] is True
    assert record["lifecycle"]["artifact_manifest_absent"] is True
    assert len(list(predecessor.rglob("*.npz"))) == assay.RECOVERY_EXPECTED_NPZ_COUNT


def test_recovery_equivalence_accepts_only_authorized_normalization_and_tamper_fails(
    tmp_path: Path,
) -> None:
    predecessor = tmp_path / "predecessor"
    current = tmp_path / "current"
    predecessor.mkdir()
    current.mkdir()
    predecessor_binding = assay.RECOVERY_PREDECESSOR["binding_sha256"]
    current_binding = "c" * 64

    arrays = {
        "values": np.array(((1.0, 2.0), (3.0, 4.0)), dtype=np.float32)
    }
    for directory in (predecessor, current):
        (directory / "forward").mkdir()
        assay._deterministic_npz(directory / "targets.npz", arrays)
        assay._deterministic_npz(directory / "forward/cell.npz", arrays)
        assay._atomic_json(
            directory / "parent_binding.json",
            {"binding_sha256": assay.PARENT_BINDING, "payload": [1, 2, 3]},
        )

    for relative in assay.RECOVERY_JSON_PATHS:
        common = {
            "schema": f"test-{relative}",
            "scientific_value": 7,
            "nested": {"binding_sha256": "science-bearing-nested-value"},
        }
        assay._atomic_json(
            predecessor / relative,
            {**common, "binding_sha256": predecessor_binding},
        )
        assay._atomic_json(
            current / relative,
            {**common, "binding_sha256": current_binding},
        )

    def write_ledgers(*, altered_forward: bool = False) -> None:
        for relative in assay.RECOVERY_JSONL_PATHS:
            common = {
                "schema": f"test-{relative}",
                "scientific_value": 8,
                "nested": {"binding_sha256": "science-bearing-nested-value"},
            }
            (predecessor / relative).write_bytes(
                assay._canonical_json(
                    {**common, "binding_sha256": predecessor_binding}
                )
                + b"\n"
            )
            current_common = dict(common)
            if altered_forward and relative == "forward_cells.jsonl":
                current_common["scientific_value"] = 9
            (current / relative).write_bytes(
                assay._canonical_json(
                    {**current_common, "binding_sha256": current_binding}
                )
                + b"\n"
            )

    write_ledgers()
    common_config = {
        "counts": [28, 56, 112],
        "gates": {"frozen": 1.0},
        "parent": {"downstream_task_sha256": assay.PREVIOUS_CLARIFIED_TASK_SHA256},
    }
    assay._atomic_json(
        predecessor / "config.json",
        {
            **common_config,
            "schema": "old",
            "protocol": "old",
            "clarified_task_sha256": assay.PREVIOUS_CLARIFIED_TASK_SHA256,
            "source_manifest": {"old": "source"},
            "binding_sha256": predecessor_binding,
        },
    )
    assay._atomic_json(
        current / "config.json",
        {
            **common_config,
            "schema": "new",
            "protocol": "new",
            "clarified_task_sha256": assay.TASK_SHA256,
            "source_manifest": {"new": "source"},
            "binding_sha256": current_binding,
            "infrastructure_recovery": {"lifecycle_only": True},
            "replay_validation": {"lifecycle_only": True},
        },
    )

    equivalent = assay._recovery_equivalence(
        predecessor, current, current_binding, expected_npz_count=2
    )
    assert equivalent["equivalence_pass"] is True
    assert equivalent["npz"]["predecessor"]["file_count"] == 2
    assert len(equivalent["npz"]["predecessor"]["files"]) == 2
    assert equivalent["normalized_science_config"]["equivalent"] is True
    assert all(
        record["equivalent"]
        for record in equivalent["binding_normalized_ledgers"].values()
    )

    assay._deterministic_npz(
        current / "forward/cell.npz",
        {"values": arrays["values"] + np.float32(1.0)},
    )
    npz_tamper = assay._recovery_equivalence(
        predecessor, current, current_binding, expected_npz_count=2
    )
    assert npz_tamper["equivalence_pass"] is False
    assert npz_tamper["npz"]["equivalent"] is False

    assay._deterministic_npz(current / "forward/cell.npz", arrays)
    write_ledgers(altered_forward=True)
    ledger_tamper = assay._recovery_equivalence(
        predecessor, current, current_binding, expected_npz_count=2
    )
    assert ledger_tamper["equivalence_pass"] is False
    assert ledger_tamper["binding_normalized_ledgers"]["forward_cells.jsonl"][
        "equivalent"
    ] is False

    write_ledgers()
    current_config = json.loads((current / "config.json").read_text(encoding="utf-8"))
    current_config["counts"] = [29, 56, 112]
    assay._atomic_json(current / "config.json", current_config)
    config_tamper = assay._recovery_equivalence(
        predecessor, current, current_binding, expected_npz_count=2
    )
    assert config_tamper["equivalence_pass"] is False
    assert config_tamper["normalized_science_config"]["equivalent"] is False


def test_only_top_level_lifecycle_schemas_advance_to_v3() -> None:
    source = inspect.getsource(assay)
    for lifecycle in (
        "bench013-local-linear-reproducing-stage0-full-v3",
        "bench013-stage0-full-config-v3",
        "bench013-stage0-full-replay-v3",
        "bench013-stage0-full-analysis-v3",
        "bench013-stage0-full-completion-v3",
        "bench013-stage0-full-artifact-manifest-v3",
    ):
        assert lifecycle in source
    for science_row in (
        "bench013-stage0-full-input-field-v2",
        "bench013-stage0-full-forward-cell-v2",
        "bench013-stage0-full-effective-field-v2",
        "bench013-stage0-full-permutation-row-v2",
        "bench013-stage0-gradient-v2",
        "bench013-stage0-gradient-direction-row-v2",
        "bench013-stage0-gradient-block-row-v2",
        "bench013-stage0-gradient-status-row-v2",
    ):
        assert science_row in source
        assert science_row.removesuffix("v2") + "v3" not in source


def test_v3_recovery_preserves_archived_science_ast_and_core_source_bytes() -> None:
    root = Path(__file__).resolve().parents[1]
    archive_path = (
        root
        / "results/bench013_local_linear_stage0_full_2026-07-16/executed_sources.tar"
    )
    runner_path = "benchmarks/local_linear_reproducing_full_assay.py"
    with tarfile.open(archive_path, mode="r") as archive:
        archived_runner = archive.extractfile(runner_path)
        assert archived_runner is not None
        previous_source = archived_runner.read().decode("utf-8")
        for relative in (
            "benchmarks/local_linear_reproducing_assay.py",
            "benchmarks/local_linear_reproducing_core.py",
            "benchmarks/local_linear_reproducing_downstream_core.py",
        ):
            archived = archive.extractfile(relative)
            assert archived is not None
            assert archived.read() == (root / relative).read_bytes()

    current_source = (root / runner_path).read_text(encoding="utf-8")

    def module_records(source: str) -> tuple[dict[str, str], dict[str, str], ast.Module]:
        tree = ast.parse(source)
        functions = {
            node.name: ast.dump(node, include_attributes=False)
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
        }
        assignments = {
            node.targets[0].id: ast.dump(node.value, include_attributes=False)
            for node in tree.body
            if isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        }
        return functions, assignments, tree

    previous_functions, previous_assignments, previous_tree = module_records(
        previous_source
    )
    current_functions, current_assignments, current_tree = module_records(current_source)
    for name in (
        "HEIGHT",
        "WIDTH",
        "PIXELS",
        "FORWARD_CELLS",
        "LOGICAL_FIELDS",
        "PERMUTATION_SEED",
        "GRADIENT_SEED",
        "GRADIENT_STEP",
        "PERMUTATION_ORDERS",
        "SHORT_CIRCUIT",
        "GATES",
    ):
        assert current_assignments[name] == previous_assignments[name]
    for name in (
        "_evaluate",
        "_forward_arrays",
        "_metrics_and_gates",
        "_raw_metrics_and_gates",
        "_range_excursion_diagnostic",
        "_gradient_audit",
        "downstream_controls",
        "_dimensionless_gradients",
        "_gradient_block_metrics",
        "_gradient_loss",
    ):
        assert current_functions[name] == previous_functions[name]

    def normalized_science_config(tree: ast.Module) -> list[tuple[str, str]]:
        run = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "run"
        )
        assignment = next(
            node
            for node in ast.walk(run)
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "science"
                for target in node.targets
            )
        )
        assert isinstance(assignment.value, ast.Dict)
        return [
            (
                ast.dump(key, include_attributes=False),
                ast.dump(value, include_attributes=False),
            )
            for key, value in zip(
                assignment.value.keys, assignment.value.values, strict=True
            )
            if not (
                isinstance(key, ast.Constant)
                and key.value in assay.RECOVERY_CONFIG_DROP_KEYS
            )
        ]

    assert normalized_science_config(current_tree) == normalized_science_config(
        previous_tree
    )
    science_row_schemas = {
        "bench013-stage0-full-controls-v2",
        "bench013-stage0-full-input-field-v2",
        "bench013-stage0-full-forward-cell-v2",
        "bench013-stage0-full-effective-field-v2",
        "bench013-stage0-full-permutation-row-v2",
        "bench013-stage0-gradient-v2",
        "bench013-stage0-gradient-direction-row-v2",
        "bench013-stage0-gradient-block-row-v2",
        "bench013-stage0-gradient-status-row-v2",
    }
    previous_science_rows = sorted(
        node.value
        for node in ast.walk(previous_tree)
        if isinstance(node, ast.Constant)
        and node.value in science_row_schemas
    )
    current_science_rows = sorted(
        node.value
        for node in ast.walk(current_tree)
        if isinstance(node, ast.Constant)
        and node.value in science_row_schemas
    )
    assert current_science_rows == previous_science_rows


def test_real_early_artifact_verifies_against_frozen_parent_hashes() -> None:
    root = Path(__file__).resolve().parents[1]
    parent = root / "results/bench013_local_linear_stage0_early_2026-07-16"
    verified = assay.verify_parent(parent)
    assert verified["binding_sha256"] == assay.PARENT_BINDING
    assert verified["critical_hashes"] == assay.PARENT_HASHES
    assert verified["manifest_file_count"] > 0
    predecessor_config = json.loads(
        (
            root
            / assay.RECOVERY_PREDECESSOR["path"]
            / "config.json"
        ).read_text(encoding="utf-8")
    )
    assert verified == predecessor_config["parent"]


def test_new_core_plumbing_controls_are_valid_and_negative_ranks_are_exact() -> None:
    controls, raw = assay.downstream_controls()
    assert controls["valid"] is True
    assert controls["positive"]["math64_accepted"] is True
    assert controls["positive"]["cand32_accepted"] is True
    assert controls["negative_collinear"]["math64_rank_max"] == 2
    assert controls["negative_one_node"]["math64_rank_max"] == 1
    assert controls["negative_one_node"]["active_max"] == 1
    assert raw["positive_math64_output"].shape == (9, 9, 3)


def test_shared_constant_evaluation_uses_requested_target_colors() -> None:
    arrays = _synthetic_field()
    permutation = np.arange(4, dtype=np.int64)
    mathematical, same_state, candidate, exact, stored = assay._evaluate(
        arrays, "affine_xy", permutation
    )
    truth = early.target_image("affine_xy", dtype=np.float64)
    assert np.max(np.abs(mathematical.output.numpy() - truth)) <= 1e-12
    assert np.max(np.abs(same_state.output.numpy() - truth)) <= 2e-6
    assert np.max(np.abs(candidate.output.numpy().astype(np.float64) - truth)) <= 2e-5
    assert not np.allclose(exact, arrays["colors"].astype(np.float64))
    np.testing.assert_array_equal(stored, exact.astype(np.float32))


def test_permutation_global_pcg64_stream_has_frozen_first_two_draws() -> None:
    rng = np.random.Generator(np.random.PCG64(assay.PERMUTATION_SEED))
    first = rng.permutation(28).astype(np.int64)
    second = rng.permutation(28).astype(np.int64)
    assert assay._array_record(first)["record_sha256"] == (
        "7d539247b0c602b86f2fb6129752f4e2de83588a1bb93466ab67f0d3e57dc96a"
    )
    assert assay._array_record(second)["record_sha256"] == (
        "57efd65a871205f7fb7d49d1f77ed72f7a8af9b4319720dfdbe099db0f5ccf6b"
    )
    assert early.cell_specs()[0]["cell_id"] == "shared_constant__affine_xy__n0028__s0"


def test_gradient_global_pcg64_stream_and_exact_n56_shapes_are_frozen() -> None:
    rng = np.random.Generator(np.random.PCG64(assay.GRADIENT_SEED))
    cotangent = rng.standard_normal((assay.HEIGHT, assay.WIDTH, 3))
    cotangent /= np.linalg.norm(cotangent.reshape(-1))
    assert assay._array_record(cotangent)["record_sha256"] == (
        "0cde661e3aa6924091ad3026883e2c239262e795009b6f65e3a5cf2778fb43bb"
    )
    assert assay._array_record(cotangent.astype(np.float32))["record_sha256"] == (
        "1059b58b38d0b3e39ad2455d2e88e899c32d99e34c8fa43e9058ecb122f25e17"
    )
    shapes = ((56, 2), (56, 2), (56,), (56, 3))
    expected_hashes = (
        "d8c6f5162b8af1e639b3385277ec10839fc994b9102f92e8db659a15c52ec32b",
        "e043985133e5d514253b45ae4e792196a49f64651dfa19ff2cfe1ff7708983aa",
        "0d8bd4b1893a5ad46edcb78c53b02233ebabca0a8378ebdce5a597cdfb1e8adf",
        "8f8a060703132a0ff6a5dac02a7be555ad7e916e285c988db4d6906078057847",
        "fe4894d2f54c803a4f05af0168432241b5b891fd57859376921766bc774e1eda",
        "71b7d83b4a658c1653b73872a66b7ad84fcde9110caa6868d2ac3f703d44f549",
        "abd966f0aef6e0f86a580a463ac661859ab27d45794ecd19bbd7232192bff402",
        "d39208270eaa60f34d9bdd1652dd686c4c1ca9a669a1bac26c27c187d054ccdc",
    )
    observed: list[str] = []
    for shape in shapes:
        for _ in range(2):
            direction = rng.standard_normal(shape)
            direction /= np.linalg.norm(direction.reshape(-1))
            observed.append(assay._array_record(direction)["record_sha256"])
    assert tuple(observed) == expected_hashes


def test_dimensionless_gradient_transform_has_frozen_chain_rule() -> None:
    gradients = (
        torch.ones((2, 2), dtype=torch.float64),
        torch.full((2, 2), 2.0, dtype=torch.float64),
        torch.full((2,), 3.0, dtype=torch.float64),
        torch.full((2, 3), 4.0, dtype=torch.float64),
    )
    transformed = assay._dimensionless_gradients(gradients, 64, 64)
    torch.testing.assert_close(
        transformed["means_over_l"], torch.full((2, 2), 63.0, dtype=torch.float64)
    )
    torch.testing.assert_close(transformed["log_scales"], gradients[1])
    torch.testing.assert_close(
        transformed["rotations_over_pi"], gradients[2] * np.pi
    )
    torch.testing.assert_close(transformed["colors"], gradients[3])


def test_full_gradient_block_gate_keeps_relative_error_diagnostic_only() -> None:
    reference = torch.zeros(2, dtype=torch.float64)
    candidate = torch.tensor((1.0e-5, 0.0), dtype=torch.float32)
    row = assay._gradient_block_metrics("log_scales", reference, candidate)
    assert row["relative_l2_error"] > 1.0e6
    assert row["absolute_l2_error"] <= row["mixed_l2_tolerance"]
    assert row["mixed_l2_tolerance"] == assay.GATES["gradient_full_abs"]
    assert row["pass"] is True


def test_synthetic_gradient_audit_executes_all_blocks_with_frozen_geometry() -> None:
    arrays = _synthetic_field()
    summary, raw = assay._gradient_audit(arrays, "affine_xy")
    assert summary["coordinate_blocks"] == list(assay.GRADIENT_BLOCKS)
    assert len(summary["directions"]) == 8
    assert len(summary["blocks"]) == 4
    assert summary["directional_pass"] is True
    assert summary["block_pass"] is True
    assert summary["gradient_pass"] is True
    assert assay._gradient_raw_shape_contract(raw, 4)
    np.testing.assert_array_equal(raw["frozen_radii"], arrays["radii"])
    np.testing.assert_array_equal(raw["base_means_float32"], arrays["means"])
    near_null = next(row for row in summary["blocks"] if row["block"] == "log_scales")
    assert near_null["relative_l2_error"] > 1.0
    assert near_null["absolute_l2_error"] <= near_null["mixed_l2_tolerance"]
    assert near_null["pass"] is True


def test_stored_float32_conics_are_isolated_to_registered_paths_and_keep_parity() -> None:
    arrays = _synthetic_field()
    permutation = np.arange(4, dtype=np.int64)
    mathematical, same_state, candidate, exact, stored = assay._evaluate(
        arrays, "checker8", permutation
    )

    poisoned_colors = {name: value.copy() for name, value in arrays.items()}
    poisoned_colors["colors"][:] = np.float32(-123.0)
    math_color, same_color, cand_color, exact_color, stored_color = assay._evaluate(
        poisoned_colors, "checker8", permutation
    )
    np.testing.assert_array_equal(math_color.output.numpy(), mathematical.output.numpy())
    np.testing.assert_array_equal(same_color.output.numpy(), same_state.output.numpy())
    np.testing.assert_array_equal(cand_color.output.numpy(), candidate.output.numpy())
    np.testing.assert_array_equal(exact_color, exact)
    np.testing.assert_array_equal(stored_color, stored)

    poisoned_conics = {name: value.copy() for name, value in arrays.items()}
    factors = np.array((0.35, 1.8, 0.7, 2.5), dtype=np.float32)
    poisoned_conics["conics_float32"][:, 0] *= factors
    poisoned_conics["conics_float32"][:, 2] *= factors[::-1]
    math_poison, same_poison, cand_poison, _, _ = assay._evaluate(
        poisoned_conics, "checker8", permutation
    )
    np.testing.assert_array_equal(math_poison.output.numpy(), mathematical.output.numpy())
    assert np.max(np.abs(same_poison.output.numpy() - same_state.output.numpy())) > 0.05
    assert np.max(np.abs(cand_poison.output.numpy() - candidate.output.numpy())) > 0.05
    assert np.max(
        np.abs(cand_poison.output.numpy().astype(np.float64) - same_poison.output.numpy())
    ) < 2e-5
    assert bool(cand_poison.solve_pass.all())


def test_replay_contract_helpers_reject_mask_representative_and_shape_tampering() -> None:
    arrays = _synthetic_field()
    mathematical, same_state, candidate, _, _ = assay._evaluate(
        arrays, "checker8", np.arange(4, dtype=np.int64)
    )
    raw = assay._forward_arrays(mathematical, same_state, candidate)
    assert assay._candidate_mask_contract(raw)
    assert assay._candidate_eigenvalue_contract(raw)
    tampered_mask = dict(raw)
    tampered_mask["candidate_mass_pass"] = raw["candidate_mass_pass"].copy()
    tampered_mask["candidate_mass_pass"][0] = ~tampered_mask["candidate_mass_pass"][0]
    assert not assay._candidate_mask_contract(tampered_mask)
    tampered_eigenvalues = dict(raw)
    tampered_eigenvalues["candidate_covariance_eigenvalues"] = raw[
        "candidate_covariance_eigenvalues"
    ].copy()
    tampered_eigenvalues["candidate_covariance_eigenvalues"][0, 0] += np.float32(1e-3)
    assert not assay._candidate_eigenvalue_contract(tampered_eigenvalues)
    tampered_condition = dict(raw)
    tampered_condition["candidate_condition_numbers"] = raw[
        "candidate_condition_numbers"
    ].copy()
    tampered_condition["candidate_condition_numbers"][0] = np.nextafter(
        tampered_condition["candidate_condition_numbers"][0], np.float32(np.inf)
    )
    assert not assay._candidate_eigenvalue_contract(tampered_condition)
    tampered_covariance = dict(raw)
    tampered_covariance["candidate_covariance"] = raw["candidate_covariance"].copy()
    tampered_covariance["candidate_covariance"][0, 0, 0] += np.float32(1e-3)
    assert not assay._candidate_eigenvalue_contract(tampered_covariance)

    effective_keys = {"mathematical_partition", "candidate_partition"}
    effective = {name: raw[name].copy() for name in effective_keys}
    assert assay._effective_arrays_match(effective, raw, effective_keys)
    tampered_effective = {name: value.copy() for name, value in effective.items()}
    tampered_effective["candidate_partition"][0] += np.float32(1e-3)
    assert not assay._effective_arrays_match(tampered_effective, raw, effective_keys)

    count = 56
    contributors = 7
    gradient_raw: dict[str, np.ndarray] = {
        "cotangent_float64": np.zeros((64, 64, 3), dtype=np.float64),
        "cotangent_float32": np.zeros((64, 64, 3), dtype=np.float32),
        "contributors_gid": np.zeros(contributors, dtype=np.int64),
        "contributors_px": np.zeros(contributors, dtype=np.int64),
        "contributors_py": np.zeros(contributors, dtype=np.int64),
        "frozen_radii": np.zeros((count, 2), dtype=np.int64),
        "base_means_float32": np.zeros((count, 2), dtype=np.float32),
        "base_log_scales_float32": np.zeros((count, 2), dtype=np.float32),
        "base_rotations_float32": np.zeros(count, dtype=np.float32),
        "base_colors_float32": np.zeros((count, 3), dtype=np.float32),
        "directional_loss_plus64": np.zeros(8, dtype=np.float64),
        "directional_loss_minus64": np.zeros(8, dtype=np.float64),
    }
    block_shapes = ((count, 2), (count, 2), (count,), (count, 3))
    for block, shape in zip(assay.GRADIENT_BLOCKS, block_shapes, strict=True):
        gradient_raw[f"directions__{block}"] = np.zeros((2, *shape), dtype=np.float64)
        gradient_raw[f"gradient64__{block}"] = np.zeros(shape, dtype=np.float64)
        gradient_raw[f"gradient32__{block}"] = np.zeros(shape, dtype=np.float32)
        gradient_raw[f"raw_gradient64__{block}"] = np.zeros(shape, dtype=np.float64)
        gradient_raw[f"raw_gradient32__{block}"] = np.zeros(shape, dtype=np.float32)
    assert assay._gradient_raw_shape_contract(gradient_raw, count)
    tampered_gradient = dict(gradient_raw)
    tampered_gradient["gradient32__colors"] = np.zeros((count, 2), dtype=np.float32)
    assert not assay._gradient_raw_shape_contract(tampered_gradient, count)


def test_componentwise_float32_backward_error_accepts_roundoff_and_rejects_tamper() -> None:
    assert assay.REPLAY_FLOAT32_BACKWARD_ERROR_MULTIPLIER == 128.0
    matrix = np.array(
        (((2.0, 0.25), (0.25, 1.0)), ((0.7, -0.1), (-0.1, 1.3))),
        dtype=np.float32,
    )
    slopes = np.array(
        (
            ((0.3, -0.7, 0.2), (0.9, 0.1, -0.4)),
            ((-0.2, 0.5, 0.8), (0.4, -0.6, 0.3)),
        ),
        dtype=np.float32,
    )
    cross = np.einsum("pij,pjk->pik", matrix, slopes).astype(np.float32)
    assert assay._componentwise_backward_error_contract(matrix, slopes, cross)
    tampered_slopes = slopes.copy()
    tampered_slopes[0, 0, 0] += np.float32(1e-3)
    assert not assay._componentwise_backward_error_contract(
        matrix, tampered_slopes, cross
    )

    inverse_mean = np.array(((0.4, -0.2), (0.7, 0.1)), dtype=np.float32)
    mean = np.einsum("pij,pj->pi", matrix, inverse_mean).astype(np.float32)
    assert assay._componentwise_backward_error_contract(matrix, inverse_mean, mean)
    tampered_inverse = inverse_mean.copy()
    tampered_inverse[1, 1] -= np.float32(1e-3)
    assert not assay._componentwise_backward_error_contract(
        matrix, tampered_inverse, mean
    )


def test_replay_ledger_metadata_rejects_order_and_gradient_status_tampering() -> None:
    binding = "binding"
    cell = {"cell_id": "cell", "target": "constant"}
    permutation_row = {
        **cell,
        "binding_sha256": binding,
        "order_index": 0,
        "order": "identity",
    }
    assert assay._permutation_row_identity_contract(permutation_row, 0, cell, binding)
    wrong_order_index = {**permutation_row, "order_index": 1}
    assert not assay._permutation_row_identity_contract(wrong_order_index, 0, cell, binding)

    gradient = {"status": "completed", "gradient_pass": True}
    status_row = {
        "schema": "bench013-stage0-gradient-status-row-v2",
        "binding_sha256": binding,
        "cell_id": "selected",
        "status": "completed",
        "row_type": "status",
        "gradient_pass": True,
    }
    assert assay._gradient_jsonl_metadata_contract(
        [status_row], gradient, binding, "selected"
    )
    wrong_status = {**status_row, "status": "not_reached"}
    assert not assay._gradient_jsonl_metadata_contract(
        [wrong_status], gradient, binding, "selected"
    )
    wrong_pass = {**status_row, "gradient_pass": False}
    assert not assay._gradient_jsonl_metadata_contract(
        [wrong_pass], gradient, binding, "selected"
    )


def test_scientific_cell_loops_have_no_short_circuit_statements() -> None:
    tree = ast.parse(textwrap.dedent(inspect.getsource(assay.run)))
    cell_loops = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.For)
        and isinstance(node.target, ast.Name)
        and node.target.id == "cell"
        and isinstance(node.iter, ast.Name)
        and node.iter.id == "cells"
    ]
    assert len(cell_loops) == 2
    for loop in cell_loops:
        assert not any(
            isinstance(node, (ast.Break, ast.Continue, ast.Return))
            for node in ast.walk(loop)
        )


def test_recovery_preflight_equivalence_replay_and_lifecycle_order_are_frozen() -> None:
    source = textwrap.dedent(inspect.getsource(assay.run))
    preflight = source.index(
        "predecessor_verification = verify_recovery_predecessor(root)"
    )
    create_outdir = source.index("outdir.mkdir(parents=True)")
    timing_written = source.index('_atomic_json(outdir / "timing.json", timing)')
    equivalence_computed = source.index("recovery_equivalence = _recovery_equivalence(")
    equivalence_written = source.index(
        "_atomic_json(recovery_equivalence_path, recovery_equivalence)"
    )
    replay_started = source.index("replay = replay_full_artifacts(outdir, root, parent)")
    incomplete_boundary = source.index("if not complete:")
    outcome_interpretation = source.index("forward_pass = all(")
    assert preflight < create_outdir
    assert timing_written < equivalence_computed < equivalence_written < replay_started
    assert incomplete_boundary < outcome_interpretation
    assert 'and recovery_equivalence["equivalence_pass"]' in source

    replay_source = textwrap.dedent(inspect.getsource(assay.replay_full_artifacts))
    assert "predecessor_verification = verify_recovery_predecessor(root)" in replay_source
    assert "replayed_recovery_equivalence = _recovery_equivalence(" in replay_source
    assert (
        '_nested_close(stored_recovery_equivalence, replayed_recovery_equivalence)'
        in replay_source
    )
    assert 'checks["recovery_equivalence"]' in replay_source
    assert '"artifact_audit_passed": all(checks.values())' in replay_source
    invalid_boundary = replay_source.index("if not all(checks.values()):")
    science_ledger_read = replay_source.index(
        'stored_controls = json.loads((outdir / "controls.json")'
    )
    assert invalid_boundary < science_ledger_read
    assert '"outcome_ledgers_interpreted": False' in replay_source


def test_invalid_recovery_analysis_is_outcome_free_and_never_authorizes_stage1() -> None:
    analysis = assay._outcome_free_unavailable_analysis(
        binding="binding",
        replay={"artifact_audit_passed": False},
        controls_valid=True,
        source_archive_record={"path": "executed_sources.tar"},
        timing={"diagnostic_only": True},
        recovery_equivalence={"equivalence_pass": False},
        recovery_equivalence_file_sha256="e" * 64,
    )
    assert analysis["complete"] is False
    assert analysis["decision"] == "unavailable"
    assert analysis["outcome_ledgers_interpreted"] is False
    assert analysis["stage1_authorized"] is False
    assert analysis["recovery_equivalence"]["equivalence_pass"] is False
    assert not {
        "forward",
        "effective_weights",
        "permutation",
        "gradient",
        "scientific_pass",
    } & analysis.keys()


def test_run_cannot_shadow_source_archive_metadata_with_npz_handle() -> None:
    source = textwrap.dedent(inspect.getsource(assay.run))
    tree = ast.parse(source)
    stored_names = [
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
    ]
    assert stored_names.count("source_archive_record") == 1
    assert "archive" not in stored_names
    assert "baseline_npz" in stored_names
    assert source.count('"source_archive": source_archive_record') == 2


def test_forward_table_contract_and_nonfinite_json_are_explicit() -> None:
    shapes = assay._expected_forward_shapes()
    assert shapes["candidate_accumulators"] == (4096, 15)
    assert shapes["mathematical_covariance_eigenvalues"] == (4096, 2)
    assert shapes["candidate_inverse_mean"] == (4096, 2)
    assert assay._json_safe(float("inf")) == {"nonfinite": "positive_infinity"}
    assert assay._json_safe(float("nan")) == {"nonfinite": "nan"}


def test_exact_color_extrema_drive_overshoot_diagnostic() -> None:
    exact = np.array(((0.100000001, 0.2, 0.3), (0.899999999, 0.8, 0.7)))
    output = np.broadcast_to(np.array((0.1, 0.8, 0.7)), (64, 64, 3)).copy()
    output[7, 9, 0] = 0.05
    diagnostic = assay._range_excursion_diagnostic(output, exact)
    assert diagnostic["sample_min_per_channel"][0] == exact[:, 0].min()
    assert diagnostic["maximum"] == exact[:, 0].min() - 0.05
    assert diagnostic["worst"] == {"x": 9, "y": 7, "channel": 0, "side": "below"}
