from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from benchmarks import clic2020_subset_acquire as A
from benchmarks import sgi_entropy_oracle as O
from benchmarks import sgi_entropy_oracle_run as R
from structsplat import codec
from structsplat.config import FitConfig
from structsplat.gaussians import GaussianField


@pytest.fixture(autouse=True)
def _bind_required_system_libstdcxx(monkeypatch):
    monkeypatch.setenv("LD_PRELOAD", str(R.REQUIRED_LD_PRELOAD))


def _fake_environment() -> dict:
    return {"synthetic": True, "device": "test-double"}


def _fake_prebuild(binary: Path) -> dict:
    return {
        "extension_binary": str(binary),
        "extension_binary_bytes": binary.stat().st_size,
        "extension_binary_sha256": R.sha256_file(binary),
        "synthetic_forward_backward_sha256": "1" * 64,
        "renderer": "cuda",
        "scientific_pixels_accessed": False,
    }


def _remote_binding() -> dict:
    return {
        "requested_url": A.FROZEN_URL,
        "resolved_url": A.FROZEN_URL,
        "status": 200,
        "etag": '"synthetic"',
        "last_modified": "Thu, 21 Aug 2025 00:16:34 GMT",
        "content_length": A.EXPECTED_CONTENT_LENGTH,
        "accept_ranges": "bytes",
        "gcs_generation": A.EXPECTED_GCS_GENERATION,
    }


def _synthetic_acquisition(root: Path) -> dict:
    root.mkdir()
    members = []
    archive_index = []
    for name in A.FROZEN_NAMES:
        development = name in A.DEVELOPMENT_NAMES
        payload = b"synthetic-encoded-png-" + name.encode("ascii")
        path = f"professional_valid/{name}.png"
        row = {
            "name": name,
            "partition": "development" if development else "confirmation",
            "archive_path": path,
            "archive_path_nfc": path,
            "crc32": "00000000",
            "compressed_size": len(payload),
            "uncompressed_size": len(payload),
            "compression_method": 8,
            "flag_bits": 0,
            "unix_mode": 0o100644,
            "is_directory": False,
            "payload_extracted": development,
            "local_path": f"{name}.png" if development else None,
        }
        archive_index.append(
            {
                key: row[key]
                for key in (
                    "archive_path",
                    "archive_path_nfc",
                    "crc32",
                    "compressed_size",
                    "uncompressed_size",
                    "compression_method",
                    "flag_bits",
                    "unix_mode",
                    "is_directory",
                )
            }
        )
        if development:
            target = root / f"{name}.png"
            target.write_bytes(payload)
            row.update(
                {
                    "byte_size": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "download_crc32": "00000000",
                }
            )
        members.append(row)
    binding = _remote_binding()
    return A.write_manifest(
        root,
        {
            "schema": A.MANIFEST_SCHEMA,
            "development_names": list(A.DEVELOPMENT_NAMES),
            "confirmation_names": list(A.CONFIRMATION_NAMES),
            "archive": binding,
            "post_acquisition_head": copy.deepcopy(binding),
            "remotezip_version": A.REMOTEZIP_VERSION,
            "archive_index": archive_index,
            "archive_index_payload_sha256": hashlib.sha256(
                A.canonical_json(archive_index)
            ).hexdigest(),
            "members": members,
            "range_requests": [{"requested_range": "bytes=-65536"}],
            "stage_process": {
                "started_utc": "2026-07-16T00:00:00Z",
                "ended_utc": "2026-07-16T00:00:01Z",
                "wall_seconds": 1.0,
                "pid": 1,
                "parent_pid": 0,
                "working_directory": "/synthetic",
                "python_executable": "/usr/bin/python",
                "invocation_argv": ["python", "-m", "benchmarks.clic2020_subset_acquire"],
            },
        },
    )


def _field(n: int, bits: tuple[int, int, int, int]) -> tuple[GaussianField, codec.CodecConfig]:
    index = np.arange(n, dtype=np.float32)
    means = np.column_stack([np.remainder(index * 7.0, 15.0), np.remainder(index * 11.0, 767.0)])
    scales = np.column_stack(
        [0.6 + np.remainder(index, 13.0) / 20.0, 0.7 + np.remainder(index, 17.0) / 25.0]
    )
    rotations = np.remainder(index * 0.17, np.pi)
    colors = np.column_stack(
        [
            np.remainder(index, 251.0) / 250.0,
            np.remainder(index * 3.0, 241.0) / 240.0,
            np.remainder(index * 5.0, 239.0) / 238.0,
        ]
    )
    field = GaussianField.from_numpy(means, scales, rotations, colors)
    return field, codec.CodecConfig(
        bits_means=bits[0],
        bits_scales=bits[1],
        bits_rot=bits[2],
        bits_colors=bits[3],
        morton_reorder=True,
        color_delta=True,
        means_byte_planar=True,
        rot_circular=True,
        stream_codec="zlib",
        zlib_level=9,
    )


def _scientific_blobs(tmp_path: Path) -> list[dict]:
    cells = []
    for bits in O.FROZEN_BIT_TUPLES:
        field, config = _field(O.FROZEN_GAUSSIAN_COUNT, bits)
        blob = codec.encode(field, 768, 16, config, FitConfig(renderer="cuda"))
        # The same tuple stream can back multiple synthetic image IDs; IDs are external metadata.
        for image_id in O.FROZEN_IMAGE_IDS:
            directory = tmp_path / "streams" / image_id / R._bit_label(bits)
            directory.mkdir(parents=True)
            path = directory / "stream.sspl1"
            path.write_bytes(blob)
            cells.append(
                {
                    "image_id": image_id,
                    "bit_tuple": list(bits),
                    "sspl1": {"path": path.relative_to(tmp_path).as_posix()},
                    "oracle_row": O.make_oracle_row(blob, image_id),
                    "cell_record_sha256": hashlib.sha256(f"{image_id}-{bits}".encode()).hexdigest(),
                }
            )
    # The lifecycle's scientific order is image-major, tuple-minor.
    by_key = {(cell["image_id"], tuple(cell["bit_tuple"])): cell for cell in cells}
    return [
        by_key[(image_id, bits)] for image_id in O.FROZEN_IMAGE_IDS for bits in O.FROZEN_BIT_TUPLES
    ]


def test_frozen_runner_configuration_has_no_coder_or_structural_edits():
    frozen = R.resolved_configs()
    assert tuple(frozen["development_names"]) == O.FROZEN_IMAGE_IDS
    assert frozen["gaussian_count"] == 8192
    assert frozen["fit_iterations"] == 4000
    assert frozen["qat_iterations"] == 150
    assert frozen["renderer"] == "cuda"
    assert frozen["init"]["strategy"] == "quadtree_wse"
    assert frozen["init"]["opacity_mode"] == "none"
    assert frozen["fit"]["checkpoint_policy"] == "terminal"
    assert frozen["fit"]["split_every"] is None
    assert frozen["fit"]["prune_every"] is None
    assert frozen["fit"]["covariance_filter_mode"] == "none"
    assert [
        tuple(row[key] for key in ("bits_means", "bits_scales", "bits_rot", "bits_colors"))
        for row in frozen["codec_arms"]
    ] == list(O.FROZEN_BIT_TUPLES)


def test_source_closure_binds_protocol_runner_tests_and_all_production_cuda():
    paths = {R._relative(path) for path in R.source_paths()}
    assert "tasks/COMP-008-mean-conditioned-entropy-oracle.md" in paths
    assert "benchmarks/clic2020_subset_acquire.py" in paths
    assert "benchmarks/sgi_entropy_oracle.py" in paths
    assert "benchmarks/sgi_entropy_oracle_run.py" in paths
    assert "tests/test_clic2020_subset_acquire.py" in paths
    assert "tests/test_sgi_entropy_oracle.py" in paths
    assert "tests/test_sgi_entropy_oracle_run.py" in paths
    assert "src/structsplat/codec.py" in paths
    assert "src/structsplat/fit.py" in paths
    assert "src/structsplat/cuda/render_ext.cpp" in paths
    assert "src/structsplat/cuda/render_ext.cu" in paths


def test_preflight_binds_sources_bootstrap_environment_and_prebuilt_binary(tmp_path):
    binary = tmp_path / "synthetic_extension.so"
    binary.write_bytes(b"synthetic-cuda-extension")
    outdir = tmp_path / "artifact"
    record = R.preflight(
        outdir,
        prebuild_fn=lambda: _fake_prebuild(binary),
        environment_fn=_fake_environment,
    )
    assert record["scientific_pixels_accessed"] is False
    assert record["bootstrap"]["sha256"] == O.BOOTSTRAP_MATRIX_SHA256
    assert record["runtime_abi"] == R.runtime_abi_record()
    assert record["synthetic_self_checks"]["ok"] is True
    assert record["synthetic_self_checks"]["scientific_pixels_accessed"] is False
    assert [tuple(item["bit_tuple"]) for item in record["synthetic_self_checks"]["checks"]] == list(
        O.FROZEN_BIT_TUPLES
    )
    assert record["stage_process"]["started_utc"]
    assert record["stage_process"]["ended_utc"]
    assert record["stage_process"]["invocation_argv"]
    assert (outdir / R.BOOTSTRAP_NAME).stat().st_size == 800_000
    assert (outdir / R.SOURCE_ARCHIVE_NAME).is_file()
    assert R.verify_preflight(outdir, environment_fn=_fake_environment) == record
    with pytest.raises(R.LifecycleError, match="must be empty"):
        R.preflight(
            outdir,
            prebuild_fn=lambda: _fake_prebuild(binary),
            environment_fn=_fake_environment,
        )


def test_preflight_captures_environment_before_cuda_prebuild_mutates_process(tmp_path):
    binary = tmp_path / "synthetic_extension.so"
    binary.write_bytes(b"synthetic-cuda-extension")
    events = []

    def environment():
        events.append("environment")
        return _fake_environment()

    def prebuild():
        events.append("cuda-prebuild")
        return _fake_prebuild(binary)

    R.preflight(tmp_path / "artifact", prebuild_fn=prebuild, environment_fn=environment)
    assert events == ["environment", "cuda-prebuild"]


def test_preflight_rejects_bootstrap_drift(tmp_path):
    binary = tmp_path / "extension.so"
    binary.write_bytes(b"x")
    outdir = tmp_path / "artifact"
    R.preflight(
        outdir,
        prebuild_fn=lambda: _fake_prebuild(binary),
        environment_fn=_fake_environment,
    )
    bootstrap = outdir / R.BOOTSTRAP_NAME
    bootstrap.chmod(0o644)
    bootstrap.write_bytes(b"tampered")
    with pytest.raises(R.LifecycleError, match="bootstrap"):
        R.verify_preflight(outdir, environment_fn=_fake_environment)


def test_preflight_rejects_frozen_source_archive_drift(tmp_path):
    binary = tmp_path / "extension.so"
    binary.write_bytes(b"x")
    outdir = tmp_path / "artifact"
    R.preflight(
        outdir,
        prebuild_fn=lambda: _fake_prebuild(binary),
        environment_fn=_fake_environment,
    )
    archive = outdir / R.SOURCE_ARCHIVE_NAME
    archive.chmod(0o644)
    with archive.open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(R.LifecycleError, match="source snapshot archive"):
        R.verify_preflight(outdir, environment_fn=_fake_environment)


def test_runtime_abi_explicitly_binds_little_endian_numpy_and_zlib(monkeypatch):
    record = R.runtime_abi_record()
    assert record["sys_byteorder"] == "little"
    assert record["numpy_version"] == np.__version__
    assert record["numpy_little_endian_u16_is_native"] is True
    assert record["zlib_compile_version"]
    assert record["zlib_runtime_version"]
    assert record["libstdcxx_preload"]["raw"] == str(R.REQUIRED_LD_PRELOAD)
    assert record["libstdcxx_preload"]["sha256"] == R.sha256_file(R.REQUIRED_LD_PRELOAD.resolve())
    monkeypatch.setattr(R.sys, "byteorder", "big")
    with pytest.raises(R.LifecycleError, match="little-endian"):
        R.runtime_abi_record()


def test_runtime_abi_requires_exact_sole_system_libstdcxx(monkeypatch):
    monkeypatch.delenv("LD_PRELOAD")
    with pytest.raises(R.LifecycleError, match="exact sole LD_PRELOAD"):
        R.runtime_abi_record()
    monkeypatch.setenv("LD_PRELOAD", f"{R.REQUIRED_LD_PRELOAD}:/tmp/foreign.so")
    with pytest.raises(R.LifecycleError, match="missing library|exact sole LD_PRELOAD"):
        R.runtime_abi_record()


def test_module_origins_are_the_captured_repository_sources():
    origins = R.module_origin_record()
    assert set(origins) == {
        "structsplat",
        "structsplat.codec",
        "structsplat.config",
        "structsplat.cuda_render",
        "structsplat.fit",
        "structsplat.gaussians",
        "structsplat.init",
        "structsplat.metrics",
        "structsplat.render",
    }
    source_root = (R.REPO_ROOT / "src" / "structsplat").resolve()
    assert all(Path(origin).is_relative_to(source_root) for origin in origins.values())


def test_acquisition_verifier_allows_only_development_payloads(tmp_path):
    acquisition_dir = tmp_path / "acquired"
    manifest = _synthetic_acquisition(acquisition_dir)
    assert R.verify_acquisition(acquisition_dir) == manifest
    assert all(
        member["payload_extracted"] is False and member["local_path"] is None
        for member in manifest["members"]
        if member["partition"] == "confirmation"
    )

    leaked = acquisition_dir / f"{A.CONFIRMATION_NAMES[0]}.png"
    leaked.write_bytes(b"must-never-be-opened")
    with pytest.raises(R.LifecycleError, match="only the eight development"):
        R.verify_acquisition(acquisition_dir)


def test_acquisition_verifier_rejects_forged_confirmation_payload_flag(tmp_path):
    acquisition_dir = tmp_path / "acquired"
    _synthetic_acquisition(acquisition_dir)
    manifest_path = acquisition_dir / A.MANIFEST_NAME
    manifest = json_load(manifest_path)
    manifest["members"][8]["payload_extracted"] = True
    core = dict(manifest)
    core.pop("manifest_payload_sha256")
    manifest["manifest_payload_sha256"] = hashlib.sha256(A.canonical_json(core)).hexdigest()
    manifest_path.write_bytes(A.canonical_json(manifest))
    (acquisition_dir / A.MANIFEST_HASH_NAME).write_text(
        f"{R.sha256_file(manifest_path)}  {A.MANIFEST_NAME}\n",
        encoding="ascii",
    )
    with pytest.raises(R.LifecycleError, match="not sealed"):
        R.verify_acquisition(acquisition_dir)


def test_acquisition_verifier_rejects_forged_process_provenance(tmp_path):
    acquisition_dir = tmp_path / "acquired"
    _synthetic_acquisition(acquisition_dir)
    manifest_path = acquisition_dir / A.MANIFEST_NAME
    manifest = json_load(manifest_path)
    manifest["stage_process"]["wall_seconds"] = -1.0
    core = dict(manifest)
    core.pop("manifest_payload_sha256")
    manifest["manifest_payload_sha256"] = hashlib.sha256(A.canonical_json(core)).hexdigest()
    manifest_path.write_bytes(A.canonical_json(manifest))
    (acquisition_dir / A.MANIFEST_HASH_NAME).write_text(
        f"{R.sha256_file(manifest_path)}  {A.MANIFEST_NAME}\n",
        encoding="ascii",
    )
    with pytest.raises(R.LifecycleError, match="wall time"):
        R.verify_acquisition(acquisition_dir)


def json_load(path: Path) -> dict:
    import json

    return json.loads(path.read_text(encoding="utf-8"))


def test_prepare_rgb_png_uses_exact_lanczos_scale_and_persists_hashes(tmp_path):
    source = tmp_path / "source.png"
    pixels = np.arange(2 * 4 * 3, dtype=np.uint8).reshape(2, 4, 3)
    Image.fromarray(pixels, mode="RGB").save(source)
    destination = tmp_path / "derived.png"
    record = R.prepare_rgb_png(source, destination, max_side=8)
    assert record["native_rgb_dimensions_hw"] == [2, 4]
    assert record["derived_rgb_dimensions_hw"] == [4, 8]
    assert record["resize_scale"] == 2.0
    assert record["source_encoded_sha256"] == R.sha256_file(source)
    assert record["derived_encoded_sha256"] == R.sha256_file(destination)
    with Image.open(destination) as image:
        derived = np.asarray(image.convert("RGB"), dtype=np.uint8)
    assert record["derived_rgb_pixel_sha256"] == R._pixel_sha256(derived)


def test_prepare_dev_dispatches_only_exact_development_names(tmp_path, monkeypatch):
    outdir = tmp_path / "artifact"
    outdir.mkdir()
    acquisition_dir = tmp_path / "acquired"
    acquired = _synthetic_acquisition(acquisition_dir)
    calls = []

    monkeypatch.setattr(
        R,
        "verify_preflight",
        lambda *args, **kwargs: {
            "binding_sha256": "a" * 64,
            "remote_object": {
                "url": A.FROZEN_URL,
                "content_length": A.EXPECTED_CONTENT_LENGTH,
                "gcs_generation": A.EXPECTED_GCS_GENERATION,
                "remotezip_version": A.REMOTEZIP_VERSION,
            },
        },
    )
    monkeypatch.setattr(R, "verify_acquisition", lambda path: acquired)

    def fake_prepare(source: Path, destination: Path):
        calls.append(source.stem)
        destination.write_bytes(b"synthetic-derived")
        return {
            "source_mode": "RGB",
            "source_encoded_bytes": source.stat().st_size,
            "source_encoded_sha256": R.sha256_file(source),
            "native_rgb_dimensions_hw": [1, 1],
            "native_rgb_pixel_sha256": "b" * 64,
            "resize_scale": 768.0,
            "derived_path": destination.as_posix(),
            "derived_encoded_bytes": destination.stat().st_size,
            "derived_encoded_sha256": R.sha256_file(destination),
            "derived_rgb_dimensions_hw": [768, 768],
            "derived_rgb_pixel_sha256": "c" * 64,
        }

    monkeypatch.setattr(R, "prepare_rgb_png", fake_prepare)
    manifest = R.prepare_development(outdir, acquisition_dir, environment_fn=_fake_environment)
    assert tuple(calls) == A.DEVELOPMENT_NAMES
    assert not set(calls) & set(A.CONFIRMATION_NAMES)
    assert manifest["confirmation_payloads_accessed"] is False
    assert manifest["stage_process"]["started_utc"]
    assert manifest["stage_process"]["ended_utc"]


def test_symbol_reconstruction_agrees_exactly_with_production_decode():
    field, config = _field(37, O.FROZEN_BIT_TUPLES[0])
    blob = codec.encode(field, 768, 16, config, FitConfig(renderer="cuda"))
    parsed = O.parse_sspl1(blob)
    expected = R._symbols_to_field_arrays(parsed)
    direct = R._field_arrays(codec.decode(blob, device="cpu"))
    assert set(expected) == set(direct)
    for name in expected:
        np.testing.assert_array_equal(expected[name], direct[name])
    integrity = R._decoded_integrity(blob)
    assert integrity["boundary_state_hash_exact"] is True
    assert integrity["production_vs_independent_max_abs_lte_1e_6"] is True
    assert integrity["decoded_field_atol"] == 1e-6
    assert integrity["cpu_reference_render_diagnostic"]["validity_gate"] is False
    assert integrity["cpu_reference_render_diagnostic"]["status"] == ("not-run-optional-diagnostic")


def test_mean_range_qat_mismatch_is_persisted_as_diagnostic_only():
    field, _ = _field(5, O.FROZEN_BIT_TUPLES[0])
    field.means[0, 0] = -0.2
    field.means[1, 1] = 768.1
    diagnostic = R._mean_range_diagnostic(field, 768, 16)
    assert diagnostic["off_image_mean_count"] == 2
    assert diagnostic["encoded_mean_range_lo_xy"][0] == -1.0
    assert diagnostic["encoded_mean_range_hi_xy"][1] == 769.0
    assert diagnostic["encoded_mean_range_differs_from_image_box"] is True
    assert "claim_limitation" in diagnostic


def test_field_state_and_npz_roundtrip_are_exact(tmp_path):
    field, _ = _field(19, O.FROZEN_BIT_TUPLES[0])
    record = R._persist_field(field, tmp_path / "field.npz")
    assert record["n"] == 19
    loaded = R._verify_field_artifact(tmp_path, {**record, "path": "field.npz"})
    assert R.field_state(loaded)["state_sha256"] == record["state_sha256"]


def test_base_record_roundtrip_validates_frozen_configs_and_dict_history(tmp_path):
    image_id = R.DEVELOPMENT_NAMES[0]
    directory = tmp_path / "fields" / image_id
    directory.mkdir(parents=True)
    field, _ = _field(R.GAUSSIAN_COUNT, O.FROZEN_BIT_TUPLES[0])
    initial = R._persist_field(field, directory / "initial.npz")
    base = R._persist_field(field, directory / "base.npz")
    initial["path"] = f"fields/{image_id}/initial.npz"
    base["path"] = f"fields/{image_id}/base.npz"
    history = {
        "iterations_run": R.FIT_ITERATIONS,
        "selected_iter": R.FIT_ITERATIONS,
        "selected_psnr": 20.0,
        "fit_seconds_internal": 1.0,
        "history": {
            "iter": [0, R.FIT_ITERATIONS - 1],
            "n_gaussians": [R.GAUSSIAN_COUNT, R.GAUSSIAN_COUNT],
        },
    }
    history_path = directory / "fit_history.json"
    history_path.write_bytes(R.canonical_json(history))
    frozen = R.resolved_configs()
    process = {
        "started_utc": "2026-07-16T00:00:00Z",
        "ended_utc": "2026-07-16T00:00:01Z",
        "wall_seconds": 1.0,
        "pid": 1,
        "parent_pid": 0,
        "working_directory": "/synthetic",
        "python_executable": "/usr/bin/python",
        "invocation_argv": ["python", "-m", "benchmarks.sgi_entropy_oracle_run"],
    }
    record = R._seal(
        {
            "schema": R.BASE_SCHEMA,
            "protocol": R.PROTOCOL,
            "image_id": image_id,
            "target_pixel_sha256": "a" * 64,
            "dimensions_hw": [768, 16],
            "init_config": frozen["init"],
            "structure_tensor_config": frozen["structure_tensor"],
            "fit_config": frozen["fit"],
            "initial_field": initial,
            "base_field": base,
            "fit_history": {
                "path": f"fields/{image_id}/fit_history.json",
                "bytes": history_path.stat().st_size,
                "sha256": R.sha256_file(history_path),
            },
            "timings": {"initialization_wall_seconds": 0.1, "ordinary_fit_wall_seconds": 1.0},
            "unsealed_staging_restarted_after_interruption": False,
            "stage_process": process,
        },
        "base_record_sha256",
    )
    (directory / "base_record.json").write_bytes(R.canonical_json(record))
    assert R._load_base_record(tmp_path, image_id) == record


def test_append_only_journal_rejects_truncation_and_noncanonical_json(tmp_path):
    path = tmp_path / "journal.jsonl"
    R._append_jsonl(path, {"b": 2, "a": 1})
    assert R._load_jsonl(path) == [{"a": 1, "b": 2}]
    path.write_bytes(path.read_bytes().rstrip(b"\n"))
    with pytest.raises(R.LifecycleError, match="truncated"):
        R._load_jsonl(path)


def test_killed_process_tail_and_unsealed_staging_are_recoverable(tmp_path):
    journal = tmp_path / R.JOURNAL_NAME
    R._append_jsonl(journal, {"committed": True})
    with journal.open("ab") as handle:
        handle.write(b'{"uncommitted":')
    repair = R._repair_truncated_jsonl_tail(journal)
    assert repair["discarded_fragment_bytes"] > 0
    assert R._load_jsonl(journal) == [{"committed": True}]

    staging = tmp_path / "fields" / f"{R.DEVELOPMENT_NAMES[0]}.part"
    staging.mkdir(parents=True)
    (staging / "initial.npz").write_bytes(b"unsealed")
    R._assert_known_artifact_layout(tmp_path, allow_staging=True)
    removed = R._remove_unsealed_staging_directories(tmp_path)
    assert removed[0]["path"].endswith(".part")
    assert not staging.exists()
    R._assert_known_artifact_layout(tmp_path, allow_staging=False)


def test_artifact_layout_rejects_foreign_top_level_and_nested_files(tmp_path):
    (tmp_path / "foreign.txt").write_text("foreign")
    with pytest.raises(R.LifecycleError, match="unexpected top-level"):
        R._assert_known_artifact_layout(tmp_path, allow_staging=False)
    (tmp_path / "foreign.txt").unlink()
    fields = tmp_path / "fields" / R.DEVELOPMENT_NAMES[0]
    fields.mkdir(parents=True)
    for name in ("initial.npz", "base.npz", "fit_history.json", "base_record.json"):
        (fields / name).write_bytes(b"placeholder")
    (fields / "foreign.bin").write_bytes(b"foreign")
    with pytest.raises(R.LifecycleError, match="unexpected artifact inventory"):
        R._assert_known_artifact_layout(tmp_path, allow_staging=False)


def test_run_invocation_journal_is_sealed_and_must_cover_cells(tmp_path):
    cell_hash = "c" * 64
    invocation = R._seal(
        {
            "schema": "structsplat.comp008.run-dev-invocation.v1",
            "protocol": R.PROTOCOL,
            "stage": "run-dev",
            "preflight_binding_sha256": "a" * 64,
            "targets_sha256": "b" * 64,
            "requested_image_id": R.DEVELOPMENT_NAMES[0],
            "validated_or_completed_cells": [cell_hash],
            "unsealed_staging_repairs": [],
            "truncated_tail_repairs": [],
            "stage_process": {
                "started_utc": "2026-07-16T00:00:00Z",
                "ended_utc": "2026-07-16T00:00:01Z",
                "wall_seconds": 1.0,
                "pid": 1,
                "parent_pid": 0,
                "working_directory": "/synthetic",
                "python_executable": "/usr/bin/python",
                "invocation_argv": ["python", "-m", "benchmarks.sgi_entropy_oracle_run"],
            },
        },
        "run_invocation_sha256",
    )
    R._append_jsonl(tmp_path / R.RUN_INVOCATIONS_NAME, invocation)
    loaded = R._load_run_invocations(
        tmp_path,
        preflight_binding_sha256="a" * 64,
        targets_sha256="b" * 64,
    )
    R._verify_run_invocation_coverage(loaded, [{"cell_record_sha256": cell_hash}])
    with pytest.raises(R.LifecycleError, match="coverage"):
        R._verify_run_invocation_coverage(loaded, [{"cell_record_sha256": "d" * 64}])


def test_run_dev_rejects_confirmation_id_before_any_pixel_decode(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "verify_preflight", lambda *args, **kwargs: {"binding_sha256": "a" * 64})
    monkeypatch.setattr(
        R,
        "_load_target_manifest",
        lambda path: {"preflight_binding_sha256": "a" * 64, "targets": []},
    )
    with pytest.raises(R.LifecycleError, match="non-development"):
        R.run_development(
            tmp_path, image_id=A.CONFIRMATION_NAMES[0], environment_fn=_fake_environment
        )


def test_analyze_rebuilds_every_row_from_blobs_and_uses_frozen_image_order(tmp_path, monkeypatch):
    cells = _scientific_blobs(tmp_path)
    monkeypatch.setattr(R, "verify_preflight", lambda *args, **kwargs: {"binding_sha256": "a" * 64})
    monkeypatch.setattr(R, "_assert_known_artifact_layout", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        R,
        "_load_target_manifest",
        lambda path: {"targets_sha256": "b" * 64, "preflight_binding_sha256": "a" * 64},
    )
    monkeypatch.setattr(R, "_complete_cells", lambda path: cells)
    monkeypatch.setattr(R, "_load_run_invocations", lambda *args, **kwargs: [])
    monkeypatch.setattr(R, "_verify_run_invocation_coverage", lambda *args, **kwargs: None)
    result = R.analyze(tmp_path, environment_fn=_fake_environment)
    assert result["scientific_scope"] == "necessary-rate-lower-bound-only"
    for tuple_result in result["tuple_results"]:
        assert tuple(tuple_result["image_ids"]) == O.FROZEN_IMAGE_IDS
    analysis_record = R._read_object(tmp_path / R.ANALYSIS_RECORD_NAME)
    assert analysis_record["stage_process"]["started_utc"]
    assert analysis_record["stage_process"]["ended_utc"]


def test_analyze_does_not_trust_a_resealed_journal_row(tmp_path, monkeypatch):
    cells = _scientific_blobs(tmp_path)
    forged = copy.deepcopy(cells)
    forged[0]["oracle_row"]["ratio_numerator"] += 1
    forged[0]["oracle_row"] = O.seal_record(forged[0]["oracle_row"])
    monkeypatch.setattr(R, "verify_preflight", lambda *args, **kwargs: {"binding_sha256": "a" * 64})
    monkeypatch.setattr(R, "_assert_known_artifact_layout", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        R,
        "_load_target_manifest",
        lambda path: {"targets_sha256": "b" * 64, "preflight_binding_sha256": "a" * 64},
    )
    monkeypatch.setattr(R, "_complete_cells", lambda path: forged)
    monkeypatch.setattr(R, "_load_run_invocations", lambda *args, **kwargs: [])
    monkeypatch.setattr(R, "_verify_run_invocation_coverage", lambda *args, **kwargs: None)
    with pytest.raises(R.LifecycleError, match="differs from its persisted SSPL1"):
        R.analyze(tmp_path, environment_fn=_fake_environment)


def test_replay_reparses_streams_without_loading_target_pixels(tmp_path, monkeypatch):
    cells = _scientific_blobs(tmp_path)
    monkeypatch.setattr(R, "verify_preflight", lambda *args, **kwargs: {"binding_sha256": "a" * 64})
    monkeypatch.setattr(R, "_assert_known_artifact_layout", lambda *args, **kwargs: None)
    monkeypatch.setattr(R, "_load_run_invocations", lambda *args, **kwargs: [])
    monkeypatch.setattr(R, "_verify_run_invocation_coverage", lambda *args, **kwargs: None)

    def target_manifest(path: Path, *, decode_pixels: bool = True):
        assert decode_pixels is False
        return {"targets_sha256": "b" * 64, "preflight_binding_sha256": "a" * 64}

    monkeypatch.setattr(R, "_load_target_manifest", target_manifest)
    monkeypatch.setattr(R, "_complete_cells", lambda path: cells)
    # Analyze with a direct target-manifest test double, then switch to the no-decode assertion.
    monkeypatch.setattr(
        R,
        "_load_target_manifest",
        lambda path: {"targets_sha256": "b" * 64, "preflight_binding_sha256": "a" * 64},
    )
    R.analyze(tmp_path, environment_fn=_fake_environment)
    monkeypatch.setattr(R, "_load_target_manifest", target_manifest)
    result = R.replay(tmp_path, environment_fn=_fake_environment)
    assert result["ok"] is True
    assert result["oracle_rows_aggregates_gates_and_decision_exact"] is True
    assert result["pixel_payloads_decoded"] is False
    assert result["confirmation_payloads_accessed"] is False
    assert result["stage_process"]["started_utc"]
    assert result["stage_process"]["ended_utc"]
