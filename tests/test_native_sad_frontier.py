from __future__ import annotations

import inspect
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import pytest

from benchmarks import fair_density_control_compare as fair
from benchmarks import native_sad_frontier as B
from structsplat.config import FitConfig


def _native_row(repeat: int, count: int) -> dict:
    return {"status": "ok", "repeat": repeat, "sad_final_active_sites": count}


def test_protocol_v6_matrix_claim_and_predecessor_constants_are_frozen() -> None:
    assert B.PROTOCOL == "bench016-native-sad-frontier-v6"
    assert B.SCHEMA_VERSION == 6
    assert B.NATIVE_REPEATS == (0, 1, 2)
    assert B.SS_ARMS == ("ss_same_n", "ss_equal_terminal_coordinates")
    assert B.BOOTSTRAP_SEED == 0
    assert B.COMMON_AUC_START == 1 / 4000
    assert B.V4_BINDING_FILE_SHA256 == (
        "473f458ec5d3729829dd34b371ab7e6b82f5d87f2e0d57e140afb1b69c65ce71"
    )
    assert B.V4_INVALIDITY_SHA256 == (
        "a3b9c696ed0fc0d574f4e177353ffec6eb5ba2600aafca640425a33a35d2f447"
    )
    assert B.V4_INVALIDITY_BUNDLE_MANIFEST_SHA256 == (
        "338220bc9b1cf8a96508bb5be7d23a8c6f2217b1dd3aeab3b9ae0ff7bdf9f1f0"
    )
    assert B.V5_BINDING_FILE_SHA256 == (
        "a87aeed30594bf12f7fe711df332ef794c13e589c38fe7ff103fa90fe8b1b6f3"
    )
    assert B.V5_INVALIDITY_SHA256 == (
        "333bd6eeb13bc4931e46808f3dbb10e4fa63ddeafe03481378002b07abce39b4"
    )
    assert B.V5_INVALIDITY_BUNDLE_MANIFEST_SHA256 == (
        "8eac6e6210de179c1a39251051f38102238d9a5634489a160709b106a5e664c7"
    )
    assert B.V5_INVALIDITY_BUNDLE_FILE_COUNT == 165
    assert B.V5_INVALIDITY_BUNDLE_TOTAL_BYTES == 31_140_605


def test_parse_sad_trace_and_native_post_update_clock() -> None:
    lines = []
    iterations = [*range(0, 4000, 20), 3999]
    for iteration in iterations:
        lines.append(
            f"Iter {iteration:4d} | PSNR: {20 + iteration / 1000:.2f} dB | "
            f"Active: {128000 - iteration}/128000 | 10.0 it/s | {iteration + 1:.1f}s"
        )
    trace = B.parse_sad_trace("\n".join(lines))
    aligned = B.native_progress_trace(trace)

    assert [row["iteration"] for row in trace] == iterations
    assert aligned[0]["progress"] == 1 / 4000
    assert aligned[1]["progress"] == 21 / 4000
    assert aligned[-1] == {"progress": 1.0, "psnr": 24.0}


def test_parse_sad_trace_rejects_missing_terminal() -> None:
    stdout = "Iter    0 | PSNR: 10.00 dB | Active: 128000/128000 | 1.0 it/s | 1.0s"
    with pytest.raises(RuntimeError, match="incomplete"):
        B.parse_sad_trace(stdout)


def test_structsplat_trace_uses_terminal_trajectory_not_selected_quality() -> None:
    history = [
        {"iteration": 0, "psnr": 10.0},
        {"iteration": 20, "psnr": 20.0},
        {"iteration": 3999, "psnr": 30.0},
    ]
    trace = B.structsplat_progress_trace(history, trajectory_terminal_psnr=27.0)

    assert trace[0] == {"progress": 0.0, "psnr": 10.0}
    assert trace[-1] == {"progress": 1.0, "psnr": 27.0}
    assert trace[-1]["psnr"] != 99.0  # a hypothetical selected checkpoint cannot enter AUC


def test_auc_restricts_to_common_closed_interval_and_normalizes_length() -> None:
    constant = [
        {"progress": 0.0, "psnr": 7.0},
        {"progress": 1 / 4000, "psnr": 20.0},
        {"progress": 1.0, "psnr": 20.0},
    ]
    assert B.normalized_psnr_auc(constant) == pytest.approx(20.0)

    native = [
        {"progress": 1 / 4000, "psnr": 10.0},
        {"progress": 1.0, "psnr": 30.0},
    ]
    control = [
        {"progress": 0.0, "psnr": 0.0},
        {"progress": 0.5, "psnr": 20.0},
        {"progress": 1.0, "psnr": 30.0},
    ]
    aggregate = B.aggregate_traces([native, control])
    assert aggregate[0]["progress"] == 1 / 4000
    assert aggregate[-1]["progress"] == 1.0
    assert all(
        left["progress"] < right["progress"]
        for left, right in zip(aggregate, aggregate[1:])
    )


def test_conservative_control_budgets_use_max_native_count_and_ceil_coordinates() -> None:
    rows = [_native_row(0, 101), _native_row(1, 103), _native_row(2, 102)]
    budgets = B.conservative_control_budgets(rows)

    assert budgets["ss_same_n"] == 103
    assert budgets["ss_equal_terminal_coordinates"] == 129
    slack = 8 * budgets["ss_equal_terminal_coordinates"] - 10 * 103
    assert slack == 2
    assert slack in range(8)


def test_conservative_control_budgets_require_all_three_repeats() -> None:
    with pytest.raises(ValueError, match="exactly three"):
        B.conservative_control_budgets([_native_row(0, 100), _native_row(1, 100)])


def test_frozen_fair_density_signature_has_exact_cadence_and_five_additions() -> None:
    final_count = 1_003
    start_count = max(16, round(0.5 * final_count))
    difference = final_count - start_count
    base = FitConfig(
        iters=4000,
        renderer="cuda",
        pixel_loss="l1",
        ssim_weight=0.3,
        log_every=20,
    )
    signature = fair._method_fit_signature(  # noqa: SLF001
        B.SS_METHOD,
        base,
        final_count,
        start_count,
        B.SS_GROWTH_WAVES,
    )

    assert difference >= 21
    assert signature.split_every == 666
    assert signature.split_count == int(np.ceil(difference / 5))
    assert signature.max_gaussians == final_count
    assert signature.split_schedule_stops_at_max is False
    assert signature.log_every == 20


def test_worker_source_enforces_exact_growth_event_contract() -> None:
    source = inspect.getsource(B._run_control_worker)  # noqa: SLF001
    assert "difference < 21" in source
    assert "expected_split_iters = [665, 1331, 1997, 2663, 3329]" in source
    assert "len(split_events) != 5" in source
    assert 'sum(int(event["added"]) for event in split_events) != difference' in source
    assert '"log_every": LOG_FREQUENCY' in source
    assert '"split_schedule_stops_at_max": False' in source


def test_target_derivation_uses_round_lanczos_and_persists_hashes(tmp_path: Path) -> None:
    source = tmp_path / "0057.png"
    pixels = np.arange(3 * 5 * 3, dtype=np.uint8).reshape(3, 5, 3)
    Image.fromarray(pixels, mode="RGB").save(source)
    target = tmp_path / "target.png"

    record = B._prepare_one_target(source, target, "0057")  # noqa: SLF001

    assert (record["derived_width"], record["derived_height"]) == (768, 461)
    assert record["source_orientation"] == record["derived_orientation"] == "landscape"
    assert record["source_byte_sha256"] == B._sha256_file(source)  # noqa: SLF001
    assert record["derived_byte_sha256"] == B._sha256_file(target)  # noqa: SLF001
    assert record["pixel_count"] == 768 * 461


def test_bootstrap_is_seed_zero_deterministic_and_uses_linear_quantiles() -> None:
    effects = [float(value) for value in range(8)]
    first = B._bootstrap_effects(effects, seed=0, replicates=1000)  # noqa: SLF001
    second = B._bootstrap_effects(effects, seed=0, replicates=1000)  # noqa: SLF001

    assert first == second
    assert first["seed"] == 0
    assert first["rng"] == "numpy.PCG64"
    assert len(first["indices_sha256"]) == 64


def test_artifact_manifest_excludes_lifecycle_and_detects_tampering(tmp_path: Path) -> None:
    (tmp_path / "payload.bin").write_bytes(b"bound")
    (tmp_path / "replay.json").write_text("{}", encoding="utf-8")
    manifest = B._artifact_manifest(tmp_path)  # noqa: SLF001

    assert [row["path"] for row in manifest["files"]] == ["payload.bin"]
    assert B._validate_artifact_manifest(tmp_path, manifest)  # noqa: SLF001
    (tmp_path / "payload.bin").write_bytes(b"tampered")
    assert not B._validate_artifact_manifest(tmp_path, manifest)  # noqa: SLF001


def test_monitored_process_persists_timestamped_rss_samples(tmp_path: Path) -> None:
    stdout = tmp_path / "stdout.txt"
    resources = tmp_path / "resources.json"
    result = B._timed_command(  # noqa: SLF001
        [sys.executable, "-c", "import time; x=bytearray(1000000); time.sleep(.08)"],
        cwd=tmp_path,
        env=dict(**__import__("os").environ),
        stdout_path=stdout,
        resource_path=resources,
    )
    payload = json.loads(resources.read_text(encoding="utf-8"))

    assert result["returncode"] == 0
    assert result["peak_rss_bytes"] is not None
    assert len(payload["rss_samples"]) >= 2
    assert all("elapsed_ns" in row and "bytes" in row for row in payload["rss_samples"])
    assert result["resource_sha256"] == B._sha256_file(resources)  # noqa: SLF001


def test_control_fit_artifact_and_fresh_cold_workers_are_separate_process_stages() -> None:
    fit_command = B._control_worker_command(  # noqa: SLF001
        Path("contract.json"), Path("cell")
    )
    artifact_command = B._control_artifact_worker_command(  # noqa: SLF001
        Path("out"), Path("cell")
    )
    cold_command = B._control_cold_worker_command(Path("cold_contract.json"))  # noqa: SLF001
    worker_source = inspect.getsource(B._run_control_worker)  # noqa: SLF001
    artifact_source = inspect.getsource(B._run_control_artifact_worker)  # noqa: SLF001
    cold_source = inspect.getsource(B._run_control_cold_worker)  # noqa: SLF001

    assert "_control-worker" in fit_command
    assert "_control-artifact-worker" in artifact_command
    assert "_control-cold-worker" in cold_command
    assert "codec.encode" not in worker_source
    assert "_prepared_structsplat_timing" not in worker_source
    assert "_validate_binding" not in worker_source
    assert "_validate_targets" not in worker_source
    assert "codec.encode" in artifact_source
    assert "decode_and_render" not in artifact_source
    assert "decode_and_render" in cold_source
    assert "codec.encode" not in cold_source
    assert "_validate_binding" not in cold_source
    assert "_validate_targets" not in cold_source
    assert cold_source.index("started = time.perf_counter()") < cold_source.index(
        "stream_path.read_bytes()"
    )
    assert cold_source.index("torch.cuda.synchronize()") < cold_source.index(
        "cold_seconds = time.perf_counter() - started"
    )


def test_cli_exposes_resumable_v6_stages() -> None:
    parser = B.build_parser()
    for command in (
        ["preflight", "--outdir", "x"],
        ["prepare-targets", "--outdir", "x", "--source-dir", "y"],
        ["run-sad", "--outdir", "x"],
        ["run-structsplat", "--outdir", "x"],
        ["analyze", "--outdir", "x"],
        ["replay", "--outdir", "x"],
    ):
        assert parser.parse_args(command).stage == command[0]


def test_preflight_binds_acquisition_helper_and_never_opens_source_dir() -> None:
    sources = B._source_files()  # noqa: SLF001
    assert {
        "adapter_tests",
        "benchmarks_package_init",
        "div2k_subset_acquire",
        "storage_budget",
        "structsplat_package_init",
        "density",
    } <= sources.keys()
    assert sources["storage_budget"] == (
        B.ROOT / "benchmarks" / "storage_budget.py"
    ).resolve()
    assert sources["density"] == (B.ROOT / "src" / "structsplat" / "density.py").resolve()
    source = inspect.getsource(B.run_preflight)
    assert "source_dir.iterdir" not in source
    assert "Image.open" not in source


def test_central_metrics_are_forced_to_cpu(tmp_path: Path) -> None:
    target = tmp_path / "target.png"
    reconstruction = tmp_path / "reconstruction.png"
    pixels = np.full((16, 16, 3), 127, dtype=np.uint8)
    Image.fromarray(pixels, mode="RGB").save(target)
    Image.fromarray(pixels, mode="RGB").save(reconstruction)

    metrics = B._central_metrics(  # noqa: SLF001
        target,
        reconstruction,
        require_lpips=True,
        lpips_function=lambda pred, expected: float(pred.device.type == expected.device.type == "cpu"),
    )
    assert metrics["central_lpips_alex"] == 1.0


def test_v6_binds_sealed_v4_and_v5_no_decisions_without_claiming_pre_access() -> None:
    for artifact_dir, binding_hash, invalidity_hash, bundle_hash in (
        (
            B.V4_ARTIFACT_DIR,
            B.V4_BINDING_FILE_SHA256,
            B.V4_INVALIDITY_SHA256,
            B.V4_INVALIDITY_BUNDLE_MANIFEST_SHA256,
        ),
        (
            B.V5_ARTIFACT_DIR,
            B.V5_BINDING_FILE_SHA256,
            B.V5_INVALIDITY_SHA256,
            B.V5_INVALIDITY_BUNDLE_MANIFEST_SHA256,
        ),
    ):
        invalidity_path = artifact_dir / "invalidity.json"
        binding_path = artifact_dir / "binding.json"
        bundle_path = artifact_dir / "invalidity_bundle_manifest.json"
        assert B._sha256_file(invalidity_path) == invalidity_hash  # noqa: SLF001
        assert B._sha256_file(binding_path) == binding_hash  # noqa: SLF001
        assert B._sha256_file(bundle_path) == bundle_hash  # noqa: SLF001
        invalidity = json.loads(invalidity_path.read_text(encoding="utf-8"))
        bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
        members = {record["path"]: record["sha256"] for record in bundle["files"]}
        assert invalidity["status"] == "invalid_no_decision"
        assert invalidity["scientific_gate_evaluation_performed"] is False
        assert members["binding.json"] == binding_hash
        assert members["invalidity.json"] == invalidity_hash

    source = inspect.getsource(B._binding_payload)  # noqa: SLF001
    assert '"created_before_any_scientific_target_access": False' in source
    assert '"prior_scientific_target_access_disclosed": True' in source
    assert '"predecessor_v4_invalidity": predecessor_v4' in source
    assert '"predecessor_v5_invalidity": predecessor_v5' in source


def test_native_training_replay_diagnostics_have_no_numeric_validity_gate() -> None:
    training = np.zeros((2, 2, 3), dtype=np.uint8)
    primary = training.copy()
    primary[0, 0] = [1, 2, 9]
    training_metrics = {
        "central_psnr": 20.0,
        "central_ssim": 0.8,
        "central_proxy_ms_ssim": 0.7,
        "central_lpips_alex": 0.2,
    }
    primary_metrics = {
        "central_psnr": 21.0,
        "central_ssim": 0.81,
        "central_proxy_ms_ssim": 0.72,
        "central_lpips_alex": 0.19,
    }

    diagnostic = B._native_training_replay_diagnostics(  # noqa: SLF001
        training, primary, training_metrics, primary_metrics
    )

    assert diagnostic["max_uint8_difference"] == 9
    assert diagnostic["per_channel_max_uint8_difference"] == {"r": 1, "g": 2, "b": 9}
    assert diagnostic["differing_pixel_fraction"] == pytest.approx(0.25)
    assert diagnostic["differing_channel_code_fraction"] == pytest.approx(0.25)
    assert diagnostic["differing_pixel_count"] == 1
    assert diagnostic["differing_channel_code_count"] == 3
    assert diagnostic["primary_minus_training_metric_delta"]["central_psnr"] == 1.0
    assert "pass" not in diagnostic
    assert "threshold" not in diagnostic

    source = inspect.getsource(B.run_sad)
    assert '"diagnostic_training_metrics": diagnostic_training_metrics' in source
    assert "**primary_metrics" in source
    assert "**diagnostic_training_metrics" not in source
    assert "max_pixel_difference >" not in source
    assert "cold_psnr_drift >" not in source


def _prepared_replay_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[dict, np.ndarray, dict, Path, Path, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    primary = np.full((2, 3, 3), 127, dtype=np.uint8)
    target = tmp_path / "target.png"
    primary_path = tmp_path / "primary.png"
    primary_stdout = tmp_path / "primary_stdout.txt"
    sites = tmp_path / "sites.txt"
    binary = tmp_path / "sad_cuda"
    Image.fromarray(primary, mode="RGB").save(target)
    Image.fromarray(primary, mode="RGB").save(primary_path)
    primary_stdout.write_text("primary", encoding="utf-8")
    sites.write_text(
        "# Total sites: 1\n0 0 0 0 0 0 0 1 0 0\n",
        encoding="utf-8",
    )

    def fake_metrics(_target: Path, reconstruction: Path) -> dict[str, float]:
        pixels = np.asarray(Image.open(reconstruction).convert("RGB"), dtype=np.uint8)
        value = float(np.mean(pixels, dtype=np.float64) / 255.0)
        return {
            "central_psnr": 20.0 + value,
            "central_ssim": 0.7 + value / 10.0,
            "central_proxy_ms_ssim": 0.6 + value / 10.0,
            "central_lpips_alex": 0.3 - value / 10.0,
        }

    monkeypatch.setattr(B, "_central_metrics", fake_metrics)
    primary_metrics = fake_metrics(target, primary_path)
    records = []
    for sample_index in range(B.RENDER_WARMUPS + B.RENDER_SAMPLES):
        reconstruction = tmp_path / f"sample_{sample_index:02d}.png"
        stdout = tmp_path / f"sample_{sample_index:02d}_stdout.txt"
        sample = primary.copy()
        if sample_index == 0:
            sample[0, 0, 1] += 1  # valid upstream variation, never a numeric integrity failure
        Image.fromarray(sample, mode="RGB").save(reconstruction)
        stdout.write_text(
            "Loaded 1 sites\n"
            "Render size: 3x2\n"
            "Candidate build: 1.000 ms | Render: 2.000 ms\n"
            f"Saved: {reconstruction.resolve()}\n",
            encoding="utf-8",
        )
        command = B._cold_sad_command(binary, sites, reconstruction, 3, 2)  # noqa: SLF001
        semantics = B._parse_render_stdout_semantics(  # noqa: SLF001
            stdout.read_text(encoding="utf-8"),
            expected_site_count=1,
            expected_width=3,
            expected_height=2,
            expected_saved_path=reconstruction,
        )
        sample_metrics = fake_metrics(target, reconstruction)
        variation = B._reconstruction_variation_diagnostics(  # noqa: SLF001
            primary,
            sample,
            primary_metrics,
            sample_metrics,
            metric_delta_name="sample_minus_primary_metric_delta",
        )
        started_ns = (sample_index + 1) * 10_000_000
        ended_ns = started_ns + 2_000_000
        records.append(
            {
                "sample_index": sample_index,
                "invocation_ordinal_after_primary": sample_index + 1,
                "sample_role": (
                    "discarded_warmup"
                    if sample_index < B.RENDER_WARMUPS
                    else "retained_timing_sample"
                ),
                "reconstruction_path": str(reconstruction),
                "reconstruction_sha256": B._sha256_file(reconstruction),  # noqa: SLF001
                "reconstruction_pixel_sha256": B._pixel_sha256(sample),  # noqa: SLF001
                "primary_replay_pixel_sha256": B._pixel_sha256(primary),  # noqa: SLF001
                "command_argv": command,
                "command_sha256": B._command_sha256(command),  # noqa: SLF001
                "cwd": str(tmp_path.resolve()),
                "config_path": str((tmp_path / "training_config.json").resolve()),
                "sites_sha256": B._sha256_file(sites),  # noqa: SLF001
                "stdout_path": str(stdout),
                "stdout_sha256": B._sha256_file(stdout),  # noqa: SLF001
                "stdout_semantics": semantics,
                "returncode": 0,
                "process_started_monotonic_ns": started_ns,
                "process_ended_monotonic_ns": ended_ns,
                "full_process_wall_ms": 2.0,
                "candidate_build_ms": 1.0,
                "render_ms": 2.0,
                "central_metrics": sample_metrics,
                "primary_replay_variation_diagnostics": variation,
            }
        )
    row = {
        "prepared_render_records": records,
        "primary_replay_reconstruction_path": str(primary_path),
        "primary_replay_stdout_path": str(primary_stdout),
        "sites_path": str(sites),
        "sad_final_active_sites": 1,
        "width": 3,
        "height": 2,
    }
    return row, primary, primary_metrics, target, binary, tmp_path


def _validate_prepared_fixture(
    fixture: tuple[dict, np.ndarray, dict, Path, Path, Path],
) -> tuple[list[float], list[float], dict]:
    row, primary, primary_metrics, target, binary, cwd = fixture
    return B._validate_native_prepared_replays(  # noqa: SLF001
        row,
        primary,
        primary_metrics,
        target_path=target,
        binary=binary,
        cwd=cwd,
    )


def test_native_prepared_replay_accepts_variation_and_rejects_hash_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _prepared_replay_fixture(tmp_path, monkeypatch)
    row = fixture[0]

    candidate, render, aggregate = _validate_prepared_fixture(fixture)
    assert len(row["prepared_render_records"]) == 53
    assert candidate == [1.0] * 51
    assert render == [2.0] * 51
    assert aggregate["exact_primary_pixel_match_count"] == 52
    assert aggregate["unique_pixel_hash_count"] == 2
    assert aggregate["max_uint8_difference"] == 1

    row["prepared_render_records"][17]["reconstruction_pixel_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="shape/pixel-hash drift"):
        _validate_prepared_fixture(fixture)


def test_native_prepared_replay_rejects_alias_and_command_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _prepared_replay_fixture(tmp_path, monkeypatch)
    row = fixture[0]
    row["prepared_render_records"][1]["reconstruction_path"] = row[
        "prepared_render_records"
    ][0]["reconstruction_path"]

    with pytest.raises(RuntimeError, match="paths must be unique"):
        _validate_prepared_fixture(fixture)

    fixture = _prepared_replay_fixture(tmp_path / "command", monkeypatch)
    fixture[0]["prepared_render_records"][7]["command_argv"] = ["wrong"]
    with pytest.raises(RuntimeError, match="identity/file/hash drift"):
        _validate_prepared_fixture(fixture)


def test_native_replay_semantics_and_process_boundaries_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "output.png"
    valid = (
        "Loaded 1 sites\nRender size: 3x2\n"
        "Candidate build: 1.000 ms | Render: 2.000 ms\n"
        f"Saved: {output.resolve()}\n"
    )
    semantics = B._parse_render_stdout_semantics(  # noqa: SLF001
        valid,
        expected_site_count=1,
        expected_width=3,
        expected_height=2,
        expected_saved_path=output,
    )
    assert semantics["loaded_site_count"] == 1
    with pytest.raises(RuntimeError, match="site-count semantics"):
        B._parse_render_stdout_semantics(  # noqa: SLF001
            valid.replace("Loaded 1 sites", "Loaded 2 sites"),
            expected_site_count=1,
            expected_width=3,
            expected_height=2,
            expected_saved_path=output,
        )
    with pytest.raises(RuntimeError, match="saved-path semantics"):
        B._parse_render_stdout_semantics(  # noqa: SLF001
            valid.replace(str(output.resolve()), str((tmp_path / "wrong.png").resolve())),
            expected_site_count=1,
            expected_width=3,
            expected_height=2,
            expected_saved_path=output,
        )

    fixture = _prepared_replay_fixture(tmp_path / "boundary", monkeypatch)
    fixture[0]["prepared_render_records"][9]["process_ended_monotonic_ns"] = -1
    with pytest.raises(RuntimeError, match="process-boundary drift"):
        _validate_prepared_fixture(fixture)


def test_role_fixed_primary_precedes_timing_and_variation_has_no_numeric_gate() -> None:
    source = inspect.getsource(B.run_sad)
    primary_launch = source.index("primary_timing = _wall_command")
    timing_loop = source.index("for sample_index in range")
    assert primary_launch < timing_loop
    assert '"primary_replay_retry_count": 0' in source
    assert "sample_pixel_sha256 != primary_pixel_sha256" not in source
    validator_source = inspect.getsource(B._validate_native_prepared_replays)  # noqa: SLF001
    assert "sample_pixel_sha256 !=" not in validator_source
    assert "max_uint8_difference >" not in validator_source
    assert "differing_pixel_fraction >" not in validator_source


def test_jfa_seed_collision_diagnostics_match_cuda_cast_and_clamp(tmp_path: Path) -> None:
    sites = tmp_path / "sites.txt"
    sites.write_text(
        "# Total sites: 4\n"
        "1.1 2.1 0 0 0 0 0 1 0 0\n"
        "1.9 2.8 0 0 0 0 0 1 0 0\n"
        "4.9 3.9 0 0 0 0 0 1 0 0\n"
        "99 99 0 0 0 0 0 1 0 0\n",
        encoding="utf-8",
    )
    result = B._jfa_seed_collision_diagnostics(sites, 5, 4, 1)  # noqa: SLF001
    assert result["site_count"] == 4
    assert result["collision_home_pixel_count"] == 2
    assert result["sites_in_collision_home_pixels"] == 4
    assert result["excess_colliding_site_writes"] == 2
    assert result["max_home_pixel_multiplicity"] == 2


def test_v6_auc_and_representation_scopes_are_explicit_and_non_deployable() -> None:
    protocol_source = inspect.getsource(B._protocol_spec)  # noqa: SLF001
    analysis_source = inspect.getsource(B._analyze_persisted)  # noqa: SLF001
    assert '"auc_scope": "internal_training_state_normalized_iteration"' in protocol_source
    assert "median_internal_training_state_auc_gain_at_least_0p5db" in analysis_source
    assert "median_auc_gain_at_least_0p5db" not in analysis_source
    assert '"sad_recipient_replayable": True' in protocol_source
    assert '"sad_deployable": False' in protocol_source
    assert '"sad_self_contained": False' in protocol_source
    assert '"sad_compressed_representation": False' in protocol_source
    assert "terminate BENCH-016 as invalid/no-decision; no v7 repair" in protocol_source


def test_codec_config_remains_unmodified_in_postfit_worker() -> None:
    source = inspect.getsource(B._run_control_artifact_worker)  # noqa: SLF001
    assert "codec_config = codec.CodecConfig()" in source
    assert "qat" not in source.lower()


def test_resume_keys_require_exact_protocol_binding_and_targets(tmp_path: Path) -> None:
    ledger = tmp_path / "rows.jsonl"
    rows = [
        {
            "status": "ok",
            "protocol": B.PROTOCOL,
            "binding_file_sha256": "a" * 64,
            "targets_file_sha256": "b" * 64,
            "image_id": "0057",
            "repeat": 0,
        },
        {
            "status": "ok",
            "protocol": "stale",
            "binding_file_sha256": "a" * 64,
            "targets_file_sha256": "b" * 64,
            "image_id": "0171",
            "repeat": 0,
        },
        {
            "status": "ok",
            "protocol": B.PROTOCOL,
            "binding_file_sha256": "c" * 64,
            "targets_file_sha256": "b" * 64,
            "image_id": "0285",
            "repeat": 0,
        },
    ]
    ledger.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    keys = B._existing_success_keys(  # noqa: SLF001
        ledger,
        ("image_id", "repeat"),
        binding_file_sha256="a" * 64,
        targets_file_sha256="b" * 64,
    )

    assert keys == {("0057", 0)}


def test_native_initial_state_tamper_is_rejected() -> None:
    binding = {"upstream": {"resolved_config": {"DEFAULT_SITES": 128_000}}}
    trace = [{"capacity_sites": 128_000}, {"capacity_sites": 64_000}]
    row = {
        "sad_initial_active_sites": 128_000,
        "sad_allocated_capacity_sites_exposed": 128_000,
    }
    B._validate_native_initial_capacity_state(row, binding, trace)  # noqa: SLF001

    row["sad_initial_active_sites"] = 127_999
    with pytest.raises(RuntimeError, match="initial/capacity"):
        B._validate_native_initial_capacity_state(row, binding, trace)  # noqa: SLF001


def test_v6_native_primary_residual_is_never_named_as_isolated_parse_upload() -> None:
    source = inspect.getsource(B.run_sad)
    assert "primary_replay_non_kernel_residual_upper_bound_ms" in source
    assert "cold_process_parse_upload_overhead_upper_ms" not in source
    assert "txt_parse_and_host_to_device_upload" in source


def test_preflight_refuses_nonempty_output_before_environment_or_source_checks(
    tmp_path: Path,
) -> None:
    (tmp_path / "foreign.txt").write_text("occupied", encoding="utf-8")
    with pytest.raises(RuntimeError, match="empty output directory"):
        B.run_preflight(tmp_path, tmp_path / "missing-sad")


def test_replay_derivation_helpers_detect_metric_timing_and_resource_tampering(
    tmp_path: Path,
) -> None:
    metrics = {
        "central_psnr": 20.0,
        "central_ssim": 0.8,
        "central_proxy_ms_ssim": 0.7,
        "central_lpips_alex": 0.2,
    }
    B._validate_metric_derivation(dict(metrics), metrics, "test")  # noqa: SLF001
    tampered = dict(metrics, central_psnr=21.0)
    with pytest.raises(RuntimeError, match="central_psnr"):
        B._validate_metric_derivation(tampered, metrics, "test")  # noqa: SLF001

    row = {"prepared_render_ms_samples": [float(index) for index in range(51)]}
    row["prepared_render_ms_median"] = 25.0
    B._validate_prepared_timing(row, "test")  # noqa: SLF001
    row["prepared_render_ms_median"] = 24.0
    with pytest.raises(RuntimeError, match="prepared_render_ms_median"):
        B._validate_prepared_timing(row, "test")  # noqa: SLF001

    resource_path = tmp_path / "resources.json"
    resource = {
        "rss_samples": [{"bytes": 100}],
        "gpu_samples": [{"bytes": 50}],
        "peak_rss_bytes": 100,
        "peak_gpu_device_bytes": 50,
        "process_started_monotonic_ns": 1_000,
        "process_ended_monotonic_ns": 1_000_001_000,
        "process_wall_seconds": 1.0,
    }
    resource_path.write_text(json.dumps(resource), encoding="utf-8")
    resource_row = {
        "resource_samples_path": str(resource_path),
        "peak_rss_bytes": 100,
        "peak_gpu_device_bytes": 50,
        "training_wall_seconds": 1.0,
    }
    B._validate_resource_derivation(resource_row)  # noqa: SLF001
    resource_row["peak_rss_bytes"] = 99
    with pytest.raises(RuntimeError, match="peak RSS"):
        B._validate_resource_derivation(resource_row)  # noqa: SLF001
