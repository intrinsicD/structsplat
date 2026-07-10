from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from benchmarks import native_reference_compare as N


def _checkout_fingerprint(repo: Path, commit: str) -> dict:
    return {
        "repo_root": str(repo),
        "repo_commit": commit,
        "repo_tree": "tree",
        "repo_tracked_diff_sha256": hashlib.sha256(b"").hexdigest(),
        "repo_python_source_sha256": "repo-python",
        "repo_untracked_files": [],
        "gsplat_root": str(repo / "gsplat"),
        "gsplat_commit": "gsplat-commit",
        "gsplat_tree": "gsplat-tree",
        "gsplat_tracked_diff_sha256": hashlib.sha256(b"").hexdigest(),
        "gsplat_python_source_sha256": "gsplat-python",
        "gsplat_untracked_files": [],
    }


def _source_fingerprint() -> dict:
    return {
        "native_adapter_source_sha256": "adapter-source",
        "native_harness_source_sha256": "harness-source",
        "metric_protocol_source_sha256": "metric-source",
    }


def _environment_fingerprint() -> dict:
    return {
        "python_executable": str(Path(__file__).resolve()),
        "python_version": "3.test",
        "torch_version": "2.test",
        "torch_cuda_version": None,
        "numpy_version": "1.test",
        "cuda_devices": [],
        "metric_dependency_versions": {
            "lpips": None,
            "pillow": "test",
            "pytorch-msssim": "test",
            "torchvision": None,
        },
        "metric_device": "cpu",
    }


def _manifest(repo: Path, reconstruction: Path, **overrides):
    extension = repo / "gsplat" / "gsplat" / "csrc.so"
    iterations = list(range(10))
    data = {
        "schema_version": N.SCHEMA_VERSION,
        "adapter_revision": N.ADAPTER_REVISION,
        "method": N.METHOD,
        "repo_root": str(repo),
        "repo_commit": "abc123",
        **_environment_fingerprint(),
        "gsplat_csrc": str(extension),
        "gsplat_csrc_sha256": (
            hashlib.sha256(extension.read_bytes()).hexdigest()
            if extension.is_file()
            else None
        ),
        "source_path": str(reconstruction),
        "height": 12,
        "width": 16,
        "seed": 0,
        "requested_iters": 10,
        "budget_cap": 64,
        "start_gaussians": 32,
        "growth_waves_requested": 2,
        "n_gaussians": 64,
        "iterations_run": 10,
        "native_reported_psnr": 30.0,
        "selection_policy": "best_training_psnr_checkpoint",
        "selected_iter": 10,
        "init_seconds": 0.1,
        "fit_seconds": 1.0,
        "total_seconds": 1.1,
        "render_ms_mean": 1.0,
        "render_ms_median": 1.0,
        "render_ms_p10": 0.9,
        "render_ms_p90": 1.1,
        "render_fps_median": 1000.0,
        "reconstruction_npy": str(reconstruction),
        "history": {
            "iter": iterations,
            "psnr": np.linspace(20.0, 30.0, 10).tolist(),
            "elapsed": np.linspace(0.1, 1.0, 10).tolist(),
            "n_gaussians": [64] * 10,
        },
    }
    data.update(overrides)
    return data


def test_artifact_source_id_binds_canonical_path_and_content(tmp_path: Path):
    left_path = tmp_path / "left" / "foo.png"
    right_path = tmp_path / "right" / "foo.png"
    left_path.parent.mkdir()
    right_path.parent.mkdir()
    left_path.write_bytes(b"same bytes")
    right_path.write_bytes(b"same bytes")
    alias = tmp_path / "foo-alias.png"
    alias.symlink_to(left_path)
    content_hash = "a" * 64
    left = N._artifact_source_id(left_path, content_hash)
    right = N._artifact_source_id(right_path, content_hash)

    assert left.startswith("foo_")
    assert right.startswith("foo_")
    assert left != right
    assert N._artifact_source_id(alias, content_hash) == left


def test_cell_key_invalidates_pixels_sources_and_harness_revisions():
    row = {
        "schema_version": N.SCHEMA_VERSION,
        "adapter_revision": N.ADAPTER_REVISION,
        "metric_protocol_revision": N.METRIC_PROTOCOL_REVISION,
        "protocol": N.PROTOCOL,
        "method": N.METHOD,
        "source_path": "source.png",
        "source_sha256": "source-a",
        "target_sha256": "target-a",
        "target_pixel_sha256": "pixels-a",
        "native_adapter_source_sha256": "adapter-a",
        "native_harness_source_sha256": "harness-a",
        "metric_protocol_source_sha256": "metrics-a",
        **_environment_fingerprint(),
        "max_side": 160,
        "budget_cap": 640,
        "start_gaussians": 320,
        "requested_iters": 500,
        "seed": 0,
    }
    key = N._cell_key(row)
    for name, value in (
        ("source_sha256", "source-b"),
        ("target_sha256", "target-b"),
        ("target_pixel_sha256", "pixels-b"),
        ("native_adapter_source_sha256", "adapter-b"),
        ("native_harness_source_sha256", "harness-b"),
        ("metric_protocol_source_sha256", "metrics-b"),
        ("torch_version", "torch-b"),
    ):
        assert N._cell_key({**row, name: value}) != key


def test_write_outputs_compacts_stale_resume_journal(tmp_path: Path):
    (tmp_path / "metrics.jsonl").write_text(
        json.dumps({"status": "ok", "method": "stale"}) + "\n",
        encoding="utf-8",
    )

    N._write_outputs([], tmp_path, [])

    assert (tmp_path / "metrics.jsonl").read_text(encoding="utf-8") == ""


def test_validate_manifest_accepts_finite_exact_shape(tmp_path: Path):
    repo = tmp_path / "repo"
    extension = repo / "gsplat" / "gsplat" / "csrc.so"
    extension.parent.mkdir(parents=True)
    extension.write_bytes(b"native")
    reconstruction = tmp_path / "render.npy"
    np.save(reconstruction, np.full((12, 16, 3), 0.5, dtype=np.float32))

    result = N._validate_manifest(
        _manifest(repo, reconstruction),
        repo=repo,
        repo_commit="abc123",
        target_path=reconstruction,
        target_shape=(12, 16),
        budget=64,
        start_gaussians=32,
        requested_iters=10,
        seed=0,
        growth_waves=2,
        expected_extension=extension,
        expected_extension_sha256=hashlib.sha256(extension.read_bytes()).hexdigest(),
        environment_fingerprint=_environment_fingerprint(),
    )

    assert result.shape == (12, 16, 3)
    assert result.dtype == np.float32


def test_validate_manifest_rejects_cap_shape_and_extension_provenance(tmp_path: Path):
    repo = tmp_path / "repo"
    extension = repo / "gsplat" / "gsplat" / "csrc.so"
    extension.parent.mkdir(parents=True)
    extension.write_bytes(b"native")
    reconstruction = tmp_path / "render.npy"
    np.save(reconstruction, np.zeros((12, 16, 3), dtype=np.float32))

    bad_cap = _manifest(repo, reconstruction, n_gaussians=65)
    try:
        N._validate_manifest(
            bad_cap,
            repo=repo,
            repo_commit="abc123",
            target_path=reconstruction,
            target_shape=(12, 16),
            budget=64,
            start_gaussians=32,
            requested_iters=10,
            seed=0,
            growth_waves=2,
            expected_extension=extension,
            expected_extension_sha256=hashlib.sha256(extension.read_bytes()).hexdigest(),
        )
    except ValueError as exc:
        assert "exceeded count cap" in str(exc)
    else:
        raise AssertionError("count-cap violation was accepted")

    bad_shape_path = tmp_path / "bad_shape.npy"
    np.save(bad_shape_path, np.zeros((12, 15, 3), dtype=np.float32))
    try:
        N._validate_manifest(
            _manifest(repo, bad_shape_path),
            repo=repo,
            repo_commit="abc123",
            target_path=bad_shape_path,
            target_shape=(12, 16),
            budget=64,
            start_gaussians=32,
            requested_iters=10,
            seed=0,
            growth_waves=2,
            expected_extension=extension,
            expected_extension_sha256=hashlib.sha256(extension.read_bytes()).hexdigest(),
        )
    except ValueError as exc:
        assert "does not match target" in str(exc)
    else:
        raise AssertionError("shape mismatch was accepted")

    other = tmp_path / "other" / "csrc.so"
    other.parent.mkdir()
    other.write_bytes(b"wrong extension")
    try:
        N._validate_manifest(
            _manifest(
                repo,
                reconstruction,
                gsplat_csrc=str(other),
                gsplat_csrc_sha256=hashlib.sha256(other.read_bytes()).hexdigest(),
            ),
            repo=repo,
            repo_commit="abc123",
            target_path=reconstruction,
            target_shape=(12, 16),
            budget=64,
            start_gaussians=32,
            requested_iters=10,
            seed=0,
            growth_waves=2,
            expected_extension=extension,
            expected_extension_sha256=hashlib.sha256(extension.read_bytes()).hexdigest(),
        )
    except ValueError as exc:
        assert "outside the requested checkout" in str(exc)
    else:
        raise AssertionError("foreign extension provenance was accepted")


def test_validate_manifest_rejects_commit_hash_axes_and_invalid_history(tmp_path: Path):
    repo = tmp_path / "repo"
    extension = repo / "gsplat" / "gsplat" / "csrc.so"
    extension.parent.mkdir(parents=True)
    extension.write_bytes(b"native")
    reconstruction = tmp_path / "render.npy"
    np.save(reconstruction, np.zeros((12, 16, 3), dtype=np.float32))
    extension_hash = hashlib.sha256(extension.read_bytes()).hexdigest()

    def validate(manifest):
        return N._validate_manifest(
            manifest,
            repo=repo,
            repo_commit="abc123",
            target_path=reconstruction,
            target_shape=(12, 16),
            budget=64,
            start_gaussians=32,
            requested_iters=10,
            seed=0,
            growth_waves=2,
            expected_extension=extension,
            expected_extension_sha256=extension_hash,
        )

    with pytest.raises(ValueError, match="commit does not match"):
        validate(_manifest(repo, reconstruction, repo_commit="different"))
    with pytest.raises(ValueError, match="extension hash"):
        validate(_manifest(repo, reconstruction, gsplat_csrc_sha256="0" * 64))
    with pytest.raises(ValueError, match="axis seed"):
        validate(_manifest(repo, reconstruction, seed=1))
    bad_history = _manifest(repo, reconstruction)
    bad_history["history"]["elapsed"][-1] = 0.0
    with pytest.raises(ValueError, match="nondecreasing"):
        validate(bad_history)

    with pytest.raises(ValueError, match="environment torch_version"):
        N._validate_manifest(
            _manifest(repo, reconstruction, torch_version="different"),
            repo=repo,
            repo_commit="abc123",
            target_path=reconstruction,
            target_shape=(12, 16),
            budget=64,
            start_gaussians=32,
            requested_iters=10,
            seed=0,
            growth_waves=2,
            expected_extension=extension,
            expected_extension_sha256=extension_hash,
            environment_fingerprint=_environment_fingerprint(),
        )


def test_central_metrics_recompute_and_do_not_impute_lpips():
    target = np.full((16, 16, 3), 0.5, dtype=np.float32)
    reconstruction = target.copy()

    metrics = N._central_metrics(
        reconstruction,
        target,
        device="cpu",
        want_lpips=False,
    )

    assert metrics["psnr"] >= 100.0
    assert metrics["ms_ssim"] > 0.999
    assert metrics["lpips"] is None


def test_target_hits_and_resume_key_include_protocol_and_commit():
    history = {"iter": [0, 4, 9], "psnr": [20.0, 24.5, 28.0]}
    assert N._target_hits(history, [22.0, 26.0, 30.0]) == {
        "22.0": 4,
        "26.0": 9,
        "30.0": None,
    }
    row = {
        "schema_version": N.SCHEMA_VERSION,
        "adapter_revision": N.ADAPTER_REVISION,
        "metric_protocol_revision": N.METRIC_PROTOCOL_REVISION,
        "protocol": N.PROTOCOL,
        "method": N.METHOD,
        "repo_commit": "a",
        "gsplat_csrc_sha256": "hash-a",
        "source_path": "image.png",
        "max_side": 160,
        "budget_cap": 640,
        "start_gaussians": 320,
        "requested_iters": 500,
        "seed": 0,
        "growth_waves_requested": 4,
        "render_warmup": 10,
        "render_repeats": 50,
        "lpips_requested": True,
        "device": "cuda",
    }
    assert N._cell_key(row) != N._cell_key({**row, "repo_commit": "b"})
    assert N._cell_key(row) != N._cell_key({**row, "protocol": "native_default"})
    assert N._cell_key(row) != N._cell_key({**row, "growth_waves_requested": 5})
    assert N._cell_key(row) != N._cell_key({**row, "lpips_requested": False})
    assert N._cell_key(row) != N._cell_key({**row, "gsplat_csrc_sha256": "hash-b"})


def test_paired_native_comparison_reports_image_clustered_tradeoff(tmp_path: Path):
    native_rows = []
    structsplat_rows = []
    for image in ("a", "b"):
        source = str((tmp_path / f"{image}.png").resolve())
        target_path = tmp_path / f"{image}_target.png"
        target_path.write_bytes(f"target-{image}".encode())
        target_pixel_sha256 = f"pixel-hash-{image}"
        for seed in (0, 1):
            native_rows.append({
                "status": "ok",
                "method": N.METHOD,
                "source_id": f"{image}-source-id",
                "source_path": source,
                "target_path": str(target_path),
                "target_pixel_sha256": target_pixel_sha256,
                "max_side": 160,
                "budget_cap": 640,
                "start_gaussians": 320,
                "growth_waves_requested": 4,
                "requested_iters": 500,
                "seed": seed,
                "psnr": 29.0,
                "ms_ssim": 0.89,
                "lpips": 0.20,
                "auc_psnr": 23.0,
                "fit_seconds": 0.5,
                "total_seconds": 0.6,
            })
            structsplat_rows.append({
                "status": "ok",
                "method": "structsplat_best_default",
                "source_path": source,
                "target_path": str(target_path),
                "target_pixel_sha256": target_pixel_sha256,
                "max_side": 160,
                "final_budget": 640,
                "start_budget": 320,
                "growth_waves": 4,
                "iters": 500,
                "seed": seed,
                "psnr": 30.0,
                "ms_ssim": 0.90,
                "lpips": 0.18,
                "auc_psnr": 24.0,
                "fit_seconds": 1.0,
                "total_seconds": 1.1,
            })

    pairs, summary = N._paired_native_vs_structsplat(native_rows, structsplat_rows)

    assert len(pairs) == 4
    assert {pair["image"] for pair in pairs} == {"a-source-id", "b-source-id"}
    assert summary is not None
    assert summary["pairs"] == 4
    assert summary["images"] == 2
    assert summary["gain_psnr"] == -1.0
    assert summary["gain_fit_seconds"] == 0.5
    assert summary["gain_lpips"] < 0.0
    assert summary["sample_relation"] == "tradeoff"
    assert summary["supported_relation_95ci"] == "tradeoff"

    checkpoint_rows = [
        {**row, "method": "structsplat_best_checkpoint", "psnr": 31.0}
        for row in structsplat_rows
    ]
    candidate_pairs, candidate_summary = N._paired_native_vs_structsplat(
        native_rows,
        checkpoint_rows,
        baseline_method="structsplat_best_checkpoint",
    )
    assert len(candidate_pairs) == 4
    assert candidate_summary is not None
    assert candidate_summary["baseline_method"] == "structsplat_best_checkpoint"
    assert candidate_summary["gain_psnr"] == -2.0

    mismatched_start = [{**row, "start_budget": 640} for row in structsplat_rows]
    rejected_pairs, rejected_summary = N._paired_native_vs_structsplat(
        native_rows,
        mismatched_start,
        require_start_match=True,
    )
    assert rejected_pairs == []
    assert rejected_summary is None


def test_native_subprocess_failure_becomes_error_row(tmp_path: Path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    target = np.full((16, 16, 3), 0.5, dtype=np.float32)
    target_path = tmp_path / "target.png"
    target_path.write_bytes(b"placeholder")

    monkeypatch.setattr(
        N.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=7,
            stdout="",
            stderr="native import failed\n",
        ),
    )
    row = N._run_cell(
        repo=repo,
        repo_commit="abc",
        checkout_fingerprint=_checkout_fingerprint(repo, "abc"),
        source_fingerprint=_source_fingerprint(),
        environment_fingerprint=_environment_fingerprint(),
        expected_extension=repo / "gsplat" / "gsplat" / "csrc.so",
        expected_extension_sha256="unavailable-after-failed-subprocess",
        source_path=tmp_path / "source.png",
        source_sha256="source-hash",
        target_path=target_path,
        target_sha256="target-file-hash",
        target_pixel_sha256=N._pixel_sha256(target),
        target=target,
        max_side=16,
        budget=64,
        start_gaussians=32,
        iters=10,
        seed=0,
        growth_waves=2,
        render_warmup=0,
        render_repeats=1,
        device="cpu",
        want_lpips=False,
        cell_dir=tmp_path / "cell",
    )

    assert row["status"] == "error"
    assert row["target_pixel_sha256"] == N._pixel_sha256(target)
    assert "exit 7" in row["error"]
    assert Path(row["stderr_path"]).read_text(encoding="utf-8") == "native import failed\n"


def test_native_success_ingests_manifest_and_recomputes_metrics(tmp_path: Path, monkeypatch):
    repo = tmp_path / "repo"
    extension = repo / "gsplat" / "gsplat" / "csrc.so"
    extension.parent.mkdir(parents=True)
    extension.write_bytes(b"native")
    target = np.full((16, 16, 3), 0.5, dtype=np.float32)
    target_path = tmp_path / "target.png"
    target_path.write_bytes(b"placeholder")
    cell_dir = tmp_path / "cell"
    python_alias = tmp_path / "python-alias"
    python_alias.symlink_to(Path(__file__).resolve())

    def fake_run(*args, **kwargs):
        cell_dir.mkdir(parents=True, exist_ok=True)
        reconstruction = cell_dir / "reconstruction.npy"
        np.save(reconstruction, target)
        manifest = _manifest(
            repo,
            reconstruction,
            repo_commit="abc",
            source_path=str(target_path),
            height=16,
            width=16,
            fit_seconds=1.0,
            render_fps_median=1000.0,
            parameter_bpp_float32=8.0,
            reconstruction_png=str(cell_dir / "reconstruction.png"),
            python_executable=str(python_alias),
        )
        (cell_dir / "result.json").write_text(json.dumps(manifest), encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="ok\n", stderr="")

    monkeypatch.setattr(N.subprocess, "run", fake_run)
    row = N._run_cell(
        repo=repo,
        repo_commit="abc",
        checkout_fingerprint=_checkout_fingerprint(repo, "abc"),
        source_fingerprint=_source_fingerprint(),
        environment_fingerprint=_environment_fingerprint(),
        expected_extension=extension,
        expected_extension_sha256=hashlib.sha256(extension.read_bytes()).hexdigest(),
        source_path=tmp_path / "source.png",
        source_sha256="source-hash",
        target_path=target_path,
        target_sha256="target-file-hash",
        target_pixel_sha256=N._pixel_sha256(target),
        target=target,
        max_side=16,
        budget=64,
        start_gaussians=32,
        iters=10,
        seed=0,
        growth_waves=2,
        render_warmup=0,
        render_repeats=1,
        device="cpu",
        want_lpips=False,
        cell_dir=cell_dir,
    )

    assert row["status"] == "ok"
    assert row["native_adapter_source_sha256"] == "adapter-source"
    assert row["native_harness_source_sha256"] == "harness-source"
    assert row["metric_protocol_source_sha256"] == "metric-source"
    assert row["torch_version"] == "2.test"
    assert row["python_executable"] == _environment_fingerprint()["python_executable"]
    assert row["psnr"] >= 100.0
    assert row["lpips"] is None
    assert row["auc_psnr"] == 25.0
    assert row["iters_to_targets"]["28.0"] == 8
    structsplat = {
        "status": "ok",
        "method": "structsplat_best_default",
        "source_path": str(tmp_path / "source.png"),
        "target_pixel_sha256": N._pixel_sha256(target),
        "max_side": 16,
        "final_budget": 64,
        "start_budget": 32,
        "growth_waves": 2,
        "iters": 10,
        "seed": 0,
        "psnr": row["psnr"],
        "ms_ssim": row["ms_ssim"],
        "lpips": row["lpips"],
        "auc_psnr": row["auc_psnr"],
        "fit_seconds": row["fit_seconds"],
        "total_seconds": row["total_seconds"],
    }
    pairs, summary = N._paired_native_vs_structsplat(
        [row], [structsplat], require_start_match=True
    )
    assert len(pairs) == 1
    assert pairs[0]["target_pixels_verified"] is True
    assert pairs[0]["start_gaussians_verified"] is True
    assert summary is not None and summary["pairs"] == 1
