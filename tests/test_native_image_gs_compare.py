from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from benchmarks import native_image_gs_compare as I


def _repo_state() -> dict:
    return {
        "repo_commit": "imagegs-commit",
        "repo_tree": "imagegs-tree",
        "repo_dirty": False,
        "repo_diff_sha256": "0" * 64,
        "repo_status": [],
        "repo_origin_url": "https://github.com/NYU-ICL/image-gs.git",
    }


def _environment(tmp_path: Path) -> dict:
    root = tmp_path / "env"
    python = root / "bin" / "python"
    gsplat = root / "lib" / "python" / "site-packages" / "gsplat" / "csrc.so"
    fused = root / "lib" / "python" / "site-packages" / "fused_ssim_cuda.so"
    python.parent.mkdir(parents=True)
    gsplat.parent.mkdir(parents=True)
    fused.parent.mkdir(parents=True, exist_ok=True)
    python.write_bytes(b"python")
    gsplat.write_bytes(b"gsplat")
    fused.write_bytes(b"fused")
    return {
        "python_executable": str(python),
        "python_version": "3.12-test",
        "environment_root": str(root),
        "torch_version": "2.9-test",
        "torch_cuda_version": "12.8",
        "cuda_available": True,
        "cuda_device_count": 1,
        "cuda_devices": [
            {
                "index": 0,
                "name": "Test GPU",
                "total_memory": 1024,
                "compute_capability": [9, 0],
            }
        ],
        "gsplat_module": str(gsplat.parent / "__init__.py"),
        "gsplat_csrc": str(gsplat),
        "gsplat_csrc_sha256": hashlib.sha256(gsplat.read_bytes()).hexdigest(),
        "gsplat_python_source_sha256": "gsplat-python-hash",
        "repo_gsplat_python_source_sha256": "gsplat-python-hash",
        "gsplat_direct_url_path": str(root / "gsplat-direct.json"),
        "gsplat_direct_url": {"url": "file:///repo/gsplat"},
        "fused_ssim_module": str(fused.parent / "fused_ssim" / "__init__.py"),
        "fused_ssim_csrc": str(fused),
        "fused_ssim_csrc_sha256": hashlib.sha256(fused.read_bytes()).hexdigest(),
        "fused_ssim_python_sha256": "fused-python-hash",
        "native_dependency_versions": {
            name: f"test-{name}" for name in I.DEPENDENCY_DISTRIBUTIONS
        },
        "fused_ssim_direct_url_path": str(root / "fused-direct.json"),
        "fused_ssim_direct_url": {
            "url": "https://github.com/rahul-goel/fused-ssim/",
            "vcs_info": {"commit_id": I.FUSED_SSIM_COMMIT},
        },
        "fused_ssim_commit": I.FUSED_SSIM_COMMIT,
        "libstdcxx_preload": None,
        "libstdcxx_sha256": None,
        "native_adapter_source_sha256": "adapter-source-hash",
        "native_harness_source_sha256": "harness-source-hash",
        "shared_compare_source_sha256": "shared-source-hash",
        "structsplat_metrics_source_sha256": "metrics-source-hash",
        "metric_python_version": "3.12-central",
        "metric_torch_version": "2.9-central",
        "metric_numpy_version": "2.0-central",
        "metric_lpips_version": "0.1.4",
        "metric_torchvision_version": "0.19.1",
    }


def _manifest(
    repo: Path,
    environment: dict,
    target_path: Path,
    reconstruction_path: Path,
    **overrides,
) -> dict:
    data = {
        "schema_version": I.SCHEMA_VERSION,
        "adapter_revision": I.ADAPTER_REVISION,
        "method": I.METHOD,
        "profile": "matched_steps_fixed_n",
        "native_device": "cuda:0",
        "native_gpu_name": "Test GPU",
        "adapter_source_sha256": environment["native_adapter_source_sha256"],
        "repo_root": str(repo),
        **_repo_state(),
        **environment,
        "source_path": str(target_path),
        "height": 16,
        "width": 16,
        "seed": 0,
        "requested_iters": 10,
        "effective_max_steps": 10,
        "iterations_run": 10,
        "selection_policy": "terminal_step",
        "selected_iter": 10,
        "budget_cap": 64,
        "start_gaussians": 64,
        "requested_start_gaussians": 64,
        "n_gaussians": 64,
        "growth_waves_requested": 4,
        "native_reported_psnr": 30.0,
        "native_reported_ssim": 0.9,
        "native_internal_optimization_seconds": 0.8,
        "native_internal_render_seconds": 0.2,
        "init_seconds": 0.1,
        "fit_seconds": 1.0,
        "total_seconds": 1.1,
        "render_ms_mean": 1.0,
        "render_ms_median": 1.0,
        "render_ms_p10": 0.9,
        "render_ms_p90": 1.1,
        "render_fps_median": 1000.0,
        "analytical_payload_bytes": 1024.0,
        "analytical_bpp": 32.0,
        "actual_codec_bytes": None,
        "actual_bpp": None,
        "reconstruction_npy": str(reconstruction_path),
        "reconstruction_png": str(reconstruction_path.with_suffix(".png")),
        "reconstruction_sha256": hashlib.sha256(reconstruction_path.read_bytes()).hexdigest(),
        "reconstruction_dtype": "float32",
        "reconstruction_shape": [16, 16, 3],
        "target_sha256": hashlib.sha256(target_path.read_bytes()).hexdigest(),
        "terminal_metrics_patch": True,
        "upstream_args": {
            "disable_prog_optim": True,
            "disable_lr_schedule": True,
            "quantize": False,
            "topk": 10,
            "disable_tiles": False,
            "num_gaussians": 64,
            "max_steps": 10,
        },
        "history": {
            "iter": [0, 5, 10],
            "psnr": [20.0, 25.0, 30.0],
            "ssim": [0.5, 0.7, 0.9],
            "elapsed": [0.0, 0.5, 1.0],
            "n_gaussians": [64, 64, 64],
        },
    }
    data.update(overrides)
    return data


def _validate(manifest: dict, repo: Path, environment: dict, target_path: Path):
    return I._validate_manifest(
        manifest,
        repo=repo,
        repo_state=_repo_state(),
        environment=environment,
        target_path=target_path,
        target_shape=(16, 16),
        profile="matched_steps_fixed_n",
        budget=64,
        requested_start=64,
        requested_iters=10,
        seed=0,
        growth_waves=4,
        target_sha256=hashlib.sha256(target_path.read_bytes()).hexdigest(),
        native_device="cuda:0",
    )


def test_validate_image_gs_manifest_accepts_sparse_native_trajectory(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    environment = _environment(tmp_path)
    target_path = tmp_path / "target.png"
    target_path.write_bytes(b"target")
    reconstruction = tmp_path / "reconstruction.npy"
    np.save(reconstruction, np.full((16, 16, 3), 0.5, dtype=np.float32))

    result = _validate(
        _manifest(repo, environment, target_path, reconstruction),
        repo,
        environment,
        target_path,
    )

    assert result.shape == (16, 16, 3)
    assert result.dtype == np.float32


def test_validate_image_gs_manifest_rejects_build_axis_and_codec_mislabel(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    environment = _environment(tmp_path)
    target_path = tmp_path / "target.png"
    target_path.write_bytes(b"target")
    reconstruction = tmp_path / "reconstruction.npy"
    np.save(reconstruction, np.zeros((16, 16, 3), dtype=np.float32))
    base = _manifest(repo, environment, target_path, reconstruction)

    with pytest.raises(ValueError, match="gsplat_csrc_sha256"):
        _validate({**base, "gsplat_csrc_sha256": "wrong"}, repo, environment, target_path)
    with pytest.raises(ValueError, match="seed"):
        _validate({**base, "seed": 1}, repo, environment, target_path)
    with pytest.raises(ValueError, match="actual codec"):
        _validate({**base, "actual_bpp": 1.0}, repo, environment, target_path)


def test_image_gs_resume_key_covers_profile_build_and_timing_axes():
    row = {
        "schema_version": I.SCHEMA_VERSION,
        "adapter_revision": I.ADAPTER_REVISION,
        "metric_protocol_revision": I.METRIC_PROTOCOL_REVISION,
        "protocol": "matched_steps_fixed_n",
        "profile": "matched_steps_fixed_n",
        "method": I.METHOD,
        "repo_commit": "commit",
        "repo_tree": "tree",
        "repo_diff_sha256": "diff",
        "gsplat_csrc_sha256": "gsplat",
        "fused_ssim_csrc_sha256": "fused",
        "fused_ssim_commit": I.FUSED_SSIM_COMMIT,
        "fused_ssim_python_sha256": "fused-python-hash",
        "native_dependency_versions": {
            name: f"test-{name}" for name in I.DEPENDENCY_DISTRIBUTIONS
        },
        "python_executable": "/env/python",
        "python_version": "3.12-test",
        "environment_root": "/env",
        "torch_version": "2.9-test",
        "torch_cuda_version": "12.8",
        "cuda_devices": [{"index": 0, "name": "Test GPU"}],
        "libstdcxx_preload": None,
        "libstdcxx_sha256": None,
        "native_adapter_source_sha256": "adapter-source-hash",
        "native_harness_source_sha256": "harness-source-hash",
        "shared_compare_source_sha256": "shared-source-hash",
        "structsplat_metrics_source_sha256": "metrics-source-hash",
        "metric_python_version": "3.12-central",
        "metric_torch_version": "2.9-central",
        "metric_numpy_version": "2.0-central",
        "metric_lpips_version": "0.1.4",
        "metric_torchvision_version": "0.19.1",
        "source_path": "/image.png",
        "source_sha256": "source-hash",
        "target_sha256": "target-hash",
        "target_pixel_sha256": "target-pixel-hash",
        "max_side": 160,
        "budget_cap": 640,
        "requested_start_gaussians": 640,
        "requested_iters": 500,
        "seed": 0,
        "growth_waves_requested": 4,
        "render_warmup": 10,
        "render_repeats": 50,
        "lpips_requested": True,
        "native_device": "cuda:0",
        "metric_device": "cuda",
    }

    assert I._cell_key(row) != I._cell_key({**row, "profile": "siggraph25"})
    assert I._cell_key(row) != I._cell_key({**row, "gsplat_csrc_sha256": "other"})
    assert I._cell_key(row) != I._cell_key({**row, "source_sha256": "other"})
    assert I._cell_key(row) != I._cell_key({**row, "torch_version": "other"})
    assert I._cell_key(row) != I._cell_key(
        {**row, "native_dependency_versions": {"numpy": "other"}}
    )
    assert I._cell_key(row) != I._cell_key({**row, "native_device": "cuda:1"})
    assert I._cell_key(row) != I._cell_key({**row, "render_repeats": 100})
    assert I._cell_key(row) != I._cell_key({**row, "lpips_requested": False})


def test_image_gs_jsonl_loader_ignores_only_torn_final_line(tmp_path: Path):
    path = tmp_path / "metrics.jsonl"
    path.write_text('{"status":"ok"}\n{"status":', encoding="utf-8")
    assert I._load_jsonl(path) == [{"status": "ok"}]

    path.write_text('{"status":\n{"status":"ok"}\n', encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        I._load_jsonl(path)


def test_image_gs_subprocess_failure_becomes_error_row(tmp_path: Path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    environment = _environment(tmp_path)
    target = np.full((16, 16, 3), 0.5, dtype=np.float32)
    target_path = tmp_path / "target.png"
    target_path.write_bytes(b"target")
    source_path = tmp_path / "source.png"
    source_path.write_bytes(b"source")
    monkeypatch.setattr(
        I.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=9, stdout="", stderr="native image-gs failed\n"
        ),
    )

    row = I._run_cell(
        python=Path(environment["python_executable"]),
        repo=repo,
        repo_state=_repo_state(),
        environment=environment,
        preload=None,
        source_path=source_path,
        target_path=target_path,
        target=target,
        max_side=16,
        profile="matched_steps_fixed_n",
        budget=64,
        start_gaussians=64,
        iters=10,
        seed=0,
        growth_waves=4,
        render_warmup=0,
        render_repeats=1,
        device="cpu",
        native_device="cuda:0",
        want_lpips=False,
        cell_dir=tmp_path / "cell",
    )

    assert row["status"] == "error"
    assert "exit 9" in row["error"]
    assert Path(row["stderr_path"]).read_text() == "native image-gs failed\n"


def test_image_gs_success_recomputes_central_metrics(tmp_path: Path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    environment = _environment(tmp_path)
    target = np.full((16, 16, 3), 0.5, dtype=np.float32)
    target_path = tmp_path / "target.png"
    target_path.write_bytes(b"target")
    source_path = tmp_path / "source.png"
    source_path.write_bytes(b"source")
    cell_dir = tmp_path / "cell"

    def fake_run(*args, **kwargs):
        cell_dir.mkdir(parents=True, exist_ok=True)
        reconstruction = cell_dir / "reconstruction.npy"
        np.save(reconstruction, target)
        manifest = _manifest(repo, environment, target_path, reconstruction)
        (cell_dir / "result.json").write_text(json.dumps(manifest), encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="ok\n", stderr="")

    monkeypatch.setattr(I.subprocess, "run", fake_run)
    row = I._run_cell(
        python=Path(environment["python_executable"]),
        repo=repo,
        repo_state=_repo_state(),
        environment=environment,
        preload=None,
        source_path=source_path,
        target_path=target_path,
        target=target,
        max_side=16,
        profile="matched_steps_fixed_n",
        budget=64,
        start_gaussians=64,
        iters=10,
        seed=0,
        growth_waves=4,
        render_warmup=0,
        render_repeats=1,
        device="cpu",
        native_device="cuda:0",
        want_lpips=False,
        cell_dir=cell_dir,
    )

    assert row["status"] == "ok"
    assert row["psnr"] >= 100.0
    assert row["lpips"] is None
    assert row["auc_psnr"] == pytest.approx(25.0)
    assert row["iters_to_targets"]["28.0"] == 10
