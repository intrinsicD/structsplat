from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from benchmarks import native_gaussianimage_compare as G
from benchmarks.native_runners import gaussianimage as runner


def test_runner_profiles_preserve_official_model_axes():
    assert runner._profile_model("matched_steps_fixed_n") == (
        "GaussianImage_Cholesky",
        "gaussianimage_cholesky.py",
    )
    assert runner._profile_model("release_rs") == (
        "GaussianImage_RS",
        "gaussianimage_rs.py",
    )
    runner._validate_profile("paper_cholesky_50k_n70k", 70_000, 50_000)
    with pytest.raises(ValueError, match="requires"):
        runner._validate_profile("paper_cholesky_50k_n70k", 640, 500)


def test_native_command_uses_isolated_python_and_fixed_count(tmp_path: Path):
    command = G._native_command(
        native_python=tmp_path / "env" / "bin" / "python",
        repo=tmp_path / "GaussianImage",
        image=tmp_path / "target.png",
        cell_dir=tmp_path / "cell",
        profile="matched_steps_fixed_n",
        budget=640,
        iters=500,
        seed=1,
        render_warmup=3,
        render_repeats=7,
        device="cuda:0",
    )
    assert command[0] == str(tmp_path / "env" / "bin" / "python")
    assert command[command.index("--num-gaussians") + 1] == "640"
    assert command[command.index("--iterations") + 1] == "500"
    assert command[command.index("--profile") + 1] == "matched_steps_fixed_n"


def test_cell_key_invalidates_shared_comparison_revision():
    row = {
        "schema_version": G.SCHEMA_VERSION,
        "adapter_revision": G.ADAPTER_REVISION,
        "metric_protocol_revision": G.METRIC_PROTOCOL_REVISION,
        "protocol": G.PROTOCOL,
        "profile": "matched_steps_fixed_n",
        "source_path": "target.png",
        "target_pixel_sha256": "pixels",
        "max_side": 160,
        "budget_cap": 640,
        "requested_iters": 500,
        "seed": 0,
        "shared_compare_source_sha256": "shared-a",
    }
    assert G._cell_key(row) != G._cell_key(
        {**row, "shared_compare_source_sha256": "shared-b"}
    )


def test_write_outputs_compacts_stale_resume_journal(tmp_path: Path):
    (tmp_path / "metrics.jsonl").write_text(
        json.dumps({"status": "ok", "method": "stale"}) + "\n",
        encoding="utf-8",
    )

    G._write_outputs([], tmp_path, [], baseline_methods=[])

    assert (tmp_path / "metrics.jsonl").read_text(encoding="utf-8") == ""


def test_manifest_validation_requires_exact_extension_and_terminal_state(tmp_path: Path):
    repo = tmp_path / "repo"
    extension = repo / "gsplat" / "gsplat" / "csrc.so"
    extension.parent.mkdir(parents=True)
    extension.write_bytes(b"official-extension")
    target_path = tmp_path / "target.png"
    target_path.write_bytes(b"target")
    reconstruction_path = tmp_path / "reconstruction.npy"
    np.save(reconstruction_path, np.zeros((8, 9, 3), dtype=np.float32))
    checkpoint_path = tmp_path / "terminal.pth"
    checkpoint_path.write_bytes(b"checkpoint")
    fingerprint = {
        "repo_root": str(repo),
        "repo_commit": "repo-commit",
        "gsplat_commit": "gsplat-commit",
        "gsplat_csrc": str(extension),
        "gsplat_csrc_sha256": G._sha256(extension),
        "repo_tracked_diff_sha256": "clean-repo",
        "gsplat_tracked_diff_sha256": "clean-gsplat",
    }
    manifest = {
        "schema_version": G.SCHEMA_VERSION,
        "adapter_revision": G.ADAPTER_REVISION,
        "method": G.METHOD,
        "profile": "matched_steps_fixed_n",
        "repo_root": str(repo),
        "repo_commit": "repo-commit",
        "gsplat_commit": "gsplat-commit",
        "gsplat_csrc": str(extension),
        "gsplat_csrc_sha256": G._sha256(extension),
        "source_path": str(target_path),
        "height": 8,
        "width": 9,
        "seed": 0,
        "requested_iters": 3,
        "iterations_run": 3,
        "budget_cap": 10,
        "start_gaussians": 10,
        "n_gaussians": 10,
        "selected_n_gaussians": 10,
        "selected_iter": 3,
        "selection_policy": "terminal",
        "repo_state": {"tracked_diff_sha256": "clean-repo"},
        "gsplat_state": {"tracked_diff_sha256": "clean-gsplat"},
        "init_seconds": 0.1,
        "fit_seconds": 0.2,
        "total_seconds": 0.3,
        "render_ms_mean": 1.0,
        "render_ms_median": 1.0,
        "render_ms_p10": 0.9,
        "render_ms_p90": 1.1,
        "render_fps_median": 1000.0,
        "native_reported_psnr": 20.0,
        "history": {
            "iter": [0, 1, 2],
            "psnr": [10.0, 15.0, 20.0],
            "elapsed": [0.1, 0.2, 0.3],
            "n_gaussians": [10, 10, 10],
        },
        "reconstruction_npy": str(reconstruction_path),
        "checkpoint_path": str(checkpoint_path),
    }

    reconstruction = G._validate_manifest(
        manifest,
        fingerprint=fingerprint,
        target_path=target_path,
        target_shape=(8, 9),
        profile="matched_steps_fixed_n",
        budget=10,
        iters=3,
        seed=0,
    )
    assert reconstruction.shape == (8, 9, 3)

    manifest["selection_policy"] = "best_psnr"
    with pytest.raises(ValueError, match="terminal state"):
        G._validate_manifest(
            manifest,
            fingerprint=fingerprint,
            target_path=target_path,
            target_shape=(8, 9),
            profile="matched_steps_fixed_n",
            budget=10,
            iters=3,
            seed=0,
        )
