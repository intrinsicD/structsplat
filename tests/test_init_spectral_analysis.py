# ruff: noqa: E402
import json

import numpy as np
import pytest
from PIL import Image

torch = pytest.importorskip("torch")

from benchmarks.init_spectral_analysis import (
    edge_anisotropy_signature,
    nearest_spacing_metrics,
    radial_power_profile,
    run_analysis,
    spectral_trough_metrics,
)
from structsplat import structure_tensor as st


def test_radial_power_profile_returns_finite_trough_metrics():
    rng = np.random.default_rng(0)
    points = rng.random((32, 2)) * 8.0

    profile = radial_power_profile(points, 8, 8, bins=8)
    metrics = spectral_trough_metrics(profile)

    assert len(profile["power"]) == 8
    assert metrics["spectral_low_mid_ratio"] is not None
    assert np.isfinite(metrics["spectral_low_mid_ratio"])


def test_nearest_spacing_p05_tracks_separation():
    rng = np.random.default_rng(0)
    random_points = rng.random((64, 2)) * 32.0
    yy, xx = np.mgrid[0:8, 0:8]
    grid_points = np.stack([xx.ravel() * 4.0, yy.ravel() * 4.0], axis=1)

    random_stats = nearest_spacing_metrics(random_points)
    grid_stats = nearest_spacing_metrics(grid_points)

    assert grid_stats["nn_p05"] > random_stats["nn_p05"]


def test_edge_anisotropy_signature_projects_nn_displacements():
    h, w = 32, 32
    zeros = np.zeros((h, w), dtype=np.float32)
    coherence = np.ones((h, w), dtype=np.float32)
    label = np.ones((h, w), dtype=np.uint8)
    tensor = st.StructureTensor(
        lam1=zeros,
        lam2=zeros,
        across_edge_angle=zeros,
        coherence=coherence,
        energy=np.ones((h, w), dtype=np.float32),
        label=label,
        energy_ref=1.0,
    )
    # across angle 0 => normal is x, tangent is y. Nearest neighbors are vertical.
    points = np.array([[16.0, y] for y in range(2, 30, 4)], dtype=np.float64)

    metrics = edge_anisotropy_signature(points, tensor)

    assert metrics["edge_samples"] == len(points)
    assert metrics["edge_nn_tangent"] > 0
    assert metrics["edge_nn_normal"] == pytest.approx(0.0)
    assert metrics["edge_nn_tangent_normal_ratio"] is None


def test_run_analysis_writes_metrics_and_config(tmp_path):
    img = np.zeros((32, 40, 3), dtype=np.float32)
    img[:, 20:] = [0.9, 0.8, 0.2]
    path = tmp_path / "edge.png"
    Image.fromarray((img * 255).astype(np.uint8)).save(path)

    rows = run_analysis(
        [path],
        outdir=tmp_path / "out",
        strategies=["iso_blue_noise", "aniso_onedge"],
        num_gaussians=48,
        max_axis_ratios=[2.0, 4.0],
        coherence_powers=[1.0],
        candidate_oversample=2.0,
        max_side=40,
        bins=12,
        device="cpu",
    )

    assert len(rows) == 3  # isotropic uses one sweep cell; anisotropic uses both ratio caps
    assert all(row["n_out"] == 48 for row in rows)
    assert (tmp_path / "out" / "init_spectral_metrics.csv").exists()
    assert (tmp_path / "out" / "summary.md").exists()
    config = json.loads((tmp_path / "out" / "init_spectral_config.json").read_text())
    assert config["resolved"]["num_gaussians"] == 48
