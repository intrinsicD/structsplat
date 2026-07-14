import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from structsplat import structure_tensor as st
from structsplat import render as render_ops
from structsplat.config import StructureTensorConfig
from structsplat.gaussians import GaussianField
from structsplat.visualize import (
    FigureConfig,
    LABEL_COLORS,
    generate_figure_bundle,
    labels_to_rgb,
    main,
    responsibility_diagnostics,
    tensor_glyphs,
)


def _vertical_edge(height: int = 48, width: int = 48) -> np.ndarray:
    image = np.zeros((height, width, 3), dtype=np.float32)
    image[:, width // 2 :] = 1.0
    return image


def _write_test_image(path: Path) -> None:
    height, width = 32, 40
    yy, xx = np.mgrid[0:height, 0:width]
    image = np.stack([xx / (width - 1), yy / (height - 1), ((xx + 2 * yy) % 13) / 12.0], axis=2)
    Image.fromarray(np.round(image * 255).astype(np.uint8), mode="RGB").save(path)


def test_label_palette_is_exact_and_complete():
    labels = np.asarray([[0, 1, 2]], dtype=np.uint8)
    rgb = labels_to_rgb(labels)
    for index, label in enumerate((0, 1, 2)):
        assert tuple(rgb[0, index]) == LABEL_COLORS[label]


def test_vertical_edge_glyphs_follow_the_tangent():
    tensor = st.compute(_vertical_edge(), StructureTensorConfig())
    glyphs = tensor_glyphs(tensor, spacing=8, min_coherence=0.1)
    near_edge = glyphs[np.abs(glyphs[:, 0] - 24) <= 4]
    assert len(near_edge) > 0
    # A vertical edge has a horizontal gradient and a vertical tangent (theta modulo pi).
    assert np.mean(np.abs(np.cos(near_edge[:, 2])) < 0.2) > 0.8


def test_responsibility_diagnostic_identities_and_bounds():
    field = GaussianField.from_numpy(
        means=np.asarray([[7.0, 8.0], [9.0, 8.0], [8.0, 10.0]], dtype=np.float32),
        scales=np.asarray([[4.0, 2.0], [4.0, 2.0], [2.0, 3.0]], dtype=np.float32),
        angles=np.asarray([0.0, 0.0, np.pi / 2], dtype=np.float32),
        colors=np.asarray([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32),
    )
    diagnostic = responsibility_diagnostics(field, 16, 16, sigma_cutoff=3.0, chunk=64)
    covered = diagnostic.active_count > 0
    # An additive render with unit RGB is exactly the normalized renderer's denominator.
    denominator_oracle = (
        render_ops.render_additive(
            field.means,
            field.conics(),
            torch.ones_like(field.colors),
            field.radii(3.0),
            16,
            16,
            chunk=64,
            sigma_cutoff=3.0,
        )[..., 0]
        .detach()
        .cpu()
        .numpy()
    )

    assert np.allclose(diagnostic.denominator, denominator_oracle, atol=1e-6)
    assert np.allclose(diagnostic.responsibility_sum[covered], 1.0)
    assert np.all(diagnostic.effective_count[covered] >= 1.0 - 1e-5)
    assert np.all(diagnostic.effective_count[covered] <= diagnostic.active_count[covered] + 1e-5)
    assert np.all(diagnostic.entropy[covered] >= -1e-6)
    assert np.all(
        diagnostic.entropy[covered]
        <= np.log(np.maximum(diagnostic.active_count[covered], 1)) + 1e-5
    )
    assert np.all((diagnostic.dominant_owner[covered] >= 0))
    assert np.all((diagnostic.dominant_owner[covered] < field.n))


def test_figure_bundle_is_deterministic_and_records_claim_limits(tmp_path: Path):
    image_path = tmp_path / "input.png"
    _write_test_image(image_path)
    config = FigureConfig(
        max_side=32,
        num_gaussians=20,
        candidate_oversample=2.0,
        glyph_spacing=8,
        glyph_length=4.0,
        ellipse_limit=20,
        metric_ellipse_limit=10,
        montage_columns=3,
    )
    first = generate_figure_bundle(image_path, tmp_path / "first", config)
    second = generate_figure_bundle(image_path, tmp_path / "second", config)

    assert first["outputs"] == second["outputs"]
    assert "initialization" in first["claim_limit"].lower()
    assert first["source"]["source_sha256"] == second["source"]["source_sha256"]
    assert first["implementation_sources"] == second["implementation_sources"]
    assert any(
        path.endswith("src/structsplat/visualize.py") for path in first["implementation_sources"]
    )
    assert first["diagnostic_checks"]["max_responsibility_sum_error_on_covered"] == 0.0
    assert first["diagnostic_checks"]["max_effective_count_bound_violation"] <= 1e-5
    assert (tmp_path / "first" / "diagnostics.npz").is_file()
    assert (tmp_path / "first" / "method_diagnostics_montage.png").is_file()
    overview_path = tmp_path / "first" / "method_overview.svg"
    overview = overview_path.read_text(encoding="utf-8")
    assert ET.parse(overview_path).getroot().tag.endswith("svg")
    assert "source-derived maps" in overview
    assert "SSPL1 bytes" in overview
    assert "Normalized renderer" in overview

    with np.load(tmp_path / "first" / "diagnostics.npz") as arrays:
        assert arrays["tensor_energy"].shape == (26, 32)
        assert arrays["field_means_xy"].shape == (20, 2)
        assert np.isclose(arrays["sampling_density_pmf"].sum(), 1.0)


def test_figure_cli_smoke(tmp_path: Path):
    image_path = tmp_path / "input.png"
    output = tmp_path / "cli"
    _write_test_image(image_path)
    main(
        [
            str(image_path),
            "--outdir",
            str(output),
            "--max-side",
            "24",
            "--num-gaussians",
            "12",
            "--candidate-oversample",
            "2",
            "--glyph-spacing",
            "6",
            "--ellipse-limit",
            "12",
            "--metric-ellipse-limit",
            "6",
        ]
    )
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["diagnostic_checks"]["field_count"] == 12
    assert (output / "tensor_tangents.png").is_file()
