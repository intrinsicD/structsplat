import json

import numpy as np
from PIL import Image

from scripts.check_report_bundle import check_bundle
from scripts.experiments import abl002_renderer_report as report


def test_abl002_renderer_report_emits_visual_curves_and_portable_bundle(tmp_path):
    source = tmp_path / "images"
    source.mkdir()
    height, width = 18, 24
    yy, xx = np.mgrid[:height, :width]
    pixels = np.stack(
        (
            xx / max(1, width - 1),
            yy / max(1, height - 1),
            ((xx + yy) % 7) / 6.0,
        ),
        axis=2,
    )
    Image.fromarray(np.rint(pixels * 255).astype(np.uint8), mode="RGB").save(
        source / "fixture.png"
    )
    outdir = tmp_path / "report"

    result = report.main(
        [
            str(source),
            str(outdir),
            "--budgets",
            "12",
            "--seeds",
            "0",
            "--iters",
            "2",
            "--max-side",
            "24",
            "--log-every",
            "1",
            "--snapshot-steps",
            "0",
            "1",
            "2",
            "--renderers",
            "normalized",
            "additive",
            "--device",
            "cpu",
            "--quiet",
        ]
    )

    assert result == 0
    assert check_bundle(outdir, allow_dirty=True) == []
    rows = json.loads((outdir / "metrics.json").read_text(encoding="utf-8"))
    assert len(rows) == 2
    assert {row["renderer_equation"] for row in rows} == {"normalized", "additive"}
    for row in rows:
        assert len(row["curves"]) == 3
        assert len(row["snapshots"]) == 3
        assert all(
            key in row["curves"][0]
            for key in (
                "loss",
                "psnr",
                "ssim",
                "ms_ssim",
                "mse",
                "mae",
                "cvar99_mse",
                "p99_mse",
                "interior_hole_fraction",
                "render_out_of_range_fraction",
                "elapsed_seconds",
            )
        )
        for artifact in (
            "target_png",
            "reconstruction_png",
            "error_png",
            "field_npz",
            "history_json",
            "config_json",
        ):
            assert (outdir / row[artifact]).is_file()

    index = (outdir / "index.html").read_text(encoding="utf-8")
    assert "Visual comparisons" in index
    assert "Aggregate trajectories" in index
    assert "Optimization objective over attempted steps" in index
    assert "normalized reconstruction" in index
    assert "additive reconstruction" in index
