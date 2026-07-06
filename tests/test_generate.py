import numpy as np
import torch

from structsplat.config import GenConfig
from structsplat.gaussians import GaussianField
from structsplat.generate import generate, refine_field


class FakeGuidance:
    def sample_raster(self, prompt, cfg):
        y, x = np.mgrid[0:cfg.height, 0:cfg.width].astype(np.float32)
        cx, cy = 0.62 * cfg.width, 0.38 * cfg.height
        blob = np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (0.08 * cfg.width * cfg.height))
        img = np.zeros((cfg.height, cfg.width, 3), dtype=np.float32)
        img[..., 0] = 0.15 + 0.8 * blob
        img[..., 1] = 0.15 + 0.25 * (x / max(cfg.width - 1, 1))
        img[..., 2] = 0.2 + 0.35 * (y / max(cfg.height - 1, 1))
        return np.clip(img, 0.0, 1.0)

    def sds_loss(self, image_bchw, prompt):
        _, _, h, w = image_bchw.shape
        yy, xx = torch.meshgrid(
            torch.linspace(0.0, 1.0, h, device=image_bchw.device, dtype=image_bchw.dtype),
            torch.linspace(0.0, 1.0, w, device=image_bchw.device, dtype=image_bchw.dtype),
            indexing="ij",
        )
        target = torch.stack(
            [
                torch.exp(-((xx - 0.72) ** 2 + (yy - 0.28) ** 2) / 0.035),
                0.2 + 0.5 * xx,
                0.2 + 0.4 * yy,
            ],
            dim=0,
        ).unsqueeze(0)
        diff = image_bchw - target
        loss = (diff ** 2).mean()
        return loss, {
            "sds_grad_norm": float(torch.linalg.vector_norm(diff.detach()).cpu()),
            "timestep_mean": 123.0,
        }


def test_sds_refine_backprops_to_gaussian_parameters():
    means = np.array([[4.0, 5.0], [10.0, 9.0], [7.0, 12.0]], dtype=np.float32)
    scales = np.array([[3.0, 1.2], [1.5, 3.5], [2.5, 1.0]], dtype=np.float32)
    angles = np.array([0.25, 0.9, 1.4], dtype=np.float32)
    colors = np.array([[0.8, 0.1, 0.1], [0.1, 0.7, 0.2], [0.2, 0.2, 0.8]], dtype=np.float32)
    opacities = np.array([0.2, 0.0, -0.3], dtype=np.float32)
    field = GaussianField.from_numpy(means, scales, angles, colors, opacities=opacities)
    cfg = GenConfig(
        height=16,
        width=16,
        num_gaussians=3,
        refine_steps=1,
        renderer="additive",
        render_chunk=1,
        grad_clip_norm=None,
        save_resolutions=[16],
    )

    out = refine_field(field, FakeGuidance(), cfg, "red icon", verbose=False)
    hist = out["history"]

    for key in ("loss", "grad_means", "grad_scales", "grad_color", "grad_opacity"):
        assert np.isfinite(hist[key][-1])
    assert hist["grad_means"][-1] > 0
    assert hist["grad_scales"][-1] > 0
    assert hist["grad_color"][-1] > 0
    assert hist["grad_opacity"][-1] > 0
    assert torch.isfinite(out["render"]).all()


def test_generate_with_fake_guidance_saves_artifacts(tmp_path):
    import pytest

    pytest.importorskip("PIL")
    cfg = GenConfig(
        height=16,
        width=16,
        num_gaussians=8,
        fit_iters=1,
        refine_steps=1,
        init_strategy="grid",
        renderer="additive",
        render_chunk=1,
        grad_clip_norm=None,
        save_resolutions=[16, 32],
    )

    out = generate(
        "flat red app icon",
        cfg,
        guidance=FakeGuidance(),
        outdir=str(tmp_path),
        verbose=False,
    )

    assert out["field"].n == 8
    for name in (
        "sample.png",
        "fit.png",
        "refined.png",
        "field.npz",
        "render_16px.png",
        "render_32px.png",
        "metadata.json",
    ):
        assert (tmp_path / name).exists()
