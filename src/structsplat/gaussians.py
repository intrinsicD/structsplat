"""2D Gaussian field with the rotation-scaling (RS) parameterization (ADR-0002).

Each Gaussian: mean mu in pixel coords, log-scales (log sx, log sy), rotation theta, color (3).
8 trainable scalars per Gaussian (matching GaussianImage's parameter count; opacity folded into
unbounded color). RS is preferred over Cholesky because it decouples orientation from extent,
which is exactly what the structure-tensor init wants to set directly.

Requires torch.
"""
from __future__ import annotations
import numpy as np
import torch


class GaussianField:
    def __init__(self, means, log_scales, rotations, colors):
        self.means = means            # (N,2) float
        self.log_scales = log_scales  # (N,2)
        self.rotations = rotations    # (N,)
        self.colors = colors          # (N,3)

    @classmethod
    def from_numpy(cls, means, scales, angles, colors, device="cpu", dtype=torch.float32):
        t = lambda a: torch.as_tensor(np.asarray(a), device=device, dtype=dtype)
        scales = np.clip(np.asarray(scales), 1e-3, None)
        return cls(t(means), torch.log(t(scales)), t(angles).reshape(-1), t(colors))

    @property
    def n(self) -> int:
        return self.means.shape[0]

    def trainable(self) -> "GaussianField":
        for p in (self.means, self.log_scales, self.rotations, self.colors):
            p.requires_grad_(True)
        return self

    def parameter_groups(self, lr_means, lr_scales, lr_rot, lr_color):
        return [
            {"params": [self.means], "lr": lr_means},
            {"params": [self.log_scales], "lr": lr_scales},
            {"params": [self.rotations], "lr": lr_rot},
            {"params": [self.colors], "lr": lr_color},
        ]

    def scales(self):
        return torch.exp(self.log_scales)

    def conics(self):
        """Return (a,b,c): the unique entries of the inverse covariance Sigma^-1=[[a,b],[b,c]]."""
        s = self.scales()
        inv_sx2 = 1.0 / (s[:, 0] ** 2)
        inv_sy2 = 1.0 / (s[:, 1] ** 2)
        c = torch.cos(self.rotations)
        sn = torch.sin(self.rotations)
        a = c * c * inv_sx2 + sn * sn * inv_sy2
        b = c * sn * (inv_sx2 - inv_sy2)
        cc = sn * sn * inv_sx2 + c * c * inv_sy2
        return torch.stack([a, b, cc], dim=1)

    def radii(self, sigma_cutoff: float):
        # eigenvalues of Sigma are sx^2, sy^2 -> max std is max(sx,sy).
        # Detached on purpose: radii only set the tile/bbox extent, never the loss gradient.
        with torch.no_grad():
            smax = self.scales().max(dim=1).values
            return torch.clamp((sigma_cutoff * smax).ceil().long(), min=1)

    @torch.no_grad()
    def save(self, path: str):
        np.savez(path, means=self.means.detach().cpu().numpy(),
                 log_scales=self.log_scales.detach().cpu().numpy(),
                 rotations=self.rotations.detach().cpu().numpy(),
                 colors=self.colors.detach().cpu().numpy())

    @classmethod
    def load(cls, path: str, device="cpu"):
        z = np.load(path)
        t = lambda k: torch.as_tensor(z[k], device=device, dtype=torch.float32)
        return cls(t("means"), t("log_scales"), t("rotations"), t("colors"))
