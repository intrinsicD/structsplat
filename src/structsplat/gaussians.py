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
    def __init__(self, means, log_scales, rotations, colors, opacities=None, scale_max=None):
        self.means = means            # (N,2) float
        self.log_scales = log_scales  # (N,2)
        self.rotations = rotations    # (N,)
        self.colors = colors          # (N,3)
        self.opacities = opacities    # optional logits, (N,)
        self.scale_max = scale_max    # optional per-Gaussian optimization cap, (N,2)

    @classmethod
    def from_numpy(cls, means, scales, angles, colors, opacities=None, scale_max=None,
                   device="cpu", dtype=torch.float32):
        def t(a):
            # Always copy: torch.as_tensor is zero-copy for float32 CPU ndarrays, so without
            # the clone an optimizer step would mutate the caller's init arrays in place
            # (verified via np.shares_memory). The field owns its parameters (CORE-004).
            return torch.as_tensor(np.asarray(a), device=device, dtype=dtype).clone()

        scales = np.clip(np.asarray(scales), 1e-3, None)
        opacity_t = None if opacities is None else t(opacities).reshape(-1)
        scale_max_t = None if scale_max is None else t(scale_max)
        return cls(t(means), torch.log(t(scales)), t(angles).reshape(-1), t(colors),
                   opacity_t, scale_max_t)

    @property
    def n(self) -> int:
        return self.means.shape[0]

    def detached(self) -> "GaussianField":
        """Return a leaf-tensor copy suitable for rebuilding an optimizer."""
        return GaussianField(
            self.means.detach().clone(),
            self.log_scales.detach().clone(),
            self.rotations.detach().clone(),
            self.colors.detach().clone(),
            None if self.opacities is None else self.opacities.detach().clone(),
            None if self.scale_max is None else self.scale_max.detach().clone(),
        )

    def subset(self, idx) -> "GaussianField":
        return GaussianField(
            self.means.detach()[idx].clone(),
            self.log_scales.detach()[idx].clone(),
            self.rotations.detach()[idx].clone(),
            self.colors.detach()[idx].clone(),
            None if self.opacities is None else self.opacities.detach()[idx].clone(),
            None if self.scale_max is None else self.scale_max.detach()[idx].clone(),
        )

    def append(self, other: "GaussianField") -> "GaussianField":
        def cat(a, b):
            return torch.cat([a.detach(), b.detach()], dim=0).clone()

        def cat_optional(a, b):
            if a is None and b is None:
                return None
            # opacities=None renders as opacity 1.0; the missing side must keep that
            # appearance after the append, so pad with a saturating logit (sigmoid(10)
            # ~ 0.99995), not zeros (which would silently mean opacity 0.5)
            if a is None:
                a = torch.full((self.n,), 10.0, device=b.device, dtype=b.dtype)
            if b is None:
                b = torch.full((other.n,), 10.0, device=a.device, dtype=a.dtype)
            return cat(a, b)

        def cat_scale_max(a, b):
            if a is None and b is None:
                return None
            if a is None:
                a = torch.full((self.n, 2), float("inf"), device=b.device, dtype=b.dtype)
            if b is None:
                b = torch.full((other.n, 2), float("inf"), device=a.device, dtype=a.dtype)
            return cat(a, b)

        return GaussianField(
            cat(self.means, other.means),
            cat(self.log_scales, other.log_scales),
            cat(self.rotations, other.rotations),
            cat(self.colors, other.colors),
            cat_optional(self.opacities, other.opacities),
            cat_scale_max(self.scale_max, other.scale_max),
        )

    def trainable(self) -> "GaussianField":
        params = [self.means, self.log_scales, self.rotations, self.colors]
        if self.opacities is not None:
            params.append(self.opacities)
        for p in params:
            p.requires_grad_(True)
        return self

    def parameter_groups(self, lr_means, lr_scales, lr_rot, lr_color, lr_opacity=1e-2):
        groups = [
            {"params": [self.means], "lr": lr_means},
            {"params": [self.log_scales], "lr": lr_scales},
            {"params": [self.rotations], "lr": lr_rot},
            {"params": [self.colors], "lr": lr_color},
        ]
        if self.opacities is not None:
            groups.append({"params": [self.opacities], "lr": lr_opacity})
        return groups

    def scales(self):
        return torch.exp(self.log_scales)

    def opacity_values(self):
        if self.opacities is None:
            return None
        return torch.sigmoid(self.opacities)

    def conics(self, dilation: float = 0.0):
        """Return (a,b,c): the unique entries of the inverse covariance Sigma^-1=[[a,b],[b,c]].

        `dilation` adds an isotropic EWA-style low-pass to the covariance (Sigma + d*I);
        exact under RS since Sigma + d*I = R diag(sx^2+d, sy^2+d) R^T.
        """
        if dilation < 0.0:
            # A negative dilation can drive sx^2+d <= 0, giving negative inverse variances,
            # exp overflow, and inf/inf = NaN through the normalized division. Fail loudly
            # here rather than silently emitting NaN images mid-fit (CORE-004).
            raise ValueError(f"dilation must be >= 0, got {dilation}")
        s = self.scales()
        inv_sx2 = 1.0 / (s[:, 0] ** 2 + dilation)
        inv_sy2 = 1.0 / (s[:, 1] ** 2 + dilation)
        c = torch.cos(self.rotations)
        sn = torch.sin(self.rotations)
        a = c * c * inv_sx2 + sn * sn * inv_sy2
        b = c * sn * (inv_sx2 - inv_sy2)
        cc = sn * sn * inv_sx2 + c * c * inv_sy2
        return torch.stack([a, b, cc], dim=1)

    def radii(self, sigma_cutoff: float, dilation: float = 0.0):
        # Per-axis support half-widths (N,2): the AABB of the sigma_cutoff ellipse.
        # Sigma_xx = c^2 sx^2 + s^2 sy^2, Sigma_yy = s^2 sx^2 + c^2 sy^2; an elongated
        # Gaussian gets a tight rectangle instead of a square sized by its major axis.
        # Detached on purpose: radii only set the tile/bbox extent, never the loss gradient.
        if dilation < 0.0:
            raise ValueError(f"dilation must be >= 0, got {dilation}")
        with torch.no_grad():
            s2 = self.scales() ** 2 + dilation
            c = torch.cos(self.rotations)
            sn = torch.sin(self.rotations)
            var_x = c * c * s2[:, 0] + sn * sn * s2[:, 1]
            var_y = sn * sn * s2[:, 0] + c * c * s2[:, 1]
            r = sigma_cutoff * torch.sqrt(torch.stack([var_x, var_y], dim=1))
            return torch.clamp(r.ceil().long(), min=1)

    @torch.no_grad()
    def save(self, path: str):
        data = {
            "means": self.means.detach().cpu().numpy(),
            "log_scales": self.log_scales.detach().cpu().numpy(),
            "rotations": self.rotations.detach().cpu().numpy(),
            "colors": self.colors.detach().cpu().numpy(),
        }
        if self.opacities is not None:
            data["opacities"] = self.opacities.detach().cpu().numpy()
        if self.scale_max is not None:
            data["scale_max"] = self.scale_max.detach().cpu().numpy()
        np.savez(path, **data)

    @classmethod
    def load(cls, path: str, device="cpu"):
        z = np.load(path)

        def t(k):
            return torch.as_tensor(z[k], device=device, dtype=torch.float32)

        opacities = t("opacities") if "opacities" in z.files else None
        scale_max = t("scale_max") if "scale_max" in z.files else None
        return cls(t("means"), t("log_scales"), t("rotations"), t("colors"), opacities, scale_max)
