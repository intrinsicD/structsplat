"""Typed configuration objects. Pure-Python (no torch), safe to import anywhere."""
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class StructureTensorConfig:
    grad_sigma: float = 1.0          # pre-smoothing of the image before gradients
    tensor_sigma: float = 2.0        # rho: smoothing of the outer-product tensor field
    # classification thresholds, expressed as fractions of the 99th-percentile energy
    flat_frac: float = 0.02          # energy below this -> flat
    corner_frac: float = 0.15        # smaller eigenvalue above this*ref -> corner


@dataclass
class InitConfig:
    strategy: str = "aniso_flanking"  # see structsplat.init.STRATEGIES
    num_gaussians: int = 20000
    candidate_oversample: float = 6.0  # WSE draws oversample*N candidates
    density_base: float = 0.05         # floor so flat regions still get some coverage
    density_power: float = 1.0         # density ~ energy**power
    # anisotropy: axis ratio cap for edge Gaussians (major/minor); 1.0 => isotropic
    max_axis_ratio: float = 6.0
    init_scale_mult: float = 1.0       # multiply local spacing to get initial std
    flank_offset_frac: float = 0.5     # edge center offset in units of local minor spacing
    seed: int = 0


@dataclass
class FitConfig:
    iters: int = 2000
    lr_means: float = 2e-3
    lr_scales: float = 5e-3
    lr_rot: float = 2e-3
    lr_color: float = 1e-2
    ssim_weight: float = 0.3           # loss = (1-w)*L1 + w*(1-SSIM); Instant-GI/AIR default
    target_psnr: float | None = None   # record iters-to-target if set
    log_every: int = 100
    sigma_cutoff: float = 3.0          # render support radius in std devs
    render_chunk: int = 4096


@dataclass
class PyramidConfig:
    levels: int = 4
    # fraction of the total budget placed at each level (coarse -> fine). Must sum ~1.
    level_fractions: list[float] = field(default_factory=lambda: [0.1, 0.2, 0.3, 0.4])
    iters_per_level: int = 500
    residual_grad_sigma: float = 0.8   # structure tensor of the residual is sharper
