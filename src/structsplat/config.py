"""Typed configuration objects. Pure-Python (no torch), safe to import anywhere."""
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class StructureTensorConfig:
    grad_sigma: float = 1.0          # pre-smoothing of the image before gradients
    tensor_sigma: float = 2.0        # rho: smoothing of the outer-product tensor field
    gradient_operator: str = "central"  # central, sobel, or scharr
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
    density_mode: str = "structure"    # structure, gradient, variance, hybrid, or uniform
    sampling_mode: str = "wse"         # wse, density_random, or jittered_grid
    # anisotropy: axis ratio cap for edge Gaussians (major/minor); 1.0 => isotropic
    max_axis_ratio: float = 6.0
    coherence_power: float = 1.0     # maps coherence -> anisotropy; >1 is more conservative
    scale_mode: str = "spacing"        # spacing or uniform
    init_scale_mult: float = 1.0       # multiply local spacing to get initial std
    flank_offset_frac: float = 0.5     # edge center offset in units of local minor spacing
    color_mode: str = "bilinear"       # bilinear, local_mean, or two_sided
    color_radius: float = 1.5          # local mean radius or extra side-sample offset
    opacity_mode: str = "none"         # none or constant
    init_opacity: float = 0.9
    seed: int = 0


@dataclass
class FitConfig:
    iters: int = 2000
    lr_means: float = 2e-3
    lr_scales: float = 5e-3
    lr_rot: float = 2e-3
    lr_color: float = 1e-2
    lr_opacity: float = 1e-2
    optimizer: str = "adam"            # adam or adamw
    pixel_loss: str = "l1"             # l1, l2, or charbonnier; SSIM term is mixed separately
    charbonnier_eps: float = 1e-3
    loss_warmup_iters: int = 0
    loss_warmup_pixel_loss: str = "l2"
    ssim_weight: float = 0.3           # loss = (1-w)*L1 + w*(1-SSIM); Instant-GI/AIR default
    compute_lpips: bool = False        # opt-in: loads a separate AlexNet LPIPS model
    target_psnr: float | None = None   # record iters-to-target if set
    target_psnrs: list[float] = field(default_factory=list)
    log_every: int = 100
    sigma_cutoff: float = 3.0          # render support radius in std devs
    render_chunk: int = 512            # reference renderer: lower chunk cuts peak memory
    renderer: str = "normalized"       # normalized or additive
    lr_schedule: str = "none"          # none, step, or cosine
    lr_decay_every: int | None = None
    lr_decay_gamma: float = 0.5
    prune_every: int | None = None
    prune_min_activity: float = 0.0     # unnormalized weight sum; <=0 disables pruning
    prune_keep_min: int = 16
    split_every: int | None = None
    split_count: int = 0
    split_mode: str = "duplicate"      # duplicate or residual_add
    split_scale: float = 0.7
    max_gaussians: int | None = None


@dataclass
class PyramidConfig:
    levels: int = 4
    # fraction of the total budget placed at each level (coarse -> fine). Must sum ~1.
    level_fractions: list[float] = field(default_factory=lambda: [0.1, 0.2, 0.3, 0.4])
    iters_per_level: int = 500
    residual_grad_sigma: float = 0.8   # structure tensor of the residual is sharper
    level_grad_sigmas: list[float] | None = None
    level_tensor_sigmas: list[float] | None = None
    evaluate_prefixes: bool = True
