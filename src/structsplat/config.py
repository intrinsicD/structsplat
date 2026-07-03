"""Typed configuration objects. Pure-Python (no torch), safe to import anywhere."""
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class StructureTensorConfig:
    grad_sigma: float = 1.0          # pre-smoothing of the image before gradients
    tensor_sigma: float = 2.0        # rho: smoothing of the outer-product tensor field
    gradient_operator: str = "central"  # central, sobel, or scharr
    color_space: str = "luma"        # luma (Rec.709 gray) or rgb (Di Zenzo multi-channel sum)
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
    # wse, density_random, jittered_grid, dart_throwing, halton, farthest_point, or cvt
    sampling_mode: str = "wse"
    # anisotropy: axis ratio cap for edge Gaussians (major/minor); 1.0 => isotropic
    max_axis_ratio: float = 6.0
    coherence_power: float = 1.0     # maps coherence -> anisotropy; >1 is more conservative
    orientation_mode: str = "tensor"   # tensor (strategy default), random, or zero
    scale_mode: str = "spacing"        # spacing, uniform, or knn
    init_scale_mult: float = 1.0       # multiply local spacing to get initial std
    scale_cap_mode: str = "none"       # none, hard, or feature
    scale_cap_max: float | None = None # absolute sigma cap, in pixels
    scale_feature_sigma: float = 3.0   # feature mode: visible half-length ~= sigma*value
    scale_feature_min: float = 0.75    # minimum adaptive sigma cap
    scale_feature_energy_frac: float = 0.25  # stop edge run when energy drops below this local frac
    flank_offset_frac: float = 0.5     # edge center offset in units of local minor spacing
    color_mode: str = "bilinear"       # bilinear, local_mean, two_sided, or aggregate (quadtree)
    color_radius: float = 1.5          # local mean radius or extra side-sample offset
    opacity_mode: str = "none"         # none or constant
    init_opacity: float = 0.9
    seed: int = 0


@dataclass
class FitConfig:
    iters: int = 2000
    # LRs are ~pixels/step for means under Adam; the old 2e-3 left positions nearly frozen.
    # Retuned on the init sweep (ADR-0008): ~6x faster to a target PSNR, +2 dB at fixed iters.
    lr_means: float = 5e-2
    lr_scales: float = 3e-2
    lr_rot: float = 1e-2
    lr_color: float = 3e-2
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
    aa_dilation: float = 0.0           # EWA-style low-pass: render with Sigma + d*I (px^2)
    render_chunk: int = 512            # reference renderer: lower chunk cuts peak memory
    renderer: str = "normalized"       # normalized, additive, cuda, cuda_additive, or gsplat
    lr_schedule: str = "none"          # none, step, or cosine
    lr_decay_every: int | None = None
    lr_decay_gamma: float = 0.5
    prune_every: int | None = None
    # unnormalized weight sum; <=0 disables pruning. When opacities are present the criterion is
    # opacity-weighted (activity * sigmoid(opacity)), so the threshold is in the same weight-sum
    # units scaled by opacity — a fully transparent Gaussian scores 0 and is pruned (FIT-002).
    prune_min_activity: float = 0.0
    prune_keep_min: int = 16
    split_every: int | None = None
    split_count: int = 0
    split_mode: str = "duplicate"      # duplicate, support_duplicate, residual_add, residual_tensor_add
    split_scale: float = 0.7
    # residual_tensor_add anisotropy, mirroring InitConfig semantics so the densifier and the
    # init agree on what anisotropy means: ratio = 1 + (max_axis_ratio-1)*coherence**power.
    densify_max_axis_ratio: float = 6.0
    densify_coherence_power: float = 1.0
    max_gaussians: int | None = None
    early_stop_patience: int | None = None  # logged evals without improvement; None disables
    early_stop_min_delta: float = 0.0       # PSNR improvement required to reset patience
    early_stop_min_iters: int = 0           # do not early-stop before this iteration

    def __post_init__(self):
        # aa_dilation adds Sigma + d*I; a negative value yields negative inverse variances and
        # NaN renders (CORE-004). Reject it at construction rather than mid-fit.
        if self.aa_dilation < 0.0:
            raise ValueError(f"aa_dilation must be >= 0, got {self.aa_dilation}")


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
