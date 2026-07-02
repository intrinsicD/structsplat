"""COMP-001: quantization + entropy coding of a fitted GaussianField (rate-distortion).

Encoding pipeline (GaussianImage-style uniform quantization, plus two representation-specific
tricks):
  * The normalized renderer is order-independent, so Gaussians are freely reordered along a
    Morton (Z-order) curve; positions are then delta-coded, which zlib compresses well.
  * Rotation is canonicalized to [0, pi): a 2D Gaussian is invariant under theta -> theta + pi.

Attribute layout per Gaussian (defaults): means 2x16 bit fixed-point over the image extent,
log-scales 2x8 bit over the fitter's clamp range [log 0.35, log max(H, W)], rotation 8 bit,
colors 3x8 bit over per-channel [min, max] stored in the header (colors are unbounded — opacity
is folded in, so the range is data-dependent). Streams are zlib-compressed separately.

`qat_finetune` runs a short straight-through-estimator fine-tune so the parameters settle onto
the quantization lattice before encoding (recovers most of the coarse-bit PSNR loss).

Requires torch (post-fit stage); the pack/unpack math is plain NumPy.
"""
from __future__ import annotations
import json
import struct
import zlib
from dataclasses import dataclass

import numpy as np
import torch

from .config import FitConfig
from .gaussians import GaussianField
from .render import render
from . import metrics as M  # noqa: N812

_MAGIC = b"SSPL1"
_SCALE_LO = float(np.log(0.35))  # matches fit.py's log-scale clamp


@dataclass
class CodecConfig:
    bits_means: int = 16      # per coordinate
    bits_scales: int = 8      # per log-scale
    bits_rot: int = 8
    bits_colors: int = 8      # per channel
    morton_reorder: bool = True
    zlib_level: int = 9
    # per-channel color ranges; computed from data when None, fixed during QAT
    color_lo: list[float] | None = None
    color_hi: list[float] | None = None


def _morton_order(x: np.ndarray, y: np.ndarray, H: int, W: int) -> np.ndarray:
    """Z-order sort indices for positions (locality => small position deltas)."""
    xi = np.clip((x / max(W - 1, 1) * 1023).round().astype(np.uint32), 0, 1023).astype(np.uint64)
    yi = np.clip((y / max(H - 1, 1) * 1023).round().astype(np.uint32), 0, 1023).astype(np.uint64)
    code = np.zeros_like(xi)
    for b in range(10):
        code |= ((xi >> np.uint64(b)) & np.uint64(1)) << np.uint64(2 * b)
        code |= ((yi >> np.uint64(b)) & np.uint64(1)) << np.uint64(2 * b + 1)
    return np.argsort(code, kind="stable")


def _quant(x: np.ndarray, lo, hi, bits: int) -> np.ndarray:
    levels = (1 << bits) - 1
    t = np.clip((x - lo) / np.maximum(np.asarray(hi) - lo, 1e-12), 0.0, 1.0)
    return np.round(t * levels).astype(np.uint32)


def _dequant(q: np.ndarray, lo, hi, bits: int) -> np.ndarray:
    levels = (1 << bits) - 1
    return (lo + (q.astype(np.float64) / levels) * (np.asarray(hi) - lo)).astype(np.float32)


def _pack(q: np.ndarray, bits: int) -> bytes:
    dtype = np.uint8 if bits <= 8 else (np.uint16 if bits <= 16 else np.uint32)
    return q.astype(dtype).tobytes()


def _unpack(raw: bytes, bits: int, count: int) -> np.ndarray:
    dtype = np.uint8 if bits <= 8 else (np.uint16 if bits <= 16 else np.uint32)
    return np.frombuffer(raw, dtype=dtype, count=count).astype(np.uint32)


def _params(field: GaussianField):
    means = field.means.detach().cpu().numpy().astype(np.float64)
    log_scales = field.log_scales.detach().cpu().numpy().astype(np.float64)
    theta = np.mod(field.rotations.detach().cpu().numpy().astype(np.float64), np.pi)
    colors = field.colors.detach().cpu().numpy().astype(np.float64)
    return means, log_scales, theta, colors


def color_ranges(field: GaussianField) -> tuple[list[float], list[float]]:
    c = field.colors.detach().cpu().numpy()
    return c.min(axis=0).tolist(), c.max(axis=0).tolist()


def encode(field: GaussianField, H: int, W: int, cfg: CodecConfig | None = None) -> bytes:
    cfg = cfg or CodecConfig()
    means, log_scales, theta, colors = _params(field)
    n = means.shape[0]
    scale_hi = float(np.log(max(H, W)))
    clo = np.asarray(cfg.color_lo if cfg.color_lo is not None else colors.min(axis=0))
    chi = np.asarray(cfg.color_hi if cfg.color_hi is not None else colors.max(axis=0))

    if cfg.morton_reorder:
        order = _morton_order(means[:, 0], means[:, 1], H, W)
        means, log_scales, theta, colors = (a[order] for a in (means, log_scales, theta, colors))

    q_means = _quant(means, [0.0, 0.0], [W - 1.0, H - 1.0], cfg.bits_means)
    if cfg.morton_reorder:  # delta along the Morton curve; zlib eats small deltas
        q_means = np.diff(q_means, axis=0, prepend=np.zeros((1, 2), np.uint32)).astype(np.uint32)
        q_means &= (1 << cfg.bits_means) - 1  # wraparound-safe modular deltas
    q_scales = _quant(log_scales, _SCALE_LO, scale_hi, cfg.bits_scales)
    q_rot = _quant(theta, 0.0, np.pi, cfg.bits_rot)
    q_colors = _quant(colors, clo, chi, cfg.bits_colors)

    streams = [
        _pack(q_means.T.ravel(), cfg.bits_means),   # planar: x deltas then y deltas
        _pack(q_scales.T.ravel(), cfg.bits_scales),
        _pack(q_rot, cfg.bits_rot),
        _pack(q_colors.T.ravel(), cfg.bits_colors),
    ]
    header = json.dumps({
        "n": int(n), "H": int(H), "W": int(W),
        "bits": [cfg.bits_means, cfg.bits_scales, cfg.bits_rot, cfg.bits_colors],
        "morton": bool(cfg.morton_reorder),
        "color_lo": clo.tolist(), "color_hi": chi.tolist(),
    }).encode()
    blob = _MAGIC + struct.pack("<I", len(header)) + header
    for s in streams:
        z = zlib.compress(s, cfg.zlib_level)
        blob += struct.pack("<I", len(z)) + z
    return blob


def decode(blob: bytes, device: str = "cpu") -> GaussianField:
    assert blob[:5] == _MAGIC, "not a structsplat codec blob"
    off = 5
    (hlen,) = struct.unpack_from("<I", blob, off)
    off += 4
    h = json.loads(blob[off:off + hlen])
    off += hlen
    n, H, W = h["n"], h["H"], h["W"]
    b_means, b_scales, b_rot, b_colors = h["bits"]

    raws = []
    for _ in range(4):
        (zlen,) = struct.unpack_from("<I", blob, off)
        off += 4
        raws.append(zlib.decompress(blob[off:off + zlen]))
        off += zlen

    q_means = _unpack(raws[0], b_means, 2 * n).reshape(2, n).T
    if h["morton"]:
        q_means = np.cumsum(q_means, axis=0, dtype=np.uint64).astype(np.uint32)
        q_means &= (1 << b_means) - 1
    q_scales = _unpack(raws[1], b_scales, 2 * n).reshape(2, n).T
    q_rot = _unpack(raws[2], b_rot, n)
    q_colors = _unpack(raws[3], b_colors, 3 * n).reshape(3, n).T

    means = _dequant(q_means, [0.0, 0.0], [W - 1.0, H - 1.0], b_means)
    log_scales = _dequant(q_scales, _SCALE_LO, float(np.log(max(H, W))), b_scales)
    theta = _dequant(q_rot, 0.0, np.pi, b_rot)
    colors = _dequant(q_colors, np.asarray(h["color_lo"]), np.asarray(h["color_hi"]), b_colors)

    def t(a):
        return torch.as_tensor(np.ascontiguousarray(a), device=device, dtype=torch.float32)

    return GaussianField(t(means), t(log_scales), t(theta), t(colors))


def _ste(x: torch.Tensor, lo, hi, bits: int) -> torch.Tensor:
    """Differentiable fake-quantization: forward rounds to the lattice, backward is identity."""
    levels = (1 << bits) - 1
    lo = torch.as_tensor(lo, device=x.device, dtype=x.dtype)
    hi = torch.as_tensor(hi, device=x.device, dtype=x.dtype)
    t = ((x - lo) / (hi - lo).clamp_min(1e-12)).clamp(0.0, 1.0)
    q = torch.round(t * levels) / levels * (hi - lo) + lo
    return x + (q - x).detach()


def quantized_view(field: GaussianField, H: int, W: int, cfg: CodecConfig) -> GaussianField:
    """A GaussianField whose parameters are fake-quantized with straight-through gradients."""
    scale_hi = float(np.log(max(H, W)))
    clo = torch.as_tensor(cfg.color_lo, dtype=field.colors.dtype, device=field.colors.device)
    chi = torch.as_tensor(cfg.color_hi, dtype=field.colors.dtype, device=field.colors.device)
    means = torch.stack([_ste(field.means[:, 0], 0.0, W - 1.0, cfg.bits_means),
                         _ste(field.means[:, 1], 0.0, H - 1.0, cfg.bits_means)], dim=1)
    log_scales = _ste(field.log_scales, _SCALE_LO, scale_hi, cfg.bits_scales)
    theta = _ste(torch.remainder(field.rotations, torch.pi), 0.0, float(np.pi), cfg.bits_rot)
    colors = _ste(field.colors, clo, chi, cfg.bits_colors)
    return GaussianField(means, log_scales, theta, colors)


def qat_finetune(field: GaussianField, target: torch.Tensor, fcfg: FitConfig,
                 ccfg: CodecConfig, iters: int = 150, verbose: bool = False) -> CodecConfig:
    """Fine-tune `field` in place through fake-quantized rendering (STE).

    Color ranges are frozen up front (a moving quantization range defeats convergence);
    the returned CodecConfig carries them and MUST be the one passed to encode().
    """
    H, W = target.shape[0], target.shape[1]
    if ccfg.color_lo is None or ccfg.color_hi is None:
        lo, hi = color_ranges(field)
        ccfg = CodecConfig(**{**ccfg.__dict__, "color_lo": lo, "color_hi": hi})
    field.trainable()
    opt = torch.optim.Adam(field.parameter_groups(fcfg.lr_means, fcfg.lr_scales,
                                                  fcfg.lr_rot, fcfg.lr_color))
    for it in range(iters):
        qf = quantized_view(field, H, W, ccfg)
        img = render(qf.means, qf.conics(fcfg.aa_dilation), qf.colors,
                     qf.radii(fcfg.sigma_cutoff, fcfg.aa_dilation), H, W, fcfg.render_chunk)
        pix = (img - target).abs().mean()
        loss = (1 - fcfg.ssim_weight) * pix + fcfg.ssim_weight * (1 - M.ssim(img, target))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        with torch.no_grad():
            field.log_scales.clamp_(_SCALE_LO, float(np.log(max(H, W))))
        if verbose and it % 50 == 0:
            print(f"  qat iter {it:4d} loss {loss.item():.5f}")
    return ccfg


@torch.no_grad()
def rd_point(field: GaussianField, target: torch.Tensor, fcfg: FitConfig,
             ccfg: CodecConfig | None = None) -> dict:
    """Encode -> decode -> measure. bpp is the actual bitstream size over H*W."""
    H, W = target.shape[0], target.shape[1]
    ccfg = ccfg or CodecConfig()
    blob = encode(field, H, W, ccfg)
    dec = decode(blob, device=str(target.device))
    img = render(dec.means, dec.conics(fcfg.aa_dilation), dec.colors,
                 dec.radii(fcfg.sigma_cutoff, fcfg.aa_dilation), H, W, fcfg.render_chunk)
    bits = [ccfg.bits_means, ccfg.bits_scales, ccfg.bits_rot, ccfg.bits_colors]
    raw_bits = field.n * (2 * bits[0] + 2 * bits[1] + bits[2] + 3 * bits[3])
    return {
        "bpp": 8.0 * len(blob) / (H * W),
        "raw_bpp": raw_bits / (H * W),
        "bytes": len(blob),
        "psnr": M.psnr(img, target),
        "ms_ssim": M.ms_ssim(img, target),
        "n_gaussians": field.n,
        "bits": bits,
    }
