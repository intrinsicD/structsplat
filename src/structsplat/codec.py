"""COMP-001: quantization + entropy coding of a fitted GaussianField (rate-distortion).

Encoding pipeline (GaussianImage-style uniform quantization, plus two representation-specific
tricks):
  * The normalized renderer is order-independent, so Gaussians are freely reordered along a
    Morton (Z-order) curve; positions are then delta-coded, which zlib compresses well.
  * Rotation is canonicalized to [0, pi): a 2D Gaussian is invariant under theta -> theta + pi.

Attribute layout per Gaussian (defaults): means 2x16 bit fixed-point over the fitted extent,
log-scales 2x8 bit over per-field [min, max] stored in the header, rotation 8 bit, colors 3x8 bit
over per-channel [min, max] stored in the header (colors are unbounded — opacity is folded in, so
the range is data-dependent). Streams are zlib-compressed separately.

`qat_finetune` runs a short straight-through-estimator fine-tune so the parameters settle onto
the quantization lattice before encoding (recovers most of the coarse-bit PSNR loss).

Requires torch (post-fit stage); the pack/unpack math is plain NumPy.
"""
from __future__ import annotations
from io import BytesIO
import json
import math
import struct
import zlib
from dataclasses import dataclass

import numpy as np
import torch

from .config import FitConfig
from .gaussians import GaussianField
from .render import render_field
from .fit import _pixel_loss
from . import metrics as M  # noqa: N812

_MAGIC = b"SSPL1"
_SCALE_LO = float(np.log(0.35))  # matches fit.py's log-scale clamp


def _ensure_constant_color_field(field: GaussianField) -> None:
    if field.color_grads is not None:
        raise ValueError(
            "StructSplat codec v1 does not support affine color fields; "
            "save NPZ fields or fit with color_basis='constant'"
        )


@dataclass
class CodecConfig:
    bits_means: int = 16      # per coordinate
    bits_scales: int = 8      # per log-scale
    bits_rot: int = 8
    bits_colors: int = 8      # per channel
    bits_opacity: int = 8     # only used when the field carries opacities
    morton_reorder: bool = True
    color_delta: bool = True  # delta-code colors along the encoded order (Morton by default)
    means_byte_planar: bool = True  # split uint16/uint32 mean streams into byte planes
    rot_circular: bool = True  # 2^bits bins over [0, pi), with wraparound to bin 0
    stream_codec: str = "zlib"  # zlib or png; png stores byte streams as Morton-ordered planes
    zlib_level: int = 9
    # per-channel color ranges; computed from data when None, fixed during QAT
    color_lo: list[float] | None = None
    color_hi: list[float] | None = None
    # per-axis log-scale ranges; computed from data when None, fixed during QAT
    scale_lo: list[float] | None = None
    scale_hi: list[float] | None = None


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


def _quant_circular(x: np.ndarray, period: float, bits: int) -> np.ndarray:
    levels = 1 << bits
    t = np.mod(x, period) / period
    return np.mod(np.round(t * levels).astype(np.int64), levels).astype(np.uint32)


def _dequant_circular(q: np.ndarray, period: float, bits: int) -> np.ndarray:
    levels = 1 << bits
    return ((q.astype(np.float64) / levels) * period).astype(np.float32)


def _pack_dtype(bits: int) -> np.dtype:
    return np.dtype(np.uint8 if bits <= 8 else (np.uint16 if bits <= 16 else np.uint32))


def _pack(q: np.ndarray, bits: int) -> bytes:
    return q.astype(_pack_dtype(bits)).tobytes()


def _unpack(raw: bytes, bits: int, count: int) -> np.ndarray:
    dtype = _pack_dtype(bits)
    return np.frombuffer(raw, dtype=dtype, count=count).astype(np.uint32)


def _pack_byte_planar(q: np.ndarray, bits: int) -> bytes:
    dtype = _pack_dtype(bits)
    raw = q.astype(dtype).tobytes()
    itemsize = dtype.itemsize
    if itemsize == 1:
        return raw
    return np.frombuffer(raw, dtype=np.uint8).reshape(-1, itemsize).T.ravel().tobytes()


def _unpack_byte_planar(raw: bytes, bits: int, count: int) -> np.ndarray:
    dtype = _pack_dtype(bits)
    itemsize = dtype.itemsize
    if itemsize == 1:
        return _unpack(raw, bits, count)
    planes = np.frombuffer(raw, dtype=np.uint8, count=count * itemsize).reshape(itemsize, count)
    interleaved = np.ascontiguousarray(planes.T)
    return np.frombuffer(interleaved.tobytes(), dtype=dtype, count=count).astype(np.uint32)


def _png_encode_bytes(raw: bytes) -> bytes:
    from PIL import Image

    arr = np.frombuffer(raw, dtype=np.uint8)
    side = max(1, int(math.ceil(math.sqrt(arr.size))))
    padded = np.zeros(side * side, dtype=np.uint8)
    padded[:arr.size] = arr
    image = Image.fromarray(padded.reshape(side, side))
    out = BytesIO()
    image.save(out, format="PNG", optimize=True)
    return out.getvalue()


def _png_decode_bytes(payload: bytes, raw_len: int) -> bytes:
    from PIL import Image

    image = Image.open(BytesIO(payload)).convert("L")
    arr = np.asarray(image, dtype=np.uint8).reshape(-1)
    if arr.size < raw_len:
        raise ValueError(f"PNG stream too short: got {arr.size} bytes, expected {raw_len}")
    return arr[:raw_len].tobytes()


def _encode_stream(raw: bytes, cfg: CodecConfig) -> bytes:
    if cfg.stream_codec == "zlib":
        return zlib.compress(raw, cfg.zlib_level)
    if cfg.stream_codec == "png":
        return _png_encode_bytes(raw)
    raise ValueError(f"unknown stream_codec {cfg.stream_codec!r}; expected 'zlib' or 'png'")


def _decode_stream(payload: bytes, codec: str, raw_len: int | None) -> bytes:
    if codec == "zlib":
        return zlib.decompress(payload)
    if codec == "png":
        if raw_len is None:
            raise ValueError("PNG stream requires stream_raw_lengths in the codec header")
        return _png_decode_bytes(payload, raw_len)
    raise ValueError(f"unknown stream_codec {codec!r}; expected 'zlib' or 'png'")


def _params(field: GaussianField):
    means = field.means.detach().cpu().numpy().astype(np.float64)
    # Codec v1 has no per-row covariance-filter stream. Fold the fixed isotropic variance into
    # the two RS scales; this preserves the exact rendered covariance without spending bits on
    # derivable birth-cohort metadata.
    log_scales = field.effective_log_scales().detach().cpu().numpy().astype(np.float64)
    theta = np.mod(field.rotations.detach().cpu().numpy().astype(np.float64), np.pi)
    colors = field.colors.detach().cpu().numpy().astype(np.float64)
    opacity_p = None
    if field.opacities is not None:  # store as probabilities: bounded [0,1] quantization range
        opacity_p = torch.sigmoid(field.opacities).detach().cpu().numpy().astype(np.float64)
    return means, log_scales, theta, colors, opacity_p


def color_ranges(field: GaussianField) -> tuple[list[float], list[float]]:
    c = field.colors.detach().cpu().numpy()
    return c.min(axis=0).tolist(), c.max(axis=0).tolist()


def scale_ranges(field: GaussianField) -> tuple[list[float], list[float]]:
    s = field.effective_log_scales().detach().cpu().numpy()
    return s.min(axis=0).tolist(), s.max(axis=0).tolist()


def _range_pair(
    lo: list[float] | None,
    hi: list[float] | None,
    fallback: tuple[list[float], list[float]],
    name: str,
) -> tuple[np.ndarray, np.ndarray]:
    if (lo is None) != (hi is None):
        raise ValueError(f"{name}_lo and {name}_hi must be provided together")
    if lo is None or hi is None:
        lo, hi = fallback
    return np.asarray(lo, dtype=np.float64), np.asarray(hi, dtype=np.float64)


def _means_extent(means: np.ndarray, H: int, W: int) -> tuple[list[float], list[float]]:
    """Quantization domain for means: the image box unioned with the fitted means' actual range.

    fit() never clamps means, so a Gaussian can sit off-image; quantizing over just [0, W-1]
    snapped it to the border with error unbounded by the lattice step. Covering the real extent
    (padded to the image box) bounds the round-trip error by one lattice step (COMP-002).
    """
    lo = [float(min(0.0, np.floor(means[:, 0].min()))), float(min(0.0, np.floor(means[:, 1].min())))]
    hi = [float(max(W - 1.0, np.ceil(means[:, 0].max()))),
          float(max(H - 1.0, np.ceil(means[:, 1].max())))]
    return lo, hi


def encode(field: GaussianField, H: int, W: int, cfg: CodecConfig | None = None,
           fcfg: FitConfig | None = None) -> bytes:
    _ensure_constant_color_field(field)
    cfg = cfg or CodecConfig()
    fcfg = fcfg or FitConfig()
    means, log_scales, theta, colors, opacity_p = _params(field)
    n = means.shape[0]
    clo, chi = _range_pair(cfg.color_lo, cfg.color_hi, (colors.min(axis=0), colors.max(axis=0)),
                           "color")
    slo, shi = _range_pair(
        cfg.scale_lo, cfg.scale_hi, (log_scales.min(axis=0), log_scales.max(axis=0)), "scale"
    )
    means_lo, means_hi = _means_extent(means, H, W)

    if cfg.morton_reorder:
        order = _morton_order(means[:, 0], means[:, 1], H, W)
        means, log_scales, theta, colors = (a[order] for a in (means, log_scales, theta, colors))
        if opacity_p is not None:
            opacity_p = opacity_p[order]

    q_means = _quant(means, means_lo, means_hi, cfg.bits_means)
    if cfg.morton_reorder:  # delta along the Morton curve; zlib eats small deltas
        q_means = np.diff(q_means, axis=0, prepend=np.zeros((1, 2), np.uint32)).astype(np.uint32)
        q_means &= (1 << cfg.bits_means) - 1  # wraparound-safe modular deltas
    q_scales = _quant(log_scales, slo, shi, cfg.bits_scales)
    q_rot = (
        _quant_circular(theta, np.pi, cfg.bits_rot)
        if cfg.rot_circular else _quant(theta, 0.0, np.pi, cfg.bits_rot)
    )
    q_colors = _quant(colors, clo, chi, cfg.bits_colors)
    color_delta = bool(cfg.color_delta and cfg.morton_reorder)
    if color_delta:
        q_colors = np.diff(q_colors, axis=0, prepend=np.zeros((1, 3), np.uint32)).astype(np.uint32)
        q_colors &= (1 << cfg.bits_colors) - 1

    q_means_flat = q_means.T.ravel()
    streams = [
        (
            _pack_byte_planar(q_means_flat, cfg.bits_means)
            if cfg.means_byte_planar else _pack(q_means_flat, cfg.bits_means)
        ),  # planar: x deltas then y deltas
        _pack(q_scales.T.ravel(), cfg.bits_scales),
        _pack(q_rot, cfg.bits_rot),
        _pack(q_colors.T.ravel(), cfg.bits_colors),
    ]
    if opacity_p is not None:
        streams.append(_pack(_quant(opacity_p, 0.0, 1.0, cfg.bits_opacity), cfg.bits_opacity))
    header_data = {
        "n": int(n), "H": int(H), "W": int(W),
        "bits": [cfg.bits_means, cfg.bits_scales, cfg.bits_rot, cfg.bits_colors],
        "morton": bool(cfg.morton_reorder),
        "color_delta": color_delta,
        "means_byte_planar": bool(cfg.means_byte_planar),
        "rot_circular": bool(cfg.rot_circular),
        "stream_codec": cfg.stream_codec,
        "stream_raw_lengths": [len(s) for s in streams],
        "color_lo": clo.tolist(), "color_hi": chi.tolist(),
        "scale_lo": slo.tolist(), "scale_hi": shi.tolist(),
        "means_lo": means_lo, "means_hi": means_hi,
        "has_opacity": opacity_p is not None,
        "bits_opacity": cfg.bits_opacity,
        # render semantics so the blob is self-describing (COMP-002): decode_and_render needs no
        # out-of-band FitConfig. scale_max is intentionally NOT stored — it is a fit-time
        # optimization cap, irrelevant to the frozen decoded field.
        "renderer": fcfg.renderer,
        "aa_dilation": float(fcfg.aa_dilation),
        "sigma_cutoff": float(fcfg.sigma_cutoff),
        "support_fade": bool(fcfg.support_fade),
        "render_chunk": int(fcfg.render_chunk),
    }
    # Keep default-off codec bytes exactly backward-compatible; the marker is diagnostic only
    # and is needed only when materialization actually occurred.
    if field.filter_variance is not None:
        header_data["covariance_filter_materialized"] = True
    header = json.dumps(header_data).encode()
    blob = _MAGIC + struct.pack("<I", len(header)) + header
    for s in streams:
        payload = _encode_stream(s, cfg)
        blob += struct.pack("<I", len(payload)) + payload
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
    has_opacity = bool(h.get("has_opacity", False))
    stream_codec = h.get("stream_codec", "zlib")
    stream_raw_lengths = h.get("stream_raw_lengths", [])

    raws = []
    for i in range(5 if has_opacity else 4):
        (zlen,) = struct.unpack_from("<I", blob, off)
        off += 4
        raw_len = int(stream_raw_lengths[i]) if i < len(stream_raw_lengths) else None
        raws.append(_decode_stream(blob[off:off + zlen], stream_codec, raw_len))
        off += zlen

    if h.get("means_byte_planar", False):
        q_means = _unpack_byte_planar(raws[0], b_means, 2 * n).reshape(2, n).T
    else:
        q_means = _unpack(raws[0], b_means, 2 * n).reshape(2, n).T
    if h["morton"]:
        q_means = np.cumsum(q_means, axis=0, dtype=np.uint64).astype(np.uint32)
        q_means &= (1 << b_means) - 1
    q_scales = _unpack(raws[1], b_scales, 2 * n).reshape(2, n).T
    q_rot = _unpack(raws[2], b_rot, n)
    q_colors = _unpack(raws[3], b_colors, 3 * n).reshape(3, n).T
    if h.get("color_delta", False):
        q_colors = np.cumsum(q_colors, axis=0, dtype=np.uint64).astype(np.uint32)
        q_colors &= (1 << b_colors) - 1

    # older blobs (pre-COMP-002) quantized means over the image box
    means_lo = h.get("means_lo", [0.0, 0.0])
    means_hi = h.get("means_hi", [W - 1.0, H - 1.0])
    means = _dequant(q_means, means_lo, means_hi, b_means)
    # older blobs (pre-COMP-003) quantized log-scales over the static fitter clamp range
    scale_lo = h.get("scale_lo", [_SCALE_LO, _SCALE_LO])
    scale_hi = h.get("scale_hi", [float(np.log(max(H, W))), float(np.log(max(H, W)))])
    log_scales = _dequant(q_scales, scale_lo, scale_hi, b_scales)
    theta = (
        _dequant_circular(q_rot, np.pi, b_rot)
        if h.get("rot_circular", False) else _dequant(q_rot, 0.0, np.pi, b_rot)
    )
    colors = _dequant(q_colors, np.asarray(h["color_lo"]), np.asarray(h["color_hi"]), b_colors)
    opacities = None
    if has_opacity:
        b_op = h["bits_opacity"]
        p = _dequant(_unpack(raws[4], b_op, n), 0.0, 1.0, b_op).clip(1e-4, 1.0 - 1e-4)
        opacities = np.log(p / (1.0 - p)).astype(np.float32)

    def t(a):
        return torch.as_tensor(np.ascontiguousarray(a), device=device, dtype=torch.float32)

    return GaussianField(t(means), t(log_scales), t(theta), t(colors),
                         None if opacities is None else t(opacities))


def blob_header(blob: bytes) -> dict:
    """Parse just the JSON header of a codec blob (n, H, W, bits, render semantics, ...)."""
    assert blob[:5] == _MAGIC, "not a structsplat codec blob"
    (hlen,) = struct.unpack_from("<I", blob, 5)
    return json.loads(blob[9:9 + hlen])


def decode_and_render(blob: bytes, device: str = "cpu") -> torch.Tensor:
    """Decode a blob and render it using the render semantics stored in its header.

    Self-contained: needs no out-of-band FitConfig (COMP-002). Returns an (H, W, 3) tensor.
    """
    h = blob_header(blob)
    field = decode(blob, device=device)
    H, W = h["H"], h["W"]
    aa = float(h.get("aa_dilation", 0.0))
    sigma = float(h.get("sigma_cutoff", 3.0))
    support_fade = bool(h.get("support_fade", False))
    mode = h.get("renderer", "normalized")
    chunk = int(h.get("render_chunk", 512))
    return render_field(field.means, field.conics(aa), field.colors,
                        field.radii(sigma, aa), H, W, chunk, mode, field.opacity_values(),
                        scales=field.scales(), rotations=field.rotations,
                        support_fade=support_fade, sigma_cutoff=sigma)


def _ste(x: torch.Tensor, lo, hi, bits: int) -> torch.Tensor:
    """Differentiable fake-quantization: forward rounds to the lattice, backward is identity."""
    levels = (1 << bits) - 1
    lo = torch.as_tensor(lo, device=x.device, dtype=x.dtype)
    hi = torch.as_tensor(hi, device=x.device, dtype=x.dtype)
    t = ((x - lo) / (hi - lo).clamp_min(1e-12)).clamp(0.0, 1.0)
    q = torch.round(t * levels) / levels * (hi - lo) + lo
    return x + (q - x).detach()


def _ste_circular(x: torch.Tensor, period: float, bits: int) -> torch.Tensor:
    levels = 1 << bits
    period_t = torch.as_tensor(period, device=x.device, dtype=x.dtype)
    wrapped = torch.remainder(x, period_t)
    q = torch.remainder(torch.round(wrapped / period_t * levels), levels) / levels * period_t
    return wrapped + (q - wrapped).detach()


def quantized_view(field: GaussianField, H: int, W: int, cfg: CodecConfig) -> GaussianField:
    """A GaussianField whose parameters are fake-quantized with straight-through gradients."""
    _ensure_constant_color_field(field)
    clo, chi = color_ranges(field) if cfg.color_lo is None or cfg.color_hi is None else (
        cfg.color_lo, cfg.color_hi
    )
    slo, shi = scale_ranges(field) if cfg.scale_lo is None or cfg.scale_hi is None else (
        cfg.scale_lo, cfg.scale_hi
    )
    slo = torch.as_tensor(slo, dtype=field.log_scales.dtype, device=field.log_scales.device)
    shi = torch.as_tensor(shi, dtype=field.log_scales.dtype, device=field.log_scales.device)
    clo = torch.as_tensor(clo, dtype=field.colors.dtype, device=field.colors.device)
    chi = torch.as_tensor(chi, dtype=field.colors.dtype, device=field.colors.device)
    means = torch.stack([_ste(field.means[:, 0], 0.0, W - 1.0, cfg.bits_means),
                         _ste(field.means[:, 1], 0.0, H - 1.0, cfg.bits_means)], dim=1)
    log_scales = _ste(field.effective_log_scales(), slo, shi, cfg.bits_scales)
    theta = (
        _ste_circular(field.rotations, float(np.pi), cfg.bits_rot)
        if cfg.rot_circular
        else _ste(torch.remainder(field.rotations, torch.pi), 0.0, float(np.pi), cfg.bits_rot)
    )
    colors = _ste(field.colors, clo, chi, cfg.bits_colors)
    opacities = None
    if field.opacities is not None:
        # quantize in probability space (matching encode), back through the logit so
        # opacity_values() reproduces the exact decoded probabilities
        p = _ste(torch.sigmoid(field.opacities), 0.0, 1.0, cfg.bits_opacity)
        p = p.clamp(1e-4, 1.0 - 1e-4)
        opacities = torch.log(p / (1.0 - p))
    return GaussianField(means, log_scales, theta, colors, opacities)


def qat_finetune(field: GaussianField, target: torch.Tensor, fcfg: FitConfig,
                 ccfg: CodecConfig, iters: int = 150, verbose: bool = False) -> CodecConfig:
    """Fine-tune `field` in place through fake-quantized rendering (STE).

    Color ranges are frozen up front (a moving quantization range defeats convergence);
    the returned CodecConfig carries them and MUST be the one passed to encode().
    """
    H, W = target.shape[0], target.shape[1]
    updates = {}
    if ccfg.color_lo is None or ccfg.color_hi is None:
        lo, hi = color_ranges(field)
        updates.update({"color_lo": lo, "color_hi": hi})
    if ccfg.scale_lo is None or ccfg.scale_hi is None:
        lo, hi = scale_ranges(field)
        updates.update({"scale_lo": lo, "scale_hi": hi})
    if updates:
        ccfg = CodecConfig(**{**ccfg.__dict__, **updates})
    scale_lo = torch.as_tensor(ccfg.scale_lo, dtype=field.log_scales.dtype,
                               device=field.log_scales.device)
    scale_hi = torch.as_tensor(ccfg.scale_hi, dtype=field.log_scales.dtype,
                               device=field.log_scales.device)
    field.trainable()
    opt = torch.optim.Adam(field.parameter_groups(fcfg.lr_means, fcfg.lr_scales,
                                                  fcfg.lr_rot, fcfg.lr_color, fcfg.lr_opacity))
    for it in range(iters):
        qf = quantized_view(field, H, W, ccfg)
        # render through the field's actual compositing model, not a hardcoded normalized one:
        # an additive/CUDA/gsplat-fit field must fine-tune under its own renderer (COMP-002)
        img = render_field(qf.means, qf.conics(fcfg.aa_dilation), qf.colors,
                           qf.radii(fcfg.sigma_cutoff, fcfg.aa_dilation), H, W, fcfg.render_chunk,
                           fcfg.renderer, qf.opacity_values(),
                           scales=qf.scales(), rotations=qf.rotations,
                           support_fade=fcfg.support_fade, sigma_cutoff=fcfg.sigma_cutoff)
        pix = _pixel_loss(img, target, fcfg.pixel_loss, fcfg.charbonnier_eps)
        loss = (1 - fcfg.ssim_weight) * pix + fcfg.ssim_weight * (1 - M.ssim(img, target))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        with torch.no_grad():
            if field.filter_variance is None:
                base_lo, base_hi = scale_lo, scale_hi
            else:
                variance = field.filter_variance.to(
                    device=field.log_scales.device, dtype=field.log_scales.dtype
                ).reshape(-1, 1)
                base_lo = 0.5 * torch.log(
                    (torch.exp(2.0 * scale_lo) - variance).clamp_min(1e-6)
                )
                base_hi = 0.5 * torch.log(
                    (torch.exp(2.0 * scale_hi) - variance).clamp_min(1e-6)
                )
            field.log_scales.copy_(
                torch.maximum(torch.minimum(field.log_scales, base_hi), base_lo)
            )
        if verbose and it % 50 == 0:
            print(f"  qat iter {it:4d} loss {loss.item():.5f}")
    return ccfg


@torch.no_grad()
def rd_point(field: GaussianField, target: torch.Tensor, fcfg: FitConfig,
             ccfg: CodecConfig | None = None) -> dict:
    """Encode -> decode -> measure. bpp is the actual bitstream size over H*W."""
    H, W = target.shape[0], target.shape[1]
    ccfg = ccfg or CodecConfig()
    blob = encode(field, H, W, ccfg, fcfg)                 # blob carries render semantics
    img = decode_and_render(blob, device=str(target.device))
    # display-referred RD: clamp to [0,1] before metrics so bpp-vs-quality is comparable to
    # display baselines (JPEG / GaussianImage) that cannot represent out-of-range values (COMP-002)
    img = img.clamp(0.0, 1.0)
    bits = [ccfg.bits_means, ccfg.bits_scales, ccfg.bits_rot, ccfg.bits_colors]
    raw_bits = field.n * (2 * bits[0] + 2 * bits[1] + bits[2] + 3 * bits[3])
    if field.opacities is not None:
        raw_bits += field.n * ccfg.bits_opacity
    return {
        "bpp": 8.0 * len(blob) / (H * W),
        "raw_bpp": raw_bits / (H * W),
        "bytes": len(blob),
        "psnr": M.psnr(img, target),
        "ms_ssim": M.ms_ssim(img, target),
        "n_gaussians": field.n,
        "bits": bits,
    }
