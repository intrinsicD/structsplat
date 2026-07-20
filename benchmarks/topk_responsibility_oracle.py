"""Oracle ceiling for per-pixel top-K normalized-renderer responsibility.

This benchmark reconstructs the current normalized renderer tile by tile, keeps only the K
largest unnormalized weights at each pixel, and reports the resulting reconstruction quality and
the *ideal* contribution-count reduction.  Because finding the winners still requires evaluating
all candidate weights in this implementation, the reported reduction is an oracle work ceiling,
not measured renderer acceleration.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shlex
import subprocess
import sys
import time
from typing import Any

import numpy as np
import torch

from benchmarks.common import load_image, write_csv
from structsplat import metrics as M
from structsplat.cuda_render import _build_tile_index
from structsplat.gaussians import GaussianField
from structsplat.render import _EPS, _support_weight, _tile_bounds, render_field


DEFAULT_IMAGE = "tests/test_images/COCO_train2014_000000000009.jpg"
DEFAULT_FIELD = (
    "results/coco4_current_20k_500/"
    "COCO_train2014_000000000009_aniso_flanking_20k_500.npz"
)
DEFAULT_KS = (1, 2, 4, 8, 16)
SOURCE_PATHS = (
    "benchmarks/topk_responsibility_oracle.py",
    "benchmarks/common.py",
    "src/structsplat/cuda_render.py",
    "src/structsplat/gaussians.py",
    "src/structsplat/metrics.py",
    "src/structsplat/render.py",
)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(args: list[str]) -> str | None:
    try:
        return subprocess.run(
            ["git", *args], check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _pixel_colors(
    field: GaussianField,
    candidates: torch.Tensor,
    dx: torch.Tensor,
    dy: torch.Tensor,
) -> torch.Tensor:
    """Return P x G x 3 colors, including the optional affine color basis."""
    base = field.colors[candidates].unsqueeze(0).expand(dx.shape[0], -1, -1)
    if field.color_grads is None:
        return base
    scales = field.scales()[candidates]
    angles = field.rotations[candidates]
    cosine, sine = torch.cos(angles), torch.sin(angles)
    local_x = (cosine * dx + sine * dy) / scales[:, 0].clamp_min(1e-6)
    local_y = (-sine * dx + cosine * dy) / scales[:, 1].clamp_min(1e-6)
    gradients = field.color_grads[candidates]
    return (
        base
        + local_x[..., None] * gradients[None, :, 0, :]
        + local_y[..., None] * gradients[None, :, 1, :]
    )


@torch.no_grad()
def topk_responsibility_oracle(
    field: GaussianField,
    height: int,
    width: int,
    *,
    ks: tuple[int, ...] = DEFAULT_KS,
    tile_size: int = 16,
    sigma_cutoff: float = 3.0,
    support_fade: bool = False,
    aa_dilation: float = 0.0,
) -> dict[str, Any]:
    """Render exact full and oracle top-K normalized mixtures.

    Responsibility ordering is identical to weight ordering at a fixed pixel because every
    candidate shares the same normalization denominator.  The full reconstruction uses the
    renderer's exact rounded, clipped support rectangles.  ``positive_weight_contributions`` can
    be smaller than the rectangle count because float32 ``exp(-q/2)`` may underflow to zero in
    extreme rectangle corners; no additional hard cutoff is applied.
    """
    if height <= 0 or width <= 0:
        raise ValueError("height and width must be positive")
    if tile_size <= 0 or tile_size * tile_size > 1024:
        raise ValueError("tile_size must be positive with tile_size^2 <= 1024")
    resolved_ks = tuple(sorted(set(int(k) for k in ks)))
    if not resolved_ks or resolved_ks[0] <= 0:
        raise ValueError("ks must contain positive integers")
    if field.n <= 0:
        raise ValueError("the oracle requires a non-empty Gaussian field")

    device, dtype = field.means.device, field.means.dtype
    means = field.means.detach()
    conics = field.conics(aa_dilation).detach()
    radii = field.radii(sigma_cutoff, aa_dilation)
    opacities = field.opacity_values()
    if opacities is not None:
        opacities = opacities.detach().to(device=device, dtype=dtype)

    tile_gids, tile_offsets = _build_tile_index(means, radii, height, width, tile_size)
    offsets = tile_offsets.detach().cpu().tolist()
    x0, y0, support_width, support_area = _tile_bounds(
        means, radii, height, width
    )
    support_height = torch.where(
        support_width > 0,
        torch.div(support_area, support_width.clamp_min(1), rounding_mode="floor"),
        torch.zeros_like(support_width),
    )

    pixels = height * width
    full = torch.zeros(pixels, 3, device=device, dtype=dtype)
    full_denominator = torch.zeros(pixels, device=device, dtype=dtype)
    topk = {
        k: torch.zeros(pixels, 3, device=device, dtype=dtype) for k in resolved_ks
    }
    topk_denominators = {
        k: torch.zeros(pixels, device=device, dtype=dtype) for k in resolved_ks
    }
    active_per_pixel = torch.zeros(pixels, device=device, dtype=torch.int64)
    rectangle_per_pixel = torch.zeros(pixels, device=device, dtype=torch.int64)
    candidate_per_pixel = torch.zeros(pixels, device=device, dtype=torch.int64)
    retained_counts = {k: 0 for k in resolved_ks}
    rectangle_evaluations = 0
    positive_weight_contributions = 0
    tile_candidate_evaluations = 0

    tiles_x = (width + tile_size - 1) // tile_size
    num_tiles = len(offsets) - 1
    for tile in range(num_tiles):
        candidates = tile_gids[offsets[tile]:offsets[tile + 1]]
        if candidates.numel() == 0:
            continue
        tile_x, tile_y = tile % tiles_x, tile // tiles_x
        px_range = torch.arange(
            tile_x * tile_size,
            min((tile_x + 1) * tile_size, width),
            device=device,
        )
        py_range = torch.arange(
            tile_y * tile_size,
            min((tile_y + 1) * tile_size, height),
            device=device,
        )
        py_grid, px_grid = torch.meshgrid(py_range, px_range, indexing="ij")
        px, py = px_grid.reshape(-1), py_grid.reshape(-1)
        flat = py * width + px

        dx = px[:, None].to(dtype) - means[candidates, 0]
        dy = py[:, None].to(dtype) - means[candidates, 1]
        candidate_conics = conics[candidates]
        q = (
            candidate_conics[:, 0] * dx.square()
            + 2.0 * candidate_conics[:, 1] * dx * dy
            + candidate_conics[:, 2] * dy.square()
        )
        rectangle_mask = (
            (px[:, None] >= x0[candidates])
            & (px[:, None] < x0[candidates] + support_width[candidates])
            & (py[:, None] >= y0[candidates])
            & (py[:, None] < y0[candidates] + support_height[candidates])
        )
        weights = _support_weight(q, sigma_cutoff, support_fade) * rectangle_mask
        if opacities is not None:
            weights = weights * opacities[candidates]
        pixel_colors = _pixel_colors(field, candidates, dx, dy)

        denominator = weights.sum(dim=1)
        numerator = (weights[..., None] * pixel_colors).sum(dim=1)
        full[flat] = numerator / (denominator[:, None] + _EPS)
        full_denominator[flat] = denominator

        rectangle_count = rectangle_mask.sum(dim=1)
        active_count = (weights > 0).sum(dim=1)
        rectangle_per_pixel[flat] = rectangle_count
        active_per_pixel[flat] = active_count
        candidate_per_pixel[flat] = candidates.numel()
        rectangle_evaluations += int(rectangle_count.sum().item())
        positive_weight_contributions += int(active_count.sum().item())
        tile_candidate_evaluations += int(weights.numel())

        largest = min(resolved_ks[-1], int(candidates.numel()))
        values, indices = torch.topk(weights, largest, dim=1, largest=True, sorted=True)
        selected_colors = torch.gather(
            pixel_colors, 1, indices[..., None].expand(-1, -1, 3)
        )
        cumulative_denominator = values.cumsum(dim=1)
        cumulative_numerator = (values[..., None] * selected_colors).cumsum(dim=1)
        for k in resolved_ks:
            retained = min(k, largest)
            denominator_k = cumulative_denominator[:, retained - 1]
            topk[k][flat] = (
                cumulative_numerator[:, retained - 1]
                / (denominator_k[:, None] + _EPS)
            )
            topk_denominators[k][flat] = denominator_k
            retained_counts[k] += int((values[:, :retained] > 0).sum().item())

    return {
        "full": full.view(height, width, 3),
        "full_denominator": full_denominator.view(height, width),
        "topk": {k: value.view(height, width, 3) for k, value in topk.items()},
        "topk_denominators": {
            k: value.view(height, width) for k, value in topk_denominators.items()
        },
        "active_per_pixel": active_per_pixel.view(height, width),
        "rectangle_per_pixel": rectangle_per_pixel.view(height, width),
        "candidate_per_pixel": candidate_per_pixel.view(height, width),
        "retained_counts": retained_counts,
        "rectangle_evaluations": rectangle_evaluations,
        "positive_weight_contributions": positive_weight_contributions,
        "tile_candidate_evaluations": tile_candidate_evaluations,
        "tile_references": int(tile_gids.numel()),
        "num_tiles": num_tiles,
    }


def _quantiles(values: torch.Tensor) -> dict[str, float]:
    if values.numel() == 0:
        return {name: float("nan") for name in ("min", "p05", "p50", "p95", "max")}
    probs = torch.tensor([0.0, 0.05, 0.5, 0.95, 1.0], device=values.device)
    result = torch.quantile(values.float(), probs).detach().cpu().tolist()
    return dict(zip(("min", "p05", "p50", "p95", "max"), map(float, result)))


def _source_provenance(root: Path) -> dict[str, Any]:
    hashes = {}
    combined = hashlib.sha256()
    for relative in SOURCE_PATHS:
        path = root / relative
        value = _sha256(path)
        hashes[relative] = value
        combined.update(relative.encode("utf-8") + b"\0" + value.encode("ascii") + b"\n")
    status = _git(["status", "--porcelain", "--", *SOURCE_PATHS])
    return {
        "git_commit": _git(["rev-parse", "HEAD"]),
        "relevant_sources_dirty": bool(status),
        "relevant_source_status": status.splitlines() if status else [],
        "combined_sha256": combined.hexdigest(),
        "files": hashes,
    }


def _environment(device: torch.device) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "platform": platform.platform(),
        "python": sys.version,
        "python_executable": sys.executable,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "numpy": np.__version__,
        "device": str(device),
        "environment": {
            name: os.environ.get(name)
            for name in (
                "CUDA_VISIBLE_DEVICES",
                "LD_PRELOAD",
                "PYTHONPATH",
                "TORCH_EXTENSIONS_DIR",
            )
        },
    }
    if device.type == "cuda":
        payload["cuda_device_name"] = torch.cuda.get_device_name(device)
        payload["cuda_capability"] = list(torch.cuda.get_device_capability(device))
    return payload


def _reproduction_command(args: argparse.Namespace) -> str:
    command = [
        "python", "-m", "benchmarks.topk_responsibility_oracle",
        "--image", args.image,
        "--field", args.field,
        "--outdir", args.outdir,
        "--device", args.device,
        "--validation-renderer", args.validation_renderer,
        "--ks", *[str(k) for k in args.ks],
        "--tile-size", str(args.tile_size),
        "--sigma-cutoff", str(args.sigma_cutoff),
        "--aa-dilation", str(args.aa_dilation),
        "--parity-max-abs", str(args.parity_max_abs),
    ]
    if args.support_fade:
        command.append("--support-fade")
    prefixes = []
    for name in ("LD_PRELOAD", "PYTHONPATH", "TORCH_EXTENSIONS_DIR"):
        if os.environ.get(name):
            prefixes.append(f"{name}={shlex.quote(os.environ[name])}")
    return " ".join([*prefixes, shlex.join(command)])


def _write_summary(
    outdir: Path,
    *,
    config: dict[str, Any],
    full_metrics: dict[str, float],
    validation: dict[str, Any],
    diagnostics: dict[str, Any],
    rows: list[dict[str, Any]],
) -> None:
    lines = [
        "# Top-K responsibility oracle ceiling",
        "",
        "This is an **oracle work ceiling, not a measured speedup**. The implementation first",
        "evaluates every current tile candidate and only then selects the per-pixel top-K weights.",
        "A deployable kernel would need a cheaper exact/approximate winner-search mechanism.",
        "",
        "The oracle matches the normalized renderer's rounded, clipped support rectangles. It",
        "does not add a q cutoff. `positive_weight_contributions` excludes only weights that are",
        "numerically zero (including float32 exponential underflow or support fade).",
        "",
        f"Input field: `{config['field']}` ({config['n_gaussians']:,} Gaussians).",
        f"Input image: `{config['image']}` ({config['width']}x{config['height']}).",
        f"Full oracle quality: {full_metrics['target_psnr']:.5f} dB PSNR, "
        f"{full_metrics['target_ms_ssim']:.6f} MS-SSIM.",
        f"Full-oracle / `{config['validation_renderer']}` parity: "
        f"max abs {validation['max_abs']:.3g}, mean abs {validation['mean_abs']:.3g} "
        f"(gate {validation['max_abs_gate']:.3g}, passed={validation['passed']}).",
        "",
        "| K | Target PSNR | Delta vs full | MS-SSIM | PSNR to full | Mean abs to full | "
        "Mean mass | p05 mass | Ideal positive-work reduction | Ideal rectangle-work reduction |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['k']} | {row['target_psnr']:.5f} | "
            f"{row['target_psnr_delta_vs_full']:+.5f} | {row['target_ms_ssim']:.6f} | "
            f"{row['psnr_to_full']:.4f} | {row['mean_abs_to_full']:.6f} | "
            f"{row['retained_responsibility_mean']:.6f} | "
            f"{row['retained_responsibility_p05']:.6f} | "
            f"{row['ideal_positive_work_reduction_fraction']:.2%} | "
            f"{row['ideal_rectangle_work_reduction_fraction']:.2%} |"
        )
    lines.extend([
        "",
        f"Rectangle contributions: {diagnostics['rectangle_evaluations']:,}; numerically "
        f"positive contributions: {diagnostics['positive_weight_contributions']:,}; tile "
        f"candidate-pixel pairs actually formed by this audit: "
        f"{diagnostics['tile_candidate_evaluations']:,}.",
        "",
        "The two reduction columns count retained color/normalization contributions relative to",
        "positive weights and current clipped-rectangle evaluations. They assume an oracle reveals",
        "the winners for free and therefore must not be quoted as throughput, latency, or FLOP",
        "measurements. `audit_seconds` in `result.json` is provenance only.",
        "",
        "## Reproduce",
        "",
        "```bash",
        config["reproduction_command"],
        "```",
        "",
        f"Relevant-source combined SHA-256: `{config['source']['combined_sha256']}`.",
        f"Field SHA-256: `{config['field_sha256']}`.",
        f"Image SHA-256: `{config['image_sha256']}`.",
        "",
        "Machine-readable outputs: `rows.csv`, `rows.json`, `result.json`, and `config.json`.",
        "",
    ])
    outdir.joinpath("summary.md").write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> list[dict[str, Any]]:
    root = Path(__file__).resolve().parents[1]
    image_path = Path(args.image)
    field_path = Path(args.field)
    if not image_path.is_file():
        raise FileNotFoundError(image_path)
    if not field_path.is_file():
        raise FileNotFoundError(field_path)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    device = torch.device(args.device)
    if args.validation_renderer.startswith("cuda") and device.type != "cuda":
        raise ValueError("a CUDA validation renderer requires --device cuda")

    image = load_image(image_path)
    height, width = image.shape[:2]
    target = torch.as_tensor(image, device=device, dtype=torch.float32)
    field = GaussianField.load(str(field_path), device=device)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    resolved = argparse.Namespace(**vars(args))
    resolved.ks = tuple(sorted(set(args.ks)))
    config = {
        "protocol": "topk_responsibility_oracle_v1",
        "claim_scope": "single-field mechanism audit; oracle work ceiling, not measured speed",
        "image": str(image_path),
        "image_sha256": _sha256(image_path),
        "field": str(field_path),
        "field_sha256": _sha256(field_path),
        "width": width,
        "height": height,
        "n_gaussians": field.n,
        "ks": list(resolved.ks),
        "tile_size": args.tile_size,
        "sigma_cutoff": args.sigma_cutoff,
        "support_fade": args.support_fade,
        "aa_dilation": args.aa_dilation,
        "validation_renderer": args.validation_renderer,
        "parity_max_abs": args.parity_max_abs,
        "responsibility_definition": (
            "w_i(p) / sum_j w_j(p); ordering equals unnormalized w_i(p) ordering"
        ),
        "support_definition": (
            "exact renderer rounded/clipped rectangles; no added q cutoff; positive count "
            "excludes numerically zero float32 weights"
        ),
        "reproduction_command": _reproduction_command(resolved),
        "source": _source_provenance(root),
        "environment": _environment(device),
    }
    outdir.joinpath("config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )

    _sync(device)
    started = time.perf_counter()
    oracle = topk_responsibility_oracle(
        field,
        height,
        width,
        ks=resolved.ks,
        tile_size=args.tile_size,
        sigma_cutoff=args.sigma_cutoff,
        support_fade=args.support_fade,
        aa_dilation=args.aa_dilation,
    )
    _sync(device)
    audit_seconds = time.perf_counter() - started

    reference = render_field(
        field.means,
        field.conics(args.aa_dilation),
        field.colors,
        field.radii(args.sigma_cutoff, args.aa_dilation),
        height,
        width,
        mode=args.validation_renderer,
        opacities=field.opacity_values(),
        scales=field.effective_scales(args.aa_dilation),
        rotations=field.rotations,
        support_fade=args.support_fade,
        sigma_cutoff=args.sigma_cutoff,
        color_grads=field.color_grads,
    )
    full = oracle["full"]
    difference = (full - reference).abs()
    validation = {
        "renderer": args.validation_renderer,
        "max_abs": float(difference.max().item()),
        "mean_abs": float(difference.mean().item()),
        "psnr_to_validator": M.psnr(full, reference),
        "max_abs_gate": args.parity_max_abs,
    }
    validation["passed"] = validation["max_abs"] <= args.parity_max_abs
    if not validation["passed"]:
        raise RuntimeError(
            f"full oracle parity failed: max abs {validation['max_abs']:.6g} > "
            f"{args.parity_max_abs:.6g}"
        )

    full_metrics = {
        "target_psnr": M.psnr(full, target),
        "target_ms_ssim": M.ms_ssim(full, target),
    }
    covered = oracle["full_denominator"] > 0
    rows: list[dict[str, Any]] = []
    for k in resolved.ks:
        reconstruction = oracle["topk"][k]
        retained_mass = (
            oracle["topk_denominators"][k][covered]
            / oracle["full_denominator"][covered]
        )
        mass_quantiles = _quantiles(retained_mass)
        retained = oracle["retained_counts"][k]
        target_psnr = M.psnr(reconstruction, target)
        rows.append({
            "k": k,
            "target_psnr": target_psnr,
            "target_psnr_delta_vs_full": target_psnr - full_metrics["target_psnr"],
            "target_ms_ssim": M.ms_ssim(reconstruction, target),
            "psnr_to_full": M.psnr(reconstruction, full),
            "mean_abs_to_full": float((reconstruction - full).abs().mean().item()),
            "max_abs_to_full": float((reconstruction - full).abs().max().item()),
            "retained_responsibility_mean": float(retained_mass.mean().item()),
            "retained_responsibility_min": mass_quantiles["min"],
            "retained_responsibility_p05": mass_quantiles["p05"],
            "retained_responsibility_p50": mass_quantiles["p50"],
            "retained_responsibility_p95": mass_quantiles["p95"],
            "retained_responsibility_max": mass_quantiles["max"],
            "retained_contributions": retained,
            "ideal_positive_work_reduction_fraction": 1.0 - (
                retained / oracle["positive_weight_contributions"]
            ),
            "ideal_rectangle_work_reduction_fraction": 1.0 - (
                retained / oracle["rectangle_evaluations"]
            ),
        })

    diagnostics = {
        "audit_seconds_not_renderer_speed": audit_seconds,
        "rectangle_evaluations": oracle["rectangle_evaluations"],
        "positive_weight_contributions": oracle["positive_weight_contributions"],
        "tile_candidate_evaluations": oracle["tile_candidate_evaluations"],
        "tile_references": oracle["tile_references"],
        "num_tiles": oracle["num_tiles"],
        "covered_pixels": int(covered.sum().item()),
        "active_contributions_per_pixel": _quantiles(
            oracle["active_per_pixel"].float()
        ),
        "rectangle_contributions_per_pixel": _quantiles(
            oracle["rectangle_per_pixel"].float()
        ),
        "tile_candidates_per_pixel": _quantiles(
            oracle["candidate_per_pixel"].float()
        ),
    }
    result = {
        "config": config,
        "full_metrics": full_metrics,
        "validation": validation,
        "diagnostics": diagnostics,
        "rows": rows,
    }
    write_csv(outdir / "rows.csv", rows)
    outdir.joinpath("rows.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )
    outdir.joinpath("result.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    _write_summary(
        outdir,
        config=config,
        full_metrics=full_metrics,
        validation=validation,
        diagnostics=diagnostics,
        rows=rows,
    )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--field", default=DEFAULT_FIELD)
    parser.add_argument(
        "--outdir", default="results/topk_responsibility_oracle_2026-07-15"
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--validation-renderer", default="cuda")
    parser.add_argument("--ks", type=int, nargs="+", default=list(DEFAULT_KS))
    parser.add_argument("--tile-size", type=int, default=16)
    parser.add_argument("--sigma-cutoff", type=float, default=3.0)
    parser.add_argument("--support-fade", action="store_true")
    parser.add_argument("--aa-dilation", type=float, default=0.0)
    parser.add_argument("--parity-max-abs", type=float, default=2e-5)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
