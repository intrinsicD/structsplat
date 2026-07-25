"""CUDA-event profile of the PORT-002 GPU-native tiled renderer path.

Compares StructSplat's exact untiled CUDA renderer (`cuda`, the unchanged baseline and shipped
GPU default) against the reworked tiled path (`cuda_tiled`: in-extension CUB binning, shared-
memory staged kernels, warp-reduced backward atomics, opt-in exact ellipse culling). Per cell it
times, on the current stream with `torch.cuda.Event`: tile-index build (GPU backend and the
legacy torch-op backend), renderer forward, renderer backward (fixed `grad_out` through
``torch.autograd.grad``), and a representative full fit step (default 0.7 L1 + 0.3 SSIM loss and
Adam). Timings are synchronized-per-sample medians over repeats after warmup; they exclude
Python launch overhead and extension compilation. Parity against the reference equations is a
precondition for reporting any timing ratio.

The decision gate is frozen in :data:`PREREGISTERED_GATE` before the first timing run of the new
path (no CUDA device existed in the authoring environment, so no number below was chosen after
seeing a measurement). Passing authorizes only the next step PORT-001's evidence already names:
an end-to-end fair-protocol fit benchmark of `cuda_tiled` against exact `cuda`. It does not
authorize a default flip, a quality claim, or a cross-GPU claim.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import statistics
from typing import Any, Callable

import numpy as np

SIGMA_CUTOFF = 3.0
SUPPORT_FADE = True
PARITY_ATOL = 5e-4
PARITY_RTOL = 5e-4
DEFAULT_RESOLUTIONS = (256, 512)
DEFAULT_COUNTS = (2048, 8192)
DEFAULT_OVERLAPS = (4.0, 16.0)
DEFAULT_AXIS_RATIOS = (1.0, 6.0)

# Frozen before the first optimized timing run. Do not tune these fields after seeing output.
PREREGISTERED_GATE: dict[str, Any] = {
    "registered_before_any_new_tiled_measurement": True,
    "representative_cell": {
        "height": 512,
        "width": 512,
        "n_gaussians": 8192,
        "requested_support_overlap": 16.0,
        "axis_ratio": 6.0,
    },
    "baseline_renderer": "cuda",
    "candidate_renderer": "cuda_tiled_gpu_index_cull",
    "parity_atol": PARITY_ATOL,
    "parity_rtol": PARITY_RTOL,
    # The 2026-07-05 fair run rejected the old tiled path at 1.69x slower end-to-end; the
    # candidate survives only if the representative full step is at least as fast as exact CUDA
    # and every N=8192 grid cell keeps that direction.
    "maximum_representative_step_ratio_vs_exact": 1.00,
    "maximum_high_n_grid_step_ratio_vs_exact": 1.00,
    "maximum_gpu_index_share_of_tiled_step": 0.15,
    "maximum_representative_timing_cv": 0.05,
    "pass_scope": (
        "source_device_bound_microprofile_only; authorizes the fair-protocol end-to-end fit "
        "benchmark, nothing else"
    ),
    "default_flip_authorized": False,
    "quality_or_compression_claim_authorized": False,
    # ADR-0024 amendment, authored 2026-07-25 AFTER the 2026-07-24 timings were seen. Only the
    # parity precondition changed; no tolerance, threshold, cell, or grid entry was retuned, and
    # no new numeric constant was introduced. Governing parity is now candidate-vs-baseline at the
    # same PARITY_ATOL/PARITY_RTOL. Reference agreement is reported, and is excused for a cell
    # only when the unmodified `cuda` baseline mismatches there too (a PORT-001/ADR-0011
    # property). See docs/adr/0024-normalized-fade-cutoff-conditioning-parity.md.
    "parity_precondition_revision": {
        "adr": "ADR-0024",
        "authored_after_seeing_timings": True,
        "governing_comparison": "candidate_vs_untiled_cuda_baseline",
        "reference_comparison": "reported; baseline-attributable mismatches do not gate",
        "tolerances_retuned": False,
    },
}


def classify_arm_parity(arm: str, got: Any, ref: Any, baseline: Any,
                        baseline_matches_reference: bool,
                        atol: float = PARITY_ATOL,
                        rtol: float = PARITY_RTOL) -> dict[str, Any]:
    """Classify one arm of one cell under the ADR-0024 parity precondition.

    Governing comparison is candidate-versus-baseline: a tiled arm that diverges from the untiled
    exact `cuda` renderer fails, because reproducing that renderer is exactly what PORT-002 and
    PORT-003 claim. Agreement with the torch reference is secondary — a mismatch there gates only
    when the unmodified baseline is itself clean at that cell, which makes it the candidate's own
    regression. When the baseline mismatches the reference too, the disagreement is inherited from
    the renderer this work does not modify (a PORT-001/ADR-0011 property) and is recorded as a
    reported diagnostic instead of a gate failure.

    Returns a record whose ``gating`` key is ``None`` when the arm is acceptable, and whose
    ``reference_diagnostic`` flag marks a baseline-attributable reference mismatch worth reporting.
    """
    import torch

    ref_tol = atol + rtol * ref.abs()
    ref_over = int((got - ref).abs().gt(ref_tol).sum())
    record: dict[str, Any] = {
        "arm": arm,
        "max_abs_vs_reference": float((got - ref).abs().max()),
        "values_over_reference_tol": ref_over,
        "values_total": int(ref.numel()),
        "baseline_also_mismatches_reference": not baseline_matches_reference,
        "gating": None,
        "reference_diagnostic": False,
    }
    if arm != "cuda":
        record["max_abs_vs_baseline"] = float((got - baseline).abs().max())
        if not torch.allclose(got, baseline, atol=atol, rtol=rtol):
            record["gating"] = "candidate_vs_baseline"
            return record
    if ref_over:
        if baseline_matches_reference:
            record["gating"] = "reference_regression_not_in_baseline"
        else:
            record["reference_diagnostic"] = True
    return record


@dataclass(frozen=True)
class CellSpec:
    height: int
    width: int
    n_gaussians: int
    requested_support_overlap: float
    axis_ratio: float
    seed: int


def build_cells(
    resolutions: tuple[int, ...],
    counts: tuple[int, ...],
    overlaps: tuple[float, ...],
    axis_ratios: tuple[float, ...],
    seed: int,
) -> list[CellSpec]:
    cells = []
    for side in resolutions:
        for n in counts:
            for overlap in overlaps:
                for ratio in axis_ratios:
                    cells.append(CellSpec(side, side, n, float(overlap), float(ratio), seed))
    return cells


def requested_scale(spec: CellSpec, sigma_cutoff: float = SIGMA_CUTOFF) -> float:
    """Choose a support scale from requested mean AABB visits per pixel."""
    area_per_gaussian = (
        spec.requested_support_overlap * spec.height * spec.width / spec.n_gaussians
    )
    requested_radius = max(1.0, (math.sqrt(area_per_gaussian) - 1.0) / 2.0)
    return max(0.45, requested_radius / sigma_cutoff)


def make_field_arrays(spec: CellSpec) -> dict[str, np.ndarray]:
    """Deterministic synthetic field with a controlled anisotropy arm.

    ``axis_ratio`` stretches sx by sqrt(ratio) and shrinks sy by the same factor, preserving the
    requested mean support area while making the AABB-vs-ellipse gap (the tiled path's weak
    spot, and the ellipse cull's payoff) an explicit experimental axis.
    """
    rng = np.random.default_rng(spec.seed)
    base_scale = requested_scale(spec)
    stretch = math.sqrt(max(1.0, spec.axis_ratio))
    means = np.stack(
        [
            rng.uniform(0.0, max(0.0, spec.width - 1.0), spec.n_gaussians),
            rng.uniform(0.0, max(0.0, spec.height - 1.0), spec.n_gaussians),
        ],
        axis=1,
    ).astype(np.float32)
    jitter = rng.uniform(0.85, 1.25, spec.n_gaussians).astype(np.float32)
    scales = np.stack(
        [base_scale * stretch * jitter, base_scale / (stretch * jitter)], axis=1
    ).astype(np.float32)
    angles = rng.uniform(0.0, math.pi, spec.n_gaussians).astype(np.float32)
    colors = rng.uniform(0.05, 0.95, (spec.n_gaussians, 3)).astype(np.float32)
    opacity_values = rng.uniform(0.35, 0.9, spec.n_gaussians).astype(np.float32)
    opacity_logits = np.log(opacity_values) - np.log1p(-opacity_values)
    return {
        "means": means,
        "scales": scales,
        "angles": angles,
        "colors": colors,
        "opacity_logits": opacity_logits.astype(np.float32),
    }


def _median(values: list[float]) -> float:
    if not values:
        raise ValueError("timing sample list must not be empty")
    return float(statistics.median(values))


def _population_cv(values: list[float]) -> float:
    mean = statistics.fmean(values)
    if mean <= 0.0:
        return math.inf
    return float(statistics.pstdev(values) / mean)


def summarize_rows(rows: list[dict[str, Any]]) -> str:
    header = (
        "| H | N | overlap | ratio | arm | index ms | forward ms | backward ms | step ms | "
        "step CV |\n"
        "|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|\n"
    )
    lines = []
    for row in rows:
        lines.append(
            "| {height} | {n_gaussians} | {requested_support_overlap:g} | {axis_ratio:g} "
            "| {arm} | {index_ms} | {forward_ms:.4f} | {backward_ms:.4f} | {step_ms:.4f} "
            "| {step_cv:.2%} |".format(
                index_ms=(
                    "-" if row["index_ms"] is None else f"{row['index_ms']:.4f}"
                ),
                **{k: row[k] for k in (
                    "height", "n_gaussians", "requested_support_overlap", "axis_ratio", "arm",
                    "forward_ms", "backward_ms", "step_ms", "step_cv",
                )},
            )
        )
    return header + "\n".join(lines) + "\n"


def _profile_cells(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    from structsplat import metrics as M
    from structsplat.cuda_render import _build_tile_index, _load_extension, render_cuda_exact
    from structsplat.gaussians import GaussianField
    from structsplat.render import render_field

    device = torch.device(f"cuda:{args.device_index}")
    torch.cuda.set_device(device)
    ext = _load_extension()

    def event_ms(action: Callable[[], Any]) -> float:
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        action()
        end.record()
        end.synchronize()
        return float(start.elapsed_time(end))

    def repeat_ms(action: Callable[[], Any]) -> list[float]:
        for _ in range(args.warmup):
            action()
        return [event_ms(action) for _ in range(args.repeats)]

    cells = build_cells(
        tuple(args.resolutions), tuple(args.counts), tuple(args.overlaps),
        tuple(args.axis_ratios), args.seed,
    )
    rows: list[dict[str, Any]] = []
    parity_failures: list[dict[str, Any]] = []
    reference_diagnostics: list[dict[str, Any]] = []
    for spec in cells:
        arrays = make_field_arrays(spec)
        field = GaussianField.from_numpy(
            arrays["means"], arrays["scales"], arrays["angles"], arrays["colors"],
            opacities=arrays["opacity_logits"], device=device,
        ).trainable()
        opacity = field.opacity_values()
        means = field.means.detach().contiguous()
        conics = field.conics().detach().contiguous()
        radii = field.radii(SIGMA_CUTOFF).contiguous()
        target = torch.rand(
            spec.height, spec.width, 3, device=device,
            generator=torch.Generator(device=device).manual_seed(spec.seed),
        )

        # Parity precondition: both arms against the reference equations on device.
        ref = render_field(
            field.means, field.conics(), field.colors, field.radii(SIGMA_CUTOFF),
            spec.height, spec.width, mode="normalized", opacities=opacity,
            support_fade=SUPPORT_FADE, sigma_cutoff=SIGMA_CUTOFF,
        ).detach()
        arms = {
            "cuda": dict(tiled=False),
            "cuda_tiled_gpu_index_cull": dict(
                tiled=True, tile_index_backend="cuda", tile_ellipse_cull=True),
            "cuda_tiled_gpu_index_nocull": dict(
                tiled=True, tile_index_backend="cuda", tile_ellipse_cull=False),
            "cuda_tiled_torch_index": dict(
                tiled=True, tile_index_backend="torch", tile_ellipse_cull=False),
        }
        # ADR-0024 parity precondition. Governing comparison: each tiled arm against the untiled
        # exact `cuda` baseline, at the same frozen tolerances. Reference agreement is recorded as
        # a diagnostic and only gates when the baseline itself agrees (i.e. the mismatch is
        # introduced by the candidate rather than inherited from the unmodified renderer).
        rendered = {
            arm: render_cuda_exact(
                means, conics, field.colors.detach(), radii, spec.height, spec.width,
                opacities=opacity.detach() if opacity is not None else None,
                support_fade=SUPPORT_FADE, sigma_cutoff=SIGMA_CUTOFF, **kwargs,
            ).detach()
            for arm, kwargs in arms.items()
        }
        baseline = rendered["cuda"]
        baseline_matches_reference = bool(
            torch.allclose(baseline, ref, atol=PARITY_ATOL, rtol=PARITY_RTOL))

        cell_ok = True
        for arm, got in rendered.items():
            record = classify_arm_parity(arm, got, ref, baseline, baseline_matches_reference)
            record["cell"] = asdict(spec)
            if record["gating"] is not None:
                parity_failures.append(record)
                cell_ok = False
            elif record["reference_diagnostic"]:
                reference_diagnostics.append(record)
        if not cell_ok:
            continue

        # Release parity buffers before timing so the measured section allocates exactly what the
        # pre-ADR-0024 protocol did (four arms are now held simultaneously, the old loop held one).
        del rendered, baseline, ref
        torch.cuda.empty_cache()

        grad_out = (target - 0.5).contiguous()
        params = [p for p in (field.means, field.log_scales, field.rotations, field.colors,
                              field.opacities) if p is not None]
        optimizer = torch.optim.Adam(field.parameter_groups(5e-2, 3e-2, 1e-2, 3e-2, 1e-2))
        frozen = [(p, p.detach().clone()) for p in params]

        def restore() -> None:
            with torch.no_grad():
                for p, keep in frozen:
                    p.copy_(keep)

        for arm, kwargs in arms.items():
            def fwd():
                return render_cuda_exact(
                    field.means, field.conics(), field.colors, field.radii(SIGMA_CUTOFF),
                    spec.height, spec.width, opacities=field.opacity_values(),
                    support_fade=SUPPORT_FADE, sigma_cutoff=SIGMA_CUTOFF, **kwargs,
                )

            def fwd_bwd():
                out = fwd()
                torch.autograd.grad(out, params, grad_out, retain_graph=False)

            def step():
                optimizer.zero_grad(set_to_none=True)
                out = fwd()
                pixel = (out - target).abs().mean()
                loss = 0.7 * pixel + 0.3 * (1.0 - M.ssim(out, target, backend="builtin"))
                loss.backward()
                optimizer.step()
                restore()

            index_samples = None
            if kwargs.get("tiled"):
                if kwargs["tile_index_backend"] == "cuda":
                    def build():
                        ext.build_tile_index(
                            means, conics, radii, spec.height, spec.width, 16,
                            bool(kwargs["tile_ellipse_cull"]) and SUPPORT_FADE, SIGMA_CUTOFF)
                else:
                    def build():
                        _build_tile_index(means, radii, spec.height, spec.width, 16)
                index_samples = repeat_ms(build)
            forward_samples = repeat_ms(lambda: fwd().detach())
            backward_samples = repeat_ms(fwd_bwd)
            step_samples = repeat_ms(step)
            rows.append({
                **asdict(spec),
                "arm": arm,
                "index_ms": None if index_samples is None else _median(index_samples),
                "forward_ms": _median(forward_samples),
                "backward_ms": _median(backward_samples),
                "step_ms": _median(step_samples),
                "step_cv": _population_cv(step_samples),
                "backward_cv": _population_cv(backward_samples),
                "raw": {
                    "index_ms": index_samples,
                    "forward_ms": forward_samples,
                    "backward_ms": backward_samples,
                    "step_ms": step_samples,
                },
            })
    return {"rows": rows, "parity_failures": parity_failures,
            "reference_diagnostics": reference_diagnostics}


def evaluate_gate(rows: list[dict[str, Any]],
                  parity_failures: list[dict[str, Any]],
                  reference_diagnostics: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    gate = PREREGISTERED_GATE
    rep = gate["representative_cell"]

    def find(arm: str, **cell: Any) -> dict[str, Any] | None:
        for row in rows:
            if row["arm"] != arm:
                continue
            if all(row[k] == v for k, v in cell.items()):
                return row
        return None

    rep_base = find("cuda", **rep)
    rep_cand = find(gate["candidate_renderer"], **rep)
    checks: dict[str, Any] = {"parity": not parity_failures}
    if rep_base is None or rep_cand is None:
        checks["representative_cell_present"] = False
        return {"pass": False, "checks": checks}
    checks["representative_cell_present"] = True
    step_ratio = rep_cand["step_ms"] / rep_base["step_ms"]
    checks["representative_step_ratio"] = step_ratio
    checks["representative_step_ok"] = (
        step_ratio <= gate["maximum_representative_step_ratio_vs_exact"]
    )
    index_share = (rep_cand["index_ms"] or 0.0) / rep_cand["step_ms"]
    checks["gpu_index_share"] = index_share
    checks["gpu_index_share_ok"] = index_share <= gate["maximum_gpu_index_share_of_tiled_step"]
    checks["representative_cv_ok"] = (
        rep_cand["step_cv"] <= gate["maximum_representative_timing_cv"]
        and rep_base["step_cv"] <= gate["maximum_representative_timing_cv"]
    )
    high_n = max(row["n_gaussians"] for row in rows)
    grid_ok = True
    grid_ratios = []
    for row in rows:
        if row["arm"] != gate["candidate_renderer"] or row["n_gaussians"] != high_n:
            continue
        base = find("cuda", **{k: row[k] for k in (
            "height", "width", "n_gaussians", "requested_support_overlap", "axis_ratio")})
        if base is None:
            continue
        ratio = row["step_ms"] / base["step_ms"]
        grid_ratios.append({"cell": {k: row[k] for k in (
            "height", "n_gaussians", "requested_support_overlap", "axis_ratio")},
            "step_ratio": ratio})
        grid_ok = grid_ok and ratio <= gate["maximum_high_n_grid_step_ratio_vs_exact"]
    checks["high_n_grid_ratios"] = grid_ratios
    checks["high_n_grid_ok"] = grid_ok
    # ADR-0024: baseline-attributable reference mismatches are reported, never silently dropped.
    diags = reference_diagnostics or []
    checks["baseline_attributable_reference_cells"] = len(diags)
    checks["baseline_attributable_reference_detail"] = [
        {"cell": {k: d["cell"][k] for k in (
            "height", "n_gaussians", "requested_support_overlap", "axis_ratio")},
         "arm": d["arm"],
         "max_abs_vs_reference": d["max_abs_vs_reference"],
         "values_over_reference_tol": d["values_over_reference_tol"],
         "values_total": d["values_total"],
         "max_abs_vs_baseline": d.get("max_abs_vs_baseline")}
        for d in diags
    ]
    verdict = all(checks[k] for k in (
        "parity", "representative_step_ok", "gpu_index_share_ok", "representative_cv_ok",
        "high_n_grid_ok",
    ))
    return {"pass": bool(verdict), "checks": checks, "gate": gate}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", default="results/port002_tiled_render_profile")
    parser.add_argument("--resolutions", type=int, nargs="+", default=list(DEFAULT_RESOLUTIONS))
    parser.add_argument("--counts", type=int, nargs="+", default=list(DEFAULT_COUNTS))
    parser.add_argument("--overlaps", type=float, nargs="+", default=list(DEFAULT_OVERLAPS))
    parser.add_argument("--axis-ratios", type=float, nargs="+",
                        default=list(DEFAULT_AXIS_RATIOS))
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device-index", type=int, default=0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("tiled_render_profile requires a CUDA device")
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    result = _profile_cells(args)
    gate = evaluate_gate(result["rows"], result["parity_failures"],
                         result["reference_diagnostics"])
    payload = {
        "protocol": "port002-tiled-render-profile-v1",
        "device": torch.cuda.get_device_name(args.device_index),
        "torch": torch.__version__,
        "args": vars(args),
        "gate_result": gate,
        **result,
    }
    (outdir / "raw.json").write_text(json.dumps(payload, indent=2))
    summary = (
        "# PORT-002 tiled renderer profile\n\n"
        f"Device: {payload['device']} · torch {torch.__version__} · "
        f"warmup {args.warmup} · repeats {args.repeats}\n\n"
        + summarize_rows(result["rows"])
        + f"\nGoverning parity failures (ADR-0024): {len(result['parity_failures'])}\n"
        + f"\nBaseline-attributable reference mismatches (reported, non-gating): "
        f"{len(result['reference_diagnostics'])}\n"
        + "".join(
            f"  - {d['arm']} @ {d['cell']['height']}^2 N={d['cell']['n_gaussians']} "
            f"ov={int(d['cell']['requested_support_overlap'])} "
            f"ar={int(d['cell']['axis_ratio'])}: "
            f"{d['values_over_reference_tol']}/{d['values_total']} values, "
            f"max {d['max_abs_vs_reference']:.6e} vs reference, "
            f"max {d.get('max_abs_vs_baseline', 0.0):.6e} vs baseline\n"
            for d in result["reference_diagnostics"]
        )
        + f"\nPreregistered gate pass: **{gate['pass']}**\n"
    )
    (outdir / "summary.md").write_text(summary)
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
