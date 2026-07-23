#!/usr/bin/env python3
"""Run Janelle C0001 with the transactional mask-contained safe-commit schedule.

The runner intentionally starts from the same 4,500 WSE + 500 boundary initialization used by
``fit_janelle_complete_refinement.py`` so the schedule is the only changed variable.  It writes
accepted reconstruction/error checkpoints, the complete rejected-trial audit trail, the full
field, an optional byte-capped ``.rtgsv``, and one comparison ``index.html``.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import html
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import fit_janelle_complete_refinement as legacy  # noqa: E402
from structsplat.config import FitConfig  # noqa: E402
from structsplat.safe_schedule import (  # noqa: E402
    PhaseBudget,
    SafeScheduleConfig,
    run_safe_schedule,
)


DEFAULT_CAPTURE_ROOT = legacy.DEFAULT_CAPTURE_ROOT
DEFAULT_REALTIME_ROOT = legacy.DEFAULT_REALTIME_ROOT
DEFAULT_OUT = REPOSITORY_ROOT / "runs/janelle_C0001_safe_commit_schedule_20260723"


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPOSITORY_ROOT), *args],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _source_provenance(out: Path) -> dict[str, Any]:
    sources = (
        Path(__file__).resolve(),
        REPOSITORY_ROOT / "src/structsplat/safe_schedule.py",
        REPOSITORY_ROOT / "src/structsplat/fit.py",
        REPOSITORY_ROOT / "src/structsplat/config.py",
        REPOSITORY_ROOT / "scripts/fit_janelle_complete_refinement.py",
        REPOSITORY_ROOT / "scripts/fit_janelle_mask_contained.py",
    )
    records = []
    for source in sources:
        payload = source.read_bytes()
        relative = Path("provenance") / source.name
        destination = out / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        legacy._atomic_text(destination, payload.decode("utf-8"))
        records.append({
            "source": str(source),
            "snapshot": relative.as_posix(),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "bytes": len(payload),
        })
    status = _git("status", "--short")
    return {
        "commit": _git("rev-parse", "HEAD"),
        "status": status,
        "status_sha256": hashlib.sha256(status.encode()).hexdigest(),
        "executed_source_snapshots": records,
    }


def _phase(
    original: PhaseBudget,
    *,
    steps: int,
    block_steps: int,
    target: int | None,
) -> PhaseBudget:
    return replace(
        original,
        max_steps=int(steps),
        block_steps=min(int(block_steps), int(steps)),
        target_gaussians=target,
    )


def build_schedule(args: argparse.Namespace) -> SafeScheduleConfig:
    defaults = SafeScheduleConfig()
    return SafeScheduleConfig(
        capacity=int(args.capacity),
        coverage_target_gaussians=int(args.coverage_target),
        detail_target_gaussians=int(args.detail_target),
        coverage_tau=float(args.coverage_tau),
        boundary_band=float(args.boundary_band),
        interior_hole_target=float(args.interior_hole_target),
        boundary_hole_target=float(args.boundary_hole_target),
        stale_patience=int(args.stale_patience),
        event_min_count=int(args.event_min_count),
        recovery_steps=int(args.recovery_steps),
        event_spacing_px=float(args.event_spacing),
        event_oversample=defaults.event_oversample,
        coverage_birth_count=int(args.coverage_birth_count),
        detail_birth_count=int(args.detail_birth_count),
        detail_split_count=int(args.detail_split_count),
        boundary_birth_count=int(args.boundary_birth_count),
        redistribution_count=int(args.redistribution_count),
        redistribution_min_responsibility=float(args.max_donor_responsibility),
        split_shrink=defaults.split_shrink,
        hole_opacity=defaults.hole_opacity,
        covered_birth_opacity=defaults.covered_birth_opacity,
        merge_envelope_inflation=defaults.merge_envelope_inflation,
        refinement_policy=str(args.refinement_policy),
        local_start_phase=str(args.local_start_phase),
        local_seed_count=int(args.local_seed_count),
        local_neighbor_count=int(args.local_neighbor_count),
        topology_neighbor_count=int(args.topology_neighbor_count),
        boundary_recycle_at_capacity=bool(args.boundary_recycle_at_capacity),
        pareto_safe_checkpoints=bool(args.pareto_safe_checkpoints),
        pareto_checkpoint_every=int(args.pareto_checkpoint_every),
        event_color_solve=bool(args.event_color_solve),
        boundary_residual_mse_threshold=float(
            args.boundary_residual_mse_threshold
        ),
        boundary_residual_component_target=int(
            args.boundary_residual_component_target
        ),
        boundary_residual_min_pixels=int(args.boundary_residual_min_pixels),
        tolerances=replace(
            defaults.tolerances,
            guard_p99=bool(args.guard_p99),
        ),
        bootstrap=_phase(
            defaults.bootstrap,
            steps=args.bootstrap_steps,
            block_steps=args.block_steps,
            target=5_000,
        ),
        coverage=_phase(
            defaults.coverage,
            steps=args.coverage_steps,
            block_steps=args.block_steps,
            target=args.coverage_target,
        ),
        detail=_phase(
            defaults.detail,
            steps=args.detail_steps,
            block_steps=args.block_steps,
            target=args.detail_target,
        ),
        boundary=_phase(
            defaults.boundary,
            steps=args.boundary_steps,
            block_steps=args.block_steps,
            target=args.capacity,
        ),
        redistribution=_phase(
            defaults.redistribution,
            steps=args.redistribution_steps,
            block_steps=args.block_steps,
            target=args.capacity,
        ),
        polish=_phase(
            defaults.polish,
            steps=args.polish_steps,
            block_steps=args.block_steps,
            target=args.capacity,
        ),
    )


def _base_fit_config(args: argparse.Namespace) -> FitConfig:
    return FitConfig(
        iters=1,
        renderer="cuda_tiled",
        render_chunk=512,
        pixel_loss="l2",
        ssim_weight=0.0,
        loss_weighting="mask",
        mask_contain=True,
        mask_margin=float(args.mask_margin),
        mask_cap_mode="anisotropic",
        mask_cap_refresh_every=100,
        mask_undercoverage_band=float(args.boundary_band),
        mask_undercoverage_tau=float(args.coverage_tau),
        mask_undercoverage_every=int(args.coverage_loss_every),
        support_fade=True,
        checkpoint_policy="terminal",
        split_scale=0.35,
        split_oversample=8.0,
        split_min_spacing=1.0,
        densify_max_axis_ratio=6.0,
        densify_coherence_power=1.0,
        color_solve_lambda=1e-4,
        color_solve_maxiter=32,
        compute_lpips=False,
        log_every=1,
    )


def _write_index(
    out: Path,
    snapshots: list[dict[str, Any]],
    rejected: list[dict[str, Any]],
) -> None:
    rows = []
    for item in snapshots:
        snapshot = item["snapshot"]
        schedule_metrics = item["schedule_metrics"]
        rows.append(
            "<tr>"
            f"<td>{snapshot['ordinal']}</td>"
            f"<td>{html.escape(item['phase'])}<br>{html.escape(item['event'])}</td>"
            f"<td>{item['attempted_steps']}</td>"
            f"<td>{item['accepted_steps']}</td>"
            f"<td>{snapshot['n_gaussians']}</td>"
            f"<td>{schedule_metrics['foreground_psnr_db']:.3f}</td>"
            f"<td>{schedule_metrics['boundary_psnr_db']:.3f}</td>"
            f"<td>{100*schedule_metrics['interior_hole_fraction']:.3f}%</td>"
            f"<td>{100*schedule_metrics['boundary_hole_fraction']:.3f}%</td>"
            f"<td>{schedule_metrics['cvar99_mse']:.6f}</td>"
            f"<td>{schedule_metrics['p99_mse']:.6f}</td>"
            f"<td><img src='{html.escape(snapshot['reconstruction'])}'></td>"
            f"<td><img src='{html.escape(snapshot['absolute_error_x4'])}'></td>"
            "</tr>"
        )
    rejected_rows = []
    for item in rejected:
        candidate = item.get("candidate")
        candidate_text = "—"
        if candidate:
            candidate_text = (
                f"FG {candidate['foreground_psnr_db']:.3f} dB; "
                f"boundary {candidate['boundary_psnr_db']:.3f} dB"
            )
        rejected_rows.append(
            "<tr>"
            f"<td>{html.escape(item['phase'])}</td>"
            f"<td>{html.escape(item['event'])}</td>"
            f"<td>{html.escape(', '.join(item.get('reasons', [])))}</td>"
            f"<td>{html.escape(candidate_text)}</td>"
            "</tr>"
        )
    document = """<!doctype html><meta charset="utf-8">
<title>Janelle C0001 — safe-commit schedule</title>
<style>
body{font-family:system-ui,sans-serif;background:#101114;color:#eee;margin:20px}
table{border-collapse:collapse;width:100%;margin:18px 0}
th,td{border:1px solid #41444b;padding:6px;vertical-align:top}
th{position:sticky;top:0;background:#22252b;z-index:2}
img{width:480px;max-width:38vw;height:auto;background:#000}
.note{max-width:1200px;color:#c7cbd3;line-height:1.45}
.ok{color:#8fda9b}.bad{color:#ff9b9b}
</style>
<h1>Janelle frame_00008 / C0001 — transactional safe-commit schedule</h1>
<p class="note">Only accepted states appear in the image timeline. Every accepted state is
Pareto-nonworse in full-resolution foreground MSE, boundary MSE, CVaR99, reachable interior
holes, reachable boundary holes and outside-mask coverage (and p99 when its guard is enabled).
Rejected trials are audited below
and never mutate the selected field or its optimizer moments.</p>
<h2>Accepted reconstruction sequence</h2>
<table><thead><tr><th>#</th><th>phase/event</th><th>attempted steps</th>
<th>accepted steps</th><th>N</th><th>FG PSNR</th><th>boundary PSNR</th>
<th>interior holes</th><th>boundary holes</th><th>CVaR99 MSE</th>
<th>p99 MSE</th>
<th>reconstruction</th><th>error ×4</th></tr></thead><tbody>""" \
        + "\n".join(rows) + """</tbody></table>
<h2>Rejected trials</h2>
<table><thead><tr><th>phase</th><th>event</th><th>gate reason</th>
<th>trial metrics</th></tr></thead><tbody>""" \
        + ("\n".join(rejected_rows) if rejected_rows else
           "<tr><td colspan='4'>No rejected trials.</td></tr>") \
        + "</tbody></table>\n"
    legacy._atomic_text(out / "index.html", document)


def run(args: argparse.Namespace) -> None:
    import torch
    import torch.nn.functional as F

    started = time.perf_counter()
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        raise RuntimeError(f"output directory is not empty: {out}")
    out.mkdir(parents=True, exist_ok=True)
    repository = _source_provenance(out)

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("the native Janelle safe schedule requires CUDA")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)

    bridge = legacy._load_bridge(args.realtime_root.resolve())
    source = legacy._find_source(
        bridge, args.capture_root.resolve(), args.frame, args.view_id
    )
    prepared = bridge.CONVERTER.prepare_source_view(source)
    if prepared.alpha_crop is None:
        raise RuntimeError("prepared Janelle source lost its authoritative mask")
    mask_cpu = prepared.alpha_crop.bool().contiguous()
    mask = mask_cpu.to(device)
    target = (prepared.rgb * mask_cpu[..., None].to(prepared.rgb)).to(device).contiguous()
    height, width = mask.shape
    outside = (~mask).to(target)[None, None]
    near_outside = F.max_pool2d(outside, kernel_size=5, stride=1, padding=2)[0, 0] > 0
    visible_boundary = mask & near_outside

    field, _, init_cfg, tensor_cfg, _, boundary_added = legacy._initial_field(
        target, mask, source.seed, device, float(args.mask_margin)
    )
    if field.opacities is None:
        field.opacities = torch.full(
            (field.n,), 10.0, device=device, dtype=target.dtype
        )
    base_cfg = _base_fit_config(args)
    schedule = build_schedule(args)
    schedule.validate(field.n)

    checkpoints = out / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    field.save(str(checkpoints / "000_initial_5000.npz"))
    writer = legacy.ProgressWriter(
        out, target, mask, visible_boundary, base_cfg, args.preview_width
    )
    initial_snapshot = writer.snapshot(
        field,
        stage="initialization",
        global_iter=0,
        event="5000_with_boundary",
        native=False,
    )
    accepted_snapshots: list[dict[str, Any]] = []
    rejected_records: list[dict[str, Any]] = []
    checkpoint_ordinal = 1

    def observe(selected_field, record: dict[str, Any]) -> None:
        nonlocal checkpoint_ordinal
        if record["phase"] == "initialization":
            accepted_snapshots.append({
                "phase": record["phase"],
                "event": record["event"],
                "attempted_steps": record["global_attempted_steps"],
                "accepted_steps": record["global_accepted_steps"],
                "schedule_metrics": record["selected"],
                "snapshot": initial_snapshot,
            })
            return
        if not record["accepted"]:
            rejected_records.append(record)
            return
        # Phase-end markers duplicate the immediately preceding selected state. Keep them in the
        # JSON audit but avoid redundant JPEGs in the visual timeline.
        if record["event"] == "phase_end":
            return
        snapshot = writer.snapshot(
            selected_field,
            stage=record["phase"],
            global_iter=record["global_attempted_steps"],
            event=record["event"],
            native=False,
        )
        accepted_snapshots.append({
            "phase": record["phase"],
            "event": record["event"],
            "attempted_steps": record["global_attempted_steps"],
            "accepted_steps": record["global_accepted_steps"],
            "schedule_metrics": record["selected"],
            "snapshot": snapshot,
        })
        if record["event"] != "global_fit":
            selected_field.save(
                str(checkpoints / (
                    f"{checkpoint_ordinal:03d}_{record['phase']}_{record['event']}.npz"
                ))
            )
            checkpoint_ordinal += 1
        _write_index(out, accepted_snapshots, rejected_records)

    result = run_safe_schedule(
        field,
        target,
        mask_cpu,
        base_cfg,
        schedule,
        observer=observe,
        verbose=True,
    )
    field = result["field"]
    final_snapshot = writer.snapshot(
        field,
        stage="final_full_field",
        global_iter=result["attempted_steps"],
        event="terminal",
        native=True,
    )
    field_path = out / "C0001_safe_commit_full.npz"
    field.save(str(field_path))
    field_sha = bridge.file_sha256(field_path)

    # Optimizer tensors are intentionally not serialized in the reader-facing run record. Every
    # accepted/rejected transition and all selected metrics remain fully auditable.
    schedule_record = {
        key: value for key, value in result.items()
        if key not in ("field", "optimizer_state")
    }
    legacy._atomic_json(out / "schedule_history.json", schedule_record)

    archive_record = None
    if not args.no_archive:
        activity = (
            bridge.CONVERTER._component_activity(
                field, mask, height=height, width=width
            )
            .detach()
            .cpu()
            .numpy()
        )
        binding = bridge.CONVERTER._implementation_binding()
        digest_payload = {
            "schedule": asdict(schedule),
            "fit": asdict(base_cfg),
            "repository": repository,
        }
        fit_digest = hashlib.sha256(
            json.dumps(
                legacy._jsonable(digest_payload),
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        observation = bridge.field_to_observation(
            field,
            canvas_size=(prepared.camera.height, prepared.camera.width),
            fit_window=prepared.fit_window,
            blend_mode="normalized",
            sigma_cutoff=base_cfg.sigma_cutoff,
            support_fade_alpha=1.0,
            aa_dilation=base_cfg.aa_dilation,
            view_id=source.view_id,
            n_init=5_000,
            producer_version=binding["structsplat_version"],
            producer_source_digest=binding["structsplat_source_sha256"],
            fit_config_digest=fit_digest,
        ).to("cpu")
        archive_path = out / "gaussians2d/C0001.rtgsv"
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        strict_view = bridge.CONVERTER._save_with_backoff(
            archive_path,
            prepared,
            observation,
            np.asarray(activity, dtype=np.float64),
        )
        archive_render = bridge._render_selected(
            strict_view.observation, device, "cuda_tiled"
        )
        archive_metrics = legacy._metric_record(
            archive_render, target, mask, visible_boundary
        )
        archive_record = {
            "path": str(archive_path.relative_to(out)),
            "bytes": int(strict_view.bytes),
            "sha256": bridge.file_sha256(archive_path),
            "n_gaussians": int(strict_view.observation.n),
            "rows_removed_from_full_field": int(field.n - strict_view.observation.n),
            "metrics": archive_metrics,
            "selection": "legacy transport activity backoff after safe full-field optimization",
            "warning": (
                "transport backoff is still not boundary-error-aware; the full-field safe "
                "result is authoritative"
            ),
        }

    run_record = {
        "schema": "structsplat.janelle.safe_commit_schedule.v1",
        "source": bridge._source_record(source),
        "fit_window": list(prepared.fit_window),
        "fit_size": [width, height],
        "foreground_pixels": int(mask.sum()),
        "visible_boundary_pixels": int(visible_boundary.sum()),
        "initialization": {
            "general_quadtree_wse": 4_500,
            "boundary_residual_tangent": boundary_added,
            "total": 5_000,
            "init_config": asdict(init_cfg),
            "structure_tensor": asdict(tensor_cfg),
            "metrics": initial_snapshot,
        },
        "schedule": asdict(schedule),
        "fit_config": asdict(base_cfg),
        "result": {
            "metrics": result["metrics"],
            "attempted_steps": result["attempted_steps"],
            "accepted_steps": result["accepted_steps"],
            "converged": result["converged"],
            "seconds": result["seconds"],
            "full_field": {
                "path": field_path.name,
                "sha256": field_sha,
                "n_gaussians": int(field.n),
                "snapshot": final_snapshot,
            },
            "archive": archive_record,
        },
        "repository": repository,
        "environment": {
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "numpy": np.__version__,
        },
        "timing": {
            "total_seconds": time.perf_counter() - started,
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        },
    }
    legacy._atomic_json(out / "run_config.json", run_record)
    legacy._atomic_json(out / "summary.json", {
        "source": f"{args.frame}/{args.view_id.upper()}",
        "initial": initial_snapshot,
        "final": final_snapshot,
        "safe_metrics": result["metrics"],
        "archive": archive_record,
        "accepted_transitions": sum(
            1 for record in result["history"] if record["accepted"]
        ),
        "rejected_transitions": sum(
            1 for record in result["history"] if not record["accepted"]
        ),
    })
    _write_index(out, accepted_snapshots, rejected_records)
    print(
        f"\ncomplete: n={field.n}, "
        f"fg={result['metrics']['foreground_psnr_db']:.3f} dB, "
        f"boundary={result['metrics']['boundary_psnr_db']:.3f} dB, "
        f"attempted/accepted steps={result['attempted_steps']}/"
        f"{result['accepted_steps']}; out={out}",
        flush=True,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-root", type=Path, default=DEFAULT_CAPTURE_ROOT)
    parser.add_argument("--realtime-root", type=Path, default=DEFAULT_REALTIME_ROOT)
    parser.add_argument("--frame", default="frame_00008")
    parser.add_argument("--view-id", default="C0001")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--mask-margin", type=float, default=0.75)
    parser.add_argument("--preview-width", type=int, default=1200)
    parser.add_argument("--capacity", type=int, default=11_000)
    parser.add_argument("--coverage-target", type=int, default=8_000)
    parser.add_argument("--detail-target", type=int, default=10_000)
    parser.add_argument("--coverage-tau", type=float, default=0.05)
    parser.add_argument("--coverage-loss-every", type=int, default=8)
    parser.add_argument("--boundary-band", type=float, default=4.0)
    parser.add_argument("--interior-hole-target", type=float, default=1e-3)
    parser.add_argument("--boundary-hole-target", type=float, default=1e-2)
    parser.add_argument("--bootstrap-steps", type=int, default=1_250)
    parser.add_argument("--coverage-steps", type=int, default=3_000)
    parser.add_argument("--detail-steps", type=int, default=3_000)
    parser.add_argument("--boundary-steps", type=int, default=2_000)
    parser.add_argument("--redistribution-steps", type=int, default=2_000)
    parser.add_argument("--polish-steps", type=int, default=2_000)
    parser.add_argument("--block-steps", type=int, default=250)
    parser.add_argument("--recovery-steps", type=int, default=80)
    parser.add_argument("--stale-patience", type=int, default=3)
    parser.add_argument(
        "--refinement-policy",
        choices=("global", "local_neighborhood"),
        default="global",
    )
    parser.add_argument(
        "--local-start-phase",
        choices=("coverage", "detail", "boundary", "redistribution", "polish"),
        default="coverage",
    )
    parser.add_argument("--local-seed-count", type=int, default=384)
    parser.add_argument("--local-neighbor-count", type=int, default=8)
    parser.add_argument("--topology-neighbor-count", type=int, default=8)
    parser.add_argument("--boundary-recycle-at-capacity", action="store_true")
    parser.add_argument("--pareto-safe-checkpoints", action="store_true")
    parser.add_argument("--pareto-checkpoint-every", type=int, default=50)
    parser.add_argument("--event-color-solve", action="store_true")
    parser.add_argument("--guard-p99", action="store_true")
    parser.add_argument(
        "--boundary-residual-mse-threshold", type=float, default=1e-2
    )
    parser.add_argument(
        "--boundary-residual-component-target", type=int, default=0
    )
    parser.add_argument("--boundary-residual-min-pixels", type=int, default=4)
    parser.add_argument("--event-min-count", type=int, default=8)
    parser.add_argument("--event-spacing", type=float, default=5.0)
    parser.add_argument("--coverage-birth-count", type=int, default=512)
    parser.add_argument("--detail-birth-count", type=int, default=256)
    parser.add_argument("--detail-split-count", type=int, default=128)
    parser.add_argument("--boundary-birth-count", type=int, default=128)
    parser.add_argument("--redistribution-count", type=int, default=64)
    parser.add_argument("--max-donor-responsibility", type=float, default=0.25)
    parser.add_argument("--no-archive", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.mask_margin < 0.72:
        raise ValueError("--mask-margin must stay above the ~0.71 px containment floor")
    positive = (
        args.preview_width,
        args.capacity,
        args.coverage_target,
        args.detail_target,
        args.bootstrap_steps,
        args.coverage_steps,
        args.detail_steps,
        args.boundary_steps,
        args.redistribution_steps,
        args.polish_steps,
        args.block_steps,
        args.recovery_steps,
        args.local_seed_count,
        args.pareto_checkpoint_every,
        args.boundary_residual_min_pixels,
        args.event_min_count,
        args.coverage_loss_every,
    )
    if any(value <= 0 for value in positive):
        raise ValueError("all size, step, and event-count arguments must be positive")
    if not 5_000 <= args.coverage_target <= args.detail_target <= args.capacity:
        raise ValueError(
            "expected 5000 <= --coverage-target <= --detail-target <= --capacity"
        )
    if not 0.0 < args.max_donor_responsibility <= 1.0:
        raise ValueError("--max-donor-responsibility must be in (0, 1]")
    if args.local_neighbor_count < 0 or args.topology_neighbor_count < 0:
        raise ValueError("neighbor counts must be nonnegative")
    if args.boundary_residual_mse_threshold < 0.0:
        raise ValueError("--boundary-residual-mse-threshold must be nonnegative")
    if args.boundary_residual_component_target < 0:
        raise ValueError("--boundary-residual-component-target must be nonnegative")
    legacy._prepend_source_roots(args.realtime_root.resolve())
    run(args)


if __name__ == "__main__":
    main()
