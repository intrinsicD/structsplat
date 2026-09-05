#!/usr/bin/env python3
"""HIER-033 finite count-funded operator oracle.
Formal: python scripts/experiments/hier033_operator_oracle.py OUT --approved-protocol-digest DIGEST
Wiring: add --smoke, using translation perturbation77 and two recovery steps only.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
import math
import platform
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for directory in (ROOT, ROOT / "src"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from benchmarks.hier_additive_controls import (  # noqa: E402
    ControlConfig, additive_render, fit_control, pack,
)
from benchmarks.hier_operator_oracle import (  # noqa: E402
    DAMPING, DONORS, FAMILIES, MAGNITUDES, SHAPE, TRUST, action_names, finite_actions, fixture,
)
from benchmarks.hier_research_report import (  # noqa: E402
    ResearchBundle, protocol_digest, save_rgb, sha256, write_json,
)

SOURCES = [
    "scripts/experiments/hier033_operator_oracle.py", "benchmarks/hier_operator_oracle.py",
    "benchmarks/hier_additive_controls.py", "benchmarks/hier_research_report.py",
    "scripts/check_report_bundle.py", "src/structsplat/pixel_gradient.py",
    "src/structsplat/gaussians.py", "src/structsplat/render.py", "src/structsplat/metrics.py",
]
PROTOCOL = {
    "task": "HIER-033", "version": 1, "device": "cpu", "dtype": "float32", "threads": 1,
    "cpu": "11th Gen Intel(R) Core(TM) i9-11900KF @ 3.50GHz",
    "renderer": "additive", "sigma_cutoff": 3, "support_fade_alpha": 1,
    "families": list(FAMILIES), "seeds": [0, 1, 2], "shape": list(SHAPE), "n_gaussians": 3,
    "seed_role": "three deterministic angle/donor-cost conditions, not independent random samples",
    "actions": action_names(), "magnitudes": list(MAGNITUDES), "donors": list(DONORS),
    "proposal_trust": list(TRUST), "proposal_damping": DAMPING,
    "proposal": "parent-only group GN; symmetric split min-eigenvector with paid donor; residual-peak donor replacement",
    "prediction": "local quadratic continuous; split Hessian plus exact linear donor cost; birth single-pixel proxy minus donor cost",
    "recovery": asdict(ControlConfig(steps=20)),
    "terminal": "exact20 fresh-state Adam updates per action, no best-checkpoint selection",
    "objective": "raw 0.5 mean squared RGB error; full image, no mask",
    "scoring": "raw PSNR with MSEfloor1e-12; display-clamped MS-SSIM/LPIPS",
    "primary": "local packet selection regret versus privileged unrestricted finite oracle before and after recovery",
    "normalized_regret_limit": 0.1, "required_fraction": 0.8, "objective_floor": 1e-8,
    "gate": "at least80% of the SAME conditions pass the regret limit at BOTH phases; all cases valid",
    "cancellation_coherence_max": 0.01, "activity_min": 1e-8, "gain_margin": 1e-8,
    "secondary": "search for high-activity/low-coherence cases where continuous beats every funded split",
    "cold_parity_max_abs": 1e-7, "worker_timeout_seconds": 600,
    "warmup": "translation perturbation77 with two Adam steps, once per case worker",
    "pairing": "family/condition; every candidate receives the same20-step recovery",
    "data_role": "procedural mechanism development only; no natural/held-out/default claim",
    "timing": "descriptive shared CPU; report proposal time and complete per-action recovery time",
    "render_work_accounting": {
        "row_forward_and_gradient_counters": "recovery only",
        "shared_per_case_renders": {"warmup_target": 1, "warmup_fit": 3, "target": 1, "proposal_base": 1},
        "extra_per_cell_renders": {"immediate": 1, "cold": 1},
        "shared_per_case_gradient_work": {"warmup_backwards": 2, "analytic_pixel_packet": 1},
        "scope": "completed cells; failed cells may have additional partial work",
    },
    "missing_policy": "retain successes and failures; any incomplete case excludes whole-atlas positive verdict",
    "forbidden": ["native-method labels", "threshold tuning", "free donor capacity", "selective rerun", "default promotion"],
}


def objective(raw, target):
    return float(0.5 * (raw.double() - target.double()).square().mean())


def worker(request_path):
    import torch
    from structsplat.gaussians import GaussianField
    from structsplat.metrics import LPIPS, ms_ssim

    torch.set_num_threads(PROTOCOL["threads"])
    cpu = next(line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
               if line.startswith("model name"))
    if cpu != PROTOCOL["cpu"]:
        raise RuntimeError("frozen CPU unavailable")
    request = json.loads(Path(request_path).read_text())
    root = Path(request["root"])
    case_dir = Path(request_path).parent
    warm, warm_target = fixture("translation", 77)
    fit_control(warm, warm_target, ControlConfig(steps=2))
    field, target = fixture(request["family"], request["seed"])
    start = time.perf_counter()
    actions, packet, base = finite_actions(field, target)
    proposal_seconds = time.perf_counter() - start
    base_objective = objective(base, target)
    arrays = {name: getattr(packet, name).cpu().numpy() for name in
              ("signed", "absolute", "contribution_square", "gram", "split_matrix", "support_count")}
    np.savez(case_dir / "gradient_packet.npz", **arrays)
    cfg = replace(ControlConfig(**PROTOCOL["recovery"]),
                  steps=2 if request["smoke"] else PROTOCOL["recovery"]["steps"])
    for action in actions:
        cell_id = f'{request["family"]}_s{request["seed"]}_{action.name}'
        directory = root / "cells" / cell_id
        directory.mkdir(parents=True)
        cell_request = {key: request[key] for key in ("family", "seed", "smoke")}
        cell_request.update(cell_id=cell_id, action=action.name)
        write_json(directory / "request.json", cell_request)
        field.save(directory / "base_field.npz")
        action.field.save(directory / "input_field.npz")
        np.save(directory / "target.npy", target.numpy())
        np.save(directory / "base_reconstruction.npy", base.numpy())
        immediate = additive_render(action.field, *SHAPE).detach()
        np.save(directory / "immediate_reconstruction.npy", immediate.numpy())
        immediate_objective = objective(immediate, target)
        write_json(directory / "config.json", {"request": cell_request, "recovery": asdict(cfg),
            "action_family": action.family, "donor": action.donor, "magnitude": action.magnitude,
            "predicted_gain": action.predicted_gain, "torch": torch.__version__, "numpy": np.__version__,
            "cpu": cpu, "platform": platform.platform(), "threads": torch.get_num_threads(),
            "base_field_sha256": sha256(directory / "base_field.npz"),
            "input_field_sha256": sha256(directory / "input_field.npz"),
            "target_sha256": sha256(directory / "target.npy")})
        with (directory / "progress.jsonl").open("w") as progress:
            def callback(row):
                progress.write(json.dumps(row, allow_nan=False, sort_keys=True) + "\n")
                progress.flush()
            result, raw, history, elapsed = fit_control(action.field, target, cfg, callback=callback)
        result.save(directory / "field.npz")
        cold = GaussianField.load(directory / "field.npz")
        cold_error = float((additive_render(cold, *SHAPE) - raw).abs().max())
        if not torch.equal(pack(cold), pack(result)) or result.n != PROTOCOL["n_gaussians"]:
            raise RuntimeError("cold parameters or count changed")
        if cold_error > PROTOCOL["cold_parity_max_abs"]:
            raise RuntimeError("cold reference replay failed")
        np.save(directory / "reconstruction.npy", raw.numpy())
        save_rgb(directory / "target.png", target.numpy())
        save_rgb(directory / "reconstruction.png", raw.numpy())
        save_rgb(directory / "immediate.png", immediate.numpy())
        save_rgb(directory / "error.png", (raw - target).abs().numpy() * 4)
        write_json(directory / "history.json", {"checkpoints": history, "nominal_iterations": cfg.steps})
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(9, 3))
        for axis, key, label in zip(axes, ("iteration", "elapsed_seconds"),
                                   ("Recovery updates", "Recovery seconds (shared CPU)")):
            axis.plot([h[key] for h in history], [h["psnr"] for h in history], label="edited state + recovery")
            axis.axhline(-10 * math.log10(max(2 * base_objective, 1e-12)), color="gray", linestyle="--",
                         label="original unedited state")
            axis.set(xlabel=label, ylabel="Raw PSNR (dB)")
            axis.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(directory / "curves.png", dpi=120)
        plt.close(fig)
        lpips = LPIPS.distance(raw.clamp(0, 1), target)
        if lpips is None:
            raise RuntimeError("required LPIPS unavailable")
        final_objective = objective(raw, target)
        signed_position = float(packet.signed[0, :2].abs().sum())
        absolute_position = float(packet.absolute[0, :2].sum())
        row = {**cell_request, "status": "ok", "action_family": action.family, "donor": action.donor,
               "magnitude": action.magnitude, "predicted_gain": action.predicted_gain,
               "base_objective": base_objective, "immediate_objective": immediate_objective,
               "terminal_objective": final_objective,
               "immediate_gain": base_objective - immediate_objective,
               "recovered_gain": base_objective - final_objective,
               "position_activity": absolute_position,
               "position_coherence": signed_position / max(absolute_position, 1e-30),
               "split_min_eigenvalue": float(torch.linalg.eigvalsh(packet.split_matrix[0])[0]),
               "n_gaussians": result.n, "iterations_run": len(history) - 1,
               "selected_iteration": len(history) - 1,
               "psnr": -10 * math.log10(max(2 * final_objective, 1e-12)),
               "ms_ssim": float(ms_ssim(raw.clamp(0, 1), target)), "lpips": lpips,
               "total_seconds": elapsed, "proposal_seconds": proposal_seconds,
               "forward_evaluations": history[-1]["forward_evaluations"],
               "gradient_evaluations": history[-1]["gradient_evaluations"],
               "counter_scope": "recovery only", "shared_case_render_evaluations": 6,
               "extra_cell_render_evaluations": 2,
               "total_cell_render_evaluations": history[-1]["forward_evaluations"] + 2,
               "cold_render_max_abs": cold_error,
               "artifacts": {name: f"cells/{cell_id}/{name}" for name in (
                   "base_field.npz", "input_field.npz", "field.npz", "target.npy", "base_reconstruction.npy",
                   "immediate_reconstruction.npy", "reconstruction.npy", "target.png", "immediate.png",
                   "reconstruction.png", "error.png", "curves.png", "history.json", "config.json", "progress.jsonl")}}
        row["artifacts"]["gradient_packet"] = f'cases/{request["case_id"]}/gradient_packet.npz'
        write_json(directory / "row.json", row)
        print(json.dumps({"cell_id": cell_id, "status": "ok"}), flush=True)


def evaluate_results(root, rows):
    cases = []
    for family in PROTOCOL["families"]:
        for seed in PROTOCOL["seeds"]:
            group = [r for r in rows if r["family"] == family and r["seed"] == seed]
            complete = (len(group) == len(PROTOCOL["actions"])
                        and {r["action"] for r in group} == set(PROTOCOL["actions"])
                        and all(r["status"] == "ok" for r in group))
            if not complete:
                cases.append({"family": family, "seed": seed, "complete": False})
                continue
            directories = [root / "cells" / r["cell_id"] for r in group]
            same_inputs = all(len({sha256(d / name) for d in directories}) == 1
                              for name in ("base_field.npz", "target.npy"))
            integrity = same_inputs and all(r["n_gaussians"] == PROTOCOL["n_gaussians"]
                and r["iterations_run"] == r["selected_iteration"] == PROTOCOL["recovery"]["steps"]
                and r["cold_render_max_abs"] <= PROTOCOL["cold_parity_max_abs"] for r in group)
            order = {name: i for i, name in enumerate(PROTOCOL["actions"])}
            def best(pool, metric):
                return max(pool, key=lambda r: (r[metric], -order[r["action"]]))
            selected = best(group, "predicted_gain")
            denominator = max(selected["base_objective"], PROTOCOL["objective_floor"])
            record = {"family": family, "seed": seed, "complete": True, "integrity": integrity,
                      "predicted_choice": selected["action"],
                      "position_activity": selected["position_activity"],
                      "position_coherence": selected["position_coherence"]}
            for name, pool in (("unrestricted", group),
                               ("continuous", [r for r in group if r["action_family"] in ("move", "scale", "rotate", "color", "noop")]),
                               ("split", [r for r in group if r["action_family"] in ("split", "noop")])):
                for phase, metric in (("immediate", "immediate_gain"), ("recovered", "recovered_gain")):
                    oracle = best(pool, metric)
                    record[f"{name}_{phase}_oracle"] = oracle["action"]
                    record[f"{name}_{phase}_gain"] = oracle[metric]
            for phase, metric in (("immediate", "immediate_gain"), ("recovered", "recovered_gain")):
                regret = max(0.0, record[f"unrestricted_{phase}_gain"] - selected[metric]) / denominator
                record[f"{phase}_normalized_regret"] = regret
            record["joint_low_regret"] = all(record[f"{phase}_normalized_regret"]
                <= PROTOCOL["normalized_regret_limit"] for phase in ("immediate", "recovered"))
            record["cancellation_counterexample"] = (
                selected["position_activity"] > PROTOCOL["activity_min"]
                and selected["position_coherence"] < PROTOCOL["cancellation_coherence_max"]
                and record["continuous_immediate_gain"] > record["split_immediate_gain"] + PROTOCOL["gain_margin"])
            cases.append(record)
    all_complete = all(c["complete"] and c.get("integrity", False) for c in cases)
    fractions = {phase: sum(c.get(f"{phase}_normalized_regret", float("inf"))
                            <= PROTOCOL["normalized_regret_limit"] for c in cases) / len(cases)
                 for phase in ("immediate", "recovered")}
    joint_fraction = sum(c.get("joint_low_regret", False) for c in cases) / len(cases)
    decision = {"cases": cases, "complete_integrity": all_complete, "low_regret_fractions": fractions,
                "joint_low_regret_fraction": joint_fraction,
                "passes_local_selector_gate": all_complete and joint_fraction >= PROTOCOL["required_fraction"],
                "scope": "finite procedural atlas only", "default_promotion": False, "speed_claim": False}
    write_json(root / "decision.json", decision)
    return decision


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("outdir", nargs="?")
    parser.add_argument("--worker")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--approved-protocol-digest")
    parser.add_argument("--print-protocol-digest", action="store_true")
    args = parser.parse_args()
    if args.worker:
        worker(args.worker)
        return
    digest = protocol_digest(PROTOCOL, SOURCES)
    if args.print_protocol_digest:
        print(digest)
        return
    if not args.outdir or (not args.smoke and args.approved_protocol_digest != digest):
        parser.error("formal execution requires outdir and exact approved digest")
    workloads = [(f, s) for f in PROTOCOL["families"] for s in PROTOCOL["seeds"]]
    if args.smoke:
        workloads = [("translation", 77)]
    identities = [f"{f}_s{s}_{action}" for f, s in workloads for action in PROTOCOL["actions"]]
    bundle = ResearchBundle(args.outdir, task="HIER-033", protocol=PROTOCOL, digest=digest,
                            expected_cells=identities, diagnostic=args.smoke, source_paths=SOURCES)
    rows = []
    for family, seed in workloads:
        case_id = f"{family}_s{seed}"
        case_dir = bundle.root / "cases" / case_id
        case_dir.mkdir(parents=True)
        request = {"family": family, "seed": seed, "case_id": case_id, "smoke": args.smoke,
                   "root": str(bundle.root)}
        path = case_dir / "request.json"
        write_json(path, request)
        error = None
        try:
            with (case_dir / "worker.log").open("w") as log:
                subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker", str(path)],
                               stdout=log, stderr=subprocess.STDOUT, check=True,
                               timeout=PROTOCOL["worker_timeout_seconds"])
        except (subprocess.SubprocessError, OSError) as exc:
            error = str(exc)
        for action in PROTOCOL["actions"]:
            cell_id = f"{case_id}_{action}"
            row_path = bundle.root / "cells" / cell_id / "row.json"
            if row_path.is_file():
                row = json.loads(row_path.read_text())
            else:
                row = {"cell_id": cell_id, "family": family, "seed": seed, "action": action,
                       "smoke": args.smoke, "status": "error", "error": error or "missing worker result",
                       "artifacts": {"worker.log": f"cases/{case_id}/worker.log"}}
            rows.append(row)
        print(json.dumps({"case_id": case_id, "complete_cells": sum(
            r["status"] == "ok" for r in rows if r["family"] == family and r["seed"] == seed)}), flush=True)
    evaluate_results(bundle.root, rows)
    for row in rows:
        row.setdefault("artifacts", {})["decision"] = "decision.json"
    bundle.finish(rows, title="HIER-033 — Count-funded finite-edit oracle",
        interpretation="Finite procedural mechanism atlas, not a native method or natural-image quality result. Every edit retains three Gaussians and pays donor damage. All actions receive identical fresh-state Adam recovery; privileged finite oracles are labelled explicitly. Forward/gradient counters are recovery-only; separate counters enumerate immediate/cold and shared setup renders.")


if __name__ == "__main__":
    main()
