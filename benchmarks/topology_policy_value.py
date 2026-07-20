"""BENCH-012 spatial-connectivity policy-value assay.

This benchmark keeps StructSplat's grammar and finite replacement primitive fixed.  It asks only
whether a target-blind spatial connectivity distance selects a better residual replacement than
the immediate aligned objective after continuous Adam recovery.  See
``tasks/BENCH-012-topology-policy-value.md`` for the frozen protocol and claim boundary.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import resource
import subprocess
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from benchmarks.marginal_cold_stream_rd import (
    ACTION_COUNT,
    _action_bank,
    _field_sha256,
    _replace_field,
)
from benchmarks.topology_policy_value_core import stable_acpd_select, target_suite
from structsplat import metrics as M
from structsplat.config import FitConfig
from structsplat.fit import _carry_adam_state, _make_optimizer, _pixel_loss, _render
from structsplat.gaussians import GaussianField


PROTOCOL = "bench012-spatial-connectivity-policy-value-v1"
SIZE = 64
COUNT = 64
SEEDS = (0, 1)
PARENT_STEPS = 600
ROLLOUT_STEP = 20
FINAL_STEP = 100
LOG_EVERY = 5
BOOTSTRAP_REPLICATES = 20_000
BOOTSTRAP_SEED = 20260716

SOURCE_PATHS = (
    "benchmarks/topology_policy_value.py",
    "benchmarks/topology_policy_value_core.py",
    "benchmarks/marginal_cold_stream_rd.py",
    "src/structsplat/config.py",
    "src/structsplat/fit.py",
    "src/structsplat/gaussians.py",
    "src/structsplat/metrics.py",
    "src/structsplat/render.py",
)


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_sha256(value: torch.Tensor) -> str:
    array = value.detach().cpu().contiguous().numpy()
    payload = {
        "dtype": str(array.dtype),
        "shape": list(array.shape),
        "bytes": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
    }
    return _sha256_bytes(_canonical_json(payload))


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(payload, encoding="utf-8")
    os.replace(tmp, path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"invalid JSONL at {path}:{line_number}") from exc
    return rows


def _append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _canonical_json(dict(row)) + b"\n"
    with path.open("ab", buffering=0) as handle:
        handle.write(payload)
        os.fsync(handle.fileno())


def _git_metadata(root: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        result = subprocess.run(
            ["git", *args], cwd=root, check=True, capture_output=True, text=True
        )
        return result.stdout.strip()

    try:
        commit = run("rev-parse", "HEAD")
        branch = run("branch", "--show-current")
        status = run("status", "--porcelain=v1", "--untracked-files=all")
        diff = subprocess.run(
            ["git", "diff", "--binary", "--no-ext-diff"],
            cwd=root,
            check=True,
            capture_output=True,
        ).stdout
        return {
            "commit": commit,
            "branch": branch,
            "dirty": bool(status),
            "status_sha256": _sha256_bytes(status.encode("utf-8")),
            "tracked_diff_sha256": _sha256_bytes(diff),
        }
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "branch": None, "dirty": None}


def _environment(root: Path) -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "device": "cpu",
        "torch_num_threads": torch.get_num_threads(),
        "git": _git_metadata(root),
    }


def _source_manifest(root: Path) -> dict[str, str]:
    manifest: dict[str, str] = {}
    for relative in SOURCE_PATHS:
        path = root / relative
        if not path.is_file():
            raise RuntimeError(f"missing BENCH-012 source dependency: {relative}")
        manifest[relative] = _sha256_file(path)
    return manifest


def _science_config(root: Path) -> dict[str, Any]:
    suite = target_suite()
    target_manifest = {
        name: {
            "family": item["family"],
            "variant": int(item["variant"]),
            "expected_betti": list(item["expected_betti"]),
            "image_sha256": _sha256_bytes(
                np.asarray(item["image"], dtype=np.float32).tobytes(order="C")
            ),
        }
        for name, item in sorted(suite.items())
    }
    return {
        "protocol": PROTOCOL,
        "size": SIZE,
        "count": COUNT,
        "seeds": list(SEEDS),
        "parent_steps": PARENT_STEPS,
        "rollout_step": ROLLOUT_STEP,
        "final_step": FINAL_STEP,
        "log_every": LOG_EVERY,
        "action_count": ACTION_COUNT,
        "objective": "0.7*rgb_l1 + 0.3*(1-builtin_ssim)",
        "renderer": "normalized_cpu_reference",
        "topology": {
            "thresholds": [0.4, 0.5, 0.6],
            "foreground_rule": "mean(rgb) >= threshold",
            "dual_conventions": ["foreground8_background4", "foreground4_background8"],
            "anchors": "{1,5,...,61}^2",
            "fallback": "immediate_aligned_objective",
        },
        "gate": {
            "changed_targets_min": 12,
            "changed_families_min": 4,
            "mean_psnr_gain_min_db": 0.15,
            "disagreement_wins_min": 10,
            "disagreement_win_fraction_min": 0.80,
            "family_floor_db": -0.10,
            "rollout20_floor_db": -0.05,
        },
        "target_manifest": target_manifest,
        "source_manifest": _source_manifest(root),
    }


def _binding(root: Path) -> tuple[dict[str, Any], str]:
    science = _science_config(root)
    return science, _sha256_bytes(_canonical_json(science))


def _fit_config() -> FitConfig:
    return FitConfig(
        iters=PARENT_STEPS,
        renderer="normalized",
        pixel_loss="l1",
        ssim_weight=0.3,
        ssim_backend="builtin",
        lr_schedule="none",
        log_every=LOG_EVERY,
        checkpoint_policy="terminal",
        support_fade=False,
        aa_dilation=0.0,
        covariance_filter_mode="none",
        color_basis="constant",
        color_solve_schedule="none",
        qat_mode="off",
        prune_every=None,
        split_every=None,
        split_count=0,
        relocate_every=None,
        relocate_count=0,
    )


def initial_field(seed: int, *, dtype: torch.dtype = torch.float32) -> GaussianField:
    """Target-independent jittered 8x8 grid used by every BENCH-012 target."""
    if seed not in SEEDS:
        raise ValueError(f"seed must be one of {SEEDS}, got {seed}")
    rng = np.random.default_rng(12012 + int(seed))
    axis = np.linspace(4.0, 59.0, 8, dtype=np.float32)
    means = np.asarray([(x, y) for y in axis for x in axis], dtype=np.float32)
    means += rng.uniform(-0.35, 0.35, size=means.shape).astype(np.float32)
    scales = np.full((COUNT, 2), 7.0, dtype=np.float32)
    scales *= np.exp(rng.uniform(-0.025, 0.025, size=scales.shape)).astype(np.float32)
    angles = rng.uniform(-0.04, 0.04, size=COUNT).astype(np.float32)
    colors = np.full((COUNT, 3), 0.5, dtype=np.float32)
    colors += rng.uniform(-0.01, 0.01, size=colors.shape).astype(np.float32)
    return GaussianField.from_numpy(
        means, scales, angles, colors, device="cpu", dtype=dtype
    )


def _objective(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pixel = _pixel_loss(prediction, target, "l1")
    return 0.7 * pixel + 0.3 * (1.0 - M.ssim(prediction, target, backend="builtin"))


@torch.no_grad()
def _evaluate(field: GaussianField, target: torch.Tensor, cfg: FitConfig) -> dict[str, Any]:
    render = _render(field, cfg, SIZE, SIZE)
    objective = _objective(render, target)
    mse = M.mse(render, target).clamp_min(1e-12)
    return {
        "objective": float(objective.detach().cpu()),
        "mse": float(mse.detach().cpu()),
        "psnr_db": float(M.psnr_from_mse(mse).detach().cpu()),
        "ssim": float(M.ssim(render, target, backend="builtin").detach().cpu()),
        "render_sha256": _tensor_sha256(render),
        "render": render.detach().clone(),
    }


def _public_metrics(record: Mapping[str, Any], step: int) -> dict[str, Any]:
    return {
        "step": int(step),
        "objective": float(record["objective"]),
        "mse": float(record["mse"]),
        "psnr_db": float(record["psnr_db"]),
        "ssim": float(record["ssim"]),
        "render_sha256": str(record["render_sha256"]),
    }


def _advance(
    field: GaussianField,
    optimizer: torch.optim.Optimizer,
    target: torch.Tensor,
    cfg: FitConfig,
    *,
    start_step: int,
    end_step: int,
    history: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    if end_step < start_step:
        raise ValueError("end_step must be >= start_step")
    records = [] if history is None else history
    if not records:
        records.append(_public_metrics(_evaluate(field, target, cfg), start_step))
    lo, hi = math.log(0.35), math.log(SIZE)
    for step in range(start_step + 1, end_step + 1):
        prediction = _render(field, cfg, SIZE, SIZE)
        loss = _objective(prediction, target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            field.log_scales.clamp_(lo, hi)
        if step % LOG_EVERY == 0 or step in (ROLLOUT_STEP, FINAL_STEP):
            records.append(_public_metrics(_evaluate(field, target, cfg), step))
    return records


def _optimizer_sha256(optimizer: torch.optim.Optimizer) -> str:
    groups = []
    for group_index, group in enumerate(optimizer.param_groups):
        if len(group["params"]) != 1:
            raise RuntimeError("BENCH-012 expects one tensor per optimizer group")
        parameter = group["params"][0]
        state = optimizer.state.get(parameter, {})
        state_payload = {}
        for key, value in sorted(state.items()):
            if torch.is_tensor(value):
                state_payload[key] = {
                    "tensor_sha256": _tensor_sha256(value),
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                }
            else:
                state_payload[key] = value
        groups.append(
            {
                "group": group_index,
                "lr": float(group["lr"]),
                "parameter_sha256": _tensor_sha256(parameter),
                "state": state_payload,
            }
        )
    return _sha256_bytes(_canonical_json(groups))


def _clone_no_action(
    parent: GaussianField, optimizer: torch.optim.Optimizer, cfg: FitConfig
) -> tuple[GaussianField, torch.optim.Optimizer]:
    child = parent.detached().trainable()
    child_optimizer = _carry_adam_state(optimizer, child, cfg, keep=None, n_new=0)
    return child, child_optimizer


def _replacement_with_state(
    parent: GaussianField,
    optimizer: torch.optim.Optimizer,
    candidate: GaussianField,
    donor_index: int,
    cfg: FitConfig,
) -> tuple[GaussianField, torch.optim.Optimizer, dict[str, Any]]:
    keep = torch.ones(parent.n, dtype=torch.bool, device=parent.means.device)
    keep[int(donor_index)] = False
    child = _replace_field(parent, candidate, donor_index).trainable()
    child_optimizer = _carry_adam_state(optimizer, child, cfg, keep=keep, n_new=1)

    continuation = {"survivor_state_exact": True, "new_row_zero_moments": True}
    for old_group, new_group in zip(optimizer.param_groups, child_optimizer.param_groups):
        old_param = old_group["params"][0]
        new_param = new_group["params"][0]
        old_state = optimizer.state.get(old_param, {})
        new_state = child_optimizer.state.get(new_param, {})
        if set(old_state) != set(new_state):
            raise RuntimeError("optimizer state keys changed across replacement")
        for key, old_value in old_state.items():
            new_value = new_state[key]
            if torch.is_tensor(old_value) and old_value.shape == old_param.shape:
                expected = old_value.detach()[keep]
                if not torch.equal(new_value[:-1], expected):
                    continuation["survivor_state_exact"] = False
                if not torch.count_nonzero(new_value[-1]).eq(0):
                    continuation["new_row_zero_moments"] = False
            elif torch.is_tensor(old_value):
                if not torch.equal(new_value, old_value):
                    raise RuntimeError("global Adam tensor state changed across replacement")
            elif new_value != old_value:
                raise RuntimeError("global Adam scalar state changed across replacement")
    if not all(continuation.values()):
        raise RuntimeError(f"optimizer continuation invariant failed: {continuation}")
    continuation["optimizer_sha256"] = _optimizer_sha256(child_optimizer)
    return child, child_optimizer, continuation


def _candidate_work(candidate: GaussianField, cfg: FitConfig) -> dict[str, Any]:
    if candidate.n != 1:
        raise ValueError("candidate work requires one row")
    radius = candidate.radii(cfg.sigma_cutoff, cfg.aa_dilation)[0].detach().cpu().long()
    mean = candidate.means[0].detach().cpu()
    ix, iy = int(torch.round(mean[0])), int(torch.round(mean[1]))
    rx, ry = int(radius[0]), int(radius[1])
    x0, x1, y0, y1 = ix - rx, ix + rx, iy - ry, iy + ry
    untruncated = x0 >= 0 and y0 >= 0 and x1 < SIZE and y1 < SIZE
    clipped_x0, clipped_x1 = max(0, x0), min(SIZE - 1, x1)
    clipped_y0, clipped_y1 = max(0, y0), min(SIZE - 1, y1)
    count = max(0, clipped_x1 - clipped_x0 + 1) * max(0, clipped_y1 - clipped_y0 + 1)
    return {
        "radius": [rx, ry],
        "rounded_center": [ix, iy],
        "untruncated": bool(untruncated),
        "support_work_count": int(count),
    }


def _feasible_candidates(
    candidates: Sequence[GaussianField], cfg: FitConfig
) -> tuple[list[int], list[dict[str, Any]]]:
    if len(candidates) != ACTION_COUNT:
        raise RuntimeError(f"expected {ACTION_COUNT} candidates, got {len(candidates)}")
    scales = [candidate.log_scales.detach().cpu() for candidate in candidates]
    rotations = [candidate.rotations.detach().cpu() for candidate in candidates]
    if not all(torch.equal(value, scales[0]) for value in scales[1:]):
        raise RuntimeError("COMP-006 candidates no longer share one frozen scale primitive")
    if not all(torch.equal(value, rotations[0]) for value in rotations[1:]):
        raise RuntimeError("COMP-006 candidates no longer share one frozen rotation primitive")
    work = [_candidate_work(candidate, cfg) for candidate in candidates]
    untruncated_counts = [
        item["support_work_count"] for item in work if item["untruncated"]
    ]
    if not untruncated_counts:
        return [], work
    modal_count = Counter(untruncated_counts).most_common(1)[0][0]
    feasible = [
        index
        for index, item in enumerate(work)
        if item["untruncated"] and item["support_work_count"] == modal_count
    ]
    for index, item in enumerate(work):
        item["modal_work_count"] = int(modal_count)
        item["feasible"] = index in feasible
    return feasible, work


def _history_at(history: Sequence[Mapping[str, Any]], step: int) -> dict[str, Any]:
    matches = [dict(row) for row in history if int(row["step"]) == int(step)]
    if len(matches) != 1:
        raise RuntimeError(f"trajectory has {len(matches)} rows at step {step}")
    return matches[0]


def _select_min(
    values: Mapping[int, float], feasible: Sequence[int]
) -> int:
    if not feasible:
        raise ValueError("cannot select from an empty feasible set")
    return min((float(values[index]), int(index)) for index in feasible)[1]


def _strip_render(record: dict[str, Any]) -> tuple[dict[str, Any], torch.Tensor]:
    render = record.pop("render")
    return record, render


def run_cell(
    *,
    target_name: str,
    target_item: Mapping[str, Any],
    seed: int,
    binding_sha256: str,
) -> dict[str, Any]:
    started = time.perf_counter()
    cfg = _fit_config()
    target_array = np.asarray(target_item["image"], dtype=np.float32)
    if target_array.shape == (SIZE, SIZE):
        target_array = np.repeat(target_array[..., None], 3, axis=2)
    if target_array.shape != (SIZE, SIZE, 3):
        raise RuntimeError(f"unexpected target shape {target_array.shape}")
    target = torch.as_tensor(target_array, dtype=torch.float32, device="cpu")

    parent = initial_field(seed).trainable()
    optimizer = _make_optimizer(parent, cfg)
    parent_history = _advance(
        parent,
        optimizer,
        target,
        cfg,
        start_step=0,
        end_step=PARENT_STEPS,
    )
    parent_field_sha256 = _field_sha256(parent)
    parent_optimizer_sha256 = _optimizer_sha256(optimizer)
    parent_terminal = _history_at(parent_history, PARENT_STEPS)

    candidates, donor_index, action_metadata = _action_bank(parent, target)
    if _field_sha256(parent) != parent_field_sha256:
        raise RuntimeError("action-bank construction mutated the fitted parent")
    feasible, work = _feasible_candidates(candidates, cfg)
    if len(feasible) < 4:
        raise RuntimeError(
            f"{target_name}/seed{seed} has only {len(feasible)} work-matched candidates"
        )

    step0: dict[int, dict[str, Any]] = {}
    step0_renders: dict[int, np.ndarray] = {}
    branch_fields: dict[int, GaussianField] = {}
    branch_optimizers: dict[int, torch.optim.Optimizer] = {}
    trajectories: dict[int, list[dict[str, Any]]] = {}
    continuation: dict[int, dict[str, Any]] = {}
    for action_index in feasible:
        field, branch_optimizer, state = _replacement_with_state(
            parent, optimizer, candidates[action_index], donor_index, cfg
        )
        evaluated = _evaluate(field, target, cfg)
        render = evaluated.pop("render")
        step0[action_index] = _public_metrics(evaluated, 0)
        step0_renders[action_index] = render.detach().cpu().numpy()
        branch_fields[action_index] = field
        branch_optimizers[action_index] = branch_optimizer
        trajectories[action_index] = [dict(step0[action_index])]
        continuation[action_index] = state

    immediate = _select_min(
        {index: step0[index]["objective"] for index in feasible}, feasible
    )
    topology_payload = stable_acpd_select(target_array, step0_renders)
    topology_selected = topology_payload.get("selected_index")
    if topology_selected is not None:
        topology_selected = int(topology_selected)
    topology_fallback = topology_selected not in feasible
    topology = immediate if topology_fallback else int(topology_selected)
    residual = min(feasible)

    for action_index in feasible:
        trajectories[action_index] = _advance(
            branch_fields[action_index],
            branch_optimizers[action_index],
            target,
            cfg,
            start_step=0,
            end_step=ROLLOUT_STEP,
            history=trajectories[action_index],
        )
    rollout20 = _select_min(
        {
            index: _history_at(trajectories[index], ROLLOUT_STEP)["objective"]
            for index in feasible
        },
        feasible,
    )
    selected_actions = sorted({residual, immediate, topology, rollout20})
    for action_index in selected_actions:
        trajectories[action_index] = _advance(
            branch_fields[action_index],
            branch_optimizers[action_index],
            target,
            cfg,
            start_step=ROLLOUT_STEP,
            end_step=FINAL_STEP,
            history=trajectories[action_index],
        )

    no_action_field, no_action_optimizer = _clone_no_action(parent, optimizer, cfg)
    no_action_history = _advance(
        no_action_field,
        no_action_optimizer,
        target,
        cfg,
        start_step=0,
        end_step=FINAL_STEP,
    )

    if _field_sha256(parent) != parent_field_sha256:
        raise RuntimeError("action/recovery branches mutated the fitted parent")
    if _optimizer_sha256(optimizer) != parent_optimizer_sha256:
        raise RuntimeError("action/recovery branches mutated the parent optimizer")

    policy_actions = {
        "residual": residual,
        "immediate": immediate,
        "topology": topology,
        "rollout20": rollout20,
    }
    row = {
        "schema": "bench012-cell-v1",
        "protocol": PROTOCOL,
        "binding_sha256": binding_sha256,
        "cell_id": f"{target_name}:seed{seed}",
        "target_name": target_name,
        "family": str(target_item["family"]),
        "variant": int(target_item["variant"]),
        "seed": int(seed),
        "target_sha256": _tensor_sha256(target),
        "expected_betti": list(target_item["expected_betti"]),
        "parent": {
            "field_sha256": parent_field_sha256,
            "optimizer_sha256": parent_optimizer_sha256,
            "terminal": parent_terminal,
        },
        "action_bank": action_metadata,
        "donor_index": int(donor_index),
        "candidate_work": work,
        "feasible_actions": feasible,
        "continuation": {str(index): continuation[index] for index in feasible},
        "step0": {str(index): step0[index] for index in feasible},
        "topology_selection": topology_payload,
        "topology_fallback": bool(topology_fallback),
        "policies": policy_actions,
        "selected_action_trajectories": {
            str(index): trajectories[index] for index in selected_actions
        },
        "all_action_step20": {
            str(index): _history_at(trajectories[index], ROLLOUT_STEP)
            for index in feasible
        },
        "no_action_trajectory": no_action_history,
        "integrity": {
            "parent_field_immutable": True,
            "parent_optimizer_immutable": True,
            "all_survivor_states_exact": all(
                item["survivor_state_exact"] for item in continuation.values()
            ),
            "all_new_rows_zero_moments": all(
                item["new_row_zero_moments"] for item in continuation.values()
            ),
            "work_matched_candidates": len(feasible),
        },
        "wall_seconds": time.perf_counter() - started,
        "peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
    }
    row["scientific_sha256"] = _sha256_bytes(_canonical_json(_scientific_cell(row)))
    return row


def _scientific_cell(row: Mapping[str, Any]) -> dict[str, Any]:
    drop = {"wall_seconds", "peak_rss_kib", "scientific_sha256"}
    return {key: value for key, value in row.items() if key not in drop}


def prepare_run(outdir: Path, root: Path) -> tuple[dict[str, Any], str]:
    science, binding_sha256 = _binding(root)
    config_path = outdir / "config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if config.get("binding_sha256") != binding_sha256:
            raise RuntimeError("BENCH-012 resume binding differs from existing output")
    else:
        config = {
            "schema": "bench012-config-v1",
            "binding_sha256": binding_sha256,
            "science": science,
            "environment": _environment(root),
            "created_at_unix": time.time(),
        }
        _atomic_json(config_path, config)
    return config, binding_sha256


def run(
    outdir: Path,
    *,
    root: Path,
    shard_index: int = 0,
    num_shards: int = 1,
    max_cells: int | None = None,
) -> None:
    if num_shards <= 0 or not 0 <= shard_index < num_shards:
        raise ValueError("invalid shard index/count")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    _config, binding_sha256 = prepare_run(outdir, root)
    rows = _read_jsonl(outdir / "cells.jsonl")
    by_id = {row["cell_id"]: row for row in rows}
    if len(by_id) != len(rows):
        raise RuntimeError("duplicate BENCH-012 cell IDs")
    for row in rows:
        if row.get("binding_sha256") != binding_sha256:
            raise RuntimeError("existing BENCH-012 cell has stale binding")
        if row.get("scientific_sha256") != _sha256_bytes(
            _canonical_json(_scientific_cell(row))
        ):
            raise RuntimeError(f"scientific hash mismatch for {row.get('cell_id')}")

    suite = target_suite()
    jobs = [
        (name, item, seed)
        for name, item in sorted(suite.items())
        for seed in SEEDS
    ]
    jobs = [job for index, job in enumerate(jobs) if index % num_shards == shard_index]
    completed = 0
    for target_name, item, seed in jobs:
        cell_id = f"{target_name}:seed{seed}"
        if cell_id in by_id:
            continue
        if max_cells is not None and completed >= max_cells:
            break
        try:
            row = run_cell(
                target_name=target_name,
                target_item=item,
                seed=seed,
                binding_sha256=binding_sha256,
            )
        except Exception as exc:
            error = {
                "schema": "bench012-error-v1",
                "protocol": PROTOCOL,
                "binding_sha256": binding_sha256,
                "cell_id": cell_id,
                "target_name": target_name,
                "family": str(item["family"]),
                "variant": int(item["variant"]),
                "seed": int(seed),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "scientific_outcomes_scored": False,
                "disposition": "unavailable_close_without_retuning",
            }
            _append_jsonl(outdir / "errors.jsonl", error)
            raise
        _append_jsonl(outdir / "cells.jsonl", row)
        by_id[cell_id] = row
        completed += 1
        print(
            f"BENCH-012 shard={shard_index}/{num_shards} {cell_id} "
            f"{row['wall_seconds']:.2f}s feasible={len(row['feasible_actions'])} "
            f"policies={row['policies']}",
            flush=True,
        )
    marker = {
        "binding_sha256": binding_sha256,
        "shard_index": shard_index,
        "num_shards": num_shards,
        "expected_cells": len(jobs),
        "present_cells": sum(f"{name}:seed{seed}" in by_id for name, _, seed in jobs),
    }
    _atomic_json(outdir / "shards" / f"{shard_index:03d}-of-{num_shards:03d}.json", marker)


def _trajectory_for_policy(cell: Mapping[str, Any], policy: str) -> list[dict[str, Any]]:
    if policy == "no_action":
        return [dict(row) for row in cell["no_action_trajectory"]]
    action_index = int(cell["policies"][policy])
    return [dict(row) for row in cell["selected_action_trajectories"][str(action_index)]]


def _auc(history: Sequence[Mapping[str, Any]], metric: str) -> float:
    rows = sorted(history, key=lambda row: int(row["step"]))
    steps = np.asarray([float(row["step"]) for row in rows], dtype=np.float64)
    values = np.asarray([float(row[metric]) for row in rows], dtype=np.float64)
    if steps[0] != 0 or steps[-1] != FINAL_STEP or np.any(np.diff(steps) <= 0):
        raise RuntimeError("invalid trajectory axis for AUC")
    return float(np.trapezoid(values, steps) / float(FINAL_STEP))


def _bootstrap_ci(
    target_rows: Sequence[Mapping[str, Any]], key: str
) -> tuple[float, float]:
    by_family: dict[str, list[float]] = defaultdict(list)
    for row in target_rows:
        by_family[str(row["family"])].append(float(row[key]))
    families = sorted(by_family)
    if len(families) != 6 or any(len(by_family[family]) != 4 for family in families):
        raise RuntimeError("family bootstrap requires exactly 6 families x 4 targets")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    for index in range(BOOTSTRAP_REPLICATES):
        selected = rng.integers(0, len(families), size=len(families))
        values: list[float] = []
        for family_index in selected:
            values.extend(by_family[families[int(family_index)]])
        samples[index] = np.mean(values)
    low, high = np.quantile(samples, [0.025, 0.975])
    return float(low), float(high)


def _sign_test_one_sided(wins: int, total: int) -> float:
    if not 0 <= wins <= total:
        raise ValueError("invalid sign-test counts")
    return float(sum(math.comb(total, k) for k in range(wins, total + 1)) / (2**total))


def _validate_complete(
    rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> None:
    expected = {
        f"{name}:seed{seed}"
        for name in config["science"]["target_manifest"]
        for seed in SEEDS
    }
    ids = [str(row["cell_id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise RuntimeError("duplicate BENCH-012 cells")
    if set(ids) != expected:
        missing = sorted(expected - set(ids))
        extra = sorted(set(ids) - expected)
        raise RuntimeError(f"incomplete BENCH-012 cells missing={missing} extra={extra}")
    for row in rows:
        if row["binding_sha256"] != config["binding_sha256"]:
            raise RuntimeError("BENCH-012 binding mismatch")
        if not all(bool(value) for key, value in row["integrity"].items() if key != "work_matched_candidates"):
            raise RuntimeError(f"integrity failure in {row['cell_id']}")
        if int(row["integrity"]["work_matched_candidates"]) < 4:
            raise RuntimeError(f"insufficient feasible actions in {row['cell_id']}")
        for policy in ("residual", "immediate", "topology", "rollout20"):
            _history_at(_trajectory_for_policy(row, policy), FINAL_STEP)
        _history_at(_trajectory_for_policy(row, "no_action"), FINAL_STEP)


def analyze(outdir: Path) -> dict[str, Any]:
    config = json.loads((outdir / "config.json").read_text(encoding="utf-8"))
    rows = _read_jsonl(outdir / "cells.jsonl")
    _validate_complete(rows, config)
    by_target: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_target[str(row["target_name"])].append(row)
    if len(by_target) != 24 or any(len(cells) != 2 for cells in by_target.values()):
        raise RuntimeError("BENCH-012 inference requires 24 targets x 2 seeds")

    target_effects: dict[str, list[dict[str, Any]]] = defaultdict(list)
    target_summary: list[dict[str, Any]] = []
    controls = ("immediate", "residual", "no_action", "rollout20")
    for target_name, cells in sorted(by_target.items()):
        cells = sorted(cells, key=lambda row: int(row["seed"]))
        changed_both = all(
            not bool(cell["topology_fallback"])
            and int(cell["policies"]["topology"]) != int(cell["policies"]["immediate"])
            for cell in cells
        )
        summary = {
            "target_name": target_name,
            "family": cells[0]["family"],
            "variant": int(cells[0]["variant"]),
            "topology_changed_both_seeds": changed_both,
        }
        for control in controls:
            seed_effects = []
            for cell in cells:
                topology_history = _trajectory_for_policy(cell, "topology")
                control_history = _trajectory_for_policy(cell, control)
                topology20 = _history_at(topology_history, ROLLOUT_STEP)
                control20 = _history_at(control_history, ROLLOUT_STEP)
                topology100 = _history_at(topology_history, FINAL_STEP)
                control100 = _history_at(control_history, FINAL_STEP)
                seed_effects.append(
                    {
                        "psnr_gain_step20": float(topology20["psnr_db"])
                        - float(control20["psnr_db"]),
                        "loss_gain_step20": float(control20["objective"])
                        - float(topology20["objective"]),
                        "psnr_gain_step100": float(topology100["psnr_db"])
                        - float(control100["psnr_db"]),
                        "loss_gain_step100": float(control100["objective"])
                        - float(topology100["objective"]),
                        "psnr_auc_gain": _auc(topology_history, "psnr_db")
                        - _auc(control_history, "psnr_db"),
                    }
                )
            effect = {
                "target_name": target_name,
                "family": cells[0]["family"],
                "variant": int(cells[0]["variant"]),
                "changed": changed_both,
            }
            for key in seed_effects[0]:
                effect[key] = float(np.mean([item[key] for item in seed_effects]))
            target_effects[control].append(effect)
            for key, value in effect.items():
                if key not in {"target_name", "family", "variant", "changed"}:
                    summary[f"topology_vs_{control}_{key}"] = value
        target_summary.append(summary)

    comparisons: dict[str, Any] = {}
    for control, effects in target_effects.items():
        comparison: dict[str, Any] = {}
        for key in (
            "psnr_gain_step20",
            "loss_gain_step20",
            "psnr_gain_step100",
            "loss_gain_step100",
            "psnr_auc_gain",
        ):
            values = [float(row[key]) for row in effects]
            low, high = _bootstrap_ci(effects, key)
            comparison[key] = {
                "mean": float(np.mean(values)),
                "median": float(np.median(values)),
                "ci95": [low, high],
            }
        family_means = {
            family: float(
                np.mean(
                    [row["psnr_gain_step100"] for row in effects if row["family"] == family]
                )
            )
            for family in sorted({str(row["family"]) for row in effects})
        }
        comparison["family_mean_psnr_gain_step100"] = family_means
        comparisons[control] = comparison

    changed = [row for row in target_effects["immediate"] if bool(row["changed"])]
    changed_families = sorted({str(row["family"]) for row in changed})
    wins = sum(float(row["psnr_gain_step100"]) > 0.0 for row in changed)
    sign_p = _sign_test_one_sided(wins, len(changed)) if changed else 1.0

    gate_items = {
        "changed_targets": len(changed) >= 12,
        "changed_families": len(changed_families) >= 4,
        "material_psnr": all(
            comparisons[control]["psnr_gain_step100"]["mean"] >= 0.15
            for control in ("immediate", "residual", "no_action")
        ),
        "positive_psnr_ci": all(
            comparisons[control]["psnr_gain_step100"]["ci95"][0] > 0.0
            for control in ("immediate", "residual", "no_action")
        ),
        "positive_loss_ci": all(
            comparisons[control]["loss_gain_step100"]["ci95"][0] > 0.0
            for control in ("immediate", "residual", "no_action")
        ),
        "disagreement_wins": wins >= 10
        and bool(changed)
        and wins / len(changed) >= 0.80,
        "step20_no_reversal": comparisons["immediate"]["psnr_gain_step20"]["mean"]
        >= 0.0
        and comparisons["immediate"]["loss_gain_step20"]["mean"] >= 0.0,
        "positive_auc_ci": all(
            comparisons[control]["psnr_auc_gain"]["ci95"][0] > 0.0
            for control in ("immediate", "no_action")
        ),
        "family_floor": min(
            comparisons["immediate"]["family_mean_psnr_gain_step100"].values()
        )
        >= -0.10,
        "rollout20_guard": comparisons["rollout20"]["psnr_gain_step100"]["mean"]
        >= -0.05,
        "integrity": True,
    }
    decision = "pass" if all(gate_items.values()) else "fail_close_without_retuning"
    analysis = {
        "schema": "bench012-analysis-v1",
        "protocol": PROTOCOL,
        "binding_sha256": config["binding_sha256"],
        "targets": len(target_summary),
        "cells": len(rows),
        "changed_targets_both_seeds": len(changed),
        "changed_families": changed_families,
        "disagreement_psnr_wins_step100": wins,
        "disagreement_sign_test_one_sided_p": sign_p,
        "comparisons": comparisons,
        "gate_items": gate_items,
        "decision": decision,
        "claim_boundary": {
            "allocation_policy": decision == "pass",
            "quality": False,
            "convergence": False,
            "performance": False,
            "compression": False,
            "expressiveness": False,
            "production_change": False,
        },
    }
    _atomic_json(outdir / "target_effects.json", target_summary)
    _atomic_json(outdir / "analysis.json", analysis)
    return analysis


def compare_replay(primary: Path, replay: Path) -> dict[str, Any]:
    primary_config = json.loads((primary / "config.json").read_text(encoding="utf-8"))
    replay_config = json.loads((replay / "config.json").read_text(encoding="utf-8"))
    if primary_config["binding_sha256"] != replay_config["binding_sha256"]:
        raise RuntimeError("BENCH-012 replay bindings differ")
    primary_rows = {row["cell_id"]: row for row in _read_jsonl(primary / "cells.jsonl")}
    replay_rows = {row["cell_id"]: row for row in _read_jsonl(replay / "cells.jsonl")}
    exact = primary_rows.keys() == replay_rows.keys()
    mismatches = []
    for cell_id in sorted(primary_rows.keys() & replay_rows.keys()):
        if _scientific_cell(primary_rows[cell_id]) != _scientific_cell(replay_rows[cell_id]):
            exact = False
            mismatches.append(cell_id)
    result = {
        "schema": "bench012-replay-v1",
        "binding_sha256": primary_config["binding_sha256"],
        "primary_cells": len(primary_rows),
        "replay_cells": len(replay_rows),
        "scientific_exact": exact,
        "mismatched_cells": mismatches,
    }
    _atomic_json(primary / "replay_audit.json", result)
    return result


def controls(root: Path) -> dict[str, Any]:
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    suite = target_suite()
    if len(suite) != 24:
        raise RuntimeError(f"target suite has {len(suite)} targets, expected 24")
    a = initial_field(0)
    b = initial_field(0)
    c = initial_field(1)
    if _field_sha256(a) != _field_sha256(b):
        raise RuntimeError("target-independent seed initialization is not deterministic")
    if _field_sha256(a) == _field_sha256(c):
        raise RuntimeError("BENCH-012 repeated seeds are identical")
    science, binding_sha256 = _binding(root)
    result = {
        "protocol": PROTOCOL,
        "binding_sha256": binding_sha256,
        "targets": len(suite),
        "families": sorted({item["family"] for item in suite.values()}),
        "source_files": len(science["source_manifest"]),
        "initialization_deterministic": True,
        "seed_repeated_measurements_distinct": True,
    }
    return result


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("controls")
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--outdir", type=Path, required=True)
    run_parser.add_argument("--shard-index", type=int, default=0)
    run_parser.add_argument("--num-shards", type=int, default=1)
    run_parser.add_argument("--max-cells", type=int, default=None)
    analyze_parser = subparsers.add_parser("analyze")
    analyze_parser.add_argument("--outdir", type=Path, required=True)
    replay_parser = subparsers.add_parser("compare-replay")
    replay_parser.add_argument("--primary", type=Path, required=True)
    replay_parser.add_argument("--replay", type=Path, required=True)
    args = parser.parse_args(argv)
    root = _root()
    if args.command == "controls":
        print(json.dumps(controls(root), indent=2, sort_keys=True))
        return 0
    if args.command == "run":
        run(
            args.outdir,
            root=root,
            shard_index=args.shard_index,
            num_shards=args.num_shards,
            max_cells=args.max_cells,
        )
        return 0
    if args.command == "analyze":
        result = analyze(args.outdir)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["decision"] == "pass" else 2
    if args.command == "compare-replay":
        result = compare_replay(args.primary, args.replay)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["scientific_exact"] else 2
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
