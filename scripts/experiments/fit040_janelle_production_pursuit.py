#!/usr/bin/env python3
"""Replay FIT-040's production pursuit tail on the persisted Janelle current state."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import sys

import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
for _root in (REPOSITORY_ROOT, REPOSITORY_ROOT / "src"):
    _text = str(_root)
    if _text not in sys.path:
        sys.path.insert(0, _text)

from scripts.experiments.fit032_janelle_dipole_screen import (  # noqa: E402
    _base_config,
    _prepare_current_job,
    _scaled_field,
)
from structsplat.gaussians import GaussianField  # noqa: E402
from structsplat.safe_schedule import (  # noqa: E402
    PhaseBudget,
    SafeScheduleConfig,
    run_safe_schedule,
)


DEFAULT_BASE_JOB = (
    REPOSITORY_ROOT
    / "runs/fit032_current_base_20260728/runs/current/C0001/seed_0"
)
DEFAULT_REFERENCE_FIELD = (
    REPOSITORY_ROOT
    / "runs/fit039_janelle_exclusion_screen_20260728/fields/radius_0_selected.npz"
)
DEFAULT_REFERENCE_RESULT = (
    REPOSITORY_ROOT / "runs/fit039_janelle_exclusion_screen_20260728/result.json"
)
DEFAULT_OUT = REPOSITORY_ROOT / "runs/fit040_janelle_production_pursuit_20260728"
FIELD_TENSORS = (
    "means",
    "log_scales",
    "rotations",
    "colors",
    "opacities",
    "scale_max",
    "color_grads",
    "background_mask",
    "filter_variance",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _site_sha256(sites, *, canonical: bool) -> str:
    values = [int(site) for site in sites]
    if canonical:
        values.sort()
    digest = hashlib.sha256()
    for site in values:
        digest.update(site.to_bytes(8, byteorder="little", signed=True))
    return digest.hexdigest()


def _atomic_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _disabled_phase(phase: PhaseBudget, count: int) -> PhaseBudget:
    return replace(
        phase,
        max_steps=0,
        block_steps=1,
        target_gaussians=int(count),
    )


def _field_comparison(
    candidate: GaussianField,
    reference: GaussianField,
    *,
    inherited_rows: int,
) -> dict[str, object]:
    def canonical_order(field: GaussianField) -> torch.Tensor:
        tail = [
            (int(y), int(x), inherited_rows + offset)
            for offset, (x, y) in enumerate(
                field.means[inherited_rows:].detach().cpu().tolist()
            )
        ]
        ordered_tail = [row for _, _, row in sorted(tail)]
        return torch.tensor(
            [*range(inherited_rows), *ordered_tail],
            device=field.means.device,
            dtype=torch.long,
        )

    candidate_order = canonical_order(candidate)
    reference_order = canonical_order(reference)
    tensors = {}
    all_exact = candidate.n == reference.n
    for name in FIELD_TENSORS:
        candidate_value = getattr(candidate, name)
        reference_value = getattr(reference, name)
        if candidate_value is None or reference_value is None:
            exact = candidate_value is None and reference_value is None
            tensors[name] = {
                "exact": exact,
                "max_abs_delta": None,
            }
            all_exact = all_exact and exact
            continue
        if candidate_value.shape == reference_value.shape:
            candidate_value = candidate_value[candidate_order]
            reference_value = reference_value[reference_order]
        exact = torch.equal(candidate_value, reference_value)
        if candidate_value.shape == reference_value.shape:
            if candidate_value.dtype == torch.bool:
                maximum = float(
                    (candidate_value != reference_value).to(torch.float32).max()
                )
            else:
                maximum = float(
                    (candidate_value - reference_value).abs().max()
                )
        else:
            maximum = float("inf")
        tensors[name] = {
            "exact": exact,
            "max_abs_delta": maximum,
        }
        all_exact = all_exact and exact
    color_delta = tensors["colors"]["max_abs_delta"]
    geometry_exact = all(
        record["exact"]
        for name, record in tensors.items()
        if name != "colors"
    )
    return {
        "candidate_rows": candidate.n,
        "reference_rows": reference.n,
        "inherited_rows": inherited_rows,
        "tail_rows_compared_in_canonical_site_order": True,
        "all_tensors_exact": all_exact,
        "geometry_exact": geometry_exact,
        "colors_match_within_1e6": (
            color_delta is not None and color_delta <= 1e-6
        ),
        "tensors": tensors,
    }


def run(args: argparse.Namespace) -> None:
    if args.out.exists() and any(args.out.iterdir()):
        raise RuntimeError(f"output directory is not empty: {args.out}")
    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA but torch.cuda.is_available() is false")
    torch.manual_seed(0)

    prepared = _prepare_current_job(args.base_job)
    target = torch.as_tensor(
        prepared["target"],
        device=device,
        dtype=torch.float32,
    ).contiguous()
    mask = torch.as_tensor(
        prepared["mask"],
        device=device,
        dtype=torch.bool,
    )
    base = _scaled_field(prepared["field_path"], device, 1.0, 1.0)
    # The persisted base is already the output of the ordinary schedule. A fresh
    # run would otherwise repeat its entry-only global color solve, which does not
    # occur when the pursuit tail follows polish in the same production run.
    # One float32 iteration with an overwhelming finite ridge is an exact no-op on
    # this field; pursuit uses its own schedule-pinned 64-iteration partial solver.
    cfg = replace(
        _base_config(args),
        color_solve_maxiter=1,
        color_solve_lambda=1e30,
    )
    defaults = SafeScheduleConfig()
    schedule = SafeScheduleConfig(
        capacity=base.n,
        storage_policy="dynamic",
        boundary_enabled=True,
        coverage_target_gaussians=base.n,
        detail_target_gaussians=base.n,
        coverage_tau=float(args.coverage_tau),
        boundary_band=float(args.boundary_band),
        pursuit_tail_enabled=True,
        bootstrap=_disabled_phase(defaults.bootstrap, base.n),
        coverage=_disabled_phase(defaults.coverage, base.n),
        detail=_disabled_phase(defaults.detail, base.n),
        boundary=_disabled_phase(defaults.boundary, base.n),
        redistribution=_disabled_phase(defaults.redistribution, base.n),
        polish=_disabled_phase(defaults.polish, base.n),
    )
    result = run_safe_schedule(
        base,
        target,
        mask,
        cfg,
        schedule,
        verbose=not args.quiet,
    )
    field_path = args.out / "field.npz"
    result["field"].save(str(field_path))
    reference = GaussianField.load(str(args.reference_field), device=device)
    comparison = _field_comparison(
        result["field"],
        reference,
        inherited_rows=base.n,
    )
    pursuit = result["pursuit_tail"]
    reference_payload = json.loads(
        args.reference_result.read_text(encoding="utf-8")
    )
    reference_arm = next(
        arm
        for arm in reference_payload["arms"]
        if int(arm["exclusion_radius"]) == 0
    )
    reference_row = next(
        row for row in reference_arm["rows"] if row["target_passed"]
    )
    reference_sites = [
        site
        for row in reference_arm["rows"][: int(reference_row["stage"])]
        for site in row["batch_sites_flat"]
    ]
    reference_site_set_sha256 = _site_sha256(
        reference_sites,
        canonical=True,
    )
    acceptance = {
        "target_reached": bool(pursuit["target_reached"]),
        "minimum_rows_reproduced": int(pursuit["activated_rows"]) == 768,
        "protected_safe": not any(
            wave["protected_reasons"]
            for wave in pursuit["waves"]
            if wave.get("accepted")
        ),
        "outside_exact_zero": (
            float(result["metrics"]["outside_max_abs"]) == 0.0
            and float(result["metrics"]["outside_coverage_max"]) == 0.0
        ),
        "prototype_field_match_within_1e6": bool(
            comparison["geometry_exact"]
            and comparison["colors_match_within_1e6"]
        ),
        "prototype_site_set_hash_exact": (
            pursuit["site_set_sha256"] == reference_site_set_sha256
        ),
    }
    payload = {
        "schema": "structsplat.fit040.production_pursuit.v1",
        "task": "FIT-040",
        "source": {
            "base_job": str(args.base_job.resolve()),
            "base_field": str(prepared["field_path"]),
            "base_field_sha256": _sha256(prepared["field_path"]),
            "target_pixel_source": str(args.base_job / "target.png"),
            "target_pixel_sha256": _sha256(args.base_job / "target.png"),
            "mask": str(prepared["mask_path"]),
            "mask_sha256": _sha256(prepared["mask_path"]),
            "reference_field": str(args.reference_field.resolve()),
            "reference_field_sha256": _sha256(args.reference_field),
            "reference_result": str(args.reference_result.resolve()),
            "reference_result_sha256": _sha256(args.reference_result),
        },
        "environment": {
            "device": str(device),
            "gpu": (
                torch.cuda.get_device_name(device)
                if device.type == "cuda"
                else None
            ),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
        },
        "fit_config": asdict(cfg),
        "replay_adapter": (
            "entry global color solve made an exact no-op because base_job is post-schedule"
        ),
        "schedule": asdict(schedule),
        "pursuit_tail": pursuit,
        "metrics": result["metrics"],
        "storage": result["storage"],
        "converged": result["converged"],
        "seconds": result["seconds"],
        "field": {
            "path": str(field_path),
            "sha256": _sha256(field_path),
        },
        "prototype_comparison": comparison,
        "prototype_sites": {
            "production_ordered_sha256": pursuit["site_sha256"],
            "prototype_ordered_sha256": reference_row["all_sites_sha256"],
            "ordered_hash_exact": (
                pursuit["site_sha256"] == reference_row["all_sites_sha256"]
            ),
            "production_set_sha256": pursuit["site_set_sha256"],
            "prototype_set_sha256": reference_site_set_sha256,
            "set_hash_exact": (
                pursuit["site_set_sha256"] == reference_site_set_sha256
            ),
        },
        "acceptance": acceptance,
        "all_acceptance_checks_passed": all(acceptance.values()),
    }
    _atomic_json(args.out / "result.json", payload)
    print(json.dumps(payload["acceptance"], indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-job", type=Path, default=DEFAULT_BASE_JOB)
    parser.add_argument(
        "--reference-field",
        type=Path,
        default=DEFAULT_REFERENCE_FIELD,
    )
    parser.add_argument(
        "--reference-result",
        type=Path,
        default=DEFAULT_REFERENCE_RESULT,
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--renderer", default="cuda")
    parser.add_argument("--mask-margin", type=float, default=0.75)
    parser.add_argument("--boundary-band", type=float, default=4.0)
    parser.add_argument("--coverage-tau", type=float, default=0.05)
    parser.add_argument("--quiet", action="store_true")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
