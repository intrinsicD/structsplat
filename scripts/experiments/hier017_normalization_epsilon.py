#!/usr/bin/env python3
"""Run HIER-017's source-bound normalization-epsilon coverage diagnostic."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from html import escape
import json
import math
import platform
from pathlib import Path
import shlex
import shutil
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
for candidate in (ROOT, ROOT / "src"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from scripts.experiments import hier005_pixel_contraction as report_utils  # noqa: E402
from scripts.experiments import hier013_global_projection_development as h13  # noqa: E402
from scripts.experiments import hier015_geometry_escape as h15  # noqa: E402
from scripts.experiments import hier016_normalized_tail_repair as h16  # noqa: E402
from structsplat.config import FitConfig, StructureTensorConfig  # noqa: E402
from structsplat.fit import _normalized_color_denominator, _render, fit  # noqa: E402
from structsplat.gaussians import GaussianField  # noqa: E402
from structsplat.init import build_field  # noqa: E402
from structsplat.pixel_contraction import contract_image  # noqa: E402


REPORT_SCHEMA = "structsplat.hier017_normalization_epsilon.diagnostic.v1"
EPS_BASELINE = 1e-8
EPS_CANDIDATE = 1e-12
DEVELOPMENT_BINDINGS = {
    "COCO_train2014_000000206968.jpg": (
        "a0b28389459001bce14be52f2c37195f308909ff75b8ae21ec357d1c05857ac6"
    ),
    "COCO_train2014_000000265833.jpg": (
        "52adab433db8427ef58d526e9a766fd5826336f967d81b2000ba9ca9b17ecb7f"
    ),
    "COCO_train2014_000000048658.jpg": (
        "5ba9a6424106155ce4903e859e5af9d8a7e03bb67490dec38e327b850143b3f4"
    ),
    "COCO_train2014_000000170371.jpg": (
        "29b1ea26c79133987846e8e4930eec2c470e5cf639227cf616c9970a65d1b108"
    ),
}
SELECTION_DIGESTS = {
    "COCO_train2014_000000206968.jpg": (
        "00009784c5ab7287963bf1eced12bd8ab0b3a59a1a2592a4977b9b2553429bb5"
    ),
    "COCO_train2014_000000265833.jpg": (
        "00034eef72fa7e4009010071f92bbb8331f81f63a1a967226392c7e5896bcfe2"
    ),
    "COCO_train2014_000000048658.jpg": (
        "000397c192d0cd0a303d05f72f43313b588546457fd02f938a18c1e6318af891"
    ),
    "COCO_train2014_000000170371.jpg": (
        "00080f4ccc21f72043039d698a27e8abbf37ae623e848c8b41ed64a25e632bfd"
    ),
}
CONTROL_ARM = "h005_control"
EPS8_ARM = "direct_eps1e8"
DECODE_ARM = "decode_eps1e12"
EPS12_ARM = "fit_eps1e12"
DEVELOPMENT_ARMS = (CONTROL_ARM, EPS8_ARM, DECODE_ARM, EPS12_ARM)
REPLAY_ARMS = (EPS8_ARM, EPS12_ARM)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=("development", "replay_h15", "replay_h16", "replay_tests"),
        required=True,
    )
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--development-decision", type=Path)
    parser.add_argument("--control-metrics", type=Path)
    parser.add_argument(
        "--recover-from",
        type=Path,
        help="copy a complete raw development run and regenerate reports without rerunning cells",
    )
    parser.add_argument("--target-gaussians", type=int, default=7000)
    parser.add_argument("--max-side", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--direct-fit-steps", type=int, default=750)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--additive-renderer", default="cuda_additive")
    parser.add_argument("--direct-renderer", default="cuda")
    parser.add_argument("--render-chunk", type=int, default=256)
    parser.add_argument("--lpips", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--error-scale", type=float, default=4.0)
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    frozen = {
        "target_gaussians": 7000,
        "max_side": 512,
        "seed": 0,
        "direct_fit_steps": 750,
        "device": "cuda",
        "additive_renderer": "cuda_additive",
        "direct_renderer": "cuda",
        "render_chunk": 256,
        "lpips": True,
    }
    for name, expected in frozen.items():
        if getattr(args, name) != expected:
            raise SystemExit(
                f"frozen HIER-017 protocol requires --{name.replace('_', '-')} {expected}"
            )
    if not math.isfinite(args.error_scale) or args.error_scale <= 0.0:
        raise SystemExit("--error-scale must be finite and positive")
    replay = args.phase != "development"
    if replay and args.development_decision is None:
        raise SystemExit("replay requires --development-decision")
    if not replay and args.development_decision is not None:
        raise SystemExit("development does not accept --development-decision")
    if args.recover_from is not None and args.phase != "development":
        raise SystemExit("--recover-from is valid only for development")
    needs_controls = args.phase in ("replay_h15", "replay_h16")
    if needs_controls != (args.control_metrics is not None):
        raise SystemExit("replay_h15/replay_h16 require --control-metrics; other phases reject it")


def _validate_development_decision(args: argparse.Namespace) -> dict[str, object] | None:
    if args.phase == "development":
        return None
    decision = json.loads(args.development_decision.read_text(encoding="utf-8"))
    if decision.get("schema") != REPORT_SCHEMA or decision.get("phase") != "development":
        raise SystemExit("--development-decision is not a HIER-017 development decision")
    if EPS12_ARM not in decision.get("numeric_candidates", []):
        raise SystemExit("fit_eps1e12 did not pass the frozen development numeric gate")
    return decision


def _bound_paths(directory: Path, bindings: dict[str, str]) -> list[Path]:
    paths = [directory / name for name in bindings]
    actual = {
        path.name: report_utils._sha256(path)
        for path in paths
        if path.is_file()
    }
    if actual != bindings:
        raise SystemExit(f"source binding mismatch: expected {bindings}, got {actual}")
    return [path.resolve() for path in paths]


def _discover_images(args: argparse.Namespace) -> list[Path]:
    if args.phase == "development":
        return _bound_paths(args.images, DEVELOPMENT_BINDINGS)
    if args.phase == "replay_h15":
        return _bound_paths(args.images, h15.DEVELOPMENT_BINDINGS)
    if args.phase == "replay_h16":
        return _bound_paths(args.images, h16.DEVELOPMENT_BINDINGS)
    return h13._discover_sources([args.images])


def _direct_configs(args: argparse.Namespace, epsilon: float):
    init_config, fit_config = h15._direct_configs(args)
    return init_config, replace(fit_config, normalization_eps=epsilon)


def _snapshot_sources(output_root: Path) -> list[dict[str, object]]:
    paths = (
        Path(__file__).resolve(),
        ROOT / "src" / "structsplat" / "config.py",
        ROOT / "src" / "structsplat" / "render.py",
        ROOT / "src" / "structsplat" / "cuda_render.py",
        ROOT / "src" / "structsplat" / "fit.py",
        ROOT / "src" / "structsplat" / "codec.py",
        ROOT / "scripts" / "experiments" / "hier015_geometry_escape.py",
        ROOT / "tasks" / "HIER-017-normalization-epsilon-coverage.md",
    )
    records: list[dict[str, object]] = []
    for source in paths:
        relative = source.relative_to(ROOT)
        destination = output_root / "source_snapshot" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        records.append(
            {
                "repository_path": str(relative),
                "snapshot_path": str(destination.relative_to(output_root)),
                "bytes": destination.stat().st_size,
                "sha256": report_utils._sha256(destination),
            }
        )
    return records


def _seed_everything(seed: int) -> None:
    import torch

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _run_fit(
    initial: GaussianField,
    image: np.ndarray,
    fit_config: FitConfig,
) -> tuple[GaussianField, dict[str, object], int]:
    import torch

    _seed_everything(0)
    torch.cuda.reset_peak_memory_stats()
    target = torch.as_tensor(
        image, device=initial.means.device, dtype=torch.float32
    ).contiguous()
    result = fit(initial.detached(), target, fit_config, verbose=False)
    return result["field"], result, int(torch.cuda.max_memory_allocated())


def _render_numpy(field: GaussianField, cfg: FitConfig, height: int, width: int) -> np.ndarray:
    import torch

    with torch.no_grad():
        value = _render(field, cfg, height, width)
    return value.detach().cpu().numpy().astype(np.float32, copy=False)


def _pseudo_decode_result(rendered: np.ndarray) -> dict[str, object]:
    import torch

    return {
        "render": torch.as_tensor(np.array(rendered, copy=True)),
        "history": {
            "kind": "decode_only_epsilon_attribution",
            "source_fit_epsilon": EPS_BASELINE,
            "render_epsilon": EPS_CANDIDATE,
        },
        "fit_seconds": 0.0,
        "iterations_run": 0,
        "selected_iter": 0,
        # The persisted field was optimized under 1e-8; only this cold interpretation uses 1e-12.
        "normalization_eps": EPS_BASELINE,
    }


def _quantiles(values: np.ndarray) -> dict[str, float]:
    return {
        f"q{label}": float(np.quantile(values, quantile))
        for label, quantile in (
            ("000", 0.0),
            ("001", 0.001),
            ("010", 0.01),
            ("500", 0.5),
            ("990", 0.99),
            ("999", 0.999),
            ("1000", 1.0),
        )
    }


def _denominator_telemetry(
    *,
    field: GaussianField,
    cfg: FitConfig,
    image: np.ndarray,
    reconstruction: np.ndarray,
    artifact_dir: Path,
) -> dict[str, object]:
    height, width = image.shape[:2]
    denominator = (
        _normalized_color_denominator(field, cfg, height, width)
        .reshape(height, width)
        .detach()
        .cpu()
        .numpy()
        .astype(np.float64)
    )
    attenuation = denominator / (denominator + float(cfg.normalization_eps))
    pixel_rmse = np.sqrt(
        np.mean(
            (reconstruction.astype(np.float64) - image.astype(np.float64)) ** 2,
            axis=2,
        )
    )
    bands = {
        "zero": denominator == 0.0,
        "positive_below_1e12": (denominator > 0.0) & (denominator < EPS_CANDIDATE),
        "1e12_to_1e8": (denominator >= EPS_CANDIDATE) & (denominator < EPS_BASELINE),
        "at_least_1e8": denominator >= EPS_BASELINE,
    }
    band_records: dict[str, dict[str, float | int | None]] = {}
    for name, active in bands.items():
        values = pixel_rmse[active]
        band_records[name] = {
            "count": int(active.sum()),
            "mean_pixel_rmse": float(values.mean()) if values.size else None,
            "max_pixel_rmse": float(values.max()) if values.size else None,
        }
    record: dict[str, object] = {
        "normalization_eps": float(cfg.normalization_eps),
        "denominator_quantiles": _quantiles(denominator),
        "attenuation_quantiles": _quantiles(attenuation),
        "denominator_exact_zero_count": int(np.count_nonzero(denominator == 0.0)),
        "denominator_below_1e12_count": int(np.count_nonzero(denominator < EPS_CANDIDATE)),
        "denominator_below_1e8_count": int(np.count_nonzero(denominator < EPS_BASELINE)),
        "error_by_denominator_band": band_records,
    }
    report_utils._write_json(artifact_dir / "denominator.json", record)
    np.savez_compressed(
        artifact_dir / "denominator.npz",
        denominator=denominator.astype(np.float32),
        attenuation=attenuation.astype(np.float32),
        pixel_rmse=pixel_rmse.astype(np.float32),
    )
    return {
        **record,
        "denominator_min": float(denominator.min()),
        "denominator_q001": float(np.quantile(denominator, 0.001)),
        "denominator_q01": float(np.quantile(denominator, 0.01)),
        "attenuation_min": float(attenuation.min()),
    }


def _write_cell(
    *,
    output_root: Path,
    image_path: Path,
    image: np.ndarray,
    mask: np.ndarray,
    raster: dict[str, object],
    arm: str,
    field: GaussianField,
    fit_result: dict[str, object],
    init_seconds: float,
    control_reconstruction: np.ndarray,
    peak_cuda_bytes: int,
    fit_config: FitConfig,
    init_hash: str,
    args: argparse.Namespace,
    extra_row: dict[str, object] | None = None,
    schema: str = REPORT_SCHEMA,
) -> tuple[dict[str, object], np.ndarray]:
    extras: dict[str, object] = {
        "semantic_family": "normalized_weighted_sum_configurable_epsilon_v1",
        "normalization_eps": float(fit_config.normalization_eps),
        "fit_normalization_eps": float(fit_result["normalization_eps"]),
        "render_normalization_eps": float(fit_config.normalization_eps),
        "initial_field_content_sha256": init_hash,
    }
    if extra_row:
        extras.update(extra_row)
    row = h15._write_direct_cell(
        output_root=output_root,
        image_path=image_path,
        image=image,
        mask=mask,
        raster=raster,
        field=field,
        fit_result=fit_result,
        init_seconds=init_seconds,
        control_reconstruction=control_reconstruction,
        peak_cuda_bytes=peak_cuda_bytes,
        fit_config=fit_config,
        args=args,
        arm=arm,
        schema=schema,
        extra_row=extras,
    )
    artifact_dir = output_root / str(row["artifact_dir"])
    cold = _render_numpy(field, fit_config, image.shape[0], image.shape[1])
    telemetry = _denominator_telemetry(
        field=field,
        cfg=fit_config,
        image=image,
        reconstruction=cold,
        artifact_dir=artifact_dir,
    )
    row.update(telemetry)
    report_utils._write_json(artifact_dir / "row.json", row)
    return row, cold


def _display_pixel_rmse(reconstruction: np.ndarray, image: np.ndarray) -> np.ndarray:
    displayed = np.rint(np.clip(reconstruction, 0.0, 1.0) * 255.0) / 255.0
    target = np.rint(np.clip(image, 0.0, 1.0) * 255.0) / 255.0
    return np.sqrt(np.mean((displayed - target) ** 2, axis=2))


def _epsilon_sensitivity(
    *,
    image: np.ndarray,
    baseline: np.ndarray,
    decoded: np.ndarray,
    candidate: np.ndarray,
) -> dict[str, object]:
    raw_delta = np.max(np.abs(decoded.astype(np.float64) - baseline), axis=2)
    sensitive = raw_delta >= 1.0 / 255.0
    baseline_error = _display_pixel_rmse(baseline, image)
    candidate_error = _display_pixel_rmse(candidate, image)
    improvement = baseline_error - candidate_error
    if sensitive.any():
        masked = np.where(sensitive, improvement, -np.inf)
        y, x = np.unravel_index(int(np.argmax(masked)), masked.shape)
        maximum = float(masked[y, x])
        coordinate: list[int] | None = [int(y), int(x)]
    else:
        maximum = 0.0
        coordinate = None
    max_y, max_x = np.unravel_index(int(np.argmax(raw_delta)), raw_delta.shape)
    return {
        "epsilon_sensitive_pixel_count_8bit": int(sensitive.sum()),
        "epsilon_sensitive_display_error_improvement_max": maximum,
        "epsilon_sensitive_best_yx": coordinate,
        "epsilon_interpretation_delta_max": float(raw_delta[max_y, max_x]),
        "epsilon_interpretation_delta_max_yx": [int(max_y), int(max_x)],
        "epsilon_interpretation_baseline_rgb": baseline[max_y, max_x].tolist(),
        "epsilon_interpretation_eps12_rgb": decoded[max_y, max_x].tolist(),
        "epsilon_interpretation_source_rgb": image[max_y, max_x].tolist(),
    }


def _pairs(
    rows: list[dict[str, object]], arm: str, control_arm: str
) -> list[dict[str, object]]:
    controls = {
        str(row["image"]): row for row in rows if row["arm"] == control_arm
    }
    pairs: list[dict[str, object]] = []
    for row in rows:
        control = controls.get(str(row["image"]))
        if row["arm"] != arm or control is None:
            continue
        pairs.append(
            {
                "image": row["image"],
                "mse_ratio": float(row["masked_mse"]) / float(control["masked_mse"]),
                "psnr_delta_db": float(row["psnr_db"]) - float(control["psnr_db"]),
                "ms_ssim_delta": float(row["ms_ssim"]) - float(control["ms_ssim"]),
                "lpips_delta": float(row["lpips"]) - float(control["lpips"]),
                "pixel_max_delta": float(row["artifact_pixel_rmse_max"])
                - float(control["artifact_pixel_rmse_max"]),
                "patch7_max_delta": float(row["artifact_patch_rmse_max_7"])
                - float(control["artifact_patch_rmse_max_7"]),
                "n_gaussians": row["n_gaussians"],
                "normalization_eps": row["normalization_eps"],
                "fit_normalization_eps": row["fit_normalization_eps"],
                "render_normalization_eps": row["render_normalization_eps"],
                "initial_field_content_sha256": row["initial_field_content_sha256"],
                "control_initial_field_content_sha256": control.get(
                    "initial_field_content_sha256"
                ),
                "maintained_render_parity_max_abs": row[
                    "maintained_render_parity_max_abs"
                ],
                "repeated_render_parity_max_abs": row[
                    "repeated_render_parity_max_abs"
                ],
                "epsilon_sensitive_display_error_improvement_max": row.get(
                    "epsilon_sensitive_display_error_improvement_max", 0.0
                ),
            }
        )
    return pairs


def _finite_pairs(pairs: list[dict[str, object]]) -> bool:
    keys = (
        "mse_ratio",
        "psnr_delta_db",
        "ms_ssim_delta",
        "lpips_delta",
        "pixel_max_delta",
        "patch7_max_delta",
    )
    try:
        return all(math.isfinite(float(pair[key])) for pair in pairs for key in keys)
    except (TypeError, ValueError):
        return False


def _development_decision(
    rows: list[dict[str, object]], attempts: list[dict[str, object]]
) -> dict[str, object]:
    h005_pairs = _pairs(rows, EPS12_ARM, CONTROL_ARM)
    eps8_pairs = _pairs(rows, EPS12_ARM, EPS8_ARM)
    decode_pairs = _pairs(rows, DECODE_ARM, EPS8_ARM)
    gates = {
        "complete_four_h005_pairs": len(h005_pairs) == 4,
        "complete_four_eps8_pairs": len(eps8_pairs) == 4,
        "complete_four_decode_pairs": len(decode_pairs) == 4,
        "all_finite": _finite_pairs(h005_pairs + eps8_pairs + decode_pairs),
        "all_exact_count": all(int(pair["n_gaussians"]) == 7000 for pair in eps8_pairs),
        "all_eps_consistent": all(
            float(pair["normalization_eps"]) == EPS_CANDIDATE
            and float(pair["fit_normalization_eps"]) == EPS_CANDIDATE
            and float(pair["render_normalization_eps"]) == EPS_CANDIDATE
            for pair in eps8_pairs
        ),
        "all_same_initialization": all(
            pair["initial_field_content_sha256"]
            == pair["control_initial_field_content_sha256"]
            for pair in eps8_pairs
        ) if len(eps8_pairs) == 4 else False,
        "all_parity_le_2e_5": all(
            float(pair["maintained_render_parity_max_abs"]) <= 2e-5
            and float(pair["repeated_render_parity_max_abs"]) <= 2e-5
            for pair in eps8_pairs
        ),
        "all_psnr_gain_vs_h005_ge_2_db": all(
            float(pair["psnr_delta_db"]) >= 2.0 for pair in h005_pairs
        ),
        "all_pixel_max_noninferior_vs_h005": all(
            float(pair["pixel_max_delta"]) <= 1e-12 for pair in h005_pairs
        ),
        "all_patch7_max_noninferior_vs_h005": all(
            float(pair["patch7_max_delta"]) <= 1e-12 for pair in h005_pairs
        ),
        "mean_ms_ssim_noninferior_vs_h005": (
            float(np.mean([pair["ms_ssim_delta"] for pair in h005_pairs])) >= -1e-7
            if h005_pairs else False
        ),
        "mean_lpips_noninferior_vs_h005": (
            float(np.mean([pair["lpips_delta"] for pair in h005_pairs])) <= 1e-7
            if h005_pairs else False
        ),
        "all_mse_noninferior_vs_eps8": all(
            float(pair["mse_ratio"]) <= 1.0 + 1e-8 for pair in eps8_pairs
        ),
        "all_pixel_max_noninferior_vs_eps8": all(
            float(pair["pixel_max_delta"]) <= 1e-12 for pair in eps8_pairs
        ),
        "all_patch7_max_noninferior_vs_eps8": all(
            float(pair["patch7_max_delta"]) <= 1e-12 for pair in eps8_pairs
        ),
        "mean_ms_ssim_delta_vs_eps8_ge_neg_0_001": (
            float(np.mean([pair["ms_ssim_delta"] for pair in eps8_pairs])) >= -0.001
            if eps8_pairs else False
        ),
        "mean_lpips_delta_vs_eps8_le_0_002": (
            float(np.mean([pair["lpips_delta"] for pair in eps8_pairs])) <= 0.002
            if eps8_pairs else False
        ),
        "at_least_one_epsilon_sensitive_improvement_ge_1_255": any(
            float(pair["epsilon_sensitive_display_error_improvement_max"])
            >= 1.0 / 255.0
            for pair in eps8_pairs
        ),
    }
    candidates = [EPS12_ARM] if all(gates.values()) else []
    return {
        "schema": REPORT_SCHEMA,
        "status": "diagnostic",
        "phase": "development",
        "gates": gates,
        "h005_pairs": h005_pairs,
        "eps8_pairs": eps8_pairs,
        "decode_attribution_pairs": decode_pairs,
        "attempt_count": len(attempts),
        "failure_count": sum(record.get("status") != "ok" for record in attempts),
        "numeric_candidates": candidates,
        "numeric_disposition": candidates[0] if candidates else "no_epsilon_candidate",
        "visual_review_required": True,
        "interpretation": (
            "Numeric candidate requires frozen full-frame/worst-crop visual review before replay."
            if candidates
            else "The fixed epsilon candidate misses a frozen numeric gate; do not replay."
        ),
    }


def _external_control_pairs(
    rows: list[dict[str, object]], control_metrics: Path | None
) -> list[dict[str, object]]:
    if control_metrics is None:
        return []
    report = json.loads(control_metrics.read_text(encoding="utf-8"))
    controls = {
        str(row["image"]): row
        for row in report["rows"]
        if row["arm"] == CONTROL_ARM
    }
    candidates = {
        str(row["image"]): row for row in rows if row["arm"] == EPS12_ARM
    }
    return [
        {
            "image": image,
            "psnr_delta_db": float(candidates[image]["psnr_db"])
            - float(control["psnr_db"]),
            "pixel_max_delta": float(candidates[image]["artifact_pixel_rmse_max"])
            - float(control["artifact_pixel_rmse_max"]),
            "patch7_max_delta": float(candidates[image]["artifact_patch_rmse_max_7"])
            - float(control["artifact_patch_rmse_max_7"]),
        }
        for image, control in sorted(controls.items())
        if image in candidates
    ]


def _replay_decision(
    rows: list[dict[str, object]], attempts: list[dict[str, object]], args: argparse.Namespace
) -> dict[str, object]:
    pairs = _pairs(rows, EPS12_ARM, EPS8_ARM)
    expected = 16 if args.phase == "replay_tests" else 4
    clauses = {
        "complete_pairs": len(pairs) == expected,
        "all_finite": _finite_pairs(pairs),
        "all_exact_count": all(int(pair["n_gaussians"]) == 7000 for pair in pairs),
        "all_eps_consistent": all(
            float(pair["normalization_eps"]) == EPS_CANDIDATE
            and float(pair["fit_normalization_eps"]) == EPS_CANDIDATE
            and float(pair["render_normalization_eps"]) == EPS_CANDIDATE
            for pair in pairs
        ),
        "all_parity_le_2e_5": all(
            float(pair["maintained_render_parity_max_abs"]) <= 2e-5
            and float(pair["repeated_render_parity_max_abs"]) <= 2e-5
            for pair in pairs
        ),
        "all_mse_noninferior_vs_eps8": all(
            float(pair["mse_ratio"]) <= 1.0 + 1e-8 for pair in pairs
        ),
        "all_pixel_max_noninferior_vs_eps8": all(
            float(pair["pixel_max_delta"]) <= 1e-12 for pair in pairs
        ),
        "all_patch7_max_noninferior_vs_eps8": all(
            float(pair["patch7_max_delta"]) <= 1e-12 for pair in pairs
        ),
    }
    external = _external_control_pairs(rows, args.control_metrics)
    if args.phase in ("replay_h15", "replay_h16"):
        clauses.update(
            {
                "complete_external_control_pairs": len(external) == 4,
                "all_psnr_gain_vs_h005_ge_2_db": all(
                    pair["psnr_delta_db"] >= 2.0 for pair in external
                ),
                "all_pixel_max_noninferior_vs_h005": all(
                    pair["pixel_max_delta"] <= 1e-12 for pair in external
                ),
                "all_patch7_max_noninferior_vs_h005": all(
                    pair["patch7_max_delta"] <= 1e-12 for pair in external
                ),
            }
        )
    return {
        "schema": REPORT_SCHEMA,
        "status": "diagnostic",
        "phase": args.phase,
        "pairs": pairs,
        "external_h005_pairs": external,
        "clauses": clauses,
        "attempt_count": len(attempts),
        "failure_count": sum(record.get("status") != "ok" for record in attempts),
        "bounded_bank_pass": all(clauses.values()),
        "interpretation": "Consumed reporting replay; no retuning or held-out claim.",
    }


def _write_report(
    output_root: Path,
    rows: list[dict[str, object]],
    decision: dict[str, object],
    command: str,
) -> None:
    table: list[str] = []
    cards: list[str] = []
    for row in rows:
        artifact = escape(str(row["artifact_dir"]))
        epsilon = row.get("normalization_eps")
        epsilon_text = "—" if epsilon is None else f"{float(epsilon):.0e}"
        coverage_link = (
            f" · <a href='{artifact}/denominator.json'>coverage</a>"
            if epsilon is not None
            else ""
        )
        table.append(
            "<tr>"
            f"<td>{escape(str(row['image']))}</td><td>{escape(str(row['arm']))}</td>"
            f"<td>{epsilon_text}</td>"
            f"<td>{int(row['n_gaussians'])}</td><td>{float(row['psnr_db']):.3f}</td>"
            f"<td>{float(row['ms_ssim']):.5f}</td><td>{float(row['lpips']):.5f}</td>"
            f"<td>{float(row['artifact_pixel_rmse_max']):.4f}</td>"
            f"<td>{float(row['artifact_patch_rmse_max_7']):.4f}</td>"
            f"<td>{int(row.get('denominator_below_1e8_count', 0))}</td>"
            f"<td><a href='{artifact}/reconstruction.png'>full</a> · "
            f"<a href='{artifact}/reconstruction_crop.png'>crop</a>"
            f"{coverage_link}</td></tr>"
        )
        cards.append(
            f"<section><h3>{escape(str(row['image']))} — {escape(str(row['arm']))}</h3>"
            f"<img src='{artifact}/source.png'><img src='{artifact}/reconstruction.png'>"
            f"<img src='{artifact}/error.png'><img src='{artifact}/reconstruction_crop.png'>"
            "</section>"
        )
    decision_text = escape(json.dumps(decision, indent=2, sort_keys=True))
    html = f"""<!doctype html><html><head><meta charset='utf-8'>
<title>HIER-017 normalization epsilon</title><style>
body{{font-family:system-ui;margin:2rem;max-width:1600px}}table{{border-collapse:collapse}}
th,td{{border:1px solid #bbb;padding:.35rem}}img{{max-width:300px;max-height:240px;margin:.25rem}}
section{{border-top:1px solid #aaa;margin-top:1rem}}pre,code{{white-space:pre-wrap}}
</style></head><body><h1>HIER-017 normalization epsilon — {escape(str(decision['phase']))}</h1>
<p>Dirty-source diagnostic. Decode-only epsilon reinterpretation is attribution, never a candidate.</p>
<p><code>{escape(command)}</code></p><p><a href='config.json'>config</a> ·
<a href='decision.json'>decision</a> · <a href='metrics.json'>metrics</a> ·
<a href='metrics.jsonl'>JSONL</a> · <a href='metrics.csv'>CSV</a> ·
<a href='attempts.json'>attempts</a> · <a href='manifest.json'>manifest</a></p>
<h2>Decision</h2><pre>{decision_text}</pre><h2>Cells</h2><table><tr><th>image</th>
<th>arm</th><th>epsilon</th><th>N</th><th>PSNR</th><th>MS-SSIM</th><th>LPIPS</th>
<th>pixel max</th><th>7x7 max</th><th>den&lt;1e-8</th><th>artifacts</th></tr>
{''.join(table)}</table><h2>Visual audit</h2>{''.join(cards)}</body></html>"""
    (output_root / "index.html").write_text(html, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    _validate_args(args)
    _validate_development_decision(args)
    images = _discover_images(args)
    output_root = args.out.resolve()

    import torch

    command = shlex.join([sys.executable, str(Path(__file__).resolve()), *(argv or sys.argv[1:])])
    if args.recover_from is not None:
        source_root = args.recover_from.resolve()
        if not source_root.is_dir():
            raise SystemExit(f"recovery source is not a directory: {source_root}")
        source_metrics = source_root / "metrics.json"
        source_attempts = source_root / "attempts.json"
        source_decision = source_root / "decision.json"
        if not all(path.is_file() for path in (source_metrics, source_attempts, source_decision)):
            raise SystemExit("recovery source lacks metrics.json, attempts.json, or decision.json")
        metrics_payload = json.loads(source_metrics.read_text(encoding="utf-8"))
        attempts_payload = json.loads(source_attempts.read_text(encoding="utf-8"))
        decision = json.loads(source_decision.read_text(encoding="utf-8"))
        rows = metrics_payload.get("rows", [])
        attempts = attempts_payload.get("attempts", [])
        expected_cells = len(DEVELOPMENT_BINDINGS) * len(DEVELOPMENT_ARMS)
        if (
            metrics_payload.get("schema") != REPORT_SCHEMA
            or decision.get("schema") != REPORT_SCHEMA
            or len(rows) != expected_cells
            or len(attempts) != expected_cells
            or any(record.get("status") != "ok" for record in attempts)
        ):
            raise SystemExit("recovery source is not a complete successful HIER-017 run")
        shutil.copytree(source_root, output_root)
        recovery_snapshot = output_root / "recovery_source_snapshot" / Path(__file__).name
        recovery_snapshot.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(Path(__file__).resolve(), recovery_snapshot)
        decision["recovered_from_complete_raw_run"] = True
        decision["cell_computation_rerun"] = False
        report_utils._write_json(output_root / "decision.json", decision)
        report_utils._write_json(
            output_root / "recovery.json",
            {
                "schema": REPORT_SCHEMA,
                "status": "diagnostic",
                "command": command,
                "source_path": str(source_root),
                "source_metrics_sha256": report_utils._sha256(source_metrics),
                "source_attempts_sha256": report_utils._sha256(source_attempts),
                "source_decision_sha256": report_utils._sha256(source_decision),
                "cell_computation_rerun": False,
                "recovery_action": "copy complete raw run; regenerate report and manifest",
                "recovery_driver_snapshot": str(recovery_snapshot.relative_to(output_root)),
                "recovery_driver_sha256": report_utils._sha256(recovery_snapshot),
            },
        )
        h15._write_tables(output_root, rows, schema=REPORT_SCHEMA)
        _write_report(output_root, rows, decision, command)
        h15._write_manifest(output_root, schema=REPORT_SCHEMA)
        print(json.dumps(decision, indent=2, sort_keys=True))
        return 0

    output_root.mkdir(parents=True, exist_ok=False)
    contraction_config = h15._contraction_config(args)
    init_config, eps8_config = _direct_configs(args, EPS_BASELINE)
    _, eps12_config = _direct_configs(args, EPS_CANDIDATE)
    arms = DEVELOPMENT_ARMS if args.phase == "development" else REPLAY_ARMS
    control_metrics = None
    if args.control_metrics is not None:
        control_metrics = {
            "path": str(args.control_metrics.resolve()),
            "sha256": report_utils._sha256(args.control_metrics),
        }
    config = {
        "schema": REPORT_SCHEMA,
        "status": "diagnostic",
        "phase": args.phase,
        "command": command,
        "args": vars(args),
        "arms": list(arms),
        "development_selection_digests": SELECTION_DIGESTS,
        "sources": [
            {"path": str(path), "sha256": report_utils._sha256(path)} for path in images
        ],
        "contraction": asdict(contraction_config),
        "direct_init": asdict(init_config),
        "direct_eps1e8": asdict(eps8_config),
        "fit_eps1e12": asdict(eps12_config),
        "decode_eps1e12": {
            "source_fit_epsilon": EPS_BASELINE,
            "render_epsilon": EPS_CANDIDATE,
            "candidate_eligible": False,
        },
        "control_metrics": control_metrics,
        "source_snapshots": _snapshot_sources(output_root),
        "git": report_utils._git_record(),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        },
        "limitations": [
            "Dirty-source, one-seed diagnostic without distinct prospective review.",
            "CUDA accumulation is numerically, not bit, reproducible.",
            "Decode-only epsilon reinterpretation is causal attribution, never selection.",
            "Consumed replays are reporting-only and cannot tune the recipe.",
        ],
    }
    report_utils._write_json(output_root / "config.json", config)

    rows: list[dict[str, object]] = []
    attempts: list[dict[str, object]] = []

    def record(image_path: Path, arm: str, started: float, error: Exception | None = None) -> None:
        item: dict[str, object] = {
            "image": image_path.stem,
            "arm": arm,
            "status": "ok" if error is None else "error",
            "elapsed_seconds": time.perf_counter() - started,
        }
        if error is not None:
            item["error"] = f"{type(error).__name__}: {error}"[:1000]
        attempts.append(item)
        report_utils._write_json(
            output_root / "attempts.json",
            {"schema": REPORT_SCHEMA, "status": "diagnostic", "attempts": attempts},
        )

    def persist() -> None:
        h15._write_tables(output_root, rows, schema=REPORT_SCHEMA)

    run_started = time.perf_counter()
    for image_path in images:
        load_started = time.perf_counter()
        try:
            image, loaded_mask, raster = report_utils._load_evaluation_raster(
                image_path, None, max_side=args.max_side, mask_threshold=0.5
            )
            if loaded_mask is not None:
                raise RuntimeError("HIER-017 requires a generated full-frame mask")
        except Exception as exc:
            for arm in arms:
                record(image_path, arm, load_started, exc)
            continue
        mask = np.ones(image.shape[:2], dtype=bool)

        control_reconstruction = image
        if CONTROL_ARM in arms:
            control_started = time.perf_counter()
            try:
                _seed_everything(args.seed)
                torch.cuda.reset_peak_memory_stats()
                control = contract_image(image, contraction_config, mask=mask)
                control_seconds = time.perf_counter() - control_started
                row = h15._write_observation_cell(
                    output_root=output_root,
                    image_path=image_path,
                    image=image,
                    mask=mask,
                    raster=raster,
                    arm=CONTROL_ARM,
                    field=control.field,
                    control_field=control.field,
                    control_reconstruction=control.reconstruction,
                    expected=control.reconstruction,
                    contraction_seconds=control_seconds,
                    method_seconds=0.0,
                    projection=None,
                    alternating=None,
                    peak_cuda_bytes=int(torch.cuda.max_memory_allocated()),
                    args=args,
                    schema=REPORT_SCHEMA,
                )
                rows.append(row)
                control_reconstruction = control.reconstruction
                record(image_path, CONTROL_ARM, control_started)
                persist()
            except Exception as exc:
                record(image_path, CONTROL_ARM, control_started, exc)

        init_started = time.perf_counter()
        try:
            _seed_everything(args.seed)
            initial = build_field(
                image, init_config, StructureTensorConfig(), device=args.device
            )
            init_seconds = time.perf_counter() - init_started
            init_hash = h15._gaussian_content_hash(initial)
        except Exception as exc:
            for arm in (candidate for candidate in arms if candidate != CONTROL_ARM):
                record(image_path, arm, init_started, exc)
            continue

        eps8_field = None
        eps8_render = None
        eps8_started = time.perf_counter()
        try:
            eps8_field, eps8_result, eps8_peak = _run_fit(initial, image, eps8_config)
            row, eps8_render = _write_cell(
                output_root=output_root,
                image_path=image_path,
                image=image,
                mask=mask,
                raster=raster,
                arm=EPS8_ARM,
                field=eps8_field,
                fit_result=eps8_result,
                init_seconds=init_seconds,
                control_reconstruction=control_reconstruction,
                peak_cuda_bytes=eps8_peak,
                fit_config=eps8_config,
                init_hash=init_hash,
                args=args,
                extra_row={"same_initialization_contract": True},
            )
            rows.append(row)
            record(image_path, EPS8_ARM, eps8_started)
            persist()
        except Exception as exc:
            record(image_path, EPS8_ARM, eps8_started, exc)

        decoded_render = None
        if DECODE_ARM in arms:
            decode_started = time.perf_counter()
            if eps8_field is None or eps8_render is None:
                record(
                    image_path,
                    DECODE_ARM,
                    decode_started,
                    RuntimeError("direct_eps1e8 source field failed"),
                )
            else:
                try:
                    decoded_render = _render_numpy(
                        eps8_field, eps12_config, image.shape[0], image.shape[1]
                    )
                    row, decoded_render = _write_cell(
                        output_root=output_root,
                        image_path=image_path,
                        image=image,
                        mask=mask,
                        raster=raster,
                        arm=DECODE_ARM,
                        field=eps8_field,
                        fit_result=_pseudo_decode_result(decoded_render),
                        init_seconds=0.0,
                        control_reconstruction=eps8_render,
                        peak_cuda_bytes=0,
                        fit_config=eps12_config,
                        init_hash=init_hash,
                        args=args,
                        extra_row={
                            "attribution_only": True,
                            "source_fit_normalization_eps": EPS_BASELINE,
                            "same_field_as_eps8": True,
                        },
                    )
                    rows.append(row)
                    record(image_path, DECODE_ARM, decode_started)
                    persist()
                except Exception as exc:
                    record(image_path, DECODE_ARM, decode_started, exc)

        eps12_started = time.perf_counter()
        try:
            eps12_field, eps12_result, eps12_peak = _run_fit(initial, image, eps12_config)
            eps12_row, eps12_render = _write_cell(
                output_root=output_root,
                image_path=image_path,
                image=image,
                mask=mask,
                raster=raster,
                arm=EPS12_ARM,
                field=eps12_field,
                fit_result=eps12_result,
                init_seconds=init_seconds,
                control_reconstruction=(
                    eps8_render if eps8_render is not None else control_reconstruction
                ),
                peak_cuda_bytes=eps12_peak,
                fit_config=eps12_config,
                init_hash=init_hash,
                args=args,
                extra_row={"same_initialization_contract": True},
            )
            if eps8_render is not None:
                if decoded_render is None:
                    decoded_render = _render_numpy(
                        eps8_field, eps12_config, image.shape[0], image.shape[1]
                    )
                sensitivity = _epsilon_sensitivity(
                    image=image,
                    baseline=eps8_render,
                    decoded=decoded_render,
                    candidate=eps12_render,
                )
                eps12_row.update(sensitivity)
                artifact_dir = output_root / str(eps12_row["artifact_dir"])
                report_utils._write_json(artifact_dir / "epsilon_sensitivity.json", sensitivity)
                report_utils._write_json(artifact_dir / "row.json", eps12_row)
            rows.append(eps12_row)
            record(image_path, EPS12_ARM, eps12_started)
            persist()
        except Exception as exc:
            record(image_path, EPS12_ARM, eps12_started, exc)

    decision = (
        _development_decision(rows, attempts)
        if args.phase == "development"
        else _replay_decision(rows, attempts, args)
    )
    decision["elapsed_seconds"] = time.perf_counter() - run_started
    report_utils._write_json(output_root / "decision.json", decision)
    h15._write_tables(output_root, rows, schema=REPORT_SCHEMA)
    _write_report(output_root, rows, decision, command)
    h15._write_manifest(output_root, schema=REPORT_SCHEMA)
    print(json.dumps(decision, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
