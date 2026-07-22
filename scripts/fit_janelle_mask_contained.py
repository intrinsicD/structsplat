#!/usr/bin/env python3
"""Fit the masked Janelle captures into capped realtime-gs compact views.

This is a task-specific, resumable bridge between StructSplat's CORE-010 hard mask
containment and realtime-gs's self-contained ``.rtgsv`` observation format.  It deliberately
does not delete the source RGB or mask directories.

The complete compact view (Gaussian field, camera, provenance, and packed alpha) is capped at
168,000 decimal bytes.  A larger fitted candidate is activity-pruned only at serialization time,
so fitting still reaches the requested capacity frontier.  Per-view JSON and CSV sidecars retain
the convergence curve, target hits, stage plateaus, and wall-clock timings.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REALTIME_ROOT = Path("/home/alex/Documents/realtime-gs")
DEFAULT_DATASET_ROOT = Path("/home/alex/Dropbox/Work/Janelle")


class _HelpFormatter(
    argparse.ArgumentDefaultsHelpFormatter,
    argparse.RawDescriptionHelpFormatter,
):
    """Preserve lifecycle notes while showing every resolved default."""


def _add_common(parser: argparse.ArgumentParser) -> None:
    inputs = parser.add_argument_group("input and execution")
    inputs.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help=(
            "Janelle dataset root searched by realtime-gs for calibrated frame_*/rgb and "
            "frame_*/mask inputs. Outputs are written below each source frame's gaussians2d "
            "directory."
        ),
    )
    inputs.add_argument(
        "--realtime-root",
        type=Path,
        default=DEFAULT_REALTIME_ROOT,
        help=(
            "realtime-gs source checkout that supplies calibrated preprocessing, compact-view "
            "serialization, and the 168,000-byte format contract. It must match "
            "STRUCTSPLAT_REALTIME_ROOT when that environment variable is set."
        ),
    )
    inputs.add_argument(
        "--device",
        default="cuda:0",
        help=(
            "PyTorch CUDA device used for initialization, fitting, export scoring, and final "
            "metrics. This production bridge rejects CPU devices."
        ),
    )

    capacity = parser.add_argument_group("capacity and growth lifecycle")
    capacity.add_argument(
        "--candidate-gaussians",
        type=int,
        default=5_750,
        help=(
            "Working-field row ceiling before serialization. The initializer builds this many "
            "candidate rows and growth cannot exceed it. The final .rtgsv may contain fewer rows "
            "because export drops the lowest total-foreground-activity rows until the complete "
            "file fits 168,000 bytes."
        ),
    )
    capacity.add_argument(
        "--start-gaussians",
        type=int,
        default=5_750,
        help=(
            "Prefix of the candidate pool present at optimizer entry. It must be at least 16 and "
            "no larger than --candidate-gaussians. Equal values produce a fixed-count fit; a "
            "smaller value requires the selected growth lifecycle to add the difference."
        ),
    )
    capacity.add_argument(
        "--growth-mode",
        choices=("residual", "boundary"),
        default="residual",
        help=(
            "How rows are born when start < candidate. 'residual' enables generic "
            "residual-tensor split events controlled by --growth-waves/--growth-every. "
            "'boundary' disables generic splits and enables tangent-aligned boundary-residual "
            "births controlled by --mask-boundary-add-every/-count. Neither mode enables "
            "fit-time pruning, relocation, adaptive counting, or merging."
        ),
    )
    capacity.add_argument(
        "--growth-waves",
        type=int,
        default=4,
        help=(
            "Residual mode only: number of planned addition waves. The per-wave count is "
            "ceil((candidate - start) / waves), capped at the candidate ceiling. Accepted but "
            "inactive in boundary mode."
        ),
    )
    capacity.add_argument(
        "--growth-every",
        type=int,
        default=1_000,
        help=(
            "Residual mode only: optimizer iterations between addition waves. Accepted but "
            "inactive in boundary mode."
        ),
    )
    capacity.add_argument(
        "--mask-boundary-add-every",
        type=int,
        help=(
            "Boundary mode only: optimizer iterations between tangent-aligned births. Boundary "
            "mode requires this interval and a positive --mask-boundary-add-count; leaving it "
            "unset disables boundary births in residual mode."
        ),
    )
    capacity.add_argument(
        "--mask-boundary-add-count",
        type=int,
        default=0,
        help=(
            "Boundary mode only: maximum rows born at each boundary event. The last event is "
            "trimmed to the candidate ceiling. Zero disables boundary births and is invalid when "
            "--growth-mode=boundary."
        ),
    )
    capacity.add_argument(
        "--mask-boundary-add-band",
        type=float,
        default=4.0,
        help=(
            "Boundary mode only: inside-mask signed-distance band, in pixels, whose residual "
            "peaks are eligible as birth sites."
        ),
    )
    capacity.add_argument(
        "--mask-boundary-add-spacing",
        type=float,
        default=5.0,
        help=(
            "Boundary mode only: approximate minimum pixel spacing between new sites. It also "
            "sets their initial along-boundary scale to 0.75 times this value before containment "
            "recertification."
        ),
    )

    optimization = parser.add_argument_group("optimization, logging, and stopping")
    optimization.add_argument(
        "--max-iters",
        type=int,
        default=50_000,
        help=(
            "Hard optimizer-iteration horizon for each view. A view may finish earlier under the "
            "PSNR plateau rule."
        ),
    )
    optimization.add_argument(
        "--log-every",
        type=int,
        default=100,
        help=(
            "Iterations between PSNR/loss history rows and plateau evaluations. Early-stop "
            "patience is counted in these logged evaluations, not optimizer steps."
        ),
    )
    optimization.add_argument(
        "--early-stop-patience",
        type=int,
        default=6,
        help=(
            "Number of eligible logged evaluations without more than "
            "--early-stop-min-delta PSNR improvement before stopping."
        ),
    )
    optimization.add_argument(
        "--early-stop-min-delta",
        type=float,
        default=0.005,
        help="Minimum PSNR improvement, in dB, that resets plateau patience.",
    )
    optimization.add_argument(
        "--early-stop-min-iters",
        type=int,
        default=6_500,
        help=(
            "Iteration before which plateau stopping is forbidden. Validation requires every "
            "planned growth event to finish before this point and this value to remain below "
            "--max-iters."
        ),
    )

    representation = parser.add_argument_group("initialization and rendering")
    representation.add_argument(
        "--strategy",
        default="quadtree_wse",
        help=(
            "StructSplat initialization strategy used to build the terminal candidate pool. The "
            "pool is always WSE-progressively ordered so --start-gaussians selects a nested "
            "prefix. Invalid strategy names are rejected by InitConfig."
        ),
    )
    representation.add_argument(
        "--renderer",
        default="cuda_tiled",
        help=(
            "StructSplat renderer used during fitting and central rescoring. Production normally "
            "uses cuda_tiled normalized-renderer semantics; changing it changes the fitted "
            "objective and is recorded in provenance."
        ),
    )
    representation.add_argument(
        "--init-scale-mult",
        type=float,
        default=0.35,
        help=(
            "Positive multiplier applied to initializer footprint scales. This affects the "
            "candidate pool in both growth modes."
        ),
    )
    representation.add_argument(
        "--split-scale",
        type=float,
        default=0.35,
        help=(
            "Residual mode only: positive scale multiplier for newly added residual-tensor rows. "
            "Accepted but inactive for boundary births, whose scales come from mask depth and "
            "--mask-boundary-add-spacing."
        ),
    )
    representation.add_argument(
        "--seed-offset",
        type=int,
        default=0,
        help=(
            "Integer added to each source view's deterministic seed. Change it to obtain a "
            "different but reproducible initialization; it is bound into the run contract."
        ),
    )

    containment = parser.add_argument_group("mask containment and boundary coverage")
    containment.add_argument(
        "--mask-margin",
        type=float,
        default=1.0,
        help=(
            "Positive pixel safety margin used to erode the admissible mean region and cap "
            "Gaussian support. Larger values improve separation from the mask edge but create a "
            "wider unreachable inner boundary band."
        ),
    )
    containment.add_argument(
        "--mask-cap-mode",
        choices=("isotropic", "anisotropic"),
        default="isotropic",
        help=(
            "Scale-containment certificate. 'isotropic' caps both axes by local mask depth; "
            "'anisotropic' retains the across-boundary cap while certifying longer tangent axes."
        ),
    )
    containment.add_argument(
        "--mask-cap-refresh-every",
        type=int,
        default=10,
        help=(
            "Iterations between full anisotropic cap recertifications. Projection and stored-cap "
            "clamping still run every iteration, and entry/topology/terminal states always receive "
            "a full recertification."
        ),
    )
    containment.add_argument(
        "--mask-undercoverage-weight",
        type=float,
        default=0.0,
        help=(
            "Non-negative weight of the inside-boundary undercoverage hinge. Zero disables the "
            "term, making --mask-undercoverage-band and --mask-undercoverage-tau inactive."
        ),
    )
    containment.add_argument(
        "--mask-undercoverage-band",
        type=float,
        default=3.0,
        help=(
            "Pixel width beyond --mask-margin supervised by the undercoverage hinge. Active only "
            "when --mask-undercoverage-weight is positive."
        ),
    )
    containment.add_argument(
        "--mask-undercoverage-tau",
        type=float,
        default=0.1,
        help=(
            "Raw Gaussian weight-sum target counted as covered by the boundary undercoverage "
            "hinge. Active only when --mask-undercoverage-weight is positive."
        ),
    )

    diagnostics = parser.add_argument_group("terminal diagnostics")
    diagnostics.add_argument(
        "--no-lpips",
        action="store_true",
        help=(
            "Skip the optional AlexNet LPIPS measurement on the final serialized field. This "
            "does not change fitting, because LPIPS is never part of the optimizer objective."
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=_HelpFormatter,
        epilog=(
            "Run '%(prog)s convert --help' for every fitting parameter, resolved defaults, "
            "lifecycle interactions, and current cleanup limitations."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True, title="commands")
    convert = subparsers.add_parser(
        "convert",
        help="discover masked views and fit resumable capped compact fields",
        description=(
            "Discover calibrated masked Janelle views, fit each view, and serialize a strictly "
            "verified realtime-gs compact dataset."
        ),
        formatter_class=_HelpFormatter,
        epilog=(
            "Lifecycle limitations:\n"
            "  * residual and boundary are birth modes; this wrapper does not expose fit-time\n"
            "    pruning, relocation, adaptive counting, or merging.\n"
            "  * export independently enforces the 168,000-byte complete-file cap by total\n"
            "    foreground support activity, so serialized rows can be fewer than candidate rows.\n"
            "  * options described as inactive are still recorded in the resolved RunSpec but do\n"
            "    not affect that lifecycle mode."
        ),
    )
    _add_common(convert)
    selection = convert.add_argument_group("dataset selection and dry inspection")
    selection.add_argument(
        "--max-views",
        type=int,
        help=(
            "Process only the first N discovered views. Useful for a pilot; partial runs do not "
            "write a complete frame manifest and skip whole-dataset verification."
        ),
    )
    selection.add_argument(
        "--inventory-only",
        action="store_true",
        help=(
            "Print the selected frame/view RGB and mask pairs, then exit without initializing, "
            "fitting, or writing outputs. Honors --max-views."
        ),
    )

    worker = subparsers.add_parser(
        "_worker",
        help="internal single-view subprocess used by convert",
        description=(
            "Internal single-view worker. Use the convert command for normal production runs."
        ),
        formatter_class=_HelpFormatter,
    )
    _add_common(worker)
    worker_inputs = worker.add_argument_group("internal view selection")
    worker_inputs.add_argument(
        "--frame",
        type=Path,
        required=True,
        help="Exact source frame directory containing the requested RGB and mask view.",
    )
    worker_inputs.add_argument(
        "--view-id",
        required=True,
        help="Camera/view identifier to fit inside --frame, for example C0001.",
    )
    return parser


# Help must remain available before importing CUDA, torch, or the sibling realtime-gs checkout.
# argparse exits here after printing any top-level or subcommand help request.
if __name__ == "__main__" and any(token in ("-h", "--help") for token in sys.argv[1:]):
    build_parser().parse_args()


def _prepend_source_roots(realtime_root: Path) -> None:
    roots = (REPOSITORY_ROOT / "src", realtime_root / "src")
    for root in reversed(roots):
        text = str(root.resolve())
        if text in sys.path:
            sys.path.remove(text)
        sys.path.insert(0, text)


_REALTIME_ROOT = Path(
    os.environ.get("STRUCTSPLAT_REALTIME_ROOT", str(DEFAULT_REALTIME_ROOT))
).resolve()
_prepend_source_roots(_REALTIME_ROOT)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from rtgs.core.observation2d import GaussianObservationField  # noqa: E402
from rtgs.data.compact_views import (  # noqa: E402
    COMPACT_VIEW_BYTE_CAP,
    CompactDataset,
    CompactView,
    file_sha256,
)
from rtgs.image2gs.structsplat_backend import field_to_observation  # noqa: E402
from structsplat import metrics as M  # noqa: E402
from structsplat.config import (  # noqa: E402
    FitConfig,
    InitConfig,
    StructureTensorConfig,
)
from structsplat.fit import fit  # noqa: E402
from structsplat.gaussians import GaussianField  # noqa: E402
from structsplat.init import build_masked_field  # noqa: E402
from structsplat.render import render_field  # noqa: E402


BYTE_CAP = 168_000
FOREGROUND_TARGETS = (22.0, 24.0, 26.0, 28.0, 30.0, 32.0, 35.0, 40.0)


def _load_converter(realtime_root: Path):
    path = realtime_root / "scripts/convert_datasets_to_gaussians2d.py"
    if not path.is_file():
        raise RuntimeError(f"realtime-gs converter not found: {path}")
    spec = importlib.util.spec_from_file_location("rtgs_janelle_converter", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load realtime-gs converter: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


CONVERTER = _load_converter(_REALTIME_ROOT)


@dataclass(frozen=True)
class RunSpec:
    candidate_gaussians: int = 5_750
    start_gaussians: int = 5_750
    growth_waves: int = 4
    growth_every: int = 1_000
    max_iters: int = 50_000
    log_every: int = 100
    early_stop_patience: int = 6
    early_stop_min_delta: float = 0.005
    early_stop_min_iters: int = 6_500
    strategy: str = "quadtree_wse"
    renderer: str = "cuda_tiled"
    mask_margin: float = 1.0
    mask_cap_mode: str = "isotropic"
    mask_cap_refresh_every: int = 10
    mask_undercoverage_weight: float = 0.0
    mask_undercoverage_band: float = 3.0
    mask_undercoverage_tau: float = 0.1
    mask_boundary_add_every: int | None = None
    mask_boundary_add_count: int = 0
    mask_boundary_add_band: float = 4.0
    mask_boundary_add_spacing: float = 5.0
    growth_mode: str = "residual"
    init_scale_mult: float = 0.35
    split_scale: float = 0.35
    seed_offset: int = 0
    compute_lpips: bool = True

    def validate(self) -> None:
        if self.start_gaussians < 16:
            raise ValueError("start_gaussians must be at least 16")
        if self.candidate_gaussians < self.start_gaussians:
            raise ValueError("candidate_gaussians must be >= start_gaussians")
        if self.growth_waves <= 0:
            raise ValueError("growth_waves must be positive")
        if self.growth_every <= 0 or self.log_every <= 0:
            raise ValueError("growth_every and log_every must be positive")
        if self.growth_mode not in ("residual", "boundary"):
            raise ValueError("growth_mode must be residual or boundary")
        add_total = self.candidate_gaussians - self.start_gaussians
        if self.growth_mode == "boundary":
            if add_total <= 0:
                raise ValueError("boundary growth requires start_gaussians < candidate_gaussians")
            if self.mask_boundary_add_every is None or self.mask_boundary_add_count <= 0:
                raise ValueError(
                    "boundary growth requires mask_boundary_add_every and a positive "
                    "mask_boundary_add_count"
                )
            events_needed = math.ceil(add_total / self.mask_boundary_add_count)
            final_growth = self.mask_boundary_add_every * events_needed - 1
        else:
            final_growth = self.growth_every * self.growth_waves - 1
        if final_growth >= self.early_stop_min_iters:
            raise ValueError("all growth must finish before early-stop eligibility")
        if self.early_stop_min_iters >= self.max_iters:
            raise ValueError("early_stop_min_iters must be below max_iters")
        if self.early_stop_patience <= 0 or self.early_stop_min_delta < 0:
            raise ValueError("invalid plateau criterion")
        if self.init_scale_mult <= 0.0 or self.split_scale <= 0.0:
            raise ValueError("initial and split scale multipliers must be positive")
        if BYTE_CAP != COMPACT_VIEW_BYTE_CAP:
            raise RuntimeError(
                f"byte-cap mismatch: task={BYTE_CAP}, realtime-gs={COMPACT_VIEW_BYTE_CAP}"
            )


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, value: object) -> None:
    _atomic_bytes(path, json.dumps(value, indent=2, allow_nan=False).encode() + b"\n")


def _atomic_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _git_record(root: Path) -> dict[str, Any]:
    def run(*args: str) -> bytes:
        return subprocess.run(
            ["git", "-C", str(root), *args],
            check=True,
            stdout=subprocess.PIPE,
        ).stdout

    try:
        commit = run("rev-parse", "HEAD").decode().strip()
        status = run("status", "--short")
        diff = run("diff", "--binary", "HEAD", "--", "src/structsplat", "scripts")
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "status_sha256": None, "relevant_diff_sha256": None}
    return {
        "commit": commit,
        "status_sha256": _sha256_bytes(status),
        "relevant_diff_sha256": _sha256_bytes(diff),
    }


def _source_record(source: Any) -> dict[str, Any]:
    return {
        "view_id": source.view_id,
        "rgb": {"path": str(source.rgb_path), "sha256": source.rgb_sha256},
        "mask": {"path": str(source.mask_path), "sha256": source.mask_sha256},
        "calibration": {
            "path": str(source.calibration_path),
            "sha256": source.calibration_sha256,
        },
        "seed": int(source.seed),
    }


def _fit_configs(
    source: Any,
    prepared: Any,
    spec: RunSpec,
) -> tuple[InitConfig, StructureTensorConfig, FitConfig, dict[float, float], float]:
    if prepared.alpha_crop is None:
        raise RuntimeError(f"{source.view_id} has no mask; hard-contained fitting requires one")
    mask_fraction = float(prepared.alpha_crop.float().mean())
    if not 0.0 < mask_fraction <= 1.0:
        raise RuntimeError(f"{source.view_id} has an empty or invalid cropped mask")
    # Outside-mask pixels are exactly zero in both target and render.  Therefore foreground
    # PSNR = full-matted-crop PSNR + 10*log10(foreground_fraction).
    psnr_shift = 10.0 * math.log10(mask_fraction)
    target_map = {
        float(foreground): float(foreground - psnr_shift) for foreground in FOREGROUND_TARGETS
    }
    add_total = spec.candidate_gaussians - spec.start_gaussians
    split_count = int(math.ceil(add_total / spec.growth_waves)) if add_total else 0
    boundary_growth = spec.growth_mode == "boundary"
    init_cfg = InitConfig(
        strategy=spec.strategy,
        # Build the terminal WSE pool so the start can use its progressive prefix with
        # terminal-density footprints.  A low-count field initialized at its own spacing creates
        # enormous supports and makes the diagnostic prefix much slower than the terminal fit.
        num_gaussians=spec.candidate_gaussians,
        seed=int(source.seed) + spec.seed_offset,
        sampling_mode="wse",
        wse_progressive_order=True,
        init_scale_mult=spec.init_scale_mult,
        scale_cap_mode="none",
        background_fraction=0.0,
        background_grid=0,
    )
    tensor_cfg = StructureTensorConfig()
    fit_cfg = FitConfig(
        iters=spec.max_iters,
        target_psnrs=list(target_map.values()),
        renderer=spec.renderer,
        render_chunk=512,
        pixel_loss="l1",
        ssim_weight=0.3,
        ssim_backend="builtin",
        loss_weighting="mask",
        mask_contain=True,
        mask_margin=spec.mask_margin,
        mask_coverage_weight=0.0,
        mask_cap_mode=spec.mask_cap_mode,
        mask_cap_refresh_every=spec.mask_cap_refresh_every,
        mask_undercoverage_weight=spec.mask_undercoverage_weight,
        mask_undercoverage_band=spec.mask_undercoverage_band,
        mask_undercoverage_tau=spec.mask_undercoverage_tau,
        mask_boundary_add_every=(spec.mask_boundary_add_every if boundary_growth else None),
        mask_boundary_add_count=(spec.mask_boundary_add_count if boundary_growth else 0),
        mask_boundary_add_band=spec.mask_boundary_add_band,
        mask_boundary_add_spacing=spec.mask_boundary_add_spacing,
        support_fade=True,
        log_every=spec.log_every,
        split_every=(
            spec.growth_every if add_total and spec.growth_mode == "residual" else None
        ),
        split_count=(split_count if spec.growth_mode == "residual" else 0),
        split_mode="residual_tensor_add",
        split_scale=spec.split_scale,
        max_gaussians=spec.candidate_gaussians,
        split_schedule_stops_at_max=True,
        checkpoint_policy="best_psnr_final_count",
        early_stop_patience=spec.early_stop_patience,
        early_stop_min_delta=spec.early_stop_min_delta,
        early_stop_min_iters=spec.early_stop_min_iters,
        compute_lpips=False,
    )
    return init_cfg, tensor_cfg, fit_cfg, target_map, psnr_shift


def _contract(
    source: Any,
    prepared: Any,
    spec: RunSpec,
    init_cfg: InitConfig,
    tensor_cfg: StructureTensorConfig,
    fit_cfg: FitConfig,
    target_map: dict[float, float],
) -> dict[str, Any]:
    binding = CONVERTER._implementation_binding()
    return {
        "schema": "structsplat.janelle_mask_contained_fit.v1",
        "source": _source_record(source),
        "provider": binding,
        "runner_sha256": file_sha256(Path(__file__).resolve()),
        "repository": _git_record(REPOSITORY_ROOT),
        "preprocessing": {
            "calibrated_undistortion": True,
            "crop": "rtgs._crop_to_mask_margin_fraction_0.05",
            "fit_window": list(prepared.fit_window),
            "fit_size": [int(prepared.rgb.shape[1]), int(prepared.rgb.shape[0])],
            "foreground_pixels": int(prepared.alpha_crop.bool().sum()),
            "foreground_fraction_of_fit_window": float(prepared.alpha_crop.float().mean()),
            "rgb_target": "binary-mask-matted calibrated RGB",
            "alpha": "authoritative undistorted binary mask, packed losslessly in rtgsv",
        },
        "initializer": asdict(init_cfg),
        "structure_tensor": asdict(tensor_cfg),
        "fit": asdict(fit_cfg),
        "foreground_to_matted_target_psnr": {str(key): value for key, value in target_map.items()},
        "serialization": {
            "format": "rtgs.compact_view.v1",
            "complete_file_byte_cap_decimal": BYTE_CAP,
            "candidate_gaussians": spec.candidate_gaussians,
            "backoff": "foreground-activity-ranked deterministic subset",
        },
        "capacity_schedule": {
            "pool_gaussians": spec.candidate_gaussians,
            "start_gaussians": spec.start_gaussians,
            "mode": (
                "direct_full_candidate"
                if spec.start_gaussians == spec.candidate_gaussians
                else f"progressive_{spec.growth_mode}_growth"
            ),
            "pool_ordering": "wse_progressive_order",
        },
    }


def _fit_digest(contract: dict[str, Any]) -> str:
    # The working-tree status is useful provenance, but unrelated IDE-file churn must not make a
    # completed view unresumable.  Fit semantics are already bound by the runner digest, provider
    # source digest, relevant source diff, source hashes, and every preprocessing/config value.
    semantic_contract = dict(contract)
    repository = dict(semantic_contract["repository"])
    repository.pop("status_sha256", None)
    semantic_contract["repository"] = repository
    return _sha256_bytes(_canonical_json(semantic_contract))


def _history_rows(history: dict[str, Any], psnr_shift: float) -> list[dict[str, Any]]:
    keys = (
        "iter",
        "elapsed",
        "n_gaussians",
        "loss",
        "psnr",
        "support_fade_alpha",
        "geometry_loss",
        "loss_target_full_weight",
    )
    count = len(history.get("iter", []))
    rows: list[dict[str, Any]] = []
    for index in range(count):
        row: dict[str, Any] = {}
        for key in keys:
            values = history.get(key, [])
            row[key] = values[index] if index < len(values) else None
        row["psnr_foreground"] = float(row["psnr"]) + psnr_shift
        row["psnr_matted_crop"] = row.pop("psnr")
        row["elapsed_seconds"] = row.pop("elapsed")
        rows.append(row)
    return rows


def _stage_plateaus(rows: list[dict[str, Any]], spec: RunSpec) -> list[dict[str, Any]]:
    by_count: list[list[dict[str, Any]]] = []
    for row in rows:
        if not by_count or by_count[-1][-1]["n_gaussians"] != row["n_gaussians"]:
            by_count.append([row])
        else:
            by_count[-1].append(row)
    events: list[dict[str, Any]] = []
    for stage in by_count:
        best = -math.inf
        stale = 0
        for row in stage:
            psnr = float(row["psnr_foreground"])
            if psnr > best + spec.early_stop_min_delta:
                best = psnr
                stale = 0
            else:
                stale += 1
            if stale >= spec.early_stop_patience:
                events.append(
                    {
                        "n_gaussians": int(row["n_gaussians"]),
                        "iteration": int(row["iter"]),
                        "elapsed_seconds": float(row["elapsed_seconds"]),
                        "best_foreground_psnr": best,
                        "criterion": (
                            f"{spec.early_stop_patience} logged evaluations without "
                            f">{spec.early_stop_min_delta:g} dB improvement"
                        ),
                    }
                )
                break
    return events


def _auc(rows: list[dict[str, Any]], field: str) -> float | None:
    if not rows:
        return None
    if len(rows) == 1:
        return float(rows[0][field])
    x = np.asarray([row["iter"] for row in rows], dtype=np.float64)
    y = np.asarray([row[field] for row in rows], dtype=np.float64)
    if x[-1] <= x[0]:
        return float(y[-1])
    return float(np.trapezoid(y, x) / (x[-1] - x[0]))


def _selected_field(observation: GaussianObservationField, device: torch.device) -> GaussianField:
    amplitudes = observation.amplitudes.to(device)
    opacities = None
    if not bool(torch.equal(amplitudes, torch.ones_like(amplitudes))):
        probabilities = amplitudes.clamp(1e-6, 1.0 - 1e-6)
        opacities = torch.logit(probabilities)
    return GaussianField(
        observation.local_means().to(device),
        observation.log_scales.to(device),
        observation.rotations.to(device),
        observation.colors.to(device),
        opacities,
        None,
        None if observation.color_grads is None else observation.color_grads.to(device),
        None,
        (None if observation.filter_variance is None else observation.filter_variance.to(device)),
    )


@torch.no_grad()
def _render_selected(
    observation: GaussianObservationField,
    device: torch.device,
    renderer: str,
) -> torch.Tensor:
    field = _selected_field(observation, device)
    _, _, width, height = observation.fit_window
    return render_field(
        field.means,
        field.conics(observation.aa_dilation),
        field.colors,
        field.radii(observation.sigma_cutoff, observation.aa_dilation),
        height,
        width,
        chunk=512,
        mode=renderer,
        opacities=field.opacity_values(),
        scales=field.effective_scales(observation.aa_dilation),
        rotations=field.rotations,
        support_fade=observation.support_fade_alpha > 0.0,
        support_fade_alpha=observation.support_fade_alpha,
        sigma_cutoff=observation.sigma_cutoff,
        color_grads=field.color_grads,
    )


def _psnr_from_mse(value: torch.Tensor) -> float:
    return float(-10.0 * torch.log10(value.clamp_min(1e-12)))


@torch.no_grad()
def _count_frontier(
    observation: GaussianObservationField,
    activity: np.ndarray,
    selected_count: int,
    prepared: Any,
    device: torch.device,
    renderer: str,
) -> dict[str, Any]:
    """Measure deterministic activity-ranked subsets without claiming independent convergence."""
    candidate_count = observation.n
    counts = {
        max(64, (int(candidate_count * fraction) // 32) * 32) for fraction in (0.25, 0.5, 0.75, 0.9)
    }
    counts.update((selected_count, candidate_count))
    counts = {count for count in counts if 0 < count <= candidate_count}
    order = CONVERTER._rank_components(activity)
    target = (prepared.rgb * prepared.alpha_crop[..., None].to(prepared.rgb)).to(device)
    mask = prepared.alpha_crop.to(device).bool()
    rows: list[dict[str, Any]] = []
    for count in sorted(counts):
        indices = np.sort(order[:count])
        subset = (
            observation
            if count == candidate_count
            else CONVERTER._subset_observation(observation, indices)
        )
        render = _render_selected(subset, device, renderer)
        error = render - target
        foreground_mse = error[mask].square().mean()
        matted_crop_mse = error.square().mean()
        outside = render[~mask].abs()
        rows.append(
            {
                "gaussians": count,
                "foreground_psnr_raw": _psnr_from_mse(foreground_mse),
                "matted_crop_psnr_raw": _psnr_from_mse(matted_crop_mse),
                "outside_mask_max_abs": (float(outside.max()) if outside.numel() else 0.0),
            }
        )
    selected = next(row for row in rows if row["gaussians"] == selected_count)
    tolerance_db = 0.10
    near = [
        row
        for row in rows
        if row["gaussians"] <= selected_count
        and row["foreground_psnr_raw"] >= selected["foreground_psnr_raw"] - tolerance_db
    ]
    earliest = min(near, key=lambda row: row["gaussians"])
    return {
        "interpretation": (
            "post-fit activity-ranked subset diagnostic; this is a count-saturation proxy, "
            "not an independently converged lower-count fit"
        ),
        "near_equivalence_tolerance_db": tolerance_db,
        "serialized_count_reference": selected_count,
        "earliest_near_equivalent": earliest,
        "substantially_smaller_near_equivalent": earliest["gaussians"] <= 0.8 * selected_count,
        "frontier": rows,
    }


@torch.no_grad()
def _final_metrics(
    observation: GaussianObservationField,
    prepared: Any,
    device: torch.device,
    spec: RunSpec,
) -> tuple[dict[str, Any], torch.Tensor]:
    started = time.perf_counter()
    render = _render_selected(observation, device, spec.renderer)
    target = (prepared.rgb * prepared.alpha_crop[..., None].to(prepared.rgb)).to(device)
    mask = prepared.alpha_crop.to(device).bool()
    error = render - target
    foreground_mse = error[mask].square().mean()
    foreground_mae = error[mask].abs().mean()
    clamped = render.clamp(0, 1)
    clamped_error = clamped - target
    clamped_foreground_mse = clamped_error[mask].square().mean()
    outside_values = render[~mask].abs()
    outside_max = float(outside_values.max()) if outside_values.numel() else 0.0
    outside_mean = float(outside_values.mean()) if outside_values.numel() else 0.0

    outside = (~mask).to(render).view(1, 1, *mask.shape)
    near_outside = F.max_pool2d(outside, kernel_size=5, stride=1, padding=2)[0, 0] > 0
    boundary = mask & near_outside
    boundary_mse = error[boundary].square().mean() if bool(boundary.any()) else None

    masked_similarity = CONVERTER.MaskedSSIM(target, mask.to(target))
    foreground_ssim = float(masked_similarity(render))
    foreground_ssim_clamped = float(masked_similarity(clamped))
    lpips = None
    lpips_error = None
    if spec.compute_lpips:
        try:
            lpips = M.LPIPS.distance(clamped, target)
        except Exception as error_value:  # optional diagnostic must not invalidate the field
            lpips_error = f"{type(error_value).__name__}: {error_value}"
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    metrics = {
        "foreground_psnr_raw": _psnr_from_mse(foreground_mse),
        "foreground_psnr_clamped": _psnr_from_mse(clamped_foreground_mse),
        "foreground_mse_raw": float(foreground_mse),
        "foreground_mae_raw": float(foreground_mae),
        "foreground_ssim_raw": foreground_ssim,
        "foreground_ssim_clamped": foreground_ssim_clamped,
        "matted_crop_psnr_raw": M.psnr(render, target),
        "matted_crop_psnr_clamped": M.psnr(clamped, target),
        "matted_crop_ssim_raw": float(M.ssim(render, target)),
        "matted_crop_ssim_clamped": float(M.ssim(clamped, target)),
        "matted_crop_ms_ssim_raw": M.ms_ssim(render, target),
        "matted_crop_ms_ssim_clamped": M.ms_ssim(clamped, target),
        "lpips_alex_clamped_matted_crop": lpips,
        "lpips_error": lpips_error,
        "boundary_band_definition": "inside pixels within a 5x5 square dilation of outside",
        "boundary_band_pixels": int(boundary.sum()),
        "boundary_psnr_raw": (None if boundary_mse is None else _psnr_from_mse(boundary_mse)),
        "outside_mask_max_abs": outside_max,
        "outside_mask_mean_abs": outside_mean,
        "outside_mask_exact_zero": outside_max == 0.0,
        "metric_seconds": time.perf_counter() - started,
    }
    return metrics, render


def _target_hits(
    output: dict[str, Any],
    target_map: dict[float, float],
) -> dict[str, int | None]:
    recorded = output.get("iters_to_targets", {})
    return {
        str(foreground): recorded.get(str(float(matted)))
        for foreground, matted in target_map.items()
    }


def _sidecar_paths(source: Any) -> tuple[Path, Path, Path]:
    output_dir = source.frame / "gaussians2d"
    return (
        output_dir / f"{source.view_id}.rtgsv",
        output_dir / "fitting" / f"{source.view_id}.json",
        output_dir / "fitting" / f"{source.view_id}_history.csv",
    )


def _existing_valid(source: Any, digest: str) -> CompactView | None:
    output_path, metrics_path, history_path = _sidecar_paths(source)
    if not output_path.is_file() or not metrics_path.is_file() or not history_path.is_file():
        return None
    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        view = CompactView.load(output_path)
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    if not (
        metrics.get("status") == "ok"
        and metrics.get("fit_config_digest") == digest
        and metrics.get("source") == _source_record(source)
        and metrics.get("output", {}).get("sha256") == file_sha256(output_path)
        and metrics.get("output", {}).get("bytes") == output_path.stat().st_size
        and metrics.get("history", {}).get("rows", 0) > 0
        and metrics.get("history", {}).get("bytes") == history_path.stat().st_size
        and metrics.get("history", {}).get("sha256") == file_sha256(history_path)
        and output_path.stat().st_size <= BYTE_CAP
        and metrics_path.stat().st_size <= BYTE_CAP
        and history_path.stat().st_size <= BYTE_CAP
        and view.observation.fit_config_digest == digest
        and view.source["rgb"]["sha256"] == source.rgb_sha256
        and view.source["mask"]["sha256"] == source.mask_sha256
    ):
        return None
    return view


def fit_one(source: Any, spec: RunSpec, device_text: str) -> dict[str, Any]:
    spec.validate()
    if source.mask_path is None:
        raise RuntimeError(f"{source.view_id}: source has no mask")
    total_started = time.perf_counter()
    prepare_started = time.perf_counter()
    prepared = CONVERTER.prepare_source_view(source)
    prepare_seconds = time.perf_counter() - prepare_started
    init_cfg, tensor_cfg, fit_cfg, target_map, psnr_shift = _fit_configs(source, prepared, spec)
    contract = _contract(source, prepared, spec, init_cfg, tensor_cfg, fit_cfg, target_map)
    digest = _fit_digest(contract)
    existing = _existing_valid(source, digest)
    if existing is not None:
        print(
            f"[{source.view_id}] verified existing {existing.path} "
            f"({existing.bytes} bytes, {existing.observation.n} Gaussians); skipping",
            flush=True,
        )
        return json.loads(_sidecar_paths(source)[1].read_text(encoding="utf-8"))
    output_path, metrics_path, history_path = _sidecar_paths(source)
    if output_path.exists() or metrics_path.exists() or history_path.exists():
        raise RuntimeError(
            f"{source.view_id}: partial or mismatched outputs exist; inspect before replacing: "
            f"{output_path.parent}"
        )

    device = torch.device(device_text)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("the Janelle production fit requires a CUDA device")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    mask_cpu = prepared.alpha_crop.bool().contiguous()
    target_cpu = (prepared.rgb * mask_cpu[..., None].to(prepared.rgb)).contiguous()
    image_np = target_cpu.numpy().astype(np.float32, copy=False)
    mask_np = mask_cpu.numpy()

    init_started = time.perf_counter()
    # contain=False avoids constructing the expensive SDF twice before fit(); masked density and
    # color dilation are still applied, and fit() hard-contains the field before its first render.
    field_pool = build_masked_field(
        image_np,
        mask_np,
        init_cfg,
        tensor_cfg,
        device=str(device),
        sigma_cutoff=fit_cfg.sigma_cutoff,
        mask_margin=fit_cfg.mask_margin,
        contain=False,
    )
    if field_pool.n != spec.candidate_gaussians:
        raise RuntimeError(
            f"{source.view_id}: initializer returned {field_pool.n} rows, "
            f"expected {spec.candidate_gaussians}"
        )
    field = field_pool.subset(slice(0, spec.start_gaussians))
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    init_seconds = time.perf_counter() - init_started
    print(
        f"[{source.view_id}] prepared {prepared.fit_window[2]}x{prepared.fit_window[3]} "
        f"crop in {prepare_seconds:.1f}s; initialized {field.n} in {init_seconds:.1f}s",
        flush=True,
    )
    target = torch.as_tensor(image_np, device=device)
    torch.cuda.synchronize(device)
    fit_call_started = time.perf_counter()
    output = fit(field, target, fit_cfg, mask=mask_np, verbose=True)
    torch.cuda.synchronize(device)
    fit_call_seconds = time.perf_counter() - fit_call_started
    if output["n_gaussians"] != spec.candidate_gaussians:
        raise RuntimeError(
            f"{source.view_id}: fit ended at {output['n_gaussians']} Gaussians, "
            f"expected candidate cap {spec.candidate_gaussians}"
        )

    history_rows = _history_rows(output["history"], psnr_shift)
    plateaus = _stage_plateaus(history_rows, spec)
    activity_started = time.perf_counter()
    activity = (
        CONVERTER._component_activity(
            output["field"],
            mask_cpu.to(device),
            height=prepared.fit_window[3],
            width=prepared.fit_window[2],
        )
        .detach()
        .cpu()
        .numpy()
    )
    binding = CONVERTER._implementation_binding()
    observation = field_to_observation(
        output["field"],
        canvas_size=(prepared.camera.height, prepared.camera.width),
        fit_window=prepared.fit_window,
        blend_mode="normalized",
        sigma_cutoff=fit_cfg.sigma_cutoff,
        support_fade_alpha=1.0,
        aa_dilation=fit_cfg.aa_dilation,
        view_id=source.view_id,
        n_init=spec.start_gaussians,
        producer_version=binding["structsplat_version"],
        producer_source_digest=binding["structsplat_source_sha256"],
        fit_config_digest=digest,
    ).to("cpu")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    CONVERTER._save_with_backoff(
        output_path,
        prepared,
        observation,
        np.asarray(activity, dtype=np.float64),
    )
    serialization_seconds = time.perf_counter() - activity_started
    strict_view = CompactView.load(output_path)
    if strict_view.bytes > BYTE_CAP or strict_view.observation.n > spec.candidate_gaussians:
        raise RuntimeError(f"{source.view_id}: strict compact-view cap/count validation failed")
    frontier_started = time.perf_counter()
    count_frontier = _count_frontier(
        observation,
        np.asarray(activity, dtype=np.float64),
        strict_view.observation.n,
        prepared,
        device,
        spec.renderer,
    )
    if any(row["outside_mask_max_abs"] != 0.0 for row in count_frontier["frontier"]):
        raise RuntimeError(f"{source.view_id}: a count-frontier subset leaked outside the mask")
    count_frontier_seconds = time.perf_counter() - frontier_started
    final_metrics, _ = _final_metrics(strict_view.observation, prepared, device, spec)
    if not final_metrics["outside_mask_exact_zero"]:
        raise RuntimeError(
            f"{source.view_id}: selected compact field is nonzero outside the mask "
            f"(max={final_metrics['outside_mask_max_abs']})"
        )

    selected_count = strict_view.observation.n
    output_bytes = int(output_path.stat().st_size)
    canvas_pixels = prepared.camera.width * prepared.camera.height
    fit_pixels = prepared.fit_window[2] * prepared.fit_window[3]
    early = [event for event in plateaus if event["n_gaussians"] < selected_count]
    convergence_reason = "psnr_plateau" if output["stopped_early"] else "max_horizon"
    result = {
        "status": "ok",
        "source": _source_record(source),
        "fit_config_digest": digest,
        "contract": contract,
        "output": {
            "path": str(output_path),
            "bytes": output_bytes,
            "byte_cap_decimal": BYTE_CAP,
            "unused_bytes": BYTE_CAP - output_bytes,
            "actual_complete_file_bpp_native_canvas": 8.0 * output_bytes / canvas_pixels,
            "actual_complete_file_bpp_fit_window": 8.0 * output_bytes / fit_pixels,
            "sha256": file_sha256(output_path),
            "candidate_gaussians": spec.candidate_gaussians,
            "serialized_gaussians": selected_count,
            "serialization_pruned_gaussians": spec.candidate_gaussians - selected_count,
            "has_exact_alpha": strict_view.alpha is not None,
        },
        "fit": {
            "start_gaussians": spec.start_gaussians,
            "candidate_gaussians": spec.candidate_gaussians,
            "iterations_run": int(output["iterations_run"]),
            "stopped_early": bool(output["stopped_early"]),
            "stopped_at": output["stopped_at"],
            "convergence_stop_reason": convergence_reason,
            "checkpoint_policy": output["checkpoint_policy"],
            "selected_iter": int(output["selected_iter"]),
            "selected_from_checkpoint": bool(output["selected_from_checkpoint"]),
            "trajectory_terminal_psnr_matted_crop": output["trajectory_terminal_psnr"],
            "selected_psnr_matted_crop": output["selected_psnr"],
            "target_iters_foreground_psnr": _target_hits(output, target_map),
            "early_count_plateaus": plateaus,
            "converged_earlier_below_serialized_count": bool(early),
            "first_early_convergence": early[0] if early else None,
            "split_events": output["history"]["split_events"],
            "boundary_add_events": output["history"]["boundary_add_events"],
            "checkpoint_history": output["checkpoint_history"],
            "auc_foreground_psnr_observed": _auc(history_rows, "psnr_foreground"),
            "auc_matted_crop_psnr_observed": _auc(history_rows, "psnr_matted_crop"),
            "count_saturation": count_frontier,
        },
        "metrics": final_metrics,
        "timing": {
            "prepare_seconds": prepare_seconds,
            "initialization_seconds": init_seconds,
            # StructSplat's native timer deliberately starts after mask-SDF construction and
            # stops before its terminal metric pass.  Keep it for benchmark comparability, and
            # also retain the complete fit() call wall time requested for this conversion.
            "fit_seconds": float(output["fit_seconds"]),
            "fit_call_wall_seconds": fit_call_seconds,
            "fit_setup_and_terminal_eval_seconds": (
                fit_call_seconds - float(output["fit_seconds"])
            ),
            "activity_and_serialization_seconds": serialization_seconds,
            "count_frontier_seconds": count_frontier_seconds,
            "metric_seconds": final_metrics["metric_seconds"],
            "total_seconds": time.perf_counter() - total_started,
        },
        "resources": {
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(device),
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
            "torch_version": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
            "numpy_version": np.__version__,
        },
    }
    _atomic_csv(history_path, history_rows, list(history_rows[0]))
    result["history"] = {
        "path": str(history_path),
        "rows": len(history_rows),
        "bytes": history_path.stat().st_size,
        "sha256": file_sha256(history_path),
    }
    _atomic_json(metrics_path, result)
    for sidecar in (metrics_path, history_path):
        if sidecar.stat().st_size > BYTE_CAP:
            raise RuntimeError(f"metrics sidecar exceeds 168,000 bytes: {sidecar}")
    print(
        f"[{source.view_id}] converged={convergence_reason} at {output['iterations_run']} "
        f"iters; serialized {selected_count} Gaussians / {strict_view.bytes} bytes; "
        f"foreground PSNR={final_metrics['foreground_psnr_clamped']:.3f} dB",
        flush=True,
    )
    return result


def _source_for_worker(args: argparse.Namespace) -> Any:
    frames = CONVERTER.discover_source_frames(args.dataset_root.resolve())
    matches = [
        view
        for frame in frames
        for view in frame.views
        if frame.path.resolve() == args.frame.resolve() and view.view_id == args.view_id.upper()
    ]
    if len(matches) != 1:
        raise RuntimeError(f"worker source is not unique: frame={args.frame}, view={args.view_id}")
    return matches[0]


def _spec_from_args(args: argparse.Namespace) -> RunSpec:
    return RunSpec(
        candidate_gaussians=args.candidate_gaussians,
        start_gaussians=args.start_gaussians,
        growth_waves=args.growth_waves,
        growth_every=args.growth_every,
        max_iters=args.max_iters,
        log_every=args.log_every,
        early_stop_patience=args.early_stop_patience,
        early_stop_min_delta=args.early_stop_min_delta,
        early_stop_min_iters=args.early_stop_min_iters,
        strategy=args.strategy,
        renderer=args.renderer,
        mask_margin=args.mask_margin,
        mask_cap_mode=args.mask_cap_mode,
        mask_cap_refresh_every=args.mask_cap_refresh_every,
        mask_undercoverage_weight=args.mask_undercoverage_weight,
        mask_undercoverage_band=args.mask_undercoverage_band,
        mask_undercoverage_tau=args.mask_undercoverage_tau,
        mask_boundary_add_every=args.mask_boundary_add_every,
        mask_boundary_add_count=args.mask_boundary_add_count,
        mask_boundary_add_band=args.mask_boundary_add_band,
        mask_boundary_add_spacing=args.mask_boundary_add_spacing,
        growth_mode=args.growth_mode,
        init_scale_mult=args.init_scale_mult,
        split_scale=args.split_scale,
        seed_offset=args.seed_offset,
        compute_lpips=not args.no_lpips,
    )


def _worker_command(source: Any, args: argparse.Namespace) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "_worker",
        "--dataset-root",
        str(args.dataset_root.resolve()),
        "--realtime-root",
        str(args.realtime_root.resolve()),
        "--frame",
        str(source.frame),
        "--view-id",
        source.view_id,
        "--device",
        args.device,
        "--candidate-gaussians",
        str(args.candidate_gaussians),
        "--start-gaussians",
        str(args.start_gaussians),
        "--growth-waves",
        str(args.growth_waves),
        "--growth-every",
        str(args.growth_every),
        "--max-iters",
        str(args.max_iters),
        "--log-every",
        str(args.log_every),
        "--early-stop-patience",
        str(args.early_stop_patience),
        "--early-stop-min-delta",
        str(args.early_stop_min_delta),
        "--early-stop-min-iters",
        str(args.early_stop_min_iters),
        "--strategy",
        args.strategy,
        "--renderer",
        args.renderer,
        "--mask-margin",
        str(args.mask_margin),
        "--mask-cap-mode",
        args.mask_cap_mode,
        "--mask-cap-refresh-every",
        str(args.mask_cap_refresh_every),
        "--mask-undercoverage-weight",
        str(args.mask_undercoverage_weight),
        "--mask-undercoverage-band",
        str(args.mask_undercoverage_band),
        "--mask-undercoverage-tau",
        str(args.mask_undercoverage_tau),
        "--mask-boundary-add-count",
        str(args.mask_boundary_add_count),
        "--mask-boundary-add-band",
        str(args.mask_boundary_add_band),
        "--mask-boundary-add-spacing",
        str(args.mask_boundary_add_spacing),
        "--growth-mode",
        args.growth_mode,
        *(
            ["--mask-boundary-add-every", str(args.mask_boundary_add_every)]
            if args.mask_boundary_add_every is not None
            else []
        ),
        "--init-scale-mult",
        str(args.init_scale_mult),
        "--split-scale",
        str(args.split_scale),
        "--seed-offset",
        str(args.seed_offset),
        *(["--no-lpips"] if args.no_lpips else []),
    ]


def _read_result(source: Any) -> dict[str, Any]:
    path = _sidecar_paths(source)[1]
    return json.loads(path.read_text(encoding="utf-8"))


def _write_frame_outputs(frame: Any, spec: RunSpec) -> None:
    CONVERTER._write_frame_manifest(frame)
    results = [_read_result(source) for source in frame.views]
    summary_rows = []
    for result in results:
        summary_rows.append(
            {
                "view_id": result["source"]["view_id"],
                "bytes": result["output"]["bytes"],
                "unused_bytes": result["output"]["unused_bytes"],
                "serialized_gaussians": result["output"]["serialized_gaussians"],
                "actual_complete_file_bpp_native_canvas": result["output"][
                    "actual_complete_file_bpp_native_canvas"
                ],
                "iterations_run": result["fit"]["iterations_run"],
                "convergence_stop_reason": result["fit"]["convergence_stop_reason"],
                "converged_earlier_below_serialized_count": result["fit"][
                    "converged_earlier_below_serialized_count"
                ],
                "first_early_convergence_gaussians": (
                    None
                    if result["fit"]["first_early_convergence"] is None
                    else result["fit"]["first_early_convergence"]["n_gaussians"]
                ),
                "earliest_near_equivalent_subset_gaussians": result["fit"]["count_saturation"][
                    "earliest_near_equivalent"
                ]["gaussians"],
                "substantially_smaller_near_equivalent_subset": result["fit"]["count_saturation"][
                    "substantially_smaller_near_equivalent"
                ],
                "foreground_psnr_clamped": result["metrics"]["foreground_psnr_clamped"],
                "boundary_psnr_raw": result["metrics"]["boundary_psnr_raw"],
                "foreground_ssim_clamped": result["metrics"]["foreground_ssim_clamped"],
                "matted_crop_ms_ssim_clamped": result["metrics"]["matted_crop_ms_ssim_clamped"],
                "lpips_alex_clamped_matted_crop": result["metrics"][
                    "lpips_alex_clamped_matted_crop"
                ],
                "fit_seconds": result["timing"]["fit_seconds"],
                "fit_call_wall_seconds": result["timing"]["fit_call_wall_seconds"],
                "total_seconds": result["timing"]["total_seconds"],
                "outside_mask_exact_zero": result["metrics"]["outside_mask_exact_zero"],
                "sha256": result["output"]["sha256"],
            }
        )
    output_dir = frame.path / "gaussians2d"
    _atomic_csv(output_dir / "fitting_summary.csv", summary_rows, list(summary_rows[0]))
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    run_record = {
        "schema": "structsplat.janelle_mask_contained_frame_run.v1",
        "frame": str(frame.path),
        "spec": asdict(spec),
        "views": len(results),
        "all_converged": all(
            result["fit"]["convergence_stop_reason"] == "psnr_plateau" for result in results
        ),
        "all_within_byte_cap": all(result["output"]["bytes"] <= BYTE_CAP for result in results),
        "all_exact_zero_outside": all(
            result["metrics"]["outside_mask_exact_zero"] for result in results
        ),
        "manifest_sha256": file_sha256(output_dir / "manifest.json"),
        "dataset_semantic_digest": manifest["semantic_digest"],
    }
    _atomic_json(output_dir / "run_config.json", run_record)


def _verify(frames: list[Any], spec: RunSpec) -> None:
    for frame in frames:
        dataset = CompactDataset.load(frame.path / "gaussians2d")
        expected = [source.view_id for source in frame.views]
        if [view.view_id for view in dataset.views] != expected:
            raise RuntimeError(f"manifest view order mismatch: {frame.path}")
        for source, view in zip(frame.views, dataset.views, strict=True):
            result = _read_result(source)
            path, metrics_path, history_path = _sidecar_paths(source)
            if not (
                view.bytes <= BYTE_CAP
                and result["status"] == "ok"
                and result["output"]["bytes"] == view.bytes
                and result["output"]["sha256"] == file_sha256(path)
                and view.observation.fit_config_digest == result["fit_config_digest"]
                and view.source["rgb"]["sha256"] == source.rgb_sha256
                and view.source["mask"]["sha256"] == source.mask_sha256
                and result["metrics"]["outside_mask_exact_zero"] is True
                and metrics_path.stat().st_size <= BYTE_CAP
                and history_path.stat().st_size <= BYTE_CAP
                and result["history"]["bytes"] == history_path.stat().st_size
                and result["history"]["sha256"] == file_sha256(history_path)
            ):
                raise RuntimeError(f"strict verification failed: {path}")
        run_config = json.loads(
            (frame.path / "gaussians2d/run_config.json").read_text(encoding="utf-8")
        )
        if run_config["spec"] != asdict(spec):
            raise RuntimeError(f"run config mismatch: {frame.path}")


def run(args: argparse.Namespace) -> None:
    spec = _spec_from_args(args)
    spec.validate()
    frames = CONVERTER.discover_source_frames(args.dataset_root.resolve())
    frames = [frame for frame in frames if (frame.path / "mask").is_dir()]
    sources = [source for frame in frames for source in frame.views]
    if not frames or any(source.mask_path is None for source in sources):
        raise RuntimeError("dataset selection must contain only frames with complete masks")
    if args.max_views is not None:
        sources = sources[: args.max_views]
    print(
        f"discovered {sum(len(frame.views) for frame in frames)} masked views across "
        f"{len(frames)} frames; processing {len(sources)}",
        flush=True,
    )
    if args.inventory_only:
        for source in sources:
            print(
                f"{source.frame}/{source.view_id}: {source.rgb_path.name} + {source.mask_path.name}"
            )
        return
    worker_environment = os.environ.copy()
    worker_environment["STRUCTSPLAT_REALTIME_ROOT"] = str(args.realtime_root.resolve())
    worker_environment["PYTHONPATH"] = os.pathsep.join(
        [
            str((REPOSITORY_ROOT / "src").resolve()),
            str((args.realtime_root / "src").resolve()),
            worker_environment.get("PYTHONPATH", ""),
        ]
    ).rstrip(os.pathsep)
    for index, source in enumerate(sources, start=1):
        print(
            f"[{index}/{len(sources)}] "
            f"{source.frame.relative_to(args.dataset_root.resolve())}/{source.view_id}",
            flush=True,
        )
        subprocess.run(
            _worker_command(source, args),
            check=True,
            env=worker_environment,
        )
    processed = {(source.frame, source.view_id) for source in sources}
    for frame in frames:
        if all(
            (frame.path, source.view_id) in processed or _sidecar_paths(source)[0].is_file()
            for source in frame.views
        ):
            _write_frame_outputs(frame, spec)
    if args.max_views is None:
        _verify(frames, spec)
        print(f"strictly verified {sum(len(frame.views) for frame in frames)} compact views")
    else:
        print("partial run complete; frame manifests are written only when complete")


def main() -> None:
    args = build_parser().parse_args()
    if args.realtime_root.resolve() != _REALTIME_ROOT:
        raise RuntimeError(
            "--realtime-root must match STRUCTSPLAT_REALTIME_ROOT because imports are bound "
            "at process start"
        )
    if args.command == "_worker":
        source = _source_for_worker(args)
        fit_one(source, _spec_from_args(args), args.device)
        return
    if args.max_views is not None and args.max_views <= 0:
        raise ValueError("--max-views must be positive")
    run(args)


if __name__ == "__main__":
    main()
