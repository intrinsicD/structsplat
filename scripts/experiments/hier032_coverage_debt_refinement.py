#!/usr/bin/env python3
"""Coverage-debt refinement of the frozen HIER-031 exact-7k field (HIER-032).

Formal development command (run from the clean protocol commit):

PYTHONPATH=src python scripts/experiments/hier032_coverage_debt_refinement.py \
  /home/alex/Dropbox/Work/Janelle/2025_03_07_stage_with_fabric/frame_00008/rgb/C0001.jpg \
  /home/alex/Dropbox/Work/Janelle/2025_03_07_stage_with_fabric/frame_00008/mask/mask_C0001.png \
  /home/alex/Documents/structsplat/results/hier032_janelle_c0001_s1200_coverage_debt_s0_development_2026-08-12 \
  --base-bundle /home/alex/Documents/structsplat/results/hier031_janelle_c0001_s1200_exact7k_boundary_detail_s0_diagnostic_2026-08-12 \
  --max-side 1200 --seed 0 --device cuda --lpips

The driver is task-scoped and default-off.  It never changes the Gaussian count, relaxes the
mask, masks a render after the fact, or optimizes successor geometry after placement.
"""

from __future__ import annotations

import argparse
from collections import deque
import csv
from dataclasses import dataclass, replace
import hashlib
from html import escape
import json
import math
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
for candidate_root in (ROOT, ROOT / "src"):
    if str(candidate_root) not in sys.path:
        sys.path.insert(0, str(candidate_root))

from scripts.experiments import hier022_additive_continuation as h22  # noqa: E402
from scripts.experiments import hier029_janelle_mask_diagnostic as h29  # noqa: E402
from scripts.experiments import hier030_janelle_7k_contained_diagnostic as h30  # noqa: E402
from scripts.experiments import hier031_exact7k_masked_boundary_detail as h31  # noqa: E402
from structsplat import mask as mask_geometry  # noqa: E402
from structsplat.cli import save_error_heatmap, save_image  # noqa: E402
from structsplat.detail_pursuit import gaussian_blur  # noqa: E402
from structsplat.endpoint_appearance_projection import project_additive_endpoint  # noqa: E402
from structsplat.gaussians import GaussianField  # noqa: E402


REPORT_SCHEMA = "structsplat.hier032_coverage_debt_refinement.diagnostic.v1"
SOURCE_SHA256 = h31.SOURCE_SHA256
MASK_SHA256 = h31.MASK_SHA256
NATIVE_SHAPE = h31.NATIVE_SHAPE
EVALUATION_SHAPE = h31.EVALUATION_SHAPE
CAPACITY = 7_000
COVERAGE_THRESHOLD = 0.05
MASK_MARGIN = h31.MASK_MARGIN
SIGMA_CUTOFF = h31.SIGMA_CUTOFF
MICRO_SCALE = h31.MICRO_SCALE
ORDINARY_MIN_SCALE = h31.ORDINARY_MIN_SCALE
MAX_WAVES = 4
MAX_SUCCESSOR_PLACEMENTS = 1_536
DETAIL_ROWS = 128
DETAIL_NMS_RADIUS = 2
DETAIL_SDF_MAX = 4.0
DETAIL_BLUR_SIGMA = 1.5
CONTAINMENT_LIMIT = 1e-7
INTERIOR_PSNR_FLOOR_DB = 35.2631
HAIR_CROP_BOUNDS = (54, 434, 839, 530)
BOUNDARY_CROP_BOUNDS = (478, 569, 574, 665)
FOUR_ARRAY_KEYS = frozenset(("means", "log_scales", "rotations", "colors"))
EXPECTED_GPU_NAME = "NVIDIA GeForce RTX 3050"
EXPECTED_SOURCE_BRANCH = "agent/hier032-coverage-debt-refinement"
COEFFICIENT_ABS_LIMIT = 16.0
RENDER_PARITY_LIMIT = 2e-5
BASE_BUNDLE = (
    ROOT
    / "results/hier031_janelle_c0001_s1200_exact7k_boundary_detail_s0_diagnostic_2026-08-12"
)
BASE_FIELD_REL = Path("artifacts/deep_only_terminal_closure_n7000/field.gaussian.npz")
BASE_FIELD_SHA256 = "a0a080ccbd255ce51f11489cd504956a1c5181a495bbca2b4bf74ecb0995c1db"
BASE_DECISION_SHA256 = "52016532a23290b12c45b2b9a75c2fc7e3fb0d3001cd19924f30a1a52eb8e2a8"
ARMS = (
    "hier031_selected_control_n7000",
    "fallback_per_weak_pixel_n7000",
    "component_set_cover_n7000",
    "component_set_cover_contribution_merge_n7000",
    "coverage_then_boundary_highpass_n7000",
)

PROTOCOL = {
    "task": "HIER-032",
    "schema": REPORT_SCHEMA,
    "source_identity": {
        "source_sha256": SOURCE_SHA256,
        "mask_sha256": MASK_SHA256,
        "native_shape": list(NATIVE_SHAPE),
        "evaluation_shape": list(EVALUATION_SHAPE),
        "max_side": 1200,
        "rgb_resampling": "pillow_lanczos",
        "mask_resampling": "pillow_nearest",
        "seed": 0,
        "base_field_relative_path": str(BASE_FIELD_REL),
        "base_field_sha256": BASE_FIELD_SHA256,
        "base_decision_sha256": BASE_DECISION_SHA256,
        "base_selected_arm": "deep_only_terminal_closure_n7000",
        "clean_source_required": True,
        "source_branch": EXPECTED_SOURCE_BRANCH,
    },
    "environment": {
        "device": "cuda",
        "gpu_name": EXPECTED_GPU_NAME,
        "renderer": "cuda_additive",
        "render_chunk": 256,
        "lpips_required": True,
        "cuda_replay_scope": "source/device/version-bound; not bit-exact",
    },
    "representation": {
        "capacity": CAPACITY,
        "persisted_arrays": sorted(FOUR_ARRAY_KEYS),
        "support_fade": "C0 compact support",
        "support_fade_alpha": 1.0,
        "mask_margin_px": MASK_MARGIN,
        "sigma_cutoff": SIGMA_CUTOFF,
        "geometry_frozen_after_placement": True,
        "containment_limit": CONTAINMENT_LIMIT,
        "render_parity_limit": RENDER_PARITY_LIMIT,
        "decoded_field_hash_required": True,
    },
    "coverage": {
        "debt": "max(0, 0.05 - unit_coverage) inside raw mask",
        "weak_predicate": "unit_coverage < 0.05",
        "raw_hole_predicate": "unit_coverage <= 0.0",
        "threshold": COVERAGE_THRESHOLD,
        "connectivity": 8,
        "component_order": "row-major deterministic",
        "exact_rerender_after_each_wave": True,
        "max_waves": MAX_WAVES,
        "max_successor_placements": MAX_SUCCESSOR_PLACEMENTS,
    },
    "candidate_bank": {
        "fallback_scale_px": MICRO_SCALE,
        "fallback_at_every_weak_pixel": True,
        "inward_normal_offsets_px": [1, 2, 3, 4],
        "tangent_proposal_scale_px": [16.0, MICRO_SCALE],
        "certificate": "ADR-0019 station-ball mask-tangent cap",
        "incidence": "exact C0-faded candidate-to-current-deficit sparse incidence",
        "greedy_priority": [
            "newly_satisfied_weak_pixels_desc",
            "covered_deficit_mass_desc",
            "weighted_appearance_variance_asc",
            "stable_candidate_id_asc",
        ],
    },
    "donor_funding": {
        "pairing": "disjoint ordinary mutual-nearest pairs",
        "ordinary_scale_min_exclusive_px": 0.20,
        "both_centres_sdf_min_exclusive_px": 2.0,
        "covariance_envelope_multiplier": 1.05,
        "recertification": "ADR-0019 anisotropic station-ball",
        "existing_micro_exemption_max_px": 0.081,
        "absorbed_rows_per_successor": 1,
        "hier031_ranking": "distance/scale + 0.25 color + 0.10 log-scale + 0.10 axial-angle",
        "contribution_ranking": [
            "exact_local_additive_merge_sse_asc",
            "exact_local_additive_sse_delta_asc",
            "stable_pair_id_asc",
        ],
        "contribution_color": "exact local RGB least squares",
        "projection": {
            "renderer": "cuda_additive",
            "support_fade_alpha": 1.0,
            "tolerance": 1e-6,
            "max_iterations": 48,
            "ridge": 1e-8,
            "coefficient_abs_limit": COEFFICIENT_ABS_LIMIT,
            "regularization_center": "input",
            "solver_start": "input",
            "frozen_base_mode": "explicit",
            "allow_unsafe_stage_zero_reconditioning": False,
            "selection_mode": "transaction",
        },
    },
    "detail": {
        "rows": DETAIL_ROWS,
        "score": "absolute high-pass residual after Gaussian blur",
        "nms_radius_px": DETAIL_NMS_RADIUS,
        "sdf_max_px": DETAIL_SDF_MAX,
        "blur_sigma_px": DETAIL_BLUR_SIGMA,
        "orientation": "source-luminance image tangent with mask-tangent fallback",
        "certificate": "maximum ADR-0019 station-ball tangent ellipse",
        "geometry_fit_after_placement": False,
    },
    "metrics": {
        "quality": [
            "foreground PSNR/MS-SSIM/LPIPS/SSIM",
            "boundary<=4 and interior>4 PSNR",
            "fixed hair and boundary crop PSNR",
            "high-pass/Laplacian/Sobel error",
            "pixel and 7x7 maxima",
        ],
        "coverage_quantiles": ["min", "q0.1%", "q1%", "q5%", "median", "q95%", "max"],
        "hair_crop_bounds_xyxy": list(HAIR_CROP_BOUNDS),
        "boundary_crop_bounds_xyxy": list(BOUNDARY_CROP_BOUNDS),
        "absolute_error_view_scale": 4.0,
    },
    "arms": list(ARMS),
    "arm_relationships": {
        "arm3_arm4_first_wave_placement_equal": True,
        "arm4_arm5_full_coverage_placement_equal": True,
    },
    "acceptance": {
        "exact_count": CAPACITY,
        "exact_four_array_payload": True,
        "zero_raw_holes": True,
        "zero_pixels_below_coverage_threshold": True,
        "support_and_reconstruction_outside_max": CONTAINMENT_LIMIT,
        "render_parity_max": RENDER_PARITY_LIMIT,
        "boundary_and_hair_strictly_improve_over_control": True,
        "interior_psnr_floor_db": INTERIOR_PSNR_FLOOR_DB,
        "coefficient_abs_max": COEFFICIENT_ABS_LIMIT,
    },
    "decision": {
        "matrix_error_policy": "any missing/error arm makes the matrix incomplete and forbids selection",
        "report_gate_requires_all_five_arms_ok": True,
        "selection": "highest foreground PSNR among arms passing every frozen clause; arm order breaks exact ties",
        "negative_tradeoff_order": [
            "weak_pixels_asc",
            "boundary_psnr_desc",
            "hair_psnr_desc",
            "foreground_psnr_desc",
            "arm_order_asc",
        ],
    },
    "execution": {
        "new_empty_output_required": True,
        "resume_or_in_place_repair_forbidden": True,
        "attempt_order": list(ARMS),
        "completed_marker_means_attempt_ledger_sealed_not_success": True,
        "portable_html_external_links_forbidden": True,
    },
    "forbidden": [
        "count scaling",
        "mask relaxation",
        "post-render masking",
        "successor geometry optimization",
        "default changes",
    ],
}
FROZEN_PROTOCOL_DIGEST = "402588c6c32a93ac1dca615ad50d2cf15248892beaaae1bf80cd9f9e253c9898"


@dataclass(frozen=True)
class Candidate:
    candidate_id: int
    component_label: int
    kind: str
    source_flat: int
    mean_x: float
    mean_y: float
    scale_x: float
    scale_y: float
    rotation: float
    appearance_variance: float = 0.0
    weak_indices: tuple[int, ...] = ()
    weights: tuple[float, ...] = ()


@dataclass(frozen=True)
class CandidateBank:
    candidates: tuple[Candidate, ...]
    weak_flats: np.ndarray
    deficits: np.ndarray
    labels: np.ndarray
    component_records: tuple[dict[str, object], ...]
    detector_seconds: float
    incidence_seconds: float
    incidence_edges: int


class CoverageClosureError(RuntimeError):
    """Coverage debt could not be closed under the frozen placement budget."""


class DonorFundingError(RuntimeError):
    """The exact-N field has too few eligible merge donors for a frozen allocation."""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, nargs="?")
    parser.add_argument("mask", type=Path, nargs="?")
    parser.add_argument("out", type=Path, nargs="?")
    parser.add_argument("--base-bundle", type=Path, default=BASE_BUNDLE)
    parser.add_argument("--max-side", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--render-chunk", type=int, default=256)
    parser.add_argument("--error-scale", type=float, default=4.0)
    parser.add_argument("--lpips", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--print-protocol-digest", action="store_true")
    return parser


def _protocol_digest() -> str:
    payload = json.dumps(PROTOCOL, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validate_args(args: argparse.Namespace) -> None:
    if args.print_protocol_digest:
        return
    if _protocol_digest() != FROZEN_PROTOCOL_DIGEST:
        raise SystemExit("HIER-032 executable protocol differs from its frozen digest")
    for name in ("image", "mask", "out"):
        if getattr(args, name) is None:
            raise SystemExit(f"{name} is required unless --print-protocol-digest is used")
    frozen = {
        "max_side": 1200,
        "seed": 0,
        "device": "cuda",
        "render_chunk": 256,
        "lpips": True,
    }
    for name, expected in frozen.items():
        if getattr(args, name) != expected:
            raise SystemExit(
                f"frozen HIER-032 development run requires {name}={expected!r}, "
                f"got {getattr(args, name)!r}"
            )
    if args.error_scale != 4.0:
        raise SystemExit(
            f"frozen HIER-032 development run requires error_scale=4.0, got {args.error_scale!r}"
        )
    for name in ("image", "mask"):
        path = getattr(args, name)
        if not path.is_file():
            raise SystemExit(f"{name} does not exist: {path}")
    if h22.report_utils._sha256(args.image) != SOURCE_SHA256:
        raise SystemExit("Janelle source SHA-256 differs from the HIER-032 binding")
    if h22.report_utils._sha256(args.mask) != MASK_SHA256:
        raise SystemExit("Janelle mask SHA-256 differs from the HIER-032 binding")
    base_field = args.base_bundle / BASE_FIELD_REL
    if not (args.base_bundle / "COMPLETED").is_file() or not base_field.is_file():
        raise SystemExit(f"immutable HIER-031 selected bundle is unavailable: {args.base_bundle}")
    if h22.report_utils._sha256(base_field) != BASE_FIELD_SHA256:
        raise SystemExit("HIER-031 selected field SHA-256 differs from the frozen binding")
    decision_path = args.base_bundle / "decision.json"
    if (
        not decision_path.is_file()
        or h22.report_utils._sha256(decision_path) != BASE_DECISION_SHA256
    ):
        raise SystemExit("HIER-031 decision receipt differs from the frozen binding")
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    if decision.get("selected_arm") != "deep_only_terminal_closure_n7000":
        raise SystemExit("HIER-031 decision no longer selects the frozen control field")


def _command() -> str:
    return " ".join(shlex.quote(value) for value in sys.argv)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _pure_field(field: GaussianField) -> GaussianField:
    return GaussianField(
        field.means.detach().clone(),
        field.log_scales.detach().clone(),
        field.rotations.detach().clone(),
        field.colors.detach().clone(),
    )


def _field_state_sha256(field: GaussianField) -> str:
    """Hash the exact decoded four-array float state independent of the NPZ container."""

    digest = hashlib.sha256(b"structsplat.gaussian-field-four-array-state.v1\0")
    for name in sorted(FOUR_ARRAY_KEYS):
        array = np.ascontiguousarray(getattr(field, name).detach().cpu().numpy())
        encoded_name = name.encode("ascii")
        descriptor = json.dumps(
            {"dtype": array.dtype.str, "shape": list(array.shape)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
        raw = array.tobytes(order="C")
        digest.update(len(encoded_name).to_bytes(4, "little"))
        digest.update(encoded_name)
        digest.update(len(descriptor).to_bytes(8, "little"))
        digest.update(descriptor)
        digest.update(len(raw).to_bytes(8, "little"))
        digest.update(raw)
    return digest.hexdigest()


def _field_state_max_abs(first: GaussianField, second: GaussianField) -> float:
    if first.n != second.n:
        return float("inf")
    return max(
        float(
            np.max(
                np.abs(
                    getattr(first, name).detach().cpu().numpy().astype(np.float64)
                    - getattr(second, name).detach().cpu().numpy().astype(np.float64)
                ),
                initial=0.0,
            )
        )
        for name in FOUR_ARRAY_KEYS
    )


def _label_components8(binary: np.ndarray) -> tuple[np.ndarray, tuple[dict[str, object], ...]]:
    """Label True pixels with deterministic row-major 8-connectivity."""

    active = np.asarray(binary, dtype=bool)
    labels = np.zeros(active.shape, dtype=np.int32)
    height, width = active.shape
    records: list[dict[str, object]] = []
    label = 0
    for y0, x0 in np.argwhere(active):
        if labels[y0, x0] != 0:
            continue
        label += 1
        labels[y0, x0] = label
        queue = deque([(int(y0), int(x0))])
        flats: list[int] = []
        while queue:
            y, x = queue.popleft()
            flats.append(y * width + x)
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    yy, xx = y + dy, x + dx
                    if (
                        0 <= yy < height
                        and 0 <= xx < width
                        and active[yy, xx]
                        and labels[yy, xx] == 0
                    ):
                        labels[yy, xx] = label
                        queue.append((yy, xx))
        records.append(
            {
                "label": label,
                "pixels": len(flats),
                "first_flat": min(flats),
                "flats": sorted(flats),
            }
        )
    return labels, tuple(records)


def _row_weights(
    mean: np.ndarray,
    scales: np.ndarray,
    rotation: float,
    xs: np.ndarray,
    ys: np.ndarray,
) -> np.ndarray:
    """Exact C0-faded unit weights at integer pixel centres for one RS row."""

    dx = np.asarray(xs, dtype=np.float64) - float(mean[0])
    dy = np.asarray(ys, dtype=np.float64) - float(mean[1])
    c, s = math.cos(float(rotation)), math.sin(float(rotation))
    sx = max(float(scales[0]), 1e-8)
    sy = max(float(scales[1]), 1e-8)
    local_x = (c * dx + s * dy) / sx
    local_y = (-s * dx + c * dy) / sy
    q = local_x * local_x + local_y * local_y
    cutoff = math.exp(-0.5 * SIGMA_CUTOFF**2)
    return np.where(q <= SIGMA_CUTOFF**2, np.maximum(np.exp(-0.5 * q) - cutoff, 0.0), 0.0)


def _fallback_candidates(
    coverage: np.ndarray,
    inside: np.ndarray,
) -> CandidateBank:
    started = time.perf_counter()
    weak = np.asarray(inside, dtype=bool) & (np.asarray(coverage) < COVERAGE_THRESHOLD)
    labels, components = _label_components8(weak)
    weak_flats = np.flatnonzero(weak.reshape(-1)).astype(np.int64, copy=False)
    deficits = COVERAGE_THRESHOLD - coverage.reshape(-1)[weak_flats].astype(np.float64)
    width = inside.shape[1]
    peak = 1.0 - math.exp(-0.5 * SIGMA_CUTOFF**2)
    candidates = tuple(
        Candidate(
            candidate_id=index,
            component_label=int(labels[flat // width, flat % width]),
            kind="fallback",
            source_flat=int(flat),
            mean_x=float(flat % width),
            mean_y=float(flat // width),
            scale_x=MICRO_SCALE,
            scale_y=MICRO_SCALE,
            rotation=0.0,
            weak_indices=(index,),
            weights=(peak,),
        )
        for index, flat in enumerate(weak_flats)
    )
    elapsed = time.perf_counter() - started
    return CandidateBank(
        candidates=candidates,
        weak_flats=weak_flats,
        deficits=deficits,
        labels=labels,
        component_records=components,
        detector_seconds=elapsed,
        incidence_seconds=0.0,
        incidence_edges=len(candidates),
    )


def _certify_tangent_geometries(
    inside: np.ndarray,
    means: np.ndarray,
    tangents: np.ndarray,
    *,
    torch_module=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return unchanged, station-ball-certified tangent candidates and a validity mask."""

    means = np.asarray(means, dtype=np.float32).reshape(-1, 2)
    tangents = np.asarray(tangents, dtype=np.float32).reshape(-1, 2)
    if not len(means):
        return (
            means,
            np.zeros((0, 2), dtype=np.float32),
            np.zeros(0, dtype=np.float32),
            np.zeros(0, dtype=bool),
        )
    norm = np.linalg.norm(tangents.astype(np.float64), axis=1)
    valid_tangent = norm > 1e-8
    unit = np.zeros_like(tangents)
    unit[valid_tangent] = tangents[valid_tangent] / norm[valid_tangent, None]
    angles = np.arctan2(unit[:, 1], unit[:, 0]).astype(np.float32)
    proposed_scales = np.repeat(
        np.asarray([[16.0, MICRO_SCALE]], dtype=np.float32), len(means), axis=0
    )
    field = GaussianField.from_numpy(
        means,
        proposed_scales,
        angles,
        np.zeros((len(means), 3), dtype=np.float32),
        device="cpu",
    )
    from structsplat.fit import _MaskConstraint

    constraint = _MaskConstraint.from_mask(
        inside,
        field.means.device,
        field.means.dtype,
        SIGMA_CUTOFF,
        MASK_MARGIN,
        aa_dilation=0.0,
        min_scale=MICRO_SCALE,
        cap_mode="anisotropic",
    )
    original = field.means.detach().clone()
    constraint.apply(field, aa_dilation=0.0, refresh=True)
    certified_means = field.means.detach().cpu().numpy().astype(np.float32, copy=False)
    scales = field.scales().detach().cpu().numpy().astype(np.float32, copy=False)
    moved = np.max(np.abs(certified_means - original.cpu().numpy()), axis=1) > 1e-5
    ix = np.rint(certified_means[:, 0]).astype(np.int64)
    iy = np.rint(certified_means[:, 1]).astype(np.int64)
    in_bounds = (
        (ix >= 0)
        & (ix < inside.shape[1])
        & (iy >= 0)
        & (iy < inside.shape[0])
    )
    centre_inside = np.zeros(len(means), dtype=bool)
    centre_inside[in_bounds] = inside[iy[in_bounds], ix[in_bounds]]
    valid = valid_tangent & ~moved & centre_inside & (scales[:, 1] >= MICRO_SCALE - 1e-6)
    return certified_means, scales, angles, valid


def _attach_incidence(
    raw_candidates: list[Candidate],
    weak_flats: np.ndarray,
    source: np.ndarray,
) -> tuple[tuple[Candidate, ...], int]:
    width = source.shape[1]
    ys = weak_flats // width
    xs = weak_flats - ys * width
    colors = source[ys, xs].astype(np.float64)
    output: list[Candidate] = []
    edges = 0
    for candidate_id, candidate in enumerate(raw_candidates):
        weights = _row_weights(
            np.asarray([candidate.mean_x, candidate.mean_y]),
            np.asarray([candidate.scale_x, candidate.scale_y]),
            candidate.rotation,
            xs,
            ys,
        )
        indices = np.flatnonzero(weights > 0.0)
        if not len(indices):
            continue
        w = weights[indices]
        total = float(w.sum())
        mean = np.sum(colors[indices] * w[:, None], axis=0) / max(total, 1e-12)
        variance = float(np.sum(w[:, None] * np.square(colors[indices] - mean)) / max(3.0 * total, 1e-12))
        output.append(
            replace(
                candidate,
                candidate_id=len(output),
                appearance_variance=variance,
                weak_indices=tuple(int(value) for value in indices),
                weights=tuple(float(value) for value in w),
            )
        )
        edges += len(indices)
    return tuple(output), edges


def _build_candidate_bank(
    coverage: np.ndarray,
    inside: np.ndarray,
    sdf: np.ndarray,
    source: np.ndarray,
    *,
    torch_module=None,
) -> CandidateBank:
    """Create guaranteed fallbacks plus certified inward-offset tangent ellipses."""

    detected_at = time.perf_counter()
    weak = np.asarray(inside, dtype=bool) & (np.asarray(coverage) < COVERAGE_THRESHOLD)
    labels, components = _label_components8(weak)
    weak_flats = np.flatnonzero(weak.reshape(-1)).astype(np.int64, copy=False)
    deficits = COVERAGE_THRESHOLD - coverage.reshape(-1)[weak_flats].astype(np.float64)
    detector_seconds = time.perf_counter() - detected_at
    width = inside.shape[1]
    normals = mask_geometry.boundary_normals(sdf)
    raw: list[Candidate] = []
    proposed_means: list[tuple[float, float]] = []
    proposed_tangents: list[tuple[float, float]] = []
    proposal_meta: list[tuple[int, int, int]] = []
    for flat in weak_flats:
        y, x = divmod(int(flat), width)
        component = int(labels[y, x])
        raw.append(
            Candidate(
                candidate_id=len(raw),
                component_label=component,
                kind="fallback",
                source_flat=int(flat),
                mean_x=float(x),
                mean_y=float(y),
                scale_x=MICRO_SCALE,
                scale_y=MICRO_SCALE,
                rotation=0.0,
            )
        )
        nx, ny = (float(normals[y, x, 0]), float(normals[y, x, 1]))
        if math.hypot(nx, ny) <= 1e-8:
            continue
        tangent = (-ny, nx)
        for offset in (1, 2, 3, 4):
            proposed_means.append((x + offset * nx, y + offset * ny))
            proposed_tangents.append(tangent)
            proposal_meta.append((component, int(flat), offset))
    if proposed_means:
        means, scales, angles, valid = _certify_tangent_geometries(
            inside,
            np.asarray(proposed_means, dtype=np.float32),
            np.asarray(proposed_tangents, dtype=np.float32),
            torch_module=torch_module,
        )
        seen: set[tuple[int, int, int, int]] = set()
        for index in np.flatnonzero(valid):
            component, source_flat, offset = proposal_meta[int(index)]
            key = (
                int(round(float(means[index, 0]) * 1_000)),
                int(round(float(means[index, 1]) * 1_000)),
                int(round(float(scales[index, 0]) * 1_000)),
                int(round(float(angles[index]) * 1_000)),
            )
            if key in seen:
                continue
            seen.add(key)
            raw.append(
                Candidate(
                    candidate_id=len(raw),
                    component_label=component,
                    kind=f"inward_offset_{offset}",
                    source_flat=source_flat,
                    mean_x=float(means[index, 0]),
                    mean_y=float(means[index, 1]),
                    scale_x=float(scales[index, 0]),
                    scale_y=float(scales[index, 1]),
                    rotation=float(angles[index]),
                )
            )
    incidence_at = time.perf_counter()
    candidates, edges = _attach_incidence(raw, weak_flats, source)
    incidence_seconds = time.perf_counter() - incidence_at
    fallbacks = {candidate.source_flat for candidate in candidates if candidate.kind == "fallback"}
    if fallbacks != set(int(value) for value in weak_flats):
        raise CoverageClosureError("candidate bank lost a guaranteed weak-pixel fallback")
    return CandidateBank(
        candidates=candidates,
        weak_flats=weak_flats,
        deficits=deficits,
        labels=labels,
        component_records=components,
        detector_seconds=detector_seconds,
        incidence_seconds=incidence_seconds,
        incidence_edges=edges,
    )


def _greedy_cover(bank: CandidateBank) -> tuple[list[Candidate], dict[str, object]]:
    """Greedily close every deficit using the frozen four-level stable priority."""

    started = time.perf_counter()
    remaining = bank.deficits.astype(np.float64, copy=True)
    available = np.ones(len(bank.candidates), dtype=bool)
    selected: list[Candidate] = []
    while bool((remaining > 1e-12).any()):
        best_index = -1
        best_key: tuple[float, float, float, int] | None = None
        for index, candidate in enumerate(bank.candidates):
            if not available[index]:
                continue
            incidence = np.asarray(candidate.weak_indices, dtype=np.int64)
            weights = np.asarray(candidate.weights, dtype=np.float64)
            active = remaining[incidence] > 1e-12
            if not bool(active.any()):
                continue
            rem = remaining[incidence[active]]
            contribution = weights[active]
            newly_satisfied = int(np.count_nonzero(contribution + 1e-12 >= rem))
            deficit_mass = float(np.minimum(contribution, rem).sum())
            key = (
                -float(newly_satisfied),
                -deficit_mass,
                float(candidate.appearance_variance),
                int(candidate.candidate_id),
            )
            if best_key is None or key < best_key:
                best_key = key
                best_index = index
        if best_index < 0:
            unresolved = int(np.count_nonzero(remaining > 1e-12))
            raise CoverageClosureError(f"greedy cover stalled with {unresolved} weak pixels")
        candidate = bank.candidates[best_index]
        incidence = np.asarray(candidate.weak_indices, dtype=np.int64)
        weights = np.asarray(candidate.weights, dtype=np.float64)
        remaining[incidence] = np.maximum(0.0, remaining[incidence] - weights)
        available[best_index] = False
        selected.append(candidate)
    elapsed = time.perf_counter() - started
    kinds: dict[str, int] = {}
    for candidate in selected:
        kinds[candidate.kind] = kinds.get(candidate.kind, 0) + 1
    return selected, {
        "selector_seconds": elapsed,
        "weak_pixels": int(len(bank.weak_flats)),
        "deficit_mass": float(bank.deficits.sum()),
        "candidate_count": len(bank.candidates),
        "incidence_edges": bank.incidence_edges,
        "selected_count": len(selected),
        "candidate_compression": (
            float(len(bank.weak_flats)) / len(selected) if selected else 1.0
        ),
        "selected_kinds": kinds,
        "complete": bool(np.all(remaining <= 1e-12)),
    }


def _candidate_record(candidate: Candidate) -> dict[str, object]:
    return {
        "candidate_id": candidate.candidate_id,
        "component_label": candidate.component_label,
        "kind": candidate.kind,
        "source_flat": candidate.source_flat,
        "mean": [candidate.mean_x, candidate.mean_y],
        "scales": [candidate.scale_x, candidate.scale_y],
        "rotation": candidate.rotation,
        "appearance_variance": candidate.appearance_variance,
        "covered_weak_pixels": len(candidate.weak_indices),
    }


def _render(field: GaussianField, shape: tuple[int, int], args, torch):
    return h31._render(field, "cuda_additive", shape, args, torch)


def _coverage(field: GaussianField, shape: tuple[int, int], args, torch) -> np.ndarray:
    return h31._coverage(field, shape, args, torch)


def _pair_geometry(field: GaussianField, sdf: np.ndarray, inside: np.ndarray, torch):
    """Build recertified envelope proposals for eligible HIER-031 mutual-nearest pairs."""

    from structsplat.fit import _MaskConstraint
    from structsplat.safe_schedule import _production_mutual_nearest_pairs
    from structsplat.triage import _row_covariances

    first_all, second_all, distance_all = _production_mutual_nearest_pairs(
        field.means.detach()
    )
    if first_all.numel() == 0:
        raise DonorFundingError("no mutual-nearest donor pairs remain")
    scales = field.scales().detach()
    colors = field.colors.detach()
    means = field.means.detach()
    ordinary = (scales[first_all].amin(dim=1) > 0.20) & (
        scales[second_all].amin(dim=1) > 0.20
    )
    ia = means[first_all].round().long()
    ib = means[second_all].round().long()
    sdf_tensor = torch.as_tensor(sdf, device=means.device, dtype=means.dtype)
    depth = (sdf_tensor[ia[:, 1], ia[:, 0]] > 2.0) & (
        sdf_tensor[ib[:, 1], ib[:, 0]] > 2.0
    )
    eligible = (ordinary & depth).nonzero(as_tuple=False).reshape(-1)
    first = first_all[eligible]
    second = second_all[eligible]
    distance = distance_all[eligible]
    if first.numel() == 0:
        raise DonorFundingError("no SDF>2 ordinary mutual-nearest donor pairs remain")
    if torch.unique(torch.cat([first, second])).numel() != 2 * first.numel():
        raise RuntimeError("mutual-nearest donor pairs are not disjoint")

    midpoint = 0.5 * (means[first] + means[second])
    cov_a = _row_covariances(field, first)
    cov_b = _row_covariances(field, second)
    da = means[first] - midpoint
    db = means[second] - midpoint
    spatial_a = cov_a + da[:, :, None] * da[:, None, :]
    spatial_b = cov_b + db[:, :, None] * db[:, None, :]
    moment = 0.5 * (spatial_a + spatial_b)
    values, vectors = torch.linalg.eigh(moment)
    inverse_sqrt = (
        vectors
        @ torch.diag_embed(values.clamp_min(1e-8).rsqrt())
        @ vectors.transpose(1, 2)
    )
    envelope = torch.maximum(
        torch.linalg.eigvalsh(inverse_sqrt @ spatial_a @ inverse_sqrt)[:, -1],
        torch.linalg.eigvalsh(inverse_sqrt @ spatial_b @ inverse_sqrt)[:, -1],
    ).clamp_min(1.0)
    covariance = moment * envelope[:, None, None] * 1.05**2
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    eigenvalues = eigenvalues.clamp_min(ORDINARY_MIN_SCALE**2)
    major = eigenvectors[:, :, 1]
    merged_scales = torch.stack(
        [torch.sqrt(eigenvalues[:, 1]), torch.sqrt(eigenvalues[:, 0])], dim=1
    )
    rotations = torch.atan2(major[:, 1], major[:, 0])
    area_a = scales[first].prod(dim=1)
    area_b = scales[second].prod(dim=1)
    merged_area = merged_scales.prod(dim=1).clamp_min(1e-8)
    area_colors = (
        area_a[:, None] * colors[first] + area_b[:, None] * colors[second]
    ) / merged_area[:, None]

    proposed = GaussianField(
        midpoint.clone(),
        torch.log(merged_scales),
        rotations.clone(),
        area_colors.clone(),
    )
    constraint = _MaskConstraint.from_mask(
        inside,
        means.device,
        means.dtype,
        SIGMA_CUTOFF,
        MASK_MARGIN,
        aa_dilation=0.0,
        min_scale=ORDINARY_MIN_SCALE,
        cap_mode="anisotropic",
        undercoverage_band=4.0,
    )
    constraint.apply(proposed, aa_dilation=0.0, refresh=True)
    proposed = _pure_field(proposed)

    color_delta = torch.linalg.norm(colors[first] - colors[second], dim=1)
    sorted_scales = torch.sort(scales, dim=1).values.clamp_min(1e-6)
    scale_delta = torch.linalg.norm(
        torch.log(sorted_scales[first]) - torch.log(sorted_scales[second]), dim=1
    )
    angle_delta = field.rotations.detach()[first] - field.rotations.detach()[second]
    axial_delta = 0.5 * torch.abs(
        torch.atan2(torch.sin(2.0 * angle_delta), torch.cos(2.0 * angle_delta))
    )
    pair_scale = 0.5 * (
        torch.sqrt(scales[first].prod(dim=1))
        + torch.sqrt(scales[second].prod(dim=1))
    ).clamp_min(ORDINARY_MIN_SCALE)
    hier031_score = (
        distance / pair_scale
        + 0.25 * color_delta
        + 0.10 * scale_delta
        + 0.10 * axial_delta
    )
    return {
        "all_pair_count": int(first_all.numel()),
        "first": first,
        "second": second,
        "distance": distance,
        "means": proposed.means,
        "scales": proposed.scales(),
        "rotations": proposed.rotations,
        "area_colors": proposed.colors,
        "hier031_score": hier031_score,
    }


def _row_bbox(
    mean: np.ndarray,
    scales: np.ndarray,
    rotation: float,
    shape: tuple[int, int],
) -> tuple[int, int, int, int]:
    c, s = math.cos(float(rotation)), math.sin(float(rotation))
    sx, sy = float(scales[0]), float(scales[1])
    radius_x = SIGMA_CUTOFF * math.sqrt((sx * c) ** 2 + (sy * s) ** 2)
    radius_y = SIGMA_CUTOFF * math.sqrt((sx * s) ** 2 + (sy * c) ** 2)
    height, width = shape
    return (
        max(0, int(math.floor(float(mean[0]) - radius_x))),
        max(0, int(math.floor(float(mean[1]) - radius_y))),
        min(width, int(math.ceil(float(mean[0]) + radius_x)) + 1),
        min(height, int(math.ceil(float(mean[1]) + radius_y)) + 1),
    )


def _local_merge_fit(
    reconstruction: np.ndarray,
    target: np.ndarray,
    inside: np.ndarray,
    row_a: tuple[np.ndarray, np.ndarray, float, np.ndarray],
    row_b: tuple[np.ndarray, np.ndarray, float, np.ndarray],
    merged: tuple[np.ndarray, np.ndarray, float, np.ndarray],
) -> dict[str, object]:
    """Fit one merged RGB row and score its exact local additive reconstruction SSE."""

    mean_a, scales_a, rotation_a, color_a = row_a
    mean_b, scales_b, rotation_b, color_b = row_b
    mean_m, scales_m, rotation_m, area_color = merged
    boxes = (
        _row_bbox(mean_a, scales_a, rotation_a, inside.shape),
        _row_bbox(mean_b, scales_b, rotation_b, inside.shape),
        _row_bbox(mean_m, scales_m, rotation_m, inside.shape),
    )
    x0 = min(box[0] for box in boxes)
    y0 = min(box[1] for box in boxes)
    x1 = max(box[2] for box in boxes)
    y1 = max(box[3] for box in boxes)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    active = inside[y0:y1, x0:x1]
    xs = xx[active].astype(np.float64, copy=False)
    ys = yy[active].astype(np.float64, copy=False)
    if not len(xs):
        return {
            "coefficient": np.asarray(area_color, dtype=np.float64),
            "merge_sse": float("inf"),
            "area_color_sse": float("inf"),
            "baseline_sse": float("inf"),
            "delta_sse": float("inf"),
            "pixels": 0,
        }
    wa = _row_weights(mean_a, scales_a, rotation_a, xs, ys)
    wb = _row_weights(mean_b, scales_b, rotation_b, xs, ys)
    wm = _row_weights(mean_m, scales_m, rotation_m, xs, ys)
    current = reconstruction[y0:y1, x0:x1][active].astype(np.float64)
    desired = target[y0:y1, x0:x1][active].astype(np.float64)
    other = current - wa[:, None] * np.asarray(color_a) - wb[:, None] * np.asarray(color_b)
    denominator = float(np.dot(wm, wm))
    if denominator <= 1e-12:
        coefficient = np.asarray(area_color, dtype=np.float64)
    else:
        coefficient = np.sum(wm[:, None] * (desired - other), axis=0) / denominator
    fitted = other + wm[:, None] * coefficient
    area_fitted = other + wm[:, None] * np.asarray(area_color, dtype=np.float64)
    baseline_sse = float(np.square(current - desired).sum())
    merge_sse = float(np.square(fitted - desired).sum())
    return {
        "coefficient": coefficient,
        "merge_sse": merge_sse,
        "area_color_sse": float(np.square(area_fitted - desired).sum()),
        "baseline_sse": baseline_sse,
        "delta_sse": merge_sse - baseline_sse,
        "pixels": int(len(xs)),
    }


def _evaluate_pair_errors(
    field: GaussianField,
    pair_data: dict[str, object],
    reconstruction: np.ndarray,
    target: np.ndarray,
    inside: np.ndarray,
    pair_indices: np.ndarray,
) -> dict[str, np.ndarray]:
    means = field.means.detach().cpu().numpy()
    scales = field.scales().detach().cpu().numpy()
    rotations = field.rotations.detach().cpu().numpy()
    colors = field.colors.detach().cpu().numpy()
    first = pair_data["first"].detach().cpu().numpy()
    second = pair_data["second"].detach().cpu().numpy()
    merge_means = pair_data["means"].detach().cpu().numpy()
    merge_scales = pair_data["scales"].detach().cpu().numpy()
    merge_rotations = pair_data["rotations"].detach().cpu().numpy()
    area_colors = pair_data["area_colors"].detach().cpu().numpy()
    coefficients = np.zeros((len(pair_indices), 3), dtype=np.float64)
    merge_sse = np.zeros(len(pair_indices), dtype=np.float64)
    area_sse = np.zeros(len(pair_indices), dtype=np.float64)
    baseline_sse = np.zeros(len(pair_indices), dtype=np.float64)
    delta_sse = np.zeros(len(pair_indices), dtype=np.float64)
    pixels = np.zeros(len(pair_indices), dtype=np.int64)
    for out_index, raw_index in enumerate(pair_indices):
        index = int(raw_index)
        a, b = int(first[index]), int(second[index])
        fitted = _local_merge_fit(
            reconstruction,
            target,
            inside,
            (means[a], scales[a], float(rotations[a]), colors[a]),
            (means[b], scales[b], float(rotations[b]), colors[b]),
            (
                merge_means[index],
                merge_scales[index],
                float(merge_rotations[index]),
                area_colors[index],
            ),
        )
        coefficients[out_index] = fitted["coefficient"]
        merge_sse[out_index] = fitted["merge_sse"]
        area_sse[out_index] = fitted["area_color_sse"]
        baseline_sse[out_index] = fitted["baseline_sse"]
        delta_sse[out_index] = fitted["delta_sse"]
        pixels[out_index] = fitted["pixels"]
    return {
        "coefficients": coefficients,
        "merge_sse": merge_sse,
        "area_color_sse": area_sse,
        "baseline_sse": baseline_sse,
        "delta_sse": delta_sse,
        "pixels": pixels,
    }


def _contribution_order(errors: dict[str, np.ndarray]) -> np.ndarray:
    pair_ids = np.arange(len(errors["merge_sse"]), dtype=np.int64)
    return np.lexsort((pair_ids, errors["delta_sse"], errors["merge_sse"]))


def _select_donors(
    field: GaussianField,
    pair_data: dict[str, object],
    count: int,
    reconstruction: np.ndarray,
    target: np.ndarray,
    inside: np.ndarray,
    mode: str,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, object]]:
    eligible = int(pair_data["first"].numel())
    if eligible < count:
        raise DonorFundingError(
            f"only {eligible} eligible mutual-nearest donor pairs for {count} placements"
        )
    started = time.perf_counter()
    if mode == "hier031":
        score = pair_data["hier031_score"].detach().cpu().numpy().astype(np.float64)
        order = np.argsort(score, kind="stable")
        selected = order[:count]
        errors = _evaluate_pair_errors(
            field, pair_data, reconstruction, target, inside, selected
        )
        ranking = score[selected]
    elif mode == "contribution":
        all_indices = np.arange(eligible, dtype=np.int64)
        all_errors = _evaluate_pair_errors(
            field, pair_data, reconstruction, target, inside, all_indices
        )
        order = _contribution_order(all_errors)
        selected = order[:count]
        errors = {key: value[selected] for key, value in all_errors.items()}
        ranking = all_errors["merge_sse"][selected]
    else:
        raise ValueError(f"unknown donor mode {mode!r}")
    telemetry = {
        "mode": mode,
        "candidate_pairs": int(pair_data["all_pair_count"]),
        "eligible_pairs": eligible,
        "selected_pairs": count,
        "selector_seconds": time.perf_counter() - started,
        "ranking_min": float(np.min(ranking)),
        "ranking_mean": float(np.mean(ranking)),
        "ranking_max": float(np.max(ranking)),
        "local_merge_sse_min": float(np.min(errors["merge_sse"])),
        "local_merge_sse_mean": float(np.mean(errors["merge_sse"])),
        "local_merge_sse_max": float(np.max(errors["merge_sse"])),
        "local_delta_sse_min": float(np.min(errors["delta_sse"])),
        "local_delta_sse_mean": float(np.mean(errors["delta_sse"])),
        "local_delta_sse_max": float(np.max(errors["delta_sse"])),
        "local_area_color_sse_mean": float(np.mean(errors["area_color_sse"])),
        "local_pixels_total": int(np.sum(errors["pixels"])),
    }
    return selected.astype(np.int64, copy=False), errors, telemetry


def _assemble_funded_field(
    field: GaussianField,
    pair_data: dict[str, object],
    selected: np.ndarray,
    selected_colors: np.ndarray,
    placements: list[Candidate],
    target: np.ndarray,
    inside: np.ndarray,
    args: argparse.Namespace,
    torch,
) -> tuple[GaussianField, dict[str, object]]:
    """Replace one row per donor pair, append the same number of successors, and project RGB."""

    from structsplat.fit import _MaskConstraint

    selected_t = torch.as_tensor(selected, device=field.means.device, dtype=torch.long)
    keep_rows = pair_data["first"][selected_t]
    absorbed_rows = pair_data["second"][selected_t]
    if len(placements) != len(absorbed_rows):
        raise ValueError("funding/placement cardinality differs")
    trial = field.detached()
    trial.means[keep_rows] = pair_data["means"][selected_t]
    trial.log_scales[keep_rows] = torch.log(pair_data["scales"][selected_t])
    trial.rotations[keep_rows] = pair_data["rotations"][selected_t]
    trial.colors[keep_rows] = torch.as_tensor(
        selected_colors,
        device=field.means.device,
        dtype=field.means.dtype,
    )
    retain = torch.ones(field.n, device=field.means.device, dtype=torch.bool)
    retain[absorbed_rows] = False
    funded = trial.subset(retain)
    constraint = _MaskConstraint.from_mask(
        inside,
        field.means.device,
        field.means.dtype,
        SIGMA_CUTOFF,
        MASK_MARGIN,
        aa_dilation=0.0,
        min_scale=ORDINARY_MIN_SCALE,
        cap_mode="anisotropic",
        undercoverage_band=4.0,
    )
    existing_micro = funded.scales().detach().amax(dim=1) <= 0.081
    constraint.apply(funded, aa_dilation=0.0, refresh=True, exempt=existing_micro)
    funded = _pure_field(funded)
    funded_render = _render(funded, inside.shape, args, torch)
    residual = target - funded_render.detach().cpu().numpy()
    means = np.asarray([[row.mean_x, row.mean_y] for row in placements], dtype=np.float32)
    scales = np.asarray([[row.scale_x, row.scale_y] for row in placements], dtype=np.float32)
    rotations = np.asarray([row.rotation for row in placements], dtype=np.float32)
    ix = np.rint(means[:, 0]).astype(np.int64)
    iy = np.rint(means[:, 1]).astype(np.int64)
    if not np.all(inside[iy, ix]):
        raise RuntimeError("successor placement contains a centre outside the mask")
    peak = 1.0 - math.exp(-0.5 * SIGMA_CUTOFF**2)
    initial_colors = residual[iy, ix] / peak
    successor = GaussianField.from_numpy(
        means,
        scales,
        rotations,
        initial_colors,
        device=field.means.device,
    )
    proposal = funded.append(successor)
    if proposal.n != CAPACITY:
        raise RuntimeError(f"count-neutral funding produced {proposal.n} rows")
    projection = project_additive_endpoint(
        proposal,
        target,
        config=h30._projection_config(args, contained=True),
        device=args.device,
        mask=inside,
    )
    output = _pure_field(projection.field)
    if output.n != CAPACITY:
        raise RuntimeError("global coefficient projection changed the Gaussian count")
    coefficient_abs_max = float(output.colors.detach().abs().max().cpu())
    if not math.isfinite(coefficient_abs_max) or coefficient_abs_max > COEFFICIENT_ABS_LIMIT:
        raise RuntimeError(
            "global coefficient projection selected an unbounded endpoint: "
            f"{coefficient_abs_max:.9g} > {COEFFICIENT_ABS_LIMIT:g}"
        )
    return output, {
        "keep_rows": keep_rows.detach().cpu().tolist(),
        "absorbed_rows": absorbed_rows.detach().cpu().tolist(),
        "coefficient_abs_max": coefficient_abs_max,
        "projection": h31._projection_record(projection),
    }


def _fund_placements(
    field: GaussianField,
    placements: list[Candidate],
    target: np.ndarray,
    inside: np.ndarray,
    sdf: np.ndarray,
    args: argparse.Namespace,
    torch,
    *,
    donor_mode: str,
) -> tuple[GaussianField, dict[str, object]]:
    if not placements:
        raise ValueError("funding requires at least one successor placement")
    before = _render(field, inside.shape, args, torch).detach().cpu().numpy()
    pair_data = _pair_geometry(field, sdf, inside, torch)
    selected, errors, telemetry = _select_donors(
        field,
        pair_data,
        len(placements),
        before,
        target,
        inside,
        donor_mode,
    )
    selected_t = torch.as_tensor(selected, device=field.means.device, dtype=torch.long)
    keep_rows = pair_data["first"][selected_t]
    absorbed_rows = pair_data["second"][selected_t]
    donor_geometry = {
        "donor_keep_means": field.means.detach()[keep_rows].cpu().tolist(),
        "donor_absorbed_means": field.means.detach()[absorbed_rows].cpu().tolist(),
    }
    if donor_mode == "contribution":
        colors = errors["coefficients"]
    else:
        colors = pair_data["area_colors"][
            selected_t
        ].detach().cpu().numpy()
    output, assembly = _assemble_funded_field(
        field,
        pair_data,
        selected,
        colors,
        placements,
        target,
        inside,
        args,
        torch,
    )
    return output, {**telemetry, **donor_geometry, **assembly}


def _placement_digest(placements: list[Candidate]) -> str:
    payload = [
        {
            "mean": [row.mean_x, row.mean_y],
            "scales": [row.scale_x, row.scale_y],
            "rotation": row.rotation,
            "kind": row.kind,
            "source_flat": row.source_flat,
        }
        for row in placements
    ]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _coverage_state(coverage: np.ndarray, inside: np.ndarray) -> dict[str, object]:
    weak = inside & (coverage < COVERAGE_THRESHOLD)
    labels, components = _label_components8(weak)
    active = coverage[inside].astype(np.float64)
    deficit = np.maximum(0.0, COVERAGE_THRESHOLD - active)
    return {
        "weak_pixels": int(weak.sum()),
        "components": len(components),
        "largest_component": max((int(row["pixels"]) for row in components), default=0),
        "deficit_mass": float(deficit.sum()),
        "raw_holes": int((inside & (coverage <= 0.0)).sum()),
        "labels": labels,
    }


def _close_coverage(
    base: GaussianField,
    target: np.ndarray,
    source: np.ndarray,
    inside: np.ndarray,
    sdf: np.ndarray,
    args: argparse.Namespace,
    torch,
    *,
    placement_mode: str,
    donor_mode: str,
) -> tuple[GaussianField, list[dict[str, object]], dict[str, object]]:
    field = _pure_field(base)
    history: list[dict[str, object]] = []
    placements_total: list[Candidate] = []
    donor_keep: list[int] = []
    donor_absorbed: list[int] = []
    first_labels: np.ndarray | None = None
    detector_seconds = 0.0
    selector_seconds = 0.0
    incidence_seconds = 0.0
    for wave in range(MAX_WAVES):
        coverage_before = _coverage(field, inside.shape, args, torch)
        before = _coverage_state(coverage_before, inside)
        if first_labels is None:
            first_labels = np.asarray(before["labels"], dtype=np.int32)
        record: dict[str, object] = {
            "wave": wave,
            "weak_pixels_before": before["weak_pixels"],
            "components_before": before["components"],
            "deficit_mass_before": before["deficit_mass"],
            "raw_holes_before": before["raw_holes"],
        }
        if int(before["weak_pixels"]) == 0:
            record["status"] = "closed"
            history.append(record)
            break
        if placement_mode == "fallback":
            bank = _fallback_candidates(coverage_before, inside)
            placements = list(bank.candidates)
            selector = {
                "selector_seconds": 0.0,
                "weak_pixels": len(bank.weak_flats),
                "deficit_mass": float(bank.deficits.sum()),
                "candidate_count": len(bank.candidates),
                "incidence_edges": bank.incidence_edges,
                "selected_count": len(placements),
                "candidate_compression": 1.0,
                "selected_kinds": {"fallback": len(placements)},
                "complete": True,
            }
        elif placement_mode == "set_cover":
            bank = _build_candidate_bank(
                coverage_before, inside, sdf, source, torch_module=torch
            )
            placements, selector = _greedy_cover(bank)
        else:
            raise ValueError(f"unknown placement mode {placement_mode!r}")
        detector_seconds += bank.detector_seconds
        incidence_seconds += bank.incidence_seconds
        selector_seconds += float(selector["selector_seconds"])
        if len(placements_total) + len(placements) > MAX_SUCCESSOR_PLACEMENTS:
            raise CoverageClosureError(
                f"coverage closure requires more than {MAX_SUCCESSOR_PLACEMENTS} successors"
            )
        funded, funding = _fund_placements(
            field,
            placements,
            target,
            inside,
            sdf,
            args,
            torch,
            donor_mode=donor_mode,
        )
        coverage_after = _coverage(funded, inside.shape, args, torch)
        after = _coverage_state(coverage_after, inside)
        record.update(
            {
                "status": "allocated",
                "candidate_bank": {
                    "detector_seconds": bank.detector_seconds,
                    "incidence_seconds": bank.incidence_seconds,
                    "component_count": len(bank.component_records),
                    **selector,
                },
                "placement_digest": _placement_digest(placements),
                "placements": [_candidate_record(row) for row in placements],
                "funding": funding,
                "weak_pixels_after": after["weak_pixels"],
                "components_after": after["components"],
                "deficit_mass_after": after["deficit_mass"],
                "raw_holes_after": after["raw_holes"],
            }
        )
        history.append(record)
        placements_total.extend(placements)
        donor_keep.extend(int(value) for value in funding["keep_rows"])
        donor_absorbed.extend(int(value) for value in funding["absorbed_rows"])
        field = funded
    final_coverage = _coverage(field, inside.shape, args, torch)
    final_state = _coverage_state(final_coverage, inside)
    if int(final_state["weak_pixels"]) != 0:
        raise CoverageClosureError(
            f"coverage closure ended with {final_state['weak_pixels']} pixels below 0.05"
        )
    metadata = {
        "placement_mode": placement_mode,
        "donor_mode": donor_mode,
        "waves": sum(row.get("status") == "allocated" for row in history),
        "successor_placements": len(placements_total),
        "placement_digest": _placement_digest(placements_total),
        "detector_seconds": detector_seconds,
        "incidence_seconds": incidence_seconds,
        "selector_seconds": selector_seconds,
        "placements": [_candidate_record(row) for row in placements_total],
        "donor_keep_rows": donor_keep,
        "donor_absorbed_rows": donor_absorbed,
        "initial_component_labels": first_labels,
        "final_coverage": final_coverage,
    }
    return field, history, metadata


def _deterministic_nms(
    score: np.ndarray,
    eligible: np.ndarray,
    count: int,
    radius: int,
) -> np.ndarray:
    flat = np.flatnonzero(eligible.reshape(-1))
    values = score.reshape(-1)[flat].astype(np.float64)
    order = np.lexsort((flat, -values))
    suppressed = np.zeros(eligible.shape, dtype=bool)
    selected: list[int] = []
    height, width = eligible.shape
    for index in order:
        site = int(flat[index])
        y, x = divmod(site, width)
        if suppressed[y, x]:
            continue
        selected.append(site)
        y0, y1 = max(0, y - radius), min(height, y + radius + 1)
        x0, x1 = max(0, x - radius), min(width, x + radius + 1)
        suppressed[y0:y1, x0:x1] = True
        if len(selected) >= count:
            break
    return np.asarray(selected, dtype=np.int64)


def _detail_candidates(
    field: GaussianField,
    target: np.ndarray,
    source: np.ndarray,
    inside: np.ndarray,
    sdf: np.ndarray,
    args: argparse.Namespace,
    torch,
) -> tuple[list[Candidate], dict[str, object]]:
    started = time.perf_counter()
    rendered = _render(field, inside.shape, args, torch)
    target_tensor = torch.as_tensor(target, device=rendered.device, dtype=rendered.dtype)
    residual = rendered - target_tensor
    highpass = residual - gaussian_blur(residual, DETAIL_BLUR_SIGMA)
    score = highpass.abs().mean(dim=2).detach().cpu().numpy().astype(np.float32)
    eligible = inside & (sdf <= DETAIL_SDF_MAX)
    # Select a reserve larger than 128 before certificate filtering. NMS itself stays fixed.
    sites = _deterministic_nms(
        score, eligible, min(int(eligible.sum()), DETAIL_ROWS * 4), DETAIL_NMS_RADIUS
    )
    width = inside.shape[1]
    luminance = np.tensordot(
        source.astype(np.float64),
        np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float64),
        axes=([2], [0]),
    )
    gy, gx = np.gradient(luminance)
    mask_normals = mask_geometry.boundary_normals(sdf)
    means: list[tuple[float, float]] = []
    tangents: list[tuple[float, float]] = []
    retained_sites: list[int] = []
    tangent_fallbacks = 0
    for site in sites:
        y, x = divmod(int(site), width)
        tangent = (-float(gy[y, x]), float(gx[y, x]))
        if math.hypot(*tangent) <= 1e-8:
            nx, ny = mask_normals[y, x]
            tangent = (-float(ny), float(nx))
            tangent_fallbacks += 1
        if math.hypot(*tangent) <= 1e-8:
            continue
        means.append((float(x), float(y)))
        tangents.append(tangent)
        retained_sites.append(int(site))
    certified_means, scales, angles, valid = _certify_tangent_geometries(
        inside,
        np.asarray(means, dtype=np.float32),
        np.asarray(tangents, dtype=np.float32),
        torch_module=torch,
    )
    candidates: list[Candidate] = []
    for index in np.flatnonzero(valid):
        site = retained_sites[int(index)]
        candidates.append(
            Candidate(
                candidate_id=len(candidates),
                component_label=0,
                kind="boundary_highpass",
                source_flat=site,
                mean_x=float(certified_means[index, 0]),
                mean_y=float(certified_means[index, 1]),
                scale_x=float(scales[index, 0]),
                scale_y=float(scales[index, 1]),
                rotation=float(angles[index]),
            )
        )
        if len(candidates) == DETAIL_ROWS:
            break
    if len(candidates) != DETAIL_ROWS:
        raise CoverageClosureError(
            f"detail selector certified {len(candidates)} of {DETAIL_ROWS} required strokes"
        )
    selected_scores = np.asarray(
        [score[row.source_flat // width, row.source_flat % width] for row in candidates]
    )
    return candidates, {
        "requested": DETAIL_ROWS,
        "selected": len(candidates),
        "eligible_pixels": int(eligible.sum()),
        "nms_radius": DETAIL_NMS_RADIUS,
        "sdf_max": DETAIL_SDF_MAX,
        "blur_sigma": DETAIL_BLUR_SIGMA,
        "tangent": "source-luminance finite-difference image tangent",
        "tangent_fallbacks": tangent_fallbacks,
        "score_min": float(selected_scores.min()),
        "score_mean": float(selected_scores.mean()),
        "score_max": float(selected_scores.max()),
        "selector_seconds": time.perf_counter() - started,
        "placement_digest": _placement_digest(candidates),
    }


def _append_boundary_detail(
    base: GaussianField,
    target: np.ndarray,
    source: np.ndarray,
    inside: np.ndarray,
    sdf: np.ndarray,
    args: argparse.Namespace,
    torch,
) -> tuple[GaussianField, list[dict[str, object]], dict[str, object]]:
    coverage_before = _coverage(base, inside.shape, args, torch)
    state_before = _coverage_state(coverage_before, inside)
    if int(state_before["weak_pixels"]) != 0:
        raise CoverageClosureError("detail batch requires a coverage-closed input field")
    placements, selection = _detail_candidates(
        base, target, source, inside, sdf, args, torch
    )
    field, funding = _fund_placements(
        base,
        placements,
        target,
        inside,
        sdf,
        args,
        torch,
        donor_mode="contribution",
    )
    coverage_after = _coverage(field, inside.shape, args, torch)
    state_after = _coverage_state(coverage_after, inside)
    if int(state_after["weak_pixels"]) != 0:
        raise CoverageClosureError(
            f"fixed detail batch reopened {state_after['weak_pixels']} weak pixels"
        )
    history = [
        {
            "stage": "boundary_highpass",
            "selection": selection,
            "funding": funding,
            "weak_pixels_before": state_before["weak_pixels"],
            "weak_pixels_after": state_after["weak_pixels"],
            "deficit_mass_after": state_after["deficit_mass"],
        }
    ]
    metadata = {
        "placement_mode": "boundary_highpass",
        "donor_mode": "contribution",
        "waves": 1,
        "successor_placements": DETAIL_ROWS,
        "placement_digest": _placement_digest(placements),
        "detector_seconds": 0.0,
        "incidence_seconds": 0.0,
        "selector_seconds": selection["selector_seconds"],
        "placements": [_candidate_record(row) for row in placements],
        "donor_keep_rows": funding["keep_rows"],
        "donor_absorbed_rows": funding["absorbed_rows"],
        "initial_component_labels": np.asarray(state_before["labels"]),
        "final_coverage": coverage_after,
    }
    return field, history, metadata


def _psnr(mse: float) -> float:
    return float("inf") if mse <= 0.0 else float(-10.0 * math.log10(mse))


def _masked_region_quality(
    reconstruction: np.ndarray,
    target: np.ndarray,
    region: np.ndarray,
    prefix: str,
) -> dict[str, float | int]:
    count = int(np.count_nonzero(region))
    if count == 0:
        raise ValueError(f"{prefix} metric region is empty")
    error = reconstruction[region].astype(np.float64) - target[region].astype(np.float64)
    mse = float(np.square(error).mean())
    mae = float(np.abs(error).mean())
    return {
        f"{prefix}_pixels": count,
        f"{prefix}_mse": mse,
        f"{prefix}_mae": mae,
        f"{prefix}_psnr_db": _psnr(mse),
    }


def _crop_region(
    inside: np.ndarray,
    bounds: tuple[int, int, int, int],
) -> np.ndarray:
    x0, y0, x1, y1 = bounds
    region = np.zeros_like(inside)
    region[y0:y1, x0:x1] = inside[y0:y1, x0:x1]
    return region


def _coverage_telemetry(coverage: np.ndarray, inside: np.ndarray) -> dict[str, object]:
    active = coverage[inside].astype(np.float64)
    weak = inside & (coverage < COVERAGE_THRESHOLD)
    labels, components = _label_components8(weak)
    deficit = np.maximum(0.0, COVERAGE_THRESHOLD - active)
    return {
        "coverage_lt_005_pixels": int(weak.sum()),
        "coverage_lt_005_components": len(components),
        "coverage_lt_005_largest_component": max(
            (int(row["pixels"]) for row in components), default=0
        ),
        "coverage_deficit_mass": float(deficit.sum()),
        "coverage_inside_min": float(active.min()),
        "coverage_inside_q001": float(np.quantile(active, 0.001)),
        "coverage_inside_q01": float(np.quantile(active, 0.01)),
        "coverage_inside_q05": float(np.quantile(active, 0.05)),
        "coverage_inside_q50": float(np.quantile(active, 0.50)),
        "coverage_inside_q95": float(np.quantile(active, 0.95)),
        "coverage_inside_max": float(active.max()),
        "coverage_component_labels": labels,
    }


def _funding_records(history: list[dict[str, object]]) -> list[dict[str, object]]:
    return [
        row["funding"]
        for row in history
        if isinstance(row.get("funding"), dict)
    ]


def _method_telemetry(
    history: list[dict[str, object]],
    metadata: dict[str, object],
) -> dict[str, object]:
    funding = _funding_records(history)
    candidate_banks = [
        row["candidate_bank"]
        for row in history
        if isinstance(row.get("candidate_bank"), dict)
    ]

    def sum_key(records: list[dict[str, object]], key: str) -> float:
        return float(sum(float(row.get(key, 0.0)) for row in records))

    def mean_key(records: list[dict[str, object]], key: str) -> float:
        values = [float(row[key]) for row in records if row.get(key) is not None]
        return float(np.mean(values)) if values else 0.0

    selected = int(metadata.get("successor_placements", 0))
    initial_weak = next(
        (int(row["weak_pixels_before"]) for row in history if "weak_pixels_before" in row),
        0,
    )
    return {
        "allocation_waves": int(metadata.get("waves", 0)),
        "successor_placements": selected,
        "initial_weak_pixels": initial_weak,
        "candidate_count_total": int(sum_key(candidate_banks, "candidate_count")),
        "candidate_incidence_edges": int(sum_key(candidate_banks, "incidence_edges")),
        "candidate_compression": (
            float(initial_weak) / selected if selected else 0.0
        ),
        "coverage_detector_seconds": float(metadata.get("detector_seconds", 0.0)),
        "candidate_incidence_seconds": float(metadata.get("incidence_seconds", 0.0)),
        "candidate_selector_seconds": float(metadata.get("selector_seconds", 0.0)),
        "donor_selector_seconds": sum_key(funding, "selector_seconds"),
        "donor_candidate_pairs_mean": mean_key(funding, "candidate_pairs"),
        "donor_eligible_pairs_mean": mean_key(funding, "eligible_pairs"),
        "donor_local_merge_sse_mean": mean_key(funding, "local_merge_sse_mean"),
        "donor_local_delta_sse_mean": mean_key(funding, "local_delta_sse_mean"),
        "donor_local_area_color_sse_mean": mean_key(
            funding, "local_area_color_sse_mean"
        ),
    }


def _component_view(labels: np.ndarray, inside: np.ndarray) -> np.ndarray:
    view = np.zeros((*inside.shape, 3), dtype=np.float32)
    view[inside] = 0.04
    for label in range(1, int(labels.max(initial=0)) + 1):
        color = np.asarray(
            [
                ((label * 73) % 251) / 250.0,
                ((label * 151 + 31) % 251) / 250.0,
                ((label * 199 + 67) % 251) / 250.0,
            ],
            dtype=np.float32,
        )
        view[labels == label] = color
    return view


def _overlay_points(
    source: np.ndarray,
    inside: np.ndarray,
    groups: list[tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    view = source * inside[..., None].astype(np.float32)
    height, width = inside.shape
    for points, color in groups:
        for x_value, y_value in np.asarray(points).reshape(-1, 2):
            x, y = int(round(float(x_value))), int(round(float(y_value)))
            if not (0 <= x < width and 0 <= y < height):
                continue
            y0, y1 = max(0, y - 1), min(height, y + 2)
            x0, x1 = max(0, x - 1), min(width, x + 2)
            view[y0:y1, x0:x1] = color
    return view


def _save_crop_triplet(
    artifact_dir: Path,
    stem: str,
    source: np.ndarray,
    reconstruction: np.ndarray,
    target: np.ndarray,
    bounds: tuple[int, int, int, int],
    error_scale: float,
) -> None:
    absolute = np.repeat(
        np.clip(
            np.mean(np.abs(reconstruction.astype(np.float64) - target), axis=2)
            * error_scale,
            0.0,
            1.0,
        )[..., None],
        3,
        axis=2,
    ).astype(np.float32)
    h22.viz_utils._save_crop(artifact_dir / f"{stem}_source_crop.png", source, bounds)
    h22.viz_utils._save_crop(
        artifact_dir / f"{stem}_reconstruction_crop.png", reconstruction, bounds
    )
    h22.viz_utils._save_crop(artifact_dir / f"{stem}_error_crop.png", absolute, bounds)


def _projection_history(history: list[dict[str, object]]) -> list[dict[str, object]]:
    return [
        row["funding"]["projection"]
        for row in history
        if isinstance(row.get("funding"), dict)
        and isinstance(row["funding"].get("projection"), dict)
    ]


def _write_arm(
    output_root: Path,
    arm: str,
    method: dict[str, object],
    source: np.ndarray,
    inside: np.ndarray,
    geometry: dict[str, np.ndarray],
    args: argparse.Namespace,
    torch,
) -> dict[str, object]:
    artifact_dir = output_root / "artifacts" / arm
    artifact_dir.mkdir(parents=True, exist_ok=False)
    field = _pure_field(method["field"])
    if field.n != CAPACITY:
        raise RuntimeError(f"{arm} has {field.n} rows before persistence")
    expected = _render(field, inside.shape, args, torch).detach().cpu().numpy()
    field_path = artifact_dir / "field.gaussian.npz"
    field.save(str(field_path))
    with np.load(field_path) as stored:
        field_keys = sorted(stored.files)
    decoded = GaussianField.load(str(field_path), device=args.device)
    in_memory_state_sha256 = _field_state_sha256(field)
    decoded_state_sha256 = _field_state_sha256(decoded)
    decoded_state_max_abs = _field_state_max_abs(field, decoded)
    if (
        decoded_state_sha256 != in_memory_state_sha256
        or decoded_state_max_abs != 0.0
    ):
        raise RuntimeError(f"{arm} four-array decoded state differs from the persisted input")
    cold = _render(decoded, inside.shape, args, torch).detach().cpu().numpy()
    repeated = _render(decoded, inside.shape, args, torch).detach().cpu().numpy()
    maintained_parity = float(
        np.max(np.abs(expected.astype(np.float64) - cold.astype(np.float64)))
    )
    repeated_parity = float(
        np.max(np.abs(repeated.astype(np.float64) - cold.astype(np.float64)))
    )
    coverage = _coverage(decoded, inside.shape, args, torch)
    target = source * inside[..., None].astype(np.float32)
    outside_reconstruction = np.abs(cold[~inside])
    outside_coverage = np.abs(coverage[~inside])
    means = decoded.means.detach().cpu().numpy()
    ix = np.rint(means[:, 0]).astype(np.int64)
    iy = np.rint(means[:, 1]).astype(np.int64)
    in_bounds = (
        (ix >= 0)
        & (ix < inside.shape[1])
        & (iy >= 0)
        & (iy < inside.shape[0])
    )
    centres_inside = np.zeros(decoded.n, dtype=bool)
    centres_inside[in_bounds] = inside[iy[in_bounds], ix[in_bounds]]
    containment_pass = bool(
        centres_inside.all()
        and float(outside_reconstruction.max(initial=0.0)) <= CONTAINMENT_LIMIT
        and float(outside_coverage.max(initial=0.0)) <= CONTAINMENT_LIMIT
    )
    if not containment_pass:
        raise RuntimeError(f"{arm} failed exact mask containment")
    coefficient_abs_max = float(decoded.colors.detach().abs().max().cpu())
    if not math.isfinite(coefficient_abs_max) or coefficient_abs_max > COEFFICIENT_ABS_LIMIT:
        raise RuntimeError(
            f"{arm} coefficient bound differs: {coefficient_abs_max:.9g}"
        )
    full_metrics, foreground = h29._metric_domains(cold, source, inside, args)
    regional = h31._regional_quality(cold, target, inside, geometry["sdf"])
    detail = h31._detail_metrics(cold, target, geometry["sdf"], args, torch)
    hair = _masked_region_quality(
        cold, target, _crop_region(inside, HAIR_CROP_BOUNDS), "hair"
    )
    boundary_crop = _masked_region_quality(
        cold,
        target,
        _crop_region(inside, BOUNDARY_CROP_BOUNDS),
        "boundary_crop",
    )
    holes = h31._coverage_metrics(
        coverage,
        inside,
        geometry["sdf"],
        geometry["ridge"],
        geometry["isotropic_unreachable"],
    )
    coverage_record = _coverage_telemetry(coverage, inside)
    component_labels = coverage_record.pop("coverage_component_labels")
    metadata = method.get("metadata", {})
    history = method.get("history", [])
    telemetry = _method_telemetry(history, metadata)
    bounds = h29._save_visuals(
        artifact_dir,
        source,
        cold,
        inside,
        "masked_foreground",
        args.error_scale,
    )
    save_error_heatmap(
        str(artifact_dir / "absolute_error.png"),
        cold - target,
        scale=args.error_scale,
    )
    _save_crop_triplet(
        artifact_dir,
        "hair",
        target,
        cold,
        target,
        HAIR_CROP_BOUNDS,
        args.error_scale,
    )
    _save_crop_triplet(
        artifact_dir,
        "boundary",
        target,
        cold,
        target,
        BOUNDARY_CROP_BOUNDS,
        args.error_scale,
    )
    coverage_debt = np.maximum(0.0, COVERAGE_THRESHOLD - coverage) * inside
    debt_view = np.zeros((*inside.shape, 3), dtype=np.float32)
    debt_view[inside] = 0.03
    debt_view[..., 0] += np.clip(coverage_debt / COVERAGE_THRESHOLD, 0.0, 1.0)
    save_image(str(artifact_dir / "coverage_debt.png"), debt_view)
    initial_labels = metadata.get("initial_component_labels")
    if initial_labels is None:
        initial_labels = component_labels
    save_image(
        str(artifact_dir / "components.png"),
        _component_view(np.asarray(initial_labels), inside),
    )
    placements = metadata.get("placements", [])
    placement_points = np.asarray(
        [row["mean"] for row in placements], dtype=np.float32
    ).reshape(-1, 2)
    funding = _funding_records(history)
    donor_keep = np.asarray(
        [point for row in funding for point in row.get("donor_keep_means", [])],
        dtype=np.float32,
    ).reshape(-1, 2)
    donor_absorbed = np.asarray(
        [point for row in funding for point in row.get("donor_absorbed_means", [])],
        dtype=np.float32,
    ).reshape(-1, 2)
    save_image(
        str(artifact_dir / "placement.png"),
        _overlay_points(
            source,
            inside,
            [(placement_points, np.asarray([1.0, 0.85, 0.0], dtype=np.float32))],
        ),
    )
    save_image(
        str(artifact_dir / "donors.png"),
        _overlay_points(
            source,
            inside,
            [
                (donor_keep, np.asarray([0.0, 1.0, 1.0], dtype=np.float32)),
                (donor_absorbed, np.asarray([1.0, 0.0, 1.0], dtype=np.float32)),
            ],
        ),
    )
    save_image(
        str(artifact_dir / "unit_coverage.png"),
        np.clip(coverage / max(float(coverage.max(initial=0.0)), 1e-8), 0.0, 1.0),
    )
    h22.viz_utils._save_crop(
        artifact_dir / "source_crop.png", target, tuple(int(v) for v in bounds)
    )
    h22.viz_utils._save_crop(
        artifact_dir / "reconstruction_crop.png", cold, tuple(int(v) for v in bounds)
    )
    absolute = np.repeat(
        np.clip(np.mean(np.abs(cold - target), axis=2)[..., None] * args.error_scale, 0.0, 1.0),
        3,
        axis=2,
    )
    h22.viz_utils._save_crop(
        artifact_dir / "error_crop.png", absolute, tuple(int(v) for v in bounds)
    )
    np.savez_compressed(
        artifact_dir / "analysis.npz",
        mask=inside,
        sdf=geometry["sdf"],
        reconstruction_raw=cold,
        error_raw=cold - target,
        unit_coverage=coverage,
        coverage_debt=coverage_debt,
        component_labels=np.asarray(initial_labels),
        placement_means=placement_points,
        donor_keep_means=donor_keep,
        donor_absorbed_means=donor_absorbed,
        hair_crop_bounds=np.asarray(HAIR_CROP_BOUNDS, dtype=np.int32),
        boundary_crop_bounds=np.asarray(BOUNDARY_CROP_BOUNDS, dtype=np.int32),
    )
    _write_json(
        artifact_dir / "fit_history.json",
        {"schema": REPORT_SCHEMA, "history": history},
    )
    _write_json(
        artifact_dir / "projection_history.json",
        {"schema": REPORT_SCHEMA, "history": _projection_history(history)},
    )
    json_metadata = {
        key: value
        for key, value in metadata.items()
        if key not in {"initial_component_labels", "final_coverage"}
    }
    _write_json(
        artifact_dir / "geometry_history.json",
        {
            "schema": REPORT_SCHEMA,
            "arm": arm,
            "geometry_frozen_after_placement": True,
            "metadata": json_metadata,
        },
    )
    _write_json(
        artifact_dir / "candidate_history.json",
        {"schema": REPORT_SCHEMA, "history": history},
    )
    scales = decoded.scales().detach().cpu().numpy()
    row: dict[str, object] = {
        "schema": REPORT_SCHEMA,
        "status": "diagnostic",
        "image": "C0001",
        "seed": args.seed,
        "arm": arm,
        "renderer": "cuda_additive",
        "target_gaussians": CAPACITY,
        "n_gaussians": decoded.n,
        "artifact_dir": str(artifact_dir.relative_to(output_root)),
        "field_file_sha256": h22.report_utils._sha256(field_path),
        "field_file_bytes": field_path.stat().st_size,
        "in_memory_field_state_sha256": in_memory_state_sha256,
        "decoded_field_state_sha256": decoded_state_sha256,
        "decoded_field_state_max_abs": decoded_state_max_abs,
        "field_npz_keys": field_keys,
        "four_array_endpoint_exact": set(field_keys) == FOUR_ARRAY_KEYS,
        "masked_mse": foreground["masked_mse"],
        "foreground_mse": foreground["masked_mse"],
        "psnr_db": foreground["psnr_db"],
        "ms_ssim": foreground["ms_ssim"],
        "lpips": foreground["lpips"],
        "ssim": foreground["ssim"],
        "full_psnr_db": full_metrics["psnr_db"],
        "pixel_rmse_max": foreground["artifact_pixel_rmse_max"],
        "patch7_rmse_max": foreground["artifact_patch_rmse_max_7"],
        "maintained_render_parity_max_abs": maintained_parity,
        "repeated_render_parity_max_abs": repeated_parity,
        "centres_inside_mask": int(centres_inside.sum()),
        "centres_outside_mask": int((~centres_inside).sum()),
        "unit_coverage_outside_abs_max": float(outside_coverage.max(initial=0.0)),
        "reconstruction_outside_abs_max": float(
            outside_reconstruction.max(initial=0.0)
        ),
        "containment_pass": containment_pass,
        "scale_min_px": float(scales.min()),
        "scale_max_px": float(scales.max()),
        "coefficient_abs_max": coefficient_abs_max,
        "method_seconds": float(method.get("seconds", 0.0)),
        **regional,
        **detail,
        **hair,
        **boundary_crop,
        **holes,
        **coverage_record,
        **telemetry,
    }
    _write_json(artifact_dir / "row.json", row)
    return row


def _gate_row(row: dict[str, object], control: dict[str, object]) -> dict[str, bool]:
    return {
        "exact_count": row.get("n_gaussians") == CAPACITY,
        "four_array_payload": row.get("four_array_endpoint_exact") is True,
        "zero_raw_holes": row.get("raw_hole_pixels") == 0,
        "zero_weak_pixels": row.get("coverage_lt_005_pixels") == 0,
        "support_outside": float(row.get("unit_coverage_outside_abs_max", float("inf")))
        <= CONTAINMENT_LIMIT,
        "reconstruction_outside": float(
            row.get("reconstruction_outside_abs_max", float("inf"))
        )
        <= CONTAINMENT_LIMIT,
        "field_parity": float(row.get("maintained_render_parity_max_abs", float("inf")))
        <= RENDER_PARITY_LIMIT,
        "boundary_improved": float(row.get("boundary_le4_psnr_db", -float("inf")))
        > float(control["boundary_le4_psnr_db"]),
        "hair_improved": float(row.get("hair_psnr_db", -float("inf")))
        > float(control["hair_psnr_db"]),
        "interior_floor": float(row.get("interior_gt4_psnr_db", -float("inf")))
        >= INTERIOR_PSNR_FLOOR_DB,
    }


def _decision(
    rows: list[dict[str, object]],
    attempts: list[dict[str, object]],
) -> dict[str, object]:
    by_arm = {str(row["arm"]): row for row in rows}
    matrix_complete = (
        [str(row.get("arm")) for row in attempts] == list(ARMS)
        and all(row.get("status") == "ok" for row in attempts)
        and set(by_arm) == set(ARMS)
    )
    control = by_arm.get(ARMS[0])
    gates: dict[str, dict[str, bool]] = {}
    if control is not None:
        for arm in ARMS[1:]:
            row = by_arm.get(arm)
            if row is not None:
                gates[arm] = _gate_row(row, control)
                row["acceptance_clauses"] = gates[arm]
                row["acceptance_pass"] = all(gates[arm].values())
    passing = (
        [
            by_arm[arm]
            for arm in ARMS[1:]
            if bool(by_arm[arm].get("acceptance_pass"))
        ]
        if matrix_complete
        else []
    )
    passing.sort(key=lambda row: (-float(row["psnr_db"]), ARMS.index(str(row["arm"]))))
    selected = passing[0] if passing else None
    tradeoffs = [by_arm[arm] for arm in ARMS[1:] if arm in by_arm]
    tradeoffs.sort(
        key=lambda row: (
            int(row["coverage_lt_005_pixels"]),
            -float(row["boundary_le4_psnr_db"]),
            -float(row["hair_psnr_db"]),
            -float(row["psnr_db"]),
            ARMS.index(str(row["arm"])),
        )
    )
    best_tradeoff = tradeoffs[0] if tradeoffs else None
    return {
        "schema": REPORT_SCHEMA,
        "status": "diagnostic",
        "complete": matrix_complete,
        "all_arms_succeeded": matrix_complete,
        "formal_claim_ready": False,
        "selected_arm": None if selected is None else selected["arm"],
        "selected_method": selected is not None,
        "selection_reason": (
            "highest foreground PSNR among arms passing all frozen clauses"
            if selected is not None
            else "matrix incomplete; no selection authorized"
            if not matrix_complete
            else "no arm passes every frozen count, coverage, containment, boundary, hair, and interior clause"
        ),
        "best_tradeoff_arm": None if best_tradeoff is None else best_tradeoff["arm"],
        "control_arm": ARMS[0],
        "control_boundary_psnr_db": None if control is None else control["boundary_le4_psnr_db"],
        "control_hair_psnr_db": None if control is None else control["hair_psnr_db"],
        "interior_psnr_floor_db": INTERIOR_PSNR_FLOOR_DB,
        "gates": gates,
        "limits": [
            "one exposed Janelle raster, seed, and device",
            "the HIER-031 input field was produced by a dirty-source sequential diagnostic",
            "development evidence only; no held-out, rate, native-resolution, or default claim",
            "the component/cover relationship is a task-specific recombination of known components",
        ],
    }


def _write_tables(output_root: Path, rows: list[dict[str, object]]) -> None:
    ordered = sorted(rows, key=lambda row: ARMS.index(str(row["arm"])))
    _write_json(
        output_root / "metrics.json",
        {"schema": REPORT_SCHEMA, "status": "diagnostic", "rows": ordered},
    )
    with (output_root / "metrics.jsonl").open("w", encoding="utf-8") as stream:
        for row in ordered:
            stream.write(json.dumps(row, sort_keys=True, default=str) + "\n")
    fields = sorted({key for row in ordered for key in row})
    with (output_root / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in ordered:
            writer.writerow(
                {
                    key: (
                        str(row.get(key))
                        if isinstance(row.get(key), (dict, list, tuple))
                        else row.get(key)
                    )
                    for key in fields
                }
            )


def _snapshot_sources(output_root: Path) -> list[dict[str, object]]:
    paths = (
        Path(__file__).resolve(),
        ROOT / "scripts/check_report_bundle.py",
        ROOT / "scripts/experiments/hier031_exact7k_masked_boundary_detail.py",
        ROOT / "src/structsplat/fit.py",
        ROOT / "src/structsplat/endpoint_appearance_projection.py",
        ROOT / "src/structsplat/mask.py",
        ROOT / "tests/test_hier032_coverage_debt_refinement.py",
        ROOT / "tasks/HIER-032-coverage-debt-refinement.md",
        ROOT / "docs/research/2026-08-12-hier032-coverage-debt-portfolio.md",
    )
    records: list[dict[str, object]] = []
    for source in paths:
        destination = output_root / "source_snapshot" / source.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        records.append(
            {
                "repository_path": str(source.relative_to(ROOT)),
                "snapshot_path": str(destination.relative_to(output_root)),
                "bytes": destination.stat().st_size,
                "sha256": h22.report_utils._sha256(destination),
            }
        )
    return records


def _method_for_arm(
    arm: str,
    source: np.ndarray,
    inside: np.ndarray,
    geometry: dict[str, np.ndarray],
    args: argparse.Namespace,
    torch,
) -> dict[str, object]:
    started = time.perf_counter()
    base_path = args.base_bundle / BASE_FIELD_REL
    base = _pure_field(GaussianField.load(str(base_path), device=args.device))
    target = source * inside[..., None].astype(np.float32)
    if arm == ARMS[0]:
        return {
            "field": base,
            "history": [],
            "metadata": {
                "operator": "frozen HIER-031 selected control",
                "base_field_sha256": BASE_FIELD_SHA256,
                "placement_mode": "none",
                "donor_mode": "none",
                "waves": 0,
                "successor_placements": 0,
                "placements": [],
            },
            "seconds": time.perf_counter() - started,
        }
    if arm == ARMS[1]:
        field, history, metadata = _close_coverage(
            base,
            target,
            source,
            inside,
            geometry["sdf"],
            args,
            torch,
            placement_mode="fallback",
            donor_mode="hier031",
        )
    elif arm == ARMS[2]:
        field, history, metadata = _close_coverage(
            base,
            target,
            source,
            inside,
            geometry["sdf"],
            args,
            torch,
            placement_mode="set_cover",
            donor_mode="hier031",
        )
    elif arm == ARMS[3]:
        field, history, metadata = _close_coverage(
            base,
            target,
            source,
            inside,
            geometry["sdf"],
            args,
            torch,
            placement_mode="set_cover",
            donor_mode="contribution",
        )
        comparison_path = args.out / "artifacts" / ARMS[2] / "fit_history.json"
        if not comparison_path.is_file():
            raise RuntimeError(
                "contribution-aware arm requires the successful set-cover comparator history"
            )
        old_history = json.loads(comparison_path.read_text(encoding="utf-8"))["history"]
        old_first = next(
            row["placement_digest"]
            for row in old_history
            if row.get("status") == "allocated"
        )
        new_first = next(
            row["placement_digest"]
            for row in history
            if row.get("status") == "allocated"
        )
        if old_first != new_first:
            raise RuntimeError(
                "contribution-aware arm changed the frozen first-wave placement"
            )
        metadata["first_wave_matches_hier031_funding_arm"] = True
    elif arm == ARMS[4]:
        covered, coverage_history, coverage_metadata = _close_coverage(
            base,
            target,
            source,
            inside,
            geometry["sdf"],
            args,
            torch,
            placement_mode="set_cover",
            donor_mode="contribution",
        )
        comparison_path = args.out / "artifacts" / ARMS[3] / "geometry_history.json"
        if not comparison_path.is_file():
            raise RuntimeError(
                "detail arm requires the successful contribution-aware coverage comparator"
            )
        comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
        expected_digest = comparison.get("metadata", {}).get("placement_digest")
        if expected_digest != coverage_metadata["placement_digest"]:
            raise RuntimeError(
                "detail arm changed the frozen contribution-aware coverage placement"
            )
        field, detail_history, detail_metadata = _append_boundary_detail(
            covered,
            target,
            source,
            inside,
            geometry["sdf"],
            args,
            torch,
        )
        history = coverage_history + detail_history
        metadata = {
            **coverage_metadata,
            "operator": "coverage closure followed by one frozen 128-row boundary high-pass batch",
            "placement_mode": "set_cover_then_boundary_highpass",
            "waves": int(coverage_metadata["waves"]) + 1,
            "successor_placements": int(coverage_metadata["successor_placements"])
            + DETAIL_ROWS,
            "placements": list(coverage_metadata["placements"])
            + list(detail_metadata["placements"]),
            "selector_seconds": float(coverage_metadata["selector_seconds"])
            + float(detail_metadata["selector_seconds"]),
            "donor_keep_rows": list(coverage_metadata["donor_keep_rows"])
            + list(detail_metadata["donor_keep_rows"]),
            "donor_absorbed_rows": list(coverage_metadata["donor_absorbed_rows"])
            + list(detail_metadata["donor_absorbed_rows"]),
            "final_coverage": detail_metadata["final_coverage"],
            "coverage_placement_digest": coverage_metadata["placement_digest"],
            "coverage_placement_matches_contribution_arm": True,
            "detail_placement_digest": detail_metadata["placement_digest"],
        }
    else:
        raise ValueError(f"unknown HIER-032 arm {arm!r}")
    metadata.setdefault(
        "operator",
        f"{metadata['placement_mode']} placement with {metadata['donor_mode']} donor funding",
    )
    return {
        "field": field,
        "history": history,
        "metadata": metadata,
        "seconds": time.perf_counter() - started,
    }


def _fmt(value: object, digits: int = 4) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{digits}f}"
    return escape(str(value))


def _write_report(
    output_root: Path,
    rows: list[dict[str, object]],
    attempts: list[dict[str, object]],
    decision: dict[str, object],
) -> None:
    ordered = sorted(rows, key=lambda row: ARMS.index(str(row["arm"])))
    row_html: list[str] = []
    cards: list[str] = []
    for row in ordered:
        arm = str(row["arm"])
        artifact = str(row["artifact_dir"])
        row_html.append(
            "<tr>"
            f"<td><code>{escape(arm)}</code></td>"
            f"<td>{_fmt(row['n_gaussians'], 0)}</td>"
            f"<td>{_fmt(row['psnr_db'])}</td>"
            f"<td>{_fmt(row['boundary_le4_psnr_db'])}</td>"
            f"<td>{_fmt(row['hair_psnr_db'])}</td>"
            f"<td>{_fmt(row['interior_gt4_psnr_db'])}</td>"
            f"<td>{_fmt(row['ms_ssim'], 6)}</td>"
            f"<td>{_fmt(row['lpips'], 6)}</td>"
            f"<td>{_fmt(row['coverage_lt_005_pixels'], 0)}</td>"
            f"<td>{_fmt(row['coverage_deficit_mass'], 6)}</td>"
            f"<td>{_fmt(row['successor_placements'], 0)}</td>"
            f"<td>{_fmt(row.get('acceptance_pass'))}</td>"
            "</tr>"
        )
        cards.append(
            f"<section><h3><code>{escape(arm)}</code></h3>"
            f"<p>PSNR {_fmt(row['psnr_db'])} dB; boundary {_fmt(row['boundary_le4_psnr_db'])} dB; "
            f"hair {_fmt(row['hair_psnr_db'])} dB; interior {_fmt(row['interior_gt4_psnr_db'])} dB. "
            f"Weak pixels {_fmt(row['coverage_lt_005_pixels'], 0)} in "
            f"{_fmt(row['coverage_lt_005_components'], 0)} components; deficit "
            f"{_fmt(row['coverage_deficit_mass'], 6)}.</p>"
            "<div class='grid'>"
            f"<figure><a href='{artifact}/source.png'><img src='{artifact}/source.png'></a><figcaption>source / target</figcaption></figure>"
            f"<figure><a href='{artifact}/reconstruction.png'><img src='{artifact}/reconstruction.png'></a><figcaption>reconstruction</figcaption></figure>"
            f"<figure><a href='{artifact}/error.png'><img src='{artifact}/error.png'></a><figcaption>absolute error</figcaption></figure>"
            f"<figure><a href='{artifact}/coverage_debt.png'><img src='{artifact}/coverage_debt.png'></a><figcaption>coverage debt</figcaption></figure>"
            f"<figure><a href='{artifact}/components.png'><img src='{artifact}/components.png'></a><figcaption>detected components</figcaption></figure>"
            f"<figure><a href='{artifact}/placement.png'><img src='{artifact}/placement.png'></a><figcaption>successor placements</figcaption></figure>"
            f"<figure><a href='{artifact}/donors.png'><img src='{artifact}/donors.png'></a><figcaption>donor pairs</figcaption></figure>"
            f"<figure><a href='{artifact}/unit_coverage.png'><img src='{artifact}/unit_coverage.png'></a><figcaption>unit coverage</figcaption></figure>"
            "</div><div class='grid crops'>"
            f"<figure><a href='{artifact}/source_crop.png'><img src='{artifact}/source_crop.png'></a><figcaption>worst-region target</figcaption></figure>"
            f"<figure><a href='{artifact}/reconstruction_crop.png'><img src='{artifact}/reconstruction_crop.png'></a><figcaption>worst-region reconstruction</figcaption></figure>"
            f"<figure><a href='{artifact}/error_crop.png'><img src='{artifact}/error_crop.png'></a><figcaption>worst-region error</figcaption></figure>"
            f"<figure><a href='{artifact}/hair_source_crop.png'><img src='{artifact}/hair_source_crop.png'></a><figcaption>fixed hair target</figcaption></figure>"
            f"<figure><a href='{artifact}/hair_reconstruction_crop.png'><img src='{artifact}/hair_reconstruction_crop.png'></a><figcaption>fixed hair reconstruction</figcaption></figure>"
            f"<figure><a href='{artifact}/hair_error_crop.png'><img src='{artifact}/hair_error_crop.png'></a><figcaption>fixed hair error</figcaption></figure>"
            f"<figure><a href='{artifact}/boundary_source_crop.png'><img src='{artifact}/boundary_source_crop.png'></a><figcaption>fixed boundary target</figcaption></figure>"
            f"<figure><a href='{artifact}/boundary_reconstruction_crop.png'><img src='{artifact}/boundary_reconstruction_crop.png'></a><figcaption>fixed boundary reconstruction</figcaption></figure>"
            f"<figure><a href='{artifact}/boundary_error_crop.png'><img src='{artifact}/boundary_error_crop.png'></a><figcaption>fixed boundary error</figcaption></figure>"
            "</div>"
            f"<p><a href='{artifact}/field.gaussian.npz'>field</a> · "
            f"<a href='{artifact}/row.json'>row</a> · "
            f"<a href='{artifact}/fit_history.json'>allocation history</a> · "
            f"<a href='{artifact}/analysis.npz'>analysis arrays</a></p></section>"
        )
    attempt_html = "".join(
        "<li>"
        f"<code>{escape(str(row['arm']))}</code>: {escape(str(row['status']))}"
        + (
            f" — {escape(str(row.get('error')))}"
            if row.get("status") != "ok"
            else ""
        )
        + "</li>"
        for row in attempts
    )
    selected = decision.get("selected_arm") or "none"
    html = f"""<!doctype html>
<html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>
<title>HIER-032 coverage-debt refinement</title>
<style>
body{{font-family:system-ui,sans-serif;max-width:1500px;margin:auto;padding:24px;background:#101217;color:#e8ecf3}}a{{color:#83c9ff}}code{{color:#ffd580}}table{{border-collapse:collapse;width:100%;font-size:.86rem}}th,td{{border:1px solid #394150;padding:6px;text-align:right}}th:first-child,td:first-child{{text-align:left}}section{{margin:30px 0;padding:18px;background:#181c24;border-radius:10px}}.callout{{padding:14px;background:#202839;border-left:5px solid #69b7ff}}.warn{{border-color:#ffba63}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:12px}}figure{{margin:0}}img{{width:100%;height:auto;background:#000}}figcaption{{font-size:.8rem;color:#b5bdc9}}.crops img{{image-rendering:auto}}
</style></head><body>
<h1>HIER-032 — Coverage-debt refinement at exact N=7,000</h1>
<div class='callout'><strong>Disposition:</strong> selected method <code>{escape(str(selected))}</code>. {escape(str(decision['selection_reason']))}</div>
<p>This clean-commit, prospectively reviewed development run continues from the hash-bound HIER-031 selected field. It changes neither the 7,000-row count nor the four-array payload, mask, renderer equation, or maintained defaults.</p>
<p>Known foundations: pixel-error densification (arXiv:2404.06109), cancellation-resistant detail detection (arXiv:2404.10484), and coverage/set-cover medial selection (arXiv:2110.00965). Their task-specific relationship is tested here without a novelty claim; full citations are preserved in <code>research_context.md</code>.</p>
<p>Raw artifacts: <a href='manifest.json'>manifest</a> · <a href='config.json'>config</a> · <a href='protocol.json'>protocol</a> · <a href='research_context.md'>research context</a> · <a href='metrics.json'>JSON</a> · <a href='metrics.jsonl'>JSONL</a> · <a href='metrics.csv'>CSV</a> · <a href='decision.json'>decision</a> · <a href='attempts.json'>execution ledger</a>.</p>
<h2>Frozen acceptance matrix</h2>
<table><tr><th>arm</th><th>N</th><th>PSNR</th><th>boundary PSNR</th><th>hair PSNR</th><th>interior PSNR</th><th>MS-SSIM</th><th>LPIPS</th><th>weak pixels</th><th>deficit</th><th>placements</th><th>pass</th></tr>{''.join(row_html)}</table>
<h2>Execution-error ledger</h2><ul>{attempt_html}</ul>
<h2>Visual and telemetry cards</h2>{''.join(cards)}
<h2>Limitations</h2><ul>{''.join(f'<li>{escape(str(item))}</li>' for item in decision['limits'])}</ul>
</body></html>"""
    (output_root / "index.html").write_text(html, encoding="utf-8")


def _write_manifest(output_root: Path) -> None:
    records = []
    for path in sorted(output_root.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            records.append(
                {
                    "path": str(path.relative_to(output_root)),
                    "bytes": path.stat().st_size,
                    "sha256": h22.report_utils._sha256(path),
                }
            )
    _write_json(
        output_root / "manifest.json",
        {"schema": REPORT_SCHEMA, "status": "diagnostic", "files": records},
    )


def main() -> None:
    args = _parser().parse_args()
    _validate_args(args)
    if args.print_protocol_digest:
        print(_protocol_digest())
        return
    if args.resume:
        raise SystemExit("formal HIER-032 evidence does not resume or repair an executed bundle")
    if (args.out / "COMPLETED").is_file():
        raise SystemExit(f"completed HIER-032 bundle is immutable: {args.out}")
    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit(f"formal HIER-032 output must be a new empty directory: {args.out}")
    args.out.mkdir(parents=True, exist_ok=True)

    import torch

    if not torch.cuda.is_available():
        raise SystemExit("frozen HIER-032 development run requires CUDA")
    gpu_name = torch.cuda.get_device_name(torch.device(args.device))
    if gpu_name != EXPECTED_GPU_NAME:
        raise SystemExit(
            f"frozen HIER-032 development run requires {EXPECTED_GPU_NAME!r}, got {gpu_name!r}"
        )
    git_record = h22._git_record()
    if git_record.get("dirty") is not False:
        raise SystemExit("formal HIER-032 evidence requires a clean source commit")
    if git_record.get("branch") != EXPECTED_SOURCE_BRANCH:
        raise SystemExit(
            "formal HIER-032 evidence requires the clean named source branch "
            f"{EXPECTED_SOURCE_BRANCH!r}, got {git_record.get('branch')!r}"
        )
    source, inside, raster = h22.report_utils._load_evaluation_raster(
        args.image, args.mask, max_side=args.max_side, mask_threshold=0.5
    )
    if inside is None or source.shape[:2] != EVALUATION_SHAPE:
        raise RuntimeError(
            f"expected masked evaluation raster {EVALUATION_SHAPE}, got {source.shape}"
        )
    if (raster["original_height"], raster["original_width"]) != NATIVE_SHAPE:
        raise RuntimeError(f"native Janelle shape differs: {raster!r}")
    _, geometry = h31._feasibility_audit(inside)
    input_dir = args.out / "input"
    input_dir.mkdir()
    save_image(str(input_dir / "source.png"), source)
    save_image(str(input_dir / "mask.png"), inside.astype(np.float32))
    save_image(
        str(input_dir / "foreground_black_matted.png"),
        source * inside[..., None].astype(np.float32),
    )
    snapshots = _snapshot_sources(args.out)
    _write_json(args.out / "environment.json", h22._environment(torch))
    _write_json(
        args.out / "protocol.json",
        {"digest": _protocol_digest(), "protocol": PROTOCOL},
    )
    _write_json(
        args.out / "config.json",
        {
            "schema": REPORT_SCHEMA,
            "status": "diagnostic",
            "command": _command(),
            "git": git_record,
            "source_snapshots": snapshots,
            "arguments": vars(args),
            "source": {
                "path": str(args.image.resolve()),
                "sha256": SOURCE_SHA256,
                "native_shape": list(NATIVE_SHAPE),
            },
            "mask": {
                "path": str(args.mask.resolve()),
                "sha256": MASK_SHA256,
                "active_pixels": int(inside.sum()),
                "margin_px": MASK_MARGIN,
            },
            "raster": raster,
            "protocol_digest": _protocol_digest(),
            "base_field": {
                "path": str((args.base_bundle / BASE_FIELD_REL).resolve()),
                "sha256": BASE_FIELD_SHA256,
                "decision_sha256": BASE_DECISION_SHA256,
            },
            "formal_source_clean": True,
            "formal_claim_ready": False,
        },
    )
    (args.out / "NATURAL_STARTED").write_text(
        "HIER-032 source, mask, and HIER-031 field were hash-bound after protocol review.\n",
        encoding="utf-8",
    )
    (args.out / "research_context.md").write_text(
        "# HIER-032 research context\n\n"
        "This task combines pixel-error densification, absolute high-pass/detail detection, and "
        "coverage/set-cover selection under a fixed exact-count donor constraint. These are known "
        "components; only their source-bound relationship is tested.\n\n"
        "- https://arxiv.org/abs/2404.06109\n"
        "- https://arxiv.org/abs/2404.10484\n"
        "- https://arxiv.org/abs/2110.00965\n",
        encoding="utf-8",
    )
    with (args.out / "git.diff").open("wb") as stream:
        subprocess.run(
            ["git", "diff", "--binary", "HEAD"],
            cwd=ROOT,
            check=False,
            stdout=stream,
        )

    rows: list[dict[str, object]] = []
    attempts: list[dict[str, object]] = []
    for arm in ARMS:
        started = time.perf_counter()
        try:
            method = _method_for_arm(arm, source, inside, geometry, args, torch)
            row = _write_arm(
                args.out, arm, method, source, inside, geometry, args, torch
            )
            rows.append(row)
            attempts.append(
                {
                    "arm": arm,
                    "status": "ok",
                    "elapsed_seconds": time.perf_counter() - started,
                }
            )
        except Exception as exc:
            attempts.append(
                {
                    "arm": arm,
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}"[:4000],
                    "elapsed_seconds": time.perf_counter() - started,
                }
            )
            if not args.quiet:
                print(f"{arm}: {type(exc).__name__}: {exc}", file=sys.stderr)
        finally:
            _write_tables(args.out, rows)
            _write_json(
                args.out / "attempts.json",
                {"schema": REPORT_SCHEMA, "status": "diagnostic", "attempts": attempts},
            )
            torch.cuda.empty_cache()

    decision = _decision(rows, attempts)
    # _decision attaches frozen gate receipts to successful rows; synchronize every projection.
    for row in rows:
        _write_json(args.out / str(row["artifact_dir"]) / "row.json", row)
    _write_tables(args.out, rows)
    _write_json(args.out / "decision.json", decision)
    _write_report(args.out, rows, attempts, decision)
    (args.out / "COMPLETED").write_text(
        "HIER-032 five-arm development protocol attempted; bundle is immutable.\n",
        encoding="utf-8",
    )
    _write_manifest(args.out)
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
