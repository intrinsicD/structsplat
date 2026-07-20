"""Native dynamic-support recovery engine for BENCH-009 Stage 1.

The immediate scientific runner owns candidate selection and the step-0 solve.  This module starts
from that *frozen* state, verifies its exact CPU state/packet/render/support/metric hashes, and then
runs the preregistered fresh-Adam recovery trajectory.  It deliberately has no selection code and
does not import transient scientific-runner internals.

There are eight immediate action labels but only four matched-six-DOF actions enter the evidence
analyzer.  The other four causal-capacity actions and the no-action control still receive complete
raw recovery records.  Step 0 is a logical link backed by the one required verification rerender;
only steps 20 and 100 are newly rendered recovery checkpoints.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import shlex
import sys
import time
from typing import Any, Callable, Iterable, Mapping, Sequence

import torch

from benchmarks.residual_tangent_auction_matrixfree import tensor_sha256
from benchmarks.residual_tangent_auction_stage1_analysis import (
    CROSS_FIT_SCOPE,
    MATCHED_LEDGER,
    RECORD_KIND as EVIDENCE_RECORD_KIND,
    SCHEMA_ID as EVIDENCE_SCHEMA_ID,
    stable_cell_id,
)
from benchmarks.residual_tangent_auction_stage1_core import (
    BirthGeometry,
    checkerboard_fold_masks,
    freeze_pixel_support,
    make_no_opacity_parameterization,
    render_normalized,
)
from benchmarks.residual_tangent_auction_stage1_runner import field_sha256
from structsplat.gaussians import GaussianField


PROTOCOL = "BENCH-009 Stage-1 native dynamic-support recovery v1"
RAW_CHECKPOINT_SCHEMA = "bench009.stage1.recovery-checkpoint.v1"
RAW_TRAJECTORY_SCHEMA = "bench009.stage1.recovery-trajectory.v1"
RAW_CHECKPOINT_KIND = "recovery_checkpoint"
RAW_TRAJECTORY_KIND = "recovery_trajectory_terminal"

ACTION_NAMES = (
    "full_j",
    "j_plus_affine6",
    "j_plus_carrier6",
    "j_plus_birth2_rgb6",
    "tangent6",
    "affine6",
    "carrier6",
    "birth2_rgb6",
)
MATCHED_ACTIONS = ("tangent6", "affine6", "carrier6", "birth2_rgb6")
BASELINE_ACTION = "no_action"
PACKET_KINDS = ("none", "affine", "carrier", "birth")

PRIMARY_DAMPING = 1e-3
PRIMARY_RCOND = 1e-5
TRUST_RADII = (0.25, 0.75)
RECOVERY_STEPS = 100
LOGICAL_CHECKPOINTS = (0, 20, 100)
NEW_RENDER_CHECKPOINTS = (20, 100)
EXPECTED_TRAJECTORIES = 816
EXPECTED_LOGICAL_RECORDS = 2_448
EXPECTED_NEW_RENDER_RECORDS = 1_632

LR_MEANS = 5e-2
LR_LOG_SCALES = 3e-2
LR_ROTATIONS = 1e-2
LR_BASE_RGB = 3e-2
LR_PACKET_RGB = 3e-2
ADAM_BETAS = (0.9, 0.999)
ADAM_EPS = 1e-8
LOG_SCALE_MIN = math.log(0.35)
LOG_SCALE_MAX = math.log(64.0)

SOURCE_PATHS = (
    "benchmarks/residual_tangent_auction_stage1_recovery.py",
    "benchmarks/residual_tangent_auction_stage1_science.py",
    "benchmarks/residual_tangent_auction_stage1.py",
    "benchmarks/residual_tangent_auction_stage1_core.py",
    "benchmarks/residual_tangent_auction_stage1_runner.py",
    "benchmarks/residual_tangent_auction_matrixfree.py",
    "benchmarks/residual_tangent_auction_stage1_analysis.py",
    "src/structsplat/gaussians.py",
    "src/structsplat/render.py",
    "tasks/BENCH-009-residual-tangent-auction.md",
    "tests/test_residual_tangent_auction_stage1_recovery.py",
    "tests/test_residual_tangent_auction_stage1_science.py",
    "tests/test_residual_tangent_auction_stage1_analysis.py",
    "pyproject.toml",
)


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _live_source_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parents[1]
    missing = [relative for relative in SOURCE_PATHS if not (root / relative).is_file()]
    if missing:
        raise RuntimeError(f"recovery source binding is incomplete: {missing}")
    return {relative: _sha256_file(root / relative) for relative in SOURCE_PATHS}


def _strict_json(path: Path) -> dict[str, Any]:
    value = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=lambda item: (_ for _ in ()).throw(
            ValueError(f"non-strict JSON constant {item} in {path}")
        ),
    )
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {path}")
    return value


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _stable_prefixed_id(prefix: str, value: Mapping[str, Any]) -> str:
    return prefix + hashlib.sha256(_canonical_json(dict(value))).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _strict_field(field: GaussianField, *, expected_n: int | None = None) -> None:
    if field.opacities is not None:
        raise ValueError("BENCH-009 recovery forbids opacity")
    if field.color_grads is not None:
        raise ValueError("immediate base state must not embed affine color gradients")
    if field.filter_variance is not None:
        raise ValueError("BENCH-009 recovery forbids covariance-filter state")
    if expected_n is not None and field.n != expected_n:
        raise ValueError("recovery changed the number of base Gaussians")
    if field.n <= 0:
        raise ValueError("recovery requires a non-empty parent")
    for value in (field.means, field.log_scales, field.rotations, field.colors):
        if value.device.type != "cpu":
            raise ValueError("exact source verification requires CPU tensors")
        if not bool(torch.isfinite(value).all()):
            raise ValueError("field state contains non-finite values")


def _as_six(value: torch.Tensor | Sequence[float]) -> torch.Tensor:
    tensor = torch.as_tensor(value).reshape(-1)
    if tensor.numel() != 6 or not bool(torch.isfinite(tensor).all()):
        raise ValueError("packet coefficients must contain six finite values")
    if tensor.device.type != "cpu":
        raise ValueError("packet coefficients must be on CPU")
    return tensor.detach().clone()


@dataclass(frozen=True)
class RecoveryPacket:
    """Frozen packet identity and trainable-at-recovery step-0 coefficients."""

    kind: str = "none"
    coefficients: torch.Tensor | None = None
    parent_row: int | None = None
    frequency_xy: tuple[float, float] | None = None
    phase_radians: float = 0.0
    birth_geometries: tuple[BirthGeometry, BirthGeometry] | None = None

    def validate(self, field: GaussianField) -> None:
        if self.kind not in PACKET_KINDS:
            raise ValueError(f"unknown recovery packet kind {self.kind!r}")
        if self.kind == "none":
            if any(
                value is not None
                for value in (
                    self.coefficients,
                    self.parent_row,
                    self.frequency_xy,
                    self.birth_geometries,
                )
            ):
                raise ValueError("none packet must not carry extension state")
            return
        if self.coefficients is None:
            raise ValueError("appearance packet lacks six step-0 coefficients")
        _as_six(self.coefficients)
        if self.kind in ("affine", "carrier"):
            if self.parent_row is None or not 0 <= int(self.parent_row) < field.n:
                raise ValueError("packet parent row is out of range")
        if self.kind == "carrier":
            if self.frequency_xy is None or not all(
                math.isfinite(float(value)) for value in self.frequency_xy
            ):
                raise ValueError("carrier packet lacks a finite frozen frequency")
            if not math.isfinite(float(self.phase_radians)):
                raise ValueError("carrier phase must be finite")
        elif self.frequency_xy is not None:
            raise ValueError("only a carrier packet may carry frequency state")
        if self.kind == "birth":
            if self.birth_geometries is None or len(self.birth_geometries) != 2:
                raise ValueError("birth packet needs exactly two frozen geometries")
            for geometry in self.birth_geometries:
                geometry.validate()
        elif self.birth_geometries is not None:
            raise ValueError("only a birth packet may carry birth geometry")

    def detached(self) -> RecoveryPacket:
        return RecoveryPacket(
            kind=self.kind,
            coefficients=(
                None if self.coefficients is None else self.coefficients.detach().clone()
            ),
            parent_row=self.parent_row,
            frequency_xy=self.frequency_xy,
            phase_radians=float(self.phase_radians),
            birth_geometries=self.birth_geometries,
        )


def packet_for_action(
    action: str,
    coefficients: torch.Tensor | Sequence[float] | None = None,
    *,
    identity: Mapping[str, Any] | None = None,
) -> RecoveryPacket:
    """Construct the packet protocol from generic immediate-runner action fields."""

    if action not in (*ACTION_NAMES, BASELINE_ACTION):
        raise ValueError(f"unknown recovery action {action!r}")
    identity = {} if identity is None else dict(identity)
    if action in ("full_j", "tangent6", BASELINE_ACTION):
        if coefficients is not None and torch.as_tensor(coefficients).numel() != 0:
            raise ValueError(f"{action} must encode its update in the base field")
        return RecoveryPacket()
    if coefficients is None:
        raise ValueError(f"{action} requires extension coefficients")
    if "affine" in action:
        return RecoveryPacket(
            kind="affine",
            coefficients=_as_six(coefficients),
            parent_row=int(identity["parent_row"]),
        )
    if "carrier" in action:
        frequency = identity.get("frequency_xy")
        if not isinstance(frequency, Sequence) or len(frequency) != 2:
            raise ValueError("carrier identity lacks frequency_xy")
        return RecoveryPacket(
            kind="carrier",
            coefficients=_as_six(coefficients),
            parent_row=int(identity["parent_row"]),
            frequency_xy=(float(frequency[0]), float(frequency[1])),
            phase_radians=float(identity.get("phase_radians", 0.0)),
        )
    geometries_value = identity.get("geometries")
    if not isinstance(geometries_value, Sequence) or len(geometries_value) != 2:
        raise ValueError("birth identity lacks two geometries")
    geometries: list[BirthGeometry] = []
    for value in geometries_value:
        if isinstance(value, BirthGeometry):
            geometries.append(value)
        elif isinstance(value, Mapping):
            geometries.append(
                BirthGeometry(
                    mean_xy=tuple(float(item) for item in value["mean_xy"]),  # type: ignore[arg-type]
                    scale_xy=tuple(float(item) for item in value["scale_xy"]),  # type: ignore[arg-type]
                    angle_radians=float(value["angle_radians"]),
                )
            )
        else:
            raise ValueError("birth geometry must be a mapping or BirthGeometry")
    return RecoveryPacket(
        kind="birth",
        coefficients=_as_six(coefficients),
        birth_geometries=(geometries[0], geometries[1]),
    )


def packet_sha256(packet: RecoveryPacket) -> str:
    payload: dict[str, Any] = {
        "kind": packet.kind,
        "parent_row": packet.parent_row,
        "frequency_xy": packet.frequency_xy,
        "phase_radians": packet.phase_radians,
        "birth_geometries": (
            None
            if packet.birth_geometries is None
            else [asdict(value) for value in packet.birth_geometries]
        ),
        "coefficients_sha256": (
            None if packet.coefficients is None else tensor_sha256(packet.coefficients)
        ),
    }
    return _sha256_json(payload)


def action_state_sha256(
    action: str, field: GaussianField, packet: RecoveryPacket
) -> str:
    return _sha256_json(
        {
            "action": action,
            "base_field_sha256": field_sha256(field),
            "packet_sha256": packet_sha256(packet),
        }
    )


@dataclass(frozen=True)
class ImmediateActionState:
    """Source-bound state handed from the immediate runner to recovery."""

    source_cell_id: str
    action: str
    base_field: GaussianField
    packet: RecoveryPacket
    expected_state_sha256: str
    expected_packet_sha256: str
    expected_render_sha256: str
    expected_support_sha256: str
    expected_metric_sha256: str
    predicted_gain: float
    residual_energy_reduction: float

    def validate(self, parent: GaussianField) -> None:
        if not self.source_cell_id:
            raise ValueError("step-0 source cell ID must be non-empty")
        if self.action not in (*ACTION_NAMES, BASELINE_ACTION):
            raise ValueError(f"unknown immediate action {self.action!r}")
        _strict_field(self.base_field, expected_n=parent.n)
        self.packet.validate(self.base_field)
        expected_kind = (
            "none"
            if self.action in ("full_j", "tangent6", BASELINE_ACTION)
            else "affine"
            if "affine" in self.action
            else "carrier"
            if "carrier" in self.action
            else "birth"
        )
        if self.packet.kind != expected_kind:
            raise ValueError("action and packet kind disagree")
        for value in (
            self.expected_state_sha256,
            self.expected_packet_sha256,
            self.expected_render_sha256,
            self.expected_support_sha256,
            self.expected_metric_sha256,
        ):
            if not _is_sha256(value):
                raise ValueError("source verification hashes must be lowercase SHA-256")
        if not math.isfinite(self.predicted_gain):
            raise ValueError("predicted gain must be finite")
        if not math.isfinite(self.residual_energy_reduction):
            raise ValueError("residual-energy reduction must be finite")


def _mask_channels(
    mask: torch.Tensor | Sequence[bool], height: int, width: int
) -> torch.Tensor:
    value = torch.as_tensor(mask, dtype=torch.bool, device="cpu")
    if value.shape == (height, width):
        value = value.unsqueeze(-1).expand(height, width, 3)
    value = value.reshape(-1)
    if value.numel() != height * width * 3:
        raise ValueError("scoring mask has the wrong number of RGB entries")
    if not bool(value.any()):
        raise ValueError("scoring mask selects no entries")
    return value


def _validate_masks(
    discovery: torch.Tensor | Sequence[bool],
    heldout: torch.Tensor | Sequence[bool],
    height: int,
    width: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    fit = _mask_channels(discovery, height, width)
    score = _mask_channels(heldout, height, width)
    if bool((fit & score).any()) or not bool((fit | score).all()):
        raise ValueError("discovery and held-out masks must be disjoint and exhaustive")
    return fit, score


def _masked_mse(
    render: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> float:
    return float((render - target).detach().reshape(-1)[mask].square().mean())


def _psnr(mse: float) -> float:
    return float(-10.0 * math.log10(max(float(mse), 1e-30)))


def source_metric_payload(
    render: torch.Tensor,
    target: torch.Tensor,
    discovery_mask: torch.Tensor,
    heldout_mask: torch.Tensor,
) -> dict[str, float]:
    """Canonical source metrics whose hash is copied into the step-0 link."""

    discovery_mse = _masked_mse(render, target, discovery_mask)
    heldout_mse = _masked_mse(render, target, heldout_mask)
    full_mse = float((render - target).detach().square().mean())
    return {
        "discovery_mse": discovery_mse,
        "heldout_mse": heldout_mse,
        "full_mse": full_mse,
        "heldout_psnr_db": _psnr(heldout_mse),
    }


def source_metric_sha256(
    render: torch.Tensor,
    target: torch.Tensor,
    discovery_mask: torch.Tensor,
    heldout_mask: torch.Tensor,
) -> str:
    return _sha256_json(
        source_metric_payload(render, target, discovery_mask, heldout_mask)
    )


@dataclass(frozen=True)
class _SupportSnapshot:
    base_membership: torch.Tensor
    birth_membership: torch.Tensor | None

    @property
    def base_sha256(self) -> str:
        return tensor_sha256(self.base_membership)

    @property
    def birth_sha256(self) -> str | None:
        return (
            None
            if self.birth_membership is None
            else tensor_sha256(self.birth_membership)
        )

    @property
    def bundle_sha256(self) -> str:
        return _sha256_json(
            {
                "base_membership_sha256": self.base_sha256,
                "birth_membership_sha256": self.birth_sha256,
            }
        )


def _birth_field(
    packet: RecoveryPacket,
    coefficients: torch.Tensor,
    *,
    dtype: torch.dtype,
) -> GaussianField:
    if packet.birth_geometries is None:
        raise RuntimeError("birth packet lacks geometry")
    means = torch.tensor(
        [value.mean_xy for value in packet.birth_geometries], dtype=dtype
    )
    scales = torch.tensor(
        [value.scale_xy for value in packet.birth_geometries], dtype=dtype
    )
    rotations = torch.tensor(
        [value.angle_radians for value in packet.birth_geometries], dtype=dtype
    )
    return GaussianField(means, torch.log(scales), rotations, coefficients.reshape(2, 3))


def _combined_birth_field(
    field: GaussianField, packet: RecoveryPacket, coefficients: torch.Tensor
) -> GaussianField:
    births = _birth_field(packet, coefficients, dtype=field.means.dtype)
    return GaussianField(
        torch.cat((field.means, births.means)),
        torch.cat((field.log_scales, births.log_scales)),
        torch.cat((field.rotations, births.rotations)),
        torch.cat((field.colors, births.colors)),
    )


def _support_snapshot(
    field: GaussianField,
    packet: RecoveryPacket,
    coefficients: torch.Tensor | None,
    height: int,
    width: int,
    sigma_cutoff: float,
) -> _SupportSnapshot:
    base = freeze_pixel_support(
        field, height, width, sigma_cutoff=sigma_cutoff
    ).membership
    birth = None
    if packet.kind == "birth":
        if coefficients is None:
            raise RuntimeError("birth support lacks coefficients")
        birth = freeze_pixel_support(
            _birth_field(packet, coefficients, dtype=field.means.dtype),
            height,
            width,
            sigma_cutoff=sigma_cutoff,
        ).membership
    return _SupportSnapshot(base.detach().clone(), None if birth is None else birth.detach().clone())


def _membership_delta(before: torch.Tensor, after: torch.Tensor) -> dict[str, Any]:
    if before.shape != after.shape:
        raise ValueError("support snapshots have incompatible shapes")
    before = before.to(dtype=torch.bool, device="cpu")
    after = after.to(dtype=torch.bool, device="cpu")
    added = after & ~before
    removed = before & ~after
    xor = added | removed
    return {
        "before_membership_sha256": tensor_sha256(before),
        "after_membership_sha256": tensor_sha256(after),
        "added_memberships": int(added.sum()),
        "removed_memberships": int(removed.sum()),
        "xor_memberships": int(xor.sum()),
        "added_membership_sha256": tensor_sha256(added),
        "removed_membership_sha256": tensor_sha256(removed),
        "xor_membership_sha256": tensor_sha256(xor),
    }


def _packet_coefficients(packet: RecoveryPacket) -> torch.Tensor | None:
    if packet.coefficients is None:
        return None
    return packet.coefficients.detach().clone().requires_grad_(True)


def _render_state(
    field: GaussianField,
    packet: RecoveryPacket,
    coefficients: torch.Tensor | None,
    height: int,
    width: int,
    *,
    sigma_cutoff: float,
    render_chunk: int,
) -> torch.Tensor:
    """Render with native support recomputed from the current state on every call."""

    if packet.kind == "none":
        return render_normalized(
            field,
            height,
            width,
            sigma_cutoff=sigma_cutoff,
            render_chunk=render_chunk,
        )
    if coefficients is None:
        raise RuntimeError("appearance packet lacks trainable coefficients")
    if packet.kind == "affine":
        gradients = torch.zeros(
            field.n, 2, 3, device="cpu", dtype=field.colors.dtype
        )
        gradients[int(packet.parent_row)] = coefficients.reshape(2, 3)
        return render_normalized(
            field,
            height,
            width,
            sigma_cutoff=sigma_cutoff,
            render_chunk=render_chunk,
            color_grads=gradients,
        )
    if packet.kind == "carrier":
        from benchmarks.residual_tangent_auction_stage1_core import carrier_six_columns

        base = render_normalized(
            field,
            height,
            width,
            sigma_cutoff=sigma_cutoff,
            render_chunk=render_chunk,
        )
        columns = carrier_six_columns(
            field,
            int(packet.parent_row),
            packet.frequency_xy,  # type: ignore[arg-type]
            height,
            width,
            phase_radians=packet.phase_radians,
            sigma_cutoff=sigma_cutoff,
        )
        return base + (columns @ coefficients).reshape_as(base)
    combined = _combined_birth_field(field, packet, coefficients)
    return render_normalized(
        combined,
        height,
        width,
        sigma_cutoff=sigma_cutoff,
        render_chunk=render_chunk,
    )


def capture_immediate_action_state(
    *,
    source_cell_id: str,
    action: str,
    base_field: GaussianField,
    packet: RecoveryPacket,
    target: torch.Tensor,
    discovery_mask: torch.Tensor | Sequence[bool],
    heldout_mask: torch.Tensor | Sequence[bool],
    predicted_gain: float = 0.0,
    residual_energy_reduction: float = 0.0,
    sigma_cutoff: float = 3.0,
    render_chunk: int = 512,
) -> ImmediateActionState:
    """Source-side helper for emitting the exact hash contract consumed by recovery."""

    _strict_field(base_field)
    packet.validate(base_field)
    target = torch.as_tensor(target, dtype=base_field.means.dtype, device="cpu")
    if target.ndim != 3 or target.shape[-1] != 3 or not bool(torch.isfinite(target).all()):
        raise ValueError("target must be finite HxWx3 CPU RGB")
    height, width = int(target.shape[0]), int(target.shape[1])
    discovery, heldout = _validate_masks(
        discovery_mask, heldout_mask, height, width
    )
    coefficients = None if packet.coefficients is None else packet.coefficients
    render = _render_state(
        base_field,
        packet,
        coefficients,
        height,
        width,
        sigma_cutoff=sigma_cutoff,
        render_chunk=render_chunk,
    )
    support = _support_snapshot(
        base_field, packet, coefficients, height, width, sigma_cutoff
    )
    return ImmediateActionState(
        source_cell_id=source_cell_id,
        action=action,
        base_field=base_field.detached(),
        packet=packet.detached(),
        expected_state_sha256=action_state_sha256(action, base_field, packet),
        expected_packet_sha256=packet_sha256(packet),
        expected_render_sha256=tensor_sha256(render),
        expected_support_sha256=support.bundle_sha256,
        expected_metric_sha256=source_metric_sha256(
            render, target, discovery, heldout
        ),
        predicted_gain=float(predicted_gain),
        residual_energy_reduction=float(residual_energy_reduction),
    )


def immediate_state_from_payload(
    parent: GaussianField,
    payload: Mapping[str, Any],
    *,
    tensor_resolver: Callable[[Mapping[str, Any], str], torch.Tensor] | None = None,
) -> ImmediateActionState:
    """Reconstruct a source state from generic list- or bound-NPZ-backed immediate fields.

    The payload must expose ``base_parameter_step``, ``extension_coefficients``, and ``identity``.
    A caller handling bound NPZ state can provide ``tensor_resolver``; otherwise list/tensor values
    are read directly.  Hashes are never synthesized here: they must have been emitted by the
    source-side immediate runner.
    """

    def tensor(name: str) -> torch.Tensor:
        value = (
            tensor_resolver(payload, name)
            if tensor_resolver is not None
            else torch.as_tensor(payload[name], dtype=parent.means.dtype)
        )
        return value.to(device="cpu", dtype=parent.means.dtype)

    action = str(payload["action"])
    parameterization = make_no_opacity_parameterization(
        parent, int(payload["height"]), int(payload["width"])
    )
    base_step = tensor("base_parameter_step").reshape(-1)
    base_field = parameterization.field_at(base_step).detached()
    extension = (
        None
        if action in ("full_j", "tangent6", BASELINE_ACTION)
        else tensor("extension_coefficients").reshape(-1)
    )
    packet = packet_for_action(
        action, extension, identity=payload.get("identity")
    )
    hashes = payload["source_hashes"]
    if not isinstance(hashes, Mapping):
        raise ValueError("immediate payload lacks source_hashes mapping")
    return ImmediateActionState(
        source_cell_id=str(payload.get("source_cell_id", payload.get("cell_id", ""))),
        action=action,
        base_field=base_field,
        packet=packet,
        expected_state_sha256=str(hashes["state_sha256"]),
        expected_packet_sha256=str(hashes["packet_sha256"]),
        expected_render_sha256=str(hashes["native_render_sha256"]),
        expected_support_sha256=str(hashes["native_support_sha256"]),
        expected_metric_sha256=str(hashes["metric_sha256"]),
        predicted_gain=float(payload.get("predicted_gain", 0.0)),
        residual_energy_reduction=float(
            payload.get("residual_energy_reduction", 0.0)
        ),
    )


@dataclass(frozen=True)
class RecoveryContext:
    binding_sha256: str
    family: str
    seed: int
    parent_checkpoint: int
    fold: int
    trust_radius: float | None
    parent_id: str
    trajectory_id: str | None = None
    recovery_binding_sha256: str | None = None
    source_binding_sha256: str | None = None
    manifest_recovery_cell_id: str | None = None

    def validate(self, action: str) -> None:
        if not _is_sha256(self.binding_sha256):
            raise ValueError("recovery binding must be a lowercase SHA-256")
        if not self.family or not self.parent_id:
            raise ValueError("recovery family and parent ID must be non-empty")
        if self.trajectory_id is not None and not self.trajectory_id.startswith(
            "bench009-recovery-plan:"
        ):
            raise ValueError("explicit recovery trajectory ID is not a frozen plan ID")
        for name, value in (
            ("recovery_binding_sha256", self.recovery_binding_sha256),
            ("source_binding_sha256", self.source_binding_sha256),
            ("manifest_recovery_cell_id", self.manifest_recovery_cell_id),
        ):
            if value is not None and not _is_sha256(value):
                raise ValueError(f"{name} must be a lowercase SHA-256")
        if self.fold not in (0, 1):
            raise ValueError("recovery fold must be 0 or 1")
        if action == BASELINE_ACTION:
            if self.trust_radius is not None:
                raise ValueError("no-action recovery has no trust-radius axis")
        elif self.trust_radius not in TRUST_RADII:
            raise ValueError("action recovery trust radius must be 0.25 or 0.75")


@dataclass(frozen=True)
class RecoveryResult:
    checkpoint_records: tuple[dict[str, Any], ...]
    evidence_records: tuple[dict[str, Any], ...]
    terminal_record: dict[str, Any]


def _trajectory_key(context: RecoveryContext, source: ImmediateActionState) -> dict[str, Any]:
    return {
        "protocol": PROTOCOL,
        "binding_sha256": context.binding_sha256,
        "family": context.family,
        "seed": context.seed,
        "parent_checkpoint": context.parent_checkpoint,
        "parent_id": context.parent_id,
        "fold": context.fold,
        "trust_radius": context.trust_radius,
        "action": source.action,
        "step0_source_cell_id": source.source_cell_id,
        "damping": PRIMARY_DAMPING,
        "rcond": PRIMARY_RCOND,
        "manifest_recovery_cell_id": context.manifest_recovery_cell_id,
        "recovery_binding_sha256": context.recovery_binding_sha256,
        "source_binding_sha256": context.source_binding_sha256,
    }


def _checkpoint_id(trajectory_id: str, checkpoint: int) -> str:
    return _stable_prefixed_id(
        "bench009-recovery-checkpoint:",
        {"trajectory_id": trajectory_id, "checkpoint": checkpoint},
    )


def _optimizer(field: GaussianField, coefficients: torch.Tensor | None) -> torch.optim.Adam:
    groups: list[dict[str, Any]] = [
        {"params": [field.means], "lr": LR_MEANS, "name": "means"},
        {"params": [field.log_scales], "lr": LR_LOG_SCALES, "name": "log_scales"},
        {"params": [field.rotations], "lr": LR_ROTATIONS, "name": "rotations"},
        {"params": [field.colors], "lr": LR_BASE_RGB, "name": "base_rgb"},
    ]
    if coefficients is not None:
        groups.append(
            {"params": [coefficients], "lr": LR_PACKET_RGB, "name": "packet_rgb"}
        )
    return torch.optim.Adam(groups, betas=ADAM_BETAS, eps=ADAM_EPS)


def _checkpoint_record(
    *,
    context: RecoveryContext,
    source: ImmediateActionState,
    trajectory_id: str,
    checkpoint: int,
    field: GaussianField,
    coefficients: torch.Tensor | None,
    render: torch.Tensor,
    target: torch.Tensor,
    discovery: torch.Tensor,
    heldout: torch.Tensor,
    parent_mse: float,
    parent_support: _SupportSnapshot,
    previous_support: _SupportSnapshot | None,
    current_support: _SupportSnapshot,
    optimizer_steps_completed: int,
    elapsed_seconds: float,
) -> dict[str, Any]:
    discovery_mse = _masked_mse(render, target, discovery)
    heldout_mse = _masked_mse(render, target, heldout)
    packet = RecoveryPacket(
        kind=source.packet.kind,
        coefficients=(
            None if coefficients is None else coefficients.detach().clone()
        ),
        parent_row=source.packet.parent_row,
        frequency_xy=source.packet.frequency_xy,
        phase_radians=source.packet.phase_radians,
        birth_geometries=source.packet.birth_geometries,
    )
    record: dict[str, Any] = {
        "schema": RAW_CHECKPOINT_SCHEMA,
        "record_kind": RAW_CHECKPOINT_KIND,
        "checkpoint_id": _checkpoint_id(trajectory_id, checkpoint),
        "trajectory_id": trajectory_id,
        "status": "ok",
        "error": None,
        "protocol": PROTOCOL,
        "binding_sha256": context.binding_sha256,
        "family": context.family,
        "seed": context.seed,
        "parent_checkpoint": context.parent_checkpoint,
        "parent_id": context.parent_id,
        "fold": context.fold,
        "trust_radius": context.trust_radius,
        "action": source.action,
        "damping": PRIMARY_DAMPING,
        "rcond": PRIMARY_RCOND,
        "logical_checkpoint": checkpoint,
        "newly_rendered_recovery_checkpoint": checkpoint in NEW_RENDER_CHECKPOINTS,
        "step0_source_cell_id": source.source_cell_id,
        "step0_link_verified": True,
        "optimizer_steps_completed": optimizer_steps_completed,
        "optimizer_objective": "discovery_tile_rgb_mse",
        "scoring_policy": "heldout_tiles_without_refit",
        "discovery_mask_sha256": tensor_sha256(discovery),
        "scoring_mask_sha256": tensor_sha256(heldout),
        "state_sha256": action_state_sha256(source.action, field, packet),
        "base_field_sha256": field_sha256(field),
        "packet_sha256": packet_sha256(packet),
        "native_render_sha256": tensor_sha256(render),
        "native_support_sha256": current_support.bundle_sha256,
        "source_state_sha256": source.expected_state_sha256,
        "source_packet_sha256": source.expected_packet_sha256,
        "source_native_render_sha256": source.expected_render_sha256,
        "source_native_support_sha256": source.expected_support_sha256,
        "source_metric_sha256": source.expected_metric_sha256,
        "recovery_binding_sha256": context.recovery_binding_sha256,
        "source_binding_sha256": context.source_binding_sha256,
        "manifest_recovery_cell_id": context.manifest_recovery_cell_id,
        "discovery_mse": discovery_mse,
        "heldout_mse": heldout_mse,
        "parent_heldout_mse": parent_mse,
        "heldout_realized_gain": parent_mse - heldout_mse,
        "heldout_psnr_db": _psnr(heldout_mse),
        "support_relative_parent": _membership_delta(
            parent_support.base_membership, current_support.base_membership
        ),
        "support_relative_previous_checkpoint": (
            None
            if previous_support is None
            else _membership_delta(
                previous_support.base_membership, current_support.base_membership
            )
        ),
        "birth_support": (
            None
            if current_support.birth_membership is None
            else {
                "membership_sha256": current_support.birth_sha256,
                "membership_count": int(current_support.birth_membership.sum()),
                "relative_previous_checkpoint": (
                    None
                    if previous_support is None
                    or previous_support.birth_membership is None
                    else _membership_delta(
                        previous_support.birth_membership,
                        current_support.birth_membership,
                    )
                ),
            }
        ),
        "wall_seconds_from_trajectory_start": elapsed_seconds,
        "peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "topology_events": "disabled",
        "opacity": "disabled",
        "qat": "disabled",
        "color_solves": "disabled",
        "packet_reselection": "disabled",
        "native_dynamic_support_every_step": True,
    }
    if checkpoint == 0:
        if record["state_sha256"] != source.expected_state_sha256:
            raise RuntimeError("step-0 state hash changed while recording verified state")
        if record["packet_sha256"] != source.expected_packet_sha256:
            raise RuntimeError("step-0 packet hash changed while recording verified state")
        if record["native_render_sha256"] != source.expected_render_sha256:
            raise RuntimeError("step-0 render hash changed while recording verified state")
        if record["native_support_sha256"] != source.expected_support_sha256:
            raise RuntimeError("step-0 support hash changed while recording verified state")
    _canonical_json(record)
    return record


def _evidence_record(
    raw: Mapping[str, Any],
    source: ImmediateActionState,
) -> dict[str, Any]:
    if source.action not in MATCHED_ACTIONS:
        raise ValueError("only matched-six-DOF actions have analyzer evidence rows")
    checkpoint = int(raw["logical_checkpoint"])
    if checkpoint not in NEW_RENDER_CHECKPOINTS:
        raise ValueError("recovery evidence contains only checkpoints 20 and 100")
    record: dict[str, Any] = {
        "schema": EVIDENCE_SCHEMA_ID,
        "record_kind": EVIDENCE_RECORD_KIND,
        "component": "crossfit_recovery",
        "status": "ok",
        "error": None,
        "binding_sha256": raw["binding_sha256"],
        "selection_scope": CROSS_FIT_SCOPE,
        "ledger": MATCHED_LEDGER,
        "family": raw["family"],
        "seed": raw["seed"],
        "parent_checkpoint": raw["parent_checkpoint"],
        "fold": raw["fold"],
        "action": raw["action"],
        "damping": PRIMARY_DAMPING,
        "rcond": PRIMARY_RCOND,
        "trust_radius": raw["trust_radius"],
        "checkpoint": checkpoint,
        "scoring_mask_sha256": raw["scoring_mask_sha256"],
        "parent_mse": raw["parent_heldout_mse"],
        "predicted_gain": source.predicted_gain,
        "realized_gain": raw["heldout_realized_gain"],
        "residual_energy_reduction": source.residual_energy_reduction,
        "psnr_db": raw["heldout_psnr_db"],
        "metadata": {
            "trajectory_id": raw["trajectory_id"],
            "checkpoint_id": raw["checkpoint_id"],
            "step0_source_cell_id": source.source_cell_id,
            "state_sha256": raw["state_sha256"],
            "packet_sha256": raw["packet_sha256"],
            "native_render_sha256": raw["native_render_sha256"],
            "native_support_sha256": raw["native_support_sha256"],
            "recovery_binding_sha256": raw["recovery_binding_sha256"],
            "source_binding_sha256": raw["source_binding_sha256"],
            "manifest_recovery_cell_id": raw["manifest_recovery_cell_id"],
        },
    }
    record["cell_id"] = stable_cell_id(record)
    _canonical_json(record)
    return record


def run_recovery(
    *,
    parent: GaussianField,
    source: ImmediateActionState,
    target: torch.Tensor,
    discovery_mask: torch.Tensor | Sequence[bool],
    heldout_mask: torch.Tensor | Sequence[bool],
    context: RecoveryContext,
    sigma_cutoff: float = 3.0,
    render_chunk: int = 512,
) -> RecoveryResult:
    """Verify step 0 and execute one exact, continuous 100-step recovery trajectory."""

    torch.use_deterministic_algorithms(True)
    _strict_field(parent)
    source.validate(parent)
    context.validate(source.action)
    target = torch.as_tensor(target, dtype=parent.means.dtype, device="cpu")
    if target.ndim != 3 or target.shape[-1] != 3 or not bool(torch.isfinite(target).all()):
        raise ValueError("target must be finite HxWx3 CPU RGB")
    height, width = int(target.shape[0]), int(target.shape[1])
    discovery, heldout = _validate_masks(
        discovery_mask, heldout_mask, height, width
    )
    if action_state_sha256(source.action, source.base_field, source.packet) != source.expected_state_sha256:
        raise RuntimeError("step-0 source state hash mismatch before rendering")
    if packet_sha256(source.packet) != source.expected_packet_sha256:
        raise RuntimeError("step-0 source packet hash mismatch before rendering")

    field = source.base_field.detached().trainable()
    coefficients = _packet_coefficients(source.packet)
    packet = source.packet.detached()
    started = time.perf_counter()

    # This is the single setup rerender required by the contract.  It is reused as the logical
    # step-0 checkpoint and is not followed by a redundant step-0 render.
    step0_render = _render_state(
        field,
        packet,
        coefficients,
        height,
        width,
        sigma_cutoff=sigma_cutoff,
        render_chunk=render_chunk,
    )
    if tensor_sha256(step0_render) != source.expected_render_sha256:
        raise RuntimeError("step-0 native render hash mismatch")
    step0_support = _support_snapshot(
        field, packet, coefficients, height, width, sigma_cutoff
    )
    if step0_support.bundle_sha256 != source.expected_support_sha256:
        raise RuntimeError("step-0 native support hash mismatch")
    if source_metric_sha256(
        step0_render, target, discovery, heldout
    ) != source.expected_metric_sha256:
        raise RuntimeError("step-0 source metric hash mismatch")

    # In the no-action arm the verified step-0 render *is* the parent render. Reusing it preserves
    # the contract's one setup rerender instead of silently rendering baseline step 0 twice.
    parent_render = (
        step0_render
        if source.action == BASELINE_ACTION
        else render_normalized(
            parent,
            height,
            width,
            sigma_cutoff=sigma_cutoff,
            render_chunk=render_chunk,
        )
    )
    parent_mse = _masked_mse(parent_render, target, heldout)
    parent_support = _support_snapshot(
        parent, RecoveryPacket(), None, height, width, sigma_cutoff
    )
    trajectory_key = _trajectory_key(context, source)
    trajectory_id = context.trajectory_id or _stable_prefixed_id(
        "bench009-recovery:", trajectory_key
    )
    raw: list[dict[str, Any]] = [
        _checkpoint_record(
            context=context,
            source=source,
            trajectory_id=trajectory_id,
            checkpoint=0,
            field=field,
            coefficients=coefficients,
            render=step0_render,
            target=target,
            discovery=discovery,
            heldout=heldout,
            parent_mse=parent_mse,
            parent_support=parent_support,
            previous_support=None,
            current_support=step0_support,
            optimizer_steps_completed=0,
            elapsed_seconds=time.perf_counter() - started,
        )
    ]

    optimizer = _optimizer(field, coefficients)
    if optimizer.state:
        raise RuntimeError("fresh recovery Adam unexpectedly inherited moments")
    previous_support = step0_support
    for step in range(1, RECOVERY_STEPS + 1):
        optimizer.zero_grad(set_to_none=True)
        render = _render_state(
            field,
            packet,
            coefficients,
            height,
            width,
            sigma_cutoff=sigma_cutoff,
            render_chunk=render_chunk,
        )
        loss = (render - target).reshape(-1)[discovery].square().mean()
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(f"non-finite recovery objective at step {step}")
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            field.log_scales.clamp_(LOG_SCALE_MIN, LOG_SCALE_MAX)
        if step in NEW_RENDER_CHECKPOINTS:
            checkpoint_render = _render_state(
                field,
                packet,
                coefficients,
                height,
                width,
                sigma_cutoff=sigma_cutoff,
                render_chunk=render_chunk,
            )
            if not bool(torch.isfinite(checkpoint_render).all()):
                raise RuntimeError(f"non-finite recovery render at step {step}")
            current_support = _support_snapshot(
                field, packet, coefficients, height, width, sigma_cutoff
            )
            raw.append(
                _checkpoint_record(
                    context=context,
                    source=source,
                    trajectory_id=trajectory_id,
                    checkpoint=step,
                    field=field,
                    coefficients=coefficients,
                    render=checkpoint_render,
                    target=target,
                    discovery=discovery,
                    heldout=heldout,
                    parent_mse=parent_mse,
                    parent_support=parent_support,
                    previous_support=previous_support,
                    current_support=current_support,
                    optimizer_steps_completed=step,
                    elapsed_seconds=time.perf_counter() - started,
                )
            )
            previous_support = current_support

    if [value["logical_checkpoint"] for value in raw] != list(LOGICAL_CHECKPOINTS):
        raise RuntimeError("recovery did not produce the exact logical checkpoint ledger")
    evidence = tuple(
        _evidence_record(value, source)
        for value in raw
        if source.action in MATCHED_ACTIONS
        and value["logical_checkpoint"] in NEW_RENDER_CHECKPOINTS
    )
    terminal = {
        "schema": RAW_TRAJECTORY_SCHEMA,
        "record_kind": RAW_TRAJECTORY_KIND,
        "trajectory_id": trajectory_id,
        "status": "ok",
        "error": None,
        **trajectory_key,
        "logical_checkpoint_ids": [value["checkpoint_id"] for value in raw],
        "logical_checkpoint_count": 3,
        "new_render_checkpoint_count": 2,
        "evidence_cell_ids": [value["cell_id"] for value in evidence],
        "optimizer": {
            "name": "Adam",
            "fresh_state": True,
            "betas": list(ADAM_BETAS),
            "eps": ADAM_EPS,
            "weight_decay": 0.0,
            "learning_rates": {
                "means": LR_MEANS,
                "log_scales": LR_LOG_SCALES,
                "rotations": LR_ROTATIONS,
                "base_rgb": LR_BASE_RGB,
                "packet_rgb": LR_PACKET_RGB,
            },
            "continuous_steps": RECOVERY_STEPS,
            "log_scale_clamp": [LOG_SCALE_MIN, LOG_SCALE_MAX],
        },
        "wall_seconds": time.perf_counter() - started,
        "peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
    }
    _canonical_json(terminal)
    return RecoveryResult(tuple(raw), evidence, terminal)


@dataclass(frozen=True)
class RecoveryParentKey:
    family: str
    seed: int
    parent_checkpoint: int
    parent_id: str


@dataclass(frozen=True)
class RecoveryPlanEntry:
    family: str
    seed: int
    parent_checkpoint: int
    parent_id: str
    fold: int
    action: str
    trust_radius: float | None
    step0_source_cell_id: str
    manifest_recovery_cell_id: str = ""

    @property
    def stable_key(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def trajectory_id(self) -> str:
        return _stable_prefixed_id("bench009-recovery-plan:", self.stable_key)


def build_recovery_plan(
    parents: Iterable[RecoveryParentKey],
    source_cell_ids: Mapping[tuple[str, int, float | None, str], str],
    *,
    require_frozen_count: bool = True,
) -> tuple[RecoveryPlanEntry, ...]:
    """Build the frozen 768 action + 48 baseline trajectory manifest."""

    from benchmarks import residual_tangent_auction_stage1 as manifest_module

    recovery_rows = manifest_module.build_manifest()["recovery_plan"]
    manifest_ids: dict[tuple[str, int, int, int, str, float | None], str] = {}
    for row in recovery_rows:
        key = (
            str(row["family"]),
            int(row["seed"]),
            int(row["parent_iters"]),
            int(row["heldout_fold"]),
            str(row["action"]),
            None if row["trust_radius"] is None else float(row["trust_radius"]),
        )
        if key in manifest_ids:
            raise RuntimeError(f"duplicate frozen recovery manifest axes {key}")
        manifest_ids[key] = str(row["cell_id"])
    result: list[RecoveryPlanEntry] = []
    ordered = sorted(
        parents,
        key=lambda item: (
            item.family,
            item.seed,
            item.parent_checkpoint,
            item.parent_id,
        ),
    )
    for parent in ordered:
        for fold in (0, 1):
            baseline_key = (parent.parent_id, fold, None, BASELINE_ACTION)
            if baseline_key not in source_cell_ids:
                raise ValueError(f"missing baseline source link {baseline_key}")
            result.append(
                RecoveryPlanEntry(
                    **asdict(parent),
                    fold=fold,
                    action=BASELINE_ACTION,
                    trust_radius=None,
                    step0_source_cell_id=source_cell_ids[baseline_key],
                    manifest_recovery_cell_id=manifest_ids[
                        (
                            parent.family,
                            parent.seed,
                            parent.parent_checkpoint,
                            fold,
                            BASELINE_ACTION,
                            None,
                        )
                    ],
                )
            )
            for radius in TRUST_RADII:
                for action in ACTION_NAMES:
                    key = (parent.parent_id, fold, radius, action)
                    if key not in source_cell_ids:
                        raise ValueError(f"missing immediate-action source link {key}")
                    result.append(
                        RecoveryPlanEntry(
                            **asdict(parent),
                            fold=fold,
                            action=action,
                            trust_radius=radius,
                            step0_source_cell_id=source_cell_ids[key],
                            manifest_recovery_cell_id=manifest_ids[
                                (
                                    parent.family,
                                    parent.seed,
                                    parent.parent_checkpoint,
                                    fold,
                                    action,
                                    radius,
                                )
                            ],
                        )
                    )
    validate_recovery_plan(
        result, expected_count=EXPECTED_TRAJECTORIES if require_frozen_count else None
    )
    return tuple(result)


def validate_recovery_plan(
    entries: Sequence[RecoveryPlanEntry], *, expected_count: int | None = EXPECTED_TRAJECTORIES
) -> None:
    ids = [entry.trajectory_id for entry in entries]
    if len(ids) != len(set(ids)):
        raise ValueError("recovery plan contains duplicate stable trajectories")
    if expected_count is not None and len(entries) != expected_count:
        raise ValueError(
            f"recovery plan has {len(entries)} trajectories, expected {expected_count}"
        )
    for entry in entries:
        if entry.fold not in (0, 1):
            raise ValueError("recovery plan fold must be 0 or 1")
        if entry.action == BASELINE_ACTION:
            if entry.trust_radius is not None:
                raise ValueError("baseline plan entry must not have a trust radius")
        elif entry.action not in ACTION_NAMES or entry.trust_radius not in TRUST_RADII:
            raise ValueError("invalid action recovery plan entry")
        if not entry.step0_source_cell_id:
            raise ValueError("recovery plan entry lacks a source link")
        if not _is_sha256(entry.manifest_recovery_cell_id):
            raise ValueError("recovery plan entry lacks a frozen manifest cell ID")
    if expected_count == EXPECTED_TRAJECTORIES:
        from benchmarks import residual_tangent_auction_stage1 as manifest_module

        expected_manifest_ids = {
            str(row["cell_id"])
            for row in manifest_module.build_manifest()["recovery_plan"]
        }
        observed_manifest_ids = {
            entry.manifest_recovery_cell_id for entry in entries
        }
        if observed_manifest_ids != expected_manifest_ids:
            raise ValueError("generated recovery plan is not 1:1 with frozen recovery_plan")


@dataclass(frozen=True)
class _IntegrationInputs:
    science_dir: Path
    runner_dir: Path
    science_config: dict[str, Any]
    runner_config: dict[str, Any]
    immediate: dict[str, dict[str, Any]]
    parents: dict[tuple[str, int, str], dict[str, Any]]
    targets: dict[str, dict[str, Any]]
    baselines: dict[tuple[str, int, str, int], dict[str, Any]]


def _load_integration_inputs(
    science_dir: str | Path, runner_dir: str | Path
) -> _IntegrationInputs:
    """Load and source-validate the immediate, parent, target, and baseline artifacts."""

    from benchmarks import residual_tangent_auction_stage1 as manifest_module

    science_root = Path(science_dir).resolve()
    runner_root = Path(runner_dir).resolve()
    science_config_path = science_root / "science_config.json"
    runner_config_path = runner_root / "runner_config.json"
    immediate_path = science_root / "immediate_cells.jsonl"
    parents_path = runner_root / "parents.jsonl"
    for path in (
        science_config_path,
        runner_config_path,
        immediate_path,
        parents_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(f"required recovery integration artifact is missing: {path}")
    science_config = _strict_json(science_config_path)
    runner_config = _strict_json(runner_config_path)
    science_binding = science_config.get("binding")
    runner_binding = runner_config.get("binding")
    if not isinstance(science_binding, dict) or not isinstance(runner_binding, dict):
        raise RuntimeError("upstream configs lack source bindings")
    if science_config.get("binding_sha256") != _sha256_json(science_binding):
        raise RuntimeError("science config binding hash mismatch")
    if runner_config.get("binding_sha256") != _sha256_json(runner_binding):
        raise RuntimeError("runner config binding hash mismatch")
    root = Path(__file__).resolve().parents[1]
    for label, binding in (("science", science_binding), ("runner", runner_binding)):
        sources = binding.get("source_sha256")
        if not isinstance(sources, dict):
            raise RuntimeError(f"{label} binding lacks source hashes")
        stale = [
            relative
            for relative, expected in sources.items()
            if not (root / relative).is_file()
            or _sha256_file(root / relative) != expected
        ]
        if stale:
            raise RuntimeError(f"{label} source binding is stale: {stale[:5]}")
    if science_binding.get("runner_binding_sha256") != runner_config.get(
        "binding_sha256"
    ):
        raise RuntimeError("science artifact is bound to a different parent runner")

    immediate = _existing_jsonl(immediate_path, "cell_id")
    if len(immediate) != 4_608:
        raise RuntimeError(
            f"recovery requires 4,608 terminal immediate rows, found {len(immediate)}"
        )
    if any(value.get("status") != "ok" for value in immediate.values()):
        raise RuntimeError("immediate component contains terminal error rows")
    expected_immediate_ids = {
        str(value["cell_id"])
        for value in manifest_module.build_manifest()["action_cells"]
        if value["cell_kind"] == "core_immediate"
    }
    if set(immediate) != expected_immediate_ids:
        missing = len(expected_immediate_ids - set(immediate))
        unexpected = len(set(immediate) - expected_immediate_ids)
        raise RuntimeError(
            "immediate rows do not exactly match the frozen manifest IDs: "
            f"missing={missing}, unexpected={unexpected}"
        )
    science_hash = science_config.get("binding_sha256")
    if any(value.get("binding_sha256") != science_hash for value in immediate.values()):
        raise RuntimeError("immediate component mixes stale science bindings")

    parent_values = _existing_jsonl(parents_path, "parent_id")
    if len(parent_values) != 24:
        raise RuntimeError(f"recovery requires exactly 24 parents, found {len(parent_values)}")
    parents: dict[tuple[str, int, str], dict[str, Any]] = {}
    for record in parent_values.values():
        key = (
            str(record["target_id"]),
            int(record["seed"]),
            str(record["horizon_name"]),
        )
        if key in parents:
            raise RuntimeError(f"duplicate parent axes {key}")
        if record.get("binding_sha256") != runner_config.get("binding_sha256"):
            raise RuntimeError("parent row has a stale runner binding")
        field_path = Path(str(record["field_path"]))
        if not field_path.is_file() or _sha256_file(field_path) != record.get(
            "field_file_sha256"
        ):
            raise RuntimeError(f"parent field file is missing or changed: {field_path}")
        parents[key] = record

    suite = manifest_module.target_suite()
    targets = {str(value["target_id"]): value for value in suite.values()}
    baseline_rows = [
        value
        for value in manifest_module.build_manifest()["action_cells"]
        if value["cell_kind"] == "no_action_baseline"
        and value["selection_scope"] == "crossfit"
    ]
    baselines: dict[tuple[str, int, str, int], dict[str, Any]] = {}
    for record in baseline_rows:
        key = (
            str(record["target_id"]),
            int(record["seed"]),
            str(record["parent_horizon"]),
            int(record["heldout_fold"]),
        )
        if key in baselines:
            raise RuntimeError(f"duplicate baseline manifest axes {key}")
        baselines[key] = record
    if len(baselines) != 48:
        raise RuntimeError("baseline manifest must contain exactly 48 cross-fit rows")
    return _IntegrationInputs(
        science_root,
        runner_root,
        science_config,
        runner_config,
        immediate,
        parents,
        targets,
        baselines,
    )


def _primary_action_records(
    inputs: _IntegrationInputs,
) -> dict[tuple[str, int, float | None, str], dict[str, Any]]:
    selected: dict[tuple[str, int, float | None, str], dict[str, Any]] = {}
    for record in inputs.immediate.values():
        if (
            record.get("selection_scope") != "crossfit"
            or float(record["damping"]) != PRIMARY_DAMPING
            or float(record["rcond"]) != PRIMARY_RCOND
        ):
            continue
        key = (
            str(record["parent_id"]) if "parent_id" in record else "",
            int(record["heldout_fold"]),
            float(record["trust_radius"]),
            str(record["action"]),
        )
        # Immediate records use target/seed/horizon axes rather than parent_id. Resolve that
        # source-bound identity through the verified parent ledger.
        parent_key = (
            str(record["target_id"]),
            int(record["seed"]),
            str(record["parent_horizon"]),
        )
        parent = inputs.parents.get(parent_key)
        if parent is None:
            raise RuntimeError(f"immediate row lacks a verified parent {parent_key}")
        key = (str(parent["parent_id"]), key[1], key[2], key[3])
        if key in selected:
            raise RuntimeError(f"duplicate primary recovery source {key}")
        selected[key] = record
    if len(selected) != 768:
        raise RuntimeError(f"primary recovery source set has {len(selected)} rows, expected 768")
    return selected


def _parent_field(record: Mapping[str, Any]) -> GaussianField:
    field = GaussianField.load(str(record["field_path"]), device="cpu")
    if field_sha256(field) != record["field_sha256"]:
        raise RuntimeError("parent tensor hash does not match its source-bound record")
    _strict_field(field)
    return field


def _target_and_masks(
    inputs: _IntegrationInputs,
    target_id: str,
    field: GaussianField,
    fold: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    target_record = inputs.targets.get(target_id)
    if target_record is None:
        raise RuntimeError(f"unknown source-bound target {target_id}")
    target = torch.as_tensor(target_record["image"], dtype=field.means.dtype)
    discovery, heldout = checkerboard_fold_masks(
        int(target.shape[0]), int(target.shape[1]), fold
    )
    return target, discovery.to(device="cpu"), heldout.to(device="cpu")


def _assert_close_metric(name: str, actual: float, expected: Any) -> None:
    if not isinstance(expected, (int, float)) or not math.isfinite(float(expected)):
        raise RuntimeError(f"source {name} is not finite")
    if not math.isclose(actual, float(expected), rel_tol=1e-7, abs_tol=1e-12):
        raise RuntimeError(
            f"source {name} mismatch: reconstructed={actual}, recorded={expected}"
        )


def _reconstruct_integration_source(
    inputs: _IntegrationInputs,
    entry: RecoveryPlanEntry,
    *,
    expected_manifest: Mapping[str, Any] | None = None,
) -> tuple[GaussianField, torch.Tensor, torch.Tensor, torch.Tensor, ImmediateActionState]:
    from benchmarks import residual_tangent_auction_stage1 as manifest_module

    parent_record = next(
        (
            value
            for value in inputs.parents.values()
            if value["parent_id"] == entry.parent_id
        ),
        None,
    )
    if parent_record is None:
        raise RuntimeError(f"recovery plan references unknown parent {entry.parent_id}")
    manifest_recovery_row = next(
        (
            row
            for row in manifest_module.build_manifest()["recovery_plan"]
            if row["cell_id"] == entry.manifest_recovery_cell_id
        ),
        None,
    )
    if manifest_recovery_row is None:
        raise RuntimeError("recovery plan references an unknown frozen manifest cell")
    parent = _parent_field(parent_record)
    target, discovery, heldout = _target_and_masks(
        inputs, str(parent_record["target_id"]), parent, entry.fold
    )
    height, width = int(target.shape[0]), int(target.shape[1])
    if entry.action == BASELINE_ACTION:
        baseline = inputs.baselines[
            (
                str(parent_record["target_id"]),
                int(parent_record["seed"]),
                str(parent_record["horizon_name"]),
                entry.fold,
            )
        ]
        if baseline["cell_id"] != entry.step0_source_cell_id:
            raise RuntimeError("baseline source link does not match the frozen manifest")
        source = capture_immediate_action_state(
            source_cell_id=str(baseline["cell_id"]),
            action=BASELINE_ACTION,
            base_field=parent,
            packet=RecoveryPacket(),
            target=target,
            discovery_mask=discovery,
            heldout_mask=heldout,
        )
        upstream_sha = _sha256_json(baseline)
        upstream_kind = "parent_native_baseline"
        upstream_render_sha = source.expected_render_sha256
        upstream_base_support_sha = freeze_pixel_support(
            parent, height, width
        ).membership
        upstream_base_support_hash = tensor_sha256(upstream_base_support_sha)
    else:
        record = inputs.immediate.get(entry.step0_source_cell_id)
        if record is None:
            raise RuntimeError("action source link is absent from immediate_cells.jsonl")
        required = {
            "status": "ok",
            "selection_scope": "crossfit",
            "action": entry.action,
            "heldout_fold": entry.fold,
            "damping": PRIMARY_DAMPING,
            "rcond": PRIMARY_RCOND,
            "trust_radius": entry.trust_radius,
        }
        drift = [key for key, value in required.items() if record.get(key) != value]
        if drift:
            raise RuntimeError(f"immediate recovery source axes drifted: {drift}")
        parameterization = make_no_opacity_parameterization(parent, height, width)
        base_step = torch.tensor(
            record["base_parameter_step"], dtype=parent.means.dtype
        ).reshape(-1)
        extension = torch.tensor(
            record["extension_coefficients"], dtype=parent.means.dtype
        ).reshape(-1)
        if tensor_sha256(base_step) != record["base_parameter_step_sha256"]:
            raise RuntimeError("serialized base parameter step hash mismatch")
        if tensor_sha256(extension) != record["extension_coefficients_sha256"]:
            raise RuntimeError("serialized extension coefficient hash mismatch")
        base_field = parameterization.field_at(base_step).detached()
        if field_sha256(base_field) != record["updated_field_sha256"]:
            raise RuntimeError("reconstructed immediate field hash mismatch")
        packet = packet_for_action(
            entry.action,
            None
            if entry.action in ("full_j", "tangent6")
            else extension,
            identity=record.get("identity"),
        )
        render = _render_state(
            base_field,
            packet,
            packet.coefficients,
            height,
            width,
            sigma_cutoff=3.0,
            render_chunk=512,
        )
        if tensor_sha256(render) != record["native_render_sha256"]:
            raise RuntimeError("reconstructed immediate native-render hash mismatch")
        support = _support_snapshot(
            base_field, packet, packet.coefficients, height, width, 3.0
        )
        recorded_support = record.get("support")
        if not isinstance(recorded_support, dict):
            raise RuntimeError("immediate row lacks native support provenance")
        if support.base_sha256 != recorded_support.get("native_membership_sha256"):
            raise RuntimeError("reconstructed immediate base-support hash mismatch")
        parent_membership = freeze_pixel_support(parent, height, width).membership
        if tensor_sha256(parent_membership) != recorded_support.get(
            "parent_membership_sha256"
        ):
            raise RuntimeError("immediate parent-support hash mismatch")
        delta = _membership_delta(parent_membership, support.base_membership)
        for name in (
            "added_memberships",
            "removed_memberships",
            "xor_memberships",
        ):
            if delta[name] != recorded_support.get(name):
                raise RuntimeError(f"immediate support {name} mismatch")
        if packet.kind == "birth" and support.birth_sha256 != record.get(
            "birth_membership_sha256"
        ):
            raise RuntimeError("reconstructed birth-support hash mismatch")
        base_render = render_normalized(parent, height, width)
        _assert_close_metric(
            "base_mse", _masked_mse(base_render, target, heldout), record["base_mse"]
        )
        _assert_close_metric(
            "realized_mse", _masked_mse(render, target, heldout), record["realized_mse"]
        )
        source = capture_immediate_action_state(
            source_cell_id=str(record["cell_id"]),
            action=entry.action,
            base_field=base_field,
            packet=packet,
            target=target,
            discovery_mask=discovery,
            heldout_mask=heldout,
            predicted_gain=float(record["predicted_mse_gain"]),
            residual_energy_reduction=(
                float(record["predicted_mse_gain"]) * int(heldout.sum())
            ),
        )
        canonical_source_hashes = {
            "state_sha256": source.expected_state_sha256,
            "packet_sha256": source.expected_packet_sha256,
            "native_render_sha256": source.expected_render_sha256,
            "native_support_sha256": source.expected_support_sha256,
            "metric_sha256": source.expected_metric_sha256,
        }
        if record.get("recovery_source_eligible") is not True:
            raise RuntimeError("primary science row is not marked recovery-source eligible")
        if record.get("source_hashes") != canonical_source_hashes:
            raise RuntimeError(
                "science source_hashes do not exactly match canonical reconstruction"
            )
        upstream_sha = _sha256_json(record)
        upstream_kind = "science_primary_crossfit_immediate"
        upstream_render_sha = str(record["native_render_sha256"])
        upstream_base_support_hash = str(recorded_support["native_membership_sha256"])

    canonical = {
        "state_sha256": source.expected_state_sha256,
        "packet_sha256": source.expected_packet_sha256,
        "native_render_sha256": source.expected_render_sha256,
        "native_support_sha256": source.expected_support_sha256,
        "metric_sha256": source.expected_metric_sha256,
    }
    source_binding = {
        "upstream_kind": upstream_kind,
        "upstream_cell_id": entry.step0_source_cell_id,
        "upstream_record_sha256": upstream_sha,
        "upstream_native_render_sha256": upstream_render_sha,
        "upstream_base_support_sha256": upstream_base_support_hash,
        "parent_field_sha256": str(parent_record["field_sha256"]),
        "parent_record_sha256": _sha256_json(parent_record),
        "target_pixel_sha256": str(parent_record["target_pixel_sha256"]),
        "manifest_recovery_cell_id": entry.manifest_recovery_cell_id,
        "manifest_recovery_row_sha256": _sha256_json(manifest_recovery_row),
        "canonical_source_hashes": canonical,
    }
    if entry.action == BASELINE_ACTION:
        source_binding["baseline_manifest_row_sha256"] = upstream_sha
    if expected_manifest is not None:
        if expected_manifest.get("source_binding") != source_binding:
            raise RuntimeError("reconstructed recovery source binding is stale")
        if expected_manifest.get("source_binding_sha256") != _sha256_json(source_binding):
            raise RuntimeError("persisted recovery source binding hash mismatch")
    return parent, target, discovery, heldout, source


def _integration_plan(
    inputs: _IntegrationInputs,
) -> tuple[tuple[RecoveryPlanEntry, ...], dict[tuple[str, int, float | None, str], dict[str, Any]]]:
    primary = _primary_action_records(inputs)
    parent_keys: list[RecoveryParentKey] = []
    links: dict[tuple[str, int, float | None, str], str] = {}
    for (target_id, seed, horizon), parent in sorted(inputs.parents.items()):
        target = inputs.targets[target_id]
        parent_keys.append(
            RecoveryParentKey(
                family=str(target["family"]),
                seed=seed,
                parent_checkpoint=int(parent["iterations"]),
                parent_id=str(parent["parent_id"]),
            )
        )
        for fold in (0, 1):
            baseline = inputs.baselines[(target_id, seed, horizon, fold)]
            links[(str(parent["parent_id"]), fold, None, BASELINE_ACTION)] = str(
                baseline["cell_id"]
            )
            for radius in TRUST_RADII:
                for action in ACTION_NAMES:
                    key = (str(parent["parent_id"]), fold, radius, action)
                    links[key] = str(primary[key]["cell_id"])
    return build_recovery_plan(parent_keys, links), primary


def prepare_recovery_sources(
    *,
    science_dir: str | Path,
    runner_dir: str | Path,
    outdir: str | Path,
) -> dict[str, Any]:
    """Verify upstream rows and persist the explicit canonical recovery-source rebind."""

    inputs = _load_integration_inputs(science_dir, runner_dir)
    from benchmarks import residual_tangent_auction_stage1 as manifest_module

    recovery_manifest_by_id = {
        str(row["cell_id"]): row
        for row in manifest_module.build_manifest()["recovery_plan"]
    }
    plan, _primary = _integration_plan(inputs)
    output = Path(outdir).resolve()
    binding = {
        "protocol": PROTOCOL,
        "source_sha256": _live_source_hashes(),
        "science_config_sha256": _sha256_file(inputs.science_dir / "science_config.json"),
        "science_binding_sha256": inputs.science_config["binding_sha256"],
        "immediate_cells_sha256": _sha256_file(inputs.science_dir / "immediate_cells.jsonl"),
        "runner_config_sha256": _sha256_file(inputs.runner_dir / "runner_config.json"),
        "runner_binding_sha256": inputs.runner_config["binding_sha256"],
        "parents_jsonl_sha256": _sha256_file(inputs.runner_dir / "parents.jsonl"),
        "science_dir": str(inputs.science_dir),
        "runner_dir": str(inputs.runner_dir),
        "selection_scope": "crossfit_only",
        "selected_damping": PRIMARY_DAMPING,
        "selected_rcond": PRIMARY_RCOND,
        "trajectory_count": EXPECTED_TRAJECTORIES,
        "logical_checkpoint_count": EXPECTED_LOGICAL_RECORDS,
        "new_recovery_render_count": EXPECTED_NEW_RENDER_RECORDS,
    }
    binding_sha256 = _sha256_json(binding)
    source_records: list[dict[str, Any]] = []
    for entry in plan:
        parent, target, discovery, heldout, source = _reconstruct_integration_source(
            inputs, entry
        )
        canonical = {
            "state_sha256": source.expected_state_sha256,
            "packet_sha256": source.expected_packet_sha256,
            "native_render_sha256": source.expected_render_sha256,
            "native_support_sha256": source.expected_support_sha256,
            "metric_sha256": source.expected_metric_sha256,
        }
        upstream = (
            inputs.baselines[
                (
                    next(
                        str(value["target_id"])
                        for value in inputs.parents.values()
                        if value["parent_id"] == entry.parent_id
                    ),
                    entry.seed,
                    "underfit" if entry.parent_checkpoint == 24 else "near_plateau",
                    entry.fold,
                )
            ]
            if entry.action == BASELINE_ACTION
            else inputs.immediate[entry.step0_source_cell_id]
        )
        parent_record = next(
            value
            for value in inputs.parents.values()
            if value["parent_id"] == entry.parent_id
        )
        support = _support_snapshot(
            source.base_field,
            source.packet,
            source.packet.coefficients,
            int(target.shape[0]),
            int(target.shape[1]),
            3.0,
        )
        source_binding = {
            "upstream_kind": (
                "parent_native_baseline"
                if entry.action == BASELINE_ACTION
                else "science_primary_crossfit_immediate"
            ),
            "upstream_cell_id": entry.step0_source_cell_id,
            "upstream_record_sha256": _sha256_json(upstream),
            "upstream_native_render_sha256": (
                source.expected_render_sha256
                if entry.action == BASELINE_ACTION
                else str(upstream["native_render_sha256"])
            ),
            "upstream_base_support_sha256": support.base_sha256,
            "parent_field_sha256": str(parent_record["field_sha256"]),
            "parent_record_sha256": _sha256_json(parent_record),
            "target_pixel_sha256": str(parent_record["target_pixel_sha256"]),
            "manifest_recovery_cell_id": entry.manifest_recovery_cell_id,
            "manifest_recovery_row_sha256": _sha256_json(
                recovery_manifest_by_id[entry.manifest_recovery_cell_id]
            ),
            "canonical_source_hashes": canonical,
        }
        if entry.action == BASELINE_ACTION:
            source_binding["baseline_manifest_row_sha256"] = _sha256_json(upstream)
        source_records.append(
            {
                "trajectory_id": entry.trajectory_id,
                "binding_sha256": binding_sha256,
                "plan": entry.stable_key,
                "source_binding": source_binding,
                "source_binding_sha256": _sha256_json(source_binding),
                "discovery_mask_sha256": tensor_sha256(discovery),
                "scoring_mask_sha256": tensor_sha256(heldout),
                "parent_state_sha256": field_sha256(parent),
            }
        )
    source_path = output / "recovery_sources.jsonl"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = source_path.with_suffix(".jsonl.tmp")
    temporary.write_text(
        "".join(
            json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
            + "\n"
            for value in source_records
        ),
        encoding="utf-8",
    )
    os.replace(temporary, source_path)
    config = {
        "binding": binding,
        "binding_sha256": binding_sha256,
        "recovery_sources_sha256": _sha256_file(source_path),
        "prepared_source_count": len(source_records),
        "commands": {
            "prepare": (
                f"PYTHONPATH=src:. python -m {__name__} --phase prepare "
                f"--science-dir {shlex.quote(str(inputs.science_dir))} "
                f"--runner-dir {shlex.quote(str(inputs.runner_dir))} "
                f"--outdir {shlex.quote(str(output))}"
            ),
            "run": (
                f"PYTHONPATH=src:. python -m {__name__} --phase run "
                f"--outdir {shlex.quote(str(output))}"
            ),
        },
        "scientific_interpretation_present": False,
        "method_promotion_authorized": False,
        "compression_claim_authorized": False,
    }
    config_path = output / "recovery_config.json"
    if config_path.exists() and _strict_json(config_path) != config:
        raise RuntimeError("recovery source binding changed; use a fresh output directory")
    _write_json_atomic(config_path, config)
    return config


def _load_prepared_recovery(
    outdir: str | Path,
) -> tuple[dict[str, Any], _IntegrationInputs, tuple[RecoveryPlanEntry, ...], dict[str, dict[str, Any]]]:
    output = Path(outdir).resolve()
    config = _strict_json(output / "recovery_config.json")
    binding = config.get("binding")
    if not isinstance(binding, dict) or config.get("binding_sha256") != _sha256_json(binding):
        raise RuntimeError("prepared recovery binding hash mismatch")
    if binding.get("source_sha256") != _live_source_hashes():
        raise RuntimeError("prepared recovery source binding is stale")
    sources_path = output / "recovery_sources.jsonl"
    if _sha256_file(sources_path) != config.get("recovery_sources_sha256"):
        raise RuntimeError("prepared recovery source manifest changed")
    science_dir = Path(str(binding["science_dir"]))
    runner_dir = Path(str(binding["runner_dir"]))
    artifact_hashes = {
        science_dir / "science_config.json": binding["science_config_sha256"],
        science_dir / "immediate_cells.jsonl": binding["immediate_cells_sha256"],
        runner_dir / "runner_config.json": binding["runner_config_sha256"],
        runner_dir / "parents.jsonl": binding["parents_jsonl_sha256"],
    }
    stale = [str(path) for path, expected in artifact_hashes.items() if _sha256_file(path) != expected]
    if stale:
        raise RuntimeError(f"prepared recovery upstream artifacts are stale: {stale}")
    inputs = _load_integration_inputs(science_dir, runner_dir)
    sources = _existing_jsonl(sources_path, "trajectory_id")
    entries = tuple(
        RecoveryPlanEntry(**record["plan"])
        for _identifier, record in sorted(sources.items())
    )
    validate_recovery_plan(entries)
    if any(record.get("binding_sha256") != config["binding_sha256"] for record in sources.values()):
        raise RuntimeError("prepared source manifest mixes recovery bindings")
    return config, inputs, entries, sources


def run_prepared_recovery(
    *,
    outdir: str | Path,
    shard_index: int = 0,
    shard_count: int = 1,
    max_trajectories: int | None = None,
) -> dict[str, Any]:
    """Execute a resumable shard from the verified 816-source manifest."""

    config, inputs, entries, sources = _load_prepared_recovery(outdir)
    if shard_count <= 0 or not 0 <= shard_index < shard_count:
        raise ValueError("invalid recovery shard index/count")
    chosen = [
        entry
        for entry in entries
        if int(hashlib.sha256(entry.trajectory_id.encode("ascii")).hexdigest(), 16)
        % shard_count
        == shard_index
    ]
    if max_trajectories is not None:
        if max_trajectories <= 0:
            raise ValueError("max_trajectories must be positive")
        chosen = chosen[:max_trajectories]
    # Analyzer-visible evidence must share the immediate component's scientific binding. The
    # stronger recovery source/code rebind remains independently persisted in recovery_config and
    # recovery_sources, and is checked above before any trajectory is admitted.
    binding_sha256 = str(config["binding"]["science_binding_sha256"])

    def executor(entry: RecoveryPlanEntry) -> RecoveryResult:
        manifest = sources[entry.trajectory_id]
        parent, target, discovery, heldout, source = _reconstruct_integration_source(
            inputs, entry, expected_manifest=manifest
        )
        return run_recovery(
            parent=parent,
            source=source,
            target=target,
            discovery_mask=discovery,
            heldout_mask=heldout,
            context=RecoveryContext(
                binding_sha256=binding_sha256,
                family=entry.family,
                seed=entry.seed,
                parent_checkpoint=entry.parent_checkpoint,
                fold=entry.fold,
                trust_radius=entry.trust_radius,
                parent_id=entry.parent_id,
                trajectory_id=entry.trajectory_id,
                recovery_binding_sha256=str(config["binding_sha256"]),
                source_binding_sha256=str(manifest["source_binding_sha256"]),
                manifest_recovery_cell_id=entry.manifest_recovery_cell_id,
            ),
        )

    output = Path(outdir).resolve()
    execution_error: str | None = None
    try:
        counts = execute_recovery_plan(
            chosen,
            executor,
            binding_sha256=binding_sha256,
            terminal_jsonl=output / "recovery_trajectories.jsonl",
            checkpoint_jsonl=output / "recovery_checkpoints.jsonl",
            evidence_jsonl=output / "matched_recovery_evidence.jsonl",
        )
    except Exception as exc:
        execution_error = f"{type(exc).__name__}: {exc}"
        counts = {"selected": len(chosen), "skipped_terminal": 0, "ok": 0, "error": 1}
    validation = validate_final_recovery_ledger(output)
    state = {
        "phase": "run",
        **counts,
        "shard_index": shard_index,
        "shard_count": shard_count,
        "max_trajectories": max_trajectories,
        "binding_sha256": binding_sha256,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "execution_error": execution_error,
        "component_complete": validation["terminal_trajectories"]
        == EXPECTED_TRAJECTORIES,
        "component_valid": execution_error is None and validation["component_valid"],
        "validation": validation,
        "scientific_interpretation_present": False,
        "method_promotion_authorized": False,
        "compression_claim_authorized": False,
    }
    _write_json_atomic(output / "recovery_validation.json", validation)
    _write_json_atomic(output / "recovery_run_state.json", state)
    return state


def _append_jsonl(path: Path, record: Mapping[str, Any]) -> None:
    payload = json.dumps(
        dict(record), sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(payload + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _existing_jsonl(path: Path, key: str) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    result: dict[str, dict[str, Any]] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            raise ValueError(f"blank JSONL line at {path}:{line_number}")
        value = json.loads(
            line,
            parse_constant=lambda item: (_ for _ in ()).throw(
                ValueError(f"non-strict JSON constant {item}")
            ),
        )
        if not isinstance(value, dict) or key not in value:
            raise ValueError(f"{path}:{line_number} lacks {key!r}")
        identifier = str(value[key])
        if identifier in result:
            raise ValueError(f"duplicate {key}={identifier} in {path}")
        result[identifier] = value
    return result


def _error_evidence_records(
    entry: RecoveryPlanEntry, binding_sha256: str, error: str
) -> tuple[dict[str, Any], ...]:
    if entry.action not in MATCHED_ACTIONS:
        return ()
    result: list[dict[str, Any]] = []
    for checkpoint in NEW_RENDER_CHECKPOINTS:
        record: dict[str, Any] = {
            "schema": EVIDENCE_SCHEMA_ID,
            "record_kind": EVIDENCE_RECORD_KIND,
            "component": "crossfit_recovery",
            "status": "error",
            "error": error,
            "binding_sha256": binding_sha256,
            "selection_scope": CROSS_FIT_SCOPE,
            "ledger": MATCHED_LEDGER,
            "family": entry.family,
            "seed": entry.seed,
            "parent_checkpoint": entry.parent_checkpoint,
            "fold": entry.fold,
            "action": entry.action,
            "damping": PRIMARY_DAMPING,
            "rcond": PRIMARY_RCOND,
            "trust_radius": entry.trust_radius,
            "checkpoint": checkpoint,
        }
        record["cell_id"] = stable_cell_id(record)
        result.append(record)
    return tuple(result)


def execute_recovery_plan(
    entries: Sequence[RecoveryPlanEntry],
    executor: Callable[[RecoveryPlanEntry], RecoveryResult],
    *,
    binding_sha256: str,
    terminal_jsonl: str | Path,
    checkpoint_jsonl: str | Path,
    evidence_jsonl: str | Path,
    shard_index: int = 0,
    shard_count: int = 1,
) -> dict[str, int]:
    """Run a deterministic shard with terminal-error and duplicate-safe resume behavior."""

    validate_recovery_plan(entries, expected_count=None)
    if not _is_sha256(binding_sha256):
        raise ValueError("plan execution binding must be a lowercase SHA-256")
    if shard_count <= 0 or not 0 <= shard_index < shard_count:
        raise ValueError("invalid recovery shard index/count")
    terminal_path = Path(terminal_jsonl)
    checkpoint_path = Path(checkpoint_jsonl)
    evidence_path = Path(evidence_jsonl)
    terminals = _existing_jsonl(terminal_path, "trajectory_id")
    checkpoints = _existing_jsonl(checkpoint_path, "checkpoint_id")
    evidence = _existing_jsonl(evidence_path, "cell_id")
    counts = {"selected": 0, "skipped_terminal": 0, "ok": 0, "error": 0}
    for entry in entries:
        shard_value = int(hashlib.sha256(entry.trajectory_id.encode("ascii")).hexdigest(), 16)
        if shard_value % shard_count != shard_index:
            continue
        counts["selected"] += 1
        if entry.trajectory_id in terminals:
            terminal = terminals[entry.trajectory_id]
            if terminal.get("status") != "ok":
                raise RuntimeError(
                    f"cannot resume terminal error trajectory {entry.trajectory_id}"
                )
            if terminal.get("binding_sha256") != binding_sha256:
                raise RuntimeError(
                    f"cannot resume stale-binding trajectory {entry.trajectory_id}"
                )
            if terminal.get("step0_source_cell_id") != entry.step0_source_cell_id:
                raise RuntimeError("resumed terminal has the wrong step-0 source link")
            if (
                terminal.get("manifest_recovery_cell_id")
                != entry.manifest_recovery_cell_id
            ):
                raise RuntimeError("resumed terminal has the wrong recovery manifest link")
            checkpoint_ids = terminal.get("logical_checkpoint_ids")
            if not isinstance(checkpoint_ids, list) or len(checkpoint_ids) != 3:
                raise RuntimeError("resumed terminal has an incomplete checkpoint ledger")
            linked = [checkpoints.get(str(identifier)) for identifier in checkpoint_ids]
            if any(record is None for record in linked):
                raise RuntimeError("resumed terminal references missing checkpoint rows")
            if [record["logical_checkpoint"] for record in linked if record] != [0, 20, 100]:
                raise RuntimeError("resumed terminal checkpoint steps are invalid")
            if any(
                record.get("status") != "ok"
                or record.get("binding_sha256") != binding_sha256
                or record.get("trajectory_id") != entry.trajectory_id
                for record in linked
                if record is not None
            ):
                raise RuntimeError("resumed terminal checkpoint rows are stale or invalid")
            evidence_ids = terminal.get("evidence_cell_ids")
            expected_evidence_count = 2 if entry.action in MATCHED_ACTIONS else 0
            if not isinstance(evidence_ids, list) or len(evidence_ids) != expected_evidence_count:
                raise RuntimeError("resumed terminal evidence ledger is incomplete")
            if any(
                identifier not in evidence
                or evidence[identifier].get("status") != "ok"
                or evidence[identifier].get("binding_sha256") != binding_sha256
                for identifier in evidence_ids
            ):
                raise RuntimeError("resumed terminal evidence rows are stale or missing")
            counts["skipped_terminal"] += 1
            continue
        orphan_checkpoints = [
            record
            for record in checkpoints.values()
            if record.get("trajectory_id") == entry.trajectory_id
        ]
        orphan_evidence = [
            record
            for record in evidence.values()
            if isinstance(record.get("metadata"), dict)
            and record["metadata"].get("trajectory_id") == entry.trajectory_id
        ]
        if orphan_checkpoints or orphan_evidence:
            raise RuntimeError(
                "partial trajectory rows exist without a terminal; refusing unsafe resume"
            )
        try:
            result = executor(entry)
            if result.terminal_record.get("status") != "ok":
                raise ValueError("executor returned a non-ok terminal record")
            for record in result.checkpoint_records:
                identifier = str(record["checkpoint_id"])
                if identifier not in checkpoints:
                    _append_jsonl(checkpoint_path, record)
                    checkpoints[identifier] = record
                elif checkpoints[identifier] != record:
                    raise RuntimeError("stale conflicting resumed checkpoint record")
            for record in result.evidence_records:
                identifier = str(record["cell_id"])
                if identifier not in evidence:
                    _append_jsonl(evidence_path, record)
                    evidence[identifier] = record
                elif evidence[identifier] != record:
                    raise RuntimeError("stale conflicting resumed evidence record")
            terminal = result.terminal_record
            if terminal.get("trajectory_id") != entry.trajectory_id:
                # The execution trajectory binds more context than the plan ID.  Persist both and
                # use the plan ID as the resumable manifest key.
                terminal = {
                    **terminal,
                    "execution_trajectory_id": terminal.get("trajectory_id"),
                    "trajectory_id": entry.trajectory_id,
                }
            _append_jsonl(terminal_path, terminal)
            terminals[entry.trajectory_id] = terminal
            counts["ok"] += 1
        except Exception as exc:  # terminal error rows are part of the fail-closed protocol
            error_text = f"{type(exc).__name__}: {exc}"
            for record in _error_evidence_records(entry, binding_sha256, error_text):
                identifier = str(record["cell_id"])
                if identifier not in evidence:
                    _append_jsonl(evidence_path, record)
                    evidence[identifier] = record
            terminal = {
                "schema": RAW_TRAJECTORY_SCHEMA,
                "record_kind": RAW_TRAJECTORY_KIND,
                "trajectory_id": entry.trajectory_id,
                "status": "error",
                "error": error_text,
                "protocol": PROTOCOL,
                "binding_sha256": binding_sha256,
                **entry.stable_key,
            }
            _append_jsonl(terminal_path, terminal)
            terminals[entry.trajectory_id] = terminal
            counts["error"] += 1
            raise RuntimeError(
                f"recovery trajectory failed and was marked terminal error: {entry.trajectory_id}"
            ) from exc
    return counts


def validate_final_recovery_ledger(outdir: str | Path) -> dict[str, Any]:
    """Fail closed unless the complete frozen recovery component is internally exact."""

    errors: list[str] = []
    output = Path(outdir).resolve()
    try:
        config, _inputs, entries, sources = _load_prepared_recovery(output)
        science_binding = str(config["binding"]["science_binding_sha256"])
        recovery_binding = str(config["binding_sha256"])
        terminals = _existing_jsonl(
            output / "recovery_trajectories.jsonl", "trajectory_id"
        )
        checkpoints = _existing_jsonl(
            output / "recovery_checkpoints.jsonl", "checkpoint_id"
        )
        evidence = _existing_jsonl(
            output / "matched_recovery_evidence.jsonl", "cell_id"
        )
    except Exception as exc:
        return {
            "component_valid": False,
            "errors": [f"artifact load failed: {type(exc).__name__}: {exc}"],
            "terminal_trajectories": 0,
            "logical_checkpoints": 0,
            "new_recovery_renders": 0,
            "matched_evidence_rows": 0,
        }

    expected_trajectory_ids = {entry.trajectory_id for entry in entries}
    if len(terminals) != EXPECTED_TRAJECTORIES or set(terminals) != expected_trajectory_ids:
        errors.append(
            f"terminal ledger mismatch: observed={len(terminals)}, expected={EXPECTED_TRAJECTORIES}"
        )
    if len(checkpoints) != EXPECTED_LOGICAL_RECORDS:
        errors.append(
            f"checkpoint ledger mismatch: observed={len(checkpoints)}, expected={EXPECTED_LOGICAL_RECORDS}"
        )
    if len(evidence) != 768:
        errors.append(f"evidence ledger mismatch: observed={len(evidence)}, expected=768")
    if len({entry.manifest_recovery_cell_id for entry in entries}) != EXPECTED_TRAJECTORIES:
        errors.append("recovery manifest links are not one-to-one")
    if len({entry.step0_source_cell_id for entry in entries}) != EXPECTED_TRAJECTORIES:
        errors.append("step-0 source links are not one-to-one")

    entry_by_id = {entry.trajectory_id: entry for entry in entries}
    expected_checkpoint_ids: set[str] = set()
    expected_evidence_ids: set[str] = set()
    new_render_count = 0
    for trajectory_id, entry in entry_by_id.items():
        terminal = terminals.get(trajectory_id)
        source_manifest = sources.get(trajectory_id)
        if terminal is None or source_manifest is None:
            continue
        source_binding_sha = source_manifest.get("source_binding_sha256")
        if terminal.get("status") != "ok" or terminal.get("error") is not None:
            errors.append(f"non-ok terminal trajectory {trajectory_id}")
        for name, expected in (
            ("binding_sha256", science_binding),
            ("recovery_binding_sha256", recovery_binding),
            ("source_binding_sha256", source_binding_sha),
            ("step0_source_cell_id", entry.step0_source_cell_id),
            ("manifest_recovery_cell_id", entry.manifest_recovery_cell_id),
        ):
            if terminal.get(name) != expected:
                errors.append(f"terminal {trajectory_id} has stale {name}")
        wanted_checkpoint_ids = [
            _checkpoint_id(trajectory_id, checkpoint)
            for checkpoint in LOGICAL_CHECKPOINTS
        ]
        expected_checkpoint_ids.update(wanted_checkpoint_ids)
        if terminal.get("logical_checkpoint_ids") != wanted_checkpoint_ids:
            errors.append(f"terminal {trajectory_id} checkpoint links are invalid")
        if terminal.get("logical_checkpoint_count") != 3:
            errors.append(f"terminal {trajectory_id} logical count is invalid")
        if terminal.get("new_render_checkpoint_count") != 2:
            errors.append(f"terminal {trajectory_id} new-render count is invalid")
        canonical_hashes = source_manifest["source_binding"]["canonical_source_hashes"]
        linked_rows: list[dict[str, Any]] = []
        for checkpoint, checkpoint_id in zip(
            LOGICAL_CHECKPOINTS, wanted_checkpoint_ids, strict=True
        ):
            row = checkpoints.get(checkpoint_id)
            if row is None:
                errors.append(f"missing checkpoint {checkpoint_id}")
                continue
            linked_rows.append(row)
            expected_new = checkpoint in NEW_RENDER_CHECKPOINTS
            new_render_count += int(row.get("newly_rendered_recovery_checkpoint") is True)
            checks = {
                "status": "ok",
                "error": None,
                "trajectory_id": trajectory_id,
                "logical_checkpoint": checkpoint,
                "newly_rendered_recovery_checkpoint": expected_new,
                "binding_sha256": science_binding,
                "recovery_binding_sha256": recovery_binding,
                "source_binding_sha256": source_binding_sha,
                "manifest_recovery_cell_id": entry.manifest_recovery_cell_id,
                "step0_source_cell_id": entry.step0_source_cell_id,
            }
            if any(row.get(name) != value for name, value in checks.items()):
                errors.append(f"checkpoint {checkpoint_id} failed stable-link validation")
            if checkpoint == 0:
                source_checks = {
                    "state_sha256": canonical_hashes["state_sha256"],
                    "packet_sha256": canonical_hashes["packet_sha256"],
                    "native_render_sha256": canonical_hashes["native_render_sha256"],
                    "native_support_sha256": canonical_hashes["native_support_sha256"],
                    "source_metric_sha256": canonical_hashes["metric_sha256"],
                }
                if any(row.get(name) != value for name, value in source_checks.items()):
                    errors.append(f"checkpoint {checkpoint_id} does not equal its source")
        evidence_ids = terminal.get("evidence_cell_ids")
        expected_count = 2 if entry.action in MATCHED_ACTIONS else 0
        if not isinstance(evidence_ids, list) or len(evidence_ids) != expected_count:
            errors.append(f"terminal {trajectory_id} evidence links are invalid")
        else:
            expected_evidence_ids.update(str(value) for value in evidence_ids)
            for cell_id in evidence_ids:
                row = evidence.get(str(cell_id))
                if row is None:
                    errors.append(f"missing evidence row {cell_id}")
                    continue
                metadata = row.get("metadata")
                if (
                    row.get("status") != "ok"
                    or row.get("binding_sha256") != science_binding
                    or not isinstance(metadata, dict)
                    or metadata.get("trajectory_id") != trajectory_id
                    or metadata.get("recovery_binding_sha256") != recovery_binding
                    or metadata.get("source_binding_sha256") != source_binding_sha
                    or metadata.get("manifest_recovery_cell_id")
                    != entry.manifest_recovery_cell_id
                ):
                    errors.append(f"evidence row {cell_id} failed audited-join validation")

    if set(checkpoints) != expected_checkpoint_ids:
        errors.append("checkpoint IDs are missing or unexpected")
    if set(evidence) != expected_evidence_ids:
        errors.append("evidence IDs are missing or unexpected")
    if new_render_count != EXPECTED_NEW_RENDER_RECORDS:
        errors.append(
            f"new-render ledger mismatch: observed={new_render_count}, expected={EXPECTED_NEW_RENDER_RECORDS}"
        )
    # Reuse the strict analyzer parser for schema, stable IDs, finite metrics, and binding.
    from benchmarks.residual_tangent_auction_stage1_analysis import load_jsonl

    loaded_evidence = load_jsonl(
        output / "matched_recovery_evidence.jsonl",
        expected_ids=expected_evidence_ids,
        expected_binding_sha256=science_binding,
    )
    if not loaded_evidence.available:
        errors.extend(f"analyzer: {value}" for value in loaded_evidence.errors[:20])
    return {
        "component_valid": not errors,
        "errors": errors[:100],
        "terminal_trajectories": len(terminals),
        "logical_checkpoints": len(checkpoints),
        "new_recovery_renders": new_render_count,
        "matched_evidence_rows": len(evidence),
        "science_binding_sha256": science_binding,
        "recovery_binding_sha256": recovery_binding,
    }


def toy_checkerboard_masks(height: int, width: int, fold: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Public convenience wrapper used by source integrations and toy preflight."""

    discovery, heldout = checkerboard_fold_masks(height, width, fold)
    return discovery.to(device="cpu"), heldout.to(device="cpu")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("prepare", "run"), required=True)
    parser.add_argument(
        "--outdir", default="results/bench009_tangent_auction_stage1_recovery"
    )
    parser.add_argument("--science-dir", default=None)
    parser.add_argument("--runner-dir", default=None)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--max-trajectories", type=int, default=None)
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    if args.phase == "prepare":
        if args.science_dir is None or args.runner_dir is None:
            parser.error("prepare requires --science-dir and --runner-dir")
        if args.shard_index != 0 or args.shard_count != 1 or args.max_trajectories is not None:
            parser.error("prepare does not accept run shard/limit options")
        result = prepare_recovery_sources(
            science_dir=args.science_dir,
            runner_dir=args.runner_dir,
            outdir=args.outdir,
        )
    else:
        if args.science_dir is not None or args.runner_dir is not None:
            parser.error("run reads upstream paths from the prepared binding")
        result = run_prepared_recovery(
            outdir=args.outdir,
            shard_index=args.shard_index,
            shard_count=args.shard_count,
            max_trajectories=args.max_trajectories,
        )
    invocation = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "command": shlex.join([sys.executable, "-m", __name__, *(argv or sys.argv[1:])]),
        "phase": args.phase,
        "result": result,
    }
    _append_jsonl(Path(args.outdir) / "recovery_invocations.jsonl", invocation)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    if args.phase == "run" and result.get("component_valid") is not True:
        return 1
    return 0


__all__ = [
    "ACTION_NAMES",
    "ADAM_BETAS",
    "ADAM_EPS",
    "BASELINE_ACTION",
    "EXPECTED_LOGICAL_RECORDS",
    "EXPECTED_NEW_RENDER_RECORDS",
    "EXPECTED_TRAJECTORIES",
    "ImmediateActionState",
    "LOGICAL_CHECKPOINTS",
    "MATCHED_ACTIONS",
    "NEW_RENDER_CHECKPOINTS",
    "PROTOCOL",
    "RECOVERY_STEPS",
    "RecoveryContext",
    "RecoveryPacket",
    "RecoveryParentKey",
    "RecoveryPlanEntry",
    "RecoveryResult",
    "action_state_sha256",
    "build_recovery_plan",
    "capture_immediate_action_state",
    "execute_recovery_plan",
    "immediate_state_from_payload",
    "packet_for_action",
    "packet_sha256",
    "prepare_recovery_sources",
    "run_recovery",
    "run_prepared_recovery",
    "source_metric_payload",
    "source_metric_sha256",
    "toy_checkerboard_masks",
    "validate_recovery_plan",
    "validate_final_recovery_ledger",
]


if __name__ == "__main__":
    raise SystemExit(main())
