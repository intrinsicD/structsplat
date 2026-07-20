"""Quality authority helpers for the frozen COMP-011 SSP2V experiment.

This module deliberately separates target/render tensor construction, exact CPU metrics,
conservative three-render envelopes, and target-blind candidate selection from the lifecycle
runner.  It never opens a development asset by itself.  The lifecycle layer is responsible for
supplying already-authorized immutable bytes and for spawning one fresh metric worker per arm and
render repetition.
"""

from __future__ import annotations

import hashlib
import importlib.util
import math
import os
import re
import stat
import struct
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np


QUALITY_SCHEMA = "structsplat.comp011.quality.v1"
RENDERER_RELATIVE_PATH = "runtime/renderer/structsplat_render_ext.so"
REQUIRED_RENDERER_PRELOAD = "/usr/lib/x86_64-linux-gnu/libstdc++.so.6"
FROZEN_THREAD_ENVIRONMENT = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}
METRIC_NAMES = ("psnr", "ms_ssim", "lpips")
VARIANT_LABELS = (
    "flat_s2_k64_l192",
    "flat_s2_k128_l384",
    "flat_s3_k64_l320",
    "flat_s3_k128_l640",
    "rvq_s2_k64",
    "rvq_s2_k128",
    "rvq_s3_k64",
    "rvq_s3_k128",
)
RENDER_SCHEDULE = (
    ("primary", *VARIANT_LABELS),
    (*reversed(VARIANT_LABELS), "primary"),
    ("primary", *VARIANT_LABELS),
)

PSNR_SPREAD_CAP = 0.001
MS_SSIM_SPREAD_CAP = 0.00001
LPIPS_SPREAD_CAP = 0.00001
PSNR_THRESHOLD = 0.05
MS_SSIM_THRESHOLD = 0.0005
LPIPS_THRESHOLD = 0.001
PSNR_EXCLUSION_HALF_WIDTH = 0.002
MS_SSIM_EXCLUSION_HALF_WIDTH = 0.00002
LPIPS_EXCLUSION_HALF_WIDTH = 0.00002
MEDIAN_PSNR_PASS_MAX = 0.018
MEDIAN_PSNR_FAIL_MIN_EXCLUSIVE = 0.022

_METRIC_WORKER_CONFIGURED = False
_PERSISTED_RENDERER_CAPABILITY: dict[str, Any] | None = None


class QualityError(ValueError):
    """A frozen COMP-011 quality invariant was violated."""


@dataclass(frozen=True)
class MetricTriple:
    """One exact fresh-worker metric result."""

    psnr: float
    ms_ssim: float
    lpips: float

    def __post_init__(self) -> None:
        for name in METRIC_NAMES:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise QualityError(f"{name} must be a real scalar")
            if not math.isfinite(float(value)):
                raise QualityError(f"{name} must be finite")

    def as_dict(self) -> dict[str, float]:
        return {name: float(getattr(self, name)) for name in METRIC_NAMES}


def sha256_bytes(payload: bytes) -> str:
    """Return a lowercase SHA-256 digest."""
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    """Hash a regular file without following a changing path after validation."""
    resolved = Path(path).resolve(strict=True)
    if not resolved.is_file():
        raise QualityError(f"not a regular file: {resolved}")
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_digest(value: str, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise QualityError(f"{name} must be a lowercase SHA-256 digest")
    return value


def tensor_record(tensor: Any) -> dict[str, Any]:
    """Bind a finite CPU contiguous tensor's dtype, shape, and physical bytes."""
    import torch

    if not isinstance(tensor, torch.Tensor):
        raise QualityError("tensor_record requires a torch.Tensor")
    if tensor.device.type != "cpu" or not tensor.is_contiguous():
        raise QualityError("tensor_record requires a contiguous CPU tensor")
    if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
        raise QualityError("tensor_record refuses non-finite values")
    array = tensor.detach().numpy()
    payload = array.tobytes(order="C")
    return {
        "dtype": str(tensor.dtype),
        "shape": list(tensor.shape),
        "bytes": len(payload),
        "sha256": sha256_bytes(payload),
    }


def strict_target_tensor(target_u8: Any) -> Any:
    """Apply the frozen strict-RGB ``uint8/255`` conversion on CPU."""
    import torch

    if isinstance(target_u8, np.ndarray):
        if target_u8.dtype != np.uint8 or target_u8.ndim != 3 or target_u8.shape[2] != 3:
            raise QualityError("target must be uint8[H,W,3]")
        if not target_u8.flags.c_contiguous:
            raise QualityError("target array must be C contiguous")
        source = torch.from_numpy(target_u8.copy(order="C"))
    elif isinstance(target_u8, torch.Tensor):
        source = target_u8
    else:
        raise QualityError("target must be a NumPy array or torch tensor")
    if (
        source.device.type != "cpu"
        or source.dtype != torch.uint8
        or source.ndim != 3
        or source.shape[2] != 3
        or not source.is_contiguous()
    ):
        raise QualityError("target must be contiguous CPU uint8[H,W,3]")
    hwc = source.to(dtype=torch.float32, device="cpu") / 255.0
    return hwc.permute(2, 0, 1).unsqueeze(0).contiguous()


def strict_render_tensor(render_hwc: Any) -> Any:
    """Clamp one CUDA/CPU float32 HWC render and create a fresh CPU NCHW tensor."""
    import torch

    if not isinstance(render_hwc, torch.Tensor):
        raise QualityError("render must be a torch tensor")
    if render_hwc.dtype != torch.float32 or render_hwc.ndim != 3 or render_hwc.shape[2] != 3:
        raise QualityError("render must be float32[H,W,3]")
    if not bool(torch.isfinite(render_hwc).all()):
        raise QualityError("render contains a non-finite value")
    clamped_device = render_hwc.clamp(0.0, 1.0).detach().contiguous()
    if clamped_device.device.type == "cuda":
        torch.cuda.synchronize(clamped_device.device)
    clamped_hwc = clamped_device.to(device="cpu", dtype=torch.float32).contiguous().clone()
    return clamped_hwc.permute(2, 0, 1).unsqueeze(0).contiguous()


def _validate_metric_tensors(candidate: Any, target: Any) -> tuple[int, int]:
    import torch

    for name, tensor in (("candidate", candidate), ("target", target)):
        if not isinstance(tensor, torch.Tensor):
            raise QualityError(f"{name} must be a torch tensor")
        if tensor.device.type != "cpu" or tensor.dtype != torch.float32:
            raise QualityError(f"{name} must be CPU float32")
        if tensor.ndim != 4 or tensor.shape[0] != 1 or tensor.shape[1] != 3:
            raise QualityError(f"{name} must have shape [1,3,H,W]")
        if not tensor.is_contiguous() or not bool(torch.isfinite(tensor).all()):
            raise QualityError(f"{name} must be finite and contiguous")
        if bool((tensor < 0).any()) or bool((tensor > 1).any()):
            raise QualityError(f"{name} must lie in [0,1]")
    if candidate.shape != target.shape:
        raise QualityError("candidate and target shapes differ")
    return int(candidate.shape[2]), int(candidate.shape[3])


def lpips_state_dict_sha256(model: Any) -> str:
    """Hash LPIPS parameters over sorted key/dtype/shape/contiguous bytes."""
    import torch

    state = model.state_dict()
    digest = hashlib.sha256()
    for key in sorted(state):
        value = state[key]
        if not isinstance(value, torch.Tensor):
            raise QualityError(f"LPIPS state entry {key!r} is not a tensor")
        tensor = value.detach().to(device="cpu").contiguous()
        encoded_key = key.encode("utf-8")
        encoded_dtype = str(tensor.dtype).encode("ascii")
        digest.update(struct.pack("<I", len(encoded_key)))
        digest.update(encoded_key)
        digest.update(struct.pack("<I", len(encoded_dtype)))
        digest.update(encoded_dtype)
        digest.update(struct.pack("<I", tensor.ndim))
        digest.update(np.asarray(tensor.shape, dtype="<i8").tobytes())
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def exact_metric_values(candidate: Any, target: Any) -> tuple[MetricTriple, str]:
    """Execute the frozen PSNR/MS-SSIM/LPIPS calls in a fresh single-thread worker.

    Callers must invoke this function in a fresh process.  Setting inter-op threads after Torch
    parallel work has started is intentionally allowed to fail closed.
    """
    configure_metric_worker()

    import lpips
    import pytorch_msssim
    import torch

    height, width = _validate_metric_tensors(candidate, target)
    with torch.inference_mode():
        delta = candidate.to(torch.float64) - target.to(torch.float64)
        mse_tensor = delta.square().sum(dtype=torch.float64) / (3 * height * width)
        mse = float(mse_tensor)
        if not math.isfinite(mse) or mse <= 0.0:
            raise QualityError("metric MSE must be finite and strictly positive")
        psnr = 10.0 * math.log10(1.0 / mse)
        ms_ssim = (
            pytorch_msssim.ms_ssim(
                candidate,
                target,
                data_range=1.0,
                size_average=True,
                win_size=11,
                win_sigma=1.5,
                win=None,
                weights=[0.0448, 0.2856, 0.3001, 0.2363, 0.1333],
                K=(0.01, 0.03),
            )
            .reshape(())
            .item()
        )
        lpips_model = (
            lpips.LPIPS(
                pretrained=True,
                net="alex",
                version="0.1",
                lpips=True,
                spatial=False,
                pnet_rand=False,
                pnet_tune=False,
                use_dropout=True,
                model_path=None,
                eval_mode=True,
                verbose=False,
            )
            .to(device="cpu", dtype=torch.float32)
            .eval()
            .requires_grad_(False)
        )
        state_sha256 = lpips_state_dict_sha256(lpips_model)
        lpips_value = (
            lpips_model(
                2.0 * candidate - 1.0,
                2.0 * target - 1.0,
                retPerLayer=False,
                normalize=False,
            )
            .reshape(())
            .item()
        )
    return MetricTriple(float(psnr), float(ms_ssim), float(lpips_value)), state_sha256


def configure_metric_worker() -> dict[str, Any]:
    """Initialize one fresh metric worker once, then verify its frozen state.

    PyTorch rejects a second ``set_num_interop_threads`` call even when the value is unchanged.
    COMP-011 preflight intentionally evaluates the same tensors twice in one fresh worker, so
    initialization cannot be repeated by the metric body.
    """
    global _METRIC_WORKER_CONFIGURED

    environment = frozen_thread_environment()
    import torch

    if not _METRIC_WORKER_CONFIGURED:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.manual_seed(0)
        _METRIC_WORKER_CONFIGURED = True
    if (
        torch.get_num_threads() != 1
        or torch.get_num_interop_threads() != 1
        or not torch.are_deterministic_algorithms_enabled()
        or torch.initial_seed() != 0
    ):
        raise QualityError("metric-worker deterministic/thread state drifted")
    return {
        "thread_environment": environment,
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "deterministic_algorithms": True,
        "manual_seed": 0,
    }


def metric_repeat_proof(candidate: Any, target: Any) -> dict[str, Any]:
    """Run the exact metric body twice in one worker and compare scalar IEEE bytes."""
    first, first_state = exact_metric_values(candidate, target)
    second, second_state = exact_metric_values(candidate, target)
    first_bytes = struct.pack("<3d", first.psnr, first.ms_ssim, first.lpips)
    second_bytes = struct.pack("<3d", second.psnr, second.ms_ssim, second.lpips)
    return {
        "first": first.as_dict(),
        "second": second.as_dict(),
        "first_ieee754_sha256": sha256_bytes(first_bytes),
        "second_ieee754_sha256": sha256_bytes(second_bytes),
        "lpips_state_sha256": first_state,
        "metric_ieee754_bytes_identical": first_bytes == second_bytes,
        "lpips_state_identical": first_state == second_state,
    }


def repeat_spreads(repetitions: Sequence[MetricTriple]) -> dict[str, float]:
    """Compute the max-minus-min spread of exactly three render repetitions."""
    if len(repetitions) != 3:
        raise QualityError("quality requires exactly three metric repetitions")
    return {
        name: max(float(getattr(item, name)) for item in repetitions)
        - min(float(getattr(item, name)) for item in repetitions)
        for name in METRIC_NAMES
    }


def spreads_pass(spreads: Mapping[str, float]) -> bool:
    """Apply all three atomic repeat-spread caps."""
    if set(spreads) != set(METRIC_NAMES):
        raise QualityError("spread record must contain exactly the three metrics")
    for value in spreads.values():
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise QualityError("spread values must be finite real scalars")
        if value < 0.0:
            raise QualityError("spread values cannot be negative")
    return (
        float(spreads["psnr"]) <= PSNR_SPREAD_CAP
        and float(spreads["ms_ssim"]) <= MS_SSIM_SPREAD_CAP
        and float(spreads["lpips"]) <= LPIPS_SPREAD_CAP
    )


def conservative_envelope(
    primary: Sequence[MetricTriple], candidate: Sequence[MetricTriple]
) -> dict[str, Any]:
    """Build frozen worst-endpoint quality losses from two three-render arms."""
    primary_spread = repeat_spreads(primary)
    candidate_spread = repeat_spreads(candidate)
    if not spreads_pass(primary_spread) or not spreads_pass(candidate_spread):
        raise QualityError("atomic repeat-spread cap exceeded")
    return {
        "primary_spread": primary_spread,
        "candidate_spread": candidate_spread,
        "psnr_loss": max(item.psnr for item in primary) - min(item.psnr for item in candidate),
        "ms_ssim_loss": max(item.ms_ssim for item in primary)
        - min(item.ms_ssim for item in candidate),
        "lpips_increase": max(item.lpips for item in candidate)
        - min(item.lpips for item in primary),
    }


def exclusion_class(value: float, threshold: float, half_width: float) -> str:
    """Classify a lower-is-better loss against the frozen decision exclusion band."""
    for name, scalar in (("value", value), ("threshold", threshold), ("half_width", half_width)):
        if isinstance(scalar, bool) or not isinstance(scalar, (int, float)):
            raise QualityError(f"{name} must be a real scalar")
        if not math.isfinite(float(scalar)):
            raise QualityError(f"{name} must be finite")
    if threshold < 0.0 or half_width < 0.0 or half_width > threshold:
        raise QualityError("invalid exclusion threshold or half-width")
    if value <= threshold - half_width:
        return "definitely_pass"
    if value > threshold + half_width:
        return "definitely_fail"
    return "ambiguous"


def classify_envelope(envelope: Mapping[str, Any]) -> dict[str, Any]:
    """Classify all metrics and the candidate as pass/fail/ambiguous."""
    required = {"psnr_loss", "ms_ssim_loss", "lpips_increase"}
    if not isinstance(envelope, Mapping) or not required.issubset(envelope):
        raise QualityError("quality envelope is missing a conservative metric")
    classes = {
        "psnr": exclusion_class(envelope["psnr_loss"], PSNR_THRESHOLD, PSNR_EXCLUSION_HALF_WIDTH),
        "ms_ssim": exclusion_class(
            envelope["ms_ssim_loss"],
            MS_SSIM_THRESHOLD,
            MS_SSIM_EXCLUSION_HALF_WIDTH,
        ),
        "lpips": exclusion_class(
            envelope["lpips_increase"], LPIPS_THRESHOLD, LPIPS_EXCLUSION_HALF_WIDTH
        ),
    }
    if all(value == "definitely_pass" for value in classes.values()):
        state = "definitely_qualified"
    elif any(value == "definitely_fail" for value in classes.values()):
        state = "definitely_failed"
    else:
        state = "ambiguous"
    return {"metrics": classes, "state": state}


def _variant_sort_key(record: Mapping[str, Any]) -> tuple[Any, ...]:
    label = record.get("label")
    if label not in VARIANT_LABELS:
        raise QualityError(f"unknown SSP2V variant label {label!r}")
    family = record.get("family")
    family_rank = {"flat": 0, "rvq": 1}.get(family)
    if family_rank is None:
        raise QualityError("variant family must be flat or rvq")
    matched = record.get("matched_stage_count")
    base_k = record.get("base_k")
    complete_bytes = record.get("complete_bytes")
    blob = record.get("complete_blob")
    if (
        isinstance(matched, bool)
        or not isinstance(matched, int)
        or matched not in (2, 3)
        or isinstance(base_k, bool)
        or not isinstance(base_k, int)
        or base_k not in (64, 128)
        or isinstance(complete_bytes, bool)
        or not isinstance(complete_bytes, int)
        or complete_bytes <= 0
        or not isinstance(blob, bytes)
        or len(blob) != complete_bytes
    ):
        raise QualityError("variant sort metadata is malformed")
    expected_label = (
        f"flat_s{matched}_k{base_k}_l{(2 * matched - 1) * base_k}"
        if family == "flat"
        else f"rvq_s{matched}_k{base_k}"
    )
    if label != expected_label:
        raise QualityError("variant label disagrees with family/stage/K descriptor")
    return (
        complete_bytes,
        family_rank,
        matched,
        base_k,
        label.encode("ascii"),
        blob,
    )


def select_variant(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Apply the frozen byte order and ambiguity-sensitive quality selection."""
    if len(records) != 8:
        raise QualityError("selection requires exactly eight variants")
    labels = [record.get("label") for record in records]
    if set(labels) != set(VARIANT_LABELS) or len(set(labels)) != 8:
        raise QualityError("selection requires each frozen variant exactly once")
    ordered = sorted(records, key=_variant_sort_key)
    for position, record in enumerate(ordered):
        state = record.get("qualification_state")
        if state == "definitely_failed":
            continue
        if state == "ambiguous":
            return {
                "status": "invalid_no_decision",
                "reason": "ambiguous_variant_can_affect_selection",
                "position": position,
                "label": record["label"],
                "ordered_labels": [item["label"] for item in ordered],
            }
        if state == "definitely_qualified":
            return {
                "status": "selected",
                "position": position,
                "label": record["label"],
                "complete_bytes": int(record["complete_bytes"]),
                "ordered_labels": [item["label"] for item in ordered],
            }
        raise QualityError(f"invalid qualification state {state!r}")
    return {
        "status": "valid_method_failure",
        "reason": "all_variants_definitely_failed",
        "ordered_labels": [item["label"] for item in ordered],
    }


def conservative_median_psnr_class(losses: Mapping[int, float]) -> dict[str, Any]:
    """Compute the frozen even-sample median and its exclusion-band class."""
    if not isinstance(losses, Mapping) or set(losses) != set(range(8)):
        raise QualityError("tuple median requires frozen image indices 0..7 exactly once")
    values: list[tuple[float, int]] = []
    for index in range(8):
        value = losses[index]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise QualityError("PSNR losses must be finite real scalars")
        values.append((float(value), index))
    ordered = sorted(values)
    median = (ordered[3][0] + ordered[4][0]) / 2.0
    if median <= MEDIAN_PSNR_PASS_MAX:
        state = "definitely_pass"
    elif median > MEDIAN_PSNR_FAIL_MIN_EXCLUSIVE:
        state = "definitely_fail"
    else:
        state = "ambiguous"
    return {
        "median_psnr_loss": median,
        "state": state,
        "ordered_image_indices": [index for _, index in ordered],
    }


def replay_endpoint_match(original: Mapping[str, float], replay: Mapping[str, float]) -> bool:
    """Check one persisted metric endpoint against replay tolerances."""
    if set(original) != set(METRIC_NAMES) or set(replay) != set(METRIC_NAMES):
        raise QualityError("replay endpoints must contain exactly the three metrics")
    tolerances = {"psnr": 0.001, "ms_ssim": 0.00001, "lpips": 0.00001}
    for name in METRIC_NAMES:
        left = float(original[name])
        right = float(replay[name])
        if not math.isfinite(left) or not math.isfinite(right):
            raise QualityError("replay endpoint is non-finite")
        if abs(right - left) > tolerances[name]:
            return False
    return True


def _artifact_relative_path(artifact_root: Path, relative_path: str) -> tuple[Path, Path]:
    if not isinstance(relative_path, str) or relative_path != RENDERER_RELATIVE_PATH:
        raise QualityError("renderer must use the frozen artifact-relative path")
    if "\\" in relative_path:
        raise QualityError("renderer relative path must use POSIX separators")
    relative = Path(relative_path)
    if relative.is_absolute() or any(part in ("", ".", "..") for part in relative.parts):
        raise QualityError("renderer path must be normalized and artifact relative")
    root_path = Path(artifact_root)
    if root_path.is_symlink():
        raise QualityError("artifact root may not be a symlink")
    root = root_path.resolve(strict=True)
    if not root.is_dir():
        raise QualityError("artifact root is not a directory")
    candidate = root.joinpath(*relative.parts)
    current = root
    for part in relative.parts[:-1]:
        current = current / part
        info = os.lstat(current)
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            raise QualityError("renderer artifact parent is a symlink or non-directory")
    info = os.lstat(candidate)
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise QualityError("renderer artifact is a symlink or non-regular file")
    resolved = candidate.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as exc:  # pragma: no cover - lstat checks already reject escapes
        raise QualityError("renderer resolves outside the artifact") from exc
    return root, resolved


def _hash_regular_no_follow(path: Path) -> tuple[int, str]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        named = os.stat(path, follow_symlinks=False)
        if not stat.S_ISREG(opened.st_mode) or (opened.st_dev, opened.st_ino) != (
            named.st_dev,
            named.st_ino,
        ):
            raise QualityError("renderer file identity changed during verification")
        digest = hashlib.sha256()
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        return int(opened.st_size), digest.hexdigest()
    finally:
        os.close(descriptor)


def renderer_elf_identity(path: Path) -> dict[str, Any]:
    """Bind ELF class/machine/build-id plus dynamic and loader command outputs."""
    import torch

    torch_lib = str(Path(torch.__file__).resolve().parent / "lib")
    inherited_library_path = os.environ.get("LD_LIBRARY_PATH", "")
    effective_library_path = (
        torch_lib
        if not inherited_library_path
        else f"{torch_lib}{os.pathsep}{inherited_library_path}"
    )
    command_environment = os.environ.copy()
    command_environment["LD_LIBRARY_PATH"] = effective_library_path
    commands = {
        "header": ("readelf", "--file-header", str(path)),
        "notes": ("readelf", "--notes", str(path)),
        "dynamic": ("readelf", "--dynamic", str(path)),
        "ldd": ("ldd", str(path)),
    }
    outputs: dict[str, str] = {}
    for name, command in commands.items():
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            env=command_environment,
        )
        if completed.stderr:
            raise QualityError(f"{name} emitted unexpected stderr")
        outputs[name] = completed.stdout
    normalized_ldd = re.sub(r"\s+\(0x[0-9a-fA-F]+\)", "", outputs["ldd"])
    if "not found" in normalized_ldd:
        raise QualityError("renderer loader dependency is unresolved")
    class_match = re.search(r"^\s*Class:\s*(\S+)", outputs["header"], re.MULTILINE)
    machine_match = re.search(r"^\s*Machine:\s*(.+?)\s*$", outputs["header"], re.MULTILINE)
    build_match = re.search(r"Build ID:\s*([0-9a-fA-F]+)", outputs["notes"])
    if class_match is None or machine_match is None or build_match is None:
        raise QualityError("renderer ELF identity fields are incomplete")
    return {
        "elf_class": class_match.group(1),
        "elf_machine": machine_match.group(1),
        "build_id": build_match.group(1).lower(),
        "header_sha256": sha256_bytes(outputs["header"].encode("utf-8")),
        "notes_sha256": sha256_bytes(outputs["notes"].encode("utf-8")),
        "dynamic_sha256": sha256_bytes(outputs["dynamic"].encode("utf-8")),
        "ldd_sha256": sha256_bytes(normalized_ldd.encode("utf-8")),
        "ldd_output": normalized_ldd,
        "effective_ld_library_path": effective_library_path,
    }


def verify_persisted_binary(
    artifact_root: Path,
    relative_path: str,
    expected_sha256: str,
    expected_bytes: int,
    expected_elf_identity: Mapping[str, Any],
) -> Path:
    """Verify the exact artifact-relative renderer copy before any import attempt."""
    _require_digest(expected_sha256, "expected renderer SHA-256")
    if (
        isinstance(expected_bytes, bool)
        or not isinstance(expected_bytes, int)
        or expected_bytes <= 0
    ):
        raise QualityError("expected renderer byte length must be positive")
    _, resolved = _artifact_relative_path(artifact_root, relative_path)
    observed_bytes, observed_sha256 = _hash_regular_no_follow(resolved)
    if observed_bytes != expected_bytes:
        raise QualityError("persisted renderer size/type mismatch")
    if observed_sha256 != expected_sha256:
        raise QualityError("persisted renderer SHA-256 mismatch")
    if not isinstance(expected_elf_identity, Mapping):
        raise QualityError("expected renderer ELF identity must be a mapping")
    if renderer_elf_identity(resolved) != dict(expected_elf_identity):
        raise QualityError("persisted renderer ELF/loader identity mismatch")
    return resolved


def _renderer_process_maps_exact(verified: Path) -> bool:
    if not sys.platform.startswith("linux"):
        return True
    maps = Path("/proc/self/maps").read_text(encoding="utf-8")
    renderer_lines = [line for line in maps.splitlines() if "structsplat_render_ext" in line]
    return bool(renderer_lines) and all(str(verified) in line for line in renderer_lines)


def load_persisted_renderer(
    artifact_root: Path,
    relative_path: str,
    expected_sha256: str,
    expected_bytes: int,
    expected_elf_identity: Mapping[str, Any],
) -> ModuleType:
    """Load only an already-copied CUDA extension and fail if JIT/cache state is resident."""
    global _PERSISTED_RENDERER_CAPABILITY

    frozen_runtime_environment()
    verified = verify_persisted_binary(
        artifact_root,
        relative_path,
        expected_sha256,
        expected_bytes,
        expected_elf_identity,
    )
    if "structsplat_render_ext" in sys.modules:
        raise QualityError("renderer extension is already resident before persisted load")

    import torch.utils.cpp_extension as cpp_extension
    from structsplat import cuda_render

    if getattr(cuda_render, "_EXT", None) is not None:
        raise QualityError("StructSplat renderer state is already initialized")
    original_load = cpp_extension.load

    def forbidden_jit(*_args: Any, **_kwargs: Any) -> Any:
        raise QualityError("JIT renderer build/load is forbidden after preflight")

    cpp_extension.load = forbidden_jit
    try:
        spec = importlib.util.spec_from_file_location("structsplat_render_ext", verified)
        if spec is None or spec.loader is None:
            raise QualityError("cannot create persisted renderer import specification")
        module = importlib.util.module_from_spec(spec)
        sys.modules["structsplat_render_ext"] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop("structsplat_render_ext", None)
            raise
    finally:
        cpp_extension.load = original_load
    loaded_path_raw = getattr(module, "__file__", None)
    if not isinstance(loaded_path_raw, str) or Path(loaded_path_raw).resolve() != verified:
        sys.modules.pop("structsplat_render_ext", None)
        raise QualityError("persisted renderer module origin mismatch")
    observed_bytes, observed_sha256 = _hash_regular_no_follow(verified)
    if observed_bytes != expected_bytes or observed_sha256 != expected_sha256:
        sys.modules.pop("structsplat_render_ext", None)
        raise QualityError("persisted renderer changed during import")
    if not _renderer_process_maps_exact(verified):
        sys.modules.pop("structsplat_render_ext", None)
        raise QualityError("renderer process mapping is absent or includes a non-artifact path")
    cuda_render._EXT = module  # noqa: SLF001 -- the frozen loader intentionally injects it.
    _PERSISTED_RENDERER_CAPABILITY = {
        "module": module,
        "path": str(verified),
        "relative_path": relative_path,
        "sha256": expected_sha256,
        "bytes": expected_bytes,
        "elf_identity": dict(expected_elf_identity),
    }
    return module


def render_boundary_once(
    header: Any, arrays: Mapping[str, np.ndarray], device: Any = "cuda:0"
) -> Any:
    """Render one exact decoded boundary through the already-loaded production CUDA adapter."""
    import torch

    from structsplat import cuda_render
    from structsplat.gaussians import GaussianField
    from structsplat.render import render_field

    capability = _PERSISTED_RENDERER_CAPABILITY
    if capability is None or capability.get("module") is not getattr(cuda_render, "_EXT", None):
        raise QualityError("persisted renderer capability was not established in this process")
    module_path = getattr(capability["module"], "__file__", None)
    if not isinstance(module_path, str) or Path(module_path).resolve() != Path(capability["path"]):
        raise QualityError("persisted renderer capability origin drifted")
    required = {"means", "log_scales", "rotations", "colors"}
    if set(arrays) != required:
        raise QualityError(
            "decoded boundary must contain exactly means/log_scales/rotations/colors"
        )
    expected_shapes = {
        "means": (8192, 2),
        "log_scales": (8192, 2),
        "rotations": (8192,),
        "colors": (8192, 3),
    }
    for name, expected_shape in expected_shapes.items():
        value = arrays[name]
        if (
            not isinstance(value, np.ndarray)
            or value.dtype != np.float32
            or value.shape != expected_shape
            or not value.flags.c_contiguous
            or not np.isfinite(value).all()
        ):
            raise QualityError(f"decoded boundary {name} is not canonical float32{expected_shape}")
    if int(header.n) != 8192 or int(header.renderer_id) != 1 or int(header.support_fade) != 0:
        raise QualityError("decoded header violates frozen renderer/N semantics")
    torch_device = torch.device(device)
    if torch_device.type != "cuda":
        raise QualityError("production quality render requires a CUDA device")
    field = GaussianField(
        torch.as_tensor(arrays["means"], device=torch_device, dtype=torch.float32).clone(),
        torch.as_tensor(arrays["log_scales"], device=torch_device, dtype=torch.float32).clone(),
        torch.as_tensor(arrays["rotations"], device=torch_device, dtype=torch.float32).clone(),
        torch.as_tensor(arrays["colors"], device=torch_device, dtype=torch.float32).clone(),
        opacities=None,
    )
    torch.cuda.synchronize(torch_device)
    with torch.inference_mode():
        image = render_field(
            field.means,
            field.conics(float(header.aa_dilation)),
            field.colors,
            field.radii(float(header.sigma_cutoff), float(header.aa_dilation)),
            int(header.height),
            int(header.width),
            int(header.render_chunk),
            "cuda",
            scales=field.effective_scales(float(header.aa_dilation)),
            rotations=field.rotations,
            support_fade=bool(header.support_fade),
            sigma_cutoff=float(header.sigma_cutoff),
        )
    torch.cuda.synchronize(torch_device)
    if image.shape != (int(header.height), int(header.width), 3):
        raise QualityError("renderer returned a noncanonical HWC shape")
    return strict_render_tensor(image)


def repeat_fixture_arrays() -> dict[str, np.ndarray]:
    """Construct the target-free frozen N=8192 renderer repeat fixture on CPU."""
    count = 8192
    means = np.empty((count, 2), dtype=np.float32)
    physical_scales = np.empty((count, 2), dtype=np.float32)
    rotations = np.empty(count, dtype=np.float32)
    colors = np.empty((count, 3), dtype=np.float32)
    prefix = b"structsplat-comp011-render-repeat-v1\0field\0"
    for index in range(count):
        digest = hashlib.sha256(prefix + struct.pack("<I", index)).digest()
        means[index] = (np.float32(48 + digest[0] % 96), np.float32(48 + digest[1] % 96))
        physical_scales[index] = (
            np.float32(1.0 + (digest[2] % 16) / 4.0),
            np.float32(1.0 + (digest[3] % 16) / 4.0),
        )
        rotations[index] = np.float32((digest[4] % 32) * math.pi / 32)
        colors[index] = (
            np.float32(digest[5] / 255.0),
            np.float32(digest[6] / 255.0),
            np.float32(digest[7] / 255.0),
        )
    return {
        "means": means,
        "physical_scales": physical_scales,
        "rotations": rotations,
        "colors": colors,
    }


def repeat_fixture_target() -> np.ndarray:
    """Construct the frozen synthetic 192x192 strict-RGB target samples."""
    target = np.empty((192, 192, 3), dtype=np.uint8)
    prefix = b"structsplat-comp011-render-repeat-v1\0target\0"
    for y in range(192):
        for x in range(192):
            digest = hashlib.sha256(prefix + struct.pack("<HH", y, x)).digest()
            target[y, x] = (digest[0], digest[1], digest[2])
    return target


def build_repeat_fixture(device: Any = "cuda:0") -> tuple[Any, Any, dict[str, Any]]:
    """Build the frozen repeat field through ``GaussianField.from_numpy`` exactly once."""
    import torch

    from structsplat.gaussians import GaussianField

    arrays = repeat_fixture_arrays()
    field = GaussianField.from_numpy(
        means=arrays["means"],
        scales=arrays["physical_scales"],
        angles=arrays["rotations"],
        colors=arrays["colors"],
        opacities=None,
        scale_max=None,
        device=device,
        dtype=torch.float32,
        color_grads=None,
        background_mask=None,
        filter_variance=None,
    )
    target = strict_target_tensor(repeat_fixture_target())
    records = {
        name: {
            "dtype": value.dtype.str,
            "shape": list(value.shape),
            "sha256": sha256_bytes(value.tobytes(order="C")),
        }
        for name, value in arrays.items()
    }
    for name, value in (
        ("field_log_scales", field.log_scales),
        ("field_scales", field.scales()),
        ("field_effective_scales", field.effective_scales(0.0)),
        ("field_conics", field.conics(0.0)),
        ("field_radii", field.radii(3.0, 0.0)),
    ):
        records[name] = tensor_record(value.detach().to(device="cpu").contiguous())
    records["target_u8"] = {
        "dtype": "uint8",
        "shape": [192, 192, 3],
        "sha256": sha256_bytes(repeat_fixture_target().tobytes(order="C")),
    }
    records["target_tensor"] = tensor_record(target)
    return field, target, records


def frozen_thread_environment() -> dict[str, str]:
    """Return and validate the four single-thread worker variables."""
    values = {name: os.environ.get(name, "") for name in FROZEN_THREAD_ENVIRONMENT}
    if values != FROZEN_THREAD_ENVIRONMENT:
        raise QualityError("all frozen metric-worker thread variables must equal 1")
    return values


def frozen_runtime_environment() -> dict[str, Any]:
    """Validate the renderer preload and all frozen single-thread variables."""
    threads = frozen_thread_environment()
    preload = os.environ.get("LD_PRELOAD")
    if preload != REQUIRED_RENDERER_PRELOAD:
        raise QualityError("renderer worker LD_PRELOAD differs from the frozen system libstdc++")
    return {"thread_environment": threads, "ld_preload": preload}
