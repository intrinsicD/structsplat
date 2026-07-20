from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch

from benchmarks import ssp2v_quality as quality


def _triple(psnr: float = 40.0, ms_ssim: float = 0.99, lpips: float = 0.01):
    return quality.MetricTriple(psnr, ms_ssim, lpips)


def _variant(
    label: str,
    complete_bytes: int,
    state: str = "definitely_failed",
) -> dict[str, object]:
    family, raw_stages, raw_k, *tail = label.split("_")
    stages = int(raw_stages[1:])
    base_k = int(raw_k[1:])
    return {
        "label": label,
        "family": family,
        "matched_stage_count": stages,
        "base_k": base_k,
        "complete_bytes": complete_bytes,
        "complete_blob": bytes([quality.VARIANT_LABELS.index(label)]) * complete_bytes,
        "qualification_state": state,
    }


def test_strict_target_and_render_conversion_are_exact() -> None:
    source = np.array([[[0, 127, 255], [1, 2, 3]]], dtype=np.uint8)
    target = quality.strict_target_tensor(source)
    assert target.shape == (1, 3, 1, 2)
    assert target.dtype == torch.float32
    assert target[0, 0, 0, 1].item() == torch.tensor(1, dtype=torch.float32).div(255).item()

    render = torch.tensor([[[-1.0, 0.5, 2.0], [0.25, 0.75, 1.0]]], dtype=torch.float32)
    candidate = quality.strict_render_tensor(render)
    assert candidate.shape == (1, 3, 1, 2)
    assert candidate.flatten().tolist() == [0.0, 0.25, 0.5, 0.75, 1.0, 1.0]
    with pytest.raises(quality.QualityError, match="uint8"):
        quality.strict_target_tensor(source.astype(np.int16))
    with pytest.raises(quality.QualityError, match="finite"):
        quality.strict_render_tensor(torch.full((1, 1, 3), float("nan")))


def test_metric_triple_and_tensor_records_reject_noncanonical_values() -> None:
    with pytest.raises(quality.QualityError, match="finite"):
        quality.MetricTriple(float("nan"), 0.9, 0.1)
    tensor = torch.arange(6, dtype=torch.float32).reshape(1, 3, 1, 2)
    record = quality.tensor_record(tensor)
    assert record["bytes"] == 24
    assert record["sha256"] == hashlib.sha256(tensor.numpy().tobytes()).hexdigest()
    noncontiguous = torch.arange(12, dtype=torch.float32).reshape(1, 3, 2, 2).transpose(2, 3)
    assert not noncontiguous.is_contiguous()
    with pytest.raises(quality.QualityError, match="contiguous"):
        quality.tensor_record(noncontiguous)


def test_exact_metrics_repeat_in_one_fresh_worker_with_identical_ieee_bytes() -> None:
    code = """
import json
import numpy as np
from benchmarks.ssp2v_quality import metric_repeat_proof, repeat_fixture_target, strict_target_tensor
target_u8 = repeat_fixture_target()
candidate_u8 = target_u8.copy()
candidate_u8[0, 0, 0] ^= np.uint8(1)
proof = metric_repeat_proof(strict_target_tensor(candidate_u8), strict_target_tensor(target_u8))
print(json.dumps(proof, sort_keys=True))
"""
    environment = os.environ.copy()
    environment.update(quality.FROZEN_THREAD_ENVIRONMENT)
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    proof = json.loads(completed.stdout.splitlines()[-1])
    assert proof["metric_ieee754_bytes_identical"] is True
    assert proof["lpips_state_identical"] is True


def test_repeat_spreads_and_conservative_envelope_use_worst_endpoints() -> None:
    primary = [
        _triple(40.0, 0.99000, 0.01000),
        _triple(40.0005, 0.990004, 0.010004),
        _triple(40.0002, 0.990002, 0.010002),
    ]
    candidate = [
        _triple(39.96, 0.9896, 0.0107),
        _triple(39.9605, 0.989604, 0.010704),
        _triple(39.9602, 0.989602, 0.010702),
    ]
    envelope = quality.conservative_envelope(primary, candidate)
    assert envelope["psnr_loss"] == pytest.approx(40.0005 - 39.96)
    assert envelope["ms_ssim_loss"] == pytest.approx(0.990004 - 0.9896)
    assert envelope["lpips_increase"] == pytest.approx(0.010704 - 0.01)
    assert quality.classify_envelope(envelope)["state"] == "definitely_qualified"

    over = list(candidate)
    over[2] = _triple(39.962, 0.989602, 0.010702)
    with pytest.raises(quality.QualityError, match="spread"):
        quality.conservative_envelope(primary, over)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0.048, "definitely_pass"),
        (math.nextafter(0.048, math.inf), "ambiguous"),
        (0.052, "ambiguous"),
        (math.nextafter(0.05 + 0.002, math.inf), "definitely_fail"),
    ],
)
def test_exclusion_band_boundaries(value: float, expected: str) -> None:
    assert quality.exclusion_class(value, 0.05, 0.002) == expected


def test_selection_uses_complete_bytes_and_fails_closed_on_early_ambiguity() -> None:
    records = [_variant(label, 200 + index) for index, label in enumerate(quality.VARIANT_LABELS)]
    records[4]["complete_bytes"] = 100
    records[4]["complete_blob"] = b"x" * 100
    records[4]["qualification_state"] = "definitely_qualified"
    selected = quality.select_variant(records)
    assert selected["status"] == "selected"
    assert selected["label"] == "rvq_s2_k64"

    records[3]["complete_bytes"] = 99
    records[3]["complete_blob"] = b"y" * 99
    records[3]["qualification_state"] = "ambiguous"
    result = quality.select_variant(records)
    assert result["status"] == "invalid_no_decision"
    assert result["label"] == records[3]["label"]


def test_selection_tie_order_and_all_failed_branch() -> None:
    records = [_variant(label, 100) for label in quality.VARIANT_LABELS]
    failed = quality.select_variant(records)
    assert failed["status"] == "valid_method_failure"
    assert failed["ordered_labels"] == list(quality.VARIANT_LABELS)
    records[-1]["qualification_state"] = "definitely_qualified"
    assert quality.select_variant(records)["label"] == quality.VARIANT_LABELS[-1]
    records[0]["label"] = records[1]["label"]
    with pytest.raises(quality.QualityError, match="each frozen"):
        quality.select_variant(records)


@pytest.mark.parametrize(
    ("center", "expected"),
    [
        (0.018, "definitely_pass"),
        (0.020, "ambiguous"),
        (0.022, "ambiguous"),
        (0.022001, "definitely_fail"),
    ],
)
def test_conservative_median_psnr_band(center: float, expected: str) -> None:
    result = quality.conservative_median_psnr_class({index: center for index in range(8)})
    assert result["state"] == expected
    assert result["median_psnr_loss"] == center


def test_conservative_median_requires_frozen_image_indices() -> None:
    with pytest.raises(quality.QualityError, match="indices"):
        quality.conservative_median_psnr_class({index + 1: 0.0 for index in range(8)})


def test_replay_endpoint_tolerances_are_inclusive() -> None:
    original = {"psnr": 40.0, "ms_ssim": 0.99, "lpips": 0.01}
    replay = {"psnr": 40.001, "ms_ssim": 0.99001, "lpips": 0.01001}
    assert quality.replay_endpoint_match(original, replay)
    replay["psnr"] = math.nextafter(40.001, math.inf)
    assert not quality.replay_endpoint_match(original, replay)


def test_render_schedule_is_exact_and_covers_every_arm_three_times() -> None:
    assert quality.RENDER_SCHEDULE[0][0] == "primary"
    assert quality.RENDER_SCHEDULE[1][-1] == "primary"
    assert quality.RENDER_SCHEDULE[1][:-1] == tuple(reversed(quality.VARIANT_LABELS))
    assert quality.RENDER_SCHEDULE[2] == quality.RENDER_SCHEDULE[0]
    assert all(
        sum(arm in repetition for repetition in quality.RENDER_SCHEDULE) == 3
        for arm in ("primary", *quality.VARIANT_LABELS)
    )


def test_render_boundary_uses_decoded_log_scales_in_production_field_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from structsplat import cuda_render, render as render_module

    count = 8192
    arrays = {
        "means": np.tile(np.array([[2.0, 3.0]], dtype=np.float32), (count, 1)),
        "log_scales": np.tile(np.log(np.array([[2.0, 4.0]], dtype=np.float32)), (count, 1)),
        "rotations": np.full(count, 0.25, dtype=np.float32),
        "colors": np.tile(np.array([[0.1, 0.2, 0.3]], dtype=np.float32), (count, 1)),
    }
    header = SimpleNamespace(
        n=count,
        renderer_id=1,
        aa_dilation=0.5,
        sigma_cutoff=3.0,
        height=4,
        width=5,
        render_chunk=512,
        support_fade=0,
    )
    observed: dict[str, object] = {}

    def fake_render(means, conics, colors, radii, height, width, chunk, mode, **kwargs):
        observed.update(
            means=means.detach().clone(),
            conics=conics.detach().clone(),
            colors=colors.detach().clone(),
            radii=radii.detach().clone(),
            height=height,
            width=width,
            chunk=chunk,
            mode=mode,
            kwargs=kwargs,
        )
        return torch.full((height, width, 3), 0.25, dtype=torch.float32, device=means.device)

    fake_module = SimpleNamespace(__file__=str(Path(__file__).resolve()))
    monkeypatch.setattr(cuda_render, "_EXT", fake_module)
    monkeypatch.setattr(
        quality,
        "_PERSISTED_RENDERER_CAPABILITY",
        {
            "module": fake_module,
            "path": str(Path(__file__).resolve()),
            "relative_path": quality.RENDERER_RELATIVE_PATH,
        },
    )
    monkeypatch.setattr(render_module, "render_field", fake_render)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda _device: None)
    image = quality.render_boundary_once(header, arrays, device="cuda:0")
    assert image.shape == (1, 3, 4, 5)
    assert observed["mode"] == "cuda"
    assert observed["chunk"] == 512
    kwargs = observed["kwargs"]
    assert isinstance(kwargs, dict)
    expected_scales = torch.sqrt(torch.tensor([[4.0, 16.0]]) + 0.5)
    assert torch.allclose(kwargs["scales"][0].cpu(), expected_scales[0])
    assert kwargs["support_fade"] is False
    assert kwargs["sigma_cutoff"] == 3.0
    # A double-log or physical-scale constructor substitution changes these exact products.
    angle = torch.tensor(0.25)
    variances = torch.tensor([4.0, 16.0]) + 0.5
    expected_radii = (
        torch.ceil(
            3.0
            * torch.sqrt(
                torch.stack(
                    (
                        torch.cos(angle).square() * variances[0]
                        + torch.sin(angle).square() * variances[1],
                        torch.sin(angle).square() * variances[0]
                        + torch.cos(angle).square() * variances[1],
                    )
                )
            )
        )
        .to(torch.long)
        .reshape(1, 2)
    )
    assert torch.equal(observed["radii"][0].cpu(), expected_radii[0])


def test_render_boundary_rejects_missing_persisted_capability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from structsplat import cuda_render

    monkeypatch.setattr(cuda_render, "_EXT", None)
    monkeypatch.setattr(quality, "_PERSISTED_RENDERER_CAPABILITY", None)
    with pytest.raises(quality.QualityError, match="capability"):
        quality.render_boundary_once(SimpleNamespace(), {})


def test_repeat_fixture_is_deterministic_and_protocol_shaped() -> None:
    first = quality.repeat_fixture_arrays()
    second = quality.repeat_fixture_arrays()
    assert set(first) == {"means", "physical_scales", "rotations", "colors"}
    for name in first:
        assert first[name].dtype == np.float32
        assert np.array_equal(first[name], second[name])
    assert first["means"].shape == (8192, 2)
    assert np.all((first["physical_scales"] >= 1.0) & (first["physical_scales"] <= 4.75))
    target = quality.repeat_fixture_target()
    assert target.shape == (192, 192, 3)
    assert target.dtype == np.uint8
    assert np.array_equal(target, quality.repeat_fixture_target())
    assert hashlib.sha256(first["means"].tobytes()).hexdigest() == (
        "e0d6ba9921377d5360e8079019f13ccfcc87e5c8abc3c495e37e02ad54359418"
    )
    assert hashlib.sha256(first["physical_scales"].tobytes()).hexdigest() == (
        "4be16bd7492d7878e73ca3a61dac09be345d884b228de3bd3d71af38b25afaa7"
    )
    assert hashlib.sha256(target.tobytes()).hexdigest() == (
        "652a1f8ba66a2eda8034cb9be89f506450edcfb6677875ce490643a777be79d2"
    )


def test_repeat_fixture_from_numpy_has_exact_single_log_boundary() -> None:
    field, target, records = quality.build_repeat_fixture("cpu")
    arrays = quality.repeat_fixture_arrays()
    expected_log = torch.log(torch.from_numpy(arrays["physical_scales"]).clone())
    assert torch.equal(field.log_scales, expected_log)
    assert torch.equal(field.scales(), torch.exp(expected_log))
    assert records["field_log_scales"]["sha256"] == (
        "fc0c8d0ddc3c33832bdbc2a77a2ae451186394f5f3f12a0307ad53ffe1dfc46b"
    )
    assert records["target_tensor"]["sha256"] == (
        "146cc2ce66777db06e3c7b851d1771f19f8072c9e0f2d2782f2c413abb4b121b"
    )
    assert target.shape == (1, 3, 192, 192)


def _fixture_renderer(tmp_path: Path) -> tuple[Path, Path, str, dict[str, str]]:
    path = tmp_path / quality.RENDERER_RELATIVE_PATH
    path.parent.mkdir(parents=True)
    path.write_bytes(b"ELF fixture")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return tmp_path, path, digest, {"fixture": "elf"}


def test_verify_persisted_binary_checks_artifact_size_hash_and_elf(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, path, digest, elf = _fixture_renderer(tmp_path)
    monkeypatch.setattr(quality, "renderer_elf_identity", lambda _path: elf)
    assert (
        quality.verify_persisted_binary(
            root, quality.RENDERER_RELATIVE_PATH, digest, len(path.read_bytes()), elf
        )
        == path.resolve()
    )
    with pytest.raises(quality.QualityError, match="size"):
        quality.verify_persisted_binary(root, quality.RENDERER_RELATIVE_PATH, digest, 1, elf)
    with pytest.raises(quality.QualityError, match="SHA"):
        quality.verify_persisted_binary(
            root, quality.RENDERER_RELATIVE_PATH, "0" * 64, len(path.read_bytes()), elf
        )
    with pytest.raises(quality.QualityError, match="ELF"):
        quality.verify_persisted_binary(
            root, quality.RENDERER_RELATIVE_PATH, digest, len(path.read_bytes()), {"wrong": 1}
        )


def test_persisted_binary_rejects_absolute_cache_and_symlink_substitution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, path, digest, elf = _fixture_renderer(tmp_path)
    monkeypatch.setattr(quality, "renderer_elf_identity", lambda _path: elf)
    with pytest.raises(quality.QualityError, match="artifact-relative"):
        quality.verify_persisted_binary(root, str(path.resolve()), digest, path.stat().st_size, elf)
    original = path.with_name("original-cache.so")
    original.write_bytes(path.read_bytes())
    path.unlink()
    path.symlink_to(original)
    with pytest.raises(quality.QualityError, match="symlink"):
        quality.verify_persisted_binary(
            root, quality.RENDERER_RELATIVE_PATH, digest, original.stat().st_size, elf
        )


def test_persisted_loader_rejects_resident_cache_module(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, path, digest, elf = _fixture_renderer(tmp_path)
    monkeypatch.setattr(quality, "renderer_elf_identity", lambda _path: elf)
    for name, value in quality.FROZEN_THREAD_ENVIRONMENT.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("LD_PRELOAD", quality.REQUIRED_RENDERER_PRELOAD)
    monkeypatch.setitem(sys.modules, "structsplat_render_ext", object())
    with pytest.raises(quality.QualityError, match="already resident"):
        quality.load_persisted_renderer(
            root, quality.RENDERER_RELATIVE_PATH, digest, path.stat().st_size, elf
        )


def test_persisted_loader_injects_exact_module_and_blocks_jit_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from structsplat import cuda_render

    root, path, digest, elf = _fixture_renderer(tmp_path)
    module = ModuleType("structsplat_render_ext")
    module.__file__ = str(path.resolve())

    class Loader:
        @staticmethod
        def exec_module(_module):
            return None

    spec = SimpleNamespace(loader=Loader())
    monkeypatch.setattr(quality, "renderer_elf_identity", lambda _path: elf)
    monkeypatch.setattr(importlib.util, "spec_from_file_location", lambda *_args: spec)
    monkeypatch.setattr(importlib.util, "module_from_spec", lambda _spec: module)
    monkeypatch.setattr(quality, "_renderer_process_maps_exact", lambda verified: verified == path)
    monkeypatch.setattr(cuda_render, "_EXT", None)
    monkeypatch.setattr(quality, "_PERSISTED_RENDERER_CAPABILITY", None)
    monkeypatch.delitem(sys.modules, "structsplat_render_ext", raising=False)
    for name, value in quality.FROZEN_THREAD_ENVIRONMENT.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("LD_PRELOAD", quality.REQUIRED_RENDERER_PRELOAD)
    loaded = quality.load_persisted_renderer(
        root, quality.RENDERER_RELATIVE_PATH, digest, path.stat().st_size, elf
    )
    assert loaded is module
    assert cuda_render._EXT is module
    assert quality._PERSISTED_RENDERER_CAPABILITY["relative_path"] == (
        quality.RENDERER_RELATIVE_PATH
    )
    sys.modules.pop("structsplat_render_ext", None)
