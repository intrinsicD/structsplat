import json
from pathlib import Path

import numpy as np
import pytest

from benchmarks import actual_rate_phase_diagram as B


def _row(num_bytes, psnr, *, n=100, height=10, width=10):
    return {
        "status": "ok",
        "bytes": num_bytes,
        "bpp": B.actual_bpp(num_bytes, height, width),
        "psnr": psnr,
        "n_gaussians": n,
        "height": height,
        "width": width,
    }


def test_actual_bpp_and_byte_cap_keep_original_integer_denominator():
    assert B.exact_byte_cap(0.5, 10, 10) == 6
    assert B.actual_bpp(6, 10, 10) == pytest.approx(0.48)
    assert B.actual_bpp(7, 10, 10) == pytest.approx(0.56)
    with pytest.raises(ValueError):
        B.actual_bpp(1, 0, 10)


def test_resolution_normalized_candidate_ladder_is_half_up_and_has_no_absolute_ceiling():
    assert B.candidate_count(1_000_000, 1.0, 10.0, 1.0) == 12_500
    # The ceiling scales with resolution; it is not a fixed 20k cap.
    assert B.candidate_count(4_000_000, 4.0, 10.0, 1.0) == 200_000
    ladder = B.candidate_ladder(1_000_000, (0.5, 1.0), 10.0, (0.75, 1.0, 4 / 3))
    assert ladder == sorted(set(ladder))
    assert len(ladder) == 6


def test_rdo_selects_best_measured_psnr_under_exact_cap_and_never_interpolates():
    rows = [_row(5, 20.0, n=50), _row(6, 19.0, n=60), _row(7, 30.0, n=70)]
    selected = B.select_under_cap(rows, 0.5, 10, 10)
    assert selected["bytes"] == 5
    assert selected["psnr"] == 20.0
    assert B.select_under_cap([rows[-1]], 0.5, 10, 10) is None


def test_monotone_envelope_drops_dominated_and_equal_rate_rows():
    rows = [
        _row(5, 20.0),
        _row(5, 19.0),
        _row(6, 19.5),
        _row(7, 21.0),
        _row(8, 21.0),
        _row(9, 22.0),
    ]
    envelope = B.monotone_envelope(rows)
    assert [(row["bytes"], row["psnr"]) for row in envelope] == [
        (5, 20.0), (7, 21.0), (9, 22.0)
    ]


def test_bd_rate_uses_common_measured_interval_without_extrapolation():
    pytest.importorskip("scipy")
    qualities = [20.0, 22.0, 24.0, 26.0]
    reference = [
        {**_row(index + 10, quality), "bpp": rate}
        for index, (quality, rate) in enumerate(zip(qualities, [0.5, 1.0, 2.0, 4.0]))
    ]
    candidate = [
        {**_row(index + 20, quality), "bpp": 0.8 * rate}
        for index, (quality, rate) in enumerate(zip(qualities, [0.5, 1.0, 2.0, 4.0]))
    ]
    result = B.bd_rate(candidate, reference)
    assert result["defined"] is True
    assert result["bd_rate_percent"] == pytest.approx(-20.0, abs=1e-8)
    assert result["psnr_interval"] == [20.0, 26.0]
    assert result["no_extrapolation"] is True


def test_bd_rate_is_explicitly_undefined_for_sparse_or_disjoint_curves():
    pytest.importorskip("scipy")
    sparse = [{**_row(10 + i, 20.0 + i), "bpp": 0.5 + i} for i in range(3)]
    assert B.bd_rate(sparse, sparse)["reason"] == "fewer_than_four_envelope_points"
    low = [{**_row(10 + i, 10.0 + i), "bpp": 0.5 + i} for i in range(4)]
    high = [{**_row(20 + i, 20.0 + i), "bpp": 0.5 + i} for i in range(4)]
    assert B.bd_rate(low, high)["reason"] == "no_common_measured_psnr_interval"


def test_mechanism_metrics_define_positive_bleed_for_lost_step_contrast():
    target = np.zeros((32, 32, 3), dtype=np.float32)
    target[:, 16:] = 1.0
    perfect = B.mechanism_metrics(target, target)
    flat_prediction = np.full_like(target, 0.5)
    blurred = B.mechanism_metrics(flat_prediction, target)
    assert perfect["edge_mse"] == pytest.approx(0.0)
    assert perfect["signed_cross_edge_bleed"] == pytest.approx(0.0, abs=1e-12)
    assert blurred["edge_mse"] > 0.0
    assert blurred["signed_cross_edge_bleed"] > 0.0


def test_cluster_bootstrap_and_holm_are_deterministic():
    first = B.cluster_bootstrap([1.0, 2.0, 3.0], samples=1000, seed=3)
    second = B.cluster_bootstrap([1.0, 2.0, 3.0], samples=1000, seed=3)
    assert first == second
    adjusted = B.holm_adjust({"a": 0.01, "b": 0.04, "c": 0.2})
    assert adjusted == {"a": pytest.approx(0.03), "b": pytest.approx(0.08), "c": pytest.approx(0.2)}


def test_jsonl_resume_ignores_only_an_incomplete_final_append(tmp_path):
    path = tmp_path / "journal.jsonl"
    path.write_text('{"key": 1}\n{"key":')
    assert B.read_jsonl(path) == [{"key": 1}]
    path.write_text('{"key": broken}\n')
    with pytest.raises(ValueError, match="invalid JSONL"):
        B.read_jsonl(path)


def test_decoded_field_state_parity_is_renderer_independent():
    torch = pytest.importorskip("torch")
    from structsplat.gaussians import GaussianField

    reference = GaussianField(
        torch.tensor([[1.0, 2.0]]),
        torch.tensor([[0.0, 0.5]]),
        torch.tensor([0.25]),
        torch.tensor([[0.1, 0.2, 0.3]]),
    )
    identical = reference.detached()
    assert B._decoded_field_state_max_abs(reference, identical) == 0.0
    assert B._decoded_field_state_sha256(reference) == B._decoded_field_state_sha256(identical)

    changed = reference.detached()
    changed.colors[0, 0] += 1e-4
    assert B._decoded_field_state_max_abs(reference, changed) == pytest.approx(1e-4, abs=1e-8)
    assert B._decoded_field_state_sha256(reference) != B._decoded_field_state_sha256(changed)

    changed.opacities = torch.tensor([0.0])
    with pytest.raises(RuntimeError, match="optional-state mismatch"):
        B._decoded_field_state_max_abs(reference, changed)


def test_frozen_stage0a_manifest_hashes_sources_and_equal_arm_counts(tmp_path):
    root = Path(__file__).resolve().parents[1]
    images = sorted((root / "tests" / "test_images").glob("*.jpg"))
    path = tmp_path / "manifest.json"
    manifest = B.freeze_manifest(
        images,
        root,
        path,
        stage="stage0a",
        bytes_per_gaussian=10.0,
        targets=(0.01,),
        multipliers=(1.0,),
        bit_mixes=((10, 5, 5, 5),),
        arms=("tensor_wse", "random"),
        iters=1,
        renderer="cuda",
        compute_lpips=False,
    )
    loaded = B.load_manifest(path)
    assert loaded["manifest_sha256"] == manifest["manifest_sha256"]
    assert {image["id"] for image in loaded["images"]} == set(B.STAGE_IDS["stage0a"])
    assert loaded["fit_config"]["renderer"] == "cuda"
    assert loaded["renderer_protocol"]["equation"] == "normalized_weighted_sum"
    assert loaded["renderer_protocol"]["implementation"] == "owned_exact_cuda"
    assert B.dry_run_plan(loaded)["fits"] == 8
    B.validate_manifest_sources(loaded, root)
    tampered = json.loads(path.read_text())
    tampered["fit_config"]["iters"] = 2
    path.write_text(json.dumps(tampered))
    with pytest.raises(ValueError, match="manifest hash mismatch"):
        B.load_manifest(path)


def test_stage_gate_passes_only_all_preregistered_conditions():
    good = {
        "n_images": 2, "mean": 0.3, "ci95": [0.1, 0.5], "p_two_sided": 0.01,
    }
    comparison = {
        "comparisons": {
            "local_slic_sobel_control": {
                "target_psnr_delta_db": {"0.5": good, "1": good},
                "bd_rate_percent": {"n_images": 0, "mean": None, "ci95": [None, None]},
                "fit_plus_search_time_ratio": {"mean": 1.05},
                "edge_mse_delta": {"mean": -0.001},
                "signed_cross_edge_bleed_delta": {"mean": 0.001},
                "texture_mse_relative_delta": {"mean": 0.02},
            }
        }
    }
    manifest = {
        "stage": "stage1",
        "images": [{"id": "a"}, {"id": "b"}],
        "targets_bpp": [0.5, 1.0],
        "arms": [
            {"name": "tensor_wse"},
            {"name": "local_slic_sobel_control"},
        ],
        "mechanism_protocol": {"texture_failure_relative_threshold": 0.05},
    }
    gate = B.evaluate_stage_gate(manifest, comparison)
    assert gate["pass"] is True and gate["stage2_authorized"] is True
    comparison["comparisons"]["local_slic_sobel_control"]["fit_plus_search_time_ratio"] = {
        "mean": 1.11
    }
    assert B.evaluate_stage_gate(manifest, comparison)["pass"] is False


def test_tiny_run_is_cold_scored_journaled_and_resumable(tmp_path):
    pytest.importorskip("torch")
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    image[:, 16:] = 255
    source = tmp_path / "toy.png"
    from PIL import Image

    Image.fromarray(image).save(source)
    manifest_path = tmp_path / "manifest.json"
    outdir = tmp_path / "run"
    B.freeze_manifest(
        [source],
        tmp_path,
        manifest_path,
        stage="stage0a",
        bytes_per_gaussian=100.0,
        targets=(8.0,),
        multipliers=(1.0,),
        bit_mixes=((10, 5, 5, 5),),
        arms=("random",),
        iters=1,
        min_gaussians=8,
        compute_lpips=False,
        enforce_stage_ids=False,
    )
    first = B.run_manifest(manifest_path, tmp_path, outdir, device="cpu", verbose=False)
    assert first["new_fits"] == 1
    candidate = B.read_jsonl(outdir / "journals" / "candidates.jsonl")[-1]
    assert candidate["status"] == "ok"
    assert candidate["cold_parity_max_abs"] <= candidate["cold_parity_atol"]
    assert candidate["cold_parity_domain"] == "persisted_stream_decoded_field_state"
    assert candidate["in_memory_decoded_field_sha256"] == candidate["cold_decoded_field_sha256"]
    assert candidate["bytes"] == (outdir / candidate["stream_path"]).stat().st_size
    second = B.run_manifest(manifest_path, tmp_path, outdir, device="cpu", verbose=False)
    assert second["new_fits"] == 0
    assert second["completed_candidates"] == 1
    third = B.run_manifest(
        manifest_path,
        tmp_path,
        outdir,
        device="cpu",
        revalidate_candidates=True,
        verbose=False,
    )
    assert third["new_fits"] == 0
    assert third["completed_candidates"] == 1
    revalidated = B.read_jsonl(outdir / "journals" / "candidates.jsonl")[-1]
    assert revalidated["candidate_revalidated"] is True
    assert revalidated["stream_reencoded_identical"] is True
    analysis = B.analyze_manifest(
        manifest_path, tmp_path, outdir, device="cpu", make_figures=False
    )
    assert analysis["completion"]["complete"] is True
    selected = json.loads((outdir / "analysis" / "selected.json").read_text())
    assert selected[0]["selection_status"] == "ok"
    assert (outdir / "index.html").is_file()
