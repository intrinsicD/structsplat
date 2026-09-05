"""FIT-051's immutable parent transfer and bounded decision helpers."""
from __future__ import annotations

import math
import statistics

ARMS = ["noop", "legacy_cg32", "actual_cg_ray", "actual_gradient_ray", "actual_jacobi_ray",
        "native_gradient_ray", "adam32"]
DIRECTIONS = {"actual_cg_ray": "cg", "actual_gradient_ray": "streaming_gradient",
              "actual_jacobi_ray": "streaming_jacobi", "native_gradient_ray": "native_gradient"}
IMAGE_IDS, SEEDS = [9, 25, 30, 34], [0, 1]
PARENT_MANIFEST_SHA256 = "5c5629df090b7946f1ee85ab98d988921998096888261c40192e7cd3a7a4f427"
PARENT_PROTOCOL_DIGEST = "edad3bb041fb2e34697a101f729402bf12470c411a5abe2567a91be04a027153"
PARENT_SOURCE_COMMIT = "e2bf6ae6e06ca8050d9aa8a93d713679a0c9c150"
PARENT_FILES = {
    "parents/coco000000000009_s0/config.json": "48bafbe98db8753215f91572ea5b9306637065baaa4df3299c3ec105b95d892c",
    "parents/coco000000000009_s0/field.npz": "cb6ab940dc00cb3a1a3f399120123071f16f8aef32ce249432d6a9a2da0a7af4",
    "parents/coco000000000009_s0/history.json": "6d860939b018122179fbd324c584791cac1ba6761d3e4a196fe8468d7e46e426",
    "parents/coco000000000009_s0/initial_field.npz": "7cf15a06f8ac7500fda051246228cc274c26cdc1624694cdb07575d349bbc0c4",
    "parents/coco000000000009_s0/optimizer_state.pt": "f8372e2088562f4268aec895dec8fe84a47a5da34e623dcff4a4c7fea9b8aee1",
    "parents/coco000000000009_s0/target.npy": "914d88f4f07f55c2f7ba0c473059119c5eb5ed5dfb83febfbe12856373654213",
    "parents/coco000000000009_s1/config.json": "02fd74a76ee8e5ae38529a7b26d670e8ecab8cb9e50fab94b3775ff45355f35d",
    "parents/coco000000000009_s1/field.npz": "4cf04cf51a886ae45815d2c35ed3feb0e73c6c10a22dd7fdef627e928741ca0e",
    "parents/coco000000000009_s1/history.json": "6df9b29f7e2d3110bce2b0957df016791c1cbbd785d41ac877aac859168ae525",
    "parents/coco000000000009_s1/initial_field.npz": "789967fe2792f54419c1874584f0ee2d4326c7cb7ae2ab873169445de4ced62c",
    "parents/coco000000000009_s1/optimizer_state.pt": "507b5200309db0ef22bd936f49ae1ee5d4bfbd2f852282a03a133e662d99580e",
    "parents/coco000000000009_s1/target.npy": "914d88f4f07f55c2f7ba0c473059119c5eb5ed5dfb83febfbe12856373654213",
    "parents/coco000000000025_s0/config.json": "d9f609cece5110f1e044e62e4283a8400e1ea34617b286c7234f6da3bdf31c17",
    "parents/coco000000000025_s0/field.npz": "22ff4e3d17991192d2d500685e9006a4b6e70d1151ca11ec3aa885285a127238",
    "parents/coco000000000025_s0/history.json": "b092c35b69591a5a510e41a4c1eacdb2d6bfe6c5ff22d842b54316719eba4cd5",
    "parents/coco000000000025_s0/initial_field.npz": "ab1e392873f110d3f533adb1af8a6c8ba077d7ec5780b718713a3e061604951c",
    "parents/coco000000000025_s0/optimizer_state.pt": "2c288b5e5baed4c0946cf5156a31bb8a30c19f8402303caa95aa1d1670d4ed63",
    "parents/coco000000000025_s0/target.npy": "77fd5212f1e1c34f8ab55ca7099bcb120566508fefe6d685780ce79ec5dc35c6",
    "parents/coco000000000025_s1/config.json": "57ce8b235680dcac8474e0f604a4bca9c02dda2dc56b2867f2bfddfcc46618f8",
    "parents/coco000000000025_s1/field.npz": "aa71c7271b284afc1e235c7f4e76a4972b08d5426294fa35654a2c4889093af2",
    "parents/coco000000000025_s1/history.json": "a4f3a59d70c27c035342d77b825e49efea0e71df82fe1b416b04226edc946ff6",
    "parents/coco000000000025_s1/initial_field.npz": "cd0bacf57d1b3c4577c1463c6d1c5d12fad34a7baf666d0d0b74abd05d7da454",
    "parents/coco000000000025_s1/optimizer_state.pt": "4050eafaf2878bae28bdf5c060476ba1c53846bb08e05ece1c7d4cf4a5d66722",
    "parents/coco000000000025_s1/target.npy": "77fd5212f1e1c34f8ab55ca7099bcb120566508fefe6d685780ce79ec5dc35c6",
    "parents/coco000000000030_s0/config.json": "37e2627be98fe6e983d4e3ead00e1a893a59f37913530d1c59c38e7a60163d8e",
    "parents/coco000000000030_s0/field.npz": "00da077aa51ae4b02556ed37d1c310f29f97cd8c867ea7871419124052adcf8c",
    "parents/coco000000000030_s0/history.json": "78a6f73b4fb3ba5c62dfc439e0d872b4cb1ff4ba0477075bb2c454fdb270442a",
    "parents/coco000000000030_s0/initial_field.npz": "56e56ad4733f0091efdc26883f94af6b4cb7fdf678fe631a3d9f2a079d44471e",
    "parents/coco000000000030_s0/optimizer_state.pt": "a156fc7cb2ea990b745660004b1c032dbfc0c68d93de2062d823643ddedbd9fd",
    "parents/coco000000000030_s0/target.npy": "f24fa88987cdf1ab933ceb910f3d418bf93ef98360799b4854fa32ffa53c562f",
    "parents/coco000000000030_s1/config.json": "e6c08db58307b50d25b44350e814bdba2cf5f3926fd7b116d7fb4b44af149c92",
    "parents/coco000000000030_s1/field.npz": "9071df653048b8e8df4b7995cbe57228e9dbeb241b309669b62b657276921a7e",
    "parents/coco000000000030_s1/history.json": "179fb1219457194c1aa1c5fe2a9c84464e0cc77b170fee1f6d1d1f71cd3949a7",
    "parents/coco000000000030_s1/initial_field.npz": "de3c6809c81eb65a21bdc9d25937300c2c713894e1f0b2cce8b79b88242cf5db",
    "parents/coco000000000030_s1/optimizer_state.pt": "4face6cf5b7e35ea86f3e1ec4b4234f3de1619e0f2e7e2230313baf30dc22c52",
    "parents/coco000000000030_s1/target.npy": "f24fa88987cdf1ab933ceb910f3d418bf93ef98360799b4854fa32ffa53c562f",
    "parents/coco000000000034_s0/config.json": "776d31b1c202e6afd1763122cfd722923c588bffd49a2d2f030d96401fdc936a",
    "parents/coco000000000034_s0/field.npz": "8398e3af4b486a0809d5941356669f1332b4e5944e156c31f7eee1007c2f8c74",
    "parents/coco000000000034_s0/history.json": "66b6a5c8009428b75f2a1e8e1d96d6165f3b055f1b9da43e5eb030bcfbcce70e",
    "parents/coco000000000034_s0/initial_field.npz": "cf9152ba0af253efb7e16fe7db91ad5965ce2c9f57c4ace3150c960f45847b56",
    "parents/coco000000000034_s0/optimizer_state.pt": "84d55622630608ca5eca73e2c7cc0d0d0b9d81bb806eb382d04c001e02b7b74c",
    "parents/coco000000000034_s0/target.npy": "1885ed269e0c882fadc488c3e461adacb474b1c811f3bb3b40445d3a013225f5",
    "parents/coco000000000034_s1/config.json": "87e5bcaff3d71cba34c3a74f6ed37e8107626d904c0664a92f2757a50ea0cc46",
    "parents/coco000000000034_s1/field.npz": "23a05caaa99b2516d6eea7ac9c6224a3d2f808fa89d1729f8755e3701ae3536e",
    "parents/coco000000000034_s1/history.json": "1382e1fe95d95da0135d46e24da86ed3bb6b206a2db58284028c87b73f47d37d",
    "parents/coco000000000034_s1/initial_field.npz": "48131f924047da6dd399dda9faad8c2d9bf98ed14a5c9ec554850dbeb099e66c",
    "parents/coco000000000034_s1/optimizer_state.pt": "7f87714cffce8d381ad66e31024c9977f28d65b04b60294f575f49551f69b9cb",
    "parents/coco000000000034_s1/target.npy": "1885ed269e0c882fadc488c3e461adacb474b1c811f3bb3b40445d3a013225f5",
}


def parent_id(image, seed):
    return f"coco{image:012d}_s{seed}"


def expected_cells(protocol=None, diagnostic=False):
    if diagnostic:
        return [f"procedural_s77_{arm}" for arm in ARMS]
    return [f"{parent_id(image, seed)}_{arm}" for image in IMAGE_IDS for seed in SEEDS for arm in ARMS]


def finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def compare_rows(rows, protocol, problems, *, diagnostic=False):
    """No utility conclusion can bypass complete row/metric/gate validation."""
    from scripts.experiments.fit050_color_ray import _quality_record
    from structsplat.safe_schedule import CommitTolerances, safe_commit_decision
    expected = expected_cells(protocol, diagnostic)
    ids = [row.get("cell_id") for row in rows]
    if len(ids) != len(set(ids)) or set(ids) != set(expected):
        problems.append("FIT-051 matrix differs from the complete frozen parent/arm product")
    count = 16 if diagnostic else 2000
    for row in rows:
        try:
            pid = "procedural_s77" if diagnostic else parent_id(row["image_id"], row["seed"])
            if (row["method"] not in ARMS or row["parent_id"] != pid
                    or row["cell_id"] != f"{pid}_{row['method']}"):
                raise ValueError("row identity disagrees")
            if not diagnostic and (row["image_id"] not in IMAGE_IDS or row["seed"] not in SEEDS):
                raise ValueError("unexpected parent")
            if row["status"] == "error":
                if not row.get("error"):
                    raise ValueError("error cell lacks explanation")
                continue
            if row["status"] != "ok" or type(row["n_gaussians"]) is not int or row["n_gaussians"] != count:
                raise ValueError("wrong status/count")
            for key in ("accepted", "coefficients_changed", "cold_parameters_exact"):
                if type(row.get(key)) is not bool:
                    raise ValueError(f"invalid {key}")
            steps = (2 if diagnostic else 32) if row["method"] == "adam32" else 0
            if (type(row["iterations_run"]) is not int or type(row["selected_iteration"]) is not int
                    or row["iterations_run"] != steps or row["selected_iteration"] != (steps if row["accepted"] else 0)):
                raise ValueError("wrong attempted/selected horizon")
            for key in ("psnr", "raw_mse", "ms_ssim", "transaction_seconds", "total_seconds",
                        "cold_render_max_abs", "selected_replay_max_abs"):
                if not finite_number(row.get(key)):
                    raise ValueError(f"nonfinite {key}")
            if not diagnostic and not finite_number(row.get("lpips")):
                raise ValueError("nonfinite LPIPS")
            if row["raw_mse"] < 0 or abs(row["psnr"] + 10 * math.log10(max(row["raw_mse"], 1e-12))) > 1e-8:
                raise ValueError("raw PSNR/MSE convention differs")
            if (row["total_seconds"] != row["transaction_seconds"] or row["total_seconds"] <= 0
                    or not 0 <= row["cold_render_max_abs"] <= 2e-5
                    or not 0 <= row["selected_replay_max_abs"] <= 2e-5
                    or not row["cold_parameters_exact"]):
                raise ValueError("runtime or selected/cold replay failed")
            before = _quality_record(row["parent_protected_metrics"], count)
            selected = _quality_record(row["protected_metrics"], count)
            _quality_record(row["reporting_metrics"], count)
            if row["accepted"]:
                allowed = [2.0 ** -k for k in range(6)] if row["method"] in DIRECTIONS else [1.0]
                if ((row["method"] in DIRECTIONS and not row["coefficients_changed"])
                        or row["selected_fraction"] not in allowed
                        or not safe_commit_decision(before, selected, CommitTolerances())[0]):
                    raise ValueError("accepted field fails unchanged gate or nonzero change")
            elif (row["coefficients_changed"] or row["selected_fraction"] != 0
                  or row["parent_protected_metrics"] != row["protected_metrics"]):
                raise ValueError("rollback did not restore the exact parent")
            if row["method"] != "adam32" and row["noncolor_changed_fields"]:
                raise ValueError("RGB-only method changed another field")
            counts = row["counts"]
            required_counts = {"quality_evaluations", "gaussian_renders", "raw_coverage_passes",
                               "basis_denominator_passes", "basis_apply_calls", "basis_transpose_calls"}
            if (not isinstance(counts, dict) or not required_counts <= counts.keys()
                    or any(type(value) is not int or value < 0 for value in counts.values())):
                raise ValueError("invalid exact work counts")
            if not diagnostic:
                for key, filename in (("parent_field_sha256", "field.npz"),
                                      ("parent_optimizer_sha256", "optimizer_state.pt"),
                                      ("target_sha256", "target.npy")):
                    if row[key] != PARENT_FILES[f"parents/{pid}/{filename}"]:
                        raise ValueError("row parent binding differs from frozen imported parent")
        except (KeyError, TypeError, ValueError) as exc:
            problems.append(f"FIT-051 {row.get('cell_id')}: {exc}")


def summarize_rows(rows, protocol, *, diagnostic=False):
    problems = []
    compare_rows(rows, protocol, problems, diagnostic=diagnostic)
    if problems:
        return {"records": [{"method": method, "pairs": [], "median_image_averaged_gain_db": None,
                             "passes_utility_gate": False, "complete_matrix": False} for method in ARMS[1:]],
                "problems": problems, "complete_matrix": False, "speed_claim": False,
                "default_promotion": False, "data_role": protocol["data_role"]}
    complete = not problems and all(row.get("status") == "ok" for row in rows)
    lookup = {(r["parent_id"], r["method"]): r for r in rows if r.get("status") == "ok"}
    records = []
    for method in ARMS[1:]:
        pairs = []
        for image in IMAGE_IDS:
            for seed in SEEDS:
                pid = parent_id(image, seed)
                candidate, baseline = lookup.get((pid, method)), lookup.get((pid, "noop"))
                if candidate is None or baseline is None:
                    continue
                pairs.append({"image_id": image, "seed": seed,
                    "gain_db": candidate["psnr"] - baseline["psnr"],
                    "ms_ssim_difference": candidate["ms_ssim"] - baseline["ms_ssim"],
                    "lpips_difference": candidate["lpips"] - baseline["lpips"],
                    "accepted": candidate["accepted"], "coefficients_changed": candidate["coefficients_changed"],
                    "comparators": {control: {"gain_db": candidate["psnr"] - lookup[pid, control]["psnr"],
                        "transaction_seconds_ratio": candidate["transaction_seconds"] / lookup[pid, control]["transaction_seconds"]}
                        for control in ("legacy_cg32", "actual_cg_ray", "adam32") if (pid, control) in lookup}})
        image_gains = [statistics.mean(p["gain_db"] for p in pairs if p["image_id"] == image)
                       for image in IMAGE_IDS if sum(p["image_id"] == image for p in pairs) == 2]
        median = statistics.median(image_gains) if len(image_gains) == 4 else None
        utility = (complete and not diagnostic and len(pairs) == 8 and median is not None and median >= 0.1
                   and all(p["gain_db"] >= -0.01 and p["ms_ssim_difference"] >= -0.001
                           and p["lpips_difference"] <= 0.002 for p in pairs)
                   and any(p["accepted"] and p["coefficients_changed"] for p in pairs))
        records.append({"method": method, "pairs": pairs, "median_image_averaged_gain_db": median,
                        "passes_utility_gate": bool(utility), "complete_matrix": complete})
    return {"records": records, "problems": problems, "complete_matrix": complete,
            "speed_claim": False, "default_promotion": False, "data_role": protocol["data_role"]}


def quality_inputs(field, target, cfg, mask, constraint, tau):
    """The maintained reference quality expression, returning its already-computed raw inputs."""
    from structsplat.safe_schedule import _quality_from_render, _quality_render_inputs
    raw, denominator = _quality_render_inputs(field, cfg, *target.shape[:2], mask, tau)
    quality = _quality_from_render(raw, target, denominator, mask, constraint, tau, field.n,
                                   tail_backend="reference")
    return quality, raw, denominator


def run_endpoint(parent, target, cfg, mask, constraint, schedule, method, optimizer_state, *, diagnostic=False):
    """Same maintained CG/Adam endpoint controls, with rejected fields retained for audit."""
    import copy
    from dataclasses import asdict, replace
    import time
    import torch
    from benchmarks.fit050_controls import _changed_noncolors, sync
    from structsplat.fit import _solve_colors_normalized, fit
    from structsplat.safe_schedule import safe_commit_decision
    if method not in ("noop", "legacy_cg32", "adam32"):
        raise ValueError("unknown endpoint control")
    cfg = replace(cfg, quality_coverage_backend="reference", quality_tail_backend="reference")
    sync(target.device)
    started = time.perf_counter()
    before, raw, den = quality_inputs(parent, target, cfg, mask, constraint, schedule.coverage_tau)
    arrays = {"parent_render": raw.detach(), "parent_denominator": den.detach()}
    field = parent.detached()
    counts = {"quality_evaluations": 1, "gaussian_renders": 1, "raw_coverage_passes": 1,
              "basis_denominator_passes": 0, "basis_apply_calls": 0,
              "basis_transpose_calls": 0, "gradient_evaluations": 0}
    metadata = {"arm": method, "parent_metrics": before.to_dict(), "selected_metrics": before.to_dict(),
        "quality_coverage_backend": "reference", "quality_tail_backend": "reference", "counts": counts,
        "accepted": False, "coefficients_changed": False, "selected_fraction": 0.0,
        "selected_alpha": 0.0, "rollback_reason": "noop", "trials": []}
    history = {}
    if method == "legacy_cg32":
        cg_cfg = replace(cfg, color_solve_maxiter=32)
        stats = _solve_colors_normalized(field, target, cg_cfg, *target.shape[:2], support_fade_alpha=1.0)
        metadata.update({"legacy_cg": stats, "candidate_config": asdict(cg_cfg)})
        counts["basis_denominator_passes"] = stats["denominator_calls"]
        counts["basis_apply_calls"] = stats["basis_apply_calls"]
        counts["basis_transpose_calls"] = stats["basis_transpose_calls"]
    elif method == "adam32":
        steps = 2 if diagnostic else 32
        candidate_cfg = replace(cfg, iters=steps, log_every=1)
        result = fit(field, target, candidate_cfg, verbose=False,
                     optimizer_state=copy.deepcopy(optimizer_state), return_optimizer_state=True)
        field, history = result["field"], result["history"]
        metadata.update({"candidate_config": asdict(candidate_cfg), "iterations_run": result["iterations_run"],
                         "candidate_fit_seconds": result["fit_seconds"]})
        counts["gaussian_renders"] += steps + 1
        counts["gradient_evaluations"] = steps
    candidate_field = field if method != "noop" else None
    selected, quality = field, before
    if method != "noop":
        candidate, candidate_raw, candidate_den = quality_inputs(field, target, cfg, mask, constraint,
                                                                schedule.coverage_tau)
        arrays.update({"candidate_render": candidate_raw.detach(), "candidate_denominator": candidate_den.detach()})
        counts["quality_evaluations"] += 1
        counts["gaussian_renders"] += 1
        counts["raw_coverage_passes"] += 1
        accepted, reasons = safe_commit_decision(before, candidate, schedule.tolerances,
                                                schedule.hole_regression_budget)
        metadata.update({"candidate_metrics": candidate.to_dict(), "candidate_reasons": reasons,
                         "accepted": accepted})
        if accepted:
            quality = candidate
            metadata.update({"selected_fraction": 1.0, "selected_alpha": 1.0, "rollback_reason": None})
        else:
            selected = parent.detached()
            metadata["rollback_reason"] = "candidate_rejected"
    metadata.update({"selected_metrics": quality.to_dict(),
        "noncolor_changed_fields": _changed_noncolors(parent, selected),
        "coefficients_changed": not torch.equal(parent.colors, selected.colors),
        "foreground_mse_improved": quality.foreground_mse < before.foreground_mse})
    sync(target.device)
    metadata["transaction_seconds"] = time.perf_counter() - started
    return selected, quality, metadata, arrays, history, candidate_field
