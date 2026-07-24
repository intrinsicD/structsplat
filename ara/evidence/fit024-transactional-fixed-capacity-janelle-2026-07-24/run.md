# FIT-024 transactional fixed-capacity storage — Janelle C0001 development check

## Scope

Source-bound one-image, one-seed, one-device engineering check of storage policy only. This is
development evidence, not held-out confirmation, a general fitting-quality equivalence proof, a
production speedup, or authorization to change the default.

## Claim disposition

| claim | kind and scope | evidence | disposition |
|---|---|---|---|
| Fixed storage retains FIT-023 topology, recovery, checkpoints, and gates | descriptive plumbing | executed source snapshots and resolved configs | confirm |
| Fixed storage causes no quality regression detectable beyond dynamic A/A variation on this image | measured single-image development | three final fields, histories, and cold rescoring | confirm narrowly |
| Fixed storage reduces peak allocated GPU memory materially | measured single-device resource | `torch.cuda.max_memory_allocated` | narrow to 11.3–11.8 MiB (about 0.5%) |
| Fixed storage materially accelerates the full pipeline | measured single-device resource | schedule/total timings and attempted-step normalization | refute on this image |
| Fixed and dynamic runs must choose identical events | asserted determinism | dynamic A and B chose different paths; CUDA uses atomic accumulation | refute |
| Fixed storage should become the repository default | production/default | only one image and seed | unauthorized |

## Protocol

- Source: `frame_00008/C0001`, RGB SHA-256
  `ae24fe99d3f8edbd04cd2c85ebc4fe9bfd95abe878c22abb7691cadcfc5c411b`, mask SHA-256
  `94dcbf7005dbeb1d183e259a569d783aa5df900255e763385bed91f02d3b80c3`, seed
  `1559856117`.
- Repository commit: `8cfe30d8709dcdf19bf1a10d1d72011b54a89ac5`, dirty working tree with exact
  executed source snapshots preserved in every arm. All three status hashes equal
  `3309951b77cd98f8c295cb111cc27039e18fdd90e046f515e381ebfe6fdd3874`.
- Device: NVIDIA GeForce RTX 4090; Torch 2.7.0+cu126; CUDA runtime 12.6.
- Common method: FIT-023 checkpoint-only winner, global refinement, `cuda_tiled`, 5,000-row
  initialization, 11,000-row capacity, state-matched Pareto checkpoints every 50 steps, event
  color solve off, identical full foreground/boundary/CVaR/hole/outside commit gate.
- Arms ran sequentially in one Python process: dynamic A, fixed capacity, dynamic B. Only
  `SafeScheduleConfig.storage_policy` differs; source, initialization, fit config, all other
  schedule fields, environment, and loaded process are shared.
- `--no-archive` excludes `.rtgsv` and every rate/codec claim.

Exact execution:

```bash
/home/alex/miniconda3/bin/python - <<'PY'
from pathlib import Path
from scripts import fit_janelle_safe_commit_schedule as runner

common = [
    "--capture-root", "/home/alex/Dropbox/Work/Janelle/2025_03_07_stage_with_fabric",
    "--realtime-root", "/home/alex/Documents/realtime-gs",
    "--frame", "frame_00008", "--view-id", "C0001", "--device", "cuda:0",
    "--preview-width", "1200", "--capacity", "11000",
    "--pareto-safe-checkpoints", "--pareto-checkpoint-every", "50", "--no-archive",
]
root = Path("runs/janelle_C0001_storage_ab_active_shape_20260724")
for arm, policy in (
    ("dynamic_a", "dynamic"),
    ("fixed_capacity", "fixed_capacity"),
    ("dynamic_b", "dynamic"),
):
    args = runner.build_parser().parse_args(
        common + ["--storage-policy", policy, "--out", str(root / arm)]
    )
    runner.run(args)
PY
```

Comparison generation:

```bash
python scripts/compare_janelle_safe_schedule_variants.py \
  --run 'Dynamic A=runs/janelle_C0001_storage_ab_active_shape_20260724/dynamic_a' \
  --run 'Fixed capacity=runs/janelle_C0001_storage_ab_active_shape_20260724/fixed_capacity' \
  --run 'Dynamic B=runs/janelle_C0001_storage_ab_active_shape_20260724/dynamic_b' \
  --out runs/janelle_C0001_storage_ab_active_shape_20260724
```

## Raw result

Lower is better for all columns after boundary PSNR.

| arm | FG PSNR | boundary PSNR | CVaR99 MSE | p99 MSE | interior holes | boundary holes |
|---|---:|---:|---:|---:|---:|---:|
| dynamic A | 27.0670227 | 11.4205702 | .1520401 | .0149374 | 1.4159% | 29.7169% |
| fixed capacity | 27.0629022 | 11.3997668 | .1525004 | .0147774 | 1.4643% | 28.6279% |
| dynamic B | 27.0629053 | 11.3944529 | .1529829 | .0145605 | 1.4755% | 28.6682% |

Fixed foreground PSNR is `0.0000031 dB` below the lower dynamic endpoint; boundary PSNR, CVaR99,
p99, and interior holes lie between the controls. Fixed boundary holes are favorably `0.0403`
percentage points below the better dynamic control. All outside render and raw-coverage maxima
are exactly zero.

| arm | attempted / accepted steps | schedule | total | peak allocated GPU memory |
|---|---:|---:|---:|---:|
| dynamic A | 17,511 / 1,893 | 271.900 s | 407.025 s | 2,246,110,208 B (2142.1 MiB) |
| fixed capacity | 17,831 / 1,939 | 272.124 s | 405.496 s | 2,233,716,736 B (2130.2 MiB) |
| dynamic B | 18,828 / 1,813 | 285.190 s | 418.433 s | 2,245,528,064 B (2141.5 MiB) |

Fixed peak allocation is 11.8/11.3 MiB below dynamic A/B and 11.5 MiB below their mean. Fixed
schedule time is 0.08% slower than dynamic A and 4.58% faster than dynamic B, but the arms execute
different numbers of safe recovery steps. Schedule milliseconds per attempted step are
15.527/15.261/15.147 for dynamic A/fixed/dynamic B: fixed lies inside the dynamic range. The
correct disposition is runtime-neutral, not faster.

## Scientist pass

- All required fields, configs, histories, native images, HTML pages, and source snapshots exist.
- Source, seed, environment, fit config, initialization, schedule excluding `storage_policy`,
  repository status, and executed-source hashes match across all arms.
- The three serialized initial fields are byte-identical, SHA-256
  `cfa13a084195149c637c0e50297dc010d812f308bfb4703ac55abb72dc26999b`.
- Every snapshotted source byte count and SHA-256 matches its manifest. The final NPZ SHA-256s are
  `5d80f4b31266bc91a16d41f7f832e4c5ae70d257bd2c1aaf44a0c60545df7885`
  (dynamic A), `2d54268f0cfe0f7bc1ce65594735df3a90742e732223e9b2ad45a8c3c59ccd81`
  (fixed), and `85f193ccefeb1627ad61afafc3a1e802df9f1d18849d44589c73c51385e413b1`
  (dynamic B).
- Cold NPZ reload and full recomputation matched every stored protected metric exactly in the
  audited environment for all three fields. All three final native reconstructions are RGB
  3964x1444.
- Reapplying the safe gate over accepted non-marker transitions found zero failures. Summed
  attempted/accepted steps match every top-level history.
- Dynamic A and dynamic B choose different accepted proposals and differ by `0.00412 dB`
  foreground and `0.02612 dB` boundary at termination. CUDA atomic accumulation is therefore an
  observed comparison confound; event-sequence identity is not attributed to storage.
- Focused changed-surface tests pass 47/47. The broader fit/mask surface passes 185/186; the
  pre-existing last-bit no-mask repeat test fails once by `1.907e-6 dB` and passes immediately in
  isolation. The portable gate passes 1,384, skips 4, and retains three unrelated baseline
  failures: affine rank-deficiency condition-number expectation, unavailable Torch CUDA
  `pci_bus_id`, and a filesystem race assertion.
- The executed run used the pre-audit telemetry label `peak_physical_rows`, although that value
  was only the final physical row count on the dynamic path. The implementation now truthfully
  emits `final_physical_rows`; no run measurement or decision changed.
- Fixed shape eliminates topology-driven append/pad resizing, not all allocation: detached
  transaction proposals and Pareto checkpoints still clone capacity-shaped scratch tensors.
- `converged=true` means the polish transaction reached its deterministic fixed point. Configured
  0.1% interior and 1% boundary coverage targets remain unmet.
- The exact CUDA extension ELF was not archived. All arms ran in one process and share the loaded
  extension, but no cross-machine or cross-version timing claim follows.

Comparison artifacts hash to
`d6ed933615796290802e1c7cf73e3334d765a64e36fceb4a4e59323ef7855069`
(`comparison.json`) and
`316d3a4c11e8af4c93a321a5f4241ff458a5b5386ad2dc617c6d5bb9face5ccb`
(`index.html`). The machine-readable scientist-pass summary is
`ara/evidence/fit024-transactional-fixed-capacity-janelle-2026-07-24/audit.json`
(mirrored in the run root), SHA-256
`56c5e27847e2f00cd7b2c4f4ca27b69903c8fe96f7c8033f7446e4859bbc7e43`.

## Development and unresolved gaps

This image was already exposed by FIT-023 and is development data. No protected confirmation data
was consumed. Multi-image and multi-seed evidence is required for a default decision. A meaningful
speed claim additionally needs repeated wall-clock runs with a fixed accepted-work trace or a
separate allocator/topology microprofile; the current safe auctions legitimately execute different
work.

## Disposition

Confirm fixed-capacity storage as an opt-in engineering implementation with no detected
source-bound quality regression beyond dynamic A/A variation on this image. Narrow the resource
result to a roughly 0.5% reduction in peak allocated GPU memory. Refute a material speedup claim
for this run and leave `dynamic` as the default.
