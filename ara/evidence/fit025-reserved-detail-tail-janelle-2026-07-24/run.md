# FIT-025 reserved detail tail — Janelle C0001 development comparison

## Outcome

At identical 12,024-row fixed physical storage, the generic +512 active-budget arm is the bounded
development winner. It is strictly better than the specialized +512 detail tail on foreground,
boundary, CVaR99, p99, interior coverage, and boundary coverage, with exact zero outside the mask.
It is also the fastest observed arm, although one sequential run per arm with different auction
work does not support a general speed or throughput claim.

The specialized tail is functional and Pareto-safe: all four 128-row transactions improved the
selected state. Its gain per row fell by about 12x from the first to fourth wave, and it spent rows
only on already-covered deep-interior texture while large coverage defects remained. Keep the tail
opt-in. Do not raise or tune its stopping threshold from this exposed single image.

## Claim disposition

| claim | kind and scope | evidence | disposition |
|---|---|---|---|
| Physical capacity can exceed the ordinary active ceiling without exposing reserve rows | engineering invariant | config, phase history, storage telemetry, tests | confirm |
| The specialized tail activates only error-selected, covered interior detail in safe transactions | mechanism | executed source, per-wave metadata, gate replay | confirm |
| A specialized +512 tail is better than a generic +512 active budget | single-image development comparison | matched three-arm terminal fields | refute |
| The generic +512 arm is fastest | observed wall time | one sequential execution per arm | confirm only for this execution |
| Pooling itself caused the timing ordering | storage-policy claim | all three arms are fixed-capacity | not tested; unauthorized |
| A nonzero gain-per-row threshold should become default | policy/default | threshold was zero and image is exposed | unauthorized |
| FIT-025 supports a repository-wide default change | generalization/default | one image, seed, and device | unauthorized |

## Protocol

- Source: `frame_00008/C0001`, RGB SHA-256
  `ae24fe99d3f8edbd04cd2c85ebc4fe9bfd95abe878c22abb7691cadcfc5c411b`, mask SHA-256
  `94dcbf7005dbeb1d183e259a569d783aa5df900255e763385bed91f02d3b80c3`, seed
  `1559856117`.
- Device: NVIDIA GeForce RTX 4090; Torch 2.7.0+cu126; CUDA runtime 12.6.
- Execution commit: `8cfe30d8709dcdf19bf1a10d1d72011b54a89ac5`, with dirty status SHA-256
  `d4635f3de6c9f2abcf4db5def50af93607037aa650ad5f1fe290c4142deea073`.
  The manifest and every arm preserve exact executed-source hashes/snapshots.
- Common method: FIT-023 checkpoint-only global schedule, `cuda_tiled`, 5,000-row identical
  initialization, Pareto checkpoints every 50 steps, event color solve off, and the same
  full-resolution foreground/boundary/CVaR/hole/outside gate.
- Common storage: `fixed_capacity`, physical capacity 12,024. All arms start at the same 5,000
  active rows and use capacity-shaped field and Adam storage.
- Baseline: ordinary active ceiling 11,000; no tail.
- Generic +512: ordinary active ceiling 11,512; no tail.
- Adaptive +512: ordinary active ceiling 11,000; up to four 128-row post-color-solve detail-tail
  waves, zero additional gain-per-row floor.
- `--no-archive` excludes transport compression and every byte/rate claim.

The source-bound arm commands are preserved in
`runs/janelle_C0001_detail_tail_ablation_20260724/experiment_config.json`. The experiment driver is:

```bash
/home/alex/miniconda3/bin/python \
  scripts/run_janelle_detail_tail_ablation.py \
  --out runs/janelle_C0001_detail_tail_ablation_20260724
```

## Results

Lower is better for every metric after boundary PSNR. Hole columns are percentages.

| arm | N | FG PSNR | boundary PSNR | CVaR99 MSE | p99 MSE | interior holes | boundary holes |
|---|---:|---:|---:|---:|---:|---:|---:|
| fixed 11k baseline | 11,000 | 27.065330 | 11.412448 | .1523867 | .0148618 | 1.4526% | 28.6376% |
| generic +512 | 11,512 | **27.219252** | **11.582541** | **.1471049** | **.0139876** | **.9236%** | **25.5772%** |
| adaptive tail +512 | 11,512 | 27.106930 | 11.432527 | .1511119 | .0142505 | 1.4343% | 28.4278% |

Versus baseline, generic +512 gains `+0.153922/+0.170092 dB` foreground/boundary, reduces
CVaR99/p99 by `3.47%/5.88%`, and reduces interior/boundary holes by `0.5290/3.0604` percentage
points. Versus the equal-count adaptive tail it gains `+0.112322/+0.150013 dB`, reduces
CVaR99/p99 by `2.65%/1.84%`, and reduces holes by `0.5107/2.8507` percentage points. Both outside
metrics are exactly zero for every arm.

| arm | attempted / accepted steps | schedule | total | peak allocated GPU memory |
|---|---:|---:|---:|---:|
| fixed 11k baseline | 17,591 / 1,849 | 276.097 s | 416.037 s | 2,132.6 MiB |
| generic +512 | 17,191 / 1,970 | **272.070 s** | **406.288 s** | 2,163.7 MiB |
| adaptive tail +512 | 21,111 / 2,229 | 315.321 s | 450.837 s | 2,162.6 MiB |

Generic +512 is 9.749 s faster end to end than baseline and 44.549 s faster than adaptive in this
execution. The adaptive arm attempts 3,920 more recovery steps than generic. Because safe auctions
follow different trajectories, these wall times rank the complete observed arms; they do not
measure allocator speed. FIT-024 remains the fixed-versus-dynamic storage evidence.

## Tail behavior

Every selected tail proposal is a 128-row detail birth; no split wins. The full Pareto gate accepts
all four waves and the phase stops at its 512-row budget.

| wave | active N | selected recovery step | aggregate gain | gain / new row | FG PSNR after | boundary PSNR after |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 11,128 | 80 | .00426253 | 3.3301e-5 | 27.082454 | 11.419483 |
| 2 | 11,256 | 80 | .00259221 | 2.0252e-5 | 27.091150 | 11.426482 |
| 3 | 11,384 | 50 | .00087161 | 6.8095e-6 | 27.094050 | 11.428691 |
| 4 | 11,512 | 80 | .00035669 | 2.7866e-6 | 27.095416 | 11.428927 |

This is clear diminishing return, but a `1e-5` gain-per-row floor would be a post-hoc hypothesis:
on the recorded trace it would reject the third full batch, yet the resulting post-color/polish
trajectory has not been run. If the tail is revisited, preregister that floor and compare it on
new development images rather than promoting it from this trace.

## Scientist pass

- Source, seed, environment, fit configuration, fit window/size, physical capacity, storage
  policy, and the initial NPZ are identical across arms. The initial NPZ SHA-256 is
  `cfa13a084195149c637c0e50297dc010d812f308bfb4703ac55abb72dc26999b`.
- Relative to baseline, the only schedule differences are the intended ordinary target/active
  ceiling fields for generic +512 and the tail reserve/polish target for adaptive +512.
- All manifest and per-arm source hashes, executed snapshots, terminal field hashes, required
  artifacts, and 3964x1444 native image dimensions pass.
- Cold NPZ reload and full CUDA recomputation reproduce every terminal protected metric exactly
  (`max_abs_delta=0`) for all three fields.
- Independent gate replay passes 96 accepted and 62 rejected non-marker transitions. History
  continuity and summed attempted/accepted steps match the top-level records in every arm.
- All fields are finite and preserve exact zero outside both render and raw coverage.
- Focused FIT-025 runner/schedule tests pass 29/29. The broader changed fit/storage surface passes
  155/155, and Ruff passes on all changed Python sources.
- The broad non-slow repository invocation completed with 1,807 passed and 5 skipped, but retained
  90 failures and 4 collection/setup errors from unavailable frozen bundles, Landlock ABI,
  deterministic-thread environment contracts, sealed hardware/hash checks, an existing affine
  rank-condition expectation, source-manifest gaps, and one known CUDA last-bit assertion. No
  FIT-025 changed-surface test failed; the repository-wide gate is not represented as green.
- The exact CUDA extension ELF was not archived. Runs were not replicated or order-randomized.

The machine-readable audit is `audit.json`. Comparison artifacts hash to
`bd4f5de7bc39019bde6654e7441d56a282cfbd9aed4121669eb3a62a21218ddb`
(`comparison.json`) and
`7752b3c2c1e7e3abe6a82e7b837d2a1875783e8a2552b038ade51e3aae58f792`
(`index.html`).

## Disposition

Accept the physical-capacity/base-active-limit separation and keep the detail-tail mechanism
available as an opt-in experiment. For this Janelle development workflow, prefer a 12,024-row
physical pool with an 11,512-row generic active ceiling and leave the remaining physical headroom
inactive. Do not promote the specialized tail, a nonzero threshold, or a repository default from
this one image.
