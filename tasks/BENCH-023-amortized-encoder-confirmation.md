# BENCH-023 — Amortized encoder confirmation

## Context

FF-002/003 can potentially replace most per-image optimization with a permutation-invariant
predictor plus short refinement, but earlier learned plans predate the selected Field V2 semantics,
codec, downstream objective, and actual-byte accounting. The learned lane must be rebased and
tested as an optional latency tier rather than assumed to inherit the iterative pipeline's quality.

## Goal

Determine whether the rebased predictor delivers a practically useful quality-rate-latency
frontier and whether its up-front training cost amortizes for the intended capture distribution.

## Non-goals

- Making the predictor required for Field V2 or changing CORE-014/015 defaults in this task.
- Random image splits that leak adjacent frames/cameras, or excluding training/data-preparation
  cost from the break-even analysis.
- Comparing variable-byte learned outputs against a fixed-row iterative baseline.

## Protocol requirements

- Freeze whole-frame/capture train/development/confirmation groups, camera policy, preprocessing,
  teacher provenance, Field V2/codec versions, byte targets, seeds, hardware, training budget,
  inference/refinement budgets, metrics, and missing/failure policy.
- Compare direct inference and `0/50/200/500` refinement steps against CORE-014 at equal complete
  bytes and matched quality targets; include simple warm-start controls.
- Count training, teacher generation, checkpoint selection, inference, refinement, encode/decode,
  resident model memory, and per-image memory. Report break-even image counts for declared reuse
  scenarios rather than one universal amortization claim.

## Acceptance criteria

- [ ] The split audit demonstrates no adjacent-frame/camera/capture leakage and labels any
      Janelle-only result as workload-specific.
- [ ] Report PSNR/MS-SSIM/LPIPS, BENCH-019 downstream response, complete bytes/bpp, inference and
      end-to-end latency, refinement convergence, peak/resident memory, failures, and training
      cost/energy proxy where available.
- [ ] Compare `0/50/200/500` steps and CORE-014 using the same cold/warm definitions and codec;
      visual failures and long-tail latency are included.
- [ ] A predeclared gate decides whether the predictor is a supported fast tier, remains research,
      or is killed. Any headline speedup must satisfy frozen quality/downstream/rate guardrails.
- [ ] Break-even analysis exposes dataset size, retraining frequency, hardware utilization, and
      uncertainty; unamortized and amortized numbers are both visible.
- [ ] Portable report, independent results audit, ARA disposition, FF task/docs synchronization,
      and `./scripts/verify.sh` pass.

## Interfaces touched

FF-002/003 model/training/evaluation code, Field V2/codec adapters, maintained benchmark/report
tooling, `ara/evidence/`, `docs/additive_field_v2.md`, this task, and the Index.

## Depends on

FF-002/003, BENCH-022/025, COMP-013/014, BENCH-002

## Agent workflow

- Driver: pending
- Reviewer: pending
- Turn: driver
- Reviewed revision: pending

### Handoff log

Append exact `### Handoff`, `### Review`, and pre-run `### Protocol review` blocks using
`tasks/README.md`.

## Notes

A strong learned result remains an optional latency tier until independently confirmed across a
capture distribution broad enough for the intended product claim.
