# BENCH-022 — Additive production confirmation

## Context

Component screens can overstate an end-to-end improvement because setup, rejected work, complete
bytes, codec distortion, cold decode, and realtime-gs response interact. CORE-014 must therefore
face one sealed production-shaped confirmation against the incumbent native-additive and current
normalized pipelines before any default change.

## Goal

An independently audited go/no-go decision for the integrated Observation Field V2 pipeline on
quality, downstream utility, convergence, complete rate, latency, memory, and robustness.

## Non-goals

- Tuning the candidate or baselines on confirmation frames.
- Hiding a failed endpoint behind a favorable intermediate checkpoint or composite score.
- Changing defaults; a positive decision only authorizes CORE-015.

## Protocol requirements

- Freeze public and Janelle development/confirmation splits, source/prepared hashes, masks, camera
  records, pipeline/config digests, commits/environments, seeds, complete-byte targets, time/work
  caps, checkpoints, metrics, noninferiority/superiority margins, and failure policy.
- Compare integrated Field V2 against current maintained normalized and incumbent native additive
  pipelines. Include matched row/raw-byte diagnostics where needed, but make complete-byte and
  end-to-end-time lanes primary.
- Run the BENCH-019 fixed downstream protocol without per-field tuning and preserve frame/capture
  as the unit of inference.

## Acceptance criteria

- [ ] A distinct reviewer approves the prospective protocol digest before any result-bearing run;
      confirmation data remains sealed until the final candidate digest is frozen.
- [ ] Report complete bytes/bpp and RD curves, PSNR/MS-SSIM/LPIPS, boundary/alpha/coverage/query
      diagnostics, downstream responses, convergence/time-to-target, renderer/sampled work,
      end-to-end wall time, encode/cold-decode/query time, peak/resident memory, and failures.
- [ ] Timing includes preparation, method-specific setup, rejected work, evaluations, checkpoints,
      serialization, decode, and synchronization. Warm and cold results are not mixed.
- [ ] The go/no-go rule evaluates all predeclared quality, downstream, rate, speed, memory, and
      reliability guardrails; heterogeneous trade-offs are reported rather than collapsed after
      outcome access.
- [ ] A negative result names the failed boundary and closes or re-scopes CORE-015. A positive
      result binds the exact profile/codec/config digest that CORE-015 may promote.
- [ ] Portable report, raw tables, representative visuals, independent results audit, ARA
      disposition, docs/task synchronization, and `./scripts/verify.sh` pass.

## Interfaces touched

Maintained benchmark/external-run adapters, report and audit tooling, `ara/evidence/`, ADR/docs,
`docs/additive_field_v2.md`, this task, CORE-015, and the Index.

## Depends on

CORE-014, BENCH-019/020/021/025, COMP-013/014, PORT-006, BENCH-002

## Agent workflow

- Driver: pending
- Reviewer: pending
- Turn: driver
- Reviewed revision: pending

### Handoff log

Append exact `### Handoff`, `### Review`, and pre-run `### Protocol review` blocks using
`tasks/README.md`. Protocol and results reviewers must be distinct from the Driver.

## Notes

The architecture document proposes targets, but this task must freeze exact margins from pilot
variance and product constraints before opening the confirmation data.
