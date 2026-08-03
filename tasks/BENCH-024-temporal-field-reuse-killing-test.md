# BENCH-024 — Temporal field-reuse killing test

## Context

Adjacent frames from one fixed camera may permit geometry warm starts or a compact delta stream,
which could reduce both fitting work and sequence bytes. Cross-camera mixing is semantically unsafe
for this first test, and building a temporal codec before measuring correspondence stability would
be expensive. A bounded oracle/killing test should determine whether the opportunity exists.

## Goal

Measure the upper bound and practical first-order value of same-camera temporal warm starts,
shared geometry, and delta coding under complete sequence-byte and latency accounting.

## Non-goals

- Changing the Field V2 bitstream, implementing a production temporal codec, or sharing rows
  across different cameras.
- Random-access-hostile sequence compression without explicitly measuring seek/keyframe costs.
- Selecting favorable motion clips after outcomes are observed.

## Protocol requirements

- Freeze contiguous same-camera sequences spanning low, moderate, and difficult motion/occlusion;
  keyframe intervals; independent, warm-start, shared-geometry-oracle, and delta-entropy arms;
  complete byte/time accounting; recovery rules; and quality/downstream guardrails.
- Keep CORE-014 semantics, codec, and per-frame byte target unchanged except for explicitly counted
  temporal side information. Evaluate drift, scene cuts, topology churn, and seek-from-keyframe.
- Report both an oracle with future correspondence information and a deployable causal arm; never
  present the oracle as achieved performance.

## Acceptance criteria

- [ ] Sequence/camera identities, split policy, motion strata, keyframes, arms, budgets, seeds,
      metrics, and killing thresholds are frozen in a reviewed protocol before execution.
- [ ] Report per-frame and total sequence complete bytes, fit/end-to-end latency, PSNR/MS-SSIM/
      LPIPS, BENCH-019 downstream response, row identity/churn, drift, failures, cold random-seek
      latency, memory, and recovery after cuts/occlusion.
- [ ] Cross-camera reuse is absent from code and protocol; any exploratory cross-camera oracle is
      separately labelled and cannot support the primary decision.
- [ ] The decision either kills temporal work, authorizes a separate codec/method task with a
      quantified opportunity and boundary, or records the result unavailable. This task does not
      mutate COMP-013's stream grammar.
- [ ] Portable report, independent audit, ARA disposition, docs/task synchronization, and
      `./scripts/verify.sh` pass.

## Interfaces touched

Bounded sequence experiment driver, CORE-014/COMP-013 adapters, report/audit tooling,
`ara/evidence/`, `docs/additive_field_v2.md`, this task, and the Index.

## Depends on

CORE-014, COMP-013/014, BENCH-022/025, BENCH-002

## Agent workflow

- Driver: pending
- Reviewer: pending
- Turn: driver
- Reviewed revision: pending

### Handoff log

Append exact `### Handoff`, `### Review`, and pre-run `### Protocol review` blocks using
`tasks/README.md`.

## Notes

This is intentionally off the critical path. A positive screen creates a separately reviewed
temporal design task; it does not retroactively expand the production codec.
