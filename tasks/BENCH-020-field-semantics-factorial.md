# BENCH-020 — Field semantics and alpha-policy factorial

## Context

The live Janelle diagnostic changes compositor, parameterization, initialization, containment,
mask loss, topology schedule, and commit policy simultaneously. Its custom no-boundary arm also
retains outside-regression vetoes while zero-weighting outside pixels, so it is not a clean
boundary control. BENCH-019 and CORE-013 make it possible to isolate the representation itself.

## Goal

Select the Field V2 semantic candidate and alpha policy through a matched development screen and
sealed confirmation before any production optimizer or codec is built around it.

## Candidate arms

- incumbent native additive `weight * color` field;
- direct additive RGB coefficient field with no structural mass;
- additive RGB coefficient plus independently supervised structural mass, if BENCH-019 identifies
  a valid target;
- normalized plain fit without the transactional schedule;
- current normalized maintained pipeline;
- alpha-gated and hard-contained boundary policies wherever the field semantics support both.

Within the additive candidates, run a preregistered fixed-geometry elimination for coefficient/DC
semantics: zero-DC nonnegative coefficients versus a counted alpha-gated DC term plus bounded/signed
residual coefficients. CORE-009 is the background control. Advance at most the frozen finalists to
the full semantic/alpha factorial; do not infer the domain from whichever solver happens to run.

Run a fixed-row lane and an equal-canonical-raw-byte lane. Native-authentic external methods remain
separate from the matched semantic arms.

## Non-goals

- Searching losses, topology policies, learned initializers, quantizers, or CUDA optimizations.
- Repairing an arm after outcomes are visible or using the live frame_00008 diagnostic as
  confirmation data.
- Promoting a default; a win selects the contract used by later tasks.

## Acceptance criteria

- [ ] Before execution, freeze a metadata-selected development set, disjoint sealed confirmation
      set, at least three seeds, source/prepared hashes, exact arms, initial geometry, row/raw-byte
      lanes, iteration and wall-time budgets, checkpoint rule, metrics, downstream protocol, and
      missing/error policy.
- [ ] A/A tests prove identical pixels, masks, initial geometry, seed streams, and requested work
      across matched arms. The no-boundary control has a loss/gate/profile contract consistent with
      its declared policy.
- [ ] Coefficient domain, DC/background bytes, alpha matting, authoritative pre-clamp render, and
      evaluation clipping are explicit; fixed-geometry elimination and advancement replay from raw
      rows under a frozen rule.
- [ ] Report foreground/boundary PSNR, MS-SSIM, LPIPS, alpha/outside checks, Stage-1 objective from
      BENCH-019, downstream responses, rows, canonical raw bytes, time-to-target, PSNR-time AUC,
      wall time, and peak memory.
- [ ] Killing screen: stop if neither additive candidate is image-quality noninferior and
      downstream-favorable against both the incumbent additive and normalized plain controls under
      a matched lane. Exact margins and statistical rules are frozen before target access.
- [ ] Confirmation evaluates one sealed semantic/alpha choice once. A heterogeneous trade-off is
      recorded as such; no hidden scalar score decides it.
- [ ] The result produces a reviewed ADR that selects direct additive, dual additive, normalized,
      or no new contract. ADR-0003/0006 history is preserved.
- [ ] Portable report, independent results audit, ARA disposition, docs/task synchronization, and
      `./scripts/verify.sh` pass.

## Interfaces touched

Bounded driver under `scripts/experiments/`, CORE-013 adapters, maintained report telemetry/tests,
`docs/adr/`, `docs/additive_field_v2.md`, `ara/evidence/`, this task, and the Index.

## Depends on

BENCH-019, CORE-013, BENCH-002, ADR-0003/0006

## Agent workflow

- Driver: pending
- Reviewer: pending
- Turn: driver
- Reviewed revision: pending

### Handoff log

Append exact `### Handoff`, `### Review`, and pre-run `### Protocol review` blocks using
`tasks/README.md`.

## Notes

This task is the semantic gate. A negative or unavailable outcome keeps the current production
default and blocks CORE-014/015; it does not invite a post-hoc loss sweep.
