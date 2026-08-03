# PORT-006 — Additive end-to-end acceleration

## Context

The present conversion wall time combines algorithmic work (many full-resolution optimization
passes and retries) with implementation overhead (Python orchestration, materialized images,
separate loss/backward/update kernels, synchronization, and allocation churn). BENCH-021 can only
select work-saving mechanisms under reference accounting; it cannot establish production latency
until the frozen winning recipe has a production-shaped implementation and explicit parity against
the reference equation.

## Goal

An accelerated additive training path for the BENCH-020 semantics and BENCH-021-selected recipe,
including any selected FIT-046/047 operations, with end-to-end wall-time evidence,
numerical/gradient parity, bounded memory, and no silent change to the optimized objective.

## Non-goals

- Using lower precision, fewer evaluated pixels, or altered support as an unlabelled speedup.
- Optimizing rejected semantic/fit candidates or rewriting the whole repository around CUDA.
- Reporting kernel microbenchmarks as conversion speedups.

## Optimization scope

- Profile preparation, initialization, tile binning, raster forward/backward, loss, optimizer,
  topology updates, checkpoint/evaluation, serialization, synchronization, and host/device copies.
- Extend the exact additive CUDA path with the selected training/backward features. Consider fused
  MSE/adjoint/optimizer operations, persistent/reused buffers, captured graphs, deterministic tile
  lists, and FIT-047 sampling only where the profiler identifies material cost.
- Preserve a small CPU/reference oracle and explicit slow fallback for unsupported shapes/devices.

## Acceptance criteria

- [ ] A committed profiler artifact attributes representative end-to-end time and peak memory by
      stage before optimization; preparation and serialization are not excluded.
- [ ] Forward values, loss, parameter gradients, optimizer steps, masks/alpha, and edge cases match
      reference fixtures within frozen dtype-specific tolerances; finite-difference checks cover
      position, scale, rotation, and appearance.
- [ ] Deterministic mode replays within its declared contract and fast mode is explicitly labelled;
      failures never return a partially updated field as successful.
- [ ] A benchmark reports kernel and end-to-end time, setup/amortization, renderer work, memory,
      target GPU/software environment, and final quality/downstream parity against the same recipe.
- [ ] Production-promotion gate: at least `1.5x` representative end-to-end speedup with no frozen
      quality/downstream regression; a `3x` result is a research target, not an acceptance promise.
- [ ] Import boundaries, CPU portability, focused performance/parity tests, report/audit, ARA
      disposition, docs/task synchronization, and `./scripts/verify.sh` pass.

## Interfaces touched

CUDA extension and additive raster/backward bindings, fit/loss/optimizer orchestration, profiling
and benchmark scripts, CPU oracle/tests, build configuration, `docs/additive_field_v2.md`, this
task, and the Index.

## Depends on

BENCH-020/021, FIT-046/047, ADR-0011/0024

## Agent workflow

- Driver: pending
- Reviewer: pending
- Turn: driver
- Reviewed revision: pending

### Handoff log

Append exact `### Handoff`, `### Review`, and pre-run `### Protocol review` blocks using
`tasks/README.md`. A differentiable-graphics review under `structsplat-review` is mandatory.

## Notes

If algorithmic work still dominates after the implementation profile, the evidence should direct
effort back to BENCH-021/FIT tasks rather than accumulating low-impact kernels.
PORT-001/002/003/005 provide reusable implementation and profiling infrastructure, but their
unrelated open deliverables do not block this bounded additive path.
