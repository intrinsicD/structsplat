# FIT-030 — Byte-priced topology and precision allocation

## Context

The original design correctly separated coverage, detail, and rate, but it priced rate with a
future `D + lambda R` concept while depending on normalized transactional-schedule diagnoses.
Field V2 and COMP-013 provide a cleaner boundary: topology and precision proposals can be screened
with cheap marginal estimates, then replayed through one complete codec so rows, attributes,
indexes, alpha, and headers are priced in actual bytes. FIT-045 supplies the regional action
control; FIT-046 supplies the conditional appearance solve.

## Goal

A default-off controller that reaches a requested complete-byte target by choosing among bounded
topology and per-attribute precision actions according to measured marginal distortion reduction
per byte, with exact codec replay and matched fixed-N/fixed-precision controls.

## Action grammar

- `noop` / stop;
- birth or split in an eligible region;
- merge or prune where coverage/quality feasibility remains satisfied;
- raise or lower a declared quantization tier for a spatial group or attribute group;
- re-solve affected additive coefficients after a topology action.

The initial grammar is finite and versioned. It does not permit mixed primitive families, temporal
dependencies, learned code generators, or arbitrary per-row bit widths.

## Objective and controller

Use an estimated `Delta D / Delta R` only to rank proposals. At every accepted batch, encode the
complete candidate with COMP-013, measure exact `R`, decode it cold, and recompute full distortion
and feasibility. Optimize toward a target-byte constraint (with a `D + lambda R` sweep retained as
a diagnostic), restore the best feasible cold-decoded checkpoint, and stop when no admissible
action clears the frozen marginal-gain/work rule or the hard budget expires.

## Non-goals

- Selecting field semantics, regional allocation, or codec grammar inside this task.
- Treating row count, analytical bits, entropy estimates, or tensor-body bytes as actual rate.
- Replacing alpha with learned occupancy or using per-block global rollback as the controller.
- Promoting a default before CORE-014 and BENCH-022.

## Acceptance criteria

- [ ] The action grammar, feasibility predicates, proposal estimator, tie order, byte-target
      tolerance, stopping rule, checkpoint policy, and exact-replay cadence are deterministic and
      frozen in configuration.
- [ ] Every accepted decision has a replay ledger containing parent hash, action, estimated
      `Delta D/Delta R`, complete encoded bytes before/after, cold-decoded metrics, work/time, and
      accept/reject reason; ledgers replay from source/config/seed.
- [ ] Unit/property tests cover exact budget boundaries, non-monotone codec bytes, zero/negative
      gain, overshoot recovery, ties, merge/birth reversals, precision changes, corruption/failure,
      masks, and best-feasible restoration.
- [ ] A preregistered screen compares byte-priced joint actions, topology-only, precision-only,
      fixed-N scalar-QAT, and the strongest fixed-byte control at matched complete byte targets,
      work limits, data, and seeds.
- [ ] Report PSNR/MS-SSIM/LPIPS, BENCH-019 downstream objective, complete bytes/bpp, target error,
      action mix, estimator calibration, encode/decode overhead, convergence/AUC, total wall time,
      and memory. Estimated and exact rate are never pooled.
- [ ] The selected controller improves the predeclared complete-byte frontier or is retired; one
      exact config digest and fallback is passed to CORE-014 only after an independent audit.
- [ ] Focused tests, portable report, results audit, ARA disposition, docs/task synchronization,
      and `./scripts/verify.sh` pass.

## Interfaces touched

New bounded rate-controller module and fit hooks, COMP-013 encode/decode/rate API, topology and
quantization configuration, replay ledger/report tooling, tests, `docs/additive_field_v2.md`, this
task, and the Index.

## Depends on

BENCH-021/025, COMP-013/014, FIT-045/046, BENCH-002

## Agent workflow

- Driver: pending
- Reviewer: pending
- Turn: driver
- Reviewed revision: pending

### Handoff log

Append exact `### Handoff`, `### Review`, and pre-run `### Protocol review` blocks using
`tasks/README.md`. The codec/config digest is part of the frozen protocol.

## Notes

The normalized safe-schedule findings in O87--O89 remain historical motivation, not prerequisites
or evidence that this controller will win. BENCH-017/FIT-028/029/BENCH-018 continue independently
for the maintained normalized pipeline.
If BENCH-025 rejects structured coding, COMP-014's terminal no-code disposition selects COMP-013;
if it authorizes COMP-014, the marginal-rate oracle must use that completed stream instead.
