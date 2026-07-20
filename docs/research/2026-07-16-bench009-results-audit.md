# BENCH-009 Stage-1 results audit

**Audit date:** 2026-07-16
**Experiment:** rate/DOF-priced residual tangent-space auction
**Verdict:** ledger-complete and internally exact; frozen causal-validity gate failed; no method,
expressiveness, performance, convergence-speed, or compression claim is authorized.

## Answer first

BENCH-009 did not identify a richer StructSplat primitive. Its complete Stage-1 development and
recovery ledgers are numerically intact, but the causal assay is unavailable because the retained
base and independently truncated joint projector spaces are not guaranteed to be nested. Exact
saved rows contain negative values for a quantity defined as incremental projector energy.

This does not show that affine appearance or localized carriers are universally useless. It shows
that the frozen v3 causal instrument cannot decide whether they add capacity outside the current
grammar. Independently, both candidates fail the v3 utility screen: immediate and checkpoint-20
held-out effects are negative on average, and no candidate survives the frozen two-radius,
two-horizon rule.

## Independently verified artifact facts

- `24` source-bound parents from `12` continuous trajectories are present and hash-exact.
- Immediate science has `4,608` unique terminal `ok` rows: `3,072` cross-fit and `1,536` excluded
  all-pixel-oracle rows.
- The auxiliary ledgers contain `1,152` diagnostic scores, `2,304` immediate matched-evidence
  rows, `288` carrier-bank sensitivity rows, `72` selections, and `72` completed scope units.
- Recovery has `816` sources, `816` terminal trajectories, and `2,448` logical checkpoints:
  exactly `816` each at steps `0`, `20`, and `100`, including `1,632` new recovery renders.
- All `768` action step-0 joins and `48` no-action joins reproduce the saved state, packet,
  render, support, and metric hashes.
- The combined immediate/recovery matched-evidence union has `3,072` rows and SHA-256
  `22d49ee2...cf7c8`.
- The canonical causal audit reproduces byte-identically with SHA-256
  `b209cc57a865ce0c9cd28a9a9fd65a3d25ac0d86b0a6c2d03047b0842025ec72` and expected exit code
  `2` (`status=unavailable`).
- `65` focused BENCH-009 tests passed in the independent referee run.

The audit's `23` errors should not be read as missing or corrupt raw rows. They are `21` invalid
projector values plus two fail-closed matched-analyzer propagation errors. Binding, manifest,
shard-union, and step-0 integrity checks passed.

## Severity-ranked scientific findings

### 1. Frozen causal calibration fails

Across all `3,072` cross-fit immediate rows, `1,628` exceed the preregistered prediction floor.
Spearman predicted-versus-realized gain is `0.268549`, below the required `0.8`; median
realized/predicted gain is `0.87667`. Every one of the eight causal action-by-parent-horizon
strata fails. The full-`J` strata have correlations `0.1916` and `-0.1611`; the six joint causal
strata range from `-0.1372` to `0.1163`.

The matched six-DOF actions calibrate well—affine and carrier are approximately `0.999`, births
are at least `0.9858`, and tangent-six is at least `0.8568`—but that does not validate the failed
full-`J` causal actions.

### 2. Literal incremental projector energy is invalid

At the two frozen relative rank thresholds, cross-fit near-plateau rows contain:

| Candidate | `rcond` | Negative units / 24 | Minimum increment |
|---|---:|---:|---:|
| affine | `1e-5` | `4` | `-5.22e-8` |
| affine | `1e-4` | `4` | `-2.31e-7` |
| carrier | `1e-5` | `3` | `-1.49e-7` |
| carrier | `1e-4` | `4` | `-4.73e-7` |

The implementation first residualizes `A` against the retained base basis, then independently
scales and truncates `[J,A]` relative to the joint maximum singular value. Consequently the
retained base and joint subspaces need not be nested. Float32 subtraction may amplify the symptom,
but positive aggregate fractions cannot rescue a statistic that violates its row-level invariant.
Neither “outside `J`” nor “inside `J`” is established.

### 3. Both richer candidates fail the frozen v3 utility screen

Against each unit's stronger tangent-six/two-birth control, mean held-out PSNR differences are:

| Candidate | Radius | Immediate | Step 20 | Step 100 |
|---|---:|---:|---:|---:|
| affine | `0.25` | `-0.631 dB` | `-0.752 dB` | `+0.002 dB` |
| affine | `0.75` | `-0.779 dB` | `-0.644 dB` | `+0.059 dB` |
| carrier | `0.25` | `-0.184 dB` | `-0.545 dB` | `+0.413 dB` |
| carrier | `0.75` | `-0.315 dB` | `-0.439 dB` | `+0.372 dB` |

Carrier's late recovery is descriptive only. At radius `0.25`, its step-100 bootstrap lower bound
is positive (`+0.066 dB`), but the `0.75` lower bound is negative and both immediate/step-20 gates
fail. The matched survival ratio itself is undefined because the stronger-control predicted-energy
totals are nonpositive (`-0.2708` and `-0.4440`). This is a three-valued reporting limitation, not
missing recovery evidence; the necessary utility conjuncts already fail.

### 4. Objective mismatch blocks an optimizer-deficit claim

Parents were trained with `0.7 L1 + 0.3 (1-SSIM)`, while the causal assay and recovery optimize
RGB MSE. Later MSE recovery therefore does not prove that the original optimizer failed its own
objective. Full-`J` immediate actions also lose against the parent on average. Report only
descriptive RGB-MSE recovery under the changed objective.

### 5. Provenance and resource accounting are not archival-grade

- Saved runner/science/recovery commands serialize `-m __main__`, so they are not directly
  rerunnable.
- The canonical recovery directory preserves preparation and a final zero-work invocation; the
  eight work-producing commands remain in shard directories and are not bound into the canonical
  config.
- The causal audit has no self-binding over auditor/input bytes.
- The preflight pass and toy full-path pass exist, but the latter is not cryptographically linked
  to the development config.
- Candidate search and matrix-free range/factorization work occur outside the per-action timer.
  The saved `1519.13 s` action total is therefore not end-to-end runtime.

Before synchronizing the stale task status, the current executed source bytes were preserved as
`results/bench009_tangent_auction_stage1_causal_audit_v3/executed_sources_v3.tar`, SHA-256
`20b3b08ec57f286e25249b76d927d525459bfdd62e6baf91899bc94fdceec4f3` (`24` files). This repairs
the missing byte snapshot for the present v3 workspace but does not retroactively bind it into the
original experiment configs.

## Claim disposition by requested axis

| Axis | Permissible conclusion |
|---|---|
| Quality | No promoted improvement. On the spent 64x64 development screen, both richer candidates fail the frozen utility decision. |
| Convergence | No speed/default claim. Carrier's positive step-100 mean is late, radius-sensitive descriptive evidence after fresh-Adam MSE recovery. |
| Performance | Unavailable. Search/factorization cost is omitted from action timing and no end-to-end implementation was measured. |
| Compression | Unavailable. Six coefficients and provisional selector bytes are not a complete cold-decodable stream or actual rate. |
| Expressiveness | Unavailable. The non-nested projector invalidates both outside-`J` and inside-`J` classifications. |
| Optimization | Unavailable as a causal claim because parent and assay objectives differ and full-`J` calibration fails. |
| Robustness | Matched-grid calibration and winner stability pass narrowly; causal calibration and the governing scientific decision fail. |
| Production | No default, grammar, allocator, CUDA, or codec change is authorized. |

## Evidence-selected follow-up

Do not consume disjoint targets yet. BENCH-011 preregisters a spent-data-only diagnostic of the
minimal algebraic repair:

1. keep the existing selected affine/carrier identities;
2. freeze one base basis `Q_J`;
3. residualize the physical six-column design directly in float64;
4. retain extension singular directions relative to the original unscaled `A`;
5. compute the incremental energy as the square norm in the nested extension basis; and
6. fit physical coefficients on discovery rows and test calibration on `96` native held-out
   renders.

If any of four candidate-by-radius calibration strata fails, close the formulation without
retuning. Only an all-strata pass may authorize preregistering a disjoint, objective-aligned assay;
even that would not promote a method.

## Reproduction

```bash
PYTHONPATH=src:. python -m benchmarks.residual_tangent_auction_stage1_causal_audit \
  --runner-dir results/bench009_tangent_auction_stage1_runner_v3 \
  --science-dir results/bench009_tangent_auction_stage1_science_v3 \
  --recovery-dir results/bench009_tangent_auction_stage1_recovery_v3 \
  --output /tmp/bench009-causal-audit-rerun.json
```

Expected exit code: `2`. The rerun must be byte-identical to the canonical audit above.
