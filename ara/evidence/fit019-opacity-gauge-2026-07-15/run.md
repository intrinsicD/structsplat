# FIT-019 opacity-gauge allocation guard

**Verdict:** exact opacity refinement is a real normalized-renderer gauge and row-wise top-k
allocation does not commute with it. Aggregate-first scoring restores canonical group actions, but
the preregistered recovery-utility guard fails. Keep the mechanism benchmark-only; do not add
production quotient or lineage state.

## Audit-corrected protocol

The first frozen execution exposed the intended commutation and negative recovery result. An
independent post-run audit then found that v1 logged but did not gate alpha-0.7 relative/top-k
parity, used set rather than count-aware multiset agreement, and checked only the immediate count.
All omitted checks passed when recomputed, but v1 was not accepted.

V2 corrected only executable gate/provenance fidelity: both-alpha relative/action checks,
both-seed physical-group multiset gating, Stage-A numerical gating, recovered counts, projected
top-k metrics, exact 128-cell validation, fully expanded component configs, CPU/libc identity, and
a 24-file source snapshot. Targets, hashes, seeds, N=32+8, prefit 40, arms, thresholds, actions,
and 20/100 horizons stayed fixed. All 4,352 shared non-timing v1/v2 row comparisons are exact.

## Commutation result

Eight 48x48 procedural target families x seeds `{0,1}` each produce a canonical N=32 field and an
exact gauge view in which every even group becomes two co-located half-opacity rows.

| Check | V2 result |
|---|---:|
| Maximum render absolute error | `8.3446503e-7` |
| Maximum quotient relative error, alpha 0.7 | `2.7007577e-6` |
| Maximum quotient relative error, alpha 1 | `2.7470564e-6` |
| Both-alpha quotient top-8 matches | `16/16` checkpoints |
| Raw alpha-1 multiset changed on both seeds | `8/8` target families |
| Quotient alpha-1 ordered canonical/gauge actions | `16/16` equal |
| Immediate/post-20/post-100 count | all exactly `40` |

The commutation decision is `confirmed=true`. Raw alpha-1 projected multiset Jaccard ranges from
`0.4545` to `0.6000`; quotient projected order/multiset/unique agreement is exactly one.

## Recovery result

Every ordered selection is replayed on the same canonical checkpoint with eight sequential
moment-preserving births. Recovery is an independent fresh Adam restart at each horizon, not
production in-run optimizer-state continuation.

| Arm | Immediate PSNR | Post-20 PSNR | Post-100 PSNR | AUC-100 | Unique groups |
|---|---:|---:|---:|---:|---:|
| `support_canonical` | 24.4108 | 29.6021 | 35.7529 | 32.4442 | 8.00 |
| `responsibility_alpha1_gauge_row` | 24.4238 | 29.3245 | 36.7738 | 32.6394 | 5.38 |
| `quotient_alpha1_gauge` | 24.3191 | 29.5356 | 36.1731 | 32.4894 | 8.00 |

Quotient minus raw gauge-row alpha 1 is `+0.211079 dB` at post-20 but positive on only 5/8 target
families, then `-0.600711 dB` at post-100. Quotient minus canonical support is `-0.066534 dB` at
post-20 and `+0.420178 dB` at post-100. The candidate fails target-family breadth, late retention,
and the post-20 support floor. `utility.survives=false`.

Primary/replay timing overhead is `+1.3778%`/`-1.2604%`. Both pass the 15% accounting bound, but
the sign change precludes a speed claim.

## Response diagnostic

Quotient-minus-raw post-20/post-100 signs reverse in 7/16 cells and 3/8 target means. Median effects
are only `+0.0390/-0.0117 dB`; the Pearson/Spearman horizon association is `-0.593/-0.182` and is
strongly influenced by the sinusoid. This post-hoc result motivates a new dense trajectory test; it
does not establish a general inverse relation between early and late recovery.

## Reproduction and integrity

The primary and replay were run without changing hashed source. Algebra is byte-identical. After
removing only `score/action/intervention/fit20/fit100/total100_seconds`, all 128 rows are exact with
normalized SHA-256
`a6ae37d0423b3014361b40453be039634a0bbf92f69087e6d586958d2f919032`. After additionally removing
the derived timing overhead and timing gate, aggregates are exact with SHA-256
`7a7e6a8c2dd792a092e64364e1dd4ae2441cb1a795524f2666e5193357bcb4ed`.

The source snapshot includes the executable benchmark, helpers, preregistration/task, tests,
`pyproject.toml`, and every top-level StructSplat Python module. Combined source SHA-256 is
`89f52281e5596e7225cf278be74eeaabc423c54db2f176ecf4d5bfa5d2b99f23`.

Residual caveat: wheel/library binaries and a detailed CPU model are not snapshotted, so exact
cross-host binary reconstruction is not guaranteed. The same-source same-host replay is exact for
every deterministic payload.
