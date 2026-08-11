# HIER-024 gauge-geometry appearance-projection diagnostic

## Scope and receipts

This is consumed development evidence on four mechanically selected, historically used DIV2K
images, two seeds, N=640, max-side 160, 500 attempted fit steps, and one RTX 4090. It is
dirty-source and producer-reviewed, not held-out confirmation or a default, semantic, codec,
novelty, or general representation-limit result.

The first output without a suffix is an immutable invalid harness run. All fits, projections,
selection metrics, and artifacts executed, but the reused HIER-023 row writer indexed its own
four-name selection map before HIER-024 could replace the metadata, so all 40 final row emissions
failed. It is excluded from method evidence. The repaired bundle is
`results/hier024_div2k4_s160_n640_i500_s01_diagnostic_rerun1_2026-08-11`.

- manifest: `f761b2834aa394d5a6c3af5648460ac0b67ea108d252d6ce03a487f06f738b9a`
- metrics: `14a3d40938dfcf2e054bd6ced3e20021dcddacc0158976a95ba3ab6993a6b074`
- decision: `ffecaecb5cf60fb79840d2ac55d85a8e3a6fdea6f33f8da914a36533a56467c1`
- report checker: passes with `--allow-dirty`

## Frozen causal test

Ordinary additive and HIER-023 no-reset unit-gauge endpoints each receive the same existing
matrix-free all-row direct-additive RGB PCG solve: tolerance `1e-6`, at most 48 iterations, ridge
`1e-8`, coefficient limit 16, input-centered start/regularization, explicit frozen base, and no
stage-zero reconditioning. A second target-known transaction selects a proposal only when raw MSE
strictly improves and MS-SSIM, LPIPS, pixel maximum, and complete 7x7 maximum satisfy their frozen
safety clauses. Geometry and count cannot change, and the result is an ordinary additive
`GaussianField` with no opacity, mass, denominator, optimizer, target, or auxiliary RGB payload.

## Aggregate results

| arm | PSNR | MS-SSIM | LPIPS | pixel max | 7x7 max | PSNR AUC | fit s | projection s | selected |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| normalized plain | 28.4194 | 0.978867 | 0.109896 | 0.403918 | 0.128780 | 27.1637 | 0.846 | 0.000 | 0/8 |
| additive plain | 27.7412 | 0.977301 | 0.160934 | 0.358699 | 0.131278 | 25.6468 | 0.795 | 0.000 | 0/8 |
| additive projected safe | 27.8711 | 0.977625 | 0.159631 | 0.345144 | 0.128037 | 25.6468 | 0.795 | 0.181 | 7/8 |
| gauge no reset | 27.7097 | 0.977462 | 0.147372 | 0.378696 | 0.138540 | 26.6612 | 1.207 | 0.000 | 0/8 |
| gauge projected safe | 27.8816 | 0.977945 | 0.145454 | 0.347652 | 0.131001 | 26.6612 | 1.207 | 0.179 | 7/8 |

The solve gains `0.12996 dB` on ordinary-additive geometry and `0.17191 dB` on gauge geometry.
The difference is only `0.04195 dB`, below the frozen `0.05 dB` causal threshold. The candidate
therefore ends just `0.01046 dB` above projected additive rather than the required `0.10 dB`, and
closes only 1.91% of the positive `0.54827 dB` normalized/projected-additive gap rather than half.
It improves mean MS-SSIM by `0.000320` and LPIPS by `0.014177`, but worsens mean pixel and 7x7
maxima by `0.002508/0.002964`. Per-cell LPIPS and local guards also fail. The hold deviation reaches
`0.06724 dB` in one direction and `0.05220 dB` in the other on two cells, narrowly failing its
`0.05 dB` guard.

## Transaction, integrity, and visual audit

Each projected arm selects seven proposals. The ordinary `0800` seed-1 proposal rolls back on the
7x7 clause; gauge `0571` seed 1 rolls back on LPIPS. Both returned fields exactly match their
incoming digests. All 16 projected rows preserve geometry and count. Maximum persisted coefficient
is `1.74095` for projected additive and `2.58119` for projected gauge; maximum maintained-render
parity is `1.70e-6/3.19e-6`, well inside the `2e-5` limit. No unsafe or failed proposal is hidden.

Native full-frame and worst-crop review finds no new lattice, checker, ringing, hole, or wash. All
arms share the expected N=640 loss of fine detail. The largest gauge local regression (`0571`,
seed 1) is a broad misplaced color lobe and scene-content displacement, while other seeds reverse
which geometry is better. That seed-sensitive support error is consistent with a basis-placement
limit, not an appearance coefficient left unsolved.

## Disposition

The frozen mechanism gate fails. Fixed-geometry coefficient optimization is not the missing
mechanism behind normalized rendering's fixed-count advantage: both bases benefit, nearly equally,
and neither closes the gap. Keep the projection wrapper and unit gauge default-off as diagnostic
infrastructure; do not retune this consumed bank. The next admissible pure-additive experiment must
change basis geometry or topology under a new task, output, and mechanically bound data selection.
A counted broad low-frequency Gaussian layer plus residual/detail Gaussians is the cheapest next
discriminator because it changes the additive span while retaining one denominator-free Gaussian
sum and exact N.

## Limits

Historically consumed images, one device, dirty sources, producer review, small resolution/count,
and target-known projection prevent confirmation, deployment, speed, rate, or novelty claims. The
result rejects the tested fixed-basis explanation; it does not prove normalization mathematically
necessary or rule out a better pure-additive Gaussian basis.
