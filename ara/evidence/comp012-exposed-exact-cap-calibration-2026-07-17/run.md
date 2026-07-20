# COMP-012 exposed exact-cap calibration

**Date:** 2026-07-17 through 2026-07-18
**Scope:** already-exposed COMP-011 development cells only; unsealed mechanics calibration
**Verdict:** promising exact-byte signal plus a perceptual counterexample and three rejected
safeguards; `NO_GO_PRE_DATA`

## Visual comparison

The focused [visual report](index.html) contains only the target, frozen SSP2E baseline, and
equal-byte SSP2F fitted image for the two exposed 12-bit cells. Its tables report complete bytes,
bpp, PSNR, MS-SSIM, and LPIPS; the targets are unscored references. The six lossless RGB PNGs and
the exact fresh-render metric/provenance snapshot are under [`visual/`](visual/). These fresh
single-render values support the displayed images, while the report labels the historical
three-repeat conservative deltas separately.

## What was tested

The probe held geometry, row order, header/range values, non-RGB symbols, the SSP2F decoder, and
the complete-stream cap fixed. It changed only absolute RGB symbols. Every priced edit used the
exact empirical-frequency normalization, order-sensitive arithmetic size, whole-model zlib-9
size, and complete SSP2F byte formula. Sweep checkpoints were reconciled against a fresh CPU
render and an ordinary SSP2F encode.

The cap was the strongest exact incumbent's complete SSP2E size, so the candidate and incumbent
have equal complete bytes but different wire formats. The CPU search objective was clamped target
pixel SSE. Persisted quality used the frozen COMP-011 v2r36 CUDA renderer and metric environment.
The recorded source/target inputs are only the named already-exposed COMP-011 development cells.
The unsealed harness records `confirmation_accessed=false` and contains no COMP-012 data path, but
it is not an access-control proof.

## Equal-byte persisted-CUDA results

The conservative comparison is minimum candidate PSNR/MS-SSIM minus maximum incumbent value, and
maximum candidate LPIPS minus minimum incumbent value. Thus a positive PSNR/MS-SSIM value and a
negative LPIPS value are favorable.

| Exposed cell | Bytes candidate/incumbent | Search state | PSNR delta | MS-SSIM delta | LPIPS delta |
|---|---:|---|---:|---:|---:|
| `jason-briscoe-149782/[12,6,6,8]` | 51,549 / 51,549 | 8 coarse + 10 fine; 15 final-sweep accepts | +0.931405480 dB | +0.000196278 | -0.000448857 |
| `nomao-saeki-33553/[12,6,6,8]` | 52,223 / 52,223 | 8 coarse + 10 fine; 103 final-sweep accepts | +0.860618088 dB | +0.000171006 | -0.002742201 |
| `jason-briscoe-149782/[16,8,8,8]`, target MSE | 64,662 / 64,662 | fine-sweep local zero after 8 coarse + 17 fine | +0.461592326 dB | -0.000652313 | +0.000356536 |
| `jason-briscoe-149782/[16,8,8,8]`, target MSE + 0.25 source-render MSE | 64,662 / 64,662 | fine-sweep local zero after 8 coarse + 10 fine | +0.392951072 dB | -0.000737071 | +0.000623705 |

The first two rows have three fresh candidate repetitions and three frozen incumbent repetitions.
The unregularized 16-bit candidate has only one fresh final-candidate repetition, so its deltas are
descriptive conservative comparisons against the three incumbent repetitions, not a complete
repeat schedule. The regularized 16-bit candidate has three fresh repetitions. Deltas and raw rows
are rounded for display; no candidate metric JSONL was preserved, so last-digit precision is not an
execution-sealed claim.

Here, “fine-sweep local zero” means zero accepted moves in one full sweep over the in-range
`{-3,-1,+1,+3}` symbol offsets, with required objective improvement greater than
`1e-12*max(1,abs(objective))` and complete bytes no greater than the cap. It is not the task's
full-label `core_terminal` certificate.

### Raw metric repetitions

| State | PSNR | MS-SSIM | LPIPS |
|---|---|---|---|
| Jason 12-bit incumbent | 36.3326681000, 36.3326680632, 36.3326680571 | 0.9938552380, 0.9938552380, 0.9938554168 | 0.0256597009, 0.0256596953, 0.0256596897 |
| Jason 12-bit candidate | 37.2640736191, 37.2640735800, 37.2640735928 | 0.9940516949, 0.9940521121, 0.9940517545 | 0.0252108239, 0.0252108201, 0.0252108332 |
| Nomao 12-bit incumbent | 35.2220703443, 35.2220703817, 35.2220703675 | 0.9848036766, 0.9848036766, 0.9848036170 | 0.1129217297, 0.1129217371, 0.1129217446 |
| Nomao 12-bit candidate | 36.0826885010, 36.0826884964, 36.0826884700 | 0.9849748611, 0.9849748015, 0.9849746823 | 0.1101795286, 0.1101795286, 0.1101795137 |
| Jason 16-bit incumbent | 38.3150065927, 38.3150065783, 38.3150065636 | 0.9951140285, 0.9951139092, 0.9951140285 | 0.0214586426, 0.0214586370, 0.0214586481 |
| Jason 16-bit target-MSE candidate | 38.7765989189 | 0.9944617152 | 0.0218151733 |
| Jason 16-bit proximal candidate | 38.7079576645, 38.7079577368, 38.7079577295 | 0.9943771362, 0.9943771362, 0.9943769574 | 0.0220823418, 0.0220823400, 0.0220823288 |

## Search mechanics and cost

| Probe | Accepted edits | Objective queries | Exact byte queries | Sweep time |
|---|---:|---:|---:|---:|
| Jason 12-bit | 24,763 | 1,376,228 | 109,760 | 257.61 s |
| Nomao 12-bit | 25,654 | 1,376,255 | 103,175 | 256.46 s |
| Jason 16-bit target MSE | 18,110 | 2,064,173 | 176,705 | 397.90 s |
| Jason 16-bit proximal | 16,814 | 1,376,125 | 107,682 | 266.02 s |
| Jason 16-bit gradient repair | 4,035 | 1,179,504 | 149,584 | 461.27 s |

Times sum only recorded sweep bodies. They exclude renderer-bridge construction (8.05 s for Jason
12-bit, 8.00 s for Nomao 12-bit, 15.65 s across the two Jason 16-bit target-MSE processes, 9.53 s
for the proximal probe, and 9.99 s for the gradient probe), persisted-CUDA rescoring, and artifact
I/O.

The Jason 12-bit candidate blob is
`3a904052197bdf0b7291fab82707f5ff5b8c6cd8aa3b43a636535544e84b8c6e`; the Nomao 12-bit
candidate is `73ce43478766b34c4c81c6facc565765df0369cd389b6441717308c2593fe6cd`.
The locally-zero Jason 16-bit target-MSE candidate is
`b36d672455501e790fa84785952617e8907b2df672ca308072c8eb9226971bdb`; the proximal candidate is
`2fbf66f54444003768d037946067f373a6e87c092fd4304b27911f10bd223fd3`.

Maximum checkpoint render drift was at most `1.33e-15`; maximum observed objective drift was
`1.876e-12`. Every final blob had the requested complete size and passed the ordinary
encode/oracle check.

## Unsealed structural-proxy diagnostic

A read-only endpoint diagnostic compared five simple residual-image proxies on the incumbent and
candidate renders. This unsealed inline diagnostic was not preserved as executable source; the
percentages are descriptive, post-hoc, and not independently reproducible from this note.

| Proxy change versus incumbent | Jason 16-bit target MSE | Jason 16-bit proximal | Jason 12-bit | Nomao 12-bit |
|---|---:|---:|---:|---:|
| 4-neighbor gradient MSE | +0.589% | +0.840% | -5.580% | -0.565% |
| Sobel MSE | +0.605% | +1.634% | -12.376% | -2.713% |
| Laplacian residual | +0.629% | +0.771% | -4.930% | -0.121% |
| Low-pass residual | -27.242% | -24.241% | -40.024% | -40.068% |
| Target-weighted Sobel | -7.339% | -6.029% | -21.298% | -2.198% |

Low-pass and target-weighted Sobel do not even separate these four observations: they improve
strongly on the 16-bit endpoints whose MS-SSIM and LPIPS regress. Post hoc, the signs of the plain
gradient, Sobel, and Laplacian rows separate the two all-three-metric improvements from the two
PSNR-up/perceptual-down endpoints. This is not predictive evidence. Four-neighbor gradients were
selected for one killing test because they use the smallest stencil and admit exact local updates.

Starting from the locally-zero Jason 16-bit target-MSE state, the repair minimized forward
horizontal/vertical residual-gradient SSE under:

```text
complete SSP2F bytes <= 64,662
CPU target MSE <= incumbent CPU MSE * 10^(-0.10/10)
```

The `+0.10 dB` floor is a proposed old-data calibration constant only. It is not inherited
COMP-012 authority; the task's eventual effect-size `G` remains a freeze blocker.

After the 12 fine sweeps fixed in the unsealed script before this run, the repair had accepted
4,035 edits and reduced its gradient objective from `433.8177521751637` to
`428.2334493761749`, a decrease of `5.5843027989888` or `1.287246262051%`. It finished only
`0.000224849958` objective units below the PSNR-floor ceiling and still accepted 75 edits in the
final sweep. Its exact 64,662-byte blob is
`311899883ce9ce4692b190ac944b880f90afa06b5eda4f92295f86ed1eb0ebc7`.

Three fresh persisted-CUDA repetitions were:

| PSNR | MS-SSIM | LPIPS |
|---:|---:|---:|
| 38.4150118992 | 0.9934103489 | 0.0230688285 |
| 38.4150117796 | 0.9934103489 | 0.0230688266 |
| 38.4150118096 | 0.9934101105 | 0.0230688285 |

The conservative deltas versus the incumbent are `+0.1000051869 dB` PSNR,
`-0.0017039180` MS-SSIM, and `+0.0016101915` LPIPS. This is substantially worse perceptually than
the unregularized target-MSE endpoint. This fixed four-neighbor objective, `+0.10 dB` floor, and
12-sweep schedule is rejected on this one cell without same-cell tuning. The nonzero final sweep
means it is not an optimizer-terminal result and does not reject gradient objectives as a class.
The static endpoint sign separation did not predict this optimization trajectory.

Gradient-repair script SHA-256:
`f59aaba239d5efe2e71e8ae99e6075fa942b1c3bf0c428e0b610fba97f202fb4`. It remains an unsealed
`/tmp` calibration harness, not task-bound source.

## Early-trajectory falsification

A deterministic rerun of the same exposed Jason 16-bit target-MSE path saved the starting state,
all eight coarse-sweep states, and the first two fine-sweep states. All replayed blob hashes and
all ten available sweep-state hashes matched the earlier run. Each checkpoint and the SSP2E
incumbent then received one frozen persisted-CUDA metric call. This is a trajectory diagnostic,
not a repeat-complete metric schedule or a valid post-hoc checkpoint-selection policy.

Deltas below are against the freshly scored 64,662-byte SSP2E incumbent
(`38.3150065681` PSNR, `0.9951139092` MS-SSIM, `0.0214586332` LPIPS):

| Checkpoint | Complete bytes | PSNR delta | MS-SSIM delta | LPIPS delta |
|---|---:|---:|---:|---:|
| Starting `step3` | 62,290 | -1.612855940 dB | -0.005719900 | +0.009318775 |
| Coarse 1 | 62,401 | -0.331091144 dB | -0.004361033 | +0.005955009 |
| Coarse 2 | 62,395 | +0.079616569 dB | -0.002880156 | +0.003832411 |
| Coarse 3 | 62,398 | +0.193604128 dB | -0.002372324 | +0.002918275 |
| Coarse 4 | 62,403 | +0.229915755 dB | -0.002189219 | +0.002654023 |
| Coarse 5 | 62,399 | +0.243560019 dB | -0.002097428 | +0.002592701 |
| Coarse 6 | 62,405 | +0.248586556 dB | -0.002066493 | +0.002515785 |
| Coarse 7 | 62,396 | +0.250571455 dB | -0.002046108 | +0.002519475 |
| Coarse 8 | 62,402 | +0.251264057 dB | -0.002043605 | +0.002549114 |
| Fine 1 | 64,662 | +0.426089848 dB | -0.000888824 | +0.000684662 |
| Fine 2 | 64,662 | +0.445779503 dB | -0.000766039 | +0.000544777 |

No saved checkpoint simultaneously improves all three metrics. Among the saved checkpoints, PSNR
first exceeds the incumbent at coarse sweep 2, where both perceptual metrics are already
unfavorable. Against the earlier three-repeat incumbent envelope, the equal-byte fine-2 row is
approximately `+0.445779478 dB`, `-0.000766158` MS-SSIM, and `+0.000544773` LPIPS. This rules out
selecting any of these 11 saved/scored checkpoints as the three-metric repair. It does not evaluate
unsaved within-sweep states, fine-sweep endpoints 3--16, or another path.

The machine-readable readout hashes to
`4ec45df52de95d010d985442fe734740633a62662a55184c4e9dfc95d4c9d824`; its search result hashes
to `143d1fdaf8ea66d8b344f0990c6fc29c6f441c97c1b1012a2023854758ec5afd`, and the unsealed
trajectory script hashes to
`d2fb01c87b43372d1c50e0db177d843f877094c5c9adb5a117237bc5d0f53132`.

## Actual-MS-SSIM gradient repair

A predeclared, independently audited exposed-only probe tested the next calibration candidate:
use the exact five-scale `pytorch_msssim.ms_ssim` gradient only to rank legal RGB symbol moves,
while independently enforcing exact complete bytes and an authoritative CPU-MSE floor equivalent
to at least `+0.10 dB` over the cold-decoded incumbent. Only dyadic accepted prefixes were eligible
for exact metric evaluation and selection. The fixed cap allowed at most 226 metric evaluations;
the run used 13.

The search stopped after its first macro. The origin's deterministic CPU MS-SSIM was
`0.994461715221405`; the best evaluated legal prefix contained one accepted edit and scored
`0.9944531917572021`. Because no evaluated prefix strictly improved the actual metric, the probe
selected no prefix. Its final 64,662-byte blob is therefore bit-identical to the starting
target-MSE endpoint:
`b36d672455501e790fa84785952617e8907b2df672ca308072c8eb9226971bdb`.

Three fresh persisted-CUDA falsifier repetitions on that immutable final blob were:

| PSNR | MS-SSIM | LPIPS |
|---:|---:|---:|
| 38.7765988856 | 0.9944614768 | 0.0218151808 |
| 38.7765988878 | 0.9944615364 | 0.0218151696 |
| 38.7765988737 | 0.9944613576 | 0.0218151789 |

The exact conservative deltas versus the three incumbent rows are `+0.461592280977 dB` PSNR,
`-0.000652670860` MS-SSIM, and `+0.000356543809` LPIPS. Under this fixed `+0.10 dB` floor and
64,662-byte cap, the origin-gradient ranker with dyadic-prefix selection failed on this exposed
cell. This does not reject actual perceptual objectives generally: the gradient is only a
first-order ranking of discrete, rate-coupled edits, and non-dyadic shadow states were deliberately
unscored and unselectable.

The stable search script hashes to
`1d0108d612be44244b8897de5d9515111790310d4ed60576ec2a08831305b4e9`; two independent
pre-execution audits returned GO for exactly one nonpromotable exposed calibration run. The
manually collated machine-readable readout is `actual_ms_repair_readout.json`; the three original
metric stdout records were not separately preserved, so it is not an execution seal. The
temporary result hashes to
`78c6e61c17e2426668c482725ac968306c14894722c566efca1c716dad951e2f`, and the macro record
hashes to `563b773ea104520329cfc70958652902afe447a4b67ffcbe21012ffd571b4617`.

## Interpretation

Two exposed 12-bit cells show the same narrow calibration signal: a target-guided SSP2F RGB search
found a candidate at the SSP2E complete-byte cap whose three frozen metrics were favorable. This is
a second exposed-cell instance, not an independent replication. The runs do not isolate unused
headroom as the cause, and both are nonterminal.

On one exposed 16-bit cell, target-MSE checkpoints with higher PSNR had worse MS-SSIM and LPIPS
than the incumbent, including every saved checkpoint after PSNR first crossed the incumbent. This
association does not establish that MSE descent caused the perceptual loss. The fixed
`lambda=0.25` source-render penalty was descriptively dominated by the unregularized endpoint, and
the fixed four-neighbor repair also failed under its one floor and schedule. These reject those
exact settings and selection among these saved checkpoints on this cell, not parameter-space
proximity, edge objectives, or checkpoint selection as general classes.

The direct actual-MS-SSIM gradient probe stopped after its first macro: all evaluated dyadic
prefixes were worse than its origin. This motivates testing constrained multi-edit or
representation-level proposals, but does not distinguish them from other local trust-region,
line-search, or non-dyadic policies. It is not evidence that StructSplat cannot be improved.

## Claim and provenance boundary

- These are ad-hoc `/tmp` runs, not source-bound or replayed lifecycle artifacts.
- The four exact streams used by the focused comparison are now archived under `streams/`:
  Jason SSP2E/SSP2F SHA-256
  `11f695461712e21eee5a5f7a9f8c3611d8d64a80766046905aef74aa5212e799` /
  `3a904052197bdf0b7291fab82707f5ff5b8c6cd8aa3b43a636535544e84b8c6e` at
  `51,549` bytes each, and Nomao SSP2E/SSP2F SHA-256
  `0fb494bbae298e695fca45fa16fd900f0e6bb2d6f1b84aa9151d7c2662743c52` /
  `73ce43478766b34c4c81c6facc565765df0369cd389b6441717308c2593fe6cd` at
  `52,223` bytes each. Archival makes the byte/hash comparison portable; it does not upgrade the
  ad-hoc runs into source-bound or held-out evidence.
- Source/target SHA-256 pairs are
  `ba0a67fb4997812432523da6763f231afbd1eb67796ecdcb95139e0602e3c193` /
  `28bfbd8a839cb1d2945ebe5fe8fc63d6545e3467a0b9cf3f5c18be15856b2523` for Jason,
  `69fd104e1f0d6c5f12ff0496cb353ad13433ac3c13a23242980f0790d7722db8` /
  `b5db359139833a7fc4913122b94529b651ded1f6679b55e98614a7f5d57b9e22` for Nomao,
  and the Jason 16-bit source is
  `f415107a319832a5375f605068244c8a19cb37bc3da2a40317b1208b168176d6`.
- The incumbent metric rows came from `actual_rows.jsonl`, SHA-256
  `668195c011e3f693a65e88a3d0236adac6f0b5a91eb1fd92cf6e2e9d05cc64e9`; all three cited
  incumbent rows were valid `ssp2e` rows.
- Audit-time hashes of the mutable `/tmp/result.json` files are
  `524bcd19b8495b1bcdcb01240ed4c7b86fbe22cb1a03c8da064ab99a3c2d55ba` (Jason 12-bit),
  `e5c34eaa18d91fa38247782da51bd71b0da133e6203f4a5423cec19a636d2cbc` (Nomao 12-bit),
  `3d0ccf04cd9de2d68a3a09a21d463b28106a159cf83717169e11ed36926c16f1` and
  `18fbf5eccf505083a431c7245eee3e07c717e7e45ab82eaa2b13c0368c8e2149` (Jason 16-bit target
  MSE), `08c861314990fb864331c100c21847315e0d71c1a94736c73911e741c3f86828`
  (proximal), and `9705ae3dc8d44b532b34e822538c13a63c82bad722e267a8bbb72cff881a3f1c`
  (gradient). They are provenance aids, not execution seals.
- The surviving probe script currently hashes to
  `a056723e006447df873ff91cef0df6adae97a1099e405c3776b6bc82ad7f4a9d`, but it was generalized
  between probes and is not an execution seal for every row.
- The persisted metric probe hashes to
  `1521573868aecdf42761059f5bbab9accd1e7a0a1d8f0bf8ab1428e7578e66f4`.
- The contextual SSP2E metric probe used for the fresh trajectory incumbent hashes to
  `fb9784e21f0f0dda1c4880b973c1decddd19d184fa5c5a4b53aca472578608b0`.
- Persisted renderer SHA-256:
  `a18b09c160abd7d509c0886f898456f2b59042b10a0cb778a2bd21ba0ce328b3`.
- LPIPS state SHA-256:
  `a52b7367d9d082a8988b25abcbae343cd1ad25c9c72c1fe592f571d00e470fc6`.
- The earlier repaired COMP-012 draft reviewed alongside this calibration hashes to
  `335366dc0e925de2b18f2a131e3ad902b8f2e3a13fbb4e23d68cc5c4327bd5f7`; it did not bind these
  earlier calibration runs.
- A fine-sweep local zero is not the task's exhaustive full-label `core_terminal` certificate.
- Equal complete bytes here compare an SSP2F candidate with the strongest SSP2E incumbent cap;
  they do not isolate a same-format or equal-decoder-complexity effect.
- No result here supports a held-out improvement, default change, runtime/compression benefit,
  convergence claim, expressiveness claim, state-of-the-art claim, or access to COMP-012 data.
