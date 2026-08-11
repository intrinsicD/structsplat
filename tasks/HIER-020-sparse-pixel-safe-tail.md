# HIER-020 — Sparse pixel-safe confidence-tail payload

## Context

HIER-019's zero-capacity twice-scale self-prior solved the fresh local counterexample
numerically: on `489983` it improved raw MSE 2.78%, worst pixel 0.2611, worst 7x7 patch 0.0559,
and MS-SSIM. The frozen whole-mask transaction nevertheless rejected it because LPIPS rose
`0.0008545`; 76 of 236 low-confidence pixels moved in the wrong RGB-error direction even though
160 helped. The selected baseline then remained 0.187 worse than HIER-005 at its literal worst
pixel, so HIER-019 correctly prohibited all replay. Its separate cold-render timing clause was
also invalidated by an unsynchronized baseline CUDA timer, but that instrumentation defect does
not affect the local-quality rejection.

The encoder has the source image. Instead of applying every `D < 1e-8` prior substitution, it can
retain only sites that are independently safe in both raw and displayed RGB error. For HIER-019's
baseline `C0`, recovered value `R = C0 + eps/(D+eps) P2`, and target `T`, select a low-confidence
pixel only when:

```text
sum_c (R_c - T_c)^2 <  sum_c (C0_c - T_c)^2
and
sum_c (q8(R_c) - q8(T_c))^2 <= sum_c (q8(C0_c) - q8(T_c))^2.
```

This pointwise certificate implies non-regression of raw full-frame MSE, displayed pixel maximum,
and every complete-window patch RMSE. Store selected raster-flat indices explicitly as sorted
little-endian uint32 values after a 16-byte `SST1` header (`magic,height,width,count`). Explicit
coordinates avoid relying on a threshold-near CUDA reduction to reproduce bit ordering. The
decoder recomputes the same-field recovered image and substitutes only those coordinates.

Exploratory attribution on all 20 exposed HIER-015--019 direct fields fixed this rule before the
fresh bank below. It selected 425 pixels total, never regressed raw MSE, displayed pixel/7x7 max,
or MS-SSIM, and improved the known HIER-015/016/HIER-019 local failures. One single-pixel HIER-016
candidate increased LPIPS by `0.0001646`, so a final whole-image transaction still selects the
ordinary render unless candidate MS-SSIM and LPIPS are noninferior. These exposed results motivate
the prospective test; they are not confirmation evidence.

## Goal

Determine prospectively whether a small explicit pixel-safe tail sidecar makes direct normalized
exact-7k fitting uniformly noninferior to its own baseline and closes every HIER-005 local
counterexample; if so, replay the frozen method across all consumed lineage banks and the complete
repository `tests/test_images` bank.

## Non-goals

- Do not change/refit the direct field, its 7,000 rows, normalization equation/floor, initializer,
  optimizer, objective, horizon, topology, or maintained defaults.
- Do not tune the `D < 1e-8` proposal set, twice-scale prior, two pointwise inequalities, SST1
  layout, perceptual transaction, or gates on the fresh sources.
- Do not hide side information: report raw SST1 bytes and field-plus-sidecar reference bytes. Do
  not call NPZ+SST1 a production codec or compare rate without a complete selected stream.
- Do not use source pixels at decode; only the encoder constructs SST1. Do not claim novelty for
  residual/error-safe sparse coding, normalized convolution, or target-known RDO.

## Prospective bank

Before opening any selected pixels, unreferenced COCO `train2014` basenames were sorted by
`SHA256("HIER-020-v1:" + basename)` and the first four bound:

| source | selection SHA-256 | file SHA-256 |
|---|---|---|
| `COCO_train2014_000000046728.jpg` | `00000931c0b426b1e276a951bc8aa1c50203aa2a50411c9ee2072d9aa1e6992e` | `72b147fd1ccf3e29e0e85ac556447009901841a98755bc0caee1d35d7de9d07c` |
| `COCO_train2014_000000036289.jpg` | `0001a21eccc35740069a5b8f008c77f0f91df5d09eacbb51b8d40308f254cd54` | `bd1d848a1d12ab05546b36d46be4090fb7dd28acea7f4c759ffb2b818faac8bb` |
| `COCO_train2014_000000466403.jpg` | `00022135883480d3bd755a22b3d08b6070940afe685d1a3d5d5661f6d6044d38` | `e3be2be68738ade64381301532276fef62d513684cf3e7fcdc2a94572d30c980` |
| `COCO_train2014_000000072902.jpg` | `00029855fe2d51bc41b0eee30f1e7abbbe0d36e4a55b8f3b06b8aec4ae32db65` | `118efde8863cbab7a350ff3d6f4cc3ce73130f5c85c797ba40271f1b558bb81d` |

All use deterministic maximum-side-512 Pillow LANCZOS rasters, full masks, exact N=7,000, seed
0, LPIPS, immutable directories, and HIER-015's unchanged direct fit at `eps=1e-8`.

## Phase A — fresh development

For each source run HIER-005 once and direct normalized fit once. Persist ordinary baseline,
unmasked HIER-019 proposal, pixel-safe SST1 candidate, and selected interpretation. Candidate
selection requires nonempty SST1, exact cold decode of the sidecar, unchanged field hash/count,
finite output, exact equality outside stored coordinates, repeated parity `<=2e-5`, pointwise raw
strict improvement and display non-regression at every stored site, and verified whole-image raw
MSE/pixel/7x7 non-regression. Select SST1 only if MS-SSIM delta is `>=-1e-7`, LPIPS delta
`<=1e-7`, and at least one raw/local metric improves beyond the same tolerances; otherwise select
the zero-byte ordinary mode.

The selected portfolio passes only if all four cells are complete, exact, finite, persistence-
clean, and transaction-safe; every selected image gains at least 2 dB versus HIER-005 without
worsening raw MSE, MS-SSIM, LPIPS, displayed pixel maximum, or 7x7 maximum. Median synchronized
encoder algorithm time including candidate construction must be `<=1.25x` direct fit and median
synchronized selected-tail decode time `<=5x` one ordinary cold render. Full frames, selected-site
overlays, candidate deltas, and common-coordinate crops must show no speckle, seam, ringing,
lattice/checker, wash, or blur. If the bank contains a baseline local failure, SST1 must repair it.

Any failure stops without replay or threshold/payload changes.

## Phase B — frozen consumed replays

Only after Phase A passes, apply the exact rule to persisted HIER-015--019 direct fields (20
consumed COCO images). Require no per-image raw/structural/perceptual/local regression, exact
sidecar replay, and repair of all recorded direct-versus-HIER-005 local failures. No outcome can
alter the rule.

Then run HIER-005/direct/SST1/selected once on all 16 HIER-013 `tests/test_images` sources. Require
every selected direct result to gain at least 2 dB and be individually noninferior in raw MSE,
MS-SSIM, LPIPS, displayed pixel maximum, and 7x7 maximum versus HIER-005, with clean visuals and
complete timing/side-byte accounting. Any counterexample rejects the bounded “works everywhere”
statement and remains visible.

## Acceptance criteria

- [ ] Typed default-off SST1 selection/encode/decode proves canonical hostile-input parsing,
      exact index replay, field immutability/count, pointwise raw/display safety, implied global
      MSE/pixel/patch safety, and CPU/CUDA parity in focused tests.
- [ ] The source-bound driver records exact side bytes, hashes, candidate/selected metrics,
      confidence/site telemetry, synchronized work, persistence, and portable visuals.
- [ ] Phase A runs exactly once; only a numeric and visual pass accesses consumed/test replay.
- [ ] All counterexamples are retained without feedback into tuning.
- [ ] Results receive adversarial audit, ARA/task/docs synchronization, bundle checks, focused
      and full verification; production codec/default integration remains separately reviewed.

## Interfaces touched

`src/structsplat/tail_recovery.py`, one HIER-020 driver, focused tests, report schema registration,
architecture docs, ARA records, this task, Index, and session brief. Maintained codec/CLI/pipeline
defaults remain unchanged.

## Depends on

HIER-019/018/017/016/015/005, ADR-0003/0006, CORE-013, BENCH-002

## Reversible fallback

Omit SST1 and use the unchanged ordinary normalized field/render. A failed global transaction
stores no sidecar and selects that exact baseline.

## Agent workflow

- Driver: codex
- Reviewer: codex
- Turn: reviewer
- Reviewed revision: report manifest `d5611111b3a8fe3959345f31305d5217898ad0e3a94867c4dfa6748ea34b2cdf`

### Handoff log

Append exact `### Handoff` and `### Review` blocks using `tasks/README.md`. This is a dirty-source
diagnostic without formal prospective review.

### Handoff

#### Objective

Test canonical pointwise-safe SST1 selection across fresh, lineage, and repository banks.

#### Changes

Added hostile-input-safe SST1 encode/decode, pointwise certificates, optimized sparse decode,
fresh/replay drivers, focused tests, report registration, and immutable raw/review bundles.

#### Evidence

Fresh and 20-field reviewed bundles pass, but `results/hier020_tests_test_images_2026-08-10`
(manifest `d5611111b3a8fe3959345f31305d5217898ad0e3a94867c4dfa6748ea34b2cdf`) fails on `000009` and
`000034`: LPIPS rollback leaves two pixel and one 7x7 regressions. The final bank is 14/16.

#### Assumptions

SST1 is explicit side information outside the unchanged field; target-known RDO is encoder work.

#### Uncertainties

One seed/device, consumed test images, producer visual review, and no complete codec/rate evidence.

#### Review focus

Canonical parsing, pointwise-to-window safety, optimized-decode parity, timing recovery, and
retention of both counterexamples.

#### Protected actions not taken

No threshold retune, field refit, default/codec integration, commit, push, or result deletion.

#### Recommended next action

Treat perceptual spatial coherence as the remaining mechanism and freeze a coherent patch test.

### Review

#### Verdict

Provisionally accepted (self-reviewed)

#### Self-reviewed

Yes

#### Correctness

SST1 parsing is canonical and hostile-input-safe; selected coordinates have pointwise certificates,
optimized decode matches the reference, and rejected transactions preserve the ordinary render.
The repository bank's two LPIPS rollbacks are retained as failures rather than tuned away.

#### Evidence quality

Fresh, lineage, and 16-image repository bundles preserve exact payloads, hashes, timing, visual
review, and both counterexamples. The final 14/16 result is consumed, one-seed/device, and reviewed
only by the producer.

#### Simplicity

The field stays unchanged and the explicit sidecar is small and removable, but it is outside the
pure-Gaussian representation.

#### Missing cases

There is no held-out confirmation, actual entropy-coded rate study, complete-stream integration,
or independent visual review.

#### Required changes

None for retaining the negative 14/16 result. Do not promote SST1 as an everywhere-working method.

#### Optional improvements

Use `000009` and `000034` as frozen counterexamples for a coherent residual mechanism.
