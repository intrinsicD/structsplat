# HIER-021 — Low-coverage 7x7 RGB exception patches

## Context

HIER-020 isolates the remaining exact-7k failure. Its source-known SST1 selector is pointwise safe,
passes a fresh four-image bank, repairs all four local failures in 20 persisted HIER-015--019
fields, and leaves 14/16 `tests/test_images` rows fully noninferior to HIER-005. On COCO `000009`
and `000034`, however, the individually safe same-field substitutions raise whole-image LPIPS by
`0.0002333` and `0.0002725`. The frozen transaction therefore selects the ordinary render and
leaves worst-pixel regressions on both images plus a 7x7 regression on `000034`.

Post-failure attribution on these consumed images found the problem is spatial coherence rather
than RGB direction: replacing only isolated low-coverage pixels with exact source RGB still raises
LPIPS, while replacing the union of radius-3 Chebyshev neighborhoods (the same 7x7 support as the
artifact metric) improves LPIPS, MS-SSIM, MSE, and both local maxima. The two candidates use 373
and 586 changed pixels, costing 2,627 and 4,118 raw reference bytes under the format below. This
is exposed motivation, not confirmation evidence.

## Goal

Determine prospectively whether a small, explicit source-RGB exception layer over coherent
low-coverage neighborhoods makes the selected exact-7k portfolio noninferior to HIER-005 on every
global and local metric, then replay the frozen rule on all prior fields and `tests/test_images`.

## Non-goals

- Do not describe SPT1 as a repair inside the 7,000-Gaussian field. It is an additional residual
  representation with visible source-derived bytes.
- Do not change/refit the direct field, its row count, initializer, optimizer, horizon, normalized
  equation, epsilon, topology, or maintained defaults.
- Do not tune the coverage threshold, radius, payload layout, or transaction after opening the
  prospective bank.
- Do not call NPZ+SPT1 a production codec or make an actual-rate claim without a complete selected
  stream. Report raw payload and field-plus-payload reference bytes only.
- Do not claim novelty for sparse residual coding, exception maps, ROI patches, or target-known RDO.

## Frozen candidate

Let `C0` be the ordinary direct normalized render, `D` its unnormalized coverage denominator, and
`T8` the encoder's displayed 8-bit source. Seed sites are exactly `D < 1e-8`. Expand the seed mask
by a fixed Chebyshev radius 3 (a clipped 7x7 square), then retain only expanded sites where
`q8(C0) != T8` and assigning `T8/255` strictly reduces raw RGB SSE. The candidate copies `T8/255`
at retained sites and is bit-identical to `C0` elsewhere.

SPT1 is canonical and source-free at decode:

```text
header:  magic[4]="SPT1", height:u32le, width:u32le, count:u32le
record:  raster_flat_index:u32le, red:u8, green:u8, blue:u8   # exactly 7 bytes
order:   strictly increasing raster-flat index, no duplicates or trailing bytes
```

The decoder ordinarily renders the unchanged field once, then assigns the stored RGB8 records.
Candidate selection requires a nonempty canonical payload, exact encode/decode, unchanged field
hash/count, finite output, exact equality outside records, repeated parity `<=2e-5`, pointwise raw
strict improvement/display exactness, and verified whole-image MSE/pixel/7x7 non-regression.
Select SPT1 only if MS-SSIM delta is `>=-1e-7`, LPIPS delta `<=1e-7`, and at least one raw/local
metric materially improves; otherwise select the zero-byte ordinary mode.

## Prospective bank

Before opening any selected pixels, unreferenced COCO `train2014` basenames were sorted by
`SHA256("HIER-021-v1:" + basename)` and the first four bound:

| source | selection SHA-256 | file SHA-256 |
|---|---|---|
| `COCO_train2014_000000233566.jpg` | `000011281017d254c9c240e2c98a24bc0dc31e0c6506b49710f2d7a5a32424d3` | `064d1aa1f712c2c9850813b7745b0c77c57ab1aa0a5716574157dcfe90e5ec56` |
| `COCO_train2014_000000455444.jpg` | `0003568a57cd479208dcbdb645ac09f259d577df98745860df05d7b064d9b50a` | `f1e87849fed2d66758df0fdb1970b4d3a21c013314adae154aa10e73565da340` |
| `COCO_train2014_000000552149.jpg` | `0004a29dcfd3a3e88dbca2c44d4cebb2a9d4a4226d367cae0bd3f3831c8736e5` | `c6d43328bede507e143a443711133df6312dd71566a39d5002014dd61f4c0c50` |
| `COCO_train2014_000000439248.jpg` | `00055b45e8344269f5b0a00c3c556114f2828f4ac4e6cdd664ef265c4dff450f` | `585f84581a42a57f6ba5bee526088c0e45b6694d20321a46fc3042a1a0628582` |

All use deterministic maximum-side-512 Pillow LANCZOS rasters, full masks, exact N=7,000, seed
0, LPIPS, immutable directories, and HIER-015's direct normalized fit at `eps=1e-8`.

## Phase A — fresh development

Run HIER-005 once and direct normalized fit once per source. Persist ordinary baseline, seed and
expanded masks, SPT1 candidate, selected interpretation, exact payload bytes/hashes, pointwise and
global certificates, synchronized encoder/decode work, common-coordinate crops, site overlays,
and full frames. Charge both candidate construction and the target-known whole-image metric
transaction to encoder overhead. The selected portfolio passes only if all four cells are
complete, exact, finite,
transaction-safe, and every selected direct result gains at least 2 dB without worsening MSE,
MS-SSIM, LPIPS, displayed pixel maximum, or 7x7 maximum versus HIER-005. Median encoder overhead
must be `<=1.25x` direct fitting and median selected decode `<=2x` ordinary rendering. If a fresh
baseline local failure exists, SPT1 must repair it. Native visual review must find no patch seam,
speckle, ringing, checker/lattice structure, wash, or blur.

Any failure stops without replay or rule changes.

## Phase B — frozen replays

Only after Phase A passes, apply the exact rule without refitting to 24 persisted fresh/lineage
direct fields: four each from HIER-015--020. Require no per-image
MSE/MS-SSIM/LPIPS/pixel/7x7 regression, exact decode, and repair of every recorded
direct-versus-HIER-005 local failure.

Then apply it to the 16 persisted direct fields from HIER-020's complete `tests/test_images` run,
for 40 total no-refit replay fields.
Require every selected result to gain at least 2 dB and be individually noninferior in MSE,
MS-SSIM, LPIPS, displayed pixel maximum, and 7x7 maximum versus its frozen HIER-005 control. Any
counterexample rejects the bounded “works on the repository bank” statement and stays visible.

## Acceptance criteria

- [ ] Typed default-off SPT1 encode/decode rejects hostile input and proves exact RGB/index replay,
      field immutability/count, pointwise raw/display safety, outside identity, and CPU/CUDA parity.
- [ ] The source-bound fresh driver and no-refit replay record complete side-byte, timing, metric,
      provenance, persistence, and visual evidence.
- [ ] The prospective bank runs exactly once before any frozen replay.
- [ ] Results receive adversarial audit, ARA/task/docs synchronization, bundle checks, focused and
      full verification; production codec/default integration remains separately reviewed.

## Interfaces touched

One default-off source-patch module, HIER-021 experiment/replay drivers, focused tests, report
schema registration, architecture docs, ARA records, this task, Index, and session brief.

## Depends on

HIER-020/019/005, ADR-0003/0006, CORE-013, BENCH-002

## Reversible fallback

Omit SPT1 and use the exact ordinary normalized render from the unchanged field.

## Agent workflow

- Driver: codex
- Reviewer: codex
- Turn: reviewer
- Reviewed revision: report manifest `9d0e1ba945b885c9105377a85266f66dbff624549403b1fa6e6b6ff398cb4418`

### Handoff log

Append exact `### Handoff` and `### Review` blocks using `tasks/README.md`. This remains a
dirty-source diagnostic without formal prospective review.

### Handoff

#### Objective

Test whether coherent source-RGB exceptions close the bounded exact-7k portfolio everywhere in
the repository bank without changing any Gaussian field.

#### Changes

Added lazy default-off SPT1 encode/decode and pointwise selection, frozen fresh/no-refit drivers,
hostile-input and CPU/CUDA tests, report-schema support, corrected full encoder/RDO accounting, and
the synchronized task/docs/ARA evidence record.

#### Evidence

The reviewed fresh bundle passes at manifest
`413e56064a228184cec423e22c33bb097577f10e5aec3449c9fa47055edb273d`. The reviewed 40-field
replay passes at manifest `9d0e1ba945b885c9105377a85266f66dbff624549403b1fa6e6b6ff398cb4418`:
24 selected fields, 20,137 records/141,343 bytes, unchanged hashes, and all nine local failures
repaired. The results audit is
`ara/evidence/hier015-hier021-exact7k-portfolio-2026-08-10/run.md`.

#### Assumptions

SPT1 is an explicit target-known RGB residual; raw NPZ+SPT1 bytes are reference accounting only.

#### Uncertainties

The fresh bank has no local failure; the replay is consumed, one-seed/device, dirty-source, and
producer-reviewed. Arbitrary images, actual rate, full-resolution, and production integration are
unproved.

#### Review focus

SPT1 canonicality, exact outside identity, field immutability, global rollback, source-free decode,
the 24+16=40 scope correction, and full encoder/RDO timing recalculation.

#### Protected actions not taken

No maintained pipeline/default, pure-Gaussian claim, complete codec, result mutation, commit, push,
or unrelated IDE/user file changed.

#### Recommended next action

Obtain distinct scientific/code review. Any integration must be a separate complete-stream task;
any pure-field successor must avoid source RGB exceptions.

### Review

#### Verdict

Provisionally accepted (self-reviewed)

#### Self-reviewed

Yes

#### Correctness

SPT1 is canonical, bounded, and hostile-input-safe; selection is pointwise SSE-improving, decode is
exactly identity outside stored coordinates, and all replayed Gaussian field hashes remain
unchanged. The audit corrected the replay scope to 24 prior plus 16 repository fields and corrected
future telemetry to include candidate construction and metric/RDO work in encoder time.

#### Evidence quality

Reviewed fresh and 40-field no-refit bundles pass all frozen gates; the latter repairs all nine
known local failures with 20,137 records in 141,343 bytes. Exact payload reparsing, native visual
inspection, parity checks, and independent timing recomputation agree. The bank is consumed,
one-seed/device, dirty-source, and producer-reviewed; the fresh bank contains no baseline failure.

#### Simplicity

The implementation is a lazy, default-off seven-byte-per-record sidecar over an unchanged field.
That clarity also exposes the main limitation: it is source-RGB residual coding, not a Gaussian
field improvement.

#### Missing cases

Held-out sources, full-resolution evaluation, actual compressed rate, complete-stream decode,
multi-seed/device confirmation, and independent review are absent.

#### Required changes

None for the bounded diagnostic. Do not change the maintained default or claim universal,
pure-Gaussian, or codec-level success without a new prospective task and distinct review.

#### Optional improvements

Run a held-out complete-stream rate-distortion study and compare coherent residual coding against a
standard image-residual codec before considering integration.
