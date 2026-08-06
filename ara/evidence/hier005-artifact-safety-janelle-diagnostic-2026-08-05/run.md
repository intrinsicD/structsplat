# HIER-005 artifact-safety factorial and bounded repair diagnostic

## Scope and verdict boundary

This is a user-authorized, post-hoc, dirty-worktree diagnostic on one exposed downscaled Janelle
image. It tests whether support cutoff/fade or topology-frozen all-active recovery removes the
localized cell/hole artifacts seen after pixel-field contraction, then executes the predeclared
bounded repair because every fixed-count 4k arm fails. It is not preregistered, independently
reviewed, held out, equal-FLOP/equal-wall-time, a complete-codec comparison, or claim-ready.

The implementation source set is bound in this order:

1. `src/structsplat/pixel_contraction.py`
2. `scripts/experiments/hier005_pixel_contraction.py`
3. `scripts/experiments/hier005_artifact_repair.py`
4. `scripts/check_report_bundle.py`
5. `tests/test_pixel_contraction.py`
6. `docs/architecture.md`
7. `docs/additive_field_v2.md`

The SHA-256 of the ordered `sha256sum` ledger is
`53c3b32baf0d6bb3bddf3ff376a6662002416d89423ed851f0b2d931106eab32`.
CUDA atomic-gradient order means optimizer fields are numerically, not bit, reproducible.

## Source, raster, metrics, and gate

- RGB: native `C0001.jpg`, 5,328x4,608, 14,268,226 bytes, SHA-256
  `ae24fe99d3f8edbd04cd2c85ebc4fe9bfd95abe878c22abb7691cadcfc5c411b`.
- Mask SHA-256:
  `94dcbf7005dbeb1d183e259a569d783aa5df900255e763385bed91f02d3b80c3`.
- Evaluation: 512x443, Pillow LANCZOS RGB and nearest mask resize, 15,929 active pixels,
  29,263-byte black-matted evaluation PNG.
- Counts: exact 4,096 and 8,192 signed direct-additive rows. Support is hard/faded at cutoff
  3.0/4.5 sigma. Recovery is touched-only interleaving `16x50` or topology-frozen all-active
  error-weighted terminal polish `1x800`.
- PSNR/MSE use foreground pixels. SSIM/MS-SSIM/LPIPS use the full black-matted raster. Localized
  metrics use the exact displayed 8-bit PNGs: foreground pixel RGB-RMSE tails/max and maximum
  complete in-canvas black-matted patch RMSE at 3/7/15/31 pixels.
- The provisional exposed-raster gate is pixel max `<=0.02` and 7x7 patch max `<=0.01`. It is a
  development safety threshold, not a calibrated human-observer claim.

## Fixed-count factorial outcomes

Every row below has a cold lossless-field load, two maintained renders, parity below `2e-6`, full
reconstruction/error visuals, raw JSON/JSONL/CSV, and 53 standalone SVG curves. “Terminal” means
one all-active error-weighted block only after topology is frozen.

| support/recovery | N | PSNR | MS-SSIM | LPIPS | pixel q99 | q99.9 | pixel max | patch 3 | patch 7 | patch 15 | patch 31 | gate | total s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| hard3 touched | 4096 | 30.481 | 0.997020 | 0.023741 | 0.1020 | 0.1319 | 0.2058 | 0.0998 | 0.0707 | 0.0613 | 0.0483 | FAIL | 13.954 |
| hard3 touched | 8192 | 52.356 | 0.999981 | 0.000017 | 0.0075 | 0.0106 | 0.0148 | 0.0072 | 0.0053 | 0.0039 | 0.0032 | PASS | 9.175 |
| hard3 terminal | 4096 | 36.695 | 0.999044 | 0.008887 | 0.0469 | 0.0631 | 0.0736 | 0.0508 | 0.0335 | 0.0251 | 0.0205 | FAIL | 14.174 |
| hard3 terminal | 8192 | 44.396 | 0.999842 | 0.001553 | 0.0310 | 0.0425 | 0.0524 | 0.0326 | 0.0227 | 0.0159 | 0.0121 | FAIL | 10.513 |
| hard4.5 touched | 4096 | 30.516 | 0.997091 | 0.023144 | 0.1009 | 0.1303 | 0.2034 | 0.1017 | 0.0712 | 0.0615 | 0.0498 | FAIL | 17.159 |
| hard4.5 touched | 8192 | 52.339 | 0.999981 | 0.000017 | 0.0075 | 0.0106 | 0.0148 | 0.0072 | 0.0053 | 0.0039 | 0.0032 | PASS | 11.754 |
| hard4.5 terminal | 4096 | 36.669 | 0.999076 | 0.009155 | 0.0460 | 0.0628 | 0.1095 | 0.0627 | 0.0324 | 0.0220 | 0.0191 | FAIL | 16.395 |
| hard4.5 terminal | 8192 | 44.461 | 0.999845 | 0.001617 | 0.0310 | 0.0425 | 0.0513 | 0.0326 | 0.0226 | 0.0156 | 0.0117 | FAIL | 11.994 |
| fade3 touched | 4096 | 32.750 | 0.997118 | 0.008995 | 0.0889 | 0.1308 | 0.2877 | 0.1069 | 0.0619 | 0.0503 | 0.0422 | FAIL | 19.686 |
| fade3 touched | 8192 | 50.058 | 0.999988 | 0.000017 | 0.0104 | 0.0177 | 0.0337 | 0.0161 | 0.0093 | 0.0064 | 0.0047 | FAIL | 13.799 |
| fade3 terminal | 4096 | 36.922 | 0.998098 | 0.004198 | 0.0483 | 0.0686 | 0.1319 | 0.0542 | 0.0371 | 0.0268 | 0.0197 | FAIL | 17.359 |
| fade3 terminal | 8192 | 50.640 | 0.999990 | 0.000011 | 0.0104 | 0.0178 | 0.0280 | 0.0155 | 0.0088 | 0.0062 | 0.0047 | FAIL | 13.138 |
| fade4.5 touched | 4096 | 30.503 | 0.997125 | 0.022990 | 0.1029 | 0.1306 | 0.2149 | 0.1009 | 0.0716 | 0.0621 | 0.0499 | FAIL | 17.773 |
| fade4.5 touched | 8192 | 52.368 | 0.999981 | 0.000016 | 0.0075 | 0.0106 | 0.0148 | 0.0072 | 0.0053 | 0.0039 | 0.0032 | PASS | 12.018 |
| fade4.5 terminal | 4096 | 36.587 | 0.999087 | 0.009317 | 0.0468 | 0.0670 | 0.1080 | 0.0626 | 0.0318 | 0.0218 | 0.0191 | FAIL | 16.627 |
| fade4.5 terminal | 8192 | 44.608 | 0.999847 | 0.001636 | 0.0305 | 0.0408 | 0.0493 | 0.0315 | 0.0226 | 0.0156 | 0.0116 | FAIL | 11.493 |

Only three rows pass: the hard3, hard4.5, and fade4.5 touched 8k arms. They are effectively the
same local-error result; fade4.5 subtracts only the negligible 4.5-sigma tail. Larger hard support
does not improve the visible morphology. Full fade at 3 sigma changes the solution materially but
worsens enough 8k pixels to fail and produces broad ripple/cell residuals at 4k. The support
hypothesis therefore fails this bounded test.

Topology-frozen terminal optimization is safer than letting all-active geometry alter the later
topology path, but it is still not safe: it improves every 4k average score without passing the
local gate, and it turns each hard/fade4.5 8k pass into a failure. Hard3 touched remains the
simplest passing 8k configuration; no terminal all-active replacement is selected.

At 4k every arm fails. The frozen selection rule minimizes
`max(pixel_max/0.02, patch7_max/0.01)`, then maximizes PSNR. Hard3 terminal wins with normalized
display violation `3.6788` and becomes the sole repair base. The higher-PSNR fade3-terminal row is
not selected because its pixel maximum is much worse.

## Bounded local repair

Each nonzero row below independently forks the exact persisted hard3-terminal 4,096-row field.
Residual centers use stable descending raw RGB MSE with Chebyshev-radius-1 NMS. Appended rows have
fixed 0.75 px isotropic scale and zero rotation; the complete base and rescue geometry are frozen;
only rescue RGB is optimized for 400 Adam steps at LR 0.05. The objective is masked MSE plus four
times the worst 1% pixel-MSE mean. The unchanged base and every candidate are ordered by normalized
raw pixel/7x7 violation and then SSE. Displayed PNG metrics remain final.

| limit | added | achieved N | selected step | PSNR | MS-SSIM | LPIPS | pixel q99 | q99.9 | pixel max | patch 7 | raw violation | gate | estimated bytes | repair s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| 0 | 0 | 4096 | -1 | 36.695 | 0.999044 | 0.008887 | 0.0469 | 0.0631 | 0.0736 | 0.0335 | 3.688 | FAIL | 159,424 | 0.000 |
| 102 | 102 | 4198 | 206 | 36.909 | 0.999052 | 0.008287 | 0.0435 | 0.0517 | 0.0642 | 0.0283 | 3.212 | FAIL | 162,688 | 0.606 |
| 205 | 205 | 4301 | 373 | 37.080 | 0.999068 | 0.007958 | 0.0411 | 0.0513 | 0.0648 | 0.0278 | 3.229 | FAIL | 165,984 | 0.514 |
| 410 | 410 | 4506 | 333 | 37.506 | 0.999105 | 0.006864 | 0.0380 | 0.0500 | 0.0655 | 0.0271 | 3.233 | FAIL | 172,544 | 0.512 |

The 102-row arm is the best repair under the frozen raw checkpoint priority, but every repair row
fails the displayed gate. The 410-row arm improves average fidelity and broad patch error most,
yet leaves the same localized maximum and therefore cannot be selected as artifact-free. All
persisted candidates retain the entire 4,096-row prefix bit-exact and have maintained/in-memory
and repeated-render parity below `2.4e-7` max absolute.

The report was re-executed after correcting the general declared-alpha output path; the actual
Janelle objective mask equals the declared alpha, so the verdict did not change. Three CUDA
executions made the nondeterminism boundary visible: the 102-row and 410-row selected steps and
display maxima were stable, while the 205-row selected step moved from 317 to 387/373 and its PSNR
stayed within 0.007 dB. Every execution failed all four display gates and selected 102 rows as the
lowest raw violation. The original non-v2 bundle is preserved but explicitly superseded by v2.

The same-raster PNG is already 5.45x smaller than the base's 159,424-byte uncoded payload and
5.90x smaller than the 410-row repair's 172,544 bytes. Rescue improves neither exact-count status
nor compression. The requested 4k operating point is infeasible under this method and provisional
gate; the threshold is not weakened.

## Visual audit

The 1:1 evaluation-pixel inspection agrees with the metrics. Hard3 touched 4k has a dense
quadtree/cell imprint. Hard3 terminal attenuates it into smaller speckles but does not remove it.
Fade3 terminal replaces some square structure with broader ripple-like residuals. Hard3 touched
8k is visually clean at the exposed scale and passes both maxima. Hard/fade4.5 terminal 8k
reintroduces localized clusters. The 102/205/410 rescue ladders progressively lower diffuse error
but retain the dominant small region around the reported worst pixel; the full images, fixed-scale
errors, 96x96 worst-neighborhood source/reconstruction/error crops, and rescue-center overlays are
in the repair report.

## Payload and timing interpretation

The fixed-count estimated/canonical raw payloads are 159,424 bytes (5.623 bpp) at 4k and 290,496
bytes (10.246 bpp) at 8k; the reference NPZs add container overhead. These are not complete coded
streams. Native-JPEG ratios compare different resolutions and are inapplicable to a compression
claim. Matrix timings are one live RTX-3050 observation; equal attempted Adam steps do not equal
optimized rows, FLOPs, or wall time. The sub-second rescue timings reuse an already contracted
field and exclude its topology/recovery construction, so they are not end-to-end convergence
measurements.

## Artifact receipts and integrity audit

The hashes in each row are metrics / manifest / HTML SHA-256:

| arm | metrics | manifest | HTML |
|---|---|---|---|
| hard3 touched | `a6e5c79ba8ff9e1081bb301b2098b95684957ca43c88bb546eeabb02fb5dc0df` | `abef5eab2a54a0bb09737218a734f2c83da80805ec037156b70ccfbb37b0523b` | `25948376ff3ba775095bae13cb1f1d77adf79021c9f9c8a1f6ed5968313fd1e2` |
| hard3 terminal | `17bed26d854b30f71a542934db635928e3ef6617d72966fec658f0f8687b10ce` | `794bc2cb2051f4e6009bb57a3fab576a0fe2b167823885c0de59207a01bc51e0` | `dfa48f8880ce0f76bec3a17f1e0f342d0722c87228eedfe37f0279f69de712de` |
| hard4.5 touched | `59f9d929a0a8f13564fbd3fdae387c63841582e786d7cd9c01a83f5dd8a16651` | `678f707385655e2e38c1a3d79fe3066e449f9cc05a49a9128b89314cea5a089f` | `bdb7ab3ab019f77697764363baa4db4dd7bb4101f591c99b6fe4678b94dba4d0` |
| hard4.5 terminal | `244ad219a62d6af81bc2839496d238bfdb475798c9107d8ab178184614d086d9` | `79d2bb599e4da5dd764268cba0b623292f0589f4a5f87c2e95f7eb47e123e4e5` | `90697c442e16a6ee45494b5c87bddd566493671703a8943002219d47f6cff2ef` |
| fade3 touched | `fbc3369878524fdec98d919cfbd602a80632c1585a7bb17743bdbd63c21e6166` | `99a67759f368b68bad289a7f4e7b8fa385184b2e8fab189191127f84194130a5` | `7cdb5a491329858dc923ca2406648abdad345cae3e38f20f4f3ea3046fb82e50` |
| fade3 terminal | `2408e9bae076803806122ffcf4cd1b112ef17c0cb0b132421cbe67247c65105c` | `7089745e98569a250c91d61f24064f36379dd3b4f312a78aba165934776dc31e` | `ce5931ab0c50c6d86a290a9d9350850fb4d4ad893ed736824fe5712a89a8cd27` |
| fade4.5 touched | `42615ca938ba4b2abea2befca7a66a2e2b748319de726462d6c24535cccf6bf9` | `00b7b15778a3e161b59eb9d82ac9e901adb7a4c102f9f180b7587acc7366cbd4` | `7aea5cde788f3555779bdbce1904ca2282e62e14e8259db14b6c1bf1a53608ca` |
| fade4.5 terminal | `313b824bc04a46fb93e759f6ab670ee3c13e65ca915f63aa4eea3ecd979721c6` | `df4593536adf8fc2db08b2d439e74163603e46f50df6a4730ebdf9674ed4d984` | `671656b3b26f87e36ff9ddda8bc1b361d82a28816963d51eb97455ccb39b1ca1` |
| local repair v2 | `678f1427f202c351660f2898f7275ad698b0384eeabb8f7b6118acd3d4d752a4` | `5c33dfa259f3a6c4fe7295d133495dba1d2080ec7384bbed4699666a186e3aec` | `422e32bdcd20a87b3b76c8195dc7d8edc4cdeaa72fe2b7e3ef399a61dcdba272` |

All eight matrix manifests verify all 75 listed files, all 73 local HTML links resolve within each
bundle, every report has two finite exact-count rows and 53 SVG curves, source/mask/raster identity
matches, artifact-gate arithmetic recomputes, and cold/repeated parity passes. The repair report's
`verification.json` SHA-256 is
`bf3153997fdef5e227ca3b8169dbe3ca56be840a896e6e5ca6909e1ea363f262`; it verifies 86 manifest
entries, 83 contained links, four metric rows, base-prefix identity, and render parity. All nine
authoritative bundles also pass `scripts/check_report_bundle.py` under their explicit non-claim
HIER-005 schemas. The matrix
contains all 16 cells including failures; the repair contains the base and all three failures.

## Diagnostic outputs

- `results/hier005_janelle_artifact_hard3_touched_2026-08-05/index.html`
- `results/hier005_janelle_artifact_hard3_terminal_2026-08-05/index.html`
- `results/hier005_janelle_artifact_hard45_touched_2026-08-05/index.html`
- `results/hier005_janelle_artifact_hard45_terminal_2026-08-05/index.html`
- `results/hier005_janelle_artifact_fade3_touched_2026-08-05/index.html`
- `results/hier005_janelle_artifact_fade3_terminal_2026-08-05/index.html`
- `results/hier005_janelle_artifact_fade45_touched_2026-08-05/index.html`
- `results/hier005_janelle_artifact_fade45_terminal_2026-08-05/index.html`
- `results/hier005_janelle_artifact_local_repair_2026-08-05/index.html` — superseded after the
  general declared-alpha correction; retained as a negative diagnostic receipt.
- `results/hier005_janelle_artifact_local_repair_v2_2026-08-05/index.html` — authoritative repair
  report.

## Verdict and required next evidence

For this exposed raster, 8k hard3 touched recovery is the lowest-complexity tested artifact-safe
operating point. The tested support changes, terminal all-active geometry polish, and bounded
fixed-geometry RGB rescue do not make 4k artifact-safe. HIER-005 must report 4k as infeasible
rather than promote the highest-PSNR failure.

The next method iteration should not tune these exposed arms further. A fresh predeclared study
should treat local-error feasibility as a constraint and test a genuinely stronger fallback—local
uncontraction/preserved pixel leaves or a complete-codec residual exception channel—on disjoint
images, with deterministic/repeated recovery, complete bytes, matched work, and a distinct
scientific review. No semantic, recovery default, production pipeline, convergence-speed, or
compression decision changes here.
