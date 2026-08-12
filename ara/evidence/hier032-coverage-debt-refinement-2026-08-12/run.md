# HIER-032 coverage-debt refinement

## Evidence class

Prospectively reviewed, clean-commit, exposed-source development evidence on canonical Janelle
C0001. The immutable exact-N7,000 run is a protocol-compliant negative: four arms persisted valid
fields, the fixed detail arm failed closed, every completed successor missed the frozen interior
floor, and no method was selected. This is not claim-ready, held-out, native-resolution,
actual-rate, default, or novelty evidence.

The tested relationship combines known pixel-error allocation, cancellation-resistant absolute
detail signals, and medial/set-cover selection. These components and the relationship-level prior
art audit are recorded in `docs/research/2026-08-12-hier032-coverage-debt-portfolio.md`; HIER-032
makes no novelty claim.

## Frozen identity and command

- Reviewed source commit: `f4cc2996d525b128ea511b96e3a7357009f347d7` on
  `agent/hier032-coverage-debt-refinement`, clean status and empty diff.
- Protocol digest:
  `402588c6c32a93ac1dca615ad50d2cf15248892beaaae1bf80cd9f9e253c9898`.
- Source SHA-256:
  `ae24fe99d3f8edbd04cd2c85ebc4fe9bfd95abe878c22abb7691cadcfc5c411b`.
- Mask SHA-256:
  `94dcbf7005dbeb1d183e259a569d783aa5df900255e763385bed91f02d3b80c3`.
- HIER-031 base-field SHA-256:
  `a0a080ccbd255ce51f11489cd504956a1c5181a495bbca2b4bf74ecb0995c1db`.
- HIER-031 decision SHA-256:
  `52016532a23290b12c45b2b9a75c2fc7e3fb0d3001cd19924f30a1a52eb8e2a8`.
- Raster/seed/device: 1200x1038, seed 0, NVIDIA GeForce RTX 3050 8 GiB, CUDA
  12.8, PyTorch 2.9.0+cu128, required LPIPS 0.1.4, 256-row render chunks.
- Driver/checker/focused-test SHA-256:
  `f572b27a7da13db3d2bf02512b44e11daac423c8fdb47596c875cf91cc6f79da`,
  `ba62e5ea809e156abfb6ac103e8eaaa9706fcb75df61053c75129b9aa9d569d4`, and
  `ab503fe8906cdfef8d2322789f0367891f8438d3e0a6bf56f5809e2cb346f529`.

Executed from a clean named linked worktree:

```bash
PYTHONPATH=src python scripts/experiments/hier032_coverage_debt_refinement.py \
  /home/alex/Dropbox/Work/Janelle/2025_03_07_stage_with_fabric/frame_00008/rgb/C0001.jpg \
  /home/alex/Dropbox/Work/Janelle/2025_03_07_stage_with_fabric/frame_00008/mask/mask_C0001.png \
  /home/alex/Documents/structsplat/results/hier032_janelle_c0001_s1200_coverage_debt_s0_development_2026-08-12 \
  --base-bundle /home/alex/Documents/structsplat/results/hier031_janelle_c0001_s1200_exact7k_boundary_detail_s0_diagnostic_2026-08-12 \
  --max-side 1200 --seed 0 --device cuda --lpips
```

## Frozen result

| arm | weak | placements | foreground PSNR | boundary PSNR | hair PSNR | interior PSNR | floor margin |
|---|---:|---:|---:|---:|---:|---:|---:|
| HIER-031 control | 743 | 0 | 23.858894 | 14.836670 | 22.922862 | 36.067722 | +0.804622 |
| per-pixel fallback | 0 | 775 | 24.981595 | 16.763009 | 24.247050 | 31.149877 | -4.113223 |
| component set cover | 0 | 770 | 24.928731 | 16.720653 | 24.167494 | 31.058401 | -4.204699 |
| contribution-aware merge | 0 | 774 | 24.384085 | 16.468017 | 23.862657 | 29.587986 | -5.675114 |
| coverage then boundary high-pass | error | — | — | — | — | — | — |

The control has 743 weak pixels in 483 components and deficit mass
`22.806974707730113`. Each completed successor closes all weak pixels and raw holes, preserves
exact containment, and improves foreground, boundary, and hair PSNR. Those gains are not safe
successes: every candidate fails only the frozen `35.2631 dB` interior floor, losing
4.92--6.48 dB of interior PSNR relative to the control. MS-SSIM also falls from `0.972107` to
`0.950433/0.949416/0.937117`, while LPIPS worsens from `0.120845` to
`0.164266/0.165885/0.172939`.

The simple fallback is only the lexicographic best tradeoff, not a selected method. Set cover
saves five placements (`770` versus `775`) while slightly worsening every reported quality
metric. Contribution-aware ranking lowers local donor merge SSE but needs four more placements
than set cover and is materially worse globally and visually. Its first-wave placement digest
matches set cover exactly:
`382173270f5942e536dc50c084c31c531154082bb42418d974491b4e2804e66a`.

The final arm reproduced the contribution-aware coverage placement and reached the fixed 128-row
detail batch. Funding that batch reopened nine weak pixels, so the driver raised the prospectively
frozen error:

```text
CoverageClosureError: fixed detail batch reopened 9 weak pixels
```

No partial arm-5 field or history was persisted because arm artifacts are written only after a
method returns successfully. The sealed error event and its code path are supported; the
nine-pixel morphology is not independently inspectable.

## Integrity and report validation

The portable report contains 154 files; all 153 manifest entries match exact relative paths,
byte counts, and SHA-256 hashes. The four persisted fields independently pass:

- exactly 7,000 rows and only `means`, `log_scales`, `rotations`, and `colors`;
- equal in-memory and cold decoded-state hashes with max state difference `0.0`;
- zero centres outside, zero unit support outside, and zero reconstruction outside;
- finite coefficients at or below the frozen absolute limit 16.0;
- maintained/repeated render parity at most `9.5367431640625e-7`, below `2e-5`.

Persisted field receipts:

| arm | NPZ SHA-256 | decoded-state SHA-256 |
|---|---|---|
| control | `a0a080ccbd255ce51f11489cd504956a1c5181a495bbca2b4bf74ecb0995c1db` | `deb837edb412569a6a93ff7371a52da6335a5ea0e27a4edad813541c927925a1` |
| fallback | `ebfce8da4a696485c013da5b0c8f79e138df10217f3de91337dd543c4bfc4c78` | `0d0d3311d0b173cea80ef091e51f1ac605e3ebfee4e8aaddec4affcc92b56017` |
| set cover | `2f92caf2785eef04fe1ac8444466ad23046bd4eb067368b71529f784883e08e1` | `f17943a958b65300c46dce80655f001641f8af4f49d5d31cd57651d91008d42f` |
| contribution | `aee7e1dc547f0f00cbc616386f592a6a32bb7a2b419e5717ce081c42487eda04` | `9693ec3b6d616930c52e39885d4685ee4921e1dd4fd088718d5cae257e190ac7` |

`scripts/check_report_bundle.py` intentionally exits 1 with four problems: the exact five-arm
success matrix is absent, arm 5 has no success relationship receipt, the attempt ledger contains
an error, and the decision is not a complete-success decision. This is the frozen fail-closed
policy, not evidence corruption. The checker-pass task item therefore remains explicitly unmet;
the bundle must not be repaired or rerun.

The `index.html` report was opened under headless Chrome and correctly renders the null
disposition, four-row metric table, fifth-arm error, limitations, and local visual cards. Native
source/reconstruction/error, hair, boundary, coverage, placement, and donor views were inspected.
Coverage debt and thin-edge continuity improve locally, but oscillatory garment/body texture error
spreads into the interior; contribution-aware funding is visually worst. Arm 5 has no legitimate
visual endpoint.

## Immutable report hashes

- Manifest: `598d7f59ed87c2c5f0bbb6d17e32e2c8c236f7b3174640052a85b146392beb14`.
- Metrics JSON: `912b6853b24a98809cd629848ca97a7fc19b363627e3132f48a322ba8e3c6dcb`.
- Attempts: `65742d3733fb193a9b2985fa9a848df0328f3d9ee7738a2b152a90bc7726c024`.
- Decision: `350d5b641287533a41b7bf2d840e9575ea62cea45999e0969b94312fba038da6`.
- Config: `d98c99340860bb371197e37b1704e296f43a5ede74165bf5cfed779b966697cd`.
- Protocol: `b85e37d235f4606ab74d035201245dd907275e7f0bcbcee3ae0dea93960399fe`.
- HTML: `eb8218edd8d431cf8a35693dff93926be24f63501a59e8f5ba2446638ee4c57b`.

## Decision and limitations

Close HIER-032 as a completed negative task. Select no method; preserve the fallback endpoint only
as diagnostic best-tradeoff evidence; leave every maintained default unchanged. The run rejects
the tested relationship at this exact scope: naïve positive-coverage closure consumes rows whose
global appearance projection damages the protected interior, set cover gives negligible capacity
compression, contribution-aware local merge ranking does not translate into better global
quality, and the fixed high-pass batch can reopen debt.

The evidence covers one exposed raster, one seed, one RTX 3050, and a dirty-diagnostic-lineage
HIER-031 base field. It has no fresh images, statistical replication, native resolution,
actual-rate accounting, compression/downstream result, or persisted arm-5 failure state. Any
investigation of the nine reopened pixels requires a new prospectively reviewed task, explicit
failure-state persistence, and a new immutable output directory. Do not repair, rescue, retune,
or rerun HIER-032.
