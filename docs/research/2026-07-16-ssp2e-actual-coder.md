# SSP2E actual coder and captured replay: audited development result

## Decision

COMP-009's frozen decision is **`ABANDON_FIXED_SSP2E_V1`**. The exact spatially conditioned
arithmetic coder is real, lossless at the quantized-field boundary, and cheap enough to decode,
but it saves only about 1.3% against the strongest factorized control available inside COMP-009.
It misses the preregistered 10% complete-rate effect by a wide margin on both tuples. The fixed
16x16/rank-8 formulation is closed without retuning.

The experiment also exposed a separate positive engineering result: replacing SSPL1's JSON/zlib
framing with exact factorized arithmetic streams reduces complete bytes by 3.6--4.6% with no
decoded-field or quality change. This is a credible codec-baseline correction, not evidence for
the spatial model and not yet confirmation.

## What was implemented

SSP2E conditions five already-quantized attribute streams (`scale_x`, `scale_y`, `R`, `G`, `B`)
on decoded mean cells. It uses a deterministic 16x16 grid, a fixed rank-8 binary-linear logistic
head, eight frozen starts, exact Q24 fitting objectives, finite positive CDFs, and a canonical
32-bit arithmetic coder with Python/C++ payload and decode parity. Every complete stream contains
its framing, model state, CDF/table identity, and arithmetic termination.

Three exact controls separate different explanations:

- SSP2S applies the same model to a frozen shuffle of mean assignments;
- SSP2L uses empirical factorized arithmetic models; and
- SSP2F uses the fixed factorized CDF assets.

All formats cold-decode to the same eight ordered absolute symbol arrays and the same float32 field
boundary as SSPL1. Consequently, candidate image quality and representation expressiveness are
exactly unchanged by construction.

## Complete-rate result

The primary comparator in COMP-009 was the smaller of SSP2L and SSP2F for each cell. Floating
ratios below are diagnostics; gates used exact integer products and the frozen 100,000-row
bootstrap matrix.

| Tuple | GM SSP2E/primary | Strict wins | Worst ratio | Bootstrap 97.5% upper | Spatial modeled SSP2E/SSP2S | Decode resource result |
|---|---:|---:|---:|---:|---:|---|
| `(12,6,6,8)` | 0.986953715 | 6/8 | 1.018560297 | 0.999583774 | 0.959093814 (8/8 wins) | pass |
| `(16,8,8,8)` | 0.987294104 | 6/8 | 1.008517965 | 0.997034471 | 0.961629155 (8/8 wins) | pass |

Only the worst-image bound passed in the complete-rate endpoint. The required geometric mean was
at most `0.90`, at least seven images had to win, and the bootstrap upper bound had to be below
`0.95`. The position-conditioning attribution was directionally consistent—eight of eight modeled
payloads beat the shuffled control—but its aggregate savings were only 3.8--4.1%, below the frozen
10% mechanism gate. This is evidence that position contains some predictive information, not that
the chosen model is a useful complete codec.

Against shipped SSPL1, the exact complete streams give:

| Tuple | factorized primary/SSPL1 | SSP2L/SSPL1 | SSP2E/SSPL1 |
|---|---:|---:|---:|
| `(12,6,6,8)` | 0.953930 (4.607% smaller) | 0.956874 | 0.941490 |
| `(16,8,8,8)` | 0.964379 (3.562% smaller) | 0.965891 | 0.952130 |

Every primary and SSP2E comparison with SSPL1 wins on all eight images. Part of that gain is leaner
binary framing; COMP-011 therefore introduces the same-container SSP2Z control before making any
new representation claim. SSP2E remains an eligible exact baseline even though it failed its own
promotion gate: an abandoned method does not cease to be a strong comparator.

## Convergence and performance

Every prescribed fit converged. SSP2E required 1--11 sweeps across starts and cells; SSP2S usually
converged in one sweep and never exceeded three. Median SSP2E eight-start fit/selection time was
about 4.56 s, shuffled fitting about 3.17 s, and complete encode plus dual cold decode about
5.06 s. Median end-to-end cell execution was about 13.76 s. These are encoder diagnostics, not
ordinary StructSplat fitting convergence.

Fresh-process bytes-to-boundary decoding passed every frozen resource gate. Across the two tuples,
upper-median time ratios were about `1.026` and `1.042`, worst ratios about `1.066` and `1.079`,
and peak-RSS ratios were `1.0`. Renderer parity was nongating: all sixteen cells agreed within
`5e-4` (observed maximum absolute difference `9.5367e-7`) but were not bit-exact because of CUDA
atomic ordering.

## Why the optimistic oracle did not transfer

COMP-008 gave every spatial cell its exact empirical distribution, rounded entropy down to whole
bytes, and charged no finite-CDF loss, arithmetic redundancy, learned-model mismatch, or decoder
work. That lower bound suggested roughly 17.3--17.6% headroom. The implemented model had to pay all
of those costs and generalize five distributions from a deliberately tiny fixed head. The actual
spatial attribution was only about four percent even before full model/framing effects, leaving no
route to the preregistered ten-percent margin.

This outcome is consistent with the broader literature rather than surprising. SGI transfers
decoded spatial structure into entropy prediction, and HAC uses hash-grid context for Gaussian
attributes, but both operate with richer jointly learned representations and entropy models.
[RDO-Gaussian](https://arxiv.org/html/2406.01597) explicitly optimizes rate and distortion during
entropy-constrained VQ instead of attaching a fixed post-hoc context. The failed result therefore
closes this small frozen transplant, not spatial conditioning in general.

## Lifecycle repair and claim boundary

The valid COMP-009 artifact is
`results/comp009_ssp2e_actual_dev_v1_2026-07-16`. Its original captured replay first exposed
absolute asset-path identity and then a non-reproducible CUDA JIT binary hash. The renderer was
render-only and nongating, but the exact preflight binary had not been persisted. COMP-010 v2r2
therefore allowed exactly two checked lifecycle exceptions: substitute only the two verified asset
path strings, and reuse the sealed nongating renderer proof through the unchanged captured
verifier while forbidding renderer load/build.

Two randomized captured-source codec replays reproduced all sixteen non-timing execution
identities and all 64 complete streams exactly, recomputed `ABANDON_FIXED_SSP2E_V1`, and left the
828-file COMP-009 artifact unchanged. Independent post-result audit returned GO.

Canonical anchors:

```text
COMP-009 preflight       b4d843ce22356839fa3fa39082044e61cc7ee82cc71176780a192d811d63fadc
COMP-009 source archive  fac4ca0978891b3cd16d477ffc04d12dc120168478192293d39af635fca7eb50
COMP-009 run             cc21fc84aab5cd55ecfffc358f67782013d7a7cf56ffc9b8ef34f8ea98de0a24
COMP-009 benchmark       88a706da64836544d997b177b3b4dda2610e6ede3eceb457e98a4aa7899fe162
COMP-009 analysis        f10c4a1906e4bd240c10253508f38c4053cfe332dec13b220262fee7ae990b30
COMP-010 retry preflight a00a7a553468dbe1f2e00e1d6e5ecb38fb3227356e090c73bbf564b39ec1bc00
COMP-010 repair          adbf2c48e1721b6d4b74960211e08ec3c770bebde1d068ddfc88e5a83f78f3a8
COMP-010 child           b482a866742d472d02d7c80db1c86640e5e17a85fdf025a56096173c08fd7ac4
COMP-010 worker          0d7d74a8e6f6a552eeb4f1d3df4a0aeb8f6e6977496400d7c65845c4fd494309
```

The repair does not replay the renderer binary, remeasure persisted resource timings, or add any
quality, convergence, performance, expressiveness, or compression evidence. It repairs only the
negative codec decision's captured-source provenance.

## Next experiment

Do not enlarge the grid, head, start menu, or fit budget on these cells. The selected disjoint
development assay is COMP-011: replace only RGB inside a same-container control with a frozen menu
of deterministic flat and residual codebooks, count every codebook/model/index/framing byte, and
select only streams that conservatively preserve PSNR, MS-SSIM, and LPIPS. GaussianImage provides
the direct transfer precedent: it applies RVQ to weighted colors, while its ablation warns that
RVQ without commitment-aware training can severely damage rate-distortion
([paper](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/01421.pdf),
[official code](https://github.com/Xinjie-Q/GaussianImage)). COMP-011 is intentionally the harder
post-training test; failure points toward a separately preregistered renderer-aware STE/ECVQ assay
on new development data, not retuning on these cells.
