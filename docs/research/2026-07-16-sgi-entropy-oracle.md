# SGI-inspired entropy oracle: audited development result

## Decision

COMP-008's frozen development decision is **`ORACLE_INCONCLUSIVE_IMPLEMENT_CODER`**. Both bit
tuples survive every necessary-condition gate. This is not a compression win: the experiment gave
each spatial cell its exact empirical distributions, rounded entropy downward to whole bytes, and
did not pay finite-CDF loss, arithmetic-coder redundancy, learned-model error, or decode compute.
It establishes only that the exact SSP2E coder is worth implementing.

The source mechanism is [SGI](https://arxiv.org/html/2603.07789), pinned here to its
[official code at commit `1aa6e1f...`](https://github.com/zx-pan/SGI/tree/1aa6e1f99026323f73a90f0a0d5c0af7080d51bb).
SGI combines seed-local generated structure, coarse-to-fine fitting, adaptive quantization, and a
binary hash-grid entropy model. [HAC](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/01178.pdf)
already establishes hash-grid-assisted context prediction for Gaussian attributes. COMP-008
transfers only the weakest causal role: decoded position conditions probabilities for five already
quantized attribute streams. It does not reproduce either system.

## Frozen experiment

Eight CLIC 2020 Professional validation images were extracted by ranged access only after a valid
source/environment preflight. Eight separately named confirmation images remained central-directory
metadata only: no payload, hash, local path, dimensions, or pixels were accessed.

Each development image used one `N=8192`, seed-0, constant-RGB, opacity-free `quadtree_wse` field
fit for exactly 4,000 CUDA updates at max side 768. Two independent 150-step QAT copies used tuples
`(12,6,6,8)` and `(16,8,8,8)`, then produced complete ordinary SSPL1 zlib-9 streams. The oracle
cold-parsed the exact ordered integer symbols and evaluated a fixed `16x16` mean-cell model for
independent `scale_x`, `scale_y`, `R`, `G`, and `B` streams.

For each image, the optimistic SSP2E lower bound was

```text
L = 656 fixed bytes
  + SSPL1 mean zlib payload
  + SSPL1 rotation zlib payload
  + floor(exact conditional entropy / 8),
```

and the control was the complete SSPL1 size `S`. The entropy floor and every gate were certified
with integer products; floating values below are diagnostics only.

## Result

| Frozen tuple | Geometric mean `L/S` | Strict wins | Worst `L/S` | Bootstrap 97.5% upper GM | Result |
|---|---:|---:|---:|---:|---|
| `(12,6,6,8)` | 0.827113535 | 8/8 | 0.839207568 | 0.833103673 | survive |
| `(16,8,8,8)` | 0.824496442 | 8/8 | 0.837357790 | 0.830102725 | survive |

The per-image exact ratios were:

| Image | `(12,6,6,8)` | `(16,8,8,8)` |
|---|---:|---:|
| nomao-saeki-33553 | 46638/56485 | 57663/70121 |
| martyn-seddon-220 | 46558/55694 | 57704/68912 |
| zugr-108 | 46435/56205 | 57385/69584 |
| jason-briscoe-149782 | 45316/54774 | 56250/68106 |
| martin-wessely-211 | 46216/55071 | 56939/68431 |
| stefan-kunze-26931 | 46394/55985 | 57263/69500 |
| vita-vilcina-3055 | 44917/55897 | 55711/69244 |
| philippe-wuyts-45997 | 46108/55498 | 56897/68924 |

All 100,000 bootstrap products were strictly below the 0.95 threshold for both tuples. The result
leaves only roughly 7--8 percentage points between the oracle and the required 0.90 actual-rate
margin. A real rank-8 binary linear logistic model, stronger factorized baselines, finite CDFs, and
termination bytes can readily consume that slack. Survival is therefore genuinely inconclusive.

## Integrity and lifecycle

The valid artifact is
`results/comp008_sgi_entropy_oracle_dev_v3_2026-07-16`. It contains eight bases and sixteen unique
QAT cells from one uninterrupted run invocation. The legal order was
`preflight -> acquire -> prepare-dev -> run-dev -> analyze -> replay`; no truncated journal,
staging repair, resumed cell, or imported predecessor result occurred.

Independent quantitative review cold-parsed all sixteen SSPL1 streams and reproduced every
absolute symbol, integer entropy certificate, byte total, ratio, exact bootstrap rank, and gate
without discrepancy. Independent lifecycle review verified the complete source archive, 30/30
bound sources, process coverage, canonical zlib streams, and captured confirmation-sealing
evidence. Production and independent decoded boundary tensors agreed exactly in every cell with
per-tensor maximum absolute difference `0.0`.

Canonical anchors:

```text
preflight binding       9c5dca682d490567a56ce4fc4043112a5bacf72b2fee908e95e86e691a1ea0f2
source snapshot         ee35792c...d74
targets manifest seal   0b08414bd2d72ff3b3e889807e9754b9454cf6f8e015f7f57df3a23f2fda57ab
cells.jsonl             9f14774f...ccb8b
analysis file           d729d64b...c104
analysis seal           3368a3f96926b0eaf2e18119ae587d4bb3822c450c0f6f5a33e883c4e87e6792
replay seal             95829d0c563842adcc0a7788d7ff02a8e6bd74596f80a3a3966d78c2da9b1397
```

The earlier v1/v2 development attempts remain explicit pre-pixel invalid artifacts. V1 failed the
required C++ ABI load. V2 measured its environment after CUDA had mutated module state and failed
closed before Pillow opened an image. Valid v3 used a fresh preflight and the later acquisition-v2
artifact.

## Diagnostics and research axes

- **Compression:** promising necessary-condition evidence only. No SSP2E bytes yet exist.
- **Quality:** unchanged by hypothesis; candidate and SSPL1 must decode identical symbols/tensors.
- **Ordinary convergence:** not tested by the entropy oracle. The shared base fits are inputs, not
  competing training methods. One difficult image's terminal PSNR was about 1.38 dB below its
  logged early peak, illustrating why terminal-field rate and convergence must not be conflated.
- **Entropy-model convergence:** not tested; the oracle fits exact histograms with no optimizer.
- **Performance:** no candidate decoder exists, so there is no speed or memory result.
- **Expressiveness:** exactly unchanged. Conditioning changes code length, not the decoded function
  class.

All sixteen QAT cells show the declared shipped-path mean-domain mismatch: 38--59 means lie outside
the image box, and the encoder's final mean extent differs from the QAT fake-quantizer domain. This
does not invalidate a bound computed from final cold-parsed symbols, but no result should call the
QAT mean lattice fully matched.

## Next experiment

Proceed only to COMP-009: implement the exact self-contained SSP2E arithmetic coder, deterministic
256-byte grid/100-byte head fitter, complete low-overhead and empirical factorized arithmetic
controls, and shuffled-position causal control. Count every byte, reproduce symbols exactly, use a
native decoder for resource measurements, and require both tuples. Confirmation remains sealed.

If actual SSP2E fails, do not enlarge or retune the same context model. The adversarially selected
materially different fallback is a complete-stream product/residual VQ assay with equal total
codebook capacity and all codebooks, group labels, indices, ranges, framing, and decoder work
counted. Generic VQ novelty is already occupied by methods such as
[GaussianImage](https://github.com/Xinjie-Q/GaussianImage/tree/d53393bee7c9fbb24e3510614e3ff2c85b8fbbc1)
and [RDO-Gaussian](https://arxiv.org/html/2406.01597); the defensible contribution would be rigorous
StructSplat-specific evidence, not a new VQ class.
