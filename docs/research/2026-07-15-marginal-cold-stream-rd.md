# COMP-006 marginal cold-stream rate--distortion attribution

**Date:** 2026-07-15
**Decision:** stop the frozen standard-birth/cold-SSPL1 lineage
**Task:** [`COMP-006`](../../tasks/COMP-006-marginal-cold-stream-rd.md)
**Benchmark:** [`benchmarks/marginal_cold_stream_rd.py`](../../benchmarks/marginal_cold_stream_rd.py)
**Primary artifact:** `results/comp006_marginal_rd_dev_v1_2026-07-15/`
**Exact replay:** `results/comp006_marginal_rd_dev_v1_replay_2026-07-15/`

## Executive result

One extra residual-placed standard Gaussian is not the best use of a small SSPL1 byte allowance in
this experiment. At the preregistered `matched no-edit + 16 bytes` cap and after 20 fresh-Adam QAT
steps, the best of 16 births lost `-1.0714 dB` mean paired PSNR to the strongest feasible union of
no-edit, 16 matched birth-for-death replacements, and 875 global precision mixes. The median was
`-0.9533 dB`, the family-stratified target-bootstrap 95% interval was
`[-1.2873, -0.8417] dB`, and every family mean was negative. All integrity and replay checks pass,
so the result is final and the confirmation split remains untouched.

Actual rate nevertheless matters. The complete-stream and nominal-raw-bit oracles selected the
same exact row in only 14/36 cells. Complete-stream selection recovered `+0.2131 dB` mean PSNR over
the proxy selection among these search candidates, but it changed the broad action class in only
2/36 cells and did not create a structural-birth advantage. The reusable result is an operational
RD audit, not a production allocator.

## State of the art and what was reused

COMP-006 is an application of established rate--distortion ideas. Its contribution boundary is
repository-specific evidence and fail-closed infrastructure, not a new coding principle.
The broader renderer, allocation, growth, amortization, structured-coding, richer-primitive, and
learned-codec map is in the
[`frontier reuse report`](2026-07-15-frontier-reuse-experiments.md); this section isolates the
lineage needed to interpret E4.

| Lineage | How the method works | Relation to COMP-006 |
|---|---|---|
| Operational RDO | An encoder enumerates realizable modes, measures coded rate and distortion, and minimizes `D + lambda R` or selects the lowest distortion below an integer cap. The relevant rate is the realized conditional code length, including syntax and side information. See [Sullivan and Wiegand, 1998](https://www.microsoft.com/en-us/research/publication/rate-distortion-optimization-for-video-compression/). | Every heterogeneous action is fully encoded and selected under an exact integer cap. There is no learned local price or deployable search policy. |
| EBCOT and tree RDO | EBCOT emits embedded code-block streams with valid truncation points, then allocates bytes across blocks by distortion-reduction/rate slopes. Tree coders compare parent and descendant representations and optimize a subtree by Lagrangian pruning or dynamic programming. See [Taubman, 2000](https://doi.org/10.1109/83.847830), [Chou, Lookabaugh, and Gray, 1989](https://doi.org/10.1109/18.32124), and [Ramchandran and Vetterli, 1993](https://doi.org/10.1109/83.217221). | Post-encode selection is analogous. SSPL1 candidates are independent complete containers, not nested truncations or an additive tree, so their local slopes cannot be composed. |
| Sparse approximation and MDL | Matching pursuit proposes residual-explaining atoms. Compression-aware variants charge atom identity, geometry, coefficient, quantizer, order, and support. MDL accepts added structure only when its distortion reduction repays model plus residual description length. See [Mallat and Zhang, 1993](https://doi.org/10.1109/78.258082), [Ryen, Schuster, and Katsaggelos, 2004](https://doi.org/10.1109/TSP.2004.826184), and [Barron, Rissanen, and Yu, 1998](https://research.ibm.com/publications/the-minimum-description-length-principle-in-coding-and-modeling). | Residual births are greedy atom proposals; birth-for-death reallocates fixed support; full SSPL1 length is an empirical description length. The benchmark does not solve global subset selection or prove an MDL result. |
| Learned image codecs | A transform produces quantized latents; hyperpriors and autoregressive, hierarchical, or graph contexts predict symbol distributions; arithmetic coding realizes those rates; training minimizes expected `R + lambda D`, including side latents. See the [scale hyperprior](https://research.google/pubs/variational-image-compression-with-a-scale-hyperprior/), [joint hierarchical/autoregressive prior](https://papers.nips.cc/paper_files/paper/2018/hash/53edebc543333dfbf7c5933af792c9c4-Abstract.html), [HPCM](https://openaccess.thecvf.com/content/ICCV2025/html/Li_Learned_Image_Compression_with_Hierarchical_Progressive_Context_Modeling_ICCV_2025_paper.html), and [GLIC](https://openaccess.thecvf.com/content/CVPR2026/html/Chen_Adaptive_Learned_Image_Compression_with_Graph_Neural_Networks_CVPR_2026_paper.html). | The insistence on actual rate and side information transfers. These trained, heavy-decoder codecs are outside-class context, not common-protocol SSPL1 baselines. |
| Gaussian representation/entropy co-design | [GaussianImage](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/1421_ECCV_2024_paper.php) directly fits 2D Gaussians and adds vector quantization. [GaussianImage++](https://ojs.aaai.org/index.php/AAAI/article/view/37572) couples distortion-driven densification and context-aware filtering with attribute-separated learnable scalar quantizers and QAT. Adjacent 3D [HAC](https://eccv.ecva.net/virtual/2024/poster/1306) predicts quantized attributes from anchors and a hash-grid context. [SGI](https://openaccess.thecvf.com/content/CVPR2026/papers/Pan_SGI_Structured_2D_Gaussians_for_Efficient_and_Compact_Large_Image_CVPR_2026_paper.pdf) uses seeds and small MLPs to generate local Gaussian groups, then entropy-codes seed attributes with binary hash-grid context. | These methods show that regularity saves bytes only when decoder structure and an entropy model exploit it. COMP-006 tests direct standard rows in SSPL1/zlib; it does not test generators, learned contexts, VQ, or a richer atom. |

The reused mechanism is therefore: propose sparse structural edits, charge every complete cold
container byte in the MDL/RDO spirit, compare against fixed-count support reallocation and global
precision allocation, then choose under an integer cap. The search-bounded novelty label is
**known RDO/MDL principles; useful new evidence in this explicit normalized 2D-Gaussian codec**.

## Frozen experiment

The target manifest contains six procedural families with six deterministic `64x64` variants.
Even variants `{0,2,4}` form 18 development targets; odd variants `{1,3,5}` were hash-frozen as
confirmation and never fit or scored. Seeds `{0,1}` are repeated measurements and are averaged
before target-level inference.

For each target and seed, the benchmark initializes and fits one N=64 no-opacity constant-RGB
field, applies 40 base-QAT steps, serializes it at bits `(12,6,6,6)`, persists it, and starts every
branch from the cold-decoded field with fresh Adam state. It creates one frozen bank of 16
residual-local-maximum rows. `birth_i` appends row `i`; `replace_i` removes one fixed lowest-
activity donor and appends that identical row. No-edit, birth, and replacement are evaluated at
recovery steps 0 and 20. The step-20 no-edit field is also encoded at all 875 precision mixes with
means bits `10..16` and scale, rotation, and color bits independently `4..8`.

Rate is `len(complete counterfactual SSPL1) - len(matched no-edit SSPL1)`. It includes the header,
ranges, framing, every compressed stream, and all zlib/Morton context changes. This is not an
incremental patch cost, additive row price, Shapley attribution, or sequential-edit budget.
Selection uses cold-decoded MSE; PSNR, SSIM, and MS-SSIM are reported secondarily.

## Integrity and replay

| Check | Result |
|---|---:|
| Development cells | 36/36 |
| Complete cold streams | 33,840/33,840 |
| Cell/stream counts in replay | 36 / 33,840 |
| Protocol/source/cells/streams/analysis replay checks | all exact |
| Odd-variant fitted cells | 0 |
| Primary decision | stop |

Both runs used deterministic single-thread CPU execution. Every stream was persisted, hashed,
cold-read, decoded, rendered, and checked against header, cardinality, finite-value, component-
partition, target, action, source, and protocol invariants. Replay timing was excluded from exact
comparison; all scientific fields and stream hashes match.

Frozen hashes:

- target manifest: `a1ea1ea5be41e36c3e4a8557d01ce721167545d680a6945522c935b832f60f0e`;
- precision grid: `c92e2fbb773e955b5c2b60a18592cbc9f012a6cc025a0f7d0578b90a49c42ab4`;
- protocol: `137359fbe8447dad5e585d27f0b2b1fe58bc8ec89b0f31e92cd37c2fa543acf2`;
- source snapshot: `e92c6aed8b57bc4382fb0ebc452bdecf2ef41c0d1521a0b87654695c0d20e175`;
- normalized environment: `efa05bc74f417a66e4c32a586649a6b102a3ab91fe4336be99c378b59ec71d47`.

## Preregistered result

| Gate | Requirement | Result | Pass |
|---|---:|---:|:---:|
| Feasible target/seed cells | at least 90% | 36/36 | yes |
| Mean target-level birth advantage | at least +0.15 dB | -1.0714 dB | no |
| Family-bootstrap lower bound | greater than 0 | -1.2873 dB | no |
| Median target-level birth advantage | at least +0.10 dB | -0.9533 dB | no |
| Positive family means | at least 4/6 | 0/6 | no |
| Integrity and exact replay | all pass | all pass | yes |

| Family | Birth minus strongest-control PSNR |
|---|---:|
| Annular sectors | -1.2658 dB |
| Bezier strokes | -0.3920 dB |
| Curved step | -1.4032 dB |
| Hard disks | -1.3664 dB |
| Quadratic patch | -0.5666 dB |
| Three-region junction | -1.4346 dB |

The control did not win because birth was inert. Relative to the matched no-edit field, selected
birth improved PSNR by `+0.9267 dB` on average, while the strongest control improved it by
`+1.9982 dB`. Birth also trailed control by `-0.001401` SSIM and `-0.000865` MS-SSIM. The issue is
opportunity cost under complete-stream rate, not an absence of residual signal.

## Exploratory failure analysis

These diagnostics were computed after the frozen decision and cannot redefine its gate.

At the primary cap, precision reallocation was the strongest control in 23/36 cells and matched
replacement in 13/36. After allowing birth into the overall competition, the exact winner was
precision in 22 cells, replacement in 9, and birth in 5. Of the 23 selected precision controls,
19 reduced means to 10 bits and 17 raised colors to 8 bits. On those cells the mean complete-stream
component changes relative to base were `-43.96` means bytes, `+7.39` scale bytes, `+1.87`
rotation bytes, and `+37.57` color bytes, for only `+2.87` total bytes. The codec frequently gained
more by moving precision from geometry to appearance than by transmitting another row.

Selected birth streams cost `+9.67` bytes on average, ranging from `+1` to `+16`. Selected controls
cost `+2.64` bytes on average, ranging from `-25` to `+15`; 13 control deltas were negative because
independent zlib contexts are non-monotone. That is precisely why ratios and a fixed per-row byte
price would be invalid here.

The nominal-raw-bit and actual-byte oracles agreed on the exact row in 14/36 cells (`38.89%`) but
on the broad branch in 34/36 (`94.44%`). All proxy winners were also feasible under the actual cap.
Actual-byte selection improved PSNR over proxy selection by `+0.2131 dB` on average, with a
maximum of `+0.9512 dB`; almost all disagreement was which precision mix to use, not whether to
grow the field. This supports exact-RD selection infrastructure while rejecting a stronger claim
that count-proxy failure makes standard birth competitive.

| Recovery/cap | Feasible cells | Mean birth advantage | Birth exact winners |
|---|---:|---:|---:|
| step 0, +16 bytes | 36/36 | -1.5868 dB | 6/36 |
| step 20, +8 bytes | 33/36 | -1.4430 dB | 2/36 |
| step 20, +16 bytes | 36/36 | -1.0714 dB | 5/36 |
| step 20, +32 bytes | 36/36 | -1.0804 dB | 5/36 |

Twenty recovery steps narrow the primary deficit by about `0.52 dB`, but neither the recovery nor
the preregistered cap sensitivities reverse it. The +8 value is conditional on its 33 feasible
cells; only +16 is inferential.

Timing remains descriptive. Deduplicated median 20-step recovery was `0.2768 s` for birth and
`0.2752 s` for replacement; median encode time was `0.395 ms` and `0.394 ms`, respectively.
Shards ran concurrently and this is an exhaustive CPU audit, so these near-equalities are not a
controlled performance result or speed claim.

## Requested-axis verdict

| Axis | Evidence | Decision |
|---|---|---|
| Quality | Birth improves no-edit, but the matched control improves about twice as much and wins by 1.0714 dB. | No promoted quality method; retire one-more-standard-row as the next lever. |
| Convergence | Fresh 20-step QAT narrows but does not close the deficit; only steps 0 and 20 were tested. | No convergence claim. Any continuation study needs a new hypothesis, optimizer-state control, longer curves, and disjoint targets. |
| Performance | Single-thread CPU timings describe this exhaustive audit, not a kernel, renderer, or deployable selector. | Not tested; continue exact per-Gaussian backward/fusion profiling independently. |
| Compression | Exact bytes materially change fine-grained selection, but standard birth loses to precision/replacement across every family. | Stop this structural-birth formulation; keep the exact-byte oracle for new codec hypotheses. |
| Expressiveness | Every action uses the same constant-RGB Gaussian schema; SSPL1 rejects affine or frequency-bearing atoms. | Not tested. A new versioned codec and equal-actual-byte atom suite are required. |

## Claim boundary and limitations

- This rejects only N=64 residual-local-maximum standard births under SSPL1/zlib, a +16-byte cap,
  20 fresh-QAT steps, and six small procedural families. It does not reject structural compression,
  natural-image densification, moment splits, opacity fields, generators, learned entropy models,
  or other codecs.
- Replacement is an oracle control over 16 candidates with one fixed lowest-activity donor, not a
  promoted policy. Precision candidates intentionally received no per-mix QAT, making that control
  weaker; losing to it is decisive against birth but does not estimate the best precision codec.
- This is not optimizer continuation, long-horizon convergence, speed, peak memory, support work,
  decoder latency, natural-image population performance, or image-compression SOTA evidence.
- The search oracles are expensive and non-sequential. Complete-container deltas may be negative
  and cannot be added to predict a multi-edit stream.
- GaussianImage++, SGI, HAC, HPCM, GLIC, and conventional codecs use different representations,
  training data, resolutions, decoders, rates, and objectives. They are mechanistic context, not
  head-to-head baselines.

## Post-replay hardening

An independent review found no result blocker. After the exact replay decision was frozen, the
current benchmark was hardened to recompute row keys, cross-bind stream/cell/parent/action
metadata, bind environment provenance on resume and replay, include the FIT-020 disjointness
dependency in future source snapshots, and replace the preliminary human overview with the final
replay decision. These changes do not touch targets, candidates, fitting, QAT, encoding, metrics,
selection, gates, or protocol hash.

The hardened validator rechecked all 33,840 rows in both frozen runs and reproduced the same final
summary; no scientific cells were rerun. The frozen evidence remains bound to source SHA-256
`e92c6aed8b57bc4382fb0ebc452bdecf2ef41c0d1521a0b87654695c0d20e175`, while the reviewed
post-evidence tree has combined relevant-source SHA-256
`3a4f1ff6a39029409afb188e8c6d1dbaff43f3889d92b95a8705726e518f274e`. Ruff, 12 focused tests,
and the full 552-test suite pass on the hardened tree.

## Reproduction

The primary and replay used the same three-shard sequence with different output directories:

```bash
PYTHONPATH=src:. python -m benchmarks.marginal_cold_stream_rd run \
  --outdir results/comp006_marginal_rd_dev_v1_2026-07-15 \
  --split development --shard-index 0 --num-shards 3
# Run shard-index 1 and 2 with the same arguments, in parallel or sequentially.

PYTHONPATH=src:. python -m benchmarks.marginal_cold_stream_rd finalize-run \
  --outdir results/comp006_marginal_rd_dev_v1_2026-07-15 \
  --split development --num-shards 3
PYTHONPATH=src:. python -m benchmarks.marginal_cold_stream_rd analyze \
  --outdir results/comp006_marginal_rd_dev_v1_2026-07-15 --split development

PYTHONPATH=src:. python -m benchmarks.marginal_cold_stream_rd verify-replay \
  --primary results/comp006_marginal_rd_dev_v1_2026-07-15 \
  --replay results/comp006_marginal_rd_dev_v1_replay_2026-07-15
```

The primary `final_summary.json` is the final decision record; `replay_comparison.json` contains the
exact checks. Raw `cells.jsonl`, `streams.jsonl`, persisted `.sspl` files, source snapshot,
configuration, target manifest, selections, aggregates, and portable `index.html` remain in the
ignored result artifact.

## Decision and next work

Do not add a marginal-byte birth score, search oracle, or structural-birth default to production.
Do not tune cap, actions, donor, horizon, bit grid, or the exposed development targets, and do not
score confirmation. ADR-0016 records this boundary.

Two independent paths remain scientifically clean. The performance path is exact per-Gaussian
backward accumulation/fusion with forward/backward parity and end-to-end profiling. The
compression/expressiveness path is a real versioned codec for compact luminance slope/affine and a
WIPES-like carrier, evaluated against constant-RGB birth, replacement, and per-mix-QAT precision
at equal complete bytes and decode work. The latter is a new grammar and task, not a COMP-006
rescue. Within the existing grammar, the evidence points first to attribute/rate co-design: real
context/range coding, per-group learnable quantization, and `R + lambda D` optimization, always
measured as a complete cold stream.
