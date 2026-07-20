# Target-free implicit RGB-lattice feasibility proof

## Status and claim boundary

**Date:** 2026-07-16
**Status:** deterministic synthetic mechanics evidence; not a StructSplat method result

This proof asks one deliberately narrow question before spending another protected-data screen:
can a color representation with no explicit codebook ever beat the exact representation mechanics
of the frozen COMP-011 flat/RVQ menu? The answer is **yes on some source distributions and no on
others**. That is enough to retain an implicit-lattice/VQ union as a conditional next direction, but
not enough to change COMP-011 or claim improved real-image compression.

The experiment opened no COMP-011 stream, target, development payload, or confirmation payload. It
does not select a COMP-011 arm, alter a frozen task, authorize a data run, measure renderer quality,
or measure a complete SSPL/SSP2V file. Its conclusions are restricted to synthetic RGB symbol SSE
and syntax-specific representation bytes.

Implementation and tests:

- `benchmarks/implicit_lattice_rgb_proof.py`
- `tests/test_implicit_lattice_rgb_proof.py`
- command: `python -m benchmarks.implicit_lattice_rgb_proof --fixture correlated --fixture palette
  --fixture uniform`
- deterministic evidence SHA-256:
  `5f1585b57cae90be9fa92cdfb0a3dedeb6d082d6595446ab23eb670b36d9260a`

## Mechanism and byte accounting

The explicit control is the production COMP-011 eight-arm menu: flat or residual VQ, two or three
matched stages, and base `K` of 64 or 128. It uses the native Lloyd fitter, production SSP2V
descriptor/codebook/model serialization, and the native arithmetic coder.

The implicit control uses either RGB or the exactly reversible integer YCoCg-R lift, one integer
step per coordinate, deterministic nearest-coordinate rounding, an empirical positive-frequency
model, and one native arithmetic stream per coordinate. RGB scalar quantization is a cubic
lattice; the lifted YCoCg-R construction is more precisely a structured implicit scalar
quantizer, not a general SALVQ-style Euclidean lattice. The tested steps are target-blind and
finite: RGB `(s,s,s)` and YCoCg-R `(s,2s,2s)` for `s in {1,2,4,8,16,32}`. It performs no iterative
fit and stores no codebook.

Both sides charge all representation-specific bytes:

- exact 20-byte directory entries;
- exact descriptors (26 bytes for the hypothetical `ILQ1` lattice syntax);
- serialized explicit codebooks where applicable;
- compressed empirical frequency models; and
- actual native arithmetic payloads.

Common outer framing and the unchanged mean, scale, rotation, and geometry payloads are excluded
because they cancel exactly in this feasibility comparison. Thus these are not complete-stream
rates. The hypothetical lattice descriptor is fully charged but has not been integrated into or
decoder-benchmarked as a shipping format.

## Deterministic fixtures

Each fixture contains 8,192 RGB rows generated from a domain-separated SHA-256 construction:

| Fixture | RGB SHA-256 | Unique RGB rows | Intended stress |
|---|---|---:|---|
| correlated | `5c16fe619aec37e49dbf8e29ab1d6a8e467a6c31d0eb678c279c913ea18a003a` | 8,019 | strong Y/Co/Cg correlation |
| palette | `94df89cfc735d982b4a630b4e5f44955c2ec0e8ada47a727b86081f48d135332` | 7,435 | clustered colors with small jitter |
| uniform | `072ca733f038da0cee0db9eb946ade0ca9640d2af2dc348ab97b23db686b8853` | 8,192 | unstructured full-RGB support |

The palette fixture is an adversarial control, not a favorable showcase.

## Results

Selected matched operating points expose both the opportunity and its limit:

| Fixture | Implicit point | Bytes | SSE | Explicit comparison | Bytes | SSE | Relative bytes | Relative SSE |
|---|---|---:|---:|---|---:|---:|---:|---:|
| correlated | YCoCg `(1,2,2)` | 17,609 | 7,865 | RVQ, 3 stages, K=64 | 19,012 | 8,053 | -7.380% | -2.335% |
| correlated | YCoCg `(4,8,8)` | 11,377 | 107,887 | flat, 3 stages, K=128 | 12,031 | 124,148 | -5.436% | -13.098% |
| uniform | RGB `(16,16,16)` | 12,712 | 521,467 | RVQ, 2 stages, K=64 | 13,152 | 631,604 | -3.345% | -17.438% |
| uniform | RGB `(8,8,8)` | 15,751 | 135,005 | RVQ, 2 stages, K=128 | 15,855 | 247,153 | -0.656% | -45.376% |
| palette | RGB `(4,4,4)` | 18,082 | 35,962 | RVQ, 3 stages, K=64 | 18,127 | 0 | -0.248% | worse (nonzero vs exact) |

On the correlated fixture, the implicit YCoCg representation strictly dominates explicit VQ at
two useful rate regions. On uniform RGB, a plain scalar lattice also wins selected matched points.
On the palette fixture, RVQ reconstructs every row exactly for only 45 additional bytes; the
implicit point cannot dominate it. The joint Pareto frontier likewise contains implicit and
explicit points rather than a universal winner.

The result is therefore not “lattices beat VQ.” It is evidence for a source-distribution-dependent
union: implicit lattices avoid codebook cost and optimization when colors occupy a simple regular
coordinate structure, while explicit VQ efficiently represents irregular clusters.

## Convergence and performance diagnostics

The implicit controls terminate after direct transform, rounding, frequency construction, and
coding; they have no Lloyd initialization, local optimum, or convergence failure mode. That is a
mechanical convergence advantage, not evidence about end-to-end fitting convergence.

In one nongating local run, all eight explicit points versus all twelve implicit points took:

| Fixture | Explicit VQ fit+code | Implicit transform+code |
|---|---:|---:|
| correlated | 2.149 s | 0.176 s |
| palette | 1.385 s | 0.246 s |
| uniform | 4.105 s | 0.251 s |

Wall time is intentionally excluded from the evidence digest. These single-run menu totals are
diagnostic only: they do not isolate decoder cost, complete-stream I/O, GPU behavior, or robust
latency statistics. The current proof tests also do not execute and seal the full proof output;
the recorded digest was reproduced locally but is not bound to a Python, NumPy, zlib, compiler, or
native-source identity. Those bindings are required before promoting this to a formal experiment.

## Relation to prior art

[SALVQ](https://arxiv.org/html/2509.13482) is the direct transfer source: an implicitly generated
lattice replaces an explicit codebook and entropy-codes integer lattice coordinates. The present
proof does not claim that mechanism as novel. It tests whether its central economic prediction can
survive in StructSplat's much smaller, post-hoc three-dimensional RGB recipient.

The reversible color lift is also established codec machinery. Cross-color prediction in
[lossless point-cloud attribute coding](https://arxiv.org/abs/2303.12917) and
[MPEG G-PCC](https://www.mpeg.org/standards/MPEG-I/9/) make clear that YCoCg/component coupling is
a control or transfer, not a novelty claim.

## Frozen conditional direction and stop rules

COMP-011 remains first. Only if its quality-qualified explicit arms have competitive index payloads
but lose because codebook/model/framing bytes bind should a new, disjoint development task test the
following fixed union:

1. a small preregistered family of implicit RGB/reversible-color lattices;
2. the strongest explicit VQ controls without retuning their menu; and
3. either complete-byte oracle selection (an upper bound) or a separately frozen, target-blind
   selector based only on source statistics and charged syntax.

The useful hypothesis is that simple statistics such as occupancy, residual entropy, or
codebook-byte share can select the appropriate representation family. That selector must not see
render targets or confirmation data. Its own identifier, parameters, and directory/descriptor
costs must be charged.

Stop without expanding the family if any of the following holds on the new development split:

- the implicit arm does not improve complete bytes at a fixed conservative quality qualification;
- its apparent gain disappears after full framing, model precision, decoder work, and transform
  metadata are charged;
- a target-blind selector cannot recover most of the oracle union's gain robustly; or
- the benefit requires per-image step rescue, an unpriced learned basis/codebook, or target-guided
  choice.

Do not run this branch when COMP-011 fails quality rather than explicit-model cost. That outcome
belongs to the separately frozen renderer-Gram distortion branch.

### Draft of the smallest target-blind union screen

This is an adversarially reduced draft, not a frozen task. Trigger it only if a valid COMP-011 run
passes its quality, convergence, and resource requirements but fails rate, and subtracting the
selected arm's exact codebook payload plus its 20-byte directory entry would make the original
rate gate pass. Any other COMP-011 outcome forbids this screen.

Use eight new, metadata-selected development identities and reserve eight disjoint identities
without extracting them. For each bit tuple, freeze an explicit anchor from the old COMP-011
development result: the minimum-geometric-mean-byte arm that definitely qualified on all eight
images, with descriptor order breaking ties. Test exactly the eight unchanged explicit arms plus
the twelve RGB/YCoCg-R points above.

Before target import, calculate for every candidate its RGB source-symbol SSE, maximum absolute
component error, absolute signed-error sum per channel, and clipping component/row counts. A
candidate is eligible only if it is no worse than the frozen explicit anchor on every guard. Then
seal:

- `E_i`: the smallest complete stream among eligible explicit candidates; and
- `U_i`: the smallest complete stream among all eligible explicit and implicit candidates.

Tie order is explicit before implicit, then frozen arm order, then unsigned blob bytes. No learned
threshold, occupancy rule, entropy diagnostic, renderer output, or target metric may affect this
choice. After target import, an oracle `O_i` may be computed only as a diagnostic: the smallest
union candidate that definitely passes target quality.

In addition to every original COMP-011 endpoint, require per tuple:

- geometric-mean `U/E <= 0.98`, at least four of eight strict wins, and a frozen-bootstrap 97.5th
  percentile below one;
- positive oracle headroom and at least 80% recovery of its log-rate gain, encoded without
  floating ambiguity as `product(E_i) * product(O_i)^4 >= product(U_i)^5`; and
- total 20-arm encoder search wall time/RSS and selected cold-decode wall time/RSS no more than
  1.20 times their explicit controls.

Both bit tuples must pass. Close the fixed union if oracle headroom is below 2%, the source-only
selector misses quality in any cell, fewer than four strict wins occur, exact framing erases the
gain, resource limits fail, or success needs a new step, learned basis, target-derived threshold,
per-image rescue, or unpriced side information. Generic novelty is explicitly disclaimed; the
possible contribution is recipient-specific complete-stream evidence under StructSplat's
normalized renderer.

## Axis conclusions

| Axis | Evidence from this proof |
|---|---|
| Quality | no renderer or target-quality evidence; exact synthetic RGB SSE only |
| Convergence | implicit scalar lattices are deterministic and iterative-free; no end-to-end convergence claim |
| Performance | promising single-run encoder diagnostic; no robust encoder/decoder benchmark |
| Compression | measured syntax-specific dominance on two synthetic regimes, with a decisive palette counterexample; no integrated container, cold decode, or complete-stream gain |
| Expressiveness | a union is more distributionally expressive as a codec family, but decoded StructSplat primitives are unchanged |

The hard lesson is simple: removing a codebook can be valuable, but only when the source geometry
is regular enough that the missing codebook was not carrying essential irregular structure.
