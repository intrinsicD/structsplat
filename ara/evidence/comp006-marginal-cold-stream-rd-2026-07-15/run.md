# COMP-006 marginal cold-stream RD development decision

## Scope

- Frozen task: `tasks/COMP-006-marginal-cold-stream-rd.md`
- Primary: `results/comp006_marginal_rd_dev_v1_2026-07-15/`
- Exact replay: `results/comp006_marginal_rd_dev_v1_replay_2026-07-15/`
- Split: development only; odd variants `{1,3,5}` remained unfit and unscored
- Independent unit: 18 targets from six procedural families; seeds `{0,1}` were averaged repeats
- Primary comparison: best of 16 standard births versus no-edit, 16 matched fixed-donor
  replacements, and 875 precision mixes after 20 fresh-QAT steps at base stream +16 bytes

## Frozen evidence

| Item | Primary | Replay |
|---|---:|---:|
| Cells | 36 | 36 |
| Persisted/cold-validated streams | 33,840 | 33,840 |
| Protocol/source/cells/streams/analysis | exact | exact |

- Target manifest SHA-256:
  `a1ea1ea5be41e36c3e4a8557d01ce721167545d680a6945522c935b832f60f0e`
- Precision-grid SHA-256:
  `c92e2fbb773e955b5c2b60a18592cbc9f012a6cc025a0f7d0578b90a49c42ab4`
- Protocol SHA-256:
  `137359fbe8447dad5e585d27f0b2b1fe58bc8ec89b0f31e92cd37c2fa543acf2`
- Frozen relevant-source SHA-256:
  `e92c6aed8b57bc4382fb0ebc452bdecf2ef41c0d1521a0b87654695c0d20e175`
- Environment SHA-256:
  `efa05bc74f417a66e4c32a586649a6b102a3ab91fe4336be99c378b59ec71d47`

## Preregistered result

| Gate | Result | Pass |
|---|---:|:---:|
| Feasible cells at least 90% | 36/36 | yes |
| Mean birth advantage at least +0.15 dB | -1.071442 dB | no |
| Family-bootstrap lower bound above zero | CI `[-1.287276, -0.841740] dB` | no |
| Median birth advantage at least +0.10 dB | -0.953260 dB | no |
| At least four positive family means | 0/6 | no |
| Integrity and exact replay | all checks true | yes |

Final decision: **stop**. Confirmation was not authorized or run. All 18 seed-averaged target
advantages and all six family means were negative.

## Descriptive diagnostics

- Strongest control: precision in 23/36 cells; replacement in 13/36.
- Exact overall winner: precision 22, replacement 9, birth 5.
- Selected birth improved no-edit by +0.926749 dB; the strongest control improved by
  +1.998191 dB.
- Exact-byte/raw-bit row agreement was 14/36. Exact selection gained +0.213081 dB mean PSNR over
  the raw-bit oracle, but all 22 disagreements were within control allocation; branch class agreed
  in 34/36 and structural-birth selection never changed.
- Nineteen of 23 selected precision controls used 10 mean bits and 17/23 used 8 color bits,
  identifying a post-hoc geometry-to-appearance precision reallocation pattern.
- Selected complete-stream birth deltas averaged +9.67 bytes. Controls averaged +2.64 bytes and
  were negative in 13 cells, confirming that these zlib-container counterfactuals are non-additive
  and cannot be interpreted as incremental row prices.

## Review and validation

Independent quantitative, literature/novelty, and implementation/reproducibility audits found no
blocking defect. After the exact result was frozen, non-scientific validation hardening added row-
key recomputation, stream/cell/parent/action cross-binding, environment resume/replay binding,
future transitive-source capture, and a final human overview. The hardened analyzer revalidated
all rows in both frozen runs without rerunning scientific cells. Current relevant-source SHA-256 is
`3a4f1ff6a39029409afb188e8c6d1dbaff43f3889d92b95a8705726e518f274e`.

- Ruff: pass
- Focused COMP-006 tests: 12 passed
- Full suite: 552 passed
- Final record: `results/comp006_marginal_rd_dev_v1_2026-07-15/final_summary.json`
- Detailed interpretation: `docs/research/2026-07-15-marginal-cold-stream-rd.md`
- Architecture decision: `docs/adr/0016-keep-marginal-cold-stream-birth-benchmark-only.md`

## Claim boundary

This refutes only the frozen standard constant-RGB birth formulation under complete SSPL1/zlib
bytes, N=64 procedural parents, a +16-byte primary cap, and 20 fresh-QAT steps. It supports exact
rate accounting as precision-allocation audit infrastructure. It does not establish an additive
local price, deployable selector, speedup, natural-image result, learned-codec comparison, richer-
atom result, or image-compression SOTA claim.
