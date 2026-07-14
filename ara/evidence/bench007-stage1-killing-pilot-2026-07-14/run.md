# BENCH-007 Stage-1 actual-rate killing pilot

**Verdict:** complete negative gate result. Tensor-metric WSE does not satisfy the preregistered
promotion rule against the strongest direct control. Stage 2 is not authorized and was not run.
This development-set result closes the exact current compression claim; it is not positive or
negative held-out evidence.

## Frozen question and protocol

The pilot asked whether tensor density/orientation plus on-edge anisotropic WSE improves
self-contained SSPL1 rate-distortion over the strongest direct nonlearned allocation control when
renderer, fitter, codec, candidate search, and decoded pixels are shared. The frozen manifest
identity is `ad16c5df6888d5d76afdeb51cf1a297b898d3fa7bf3b8e9f2bed473cca748f83`.

- DIV2K train IDs: `0001, 0115, 0229, 0343, 0457, 0571, 0685, 0799`, native dimensions.
- Arms: tensor-WSE, quadtree-WSE, local SLIC/Sobel, local gradient, uniform Euclidean WSE, and
  seeded random.
- Target caps: `0.5` and `1.0` bpp; six frozen counts per image and four bit mixes.
- Independent 1,500-iteration fits, seed 0, no QAT, owned exact-CUDA normalized renderer.
- Rate: all SSPL1 bytes divided by original native pixels; selection is best measured PSNR under
  each exact integer byte cap.
- Inference: image-cluster bootstrap and the frozen strongest-control, quality, time, and mechanism
  rules in `tasks/BENCH-007-actual-rate-structure-phase-diagram.md`.

The manifest was frozen before metric inspection from Stage 0b's median
`8.614970513660953 B/G`. Its repository record is clean commit
`31837269aa892694c697b9d45c55d8bd78aa2374`; the final all-candidate revalidation snapshot is clean
commit `8e9b5b5e48abbd1758a6402c7ad1ac602bca40a8` on an RTX 4090 with PyTorch
`2.7.0+cu126`. The final analysis and visual bundle were regenerated from clean commit
`a6560761ca7a941c4b9b3c076679c44d9c92f728`.

## Completion and integrity

The latest-cell view is complete:

| Item | Complete | Failed or missing |
|---|---:|---:|
| Independent fits | 288 / 288 | 0 |
| SSPL1 codec candidates | 1,152 / 1,152 | 0 |
| Exact-cap selected rows | 96 / 96 | 0 |
| LPIPS selected rows | 96 / 96 | 0 |
| PNG/JPEG-444/AVIF-444 context rows | 184 / 184 | 0 |

All 1,152 latest candidates were re-encoded from their saved fitted fields at the clean snapshot.
Every persisted stream was byte-identical to its previous encoding; every in-memory/persisted
decoded-field hash matched; every maximum absolute decoded-field difference was exactly `0.0`
under the unchanged `1e-6` tolerance. Stream hashes are unique across all 1,152 cells.

### Validator incident and resolution

The first execution compared two independent exact-CUDA renders of identical decoded fields. CUDA
atomic accumulation order made 180 otherwise valid candidates differ by
`1.0132789611816406e-6` to `1.3709068298339844e-6`, just above the frozen tolerance. This was a
validator artifact, not stream corruption: the failures spanned all arms and the saved fits and
streams remained valid.

Commit `1b26cfaece25ecc72f58a99381d13bf07641798c` moved parity to the correct persisted-stream
decoded-field boundary, retained the tolerance, recorded both field hashes, and added
candidate-only revalidation. The append-only journals deliberately retain the 180 historical
failure rows; status and analysis use the latest row per candidate key, where all 1,152 are now
valid. Commit `97ecdbf8e5ea32b43196984eaa6044cabfc8aafa` subsequently fixed F5 replay of a CUDA-frozen
stream, and `a6560761ca7a941c4b9b3c076679c44d9c92f728` replaced F8's obscuring inset with a dedicated
decoder-latency panel.

## Preregistered result

The strongest direct control is the local gradient-weighted control, selected by the frozen mean
PSNR rule. Tensor-WSE shows a real low-rate effect but neither the required magnitude across rates
nor the required resource/mechanism profile.

| Gate component versus local gradient | Measured result | Required | Pass |
|---|---:|---:|:---:|
| PSNR at 0.5 bpp | `+0.3457 dB`, 95% CI `[+0.1426, +0.5662]` | at least `+0.25 dB`, lower CI > 0 | yes |
| PSNR at 1.0 bpp | `+0.0089 dB`, 95% CI `[-0.1718, +0.1738]` | at least `+0.25 dB`, lower CI > 0 | no |
| PSNR BD-rate | `-4.5417%`, 95% CI `[-6.9489, -2.0516]` | at most `-10%`, upper CI < 0 | no |
| Fit plus equal search time | `1.4752x`, 95% CI `[1.3970, 1.5510]` | at most `1.10x` | no |
| Edge-band MSE delta | `-2.9473e-4` | edge or bleed improves | yes |
| Signed bleed delta | `-2.8586e-3` | edge or bleed improves | yes |
| Texture MSE relative delta | `+7.2883%`, 95% CI `[+4.7190%, +9.8167%]` | at most `+5%` | no |

Therefore quality-at-both-targets, BD-rate magnitude, time, and the full mechanism guard all fail.
The executable gate returns `pass=false`, `stage2_authorized=false`, and “Stop: do not launch or
interpret Stage 2 as a rescue run.”

The result is not a claim that structure is useless. Tensor-WSE beats the weaker local SLIC/Sobel
control by `+1.4461 dB` at 0.5 bpp and `+1.3525 dB` at 1.0 bpp, and it beats uniform/random
placement by roughly 2--2.5 dB. But the preregistered comparison is the strongest direct control,
not a convenient one. Against quadtree-WSE, tensor-WSE is slightly lower in mean PSNR by
`0.0111 dB` at 0.5 bpp and `0.0621 dB` at 1.0 bpp. Against the gradient control, the low-rate
advantage disappears at 1.0 bpp, encoder cost rises 47.5%, and texture error exceeds the guard.

## Publication and research decision

The proposed tensor-WSE actual-rate method paper is not publication-ready and should not be
rescued by tuning on these eight images. The current method claim is closed; untouched DIV2K
validation remains untouched because the gate denied access. BENCH-008 and COMP-005 are not
authorized by this result.

The reusable output is still substantial: a tested target-rate/cold-stream benchmark, a direct
control matrix, exact resource and mechanism diagnostics, and a visually audited F5--F9 bundle.
Those artifacts could support a future benchmark or negative-results narrative only after an
independently justified scope and native external methods; this pilot alone is not held-out paper
evidence.

Further method research must begin as a new question with a new null and a disjoint development
screen. The measured clue worth carrying forward is explicitly conditional: tensor structure has
a 0.5-bpp edge/bleed benefit, but it is expensive, vanishes at 1 bpp, and trades against texture.
A legitimate next ideation pass may ask whether a materially different sparse-regime mechanism
can preserve that low-rate effect under a hard compute and texture guard. It must not tune the
current tensor-WSE formulation on these eight images or silently reuse Stage 2 as a rescue set.

## Reproduction

```bash
PYTHONPATH=src python -m benchmarks.actual_rate_phase_diagram run \
  --manifest results/bench007_stage1_20260714/manifest.json \
  --data-root results/datasets/DIV2K_train_HR \
  --outdir results/bench007_stage1_20260714 --device cuda \
  --retry-failed --revalidate-candidates

PYTHONPATH=src python -m benchmarks.actual_rate_phase_diagram conventional \
  --manifest results/bench007_stage1_20260714/manifest.json \
  --data-root results/datasets/DIV2K_train_HR \
  --outdir results/bench007_stage1_20260714

PYTHONPATH=src python -m benchmarks.actual_rate_phase_diagram analyze \
  --manifest results/bench007_stage1_20260714/manifest.json \
  --data-root results/datasets/DIV2K_train_HR \
  --outdir results/bench007_stage1_20260714 --device cuda
```

This evidence directory commits the frozen manifest/snapshot, latest candidate and selected tables,
fit journal, conventional table, analysis summary, and final F5--F9 figures. The much larger fitted
fields, candidate streams, reconstructions, histories, codec files, and portable local index remain
under ignored `results/bench007_stage1_20260714/`. These hashes bind both the committed compact
record and the retained local artifacts:

| Artifact | SHA-256 |
|---|---|
| `manifest.json` | `9427f581044e6a68ad2b9a4940d36e90c89a4cce20373e07b58db64d3334942a` |
| `run_snapshot.json` | `70542ecd3491f8219de84d13cdbbc52cb450b86a4deab7e8f31b591423067674` |
| `summary.json` / local `analysis/summary.json` | `89f4d1ab7f88c2e598a246c8c92641b7e841100d2052b242fc8eaa86e59ce99f` |
| `raw_candidates.csv` / local `analysis/raw_candidates.csv` | `2a85e8c4a1760179f4ced918c17648b6dd5bc50ff56e9e13fd8a208641d3d57d` |
| `selected.json` / local `analysis/selected.json` | `00d16c24732f9662135438f88dae13a204fb4c7d900279723fdf44be420ee1d2` |
| `selected.csv` / local `analysis/selected.csv` | `cf6d5759be47571c971e05c5cdc46137ef3d03f7302b952dfd1afb7db58d0c94` |
| Local `analysis/conventional.json` | `9497a385347655849991e3fee959d4a8cf1a87a4e56467d486853611fc5adbef` |
| `conventional.csv` / local `analysis/conventional.csv` | `750c52f7924697fd25eb0e918a0d97a8b8eed627207d995c74ffe1e629b31bb3` |
| `fits.jsonl` / local `journals/fits.jsonl` | `8625ea65ee5631027e6ef47bf614e72ef5f3a291bc63541790896abfa2e1e7b2` |
| Local `index.html` | `00a1a001b856dfb1a6bea491d0a640154ac28a2bf8f1f49d37bfe0be0474fdc9` |
| `figures/f5_causal_allocation.png` | `449c95e7e1039c41922d9ec1f1da7635541dc6669cdd05ac728d97df5db5783f` |
| `figures/f6_actual_rate_phase_diagram.png` | `2077eeddbcd90c27f773ac684abac2230ca38ba5b36c44290db6c8cab59e8e87` |
| `figures/f7_mechanism.png` | `8c82b116c199684a5edad4ba44281862069a3c672754e562d8a3ae7757507b75` |
| `figures/f8_resources.png` | `32d5d5983b7531ff2fed28cc1a2d9e401c040db51f5676f74bbe8edb00dba0c6` |
| `figures/f9_qualitative_quantiles.png` | `ae5949fa943203ef836eecb6bb28c1a18dc3f1800e545b8589af36bda8a51368` |
| `logs/stage1-revalidate-clean.log` | `1baad2493fdb7a7f65f1ef954a8d4e49188cfb3c9f91f87c532c0475a403e6af` |
| `logs/stage1-analyze-v3.log` | `4de91489f92e8b35a63ab12a5a72147fe1bda8e0d0be9c2288d658e273b54f76` |

All five figures were inspected at full bundle scale. F5/F6/F7/F9 were accepted as generated; F8
was revised to separate encoder cost, decoder latency, and equal-horizon trajectories, then
regenerated and accepted. Final verification on the completed branch passed `ruff check .` and
`PYTHONPATH=src pytest -q`: 468 tests passed with one known CUDA-extension architecture-list
warning in 10.82 seconds.
