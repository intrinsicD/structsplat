# CORE-019 coherent-depth successor portfolio

Date: 2026-08-07

Scope: construction/pre-registration evidence only; no novelty, quality, speed, compression, or
commercial-use claim

## Failure being isolated

The current packet appearance is useful, but all three prior geometry units fail visually:
CompactCarve emits broad interior volumes, the first-alpha shell retains directional duplicate
silhouettes, and independently matched ray modes retain high posterior entropy and smeared
geometry. Covariance reconciliation and longer/common optimization do not repair the step-zero
surface. The successor must therefore introduce spatial coherence before Gaussian placement.

## Candidate families

| Family | Spatial coupling | Calibration use | Main risk | Disposition |
|---|---|---|---|---|
| Tune CORE-018 posterior/reciprocity | Independent rays | Search only | Consumed-scene threshold rescue; no coherent state | Rejected |
| Per-view monocular depth + scale | Within one image | Per-view affine fit | Scale seams and cross-view doubles | Control only |
| Classical packet plane sweep/SGM | Local image lattice | Exact cameras | Repeated/low texture and large bounded search | Valid future control |
| Dense correspondence/track graph | Sparse cross-view tracks | Exact cameras | Coverage holes; earlier precision failures | Anchor control |
| Feed-forward coherent multiview depth | Joint four-view field | Sim(3) to exact cameras | Large prior, domain/license dependence | Selected preflight |
| Optimize a neural scene representation first | Global via training | Exact cameras | Violates fast initialization and blurs causal test | Rejected |

The selected composition is not a new estimator. Its irreducible change is that depth is predicted
as a spatially coherent multiview field, then known calibration—not predicted cameras—owns every
output ray. Projective occlusion tests, continuous micro-contraction, and feature-adaptive WSE are
downstream compilation steps into the existing realtime-gs primitive.

## Pinned dependency and verified preflight

The official VGGT source is pinned at commit
`a288dd0f14786c93483e45524328726ab7b1b4ce`. The public `facebook/VGGT-1B` research checkpoint is
pinned at revision `860abec7937da0a4c03c41d3c269c366e82abdf9`, byte count `5,026,367,224`,
SHA-256 `f164acf60724910d8fe1578bb499d800850c7bb0948db7555c413f9fbe60467e`,
and license CC-BY-NC-4.0. It is an encoder dependency, not per-scene payload. A commercial pipeline
requires separately licensed supplied weights.

The checkpoint was reconstructed through exact HTTP byte ranges after an older Xet transfer
produced a corrupt sparse file. The first GiB was byte-equal across independent HTTP and hf-xet
1.6 reconstructions; 472 remaining ranges required HTTP 206, exact content range/length, exact repo
commit and linked ETag, then SHA-256 before write and after re-read. The complete file independently
matches the expected hash and exposes 1,797 safetensors keys.

A disposable probe used only the already-consumed CORE-018 packet decodes for cameras C0000,
C0010, C0020 and C0030. It opened no fresh scene/reporting image and performed no optimization.
The required mixed precision is bfloat16 aggregator plus float32 camera/depth heads under bfloat16
CUDA autocast; whole-model bfloat16 fails at the pinned head boundary. On the RTX 3050, one-view
and four-view 392-max-side inference took 0.388 s and 0.731 s, with four-view peak allocated/reserved
memory 3.535/4.649 GB. All predictions were finite.

After one four-camera Sim(3), known-ray scaled depth achieved 85.64% projective validity,
3.26%/25.74% median/p90 relative-depth error, and 0.0429/0.2477 median/p90 best-view RGB L1. Frozen
limits were respectively 25%, 12%/35%, and 0.15/0.30, so every controlling gate passed. Predicted
camera diagnostics also passed: leave-one-out center error 6.83% of camera diameter, median
orientation error 13.0 degrees, and median focal-relative error 14.0%. Raw untrimmed cross-cloud
distance failed, which is consistent with non-overlapping visible surfaces and makes naive cloud
nearest-neighbor fusion specifically inappropriate.

The probe artifacts remain disposable under
`/tmp/core019_vggt_packet_probe_20260807_v1/`; they justify implementation only.

## Selected mechanism and falsifier

Overlapping calibration-only four-view groups yield scaled depth maps. Each group receives one
Sim(3); each camera fuses every group estimate by robust confidence-weighted median/MAD. Candidate
points lie on known rays. Cross-view depth classifies support, compatible occlusion, and free-space
contradiction. Structural packet proposals preserve detail; a bounded low-density cover preserves
flat surfaces. Compatible local points may contract to continuous centroids, and dynamic
feature/normal/color-aware WSE removes redundant overlap to an exact budget. Surfel covariance is
tangent-aligned and uncertainty-bounded.

The selected method is killed if step-zero geometry still contains a volume, trail, duplicate
shell, floating sheet, grid imprint, boundary hole, or deleted thin structure; if its step-zero
reporting gain is below 2 dB; if the fixed-topology advantage disappears by step 500; if it is not
terminally Pareto-safe; or if its complete scene payload is not smaller than the original files.
No threshold rescue is permitted on the frozen development scene.

## Evidence boundary

`karate/frame_00005` is an exposed development scene, not held-out confirmation. Reporting cameras
`C0024`, `C0010`, `C1004`, and `C0022` were selected from calibration geometry before the frame was
opened and are excluded from every construction operation. A distinct prospective review is still
required for claim-bearing execution. Until then, implementation and any run are default-off
diagnostic evidence only.
