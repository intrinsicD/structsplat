# Fast, convergent, compressive image → 2D Gaussian field: pipeline design

Status: design proposal and ordered evidence program. It composes mechanisms the repository has
already measured and names the gates that must pass before any piece becomes a default. It does
not claim novelty or state of the art; every quantitative statement below cites an existing
artifact, and every proposed change carries a falsification criterion. Authored 2026-08-07 from
the evidence survey in this document; the ideation discipline is
`.claude/skills/structsplat-research-ideation`.

## 1. Objective

Transform one image into a 2D Gaussian field optimizing four axes at once:

```text
min  T_encode(P)  and  B_complete(P)
s.t. D(P) at matched budgets is not worse than the maintained recipe,
     cold decode + first render stay within current cost,
     every number is reproducible from logged config + seed.
```

Following the DOCS-007 review (`ara/evidence/2d-gaussian-image-fields-sota-review-2026-08-04.md`
§3.3, §11 — producer-authored, in-review), the design treats **four clocks** (offline training,
encode/conversion, cold decode/first render, warm render/query) and **complete-package bytes**
(headers, ranges, tables, alpha, padding, any side state) as the only admissible accounting.
Iterations are not a portable convergence metric; compression ratio is reported against
(1) the exact supplied source file, (2) a canonical lossless PNG, (3) raw RGB, separately.

## 2. Where the four axes stand today (baseline, with artifacts)

| Axis | Current measured state | Artifact |
|---|---|---|
| Encode speed | 270.3 s for the canonical masked 1200-px run, 11k capacity, RTX 4090; **19,156 attempted / 1,800 accepted optimizer steps (9.4%)** | `runs/janelle_C0001_current_pipeline_20260727/metrics.json` |
| Quality | 26.82 dB / 0.9889 MS-SSIM / 0.0477 LPIPS terminal on that run; init state 20.92 dB → LS color solve 21.40 dB | same |
| Convergence | headline targets 28/30/32 dB never reached on the canonical run; the only decision-grade multi-image result is ABL-006's init confirmation (`quadtree_wse` +0.0930 dB, CI [+0.0168, +0.1700] at 5k) | `ara/evidence/abl006-complete-2026-07-07/` |
| Rate | actual complete-stream **67.5 bits/Gaussian** at 16/8/8/8 and **54.5** at 12/6/6/8 (≈1.39–1.43 / 1.11–1.15 bpp at N=8192, 768×512); **−1.0 dB vs JPEG-444 at 0.5 bpp, −4.3 dB vs AVIF-444**; PNG lossless mean 11.43 bpp on the same set | `ara/evidence/comp008-sgi-entropy-oracle-2026-07-16/cells.jsonl`, `ara/evidence/bench007-stage1-killing-pilot-2026-07-14/{selected,conventional}.csv` |

Structural facts that drive the design:

- **~91% of attempted optimization is discarded** by the transactional Pareto gate; 75% of
  rejected blocks die on the absolute interior-hole veto alone; `safe_polish` attempted 3,276
  steps and committed 0 (O87/O88, `tasks/BENCH-017-full-frame-arm-screen.md`, ADR-0026).
  No experiment has attacked this yet (FIT-028, FIT-029, BENCH-018 all open).
- The tiled renderer's step is **0.63×** the exact-CUDA step at the representative cell, with the
  GPU-built tile index at 1.4% of step cost (PORT-002 preregistered profile pass,
  `results/port002_tiled_render_profile_rtx3050_head_pass/`, local-only); the warp-reduced tiled
  backward is 5.96 → 2.11 ms (PORT-003). Neither has passed the fair end-to-end *fit* benchmark,
  which is what a default flip requires (the 2026-07-05 fair run had `cuda_tiled` 1.69× slower).
- Zlib already captures most easily reachable entropy: a real factorized arithmetic coder buys
  ~4–5%, spatial context ~1.3% more against a 17% oracle, and the strongest coder failed its
  1.25× decode-cost gate by 5–24× (COMP-008/009, C47/C48). The **rotation stream (11.8% of the
  container) is the only stream zlib cannot touch** (ratio 0.93, sometimes expanding).
- Post-fit QAT recovers **+0.64 to +1.50 dB at low bit tuples** and is slightly harmful at
  16/8/8/8 (COMP-004, C10).
- The masked fine-detail-pursuit tail hits its −25%/−20% high-pass/Laplacian targets with a
  **median 384–768 added rows**, where the error-only tail needs ~6,144 and misses in 44/51
  cells (FIT-040/041, C59/C60, O92).
- BENCH-007 refuted the tensor-metric/WSE *rate-distortion* claim (C28). Structure-aware
  placement remains the supported *convergence/quality* initializer (ABL-006), and exact
  complete-stream bytes are supported for *control allocation* (+0.213 dB mean, C37) — but no
  structural-prior compression claim may be revived without a materially different question.

## 3. Design overview: two lanes, one shared substrate

The evidence supports splitting the product surface rather than tuning one schedule for
everything. Both lanes share the NumPy analysis stack, the torch/CUDA renderer, and the codec.

```text
image (H,W,3)
  │  shared analysis (NumPy): structure tensor → density → quadtree-WSE placement
  │  + fixed-geometry LS color solve
  ├── DIRECT LANE (unmasked / throughput): short staged direct fit, wave growth priced by
  │   marginal distortion per complete byte, terminal high-pass pursuit, QAT, encode
  └── GUARANTEE LANE (masked / production): current safe transactional schedule with the
      discard-rate fixes (hole budget, block sizing, polish disposition)
```

The dispatch decision — whether unmasked `convert` keeps the safe schedule or takes the direct
lane — is exactly the question BENCH-017 has already preregistered. This design does not
pre-empt it; it makes the direct lane the arm BENCH-017 screens.

## 4. The direct lane, stage by stage

**Stage 0 — byte-derived capacity.** `FitConfig.target_file_bytes` →
`pool.derive_capacity()` (exact raw SSPL1 layout) picks N. Rate control is a first-class input,
not a post-hoc quantization choice. Supported by C37's finding that exact bytes improve
precision-mix/control allocation.

**Stage 1 — placement.** `quadtree_wse` with anisotropic WSE and `wse_progressive_order=True`.
This is the one multi-image, multi-seed, CI-backed init winner (ABL-006/ADR-0013); progressive
order costs 14.2% of selection time and wins 32/32 prefix-geometry pairs (C25), keeping a
LOD-prefix option open for the codec later.

**Stage 2 — closed-form color.** Fixed-geometry least-squares color solve before any Adam step
(+0.48 dB instantly on the canonical run; the cheapest quality per second in the repo).

**Stage 3 — short direct fit.** L2 pixel loss (recipe), per-phase LR ladder, exact `cuda`
renderer today. Candidate knobs pending ABL-005 completion: `opacity=constant` (+0.91 dB at 4
pairs and shrinking) and `lr_schedule=cosine` (+0.58 dB at 4 pairs) — *screening signals only*,
promoted only if the 336-cell fair-regime matrix confirms them. Renderer flips to
`cuda_tiled` + warp-reduced backward only after the PORT-002-authorized fair end-to-end fit
benchmark passes; fused render+loss and CUDA-graph capture are the follow-on PORT-002 items.

**Stage 4 — wave growth, priced acceptance.** Grow toward capacity in bounded waves proposed by
the residual structure tensor (existing densification machinery), accepting on a **cheap scalar
gate during growth** (foreground MSE + a *budgeted* hole term) and reserving the full Pareto
metric vector for phase boundaries and the terminal state. Rationale: the all-or-nothing
absolute hole veto is the measured throughput killer (75% of rejections), and ADR-0026 already
ships the `hole_regression_budget` mechanism default-off. This is FIT-028's screen plus
BENCH-018's `block_steps` choice, not a new mechanism.

**Stage 5 — terminal detail.** Masked-lane-proven fine-detail pursuit adapted to the direct
lane: deep high-pass/NMS births in small waves with a joint frozen-base color solve
(FIT-039/040 mechanism). FIT-042 owns independent confirmation before any default. The
error-only tail (FIT-031) remains the boundary/global repair tool, not the detail tool (C60).

**Stage 6 — quantization-aware finish.** Freeze ranges, run short STE QAT at the *target* bit
tuple. Evidence says ship low tuples with QAT (12/6/6/8: +0.66 dB; 10/5/5/5: +1.50 dB) and skip
QAT at 16/8/8/8.

**Stage 7 — encode.** SSPL1 as-is (Morton delta means/colors, byte-planar means, circular
rotation, per-stream zlib). Codec work worth doing, in expected-value order:

1. Finish COMP-003 rung 4 honestly: true per-attribute sorted planes through
   `benchmarks/rate_distortion.py` (the byte-square PNG smoke was negative; the rung is not
   done).
2. Rung 5 LSQ learnable quantization ranges (unstarted; cheap; composes with QAT).
3. The rotation stream: the one pool zlib cannot compress. A candidate is predicting θ from
   already-decoded means via the *decoder-side* structure of the field and coding the residual —
   with expectations set by SSP2E's measured 1.3% context ceiling and C46's refuted log-SPD
   recoding. Killing test: if rotation-stream bytes do not drop ≥15% at equal decode cost on the
   COMP-008 8-image set, abandon.
4. Any arithmetic/range coder enters only under the existing ≤1.25× decode-cost gate that
   SSP2L/SSP2E already failed. Decode cost, not rate, is the binding constraint.

The Field V2 additive-compositor chain (BENCH-019 → CORE-013 → BENCH-020 → COMP-013 →
BENCH-025 → COMP-014) remains the long-horizon route for compositor semantics and a structured
codec; per H16 it advances through its own killing gates and must not block this lane.

## 5. The guarantee lane (masked production)

Keep the transactional safe schedule — its Pareto gate is the artifact-safety product feature —
but fix its measured waste:

- **FIT-028**: measure the interior-coverage budget; replace the absolute veto with the shipped
  ADR-0026 budget if the screen supports it.
- **FIT-029**: `safe_polish` committed zero steps at 5.8 s + 468 attempted steps on the
  canonical run; decide its disposition with the existing task's gate.
- **BENCH-018**: choose `block_steps` (the inherited 250 was never measured), after FIT-028 and
  FIT-027's cheaper SSIM (both move the optimum).
- **PORT-006** owns end-to-end acceleration once the semantics questions settle.

## 6. Ordered evidence program (what to run, in order, with kill criteria)

| # | Experiment | Owning task | Gate / kill criterion (frozen in the task where it exists) |
|---|---|---|---|
| 1 | Full-frame screen: safe schedule vs plain fit at matched budgets | BENCH-017 (todo, preregistered) | full-frame safe schedule must win paired mean PSNR, 95% CI excluding zero, Kodak-24+COCO4, ≥3 seeds, rejected trials priced as spent work; a loss dispatches unmasked convert to the direct lane |
| 2 | Hole-budget + polish disposition + block sizing | FIT-028, FIT-029, BENCH-018 | equal-quality (Pareto vector at terminal) at materially lower attempted-step count; no retuning on the Janelle image alone |
| 3 | ABL-005 completion (cosine LR, constant opacity, charbonnier, variance density, moment split) | ABL-005 (partial, 24/336 cells) | promotion only on the full fair-regime matrix; deltas are already visibly shrinking with pairs |
| 4 | Fair end-to-end fit benchmark, tiled + warp-reduced backward | PORT-002/003 (authorized by the profile pass) | default flip only on end-to-end fit-time win at parity quality; deterministic-accumulation caveat stays recorded |
| 5 | Independent fine-detail pursuit confirmation | FIT-042 | existing task gate; no default before it |
| 6 | Codec rungs 4–5 + QAT-at-low-tuple default; complete-stream RD with JPEG/AVIF/PNG context | COMP-003, COMP-001 | complete-stream bytes only (K05); JPEG-444 parity at 0.5 bpp on the frozen 8-image set is the falsifiable *target*, not an assumption; AVIF parity is not evidenced as reachable |
| 7 | Rotation-stream candidate | new task if pursued | ≥15% rotation-stream byte reduction at ≤1.25× decode cost on COMP-008's set, else abandon |

Steps 1–3 need no new mechanism code. Steps 1, 2, 4 attack encode speed; 3, 5 attack
convergence/quality; 6, 7 attack rate. Every step reuses a frozen protocol that already exists
or inherits BENCH-007's rate accounting rules (`8·bytes/(W·H)`, cold decode, no analytical
substitution).

## 7. Candidate targets (falsifiable, not promises)

- **Encode:** ≤90 s for the canonical 1200-px masked conversion at parity quality (from 270 s),
  from discard-rate work alone; ≤60 s if the tiled fit benchmark passes.
- **Convergence:** report "steps and seconds to 95% of terminal quality" per run (the Image-GS
  anchor); target ≤1/3 of current wall-clock to 95% on the direct lane.
- **Rate:** ≤7.0 complete-stream bytes/Gaussian at ≤0.1 dB QAT-recovered loss (from 8.04
  median); ~8–10× vs canonical lossless PNG at development-set quality.
- **Decode:** no regression vs current zlib decode.

## 8. Explicitly excluded (refuted or gate-failed; do not re-propose)

Feature scale caps as default (C02, INIT-008) · event color solve (C50) · specialized detail
tail at equal count (C52) · low-pass loss curriculum (C21) · signed matched-residual
densification (C24/FIT-017) · SAD α=0.7 responsibility split (C29) · opacity-gauge quotient
allocation (C32/ADR-0014) · log-SPD covariance recoding (C46) · fixed SSP2E v1 context coder
(C48) · +1 standard Gaussian per 16 B marginal edit (C36/ADR-0016) · `cuda_tiled` default
without the fair fit gate (C53) · any revival of the BENCH-007 tensor-WSE rate claim, including
consuming the DIV2K validation split (C28) · flanking as default (ADR-0013) · retained-ancestor
and artifact-first quadtree reconciliation as formulated (HIER-006/007/008/009 negatives).

## 9. Claim boundary

Nothing here authorizes a README or task-status quantitative claim. The honest current
statement remains: complete-stream Gaussian fields in this repository store an image in roughly
1/8 of its lossless-PNG bytes at development-set quality while trailing JPEG-444 by ~1 dB and
AVIF-444 by ~4–6 dB at matched bytes on the frozen development images (C26/C28 context rows).
The design's purpose is to move the four axes with gated, reproducible steps — and to record
the negatives at the BENCH-007 standard when they come.
