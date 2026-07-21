# Feasibility study: porting StructSplat to pure CUDA or Thrust

**Date:** 2026-07-21
**Status:** analysis/recommendation only — no decision taken, no ADR, no task status change.
**Question:** would porting StructSplat to "pure CUDA or Thrust" be feasible and beneficial,
given a top-tier production-performance goal?
**Scope caveat:** every number below is from committed evidence measured on one RTX 3050. No new
measurements were run for this study (no CUDA device in the authoring environment), and per the
PORT-004 language none of the cited microprofile numbers support a cross-GPU claim.

## Verdict summary

| Option | Feasible? | Beneficial? | Verdict |
|---|---|---|---|
| A. Torch-free "pure CUDA" trainer (encode side) | Yes, at multi-month cost | No, at current margins | **Do not build now** |
| B. Deeper CUDA inside the existing torch extension (PORT-002/003 lineage) | Yes | Yes — this is where the measured headroom is | **Do this; it is already the planned path** |
| C. Wholesale "port to Thrust" | Not meaningfully | — | **Category error; use CUB/Thrust only for the sort/scan stages of B** |
| D. Torch-free forward-only decode renderer (production decode, PORT-001 RHI) | Yes, easily | Yes — this is the actual production deliverable | **Do this when decode matters; "pure CUDA" is right here** |
| E. Batch/stream parallelism across images for encode throughput | Yes, trivially | Yes, near-linear | **Prefer before any rewrite** |

"Port this to pure CUDA" is therefore two different questions with opposite answers. For the
*training/encode* pipeline the answer is: the port already happened where it pays (ADR-0011's exact
CUDA renderer), and the remaining wins are kernel- and indexing-level work inside the existing
PyTorch extension, not the removal of PyTorch. For the *decode/viewer* path the answer is: yes, a
torch-free CUDA/Vulkan forward renderer is exactly the plan of record (PORT-001's open RHI
milestone) and is unusually easy here because the normalized compositor is order-independent.

## 1. What already exists (baseline for "port this")

- **Exact CUDA renderer** (`renderer=cuda` / `cuda_additive`, ADR-0011): owned PyTorch extension
  under `src/structsplat/cuda/` implementing the reference clipped-support normalized/additive
  equations with hand-written forward and backward kernels. Parity-tested against the reference,
  which stays the correctness oracle (ADR-0001).
- **Tiled variant** (`cuda_tiled`): tile-to-Gaussian index built in Python/torch each call
  (`cuda_render.py:_build_tile_index`), then one CUDA block per image tile. Measured **1.69× slower
  end-to-end than exact CUDA** on the fair difficult-four protocol (48/48 cells, +17.63 s mean per
  fit; `ara/evidence/fair-density-control-cuda-tiled-difficult4-2026-07-05/run.md`), despite lower
  theoretical work.
- **Block-reduction backward** (`cuda_block_reduce`, PORT-004): −57.5% exact-backward and −25.3%
  representative-step device time at the frozen N=2048 cell, but it failed the frozen all-grid
  direction rule (all four N=512 backward ratios ≥ 0.99, three > 1.0) and one 5% CV stability
  guard. Kept benchmark-only per the frozen gate.
- **Fused SSIM** (FIT-003): opt-in `fused_ssim` backend, −22.6% / −28.0% s/iter vs the pre-FIT-003
  loop at budgets 512/20000 on CUDA (`ara/evidence/fit003-speed-2026-07-03/summary.md`). Exact CUDA
  at budget 20000 was already ~12× faster per iteration than the CPU reference on a 96 px image.
- **Init-time math**: NumPy by invariant, already optimized asymptotically (INIT-006: quadtree
  0.20 s at n=20k, vectorized conflict graph, chunked GEMM spacing). Only the greedy WSE heap
  removal remains serial Python — deliberately, because exact-N elimination order is the
  reproducibility contract (INIT-009's terminal-set-preserving progressive order is definitionally
  sequential).
- **Packaging**: the extension is JIT-compiled at first use via `torch.utils.cpp_extension.load`
  (needs nvcc at runtime; some environments need the documented `LD_PRELOAD` libstdc++ workaround,
  ADR-0011 consequences).

## 2. Where the device time actually goes (measured)

PORT-004's representative cell (RTX 3050, 256×256, N=2048, actual overlap ≈ 16.9, exact untiled
normalized renderer, opacity+support-fade, 0.7·L1+0.3·SSIM;
`ara/evidence/port004-exact-backward-block-reduction-2026-07-16/primary/summary.md`):

| Component | ms | share of 3.425 ms step |
|---|---:|---:|
| Renderer forward | 0.506 | 14.8% |
| Renderer exact backward | 1.178 | 34.4% |
| Loss backward beyond renderer (SSIM/L1 chain) | ≈ 0.867 | 25.3% |
| Adam step | 0.435 | 12.7% |
| Remainder (loss forward, targets, bookkeeping) | ≈ 0.439 | 12.8% |

Two consequences:

1. **Amdahl bounds a renderer-only rewrite at ~2×.** Renderer forward+backward is ≈ 49% of the
   device step at this cell. Even an infinitely fast rasterizer leaves the SSIM/L1 chain, Adam, and
   bookkeeping — all already dense fused GPU work in torch — as > half the step.
2. **The single biggest measured win so far came from a reduction schedule, not from removing
   torch.** `cuda_block_reduce` cut the full step by 25% by changing ~40 lines of kernel-side
   accumulation. The equivalent claim for "remove PyTorch" has no measurement behind it anywhere in
   the repo.

At small workloads (128², N=512) the step floors at ≈ 2.03 ms with forward at 0.39 ms —
launch/fixed-overhead dominated, which is why block-reduce showed nothing there. That regime wants
fewer launches (CUDA Graphs, in-extension indexing), not different ownership of the math.

Wall-clock scale: the fair difficult-four protocol (max-side 768, 1500 iters, budgets
{2000,5000,10000}) implies ≈ 25.5 s mean per exact-CUDA fit (derived: tiled = exact + 17.63 s at
1.69×). Init is seconds-scale and once per image; the fit loop is the encode cost.

## 3. Option A — torch-free "pure CUDA" trainer: feasible, not beneficial now

**What it would require.** The extension today consumes `(means, conics, colors, radii, opacities)`
and returns gradients to those tensors; autograd carries them back through the RS→conic chain
(`gaussians.py:conics` — `exp`/`cos`/`sin` algebra), the SSIM/L1 losses, and Adam. A torch-free
trainer must reimplement, with hand-derived backward where needed:

- the RS→conic→radii parameter chain and its backward;
- L1 + windowed SSIM forward/backward (currently cuDNN-backed convs or `fused_ssim`);
- Adam with per-group LRs and schedules;
- the entire live research surface of `fit.py` (~3,000 lines): densification/relocation events,
  moment-preserving splits, color solves (a CG solver), checkpoint selection, adaptive counts,
  pruning, mask containment — the axes the stage-search sweeps;
- reproducibility plumbing (seeded config → identical trajectories) duplicated against the
  reference oracle.

**Why the cost is disproportionate:**

- The measured torch-side share of a step is either already near-optimal fused GPU work (Adam,
  SSIM) or is precisely what PORT-002's *fused loss* item would fold into the extension anyway
  without giving up autograd elsewhere.
- `fit.py` churns weekly (FIT-013..020 all landed within one month). Freezing it in C++/CUDA makes
  every future ablation axis a dual-implementation task and detaches the production trainer from
  the correctness oracle that ADR-0001 makes load-bearing.
- There is currently **no frozen production method to port**: BENCH-007 rejected the structural
  compression claim at its gate, and the priority queue explicitly ranks PORT-002/003 as "if tiled
  CUDA remains strategically important after quality work". A multi-month rewrite of a moving
  research substrate is the wrong order.

**When to revisit:** a specific fitted configuration is frozen for a product (fixed renderer, loss,
schedule, refine axes), encode latency has an on-device SLO that batching (Option E) cannot meet,
and the paired-benchmark protocol can price the rewrite against Option B's endpoint.

## 4. Option B — deeper CUDA inside the existing extension: the real port target

This is PORT-002 + PORT-003, refined by what the evidence already showed. Ordered by expected
leverage:

1. **In-extension GPU binning (CUB radix sort + device scan).** `_build_tile_index` currently
   issues ~10 torch ops per iteration (`repeat_interleave`×6, `argsort`, `bincount`, `cumsum`, …)
   over int64 intermediates. Replace with packed 32-bit `(tile_id, gid)` keys and
   `cub::DeviceRadixSort::SortPairs` + `cub::DeviceScan` on the current torch stream, with buffers
   from the torch caching allocator. Note `torch.argsort` already dispatches to CUB radix sort
   internally — the win is launch count, dispatch overhead, int64→int32 bandwidth, and dropped
   intermediates, not asymptotics. This is also the natural CUDA-graph enabler (stable shapes,
   capacity-capped buffers).
2. **Shared-memory collective staging in the tiled kernels.** `tiled_forward_kernel` re-reads each
   Gaussian's parameters (8–9 floats plus two int64 radii) from global memory per pixel per
   Gaussian and recomputes `support_bounds` per thread. The standard splatting pattern (stage Gaussians into shared memory
   in warp-cooperative chunks, then iterate the staged batch) plus 32-bit gids is the difference
   between the current 1.69×-slower tiled path and one that wins at high N. Crucially, the
   normalized compositor is **order-independent** (no depth sort, unlike 3DGS pipelines) — tile
   lists need grouping, never sorting by depth, so the pipeline is strictly simpler than the
   3DGS-style renderers it borrows the pattern from.
3. **Tighter ellipse–tile intersection** (PORT-002 approach item 1). The AABB bound is worst-case
   for exactly the elongated edge Gaussians this method produces; the 2026-07-05 tiled evidence
   named it a prerequisite.
4. **Backward reductions without global per-pair atomics** (PORT-003). The tiled backward still
   does ~9 global atomicAdds per pixel–Gaussian pair. Per-tile partials + a second-pass reduce, or
   a shape-dispatched variant that selects block-reduce only in the regime where it measured wins
   (N≳2k, high overlap), belongs in a **new task with its own preregistered gate** — PORT-004's
   frozen cell/thresholds must not be retuned, and its all-grid failure is precisely the argument
   for dispatch-by-shape rather than a single kernel.
5. **Fused render+loss partials and CUDA graph capture** (PORT-002 items 3–4). Loss forward +
   backward beyond the renderer is ≈ 38% of the representative step; the small-N floor is
   launch-bound. Graph capture needs the restructuring-event fallback PORT-002 already specifies
   (densify/relocate/prune change N).

All of this keeps autograd, the oracle relationship, every searchable fitter axis, and the
BENCH-002 paired-measurement protocol intact. No number above authorizes a speedup claim — each
item lands behind the usual paired, seeded before/after benchmark on the fair regime.

## 5. Option C — "port to Thrust": scoped to what Thrust is

Thrust is a host-side template library of parallel algorithms (sort/scan/reduce/transform) over
CUB. It cannot express the hot code here — fused per-tile gather/accumulate rasterization and its
backward are custom kernels by nature. The parts Thrust *can* express (the binning sort and offset
scan) are exactly Option B item 1, and inside a torch extension the cleaner tool is the CUB device
API on `at::cuda::getCurrentCUDAStream()` (Thrust's default execution policy brings its own
allocation/synchronization behavior that fights the torch caching allocator unless wrapped with a
custom allocator policy). So: adopt CUB (or Thrust with a torch-allocator policy) for sort/scan
inside PORT-002 — standard practice in splatting rasterizers — and drop "Thrust" as a wholesale
port target.

## 6. Option D — production decode: where "pure CUDA" is genuinely right

PORT-001's open milestones are the production decode path: packed Gaussian buffer, tiled forward,
IntrinsicEngine RHI pass, documented >1000 FPS decode target. This side is forward-only — no
autograd, no optimizer, no torch dependency needed at all — and the order-independent normalized
sum makes it a single-pass tile gather plus one divide. A standalone CUDA (or Vulkan-compute)
library consuming the packed buffer is straightforwardly feasible, small relative to any trainer
port, and is the piece that actually delivers "top notch production performance" where production
runs: decode/display.

Two production-hygiene wins come with it:

- **Determinism:** per-pixel sequential accumulation over a fixed tile list (the tiled forward's
  existing shape) is run-to-run deterministic, unlike the untiled global-atomic forward — this
  directly serves PORT-001's open deterministic-accumulation item and the reproducibility
  invariant. The trainer can keep atomics; the decoder should not need them.
- **Packaging:** a torch-free decoder removes the runtime-nvcc/JIT/`LD_PRELOAD` fragility from the
  deployment surface entirely. Independently, the training extension should ship as a precompiled
  wheel (setuptools `CUDAExtension`) for the pinned torch/CUDA matrix once any of Option B lands.

## 7. Option E — encode throughput without touching kernels

Per-image fits are independent. For farm/production encode *throughput* (images/hour rather than
single-fit latency), multi-process or multi-stream batching scales near-linearly on committed
harness infrastructure (the sweeps already shard) and costs zero rewrite risk. Exhaust this before
pricing any trainer port.

## 8. Constraints any port must preserve (non-negotiable)

1. Normalized renderer semantics (ADR-0003), exactly as ADR-0011 preserved them; the
   reference stays the oracle (ADR-0001), with parity tests per variant.
2. Reproducibility from logged config + seed; atomic accumulation nondeterminism stays documented
   (ADR-0011), and any deterministic mode is an explicit, tested option.
3. Init-time math stays NumPy and torch-free (core invariant). A GPU WSE would change elimination
   order, hence sample sets, hence every downstream result — that is a research proposal with an
   ADR and a benchmark, not a port. Its upside is also negligible: init is a one-time seconds-scale
   cost against a ~25 s fit.
4. No performance claim without the paired, seeded, frozen-gate protocol (BENCH-002 /
   `structsplat-results-audit`); PORT-004 is the template for how to pre-register kernel gates.

## 9. Recommended order

1. **Reject** the torch-free trainer rewrite for now (Section 3). Revisit only against a frozen
   production method with an on-device encode SLO.
2. **Execute PORT-002** with the refinements in Section 4 (CUB binning, 32-bit ids, shared-memory
   staging, tighter ellipse–tile bound, fused-loss option, graph capture behind the dynamic-N
   fallback).
3. **Open a new PORT task** for shape-dispatched backward reduction (PORT-003 lineage) with a
   fresh preregistered gate covering the full N/overlap grid, explicitly not reusing PORT-004's
   frozen artifacts.
4. **Split PORT-001's decode milestone** into a standalone forward-only CUDA library (packed
   buffer + parity harness + deterministic accumulation + FPS protocol), then the engine-side
   RHI/Vulkan pass on top.
5. **Ship precompiled extension wheels** for the supported torch/CUDA matrix alongside whichever
   of the above lands first.
6. Use **batch parallelism** for encode throughput in the meantime.

## Evidence index

- `ara/evidence/port004-exact-backward-block-reduction-2026-07-16/primary/summary.md` — step
  composition table and block-reduce deltas (RTX 3050).
- `tasks/PORT-004-exact-backward-block-reduction.md` — frozen gate, all-grid failure, CV guard.
- `ara/evidence/fair-density-control-cuda-tiled-difficult4-2026-07-05/run.md` — tiled 1.69× slower,
  48/48 cells, prerequisites named.
- `ara/evidence/fit003-speed-2026-07-03/summary.md` — CUDA vs CPU s/iter, fused SSIM deltas.
- `tasks/done/INIT-006-init-performance.md` — init-time cost tables after the asymptotic fixes.
- `docs/adr/0001-pytorch-reference-first.md`, `docs/adr/0011-owned-exact-cuda-renderer.md` —
  reference-first policy and exact-CUDA ownership/consequences.
- `tasks/PORT-001-cuda-rasterizer.md`, `tasks/PORT-002-gpu-native-tile-index-fused-loss.md`,
  `tasks/PORT-003-tiled-backward-reductions.md` — open production-port milestones this study
  refines.
