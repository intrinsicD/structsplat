# FIT-027: Cache target-side SSIM statistics across fit iterations

**Status: open, measured opportunity.** Prototyped and timed 2026-07-25; not implemented, because
it needs a per-fit state API and correct invalidation wherever the target changes.

## Context

PORT-002's 2026-07-25 profile showed the renderer stops being the dominant fit-step cost once the
tiled path lands. At 512² the non-renderer remainder is `~6.1--6.3 ms` and is flat in `N`, overlap,
anisotropy, and renderer arm. Direct measurement attributes it almost entirely to SSIM:

- 512²: SSIM fwd+bwd `6.531 ms`, L1 fwd+bwd `0.206 ms`, Adam `~0.19 ms` (`N`-independent);
- after the separable-window change landed in PORT-002, SSIM fwd+bwd is `~4.35 ms` at 512² and
  still the single largest term in the tiled step (`6.706 ms` total).

`_ssim_builtin_bchw` computes five blurred quantities: `blur(p)`, `blur(t)`, `blur(p*p)`,
`blur(t*t)`, `blur(p*t)`. Within a single fit the target `t` is constant, so `blur(t)` and
`blur(t*t)` — and the derived `mu_t`, `mu_t2`, `sig_t` — are recomputed every iteration to produce
the identical result, and they require no gradient.

## Goal

Reuse the target-side SSIM statistics across iterations of one fit, without changing the loss
value, its gradient, or the stateless `metrics.ssim()` contract used for reporting.

## Measured prototype

Separable blur plus cached `mu_t` / `mu_t2` / `sig_t`, against the pre-change dense builtin:

| resolution | builtin fwd+bwd | separable + cached | speedup |
|---|---|---|---|
| 256² | `1.641 ms` | `0.884 ms` | `1.86x` |
| 512² | `6.212 ms` | `3.595 ms` | `1.73x` |

Agreement with the dense builtin: SSIM value `6.5e-9` absolute at 512²; gradient `2.274e-11`
absolute, `2.0e-6` relative. The separable half of that (`1.42x`) already shipped with PORT-002;
this task is the caching half, worth the remaining `~1.2x` on SSIM.

## Approach

1. Introduce per-fit cached SSIM state (an object or an explicitly-keyed cache) holding the
   target-derived tensors, created once per fit and owned by the fitter rather than by
   `metrics.ssim()`.
2. Keep `metrics.ssim()` stateless and unchanged for reporting and for the ablation, so logged
   metric values stay comparable to prior runs.
3. Invalidate correctly wherever the target changes within a run — the pyramid path, the
   FIT-016 low-pass loss-target curriculum, and any masked or cropped target. Prefer explicit
   ownership over identity-keyed caching so a mutated-in-place target cannot silently serve stale
   statistics.
4. Confirm no interaction with `ms_ssim`, which rebuilds `t` per scale via `avg_pool2d` and would
   need one cache entry per scale if it participates at all.

## Acceptance criteria

- [ ] Loss value and gradient match the stateless builtin within the tolerances measured above on
      fixed fixtures.
- [ ] A target change mid-fit (pyramid level advance, curriculum step, mask change) provably
      invalidates the cache; a regression test drives at least one such transition and fails if
      stale statistics are served.
- [ ] `metrics.ssim()` and `metrics.ms_ssim()` remain stateless; reported metric values are
      unchanged.
- [ ] Benchmark records fit-step time before/after on the PORT-002 grid, with CVs inside the same
      `5%` limit that profile uses.
- [ ] Memory overhead is bounded and logged (three target-sized tensors per active fit).

## Interfaces touched

`src/structsplat/metrics.py`, `src/structsplat/fit.py`, `tests/test_metrics.py`,
`benchmarks/tiled_render_profile.py`.

## Depends on

PORT-002 (separable window, and the profile that exposed the bottleneck), FIT-003. Interacts with
FIT-016 and the pyramid path for invalidation.

## Notes

- **2026-07-25 decision: do not adopt the third-party `fused_ssim` dependency.** It is not on PyPI
  and not installed here, so adopting it would mean fetching and compiling third-party CUDA from
  GitHub. `docs/research/2026-07-21-cuda-thrust-port-feasibility.md` records an earlier FIT-003
  measurement of `-22.6% / -28.0%` s/iter for that backend, and this task's dependency-free route
  measures `1.73x` on the SSIM term for comparable benefit. The existing `--ssim-backend
  builtin|fused|auto` selector and `metrics._fused_ssim_fn` are **kept as-is**: they already fall
  back cleanly when the module is absent (`fused` falls back, `auto` resolves to `builtin`), so
  the option stays open for anyone whose environment does provide it. No code is removed and no
  default changes; this decision is about what StructSplat depends on, not about what it permits.
  This task is therefore the supported path to the SSIM bottleneck, and it remains the right work
  even if `fused_ssim` were later adopted, since CPU and non-CUDA runs use the builtin regardless.
- Do not fold this into PORT-002. That task's profile is a frozen, now-passed gate; adding a loss
  change to it would mix an authorized renderer result with an unauthorized loss result.
