# BENCH-016 v4: saved-site parity invalidity

## Outcome

BENCH-016 protocol v4 is **invalid / no decision**. It halted at the first native execution,
`0057`, requested `0.5 bpp`, repeat `0`, before any successful scientific row was appended. The
frozen maximum per-channel train-versus-cold difference bound (`<= 1` uint8 code) failed. No SAD
pass/fail gate was evaluated, and this event is not evidence for or against reuse of SAD.

The canonical failure record is
`results/bench016_native_sad_frontier_v4_2026-07-16/invalidity.json` (SHA-256
`a3b9c696ed0fc0d574f4e177353ffec6eb5ba2600aafca640425a33a35d2f447`). The v4 binding is
`473f458ec5d3729829dd34b371ab7e6b82f5d87f2e0d57e140afb1b69c65ce71` and remains preserved with
its executed-source archive.

## Observed failure

The 510 x 768 target produced 1,530 active exported sites. Central measurements were:

| reconstruction | PSNR | role |
|---|---:|---|
| in-training export | 35.059850 dB | transient candidate-state diagnostic |
| official fresh saved-site replay | 35.054282 dB | recipient replay under the pinned decoder |

The target-relative PSNR change was `-0.005568 dB`, while the two reconstructions had
`61.208822 dB` difference-image PSNR. Nevertheless, their sparse ownership differences included a
maximum channel change of `55`; 1,785 of 391,680 pixels (`0.455729%`) had at least one channel
whose absolute change exceeded one code. Relaxing the bound to 55, 64, or any other observed-data
threshold is prohibited.

## Cause

The saved site floats are written with nine significant digits, sufficient for float32 round-trip.
The mismatch instead comes from decoder state that is not serialized:

- native training retains a mutable, history-dependent K=8 candidate map over the original
  allocated/indexed site state;
- the pinned configuration refreshes this map by one candidate pass every eight updates;
- after optimization, active sites are filtered and compacted into TXT, but the candidate map is
  not exported;
- official render-only mode reconstructs a new map from the compact sites using one JFA round and
  sixteen candidate-refinement passes.

Consequently the training PNG and official saved-site replay are two renderings of the same
exported field under different approximate candidate maps. Exact per-pixel equality between them
is not an upstream representation invariant. Five additional default fresh replays of the failed
TXT were byte-identical to the canonical cold PNG. Separate, previously exposed COCO calibration
at both protocol site densities also produced exact repeated hashes: 10/10 at 66 sites had pixel
SHA-256 `456f014fdb5bb6d8eb3ac2e40c1aaded2644f431f64cd24095ed81b102fa2ba7`, and 20/20
at 265 sites had `907b1b1e526a851de6e1af688e68705662a99a5d7bc2fbc4cdfebf007f61724a`.
A deliberately pathological 128,000-site stress case produced ten different hashes, so
determinism is not generalized beyond the screened density regime and remains fail-closed.

## Approved repair boundary

A new protocol may repair validity without fitting a tolerance:

1. Restart every one of the 48 native and 96 StructSplat executions; import no v4 result.
2. Keep all images (including exposed `0057`), pixels, rates, methods, schedules, repetitions,
   metrics, aggregation, gates, and branch actions unchanged.
3. Use a fresh official saved-TXT render as the sole native final-quality reconstruction. Persist
   the in-training PNG and its metrics as diagnostics only.
4. Persist the output of every required fresh render-only timing invocation and require each
   decoded RGB8 pixel hash to equal the canonical saved-site replay exactly. Any mismatch is an
   integrity failure with no decision.
5. Retain the upstream convergence calculation but label it internal-training-state
   normalized-iteration PSNR AUC; it is not cold-decode AUC.
6. Call TXT plus the pinned official renderer/configuration recipient-replayable, not deployable,
   self-contained, or compressed. SAD compression remains untested.

This repair was chosen from source semantics and a threshold-free decoder invariant, not from the
observed `55`-code failure. Because one development image was exposed before v5, any repaired run
remains transparent development evidence rather than pristine preregistration or held-out
confirmation.
