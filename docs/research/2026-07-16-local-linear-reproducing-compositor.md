# BENCH-013 local-linear reproducing compositor

## Decision

**KILL the no-extra-state local-linear/moving-least-squares compositor.** The completed v3
Stage-0 assay shows that enforcing first moments independently at every pixel creates signed-weight
extrapolation and ringing under StructSplat's compact clipped support. The failure is not a rank,
theorem, or implementation-parity failure.

Canonical artifact:
`results/bench013_local_linear_stage0_full_v3_2026-07-16`.

## Integrity

- Task: `b40b9075262c8c2dc07212490a1178b0168b6da9e433cb4ca23a10957bb1d0ad`
- Binding: `2ff71a04472d061532f2b18dbf8619bf92cbc523aff9b50c96a3a6240be74adf`
- Analysis: `68db5bc6686f2b3dda430abaa0df6ffdd95698551a99cd0ae5a305cead258791`
- Completion: `cf2adf32b848e1a661c864ae6e6dab0e2a363416a26411c6374cafd9031d4e4e`
- Replay: `709d9b55f6535bbb55d61e926706b501dfe7faafd35b320d14778b38dee74ce8`
- Artifact manifest:
  `1ef0823777ff70723603ca40d4d9ba1dabdfb266deee145e19f4ba7ba14ffc1f`
- Source archive:
  `38569701e8a1eca60be1e47cad9a90e54e4e756765f5843953d18b5f7f1f21f3`

The replay completed all 108 forward cells and 432 permutation rows. It found 82 forward-cell
failures, 49 failures among 63 effective-weight fields, and 27 permutation failures. The registered
gradient cell was correctly recorded as not reached because its base forward gate failed.

## Interpretation

Local-linear regression and moving least squares can reproduce affine fields, but the reproduction
constraint replaces a positive partition with signed effective weights. Under irregular compact
support, boundary and sparse-neighborhood leverage becomes the dominant problem. The result closes
this exact no-ghost, per-pixel solve, ridge/support/threshold formulation without retuning.

The useful residual insight is narrower: affine reproduction itself remains valuable, but it must
be delivered without solving a signed local correction at every pixel. That observation motivated
BENCH-014's explicit global carrier.

## Claim boundary

This is synthetic analytic conditioning/ringing evidence. It does not show that every MLS,
reproducing-kernel, maximum-entropy, or learned coordinate method fails, and it is not a
natural-image or production-performance result.
