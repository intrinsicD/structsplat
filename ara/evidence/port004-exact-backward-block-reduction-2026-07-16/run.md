# PORT-004 exact-backward block reduction

The primary RTX-3050 representative cell reduced exact-backward time by `57.478%` and the
device-side representative step by `25.288%`. An independent run repeated large reductions
(`51.284%` and `24.760%`) but exceeded the frozen 5% stability guard
(`5.1154%` candidate-backward CV). The frozen whole-grid direction clause also fails: four
`N=512` exact-backward ratios include regressions and one complete-step ratio is `1.0052`.

The original executable predicate omitted the whole-grid clause; the live evaluator now encodes
it and fails closed. The selector remains benchmark-only. No end-to-end, default, universal,
cross-GPU, quality, convergence, compression, or expressiveness claim is authorized.

The `primary/` and `confirmation/` directories preserve aggregate/row/sample/config summaries.
Primary audit/aggregate/rows/sample SHA-256 values are
`b09b29d43507fba552a9b32ff408c7124ad6f7ec1ef84e8a2d20f6a1703874a6`,
`6fe6d4193642f389d760bf6ef675c9dd0961fbf102de87667a88a292926efe85`,
`07f6b3040cf6ad555ee3c47deee091f9c79ecfcc68ef1755c4fb2237023a29b2`,
and `009b3b7337f35fc420fd46e987254a10ee2803357dc518154b554d69148308c2`.
Confirmation aggregate/rows/sample SHA-256 values are
`1780a0c42019f3ce92009b111ba92a6b1f728b4088552a34879752086315d82f`,
`2b6082efe4ec58d8fdf7743a3aebc253d96f019a429705f9ddc6a9b3d0a2a5fd`,
and `96c88c31b06ad42ff8532ff22c1a2f9fec6eb9415b372a39771a4a3d8980c28e`.
