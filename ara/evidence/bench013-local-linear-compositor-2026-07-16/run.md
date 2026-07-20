# BENCH-013 local-linear reproducing compositor

The canonical v3 Stage-0 artifact is complete and replay-valid. Analytic affine reproduction
passes, but signed local leverage/ringing fails 82 of 108 forward cells, so the frozen decision is
`kill`. Natural-image Stage 1 and production integration remain prohibited.

The portable bundle includes the analysis, replay, completion, config, 108 forward cells,
permutation rows, and exact executed sources. Key SHA-256 values are:

- analysis `68db5bc6686f2b3dda430abaa0df6ffdd95698551a99cd0ae5a305cead258791`;
- replay `709d9b55f6535bbb55d61e926706b501dfe7faafd35b320d14778b38dee74ce8`;
- completion `cf2adf32b848e1a661c864ae6e6dab0e2a363416a26411c6374cafd9031d4e4e`;
- forward cells `eff7ea6fe5ab2e0e2a0d7b07f932c151cb25c14c39d96e89581d13b4f905a98a`;
- permutations `022c5a5192d2f84c1eb7f02b2ed4fa444c7d92b6bf4194fb8c8c13b0cf4c34f9`;
- executed sources `38569701e8a1eca60be1e47cad9a90e54e4e756765f5843953d18b5f7f1f21f3`.
