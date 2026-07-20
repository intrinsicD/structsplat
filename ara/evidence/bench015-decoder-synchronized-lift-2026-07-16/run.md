# BENCH-015 decoder-synchronized affine lift

The canonical Stage-0 artifact is complete and replay-valid. Equal-stream static quality improves
on the smooth families, but the frozen no-harm, terminal-convergence, and cold-decode gates fail.
Stage 1 and another local successor are prohibited.

Portable proof includes the analysis, replay/completion records, complete static/convergence/timing
rows, the stream ledger and all 162 exact DSLR streams, plus executed sources. Core SHA-256 values:

- analysis `62f42f280efaa0e2d0857da55c3ea7f02b507a09ea3eaa95f1bf139c038a88be`;
- replay `55d22739e29c28915ad253bfd1d65a4ef340f059c155beff0873203e143e571e`;
- static rows `a622f4ab15574adc771e249d3d17f65d00eff39aaeb02f777f79ad4f12dbc976`;
- convergence rows
  `93d1f913eeeb8f3713d676899f60d737670ab46ab340a63628c43436c6c99f84`;
- timing rows `6f611602a07f215e7f13f01d1d821dedae42666fd4f501a4a16a6fcabb6cdbec`;
- stream ledger `6114fc5e4bac4fc3245ada3e65296e9d324c339b8a89bbdca1a28b8c02ffb21f`;
- executed sources `11a578e3215ec0a0c641daa2e81bbf55491f0ce3a36e9a07486d8562feb3e97c`.
