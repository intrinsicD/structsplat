# COMP-007 gauge-free covariance codec

The audited v4 development run replayed 12,096 complete streams. The log-SPD chart fails seven of
eight frozen gates; median whole-container movement is `-0.4053%` under zlib and `+0.3426%` under
zstd versus the required `+1%`. Confirmation remains sealed.

`decision_summary.json` is a mechanical projection of the 18,472,928-byte canonical analysis
(SHA-256 `115c2e272a406b1d85313496a94c76e6a4f47c59e41b79e13f951f0ff464ea27`)
containing the binding/checks, decision, gates, and per-image results with nested per-coder
summaries. The bundle also
preserves the config, field ledger, artifact audit, and executed source archive. Their SHA-256
values are
`fbf1846492a930faf556c3ce9b8c98927c32bb1a20055ac91cc30ed9f52113f9`,
`a8d0c60a5cb5dd8b5bb1027f284386bb6261bb1f722460638d8f88a694f7f40a`,
`ad1ec6c889e818e4b1af4cc63a4f99959453534951f480554471bf14fc621aa5`,
and `cb6d0d8eb77b6b98328dd66290d8a514a746e00de07c10a2d23cc62959eb9753`.

The 12,096 streams are not vendored here, so this is portable decision/audit evidence rather than
a clean-clone cold-stream replay corpus.
