# BENCH-014 explicit affine carrier

The canonical Stage-0 artifact is complete and replay-valid. The gauge-fixed carrier improves
synthetic static quality/rank and has bounded prepared-render cost, but its six transmitted
binary16 scalars plus residual-color entropy fail every complete-byte gate; three bump cells also
fail the terminal-convergence guard. The transmitted-tail formulation is closed.

Portable proof includes `analysis.json`, replay/completion records, convergence pairs, the stream
ledger and all 144 exact AFCR streams, the artifact manifest, and executed sources. Core SHA-256
values are:

- analysis `cb33e79204e7e1eca8be80fe8cb8591c968d59afeb0f7b5f3a1df9af86a0a6c3`;
- replay `f93c571bb554eccd94fde7b347751d99316526a9d6bb753c5da6b6f992f4e024`;
- artifact manifest `3a49fb7cab3071d95e2943337c21c28d4c2601b61fcce8853a98f544015c511a`;
- convergence pairs
  `060f2cb0693084bbc86128830ea5534910be39f1d7caf1c5b182c392c10d63ce`;
- stream ledger `5656b0154c28cfb0817842a992b8ab0aa3a590b2aad8b2b941f48577e07eeff3`;
- executed sources `1c0d9cceb7f87ce2db74592d4f6f8c4dbeb019d3217b36f96ea2f5867beee8e3`.
