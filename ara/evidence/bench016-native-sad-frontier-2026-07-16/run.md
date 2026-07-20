# BENCH-016 native SAD frontier

The valid v6 development matrix contains 144 fresh rows. Native SAD passes the frozen nominal
0.5-bpp stratum, but at 2.0 bpp it fails the median `+1 dB`, worst `-0.25 dB`, and LPIPS-nonworse
gates. The joint decision is `abandon SAD reuse`; Stage 1 and production action are unauthorized.

This is an eight-image, downsampled DIV2K development result on one GPU/software environment. SAD's
TXT plus decoder is recipient-replayable, not a self-contained compressed representation.
Protocol v6 was an outcome-responsive integrity repair after v4/v5 target exposure; it retained
the frozen scientific choices, but that design-after-access limitation remains.

Portable proof:

- `analysis.json`: SHA-256
  `ec4aacf2a7c56e76c9e3f0e1ca97e49a43edeeba04e179fe5b61e852ee2e331c`;
- `binding.json`: SHA-256
  `323b41979ca115cf89f5a16579561690ecd4535768f972cb6506d02320cae42b`;
- `replay.json`: SHA-256
  `06231b926b495be57138094aa99e76381c43425e0efc27734c8d3f6ba4765211`;
- `completion.json`: SHA-256
  `558f52655b1043778698d57b3a5c13f073fc3dafecf3af49cfe02a9c6e4fad5d`.
