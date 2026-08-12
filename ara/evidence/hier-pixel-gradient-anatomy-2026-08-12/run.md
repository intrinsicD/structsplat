# Additive HIER gradient-anatomy sanity check

## Scope

This is a deterministic mathematical implementation check, not a quality experiment or method
result. It tests the constant-color, opacity-free, compact-support additive renderer used by the
HIER-031/032 four-array field.

The independent analytic expressions in check.py are compared with autograd through the repository
reference additive renderer for:

- RGB coefficient gradient;
- mean gradient;
- log-scale gradient;
- rotation gradient; and
- the second-order loss change of a symmetric coefficient-halved mean split.

## Execution

Command:

    python ara/evidence/hier-pixel-gradient-anatomy-2026-08-12/check.py

Environment recorded by the result: torch 2.9.0+cu128, CPU float64. The fixture is a deterministic
7-by-7 image with one anisotropic Gaussian, C0 support fade at three sigma, a signed RGB target,
and half squared error.

## Result

All four analytic aggregate gradients match autograd with maximum absolute error below
9e-16. For symmetric split offsets 1e-2, 3e-3, and 1e-3 pixels, the observed loss change divided by
the split-matrix prediction is 0.9999574, 0.9999962, and 0.9999996. The convergence is the expected
local second-order behavior.

The exact machine-readable output is result.json. Re-running check.py asserts a 1e-12 gradient
agreement bound and a 1e-5 finest-offset split-ratio bound.

## Interpretation boundary

This check validates the formulas and their implementation against one deterministic local
fixture. It does not establish that any gradient statistic predicts useful topology operations,
that infinitesimal split curvature predicts finite contained HIER edits, or that the proposed
HIER-033 operator oracle will pass.
