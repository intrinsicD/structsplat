# FF-001: Feed-forward init predictor (warm-start)

**Status: todo (future).**

## Goal
Predict Gaussian parameters (or a placement/probability map) in one forward pass to warm-start the
fitter, using the structure-tensor init as supervision/prior (cf. Instant-GI PPM, Fast-2DGS).

## Acceptance criteria
- [ ] Predicts positions+covariance+color from the image; fitter converges in far fewer iters.
- [ ] Compared against optimized-from-scratch and against structure-tensor init as warm-start.

## Depends on
INIT-003, FIT-001.
