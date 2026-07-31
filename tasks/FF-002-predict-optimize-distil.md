# FF-002 — Predict–optimize–distil with permutation-invariant supervision

## Context

FF-001 closed with an explicit pointer here: "future distillation work should be a new task with
its own evidence." Its `TinyGaussianPredictorNet` (`src/structsplat/predictor.py`) globally
average-pools the image and emits a fixed `num_gaussians * 10` tensor trained with row-wise MSE
against teacher rows from only three Kodak crops. Two of those choices cap the ceiling by
construction: a Gaussian field is an unordered set, so row-i-to-row-i MSE penalizes correct
fields in a different row order, and global pooling discards the spatial layout the positions
must be predicted from. The FF-001 held-out slice quantifies the gap: `learned_tensor` 23.4686 dB
beats `scratch` 22.9056 dB but sits 1.8563 dB under the hand `quadtree_wse` tensor prior at
25.3249 dB (`ara/evidence/ff001-multimage-tensor-ablation-2026-07-07/`). Predict–optimize–distil
plus a set-structured loss is the direct attack on that gap, and several equivalent Gaussian
decompositions can render nearly the same image, so the render loss — not Gaussian matching —
must be authoritative.

## Goal

A spatially structured predictor trained with permutation-invariant supervision and a
predict–optimize–distil loop, screened as four frozen comparators at matched budgets on held-out
images:

- **A** — FF-001 `TinyGaussianPredictorNet` + row-wise MSE (frozen baseline).
- **B** — local predictor + permutation-invariant set loss.
- **C** — B + differentiable render loss (render authoritative).
- **D** — C + predict–optimize–distil.

First architecture is the spatial-grid form: small U-Net/FPN → 16×16 feature map → per-cell
occupancy + K Gaussian slots → global top-N. A CNN feature pyramid with 256–512 learned queries
and a tiny 2-layer cross-attention decoder is the recorded alternative if the grid form
underfits; it is not part of the first screen.

Loss composition for B–D:
`L = lambda_render * L_image + lambda_set * L_sinkhorn + lambda_density * L_occupancy +
lambda_prior * L_orientation`, with set cost
`c_ij = w_mu ||mu_i - mu_j||^2 + w_sigma d_sigma(Sigma_i, Sigma_j) + w_c ||c_i - c_j||^2 +
w_alpha |alpha_i - alpha_j|`.

Distil loop (D): predict `G_0` → run the existing fitter 25 or 50 iterations → detach `G_K` →
train the predictor toward the refined render and refined field → periodically regenerate
refined targets. No backpropagation through the fitting trajectory in this task; offline or
periodically refreshed targets only.

Role-separated teachers, one loss component each, never averaged into one target:

- `quadtree_wse` tensor prior teaches placement, orientation, and coverage;
- the 600-iteration optimized field teaches final appearance;
- the short-refined student field teaches the corrections the student itself needs.

## Non-goals

- Backpropagating through fitter iterations (a possible successor task, only if D wins).
- Elastic/multi-budget prediction (FF-003 owns it, gated on this task's representation).
- Changing any shipped initializer default; `strategy=feedforward` stays experimental.
- Adapterizing or scaling the backbone beyond the two named forms; no SSM/attention rewrites.

## Acceptance criteria

- [ ] Comparators A–D are runnable from logged configs + seeds with matched teacher data,
      training budget, and final N; A reuses the frozen FF-001 checkpoint recipe unchanged.
- [ ] The set loss is exactly permutation-invariant (test: shuffling predicted rows leaves loss
      and gradients identical within tolerance) and the Sinkhorn/matching cost implements the
      frozen `c_ij` weights.
- [ ] Teacher export extends FF-001's protocol beyond three images to a frozen list large enough
      for a train/held-out split, with export cost recorded per teacher role.
- [ ] Held-out screen reports, per comparator: PSNR after 0/25/50/100/200 refinement iterations,
      time to 22/24/25 dB, teacher export cost, predictor inference time, original-prediction
      survival after refinement, and final Gaussian count.
- [ ] NumPy/torch split intact (predictor work stays in torch modules); no init-time module
      imports torch.
- [ ] Outcome (positive or negative) recorded as an ARA observation or claim row citing the
      evidence bundle; Index status updated in the same commit.
- [ ] `./scripts/verify.sh` passes.

## Interfaces touched

`src/structsplat/predictor.py` (new predictor classes beside the frozen FF-001 net),
`src/structsplat/fit.py` (short-refinement hooks only if a seam is missing), teacher
export/training drivers under `scripts/experiments/`, `tests/`, `tasks/INDEX.md`, `ara/`.

## Depends on

FF-001, BENCH-002

## Agent workflow

- Driver: pending
- Reviewer: pending
- Turn: driver
- Reviewed revision: pending

### Handoff log

Append exact `### Handoff` and `### Review` blocks using the schema in `tasks/README.md`.
Before a formal result-bearing run, append the prospective `### Protocol review` block from that
document and bind the exact frozen protocol digest.

## Notes

Comparator ordering isolates one mechanism per step: B−A is the set-loss/locality value, C−B is
render supervision, D−C is distillation. If B fails to beat A, stop and record the negative
before spending the render/distil budget. Success is a screen win, not a default flip; a win
authorizes a confirmation task and, afterwards, FF-003. The FF-001 measured negatives
(image-only input) stay binding: all comparators keep tensor-prior input channels available.
