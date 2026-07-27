# StructSplat — Claude Code project guide

Hierarchical, feature-aware, **anisotropic blue-noise 2D Gaussian research substrate**. A single
image is represented by oriented 2D Gaussians under a normalized compositor. BENCH-007 rejected
the training-free tensor-metric/WSE structural-prior compression claim at its development gate.
The repository remains an interpretable causal substrate for representation, ownership, renderer,
and codec hypotheses; none is currently a SOTA or default-method claim. Structure-aware
allocation, orientation, progressive coding, and generic Gaussian compression are not blanket
novelty claims. See BENCH-007 and the 2026-07-13 SOTA audit.
PyTorch reference plus exact CUDA; the sampler/rasterizer remain CUDA/Vulkan + IntrinsicEngine port
targets.

> `structsplat` is a placeholder project name. If it changes, follow the `structsplat-docs-sync` skill.

## Skill-aware routing (load the skill, then act)
This repo ships eight project skills in `.claude/skills/`. Load them by task — do not reimplement
their guidance inline.

| When you are… | Load skill |
|---|---|
| Starting any session / orienting / deciding where code goes | **structsplat-core** (always first) |
| Picking up or closing a task from `tasks/` | **structsplat-task-workflow** |
| Adding/changing an init strategy, renderer, sampler, hierarchy | **structsplat-method** |
| Developing novel research directions, cross-domain transfers, or falsifiable research portfolios | **structsplat-research-ideation** |
| Running/extending the ablation, or wiring a fitness signal | **structsplat-benchmark** |
| Auditing results, claims, benchmark bundles, or a results-bearing change | **structsplat-results-audit** |
| Reviewing a diff or self-reviewing before commit | **structsplat-review** |
| A change touches documented behavior, decisions, or task status | **structsplat-docs-sync** |

Every skill name is `structsplat-`prefixed so it cannot collide with another repository's skill
when several repos are open in one agent session. An unprefixed name here (`core`, `review`,
`docs-sync`) would be shadowed by, or shadow, a sibling repository's skill of the same name.

Typical flow: `structsplat-core` → `structsplat-task-workflow` (open the task) →
`structsplat-method` (if adding a component) → `structsplat-review` (before commit) →
`structsplat-docs-sync` (same commit). A results-bearing flow inserts `structsplat-benchmark`
→ `structsplat-results-audit` before `structsplat-review`. Explicit invocation:
"use the structsplat-method skill". For open-ended research discovery: `structsplat-core` →
`structsplat-research-ideation`; a selected candidate then re-enters `structsplat-task-workflow`
→ `structsplat-method` → `structsplat-benchmark` → `structsplat-review` →
`structsplat-docs-sync`.

## Non-negotiable invariants (full list in the `structsplat-core` skill)
1. Init-time math (`structure_tensor`, `density`, `sampling`) is **NumPy and importable without
   torch**. Autograd lives in torch modules only.
2. Images `(H,W,3)` float32 in `[0,1]`; positions `(x,y)` pixel coords.
3. The renderer is **normalized** (ADR-0003). Additive/residual compositing requires a new ADR.
4. Everything reproducible from a logged config + `InitConfig.seed`.

## Layout
`src/structsplat/` package · `tests/` pytest · `benchmarks/` ablation + fitness ·
`docs/adr/` decisions (cited as `ADR-NNNN`) · `docs/architecture.md`, `docs/theory.md`,
`docs/comparison.md` (external-method comparison), `docs/blockers_and_external_techniques.md`
(known blockers and borrowed techniques), `docs/publication_figures.md` ·
`docs/research/` dated session records · `tasks/` work items + `INDEX.md` ·
`ara/` claim + evidence ledger (see below) · `scripts/` durable tooling, with one-off experiment
drivers under `scripts/experiments/`. The supported operational scripts are
`convert.py`, `benchmark.py`, `ablation.py`, and `stage_search.py`; historical launchers live in
`deprecated_scripts/`.

## Evidence and claims (`ara/`)

`ara/` is the Agent-Native Research Artifact: this repository's claim and evidence ledger. It is
where a number becomes a claim you are allowed to repeat — and, just as often, where a claim is
recorded as refuted.

```
ara/PAPER.md                   root manifest and layer index; start here
ara/logic/claims.md            the claim ledger: C<NN> rows, each with a falsification criterion
ara/logic/problem.md           the research problem statement
ara/logic/concepts.md          crystallized concepts
ara/logic/solution/            heuristics and solution notes
ara/staging/observations.yaml  O<NN> observations awaiting promotion to a claim
ara/trace/                     exploration_tree.yaml + trace/sessions/ session history
ara/evidence/                  per-experiment evidence bundles and the evidence index
```

**When to touch it.** Before a quantitative or capability statement enters `README.md`, `docs/`,
or a task status line, it needs a `ara/logic/claims.md` row whose `Proof` cites a tracked
artifact under `ara/evidence/`, `benchmarks/`, or `tests/`. A claim row carries nine fields:
`Statement`, `Status`, `Provenance`, `Crystallized via`, `Falsification criteria`, `Proof`,
`Dependencies`, `Tags`, `From staging`.

`Status` starts with a disposition word — `supported`, `refuted`, `untested`, `unavailable`,
`hypothesis`, `superseded`, `withdrawn` — optionally followed by a scope qualifier, as in
`refuted development actual-rate claim`. A `supported` or `refuted` claim must cite at least one
artifact path that exists on disk.

`python scripts/check_ara.py` enforces the structure: required layer files, PAPER.md index
targets, claim-field completeness, status vocabulary, dependency resolution, proof-path
existence, and staging-ID resolution. It cannot judge whether a sentence overstates its
artifact — that is `structsplat-review`, and `structsplat-results-audit` for anything promoted.

BENCH-007 is the model entry: a rejected headline claim, recorded as rejected, with the gate that
killed it. Keep that standard.

## Environment
`pip install -e .` (torch, numpy, pillow, imageio). Optional metrics: `pip install -e ".[metrics]"`
(lpips, pytorch-msssim). Dev: `pip install -e ".[dev]"` then `pytest -q`.
Reference code is CPU-correct but slow at large N; use GPU and small budgets while iterating. The
remaining production/tiled CUDA/Vulkan/RHI work is `PORT-001`/002/003; ADR-0011 owns the exact
CUDA research renderer.

## Verify
Run `./scripts/verify.sh` before every commit: `ruff check` + `ruff format --check` +
`pytest -m "not slow and not integration"` + `scripts/docs_sync.py` (structural docs↔code gate) +
`scripts/check_ara.py` (claim ledger) + `scripts/check_task_policy.py` (task tree) +
`scripts/check_script_layout.py` (scripts layout). CI mirrors these steps on CPU.

`ruff check` enforces the correctness baseline pinned in `pyproject.toml`. The broader style and
import ruleset is a separate repo-wide ratchet owned by `DOCS-003`; see that task for the current
stage and its expiry.

## Definition of done (short form)
Acceptance criteria tested · NumPy/torch split intact · ADR for any real decision · docs updated in
the same commit · results reproducible.
