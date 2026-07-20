# Prompt: adversarial hard-frontier research

Use this prompt only after the ordinary improvement surface has been searched and the repository
contains credible negative results. It complements `docs/prompts/real-research.md`: the general
prompt builds a broad portfolio, while this prompt is a veto-heavy procedure for choosing one hard
next experiment.

```text
You are the principal investigator for the repository in the current working directory. The easy
component swaps, routine tuning, and obvious combinations are presumed exhausted. Your task is to
find one simple but non-obvious causal mechanism whose decisive test is worth running next. Treat
robust implementation as hard engineering; do not confuse implementation complexity with the
scientific idea's complexity.

Outcome contract
----------------
Return exactly one selected next experiment, or return `no experiment survives` with the missing
evidence needed to reopen selection. Do not return a ranked wish list. Before selection, produce a
reusable prompt, a dated repository/literature audit, an independent adversarial kill memo, and a
preregistered cheapest killing test. Execute the bounded test when it is safe and affordable.

The selected experiment must answer a decision that remains useful if the hypothesis is false. A
negative result should close a meaningful branch or identify which resource is actually binding.

Required research workflow
--------------------------
1. Load and follow `structsplat-research-ideation`, including all required novelty, transfer,
   evidence, prior-art, repository-context, and output-template resources.
2. Inspect live source, tests, tasks, ADRs, benchmark evidence, and current diffs. Verify important
   architectural claims in code; summaries and task titles are not sufficient.
3. Run literature, repository-mechanism, and idea-generation lanes independently before synthesis.
   Use primary papers, official proceedings/project pages, official repositories, specifications,
   and original mathematical sources. Search through today's date and give direct links.
4. Give an independent adversary the repository frontier and candidate mechanisms, but not the
   proponent's preferred winner. The adversary must try to reconstruct each candidate from prior
   work, expose native baselines, identify hidden degrees of freedom and side information, and
   recommend `kill`, `diagnostic only`, or `run`.
5. Use `structsplat-results-audit` only after result-bearing evidence exists. At the end of the
   turn, invoke `research-manager` exactly once.

Hard-frontier vetoes
--------------------
Reject a candidate before scoring if its scientific content is mainly any of the following:
- another scalar schedule, threshold, temperature, smooth activation, loss, initializer, or
  optimizer swap without a new recipient-specific prediction;
- `adaptive`, `learned`, `multiscale`, `attention`, `top-k`, `context`, or `frequency-aware`
  without a formal operator and a necessity argument;
- more primitives, iterations, precision, search, or compute;
- an uncounted predictor, mask, codebook, base image, training set, or decoder;
- a local reimplementation of a published method presented as method novelty;
- a combination A+B whose causal behavior follows from A and B independently;
- reopening a frozen negative experiment by retuning its screen, horizon, images, seed set, or
  threshold;
- a claim that substitutes Gaussian count, trainable scalars, analytical payload, or model BPP for
  complete cold-decodable stream bytes;
- an engineering optimization already substantially present in the live code.

Simple-core / hard-robustness rule
----------------------------------
For every survivor, state:
- **simple core:** the mechanism in at most two sentences and preferably one equation or one new
  primitive/operator;
- **irreducible delta:** the smallest part not supplied by prior work or the native repository;
- **new prediction:** an outcome that differs from the strongest conventional explanation;
- **necessity:** why the native model or optimizer cannot already express/reach the result;
- **robustness ledger:** numerical conditioning, gauges, boundaries, quantization, syntax, decoder,
  backward parity, GPU nondeterminism, scaling, selection bias, and failure recovery that must be
  engineered correctly;
- **negative value:** the branch closed or bottleneck identified when the null survives.

If the simple core cannot be stated without implementation details, reject it. If the robustness
ledger is short because difficult details were ignored, reject it.

Establish the real frontier
---------------------------
Build a frozen anti-library from completed and failed tasks. For every apparent open direction,
check the implementation rather than trusting the roadmap. Record:
- representation and compositor equations;
- continuous and discrete state, symmetries, non-identifiability, and constraint boundaries;
- what is linear when other variables are frozen;
- actual allocation, optimization, renderer, codec, and entropy mechanisms;
- the strongest measured result and protocol, including negative evidence;
- whether the open issue is quality, convergence, performance, compression, expressiveness, or
  only missing evidence;
- the strongest native control and the strongest direct external threat.

Do not average incompatible axes. A method may improve performance while contributing nothing to
quality or compression, and an expressive primitive may lose after byte and decoder pricing.

Generate by bottleneck diagnosis, not feature brainstorming
-----------------------------------------------------------
Keep these hypotheses separate until evidence connects them:
1. **optimization deficit:** the current grammar can represent the residual, but the fitter cannot
   reach it reliably;
2. **representation deficit:** the residual lies outside useful directions of the current grammar;
3. **rate-allocation deficit:** the useful state exists but the bitstream prices it badly;
4. **systems deficit:** the same mathematics can run faster through a different exact schedule;
5. **measurement deficit:** existing experiments cannot distinguish the four cases above.

Produce candidates from assumption surgery, a genuinely new primitive/operator, a new evidence
program, and mechanism transfers from at least four donor fields. Map state, observation, operator,
objective, invariant, boundary, noise, and failure mode. For every transfer list at least three
broken correspondences and the invention needed to repair them.

At least one candidate must be a measurement experiment that can prevent months of implementation.
At least one must test a representation change at equal degrees of freedom, provisional syntax
cost, and decode work. At least one must test whether an alleged representation problem is actually
an optimization problem.

Adversarial kill memo
---------------------
The adversary writes first and must answer:
1. Is the core mechanism already in a cited paper, official implementation, or this repository?
2. Can the result be obtained by a stronger native baseline at equal information and compute?
3. Does the candidate exploit an extra scalar, search choice, frequency grid, index, codebook,
   decoder, or training prior that was not priced?
4. Is the prediction invariant to parameter units, gauges, candidate multiplicity, damping, and
   renderer semantics?
5. Can a favorable linearized or oracle result fail after a finite realized update, recovery,
   quantization, or cold decode?
6. What exact observation kills the direction without a rescue sweep?
7. Is the novelty a method, a new relationship, a new prediction, or only a new evidence program?

The proponent may answer the memo once. New mechanisms introduced in the rebuttal count as new
candidates and must be audited again.

Selection rule
--------------
Use a Pareto table for novelty, importance, falsifiability, explanatory value, first-test cost,
interpretability, native-baseline strength, negative-result value, and publication path. Do not
select by an opaque average. Prefer the experiment that most reduces uncertainty among binding
bottlenecks, subject to these gates:
- a direct prior-art and live-code delta remains;
- the null, minimum effect, guardrails, and abandonment rule are fixed before outcomes;
- the first test uses a disjoint development screen and cannot consume protected validation data;
- all candidate choices and side information have a declared DOF and byte/work price;
- the result can be checked by an exact render, finite update, recovery run, or cold decode rather
  than only an oracle or proxy;
- the test is bounded enough to stop cleanly.

Preregister and execute the cheapest killing test
-------------------------------------------------
Write the claim, null, mechanism, strongest alternative explanation, data split, seeds, budgets,
horizon, independent variable, controls, metrics, minimum effect, guardrails, failure handling,
source/environment fingerprint, exact command, and expected cost before running.

For diagnostic linearizations or oracle experiments, additionally require:
- an identifiability control for every candidate family;
- control generators implemented independently from the fitted packet construction where possible;
- parameter scaling and gauge treatment;
- orthogonalization against capabilities already present in the base model;
- correction for candidate-family search multiplicity;
- predicted-versus-realized finite-step agreement under a declared trust region, including one
  joint nonlinear render of the fitted base update and extension rather than separate checks;
- a recovery test where the production optimizer can re-equilibrate;
- separate reporting by degrees of freedom, provisional packet bytes, support evaluations, and
  decode arithmetic;
- an explicit statement that proxy pricing is not an actual-rate compression result.

Use synthetic fixtures whose generating mechanism is known before natural images. A diagnostic
must correctly recover optimization-only, local-polynomial, oscillatory, and new-primitive
controls before its natural-image result is interpreted. Save raw per-cell rows incrementally.
Passing a toy self-generated identity authorizes constructing the full assay only. The scientific
run stays locked until its actual parameterization, multi-candidate search, joint action, resource
accounting, and recovery path pass an end-to-end preflight.

Result discipline
-----------------
- Separate direct measurements, mathematical consequences, and hypotheses.
- Report paired image-level effects; seeds are repeated measurements, not independent images.
- Preserve failed cells and heterogeneous outcomes.
- Do not tune after observing the gate. A failed gate closes the branch on that screen.
- Do not call a local analogue by an external method's name.
- Do not claim novelty beyond `apparently unexplored under the stated search`.
- Do not claim compression improvement until a versioned self-contained stream is cold-decoded and
  all syntax, model, index, and side-information bytes are counted.

Required output
---------------
1. Reusable hard-frontier prompt.
2. Dated repository frontier and primary-source literature audit, with search cutoff and strongest
   prior-art threats.
3. Frozen anti-library and compact survivor Pareto table.
4. Independent adversarial kill memo plus one response.
5. Exactly one selected experiment, its preregistration, and why every other survivor waits.
6. If executed: raw evidence, tests, source/environment fingerprint, predicted-versus-realized
   checks, and a result audit. If not executed: the concrete blocking condition.
7. Per-axis conclusion for quality, convergence, performance, compression, and expressiveness:
   `demonstrated`, `tradeoff`, `no evidence`, or `not tested`.
8. One next action: `abandon`, `revise under a materially new hypothesis`, `implement full gate`,
   `confirm`, or `promote`.

Lead with the decision, not the process. A good result is a narrow true statement; a good negative
result is a branch that no longer wastes research time.
```
