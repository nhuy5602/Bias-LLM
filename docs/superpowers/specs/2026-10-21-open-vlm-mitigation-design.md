# Open-VLM Visual-Prior Mitigation - Design Spec

Date: 2026-10-21
Status: proposed - awaiting implementation approval
Scope: a new `open-vlm` experiment track; existing black-box pilots and
`run-collab.ps1` remain unchanged.

## 1. Research question

Can a low-cost inference-time method reduce canonical visual-prior errors on
counterfactual visual counting tasks while preserving accuracy when the prior
is correct and on a small ordinary-VQA sanity set?

The first implementation targets Qwen2.5-VL-3B. LLaVA-OneVision-0.5B is a
follow-up adapter, not a dependency of the first slice. All experiment
summaries remain `NEEDS_EVIDENCE`.

## 2. Required scientific gates

### Step 0: validate the blur-as-prior assumption

Before contrastive decoding, run a cheap diagnostic on a deterministic smoke
subset. For each source family, evaluate clear, matched-resolution blur at a
small sigma grid, and a heavily blurred condition. Record:

- subject recognition validity;
- exact answer, prior answer, and bias-aligned indicator;
- whether blur retains the subject while removing the task detail;
- sigma selected for the method arm.

The sigma is not accepted merely because it increases bias rate. It must retain
subject validity and be selected before the held-out test arm. If blur produces
invalid or noisy outputs, contrastive decoding is marked unsupported for that
model/condition rather than tuned around the failure.

### Domain split

Use source-family disjoint splits. `animals` and `logos` are development
domains; `flags`, `chess`, `boards`, `grids`, and `illusions` are held-out test
domains where available. Sigma, contrastive weight, plausibility thresholds,
ROI settings, and any prompt choice are selected on development data only.
No test output is used for tuning.

### Answer-set scoring

Score candidate answers in a bounded set (default integers `0..20`) rather than
assuming one tokenizer token represents an answer. Decode each candidate as a
complete answer string, aggregate its sequence score, and apply an adaptive
plausibility filter based on the task's allowed answer range and format. Record
the candidate set, tokenization, score components, and rejection reason for
every invalid result.

### Compute-matched controls

Every method reports wall time, forward-pass count, peak memory, and estimated
GPU cost. Baselines include ordinary prompting, ROI, prior-contrastive
decoding, and a self-consistency control using the same number of samples as
the extra forward passes consumed by the method. A two-forward-pass method is
not compared to a one-forward-pass baseline without this accounting.

### Negative controls

The contrastive arm includes:

- unrelated noise or distortion with the same compute and image-size budget;
- a shuffled/unrelated prior image control;
- an alpha-randomized control drawn from the preregistered alpha range.

These controls test whether improvement comes from a meaningful prior rather
than from subtracting arbitrary logits.

### ROI variants

Evaluate two separately labelled variants:

1. `oracle_roi`: ground-truth mask/bounding box, an upper-bound diagnostic.
2. `automatic_roi`: Qwen2.5-VL grounding on the same image, without adding a
   second model or hidden annotation at evaluation time.

The oracle result must not be presented as deployable performance.

### No-regression checks

Run every method on:

- original, unmodified images where the canonical prior is correct; and
- a small ordinary-VQA sanity set (for example a fixed CountBench/POPE/MME
  subset, subject to license and availability verification).

Report both task accuracy and bias rate. A method that reduces counterfactual
bias by damaging ordinary recognition fails the gate.

### Statistical unit and preregistered pass/fail rule

Bootstrap confidence intervals are clustered by `source_family`; repeats and
derived resolutions are never treated as independent samples. The first
registration uses these defaults unless changed before execution:

- pass on an unseen domain only if bias rate improves by at least 10 percentage
  points and the clustered 95% interval for the improvement excludes 0;
- pass no-regression only if original-image accuracy and sanity-set accuracy
  decrease by no more than 5 percentage points, with the same clustered
  interval rule;
- otherwise label the method `FAIL` or `INCONCLUSIVE`, never retune on test.

The thresholds and domain assignment are written to the run manifest before
the first model launch.

## 3. Method order

Implement in this order:

1. deterministic open-VLM manifest and dry-run;
2. baseline + Step 0 blur diagnostic;
3. oracle/automatic ROI baseline;
4. prior-contrastive decoding with candidate answer scoring;
5. negative and compute-matched controls;
6. held-out domain evaluation and clustered bootstrap.

Representation/attention diagnostics are a separate later slice. They may log
layer/attention evidence but do not imply a causal mechanism or change model
weights. LoRA/RL is explicitly out of the first implementation because it
requires a larger budget and creates a domain-overfitting risk.

## 4. Open-VLM track boundary

Create a new schema and runner under `scripts/open_vlm/` (or an equivalently
small package) with no imports from the existing black-box controller. Keep
field names compatible where useful (`family`, `state`, `repeat`, `seed`,
`source_sha256`, `derived_sha256`, `model`, `method`, `scientific_status`), but
do not refactor or modify `scripts/collab.py` or its current pilots.

The manifest records:

- pinned VLMBias dataset revision and source license check;
- domain and source-family split;
- model/revision, processor revision, dtype/device and generation settings;
- method parameters, sigma/alpha grids, candidate answer set and plausibility
  rule;
- seed, task matrix, compute budget, stop gate and actual cost;
- source/derived hashes, prompts, raw outputs, score traces and failures.

GPU dependencies are optional and isolated in `requirements-gpu.txt`; the
dry-run and fake-backend tests must run with the repository's normal Python
environment.

## 5. Budget and RunPod safety

Do not launch a paid GPU until dry-run, fake-backend tests, and the 50-image
smoke test pass. The smoke gate stops and marks the run `BUDGET_STOPPED` when:

- Step 0 cannot validate a usable blur sigma;
- the method does not beat the preregistered development threshold; or
- invalid/failed output exceeds the preregistered limit.

The launcher records every launch/retry and starts a shutdown trap that stops
the pod after the job exits, including failure paths. No automatic `$20` top-up
is performed; a follow-up budget extension must be explicit and ledgered.

## 6. Testing and verification

Unit tests cover manifest validation, domain disjointness, deterministic task
ordering, answer-set/tokenization scoring, plausibility rejection, blur-sigma
selection, ROI provenance, negative controls, compute accounting, clustered
bootstrap, stop-gate decisions, and shutdown command generation. A fake
backend provides deterministic logits and failures; no paid model call occurs
in tests.

Required checks after each implementation slice:

```powershell
python -m unittest discover -s tests -v
python scripts/check_research_constraints.py .
git diff --check
```

The scientific status remains `NEEDS_EVIDENCE` regardless of test or model
agreement.

## 7. Non-goals

- no deletion or refactor of the existing three-CLI controller;
- no silent model, endpoint, dataset, prompt, or budget changes;
- no causal-mechanism or universal-generalization claim;
- no LoRA/RL training in the first slice;
- no tuning on held-out test domains;
- no claim that blur estimates prior until Step 0 passes.

## 8. Self-review

- Placeholder scan: no `TBD` or unbounded tuning step remains.
- Scope: the first slice is a new open-VLM track; existing experiments are
  untouched.
- Consistency: every tunable parameter is selected on development domains and
  recorded before held-out evaluation.
- Evidence: the spec preserves source-family separation, negative findings,
  clustered statistics, and `NEEDS_EVIDENCE`.
