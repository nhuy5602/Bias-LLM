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

The primary method is prior-contrastive decoding. The primary endpoint is a
joint report of accuracy, bias rate, and other-error rate on the held-out
domain composite. A method cannot pass by moving answers from
`Bias-aligned` to a different wrong answer: accuracy must be non-inferior and
other errors must not materially increase. ROI, self-consistency, noise,
shuffled-prior, and randomized-alpha arms are secondary controls.

The VLMBias dataset revision is pinned to
`3761f9fd7163577534a7816c8bc1004035f2e2a0`. A preflight must record and pass
the dataset and model licenses before any paid launch.

## 2. Required scientific gates

### Step 0: validate the blur-as-prior assumption

Before contrastive decoding, run a cheap operational smoke on 50 images, then
run the complete development set. The smoke only checks loading, token-count
parity, output format, and failure rate; it never decides scientific success.
For the complete development run, evaluate clear, matched-resolution blur at
sigma values `[1, 3, 6, 12]` (sigma 12 is the heavy condition). Record:

- subject recognition validity;
- exact answer, prior answer, and bias-aligned indicator;
- whether blur retains the subject while removing the task detail;
- sigma selected for the method arm.

The selected sigma is the smallest value satisfying all of these preregistered
development rules: subject-validity rate at least 90%, invalid output rate at
most 5%, and blur detail accuracy at least 10 percentage points below the clear
condition. If more than one value qualifies, choose the smallest. If none
qualifies, contrastive decoding is marked unsupported for that model/condition
rather than tuned around the failure.

### Domain split

Use source-family disjoint splits, including a deterministic within-domain
split: 80% of families for development and 20% for a same-domain check (at
least two families when a domain has five or more). `animals` and `logos` are
development domains; `flags`, `chess`, `boards`, `grids`, and `illusions` are
held-out test domains where available. Sigma, alpha, plausibility, ROI
settings, and prompts are selected only on development data. No test output is
used for tuning.

Before execution, the manifest must count source families. A domain with fewer
than five families is descriptive only. The primary held-out endpoint is a
single pre-registered composite of all eligible held-out domains with at least
20 total families; individual domains remain descriptive when their cluster
count is too small for a stable interval.

The domain contract is explicit:

| Domain | Answer set | Prior field | Eligibility |
|---|---|---|---|
| animals/logos/flags/boards | integer range from dataset metadata, default `0..20` | `expected_bias` | require integer `ground_truth` and `expected_bias` |
| grids | task-specific set from the row schema | `expected_bias` or verified bias-answer field | fail closed if the schema is ambiguous |
| illusions | `{yes, no}` | verified categorical bias-answer field | never coerce to integers |

The loader rejects any row whose answer type or prior-answer field cannot be
verified from the pinned dataset. It does not infer a prior from the model's
output.

### Answer-set scoring

Keep two scoring baselines separate: (a) free generation, matching the
original benchmark protocol, and (b) one-forward-pass candidate scoring. The
contrastive method uses only (b), so any gain can be compared against the
same scoring constraint without attributing the constraint's effect to
contrastive decoding.

For candidate scoring, use complete answer strings from the task's registered
answer set, aggregate sequence log-probability over all tokenizer pieces, and
apply only deterministic plausibility checks: valid format and membership in
the registered set/range. There is no hidden score-margin tuning. Record the
candidate set, tokenization, score components, and rejection reason for every
invalid result. Default integer candidates are `0..20`; other domains use the
table above.

### Compute-matched controls

Every method reports wall time, forward-pass count, peak memory, and estimated
GPU cost. Baselines include free generation, one-forward-pass candidate
scoring, ROI, prior-contrastive decoding, and self-consistency with the same
number of samples as the extra forward passes consumed by the method. A
two-forward-pass method is not compared to a one-forward-pass baseline without
this accounting.

### Negative controls

The contrastive arm includes:

- unrelated noise or distortion with the same compute and image-size budget;
- a shuffled/unrelated prior image control;
- an alpha-randomized control with 20 deterministic draws from the registered
  alpha range;
- an empty/black image prior with the same processor settings.

These controls test whether improvement comes from a meaningful prior rather
than from subtracting arbitrary logits.

### ROI variants

Evaluate two separately labelled variants:

1. `oracle_roi`: only when a dataset-provided mask/bounding box is verified.
   Otherwise use `reference_roi` for an externally created annotation and never
   call it ground truth.
2. `automatic_roi`: Qwen2.5-VL grounding on the same image, without adding a
   second model or hidden annotation at evaluation time.

The ROI prompt is neutral (for example, "locate the logo" or "locate the
animal") and must not contain a count or expected answer. The oracle/reference
result is an upper-bound diagnostic, not deployable performance.

### No-regression checks

Run every method on:

- original, unmodified images where the canonical prior is correct; and
- an expanded original-image set with at least 20 source families and at least
  200 images, including code-generated originals for flags and boards when the
  benchmark's original split is too small; and
- a small ordinary-VQA sanity set (for example a fixed CountBench/POPE/MME
  subset, subject to license and availability verification).

Report `Correct`, `Bias-aligned`, and `Other errors` (the remainder) separately,
plus accuracy and bias rate. A method that reduces counterfactual bias by
damaging ordinary recognition fails the gate.

### Statistical unit and preregistered pass/fail rule

Bootstrap confidence intervals are clustered by `source_family`; repeats and
derived resolutions are never treated as independent samples. The first
registration uses these concrete values unless changed before execution:

- development continuation requires at least a 5 percentage-point bias-rate
  reduction, point-estimate accuracy no lower than 5 points, and invalid output
  rate at most 5%; otherwise stop the primary method and redirect remaining
  budget to the attention/logit-lens diagnostic slice;
- pass on the primary held-out composite only if bias rate improves by at least
  10 percentage points, accuracy is non-inferior, and other-error rate does not
  increase by more than 5 points;
- no-regression means the lower bound of the clustered 95% interval for
  `method_accuracy - baseline_accuracy` is at least `-5` points on both the
  expanded original set and the sanity set. The interval need not exclude 0;
- the alpha grid is `[0.25, 0.5, 1.0, 2.0]`; the selected value is fixed on
  development data. The sigma grid and answer-set rules are fixed above;
- otherwise label the method `FAIL` or `INCONCLUSIVE`, never retune on test.

The thresholds and domain assignment are written to the run manifest before
the first model launch.

## 3. Method order

Implement in this order:

1. deterministic open-VLM manifest and dry-run;
2. 50-image operational smoke;
3. complete development run with free-generation and candidate-scoring
   baselines plus Step 0 blur diagnostic;
4. oracle/reference and automatic ROI baselines;
5. prior-contrastive decoding with candidate answer scoring;
6. negative and compute-matched controls;
7. held-out domain evaluation and clustered bootstrap.

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
- `min_pixels`, `max_pixels`, image-grid/token counts, Transformers/Qwen-VL
  versions, attention implementation, and dtype; clear and blur variants must
  have identical processor settings and image-token counts;
- method parameters, sigma/alpha grids, candidate answer set and plausibility
  rule;
- seed, task matrix, compute budget, stop gate and actual cost;
- source/derived hashes, prompts, raw outputs, score traces and failures.

GPU dependencies are optional and isolated in `requirements-gpu.txt`; the
dry-run and fake-backend tests must run with the repository's normal Python
environment. All new code uses `pathlib` and is exercised on Linux through a
checked-in bash launcher as well as PowerShell.

## 5. Budget and RunPod safety

Do not launch a paid GPU until dry-run, fake-backend tests, and the 50-image
smoke test pass. The smoke gate only marks `SMOKE_FAILED` for operational
failure; scientific continuation is decided only after the complete animals +
logos development run. That development gate stops and marks the run
`BUDGET_STOPPED` when:

- Step 0 cannot validate a usable blur sigma;
- the primary method misses the concrete development thresholds above; or
- invalid/failed output exceeds the preregistered limit.

The launcher records every launch/retry and starts a shutdown trap. Before pod
deletion it uploads the manifest, summary, logs, and checksums to the explicitly
configured artifact destination, verifies remote checksums, then deletes the
pod and its volume. If upload or verification fails, it does not delete the
pod and records `CLEANUP_BLOCKED`. No automatic `$20` top-up is performed; a
follow-up budget extension must be explicit and ledgered. If the primary gate
fails, remaining budget is reserved for the attention/logit-lens diagnostic
slice rather than silently discarded.

## 6. Testing and verification

Unit tests cover manifest validation, domain disjointness, deterministic task
ordering, free-generation versus candidate-scoring baselines,
answer-set/tokenization scoring, plausibility rejection, blur-sigma selection,
empty/black priors, ROI provenance, negative controls, compute accounting,
clustered bootstrap, non-inferiority bounds, stop-gate decisions, license
preflight, Linux paths, and upload-then-delete command generation. A fake
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
- no model or dataset use before license verification;
- no pod deletion before verified artifact upload.

## 8. Self-review

- Placeholder scan: no `TBD`, unspecified threshold, or unbounded tuning step
  remains; all initial sigma, alpha, invalid-rate, family-count, sample-count,
  bootstrap and non-inferiority values are concrete above.
- Scope: the first slice is a new open-VLM track; existing experiments are
  untouched.
- Consistency: every tunable parameter is selected on development domains and
  recorded before held-out evaluation; free-generation and candidate-scoring
  effects are separated.
- Evidence: the spec preserves source-family separation, negative findings,
  clustered statistics, token-count parity, license checks, cleanup evidence,
  and `NEEDS_EVIDENCE`.
