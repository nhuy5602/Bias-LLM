# Open-VLM Visual-Prior Mitigation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Linux-capable, provenance-preserving `open-vlm` track that validates blur-as-prior and evaluates prior-contrastive decoding against separated baselines and controls.

**Architecture:** A new `scripts/open_vlm/` package owns manifest validation, scoring, diagnostics, gate decisions, and runner lifecycle. A thin `scripts/open_vlm_pilot.py` CLI exposes dry-run, smoke, development, and held-out phases. Existing black-box pilots and `scripts/collab.py` are untouched.

**Tech Stack:** Python standard library, Pillow, optional pinned PyTorch/Transformers/Qwen-VL runtime, unittest, pathlib, bash launcher for Linux.

**Spec:** `docs/superpowers/specs/2026-10-21-open-vlm-mitigation-design.md`

## Global Constraints

- Keep scientific status `NEEDS_EVIDENCE`.
- Use VLMBias revision `3761f9fd7163577534a7816c8bc1004035f2e2a0` and verify dataset/model licenses before paid execution.
- Keep source families disjoint; repeats and derived resolutions are not independent samples.
- Primary method is prior-contrastive decoding; primary endpoint reports accuracy, bias rate, and other-error rate together.
- Free-generation and one-forward-pass candidate-scoring baselines are separate.
- Sigma grid is `[1, 3, 6, 12]`; alpha grid is `[0.25, 0.5, 1.0, 2.0]`; randomized-alpha control uses 20 deterministic draws.
- Subject validity must be at least 90%, invalid output at most 5%, and blur detail accuracy at least 10 points below clear before a sigma is usable.
- Development continuation requires at least 5 points bias-rate reduction, accuracy no lower than baseline minus 5 points, and invalid output at most 5%.
- Held-out pass requires at least 10 points bias-rate reduction, accuracy lower-bound non-inferiority at -5 points, and other-error increase at most 5 points.
- Clustered 95% bootstrap intervals use `source_family`; primary held-out composite needs at least 20 eligible families.
- Smoke checks operation only; complete `animals + logos` development data decides continuation.
- No pod deletion before verified artifact upload; no automatic budget top-up.

## Review Focus

- A row with a missing or ambiguous prior-answer field must be rejected, not silently treated as `Other`.
- Clear and blurred Qwen inputs must have identical processor settings and image-token counts.
- A tokenizer that splits a multi-digit answer must score the complete candidate sequence.
- A domain with too few families must not produce a falsely precise bootstrap interval.
- An upload failure must leave the pod undeleted and mark cleanup as blocked.

---

### Task 1: Manifest and Domain Contract

**Files:**
- Create: `scripts/open_vlm/__init__.py`
- Create: `scripts/open_vlm/schema.py`
- Create: `tests/test_open_vlm_schema.py`

**Interfaces:**
- `DomainSpec(name, answer_kind, candidate_values, prior_fields)` is an immutable dataclass.
- `validate_row(row: dict, domain: DomainSpec) -> dict` returns a normalized row or raises `ValueError`.
- `split_families(rows: list[dict], seed: int, dev_fraction: float = 0.8) -> dict[str, list[dict]]` returns deterministic `dev` and `same_domain_test` groups.
- `build_manifest(rows, config) -> dict` records dataset revision, license attestations, family split, seed, processor settings, method grids, and `scientific_status`.

- [ ] **Step 1: Write failing tests** for the fixed VLMBias revision, integer/boolean domain contracts, missing prior fields, disjoint family splits, minimum-family flags, and deterministic manifest serialization.
- [ ] **Step 2: Run `python -m unittest tests.test_open_vlm_schema -v` and verify failure.**
- [ ] **Step 3: Implement the dataclasses and validation with `pathlib` only.** Reject ambiguous rows and never infer a prior from model output.
- [ ] **Step 4: Run the focused tests and verify pass.**
- [ ] **Step 5: Commit** `feat: add open-vlm manifest contract`.

### Task 2: Scoring, Metrics, and Statistical Gates

**Files:**
- Create: `scripts/open_vlm/scoring.py`
- Create: `tests/test_open_vlm_scoring.py`

**Interfaces:**
- `score_candidates(candidates: list[str], sequence_score) -> dict` scores complete candidate strings over all token pieces.
- `classify_answer(prediction, row) -> Literal["Correct", "Bias-aligned", "Other errors"]`.
- `summarize(rows) -> dict` returns counts/rates for all three labels, accuracy, bias rate, and other-error rate.
- `cluster_bootstrap_delta(rows, metric, seed, draws=2000) -> dict` resamples source families and returns estimate plus 95% bounds.
- `evaluate_gate(baseline, method, development: bool) -> dict` returns `PASS`, `FAIL`, or `INCONCLUSIVE` with reasons.

- [ ] **Step 1: Write failing tests** for multi-token answers such as `13`, free-generation versus candidate scoring, all three outcome labels, 2,000 clustered draws, the -5-point non-inferiority lower bound, and the 5/10-point development/held-out gates.
- [ ] **Step 2: Run the focused tests and verify failure.**
- [ ] **Step 3: Implement deterministic scoring and family-cluster bootstrap using the standard library.**
- [ ] **Step 4: Run focused tests and verify pass.**
- [ ] **Step 5: Commit** `feat: add open-vlm scoring and clustered gates`.

### Task 3: Blur, Prior, ROI, and Negative Controls

**Files:**
- Create: `scripts/open_vlm/diagnostics.py`
- Create: `tests/test_open_vlm_diagnostics.py`

**Interfaces:**
- `SIGMAS = (1, 3, 6, 12)` and `ALPHAS = (0.25, 0.5, 1.0, 2.0)` are exported constants.
- `select_blur_sigma(records) -> dict` enforces 90% subject validity, 5% invalid limit, and 10-point detail drop.
- `build_negative_controls(task, seed) -> list[dict]` returns noise, shuffled-prior, empty/black, and 20 randomized-alpha controls.
- `validate_roi(roi, row, kind) -> dict` distinguishes `oracle_roi`, `reference_roi`, and `automatic_roi`; prompts are neutral.

- [ ] **Step 1: Write failing tests** for sigma selection, no qualifying sigma, deterministic 20-draw random alpha, empty/black controls, equal-size controls, and rejection of count-bearing ROI prompts.
- [ ] **Step 2: Run the focused tests and verify failure.**
- [ ] **Step 3: Implement Pillow transforms and provenance-only ROI validation.** Do not call an external grounding model in this task.
- [ ] **Step 4: Run focused tests and verify pass.**
- [ ] **Step 5: Commit** `feat: add blur and control diagnostics`.

### Task 4: Fake Backend, Run Phases, and Budget Ledger

**Files:**
- Create: `scripts/open_vlm/runner.py`
- Create: `scripts/open_vlm_pilot.py`
- Create: `tests/test_open_vlm_runner.py`

**Interfaces:**
- `Backend.run(task: dict, method: str) -> dict` returns prediction, score trace, forward-pass count, timing, memory, and error fields.
- `run_smoke(manifest, backend, limit=50) -> dict` performs operational checks only and may return `SMOKE_FAILED`.
- `run_development(manifest, backend) -> dict` runs animals/logos and invokes the development gate.
- `prepare_run(root, phase, rows, config) -> Path` writes a manifest before any model call.
- `safe_upload_then_delete(pod_id, artifact_dir, destination, uploader, deleter) -> dict` deletes only after remote checksum verification.

- [ ] **Step 1: Write failing tests** for zero-launch dry-run, 50-image smoke limits, launch/retry ledgering, explicit `BUDGET_STOPPED`, redirect to attention diagnostics, upload failure preserving the pod, and successful upload-then-delete.
- [ ] **Step 2: Run focused tests and verify failure.**
- [ ] **Step 3: Implement fake-backend execution, phase state, cost accounting, and cleanup lifecycle.** Keep all output under the new run directory.
- [ ] **Step 4: Add `--dry-run`, `--smoke`, `--development`, `--held-out`, and `--run-dir` CLI modes; run the focused tests.**
- [ ] **Step 5: Commit** `feat: add open-vlm dry-run and budget gates`.

### Task 5: Linux Launcher and Portability

**Files:**
- Create: `run-open-vlm.sh`
- Create: `requirements-gpu.txt` after a compatibility probe verifies exact versions.
- Create: `tests/test_open_vlm_portability.py`

**Interfaces:**
- `run-open-vlm.sh` accepts a phase and run directory, enables strict shell mode, records the PID, uploads artifacts, and invokes cleanup on success or failure.
- The launcher must use POSIX paths internally and never rely on Windows drive parsing.

- [ ] **Step 1: Write failing tests** for POSIX path manifests, bash argument handling, missing uploader, and cleanup-blocked state.
- [ ] **Step 2: Run the focused tests and verify failure.**
- [ ] **Step 3: Implement the launcher and add only verified, pinned GPU dependencies.** Record torch, Transformers, Qwen-VL utility, attention backend, CUDA, and Python versions in the manifest.
- [ ] **Step 4: Run focused tests on Windows and Linux-compatible shell, then `git diff --check`.**
- [ ] **Step 5: Commit** `chore: add open-vlm linux launcher`.

### Task 6: Qwen Baseline and Prior-Contrastive Adapter

**Files:**
- Create: `scripts/open_vlm/qwen_backend.py`
- Create: `tests/test_open_vlm_qwen_backend.py`
- Modify: `README.md` with the new dry-run and Linux commands only.

**Interfaces:**
- `QwenBackend.from_manifest(manifest) -> QwenBackend` loads the pinned model/processor lazily.
- `QwenBackend.run(task, method) -> dict` supports `free_generation`, `candidate_scoring`, `prior_contrastive`, `oracle_roi`, and `automatic_roi`.
- `score_image_pair(clear, prior, candidates, alpha) -> dict` requires equal image-token counts and records both forward passes.

- [ ] **Step 1: Write failing fake-backend tests** for dynamic-resolution parity, multi-token candidate scores, alpha application, forward-pass accounting, and failure when image-token counts differ.
- [ ] **Step 2: Run focused tests and verify failure.**
- [ ] **Step 3: Implement lazy Qwen loading, fixed `min_pixels`/`max_pixels`, `sdpa` or verified attention backend, dtype/device recording, free-generation and candidate-scoring baselines, and prior-contrastive logits subtraction.** No API keys or paid calls.
- [ ] **Step 4: Run focused tests and a zero-launch dry-run.**
- [ ] **Step 5: Commit** `feat: add qwen open-vlm adapter`.

### Task 7: End-to-End Verification and Research Handoff

**Files:**
- Modify: `README.md` with interpretation limits, license checks, and budget stop behavior.
- Modify: `docs/IMPLEMENTATION-REVIEW.md` with the new track review status.
- Create: `tests/test_open_vlm_end_to_end.py`

- [ ] **Step 1: Add an end-to-end fake run** covering manifest, smoke, development gate, held-out composite, metrics, and cleanup state.
- [ ] **Step 2: Run the full suite:** `python -m unittest discover -s tests -v`.
- [ ] **Step 3: Run constraints:** `python scripts/check_research_constraints.py .`.
- [ ] **Step 4: Run `git diff --check` and inspect for secret-like values and accidental controller changes.**
- [ ] **Step 5: Commit** `test: verify open-vlm track end to end`.

## Plan Self-Review

- Spec coverage: all primary/secondary endpoints, domain contracts, family
  counts, token parity, ROI provenance, controls, budget redirection, license
  checks, Linux launcher, and cleanup are assigned to tasks.
- Step scan: every task starts with a failing test and ends with a focused
  verification and atomic commit.
- Type consistency: `Backend.run`, `score_candidates`, `summarize`, and gate
  functions are the shared interfaces used by later tasks.
- Review focus: each listed failure mode has an owning test task.
- Proportion: the plan is additive and keeps the existing controller outside
  the change boundary.
