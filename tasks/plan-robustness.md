# Implementation Plan: Visual Signal Robustness Pilot

## Overview

Measure when visual evidence degrades enough for the canonical brand prior to
override it. Reuse six fixed families (Audi/Maserati/Mercedes-Benz coupe and SUV),
both original and counterfactual images, three derived maximum-edge levels, and
three isolated repeats per task. This is a behavioral threshold pilot, not a
causal mechanism test.

## Tasks

### Phase 1: Preparation

- [x] Add a deterministic Pillow transform with source and derived SHA256 values.
- [x] Add a Codex-only runner with prepared-run reuse and a Gemini prompt.
- [x] Add tests for family selection, transform metadata, task matrix, and prompt
  provenance.

### Checkpoint: Prepared protocol

- [x] Dry-run creates 108 tasks, zero model launches, and a complete Gemini prompt.
- [x] All derived variants point to the same six source families and pinned source
  manifest.

### Phase 2: Execution

- [x] Run Codex on the prepared run directory with the pinned model/configuration.
- [x] User runs the Gemini prompt independently in Antigravity.

### Phase 3: Audit and synthesis

- [x] Recompute source and derived hashes; parse both model outputs by brand,
  state, and resolution.
- [x] Report exact accuracy, prior-aligned rate, and any transition point without
  claiming causality.
- [x] Update Graphify and keep `NEEDS_EVIDENCE`.

### Gaussian-blur follow-up

- [x] Prompt created before execution with full source/derived hashes.
- [x] Gemini completed 108/108 valid readings.
- [x] Codex completed 104/108 valid readings; four missing outcomes retained in
  the ledger and summary.
- [x] Candidate blur transition recorded without causal overclaim.

### Matched-resolution Gaussian-blur correction

- [x] Identify and document the native-resolution confound in the prior blur run.
- [x] Add opt-in `gaussian-blur-matched` mode with source normalization before blur.
- [x] Preserve source, normalized, and derived hashes plus all three dimension sets.
- [x] Add failing-first unit/integration tests for the 108-task matched matrix.
- [x] Generate the matched prompt and zero-launch dry-run artifact.
- [x] Complete Codex and independent Gemini runs on the same manifest.
- [x] Recompute hashes and report results by brand, state, and sigma without
  upgrading `NEEDS_EVIDENCE`.

### Matched-resolution sigma-zero baseline

- [x] Add an opt-in sigma-zero no-blur baseline without changing legacy blur modes.
- [x] Preserve byte-identical normalized/derived PNGs and equal hashes for sigma zero.
- [x] Generate protocol `vlmbias-logo-gaussian-blur-matched-v2` with 144 tasks per model.
- [x] Complete Codex and independent Gemini runs on the v2 manifest.
- [x] Audit baseline-to-blur deltas and preserve model disagreement/non-results.

## Risks

- Downsampling changes both signal quality and geometry visibility; use one fixed
  transform and label the result as a robustness screen.
- The prior Gaussian-blur run mixed native source resolutions with 384px
  counterfactuals; treat it as provisional until the matched-resolution run is
  audited. Protocol v1 had no sigma-zero cell; v2 supplies an in-manifest
  no-blur baseline, but it remains a repeated-reading condition rather than an
  independent sample or causal control.
- The v2 matched protocol adds sigma zero as an in-manifest no-blur baseline;
  it is a repeated reading condition, not an independent sample or a causal
  control.
- The six-family subset is not a full benchmark; preserve family-level results.
- A flip under corruption does not prove an internal causal mechanism.
