# Implementation Plan: Mercedes-Benz Generalization Pilot

## Overview

Extend the existing logo replication runner with an opt-in third-brand arm. Use
the same pinned dataset revision, baseline/evidence-first conditions, three
repeated readings, Codex configuration, and Gemini handoff protocol. The new arm
tests behavioral generalization only; it does not establish a causal mechanism.

## Task List

### Phase 1: Runner and protocol

- [x] Add an opt-in `--include-mercedes` family set with four verified black
  original/counterfactual pairs.
- [x] Add tests for family aliases, brand-specific prompts, task counts, and the
  unchanged ten-family default.

### Checkpoint: Protocol

- [x] Dry-run produces the expected manifest and Gemini prompt with zero model
  launches.
- [x] Dataset revision and source-image family metadata remain pinned.

### Phase 2: Bounded execution

- [x] Create the Gemini prompt before execution and run the Codex arm with the
  pinned model/configuration.
- [x] User runs the Gemini prompt independently in Antigravity and writes the
  report into the same run folder.

### Checkpoint: Evidence

- [x] Verify full hashes, task counts, state labels, repeats, and raw outputs.
- [x] Keep `NEEDS_EVIDENCE` and preserve disagreement/null findings.

### Phase 3: Synthesis and map

- [x] Synthesize third-brand results without causal or universal claims.
- [x] Update Graphify artifact links, condition nodes, and next evidence gaps.

## Risks and Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Mercedes sedan lacks an original row | High | Use only the four complete pairs; record the exclusion. |
| Original/CF images are metadata-matched, not pixel-paired | High | Keep this limitation explicit; do not call the result causal. |
| Gemini report exposes only truncated hashes | Medium | Treat full-hash verification as unresolved and preserve raw manifest. |
| More tasks increase paid/runtime cost | Medium | Keep three repeats, pin model/config, checkpoint every task, and record ledger. |

## Open Questions

- Does the Codex prior-aligned behavior extend to Mercedes-Benz?
- Does the Gemini/Codex discrepancy replicate across a third brand while the
  same condition intervention remains ineffective?
- Which pixel-paired or sham-control design should follow this screening arm?

### Pixel-paired sham-control follow-up

- [x] Define six fixed source families with shared normalized source provenance.
- [x] Add original, logo-erased, and adjacent sham-erased conditions with three repeats.
- [x] Add ROI, source/normalized/derived hash, and dimension tests before execution.
- [x] Create corrected Gemini prompt before execution and run Codex on the same manifest.
- [x] Audit corrected Codex/Gemini reports; preserve model disagreement and `NEEDS_EVIDENCE`.
- [ ] Replace synthetic blur erasure with a naturalistic pixel-paired logo edit or
  an explicitly preregistered mask/texture intervention before causal claims.
