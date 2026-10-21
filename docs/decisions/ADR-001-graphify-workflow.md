# ADR-001: Keep Graphify and Gemini Handoffs Deterministic

## Status

Accepted

## Date

2026-10-19

## Context

The project uses a two-model research workflow: Codex executes bounded local
research tasks, while the user runs Gemini independently in Antigravity from a
durable prompt. Graphify is needed to preserve the workflow architecture across
sessions and to expose paths between experiment artifacts, evidence gates, and
reports.

The Graphify package can automatically choose a Gemini API backend when
`GEMINI_API_KEY` is present. That would silently cross the established model
boundary, make the handoff less independent, and make provenance and cost harder
to audit.

## Decision

Use a deterministic Graphify maintenance path:

1. Run Graphify with `GRAPHIFY_BACKEND=none` for local structural extraction.
2. Maintain the research workflow and condition-level result map in
   `collab/graphify/ARCHITECTURE.md` as a small source-oriented manifest.
3. Add the manifest and existing experiment artifacts to `graphify-out/graph.json` with
   `scripts/augment_graphify_architecture.py`.
4. Recluster and regenerate report/HTML/wiki artifacts with
   `GRAPHIFY_BACKEND=none graphify cluster-only . --no-label`, then run the
   augmenter once more so source-state metadata is retained.
5. Use `collab/graphify/GEMINI-PROMPT.md` for independent Gemini review. Gemini
   writes `GEMINI-REPORT.md`; it does not edit the graph or source files.

## Alternatives considered

### Automatic Gemini semantic extraction

Rejected. It is convenient, but it silently turns a Codex run into a Gemini API
run whenever a machine credential exists. It also blurs the independent review
boundary.

### Code-only Graphify with no architecture manifest

Rejected. It captures runners and tests but loses the research objective,
provenance gates, evidence status, and the Codex/Gemini handoff sequence.

### A hand-maintained graph with no source manifest

Rejected. Edges would have no durable source location and would drift as the
workflow changes.

## Consequences

- The current graph contains the workflow path from prompt creation through
  synthesis, condition-level baseline/evidence-first result nodes, and links to
  the existing logo intervention prompt, manifest, summary, and Gemini report.
- The Graphify run reports zero token cost because no semantic cloud backend was
  called.
- Document-level semantic extraction is intentionally not automatic. Any future
  semantic run requires explicit authorization and a separate provenance audit.
- The manifest and augmenter add a small maintenance burden, but they make the
  model boundary and evidence status inspectable and reproducible.

## Verification

- `python -m unittest discover -s tests -v` passes all 93 tests.
- Graphify diagnostic reports zero missing, dangling, self-loop, duplicate, or
  directed edge-collapse errors.
- A directed path query returns:
  `Gemini prompt handoff -> Codex execution -> Gemini Antigravity review ->
  Artifact and provenance verification -> Evidence-preserving synthesis`.
