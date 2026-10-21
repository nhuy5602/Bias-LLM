# Research Constraints

Last reviewed: 2026-10-19

This project studies visual-prior bias in vision-language models. The quality bar
below applies to experiment runners, Gemini handoffs, reports, Graphify artifacts,
and any future paper-facing text.

## Floor

- Keep the scientific status at `NEEDS_EVIDENCE` until an experiment and an
  independent evidence audit support a stronger claim. Prose consensus is not a
  pass condition.
- Do not invent data, citations, novelty, model behavior, or causal mechanisms.
  Mark claims as fact, inference, hypothesis, or unresolved issue.
- Do not call a behavioral discrepancy, linear probe, or model agreement a causal
  mechanism without an appropriate intervention and control.
- Do not silently change a model, provider, endpoint, dataset revision, prompt,
  subset, compute budget, or retry policy. Record changes in the run ledger.
- The Codex-owned research runner must not call Gemini CLI or Gemini API. Gemini
  runs only from the user-facing Antigravity prompt and writes a separate report.
- Keep source-image families separate from derived resolutions and edits. Repeated
  readings are not independent samples. Preserve negative findings and dissent.
- Do not commit or print API keys, access tokens, or secret values.
- Do not skip, delete, or weaken tests to make a run pass.

## Enforced gates

| Dimension | Rule | Checked by | Stage |
|---|---|---|---|
| Tests | 0 failing tests | `python -m unittest discover -s tests -v` | every code change |
| Graph integrity | 0 missing, dangling, self-loop, or collapsed edges | `graphify diagnose multigraph --graph graphify-out/graph.json --directed` | after graph rebuild |
| Scientific status | Every experiment summary retains `NEEDS_EVIDENCE` | `rg -n "NEEDS_EVIDENCE" collab -g SUMMARY.md` | every synthesis |
| Workflow boundary | Gemini invocation remains rejected by the Codex-only runner | `python -m unittest discover -s tests -v` and `test_invoke_rejects_gemini_in_codex_only_runner` | every runner change |
| Artifact provenance | Hashes, prompts, conditions, repeats, attempts, and unresolved gaps are retained | `GEMINI-PROMPT.md`, `GEMINI-REPORT.md`, `SUMMARY.md`, and run ledger review | every experiment handoff |
| Workspace hygiene | No whitespace errors or accidental secret material in the diff | `git diff --check` plus manual secret scan | before handoff |
| Epistemic topology | The objective reaches `NEEDS_EVIDENCE` through the audit bridge; no direct objective-to-causal edge is allowed | `python scripts/check_research_constraints.py .` | every graph rebuild |

The dependency-free combined gate is:

```powershell
python scripts/check_research_constraints.py .
```

It checks the graph structure, summary status, and secret-like values without
calling a model or changing files.

## Required evidence before a stronger scientific claim

- Pixel-paired source/counterfactual families or an explicit reason they are
  unavailable.
- Background, texture, edge, or other sham controls where a background mechanism
  is proposed.
- Matched compute and the same subsets across compared conditions.
- Independent evidence audit of raw outputs and hashes.
- A preregistered falsifier or pass/fail rule for any proposed mechanism.
- Generalization beyond the current two-brand logo pilot, or an explicit scope
  limit that says the result is only a two-brand pilot.

## Exceptions

Exceptions must name the affected artifact, reason, owner, and expiry date. An
exception does not change the scientific status or permit silent provider/model
changes.
