# Shared implementation review — 2026-10-12

Real external CLI delegation, separate from the scientific reports:

- Claude Code reviewed scheduler, timeouts, retry state and result validation. Native CLI returned budget_exhausted after producing a review; its text was recovered from the specific local session transcript without another paid model call. The failure record remains unchanged. Deliverable: collab/setup-review/reports/setup-claude-recovered.md.
- Gemini reviewed CLI adapters/output parsing and proposed a patch for the malformed Codex override keys. Deliverable: collab/setup-review/reports/setup-gemini.md.
- Codex integrated selected fixes and ran regression tests. Internal Codex subagents failed at the provider with unsupported encrypted_content; no successful internal-agent review is claimed.

Applied: preserve partial timeout diagnostics, handle bad result types/config parse errors, make nested provider errors readable, remove literal quotes from Codex override key paths, and guard process-exit races. Model/provider/authentication remain the user's existing configuration; no hardcoded research provider or new credentials.

Not applied: summing concurrent attempt durations against a wall-clock budget (that would change budget semantics); silently re-running a completed task after its output file was deleted (that would spend more without evidence of a transient failure). Report integrity failures stop instead.

Research output is a proposal, not evidence. Peer reports may still overstate causal conclusions or novelty; NEEDS_EVIDENCE is imposed by the controller regardless of prose. The current framework does not execute experiments or write a final manuscript.
