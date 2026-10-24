# Shared implementation review — 2026-10-12

Real external CLI delegation, separate from the scientific reports:

- Claude Code reviewed scheduler, timeouts, retry state and result validation. Native CLI returned budget_exhausted after producing a review; its text was recovered from the specific local session transcript without another paid model call. The failure record remains unchanged. Deliverable: collab/setup-review/reports/setup-claude-recovered.md.
- Gemini reviewed CLI adapters/output parsing and proposed a patch for the malformed Codex override keys. Deliverable: collab/setup-review/reports/setup-gemini.md.
- Codex integrated selected fixes and ran regression tests. Internal Codex subagents failed at the provider with unsupported encrypted_content; no successful internal-agent review is claimed.

Applied: preserve partial timeout diagnostics, handle bad result types/config parse errors, make nested provider errors readable, remove literal quotes from Codex override key paths, and guard process-exit races. Model/provider/authentication remain the user's existing configuration; no hardcoded research provider or new credentials.

Not applied: summing concurrent attempt durations against a wall-clock budget (that would change budget semantics); silently re-running a completed task after its output file was deleted (that would spend more without evidence of a transient failure). Report integrity failures stop instead.

Research output is a proposal, not evidence. Peer reports may still overstate causal conclusions or novelty; NEEDS_EVIDENCE is imposed by the controller regardless of prose. The current framework does not execute experiments or write a final manuscript.

## Open-VLM mitigation track - 2026-10-22

The new `open-vlm` track is additive and does not modify the three-CLI
controller. It now has a manifest/domain contract, free-generation and
candidate-scoring metrics, clustered gates, blur/negative-control diagnostics,
an offline run ledger, a checksum-gated Linux launcher, and a lazy Qwen
adapter with fake-backend coverage. The adapter has not been run on a GPU in
this workspace; package pins are availability-checked, while CUDA/attention
compatibility and model/dataset licenses remain target-image preflight gates.

All summaries and fake runs retain `NEEDS_EVIDENCE`. No experiment result or
causal mechanism claim is produced by this implementation slice.

Follow-up review fixes completed on 2026-10-24: manifest rows are converted to
backend tasks with materialized image variants; Step 0 checks blur bias
alignment against a black control before alpha selection; candidate scoring
uses a batched forward-pass contract with EOS-inclusive token traces; phase
results and cost/attempt state are checkpointed; held-out evaluation requires a
passed development gate and clustered family-weighted summaries; and cleanup
stops a failed-upload pod while preserving its volume. GPU execution remains
pending.

The preparation path is now executable from `open_vlm_pilot.py --prepare`:
it fetches the pinned Hugging Face rows, filters explicit prior-answer rows,
downloads images into the run folder, records SHA256/source-family metadata,
and blocks phase execution while license attestations are pending.
