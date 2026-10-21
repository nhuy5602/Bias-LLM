# Local research team — approved execution scope

Supersedes the earlier draft in superpowers/specs. User requested implementation and a first run on 2026-10-11.

PowerShell launcher, Python stdlib scheduler, pypdf input extraction, native Codex/Claude/Gemini CLIs. Local orchestration uses remote model APIs, not offline inference. Inherit native skills and credentials; desktop tools are not guaranteed in CLI. Agents report via stdout; controller owns artifacts.

Default: 2 rounds (independent proposals, cross-review/revisions), then three parallel syntheses = 9 logical tasks. Codex owns the pilot/implementation section, Claude owns causal/evidence review, Gemini owns novelty/generalization. Roles can rotate via `--rotate-roles` (`run_index % 3`). Max 3 concurrent members, 12 total CLI launches, 4 launches per member, 240 seconds per attempt, 1200 seconds total active time. One initial attempt + max 3 retries per task, backoff 2/4/8 seconds. Call/time quotas are enforced per agent. Claude has a $2 native per-run cap; each request gets the remaining amount, with provider-side overshoot possible. Gemini has no USD cap by user request but remains bounded by calls, timeout, and runtime. Codex cost is not reported by its CLI. Exhausted member disabled for run; permanent auth/quota/config errors fail immediately. Section failover (`--section-failover`) can reassign disabled members' sections to the available member with the lowest load without switching models or endpoints. Persist counters, member stats and completed reports across resume. No nested delegation or auto-new-run. Peer excerpts are bounded to 2,000 chars and evidence gate receives sliced context (methods & result tables only).

Optional manual duo mode: `-Duo -NewRun` runs only Codex and Gemini, assigning `pilot` and `evidence_gate` with mutual circular audit. Claude is not called and is absent from duo tasks, reports and synthesis. Use `-Duo -ContinueLatest` for one continuation round from a verified duo ledger. Duo mode requires an explicit fresh run, continuation, or duo-ledger extension and cannot auto-take over a three-agent ledger or use section failover.

Full-mode circular review & consensus: Authors are strictly prohibited from self-approving their own proposals. In Round 2 and synthesis, audits follow a circular dependency: Claude audits Codex's pilot, Codex audits Gemini's novelty, and Gemini audits Claude's evidence gate. Duo mode uses the same no-self-approval rule with mutual Codex/Gemini audit. Adoption of any proposal or mechanism into actionable recommendations requires consensus from at least 2 agents (minority dissents are preserved).

Research must propose novel mechanisms, falsifiers and cheap pilots, distinguish prior work from hypotheses, critique the seed paper, prevent source-image leakage across resolution variants, and compare identical subsets and compute budgets. Linear probing suggests mechanisms but is not universal causal proof.

This analysis framework always ends scientifically at NEEDS_EVIDENCE. Prose consensus cannot trigger SOLUTION_READY or paper writing. Experiments and an audited evidence gate are future work. Operational statuses distinguish COMPLETED, DEGRADED, FAILED and BUDGET_EXHAUSTED.
An operational round is COMPLETED only when all active synthesis roles finish; omitted or failed role synthesis makes it DEGRADED even when some synthesis reports are available.

Artifacts: collab/runs/<id>/ with source hashes, input snapshot, manifest, state, per-attempt logs, reports and SUMMARY.md (including a Workload Distribution table with tasks, attempts, runtime seconds, USD cost, and load skew warnings when an agent accounts for >50% of tasks or runtime). Default resumes matching latest run; explicit -NewRun grants a fresh budget. -DryRun makes zero model calls; -Status is read-only. OS file lock prevents competing schedulers. Timeout/cancel kills owned process tree.

The -ContinueLatest option starts one cross-review/revision round from the latest complete, hash-verified proposal round, then runs circular audits/syntheses for all active roles. It preserves source and role ownership and creates a new budget ledger. It refuses incomplete or tampered prior proposals; finish or resume those before continuing.

The -ExtendBudget option resumes an incomplete latest run without replaying completed tasks. It permits keeping or increasing elapsed-time/per-member-call ceilings, never decreasing them; it does not reset calls, attempts, costs, sources, roles, or model settings. Actual increases are recorded in the manifest.

The -RepairGemini option is only valid with -ExtendBudget. It reuses the persisted prompt for an exhausted Gemini proposal task, starts a separately ledgered repair task with a 600-second default timeout, preserves the four original timeout attempts, and generates reconciliation synthesis using the recovered report. The repair task has the normal maximum of three retries and remains bounded by the run and Gemini call/time quotas; Gemini has no USD cap.

The -RepairGeminiSynthesis option is only valid with -ExtendBudget and -ResumeRunId. It starts a separately ledgered Gemini synthesis recovery after the original synthesis task exhausts four attempts, preserves its timeout history, and then reruns synthesis from the saved peer reports. The recovery task allows at most three retries.

The -UnlimitedBudget option is valid with -ExtendBudget or -ContinueLatest. It removes global and per-member launch, elapsed-time, and USD caps for all members, including Claude's native USD cap. The per-request timeout and maximum of three retries per task remain. Quota fields are recorded as null in the manifest; actual calls, elapsed time and any CLI-reported costs continue to be recorded in the ledger.

The optional -ResumeRunId with -ExtendBudget selects an older run ledger for recovery without changing latest.json. The ID must name an existing run; source and saved-report integrity checks still apply.

The root run-series.ps1 command continues the configured series sequentially and updates its ledger after each run. It stops when a run is incomplete; a degraded Gemini synthesis may be recovered with -RepairGeminiSynthesis before continuing.

The separate `run-semantic-label-pilot.ps1` runs an exploratory prompt intervention on 20 counterfactual animal images (10 birds and 10 mammals), selected deterministically from the public 384px dataset. Each image receives generic and species-named prompts twice. At most 80 model launches total are permitted, including up to three retries per task; each launch is checkpointed with prompt/output logs. The current provider is Claude Code CLI using the machine-configured `claude-fable-5-1` model and proxy host `gpt.teamsoclo.site`; only the Read tool is enabled and Claude's native USD cap is $2 per pilot run, though proxy/provider overshoot is possible. This pilot recorded $2.3534 before stopping. Previous Gemini launches remain in the same 80-launch ledger. Provider errors make the run `DEGRADED`; `-RetryUnavailable` is available after quota recovery. It does not alter the discussion-series ledger, include an original-image baseline, or establish a causal mechanism. Scientific status remains `NEEDS_EVIDENCE`.

The `scripts/logo_duo_pilot.py` executor runs a smaller exploratory replication on two official VLMBias Audi families, pairing original and counterfactual 384px images and sending the same counting prompt to Codex and Gemini. It records source metadata, image hashes, prompts, attempts, stderr/stdout and parsed counts in `collab/experiment/logo-duo-<id>/`. The first run used 8 launches: Codex completed 4/4 and Gemini timed out 4/4, so operational status is `DEGRADED`; this is not a causal pilot because texture/edge-matched shams, hidden-state intervention and open-VLM comparison are absent.

## Implementation plan
- [x] Retry/budget/resume and subprocess tests first.
- [x] Scheduler, adapters, input extraction and bounded prompts.
- [x] Launcher, native agent instructions and README.
- [x] Circular audit, role rotation, per-agent quotas, section failover, and workload summary table.
- [x] Offline tests, dry run, then one real bounded run; inspect successes/failures.

Review focus: Windows quoting, timeout children, persisted attempts after interruption, source/prompt/report integrity, error JSON with zero exit code. Work inline in the existing unborn Git repository; preserve original PDF and unrelated files.
