# Local research team — approved execution scope

Supersedes the earlier draft in superpowers/specs. User requested implementation and a first run on 2026-10-11.

PowerShell launcher, Python stdlib scheduler, pypdf input extraction, native Codex/Claude/Gemini CLIs. Local orchestration uses remote model APIs, not offline inference. Inherit native skills and credentials; desktop tools are not guaranteed in CLI. Agents report via stdout; controller owns artifacts.

Default: 2 rounds (independent proposals, cross-review/revisions), then three parallel syntheses = 9 logical tasks. Codex owns the pilot/implementation section, Claude owns causal/evidence review, Gemini owns novelty/generalization. Max 3 concurrent members, 12 total CLI launches, 240 seconds per attempt, 1200 seconds total active time. One initial attempt + max 3 retries per task, backoff 2/4/8 seconds. The global allowance may stop a task before its retry allowance is exhausted. Exhausted member disabled for run; permanent auth/quota/config errors fail immediately. Persist counters and completed reports across resume. No nested delegation, model fallback or auto-new-run. CLI launches are not API request counts; native internal calls may occur. No universal dollar-cap claim; Claude native per-call budget $0.80 is additionally used (in-flight request can overshoot).

Research must propose novel mechanisms, falsifiers and cheap pilots, distinguish prior work from hypotheses, critique the seed paper, prevent source-image leakage across resolution variants, and compare identical subsets and compute budgets. Linear probing suggests mechanisms but is not universal causal proof.

This analysis framework always ends scientifically at NEEDS_EVIDENCE. Prose consensus cannot trigger SOLUTION_READY or paper writing. Experiments and an audited evidence gate are future work. Operational statuses distinguish COMPLETED, DEGRADED, FAILED and BUDGET_EXHAUSTED.

Artifacts: collab/runs/<id>/ with source hashes, input snapshot, manifest, state, per-attempt logs, reports and summary. Default resumes matching latest run; explicit -NewRun grants a fresh budget. -DryRun makes zero model calls; -Status is read-only. OS file lock prevents competing schedulers. Timeout/cancel kills owned process tree.

## Implementation plan
- [ ] Retry/budget/resume and subprocess tests first.
- [ ] Scheduler, adapters, input extraction and bounded prompts.
- [ ] Launcher, native agent instructions and README.
- [ ] Offline tests, dry run, then one real bounded run; inspect successes/failures.

Review focus: Windows quoting, timeout children, persisted attempts after interruption, source/prompt/report integrity, error JSON with zero exit code. Work inline in the existing unborn Git repository; preserve original PDF and unrelated files.
