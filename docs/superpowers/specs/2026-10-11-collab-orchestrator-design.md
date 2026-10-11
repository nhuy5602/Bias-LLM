# Collab Orchestrator — Design Spec

Date: 2026-10-11
Status: historical draft — superseded by ../../COLLAB.md and the user's later instructions
Command: `.\run-collab.ps1` from project root

## 1. Intent
- Run Codex, Claude, and Gemini locally on the same folder `Problem/`.
- Round 1 is read-only analysis, no edits to source PDF or code.
- Produce a solution candidate with evidence labels, then a paper skeleton only when ready.

## 2. Non-goals for this phase
- No `--implement`, no `--paper` full writing, no auto-editing source files.
- No model calls during scaffolding; `--dry-run` must work offline.

## 3. Architecture
- Single PowerShell orchestrator, local only.
- Calls three CLIs: `codex exec`, `claude -p`, `gemini -p`.
- Shared workdir is the project root; agents read `Problem/`, write only under `collab/`.

## 4. Artifacts under `collab/`
- `inputs/` — manifest, SHA256 of PDF, env metadata, CLI versions.
- `round-01/`, `round-02/`, ... — independent analysis, cross-critique, rebuttal.
- `synthesis/` — claims labeled `evidence` / `hypothesis` / `open`, plus verification criteria.
- `paper/` — created only on `SOLUTION_READY`.
- `logs/` — stdout, stderr, exit code, duration per agent call.

## 5. Flow
1. Inventory `Problem/` and extract PDF text to `collab/inputs/`.
2. Independent analysis by each selected agent.
3. Cross-critique: each agent critiques the other two.
4. Codex synthesizes a solution candidate with claim labels.
5. Repeat until no severe critique remains or `--max-rounds` is reached. Agreement is not evidence.
6. Gate: `NEEDS_EVIDENCE` stops before paper; `SOLUTION_READY` opens `paper/` skeleton.

## 6. Flags and safety
- Flags: `--dry-run`, `--max-rounds N`, `--agent codex|claude|gemini|all`, timeout, retry.
- Preflight checks for required API keys without printing values.
- Default `analysis-only`; resume from existing `collab/` artifacts.
- Never modify `Problem/` originals.

## 7. Verification
- Dry run creates folders and manifest with no model calls.
- Each agent log records exit code and duration.
- Synthesis gate file explicitly states `NEEDS_EVIDENCE` or `SOLUTION_READY` with reasons.

## 8. Open item
- Definition of "triệt để" is assumed as: labeled claims plus verification criteria, not running code. Correct this assumption before the implement phase.

## 9. Self-review 2026-10-11
- Placeholder scan: no TBD or TODO markers found.
- Consistency: analysis-only default matches no-edit rule for Problem/.
- Scope: this spec covers orchestrator only, not implement or paper phases.
- Ambiguity fixed: resume means reuse existing collab/ outputs, never rewrite inputs.
- Error handling: per-agent timeout, one retry, logs keep exit code and duration.
