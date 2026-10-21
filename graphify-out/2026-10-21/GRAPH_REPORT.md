# Graph Report - Bias LLM  (2026-10-21)

## Corpus Check
- 31 files · ~40,632 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 3 file(s) not represented in the graph (top: .log 2, (none) 1)

## Summary
- 404 nodes · 912 edges · 31 communities (16 shown, 15 thin omitted)
- Extraction: 86% EXTRACTED · 14% INFERRED · 0% AMBIGUOUS · INFERRED: 125 edges (avg confidence: 0.85)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `77de20ae`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- invoke
- logo_duo_pilot.py
- semantic_label_pilot.py
- Tasks
- SemanticLabelPilotTests
- ADR-001: Keep Graphify and Gemini Handoffs Deterministic
- SchedulerTests
- hashlib
- Collab Orchestrator — Design Spec
- .test_duo_fresh_run_uses_only_codex_and_gemini
- logo_paired_sham_pilot.py
- collab.py
- Runner
- check_project
- IntegrationTests
- Cross-Cutting Principles
- LogoRobustnessPilotTests
- Task List
- Nhóm nghiên cứu Bias VLM — Codex, Claude Code, Gemini
- digest
- Local research team — approved execution scope
- Mercedes-Benz Generalization Pilot
- AGENTS.md
- CLAUDE.md
- IMPLEMENTATION-REVIEW.md
- GEMINI.md
- todo-robustness.md

## God Nodes (most connected - your core abstractions)
1. `save()` - 35 edges
2. `IntegrationTests` - 32 edges
3. `SemanticLabelPilotTests` - 32 edges
4. `stable_digest()` - 27 edges
5. `main()` - 24 edges
6. `prepare_run()` - 24 edges
7. `Runner` - 22 edges
8. `SchedulerTests` - 17 edges
9. `invoke_cli()` - 15 edges
10. `run_process()` - 14 edges

## Surprising Connections (you probably didn't know these)
- `SchedulerTests` --uses--> `Runner`  [INFERRED]
  tests/test_collab.py → scripts/collab.py
- `IntegrationTests` --uses--> `Runner`  [INFERRED]
  tests/test_collab.py → scripts/collab.py
- `run_tasks()` --calls--> `save()`  [EXTRACTED]
  scripts/logo_duo_pilot.py → scripts/collab.py
- `invoke()` --calls--> `executable()`  [EXTRACTED]
  scripts/logo_duo_pilot.py → scripts/collab.py
- `invoke()` --calls--> `save()`  [EXTRACTED]
  scripts/logo_duo_pilot.py → scripts/collab.py

## Import Cycles
- None detected.

## Communities (31 total, 15 thin omitted)

### Community 1 - "logo_duo_pilot.py"
Cohesion: 0.08
Nodes (20): Enforced gates, Exceptions, Floor, Required evidence before a stronger scientific claim, Research Constraints, build_gemini_prompt(), build_tasks(), discover_rows() (+12 more)

### Community 2 - "semantic_label_pilot.py"
Cohesion: 0.11
Nodes (46): executable(), save(), main(), summarize(), build_blur_gemini_prompt(), build_blur_tasks(), build_gemini_prompt(), build_matched_blur_gemini_prompt() (+38 more)

### Community 3 - "Tasks"
Cohesion: 0.17
Nodes (11): Checkpoint: Prepared protocol, Gaussian-blur follow-up, Implementation Plan: Visual Signal Robustness Pilot, Matched-resolution Gaussian-blur correction, Matched-resolution sigma-zero baseline, Overview, Phase 1: Preparation, Phase 2: Execution (+3 more)

### Community 4 - "SemanticLabelPilotTests"
Cohesion: 0.05
Nodes (20): usd_floor_cents(), api_failure_summary(), bind_task_image(), build_api_request(), claude_budget_allows_call(), claude_command(), format_response(), generate_api() (+12 more)

### Community 5 - "ADR-001: Keep Graphify and Gemini Handoffs Deterministic"
Cohesion: 0.17
Nodes (11): A hand-maintained graph with no source manifest, ADR-001: Keep Graphify and Gemini Handoffs Deterministic, Alternatives considered, Automatic Gemini semantic extraction, Code-only Graphify with no architecture manifest, Consequences, Context, Date (+3 more)

### Community 6 - "SchedulerTests"
Cohesion: 0.12
Nodes (5): kill_tree(), parse_response(), run_process(), invoke(), SchedulerTests

### Community 7 - "hashlib"
Cohesion: 0.18
Nodes (8): alias_edges(), artifact_edges(), artifact_nodes(), edge(), main(), node(), source_state(), update_report_provenance()

### Community 8 - "Collab Orchestrator — Design Spec"
Cohesion: 0.18
Nodes (10): 1. Intent, 2. Non-goals for this phase, 3. Architecture, 4. Artifacts under `collab/`, 5. Flow, 6. Flags and safety, 7. Verification, 8. Open item (+2 more)

### Community 10 - "logo_paired_sham_pilot.py"
Cohesion: 0.15
Nodes (8): build_gemini_prompt(), build_tasks(), erase_region(), prepare_run(), _prompt_for(), relative_bbox(), select_original_rows(), LogoPairedShamPilotTests

### Community 14 - "collab.py"
Cohesion: 0.13
Nodes (11): build_workload_summary(), check_consensus(), get_duo_owners(), get_owners(), get_role_context(), main(), make_prompt(), prepare_inputs() (+3 more)

### Community 15 - "Runner"
Cohesion: 0.38
Nodes (4): Runner, main(), run_pilot(), write_answers()

### Community 16 - "check_project"
Cohesion: 0.20
Nodes (6): check_project(), _graph_issues(), main(), _secret_issues(), _status_issues(), ResearchConstraintsTests

### Community 17 - "IntegrationTests"
Cohesion: 0.26
Nodes (3): invoke_cli(), validate_budget_extension(), IntegrationTests

### Community 19 - "LogoRobustnessPilotTests"
Cohesion: 0.21
Nodes (3): derive_matched_blur_variant(), normalize_source_variant(), LogoRobustnessPilotTests

### Community 20 - "Task List"
Cohesion: 0.17
Nodes (11): Checkpoint: Evidence, Checkpoint: Protocol, Implementation Plan: Mercedes-Benz Generalization Pilot, Open Questions, Overview, Phase 1: Runner and protocol, Phase 2: Bounded execution, Phase 3: Synthesis and map (+3 more)

### Community 21 - "Nhóm nghiên cứu Bias VLM — Codex, Claude Code, Gemini"
Cohesion: 0.33
Nodes (5): Giới hạn mặc định, Kết quả, Lệnh khác, Nhóm nghiên cứu Bias VLM — Codex, Claude Code, Gemini, Skills và plugins

## Knowledge Gaps
- **50 isolated node(s):** `Research project: visual-prior bias`, `Project research instructions`, `Floor`, `Required evidence before a stronger scientific claim`, `Exceptions` (+45 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 118 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **15 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `invoke_cli()` connect `IntegrationTests` to `semantic_label_pilot.py`, `SemanticLabelPilotTests`, `SchedulerTests`, `collab.py`?**
  _High betweenness centrality (0.001) - this node is a cross-community bridge._
- **What connects `Research project: visual-prior bias`, `Project research instructions`, `Floor` to the rest of the system?**
  _50 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `invoke` be split into smaller, more focused modules?**
  _Cohesion score 0.125 - nodes in this community are weakly interconnected._
- **Why does `run_process()` connect `SchedulerTests` to `logo_duo_pilot.py`?**
  _High betweenness centrality (0.001) - this node is a cross-community bridge._
- **Should `logo_duo_pilot.py` be split into smaller, more focused modules?**
  _Cohesion score 0.08408408408408409 - nodes in this community are weakly interconnected._
- **Should `semantic_label_pilot.py` be split into smaller, more focused modules?**
  _Cohesion score 0.11313131313131314 - nodes in this community are weakly interconnected._
- **Should `SemanticLabelPilotTests` be split into smaller, more focused modules?**
  _Cohesion score 0.05411764705882353 - nodes in this community are weakly interconnected._
## Source-state provenance
- Architecture manifest: `collab/graphify/ARCHITECTURE.md`
- Git HEAD: `77de20aecc72acfbcc128d26d554b9e3dfc928b8`
- Worktree: **dirty** (161 changed/untracked paths)
- Worktree status SHA256: `a0ba4dd41feb59e11ffef42b00447da28041bb5cebaaceb19832b26a2bb0c907`
- Architecture manifest SHA256: `b0f319b17a4cf0ef0a9144cc5cce5287e5033d288fc5d70b4b3446b2e6cd7169`
- This report is not a clean-commit snapshot when the worktree is dirty.
