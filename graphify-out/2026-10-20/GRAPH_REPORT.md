# Graph Report - Bias LLM  (2026-10-20)

## Corpus Check
- 28 files · ~36,285 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 2 file(s) not represented in the graph (top: (none) 1, .log 1)

## Summary
- 366 nodes · 808 edges · 24 communities (12 shown, 12 thin omitted)
- Extraction: 86% EXTRACTED · 14% INFERRED · 0% AMBIGUOUS · INFERRED: 113 edges (avg confidence: 0.85)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `77de20ae`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- collab.py
- logo_duo_pilot.py
- LogoRobustnessPilotTests
- SemanticLabelPilotTests
- semantic_label_pilot.py
- ADR-001: Keep Graphify and Gemini Handoffs Deterministic
- SchedulerTests
- IntegrationTests
- Collab Orchestrator — Design Spec
- Tasks
- Task List
- check_project
- AGENTS.md
- hashlib
- Nhóm nghiên cứu Bias VLM — Codex, Claude Code, Gemini
- Local research team — approved execution scope
- Mercedes-Benz Generalization Pilot
- CLAUDE.md
- IMPLEMENTATION-REVIEW.md
- GEMINI.md
- todo-robustness.md

## God Nodes (most connected - your core abstractions)
1. `save()` - 32 edges
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
- `invoke()` --calls--> `run_process()`  [EXTRACTED]
  scripts/logo_duo_pilot.py → scripts/collab.py
- `main()` --calls--> `save()`  [EXTRACTED]
  scripts/logo_duo_pilot.py → scripts/collab.py

## Import Cycles
- None detected.

## Communities (24 total, 12 thin omitted)

### Community 0 - "collab.py"
Cohesion: 0.08
Nodes (21): build_workload_summary(), check_consensus(), digest(), get_duo_owners(), get_owners(), get_role_context(), kill_tree(), main() (+13 more)

### Community 1 - "logo_duo_pilot.py"
Cohesion: 0.08
Nodes (20): Enforced gates, Exceptions, Floor, Required evidence before a stronger scientific claim, Research Constraints, build_gemini_prompt(), build_tasks(), discover_rows() (+12 more)

### Community 3 - "SemanticLabelPilotTests"
Cohesion: 0.07
Nodes (16): api_failure_summary(), bind_task_image(), build_api_request(), claude_budget_allows_call(), format_response(), generate_api(), make_custom_command(), make_prompt() (+8 more)

### Community 4 - "semantic_label_pilot.py"
Cohesion: 0.10
Nodes (42): executable(), save(), invoke(), build_blur_gemini_prompt(), build_blur_tasks(), build_gemini_prompt(), build_tasks(), derive_blur_variant() (+34 more)

### Community 5 - "ADR-001: Keep Graphify and Gemini Handoffs Deterministic"
Cohesion: 0.17
Nodes (11): A hand-maintained graph with no source manifest, ADR-001: Keep Graphify and Gemini Handoffs Deterministic, Alternatives considered, Automatic Gemini semantic extraction, Code-only Graphify with no architecture manifest, Consequences, Context, Date (+3 more)

### Community 7 - "IntegrationTests"
Cohesion: 0.06
Nodes (7): invoke_cli(), redact(), validate_budget_extension(), IntegrationTests, fake_invoke(), fake_prepare(), invoke()

### Community 8 - "Collab Orchestrator — Design Spec"
Cohesion: 0.18
Nodes (10): 1. Intent, 2. Non-goals for this phase, 3. Architecture, 4. Artifacts under `collab/`, 5. Flow, 6. Flags and safety, 7. Verification, 8. Open item (+2 more)

### Community 9 - "Tasks"
Cohesion: 0.20
Nodes (9): Checkpoint: Prepared protocol, Gaussian-blur follow-up, Implementation Plan: Visual Signal Robustness Pilot, Overview, Phase 1: Preparation, Phase 2: Execution, Phase 3: Audit and synthesis, Risks (+1 more)

### Community 10 - "Task List"
Cohesion: 0.18
Nodes (10): Checkpoint: Evidence, Checkpoint: Protocol, Implementation Plan: Mercedes-Benz Generalization Pilot, Open Questions, Overview, Phase 1: Runner and protocol, Phase 2: Bounded execution, Phase 3: Synthesis and map (+2 more)

### Community 14 - "check_project"
Cohesion: 0.20
Nodes (6): check_project(), _graph_issues(), main(), _secret_issues(), _status_issues(), ResearchConstraintsTests

### Community 16 - "hashlib"
Cohesion: 0.18
Nodes (8): alias_edges(), artifact_edges(), artifact_nodes(), edge(), main(), node(), source_state(), update_report_provenance()

### Community 17 - "Nhóm nghiên cứu Bias VLM — Codex, Claude Code, Gemini"
Cohesion: 0.33
Nodes (5): Giới hạn mặc định, Kết quả, Lệnh khác, Nhóm nghiên cứu Bias VLM — Codex, Claude Code, Gemini, Skills và plugins

## Knowledge Gaps
- **47 isolated node(s):** `Research project: visual-prior bias`, `Project research instructions`, `Floor`, `Required evidence before a stronger scientific claim`, `Exceptions` (+42 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 111 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **12 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `invoke_cli()` connect `IntegrationTests` to `collab.py`, `semantic_label_pilot.py`, `SchedulerTests`?**
  _High betweenness centrality (0.001) - this node is a cross-community bridge._
- **What connects `Research project: visual-prior bias`, `Project research instructions`, `Floor` to the rest of the system?**
  _47 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `collab.py` be split into smaller, more focused modules?**
  _Cohesion score 0.08405797101449275 - nodes in this community are weakly interconnected._
- **Why does `run_process()` connect `collab.py` to `logo_duo_pilot.py`?**
  _High betweenness centrality (0.001) - this node is a cross-community bridge._
- **Should `logo_duo_pilot.py` be split into smaller, more focused modules?**
  _Cohesion score 0.08250355618776671 - nodes in this community are weakly interconnected._
- **Should `SemanticLabelPilotTests` be split into smaller, more focused modules?**
  _Cohesion score 0.07058823529411765 - nodes in this community are weakly interconnected._
- **Should `semantic_label_pilot.py` be split into smaller, more focused modules?**
  _Cohesion score 0.0967741935483871 - nodes in this community are weakly interconnected._
## Source-state provenance
- Architecture manifest: `collab/graphify/ARCHITECTURE.md`
- Git HEAD: `77de20aecc72acfbcc128d26d554b9e3dfc928b8`
- Worktree: **dirty** (130 changed/untracked paths)
- Worktree status SHA256: `1e87ea72f94afabd60b514e44151498d45e2c43d495bce092ee7a2261956e729`
- Architecture manifest SHA256: `7de70055a24a2ecc4dd809c9c002ce832df6537296af945922e7e3822173474e`
- This report is not a clean-commit snapshot when the worktree is dirty.
