# Graph Report - Bias LLM  (2026-10-19)

## Corpus Check
- 28 files · ~35,118 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 2 file(s) not represented in the graph (top: (none) 1, .log 1)

## Summary
- 358 nodes · 784 edges · 23 communities (13 shown, 10 thin omitted)
- Extraction: 86% EXTRACTED · 14% INFERRED · 0% AMBIGUOUS · INFERRED: 110 edges (avg confidence: 0.85)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `77de20ae`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- collab.py
- logo_duo_pilot.py
- logo_robustness_pilot.py
- hashlib
- semantic_label_pilot.py
- ADR-001: Keep Graphify and Gemini Handoffs Deterministic
- SchedulerTests
- IntegrationTests
- Collab Orchestrator — Design Spec
- Nhóm nghiên cứu Bias VLM — Codex, Claude Code, Gemini
- Task List
- check_project
- AGENTS.md
- Tasks
- Local research team — approved execution scope
- Mercedes-Benz Generalization Pilot
- CLAUDE.md
- IMPLEMENTATION-REVIEW.md
- GEMINI.md
- todo-robustness.md

## God Nodes (most connected - your core abstractions)
1. `IntegrationTests` - 32 edges
2. `SemanticLabelPilotTests` - 32 edges
3. `save()` - 31 edges
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
- `invoke()` --calls--> `save()`  [EXTRACTED]
  scripts/logo_duo_pilot.py → scripts/collab.py
- `main()` --calls--> `save()`  [EXTRACTED]
  scripts/logo_duo_pilot.py → scripts/collab.py

## Import Cycles
- None detected.

## Communities (23 total, 10 thin omitted)

### Community 0 - "collab.py"
Cohesion: 0.10
Nodes (16): build_workload_summary(), check_consensus(), digest(), get_duo_owners(), get_owners(), get_role_context(), kill_tree(), main() (+8 more)

### Community 1 - "logo_duo_pilot.py"
Cohesion: 0.08
Nodes (20): Enforced gates, Exceptions, Floor, Required evidence before a stronger scientific claim, Research Constraints, build_gemini_prompt(), build_tasks(), discover_rows() (+12 more)

### Community 2 - "logo_robustness_pilot.py"
Cohesion: 0.15
Nodes (16): executable(), run_process(), invoke(), build_gemini_prompt(), build_tasks(), derive_variant(), invoke(), main() (+8 more)

### Community 3 - "hashlib"
Cohesion: 0.18
Nodes (8): alias_edges(), artifact_edges(), artifact_nodes(), edge(), main(), node(), source_state(), update_report_provenance()

### Community 4 - "semantic_label_pilot.py"
Cohesion: 0.07
Nodes (41): save(), api_failure_summary(), bind_task_image(), build_api_request(), build_tasks(), claude_budget_allows_call(), claude_configuration(), dataset_candidates() (+33 more)

### Community 5 - "ADR-001: Keep Graphify and Gemini Handoffs Deterministic"
Cohesion: 0.17
Nodes (11): A hand-maintained graph with no source manifest, ADR-001: Keep Graphify and Gemini Handoffs Deterministic, Alternatives considered, Automatic Gemini semantic extraction, Code-only Graphify with no architecture manifest, Consequences, Context, Date (+3 more)

### Community 6 - "SchedulerTests"
Cohesion: 0.12
Nodes (5): parse_response(), usd_floor_cents(), claude_command(), invoke_claude(), SchedulerTests

### Community 7 - "IntegrationTests"
Cohesion: 0.06
Nodes (6): invoke_cli(), validate_budget_extension(), IntegrationTests, fake_invoke(), fake_prepare(), invoke()

### Community 8 - "Collab Orchestrator — Design Spec"
Cohesion: 0.18
Nodes (10): 1. Intent, 2. Non-goals for this phase, 3. Architecture, 4. Artifacts under `collab/`, 5. Flow, 6. Flags and safety, 7. Verification, 8. Open item (+2 more)

### Community 9 - "Nhóm nghiên cứu Bias VLM — Codex, Claude Code, Gemini"
Cohesion: 0.33
Nodes (5): Giới hạn mặc định, Kết quả, Lệnh khác, Nhóm nghiên cứu Bias VLM — Codex, Claude Code, Gemini, Skills và plugins

### Community 10 - "Task List"
Cohesion: 0.18
Nodes (10): Checkpoint: Evidence, Checkpoint: Protocol, Implementation Plan: Mercedes-Benz Generalization Pilot, Open Questions, Overview, Phase 1: Runner and protocol, Phase 2: Bounded execution, Phase 3: Synthesis and map (+2 more)

### Community 14 - "check_project"
Cohesion: 0.20
Nodes (6): check_project(), _graph_issues(), main(), _secret_issues(), _status_issues(), ResearchConstraintsTests

### Community 16 - "Tasks"
Cohesion: 0.22
Nodes (8): Checkpoint: Prepared protocol, Implementation Plan: Visual Signal Robustness Pilot, Overview, Phase 1: Preparation, Phase 2: Execution, Phase 3: Audit and synthesis, Risks, Tasks

## Knowledge Gaps
- **46 isolated node(s):** `Research project: visual-prior bias`, `Project research instructions`, `Floor`, `Required evidence before a stronger scientific claim`, `Exceptions` (+41 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 110 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **10 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `invoke_cli()` connect `IntegrationTests` to `collab.py`, `logo_robustness_pilot.py`, `semantic_label_pilot.py`, `SchedulerTests`?**
  _High betweenness centrality (0.001) - this node is a cross-community bridge._
- **What connects `Research project: visual-prior bias`, `Project research instructions`, `Floor` to the rest of the system?**
  _46 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `collab.py` be split into smaller, more focused modules?**
  _Cohesion score 0.09634146341463415 - nodes in this community are weakly interconnected._
- **Why does `run_process()` connect `logo_robustness_pilot.py` to `collab.py`?**
  _High betweenness centrality (0.001) - this node is a cross-community bridge._
- **Should `logo_duo_pilot.py` be split into smaller, more focused modules?**
  _Cohesion score 0.08258258258258258 - nodes in this community are weakly interconnected._
- **Should `logo_robustness_pilot.py` be split into smaller, more focused modules?**
  _Cohesion score 0.14666666666666667 - nodes in this community are weakly interconnected._
- **Should `semantic_label_pilot.py` be split into smaller, more focused modules?**
  _Cohesion score 0.07245386192754613 - nodes in this community are weakly interconnected._
## Source-state provenance
- Architecture manifest: `collab/graphify/ARCHITECTURE.md`
- Git HEAD: `77de20aecc72acfbcc128d26d554b9e3dfc928b8`
- Worktree: **dirty** (117 changed/untracked paths)
- Worktree status SHA256: `28c8914b82ce69d3c68f44b0253b67b2a0c8c09286536b200c7f31653b1a728a`
- Architecture manifest SHA256: `031b1873dd7fded81c5ad1cb25b753c3c41217a2dc1114f513564f9d51b63b67`
- This report is not a clean-commit snapshot when the worktree is dirty.
