# Graph Report - Bias LLM  (2026-10-21)

## Corpus Check
- cluster-only mode — file stats not available

## Summary
- 483 nodes · 1119 edges · 22 communities (15 shown, 7 thin omitted)
- Extraction: 70% EXTRACTED · 30% INFERRED · 0% AMBIGUOUS · INFERRED: 332 edges (avg confidence: 0.83)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `77de20ae`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- invoke
- logo_duo_pilot.py
- semantic_label_pilot.py
- Artifact and provenance verification
- usd_floor_cents
- ADR-001: Keep Graphify and Gemini Handoffs Deterministic
- SchedulerTests
- augment_graphify_architecture.py
- Codex execution
- .test_duo_fresh_run_uses_only_codex_and_gemini
- logo_paired_sham_pilot.py
- main
- Architecture manifest
- check_project
- IntegrationTests
- Cross-Cutting Principles
- synthesis_prompt
- collab.py
- Epistemic discipline and auditability

## God Nodes (most connected - your core abstractions)
1. `Artifact and provenance verification` - 47 edges
2. `Architecture manifest` - 43 edges
3. `save()` - 35 edges
4. `IntegrationTests` - 32 edges
5. `SemanticLabelPilotTests` - 32 edges
6. `stable_digest()` - 27 edges
7. `main()` - 24 edges
8. `prepare_run()` - 24 edges
9. `Runner` - 22 edges
10. `Epistemic discipline and auditability` - 20 edges

## Surprising Connections (you probably didn't know these)
- `Artifact and provenance verification` --references--> `Matched-resolution Gaussian-blur manifest`  [INFERRED]
  collab/graphify/ARCHITECTURE.md → collab/experiment/logo-robustness-20261021-181603-868764/manifest.json
- `Artifact and provenance verification` --references--> `Matched-resolution Gaussian-blur v2 manifest`  [INFERRED]
  collab/graphify/ARCHITECTURE.md → collab/experiment/logo-robustness-20261021-215013-759070/manifest.json
- `Artifact and provenance verification` --references--> `Gaussian-blur replication manifest`  [INFERRED]
  collab/graphify/ARCHITECTURE.md → collab/experiment/logo-robustness-20261020-002912-258284/manifest.json
- `Artifact and provenance verification` --references--> `Logo intervention manifest`  [INFERRED]
  collab/graphify/ARCHITECTURE.md → collab/experiment/logo-intervention-20261019/manifest.json
- `Artifact and provenance verification` --references--> `Logo intervention normalized results`  [INFERRED]
  collab/graphify/ARCHITECTURE.md → collab/experiment/logo-intervention-20261019/results-normalized.json

## Import Cycles
- None detected.

## Communities (22 total, 7 thin omitted)

### Community 1 - "logo_duo_pilot.py"
Cohesion: 0.10
Nodes (15): build_gemini_prompt(), build_tasks(), discover_rows(), download_rows(), family_from_path(), fetch_json(), fetch_rows(), get_families() (+7 more)

### Community 2 - "semantic_label_pilot.py"
Cohesion: 0.05
Nodes (64): executable(), save(), invoke(), main(), summarize(), build_blur_gemini_prompt(), build_blur_tasks(), build_gemini_prompt() (+56 more)

### Community 3 - "Artifact and provenance verification"
Cohesion: 0.09
Nodes (34): Native-resolution Gaussian-blur confound (provisional), Matched-blur sigma-zero no-blur baseline, Matched-resolution Gaussian-blur correction, Artifact and provenance verification, Shared implementation review — 2026-10-12, Gaussian-blur robustness manifest, Corrected pixel-paired sham-control manifest, Visual-signal robustness manifest (+26 more)

### Community 5 - "ADR-001: Keep Graphify and Gemini Handoffs Deterministic"
Cohesion: 0.26
Nodes (13): Persistent Graphify map, Graphify maintenance loop, A hand-maintained graph with no source manifest, ADR-001: Keep Graphify and Gemini Handoffs Deterministic, Alternatives considered, Automatic Gemini semantic extraction, Code-only Graphify with no architecture manifest, Consequences (+5 more)

### Community 6 - "SchedulerTests"
Cohesion: 0.14
Nodes (3): parse_response(), invoke_claude(), SchedulerTests

### Community 7 - "augment_graphify_architecture.py"
Cohesion: 0.29
Nodes (8): alias_edges(), artifact_edges(), artifact_nodes(), edge(), main(), node(), source_state(), update_report_provenance()

### Community 8 - "Codex execution"
Cohesion: 0.14
Nodes (22): Bounded research question, Codex execution, Gemini prompt handoff, Visual-prior bias research, Graphify Gemini audit prompt, Implementation plan, Local research team — approved execution scope, 1. Intent (+14 more)

### Community 10 - "logo_paired_sham_pilot.py"
Cohesion: 0.09
Nodes (13): build_gemini_prompt(), build_tasks(), erase_region(), prepare_run(), _prompt_for(), relative_bbox(), select_original_rows(), derive_matched_blur_variant() (+5 more)

### Community 14 - "main"
Cohesion: 0.18
Nodes (9): build_workload_summary(), digest(), main(), prepare_inputs(), redact(), run_lock(), Runner, task_name_for_prompt() (+1 more)

### Community 15 - "Architecture manifest"
Cohesion: 0.06
Nodes (59): Baseline condition, Behavioral discrepancy is not causal mechanism, Causal mechanism (unproven), Codex baseline: 0/30 exact, 30/30 bias-aligned, Codex evidence-first: 0/30 exact, 30/30 bias-aligned, Codex counterfactual result, Evidence-first condition, Gemini baseline: 30/30 exact, 0/30 bias-aligned (+51 more)

### Community 16 - "check_project"
Cohesion: 0.15
Nodes (6): check_project(), _graph_issues(), main(), _secret_issues(), _status_issues(), ResearchConstraintsTests

### Community 17 - "IntegrationTests"
Cohesion: 0.26
Nodes (3): invoke_cli(), validate_budget_extension(), IntegrationTests

### Community 19 - "synthesis_prompt"
Cohesion: 0.32
Nodes (5): get_duo_owners(), get_owners(), get_role_context(), make_prompt(), synthesis_prompt()

### Community 23 - "collab.py"
Cohesion: 0.18
Nodes (3): check_consensus(), kill_tree(), run_process()

### Community 26 - "Epistemic discipline and auditability"
Cohesion: 0.18
Nodes (13): Research project: visual-prior bias, Project research instructions, Epistemic discipline and auditability, Matched subsets and compute budgets, Original/counterfactual family separation, Claim provenance labels, Repeat versus independent sample accounting, Enforced gates (+5 more)

## Knowledge Gaps
- **7 isolated node(s):** `Active Principles`, `Matched subsets and compute budgets`, `Claim provenance labels`, `Native-resolution Gaussian-blur confound (provisional)`, `Matched-blur Codex/Gemini model disagreement` (+2 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 62 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **7 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Artifact and provenance verification` connect `Artifact and provenance verification` to `Architecture manifest`?**
  _High betweenness centrality (0.004) - this node is a cross-community bridge._
- **Are the 10 inferred relationships involving `Artifact and provenance verification` (e.g. with `Evidence-preserving synthesis` and `Gaussian-blur robustness manifest`) actually correct?**
  _`Artifact and provenance verification` has 10 INFERRED edges - model-reasoned connections that need verification._
- **What connects `Active Principles`, `Matched subsets and compute budgets`, `Claim provenance labels` to the rest of the system?**
  _7 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `invoke` be split into smaller, more focused modules?**
  _Cohesion score 0.125 - nodes in this community are weakly interconnected._
- **Why does `Gemini Antigravity review` connect `Architecture manifest` to `Artifact and provenance verification`?**
  _High betweenness centrality (0.002) - this node is a cross-community bridge._
- **Are the 43 inferred relationships involving `Architecture manifest` (e.g. with `Graphify Gemini audit prompt` and `Gaussian-blur Gemini prompt`) actually correct?**
  _`Architecture manifest` has 43 INFERRED edges - model-reasoned connections that need verification._
- **Should `logo_duo_pilot.py` be split into smaller, more focused modules?**
  _Cohesion score 0.10483870967741936 - nodes in this community are weakly interconnected._
## Source-state provenance
- Architecture manifest: `collab/graphify/ARCHITECTURE.md`
- Git HEAD: `77de20aecc72acfbcc128d26d554b9e3dfc928b8`
- Worktree: **dirty** (161 changed/untracked paths)
- Worktree status SHA256: `a0ba4dd41feb59e11ffef42b00447da28041bb5cebaaceb19832b26a2bb0c907`
- Architecture manifest SHA256: `b0f319b17a4cf0ef0a9144cc5cce5287e5033d288fc5d70b4b3446b2e6cd7169`
- This report is not a clean-commit snapshot when the worktree is dirty.
