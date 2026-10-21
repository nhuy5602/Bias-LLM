"""Add the project workflow manifest to the persistent Graphify graph.

The normal Graphify semantic pass may use a cloud provider. This small deterministic
augmenter keeps the research workflow represented without making a model call. All
manual nodes and edges point back to ARCHITECTURE.md, so the provenance is explicit.
"""

from __future__ import annotations

import json
import hashlib
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GRAPH_PATH = ROOT / "graphify-out" / "graph.json"
SOURCE = "collab/graphify/ARCHITECTURE.md"


def node(
    node_id: str,
    label: str,
    location: str,
    *,
    source_file: str = SOURCE,
    file_type: str = "concept",
    provenance: str = "MANUAL_MANIFEST",
) -> dict:
    return {
        "id": node_id,
        "label": label,
        "file_type": file_type,
        "source_file": source_file,
        "source_location": location,
        "source_url": None,
        "captured_at": None,
        "author": None,
        "contributor": None,
        "provenance": provenance,
    }


def edge(
    source: str,
    target: str,
    relation: str,
    location: str,
    *,
    provenance: str = "MANUAL_MANIFEST",
    confidence_score: float = 0.85,
) -> dict:
    return {
        "source": source,
        "target": target,
        "relation": relation,
        "confidence": "INFERRED",
        "confidence_score": confidence_score,
        "source_file": SOURCE,
        "source_location": location,
        "provenance": provenance,
        "weight": 1.0,
    }


NODES = [
    node("collab_graphify_architecture_visual_prior_bias", "Visual-prior bias research", "Purpose"),
    node("collab_graphify_architecture_bounded_question", "Bounded research question", "Execution protocol 1"),
    node("collab_graphify_architecture_gemini_prompt", "Gemini prompt handoff", "Execution protocol 2"),
    node("collab_graphify_architecture_codex_execution", "Codex execution", "Execution protocol 3"),
    node("collab_graphify_architecture_gemini_review", "Gemini Antigravity review", "Execution protocol 4"),
    node("collab_graphify_architecture_provenance_verification", "Artifact and provenance verification", "Execution protocol 5"),
    node("collab_graphify_architecture_synthesis", "Evidence-preserving synthesis", "Execution protocol 6"),
    node("collab_graphify_architecture_graphify", "Persistent Graphify map", "Roles and boundaries"),
    node("collab_graphify_architecture_needs_evidence", "NEEDS_EVIDENCE", "Purpose"),
    node("collab_graphify_architecture_original_cf_separation", "Original/counterfactual family separation", "Evidence invariants"),
    node("collab_graphify_architecture_matched_subsets_compute", "Matched subsets and compute budgets", "Evidence invariants"),
    node("collab_graphify_architecture_repeat_accounting", "Repeat versus independent sample accounting", "Evidence invariants"),
    node("collab_graphify_architecture_provenance_labels", "Claim provenance labels", "Evidence invariants"),
    node("collab_graphify_architecture_behavioral_not_causal", "Behavioral discrepancy is not causal mechanism", "Evidence invariants"),
    node("collab_graphify_architecture_intervention", "Evidence-first prompt intervention", "Current pilot state"),
    node("collab_graphify_architecture_baseline_condition", "Baseline condition", "Current pilot state 2-4"),
    node("collab_graphify_architecture_evidence_first_condition", "Evidence-first condition", "Current pilot state 2-4"),
    node("collab_graphify_architecture_codex_result", "Codex counterfactual result", "Current pilot state 2"),
    node("collab_graphify_architecture_gemini_result", "Gemini counterfactual result", "Current pilot state 3"),
    node("collab_graphify_architecture_codex_baseline_result", "Codex baseline: 0/30 exact, 30/30 bias-aligned", "Current pilot state 2"),
    node("collab_graphify_architecture_codex_evidence_first_result", "Codex evidence-first: 0/30 exact, 30/30 bias-aligned", "Current pilot state 2"),
    node("collab_graphify_architecture_gemini_baseline_result", "Gemini baseline: 30/30 exact, 0/30 bias-aligned", "Current pilot state 3"),
    node("collab_graphify_architecture_gemini_evidence_first_result", "Gemini evidence-first: 30/30 exact, 0/30 bias-aligned", "Current pilot state 3"),
    node("collab_graphify_architecture_unresolved_controls", "Missing causal controls and generalization", "Current pilot state 5"),
    node(
        "collab_graphify_architecture_gaussian_blur_resolution_confound",
        "Native-resolution Gaussian-blur confound (provisional)",
        "Resolution-matched Gaussian-blur correction",
    ),
    node(
        "collab_graphify_architecture_matched_resolution_blur",
        "Matched-resolution Gaussian-blur correction",
        "Resolution-matched Gaussian-blur correction",
    ),
    node(
        "collab_graphify_architecture_matched_blur_audit",
        "Matched-blur hash and dimension audit",
        "Resolution-matched Gaussian-blur correction",
    ),
    node(
        "collab_graphify_architecture_matched_blur_model_disagreement",
        "Matched-blur Codex/Gemini model disagreement",
        "Resolution-matched Gaussian-blur correction",
    ),
    node(
        "collab_graphify_architecture_matched_blur_sigma_zero_baseline",
        "Matched-blur sigma-zero no-blur baseline",
        "Resolution-matched Gaussian-blur correction",
    ),
    node(
        "collab_graphify_architecture_matched_blur_v2_pending_audit",
        "Matched-blur v2 Codex/Gemini model disagreement",
        "Resolution-matched Gaussian-blur correction",
    ),
    node(
        "collab_graphify_architecture_paired_sham_model_disagreement",
        "Pixel-paired sham pilot model disagreement",
        "Pixel-paired sham-control pilot",
    ),
    node(
        "collab_graphify_architecture_superseded_paired_sham_run",
        "Superseded paired-sham run with invalid provenance",
        "Pixel-paired sham-control pilot",
    ),
    node(
        "collab_graphify_architecture_epistemic_discipline_auditability",
        "Epistemic discipline and auditability",
        "Binding bridge",
        provenance="USER_SUPPLIED_SYNTHESIS",
    ),
    node(
        "collab_graphify_architecture_causal_mechanism_unproven",
        "Causal mechanism (unproven)",
        "Binding bridge",
        provenance="USER_SUPPLIED_SYNTHESIS",
    ),
    node("collab_graphify_architecture_maintenance_loop", "Graphify maintenance loop", "Graph maintenance"),
]


EDGES = [
    edge("collab_graphify_architecture_visual_prior_bias", "collab_graphify_architecture_bounded_question", "rationale_for", "Execution protocol 1"),
    edge("collab_graphify_architecture_bounded_question", "collab_graphify_architecture_gemini_prompt", "references", "Execution protocol 2"),
    edge("collab_graphify_architecture_gemini_prompt", "collab_graphify_architecture_codex_execution", "references", "Execution protocol 2-3"),
    edge("collab_graphify_architecture_codex_execution", "collab_graphify_architecture_gemini_review", "references", "Execution protocol 3-4"),
    edge("collab_graphify_architecture_gemini_review", "collab_graphify_architecture_provenance_verification", "references", "Execution protocol 4-5"),
    edge("collab_graphify_architecture_provenance_verification", "collab_graphify_architecture_synthesis", "references", "Execution protocol 5-6"),
    edge("collab_graphify_architecture_synthesis", "collab_graphify_architecture_needs_evidence", "rationale_for", "Purpose"),
    edge("collab_graphify_architecture_graphify", "collab_graphify_architecture_provenance_verification", "references", "Roles and boundaries"),
    edge("collab_graphify_architecture_original_cf_separation", "collab_graphify_architecture_provenance_verification", "rationale_for", "Evidence invariants"),
    edge("collab_graphify_architecture_repeat_accounting", "collab_graphify_architecture_provenance_verification", "rationale_for", "Evidence invariants"),
    edge("collab_graphify_architecture_behavioral_not_causal", "collab_graphify_architecture_synthesis", "rationale_for", "Evidence invariants"),
    edge("collab_graphify_architecture_intervention", "collab_graphify_architecture_codex_result", "references", "Current pilot state 2"),
    edge("collab_graphify_architecture_intervention", "collab_graphify_architecture_gemini_result", "references", "Current pilot state 3"),
    edge("collab_graphify_architecture_intervention", "collab_graphify_architecture_baseline_condition", "references", "Current pilot state 2-4"),
    edge("collab_graphify_architecture_intervention", "collab_graphify_architecture_evidence_first_condition", "references", "Current pilot state 2-4"),
    edge("collab_graphify_architecture_baseline_condition", "collab_graphify_architecture_codex_baseline_result", "references", "Current pilot state 2"),
    edge("collab_graphify_architecture_baseline_condition", "collab_graphify_architecture_gemini_baseline_result", "references", "Current pilot state 3"),
    edge("collab_graphify_architecture_evidence_first_condition", "collab_graphify_architecture_codex_evidence_first_result", "references", "Current pilot state 2"),
    edge("collab_graphify_architecture_evidence_first_condition", "collab_graphify_architecture_gemini_evidence_first_result", "references", "Current pilot state 3"),
    edge("collab_graphify_architecture_codex_baseline_result", "collab_graphify_architecture_behavioral_not_causal", "rationale_for", "Evidence invariants"),
    edge("collab_graphify_architecture_codex_evidence_first_result", "collab_graphify_architecture_behavioral_not_causal", "rationale_for", "Evidence invariants"),
    edge("collab_graphify_architecture_gemini_baseline_result", "collab_graphify_architecture_behavioral_not_causal", "rationale_for", "Evidence invariants"),
    edge("collab_graphify_architecture_gemini_evidence_first_result", "collab_graphify_architecture_behavioral_not_causal", "rationale_for", "Evidence invariants"),
    edge("collab_graphify_architecture_codex_result", "collab_graphify_architecture_behavioral_not_causal", "rationale_for", "Evidence invariants"),
    edge("collab_graphify_architecture_gemini_result", "collab_graphify_architecture_behavioral_not_causal", "rationale_for", "Evidence invariants"),
    edge("collab_graphify_architecture_unresolved_controls", "collab_graphify_architecture_needs_evidence", "rationale_for", "Current pilot state 5"),
    edge(
        "collab_graphify_architecture_gaussian_blur_resolution_confound",
        "collab_graphify_architecture_matched_resolution_blur",
        "motivates",
        "Resolution-matched Gaussian-blur correction",
    ),
    edge(
        "collab_graphify_architecture_matched_resolution_blur",
        "collab_graphify_architecture_provenance_verification",
        "requires",
        "Resolution-matched Gaussian-blur correction",
    ),
    edge(
        "collab_graphify_architecture_matched_blur_audit",
        "collab_graphify_architecture_synthesis",
        "supports",
        "Resolution-matched Gaussian-blur correction",
    ),
    edge(
        "collab_graphify_architecture_matched_blur_model_disagreement",
        "collab_graphify_architecture_behavioral_not_causal",
        "rationale_for",
        "Resolution-matched Gaussian-blur correction",
    ),
    edge(
        "collab_graphify_architecture_matched_blur_sigma_zero_baseline",
        "collab_graphify_architecture_matched_resolution_blur",
        "constrains",
        "Resolution-matched Gaussian-blur correction",
    ),
    edge(
        "collab_graphify_architecture_matched_blur_v2_pending_audit",
        "collab_graphify_architecture_needs_evidence",
        "rationale_for",
        "Resolution-matched Gaussian-blur correction",
    ),
    edge(
        "collab_graphify_architecture_matched_blur_v2_pending_audit",
        "collab_graphify_architecture_behavioral_not_causal",
        "rationale_for",
        "Resolution-matched Gaussian-blur correction",
    ),
    edge(
        "collab_graphify_architecture_paired_sham_model_disagreement",
        "collab_graphify_architecture_behavioral_not_causal",
        "rationale_for",
        "Pixel-paired sham-control pilot",
    ),
    edge(
        "collab_graphify_architecture_superseded_paired_sham_run",
        "collab_graphify_architecture_provenance_verification",
        "references",
        "Pixel-paired sham-control pilot",
    ),
    edge("collab_graphify_architecture_graphify", "collab_graphify_architecture_maintenance_loop", "references", "Graph maintenance"),
    edge(
        "collab_graphify_architecture_visual_prior_bias",
        "collab_graphify_architecture_epistemic_discipline_auditability",
        "rationale_for",
        "Binding bridge",
        provenance="USER_SUPPLIED_SYNTHESIS",
        confidence_score=0.75,
    ),
    edge(
        "collab_graphify_architecture_graphify",
        "collab_graphify_architecture_epistemic_discipline_auditability",
        "references",
        "Binding bridge",
        provenance="USER_SUPPLIED_SYNTHESIS",
        confidence_score=0.75,
    ),
    edge(
        "collab_graphify_architecture_original_cf_separation",
        "collab_graphify_architecture_epistemic_discipline_auditability",
        "rationale_for",
        "Binding bridge",
        provenance="USER_SUPPLIED_SYNTHESIS",
        confidence_score=0.75,
    ),
    edge(
        "collab_graphify_architecture_matched_subsets_compute",
        "collab_graphify_architecture_epistemic_discipline_auditability",
        "rationale_for",
        "Binding bridge",
        provenance="USER_SUPPLIED_SYNTHESIS",
        confidence_score=0.75,
    ),
    edge(
        "collab_graphify_architecture_repeat_accounting",
        "collab_graphify_architecture_epistemic_discipline_auditability",
        "rationale_for",
        "Binding bridge",
        provenance="USER_SUPPLIED_SYNTHESIS",
        confidence_score=0.75,
    ),
    edge(
        "collab_graphify_architecture_provenance_labels",
        "collab_graphify_architecture_epistemic_discipline_auditability",
        "rationale_for",
        "Binding bridge",
        provenance="USER_SUPPLIED_SYNTHESIS",
        confidence_score=0.75,
    ),
    edge(
        "collab_graphify_architecture_behavioral_not_causal",
        "collab_graphify_architecture_epistemic_discipline_auditability",
        "rationale_for",
        "Binding bridge",
        provenance="USER_SUPPLIED_SYNTHESIS",
        confidence_score=0.75,
    ),
    edge(
        "collab_graphify_architecture_epistemic_discipline_auditability",
        "collab_graphify_architecture_needs_evidence",
        "rationale_for",
        "Binding bridge",
        provenance="USER_SUPPLIED_SYNTHESIS",
        confidence_score=0.75,
    ),
    edge(
        "collab_graphify_architecture_unresolved_controls",
        "collab_graphify_architecture_causal_mechanism_unproven",
        "references",
        "Binding bridge",
        provenance="USER_SUPPLIED_SYNTHESIS",
        confidence_score=0.75,
    ),
    edge(
        "collab_graphify_architecture_causal_mechanism_unproven",
        "collab_graphify_architecture_needs_evidence",
        "rationale_for",
        "Binding bridge",
        provenance="USER_SUPPLIED_SYNTHESIS",
        confidence_score=0.75,
    ),
]


ARTIFACTS = [
    ("collab_graphify_architecture_manifest", "Architecture manifest", SOURCE, "source", "document"),
    ("collab_graphify_gemini_prompt", "Graphify Gemini audit prompt", "collab/graphify/GEMINI-PROMPT.md", "Run metadata", "document"),
    ("logo_intervention_summary", "Logo intervention summary", "collab/experiment/logo-intervention-20261019/SUMMARY.md", "Summary", "document"),
    ("logo_intervention_manifest", "Logo intervention manifest", "collab/experiment/logo-intervention-20261019/manifest.json", "Manifest", "document"),
    ("logo_intervention_gemini_prompt", "Logo intervention Gemini prompt", "collab/experiment/logo-intervention-20261019/GEMINI-PROMPT.md", "Protocol", "document"),
    ("logo_intervention_gemini_report", "Logo intervention Gemini report", "collab/experiment/logo-intervention-20261019/GEMINI-REPORT.md", "Report", "document"),
    ("logo_intervention_normalized_results", "Logo intervention normalized results", "collab/experiment/logo-intervention-20261019/results-normalized.json", "Normalized results", "document"),
    ("logo_third_brand_summary", "Third-brand generalization summary", "collab/experiment/logo-duo-20261019-005644-359883/SUMMARY.md", "Third-brand generalization pilot", "document"),
    ("logo_third_brand_manifest", "Third-brand generalization manifest", "collab/experiment/logo-duo-20261019-005644-359883/manifest.json", "Third-brand generalization pilot", "document"),
    ("logo_third_brand_gemini_prompt", "Third-brand Gemini prompt", "collab/experiment/logo-duo-20261019-005644-359883/GEMINI-PROMPT.md", "Third-brand generalization pilot", "document"),
    ("logo_third_brand_gemini_report", "Third-brand Gemini report", "collab/experiment/logo-duo-20261019-005644-359883/GEMINI-REPORT.md", "Third-brand generalization pilot", "document"),
    ("logo_third_brand_normalized_results", "Third-brand normalized results", "collab/experiment/logo-duo-20261019-005644-359883/results-normalized.json", "Third-brand generalization pilot", "document"),
    ("logo_robustness_summary", "Visual-signal robustness summary", "collab/experiment/logo-robustness-20261019-022124-977565/SUMMARY.md", "Visual-signal robustness pilot", "document"),
    ("logo_robustness_manifest", "Visual-signal robustness manifest", "collab/experiment/logo-robustness-20261019-022124-977565/manifest.json", "Visual-signal robustness pilot", "document"),
    ("logo_robustness_gemini_prompt", "Visual-signal robustness Gemini prompt", "collab/experiment/logo-robustness-20261019-022124-977565/GEMINI-PROMPT.md", "Visual-signal robustness pilot", "document"),
    ("logo_robustness_gemini_report", "Visual-signal robustness Gemini report", "collab/experiment/logo-robustness-20261019-022124-977565/GEMINI-REPORT.md", "Visual-signal robustness pilot", "document"),
    ("logo_blur_summary", "Gaussian-blur robustness summary", "collab/experiment/logo-robustness-20261019-231529-364544/SUMMARY.md", "Gaussian-blur robustness pilot", "document"),
    ("logo_blur_manifest", "Gaussian-blur robustness manifest", "collab/experiment/logo-robustness-20261019-231529-364544/manifest.json", "Gaussian-blur robustness pilot", "document"),
    ("logo_blur_gemini_prompt", "Gaussian-blur Gemini prompt", "collab/experiment/logo-robustness-20261019-231529-364544/GEMINI-PROMPT.md", "Gaussian-blur robustness pilot", "document"),
    ("logo_blur_gemini_report", "Gaussian-blur Gemini report", "collab/experiment/logo-robustness-20261019-231529-364544/GEMINI-REPORT.md", "Gaussian-blur robustness pilot", "document"),
    ("logo_blur_replication_summary", "Gaussian-blur replication summary", "collab/experiment/logo-robustness-20261020-002912-258284/SUMMARY.md", "Gaussian-blur replication", "document"),
    ("logo_blur_replication_manifest", "Gaussian-blur replication manifest", "collab/experiment/logo-robustness-20261020-002912-258284/manifest.json", "Gaussian-blur replication", "document"),
    ("logo_blur_replication_gemini_prompt", "Gaussian-blur replication Gemini prompt", "collab/experiment/logo-robustness-20261020-002912-258284/GEMINI-PROMPT.md", "Gaussian-blur replication", "document"),
    ("logo_blur_replication_gemini_report", "Gaussian-blur replication Gemini report", "collab/experiment/logo-robustness-20261020-002912-258284/GEMINI-REPORT.md", "Gaussian-blur replication", "document"),
    ("logo_blur_matched_summary", "Matched-resolution Gaussian-blur summary", "collab/experiment/logo-robustness-20261021-181603-868764/SUMMARY.md", "Resolution-matched Gaussian-blur correction", "document"),
    ("logo_blur_matched_manifest", "Matched-resolution Gaussian-blur manifest", "collab/experiment/logo-robustness-20261021-181603-868764/manifest.json", "Resolution-matched Gaussian-blur correction", "document"),
    ("logo_blur_matched_gemini_prompt", "Matched-resolution Gaussian-blur Gemini prompt", "collab/experiment/logo-robustness-20261021-181603-868764/GEMINI-PROMPT.md", "Resolution-matched Gaussian-blur correction", "document"),
    ("logo_blur_matched_gemini_report", "Matched-resolution Gaussian-blur Gemini report", "collab/experiment/logo-robustness-20261021-181603-868764/GEMINI-REPORT.md", "Resolution-matched Gaussian-blur correction", "document"),
    ("logo_blur_matched_results", "Matched-resolution Gaussian-blur Codex results", "collab/experiment/logo-robustness-20261021-181603-868764/results.json", "Resolution-matched Gaussian-blur correction", "document"),
    ("logo_blur_matched_state", "Matched-resolution Gaussian-blur run state", "collab/experiment/logo-robustness-20261021-181603-868764/state.json", "Resolution-matched Gaussian-blur correction", "document"),
    ("logo_blur_matched_v2_summary", "Matched-resolution Gaussian-blur v2 summary", "collab/experiment/logo-robustness-20261021-215013-759070/SUMMARY.md", "Resolution-matched Gaussian-blur v2", "document"),
    ("logo_blur_matched_v2_manifest", "Matched-resolution Gaussian-blur v2 manifest", "collab/experiment/logo-robustness-20261021-215013-759070/manifest.json", "Resolution-matched Gaussian-blur v2", "document"),
    ("logo_blur_matched_v2_gemini_prompt", "Matched-resolution Gaussian-blur v2 Gemini prompt", "collab/experiment/logo-robustness-20261021-215013-759070/GEMINI-PROMPT.md", "Resolution-matched Gaussian-blur v2", "document"),
    ("logo_blur_matched_v2_gemini_report", "Matched-resolution Gaussian-blur v2 Gemini report", "collab/experiment/logo-robustness-20261021-215013-759070/GEMINI-REPORT.md", "Resolution-matched Gaussian-blur v2", "document"),
    ("logo_blur_matched_v2_results", "Matched-resolution Gaussian-blur v2 Codex results", "collab/experiment/logo-robustness-20261021-215013-759070/results.json", "Resolution-matched Gaussian-blur v2", "document"),
    ("logo_blur_matched_v2_state", "Matched-resolution Gaussian-blur v2 run state", "collab/experiment/logo-robustness-20261021-215013-759070/state.json", "Resolution-matched Gaussian-blur v2", "document"),
    ("logo_paired_sham_summary", "Corrected pixel-paired sham-control summary", "collab/experiment/logo-paired-sham-20261021-225532-353740/SUMMARY.md", "Pixel-paired sham-control pilot", "document"),
    ("logo_paired_sham_manifest", "Corrected pixel-paired sham-control manifest", "collab/experiment/logo-paired-sham-20261021-225532-353740/manifest.json", "Pixel-paired sham-control pilot", "document"),
    ("logo_paired_sham_gemini_prompt", "Corrected pixel-paired sham-control Gemini prompt", "collab/experiment/logo-paired-sham-20261021-225532-353740/GEMINI-PROMPT.md", "Pixel-paired sham-control pilot", "document"),
    ("logo_paired_sham_gemini_report", "Corrected pixel-paired sham-control Gemini report", "collab/experiment/logo-paired-sham-20261021-225532-353740/GEMINI-REPORT.md", "Pixel-paired sham-control pilot", "document"),
    ("logo_paired_sham_results", "Corrected pixel-paired sham-control Codex results", "collab/experiment/logo-paired-sham-20261021-225532-353740/results.json", "Pixel-paired sham-control pilot", "document"),
    ("logo_paired_sham_state", "Corrected pixel-paired sham-control run state", "collab/experiment/logo-paired-sham-20261021-225532-353740/state.json", "Pixel-paired sham-control pilot", "document"),
    ("logo_paired_sham_superseded_summary", "Superseded paired-sham summary", "collab/experiment/logo-paired-sham-20261021-223907-817729/SUMMARY.md", "Superseded paired-sham artifact", "document"),
    ("logo_paired_sham_superseded_manifest", "Superseded paired-sham manifest", "collab/experiment/logo-paired-sham-20261021-223907-817729/manifest.json", "Superseded paired-sham artifact", "document"),
    ("logo_paired_sham_superseded_gemini_report", "Superseded paired-sham Gemini report", "collab/experiment/logo-paired-sham-20261021-223907-817729/GEMINI-REPORT.md", "Superseded paired-sham artifact", "document"),
]


def artifact_nodes() -> list[dict]:
    return [
        node(node_id, label, location, source_file=source_file, file_type=file_type)
        for node_id, label, source_file, location, file_type in ARTIFACTS
        if (ROOT / source_file).is_file()
    ]


def artifact_edges() -> list[dict]:
    by_id = {item[0]: item for item in ARTIFACTS if (ROOT / item[2]).is_file()}
    edges = []
    for artifact_id in by_id:
        if artifact_id != "collab_graphify_architecture_manifest":
            edges.append(edge("collab_graphify_architecture_manifest", artifact_id, "references", "Graph maintenance"))
    if "collab_graphify_gemini_prompt" in by_id:
        edges.append(edge("collab_graphify_architecture_gemini_prompt", "collab_graphify_gemini_prompt", "references", "Roles and boundaries"))
    if "logo_intervention_manifest" in by_id:
        edges.append(edge("collab_graphify_architecture_provenance_verification", "logo_intervention_manifest", "references", "Current pilot state 5"))
    if "logo_intervention_summary" in by_id:
        edges.append(edge("collab_graphify_architecture_codex_result", "logo_intervention_summary", "references", "Current pilot state 2"))
    if "logo_intervention_gemini_report" in by_id:
        edges.append(edge("collab_graphify_architecture_gemini_result", "logo_intervention_gemini_report", "references", "Current pilot state 3"))
    if "logo_intervention_gemini_prompt" in by_id:
        edges.append(edge("logo_intervention_gemini_prompt", "collab_graphify_architecture_gemini_review", "references", "Protocol"))
    if "logo_intervention_normalized_results" in by_id:
        edges.append(edge("collab_graphify_architecture_provenance_verification", "logo_intervention_normalized_results", "references", "Current pilot state 5"))
    if "logo_third_brand_summary" in by_id:
        edges.append(edge("collab_graphify_architecture_codex_result", "logo_third_brand_summary", "references", "Third-brand generalization pilot"))
    if "logo_third_brand_manifest" in by_id:
        edges.append(edge("collab_graphify_architecture_provenance_verification", "logo_third_brand_manifest", "references", "Third-brand generalization pilot"))
    if "logo_third_brand_gemini_report" in by_id:
        edges.append(edge("collab_graphify_architecture_gemini_result", "logo_third_brand_gemini_report", "references", "Third-brand generalization pilot"))
    if "logo_third_brand_gemini_prompt" in by_id:
        edges.append(edge("logo_third_brand_gemini_prompt", "collab_graphify_architecture_gemini_review", "references", "Third-brand generalization pilot"))
    if "logo_third_brand_normalized_results" in by_id:
        edges.append(edge("collab_graphify_architecture_provenance_verification", "logo_third_brand_normalized_results", "references", "Third-brand generalization pilot"))
    if "logo_robustness_summary" in by_id:
        edges.append(edge("collab_graphify_architecture_codex_result", "logo_robustness_summary", "references", "Visual-signal robustness pilot"))
    if "logo_robustness_manifest" in by_id:
        edges.append(edge("collab_graphify_architecture_provenance_verification", "logo_robustness_manifest", "references", "Visual-signal robustness pilot"))
    if "logo_robustness_gemini_report" in by_id:
        edges.append(edge("collab_graphify_architecture_gemini_result", "logo_robustness_gemini_report", "references", "Visual-signal robustness pilot"))
    if "logo_robustness_gemini_prompt" in by_id:
        edges.append(edge("logo_robustness_gemini_prompt", "collab_graphify_architecture_gemini_review", "references", "Visual-signal robustness pilot"))
    if "logo_blur_summary" in by_id:
        edges.append(edge("collab_graphify_architecture_codex_result", "logo_blur_summary", "references", "Gaussian-blur robustness pilot"))
    if "logo_blur_manifest" in by_id:
        edges.append(edge("collab_graphify_architecture_provenance_verification", "logo_blur_manifest", "references", "Gaussian-blur robustness pilot"))
    if "logo_blur_gemini_report" in by_id:
        edges.append(edge("collab_graphify_architecture_gemini_result", "logo_blur_gemini_report", "references", "Gaussian-blur robustness pilot"))
    if "logo_blur_gemini_prompt" in by_id:
        edges.append(edge("logo_blur_gemini_prompt", "collab_graphify_architecture_gemini_review", "references", "Gaussian-blur robustness pilot"))
    if "logo_blur_replication_summary" in by_id:
        edges.append(edge("logo_blur_replication_summary", "collab_graphify_architecture_synthesis", "references", "Gaussian-blur replication"))
    if "logo_blur_replication_manifest" in by_id:
        edges.append(edge("collab_graphify_architecture_provenance_verification", "logo_blur_replication_manifest", "references", "Gaussian-blur replication"))
    if "logo_blur_replication_gemini_report" in by_id:
        edges.append(edge("collab_graphify_architecture_gemini_result", "logo_blur_replication_gemini_report", "references", "Gaussian-blur replication"))
    if "logo_blur_replication_gemini_prompt" in by_id:
        edges.append(edge("logo_blur_replication_gemini_prompt", "collab_graphify_architecture_gemini_review", "references", "Gaussian-blur replication"))
    if "logo_blur_matched_summary" in by_id:
        edges.append(edge("logo_blur_matched_summary", "collab_graphify_architecture_synthesis", "references", "Resolution-matched Gaussian-blur correction"))
    if "logo_blur_matched_manifest" in by_id:
        edges.append(edge("collab_graphify_architecture_provenance_verification", "logo_blur_matched_manifest", "references", "Resolution-matched Gaussian-blur correction"))
    if "logo_blur_matched_gemini_report" in by_id:
        edges.append(edge("collab_graphify_architecture_gemini_result", "logo_blur_matched_gemini_report", "references", "Resolution-matched Gaussian-blur correction"))
    if "logo_blur_matched_gemini_prompt" in by_id:
        edges.append(edge("logo_blur_matched_gemini_prompt", "collab_graphify_architecture_gemini_review", "references", "Resolution-matched Gaussian-blur correction"))
    if "logo_blur_matched_results" in by_id:
        edges.append(edge("collab_graphify_architecture_matched_blur_audit", "logo_blur_matched_results", "references", "Resolution-matched Gaussian-blur correction"))
    if "logo_blur_matched_state" in by_id:
        edges.append(edge("collab_graphify_architecture_matched_blur_audit", "logo_blur_matched_state", "references", "Resolution-matched Gaussian-blur correction"))
    if "logo_blur_matched_v2_summary" in by_id:
        edges.append(edge("logo_blur_matched_v2_summary", "collab_graphify_architecture_synthesis", "references", "Resolution-matched Gaussian-blur v2"))
    if "logo_blur_matched_v2_manifest" in by_id:
        edges.append(edge("collab_graphify_architecture_provenance_verification", "logo_blur_matched_v2_manifest", "references", "Resolution-matched Gaussian-blur v2"))
    if "logo_blur_matched_v2_gemini_report" in by_id:
        edges.append(edge("collab_graphify_architecture_gemini_result", "logo_blur_matched_v2_gemini_report", "references", "Resolution-matched Gaussian-blur v2"))
    if "logo_blur_matched_v2_gemini_prompt" in by_id:
        edges.append(edge("logo_blur_matched_v2_gemini_prompt", "collab_graphify_architecture_gemini_review", "references", "Resolution-matched Gaussian-blur v2"))
    if "logo_blur_matched_v2_results" in by_id:
        edges.append(edge("collab_graphify_architecture_matched_blur_v2_pending_audit", "logo_blur_matched_v2_results", "references", "Resolution-matched Gaussian-blur v2"))
    if "logo_blur_matched_v2_state" in by_id:
        edges.append(edge("collab_graphify_architecture_matched_blur_v2_pending_audit", "logo_blur_matched_v2_state", "references", "Resolution-matched Gaussian-blur v2"))
    if "logo_paired_sham_summary" in by_id:
        edges.append(edge("logo_paired_sham_summary", "collab_graphify_architecture_synthesis", "references", "Pixel-paired sham-control pilot"))
    if "logo_paired_sham_manifest" in by_id:
        edges.append(edge("collab_graphify_architecture_provenance_verification", "logo_paired_sham_manifest", "references", "Pixel-paired sham-control pilot"))
    if "logo_paired_sham_gemini_report" in by_id:
        edges.append(edge("collab_graphify_architecture_gemini_result", "logo_paired_sham_gemini_report", "references", "Pixel-paired sham-control pilot"))
    if "logo_paired_sham_gemini_prompt" in by_id:
        edges.append(edge("logo_paired_sham_gemini_prompt", "collab_graphify_architecture_gemini_review", "references", "Pixel-paired sham-control pilot"))
    if "logo_paired_sham_results" in by_id:
        edges.append(edge("collab_graphify_architecture_paired_sham_model_disagreement", "logo_paired_sham_results", "references", "Pixel-paired sham-control pilot"))
    if "logo_paired_sham_state" in by_id:
        edges.append(edge("collab_graphify_architecture_paired_sham_model_disagreement", "logo_paired_sham_state", "references", "Pixel-paired sham-control pilot"))
    if "logo_paired_sham_superseded_summary" in by_id:
        edges.append(edge("collab_graphify_architecture_superseded_paired_sham_run", "logo_paired_sham_superseded_summary", "references", "Superseded paired-sham artifact"))
    if "logo_paired_sham_superseded_manifest" in by_id:
        edges.append(edge("collab_graphify_architecture_superseded_paired_sham_run", "logo_paired_sham_superseded_manifest", "references", "Superseded paired-sham artifact"))
    if "logo_paired_sham_superseded_gemini_report" in by_id:
        edges.append(edge("collab_graphify_architecture_superseded_paired_sham_run", "logo_paired_sham_superseded_gemini_report", "references", "Superseded paired-sham artifact"))
    return edges


def alias_edges(nodes: list[dict]) -> list[dict]:
    """Connect document headings to curated concepts using source-scoped rules.

    Graphify's AST document nodes are useful navigation anchors but often have
    degree zero. These aliases are intentionally conservative: the source file
    determines the target concept, and every edge records CURATED_ALIAS provenance.
    """
    targets = {
        "epistemic": "collab_graphify_architecture_epistemic_discipline_auditability",
        "graphify": "collab_graphify_architecture_graphify",
        "execution": "collab_graphify_architecture_codex_execution",
        "gemini": "collab_graphify_architecture_gemini_review",
        "provenance": "collab_graphify_architecture_provenance_verification",
        "objective": "collab_graphify_architecture_visual_prior_bias",
        "synthesis": "collab_graphify_architecture_synthesis",
    }
    seen = set()
    aliases = []
    for item in nodes:
        if item.get("file_type") != "document" or item.get("provenance") == "MANUAL_MANIFEST":
            continue
        source = str(item.get("source_file", "")).replace("\\", "/")
        if source in {"AGENTS.md", "CLAUDE.md", "GEMINI.md", "CONSTRAINTS.md"}:
            target = targets["epistemic"]
        elif source == "README.md":
            target = targets["objective"]
        elif source == "docs/COLLAB.md" or source.startswith("docs/superpowers/specs/"):
            target = targets["execution"]
        elif source.startswith("docs/decisions/"):
            target = targets["graphify"]
        elif source == "docs/IMPLEMENTATION-REVIEW.md" or source.startswith("tasks/"):
            target = targets["provenance"]
        elif source.startswith("collab/graphify/") or source.startswith("collab/experiment/"):
            target = targets["gemini"] if "PROMPT" in str(item.get("label", "")).upper() else targets["provenance"]
        else:
            continue
        if item["id"] == target:
            continue
        identity = (item["id"], target, "references")
        if identity in seen:
            continue
        seen.add(identity)
        aliases.append(edge(
            item["id"], target, "references",
            f"Curated alias: {source}:{item.get('source_location', item.get('label', 'document'))}",
            provenance="CURATED_ALIAS",
            confidence_score=0.75,
        ))
    return aliases


def source_state() -> dict:
    def git(*args: str) -> str:
        try:
            result = subprocess.run(
                ["git", *args], cwd=ROOT, capture_output=True, text=True, check=True
            )
        except (OSError, subprocess.CalledProcessError):
            return "unavailable"
        return result.stdout.strip()

    status = git("status", "--porcelain", "--untracked-files=all")
    manifest = ROOT / SOURCE
    manifest_hash = hashlib.sha256(manifest.read_bytes()).hexdigest() if manifest.is_file() else "missing"
    return {
        "head": git("rev-parse", "HEAD"),
        "worktree": "dirty" if status else "clean",
        "change_count": len(status.splitlines()) if status else 0,
        "status_sha256": hashlib.sha256(status.encode("utf-8")).hexdigest(),
        "architecture_manifest_sha256": manifest_hash,
    }


def update_report_provenance(state: dict) -> None:
    report_path = ROOT / "graphify-out" / "GRAPH_REPORT.md"
    if not report_path.is_file():
        return
    text = report_path.read_text(encoding="utf-8")
    marker = "## Source-state provenance"
    if marker in text:
        text = text.split(marker, 1)[0].rstrip() + "\n"
    block = (
        f"\n{marker}\n"
        f"- Architecture manifest: `{SOURCE}`\n"
        f"- Git HEAD: `{state['head']}`\n"
        f"- Worktree: **{state['worktree']}** ({state['change_count']} changed/untracked paths)\n"
        f"- Worktree status SHA256: `{state['status_sha256']}`\n"
        f"- Architecture manifest SHA256: `{state['architecture_manifest_sha256']}`\n"
        "- This report is not a clean-commit snapshot when the worktree is dirty.\n"
    )
    report_path.write_text(text + block, encoding="utf-8")


def main() -> None:
    if not GRAPH_PATH.exists():
        raise SystemExit(f"missing Graphify graph: {GRAPH_PATH}")
    data = json.loads(GRAPH_PATH.read_text(encoding="utf-8"))
    nodes = data.setdefault("nodes", [])
    links = data.setdefault("links", [])
    all_nodes = NODES + artifact_nodes()
    all_edges = EDGES + artifact_edges()
    existing_nodes = {str(item.get("id")) for item in nodes}
    existing_links = {
        (str(item.get("source")), str(item.get("target")), str(item.get("relation", "")))
        for item in links
    }
    nodes.extend(item for item in all_nodes if item["id"] not in existing_nodes)
    all_edges.extend(alias_edges(nodes))
    links.extend(
        item
        for item in all_edges
        if (item["source"], item["target"], item["relation"]) not in existing_links
    )
    data["directed"] = True
    data.setdefault("graph", {})["architecture_manifest"] = SOURCE
    data["graph"]["architecture_augmenter"] = "scripts/augment_graphify_architecture.py"
    state = source_state()
    data["graph"]["architecture_source_state"] = state
    GRAPH_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    update_report_provenance(state)
    print(f"Graph augmented: {len(all_nodes)} architecture/artifact nodes, {len(all_edges)} architecture/artifact edges")
    print(f"Graph totals: {len(nodes)} nodes, {len(links)} links")


if __name__ == "__main__":
    main()
