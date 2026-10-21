"""Run the project's low-cost research quality gates.

This checker is intentionally deterministic and dependency-free. It does not
call a model, fetch the network, or modify artifacts.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


SECRET_PATTERNS = (
    re.compile(r"AIza[0-9A-Za-z_-]{20,}"),
    re.compile(r"sk-[A-Za-z0-9]{20,}"),
    re.compile(r"-----BEGIN (?:RSA|OPENSSH|EC) PRIVATE KEY-----"),
)
TEXT_SUFFIXES = {
    ".json",
    ".md",
    ".py",
    ".ps1",
    ".sh",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
VALID_CONFIDENCE = {"EXTRACTED", "INFERRED", "AMBIGUOUS"}
BRIDGE_ID = "collab_graphify_architecture_epistemic_discipline_auditability"
OBJECTIVE_ID = "collab_graphify_architecture_visual_prior_bias"
CAUSAL_ID = "collab_graphify_architecture_causal_mechanism_unproven"
NEEDS_EVIDENCE_ID = "collab_graphify_architecture_needs_evidence"


def _graph_issues(graph_path: Path) -> list[str]:
    if not graph_path.exists():
        return [f"missing Graphify graph: {graph_path}"]
    try:
        graph = json.loads(graph_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"unreadable Graphify graph: {exc}"]

    nodes = graph.get("nodes")
    links = graph.get("links", graph.get("edges"))
    issues: list[str] = []
    if not isinstance(nodes, list) or not isinstance(links, list):
        return ["Graphify graph must contain list-valued nodes and links"]

    metadata = graph.get("graph", {})
    if not metadata.get("architecture_manifest"):
        issues.append("Graphify graph is missing architecture manifest provenance")
    if not metadata.get("architecture_source_state"):
        issues.append("Graphify graph is missing source-state provenance")

    node_ids = {str(item.get("id")) for item in nodes if isinstance(item, dict)}
    seen: set[tuple[str, str, str]] = set()
    edge_pairs: set[tuple[str, str]] = set()
    for index, link in enumerate(links):
        if not isinstance(link, dict):
            issues.append(f"link {index} is not an object")
            continue
        source = str(link.get("source"))
        target = str(link.get("target"))
        relation = str(link.get("relation", ""))
        if source not in node_ids or target not in node_ids:
            issues.append(f"dangling edge at link {index}: {source} -> {target}")
        if source == target:
            issues.append(f"self-loop edge at link {index}: {source}")
        identity = (source, target, relation)
        edge_pairs.add((source, target))
        if identity in seen:
            issues.append(f"duplicate edge at link {index}: {source} -> {target} ({relation})")
        seen.add(identity)
        confidence = link.get("confidence")
        if confidence is not None and confidence not in VALID_CONFIDENCE:
            issues.append(f"invalid edge confidence at link {index}: {confidence}")
    if not node_ids:
        issues.append("Graphify graph has no nodes")
    bridge_ids = {BRIDGE_ID, OBJECTIVE_ID, CAUSAL_ID, NEEDS_EVIDENCE_ID}
    if bridge_ids & node_ids:
        missing_bridge_nodes = sorted(bridge_ids - node_ids)
        if missing_bridge_nodes:
            issues.append("epistemic bridge is incomplete; missing nodes: " + ", ".join(missing_bridge_nodes))
        required_edges = {
            (OBJECTIVE_ID, BRIDGE_ID),
            (BRIDGE_ID, NEEDS_EVIDENCE_ID),
            (CAUSAL_ID, NEEDS_EVIDENCE_ID),
        }
        missing_edges = sorted(required_edges - edge_pairs)
        if missing_edges:
            issues.append("epistemic bridge is missing edges: " + ", ".join(f"{a}->{b}" for a, b in missing_edges))
        if (OBJECTIVE_ID, CAUSAL_ID) in edge_pairs:
            issues.append("direct objective-to-causal edge violates epistemic topology")
    return issues


def _status_issues(collab_root: Path) -> list[str]:
    summaries = sorted(collab_root.rglob("SUMMARY.md")) if collab_root.exists() else []
    if not summaries:
        return [f"no experiment summaries found under {collab_root}"]
    issues = []
    for path in summaries:
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            issues.append(f"unreadable summary {path}: {exc}")
            continue
        if "NEEDS_EVIDENCE" not in text:
            issues.append(f"summary missing NEEDS_EVIDENCE: {path}")
    return issues


def _secret_issues(root: Path) -> list[str]:
    issues = []
    skipped_dirs = {".git", ".venv", "__pycache__", "graphify-out"}
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        if any(part in skipped_dirs for part in path.parts):
            continue
        try:
            if path.stat().st_size > 2_000_000:
                continue
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        if any(pattern.search(text) for pattern in SECRET_PATTERNS):
            issues.append(f"secret-like value found in {path}")
    return issues


def check_project(root: str | Path) -> list[str]:
    """Return deterministic quality-gate failures for *root*."""

    root = Path(root).resolve()
    issues = _graph_issues(root / "graphify-out" / "graph.json")
    issues.extend(_status_issues(root / "collab"))
    issues.extend(_secret_issues(root))
    return issues


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", default=".", help="project root")
    args = parser.parse_args(argv)
    issues = check_project(args.root)
    if issues:
        for issue in issues:
            print(f"FAIL: {issue}")
        return 1
    print("Research constraints: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
