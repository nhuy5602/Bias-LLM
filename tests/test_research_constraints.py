import json
import tempfile
import unittest
from pathlib import Path

from scripts.check_research_constraints import check_project


class ResearchConstraintsTests(unittest.TestCase):
    def write_project(self, graph, summary="Scientific: NEEDS_EVIDENCE\n", source="safe text\n"):
        root = Path(self.tmp.name)
        (root / "graphify-out").mkdir()
        (root / "collab" / "runs" / "sample").mkdir(parents=True)
        graph.setdefault("graph", {})["architecture_manifest"] = "collab/graphify/ARCHITECTURE.md"
        graph["graph"]["architecture_source_state"] = {"head": "test", "worktree": "clean"}
        (root / "graphify-out" / "graph.json").write_text(
            json.dumps(graph), encoding="utf-8"
        )
        (root / "collab" / "runs" / "sample" / "SUMMARY.md").write_text(
            summary, encoding="utf-8"
        )
        (root / "notes.txt").write_text(source, encoding="utf-8")
        return root

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def test_valid_project_has_no_constraint_violations(self):
        graph = {
            "nodes": [{"id": "a"}, {"id": "b"}],
            "links": [
                {
                    "source": "a",
                    "target": "b",
                    "relation": "references",
                    "confidence": "EXTRACTED",
                }
            ],
        }
        self.assertEqual(check_project(self.write_project(graph)), [])

    def test_graph_integrity_reports_dangling_and_self_loop_edges(self):
        graph = {
            "nodes": [{"id": "a"}],
            "links": [
                {"source": "a", "target": "missing", "relation": "references"},
                {"source": "a", "target": "a", "relation": "references"},
            ],
        }
        issues = check_project(self.write_project(graph))
        self.assertTrue(any("dangling" in issue for issue in issues))
        self.assertTrue(any("self-loop" in issue for issue in issues))

    def test_missing_scientific_status_is_reported(self):
        graph = {"nodes": [{"id": "a"}], "links": []}
        issues = check_project(self.write_project(graph, summary="Scientific: TBD\n"))
        self.assertTrue(any("NEEDS_EVIDENCE" in issue for issue in issues))

    def test_secret_like_value_is_reported(self):
        graph = {"nodes": [{"id": "a"}], "links": []}
        issues = check_project(self.write_project(graph, source="AIza" + "A" * 30))
        self.assertTrue(any("secret" in issue.lower() for issue in issues))

    def test_epistemic_bridge_rejects_direct_objective_to_causal_edge(self):
        objective = "collab_graphify_architecture_visual_prior_bias"
        bridge = "collab_graphify_architecture_epistemic_discipline_auditability"
        causal = "collab_graphify_architecture_causal_mechanism_unproven"
        needs = "collab_graphify_architecture_needs_evidence"
        graph = {
            "nodes": [{"id": node_id} for node_id in (objective, bridge, causal, needs)],
            "links": [
                {"source": objective, "target": bridge, "relation": "rationale_for"},
                {"source": bridge, "target": needs, "relation": "rationale_for"},
                {"source": causal, "target": needs, "relation": "rationale_for"},
                {"source": objective, "target": causal, "relation": "references"},
            ],
        }
        issues = check_project(self.write_project(graph))
        self.assertTrue(any("direct objective-to-causal" in issue for issue in issues))


if __name__ == "__main__":
    unittest.main()
