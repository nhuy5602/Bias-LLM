import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))

from open_vlm.runner import (  # noqa: E402
    prepare_run,
    run_development,
    run_held_out,
    run_no_regression,
    run_smoke,
    safe_upload_then_delete,
)


def manifest_rows(count=60):
    rows = []
    for index in range(count):
        domain = 'animals' if index % 2 == 0 else 'logos'
        rows.append({
            'domain': domain,
            'family': f'{domain}-{index:03}',
            'image_path': f'images/{index:03}.png',
            'ground_truth': 4,
            'expected_bias': 5,
            'prediction': None,
            'task_id': f'task-{index:03}',
        })
    return rows


class FakeBackend:
    def __init__(self, fail_once=False, method_predictions=None):
        self.calls = []
        self.fail_once = fail_once
        self.failed = set()
        self.method_predictions = method_predictions or {}

    def run(self, task, method):
        self.calls.append((task['task_id'], method))
        if self.fail_once and task['task_id'] not in self.failed:
            self.failed.add(task['task_id'])
            raise RuntimeError('transient backend error')
        return {
            'prediction': self.method_predictions.get(method, 4),
            'score_trace': {'method': method},
            'forward_passes': 2 if method == 'prior_contrastive' else 1,
            'seconds': 0.01,
            'peak_memory_mb': 10,
        }


class OpenVlmRunnerTests(unittest.TestCase):
    def config(self):
        return {
            'dataset_revision': '3761f9fd7163577534a7816c8bc1004035f2e2a0',
            'license_attestations': {'dataset': 'verified', 'model': 'verified'},
            'seed': 4,
            'processor': {'min_pixels': 256, 'max_pixels': 512},
            'method': {'sigmas': [1, 3, 6, 12], 'alphas': [0.25, 0.5, 1.0, 2.0]},
        }

    def test_prepare_run_writes_manifest_without_model_calls(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = prepare_run(Path(directory), 'smoke', manifest_rows(2), self.config())
            self.assertTrue((run_dir / 'manifest.json').is_file())
            state = json.loads((run_dir / 'state.json').read_text(encoding='utf-8'))
            self.assertEqual(state['status'], 'PREPARED')
            self.assertEqual(state['calls'], 0)

    def test_smoke_limits_to_50_and_does_not_make_scientific_decision(self):
        backend = FakeBackend()
        result = run_smoke({'rows': manifest_rows(), 'scientific_status': 'NEEDS_EVIDENCE'}, backend, limit=50)
        self.assertEqual(result['status'], 'COMPLETED')
        self.assertEqual(result['calls'], 50)
        self.assertEqual(result['scientific_decision'], 'NOT_EVALUATED')

    def test_smoke_dry_run_makes_zero_calls(self):
        backend = FakeBackend()
        result = run_smoke({'rows': manifest_rows(), 'scientific_status': 'NEEDS_EVIDENCE'}, backend, dry_run=True)
        self.assertEqual(result['status'], 'DRY_RUN')
        self.assertEqual(result['calls'], 0)
        self.assertEqual(backend.calls, [])

    def test_retry_is_recorded_and_successful_task_is_not_replayed(self):
        backend = FakeBackend(fail_once=True)
        result = run_smoke({'rows': manifest_rows(1), 'scientific_status': 'NEEDS_EVIDENCE'}, backend, max_retries=1)
        self.assertEqual(result['status'], 'COMPLETED')
        self.assertEqual(len(backend.calls), 2)
        self.assertEqual(result['attempts'], 2)

    def test_development_gate_redirects_when_method_fails(self):
        rows = manifest_rows(10)
        backend = FakeBackend(method_predictions={
            'free_generation': 5, 'candidate_scoring': 5, 'prior_contrastive': 7,
        })
        result = run_development({'rows': rows, 'scientific_status': 'NEEDS_EVIDENCE'}, backend)
        self.assertEqual(result['status'], 'BUDGET_STOPPED')
        self.assertEqual(result['next_action'], 'attention_logit_lens')

    def test_development_gates_contrastive_against_candidate_scoring(self):
        rows = manifest_rows(10)
        backend = FakeBackend(method_predictions={
            'free_generation': 4, 'candidate_scoring': 5, 'prior_contrastive': 4,
        })
        result = run_development({'rows': rows, 'scientific_status': 'NEEDS_EVIDENCE'}, backend)
        self.assertEqual(result['status'], 'PASS')
        self.assertEqual(result['baseline_candidate']['accuracy'], 0.0)
        self.assertEqual(result['method']['accuracy'], 1.0)
        self.assertIn(('task-000', 'candidate_scoring'), backend.calls)

    def test_development_uses_only_registered_dev_families(self):
        rows = manifest_rows(10)
        for row in rows:
            row['task_id'] = row['family']
        manifest = {
            'rows': rows,
            'family_split': {
                'animals': {'dev_families': ['animals-000']},
                'logos': {'dev_families': ['logos-001']},
            },
        }
        backend = FakeBackend(method_predictions={
            'free_generation': 5, 'candidate_scoring': 5, 'prior_contrastive': 4,
        })
        run_development(manifest, backend)
        called_ids = {task_id for task_id, _method in backend.calls}
        self.assertEqual(called_ids, {'animals-000', 'logos-001'})

    def test_held_out_returns_clustered_primary_gate(self):
        rows = [{
            'domain': 'flags',
            'family': f'flags-{index:02}',
            'image_path': f'images/flags-{index:02}.png',
            'ground_truth': 4,
            'prior_answer': 5,
            'expected_bias': 5,
            'task_id': f'flag-task-{index:02}',
        } for index in range(20)]
        backend = FakeBackend(method_predictions={
            'free_generation': 4, 'candidate_scoring': 5, 'prior_contrastive': 4,
        })
        result = run_held_out({'rows': rows, 'scientific_status': 'NEEDS_EVIDENCE'}, backend)
        self.assertEqual(result['status'], 'PASS')
        self.assertEqual(result['baseline_candidate']['accuracy'], 0.0)
        self.assertEqual(result['gate']['status'], 'PASS')

    def test_no_regression_requires_both_registered_sets(self):
        result = run_no_regression({'rows': []}, FakeBackend())
        self.assertEqual(result['status'], 'INCONCLUSIVE')

    def test_upload_failure_preserves_pod(self):
        with tempfile.TemporaryDirectory() as directory:
            artifact_dir = Path(directory)
            (artifact_dir / 'summary.json').write_text('{}', encoding='utf-8')
            deleted = []

            def uploader(*_args):
                raise RuntimeError('remote unavailable')

            result = safe_upload_then_delete('pod-1', artifact_dir, 'remote', uploader, deleted.append)
            self.assertEqual(result['status'], 'CLEANUP_BLOCKED')
            self.assertEqual(deleted, [])

    def test_successful_upload_verifies_then_deletes_pod(self):
        with tempfile.TemporaryDirectory() as directory:
            artifact_dir = Path(directory)
            (artifact_dir / 'summary.json').write_text('{}', encoding='utf-8')
            deleted = []
            expected = None

            def uploader(_artifact_dir, _destination):
                return {'checksum': expected}

            from open_vlm.runner import artifact_checksum
            expected = artifact_checksum(artifact_dir)
            result = safe_upload_then_delete('pod-1', artifact_dir, 'remote', uploader, deleted.append)
            self.assertEqual(result['status'], 'DELETED')
            self.assertEqual(deleted, ['pod-1'])


if __name__ == '__main__':
    unittest.main()
