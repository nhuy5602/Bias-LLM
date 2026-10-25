import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))

from open_vlm.runner import (  # noqa: E402
    build_backend_task,
    prepare_run,
    run_development,
    run_held_out,
    run_no_regression,
    run_step0,
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
        fallback = 5 if method == 'self_consistency' else 4
        return {
            'prediction': self.method_predictions.get(method, fallback),
            'score_trace': {'method': method},
            'forward_passes': 2 if method == 'prior_contrastive' else 1,
            'seconds': 0.01,
            'peak_memory_mb': 10,
        }


class CostBackend(FakeBackend):
    def run(self, task, method):
        result = super().run(task, method)
        result['cost_usd'] = 1.0
        return result


class Step0Backend(FakeBackend):
    def __init__(self):
        super().__init__()
        self.subject_tasks = []

    def run(self, task, method):
        self.calls.append((task['task_id'], method))
        if method == 'subject_validity':
            self.subject_tasks.append(dict(task))
            return {'prediction': task['candidates'][1] if task.get('condition') == 'black'
                    else task['candidates'][0],
                    'forward_passes': 1, 'seconds': 0.01}
        condition = task.get('condition')
        prediction = {'clear': 4, 'blur': 5, 'black': 4}.get(condition, 4)
        return {'prediction': prediction, 'forward_passes': 1, 'seconds': 0.01}


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

    def test_backend_task_maps_manifest_fields_and_candidate_values(self):
        row = manifest_rows(1)[0]
        task = build_backend_task(row, sigma=3, alpha=1.0)
        self.assertEqual(task['candidates'], [str(value) for value in range(21)])
        self.assertEqual(task['alpha'], 1.0)
        self.assertIn('image_path', task)

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

    def test_smoke_checkpoints_results_and_state_when_run_dir_is_present(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = prepare_run(Path(directory), 'smoke', manifest_rows(2), self.config())
            manifest = json.loads((run_dir / 'manifest.json').read_text(encoding='utf-8'))
            result = run_smoke(manifest, FakeBackend(), limit=2)
            self.assertEqual(result['status'], 'COMPLETED')
            self.assertTrue((run_dir / 'results' / 'candidate_scoring.jsonl').is_file())
            self.assertTrue((run_dir / 'results' / 'smoke-summary.json').is_file())
            state = json.loads((run_dir / 'state.json').read_text(encoding='utf-8'))
            self.assertEqual(state['calls'], 2)
            self.assertEqual(state['phase_status'], 'COMPLETED')

    def test_budget_ceiling_stops_after_cost_is_recorded(self):
        with tempfile.TemporaryDirectory() as directory:
            config = {**self.config(), 'budget_usd': 0.5}
            run_dir = prepare_run(Path(directory), 'smoke', manifest_rows(2), config)
            manifest = json.loads((run_dir / 'manifest.json').read_text(encoding='utf-8'))
            result = run_smoke(manifest, CostBackend(), limit=2)
            self.assertEqual(result['status'], 'SMOKE_FAILED')
            state = json.loads((run_dir / 'state.json').read_text(encoding='utf-8'))
            self.assertEqual(state['status'], 'BUDGET_STOPPED')
            self.assertGreater(state['cost_usd'], 0.5)

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
        result = run_development({'rows': rows, 'scientific_status': 'NEEDS_EVIDENCE',
                                  'step0_records': [{'sigma': 3, 'subject_valid_rate': 1.0,
                                                     'invalid_rate': 0.0, 'clear_accuracy': 1.0,
                                                     'blur_accuracy': 0.0, 'blur_bias_rate': 1.0,
                                                     'black_bias_rate': 0.0}]}, backend)
        self.assertEqual(result['status'], 'GATE_STOPPED')
        self.assertEqual(result['next_action'], 'attention_logit_lens')

    def test_development_gates_contrastive_against_candidate_scoring(self):
        rows = manifest_rows(10)
        backend = FakeBackend(method_predictions={
            'free_generation': 4, 'candidate_scoring': 5, 'prior_contrastive': 4,
        })
        result = run_development({'rows': rows, 'scientific_status': 'NEEDS_EVIDENCE',
                                  'step0_records': [{'sigma': 3, 'subject_valid_rate': 1.0,
                                                     'invalid_rate': 0.0, 'clear_accuracy': 1.0,
                                                     'blur_accuracy': 0.0, 'blur_bias_rate': 1.0,
                                                     'black_bias_rate': 0.0}]}, backend)
        self.assertEqual(result['status'], 'PASS')
        self.assertEqual(result['baseline_candidate']['accuracy'], 0.0)
        self.assertEqual(result['method']['accuracy'], 1.0)
        self.assertIn(('task-000', 'candidate_scoring'), backend.calls)

    def test_step0_selects_blur_sigma_using_black_control(self):
        rows = manifest_rows(10)
        backend = Step0Backend()
        result = run_step0({'rows': rows}, backend)
        self.assertEqual(result['status'], 'PASS')
        self.assertEqual(result['selected_sigma'], 1)
        self.assertEqual(result['records'][0]['blur_bias_rate'], 1.0)
        self.assertTrue(backend.subject_tasks)
        self.assertEqual(backend.subject_tasks[0]['candidates'], ['animals', 'other'])
        self.assertNotIn('clearly visible', backend.subject_tasks[0]['prompt'])

    def test_step0_stops_before_model_calls_when_calibration_is_underpowered(self):
        backend = Step0Backend()
        result = run_step0({'rows': manifest_rows(2)}, backend)
        self.assertEqual(result['status'], 'INCONCLUSIVE')
        self.assertIsNone(result['selected_sigma'])
        self.assertEqual(backend.calls, [])
        self.assertIn('at least 5 source families', result['reason'])

    def test_step0_calibration_uses_original_rows_and_calibration_thresholds(self):
        rows = []
        for index in range(10):
            rows.append({
                'domain': 'animals',
                'family': f'animals:standard-{index:02}',
                'source_split': 'original',
                'image_path': f'calibration/{index:02}.png',
                'ground_truth': 4,
                'expected_bias': 5,
                'subject_label': 'animal',
            })
        backend = Step0Backend()
        result = run_step0({
            'rows': [],
            'step0_rows': rows,
            'step0_bias_rows': rows,
            'step0_calibration': True,
            'step0_thresholds': {
                'min_families': 10,
                'min_clear_accuracy': 0.80,
                'min_detail_drop': 0.20,
                'max_black_subject_validity': 0.10,
            },
        }, backend)
        self.assertEqual(result['status'], 'PASS')
        self.assertEqual(result['selected_sigma'], 1)

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
            'step0_records': [{'sigma': 3, 'subject_valid_rate': 1.0,
                               'invalid_rate': 0.0, 'clear_accuracy': 1.0,
                               'blur_accuracy': 0.0, 'blur_bias_rate': 1.0,
                               'black_bias_rate': 0.0}],
        }
        backend = FakeBackend(method_predictions={
            'free_generation': 5, 'candidate_scoring': 5, 'prior_contrastive': 4,
        })
        run_development(manifest, backend)
        called_ids = {task_id for task_id, _method in backend.calls
                      if '-sigma' not in task_id and '-alpha' not in task_id}
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
        result = run_held_out({'rows': rows, 'scientific_status': 'NEEDS_EVIDENCE',
                               'development_status': 'PASS'}, backend)
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

    def test_upload_failure_stops_pod_but_preserves_volume(self):
        with tempfile.TemporaryDirectory() as directory:
            artifact_dir = Path(directory)
            stopped = []
            result = safe_upload_then_delete(
                'pod-1', artifact_dir, 'remote',
                lambda *_args: (_ for _ in ()).throw(RuntimeError('remote unavailable')),
                lambda pod_id, **kwargs: stopped.append((pod_id, kwargs)),
                stopper=lambda pod_id, **kwargs: stopped.append((pod_id, kwargs)),
            )
            self.assertEqual(result['status'], 'STOPPED_CLEANUP_BLOCKED')
            self.assertEqual(stopped, [('pod-1', {'preserve_volume': True})])

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
