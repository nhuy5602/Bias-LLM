import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))

from open_vlm.runner import (  # noqa: E402
    artifact_checksum,
    prepare_run,
    run_development,
    run_held_out,
    run_smoke,
    safe_upload_then_delete,
)


class PassingFakeBackend:
    def run(self, task, method):
        prediction = task['prior_answer'] if method in {'free_generation', 'candidate_scoring', 'self_consistency'} else task['ground_truth']
        return {
            'prediction': prediction,
            'score_trace': {'method': method},
            'forward_passes': 2 if method == 'prior_contrastive' else 1,
            'seconds': 0.01,
            'peak_memory_mb': 1,
        }


def rows():
    values = []
    for domain in ('animals', 'logos'):
        for index in range(5):
            values.append({
                'domain': domain,
                'family': f'{domain}-{index}',
                'image_path': f'images/{domain}-{index}.png',
                'ground_truth': 4,
                'expected_bias': 5,
            })
    for index in range(20):
        values.append({
            'domain': 'flags',
            'family': f'flags-{index}',
            'image_path': f'images/flags-{index}.png',
            'ground_truth': 4,
            'expected_bias': 5,
        })
    return values


class OpenVlmEndToEndTests(unittest.TestCase):
    def config(self):
        return {
            'dataset_revision': '3761f9fd7163577534a7816c8bc1004035f2e2a0',
            'license_attestations': {'dataset': 'verified', 'model': 'verified'},
            'seed': 20261022,
            'processor': {'min_pixels': 256, 'max_pixels': 512},
            'method': {'sigmas': [1, 3, 6, 12], 'alphas': [0.25, 0.5, 1.0, 2.0]},
            'step0_records': [{'sigma': 3, 'subject_valid_rate': 1.0,
                               'invalid_rate': 0.0, 'clear_accuracy': 1.0,
                               'blur_accuracy': 0.0, 'blur_bias_rate': 1.0,
                               'black_bias_rate': 0.0}],
        }

    def test_fake_run_covers_manifest_phases_metrics_and_cleanup(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = prepare_run(Path(directory), 'development', rows(), self.config())
            manifest = json.loads((run_dir / 'manifest.json').read_text(encoding='utf-8'))
            backend = PassingFakeBackend()
            smoke = run_smoke(manifest, backend, limit=50)
            development = run_development(manifest, backend)
            held_out = run_held_out(manifest, backend)
            self.assertEqual(smoke['scientific_decision'], 'NOT_EVALUATED')
            self.assertEqual(development['status'], 'PASS')
            self.assertEqual(held_out['status'], 'PASS')
            self.assertEqual(development['baseline_candidate']['accuracy'], 0.0)
            self.assertEqual(development['method']['accuracy'], 1.0)
            self.assertEqual(development['method']['other_error_rate'], 0.0)

            deleted = []
            checksum = artifact_checksum(run_dir)
            cleanup = safe_upload_then_delete(
                'pod-test', run_dir, 'remote',
                lambda *_args: {'checksum': checksum}, deleted.append,
            )
            self.assertEqual(cleanup['status'], 'DELETED')
            self.assertEqual(deleted, ['pod-test'])


if __name__ == '__main__':
    unittest.main()
