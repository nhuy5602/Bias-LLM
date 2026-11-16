import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))

import open_vlm_pilot
from open_vlm.schema import DATASET_REVISION, execution_source_fingerprint
from open_vlm.dataset import load_visual_icl_rows, visual_icl_animal_family
from open_vlm.qwen_backend import QwenBackend


class FakeBackend:
    def __init__(self, mismatch=False):
        self.tasks = []
        self.mismatch = mismatch

    def run(self, task, method):
        self.tasks.append(task)
        demos = task['demonstrations']
        correct = not demos or all(d['answer'] == '3' for d in demos[:2])
        truth = task['ground_truth']
        prediction = truth if truth % 2 == 0 or (demos and correct) else task['prior_answer']
        return {'prediction': str(prediction), 'input_token_count': 100 + len(demos)
                + int(self.mismatch and bool(demos) and not correct),
                'forward_passes': 1, 'image_encodes': 1 + len(demos), 'raw': str(prediction)}


class VisualIclTests(unittest.TestCase):
    def test_dataset_loader_crosses_single_label_pages_and_deduplicates_families(self):
        def source(family, truth):
            return {'ID': family, 'topic': 'Animals', 'image_path': f'{family}_384.png',
                    'metadata': {'source_image': f'{family}_0_1_384.png'}, 'ground_truth': truth,
                    'expected_bias': truth - 1}

        with patch('open_vlm.dataset.fetch_rows', side_effect=[
            [source('mammal-a', 5)] * 99 + [source('mammal-b', 5)],
            [source('bird-a', 3), source('bird-b', 3)],
        ]):
            rows = load_visual_icl_rows(4, 42, None, None)
        self.assertEqual(sorted(row['ground_truth'] for row in rows), [3, 3, 5, 5])
        self.assertEqual(len({row['family'] for row in rows}), 4)

    def test_visual_icl_family_ignores_question_resolution_and_original_split(self):
        rows = [
            {'family': 'animals:redfox-q1-384', 'metadata': {'source_image': 'red fox_1_1_384.png'}},
            {'family': 'animals:redfox-q2-1152', 'metadata': {'source_image': 'red fox_1_1_1152.png'}},
            {'family': 'animals:redfox-original', 'metadata': {'source_image': 'red fox.png'}},
        ]
        self.assertEqual({visual_icl_animal_family(row) for row in rows}, {'animals:redfox'})

    def test_backend_sends_demonstrations_without_exposing_query_answer(self):
        class Processor:
            def apply_chat_template(self, messages, **kwargs):
                self.messages = messages
                return Inputs(token_type_ids=None)

            def batch_decode(self, generated, **kwargs):
                return ['5']

        class Inputs(dict):
            input_ids = SimpleNamespace(shape=(1, 27))

            def to(self, device):
                return self

        class TokenIds:
            def __getitem__(self, index):
                if index[1].start != 27:
                    raise AssertionError('Prompt must be trimmed')
                return SimpleNamespace(shape=(1, 2))

        processor = Processor()
        backend = QwenBackend(model=SimpleNamespace(generate=lambda **kw: TokenIds()), processor=processor)
        result = backend.run({
            'image': '/query.png', 'prompt': 'Count legs.', 'candidates': ['3', '5'],
            'ground_truth': 3, 'demonstrations': [{'image': '/demo.png', 'answer': '5'}],
        }, 'visual_icl')
        self.assertEqual([turn['role'] for turn in processor.messages], ['user', 'assistant', 'user'])
        self.assertEqual(processor.messages[1]['content'][0]['text'], '5')
        self.assertEqual(processor.messages[2]['content'][0]['path'], '/query.png')
        self.assertEqual(result['prediction'], '5')
        self.assertEqual(result['image_encodes'], 2)
        self.assertEqual(result['input_token_count'], 27)

    def prepare(self, run_dir):
        def row(family, truth, prior, split='main'):
            name = f'{family}.png'
            content = name.encode()
            (run_dir / name).write_bytes(content)
            return {'family': family, 'domain': 'animals', 'source_id': family,
                    'image_path': name, 'source_sha256': hashlib.sha256(content).hexdigest(),
                    'ground_truth': truth, 'expected_bias': prior, 'prior_answer': prior,
                    'source_split': split}

        demos = [row(f'demo{i}', 3 if i < 2 else 5, 2 if i < 2 else 4) for i in range(4)]
        test = [row(f'test{i}', 3, 2) for i in range(20)]
        original = [row(f'original{i}', 2 if i < 10 else 4, None, 'original') for i in range(20)]
        manifest = {
            'phase': 'visual-icl', 'seed': 42, 'dataset_revision': DATASET_REVISION,
            'model': {'id': 'Qwen/Qwen3-VL-32B-Instruct', 'revision': 'a' * 40},
            'execution_sources': execution_source_fingerprint(),
            'license_attestations': {'dataset': 'verified', 'model': 'verified'},
            'runtime': {'device': 'cpu'},
            'rows': demos + test, 'step0_rows': original,
            'family_split': {'animals': {'dev_families': [r['family'] for r in demos],
                                         'same_domain_test_families': [r['family'] for r in test]}},
        }
        manifest['visual_icl_demo_families'] = ['demo1', 'demo0', 'demo2', 'demo3']
        self.save(run_dir, manifest)
        return manifest

    @staticmethod
    def save(run_dir, manifest):
        (run_dir / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')

    def run_cli(self, run_dir, backend, phase='--visual-icl'):
        output = io.StringIO()
        with patch.object(open_vlm_pilot.QwenBackend, 'from_manifest', return_value=backend):
            with contextlib.redirect_stdout(output):
                code = open_vlm_pilot.main([phase, '--run-dir', str(run_dir)])
        return code, json.loads(output.getvalue())

    def test_cli_evaluates_matched_arms_and_preserves_attempts(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            self.prepare(run_dir)
            backend = FakeBackend()
            code, result = self.run_cli(run_dir, backend)
            self.assertEqual((code, result['status']), (0, 'PASS'))
            self.assertEqual(result['scientific_status'], 'NEEDS_EVIDENCE')
            self.assertEqual(len(backend.tasks), 120)
            one = [t for t in backend.tasks if t['family'] == 'test0']
            self.assertEqual(len(one), 3)
            self.assertTrue(all('sigma' not in task and 'prior_image' not in task for task in one))
            zero, aligned, shuffled = [t['demonstrations'] for t in one]
            self.assertEqual(zero, [])
            self.assertEqual([d['image'] for d in aligned], [d['image'] for d in shuffled])
            self.assertEqual(sorted(d['answer'] for d in aligned), sorted(d['answer'] for d in shuffled))
            self.assertTrue(all(a['answer'] != b['answer'] for a, b in zip(aligned, shuffled)))
            self.assertEqual(len(list(run_dir.glob('logs/**/intent.json'))), 120)
            code, result = self.run_cli(run_dir, backend)
            self.assertEqual((code, result['status']), (1, 'INCONCLUSIVE'))
            self.assertEqual(len(backend.tasks), 120)

    def test_dry_run_rejects_demo_leakage_and_corrupt_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            manifest = self.prepare(run_dir)
            backend = FakeBackend()
            code, result = self.run_cli(run_dir, backend, '--dry-run')
            self.assertEqual((code, result['status']), (0, 'DRY_RUN'))
            manifest['visual_icl_demo_families'][0] = 'test0'
            self.save(run_dir, manifest)
            _, result = self.run_cli(run_dir, backend, '--dry-run')
            self.assertEqual(result['status'], 'INCONCLUSIVE')
            manifest['visual_icl_demo_families'][0] = 'demo1'
            self.save(run_dir, manifest)
            (run_dir / 'demo0.png').write_bytes(b'changed')
            _, result = self.run_cli(run_dir, backend, '--dry-run')
            self.assertEqual(result['status'], 'INCONCLUSIVE')
            self.assertFalse(backend.tasks)

    def test_token_mismatch_cannot_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            self.prepare(run_dir)
            code, result = self.run_cli(run_dir, FakeBackend(mismatch=True))
            self.assertEqual((code, result['status']), (1, 'INCONCLUSIVE'))
            self.assertIn('token', result['reason'].lower())

    def test_backend_failure_stops_new_inference_and_retains_intent(self):
        class FailingBackend(FakeBackend):
            def run(self, task, method):
                self.tasks.append(task)
                raise RuntimeError('GPU unavailable')

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            self.prepare(run_dir)
            backend = FailingBackend()
            code, result = self.run_cli(run_dir, backend)
            self.assertEqual((code, result['status']), (1, 'INCONCLUSIVE'))
            self.assertEqual(len(backend.tasks), 1)
            self.assertEqual(len(list(run_dir.glob('logs/**/intent.json'))), 1)
            self.assertEqual(result['failures'][0]['error'], 'GPU unavailable')
