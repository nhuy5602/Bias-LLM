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
from open_vlm.dataset import (load_visual_icl_rows, load_vlmbias_calibration_rows,
                              visual_icl_animal_family)
from open_vlm.qwen_backend import QwenBackend
from open_vlm.visual_icl import CONTEXT_COUNT, MIN_FAMILIES, register_visual_icl


class FakeBackend:
    def __init__(self, mismatch=False, fail_first=False):
        self.tasks = []
        self.mismatch = mismatch
        self.fail_first = fail_first

    def run(self, task, method):
        self.tasks.append(task)
        if self.fail_first and len(self.tasks) == 1:
            raise RuntimeError('model initialization failed')
        demos = task['demonstrations']
        truth = task['ground_truth']
        prediction = truth if (task['evaluation_split'] == 'original'
                               or task['arm'] == 'visual_icl') else task['prior_answer']
        return {'prediction': str(prediction), 'input_token_count': 100 + len(demos)
                + int(self.mismatch and task['arm'] == 'shuffled_labels'),
                'forward_passes': 1, 'image_encodes': 1 + len(demos), 'raw': str(prediction)}


class VisualIclTests(unittest.TestCase):
    @contextlib.contextmanager
    def small_protocol(self):
        with patch('open_vlm.visual_icl.MIN_FAMILIES', 2), \
             patch('open_vlm.visual_icl.CONTEXT_COUNT', 1):
            yield

    def test_dataset_loader_crosses_single_label_pages_and_deduplicates_families(self):
        def source(family, truth):
            return {'ID': family, 'topic': 'Animals', 'image_path': f'{family}_384.png',
                    'metadata': {'source_image': f'{family}_0_1_384.png'}, 'ground_truth': truth,
                    'expected_bias': truth - 1}

        def fetch_rows(*, where, **_kwargs):
            if where.endswith('= 3'):
                return [source('bird-a', 3), source('bird-b', 3)]
            return [source('mammal-a', 5)] * 99 + [source('mammal-b', 5)]

        with patch('open_vlm.dataset.fetch_rows', side_effect=fetch_rows):
            rows = load_visual_icl_rows(4, 42, None, None)
        self.assertEqual(sorted(row['ground_truth'] for row in rows), [3, 3, 5, 5])
        self.assertEqual(len({row['family'] for row in rows}), 4)

    def test_dataset_loader_filters_for_animals_before_paging(self):
        requests = []

        def fetch_rows(*, offsets, where, **_kwargs):
            requests.append((offsets, where))
            return []

        with patch('open_vlm.dataset.fetch_rows', side_effect=fetch_rows):
            with self.assertRaisesRegex(ValueError, 'distinct families'):
                load_visual_icl_rows(4, 42, None, {})
        self.assertEqual(requests, [
            ((0,), '"topic" = \'Animals\' AND "ground_truth" = 3'),
            ((0,), '"topic" = \'Animals\' AND "ground_truth" = 5'),
        ])

    def test_original_loader_filters_for_animals_before_paging(self):
        requests = []

        def fetch_rows(*, offsets, where, **_kwargs):
            requests.append((offsets, where))
            return []

        with patch('open_vlm.dataset.fetch_rows', side_effect=fetch_rows):
            with self.assertRaisesRegex(ValueError, 'Original VLMBias calibration'):
                load_vlmbias_calibration_rows(
                    ('animals',), max_per_domain=2, opener=None, page_cache={},
                    family_key=visual_icl_animal_family, where='"topic" = \'Animals\'')
        self.assertEqual(requests, [((0,), '"topic" = \'Animals\'')])

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

    def prepare(self, run_dir, per_label=20, original_count=20):
        def row(family, truth, prior, split='main'):
            name = f'{family}.png'
            content = name.encode()
            (run_dir / name).write_bytes(content)
            return {'family': family, 'domain': 'animals', 'source_id': family,
                    'image_path': name, 'source_sha256': hashlib.sha256(content).hexdigest(),
                    'ground_truth': truth, 'expected_bias': prior, 'prior_answer': prior,
                    'source_split': split}

        rows = [row(f'counterfactual-three-{index:02}', 3, 2) for index in range(per_label)]
        rows.extend(row(f'counterfactual-five-{index:02}', 5, 4) for index in range(per_label))
        original = [row(f'original-{index:02}', 2 if index < original_count // 2 else 4, None, 'original')
                    for index in range(original_count)]
        manifest = {
            'phase': 'visual-icl', 'seed': 42, 'dataset_revision': DATASET_REVISION,
            'model': {'id': 'Qwen/Qwen3-VL-32B-Instruct', 'revision': 'a' * 40},
            'execution_sources': execution_source_fingerprint(),
            'license_attestations': {'dataset': 'verified', 'model': 'verified'},
            'runtime': {'device': 'cpu', 'min_gpu_memory_gb': 80},
            'rows': rows, 'step0_rows': original,
            'family_split': {'animals': {'dev_families': [], 'same_domain_test_families': []}},
        }
        register_visual_icl(manifest)
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

    def test_contexts_are_balanced_ordered_and_disjoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = self.prepare(Path(tmp))
            contexts = manifest['visual_icl_contexts']
            self.assertEqual(len(contexts), CONTEXT_COUNT)
            self.assertEqual(manifest['visual_icl_registration']['expected_counterfactual_accuracy_gain'], 0.15)
            self.assertEqual(len(manifest['family_split']['animals']['same_domain_test_families']), MIN_FAMILIES)
            self.assertEqual(len({family for context in contexts for family in context['demo_families']}),
                             CONTEXT_COUNT * 4)
            orders = {tuple(context['answer_labels']) for context in contexts}
            self.assertGreater(len(orders), 1)
            for context in contexts:
                self.assertEqual(sorted(context['answer_labels']), [3, 3, 5, 5])

    def test_cli_evaluates_contexts_with_answer_space_control_and_preserves_attempts(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            with self.small_protocol():
                self.prepare(run_dir, per_label=4, original_count=2)
                backend = FakeBackend()
                code, result = self.run_cli(run_dir, backend)
            self.assertEqual((code, result['status']), (0, 'PASS'))
            self.assertEqual(result['scientific_status'], 'NEEDS_EVIDENCE')
            self.assertEqual(len(backend.tasks), 22)
            one = [task for task in backend.tasks if task['family'] == 'counterfactual-three-00']
            self.assertEqual(len(one), 4)
            self.assertEqual({task['arm'] for task in one},
                             {'zero_shot', 'answer_space_control', 'visual_icl', 'shuffled_labels'})
            self.assertTrue(all('sigma' not in task and 'prior_image' not in task for task in one))
            zero = [task for task in one if task['arm'] == 'zero_shot']
            answer_space = [task for task in one if task['arm'] == 'answer_space_control']
            self.assertEqual(len(zero), 1)
            self.assertEqual(len(answer_space), 1)
            self.assertIn('either 3 or 5', answer_space[0]['prompt'])
            aligned = next(task['demonstrations'] for task in one if task['arm'] == 'visual_icl')
            shuffled = next(task['demonstrations'] for task in one if task['arm'] == 'shuffled_labels')
            self.assertEqual([d['image'] for d in aligned], [d['image'] for d in shuffled])
            self.assertEqual(sorted(d['answer'] for d in aligned), sorted(d['answer'] for d in shuffled))
            self.assertTrue(all(a['answer'] != b['answer'] for a, b in zip(aligned, shuffled)))
            self.assertEqual(len(list(run_dir.glob('logs/**/intent.json'))), 22)
            with self.small_protocol():
                code, result = self.run_cli(run_dir, backend)
            self.assertEqual((code, result['status']), (0, 'PASS'))
            self.assertEqual(len(backend.tasks), 22)

    def test_dry_run_rejects_demo_leakage_and_corrupt_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            with self.small_protocol():
                manifest = self.prepare(run_dir, per_label=4, original_count=2)
                backend = FakeBackend()
                code, result = self.run_cli(run_dir, backend, '--dry-run')
            self.assertEqual((code, result['status']), (0, 'DRY_RUN'))
            manifest['visual_icl_contexts'][0]['demo_families'][0] = \
                manifest['family_split']['animals']['same_domain_test_families'][0]
            self.save(run_dir, manifest)
            with self.small_protocol():
                _, result = self.run_cli(run_dir, backend, '--dry-run')
            self.assertEqual(result['status'], 'INCONCLUSIVE')
            with self.small_protocol():
                register_visual_icl(manifest)
            self.save(run_dir, manifest)
            selected_demo = manifest['visual_icl_contexts'][0]['demo_families'][0]
            (run_dir / f'{selected_demo}.png').write_bytes(b'changed')
            with self.small_protocol():
                _, result = self.run_cli(run_dir, backend, '--dry-run')
            self.assertEqual(result['status'], 'INCONCLUSIVE')
            self.assertFalse(backend.tasks)

    def test_token_mismatch_cannot_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            with self.small_protocol():
                self.prepare(run_dir, per_label=4, original_count=2)
                code, result = self.run_cli(run_dir, FakeBackend(mismatch=True))
            self.assertEqual((code, result['status']), (1, 'INCONCLUSIVE'))
            self.assertIn('token', result['reason'].lower())

    def test_backend_failure_can_resume_without_preparing_a_new_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            with self.small_protocol():
                self.prepare(run_dir, per_label=4, original_count=2)
                backend = FakeBackend(fail_first=True)
                code, result = self.run_cli(run_dir, backend)
            self.assertEqual((code, result['status']), (1, 'INCONCLUSIVE'))
            self.assertEqual(len(backend.tasks), 1)
            self.assertEqual(len(list(run_dir.glob('logs/**/intent.json'))), 1)
            self.assertEqual(result['failures'][0]['error'], 'model initialization failed')
            resumed = FakeBackend()
            with self.small_protocol():
                code, result = self.run_cli(run_dir, resumed)
            self.assertEqual((code, result['status']), (0, 'PASS'))
            self.assertEqual(len(resumed.tasks), 22)
            first_task = run_dir / 'logs' / 'visual_icl' / 'visual-icl-0000-zero_shot'
            self.assertTrue((first_task / '1' / 'result.json').is_file())
            self.assertTrue((first_task / '2' / 'result.json').is_file())
