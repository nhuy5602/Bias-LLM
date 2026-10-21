import sys
import json
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from logo_duo_pilot import (DATASET, FAMILIES, GEMINI_MODEL, MERCEDES_FAMILIES,
                             REPEATS, REVISION, build_gemini_prompt, build_tasks,
                             family_from_path, fetch_rows, get_families, invoke,
                             main, parse_count, run_tasks, select_rows)


class LogoDuoPilotTests(unittest.TestCase):
    def test_parse_count_accepts_braced_integer_only(self):
        self.assertEqual(parse_count('{5}'), 5)
        self.assertEqual(parse_count(' 5 '), 5)
        self.assertIsNone(parse_count('five'))
        self.assertIsNone(parse_count('{5} because...'))

    def test_select_rows_requires_all_ten_family_pairs(self):
        rows = []
        for family, values in FAMILIES.items():
            brand = family.split('-')[0]
            original_gt = '4' if brand == 'audi' else '3'
            rows.extend([
                {'image_path': f'original_images/images/{family.replace("-", "_")}_0.png',
                 'state': 'original', 'family': family, 'ground_truth': original_gt,
                 'expected_bias': None},
                {'image_path': f'vlms-are-biased-notitle/car_logos/images/{family}_384.png',
                 'state': 'counterfactual', 'family': family,
                 'ground_truth': str(values['bias']), 'expected_bias': str(values['ground_truth'])},
            ])
        selected = select_rows(rows)
        self.assertEqual(len(selected), 20)
        self.assertEqual({r['family'] for r in selected}, set(FAMILIES))
        self.assertEqual({r['state'] for r in selected}, {'original', 'counterfactual'})

    def test_build_tasks_runs_three_codex_repeats_per_state(self):
        rows = [
            {'family': 'audi-black-convertible', 'state': 'original', 'ground_truth': '4', 'expected_bias': None, 'local_path': 'original.png', 'prompt': 'Count the circles.'},
            {'family': 'audi-black-convertible', 'state': 'counterfactual', 'ground_truth': '5', 'expected_bias': '4', 'local_path': 'cf.png', 'prompt': 'Count the circles.'},
            {'family': 'maserati-black-convertible', 'state': 'original', 'ground_truth': '3', 'expected_bias': None, 'local_path': 'maserati-original.png', 'prompt': 'Count the prongs.'},
            {'family': 'maserati-black-convertible', 'state': 'counterfactual', 'ground_truth': '5', 'expected_bias': '3', 'local_path': 'maserati-cf.png', 'prompt': 'Count the prongs.'},
        ]
        tasks = build_tasks(rows)
        self.assertEqual(len(tasks), 12)
        self.assertEqual({t['agent'] for t in tasks}, {'codex'})
        self.assertEqual({t['state'] for t in tasks}, {'original', 'counterfactual'})
        self.assertEqual({t['repeat'] for t in tasks}, set(range(1, REPEATS + 1)))
        self.assertEqual({t['prompt'] for t in tasks if t['family'].startswith('audi-')},
                         {'Count the overlapping circles in this Audi logo. Return one integer only.'})
        self.assertEqual({t['prompt'] for t in tasks if t['family'].startswith('maserati-')},
                         {'Count the prongs in this Maserati logo. Return one integer only.'})

    def test_expanded_pilot_uses_ten_fixed_families(self):
        self.assertEqual(len(FAMILIES), 10)
        self.assertEqual(len(set(FAMILIES)), 10)

    def test_mercedes_arm_is_opt_in_and_uses_four_complete_pairs(self):
        families = get_families(include_mercedes=True)
        self.assertEqual(len(families), 14)
        self.assertEqual(set(MERCEDES_FAMILIES), {
            'mercedes-black-convertible', 'mercedes-black-coupe',
            'mercedes-black-hatchback', 'mercedes-black-suv',
        })
        self.assertEqual(
            family_from_path('mercedes_benz-black-coupe_384.png', families),
            'mercedes-black-coupe',
        )
        self.assertEqual(
            family_from_path('mercedes-benz_black_coupe_0.png', families),
            'mercedes-black-coupe',
        )
        rows = []
        for family, values in families.items():
            rows.extend([
                {'family': family, 'state': 'original', 'ground_truth': values['ground_truth'],
                 'expected_bias': None},
                {'family': family, 'state': 'counterfactual', 'ground_truth': values['bias'],
                 'expected_bias': values['ground_truth']},
            ])
        self.assertEqual(len(select_rows(rows, families)), 28)

    def test_mercedes_tasks_use_brand_specific_prompt(self):
        rows = [{'family': 'mercedes-black-coupe', 'state': 'counterfactual',
                 'ground_truth': 4, 'expected_bias': 3, 'local_path': 'cf.png',
                 'prompt': 'Count points.'}]
        tasks = build_tasks(rows, get_families(include_mercedes=True))
        self.assertEqual(len(tasks), REPEATS)
        self.assertEqual(
            {task['prompt'] for task in tasks},
            {'Count the points on the star in this Mercedes-Benz logo. Return one integer only.'},
        )

    def test_prepared_run_dir_can_be_checked_without_launching_models(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder / 'manifest.json').write_text(json.dumps({'tasks': []}), encoding='utf-8')
            self.assertEqual(main(['--run-dir', directory, '--dry-run']), 0)

    def test_gemini_prompt_pins_model_and_labels_repeats_as_non_independent(self):
        rows = [{'family': 'audi-black-convertible', 'state': 'original',
                 'local_path': 'original.png', 'sha256': 'abc'}]
        prompt = build_gemini_prompt(rows)
        self.assertIn(GEMINI_MODEL, prompt)
        self.assertIn('stop before running', prompt.lower())
        self.assertIn('not independent samples', prompt.lower())

    def test_dataset_row_requests_pin_the_recorded_revision(self):
        from unittest.mock import patch
        urls = []

        def fake_fetch(url):
            urls.append(url)
            return {'rows': []}

        with patch('logo_duo_pilot.fetch_json', side_effect=fake_fetch):
            fetch_rows('main', (0,))

        self.assertEqual(len(urls), 1)
        self.assertIn('revision=' + REVISION, urls[0])
        self.assertIn('dataset=' + DATASET.replace('/', '%2F'), urls[0])

    def test_run_tasks_checkpoints_launch_before_call_and_records_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            task = {'task_id': 'audi-original-codex', 'family': 'audi', 'state': 'original',
                    'agent': 'codex', 'image_path': 'image.png', 'prompt': 'Count circles.',
                    'ground_truth': 4, 'expected_bias': None}
            checkpoints = []

            def invoke(member, prompt, image_path, attempt_dir, timeout):
                state = __import__('json').loads((folder / 'state.json').read_text(encoding='utf-8'))
                checkpoints.append((member, state['calls'], state['tasks'][task['task_id']]['status']))
                raise RuntimeError('provider unavailable')

            results, state = run_tasks(folder, [task], 10, invoke_fn=invoke)
            self.assertEqual(checkpoints, [('codex', 1, 'RUNNING')])
            self.assertEqual(state['calls'], 1)
            self.assertEqual(state['tasks'][task['task_id']]['status'], 'FAILED')
            self.assertEqual(state['status'], 'DEGRADED')
            self.assertFalse(results[0]['valid'])
            attempt = folder / 'logs' / task['task_id'] / '1'
            detail = json.loads((attempt / 'attempt.json').read_text(encoding='utf-8'))
            self.assertEqual(detail['status'], 'FAILED')
            self.assertEqual((attempt / 'stderr.txt').read_text(encoding='utf-8'), 'provider unavailable')

    def test_invoke_rejects_gemini_in_codex_only_runner(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                invoke('gemini', 'Count.', Path(directory) / 'image.png', Path(directory) / 'attempt', 1)


if __name__ == '__main__':
    unittest.main()
