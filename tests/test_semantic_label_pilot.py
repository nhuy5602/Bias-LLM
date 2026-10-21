import io
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from semantic_label_pilot import (bind_task_image, build_tasks, format_response, make_prompt, parse_count,
                                  image_extension, make_custom_command, migrate_v1_run, select_samples,
                                  migrate_v2_run, migrate_v3_api_run, migrate_v4_model_run,
                                  migrate_v5_model_run, migrate_v6_model_run, migrate_v7_model_run,
                                  migrate_v8_model_run, migrate_v9_model_run,
                                  migrate_v10_claude_run, claude_command,
                                  migrate_v11_label_run,
                                  claude_budget_allows_call,
                                  species_from_row, stable_digest,
                                  build_api_request, generate_api, parse_api_response, pilot_status,
                                  api_failure_summary, task_retry_exhausted, validate_launch_ledger)


class SemanticLabelPilotTests(unittest.TestCase):
    def candidates(self):
        return [
            {
                'species': f'{group}-{i}',
                'animal_group': group,
                'image_path': f'{group}-{i}.png',
                'image_url': f'https://example.test/{group}-{i}.png',
                'ground_truth': '3' if group == 'bird' else '5',
                'expected_bias': '2' if group == 'bird' else '4',
            }
            for group in ('bird', 'mammal') for i in range(15)
        ]

    def test_selector_is_seeded_balanced_and_unique(self):
        candidates = self.candidates()
        first = select_samples(candidates, count=20, seed=17)
        second = select_samples(candidates, count=20, seed=17)
        self.assertEqual(first, second)
        self.assertEqual(len({sample['species'] for sample in first}), 20)
        self.assertEqual(sum(sample['animal_group'] == 'bird' for sample in first), 10)
        self.assertEqual(sum(sample['animal_group'] == 'mammal' for sample in first), 10)

    def test_task_matrix_has_80_calls_and_balances_each_image_and_prompt(self):
        samples = select_samples(self.candidates(), count=20, seed=17)
        tasks = build_tasks(samples)
        self.assertEqual(len(tasks), 80)
        for sample in samples:
            rows = [task for task in tasks if task['species'] == sample['species']]
            self.assertEqual(len(rows), 4)
            self.assertEqual({task['condition'] for task in rows}, {'generic', 'named'})
            self.assertEqual({task['replicate'] for task in rows}, {1, 2})

    def test_treatment_changes_only_the_species_label(self):
        generic = make_prompt({'species': 'horse', 'condition': 'generic', 'image_path': 'image-01.jpg'})
        named = make_prompt({'species': 'horse', 'condition': 'named', 'image_path': 'image-01.jpg'})
        self.assertIn('this animal', generic)
        self.assertNotIn('horse', generic)
        self.assertIn('this horse', named)
        self.assertEqual(generic.replace('this animal', 'this horse'), named)

    def test_gemini_command_uses_multimodal_at_file_injection(self):
        prompt = make_custom_command('Count the legs of this egret.', 'C:/pilot/images/image-01.jpg')
        self.assertIn('Count the legs of this egret.', prompt)
        self.assertIn('@{C:/pilot/images/image-01.jpg}', prompt)
        self.assertIn('image', prompt.lower())

    def test_image_reference_is_resolved_inside_pilot_run_folder(self):
        task = {'image_path': 'images/image-01.jpg'}
        bound = bind_task_image(Path('C:/pilot/run'), task)
        self.assertEqual(bound['image_path'], (Path('C:/pilot/run') / 'images/image-01.jpg').resolve().as_posix())

    def test_parse_count_accepts_one_integer_only(self):
        self.assertEqual(parse_count('{5}'), 5)
        self.assertEqual(parse_count(' 5\n'), 5)
        self.assertIsNone(parse_count('I see 5 legs'))
        self.assertIsNone(parse_count('{5} or {4}'))

    def test_short_integer_answer_is_saved_as_a_valid_runner_report(self):
        report = format_response('5')
        self.assertGreaterEqual(len(report), 80)
        self.assertEqual(parse_count(json.loads(report)['response']), 5)

    def test_resume_rejects_launch_count_lower_than_saved_attempts(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            attempt = folder / 'logs' / 'image-01' / '1'
            attempt.mkdir(parents=True)
            with self.assertRaisesRegex(ValueError, 'launch ledger'):
                validate_launch_ledger(folder, {'calls': 0})

    def test_dataset_image_format_falls_back_to_binary_signature(self):
        self.assertEqual(image_extension('binary/octet-stream', b'\x89PNG\r\n\x1a\nrest'), '.png')
        self.assertEqual(image_extension('image/jpeg', b'\xff\xd8rest'), '.jpg')
        self.assertIsNone(image_extension('binary/octet-stream', b'unknown'))

    def test_species_name_comes_from_dataset_id_not_resolution_filename_suffix(self):
        self.assertEqual(species_from_row({'ID': 'animal_089_secretarybird_notitle_Q1_px384',
                                           'image_path': 'secretarybird_2_3_384.png'}), 'secretary bird')
        self.assertEqual(species_from_row({'ID': 'animal_002_zebra_notitle_Q1_px384',
                                           'image_path': 'zebra_5_3_384.png'}), 'zebra')

    def test_migration_preserves_bad_attempt_and_supersedes_with_corrected_task(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder / 'images').mkdir()
            image = folder / 'images/image-01.jpg'
            image.write_bytes(b'\xff\xd8sample')
            sample = {'species': 'egret 0 3 384', 'animal_group': 'bird', 'ground_truth': '3',
                      'expected_bias': '2', 'dataset_id': 'animal_069_egret_notitle_Q1_px384',
                      'image_path': 'egret_0_3_384.png', 'local_path': 'images/image-01.jpg',
                      'image_sha256': hashlib.sha256(image.read_bytes()).hexdigest()}
            task = {'task_id': 'image-01-named-1', 'sample_id': 'image-01', 'species': sample['species'],
                    'animal_group': 'bird', 'image_path': sample['local_path'], 'condition': 'named',
                    'replicate': 1, 'ground_truth': 3, 'expected_bias': 2}
            manifest = {'protocol': 'semantic-label-cf-v1', 'created': 'now', 'samples': [sample],
                        'tasks': [task]}
            manifest['protocol_hash'] = stable_digest({k: v for k, v in manifest.items() if k != 'created'})
            (folder / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
            state = {'calls': 1, 'pilot_protocol_hash': manifest['protocol_hash'], 'tasks': {
                task['task_id']: {'attempts': 1, 'status': 'RUNNING', 'prompt_hash': 'old'}}}
            (folder / 'state.json').write_text(json.dumps(state), encoding='utf-8')
            migrate_v1_run(folder, manifest)
            updated = json.loads((folder / 'manifest.json').read_text(encoding='utf-8'))
            saved_state = json.loads((folder / 'state.json').read_text(encoding='utf-8'))
            self.assertTrue((folder / 'manifest-v1.json').is_file())
            self.assertEqual(updated['protocol'], 'semantic-label-cf-v2')
            self.assertEqual(updated['samples'][0]['species'], 'egret')
            self.assertEqual(updated['tasks'][0]['task_id'], task['task_id'] + '-corrected')
            self.assertEqual(updated['protocol_hash'], stable_digest(
                {k: v for k, v in updated.items() if k not in ('created', 'protocol_hash')}))
            self.assertEqual(saved_state['calls'], 1)
            self.assertEqual(saved_state['tasks'][task['task_id']]['status'], 'SUPERSEDED')

    def test_v2_migration_preserves_attempts_and_reenables_gemini_within_same_cap(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            task = {'task_id': 'image-03-named-1-corrected', 'image_path': 'images/image-03.jpg',
                    'species': 'egret', 'condition': 'named'}
            manifest = {'protocol': 'semantic-label-cf-v2', 'created': 'now', 'tasks': [task],
                        'samples': [], 'migration': {'from': 'semantic-label-cf-v1',
                            'superseded_task': 'image-03-named-1', 'reason': 'species parsing'}}
            manifest['protocol_hash'] = stable_digest({k: v for k, v in manifest.items()
                                                       if k not in ('created', 'protocol_hash')})
            (folder / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
            state = {'calls': 5, 'pilot_protocol_hash': manifest['protocol_hash'],
                     'unavailable': {'gemini': 'CLI timeout after 240.0s'}, 'tasks': {
                         task['task_id']: {'attempts': 4, 'status': 'FAILED', 'error': 'timeout'}}}
            (folder / 'state.json').write_text(json.dumps(state), encoding='utf-8')
            (folder / 'logs').mkdir()
            (folder / 'reports').mkdir()
            migrate_v2_run(folder, manifest)
            updated = json.loads((folder / 'manifest.json').read_text(encoding='utf-8'))
            saved_state = json.loads((folder / 'state.json').read_text(encoding='utf-8'))
            self.assertTrue((folder / 'manifest-v2.json').is_file())
            self.assertEqual(updated['protocol'], 'semantic-label-cf-v3')
            self.assertEqual(updated['tasks'][0]['task_id'], task['task_id'] + '-path-fixed')
            self.assertEqual(saved_state['calls'], 5)
            self.assertEqual(saved_state['tasks'][task['task_id']]['status'], 'SUPERSEDED')
            self.assertEqual(saved_state['unavailable'], {})

    def test_task_prompt_uses_bound_image_absolute_path(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            image = folder / 'images/image-01.jpg'
            image.parent.mkdir()
            image.write_bytes(b'image')
            task = bind_task_image(folder, {'image_path': 'images/image-01.jpg', 'species': 'egret',
                                             'condition': 'named'})
            self.assertIn(str(image.resolve()).replace('\\', '/'), make_prompt(task))

    def test_direct_api_request_contains_inline_image_and_never_the_key(self):
        request = build_api_request('Count the legs of this animal.', b'jpeg-data', 'image/jpeg')
        self.assertEqual(request['contents'][0]['parts'][0], {'text': 'Count the legs of this animal.'})
        self.assertEqual(request['contents'][0]['parts'][1], {
            'inlineData': {'mimeType': 'image/jpeg', 'data': 'anBlZy1kYXRh'}})
        self.assertNotIn('key', json.dumps(request).lower())

    def test_api_response_parser_reads_first_text_part_and_rejects_api_error(self):
        payload = {'candidates': [{'content': {'parts': [{'text': '5'}]}}]}
        self.assertEqual(parse_api_response(payload), '5')
        with self.assertRaisesRegex(RuntimeError, 'quota'):
            parse_api_response({'error': {'message': 'quota exceeded'}})

    def test_api_failure_summary_distinguishes_model_quota_and_load_errors(self):
        self.assertIn('unavailable to new users', api_failure_summary('Model no longer available to new users'))
        self.assertIn('quota/billing', api_failure_summary('quota exceeded'))
        self.assertIn('high demand', api_failure_summary('model is currently experiencing high demand'))

    def test_one_invocation_canary_does_not_mark_the_pilot_global_budget_exhausted(self):
        self.assertEqual(pilot_status(False, 8, 1000), 'INCOMPLETE')
        self.assertEqual(pilot_status(False, 80, 1000), 'BUDGET_EXHAUSTED')
        self.assertEqual(pilot_status(True, 8, 1000), 'COMPLETED')
        self.assertEqual(pilot_status(False, 9, 1000, unavailable=True), 'DEGRADED')

    def test_exhausted_task_is_skipped_on_resume_but_pending_task_is_not(self):
        self.assertTrue(task_retry_exhausted({'attempts': 4, 'status': 'FAILED'}))
        self.assertTrue(task_retry_exhausted({'attempts': 4, 'status': 'BUDGET_EXHAUSTED'}))
        self.assertFalse(task_retry_exhausted({'attempts': 3, 'status': 'FAILED'}))
        self.assertFalse(task_retry_exhausted({'attempts': 4, 'status': 'SUPERSEDED'}))

    def test_api_call_keeps_key_in_header_and_out_of_saved_request(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / 'sample.jpg'
            image.write_bytes(b'\xff\xd8image')
            attempt = root / 'logs' / 'task' / '1'
            attempt.mkdir(parents=True)
            response = io.BytesIO(b'{"candidates":[{"content":{"parts":[{"text":"5"}]}}]}')
            with patch('semantic_label_pilot.urllib.request.urlopen', return_value=response) as urlopen:
                result = generate_api('Count legs.', image, attempt, 3, model='gemini-2.5-pro',
                                      api_key='secret-do-not-log')
            request = urlopen.call_args.args[0]
            self.assertEqual(result, '5')
            self.assertEqual(request.get_header('X-goog-api-key'), 'secret-do-not-log')
            artifact = (attempt / 'api-request.json').read_text(encoding='utf-8')
            self.assertNotIn('secret-do-not-log', artifact)
            self.assertIn('gemini-2.5-pro', request.full_url)

    def test_v3_api_migration_preserves_seven_launches_and_versions_task_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            image = folder / 'images/image-01.jpg'
            image.parent.mkdir()
            image.write_bytes(b'\xff\xd8image')
            sample = {'local_path': 'images/image-01.jpg',
                      'image_sha256': hashlib.sha256(image.read_bytes()).hexdigest()}
            tasks = [{'task_id': f'image-01-{condition}-{rep}', 'condition': condition,
                      'replicate': rep, 'image_path': sample['local_path']}
                     for condition in ('generic', 'named') for rep in (1, 2)]
            manifest = {'protocol': 'semantic-label-cf-v3', 'created': 'now', 'samples': [sample],
                        'tasks': tasks, 'superseded_tasks': [], 'calls_max': 80}
            manifest['protocol_hash'] = stable_digest({k: v for k, v in manifest.items()
                                                       if k not in ('created', 'protocol_hash')})
            (folder / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
            state = {'calls': 7, 'status': 'INTERRUPTED',
                     'pilot_protocol_hash': manifest['protocol_hash'], 'tasks': {
                         tasks[0]['task_id']: {'attempts': 2, 'status': 'RUNNING'},
                         **{task['task_id']: {'attempts': 0} for task in tasks[1:]}}}
            (folder / 'state.json').write_text(json.dumps(state), encoding='utf-8')
            migrate_v3_api_run(folder, manifest)
            updated = json.loads((folder / 'manifest.json').read_text(encoding='utf-8'))
            saved_state = json.loads((folder / 'state.json').read_text(encoding='utf-8'))
            self.assertTrue((folder / 'manifest-v3.json').is_file())
            self.assertEqual(updated['protocol'], 'semantic-label-cf-v4')
            self.assertEqual(updated['model'], 'gemini-2.5-pro')
            self.assertEqual(len(updated['tasks']), 4)
            self.assertTrue(all(task['task_id'].endswith('-api') for task in updated['tasks']))
            self.assertEqual(saved_state['calls'], 7)
            self.assertEqual(saved_state['tasks'][tasks[0]['task_id']]['attempts'], 2)
            self.assertEqual(saved_state['tasks'][tasks[0]['task_id']]['status'], 'SUPERSEDED')

    def test_v4_model_migration_records_explicit_user_choice_without_resetting_calls(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            task = {'task_id': 'image-01-generic-1-api', 'condition': 'generic'}
            manifest = {'protocol': 'semantic-label-cf-v4', 'created': 'now', 'model': 'gemini-2.5-pro',
                        'tasks': [task], 'samples': [], 'transport': 'Gemini Developer API v1beta generateContent'}
            manifest['protocol_hash'] = stable_digest({k: v for k, v in manifest.items()
                                                       if k not in ('created', 'protocol_hash')})
            (folder / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
            state = {'calls': 8, 'status': 'INCOMPLETE', 'pilot_protocol_hash': manifest['protocol_hash'],
                     'tasks': {task['task_id']: {'attempts': 1, 'status': 'BUDGET_EXHAUSTED'}}}
            (folder / 'state.json').write_text(json.dumps(state), encoding='utf-8')
            migrate_v4_model_run(folder, manifest)
            updated = json.loads((folder / 'manifest.json').read_text(encoding='utf-8'))
            saved_state = json.loads((folder / 'state.json').read_text(encoding='utf-8'))
            self.assertTrue((folder / 'manifest-v4.json').is_file())
            self.assertEqual(updated['protocol'], 'semantic-label-cf-v5')
            self.assertEqual(updated['model'], 'gemini-3.1-pro-preview')
            self.assertEqual(saved_state['calls'], 8)
            self.assertEqual(saved_state['tasks'][task['task_id']]['attempts'], 1)

    def test_v5_model_migration_preserves_quota_circuit_until_explicit_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            manifest = {'protocol': 'semantic-label-cf-v5', 'created': 'now',
                        'model': 'gemini-3.1-pro-preview', 'tasks': [], 'samples': []}
            manifest['protocol_hash'] = stable_digest({k: v for k, v in manifest.items()
                                                       if k not in ('created', 'protocol_hash')})
            (folder / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
            state = {'calls': 9, 'status': 'DEGRADED', 'pilot_protocol_hash': manifest['protocol_hash'],
                     'unavailable': {'gemini': 'quota exceeded'}, 'tasks': {}}
            (folder / 'state.json').write_text(json.dumps(state), encoding='utf-8')
            migrate_v5_model_run(folder, manifest)
            updated = json.loads((folder / 'manifest.json').read_text(encoding='utf-8'))
            saved_state = json.loads((folder / 'state.json').read_text(encoding='utf-8'))
            self.assertTrue((folder / 'manifest-v5.json').is_file())
            self.assertEqual(updated['protocol'], 'semantic-label-cf-v6')
            self.assertEqual(updated['model'], 'gemini-3.8-flash')
            self.assertEqual(saved_state['calls'], 9)
            self.assertIn('gemini', saved_state['unavailable'])

    def test_v6_model_migration_preserves_launches_and_task_attempts(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            task = {'task_id': 'image-16-named-1-api', 'condition': 'named'}
            manifest = {'protocol': 'semantic-label-cf-v6', 'created': 'now',
                        'model': 'gemini-3.8-flash', 'tasks': [task], 'samples': []}
            manifest['protocol_hash'] = stable_digest({k: v for k, v in manifest.items()
                                                       if k not in ('created', 'protocol_hash')})
            (folder / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
            state = {'calls': 16, 'status': 'INCOMPLETE', 'pilot_protocol_hash': manifest['protocol_hash'],
                     'tasks': {task['task_id']: {'attempts': 1, 'status': 'BUDGET_EXHAUSTED'}},
                     'unavailable': {}}
            (folder / 'state.json').write_text(json.dumps(state), encoding='utf-8')
            migrate_v6_model_run(folder, manifest)
            updated = json.loads((folder / 'manifest.json').read_text(encoding='utf-8'))
            saved_state = json.loads((folder / 'state.json').read_text(encoding='utf-8'))
            self.assertTrue((folder / 'manifest-v6.json').is_file())
            self.assertEqual(updated['protocol'], 'semantic-label-cf-v7')
            self.assertEqual(updated['model'], 'gemini-2.5-flash')
            self.assertEqual(saved_state['calls'], 16)
            self.assertEqual(saved_state['tasks'][task['task_id']]['attempts'], 1)

    def test_v7_model_migration_restores_gemini_38_flash(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            task = {'task_id': 'image-16-named-1-api', 'condition': 'named'}
            manifest = {'protocol': 'semantic-label-cf-v7', 'created': 'now',
                        'model': 'gemini-2.5-flash', 'tasks': [task], 'samples': []}
            manifest['protocol_hash'] = stable_digest({k: v for k, v in manifest.items()
                                                       if k not in ('created', 'protocol_hash')})
            (folder / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
            state = {'calls': 18, 'status': 'INCOMPLETE', 'pilot_protocol_hash': manifest['protocol_hash'],
                     'tasks': {task['task_id']: {'attempts': 3, 'status': 'BUDGET_EXHAUSTED'}},
                     'unavailable': {}}
            (folder / 'state.json').write_text(json.dumps(state), encoding='utf-8')
            migrate_v7_model_run(folder, manifest)
            updated = json.loads((folder / 'manifest.json').read_text(encoding='utf-8'))
            saved_state = json.loads((folder / 'state.json').read_text(encoding='utf-8'))
            self.assertTrue((folder / 'manifest-v7.json').is_file())
            self.assertEqual(updated['protocol'], 'semantic-label-cf-v8')
            self.assertEqual(updated['model'], 'gemini-3.8-flash')
            self.assertEqual(saved_state['calls'], 18)
            self.assertEqual(saved_state['tasks'][task['task_id']]['attempts'], 3)

    def test_v8_model_migration_records_requested_31_pro_and_preserves_ledger(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            task = {'task_id': 'image-16-named-2-api', 'condition': 'named'}
            manifest = {'protocol': 'semantic-label-cf-v8', 'created': 'now',
                        'model': 'gemini-3.8-flash', 'tasks': [task], 'samples': []}
            manifest['protocol_hash'] = stable_digest({k: v for k, v in manifest.items()
                                                       if k not in ('created', 'protocol_hash')})
            (folder / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
            state = {'calls': 21, 'status': 'INCOMPLETE', 'pilot_protocol_hash': manifest['protocol_hash'],
                     'tasks': {task['task_id']: {'attempts': 2, 'status': 'BUDGET_EXHAUSTED'}},
                     'unavailable': {}}
            (folder / 'state.json').write_text(json.dumps(state), encoding='utf-8')
            migrate_v8_model_run(folder, manifest)
            updated = json.loads((folder / 'manifest.json').read_text(encoding='utf-8'))
            saved_state = json.loads((folder / 'state.json').read_text(encoding='utf-8'))
            self.assertTrue((folder / 'manifest-v8.json').is_file())
            self.assertEqual(updated['protocol'], 'semantic-label-cf-v9')
            self.assertEqual(updated['model'], 'gemini-3.1-pro-preview')
            self.assertEqual(saved_state['calls'], 21)
            self.assertEqual(saved_state['tasks'][task['task_id']]['attempts'], 2)

    def test_v9_model_migration_restores_gemini_38_flash(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            task = {'task_id': 'image-16-named-2-api', 'condition': 'named'}
            manifest = {'protocol': 'semantic-label-cf-v9', 'created': 'now',
                        'model': 'gemini-3.1-pro-preview', 'tasks': [task], 'samples': []}
            manifest['protocol_hash'] = stable_digest({k: v for k, v in manifest.items()
                                                       if k not in ('created', 'protocol_hash')})
            (folder / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
            state = {'calls': 22, 'status': 'DEGRADED', 'pilot_protocol_hash': manifest['protocol_hash'],
                     'tasks': {task['task_id']: {'attempts': 3, 'status': 'FAILED'}},
                     'unavailable': {'gemini': 'quota exceeded'}}
            (folder / 'state.json').write_text(json.dumps(state), encoding='utf-8')
            migrate_v9_model_run(folder, manifest)
            updated = json.loads((folder / 'manifest.json').read_text(encoding='utf-8'))
            saved_state = json.loads((folder / 'state.json').read_text(encoding='utf-8'))
            self.assertTrue((folder / 'manifest-v9.json').is_file())
            self.assertEqual(updated['protocol'], 'semantic-label-cf-v10')
            self.assertEqual(updated['model'], 'gemini-3.8-flash')
            self.assertEqual(saved_state['calls'], 22)
            self.assertEqual(saved_state['tasks'][task['task_id']]['attempts'], 3)

    def test_claude_command_is_read_only_and_enforces_budget(self):
        command = claude_command(1.25)
        self.assertIn('--max-budget-usd', command)
        self.assertIn('1.25', command)
        self.assertIn('--tools', command)
        self.assertIn('Read', command)
        self.assertIn('--allowedTools', command)
        self.assertNotIn('Bash', command)

    def test_claude_skips_launch_when_remaining_budget_is_below_observed_call_floor(self):
        self.assertTrue(claude_budget_allows_call(0.0, 2.0))
        self.assertTrue(claude_budget_allows_call(0.98, 2.0))
        self.assertTrue(claude_budget_allows_call(1.0, 2.0))
        self.assertFalse(claude_budget_allows_call(1.01, 2.0))
        self.assertFalse(claude_budget_allows_call(1.99, 2.0))

    def test_v10_claude_migration_preserves_gemini_history_and_run_cap(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            task = {'task_id': 'image-16-named-2-api', 'condition': 'named',
                    'image_path': 'images/image-16.jpg'}
            manifest = {'protocol': 'semantic-label-cf-v10', 'created': 'now',
                        'model': 'gemini-3.8-flash', 'transport': 'Gemini API',
                        'tasks': [task], 'samples': [], 'calls_max': 80}
            manifest['protocol_hash'] = stable_digest({k: v for k, v in manifest.items()
                                                       if k not in ('created', 'protocol_hash')})
            (folder / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
            state = {'calls': 23, 'status': 'DEGRADED', 'pilot_protocol_hash': manifest['protocol_hash'],
                     'unavailable': {'gemini': 'high demand'}, 'tasks': {
                         task['task_id']: {'attempts': 2, 'status': 'BUDGET_EXHAUSTED'}},
                     'member_stats': {'gemini': {'calls': 23, 'attempts': 23, 'seconds': 900,
                                                 'cost_usd': 0, 'tasks': 0},
                                      'claude': {'calls': 0, 'attempts': 0, 'seconds': 0,
                                                 'cost_usd': 0, 'tasks': 0}}}
            (folder / 'state.json').write_text(json.dumps(state), encoding='utf-8')
            migrate_v10_claude_run(folder, manifest, model='claude-fable-5-1',
                                   endpoint='gpt.teamsoclo.site')
            updated = json.loads((folder / 'manifest.json').read_text(encoding='utf-8'))
            saved_state = json.loads((folder / 'state.json').read_text(encoding='utf-8'))
            self.assertTrue((folder / 'manifest-v10.json').is_file())
            self.assertEqual(updated['protocol'], 'semantic-label-cf-v11')
            self.assertEqual(updated['provider'], 'claude-code')
            self.assertEqual(updated['model'], 'claude-fable-5-1')
            self.assertEqual(updated['tasks'][0]['task_id'], task['task_id'] + '-claude')
            self.assertEqual(saved_state['calls'], 23)
            self.assertEqual(saved_state['tasks'][task['task_id']]['attempts'], 2)
            self.assertEqual(saved_state['member_stats']['claude']['calls'], 0)
            self.assertEqual(updated['calls_max'], 80)

    def test_v11_label_migration_supersedes_bad_named_result_but_keeps_generic_result(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            samples = [{'species': 'giraffe', 'animal_group': 'mammal', 'ground_truth': '5',
                        'expected_bias': '4', 'local_path': 'images/image-01.jpg',
                        'image_sha256': 'image-hash'}]
            canonical = build_tasks(samples)
            active = [dict(task, task_id=f"{task['task_id']}-api-claude") for task in canonical]
            named = next(t for t in active if t['condition'] == 'named' and t['replicate'] == 1)
            generic = next(t for t in active if t['condition'] == 'generic' and t['replicate'] == 1)
            named['species'] = 'giraffe 3 2 384'
            manifest = {'protocol': 'semantic-label-cf-v11', 'created': 'now', 'provider': 'claude',
                        'model': 'claude-fable-5-1', 'endpoint_host': 'proxy.example',
                        'tasks': active, 'samples': samples, 'calls_max': 80, 'claude_usd_cap': 2}
            manifest['protocol_hash'] = stable_digest({k: v for k, v in manifest.items()
                                                       if k not in ('created', 'protocol_hash')})
            (folder / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
            state = {'calls': 27, 'status': 'DEGRADED', 'pilot_protocol_hash': manifest['protocol_hash'],
                     'unavailable': {'claude': 'max-budget'}, 'tasks': {
                         named['task_id']: {'attempts': 1, 'status': 'DONE'},
                         generic['task_id']: {'attempts': 1, 'status': 'DONE'}},
                     'member_stats': {'claude': {'calls': 4, 'attempts': 4, 'seconds': 100,
                                                 'cost_usd': 2.3534, 'tasks': 3}}}
            (folder / 'state.json').write_text(json.dumps(state), encoding='utf-8')
            (folder / 'reports').mkdir()
            (folder / 'reports' / f"{named['task_id']}.md").write_text('{"response":"4"}', encoding='utf-8')
            (folder / 'reports' / f"{generic['task_id']}.md").write_text('{"response":"4"}', encoding='utf-8')
            migrate_v11_label_run(folder, manifest)
            updated = json.loads((folder / 'manifest.json').read_text(encoding='utf-8'))
            saved_state = json.loads((folder / 'state.json').read_text(encoding='utf-8'))
            corrected_named = next(t for t in updated['tasks']
                                   if t['condition'] == 'named' and t['replicate'] == 1)
            kept_generic = next(t for t in updated['tasks']
                                if t['condition'] == 'generic' and t['replicate'] == 1)
            self.assertEqual(updated['protocol'], 'semantic-label-cf-v12')
            self.assertEqual(corrected_named['species'], 'giraffe')
            self.assertTrue(corrected_named['task_id'].endswith('-labels-fixed'))
            self.assertEqual(kept_generic['species'], 'giraffe')
            self.assertEqual(kept_generic['task_id'], generic['task_id'])
            self.assertEqual(saved_state['tasks'][named['task_id']]['status'], 'SUPERSEDED')
            self.assertEqual(saved_state['tasks'][generic['task_id']]['status'], 'DONE')
            self.assertTrue((folder / 'reports' / f"{named['task_id']}.md").is_file())
            self.assertEqual(saved_state['calls'], 27)

    def test_claude_command_is_read_only_and_enforces_budget(self):
        command = claude_command(1.25)
        self.assertIn('--max-budget-usd', command)
        self.assertIn('1.25', command)
        self.assertIn('--tools', command)
        self.assertIn('Read', command)
        self.assertIn('--allowedTools', command)
        self.assertNotIn('Bash', command)

    def test_v10_claude_migration_preserves_gemini_history_and_run_cap(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            task = {'task_id': 'image-16-named-2-api', 'condition': 'named',
                    'image_path': 'images/image-16.jpg'}
            manifest = {'protocol': 'semantic-label-cf-v10', 'created': 'now',
                        'model': 'gemini-3.8-flash', 'transport': 'Gemini API',
                        'tasks': [task], 'samples': [], 'calls_max': 80}
            manifest['protocol_hash'] = stable_digest({k: v for k, v in manifest.items()
                                                       if k not in ('created', 'protocol_hash')})
            (folder / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
            state = {'calls': 23, 'status': 'DEGRADED', 'pilot_protocol_hash': manifest['protocol_hash'],
                     'unavailable': {'gemini': 'high demand'}, 'tasks': {
                         task['task_id']: {'attempts': 2, 'status': 'BUDGET_EXHAUSTED'}},
                     'member_stats': {'gemini': {'calls': 23, 'attempts': 23, 'seconds': 900,
                                                 'cost_usd': 0, 'tasks': 0},
                                      'claude': {'calls': 0, 'attempts': 0, 'seconds': 0,
                                                 'cost_usd': 0, 'tasks': 0}}}
            (folder / 'state.json').write_text(json.dumps(state), encoding='utf-8')
            migrate_v10_claude_run(folder, manifest, model='claude-fable-5-1',
                                   endpoint='gpt.teamsoclo.site')
            updated = json.loads((folder / 'manifest.json').read_text(encoding='utf-8'))
            saved_state = json.loads((folder / 'state.json').read_text(encoding='utf-8'))
            self.assertTrue((folder / 'manifest-v10.json').is_file())
            self.assertEqual(updated['protocol'], 'semantic-label-cf-v11')
            self.assertEqual(updated['provider'], 'claude')
            self.assertEqual(updated['model'], 'claude-fable-5-1')
            self.assertEqual(updated['tasks'][0]['task_id'], task['task_id'] + '-claude')
            self.assertEqual(saved_state['calls'], 23)
            self.assertEqual(saved_state['tasks'][task['task_id']]['attempts'], 2)
            self.assertEqual(saved_state['member_stats']['claude']['calls'], 0)
            self.assertEqual(updated['calls_max'], 80)


if __name__ == '__main__':
    unittest.main()
