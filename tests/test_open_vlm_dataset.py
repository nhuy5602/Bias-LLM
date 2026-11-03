import json
import sys
import tempfile
import unittest
import urllib.error
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))

from open_vlm.dataset import (  # noqa: E402
    BASE_ROWS_URL,
    DATASET_REVISION,
    fetch_rows,
    load_vlmbias_rows,
    load_vlmbias_calibration_rows,
    normalize_calibration_row,
    normalize_row,
    prepare_vlmbias_run,
    source_family,
)


class Response:
    def __init__(self, data):
        self.data = data

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return self.data


class OpenVlmDatasetTests(unittest.TestCase):
    def row(self, domain='Animals'):
        return {
            'ID': 'row-1',
            'topic': domain,
            'image_path': 'animals/cat_384.png',
            'ground_truth': 4,
            'expected_bias': 5,
            'image': {'src': 'https://example.test/cat.png'},
            'metadata': json.dumps({'group_id': 'family-1'}),
        }

    def test_normalize_row_records_domain_family_and_prior(self):
        normalized = normalize_row(self.row())
        self.assertEqual(normalized['domain'], 'animals')
        self.assertEqual(normalized['family'], 'animals:family-1')
        self.assertEqual(normalized['expected_bias'], 5)
        self.assertEqual(normalized['image_url'], 'https://example.test/cat.png')

    def test_normalize_row_infers_subject_label_from_source_metadata(self):
        animal = normalize_row({
            'ID': 'animal_006_buffalo_notitle_Q1_px1152',
            'topic': 'Animals',
            'image_path': 'animals/buffalo_2_3_1152.png',
            'ground_truth': 5,
            'expected_bias': 4,
            'image': {'src': 'https://example.test/buffalo.png'},
            'metadata': '{}',
        })
        logo = normalize_row({
            'ID': 'shoe_0010_notitle_Q1_px1152',
            'topic': 'Logos',
            'image_path': 'logos/adidas-black-basketball-5_1152.png',
            'ground_truth': 4,
            'expected_bias': 3,
            'image': {'src': 'https://example.test/adidas.png'},
            'metadata': json.dumps({'bias_type': 'adidas'}),
        })
        self.assertEqual(animal['subject_label'], 'buffalo')
        self.assertEqual(logo['subject_label'], 'adidas')

    def test_source_family_falls_back_to_resolution_stripped_path(self):
        row = self.row()
        row['metadata'] = '{}'
        row['image_path'] = 'animals/cat_384.png'
        self.assertEqual(source_family(row, 'animals'), 'animals:cat')

    def test_fetch_rows_pins_revision_in_url(self):
        urls = []

        def opener(request, timeout):
            urls.append(request.full_url)
            return Response(json.dumps({'rows': []}).encode('utf-8'))

        fetch_rows(offsets=(0,), opener=opener)
        self.assertEqual(len(urls), 1)
        self.assertIn(DATASET_REVISION, urls[0])
        self.assertTrue(urls[0].startswith(BASE_ROWS_URL))

    def test_fetch_rows_retries_transient_gateway_errors(self):
        attempts = []

        def opener(request, timeout):
            attempts.append(request.full_url)
            if len(attempts) < 3:
                raise urllib.error.HTTPError(request.full_url, 502, 'bad gateway', {}, None)
            return Response(json.dumps({'rows': []}).encode('utf-8'))

        fetch_rows(offsets=(0,), opener=opener, retries=3, backoff_seconds=0)
        self.assertEqual(len(attempts), 3)

    def test_fetch_rows_honors_retry_after_for_rate_limits(self):
        attempts = []

        def opener(request, timeout):
            attempts.append(request.full_url)
            if len(attempts) == 1:
                raise urllib.error.HTTPError(
                    request.full_url, 429, 'too many requests', {'Retry-After': '7'}, None)
            return Response(json.dumps({'rows': []}).encode('utf-8'))

        with patch('open_vlm.dataset.time.sleep') as sleep:
            fetch_rows(offsets=(0,), opener=opener, retries=1, backoff_seconds=1)

        self.assertEqual(len(attempts), 2)
        sleep.assert_called_once_with(7.0)

    def test_fetch_rows_does_not_retry_not_found(self):
        attempts = []

        def opener(request, timeout):
            attempts.append(request.full_url)
            raise urllib.error.HTTPError(request.full_url, 404, 'not found', {}, None)

        with self.assertRaises(urllib.error.HTTPError):
            fetch_rows(offsets=(0,), opener=opener, retries=3, backoff_seconds=0)
        self.assertEqual(len(attempts), 1)

    def test_load_rows_filters_domains_and_limits_per_domain(self):
        rows = [self.row('Animals'), self.row('Logos')]
        rows[1]['ID'] = 'row-2'

        def opener(_request, timeout):
            return Response(json.dumps({'rows': [{'row': row} for row in rows]}).encode('utf-8'))

        selected = load_vlmbias_rows(('animals',), max_per_domain=1, page_limit=1, opener=opener)
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]['domain'], 'animals')

    def test_load_rows_limits_distinct_source_families(self):
        rows = [self.row('Animals'), self.row('Animals')]
        rows[1]['ID'] = 'row-2'
        rows[1]['image_path'] = 'animals/dog_384.png'
        rows[1]['metadata'] = json.dumps({'group_id': 'family-2'})

        def opener(_request, timeout):
            return Response(json.dumps({'rows': [{'row': row} for row in rows]}).encode('utf-8'))

        selected = load_vlmbias_rows(('animals',), max_per_domain=2, page_limit=1, opener=opener)
        self.assertEqual({row['family'] for row in selected}, {'animals:family-1', 'animals:family-2'})

    def test_load_rows_fetches_until_requested_unique_family_count(self):
        duplicate_rows = [self.row('Animals'), self.row('Animals')]
        duplicate_rows[1]['ID'] = 'row-2'
        duplicate_rows[1]['image_path'] = 'animals/cat_768.png'
        additional = self.row('Animals')
        additional.update({
            'ID': 'row-3',
            'image_path': 'animals/dog_384.png',
            'metadata': json.dumps({'group_id': 'family-2'}),
        })
        page_calls = 0

        def opener(_request, timeout):
            nonlocal page_calls
            page_calls += 1
            page_rows = duplicate_rows if page_calls == 1 else [additional]
            return Response(json.dumps({'rows': [{'row': row} for row in page_rows]}).encode('utf-8'))

        selected = load_vlmbias_rows(
            ('animals',), max_per_domain=2, page_limit=2, opener=opener)
        self.assertEqual({row['family'] for row in selected}, {'animals:family-1', 'animals:family-2'})
        self.assertEqual(page_calls, 2)

    def test_load_calibration_rows_uses_original_split_and_distinct_families(self):
        rows = []
        for family in ('elephant', 'buffalo', 'camel'):
            rows.append({
                'ID': f'{family}_prompt1',
                'topic': 'Animals',
                'type_of_question': 'Counting/Illusion',
                'image_path': f'original_images/{family}.png',
                'ground_truth': 4,
                'expected_bias': None,
                'image': {'src': f'https://example.test/{family}.png'},
                'metadata': '{}',
            })

        def opener(request, timeout):
            self.assertIn('split=original', request.full_url)
            return Response(json.dumps({'rows': [{'row': row} for row in rows]}).encode('utf-8'))

        selected = load_vlmbias_calibration_rows(('animals',), max_per_domain=2,
                                                 page_limit=1, opener=opener)
        self.assertEqual(len(selected), 2)
        self.assertEqual(len({row['family'] for row in selected}), 2)
        self.assertTrue({row['family'] for row in selected} <= {
            'animals:elephant', 'animals:buffalo', 'animals:camel',
        })
        self.assertTrue(all(row['source_split'] == 'original' for row in selected))
        self.assertTrue(all(row['expected_bias'] is None for row in selected))

        logo = normalize_calibration_row({
            'ID': 'audi_prompt1',
            'topic': 'Logos',
            'type_of_question': 'Counting/Illusion',
            'image_path': 'original_images/images/audi_black_coupe_0.png',
            'ground_truth': 4,
            'expected_bias': None,
            'image': {'src': 'https://example.test/audi.png'},
            'metadata': '{}',
        })
        self.assertEqual(logo['family'], 'logos:audi-black-coupe')

    def test_calibration_sample_balances_animal_counts_and_logo_brands(self):
        rows = []
        for index in range(12):
            rows.append({
                'ID': f'animal_{index:03d}_species{index}_prompt1',
                'topic': 'Animals',
                'type_of_question': 'Counting/Illusion',
                'image_path': f'original_images/animals/species_{index}.png',
                'ground_truth': 2 if index < 6 else 4,
                'expected_bias': None,
                'image': {'src': f'https://example.test/animal-{index}.png'},
                'metadata': '{}',
            })
        for brand in ('audi', 'maserati', 'mercedes'):
            for index in range(4):
                rows.append({
                    'ID': f'{brand}_{index}_prompt1',
                    'topic': 'Logos',
                    'type_of_question': 'Counting/Illusion',
                    'image_path': f'original_images/logos/{brand}_black_body-{index}.png',
                    'ground_truth': 4,
                    'expected_bias': None,
                    'image': {'src': f'https://example.test/{brand}-{index}.png'},
                    'metadata': json.dumps({
                        'bias_type': brand, 'color': 'black', 'body_type': f'body-{index}',
                    }),
                })

        def opener(_request, timeout):
            return Response(json.dumps({'rows': [{'row': row} for row in rows]}).encode('utf-8'))

        selected = load_vlmbias_calibration_rows(
            ('animals', 'logos'), max_per_domain=6, page_limit=1, opener=opener, seed=17)
        repeated = load_vlmbias_calibration_rows(
            ('animals', 'logos'), max_per_domain=6, page_limit=1, opener=opener, seed=17)
        animals = [row for row in selected if row['domain'] == 'animals']
        logos = [row for row in selected if row['domain'] == 'logos']
        self.assertEqual([row['family'] for row in selected],
                         [row['family'] for row in repeated])
        self.assertEqual({count: sum(row['ground_truth'] == count for row in animals)
                          for count in (2, 4)}, {2: 3, 4: 3})
        self.assertEqual({brand: sum(row['metadata']['bias_type'] == brand for row in logos)
                          for brand in ('audi', 'maserati', 'mercedes')},
                         {'audi': 2, 'maserati': 2, 'mercedes': 2})

    def test_calibration_loader_does_not_count_duplicate_families_toward_limit(self):
        duplicate = {
            'ID': 'animal_001_buffalo_prompt1',
            'topic': 'Animals',
            'type_of_question': 'Counting/Illusion',
            'image_path': 'original_images/buffalo.png',
            'ground_truth': 4,
            'expected_bias': None,
            'image': {'src': 'https://example.test/buffalo.png'},
            'metadata': '{}',
        }
        additional = [
            {**duplicate, 'ID': 'animal_002_bison_prompt1',
             'image_path': 'original_images/bison.png',
             'image': {'src': 'https://example.test/bison.png'}},
            {**duplicate, 'ID': 'animal_003_bird_prompt1',
             'image_path': 'original_images/bird.png', 'ground_truth': 2,
             'image': {'src': 'https://example.test/bird.png'}},
        ]
        page_calls = 0

        def opener(_request, timeout):
            nonlocal page_calls
            page_calls += 1
            page_rows = [duplicate] * 100 if page_calls == 1 else additional
            return Response(json.dumps({'rows': [{'row': row} for row in page_rows]}).encode('utf-8'))

        selected = load_vlmbias_calibration_rows(
            ('animals',), max_per_domain=2, page_limit=2, opener=opener)
        self.assertEqual(len({row['family'] for row in selected}), 2)
        self.assertEqual(page_calls, 2)

    def test_calibration_loader_excludes_alpha_development_families(self):
        rows = []
        for family, count in (
                ('elk', 4), ('warthog', 4), ('buffalo', 4), ('bison', 4),
                ('bird-one', 2), ('bird-two', 2), ('bird-three', 2), ('bird-four', 2)):
            rows.append({
                'ID': f'{family}_prompt1',
                'topic': 'Animals',
                'type_of_question': 'Counting/Illusion',
                'image_path': f'original_images/{family}.png',
                'ground_truth': count,
                'expected_bias': None,
                'image': {'src': f'https://example.test/{family}.png'},
                'metadata': '{}',
            })

        def opener(_request, timeout):
            return Response(json.dumps({'rows': [{'row': row} for row in rows]}).encode('utf-8'))

        selected = load_vlmbias_calibration_rows(
            ('animals',), max_per_domain=4, page_limit=1, opener=opener,
            excluded_families={'animals:elk', 'animals:warthog'}, seed=9)
        counts = {count: sum(row['ground_truth'] == count for row in selected)
                  for count in (2, 4)}
        self.assertEqual(counts, {2: 2, 4: 2})
        self.assertFalse({'animals:elk', 'animals:warthog'} &
                         {row['family'] for row in selected})

    def test_prepare_run_downloads_images_and_preserves_normalized_contract(self):
        image_buffer = BytesIO()
        Image.new('RGB', (8, 8), (20, 30, 40)).save(image_buffer, format='PNG')
        image_bytes = image_buffer.getvalue()
        rows = [self.row('Animals')]

        def opener(request, timeout):
            if request.full_url.startswith(BASE_ROWS_URL):
                return Response(json.dumps({'rows': [{'row': rows[0]}]}).encode('utf-8'))
            return Response(image_bytes)

        with tempfile.TemporaryDirectory() as directory:
            run_dir = prepare_vlmbias_run(
                Path(directory), 'development', ['animals'], {
                    'dataset_revision': DATASET_REVISION,
                    'license_attestations': {'dataset': 'pending', 'model': 'pending'},
                    'seed': 7,
                }, max_per_domain=1, opener=opener)
            manifest = json.loads((run_dir / 'manifest.json').read_text(encoding='utf-8'))
            self.assertTrue(manifest['execution_blocked'])
            self.assertEqual(manifest['rows'][0]['candidate_values'], list(range(21)))
            self.assertTrue((run_dir / manifest['rows'][0]['image_path']).is_file())

    def test_prepare_run_can_attach_original_calibration_rows(self):
        image_buffer = BytesIO()
        Image.new('RGB', (8, 8), (20, 30, 40)).save(image_buffer, format='PNG')
        image_bytes = image_buffer.getvalue()
        main_row = self.row('Animals')
        original_row = {
            'ID': 'elephant_prompt1',
            'topic': 'Animals',
            'type_of_question': 'Counting/Illusion',
            'image_path': 'original_images/elephant.png',
            'ground_truth': 4,
            'expected_bias': None,
            'image': {'src': 'https://example.test/elephant.png'},
            'metadata': '{}',
        }

        def opener(request, timeout):
            if 'split=original' in request.full_url:
                return Response(json.dumps({'rows': [{'row': original_row}]}).encode('utf-8'))
            if request.full_url.startswith(BASE_ROWS_URL):
                return Response(json.dumps({'rows': [{'row': main_row}]}).encode('utf-8'))
            return Response(image_bytes)

        with tempfile.TemporaryDirectory() as directory:
            run_dir = prepare_vlmbias_run(
                Path(directory), 'development', ['animals'], {
                    'dataset_revision': DATASET_REVISION,
                    'license_attestations': {'dataset': 'pending', 'model': 'pending'},
                    'seed': 7,
                }, max_per_domain=1, calibration_max_per_domain=1, opener=opener)
            manifest = json.loads((run_dir / 'manifest.json').read_text(encoding='utf-8'))
            self.assertTrue(manifest['step0_calibration'])
            self.assertEqual(manifest['step0_rows'][0]['source_split'], 'original')
            self.assertTrue((run_dir / manifest['step0_rows'][0]['image_path']).is_file())

    def test_prepare_run_registers_disjoint_counterfactual_bias_calibration_rows(self):
        image_buffer = BytesIO()
        Image.new('RGB', (8, 8), (20, 30, 40)).save(image_buffer, format='PNG')
        image_bytes = image_buffer.getvalue()
        main_rows = []
        for index in range(20):
            row = self.row('Animals')
            row.update({
                'ID': f'counterfactual-{index}',
                'image_path': f'animals/family-{index}_384.png',
                'metadata': json.dumps({'group_id': f'family-{index}'}),
                'ground_truth': 5,
                'expected_bias': 4,
            })
            main_rows.append(row)
        original_rows = [{
            'ID': f'family-{index}_prompt1',
            'topic': 'Animals',
            'type_of_question': 'Counting/Illusion',
            'image_path': f'original_images/family-{index}.png',
            'ground_truth': 2 if index % 2 == 0 else 4,
            'expected_bias': None,
            'image': {'src': f'https://example.test/family-{index}.png'},
            'metadata': '{}',
        } for index in range(20)]

        def opener(request, timeout):
            if 'split=original' in request.full_url:
                response_rows = original_rows
            elif request.full_url.startswith(BASE_ROWS_URL):
                response_rows = main_rows
            else:
                return Response(image_bytes)
            return Response(json.dumps({'rows': [{'row': row} for row in response_rows]}).encode('utf-8'))

        with tempfile.TemporaryDirectory() as directory:
            run_dir = prepare_vlmbias_run(
                Path(directory), 'development', ['animals'], {
                    'dataset_revision': DATASET_REVISION,
                    'license_attestations': {'dataset': 'pending', 'model': 'pending'},
                    'seed': 7,
                }, max_per_domain=20, calibration_max_per_domain=5, opener=opener)
            manifest = json.loads((run_dir / 'manifest.json').read_text(encoding='utf-8'))
            bias_paths_exist = all((run_dir / row['image_path']).is_file()
                                   for row in manifest['step0_bias_rows'])

        split = manifest['family_split']['animals']
        dev_families = set(split['dev_families'])
        test_families = set(split['same_domain_test_families'])
        bias_rows = manifest['step0_bias_rows']
        bias_families = {row['family'] for row in bias_rows}
        self.assertEqual(len(bias_families), 5)
        self.assertEqual(bias_families, test_families)
        self.assertFalse(bias_families & dev_families)
        original_families = {row['family'] for row in manifest['step0_rows']}
        self.assertFalse(original_families & dev_families)
        self.assertEqual(manifest['step0_excluded_families'], sorted(dev_families))
        self.assertTrue(all(row['source_split'] == 'main' for row in bias_rows))
        self.assertTrue(all(row['expected_bias'] is not None for row in bias_rows))
        self.assertTrue(bias_paths_exist)


if __name__ == '__main__':
    unittest.main()
