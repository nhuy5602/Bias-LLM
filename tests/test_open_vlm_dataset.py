import json
import sys
import tempfile
import unittest
import urllib.error
from io import BytesIO
from pathlib import Path

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
        self.assertEqual([row['family'] for row in selected],
                         ['animals:buffalo', 'animals:camel'])
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


if __name__ == '__main__':
    unittest.main()
