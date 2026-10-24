import json
import sys
import tempfile
import unittest
from io import BytesIO
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))

from open_vlm.dataset import (  # noqa: E402
    BASE_ROWS_URL,
    DATASET_REVISION,
    fetch_rows,
    load_vlmbias_rows,
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

    def test_load_rows_filters_domains_and_limits_per_domain(self):
        rows = [self.row('Animals'), self.row('Logos')]
        rows[1]['ID'] = 'row-2'

        def opener(_request, timeout):
            return Response(json.dumps({'rows': [{'row': row} for row in rows]}).encode('utf-8'))

        selected = load_vlmbias_rows(('animals',), max_per_domain=1, page_limit=1, opener=opener)
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]['domain'], 'animals')

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


if __name__ == '__main__':
    unittest.main()
