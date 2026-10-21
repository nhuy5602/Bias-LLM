import json
import sys
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))

from open_vlm.schema import (  # noqa: E402
    DATASET_REVISION,
    DomainSpec,
    build_manifest,
    split_families,
    validate_row,
)


class OpenVlmSchemaTests(unittest.TestCase):
    def integer_domain(self):
        return DomainSpec(
            name='animals',
            answer_kind='integer',
            candidate_values=tuple(range(21)),
            prior_fields=('expected_bias',),
        )

    def boolean_domain(self):
        return DomainSpec(
            name='illusions',
            answer_kind='categorical',
            candidate_values=('yes', 'no'),
            prior_fields=('expected_bias',),
        )

    def test_domain_spec_is_immutable(self):
        domain = self.integer_domain()
        with self.assertRaises(FrozenInstanceError):
            domain.name = 'logos'

    def test_integer_row_is_normalized_without_inference(self):
        row = validate_row({
            'domain': 'animals',
            'family': 'animal-01',
            'ground_truth': 4,
            'expected_bias': 5,
            'image_path': 'images/animal-01.png',
        }, self.integer_domain())
        self.assertEqual(row['domain'], 'animals')
        self.assertEqual(row['ground_truth'], 4)
        self.assertEqual(row['prior_answer'], 5)
        self.assertEqual(row['answer_kind'], 'integer')

    def test_categorical_row_is_case_normalized(self):
        row = validate_row({
            'domain': 'illusions',
            'family': 'illusion-01',
            'ground_truth': 'yes',
            'expected_bias': 'NO',
            'image_path': 'images/illusion-01.png',
        }, self.boolean_domain())
        self.assertEqual(row['ground_truth'], 'yes')
        self.assertEqual(row['prior_answer'], 'no')
        self.assertEqual(row['candidate_values'], ['yes', 'no'])

    def test_missing_prior_field_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'prior answer'):
            validate_row({
                'domain': 'animals',
                'family': 'animal-01',
                'ground_truth': 4,
                'image_path': 'images/animal-01.png',
            }, self.integer_domain())

    def test_ambiguous_or_out_of_range_answer_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'candidate'):
            validate_row({
                'domain': 'animals',
                'family': 'animal-01',
                'ground_truth': 21,
                'expected_bias': 5,
                'image_path': 'images/animal-01.png',
            }, self.integer_domain())

    def test_split_is_family_disjoint_and_deterministic(self):
        rows = [
            {'domain': 'animals', 'family': f'family-{index:02}', 'row_id': index}
            for index in range(10)
        ]
        first = split_families(rows, seed=17)
        second = split_families(rows, seed=17)
        self.assertEqual(first, second)
        self.assertEqual(len(first['dev']), 8)
        self.assertEqual(len(first['same_domain_test']), 2)
        self.assertTrue({r['family'] for r in first['dev']}.isdisjoint(
            {r['family'] for r in first['same_domain_test']}))

    def test_manifest_records_revision_licenses_and_minimum_family_status(self):
        rows = [
            {
                'domain': 'animals',
                'family': f'family-{index:02}',
                'ground_truth': 4,
                'expected_bias': 5,
                'image_path': f'images/{index:02}.png',
            }
            for index in range(4)
        ]
        config = {
            'dataset': 'anvo25/vlms-are-biased',
            'dataset_revision': DATASET_REVISION,
            'license_attestations': {'dataset': 'verified', 'model': 'verified'},
            'seed': 20261022,
            'processor': {'min_pixels': 256, 'max_pixels': 512},
            'model': {'id': 'Qwen/Qwen2.5-VL-3B-Instruct', 'revision': 'main'},
            'runtime': {'device': 'cuda', 'dtype': 'bfloat16', 'attention': 'sdpa'},
            'method': {'sigmas': [1, 3, 6, 12], 'alphas': [0.25, 0.5, 1.0, 2.0]},
        }
        manifest = build_manifest(rows, config)
        self.assertEqual(manifest['dataset_revision'], DATASET_REVISION)
        self.assertEqual(manifest['scientific_status'], 'NEEDS_EVIDENCE')
        self.assertEqual(manifest['model']['id'], 'Qwen/Qwen2.5-VL-3B-Instruct')
        self.assertEqual(manifest['runtime']['attention'], 'sdpa')
        self.assertFalse(manifest['family_split']['animals']['eligible'])
        self.assertEqual(manifest['family_split']['animals']['family_count'], 4)
        serialized = json.dumps(manifest, sort_keys=True, ensure_ascii=True)
        self.assertEqual(serialized, json.dumps(manifest, sort_keys=True, ensure_ascii=True))

    def test_manifest_rejects_unverified_license(self):
        with self.assertRaisesRegex(ValueError, 'license'):
            build_manifest([], {
                'dataset_revision': DATASET_REVISION,
                'license_attestations': {'dataset': 'pending', 'model': 'verified'},
            })


if __name__ == '__main__':
    unittest.main()
