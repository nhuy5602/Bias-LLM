import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))

from open_vlm.diagnostics import (  # noqa: E402
    ALPHAS,
    SIGMAS,
    build_negative_controls,
    select_blur_sigma,
    validate_roi,
)


class OpenVlmDiagnosticsTests(unittest.TestCase):
    def test_sigma_constants_are_preregistered(self):
        self.assertEqual(SIGMAS, (1, 3, 6, 12))
        self.assertEqual(ALPHAS, (0.25, 0.5, 1.0, 2.0))

    def test_select_blur_sigma_uses_smallest_qualifying_value(self):
        records = [
            {'sigma': 1, 'subject_valid_rate': 1.0, 'invalid_rate': 0.01,
             'clear_accuracy': 0.90, 'blur_accuracy': 0.86},
            {'sigma': 3, 'subject_valid_rate': 0.95, 'invalid_rate': 0.01,
             'clear_accuracy': 0.90, 'blur_accuracy': 0.75},
            {'sigma': 6, 'subject_valid_rate': 0.99, 'invalid_rate': 0.01,
             'clear_accuracy': 0.90, 'blur_accuracy': 0.60},
        ]
        result = select_blur_sigma(records)
        self.assertEqual(result['status'], 'PASS')
        self.assertEqual(result['sigma'], 3)

    def test_sigma_selection_rejects_no_qualifying_condition(self):
        records = [{
            'sigma': sigma,
            'subject_valid_rate': 0.80,
            'invalid_rate': 0.10,
            'clear_accuracy': 0.90,
            'blur_accuracy': 0.50,
        } for sigma in SIGMAS]
        result = select_blur_sigma(records)
        self.assertEqual(result['status'], 'UNSUPPORTED')
        self.assertIsNone(result['sigma'])

    def test_negative_controls_are_deterministic_and_size_matched(self):
        task = {'task_id': 'family-1', 'size': [64, 48], 'prior_image_path': 'prior.png'}
        first = build_negative_controls(task, seed=11)
        second = build_negative_controls(task, seed=11)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 23)
        self.assertEqual({control['kind'] for control in first}, {
            'noise', 'shuffled_prior', 'empty_black', 'random_alpha',
        })
        self.assertTrue(all(control['size'] == [64, 48] for control in first))
        self.assertEqual(sum(control['kind'] == 'random_alpha' for control in first), 20)
        self.assertTrue(all(control['alpha'] in ALPHAS for control in first if control['kind'] == 'random_alpha'))

    def test_roi_rejects_count_bearing_prompt(self):
        with self.assertRaisesRegex(ValueError, 'neutral'):
            validate_roi(
                {'bbox': [1, 2, 20, 20], 'prompt': 'Count the logo.'},
                {'family': 'logo-1', 'image_size': [64, 48]},
                'reference_roi',
            )

    def test_roi_labels_reference_and_automatic_without_ground_truth_claim(self):
        row = {'family': 'logo-1', 'image_size': [64, 48]}
        reference = validate_roi(
            {'bbox': [1, 2, 20, 20], 'prompt': 'Locate the logo.', 'source': 'annotator'},
            row,
            'reference_roi',
        )
        automatic = validate_roi(
            {'bbox': [1, 2, 20, 20], 'prompt': 'Locate the logo.', 'source': 'qwen'},
            row,
            'automatic_roi',
        )
        self.assertEqual(reference['kind'], 'reference_roi')
        self.assertEqual(automatic['kind'], 'automatic_roi')
        self.assertFalse(reference['ground_truth'])
        self.assertFalse(automatic['ground_truth'])

    def test_oracle_roi_requires_verified_dataset_geometry(self):
        with self.assertRaisesRegex(ValueError, 'ground-truth'):
            validate_roi(
                {'bbox': [1, 2, 20, 20], 'prompt': 'Locate the logo.'},
                {'family': 'logo-1', 'image_size': [64, 48]},
                'oracle_roi',
            )
        result = validate_roi(
            {'bbox': [1, 2, 20, 20], 'prompt': 'Locate the logo.', 'source': 'dataset_bbox'},
            {'family': 'logo-1', 'image_size': [64, 48], 'dataset_bbox_verified': True},
            'oracle_roi',
        )
        self.assertTrue(result['ground_truth'])


if __name__ == '__main__':
    unittest.main()
