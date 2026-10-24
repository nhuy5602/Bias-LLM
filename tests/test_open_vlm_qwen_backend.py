import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))

from open_vlm.qwen_backend import QwenBackend  # noqa: E402


class FakeQwenModel:
    def __init__(self):
        self.generated = []

    def generate(self, image, prompt):
        self.generated.append((image, prompt))
        return '4'

    def score_image(self, image, candidate):
        values = image['scores']
        value = values[candidate]
        return value if isinstance(value, list) else [value]

    def score_image_batch(self, image, candidates):
        return {candidate: self.score_image(image, candidate) for candidate in candidates}


class OpenVlmQwenBackendTests(unittest.TestCase):
    def backend(self):
        return QwenBackend(
            model=FakeQwenModel(),
            processor=None,
            manifest={
                'model': {'id': 'Qwen/Qwen2.5-VL-3B-Instruct'},
                'processor': {'min_pixels': 256, 'max_pixels': 512},
                'runtime': {'device': 'cuda', 'dtype': 'bfloat16', 'attention': 'sdpa'},
            },
        )

    def test_from_manifest_is_lazy_and_records_runtime_settings(self):
        backend = QwenBackend.from_manifest({
            'model': {'id': 'Qwen/Qwen2.5-VL-3B-Instruct'},
            'processor': {'min_pixels': 256, 'max_pixels': 512},
            'runtime': {'device': 'cuda', 'dtype': 'bfloat16', 'attention': 'sdpa'},
        })
        self.assertIsNone(backend.model)
        self.assertEqual(backend.runtime['min_pixels'], 256)
        self.assertEqual(backend.runtime['max_pixels'], 512)
        self.assertEqual(backend.runtime['attention'], 'sdpa')

    def test_candidate_scoring_handles_multi_token_answer(self):
        backend = self.backend()
        image = {
            'image_tokens': 8,
            'scores': {'13': [-0.1, -0.1], '4': [-0.6]},
        }
        result = backend.run({
            'image': image,
            'candidates': ['13', '4'],
            'prompt': 'Count the objects.',
        }, 'candidate_scoring')
        self.assertEqual(result['prediction'], '13')
        self.assertEqual(result['forward_passes'], 1)
        self.assertEqual(result['image_token_count'], 8)
        self.assertEqual(result['forward_passes'], 1)

    def test_candidate_scoring_uses_one_batched_forward_pass(self):
        backend = self.backend()
        image = {'image_tokens': 8, 'scores': {'13': [-0.1, -0.1], '4': [-0.6]}}
        result = backend.run({'image': image, 'candidates': ['13', '4']}, 'candidate_scoring')
        self.assertEqual(result['forward_passes'], 1)
        self.assertEqual(result['score_trace']['candidate_token_counts'], {'13': 2, '4': 1})

    def test_prior_contrastive_requires_equal_image_token_counts(self):
        backend = self.backend()
        clear = {'image_tokens': 8, 'scores': {'4': [-0.2], '5': [-0.8]}}
        prior = {'image_tokens': 8, 'scores': {'4': [-0.8], '5': [-0.2]}}
        result = backend.score_image_pair(clear, prior, ['4', '5'], alpha=1.0)
        self.assertEqual(result['prediction'], '4')
        self.assertEqual(result['forward_passes'], 2)
        self.assertEqual(result['image_token_count'], 8)
        self.assertEqual(result['alpha'], 1.0)
        self.assertEqual(result['formula'], 'clear_minus_alpha_prior')
        with self.assertRaisesRegex(ValueError, 'image-token count'):
            backend.score_image_pair(clear, {**prior, 'image_tokens': 16}, ['4', '5'], alpha=1.0)

    def test_alpha_changes_contrastive_scores(self):
        backend = self.backend()
        clear = {'image_tokens': 8, 'scores': {'4': [-0.2], '5': [-0.8]}}
        prior = {'image_tokens': 8, 'scores': {'4': [-0.1], '5': [-0.9]}}
        result = backend.score_image_pair(clear, prior, ['4', '5'], alpha=2.0)
        self.assertAlmostEqual(result['scores']['4'], 0.0)
        self.assertAlmostEqual(result['scores']['5'], 1.0)
        self.assertEqual(result['prediction'], '5')

    def test_normalized_contrastive_formula_is_explicit(self):
        backend = self.backend()
        clear = {'image_tokens': 8, 'scores': {'4': [-0.2], '5': [-0.8]}}
        prior = {'image_tokens': 8, 'scores': {'4': [-0.1], '5': [-0.9]}}
        result = backend.score_image_pair(clear, prior, ['4', '5'], alpha=2.0,
                                          formula='normalized_vcd')
        self.assertEqual(result['formula'], 'normalized_vcd')

    def test_free_generation_and_roi_methods_keep_method_metadata(self):
        backend = self.backend()
        task = {
            'image': {'image_tokens': 8, 'scores': {'4': [-0.1]}},
            'roi_image': {'image_tokens': 8, 'scores': {'4': [-0.1]}},
            'candidates': ['4'],
            'prompt': 'Count the objects.',
        }
        free = backend.run(task, 'free_generation')
        roi = backend.run(task, 'automatic_roi')
        self.assertEqual(free['prediction'], '4')
        self.assertEqual(free['forward_passes'], 1)
        self.assertEqual(roi['method'], 'automatic_roi')
        self.assertEqual(roi['roi_kind'], 'automatic_roi')

    def test_self_consistency_matches_two_forward_pass_budget(self):
        backend = self.backend()
        result = backend.run({
            'image': {'image_tokens': 8, 'scores': {'4': [-0.1]}},
            'candidates': ['4'],
            'prompt': 'Count the objects.',
            'samples': 2,
            'temperature': 0.7,
        }, 'self_consistency')
        self.assertEqual(result['prediction'], '4')
        self.assertEqual(result['forward_passes'], 2)
        self.assertEqual(len(result['raw_samples']), 2)


if __name__ == '__main__':
    unittest.main()
