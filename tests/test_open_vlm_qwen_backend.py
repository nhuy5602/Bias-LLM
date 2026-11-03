import sys
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

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

    def score_candidate_lens(self, image, candidates):
        return [
            {'layer': 0, 'scores': {candidate: [-0.1] if candidate == '5' else [-0.5]
                                    for candidate in candidates}},
            {'layer': 1, 'scores': {candidate: [-0.6] if candidate == '5' else [-0.1]
                                    for candidate in candidates}},
        ]


class FakeTensor:
    def __init__(self, shape):
        self.shape = tuple(shape)

    def repeat(self, *repeats):
        if len(repeats) == 1 and isinstance(repeats[0], tuple):
            repeats = repeats[0]
        return FakeTensor(tuple(size * repeat for size, repeat in zip(self.shape, repeats)))

    def repeat_interleave(self, count, dim=0):
        shape = list(self.shape)
        shape[dim] *= count
        return FakeTensor(shape)


class FakeNoGradTorch:
    def __init__(self):
        self.grad_enabled = True

    class _NoGrad:
        def __init__(self, torch):
            self.torch = torch
            self.previous = torch.grad_enabled

        def __enter__(self):
            self.torch.grad_enabled = False

        def __exit__(self, *_args):
            self.torch.grad_enabled = self.previous

    def no_grad(self):
        return self._NoGrad(self)

    def tensor(self, values, **_kwargs):
        return values

    def cat(self, _values, **_kwargs):
        return object()

    def ones(self, *_args, **_kwargs):
        return object()

    def arange(self, start, stop, **_kwargs):
        return list(range(start, stop))

    def log_softmax(self, logits, dim):
        if self.grad_enabled:
            raise AssertionError('log_softmax ran with gradients enabled')
        return logits


class FakeInputIds:
    shape = (1, 2)
    dtype = 'int64'

    def repeat(self, *_args):
        return self


class FakeInputs(dict):
    def __init__(self):
        self.input_ids = FakeInputIds()
        super().__init__(input_ids=self.input_ids, attention_mask=object())


class FakeHiddenState:
    def __init__(self, torch):
        self.torch = torch

    def index_select(self, _dim, _positions):
        if self.torch.grad_enabled:
            raise AssertionError('hidden-state projection ran with gradients enabled')
        return self


class FakeScoreScalar:
    def __init__(self, torch, value):
        self.torch = torch
        self.value = value

    def __float__(self):
        if self.torch.grad_enabled:
            raise AssertionError('log-probability scalarized with gradients enabled')
        return self.value


class FakeProjectedScores:
    def __init__(self, torch):
        self.torch = torch

    def __getitem__(self, index):
        return FakeScoreScalar(self.torch, index[-1] / 10)


class FakeLogits:
    def __init__(self, torch):
        self.torch = torch

    def float(self):
        if self.torch.grad_enabled:
            raise AssertionError('lm_head output cast with gradients enabled')
        return FakeProjectedScores(self.torch)


class FakeHFQwenModel:
    def __init__(self, torch):
        self.torch = torch
        self.model = self
        self.language_model = self
        self.norm = self._norm
        self.lm_head = self._lm_head

    def __call__(self, **_inputs):
        if self.torch.grad_enabled:
            raise AssertionError('model forward ran with gradients enabled')
        return type('Outputs', (), {
            'hidden_states': (FakeHiddenState(self.torch), FakeHiddenState(self.torch)),
        })()

    def _norm(self, hidden):
        if self.torch.grad_enabled:
            raise AssertionError('RMSNorm ran with gradients enabled')
        return hidden

    def _lm_head(self, _hidden):
        if self.torch.grad_enabled:
            raise AssertionError('lm_head ran with gradients enabled')
        return FakeLogits(self.torch)


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
        self.assertTrue(backend.runtime['image_processor_use_fast'])

    def test_loader_passes_manifest_processor_mode_to_transformers(self):
        for use_fast in (True, False):
            with self.subTest(use_fast=use_fast):
                torch = ModuleType('torch')
                torch.bfloat16 = object()
                processor_calls = []
                model_calls = []

                class FakeModel:
                    def to(self, _device):
                        return self

                    def eval(self):
                        return self

                class FakeModelLoader:
                    @staticmethod
                    def from_pretrained(*_args, **kwargs):
                        model_calls.append(kwargs)
                        return FakeModel()

                class FakeProcessorLoader:
                    @staticmethod
                    def from_pretrained(*_args, **kwargs):
                        processor_calls.append(kwargs)
                        return object()

                transformers = ModuleType('transformers')
                transformers.AutoProcessor = FakeProcessorLoader
                transformers.Qwen2_5_VLForConditionalGeneration = FakeModelLoader
                with patch.dict(sys.modules, {'torch': torch, 'transformers': transformers}), \
                     patch('open_vlm.qwen_backend._package_version', return_value='4.57.6'):
                    backend = QwenBackend(manifest={
                        'model': {'id': 'Qwen/Qwen2.5-VL-3B-Instruct',
                                  'revision': '66285546d2b821cf421d4f5eb2576359d3770cd3'},
                        'runtime': {'device': 'cpu', 'dtype': 'bfloat16',
                                    'image_processor_use_fast': use_fast},
                    })
                    backend._ensure_loaded()

                self.assertEqual(processor_calls[0]['use_fast'], use_fast)
                self.assertEqual(model_calls[0]['revision'], '66285546d2b821cf421d4f5eb2576359d3770cd3')
                self.assertEqual(processor_calls[0]['revision'], '66285546d2b821cf421d4f5eb2576359d3770cd3')
                self.assertIn('dtype', model_calls[0])
                self.assertNotIn('torch_dtype', model_calls[0])

    def test_loader_rejects_missing_model_revision_before_import(self):
        backend = QwenBackend(manifest={'model': {'id': 'Qwen/Qwen2.5-VL-3B-Instruct'}})
        with self.assertRaisesRegex(RuntimeError, 'model.revision.*commit SHA'):
            backend._ensure_loaded()

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

    def test_step0_scores_the_selected_condition_image(self):
        backend = self.backend()
        result = backend.run({
            'image': {'image_tokens': 8, 'scores': {'yes': [-0.1], 'no': [-0.6]}},
            'candidates': ['yes', 'no'],
        }, 'step0')
        self.assertEqual(result['method'], 'step0')
        self.assertEqual(result['prediction'], 'yes')
        self.assertEqual(result['forward_passes'], 1)

    def test_subject_validity_uses_registered_label_candidates(self):
        backend = self.backend()
        result = backend.run({
            'image': {'image_tokens': 8, 'scores': {'cow': [-0.1], 'other': [-0.6]}},
            'candidates': ['cow', 'other'],
        }, 'subject_validity')
        self.assertEqual(result['prediction'], 'cow')
        self.assertEqual(set(result['scores']), {'cow', 'other'})

    def test_subject_validity_can_calibrate_against_empty_image_prior(self):
        backend = self.backend()
        result = backend.run({
            'image': {'image_tokens': 8, 'scores': {'cow': [-0.1], 'other': [-0.6]}},
            'empty_image': {'image_tokens': 8, 'scores': {'cow': [-0.2], 'other': [-0.2]}},
            'candidates': ['cow', 'other'],
        }, 'subject_validity')
        self.assertEqual(result['prediction'], 'cow')
        self.assertTrue(result['calibrated'])
        black = backend.run({
            'image': {'image_tokens': 8, 'scores': {'cow': [-0.2], 'other': [-0.2]}},
            'empty_image': {'image_tokens': 8, 'scores': {'cow': [-0.2], 'other': [-0.2]}},
            'candidates': ['cow', 'other'],
        }, 'subject_validity')
        self.assertIsNone(black['prediction'])

    def test_visual_inputs_repeat_flattened_patches_for_each_candidate(self):
        backend = self.backend()
        repeated = backend._repeat_model_input('pixel_values', FakeTensor((1976, 1176)), 2)
        grid = backend._repeat_model_input('image_grid_thw', FakeTensor((1, 3)), 2)
        self.assertEqual(repeated.shape, (3952, 1176))
        self.assertEqual(grid.shape, (2, 3))

    def test_logit_lens_reports_ground_truth_vs_prior_delta_by_layer(self):
        backend = self.backend()
        result = backend.run({
            'image': {'image_tokens': 8},
            'candidates': ['5', '4'],
            'ground_truth': 5,
            'prior_answer': 4,
            'prompt': 'Count and return a number.',
        }, 'logit_lens')
        self.assertEqual(result['method'], 'logit_lens')
        self.assertEqual(result['forward_passes'], 1)
        self.assertEqual(result['image_encodes'], 2)
        self.assertEqual([row['prior_minus_truth_logprob'] for row in result['layerwise_scores']],
                         [-0.4, 0.5])
        self.assertEqual([row['prediction'] for row in result['layerwise_scores']], ['5', '4'])

    def test_logit_lens_hf_projection_runs_all_tensor_ops_without_gradients(self):
        torch = FakeNoGradTorch()
        backend = self.backend()
        backend.model = FakeHFQwenModel(torch)
        backend.processor = type('Processor', (), {
            'tokenizer': type('Tokenizer', (), {'pad_token_id': 0})(),
        })()
        with patch.object(backend, '_prepare_hf', return_value=(FakeInputs(), None)), \
             patch.object(backend, '_candidate_token_ids', side_effect=[[1], [2]]), \
             patch.dict(sys.modules, {'torch': torch}):
            layers, trace = backend._score_candidates_by_layer('image.png', ['truth', 'prior'])

        self.assertEqual(layers[0]['scores'], {'truth': [0.1], 'prior': [0.2]})
        self.assertEqual(trace['forward_passes'], 1)
        self.assertTrue(torch.grad_enabled)

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

    def test_runtime_metrics_and_cost_are_recorded(self):
        backend = QwenBackend(
            model=FakeQwenModel(), processor=None,
            manifest={'runtime': {'gpu_cost_usd_per_hour': 3600.0}},
        )
        result = backend.run({'image': {'image_tokens': 8, 'scores': {'4': [-0.1]}},
                              'candidates': ['4']}, 'candidate_scoring')
        self.assertIn('seconds', result)
        self.assertIn('peak_memory_mb', result)
        self.assertIn('cost_usd', result)
        self.assertGreaterEqual(result['cost_usd'], 0.0)


if __name__ == '__main__':
    unittest.main()
