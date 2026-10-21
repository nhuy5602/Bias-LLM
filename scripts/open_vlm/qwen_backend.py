"""Lazy Qwen2.5-VL adapter with testable score-pair contrastive decoding."""

import math
from pathlib import Path

from .scoring import score_candidates


class QwenBackend:
    def __init__(self, model=None, processor=None, manifest=None):
        manifest = manifest or {}
        model_config = manifest.get('model', {})
        processor_config = manifest.get('processor', {})
        runtime_config = manifest.get('runtime', {})
        self.model = model
        self.processor = processor
        self.manifest = manifest
        self.model_id = model_config.get('id', 'Qwen/Qwen2.5-VL-3B-Instruct')
        self.runtime = {
            'min_pixels': int(processor_config.get('min_pixels', 256 * 28 * 28)),
            'max_pixels': int(processor_config.get('max_pixels', 512 * 28 * 28)),
            'device': runtime_config.get('device', 'cuda'),
            'dtype': runtime_config.get('dtype', 'bfloat16'),
            'attention': runtime_config.get('attention', 'sdpa'),
        }
        self._active_prompt = 'Return only the requested answer.'

    @classmethod
    def from_manifest(cls, manifest):
        return cls(manifest=manifest)

    def _ensure_loaded(self):
        if self.model is not None and self.processor is not None:
            return
        try:
            import torch
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        except ImportError as exc:
            raise RuntimeError(
                'Qwen backend requires requirements-gpu.txt on the target Linux image') from exc
        dtype = getattr(torch, self.runtime['dtype'], torch.bfloat16)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=dtype,
            attn_implementation=self.runtime['attention'],
        ).to(self.runtime['device']).eval()
        self.processor = AutoProcessor.from_pretrained(
            self.model_id,
            min_pixels=self.runtime['min_pixels'],
            max_pixels=self.runtime['max_pixels'],
        )
        self.runtime['transformers_version'] = _package_version('transformers')
        self.runtime['torch_version'] = _package_version('torch')

    @staticmethod
    def _image_token_count(image):
        if isinstance(image, dict):
            if image.get('image_tokens') is not None:
                return int(image['image_tokens'])
            grid = image.get('image_grid_thw')
            if grid is not None:
                values = list(grid[0]) if grid and isinstance(grid[0], (list, tuple)) else list(grid)
                return math.prod(int(value) for value in values)
        value = getattr(image, 'image_tokens', None)
        if value is not None:
            return int(value)
        raise ValueError('Image token count is required for parity checks')

    def _score_image(self, image, candidate):
        if isinstance(image, dict) and 'scores' in image:
            return image['scores'][candidate]
        scorer = getattr(self.model, 'score_image', None)
        if scorer is not None:
            return scorer(image, candidate)
        return self._score_hf_image(image, candidate)

    def _generate(self, image, prompt):
        if isinstance(image, dict):
            generator = getattr(self.model, 'generate', None)
            if generator is None:
                raise RuntimeError('Injected Qwen model has no generate method')
            return generator(image, prompt)
        self._ensure_loaded()
        messages = [{'role': 'user', 'content': [
            {'type': 'image', 'image': str(Path(image).resolve())},
            {'type': 'text', 'text': prompt},
        ]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        from qwen_vl_utils import process_vision_info
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(text=[text], images=image_inputs, videos=video_inputs,
                                padding=True, return_tensors='pt').to(self.runtime['device'])
        output_ids = self.model.generate(**inputs, max_new_tokens=16)
        generated = output_ids[:, inputs.input_ids.shape[1]:]
        return self.processor.batch_decode(generated, skip_special_tokens=True)[0].strip()

    def _score_hf_image(self, image, candidate):
        self._ensure_loaded()
        messages = [{'role': 'user', 'content': [
            {'type': 'image', 'image': str(Path(image).resolve())},
            {'type': 'text', 'text': self._active_prompt},
        ]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        from qwen_vl_utils import process_vision_info
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(text=[text], images=image_inputs, videos=video_inputs,
                                padding=True, return_tensors='pt').to(self.runtime['device'])
        candidate_ids = self.processor.tokenizer(
            candidate, add_special_tokens=False, return_tensors='pt').input_ids.to(self.runtime['device'])
        prompt_length = inputs.input_ids.shape[1]
        inputs.input_ids = __import__('torch').cat([inputs.input_ids, candidate_ids], dim=1)
        inputs.attention_mask = __import__('torch').ones_like(inputs.input_ids)
        with __import__('torch').no_grad():
            logits = self.model(**inputs).logits[0]
        log_probs = __import__('torch').log_softmax(logits, dim=-1)
        token_scores = []
        for offset, token_id in enumerate(candidate_ids[0]):
            token_scores.append(float(log_probs[prompt_length - 1 + offset, token_id]))
        return token_scores

    def score_image_pair(self, clear, prior, candidates, alpha):
        clear_tokens = self._image_token_count(clear)
        prior_tokens = self._image_token_count(prior)
        if clear_tokens != prior_tokens:
            raise ValueError('clear/prior image-token count must match')
        clear_scores = score_candidates(candidates, lambda candidate: self._score_image(clear, candidate))['scores']
        prior_scores = score_candidates(candidates, lambda candidate: self._score_image(prior, candidate))['scores']
        contrastive_scores = {
            candidate: clear_scores[candidate] - float(alpha) * prior_scores[candidate]
            for candidate in candidates
        }
        prediction = max(enumerate(candidates), key=lambda item: (contrastive_scores[item[1]], -item[0]))[1]
        return {
            'prediction': prediction,
            'scores': contrastive_scores,
            'clear_scores': clear_scores,
            'prior_scores': prior_scores,
            'alpha': float(alpha),
            'forward_passes': 2,
            'image_token_count': clear_tokens,
        }

    def run(self, task: dict, method: str) -> dict:
        candidates = list(task.get('candidates', []))
        prompt = task.get('prompt', self._active_prompt)
        self._active_prompt = prompt
        try:
            if method == 'free_generation':
                return {
                    'method': method,
                    'prediction': self._generate(task['image'], prompt),
                    'forward_passes': 1,
                    'image_token_count': self._image_token_count(task['image']),
                }
            if method == 'candidate_scoring':
                result = score_candidates(candidates, lambda candidate: self._score_image(task['image'], candidate))
                return {**result, 'method': method, 'forward_passes': 1,
                        'image_token_count': self._image_token_count(task['image'])}
            if method == 'prior_contrastive':
                return {**self.score_image_pair(task['image'], task['prior_image'], candidates,
                                                task['alpha']), 'method': method}
            if method in {'oracle_roi', 'automatic_roi'}:
                roi_image = task.get('roi_image')
                if roi_image is None:
                    raise ValueError(f'{method} requires roi_image')
                result = score_candidates(candidates, lambda candidate: self._score_image(roi_image, candidate))
                return {**result, 'method': method, 'roi_kind': method,
                        'forward_passes': 1,
                        'image_token_count': self._image_token_count(roi_image)}
            raise ValueError(f'Unsupported Qwen method: {method}')
        finally:
            self._active_prompt = 'Return only the requested answer.'


def _package_version(name):
    try:
        from importlib.metadata import version
        return version(name)
    except Exception:
        return 'unknown'
