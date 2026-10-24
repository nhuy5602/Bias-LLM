"""Lazy Qwen2.5-VL adapter with testable score-pair contrastive decoding."""

import math
import time
from pathlib import Path

from .scoring import parse_free_generation, score_candidates


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
        self._last_image_token_count = None

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

    def _prepare_hf(self, image):
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
        grid = inputs.get('image_grid_thw')
        if grid is None:
            raise ValueError('Qwen processor did not return image_grid_thw')
        shape = getattr(grid, 'shape', ())
        values = list(grid[0]) if len(grid) and len(shape) > 1 else list(grid)
        token_count = math.prod(int(value) for value in values)
        self._last_image_token_count = token_count
        return inputs, token_count

    def _resolve_image_token_count(self, image):
        try:
            return self._image_token_count(image)
        except ValueError:
            if isinstance(image, (str, Path)):
                return self._prepare_hf(image)[1]
            raise

    def _score_image(self, image, candidate):
        if isinstance(image, dict) and 'scores' in image:
            return image['scores'][candidate]
        scorer = getattr(self.model, 'score_image', None)
        if scorer is not None:
            return scorer(image, candidate)
        return self._score_hf_image(image, candidate)

    def _score_candidates(self, image, candidates):
        if isinstance(image, dict) and hasattr(self.model, 'score_image_batch'):
            scores = self.model.score_image_batch(image, candidates)
            return scores, {
                'candidate_token_counts': {
                    candidate: len(scores[candidate]) if isinstance(scores[candidate], (list, tuple)) else 1
                    for candidate in candidates
                },
                'forward_passes': 1,
                'image_encodes': 1,
                'sequence_evaluations': len(candidates),
            }
        if isinstance(image, dict):
            scores = {candidate: self._score_image(image, candidate) for candidate in candidates}
            return scores, {
                'candidate_token_counts': {
                    candidate: len(scores[candidate]) if isinstance(scores[candidate], (list, tuple)) else 1
                    for candidate in candidates
                },
                'forward_passes': 1,
                'image_encodes': 1,
                'sequence_evaluations': len(candidates),
            }
        return self._score_hf_candidates(image, candidates)

    def _candidate_token_ids(self, candidate):
        token_ids = list(self.processor.tokenizer(
            candidate, add_special_tokens=False).input_ids)
        eos = getattr(self.processor.tokenizer, 'eos_token_id', None)
        if eos is not None and (not token_ids or token_ids[-1] != eos):
            token_ids.append(eos)
        return token_ids

    def _score_hf_candidates(self, image, candidates):
        inputs, _ = self._prepare_hf(image)
        import torch
        candidate_ids = [self._candidate_token_ids(candidate) for candidate in candidates]
        max_candidate = max(len(ids) for ids in candidate_ids)
        base_ids = inputs.input_ids
        prompt_length = base_ids.shape[1]
        pad_id = getattr(self.processor.tokenizer, 'pad_token_id', 0) or 0
        padded = [ids + [pad_id] * (max_candidate - len(ids)) for ids in candidate_ids]
        candidate_tensor = torch.tensor(padded, device=self.runtime['device'], dtype=base_ids.dtype)
        full_ids = torch.cat([base_ids.repeat(len(candidates), 1), candidate_tensor], dim=1)
        attention = torch.ones_like(full_ids)
        model_inputs = {}
        for key, value in inputs.items():
            if key in {'input_ids', 'attention_mask'}:
                continue
            if hasattr(value, 'repeat_interleave') and getattr(value, 'shape', (1,))[0] == 1:
                model_inputs[key] = value.repeat_interleave(len(candidates), dim=0)
            else:
                model_inputs[key] = value
        model_inputs.update({'input_ids': full_ids, 'attention_mask': attention})
        with torch.no_grad():
            logits = self.model(**model_inputs).logits
        log_probs = torch.log_softmax(logits, dim=-1)
        scores = {}
        token_counts = {}
        for index, (candidate, ids) in enumerate(zip(candidates, candidate_ids)):
            values = [float(log_probs[index, prompt_length - 1 + offset, token_id])
                      for offset, token_id in enumerate(ids)]
            scores[candidate] = values
            token_counts[candidate] = len(ids)
        return scores, {'candidate_token_counts': token_counts, 'forward_passes': 1,
                        'image_encodes': 1, 'sequence_evaluations': len(candidates)}

    def _with_runtime(self, result, started):
        seconds = time.perf_counter() - started
        peak_memory_mb = 0.0
        try:
            import torch
            if torch.cuda.is_available():
                peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        except (ImportError, RuntimeError):
            pass
        runtime = self.manifest.get('runtime', {})
        rate = float(runtime.get('gpu_cost_usd_per_hour', 0.0) or 0.0)
        result.setdefault('seconds', seconds)
        result.setdefault('peak_memory_mb', peak_memory_mb)
        result.setdefault('cost_usd', seconds * rate / 3600.0)
        result.setdefault('cost_estimate_source', 'gpu_cost_usd_per_hour' if rate else 'unconfigured')
        result.setdefault('image_encodes', result.get('forward_passes', 0))
        result.setdefault('sequence_evaluations', 0)
        return result

    def _generate(self, image, prompt, do_sample=False, temperature=1.0):
        if isinstance(image, dict):
            generator = getattr(self.model, 'generate', None)
            if generator is None:
                raise RuntimeError('Injected Qwen model has no generate method')
            return generator(image, prompt)
        self._active_prompt = prompt
        inputs, _ = self._prepare_hf(image)
        output_ids = self.model.generate(**inputs, max_new_tokens=16,
                                         do_sample=do_sample, temperature=temperature)
        generated = output_ids[:, inputs.input_ids.shape[1]:]
        return self.processor.batch_decode(generated, skip_special_tokens=True)[0].strip()

    def _score_hf_image(self, image, candidate):
        inputs, _ = self._prepare_hf(image)
        import torch
        candidate_ids = self.processor.tokenizer(
            candidate, add_special_tokens=False, return_tensors='pt').input_ids.to(self.runtime['device'])
        prompt_length = inputs.input_ids.shape[1]
        inputs.input_ids = torch.cat([inputs.input_ids, candidate_ids], dim=1)
        inputs.attention_mask = torch.ones_like(inputs.input_ids)
        with torch.no_grad():
            logits = self.model(**inputs).logits[0]
        log_probs = torch.log_softmax(logits, dim=-1)
        token_scores = []
        for offset, token_id in enumerate(candidate_ids[0]):
            token_scores.append(float(log_probs[prompt_length - 1 + offset, token_id]))
        return token_scores

    def score_image_pair(self, clear, prior, candidates, alpha, formula='clear_minus_alpha_prior'):
        clear_tokens = self._resolve_image_token_count(clear)
        prior_tokens = self._resolve_image_token_count(prior)
        if clear_tokens != prior_tokens:
            raise ValueError('clear/prior image-token count must match')
        clear_raw, clear_trace = self._score_candidates(clear, candidates)
        prior_raw, prior_trace = self._score_candidates(prior, candidates)
        clear_scores = {candidate: sum(float(value) for value in (values if isinstance(values, (list, tuple)) else [values]))
                        for candidate, values in clear_raw.items()}
        prior_scores = {candidate: sum(float(value) for value in (values if isinstance(values, (list, tuple)) else [values]))
                        for candidate, values in prior_raw.items()}
        if formula == 'normalized_vcd':
            contrastive_scores = {
                candidate: (1.0 + float(alpha)) * clear_scores[candidate] - float(alpha) * prior_scores[candidate]
                for candidate in candidates
            }
        elif formula == 'clear_minus_alpha_prior':
            contrastive_scores = {
                candidate: clear_scores[candidate] - float(alpha) * prior_scores[candidate]
                for candidate in candidates
            }
        else:
            raise ValueError(f'Unsupported contrastive formula: {formula}')
        prediction = max(enumerate(candidates), key=lambda item: (contrastive_scores[item[1]], -item[0]))[1]
        return {
            'prediction': prediction,
            'scores': contrastive_scores,
            'clear_scores': clear_scores,
            'prior_scores': prior_scores,
            'alpha': float(alpha),
            'formula': formula,
            'forward_passes': clear_trace['forward_passes'] + prior_trace['forward_passes'],
            'score_trace': {'clear': clear_trace, 'prior': prior_trace},
            'image_encodes': clear_trace.get('image_encodes', 1) + prior_trace.get('image_encodes', 1),
            'sequence_evaluations': clear_trace.get('sequence_evaluations', len(candidates))
            + prior_trace.get('sequence_evaluations', len(candidates)),
            'image_token_count': clear_tokens,
        }

    def run(self, task: dict, method: str) -> dict:
        started = time.perf_counter()
        candidates = list(task.get('candidates', []))
        prompt = task.get('prompt', self._active_prompt)
        self._active_prompt = prompt
        try:
            if method == 'free_generation':
                raw = self._generate(task['image'], prompt)
                prediction = parse_free_generation(raw, candidates) if candidates else raw
                return self._with_runtime({
                    'method': method,
                    'prediction': prediction,
                    'raw': raw,
                    'forward_passes': 1,
                    'image_token_count': self._resolve_image_token_count(task['image']),
                }, started)
            if method == 'self_consistency':
                samples = int(task.get('samples', 2))
                temperature = float(task.get('temperature', 0.7))
                raw_samples = [self._generate(task['image'], prompt, do_sample=True,
                                              temperature=temperature) for _ in range(samples)]
                parsed = [parse_free_generation(raw, candidates) for raw in raw_samples]
                valid = [value for value in parsed if value is not None]
                prediction = max(set(valid), key=valid.count) if valid else None
                return self._with_runtime({
                    'method': method,
                    'prediction': prediction,
                    'raw_samples': raw_samples,
                    'forward_passes': samples,
                    'temperature': temperature,
                    'image_token_count': self._resolve_image_token_count(task['image']),
                }, started)
            if method == 'candidate_scoring':
                raw_scores, trace = self._score_candidates(task['image'], candidates)
                result = score_candidates(candidates, lambda candidate: raw_scores[candidate])
                return self._with_runtime({**result, 'method': method, 'forward_passes': trace['forward_passes'],
                        'score_trace': trace,
                        'image_token_count': self._resolve_image_token_count(task['image'])}, started)
            if method == 'subject_validity':
                subject_candidates = ['yes', 'no']
                raw_scores, trace = self._score_candidates(task['image'], subject_candidates)
                result = score_candidates(subject_candidates, lambda candidate: raw_scores[candidate])
                return self._with_runtime({**result, 'method': method, 'forward_passes': trace['forward_passes'],
                        'score_trace': trace,
                        'image_token_count': self._resolve_image_token_count(task['image'])}, started)
            if method in {'prior_contrastive', 'prior_contrastive_control'}:
                return self._with_runtime({**self.score_image_pair(task['image'], task['prior_image'], candidates,
                                                task['alpha'], task.get('formula', 'clear_minus_alpha_prior')),
                        'method': method}, started)
            if method in {'oracle_roi', 'automatic_roi'}:
                roi_image = task.get('roi_image')
                if roi_image is None:
                    raise ValueError(f'{method} requires roi_image')
                raw_scores, trace = self._score_candidates(roi_image, candidates)
                result = score_candidates(candidates, lambda candidate: raw_scores[candidate])
                return self._with_runtime({**result, 'method': method, 'roi_kind': method,
                        'forward_passes': trace['forward_passes'], 'score_trace': trace,
                        'image_token_count': self._resolve_image_token_count(roi_image)}, started)
            raise ValueError(f'Unsupported Qwen method: {method}')
        finally:
            self._active_prompt = 'Return only the requested answer.'


def _package_version(name):
    try:
        from importlib.metadata import version
        return version(name)
    except Exception:
        return 'unknown'
