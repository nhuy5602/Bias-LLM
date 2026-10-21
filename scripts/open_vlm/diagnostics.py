"""Pre-method diagnostics and controls for the open-VLM track."""

import random
import re

SIGMAS = (1, 3, 6, 12)
ALPHAS = (0.25, 0.5, 1.0, 2.0)
MIN_SUBJECT_VALIDITY = 0.90
MAX_INVALID_RATE = 0.05
MIN_DETAIL_DROP = 0.10
RANDOM_ALPHA_DRAWS = 20


def select_blur_sigma(records) -> dict:
    reasons = []
    for record in sorted(records, key=lambda item: item['sigma']):
        detail_drop = record['clear_accuracy'] - record['blur_accuracy']
        if (record['subject_valid_rate'] >= MIN_SUBJECT_VALIDITY
                and record['invalid_rate'] <= MAX_INVALID_RATE
                and detail_drop >= MIN_DETAIL_DROP):
            return {
                'status': 'PASS',
                'sigma': record['sigma'],
                'subject_validity': record['subject_valid_rate'],
                'invalid_rate': record['invalid_rate'],
                'detail_drop': detail_drop,
            }
        reasons.append({
            'sigma': record['sigma'],
            'subject_validity': record['subject_valid_rate'],
            'invalid_rate': record['invalid_rate'],
            'detail_drop': detail_drop,
        })
    return {'status': 'UNSUPPORTED', 'sigma': None, 'reasons': reasons}


def _image_size(task):
    size = task.get('size')
    if not isinstance(size, (list, tuple)) or len(size) != 2:
        raise ValueError('Control task requires a two-element image size')
    width, height = (int(size[0]), int(size[1]))
    if width <= 0 or height <= 0:
        raise ValueError('Control image size must be positive')
    return [width, height]


def build_negative_controls(task, seed) -> list[dict]:
    size = _image_size(task)
    controls = [
        {'kind': 'noise', 'size': size, 'seed': seed, 'alpha': None},
        {'kind': 'shuffled_prior', 'size': size, 'seed': seed,
         'prior_image_path': task.get('prior_image_path')},
        {'kind': 'empty_black', 'size': size, 'seed': seed, 'alpha': None},
    ]
    rng = random.Random(seed)
    controls.extend({
        'kind': 'random_alpha',
        'size': size,
        'seed': seed,
        'draw': index,
        'alpha': rng.choice(ALPHAS),
    } for index in range(RANDOM_ALPHA_DRAWS))
    return controls


def validate_roi(roi, row, kind) -> dict:
    if kind not in {'oracle_roi', 'reference_roi', 'automatic_roi'}:
        raise ValueError(f'Unsupported ROI kind: {kind}')
    prompt = str(roi.get('prompt', '')).strip()
    if (not prompt or re.search(r'\b(count|counting|how many|number)\b|[0-9]', prompt.casefold())):
        raise ValueError('ROI prompt must be neutral and contain no count hint')
    size = _image_size({'size': row.get('image_size')})
    bbox = roi.get('bbox')
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        raise ValueError('ROI requires four bbox coordinates')
    x1, y1, x2, y2 = (int(value) for value in bbox)
    if not (0 <= x1 < x2 <= size[0] and 0 <= y1 < y2 <= size[1]):
        raise ValueError('ROI bbox is outside image bounds')
    ground_truth = kind == 'oracle_roi'
    if ground_truth and not row.get('dataset_bbox_verified', False):
        raise ValueError('oracle_roi requires verified ground-truth geometry')
    return {
        'kind': kind,
        'bbox': [x1, y1, x2, y2],
        'prompt': prompt,
        'source': roi.get('source', 'unknown'),
        'ground_truth': ground_truth,
    }
