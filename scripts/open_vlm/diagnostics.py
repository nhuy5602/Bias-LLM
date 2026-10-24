"""Pre-method diagnostics and controls for the open-VLM track."""

import random
import re
import hashlib
from pathlib import Path

from PIL import Image, ImageFilter

SIGMAS = (1, 3, 6, 12)
ALPHAS = (0.25, 0.5, 1.0, 2.0)
MIN_SUBJECT_VALIDITY = 0.90
MAX_INVALID_RATE = 0.05
MIN_DETAIL_DROP = 0.10
RANDOM_ALPHA_DRAWS = 20
MIN_BIAS_ALIGNMENT = 0.50
MIN_BIAS_LIFT_OVER_BLACK = 0.10
MAX_BLACK_SUBJECT_VALIDITY = 0.20


def select_blur_sigma(records) -> dict:
    reasons = []
    for record in sorted(records, key=lambda item: item['sigma']):
        detail_drop = record['clear_accuracy'] - record['blur_accuracy']
        if (record['subject_valid_rate'] >= MIN_SUBJECT_VALIDITY
                and record['invalid_rate'] <= MAX_INVALID_RATE
                and detail_drop >= MIN_DETAIL_DROP
                and record.get('blur_bias_rate', 0.0) >= MIN_BIAS_ALIGNMENT
                and record.get('blur_bias_rate', 0.0) - record.get('black_bias_rate', 0.0)
                >= MIN_BIAS_LIFT_OVER_BLACK
                and record.get('black_subject_valid_rate', 0.0) <= MAX_BLACK_SUBJECT_VALIDITY):
            return {
                'status': 'PASS',
                'sigma': record['sigma'],
                'subject_validity': record['subject_valid_rate'],
                'invalid_rate': record['invalid_rate'],
                'detail_drop': detail_drop,
                'blur_bias_rate': record.get('blur_bias_rate', 0.0),
                'black_bias_rate': record.get('black_bias_rate', 0.0),
                'black_subject_valid_rate': record.get('black_subject_valid_rate', 0.0),
            }
        reasons.append({
            'sigma': record['sigma'],
            'subject_validity': record['subject_valid_rate'],
            'invalid_rate': record['invalid_rate'],
            'detail_drop': detail_drop,
            'blur_bias_rate': record.get('blur_bias_rate', 0.0),
            'black_bias_rate': record.get('black_bias_rate', 0.0),
            'black_subject_valid_rate': record.get('black_subject_valid_rate', 0.0),
        })
    return {'status': 'UNSUPPORTED', 'sigma': None, 'reasons': reasons}


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def materialize_image_variants(source_path, output_dir, sigma, seed=0):
    source_path = Path(source_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_sha256 = _sha256(source_path)
    stem = f'{source_path.stem}-{source_sha256[:12]}'
    blur_path = output_dir / f'{stem}-blur-sigma{sigma}.png'
    black_path = output_dir / f'{stem}-black.png'
    noise_path = output_dir / f'{stem}-noise-seed{seed}.png'
    shuffled_path = output_dir / f'{stem}-shuffled.png'
    if all(path.is_file() for path in (blur_path, black_path, noise_path, shuffled_path)):
        with Image.open(source_path) as opened:
            size = list(opened.size)
        sigma_pixels = float(sigma) * min(size) / 384.0
        return {
            'source_path': str(source_path), 'blur_path': str(blur_path),
            'black_path': str(black_path), 'noise_path': str(noise_path),
            'shuffled_prior_path': str(shuffled_path), 'size': size,
            'source_sha256': source_sha256, 'blur_sha256': _sha256(blur_path),
            'black_sha256': _sha256(black_path), 'noise_sha256': _sha256(noise_path),
            'shuffled_prior_sha256': _sha256(shuffled_path), 'sigma': sigma,
            'sigma_pixels': sigma_pixels,
        }
    with Image.open(source_path) as opened:
        image = opened.convert('RGB')
        size = list(image.size)
        sigma_pixels = float(sigma) * min(size) / 384.0
        image.filter(ImageFilter.GaussianBlur(sigma_pixels)).save(blur_path, format='PNG', optimize=False)
        Image.new('RGB', image.size, (0, 0, 0)).save(black_path, format='PNG', optimize=False)
        noise = Image.effect_noise(image.size, 127).convert('RGB')
        noise.save(noise_path, format='PNG', optimize=False)
        image.transpose(Image.Transpose.FLIP_LEFT_RIGHT).save(shuffled_path, format='PNG', optimize=False)
    return {
        'source_path': str(source_path),
        'blur_path': str(blur_path),
        'black_path': str(black_path),
        'noise_path': str(noise_path),
        'shuffled_prior_path': str(shuffled_path),
        'size': size,
        'source_sha256': source_sha256,
        'blur_sha256': _sha256(blur_path),
        'black_sha256': _sha256(black_path),
        'noise_sha256': _sha256(noise_path),
        'shuffled_prior_sha256': _sha256(shuffled_path),
        'sigma': sigma,
        'sigma_pixels': sigma_pixels,
    }


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
