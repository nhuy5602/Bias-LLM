import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from logo_robustness_pilot import (
    BLUR_SIGMAS,
    FAMILIES,
    MATCHED_BLUR_MAX_EDGE,
    MATCHED_BLUR_SIGMAS,
    RESOLUTION_LEVELS,
    REPEATS,
    build_blur_gemini_prompt,
    build_blur_tasks,
    build_gemini_prompt,
    build_matched_blur_gemini_prompt,
    build_matched_blur_tasks,
    build_tasks,
    derive_blur_variant,
    derive_matched_blur_variant,
    derive_variant,
    normalize_source_variant,
    prepare_matched_blur_run,
    select_source_rows,
)


class LogoRobustnessPilotTests(unittest.TestCase):
    def source_rows(self):
        rows = []
        for family, config in FAMILIES.items():
            rows.extend([
                {'family': family, 'state': 'original', 'ground_truth': config['ground_truth'],
                 'expected_bias': None, 'local_path': f'{family}-original.png',
                 'sha256': hashlib.sha256(f'{family}-original'.encode()).hexdigest()},
                {'family': family, 'state': 'counterfactual', 'ground_truth': config['counterfactual'],
                 'expected_bias': config['ground_truth'], 'local_path': f'{family}-counterfactual.png',
                 'sha256': hashlib.sha256(f'{family}-counterfactual'.encode()).hexdigest()},
            ])
        return rows

    def test_select_source_rows_requires_all_six_pairs(self):
        selected = select_source_rows(self.source_rows())
        self.assertEqual(len(selected), 12)
        self.assertEqual({row['family'] for row in selected}, set(FAMILIES))
        self.assertEqual({row['state'] for row in selected}, {'original', 'counterfactual'})

    def test_derive_variant_records_source_and_derived_hashes(self):
        from PIL import Image

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'source.png'
            target = root / 'derived.png'
            Image.new('RGB', (400, 200), (255, 0, 0)).save(source)
            metadata = derive_variant(source, target, 192)
            self.assertEqual(metadata['max_edge'], 192)
            self.assertEqual(metadata['source_sha256'], hashlib.sha256(source.read_bytes()).hexdigest())
            self.assertEqual(metadata['derived_sha256'], hashlib.sha256(target.read_bytes()).hexdigest())
            with Image.open(target) as image:
                self.assertLessEqual(max(image.size), 192)

    def test_task_matrix_has_six_families_two_states_three_levels_three_repeats(self):
        variants = []
        for row in self.source_rows():
            for level in RESOLUTION_LEVELS:
                variants.append({**row, 'resolution': level,
                                 'derived_path': f"{row['family']}-{row['state']}-{level}.png",
                                 'derived_sha256': 'd' * 64})
        tasks = build_tasks(variants)
        self.assertEqual(len(tasks), 6 * 2 * len(RESOLUTION_LEVELS) * REPEATS)
        self.assertEqual({task['resolution'] for task in tasks}, set(RESOLUTION_LEVELS))
        self.assertEqual({task['repeat'] for task in tasks}, set(range(1, REPEATS + 1)))
        self.assertEqual({task['agent'] for task in tasks}, {'codex'})

    def test_gemini_prompt_contains_provenance_and_resolution_levels(self):
        row = self.source_rows()[0]
        variants = [{**row, 'resolution': level, 'derived_path': f'image-{level}.png',
                     'derived_sha256': 'd' * 64} for level in RESOLUTION_LEVELS]
        prompt = build_gemini_prompt(variants)
        self.assertIn('NEEDS_EVIDENCE', prompt)
        self.assertIn('source SHA256', prompt)
        for level in RESOLUTION_LEVELS:
            self.assertIn(str(level), prompt)
        self.assertIn('d' * 64, prompt)

    def test_blur_variant_preserves_dimensions_and_records_sigma(self):
        from PIL import Image

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'source.png'
            target = root / 'blurred.png'
            Image.new('RGB', (120, 80), (255, 0, 0)).save(source)
            metadata = derive_blur_variant(source, target, 3.0)
            self.assertEqual(metadata['blur_sigma'], 3.0)
            self.assertEqual(metadata['source_size'], [120, 80])
            self.assertEqual(metadata['derived_size'], [120, 80])
            self.assertEqual(len(metadata['source_sha256']), 64)
            self.assertEqual(len(metadata['derived_sha256']), 64)

    def test_blur_task_matrix_and_prompt_use_three_sigmas(self):
        variants = []
        for row in self.source_rows():
            for sigma in BLUR_SIGMAS:
                variants.append({**row, 'blur_sigma': sigma,
                                 'derived_path': f"{row['family']}-{sigma}.png",
                                 'derived_sha256': 'e' * 64})
        tasks = build_blur_tasks(variants)
        self.assertEqual(len(tasks), 6 * 2 * len(BLUR_SIGMAS) * REPEATS)
        self.assertEqual({task['corruption'] for task in tasks}, {'gaussian_blur'})
        prompt = build_blur_gemini_prompt(variants)
        self.assertIn('GaussianBlur', prompt)
        self.assertIn('Blur sigma', prompt)
        for sigma in BLUR_SIGMAS:
            self.assertIn(f'| {sigma:g} |', prompt)

    def test_normalize_source_variant_records_source_and_normalized_provenance(self):
        from PIL import Image

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'source.png'
            normalized = root / 'normalized.png'
            Image.new('RGB', (1024, 829), (255, 0, 0)).save(source)
            metadata = normalize_source_variant(source, normalized, MATCHED_BLUR_MAX_EDGE)
            self.assertEqual(metadata['max_edge'], MATCHED_BLUR_MAX_EDGE)
            self.assertEqual(metadata['source_size'], [1024, 829])
            self.assertEqual(metadata['normalized_size'], [384, 311])
            self.assertEqual(metadata['source_sha256'], hashlib.sha256(source.read_bytes()).hexdigest())
            self.assertEqual(metadata['normalized_sha256'], hashlib.sha256(normalized.read_bytes()).hexdigest())
            with Image.open(normalized) as image:
                self.assertLessEqual(max(image.size), MATCHED_BLUR_MAX_EDGE)

    def test_matched_blur_variant_preserves_three_hashes_and_dimensions(self):
        from PIL import Image

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'source.png'
            normalized = root / 'normalized.png'
            blurred = root / 'blurred.png'
            Image.new('RGB', (1024, 829), (255, 0, 0)).save(source)
            normalization = normalize_source_variant(source, normalized, MATCHED_BLUR_MAX_EDGE)
            metadata = derive_matched_blur_variant(normalized, blurred, 3.0, normalization)
            self.assertEqual(metadata['blur_sigma'], 3.0)
            self.assertEqual(metadata['source_size'], [1024, 829])
            self.assertEqual(metadata['normalized_size'], [384, 311])
            self.assertEqual(metadata['derived_size'], [384, 311])
            self.assertEqual(metadata['source_sha256'], hashlib.sha256(source.read_bytes()).hexdigest())
            self.assertEqual(metadata['normalized_sha256'], hashlib.sha256(normalized.read_bytes()).hexdigest())
            self.assertEqual(metadata['derived_sha256'], hashlib.sha256(blurred.read_bytes()).hexdigest())

    def test_matched_blur_sigma_zero_is_the_normalized_baseline(self):
        from PIL import Image

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'source.png'
            normalized = root / 'normalized.png'
            baseline = root / 'baseline.png'
            Image.new('RGB', (1024, 829), (255, 0, 0)).save(source)
            normalization = normalize_source_variant(source, normalized, MATCHED_BLUR_MAX_EDGE)
            metadata = derive_matched_blur_variant(normalized, baseline, 0.0, normalization)
            self.assertEqual(metadata['blur_sigma'], 0.0)
            self.assertEqual(metadata['normalized_sha256'], metadata['derived_sha256'])
            self.assertEqual(normalized.read_bytes(), baseline.read_bytes())

    def test_matched_blur_task_matrix_and_prompt_pin_normalization(self):
        variants = []
        for row in self.source_rows():
            for sigma in MATCHED_BLUR_SIGMAS:
                variants.append({
                    **row,
                    'max_edge': MATCHED_BLUR_MAX_EDGE,
                    'blur_sigma': sigma,
                    'source_size': [1024, 829],
                    'normalized_size': [384, 311],
                    'derived_size': [384, 311],
                    'normalized_path': f"{row['family']}-{row['state']}-normalized.png",
                    'derived_path': f"{row['family']}-{row['state']}-sigma{sigma:g}.png",
                    'normalized_sha256': 'n' * 64,
                    'derived_sha256': 'e' * 64,
                })
        tasks = build_matched_blur_tasks(variants)
        self.assertEqual(len(tasks), 6 * 2 * len(MATCHED_BLUR_SIGMAS) * REPEATS)
        self.assertEqual({task['corruption'] for task in tasks}, {'gaussian_blur_matched'})
        self.assertEqual({task['max_edge'] for task in tasks}, {MATCHED_BLUR_MAX_EDGE})
        self.assertTrue(all('-r384-' in task['task_id'] for task in tasks))
        prompt = build_matched_blur_gemini_prompt(variants)
        self.assertIn('Normalized SHA256', prompt)
        self.assertIn('normalize', prompt.lower())
        self.assertIn('384', prompt)
        self.assertIn('n' * 64, prompt)
        self.assertIn('e' * 64, prompt)

    def test_prepare_matched_blur_run_writes_resolution_matched_manifest(self):
        from PIL import Image

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_run = root / 'source-run'
            source_images = source_run / 'images'
            source_images.mkdir(parents=True)
            rows = []
            for family, config in FAMILIES.items():
                for state, truth, expected_bias, size in (
                    ('original', config['ground_truth'], None, (1024, 829)),
                    ('counterfactual', config['counterfactual'], config['ground_truth'], (384, 266)),
                ):
                    image_path = source_images / f'{family}-{state}.png'
                    Image.new('RGB', size, (255, 0, 0)).save(image_path)
                    rows.append({
                        'family': family,
                        'state': state,
                        'ground_truth': truth,
                        'expected_bias': expected_bias,
                        'local_path': str(image_path),
                        'sha256': hashlib.sha256(image_path.read_bytes()).hexdigest(),
                    })
            (source_run / 'manifest.json').write_text(json.dumps({'rows': rows}), encoding='utf-8')
            prepared = root / 'prepared'
            manifest = prepare_matched_blur_run(prepared, source_run)
            self.assertEqual(manifest['protocol'], 'vlmbias-logo-gaussian-blur-matched-v2')
            self.assertEqual(manifest['mode'], 'gaussian_blur_matched')
            self.assertEqual(manifest['normalization_max_edge'], MATCHED_BLUR_MAX_EDGE)
            self.assertEqual(len(manifest['variants']), 12 * len(MATCHED_BLUR_SIGMAS))
            self.assertEqual(len(manifest['tasks']), 144)
            self.assertEqual(len({variant['normalized_path'] for variant in manifest['variants']}), 12)
            self.assertTrue(all(variant['normalized_size'][0] <= 384 for variant in manifest['variants']))
            self.assertTrue(all(max(variant['derived_size']) <= 384 for variant in manifest['variants']))
            self.assertTrue((prepared / 'GEMINI-PROMPT.md').is_file())


if __name__ == '__main__':
    unittest.main()
