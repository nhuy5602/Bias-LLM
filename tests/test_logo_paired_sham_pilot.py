import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from logo_paired_sham_pilot import (
    CONDITIONS,
    FAMILIES,
    ROI_FRACTIONS,
    REPEATS,
    build_gemini_prompt,
    build_tasks,
    erase_region,
    prepare_run,
    relative_bbox,
    sha256,
)


class LogoPairedShamPilotTests(unittest.TestCase):
    @staticmethod
    def image_size(path):
        with Image.open(path) as image:
            return list(image.size)

    def source_rows(self, root):
        rows = []
        image_dir = Path(root) / 'images'
        image_dir.mkdir(parents=True, exist_ok=True)
        for family, config in FAMILIES.items():
            path = image_dir / f'{family}-original.png'
            image = Image.new('RGB', (1024, 829), (40, 40, 40))
            for x in range(1024):
                for y in range(829):
                    image.putpixel((x, y), ((x + y) % 256, 40, 40))
            image.save(path, format='PNG', optimize=False)
            rows.append({
                'family': family,
                'state': 'original',
                'ground_truth': config['ground_truth'],
                'expected_bias': None,
                'local_path': str(path),
                'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            })
        return rows

    def test_relative_bbox_stays_inside_normalized_image(self):
        for family in FAMILIES:
            bbox = relative_bbox((384, 266), family, sham=False)
            sham = relative_bbox((384, 266), family, sham=True)
            for candidate in (bbox, sham):
                self.assertGreaterEqual(candidate[0], 0)
                self.assertGreaterEqual(candidate[1], 0)
                self.assertLessEqual(candidate[2], 384)
                self.assertLessEqual(candidate[3], 266)
                self.assertLess(candidate[0], candidate[2])
                self.assertLess(candidate[1], candidate[3])
            self.assertNotEqual(bbox, sham)

    def test_logo_rois_use_compact_family_specific_bands(self):
        self.assertEqual(ROI_FRACTIONS['audi-black-coupe'], (0.42, 0.54, 0.58, 0.64))
        self.assertEqual(ROI_FRACTIONS['audi-black-suv'], (0.42, 0.48, 0.58, 0.60))
        self.assertEqual(ROI_FRACTIONS['maserati-black-coupe'], (0.45, 0.56, 0.55, 0.68))
        self.assertEqual(ROI_FRACTIONS['maserati-black-suv'], (0.45, 0.48, 0.55, 0.62))
        self.assertEqual(ROI_FRACTIONS['mercedes-black-coupe'], (0.43, 0.48, 0.57, 0.62))
        self.assertEqual(ROI_FRACTIONS['mercedes-black-suv'], (0.44, 0.42, 0.56, 0.56))

    def test_erase_region_preserves_dimensions_and_changes_only_roi(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'source.png'
            target = root / 'target.png'
            image = Image.new('RGB', (64, 48), (120, 80, 40))
            for x in range(20, 40):
                for y in range(15, 30):
                    image.putpixel((x, y), (255, 255, 255))
            image.save(source, format='PNG', optimize=False)
            metadata = erase_region(source, target, (20, 15, 40, 30), blur_radius=6)
            self.assertEqual(metadata['source_size'], [64, 48])
            self.assertEqual(metadata['derived_size'], [64, 48])
            with Image.open(source) as before, Image.open(target) as after:
                for x in range(64):
                    for y in range(48):
                        if not (20 <= x < 40 and 15 <= y < 30):
                            self.assertEqual(before.getpixel((x, y)), after.getpixel((x, y)))
            self.assertNotEqual(source.read_bytes(), target.read_bytes())

    def test_task_matrix_has_six_families_three_conditions_three_repeats(self):
        variants = []
        for family, config in FAMILIES.items():
            for condition in CONDITIONS:
                variants.append({
                    'family': family,
                    'condition': condition,
                    'derived_path': f'{family}-{condition}.png',
                    'source_sha256': 's' * 64,
                    'normalized_sha256': 'n' * 64,
                    'derived_sha256': 'd' * 64,
                    'source_size': [1024, 829],
                    'normalized_size': [384, 311],
                    'derived_size': [384, 311],
                    'bbox': [10, 10, 20, 20],
                    'sham_bbox': [30, 10, 40, 20],
                    'ground_truth': 0 if condition == 'logo_erased' else config['ground_truth'],
                    'expected_bias': config['ground_truth'] if condition == 'logo_erased' else None,
                })
        tasks = build_tasks(variants)
        self.assertEqual(len(tasks), 6 * len(CONDITIONS) * REPEATS)
        self.assertEqual({task['condition'] for task in tasks}, set(CONDITIONS))
        self.assertEqual({task['repeat'] for task in tasks}, set(range(1, REPEATS + 1)))
        self.assertEqual({task['agent'] for task in tasks}, {'codex'})

    def test_prompt_records_pixel_pair_and_sham_provenance(self):
        variant = {
            'family': 'audi-black-coupe',
            'condition': 'logo_erased',
            'derived_path': 'logo-erased.png',
            'source_sha256': 's' * 64,
            'normalized_sha256': 'n' * 64,
            'derived_sha256': 'd' * 64,
            'source_size': [1024, 829],
            'normalized_size': [384, 311],
            'derived_size': [384, 311],
            'bbox': [160, 140, 224, 175],
            'sham_bbox': [250, 140, 314, 175],
            'ground_truth': 0,
            'expected_bias': 4,
        }
        prompt = build_gemini_prompt([variant])
        self.assertIn('pixel-paired', prompt.lower())
        self.assertIn('sham', prompt.lower())
        self.assertIn('NEEDS_EVIDENCE', prompt)
        self.assertIn('s' * 64, prompt)
        self.assertIn('n' * 64, prompt)
        self.assertIn('d' * 64, prompt)
        self.assertIn('logo_erased', prompt)

    def test_prepare_run_creates_shared_source_pairs_and_three_conditions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_run = root / 'source-run'
            rows = self.source_rows(source_run)
            (source_run / 'manifest.json').write_text(json.dumps({'rows': rows}), encoding='utf-8')
            prepared = root / 'prepared'
            manifest = prepare_run(prepared, source_run)
            self.assertEqual(manifest['protocol'], 'vlmbias-logo-pixel-paired-sham-v1')
            self.assertEqual(len(manifest['variants']), 6 * len(CONDITIONS))
            self.assertEqual(len(manifest['tasks']), 6 * len(CONDITIONS) * REPEATS)
            for family in FAMILIES:
                group = [v for v in manifest['variants'] if v['family'] == family]
                self.assertEqual({v['condition'] for v in group}, set(CONDITIONS))
                self.assertEqual(len({v['source_sha256'] for v in group}), 1)
                self.assertEqual(len({v['normalized_sha256'] for v in group}), 1)
                for variant in group:
                    self.assertEqual(variant['source_sha256'], sha256(variant['source_path']))
                    self.assertEqual(variant['normalized_sha256'], sha256(variant['normalized_path']))
                    self.assertEqual(variant['derived_sha256'], sha256(variant['derived_path']))
                    self.assertEqual(variant['source_size'], self.image_size(variant['source_path']))
                    self.assertEqual(variant['normalized_size'], self.image_size(variant['normalized_path']))
                    self.assertEqual(variant['derived_size'], self.image_size(variant['derived_path']))
                original = next(v for v in group if v['condition'] == 'original')
                self.assertEqual(original['normalized_sha256'], original['derived_sha256'])
            self.assertTrue((prepared / 'GEMINI-PROMPT.md').is_file())


if __name__ == '__main__':
    unittest.main()
