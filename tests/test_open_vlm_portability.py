import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))

from open_vlm.runner import safe_upload_then_delete  # noqa: E402
from open_vlm.schema import DomainSpec, validate_row  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]


class OpenVlmPortabilityTests(unittest.TestCase):
    def test_manifest_paths_are_posix_normalized(self):
        row = validate_row({
            'domain': 'animals',
            'family': 'animal-1',
            'ground_truth': 4,
            'expected_bias': 5,
            'image_path': r'images\animal-1.png',
        }, DomainSpec('animals', 'integer', tuple(range(21)), ('expected_bias',)))
        self.assertEqual(row['image_path'], 'images/animal-1.png')

    def test_launcher_has_strict_mode_and_cleanup_hooks(self):
        script = (ROOT / 'run-open-vlm.sh').read_text(encoding='utf-8')
        self.assertIn('set -euo pipefail', script)
        self.assertIn('trap cleanup EXIT', script)
        self.assertIn('OPEN_VLM_UPLOADER', script)
        self.assertIn('OPEN_VLM_DELETER', script)
        self.assertIn('OPEN_VLM_STOPPER', script)
        self.assertIn('OPEN_VLM_TIMEOUT_SECONDS', script)
        self.assertIn('no-regression', script)

    def test_launcher_rejects_missing_arguments(self):
        bash = None
        if os.name == 'nt':
            candidate = Path(r'C:\Program Files\Git\bin\bash.exe')
            bash = str(candidate) if candidate.is_file() else None
        if bash is None:
            bash = shutil.which('bash')
        if bash is None:
            self.skipTest('bash is unavailable on this host')
        result = subprocess.run([bash, str(ROOT / 'run-open-vlm.sh')], capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn('usage', result.stderr.lower())

    def test_checksum_mismatch_blocks_cleanup(self):
        with tempfile.TemporaryDirectory() as directory:
            artifact_dir = Path(directory)
            (artifact_dir / 'manifest.json').write_text(json.dumps({'ok': True}), encoding='utf-8')
            deleted = []
            result = safe_upload_then_delete(
                'pod-1', artifact_dir, 'remote',
                lambda *_args: {'checksum': 'wrong'}, deleted.append,
            )
            self.assertEqual(result['status'], 'CLEANUP_BLOCKED')
            self.assertEqual(deleted, [])

    def test_gpu_requirements_are_pinned(self):
        lines = [line.strip() for line in (ROOT / 'requirements-gpu.txt').read_text().splitlines()
                 if line.strip() and not line.lstrip().startswith('#')]
        self.assertGreaterEqual(len(lines), 4)
        self.assertTrue(all('==' in line for line in lines))


if __name__ == '__main__':
    unittest.main()
