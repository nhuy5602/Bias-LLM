import json
import contextlib
import io
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))

from open_vlm.runner import safe_upload_then_delete  # noqa: E402
from open_vlm.schema import DomainSpec, execution_source_fingerprint, validate_row  # noqa: E402
import open_vlm_pilot  # noqa: E402
from open_vlm_pilot import main as open_vlm_main  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
MODEL_COMMIT = '66285546d2b821cf421d4f5eb2576359d3770cd3'


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
        self.assertIn('open_vlm_runpod_guard.py', script)
        self.assertIn('OPEN_VLM_WATCHDOG_CLI', script)
        self.assertIn('OPEN_VLM_BALANCE_BEFORE_USD', script)
        self.assertIn('OPEN_VLM_SPEND_CAP_USD', script)
        self.assertIn('no-regression', script)
        self.assertIn('logit-lens', script)
        self.assertIn('open_vlm_pilot.py', script)

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

    def test_launcher_requires_watchdog_registration_for_pod_runs(self):
        bash = None
        if os.name == 'nt':
            candidate = Path(r'C:\Program Files\Git\bin\bash.exe')
            bash = str(candidate) if candidate.is_file() else None
        if bash is None:
            bash = shutil.which('bash')
        if bash is None:
            self.skipTest('bash is unavailable on this host')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / 'run'
            run_dir.mkdir()
            command_marker = root / 'command-ran'
            command = root / 'command.sh'
            command.write_text(
                '#!/usr/bin/env bash\necho ran > "' + str(command_marker).replace('\\', '/') + '"\n',
                encoding='utf-8')
            command.chmod(0o755)
            env = os.environ.copy()
            env.update({'OPEN_VLM_POD_ID': 'pod-1', 'OPEN_VLM_COMMAND': str(command)})

            result = subprocess.run(
                [bash, str(ROOT / 'run-open-vlm.sh'), 'smoke', str(run_dir)],
                env=env, capture_output=True, text=True,
            )

        self.assertEqual(result.returncode, 2)
        self.assertIn('watchdog', result.stderr.lower())
        self.assertFalse(command_marker.exists())

    def test_launcher_dispatches_logit_lens_to_cli_when_no_override(self):
        bash = None
        if os.name == 'nt':
            candidate = Path(r'C:\Program Files\Git\bin\bash.exe')
            bash = str(candidate) if candidate.is_file() else None
        if bash is None:
            bash = shutil.which('bash')
        if bash is None:
            self.skipTest('bash is unavailable on this host')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / 'run'
            run_dir.mkdir()
            (run_dir / 'manifest.json').write_text('{}', encoding='utf-8')
            marker = root / 'argv.txt'
            python_stub = root / 'python-stub.sh'
            python_stub.write_text(
                '#!/usr/bin/env bash\nprintf "%s\\n" "$@" > "' + str(marker).replace('\\', '/') + '"\n',
                encoding='utf-8')
            python_stub.chmod(0o755)
            env = os.environ.copy()
            env.update({'OPEN_VLM_PYTHON': str(python_stub)})
            result = subprocess.run(
                [bash, str(ROOT / 'run-open-vlm.sh'), 'logit-lens', str(run_dir)],
                env=env, capture_output=True, text=True,
            )
            self.assertEqual(result.returncode, 3)
            args = marker.read_text(encoding='utf-8').splitlines()
            self.assertIn('--logit-lens', args)
            self.assertIn('--run-dir', args)
            self.assertIn(str(run_dir), args)
            self.assertIn('CLEANUP_BLOCKED', result.stderr)

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

    def test_uploader_nonzero_exit_stops_pod(self):
        git_bash = Path(r'C:\Program Files\Git\bin\bash.exe')
        bash = str(git_bash) if os.name == 'nt' and git_bash.is_file() else shutil.which('bash')
        if bash is None:
            self.skipTest('bash is unavailable on this host')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / 'run'
            run_dir.mkdir()
            (run_dir / 'manifest.json').write_text('{}', encoding='utf-8')
            uploader = root / 'uploader.sh'
            uploader.write_text('#!/usr/bin/env bash\nexit 7\n', encoding='utf-8')
            runpodctl = root / 'runpodctl.sh'
            runpodctl.write_text(
                '#!/usr/bin/env bash\n'
                'if [[ "$1" == user ]]; then echo \'{"clientBalance":12.5}\'; '
                'elif [[ "$1 $2" == "pod list" ]]; then echo \'[{"id":"pod-1"}]\'; '
                'fi\n', encoding='utf-8')
            stopper = root / 'stopper.sh'
            stopper.write_text('#!/usr/bin/env bash\necho stopped > "$(dirname "$0")/stopped"\n', encoding='utf-8')
            deleter = root / 'deleter.sh'
            deleter.write_text('#!/usr/bin/env bash\necho deleted > "$(dirname "$0")/deleted"\n', encoding='utf-8')
            for script in (uploader, runpodctl, stopper, deleter):
                script.chmod(0o755)
            env = os.environ.copy()
            env.update({
                'OPEN_VLM_ARTIFACT_DESTINATION': 'remote',
                'OPEN_VLM_UPLOADER': str(uploader),
                'OPEN_VLM_STOPPER': str(stopper),
                'OPEN_VLM_DELETER': str(deleter),
                'OPEN_VLM_POD_ID': 'pod-1',
                'OPEN_VLM_WATCHDOG_CLI': str(runpodctl),
                'OPEN_VLM_BALANCE_BEFORE_USD': '12.5',
                'OPEN_VLM_SPEND_CAP_USD': '0.1',
                'OPEN_VLM_WATCHDOG_TIMEOUT_SECONDS': '60',
                'OPEN_VLM_WATCHDOG_POLL_SECONDS': '1',
                'OPEN_VLM_PYTHON': 'python',
            })
            result = subprocess.run(
                [bash, str(ROOT / 'run-open-vlm.sh'), 'smoke', str(run_dir)],
                env=env, capture_output=True, text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertTrue((root / 'stopped').is_file())
            self.assertFalse((root / 'deleted').exists())

    def test_gpu_requirements_are_pinned(self):
        lines = [line.strip() for line in (ROOT / 'requirements-gpu.txt').read_text().splitlines()
                 if line.strip() and not line.lstrip().startswith('#')]
        self.assertGreaterEqual(len(lines), 4)
        self.assertTrue(all('==' in line for line in lines))

    def test_cli_returns_nonzero_when_logit_lens_is_inconclusive(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            (run_dir / 'manifest.json').write_text(json.dumps({
                'phase': 'development',
                'execution_blocked': False,
                'runtime': {'device': 'cpu'},
            }), encoding='utf-8')
            with patch.object(open_vlm_pilot.QwenBackend, 'from_manifest', return_value=object()):
                with patch.object(open_vlm_pilot, 'run_logit_lens', return_value={
                    'phase': 'logit-lens', 'status': 'INCONCLUSIVE', 'calls': 0,
                }):
                    with contextlib.redirect_stdout(io.StringIO()):
                        result = open_vlm_main(['--logit-lens', '--run-dir', str(run_dir)])
        self.assertEqual(result, 1)

    def test_gpu_cli_requires_explicit_cost_and_budget(self):
        with patch('open_vlm_pilot.prepare_vlmbias_run') as prepare:
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    open_vlm_main(['--prepare', '--root', 'runs', '--model-revision', MODEL_COMMIT,
                                   '--unlimited-budget'])
        prepare.assert_not_called()
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            (run_dir / 'manifest.json').write_text(json.dumps({
                'execution_blocked': False,
                'runtime': {'device': 'cuda', 'gpu_cost_usd_per_hour': 0.89},
                'budget_usd': None,
                'model': {'id': 'Qwen/Qwen2.5-VL-3B-Instruct', 'revision': MODEL_COMMIT},
            }), encoding='utf-8')
            with patch.object(open_vlm_pilot.QwenBackend, 'from_manifest') as backend:
                with contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit):
                        open_vlm_main(['--logit-lens', '--run-dir', str(run_dir)])
            backend.assert_not_called()

    def test_prepare_accepts_explicit_unlimited_budget_mode(self):
        with patch('open_vlm_pilot.prepare_vlmbias_run', return_value=Path('prepared')) as prepare:
            with contextlib.redirect_stdout(io.StringIO()):
                result = open_vlm_main([
                    '--prepare', '--root', 'runs', '--domains', 'animals', 'logos',
                    '--model-revision', MODEL_COMMIT,
                    '--max-per-domain', '20', '--calibration-max-per-domain', '5',
                    '--dataset-license', 'verified', '--model-license', 'verified',
                    '--gpu-cost-per-hour', '0.89', '--unlimited-budget',
                ])
        self.assertEqual(result, 0)
        config = prepare.call_args.args[3]
        self.assertEqual(config['budget_mode'], 'unlimited')
        self.assertIsNone(config['budget_usd'])
        self.assertEqual(config['model']['revision'], MODEL_COMMIT)

    def test_visual_icl_preparation_registers_80_gb_gpu_requirement_and_larger_sample(self):
        with patch('open_vlm_pilot.prepare_vlmbias_run', return_value=Path('prepared')) as prepare:
            with contextlib.redirect_stdout(io.StringIO()):
                result = open_vlm_main([
                    '--prepare-visual-icl', '--root', 'runs', '--model-revision', MODEL_COMMIT,
                    '--dataset-license', 'verified', '--model-license', 'verified',
                    '--gpu-cost-per-hour', '1.50', '--unlimited-budget',
                ])
        self.assertEqual(result, 0)
        self.assertEqual(prepare.call_args.args[1], 'visual-icl')
        self.assertEqual(prepare.call_args.kwargs['max_per_domain'], 40)
        self.assertEqual(prepare.call_args.kwargs['calibration_max_per_domain'], 20)
        self.assertEqual(prepare.call_args.args[3]['runtime']['min_gpu_memory_gb'], 80)

    def test_prepared_manifest_fingerprint_matches_current_execution_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / 'run'

            def prepare(_root, _phase, _domains, config, **_kwargs):
                from open_vlm.schema import build_manifest
                manifest = build_manifest([{
                    'domain': 'animals', 'family': 'animal-01', 'ground_truth': 4,
                    'expected_bias': 5, 'image_path': 'images/a.png',
                }], config)
                manifest['run_dir'] = str(run_dir)
                run_dir.mkdir()
                (run_dir / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
                return run_dir

            with patch('open_vlm_pilot.prepare_vlmbias_run', side_effect=prepare):
                with contextlib.redirect_stdout(io.StringIO()):
                    result = open_vlm_main([
                        '--prepare', '--root', str(root), '--domains', 'animals',
                        '--dataset-license', 'verified', '--model-license', 'verified',
                        '--model-revision', MODEL_COMMIT, '--gpu-cost-per-hour', '0.89',
                        '--unlimited-budget',
                    ])

            self.assertEqual(result, 0)
            manifest = json.loads((run_dir / 'manifest.json').read_text(encoding='utf-8'))
            self.assertEqual(manifest['model']['revision'], MODEL_COMMIT)
            self.assertEqual(manifest['execution_sources'], execution_source_fingerprint())

    def test_prepare_rejects_missing_model_revision(self):
        with patch('open_vlm_pilot.prepare_vlmbias_run') as prepare:
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    open_vlm_main(['--prepare', '--root', 'runs', '--gpu-cost-per-hour', '0.89',
                                   '--unlimited-budget'])
        prepare.assert_not_called()

    def test_prepare_rejects_mutable_model_revision(self):
        with patch('open_vlm_pilot.prepare_vlmbias_run') as prepare:
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    open_vlm_main(['--prepare', '--root', 'runs', '--model-revision', 'main',
                                   '--gpu-cost-per-hour', '0.89', '--unlimited-budget'])
        prepare.assert_not_called()

    def test_gpu_cli_rejects_unpinned_model_revision(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            (run_dir / 'manifest.json').write_text(json.dumps({
                'execution_blocked': False,
                'runtime': {'gpu_cost_usd_per_hour': 0.89},
                'budget_mode': 'unlimited',
                'budget_usd': None,
                'model': {'id': 'Qwen/Qwen2.5-VL-3B-Instruct'},
            }), encoding='utf-8')
            with patch.object(open_vlm_pilot.QwenBackend, 'from_manifest') as backend:
                with contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit):
                        open_vlm_main(['--logit-lens', '--run-dir', str(run_dir)])
            backend.assert_not_called()

    def test_gpu_cli_rejects_missing_execution_source_fingerprint(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            (run_dir / 'manifest.json').write_text(json.dumps({
                'execution_blocked': False,
                'runtime': {'device': 'cuda', 'gpu_cost_usd_per_hour': 0.89},
                'budget_mode': 'unlimited',
                'budget_usd': None,
                'model': {'id': 'Qwen/Qwen2.5-VL-3B-Instruct', 'revision': MODEL_COMMIT},
            }), encoding='utf-8')
            with patch.object(open_vlm_pilot.QwenBackend, 'from_manifest') as backend:
                with contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit):
                        open_vlm_main(['--logit-lens', '--run-dir', str(run_dir)])
            backend.assert_not_called()

    def test_gpu_cli_rejects_changed_execution_source(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            (run_dir / 'manifest.json').write_text(json.dumps({
                'execution_blocked': False,
                'runtime': {'device': 'cuda', 'gpu_cost_usd_per_hour': 0.89},
                'budget_mode': 'unlimited',
                'budget_usd': None,
                'model': {'id': 'Qwen/Qwen2.5-VL-3B-Instruct', 'revision': MODEL_COMMIT},
                'execution_sources': {
                    'algorithm': 'sha256', 'files': {}, 'tree_sha256': '0' * 64,
                },
            }), encoding='utf-8')
            with patch.object(open_vlm_pilot.QwenBackend, 'from_manifest') as backend:
                with contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit):
                        open_vlm_main(['--logit-lens', '--run-dir', str(run_dir)])
            backend.assert_not_called()

    def test_prepare_rejects_missing_cost_and_budget_mode(self):
        with patch('open_vlm_pilot.prepare_vlmbias_run') as prepare:
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    open_vlm_main(['--prepare', '--root', 'runs', '--model-revision', MODEL_COMMIT])
        prepare.assert_not_called()

    def test_cli_dispatches_logit_lens_phase(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            (run_dir / 'manifest.json').write_text(json.dumps({
                'phase': 'development',
                'execution_blocked': False,
                'runtime': {'device': 'cpu'},
            }), encoding='utf-8')
            with patch.object(open_vlm_pilot.QwenBackend, 'from_manifest', return_value=object()):
                with patch.object(open_vlm_pilot, 'run_logit_lens', return_value={
                    'phase': 'logit-lens', 'status': 'COMPLETED_EXPLORATORY', 'calls': 1,
                }) as run:
                    with contextlib.redirect_stdout(io.StringIO()):
                        result = open_vlm_main(['--logit-lens', '--run-dir', str(run_dir)])
        self.assertEqual(result, 0)
        run.assert_called_once()


if __name__ == '__main__':
    unittest.main()
