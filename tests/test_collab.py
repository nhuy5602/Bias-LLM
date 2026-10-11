import sys
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from collab import Runner, parse_response, run_process

class SchedulerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.calls = []
    def make(self, fn, **kw):
        return Runner(self.root, fn, max_calls=kw.get('max_calls', 12),
                      retries=3, timeout=1, max_seconds=30, backoff=0)
    def timeout_invoke(self, member, prompt, folder, timeout):
        self.calls.append(member)
        raise TimeoutError('test timeout')
    def success(self, member, prompt, folder, timeout):
        self.calls.append(member)
        return 'A research proposal with hypotheses and testable predictions. ' * 4
    def test_max_three_retries_and_circuit_breaker(self):
        r = self.make(self.timeout_invoke)
        self.assertIsNone(r.task('r1-codex', 'codex', 'prompt'))
        self.assertEqual(len(self.calls), 4)
        r.task('r2-codex', 'codex', 'prompt')
        self.assertEqual(len(self.calls), 4)
    def test_global_budget_persists_across_resume(self):
        self.make(self.timeout_invoke, max_calls=2).task('r1', 'codex', 'prompt')
        self.make(self.success, max_calls=2).task('r2', 'claude', 'prompt')
        self.assertEqual(len(self.calls), 2)
    def test_completed_task_is_not_paid_twice(self):
        self.make(self.success).task('r1', 'codex', 'prompt')
        self.make(self.timeout_invoke).task('r1', 'codex', 'prompt')
        self.assertEqual(len(self.calls), 1)
    def test_changed_prompt_or_report_is_not_silently_reused(self):
        self.make(self.success).task('r1', 'codex', 'prompt')
        with self.assertRaises(ValueError):
            self.make(self.success).task('r1', 'codex', 'changed prompt')
        (self.root / 'reports/r1.md').write_text('tampered')
        with self.assertRaises(ValueError):
            self.make(self.success).task('r1', 'codex', 'prompt')
    def test_auth_error_does_not_retry(self):
        def auth(*args):
            self.calls.append('auth')
            raise RuntimeError('401 invalid API key')
        self.make(auth).task('r1', 'gemini', 'prompt')
        self.assertEqual(len(self.calls), 1)

    def test_budget_and_workspace_trust_errors_do_not_retry(self):
        for message in ('Reached maximum budget ($1)', 'folder is not trusted'):
            with tempfile.TemporaryDirectory() as d:
                calls = []
                def fail(*args):
                    calls.append(1)
                    raise RuntimeError(message)
                Runner(Path(d), fail, backoff=0).task('x', 'claude', 'p')
                self.assertEqual(len(calls), 1)
    def test_real_subprocess_stdin_and_timeout(self):
        out, err, code = run_process(
            [sys.executable, '-c', 'import sys; print(sys.stdin.read())'],
            'Unicode: tiếng Việt', self.root, 5)
        self.assertIn('tiếng Việt', out)
        self.assertEqual(code, 0)
        with self.assertRaises(TimeoutError):
            run_process([sys.executable, '-c', 'import time; time.sleep(10)'],
                        '', self.root, 0.2)
    def test_cli_error_envelopes_fail_even_with_zero_exit(self):
        with self.assertRaises(RuntimeError):
            parse_response('claude', '{"is_error":true,"result":"quota exceeded"}')
        with self.assertRaises(RuntimeError):
            parse_response('gemini', '{"error":{"message":"invalid API key"}}')
        self.assertEqual(parse_response('gemini', '{"response":"hello"}'), 'hello')

    def test_nested_error_is_clear(self):
        with self.assertRaisesRegex(RuntimeError, '^quota exceeded$'):
            parse_response('gemini', '{"error":{"message":"quota exceeded"}}')

    def test_invalid_invoke_output_and_config_fail_once(self):
        for value in (None, ValueError('invalid transport in config')):
            with tempfile.TemporaryDirectory() as d:
                calls = []
                def invoke(*a):
                    calls.append(1)
                    if isinstance(value, Exception):
                        raise value
                    return value
                r = Runner(Path(d), invoke, max_calls=1, backoff=0)
                self.assertIsNone(r.task('x','codex','p'))
                self.assertEqual(len(calls), 1)

    def test_timeout_preserves_diagnostics(self):
        with self.assertRaises(TimeoutError) as cm:
            run_process([sys.executable, '-c', 'import time; print("partial", flush=True); time.sleep(10)'],
                        '', self.root, 0.3)
        self.assertIn('partial', cm.exception.stdout)


class IntegrationTests(unittest.TestCase):

    def test_codex_adapter_preserves_provider_and_does_not_quote_keys(self):
        from unittest.mock import patch
        from collab import invoke_cli
        import os
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            (p / 'config.toml').write_text(
                'model="existing-model"\n[mcp_servers.node_repl]\ncommand="unused"\n'
                '[plugins."chrome@openai-bundled"]\nenabled=true\n', encoding='utf-8')
            (p / 'answer.txt').write_text('a validated report', encoding='utf-8')
            with patch.dict(os.environ, {'CODEX_HOME': d}), \
                 patch('collab.run_process', return_value=('', '', 0)) as proc:
                self.assertEqual(invoke_cli('codex','p',p,5), 'a validated report')
                cmd = proc.call_args.args[0]
                self.assertIn('mcp_servers.node_repl.enabled=false', cmd)
                self.assertIn('plugins.chrome@openai-bundled.enabled=false', cmd)
                self.assertNotIn('--ignore-user-config', cmd)
                self.assertNotIn('--model', cmd)

    def test_concurrent_budget_and_other_members_survive(self):
        from concurrent.futures import ThreadPoolExecutor
        import time
        with tempfile.TemporaryDirectory() as d:
            calls = []
            def invoke(member, *args):
                calls.append(member)
                if member == 'gemini':
                    raise RuntimeError('401 invalid API key')
                time.sleep(0.03)
                return 'Evidence and a falsifiable hypothesis. ' * 5
            r = Runner(Path(d), invoke, max_calls=3, backoff=0)
            with ThreadPoolExecutor(max_workers=3) as pool:
                answers = list(pool.map(lambda m: r.task(m, m, 'p'), ('codex','claude','gemini')))
            self.assertEqual(len(calls), 3)
            self.assertTrue(answers[0] and answers[1])
            self.assertIsNone(answers[2])
            self.assertIsNone(r.task('extra', 'claude', 'p'))
            self.assertEqual(len(calls), 3)

    def test_one_controller_lock(self):
        from collab import run_lock
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'controller.lock'
            with run_lock(path):
                with self.assertRaises(RuntimeError):
                    with run_lock(path):
                        pass
            with run_lock(path):
                pass

    def test_interrupted_attempt_counts_on_resume(self):
        import json
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            r = Runner(p, lambda *a: 'test report ' * 20, retries=3)
            from collab import digest, save
            r.state['calls'] = 3
            r.state['tasks']['x'] = {'attempts': 3, 'prompt_hash': digest('p'), 'status': 'RUNNING'}
            r.persist()
            r2 = Runner(p, lambda *a: (_ for _ in ()).throw(TimeoutError('timeout')), backoff=0)
            r2.task('x', 'codex', 'p')
            self.assertEqual(r2.state['calls'], 4)
            self.assertIn('codex', r2.state['unavailable'])

    def test_whole_workflow_no_network_and_resume(self):
        from unittest.mock import patch
        import collab
        import json
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / 'Problem').mkdir()
            (root / 'Problem/source.pdf').write_bytes(b'fixture')
            calls = []
            def fake_prepare(folder, files):
                (folder / 'inputs').mkdir(parents=True)
                (folder / 'inputs/context.txt').write_text('source', encoding='utf-8')
                return 'source'
            def fake_invoke(member, prompt, folder, timeout):
                calls.append(member)
                return 'A hypothesis, falsifier and experimental evidence required. ' * 10
            with patch.object(collab, 'ROOT', root), patch.object(collab, 'prepare_inputs', fake_prepare), \
                 patch.object(collab, 'invoke_cli', fake_invoke), patch.object(sys, 'argv', ['collab.py']), \
                 patch.object(collab.signal, 'signal'):
                self.assertEqual(collab.main(), 0)
                self.assertEqual(len(calls), 9)
                self.assertEqual(collab.main(), 0)
                self.assertEqual(len(calls), 9)
            runid = json.loads((root / 'collab/latest.json').read_text())['id']
            state = json.loads((root / 'collab/runs' / runid / 'state.json').read_text())
            self.assertEqual(state['scientific_status'], 'NEEDS_EVIDENCE')
            self.assertEqual(state['status'], 'COMPLETED')


if __name__ == '__main__':
    unittest.main()
