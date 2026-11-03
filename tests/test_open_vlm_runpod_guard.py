import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))

from open_vlm_runpod_guard import monitor_pod  # noqa: E402


class OpenVlmRunpodGuardTests(unittest.TestCase):
    def test_guard_returns_when_stopped_pod_remains_in_list(self):
        calls = []

        def cli(_path, *args):
            calls.append(args)
            if args == ('pod', 'list'):
                return [{'id': 'pod-1', 'runtimeStatus': 'stopped'}]
            raise AssertionError(args)

        with tempfile.TemporaryDirectory() as directory:
            result = monitor_pod(
                cli_path='runpodctl', pod_id='pod-1', balance_before_usd=12.0,
                spend_cap_usd=3.0, timeout_seconds=3600, poll_seconds=30,
                log_path=Path(directory) / 'watchdog.jsonl', cli=cli,
                clock=lambda: 0.0, sleep=lambda _seconds: self.fail('stopped pod should return immediately'))

        self.assertEqual(result['status'], 'STOP_CONFIRMED')
        self.assertEqual(calls, [('pod', 'list')])

    def test_guard_stops_pod_when_account_balance_delta_reaches_cap(self):
        calls = []
        active = [True]
        sleeps = []

        def cli(_path, *args):
            calls.append(args)
            if args == ('pod', 'list'):
                return [{'id': 'pod-1'}] if active[0] else []
            if args == ('user',):
                return {'clientBalance': 9.5}
            if args == ('pod', 'stop', 'pod-1'):
                active[0] = False
                return {'id': 'pod-1', 'desiredStatus': 'EXITED'}
            raise AssertionError(args)

        with tempfile.TemporaryDirectory() as directory:
            log_path = Path(directory) / 'watchdog.jsonl'
            result = monitor_pod(
                cli_path='runpodctl', pod_id='pod-1', balance_before_usd=12.5,
                spend_cap_usd=3.0, timeout_seconds=3600, poll_seconds=30,
                log_path=log_path, cli=cli, clock=lambda: 0.0,
                sleep=sleeps.append)
            events = [json.loads(line) for line in log_path.read_text(encoding='utf-8').splitlines()]

        self.assertEqual(result['status'], 'STOP_CONFIRMED')
        self.assertEqual(sleeps, [30])
        self.assertIn(('pod', 'stop', 'pod-1'), calls)
        self.assertFalse(any(args[0] == 'pod' and args[1] == 'delete' for args in calls))
        self.assertEqual(events[-2]['event'], 'stop_submitted')
        self.assertEqual(events[-2]['reason'], 'spend_cap')
        self.assertEqual(events[-1]['event'], 'pod_inactive')

    def test_guard_stops_active_pod_at_registered_timeout(self):
        calls = []
        now = [0.0]
        active = [True]

        def cli(_path, *args):
            calls.append(args)
            if args == ('pod', 'list'):
                return [{'id': 'pod-1'}] if active[0] else []
            if args == ('user',):
                return {'clientBalance': 12.5}
            if args == ('pod', 'stop', 'pod-1'):
                active[0] = False
                return {'id': 'pod-1', 'desiredStatus': 'EXITED'}
            raise AssertionError(args)

        with tempfile.TemporaryDirectory() as directory:
            log_path = Path(directory) / 'watchdog.jsonl'
            result = monitor_pod(
                cli_path='runpodctl', pod_id='pod-1', balance_before_usd=12.5,
                spend_cap_usd=3.0, timeout_seconds=60, poll_seconds=60,
                log_path=log_path, cli=cli, clock=lambda: now[0],
                sleep=lambda seconds: now.__setitem__(0, now[0] + seconds))

        self.assertEqual(result['status'], 'STOP_CONFIRMED')
        self.assertIn(('pod', 'stop', 'pod-1'), calls)

    def test_guard_keeps_retrying_stop_until_runpod_confirms_it(self):
        stop_attempts = 0
        sleeps = []
        active = [True]

        def cli(_path, *args):
            nonlocal stop_attempts
            if args == ('pod', 'list'):
                return [{'id': 'pod-1'}] if active[0] else []
            if args == ('user',):
                return {'clientBalance': 9.0}
            if args == ('pod', 'stop', 'pod-1'):
                stop_attempts += 1
                if stop_attempts < 4:
                    raise RuntimeError('temporary API error')
                active[0] = False
                return {'id': 'pod-1', 'desiredStatus': 'EXITED'}
            raise AssertionError(args)

        with tempfile.TemporaryDirectory() as directory:
            result = monitor_pod(
                cli_path='runpodctl', pod_id='pod-1', balance_before_usd=12.5,
                spend_cap_usd=3.0, timeout_seconds=3600, poll_seconds=30,
                log_path=Path(directory) / 'watchdog.jsonl', cli=cli,
                clock=lambda: 0.0, sleep=sleeps.append)

        self.assertEqual(result['status'], 'STOP_CONFIRMED')
        self.assertEqual(stop_attempts, 4)
        self.assertEqual(sleeps, [30, 30, 30, 30])

    def test_guard_does_not_repeat_successful_stop_while_pod_is_shutting_down(self):
        list_calls = 0
        stop_calls = 0
        sleeps = []

        def cli(_path, *args):
            nonlocal list_calls, stop_calls
            if args == ('pod', 'list'):
                list_calls += 1
                return [{'id': 'pod-1'}] if list_calls < 4 else []
            if args == ('user',):
                return {'clientBalance': 9.0}
            if args == ('pod', 'stop', 'pod-1'):
                stop_calls += 1
                return {'id': 'pod-1', 'desiredStatus': 'EXITED'}
            raise AssertionError(args)

        with tempfile.TemporaryDirectory() as directory:
            result = monitor_pod(
                cli_path='runpodctl', pod_id='pod-1', balance_before_usd=12.0,
                spend_cap_usd=3.0, timeout_seconds=3600, poll_seconds=30,
                log_path=Path(directory) / 'watchdog.jsonl', cli=cli,
                clock=lambda: 0.0, sleep=sleeps.append)

        self.assertEqual(result['status'], 'STOP_CONFIRMED')
        self.assertEqual(stop_calls, 1)
        self.assertEqual(sleeps, [30, 30, 30])
