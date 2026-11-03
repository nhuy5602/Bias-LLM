"""Stop an active Open-VLM pod at its spend or time ceiling."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time


def _run_cli(cli_path, *args):
    result = subprocess.run(
        [str(cli_path), *args], capture_output=True, text=True, timeout=30, check=False)
    if result.returncode:
        raise RuntimeError(f'runpodctl command failed with exit code {result.returncode}')
    return json.loads(result.stdout)


def _pod_records(payload):
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        return payload.get('pods') or []
    return []


def _pod_is_inactive(pod):
    runtime_status = str(pod.get('runtimeStatus') or '').strip().casefold()
    return runtime_status in {'stopped', 'terminated'}


def _write_event(log_path, event, **fields):
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'event': event,
        **fields,
    }
    with log_path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(record, sort_keys=True) + '\n')


def monitor_pod(*, cli_path, pod_id, balance_before_usd, spend_cap_usd,
                timeout_seconds, poll_seconds, log_path, cli=_run_cli,
                clock=time.monotonic, sleep=time.sleep):
    if not pod_id or balance_before_usd <= 0 or spend_cap_usd <= 0:
        raise ValueError('Pod ID, positive baseline balance, and positive spend cap are required')
    if timeout_seconds <= 0 or poll_seconds <= 0:
        raise ValueError('Timeout and poll interval must be positive')

    started = clock()
    failures = 0
    requested_stop_reason = None
    stop_submitted = False
    _write_event(log_path, 'watchdog_started', pod_id=pod_id,
                 balance_before_usd=balance_before_usd,
                 spend_cap_usd=spend_cap_usd, timeout_seconds=timeout_seconds)
    while True:
        elapsed = clock() - started
        stop_reason = requested_stop_reason
        try:
            pods = _pod_records(cli(cli_path, 'pod', 'list'))
            pod = next((item for item in pods if str(item.get('id')) == str(pod_id)), None)
            if pod is None or _pod_is_inactive(pod):
                status = 'STOP_CONFIRMED' if requested_stop_reason or pod is not None else 'POD_NOT_ACTIVE'
                _write_event(log_path, 'pod_inactive', pod_id=pod_id,
                             stop_reason=requested_stop_reason,
                             runtime_status=pod.get('runtimeStatus') if pod else None)
                return {'status': status, 'pod_id': pod_id,
                        'reason': requested_stop_reason}
            if stop_reason is None and elapsed >= timeout_seconds:
                stop_reason = 'timeout'
            if stop_reason is None:
                balance = float(cli(cli_path, 'user')['clientBalance'])
                spend_delta = max(0.0, float(balance_before_usd) - balance)
                _write_event(log_path, 'preflight_poll', pod_id=pod_id,
                             balance_usd=balance, balance_delta_usd=spend_delta)
                if spend_delta >= spend_cap_usd:
                    stop_reason = 'spend_cap'
            failures = 0
        except Exception as exc:
            failures += 1
            _write_event(log_path, 'api_error', pod_id=pod_id,
                         consecutive_failures=failures,
                         error_type=type(exc).__name__)
            if failures >= 3 and stop_reason is None:
                stop_reason = 'api_unavailable'
        if stop_reason and requested_stop_reason is None:
            requested_stop_reason = stop_reason

        if requested_stop_reason:
            if not stop_submitted:
                try:
                    cli(cli_path, 'pod', 'stop', str(pod_id))
                    stop_submitted = True
                    _write_event(log_path, 'stop_submitted', pod_id=pod_id,
                                 reason=requested_stop_reason, volume_preserved=True)
                except Exception as exc:
                    failures += 1
                    _write_event(log_path, 'stop_failed', pod_id=pod_id,
                                 reason=requested_stop_reason, consecutive_failures=failures,
                                 error_type=type(exc).__name__)
            sleep(poll_seconds)
            continue

        remaining = timeout_seconds - (clock() - started)
        sleep(min(poll_seconds, max(0.0, remaining)))


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--cli-path', required=True)
    parser.add_argument('--pod-id', required=True)
    parser.add_argument('--balance-before-usd', type=float, required=True)
    parser.add_argument('--spend-cap-usd', type=float, required=True)
    parser.add_argument('--timeout-seconds', type=int, default=3600)
    parser.add_argument('--poll-seconds', type=int, default=60)
    parser.add_argument('--log-path', type=Path, required=True)
    args = parser.parse_args(argv)
    result = monitor_pod(
        cli_path=args.cli_path, pod_id=args.pod_id,
        balance_before_usd=args.balance_before_usd,
        spend_cap_usd=args.spend_cap_usd,
        timeout_seconds=args.timeout_seconds, poll_seconds=args.poll_seconds,
        log_path=args.log_path)
    print(json.dumps(result, sort_keys=True))
    return 0 if result['status'] in {'POD_NOT_ACTIVE', 'STOP_CONFIRMED'} else 1


if __name__ == '__main__':
    raise SystemExit(main())
