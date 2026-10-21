"""Offline-first run lifecycle for the open-VLM experiment track."""

import hashlib
import json
from datetime import datetime
from pathlib import Path

from .scoring import evaluate_gate, summarize
from .schema import build_manifest


class Backend:
    def run(self, task: dict, method: str) -> dict:
        raise NotImplementedError


def _run_tasks(rows, backend, method, limit=None, max_retries=0, dry_run=False):
    rows = list(rows[:limit] if limit is not None else rows)
    if dry_run:
        return [], 0, []
    results = []
    attempts = 0
    failures = []
    for index, row in enumerate(rows):
        task = dict(row)
        task.setdefault('task_id', f'{method}-{index:05}')
        for attempt in range(1, max_retries + 2):
            attempts += 1
            try:
                result = {
                    **task,
                    **dict(backend.run(task, method)),
                    'task_id': task['task_id'],
                    'method': method,
                    'attempt': attempt,
                }
                results.append(result)
                break
            except Exception as exc:
                if attempt > max_retries:
                    failures.append({'task_id': task['task_id'], 'method': method, 'error': str(exc)})
                    results.append({**task, 'method': method, 'attempt': attempt,
                                    'prediction': None, 'error': str(exc)})
    return results, attempts, failures


def prepare_run(root, phase, rows, config) -> Path:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    run_dir = root / f'open-vlm-{phase}'
    suffix = 2
    while run_dir.exists():
        run_dir = root / f'open-vlm-{phase}-{suffix}'
        suffix += 1
    run_dir.mkdir()
    manifest = build_manifest(rows, config)
    manifest['phase'] = phase
    manifest['created'] = datetime.now().astimezone().isoformat()
    (run_dir / 'manifest.json').write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=True), encoding='utf-8')
    (run_dir / 'state.json').write_text(json.dumps({
        'phase': phase,
        'status': 'PREPARED',
        'calls': 0,
        'scientific_status': 'NEEDS_EVIDENCE',
    }, indent=2), encoding='utf-8')
    return run_dir


def run_smoke(manifest, backend, limit=50, max_retries=0, dry_run=False) -> dict:
    rows = list(manifest.get('rows', []))
    results, calls, failures = _run_tasks(
        rows, backend, 'candidate_scoring', limit=limit,
        max_retries=max_retries, dry_run=dry_run)
    status = 'DRY_RUN' if dry_run else ('SMOKE_FAILED' if failures else 'COMPLETED')
    return {
        'phase': 'smoke',
        'status': status,
        'calls': calls,
        'attempts': calls,
        'limit': min(limit, len(rows)),
        'failures': failures,
        'results': results,
        'scientific_decision': 'NOT_EVALUATED',
    }


def _development_rows(manifest):
    return [row for row in manifest.get('rows', []) if row.get('domain') in {'animals', 'logos'}]


def run_development(manifest, backend, max_retries=0, dry_run=False) -> dict:
    rows = _development_rows(manifest)
    baseline_rows, baseline_calls, baseline_failures = _run_tasks(
        rows, backend, 'free_generation', max_retries=max_retries, dry_run=dry_run)
    method_rows, method_calls, method_failures = _run_tasks(
        rows, backend, 'prior_contrastive', max_retries=max_retries, dry_run=dry_run)
    if dry_run:
        return {'phase': 'development', 'status': 'DRY_RUN', 'calls': 0}
    baseline = summarize(baseline_rows)
    method = summarize(method_rows)
    gate = evaluate_gate(baseline, method, development=True)
    failures = baseline_failures + method_failures
    if failures or gate['status'] == 'FAIL':
        status = 'BUDGET_STOPPED'
        next_action = 'attention_logit_lens'
    else:
        status = gate['status']
        next_action = 'held_out_evaluation' if status == 'PASS' else 'review'
    return {
        'phase': 'development',
        'status': status,
        'calls': baseline_calls + method_calls,
        'baseline': baseline,
        'method': method,
        'gate': gate,
        'failures': failures,
        'next_action': next_action,
    }


def run_held_out(manifest, backend, max_retries=0, dry_run=False) -> dict:
    rows = [row for row in manifest.get('rows', []) if row.get('domain') not in {'animals', 'logos'}]
    if dry_run:
        return {'phase': 'held-out', 'status': 'DRY_RUN', 'calls': 0}
    baseline_rows, baseline_calls, baseline_failures = _run_tasks(
        rows, backend, 'free_generation', max_retries=max_retries)
    method_rows, method_calls, method_failures = _run_tasks(
        rows, backend, 'prior_contrastive', max_retries=max_retries)
    return {
        'phase': 'held-out',
        'status': 'COMPLETED' if not baseline_failures + method_failures else 'INCONCLUSIVE',
        'calls': baseline_calls + method_calls,
        'baseline': summarize(baseline_rows),
        'method': summarize(method_rows),
        'failures': baseline_failures + method_failures,
    }


def artifact_checksum(artifact_dir) -> str:
    artifact_dir = Path(artifact_dir)
    digest = hashlib.sha256()
    for path in sorted(path for path in artifact_dir.rglob('*') if path.is_file()):
        digest.update(path.relative_to(artifact_dir).as_posix().encode('utf-8'))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def safe_upload_then_delete(pod_id, artifact_dir, destination, uploader, deleter) -> dict:
    try:
        local_checksum = artifact_checksum(artifact_dir)
        response = uploader(Path(artifact_dir), destination)
        if response.get('checksum') != local_checksum:
            return {'status': 'CLEANUP_BLOCKED', 'reason': 'remote checksum mismatch'}
        deleter(pod_id)
        return {'status': 'DELETED', 'checksum': local_checksum}
    except Exception as exc:
        return {'status': 'CLEANUP_BLOCKED', 'reason': str(exc)}
