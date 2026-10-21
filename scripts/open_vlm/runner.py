"""Offline-first run lifecycle for the open-VLM experiment track."""

import hashlib
import json
from datetime import datetime
from pathlib import Path

from .scoring import classify_answer, cluster_bootstrap_delta, evaluate_gate, summarize
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
        task.setdefault('task_id', f'task-{index:05}')
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
    split = manifest.get('family_split', {})
    allowed = {
        family
        for domain in ('animals', 'logos')
        for family in split.get(domain, {}).get('dev_families', [])
    }
    rows = [row for row in manifest.get('rows', []) if row.get('domain') in {'animals', 'logos'}]
    return [row for row in rows if not allowed or row.get('family') in allowed]


def run_development(manifest, backend, max_retries=0, dry_run=False) -> dict:
    rows = _development_rows(manifest)
    free_rows, free_calls, free_failures = _run_tasks(
        rows, backend, 'free_generation', max_retries=max_retries, dry_run=dry_run)
    baseline_rows, baseline_calls, baseline_failures = _run_tasks(
        rows, backend, 'candidate_scoring', max_retries=max_retries, dry_run=dry_run)
    method_rows, method_calls, method_failures = _run_tasks(
        rows, backend, 'prior_contrastive', max_retries=max_retries, dry_run=dry_run)
    if dry_run:
        return {'phase': 'development', 'status': 'DRY_RUN', 'calls': 0}
    baseline_free = summarize(free_rows)
    baseline = summarize(baseline_rows)
    method = summarize(method_rows)
    gate = evaluate_gate(baseline, method, development=True)
    failures = free_failures + baseline_failures + method_failures
    if failures or gate['status'] == 'FAIL':
        status = 'BUDGET_STOPPED'
        next_action = 'attention_logit_lens'
    else:
        status = gate['status']
        next_action = 'held_out_evaluation' if status == 'PASS' else 'review'
    return {
        'phase': 'development',
        'status': status,
        'calls': free_calls + baseline_calls + method_calls,
        'baseline_free': baseline_free,
        'baseline_candidate': baseline,
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
    free_rows, free_calls, free_failures = _run_tasks(
        rows, backend, 'free_generation', max_retries=max_retries)
    baseline_rows, baseline_calls, baseline_failures = _run_tasks(
        rows, backend, 'candidate_scoring', max_retries=max_retries)
    method_rows, method_calls, method_failures = _run_tasks(
        rows, backend, 'prior_contrastive', max_retries=max_retries)
    baseline = summarize(baseline_rows)
    method = summarize(method_rows)
    paired = _paired_deltas(baseline_rows, method_rows)
    seed = int(manifest.get('seed', 0))
    method['bias_delta_ci'] = cluster_bootstrap_delta(paired, 'bias_delta', seed)
    method['accuracy_delta_ci'] = cluster_bootstrap_delta(paired, 'accuracy_delta', seed + 1)
    method['other_error_delta_ci'] = cluster_bootstrap_delta(paired, 'other_error_delta', seed + 2)
    gate = evaluate_gate(baseline, method, development=False)
    failures = free_failures + baseline_failures + method_failures
    family_count = len({row.get('family') for row in rows})
    if family_count < 20 and not failures:
        gate = {'status': 'INCONCLUSIVE', 'reasons': ['held-out composite needs at least 20 source families']}
    return {
        'phase': 'held-out',
        'status': gate['status'],
        'calls': free_calls + baseline_calls + method_calls,
        'baseline_free': summarize(free_rows),
        'baseline_candidate': baseline,
        'baseline': baseline,
        'method': method,
        'gate': gate,
        'failures': failures,
    }


def run_no_regression(manifest, backend, max_retries=0, dry_run=False) -> dict:
    sets = {
        'original': manifest.get('original_rows', []),
        'sanity': manifest.get('sanity_rows', []),
    }
    if any(not rows for rows in sets.values()):
        return {'phase': 'no-regression', 'status': 'INCONCLUSIVE',
                'reason': 'original_rows and sanity_rows are required'}
    results = {}
    for name, rows in sets.items():
        baseline_rows, _, baseline_failures = _run_tasks(
            rows, backend, 'candidate_scoring', max_retries=max_retries, dry_run=dry_run)
        method_rows, _, method_failures = _run_tasks(
            rows, backend, 'prior_contrastive', max_retries=max_retries, dry_run=dry_run)
        if dry_run:
            return {'phase': 'no-regression', 'status': 'DRY_RUN', 'calls': 0}
        paired = _paired_deltas(baseline_rows, method_rows)
        ci = cluster_bootstrap_delta(paired, 'accuracy_delta', int(manifest.get('seed', 0)))
        results[name] = {
            'baseline': summarize(baseline_rows),
            'method': summarize(method_rows),
            'accuracy_delta_ci': ci,
            'failures': baseline_failures + method_failures,
            'status': 'PASS' if ci['family_count'] >= 5 and ci['lower'] >= -0.05 else 'INCONCLUSIVE',
        }
    status = 'PASS' if all(result['status'] == 'PASS' for result in results.values()) else 'INCONCLUSIVE'
    return {'phase': 'no-regression', 'status': status, 'sets': results}


def _paired_deltas(baseline_rows, method_rows):
    baseline_by_task = {row['task_id']: row for row in baseline_rows}
    paired = []
    for method_row in method_rows:
        baseline_row = baseline_by_task.get(method_row['task_id'])
        if baseline_row is None:
            continue
        baseline_label = classify_answer(baseline_row.get('prediction'), baseline_row)
        method_label = classify_answer(method_row.get('prediction'), method_row)
        paired.append({
            'family': method_row['family'],
            'bias_delta': float(baseline_label == 'Bias-aligned') - float(method_label == 'Bias-aligned'),
            'accuracy_delta': float(method_label == 'Correct') - float(baseline_label == 'Correct'),
            'other_error_delta': float(method_label == 'Other errors') - float(baseline_label == 'Other errors'),
        })
    if not paired:
        raise ValueError('Held-out evaluation requires paired baseline/method rows')
    return paired


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
