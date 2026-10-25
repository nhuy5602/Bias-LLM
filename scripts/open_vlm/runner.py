"""Offline-first run lifecycle for the open-VLM experiment track."""

import hashlib
import json
from datetime import datetime
from pathlib import Path

from .diagnostics import ALPHAS, SIGMAS, build_negative_controls, materialize_image_variants, select_blur_sigma
from .dataset import infer_subject_label
from .scoring import classify_answer, cluster_bootstrap_delta, evaluate_gate, summarize, summarize_clustered
from .schema import build_manifest


class Backend:
    def run(self, task: dict, method: str) -> dict:
        raise NotImplementedError


def build_backend_task(row, sigma=3, alpha=1.0, artifact_dir=None, condition='clear', source_root=None,
                       formula='clear_minus_alpha_prior'):
    task = dict(row)
    task['image_path'] = str(row.get('image_path', '')).replace('\\', '/')
    task['candidates'] = [str(value) for value in row.get('candidate_values', range(21))]
    task['alpha'] = float(alpha)
    task['sigma'] = sigma
    task['condition'] = condition
    task['formula'] = formula
    if 'image' in row:
        task['image'] = row['image']
        task['prior_image'] = row.get('prior_image', row['image'])
        return task
    source = Path(task['image_path'])
    if source_root and not source.is_absolute():
        source = Path(source_root) / source
    if source.is_file() and artifact_dir is not None:
        variants = materialize_image_variants(source, Path(artifact_dir) / 'derived', sigma,
                                              seed=int(row.get('seed', 0)))
        task.update(variants)
        task['image'] = {
            'clear': str(source),
            'blur': variants['blur_path'],
            'black': variants['black_path'],
        }.get(condition, str(source))
        task['prior_image'] = variants['blur_path']
        task['black_image'] = variants['black_path']
        task['noise_image'] = variants['noise_path']
        task['shuffled_prior_image'] = variants['shuffled_prior_path']
    return task


def _artifact_dir(manifest):
    value = manifest.get('run_dir') or manifest.get('artifact_dir')
    return Path(value) if value else None


def _method_config(manifest):
    method = manifest.get('method', {})
    return manifest.get('source_root') or None, method.get('contrastive_formula', 'clear_minus_alpha_prior')


def _finalize_phase(manifest, phase, payload):
    artifact_dir = _artifact_dir(manifest)
    if artifact_dir is None:
        return
    _write_json(artifact_dir / 'results' / f'{phase}-summary.json', payload)
    state_path = artifact_dir / 'state.json'
    state = json.loads(state_path.read_text(encoding='utf-8')) if state_path.is_file() else {}
    state.update({
        'phase': phase,
        'phase_status': payload.get('status'),
        'scientific_status': 'NEEDS_EVIDENCE',
    })
    if phase == 'development':
        state['development_status'] = payload.get('status')
        if payload.get('selected_sigma') is not None:
            state['selected_sigma'] = payload['selected_sigma']
        if payload.get('selected_alpha') is not None:
            state['selected_alpha'] = payload['selected_alpha']
    _write_json(state_path, state)


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True, default=str), encoding='utf-8')


def _checkpoint_task(artifact_dir, method, result, attempts, failures):
    if artifact_dir is None:
        return True
    task_id = result.get('task_id', 'unknown')
    attempt = result.get('attempt', attempts)
    task_dir = artifact_dir / 'logs' / method / task_id / str(attempt)
    _write_json(task_dir / 'result.json', result)
    raw = result.get('raw')
    if raw is not None:
        (task_dir / 'raw.txt').write_text(str(raw), encoding='utf-8')
    results_path = artifact_dir / 'results' / f'{method}.jsonl'
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with results_path.open('a', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(result, sort_keys=True, ensure_ascii=True, default=str) + '\n')
    state_path = artifact_dir / 'state.json'
    state = json.loads(state_path.read_text(encoding='utf-8')) if state_path.is_file() else {}
    state.update({
        'status': 'RUNNING',
        'calls': int(state.get('calls', 0)) + 1,
        'attempts': int(state.get('attempts', 0)) + 1,
        'forward_passes': int(state.get('forward_passes', 0)) + int(result.get('forward_passes', 0) or 0),
        'cost_usd': float(state.get('cost_usd', 0.0)) + float(result.get('cost_usd', 0.0) or 0.0),
        'last_method': method,
        'last_task_id': task_id,
        'failure_count': int(state.get('failure_count', 0)) + (1 if result.get('error') else 0),
    })
    budget = state.get('budget_usd')
    if budget is not None and state['cost_usd'] > float(budget):
        state['status'] = 'BUDGET_STOPPED'
        state['budget_exceeded'] = True
    _write_json(state_path, state)
    return state.get('status') != 'BUDGET_STOPPED'


def _run_tasks(rows, backend, method, limit=None, max_retries=0, dry_run=False,
               artifact_dir=None, sigma=3, alpha=1.0, condition='clear', source_root=None,
               formula='clear_minus_alpha_prior'):
    rows = list(rows[:limit] if limit is not None else rows)
    if dry_run:
        return [], 0, []
    results = []
    attempts = 0
    failures = []
    for index, row in enumerate(rows):
        if artifact_dir is not None:
            state_path = artifact_dir / 'state.json'
            if state_path.is_file() and json.loads(state_path.read_text(encoding='utf-8')).get('status') == 'BUDGET_STOPPED':
                failures.append({'task_id': row.get('task_id', f'task-{index:05}'),
                                 'method': method, 'error': 'budget exceeded'})
                break
        task = build_backend_task(row, sigma=sigma, alpha=alpha, artifact_dir=artifact_dir,
                                  condition=condition, source_root=source_root, formula=formula)
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
                if not _checkpoint_task(artifact_dir, method, result, attempts, failures):
                    failures.append({'task_id': task['task_id'], 'method': method, 'error': 'budget exceeded'})
                    return results, attempts, failures
                break
            except Exception as exc:
                attempt_result = {**task, 'method': method, 'attempt': attempt,
                                  'prediction': None, 'error': str(exc)}
                _checkpoint_task(artifact_dir, method, attempt_result, attempts, failures)
                if attempt > max_retries:
                    failures.append({'task_id': task['task_id'], 'method': method, 'error': str(exc)})
                    result = attempt_result
                    results.append(result)
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
    manifest['run_dir'] = run_dir.as_posix()
    manifest['created'] = datetime.now().astimezone().isoformat()
    (run_dir / 'manifest.json').write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=True), encoding='utf-8')
    (run_dir / 'state.json').write_text(json.dumps({
        'phase': phase,
        'status': 'PREPARED',
        'calls': 0,
        'scientific_status': 'NEEDS_EVIDENCE',
        'cost_usd': 0.0,
        'forward_passes': 0,
        'budget_usd': manifest.get('budget_usd'),
    }, indent=2), encoding='utf-8')
    return run_dir


def run_smoke(manifest, backend, limit=50, max_retries=0, dry_run=False) -> dict:
    rows = _select_smoke_rows(manifest.get('rows', []), limit)
    artifact_dir = _artifact_dir(manifest)
    source_root, formula = _method_config(manifest)
    results, calls, failures = _run_tasks(
        rows, backend, 'candidate_scoring', max_retries=max_retries,
        dry_run=dry_run, artifact_dir=artifact_dir, source_root=source_root, formula=formula)
    status = 'DRY_RUN' if dry_run else ('SMOKE_FAILED' if failures else 'COMPLETED')
    payload = {
        'phase': 'smoke',
        'status': status,
        'calls': calls,
        'attempts': calls,
        'limit': len(rows),
        'failures': failures,
        'results': results,
        'scientific_decision': 'NOT_EVALUATED',
    }
    _finalize_phase(manifest, 'smoke', payload)
    return payload


def _select_smoke_rows(rows, limit):
    grouped = {}
    for row in sorted(rows, key=lambda item: (item.get('domain', ''), item.get('family', ''), item.get('image_path', ''))):
        grouped.setdefault(row.get('domain', ''), []).append(row)
    selected = []
    while len(selected) < limit and grouped:
        for domain in sorted(list(grouped)):
            values = grouped[domain]
            if values:
                selected.append(values.pop(0))
            if not values:
                grouped.pop(domain, None)
            if len(selected) >= limit:
                break
    return selected


def _development_rows(manifest):
    split = manifest.get('family_split', {})
    allowed = {
        family
        for domain in ('animals', 'logos')
        for family in split.get(domain, {}).get('dev_families', [])
    }
    rows = [row for row in manifest.get('rows', []) if row.get('domain') in {'animals', 'logos'}]
    return [row for row in rows if not allowed or row.get('family') in allowed]


def _step0_rows(manifest):
    if 'step0_rows' in manifest:
        return list(manifest.get('step0_rows') or [])
    return _development_rows(manifest)


def run_step0(manifest, backend, max_retries=0, dry_run=False) -> dict:
    calibration = bool(manifest.get('step0_calibration'))
    thresholds = manifest.get('step0_thresholds', {}) if calibration else {}
    selection_kwargs = {
        'min_detail_drop': float(thresholds.get('min_detail_drop', 0.0 if calibration else 0.10)),
        'require_bias_alignment': not calibration,
        'max_black_subject_validity': float(thresholds.get(
            'max_black_subject_validity', 0.20 if calibration else 0.20)),
        'min_clear_accuracy': float(thresholds.get('min_clear_accuracy', 0.0)),
    }
    preset = manifest.get('step0_records')
    if preset is not None:
        selected = select_blur_sigma(preset, **selection_kwargs)
        payload = {
            'phase': 'step0',
            'status': 'PASS' if selected['status'] == 'PASS' else 'UNSUPPORTED',
            'selected_sigma': selected.get('sigma'),
            'records': preset,
            'selection': selected,
        }
        _finalize_phase(manifest, 'step0', payload)
        return payload
    rows = _step0_rows(manifest)
    if dry_run:
        payload = {'phase': 'step0', 'status': 'DRY_RUN', 'selected_sigma': None, 'records': []}
        _finalize_phase(manifest, 'step0', payload)
        return payload
    family_count = len({(row.get('domain'), row.get('family')) for row in rows})
    min_families = int(thresholds.get('min_families', 5))
    if family_count < min_families:
        payload = {
            'phase': 'step0',
            'status': 'INCONCLUSIVE',
            'selected_sigma': None,
            'records': [],
            'reason': f'Step 0 calibration requires at least {min_families} source families; observed {family_count}',
        }
        _finalize_phase(manifest, 'step0', payload)
        return payload
    records = []
    artifact_dir = _artifact_dir(manifest)
    source_root, formula = _method_config(manifest)
    for sigma in SIGMAS:
        condition_results = {}
        for condition in ('clear', 'blur', 'black'):
            condition_rows = [
                {**row, 'task_id': f"{row.get('task_id', row.get('family', index))}-{condition}-sigma{sigma}"}
                for index, row in enumerate(rows)
            ]
            values, _, _ = _run_tasks(
                condition_rows, backend, 'step0', max_retries=max_retries,
                artifact_dir=artifact_dir, sigma=sigma, condition=condition,
                source_root=source_root, formula=formula)
            condition_results[condition] = summarize_clustered(values)
        subject_rates = {}
        for subject_condition in ('blur', 'black'):
            subject_rows = [
                {**row, 'subject_label': infer_subject_label(row, row.get('domain')),
                 'candidate_values': [infer_subject_label(row, row.get('domain')), 'other'],
                 'ground_truth': infer_subject_label(row, row.get('domain')),
                 'expected_bias': 'other',
                 'empty_image': row.get('black_image'),
                 'prompt': (
                     f"Which label best describes the visible subject? Answer exactly one of "
                     f"{{{infer_subject_label(row, row.get('domain'))}, other}}."
                 ),
                 'task_id': f"{row.get('task_id', row.get('family', index))}-subject-{subject_condition}-sigma{sigma}"}
                for index, row in enumerate(rows)
            ]
            subject_values, _, _ = _run_tasks(
                subject_rows, backend, 'subject_validity', max_retries=max_retries,
                artifact_dir=artifact_dir, sigma=sigma, condition=subject_condition,
                source_root=source_root, formula=formula)
            subject_rates[subject_condition] = (
                sum(value.get('prediction') == value.get('subject_label') for value in subject_values)
                / max(len(subject_values), 1)
            )
        clear = condition_results['clear']
        blur = condition_results['blur']
        black = condition_results['black']
        records.append({
            'sigma': sigma,
            'subject_valid_rate': subject_rates['blur'],
            'invalid_rate': blur['invalid_rate'],
            'clear_accuracy': clear['accuracy'],
            'blur_accuracy': blur['accuracy'],
            'blur_bias_rate': blur['bias_rate'],
            'black_bias_rate': black['bias_rate'],
            'black_subject_valid_rate': subject_rates['black'],
            'subject_check': 'label_recognition',
            'clear': clear,
            'blur': blur,
            'black': black,
        })
    selected = select_blur_sigma(records, **selection_kwargs)
    payload = {
        'phase': 'step0',
        'status': 'PASS' if selected['status'] == 'PASS' else 'UNSUPPORTED',
        'selected_sigma': selected.get('sigma'),
        'records': records,
        'selection': selected,
    }
    _finalize_phase(manifest, 'step0', payload)
    return payload


def run_negative_controls(manifest, backend, sigma, alpha, max_retries=0) -> dict:
    rows = _development_rows(manifest)
    artifact_dir = _artifact_dir(manifest)
    source_root, formula = _method_config(manifest)
    records = []
    failures = []
    calls = 0
    prepared = []
    for index, row in enumerate(rows):
        task = build_backend_task(row, sigma=sigma, alpha=alpha, artifact_dir=artifact_dir,
                                  source_root=source_root, formula=formula)
        if not task.get('black_image') or not task.get('noise_image'):
            continue
        size = task.get('size')
        if not size:
            continue
        prepared.append((row, task, index))
    for row, task, index in prepared:
        controls = build_negative_controls({
            'size': size,
            'prior_image_path': task.get('prior_image'),
        }, seed=int(row.get('seed', index)))
        for control in controls:
            control_task = dict(task)
            control_task['task_id'] = f"{task.get('task_id', task.get('family', index))}-control-{control['kind']}-{control.get('draw', 0)}"
            control_task['control_kind'] = control['kind']
            control_task['alpha'] = control.get('alpha') or alpha
            other_tasks = [candidate for candidate_row, candidate, _ in prepared
                           if candidate_row.get('domain') == row.get('domain')
                           and candidate_row.get('family') != row.get('family')]
            shuffled_prior = other_tasks[0].get('prior_image') if other_tasks else task['shuffled_prior_image']
            control_task['shuffled_prior_family'] = (other_tasks[0].get('family')
                                                     if other_tasks else None)
            control_task['prior_image'] = {
                'empty_black': task['black_image'],
                'noise': task['noise_image'],
                'shuffled_prior': shuffled_prior,
                'random_alpha': task['prior_image'],
            }.get(control['kind'], task['prior_image'])
            try:
                result = dict(backend.run(control_task, 'prior_contrastive_control'))
                result = {**control_task, **result}
                result.update({'task_id': control_task['task_id'], 'control_kind': control['kind'],
                               'alpha': control_task['alpha'], 'method': 'prior_contrastive_control'})
                records.append(result)
                _checkpoint_task(artifact_dir, 'prior_contrastive_control', result, calls, failures)
            except Exception as exc:
                failure = {'task_id': control_task['task_id'], 'error': str(exc),
                           'method': 'prior_contrastive_control', 'prediction': None}
                failures.append(failure)
                _checkpoint_task(artifact_dir, 'prior_contrastive_control', failure, calls, failures)
            calls += 1
    return {
        'status': 'COMPLETED' if not failures else 'DEGRADED',
        'calls': calls,
        'records': records,
        'failures': failures,
        'available': bool(records or failures),
    }


def run_development(manifest, backend, max_retries=0, dry_run=False) -> dict:
    rows = _development_rows(manifest)
    artifact_dir = _artifact_dir(manifest)
    source_root, formula = _method_config(manifest)
    step0 = run_step0(manifest, backend, max_retries=max_retries, dry_run=dry_run)
    if step0['status'] != 'PASS':
        payload = {'phase': 'development', 'status': 'GATE_STOPPED', 'calls': 0,
                   'step0': step0, 'next_action': 'attention_logit_lens'}
        _finalize_phase(manifest, 'development', payload)
        return payload
    sigma = int(step0.get('selected_sigma') or manifest.get('selected_sigma', 3))
    alpha = float(manifest.get('selected_alpha', 1.0))
    free_rows, free_calls, free_failures = _run_tasks(
        rows, backend, 'free_generation', max_retries=max_retries, dry_run=dry_run,
        artifact_dir=artifact_dir, sigma=sigma, alpha=alpha, source_root=source_root, formula=formula)
    baseline_rows, baseline_calls, baseline_failures = _run_tasks(
        rows, backend, 'candidate_scoring', max_retries=max_retries, dry_run=dry_run,
        artifact_dir=artifact_dir, sigma=sigma, alpha=alpha, source_root=source_root, formula=formula)
    self_consistency_rows, self_consistency_calls, self_consistency_failures = _run_tasks(
        [{**row, 'samples': 2, 'temperature': 0.7} for row in rows], backend,
        'self_consistency', max_retries=max_retries, dry_run=dry_run,
        artifact_dir=artifact_dir, sigma=sigma, alpha=alpha, source_root=source_root, formula=formula)
    alpha_records = []
    for alpha in ALPHAS:
        alpha_rows = [
            {**row, 'task_id': f"{row.get('task_id', row.get('family', index))}-alpha{alpha:g}"}
            for index, row in enumerate(rows)
        ]
        candidate_rows, candidate_calls, candidate_failures = _run_tasks(
            alpha_rows, backend, 'prior_contrastive', max_retries=max_retries, dry_run=dry_run,
            artifact_dir=artifact_dir, sigma=sigma, alpha=alpha, source_root=source_root, formula=formula)
        alpha_records.append({
            'alpha': alpha,
            'rows': candidate_rows,
            'calls': candidate_calls,
            'failures': candidate_failures,
        })
    if dry_run:
        payload = {'phase': 'development', 'status': 'DRY_RUN', 'calls': 0, 'step0': step0}
        _finalize_phase(manifest, 'development', payload)
        return payload
    baseline_free = summarize(free_rows)
    baseline = summarize_clustered(baseline_rows)
    ranked = []
    for record in alpha_records:
        summary = summarize_clustered(record['rows'])
        gate = evaluate_gate(baseline, summary, development=True)
        ranked.append((gate['status'] == 'PASS', baseline['bias_rate'] - summary['bias_rate'],
                       summary['accuracy'], -summary['other_error_rate'], -record['alpha'], record, summary, gate))
    _, _, _, _, _, selected_alpha, method, gate = max(ranked, key=lambda item: item[:5])
    alpha = selected_alpha['alpha']
    method_failures = [failure for record in alpha_records for failure in record['failures']]
    controls = run_negative_controls(manifest, backend, sigma, alpha, max_retries=max_retries)
    self_consistency = summarize_clustered(self_consistency_rows)
    control_summaries = {}
    for kind in sorted({record.get('control_kind') for record in controls['records']}):
        control_summaries[kind] = summarize_clustered(
            [record for record in controls['records'] if record.get('control_kind') == kind])
    comparator_bias = [baseline['bias_rate'] - summary['bias_rate'] for summary in control_summaries.values()]
    comparator_bias.append(baseline['bias_rate'] - self_consistency['bias_rate'])
    if comparator_bias:
        method['control_margin'] = (baseline['bias_rate'] - method['bias_rate']) - max(comparator_bias)
    gate = evaluate_gate(baseline, method, development=True)
    if manifest.get('require_controls') and not controls['available']:
        gate['reasons'].append('negative controls unavailable on prepared images')
        gate['status'] = 'FAIL'
    failures = free_failures + baseline_failures + self_consistency_failures + method_failures
    if failures or gate['status'] == 'FAIL':
        status = 'GATE_STOPPED'
        next_action = 'attention_logit_lens'
    else:
        status = gate['status']
        next_action = 'held_out_evaluation' if status == 'PASS' else 'review'
    payload = {
        'phase': 'development',
        'status': status,
        'calls': free_calls + baseline_calls + self_consistency_calls + sum(record['calls'] for record in alpha_records) + controls['calls'],
        'baseline_free': baseline_free,
        'baseline_candidate': baseline,
        'baseline_self_consistency': summarize_clustered(self_consistency_rows),
        'baseline': baseline,
        'method': method,
        'gate': gate,
        'failures': failures,
        'controls': controls,
        'next_action': next_action,
        'step0': step0,
        'selected_sigma': sigma,
        'selected_alpha': alpha,
        'alpha_records': [
            {key: value for key, value in record.items() if key != 'rows'}
            for record in alpha_records
        ],
    }
    _finalize_phase(manifest, 'development', payload)
    return payload


def run_held_out(manifest, backend, max_retries=0, dry_run=False) -> dict:
    artifact_dir = _artifact_dir(manifest)
    source_root, formula = _method_config(manifest)
    state = {}
    if artifact_dir is not None and (artifact_dir / 'state.json').is_file():
        state = json.loads((artifact_dir / 'state.json').read_text(encoding='utf-8'))
    development_status = manifest.get('development_status', state.get('development_status'))
    if development_status != 'PASS':
        payload = {'phase': 'held-out', 'status': 'INCONCLUSIVE',
                   'reasons': ['development gate must PASS before held-out evaluation'], 'calls': 0}
        _finalize_phase(manifest, 'held-out', payload)
        return payload
    split = manifest.get('family_split', {})
    eligible_domains = {
        domain for domain, meta in split.items()
        if domain not in {'animals', 'logos'} and meta.get('eligible')
    }
    rows = [row for row in manifest.get('rows', [])
            if row.get('domain') not in {'animals', 'logos'}
            and (not split or row.get('domain') in eligible_domains)]
    selected_sigma = int(state.get('selected_sigma', manifest.get('selected_sigma', 3)))
    selected_alpha = float(state.get('selected_alpha', manifest.get('selected_alpha', 1.0)))
    if dry_run:
        payload = {'phase': 'held-out', 'status': 'DRY_RUN', 'calls': 0}
        _finalize_phase(manifest, 'held-out', payload)
        return payload
    free_rows, free_calls, free_failures = _run_tasks(
        rows, backend, 'free_generation', max_retries=max_retries, artifact_dir=artifact_dir,
        sigma=selected_sigma, alpha=selected_alpha,
        source_root=source_root, formula=formula)
    baseline_rows, baseline_calls, baseline_failures = _run_tasks(
        rows, backend, 'candidate_scoring', max_retries=max_retries, artifact_dir=artifact_dir,
        sigma=selected_sigma, alpha=selected_alpha,
        source_root=source_root, formula=formula)
    method_rows, method_calls, method_failures = _run_tasks(
        rows, backend, 'prior_contrastive', max_retries=max_retries, artifact_dir=artifact_dir,
        sigma=selected_sigma, alpha=selected_alpha,
        source_root=source_root, formula=formula)
    baseline = summarize_clustered(baseline_rows)
    method = summarize_clustered(method_rows)
    paired = _paired_deltas(baseline_rows, method_rows)
    seed = int(manifest.get('seed', 0))
    method['bias_delta_ci'] = cluster_bootstrap_delta(paired, 'bias_delta', seed)
    method['accuracy_delta_ci'] = cluster_bootstrap_delta(paired, 'accuracy_delta', seed + 1)
    method['other_error_delta_ci'] = cluster_bootstrap_delta(paired, 'other_error_delta', seed + 2)
    gate = evaluate_gate(baseline, method, development=False)
    failures = free_failures + baseline_failures + method_failures
    family_count = len({row.get('family') for row in rows})
    if family_count < 20:
        gate = {'status': 'INCONCLUSIVE',
                'reasons': [f'held-out composite needs at least 20 source families; observed {family_count}',
                            f'underlying gate status: {gate["status"]}', *gate.get('reasons', [])]}
    payload = {
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
    _finalize_phase(manifest, 'held-out', payload)
    return payload


def run_no_regression(manifest, backend, max_retries=0, dry_run=False) -> dict:
    source_root, formula = _method_config(manifest)
    sets = {
        'original': manifest.get('original_rows', []),
        'sanity': manifest.get('sanity_rows', []),
    }
    if any(not rows for rows in sets.values()):
        payload = {'phase': 'no-regression', 'status': 'INCONCLUSIVE',
                   'reason': 'original_rows and sanity_rows are required'}
        _finalize_phase(manifest, 'no-regression', payload)
        return payload
    results = {}
    for name, rows in sets.items():
        baseline_rows, _, baseline_failures = _run_tasks(
            rows, backend, 'candidate_scoring', max_retries=max_retries, dry_run=dry_run,
            artifact_dir=_artifact_dir(manifest), source_root=source_root, formula=formula)
        method_rows, _, method_failures = _run_tasks(
            rows, backend, 'prior_contrastive', max_retries=max_retries, dry_run=dry_run,
            artifact_dir=_artifact_dir(manifest), source_root=source_root, formula=formula)
        if dry_run:
            payload = {'phase': 'no-regression', 'status': 'DRY_RUN', 'calls': 0}
            _finalize_phase(manifest, 'no-regression', payload)
            return payload
        paired = _paired_deltas(baseline_rows, method_rows)
        ci = cluster_bootstrap_delta(paired, 'accuracy_delta', int(manifest.get('seed', 0)))
        results[name] = {
            'baseline': summarize_clustered(baseline_rows),
            'method': summarize_clustered(method_rows),
            'accuracy_delta_ci': ci,
            'failures': baseline_failures + method_failures,
            'status': 'PASS' if ci['family_count'] >= 5 and ci['lower'] >= -0.05 else 'INCONCLUSIVE',
        }
    status = 'PASS' if all(result['status'] == 'PASS' for result in results.values()) else 'INCONCLUSIVE'
    payload = {'phase': 'no-regression', 'status': status, 'sets': results}
    _finalize_phase(manifest, 'no-regression', payload)
    return payload


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


def safe_upload_then_delete(pod_id, artifact_dir, destination, uploader, deleter, stopper=None) -> dict:
    try:
        local_checksum = artifact_checksum(artifact_dir)
        response = uploader(Path(artifact_dir), destination)
        if response.get('checksum') != local_checksum:
            if stopper is not None:
                stopper(pod_id, preserve_volume=True)
                return {'status': 'STOPPED_CLEANUP_BLOCKED', 'reason': 'remote checksum mismatch'}
            return {'status': 'CLEANUP_BLOCKED', 'reason': 'remote checksum mismatch'}
        deleter(pod_id)
        return {'status': 'DELETED', 'checksum': local_checksum}
    except Exception as exc:
        if stopper is not None:
            stopper(pod_id, preserve_volume=True)
            return {'status': 'STOPPED_CLEANUP_BLOCKED', 'reason': str(exc)}
        return {'status': 'CLEANUP_BLOCKED', 'reason': str(exc)}
