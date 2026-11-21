"""Family-disjoint visual ICL evaluation on VLMBias animal counting."""

from collections import defaultdict
import hashlib
import json
from itertools import permutations
from pathlib import Path
import random

from .qwen_backend import VISUAL_ICL_MODEL_ID
from .runner import _finalize_phase, _run_tasks
from .schema import DATASET_REVISION, SCIENTIFIC_STATUS, split_families
from .scoring import classify_answer, cluster_bootstrap_delta, evaluate_gate, summarize_clustered

PHASE = 'visual-icl'
MIN_FAMILIES = 40
CONTEXT_COUNT = 4
DEMONSTRATIONS_PER_LABEL = 2
COUNTERFACTUAL_ARMS = ('zero_shot', 'answer_space_control', 'visual_icl', 'shuffled_labels')
ORIGINAL_ARMS = ('zero_shot', 'visual_icl', 'shuffled_labels')
PROMPT = 'How many legs are visible on this animal? Return only the integer.'
ANSWER_SPACE_PROMPT = f'{PROMPT} The answer is either 3 or 5.'


def stratified_split(rows, seed):
    """Split each registered counterfactual label evenly by source family."""
    splits = [split_families([row for row in rows if row['ground_truth'] == label], seed, 0.5)
              for label in (3, 5)]
    return {f'{subset}_families': sorted(row['family'] for split in splits for row in split[subset])
            for subset in ('dev', 'same_domain_test')}


def build_visual_icl_contexts(rows, dev_families, seed):
    """Create label-balanced, distinct, ordered visual demonstration contexts."""
    dev_families = set(dev_families)
    by_label = defaultdict(list)
    for row in sorted(rows, key=lambda row: row['family']):
        if row['family'] in dev_families and row['ground_truth'] != row['expected_bias']:
            by_label[row['ground_truth']].append(row['family'])
    required = CONTEXT_COUNT * DEMONSTRATIONS_PER_LABEL
    labels = (3, 5)
    if any(len(set(by_label[label])) < required for label in labels):
        raise ValueError(f'Visual ICL needs {required} distinct development families for each answer label')

    rng = random.Random(seed)
    pools = {label: sorted(set(by_label[label])) for label in labels}
    for values in pools.values():
        rng.shuffle(values)
    orders = list(set(permutations((3, 3, 5, 5))))
    rng.shuffle(orders)
    contexts = []
    for index, order in enumerate(orders[:CONTEXT_COUNT]):
        selected = {label: [pools[label].pop() for _ in range(DEMONSTRATIONS_PER_LABEL)]
                    for label in labels}
        ordered_families = []
        for label in order:
            ordered_families.append(selected[label].pop())
        contexts.append({
            'context_id': f'context-{index + 1:02d}',
            'demo_families': ordered_families,
            'answer_labels': list(order),
        })
    return contexts


def register_visual_icl(manifest):
    rows = manifest['rows']
    split = stratified_split(rows, manifest['seed'])
    manifest['family_split']['animals'].update(split)
    manifest['visual_icl_contexts'] = build_visual_icl_contexts(
        rows, split['dev_families'], manifest['seed'])
    manifest.pop('visual_icl_demo_families', None)


def _manifest_hash(run_dir):
    return hashlib.sha256((run_dir / 'manifest.json').read_bytes()).hexdigest()


def _registered_inputs(manifest, run_dir):
    if (manifest.get('phase') != PHASE or manifest.get('dataset_revision') != DATASET_REVISION
            or manifest.get('model', {}).get('id') != VISUAL_ICL_MODEL_ID):
        raise ValueError('Visual ICL requires its own prepared run, pinned VLMBias and registered model')
    rows = manifest['rows']
    originals = manifest['step0_rows']
    split = manifest['family_split']['animals']
    dev = set(split['dev_families'])
    held_out = set(split['same_domain_test_families'])
    by_family = {row['family']: row for row in rows}
    original_families = {row['family'] for row in originals}
    if (dev & held_out or dev | held_out != by_family.keys() or len(by_family) != len(rows)
            or len(original_families) != len(originals) or original_families & dev):
        raise ValueError('Visual ICL requires disjoint development and evaluation source families')
    if len(held_out) < MIN_FAMILIES or len(original_families) < MIN_FAMILIES:
        raise ValueError(f'Visual ICL needs at least {MIN_FAMILIES} families per evaluation split')
    if any(row['source_split'] != 'original' or row['expected_bias'] is not None for row in originals):
        raise ValueError('Visual ICL original controls must come from the original split')
    if any(row['source_split'] != 'main' or type(row['expected_bias']) is not int
           or not 0 <= row['expected_bias'] <= 20 or row['expected_bias'] == row['ground_truth']
           for row in rows):
        raise ValueError('Visual ICL counterfactual rows must have distinct truth and prior answers')

    expected_contexts = build_visual_icl_contexts(rows, dev, manifest['seed'])
    if manifest.get('visual_icl_contexts') != expected_contexts:
        raise ValueError('Visual ICL contexts differ from their registered deterministic selection')

    paths = {}
    seen_hashes = set()
    for row in rows + originals:
        if row['domain'] != 'animals' or type(row['ground_truth']) is not int or not 0 <= row['ground_truth'] <= 20:
            raise ValueError('Visual ICL requires animal counting rows with integer answers 0–20')
        relative = Path(row['image_path'])
        image = (run_dir / relative).resolve()
        if relative.is_absolute() or not image.is_relative_to(run_dir) or not image.is_file():
            raise ValueError(f'Missing or non-portable image for {row["family"]}')
        digest = hashlib.sha256(image.read_bytes()).hexdigest()
        if digest != row['source_sha256'] or digest in seen_hashes:
            raise ValueError(f'Image hash mismatch or duplicate source image for {row["family"]}')
        seen_hashes.add(digest)
        paths[(row['source_split'], row['family'])] = str(image)

    contexts = []
    for context in expected_contexts:
        demos = [{
            'family': family,
            'source_id': by_family[family]['source_id'],
            'source_sha256': by_family[family]['source_sha256'],
            'image': paths[('main', family)],
            'answer': str(by_family[family]['ground_truth']),
        } for family in context['demo_families']]
        shuffled = [{**demo, 'answer': '5' if demo['answer'] == '3' else '3'} for demo in demos]
        if any(demo['answer'] == control['answer'] for demo, control in zip(demos, shuffled, strict=True)):
            raise ValueError('Shuffled-label control must change every demonstration answer')
        contexts.append({**context, 'demonstrations': demos, 'shuffled_demonstrations': shuffled})

    counterfactual = [{
        **by_family[family], 'image': paths[('main', family)], 'prompt': PROMPT,
        'prior_answer': by_family[family]['expected_bias'], 'evaluation_split': 'counterfactual',
        'candidate_values': list(range(21)),
    } for family in sorted(held_out)]
    original = [{
        **row, 'image': paths[('original', row['family'])], 'prompt': PROMPT,
        'prior_answer': row['ground_truth'], 'evaluation_split': 'original',
        'candidate_values': list(range(21)),
    } for row in sorted(originals, key=lambda row: row['family'])]
    return counterfactual, original, contexts


def _task(row, arm, manifest_sha256, context=None):
    task = {**row, 'arm': arm, 'manifest_sha256': manifest_sha256, 'demonstrations': []}
    if arm == 'answer_space_control':
        task['prompt'] = ANSWER_SPACE_PROMPT
    if context is not None:
        task['context_id'] = context['context_id']
        task['demonstrations'] = (context['demonstrations'] if arm == 'visual_icl'
                                  else context['shuffled_demonstrations'])
    return task


def _task_matrix(counterfactual, original, contexts, manifest_sha256):
    tasks = []
    for row in counterfactual:
        tasks.extend(_task(row, arm, manifest_sha256) for arm in ('zero_shot', 'answer_space_control'))
        for context in contexts:
            tasks.append(_task(row, 'visual_icl', manifest_sha256, context))
            tasks.append(_task(row, 'shuffled_labels', manifest_sha256, context))
    for row in original:
        tasks.append(_task(row, 'zero_shot', manifest_sha256))
        for context in contexts:
            tasks.append(_task(row, 'visual_icl', manifest_sha256, context))
            tasks.append(_task(row, 'shuffled_labels', manifest_sha256, context))
    for index, task in enumerate(tasks):
        task['task_id'] = f'visual-icl-{index:04d}-{task["arm"]}'
    return tasks


def _load_recorded_tasks(run_dir, tasks, manifest_sha256):
    completed = {}
    pending = []
    for task in tasks:
        task_dir = run_dir / 'logs' / 'visual_icl' / task['task_id']
        attempts = []
        if task_dir.is_dir():
            for attempt_dir in sorted(task_dir.iterdir(), key=lambda path: int(path.name)):
                if not attempt_dir.name.isdigit():
                    continue
                result_path = attempt_dir / 'result.json'
                if result_path.is_file():
                    record = json.loads(result_path.read_text(encoding='utf-8'))
                    if record.get('manifest_sha256') != manifest_sha256:
                        raise ValueError(f'Recorded task uses a different manifest: {task["task_id"]}')
                    attempts.append((int(attempt_dir.name), record))
                else:
                    attempts.append((int(attempt_dir.name), None))
        successful = [record for _, record in attempts if record is not None and not record.get('error')]
        if successful:
            completed[task['task_id']] = successful[-1]
        else:
            pending.append({**task, '_resume_attempt': max((number for number, _ in attempts), default=0)})
    return completed, pending


def _pair_interval(reference, treatment, seed, delta, key):
    reference_by_key = {key(row): row for row in reference}
    rows = []
    for row in treatment:
        baseline = reference_by_key.get(key(row))
        if baseline is None:
            raise ValueError('Visual ICL comparison is missing a paired reference task')
        rows.append({'family': row['family'], 'delta': delta(baseline, row)})
    return cluster_bootstrap_delta(rows, 'delta', seed)


def _indicator(label):
    return lambda row: int(classify_answer(row['prediction'], row) == label)


def _summary(results, split):
    arms = COUNTERFACTUAL_ARMS if split == 'counterfactual' else ORIGINAL_ARMS
    return {arm: summarize_clustered([row for row in results
                                     if row['evaluation_split'] == split and row['arm'] == arm])
            for arm in arms}


def _evaluate(results, seed):
    payload = {}
    counterfactual = [row for row in results if row['evaluation_split'] == 'counterfactual']
    original = [row for row in results if row['evaluation_split'] == 'original']
    payload['counterfactual'] = _summary(results, 'counterfactual')
    payload['original'] = _summary(results, 'original')
    by_arm = {arm: [row for row in counterfactual if row['arm'] == arm] for arm in COUNTERFACTUAL_ARMS}
    correct = _indicator('Correct')
    bias = _indicator('Bias-aligned')
    other_error = _indicator('Other errors')
    accuracy_delta = _pair_interval(by_arm['zero_shot'], by_arm['visual_icl'], seed,
                                     lambda baseline, method: correct(method) - correct(baseline),
                                     lambda row: row['family'])
    bias_delta = _pair_interval(by_arm['zero_shot'], by_arm['visual_icl'], seed,
                                 lambda baseline, method: bias(baseline) - bias(method),
                                 lambda row: row['family'])
    other_error_delta = _pair_interval(by_arm['zero_shot'], by_arm['visual_icl'], seed,
                                        lambda baseline, method: other_error(method) - other_error(baseline),
                                        lambda row: row['family'])
    versus_shuffled = _pair_interval(by_arm['shuffled_labels'], by_arm['visual_icl'], seed,
                                      lambda control, method: correct(method) - correct(control),
                                      lambda row: (row['family'], row['context_id']))
    versus_answer_space = _pair_interval(by_arm['answer_space_control'], by_arm['visual_icl'], seed,
                                          lambda control, method: correct(method) - correct(control),
                                          lambda row: row['family'])
    original_by_arm = {arm: [row for row in original if row['arm'] == arm] for arm in ORIGINAL_ARMS}
    original_delta = _pair_interval(original_by_arm['zero_shot'], original_by_arm['visual_icl'], seed,
                                    lambda baseline, method: correct(method) - correct(baseline),
                                    lambda row: row['family'])
    gate_metrics = {
        **payload['counterfactual']['visual_icl'],
        'accuracy_delta_ci': accuracy_delta,
        'bias_delta_ci': bias_delta,
        'other_error_delta_ci': other_error_delta,
    }
    gate = evaluate_gate(payload['counterfactual']['zero_shot'], gate_metrics, development=False)
    if versus_shuffled['lower'] <= 0:
        gate['reasons'].append('Accuracy gain over shuffled labels is not positive at the lower bound')
    if versus_answer_space['lower'] <= 0:
        gate['reasons'].append('Accuracy gain over answer-space control is not positive at the lower bound')
    if original_delta['lower'] < -0.05:
        gate['reasons'].append('Original-image accuracy lower bound exceeds the 5-point loss margin')
    if gate['reasons']:
        gate['status'] = 'FAIL'
    payload['counterfactual'].update({
        'accuracy_delta_vs_zero_shot': accuracy_delta,
        'bias_improvement_vs_zero_shot': bias_delta,
        'other_error_delta_vs_zero_shot': other_error_delta,
        'accuracy_delta_vs_shuffled_labels': versus_shuffled,
        'accuracy_delta_vs_answer_space_control': versus_answer_space,
    })
    payload['original']['accuracy_delta_vs_zero_shot'] = original_delta
    payload['gate'] = gate
    return payload


def run_visual_icl(manifest, backend, run_dir, *, dry_run=False):
    run_dir = Path(run_dir).resolve()
    payload = {'phase': PHASE, 'scientific_status': SCIENTIFIC_STATUS, 'calls': 0}
    try:
        counterfactual, original, contexts = _registered_inputs(manifest, run_dir)
    except (KeyError, ValueError, OSError, TypeError) as exc:
        return {**payload, 'status': 'INCONCLUSIVE', 'reason': str(exc)}
    manifest_sha256 = _manifest_hash(run_dir)
    tasks = _task_matrix(counterfactual, original, contexts, manifest_sha256)
    payload.update(
        planned_calls=len(tasks),
        contexts=[{key: value for key, value in context.items()
                   if key not in {'demonstrations', 'shuffled_demonstrations'}} for context in contexts],
    )
    if dry_run:
        return {**payload, 'status': 'DRY_RUN'}

    try:
        completed, pending = _load_recorded_tasks(run_dir, tasks, manifest_sha256)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return {**payload, 'status': 'INCONCLUSIVE', 'reason': str(exc)}
    fresh, calls, failures = _run_tasks(
        pending, backend, 'visual_icl', artifact_dir=run_dir, record_intent=True, stop_on_error=True)
    results_by_task = {**completed, **{row['task_id']: row for row in fresh if not row.get('error')}}
    payload.update(calls=calls, completed_tasks=len(results_by_task), failures=failures)
    if failures or len(results_by_task) != len(tasks):
        payload.update(status='INCONCLUSIVE', reason='Incomplete paired inference')
    else:
        results = [results_by_task[task['task_id']] for task in tasks]
        parity_fields = ('input_token_count', 'image_encodes', 'forward_passes')
        mismatches = []
        for context in contexts:
            for row in counterfactual + original:
                paired = [result for result in results if result['family'] == row['family']
                          and result.get('context_id') == context['context_id']]
                aligned = next(result for result in paired if result['arm'] == 'visual_icl')
                shuffled = next(result for result in paired if result['arm'] == 'shuffled_labels')
                if any(not aligned.get(field) or aligned.get(field) != shuffled.get(field)
                       for field in parity_fields):
                    mismatches.append(aligned['task_id'])
        payload['compute_mismatches'] = mismatches
        if mismatches:
            payload.update(status='INCONCLUSIVE', reason='Matched arms differ in recorded input tokens or compute')
        else:
            payload.update(_evaluate(results, manifest['seed']))
            payload['status'] = payload['gate']['status']
    _finalize_phase({**manifest, 'run_dir': run_dir}, PHASE, payload)
    return payload
