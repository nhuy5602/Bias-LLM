"""Preregistered, family-disjoint visual ICL comparison on VLMBias animals."""

import hashlib
import json
from pathlib import Path
import random

from .runner import _finalize_phase, _run_tasks
from .qwen_backend import VISUAL_ICL_MODEL_ID
from .schema import DATASET_REVISION, SCIENTIFIC_STATUS, split_families
from .scoring import classify_answer, cluster_bootstrap_delta, evaluate_gate, summarize_clustered

PHASE = 'visual-icl'
PROMPT = 'How many legs are visible on this animal? Return only the integer.'
MIN_FAMILIES = 20
ARMS = ('zero_shot', 'visual_icl', 'shuffled_labels')


def select_demo_families(rows, dev_families, seed):
    """Select two families per label from two eligible answer labels."""
    dev_families = set(dev_families)
    by_label = {}
    for row in sorted(rows, key=lambda row: row['family']):
        if row['family'] in dev_families and row['ground_truth'] != row['expected_bias']:
            by_label.setdefault(row['ground_truth'], []).append(row['family'])
    labels = sorted(label for label, families in by_label.items() if len(set(families)) >= 2)
    if len(labels) < 2:
        raise ValueError('Visual ICL needs two counterfactual demo labels with two distinct families each')
    rng = random.Random(seed)
    return [family for label in rng.sample(labels, 2)
            for family in rng.sample(by_label[label], 2)]


def stratified_split(rows, seed):
    splits = [split_families([row for row in rows if row['ground_truth'] == label], seed, 0.5)
              for label in (3, 5)]
    return {f'{subset}_families': sorted(row['family'] for split in splits for row in split[subset])
            for subset in ('dev', 'same_domain_test')}


def register_visual_icl(manifest):
    rows = manifest['rows']
    seed = manifest['seed']
    split = stratified_split(rows, seed)
    manifest['family_split']['animals'].update(split)
    manifest['visual_icl_demo_families'] = select_demo_families(rows, split['dev_families'], seed)


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
    if (dev & held_out or dev | held_out != by_family.keys()
            or len(by_family) != len(rows) or len(original_families) != len(originals)
            or original_families & dev):
        raise ValueError('Visual ICL development families must be disjoint from both evaluation splits')
    if len(held_out) < MIN_FAMILIES or len(original_families) < MIN_FAMILIES:
        raise ValueError(f'Visual ICL needs at least {MIN_FAMILIES} families per evaluation split')
    if any(row['source_split'] != 'original' or row['expected_bias'] is not None for row in originals):
        raise ValueError('Visual ICL original controls must come from the original split')
    if any(row['source_split'] != 'main' or type(row['expected_bias']) is not int
           or not 0 <= row['expected_bias'] <= 20 or row['expected_bias'] == row['ground_truth']
           for row in rows):
        raise ValueError('Visual ICL counterfactual rows must have distinct truth and prior answers')

    demo_families = select_demo_families(rows, dev, manifest['seed'])
    if manifest.get('visual_icl_demo_families') != demo_families:
        raise ValueError('Visual ICL demo families differ from the registered deterministic selection')
    seen_hashes = set()
    paths = {}
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

    demos = [{**{key: by_family[family][key] for key in ('family', 'source_id', 'source_sha256')},
              'image': paths[('main', family)], 'answer': str(by_family[family]['ground_truth'])}
             for family in demo_families]
    shuffled = [{**demo, 'answer': other['answer']}
                for demo, other in zip(demos, demos[2:] + demos[:2])]
    queries = []
    for row in [by_family[family] for family in sorted(held_out)] + originals:
        original = row['source_split'] == 'original'
        queries.append({
            **row, 'image': paths[(row['source_split'], row['family'])], 'prompt': PROMPT,
            'prior_answer': row['ground_truth'] if original else row['expected_bias'],
            'evaluation_split': 'original' if original else 'counterfactual',
            'candidate_values': list(range(21)),
        })
    return queries, ([], demos, shuffled)


def _paired_interval(reference, treatment, seed, metric):
    rows = [
        {'family': left['family'], 'delta': metric(right) - metric(left)}
        for left, right in zip(reference, treatment, strict=True)
    ]
    return cluster_bootstrap_delta(rows, 'delta', seed)


def _indicator(label):
    return lambda row: int(classify_answer(row['prediction'], row) == label)


def run_visual_icl(manifest, backend, run_dir, *, dry_run=False):
    run_dir = Path(run_dir).resolve()
    payload = {'phase': PHASE, 'scientific_status': SCIENTIFIC_STATUS, 'calls': 0}
    try:
        queries, contexts = _registered_inputs(manifest, run_dir)
    except (KeyError, ValueError, OSError, TypeError) as exc:
        return {**payload, 'status': 'INCONCLUSIVE', 'reason': str(exc)}
    payload['planned_calls'] = len(queries) * len(ARMS)
    payload['context_families'] = [demo['family'] for demo in contexts[1]]
    if dry_run:
        return {**payload, 'status': 'DRY_RUN'}

    # A run directory owns one attempt at this fixed comparison. An interrupted
    # run retains its manifest hash, intents and partial results for audit.
    claim = run_dir / 'visual-icl-execution.json'
    try:
        with claim.open('x', encoding='utf-8') as stream:
            json.dump({'manifest_sha256': hashlib.sha256((run_dir / 'manifest.json').read_bytes()).hexdigest()},
                      stream, indent=2)
    except FileExistsError:
        return {**payload, 'status': 'INCONCLUSIVE', 'reason': 'Visual ICL run already started'}

    tasks = [
        {**row, 'task_id': f'{index:05}-{arm}', 'arm': arm, 'demonstrations': context}
        for index, row in enumerate(queries)
        for arm, context in zip(ARMS, contexts, strict=True)
    ]
    results, calls, failures = _run_tasks(
        tasks, backend, 'visual_icl', artifact_dir=run_dir, record_intent=True, stop_on_error=True)
    payload.update(calls=calls, failures=failures)

    if failures or len(results) != len(tasks):
        payload.update(status='INCONCLUSIVE', reason='Incomplete paired inference')
    else:
        seed = manifest['seed']
        parity_fields = ('input_token_count', 'image_encodes', 'forward_passes')
        mismatches = [
            aligned['task_id']
            for aligned, shuffled in zip(results[1::3], results[2::3], strict=True)
            if any(not aligned.get(field) or aligned.get(field) != shuffled.get(field)
                   for field in parity_fields)
        ]
        for split in ('counterfactual', 'original'):
            by_arm = {arm: [row for row in results
                            if row['evaluation_split'] == split and row['arm'] == arm]
                      for arm in ARMS}
            summaries = {arm: summarize_clustered(arm_rows) for arm, arm_rows in by_arm.items()}
            baseline, aligned, shuffled = (by_arm[arm] for arm in ARMS)
            acc = _indicator('Correct')
            summaries['accuracy_delta_vs_zero_shot'] = _paired_interval(baseline, aligned, seed, acc)
            summaries['accuracy_delta_vs_shuffled'] = _paired_interval(shuffled, aligned, seed, acc)
            if split == 'counterfactual':
                gate_metrics = {
                    **summaries['visual_icl'],
                    'accuracy_delta_ci': summaries['accuracy_delta_vs_zero_shot'],
                    'bias_delta_ci': _paired_interval(aligned, baseline, seed, _indicator('Bias-aligned')),
                    'other_error_delta_ci': _paired_interval(baseline, aligned, seed, _indicator('Other errors')),
                }
                payload['gate'] = evaluate_gate(summaries['zero_shot'], gate_metrics, development=False)
                summaries['bias_improvement_vs_zero_shot'] = gate_metrics['bias_delta_ci']
                summaries['other_error_delta_vs_zero_shot'] = gate_metrics['other_error_delta_ci']
            payload[split] = summaries

        payload['compute_mismatches'] = mismatches
        gate = payload['gate']
        if payload['counterfactual']['accuracy_delta_vs_shuffled']['lower'] <= 0:
            gate['reasons'].append('Accuracy gain over shuffled labels is not positive at the lower bound')
        if payload['original']['accuracy_delta_vs_zero_shot']['lower'] < -0.05:
            gate['reasons'].append('Original-image accuracy lower bound exceeds the 5-point loss margin')
        if gate['reasons']:
            gate['status'] = 'FAIL'
        if mismatches:
            payload.update(status='INCONCLUSIVE', reason='Matched arms differ in recorded input tokens or compute')
        else:
            payload['status'] = gate['status']

    _finalize_phase({**manifest, 'run_dir': run_dir}, PHASE, payload)
    return payload
