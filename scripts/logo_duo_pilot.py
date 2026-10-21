"""Small Codex/Gemini replication on official VLMBias logo rows."""
import argparse
import hashlib
import json
import random
import re
import time
import urllib.request
from datetime import datetime
from pathlib import Path

from collab import ROOT, executable, run_process, save

DATASET = 'anvo25/vlms-are-biased'
REVISION = '3761f9fd7163577534a7816c8bc1004035f2e2a0'
BASE_ROWS = 'https://datasets-server.huggingface.co/rows'
SEED = 20261019
REPEATS = 3
CODEX_MODEL = 'gpt-5.6-sol'
CODEX_PROVIDER = 'teamsoclo'
CODEX_REASONING = 'xhigh'
GEMINI_MODEL = 'Gemini 3.8 Flash (High)'
FAMILIES = {
    'audi-black-convertible': {'brand': 'Audi', 'element': 'overlapping circles', 'ground_truth': 4, 'bias': 5},
    'audi-black-coupe': {'brand': 'Audi', 'element': 'overlapping circles', 'ground_truth': 4, 'bias': 5},
    'audi-black-hatchback': {'brand': 'Audi', 'element': 'overlapping circles', 'ground_truth': 4, 'bias': 5},
    'audi-black-sedan': {'brand': 'Audi', 'element': 'overlapping circles', 'ground_truth': 4, 'bias': 5},
    'audi-black-suv': {'brand': 'Audi', 'element': 'overlapping circles', 'ground_truth': 4, 'bias': 5},
    'maserati-black-convertible': {'brand': 'Maserati', 'element': 'prongs', 'ground_truth': 3, 'bias': 5},
    'maserati-black-coupe': {'brand': 'Maserati', 'element': 'prongs', 'ground_truth': 3, 'bias': 5},
    'maserati-black-sedan': {'brand': 'Maserati', 'element': 'prongs', 'ground_truth': 3, 'bias': 5},
    'maserati-black-sports car': {'brand': 'Maserati', 'element': 'prongs', 'ground_truth': 3, 'bias': 5},
    'maserati-black-suv': {'brand': 'Maserati', 'element': 'prongs', 'ground_truth': 3, 'bias': 5},
}
MERCEDES_FAMILIES = {
    'mercedes-black-convertible': {'brand': 'Mercedes-Benz', 'element': 'points on the star', 'ground_truth': 3, 'bias': 4},
    'mercedes-black-coupe': {'brand': 'Mercedes-Benz', 'element': 'points on the star', 'ground_truth': 3, 'bias': 4},
    'mercedes-black-hatchback': {'brand': 'Mercedes-Benz', 'element': 'points on the star', 'ground_truth': 3, 'bias': 4},
    'mercedes-black-suv': {'brand': 'Mercedes-Benz', 'element': 'points on the star', 'ground_truth': 3, 'bias': 4},
}
MERCEDES_ALIASES = {
    'mercedes-black-convertible': ('mercedes_benz-black-convertible', 'mercedes-benz_black_convertible'),
    'mercedes-black-coupe': ('mercedes_benz-black-coupe', 'mercedes-benz_black_coupe'),
    'mercedes-black-hatchback': ('mercedes_benz-black-hatchback', 'mercedes-benz_black_hatchback'),
    'mercedes-black-suv': ('mercedes_benz-black-suv', 'mercedes-benz_black_suv'),
}


def get_families(include_mercedes=False):
    families = dict(FAMILIES)
    if include_mercedes:
        families.update(MERCEDES_FAMILIES)
    return families


def parse_count(text):
    match = re.fullmatch(r'\s*\{?\s*([0-9]+)\s*\}?\s*', str(text))
    return int(match.group(1)) if match else None


def family_from_path(path, families=None):
    name = Path(path).name.casefold()
    families = families or FAMILIES
    for family in families:
        if family in name or family.replace('-', '_') in name:
            return family
    for family, aliases in MERCEDES_ALIASES.items():
        if family in families and any(alias in name for alias in aliases):
            return family
    return None


def select_rows(rows, families=None):
    families = families or FAMILIES
    selected = []
    for family in families:
        family_rows = [r for r in rows if r.get('family') == family]
        states = {r.get('state') for r in family_rows}
        if states != {'original', 'counterfactual'}:
            raise ValueError(f'Missing original/counterfactual pair for {family}: {states}')
        selected.extend(sorted(family_rows, key=lambda r: r['state']))
    return selected


def build_tasks(rows, families=None):
    families = families or FAMILIES
    tasks = []
    for row in rows:
        config = families[row['family']]
        brand = config['brand']
        element = config['element']
        for repeat in range(1, REPEATS + 1):
            tasks.append({
                'task_id': f"{row['family']}-{row['state']}-repeat-{repeat:02}-codex",
                'family': row['family'], 'state': row['state'], 'agent': 'codex',
                'repeat': repeat, 'image_path': row['local_path'],
                'prompt': f'Count the {element} in this {brand} logo. Return one integer only.',
                'ground_truth': row['ground_truth'], 'expected_bias': row.get('expected_bias'),
            })
    random.Random(SEED).shuffle(tasks)
    return tasks


def build_gemini_prompt(rows, families=None):
    families = families or FAMILIES
    lines = [
        '# Gemini Antigravity Blind Replication', '',
        f'Required model: {GEMINI_MODEL}. If this model cannot be selected, stop before running and report the deviation instead of substituting silently.',
        'Evaluate every listed image independently. Do not read Codex results or output files.',
        'Use one model/configuration for all images. Return one integer for each exact prompt.',
        'Original/counterfactual rows are matched by brand, black color, and body style metadata, not pixel-level source identity.',
        'The 3 repeats are repeated requests on the same image, not independent samples. Treat each repeat as a fresh isolated task/context; do not use earlier answers to answer later repeats.',
        'Write results to `GEMINI-REPORT.md` in this run folder. Record model/provider, task ID, repeat, and response.',
        'Do not retry silently. Keep the scientific status at `NEEDS_EVIDENCE`.', '',
        '| Task ID | Family | State | Repeat | Image | SHA256 | Prompt |',
        '|---|---|---:|---:|---|---|---|',
    ]
    for row in rows:
        config = families[row['family']]
        brand = config['brand']
        element = config['element']
        prompt = f'Count the {element} in this {brand} logo. Return one integer only.'
        for repeat in range(1, REPEATS + 1):
            task_id = f"{row['family']}-{row['state']}-repeat-{repeat:02}-gemini"
            image_path = Path(row['local_path']).resolve()
            lines.append(f"| {task_id} | {row['family']} | {row['state']} | {repeat} | `{image_path}` | `{row['sha256']}` | {prompt} |")
    return '\n'.join(lines) + '\n'


def run_tasks(folder, tasks, timeout=240, invoke_fn=None):
    folder = Path(folder)
    state_path, results_path = folder / 'state.json', folder / 'results.json'
    state = json.loads(state_path.read_text(encoding='utf-8')) if state_path.exists() else {
        'calls': 0, 'status': 'PENDING', 'scientific_status': 'NEEDS_EVIDENCE', 'tasks': {}}
    results = json.loads(results_path.read_text(encoding='utf-8')) if results_path.exists() else []
    done = {r['task_id'] for r in results}
    invoke_fn = invoke_fn or invoke
    for task in tasks:
        if task['task_id'] in done:
            continue
        attempt = state['tasks'].get(task['task_id'], {}).get('attempts', 0) + 1
        state['calls'] += 1
        state['status'] = 'RUNNING'
        state['tasks'][task['task_id']] = {'attempts': attempt, 'status': 'RUNNING'}
        task_dir = folder / 'logs' / task['task_id'] / str(attempt)
        task_dir.mkdir(parents=True, exist_ok=True)
        (task_dir / 'prompt.txt').write_text(task['prompt'], encoding='utf-8')
        save(state_path, state)
        started = time.monotonic()
        try:
            raw = invoke_fn(task['agent'], task['prompt'], task['image_path'], task_dir, timeout)
            count = parse_count(raw)
            result = {**task, 'raw': raw, 'count': count, 'valid': count is not None, 'attempt': attempt}
            state['tasks'][task['task_id']].update(status='DONE' if count is not None else 'INVALID')
            save(task_dir / 'attempt.json', {'status': 'DONE' if count is not None else 'INVALID',
                                             'seconds': time.monotonic() - started})
        except Exception as exc:
            result = {**task, 'raw': '', 'count': None, 'valid': False, 'error': str(exc), 'attempt': attempt}
            state['tasks'][task['task_id']].update(status='FAILED', error=str(exc))
            save(task_dir / 'attempt.json', {'status': 'FAILED', 'error': str(exc),
                                             'seconds': time.monotonic() - started})
            (task_dir / 'stderr.txt').write_text(str(exc), encoding='utf-8')
        results.append(result)
        done.add(task['task_id'])
        save(results_path, results)
        save(state_path, state)
    state['status'] = ('COMPLETED' if all(state['tasks'].get(t['task_id'], {}).get('status') in ('DONE', 'INVALID')
                                          for t in tasks) else 'DEGRADED')
    save(state_path, state)
    return results, state


def fetch_json(url, retries=4):
    last = None
    for attempt in range(retries):
        try:
            request = urllib.request.Request(url, headers={'User-Agent': 'visual-prior-bias-logo-pilot/1'})
            with urllib.request.urlopen(request, timeout=60) as response:
                return json.load(response)
        except Exception as exc:
            last = exc
            if attempt + 1 < retries:
                time.sleep(2 ** attempt)
    raise RuntimeError(f'Could not fetch dataset rows: {last}')


def fetch_rows(split, offsets):
    rows = []
    for offset in offsets:
        url = (f'{BASE_ROWS}?dataset={DATASET.replace("/", "%2F")}&config=default'
               f'&revision={REVISION}&split={split}&offset={offset}&length=100')
        try:
            rows.extend(item['row'] for item in fetch_json(url).get('rows', []))
        except RuntimeError:
            continue
    return rows


def discover_rows(families=None):
    families = families or FAMILIES
    main = fetch_rows('main', (1500, 1600, 1700, 1800, 1900))
    original = fetch_rows('original', (0, 100, 200, 300, 400))
    found = []
    for row in main:
        if row.get('topic') != 'Logos' or row.get('pixel') != 384 or row.get('type_of_question') != 'Q1':
            continue
        family = family_from_path(row.get('image_path', ''), families)
        if family and family in families:
            found.append({
                'family': family, 'state': 'counterfactual', 'source_split': 'main',
                'source_id': row.get('ID'), 'source_path': row.get('image_path'),
                'image_url': row['image']['src'], 'prompt': row['prompt'],
                'ground_truth': int(row['ground_truth']),
                'expected_bias': int(row['expected_bias']), 'metadata': row.get('metadata'),
            })
    for row in original:
        if row.get('topic') != 'Logos' or row.get('sub_topic') != 'cars':
            continue
        family = family_from_path(row.get('image_path', ''), families)
        if family and family in families and row.get('type_of_question') == 'Counting/Illusion':
            found.append({
                'family': family, 'state': 'original', 'source_split': 'original',
                'source_id': row.get('ID'), 'source_path': row.get('image_path'),
                'image_url': row['image']['src'], 'prompt': row['prompt'],
                'ground_truth': int(row['ground_truth']), 'expected_bias': None,
                'metadata': row.get('metadata'),
            })
    # Keep one deterministic row per family/state.
    deduped = {}
    for row in found:
        deduped.setdefault((row['family'], row['state']), row)
    rows = list(deduped.values())
    return select_rows(rows, families)


def download_rows(folder, rows):
    image_dir = folder / 'images'
    image_dir.mkdir(parents=True, exist_ok=True)
    for row in rows:
        filename = f"{row['family']}-{row['state']}.png"
        path = image_dir / filename
        request = urllib.request.Request(row['image_url'], headers={'User-Agent': 'visual-prior-bias-logo-pilot/1'})
        with urllib.request.urlopen(request, timeout=60) as response:
            data = response.read()
        path.write_bytes(data)
        row['local_path'] = str(path)
        row['sha256'] = hashlib.sha256(data).hexdigest()


def invoke(member, prompt, image_path, attempt_dir, timeout):
    if member != 'codex':
        raise ValueError('This runner is Codex-only; Gemini runs through the generated Antigravity prompt')
    attempt_dir.mkdir(parents=True, exist_ok=True)
    answer_path = attempt_dir / 'answer.txt'
    if member == 'codex':
        command = list(executable('codex')) + [
            'exec', '--model', CODEX_MODEL, '--sandbox', 'read-only',
            '-c', 'model_provider="teamsoclo"', '-c', 'model_reasoning_effort="xhigh"',
            '-c', 'approval_policy="never"', '--ephemeral',
            '--color', 'never', '--output-last-message', str(answer_path), '--image', str(image_path), '-']
        output, error, code = run_process(command, prompt, ROOT, timeout)
        result = answer_path.read_text(encoding='utf-8') if answer_path.exists() else output
    (attempt_dir / 'stdout.txt').write_text(str(result), encoding='utf-8')
    (attempt_dir / 'stderr.txt').write_text(str(error), encoding='utf-8')
    save(attempt_dir / 'process.json', {'command': command, 'exit_code': code})
    if code != 0:
        raise RuntimeError(f'{member} exit {code}: {error or result}')
    return str(result)


def summarize(folder, manifest, results, calls):
    state = json.loads((Path(folder) / 'state.json').read_text(encoding='utf-8'))
    lines = [
        '# Logo duo pilot', '', f"Operational: {state['status']}", 'Scientific: NEEDS_EVIDENCE',
        f'Codex launches: {calls}/{len(manifest["tasks"])}', '',
        f"Protocol: exploratory official-benchmark replication; {len(manifest['families'])} car-logo source families, original vs counterfactual, {REPEATS} repeats per image.",
        f"Codex: {manifest['codex']['model']} / {manifest['codex']['provider']} / {manifest['codex']['reasoning_effort']}",
        'Original/CF rows are matched by brand, black color and body style metadata, not pixel-level source identity. Repeats are repeated requests on the same image, not independent samples.',
        'Gemini Antigravity tasks are listed separately in `GEMINI-PROMPT.md`; this runner does not call Gemini.',
        'No texture/edge-matched sham, hidden-state intervention, or causal mechanism claim.', '',
        '| State | Correct | Bias-aligned | Valid N |', '|---|---:|---:|---:|',
    ]
    for state_name in ('original', 'counterfactual'):
        subset = [r for r in results if r['state'] == state_name and r['count'] is not None]
        correct = sum(r['count'] == r['ground_truth'] for r in subset)
        biased = sum(r['count'] == r['expected_bias'] for r in subset if r['expected_bias'] is not None)
        lines.append(f'| {state_name} | {correct} | {biased} | {len(subset)} |')
    lines += ['', '| Family | State | Correct | Bias-aligned | Valid N |', '|---|---|---:|---:|---:|']
    for family in manifest['families']:
        for state_name in ('original', 'counterfactual'):
            subset = [r for r in results if r['family'] == family and r['state'] == state_name and r['count'] is not None]
            correct = sum(r['count'] == r['ground_truth'] for r in subset)
            biased = sum(r['count'] == r['expected_bias'] for r in subset if r['expected_bias'] is not None)
            lines.append(f'| {family} | {state_name} | {correct} | {biased} | {len(subset)} |')
    failures = [r for r in results if not r['valid']]
    lines += ['', f'Failed/invalid Codex tasks: {len(failures)}']
    for result in failures:
        lines.append(f"- {result['task_id']}: {result.get('error', 'invalid integer response')}")
    lines += ['', 'All model outputs are descriptive. Results do not establish causality or generalization.',
              'Scientific status remains `NEEDS_EVIDENCE`.']
    (folder / 'SUMMARY.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--include-mercedes', action='store_true',
                        help='include the four complete Mercedes-Benz black pairs')
    parser.add_argument('--run-dir', type=Path,
                        help='execute an existing prepared run directory')
    parser.add_argument('--timeout', type=int, default=240)
    args = parser.parse_args(argv)
    if args.run_dir:
        folder = args.run_dir.resolve()
        manifest_path = folder / 'manifest.json'
        if not manifest_path.is_file():
            raise SystemExit(f'Missing prepared manifest: {manifest_path}')
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        if args.dry_run:
            print(f'Prepared run: {folder}; 0 model launches')
            return 0
        results, state = run_tasks(folder, manifest['tasks'], args.timeout)
        summarize(folder, manifest, results, state['calls'])
        print(folder / 'SUMMARY.md')
        return 0
    families = get_families(args.include_mercedes)
    run_id = datetime.now().strftime('logo-duo-%Y%m%d-%H%M%S-%f')
    folder = ROOT / 'collab' / 'experiment' / run_id
    folder.mkdir(parents=True, exist_ok=False)
    rows = discover_rows(families)
    download_rows(folder, rows)
    manifest = {'protocol': 'vlmbias-logo-repeat-v3-third-brand' if args.include_mercedes else 'vlmbias-logo-repeat-v2',
                'created': datetime.now().astimezone().isoformat(), 'dataset': DATASET,
                'dataset_revision': REVISION, 'families': list(families),
                'brands': sorted({config['brand'] for config in families.values()}),
                'excluded_incomplete_families': ['mercedes-black-sedan'] if args.include_mercedes else [],
                'codex': {'model': CODEX_MODEL, 'provider': CODEX_PROVIDER,
                          'reasoning_effort': CODEX_REASONING},
                'repeats': REPEATS, 'task_order_seed': SEED,
                'family_match': 'brand + black color + body style metadata; not pixel-level source identity',
                'repeat_unit': 'same image repeated per request; not independent family samples',
                'timeout_seconds': args.timeout, 'rows': rows, 'scientific_status': 'NEEDS_EVIDENCE'}
    manifest['tasks'] = build_tasks(rows, families)
    save(folder / 'manifest.json', manifest)
    prompt_path = folder / 'GEMINI-PROMPT.md'
    prompt_path.write_text(build_gemini_prompt(rows, families), encoding='utf-8')
    print(f'Gemini prompt: {prompt_path}', flush=True)
    if args.dry_run:
        print(f'Dry run: {len(manifest["tasks"])} tasks, 0 model launches; {folder}')
        return 0
    results, state = run_tasks(folder, manifest['tasks'], args.timeout)
    summarize(folder, manifest, results, state['calls'])
    print(folder / 'SUMMARY.md')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
