import argparse
import base64
import hashlib
import json
import os
import random
import re
import shutil
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from urllib.parse import urlsplit

from collab import (ROOT, digest, executable, parse_response, run_lock, run_process,
                    save, usd_floor_cents)

DATASET = 'anvo25/vlms-are-biased'
DATASET_REVISION = '3761f9fd7163577534a7816c8bc1004035f2e2a0'
API_MODEL = 'gemini-3.8-flash'
PREVIOUS_API_MODEL = 'gemini-2.5-pro'
SELECTED_PREVIOUS_API_MODEL = 'gemini-3.1-pro-preview'
SELECTED_LATEST_API_MODEL = 'gemini-3.1-pro-preview'
REJECTED_FLASH_API_MODEL = 'gemini-2.5-flash'
SEED = 20261014
SAMPLE_SIZE = 20
MAX_CALLS = 80
MAX_RETRIES = 3
CLAUDE_MIN_REMAINING_USD = 1.0
PAGE_SIZE = 100
MAIN_ROWS = 2784
COMPOUND_NAMES = {
    'africanelephant': 'african elephant', 'africanwilddog': 'african wild dog',
    'asianelephant': 'asian elephant', 'domesticcat': 'domestic cat',
    'domesticdog': 'domestic dog', 'mountaingoat': 'mountain goat',
    'redfox': 'red fox', 'secretarybird': 'secretary bird', 'waterbuffalo': 'water buffalo',
}


def stable_digest(value):
    return digest(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':')))


def _integer(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def select_samples(candidates, count=SAMPLE_SIZE, seed=SEED):
    if count < 2 or count % 2:
        raise ValueError('Sample count must be a positive even number')
    rng = random.Random(seed)
    chosen = []
    for group, prior, actual in (('bird', 2, 3), ('mammal', 4, 5)):
        group_rows = {}
        for row in candidates:
            if (row.get('animal_group') == group and _integer(row.get('expected_bias')) == prior
                    and _integer(row.get('ground_truth')) == actual and row.get('image_url')):
                group_rows.setdefault(row['species'].casefold(), row)
        options = list(group_rows.values())
        if len(options) < count // 2:
            raise ValueError(f'Need {count // 2} valid {group} images; found {len(options)}')
        rng.shuffle(options)
        chosen.extend(options[:count // 2])
    return sorted(chosen, key=lambda row: (row['animal_group'], row['species'].casefold()))


def build_tasks(samples):
    tasks = []
    for index, sample in enumerate(samples, 1):
        for condition in ('generic', 'named'):
            for replicate in (1, 2):
                tasks.append({
                    'task_id': f'image-{index:02}-{condition}-{replicate}',
                    'sample_id': f'image-{index:02}',
                    'species': sample['species'],
                    'animal_group': sample['animal_group'],
                    'image_path': sample.get('local_path', sample.get('image_path')),
                    'condition': condition,
                    'replicate': replicate,
                    'ground_truth': int(sample['ground_truth']),
                    'expected_bias': int(sample['expected_bias']),
                })
    rng = random.Random(SEED)
    rng.shuffle(tasks)
    return tasks


def make_prompt(task):
    subject = 'this animal' if task['condition'] == 'generic' else f"this {task['species']}"
    image = Path(task['image_path']).resolve().as_posix()
    return make_custom_command(f'Count the legs of {subject}. Answer with one integer only.', image)


def bind_task_image(folder, task):
    return {**task, 'image_path': (Path(folder) / task['image_path']).resolve().as_posix()}


def make_custom_command(question, image_path):
    return f'{question}\nImage: @{{{Path(image_path).resolve().as_posix()}}}'


def parse_count(response):
    match = re.fullmatch(r'\s*\{?([0-9]+)\}?\s*', str(response))
    return int(match.group(1)) if match else None


def image_extension(content_type, image_bytes):
    if image_bytes.startswith(b'\xff\xd8'):
        return '.jpg'
    if image_bytes.startswith(b'\x89PNG\r\n\x1a\n'):
        return '.png'
    return {'image/jpeg': '.jpg', 'image/png': '.png'}.get(content_type)


def species_from_row(row):
    dataset_id = row.get('ID', '')
    match = re.fullmatch(r'animal_\d+_(.+)_notitle_Q1_px384', dataset_id)
    if match:
        key = match.group(1).replace('_', '').casefold()
        return COMPOUND_NAMES.get(key, match.group(1).replace('_', ' '))
    stem = Path(row.get('image_path', dataset_id)).stem
    match = re.fullmatch(r'(.+)_\d+_\d+_384', stem)
    if match:
        return match.group(1).replace('_', ' ')
    return None


def format_response(response):
    count = parse_count(response)
    if count is None:
        raise RuntimeError(f'Gemini response is not a single integer: {str(response)[:300]}')
    return json.dumps({'response': str(response), 'parsed_count': count,
                       'validation': 'single-integer output accepted for the animal leg-count task'},
                      ensure_ascii=False)


def build_api_request(prompt, image_bytes, mime_type):
    return {'contents': [{'role': 'user', 'parts': [
        {'text': prompt},
        {'inlineData': {'mimeType': mime_type,
                        'data': base64.b64encode(image_bytes).decode('ascii')}}]}]}


def parse_api_response(payload):
    if payload.get('error'):
        error = payload['error']
        raise RuntimeError(error.get('message', json.dumps(error, ensure_ascii=False)))
    try:
        return ''.join(part['text'] for part in payload['candidates'][0]['content']['parts']
                       if 'text' in part).strip()
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError('Gemini API response has no candidate text') from exc


def api_failure_summary(error):
    text = str(error).casefold()
    if 'no longer available to new users' in text or 'model not found' in text:
        return 'Gemini status: provider rejected the selected model as unavailable to new users.'
    if 'quota' in text or 'billing' in text:
        return 'Gemini status: provider quota/billing limit reached; no further retries issued.'
    if 'high demand' in text:
        return 'Gemini status: provider reporting high demand; retry allowance exhausted for the active task.'
    return 'Gemini status: unavailable after provider errors; retry allowance exhausted for the active task.'


def generate_api(prompt, image_path, attempt, timeout, model=API_MODEL, api_key=None):
    api_key = api_key or os.environ.get('GEMINI_API_KEY') or os.environ.get('GOOGLE_API_KEY')
    if not api_key:
        raise RuntimeError('GEMINI_API_KEY or GOOGLE_API_KEY is not available in this process')
    path = Path(image_path)
    mime_type = {'.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.png': 'image/png'}.get(path.suffix.lower())
    if not mime_type:
        raise ValueError(f'Unsupported pilot image type: {path.suffix}')
    request_body = build_api_request(prompt, path.read_bytes(), mime_type)
    request_text = json.dumps(request_body, ensure_ascii=False)
    (attempt / 'api-request.json').write_text(json.dumps({
        'model': model, 'endpoint': f'https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent',
        'request': request_body,
    }, ensure_ascii=False), encoding='utf-8')
    url = f'https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent'
    request = urllib.request.Request(url, data=request_text.encode('utf-8'), method='POST', headers={
        'Content-Type': 'application/json', 'x-goog-api-key': api_key,
    })
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as exc:
        try:
            payload = json.loads(exc.read().decode('utf-8'))
            message = parse_api_response(payload)
            if 'no longer available to new users' in message.casefold() or 'model not found' in message.casefold():
                raise ValueError(message) from exc
            raise RuntimeError(message) from exc
        except (json.JSONDecodeError, UnicodeDecodeError):
            raise RuntimeError(f'Gemini API HTTP {exc.code}') from exc
    (attempt / 'api-response.json').write_text(json.dumps(payload, ensure_ascii=False), encoding='utf-8')
    return parse_api_response(payload)


def _request_json(url, retries=3):
    for attempt in range(retries + 1):
        try:
            request = urllib.request.Request(url, headers={'User-Agent': 'visual-prior-bias-pilot/1'})
            with urllib.request.urlopen(request, timeout=45) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            if exc.code not in (429, 500, 502, 503, 504) or attempt == retries:
                raise
        except (TimeoutError, urllib.error.URLError):
            if attempt == retries:
                raise
        time.sleep(2 ** attempt)
    raise RuntimeError('Dataset request retries exhausted')


def dataset_candidates():
    rows = []
    for offset in range(0, MAIN_ROWS, PAGE_SIZE):
        url = (f'https://datasets-server.huggingface.co/rows?dataset={DATASET.replace("/", "%2F")}'
               f'&config=default&split=main&offset={offset}&length={PAGE_SIZE}')
        rows.extend(_request_json(url)['rows'])

    candidates = {}
    for item in rows:
        row = item['row']
        if row.get('topic') != 'Animals' or row.get('type_of_question') != 'Q1' or row.get('pixel') != 384:
            continue
        actual, prior = _integer(row.get('ground_truth')), _integer(row.get('expected_bias'))
        if (actual, prior) not in ((3, 2), (5, 4)):
            continue
        species = species_from_row(row)
        image_url = row.get('image', {}).get('src')
        if not species or not image_url:
            continue
        group = 'bird' if actual == 3 else 'mammal'
        candidates.setdefault(species.casefold(), {
            'species': species,
            'animal_group': group,
            'ground_truth': str(actual),
            'expected_bias': str(prior),
            'image_url': image_url,
            'dataset_id': row.get('ID'),
            'image_path': row.get('image_path'),
        })
    return list(candidates.values())


def prepare_run(folder):
    manifest_path = folder / 'manifest.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        if manifest.get('protocol') == 'semantic-label-cf-v1':
            migrate_v1_run(folder, manifest)
            manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        if manifest.get('protocol') == 'semantic-label-cf-v2':
            migrate_v2_run(folder, manifest)
            manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        if manifest.get('protocol') == 'semantic-label-cf-v3':
            migrate_v3_api_run(folder, manifest)
            manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        if manifest.get('protocol') == 'semantic-label-cf-v4':
            migrate_v4_model_run(folder, manifest)
            manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        if manifest.get('protocol') == 'semantic-label-cf-v5':
            migrate_v5_model_run(folder, manifest)
            manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        if manifest.get('protocol') == 'semantic-label-cf-v6':
            migrate_v6_model_run(folder, manifest)
            manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        if manifest.get('protocol') == 'semantic-label-cf-v7':
            migrate_v7_model_run(folder, manifest)
            manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        if manifest.get('protocol') == 'semantic-label-cf-v8':
            migrate_v8_model_run(folder, manifest)
            manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        if manifest.get('protocol') == 'semantic-label-cf-v9':
            migrate_v9_model_run(folder, manifest)
            manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        if manifest.get('protocol') == 'semantic-label-cf-v10':
            model, endpoint = claude_configuration()
            migrate_v10_claude_run(folder, manifest, model=model, endpoint=endpoint)
            manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        if manifest.get('protocol') == 'semantic-label-cf-v11':
            migrate_v11_label_run(folder, manifest)
            manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        if manifest.get('protocol') != 'semantic-label-cf-v12':
            raise ValueError('Existing pilot manifest has a different protocol')
        if manifest.get('migration', {}).get('from') == 'semantic-label-cf-v1':
            repair_interrupted_migration_hash(folder, manifest)
            manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        if stable_digest({k: v for k, v in manifest.items() if k not in ('created', 'protocol_hash')}) != manifest.get('protocol_hash'):
            raise ValueError('Pilot manifest integrity check failed')
        for sample in manifest['samples']:
            image = folder / sample['local_path']
            if not image.is_file() or hashlib.sha256(image.read_bytes()).hexdigest() != sample['image_sha256']:
                raise ValueError(f"Pilot image integrity check failed: {sample['local_path']}")
        return manifest

    if not (folder / 'samples.json').exists():
        candidates = select_samples(dataset_candidates())
        samples_dir = folder / 'images'
        samples_dir.mkdir(parents=True, exist_ok=True)
        samples = []
        for index, row in enumerate(candidates, 1):
            request = urllib.request.Request(row['image_url'], headers={'User-Agent': 'visual-prior-bias-pilot/1'})
            with urllib.request.urlopen(request, timeout=45) as response:
                image_bytes = response.read()
                content_type = response.headers.get_content_type()
            extension = image_extension(content_type, image_bytes)
            if not extension:
                raise ValueError(f"Unsupported dataset image type {content_type}: {row['dataset_id']}")
            local_path = f'images/image-{index:02}{extension}'
            (folder / local_path).write_bytes(image_bytes)
            samples.append({**{k: row[k] for k in ('species', 'animal_group', 'ground_truth', 'expected_bias',
                                                      'dataset_id', 'image_path')},
                            'local_path': local_path,
                            'image_sha256': hashlib.sha256(image_bytes).hexdigest()})
        save(folder / 'samples.json', samples)
    samples = json.loads((folder / 'samples.json').read_text(encoding='utf-8'))
    manifest = {
        'protocol': 'semantic-label-cf-v10',
        'created': datetime.now().astimezone().isoformat(),
        'dataset': DATASET,
        'dataset_revision': DATASET_REVISION,
        'sampling_seed': SEED,
        'model': API_MODEL,
        'model_selection': 'gemini-3.8-flash, explicitly selected by the user; API model catalog confirms generateContent support.',
        'transport': 'Gemini Developer API v1beta generateContent',
        'authentication': 'GEMINI_API_KEY/GOOGLE_API_KEY environment header; never written to artifacts',
        'gemini_cli': subprocess_version(),
        'calls_max': MAX_CALLS,
        'max_retries_per_task': MAX_RETRIES,
        'samples': samples,
        'tasks': build_tasks(samples),
    }
    manifest['protocol_hash'] = stable_digest({k: v for k, v in manifest.items() if k != 'created'})
    save(manifest_path, manifest)
    return manifest


def migrate_v1_run(folder, manifest):
    state_path = folder / 'state.json'
    state = json.loads(state_path.read_text(encoding='utf-8')) if state_path.exists() else None
    if state is None or state.get('calls', 0) < 1 or state.get('pilot_protocol_hash') != manifest.get('protocol_hash'):
        raise ValueError('Only the hash-matched interrupted v1 pilot can be migrated safely')
    if stable_digest({k: v for k, v in manifest.items() if k not in ('created', 'protocol_hash')}) != manifest.get('protocol_hash'):
        raise ValueError('Cannot migrate a pilot manifest that fails its integrity check')
    running = [task_id for task_id, task in state.get('tasks', {}).items()
               if task.get('status') == 'RUNNING' and task.get('attempts') == 1]
    if len(running) != 1:
        raise ValueError('Expected one interrupted malformed prompt task in the v1 ledger')
    old_id = running[0]
    if (folder / 'reports' / f'{old_id}.md').exists():
        raise ValueError('Interrupted v1 task has a saved report; refusing to supersede it')
    old_manifest = folder / 'manifest-v1.json'
    if old_manifest.exists():
        raise ValueError('Pilot v1 migration snapshot already exists')
    shutil.copy2(folder / 'manifest.json', old_manifest)
    for sample in manifest['samples']:
        species = species_from_row({'ID': sample['dataset_id'], 'image_path': sample['image_path']})
        if not species:
            raise ValueError(f"Cannot derive species from dataset row: {sample['dataset_id']}")
        sample['species'] = species
    replacements, superseded = [], []
    for task in manifest['tasks']:
        if task['task_id'] != old_id:
            replacements.append(task)
            continue
        superseded.append(task)
        corrected = dict(task, task_id=f'{old_id}-corrected')
        corrected['species'] = next(s['species'] for s in manifest['samples']
                                    if s['local_path'] == task['image_path'])
        replacements.append(corrected)
    if len(superseded) != 1:
        raise ValueError('Interrupted task is missing from the v1 manifest')
    state['tasks'][old_id].update(status='SUPERSEDED',
        error='Malformed species name parsed from image filename; launch retained in ledger.')
    manifest['tasks'] = replacements
    manifest['superseded_tasks'] = superseded
    manifest['protocol'] = 'semantic-label-cf-v2'
    manifest['migration'] = {'from': 'semantic-label-cf-v1', 'superseded_task': old_id,
                             'reason': 'Species names are parsed from stable dataset IDs.'}
    manifest['protocol_hash'] = stable_digest({k: v for k, v in manifest.items()
                                               if k not in ('created', 'protocol_hash')})
    state['pilot_protocol_hash'] = manifest['protocol_hash']
    save(folder / 'manifest.json', manifest)
    save(state_path, state)


def migrate_v2_run(folder, manifest):
    state_path = folder / 'state.json'
    state = json.loads(state_path.read_text(encoding='utf-8')) if state_path.exists() else None
    if (state is None or state.get('calls', 0) > 5
            or state.get('pilot_protocol_hash') != manifest.get('protocol_hash')
            or state.get('unavailable', {}).get('gemini') != 'CLI timeout after 240.0s'):
        raise ValueError('Only the verified, failed image-path v2 run can be migrated')
    if stable_digest({k: v for k, v in manifest.items() if k not in ('created', 'protocol_hash')}) != manifest.get('protocol_hash'):
        raise ValueError('Cannot migrate a v2 manifest that fails its integrity check')
    failed = [(task_id, record) for task_id, record in state.get('tasks', {}).items()
              if record.get('status') == 'FAILED' and record.get('attempts') == 4]
    if len(failed) != 1:
        raise ValueError('Expected exactly one exhausted v2 image-path task')
    failed_id, record = failed[0]
    if not record or record.get('status') != 'FAILED' or (folder / 'reports' / f'{failed_id}.md').exists():
        raise ValueError('The v2 failed task is not eligible for transparent recovery')
    snapshot = folder / 'manifest-v2.json'
    if snapshot.exists():
        raise ValueError('Pilot v2 migration snapshot already exists')
    shutil.copy2(folder / 'manifest.json', snapshot)
    record.update(status='SUPERSEDED',
                  error='Image path resolved outside the pilot run directory; attempts retained in ledger.')
    replacement_id = f'{failed_id}-path-fixed'
    tasks, superseded = [], []
    for task in manifest['tasks']:
        if task['task_id'] == failed_id:
            superseded.append(task)
            task = dict(task, task_id=replacement_id)
        tasks.append(task)
    if not any(task['task_id'] == replacement_id for task in tasks):
        raise ValueError('The v2 manifest does not contain its recorded failed task')
    manifest['tasks'] = tasks
    manifest['superseded_tasks'] = manifest.get('superseded_tasks', []) + superseded
    manifest['migration']['from'] = 'semantic-label-cf-v2'
    manifest['migration']['superseded_task'] = replacement_id
    manifest['migration']['reason'] = 'Image paths are resolved from the pilot run directory.'
    manifest['protocol'] = 'semantic-label-cf-v3'
    manifest['protocol_hash'] = stable_digest({k: v for k, v in manifest.items()
                                                if k not in ('created', 'protocol_hash')})
    state.pop('unavailable', None)
    state['unavailable'] = {}
    state['status'] = 'INCOMPLETE'
    state['pilot_protocol_hash'] = manifest['protocol_hash']
    save(folder / 'manifest.json', manifest)
    save(state_path, state)


def migrate_v3_api_run(folder, manifest):
    state_path = folder / 'state.json'
    state = json.loads(state_path.read_text(encoding='utf-8')) if state_path.exists() else None
    if (state is None or state.get('calls') != 7 or state.get('status') != 'INTERRUPTED'
            or state.get('pilot_protocol_hash') != manifest.get('protocol_hash')):
        raise ValueError('Only the verified, seven-launch interrupted v3 pilot can migrate to API transport')
    if stable_digest({k: v for k, v in manifest.items() if k not in ('created', 'protocol_hash')}) != manifest.get('protocol_hash'):
        raise ValueError('Cannot migrate a v3 manifest that fails its integrity check')
    if any(not (folder / sample['local_path']).is_file()
           or hashlib.sha256((folder / sample['local_path']).read_bytes()).hexdigest() != sample['image_sha256']
           for sample in manifest['samples']):
        raise ValueError('Cannot migrate: a selected image is missing or changed')
    if (folder / 'manifest-v3.json').exists():
        raise ValueError('Pilot v3 migration snapshot already exists')
    shutil.copy2(folder / 'manifest.json', folder / 'manifest-v3.json')
    superseded = list(manifest['tasks'])
    manifest['tasks'] = [dict(task, task_id=f"{task['task_id']}-api") for task in superseded]
    manifest['superseded_tasks'] = manifest.get('superseded_tasks', []) + superseded
    manifest['protocol'] = 'semantic-label-cf-v4'
    manifest['model'] = PREVIOUS_API_MODEL
    manifest['model_selection'] = 'Initial API migration choice; later changed by explicit user selection.'
    manifest['transport'] = 'Gemini Developer API v1beta generateContent'
    manifest['authentication'] = 'GEMINI_API_KEY/GOOGLE_API_KEY environment header; never written to artifacts'
    manifest['migration'] = {'from': 'semantic-label-cf-v3', 'reason': 'Use explicit Gemini API transport with inline image bytes; preserve all seven previous CLI launches.'}
    for task_id, record in state.get('tasks', {}).items():
        if record.get('attempts', 0) or record.get('status') == 'RUNNING':
            record.update(status='SUPERSEDED', error='CLI transport replaced with explicitly authorized API transport; launch history retained.')
        else:
            record.update(status='SUPERSEDED', error='Task ID versioned for API transport; no launch was made.')
    manifest['protocol_hash'] = stable_digest({k: v for k, v in manifest.items()
                                                if k not in ('created', 'protocol_hash')})
    state.update(status='INCOMPLETE', unavailable={}, pilot_protocol_hash=manifest['protocol_hash'])
    save(folder / 'manifest.json', manifest)
    save(state_path, state)


def migrate_v4_model_run(folder, manifest):
    state_path = folder / 'state.json'
    state = json.loads(state_path.read_text(encoding='utf-8')) if state_path.exists() else None
    if (state is None or state.get('calls') != 8 or state.get('pilot_protocol_hash') != manifest.get('protocol_hash')
            or manifest.get('model') != 'gemini-2.5-pro' or state.get('status') != 'INCOMPLETE'):
        raise ValueError('Only the verified eight-launch v4 pilot can migrate to the selected preview model')
    if stable_digest({k: v for k, v in manifest.items() if k not in ('created', 'protocol_hash')}) != manifest.get('protocol_hash'):
        raise ValueError('Cannot migrate a v4 manifest that fails its integrity check')
    canaries = [record for record in state.get('tasks', {}).values()
                if record.get('attempts') == 1 and record.get('status') == 'BUDGET_EXHAUSTED']
    if len(canaries) != 1:
        raise ValueError('Expected exactly one failed single-launch API canary')
    snapshot = folder / 'manifest-v4.json'
    if snapshot.exists():
        raise ValueError('Pilot v4 model migration snapshot already exists')
    shutil.copy2(folder / 'manifest.json', snapshot)
    manifest['protocol'] = 'semantic-label-cf-v5'
    manifest['model'] = SELECTED_PREVIOUS_API_MODEL
    manifest['model_selection'] = 'gemini-3.1-pro-preview, explicitly selected by the user after the API rejected gemini-2.5-pro.'
    manifest['migration'] = {'from': 'semantic-label-cf-v4', 'reason': 'User selected gemini-3.1-pro-preview; all eight prior launches and retry history remain in the ledger.'}
    manifest['protocol_hash'] = stable_digest({k: v for k, v in manifest.items()
                                                if k not in ('created', 'protocol_hash')})
    state['pilot_protocol_hash'] = manifest['protocol_hash']
    save(folder / 'manifest.json', manifest)
    save(state_path, state)


def migrate_v5_model_run(folder, manifest):
    state_path = folder / 'state.json'
    state = json.loads(state_path.read_text(encoding='utf-8')) if state_path.exists() else None
    if (state is None or state.get('calls') != 9 or state.get('status') != 'DEGRADED'
            or state.get('pilot_protocol_hash') != manifest.get('protocol_hash')
            or manifest.get('model') != SELECTED_PREVIOUS_API_MODEL
            or not state.get('unavailable', {}).get('gemini')):
        raise ValueError('Only the verified nine-launch quota-degraded v5 pilot can migrate models')
    if stable_digest({k: v for k, v in manifest.items() if k not in ('created', 'protocol_hash')}) != manifest.get('protocol_hash'):
        raise ValueError('Cannot migrate a v5 manifest that fails its integrity check')
    snapshot = folder / 'manifest-v5.json'
    if snapshot.exists():
        raise ValueError('Pilot v5 model migration snapshot already exists')
    shutil.copy2(folder / 'manifest.json', snapshot)
    manifest['protocol'] = 'semantic-label-cf-v6'
    manifest['model'] = API_MODEL
    manifest['model_selection'] = 'gemini-3.8-flash, explicitly selected by the user; API model catalog confirms generateContent support.'
    manifest['migration'] = {'from': 'semantic-label-cf-v5', 'reason': 'User requested gemini-3.8-flash after quota was exhausted on gemini-3.1-pro-preview.'}
    manifest['protocol_hash'] = stable_digest({k: v for k, v in manifest.items()
                                                if k not in ('created', 'protocol_hash')})
    state['pilot_protocol_hash'] = manifest['protocol_hash']
    save(folder / 'manifest.json', manifest)
    save(state_path, state)


def migrate_v6_model_run(folder, manifest):
    state_path = folder / 'state.json'
    state = json.loads(state_path.read_text(encoding='utf-8')) if state_path.exists() else None
    if (state is None or state.get('calls') != 16 or state.get('status') != 'INCOMPLETE'
            or state.get('pilot_protocol_hash') != manifest.get('protocol_hash')
            or manifest.get('model') != 'gemini-3.8-flash'):
        raise ValueError('Only the verified 16-launch v6 pilot can migrate to Gemini 2.5 Flash')
    if stable_digest({k: v for k, v in manifest.items() if k not in ('created', 'protocol_hash')}) != manifest.get('protocol_hash'):
        raise ValueError('Cannot migrate a v6 manifest that fails its integrity check')
    snapshot = folder / 'manifest-v6.json'
    if snapshot.exists():
        raise ValueError('Pilot v6 model migration snapshot already exists')
    shutil.copy2(folder / 'manifest.json', snapshot)
    manifest['protocol'] = 'semantic-label-cf-v7'
    manifest['model'] = REJECTED_FLASH_API_MODEL
    manifest['model_selection'] = 'gemini-2.5-flash, explicitly selected by the user; API model catalog confirms generateContent support.'
    manifest['migration'] = {'from': 'semantic-label-cf-v6', 'reason': 'User requested gemini-2.5-flash; preserve all 16 previous launches and attempt history.'}
    manifest['protocol_hash'] = stable_digest({k: v for k, v in manifest.items()
                                                if k not in ('created', 'protocol_hash')})
    state['pilot_protocol_hash'] = manifest['protocol_hash']
    save(folder / 'manifest.json', manifest)
    save(state_path, state)


def migrate_v7_model_run(folder, manifest):
    state_path = folder / 'state.json'
    state = json.loads(state_path.read_text(encoding='utf-8')) if state_path.exists() else None
    if (state is None or state.get('calls') != 18 or state.get('status') != 'INCOMPLETE'
            or state.get('pilot_protocol_hash') != manifest.get('protocol_hash')
            or manifest.get('model') != REJECTED_FLASH_API_MODEL):
        raise ValueError('Only the verified 18-launch v7 pilot can migrate back to Gemini 3.8 Flash')
    if stable_digest({k: v for k, v in manifest.items() if k not in ('created', 'protocol_hash')}) != manifest.get('protocol_hash'):
        raise ValueError('Cannot migrate a v7 manifest that fails its integrity check')
    snapshot = folder / 'manifest-v7.json'
    if snapshot.exists():
        raise ValueError('Pilot v7 model migration snapshot already exists')
    shutil.copy2(folder / 'manifest.json', snapshot)
    manifest['protocol'] = 'semantic-label-cf-v8'
    manifest['model'] = API_MODEL
    manifest['model_selection'] = 'gemini-3.8-flash, restored after Google API rejected gemini-2.5-flash as unavailable to new users.'
    manifest['migration'] = {'from': 'semantic-label-cf-v7', 'reason': 'Google API rejected gemini-2.5-flash as unavailable to new users; restore gemini-3.8-flash as explicitly instructed by provider.'}
    manifest['protocol_hash'] = stable_digest({k: v for k, v in manifest.items()
                                                if k not in ('created', 'protocol_hash')})
    state['pilot_protocol_hash'] = manifest['protocol_hash']
    save(folder / 'manifest.json', manifest)
    save(state_path, state)


def migrate_v8_model_run(folder, manifest):
    state_path = folder / 'state.json'
    state = json.loads(state_path.read_text(encoding='utf-8')) if state_path.exists() else None
    if (state is None or state.get('calls') != 21 or state.get('status') != 'INCOMPLETE'
            or state.get('pilot_protocol_hash') != manifest.get('protocol_hash')
            or manifest.get('model') != 'gemini-3.8-flash'):
        raise ValueError('Only the verified 21-launch v8 pilot can migrate to Gemini 3.1 Pro Preview')
    if stable_digest({k: v for k, v in manifest.items() if k not in ('created', 'protocol_hash')}) != manifest.get('protocol_hash'):
        raise ValueError('Cannot migrate a v8 manifest that fails its integrity check')
    snapshot = folder / 'manifest-v8.json'
    if snapshot.exists():
        raise ValueError('Pilot v8 model migration snapshot already exists')
    shutil.copy2(folder / 'manifest.json', snapshot)
    manifest['protocol'] = 'semantic-label-cf-v9'
    manifest['model'] = SELECTED_LATEST_API_MODEL
    manifest['model_selection'] = 'gemini-3.1-pro-preview, explicitly selected by the user; API model catalog confirms generateContent support.'
    manifest['migration'] = {'from': 'semantic-label-cf-v8', 'reason': 'User requested gemini-3.1-pro-preview; preserve all 21 previous launches and attempt history.'}
    manifest['protocol_hash'] = stable_digest({k: v for k, v in manifest.items()
                                                if k not in ('created', 'protocol_hash')})
    state['pilot_protocol_hash'] = manifest['protocol_hash']
    save(folder / 'manifest.json', manifest)
    save(state_path, state)


def migrate_v9_model_run(folder, manifest):
    state_path = folder / 'state.json'
    state = json.loads(state_path.read_text(encoding='utf-8')) if state_path.exists() else None
    if (state is None or state.get('calls') != 22 or state.get('status') != 'DEGRADED'
            or state.get('pilot_protocol_hash') != manifest.get('protocol_hash')
            or manifest.get('model') != 'gemini-3.1-pro-preview'):
        raise ValueError('Only the verified 22-launch quota-degraded v9 pilot can migrate to Gemini 3.8 Flash')
    if stable_digest({k: v for k, v in manifest.items() if k not in ('created', 'protocol_hash')}) != manifest.get('protocol_hash'):
        raise ValueError('Cannot migrate a v9 manifest that fails its integrity check')
    snapshot = folder / 'manifest-v9.json'
    if snapshot.exists():
        raise ValueError('Pilot v9 model migration snapshot already exists')
    shutil.copy2(folder / 'manifest.json', snapshot)
    manifest['protocol'] = 'semantic-label-cf-v10'
    manifest['model'] = API_MODEL
    manifest['model_selection'] = 'gemini-3.8-flash, explicitly selected by the user; API model catalog confirms generateContent support.'
    manifest['migration'] = {'from': 'semantic-label-cf-v9', 'reason': 'User requested gemini-3.8-flash after free tier quota was rejected (limit: 0) on gemini-3.1-pro-preview.'}
    manifest['protocol_hash'] = stable_digest({k: v for k, v in manifest.items()
                                                if k not in ('created', 'protocol_hash')})
    state['pilot_protocol_hash'] = manifest['protocol_hash']
    save(folder / 'manifest.json', manifest)
    save(state_path, state)


def claude_configuration():
    settings_path = Path.home() / '.claude' / 'settings.json'
    settings = json.loads(settings_path.read_text(encoding='utf-8')) if settings_path.is_file() else {}
    configured = settings.get('env', {})
    model = os.environ.get('ANTHROPIC_MODEL') or configured.get('ANTHROPIC_MODEL') or 'Claude Code configured default'
    endpoint = os.environ.get('ANTHROPIC_BASE_URL') or configured.get('ANTHROPIC_BASE_URL')
    endpoint_host = urlsplit(endpoint).hostname if endpoint else 'api.anthropic.com (Claude Code default)'
    return model, endpoint_host


def claude_command(max_budget_usd):
    return executable('claude') + [
        '-p', '--output-format', 'json', '--permission-mode', 'dontAsk',
        '--tools', 'Read', '--allowedTools', 'Read',
        '--max-budget-usd', f'{usd_floor_cents(max_budget_usd):.2f}',
        '--strict-mcp-config', '--mcp-config', '{"mcpServers":{}}',
    ]


def claude_budget_allows_call(spent_usd, cap_usd):
    return cap_usd - spent_usd >= CLAUDE_MIN_REMAINING_USD


def invoke_claude(prompt, image_path, attempt, timeout, max_budget_usd):
    if max_budget_usd is None or max_budget_usd < 0.01:
        raise RuntimeError('Claude run budget exhausted')
    full_prompt = (f'{prompt}\n\nUse the Read tool to inspect the image at this path before answering: '
                   f'{Path(image_path).resolve().as_posix()}\nReturn only the integer count.')
    command = claude_command(max_budget_usd)
    try:
        stdout, stderr, code = run_process(command, full_prompt, ROOT, timeout)
    except TimeoutError as exc:
        (attempt / 'stdout.txt').write_text(str(getattr(exc, 'stdout', '')), encoding='utf-8')
        (attempt / 'stderr.txt').write_text(str(getattr(exc, 'stderr', '')), encoding='utf-8')
        raise
    (attempt / 'stdout.txt').write_text(stdout, encoding='utf-8')
    (attempt / 'stderr.txt').write_text(stderr, encoding='utf-8')
    save(attempt / 'process.json', {'exit_code': code, 'command': command})
    if code != 0:
        raise RuntimeError(f'claude exit {code}: {(stderr or stdout)[-2500:]}')
    return parse_response('claude', stdout)


def migrate_v10_claude_run(folder, manifest, model=None, endpoint=None):
    state_path = folder / 'state.json'
    state = json.loads(state_path.read_text(encoding='utf-8')) if state_path.exists() else None
    if (state is None or not 0 < state.get('calls', 0) < MAX_CALLS
            or state.get('pilot_protocol_hash') != manifest.get('protocol_hash')
            or manifest.get('protocol') != 'semantic-label-cf-v10'
            or manifest.get('provider', 'gemini-api') != 'gemini-api'):
        raise ValueError('Only a verified, incomplete Gemini pilot can migrate to Claude Code')
    if stable_digest({k: v for k, v in manifest.items() if k not in ('created', 'protocol_hash')}) != manifest.get('protocol_hash'):
        raise ValueError('Cannot migrate a v10 manifest that fails its integrity check')
    if any(p.is_file() for p in (folder / 'reports').glob('*.md')):
        raise ValueError('Refusing provider migration after results have been produced')
    if any(not (folder / sample['local_path']).is_file()
           or hashlib.sha256((folder / sample['local_path']).read_bytes()).hexdigest() != sample['image_sha256']
           for sample in manifest['samples']):
        raise ValueError('Cannot migrate: a selected image is missing or changed')
    snapshot = folder / 'manifest-v10.json'
    if snapshot.exists():
        raise ValueError('Pilot v10 provider migration snapshot already exists')
    shutil.copy2(folder / 'manifest.json', snapshot)
    model = model or 'Claude Code configured default'
    endpoint = endpoint or 'api.anthropic.com (Claude Code default)'
    old_tasks = list(manifest['tasks'])
    manifest['tasks'] = [dict(task, task_id=f"{task['task_id']}-claude") for task in old_tasks]
    manifest['superseded_tasks'] = manifest.get('superseded_tasks', []) + old_tasks
    manifest['protocol'] = 'semantic-label-cf-v11'
    manifest['provider'] = 'claude'
    manifest['transport'] = 'Claude Code CLI print mode; local image inspected with Read tool'
    manifest['model'] = model
    manifest['endpoint_host'] = endpoint
    manifest['claude_usd_cap'] = 2.0
    manifest['migration'] = {'from': 'semantic-label-cf-v10',
                             'reason': 'User requested moving the remaining pilot tasks to Claude; prior Gemini launches remain in the 80-launch ledger.'}
    for task_id, record in state.get('tasks', {}).items():
        record.update(status='SUPERSEDED',
                      error='Gemini task ID superseded by Claude provider migration; prior attempts retained.')
    state.setdefault('member_stats', {})
    state['member_stats'].setdefault('claude', {
        'calls': 0, 'attempts': 0, 'seconds': 0.0, 'cost_usd': 0.0, 'tasks': 0})
    state['status'] = 'INCOMPLETE'
    state['pilot_protocol_hash'] = stable_digest({k: v for k, v in manifest.items()
                                                   if k not in ('created', 'protocol_hash')})
    manifest['protocol_hash'] = state['pilot_protocol_hash']
    save(folder / 'manifest.json', manifest)
    save(state_path, state)


def migrate_v11_label_run(folder, manifest):
    state_path = folder / 'state.json'
    state = json.loads(state_path.read_text(encoding='utf-8')) if state_path.exists() else None
    if (state is None or state.get('calls', 0) > MAX_CALLS
            or state.get('pilot_protocol_hash') != manifest.get('protocol_hash')
            or manifest.get('protocol') != 'semantic-label-cf-v11'
            or manifest.get('provider') != 'claude'):
        raise ValueError('Only a verified Claude pilot may migrate its task labels')
    if stable_digest({k: v for k, v in manifest.items() if k not in ('created', 'protocol_hash')}) != manifest.get('protocol_hash'):
        raise ValueError('Cannot migrate a Claude manifest that fails its integrity check')
    snapshot = folder / 'manifest-v11.json'
    if snapshot.exists():
        raise ValueError('Pilot v11 label migration snapshot already exists')
    shutil.copy2(folder / 'manifest.json', snapshot)
    canonical = {(task['sample_id'], task['condition'], task['replicate']): task
                 for task in build_tasks(manifest['samples'])}
    state_tasks = state.setdefault('tasks', {})
    migrated, superseded = [], []
    for old in manifest['tasks']:
        key = (old['sample_id'], old['condition'], old['replicate'])
        if key not in canonical:
            raise ValueError(f'No canonical task for {key}')
        task = dict(canonical[key], task_id=old['task_id'])
        mismatch = old['condition'] == 'named' and old.get('species') != task['species']
        if mismatch:
            old_record = state_tasks.get(old['task_id'], {})
            if old_record.get('status') == 'DONE':
                old_record.update(status='SUPERSEDED',
                                  error='Named-prompt species label did not match the selected image sample; response excluded from analysis.')
            superseded.append(old)
            task['task_id'] = f"{old['task_id']}-labels-fixed"
        migrated.append(task)
    manifest['tasks'] = migrated
    manifest['superseded_tasks'] = manifest.get('superseded_tasks', []) + superseded
    manifest['protocol'] = 'semantic-label-cf-v12'
    manifest['migration'] = {'from': 'semantic-label-cf-v11',
                             'reason': 'Regenerate task species labels from the selected sample metadata; invalidate completed named prompts with malformed labels while retaining their reports.'}
    manifest['protocol_hash'] = stable_digest({k: v for k, v in manifest.items()
                                                if k not in ('created', 'protocol_hash')})
    state['pilot_protocol_hash'] = manifest['protocol_hash']
    state['status'] = 'DEGRADED' if state.get('unavailable', {}).get('claude') else 'INCOMPLETE'
    save(folder / 'manifest.json', manifest)
    save(state_path, state)


def repair_interrupted_migration_hash(folder, manifest):
    manifest_path, state_path = folder / 'manifest.json', folder / 'state.json'
    snapshot_path = folder / 'manifest-v1.json'
    if not snapshot_path.is_file():
        raise ValueError('Pilot migration snapshot is missing')
    original = json.loads(snapshot_path.read_text(encoding='utf-8'))
    if stable_digest({k: v for k, v in original.items() if k not in ('created', 'protocol_hash')}) != original.get('protocol_hash'):
        raise ValueError('Pilot migration snapshot failed its integrity check')
    state = json.loads(state_path.read_text(encoding='utf-8'))
    if state.get('calls', 0) < 1 or state.get('pilot_protocol_hash') != manifest.get('protocol_hash'):
        raise ValueError('Pilot migration ledger does not match the interrupted run')
    manifest['protocol_hash'] = stable_digest({k: v for k, v in manifest.items()
                                                if k not in ('created', 'protocol_hash')})
    state['pilot_protocol_hash'] = manifest['protocol_hash']
    save(manifest_path, manifest)
    save(state_path, state)


def subprocess_version():
    from collab import run_process
    out, _, code = run_process(executable('gemini') + ['--version'], '', ROOT, 30)
    return out.strip() if code == 0 else 'version unavailable'


def validate_launch_ledger(folder, state):
    logs = Path(folder) / 'logs'
    logged_attempts = sum(1 for attempt in logs.glob('*/*') if attempt.is_dir() and attempt.name.isdigit())
    if logged_attempts > state.get('calls', 0):
        raise ValueError('Pilot launch ledger is lower than saved attempts; refusing a paid resume')


def write_answers(folder, manifest, runner):
    answers = []
    by_id = runner.state.get('tasks', {})
    for task in manifest['tasks']:
        record = by_id.get(task['task_id'], {})
        if record.get('status') != 'DONE':
            continue
        result = json.loads((folder / 'reports' / f"{task['task_id']}.md").read_text(encoding='utf-8'))
        count = parse_count(result['response'])
        answers.append({**{k: task[k] for k in ('task_id', 'sample_id', 'species', 'animal_group', 'condition',
                                                  'replicate', 'ground_truth', 'expected_bias')},
                        'response': result['response'], 'count': count,
                        'correct': count == task['ground_truth'],
                        'bias_aligned': count == task['expected_bias']})
    save(folder / 'answers.json', answers)
    provider = manifest.get('provider', 'gemini-api')
    member = 'claude' if provider == 'claude' else 'gemini'
    provider_name = 'Claude Code CLI' if provider == 'claude' else 'Gemini Developer API v1beta'
    summary = ['# Semantic-label pilot', '', f"Operational: {runner.state.get('status', 'INCOMPLETE')}",
               'Scientific: NEEDS_EVIDENCE', f"Total model launches: {runner.state['calls']}/{MAX_CALLS}",
               f"Transport/model: {provider_name} / {manifest['model']}",
               f"Endpoint host: {manifest.get('endpoint_host', 'not recorded')}",
               f"Reported cost: ${runner.state['member_stats'][member]['cost_usd']:.4f}",
               f"Completed responses: {len(answers)}/{len(manifest['tasks'])}", '',
               'One counterfactual 384px image per species; generic vs. species-named prompt; two calls per cell.',
               'Original-image baseline not included. Repeated calls are descriptive, not independent species.', '',
               '| Prompt | Correct / valid | Bias-aligned / valid | Valid N |', '|---|---:|---:|---:|']
    claude_cap = manifest.get('claude_usd_cap')
    claude_cost = runner.state['member_stats']['claude']['cost_usd']
    if member == 'claude' and claude_cap is not None and claude_cost > claude_cap:
        summary.insert(6, f"Claude native cap overshot by ${claude_cost - claude_cap:.4f}; no further Claude launches issued.")
    elif member == 'claude' and runner.state.get('budget_stop_reason'):
        summary.insert(6, runner.state['budget_stop_reason'])
    elif member == 'claude' and runner.state.get('unavailable', {}).get(member):
        error = runner.state['unavailable'][member]
        summary.insert(6, 'Claude stopped on a provider or native budget error; details are in the attempt log.')
    elif runner.state.get('unavailable', {}).get(member):
        message = runner.state['unavailable'][member]
        summary.insert(7, api_failure_summary(message) if member == 'gemini' else f'Claude status: {message}')
    else:
        failures = [runner.state.get('tasks', {}).get(task['task_id'], {}).get('error', '')
                    for task in manifest['tasks']]
        latest = next((error for error in reversed(failures) if error), '')
        if latest:
            summary.insert(7, api_failure_summary(latest) if member == 'gemini' else f'Claude status: {latest}')
    for condition in ('generic', 'named'):
        group = [a for a in answers if a['condition'] == condition and a['count'] is not None]
        summary.append(f"| {condition} | {sum(a['correct'] for a in group)} | "
                       f"{sum(a['bias_aligned'] for a in group)} | {len(group)} |")
    summary.extend(['', 'No causal mechanism or generalization claim follows from this pilot.'])
    (folder / 'SUMMARY.md').write_text('\n'.join(summary) + '\n', encoding='utf-8')


def pilot_status(complete, calls, remaining_seconds, unavailable=False):
    if complete:
        return 'COMPLETED'
    if calls >= MAX_CALLS or remaining_seconds <= 0:
        return 'BUDGET_EXHAUSTED'
    if unavailable:
        return 'DEGRADED'
    return 'INCOMPLETE'


def task_retry_exhausted(record):
    return (record.get('attempts', 0) >= MAX_RETRIES + 1
            and record.get('status') in ('FAILED', 'BUDGET_EXHAUSTED'))


def run_pilot(folder, manifest, timeout=240, max_seconds=14400, additional_calls=MAX_CALLS,
              retry_unavailable=False):
    with run_lock(Path(ROOT) / 'collab/controller.lock'):
        tasks_by_id = {task['task_id']: task for task in manifest['tasks']}
        provider = manifest.get('provider', 'gemini-api')
        member = 'claude' if provider == 'claude' else 'gemini'

        def invoke(member, prompt, attempt, call_timeout, max_budget_usd=None):
            task_id = attempt.parent.name
            task = bind_task_image(folder, tasks_by_id[task_id])
            if provider == 'claude':
                response = invoke_claude(prompt, task['image_path'], attempt, call_timeout, max_budget_usd)
            else:
                response = generate_api(prompt, task['image_path'], attempt, call_timeout, model=manifest['model'])
            return format_response(response)

        from collab import Runner
        runner = Runner(folder, invoke, max_calls=MAX_CALLS, retries=MAX_RETRIES, timeout=timeout,
                        max_seconds=max_seconds, backoff=2, member_max_calls=MAX_CALLS,
                        member_max_seconds=max_seconds,
                        member_max_usd=manifest.get('claude_usd_cap') if provider == 'claude' else None)
        validate_launch_ledger(folder, runner.state)
        if retry_unavailable and runner.state.get('unavailable', {}).get(member):
            runner.state.setdefault('quota_recovery_events', []).append({
                'at': datetime.now().astimezone().isoformat(),
                'previous_error': runner.state['unavailable'][member],
                'authorized_by_user': True,
            })
            del runner.state['unavailable'][member]
        if additional_calls < 1:
            raise ValueError('additional_calls must be at least 1')
        runner.max_calls = min(MAX_CALLS, runner.state['calls'] + additional_calls)
        runner.state.setdefault('pilot_protocol_hash', manifest['protocol_hash'])
        if runner.state['pilot_protocol_hash'] != manifest['protocol_hash']:
            raise ValueError('Pilot protocol changed; start a new run explicitly')
        runner.persist()
        for task in manifest['tasks']:
            if runner.state['calls'] >= runner.max_calls or runner.remaining() <= 0:
                break
            prior = runner.state.get('tasks', {}).get(task['task_id'], {})
            if task_retry_exhausted(prior):
                continue
            if (member == 'claude' and not claude_budget_allows_call(
                    runner.state['member_stats']['claude']['cost_usd'], manifest['claude_usd_cap'])):
                runner.state['budget_stop_reason'] = (
                    f"Claude launch skipped: less than ${CLAUDE_MIN_REMAINING_USD:.2f} remains under the native budget cap.")
                runner.persist()
                break
            prompt = ('Count the legs of this animal.' if task['condition'] == 'generic'
                      else f"Count the legs of this {task['species']}.")
            runner.task(task['task_id'], member, prompt)
        complete = all(runner.state.get('tasks', {}).get(task['task_id'], {}).get('status') == 'DONE'
                       for task in manifest['tasks'])
        runner.state['status'] = ('BUDGET_EXHAUSTED' if runner.state.get('budget_stop_reason') else
                                  pilot_status(complete, runner.state['calls'], runner.remaining(),
                                               unavailable=bool(runner.state['unavailable'].get(member))))
        runner.state['scientific_status'] = 'NEEDS_EVIDENCE'
        runner.persist()
        write_answers(folder, manifest, runner)
    return runner.state['status']


def select_run(new_run=False):
    base = Path(ROOT) / 'collab/experiment'
    pointer = base / 'semantic-label-latest.json'
    if pointer.exists() and not new_run:
        run_id = json.loads(pointer.read_text(encoding='utf-8'))['id']
        if not re.fullmatch(r'semantic-label-\d{8}-\d{6}-\d{6}', run_id):
            raise ValueError('Invalid semantic-label run pointer')
    else:
        run_id = datetime.now().strftime('semantic-label-%Y%m%d-%H%M%S-%f')
        save(pointer, {'id': run_id})
    folder = base / run_id
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def main(argv=None):
    parser = argparse.ArgumentParser(description='Run the bounded 80-launch semantic-label pilot.')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--status', action='store_true')
    parser.add_argument('--new-run', action='store_true')
    parser.add_argument('--max-new-launches', type=int, default=MAX_CALLS)
    parser.add_argument('--retry-unavailable', action='store_true')
    parser.add_argument('--timeout', type=int, default=240)
    parser.add_argument('--max-seconds', type=int, default=14400)
    args = parser.parse_args(argv)
    if args.status:
        base = Path(ROOT) / 'collab/experiment'
        pointer = base / 'semantic-label-latest.json'
        if not pointer.exists():
            print('No semantic-label pilot yet')
            return 0
        run_id = json.loads(pointer.read_text(encoding='utf-8'))['id']
        folder = base / run_id
        print(folder)
        print((folder / 'state.json').read_text(encoding='utf-8') if (folder / 'state.json').exists()
              else 'Prepared; no model calls yet')
        return 0
    if not 1 <= args.timeout <= 600 or not 1 <= args.max_seconds <= 86400:
        raise ValueError('Timeout must be 1..600 seconds and max-seconds 1..86400')
    folder = select_run(args.new_run)
    manifest = prepare_run(folder)
    if (manifest.get('provider', 'gemini-api') == 'gemini-api'
            and not any(os.environ.get(k) for k in ('GEMINI_API_KEY', 'GOOGLE_API_KEY'))):
        raise ValueError('GEMINI_API_KEY or GOOGLE_API_KEY is not available in this process')
    print(f"Prepared {len(manifest['samples'])} image families and {len(manifest['tasks'])} tasks in {folder}")
    if args.dry_run:
        from collab import Runner
        member = 'claude' if manifest.get('provider') == 'claude' else 'gemini'
        preview = Runner(folder, lambda *unused: None, max_calls=MAX_CALLS, retries=MAX_RETRIES,
                         timeout=args.timeout, max_seconds=args.max_seconds,
                         member_max_calls=MAX_CALLS, member_max_seconds=args.max_seconds,
                         member_max_usd=manifest.get('claude_usd_cap') if member == 'claude' else None)
        write_answers(folder, manifest, preview)
        print('Dry run: 0 model launches')
        return 0
    status = run_pilot(folder, manifest, timeout=args.timeout, max_seconds=args.max_seconds,
                       additional_calls=args.max_new_launches,
                       retry_unavailable=args.retry_unavailable)
    print((folder / 'SUMMARY.md').read_text(encoding='utf-8'))
    return 0 if status == 'COMPLETED' else 2


if __name__ == '__main__':
    raise SystemExit(main())
