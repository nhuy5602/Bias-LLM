"""Codex/Gemini visual-signal robustness pilot on a fixed logo subset."""

import argparse
import hashlib
import json
import random
import time
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageFilter, ImageOps

from collab import ROOT, executable, run_process, save

REPEATS = 3
RESOLUTION_LEVELS = (384, 192, 96)
BLUR_SIGMAS = (1.0, 3.0, 6.0)
MATCHED_BLUR_MAX_EDGE = 384
MATCHED_BLUR_SIGMAS = (0.0, 1.0, 3.0, 6.0)
SEED = 20261019
CODEX_MODEL = 'gpt-6.1-sol'
CODEX_PROVIDER = 'teamsoclo'
CODEX_REASONING = 'xhigh'
GEMINI_MODEL = 'Gemini 3.8 Flash (High)'
SOURCE_RUN_DEFAULT = ROOT / 'collab' / 'experiment' / 'logo-duo-20261019-005644-359883'

FAMILIES = {
    'audi-black-coupe': {'brand': 'Audi', 'element': 'overlapping circles', 'ground_truth': 4, 'counterfactual': 5},
    'audi-black-suv': {'brand': 'Audi', 'element': 'overlapping circles', 'ground_truth': 4, 'counterfactual': 5},
    'maserati-black-coupe': {'brand': 'Maserati', 'element': 'prongs', 'ground_truth': 3, 'counterfactual': 5},
    'maserati-black-suv': {'brand': 'Maserati', 'element': 'prongs', 'ground_truth': 3, 'counterfactual': 5},
    'mercedes-black-coupe': {'brand': 'Mercedes-Benz', 'element': 'points on the star', 'ground_truth': 3, 'counterfactual': 4},
    'mercedes-black-suv': {'brand': 'Mercedes-Benz', 'element': 'points on the star', 'ground_truth': 3, 'counterfactual': 4},
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def select_source_rows(rows):
    selected = []
    for family in FAMILIES:
        family_rows = [row for row in rows if row.get('family') == family]
        states = {row.get('state') for row in family_rows}
        if states != {'original', 'counterfactual'}:
            raise ValueError(f'Missing original/counterfactual pair for {family}: {states}')
        selected.extend(sorted(family_rows, key=lambda row: row['state']))
    return selected


def derive_variant(source_path, target_path, max_edge):
    source_path = Path(source_path)
    target_path = Path(target_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source_path) as opened:
        image = ImageOps.exif_transpose(opened).convert('RGB')
        image.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
        derived_size = list(image.size)
        image.save(target_path, format='PNG', optimize=False)
        source_size = list(opened.size)
    return {
        'source_sha256': sha256(source_path),
        'derived_sha256': sha256(target_path),
        'max_edge': max_edge,
        'source_size': source_size,
        'derived_size': derived_size,
        'transform': 'Pillow EXIF transpose + RGB + aspect-preserving LANCZOS thumbnail; no enlargement; PNG output',
    }


def derive_blur_variant(source_path, target_path, sigma):
    source_path = Path(source_path)
    target_path = Path(target_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source_path) as opened:
        source_size = list(opened.size)
        image = ImageOps.exif_transpose(opened).convert('RGB')
        image = image.filter(ImageFilter.GaussianBlur(radius=float(sigma)))
        derived_size = list(image.size)
        image.save(target_path, format='PNG', optimize=False)
    return {
        'source_sha256': sha256(source_path),
        'derived_sha256': sha256(target_path),
        'blur_sigma': float(sigma),
        'source_size': source_size,
        'derived_size': derived_size,
        'transform': f'Pillow EXIF transpose + RGB + GaussianBlur(radius={float(sigma):g}); no resize; PNG output',
    }


def normalize_source_variant(source_path, target_path, max_edge=MATCHED_BLUR_MAX_EDGE):
    source_path = Path(source_path)
    target_path = Path(target_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source_path) as opened:
        source_size = list(opened.size)
        image = ImageOps.exif_transpose(opened).convert('RGB')
        image.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
        normalized_size = list(image.size)
        image.save(target_path, format='PNG', optimize=False)
    return {
        'source_sha256': sha256(source_path),
        'normalized_sha256': sha256(target_path),
        'max_edge': int(max_edge),
        'source_size': source_size,
        'normalized_size': normalized_size,
        'normalization_transform': 'Pillow EXIF transpose + RGB + aspect-preserving LANCZOS thumbnail; no enlargement; PNG output',
    }


def derive_matched_blur_variant(normalized_path, target_path, sigma, normalization):
    normalized_path = Path(normalized_path)
    target_path = Path(target_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if float(sigma) == 0.0:
        target_path.write_bytes(normalized_path.read_bytes())
        with Image.open(normalized_path) as opened:
            derived_size = list(opened.size)
        blur_transform = 'No filtering; byte-identical copy of normalized RGB PNG'
    else:
        with Image.open(normalized_path) as opened:
            image = ImageOps.exif_transpose(opened).convert('RGB')
            image = image.filter(ImageFilter.GaussianBlur(radius=float(sigma)))
            derived_size = list(image.size)
            image.save(target_path, format='PNG', optimize=False)
        blur_transform = f'Pillow GaussianBlur(radius={float(sigma):g}) applied after normalization; PNG output'
    return {
        **normalization,
        'normalized_sha256': sha256(normalized_path),
        'derived_sha256': sha256(target_path),
        'blur_sigma': float(sigma),
        'derived_size': derived_size,
        'blur_transform': blur_transform,
        'transform': f"{normalization['normalization_transform']}; {blur_transform}",
    }


def build_tasks(variants):
    tasks = []
    for variant in variants:
        config = FAMILIES[variant['family']]
        for repeat in range(1, REPEATS + 1):
            tasks.append({
                'task_id': f"{variant['family']}-{variant['state']}-r{variant['resolution']}-repeat-{repeat:02}-codex",
                'family': variant['family'],
                'brand': config['brand'],
                'state': variant['state'],
                'resolution': variant['resolution'],
                'repeat': repeat,
                'agent': 'codex',
                'image_path': variant['derived_path'],
                'source_sha256': variant.get('source_sha256', variant.get('sha256')),
                'image_sha256': variant.get('derived_sha256', variant.get('image_sha256')),
                'prompt': f"Count the {config['element']} in this {config['brand']} logo. Return one integer only.",
                'ground_truth': variant['ground_truth'],
                'expected_bias': variant.get('expected_bias'),
            })
    random.Random(SEED).shuffle(tasks)
    return tasks


def build_gemini_prompt(variants):
    lines = [
        '# Gemini Antigravity Blind Robustness Pilot', '',
        f'Required model: {GEMINI_MODEL}. If this model cannot be selected, stop before running and report the deviation instead of substituting silently.',
        'Evaluate every listed image independently. Do not read Codex results or output files.',
        'Use one model/configuration for all images. Return one integer for each exact prompt.',
        'This is a visual-signal robustness screen: resolution levels are derived from the same source image with one fixed aspect-preserving LANCZOS transform.',
        'Original/counterfactual rows are metadata-matched by brand, black color, and body style, not pixel-paired source identity.',
        'The 3 repeats are repeated readings on the same derived image, not independent samples. Keep every repeat isolated.',
        'Record the full 64-character source SHA256 and derived SHA256 values shown below in your report. Write results to `GEMINI-REPORT.md` in this run folder.',
        'Do not claim a causal mechanism. Keep the scientific status at `NEEDS_EVIDENCE`.', '',
        '| Task ID | Family | State | Max edge | Repeat | Derived image | Source SHA256 | Derived SHA256 | Prompt |',
        '|---|---|---:|---:|---:|---|---|---|---|',
    ]
    for variant in variants:
        config = FAMILIES[variant['family']]
        prompt = f"Count the {config['element']} in this {config['brand']} logo. Return one integer only."
        for repeat in range(1, REPEATS + 1):
            task_id = f"{variant['family']}-{variant['state']}-r{variant['resolution']}-repeat-{repeat:02}-gemini"
            lines.append(
                f"| {task_id} | {variant['family']} | {variant['state']} | {variant['resolution']} | {repeat} | "
                f"`{Path(variant['derived_path']).resolve()}` | `{variant.get('source_sha256', variant.get('sha256'))}` | `{variant.get('derived_sha256', variant.get('image_sha256'))}` | {prompt} |"
            )
    return '\n'.join(lines) + '\n'


def build_blur_tasks(variants):
    tasks = []
    for variant in variants:
        config = FAMILIES[variant['family']]
        sigma = float(variant['blur_sigma'])
        label = f'sigma{sigma:g}'
        for repeat in range(1, REPEATS + 1):
            tasks.append({
                'task_id': f"{variant['family']}-{variant['state']}-{label}-repeat-{repeat:02}-codex",
                'family': variant['family'], 'brand': config['brand'], 'state': variant['state'],
                'corruption': 'gaussian_blur', 'blur_sigma': sigma, 'repeat': repeat,
                'agent': 'codex', 'image_path': variant['derived_path'],
                'source_sha256': variant.get('source_sha256', variant.get('sha256')),
                'image_sha256': variant.get('derived_sha256', variant.get('image_sha256')),
                'prompt': f"Count the {config['element']} in this {config['brand']} logo. Return one integer only.",
                'ground_truth': variant['ground_truth'], 'expected_bias': variant.get('expected_bias'),
            })
    random.Random(SEED).shuffle(tasks)
    return tasks


def build_blur_gemini_prompt(variants):
    lines = [
        '# Gemini Antigravity Blind Gaussian-Blur Pilot', '',
        f'Required model: {GEMINI_MODEL}. If this model cannot be selected, stop before running and report the deviation instead of substituting silently.',
        'Evaluate every listed image independently. Do not read Codex results or output files.',
        'Use one model/configuration for all images. Return one integer for each exact prompt.',
        'This is a corruption robustness screen: every variant is derived from the same source image with EXIF transpose, RGB conversion, and one GaussianBlur radius; dimensions are unchanged.',
        'Original/counterfactual rows are metadata-matched by brand, black color, and body style, not pixel-paired source identity.',
        'The 3 repeats are repeated readings on the same blurred image, not independent samples. Keep every repeat isolated.',
        'Record the full 64-character source SHA256 and derived SHA256 values shown below in your report. Write results to `GEMINI-REPORT.md` in this run folder.',
        'Do not claim a causal mechanism. Keep the scientific status at `NEEDS_EVIDENCE`.', '',
        '| Task ID | Family | State | Blur sigma | Repeat | Derived image | Source SHA256 | Derived SHA256 | Prompt |',
        '|---|---|---:|---:|---:|---|---|---|---|',
    ]
    for variant in variants:
        config = FAMILIES[variant['family']]
        sigma = float(variant['blur_sigma'])
        prompt = f"Count the {config['element']} in this {config['brand']} logo. Return one integer only."
        for repeat in range(1, REPEATS + 1):
            task_id = f"{variant['family']}-{variant['state']}-sigma{sigma:g}-repeat-{repeat:02}-gemini"
            lines.append(
                f"| {task_id} | {variant['family']} | {variant['state']} | {sigma:g} | {repeat} | "
                f"`{Path(variant['derived_path']).resolve()}` | `{variant.get('source_sha256', variant.get('sha256'))}` | `{variant.get('derived_sha256', variant.get('image_sha256'))}` | {prompt} |"
            )
    return '\n'.join(lines) + '\n'


def build_matched_blur_tasks(variants):
    tasks = []
    for variant in variants:
        config = FAMILIES[variant['family']]
        sigma = float(variant['blur_sigma'])
        label = f'sigma{sigma:g}'
        max_edge = int(variant['max_edge'])
        for repeat in range(1, REPEATS + 1):
            tasks.append({
                'task_id': f"{variant['family']}-{variant['state']}-r{max_edge}-{label}-repeat-{repeat:02}-codex",
                'family': variant['family'], 'brand': config['brand'], 'state': variant['state'],
                'corruption': 'gaussian_blur_matched', 'max_edge': max_edge,
                'blur_sigma': sigma, 'repeat': repeat, 'agent': 'codex',
                'image_path': variant['derived_path'],
                'normalized_path': variant['normalized_path'],
                'source_sha256': variant.get('source_sha256', variant.get('sha256')),
                'normalized_sha256': variant['normalized_sha256'],
                'image_sha256': variant['derived_sha256'],
                'source_size': variant['source_size'],
                'normalized_size': variant['normalized_size'],
                'derived_size': variant['derived_size'],
                'prompt': f"Count the {config['element']} in this {config['brand']} logo. Return one integer only.",
                'ground_truth': variant['ground_truth'], 'expected_bias': variant.get('expected_bias'),
            })
    random.Random(SEED).shuffle(tasks)
    return tasks


def build_matched_blur_gemini_prompt(variants):
    lines = [
        '# Gemini Antigravity Blind Matched-Resolution Gaussian-Blur Pilot', '',
        f'Required model: {GEMINI_MODEL}. If this model cannot be selected, stop before running and report the deviation instead of substituting silently.',
        'Evaluate every listed image independently. Do not read Codex results or output files.',
        'Use one model/configuration for all images. Return one integer for each exact prompt.',
        f'Every source image is normalized first to max_edge={MATCHED_BLUR_MAX_EDGE} with EXIF transpose, RGB conversion, and aspect-preserving LANCZOS; Gaussian blur is applied only after that normalization.',
        'The sigma=0 condition is the no-blur baseline: use the normalized image content unchanged, with no additional filtering.',
        'Original/counterfactual rows are metadata-matched by brand, black color, and body style, not pixel-paired source identity.',
        'The 3 repeats are repeated readings on the same normalized-and-blurred image, not independent samples. Keep every repeat isolated.',
        'Record the full 64-character source, normalized, and derived SHA256 values and dimensions shown below in your report. Write results to `GEMINI-REPORT.md` in this run folder.',
        'Do not claim a causal mechanism. Keep the scientific status at `NEEDS_EVIDENCE`.', '',
        '| Task ID | Family | State | Max edge | Blur sigma | Repeat | Normalized image | Blurred image | Source SHA256 | Normalized SHA256 | Derived SHA256 | Dimensions | Prompt |',
        '|---|---|---|---:|---:|---:|---|---|---|---|---|---|---|',
    ]
    for variant in variants:
        config = FAMILIES[variant['family']]
        sigma = float(variant['blur_sigma'])
        max_edge = int(variant['max_edge'])
        prompt = f"Count the {config['element']} in this {config['brand']} logo. Return one integer only."
        dimensions = f"source={variant['source_size']}; normalized={variant['normalized_size']}; derived={variant['derived_size']}"
        for repeat in range(1, REPEATS + 1):
            task_id = f"{variant['family']}-{variant['state']}-r{max_edge}-sigma{sigma:g}-repeat-{repeat:02}-gemini"
            lines.append(
                f"| {task_id} | {variant['family']} | {variant['state']} | {max_edge} | {sigma:g} | {repeat} | "
                f"`{Path(variant['normalized_path']).resolve()}` | `{Path(variant['derived_path']).resolve()}` | "
                f"`{variant.get('source_sha256', variant.get('sha256'))}` | `{variant['normalized_sha256']}` | `{variant['derived_sha256']}` | {dimensions} | {prompt} |"
            )
    return '\n'.join(lines) + '\n'


def prepare_run(folder, source_run):
    folder = Path(folder)
    source_run = Path(source_run).resolve()
    folder.mkdir(parents=True, exist_ok=False)
    source_manifest_path = source_run / 'manifest.json'
    if not source_manifest_path.is_file():
        raise ValueError(f'Missing source manifest: {source_manifest_path}')
    source_manifest = json.loads(source_manifest_path.read_text(encoding='utf-8'))
    rows = select_source_rows(source_manifest['rows'])
    variants = []
    for row in rows:
        source_path = Path(row['local_path']).resolve()
        if sha256(source_path) != row['sha256']:
            raise ValueError(f'Source hash mismatch: {source_path}')
        for level in RESOLUTION_LEVELS:
            target = folder / 'images' / f"{row['family']}-{row['state']}-r{level}.png"
            transform = derive_variant(source_path, target, level)
            variants.append({
                **row,
                **transform,
                'source_path': str(source_path),
                'derived_path': str(target),
                'resolution': level,
                'source_run': str(source_run),
            })
    manifest = {
        'protocol': 'vlmbias-logo-robustness-v1',
        'created': datetime.now().astimezone().isoformat(),
        'source_run': str(source_run),
        'source_manifest_sha256': sha256(source_manifest_path),
        'families': list(FAMILIES),
        'resolution_levels': list(RESOLUTION_LEVELS),
        'repeats': REPEATS,
        'transform': variants[0]['transform'] if variants else None,
        'codex': {'model': CODEX_MODEL, 'provider': CODEX_PROVIDER, 'reasoning_effort': CODEX_REASONING},
        'scientific_status': 'NEEDS_EVIDENCE',
        'variants': variants,
    }
    manifest['tasks'] = build_tasks(variants)
    save(folder / 'manifest.json', manifest)
    (folder / 'GEMINI-PROMPT.md').write_text(build_gemini_prompt(variants), encoding='utf-8')
    return manifest


def prepare_blur_run(folder, source_run):
    folder = Path(folder)
    source_run = Path(source_run).resolve()
    folder.mkdir(parents=True, exist_ok=False)
    source_manifest_path = source_run / 'manifest.json'
    if not source_manifest_path.is_file():
        raise ValueError(f'Missing source manifest: {source_manifest_path}')
    source_manifest = json.loads(source_manifest_path.read_text(encoding='utf-8'))
    rows = select_source_rows(source_manifest['rows'])
    variants = []
    for row in rows:
        source_path = Path(row['local_path']).resolve()
        if sha256(source_path) != row['sha256']:
            raise ValueError(f'Source hash mismatch: {source_path}')
        for sigma in BLUR_SIGMAS:
            target = folder / 'images' / f"{row['family']}-{row['state']}-sigma{sigma:g}.png"
            transform = derive_blur_variant(source_path, target, sigma)
            variants.append({**row, **transform, 'source_path': str(source_path),
                             'derived_path': str(target), 'source_run': str(source_run)})
    manifest = {
        'protocol': 'vlmbias-logo-gaussian-blur-v1', 'mode': 'gaussian_blur',
        'created': datetime.now().astimezone().isoformat(), 'source_run': str(source_run),
        'source_manifest_sha256': sha256(source_manifest_path), 'families': list(FAMILIES),
        'blur_sigmas': list(BLUR_SIGMAS), 'repeats': REPEATS,
        'transform': variants[0]['transform'] if variants else None,
        'codex': {'model': CODEX_MODEL, 'provider': CODEX_PROVIDER, 'reasoning_effort': CODEX_REASONING},
        'scientific_status': 'NEEDS_EVIDENCE', 'variants': variants,
    }
    manifest['tasks'] = build_blur_tasks(variants)
    save(folder / 'manifest.json', manifest)
    (folder / 'GEMINI-PROMPT.md').write_text(build_blur_gemini_prompt(variants), encoding='utf-8')
    return manifest


def prepare_matched_blur_run(folder, source_run):
    folder = Path(folder)
    source_run = Path(source_run).resolve()
    folder.mkdir(parents=True, exist_ok=False)
    source_manifest_path = source_run / 'manifest.json'
    if not source_manifest_path.is_file():
        raise ValueError(f'Missing source manifest: {source_manifest_path}')
    source_manifest = json.loads(source_manifest_path.read_text(encoding='utf-8'))
    rows = select_source_rows(source_manifest['rows'])
    variants = []
    for row in rows:
        source_path = Path(row['local_path']).resolve()
        if sha256(source_path) != row['sha256']:
            raise ValueError(f'Source hash mismatch: {source_path}')
        normalized_path = folder / 'images' / f"{row['family']}-{row['state']}-normalized-r{MATCHED_BLUR_MAX_EDGE}.png"
        normalization = normalize_source_variant(source_path, normalized_path, MATCHED_BLUR_MAX_EDGE)
        for sigma in MATCHED_BLUR_SIGMAS:
            target = folder / 'images' / f"{row['family']}-{row['state']}-r{MATCHED_BLUR_MAX_EDGE}-sigma{sigma:g}.png"
            transform = derive_matched_blur_variant(normalized_path, target, sigma, normalization)
            variants.append({
                **row,
                **transform,
                'source_path': str(source_path),
                'normalized_path': str(normalized_path),
                'derived_path': str(target),
                'source_run': str(source_run),
            })
    manifest = {
        'protocol': 'vlmbias-logo-gaussian-blur-matched-v2',
        'mode': 'gaussian_blur_matched',
        'created': datetime.now().astimezone().isoformat(),
        'source_run': str(source_run),
        'source_manifest_sha256': sha256(source_manifest_path),
        'families': list(FAMILIES),
        'normalization_max_edge': MATCHED_BLUR_MAX_EDGE,
        'blur_sigmas': list(MATCHED_BLUR_SIGMAS),
        'repeats': REPEATS,
        'normalization_transform': variants[0]['normalization_transform'] if variants else None,
        'blur_transform': variants[0]['blur_transform'] if variants else None,
        'transform': variants[0]['transform'] if variants else None,
        'codex': {'model': CODEX_MODEL, 'provider': CODEX_PROVIDER, 'reasoning_effort': CODEX_REASONING},
        'scientific_status': 'NEEDS_EVIDENCE',
        'variants': variants,
    }
    manifest['tasks'] = build_matched_blur_tasks(variants)
    save(folder / 'manifest.json', manifest)
    (folder / 'GEMINI-PROMPT.md').write_text(build_matched_blur_gemini_prompt(variants), encoding='utf-8')
    return manifest


def run_tasks(folder, tasks, timeout=240, invoke_fn=None):
    folder = Path(folder)
    state_path, results_path = folder / 'state.json', folder / 'results.json'
    state = json.loads(state_path.read_text(encoding='utf-8')) if state_path.exists() else {
        'calls': 0, 'status': 'PENDING', 'scientific_status': 'NEEDS_EVIDENCE', 'tasks': {}}
    results = json.loads(results_path.read_text(encoding='utf-8')) if results_path.exists() else []
    done = {result['task_id'] for result in results}
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
            save(task_dir / 'attempt.json', {'status': 'DONE' if count is not None else 'INVALID', 'seconds': time.monotonic() - started})
        except Exception as exc:
            result = {**task, 'raw': '', 'count': None, 'valid': False, 'error': str(exc), 'attempt': attempt}
            state['tasks'][task['task_id']].update(status='FAILED', error=str(exc))
            save(task_dir / 'attempt.json', {'status': 'FAILED', 'error': str(exc), 'seconds': time.monotonic() - started})
            (task_dir / 'stderr.txt').write_text(str(exc), encoding='utf-8')
        results.append(result)
        done.add(task['task_id'])
        save(results_path, results)
        save(state_path, state)
    state['status'] = 'COMPLETED' if all(state['tasks'].get(task['task_id'], {}).get('status') in ('DONE', 'INVALID') for task in tasks) else 'DEGRADED'
    save(state_path, state)
    return results, state


def parse_count(text):
    text = str(text).strip()
    return int(text.strip('{}').strip()) if text.strip('{}').strip().isdigit() else None


def invoke(member, prompt, image_path, attempt_dir, timeout):
    if member != 'codex':
        raise ValueError('This runner is Codex-only; Gemini runs through the generated Antigravity prompt')
    attempt_dir.mkdir(parents=True, exist_ok=True)
    answer_path = attempt_dir / 'answer.txt'
    command = list(executable('codex')) + [
        'exec', '--model', CODEX_MODEL, '--sandbox', 'read-only',
        '-c', 'model_provider="teamsoclo"', '-c', 'model_reasoning_effort="xhigh"',
        '-c', 'approval_policy="never"', '--ephemeral', '--color', 'never',
        '--output-last-message', str(answer_path), '--image', str(image_path), '-']
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
        '# Logo visual-signal robustness pilot', '',
        f"Operational: {state['status']}", 'Scientific: NEEDS_EVIDENCE',
        f'Codex launches: {calls}/{len(manifest["tasks"])}', '',
        f"Protocol: six fixed families, original/counterfactual, max-edge levels {manifest['resolution_levels']}, {REPEATS} repeats per derived image.",
        f"Source run: `{manifest['source_run']}`; source manifest SHA256: `{manifest['source_manifest_sha256']}`",
        f"Codex: {manifest['codex']['model']} / {manifest['codex']['provider']} / {manifest['codex']['reasoning_effort']}",
        'The transform is a robustness screen, not a causal intervention. Source families and repeated readings remain separate.', '',
        '| Brand | Max edge | State | Correct | Bias-aligned | Valid N |', '|---|---:|---|---:|---:|---:|',
    ]
    for brand in ('Audi', 'Maserati', 'Mercedes-Benz'):
        for level in RESOLUTION_LEVELS:
            for state_name in ('original', 'counterfactual'):
                subset = [r for r in results if r['brand'] == brand and r['resolution'] == level and r['state'] == state_name and r['count'] is not None]
                correct = sum(r['count'] == r['ground_truth'] for r in subset)
                biased = sum(r['count'] == r['expected_bias'] for r in subset if r['expected_bias'] is not None)
                lines.append(f'| {brand} | {level} | {state_name} | {correct} | {biased} | {len(subset)} |')
    failures = [result for result in results if not result['valid']]
    lines += ['', f'Failed/invalid Codex tasks: {len(failures)}']
    for result in failures:
        lines.append(f"- {result['task_id']}: {result.get('error', 'invalid integer response')}")
    lines += ['', 'Results are descriptive robustness evidence only; they do not establish causality or universal generalization.', 'Scientific status remains `NEEDS_EVIDENCE`.']
    (Path(folder) / 'SUMMARY.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def summarize_blur(folder, manifest, results, calls):
    state = json.loads((Path(folder) / 'state.json').read_text(encoding='utf-8'))
    lines = [
        '# Logo Gaussian-blur robustness pilot', '', f"Operational: {state['status']}",
        'Scientific: NEEDS_EVIDENCE', f'Codex launches: {calls}/{len(manifest["tasks"])}', '',
        f"Protocol: six fixed families, original/counterfactual, GaussianBlur sigmas {manifest['blur_sigmas']}, {REPEATS} repeats per blurred image.",
        f"Source run: `{manifest['source_run']}`; source manifest SHA256: `{manifest['source_manifest_sha256']}`",
        f"Codex: {manifest['codex']['model']} / {manifest['codex']['provider']} / {manifest['codex']['reasoning_effort']}",
        'Gaussian blur is a robustness screen, not a causal intervention. Source families and repeats remain separate.', '',
        '| Brand | Blur sigma | State | Correct | Bias-aligned | Valid N |', '|---|---:|---|---:|---:|---:|',
    ]
    for brand in ('Audi', 'Maserati', 'Mercedes-Benz'):
        for sigma in BLUR_SIGMAS:
            for state_name in ('original', 'counterfactual'):
                subset = [r for r in results if r['brand'] == brand and float(r['blur_sigma']) == sigma and r['state'] == state_name and r['count'] is not None]
                correct = sum(r['count'] == r['ground_truth'] for r in subset)
                biased = sum(r['count'] == r['expected_bias'] for r in subset if r['expected_bias'] is not None)
                lines.append(f'| {brand} | {sigma:g} | {state_name} | {correct} | {biased} | {len(subset)} |')
    failures = [result for result in results if not result['valid']]
    lines += ['', f'Failed/invalid Codex tasks: {len(failures)}']
    for result in failures:
        lines.append(f"- {result['task_id']}: {result.get('error', 'invalid integer response')}")
    lines += ['', 'Results are descriptive robustness evidence only; they do not establish causality or universal generalization.', 'Scientific status remains `NEEDS_EVIDENCE`.']
    (Path(folder) / 'SUMMARY.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def summarize_matched_blur(folder, manifest, results, calls):
    state = json.loads((Path(folder) / 'state.json').read_text(encoding='utf-8'))
    lines = [
        '# Logo matched-resolution Gaussian-blur robustness pilot', '',
        f"Operational: {state['status']}", 'Scientific: NEEDS_EVIDENCE',
        f'Codex launches: {calls}/{len(manifest["tasks"])}', '',
        f"Protocol: six fixed families, original/counterfactual, all sources normalized to max edge {manifest['normalization_max_edge']} with sigma=0 no-blur baseline and GaussianBlur sigmas {manifest['blur_sigmas'][1:]}, {REPEATS} repeats per derived image.",
        f"Source run: `{manifest['source_run']}`; source manifest SHA256: `{manifest['source_manifest_sha256']}`",
        f"Codex: {manifest['codex']['model']} / {manifest['codex']['provider']} / {manifest['codex']['reasoning_effort']}",
        'Normalization and blur are a resolution-matched robustness screen, not a causal intervention. Source families and repeats remain separate.', '',
        '| Brand | Max edge | Blur sigma | State | Correct | Bias-aligned | Valid N |', '|---|---:|---:|---|---:|---:|---:|',
    ]
    for brand in ('Audi', 'Maserati', 'Mercedes-Benz'):
        for sigma in manifest['blur_sigmas']:
            for state_name in ('original', 'counterfactual'):
                subset = [r for r in results if r['brand'] == brand and r['max_edge'] == MATCHED_BLUR_MAX_EDGE and float(r['blur_sigma']) == sigma and r['state'] == state_name and r['count'] is not None]
                correct = sum(r['count'] == r['ground_truth'] for r in subset)
                biased = sum(r['count'] == r['expected_bias'] for r in subset if r['expected_bias'] is not None)
                lines.append(f'| {brand} | {MATCHED_BLUR_MAX_EDGE} | {sigma:g} | {state_name} | {correct} | {biased} | {len(subset)} |')
    failures = [result for result in results if not result['valid']]
    lines += ['', f'Failed/invalid Codex tasks: {len(failures)}']
    for result in failures:
        lines.append(f"- {result['task_id']}: {result.get('error', 'invalid integer response')}")
    lines += ['', 'Results are descriptive robustness evidence only; they do not establish causality or universal generalization.', 'Scientific status remains `NEEDS_EVIDENCE`.']
    (Path(folder) / 'SUMMARY.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--run-dir', type=Path, help='execute an existing prepared run directory')
    parser.add_argument('--source-run', type=Path, default=SOURCE_RUN_DEFAULT)
    parser.add_argument('--mode', choices=('resolution', 'gaussian-blur', 'gaussian-blur-matched'), default='resolution')
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
    else:
        run_id = datetime.now().strftime('logo-robustness-%Y%m%d-%H%M%S-%f')
        folder = ROOT / 'collab' / 'experiment' / run_id
        if args.mode == 'gaussian-blur':
            manifest = prepare_blur_run(folder, args.source_run)
        elif args.mode == 'gaussian-blur-matched':
            manifest = prepare_matched_blur_run(folder, args.source_run)
        else:
            manifest = prepare_run(folder, args.source_run)
        print(f'Gemini prompt: {folder / "GEMINI-PROMPT.md"}', flush=True)
        if args.dry_run:
            print(f'Dry run: {len(manifest["tasks"])} tasks, 0 model launches; {folder}')
            return 0
    results, state = run_tasks(folder, manifest['tasks'], args.timeout)
    if manifest.get('mode') == 'gaussian_blur':
        summarize_blur(folder, manifest, results, state['calls'])
    elif manifest.get('mode') == 'gaussian_blur_matched':
        summarize_matched_blur(folder, manifest, results, state['calls'])
    else:
        summarize(folder, manifest, results, state['calls'])
    print(folder / 'SUMMARY.md')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
