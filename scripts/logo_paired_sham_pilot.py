"""Pixel-paired logo-erasure and sham-control pilot."""

import argparse
import json
import random
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageFilter

from collab import ROOT, save
from logo_robustness_pilot import (
    FAMILIES as ROBUSTNESS_FAMILIES,
    MATCHED_BLUR_MAX_EDGE,
    REPEATS,
    invoke,
    normalize_source_variant,
    run_tasks,
    sha256,
)

SEED = 20261021
SHAM_BLUR_RADIUS = 12.0
SOURCE_RUN_DEFAULT = ROOT / 'collab' / 'experiment' / 'logo-duo-20261019-005644-359883'
CONDITIONS = ('original', 'logo_erased', 'sham_erased')
FAMILIES = {name: ROBUSTNESS_FAMILIES[name] for name in (
    'audi-black-coupe', 'audi-black-suv',
    'maserati-black-coupe', 'maserati-black-suv',
    'mercedes-black-coupe', 'mercedes-black-suv',
)}

# Relative ROIs were selected from the six fixed source images. They are part of
# the protocol, not an implicit image search, and are recorded in every manifest.
ROI_FRACTIONS = {
    'audi-black-coupe': (0.42, 0.54, 0.58, 0.64),
    'audi-black-suv': (0.42, 0.48, 0.58, 0.60),
    'maserati-black-coupe': (0.45, 0.56, 0.55, 0.68),
    'maserati-black-suv': (0.45, 0.48, 0.55, 0.62),
    'mercedes-black-coupe': (0.43, 0.48, 0.57, 0.62),
    'mercedes-black-suv': (0.44, 0.42, 0.56, 0.56),
}


def relative_bbox(size, family, sham=False):
    width, height = size
    x0, y0, x1, y1 = ROI_FRACTIONS[family]
    if sham:
        offset = 0.24
        x0 = min(x0 + offset, 1.0 - (x1 - x0))
        x1 = x0 + (ROI_FRACTIONS[family][2] - ROI_FRACTIONS[family][0])
    bbox = (
        max(0, round(x0 * width)),
        max(0, round(y0 * height)),
        min(width, round(x1 * width)),
        min(height, round(y1 * height)),
    )
    if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
        raise ValueError(f'Invalid ROI for {family}: {bbox} at {size}')
    return bbox


def erase_region(source_path, target_path, bbox, blur_radius=SHAM_BLUR_RADIUS):
    source_path = Path(source_path)
    target_path = Path(target_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source_path) as opened:
        image = opened.convert('RGB')
        blurred = image.filter(ImageFilter.GaussianBlur(radius=float(blur_radius)))
        image.paste(blurred.crop(bbox), bbox)
        derived_size = list(image.size)
        image.save(target_path, format='PNG', optimize=False)
        source_size = list(opened.size)
    return {
        'source_sha256': sha256(source_path),
        'derived_sha256': sha256(target_path),
        'source_size': source_size,
        'derived_size': derived_size,
        'bbox': list(bbox),
        'blur_radius': float(blur_radius),
        'transform': 'RGB GaussianBlur composite pasted only inside the recorded ROI; PNG output',
    }


def select_original_rows(rows):
    selected = []
    for family in FAMILIES:
        family_rows = [row for row in rows if row.get('family') == family and row.get('state') == 'original']
        if len(family_rows) != 1:
            raise ValueError(f'Expected one original source row for {family}: {len(family_rows)}')
        selected.append(family_rows[0])
    return selected


def _prompt_for(family, condition):
    config = FAMILIES[family]
    return f"Count the {config['element']} in this {config['brand']} logo. Return one integer only."


def build_tasks(variants):
    tasks = []
    for variant in variants:
        for repeat in range(1, REPEATS + 1):
            tasks.append({
                'task_id': f"{variant['family']}-{variant['condition']}-repeat-{repeat:02}-codex",
                'family': variant['family'], 'condition': variant['condition'],
                'agent': 'codex', 'repeat': repeat,
                'image_path': variant['derived_path'],
                'source_sha256': variant['source_sha256'],
                'normalized_sha256': variant['normalized_sha256'],
                'image_sha256': variant['derived_sha256'],
                'bbox': variant['bbox'], 'sham_bbox': variant['sham_bbox'],
                'prompt': _prompt_for(variant['family'], variant['condition']),
                'ground_truth': variant['ground_truth'],
                'expected_bias': variant.get('expected_bias'),
            })
    random.Random(SEED).shuffle(tasks)
    return tasks


def build_gemini_prompt(variants):
    lines = [
        '# Gemini Antigravity Blind Pixel-Paired Logo-Erasure and Sham Pilot', '',
        'Required model: Gemini 3.8 Flash (High). If this model cannot be selected, stop before running and report the deviation instead of substituting silently.',
        'Evaluate every listed image independently. Do not read Codex results or output files.',
        'All conditions are pixel-paired derivatives of the same normalized source image. `original` is unchanged; `logo_erased` applies the recorded blur-composite only inside the logo ROI; `sham_erased` applies the same operation and ROI size to an adjacent non-logo ROI while preserving the logo.',
        'This is a synthetic erase/sham control, not a naturalistic logo replacement. Do not call it a causal mechanism test.',
        'The 3 repeats are repeated readings on the same derived image, not independent image samples. Keep each repeat isolated.',
        'Record the full source, normalized, and derived SHA256 values plus dimensions and both ROI coordinates in your report. Write results to `GEMINI-REPORT.md` in this run folder.',
        'Keep the scientific status at `NEEDS_EVIDENCE`.', '',
        '| Task ID | Family | Condition | Repeat | Derived image | Source SHA256 | Normalized SHA256 | Derived SHA256 | Logo ROI | Sham ROI | Dimensions | Prompt |',
        '|---|---|---|---:|---|---|---|---|---|---|---|---|',
    ]
    for variant in variants:
        for repeat in range(1, REPEATS + 1):
            task_id = f"{variant['family']}-{variant['condition']}-repeat-{repeat:02}-gemini"
            dimensions = f"source={variant['source_size']}; normalized={variant['normalized_size']}; derived={variant['derived_size']}"
            lines.append(
                f"| {task_id} | {variant['family']} | {variant['condition']} | {repeat} | "
                f"`{Path(variant['derived_path']).resolve()}` | `{variant['source_sha256']}` | "
                f"`{variant['normalized_sha256']}` | `{variant['derived_sha256']}` | `{variant['bbox']}` | "
                f"`{variant['sham_bbox']}` | {dimensions} | {_prompt_for(variant['family'], variant['condition'])} |"
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
    rows = select_original_rows(source_manifest['rows'])
    variants = []
    for row in rows:
        source_path = Path(row['local_path']).resolve()
        if sha256(source_path) != row['sha256']:
            raise ValueError(f'Source hash mismatch: {source_path}')
        normalized_path = folder / 'images' / f"{row['family']}-normalized-r{MATCHED_BLUR_MAX_EDGE}.png"
        normalization = normalize_source_variant(source_path, normalized_path, MATCHED_BLUR_MAX_EDGE)
        with Image.open(normalized_path) as normalized:
            size = normalized.size
        logo_bbox = relative_bbox(size, row['family'], sham=False)
        sham_bbox = relative_bbox(size, row['family'], sham=True)
        for condition in CONDITIONS:
            target = folder / 'images' / f"{row['family']}-{condition}-r{MATCHED_BLUR_MAX_EDGE}.png"
            if condition == 'original':
                target.write_bytes(normalized_path.read_bytes())
                transform = {
                    'source_sha256': sha256(normalized_path),
                    'derived_sha256': sha256(target),
                    'source_size': list(size), 'derived_size': list(size),
                    'bbox': list(logo_bbox), 'sham_bbox': list(sham_bbox),
                    'blur_radius': 0.0,
                    'transform': 'Byte-identical copy of normalized source; no ROI edit',
                }
            else:
                bbox = logo_bbox if condition == 'logo_erased' else sham_bbox
                transform = erase_region(normalized_path, target, bbox)
                transform['bbox'] = list(logo_bbox)
                transform['sham_bbox'] = list(sham_bbox)
            variant = {
                **row, **normalization, **transform,
                'condition': condition,
                'source_state': 'original',
                'source_path': str(source_path),
                'normalized_path': str(normalized_path),
                'derived_path': str(target),
                'max_edge': MATCHED_BLUR_MAX_EDGE,
                'ground_truth': 0 if condition == 'logo_erased' else row['ground_truth'],
                'expected_bias': row['ground_truth'] if condition == 'logo_erased' else None,
                'source_run': str(source_run),
            }
            # erase_region operates on the normalized file; restore the
            # original source provenance after merging its derived metadata.
            variant['source_sha256'] = normalization['source_sha256']
            variant['source_size'] = normalization['source_size']
            variant['normalized_sha256'] = normalization['normalized_sha256']
            variant['normalized_size'] = normalization['normalized_size']
            variants.append(variant)
    manifest = {
        'protocol': 'vlmbias-logo-pixel-paired-sham-v1',
        'mode': 'pixel_paired_sham',
        'created': datetime.now().astimezone().isoformat(),
        'source_run': str(source_run),
        'source_manifest_sha256': sha256(source_manifest_path),
        'families': list(FAMILIES), 'conditions': list(CONDITIONS),
        'normalization_max_edge': MATCHED_BLUR_MAX_EDGE,
        'sham_blur_radius': SHAM_BLUR_RADIUS, 'repeats': REPEATS,
        'roi_fractions': ROI_FRACTIONS,
        'scientific_status': 'NEEDS_EVIDENCE', 'variants': variants,
        'codex': {'model': 'gpt-6.1-sol', 'provider': 'teamsoclo', 'reasoning_effort': 'xhigh'},
    }
    manifest['tasks'] = build_tasks(variants)
    save(folder / 'manifest.json', manifest)
    (folder / 'GEMINI-PROMPT.md').write_text(build_gemini_prompt(variants), encoding='utf-8')
    return manifest


def summarize(folder, manifest, results, calls):
    state = json.loads((Path(folder) / 'state.json').read_text(encoding='utf-8'))
    lines = [
        '# Pixel-paired logo-erasure and sham-control pilot', '',
        f"Operational: {state['status']}", 'Scientific: NEEDS_EVIDENCE',
        f'Codex launches: {calls}/{len(manifest["tasks"])}', '',
        f"Protocol: six original source families, pixel-paired conditions {manifest['conditions']}, max edge {manifest['normalization_max_edge']}, {REPEATS} repeats per derived image.",
        'The erase/sham operation is a synthetic robustness control, not a naturalistic edit or causal intervention.', '',
        '| Condition | Correct | Bias-aligned | Valid N |', '|---|---:|---:|---:|',
    ]
    for condition in CONDITIONS:
        subset = [r for r in results if r['condition'] == condition and r['count'] is not None]
        correct = sum(r['count'] == r['ground_truth'] for r in subset)
        biased = sum(r['count'] == r['expected_bias'] for r in subset if r['expected_bias'] is not None)
        lines.append(f'| {condition} | {correct} | {biased} | {len(subset)} |')
    lines += ['', '| Family | Condition | Correct | Bias-aligned | Valid N |', '|---|---|---:|---:|---:|']
    for family in manifest['families']:
        for condition in CONDITIONS:
            subset = [r for r in results if r['family'] == family and r['condition'] == condition and r['count'] is not None]
            correct = sum(r['count'] == r['ground_truth'] for r in subset)
            biased = sum(r['count'] == r['expected_bias'] for r in subset if r['expected_bias'] is not None)
            lines.append(f'| {family} | {condition} | {correct} | {biased} | {len(subset)} |')
    failures = [r for r in results if not r['valid']]
    lines += ['', f'Failed/invalid Codex tasks: {len(failures)}']
    for result in failures:
        lines.append(f"- {result['task_id']}: {result.get('error', 'invalid integer response')}")
    lines += ['', 'Results are descriptive pixel-paired/sham evidence only; they do not establish causality or universal generalization.', 'Scientific status remains `NEEDS_EVIDENCE`.']
    (Path(folder) / 'SUMMARY.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--source-run', type=Path, default=SOURCE_RUN_DEFAULT)
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
        run_id = datetime.now().strftime('logo-paired-sham-%Y%m%d-%H%M%S-%f')
        folder = ROOT / 'collab' / 'experiment' / run_id
        manifest = prepare_run(folder, args.source_run)
        print(f'Gemini prompt: {folder / "GEMINI-PROMPT.md"}', flush=True)
        if args.dry_run:
            print(f'Dry run: {len(manifest["tasks"])} tasks, 0 model launches; {folder}')
            return 0
    results, state = run_tasks(folder, manifest['tasks'], args.timeout, invoke_fn=invoke)
    summarize(folder, manifest, results, state['calls'])
    print(folder / 'SUMMARY.md')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
