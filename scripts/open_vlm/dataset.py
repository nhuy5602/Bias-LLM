"""Pinned VLMBias loader and run-folder preparation helpers."""

import hashlib
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path

from .schema import DATASET_REVISION, DEFAULT_DATASET, default_domain_specs, validate_calibration_row

BASE_ROWS_URL = 'https://datasets-server.huggingface.co/rows'
PAGE_SIZE = 100
DOMAIN_ALIASES = {
    'animals': 'animals',
    'animal': 'animals',
    'logos': 'logos',
    'logo': 'logos',
    'flags': 'flags',
    'flag': 'flags',
    'boards': 'boards',
    'board': 'boards',
    'chess': 'chess',
    'grids': 'grids',
    'grid': 'grids',
    'illusions': 'illusions',
    'illusion': 'illusions',
}


def _json_response(response):
    data = response.read()
    if isinstance(data, bytes):
        data = data.decode('utf-8')
    return json.loads(data)


def _open_with_retry(request, opener, retries=4, backoff_seconds=1.0):
    transient = {429, 500, 502, 503, 504}
    for attempt in range(retries + 1):
        try:
            return opener(request, timeout=60)
        except urllib.error.HTTPError as exc:
            if exc.code not in transient or attempt >= retries:
                raise
            time.sleep(backoff_seconds * (2 ** attempt))
        except urllib.error.URLError:
            if attempt >= retries:
                raise
            time.sleep(backoff_seconds * (2 ** attempt))


def fetch_rows(split='main', offsets=(0,), dataset=DEFAULT_DATASET,
               revision=DATASET_REVISION, page_size=PAGE_SIZE, opener=None,
               retries=4, backoff_seconds=1.0):
    opener = opener or urllib.request.urlopen
    rows = []
    for offset in offsets:
        query = urllib.parse.urlencode({
            'dataset': dataset,
            'config': 'default',
            'revision': revision,
            'split': split,
            'offset': offset,
            'length': page_size,
        })
        request = urllib.request.Request(
            f'{BASE_ROWS_URL}?{query}',
            headers={'User-Agent': 'visual-prior-bias-open-vlm/1'},
        )
        with _open_with_retry(request, opener, retries=retries,
                              backoff_seconds=backoff_seconds) as response:
            rows.extend(item['row'] for item in _json_response(response).get('rows', []))
    return rows


def _metadata(row):
    value = row.get('metadata')
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


_GENERIC_SUBJECT_LABELS = {
    'animal', 'animals', 'logo', 'logos', 'subject', 'unknown',
}


def infer_subject_label(row, domain=None):
    """Return a concrete recognition label without using the answer count."""
    domain = domain or infer_domain(row)
    metadata = _metadata(row)
    values = (
        row.get('subject_label'),
        metadata.get('brand'),
        metadata.get('bias_type'),
        metadata.get('species'),
        metadata.get('animal') if isinstance(metadata.get('animal'), str) else None,
    )
    for value in values:
        label = str(value or '').strip().casefold()
        if label and label not in _GENERIC_SUBJECT_LABELS:
            return label
    source_path = str(row.get('image_path') or row.get('source_path') or '')
    stem = Path(source_path).stem.casefold()
    token = re.split(r'[_\-\s]+', stem)[0]
    if token and token not in _GENERIC_SUBJECT_LABELS and not token.isdigit():
        return token
    return str(domain or 'subject').strip().casefold()


def infer_domain(row):
    raw = str(row.get('topic') or row.get('domain') or '').strip().casefold()
    return DOMAIN_ALIASES.get(raw) or DOMAIN_ALIASES.get(raw.replace('_', '').replace(' ', ''))


def source_family(row, domain=None):
    metadata = _metadata(row)
    group = metadata.get('group_id') or row.get('group_id')
    if group not in (None, ''):
        return f'{domain or infer_domain(row) or "unknown"}:{group}'
    path = str(row.get('image_path') or row.get('id') or row.get('ID') or 'unknown')
    stem = Path(path).stem.casefold()
    stem = re.sub(r'([_-](?:original|counterfactual|\d+))+$', '', stem)
    return f'{domain or infer_domain(row) or "unknown"}:{stem}'


def _image_url(row):
    image = row.get('image')
    if isinstance(image, dict):
        return image.get('src') or image.get('url')
    return row.get('image_url') or row.get('image_path')


def normalize_row(row, domain=None):
    domain = domain or infer_domain(row)
    if not domain:
        raise ValueError(f'Cannot infer VLMBias domain from row: {row.get("topic")}')
    if row.get('expected_bias') is None:
        raise ValueError('Open-VLM rows require an explicit expected_bias prior answer')
    image_url = _image_url(row)
    if not image_url:
        raise ValueError('VLMBias row has no image URL/path')
    ground_truth = row.get('ground_truth')
    expected_bias = row.get('expected_bias')
    if domain == 'illusions':
        ground_truth = str(ground_truth).strip().casefold()
        expected_bias = str(expected_bias).strip().casefold()
    else:
        ground_truth = int(ground_truth)
        expected_bias = int(expected_bias)
    return {
        'domain': domain,
        'family': source_family(row, domain),
        'source_id': row.get('ID', row.get('id')),
        'source_path': str(row.get('image_path', '')).replace('\\', '/'),
        'image_path': str(row.get('image_path') or f"raw/{row.get('ID', row.get('id', 'row'))}.png").replace('\\', '/'),
        'image_url': image_url,
        'ground_truth': ground_truth,
        'expected_bias': expected_bias,
        'prompt': row.get('prompt', ''),
        'metadata': _metadata(row),
        'subject_label': infer_subject_label(row, domain),
    }


def calibration_source_family(row, domain=None):
    """Derive a stable family key for an original-split calibration row."""
    domain = domain or infer_domain(row) or 'unknown'
    source_id = str(row.get('ID', row.get('id', ''))).strip().casefold()
    if domain == 'animals' and source_id:
        return f"animals:{source_id.split('_prompt', 1)[0]}"
    metadata = _metadata(row)
    brand = str(metadata.get('bias_type') or '').strip().casefold()
    path = str(row.get('image_path') or '').replace('\\', '/').casefold()
    stem = Path(path).stem
    stem = re.sub(r'[_-](?:original|counterfactual|\d+)$', '', stem)
    tokens = [token for token in re.split(r'[_\-\s]+', stem) if token]
    parsed_color = None
    if domain == 'logos' and not brand and len(tokens) >= 3:
        brand, parsed_color = tokens[0], tokens[1]
        tokens = tokens[2:]
    if domain == 'logos' and brand:
        color = str(metadata.get('color') or parsed_color or 'black').strip().casefold()
        body = str(metadata.get('body_type') or '').strip().casefold()
        body = Path(body).stem
        if not body:
            body = '-'.join(token for token in tokens if token not in {brand, color}) or 'unknown'
        return f'logos:{brand}-{color}-{body}'
    return f'{domain}:{stem or source_id or "unknown"}'


def normalize_calibration_row(row, domain=None):
    domain = domain or infer_domain(row)
    if domain not in {'animals', 'logos'}:
        raise ValueError(f'Calibration is registered only for animals/logos, got {domain}')
    if row.get('expected_bias') is not None:
        raise ValueError('Original calibration rows require expected_bias=null')
    image_url = _image_url(row)
    if not image_url:
        raise ValueError('Calibration row has no image URL/path')
    metadata = _metadata(row)
    normalized = {
        'domain': domain,
        'family': calibration_source_family(row, domain),
        'source_id': row.get('ID', row.get('id')),
        'source_path': str(row.get('image_path', '')).replace('\\', '/'),
        'image_path': str(row.get('image_path') or f"raw/{row.get('ID', row.get('id', 'row'))}.png").replace('\\', '/'),
        'image_url': image_url,
        'ground_truth': row.get('ground_truth'),
        'expected_bias': None,
        'prompt': row.get('prompt', ''),
        'metadata': metadata,
        'subject_label': infer_subject_label(row, domain),
        'source_split': 'original',
        'type_of_question': row.get('type_of_question'),
    }
    spec = default_domain_specs()[domain]
    return validate_calibration_row(normalized, spec)


def load_vlmbias_rows(domains, max_per_domain=20, page_limit=100, split='main', opener=None):
    domains = tuple(dict.fromkeys(domain.casefold() for domain in domains))
    rows = []
    for offset in range(0, page_limit * PAGE_SIZE, PAGE_SIZE):
        page = fetch_rows(split=split, offsets=(offset,), page_size=PAGE_SIZE, opener=opener)
        if not page:
            break
        for row in page:
            try:
                normalized = normalize_row(row)
            except (TypeError, ValueError):
                continue
            if normalized['domain'] in domains:
                rows.append(normalized)
        counts = {domain: sum(row['domain'] == domain for row in rows) for domain in domains}
        if all(counts[domain] >= max_per_domain for domain in domains):
            break
    selected = []
    for domain in domains:
        domain_rows = sorted(
            (row for row in rows if row['domain'] == domain),
            key=lambda row: (row['family'], str(row.get('source_id', '')), row['source_path']),
        )
        unique = {}
        for row in domain_rows:
            unique.setdefault(row['family'], row)
        selected.extend(list(unique.values())[:max_per_domain])
    missing = [domain for domain in domains if not any(row['domain'] == domain for row in selected)]
    if missing:
        raise ValueError(f'No pinned VLMBias rows found for domains: {missing}')
    return selected


def load_vlmbias_calibration_rows(domains=('animals', 'logos'), max_per_domain=5,
                                  page_limit=100, opener=None):
    """Load deterministic, unmodified original-split rows for Step 0."""
    domains = tuple(dict.fromkeys(domain.casefold() for domain in domains))
    rows = []
    for offset in range(0, page_limit * PAGE_SIZE, PAGE_SIZE):
        page = fetch_rows(split='original', offsets=(offset,), page_size=PAGE_SIZE, opener=opener)
        if not page:
            break
        for row in page:
            if row.get('type_of_question') not in (None, 'Counting/Illusion'):
                continue
            try:
                normalized = normalize_calibration_row(row)
            except (TypeError, ValueError):
                continue
            if normalized['domain'] in domains:
                rows.append(normalized)
        if all(sum(row['domain'] == domain for row in rows) >= max_per_domain
               for domain in domains):
            break
    selected = []
    for domain in domains:
        domain_rows = sorted((row for row in rows if row['domain'] == domain),
                             key=lambda row: (row['family'], str(row.get('source_id', ''))))
        unique = {}
        for row in domain_rows:
            unique.setdefault(row['family'], row)
        selected.extend(list(unique.values())[:max_per_domain])
    missing = [domain for domain in domains if not any(row['domain'] == domain for row in selected)]
    if missing:
        raise ValueError(f'No original VLMBias calibration rows found for domains: {missing}')
    return selected


def _sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def download_rows(rows, run_dir, opener=None, image_subdir='images'):
    opener = opener or urllib.request.urlopen
    image_dir = Path(run_dir) / image_subdir
    image_dir.mkdir(parents=True, exist_ok=True)
    downloaded = []
    for index, row in enumerate(rows):
        safe_family = re.sub(r'[^A-Za-z0-9_.-]+', '_', row['family'])
        path = image_dir / f'{index:04d}-{safe_family}.png'
        request = urllib.request.Request(row['image_url'], headers={'User-Agent': 'visual-prior-bias-open-vlm/1'})
        with opener(request, timeout=60) as response:
            data = response.read()
        path.write_bytes(data)
        downloaded.append({
            **row,
            'image_path': path.relative_to(run_dir).as_posix(),
            'source_sha256': _sha256_bytes(data),
            'source_size_bytes': len(data),
        })
    return downloaded


def prepare_vlmbias_run(root, phase, domains, config, max_per_domain=20,
                        calibration_max_per_domain=None, opener=None):
    from .runner import prepare_run

    rows = load_vlmbias_rows(domains, max_per_domain=max_per_domain, opener=opener)
    config = dict(config)
    config['source_root'] = ''
    config['prepare_only'] = True
    config['require_controls'] = True
    calibration_rows = []
    if calibration_max_per_domain:
        calibration_rows = load_vlmbias_calibration_rows(
            domains, max_per_domain=calibration_max_per_domain, opener=opener)
        config['step0_rows'] = calibration_rows
        config['step0_calibration'] = True
        config['step0_thresholds'] = {
            'min_families': len(domains) * calibration_max_per_domain,
            'min_clear_accuracy': 0.80,
            'min_subject_validity': 0.90,
            'min_detail_drop': 0.20,
            'max_black_subject_validity': 0.10,
            'max_invalid_rate': 0.05,
        }
    run_dir = prepare_run(root, phase, rows, config)
    downloaded = download_rows(rows, run_dir, opener=opener)
    downloaded_calibration = download_rows(
        calibration_rows, run_dir, opener=opener, image_subdir='calibration-images')
    manifest_path = run_dir / 'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    manifest['source_root'] = run_dir.as_posix()
    downloaded_by_id = {str(row.get('source_id')): row for row in downloaded}
    manifest['rows'] = [
        {**row, **downloaded_by_id.get(str(row.get('source_id')), {})}
        for row in manifest['rows']
    ]
    if calibration_rows:
        manifest['step0_rows'] = downloaded_calibration
    manifest['prepared_from'] = {
        'dataset': DEFAULT_DATASET,
        'revision': DATASET_REVISION,
        'split': 'main',
        'prepared_at': datetime.now().astimezone().isoformat(),
    }
    manifest['license_attestations'] = config.get('license_attestations', {
        'dataset': 'pending', 'model': 'pending',
    })
    manifest['execution_blocked'] = any(
        value != 'verified' for value in manifest['license_attestations'].values())
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=True), encoding='utf-8')
    return run_dir
