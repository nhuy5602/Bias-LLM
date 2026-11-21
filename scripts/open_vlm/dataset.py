"""Pinned VLMBias loader and run-folder preparation helpers."""

import hashlib
import json
import random
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path

from .schema import DATASET_REVISION, DEFAULT_DATASET, default_domain_specs, validate_calibration_row

BASE_ROWS_URL = 'https://datasets-server.huggingface.co/rows'
PAGE_SIZE = 100
DATASET_REQUEST_INTERVAL_SECONDS = 1.0
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


def _retry_delay(exc, attempt, backoff_seconds):
    fallback = backoff_seconds * (2 ** attempt)
    retry_after = exc.headers.get('Retry-After') if exc.headers else None
    if not retry_after:
        return fallback
    try:
        return max(0.0, float(retry_after))
    except ValueError:
        try:
            retry_at = parsedate_to_datetime(retry_after)
        except (TypeError, ValueError, OverflowError):
            return fallback
        if retry_at.tzinfo is None:
            retry_at = retry_at.replace(tzinfo=timezone.utc)
        return max(0.0, (retry_at - datetime.now(timezone.utc)).total_seconds())


def _paced_opener(opener, minimum_interval_seconds, clock=time.monotonic, sleep=time.sleep):
    previous_request = [None]

    def open_paced(request, timeout):
        now = clock()
        if previous_request[0] is not None:
            delay = minimum_interval_seconds - (now - previous_request[0])
            if delay > 0:
                sleep(delay)
        previous_request[0] = clock()
        return opener(request, timeout=timeout)

    return open_paced


def _open_with_retry(request, opener, retries=4, backoff_seconds=1.0):
    transient = {429, 500, 502, 503, 504}
    for attempt in range(retries + 1):
        try:
            return opener(request, timeout=60)
        except urllib.error.HTTPError as exc:
            if exc.code not in transient or attempt >= retries:
                raise
            time.sleep(_retry_delay(exc, attempt, backoff_seconds))
        except urllib.error.URLError:
            if attempt >= retries:
                raise
            time.sleep(backoff_seconds * (2 ** attempt))


def fetch_rows(split='main', offsets=(0,), dataset=DEFAULT_DATASET,
               revision=DATASET_REVISION, page_size=PAGE_SIZE, opener=None,
               retries=4, backoff_seconds=1.0, page_cache=None, where=None):
    opener = opener or urllib.request.urlopen
    rows = []
    for offset in offsets:
        cache_key = (dataset, revision, split, offset, page_size, where)
        if page_cache is not None and cache_key in page_cache:
            rows.extend(page_cache[cache_key])
            continue
        query_values = {
            'dataset': dataset,
            'config': 'default',
            'revision': revision,
            'split': split,
            'offset': offset,
            'length': page_size,
        }
        if where is not None:
            query_values['where'] = where
        query = urllib.parse.urlencode(query_values)
        endpoint = 'filter' if where is not None else 'rows'
        request = urllib.request.Request(
            f'https://datasets-server.huggingface.co/{endpoint}?{query}',
            headers={'User-Agent': 'visual-prior-bias-open-vlm/1'},
        )
        with _open_with_retry(request, opener, retries=retries,
                              backoff_seconds=backoff_seconds) as response:
            page = [item['row'] for item in _json_response(response).get('rows', [])]
        if page_cache is not None:
            page_cache[cache_key] = page
        rows.extend(page)
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
    source_id = str(row.get('ID', row.get('id', ''))).strip().casefold()
    source_id = source_id.split('_prompt', 1)[0]
    if source_id:
        source_id = re.sub(r'^animal_\d+_', '', source_id)
        source_id = re.sub(r'_notitle.*$', '', source_id)
        tokens = [token for token in re.split(r'[_\-\s]+', source_id) if token]
        if tokens and tokens[0] not in {'animal', 'shoe', 'logo'}:
            return ' '.join(tokens)
    source_path = str(row.get('image_path') or row.get('source_path') or '')
    stem = Path(source_path).stem.casefold()
    tokens = [token for token in re.split(r'[_\-\s]+', stem) if token]
    if tokens and tokens[0] not in _GENERIC_SUBJECT_LABELS and not tokens[0].isdigit():
        return ' '.join(tokens)
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
        'source_split': row.get('source_split') or 'main',
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


def _calibration_stratum(row):
    if row['domain'] == 'animals':
        return f"count:{row.get('ground_truth')}"
    if row['domain'] == 'logos':
        metadata = row.get('metadata') or {}
        family_label = str(row.get('family', '')).split(':', 1)[-1]
        brand = metadata.get('bias_type') or family_label.split('-', 1)[0]
        return f'brand:{str(brand).casefold()}'
    return str(row.get('family', 'unknown'))


def _select_stratified_calibration_rows(rows, domain, limit, seed, excluded_families=()):
    excluded_families = set(excluded_families)
    unique = {}
    for row in sorted(rows, key=lambda item: (item['family'], str(item.get('source_id', '')))):
        if row['family'] not in excluded_families:
            unique.setdefault(row['family'], row)

    strata = {}
    for row in unique.values():
        strata.setdefault(_calibration_stratum(row), []).append(row)

    rng = random.Random(f'{seed}:{domain}:original-calibration')
    stratum_names = sorted(strata)
    rng.shuffle(stratum_names)
    for name in stratum_names:
        rng.shuffle(strata[name])

    selected = []
    while len(selected) < limit:
        added = False
        for name in stratum_names:
            if strata[name]:
                selected.append(strata[name].pop())
                added = True
                if len(selected) == limit:
                    break
        if not added:
            break
    return selected


def load_vlmbias_rows(domains, max_per_domain=20, page_limit=100, split='main', opener=None,
                      seed=20261024, excluded_families=(), page_cache=None):
    domains = tuple(dict.fromkeys(domain.casefold() for domain in domains))
    excluded_families = set(excluded_families)
    rows = []
    for offset in range(0, page_limit * PAGE_SIZE, PAGE_SIZE):
        page = fetch_rows(split=split, offsets=(offset,), page_size=PAGE_SIZE,
                          opener=opener, page_cache=page_cache)
        if not page:
            break
        for row in page:
            try:
                normalized = normalize_row(row)
            except (TypeError, ValueError):
                continue
            if normalized['domain'] in domains and normalized['family'] not in excluded_families:
                rows.append(normalized)
        counts = {
            domain: len({row['family'] for row in rows if row['domain'] == domain})
            for domain in domains
        }
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
        families = list(unique)
        random.Random(f'{seed}:{domain}:{split}').shuffle(families)
        selected.extend(unique[family] for family in families[:max_per_domain])
    missing = [domain for domain in domains if not any(row['domain'] == domain for row in selected)]
    if missing:
        raise ValueError(f'No pinned VLMBias rows found for domains: {missing}')
    insufficient = [
        domain for domain in domains
        if len({row['family'] for row in selected if row['domain'] == domain}) < max_per_domain
    ]
    if insufficient:
        raise ValueError(
            f'VLMBias split {split!r} requires {max_per_domain} distinct families per domain; '
            f'insufficient families for: {insufficient}')
    return selected


def visual_icl_animal_family(row):
    source_image = (row.get('metadata') or {}).get('source_image')
    if not isinstance(source_image, str):
        raise ValueError('Visual ICL requires the VLMBias source_image metadata')
    stem = Path(source_image).stem.casefold()
    animal = re.sub(r'(?:_\d+)+$', '', stem)
    animal = re.sub(r'[^a-z0-9]', '', animal)
    if not animal:
        raise ValueError('Visual ICL source_image has no animal family name')
    return f'animals:{animal}'


def load_visual_icl_rows(limit, seed, opener, page_cache):
    """Collect equal numbers of 3-leg and 5-leg source families."""
    strata = {3: {}, 5: {}}
    seen_families = set()
    per_label = limit // 2
    for label in sorted(strata):
        where = f'"topic" = \'Animals\' AND "ground_truth" = {label}'
        for offset in range(0, 100 * PAGE_SIZE, PAGE_SIZE):
            page = fetch_rows(offsets=(offset,), opener=opener, page_cache=page_cache, where=where)
            for raw in page:
                row = normalize_row(raw)
                family = visual_icl_animal_family(row)
                if row['ground_truth'] == label and row['expected_bias'] != label and family not in seen_families:
                    strata[label][family] = {**row, 'family': family}
                    seen_families.add(family)
            if len(strata[label]) >= per_label or len(page) < PAGE_SIZE:
                break
    if any(len(families) < per_label for families in strata.values()):
        available = {label: len(families) for label, families in strata.items()}
        raise ValueError(
            f'Visual ICL needs {per_label} distinct families for each counterfactual label (3, 5); '
            f'found {available}')
    rng = random.Random(seed)
    return [families[family] for families in strata.values()
            for family in rng.sample(sorted(families), per_label)]


def _calibration_pool_ready(rows, domain, limit, excluded_families):
    unique = {}
    for row in rows:
        if row['family'] not in excluded_families:
            unique.setdefault(row['family'], row)
    if len(unique) < limit:
        return False

    strata = {_calibration_stratum(row) for row in unique.values()}
    if domain == 'animals':
        minimum_per_count = (limit + 1) // 2
        return all(sum(row.get('ground_truth') == count for row in unique.values())
                   >= minimum_per_count for count in (2, 4))
    if domain == 'logos':
        return len(strata) >= min(2, limit)
    return True


def load_vlmbias_calibration_rows(domains=('animals', 'logos'), max_per_domain=5,
                                  page_limit=100, opener=None, seed=20261024,
                                  excluded_families=(), page_cache=None, family_key=None, where=None):
    """Load seeded, stratified, unmodified original-split rows for Step 0."""
    domains = tuple(dict.fromkeys(domain.casefold() for domain in domains))
    excluded_families = set(excluded_families)
    rows = []
    for offset in range(0, page_limit * PAGE_SIZE, PAGE_SIZE):
        page = fetch_rows(split='original', offsets=(offset,), page_size=PAGE_SIZE,
                          opener=opener, page_cache=page_cache, where=where)
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
                if family_key is not None:
                    normalized['family'] = family_key(normalized)
                rows.append(normalized)
        if all(_calibration_pool_ready(
                [row for row in rows if row['domain'] == domain], domain,
                max_per_domain, excluded_families) for domain in domains):
            break
        if len(page) < PAGE_SIZE:
            break
    selected = []
    for domain in domains:
        domain_rows = [row for row in rows if row['domain'] == domain]
        selected.extend(_select_stratified_calibration_rows(
            domain_rows, domain, max_per_domain, seed, excluded_families))
    insufficient = [
        domain for domain in domains
        if len({row['family'] for row in selected if row['domain'] == domain}) < max_per_domain
    ]
    if insufficient:
        raise ValueError(
            f'Original VLMBias calibration requires {max_per_domain} distinct families per domain; '
            f'insufficient families for: {insufficient}')
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
        with _open_with_retry(request, opener) as response:
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
    from .visual_icl import PHASE as VISUAL_ICL_PHASE, stratified_split, register_visual_icl

    request_interval_seconds = DATASET_REQUEST_INTERVAL_SECONDS if opener is None else 0.0
    if opener is None:
        opener = _paced_opener(urllib.request.urlopen, request_interval_seconds)
    page_cache = {}
    seed = int(config.get('seed', 20261024))
    visual_icl = phase == VISUAL_ICL_PHASE
    if visual_icl:
        if (list(domains) != ['animals'] or max_per_domain < 40 or max_per_domain % 4
                or (calibration_max_per_domain or 0) < 20):
            raise ValueError('Visual ICL requires animals, a multiple of 4 main families (minimum 40), and 20+ originals')
        rows = load_visual_icl_rows(max_per_domain, seed, opener, page_cache)
    else:
        rows = load_vlmbias_rows(domains, max_per_domain=max_per_domain, opener=opener,
                                 seed=seed, page_cache=page_cache)
    config = dict(config)
    config['source_root'] = ''
    config['prepare_only'] = True
    config['require_controls'] = True
    if visual_icl:
        config['dev_fraction'] = 0.5
    elif calibration_max_per_domain:
        config['dev_fraction'] = min(float(config.get('dev_fraction', 0.8)), 0.75)
    calibration_rows = []
    bias_rows = []
    if calibration_max_per_domain:
        main_families = {row['family'] for row in rows}
        excluded_from_originals = (set(stratified_split(rows, seed)['dev_families'])
                                   if visual_icl else main_families)
        calibration_rows = load_vlmbias_calibration_rows(
            domains, max_per_domain=calibration_max_per_domain, opener=opener, seed=seed,
            excluded_families=excluded_from_originals, page_cache=page_cache,
            **({'family_key': visual_icl_animal_family, 'where': '"topic" = \'Animals\''}
               if visual_icl else {}))
        config['step0_rows'] = calibration_rows
    if calibration_max_per_domain and not visual_icl:
        bias_excluded_families = main_families | {row['family'] for row in calibration_rows}
        bias_rows = load_vlmbias_rows(
            domains, max_per_domain=calibration_max_per_domain, split='main', opener=opener,
            seed=seed, excluded_families=bias_excluded_families, page_cache=page_cache)
        config['step0_excluded_families'] = sorted(main_families)
        config['step0_bias_excluded_families'] = sorted(bias_excluded_families)
        config['step0_bias_rows'] = bias_rows
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
    downloaded_bias = download_rows(
        bias_rows, run_dir, opener=opener, image_subdir='step0-bias-images')
    manifest_path = run_dir / 'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    manifest['source_root'] = run_dir.as_posix()
    downloaded_by_id = {str(row.get('source_id')): row for row in downloaded}
    manifest['rows'] = [
        {**row, **downloaded_by_id.get(str(row.get('source_id')), {})}
        for row in manifest['rows']
    ]
    if config.get('step0_bias_rows'):
        manifest['step0_bias_rows'] = downloaded_bias
    if calibration_rows:
        manifest['step0_rows'] = downloaded_calibration
    if visual_icl:
        register_visual_icl(manifest)
    manifest['prepared_from'] = {
        'dataset': DEFAULT_DATASET,
        'revision': DATASET_REVISION,
        'split': 'main',
        'prepared_at': datetime.now().astimezone().isoformat(),
        'request_interval_seconds': request_interval_seconds,
    }
    manifest['license_attestations'] = config.get('license_attestations', {
        'dataset': 'pending', 'model': 'pending',
    })
    manifest['execution_blocked'] = any(
        value != 'verified' for value in manifest['license_attestations'].values())
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=True), encoding='utf-8')
    return run_dir
