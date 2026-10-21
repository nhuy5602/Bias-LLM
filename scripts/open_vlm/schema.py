"""Manifest and answer-contract primitives for the open-VLM track."""

from collections import defaultdict
from dataclasses import dataclass
import copy
import random
from pathlib import Path

DATASET_REVISION = '3761f9fd7163577534a7816c8bc1004035f2e2a0'
DEFAULT_DATASET = 'anvo25/vlms-are-biased'
SCIENTIFIC_STATUS = 'NEEDS_EVIDENCE'


@dataclass(frozen=True)
class DomainSpec:
    name: str
    answer_kind: str
    candidate_values: tuple
    prior_fields: tuple

    def __post_init__(self):
        if self.answer_kind not in {'integer', 'categorical'}:
            raise ValueError(f'Unsupported answer kind: {self.answer_kind}')
        if not self.candidate_values or not self.prior_fields:
            raise ValueError('DomainSpec requires candidates and prior fields')


def default_domain_specs():
    integer = tuple(range(21))
    return {
        name: DomainSpec(name, 'integer', integer, ('expected_bias',))
        for name in ('animals', 'logos', 'flags', 'boards')
    } | {
        'illusions': DomainSpec('illusions', 'categorical', ('yes', 'no'), ('expected_bias',)),
    }


def _normalize_answer(value, domain):
    if domain.answer_kind == 'integer':
        if isinstance(value, bool):
            raise ValueError(f'{domain.name} answer must be an integer')
        try:
            value = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f'{domain.name} answer must be an integer') from exc
    else:
        value = str(value).strip().casefold()
    candidates = tuple(_normalize_candidate(candidate, domain.answer_kind)
                       for candidate in domain.candidate_values)
    if value not in candidates:
        raise ValueError(f'{domain.name} answer is outside the candidate set')
    return value


def _normalize_candidate(value, answer_kind):
    if answer_kind == 'integer':
        if isinstance(value, bool):
            raise ValueError('Boolean is not an integer candidate')
        return int(value)
    return str(value).strip().casefold()


def validate_row(row: dict, domain: DomainSpec) -> dict:
    if not isinstance(row, dict):
        raise ValueError('Dataset row must be a mapping')
    if row.get('domain', domain.name) != domain.name:
        raise ValueError(f'Row domain does not match {domain.name}')
    family = str(row.get('family', '')).strip()
    if not family:
        raise ValueError('Dataset row requires a source family')
    image_path = str(row.get('image_path', '')).strip()
    if not image_path:
        raise ValueError('Dataset row requires an image path')
    if 'ground_truth' not in row or row['ground_truth'] is None:
        raise ValueError('Dataset row requires a ground-truth answer')

    prior_values = []
    for field in domain.prior_fields:
        if field in row and row[field] is not None:
            prior_values.append(_normalize_answer(row[field], domain))
    if not prior_values:
        raise ValueError('Dataset row requires a prior answer field')
    if len(set(prior_values)) != 1:
        raise ValueError('Dataset row has ambiguous prior answer fields')

    normalized = dict(row)
    normalized.update({
        'domain': domain.name,
        'family': family,
        'image_path': Path(image_path).as_posix(),
        'answer_kind': domain.answer_kind,
        'candidate_values': [_normalize_candidate(v, domain.answer_kind)
                             for v in domain.candidate_values],
        'ground_truth': _normalize_answer(row['ground_truth'], domain),
        'prior_answer': prior_values[0],
    })
    return normalized


def _row_key(row):
    return (str(row.get('family', '')), str(row.get('row_id', row.get('image_path', ''))))


def split_families(rows: list[dict], seed: int, dev_fraction: float = 0.8) -> dict[str, list[dict]]:
    if not 0 < dev_fraction < 1:
        raise ValueError('dev_fraction must be between 0 and 1')
    grouped = defaultdict(list)
    for row in rows:
        family = str(row.get('family', '')).strip()
        if not family:
            raise ValueError('Every row requires a source family')
        grouped[family].append(row)
    families = sorted(grouped)
    random.Random(seed).shuffle(families)
    dev_count = max(1, min(len(families) - 1, int(len(families) * dev_fraction))) if len(families) > 1 else len(families)
    dev_families = set(families[:dev_count])
    dev = sorted((row for family, group in grouped.items() if family in dev_families for row in group), key=_row_key)
    same_domain_test = sorted((row for family, group in grouped.items() if family not in dev_families for row in group), key=_row_key)
    return {'dev': dev, 'same_domain_test': same_domain_test}


def build_manifest(rows, config):
    config = copy.deepcopy(config)
    if config.get('dataset_revision') != DATASET_REVISION:
        raise ValueError(f'Dataset revision must be {DATASET_REVISION}')
    attestations = config.get('license_attestations', {})
    if any(attestations.get(name) != 'verified' for name in ('dataset', 'model')):
        raise ValueError('Dataset and model license attestations must be verified')

    specs = default_domain_specs()
    specs.update(config.get('domain_specs', {}))
    normalized = []
    for row in rows:
        domain_name = row.get('domain')
        if domain_name not in specs:
            raise ValueError(f'No verified domain contract for {domain_name}')
        normalized.append(validate_row(row, specs[domain_name]))

    seed = int(config.get('seed', 0))
    family_split = {}
    for domain_name in sorted({row['domain'] for row in normalized}):
        domain_rows = [row for row in normalized if row['domain'] == domain_name]
        split = split_families(domain_rows, seed)
        families = sorted({row['family'] for row in domain_rows})
        family_split[domain_name] = {
            'family_count': len(families),
            'dev_families': sorted({row['family'] for row in split['dev']}),
            'same_domain_test_families': sorted({row['family'] for row in split['same_domain_test']}),
            'eligible': len(families) >= 5,
        }

    return {
        'protocol': 'open-vlm-v1',
        'dataset': config.get('dataset', DEFAULT_DATASET),
        'dataset_revision': DATASET_REVISION,
        'license_attestations': attestations,
        'seed': seed,
        'processor': config.get('processor', {}),
        'model': config.get('model', {}),
        'runtime': config.get('runtime', {}),
        'method': config.get('method', {}),
        'family_split': family_split,
        'rows': sorted(normalized, key=_row_key),
        'scientific_status': SCIENTIFIC_STATUS,
    }
