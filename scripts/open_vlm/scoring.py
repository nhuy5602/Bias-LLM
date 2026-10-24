"""Answer scoring, clustered uncertainty, and preregistered gates."""

from collections import defaultdict
import math
import random
import re
from statistics import mean

DEVELOPMENT_BIAS_DELTA = 0.05
HELD_OUT_BIAS_DELTA = 0.10
MAX_ACCURACY_DROP = 0.05
MAX_OTHER_ERROR_INCREASE = 0.05
MAX_INVALID_RATE = 0.05
LABELS = ('Correct', 'Bias-aligned', 'Other errors')


def parse_free_generation(raw, candidates):
    """Accept only an exact registered answer, preserving free-generation invalids."""
    text = str(raw).strip()
    normalized = {str(candidate).strip().casefold(): str(candidate).strip() for candidate in candidates}
    return normalized.get(text.casefold())


def _aggregate_score(raw_score):
    if isinstance(raw_score, (int, float)):
        score = float(raw_score)
    else:
        score = sum(float(piece) for piece in raw_score)
    if not math.isfinite(score):
        raise ValueError('Candidate score must be finite')
    return score


def score_candidates(candidates: list[str], sequence_score) -> dict:
    if not candidates:
        raise ValueError('Candidate set must not be empty')
    scores = {candidate: _aggregate_score(sequence_score(candidate)) for candidate in candidates}
    best = max(enumerate(candidates), key=lambda item: (scores[item[1]], -item[0]))[1]
    return {
        'prediction': best,
        'scores': scores,
        'candidate_count': len(candidates),
        'score_type': 'sequence_logprob',
    }


def _canonical_answer(value):
    if value is None:
        return None
    if isinstance(value, bool):
        return str(value).casefold()
    if isinstance(value, (int, float)):
        return value
    text = str(value).strip().casefold()
    try:
        return int(text)
    except ValueError:
        return text


def classify_answer(prediction, row) -> str:
    prediction = _canonical_answer(prediction)
    if prediction == _canonical_answer(row.get('ground_truth')):
        return 'Correct'
    if prediction == _canonical_answer(row.get('prior_answer', row.get('expected_bias'))):
        return 'Bias-aligned'
    return 'Other errors'


def summarize(rows) -> dict:
    counts = {label: 0 for label in LABELS}
    for row in rows:
        counts[classify_answer(row.get('prediction'), row)] += 1
    total = sum(counts.values())
    denominator = total or 1
    return {
        'counts': counts,
        'total': total,
        'accuracy': counts['Correct'] / denominator,
        'bias_rate': counts['Bias-aligned'] / denominator,
        'other_error_rate': counts['Other errors'] / denominator,
        'invalid_rate': sum(row.get('prediction') is None for row in rows) / denominator,
    }


def summarize_clustered(rows) -> dict:
    grouped = defaultdict(list)
    for row in rows:
        key = (row.get('domain', ''), row.get('family'))
        grouped[key].append(row)
    if not grouped:
        return {**summarize(rows), 'family_count': 0, 'clustered': True}
    summaries = [summarize(group) for group in grouped.values()]
    count = len(summaries)
    return {
        'counts': {label: mean(summary['counts'][label] / max(summary['total'], 1) for summary in summaries)
                   for label in LABELS},
        'total': sum(summary['total'] for summary in summaries),
        'accuracy': mean(summary['accuracy'] for summary in summaries),
        'bias_rate': mean(summary['bias_rate'] for summary in summaries),
        'other_error_rate': mean(summary['other_error_rate'] for summary in summaries),
        'invalid_rate': mean(summary['invalid_rate'] for summary in summaries),
        'family_count': count,
        'clustered': True,
    }


def _quantile(values, probability):
    values = sorted(values)
    if not values:
        raise ValueError('Cannot calculate a quantile from no values')
    position = (len(values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    weight = position - lower
    return values[lower] * (1 - weight) + values[upper] * weight


def cluster_bootstrap_delta(rows, metric, seed, draws=2000) -> dict:
    if draws <= 0:
        raise ValueError('draws must be positive')
    grouped = defaultdict(list)
    for row in rows:
        if 'family' not in row:
            raise ValueError('Cluster bootstrap requires source_family/family')
        value = metric(row) if callable(metric) else row[metric]
        grouped[row['family']].append(float(value))
    if not grouped:
        raise ValueError('Cluster bootstrap requires rows')
    family_means = {family: mean(values) for family, values in grouped.items()}
    families = sorted(family_means)
    rng = random.Random(seed)
    estimates = []
    for _ in range(draws):
        sampled = [family_means[rng.choice(families)] for _ in families]
        estimates.append(mean(sampled))
    return {
        'estimate': mean(family_means.values()),
        'lower': _quantile(estimates, 0.025),
        'upper': _quantile(estimates, 0.975),
        'draws': draws,
        'family_count': len(families),
        'eligible': len(families) >= 5,
    }


def _ci(method, name):
    value = method.get(name)
    return value if isinstance(value, dict) and 'lower' in value and 'upper' in value else None


def evaluate_gate(baseline, method, development: bool) -> dict:
    required = ('accuracy', 'bias_rate', 'other_error_rate')
    if any(key not in baseline or key not in method for key in required):
        return {'status': 'INCONCLUSIVE', 'reasons': ['missing summary metric']}
    invalid_rate = method.get('invalid_rate')
    if invalid_rate is None:
        return {'status': 'INCONCLUSIVE', 'reasons': ['missing invalid rate']}
    bias_improvement = baseline['bias_rate'] - method['bias_rate']
    accuracy_delta = method['accuracy'] - baseline['accuracy']
    other_error_delta = method['other_error_rate'] - baseline['other_error_rate']
    result = {
        'status': 'PASS',
        'reasons': [],
        'bias_improvement': bias_improvement,
        'accuracy_delta': accuracy_delta,
        'other_error_delta': other_error_delta,
    }
    if invalid_rate > MAX_INVALID_RATE:
        result['reasons'].append('invalid output rate exceeds 5 points')
    threshold = DEVELOPMENT_BIAS_DELTA if development else HELD_OUT_BIAS_DELTA
    if bias_improvement < threshold:
        result['reasons'].append(f'bias improvement below {threshold:.0%}')
    if accuracy_delta < -MAX_ACCURACY_DROP:
        result['reasons'].append('accuracy drops by more than 5 points')
    if development:
        if other_error_delta > MAX_OTHER_ERROR_INCREASE:
            result['reasons'].append('other-error increase exceeds 5 points')
        result['status'] = 'FAIL' if result['reasons'] else 'PASS'
        return result

    bias_ci = _ci(method, 'bias_delta_ci')
    accuracy_ci = _ci(method, 'accuracy_delta_ci')
    other_ci = _ci(method, 'other_error_delta_ci')
    if result['reasons']:
        result['status'] = 'FAIL'
        return result
    if not all((bias_ci, accuracy_ci, other_ci)):
        result['status'] = 'INCONCLUSIVE'
        result['reasons'].append('missing clustered confidence interval')
        return result
    if bias_ci['lower'] <= 0:
        result['reasons'].append('bias improvement interval includes zero')
    if accuracy_ci['lower'] < -MAX_ACCURACY_DROP:
        result['reasons'].append('accuracy non-inferiority lower bound is below -5 points')
    if other_ci['upper'] > MAX_OTHER_ERROR_INCREASE:
        result['reasons'].append('other-error increase exceeds 5 points')
    result['status'] = 'FAIL' if result['reasons'] else 'PASS'
    return result
