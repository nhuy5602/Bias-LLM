import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))

from open_vlm.scoring import (  # noqa: E402
    classify_answer,
    cluster_bootstrap_delta,
    evaluate_gate,
    score_candidates,
    summarize,
)


class OpenVlmScoringTests(unittest.TestCase):
    def test_candidate_scoring_sums_all_token_pieces(self):
        calls = []

        def sequence_score(candidate):
            calls.append(candidate)
            return {'13': [-0.1, -0.2], '4': [-0.5]}[candidate]

        result = score_candidates(['13', '4'], sequence_score)
        self.assertEqual(result['prediction'], '13')
        self.assertAlmostEqual(result['scores']['13'], -0.3)
        self.assertEqual(calls, ['13', '4'])

    def test_candidate_scoring_supports_single_float_scores(self):
        result = score_candidates(['yes', 'no'], lambda candidate: 1.0 if candidate == 'yes' else 0.5)
        self.assertEqual(result['prediction'], 'yes')

    def test_summary_reports_correct_bias_and_other_errors(self):
        rows = [
            {'family': 'a', 'prediction': 4, 'ground_truth': 4, 'prior_answer': 5},
            {'family': 'a', 'prediction': 5, 'ground_truth': 4, 'prior_answer': 5},
            {'family': 'b', 'prediction': 7, 'ground_truth': 4, 'prior_answer': 5},
        ]
        result = summarize(rows)
        self.assertEqual(result['counts'], {'Correct': 1, 'Bias-aligned': 1, 'Other errors': 1})
        self.assertAlmostEqual(result['accuracy'], 1 / 3)
        self.assertAlmostEqual(result['bias_rate'], 1 / 3)
        self.assertAlmostEqual(result['other_error_rate'], 1 / 3)

    def test_classification_prioritizes_correct_when_answers_match(self):
        row = {'ground_truth': 'yes', 'prior_answer': 'yes'}
        self.assertEqual(classify_answer('yes', row), 'Correct')
        different_prior = {'ground_truth': 'yes', 'prior_answer': 'no'}
        self.assertEqual(classify_answer('no', different_prior), 'Bias-aligned')
        self.assertEqual(classify_answer('maybe', different_prior), 'Other errors')

    def test_cluster_bootstrap_resamples_families(self):
        rows = [
            {'family': 'a', 'delta': 0.0},
            {'family': 'a', 'delta': 0.0},
            {'family': 'b', 'delta': 1.0},
            {'family': 'b', 'delta': 1.0},
        ]
        result = cluster_bootstrap_delta(rows, 'delta', seed=3, draws=2000)
        self.assertEqual(result['draws'], 2000)
        self.assertEqual(result['family_count'], 2)
        self.assertAlmostEqual(result['estimate'], 0.5)
        self.assertLessEqual(result['lower'], result['estimate'])
        self.assertGreaterEqual(result['upper'], result['estimate'])

    def test_development_gate_uses_bias_accuracy_and_invalid_rate(self):
        baseline = {'accuracy': 0.50, 'bias_rate': 0.50, 'other_error_rate': 0.0}
        method = {
            'accuracy': 0.46,
            'bias_rate': 0.46,
            'other_error_rate': 0.10,
            'invalid_rate': 0.01,
        }
        result = evaluate_gate(baseline, method, development=True)
        self.assertEqual(result['status'], 'FAIL')
        self.assertIn('bias', ' '.join(result['reasons']))

    def test_held_out_gate_accepts_accuracy_lower_bound_of_minus_five_points(self):
        baseline = {'accuracy': 0.50, 'bias_rate': 0.50, 'other_error_rate': 0.20}
        method = {
            'accuracy': 0.46,
            'bias_rate': 0.35,
            'other_error_rate': 0.19,
            'invalid_rate': 0.01,
            'bias_delta_ci': {'lower': 0.06, 'upper': 0.25},
            'accuracy_delta_ci': {'lower': -0.05, 'upper': 0.04},
            'other_error_delta_ci': {'lower': -0.05, 'upper': 0.04},
        }
        result = evaluate_gate(baseline, method, development=False)
        self.assertEqual(result['status'], 'PASS')

    def test_held_out_gate_is_inconclusive_without_cluster_interval(self):
        baseline = {'accuracy': 0.50, 'bias_rate': 0.50, 'other_error_rate': 0.20}
        method = {
            'accuracy': 0.50,
            'bias_rate': 0.35,
            'other_error_rate': 0.20,
            'invalid_rate': 0.01,
        }
        result = evaluate_gate(baseline, method, development=False)
        self.assertEqual(result['status'], 'INCONCLUSIVE')


if __name__ == '__main__':
    unittest.main()
