import unittest
from experiments.overnight_guard import remaining_budget, parse_deadline


class BudgetTests(unittest.TestCase):
    def test_total_includes_evaluation(self):
        ledger = {'jobs': [{'candidate':'a', 'kind':'evaluation', 'state':'complete', 'elapsed_seconds':21000}]}
        self.assertEqual(remaining_budget(ledger, 'b', 1800), 600)

    def test_candidate_training_cap_survives_resume(self):
        ledger = {'jobs': [{'candidate':'a', 'kind':'training', 'state':'complete', 'elapsed_seconds':7000}]}
        self.assertEqual(remaining_budget(ledger, 'a', 1800), 200)

    def test_three_candidates(self):
        ledger = {'jobs':[{'candidate':x,'kind':'training','state':'complete','elapsed_seconds':60} for x in 'abc']}
        with self.assertRaises(RuntimeError):
            remaining_budget(ledger, 'd', 1800)
        self.assertEqual(remaining_budget(ledger, 'a', 1800), 1800)

    def test_unreconciled_job_blocks_launch(self):
        with self.assertRaises(RuntimeError):
            remaining_budget({'jobs':[{'state':'running'}]}, 'a', 1800)

    def test_explicit_deadline(self):
        self.assertEqual(parse_deadline('2026-10-27T13:30:00Z'), parse_deadline('2026-10-27T09:30:00-04:00'))
        with self.assertRaises(ValueError):
            parse_deadline('2026-10-27T09:30:00')


if __name__ == '__main__':
    unittest.main()
