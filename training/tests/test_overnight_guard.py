from contextlib import nullcontext
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from experiments.overnight_guard import (
    ensure_budget_policy, ledger_budget_policy, make_budget_policy,
    remaining_budget, parse_deadline, main, read_json, write_json, run,
)


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


class PolicyTests(unittest.TestCase):
    def setUp(self):
        self.custom = make_budget_policy(23400, 10800, 2)

    def test_new_policy_persists_and_requires_exact_match_after_reload(self):
        ledger = {'jobs': []}
        ensure_budget_policy(ledger, self.custom, ledger_exists=False)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'ledger.json'
            write_json(path, ledger)
            restored = read_json(path)
        self.assertEqual(ledger_budget_policy(restored), self.custom)
        ensure_budget_policy(restored, self.custom, ledger_exists=True)
        for changed in [make_budget_policy(23401, 10800, 2),
                        make_budget_policy(23400, 10801, 2),
                        make_budget_policy(23400, 10800, 3),
                        make_budget_policy(23400, 10000, 2),
                        make_budget_policy()]:
            with self.subTest(changed=changed), self.assertRaisesRegex(RuntimeError, 'immutable'):
                ensure_budget_policy(restored, changed, ledger_exists=True)
            self.assertEqual(restored['budget_policy'], self.custom)

    def test_existing_legacy_ledger_cannot_be_expanded_even_if_empty(self):
        for jobs in [[], [{'candidate': 'old', 'kind': 'training',
                          'state': 'complete', 'elapsed_seconds': 60}]]:
            ledger = {'jobs': jobs}
            with self.subTest(jobs=jobs), self.assertRaisesRegex(RuntimeError, 'immutable'):
                ensure_budget_policy(ledger, self.custom, ledger_exists=True)
            self.assertNotIn('budget_policy', ledger)
            ensure_budget_policy(ledger, make_budget_policy(), ledger_exists=True)
            self.assertEqual(ledger['budget_policy'], make_budget_policy(21600, 7200, 3))

    def test_custom_training_cap_accumulates_bootstrap_and_refinement(self):
        ledger = {'budget_policy': self.custom, 'jobs': [
            {'candidate': 'episode_000016', 'kind': 'training', 'state': 'complete',
             'elapsed_seconds': 7200},
            {'candidate': 'episode_000016', 'kind': 'training', 'state': 'complete',
             'elapsed_seconds': 3300},
            {'candidate': 'episode_000016', 'kind': 'evaluation', 'state': 'complete',
             'elapsed_seconds': 600},
        ]}
        self.assertEqual(remaining_budget(ledger, 'episode_000016', 1800), 300)
        self.assertEqual(remaining_budget(ledger, 'episode_000018', 12000), 10800)
        self.assertEqual(remaining_budget(ledger, 'episode_000016', 1800, 'evaluation'), 1800)

    def test_custom_total_cap_charges_evaluation_and_smoke(self):
        ledger = {'budget_policy': self.custom, 'jobs': [
            {'candidate': 'episode_000016', 'kind': 'training', 'state': 'complete',
             'elapsed_seconds': 10800},
            {'candidate': 'episode_000018', 'kind': 'training', 'state': 'complete',
             'elapsed_seconds': 10800},
            {'candidate': 'episode_000016', 'kind': 'evaluation', 'state': 'complete',
             'elapsed_seconds': 900},
            {'candidate': 'episode_000018', 'kind': 'smoke', 'state': 'complete',
             'elapsed_seconds': 600},
        ]}
        self.assertEqual(remaining_budget(ledger, 'episode_000018', 900, 'evaluation'), 300)
        self.assertEqual(remaining_budget(ledger, 'episode_000018', 900), 0)

    def test_custom_candidate_roster_cap_counts_distinct_training_ids(self):
        ledger = {'budget_policy': self.custom, 'jobs': [
            {'candidate': name, 'kind': 'training', 'state': 'complete', 'elapsed_seconds': 60}
            for name in ['episode_000016', 'episode_000018', 'episode_000016']
        ]}
        with self.assertRaisesRegex(RuntimeError, 'limit'):
            remaining_budget(ledger, 'episode_000005', 1800)
        self.assertEqual(remaining_budget(ledger, 'episode_000016', 1800), 1800)

    def test_running_job_still_blocks_custom_policy(self):
        ledger = {'budget_policy': self.custom, 'jobs': [{'state': 'running'}]}
        with self.assertRaisesRegex(RuntimeError, 'running job'):
            remaining_budget(ledger, 'episode_000016', 1800)

    def test_policy_rejects_nonfinite_nonpositive_and_noninteger_caps(self):
        for value in [float('nan'), float('inf'), -1, 0, True, '23400']:
            with self.subTest(value=value), self.assertRaises(ValueError):
                make_budget_policy(value, 10800, 2)
            with self.subTest(candidate_value=value), self.assertRaises(ValueError):
                make_budget_policy(23400, value, 2)
        for value in [0, -1, 2.5, True, '2']:
            with self.subTest(maximum_candidates=value), self.assertRaises(ValueError):
                make_budget_policy(23400, 10800, value)

    def test_malformed_persisted_policy_fails_closed(self):
        invalid = [None, {}, {**self.custom, 'schema': 'unknown'},
                   {**self.custom, 'total_budget_seconds': float('inf')},
                   {**self.custom, 'extra_setting': True}]
        for value in invalid:
            with self.subTest(policy=value), self.assertRaises(RuntimeError):
                remaining_budget({'budget_policy': value, 'jobs': []}, 'episode_000016', 1800)

    def test_cli_default_and_custom_policy_arguments(self):
        required = ['guard', '--ledger', 'ledger.json', '--name', 'bootstrap',
                    '--candidate', 'episode_000016', '--kind', 'training',
                    '--max-seconds', '7200', '--wait-queue', 'queue',
                    '--output', 'output', '--cwd', '.']
        for flags, expected in [([], make_budget_policy()),
                                (['--total-budget-seconds', '23400',
                                  '--candidate-training-budget-seconds', '10800',
                                  '--maximum-candidates', '2'], self.custom)]:
            with patch('sys.argv', required + flags + ['--', 'python', 'train.py']), \
                    patch('experiments.overnight_guard.run', return_value=0) as run:
                with self.assertRaises(SystemExit) as stopped:
                    main()
                self.assertEqual(stopped.exception.code, 0)
            args = run.call_args.args[0]
            self.assertEqual(make_budget_policy(args.total_budget_seconds,
                                               args.candidate_training_budget_seconds,
                                               args.maximum_candidates), expected)
            self.assertEqual(args.command, ['--', 'python', 'train.py'])


    def test_new_scope_policy_is_persisted_before_blocked_queue_without_launch(self):
        with tempfile.TemporaryDirectory() as directory:
            ledger_path = Path(directory) / 'ledger.json'
            args = SimpleNamespace(
                command=['--', 'python', 'train.py'], deadline_utc='2099-01-01T00:00:00Z',
                ledger=str(ledger_path), candidate='episode_000016', max_seconds=9000,
                kind='training', total_budget_seconds=23400,
                candidate_training_budget_seconds=10800, maximum_candidates=2,
                wait_queue='queue', wait_seconds=0,
            )
            with patch('experiments.overnight_guard.exclusive_lock', return_value=nullcontext()), \
                    patch('experiments.overnight_guard.external_queue_ready', return_value=(False, 'queue busy')), \
                    patch('experiments.overnight_guard.subprocess.Popen') as launch:
                with self.assertRaisesRegex(RuntimeError, 'queue busy'):
                    run(args)
                launch.assert_not_called()
            ledger = read_json(ledger_path)
            self.assertEqual(ledger['budget_policy'], self.custom)
            self.assertEqual(ledger['jobs'], [])


if __name__ == '__main__':
    unittest.main()
