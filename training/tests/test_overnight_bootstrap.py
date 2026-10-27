"""Selection rejects unusable sources and ranks measured quality over recency."""

import copy
import json
from pathlib import Path
import struct
import tempfile
import unittest

from experiments.overnight_bootstrap import eligible_sources, verify_parent_repeat, verify_unlearned_startup, file_hash


def source_run(root, name, step, auc, *, mppe=5.0, assistance=0.0, missing_state=False):
    run = root / name
    checkpoint_dir = run / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    policy = checkpoint_dir / f"policy_{step}.safetensors"
    state = checkpoint_dir / f"state_{step}.safetensors"
    fields = {"algorithm": "flash_sac", "checkpoint_set_id": f"pair_{name}_{step}",
              "environment_steps": str(step), "policy_schema_json": "{}", "critic_schema_json": "{}"}
    for path, kind in ((policy, "policy"), (state, "training_state")):
        if missing_state and kind == "training_state":
            continue
        metadata = {"format": "learner_state", "version": "1", "manifest": "[]",
                    **{"user." + key: value for key, value in {**fields, "checkpoint_kind": kind}.items()}}
        header = json.dumps({"__metadata__": metadata}).encode()
        path.write_bytes(struct.pack("<Q", len(header)) + header)
    report = run / "evaluations" / f"{step}_unassisted" / "evaluation.json"
    report.parent.mkdir(parents=True)
    report.write_text(json.dumps({
        "checkpoint": str(policy), "environment_steps": step,
        "non_finite_worlds": [], "world_count": 16,
        "start_frame": 0, "reset_mode": "explicit",
        "probe_assistance": assistance, "settling_steps": 0,
        "leaderboard": {"scores": {"add_auc": auc, "mppe_cm": mppe,
                                     "spider_sr": .95, "maniptrans_sr": .80}},
    }))
    return run, policy


class OvernightBootstrapTests(unittest.TestCase):
    def test_rejects_missing_resumable_state_even_with_high_score(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            broken, _ = source_run(root, "missing_state", 200_001_536, .99, missing_state=True)
            valid, policy = source_run(root, "valid", 100_001_792, .80)
            candidates, rejected = eligible_sources([broken, valid])
            self.assertEqual([item["policy"] for item in candidates], [str(policy)])
            self.assertEqual(len(rejected), 1)
            self.assertIn("state_200001536.safetensors", rejected[0]["reason"])
            with self.assertRaisesRegex(ValueError, "No evaluated compact state"):
                eligible_sources([broken])

    def test_rejects_assisted_evaluation(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            assisted, _ = source_run(root, "assisted", 200_001_536, .99, assistance=.05)
            valid, policy = source_run(root, "unassisted", 100_001_792, .80)
            candidates, rejected = eligible_sources([assisted, valid])
            self.assertEqual(candidates[0]["policy"], str(policy))
            self.assertEqual(len(candidates), 1)
            self.assertIn("unassisted", rejected[0]["reason"])

    def test_selects_best_auc_instead_of_newest_step(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            older, old_policy = source_run(root, "older", 100_001_792, .93)
            newer, _ = source_run(root, "newer", 349_999_104, .81)
            candidates, rejected = eligible_sources([newer, older])
            self.assertEqual(candidates[0]["policy"], str(old_policy))
            self.assertEqual(candidates[0]["environment_steps"], 100_001_792)
            self.assertEqual(rejected, [])

    def test_auc_ties_prefer_lower_mppe(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            older, old_policy = source_run(root, "older", 100_001_792, .93, mppe=3)
            newer, _ = source_run(root, "newer", 349_999_104, .93, mppe=7)
            candidates, _ = eligible_sources([newer, older])
            self.assertEqual(candidates[0]["policy"], str(old_policy))


class ParentRepeatTests(unittest.TestCase):
    def report(self):
        return {
            "checkpoint_sha256": "a" * 64, "world_count": 16, "step_count": 200,
            "start_frame": 0, "reset_mode": "explicit", "settling_steps": 0,
            "sim_control_fps": 50, "sim_physics_fps": 200,
            "parquet": "/run/input/motion.parquet", "non_finite_worlds": [],
            "leaderboard": {"scores": {"add_auc": .35575, "mppe_cm": 6.0,
                                         "spider_sr": .9375, "maniptrans_sr": .8125}},
        }

    def test_allows_small_repeat_drift_with_identical_contract(self):
        expected = self.report()
        actual = copy.deepcopy(expected)
        actual["leaderboard"]["scores"].update(add_auc=.35295, mppe_cm=6.088)
        before = copy.deepcopy((expected, actual))
        drift = verify_parent_repeat(expected, actual)
        self.assertIsInstance(drift, dict)
        self.assertEqual((expected, actual), before)

    def test_rejects_changed_checkpoint_reference_and_rollout_contract(self):
        changes = {
            "checkpoint_sha256": "b" * 64,
            "parquet": "/different/input/motion.parquet",
            "world_count": 8,
            "step_count": 199,
            "start_frame": 1,
            "reset_mode": "random",
            "settling_steps": 1,
            "sim_control_fps": 30,
            "sim_physics_fps": 100,
        }
        for key, value in changes.items():
            with self.subTest(key=key):
                expected = self.report()
                actual = copy.deepcopy(expected)
                actual[key] = value
                with self.assertRaises(RuntimeError):
                    verify_parent_repeat(expected, actual)

    def test_rejects_non_finite_rollout_worlds(self):
        expected = self.report()
        actual = copy.deepcopy(expected)
        actual["non_finite_worlds"] = [3]
        with self.assertRaises(RuntimeError):
            verify_parent_repeat(expected, actual)

    def test_rejects_large_repeat_drift_in_either_direction(self):
        for metric, delta in (("add_auc", .0101), ("mppe_cm", .5001)):
            for sign in (-1, 1):
                with self.subTest(metric=metric, sign=sign):
                    expected = self.report()
                    actual = copy.deepcopy(expected)
                    actual["leaderboard"]["scores"][metric] += sign * delta
                    with self.assertRaises(RuntimeError):
                        verify_parent_repeat(expected, actual)

    def test_requires_exact_success_rates(self):
        for metric in ("spider_sr", "maniptrans_sr"):
            with self.subTest(metric=metric):
                expected = self.report()
                actual = copy.deepcopy(expected)
                actual["leaderboard"]["scores"][metric] -= 1 / 16
                with self.assertRaises(RuntimeError):
                    verify_parent_repeat(expected, actual)

    def test_rejects_non_finite_scores(self):
        for metric in ("add_auc", "mppe_cm", "spider_sr", "maniptrans_sr"):
            for value in (float("nan"), float("inf")):
                with self.subTest(metric=metric, value=value):
                    expected = self.report()
                    actual = copy.deepcopy(expected)
                    actual["leaderboard"]["scores"][metric] = value
                    with self.assertRaises(RuntimeError):
                        verify_parent_repeat(expected, actual)


def write_tensors(path, values, *, updates=100, env_steps=2048):
    raw = path.read_bytes()
    metadata = json.loads(raw[8:8 + struct.unpack('<Q', raw[:8])[0]])['__metadata__']
    metadata.update({'user.global_update_step': str(updates), 'user.environment_steps': str(env_steps)})
    manifest, tensors, payload = [], {}, bytearray()
    formats = {'F32': 'f', 'I32': 'i', 'U32': 'I'}
    for i, (dtype, shape, data) in enumerate(values):
        key = f'state.{i:04d}'
        encoded = struct.pack('<' + formats[dtype] * len(data), *data)
        tensors[key] = {'dtype': dtype, 'shape': shape or [1], 'data_offsets': [len(payload), len(payload) + len(encoded)]}
        payload.extend(encoded)
        manifest.append({'key': key, 'shape': shape, 'dtype': {'F32': '<f4', 'I32': '<i4', 'U32': '<u4'}[dtype]})
    metadata['manifest'] = json.dumps(manifest)
    header = json.dumps({'__metadata__': metadata, **tensors}).encode()
    path.write_bytes(struct.pack('<Q', len(header)) + header + payload)


class UnlearnedStartupTests(unittest.TestCase):
    def fixture(self, root):
        from training import RUNTIME_COMMIT
        _, parent_policy = source_run(root, 'parent', 2048, .8)
        _, failed_policy = source_run(root, 'failed', 4096, .8)
        parent_state = parent_policy.with_name('state_2048.safetensors')
        failed_state = failed_policy.with_name('state_4096.safetensors')
        protected = [('F32', [2], [i + .25, i + .5]) for i in range(7)]
        tail = [('F32', [2], [0., 0.]), *[('F32', [], [1.])] * 4,
                ('F32', [2, 2], [0., 0., 0., 0.]), ('I32', [], [0]), ('I32', [], [0]),
                ('F32', [], [1.]), ('I32', [], [0]), ('U32', [2], [1, 2]),
                ('F32', [], [0.]), ('I32', [], [2048]), ('I32', [], [100])]
        values = protected + tail
        write_tensors(parent_state, values)
        policy = [('F32', [2], [1., 2.])]
        write_tensors(parent_policy, policy)
        failed_values = copy.deepcopy(values)
        failed_values[-14] = ('F32', [2], [4., 5.])
        failed_values[-4] = ('U32', [2], [7, 8])
        failed_values[-2] = ('I32', [], [4096])
        write_tensors(failed_state, failed_values, env_steps=4096)
        write_tensors(failed_policy, policy, env_steps=4096)
        output = root / 'attempt'
        output.mkdir()
        documents = {
            'latest_checkpoint.json': {'state': str(failed_state), 'policy': str(failed_policy), 'environment_steps': 4096},
            'provenance.json': {'runtime_commit': RUNTIME_COMMIT, 'replay_is_fresh': True,
                                'source_state': str(parent_state), 'source_state_sha256': file_hash(parent_state),
                                'source_policy_sha256': file_hash(parent_policy)},
            'warmstart_parity.json': {'parameter_exact_match': True, 'deterministic_action_max_abs_error': 0.,
                                     'fresh_replay_size': 0, 'environment_steps': 2048, 'global_update_step': 100},
            'resolved_config.json': {'training': {'replay': {'minimum_size': 10000}}},
        }
        for name, data in documents.items():
            (output / name).write_text(json.dumps(data))
        return parent_state, output, failed_state, failed_policy, failed_values

    def test_allows_only_unchanged_learning_state_with_warmup_changes(self):
        with tempfile.TemporaryDirectory() as folder:
            parent, output, failed, _, _ = self.fixture(Path(folder))
            before_hash = file_hash(failed)
            report = verify_unlearned_startup(parent, output)
            self.assertTrue(report['verified_zero_learning'])
            self.assertEqual(report['additional_transitions_discarded'], 2048)
            self.assertEqual(report['global_update_step'], 100)
            self.assertEqual(len(report['changed_discarded_warmup_leaves']), 3)
            self.assertEqual(file_hash(failed), before_hash)

    def test_refuses_any_changed_model_optimizer_target_or_clock_leaf(self):
        for index in (*range(7), -6, -5, -1):
            with self.subTest(index=index), tempfile.TemporaryDirectory() as folder:
                parent, output, failed, _, values = self.fixture(Path(folder))
                dtype, shape, data = values[index]
                values[index] = (dtype, shape, [data[0] + 1, *data[1:]])
                write_tensors(failed, values, env_steps=4096)
                with self.assertRaisesRegex(RuntimeError, 'protected'):
                    verify_unlearned_startup(parent, output)

    def test_refuses_changed_policy_even_if_state_is_unchanged(self):
        with tempfile.TemporaryDirectory() as folder:
            parent, output, _, policy, _ = self.fixture(Path(folder))
            write_tensors(policy, [('F32', [2], [2., 3.])], env_steps=4096)
            with self.assertRaisesRegex(RuntimeError, 'deployable actor'):
                verify_unlearned_startup(parent, output)

    def test_refuses_changed_update_metadata(self):
        with tempfile.TemporaryDirectory() as folder:
            parent, output, failed, _, values = self.fixture(Path(folder))
            write_tensors(failed, values, updates=101, env_steps=4096)
            with self.assertRaisesRegex(RuntimeError, 'global_update_step'):
                verify_unlearned_startup(parent, output)

    def test_refuses_warmup_boundary_or_nonfresh_replay(self):
        for filename, modification in (
            ('resolved_config.json', {'training': {'replay': {'minimum_size': 2048}}}),
            ('warmstart_parity.json', {'parameter_exact_match': True, 'deterministic_action_max_abs_error': 0., 'fresh_replay_size': 1}),
        ):
            with self.subTest(filename=filename), tempfile.TemporaryDirectory() as folder:
                parent, output, _, _, _ = self.fixture(Path(folder))
                (output / filename).write_text(json.dumps(modification))
                with self.assertRaises(RuntimeError):
                    verify_unlearned_startup(parent, output)


if __name__ == "__main__":
    unittest.main()
