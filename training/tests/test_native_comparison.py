import copy
import unittest

from experiments.native_chord_diagnostic import validate_replay_only_change


class NativeComparisonTests(unittest.TestCase):
    def test_capacity_change_passes_without_mutating_inputs(self):
        old = {"replay": {"capacity": 2000000, "batch_size": 2048}, "lr": 0.0003}
        new = copy.deepcopy(old)
        new["replay"]["capacity"] = 5000000
        self.assertTrue(
            validate_replay_only_change(old, new)[
                "all_other_learner_parameters_identical"
            ]
        )
        self.assertEqual(old["replay"]["capacity"], 2000000)

    def test_batch_or_lr_change_is_rejected(self):
        old = {"replay": {"capacity": 2000000, "batch_size": 2048}, "lr": 0.0003}
        for key in ("batch", "lr"):
            new = copy.deepcopy(old)
            new["replay"]["capacity"] = 5000000
            if key == "batch":
                new["replay"]["batch_size"] = 5000000
            else:
                new["lr"] = 0.0001
            with self.assertRaises(ValueError):
                validate_replay_only_change(old, new)
