import ast
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from training.leaderboard import leaderboard_row, leaderboard_scores, write_report
from training.leaderboard_metrics import pose_metrics


def poses(frames=7, worlds=3, bodies=2):
    a = np.zeros((frames, worlds, bodies, 7))
    a[..., 6] = 1
    return a, a[:, 0].copy()


class LeaderboardTests(unittest.TestCase):
    def test_native_aliases_do_not_use_chord_success_or_worst_body_mppe(self):
        scores = leaderboard_scores(
            {
                "metrics": {
                    "chord_sr": 1.0,
                    "spider_sr_uncentered": 0.25,
                    "maniptrans_sr": 0.125,
                    "add_auc": 0.51,
                    "mppe_cm": 20,
                    "mean_per_point_error_cm": 10,
                }
            }
        )
        self.assertEqual(
            scores,
            {"add_auc": 0.51, "spider_sr": 0.25, "maniptrans_sr": 0.125, "mppe_cm": 10},
        )
        self.assertIsNone(leaderboard_scores({"mppe_cm": 20})["mppe_cm"])

    def test_missing_invalid_and_zero_are_distinct(self):
        self.assertTrue(
            all(
                value is None
                for value in leaderboard_scores(
                    {"non_finite_worlds": [0], "metrics": {"spider_sr_uncentered": 1.0}}
                ).values()
            )
        )
        scores = leaderboard_scores(
            {"add_auc": float("nan"), "spider_sr": 0, "maniptrans_sr": 2, "mppe_cm": -1}
        )
        self.assertEqual(
            scores,
            {"add_auc": None, "spider_sr": 0, "maniptrans_sr": None, "mppe_cm": None},
        )
        row = leaderboard_row(
            {"leaderboard": {"scores": scores}}, tier=2, episode=16, variant="corrected"
        )
        with tempfile.TemporaryDirectory() as tmp:
            report = write_report(tmp, [row])
            text = (Path(tmp) / "leaderboard.md").read_text(encoding="utf-8")
            self.assertIn("— | 0.0000 | — | —", text)
            self.assertIn("T2: 1/30 episodes", text)
            self.assertFalse(report["official_leaderboard_score"])
            self.assertEqual(
                json.loads((Path(tmp) / "leaderboard.json").read_text())["rows"][0][
                    "spider_sr"
                ],
                0,
            )

    def test_mppe_is_mean_over_bodies_and_orientation_matters(self):
        a, r = poses(bodies=2)
        a[..., 0, 0] = 0.02
        a[..., 1, 0] = 0.08
        metrics = pose_metrics(a, r, np.zeros((2, 1, 3)))
        self.assertAlmostEqual(metrics["scores"]["mppe_cm"], 5)
        self.assertEqual(metrics["scores"]["spider_sr"], 1)
        self.assertEqual(metrics["scores"]["maniptrans_sr"], 0)
        a, r = poses(bodies=1)
        a[..., 5:7] = [1, 0]  # pi about Z; object centre stays identical.
        self.assertGreater(pose_metrics(a, r, None)["scores"]["mppe_cm"], 0)

    def test_failed_world_is_not_dropped(self):
        a, r = poses(bodies=1)
        a[2, 1, 0, 0] = np.nan
        result = pose_metrics(a, r, None)
        self.assertEqual(result["invalid_worlds"], [1])
        self.assertTrue(all(v is None for v in result["scores"].values()))

    def test_success_uses_temporal_mean_without_position_alignment(self):
        a, r = poses(frames=10, worlds=1, bodies=1)
        a[0, :, :, 0] = 0.2  # mean=2 cm: not an every-frame 3 cm test.
        scores = pose_metrics(a, r, None)["scores"]
        self.assertEqual(scores["maniptrans_sr"], 1)
        a[..., 0] = 0.2  # Must not be centred away.
        self.assertEqual(pose_metrics(a, r, None)["scores"]["spider_sr"], 0)

    def test_formulas_match_released_kaggle_implementation(self):
        path = (
            Path(__file__).resolve().parents[2]
            / "vendor/submission-kit/v2d_submission_kit/metric_code/track_2/SP-SR.py"
        )
        if not path.exists():
            self.skipTest("Released scoring kit is not installed")
        # Execute the released numerical functions unchanged, without importing
        # the unrelated Pandas submission/provenance validation dependencies.
        tree = ast.parse(path.read_text(encoding="utf-8"))
        functions = {
            "_rotation_matrix",
            "_quaternion_geodesic_angle",
            "_object_add",
            "_add_auc",
            "_per_world_pose_error",
            "_spider_success",
            "_maniptrans_success",
            "_mean_per_point_error_cm",
        }
        constants = {
            "_ADD_THRESHOLDS_M",
            "_SPIDER_POSITION_THRESHOLD_M",
            "_SPIDER_ORIENTATION_THRESHOLD_RAD",
            "_MANIPTRANS_POSITION_THRESHOLD_M",
            "_MANIPTRANS_ORIENTATION_THRESHOLD_DEG",
            "_MPPE_KEYPOINT_RADIUS_M",
            "_MPPE_KEYPOINT_DIRECTIONS",
        }
        nodes = [
            node
            for node in tree.body
            if (isinstance(node, ast.FunctionDef) and node.name in functions)
            or (
                isinstance(node, ast.Assign)
                and any(
                    isinstance(t, ast.Name) and t.id in constants for t in node.targets
                )
            )
        ]
        scope = {"np": np}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), scope)  # noqa: S102 -- pinned local numerical oracle
        oracle = SimpleNamespace(**scope)
        rng = np.random.default_rng(42)
        a, r = poses(frames=71, worlds=10, bodies=2)  # Cross both chunk boundaries.
        a[..., :3] = rng.normal(0, 0.035, a[..., :3].shape)
        a[..., 3:] += rng.normal(0, 0.08, a[..., 3:].shape)
        a[..., 3:] /= np.linalg.norm(a[..., 3:], axis=-1, keepdims=True)
        vertices = rng.normal(0, 0.1, (2, 15, 3))
        ids = np.array([0, 1])
        scores = pose_metrics(a, r, vertices, ids)["scores"]
        ref = r[:, None]
        expected = {
            "add_auc": oracle._add_auc(oracle._object_add(a, ref, vertices)),
            "spider_sr": float(oracle._spider_success(a, ref).mean()),
            "maniptrans_sr": float(oracle._maniptrans_success(a, ref, ids).mean()),
            "mppe_cm": oracle._mean_per_point_error_cm(a, ref),
        }
        for key, value in expected.items():
            self.assertAlmostEqual(scores[key], value, places=11, msg=key)


if __name__ == "__main__":
    unittest.main()
