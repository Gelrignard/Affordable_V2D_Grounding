import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from training.__main__ import batch_run, make_plan, parser
from training.config import TrainConfig


class PlanTests(unittest.TestCase):
    def test_default_scope_and_two_device_shards(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for tier in (1, 2):
                for episode in (3, 5, 11, 16, 18, 25):
                    for variant in ("original", "corrected"):
                        folder = root / "jobs" / f"tier{tier}" / str(episode) / variant
                        folder.mkdir(parents=True)
                        (folder / "job.json").write_text(
                            json.dumps(
                                {
                                    "tier": tier,
                                    "episode_index": episode,
                                    "variant": variant,
                                }
                            )
                        )
            config = root / "config.json"
            config.write_text(json.dumps(TrainConfig().to_dict()))
            path = root / "plan.json"
            args = parser().parse_args(
                [
                    "plan",
                    "--jobs",
                    str(root / "jobs"),
                    "--config",
                    str(config),
                    "--output",
                    str(path),
                ]
            )

            def load(folder):
                return json.loads((Path(folder) / "job.json").read_text())

            with patch("training.prepare.load_job", side_effect=load):
                result = make_plan(args)
                self.assertEqual(result["runs"], 8)
                plan = json.loads(path.read_text())
                self.assertEqual(
                    {e["algorithm"] for e in plan["entries"]}, {"model_based"}
                )
                identities = {
                    tuple(load(root / e["job"]).values()) for e in plan["entries"]
                }
                self.assertEqual(
                    identities,
                    {
                        (t, e, v)
                        for t in (1, 2)
                        for e in (5, 16)
                        for v in ("original", "corrected")
                    },
                )
                completed = []
                for shard in (0, 1):
                    batch = parser().parse_args(
                        [
                            "batch",
                            "--plan",
                            str(path),
                            "--runtime",
                            temp,
                            "--max-jobs",
                            "10",
                            "--shard-count",
                            "2",
                            "--shard-index",
                            str(shard),
                            "--device",
                            f"cuda:{shard}",
                            "--evaluate",
                        ]
                    )
                    with patch("training.__main__.subprocess.run") as run:
                        done = batch_run(batch)["completed"]
                        self.assertEqual(len(done), 4)
                        self.assertEqual(run.call_count, 8)
                        for call in run.call_args_list:
                            command = call.args[0]
                            if "evaluate" in command:
                                self.assertEqual(
                                    command[command.index("--device") + 1],
                                    f"cuda:{shard}",
                                )
                        for out in done:
                            saved = json.loads(
                                (root / out / "planned_config.json").read_text()
                            )
                            self.assertEqual(saved["device"], f"cuda:{shard}")
                        completed.extend(done)
                self.assertEqual(len(set(completed)), 8)
                # A retired larger plan cannot silently spend the old budget.
                plan["obsolete"] = True
                path.write_text(json.dumps(plan))
                with self.assertRaisesRegex(ValueError, "obsolete"):
                    batch_run(batch)
                # Missing one selected input must not produce a partial comparison.
                (root / "jobs/tier1/5/original/job.json").unlink()
                args.output = str(root / "incomplete.json")
                with self.assertRaisesRegex(ValueError, "missing selected"):
                    make_plan(args)
                self.assertFalse(Path(args.output).exists())


if __name__ == "__main__":
    unittest.main()
