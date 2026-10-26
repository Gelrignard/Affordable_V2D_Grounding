"""Read-only CPU audit of rewards, update health and saved unassisted rollouts."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def audit(run, baseline):
    import numpy as np
    import pyarrow.parquet as pq

    from training.leaderboard_metrics import pose_metrics

    run, baseline = Path(run), Path(baseline)
    metrics = [
        json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()
    ]
    keys = [
        k
        for k in metrics[-1]
        if k.startswith(
            ("reward/", "reward_components/", "ppo/", "termination/", "episode/")
        )
    ]
    summary = {
        "source_run": str(run),
        "metric_records": len(metrics),
        "scope": "60M-to-350M continuation; first/last 40 logged PPO iterations",
        "windows": {
            name: {k: statistics.mean(m[k] for m in window) for k in keys}
            for name, window in (("first40", metrics[:40]), ("last40", metrics[-40:]))
        },
        "assistance_scales": sorted({m["curriculum/voc_scale"] for m in metrics}),
        "skipped_update_max": max(
            m["ppo/skipped_update_fraction_per_iteration"] for m in metrics
        ),
        "learning_rate_range": [
            min(m["ppo/learning_rate"] for m in metrics),
            max(m["ppo/learning_rate"] for m in metrics),
        ],
        "reward_sum_max_absolute_error": max(
            abs(
                m["reward/total_mean_per_env_step"]
                - sum(v for k, v in m.items() if k.startswith("reward_components/"))
            )
            for m in metrics
        ),
        "evaluations": [],
    }
    config = json.loads((run / "training_config.json").read_text())
    weights = config["curriculum"]["stages"][0]["objective_weights"]
    summary["reward_weights"] = weights
    summary["direct_object_orientation_reward"] = "object_orientation" in weights
    final = json.loads((run / "checkpoints/model_7121.experiment.json").read_text())
    summary["curriculum_decisions"] = final["curriculum_state"]["decisions"]
    summary["curriculum_gate"] = final["curriculum_state"]["gate"]
    summary["same_input_as_flash_sac"] = (
        json.loads((run / "provenance.json").read_text())["job_motion_sha256"]
        == json.loads((baseline / "provenance.json").read_text())["job_motion_sha256"]
    )
    for path in sorted((run / "evaluations").glob("*/evaluation.json")):
        saved = json.loads(path.read_text())
        table = pq.read_table(path.parent / "object_rollouts.parquet")
        table = table.sort_by(
            [
                ("step", "ascending"),
                ("environment_id", "ascending"),
                ("object_body_id", "ascending"),
            ]
        )
        meta = {k.decode(): v.decode() for k, v in table.schema.metadata.items()}
        shape = tuple(
            int(meta[f"flash_chord.{key}"])
            for key in ("step_count", "world_count", "object_body_count")
        )
        suffixes = [f"position_{axis}" for axis in "xyz"] + [
            f"quaternion_{axis}" for axis in "xyzw"
        ]

        def poses(prefix, table=table, suffixes=suffixes, shape=shape):
            return np.stack(
                [table[f"{prefix}_{s}"].to_numpy() for s in suffixes], axis=-1
            ).reshape(*shape, 7)

        achieved, reference = poses("achieved"), poses("reference")
        assert table["environment_valid"].to_numpy().all()
        assert np.allclose(reference, reference[:, :1])
        vertices = np.load(path.parent / "auc_vertices.npz")["vertices"]
        recomputed = pose_metrics(achieved, reference[:, 0], vertices)
        for key, value in recomputed["scores"].items():
            expected = saved["metric_verification"]["scores"][key]
            if not np.isclose(value, expected, atol=1e-5, rtol=1e-5):
                raise ValueError(f"Saved-pose metric mismatch: {path} {key}")
        summary["evaluations"].append(
            {
                "transitions": saved["environment_steps"],
                "scores": recomputed["scores"],
                "recomputed_from_saved_poses": True,
                "object_position_error_m": saved["metrics"]["object_position_error_m"],
                "object_orientation_error_deg": saved["metrics"][
                    "object_orientation_error_deg"
                ],
                "mean_add_m": saved["metrics"]["mean_add_m"],
            }
        )
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "baseline", "output"):
        p.add_argument("--" + name, required=True)
    args = p.parse_args()
    result = audit(args.run, args.baseline)
    Path(args.output).write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "audit_passed": True,
                "metrics_recomputed": len(result["evaluations"]),
                "reward_sum_error": result["reward_sum_max_absolute_error"],
                "same_input": result["same_input_as_flash_sac"],
                "output": args.output,
            }
        )
    )
