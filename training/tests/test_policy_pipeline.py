"""CPU-only checks of fixed policy selection, supervision, and verified resume."""
import argparse
import contextlib
import copy
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

from training import policy_pipeline as p


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def checkpoint(root, steps, cfg=None):
    root.mkdir(parents=True, exist_ok=True)
    for kind, prefix in (("training_state", "state"), ("policy", "policy")):
        fields = {"algorithm": "flash_sac", "checkpoint_kind": kind,
                  "checkpoint_set_id": f"pair_{steps}", "environment_steps": str(steps),
                  "policy_schema_json": "{}", "critic_schema_json": "{}",
                  "resolved_config_json": json.dumps(cfg or {})}
        metadata = {"format": "learner_state", "version": "1",
                    **{"user." + k: v for k, v in fields.items()}}
        header = json.dumps({"__metadata__": metadata}).encode()
        (root / f"{prefix}_{steps}.safetensors").write_bytes(struct.pack("<Q", len(header)) + header)
    return p.checkpoint_record(root / f"state_{steps}.safetensors")


def job(root, tier=2):
    names = ("motion.parquet", "reference.npz", "guidance.npz", "geometry.npz",
             "assets/object.obj", "assets/object.urdf", "assets/support.usda")
    for name in names:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(name.encode())
    data = {"schema": "v2d_policy_job_v1", "tier": tier, "episode_index": 3,
            "runtime_commit": p.RUNTIME_COMMIT, "model_scope": "single_rigid_object_g1_dex3",
            "source_frames": 781, "files": {name: p.sha(root / name) for name in names}}
    write(root / "job.json", data)
    return p.job_identity(root)


class PipelineSelectionTests(unittest.TestCase):
    def records(self, *steps):
        return [{"environment_steps": step, "checkpoint_sha256": str(step)} for step in steps]

    def test_fixed_subset_ignores_outcomes(self):
        records = self.records(100, 20_000_100, 40_000_100, 60_000_100, 80_000_100, 90_000_100)
        self.assertEqual([r["environment_steps"] for r in p.choose_candidates(records, 100, 90_000_100)],
                         [90_000_100, 60_000_100, 40_000_100])

    def test_short_wall_stop_deduplicates_and_requires_progress(self):
        self.assertEqual(len(p.choose_candidates(self.records(100, 10_000_100, 25_000_100), 100, 25_000_100)), 1)
        with self.assertRaises(ValueError):
            p.choose_candidates(self.records(100), 100, 100)
        with self.assertRaises(ValueError):
            p.choose_candidates(self.records(100, 200, 200), 100, 200)

    def test_native_absolute_boundary_allows_exactly_one_collection(self):
        start = 349_999_104
        records = self.records(start, 380_000_896, 390_000_640, 410_000_384, 439_998_464)
        selected = p.choose_candidates(records, start, 439_998_464)
        self.assertEqual([r["environment_steps"] for r in selected], [439_998_464, 410_000_384, 390_000_640])
        records = self.records(start, start + 40_000_000 + 2049, start + 60_000_000 + 2049, start + 90_000_000)
        self.assertEqual([r["environment_steps"] for r in p.choose_candidates(records, start, start + 90_000_000)],
                         [start + 90_000_000, start + 40_000_000 + 2049])

    def test_ranking_prioritizes_mp_then_sp_then_auc_not_recency(self):
        def candidate(mp, sp, auc, mppe=2, steps=100):
            return {"metrics": dict(maniptrans_sr=mp, spider_sr=sp, add_auc=auc, mppe_cm=mppe),
                    "environment_steps": steps, "checkpoint_sha256": "a"}
        a, b = candidate(.875, .5, .5), candidate(.75, 1, .99)
        self.assertLess(p.rank_key(a), p.rank_key(b))
        self.assertLess(p.rank_key(candidate(.8, 1, .7)), p.rank_key(candidate(.8, .9, .99)))
        self.assertLess(p.rank_key(candidate(.8, 1, .8)), p.rank_key(candidate(.8, 1, .7)))
        self.assertLess(p.rank_key(candidate(.8, 1, .8, 1)), p.rank_key(candidate(.8, 1, .8, 2)))
        self.assertLess(p.rank_key(candidate(.8, 1, .8, steps=100)), p.rank_key(candidate(.8, 1, .8, steps=200)))
        for bad in (float("nan"), float("inf"), True, 1.01):
            with self.assertRaises(ValueError):
                p.rank_key(candidate(bad, 1, .8))

    def test_budget_reserves_three_evaluations_and_enforces_training_cap(self):
        request = {"deadline_utc": "2100-01-01T00:00:00Z", "dataset_input": "optional"}
        ledger = {"budget_policy": p.guard.make_budget_policy(11700, 10800, 1), "jobs": []}
        self.assertEqual(p.phase_caps(request, ledger, "episode", bootstrap=True, now=0), (9000, 9060))
        ledger["jobs"] = [{"candidate": "episode", "kind": "training", "state": "complete", "elapsed_seconds": 7500}]
        self.assertEqual(p.phase_caps(request, ledger, "episode", bootstrap=False, now=0), (3240, 3300))
        request["deadline_utc"] = "1970-01-01T00:16:00Z"
        with self.assertRaisesRegex(RuntimeError, "Insufficient"):
            p.phase_caps(request, ledger, "episode", bootstrap=False, now=0)

    def test_job_tier_three_and_changed_input(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertEqual(job(root, tier=3)["manifest"]["tier"], 3)
            (root / "motion.parquet").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "hash/path"):
                p.job_identity(root)

    def test_checkpoint_pair_and_hash_must_both_match(self):
        with tempfile.TemporaryDirectory() as directory:
            record = checkpoint(Path(directory), 4096)
            p.verify_checkpoint(record)
            Path(record["policy"]).write_bytes(Path(record["policy"]).read_bytes() + b"tamper")
            with self.assertRaisesRegex(ValueError, "changed"):
                p.verify_checkpoint(record)

    def test_source_identity_rejects_untracked_code(self):
        def fake_git(root, *args):
            if "--others" in args:
                return "training/shadow.py"
            return "training/policy_pipeline.py"
        with patch.object(p, "git", side_effect=fake_git):
            with self.assertRaisesRegex(ValueError, "Untracked"):
                p.source_identity()

    def test_export_preflight_fails_before_any_training(self):
        request = {"dataset_input": "released", "submission_kit": "kit"}
        identity = {"job": "portable", "manifest": {"episode_index": 3}}
        with patch("experiments.export_policy_rollout.validate_export_inputs", create=True,
                   side_effect=ValueError("Wrong released frame")) as validate:
            with self.assertRaisesRegex(ValueError, "Wrong released frame"):
                p.export_preflight(request, identity)
            validate.assert_called_once_with("portable", "released", 3, "kit")


class PipelineExecutionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.output = self.root / "output"
        self.identity = job(self.root / "job")
        self.args = argparse.Namespace(job=str(self.root / "job"), runtime=str(self.root / "runtime"),
            output=str(self.output), parent_state=None, training_budget_seconds=10800,
            total_budget_seconds=11700, deadline_utc="2100-01-01T00:00:00Z", prepare_only=False,
            dataset_input=None, submission_kit=None)
        self.calls = []
        for name, replacement in (("source_identity", lambda: {"commit": "c", "files": {}}),
                                  ("runtime_identity", lambda root: p.RUNTIME_COMMIT),
                                  ("require_gpu_platform", lambda: None)):
            self.enterContext(patch.object(p, name, replacement))
        self.enterContext(patch.object(p.guard, "exclusive_lock", lambda path: contextlib.nullcontext()))
        self.enterContext(patch.object(p.guard, "gpu_pids", return_value=[]))
        self.enterContext(patch.object(p.subprocess, "run", side_effect=self.prepare))
        self.enterContext(patch.object(p, "run_guard", side_effect=self.worker))
        self.enterContext(patch.object(p, "build_config", side_effect=lambda saved, **kw:
            {"training": {"total_environment_steps": kw["end"]}, "source": str(kw["state"])}))

    def prepare(self, command, **kwargs):
        self.assertIn("--_prepare-request", command)
        self.assertEqual(kwargs["env"]["CUDA_VISIBLE_DEVICES"], "")
        self.assertEqual(kwargs["env"]["JAX_PLATFORMS"], "cpu")
        config = self.output / "prepared/native_config.json"
        write(config, {"training": {"total_environment_steps": p.BOOTSTRAP_STEPS}})
        motion = self.output / "prepared/motion.parquet"
        motion.write_bytes(b"prepared")
        write(self.output / "preparation.json", {"job": self.identity, "native_config": str(config),
              "native_config_sha256": p.sha(config), "runtime_motion": str(motion),
              "runtime_motion_sha256": p.sha(motion), "parent": None})

    def worker(self, request, candidate, name, kind, seconds, command):
        self.calls.append((name, kind, seconds, command))
        ledger = p.guard.read_json(self.output / "ledger.json")
        ledger["jobs"].append({"name": name, "candidate": candidate, "kind": kind, "state": "complete",
                              "elapsed_seconds": 7000 if name == "bootstrap" else 2200 if name == "refinement" else 40})
        write(self.output / "ledger.json", ledger)
        if kind == "training":
            folder = self.output / name
            cfg = p.guard.read_json(self.output / (name + "_planned_config.json"))
            end = cfg["training"]["total_environment_steps"]
            latest = checkpoint(folder / "checkpoints", end, cfg)
            if name == "bootstrap":
                write(folder / "initial_checkpoint.json", checkpoint(folder / "checkpoints", 0, cfg))
            else:
                for additional in (20_000_000, 40_000_000, 60_000_000, 80_000_000):
                    checkpoint(folder / "checkpoints", p.BOOTSTRAP_STEPS + additional, cfg)
            write(folder / "resolved_config.json", cfg)
            write(folder / "status.json", {"state": "complete", "exit_code": 0, "forced_kill": False})
            write(folder / "latest_checkpoint.json", latest)
        else:
            folder = self.output / name
            policy = Path(command[command.index("--_evaluate") + 2])
            _, _, meta = p.checkpoint_pair(policy)
            steps = int(meta["environment_steps"])
            mp = 1.0 if steps == p.BOOTSTRAP_STEPS + 40_000_000 else .875
            report = {"checkpoint_sha256": p.sha(policy), "environment_steps": steps, "world_count": 16,
                      "step_count": 781, "start_frame": 0, "reset_mode": "explicit", "settling_steps": 0,
                      "metric_reference": "tier2_input_reference", "official_leaderboard_score": False,
                      "non_finite_worlds": [], "evaluation_contract": {"frame_zero": True,
                      "explicit_reset": True, "within_attempt_resets": 0, "voc_scale": 0., "reset_voc_scale": 0.,
                      "settling_steps": 0, "training_termination_disabled_during_rollout": True},
                      "leaderboard": {"scores": {"maniptrans_sr": mp, "spider_sr": 1., "add_auc": .8, "mppe_cm": 2.}}}
            write(folder / "evaluation.json", report)
            (folder / "rollout_arrays.npz").write_bytes(b"all16worlds")

    def test_full_automatic_path_and_complete_resume_without_worker_or_budget(self):
        selected = p.run(self.args)
        self.assertEqual([call[1] for call in self.calls], ["training", "training", "evaluation", "evaluation", "evaluation"])
        self.assertEqual(selected["environment_steps"], p.BOOTSTRAP_STEPS + 40_000_000)
        self.assertEqual(selected["state_sha256"], p.sha(selected["state"]))
        self.assertEqual(p.guard.read_json(self.output / "pipeline_status.json")["state"], "complete")
        with patch.object(p.guard, "gpu_pids", side_effect=AssertionError("No GPU check on complete resume")), \
             patch.object(p.guard, "remaining_budget", side_effect=AssertionError("No budget check on complete resume")):
            self.assertEqual(p.run(self.args), selected)
        self.assertEqual(len(self.calls), 5)
        Path(selected["state"]).write_bytes(Path(selected["state"]).read_bytes() + b"changed")
        with self.assertRaisesRegex(ValueError, "changed"):
            p.run(self.args)
        self.assertEqual(p.guard.read_json(self.output / "pipeline_status.json")["state"], "failed")

    def test_prepare_only_creates_no_gpu_ledger_and_resumes(self):
        self.args.prepare_only = True
        p.run(self.args)
        self.assertFalse((self.output / "ledger.json").exists())
        self.assertEqual(self.calls, [])
        p.run(self.args)
        self.assertEqual(p.guard.read_json(self.output / "pipeline_status.json")["state"], "prepared")

    def test_partial_training_is_never_restarted(self):
        self.args.prepare_only = True
        p.run(self.args)
        (self.output / "bootstrap").mkdir()
        self.args.prepare_only = False
        with self.assertRaisesRegex(RuntimeError, "duplicate launch refused"):
            p.run(self.args)
        self.assertEqual(self.calls, [])

    def test_incomplete_acquisition_cannot_start_refinement(self):
        original = self.worker
        def short_worker(*args):
            original(*args)
            if args[2] == "bootstrap":
                cfg = p.guard.read_json(self.output / "bootstrap_planned_config.json")
                write(self.output / "bootstrap/latest_checkpoint.json", checkpoint(self.output / "bootstrap/checkpoints", 100_000_000, cfg))
        with patch.object(p, "run_guard", side_effect=short_worker):
            with self.assertRaisesRegex(RuntimeError, "required"):
                p.run(self.args)
        self.assertEqual([call[0] for call in self.calls], ["bootstrap"])

    def test_export_required_before_complete_and_reverified(self):
        selected = p.run(self.args)
        request = p.guard.read_json(self.output / "request.json")
        request.update(dataset_input="released", submission_kit="kit")
        parquet, manifest = self.output / "export.parquet", self.output / "export_manifest.json"
        parquet.write_bytes(b"world0")
        write(manifest, {"state": "complete", "source_world_id": 0, "clean_reference_used": False,
                        "achieved_alignment": False, "exported_world_count": 1, "tier": 2, "episode": 3,
                        "selected_policy_manifest": {"sha256": p.sha(self.output / "selected_checkpoint.json")},
                        "validation": {"passed": True},
                        "artifact": {"sha256": p.sha(parquet)},
                        "inputs": {"policy_checkpoint": {"sha256": selected["checkpoint_sha256"]}}})
        record = {"state": "complete", "source_world_id": 0, "tier": 2, "episode": 3,
                  "parquet": str(parquet), "parquet_sha256": p.sha(parquet),
                  "manifest": str(manifest), "manifest_sha256": p.sha(manifest),
                  "selected_manifest_sha256": p.sha(self.output / "selected_checkpoint.json")}
        with patch("experiments.export_policy_rollout.export_selected", return_value=record) as export:
            self.assertEqual(p.finish_export(selected, request), record)
            self.assertEqual(p.finish_export(selected, request), record)
            self.assertEqual(export.call_count, 1)
        parquet.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "Export evidence"):
            p.finish_export(selected, request)


if __name__ == "__main__":
    unittest.main()
