"""CPU-only installation-plan checks: no package install, network, or GPU."""
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from training import RUNTIME_COMMIT
from training.setup_policy_runtime import build_plan, ORT_CUDA12_INDEX


class SetupPlanTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.runtime = self.root / "runtime"
        project = self.runtime / "robotic_grounding/flash_chord"
        project.mkdir(parents=True)
        (project / "pyproject.toml").write_text('''[project]
dependencies = ["numpy==2.4.6", "onnxruntime-gpu==1.28.0"]
[project.optional-dependencies]
train = ["jax[cuda13]==0.10.2", "flax==0.12.7"]
''')
        (project / "uv.lock").write_text('''[[package]]
name = "tfp-nightly"
version = "0.26.0.dev20260705"
wheels = [{url="https://files.pythonhosted.org/tfp-nightly-none-any.whl",hash="sha256:''' + "a" * 64 + '''"}]
''')
        self.git = patch("training.setup_policy_runtime.subprocess.check_output", side_effect=[RUNTIME_COMMIT + "\n", ""])

    def plan(self, profile="cuda12", driver=570):
        with self.git:
            return build_plan(self.runtime, self.root / "new-env", profile, driver)

    def test_cuda12_separates_ort_feed_and_never_requests_cuda13(self):
        plan = self.plan()
        self.assertFalse((self.root / "new-env").exists())
        install = plan["commands"][1]
        self.assertIn("jax[cuda12]==0.10.2", install)
        self.assertFalse(any("cuda13" in item for cmd in plan["commands"] for item in cmd))
        self.assertNotIn("onnxruntime-gpu==1.28.0", install)
        ort = plan["commands"][2]
        self.assertIn(ORT_CUDA12_INDEX, ort)
        self.assertIn("--no-deps", ort)
        self.assertTrue(any("#sha256=" in item for item in install))
        self.assertIn("CUDA_VISIBLE_DEVICES=", plan["commands"][-1])
        self.assertTrue(any("LD_LIBRARY_PATH" in item for item in plan["activation"]))

    def test_cuda13_uses_official_frozen_lock(self):
        plan = self.plan("cuda13", 580)
        self.assertIn("--frozen", plan["commands"][0])
        self.assertIn("train", plan["commands"][0])
        self.assertIsNone(plan["cuda12_constraints_sha256"])
        self.assertFalse(any(ORT_CUDA12_INDEX in cmd for cmd in plan["commands"]))

    def test_reject_driver_mismatch(self):
        with self.assertRaisesRegex(ValueError, "driver"):
            self.plan("cuda13", 570)

    def test_reject_existing_environment(self):
        (self.root / "new-env").mkdir()
        with self.assertRaisesRegex(ValueError, "new environment"):
            self.plan()

    def test_reject_unpinned_runtime(self):
        with patch("training.setup_policy_runtime.subprocess.check_output", side_effect=["wrong\n", ""]):
            with self.assertRaisesRegex(ValueError, "runtime must be clean"):
                build_plan(self.runtime, self.root / "new-env", "cuda12", 570)

    def test_reject_nonpublic_tfp_source(self):
        path = self.runtime / "robotic_grounding/flash_chord/uv.lock"
        path.write_text(path.read_text().replace("files.pythonhosted.org", "private.example"))
        with self.assertRaisesRegex(ValueError, "public upstream"):
            self.plan()


if __name__ == "__main__":
    unittest.main()
