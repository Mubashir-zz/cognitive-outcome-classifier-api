import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class DeploymentFileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.blueprint = (ROOT / "render.yaml").read_text(encoding="utf-8")
        cls.dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
        cls.benchmark = (ROOT / "scripts" / "benchmark_staging.py").read_text(encoding="utf-8")

    def test_blueprint_is_staging_only_and_manual(self):
        self.assertIn("name: cognitive-outcome-classifier-v2-staging", self.blueprint)
        self.assertIn("branch: codex/v2-rc2-validation-gated", self.blueprint)
        self.assertIn("autoDeployTrigger: off", self.blueprint)
        self.assertIn("healthCheckPath: /health", self.blueprint)

    def test_blueprint_does_not_hardcode_secrets(self):
        self.assertRegex(self.blueprint, r"key: HF_TOKEN\s+sync: false")
        self.assertRegex(self.blueprint, r"key: MODEL_REVISION\s+sync: false")
        self.assertRegex(self.blueprint, r"key: CLASSIFIER_API_KEY\s+generateValue: true")

    def test_container_honors_render_port_and_uses_one_worker(self):
        self.assertIn("${PORT:-10000}", self.dockerfile)
        self.assertIn("--workers 1", self.dockerfile)

    def test_benchmark_does_not_write_source_text(self):
        self.assertIn('"raw_outcome_text_stored": False', self.benchmark)
        self.assertIn("/diagnostics", self.benchmark)


if __name__ == "__main__":
    unittest.main()
