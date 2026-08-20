import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class DeploymentFileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.blueprint = (ROOT / "render.yaml").read_text(encoding="utf-8")
        cls.dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
        cls.benchmark = (ROOT / "scripts" / "benchmark_staging.py").read_text(encoding="utf-8")
        cls.v8_parity = (ROOT / "scripts" / "verify_v8_runtime_parity.py").read_text(encoding="utf-8")
        cls.collector = (ROOT / "scripts" / "collect_classifier_technical_results.py").read_text(encoding="utf-8")
        cls.api_smoke = (ROOT / "scripts" / "smoke_v8_api_contract.py").read_text(encoding="utf-8")
        cls.test_runner = (ROOT / "scripts" / "run_classifier_release_tests.py").read_text(encoding="utf-8")

    def test_blueprint_is_staging_only_and_manual(self):
        self.assertIn("name: cognitive-outcome-classifier-v2-staging", self.blueprint)
        self.assertIn("branch: codex/v2-rc2-validation-gated", self.blueprint)
        self.assertIn("autoDeployTrigger: off", self.blueprint)
        self.assertIn("healthCheckPath: /health", self.blueprint)
        self.assertRegex(self.blueprint, r"key: MODEL_RUNTIME\s+value: legacy_v7")
        self.assertRegex(self.blueprint, r"key: V8_MAX_INPUT_TOKENS\s+value: \"50000\"")
        self.assertRegex(self.blueprint, r"key: MODEL_MANIFEST_SHA256\s+sync: false")

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
        self.assertIn("expected-model-manifest-sha256", self.benchmark)
        self.assertIn("frozen_regression", self.benchmark)
        self.assertIn('oversized_payload = b"x" * 750_001', self.benchmark)

    def test_v8_parity_is_exact_and_does_not_write_source_text(self):
        self.assertIn("score_fixed_length", self.v8_parity)
        self.assertIn("V8ChunkedRuntime", self.v8_parity)
        self.assertIn('default=1e-12', self.v8_parity)
        self.assertIn('"record_results"', self.v8_parity)
        self.assertIn('"raw_outcome_text_stored": False', self.v8_parity)

    def test_technical_results_are_derived_from_evidence(self):
        self.assertIn('"benchmark"', self.collector)
        self.assertIn('"parity"', self.collector)
        self.assertIn('"internal_test"', self.collector)
        self.assertIn('"rollback_record"', self.collector)
        self.assertIn('"failed_evidence_checks"', self.collector)
        self.assertIn('"automatic_deployment_performed": False', self.collector)

    def test_api_smoke_exercises_full_stack_without_storing_text(self):
        self.assertIn("TestClient", self.api_smoke)
        self.assertIn('content=b"x" * 750_001', self.api_smoke)
        self.assertIn("token-overlimit", self.api_smoke)
        self.assertIn('"raw_outcome_text_stored": False', self.api_smoke)

    def test_scripts_resolve_transfer_and_repository_layouts(self):
        self.assertIn('STAGING_ROOT.name == "staging_classifier_v2"', self.v8_parity)
        self.assertIn('STAGING_ROOT.name == "staging_classifier_v2"', self.test_runner)
        self.assertIn("STAGING_TEST_DIRECTORY", self.test_runner)
        self.assertIn("TRANSFER_LAYOUT", self.collector)


if __name__ == "__main__":
    unittest.main()
