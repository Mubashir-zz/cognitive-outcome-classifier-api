import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "app" / "main.py"


class MainStaticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = MAIN.read_text(encoding="utf-8")
        cls.tree = ast.parse(cls.source)

    def function_source(self, name):
        node = next(item for item in self.tree.body if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == name)
        return ast.get_source_segment(self.source, node)

    def test_review_webhook_does_not_receive_raw_outcome_text(self):
        function = self.function_source("log_review_event")
        self.assertNotIn('"outcome_text"', function)
        self.assertIn('"source_text_sha256"', function)

    def test_validation_errors_are_non_echoing(self):
        function = self.function_source("non_echoing_validation_error")
        self.assertNotIn("input", function)
        self.assertIn("Request validation failed", function)

    def test_security_and_payload_controls_are_present(self):
        function = self.function_source("request_security_controls")
        self.assertIn("MAX_REQUEST_BYTES", function)
        self.assertIn("compare_digest", function)
        self.assertIn("413", function)
        self.assertIn("RequestBodyLimitMiddleware", self.source)
        self.assertIn("consumed > self.max_bytes", self.source)

    def test_response_exposes_provenance_and_truncation(self):
        for field in (
            "source_text_sha256", "model_sha256", "tokenizer_sha256",
            "keyword_config_sha256", "decision_rule_version", "build_commit",
            "bert_truncated", "bert_input_tokens",
            "model_manifest_sha256", "model_training_config_sha256",
            "model_development_release_sha256", "model_selection_sha256",
            "full_text_processed", "model_chunk_count", "max_chunk_start_character",
        ):
            self.assertIn(field, self.source)

    def test_no_bare_exception_suppression(self):
        bare_handlers = [
            node for node in ast.walk(self.tree)
            if isinstance(node, ast.ExceptHandler) and node.type is None
        ]
        self.assertEqual(bare_handlers, [])

    def test_authenticated_diagnostics_exposes_memory_without_text(self):
        function = self.function_source("diagnostics")
        self.assertIn("peak_rss_mb", function)
        self.assertNotIn("outcome_text", function)
        middleware = self.function_source("request_security_controls")
        self.assertNotIn('"/diagnostics"', middleware)

    def test_v8_runtime_is_explicit_and_overlimit_input_fails(self):
        self.assertIn('MODEL_RUNTIME not in {"legacy_v7", "v8_chunked"}', self.source)
        self.assertIn("load_v8_artifact_spec", self.source)
        self.assertIn("V8ChunkedRuntime", self.source)
        self.assertIn("except ModelInputTooLongError", self.source)
        self.assertIn("status_code=413", self.source)

    def test_v8_model_is_primary_and_keyword_is_review_evidence(self):
        self.assertIn('CNS_DECISION_MODE = "model_primary"', self.source)
        self.assertIn("cns_model_primary_decision", self.source)


if __name__ == "__main__":
    unittest.main()
