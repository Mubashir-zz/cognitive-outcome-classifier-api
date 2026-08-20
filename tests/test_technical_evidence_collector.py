#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


STAGING = Path(__file__).resolve().parents[1]
ROOT = STAGING.parent
CONTRACT_TEST_DIRECTORY = "staging_classifier_v2/tests" if STAGING.name == "staging_classifier_v2" else "tests"
COLLECTOR = STAGING / "scripts" / "collect_classifier_technical_results.py"
RUNNER = STAGING / "scripts" / "run_classifier_release_tests.py"
MANIFEST_SHA = "a" * 64
MODEL_SHA = "b" * 64
TOKENIZER_SHA = "c" * 64
TRAINING_SHA = "d" * 64
RELEASE_SHA = "e" * 64
SELECTION_SHA = "f" * 64
BUILD_COMMIT = "1" * 40


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def response(trial_id: str, chunks: int = 1) -> dict:
    return {
        "trial_id": trial_id,
        "cancer_type": "CNS",
        "full_text_processed": True,
        "bert_truncated": False,
        "model_chunk_count": chunks,
        "model_runtime": "v8_chunked",
        "cns_decision_mode": "model_primary",
        "model_sha256": MODEL_SHA,
        "tokenizer_sha256": TOKENIZER_SHA,
        "model_manifest_sha256": MANIFEST_SHA,
        "model_training_config_sha256": TRAINING_SHA,
        "model_development_release_sha256": RELEASE_SHA,
        "model_selection_sha256": SELECTION_SHA,
        "model_selected_seed": 20260821,
    }


def make_evidence(root: Path, mismatch: bool = False) -> dict[str, Path]:
    paths = {name: root / f"{name}.json" for name in ("benchmark", "parity", "internal", "tests", "rollback", "output")}
    parity = {
        "status": "PASS",
        "raw_outcome_text_stored": False,
        "quantization_manifest_sha256": "9" * 64 if mismatch else MANIFEST_SHA,
        "serialized_torchscript_sha256": MODEL_SHA,
        "records": 1,
        "class_agreement": 1.0,
        "maximum_absolute_probability_delta": 0.0,
        "allowed_maximum_probability_delta": 1e-12,
        "record_results": [{"record_id": "r1"}],
    }
    write_json(paths["parity"], parity)
    health = {
        "model_runtime": "v8_chunked",
        "cns_decision_mode": "model_primary",
        "prediction_authentication_enabled": True,
        "model_sha256": MODEL_SHA,
        "tokenizer_sha256": TOKENIZER_SHA,
        "model_manifest_sha256": MANIFEST_SHA,
        "model_training_config_sha256": TRAINING_SHA,
        "model_development_release_sha256": RELEASE_SHA,
        "model_selection_sha256": SELECTION_SHA,
        "model_selected_seed": 20260821,
        "build_commit": BUILD_COMMIT,
    }
    probe_ids = ["probe-cns-positive", "probe-cns-negative", "probe-cns-keyword", "probe-alias", "probe-cns-long-tail"]
    serial = [{"trial_id": item, "response": response(item, chunks=2 if item == "probe-cns-long-tail" else 1)} for item in probe_ids]
    concurrent = [{"trial_id": item, "response": response(item, chunks=2 if item == "probe-cns-long-tail" else 1)} for item in probe_ids]
    benchmark = {
        "benchmark_version": "1.1.0",
        "raw_outcome_text_stored": False,
        "cold_start_seconds": 10.0,
        "health": health,
        "diagnostics": {**health, "peak_rss_mb": 300.0},
        "serial": serial,
        "concurrent_four_requests": concurrent,
        "security_contract": {
            "unauthenticated_prediction_status": 401,
            "non_echoing_validation_status": 422,
            "payload_limit_status": 413,
        },
        "frozen_regression": {
            "status": "PASS",
            "records": 1,
            "class_agreement": 1.0,
            "maximum_absolute_probability_delta": 0.0,
            "allowed_maximum_probability_delta": 1e-6,
            "parity_evidence_sha256": digest(paths["parity"]),
        },
    }
    write_json(paths["benchmark"], benchmark)
    write_json(paths["internal"], {
        "status": "INTERNAL_TEST_EVALUATED_ONCE",
        "challenge_data_accessed": False,
        "production_promotion_decided": False,
        "internal_test_records": 337,
        "quantization_manifest_sha256": MANIFEST_SHA,
        "evaluated_torchscript_sha256": MODEL_SHA,
        "training_config_sha256": TRAINING_SHA,
        "development_release_sha256": RELEASE_SHA,
        "selection_sha256": SELECTION_SHA,
        "selected_seed": 20260821,
    })
    write_json(paths["tests"], {
        "status": "PASS",
        "raw_outcome_text_stored": False,
        "runner_sha256": digest(RUNNER),
        "build_commit": BUILD_COMMIT,
        "suites": [
            {"name": "project_unit", "arguments": ["-m", "unittest", "discover", "-s", "tests"], "status": "PASS", "return_code": 0, "tests_run": 47},
            {"name": "staging_contract", "arguments": ["-m", "unittest", "discover", "-s", CONTRACT_TEST_DIRECTORY], "status": "PASS", "return_code": 0, "tests_run": 35},
            {"name": "python_syntax", "arguments": ["fixed"], "status": "PASS", "return_code": 0, "tests_run": None},
        ],
    })
    write_json(paths["rollback"], {
        "rollback_record_version": "1.0.0",
        "status": "RECORDED_AND_RESTORE_DRILL_PASSED",
        "candidate_model_manifest_sha256": MANIFEST_SHA,
        "candidate_build_commit": BUILD_COMMIT,
        "rollback_build_commit": "2" * 40,
        "rollback_container_image_digest": "sha256:" + "3" * 64,
        "restore_drill_status": "PASS",
        "production_changed": False,
    })
    return paths


def collect(paths: dict[str, Path]) -> subprocess.CompletedProcess[str]:
    return subprocess.run([
        sys.executable, str(COLLECTOR),
        "--benchmark", str(paths["benchmark"]),
        "--parity", str(paths["parity"]),
        "--internal-test", str(paths["internal"]),
        "--test-evidence", str(paths["tests"]),
        "--rollback-record", str(paths["rollback"]),
        "--output", str(paths["output"]),
    ], cwd=ROOT, text=True, capture_output=True, check=False)


class TechnicalEvidenceCollectorTests(unittest.TestCase):
    def test_complete_hash_bound_evidence_passes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = make_evidence(Path(temporary))
            completed = collect(paths)
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            result = json.loads(paths["output"].read_text(encoding="utf-8"))
            self.assertEqual(result["status"], "PASS")
            self.assertEqual(result["failed_evidence_checks"], [])
            self.assertFalse(result["automatic_deployment_performed"])

    def test_manifest_mismatch_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = make_evidence(Path(temporary), mismatch=True)
            completed = collect(paths)
            self.assertEqual(completed.returncode, 2)
            result = json.loads(paths["output"].read_text(encoding="utf-8"))
            self.assertEqual(result["status"], "HOLD")
            self.assertIn("parity", result["failed_evidence_checks"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
