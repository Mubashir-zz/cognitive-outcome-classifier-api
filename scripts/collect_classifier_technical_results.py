#!/usr/bin/env python3
"""Derive release-gate fields from independent, hash-bound evidence artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path


SCRIPT = Path(__file__).resolve()
RUNNER = SCRIPT.with_name("run_classifier_release_tests.py")
STAGING_ROOT = SCRIPT.parents[1]
TRANSFER_LAYOUT = STAGING_ROOT.name == "staging_classifier_v2"
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
EXPECTED_SUITES = {
    "project_unit": ["-m", "unittest", "discover", "-s", "tests"],
    "staging_contract": [
        "-m", "unittest", "discover", "-s",
        "staging_classifier_v2/tests" if TRANSFER_LAYOUT else "tests",
    ],
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_object(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def finite_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--parity", type=Path, required=True)
    parser.add_argument("--internal-test", type=Path, required=True)
    parser.add_argument("--test-evidence", type=Path, required=True)
    parser.add_argument("--rollback-record", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite technical results: {args.output}")

    benchmark = load_object(args.benchmark)
    parity = load_object(args.parity)
    internal = load_object(args.internal_test)
    tests = load_object(args.test_evidence)
    rollback = load_object(args.rollback_record)
    health = benchmark.get("health", {})
    diagnostics = benchmark.get("diagnostics", {})
    regression = benchmark.get("frozen_regression", {})
    security = benchmark.get("security_contract", {})
    manifest_sha = health.get("model_manifest_sha256")
    model_sha = health.get("model_sha256")

    checks: dict[str, bool] = {}
    checks["benchmark_version"] = benchmark.get("benchmark_version") == "1.1.0"
    checks["benchmark_privacy"] = benchmark.get("raw_outcome_text_stored") is False
    checks["runtime_contract"] = (
        health.get("model_runtime") == "v8_chunked"
        and health.get("cns_decision_mode") == "model_primary"
        and health.get("prediction_authentication_enabled") is True
    )
    checks["health_hashes"] = all(HEX64.fullmatch(str(health.get(field, ""))) for field in (
        "model_sha256", "tokenizer_sha256", "model_manifest_sha256",
        "model_training_config_sha256", "model_development_release_sha256", "model_selection_sha256",
    ))
    checks["health_diagnostics_match"] = all(
        diagnostics.get(field) == health.get(field)
        for field in ("model_sha256", "tokenizer_sha256", "model_manifest_sha256", "build_commit")
    )
    checks["security_contract"] = (
        security.get("unauthenticated_prediction_status") == 401
        and security.get("non_echoing_validation_status") == 422
        and security.get("payload_limit_status") == 413
    )
    responses = [item.get("response", {}) for section in ("serial", "concurrent_four_requests") for item in benchmark.get(section, [])]
    checks["prediction_artifact_chain"] = bool(responses) and all(
        response.get("model_runtime") == "v8_chunked"
        and response.get("cns_decision_mode") == "model_primary"
        and response.get("model_sha256") == model_sha
        and response.get("tokenizer_sha256") == health.get("tokenizer_sha256")
        and response.get("model_manifest_sha256") == manifest_sha
        and response.get("model_training_config_sha256") == health.get("model_training_config_sha256")
        and response.get("model_development_release_sha256") == health.get("model_development_release_sha256")
        and response.get("model_selection_sha256") == health.get("model_selection_sha256")
        and response.get("model_selected_seed") == health.get("model_selected_seed")
        for response in responses
    )
    expected_probe_ids = {"probe-cns-positive", "probe-cns-negative", "probe-cns-keyword", "probe-alias", "probe-cns-long-tail"}
    checks["complete_probe_matrix"] = all(
        len(benchmark.get(section, [])) == len(expected_probe_ids)
        and {item.get("trial_id") for item in benchmark.get(section, [])} == expected_probe_ids
        for section in ("serial", "concurrent_four_requests")
    )
    cns_responses = [response for response in responses if response.get("cancer_type") == "CNS"]
    checks["full_text_contract"] = bool(cns_responses) and all(
        response.get("full_text_processed") is True and response.get("bert_truncated") is False
        for response in cns_responses
    )
    checks["long_text_chunked"] = any(
        item.get("trial_id") == "probe-cns-long-tail"
        and item.get("response", {}).get("model_chunk_count", 0) >= 2
        for item in benchmark.get("serial", [])
    )
    checks["frozen_regression"] = (
        regression.get("status") == "PASS"
        and regression.get("records", 0) > 0
        and regression.get("class_agreement") == 1.0
        and finite_number(regression.get("maximum_absolute_probability_delta"))
        and regression["maximum_absolute_probability_delta"] <= regression.get("allowed_maximum_probability_delta", -1)
        and regression.get("parity_evidence_sha256") == sha256(args.parity)
    )

    checks["parity"] = (
        parity.get("status") == "PASS"
        and parity.get("raw_outcome_text_stored") is False
        and parity.get("quantization_manifest_sha256") == manifest_sha
        and parity.get("serialized_torchscript_sha256") == model_sha
        and parity.get("class_agreement") == 1.0
        and parity.get("records", 0) > 0
        and len(parity.get("record_results", [])) == parity.get("records")
        and finite_number(parity.get("maximum_absolute_probability_delta"))
        and parity["maximum_absolute_probability_delta"] <= parity.get("allowed_maximum_probability_delta", -1)
    )
    checks["internal_test"] = (
        internal.get("status") == "INTERNAL_TEST_EVALUATED_ONCE"
        and internal.get("challenge_data_accessed") is False
        and internal.get("production_promotion_decided") is False
        and internal.get("internal_test_records", 0) > 0
        and internal.get("quantization_manifest_sha256") == manifest_sha
        and internal.get("evaluated_torchscript_sha256") == model_sha
        and internal.get("training_config_sha256") == health.get("model_training_config_sha256")
        and internal.get("development_release_sha256") == health.get("model_development_release_sha256")
        and internal.get("selection_sha256") == health.get("model_selection_sha256")
        and internal.get("selected_seed") == health.get("model_selected_seed")
    )

    suites = {item.get("name"): item for item in tests.get("suites", []) if isinstance(item, dict)}
    runner_bound = tests.get("runner_sha256") == sha256(RUNNER)
    suite_checks = {
        name: (
            suites.get(name, {}).get("arguments") == arguments
            and suites.get(name, {}).get("status") == "PASS"
            and suites.get(name, {}).get("return_code") == 0
            and isinstance(suites.get(name, {}).get("tests_run"), int)
            and suites[name]["tests_run"] > 0
        )
        for name, arguments in EXPECTED_SUITES.items()
    }
    syntax = suites.get("python_syntax", {})
    checks["test_evidence"] = (
        tests.get("status") == "PASS"
        and tests.get("raw_outcome_text_stored") is False
        and runner_bound
        and tests.get("build_commit") == health.get("build_commit")
        and bool(HEX40.fullmatch(str(tests.get("build_commit", ""))))
        and all(suite_checks.values())
        and syntax.get("status") == "PASS"
        and syntax.get("return_code") == 0
    )

    build_commit = str(health.get("build_commit", ""))
    checks["rollback_record"] = (
        rollback.get("rollback_record_version") == "1.0.0"
        and rollback.get("status") == "RECORDED_AND_RESTORE_DRILL_PASSED"
        and rollback.get("candidate_model_manifest_sha256") == manifest_sha
        and rollback.get("candidate_build_commit") == build_commit
        and bool(HEX40.fullmatch(build_commit))
        and bool(HEX40.fullmatch(str(rollback.get("rollback_build_commit", ""))))
        and rollback.get("rollback_build_commit") != build_commit
        and bool(DIGEST.fullmatch(str(rollback.get("rollback_container_image_digest", ""))))
        and rollback.get("restore_drill_status") == "PASS"
        and rollback.get("production_changed") is False
    )
    checks["resource_measurements"] = (
        finite_number(benchmark.get("cold_start_seconds"))
        and finite_number(diagnostics.get("peak_rss_mb"))
    )

    technical = {
        "technical_results_version": "1.0.0",
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "status": "PASS" if all(checks.values()) else "HOLD",
        "unit_tests_pass": checks["test_evidence"] and suite_checks.get("project_unit", False),
        "contract_tests_pass": checks["test_evidence"] and suite_checks.get("staging_contract", False),
        "frozen_regression_tests_pass": checks["frozen_regression"],
        "analytic_api_parity": parity.get("class_agreement", 0.0) if checks["parity"] else 0.0,
        "non_echoing_validation_errors": security.get("non_echoing_validation_status") == 422,
        "payload_limit_enforced": security.get("payload_limit_status") == 413,
        "artifact_hashes_exposed": checks["health_hashes"] and checks["prediction_artifact_chain"],
        "prediction_authentication_enabled": security.get("unauthenticated_prediction_status") == 401 and health.get("prediction_authentication_enabled") is True,
        "peak_rss_mb": diagnostics.get("peak_rss_mb") if finite_number(diagnostics.get("peak_rss_mb")) else None,
        "cold_start_seconds": benchmark.get("cold_start_seconds") if finite_number(benchmark.get("cold_start_seconds")) else None,
        "rollback_artifact_recorded": checks["rollback_record"],
        "model_runtime": health.get("model_runtime"),
        "cns_decision_mode": health.get("cns_decision_mode"),
        "model_manifest_sha256": manifest_sha,
        "parity_quantization_manifest_sha256": parity.get("quantization_manifest_sha256"),
        "internal_test_quantization_manifest_sha256": internal.get("quantization_manifest_sha256"),
        "evidence_checks": checks,
        "failed_evidence_checks": [name for name, passed in checks.items() if not passed],
        "input_sha256": {
            "benchmark": sha256(args.benchmark),
            "parity": sha256(args.parity),
            "internal_test": sha256(args.internal_test),
            "test_evidence": sha256(args.test_evidence),
            "rollback_record": sha256(args.rollback_record),
            "collector": sha256(SCRIPT),
        },
        "automatic_deployment_performed": False,
        "raw_outcome_text_stored": False,
    }
    args.output.write_text(json.dumps(technical, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(technical, indent=2))
    if technical["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
