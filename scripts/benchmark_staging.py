#!/usr/bin/env python3
"""Probe a deployed staging service and record cold-start and contract evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import requests


CASES = [
    {"cancer_type": "CNS", "outcome_text": "Montreal Cognitive Assessment at baseline and 6 months", "trial_id": "probe-cns-positive"},
    {"cancer_type": "CNS", "outcome_text": "Overall survival and progression-free survival", "trial_id": "probe-cns-negative"},
    {"cancer_type": "CNS", "outcome_text": "Trail Making Test Part B", "trial_id": "probe-cns-keyword"},
    {"cancer_type": "Head & Neck", "outcome_text": "Montreal Cognitive Assessment", "trial_id": "probe-alias"},
    {"cancer_type": "CNS", "outcome_text": " ".join(["survival"] * 430 + ["Montreal Cognitive Assessment at month six"]), "trial_id": "probe-cns-long-tail"},
]
REQUIRED_FIELDS = {
    "predicted_cognitive", "decision_basis", "bert_probability", "keyword_hit",
    "keyword_matches", "source_text_sha256", "bert_truncated", "api_version",
    "decision_rule_version", "model_sha256", "tokenizer_sha256",
    "keyword_config_sha256", "build_commit",
    "model_runtime", "cns_decision_mode", "model_manifest_sha256",
    "model_training_config_sha256", "model_development_release_sha256",
    "model_selection_sha256", "model_selected_seed", "full_text_processed",
    "model_chunk_count", "max_chunk_index", "max_chunk_start_token",
    "max_chunk_end_token", "max_chunk_start_character", "max_chunk_end_character",
}
HEX64 = re.compile(r"^[0-9a-f]{64}$")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_regression_fixture(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            record_id = str(item.get("record_id", "")).strip()
            outcome_text = item.get("outcome_text")
            if not record_id or not isinstance(outcome_text, str) or not outcome_text.strip():
                raise ValueError(f"Invalid regression fixture at line {line_number}")
            rows.append({"record_id": record_id, "outcome_text": outcome_text})
    if not rows or len({row["record_id"] for row in rows}) != len(rows):
        raise ValueError("Regression fixture must contain unique, nonempty records")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-model-manifest-sha256", required=True)
    parser.add_argument("--regression-fixture", type=Path)
    parser.add_argument("--parity-evidence", type=Path)
    parser.add_argument("--maximum-regression-probability-delta", type=float, default=1e-6)
    parser.add_argument("--maximum-wait-seconds", type=float, default=120)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite benchmark output: {args.output}")
    if not HEX64.fullmatch(args.expected_model_manifest_sha256):
        raise ValueError("Expected model-manifest SHA-256 must be exactly 64 lowercase hexadecimal characters")
    if (args.regression_fixture is None) != (args.parity_evidence is None):
        raise ValueError("Regression fixture and parity evidence must be supplied together")
    base = args.base_url.rstrip("/")
    headers = {"X-API-Key": args.api_key}

    started = time.perf_counter()
    health = None
    health_errors: list[str] = []
    while time.perf_counter() - started <= args.maximum_wait_seconds:
        try:
            response = requests.get(f"{base}/health", timeout=10)
            if response.ok:
                health = response.json()
                break
            health_errors.append(f"HTTP {response.status_code}")
        except requests.RequestException as exc:
            health_errors.append(type(exc).__name__)
        time.sleep(2)
    cold_start_seconds = time.perf_counter() - started
    if health is None:
        raise RuntimeError(f"Health check did not pass within the limit; recent errors: {health_errors[-5:]}")
    if health.get("model_runtime") != "v8_chunked" or health.get("cns_decision_mode") != "model_primary":
        raise RuntimeError("Staging service is not running the required v8 model-primary contract")
    if health.get("model_manifest_sha256") != args.expected_model_manifest_sha256:
        raise RuntimeError("Staging service manifest hash does not match the prespecified artifact")
    if health.get("prediction_authentication_enabled") is not True:
        raise RuntimeError("Prediction authentication is not enabled")

    unauthorized = requests.post(f"{base}/predict", json=CASES[0], timeout=20)
    if unauthorized.status_code != 401:
        raise RuntimeError(f"Prediction route did not reject a missing API key: HTTP {unauthorized.status_code}")

    marker = "PRIVATE-TEXT-MUST-NOT-ECHO"
    invalid = requests.post(
        f"{base}/predict", headers=headers,
        json={"cancer_type": "CNS", "outcome_text": marker, "trial_id": "x" * 129},
        timeout=20,
    )
    if invalid.status_code != 422 or marker in invalid.text:
        raise RuntimeError("Validation error was not non-echoing")

    oversized_payload = b"x" * 750_001
    oversized = requests.post(
        f"{base}/predict", headers=headers, data=oversized_payload, timeout=20
    )
    if oversized.status_code != 413:
        raise RuntimeError(f"Payload limit was not enforced: HTTP {oversized.status_code}")

    def predict(item: dict) -> dict:
        start = time.perf_counter()
        response = requests.post(f"{base}/predict", headers=headers, json=item, timeout=60)
        elapsed = time.perf_counter() - start
        response.raise_for_status()
        body = response.json()
        missing = REQUIRED_FIELDS - set(body)
        if missing:
            raise ValueError(f"Prediction response lacks required fields: {sorted(missing)}")
        if body["model_sha256"] != health.get("model_sha256") or body["tokenizer_sha256"] != health.get("tokenizer_sha256"):
            raise ValueError("Prediction and health artifact hashes differ")
        if body["model_runtime"] != "v8_chunked" or body["cns_decision_mode"] != "model_primary":
            raise ValueError("Prediction response left the required v8 model-primary contract")
        if body["model_manifest_sha256"] != health.get("model_manifest_sha256"):
            raise ValueError("Prediction and health v8 manifest hashes differ")
        if item["cancer_type"] == "CNS" and (body["full_text_processed"] is not True or body["bert_truncated"] is not False):
            raise ValueError("V8 CNS response does not certify full-text processing")
        return {"trial_id": item["trial_id"], "latency_seconds": elapsed, "response": body}

    serial = [predict(item) for item in CASES]
    long_tail = next(item for item in serial if item["trial_id"] == "probe-cns-long-tail")["response"]
    if long_tail["model_chunk_count"] is None or long_tail["model_chunk_count"] < 2:
        raise RuntimeError("Long-tail probe was not processed through multiple v8 chunks")
    with ThreadPoolExecutor(max_workers=4) as pool:
        concurrent = list(pool.map(predict, CASES))
    diagnostics_response = requests.get(f"{base}/diagnostics", headers=headers, timeout=20)
    diagnostics_response.raise_for_status()
    diagnostics = diagnostics_response.json()
    for field in ("model_sha256", "tokenizer_sha256", "model_manifest_sha256", "build_commit"):
        if diagnostics.get(field) != health.get(field):
            raise RuntimeError(f"Diagnostics and health disagree on {field}")

    regression = {
        "status": "NOT_RUN",
        "records": 0,
        "class_agreement": None,
        "maximum_absolute_probability_delta": None,
        "fixture_sha256": None,
        "parity_evidence_sha256": None,
        "raw_outcome_text_stored": False,
    }
    if args.regression_fixture is not None and args.parity_evidence is not None:
        fixture = read_regression_fixture(args.regression_fixture)
        parity = json.loads(args.parity_evidence.read_text(encoding="utf-8"))
        if parity.get("status") != "PASS":
            raise RuntimeError("Analytic/runtime parity evidence did not pass")
        if parity.get("fixture_sha256") != sha256(args.regression_fixture):
            raise RuntimeError("Parity evidence is not bound to this regression fixture")
        if parity.get("quantization_manifest_sha256") != args.expected_model_manifest_sha256:
            raise RuntimeError("Parity evidence and deployed service use different manifests")
        expected = {item["record_id"]: item for item in parity.get("record_results", [])}
        if len(expected) != len(parity.get("record_results", [])) or len(expected) != len(fixture):
            raise RuntimeError("Parity evidence contains missing or duplicate regression records")
        if set(expected) != {item["record_id"] for item in fixture}:
            raise RuntimeError("Parity record results do not exactly cover the regression fixture")
        deltas: list[float] = []
        agreements: list[bool] = []
        for item in fixture:
            record = expected[item["record_id"]]
            source_hash = hashlib.sha256(item["outcome_text"].encode("utf-8")).hexdigest()
            if record.get("source_text_sha256") != source_hash:
                raise RuntimeError(f"Regression text hash mismatch for {item['record_id']}")
            observed = predict({
                "cancer_type": "CNS",
                "outcome_text": item["outcome_text"],
                "trial_id": item["record_id"],
            })["response"]
            delta = abs(float(observed["bert_probability"]) - float(record["analytic_probability"]))
            deltas.append(delta)
            agreements.append(bool(observed["predicted_cognitive"]) is bool(record["analytic_class"]))
        regression = {
            "status": "PASS" if all(agreements) and max(deltas) <= args.maximum_regression_probability_delta else "FAIL",
            "records": len(fixture),
            "class_agreement": sum(agreements) / len(agreements),
            "maximum_absolute_probability_delta": max(deltas),
            "allowed_maximum_probability_delta": args.maximum_regression_probability_delta,
            "fixture_sha256": sha256(args.regression_fixture),
            "parity_evidence_sha256": sha256(args.parity_evidence),
            "raw_outcome_text_stored": False,
        }
        if regression["status"] != "PASS":
            raise RuntimeError(f"Frozen deployed regression failed: {regression}")

    output = {
        "benchmark_version": "1.1.0",
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "base_url": base,
        "cold_start_seconds": cold_start_seconds,
        "health": health,
        "serial": serial,
        "concurrent_four_requests": concurrent,
        "diagnostics": diagnostics,
        "frozen_regression": regression,
        "security_contract": {
            "unauthenticated_prediction_status": unauthorized.status_code,
            "non_echoing_validation_status": invalid.status_code,
            "payload_limit_status": oversized.status_code,
        },
        "raw_outcome_text_stored": False,
    }
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "PASS",
        "cold_start_seconds": round(cold_start_seconds, 3),
        "peak_rss_mb": diagnostics.get("peak_rss_mb"),
        "output": str(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
