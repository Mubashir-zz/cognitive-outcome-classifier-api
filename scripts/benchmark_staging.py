#!/usr/bin/env python3
"""Probe a deployed staging service and record cold-start and contract evidence."""

from __future__ import annotations

import argparse
import json
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
]
REQUIRED_FIELDS = {
    "predicted_cognitive", "decision_basis", "bert_probability", "keyword_hit",
    "keyword_matches", "source_text_sha256", "bert_truncated", "api_version",
    "decision_rule_version", "model_sha256", "tokenizer_sha256",
    "keyword_config_sha256", "build_commit",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--maximum-wait-seconds", type=float, default=120)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite benchmark output: {args.output}")
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

    def predict(item: dict) -> dict:
        start = time.perf_counter()
        response = requests.post(f"{base}/predict", headers=headers, json=item, timeout=60)
        elapsed = time.perf_counter() - start
        response.raise_for_status()
        body = response.json()
        missing = REQUIRED_FIELDS - set(body)
        if missing:
            raise ValueError(f"Prediction response lacks required fields: {sorted(missing)}")
        return {"trial_id": item["trial_id"], "latency_seconds": elapsed, "response": body}

    serial = [predict(item) for item in CASES]
    with ThreadPoolExecutor(max_workers=4) as pool:
        concurrent = list(pool.map(predict, CASES))
    diagnostics_response = requests.get(f"{base}/diagnostics", headers=headers, timeout=20)
    diagnostics_response.raise_for_status()
    diagnostics = diagnostics_response.json()

    output = {
        "benchmark_version": "1.0.0",
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "base_url": base,
        "cold_start_seconds": cold_start_seconds,
        "health": health,
        "serial": serial,
        "concurrent_four_requests": concurrent,
        "diagnostics": diagnostics,
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
