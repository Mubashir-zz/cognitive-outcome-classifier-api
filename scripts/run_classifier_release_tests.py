#!/usr/bin/env python3
"""Run the fixed local release-test suites and write hash-bound evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


SCRIPT = Path(__file__).resolve()
STAGING_ROOT = SCRIPT.parents[1]
TRANSFER_LAYOUT = STAGING_ROOT.name == "staging_classifier_v2"
PROJECT_ROOT = STAGING_ROOT.parent if TRANSFER_LAYOUT else STAGING_ROOT
STAGING_TEST_DIRECTORY = "staging_classifier_v2/tests" if TRANSFER_LAYOUT else "tests"
STAGING_PREFIX = "staging_classifier_v2/" if TRANSFER_LAYOUT else ""
EVALUATOR_PATH = "evaluate_classifier_release_gates.py" if TRANSFER_LAYOUT else "training/evaluate_classifier_release_gates.py"
TEST_COUNT = re.compile(r"Ran (\d+) tests? in")
HEX40 = re.compile(r"^[0-9a-f]{40}$")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def run_suite(name: str, arguments: list[str]) -> dict:
    command = [sys.executable, *arguments]
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    combined = completed.stdout + b"\n" + completed.stderr
    decoded = combined.decode("utf-8", errors="replace")
    matches = TEST_COUNT.findall(decoded)
    tests_run = int(matches[-1]) if matches else None
    return {
        "name": name,
        "arguments": arguments,
        "return_code": completed.returncode,
        "status": "PASS" if completed.returncode == 0 else "FAIL",
        "tests_run": tests_run,
        "stdout_sha256": sha256_bytes(completed.stdout),
        "stderr_sha256": sha256_bytes(completed.stderr),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--build-commit", required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite test evidence: {args.output}")
    if not HEX40.fullmatch(args.build_commit):
        raise ValueError("Build commit must be exactly 40 lowercase hexadecimal characters")

    suites = [
        run_suite("project_unit", ["-m", "unittest", "discover", "-s", "tests"]),
        run_suite("staging_contract", ["-m", "unittest", "discover", "-s", STAGING_TEST_DIRECTORY]),
        run_suite(
            "python_syntax",
            [
                "-m", "py_compile",
                EVALUATOR_PATH,
                f"{STAGING_PREFIX}app/main.py",
                f"{STAGING_PREFIX}app/logic.py",
                f"{STAGING_PREFIX}app/model_runtime.py",
                f"{STAGING_PREFIX}scripts/benchmark_staging.py",
                f"{STAGING_PREFIX}scripts/verify_v8_runtime_parity.py",
                f"{STAGING_PREFIX}scripts/smoke_v8_api_contract.py",
                f"{STAGING_PREFIX}scripts/collect_classifier_technical_results.py",
            ],
        ),
    ]
    result = {
        "test_evidence_version": "1.0.0",
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "runner_sha256": hashlib.sha256(SCRIPT.read_bytes()).hexdigest(),
        "build_commit": args.build_commit,
        "python_version": sys.version.split()[0],
        "status": "PASS" if all(item["status"] == "PASS" for item in suites) else "FAIL",
        "suites": suites,
        "raw_outcome_text_stored": False,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if result["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
