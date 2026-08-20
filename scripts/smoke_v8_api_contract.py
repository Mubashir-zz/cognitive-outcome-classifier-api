#!/usr/bin/env python3
"""Exercise the real FastAPI stack with a supplied hash-pinned v8 artifact."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve()
STAGING_ROOT = SCRIPT.parents[1]
sys.path.insert(0, str(STAGING_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=STAGING_ROOT / "hybrid_config.json")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite API smoke evidence: {args.output}")

    os.environ.update({
        "MODEL_RUNTIME": "v8_chunked",
        "MODEL_DIR": str(args.artifact_dir.resolve()),
        "MODEL_MANIFEST_SHA256": args.manifest_sha256,
        "CONFIG_PATH": str(args.config.resolve()),
        "CLASSIFIER_API_KEY": "local-contract-smoke-key",
        "V8_MAX_INPUT_TOKENS": "600",
        "BUILD_COMMIT": "0" * 40,
    })

    from fastapi.testclient import TestClient
    from app.main import app

    client = TestClient(app)
    headers = {"X-API-Key": "local-contract-smoke-key"}
    health_response = client.get("/health")
    health_response.raise_for_status()
    health = health_response.json()
    if health.get("model_runtime") != "v8_chunked" or health.get("cns_decision_mode") != "model_primary":
        raise RuntimeError("API did not start in the v8 model-primary mode")
    if health.get("model_manifest_sha256") != args.manifest_sha256:
        raise RuntimeError("API health returned an unexpected model manifest")
    if health.get("prediction_authentication_enabled") is not True:
        raise RuntimeError("API smoke requires prediction authentication")

    unauthorized = client.post("/predict", json={
        "cancer_type": "CNS", "outcome_text": "overall survival", "trial_id": "unauthorized",
    })
    long_response = client.post("/predict", headers=headers, json={
        "cancer_type": "CNS",
        "outcome_text": " ".join(["survival"] * 430 + ["Montreal Cognitive Assessment at month six"]),
        "trial_id": "synthetic-long",
    })
    long_response.raise_for_status()
    long_body = long_response.json()
    if long_body.get("model_chunk_count", 0) < 2 or long_body.get("full_text_processed") is not True or long_body.get("bert_truncated") is not False:
        raise RuntimeError("Long CNS input did not satisfy the full-text chunk contract")
    if long_body.get("model_manifest_sha256") != args.manifest_sha256:
        raise RuntimeError("Prediction manifest differs from health")

    non_cns = client.post("/predict", headers=headers, json={
        "cancer_type": "Breast", "outcome_text": "Montreal Cognitive Assessment", "trial_id": "synthetic-breast",
    })
    non_cns.raise_for_status()
    invalid_marker = "PRIVATE-TEXT-MUST-NOT-ECHO"
    invalid = client.post("/predict", headers=headers, json={
        "cancer_type": "CNS", "outcome_text": invalid_marker, "trial_id": "x" * 129,
    })
    oversized = client.post("/predict", headers=headers, content=b"x" * 750_001)
    token_overlimit = client.post("/predict", headers=headers, json={
        "cancer_type": "CNS", "outcome_text": " ".join(["survival"] * 650), "trial_id": "token-overlimit",
    })
    checks = {
        "unauthenticated_status": unauthorized.status_code,
        "long_text_chunks": long_body["model_chunk_count"],
        "long_text_full_processed": long_body["full_text_processed"],
        "long_text_truncated": long_body["bert_truncated"],
        "non_cns_keyword_prediction": non_cns.json().get("predicted_cognitive"),
        "non_echoing_validation_status": invalid.status_code,
        "validation_marker_echoed": invalid_marker in invalid.text,
        "streaming_payload_limit_status": oversized.status_code,
        "token_safety_limit_status": token_overlimit.status_code,
    }
    passed = (
        checks["unauthenticated_status"] == 401
        and checks["long_text_chunks"] >= 2
        and checks["long_text_full_processed"] is True
        and checks["long_text_truncated"] is False
        and checks["non_cns_keyword_prediction"] is True
        and checks["non_echoing_validation_status"] == 422
        and checks["validation_marker_echoed"] is False
        and checks["streaming_payload_limit_status"] == 413
        and checks["token_safety_limit_status"] == 413
    )
    result = {
        "status": "PASS_SYNTHETIC_API_CONTRACT" if passed else "FAIL",
        "model_manifest_sha256": health.get("model_manifest_sha256"),
        "model_sha256": health.get("model_sha256"),
        "tokenizer_sha256": health.get("tokenizer_sha256"),
        "checks": checks,
        "synthetic_or_local_contract_evidence_only": True,
        "scientific_performance_evaluated": False,
        "raw_outcome_text_stored": False,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
