# Neurocognitive outcome classifier — v2 release candidate 3

This folder is a staging candidate, not the production service.

It supports two explicit CNS runtimes. `legacy_v7` preserves the existing quantized detector and BERT-or-keyword rule. `v8_chunked` loads only a SHA-256-pinned quantization manifest, model, inference contract, and complete tokenizer; it scores every accepted token with the frozen overlapping-chunk policy. Under v8, the model label is primary and keyword disagreement sends the record to review without overriding that label.

A response reports model probability, keyword evidence and character offsets, decision basis, review reasons, source-text hash, full-text/chunk status, winning chunk token and character offsets, artifact/provenance hashes, and build commit. Raw outcome text is not sent to the review webhook.

Production promotion is intentionally blocked until the disjoint `CNS_challenge_set_300_BLINDED` sample receives frozen independent human labels and the release candidate passes the prespecified scientific, analytic-parity, memory, security, and regression gates.

Legacy parameters remain versioned in `hybrid_config.json`. V8 sequence length, overlap, aggregation, and threshold come from the hash-verified inference contract. Inputs beyond the configured full-text safety ceiling receive HTTP 413; they are never silently truncated.

## Local model test

Set `MODEL_DIR` to a directory containing the tokenizer files and `cns_v7_quantized.pt`, then run one worker:

```bash
CONFIG_PATH=./hybrid_config.json MODEL_DIR=/path/to/model uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers 1
```

For a frozen v8 artifact directory:

```bash
MODEL_RUNTIME=v8_chunked \
MODEL_DIR=/path/to/v8_quantized_selected \
MODEL_MANIFEST_SHA256=<exact-quantization-manifest-sha256> \
CONFIG_PATH=./hybrid_config.json \
uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers 1
```

For Hugging Face loading, omit `MODEL_DIR` and set `MODEL_REPO`, an immutable 40-character `MODEL_REVISION`, `MODEL_MANIFEST_FILENAME`, `MODEL_MANIFEST_SHA256`, and `HF_TOKEN`. Startup fails if any model, tokenizer, contract, manifest, or provenance binding differs.

The local full-stack contract smoke uses the real FastAPI request path and a supplied artifact. It checks authentication, non-echoing validation, streamed payload limits, token safety limits, multi-chunk full-text inference, and non-CNS behavior without recording input text:

```bash
python3 scripts/smoke_v8_api_contract.py \
  --artifact-dir /path/to/v8_quantized_selected \
  --manifest-sha256 <exact-quantization-manifest-sha256> \
  --output v8_api_contract_smoke.json
```

## Required staging secrets

- `HF_TOKEN` when loading the private model from Hugging Face
- `CLASSIFIER_API_KEY` to require `X-API-Key` on prediction routes
- `REVIEW_QUEUE_WEBHOOK` only if the review queue accepts metadata without raw outcome text
- `MODEL_MANIFEST_SHA256` when `MODEL_RUNTIME=v8_chunked`

## Gates before deployment

1. Blinded human adjudication completed, frozen, and analyzed with the challenge-set design weights.
2. The exact serialized v8 artifact passes calibration parity, one-time internal testing, blinded challenge scoring, and design-weighted v1.2 gates.
3. Unit, API-contract, regression, cold-start, peak-memory, and concurrency tests pass.
4. A separate staging service passes shadow comparison against production.
5. Rollback image and immutable hashes are recorded.

Passing every gate allows manual promotion review; it never deploys automatically.

## Hash-bound technical evidence

After the selected v8 artifact exists, create a small private JSONL parity fixture with `record_id` and `outcome_text`. First compare the frozen analytic scorer with the API runtime locally:

```bash
python3 scripts/verify_v8_runtime_parity.py \
  --artifact-dir /path/to/v8_quantized_selected \
  --manifest-sha256 <exact-quantization-manifest-sha256> \
  --fixture /private/path/v8_regression_fixture.jsonl \
  --output v8_runtime_parity.json
```

Then benchmark the separately deployed staging service. The output stores record IDs, hashes, timings, and aggregate regression evidence, but no outcome text:

```bash
python3 scripts/benchmark_staging.py \
  --base-url https://<staging-service> \
  --api-key <staging-key> \
  --expected-model-manifest-sha256 <exact-quantization-manifest-sha256> \
  --regression-fixture /private/path/v8_regression_fixture.jsonl \
  --parity-evidence v8_runtime_parity.json \
  --output v8_staging_benchmark.json
```

Generate local test evidence with the fixed runner:

```bash
python3 scripts/run_classifier_release_tests.py \
  --build-commit <exact-40-character-staging-build-commit> \
  --output classifier_test_evidence.json
```

Finally derive the technical gate input from five independent artifacts:

```bash
python3 scripts/collect_classifier_technical_results.py \
  --benchmark v8_staging_benchmark.json \
  --parity v8_runtime_parity.json \
  --internal-test /private/path/internal_test_evaluation.json \
  --test-evidence classifier_test_evidence.json \
  --rollback-record rollback_record.json \
  --output classifier_technical_results.json
```

The rollback record must identify the candidate manifest and build commit, a distinct 40-character rollback commit, an immutable `sha256:<64 hex>` container digest, a passed restore drill, and `production_changed: false`. Missing or inconsistent evidence produces `HOLD`; the collector and release evaluator never deploy.

## Complete readiness audit

After the scientific and technical evidence has been generated, run the independent chain auditor from the private workspace:

```bash
python3 training/verify_cns_expansion_readiness.py \
  --workspace /private/path/cns_v8 \
  --output /private/path/cns_v8/CNS_EXPANSION_READINESS_AUDIT.json
```

The auditor verifies one hash-bound chain from the frozen development release through all five seeds, calibration-only selection, serialized quantization, one-time internal testing, frozen human adjudication, post-freeze overlap flags, blinded v8 challenge predictions, design-weighted validation, staging evidence, and the manual-review gate. It has no sealed-key argument and does not deploy. Missing work reports `WAITING_FOR_REQUIRED_EXECUTIONS`; invalid or mixed evidence reports `HOLD_INVALID_OR_MIXED_EVIDENCE`.
