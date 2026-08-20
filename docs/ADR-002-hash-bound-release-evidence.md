# ADR-002: Hash-bound release evidence

**Status:** Accepted for the v8 staging candidate  
**Date:** 2026-08-20  
**Decider:** Mubashir Ahmad Khan

## Context

The release evaluator originally accepted a technical-results JSON containing booleans such as `unit_tests_pass` and `payload_limit_enforced`. That format was easy to read but did not prove which deployment, model artifact, source commit, parity run, or internal-test result produced those values. The scientific gate is only meaningful if technical evidence belongs to the exact v8 artifact being externally validated.

## Decision

Derive technical gate fields from five separate evidence artifacts: deployed staging benchmark, exact analytic/runtime parity, one-time internal-test evaluation, fixed local test execution, and rollback record. The collector cross-checks their SHA-256 chain against the quantization manifest, serialized model, tokenizer, training configuration, development release, selection, seed, and 40-character staging build commit.

The v1.2 evaluator requires the collector version, a passing complete evidence map, no failed checks, and the SHA-256 of the collector implementation. Technical evidence can only permit manual promotion review; neither collector nor evaluator changes production.

## Options considered

### Continue accepting manually assembled booleans

Simple, but it cannot distinguish measured evidence from unsupported assertions and cannot detect artifact mixing.

### One monolithic deployment test

This reduces files but couples local model access, staging credentials, internal-test evidence, and rollback systems. A failure becomes difficult to isolate, and rerunning one check can unintentionally replace another.

### Independent evidence plus hash-bound collector — selected

Each test remains independently reproducible. The collector fails closed on missing records, duplicate regression IDs, source/deployment commit mismatch, runtime-mode drift, or any model/provenance hash mismatch.

## Consequences

- A legacy deployment, different v8 artifact, different source commit, incomplete probe matrix, or hand-written boolean file cannot satisfy the v1.2 path.
- The private regression fixture remains outside evidence outputs; outputs store IDs, hashes, probabilities, deltas, and aggregate results only.
- Real peak memory, cold start, staging capacity, internal-test performance, external human validation, and rollback drill remain required; synthetic smoke evidence cannot promote a model.
- Production v1 remains unchanged until a human reviews a fully passing decision.

## Action items

1. Complete five-seed GPU training and freeze one calibration-selected candidate.
2. Run the exact-artifact internal test once.
3. Deploy the pinned artifact to a separate authenticated staging service.
4. Collect parity, benchmark, fixed-test, and rollback evidence for the same build and manifest.
5. Complete and freeze the 300-record human adjudication, then run controlled unblinding and the v1.2 decision.
