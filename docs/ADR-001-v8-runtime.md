# ADR-001: Opt-in hash-verified v8 runtime

**Status:** Accepted for staging
**Date:** 2026-08-20
**Decider:** Mubashir Ahmad Khan

## Context

The existing service loads the deployed v7 TorchScript file, truncates CNS text at 256 tokens, and labels CNS records with a BERT-or-keyword union. The frozen v8 development protocol instead requires full-text overlapping chunks, maximum-chunk aggregation, a calibration-selected threshold, and v8 as the externally validated primary detector. Production v1 must remain unchanged until every scientific and technical gate passes.

## Decision

Use one staging API with two explicit runtimes:

- `legacy_v7` remains the default and preserves current behavior.
- `v8_chunked` is opt-in. Startup requires the exact quantization-manifest SHA-256 and verifies the serialized model, complete tokenizer directory, inference contract, threshold, selected seed, training configuration, development release, and selection hashes.

V8 scores every accepted CNS token. Its label is primary; keyword evidence remains visible and disagreement requires review but cannot change the v8 label. Inputs beyond the configured token safety ceiling are rejected rather than truncated. The response exposes runtime, full-text status, chunk count, winning token/character offsets, and provenance hashes.

## Options considered

### Replace the legacy loader immediately

| Dimension | Assessment |
|---|---|
| Complexity | Low |
| Scientific risk | High before validation |
| Rollback clarity | Poor |

This would make staging easier to read but could accidentally change live behavior before v8 evidence exists.

### Dual runtime with explicit manifest pin — selected

| Dimension | Assessment |
|---|---|
| Complexity | Moderate |
| Scientific risk | Low |
| Rollback clarity | Strong |

This keeps the current service reproducible while allowing the exact frozen v8 artifact to be tested through the same request/response contract.

### Separate v8 microservice

| Dimension | Assessment |
|---|---|
| Complexity | High |
| Resource cost | High for the current free-tier target |
| Isolation | Strong |

This gives maximal isolation but duplicates authentication, review routing, monitoring, and deployment configuration.

## Consequences

- A missing or altered v8 artifact stops startup; there is no silent fallback to v7.
- V8 analytic-to-API parity can use exact artifact and tokenizer hashes.
- The API remains backward-compatible in legacy mode, while v8 adds nullable provenance/chunk fields.
- Runtime and release-gate evidence are still required before manual production promotion.
- Performance and memory must be measured again with the real trained artifact; the synthetic smoke test proves contract execution, not model quality or production capacity.
