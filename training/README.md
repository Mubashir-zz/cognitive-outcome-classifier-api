# Frozen v8 model-development pipeline

This directory contains the reproducible code for the next CNS classifier candidate. It is separate from the live service and does not authorize deployment.

The pipeline:

1. verifies a hash-pinned, challenge-text-disjoint development release;
2. trains five prespecified Bio_ClinicalBERT seeds with full-text overlapping chunks and trial-balanced loss;
3. selects one seed and threshold from CNS calibration records only;
4. hash-verifies the selected model, quantizes it, and requires parity from the reloaded serialized artifact;
5. evaluates that exact model and tokenizer on the internal test once, with a permanent lock;
6. after human labels are frozen, scores the original blinded challenge text without reading human truth or the sealed key;
7. evaluates design-weighted v8 performance under conjunctive scientific and technical release gates.

Training rows and challenge files are intentionally excluded from GitHub. The private development release contains 2,247 unique trial IDs and normalized texts: 1,573 training, 337 calibration, and 337 internal test. Nine source rows matching three challenge-text hashes were excluded before splitting.

## Development sequence

```bash
python training/train_v8_chunked_bert.py \
  --release /private/path/v8_development_release.csv \
  --config training/v8_training_config.json \
  --output-dir /private/path/v8_candidates --resume

python training/select_v8_candidate.py \
  --candidates-dir /private/path/v8_candidates \
  --config training/v8_training_config.json \
  --output /private/path/v8_candidates/v8_candidate_selection.json

python training/quantize_v8_model.py \
  --release /private/path/v8_development_release.csv \
  --config training/v8_training_config.json \
  --selection /private/path/v8_candidates/v8_candidate_selection.json \
  --output-dir /private/path/v8_candidates/v8_quantized_selected
```

The internal test requires the literal acknowledgement `EVALUATE_FROZEN_SELECTION_ONCE`. External scoring and controlled unblinding are separate, post-freeze stages documented in `MODEL_DEVELOPMENT_PROTOCOL_v2.md`. Production must not be changed from this branch.

## What is not claimed

- The recovered float v7 directory has not been proven to be the exact parent of the live quantized artifact.
- A runtime smoke test is not candidate training.
- Internal-test performance is not external validation.
- The challenge sample is deliberately enriched; unweighted performance is not population performance.
- Passing model metrics alone does not satisfy API privacy, parity, memory, cold-start, authentication, or rollback gates.
