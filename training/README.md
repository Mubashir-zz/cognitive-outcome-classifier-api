# Frozen v8 model-development pipeline

This directory contains the reproducible code for the next CNS classifier candidate. It is separate from the live v1 service and does not authorize deployment.

The pipeline:

1. verifies a hash-pinned, challenge-text-disjoint development release;
2. trains five prespecified Bio_ClinicalBERT seeds with full-text overlapping chunks;
3. selects one seed and threshold from CNS calibration records only;
4. quantizes only the selected candidate and requires parity from the final serialized artifact;
5. evaluates the frozen internal test once, with an explicit acknowledgement and a permanent selection-level lock;
6. leaves the independent 300-record challenge set untouched until human labels are frozen.

Training data and challenge files are intentionally excluded from GitHub. The local development release has 2,247 unique trial IDs and unique normalized texts, split into 1,573 training, 337 calibration, and 337 internal-test records. Nine source rows matching three challenge text hashes were excluded before splitting.

## Sequence

```bash
python training/train_v8_chunked_bert.py \
  --release /private/path/v8_development_release.csv \
  --config training/v8_training_config.json \
  --output-dir /private/path/v8_candidates

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

The internal-test command is deliberately separate and requires the literal acknowledgement `EVALUATE_FROZEN_SELECTION_ONCE`. External challenge validation and release-gate evaluation remain mandatory after independent adjudication. Production must not be changed from this branch.

## What is not claimed

- The existing float v7 directory has not been proven to be the exact parent of the live quantized artifact.
- A runtime smoke test is not candidate training.
- Internal-test performance is not external validation.
- The external challenge set is deliberately enriched; unweighted performance is not population performance.
- Passing model metrics alone does not satisfy API privacy, parity, container-memory, cold-start, or rollback gates.

