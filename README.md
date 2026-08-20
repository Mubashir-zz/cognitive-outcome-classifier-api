# Neurocognitive outcome classifier — v2 release candidate 2

This folder is a staging candidate, not the production service.

It preserves the v7 quantized CNS detector and the 66-term keyword detector while making each detector's evidence explicit. A response reports the BERT probability, binary BERT result, keyword hit and character offsets, decision basis, review reasons, source-text hash, truncation status, artifact hashes, and build commit. It does not call a keyword result a probability or send raw outcome text to the review webhook.

Production promotion is intentionally blocked until the disjoint `CNS_challenge_set_300_BLINDED` sample receives frozen independent human labels and the release candidate passes the prespecified scientific, analytic-parity, memory, security, and regression gates.

The 0.5 BERT threshold, uncertainty zone, token limit, keyword list, and review-trigger list are all versioned in `hybrid_config.json`. Long CNS inputs disclose truncation and are routed to review rather than silently treated as complete model reads.

## Local model test

Set `MODEL_DIR` to a directory containing the tokenizer files and `cns_v7_quantized.pt`, then run one worker:

```bash
CONFIG_PATH=./hybrid_config.json MODEL_DIR=/path/to/model uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers 1
```

## Required staging secrets

- `HF_TOKEN` when loading the private model from Hugging Face
- `CLASSIFIER_API_KEY` to require `X-API-Key` on prediction routes
- `REVIEW_QUEUE_WEBHOOK` only if the review queue accepts metadata without raw outcome text

## Gates before deployment

1. Blinded human adjudication completed, frozen, and analyzed with the challenge-set design weights.
2. Candidate keyword output exactly reproduces the frozen analytic rule on matched full text, or a new analytic release is generated.
3. Unit, API-contract, regression, cold-start, peak-memory, and concurrency tests pass.
4. A separate staging service passes shadow comparison against production.
5. Rollback image and immutable hashes are recorded.

Passing every gate allows manual promotion review; it never deploys automatically.
