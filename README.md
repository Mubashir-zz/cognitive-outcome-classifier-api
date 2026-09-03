# Neurocognitive Outcome Classifier — API

Predicts whether a clinical trial's registered outcome text contains a genuine
neurocognitive assessment. Built as the serving layer for a study measuring how
often oncology trials actually measure cognition.

**Live:** https://cognitive-outcome-classifier-api.onrender.com/about

It runs on Render's free tier, so the first request after a quiet period takes
30–50 seconds to wake the instance. After that it responds in well under a second.

```bash
curl -X POST https://cognitive-outcome-classifier-api.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{"cancer_type":"CNS","trial_id":"NCT00000000",
       "outcome_text":"Change from baseline in Hopkins Verbal Learning Test - Revised at 6 months"}'
```

```json
{
  "trial_id": "NCT00000000",
  "predicted_cognitive": true,
  "confidence": 0.9965,
  "method": "BERT (v7, quantized)",
  "review_recommended": false,
  "review_reason": null
}
```

## Why this is a hybrid and not one model

Cognitive outcomes are easy to spot in breast, lung and head & neck trials —
the instruments are standardised, and a keyword rule hits 99–100% on held-out
data. CNS trials are the hard case: the vocabulary is heterogeneous and full of
things that look like cognition and are not. The NANO neurologic exam. Karnofsky
performance status. A quality-of-life questionnaire's cognitive subscale.

So the routing follows the evidence rather than a preference:

| Cancer type | Route |
|---|---|
| CNS | Fine-tuned Bio_ClinicalBERT (v7), int8-quantized, TorchScript-traced |
| Breast, Lung, HeadNeck | Validated 66-term keyword rule |

A transformer for the other three types was tested and did not beat the keyword
baseline, so it is not used for them. The comparison is in
[`PHASE2_RESULTS.md`](https://github.com/Mubashir-zz/neurocognitive-outcome-classifier/blob/main/results/PHASE2_RESULTS.md).

## Review flagging

The model is a screening aid and is designed around being wrong sometimes. A
prediction is returned with `review_recommended: true` when either:

- the BERT probability lands between 0.25 and 0.75, or
- the text matches a pattern that produced confident-but-wrong predictions
  during validation (`EORTC QLQ-C30`, `Karnofsky`, imaging biomarkers,
  hospitalisation, haematological toxicity)

The second case fires *even when the model is confident*, because on those
categories confidence carries no information. Flagged predictions are optionally
posted to a review queue via `REVIEW_QUEUE_WEBHOOK`; that call is best-effort
and never blocks the response.

Full error analysis in [MODEL_CARD.md](MODEL_CARD.md).

## Endpoints

| Method | Path | Notes |
|---|---|---|
| `POST` | `/predict` | Single trial |
| `POST` | `/predict_batch` | Up to 6 trials, one batched forward pass for the CNS items |
| `GET` | `/about` | Model lineage and known limitations, served live rather than buried in docs |
| `GET` | `/health` | Liveness + device |

`cancer_type` must be one of `CNS`, `Breast`, `Lung`, `HeadNeck`.

The batch cap of 6 is not arbitrary. Batches of 25 caused repeated
out-of-memory kills on Render's 512MB tier; 6 is the size that held up under
load. The same constraint is why the served model is int8-quantized to 169MB
instead of the 433MB float32 checkpoint.

## Scoring a file of trials

```bash
python examples/score_trials.py trials.csv --out scored.csv
```

Input needs `trial_id`, `cancer_type`, `outcome_text`. Output adds the
prediction, confidence, route used, and the review flag with its reason.

## Running locally

The CNS model lives in a private Hugging Face repo, so serving needs a token
with read access to it:

```bash
pip install -r requirements.txt
export HF_TOKEN=hf_...
uvicorn main:app --reload
```

Docker:

```bash
docker build -t cognitive-classifier .
docker run -p 8000:8000 -e HF_TOKEN=hf_... cognitive-classifier
```

The Dockerfile installs CPU-only PyTorch from a separate index. The default
PyPI wheel pulls ~900MB of CUDA libraries that are never used here, and that
alone was enough to blow the memory limit.

## Evaluation against the live endpoint

196 trials scored through the deployed API on 3 September 2026, with outcome text
pulled from the ClinicalTrials.gov v2 API rather than from the stored dataset
(the stored column is truncated at ~400 characters and drops the instrument):

| | n | Agreement | Sensitivity | Specificity | Flagged |
|---|---|---|---|---|---|
| Overall | 196 | 95.9% | 93.0% | 99.0% | 32.1% |
| CNS *(BERT)* | 49 | 83.7% | 72.0% | 95.8% | 36.7% |
| Breast, Lung, Head & Neck *(keyword)* | 147 | 100% | 100% | 100% | 30.6% |

The keyword rule made no errors in either direction on 147 trials, which is the
clearest case for not routing those types through a model. CNS is the hard case
and errs toward missing cognitive outcomes rather than inventing them. Four of the
eight CNS errors were caught by the review flag — half, not all.

Method, per-trial results and the full error analysis: [`validation/`](validation/).

## Tests

```bash
pip install pytest
pytest
```

All decision rules live in `classifier.py`, which imports nothing heavier than
`json`. That means the logic — thresholds, the uncertain band, every review
trigger, the keyword rule — is tested with no model file, no token and no
network, and CI runs it on every push. `main.py` handles model loading and HTTP
and nothing else.

## Layout

```
classifier.py        decision rules, dependency-free
main.py              FastAPI app, model loading, batching
hybrid_config.json   the 66-term validated keyword list
tests/               41 tests over the decision logic
examples/            batch scoring client
Dockerfile           CPU-only build for the 512MB tier
MODEL_CARD.md        training data, metrics, failure modes
```

## Scope

Research screening — narrowing a registry pull to a reviewable candidate set.
Not for clinical decision-making, and not a substitute for reading the trial
record.

## Related

- [neurocognitive-outcome-classifier](https://github.com/Mubashir-zz/neurocognitive-outcome-classifier) — training data, model development, R and Python

## Author

Mubashir Ahmad Khan
