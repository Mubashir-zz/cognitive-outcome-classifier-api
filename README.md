# Neurocognitive Outcome Classifier -- API

A research screening tool that predicts whether a clinical trial's outcome text registers a genuine neurocognitive assessment. Built on a hybrid design: fine-tuned Bio_ClinicalBERT (v7) for CNS trials, a validated keyword rule for Breast, Lung, and Head & Neck.

**This is AI-assisted screening, not a final determination.** Every response includes a `review_recommended` flag -- predictions flagged this way should be checked by a human before being treated as ground truth.

## Setup

1. Place your v7 CNS model files (from `cns_classifier_v7.zip`) into `./model_cns/`
2. Place `hybrid_config.json` (the keyword list) in the project root -- extract this from the same zip
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Run locally:
   ```
   uvicorn main:app --reload
   ```
5. Test it:
   ```
   curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"cancer_type": "CNS", "outcome_text": "Change from baseline in Hopkins Verbal Learning Test", "trial_id": "TEST001"}'
   ```

## Endpoints

- `POST /predict` -- the main classification endpoint
- `GET /about` -- purpose, model details, and known limitations (always disclosed, not buried)
- `GET /health` -- basic health check

## Deploying for real use: Render (free tier)

Hugging Face Spaces' Docker/Gradio options require a paid plan (confirmed directly on their pricing page) -- Render has a genuine free tier for small Docker-based web services, so that's what this repo is set up for.

1. Log into render.com (GitHub sign-in works, no separate password needed)
2. New -> Web Service -> connect this GitHub repository
3. Render auto-detects the `Dockerfile` in this repo -- no extra configuration needed
4. Model files (`model_cns/` and `hybrid_config.json`, from `cns_classifier_v7.zip`) need to be added to this repo too before deploying, since Render builds directly from the repo contents

The included `Dockerfile` already uses port 8000 with `uvicorn`, matching Render's expected setup.

## Known Limitations (also served live at /about)

- A specific QoL-subscale pattern (EORTC QLQ-C30-style multi-subscale mentions) remains unresolved despite five targeted retraining rounds -- explicitly flagged via `REVIEW_TRIGGER_PATTERNS`.
- Residual confident-but-wrong rate on novel content categories not represented in training, estimated at roughly 1 per 100-250 predictions from audit testing.
- Fixes to one failure pattern have, in testing, occasionally caused regressions in unrelated previously-fixed cases. The review-flagging design assumes ongoing instability, not a fixed, known error set.
