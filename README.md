# Neurocognitive Outcome Classifier -- API

A research screening tool that predicts whether a clinical trial's outcome text registers a genuine neurocognitive assessment. Built on a hybrid design: fine-tuned Bio_ClinicalBERT (v7) for CNS trials, a validated keyword rule for Breast, Lung, and Head & Neck.

**This is AI-assisted screening, not a final determination.** Every response includes a `review_recommended` flag -- predictions flagged this way should be checked by a human before being treated as ground truth.

## Setup

1. The v7 CNS model loads automatically from Hugging Face Hub (`Mubashir-ZZ/cognitive-classifier-v7-cns`, private) -- no local model files needed.
2. Set the `HF_TOKEN` environment variable to a Hugging Face access token with read access to that private repo (Settings -> Access Tokens on huggingface.co). Never commit this token to the repo -- set it as an environment variable in Render's dashboard.
3. `hybrid_config.json` (the keyword list) is already included in this repo.
4. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
5. Run locally (with `HF_TOKEN` set in your environment):
   ```
   uvicorn main:app --reload
   ```
6. Test it:
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
4. In Render's dashboard, add an environment variable: `HF_TOKEN` = your Hugging Face access token (with read access to the private model repo)
5. Deploy -- the model downloads automatically from Hugging Face Hub at startup, no manual file upload needed

The included `Dockerfile` already uses port 8000 with `uvicorn`, matching Render's expected setup.

## Known Limitations (also served live at /about)

- A specific QoL-subscale pattern (EORTC QLQ-C30-style multi-subscale mentions) remains unresolved despite five targeted retraining rounds -- explicitly flagged via `REVIEW_TRIGGER_PATTERNS`.
- Residual confident-but-wrong rate on novel content categories not represented in training, estimated at roughly 1 per 100-250 predictions from audit testing.
- Fixes to one failure pattern have, in testing, occasionally caused regressions in unrelated previously-fixed cases. The review-flagging design assumes ongoing instability, not a fixed, known error set.
