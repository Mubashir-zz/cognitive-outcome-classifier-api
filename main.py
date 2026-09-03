"""
Neurocognitive Outcome Classifier API
--------------------------------------
Research screening tool: predicts whether a clinical trial's outcome text
registers a genuine neurocognitive assessment.

IMPORTANT: This is an AI-assisted screening tool, not a final determination.
Every prediction includes a confidence score and a review_recommended flag.
Predictions flagged for review should be checked by a human before being
treated as ground truth -- this is a core design requirement, not optional,
based on the documented residual error patterns found during development
(see MODEL_CARD.md).

NOTE ON THE MODEL FILE: this loads a quantized, TorchScript-traced version
of the v7 CNS model (169MB vs. the original ~433MB), specifically to fit
Render's free-tier 512MB RAM limit. It was validated against the same known
test cases used throughout development before being deployed.

All decision rules live in classifier.py so they can be tested without a
model file or an HF token; this module handles model loading, HTTP and I/O.
"""

import os

import requests
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download

from classifier import (
    KEYWORD_ROUTE_TYPES,
    SUPPORTED_CANCER_TYPES,
    cns_decision,
    keyword_decision,
    load_keywords,
)

MODEL_REPO = "Mubashir-ZZ/cognitive-classifier-v7-cns"   # v7 CNS BERT model on HF Hub (private)
QUANTIZED_MODEL_FILENAME = "cns_v7_quantized.pt"
HF_TOKEN = os.environ.get("HF_TOKEN")  # set in Render's environment variables, never hardcoded
REVIEW_QUEUE_WEBHOOK = os.environ.get("REVIEW_QUEUE_WEBHOOK")  # Google Apps Script web app URL

app = FastAPI(
    title="Neurocognitive Outcome Classifier",
    description=(
        "Predicts whether a clinical trial registers a neurocognitive outcome. "
        "AI-ASSISTED SCREENING TOOL -- not a final determination. See /about."
    ),
    version="1.0.0 (v7 CNS model, quantized)",
)

device = torch.device("cpu")  # free-tier instance has no GPU
COG_KEYWORDS = load_keywords()

tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO, token=HF_TOKEN)

# Download just the quantized model file (169MB) rather than going through the
# full transformers model-loading path -- smaller, and it never touches the
# original ~433MB float32 weights at all.
model_path = hf_hub_download(repo_id=MODEL_REPO, filename=QUANTIZED_MODEL_FILENAME, token=HF_TOKEN)
traced_model = torch.jit.load(model_path, map_location=device)
traced_model.eval()

MAX_BATCH_SIZE = 6  # reduced from 32 after Render logs confirmed batches of 25 caused repeated
                    # out-of-memory crashes; 6 is a conservative, verified-safe size on 512MB


class PredictionRequest(BaseModel):
    cancer_type: str = Field(..., description="One of: CNS, Breast, Lung, HeadNeck")
    outcome_text: str = Field(..., description="Full outcome-measure text from the trial registry")
    trial_id: str | None = Field(None, description="Optional NCT/TrialID for reference")


class PredictionResponse(BaseModel):
    trial_id: str | None
    predicted_cognitive: bool
    confidence: float
    method: str
    review_recommended: bool
    review_reason: str | None


class BatchPredictionRequest(BaseModel):
    items: list[PredictionRequest] = Field(..., max_length=MAX_BATCH_SIZE)


class BatchPredictionResponse(BaseModel):
    results: list[PredictionResponse]


def _logits(out):
    """The traced model's output type varies (dict/tuple/object) with export
    details -- handle all three, as validated during deployment testing."""
    if isinstance(out, dict):
        return out["logits"]
    if isinstance(out, tuple):
        return out[0]
    return out.logits


def bert_predict(text: str) -> float:
    inputs = tokenizer(
        text, truncation=True, padding=True, max_length=256, return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        logits = _logits(traced_model(inputs["input_ids"], inputs["attention_mask"]))
    return torch.softmax(logits, dim=1)[0, 1].item()


def bert_predict_batch(texts: list[str]) -> list[float]:
    """One forward pass for the whole list rather than one call per text.
    Substantially faster for bulk scoring, at the cost of a padded tensor sized
    to the longest sequence -- bounded by MAX_BATCH_SIZE to keep memory
    predictable on the free tier."""
    inputs = tokenizer(
        texts, truncation=True, padding=True, max_length=256, return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        logits = _logits(traced_model(inputs["input_ids"], inputs["attention_mask"]))
    return torch.softmax(logits, dim=1)[:, 1].tolist()


def log_to_review_queue(trial_id, cancer_type, outcome_text, predicted, confidence, reason):
    """Best-effort log to the review queue. A logging failure must never break
    the prediction response the caller is waiting on."""
    if not REVIEW_QUEUE_WEBHOOK:
        return
    try:
        requests.post(
            REVIEW_QUEUE_WEBHOOK,
            json={
                "trial_id": trial_id,
                "cancer_type": cancer_type,
                "outcome_text": outcome_text,
                "predicted_cognitive": predicted,
                "confidence": confidence,
                "review_reason": reason,
            },
            timeout=5,
        )
    except Exception:
        pass


def _require_cancer_type(cancer_type: str) -> str:
    normalised = cancer_type.strip()
    if normalised not in SUPPORTED_CANCER_TYPES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unknown cancer_type '{cancer_type}'. "
                f"Must be one of: {', '.join(SUPPORTED_CANCER_TYPES)}"
            ),
        )
    return normalised


def _require_text(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        raise HTTPException(status_code=400, detail="outcome_text cannot be empty")
    return stripped


def _response(trial_id, cancer_type, text, predicted, confidence, method, review, reason):
    if review:
        log_to_review_queue(trial_id, cancer_type, text, predicted, confidence, reason)
    return PredictionResponse(
        trial_id=trial_id,
        predicted_cognitive=predicted,
        confidence=confidence,
        method=method,
        review_recommended=review,
        review_reason=reason,
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(req: PredictionRequest):
    cancer_type = _require_cancer_type(req.cancer_type)
    text = _require_text(req.outcome_text)

    if cancer_type == "CNS":
        predicted, confidence, review, reason = cns_decision(bert_predict(text), text)
        method = "BERT (v7, quantized)"
    else:
        predicted, confidence, review, reason = keyword_decision(text, COG_KEYWORDS)
        method = "keyword_rule"

    return _response(req.trial_id, cancer_type, text, predicted, confidence, method, review, reason)


@app.post("/predict_batch", response_model=BatchPredictionResponse)
def predict_batch(req: BatchPredictionRequest):
    items = req.items
    if not items:
        raise HTTPException(status_code=400, detail="items cannot be empty")

    # Validate everything up front so a bad item never leaves a half-scored batch.
    prepared = [(_require_cancer_type(i.cancer_type), _require_text(i.outcome_text)) for i in items]

    results: list[PredictionResponse | None] = [None] * len(items)

    # CNS items share a single batched forward pass.
    cns_indices = [i for i, (ct, _) in enumerate(prepared) if ct == "CNS"]
    if cns_indices:
        probs = bert_predict_batch([prepared[i][1] for i in cns_indices])
        for idx, prob in zip(cns_indices, probs):
            text = prepared[idx][1]
            predicted, confidence, review, reason = cns_decision(prob, text)
            results[idx] = _response(
                items[idx].trial_id, "CNS", text, predicted, confidence,
                "BERT (v7, quantized)", review, reason,
            )

    # Keyword-route items are already fast enough not to need batching.
    for idx, (cancer_type, text) in enumerate(prepared):
        if cancer_type == "CNS":
            continue
        predicted, confidence, review, reason = keyword_decision(text, COG_KEYWORDS)
        results[idx] = _response(
            items[idx].trial_id, cancer_type, text, predicted, confidence,
            "keyword_rule", review, reason,
        )

    return BatchPredictionResponse(results=results)


@app.get("/about")
def about():
    return {
        "purpose": "Research screening tool for detecting neurocognitive outcome measurement in oncology clinical trials.",
        "important": "This is AI-assisted screening, NOT a final determination. Predictions with review_recommended=true must be checked by a human before being treated as ground truth.",
        "cns_model": "Fine-tuned Bio_ClinicalBERT (v7), quantized to int8 and TorchScript-traced for deployment, trained on 2,269 hand-verified trials across 4 cancer types.",
        "other_cancer_types": "Keyword-presence rule, validated at ~99-100% accuracy against hand-labeled data.",
        "keyword_route_applies_to": list(KEYWORD_ROUTE_TYPES),
        "known_limitations": [
            "A specific QoL-subscale pattern (EORTC QLQ-C30-style multi-subscale mentions) remains unresolved despite five targeted retraining rounds.",
            "Residual confident-but-wrong rate on novel content categories not represented in training, estimated at roughly 1 per 100-250 predictions from audit testing.",
            "Fixes to one failure pattern have, in testing, occasionally caused regressions in unrelated previously-fixed cases -- review flagging is designed around this instability, not just point-in-time accuracy.",
            "The deployed model is quantized (int8) for memory efficiency; validated against known test cases post-quantization with no new failure patterns introduced.",
        ],
    }


@app.get("/health")
def health():
    return {"status": "ok", "device": str(device)}
