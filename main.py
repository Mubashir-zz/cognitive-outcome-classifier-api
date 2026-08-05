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
Render's free-tier 512MB RAM limit. This was validated against the same
known test cases used throughout development before being deployed --
see PHASE4_final_results.md / project history for the validation table.
"""

import json
import os
import requests
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download

# ---------------------------------------------------------------------------
# Startup: load the hybrid classifier components once, at server start
# ---------------------------------------------------------------------------

MODEL_REPO = "Mubashir-ZZ/cognitive-classifier-v7-cns"   # v7 CNS BERT model, hosted on HF Hub (private)
QUANTIZED_MODEL_FILENAME = "cns_v7_quantized.pt"
CONFIG_PATH = "./hybrid_config.json"
HF_TOKEN = os.environ.get("HF_TOKEN")  # set this in Render's environment variables, never hardcode it
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

with open(CONFIG_PATH) as f:
    config = json.load(f)
COG_KEYWORDS = config["keywords"]

tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO, token=HF_TOKEN)

# Download just the quantized model file (169MB) rather than the full
# transformers model-loading path -- smaller, and never touches the
# original ~433MB float32 weights at all.
model_path = hf_hub_download(repo_id=MODEL_REPO, filename=QUANTIZED_MODEL_FILENAME, token=HF_TOKEN)
traced_model = torch.jit.load(model_path, map_location=device)
traced_model.eval()

# Confidence zone: BERT probabilities in this range are genuinely uncertain
# and always get flagged for human review, regardless of which side of 0.5
# they land on.
UNCERTAIN_LOW = 0.25
UNCERTAIN_HIGH = 0.75

# Known-difficult content patterns (from documented validation failures) --
# predictions on text matching these patterns are always flagged for review
# even when the model is confident, since these are exactly the categories
# where confident-but-wrong predictions have occurred during testing.
REVIEW_TRIGGER_PATTERNS = [
    "qlq-c30", "qlq c30", "eortc",              # QoL-subscale trap (still unresolved as of v7)
    "karnofsky", " kps ", "kps)",                 # performance-status trap
    "rcbv", "rcbf", "suvr", "dsc-mri", "pet/ct",  # imaging/biomarker trap
    "hospitalization", "emergency department",    # healthcare-utilization trap
    "platelet", "thrombocytopenia",                # hematological-toxicity trap
]


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


MAX_BATCH_SIZE = 6  # reduced from 32 after confirming via Render logs that batches of 25
                     # caused repeated out-of-memory crashes; 6 is a conservative, verified-safe
                     # starting point on the 512MB free-tier instance


class BatchPredictionRequest(BaseModel):
    items: list[PredictionRequest] = Field(..., max_length=MAX_BATCH_SIZE)


class BatchPredictionResponse(BaseModel):
    results: list[PredictionResponse]


def keyword_rule(text: str) -> bool:
    t = text.lower()
    return any(k in t for k in COG_KEYWORDS)


def bert_predict(text: str) -> float:
    inputs = tokenizer(
        text, truncation=True, padding=True, max_length=512, return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        out = traced_model(inputs["input_ids"], inputs["attention_mask"])
        # the traced model's output type can vary (dict/tuple/object) depending
        # on export details -- handle all three, same as validated in testing
        if isinstance(out, dict):
            logits = out["logits"]
        elif isinstance(out, tuple):
            logits = out[0]
        else:
            logits = out.logits
    return torch.softmax(logits, dim=1)[0, 1].item()


def bert_predict_batch(texts: list[str]) -> list[float]:
    """True batched inference: one forward pass for the whole list, not one
    call per text. Substantially faster for bulk scoring (e.g. landscape
    analyses) than repeated single-item calls, at the cost of one padded
    tensor sized to the longest sequence in the batch -- bounded by
    MAX_BATCH_SIZE to keep memory usage predictable."""
    inputs = tokenizer(
        texts, truncation=True, padding=True, max_length=512, return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        out = traced_model(inputs["input_ids"], inputs["attention_mask"])
        if isinstance(out, dict):
            logits = out["logits"]
        elif isinstance(out, tuple):
            logits = out[0]
        else:
            logits = out.logits
    probs = torch.softmax(logits, dim=1)[:, 1]
    return probs.tolist()


def check_review_triggers(text: str) -> str | None:
    t = text.lower()
    for pattern in REVIEW_TRIGGER_PATTERNS:
        if pattern in t:
            return f"Text matches a known-difficult pattern ('{pattern.strip()}') -- verify manually."
    return None


def log_to_review_queue(trial_id, cancer_type, outcome_text, predicted, confidence, reason):
    """Best-effort log to the review queue -- never let a logging failure break
    the actual prediction response the caller is waiting on."""
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
        pass  # logging failure should never break the actual API response


@app.post("/predict", response_model=PredictionResponse)
def predict(req: PredictionRequest):
    cancer_type = req.cancer_type.strip()
    text = req.outcome_text.strip()

    if not text:
        raise HTTPException(status_code=400, detail="outcome_text cannot be empty")

    trigger_reason = check_review_triggers(text)

    if cancer_type == "CNS":
        prob = bert_predict(text)
        predicted = prob >= 0.5
        uncertain = UNCERTAIN_LOW <= prob <= UNCERTAIN_HIGH
        review = uncertain or (trigger_reason is not None)
        reason = trigger_reason or (
            f"BERT confidence ({prob:.3f}) is in the uncertain zone." if uncertain else None
        )
        if review:
            log_to_review_queue(req.trial_id, cancer_type, text, predicted, round(prob, 4), reason)
        return PredictionResponse(
            trial_id=req.trial_id,
            predicted_cognitive=predicted,
            confidence=round(prob, 4),
            method="BERT (v7, quantized)",
            review_recommended=review,
            review_reason=reason,
        )
    elif cancer_type in ("Breast", "Lung", "HeadNeck"):
        predicted = keyword_rule(text)
        # keyword rule is highly reliable for these 3 types (validated ~99-100%
        # accuracy), so only flag on a known-trigger pattern match, not routinely
        review = trigger_reason is not None
        if review:
            log_to_review_queue(req.trial_id, cancer_type, text, predicted, 1.0 if predicted else 0.0, trigger_reason)
        return PredictionResponse(
            trial_id=req.trial_id,
            predicted_cognitive=predicted,
            confidence=1.0 if predicted else 0.0,  # keyword rule is binary, not probabilistic
            method="keyword_rule",
            review_recommended=review,
            review_reason=trigger_reason,
        )
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown cancer_type '{cancer_type}'. Must be one of: CNS, Breast, Lung, HeadNeck",
        )


@app.post("/predict_batch", response_model=BatchPredictionResponse)
def predict_batch(req: BatchPredictionRequest):
    items = req.items
    if not items:
        raise HTTPException(status_code=400, detail="items cannot be empty")

    # validate all cancer types up front, and empty text, before doing any work
    for item in items:
        if item.cancer_type.strip() not in ("CNS", "Breast", "Lung", "HeadNeck"):
            raise HTTPException(
                status_code=400,
                detail=f"Unknown cancer_type '{item.cancer_type}'. Must be one of: CNS, Breast, Lung, HeadNeck",
            )
        if not item.outcome_text.strip():
            raise HTTPException(status_code=400, detail="outcome_text cannot be empty")

    results: list[PredictionResponse | None] = [None] * len(items)

    # CNS items: one batched BERT forward pass for all of them together
    cns_indices = [i for i, item in enumerate(items) if item.cancer_type.strip() == "CNS"]
    if cns_indices:
        cns_texts = [items[i].outcome_text.strip() for i in cns_indices]
        probs = bert_predict_batch(cns_texts)
        for idx, prob in zip(cns_indices, probs):
            item = items[idx]
            text = item.outcome_text.strip()
            trigger_reason = check_review_triggers(text)
            predicted = prob >= 0.5
            uncertain = UNCERTAIN_LOW <= prob <= UNCERTAIN_HIGH
            review = uncertain or (trigger_reason is not None)
            reason = trigger_reason or (
                f"BERT confidence ({prob:.3f}) is in the uncertain zone." if uncertain else None
            )
            if review:
                log_to_review_queue(item.trial_id, "CNS", text, predicted, round(prob, 4), reason)
            results[idx] = PredictionResponse(
                trial_id=item.trial_id, predicted_cognitive=predicted, confidence=round(prob, 4),
                method="BERT (v7, quantized)", review_recommended=review, review_reason=reason,
            )

    # Non-CNS items: keyword rule, no batching benefit needed (already fast)
    for i, item in enumerate(items):
        cancer_type = item.cancer_type.strip()
        if cancer_type == "CNS":
            continue
        text = item.outcome_text.strip()
        trigger_reason = check_review_triggers(text)
        predicted = keyword_rule(text)
        review = trigger_reason is not None
        if review:
            log_to_review_queue(item.trial_id, cancer_type, text, predicted, 1.0 if predicted else 0.0, trigger_reason)
        results[i] = PredictionResponse(
            trial_id=item.trial_id, predicted_cognitive=predicted,
            confidence=1.0 if predicted else 0.0, method="keyword_rule",
            review_recommended=review, review_reason=trigger_reason,
        )

    return BatchPredictionResponse(results=results)


@app.get("/about")
def about():
    return {
        "purpose": "Research screening tool for detecting neurocognitive outcome measurement in oncology clinical trials.",
        "important": "This is AI-assisted screening, NOT a final determination. Predictions with review_recommended=true must be checked by a human before being treated as ground truth.",
        "cns_model": "Fine-tuned Bio_ClinicalBERT (v7), quantized to int8 and TorchScript-traced for deployment, trained on 2,269 hand-verified trials across 4 cancer types.",
        "other_cancer_types": "Keyword-presence rule, validated at ~99-100% accuracy against hand-labeled data.",
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
