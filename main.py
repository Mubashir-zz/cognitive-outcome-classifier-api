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
"""

import json
import os
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ---------------------------------------------------------------------------
# Startup: load the hybrid classifier components once, at server start
# ---------------------------------------------------------------------------

MODEL_REPO = "Mubashir-ZZ/cognitive-classifier-v7-cns"   # v7 CNS BERT model, hosted on HF Hub (private)
CONFIG_PATH = "./hybrid_config.json"
HF_TOKEN = os.environ.get("HF_TOKEN")  # set this in Render's environment variables, never hardcode it

app = FastAPI(
    title="Neurocognitive Outcome Classifier",
    description=(
        "Predicts whether a clinical trial registers a neurocognitive outcome. "
        "AI-ASSISTED SCREENING TOOL -- not a final determination. See /about."
    ),
    version="1.0.0 (v7 CNS model)",
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open(CONFIG_PATH) as f:
    config = json.load(f)
COG_KEYWORDS = config["keywords"]

tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO, token=HF_TOKEN)
bert_model = AutoModelForSequenceClassification.from_pretrained(MODEL_REPO, token=HF_TOKEN)
bert_model.to(device)
bert_model.eval()

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
    "qlq-c30", "qlq c30", "eortc",              # QoL-subscale trap
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


def keyword_rule(text: str) -> bool:
    t = text.lower()
    return any(k in t for k in COG_KEYWORDS)


def bert_predict(text: str) -> float:
    inputs = tokenizer(
        text, truncation=True, padding=True, max_length=512, return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        logits = bert_model(**inputs).logits
    return torch.softmax(logits, dim=1)[0, 1].item()


def check_review_triggers(text: str) -> str | None:
    t = text.lower()
    for pattern in REVIEW_TRIGGER_PATTERNS:
        if pattern in t:
            return f"Text matches a known-difficult pattern ('{pattern.strip()}') -- verify manually."
    return None


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
        return PredictionResponse(
            trial_id=req.trial_id,
            predicted_cognitive=predicted,
            confidence=round(prob, 4),
            method="BERT (v7)",
            review_recommended=review,
            review_reason=reason,
        )
    elif cancer_type in ("Breast", "Lung", "HeadNeck"):
        predicted = keyword_rule(text)
        # keyword rule is highly reliable for these 3 types (validated ~99-100%
        # accuracy), so only flag on a known-trigger pattern match, not routinely
        review = trigger_reason is not None
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


@app.get("/about")
def about():
    return {
        "purpose": "Research screening tool for detecting neurocognitive outcome measurement in oncology clinical trials.",
        "important": "This is AI-assisted screening, NOT a final determination. Predictions with review_recommended=true must be checked by a human before being treated as ground truth.",
        "cns_model": "Fine-tuned Bio_ClinicalBERT (v7), trained on 2,269 hand-verified trials across 4 cancer types.",
        "other_cancer_types": "Keyword-presence rule, validated at ~99-100% accuracy against hand-labeled data.",
        "known_limitations": [
            "A specific QoL-subscale pattern (EORTC QLQ-C30-style multi-subscale mentions) remains unresolved despite five targeted retraining rounds.",
            "Residual confident-but-wrong rate on novel content categories not represented in training, estimated at roughly 1 per 100-250 predictions from audit testing.",
            "Fixes to one failure pattern have, in testing, occasionally caused regressions in unrelated previously-fixed cases -- review flagging is designed around this instability, not just point-in-time accuracy.",
        ],
    }


@app.get("/health")
def health():
    return {"status": "ok", "device": str(device)}
