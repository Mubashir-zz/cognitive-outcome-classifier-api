"""
Decision logic for the hybrid neurocognitive-outcome classifier.

This module holds every rule that decides an outcome, deliberately kept free
of torch, transformers and FastAPI imports so the logic can be tested without
a model file, a GPU or network access. `main.py` supplies the BERT probability
for CNS trials; everything else lives here.

The two routes:
  * CNS          -> fine-tuned Bio_ClinicalBERT probability, thresholded at 0.5
  * Breast/Lung/HeadNeck -> keyword-presence rule over hybrid_config.json

Both routes go through the same review-flagging layer, which is the part that
matters clinically: the model is a screening aid, so anything landing in the
uncertain band or matching a documented failure pattern gets handed back to a
human rather than silently accepted.
"""

import json
from pathlib import Path

DEFAULT_CONFIG_PATH = Path(__file__).with_name("hybrid_config.json")

SUPPORTED_CANCER_TYPES = ("CNS", "Breast", "Lung", "HeadNeck")
KEYWORD_ROUTE_TYPES = ("Breast", "Lung", "HeadNeck")

# BERT probabilities inside this band are genuinely uncertain and are always
# flagged, whichever side of 0.5 they fall on.
UNCERTAIN_LOW = 0.25
UNCERTAIN_HIGH = 0.75

DECISION_THRESHOLD = 0.5

# Content patterns that produced confident-but-wrong predictions during
# validation. Text matching any of these is flagged for review even when the
# model is confident, because confidence is not informative on these categories.
REVIEW_TRIGGER_PATTERNS = [
    "qlq-c30", "qlq c30", "eortc",                 # QoL-subscale trap, unresolved as of v7
    "karnofsky", " kps ", "kps)",                  # performance-status trap
    "rcbv", "rcbf", "suvr", "dsc-mri", "pet/ct",   # imaging / biomarker trap
    "hospitalization", "emergency department",     # healthcare-utilization trap
    "platelet", "thrombocytopenia",                # haematological-toxicity trap
]


def load_keywords(config_path=DEFAULT_CONFIG_PATH):
    """Read the validated cognitive-instrument keyword list."""
    with open(config_path) as f:
        return json.load(f)["keywords"]


def keyword_rule(text, keywords):
    """True when any validated cognitive-instrument term appears in the text."""
    lowered = text.lower()
    return any(k in lowered for k in keywords)


def check_review_triggers(text):
    """Return a human-readable reason if the text hits a known failure pattern."""
    lowered = text.lower()
    for pattern in REVIEW_TRIGGER_PATTERNS:
        if pattern in lowered:
            return (
                f"Text matches a known-difficult pattern ('{pattern.strip()}') "
                "-- verify manually."
            )
    return None


def is_uncertain(prob):
    return UNCERTAIN_LOW <= prob <= UNCERTAIN_HIGH


def cns_decision(prob, text):
    """Apply the CNS route to an already-computed BERT probability.

    Returns (predicted_cognitive, confidence, review_recommended, review_reason).
    """
    trigger_reason = check_review_triggers(text)
    predicted = prob >= DECISION_THRESHOLD
    uncertain = is_uncertain(prob)
    review = uncertain or trigger_reason is not None
    reason = trigger_reason or (
        f"BERT confidence ({prob:.3f}) is in the uncertain zone." if uncertain else None
    )
    return predicted, round(prob, 4), review, reason


def keyword_decision(text, keywords):
    """Apply the keyword route used for Breast, Lung and Head & Neck.

    Returns (predicted_cognitive, confidence, review_recommended, review_reason).
    Confidence is binary here because the rule is binary -- reporting a
    probability would imply a calibration the rule does not have.
    """
    trigger_reason = check_review_triggers(text)
    predicted = keyword_rule(text, keywords)
    review = trigger_reason is not None
    return predicted, 1.0 if predicted else 0.0, review, trigger_reason


def validate_cancer_type(cancer_type):
    """Return the normalised cancer type, or raise ValueError."""
    normalised = cancer_type.strip()
    if normalised not in SUPPORTED_CANCER_TYPES:
        raise ValueError(
            f"Unknown cancer_type '{cancer_type}'. "
            f"Must be one of: {', '.join(SUPPORTED_CANCER_TYPES)}"
        )
    return normalised
