"""
Tests for the decision logic. No model, no network, no HF token needed --
that is the point of keeping classifier.py free of torch and FastAPI imports.
"""

import pytest

from classifier import (
    DECISION_THRESHOLD,
    KEYWORD_ROUTE_TYPES,
    UNCERTAIN_HIGH,
    UNCERTAIN_LOW,
    check_review_triggers,
    cns_decision,
    is_uncertain,
    keyword_decision,
    keyword_rule,
    load_keywords,
    validate_cancer_type,
)

KEYWORDS = load_keywords()


# --------------------------------------------------------------------------
# Keyword rule
# --------------------------------------------------------------------------

@pytest.mark.parametrize("text", [
    "Change from baseline in Hopkins Verbal Learning Test - Revised (HVLT-R)",
    "Mini-Mental State Examination (MMSE) score at 6 months",
    "Montreal Cognitive Assessment administered at each visit",
    "Trail Making Test A and B completion time",
    "FACT-Cog perceived cognitive impairment subscale",
    "Neurocognitive function measured by a standard battery",
])
def test_keyword_rule_detects_real_instruments(text):
    assert keyword_rule(text, KEYWORDS) is True


@pytest.mark.parametrize("text", [
    "Overall survival at 24 months",
    "Progression-free survival per RANO criteria",
    "Incidence of grade 3 or higher adverse events",
    "Objective response rate by RECIST 1.1",
])
def test_keyword_rule_ignores_non_cognitive_outcomes(text):
    assert keyword_rule(text, KEYWORDS) is False


def test_keyword_rule_is_case_insensitive():
    assert keyword_rule("MOCA SCORE", KEYWORDS) is True
    assert keyword_rule("moca score", KEYWORDS) is True


# --------------------------------------------------------------------------
# Review triggers -- the traps that caused confident-but-wrong predictions
# --------------------------------------------------------------------------

@pytest.mark.parametrize("text", [
    "EORTC QLQ-C30 cognitive functioning subscale",
    "Karnofsky Performance Status at 12 weeks",
    "Change in rCBV on DSC-MRI",
    "Rate of hospitalization during treatment",
    "Incidence of thrombocytopenia",
])
def test_known_traps_are_flagged(text):
    assert check_review_triggers(text) is not None


def test_clean_cognitive_text_is_not_flagged():
    assert check_review_triggers("HVLT-R total recall at 6 months") is None


# --------------------------------------------------------------------------
# CNS route
# --------------------------------------------------------------------------

def test_cns_confident_positive_is_accepted_without_review():
    predicted, confidence, review, reason = cns_decision(0.996, "HVLT-R at 6 months")
    assert predicted is True
    assert confidence == 0.996
    assert review is False
    assert reason is None


def test_cns_confident_negative_is_accepted_without_review():
    predicted, _, review, _ = cns_decision(0.01, "Overall survival at 24 months")
    assert predicted is False
    assert review is False


@pytest.mark.parametrize("prob", [UNCERTAIN_LOW, 0.4, 0.5, 0.6, UNCERTAIN_HIGH])
def test_cns_uncertain_band_is_always_flagged(prob):
    _, _, review, reason = cns_decision(prob, "Cognitive assessment of some kind")
    assert review is True
    assert "uncertain zone" in reason


def test_cns_trap_overrides_high_confidence():
    """A confident prediction on trap text still goes to a human."""
    predicted, confidence, review, reason = cns_decision(0.99, "EORTC QLQ-C30 cognitive subscale")
    assert confidence == 0.99
    assert review is True
    assert "known-difficult pattern" in reason


def test_cns_threshold_boundary():
    assert cns_decision(DECISION_THRESHOLD, "x")[0] is True
    assert cns_decision(DECISION_THRESHOLD - 0.001, "x")[0] is False


def test_uncertain_band_bounds():
    assert is_uncertain(UNCERTAIN_LOW) and is_uncertain(UNCERTAIN_HIGH)
    assert not is_uncertain(UNCERTAIN_LOW - 0.01)
    assert not is_uncertain(UNCERTAIN_HIGH + 0.01)


# --------------------------------------------------------------------------
# Keyword route
# --------------------------------------------------------------------------

def test_keyword_route_positive():
    predicted, confidence, review, reason = keyword_decision(
        "Change in MMSE score from baseline", KEYWORDS
    )
    assert predicted is True
    assert confidence == 1.0
    assert review is False
    assert reason is None


def test_keyword_route_negative():
    predicted, confidence, _, _ = keyword_decision("Overall survival", KEYWORDS)
    assert predicted is False
    assert confidence == 0.0


def test_keyword_route_flags_traps():
    _, _, review, reason = keyword_decision(
        "EORTC QLQ-C30 cognitive functioning subscale", KEYWORDS
    )
    assert review is True
    assert reason is not None


# --------------------------------------------------------------------------
# Input validation
# --------------------------------------------------------------------------

@pytest.mark.parametrize("value", ["CNS", " CNS ", "Breast", "Lung", "HeadNeck"])
def test_valid_cancer_types(value):
    assert validate_cancer_type(value) == value.strip()


@pytest.mark.parametrize("value", ["Prostate", "cns", "", "Head & Neck"])
def test_invalid_cancer_types(value):
    with pytest.raises(ValueError):
        validate_cancer_type(value)


def test_keyword_route_covers_three_types():
    assert set(KEYWORD_ROUTE_TYPES) == {"Breast", "Lung", "HeadNeck"}


def test_keyword_config_loaded():
    assert len(KEYWORDS) > 50
    assert all(k == k.lower() for k in KEYWORDS)
