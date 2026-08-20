"""Pure, model-independent decision logic for the candidate v2 classifier."""

from __future__ import annotations

from dataclasses import dataclass


CANONICAL_CANCER_TYPES = ("CNS", "Breast", "Lung", "HeadNeck")
CANCER_TYPE_ALIASES = {
    "cns": "CNS",
    "brain": "CNS",
    "breast": "Breast",
    "lung": "Lung",
    "headneck": "HeadNeck",
    "head neck": "HeadNeck",
    "head and neck": "HeadNeck",
    "head & neck": "HeadNeck",
}


@dataclass(frozen=True)
class KeywordEvidence:
    term: str
    start: int
    end: int


@dataclass(frozen=True)
class Decision:
    predicted_cognitive: bool
    decision_basis: str
    review_recommended: bool
    review_reasons: tuple[str, ...]


def keyword_evidence(text: str, keywords: list[str]) -> list[KeywordEvidence]:
    """Return every first case-insensitive substring hit with character offsets."""
    lowered = text.lower()
    evidence: list[KeywordEvidence] = []
    for term in keywords:
        start = lowered.find(term)
        if start >= 0:
            evidence.append(KeywordEvidence(term=term, start=start, end=start + len(term)))
    return evidence


def normalize_cancer_type(value: str) -> str:
    """Map common user spellings to the stable API cancer-type contract."""
    normalized = " ".join(value.strip().lower().replace("-", " ").split())
    if normalized not in CANCER_TYPE_ALIASES:
        raise ValueError("Unsupported cancer_type")
    return CANCER_TYPE_ALIASES[normalized]


def review_trigger_reasons(text: str, patterns: list[str]) -> list[str]:
    lowered = text.lower()
    return [
        f"Known-difficult pattern matched: {pattern.strip()}"
        for pattern in patterns
        if pattern in lowered
    ]


def cns_union_decision(
    bert_probability: float,
    has_keyword: bool,
    trigger_reasons: list[str] | None = None,
    *,
    threshold: float = 0.5,
    uncertain_low: float = 0.25,
    uncertain_high: float = 0.75,
) -> Decision:
    """Candidate decision rule; not production-approved until external validation."""
    if not 0.0 <= bert_probability <= 1.0:
        raise ValueError("bert_probability must be between 0 and 1")

    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")
    if not 0.0 <= uncertain_low <= uncertain_high <= 1.0:
        raise ValueError("invalid uncertainty zone")

    bert_positive = bert_probability >= threshold
    predicted = bert_positive or has_keyword
    if bert_positive and has_keyword:
        basis = "bert_and_keyword"
    elif bert_positive:
        basis = "bert_only"
    elif has_keyword:
        basis = "keyword_only"
    else:
        basis = "neither"

    reasons = list(trigger_reasons or [])
    if bert_positive != has_keyword:
        reasons.append("BERT and keyword detectors disagree")
    if uncertain_low <= bert_probability <= uncertain_high:
        reasons.append("BERT probability is in the prespecified uncertainty zone")

    return Decision(
        predicted_cognitive=predicted,
        decision_basis=basis,
        review_recommended=bool(reasons),
        review_reasons=tuple(reasons),
    )


def keyword_only_decision(
    has_keyword: bool,
    trigger_reasons: list[str] | None = None,
) -> Decision:
    reasons = tuple(trigger_reasons or [])
    return Decision(
        predicted_cognitive=has_keyword,
        decision_basis="keyword_only" if has_keyword else "neither",
        review_recommended=bool(reasons),
        review_reasons=reasons,
    )
