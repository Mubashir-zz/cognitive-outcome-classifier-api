#!/usr/bin/env python3
"""Validate evidence grounding for one schema-constrained extraction.

The source text is supplied separately and is never copied into error output.
This validator is intentionally independent of any particular model provider.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


DOMAINS = {
    "global_cognition",
    "memory",
    "attention",
    "executive_function",
    "processing_speed",
    "language",
    "visuospatial_function",
    "social_cognition",
    "other_cognitive_domain",
}
CLASSIFICATIONS = {"cognitive_outcome", "not_cognitive_outcome", "abstain"}
SUPPORT_TYPES = {"classification", "instrument", "domain", "timepoint", "exclusion"}
REQUIRED_TOP_LEVEL = {
    "schema_version",
    "record_id",
    "source_text_sha256",
    "classification",
    "cognitive_instruments",
    "cognitive_domains",
    "assessment_timepoints",
    "evidence_spans",
    "rationale",
    "abstention_reason",
    "provenance",
}
REQUIRED_PROVENANCE = {"model_provider", "model_name", "model_revision", "prompt_sha256", "temperature"}


def fail(message: str) -> None:
    raise ValueError(message)


def validate(extraction: dict, source_text: str) -> dict:
    if set(extraction) != REQUIRED_TOP_LEVEL:
        fail("Extraction has missing or unexpected top-level fields")
    if extraction["schema_version"] != "1.0.0":
        fail("Unsupported extraction schema version")
    if extraction["classification"] not in CLASSIFICATIONS:
        fail("Invalid classification")
    expected_hash = hashlib.sha256(source_text.encode("utf-8")).hexdigest()
    if extraction["source_text_sha256"] != expected_hash:
        fail("Source-text SHA-256 does not match")

    for field in ("cognitive_instruments", "cognitive_domains", "assessment_timepoints", "evidence_spans"):
        if not isinstance(extraction[field], list):
            fail(f"{field} must be an array")
    if len(set(extraction["cognitive_domains"])) != len(extraction["cognitive_domains"]):
        fail("Cognitive domains must be unique")
    if not set(extraction["cognitive_domains"]).issubset(DOMAINS):
        fail("Invalid cognitive domain")

    for index, span in enumerate(extraction["evidence_spans"], start=1):
        if set(span) != {"start", "end", "text", "supports"}:
            fail(f"Evidence span {index} has invalid fields")
        start, end = span["start"], span["end"]
        if not isinstance(start, int) or isinstance(start, bool) or not isinstance(end, int) or isinstance(end, bool):
            fail(f"Evidence span {index} offsets must be integers")
        if start < 0 or end <= start or end > len(source_text):
            fail(f"Evidence span {index} offsets are out of bounds")
        if span["text"] != source_text[start:end]:
            fail(f"Evidence span {index} does not exactly match the source text")
        if span["supports"] not in SUPPORT_TYPES:
            fail(f"Evidence span {index} has an invalid support type")

    classification = extraction["classification"]
    rationale = extraction["rationale"]
    abstention_reason = extraction["abstention_reason"]
    if classification == "cognitive_outcome":
        if not extraction["evidence_spans"]:
            fail("A cognitive-outcome decision requires exact evidence")
        if not extraction["cognitive_instruments"] and not extraction["cognitive_domains"]:
            fail("A cognitive-outcome decision requires an instrument or cognitive domain")
        if abstention_reason is not None:
            fail("A non-abstaining decision cannot include an abstention reason")
    elif classification == "not_cognitive_outcome":
        if extraction["cognitive_instruments"] or extraction["cognitive_domains"]:
            fail("A non-cognitive decision cannot assert cognitive instruments or domains")
        if abstention_reason is not None:
            fail("A non-abstaining decision cannot include an abstention reason")
    else:
        if not isinstance(abstention_reason, str) or not abstention_reason.strip():
            fail("An abstention requires a reason")
        if extraction["cognitive_instruments"] or extraction["cognitive_domains"]:
            fail("An abstention cannot assert cognitive instruments or domains")
    if classification != "abstain" and (not isinstance(rationale, str) or not rationale.strip()):
        fail("A non-abstaining decision requires a rationale")

    provenance = extraction["provenance"]
    if not isinstance(provenance, dict) or set(provenance) != REQUIRED_PROVENANCE:
        fail("Invalid provenance fields")
    if provenance["temperature"] != 0:
        fail("Extraction temperature must be zero")
    for field in ("model_provider", "model_name", "model_revision"):
        if not isinstance(provenance[field], str) or not provenance[field].strip():
            fail(f"Missing provenance field: {field}")
    prompt_hash = provenance["prompt_sha256"]
    if not isinstance(prompt_hash, str) or len(prompt_hash) != 64 or any(c not in "0123456789abcdef" for c in prompt_hash):
        fail("Invalid prompt SHA-256")

    return {
        "status": "PASS",
        "record_id": extraction["record_id"],
        "classification": classification,
        "source_text_sha256": expected_hash,
        "evidence_spans": len(extraction["evidence_spans"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--extraction", type=Path, required=True)
    parser.add_argument("--source-text", type=Path, required=True)
    args = parser.parse_args()
    extraction = json.loads(args.extraction.read_text(encoding="utf-8"))
    source_text = args.source_text.read_text(encoding="utf-8")
    print(json.dumps(validate(extraction, source_text), indent=2))


if __name__ == "__main__":
    main()

