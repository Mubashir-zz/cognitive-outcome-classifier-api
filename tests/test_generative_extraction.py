#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import unittest

from tools.validate_generative_extraction import validate


SOURCE = "Objective cognition will be assessed using the Montreal Cognitive Assessment at 6 months."
EVIDENCE = "Montreal Cognitive Assessment"
START = SOURCE.index(EVIDENCE)


def valid_extraction() -> dict:
    return {
        "schema_version": "1.0.0",
        "record_id": "synthetic-001",
        "source_text_sha256": hashlib.sha256(SOURCE.encode("utf-8")).hexdigest(),
        "classification": "cognitive_outcome",
        "cognitive_instruments": ["Montreal Cognitive Assessment"],
        "cognitive_domains": ["global_cognition"],
        "assessment_timepoints": ["6 months"],
        "evidence_spans": [
            {"start": START, "end": START + len(EVIDENCE), "text": EVIDENCE, "supports": "instrument"}
        ],
        "rationale": "A named objective cognitive instrument is registered as an outcome.",
        "abstention_reason": None,
        "provenance": {
            "model_provider": "synthetic",
            "model_name": "test-model",
            "model_revision": "immutable-test-revision",
            "prompt_sha256": "1" * 64,
            "temperature": 0,
        },
    }


class GenerativeExtractionTests(unittest.TestCase):
    def test_exact_evidence_passes(self) -> None:
        result = validate(valid_extraction(), SOURCE)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["evidence_spans"], 1)

    def test_hallucinated_evidence_fails(self) -> None:
        extraction = valid_extraction()
        extraction["evidence_spans"][0]["text"] = "Mini-Mental State Examination"
        with self.assertRaisesRegex(ValueError, "does not exactly match"):
            validate(extraction, SOURCE)

    def test_wrong_source_hash_fails(self) -> None:
        extraction = valid_extraction()
        extraction["source_text_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "SHA-256 does not match"):
            validate(extraction, SOURCE)

    def test_positive_without_evidence_fails(self) -> None:
        extraction = valid_extraction()
        extraction["evidence_spans"] = []
        with self.assertRaisesRegex(ValueError, "requires exact evidence"):
            validate(extraction, SOURCE)

    def test_abstention_requires_reason(self) -> None:
        extraction = valid_extraction()
        extraction.update(
            classification="abstain",
            cognitive_instruments=[],
            cognitive_domains=[],
            evidence_spans=[],
            rationale=None,
            abstention_reason=None,
        )
        with self.assertRaisesRegex(ValueError, "requires a reason"):
            validate(extraction, SOURCE)


if __name__ == "__main__":
    unittest.main(verbosity=2)

