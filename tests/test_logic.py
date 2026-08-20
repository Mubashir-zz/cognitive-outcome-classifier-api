import unittest
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.logic import (
    cns_model_primary_decision,
    cns_union_decision,
    keyword_evidence,
    keyword_only_decision,
    normalize_cancer_type,
    review_trigger_reasons,
)


KEYWORDS = ["moca", "trail making", "cognitive function"]


class LogicTests(unittest.TestCase):
    def test_keyword_evidence_returns_offsets(self):
        text = "Outcome assessed with the MoCA and Trail Making Test."
        evidence = keyword_evidence(text, KEYWORDS)
        self.assertEqual(
            [(item.term, text[item.start:item.end].lower()) for item in evidence],
            [("moca", "moca"), ("trail making", "trail making")],
        )

    def test_cns_union_truth_table(self):
        self.assertEqual(cns_union_decision(0.9, True).decision_basis, "bert_and_keyword")
        self.assertEqual(cns_union_decision(0.9, False).decision_basis, "bert_only")
        self.assertEqual(cns_union_decision(0.1, True).decision_basis, "keyword_only")
        self.assertEqual(cns_union_decision(0.1, False).decision_basis, "neither")
        self.assertTrue(cns_union_decision(0.1, True).review_recommended)

    def test_v8_model_primary_rule_does_not_let_keyword_override_label(self):
        decision = cns_model_primary_decision(0.1, True, threshold=0.5)
        self.assertFalse(decision.predicted_cognitive)
        self.assertEqual(decision.decision_basis, "keyword_only_not_decisive")
        self.assertIn("V8 and keyword detectors disagree", decision.review_reasons)

    def test_non_cns_keyword_rule_is_not_called_probability(self):
        decision = keyword_only_decision(True)
        self.assertTrue(decision.predicted_cognitive)
        self.assertEqual(decision.decision_basis, "keyword_only")

    def test_probability_bounds_fail_loudly(self):
        for value in (-0.1, 1.1):
            with self.assertRaises(ValueError):
                cns_union_decision(value, False)

    def test_threshold_is_explicit_and_changes_decision(self):
        self.assertTrue(cns_union_decision(0.6, False, threshold=0.5).predicted_cognitive)
        self.assertFalse(cns_union_decision(0.6, False, threshold=0.9).predicted_cognitive)

    def test_common_cancer_aliases_are_canonicalized(self):
        self.assertEqual(normalize_cancer_type("head & neck"), "HeadNeck")
        self.assertEqual(normalize_cancer_type("cns"), "CNS")
        with self.assertRaises(ValueError):
            normalize_cancer_type("melanoma")

    def test_review_patterns_are_configuration_driven(self):
        reasons = review_trigger_reasons("Karnofsky performance status", ["karnofsky"])
        self.assertEqual(reasons, ["Known-difficult pattern matched: karnofsky"])


if __name__ == "__main__":
    unittest.main()
