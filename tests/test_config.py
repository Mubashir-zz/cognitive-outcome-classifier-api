import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ConfigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads((ROOT / "hybrid_config.json").read_text(encoding="utf-8"))

    def test_decision_parameters_are_explicit(self):
        self.assertEqual(self.config["bert_threshold"], 0.5)
        self.assertEqual(self.config["bert_max_tokens"], 256)
        self.assertEqual(self.config["uncertainty_zone"], [0.25, 0.75])
        self.assertEqual(self.config["decision_rule_version"], "cns-bert-or-keyword-rc2")

    def test_keyword_dictionary_is_stable_and_unique(self):
        keywords = self.config["keywords"]
        self.assertEqual(len(keywords), 66)
        self.assertEqual(len(set(keywords)), len(keywords))
        self.assertTrue(all(term == term.lower() for term in keywords))

    def test_review_patterns_are_versioned_with_the_config(self):
        patterns = self.config["review_trigger_patterns"]
        self.assertIn("karnofsky", patterns)
        self.assertIn("qlq-c30", patterns)
        self.assertEqual(len(patterns), len(set(patterns)))


if __name__ == "__main__":
    unittest.main()
