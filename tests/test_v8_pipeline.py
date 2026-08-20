#!/usr/bin/env python3

import ast
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAINING = ROOT / "training"
if str(TRAINING) not in sys.path:
    sys.path.insert(0, str(TRAINING))

from quantize_v8_model import parity_summary
from v8_model_core import chunk_token_ids, confusion_metrics, select_threshold


class V8PipelineTests(unittest.TestCase):
    def test_chunking_is_lossless(self) -> None:
        tokens = list(range(1000))
        chunks = chunk_token_ids(tokens, content_tokens=382, overlap_tokens=64)
        covered = set()
        for chunk in chunks:
            covered.update(range(chunk.start_token, chunk.end_token))
            self.assertLessEqual(len(chunk.token_ids), 382)
        self.assertEqual(covered, set(range(1000)))

    def test_joint_threshold_constraints(self) -> None:
        result = select_threshold(
            [1, 1, 1, 0, 0, 0], [0.99, 0.90, 0.85, 0.20, 0.10, 0.01],
            minimum_sensitivity=0.90, minimum_specificity=0.95,
        )
        self.assertTrue(result["feasible_threshold_exists"])
        self.assertGreaterEqual(result["selected"]["sensitivity"], 0.90)
        self.assertGreaterEqual(result["selected"]["specificity"], 0.95)

    def test_confusion_metrics(self) -> None:
        result = confusion_metrics([1, 1, 0, 0], [0.9, 0.1, 0.8, 0.2], 0.5)
        self.assertEqual((result["tp"], result["tn"], result["fp"], result["fn"]), (1, 1, 1, 1))

    def test_quantization_class_change_fails(self) -> None:
        result = parity_summary(
            [{"NCT_or_TrialID": "A", "Probability": 0.51}],
            [{"NCT_or_TrialID": "A", "Probability": 0.49}],
            0.5, 0.1, 1.0,
        )
        self.assertFalse(result["parity_passed"])

    def test_pipeline_isolation_is_explicit(self) -> None:
        names = [
            "train_v8_chunked_bert.py", "quantize_v8_model.py",
            "evaluate_v8_internal_test.py", "score_v8_frozen_challenge.py",
        ]
        sources = {name: (TRAINING / name).read_text(encoding="utf-8") for name in names}
        for source in sources.values():
            ast.parse(source)
        self.assertNotIn('row["Split"] == "internal_test"', sources["train_v8_chunked_bert.py"])
        self.assertIn('row["Split"] == "calibration"', sources["quantize_v8_model.py"])
        self.assertIn("validate_quantized_artifacts", sources["evaluate_v8_internal_test.py"])
        self.assertIn("EVALUATE_FROZEN_SELECTION_ONCE", sources["evaluate_v8_internal_test.py"])
        self.assertNotIn('add_argument("--sealed-key"', sources["score_v8_frozen_challenge.py"])
        self.assertNotIn('add_argument("--frozen"', sources["score_v8_frozen_challenge.py"])
        self.assertIn('add_argument("--blinded"', sources["score_v8_frozen_challenge.py"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
