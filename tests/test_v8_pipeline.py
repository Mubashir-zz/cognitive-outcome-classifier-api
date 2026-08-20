#!/usr/bin/env python3

import ast
import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAINING = ROOT / "training"
if str(TRAINING) not in sys.path:
    sys.path.insert(0, str(TRAINING))

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
        labels = [1, 1, 1, 0, 0, 0]
        probabilities = [0.99, 0.90, 0.85, 0.20, 0.10, 0.01]
        result = select_threshold(labels, probabilities, minimum_sensitivity=0.90, minimum_specificity=0.95)
        self.assertTrue(result["feasible_threshold_exists"])
        self.assertGreaterEqual(result["selected"]["sensitivity"], 0.90)
        self.assertGreaterEqual(result["selected"]["specificity"], 0.95)

    def test_confusion_metrics(self) -> None:
        result = confusion_metrics([1, 1, 0, 0], [0.9, 0.1, 0.8, 0.2], 0.5)
        self.assertEqual((result["tp"], result["tn"], result["fp"], result["fn"]), (1, 1, 1, 1))
        self.assertEqual(result["balanced_accuracy"], 0.5)

    def test_pipeline_isolation_is_explicit(self) -> None:
        training = (TRAINING / "train_v8_chunked_bert.py").read_text(encoding="utf-8")
        quantize = (TRAINING / "quantize_v8_model.py").read_text(encoding="utf-8")
        internal = (TRAINING / "evaluate_v8_internal_test.py").read_text(encoding="utf-8")
        for source in (training, quantize, internal):
            ast.parse(source)
        self.assertNotIn('row["Split"] == "internal_test"', training)
        self.assertIn('row["Split"] == "calibration"', quantize)
        self.assertNotIn('row["Split"] == "internal_test"', quantize)
        self.assertIn("serialized_torchscript_parity", quantize)
        self.assertIn("EVALUATE_FROZEN_SELECTION_ONCE", internal)
        self.assertIn("internal_test_evaluated.lock", internal)
        self.assertIn("--quantization-manifest", internal)
        self.assertIn("torch.jit.load", internal)

    def test_v8_release_gate_requires_hash_bound_technical_evidence(self) -> None:
        gates = json.loads((TRAINING / "classifier_release_gates_v1_2.json").read_text(encoding="utf-8"))
        evaluator = (TRAINING / "evaluate_classifier_release_gates.py").read_text(encoding="utf-8")
        self.assertEqual(gates["scientific"]["primary_detector"], "V8")
        self.assertEqual(gates["technical"]["required_model_runtime"], "v8_chunked")
        self.assertEqual(gates["technical"]["required_cns_decision_mode"], "model_primary")
        self.assertTrue(gates["technical"]["require_artifact_chain_match"])
        self.assertTrue(gates["technical"]["require_hash_bound_evidence_collector"])
        self.assertIn("COLLECTOR_CANDIDATES", evaluator)
        self.assertIn("technical_collector_binding", evaluator)
        self.assertIn("technical_evidence_checks", evaluator)


if __name__ == "__main__":
    unittest.main(verbosity=2)
