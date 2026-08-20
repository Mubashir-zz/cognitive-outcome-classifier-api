from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.model_runtime import (
    ModelInputTooLongError,
    RuntimeConfigurationError,
    V8ChunkedRuntime,
    chunk_bounds,
    load_v8_artifact_spec,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_artifact(root: Path) -> Path:
    model = root / "model.ts"
    model.write_bytes(b"synthetic-torchscript")
    tokenizer = root / "tokenizer"
    tokenizer.mkdir()
    tokenizer_file = tokenizer / "tokenizer.json"
    tokenizer_file.write_text('{"synthetic":true}', encoding="utf-8")
    contract = root / "inference_contract.json"
    contract.write_text(json.dumps({
        "input_dtype": "int64",
        "input_shape": ["batch", 384],
        "runtime_device": "cpu",
        "token_chunking": {
            "content_tokens": 382,
            "overlap_tokens": 64,
            "aggregation": "maximum_chunk_probability",
        },
        "decision_threshold": 0.42,
    }), encoding="utf-8")
    manifest = root / "quantization_manifest.json"
    manifest.write_text(json.dumps({
        "status": "QUANTIZED_CANDIDATE_PASSED_CALIBRATION_PARITY",
        "internal_test_accessed": False,
        "challenge_data_accessed": False,
        "selected_seed": 20260820,
        "selected_threshold": 0.42,
        "training_config_sha256": "1" * 64,
        "development_release_sha256": "2" * 64,
        "selection_sha256": "3" * 64,
        "eager_quantized_parity": {"parity_passed": True},
        "serialized_torchscript_parity": {"parity_passed": True},
        "artifacts": {
            "torchscript": {"path": model.name, "sha256": sha256(model), "bytes": model.stat().st_size},
            "tokenizer_directory": tokenizer.name,
            "tokenizer_files": [{
                "path": tokenizer_file.name,
                "sha256": sha256(tokenizer_file),
                "bytes": tokenizer_file.stat().st_size,
            }],
            "inference_contract": {"path": contract.name, "sha256": sha256(contract)},
        },
    }), encoding="utf-8")
    return manifest


class ModelRuntimeTests(unittest.TestCase):
    def test_valid_artifact_is_fully_bound(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = write_artifact(root)
            spec = load_v8_artifact_spec(root, manifest.name, sha256(manifest))
            self.assertEqual(spec.threshold, 0.42)
            self.assertEqual(spec.maximum_sequence_tokens, 384)
            self.assertEqual(spec.content_tokens, 382)
            self.assertEqual(spec.model_sha256, sha256(root / "model.ts"))
            self.assertEqual(len(spec.tokenizer_sha256), 64)

    def test_tokenizer_tampering_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = write_artifact(root)
            (root / "tokenizer" / "tokenizer.json").write_text("tampered", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeConfigurationError, "Tokenizer hash mismatch"):
                load_v8_artifact_spec(root, manifest.name, sha256(manifest))

    def test_manifest_pin_is_mandatory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = write_artifact(root)
            with self.assertRaisesRegex(RuntimeConfigurationError, "MODEL_MANIFEST_SHA256"):
                load_v8_artifact_spec(root, manifest.name, "")

    def test_chunk_bounds_cover_all_tokens_with_overlap(self):
        bounds = chunk_bounds(1000, 382, 64)
        covered = set()
        for start, end in bounds:
            covered.update(range(start, end))
        self.assertEqual(covered, set(range(1000)))
        self.assertTrue(all(right[0] < left[1] for left, right in zip(bounds, bounds[1:])))

    def test_chunked_runtime_scores_full_text_and_returns_winning_offsets(self):
        class FakeVector:
            def __init__(self, values): self.values = values
            def cpu(self): return self
            def tolist(self): return self.values

        class FakeTensor:
            def __init__(self, rows): self.rows = rows
            def __getitem__(self, key):
                row_selector, column = key
                self.assert_slice(row_selector)
                return FakeVector([row[column] for row in self.rows])
            @staticmethod
            def assert_slice(value):
                if not isinstance(value, slice): raise AssertionError(value)

        class FakeNoGrad:
            def __enter__(self): return None
            def __exit__(self, *_args): return False

        class FakeTorch:
            Tensor = FakeTensor
            @staticmethod
            def no_grad(): return FakeNoGrad()
            @staticmethod
            def softmax(value, dim):
                if dim != 1: raise AssertionError(dim)
                return value

        class FakeTokenizer:
            def __call__(self, text, **_kwargs):
                words = text.split()
                offsets = []
                cursor = 0
                for word in words:
                    start = text.index(word, cursor)
                    offsets.append((start, start + len(word)))
                    cursor = start + len(word)
                return {"input_ids": list(range(1, len(words) + 1)), "offset_mapping": offsets}
            @staticmethod
            def build_inputs_with_special_tokens(ids): return [101] + list(ids) + [102]
            @staticmethod
            def num_special_tokens_to_add(pair=False):
                if pair: raise AssertionError
                return 2
            @staticmethod
            def pad(features, padding, max_length, return_tensors):
                if padding != "max_length" or return_tensors != "pt": raise AssertionError
                ids = [item["input_ids"] + [0] * (max_length - len(item["input_ids"])) for item in features]
                masks = [item["attention_mask"] + [0] * (max_length - len(item["attention_mask"])) for item in features]
                return {"input_ids": ids, "attention_mask": masks}

        class FakeModel:
            @staticmethod
            def eval(): return None
            @staticmethod
            def __call__(input_ids, _attention_mask):
                probabilities = [row[1] / 10 for row in input_ids]
                return FakeTensor([[1 - value, value] for value in probabilities])

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = write_artifact(root)
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            contract = root / "inference_contract.json"
            contract_payload = json.loads(contract.read_text(encoding="utf-8"))
            contract_payload["input_shape"] = ["batch", 6]
            contract_payload["token_chunking"] = {
                "content_tokens": 4, "overlap_tokens": 1,
                "aggregation": "maximum_chunk_probability",
            }
            contract.write_text(json.dumps(contract_payload), encoding="utf-8")
            payload["artifacts"]["inference_contract"]["sha256"] = sha256(contract)
            manifest.write_text(json.dumps(payload), encoding="utf-8")
            spec = load_v8_artifact_spec(root, manifest.name, sha256(manifest))
            runtime = V8ChunkedRuntime(spec, FakeTokenizer(), FakeModel(), FakeTorch(), inference_batch_size=2, maximum_input_tokens=20)
            result = runtime.score_one("one two three four five six seven eight nine ten")
            self.assertAlmostEqual(result.probability, 0.7)
            self.assertTrue(result.full_text_processed)
            self.assertFalse(result.truncated)
            self.assertEqual(result.chunk_count, 3)
            self.assertEqual((result.max_chunk_start_token, result.max_chunk_end_token), (6, 10))
            self.assertEqual("seven eight nine ten", "one two three four five six seven eight nine ten"[result.max_chunk_start_character:result.max_chunk_end_character])
            runtime.maximum_input_tokens = 5
            with self.assertRaises(ModelInputTooLongError):
                runtime.score_one("one two three four five six")


if __name__ == "__main__":
    unittest.main(verbosity=2)
