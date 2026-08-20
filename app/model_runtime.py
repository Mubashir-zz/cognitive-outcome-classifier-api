"""Hash-verified model runtimes for the staging classifier."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class RuntimeConfigurationError(RuntimeError):
    pass


class ModelInputTooLongError(ValueError):
    pass


def is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def contained_file(root: Path, relative: str, description: str) -> Path:
    root = root.resolve()
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise RuntimeConfigurationError(f"{description} path escapes its artifact directory") from exc
    if not path.is_file():
        raise RuntimeConfigurationError(f"{description} is missing: {relative}")
    return path


def verify_directory(root: Path, records: list[dict], description: str) -> str:
    if not root.is_dir() or not isinstance(records, list) or not records:
        raise RuntimeConfigurationError(f"{description} directory or hash records are missing")
    expected: dict[str, str] = {}
    for record in records:
        relative = Path(str(record.get("path", ""))).as_posix()
        if not relative or relative in expected:
            raise RuntimeConfigurationError(f"{description} hash records contain a blank or duplicate path")
        path = contained_file(root, relative, description)
        actual = sha256_file(path)
        if actual != record.get("sha256"):
            raise RuntimeConfigurationError(f"{description} hash mismatch: {relative}")
        if "bytes" in record and int(record["bytes"]) != path.stat().st_size:
            raise RuntimeConfigurationError(f"{description} byte-size mismatch: {relative}")
        expected[relative] = actual
    actual_paths = {
        path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()
    }
    if set(expected) != actual_paths:
        raise RuntimeConfigurationError(f"{description} directory contents differ from its manifest")
    canonical = "".join(f"{path}\0{expected[path]}\n" for path in sorted(expected))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class V8ArtifactSpec:
    root: Path
    manifest_path: Path
    manifest_sha256: str
    model_path: Path
    model_sha256: str
    tokenizer_dir: Path
    tokenizer_sha256: str
    contract_path: Path
    threshold: float
    maximum_sequence_tokens: int
    content_tokens: int
    overlap_tokens: int
    aggregation: str
    selected_seed: int
    training_config_sha256: str
    development_release_sha256: str
    selection_sha256: str


def load_v8_artifact_spec(root: Path, manifest_filename: str, expected_manifest_sha256: str) -> V8ArtifactSpec:
    if not is_sha256(expected_manifest_sha256):
        raise RuntimeConfigurationError("MODEL_MANIFEST_SHA256 must be a lowercase 64-character SHA-256")
    root = root.resolve()
    manifest_path = contained_file(root, manifest_filename, "Quantization manifest")
    manifest_sha = sha256_file(manifest_path)
    if manifest_sha != expected_manifest_sha256:
        raise RuntimeConfigurationError("Quantization manifest SHA-256 differs from the deployed pin")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "QUANTIZED_CANDIDATE_PASSED_CALIBRATION_PARITY":
        raise RuntimeConfigurationError("Quantized artifact has not passed calibration parity")
    if manifest.get("internal_test_accessed") is not False or manifest.get("challenge_data_accessed") is not False:
        raise RuntimeConfigurationError("Quantization manifest has invalid pre-evaluation provenance")
    eager = manifest.get("eager_quantized_parity", {})
    serialized = manifest.get("serialized_torchscript_parity", {})
    if eager.get("parity_passed") is not True or serialized.get("parity_passed") is not True:
        raise RuntimeConfigurationError("Eager and serialized parity must both pass")

    artifacts = manifest.get("artifacts", {})
    model_record = artifacts.get("torchscript", {})
    model_path = contained_file(root, str(model_record.get("path", "")), "TorchScript model")
    model_sha = sha256_file(model_path)
    if model_sha != model_record.get("sha256"):
        raise RuntimeConfigurationError("TorchScript model SHA-256 differs from its manifest")
    if "bytes" in model_record and int(model_record["bytes"]) != model_path.stat().st_size:
        raise RuntimeConfigurationError("TorchScript model byte size differs from its manifest")

    tokenizer_relative = str(artifacts.get("tokenizer_directory", ""))
    tokenizer_dir = (root / tokenizer_relative).resolve()
    try:
        tokenizer_dir.relative_to(root)
    except ValueError as exc:
        raise RuntimeConfigurationError("Tokenizer path escapes its artifact directory") from exc
    tokenizer_sha = verify_directory(tokenizer_dir, artifacts.get("tokenizer_files"), "Tokenizer")

    contract_record = artifacts.get("inference_contract", {})
    contract_path = contained_file(root, str(contract_record.get("path", "")), "Inference contract")
    if sha256_file(contract_path) != contract_record.get("sha256"):
        raise RuntimeConfigurationError("Inference contract SHA-256 differs from its manifest")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if contract.get("input_dtype") != "int64" or contract.get("runtime_device") != "cpu":
        raise RuntimeConfigurationError("Inference contract has an unsupported dtype or runtime device")
    shape = contract.get("input_shape")
    if not isinstance(shape, list) or len(shape) != 2 or shape[0] != "batch":
        raise RuntimeConfigurationError("Inference contract has an invalid input shape")
    maximum_sequence_tokens = int(shape[1])
    chunking = contract.get("token_chunking", {})
    content_tokens = int(chunking.get("content_tokens", 0))
    overlap_tokens = int(chunking.get("overlap_tokens", -1))
    aggregation = str(chunking.get("aggregation", ""))
    if maximum_sequence_tokens <= 2 or content_tokens + 2 != maximum_sequence_tokens:
        raise RuntimeConfigurationError("Inference contract token dimensions are inconsistent")
    if overlap_tokens < 0 or overlap_tokens >= content_tokens:
        raise RuntimeConfigurationError("Inference contract overlap is invalid")
    if aggregation != "maximum_chunk_probability":
        raise RuntimeConfigurationError("Only maximum-chunk aggregation is supported")
    threshold = float(contract.get("decision_threshold"))
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise RuntimeConfigurationError("Inference contract threshold is invalid")
    if not math.isclose(threshold, float(manifest.get("selected_threshold")), rel_tol=0, abs_tol=1e-15):
        raise RuntimeConfigurationError("Manifest and inference-contract thresholds differ")

    required_hashes = {
        "training_config_sha256": manifest.get("training_config_sha256"),
        "development_release_sha256": manifest.get("development_release_sha256"),
        "selection_sha256": manifest.get("selection_sha256"),
    }
    if any(not is_sha256(value) for value in required_hashes.values()):
        raise RuntimeConfigurationError("Quantization manifest lacks required provenance hashes")
    return V8ArtifactSpec(
        root=root,
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha,
        model_path=model_path,
        model_sha256=model_sha,
        tokenizer_dir=tokenizer_dir,
        tokenizer_sha256=tokenizer_sha,
        contract_path=contract_path,
        threshold=threshold,
        maximum_sequence_tokens=maximum_sequence_tokens,
        content_tokens=content_tokens,
        overlap_tokens=overlap_tokens,
        aggregation=aggregation,
        selected_seed=int(manifest["selected_seed"]),
        training_config_sha256=str(required_hashes["training_config_sha256"]),
        development_release_sha256=str(required_hashes["development_release_sha256"]),
        selection_sha256=str(required_hashes["selection_sha256"]),
    )


@dataclass(frozen=True)
class ModelInference:
    probability: float
    input_tokens: int
    truncated: bool
    full_text_processed: bool
    chunk_count: int
    max_chunk_index: int | None
    max_chunk_start_token: int | None
    max_chunk_end_token: int | None
    max_chunk_start_character: int | None
    max_chunk_end_character: int | None


def chunk_bounds(token_count: int, content_tokens: int, overlap_tokens: int) -> list[tuple[int, int]]:
    if token_count < 0 or content_tokens <= 0 or overlap_tokens < 0 or overlap_tokens >= content_tokens:
        raise ValueError("Invalid token chunking parameters")
    if token_count == 0:
        return [(0, 0)]
    stride = content_tokens - overlap_tokens
    result = []
    start = 0
    while start < token_count:
        end = min(start + content_tokens, token_count)
        result.append((start, end))
        if end == token_count:
            break
        start += stride
    return result


def extract_logits(output: Any, torch_module) -> Any:
    if isinstance(output, torch_module.Tensor):
        return output
    if isinstance(output, dict):
        return output["logits"]
    if isinstance(output, tuple):
        return output[0]
    return output.logits


class V8ChunkedRuntime:
    def __init__(
        self,
        spec: V8ArtifactSpec,
        tokenizer,
        model,
        torch_module,
        inference_batch_size: int = 8,
        maximum_input_tokens: int = 50_000,
    ) -> None:
        if inference_batch_size <= 0 or maximum_input_tokens <= 0:
            raise RuntimeConfigurationError("Runtime batch size and maximum input tokens must be positive")
        self.spec = spec
        self.tokenizer = tokenizer
        self.model = model
        self.torch = torch_module
        self.inference_batch_size = inference_batch_size
        self.maximum_input_tokens = maximum_input_tokens

    def score_one(self, text: str) -> ModelInference:
        encoded = self.tokenizer(
            text,
            add_special_tokens=False,
            truncation=False,
            return_offsets_mapping=True,
        )
        raw_ids = list(encoded["input_ids"])
        offsets = [tuple(value) for value in encoded["offset_mapping"]]
        if len(raw_ids) != len(offsets):
            raise RuntimeError("Tokenizer IDs and character offsets are misaligned")
        if len(raw_ids) > self.maximum_input_tokens:
            raise ModelInputTooLongError(
                f"CNS input has {len(raw_ids)} tokens; configured maximum is {self.maximum_input_tokens}"
            )
        bounds = chunk_bounds(len(raw_ids), self.spec.content_tokens, self.spec.overlap_tokens)
        prepared = []
        for index, (start, end) in enumerate(bounds):
            input_ids = self.tokenizer.build_inputs_with_special_tokens(raw_ids[start:end])
            if len(input_ids) > self.spec.maximum_sequence_tokens:
                raise RuntimeError("Prepared v8 chunk exceeds the frozen sequence length")
            prepared.append({
                "chunk_index": index,
                "start": start,
                "end": end,
                "input_ids": input_ids,
                "attention_mask": [1] * len(input_ids),
            })
        best_probability = -1.0
        best = None
        self.model.eval()
        with self.torch.no_grad():
            for start in range(0, len(prepared), self.inference_batch_size):
                batch = prepared[start:start + self.inference_batch_size]
                padded = self.tokenizer.pad(
                    [{"input_ids": item["input_ids"], "attention_mask": item["attention_mask"]} for item in batch],
                    padding="max_length",
                    max_length=self.spec.maximum_sequence_tokens,
                    return_tensors="pt",
                )
                output = self.model(padded["input_ids"], padded["attention_mask"])
                logits = extract_logits(output, self.torch)
                probabilities = self.torch.softmax(logits, dim=1)[:, 1].cpu().tolist()
                for item, probability in zip(batch, probabilities):
                    probability = float(probability)
                    if not math.isfinite(probability) or not 0 <= probability <= 1:
                        raise RuntimeError("V8 model returned an invalid probability")
                    if probability > best_probability:
                        best_probability = probability
                        best = item
        if best is None:
            raise RuntimeError("V8 model produced no chunk prediction")
        token_start, token_end = int(best["start"]), int(best["end"])
        character_start = offsets[token_start][0] if token_start < len(offsets) else 0
        character_end = offsets[token_end - 1][1] if token_end > token_start else 0
        special_tokens = int(self.tokenizer.num_special_tokens_to_add(pair=False))
        return ModelInference(
            probability=best_probability,
            input_tokens=len(raw_ids) + special_tokens,
            truncated=False,
            full_text_processed=True,
            chunk_count=len(bounds),
            max_chunk_index=int(best["chunk_index"]),
            max_chunk_start_token=token_start,
            max_chunk_end_token=token_end,
            max_chunk_start_character=int(character_start),
            max_chunk_end_character=int(character_end),
        )

    def score(self, texts: list[str]) -> list[ModelInference]:
        return [self.score_one(text) for text in texts]
