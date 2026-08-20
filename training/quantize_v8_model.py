#!/usr/bin/env python3
"""Quantize the frozen v8 candidate and prove calibration-set parity.

This stage is deliberately downstream of calibration-only candidate selection.
It never reads the internal test split or the external challenge benchmark.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from train_v8_chunked_bert import prepare_chunk_records, require_ml_dependencies, sha256_file, validate_inputs


def json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value


def parity_summary(
    reference: list[dict],
    quantized: list[dict],
    threshold: float,
    maximum_probability_delta: float,
) -> dict:
    reference_by_id = {row["NCT_or_TrialID"]: float(row["Probability"]) for row in reference}
    quantized_by_id = {row["NCT_or_TrialID"]: float(row["Probability"]) for row in quantized}
    if len(reference_by_id) != len(reference) or len(quantized_by_id) != len(quantized):
        raise ValueError("Parity inputs contain duplicate trial IDs")
    if reference_by_id.keys() != quantized_by_id.keys():
        raise ValueError("Reference and quantized predictions cover different trials")
    deltas = []
    agreements = []
    for trial_id in sorted(reference_by_id):
        reference_probability = reference_by_id[trial_id]
        quantized_probability = quantized_by_id[trial_id]
        deltas.append(abs(reference_probability - quantized_probability))
        agreements.append((reference_probability >= threshold) == (quantized_probability >= threshold))
    result = {
        "records": len(deltas),
        "class_agreement": sum(agreements) / len(agreements),
        "discordant_classifications": len(agreements) - sum(agreements),
        "maximum_absolute_probability_delta": max(deltas),
        "mean_absolute_probability_delta": sum(deltas) / len(deltas),
        "required_class_agreement": 1.0,
        "allowed_maximum_absolute_probability_delta": maximum_probability_delta,
    }
    result["parity_passed"] = (
        result["class_agreement"] == 1.0
        and result["maximum_absolute_probability_delta"] <= maximum_probability_delta
    )
    return result


def score_fixed_length(wrapper, tokenizer, rows, config, torch, DataLoader, Dataset, batch_size: int) -> list[dict]:
    chunks = prepare_chunk_records(tokenizer, rows, config)

    class InferenceDataset(Dataset):
        def __len__(self):
            return len(chunks)

        def __getitem__(self, index):
            return chunks[index]

    max_length = int(config["max_sequence_tokens"])

    def collate(features):
        padded = tokenizer.pad(
            [{"input_ids": item["input_ids"], "attention_mask": item["attention_mask"]} for item in features],
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        return padded["input_ids"], padded["attention_mask"], features

    loader = DataLoader(InferenceDataset(), batch_size=batch_size, shuffle=False, collate_fn=collate, num_workers=0)
    best = {}
    wrapper.eval()
    with torch.no_grad():
        for input_ids, attention_mask, metadata in loader:
            probabilities = torch.softmax(wrapper(input_ids, attention_mask), dim=1)[:, 1].cpu().tolist()
            for item, probability in zip(metadata, probabilities):
                current = best.get(item["trial_id"])
                if current is None or probability > current["Probability"]:
                    best[item["trial_id"]] = {
                        "NCT_or_TrialID": item["trial_id"],
                        "Cancer_Type": item["cancer_type"],
                        "Label": str(item["label"]),
                        "Probability": float(probability),
                    }
    if len(best) != len(rows):
        raise AssertionError("Quantized trial-level aggregation lost records")
    return [best[row["NCT_or_TrialID"]] for row in rows]


def write_predictions(path: Path, rows: list[dict]) -> None:
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["NCT_or_TrialID", "Cancer_Type", "Label", "Probability"],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def quantize(
    release_path: Path,
    config_path: Path,
    selection_path: Path,
    output_dir: Path,
    maximum_probability_delta: float | None,
) -> dict:
    rows, config, _manifest = validate_inputs(release_path, config_path)
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite quantization output: {output_dir}")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection.get("status") != "CANDIDATE_SELECTED_ON_CALIBRATION_ONLY":
        raise ValueError("Candidate selection is not frozen")
    if selection.get("feasible_threshold_exists") is not True:
        raise ValueError("Quantization is blocked because no prespecified feasible calibration threshold exists")
    if selection.get("internal_test_accessed") is not False or selection.get("challenge_data_accessed") is not False:
        raise ValueError("Selection provenance does not certify untouched evaluation data")
    if selection.get("training_config_sha256") != sha256_file(config_path):
        raise ValueError("Selection is not bound to this training configuration")
    quantization_config = config.get("quantization")
    if not isinstance(quantization_config, dict):
        raise ValueError("Training configuration does not pin a quantization policy")
    pinned_delta = float(quantization_config["maximum_absolute_probability_delta"])
    if maximum_probability_delta is not None and maximum_probability_delta != pinned_delta:
        raise ValueError("CLI quantization tolerance differs from the frozen training configuration")
    maximum_probability_delta = pinned_delta
    calibration_rows = [row for row in rows if row["Split"] == "calibration"]
    # The internal_test split and challenge benchmark are never scored here.
    np, torch, transformers, DataLoader, Dataset, AutoModel, AutoTokenizer, _Trainer, _TrainingArguments = require_ml_dependencies()
    model_dir = Path(selection["selected_model_dir"])
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True, use_fast=True)
    model = AutoModel.from_pretrained(str(model_dir), local_files_only=True)
    model.to("cpu").eval()

    class LogitsWrapper(torch.nn.Module):
        def __init__(self, classifier):
            super().__init__()
            self.classifier = classifier

        def forward(self, input_ids, attention_mask):
            return self.classifier(input_ids=input_ids, attention_mask=attention_mask).logits

    full_precision = LogitsWrapper(model).eval()
    reference_predictions = score_fixed_length(
        full_precision, tokenizer, calibration_rows, config, torch, DataLoader, Dataset,
        int(config["training"]["train_batch_size"]),
    )
    quantized = torch.ao.quantization.quantize_dynamic(
        full_precision,
        {torch.nn.Linear},
        dtype=torch.qint8,
    ).eval()
    quantized_predictions = score_fixed_length(
        quantized, tokenizer, calibration_rows, config, torch, DataLoader, Dataset,
        int(config["training"]["train_batch_size"]),
    )
    threshold = float(selection["selected_threshold"])
    eager_parity = parity_summary(reference_predictions, quantized_predictions, threshold, maximum_probability_delta)
    if not eager_parity["parity_passed"]:
        raise RuntimeError(f"Eager quantization parity gate failed: {json.dumps(eager_parity, sort_keys=True)}")

    output_dir.mkdir(parents=True, exist_ok=False)
    reference_path = output_dir / "calibration_predictions_full_precision.csv"
    quantized_path = output_dir / "calibration_predictions_quantized.csv"
    write_predictions(reference_path, reference_predictions)
    write_predictions(quantized_path, quantized_predictions)
    tokenizer_dir = output_dir / "tokenizer"
    tokenizer.save_pretrained(str(tokenizer_dir))
    example_ids = torch.zeros((1, int(config["max_sequence_tokens"])), dtype=torch.long)
    example_mask = torch.ones_like(example_ids)
    traced = torch.jit.trace(quantized, (example_ids, example_mask), strict=False)
    traced = torch.jit.freeze(traced.eval())
    model_path = output_dir / "model_quantized_int8.ts"
    torch.jit.save(traced, str(model_path))
    serialized = torch.jit.load(str(model_path), map_location="cpu").eval()
    serialized_predictions = score_fixed_length(
        serialized, tokenizer, calibration_rows, config, torch, DataLoader, Dataset,
        int(config["training"]["train_batch_size"]),
    )
    serialized_parity = parity_summary(reference_predictions, serialized_predictions, threshold, maximum_probability_delta)
    if not serialized_parity["parity_passed"]:
        raise RuntimeError(f"Serialized quantization parity gate failed: {json.dumps(serialized_parity, sort_keys=True)}")
    serialized_path = output_dir / "calibration_predictions_serialized_torchscript.csv"
    write_predictions(serialized_path, serialized_predictions)
    inference_contract = {
        "contract_version": "1.0.0",
        "input_names": ["input_ids", "attention_mask"],
        "input_dtype": "int64",
        "input_shape": ["batch", int(config["max_sequence_tokens"])],
        "output": "two-class logits [No, Yes]",
        "token_chunking": {
            "content_tokens": int(config["content_tokens_per_chunk"]),
            "overlap_tokens": int(config["chunk_overlap_tokens"]),
            "aggregation": config["trial_probability_aggregation"],
        },
        "decision_threshold": threshold,
        "runtime_device": "cpu",
    }
    contract_path = output_dir / "inference_contract.json"
    contract_path.write_text(json.dumps(inference_contract, indent=2) + "\n", encoding="utf-8")
    result = {
        "quantization_version": "1.0.0",
        "status": "QUANTIZED_CANDIDATE_PASSED_CALIBRATION_PARITY",
        "selection_sha256": sha256_file(selection_path),
        "training_config_sha256": sha256_file(config_path),
        "development_release_sha256": sha256_file(release_path),
        "selected_seed": selection["selected_seed"],
        "selected_threshold": threshold,
        "quantization": "PyTorch dynamic int8 quantization of Linear layers",
        "parity_scope": "calibration split only",
        "eager_quantized_parity": eager_parity,
        "serialized_torchscript_parity": serialized_parity,
        "artifacts": {
            "torchscript": {"path": model_path.name, "sha256": sha256_file(model_path), "bytes": model_path.stat().st_size},
            "tokenizer_directory": tokenizer_dir.name,
            "inference_contract": {"path": contract_path.name, "sha256": sha256_file(contract_path)},
            "reference_predictions_sha256": sha256_file(reference_path),
            "quantized_predictions_sha256": sha256_file(quantized_path),
            "serialized_torchscript_predictions_sha256": sha256_file(serialized_path),
        },
        "internal_test_accessed": False,
        "challenge_data_accessed": False,
        "production_promotion_decided": False,
        "environment": {"torch": torch.__version__, "transformers": transformers.__version__},
    }
    result_path = output_dir / "quantization_manifest.json"
    result_path.write_text(json.dumps(json_safe(result), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(json_safe(result), indent=2, allow_nan=False))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--maximum-probability-delta", type=float)
    args = parser.parse_args()
    if args.maximum_probability_delta is not None and not 0.0 <= args.maximum_probability_delta <= 1.0:
        raise ValueError("--maximum-probability-delta must be between 0 and 1")
    quantize(args.release, args.config, args.selection, args.output_dir, args.maximum_probability_delta)


if __name__ == "__main__":
    main()
