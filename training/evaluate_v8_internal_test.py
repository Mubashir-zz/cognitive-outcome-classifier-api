#!/usr/bin/env python3
"""Evaluate the frozen selected v8 candidate once on the internal test split."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path

from quantize_v8_model import score_fixed_length, validate_quantized_artifacts, write_predictions
from train_v8_chunked_bert import require_ml_dependencies, sha256_file, validate_inputs
from v8_model_core import confusion_metrics


METRICS = ("sensitivity", "specificity", "ppv", "npv", "accuracy", "f1", "balanced_accuracy", "mcc")


def json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value


def quantile(values: list[float], probability: float) -> float:
    ordered = sorted(value for value in values if math.isfinite(value))
    if not ordered:
        return float("nan")
    position = (len(ordered) - 1) * probability
    lower = math.floor(position); upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def auc(labels: list[int], probabilities: list[float]) -> float:
    positives = [value for label, value in zip(labels, probabilities) if label == 1]
    negatives = [value for label, value in zip(labels, probabilities) if label == 0]
    if not positives or not negatives:
        return float("nan")
    concordance = 0.0
    for positive in positives:
        for negative in negatives:
            concordance += 1.0 if positive > negative else 0.5 if positive == negative else 0.0
    return concordance / (len(positives) * len(negatives))


def summarize(rows: list[dict], threshold: float, bootstrap_replicates: int, seed: int) -> dict:
    labels = [int(row["Label"]) for row in rows]
    probabilities = [float(row["Probability"]) for row in rows]
    point = confusion_metrics(labels, probabilities, threshold)
    point["auc"] = auc(labels, probabilities)
    point["brier"] = sum((probability - label) ** 2 for label, probability in zip(labels, probabilities)) / len(labels)
    rng = random.Random(seed)
    bootstrap = defaultdict(list)
    for _ in range(bootstrap_replicates):
        indices = [rng.randrange(len(rows)) for _ in rows]
        sampled_labels = [labels[index] for index in indices]
        sampled_probabilities = [probabilities[index] for index in indices]
        values = confusion_metrics(sampled_labels, sampled_probabilities, threshold)
        for metric in METRICS:
            bootstrap[metric].append(float(values[metric]))
        bootstrap["auc"].append(auc(sampled_labels, sampled_probabilities))
        bootstrap["brier"].append(
            sum((probability - label) ** 2 for label, probability in zip(sampled_labels, sampled_probabilities)) / len(sampled_labels)
        )
    intervals = {
        metric: {"lower_95": quantile(values, 0.025), "upper_95": quantile(values, 0.975)}
        for metric, values in bootstrap.items()
    }
    return {"n": len(rows), "point": point, "bootstrap_intervals": intervals}


def evaluate(
    release_path: Path,
    config_path: Path,
    selection_path: Path,
    quantization_manifest_path: Path,
    output_dir: Path,
    bootstrap_replicates: int | None = None,
) -> dict:
    rows, config, _manifest = validate_inputs(release_path, config_path)
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite internal-test evaluation: {output_dir}")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection.get("status") != "CANDIDATE_SELECTED_ON_CALIBRATION_ONLY":
        raise ValueError("Selection was not frozen from calibration-only evidence")
    if selection.get("feasible_threshold_exists") is not True:
        raise ValueError("Internal-test evaluation is blocked because no prespecified feasible calibration threshold exists")
    if selection.get("internal_test_accessed") is not False or selection.get("challenge_data_accessed") is not False:
        raise ValueError("Selection provenance does not certify untouched evaluation data")
    if selection.get("training_config_sha256") != sha256_file(config_path):
        raise ValueError("Selection is not bound to this training configuration")
    quantization, quantized_model_path, tokenizer_dir = validate_quantized_artifacts(
        quantization_manifest_path, selection_path, config_path, release_path
    )
    global_lock_path = selection_path.with_suffix(selection_path.suffix + ".internal_test_evaluated.lock")
    if global_lock_path.exists():
        raise FileExistsError(f"This frozen selection already has an internal-test evaluation lock: {global_lock_path}")
    test_rows = [row for row in rows if row["Split"] == "internal_test"]
    # Challenge data are never read by this script.
    np, torch, transformers, DataLoader, Dataset, AutoModel, AutoTokenizer, _Trainer, _TrainingArguments = require_ml_dependencies()
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir), local_files_only=True, use_fast=True)
    model = torch.jit.load(str(quantized_model_path), map_location="cpu").eval()
    predictions = score_fixed_length(
        model,
        tokenizer,
        test_rows,
        config,
        torch,
        DataLoader,
        Dataset,
        batch_size=int(config["training"]["train_batch_size"]),
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    prediction_path = output_dir / "internal_test_predictions.csv"
    write_predictions(prediction_path, predictions)
    threshold = float(selection["selected_threshold"])
    replicates = bootstrap_replicates or int(config["internal_test"]["bootstrap_replicates"])
    seed = int(config["internal_test"]["bootstrap_seed"])
    by_cancer = {}
    for cancer in sorted({row["Cancer_Type"] for row in predictions}):
        subset = [row for row in predictions if row["Cancer_Type"] == cancer]
        by_cancer[cancer] = summarize(subset, threshold, replicates, seed + sum(map(ord, cancer)))
    result = {
        "evaluation_version": "1.0.0",
        "status": "INTERNAL_TEST_EVALUATED_ONCE",
        "selection_sha256": sha256_file(selection_path),
        "quantization_manifest_sha256": sha256_file(quantization_manifest_path),
        "evaluated_torchscript_sha256": sha256_file(quantized_model_path),
        "evaluated_artifact": "serialized dynamic-int8 TorchScript candidate",
        "training_config_sha256": sha256_file(config_path),
        "development_release_sha256": sha256_file(release_path),
        "selected_seed": selection["selected_seed"],
        "threshold": threshold,
        "internal_test_records": len(predictions),
        "internal_test_predictions_sha256": sha256_file(prediction_path),
        "bootstrap_replicates": replicates,
        "bootstrap_seed": seed,
        "overall": summarize(predictions, threshold, replicates, seed),
        "by_cancer_type": by_cancer,
        "challenge_data_accessed": False,
        "production_promotion_decided": False,
    }
    safe = json_safe(result)
    result_path = output_dir / "internal_test_evaluation.json"
    result_path.write_text(json.dumps(safe, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (output_dir / "INTERNAL_TEST_EVALUATED.lock").write_text(
        json.dumps({"selection_sha256": sha256_file(selection_path), "result_sha256": sha256_file(result_path)}, indent=2) + "\n",
        encoding="utf-8",
    )
    global_lock_path.write_text(
        json.dumps({
            "status": "FROZEN_SELECTION_INTERNAL_TEST_CONSUMED",
            "selection_sha256": sha256_file(selection_path),
            "result_sha256": sha256_file(result_path),
            "result_path": str(result_path.resolve()),
        }, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(safe, indent=2, allow_nan=False))
    return safe


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--quantization-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int)
    parser.add_argument("--acknowledge-final-internal-test", required=True)
    args = parser.parse_args()
    if args.acknowledge_final_internal_test != "EVALUATE_FROZEN_SELECTION_ONCE":
        raise ValueError("Exact internal-test acknowledgement is required")
    evaluate(
        args.release,
        args.config,
        args.selection,
        args.quantization_manifest,
        args.output_dir,
        args.bootstrap,
    )


if __name__ == "__main__":
    main()
