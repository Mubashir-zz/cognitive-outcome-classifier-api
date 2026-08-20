#!/usr/bin/env python3
"""Score frozen challenge texts with the exact v8 deployment artifact.

This stage runs only after human labels are frozen, but it scores the original
blinded text file and never opens the frozen human-label CSV or the sealed
sampling/model-output key. Predictions are bound to both by SHA-256 manifests.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from quantize_v8_model import score_fixed_length, validate_quantized_artifacts
from train_v8_chunked_bert import require_ml_dependencies, sha256_file, validate_inputs


FROZEN_HEADERS = [
    "Row_ID",
    "Outcome_Text_Part_1",
    "Outcome_Text_Part_2",
    "Measures_Cognition_Y_N",
    "Reviewer_Confidence_High_Medium_Low",
    "Notes",
]
OUTPUT_HEADERS = ["Row_ID", "V8_Probability", "V8_Prediction"]


def read_blinded_texts(blinded_path: Path, frozen_manifest_path: Path) -> tuple[list[dict[str, str]], dict]:
    if not frozen_manifest_path.is_file():
        raise FileNotFoundError("Frozen-label manifest is required")
    manifest = json.loads(frozen_manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "FROZEN_HUMAN_LABELS":
        raise ValueError("Manifest does not certify frozen human labels")
    if manifest.get("records") != 300 or not manifest.get("frozen_sha256"):
        raise ValueError("Frozen-label manifest does not bind 300 records")
    if manifest.get("blinded_source_sha256") != sha256_file(blinded_path):
        raise ValueError("Blinded challenge text differs from the frozen-label provenance")
    with blinded_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != FROZEN_HEADERS:
            raise ValueError("Unexpected blinded challenge schema")
        raw_rows = list(reader)
    row_ids = [row["Row_ID"] for row in raw_rows]
    if len(raw_rows) != 300 or len(set(row_ids)) != 300 or any(not value.strip() for value in row_ids):
        raise ValueError("Blinded challenge must contain 300 unique nonblank Row_ID values")
    if any(any((row[field] or "").strip() for field in FROZEN_HEADERS[3:]) for row in raw_rows):
        raise ValueError("Blinded scoring input unexpectedly contains adjudication data")
    # A constant placeholder satisfies the shared chunk encoder; no human truth
    # is present anywhere in this process.
    model_rows = [
        {
            "NCT_or_TrialID": row["Row_ID"],
            "Cancer_Type": "CNS",
            "Outcome_Text": (row["Outcome_Text_Part_1"] or "") + (row["Outcome_Text_Part_2"] or ""),
            "Label": "No",
        }
        for row in raw_rows
    ]
    if any(not row["Outcome_Text"].strip() for row in model_rows):
        raise ValueError("Blinded challenge contains blank outcome text")
    return model_rows, manifest


def score(
    blinded_path: Path,
    frozen_manifest_path: Path,
    release_path: Path,
    config_path: Path,
    selection_path: Path,
    quantization_manifest_path: Path,
    output_path: Path,
) -> dict:
    manifest_path = output_path.with_suffix(output_path.suffix + ".manifest.json")
    if output_path.exists() or manifest_path.exists():
        raise FileExistsError("Refusing to overwrite frozen v8 challenge predictions")
    rows, frozen_manifest = read_blinded_texts(blinded_path, frozen_manifest_path)
    _release_rows, config, _release_manifest = validate_inputs(release_path, config_path)
    quantization, model_path, tokenizer_dir = validate_quantized_artifacts(
        quantization_manifest_path, selection_path, config_path, release_path
    )
    np, torch, transformers, DataLoader, Dataset, _AutoModel, AutoTokenizer, _Trainer, _TrainingArguments = require_ml_dependencies()
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir), local_files_only=True, use_fast=True)
    model = torch.jit.load(str(model_path), map_location="cpu").eval()
    scored = score_fixed_length(
        model, tokenizer, rows, config, torch, DataLoader, Dataset,
        batch_size=int(config["training"]["train_batch_size"]),
    )
    threshold = float(quantization["selected_threshold"])
    output_rows = [
        {
            "Row_ID": row["NCT_or_TrialID"],
            "V8_Probability": f"{float(row['Probability']):.17g}",
            "V8_Prediction": str(int(float(row["Probability"]) >= threshold)),
        }
        for row in scored
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_HEADERS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(output_rows)
    result = {
        "manifest_version": "1.0.0",
        "status": "FROZEN_V8_CHALLENGE_PREDICTIONS",
        "records": len(output_rows),
        "frozen_labels_sha256": frozen_manifest["frozen_sha256"],
        "frozen_labels_manifest_sha256": sha256_file(frozen_manifest_path),
        "blinded_source_sha256": sha256_file(blinded_path),
        "quantization_manifest_sha256": sha256_file(quantization_manifest_path),
        "serialized_torchscript_sha256": sha256_file(model_path),
        "training_config_sha256": sha256_file(config_path),
        "development_release_sha256": sha256_file(release_path),
        "selected_threshold": threshold,
        "output_sha256": sha256_file(output_path),
        "human_truth_used_for_scoring": False,
        "reviewer_confidence_used_for_scoring": False,
        "sealed_key_opened_by_this_script": False,
        "challenge_identity_used_for_scoring": False,
        "environment": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "numpy": np.__version__,
        },
    }
    manifest_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--blinded", type=Path, required=True)
    parser.add_argument("--frozen-manifest", type=Path, required=True)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--quantization-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    score(
        args.blinded, args.frozen_manifest, args.release, args.config, args.selection,
        args.quantization_manifest, args.output,
    )


if __name__ == "__main__":
    main()
