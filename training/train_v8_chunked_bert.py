#!/usr/bin/env python3
"""Train full-text chunked Bio_ClinicalBERT candidates on the frozen v8 release.

The challenge benchmark is not read. Candidate selection uses calibration
predictions only; the internal test split is not scored by this script.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

from v8_model_core import chunk_token_ids


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_release(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {
        "NCT_or_TrialID", "Outcome_Text", "Label", "Cancer_Type", "Split",
        "Normalized_ID_SHA256", "Normalized_Text_SHA256",
    }
    if not rows or not required.issubset(rows[0]):
        raise ValueError("Unexpected v8 development-release schema")
    if any(row["Split"] not in {"train", "calibration", "internal_test"} for row in rows):
        raise ValueError("Invalid split value")
    if any(row["Label"] not in {"Yes", "No"} for row in rows):
        raise ValueError("Invalid label value")
    if len({row["Normalized_ID_SHA256"] for row in rows}) != len(rows):
        raise ValueError("Normalized trial IDs are not unique")
    if len({row["Normalized_Text_SHA256"] for row in rows}) != len(rows):
        raise ValueError("Normalized outcome texts are not unique")
    return rows


def validate_inputs(release_path: Path, config_path: Path) -> tuple[list[dict[str, str]], dict, dict]:
    rows = read_release(release_path)
    config = read_json(config_path)
    manifest_path = release_path.parent / "v8_split_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError("v8_split_manifest.json must accompany the development release")
    manifest = read_json(manifest_path)
    release_sha = sha256_file(release_path)
    if manifest.get("data_file", {}).get("sha256") != release_sha:
        raise ValueError("Development release SHA-256 does not match its split manifest")
    if config.get("development_release_sha256") != release_sha:
        raise ValueError("Training configuration is not pinned to this development release")
    if manifest.get("invariants", {}).get("challenge_text_overlap_in_release") != 0:
        raise ValueError("Development release is not challenge-text disjoint")
    if not manifest.get("challenge_blinded_source", {}).get("sealed_key_opened") is False:
        raise ValueError("Split manifest does not certify sealed-key isolation")
    revision = config.get("base_model_revision", "")
    if len(revision) != 40 or any(character not in "0123456789abcdef" for character in revision):
        raise ValueError("Base model revision must be an immutable 40-character commit SHA")
    if config.get("content_tokens_per_chunk") + 2 != config.get("max_sequence_tokens"):
        raise ValueError("Content-token and sequence-token configuration is inconsistent")
    return rows, config, manifest


def require_ml_dependencies():
    try:
        import numpy as np
        import torch
        import transformers
        from torch.utils.data import DataLoader, Dataset
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
    except ImportError as exc:
        raise RuntimeError(
            "v8 training dependencies are unavailable. Use Python 3.11 and install v8_requirements.lock.txt."
        ) from exc
    return np, torch, transformers, DataLoader, Dataset, AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments


def prepare_chunk_records(tokenizer, rows: list[dict[str, str]], config: dict) -> list[dict]:
    records = []
    for trial_index, row in enumerate(rows):
        raw_ids = tokenizer.encode(row["Outcome_Text"], add_special_tokens=False, truncation=False)
        chunks = chunk_token_ids(
            raw_ids,
            content_tokens=int(config["content_tokens_per_chunk"]),
            overlap_tokens=int(config["chunk_overlap_tokens"]),
        )
        weight = 1.0 / len(chunks)
        for chunk_index, chunk in enumerate(chunks):
            input_ids = tokenizer.build_inputs_with_special_tokens(list(chunk.token_ids))
            if len(input_ids) > int(config["max_sequence_tokens"]):
                raise AssertionError("Prepared chunk exceeds max_sequence_tokens")
            records.append({
                "trial_index": trial_index,
                "trial_id": row["NCT_or_TrialID"],
                "cancer_type": row["Cancer_Type"],
                "label": 1 if row["Label"] == "Yes" else 0,
                "chunk_index": chunk_index,
                "chunk_count": len(chunks),
                "start_token": chunk.start_token,
                "end_token": chunk.end_token,
                "sample_weight": weight,
                "input_ids": input_ids,
                "attention_mask": [1] * len(input_ids),
            })
    return records


def score_records(model, tokenizer, rows: list[dict[str, str]], config: dict, torch, DataLoader, Dataset, batch_size: int) -> list[dict]:
    chunks = prepare_chunk_records(tokenizer, rows, config)

    class InferenceDataset(Dataset):
        def __len__(self):
            return len(chunks)

        def __getitem__(self, index):
            return chunks[index]

    def collate(features):
        padded = tokenizer.pad(
            [{"input_ids": item["input_ids"], "attention_mask": item["attention_mask"]} for item in features],
            padding=True,
            return_tensors="pt",
        )
        padded["metadata"] = features
        return padded

    device = next(model.parameters()).device
    loader = DataLoader(InferenceDataset(), batch_size=batch_size, shuffle=False, collate_fn=collate, num_workers=0)
    best: dict[str, dict] = {}
    model.eval()
    with torch.no_grad():
        for batch in loader:
            metadata = batch.pop("metadata")
            batch = {key: value.to(device) for key, value in batch.items()}
            probabilities = torch.softmax(model(**batch).logits, dim=1)[:, 1].detach().cpu().tolist()
            for item, probability in zip(metadata, probabilities):
                current = best.get(item["trial_id"])
                if current is None or probability > current["Probability"]:
                    best[item["trial_id"]] = {
                        "NCT_or_TrialID": item["trial_id"],
                        "Cancer_Type": item["cancer_type"],
                        "Label": str(item["label"]),
                        "Probability": float(probability),
                        "Chunks": item["chunk_count"],
                        "Max_Chunk_Index": item["chunk_index"],
                        "Max_Chunk_Start_Token": item["start_token"],
                        "Max_Chunk_End_Token": item["end_token"],
                    }
    if len(best) != len(rows):
        raise AssertionError("Trial-level aggregation lost records")
    return [best[row["NCT_or_TrialID"]] for row in rows]


def write_predictions(path: Path, rows: list[dict]) -> None:
    headers = [
        "NCT_or_TrialID", "Cancer_Type", "Label", "Probability", "Chunks",
        "Max_Chunk_Index", "Max_Chunk_Start_Token", "Max_Chunk_End_Token",
    ]
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def artifact_hashes(directory: Path) -> list[dict]:
    output = []
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.name != "training_run_manifest.json":
            output.append({"path": str(path.relative_to(directory)), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return output


def train_candidates(
    release_path: Path,
    config_path: Path,
    output_dir: Path,
    selected_seeds: list[int] | None,
    resume: bool = False,
) -> dict:
    rows, config, split_manifest = validate_inputs(release_path, config_path)
    if output_dir.exists() and not resume:
        raise FileExistsError(f"Refusing to overwrite training output: {output_dir}")
    np, torch, transformers, DataLoader, Dataset, AutoModel, AutoTokenizer, Trainer, TrainingArguments = require_ml_dependencies()
    seeds = selected_seeds or [int(value) for value in config["training"]["seeds"]]
    unknown = sorted(set(seeds) - {int(value) for value in config["training"]["seeds"]})
    if unknown:
        raise ValueError(f"Seeds are not prespecified in the training configuration: {unknown}")
    train_rows = [row for row in rows if row["Split"] == "train"]
    calibration_rows = [row for row in rows if row["Split"] == "calibration"]
    # The internal_test split is intentionally not accessed here.
    output_dir.mkdir(parents=True, exist_ok=resume)
    root_manifest = {
        "pipeline_version": "1.0.0",
        "development_release_sha256": sha256_file(release_path),
        "split_manifest_sha256": sha256_file(release_path.parent / "v8_split_manifest.json"),
        "training_config_sha256": sha256_file(config_path),
        "base_model": config["base_model"],
        "base_model_revision": config["base_model_revision"],
        "seeds": seeds,
        "internal_test_accessed": False,
        "challenge_data_accessed": False,
        "candidate_manifests": [],
    }

    class ChunkDataset(Dataset):
        def __init__(self, records):
            self.records = records

        def __len__(self):
            return len(self.records)

        def __getitem__(self, index):
            return self.records[index]

    for seed in seeds:
        candidate_dir = output_dir / f"candidate_seed_{seed}"
        model_dir = candidate_dir / "model"
        trainer_dir = candidate_dir / "trainer"
        candidate_manifest_path = candidate_dir / "training_run_manifest.json"
        if candidate_manifest_path.exists():
            completed = read_json(candidate_manifest_path)
            prediction_path = candidate_dir / "calibration_predictions.csv"
            if (
                completed.get("seed") != seed
                or completed.get("development_release_sha256") != sha256_file(release_path)
                or completed.get("training_config_sha256") != sha256_file(config_path)
                or not prediction_path.exists()
                or completed.get("calibration_predictions_sha256") != sha256_file(prediction_path)
            ):
                raise ValueError(f"Completed candidate provenance is invalid for seed {seed}")
            print(f"Seed {seed}: verified completed candidate; skipping immutable artifacts.")
            continue
        if candidate_dir.exists() and not resume:
            raise FileExistsError(f"Incomplete candidate exists; use --resume after inspection: {candidate_dir}")
        candidate_dir.mkdir(exist_ok=resume)
        setup_path = candidate_dir / "candidate_setup.json"
        expected_setup = {
            "seed": seed,
            "development_release_sha256": sha256_file(release_path),
            "training_config_sha256": sha256_file(config_path),
            "base_model": config["base_model"],
            "base_model_revision": config["base_model_revision"],
        }
        if setup_path.exists():
            if read_json(setup_path) != expected_setup:
                raise ValueError(f"Incomplete candidate setup does not match this run for seed {seed}")
        else:
            setup_path.write_text(json.dumps(expected_setup, indent=2) + "\n", encoding="utf-8")
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except AttributeError:
            pass

        tokenizer = AutoTokenizer.from_pretrained(
            config["base_model"], revision=config["base_model_revision"], use_fast=True
        )
        model = AutoModel.from_pretrained(
            config["base_model"], revision=config["base_model_revision"], num_labels=2
        )
        chunks = prepare_chunk_records(tokenizer, train_rows, config)

        def collate(features):
            padded = tokenizer.pad(
                [{"input_ids": item["input_ids"], "attention_mask": item["attention_mask"]} for item in features],
                padding=True,
                return_tensors="pt",
            )
            padded["labels"] = torch.tensor([item["label"] for item in features], dtype=torch.long)
            padded["sample_weight"] = torch.tensor([item["sample_weight"] for item in features], dtype=torch.float32)
            return padded

        class WeightedTrainer(Trainer):
            def compute_loss(self, model, inputs, return_outputs=False, **_kwargs):
                weights = inputs.pop("sample_weight")
                labels = inputs.pop("labels")
                outputs = model(**inputs)
                losses = torch.nn.functional.cross_entropy(outputs.logits, labels, reduction="none")
                # Chunks are sampled uniformly by the Trainer. Multiplying the
                # minibatch mean by N_chunks/N_trials gives an unbiased estimate
                # of the intended trial-level objective:
                #   sum(chunk_loss / chunks_in_trial) / N_trials.
                # Normalizing by the weights inside each minibatch would instead
                # make a long trial's influence depend on which other chunks
                # happened to share its minibatch.
                loss = torch.mean(losses * weights) * (len(chunks) / len(train_rows))
                return (loss, outputs) if return_outputs else loss

        training = config["training"]
        arguments = TrainingArguments(
            output_dir=str(trainer_dir),
            overwrite_output_dir=False,
            num_train_epochs=float(training["epochs"]),
            learning_rate=float(training["learning_rate"]),
            weight_decay=float(training["weight_decay"]),
            warmup_ratio=float(training["warmup_fraction"]),
            per_device_train_batch_size=int(training["train_batch_size"]),
            gradient_accumulation_steps=int(training["gradient_accumulation_steps"]),
            max_grad_norm=float(training["maximum_gradient_norm"]),
            save_strategy="epoch",
            save_total_limit=1,
            logging_strategy="steps",
            logging_steps=25,
            report_to=[],
            seed=seed,
            data_seed=seed,
            dataloader_num_workers=0,
            remove_unused_columns=False,
            save_safetensors=True,
        )
        trainer = WeightedTrainer(
            model=model,
            args=arguments,
            train_dataset=ChunkDataset(chunks),
            data_collator=collate,
        )
        checkpoints = sorted(trainer_dir.glob("checkpoint-*")) if trainer_dir.exists() else []
        train_result = trainer.train(resume_from_checkpoint=True if checkpoints else None)
        trainer.save_model(str(model_dir))
        tokenizer.save_pretrained(str(model_dir))
        predictions = score_records(
            trainer.model,
            tokenizer,
            calibration_rows,
            config,
            torch,
            DataLoader,
            Dataset,
            batch_size=int(training["train_batch_size"]),
        )
        prediction_path = candidate_dir / "calibration_predictions.csv"
        write_predictions(prediction_path, predictions)
        manifest = {
            "candidate_version": "1.0.0",
            "seed": seed,
            "base_model": config["base_model"],
            "base_model_revision": config["base_model_revision"],
            "development_release_sha256": sha256_file(release_path),
            "training_config_sha256": sha256_file(config_path),
            "training_records": len(train_rows),
            "training_chunks": len(chunks),
            "calibration_records_scored": len(calibration_rows),
            "internal_test_accessed": False,
            "challenge_data_accessed": False,
            "train_runtime_seconds": train_result.metrics.get("train_runtime"),
            "train_loss": train_result.metrics.get("train_loss"),
            "environment": {
                "python": sys.version,
                "platform": platform.platform(),
                "torch": torch.__version__,
                "transformers": transformers.__version__,
            },
            "calibration_predictions_sha256": sha256_file(prediction_path),
            "artifacts": artifact_hashes(candidate_dir),
        }
        candidate_manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    root_manifest["candidate_manifests"] = []
    configured_seeds = [int(value) for value in config["training"]["seeds"]]
    for seed in configured_seeds:
        candidate_manifest_path = output_dir / f"candidate_seed_{seed}" / "training_run_manifest.json"
        if candidate_manifest_path.exists():
            root_manifest["candidate_manifests"].append({
                "seed": seed,
                "path": str(candidate_manifest_path.relative_to(output_dir)),
                "sha256": sha256_file(candidate_manifest_path),
            })
    root_manifest["configured_seeds"] = configured_seeds
    root_manifest["completed_seeds"] = [item["seed"] for item in root_manifest["candidate_manifests"]]
    root_manifest["status"] = (
        "ALL_PRESPECIFIED_CANDIDATES_COMPLETE"
        if root_manifest["completed_seeds"] == configured_seeds
        else "PARTIAL_RESUMABLE_TRAINING_RUN"
    )
    root_manifest_path = output_dir / "v8_training_root_manifest.json"
    temporary_manifest_path = output_dir / "v8_training_root_manifest.json.tmp"
    temporary_manifest_path.write_text(json.dumps(root_manifest, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary_manifest_path, root_manifest_path)
    return root_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seeds", nargs="*", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    rows, config, manifest = validate_inputs(args.release, args.config)
    if args.audit_only:
        print(json.dumps({
            "status": "PASS",
            "development_release_sha256": sha256_file(args.release),
            "rows": len(rows),
            "split_counts": dict(Counter(row["Split"] for row in rows)),
            "challenge_text_overlap": manifest["invariants"]["challenge_text_overlap_in_release"],
            "base_model": config["base_model"],
            "base_model_revision": config["base_model_revision"],
            "training_started": False,
        }, indent=2))
        return
    if args.output_dir is None:
        parser.error("--output-dir is required unless --audit-only is used")
    result = train_candidates(args.release, args.config, args.output_dir, args.seeds, args.resume)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
