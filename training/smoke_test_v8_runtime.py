#!/usr/bin/env python3
"""One-step runtime check for the pinned v8 model stack.

This is not model development and cannot produce a candidate. It reads only
training records, runs one forward/backward/optimizer step, and writes a
manifest explicitly marked NOT_FOR_RELEASE.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

from train_v8_chunked_bert import prepare_chunk_records, require_ml_dependencies, sha256_file, validate_inputs


def run(release_path: Path, config_path: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite smoke-test evidence: {output}")
    rows, config, _manifest = validate_inputs(release_path, config_path)
    train_rows = [row for row in rows if row["Split"] == "train"][:2]
    # Calibration, internal-test, and challenge records are never accessed.
    np, torch, transformers, _DataLoader, _Dataset, AutoModel, AutoTokenizer, _Trainer, _TrainingArguments = require_ml_dependencies()
    started = time.time()
    tokenizer = AutoTokenizer.from_pretrained(
        config["base_model"], revision=config["base_model_revision"], use_fast=True
    )
    model = AutoModel.from_pretrained(
        config["base_model"], revision=config["base_model_revision"], num_labels=2
    )
    model.train()
    chunks = prepare_chunk_records(tokenizer, train_rows, config)[:2]
    padded = tokenizer.pad(
        [{"input_ids": item["input_ids"], "attention_mask": item["attention_mask"]} for item in chunks],
        padding=True,
        return_tensors="pt",
    )
    labels = torch.tensor([item["label"] for item in chunks], dtype=torch.long)
    weights = torch.tensor([item["sample_weight"] for item in chunks], dtype=torch.float32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["training"]["learning_rate"]))
    optimizer.zero_grad(set_to_none=True)
    outputs = model(**padded)
    losses = torch.nn.functional.cross_entropy(outputs.logits, labels, reduction="none")
    loss = torch.mean(losses * weights) * (len(chunks) / len(train_rows))
    loss.backward()
    optimizer.step()
    result = {
        "smoke_test_version": "1.0.0",
        "status": "PASS_NOT_FOR_RELEASE",
        "purpose": "Verify pinned dependencies, model download, tokenizer, full forward/backward pass, and optimizer step",
        "development_release_sha256": sha256_file(release_path),
        "training_config_sha256": sha256_file(config_path),
        "base_model": config["base_model"],
        "base_model_revision": config["base_model_revision"],
        "training_records_touched": len(train_rows),
        "chunks_touched": len(chunks),
        "calibration_accessed": False,
        "internal_test_accessed": False,
        "challenge_data_accessed": False,
        "candidate_artifact_created": False,
        "loss_is_finite": bool(torch.isfinite(loss).item()),
        "elapsed_seconds": time.time() - started,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "numpy": np.__version__,
            "mps_available": bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()),
            "cuda_available": bool(torch.cuda.is_available()),
        },
    }
    if not result["loss_is_finite"]:
        raise RuntimeError("Smoke-test loss is not finite")
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.release, args.config, args.output)


if __name__ == "__main__":
    main()
