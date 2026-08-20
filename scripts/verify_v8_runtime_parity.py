#!/usr/bin/env python3
"""Compare staging v8 inference with the frozen analytic scorer on exact texts."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer


SCRIPT = Path(__file__).resolve()
STAGING_ROOT = SCRIPT.parents[1]
PROJECT_ROOT = STAGING_ROOT.parent if STAGING_ROOT.name == "staging_classifier_v2" else STAGING_ROOT
sys.path.insert(0, str(STAGING_ROOT))
training_root = PROJECT_ROOT / "training"
sys.path.insert(0, str(training_root if training_root.is_dir() else PROJECT_ROOT))

from app.model_runtime import V8ChunkedRuntime, load_v8_artifact_spec
from quantize_v8_model import score_fixed_length


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_fixture(path: Path) -> list[dict[str, str]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            record_id = str(item.get("record_id", "")).strip()
            text = item.get("outcome_text")
            if not record_id or not isinstance(text, str) or not text.strip():
                raise ValueError(f"Invalid parity fixture at line {line_number}")
            rows.append({"record_id": record_id, "outcome_text": text})
    if not rows or len({row["record_id"] for row in rows}) != len(rows):
        raise ValueError("Parity fixture must contain unique records")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--manifest-filename", default="quantization_manifest.json")
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--maximum-probability-delta", type=float, default=1e-12)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite parity evidence: {args.output}")
    fixture = read_fixture(args.fixture)
    spec = load_v8_artifact_spec(args.artifact_dir, args.manifest_filename, args.manifest_sha256)
    tokenizer = AutoTokenizer.from_pretrained(str(spec.tokenizer_dir), local_files_only=True, use_fast=True)
    model = torch.jit.load(str(spec.model_path), map_location="cpu").eval()
    runtime = V8ChunkedRuntime(
        spec, tokenizer, model, torch,
        inference_batch_size=args.batch_size,
        maximum_input_tokens=max(50_000, max(len(tokenizer.encode(row["outcome_text"], add_special_tokens=False, truncation=False)) for row in fixture)),
    )
    api_scores = runtime.score([row["outcome_text"] for row in fixture])
    analytic_rows = [{
        "NCT_or_TrialID": row["record_id"],
        "Cancer_Type": "CNS",
        "Outcome_Text": row["outcome_text"],
        "Label": "No",
    } for row in fixture]
    config = {
        "max_sequence_tokens": spec.maximum_sequence_tokens,
        "content_tokens_per_chunk": spec.content_tokens,
        "chunk_overlap_tokens": spec.overlap_tokens,
    }
    analytic_scores = score_fixed_length(
        model, tokenizer, analytic_rows, config, torch, DataLoader, Dataset, args.batch_size
    )
    deltas = []
    agreements = []
    record_results = []
    for fixture_row, api, analytic in zip(fixture, api_scores, analytic_scores):
        analytic_probability = float(analytic["Probability"])
        delta = abs(api.probability - analytic_probability)
        deltas.append(delta)
        agreement = (api.probability >= spec.threshold) == (analytic_probability >= spec.threshold)
        agreements.append(agreement)
        record_results.append({
            "record_id": fixture_row["record_id"],
            "source_text_sha256": hashlib.sha256(fixture_row["outcome_text"].encode("utf-8")).hexdigest(),
            "analytic_probability": analytic_probability,
            "analytic_class": analytic_probability >= spec.threshold,
            "runtime_probability": api.probability,
            "runtime_class": api.probability >= spec.threshold,
            "absolute_probability_delta": delta,
            "class_agreement": agreement,
        })
    result = {
        "status": "PASS" if max(deltas) <= args.maximum_probability_delta and all(agreements) else "FAIL",
        "records": len(fixture),
        "fixture_sha256": sha256(args.fixture),
        "quantization_manifest_sha256": spec.manifest_sha256,
        "serialized_torchscript_sha256": spec.model_sha256,
        "tokenizer_sha256": spec.tokenizer_sha256,
        "threshold": spec.threshold,
        "class_agreement": sum(agreements) / len(agreements),
        "maximum_absolute_probability_delta": max(deltas),
        "mean_absolute_probability_delta": sum(deltas) / len(deltas),
        "allowed_maximum_probability_delta": args.maximum_probability_delta,
        "record_results": record_results,
        "raw_outcome_text_stored": False,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if result["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
