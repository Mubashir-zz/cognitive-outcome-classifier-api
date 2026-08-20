#!/usr/bin/env python3
"""Select one v8 seed and threshold using only frozen calibration predictions."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

from v8_model_core import confusion_metrics, select_threshold


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_predictions(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"NCT_or_TrialID", "Cancer_Type", "Label", "Probability"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Unexpected calibration prediction schema: {path}")
    if len({row["NCT_or_TrialID"] for row in rows}) != len(rows):
        raise ValueError("Calibration predictions contain duplicate trial IDs")
    return rows


def finite_or_low(value: float) -> float:
    return value if math.isfinite(value) else -math.inf


def json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value


def select(candidates_dir: Path, config_path: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite selection: {output}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    calibration = config["calibration"]
    candidate_results = []
    for seed in [int(value) for value in config["training"]["seeds"]]:
        candidate_dir = candidates_dir / f"candidate_seed_{seed}"
        manifest_path = candidate_dir / "training_run_manifest.json"
        prediction_path = candidate_dir / "calibration_predictions.csv"
        if not manifest_path.exists() or not prediction_path.exists():
            raise FileNotFoundError(f"Missing frozen candidate artifacts for seed {seed}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("seed") != seed or manifest.get("internal_test_accessed") is not False:
            raise ValueError(f"Candidate manifest is invalid for seed {seed}")
        if manifest.get("calibration_predictions_sha256") != sha256(prediction_path):
            raise ValueError(f"Calibration prediction hash mismatch for seed {seed}")
        rows = read_predictions(prediction_path)
        cns = [row for row in rows if row["Cancer_Type"] == "CNS"]
        if not cns or {row["Label"] for row in cns} != {"0", "1"}:
            raise ValueError(f"CNS calibration data for seed {seed} lacks both labels")
        selection = select_threshold(
            [int(row["Label"]) for row in cns],
            [float(row["Probability"]) for row in cns],
            minimum_sensitivity=float(calibration["minimum_sensitivity"]),
            minimum_specificity=float(calibration["minimum_specificity"]),
        )
        threshold = float(selection["selected"]["threshold"])
        overall = confusion_metrics(
            [int(row["Label"]) for row in rows],
            [float(row["Probability"]) for row in rows],
            threshold,
        )
        candidate_results.append({
            "seed": seed,
            "candidate_dir": str(candidate_dir.resolve()),
            "manifest_sha256": sha256(manifest_path),
            "calibration_predictions_sha256": sha256(prediction_path),
            "cns_calibration_records": len(cns),
            "cns_threshold_selection": selection,
            "overall_calibration_at_selected_threshold": overall,
        })
    chosen = max(
        candidate_results,
        key=lambda item: (
            int(item["cns_threshold_selection"]["feasible_threshold_exists"]),
            finite_or_low(float(item["cns_threshold_selection"]["selected"]["mcc"])),
            finite_or_low(float(item["cns_threshold_selection"]["selected"]["balanced_accuracy"])),
            finite_or_low(float(item["cns_threshold_selection"]["selected"]["sensitivity"])),
            finite_or_low(float(item["cns_threshold_selection"]["selected"]["specificity"])),
            -item["seed"],
        ),
    )
    result = {
        "selection_version": "1.0.0",
        "status": "CANDIDATE_SELECTED_ON_CALIBRATION_ONLY",
        "training_config_sha256": sha256(config_path),
        "selection_scope": "CNS calibration records",
        "selected_seed": chosen["seed"],
        "selected_model_dir": str(Path(chosen["candidate_dir"]) / "model"),
        "selected_threshold": chosen["cns_threshold_selection"]["selected"]["threshold"],
        "feasible_threshold_exists": chosen["cns_threshold_selection"]["feasible_threshold_exists"],
        "internal_test_accessed": False,
        "challenge_data_accessed": False,
        "candidates": candidate_results,
    }
    safe_result = json_safe(result)
    output.write_text(json.dumps(safe_result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(safe_result, indent=2, allow_nan=False))
    return safe_result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    select(args.candidates_dir, args.config, args.output)


if __name__ == "__main__":
    main()
