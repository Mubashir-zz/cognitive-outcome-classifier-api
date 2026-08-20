#!/usr/bin/env python3
"""Audit the complete CNS v8 evidence chain without reading the sealed answer key."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import stat
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path


FROZEN_HEADERS = [
    "Row_ID", "Outcome_Text_Part_1", "Outcome_Text_Part_2",
    "Measures_Cognition_Y_N", "Reviewer_Confidence_High_Medium_Low", "Notes",
]
OVERLAP_HEADERS = ["Row_ID", "Training_Text_Overlap", "Matched_Training_Rows", "Normalized_Text_SHA256"]
PREDICTION_HEADERS = ["Row_ID", "V8_Probability", "V8_Prediction"]
EXPECTED_METRICS = {"Sensitivity", "Specificity", "Balanced_Accuracy", "MCC"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def contained(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"Artifact path escapes its root: {relative}") from exc
    return candidate


def finite_probability(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"Invalid probability: {value!r}")
    return number


@dataclass
class Stage:
    name: str
    status: str = "PASS"
    checks: dict[str, bool] = field(default_factory=dict)
    evidence: dict = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    def require(self, name: str, condition: bool, message: str) -> None:
        passed = bool(condition)
        self.checks[name] = passed
        if not passed:
            self.status = "FAIL"
            self.errors.append(message)

    def missing(self, message: str) -> None:
        self.status = "MISSING"
        self.errors.append(message)


def audit_development(workspace: Path) -> Stage:
    stage = Stage("frozen_development_inputs")
    release = workspace / "v8_development_release_2026-08-20" / "v8_development_release.csv"
    config_path = workspace / "v8_training_config.json"
    split_manifest_path = release.parent / "v8_split_manifest.json"
    for path in (release, config_path, split_manifest_path):
        if not path.is_file():
            stage.missing(f"Missing frozen development input: {path}")
            return stage
    try:
        config = load_json(config_path)
        split_manifest = load_json(split_manifest_path)
        headers, rows = read_csv(release)
        stage.require("release_hash_bound_to_config", config.get("development_release_sha256") == sha256(release), "Development release hash differs from training configuration")
        stage.require("release_hash_bound_to_split_manifest", split_manifest.get("data_file", {}).get("sha256") == sha256(release), "Development release hash differs from split manifest")
        stage.require("required_release_columns", {"NCT_or_TrialID", "Outcome_Text", "Label", "Cancer_Type", "Normalized_Text_SHA256", "Split"}.issubset(headers), "Development release schema is incomplete")
        stage.require("unique_trial_ids", len({row["NCT_or_TrialID"] for row in rows}) == len(rows), "Development release has duplicate trial IDs")
        stage.require("unique_normalized_text", len({row["Normalized_Text_SHA256"] for row in rows}) == len(rows), "Development release has duplicate normalized text")
        stage.require("valid_splits", {row["Split"] for row in rows} == {"train", "calibration", "internal_test"}, "Development release splits are invalid")
        stage.require("valid_labels", {row["Label"] for row in rows} == {"Yes", "No"}, "Development release labels are invalid")
        stage.require("challenge_overlap_zero", split_manifest.get("invariants", {}).get("challenge_text_overlap_in_release") == 0, "Development release contains challenge-text overlap")
        stage.evidence = {
            "release_sha256": sha256(release),
            "config_sha256": sha256(config_path),
            "split_manifest_sha256": sha256(split_manifest_path),
            "records": len(rows),
            "split_counts": {name: sum(row["Split"] == name for row in rows) for name in ("train", "calibration", "internal_test")},
            "configured_seeds": config.get("training", {}).get("seeds", []),
        }
    except Exception as exc:
        stage.status = "FAIL"
        stage.errors.append(f"Development-input audit failed: {exc}")
    return stage


def audit_training(workspace: Path, development: Stage) -> Stage:
    stage = Stage("five_seed_training")
    root = workspace / "v8_training_candidates_2026-08-20"
    manifest_path = root / "v8_training_root_manifest.json"
    if not manifest_path.is_file():
        stage.missing("Five-seed root manifest is absent")
        stage.evidence = {"candidate_setup_directories": sorted(path.name for path in root.glob("candidate_seed_*") if path.is_dir()) if root.is_dir() else []}
        return stage
    try:
        manifest = load_json(manifest_path)
        config_path = workspace / "v8_training_config.json"
        release = workspace / "v8_development_release_2026-08-20" / "v8_development_release.csv"
        config = load_json(config_path)
        seeds = [int(value) for value in config["training"]["seeds"]]
        stage.require("all_seeds_complete", manifest.get("status") == "ALL_PRESPECIFIED_CANDIDATES_COMPLETE", "Training root is not complete")
        stage.require("configured_seed_order", manifest.get("configured_seeds") == seeds, "Configured seeds differ from frozen configuration")
        stage.require("completed_seed_order", manifest.get("completed_seeds") == seeds, "Not all configured seeds completed in order")
        stage.require("release_binding", manifest.get("development_release_sha256") == sha256(release), "Training root release hash mismatch")
        stage.require("config_binding", manifest.get("training_config_sha256") == sha256(config_path), "Training root config hash mismatch")
        stage.require("internal_test_untouched", manifest.get("internal_test_accessed") is False, "Training accessed the internal test")
        stage.require("challenge_untouched", manifest.get("challenge_data_accessed") is False, "Training accessed challenge data")
        entries = manifest.get("candidate_manifests", [])
        stage.require("candidate_entry_count", len(entries) == len(seeds), "Training root has the wrong candidate-manifest count")
        entry_by_seed = {int(item.get("seed")): item for item in entries}
        stage.require("candidate_seed_set", set(entry_by_seed) == set(seeds), "Training root candidate seed set is wrong")
        verified = []
        for seed in seeds:
            item = entry_by_seed.get(seed, {})
            candidate_dir = root / f"candidate_seed_{seed}"
            candidate_manifest_path = contained(root, str(item.get("path", "")))
            if not candidate_manifest_path.is_file():
                raise FileNotFoundError(f"Missing candidate manifest for seed {seed}")
            if sha256(candidate_manifest_path) != item.get("sha256"):
                raise ValueError(f"Candidate manifest hash mismatch for seed {seed}")
            candidate = load_json(candidate_manifest_path)
            if candidate.get("seed") != seed or candidate.get("development_release_sha256") != sha256(release) or candidate.get("training_config_sha256") != sha256(config_path):
                raise ValueError(f"Candidate provenance mismatch for seed {seed}")
            if candidate.get("internal_test_accessed") is not False or candidate.get("challenge_data_accessed") is not False:
                raise ValueError(f"Candidate {seed} accessed held-out data")
            prediction = candidate_dir / "calibration_predictions.csv"
            if not prediction.is_file() or sha256(prediction) != candidate.get("calibration_predictions_sha256"):
                raise ValueError(f"Candidate calibration predictions mismatch for seed {seed}")
            listed = set()
            for artifact in candidate.get("artifacts", []):
                path = contained(candidate_dir, str(artifact.get("path", "")))
                if not path.is_file() or path.stat().st_size != artifact.get("bytes") or sha256(path) != artifact.get("sha256"):
                    raise ValueError(f"Candidate artifact mismatch for seed {seed}: {artifact.get('path')}")
                listed.add(str(path.relative_to(candidate_dir)))
            actual = {
                str(path.relative_to(candidate_dir))
                for path in candidate_dir.rglob("*")
                if path.is_file() and path.name != "training_run_manifest.json"
            }
            if listed != actual:
                raise ValueError(f"Candidate artifact inventory mismatch for seed {seed}")
            verified.append(seed)
        stage.require("all_candidate_artifacts_verified", verified == seeds, "Candidate artifact verification was incomplete")
        stage.evidence = {"root_manifest_sha256": sha256(manifest_path), "verified_seeds": verified}
    except Exception as exc:
        stage.status = "FAIL"
        stage.errors.append(f"Training audit failed: {exc}")
    return stage


def audit_selection(workspace: Path) -> Stage:
    stage = Stage("calibration_only_selection")
    root = workspace / "v8_training_candidates_2026-08-20"
    path = root / "v8_candidate_selection.json"
    if not path.is_file():
        stage.missing("Calibration-only candidate selection is absent")
        return stage
    try:
        selection = load_json(path)
        config_path = workspace / "v8_training_config.json"
        config = load_json(config_path)
        seeds = [int(value) for value in config["training"]["seeds"]]
        stage.require("selection_status", selection.get("status") == "CANDIDATE_SELECTED_ON_CALIBRATION_ONLY", "Selection status is invalid")
        stage.require("config_binding", selection.get("training_config_sha256") == sha256(config_path), "Selection config hash mismatch")
        stage.require("candidate_set", {int(item.get("seed")) for item in selection.get("candidates", [])} == set(seeds), "Selection does not include every configured seed")
        stage.require("selected_seed_valid", selection.get("selected_seed") in seeds, "Selected seed is not prespecified")
        stage.require("feasible_threshold", selection.get("feasible_threshold_exists") is True, "No threshold met the frozen calibration constraints")
        stage.require("threshold_valid", isinstance(selection.get("selected_threshold"), (int, float)) and 0 <= selection["selected_threshold"] <= 1, "Selected threshold is invalid")
        stage.require("internal_test_untouched", selection.get("internal_test_accessed") is False, "Selection accessed internal test")
        stage.require("challenge_untouched", selection.get("challenge_data_accessed") is False, "Selection accessed challenge data")
        for item in selection.get("candidates", []):
            seed = int(item["seed"])
            candidate_dir = root / f"candidate_seed_{seed}"
            manifest_path = candidate_dir / "training_run_manifest.json"
            prediction_path = candidate_dir / "calibration_predictions.csv"
            if sha256(manifest_path) != item.get("manifest_sha256") or sha256(prediction_path) != item.get("calibration_predictions_sha256"):
                raise ValueError(f"Selection candidate hash mismatch for seed {seed}")
        stage.evidence = {"selection_sha256": sha256(path), "selected_seed": selection.get("selected_seed"), "selected_threshold": selection.get("selected_threshold")}
    except Exception as exc:
        stage.status = "FAIL"
        stage.errors.append(f"Selection audit failed: {exc}")
    return stage


def audit_quantization(workspace: Path) -> Stage:
    stage = Stage("serialized_quantized_artifact")
    root = workspace / "v8_training_candidates_2026-08-20"
    quant_dir = root / "v8_quantized_selected"
    manifest_path = quant_dir / "quantization_manifest.json"
    selection_path = root / "v8_candidate_selection.json"
    if not manifest_path.is_file():
        stage.missing("Passing serialized quantization manifest is absent")
        return stage
    try:
        manifest = load_json(manifest_path)
        config_path = workspace / "v8_training_config.json"
        release = workspace / "v8_development_release_2026-08-20" / "v8_development_release.csv"
        config = load_json(config_path)
        selection = load_json(selection_path)
        stage.require("quantization_status", manifest.get("status") == "QUANTIZED_CANDIDATE_PASSED_CALIBRATION_PARITY", "Quantization status is invalid")
        stage.require("selection_binding", manifest.get("selection_sha256") == sha256(selection_path), "Quantization selection hash mismatch")
        stage.require("config_binding", manifest.get("training_config_sha256") == sha256(config_path), "Quantization config hash mismatch")
        stage.require("release_binding", manifest.get("development_release_sha256") == sha256(release), "Quantization release hash mismatch")
        stage.require("seed_binding", manifest.get("selected_seed") == selection.get("selected_seed"), "Quantization seed mismatch")
        stage.require("threshold_binding", manifest.get("selected_threshold") == selection.get("selected_threshold"), "Quantization threshold mismatch")
        for name in ("eager_quantized_parity", "serialized_torchscript_parity"):
            parity = manifest.get(name, {})
            stage.require(f"{name}_passed", parity.get("parity_passed") is True, f"{name} did not pass")
            stage.require(f"{name}_agreement", parity.get("class_agreement") == config["quantization"]["required_class_agreement"], f"{name} class agreement differs from frozen requirement")
            stage.require(f"{name}_delta", parity.get("maximum_absolute_probability_delta", math.inf) <= config["quantization"]["maximum_absolute_probability_delta"], f"{name} probability delta exceeds frozen tolerance")
        stage.require("internal_test_untouched", manifest.get("internal_test_accessed") is False, "Quantization accessed internal test")
        stage.require("challenge_untouched", manifest.get("challenge_data_accessed") is False, "Quantization accessed challenge data")
        artifacts = manifest.get("artifacts", {})
        model_info = artifacts.get("torchscript", {})
        model = contained(quant_dir, str(model_info.get("path", "")))
        stage.require("torchscript_verified", model.is_file() and model.stat().st_size == model_info.get("bytes") and sha256(model) == model_info.get("sha256"), "Serialized TorchScript hash or size mismatch")
        contract_info = artifacts.get("inference_contract", {})
        contract = contained(quant_dir, str(contract_info.get("path", "")))
        stage.require("contract_verified", contract.is_file() and sha256(contract) == contract_info.get("sha256"), "Inference contract hash mismatch")
        tokenizer_dir = contained(quant_dir, str(artifacts.get("tokenizer_directory", "")))
        listed = set()
        for item in artifacts.get("tokenizer_files", []):
            token_path = contained(tokenizer_dir, str(item.get("path", "")))
            if not token_path.is_file() or token_path.stat().st_size != item.get("bytes") or sha256(token_path) != item.get("sha256"):
                raise ValueError(f"Tokenizer artifact mismatch: {item.get('path')}")
            listed.add(str(token_path.relative_to(tokenizer_dir)))
        actual = {str(path.relative_to(tokenizer_dir)) for path in tokenizer_dir.rglob("*") if path.is_file()}
        stage.require("complete_tokenizer_verified", bool(listed) and listed == actual, "Tokenizer inventory is incomplete or contains extras")
        stage.evidence = {"quantization_manifest_sha256": sha256(manifest_path), "torchscript_sha256": sha256(model) if model.is_file() else None}
    except Exception as exc:
        stage.status = "FAIL"
        stage.errors.append(f"Quantization audit failed: {exc}")
    return stage


def audit_internal_test(workspace: Path) -> Stage:
    stage = Stage("one_time_internal_test")
    root = workspace / "v8_training_candidates_2026-08-20"
    output_dir = root / "v8_internal_test_once"
    result_path = output_dir / "internal_test_evaluation.json"
    if not result_path.is_file():
        stage.missing("One-time exact-artifact internal-test result is absent")
        return stage
    try:
        result = load_json(result_path)
        selection_path = root / "v8_candidate_selection.json"
        quant_path = root / "v8_quantized_selected" / "quantization_manifest.json"
        config_path = workspace / "v8_training_config.json"
        release = workspace / "v8_development_release_2026-08-20" / "v8_development_release.csv"
        quant = load_json(quant_path)
        predictions = output_dir / "internal_test_predictions.csv"
        lock = output_dir / "INTERNAL_TEST_EVALUATED.lock"
        stage.require("internal_status", result.get("status") == "INTERNAL_TEST_EVALUATED_ONCE", "Internal-test status is invalid")
        stage.require("selection_binding", result.get("selection_sha256") == sha256(selection_path), "Internal-test selection hash mismatch")
        stage.require("quantization_binding", result.get("quantization_manifest_sha256") == sha256(quant_path), "Internal-test quantization hash mismatch")
        stage.require("model_binding", result.get("evaluated_torchscript_sha256") == quant.get("artifacts", {}).get("torchscript", {}).get("sha256"), "Internal-test model hash mismatch")
        stage.require("config_binding", result.get("training_config_sha256") == sha256(config_path), "Internal-test config hash mismatch")
        stage.require("release_binding", result.get("development_release_sha256") == sha256(release), "Internal-test release hash mismatch")
        stage.require("prediction_hash", predictions.is_file() and result.get("internal_test_predictions_sha256") == sha256(predictions), "Internal-test prediction hash mismatch")
        stage.require("record_count", result.get("internal_test_records") == sum(row["Split"] == "internal_test" for row in read_csv(release)[1]), "Internal-test record count mismatch")
        stage.require("challenge_untouched", result.get("challenge_data_accessed") is False, "Internal-test evaluation accessed challenge data")
        stage.require("promotion_not_decided", result.get("production_promotion_decided") is False, "Internal-test evaluation decided production promotion")
        lock_data = load_json(lock) if lock.is_file() else {}
        stage.require("immutable_lock", lock_data.get("result_sha256") == sha256(result_path) and lock_data.get("selection_sha256") == sha256(selection_path), "Internal-test immutable lock is absent or mismatched")
        stage.evidence = {"internal_test_result_sha256": sha256(result_path), "records": result.get("internal_test_records")}
    except Exception as exc:
        stage.status = "FAIL"
        stage.errors.append(f"Internal-test audit failed: {exc}")
    return stage


def audit_human_freeze(workspace: Path) -> Stage:
    stage = Stage("frozen_human_adjudication")
    frozen = workspace / "CNS_challenge_set_300_HUMAN_FROZEN.csv"
    manifest_path = frozen.with_suffix(frozen.suffix + ".manifest.json")
    checksum_path = frozen.with_suffix(frozen.suffix + ".sha256")
    if not frozen.is_file():
        stage.missing("Frozen 300-record human adjudication is absent")
        return stage
    try:
        manifest = load_json(manifest_path)
        headers, rows = read_csv(frozen)
        ids = [row.get("Row_ID", "") for row in rows]
        labels = [row.get("Measures_Cognition_Y_N", "").strip() for row in rows]
        confidence = [row.get("Reviewer_Confidence_High_Medium_Low", "").strip() for row in rows]
        stage.require("frozen_status", manifest.get("status") == "FROZEN_HUMAN_LABELS", "Human-label manifest status is invalid")
        stage.require("schema", headers == FROZEN_HEADERS, "Frozen human-label schema is invalid")
        stage.require("records", len(rows) == 300 and manifest.get("records") == 300, "Frozen human-label record count is not 300")
        stage.require("unique_ids", len(set(ids)) == 300 and all(ids), "Frozen human labels have duplicate or blank IDs")
        stage.require("valid_labels", all(value in {"", "Yes", "No"} for value in labels), "Frozen human labels contain invalid values")
        stage.require("valid_confidence", all(value in {"High", "Medium", "Low"} for value in confidence), "Frozen human labels contain invalid confidence")
        stage.require("ambiguous_rule", all(label or (conf == "Low" and row.get("Notes", "").strip()) for row, label, conf in zip(rows, labels, confidence)), "Ambiguous human labels violate the Low-confidence note rule")
        stage.require("frozen_hash", manifest.get("frozen_sha256") == sha256(frozen), "Frozen human-label hash mismatch")
        stage.require("checksum", checksum_path.is_file() and checksum_path.read_text(encoding="utf-8").strip() == f"{sha256(frozen)}  {frozen.name}", "Frozen human-label checksum file mismatch")
        stage.require("read_only", frozen.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH) == 0, "Frozen human-label file is writable")
        stage.require("freeze_did_not_open_key", manifest.get("sealed_key_opened_by_this_script") is False, "Freeze provenance reports answer-key access")
        stage.evidence = {"frozen_labels_sha256": sha256(frozen), "adjudicated": sum(bool(value) for value in labels), "ambiguous": sum(not value for value in labels)}
    except Exception as exc:
        stage.status = "FAIL"
        stage.errors.append(f"Human-freeze audit failed: {exc}")
    return stage


def audit_overlap(workspace: Path) -> Stage:
    stage = Stage("post_freeze_text_overlap")
    output = workspace / "CNS_challenge_set_300_POST_FREEZE_TEXT_OVERLAP.csv"
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    if not output.is_file():
        stage.missing("Post-freeze training-text-overlap artifact is absent")
        return stage
    try:
        manifest = load_json(manifest_path)
        headers, rows = read_csv(output)
        frozen = workspace / "CNS_challenge_set_300_HUMAN_FROZEN.csv"
        training = workspace / "CROSS_CANCER_TRAINING_SET_v7.csv"
        if not training.is_file():
            alternative = Path("/Users/mac/Documents/Trial study/NLP project/cross cancer training /CROSS_CANCER_TRAINING_SET_v7.csv")
            training = alternative
        stage.require("overlap_status", manifest.get("status") == "POST_FREEZE_TRAINING_TEXT_OVERLAP_FLAGS", "Overlap status is invalid")
        stage.require("schema", headers == OVERLAP_HEADERS, "Overlap schema is invalid")
        stage.require("records", len(rows) == 300 and manifest.get("records") == 300, "Overlap record count is not 300")
        stage.require("unique_ids", len({row["Row_ID"] for row in rows}) == 300, "Overlap rows are not unique")
        frozen_ids = {row["Row_ID"] for row in read_csv(frozen)[1]}
        stage.require("frozen_id_set", {row["Row_ID"] for row in rows} == frozen_ids, "Overlap IDs differ from frozen human-label IDs")
        stage.require("output_hash", manifest.get("output_sha256") == sha256(output), "Overlap output hash mismatch")
        stage.require("frozen_binding", manifest.get("frozen_labels_sha256") == sha256(frozen), "Overlap frozen-label hash mismatch")
        stage.require("training_binding", training.is_file() and manifest.get("training_data_sha256") == sha256(training), "Overlap training-data hash mismatch")
        stage.require("post_freeze_only", manifest.get("created_only_after_frozen_human_labels") is True, "Overlap provenance is not post-freeze")
        stage.require("key_untouched", manifest.get("sealed_key_opened") is False, "Overlap provenance reports answer-key access")
        stage.evidence = {"overlap_sha256": sha256(output), "overlap_records": manifest.get("overlap_records"), "overlap_rate": manifest.get("overlap_rate")}
    except Exception as exc:
        stage.status = "FAIL"
        stage.errors.append(f"Overlap audit failed: {exc}")
    return stage


def audit_challenge_predictions(workspace: Path) -> Stage:
    stage = Stage("frozen_v8_challenge_predictions")
    output = workspace / "CNS_challenge_set_300_V8_FROZEN_PREDICTIONS.csv"
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    if not output.is_file():
        stage.missing("Frozen v8 challenge predictions are absent")
        return stage
    try:
        manifest = load_json(manifest_path)
        headers, rows = read_csv(output)
        frozen = workspace / "CNS_challenge_set_300_HUMAN_FROZEN.csv"
        quant_path = workspace / "v8_training_candidates_2026-08-20" / "v8_quantized_selected" / "quantization_manifest.json"
        quant = load_json(quant_path)
        threshold = float(quant["selected_threshold"])
        stage.require("prediction_status", manifest.get("status") == "FROZEN_V8_CHALLENGE_PREDICTIONS", "Challenge-prediction status is invalid")
        stage.require("schema", headers == PREDICTION_HEADERS, "Challenge-prediction schema is invalid")
        stage.require("records", len(rows) == 300 and manifest.get("records") == 300, "Challenge-prediction record count is not 300")
        stage.require("unique_ids", len({row["Row_ID"] for row in rows}) == 300, "Challenge predictions contain duplicate IDs")
        frozen_manifest_path = frozen.with_suffix(frozen.suffix + ".manifest.json")
        frozen_ids = {row["Row_ID"] for row in read_csv(frozen)[1]}
        stage.require("frozen_id_set", {row["Row_ID"] for row in rows} == frozen_ids, "Challenge-prediction IDs differ from frozen human-label IDs")
        for row in rows:
            probability = finite_probability(row["V8_Probability"])
            if row["V8_Prediction"] != str(int(probability >= threshold)):
                raise ValueError(f"Thresholded prediction mismatch at {row['Row_ID']}")
        stage.require("output_hash", manifest.get("output_sha256") == sha256(output), "Challenge-prediction output hash mismatch")
        stage.require("frozen_binding", manifest.get("frozen_labels_sha256") == sha256(frozen), "Challenge-prediction frozen-label hash mismatch")
        stage.require("frozen_manifest_binding", manifest.get("frozen_labels_manifest_sha256") == sha256(frozen_manifest_path), "Challenge-prediction frozen-label manifest hash mismatch")
        stage.require("quantization_binding", manifest.get("quantization_manifest_sha256") == sha256(quant_path), "Challenge-prediction quantization hash mismatch")
        stage.require("model_binding", manifest.get("serialized_torchscript_sha256") == quant.get("artifacts", {}).get("torchscript", {}).get("sha256"), "Challenge-prediction model hash mismatch")
        stage.require("config_binding", manifest.get("training_config_sha256") == sha256(workspace / "v8_training_config.json"), "Challenge-prediction config hash mismatch")
        stage.require("release_binding", manifest.get("development_release_sha256") == sha256(workspace / "v8_development_release_2026-08-20" / "v8_development_release.csv"), "Challenge-prediction release hash mismatch")
        stage.require("threshold_binding", manifest.get("selected_threshold") == threshold, "Challenge-prediction threshold mismatch")
        stage.require("truth_not_used", manifest.get("human_truth_used_for_scoring") is False and manifest.get("reviewer_confidence_used_for_scoring") is False, "Human labels influenced challenge scoring")
        stage.require("identity_not_used", manifest.get("challenge_identity_used_for_scoring") is False, "Challenge identity influenced scoring")
        stage.require("key_untouched", manifest.get("sealed_key_opened_by_this_script") is False, "Challenge scorer reports answer-key access")
        stage.evidence = {"predictions_sha256": sha256(output), "quantization_manifest_sha256": sha256(quant_path)}
    except Exception as exc:
        stage.status = "FAIL"
        stage.errors.append(f"Challenge-prediction audit failed: {exc}")
    return stage


def audit_validation(workspace: Path) -> Stage:
    stage = Stage("design_weighted_external_validation")
    root = workspace / "cns_challenge_validation_results"
    summary_path = root / "validation_summary.json"
    metrics_path = root / "performance_metrics.csv"
    if not summary_path.is_file() or not metrics_path.is_file():
        stage.missing("Controlled-unblinding validation summary and metrics are absent")
        return stage
    try:
        summary = load_json(summary_path)
        _headers, metrics = read_csv(metrics_path)
        quant_path = workspace / "v8_training_candidates_2026-08-20" / "v8_quantized_selected" / "quantization_manifest.json"
        stage.require("validation_status", summary.get("status") == "COMPLETE", "External validation status is not complete")
        stage.require("records", summary.get("records") == 300, "External validation did not cover 300 records")
        stage.require("v8_supplied", summary.get("v8_predictions_supplied") is True, "V8 predictions were not supplied to external validation")
        stage.require("quantization_binding", summary.get("v8_quantization_manifest_sha256") == sha256(quant_path), "External validation quantization hash mismatch")
        stage.require("frozen_binding", summary.get("frozen_labels_sha256") == sha256(workspace / "CNS_challenge_set_300_HUMAN_FROZEN.csv"), "External validation frozen-label hash mismatch")
        stage.require("overlap_flags", summary.get("training_text_overlap_flags_supplied") is True, "External validation lacks post-freeze overlap flags")
        for analysis in ("Design-weighted remaining frame", "Design-weighted strict text-disjoint subpopulation"):
            present = {row.get("Metric") for row in metrics if row.get("Detector") == "V8" and row.get("Analysis") == analysis}
            stage.require(f"{analysis}_metrics", EXPECTED_METRICS.issubset(present), f"Missing V8 metrics for {analysis}")
        stage.evidence = {"validation_summary_sha256": sha256(summary_path), "performance_metrics_sha256": sha256(metrics_path), "adjudicated": summary.get("adjudicated"), "ambiguous": summary.get("ambiguous")}
    except Exception as exc:
        stage.status = "FAIL"
        stage.errors.append(f"External-validation audit failed: {exc}")
    return stage


def audit_technical(workspace: Path) -> Stage:
    stage = Stage("hash_bound_technical_evidence")
    path = workspace / "classifier_technical_results.json"
    if not path.is_file():
        stage.missing("Real hash-bound technical-results file is absent")
        return stage
    try:
        result = load_json(path)
        quant_path = workspace / "v8_training_candidates_2026-08-20" / "v8_quantized_selected" / "quantization_manifest.json"
        quant_sha = sha256(quant_path)
        checks = result.get("evidence_checks", {})
        stage.require("technical_status", result.get("status") == "PASS", "Technical evidence status is not PASS")
        stage.require("all_evidence_checks", isinstance(checks, dict) and bool(checks) and all(value is True for value in checks.values()), "Technical evidence contains missing or failed checks")
        stage.require("no_failed_checks", result.get("failed_evidence_checks") == [], "Technical evidence lists failed checks")
        stage.require("runtime", result.get("model_runtime") == "v8_chunked" and result.get("cns_decision_mode") == "model_primary", "Technical evidence runtime contract is wrong")
        stage.require("artifact_chain", result.get("model_manifest_sha256") == quant_sha and result.get("parity_quantization_manifest_sha256") == quant_sha and result.get("internal_test_quantization_manifest_sha256") == quant_sha, "Technical evidence artifact chain is mixed")
        stage.require("no_auto_deploy", result.get("automatic_deployment_performed") is False, "Technical collector performed deployment")
        stage.evidence = {"technical_results_sha256": sha256(path), "peak_rss_mb": result.get("peak_rss_mb"), "cold_start_seconds": result.get("cold_start_seconds")}
    except Exception as exc:
        stage.status = "FAIL"
        stage.errors.append(f"Technical-evidence audit failed: {exc}")
    return stage


def audit_decision(workspace: Path) -> Stage:
    stage = Stage("manual_promotion_review_gate")
    path = workspace / "classifier_release_decision.json"
    if not path.is_file():
        stage.missing("v1.2 release decision is absent")
        return stage
    try:
        decision = load_json(path)
        validation = workspace / "cns_challenge_validation_results"
        inputs = {
            "gates": workspace / "classifier_release_gates_v1_2.json",
            "validation_summary": validation / "validation_summary.json",
            "performance_metrics": validation / "performance_metrics.csv",
            "technical_results": workspace / "classifier_technical_results.json",
        }
        stage.require("manual_review_only", decision.get("decision") == "PASS_FOR_MANUAL_PROMOTION_REVIEW", "Release decision is not a manual-promotion-review pass")
        stage.require("no_auto_deploy", decision.get("automatic_deployment_performed") is False, "Release evaluator performed deployment")
        stage.require("scientific_checks", decision.get("failed_scientific") == [] and all(value is True for value in decision.get("scientific_checks", {}).values()), "Scientific release checks are incomplete")
        stage.require("technical_checks", decision.get("failed_technical") == [] and all(value is True for value in decision.get("technical_checks", {}).values()), "Technical release checks are incomplete")
        recorded = decision.get("input_sha256", {})
        for name, input_path in inputs.items():
            stage.require(f"{name}_binding", input_path.is_file() and recorded.get(name) == sha256(input_path), f"Release decision {name} hash mismatch")
        stage.evidence = {"release_decision_sha256": sha256(path), "decision": decision.get("decision")}
    except Exception as exc:
        stage.status = "FAIL"
        stage.errors.append(f"Release-decision audit failed: {exc}")
    return stage


def audit_production(workspace: Path) -> Stage:
    stage = Stage("manual_production_promotion")
    path = workspace / "production_v8_promotion_record.json"
    if not path.is_file():
        stage.missing("Manual production-v8 promotion record is absent; production must remain unchanged before this point")
        return stage
    try:
        record = load_json(path)
        quant_path = workspace / "v8_training_candidates_2026-08-20" / "v8_quantized_selected" / "quantization_manifest.json"
        stage.require("promotion_status", record.get("status") == "PRODUCTION_V8_PROMOTION_VERIFIED", "Production promotion status is invalid")
        stage.require("manual_approval", record.get("manual_approval_recorded") is True, "Manual promotion approval is not recorded")
        stage.require("runtime", record.get("model_runtime") == "v8_chunked" and record.get("cns_decision_mode") == "model_primary", "Production is not on the v8 model-primary runtime")
        stage.require("artifact_binding", record.get("model_manifest_sha256") == sha256(quant_path), "Production model manifest differs from the validated artifact")
        stage.require("health_status", record.get("production_health_status") == "ok", "Production health verification did not pass")
        stage.require("rollback_retained", record.get("rollback_retained") is True, "Production rollback was not retained")
        stage.evidence = {"promotion_record_sha256": sha256(path), "build_commit": record.get("build_commit")}
    except Exception as exc:
        stage.status = "FAIL"
        stage.errors.append(f"Production-promotion audit failed: {exc}")
    return stage


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path)
    parser.add_argument("--require-production", action="store_true")
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    for output in (args.output, args.markdown_output):
        if output is not None and output.exists():
            raise FileExistsError(f"Refusing to overwrite readiness audit: {output}")

    stages = []
    development = audit_development(workspace)
    stages.append(development)
    stages.extend([
        audit_training(workspace, development),
        audit_selection(workspace),
        audit_quantization(workspace),
        audit_internal_test(workspace),
        audit_human_freeze(workspace),
        audit_overlap(workspace),
        audit_challenge_predictions(workspace),
        audit_validation(workspace),
        audit_technical(workspace),
        audit_decision(workspace),
    ])
    production = audit_production(workspace)
    if args.require_production:
        stages.append(production)
    failed = [stage.name for stage in stages if stage.status == "FAIL"]
    missing = [stage.name for stage in stages if stage.status == "MISSING"]
    if failed:
        overall = "HOLD_INVALID_OR_MIXED_EVIDENCE"
    elif missing:
        overall = "WAITING_FOR_REQUIRED_EXECUTIONS"
    else:
        overall = "READY_FOR_MANUAL_PROMOTION_REVIEW"
    result = {
        "readiness_audit_version": "1.0.0",
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "workspace": str(workspace),
        "status": overall,
        "require_production": args.require_production,
        "production_stage_observed_status": production.status,
        "sealed_answer_key_read_by_this_auditor": False,
        "automatic_deployment_performed": False,
        "failed_stages": failed,
        "missing_stages": missing,
        "stages": [stage.__dict__ for stage in stages],
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if args.markdown_output:
        lines = [
            "# CNS classifier expansion readiness audit", "",
            f"Status: **{overall}**", "",
            "| Stage | Status | Evidence or next requirement |", "|---|---|---|",
        ]
        for stage in stages:
            detail = "; ".join(stage.errors) if stage.errors else ", ".join(f"{key}={value}" for key, value in stage.evidence.items())
            lines.append(f"| {stage.name} | {stage.status} | {detail.replace('|', '/')} |")
        lines.extend(["", "This auditor does not read the sealed answer key and never deploys."])
        args.markdown_output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": overall, "failed_stages": failed, "missing_stages": missing, "output": str(args.output)}, indent=2))
    raise SystemExit(0 if overall == "READY_FOR_MANUAL_PROMOTION_REVIEW" else 2)


if __name__ == "__main__":
    main()
