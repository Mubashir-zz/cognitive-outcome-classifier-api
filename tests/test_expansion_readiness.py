#!/usr/bin/env python3

from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AUDITOR_CANDIDATES = [
    ROOT / "verify_cns_expansion_readiness.py",
    ROOT / "training" / "verify_cns_expansion_readiness.py",
]
AUDITOR = next((path for path in AUDITOR_CANDIDATES if path.is_file()), AUDITOR_CANDIDATES[0])


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def write_csv(path: Path, headers: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def artifact(path: Path, root: Path) -> dict:
    return {"path": str(path.relative_to(root)), "bytes": path.stat().st_size, "sha256": sha(path)}


def build_development(workspace: Path) -> tuple[Path, Path]:
    release_dir = workspace / "v8_development_release_2026-08-20"
    release = release_dir / "v8_development_release.csv"
    headers = ["NCT_or_TrialID", "Outcome_Text", "Label", "Cancer_Type", "Normalized_Text_SHA256", "Split"]
    rows = [
        {"NCT_or_TrialID": "T1", "Outcome_Text": "memory test", "Label": "Yes", "Cancer_Type": "CNS", "Normalized_Text_SHA256": "1" * 64, "Split": "train"},
        {"NCT_or_TrialID": "T2", "Outcome_Text": "survival", "Label": "No", "Cancer_Type": "CNS", "Normalized_Text_SHA256": "2" * 64, "Split": "calibration"},
        {"NCT_or_TrialID": "T3", "Outcome_Text": "attention", "Label": "Yes", "Cancer_Type": "CNS", "Normalized_Text_SHA256": "3" * 64, "Split": "calibration"},
        {"NCT_or_TrialID": "T4", "Outcome_Text": "toxicity", "Label": "No", "Cancer_Type": "CNS", "Normalized_Text_SHA256": "4" * 64, "Split": "internal_test"},
    ]
    write_csv(release, headers, rows)
    config = workspace / "v8_training_config.json"
    write_json(config, {
        "development_release_sha256": sha(release),
        "training": {"seeds": [1, 2]},
        "quantization": {"required_class_agreement": 1.0, "maximum_absolute_probability_delta": 0.02},
    })
    write_json(release_dir / "v8_split_manifest.json", {
        "data_file": {"sha256": sha(release)},
        "invariants": {"challenge_text_overlap_in_release": 0},
    })
    return release, config


def build_complete_workspace(workspace: Path) -> None:
    release, config_path = build_development(workspace)
    candidates = workspace / "v8_training_candidates_2026-08-20"
    root_entries = []
    selection_candidates = []
    for seed in (1, 2):
        candidate = candidates / f"candidate_seed_{seed}"
        model = candidate / "model" / "weights.bin"
        model.parent.mkdir(parents=True)
        model.write_bytes(f"model-{seed}".encode())
        predictions = candidate / "calibration_predictions.csv"
        write_csv(predictions, ["NCT_or_TrialID", "Cancer_Type", "Label", "Probability"], [
            {"NCT_or_TrialID": "T2", "Cancer_Type": "CNS", "Label": "0", "Probability": "0.1"},
            {"NCT_or_TrialID": "T3", "Cancer_Type": "CNS", "Label": "1", "Probability": "0.9"},
        ])
        manifest_path = candidate / "training_run_manifest.json"
        write_json(manifest_path, {
            "seed": seed,
            "development_release_sha256": sha(release),
            "training_config_sha256": sha(config_path),
            "calibration_predictions_sha256": sha(predictions),
            "internal_test_accessed": False,
            "challenge_data_accessed": False,
            "artifacts": [artifact(predictions, candidate), artifact(model, candidate)],
        })
        root_entries.append({"seed": seed, "path": str(manifest_path.relative_to(candidates)), "sha256": sha(manifest_path)})
        selection_candidates.append({"seed": seed, "manifest_sha256": sha(manifest_path), "calibration_predictions_sha256": sha(predictions)})
    write_json(candidates / "v8_training_root_manifest.json", {
        "status": "ALL_PRESPECIFIED_CANDIDATES_COMPLETE",
        "configured_seeds": [1, 2], "completed_seeds": [1, 2],
        "development_release_sha256": sha(release), "training_config_sha256": sha(config_path),
        "internal_test_accessed": False, "challenge_data_accessed": False,
        "candidate_manifests": root_entries,
    })
    selection_path = candidates / "v8_candidate_selection.json"
    write_json(selection_path, {
        "status": "CANDIDATE_SELECTED_ON_CALIBRATION_ONLY",
        "training_config_sha256": sha(config_path), "selected_seed": 1,
        "selected_threshold": 0.5, "feasible_threshold_exists": True,
        "internal_test_accessed": False, "challenge_data_accessed": False,
        "candidates": selection_candidates,
    })

    quant_dir = candidates / "v8_quantized_selected"
    model = quant_dir / "model.ts"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"quantized-model")
    tokenizer = quant_dir / "tokenizer" / "tokenizer.json"
    tokenizer.parent.mkdir()
    tokenizer.write_text("{}", encoding="utf-8")
    contract = quant_dir / "inference_contract.json"
    contract.write_text("{}", encoding="utf-8")
    quant_path = quant_dir / "quantization_manifest.json"
    parity = {"parity_passed": True, "class_agreement": 1.0, "maximum_absolute_probability_delta": 0.0}
    write_json(quant_path, {
        "status": "QUANTIZED_CANDIDATE_PASSED_CALIBRATION_PARITY",
        "selection_sha256": sha(selection_path), "training_config_sha256": sha(config_path),
        "development_release_sha256": sha(release), "selected_seed": 1, "selected_threshold": 0.5,
        "eager_quantized_parity": parity, "serialized_torchscript_parity": parity,
        "internal_test_accessed": False, "challenge_data_accessed": False,
        "artifacts": {
            "torchscript": {"path": model.name, "bytes": model.stat().st_size, "sha256": sha(model)},
            "tokenizer_directory": "tokenizer",
            "tokenizer_files": [{"path": tokenizer.name, "bytes": tokenizer.stat().st_size, "sha256": sha(tokenizer)}],
            "inference_contract": {"path": contract.name, "sha256": sha(contract)},
        },
    })

    internal = candidates / "v8_internal_test_once"
    internal_predictions = internal / "internal_test_predictions.csv"
    write_csv(internal_predictions, ["NCT_or_TrialID", "Probability"], [{"NCT_or_TrialID": "T4", "Probability": "0.1"}])
    internal_result = internal / "internal_test_evaluation.json"
    write_json(internal_result, {
        "status": "INTERNAL_TEST_EVALUATED_ONCE", "selection_sha256": sha(selection_path),
        "quantization_manifest_sha256": sha(quant_path), "evaluated_torchscript_sha256": sha(model),
        "training_config_sha256": sha(config_path), "development_release_sha256": sha(release),
        "internal_test_predictions_sha256": sha(internal_predictions), "internal_test_records": 1,
        "challenge_data_accessed": False, "production_promotion_decided": False,
    })
    write_json(internal / "INTERNAL_TEST_EVALUATED.lock", {"selection_sha256": sha(selection_path), "result_sha256": sha(internal_result)})

    frozen = workspace / "CNS_challenge_set_300_HUMAN_FROZEN.csv"
    frozen_headers = ["Row_ID", "Outcome_Text_Part_1", "Outcome_Text_Part_2", "Measures_Cognition_Y_N", "Reviewer_Confidence_High_Medium_Low", "Notes"]
    frozen_rows = [{
        "Row_ID": f"CNS-CHAL-{index:03d}", "Outcome_Text_Part_1": f"text {index}", "Outcome_Text_Part_2": "",
        "Measures_Cognition_Y_N": "Yes" if index % 2 else "No", "Reviewer_Confidence_High_Medium_Low": "High", "Notes": "",
    } for index in range(1, 301)]
    write_csv(frozen, frozen_headers, frozen_rows)
    frozen_manifest = frozen.with_suffix(frozen.suffix + ".manifest.json")
    write_json(frozen_manifest, {"status": "FROZEN_HUMAN_LABELS", "records": 300, "frozen_sha256": sha(frozen), "sealed_key_opened_by_this_script": False})
    frozen.with_suffix(frozen.suffix + ".sha256").write_text(f"{sha(frozen)}  {frozen.name}\n", encoding="utf-8")
    os.chmod(frozen, 0o444)

    training_v7 = workspace / "CROSS_CANCER_TRAINING_SET_v7.csv"
    training_v7.write_text("synthetic training\n", encoding="utf-8")
    overlap = workspace / "CNS_challenge_set_300_POST_FREEZE_TEXT_OVERLAP.csv"
    overlap_rows = [{"Row_ID": row["Row_ID"], "Training_Text_Overlap": "False", "Matched_Training_Rows": "0", "Normalized_Text_SHA256": f"{index:064x}"} for index, row in enumerate(frozen_rows, start=1)]
    write_csv(overlap, ["Row_ID", "Training_Text_Overlap", "Matched_Training_Rows", "Normalized_Text_SHA256"], overlap_rows)
    write_json(overlap.with_suffix(overlap.suffix + ".manifest.json"), {
        "status": "POST_FREEZE_TRAINING_TEXT_OVERLAP_FLAGS", "records": 300,
        "output_sha256": sha(overlap), "frozen_labels_sha256": sha(frozen),
        "training_data_sha256": sha(training_v7), "created_only_after_frozen_human_labels": True,
        "sealed_key_opened": False, "overlap_records": 0, "overlap_rate": 0.0,
    })

    challenge = workspace / "CNS_challenge_set_300_V8_FROZEN_PREDICTIONS.csv"
    prediction_rows = [{"Row_ID": row["Row_ID"], "V8_Probability": "0.9" if index % 2 else "0.1", "V8_Prediction": "1" if index % 2 else "0"} for index, row in enumerate(frozen_rows, start=1)]
    write_csv(challenge, ["Row_ID", "V8_Probability", "V8_Prediction"], prediction_rows)
    write_json(challenge.with_suffix(challenge.suffix + ".manifest.json"), {
        "status": "FROZEN_V8_CHALLENGE_PREDICTIONS", "records": 300,
        "output_sha256": sha(challenge), "frozen_labels_sha256": sha(frozen),
        "frozen_labels_manifest_sha256": sha(frozen_manifest), "quantization_manifest_sha256": sha(quant_path),
        "serialized_torchscript_sha256": sha(model), "training_config_sha256": sha(config_path),
        "development_release_sha256": sha(release), "selected_threshold": 0.5,
        "human_truth_used_for_scoring": False, "reviewer_confidence_used_for_scoring": False,
        "challenge_identity_used_for_scoring": False, "sealed_key_opened_by_this_script": False,
    })

    validation = workspace / "cns_challenge_validation_results"
    summary = validation / "validation_summary.json"
    write_json(summary, {
        "status": "COMPLETE", "records": 300, "adjudicated": 300, "ambiguous": 0,
        "v8_predictions_supplied": True, "v8_quantization_manifest_sha256": sha(quant_path),
        "frozen_labels_sha256": sha(frozen), "training_text_overlap_flags_supplied": True,
    })
    metrics = validation / "performance_metrics.csv"
    metric_rows = []
    for analysis in ("Design-weighted remaining frame", "Design-weighted strict text-disjoint subpopulation"):
        for metric in ("Sensitivity", "Specificity", "Balanced_Accuracy", "MCC"):
            metric_rows.append({"Detector": "V8", "Analysis": analysis, "Metric": metric, "Estimate": "1.0", "Lower_95": "0.9", "Upper_95": "1.0"})
    write_csv(metrics, ["Detector", "Analysis", "Metric", "Estimate", "Lower_95", "Upper_95"], metric_rows)

    technical = workspace / "classifier_technical_results.json"
    write_json(technical, {
        "status": "PASS", "evidence_checks": {"all": True}, "failed_evidence_checks": [],
        "model_runtime": "v8_chunked", "cns_decision_mode": "model_primary",
        "model_manifest_sha256": sha(quant_path), "parity_quantization_manifest_sha256": sha(quant_path),
        "internal_test_quantization_manifest_sha256": sha(quant_path), "automatic_deployment_performed": False,
        "peak_rss_mb": 300, "cold_start_seconds": 10,
    })
    gates = workspace / "classifier_release_gates_v1_2.json"
    gates.write_text("{}\n", encoding="utf-8")
    decision = workspace / "classifier_release_decision.json"
    write_json(decision, {
        "decision": "PASS_FOR_MANUAL_PROMOTION_REVIEW", "automatic_deployment_performed": False,
        "scientific_checks": {"all": True}, "technical_checks": {"all": True},
        "failed_scientific": [], "failed_technical": [],
        "input_sha256": {
            "gates": sha(gates), "validation_summary": sha(summary),
            "performance_metrics": sha(metrics), "technical_results": sha(technical),
        },
    })


def run_audit(workspace: Path, output: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run([
        sys.executable, str(AUDITOR), "--workspace", str(workspace), "--output", str(output),
    ], cwd=ROOT, text=True, capture_output=True, check=False)


class ExpansionReadinessTests(unittest.TestCase):
    def test_complete_synthetic_chain_reaches_manual_review(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            build_complete_workspace(workspace)
            output = workspace / "audit.json"
            completed = run_audit(workspace, output)
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            result = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(result["status"], "READY_FOR_MANUAL_PROMOTION_REVIEW")
            self.assertEqual(result["failed_stages"], [])
            self.assertEqual(result["missing_stages"], [])
            self.assertFalse(result["sealed_answer_key_read_by_this_auditor"])

    def test_quantized_model_tampering_holds_chain(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            build_complete_workspace(workspace)
            (workspace / "v8_training_candidates_2026-08-20" / "v8_quantized_selected" / "model.ts").write_bytes(b"tampered")
            output = workspace / "audit.json"
            completed = run_audit(workspace, output)
            self.assertEqual(completed.returncode, 2)
            result = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(result["status"], "HOLD_INVALID_OR_MIXED_EVIDENCE")
            self.assertIn("serialized_quantized_artifact", result["failed_stages"])

    def test_development_only_state_waits_without_opening_key(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            build_development(workspace)
            output = workspace / "audit.json"
            completed = run_audit(workspace, output)
            self.assertEqual(completed.returncode, 2)
            result = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(result["status"], "WAITING_FOR_REQUIRED_EXECUTIONS")
            self.assertEqual(result["failed_stages"], [])
            self.assertFalse(result["sealed_answer_key_read_by_this_auditor"])

    def test_auditor_has_no_sealed_key_input(self) -> None:
        source = AUDITOR.read_text(encoding="utf-8")
        self.assertNotIn("CNS_challenge_set_300_SEALED_KEY", source)
        self.assertNotIn('add_argument("--sealed', source)


if __name__ == "__main__":
    unittest.main(verbosity=2)
