#!/usr/bin/env python3
"""Evaluate prespecified scientific and technical gates; never deploy directly."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
COLLECTOR_CANDIDATES = (
    ROOT / "staging_classifier_v2" / "scripts" / "collect_classifier_technical_results.py",
    ROOT.parent / "scripts" / "collect_classifier_technical_results.py",
)
TECHNICAL_COLLECTOR = next((path for path in COLLECTOR_CANDIDATES if path.is_file()), COLLECTOR_CANDIDATES[0])


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gates", type=Path, required=True)
    parser.add_argument("--validation-dir", type=Path, required=True)
    parser.add_argument("--technical-results", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite release decision: {args.output}")

    gates = json.loads(args.gates.read_text(encoding="utf-8"))
    summary = json.loads((args.validation_dir / "validation_summary.json").read_text(encoding="utf-8"))
    with (args.validation_dir / "performance_metrics.csv").open(encoding="utf-8", newline="") as handle:
        metrics = list(csv.DictReader(handle))
    technical = json.loads(args.technical_results.read_text(encoding="utf-8")) if args.technical_results else {}

    detector = gates["scientific"]["primary_detector"]
    def analysis_metrics(analysis: str) -> dict[str, dict[str, str]]:
        return {
            row["Metric"]: row
            for row in metrics
            if row["Detector"] == detector and row["Analysis"] == analysis
        }

    weighted = analysis_metrics("Design-weighted remaining frame")
    strict = analysis_metrics("Design-weighted strict text-disjoint subpopulation")
    records = int(summary.get("records", 0))
    ambiguous = int(summary.get("ambiguous", records))
    high_confidence = summary.get("high_confidence_false_negatives", {})
    primary_high_confidence_false_negatives = (
        high_confidence.get(detector)
        if isinstance(high_confidence, dict)
        else None
    )
    if primary_high_confidence_false_negatives is None:
        primary_high_confidence_false_negatives = summary.get(
            f"{detector.lower()}_high_confidence_false_negatives"
        )

    def number(table: dict[str, dict[str, str]], metric: str, field: str = "Estimate") -> float:
        value = table.get(metric, {}).get(field, "")
        return float(value) if value not in {"", None} else float("nan")

    scientific_checks = {
        "human_records": records >= gates["scientific"]["required_human_records"],
        "ambiguous_rate": ambiguous / max(1, records) <= gates["scientific"]["maximum_ambiguous_rate"],
        "weighted_sensitivity": number(weighted, "Sensitivity") >= gates["scientific"]["minimum_weighted_sensitivity"],
        "weighted_sensitivity_lower_95": number(weighted, "Sensitivity", "Lower_95") >= gates["scientific"]["minimum_weighted_sensitivity_lower_95"],
        "weighted_specificity": number(weighted, "Specificity") >= gates["scientific"]["minimum_weighted_specificity"],
        "weighted_balanced_accuracy": number(weighted, "Balanced_Accuracy") >= gates["scientific"]["minimum_weighted_balanced_accuracy"],
        "weighted_mcc": number(weighted, "MCC") >= gates["scientific"]["minimum_weighted_mcc"],
        "high_confidence_false_negatives": primary_high_confidence_false_negatives is not None and primary_high_confidence_false_negatives <= gates["scientific"]["maximum_high_confidence_false_negatives"],
    }
    if gates["scientific"].get("require_v8_predictions"):
        scientific_checks["v8_predictions"] = summary.get("v8_predictions_supplied") is True
    if gates["scientific"].get("require_training_text_overlap_flags"):
        scientific_checks["training_text_overlap_flags"] = summary.get("training_text_overlap_flags_supplied") is True
        scientific_checks["training_text_overlap_rate"] = (
            summary.get("training_text_overlap_rate") is not None
            and summary["training_text_overlap_rate"] <= gates["scientific"]["maximum_training_text_overlap_rate"]
        )
    if gates["scientific"].get("require_strict_text_disjoint_analysis"):
        scientific_checks.update({
            "strict_text_disjoint_analysis_present": bool(strict),
            "strict_text_disjoint_sensitivity": number(strict, "Sensitivity") >= gates["scientific"]["strict_text_disjoint_minimum_sensitivity"],
            "strict_text_disjoint_sensitivity_lower_95": number(strict, "Sensitivity", "Lower_95") >= gates["scientific"]["strict_text_disjoint_minimum_sensitivity_lower_95"],
            "strict_text_disjoint_specificity": number(strict, "Specificity") >= gates["scientific"]["strict_text_disjoint_minimum_specificity"],
            "strict_text_disjoint_balanced_accuracy": number(strict, "Balanced_Accuracy") >= gates["scientific"]["strict_text_disjoint_minimum_balanced_accuracy"],
            "strict_text_disjoint_mcc": number(strict, "MCC") >= gates["scientific"]["strict_text_disjoint_minimum_mcc"],
        })

    expected_technical = gates["technical"]
    technical_checks = {
        "unit_tests_pass": technical.get("unit_tests_pass") is expected_technical["unit_tests_pass"],
        "contract_tests_pass": technical.get("contract_tests_pass") is expected_technical["contract_tests_pass"],
        "frozen_regression_tests_pass": technical.get("frozen_regression_tests_pass") is expected_technical["frozen_regression_tests_pass"],
        "analytic_api_parity": technical.get("analytic_api_parity", -1) >= expected_technical["analytic_api_parity"],
        "non_echoing_validation_errors": technical.get("non_echoing_validation_errors") is expected_technical["non_echoing_validation_errors"],
        "payload_limit_enforced": technical.get("payload_limit_enforced") is expected_technical["payload_limit_enforced"],
        "artifact_hashes_exposed": technical.get("artifact_hashes_exposed") is expected_technical["artifact_hashes_exposed"],
        "prediction_authentication_enabled": technical.get("prediction_authentication_enabled") is expected_technical["prediction_authentication_enabled"],
        "peak_rss_mb": technical.get("peak_rss_mb", float("inf")) <= expected_technical["maximum_peak_rss_mb"],
        "cold_start_seconds": technical.get("cold_start_seconds", float("inf")) <= expected_technical["maximum_cold_start_seconds"],
        "rollback_artifact_recorded": technical.get("rollback_artifact_recorded") is expected_technical["rollback_artifact_recorded"],
    }
    if expected_technical.get("required_model_runtime"):
        technical_checks["model_runtime"] = technical.get("model_runtime") == expected_technical["required_model_runtime"]
    if expected_technical.get("required_cns_decision_mode"):
        technical_checks["cns_decision_mode"] = technical.get("cns_decision_mode") == expected_technical["required_cns_decision_mode"]
    if expected_technical.get("required_technical_results_version"):
        technical_checks["technical_results_version"] = (
            technical.get("technical_results_version") == expected_technical["required_technical_results_version"]
        )
    if expected_technical.get("require_hash_bound_evidence_collector"):
        evidence_checks = technical.get("evidence_checks")
        technical_checks["technical_evidence_status"] = technical.get("status") == "PASS"
        technical_checks["technical_evidence_checks"] = (
            isinstance(evidence_checks, dict)
            and bool(evidence_checks)
            and all(value is True for value in evidence_checks.values())
            and technical.get("failed_evidence_checks") == []
        )
        technical_checks["technical_collector_binding"] = (
            TECHNICAL_COLLECTOR.is_file()
            and technical.get("input_sha256", {}).get("collector") == sha256(TECHNICAL_COLLECTOR)
        )
    if expected_technical.get("require_artifact_chain_match"):
        reference_manifest_sha = summary.get("v8_quantization_manifest_sha256")
        chain_values = {
            "api": technical.get("model_manifest_sha256"),
            "parity": technical.get("parity_quantization_manifest_sha256"),
            "internal_test": technical.get("internal_test_quantization_manifest_sha256"),
        }
        technical_checks["artifact_chain_reference_present"] = (
            isinstance(reference_manifest_sha, str) and len(reference_manifest_sha) == 64
        )
        technical_checks["artifact_chain_match"] = (
            technical_checks["artifact_chain_reference_present"]
            and all(value == reference_manifest_sha for value in chain_values.values())
        )
    passed = all(scientific_checks.values()) and all(technical_checks.values())
    result = {
        "gate_version": gates["gate_version"],
        "decision": "PASS_FOR_MANUAL_PROMOTION_REVIEW" if passed else "HOLD",
        "automatic_deployment_performed": False,
        "input_sha256": {
            "gates": sha256(args.gates),
            "validation_summary": sha256(args.validation_dir / "validation_summary.json"),
            "performance_metrics": sha256(args.validation_dir / "performance_metrics.csv"),
            "technical_results": sha256(args.technical_results) if args.technical_results else None,
        },
        "scientific_checks": scientific_checks,
        "technical_checks": technical_checks,
        "failed_scientific": [name for name, ok in scientific_checks.items() if not ok],
        "failed_technical": [name for name, ok in technical_checks.items() if not ok],
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if passed else 2)


if __name__ == "__main__":
    main()
