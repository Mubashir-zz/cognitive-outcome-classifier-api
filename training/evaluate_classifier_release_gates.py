#!/usr/bin/env python3
"""Evaluate prespecified scientific and technical gates; never deploy directly."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gates", type=Path, required=True)
    parser.add_argument("--validation-dir", type=Path, required=True)
    parser.add_argument("--technical-results", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

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
    passed = all(scientific_checks.values()) and all(technical_checks.values())
    result = {
        "gate_version": gates["gate_version"],
        "decision": "PASS_FOR_MANUAL_PROMOTION_REVIEW" if passed else "HOLD",
        "automatic_deployment_performed": False,
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
