#!/usr/bin/env python3
"""Dependency-light core logic for the chunked v8 classifier pipeline."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class TokenChunk:
    start_token: int
    end_token: int
    token_ids: tuple[int, ...]


def chunk_token_ids(token_ids: list[int], content_tokens: int, overlap_tokens: int) -> list[TokenChunk]:
    if content_tokens <= 0:
        raise ValueError("content_tokens must be positive")
    if overlap_tokens < 0 or overlap_tokens >= content_tokens:
        raise ValueError("overlap_tokens must be non-negative and smaller than content_tokens")
    if not token_ids:
        return [TokenChunk(0, 0, tuple())]
    stride = content_tokens - overlap_tokens
    chunks = []
    start = 0
    while start < len(token_ids):
        end = min(start + content_tokens, len(token_ids))
        chunks.append(TokenChunk(start, end, tuple(token_ids[start:end])))
        if end == len(token_ids):
            break
        start += stride
    return chunks


def aggregate_max_by_trial(rows: Iterable[tuple[str, float]]) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for trial_id, probability in rows:
        if not trial_id:
            raise ValueError("trial_id cannot be blank")
        if not math.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("probability must be finite and between 0 and 1")
        grouped[trial_id].append(float(probability))
    return {trial_id: max(values) for trial_id, values in grouped.items()}


def confusion_metrics(labels: list[int], probabilities: list[float], threshold: float) -> dict[str, float | int]:
    if len(labels) != len(probabilities) or not labels:
        raise ValueError("labels and probabilities must be non-empty and have equal length")
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")
    if any(label not in {0, 1} for label in labels):
        raise ValueError("labels must be binary")
    if any(not math.isfinite(value) or not 0 <= value <= 1 for value in probabilities):
        raise ValueError("probabilities must be finite and between 0 and 1")
    predictions = [int(value >= threshold) for value in probabilities]
    tp = sum(label == 1 and pred == 1 for label, pred in zip(labels, predictions))
    tn = sum(label == 0 and pred == 0 for label, pred in zip(labels, predictions))
    fp = sum(label == 0 and pred == 1 for label, pred in zip(labels, predictions))
    fn = sum(label == 1 and pred == 0 for label, pred in zip(labels, predictions))

    def safe(numerator: float, denominator: float) -> float:
        return numerator / denominator if denominator else float("nan")

    sensitivity = safe(tp, tp + fn)
    specificity = safe(tn, tn + fp)
    ppv = safe(tp, tp + fp)
    npv = safe(tn, tn + fn)
    accuracy = safe(tp + tn, tp + tn + fp + fn)
    f1 = safe(2 * tp, 2 * tp + fp + fn)
    balanced = (sensitivity + specificity) / 2
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = safe(tp * tn - fp * fn, denominator)
    return {
        "threshold": threshold,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "ppv": ppv,
        "npv": npv,
        "accuracy": accuracy,
        "f1": f1,
        "balanced_accuracy": balanced,
        "mcc": mcc,
    }


def threshold_candidates(probabilities: list[float]) -> list[float]:
    if not probabilities:
        raise ValueError("probabilities cannot be empty")
    values = sorted(set(float(value) for value in probabilities))
    candidates = {0.0, 0.5, 1.0}
    candidates.update(values)
    for left, right in zip(values, values[1:]):
        candidates.add((left + right) / 2)
    return sorted(candidates)


def select_threshold(
    labels: list[int],
    probabilities: list[float],
    minimum_sensitivity: float,
    minimum_specificity: float,
) -> dict:
    rows = [confusion_metrics(labels, probabilities, threshold) for threshold in threshold_candidates(probabilities)]
    for row in rows:
        row["feasible"] = bool(
            row["sensitivity"] >= minimum_sensitivity and row["specificity"] >= minimum_specificity
        )

    def finite_or_low(value: float) -> float:
        return value if math.isfinite(value) else -math.inf

    selected = max(
        rows,
        key=lambda row: (
            int(row["feasible"]),
            finite_or_low(float(row["mcc"])),
            finite_or_low(float(row["balanced_accuracy"])),
            finite_or_low(float(row["sensitivity"])),
            finite_or_low(float(row["specificity"])),
            -float(row["threshold"]),
        ),
    )
    return {
        "selected": selected,
        "feasible_threshold_exists": any(bool(row["feasible"]) for row in rows),
        "candidate_count": len(rows),
        "all_candidates": rows,
    }
