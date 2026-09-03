#!/usr/bin/env python3
"""
Evaluate the deployed classifier against the hand-labelled gold standard, using
outcome text pulled live from ClinicalTrials.gov rather than the copy stored in
the dataset.

Why the registry pull matters. The `Outcome text (for cognition check)` column
in MASTER_gold_standard_1888.csv is truncated at roughly 300-400 characters --
enough for the human labeller working alongside the registry page, but often
cutting off the outcome that names the instrument. 637 of the 1,070 trials
labelled Yes have no cognitive term anywhere in that stored column, while the
`Flagged Keyword(s)` column records the instrument the labeller actually saw.
Scoring the stored column therefore measures truncation, not the classifier.

This script:
  1. draws a label-balanced sample per cancer type (seeded)
  2. fetches the complete primary/secondary/other outcome text for each NCT
     from the ClinicalTrials.gov v2 API, and caches it
  3. scores it through the deployed endpoint
  4. reports agreement, sensitivity, specificity, routing and the review-flag
     rate, per cancer type

Standing caveat: these trials are in the model's development set, so agreement
is bounded above by memorisation and is not an external-validity estimate. The
routing behaviour, the flag rate and the truncation effect are label-independent
and are the substantive results.

    python validation/evaluate_deployed_classifier.py path/to/MASTER_gold_standard_1888.csv
"""

import argparse
import csv
import json
import os
import random
import statistics
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from urllib import error, parse, request

API = "https://cognitive-outcome-classifier-api.onrender.com"
CTGOV = "https://clinicaltrials.gov/api/v2/studies"

# The API accepts batches up to 6. On full-length registry outcome text the
# 512MB instance restarts (502/503) at 6 and still at 3 -- the tokenizer pads to
# the longest item in the batch and the instance has almost no headroom. One
# item per request with pacing is the only configuration that runs 200 trials
# without a restart. This is a real constraint of the free tier, not a tuning
# preference, and it is why results are cached per trial below.
BATCH = 1
PACING = 2.0
SEED = 2026

TYPE_COL, LABEL_COL, ID_COL = "Cancer Type", "Measures cognition? (Y/N)", "NCT/TrialID"
INCLUDE_COL, STORED_TEXT_COL = "INCLUDE? (Y/N)", "Outcome text (for cognition check)"
TYPES = ("CNS", "Breast", "Lung", "HeadNeck")


# --------------------------------------------------------------------------
# Sampling
# --------------------------------------------------------------------------

def sample(path, per_cell):
    cells = defaultdict(list)
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            ctype = (row.get(TYPE_COL) or "").strip()
            label = (row.get(LABEL_COL) or "").strip()
            nct = (row.get(ID_COL) or "").strip()
            if ctype not in TYPES or label not in ("Yes", "No"):
                continue
            if (row.get(INCLUDE_COL) or "").strip() != "Yes":
                continue
            if not nct.upper().startswith("NCT"):
                continue  # international registry IDs are not resolvable here
            cells[(ctype, label)].append({
                "trial_id": nct,
                "cancer_type": ctype,
                "label": label == "Yes",
                "stored_text": (row.get(STORED_TEXT_COL) or "").strip(),
            })

    rng = random.Random(SEED)
    picked = []
    for key in sorted(cells):
        rows = cells[key]
        rng.shuffle(rows)
        picked.extend(rows[:per_cell])
    return picked


# --------------------------------------------------------------------------
# ClinicalTrials.gov
# --------------------------------------------------------------------------

def fetch_outcome_text(nct):
    url = f"{CTGOV}/{parse.quote(nct)}?fields=protocolSection.outcomesModule"
    for attempt in range(3):
        try:
            with request.urlopen(url, timeout=60) as r:
                data = json.load(r)
            break
        except error.HTTPError as exc:
            if exc.code == 404:
                return None
            if attempt == 2:
                raise
            time.sleep(3 * (attempt + 1))
        except (error.URLError, TimeoutError):
            if attempt == 2:
                raise
            time.sleep(3 * (attempt + 1))
    module = data.get("protocolSection", {}).get("outcomesModule", {})
    parts = []
    for key in ("primaryOutcomes", "secondaryOutcomes", "otherOutcomes"):
        for outcome in module.get(key, []):
            parts.append(" ".join(filter(None, [outcome.get("measure"),
                                                outcome.get("description")])))
    return " | ".join(p for p in parts if p) or None


def build_cache(rows, cache_path):
    cache = {}
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            cache = json.load(f)
    missing = [r["trial_id"] for r in rows if r["trial_id"] not in cache]
    print(f"fetching {len(missing)} trials from ClinicalTrials.gov "
          f"({len(rows) - len(missing)} cached)")
    for n, nct in enumerate(missing, 1):
        cache[nct] = fetch_outcome_text(nct)
        if n % 20 == 0:
            print(f"  {n}/{len(missing)}", end="\r", flush=True)
        time.sleep(0.15)
    with open(cache_path, "w") as f:
        json.dump(cache, f, indent=1, sort_keys=True)
    return cache


# --------------------------------------------------------------------------
# Deployed API
# --------------------------------------------------------------------------

def wake(api):
    print("waking instance...")
    for attempt in range(6):
        try:
            with request.urlopen(f"{api}/health", timeout=60) as r:
                print(f"  health: {json.load(r)}")
                return True
        except Exception as exc:
            print(f"  attempt {attempt + 1}: {exc}", file=sys.stderr)
            time.sleep(15)
    return False


def post_batch(api, items, retries=5):
    req = request.Request(
        f"{api}/predict_batch",
        data=json.dumps({"items": items}).encode(),
        headers={"Content-Type": "application/json"},
    )
    for attempt in range(retries):
        started = time.perf_counter()
        try:
            with request.urlopen(req, timeout=180) as resp:
                return json.load(resp)["results"], time.perf_counter() - started
        except error.HTTPError as exc:
            if exc.code not in (429, 500, 502, 503, 504) or attempt == retries - 1:
                raise
            wait = 15 * (attempt + 1)
            print(f"\n  HTTP {exc.code}; instance restarting, retrying in {wait}s",
                  file=sys.stderr)
            time.sleep(wait)
        except (error.URLError, TimeoutError) as exc:
            if attempt == retries - 1:
                raise
            wait = 15 * (attempt + 1)
            print(f"\n  {exc}; retrying in {wait}s", file=sys.stderr)
            time.sleep(wait)


# --------------------------------------------------------------------------

def metrics(subset):
    if not subset:
        return None
    tp = sum(1 for r in subset if r["predicted"] and r["human_label"])
    tn = sum(1 for r in subset if not r["predicted"] and not r["human_label"])
    fp = sum(1 for r in subset if r["predicted"] and not r["human_label"])
    fn = sum(1 for r in subset if not r["predicted"] and r["human_label"])
    pos, neg = tp + fn, tn + fp
    return {
        "n": len(subset),
        "agreement": round((tp + tn) / len(subset), 4),
        "sensitivity": round(tp / pos, 4) if pos else None,
        "specificity": round(tn / neg, 4) if neg else None,
        "flagged_for_review": round(
            sum(1 for r in subset if r["review_recommended"]) / len(subset), 4),
        "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--per-cell", type=int, default=25,
                    help="trials per (cancer type x label) cell")
    ap.add_argument("--api", default=API)
    ap.add_argument("--out", default="validation")
    args = ap.parse_args()

    rows = sample(args.dataset, args.per_cell)
    print(f"{len(rows)} trials sampled: "
          f"{Counter((r['cancer_type'], 'Yes' if r['label'] else 'No') for r in rows)}")

    cache = build_cache(rows, os.path.join(args.out, "ctgov_outcome_text_cache.json"))
    rows = [r | {"outcome_text": cache[r["trial_id"]]}
            for r in rows if cache.get(r["trial_id"])]
    print(f"\n{len(rows)} trials have retrievable outcome text")

    if not wake(args.api):
        sys.exit("could not reach the API")

    score_cache_path = os.path.join(args.out, "scored_cache.json")
    score_cache = {}
    if os.path.exists(score_cache_path):
        with open(score_cache_path) as f:
            score_cache = json.load(f)

    scored, latencies = [], []
    for i in range(0, len(rows), BATCH):
        chunk = rows[i:i + BATCH]
        cached = [score_cache.get(r["trial_id"]) for r in chunk]
        if all(cached):
            results, elapsed = cached, 0.0
        else:
            results, elapsed = post_batch(args.api, [
                {"trial_id": r["trial_id"], "cancer_type": r["cancer_type"],
                 "outcome_text": r["outcome_text"]} for r in chunk
            ])
            for r, res in zip(chunk, results):
                score_cache[r["trial_id"]] = res
            with open(score_cache_path, "w") as f:
                json.dump(score_cache, f)
            latencies.append(elapsed / len(chunk))
        for src, res in zip(chunk, results):
            scored.append({
                "trial_id": src["trial_id"],
                "cancer_type": src["cancer_type"],
                "human_label": src["label"],
                "predicted": res["predicted_cognitive"],
                "confidence": res["confidence"],
                "method": res["method"],
                "review_recommended": res["review_recommended"],
                "review_reason": res["review_reason"] or "",
                "registry_text_chars": len(src["outcome_text"]),
                "stored_text_chars": len(src["stored_text"]),
            })
        print(f"  scored {len(scored)}/{len(rows)}", end="\r", flush=True)
        if elapsed:
            time.sleep(PACING)

    truncated = [r for r in scored if r["stored_text_chars"] < r["registry_text_chars"]]
    summary = {
        "run_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "endpoint": args.api,
        "text_source": "ClinicalTrials.gov API v2, full outcome module",
        "sampling": f"label-balanced, {args.per_cell} per cancer-type x label cell, seed {SEED}",
        "overall": metrics(scored),
        "by_cancer_type": {t: metrics([r for r in scored if r["cancer_type"] == t])
                           for t in TYPES},
        "routing": dict(Counter(r["method"] for r in scored)),
        "flag_reasons": dict(Counter(
            (r["review_reason"].split("(")[0].strip() or "uncertain band")
            for r in scored if r["review_recommended"])),
        "stored_text_truncation": {
            "trials_where_stored_text_is_shorter": len(truncated),
            "median_stored_chars": statistics.median(
                r["stored_text_chars"] for r in scored),
            "median_registry_chars": statistics.median(
                r["registry_text_chars"] for r in scored),
        },
        "latency_seconds_per_trial": {
            "median": round(statistics.median(latencies), 3) if latencies else None,
            "mean": round(statistics.fmean(latencies), 3) if latencies else None,
            "n_measured": len(latencies),
        },
        "caveat": (
            "These trials are in the model's development set. Agreement is "
            "bounded above by memorisation and is not an external-validity "
            "estimate. Routing, flag rate and the truncation comparison are "
            "label-independent."
        ),
    }

    os.makedirs(args.out, exist_ok=True)
    with open(f"{args.out}/deployed_classifier_scored.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(scored[0].keys()))
        w.writeheader()
        w.writerows(scored)
    with open(f"{args.out}/deployed_classifier_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\n" + json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
