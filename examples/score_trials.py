#!/usr/bin/env python3
"""
Score a CSV of trial outcome text against the deployed classifier.

Reads a CSV with columns `trial_id`, `cancer_type` and `outcome_text`, sends
them through /predict_batch in chunks, and writes a scored CSV plus a short
summary of how many predictions were flagged for human review.

    python examples/score_trials.py input.csv --out scored.csv

Against a local server instead of the deployed one:

    python examples/score_trials.py input.csv --api http://localhost:8000

The batch size default of 6 matches the server's MAX_BATCH_SIZE; larger
batches are rejected, and were what caused out-of-memory crashes on the free
tier in the first place.
"""

import argparse
import csv
import sys
import time
from urllib import error, request
import json

DEFAULT_API = "https://cognitive-outcome-classifier-api.onrender.com"
BATCH_SIZE = 6
REQUIRED_COLUMNS = {"trial_id", "cancer_type", "outcome_text"}

FIELDNAMES = [
    "trial_id", "cancer_type", "outcome_text",
    "predicted_cognitive", "confidence", "method",
    "review_recommended", "review_reason",
]


def post_batch(api, items, retries=3):
    """POST one batch, retrying on the cold-start timeout Render's free tier
    produces after ~15 minutes of inactivity."""
    payload = json.dumps({"items": items}).encode()
    req = request.Request(
        f"{api}/predict_batch",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    for attempt in range(retries):
        try:
            with request.urlopen(req, timeout=120) as resp:
                return json.load(resp)["results"]
        except (error.URLError, TimeoutError) as exc:
            if attempt == retries - 1:
                raise
            wait = 20 * (attempt + 1)
            print(f"  request failed ({exc}); retrying in {wait}s", file=sys.stderr)
            time.sleep(wait)


def chunked(rows, size):
    for i in range(0, len(rows), size):
        yield rows[i:i + size]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="CSV with trial_id, cancer_type, outcome_text")
    parser.add_argument("--out", default="scored_results.csv")
    parser.add_argument("--api", default=DEFAULT_API)
    parser.add_argument("--limit", type=int, help="only score the first N rows")
    args = parser.parse_args()

    with open(args.input, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    missing = REQUIRED_COLUMNS - set(rows[0]) if rows else REQUIRED_COLUMNS
    if missing:
        sys.exit(f"input is missing required column(s): {', '.join(sorted(missing))}")

    if args.limit:
        rows = rows[:args.limit]

    print(f"scoring {len(rows)} trials against {args.api}")

    # The first request wakes the instance if it has spun down. Give it a
    # moment rather than counting the cold start as a failure.
    try:
        with request.urlopen(f"{args.api}/health", timeout=120) as resp:
            print(f"  health: {json.load(resp)}")
    except Exception as exc:
        print(f"  health check failed ({exc}) -- continuing anyway", file=sys.stderr)

    scored = []
    for n, batch in enumerate(chunked(rows, BATCH_SIZE), start=1):
        items = [
            {
                "trial_id": r["trial_id"],
                "cancer_type": r["cancer_type"],
                "outcome_text": r["outcome_text"],
            }
            for r in batch
        ]
        for source, result in zip(batch, post_batch(args.api, items)):
            scored.append({
                "trial_id": result["trial_id"],
                "cancer_type": source["cancer_type"],
                "outcome_text": source["outcome_text"],
                "predicted_cognitive": result["predicted_cognitive"],
                "confidence": result["confidence"],
                "method": result["method"],
                "review_recommended": result["review_recommended"],
                "review_reason": result["review_reason"] or "",
            })
        print(f"  batch {n}: {len(scored)}/{len(rows)} done", end="\r", flush=True)

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(scored)

    positives = sum(1 for r in scored if r["predicted_cognitive"])
    flagged = sum(1 for r in scored if r["review_recommended"])
    print(f"\n\nwrote {args.out}")
    print(f"  predicted cognitive : {positives}/{len(scored)} ({positives / len(scored):.1%})")
    print(f"  flagged for review  : {flagged}/{len(scored)} ({flagged / len(scored):.1%})")


if __name__ == "__main__":
    main()
