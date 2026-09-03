# Deployed-classifier evaluation

Run 3 September 2026 against the live endpoint. Reproduce with:

```bash
python validation/evaluate_deployed_classifier.py path/to/MASTER_gold_standard_1888.csv
```

196 trials, drawn label-balanced at 25 per (cancer type × label) cell, seed 2026.
Four of the 200 sampled had no retrievable outcome module.

## What is being measured, and what is not

Outcome text comes from the **ClinicalTrials.gov v2 API**, not from the copy
stored in the dataset. That distinction is the whole reason this evaluation
exists — see [Truncation](#truncation-in-the-stored-dataset) below.

These trials are in the model's development set, so **agreement is bounded above
by memorisation and is not an external-validity estimate.** The routing
behaviour, the review-flag rate and the truncation comparison do not depend on
the labels, and are the substantive results.

## Results

| | n | Agreement | Sensitivity | Specificity | Flagged |
|---|---|---|---|---|---|
| **Overall** | 196 | 95.9% | 93.0% | 99.0% | 32.1% |
| CNS *(BERT)* | 49 | 83.7% | 72.0% | 95.8% | 36.7% |
| Breast *(keyword)* | 50 | 100% | 100% | 100% | 30.0% |
| Lung *(keyword)* | 49 | 100% | 100% | 100% | 32.7% |
| Head & Neck *(keyword)* | 48 | 100% | 100% | 100% | 29.2% |

Confusion, overall: 93 TP, 95 TN, 1 FP, 7 FN.

Routing behaved as designed: 49 CNS trials to BERT, 147 to the keyword rule.

Two things worth saying plainly.

**The keyword rule is perfect on all three non-CNS types.** 147/147, no errors in
either direction. That is the strongest possible confirmation of the design
decision not to route those types through a model — a transformer cannot beat
100%, and the Phase 2 comparison that found no lift was reading the situation
correctly.

**CNS remains the hard case, and it fails in the safer direction.** Sensitivity
72%, specificity 96% — the model misses cognitive outcomes more often than it
invents them. For a screening tool that is the right direction of error, but it
means the flagged queue is load-bearing, not decorative.

## Where CNS went wrong

| Trial | Confidence | Flagged | Registry text |
|---|---|---|---|
| NCT01305122 | 0.062 | no | **9 chars** |
| NCT01894061 | 0.007 | no | 3,270 |
| NCT05535166 | 0.011 | no | **20,475 chars** |
| NCT02125786 | 0.272 | yes | 6,932 |
| NCT04243005 | 0.072 | yes | 3,787 |
| NCT03345095 | 0.019 | yes | 2,699 |
| NCT03868943 | 0.479 | yes | 3,492 |
| NCT02082119 (FP) | 0.994 | no | — |

**4 of the 8 errors were flagged for human review.** Half the failures get caught
by the safety layer; half do not. That is a real limit and is not smoothed over
here.

Two of the misses are explained by their input rather than by the model.
NCT01305122 has nine characters of outcome text in the registry — nothing to
classify. NCT05535166 has 20,475 characters, and the model reads the first 256
tokens; if the cognitive outcome is listed twentieth, it is not in the window.
That is the 256-token limit doing exactly what it says on the tin, and it argues
for chunking long outcome lists rather than for retraining.

## Truncation in the stored dataset

The reason this script pulls from the registry:

| | Median characters |
|---|---|
| `Outcome text (for cognition check)` in the dataset | 400 |
| Full ClinicalTrials.gov outcome module | 1,849 |

172 of 196 trials have less text stored than the registry holds. The stored
column keeps roughly a fifth of the outcome text and routinely cuts off the
outcome that names the instrument.

An initial run against the stored column produced 74 false negatives and **zero**
false positives — the signature of missing text, not of a bad model. Scoring it
would have measured truncation.

## Operating notes

`MAX_BATCH_SIZE` is 6 and the endpoint accepts it, but on full-length registry
text the 512MB instance restarts (502/503) at 6 and again at 3. One item per
request with ~2s pacing is the only configuration that completes 200 trials
without a restart. Results are cached per trial so a restart cannot lose the run.

CNS inference runs ~4s per trial on the free-tier CPU; the keyword route is
~0.1s.

## Files

```
evaluate_deployed_classifier.py   the script
deployed_classifier_scored.csv    per-trial predictions with human labels
deployed_classifier_summary.json  metrics
ctgov_outcome_text_cache.json     fetched registry text, so re-runs are offline
scored_cache.json                 per-trial API responses, for resume
```
