# Evidence-grounded generative extractor — specification v1

## Role in the platform

The generative component is a structured evidence extractor for difficult records. It is not the gold standard and does not replace the deterministic BERT and keyword detectors. It runs on detector disagreements, BERT uncertainty, truncation, trap-language matches, and cases without adequate deterministic evidence.

The synchronous research API should return the deterministic decision immediately. Generative extraction should run asynchronously so latency, cost, or provider failure cannot block the base classifier.

## Inputs

- internal record ID
- complete registered outcome text
- SHA-256 of the exact source text
- deterministic detector outputs and review reasons
- immutable prompt version and model revision

Trial text from medical charts is out of scope for the public service. Any future chart workflow requires a separate private deployment, institutional approval, encryption, access control, audit logging, and a retention policy.

## Output contract

`generative_extractor_schema_v1.json` defines the machine contract. The extractor returns:

- `cognitive_outcome`, `not_cognitive_outcome`, or `abstain`
- named cognitive instruments
- normalized cognitive domains
- assessment timepoints
- exact source-text spans with zero-based, end-exclusive character offsets
- a short rationale
- immutable model and prompt provenance

The output omits the full source text. `validate_generative_extraction.py` receives the source separately and rejects a wrong hash, fabricated evidence, invalid offsets, unsupported positive classification, missing abstention reason, or non-zero temperature.

## Decision policy

The generative output is advisory until independently validated.

1. If deterministic and generative decisions agree and exact evidence validates, retain the deterministic class and attach the structured evidence.
2. If they disagree, route the record to human review.
3. If the generative model abstains, route to human review.
4. If grounding validation fails, discard the extraction and log a structured failure without storing source text in the error.
5. A generative decision must never silently overwrite a frozen analytic label.

## Evaluation

Evaluation uses development data that are disjoint from the 300-record challenge benchmark. Measure:

- JSON/schema success rate
- exact-span grounding rate
- hallucinated-span rate
- classification sensitivity, specificity, balanced accuracy, and MCC
- instrument and domain exact-match performance
- abstention rate and selective performance after abstention
- deterministic/generative disagreement rate
- latency, cost, and provider-error rate

Model and prompt selection must be frozen before the challenge key is opened. The challenge set is evaluated once for release evidence; it is not an iterative prompt-tuning set.

## Operational record

Each extraction event should store only the minimum required research metadata:

- record ID and source-text SHA-256
- BERT probability, keyword evidence, and deterministic decision rule version
- validated structured extraction
- model revision, prompt SHA-256, configuration version, and timestamp
- review-queue status and final human disposition when available

Any public release must remove trial text, reviewer notes, and record-level error details unless disclosure review permits them.
