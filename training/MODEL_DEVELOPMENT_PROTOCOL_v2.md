# CNS classifier model-development protocol v2

## Objective

Build a reproducible research-screening classifier for registered neurocognitive outcomes. The model must process full outcome text, preserve an auditable evidence path, and remain separate from the untouched 300-record external benchmark until training, calibration, selection, quantization, and internal evaluation are frozen.

## Frozen data partitions

The v7 source audit found 2,269 rows with valid binary labels and no conflicting normalized-text labels. Before splitting, nine rows whose normalized text matched three blinded challenge texts were excluded. Thirteen additional duplicate-text rows were removed deterministically. The resulting v8 development release contains 2,247 unique trial IDs and unique normalized texts:

- training: 1,573 records;
- calibration: 337 records;
- internal test: 337 records;
- external challenge: 300 separate blinded records, excluded from every development decision.

Splitting is deterministic within cancer-type × label strata using the frozen seed and SHA-256 ordering. The release, split manifest, challenge-exclusion hashes, and configuration are mutually hash-bound.

## Candidate architecture and training

The base encoder is `emilyalsentzer/Bio_ClinicalBERT` at commit `d5892b39a4adaed74b92212a44081509db72f87b`. Complete outcome text is tokenized into overlapping 382-content-token chunks within 384-token model sequences, with 64-token overlap. The trial probability is the maximum chunk probability. Each chunk receives inverse chunk-count weight, giving equal expected trial contribution under uniform chunk minibatch sampling.

Five seeds (20260820–20260824) use four epochs, learning rate 2×10⁻⁵, weight decay 0.01, warm-up fraction 0.10, batch size 8, gradient accumulation 4, and maximum gradient norm 1. Training reads only the training split. Each candidate scores the calibration split and writes immutable model, prediction, environment, and SHA-256 records. Interrupted candidates resume from the latest epoch checkpoint; completed candidates cannot be overwritten.

## Calibration-only selection

The operating threshold is selected on CNS calibration records only. A threshold is feasible only if sensitivity is at least 0.90 and specificity at least 0.95. Threshold ties are resolved by MCC, balanced accuracy, sensitivity, specificity, then lower threshold. Seed selection uses the same metrics, then lower seed. If no candidate has a feasible threshold, quantization and internal-test evaluation stop.

## Deployment artifact and internal test

The selected float candidate is dynamically quantized to int8 for linear layers on CPU. Both the eager quantized model and the reloaded serialized TorchScript artifact must preserve the configured fraction of calibration classifications (currently 100%) and have maximum absolute probability change no greater than 0.02. The exact serialized model and every tokenizer file are hash-bound.

Only that serialized artifact may consume the internal test. Evaluation requires an explicit acknowledgement, writes a selection-level lock, and reports overall and cancer-specific metrics with 5,000 bootstrap replicates. Internal-test results cannot be used to choose another seed or threshold; failure ends this candidate version.

## External benchmark and promotion

Human adjudication is completed offline while identities, model outputs, strata, and weights remain sealed. Labels are frozen first. V8 then scores the original blinded text—not the human-label file—and writes a prediction manifest bound to the blinded source, frozen-label manifest, model, tokenizer, threshold, configuration, and release.

Controlled unblinding evaluates v8 as the primary detector with legacy BERT, keyword, and union outputs as comparators. Remaining-frame estimates use inverse-probability design weights and stratified finite-population variance; composite-metric intervals use a Rao–Wu rescaled bootstrap. A strict normalized-text-disjoint analysis is mandatory. `classifier_release_gates_v1_2.json` is conjunctive: any missing or failed scientific or technical result yields `HOLD`, and no script deploys automatically.

Challenge failures may inform a future version only. Any materially changed model requires a new untouched external benchmark.
