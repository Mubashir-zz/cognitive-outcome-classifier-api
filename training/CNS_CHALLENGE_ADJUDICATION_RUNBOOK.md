# CNS challenge-set adjudication runbook

## Purpose

This workflow creates an independent human reference standard for the 300-record, positive-enriched CNS challenge set. The sealed key contains model outputs, sampling strata, and design weights. It must remain unopened until the completed human labels have been validated and frozen.

The challenge set is a final benchmark. Its labels must not be used to choose keywords, tune thresholds, train a model, edit prompts, or select among candidate models.

## Files and roles

- `CNS_challenge_set_300_ADJUDICATE_OFFLINE.html` — the reviewer interface. It contains only blinded row IDs and outcome text.
- `CNS_challenge_set_300_BLINDED.csv` — authoritative blinded source.
- `CNS_challenge_set_300_HUMAN_COMPLETED.csv` — created by the interface only after all 300 records are adjudicated.
- `CNS_challenge_set_300_SEALED_KEY.csv` — controlled-unblinding key. Do not open manually.
- `freeze_cns_challenge_labels.py` — validates and freezes the completed review without reading the sealed key.
- `analyze_cns_challenge_validation.R` — validates the frozen manifest first, then performs controlled unblinding and design-weighted analysis.
- `build_frozen_text_overlap_flags.py` — after label freeze, identifies outcome text previously seen under other v7 training IDs without opening the sealed key.
- `score_v8_frozen_challenge.py` — scores the original blinded text with the exact frozen v8 TorchScript/tokenizer artifacts. It verifies the frozen-label manifest but never opens the human-label CSV or sealed key.
- `classifier_release_gates_v1_2.json` and `evaluate_classifier_release_gates.py` — make v8 the primary detector and produce `HOLD` unless every prespecified scientific and technical gate passes, including strict text-disjoint sensitivity analysis.

## Step 1 — adjudicate offline

1. Open `CNS_challenge_set_300_ADJUDICATE_OFFLINE.html` in a browser. No Excel subscription, server, or internet connection is required.
2. Read the full outcome text for one record at a time.
3. Choose **Yes** only when the registered outcome explicitly measures cognition with a named cognitive instrument, a scored cognitive task, or a specific cognitive domain.
4. Choose **No** for performance status, neurological examination alone, motor function, balance or gait, generic quality of life, symptoms without a cognitive measure, imaging, survival, response, safety, pharmacokinetics, and laboratory outcomes.
5. Leave the label blank only when the text is genuinely insufficient or irreducibly ambiguous. A blank label requires **Low** confidence and an explanatory note.
6. Confidence describes certainty in the human decision, not confidence in any model.
7. Use **Export progress CSV** at the end of every review session. Browser storage is helpful but is not the authoritative backup.

The interface deliberately hides trial identity, registry, sampling stratum, detector output, probability, ontology flags, trap flags, and design weights.

## Step 2 — finish and freeze

When all 300 records are complete, click **Freeze completed review**. The browser downloads:

- `CNS_challenge_set_300_HUMAN_COMPLETED.csv`
- `CNS_challenge_set_300_HUMAN_COMPLETED.browser_manifest.json`

Move the completed CSV into this project directory. Then run:

```bash
python3 freeze_cns_challenge_labels.py \
  --blinded CNS_challenge_set_300_BLINDED.csv \
  --completed CNS_challenge_set_300_HUMAN_COMPLETED.csv \
  --reviewer "Mubashir Ahmad Khan" \
  --output CNS_challenge_set_300_HUMAN_FROZEN.csv
```

The command must create three read-only artifacts:

- `CNS_challenge_set_300_HUMAN_FROZEN.csv`
- `CNS_challenge_set_300_HUMAN_FROZEN.csv.manifest.json`
- `CNS_challenge_set_300_HUMAN_FROZEN.csv.sha256`

If any blinded text changed, a reviewer value is invalid, or an output already exists, the command stops without producing a replacement.

## Step 3 — freeze the training-text overlap audit

A blinded hash audit found that 3 of 300 challenge texts are identical, after Unicode/case/whitespace normalization, to outcome text present under other v7 training IDs. This was detected before human-label unblinding. Create record-level flags only after the human labels are frozen:

```bash
python3 build_frozen_text_overlap_flags.py \
  --frozen CNS_challenge_set_300_HUMAN_FROZEN.csv \
  --training-v7 "/Users/mac/Documents/Trial study/NLP project/cross cancer training /CROSS_CANCER_TRAINING_SET_v7.csv" \
  --output CNS_challenge_set_300_POST_FREEZE_TEXT_OVERLAP.csv
```

The output discloses overlap status but no model prediction. Do not create or inspect it before the human review is frozen.

## Step 4 — freeze v8 predictions while still blinded

After v8 candidate selection, quantization parity, and the one-time internal test are complete, score the untouched blinded challenge text. This command reads neither the frozen human-label CSV nor the sealed key:

```bash
python3 score_v8_frozen_challenge.py \
  --blinded CNS_challenge_set_300_BLINDED.csv \
  --frozen-manifest CNS_challenge_set_300_HUMAN_FROZEN.csv.manifest.json \
  --release v8_development_release_2026-08-20/v8_development_release.csv \
  --config v8_training_config.json \
  --selection /path/to/v8_candidate_selection.json \
  --quantization-manifest /path/to/v8_quantized_selected/quantization_manifest.json \
  --output CNS_challenge_set_300_V8_FROZEN_PREDICTIONS.csv
```

The command creates predictions plus a manifest binding the exact frozen-label provenance, blinded text, selection, serialized model, tokenizer, threshold, configuration, and development release.

## Step 5 — controlled unblinding and weighted validation

Only after the freeze command succeeds, run:

```bash
Rscript analyze_cns_challenge_validation.R \
  --frozen CNS_challenge_set_300_HUMAN_FROZEN.csv \
  --sealed-key CNS_challenge_set_300_SEALED_KEY.csv \
  --text-overlap-flags CNS_challenge_set_300_POST_FREEZE_TEXT_OVERLAP.csv \
  --v8-predictions CNS_challenge_set_300_V8_FROZEN_PREDICTIONS.csv \
  --output-dir cns_challenge_validation_results \
  --bootstrap 5000 \
  --seed 20260824
```

The enriched sample's raw confusion matrices are descriptive. Claims about the 2,093-record remaining frame must use the design-weighted estimates. The analysis reconstructs the frame through inverse-probability weights and reports v8 as the primary detector, with legacy BERT, keyword, and union results as comparators. It also reports confidence intervals, thresholds, subgroups, an internal error audit, and a strict text-disjoint subpopulation analysis; that analysis does not claim to reconstruct the excluded overlap units.

The unblinded internal files contain trial identity and detector results. Keep them private. A public report should exclude free-text reviewer notes and record-level error rows unless they have been manually disclosure-checked.

## Step 6 — release decision

Technical results must be derived from independent evidence; a manually written JSON file of passing booleans is not accepted by the v1.2 gate.

1. Run `staging_classifier_v2/scripts/verify_v8_runtime_parity.py` against the exact quantized artifact and a frozen private regression fixture.
2. Run `staging_classifier_v2/scripts/benchmark_staging.py` against the separate v8 staging deployment, pinned to the same quantization-manifest hash.
3. Run `staging_classifier_v2/scripts/run_classifier_release_tests.py` with the exact 40-character staging build commit.
4. Record the independently tested rollback artifact and restore drill.
5. Derive the technical file:

```bash
python3 staging_classifier_v2/scripts/collect_classifier_technical_results.py \
  --benchmark v8_staging_benchmark.json \
  --parity v8_runtime_parity.json \
  --internal-test v8_internal_test_once/internal_test_evaluation.json \
  --test-evidence classifier_test_evidence.json \
  --rollback-record rollback_record.json \
  --output classifier_technical_results.json
```

The collector cross-checks the deployed runtime and build, full-text behavior, security responses, model/tokenizer/provenance hashes, exact analytic/API parity, frozen regression, one-time internal test, fixed test suites, resource measurements, and rollback record. Any missing or mismatched input produces `HOLD`.

Then run:

```bash
python3 evaluate_classifier_release_gates.py \
  --gates classifier_release_gates_v1_2.json \
  --validation-dir cns_challenge_validation_results \
  --technical-results classifier_technical_results.json \
  --output classifier_release_decision.json
```

The evaluator verifies that the technical file came from the current collector implementation. It never deploys. A passing result means only `PASS_FOR_MANUAL_PROMOTION_REVIEW`. Any missing result or failed gate produces `HOLD`.

## Recovery rules

- If the browser closes, reopen the same HTML file in the same browser profile; progress is stored locally.
- If browser storage is lost, import the most recent progress CSV.
- If a record was labeled incorrectly before browser freeze, correct it and export again.
- If an error is discovered after the authoritative command-line freeze, do not edit the frozen file. Preserve it and create a versioned amendment with a written reason and a new hash.
- Never inspect the sealed key to make adjudication easier. That would invalidate the benchmark.

## Verification commands

```bash
python3 validate_cns_adjudication_app.py \
  --app CNS_challenge_set_300_ADJUDICATE_OFFLINE.html \
  --blinded CNS_challenge_set_300_BLINDED.csv

python3 -m unittest tests/test_cns_challenge_workflow.py -v
```

Both commands must pass before adjudication starts or resumes.
