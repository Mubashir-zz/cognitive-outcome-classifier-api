# CNS classifier v8 GPU training runbook

## Scope

This bundle trains five prespecified Bio_ClinicalBERT candidates from the leakage-filtered 2,247-record v8 development release. It contains no challenge labels, sealed-key identities, model outputs, sampling strata, or design weights.

## Frozen inputs

- Development release: 2,247 unique trial IDs and unique normalized outcome texts.
- Split: train 1,573; calibration 337; internal test 337.
- Nine v7 rows matching three challenge-text hashes were excluded before splitting.
- Base model: `emilyalsentzer/Bio_ClinicalBERT` at immutable revision `d5892b39a4adaed74b92212a44081509db72f87b`.
- Seeds: 20260820 through 20260824.
- Selection: CNS calibration split only; joint minimum sensitivity 0.90 and specificity 0.95, then MCC and balanced accuracy.
- Internal test: evaluated once only after seed and threshold are frozen.

## Run

1. Open `CNS_v8_frozen_GPU_training_v7.ipynb` in Google Colab.
2. Select a GPU runtime.
3. Ensure `CNS_V8_GPU_TRAINING_BUNDLE_V7_2026-08-20.zip` is in My Drive root.
4. Run cells in order.
5. If Colab disconnects, rerun from the first cell. Completed seed artifacts are hash-verified and skipped; incomplete seeds resume from saved epoch checkpoints.
6. Do not rename or edit candidate folders. Candidate selection and quantization will fail closed if the release, configuration, model, or tokenizer artifacts differ from their frozen hashes.

## Expected Drive outputs

`MyDrive/cns_v8/v8_training_candidates_2026-08-20/`

- five `candidate_seed_*` directories;
- `v8_training_root_manifest.json` with status `ALL_PRESPECIFIED_CANDIDATES_COMPLETE`;
- `v8_candidate_selection.json`;
- `v8_quantized_selected/quantization_manifest.json` only if serialized parity passes;
- `v8_internal_test_once/internal_test_evaluation.json` after the one-time evaluation of the exact hash-verified serialized quantized artifact.

No candidate is production-approved by training alone. After human labels are frozen, `score_v8_frozen_challenge.py` scores the original blinded text without opening the labels or sealed key. Controlled unblinding then evaluates v8 as the primary detector under `classifier_release_gates_v1_2.json`.

V7 also carries the staging v8 runtime, exact analytic/API parity verifier, deployed benchmark, fixed test-evidence runner, and hash-bound technical-results collector. These do not run during model fitting. They are used only after a candidate is frozen and a separate staging service exists.
