# Model Card — Neurocognitive Outcome Classifier (v7, CNS)

## What it does

Given the outcome-measure text of an oncology clinical trial, predicts whether
that trial registers a genuine neurocognitive assessment.

The system is a hybrid, not a single model:

| Cancer type | Route | Why |
|---|---|---|
| CNS | Fine-tuned Bio_ClinicalBERT (v7), int8-quantized, TorchScript-traced | CNS trials use the most heterogeneous and trap-prone cognitive language; a keyword rule underperforms here |
| Breast, Lung, Head & Neck | Keyword-presence rule over a validated 66-term list | Instrument vocabulary is standardised in these types; the rule already sits at ~99–100% accuracy and a model adds nothing |

Routing by cancer type came out of the Phase 2 comparison, not from a
preference for either method — see `results/PHASE2_RESULTS.md` in the
[classifier development repo](https://github.com/Mubashir-zz/neurocognitive-outcome-classifier).

## Training data

2,269 hand-verified trials across four cancer types, drawn from
ClinicalTrials.gov and international registries (ChiCTR, EU-CTR, JPRN, CTRI,
ANZCTR and others). Each trial's outcome text was read individually and
classified by hand — not keyword-matched — specifically to capture the
distinctions a keyword rule cannot make:

- a real cognitive instrument (MMSE, MoCA, HVLT-R, FACT-Cog, TMT, COWAT) counts
- the NANO neurologic exam does not
- Karnofsky / ECOG performance status does not
- a quality-of-life questionnaire's cognitive *subscale* does not

That last distinction is where most automated approaches fail, and it is the
reason the labelling was done by reading rather than by rule.

## Base model

`emilyalsentzer/Bio_ClinicalBERT` — BioBERT further pre-trained on MIMIC-III
clinical notes. BERT-base architecture, 12 layers, 768 hidden, 110M parameters.

Fine-tuned with class-weighted cross-entropy (No = 1.183, Yes = 0.866) for
4 epochs. Max input 256 tokens at serving time, 512 during training.

## Deployment form

The served model is int8-quantized and TorchScript-traced: 169MB against the
original 433MB float32 checkpoint. This was not a research choice — Render's
free tier caps at 512MB RAM and the float32 model crashed it. Post-quantization
the model was re-checked against the development test cases; no new failure
patterns appeared.

## Performance

CNS route, held-out test set (n = 85):

| Metric | Value |
|---|---|
| Sensitivity | 0.771 |
| Specificity | 0.514 |
| AUROC (all types) | 0.894 |
| F1 (all types) | 0.834 |

Keyword route, held-out test sets: 99.3% (Breast, n=152), 100% (Lung, n=77),
100% (Head & Neck, n=40).

Read the sensitivity number in context. The honest framing is that BERT beats
the keyword baseline **only** for CNS, and even there the margin is modest.
For the other three cancer types it does not beat the baseline at all, which
is why they are not routed through it.

## Review flagging — the part that matters clinically

The model is a screening aid, so the design assumes it will be wrong and
routes uncertain cases to a human rather than absorbing them silently. A
prediction is flagged when either:

1. **The BERT probability lands in 0.25–0.75.** Genuinely uncertain, whichever
   side of the 0.5 threshold it falls on.
2. **The text matches a documented failure pattern.** These are flagged even
   when the model is confident, because on these categories confidence is not
   informative:

   | Pattern | Trap |
   |---|---|
   | `qlq-c30`, `eortc` | QoL cognitive subscale read as a cognitive endpoint |
   | `karnofsky`, `kps` | Performance status read as cognition |
   | `rcbv`, `rcbf`, `suvr`, `dsc-mri`, `pet/ct` | Imaging biomarker |
   | `hospitalization`, `emergency department` | Healthcare utilisation |
   | `platelet`, `thrombocytopenia` | Haematological toxicity |

## Known limitations

- The EORTC QLQ-C30 multi-subscale pattern remains unresolved after five
  targeted retraining rounds. It is handled by flagging, not by the model.
- Residual confident-but-wrong rate on content categories absent from training,
  roughly 1 in 100–250 predictions on audit testing.
- Fixes to one failure pattern have occasionally regressed unrelated
  previously-fixed cases. The flagging layer is designed around that
  instability rather than around a fixed, known error set.
- 256-token truncation at serving. Trials with very long multi-outcome lists —
  mostly paediatric CNS trials — have later outcomes cut off.
- Specificity on CNS is 0.514. The model over-calls cognition. For a screening
  tool that is the preferable direction of error, but it means positive
  predictions carry real false-positive load and the flagged queue is not
  optional.

## Intended use

Research screening: narrowing a large registry pull to a reviewable candidate
set. It is not a substitute for reading the trial record, and it should not be
used to make a final determination about any individual trial.

## Not intended for

Clinical decision-making, patient-level inference, or any use where a wrong
label is not caught by a human downstream.
