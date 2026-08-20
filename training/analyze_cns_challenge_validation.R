#!/usr/bin/env Rscript

# Controlled unblinding and design-weighted validation for the frozen CNS
# challenge set. This script validates the frozen human-label manifest before
# it reads the sealed key.

suppressPackageStartupMessages({
  if (!requireNamespace("digest", quietly = TRUE)) stop("Package 'digest' is required.")
  if (!requireNamespace("jsonlite", quietly = TRUE)) stop("Package 'jsonlite' is required.")
})

args <- commandArgs(trailingOnly = TRUE)
get_arg <- function(flag, default = NULL) {
  hit <- match(flag, args)
  if (is.na(hit)) return(default)
  if (hit == length(args)) stop("Missing value after ", flag)
  args[[hit + 1L]]
}

frozen_path <- get_arg("--frozen")
key_path <- get_arg("--sealed-key")
text_overlap_path <- get_arg("--text-overlap-flags")
v8_predictions_path <- get_arg("--v8-predictions")
output_dir <- get_arg("--output-dir", "cns_challenge_validation_results")
bootstrap_replicates <- as.integer(get_arg("--bootstrap", "5000"))
seed <- as.integer(get_arg("--seed", "20260824"))
if (is.null(frozen_path) || is.null(key_path)) {
  stop("Usage: Rscript analyze_cns_challenge_validation.R --frozen <FROZEN.csv> --sealed-key <SEALED_KEY.csv> [--text-overlap-flags POST_FREEZE_FLAGS.csv] [--v8-predictions FROZEN_V8_PREDICTIONS.csv] [--output-dir DIR] [--bootstrap 5000]")
}
manifest_path <- paste0(frozen_path, ".manifest.json")
if (!file.exists(frozen_path) || !file.exists(manifest_path)) stop("Frozen labels and their manifest are both required.")
if (!file.exists(key_path)) stop("Sealed key not found: ", key_path)
if (dir.exists(output_dir) || file.exists(output_dir)) stop("Refusing to overwrite existing output path: ", output_dir)

sha256_file <- function(path) digest::digest(file = path, algo = "sha256", serialize = FALSE)
manifest <- jsonlite::fromJSON(manifest_path, simplifyVector = TRUE)
if (!identical(manifest$status, "FROZEN_HUMAN_LABELS")) stop("Manifest does not certify frozen human labels.")
if (!identical(unname(manifest$frozen_sha256), sha256_file(frozen_path))) stop("Frozen-label SHA-256 does not match its manifest.")
if (!identical(as.integer(manifest$records), 300L)) stop("Manifest must certify exactly 300 records.")

expected_label_headers <- c(
  "Row_ID", "Outcome_Text_Part_1", "Outcome_Text_Part_2", "Measures_Cognition_Y_N",
  "Reviewer_Confidence_High_Medium_Low", "Notes"
)
labels <- read.csv(frozen_path, check.names = FALSE, stringsAsFactors = FALSE, na.strings = character())
if (!identical(names(labels), expected_label_headers)) stop("Unexpected frozen-label schema.")
if (nrow(labels) != 300L || anyDuplicated(labels$Row_ID)) stop("Frozen labels must contain 300 unique Row_ID values.")
label_text <- trimws(labels$Measures_Cognition_Y_N)
confidence <- trimws(labels$Reviewer_Confidence_High_Medium_Low)
if (any(!label_text %in% c("", "Yes", "No"))) stop("Frozen labels contain an invalid cognition value.")
if (any(!confidence %in% c("High", "Medium", "Low"))) stop("Every frozen record must have valid reviewer confidence.")
ambiguous <- label_text == ""
if (any(ambiguous & (confidence != "Low" | !nzchar(trimws(labels$Notes))))) {
  stop("Every ambiguous label must have Low confidence and a note.")
}
if (all(ambiguous)) stop("All 300 labels are ambiguous; validation cannot proceed.")

# V8 predictions are validated before controlled unblinding. Their manifest
# must bind them to this exact frozen file; no human-truth field is accepted.
v8_predictions <- NULL
v8_manifest <- NULL
if (!is.null(v8_predictions_path)) {
  v8_manifest_path <- paste0(v8_predictions_path, ".manifest.json")
  if (!file.exists(v8_predictions_path) || !file.exists(v8_manifest_path)) stop("V8 predictions and their manifest are both required.")
  v8_manifest <- jsonlite::fromJSON(v8_manifest_path, simplifyVector = TRUE)
  if (!identical(v8_manifest$status, "FROZEN_V8_CHALLENGE_PREDICTIONS")) stop("V8 prediction manifest has the wrong status.")
  if (!identical(as.integer(v8_manifest$records), 300L)) stop("V8 prediction manifest must certify 300 records.")
  if (!identical(unname(v8_manifest$frozen_labels_sha256), sha256_file(frozen_path))) stop("V8 predictions are not bound to these frozen labels.")
  if (!identical(unname(v8_manifest$output_sha256), sha256_file(v8_predictions_path))) stop("V8 prediction SHA-256 does not match its manifest.")
  if (!identical(v8_manifest$human_truth_used_for_scoring, FALSE) || !identical(v8_manifest$sealed_key_opened_by_this_script, FALSE)) stop("V8 scoring provenance does not certify blinded inference.")
  v8_predictions <- read.csv(v8_predictions_path, check.names = FALSE, stringsAsFactors = FALSE, na.strings = character())
  expected_v8_headers <- c("Row_ID", "V8_Probability", "V8_Prediction")
  if (!identical(names(v8_predictions), expected_v8_headers)) stop("Unexpected V8 prediction schema.")
  if (nrow(v8_predictions) != 300L || anyDuplicated(v8_predictions$Row_ID) || !setequal(v8_predictions$Row_ID, labels$Row_ID)) stop("V8 predictions must match all 300 frozen Row_ID values.")
  v8_predictions <- v8_predictions[match(labels$Row_ID, v8_predictions$Row_ID), , drop = FALSE]
  v8_predictions$V8_Probability <- as.numeric(v8_predictions$V8_Probability)
  v8_predictions$V8_Prediction <- as.integer(v8_predictions$V8_Prediction)
  if (any(!is.finite(v8_predictions$V8_Probability) | v8_predictions$V8_Probability < 0 | v8_predictions$V8_Probability > 1)) stop("Invalid V8 probability.")
  if (any(!v8_predictions$V8_Prediction %in% c(0L, 1L))) stop("Invalid V8 prediction.")
  v8_threshold <- as.numeric(v8_manifest$selected_threshold)
  if (!is.finite(v8_threshold) || v8_threshold < 0 || v8_threshold > 1) stop("Invalid frozen V8 threshold.")
  if (any(v8_predictions$V8_Prediction != as.integer(v8_predictions$V8_Probability >= v8_threshold))) stop("V8 predictions do not match the frozen threshold.")
}

# The sealed key is read only after every freeze and schema gate above passes.
key <- read.csv(key_path, check.names = FALSE, stringsAsFactors = FALSE, na.strings = c("", "NA"))
required_key <- c(
  "Row_ID", "Trial_ID", "Sampling_Stratum", "Frame_N", "Sample_N",
  "Inclusion_Probability", "Design_Weight", "Full_Text_Characters",
  "Ontology_Hit", "Ontology_Families", "Trap_Hit", "Trap_Families",
  "BERT_Prediction", "BERT_Probability", "Existing_v2_Keyword_Hit",
  "Existing_v2_Union_Label"
)
if (!all(required_key %in% names(key))) stop("Sealed key is missing required fields: ", paste(setdiff(required_key, names(key)), collapse = ", "))
if (nrow(key) != 300L || anyDuplicated(key$Row_ID)) stop("Sealed key must contain 300 unique Row_ID values.")
if (!setequal(labels$Row_ID, key$Row_ID)) stop("Frozen labels and sealed key contain different Row_ID sets.")
dat <- merge(labels, key, by = "Row_ID", all = FALSE, sort = FALSE)
dat <- dat[match(labels$Row_ID, dat$Row_ID), , drop = FALSE]
if (!identical(dat$Row_ID, labels$Row_ID)) stop("Row alignment failed during controlled unblinding.")

as_binary <- function(x) {
  y <- tolower(trimws(as.character(x)))
  if (any(!is.na(y) & !y %in% c("true", "false", "1", "0"))) stop("Invalid binary field in sealed key.")
  ifelse(is.na(y), NA_integer_, as.integer(y %in% c("true", "1")))
}
dat$Truth <- ifelse(label_text == "Yes", 1L, ifelse(label_text == "No", 0L, NA_integer_))
dat$BERT <- as_binary(dat$BERT_Prediction)
dat$Keyword <- as_binary(dat$Existing_v2_Keyword_Hit)
dat$Union <- as_binary(dat$Existing_v2_Union_Label)
dat$Ontology <- as_binary(dat$Ontology_Hit)
dat$Trap <- as_binary(dat$Trap_Hit)
dat$Weight <- as.numeric(dat$Design_Weight)
dat$Frame <- as.numeric(dat$Frame_N)
dat$Sample <- as.numeric(dat$Sample_N)
dat$Pi <- as.numeric(dat$Inclusion_Probability)
dat$BERT_Probability <- as.numeric(dat$BERT_Probability)
if (!is.null(v8_predictions)) {
  if (!identical(dat$Row_ID, v8_predictions$Row_ID)) stop("V8 row alignment failed during controlled unblinding.")
  dat$V8 <- v8_predictions$V8_Prediction
  dat$V8_Probability <- v8_predictions$V8_Probability
}
dat$Training_Text_Overlap <- NA
if (!is.null(text_overlap_path)) {
  overlap_manifest_path <- paste0(text_overlap_path, ".manifest.json")
  if (!file.exists(text_overlap_path) || !file.exists(overlap_manifest_path)) stop("Text-overlap flags and their manifest are both required.")
  overlap_manifest <- jsonlite::fromJSON(overlap_manifest_path, simplifyVector = TRUE)
  if (!identical(overlap_manifest$status, "POST_FREEZE_TRAINING_TEXT_OVERLAP_FLAGS")) stop("Text-overlap manifest has the wrong status.")
  if (!identical(as.integer(overlap_manifest$records), 300L)) stop("Text-overlap manifest must certify 300 records.")
  if (!identical(unname(overlap_manifest$frozen_labels_sha256), sha256_file(frozen_path))) stop("Text-overlap flags were not built from these frozen labels.")
  if (!identical(unname(overlap_manifest$output_sha256), sha256_file(text_overlap_path))) stop("Text-overlap flag SHA-256 does not match its manifest.")
  overlap <- read.csv(text_overlap_path, check.names = FALSE, stringsAsFactors = FALSE, na.strings = character())
  expected_overlap_headers <- c("Row_ID", "Training_Text_Overlap", "Matched_Training_Rows", "Normalized_Text_SHA256")
  if (!identical(names(overlap), expected_overlap_headers)) stop("Unexpected text-overlap flag schema.")
  if (nrow(overlap) != 300L || anyDuplicated(overlap$Row_ID) || !setequal(overlap$Row_ID, dat$Row_ID)) stop("Text-overlap flags must match all 300 Row_ID values.")
  overlap <- overlap[match(dat$Row_ID, overlap$Row_ID), , drop = FALSE]
  overlap_value <- tolower(trimws(overlap$Training_Text_Overlap))
  if (any(!overlap_value %in% c("true", "false"))) stop("Invalid Training_Text_Overlap value.")
  if (any(as.integer(overlap$Matched_Training_Rows) < 0L)) stop("Matched_Training_Rows cannot be negative.")
  dat$Training_Text_Overlap <- overlap_value == "true"
  dat$Matched_Training_Rows <- as.integer(overlap$Matched_Training_Rows)
  dat$Normalized_Text_SHA256 <- overlap$Normalized_Text_SHA256
}
if (any(!is.finite(dat$Weight) | dat$Weight <= 0)) stop("Invalid design weights.")
if (any(abs(dat$Weight - 1 / dat$Pi) > 1e-5)) stop("Design weights do not equal inverse inclusion probabilities.")
if (abs(sum(dat$Weight) - 2093) > 1e-5) stop("Design weights do not reconstruct the 2,093-record frame.")
stratum_check <- unique(dat[c("Sampling_Stratum", "Frame", "Sample")])
if (any(duplicated(stratum_check$Sampling_Stratum))) stop("Frame_N or Sample_N varies within a sampling stratum.")
observed_n <- table(dat$Sampling_Stratum)
if (any(as.numeric(observed_n[stratum_check$Sampling_Stratum]) != stratum_check$Sample)) stop("Observed sample counts do not match Sample_N.")

metric_from_cells <- function(tp, tn, fp, fn) {
  tp <- as.numeric(tp); tn <- as.numeric(tn); fp <- as.numeric(fp); fn <- as.numeric(fn)
  safe <- function(a, b) if (is.finite(b) && b > 0) a / b else NA_real_
  sensitivity <- safe(tp, tp + fn)
  specificity <- safe(tn, tn + fp)
  ppv <- safe(tp, tp + fp)
  npv <- safe(tn, tn + fn)
  accuracy <- safe(tp + tn, tp + tn + fp + fn)
  f1 <- safe(2 * tp, 2 * tp + fp + fn)
  balanced <- mean(c(sensitivity, specificity), na.rm = TRUE)
  mcc_den <- sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
  mcc <- safe(tp * tn - fp * fn, mcc_den)
  c(Sensitivity = sensitivity, Specificity = specificity, PPV = ppv, NPV = npv,
    Accuracy = accuracy, F1 = f1, Balanced_Accuracy = balanced, MCC = mcc)
}

confusion <- function(truth, prediction, weight) {
  keep <- !is.na(truth) & !is.na(prediction) & !is.na(weight)
  truth <- truth[keep]; prediction <- prediction[keep]; weight <- weight[keep]
  c(
    TP = sum(weight[truth == 1 & prediction == 1]),
    TN = sum(weight[truth == 0 & prediction == 0]),
    FP = sum(weight[truth == 0 & prediction == 1]),
    FN = sum(weight[truth == 1 & prediction == 0])
  )
}

ratio_taylor <- function(y, x, dat, alpha = 0.05) {
  keep <- !is.na(y) & !is.na(x) & !is.na(dat$Weight)
  y <- y[keep]; x <- x[keep]; d <- dat[keep, , drop = FALSE]
  y_total <- sum(d$Weight * y); x_total <- sum(d$Weight * x)
  if (x_total <= 0) return(c(estimate = NA, lower = NA, upper = NA, numerator = y_total, denominator = x_total, se = NA))
  estimate <- y_total / x_total
  residual <- y - estimate * x
  variance_total <- 0
  for (stratum in unique(d$Sampling_Stratum)) {
    ii <- which(d$Sampling_Stratum == stratum)
    n_h <- unique(d$Sample[ii]); N_h <- unique(d$Frame[ii])
    if (length(n_h) != 1L || length(N_h) != 1L) stop("Inconsistent stratum metadata.")
    if (n_h > 1 && n_h < N_h) variance_total <- variance_total + N_h^2 * (1 - n_h / N_h) * stats::var(residual[ii]) / n_h
  }
  se <- sqrt(max(0, variance_total / x_total^2))
  z <- stats::qnorm(1 - alpha / 2)
  if (is.finite(estimate) && estimate > 0 && estimate < 1 && is.finite(se)) {
    eta <- stats::qlogis(estimate); se_eta <- se / (estimate * (1 - estimate))
    lower <- stats::plogis(eta - z * se_eta); upper <- stats::plogis(eta + z * se_eta)
  } else {
    lower <- max(0, estimate - z * se); upper <- min(1, estimate + z * se)
  }
  c(estimate = estimate, lower = lower, upper = upper, numerator = y_total, denominator = x_total, se = se)
}

exact_ratio <- function(success, eligible) {
  success <- as.integer(success); eligible <- as.integer(eligible)
  n <- sum(eligible); x <- sum(success)
  if (n == 0) return(c(estimate = NA, lower = NA, upper = NA, numerator = x, denominator = n))
  test <- stats::binom.test(x, n)
  c(estimate = x / n, lower = unname(test$conf.int[1]), upper = unname(test$conf.int[2]), numerator = x, denominator = n)
}

weighted_auc <- function(truth, probability, weight) {
  keep <- !is.na(truth) & !is.na(probability) & !is.na(weight)
  truth <- truth[keep]; probability <- probability[keep]; weight <- weight[keep]
  pos_total <- sum(weight[truth == 1]); neg_total <- sum(weight[truth == 0])
  if (pos_total <= 0 || neg_total <= 0) return(NA_real_)
  scores <- sort(unique(probability)); cumulative_negative <- 0; concordant <- 0
  for (score in scores) {
    at_score <- probability == score
    positive_weight <- sum(weight[at_score & truth == 1])
    negative_weight <- sum(weight[at_score & truth == 0])
    concordant <- concordant + positive_weight * (cumulative_negative + 0.5 * negative_weight)
    cumulative_negative <- cumulative_negative + negative_weight
  }
  concordant / (pos_total * neg_total)
}

detectors <- c("BERT", "Keyword", "Union", if (!is.null(v8_predictions)) "V8")
weighted_analysis_masks <- list("Design-weighted remaining frame" = rep(TRUE, nrow(dat)))
if (!all(is.na(dat$Training_Text_Overlap))) {
  weighted_analysis_masks[["Design-weighted strict text-disjoint subpopulation"]] <- !dat$Training_Text_Overlap
}
confusion_rows <- list(); metric_rows <- list(); row_id <- 1L
for (detector in detectors) {
  pred <- dat[[detector]]; truth <- dat$Truth
  for (analysis in c("Unweighted enriched challenge (descriptive only)", names(weighted_analysis_masks))) {
    analysis_mask <- if (startsWith(analysis, "Unweighted")) rep(TRUE, nrow(dat)) else weighted_analysis_masks[[analysis]]
    weights <- if (startsWith(analysis, "Unweighted")) rep(1, nrow(dat)) else dat$Weight
    weights <- weights * as.integer(analysis_mask)
    cells <- confusion(truth, pred, weights)
    confusion_rows[[length(confusion_rows) + 1L]] <- data.frame(
      Detector = detector, Analysis = analysis, TP = cells["TP"], TN = cells["TN"], FP = cells["FP"], FN = cells["FN"],
      stringsAsFactors = FALSE, check.names = FALSE
    )
    components <- list(
      Sensitivity = list(y = truth == 1 & pred == 1, x = truth == 1),
      Specificity = list(y = truth == 0 & pred == 0, x = truth == 0),
      PPV = list(y = truth == 1 & pred == 1, x = pred == 1),
      NPV = list(y = truth == 0 & pred == 0, x = pred == 0),
      Accuracy = list(y = truth == pred, x = !is.na(truth) & !is.na(pred))
    )
    core <- list()
    for (metric in names(components)) {
      item <- components[[metric]]
      valid <- !is.na(truth) & !is.na(pred) & analysis_mask
      y <- as.integer(item$y & valid); x <- as.integer(item$x & valid)
      estimate <- if (startsWith(analysis, "Unweighted")) exact_ratio(y, x) else ratio_taylor(y, x, dat)
      core[[metric]] <- estimate["estimate"]
      metric_rows[[row_id]] <- data.frame(
        Detector = detector, Analysis = analysis, Metric = metric,
        Estimate = unname(estimate["estimate"]), Lower_95 = unname(estimate["lower"]), Upper_95 = unname(estimate["upper"]),
        Numerator = unname(estimate["numerator"]), Denominator = unname(estimate["denominator"]),
        CI_Method = if (startsWith(analysis, "Unweighted")) "Clopper-Pearson exact" else "Taylor linearization with stratum FPC; logit CI",
        stringsAsFactors = FALSE, check.names = FALSE
      ); row_id <- row_id + 1L
    }
    composite <- metric_from_cells(cells["TP"], cells["TN"], cells["FP"], cells["FN"])
    for (metric in c("F1", "Balanced_Accuracy", "MCC")) {
      metric_rows[[row_id]] <- data.frame(
        Detector = detector, Analysis = analysis, Metric = metric, Estimate = unname(composite[metric]),
        Lower_95 = NA_real_, Upper_95 = NA_real_, Numerator = NA_real_, Denominator = NA_real_,
        CI_Method = if (startsWith(analysis, "Unweighted")) "Point estimate; enriched sample" else "Rao-Wu bootstrap CI added below",
        stringsAsFactors = FALSE, check.names = FALSE
      ); row_id <- row_id + 1L
    }
  }
}
confusion_table <- do.call(rbind, confusion_rows)
metric_table <- do.call(rbind, metric_rows)

replicate_weights <- function(dat) {
  rw <- dat$Weight
  for (stratum in unique(dat$Sampling_Stratum)) {
    ii <- which(dat$Sampling_Stratum == stratum); n_h <- length(ii); N_h <- unique(dat$Frame[ii])
    if (n_h > 1 && n_h < N_h) {
      counts <- as.vector(stats::rmultinom(1, size = n_h - 1L, prob = rep(1 / n_h, n_h)))
      scale <- sqrt(1 - n_h / N_h)
      rw[ii] <- dat$Weight[ii] * (1 - scale + scale * n_h / (n_h - 1) * counts)
    }
  }
  rw
}

set.seed(seed)
bootstrap_names <- c()
for (analysis in names(weighted_analysis_masks)) for (detector in detectors) for (metric in c("F1", "Balanced_Accuracy", "MCC")) {
  bootstrap_names <- c(bootstrap_names, paste(analysis, detector, metric, sep = "||"))
}
bootstrap_values <- matrix(NA_real_, nrow = bootstrap_replicates, ncol = length(bootstrap_names), dimnames = list(NULL, bootstrap_names))
for (b in seq_len(bootstrap_replicates)) {
  rw <- replicate_weights(dat)
  for (analysis in names(weighted_analysis_masks)) for (detector in detectors) {
    cells <- confusion(dat$Truth, dat[[detector]], rw * as.integer(weighted_analysis_masks[[analysis]]))
    values <- metric_from_cells(cells["TP"], cells["TN"], cells["FP"], cells["FN"])
    for (metric in c("F1", "Balanced_Accuracy", "MCC")) {
      bootstrap_values[b, paste(analysis, detector, metric, sep = "||")] <- values[metric]
    }
  }
}
for (analysis in names(weighted_analysis_masks)) for (detector in detectors) for (metric in c("F1", "Balanced_Accuracy", "MCC")) {
  ii <- metric_table$Detector == detector & metric_table$Analysis == analysis & metric_table$Metric == metric
  ci <- stats::quantile(bootstrap_values[, paste(analysis, detector, metric, sep = "||")], c(0.025, 0.975), na.rm = TRUE, names = FALSE)
  metric_table$Lower_95[ii] <- ci[1]; metric_table$Upper_95[ii] <- ci[2]
  metric_table$CI_Method[ii] <- paste0("Rao-Wu rescaled stratified bootstrap, B=", bootstrap_replicates)
}

valid_truth <- !is.na(dat$Truth)
weighted_prevalence <- ratio_taylor(as.integer(dat$Truth == 1 & valid_truth), as.integer(valid_truth), dat)
weighted_ambiguous <- ratio_taylor(as.integer(is.na(dat$Truth)), rep(1L, nrow(dat)), dat)
probability_detectors <- c("BERT", if (!is.null(v8_predictions)) "V8")
weighted_brier <- setNames(vapply(probability_detectors, function(detector) {
  probability <- dat[[paste0(detector, "_Probability")]]
  sum(dat$Weight[valid_truth] * (probability[valid_truth] - dat$Truth[valid_truth])^2) / sum(dat$Weight[valid_truth])
}, numeric(1)), probability_detectors)
weighted_auc_value <- setNames(vapply(probability_detectors, function(detector) {
  weighted_auc(dat$Truth, dat[[paste0(detector, "_Probability")]], dat$Weight)
}, numeric(1)), probability_detectors)

thresholds <- sort(unique(c(seq(0.05, 0.95, by = 0.05), 0.5)))
threshold_rows <- list()
threshold_rules <- c("BERT", "BERT_or_keyword", if (!is.null(v8_predictions)) "V8")
for (threshold in thresholds) {
  for (rule in threshold_rules) {
    probability <- if (rule == "V8") dat$V8_Probability else dat$BERT_Probability
    pred <- as.integer(probability >= threshold)
    if (rule == "BERT_or_keyword") pred <- as.integer(pred == 1 | dat$Keyword == 1)
    cells <- confusion(dat$Truth, pred, dat$Weight); metrics <- metric_from_cells(cells["TP"], cells["TN"], cells["FP"], cells["FN"])
    threshold_rows[[length(threshold_rows) + 1L]] <- data.frame(
      Rule = rule, Threshold = threshold,
      Sensitivity = unname(metrics["Sensitivity"]), Specificity = unname(metrics["Specificity"]),
      PPV = unname(metrics["PPV"]), NPV = unname(metrics["NPV"]), Accuracy = unname(metrics["Accuracy"]),
      F1 = unname(metrics["F1"]), Balanced_Accuracy = unname(metrics["Balanced_Accuracy"]), MCC = unname(metrics["MCC"]),
      stringsAsFactors = FALSE, check.names = FALSE
    )
  }
}
threshold_table <- do.call(rbind, threshold_rows)

group_summary <- function(group_name, group) {
  rows <- list()
  for (level in unique(group)) for (detector in detectors) {
    ii <- which(group == level); cells <- confusion(dat$Truth[ii], dat[[detector]][ii], dat$Weight[ii]); metrics <- metric_from_cells(cells["TP"], cells["TN"], cells["FP"], cells["FN"])
    rows[[length(rows) + 1L]] <- data.frame(Group = group_name, Level = as.character(level), Detector = detector,
      Sample_N = length(ii), Weighted_N = sum(dat$Weight[ii]), TP = cells["TP"], TN = cells["TN"], FP = cells["FP"], FN = cells["FN"],
      Sensitivity = unname(metrics["Sensitivity"]), Specificity = unname(metrics["Specificity"]),
      PPV = unname(metrics["PPV"]), NPV = unname(metrics["NPV"]), Accuracy = unname(metrics["Accuracy"]),
      F1 = unname(metrics["F1"]), Balanced_Accuracy = unname(metrics["Balanced_Accuracy"]), MCC = unname(metrics["MCC"]),
      stringsAsFactors = FALSE, check.names = FALSE)
  }
  do.call(rbind, rows)
}
subgroup_blocks <- list(
  group_summary("Sampling_Stratum", dat$Sampling_Stratum),
  group_summary("Ontology_Hit", ifelse(dat$Ontology == 1, "Yes", "No")),
  group_summary("Trap_Hit", ifelse(dat$Trap == 1, "Yes", "No")),
  group_summary("Text_Length", cut(dat$Full_Text_Characters, breaks = c(-Inf, 500, 1500, 5000, Inf), labels = c("<=500", "501-1500", "1501-5000", ">5000")))
)
if (!all(is.na(dat$Training_Text_Overlap))) {
  subgroup_blocks[[length(subgroup_blocks) + 1L]] <- group_summary("Training_Text_Overlap", ifelse(dat$Training_Text_Overlap, "Yes", "No"))
}
subgroup_table <- do.call(rbind, subgroup_blocks)

error_rows <- list()
for (detector in detectors) {
  pred <- dat[[detector]]; ii <- which(!is.na(dat$Truth) & pred != dat$Truth)
  if (length(ii)) {
    error_fields <- c("Row_ID", "Trial_ID", "Sampling_Stratum", "Ontology_Hit", "Ontology_Families", "Trap_Hit", "Trap_Families", "BERT_Probability", "Existing_v2_Keyword_Hit", "Existing_v2_Union_Label", "Reviewer_Confidence_High_Medium_Low", "Notes")
    if (!is.null(v8_predictions)) error_fields <- c(error_fields, "V8_Probability")
    if (!all(is.na(dat$Training_Text_Overlap))) error_fields <- c(error_fields, "Training_Text_Overlap", "Matched_Training_Rows")
    block <- dat[ii, error_fields]
    block$Detector <- detector; block$Truth <- dat$Truth[ii]; block$Prediction <- pred[ii]
    block$Error_Type <- ifelse(block$Truth == 1, "False negative", "False positive")
    error_rows[[length(error_rows) + 1L]] <- block
  }
}
error_table <- if (length(error_rows)) do.call(rbind, error_rows) else data.frame()

dir.create(output_dir, recursive = FALSE)
write.csv(confusion_table, file.path(output_dir, "confusion_matrices.csv"), row.names = FALSE, na = "")
write.csv(metric_table, file.path(output_dir, "performance_metrics.csv"), row.names = FALSE, na = "")
write.csv(threshold_table, file.path(output_dir, "threshold_analysis.csv"), row.names = FALSE, na = "")
write.csv(subgroup_table, file.path(output_dir, "subgroup_performance.csv"), row.names = FALSE, na = "")
write.csv(error_table, file.path(output_dir, "error_audit_internal.csv"), row.names = FALSE, na = "")
write.csv(dat, file.path(output_dir, "controlled_unblinded_internal.csv"), row.names = FALSE, na = "")

summary_object <- list(
  status = "COMPLETE",
  frozen_labels_sha256 = sha256_file(frozen_path),
  sealed_key_sha256 = sha256_file(key_path),
  records = nrow(dat),
  adjudicated = sum(valid_truth),
  ambiguous = sum(!valid_truth),
  weighted_frame_total = sum(dat$Weight),
  weighted_positive_prevalence = as.list(weighted_prevalence),
  weighted_ambiguous_rate = as.list(weighted_ambiguous),
  weighted_brier = as.list(weighted_brier),
  weighted_auc = as.list(weighted_auc_value),
  bert_weighted_brier = unname(weighted_brier["BERT"]),
  bert_weighted_auc = unname(weighted_auc_value["BERT"]),
  v8_predictions_supplied = !is.null(v8_predictions),
  v8_predictions_sha256 = if (is.null(v8_predictions)) NA_character_ else sha256_file(v8_predictions_path),
  v8_quantization_manifest_sha256 = if (is.null(v8_manifest)) NA_character_ else unname(v8_manifest$quantization_manifest_sha256),
  union_high_confidence_false_negatives = sum(dat$Truth == 1 & dat$Union == 0 & confidence == "High", na.rm = TRUE),
  high_confidence_false_negatives = as.list(setNames(vapply(detectors, function(detector) {
    sum(dat$Truth == 1 & dat[[detector]] == 0 & confidence == "High", na.rm = TRUE)
  }, integer(1)), detectors)),
  training_text_overlap_flags_supplied = !all(is.na(dat$Training_Text_Overlap)),
  training_text_overlap_records = if (all(is.na(dat$Training_Text_Overlap))) NA_integer_ else sum(dat$Training_Text_Overlap),
  training_text_overlap_rate = if (all(is.na(dat$Training_Text_Overlap))) NA_real_ else mean(dat$Training_Text_Overlap),
  bootstrap_replicates = bootstrap_replicates,
  bootstrap_seed = seed
)
jsonlite::write_json(summary_object, file.path(output_dir, "validation_summary.json"), pretty = TRUE, auto_unbox = TRUE, na = "null")

weighted_metrics <- metric_table[metric_table$Analysis == "Design-weighted remaining frame", ]
strict_metrics <- metric_table[metric_table$Analysis == "Design-weighted strict text-disjoint subpopulation", ]
fmt <- function(x, digits = 3) ifelse(is.na(x), "NA", formatC(x, format = "f", digits = digits))
report <- c(
  "# CNS Challenge-Set Validation Report",
  "",
  paste0("Frozen human-label SHA-256: `", sha256_file(frozen_path), "`"),
  paste0("Sealed-key SHA-256: `", sha256_file(key_path), "`"),
  paste0("Adjudicated records: ", sum(valid_truth), "/300; ambiguous: ", sum(!valid_truth), "."),
  paste0("Design weights reconstruct ", fmt(sum(dat$Weight), 0), " strictly held-out frame records."),
  "",
  "## Primary design-weighted estimates",
  "",
  "| Detector | Metric | Estimate | 95% CI |",
  "|---|---|---:|---:|"
)
for (i in seq_len(nrow(weighted_metrics))) {
  row <- weighted_metrics[i, ]
  report <- c(report, paste0("| ", row$Detector, " | ", row$Metric, " | ", fmt(row$Estimate), " | ", fmt(row$Lower_95), "–", fmt(row$Upper_95), " |"))
}
if (nrow(strict_metrics)) {
  report <- c(report, "", "## Strict text-disjoint sensitivity analysis", "",
    paste0(sum(dat$Training_Text_Overlap), " of 300 challenge texts matched normalized outcome text present under other v7 training IDs. The estimates below describe the text-disjoint subpopulation and do not reconstruct those excluded frame units."),
    "", "| Detector | Metric | Estimate | 95% CI |", "|---|---|---:|---:|")
  for (i in seq_len(nrow(strict_metrics))) {
    row <- strict_metrics[i, ]
    report <- c(report, paste0("| ", row$Detector, " | ", row$Metric, " | ", fmt(row$Estimate), " | ", fmt(row$Lower_95), "–", fmt(row$Upper_95), " |"))
  }
}
probability_report <- paste(vapply(probability_detectors, function(detector) {
  paste0(detector, " weighted AUC: ", fmt(weighted_auc_value[detector]), "; Brier score: ", fmt(weighted_brier[detector]))
}, character(1)), collapse = ". ")
report <- c(report, "", paste0(probability_report, "."),
  "", "The enriched challenge-set confusion matrices are descriptive only. Remaining-frame claims must use the design-weighted estimates above.")
writeLines(report, file.path(output_dir, "VALIDATION_REPORT.md"), useBytes = TRUE)
cat("PASS: controlled unblinding and validation completed in ", output_dir, "\n", sep = "")
