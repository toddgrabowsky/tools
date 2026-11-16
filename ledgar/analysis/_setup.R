# Shared preamble for every analysis script.
# Run scripts from the project root (the `ledgar/` directory):
#   Rscript analysis/02_eda_labels.R

suppressPackageStartupMessages({
  library(tidyverse)
  library(Matrix)
})

# Load the in-repo package: the installed copy if present (fast, used by
# parallel workers), otherwise straight from source.
if (requireNamespace("provisionr", quietly = TRUE)) {
  library(provisionr)
} else {
  devtools::load_all("provisionr", quiet = TRUE)
}

paths <- list(
  raw = "data/raw",
  processed = "data/processed",
  results = "results",
  figures = "figures",
  models = "models"
)
walk(paths, dir.create, recursive = TRUE, showWarnings = FALSE)

# Scripts must behave the same under any locale (Rscript defaults to C here).
invisible(Sys.setlocale("LC_CTYPE", "C.UTF-8"))

SEED <- 20261116L
set.seed(SEED)
theme_set(theme_provision())

#' Write a results table as CSV and echo where it went.
write_result <- function(x, name) {
  path <- file.path(paths$results, paste0(name, ".csv"))
  write_csv(x, path)
  cli::cli_alert_success("Wrote {.path {path}} ({nrow(x)} rows)")
  invisible(x)
}

# -- Decisions carried between scripts ----------------------------------------
# These are the *outputs* of 08_ablation.R and 09_tuning.R, recorded in one
# place so every later script uses the same pipeline. 10_select_refit.R
# asserts that chosen_model_name() really is the cross-validation winner.

#' Feature set selected by the ablation study (08).
chosen_spec <- function() {
  feature_spec(word_ngram = c(1, 2), head_ngram = c(1, 2), head_words = 20,
               weights = c(head = 0.5), min_doc_count = 2, sublinear_tf = TRUE, use_idf = TRUE,
               lowercase = TRUE, mask_numbers = TRUE)
}
