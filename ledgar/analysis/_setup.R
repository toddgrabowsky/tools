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
