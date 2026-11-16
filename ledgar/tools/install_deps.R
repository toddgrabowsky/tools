# Install the R packages the study needs (CRAN, from source if necessary).
pkgs <- c(
  # provisionr imports
  "cli", "dplyr", "ggplot2", "ggrepel", "jsonlite", "LiblineaR", "Matrix", "nanoparquet",
  "purrr", "rlang", "scales", "SparseM", "stringr", "text2vec", "tibble", "tidyr",
  # analysis scripts and report
  "tidyverse", "tidytext", "glue", "irlba", "uwot", "ggridges", "patchwork", "xgboost",
  "knitr", "rmarkdown",
  # development
  "devtools", "roxygen2", "testthat"
)
missing <- setdiff(pkgs, rownames(installed.packages()))
if (length(missing) > 0) {
  install.packages(missing, repos = "https://cloud.r-project.org", Ncpus = max(1L, parallel::detectCores() - 1L))
}
invisible(lapply(pkgs, requireNamespace))
cat("All", length(pkgs), "packages available.\n")
