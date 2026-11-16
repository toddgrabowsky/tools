#' Locations of the LEDGAR splits on the Hugging Face hub
#'
#' LEDGAR (Tuggener et al., 2020) as redistributed in LexGLUE
#' (Chalkidis et al., 2022): 80,000 contract provisions from SEC EDGAR
#' filings, each labelled with one of 100 provision types.
#'
#' @return A named character vector of URLs, one per split.
#' @export
ledgar_urls <- function() {
  base <- "https://huggingface.co/api/datasets/coastalcph/lex_glue/parquet/ledgar"
  splits <- c("train", "validation", "test")
  setNames(sprintf("%s/%s/0.parquet", base, splits), splits)
}

#' Download the LEDGAR parquet files
#'
#' Files that already exist are left alone, so the function is safe to call
#' at the top of every script.
#'
#' @param dir Directory to download into (created if needed).
#' @param overwrite Re-download files that already exist?
#' @return `dir`, invisibly.
#' @export
ledgar_download <- function(dir, overwrite = FALSE) {
  dir.create(dir, recursive = TRUE, showWarnings = FALSE)
  urls <- ledgar_urls()
  purrr::iwalk(urls, function(url, split) {
    dest <- file.path(dir, paste0(split, ".parquet"))
    if (file.exists(dest) && !overwrite) {
      return(invisible())
    }
    cli::cli_alert_info("Downloading {.val {split}} split")
    utils::download.file(url, dest, mode = "wb", quiet = TRUE)
  })
  invisible(dir)
}

#' Label names stored in a Hugging Face parquet file
#'
#' Hugging Face writes the `ClassLabel` names into the parquet key-value
#' metadata; reading them from there avoids hard-coding the label map.
#'
#' @param path Path to a parquet file written by the `datasets` library.
#' @param column Name of the class-label column.
#' @return Character vector of label names, in id order (id 0 first).
#' @export
parquet_label_names <- function(path, column = "label") {
  kv <- nanoparquet::read_parquet_metadata(path)$file_meta_data$key_value_metadata[[1]]
  info <- kv$value[kv$key == "huggingface"]
  if (length(info) != 1) {
    cli::cli_abort("No Hugging Face metadata found in {.path {path}}.")
  }
  jsonlite::fromJSON(info)$info$features[[column]]$names
}

#' Read LEDGAR into a tidy tibble
#'
#' @param dir Directory holding `train.parquet`, `validation.parquet` and
#'   `test.parquet` (see [ledgar_download()]).
#' @param splits Which splits to read.
#' @return A tibble with one row per provision and columns `id` (unique
#'   within the corpus), `split` (factor), `text`, `label_id` (0-based id
#'   used by LexGLUE) and `label` (factor whose levels follow `label_id`).
#' @export
ledgar_read <- function(dir, splits = c("train", "validation", "test")) {
  splits <- rlang::arg_match(splits, multiple = TRUE)
  paths <- setNames(file.path(dir, paste0(splits, ".parquet")), splits)
  missing <- paths[!file.exists(paths)]
  if (length(missing) > 0) {
    cli::cli_abort(c("Missing split file{?s}: {.path {missing}}.",
                     i = "Run {.fn ledgar_download} first."))
  }
  label_names <- parquet_label_names(paths[[1]])
  purrr::map(paths, nanoparquet::read_parquet) |>
    dplyr::bind_rows(.id = "split") |>
    dplyr::transmute(
      split = factor(.data$split, levels = c("train", "validation", "test")),
      text = .data$text,
      label_id = as.integer(.data$label),
      label = factor(label_names[.data$label_id + 1L], levels = label_names)
    ) |>
    dplyr::mutate(id = sprintf("%s-%05d", .data$split, dplyr::row_number()),
                  .by = "split", .before = 1) |>
    tibble::as_tibble()
}
