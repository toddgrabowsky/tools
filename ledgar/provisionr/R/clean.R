#' Normalise contract text
#'
#' A conservative, deterministic normaliser. It repairs encoding noise that
#' carries no meaning (typographic quotes and dashes, non-breaking spaces,
#' runs of whitespace or of underscores used as blanks) and optionally
#' lower-cases and masks digits. It never drops words, so it is safe to
#' apply before any tokeniser.
#'
#' @param x Character vector.
#' @param lowercase Convert to lower case?
#' @param mask_numbers Replace every run of digits (with internal `,` or `.`)
#'   by a single `0`? Useful because amounts, dates and section numbers are
#'   almost never informative about the provision type.
#' @return Character vector the same length as `x`.
#' @examples
#' normalize_text("  \u201cBase Salary\u201d  of $250,000.00\u00a0per year ____ ")
#' normalize_text("Section 12.3", mask_numbers = TRUE)
#' @export
normalize_text <- function(x, lowercase = FALSE, mask_numbers = FALSE) {
  x <- stringr::str_replace_all(x, c(
    "[\\u2018\\u2019\\u201a\\u201b`\\u00b4]" = "'",
    "[\\u201c\\u201d\\u201e\\u201f]" = "\"",
    "[\\u2010-\\u2015]" = "-",
    "[\\u00a0\\u2000-\\u200b\\u202f\\u3000]" = " ",
    "_{2,}" = " _ ",
    "\\s+" = " "
  ))
  x <- stringr::str_trim(x)
  if (lowercase) {
    x <- stringr::str_to_lower(x)
  }
  if (mask_numbers) {
    x <- stringr::str_replace_all(x, "\\d+(?:[.,]\\d+)*", "0")
  }
  x
}

#' Template fingerprint of a text
#'
#' Collapses a text to its "template": lower case, digits masked, and
#' everything except letters and the digit mask removed. Two provisions with
#' the same fingerprint differ only in numbers, punctuation, case or spacing
#' -- i.e. they are the same boilerplate.
#'
#' @param x Character vector.
#' @return Character vector of fingerprints.
#' @examples
#' text_fingerprint(c("Section 4.1  Governing Law.", "SECTION 12.7 governing law"))
#' @export
text_fingerprint <- function(x) {
  x <- normalize_text(x, lowercase = TRUE, mask_numbers = TRUE)
  x <- stringr::str_replace_all(x, "[^a-z0]+", " ")
  stringr::str_squish(x)
}

#' Flag duplicated texts and label conflicts
#'
#' Groups texts by [text_fingerprint()] (or by exact string) and reports,
#' for every row, the size of its duplicate group, whether it is a repeat of
#' an earlier row, and whether the group carries more than one label.
#'
#' @param text Character vector.
#' @param label Optional vector of labels, same length as `text`.
#' @param method `"template"` groups by fingerprint, `"exact"` by identical
#'   string.
#' @return A tibble with one row per element of `text`: `group` (integer id),
#'   `group_size`, `is_repeat` (not the first of its group) and, when `label`
#'   is given, `n_labels` and `label_conflict`.
#' @examples
#' flag_duplicates(c("Sec. 1 Notices.", "Sec. 2 Notices.", "Taxes"),
#'                 label = c("a", "b", "c"))
#' @export
flag_duplicates <- function(text, label = NULL, method = c("template", "exact")) {
  method <- rlang::arg_match(method)
  key <- if (method == "template") text_fingerprint(text) else text
  out <- tibble::tibble(group = match(key, unique(key))) |>
    dplyr::mutate(group_size = dplyr::n(),
                  is_repeat = dplyr::row_number() > 1L,
                  .by = "group")
  if (!is.null(label)) {
    stopifnot(length(label) == length(text))
    out <- out |>
      dplyr::mutate(label = .env$label) |>
      dplyr::mutate(n_labels = dplyr::n_distinct(.data$label), .by = "group") |>
      dplyr::mutate(label_conflict = .data$n_labels > 1L) |>
      dplyr::select(-"label")
  }
  out
}
