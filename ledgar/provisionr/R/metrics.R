#' Per-class precision, recall and F1
#'
#' @param truth,estimate Factors with the same levels.
#' @return A tibble with one row per class that occurs in `truth` or
#'   `estimate`: `label`, `support` (true count), `predicted` (predicted
#'   count), `precision`, `recall`, `f1`. Undefined ratios are 0, matching
#'   the convention of the LexGLUE benchmark scripts.
#' @examples
#' truth <- factor(c("a", "a", "b", "c"), levels = c("a", "b", "c"))
#' est <- factor(c("a", "b", "b", "b"), levels = c("a", "b", "c"))
#' per_class_metrics(truth, est)
#' @export
per_class_metrics <- function(truth, estimate) {
  check_same_levels(truth, estimate)
  k <- nlevels(truth)
  tp <- tabulate(as.integer(truth)[truth == estimate], k)
  support <- tabulate(as.integer(truth), k)
  predicted <- tabulate(as.integer(estimate), k)
  safe <- function(num, den) ifelse(den > 0, num / den, 0)
  tibble::tibble(
    label = factor(levels(truth), levels = levels(truth)),
    support = support, predicted = predicted,
    precision = safe(tp, predicted), recall = safe(tp, support),
    f1 = safe(2 * tp, support + predicted)
  ) |>
    dplyr::filter(.data$support > 0 | .data$predicted > 0)
}

#' Headline classification metrics
#'
#' For single-label problems micro-averaged F1 equals accuracy, so the two
#' LexGLUE headline numbers are `accuracy` (micro-F1) and `macro_f1`.
#'
#' @inheritParams per_class_metrics
#' @return A one-row tibble: `accuracy`, `macro_f1`, `weighted_f1`,
#'   `balanced_accuracy` (mean recall over classes present in `truth`).
#' @examples
#' truth <- factor(c("a", "a", "b", "c"), levels = c("a", "b", "c"))
#' est <- factor(c("a", "b", "b", "b"), levels = c("a", "b", "c"))
#' classification_metrics(truth, est)
#' @export
classification_metrics <- function(truth, estimate) {
  pc <- per_class_metrics(truth, estimate)
  tibble::tibble(
    accuracy = mean(truth == estimate),
    macro_f1 = mean(pc$f1),
    weighted_f1 = stats::weighted.mean(pc$f1, pc$support),
    balanced_accuracy = mean(pc$recall[pc$support > 0])
  )
}

#' Top-k accuracy from a score matrix
#'
#' @param truth Factor of true labels.
#' @param scores n x K score matrix with class names as column names
#'   (see [predict.provision_classifier()]).
#' @param k Integer vector of cut-offs.
#' @return A tibble with columns `k` and `accuracy`.
#' @export
top_k_accuracy <- function(truth, scores, k = c(1L, 3L, 5L)) {
  stopifnot(nrow(scores) == length(truth))
  true_score <- scores[cbind(seq_along(truth), match(as.character(truth), colnames(scores)))]
  rank_true <- rowSums(scores > true_score) + 1L
  tibble::tibble(k = as.integer(k), accuracy = purrr::map_dbl(k, \(kk) mean(rank_true <= kk)))
}

#' Most frequent confusions
#'
#' @inheritParams per_class_metrics
#' @return A tibble of off-diagonal confusion-matrix cells (`truth`,
#'   `estimate`, `n`, `share_of_truth`), most frequent first.
#' @export
confusion_pairs <- function(truth, estimate) {
  check_same_levels(truth, estimate)
  tibble::tibble(truth = truth, estimate = estimate) |>
    dplyr::mutate(support = dplyr::n(), .by = "truth") |>
    dplyr::filter(.data$truth != .data$estimate) |>
    dplyr::count(.data$truth, .data$estimate, .data$support, sort = TRUE) |>
    dplyr::mutate(share_of_truth = .data$n / .data$support) |>
    dplyr::select(-"support")
}

#' Bootstrap confidence intervals for classification metrics
#'
#' Non-parametric percentile bootstrap over examples. It answers "how much
#' would this score move with a different test sample of the same size",
#' not "with a different training set".
#'
#' @inheritParams per_class_metrics
#' @param n_boot Number of bootstrap resamples.
#' @param conf Confidence level.
#' @param seed Optional seed for reproducibility.
#' @return A tibble with one row per metric: `metric`, `estimate`, `lower`,
#'   `upper`, `std_error`. The bootstrap replicates are attached as
#'   attribute `"replicates"`.
#' @export
bootstrap_metrics <- function(truth, estimate, n_boot = 1000L, conf = 0.95, seed = NULL) {
  check_same_levels(truth, estimate)
  if (!is.null(seed)) {
    set.seed(seed)
  }
  n <- length(truth)
  reps <- purrr::map(seq_len(n_boot), function(b) {
    i <- sample.int(n, replace = TRUE)
    classification_metrics(truth[i], estimate[i])
  }) |>
    dplyr::bind_rows(.id = "replicate")
  point <- classification_metrics(truth, estimate)
  alpha <- (1 - conf) / 2
  out <- reps |>
    tidyr::pivot_longer(-"replicate", names_to = "metric") |>
    dplyr::summarise(lower = stats::quantile(.data$value, alpha),
                     upper = stats::quantile(.data$value, 1 - alpha),
                     std_error = stats::sd(.data$value), .by = "metric") |>
    dplyr::mutate(estimate = unname(unlist(point[.data$metric])), .after = "metric")
  attr(out, "replicates") <- reps
  out
}

check_same_levels <- function(truth, estimate) {
  if (!is.factor(truth) || !is.factor(estimate) || !identical(levels(truth), levels(estimate))) {
    cli::cli_abort("{.arg truth} and {.arg estimate} must be factors with identical levels.")
  }
  if (length(truth) != length(estimate)) {
    cli::cli_abort("{.arg truth} and {.arg estimate} must have the same length.")
  }
  invisible(TRUE)
}
