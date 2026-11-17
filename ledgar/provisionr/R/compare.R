#' Paired comparison of two classifiers on the same examples
#'
#' Both classifiers must have been scored on exactly the same rows. The
#' comparison uses that pairing twice: a paired bootstrap for the difference
#' in each headline metric (resampling rows, scoring both models on the same
#' resample), and an exact McNemar test on the discordant rows for accuracy.
#'
#' @param truth Factor of true labels.
#' @param estimate_a,estimate_b Factors of predictions with the levels of
#'   `truth`.
#' @param n_boot Number of bootstrap resamples.
#' @param conf Confidence level.
#' @param seed Optional seed.
#' @return A list with
#'   * `metrics`: tibble with one row per metric -- the two estimates, the
#'     difference `b - a`, its percentile interval and the share of
#'     resamples in which `b` beats `a`;
#'   * `agreement`: tibble counting rows by which model is correct;
#'   * `mcnemar`: list with the discordant counts and the exact two-sided
#'     p-value.
#' @examples
#' truth <- factor(c("x", "x", "y", "y", "y", "x"))
#' a <- factor(c("x", "y", "y", "y", "x", "x"), levels = levels(truth))
#' b <- factor(c("x", "x", "y", "y", "y", "y"), levels = levels(truth))
#' compare_predictions(truth, a, b, n_boot = 200, seed = 1)$agreement
#' @export
compare_predictions <- function(truth, estimate_a, estimate_b, n_boot = 1000L, conf = 0.95, seed = NULL) {
  check_same_levels(truth, estimate_a)
  check_same_levels(truth, estimate_b)
  if (!is.null(seed)) {
    set.seed(seed)
  }
  n <- length(truth)
  point_a <- classification_metrics(truth, estimate_a)
  point_b <- classification_metrics(truth, estimate_b)
  diffs <- purrr::map(seq_len(n_boot), function(i) {
    idx <- sample.int(n, replace = TRUE)
    classification_metrics(truth[idx], estimate_b[idx]) - classification_metrics(truth[idx], estimate_a[idx])
  }) |>
    dplyr::bind_rows()
  alpha <- (1 - conf) / 2
  metrics <- tibble::tibble(
    metric = names(point_a),
    a = unlist(point_a, use.names = FALSE),
    b = unlist(point_b, use.names = FALSE),
    difference = .data$b - .data$a,
    lower = purrr::map_dbl(.data$metric, \(m) stats::quantile(diffs[[m]], alpha)),
    upper = purrr::map_dbl(.data$metric, \(m) stats::quantile(diffs[[m]], 1 - alpha)),
    share_b_better = purrr::map_dbl(.data$metric, \(m) mean(diffs[[m]] > 0))
  )

  a_ok <- estimate_a == truth
  b_ok <- estimate_b == truth
  agreement <- tibble::tibble(
    outcome = c("both correct", "only a correct", "only b correct", "both wrong"),
    n = c(sum(a_ok & b_ok), sum(a_ok & !b_ok), sum(!a_ok & b_ok), sum(!a_ok & !b_ok))
  ) |>
    dplyr::mutate(share = .data$n / sum(.data$n))
  only_a <- agreement$n[2]
  only_b <- agreement$n[3]
  p <- if (only_a + only_b == 0) 1 else stats::binom.test(only_b, only_a + only_b, 0.5)$p.value
  list(metrics = metrics, agreement = agreement,
       mcnemar = list(only_a = only_a, only_b = only_b, p_value = p))
}

#' Temperature-scaled soft-max probabilities
#'
#' Turns arbitrary per-class scores (SVM decision values, network logits)
#' into probabilities `softmax(scores / temperature)`.
#' [fit_temperature()] chooses the temperature that minimises the negative
#' log-likelihood on held-out data, which calibrates the probabilities
#' without changing any prediction.
#'
#' @param scores n x K numeric matrix; `-Inf` marks impossible classes.
#' @param temperature Positive scalar.
#' @return n x K matrix of probabilities with the dimnames of `scores`.
#' @examples
#' s <- rbind(c(2, 1, 0), c(0, 0, 5))
#' round(softmax_scores(s), 3)
#' round(softmax_scores(s, temperature = 10), 3)
#' @export
softmax_scores <- function(scores, temperature = 1) {
  stopifnot(is.matrix(scores), temperature > 0)
  z <- scores / temperature
  z <- z - apply(z, 1, max)
  p <- exp(z)
  p / rowSums(p)
}

#' @rdname softmax_scores
#' @param truth Factor of true labels; its levels must match
#'   `colnames(scores)`.
#' @param interval Search interval for the temperature.
#' @return `fit_temperature()` returns the optimal temperature, with the
#'   attained mean negative log-likelihood as attribute `"nll"`.
#' @export
fit_temperature <- function(scores, truth, interval = c(0.01, 100)) {
  stopifnot(is.factor(truth), nrow(scores) == length(truth))
  col <- match(as.character(truth), colnames(scores) %||% levels(truth))
  nll <- function(log_t) {
    p <- softmax_scores(scores, exp(log_t))[cbind(seq_along(truth), col)]
    -mean(log(pmax(p, 1e-12)))
  }
  opt <- stats::optimize(nll, log(interval))
  structure(exp(opt$minimum), nll = opt$objective)
}
