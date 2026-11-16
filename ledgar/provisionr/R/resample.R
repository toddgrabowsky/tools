#' Assign cross-validation folds, stratified and optionally grouped
#'
#' Every class is spread as evenly as possible over the folds, however
#' rare it is. (General-purpose helpers typically pool small strata, which
#' with 100 classes would silently switch stratification off.) With
#' `groups`, all members of a group land in the same fold -- use this to
#' keep near-duplicate texts from straddling the train/assessment boundary.
#'
#' @param y Factor (or vector) of class labels.
#' @param v Number of folds.
#' @param groups Optional vector of group ids, same length as `y`.
#' @param seed Optional seed.
#' @return Integer vector of fold ids in `1..v`, same length as `y`.
#' @examples
#' y <- factor(rep(c("a", "b", "c"), c(10, 5, 2)))
#' table(y, assign_folds(y, v = 5, seed = 1))
#' @export
assign_folds <- function(y, v = 5L, groups = NULL, seed = NULL) {
  if (!is.null(seed)) {
    set.seed(seed)
  }
  n <- length(y)
  if (is.null(groups)) {
    groups <- seq_len(n)
  }
  stopifnot(length(groups) == n, v >= 2)
  # one row per group, labelled with the group's most common class
  unit <- tibble::tibble(group = groups, y = as.character(y)) |>
    dplyr::count(.data$group, .data$y) |>
    dplyr::slice_max(.data$n, n = 1, by = "group", with_ties = FALSE) |>
    dplyr::mutate(u = stats::runif(dplyr::n())) |>
    dplyr::arrange(.data$y, .data$u) |>
    # deal units out like cards within each class, starting at a random fold
    dplyr::mutate(fold = (dplyr::row_number() + sample.int(v, 1L)) %% v + 1L, .by = "y")
  unit$fold[match(groups, unit$group)]
}

#' Cross-validate one or more classifiers on raw text
#'
#' For every fold the featurizer is learned on the analysis rows only, then
#' each model is fitted and scored on the assessment rows. Several models
#' can share one pass so the (comparatively expensive) feature building is
#' done once per fold.
#'
#' @param text Character vector of texts.
#' @param y Factor of labels.
#' @param folds Integer fold ids, e.g. from [assign_folds()].
#' @param spec A [feature_spec()].
#' @param models Named list; each element is a list of arguments for
#'   [fit_classifier()], e.g. `list(svm = list(engine = "svm", cost = 0.5))`.
#' @param prepare Optional function applied to the analysis rows of each
#'   fold before fitting -- `function(text, y)` returning a list with `text`
#'   and `y`. Use it for training-only steps such as de-duplication.
#' @param workers Number of forked worker processes (folds run in parallel).
#' @return A list with `metrics` (one row per model and fold, see
#'   [classification_metrics()], plus fit and predict time in seconds) and
#'   `predictions` (one row per model and example: `row`, `fold`, `truth`,
#'   `estimate`, `margin` = best score minus runner-up).
#' @export
cv_classifier <- function(text, y, folds, spec = feature_spec(),
                          models = list(svm = list(engine = "svm")),
                          prepare = NULL, workers = 1L) {
  stopifnot(is.character(text), is.factor(y), length(text) == length(y),
            length(folds) == length(y), rlang::is_named(models))
  run_fold <- function(k) {
    test <- which(folds == k)
    analysis <- list(text = text[folds != k], y = y[folds != k])
    if (!is.null(prepare)) {
      analysis <- prepare(analysis$text, analysis$y)
    }
    featurizer <- fit_featurizer(analysis$text, spec)
    x_train <- bake_features(featurizer, analysis$text)
    x_test <- bake_features(featurizer, text[test])
    purrr::imap(models, function(args, name) {
      t_fit <- system.time(fit <- do.call(fit_classifier, c(list(x = x_train, y = analysis$y), args)))
      t_pred <- system.time(scores <- predict(fit, x_test, type = "score"))
      estimate <- factor(colnames(scores)[max.col(scores, ties.method = "first")], levels = levels(y))
      list(
        metrics = classification_metrics(y[test], estimate) |>
          dplyr::mutate(model = name, fold = k, n_features = ncol(x_train),
                        fit_seconds = t_fit[["elapsed"]], predict_seconds = t_pred[["elapsed"]],
                        .before = 1),
        predictions = tibble::tibble(model = name, row = test, fold = k, truth = y[test],
                                     estimate = estimate, margin = score_margin(scores))
      )
    })
  }
  ks <- sort(unique(folds))
  res <- if (workers > 1L) {
    parallel::mclapply(ks, run_fold, mc.cores = workers, mc.preschedule = FALSE)
  } else {
    lapply(ks, run_fold)
  }
  failed <- purrr::map_lgl(res, \(r) inherits(r, "try-error") || is.null(r))
  if (any(failed)) {
    msg <- as.character(res[failed][[1]])
    cli::cli_abort(c("{sum(failed)} fold{?s} failed.", x = "{msg}"))
  }
  res <- purrr::list_flatten(res)
  list(metrics = purrr::map(res, "metrics") |> dplyr::bind_rows(),
       predictions = purrr::map(res, "predictions") |> dplyr::bind_rows())
}

score_margin <- function(scores) {
  best <- max.col(scores, ties.method = "first")
  top <- scores[cbind(seq_len(nrow(scores)), best)]
  scores[cbind(seq_len(nrow(scores)), best)] <- -Inf
  top - apply(scores, 1, max)
}

#' Summarise cross-validation metrics across folds
#'
#' @param metrics The `metrics` element returned by [cv_classifier()].
#' @param by Grouping columns.
#' @return A tibble with the mean and the standard error over folds of each
#'   metric, per group.
#' @export
summarise_cv <- function(metrics, by = "model") {
  metrics |>
    dplyr::summarise(
      folds = dplyr::n(),
      dplyr::across(c("accuracy", "macro_f1", "weighted_f1", "balanced_accuracy"),
                    list(mean = mean, se = \(x) stats::sd(x) / sqrt(length(x))),
                    .names = "{.col}_{.fn}"),
      fit_seconds = mean(.data$fit_seconds),
      .by = dplyr::all_of(by)
    )
}

#' Target shuffling (permutation test) for a cross-validated classifier
#'
#' Repeats the whole cross-validation with the labels randomly permuted.
#' Shuffling breaks any real link between text and label while keeping the
#' class balance, the features and the modelling procedure intact, so the
#' resulting scores show what the pipeline achieves by chance alone
#' (including any optimism from the procedure itself). The observed score
#' is then located in that null distribution.
#'
#' Features are built once per fold and re-used across permutations: the
#' featurizer never sees labels, so this is exact, not an approximation.
#'
#' @inheritParams cv_classifier
#' @param model Argument list for [fit_classifier()].
#' @param n_perm Number of label permutations.
#' @param seed Optional seed.
#' @return A list with `observed` (one-row tibble of metrics), `null` (one
#'   row per permutation) and `p_value` (named vector; the permutation
#'   p-value `(1 + #{null >= observed}) / (1 + n_perm)` per metric).
#' @export
target_shuffle <- function(text, y, folds, spec = feature_spec(),
                           model = list(engine = "svm"), n_perm = 100L,
                           seed = NULL, workers = 1L) {
  stopifnot(is.character(text), is.factor(y), length(folds) == length(y))
  if (!is.null(seed)) {
    set.seed(seed)
  }
  ks <- sort(unique(folds))
  feats <- lapply(ks, function(k) {
    fz <- fit_featurizer(text[folds != k], spec)
    list(train = bake_features(fz, text[folds != k]), test = bake_features(fz, text[folds == k]))
  })
  cv_once <- function(labels) {
    estimate <- labels
    for (i in seq_along(ks)) {
      k <- ks[i]
      fit <- do.call(fit_classifier, c(list(x = feats[[i]]$train, y = labels[folds != k]), model))
      estimate[folds == k] <- predict(fit, feats[[i]]$test)
    }
    classification_metrics(labels, estimate)
  }
  observed <- cv_once(y)
  perms <- lapply(seq_len(n_perm), function(b) sample(y))
  null <- if (workers > 1L) {
    parallel::mclapply(perms, cv_once, mc.cores = workers)
  } else {
    lapply(perms, cv_once)
  }
  null <- dplyr::bind_rows(null, .id = "permutation")
  metrics <- names(observed)
  p_value <- purrr::map_dbl(setNames(metrics, metrics),
                            \(m) (1 + sum(null[[m]] >= observed[[m]])) / (1 + n_perm))
  list(observed = observed, null = null, p_value = p_value)
}
