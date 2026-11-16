#' Plot a target-shuffling null distribution against the observed score
#'
#' @param x Result of [target_shuffle()].
#' @param metric Which metric to show.
#' @param bins Number of histogram bins for the null distribution.
#' @return A ggplot.
#' @export
plot_target_shuffle <- function(x, metric = "accuracy", bins = 30) {
  col <- provision_colours()
  obs <- x$observed[[metric]]
  null <- x$null[[metric]]
  n_perm <- length(null)
  ggplot2::ggplot(tibble::tibble(value = null), ggplot2::aes(.data$value)) +
    ggplot2::geom_histogram(bins = bins, fill = col$muted, colour = col$surface, linewidth = 0.2) +
    ggplot2::geom_vline(xintercept = obs, colour = col$categorical[["blue"]], linewidth = 0.9) +
    ggplot2::annotate("text", x = obs, y = Inf, vjust = 1.6, hjust = 1.08, size = 3.3,
                      colour = col$categorical[["blue"]],
                      label = sprintf("observed %.1f%%", 100 * obs)) +
    ggplot2::annotate("text", x = max(null), y = Inf, vjust = 3.4, hjust = 0, size = 3.1,
                      colour = col$ink2,
                      label = sprintf("  shuffled labels:\n  mean %.1f%%, max %.1f%%", 100 * mean(null), 100 * max(null))) +
    ggplot2::scale_x_continuous(labels = scales::label_percent(accuracy = 1)) +
    ggplot2::scale_y_continuous(expand = ggplot2::expansion(mult = c(0, 0.08))) +
    ggplot2::labs(
      title = "Target shuffling: the model's score is far outside what chance produces",
      subtitle = sprintf("Cross-validated %s for the real labels vs %d label permutations (p %s %.3g).",
                         gsub("_", " ", metric), n_perm,
                         if (x$p_value[[metric]] <= 1 / (n_perm + 1)) "<=" else "=", x$p_value[[metric]]),
      x = sprintf("Cross-validated %s", gsub("_", " ", metric)), y = "Permutations"
    ) +
    theme_provision()
}

#' Row-normalised confusion heat map
#'
#' With many classes a full confusion matrix is mostly empty, so by default
#' only the classes involved in the most frequent confusions are shown.
#'
#' @inheritParams per_class_metrics
#' @param n_classes Show the `n_classes` classes with the most errors
#'   (as truth or as prediction); `Inf` shows all.
#' @return A ggplot.
#' @export
plot_confusion <- function(truth, estimate, n_classes = 30) {
  check_same_levels(truth, estimate)
  col <- provision_colours()
  wrong <- truth != estimate
  errs <- tabulate(as.integer(truth)[wrong], nlevels(truth)) +
    tabulate(as.integer(estimate)[wrong], nlevels(truth))
  keep <- levels(truth)[utils::head(order(errs, decreasing = TRUE), min(n_classes, nlevels(truth)))]
  cm <- tibble::tibble(truth = truth, estimate = estimate) |>
    dplyr::count(.data$truth, .data$estimate) |>
    dplyr::mutate(share = .data$n / sum(.data$n), .by = "truth") |>
    dplyr::filter(.data$truth %in% keep, .data$estimate %in% keep)
  # order classes so that confused classes sit next to each other
  wide <- stats::xtabs(share ~ truth + estimate, data = droplevels(cm))
  lv <- union(rownames(wide), colnames(wide))
  sq <- matrix(0, length(lv), length(lv), dimnames = list(lv, lv))
  sq[rownames(wide), colnames(wide)] <- wide
  sym <- sq + t(sq)
  diag(sym) <- 0
  ord <- lv[stats::hclust(stats::as.dist(max(sym) - sym), method = "average")$order]
  cm |>
    dplyr::mutate(truth = factor(.data$truth, levels = rev(ord)),
                  estimate = factor(.data$estimate, levels = ord)) |>
    ggplot2::ggplot(ggplot2::aes(.data$estimate, .data$truth, fill = .data$share)) +
    ggplot2::geom_tile(colour = col$surface, linewidth = 0.3) +
    ggplot2::scale_fill_gradientn(colours = c(col$sequential[1], col$sequential), limits = c(0, 1),
                                  labels = scales::percent, name = "Share of true class") +
    ggplot2::coord_equal() +
    ggplot2::labs(x = "Predicted class", y = "True class") +
    theme_provision(base_size = 9, grid = "none") +
    ggplot2::theme(axis.text.x = ggplot2::element_text(angle = 90, hjust = 1, vjust = 0.5),
                   axis.line = ggplot2::element_blank(),
                   legend.position = "bottom",
                   legend.title = ggplot2::element_text(vjust = 0.9),
                   legend.key.width = ggplot2::unit(1.4, "cm"),
                   legend.key.height = ggplot2::unit(0.25, "cm"))
}

#' Per-class F1 against class size
#'
#' @param per_class Result of [per_class_metrics()].
#' @param label_below Label the classes whose F1 is below this value.
#' @return A ggplot.
#' @export
plot_per_class <- function(per_class, label_below = 0.6) {
  col <- provision_colours()
  ggplot2::ggplot(per_class, ggplot2::aes(.data$support, .data$f1)) +
    ggplot2::geom_point(colour = col$categorical[["blue"]], alpha = 0.75, size = 2) +
    ggrepel::geom_text_repel(
      data = dplyr::filter(per_class, .data$f1 < label_below),
      ggplot2::aes(label = .data$label), size = 2.9, colour = col$ink2,
      max.overlaps = 30, seed = 1, min.segment.length = 0.2
    ) +
    ggplot2::scale_x_log10(labels = scales::comma) +
    ggplot2::scale_y_continuous(limits = c(0, 1), labels = scales::percent) +
    ggplot2::labs(x = "Examples of the class in the evaluation set (log scale)", y = "F1") +
    theme_provision(grid = "xy")
}

#' Dot-and-interval plot of cross-validated scores
#'
#' @param summary Result of [summarise_cv()] (or any table with
#'   `<metric>_mean` and `<metric>_se` columns).
#' @param by Column holding the configuration names.
#' @param metrics Metrics to show, one panel each.
#' @param highlight Optional configuration name to draw in the accent colour.
#' @return A ggplot. Intervals are mean +/- 2 standard errors over folds.
#' @export
plot_cv_summary <- function(summary, by = "model", metrics = c("accuracy", "macro_f1"),
                            highlight = NULL) {
  col <- provision_colours()
  long <- purrr::map(metrics, function(m) {
    tibble::tibble(config = summary[[by]], metric = m,
                   mean = summary[[paste0(m, "_mean")]], se = summary[[paste0(m, "_se")]])
  }) |>
    dplyr::bind_rows() |>
    dplyr::mutate(
      metric = factor(.data$metric, levels = metrics,
                      labels = dplyr::recode(metrics, accuracy = "Accuracy (micro-F1)",
                                             macro_f1 = "Macro-F1")),
      config = factor(.data$config, levels = unique(summary[[by]])),
      accent = .data$config %in% highlight
    )
  ggplot2::ggplot(long, ggplot2::aes(.data$mean, .data$config, colour = .data$accent)) +
    ggplot2::geom_linerange(ggplot2::aes(xmin = .data$mean - 2 * .data$se,
                                         xmax = .data$mean + 2 * .data$se), linewidth = 0.7) +
    ggplot2::geom_point(size = 2.3) +
    ggplot2::geom_text(ggplot2::aes(label = sprintf("%.3f", .data$mean)), vjust = -0.9, size = 2.9,
                       colour = col$ink2) +
    ggplot2::facet_wrap(ggplot2::vars(.data$metric), scales = "free_x") +
    ggplot2::scale_colour_manual(values = c(`FALSE` = col$categorical[["blue"]],
                                            `TRUE` = col$categorical[["orange"]]), guide = "none") +
    ggplot2::scale_x_continuous(labels = scales::label_percent(accuracy = 0.1)) +
    ggplot2::labs(x = NULL, y = NULL) +
    theme_provision(grid = "x")
}
