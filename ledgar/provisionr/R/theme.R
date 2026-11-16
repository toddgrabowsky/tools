#' Colour constants used by every provisionr plot
#'
#' A small, colour-blind-checked palette: eight categorical hues (assigned
#' in order, never recycled), a single-hue sequential ramp, a blue/red
#' diverging pair with a neutral midpoint, and ink / grid colours.
#'
#' @return A named list of hex colours.
#' @export
provision_colours <- function() {
  list(
    categorical = c(blue = "#2a78d6", orange = "#eb6834", aqua = "#1baf7a",
                    yellow = "#eda100", magenta = "#e87ba4", green = "#008300",
                    violet = "#6250d6", red = "#e34948"),
    sequential = c("#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf",
                   "#184f95", "#0d366b"),
    diverging = c(low = "#d03b3b", mid = "#f0efec", high = "#2a78d6"),
    ink = "#0b0b0b", ink2 = "#52514e", muted = "#898781",
    grid = "#e1e0d9", axis = "#c3c2b7", surface = "#fcfcfb"
  )
}

#' ggplot2 theme for provisionr figures
#'
#' Recessive grid and axes, left-aligned titles, legend on top.
#'
#' @param base_size Base font size in points.
#' @param grid Which major grid lines to keep: `"y"`, `"x"`, `"xy"` or `"none"`.
#' @return A ggplot2 theme object.
#' @export
theme_provision <- function(base_size = 11, grid = c("y", "x", "xy", "none")) {
  grid <- rlang::arg_match(grid)
  col <- provision_colours()
  line <- ggplot2::element_line(colour = col$grid, linewidth = 0.3)
  ggplot2::theme_minimal(base_size = base_size) +
    ggplot2::theme(
      plot.background = ggplot2::element_rect(fill = col$surface, colour = NA),
      panel.grid.minor = ggplot2::element_blank(),
      panel.grid.major.x = if (grid %in% c("x", "xy")) line else ggplot2::element_blank(),
      panel.grid.major.y = if (grid %in% c("y", "xy")) line else ggplot2::element_blank(),
      axis.line = ggplot2::element_line(colour = col$axis, linewidth = 0.3),
      axis.ticks = ggplot2::element_blank(),
      axis.text = ggplot2::element_text(colour = col$ink2),
      axis.title = ggplot2::element_text(colour = col$ink2, size = ggplot2::rel(0.9)),
      plot.title = ggplot2::element_text(colour = col$ink, face = "bold", size = ggplot2::rel(1.15)),
      plot.subtitle = ggplot2::element_text(colour = col$ink2, margin = ggplot2::margin(b = 8)),
      plot.caption = ggplot2::element_text(colour = col$muted, hjust = 0),
      plot.title.position = "plot",
      plot.caption.position = "plot",
      strip.text = ggplot2::element_text(colour = col$ink, face = "bold", hjust = 0),
      legend.position = "top",
      legend.justification = "left",
      legend.title = ggplot2::element_text(colour = col$ink2, size = ggplot2::rel(0.9)),
      legend.text = ggplot2::element_text(colour = col$ink2),
      plot.margin = ggplot2::margin(12, 14, 10, 12)
    )
}

#' Categorical colour / fill scales in the fixed provisionr order
#'
#' @param ... Passed to [ggplot2::scale_colour_manual()] /
#'   [ggplot2::scale_fill_manual()].
#' @return A ggplot2 scale.
#' @export
scale_colour_provision <- function(...) {
  ggplot2::scale_colour_manual(values = unname(provision_colours()$categorical), ...)
}

#' @rdname scale_colour_provision
#' @export
scale_fill_provision <- function(...) {
  ggplot2::scale_fill_manual(values = unname(provision_colours()$categorical), ...)
}

#' Save a figure with consistent size and resolution
#'
#' @param plot A ggplot.
#' @param path Output file (PNG).
#' @param width,height Size in inches.
#' @param dpi Resolution.
#' @return `path`, invisibly.
#' @export
save_figure <- function(plot, path, width = 8, height = 5, dpi = 150) {
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  ggplot2::ggsave(path, plot, width = width, height = height, dpi = dpi,
                  bg = provision_colours()$surface)
  cli::cli_alert_success("Saved {.path {path}}")
  invisible(path)
}
