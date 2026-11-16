# 02 -- EDA: label distribution and class imbalance.
#
# Questions: how skewed are the 100 classes, and do the official splits
# share one label distribution? (Only label marginals of validation/test
# are inspected here -- never their text.)
#
# Output: figures/02_*.png, results/02_label_counts.csv

source("analysis/_setup.R")
col <- provision_colours()

ledgar <- ledgar_read(paths$raw)
train <- filter(ledgar, split == "train")

label_counts <- ledgar |>
  count(split, label) |>
  mutate(share = n / sum(n), .by = split) |>
  pivot_wider(names_from = split, values_from = c(n, share), values_fill = 0) |>
  arrange(desc(n_train)) |>
  mutate(rank = row_number())
write_result(label_counts, "02_label_counts")

imbalance <- label_counts |>
  summarise(
    classes = n(),
    largest = max(n_train), smallest = min(n_train),
    ratio = largest / smallest,
    median = median(n_train),
    classes_under_100 = sum(n_train < 100),
    top10_share = sum(share_train[rank <= 10]),
    # effective number of classes = exp(Shannon entropy)
    effective_classes = exp(-sum(share_train * log(share_train))),
    majority_baseline = max(share_train)
  )
write_result(imbalance, "02_imbalance_summary")
print(imbalance, width = Inf)

# -- Figure: ranked class frequencies ---------------------------------------
p_counts <- label_counts |>
  mutate(label = fct_reorder(label, n_train),
         half = if_else(rank <= 50, "Ranks 1-50", "Ranks 51-100")) |>
  ggplot(aes(n_train, label)) +
  geom_segment(aes(x = 0, xend = n_train, yend = label), colour = col$categorical[["blue"]],
               linewidth = 0.6) +
  geom_point(colour = col$categorical[["blue"]], size = 1.6) +
  geom_text(aes(label = scales::comma(n_train)), hjust = -0.35, size = 2.6, colour = col$ink2) +
  facet_wrap(vars(half), scales = "free_y") +
  scale_x_continuous(labels = scales::comma, expand = expansion(mult = c(0, 0.14))) +
  labs(
    title = "LEDGAR classes are heavily imbalanced",
    subtitle = glue::glue(
      "Training provisions per class. Largest/smallest = {round(imbalance$ratio)}x; ",
      "{imbalance$classes_under_100} classes have fewer than 100 examples."),
    x = "Training provisions", y = NULL
  ) +
  theme_provision(base_size = 10, grid = "x")
save_figure(p_counts, file.path(paths$figures, "02_label_counts.png"), width = 10, height = 9)

# -- Figure: concentration curve --------------------------------------------
p_conc <- label_counts |>
  mutate(cum_share = cumsum(share_train)) |>
  ggplot(aes(rank, cum_share)) +
  geom_abline(slope = 1 / 100, intercept = 0, colour = col$axis, linetype = "dashed") +
  geom_line(colour = col$categorical[["blue"]], linewidth = 0.8) +
  annotate("text", x = 62, y = 0.56, label = "perfectly balanced", colour = col$muted,
           size = 3, angle = 27) +
  geom_point(data = \(d) filter(d, rank %in% c(10, 25, 50)), colour = col$categorical[["blue"]], size = 2.4) +
  geom_text(data = \(d) filter(d, rank %in% c(10, 25, 50)),
            aes(label = glue::glue("top {rank}: {scales::percent(cum_share, 1)}")),
            hjust = -0.12, vjust = 1.4, size = 3.2, colour = col$ink2) +
  scale_y_continuous(labels = scales::percent, limits = c(0, 1), expand = expansion(0)) +
  scale_x_continuous(expand = expansion(mult = c(0, 0.02))) +
  labs(
    title = "A quarter of the classes cover over half of the provisions",
    subtitle = glue::glue("Cumulative share of training data by class rank. ",
                          "Effective number of classes: {round(imbalance$effective_classes)} of 100."),
    x = "Classes, most to least frequent", y = "Cumulative share of provisions"
  ) +
  theme_provision(grid = "xy")
save_figure(p_conc, file.path(paths$figures, "02_label_concentration.png"), width = 7, height = 4.5)

# -- Figure: are the splits drawn from the same label distribution? ---------
split_cmp <- label_counts |>
  select(label, share_train, validation = share_validation, test = share_test) |>
  pivot_longer(c(validation, test), names_to = "split", values_to = "share") |>
  mutate(split = fct_inorder(split),
         ratio = share / share_train)

# Chi-squared goodness of fit of each held-out split against train shares.
gof <- map_dfr(c("validation", "test"), \(s) {
  tst <- suppressWarnings(chisq.test(label_counts[[paste0("n_", s)]], p = label_counts$share_train))
  tibble(split = s, chi_sq = unname(tst$statistic), df = unname(tst$parameter), p_value = tst$p.value)
})
write_result(gof, "02_split_label_gof")
print(gof)

# LexGLUE splits LEDGAR chronologically (train = oldest filings, test = newest),
# so some drift in the label mix is expected -- and it shows up in test.
p_split <- split_cmp |>
  filter(share > 0) |>  # 'Books' has no validation examples; undefined on a log axis
  ggplot(aes(share_train, ratio)) +
  geom_hline(yintercept = 1, colour = col$axis) +
  geom_point(colour = col$categorical[["blue"]], alpha = 0.7, size = 1.8) +
  ggrepel::geom_text_repel(
    data = \(d) filter(d, abs(log(ratio)) > log(1.45)), aes(label = label),
    size = 2.8, colour = col$ink2, max.overlaps = 20, seed = 1) +
  facet_wrap(vars(split), labeller = as_labeller(\(x) paste(str_to_title(x), "vs train"))) +
  scale_x_log10(labels = scales::label_percent(accuracy = 0.1)) +
  scale_y_continuous(trans = "log2", breaks = c(0.5, 0.67, 1, 1.5, 2),
                     labels = \(x) paste0(x, "x")) +
  labs(
    title = "The test split's label mix has drifted away from train",
    subtitle = "Class share in the held-out split relative to its share in train (1x = identical).",
    x = "Share in train (log scale)", y = "Held-out share / train share",
    caption = glue::glue("Chi-squared goodness of fit vs train shares: ",
                         "validation p = {signif(gof$p_value[1], 2)}, test p = {signif(gof$p_value[2], 2)}. ",
                         "LexGLUE's LEDGAR split is chronological.")
  ) +
  theme_provision(grid = "xy")
save_figure(p_split, file.path(paths$figures, "02_label_split_consistency.png"), width = 8, height = 4.5)
