# 12 -- Held-out test evaluation.
#
# The test split is used here for the first time, once, with the single
# model selected and frozen in 10_select_refit.R.
#
# Output: results/12_*.csv, figures/12_*.png

source("analysis/_setup.R")
col <- provision_colours()
blue <- col$categorical[["blue"]]

clean <- readRDS(file.path(paths$processed, "ledgar_clean.rds"))
test <- filter(clean, split == "test")
final <- readRDS(file.path(paths$models, "final_model.rds"))

# The final model was fitted on train + validation, so "seen before" must be
# judged against both. (LEDGAR itself contains test provisions that equal a
# training provision up to whitespace, typography or numbers.)
fit_texts <- clean$text[clean$split != "test"]
test <- test |>
  mutate(template_in_train = text_fingerprint(text) %in% text_fingerprint(fit_texts),
         exact_in_train = text %in% fit_texts)
x_test <- bake_features(final$featurizer, test$text)
scores <- predict(final$classifier, x_test, type = "score")
test <- test |>
  mutate(estimate = factor(colnames(scores)[max.col(scores, ties.method = "first")], levels = levels(label)),
         correct = estimate == label,
         margin = provisionr:::score_margin(scores))
saveRDS(select(test, id, label, estimate, correct, margin, template_in_train, n_words),
        file.path(paths$processed, "12_test_predictions.rds"))

# -- 1. Headline metrics with bootstrap confidence intervals ----------------------
boot <- bootstrap_metrics(test$label, test$estimate, n_boot = 2000, seed = SEED)
write_result(boot, "12_test_metrics")
print(boot)

# Published LexGLUE test results for context (Chalkidis et al., 2022, Table 3).
reference <- tribble(
  ~model, ~accuracy, ~macro_f1,
  "TF-IDF + SVM (LexGLUE)", 0.872, 0.824,
  "BERT (LexGLUE)", 0.876, 0.818,
  "Legal-BERT (LexGLUE)", 0.882, 0.830,
  "RoBERTa-large (LexGLUE)", 0.886, 0.836
)
# The published models were trained on the training split only, whereas the
# final model here also saw the validation split. For a like-for-like row,
# score the train-only fit of the same pipeline (saved by 10_select_refit.R)
# as well. No decision depends on this second number.
train_only <- readRDS(file.path(paths$models, "train_only_model.rds"))
train_only_metrics <- classification_metrics(
  test$label, predict(train_only$classifier, bake_features(train_only$featurizer, test$text)))
this_work <- "This work: TF-IDF + linear SVM (train + validation)"
comparison <- bind_rows(
  reference |> mutate(source = "published"),
  tibble(model = this_work,
         accuracy = boot$estimate[boot$metric == "accuracy"],
         macro_f1 = boot$estimate[boot$metric == "macro_f1"], source = "this work"),
  tibble(model = "This work: same pipeline, train split only",
         accuracy = train_only_metrics$accuracy, macro_f1 = train_only_metrics$macro_f1, source = "this work")
)
write_result(comparison, "12_benchmark_comparison")

ci <- boot |> filter(metric %in% c("accuracy", "macro_f1")) |>
  mutate(metric = recode(metric, accuracy = "Accuracy (micro-F1)", macro_f1 = "Macro-F1"))
p_bench <- comparison |>
  pivot_longer(c(accuracy, macro_f1), names_to = "metric") |>
  mutate(metric = recode(metric, accuracy = "Accuracy (micro-F1)", macro_f1 = "Macro-F1"),
         model = fct_reorder(model, value, .fun = min)) |>
  ggplot(aes(value, model)) +
  geom_linerange(data = ci |> mutate(model = this_work),
                 aes(x = estimate, xmin = lower, xmax = upper), colour = col$categorical[["orange"]], linewidth = 1) +
  geom_point(aes(colour = source), size = 2.6) +
  geom_text(aes(label = sprintf("%.1f", 100 * value)), vjust = -1, size = 3, colour = col$ink2) +
  facet_wrap(vars(metric), scales = "free_x") +
  scale_colour_manual(values = c(published = blue, `this work` = col$categorical[["orange"]]), name = NULL) +
  scale_x_continuous(labels = scales::label_percent(accuracy = 1), expand = expansion(mult = 0.15)) +
  labs(title = "Test performance in context",
       subtitle = "LEDGAR test split (10,000 provisions). Orange bar: 95% bootstrap interval (2,000 resamples of the test set).",
       x = NULL, y = NULL,
       caption = paste("Published figures: Chalkidis et al. (2022), LexGLUE; trained on the train split, reported without intervals.",
                       "Differences of a few tenths of a point are within the bootstrap interval.", sep = "\n")) +
  theme_provision(grid = "x")
save_figure(p_bench, file.path(paths$figures, "12_benchmark_comparison.png"), width = 10.5, height = 4.6)

reps <- attr(boot, "replicates")
p_boot <- reps |>
  select(accuracy, macro_f1) |>
  pivot_longer(everything(), names_to = "metric") |>
  mutate(metric = recode(metric, accuracy = "Accuracy (micro-F1)", macro_f1 = "Macro-F1")) |>
  ggplot(aes(value)) +
  geom_histogram(bins = 40, fill = blue, colour = col$surface, linewidth = 0.2) +
  geom_vline(data = ci, aes(xintercept = lower), linetype = "dashed", colour = col$ink2) +
  geom_vline(data = ci, aes(xintercept = upper), linetype = "dashed", colour = col$ink2) +
  geom_text(data = ci, aes(x = estimate, y = Inf, label = sprintf("%.3f [%.3f, %.3f]", estimate, lower, upper)),
            vjust = 1.4, size = 3.2, colour = col$ink) +
  facet_wrap(vars(metric), scales = "free_x") +
  scale_x_continuous(labels = scales::label_percent(accuracy = 0.1)) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.12))) +
  labs(title = "Sampling uncertainty of the test scores",
       subtitle = "Bootstrap distribution over 2,000 resamples of the test set; dashed lines = 95% percentile interval.",
       x = NULL, y = "Bootstrap resamples")
save_figure(p_boot, file.path(paths$figures, "12_bootstrap_distribution.png"), width = 9, height = 4)

# -- 2. Top-k accuracy --------------------------------------------------------------
topk <- top_k_accuracy(test$label, scores, k = c(1, 2, 3, 5, 10))
write_result(topk, "12_top_k_accuracy")
print(topk)

# -- 3. Per-class performance ---------------------------------------------------------
per_class <- per_class_metrics(test$label, test$estimate) |> arrange(f1)
write_result(per_class, "12_per_class")
p_class <- plot_per_class(per_class, label_below = 0.62) +
  labs(title = "Weak classes are the near-synonymous ones, not simply the rare ones",
       subtitle = glue::glue("Per-class F1 on the test split against class size. ",
                             "{sum(per_class$f1 >= 0.9)} of {nrow(per_class)} classes reach F1 >= 0.90; ",
                             "{sum(per_class$f1 < 0.6)} fall below 0.60."))
save_figure(p_class, file.path(paths$figures, "12_per_class_f1.png"), width = 9, height = 5.5)

p_f1_dist <- per_class |>
  mutate(label = fct_reorder(label, f1)) |>
  ggplot(aes(f1, label)) +
  geom_segment(aes(x = 0, xend = f1, yend = label), colour = blue, linewidth = 0.5) +
  geom_point(colour = blue, size = 1.3) +
  facet_wrap(vars(if_else(rank(-f1, ties.method = "first") <= 50, "Best 50 classes", "Worst 50 classes")),
             scales = "free_y") +
  scale_x_continuous(labels = scales::percent, limits = c(0, 1), expand = expansion(mult = c(0, 0.03))) +
  labs(title = "Per-class F1 on the test split", x = "F1", y = NULL) +
  theme_provision(base_size = 9, grid = "x")
save_figure(p_f1_dist, file.path(paths$figures, "12_per_class_f1_ranked.png"), width = 10, height = 8.5)

# -- 4. Confusions ----------------------------------------------------------------------
pairs <- confusion_pairs(test$label, test$estimate)
write_result(pairs, "12_confusion_pairs")
print(head(pairs, 12))
p_conf <- plot_confusion(test$label, test$estimate, n_classes = 30) +
  labs(title = "The largest confusions are between near-synonymous labels",
       subtitle = glue::glue("Row-normalised confusion matrix for the 30 classes involved in the most test errors.\n",
                             "The 20 most confused pairs account for ",
                             "{scales::percent(sum(head(pairs$n, 20)) / sum(pairs$n), 1)} of all errors; the rest is a long tail."))
save_figure(p_conf, file.path(paths$figures, "12_confusion_heatmap.png"), width = 9.5, height = 10)

# -- 5. Does the model know when it is unsure? Selective prediction -----------------------
risk_coverage <- test |>
  arrange(desc(margin)) |>
  mutate(coverage = row_number() / n(), accuracy = cummean(correct)) |>
  filter(row_number() %% 50 == 0)
coverage_points <- map_dfr(c(0.5, 0.7, 0.8, 0.9, 1), \(cv) slice_min(risk_coverage, abs(coverage - cv), n = 1))
write_result(select(coverage_points, coverage, accuracy, margin), "12_selective_prediction")

margin_bins <- test |>
  mutate(bin = ntile(margin, 10)) |>
  summarise(margin = median(margin), accuracy = mean(correct), n = n(), .by = bin) |>
  mutate(se = sqrt(accuracy * (1 - accuracy) / n))
write_result(margin_bins, "12_margin_deciles")

library(patchwork)
p_margin <- ggplot(margin_bins, aes(factor(bin), accuracy)) +
  geom_col(fill = blue, width = 0.65) +
  geom_errorbar(aes(ymin = accuracy - 2 * se, ymax = accuracy + 2 * se), width = 0.15, colour = col$ink2) +
  geom_text(aes(label = scales::percent(accuracy, 1)), vjust = -1.4, size = 2.9, colour = col$ink2) +
  scale_y_continuous(labels = scales::percent, limits = c(0, 1.08), breaks = seq(0, 1, 0.25)) +
  labs(subtitle = "Accuracy by decile of the SVM margin (best minus runner-up score)",
       x = "Margin decile (1 = least confident)", y = "Test accuracy")
p_cov <- ggplot(risk_coverage, aes(coverage, accuracy)) +
  geom_line(colour = blue, linewidth = 0.8) +
  geom_point(data = coverage_points, colour = blue, size = 2) +
  geom_text(data = coverage_points, aes(label = scales::percent(accuracy, 0.1)), vjust = 1.9, size = 2.9, colour = col$ink2) +
  scale_x_continuous(labels = scales::percent) +
  scale_y_continuous(labels = scales::label_percent(accuracy = 1), expand = expansion(mult = c(0.12, 0.05))) +
  labs(subtitle = "Accuracy when only the most confident share is auto-labelled",
       x = "Coverage (share of provisions auto-labelled)", y = "Accuracy on the covered share") +
  theme_provision(grid = "xy")
save_figure(p_margin + p_cov + plot_annotation(
              title = "The margin is a usable confidence score",
              theme = theme_provision()),
            file.path(paths$figures, "12_confidence.png"), width = 11, height = 4.5)

# -- 6. Slices: template seen in train, and provision length ---------------------------------
slices <- bind_rows(
  test |> mutate(slice = case_when(exact_in_train ~ "identical text in train/validation",
                                   template_in_train ~ "same template (numbers/case differ)",
                                   TRUE ~ "novel text")) |>
    summarise(group = "Boilerplate", n = n(), accuracy = mean(correct), .by = slice) |> arrange(desc(n)),
  test |> mutate(slice = cut(n_words, c(0, 30, 60, 120, 240, Inf),
                             labels = c("<= 30 words", "31-60", "61-120", "121-240", "> 240"))) |>
    summarise(group = "Length", n = n(), accuracy = mean(correct), .by = slice) |> arrange(slice)
) |> mutate(se = sqrt(accuracy * (1 - accuracy) / n))
write_result(slices, "12_slices")
print(slices)

p_slices <- slices |>
  mutate(slice = fct_inorder(as.character(slice))) |>
  ggplot(aes(accuracy, fct_rev(slice))) +
  geom_linerange(aes(xmin = accuracy - 2 * se, xmax = accuracy + 2 * se), colour = blue, linewidth = 0.7) +
  geom_point(colour = blue, size = 2.4) +
  geom_text(aes(label = glue::glue("{scales::percent(accuracy, 0.1)}  (n = {scales::comma(n)})")),
            vjust = -1, size = 2.9, colour = col$ink2) +
  facet_wrap(vars(group), scales = "free_y") +
  scale_x_continuous(labels = scales::label_percent(accuracy = 1), expand = expansion(mult = 0.12)) +
  labs(title = "Where the test accuracy comes from",
       subtitle = "Test accuracy by slice, +/- 2 binomial SE.", x = "Test accuracy", y = NULL) +
  theme_provision(grid = "x")
save_figure(p_slices, file.path(paths$figures, "12_slices.png"), width = 9.5, height = 4)
