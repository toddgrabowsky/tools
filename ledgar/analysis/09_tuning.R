# 09 -- Iteration 2: model families and hyper-parameters.
#
# All candidates share the feature set chosen in 08_ablation.R and the same
# five folds, so features are built once per fold and every comparison is
# paired. The grid covers the regularisation strength of each family, class
# re-weighting, and the multi-class formulation of the SVM.
#
# Output: results/09_*.csv, data/processed/09_tuning_predictions.rds,
#         figures/09_*.png

source("analysis/_setup.R")
col <- provision_colours()

train <- readRDS(file.path(paths$processed, "ledgar_clean.rds")) |> filter(split == "train")
folds <- readRDS(file.path(paths$processed, "cv_folds.rds"))

grid <- bind_rows(
  tibble(family = "Multinomial NB", engine = "nb", alpha = c(0.01, 0.03, 0.1, 0.3, 1)),
  tibble(family = "Complement NB", engine = "nb", complement = TRUE, alpha = c(0.01, 0.03, 0.1, 0.3, 1)),
  tibble(family = "Nearest centroid", engine = "centroid"),
  tibble(family = "Linear SVM (one-vs-rest)", engine = "svm", cost = c(0.1, 0.3, 1, 3)),
  tibble(family = "Linear SVM, class-balanced", engine = "svm", balanced = TRUE, cost = c(0.3, 1)),
  tibble(family = "Linear SVM (Crammer-Singer)", engine = "svm", multiclass = "cs", cost = c(0.1, 0.3, 1, 3, 10)),
  tibble(family = "Logistic regression", engine = "logreg", cost = c(10, 100))
) |>
  mutate(param = coalesce(cost, alpha),
         model = if_else(is.na(param), family, paste0(family, " [", if_else(is.na(cost), "alpha", "C"), "=", param, "]")))

# one argument list per row, dropping the NA (not applicable) arguments
models <- grid |>
  select(engine, alpha, complement, cost, balanced, multiclass) |>
  pmap(\(...) discard(list(...), is.na)) |>
  set_names(grid$model)

folds_file <- file.path(paths$results, "09_tuning_folds.csv")
if (!file.exists(folds_file)) {
  cv <- cv_classifier(train$text, train$label, folds, spec = chosen_spec(), models = models, workers = 5)
  write_result(cv$metrics, "09_tuning_folds")
  saveRDS(cv$predictions, file.path(paths$processed, "09_tuning_predictions.rds"))
}
tuning <- read_csv(folds_file, show_col_types = FALSE) |>
  left_join(select(grid, model, family, param), by = "model")

tuned <- summarise_cv(tuning, by = c("family", "model", "param")) |> arrange(desc(accuracy_mean))
write_result(tuned, "09_tuning")
print(select(tuned, model, accuracy_mean, accuracy_se, macro_f1_mean, macro_f1_se, fit_seconds), n = 30)

# -- Figure: regularisation paths ----------------------------------------------
p_path <- tuned |>
  filter(!is.na(param)) |>
  select(family, param, `Accuracy (micro-F1)_mean` = accuracy_mean, `Accuracy (micro-F1)_se` = accuracy_se,
         `Macro-F1_mean` = macro_f1_mean, `Macro-F1_se` = macro_f1_se) |>
  pivot_longer(c(-family, -param), names_to = c("metric", ".value"), names_sep = "_") |>
  mutate(kind = if_else(str_detect(family, "NB"), "Naive Bayes: smoothing alpha", "Linear models: cost C")) |>
  ggplot(aes(param, mean, colour = family)) +
  geom_line(linewidth = 0.7) +
  geom_pointrange(aes(ymin = mean - 2 * se, ymax = mean + 2 * se), size = 0.3, linewidth = 0.6) +
  facet_grid(metric ~ kind, scales = "free") +
  scale_x_log10() +
  scale_y_continuous(labels = scales::label_percent(accuracy = 1)) +
  scale_colour_provision(name = NULL) +
  guides(colour = guide_legend(nrow = 2)) +
  labs(title = "Regularisation matters more than the choice among linear models",
       subtitle = "5-fold CV mean +/- 2 SE on the cleaned training split, as a function of each family's main hyper-parameter.",
       x = "Hyper-parameter value (log scale)", y = NULL) +
  theme_provision(grid = "xy")
save_figure(p_path, file.path(paths$figures, "09_regularisation_paths.png"), width = 9.5, height = 6.5)

# -- Figure: best configuration per family -------------------------------------
best_per_family <- tuned |> slice_max(accuracy_mean, n = 1, by = family, with_ties = FALSE) |>
  arrange(accuracy_mean)
write_result(best_per_family, "09_best_per_family")
p_best <- plot_cv_summary(best_per_family, by = "model", highlight = last(best_per_family$model)) +
  labs(title = "Best configuration of each model family",
       subtitle = "5-fold stratified CV on the cleaned training split; mean +/- 2 SE over folds.")
save_figure(p_best, file.path(paths$figures, "09_best_per_family.png"), width = 10, height = 4.2)

# -- Is the winner's lead real? Paired fold-wise differences to the runner-ups --
winner <- tuned$model[1]
paired <- tuning |>
  select(model, fold, accuracy, macro_f1) |>
  left_join(tuning |> filter(model == winner) |> select(fold, win_acc = accuracy, win_f1 = macro_f1), by = "fold") |>
  filter(model != winner) |>
  summarise(d_accuracy = mean(win_acc - accuracy), d_accuracy_se = sd(win_acc - accuracy) / sqrt(n()),
            d_macro_f1 = mean(win_f1 - macro_f1), d_macro_f1_se = sd(win_f1 - macro_f1) / sqrt(n()),
            folds_won = sum(win_acc > accuracy),
            p_paired_t = t.test(win_acc, accuracy, paired = TRUE)$p.value, .by = model) |>
  arrange(d_accuracy)
write_result(paired, "09_winner_vs_rest")
cli::cli_alert_info("Winner: {winner}")
print(head(paired, 8), width = Inf)
