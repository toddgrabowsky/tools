# 07 -- Baselines under 5-fold stratified cross-validation (training split).
#
# Establishes the floor (majority class), cheap generative / geometric
# baselines (multinomial NB, complement NB, nearest centroid) and the two
# linear discriminative models at default settings, all on the same plain
# word 1-2-gram TF-IDF features.
#
# Output: results/07_baseline_cv_folds.csv, results/07_baseline_cv.csv,
#         data/processed/07_baseline_predictions.rds, figures/07_*.png

source("analysis/_setup.R")
col <- provision_colours()

train <- readRDS(file.path(paths$processed, "ledgar_clean.rds")) |> filter(split == "train")
folds <- assign_folds(train$label, v = 5, seed = SEED)
saveRDS(folds, file.path(paths$processed, "cv_folds.rds"))

models <- list(
  "Majority class" = list(engine = "majority"),
  "Multinomial NB" = list(engine = "nb", alpha = 0.1),
  "Complement NB" = list(engine = "nb", alpha = 0.1, complement = TRUE),
  "Nearest centroid" = list(engine = "centroid"),
  "Logistic regression (C=1)" = list(engine = "logreg", cost = 1),
  "Linear SVM (C=1)" = list(engine = "svm", cost = 1)
)

cv <- cv_classifier(train$text, train$label, folds, spec = feature_spec(word_ngram = c(1, 2)),
                    models = models, workers = 4)
write_result(cv$metrics, "07_baseline_cv_folds")
saveRDS(cv$predictions, file.path(paths$processed, "07_baseline_predictions.rds"))

baseline <- summarise_cv(cv$metrics) |> arrange(accuracy_mean)
write_result(baseline, "07_baseline_cv")
print(select(baseline, model, accuracy_mean, accuracy_se, macro_f1_mean, macro_f1_se, fit_seconds))

# -- Figure: fold-level scores ------------------------------------------------
p <- cv$metrics |>
  select(model, fold, `Accuracy (micro-F1)` = accuracy, `Macro-F1` = macro_f1) |>
  pivot_longer(c(-model, -fold), names_to = "metric") |>
  mutate(model = factor(model, levels = baseline$model)) |>
  ggplot(aes(value, model)) +
  geom_point(colour = col$categorical[["blue"]], alpha = 0.55, size = 1.8) +
  stat_summary(fun = mean, geom = "point", shape = 124, size = 6, colour = col$ink) +
  stat_summary(aes(label = after_stat(sprintf("%.3f", x))), fun = mean, geom = "text",
               vjust = -1.1, size = 3, colour = col$ink2) +
  facet_wrap(vars(metric)) +
  scale_x_continuous(labels = scales::percent, limits = c(0, 1)) +
  labs(title = "Baselines: linear discriminative models lead; rare classes hurt the others",
       subtitle = "5-fold stratified CV on the cleaned training split, word 1-2-gram TF-IDF. Dots = folds, bar = mean.",
       x = NULL, y = NULL) +
  theme_provision(grid = "x")
save_figure(p, file.path(paths$figures, "07_baselines.png"), width = 9.5, height = 4.2)
