# 10 -- Model selection and refit.
#
# 1. Selection is made on cross-validation alone (09_tuning.R).
# 2. The validation split -- never used so far, and later in time than the
#    training split -- is then used as a first hold-out: do the CV rankings
#    and CV scores carry over to unseen, newer data?
# 3. One open cleaning question is settled on the same hold-out: does
#    de-duplicating the training data help?
# 4. The selected configuration is refitted on train + validation and saved.
#    The test split is not touched in this script.
#
# Output: results/10_*.csv, figures/10_*.png, models/final_model.rds

source("analysis/_setup.R")
col <- provision_colours()

clean <- readRDS(file.path(paths$processed, "ledgar_clean.rds"))
train <- filter(clean, split == "train")
valid <- filter(clean, split == "validation")

tuned <- read_csv(file.path(paths$results, "09_tuning.csv"), show_col_types = FALSE)
best_per_family <- read_csv(file.path(paths$results, "09_best_per_family.csv"), show_col_types = FALSE)

# -- 1. Selection rule -----------------------------------------------------------
winner <- slice_max(tuned, accuracy_mean, n = 1, with_ties = FALSE)
# configurations statistically indistinguishable from the winner (within 1 SE)
within_1se <- tuned |> filter(accuracy_mean >= winner$accuracy_mean - winner$accuracy_se)
cli::cli_alert_info("CV winner: {winner$model} (accuracy {round(winner$accuracy_mean, 4)})")
cli::cli_alert_info("Within one SE of the winner: {.val {within_1se$model}}")
stopifnot(identical(chosen_model_name(), winner$model))   # _setup.R must agree with the CV result

# -- 2. Hold-out check on validation ---------------------------------------------
model_args <- function(model_name) {
  # rebuild the fit_classifier() argument list from the label used in 09_tuning.R
  family <- str_remove(model_name, " \\[.*$")
  value <- parse_number(str_extract(model_name, "\\[.*\\]") %||% NA_character_)
  switch(family,
    "Multinomial NB" = list(engine = "nb", alpha = value),
    "Complement NB" = list(engine = "nb", alpha = value, complement = TRUE),
    "Nearest centroid" = list(engine = "centroid"),
    "Linear SVM (one-vs-rest)" = list(engine = "svm", cost = value),
    "Linear SVM, class-balanced" = list(engine = "svm", cost = value, balanced = TRUE),
    "Linear SVM (Crammer-Singer)" = list(engine = "svm", cost = value, multiclass = "cs"),
    "Logistic regression" = list(engine = "logreg", cost = value)
  )
}

fz <- fit_featurizer(train$text, chosen_spec())
x_train <- bake_features(fz, train$text)
x_valid <- bake_features(fz, valid$text)

holdout_file <- file.path(paths$processed, "10_holdout_fits.rds")
if (!file.exists(holdout_file)) {
  best_per_family |>
    select(family, model, cv_accuracy = accuracy_mean, cv_accuracy_se = accuracy_se,
           cv_macro_f1 = macro_f1_mean, cv_macro_f1_se = macro_f1_se) |>
    mutate(fit = map(model, \(m) do.call(fit_classifier, c(list(x = x_train, y = train$label), model_args(m)))),
           pred = map(fit, \(f) predict(f, x_valid)),
           metrics = map(pred, \(p) classification_metrics(valid$label, p))) |>
    unnest(metrics) |>
    rename(valid_accuracy = accuracy, valid_macro_f1 = macro_f1) |>
    saveRDS(holdout_file)
}
holdout <- readRDS(holdout_file)
write_result(select(holdout, -fit, -pred, -weighted_f1, -balanced_accuracy), "10_cv_vs_validation")
print(select(holdout, model, cv_accuracy, valid_accuracy, cv_macro_f1, valid_macro_f1))

rank_agreement <- cor(holdout$cv_accuracy, holdout$valid_accuracy, method = "spearman")
cv_valid_long <- holdout |>
  select(family, `Accuracy (micro-F1)_cv` = cv_accuracy, `Accuracy (micro-F1)_valid` = valid_accuracy,
         `Macro-F1_cv` = cv_macro_f1, `Macro-F1_valid` = valid_macro_f1) |>
  pivot_longer(-family, names_to = c("metric", ".value"), names_sep = "_") |>
  mutate(linear = str_detect(family, "SVM|Logistic"))
# the four tuned linear models sit on top of each other: label them once
linear_cluster <- cv_valid_long |> filter(linear) |>
  summarise(cv = mean(cv), valid = mean(valid), .by = metric) |>
  mutate(label = "3 linear SVM variants +\nlogistic regression")
p_cv_valid <- ggplot(cv_valid_long, aes(cv, valid)) +
  geom_abline(colour = col$axis) +
  geom_point(colour = col$categorical[["blue"]], size = 2.4, alpha = 0.8) +
  ggrepel::geom_text_repel(data = \(d) filter(d, !linear), aes(label = family), size = 2.9,
                           colour = col$ink2, seed = 1) +
  geom_text(data = linear_cluster, aes(label = label), size = 2.9, colour = col$ink2, hjust = 1.08,
            vjust = 0.3, lineheight = 0.9) +
  facet_wrap(vars(metric), scales = "free") +
  scale_x_continuous(labels = scales::label_percent(accuracy = 1)) +
  scale_y_continuous(labels = scales::label_percent(accuracy = 1)) +
  labs(title = "Cross-validation and the validation hold-out agree on the ranking of model families",
       subtitle = glue::glue("Best configuration per family: 5-fold CV estimate on train vs score on the untouched validation split ",
                             "(Spearman rho = {round(rank_agreement, 2)}).\nLine: equal scores. Hold-out models train on all of train, CV models on 4/5 of it."),
       x = "5-fold CV estimate (train)", y = "Validation hold-out") +
  theme_provision(grid = "xy")
save_figure(p_cv_valid, file.path(paths$figures, "10_cv_vs_validation.png"), width = 10, height = 4.8)

# -- 2b. A non-linear check: boosted trees on LSA components -------------------------
# Every family in the CV grid is linear in the n-gram space. As a sanity check
# that linearity is not what holds performance back, fit gradient-boosted
# trees on a 100-dimensional LSA projection (train -> validation only; too
# slow for the CV grid).
xgb_file <- file.path(paths$processed, "10_xgb_lsa.rds")
if (!file.exists(xgb_file)) {
  set.seed(SEED)
  svd <- irlba::irlba(x_train, nv = 100)
  lsa_train <- as.matrix(x_train %*% svd$v)
  lsa_valid <- as.matrix(x_valid %*% svd$v)
  secs <- system.time(
    xgb <- fit_classifier(lsa_train, train$label, engine = "xgb", nrounds = 100, eta = 0.3,
                          max_depth = 6, nthread = 4)
  )[["elapsed"]]
  # the same LSA features with the selected linear model, for a like-for-like comparison
  svm_lsa <- do.call(fit_classifier, c(list(x = lsa_train, y = train$label), chosen_model()))
  saveRDS(bind_rows(
    classification_metrics(valid$label, predict(xgb, lsa_valid)) |>
      mutate(model = "xgboost on 100-d LSA", fit_seconds = secs),
    classification_metrics(valid$label, predict(svm_lsa, lsa_valid)) |>
      mutate(model = "linear SVM on 100-d LSA", fit_seconds = NA)
  ), xgb_file)
}
nonlinear <- bind_rows(
  readRDS(xgb_file),
  filter(holdout, model == winner$model) |>
    transmute(accuracy = valid_accuracy, macro_f1 = valid_macro_f1, weighted_f1, balanced_accuracy,
              model = "linear SVM on sparse TF-IDF (selected)", fit_seconds = NA)
) |> relocate(model)
write_result(nonlinear, "10_nonlinear_check")
print(nonlinear)

# -- 3. Does de-duplicating the training data help? -------------------------------
train_raw <- ledgar_read(paths$raw, "train") |> mutate(text = normalize_text(text))
fit_and_score <- function(data) {
  f <- fit_featurizer(data$text, chosen_spec())
  m <- do.call(fit_classifier, c(list(x = bake_features(f, data$text), y = data$label), chosen_model()))
  classification_metrics(valid$label, predict(m, bake_features(f, valid$text)))
}
dedup <- bind_rows(
  `raw train (60,000 rows)` = fit_and_score(train_raw),
  `cleaned train (57,947 rows)` = filter(holdout, model == winner$model) |>
    select(accuracy = valid_accuracy, macro_f1 = valid_macro_f1, weighted_f1, balanced_accuracy),
  .id = "training_data"
)
write_result(dedup, "10_dedup_effect")
print(dedup)

# -- 4. Refit on train + validation -----------------------------------------------
# Clean the pooled data jointly so a validation row repeating a training
# template is dropped as well.
pooled <- bind_rows(train, valid) |> select(id, split, text, label) |> clean_corpus()
stopifnot(all(pooled$split %in% c("train", "validation")))   # no test row can enter the fit
cli::cli_alert_info("Refit data: {nrow(pooled)} rows after joint cleaning")
final_fz <- fit_featurizer(pooled$text, chosen_spec())
final_fit <- do.call(fit_classifier, c(list(x = bake_features(final_fz, pooled$text), y = pooled$label),
                                       chosen_model()))
final <- list(featurizer = final_fz, classifier = final_fit, spec = chosen_spec(), model = chosen_model(),
              trained_on = c(rows = nrow(pooled), train = sum(pooled$split == "train"),
                             validation = sum(pooled$split == "validation")),
              cv = winner, created = Sys.time())
saveRDS(final, file.path(paths$models, "final_model.rds"))
# The train-only model is kept too: it is what the validation numbers above
# describe, and it lets 12/13 separate "more data" from "newer data".
saveRDS(list(featurizer = fz, classifier = holdout$fit[[which(holdout$model == winner$model)]]),
        file.path(paths$models, "train_only_model.rds"))
cli::cli_alert_success("Saved final model ({format(object.size(final), units = 'MB')})")
