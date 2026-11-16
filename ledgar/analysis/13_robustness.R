# 13 -- Robustness: how much should the headline number be trusted?
#
#   A. Fold-assignment noise: repeat the 5-fold CV with different fold seeds.
#   B. Boilerplate: grouped CV (template twins kept in one fold) vs plain CV,
#      and test accuracy as a function of similarity to the nearest
#      training provision.
#   C. Temporal drift: CV -> validation -> test for one frozen train-only
#      model, plus test re-weighted to the training label mix.
#   D. Learning curve: is more labelled data still worth collecting?
#   E. Error analysis: the most confident mistakes.
#
# Output: results/13_*.csv, figures/13_*.png

source("analysis/_setup.R")
col <- provision_colours()
blue <- col$categorical[["blue"]]

clean <- readRDS(file.path(paths$processed, "ledgar_clean.rds"))
train <- filter(clean, split == "train")
valid <- filter(clean, split == "validation")
test <- filter(clean, split == "test")
model <- list(selected = chosen_model())

cache <- function(name, expr) {
  file <- file.path(paths$processed, paste0("13_", name, ".rds"))
  if (!file.exists(file)) saveRDS(expr, file)
  readRDS(file)
}

# -- A. Repeated CV: sensitivity to the fold assignment -----------------------------
repeated <- cache("repeated_cv", map_dfr(1:3, \(r) {
  folds <- assign_folds(train$label, v = 5, seed = SEED + r)
  cv_classifier(train$text, train$label, folds, chosen_spec(), model, workers = 4)$metrics |>
    mutate(repeat_id = r)
}))
repeat_summary <- repeated |>
  summarise(accuracy = mean(accuracy), macro_f1 = mean(macro_f1), .by = repeat_id)
variance_parts <- tibble(
  metric = c("accuracy", "macro_f1"),
  mean = c(mean(repeated$accuracy), mean(repeated$macro_f1)),
  sd_between_folds = c(mean(tapply(repeated$accuracy, repeated$repeat_id, sd)),
                       mean(tapply(repeated$macro_f1, repeated$repeat_id, sd))),
  sd_between_repeats = c(sd(repeat_summary$accuracy), sd(repeat_summary$macro_f1))
)
write_result(variance_parts, "13_repeated_cv")
print(variance_parts)

# -- B1. Grouped vs plain CV on the *raw* training split -----------------------------
# (The cleaned split has one row per template, so twins can only straddle
# folds in the raw data.)
raw <- ledgar_read(paths$raw, "train") |> mutate(text = normalize_text(text))
groups <- flag_duplicates(raw$text)$group
grouped <- cache("grouped_cv", bind_rows(
  cv_classifier(raw$text, raw$label, assign_folds(raw$label, 5, seed = SEED), chosen_spec(), model,
                workers = 4)$metrics |> mutate(scheme = "stratified 5-fold (twins may straddle folds)"),
  cv_classifier(raw$text, raw$label, assign_folds(raw$label, 5, groups = groups, seed = SEED), chosen_spec(),
                model, workers = 4)$metrics |> mutate(scheme = "grouped 5-fold (template twins kept together)")
))
grouped_summary <- summarise_cv(grouped, by = "scheme")
write_result(grouped_summary, "13_grouped_cv")
print(select(grouped_summary, scheme, accuracy_mean, accuracy_se, macro_f1_mean, macro_f1_se))

# -- B2. Test accuracy vs similarity to the nearest training provision ---------------
test_pred <- readRDS(file.path(paths$processed, "12_test_predictions.rds"))
nn <- readRDS(file.path(paths$processed, "03_nn_similarity.rds")) |> filter(split == "test vs train")
stopifnot(nrow(nn) == nrow(test_pred))
by_sim <- test_pred |>
  mutate(sim = nn$sim,
         bin = cut(sim, c(0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.001), right = FALSE)) |>
  summarise(n = n(), accuracy = mean(correct), .by = bin) |>
  arrange(bin) |>
  mutate(se = sqrt(accuracy * (1 - accuracy) / n), share = n / sum(n))
write_result(by_sim, "13_accuracy_by_similarity")
print(by_sim)

p_sim <- ggplot(by_sim, aes(bin, accuracy)) +
  geom_col(aes(y = share), fill = col$grid, width = 0.7) +
  geom_linerange(aes(ymin = accuracy - 2 * se, ymax = accuracy + 2 * se), colour = blue, linewidth = 0.7) +
  geom_point(colour = blue, size = 2.4) +
  geom_text(aes(label = scales::percent(accuracy, 1)), vjust = -1.3, size = 3, colour = col$ink2) +
  geom_text(aes(y = share, label = scales::percent(share, 1)), vjust = -0.5, size = 2.7, colour = col$muted) +
  scale_y_continuous(labels = scales::percent, limits = c(0, 1.06), breaks = seq(0, 1, 0.25)) +
  labs(title = "Accuracy depends on how familiar a provision looks",
       subtitle = "Test accuracy (blue, +/- 2 SE) by cosine similarity to the nearest training provision; grey bars = share of test set.",
       x = "TF-IDF cosine similarity to nearest training provision", y = NULL)
save_figure(p_sim, file.path(paths$figures, "13_accuracy_by_similarity.png"), width = 9, height = 4.6)

# -- C. Temporal drift ------------------------------------------------------------------
train_only <- readRDS(file.path(paths$models, "train_only_model.rds"))
score_split <- function(d) {
  pred <- predict(train_only$classifier, bake_features(train_only$featurizer, d$text))
  list(metrics = classification_metrics(d$label, pred), pred = pred)
}
v <- score_split(valid)
t <- score_split(test)

# Label-shift adjustment: re-weight test rows so the class mix matches train.
# What is left of the gap is not explained by the changed label proportions.
w <- (prop.table(table(train$label)) / prop.table(table(test$label)))[as.character(test$label)]
w[!is.finite(w)] <- 0
test_reweighted <- weighted.mean(t$pred == test$label, w)

cv_ref <- repeated |> summarise(accuracy = mean(accuracy), macro_f1 = mean(macro_f1))
drift <- tribble(
  ~estimate, ~accuracy, ~macro_f1,
  "5-fold CV on train (random folds)", cv_ref$accuracy, cv_ref$macro_f1,
  "validation (next period)", v$metrics$accuracy, v$metrics$macro_f1,
  "test (latest period)", t$metrics$accuracy, t$metrics$macro_f1,
  "test, re-weighted to train label mix", test_reweighted, NA
) |> mutate(n = c(nrow(train), nrow(valid), nrow(test), nrow(test)),
            se = sqrt(accuracy * (1 - accuracy) / n))
write_result(drift, "13_temporal_drift")
print(drift)

p_drift <- drift |>
  mutate(estimate = fct_rev(fct_inorder(estimate))) |>
  ggplot(aes(accuracy, estimate)) +
  geom_linerange(aes(xmin = accuracy - 2 * se, xmax = accuracy + 2 * se), colour = blue, linewidth = 0.7) +
  geom_point(colour = blue, size = 2.6) +
  geom_text(aes(label = scales::percent(accuracy, 0.1)), vjust = -1.1, size = 3.1, colour = col$ink2) +
  scale_x_continuous(labels = scales::label_percent(accuracy = 1), expand = expansion(mult = 0.1)) +
  labs(title = "Performance holds up on later filings: CV, validation and test agree within a point",
       subtitle = "Accuracy of the same train-only pipeline under four evaluation schemes (+/- 2 binomial SE).\nCV models see 4/5 of the training split; the hold-out model sees all of it.",
       x = "Accuracy", y = NULL) +
  theme_provision(grid = "x")
save_figure(p_drift, file.path(paths$figures, "13_temporal_drift.png"), width = 9, height = 3.9)

# -- D. Learning curve --------------------------------------------------------------------
sizes <- c(1000, 2000, 5000, 10000, 20000, 40000, nrow(train))
curve <- cache("learning_curve", {
  jobs <- tibble(n = sizes, reps = if_else(sizes < nrow(train), 3L, 1L)) |> uncount(reps, .id = "rep")
  parallel::mclapply(seq_len(nrow(jobs)), \(i) {
    set.seed(SEED + i)
    d <- slice_sample(train, n = jobs$n[i])
    fz <- fit_featurizer(d$text, chosen_spec())
    fit <- do.call(fit_classifier, c(list(x = bake_features(fz, d$text), y = d$label), chosen_model()))
    classification_metrics(valid$label, predict(fit, bake_features(fz, valid$text))) |>
      mutate(n = jobs$n[i], rep = jobs$rep[i], classes_seen = n_distinct(d$label))
  }, mc.cores = 4) |> bind_rows()
})
curve_summary <- curve |>
  summarise(across(c(accuracy, macro_f1), list(mean = mean, min = min, max = max)), .by = n)
write_result(curve_summary, "13_learning_curve")
print(curve_summary)

# gain from the last doubling of data, in percentage points
last_doubling <- with(curve_summary, 100 * (accuracy_mean[n == max(n)] - accuracy_mean[n == 20000])) /
  log2(nrow(train) / 20000)
p_curve <- curve_summary |>
  pivot_longer(-n, names_to = c("metric", ".value"), names_pattern = "(.*)_(mean|min|max)") |>
  mutate(metric = recode(metric, accuracy = "Accuracy (micro-F1)", macro_f1 = "Macro-F1")) |>
  ggplot(aes(n, mean, colour = metric)) +
  geom_linerange(aes(ymin = min, ymax = max), linewidth = 0.7, alpha = 0.6) +
  geom_line(linewidth = 0.8) + geom_point(size = 2) +
  geom_text(data = \(d) filter(d, n == max(n)), aes(label = metric), hjust = 1, vjust = -1.3, size = 3.1,
            show.legend = FALSE) +
  scale_colour_provision(guide = "none") +
  scale_x_log10(labels = scales::comma, breaks = sizes) +
  scale_y_continuous(labels = scales::label_percent(accuracy = 1), expand = expansion(mult = c(0.05, 0.12))) +
  labs(title = glue::glue("More data still helps: about {sprintf('%.1f', last_doubling)} pp accuracy per doubling beyond 20k provisions"),
       subtitle = "Validation score vs number of training provisions (log scale); vertical ranges = 3 random subsamples.",
       x = "Training provisions", y = "Score on the validation split") +
  theme_provision(grid = "xy")
save_figure(p_curve, file.path(paths$figures, "13_learning_curve.png"), width = 9, height = 4.6)

# -- E. The most confident mistakes -----------------------------------------------------------
confident_errors <- test_pred |>
  filter(!correct) |>
  slice_max(margin, n = 25) |>
  left_join(select(test, id, text), by = "id") |>
  transmute(id, true_label = label, predicted = estimate, margin = round(margin, 2),
            text = str_trunc(text, 220))
write_result(confident_errors, "13_confident_errors")
print(select(confident_errors, true_label, predicted, margin), n = 12)
