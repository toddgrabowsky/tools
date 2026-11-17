# 14 -- Fine-tuned transformer vs TF-IDF + linear SVM.
#
# The transformer (Legal-BERT-small, fine-tuned by transformer/finetune.py)
# and the SVM are compared like for like: both trained on the train split
# only, both scored on the same 10,000 test provisions, with the same metric
# code. Anything that had to be chosen for the comparison itself (probability
# temperatures, ensemble weight) is chosen on the validation split.
#
# Output: results/14_*.csv, figures/14_*.png

source("analysis/_setup.R")
library(patchwork)
col <- provision_colours()
blue <- col$categorical[["blue"]]
orange <- col$categorical[["orange"]]
SVM <- "TF-IDF + linear SVM"
BERT <- "Legal-BERT-small (fine-tuned)"

bert_dir <- file.path(paths$models, "legal-bert-small")
clean <- readRDS(file.path(paths$processed, "ledgar_clean.rds"))
valid <- filter(clean, split == "validation")
test <- filter(clean, split == "test")
lv <- levels(test$label)

# -- Scores of both models on validation and test --------------------------------
read_logits <- function(split, ids) {
  d <- nanoparquet::read_parquet(file.path(bert_dir, paste0("logits_", split, ".parquet")))
  stopifnot(identical(d$id, ids))          # same rows, same order as the R side
  m <- as.matrix(d[, -1])
  colnames(m) <- lv                        # column j = label id j - 1
  m
}
bert_valid <- read_logits("validation", valid$id)
bert_test <- read_logits("test", test$id)

svm <- readRDS(file.path(paths$models, "train_only_model.rds"))
svm_valid <- predict(svm$classifier, bake_features(svm$featurizer, valid$text), type = "score")
t_svm <- system.time(
  svm_test <- predict(svm$classifier, bake_features(svm$featurizer, test$text), type = "score")
)[["elapsed"]]

top <- function(scores) factor(colnames(scores)[max.col(scores, ties.method = "first")], levels = lv)
pred <- tibble(id = test$id, truth = test$label, svm = top(svm_test), bert = top(bert_test))

# -- 1. Headline comparison ----------------------------------------------------------
cmp <- compare_predictions(pred$truth, pred$svm, pred$bert, n_boot = 2000, seed = SEED)
boot_svm <- bootstrap_metrics(pred$truth, pred$svm, n_boot = 2000, seed = SEED)
boot_bert <- bootstrap_metrics(pred$truth, pred$bert, n_boot = 2000, seed = SEED)

headline <- bind_rows(mutate(boot_svm, model = SVM), mutate(boot_bert, model = BERT)) |>
  filter(metric %in% c("accuracy", "macro_f1")) |>
  select(model, metric, estimate, lower, upper)
write_result(headline, "14_headline")
write_result(cmp$metrics, "14_paired_difference")
write_result(cmp$agreement |> mutate(outcome = recode(outcome, "only a correct" = "only SVM correct",
                                                      "only b correct" = "only BERT correct")),
             "14_agreement")
write_result(as_tibble(cmp$mcnemar), "14_mcnemar")
print(headline)
print(cmp$metrics)
print(cmp$agreement)
cli::cli_alert_info("McNemar exact p = {signif(cmp$mcnemar$p_value, 3)}")

metric_names <- c(accuracy = "Accuracy (micro-F1)", macro_f1 = "Macro-F1")
p_head <- headline |>
  mutate(metric = recode(metric, !!!metric_names), model = factor(model, c(BERT, SVM))) |>
  ggplot(aes(estimate, model, colour = model)) +
  geom_linerange(aes(xmin = lower, xmax = upper), linewidth = 1) +
  geom_point(size = 2.8) +
  geom_text(aes(label = sprintf("%.1f%%", 100 * estimate)), vjust = -1.1, size = 3.2, colour = col$ink2) +
  facet_wrap(vars(metric), scales = "free_x") +
  scale_colour_manual(values = setNames(c(blue, orange), c(SVM, BERT)), guide = "none") +
  scale_x_continuous(labels = scales::label_percent(accuracy = 0.1), expand = expansion(mult = 0.12)) +
  labs(subtitle = "Test score with 95% bootstrap interval", x = NULL, y = NULL) +
  theme_provision(grid = "x")

p_diff <- cmp$metrics |>
  filter(metric %in% names(metric_names)) |>
  mutate(metric = recode(metric, !!!metric_names)) |>
  ggplot(aes(difference, metric)) +
  geom_vline(xintercept = 0, colour = col$axis) +
  geom_linerange(aes(xmin = lower, xmax = upper), colour = col$ink2, linewidth = 1) +
  geom_point(colour = col$ink, size = 2.8) +
  geom_text(aes(label = sprintf("%+.2f pp  [%+.2f, %+.2f]", 100 * difference, 100 * lower, 100 * upper)),
            vjust = -1.2, size = 3.1, colour = col$ink2) +
  scale_x_continuous(labels = scales::label_percent(accuracy = 0.1, suffix = " pp"),
                     expand = expansion(mult = 0.25)) +
  labs(subtitle = "Paired difference, BERT minus SVM (same test rows resampled together)", x = NULL, y = NULL) +
  theme_provision(grid = "x")

winner_txt <- with(filter(cmp$metrics, metric == "accuracy"),
                   if (lower > 0) "the fine-tuned transformer is ahead" else if (upper < 0) "the linear model is ahead"
                   else "the two models are statistically tied on accuracy")
save_figure(p_head / p_diff + plot_layout(heights = c(1.2, 1)) + plot_annotation(
              title = glue::glue("Legal-BERT-small vs TF-IDF + SVM on the LEDGAR test split: {winner_txt}"),
              theme = theme_provision()),
            file.path(paths$figures, "14_headline.png"), width = 10, height = 6)

# -- 2. Training curve of the transformer ----------------------------------------------
# models/ is git-ignored, so keep the training record with the results
file.copy(file.path(bert_dir, "summary.json"), file.path(paths$results, "14_finetune_summary.json"), overwrite = TRUE)
file.copy(file.path(bert_dir, "history.csv"), file.path(paths$results, "14_finetune_history.csv"), overwrite = TRUE)
history <- read_csv(file.path(bert_dir, "history.csv"), show_col_types = FALSE)
summary_bert <- jsonlite::read_json(file.path(bert_dir, "summary.json"))
svm_valid_metrics <- classification_metrics(valid$label, top(svm_valid))
p_hist <- history |>
  select(epoch, `Accuracy (micro-F1)` = val_accuracy, `Macro-F1` = val_macro_f1) |>
  pivot_longer(-epoch, names_to = "metric") |>
  ggplot(aes(epoch, value)) +
  geom_hline(data = tibble(metric = c("Accuracy (micro-F1)", "Macro-F1"),
                           value = c(svm_valid_metrics$accuracy, svm_valid_metrics$macro_f1)),
             aes(yintercept = value), colour = blue, linetype = "dashed") +
  geom_line(colour = orange, linewidth = 0.8) + geom_point(colour = orange, size = 2.4) +
  geom_text(aes(label = sprintf("%.1f%%", 100 * value)), vjust = 1.9, size = 3, colour = col$ink2) +
  facet_wrap(vars(metric), scales = "free_y") +
  scale_x_continuous(breaks = history$epoch) +
  scale_y_continuous(labels = scales::label_percent(accuracy = 1), expand = expansion(mult = 0.2)) +
  labs(title = "Validation score of the transformer after each epoch",
       subtitle = glue::glue("Orange: Legal-BERT-small. Dashed blue: the SVM's validation score. ",
                             "Epoch {summary_bert$selected_epoch} was selected on validation accuracy."),
       x = "Epoch", y = NULL) +
  theme_provision(grid = "xy")
save_figure(p_hist, file.path(paths$figures, "14_training_curve.png"), width = 9, height = 4)

# -- 3. Per-class comparison --------------------------------------------------------------
per_class <- inner_join(
  per_class_metrics(pred$truth, pred$svm) |> select(label, support, f1_svm = f1),
  per_class_metrics(pred$truth, pred$bert) |> select(label, f1_bert = f1),
  by = "label"
) |> mutate(diff = f1_bert - f1_svm)
write_result(arrange(per_class, diff), "14_per_class")
cli::cli_alert_info("BERT better on {sum(per_class$diff > 0)} classes, SVM on {sum(per_class$diff < 0)}, tied on {sum(per_class$diff == 0)}")

p_class <- ggplot(per_class, aes(f1_svm, f1_bert)) +
  geom_abline(colour = col$axis) +
  geom_point(aes(size = support), colour = col$ink2, alpha = 0.55) +
  ggrepel::geom_text_repel(data = \(d) filter(d, abs(diff) > 0.1), aes(label = label), size = 2.8,
                           colour = col$ink2, seed = 1, max.overlaps = 30) +
  scale_size_area(max_size = 5, name = "Test examples") +
  scale_x_continuous(labels = scales::percent, limits = c(0, 1)) +
  scale_y_continuous(labels = scales::percent, limits = c(0, 1)) +
  coord_equal() +
  labs(title = "The same classes are hard for both models; the SVM leads on most",
       subtitle = glue::glue("Per-class F1 on test. Above the line the transformer is better ",
                             "({sum(per_class$diff > 0)} classes); below, the SVM ({sum(per_class$diff < 0)}).\n",
                             "Correlation {sprintf('%.2f', cor(per_class$f1_svm, per_class$f1_bert))}. ",
                             "Labelled: classes differing by more than 10 pp."),
       x = glue::glue("F1, {SVM}"), y = glue::glue("F1, {BERT}")) +
  theme_provision(grid = "xy")
save_figure(p_class, file.path(paths$figures, "14_per_class.png"), width = 8, height = 8.2)

# -- 4. Where do they differ? Slices ---------------------------------------------------------
nn <- readRDS(file.path(paths$processed, "03_nn_similarity.rds")) |> filter(split == "test vs train")
max_len <- summary_bert$max_length
slice_df <- pred |>
  mutate(n_words = test$n_words, sim = nn$sim,
         Length = cut(n_words, c(0, 30, 60, 120, 240, Inf),
                      labels = c("<= 30 words", "31-60", "61-120", "121-240", "> 240")),
         Familiarity = cut(sim, c(0, 0.4, 0.6, 0.8, 0.95, 1.001), right = FALSE,
                           labels = c("< 0.4", "0.4-0.6", "0.6-0.8", "0.8-0.95", ">= 0.95")))
slices <- slice_df |>
  pivot_longer(c(Length, Familiarity), names_to = "group", values_to = "slice") |>
  mutate(svm_ok = svm == truth, bert_ok = bert == truth) |>
  summarise(n = n(), svm = mean(svm_ok), bert = mean(bert_ok),
            only_svm = sum(svm_ok & !bert_ok), only_bert = sum(bert_ok & !svm_ok),
            .by = c(group, slice)) |>
  mutate(diff = bert - svm,
         # SE of a paired difference in proportions (discordant rows only)
         diff_se = sqrt(only_svm + only_bert - (only_bert - only_svm)^2 / n) / n) |>
  arrange(group, slice)
write_result(slices, "14_slices")
print(slices)

p_slices <- slices |>
  mutate(group = recode(group, Length = "Provision length",
                        Familiarity = "Cosine similarity to nearest training provision")) |>
  ggplot(aes(diff, fct_rev(slice))) +
  geom_vline(xintercept = 0, colour = col$axis) +
  geom_linerange(aes(xmin = diff - 2 * diff_se, xmax = diff + 2 * diff_se), colour = col$ink2, linewidth = 0.7) +
  geom_point(colour = col$ink, size = 2.4) +
  geom_text(aes(label = glue::glue("{sprintf('%+.1f', 100 * diff)} pp (n = {scales::comma(n)})")),
            vjust = -1, size = 2.9, colour = col$ink2) +
  facet_wrap(vars(group), scales = "free_y") +
  scale_x_continuous(labels = scales::label_percent(accuracy = 1, suffix = " pp"), expand = expansion(mult = 0.2)) +
  labs(title = "Where the two models differ",
       subtitle = glue::glue("Test accuracy of the transformer minus the SVM by slice (+/- 2 SE of the paired difference). ",
                             "The transformer reads at most {max_len} word pieces."),
       x = "Accuracy difference, BERT minus SVM", y = NULL) +
  theme_provision(grid = "x")
save_figure(p_slices, file.path(paths$figures, "14_slices.png"), width = 10, height = 4.2)

# -- 5. Do the models complement each other? Ensemble ------------------------------------------
# Probabilities via temperature-scaled soft-max; temperatures and the mixing
# weight are fitted on validation, then applied once to test.
temp_svm <- fit_temperature(svm_valid, valid$label)
temp_bert <- fit_temperature(bert_valid, valid$label)
p_svm_valid <- softmax_scores(svm_valid, temp_svm)
p_bert_valid <- softmax_scores(bert_valid, temp_bert)
weights <- tibble(w_bert = seq(0, 1, 0.05)) |>
  mutate(valid_accuracy = map_dbl(w_bert, \(w) mean(top(w * p_bert_valid + (1 - w) * p_svm_valid) == valid$label)))
w_best <- weights$w_bert[which.max(weights$valid_accuracy)]
write_result(weights, "14_ensemble_weights")
cli::cli_alert_info("temperatures: SVM {round(temp_svm, 3)}, BERT {round(temp_bert, 3)}; ensemble weight on BERT {w_best}")

p_svm_test <- softmax_scores(svm_test, temp_svm)
p_bert_test <- softmax_scores(bert_test, temp_bert)
p_ens_test <- w_best * p_bert_test + (1 - w_best) * p_svm_test
pred$ensemble <- top(p_ens_test)

boot_ens <- bootstrap_metrics(pred$truth, pred$ensemble, n_boot = 2000, seed = SEED)
cmp_ens <- compare_predictions(pred$truth, pred$svm, pred$ensemble, n_boot = 2000, seed = SEED)
oracle <- mean(pred$svm == pred$truth | pred$bert == pred$truth)
ensemble <- bind_rows(
  headline,
  boot_ens |> filter(metric %in% c("accuracy", "macro_f1")) |>
    transmute(model = glue::glue("Ensemble ({w_best} BERT + {1 - w_best} SVM)"), metric, estimate, lower, upper),
  tibble(model = "Oracle (either model correct)", metric = "accuracy", estimate = oracle)
)
write_result(ensemble, "14_ensemble")
write_result(cmp_ens$metrics, "14_ensemble_vs_svm")
print(ensemble)
print(cmp_ens$metrics)

p_ens <- ensemble |>
  filter(metric %in% names(metric_names)) |>
  mutate(metric = recode(metric, !!!metric_names), model = fct_rev(fct_inorder(as.character(model)))) |>
  ggplot(aes(estimate, model)) +
  geom_linerange(aes(xmin = lower, xmax = upper), colour = col$ink2, linewidth = 0.8, na.rm = TRUE) +
  geom_point(colour = col$ink, size = 2.5) +
  geom_text(aes(label = sprintf("%.1f%%", 100 * estimate)), vjust = -1, size = 3, colour = col$ink2) +
  facet_wrap(vars(metric), scales = "free_x") +
  scale_x_continuous(labels = scales::label_percent(accuracy = 1), expand = expansion(mult = 0.12)) +
  labs(title = glue::glue("Blending in the transformer moves the SVM by ",
                          "{sprintf('%+.1f', 100 * cmp_ens$metrics$difference[1])} pp accuracy"),
       subtitle = "Test scores with 95% bootstrap intervals. Temperatures and mixing weight were fitted on validation.\nThe oracle (right whenever either model is right) bounds what any combination could reach.",
       x = NULL, y = NULL) +
  theme_provision(grid = "x")
save_figure(p_ens, file.path(paths$figures, "14_ensemble.png"), width = 10, height = 4.2)

# -- 6. Confidence: calibration and selective prediction ------------------------------------------
conf_df <- bind_rows(
  tibble(model = SVM, confidence = apply(p_svm_test, 1, max), correct = pred$svm == pred$truth),
  tibble(model = BERT, confidence = apply(p_bert_test, 1, max), correct = pred$bert == pred$truth),
  tibble(model = "Ensemble", confidence = apply(p_ens_test, 1, max), correct = pred$ensemble == pred$truth)
) |> mutate(model = fct_inorder(model))

calibration <- conf_df |>
  mutate(bin = cut(confidence, seq(0, 1, 0.1), include.lowest = TRUE)) |>
  summarise(n = n(), confidence = mean(confidence), accuracy = mean(correct), .by = c(model, bin))
ece <- calibration |>
  summarise(ece = sum(n * abs(accuracy - confidence)) / sum(n), .by = model)
write_result(calibration, "14_calibration")
write_result(ece, "14_calibration_error")
print(ece)

coverage <- conf_df |>
  arrange(model, desc(confidence)) |>
  mutate(coverage = row_number() / n(), accuracy = cummean(correct), .by = model) |>
  filter(row_number() %% 50 == 0, .by = model)
coverage_tbl <- map_dfr(c(0.5, 0.7, 0.8, 0.9), \(cv) {
  coverage |> slice_min(abs(coverage - cv), n = 1, by = model, with_ties = FALSE) |>
    transmute(model, coverage = cv, accuracy)
})
write_result(coverage_tbl, "14_selective_prediction")

model_cols <- setNames(c(blue, orange, col$categorical[["aqua"]]), levels(conf_df$model))
p_cal <- ggplot(calibration, aes(confidence, accuracy, colour = model)) +
  geom_abline(colour = col$axis) +
  geom_line(linewidth = 0.7) + geom_point(aes(size = n), alpha = 0.8) +
  scale_colour_manual(values = model_cols, name = NULL) +
  scale_size_area(max_size = 4, guide = "none") +
  scale_x_continuous(labels = scales::percent, limits = c(0, 1)) +
  scale_y_continuous(labels = scales::percent, limits = c(0, 1)) +
  labs(subtitle = "Reliability after temperature scaling (point size = test rows)",
       x = "Predicted probability of the chosen class", y = "Observed accuracy") +
  theme_provision(grid = "xy")
p_cov <- ggplot(coverage, aes(coverage, accuracy, colour = model)) +
  geom_line(linewidth = 0.8) +
  scale_colour_manual(values = model_cols, guide = "none") +
  scale_x_continuous(labels = scales::percent) +
  scale_y_continuous(labels = scales::label_percent(accuracy = 1)) +
  labs(subtitle = "Accuracy when only the most confident share is auto-labelled",
       x = "Coverage", y = "Accuracy on the covered share") +
  theme_provision(grid = "xy")
save_figure(p_cal + p_cov + plot_annotation(
              title = "The transformer's probabilities are better calibrated; the SVM ranks its own errors as well",
              theme = theme_provision()),
            file.path(paths$figures, "14_confidence.png"), width = 11, height = 5)

# -- 7. Cost -----------------------------------------------------------------------------------------
svm_fit_seconds <- read_csv(file.path(paths$results, "09_tuning.csv"), show_col_types = FALSE) |>
  filter(model == chosen_model_name()) |> pull(fit_seconds)
cost <- tribble(
  ~model, ~parameters, ~train_minutes, ~test_predict_seconds, ~hardware,
  SVM, length(svm$classifier$model$W), svm_fit_seconds / 60 * 5 / 4, t_svm, "1 CPU core",
  BERT, summary_bert$parameters, summary_bert$train_minutes, summary_bert$test_predict_seconds, "4 CPU cores"
)
write_result(cost, "14_cost")
print(cost)

saveRDS(pred, file.path(paths$processed, "14_predictions.rds"))
