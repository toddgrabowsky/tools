# 11 -- Validation: target shuffling.
#
# Could the pipeline reach its score without any real text->label signal --
# through leakage, a bug, or the sheer flexibility of ~10^5 features? Shuffle
# the labels, rerun the *entire* cross-validation, repeat. If the pipeline is
# honest, shuffled-label scores collapse to the majority-class rate.
#
# A 100-class linear SVM on label noise converges slowly, so the test runs
# on a stratified subsample; the observed score is recomputed on the same
# subsample and folds so observed and null are strictly comparable.
#
# Output: results/11_*.csv, figures/11_target_shuffle.png

source("analysis/_setup.R")

train <- readRDS(file.path(paths$processed, "ledgar_clean.rds")) |> filter(split == "train")
N_SUB <- 6000
N_PERM <- 100

set.seed(SEED)
sub <- slice_sample(train, n = N_SUB)
folds <- assign_folds(sub$label, v = 5, seed = SEED)

null_file <- file.path(paths$processed, "11_target_shuffle.rds")
if (!file.exists(null_file)) {
  elapsed <- system.time(
    ts <- target_shuffle(sub$text, sub$label, folds, spec = chosen_spec(), model = chosen_model(),
                         n_perm = N_PERM, seed = SEED, workers = 4)
  )
  cli::cli_alert_info("target shuffling took {round(elapsed[['elapsed']] / 60, 1)} min")
  saveRDS(ts, null_file)
}
ts <- readRDS(null_file)

write_result(ts$null, "11_target_shuffle_null")
majority_rate <- max(table(sub$label)) / nrow(sub)
chance_rate <- sum(prop.table(table(sub$label))^2)   # expected accuracy of label-frequency guessing
summary_tbl <- tibble(
  metric = names(ts$observed),
  observed = unlist(ts$observed),
  null_mean = map_dbl(metric, \(m) mean(ts$null[[m]])),
  null_sd = map_dbl(metric, \(m) sd(ts$null[[m]])),
  null_max = map_dbl(metric, \(m) max(ts$null[[m]])),
  z = (observed - null_mean) / null_sd,
  p_value = ts$p_value[metric],
  n_perm = nrow(ts$null)
)
write_result(summary_tbl, "11_target_shuffle_summary")
print(summary_tbl, width = Inf)
cli::cli_alert_info("majority-class rate {round(majority_rate, 4)}; frequency-guessing rate {round(chance_rate, 4)}")

# Why resampling is essential here: fitted to shuffled labels, the same model
# reproduces its *training* labels almost perfectly. In-sample fit is no
# evidence of signal for a model with ~10^5 free parameters per class.
set.seed(SEED)
shuffled <- sample(sub$label)
fz <- fit_featurizer(sub$text, chosen_spec())
x <- bake_features(fz, sub$text)
memorisation <- tibble(
  labels = c("real", "shuffled"),
  in_sample_accuracy = c(
    mean(predict(do.call(fit_classifier, c(list(x = x, y = sub$label), chosen_model())), x) == sub$label),
    mean(predict(do.call(fit_classifier, c(list(x = x, y = shuffled), chosen_model())), x) == shuffled)),
  cross_validated_accuracy = c(ts$observed$accuracy, mean(ts$null$accuracy))
)
write_result(memorisation, "11_memorisation")
print(memorisation)

library(patchwork)
p_value_label <- function(m) glue::glue("permutation p <= {signif(ts$p_value[[m]], 2)}")
p <- plot_target_shuffle(ts, "accuracy") + labs(title = NULL, subtitle = p_value_label("accuracy")) +
  plot_target_shuffle(ts, "macro_f1") + labs(title = NULL, subtitle = p_value_label("macro_f1")) +
  plot_annotation(
    title = "Target shuffling: with shuffled labels the pipeline learns nothing",
    subtitle = glue::glue(
      "Full 5-fold CV of the selected pipeline on {scales::comma(N_SUB)} training provisions, real labels vs {N_PERM} ",
      "label permutations.\nMajority-class rate on this sample: {scales::percent(majority_rate, 0.1)}."),
    theme = theme_provision()
  )
save_figure(p, file.path(paths$figures, "11_target_shuffle.png"), width = 11, height = 4.6)

# Zoom on the null alone: it should sit at or below the majority-class rate.
col <- provision_colours()
p_null <- ts$null |>
  select(accuracy, macro_f1) |>
  pivot_longer(everything(), names_to = "metric") |>
  mutate(metric = recode(metric, accuracy = "Accuracy (micro-F1)", macro_f1 = "Macro-F1")) |>
  ggplot(aes(value)) +
  geom_histogram(bins = 30, fill = col$muted, colour = col$surface, linewidth = 0.2) +
  geom_vline(data = tibble(metric = "Accuracy (micro-F1)", x = c(majority_rate, chance_rate),
                           what = c("majority-class rate", "frequency-guessing rate")),
             aes(xintercept = x, colour = what), linewidth = 0.8) +
  facet_wrap(vars(metric), scales = "free") +
  scale_colour_provision(name = NULL) +
  scale_x_continuous(labels = scales::label_percent(accuracy = 0.1)) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.05))) +
  labs(title = "With shuffled labels the pipeline scores between random guessing and the majority rate",
       subtitle = "Cross-validated scores under shuffled labels only (the observed score is far off the right edge).",
       x = "Cross-validated score with shuffled labels", y = "Permutations")
save_figure(p_null, file.path(paths$figures, "11_target_shuffle_null.png"), width = 9, height = 4.2)
