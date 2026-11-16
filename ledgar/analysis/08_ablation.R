# 08 -- Iteration 1: which feature decisions earn their keep?
#
# Stage A (screening): 5-fold CV on a 20k stratified subsample, linear SVM
#   (C = 1), one change at a time relative to the baseline feature set.
# Stage B (confirmation): the baseline and the most promising combinations
#   are re-run on the full cleaned training split with the same folds as
#   07_baselines.R, so the numbers are directly comparable.
#
# Output: results/08_*.csv, figures/08_*.png

source("analysis/_setup.R")
col <- provision_colours()

train <- readRDS(file.path(paths$processed, "ledgar_clean.rds")) |> filter(split == "train")
svm <- list(svm = list(engine = "svm", cost = 1))

# -- Stage A: screening on a subsample ----------------------------------------
set.seed(SEED)
sub <- train |> slice_sample(n = 20000)
# guarantee every class is present at least 5 times in the subsample
sub <- bind_rows(sub, train |> filter(!id %in% sub$id) |>
                   semi_join(count(sub, label) |> filter(n < 10), by = "label")) |>
  bind_rows(train |> filter(!label %in% sub$label))
sub <- distinct(sub, id, .keep_all = TRUE)
sub_folds <- assign_folds(sub$label, v = 5, seed = SEED)

base <- list(word_ngram = c(1, 2))
variants <- list(
  "baseline: word 1-2, sublinear tf-idf, digits masked" = base,
  "word 1-gram only" = list(word_ngram = c(1, 1)),
  "word 1-3-gram" = list(word_ngram = c(1, 3)),
  "no digit masking" = c(base, list(mask_numbers = FALSE)),
  "no lower-casing" = c(base, list(lowercase = FALSE)),
  "raw tf (not sublinear)" = c(base, list(sublinear_tf = FALSE)),
  "no idf" = c(base, list(use_idf = FALSE)),
  "min_doc_count = 5" = c(base, list(min_doc_count = 5)),
  "max_doc_prop = 0.5" = c(base, list(max_doc_prop = 0.5)),
  "+ head block" = c(base, list(head_ngram = c(1, 2))),
  "+ surface statistics" = c(base, list(stats = TRUE)),
  "+ char 2-5-gram" = c(base, list(char_ngram = c(2, 5))),
  "char 2-5-gram only" = list(word_ngram = NULL, char_ngram = c(2, 5)),
  "+ head + stats" = c(base, list(head_ngram = c(1, 2), stats = TRUE))
)
# (An all-blocks variant was dropped: character n-grams take ~4x the compute
# of everything else combined and "+ char" alone is within noise of baseline.)

# Each configuration is cached so an interrupted run resumes where it stopped.
# Character n-grams need several GB per worker, so they run with fewer workers.
cached_cv <- function(name, args, text, y, folds, stage, models = svm) {
  dir <- file.path(paths$processed, stage)
  dir.create(dir, showWarnings = FALSE)
  file <- file.path(dir, paste0(make.names(name), ".rds"))
  if (!file.exists(file)) {
    cli::cli_alert_info("{stage}: {name}")
    workers <- if (is.null(args$char_ngram)) 4 else if (length(text) > 30000) 1 else 2
    cv <- cv_classifier(text, y, folds, spec = do.call(feature_spec, args), models = models, workers = workers)
    saveRDS(mutate(cv$metrics, variant = name, .before = 1), file)
  }
  readRDS(file)
}

screening <- imap_dfr(variants, \(args, name) cached_cv(name, args, sub$text, sub$label, sub_folds, "08_screening"))
write_result(screening, "08_screening_folds")

# Paired comparison against the baseline: same folds, so compare fold by fold.
base_name <- names(variants)[1]
paired <- screening |>
  select(variant, fold, accuracy, macro_f1, n_features) |>
  left_join(screening |> filter(variant == base_name) |>
              select(fold, base_acc = accuracy, base_f1 = macro_f1), by = "fold") |>
  # differences first: summarise() would otherwise overwrite `accuracy`
  # with its mean before the paired differences are taken
  mutate(d_acc = accuracy - base_acc, d_f1 = macro_f1 - base_f1) |>
  summarise(
    n_features = mean(n_features),
    accuracy = mean(accuracy), macro_f1 = mean(macro_f1),
    d_accuracy = mean(d_acc), d_accuracy_se = sd(d_acc) / sqrt(n()),
    d_macro_f1 = mean(d_f1), d_macro_f1_se = sd(d_f1) / sqrt(n()),
    folds_better = sum(d_acc > 0),
    .by = variant
  ) |>
  arrange(desc(d_accuracy))
write_result(paired, "08_screening_paired")
print(select(paired, variant, n_features, accuracy, d_accuracy, d_accuracy_se, d_macro_f1, folds_better), n = 20)

p_screen <- paired |>
  filter(variant != base_name) |>
  select(variant, `Accuracy (micro-F1)_d` = d_accuracy, `Accuracy (micro-F1)_se` = d_accuracy_se,
         `Macro-F1_d` = d_macro_f1, `Macro-F1_se` = d_macro_f1_se) |>
  pivot_longer(-variant, names_to = c("metric", ".value"), names_sep = "_") |>
  mutate(variant = fct_reorder(variant, d * (metric == "Accuracy (micro-F1)"), .fun = sum),
         direction = if_else(d >= 0, "better than baseline", "worse than baseline")) |>
  ggplot(aes(d, variant, colour = direction)) +
  geom_vline(xintercept = 0, colour = col$axis) +
  geom_linerange(aes(xmin = d - 2 * se, xmax = d + 2 * se), linewidth = 0.7) +
  geom_point(size = 2.2) +
  facet_wrap(vars(metric)) +
  scale_colour_manual(values = c("better than baseline" = col$categorical[["blue"]],
                                 "worse than baseline" = col$categorical[["red"]]), name = NULL) +
  scale_x_continuous(labels = scales::label_percent(accuracy = 0.1, suffix = " pp")) +
  labs(title = "Feature ablation: what changes relative to plain word 1-2-gram TF-IDF",
       subtitle = glue::glue("Paired fold-wise difference to the baseline, linear SVM, 5-fold CV on a ",
                             "{scales::comma(nrow(sub))}-provision subsample. Bars: +/- 2 SE of the paired difference."),
       x = "Change vs baseline (percentage points)", y = NULL) +
  theme_provision(grid = "x")
save_figure(p_screen, file.path(paths$figures, "08_ablation_screening.png"), width = 10.5, height = 5.5)

# -- Stage A2: is the comparison fair? Block weights x regularisation ----------
# Adding a unit-norm block changes the scale of every document vector, which
# for an SVM acts like changing C. A block can therefore look useless (or
# useful) merely because the effective regularisation moved. Re-test the
# extra blocks at several weights, each across a range of C.
svm_costs <- list("C=0.3" = list(engine = "svm", cost = 0.3), "C=1" = list(engine = "svm", cost = 1),
                  "C=3" = list(engine = "svm", cost = 3))
weighted <- list(
  "word only" = base,
  "+ head (w=1)" = c(base, list(head_ngram = c(1, 2))),
  "+ head (w=0.5)" = c(base, list(head_ngram = c(1, 2), weights = c(head = 0.5))),
  "+ head (w=0.25)" = c(base, list(head_ngram = c(1, 2), weights = c(head = 0.25))),
  "+ stats (w=1)" = c(base, list(stats = TRUE)),
  "+ stats (w=0.5)" = c(base, list(stats = TRUE, weights = c(stats = 0.5))),
  "+ char (w=0.5)" = c(base, list(char_ngram = c(2, 5), weights = c(char = 0.5)))
)
weights_cv <- imap_dfr(weighted, \(args, name) {
  cached_cv(name, args, sub$text, sub$label, sub_folds, "08_weights", models = svm_costs)
})
write_result(weights_cv, "08_weights_folds")

weights_summary <- weights_cv |>
  summarise(accuracy_se = sd(accuracy) / sqrt(n()), macro_f1_se = sd(macro_f1) / sqrt(n()),
            accuracy = mean(accuracy), macro_f1 = mean(macro_f1), .by = c(variant, model)) |>
  mutate(cost = parse_number(model))
write_result(weights_summary, "08_weights_summary")
weights_summary |> select(variant, model, accuracy) |>
  pivot_wider(names_from = model, values_from = accuracy) |> print()

best_word_only <- weights_summary |> filter(variant == "word only") |> slice_max(accuracy, n = 1)
p_weights <- weights_summary |>
  mutate(variant = factor(variant, levels = names(weighted)),
         block = case_when(str_detect(variant, "head") ~ "head block", str_detect(variant, "stats") ~ "surface statistics",
                           str_detect(variant, "char") ~ "char n-grams", TRUE ~ "word only")) |>
  ggplot(aes(factor(cost), variant, fill = accuracy)) +
  geom_tile(colour = col$surface, linewidth = 1.5) +
  geom_text(aes(label = sprintf("%.1f", 100 * accuracy),
                colour = accuracy > quantile(accuracy, 0.6)), size = 3.4, show.legend = FALSE) +
  scale_colour_manual(values = c(`TRUE` = "white", `FALSE` = col$ink)) +
  scale_fill_gradientn(colours = col$sequential, labels = scales::label_percent(accuracy = 0.1),
                       name = "CV accuracy") +
  scale_y_discrete(limits = rev) +
  labs(title = "A down-weighted head block is the one addition that helps at every C",
       subtitle = glue::glue("5-fold CV accuracy (%) on the {scales::comma(nrow(sub))}-provision subsample, by feature set and SVM cost.\n",
                             "At weight 1 the head block looked harmful; the fold SE is about ",
                             "{sprintf('%.2f', 100 * median(weights_summary$accuracy_se))} pp."),
       x = "SVM cost C", y = NULL) +
  theme_provision(grid = "none") +
  theme(axis.line = element_blank(), legend.position = "none")
save_figure(p_weights, file.path(paths$figures, "08_ablation_weights.png"), width = 8, height = 5)

# -- Stage A3: design of the head block -----------------------------------------
# A2 overturned the screening verdict on the head block: at weight 0.5 it
# helps at every C. Before confirming on the full data, check how many
# opening words and which n-gram range it should use (C = 1).
head_design <- list(
  "6 words, 1-2-gram" = list(head_words = 6, head_ngram = c(1, 2)),
  "12 words, 1-2-gram" = list(head_words = 12, head_ngram = c(1, 2)),
  "12 words, 1-3-gram" = list(head_words = 12, head_ngram = c(1, 3)),
  "20 words, 1-2-gram" = list(head_words = 20, head_ngram = c(1, 2)),
  "20 words, 1-3-gram" = list(head_words = 20, head_ngram = c(1, 3)),
  "35 words, 1-2-gram" = list(head_words = 35, head_ngram = c(1, 2))
)
head_cv <- imap_dfr(head_design, \(args, name) {
  cached_cv(name, c(base, args, list(weights = c(head = 0.5))), sub$text, sub$label, sub_folds, "08_head")
})
word_only_c1 <- weights_cv |> filter(variant == "word only", model == "C=1") |>
  select(fold, base_acc = accuracy, base_f1 = macro_f1)
head_summary <- head_cv |>
  left_join(word_only_c1, by = "fold") |>
  mutate(d_acc = accuracy - base_acc, d_f1 = macro_f1 - base_f1) |>
  summarise(accuracy = mean(accuracy), macro_f1 = mean(macro_f1),
            d_accuracy = mean(d_acc), d_accuracy_se = sd(d_acc) / sqrt(n()),
            d_macro_f1 = mean(d_f1), folds_better = sum(d_acc > 0), .by = variant)
write_result(head_summary, "08_head_design")
print(head_summary)

# -- Stage B: confirm on the full training split -------------------------------
folds <- readRDS(file.path(paths$processed, "cv_folds.rds"))
best_head <- head_design[[head_summary$variant[which.max(head_summary$accuracy)]]]
cli::cli_alert_info("Best head design: {head_summary$variant[which.max(head_summary$accuracy)]}")
candidates <- list(
  "word 1-2 (baseline)" = base,
  "word 1-2 + head (w=0.5)" = c(base, best_head, list(weights = c(head = 0.5))),
  "word 1-2 + head (w=0.5) + stats (w=0.5)" = c(base, best_head, list(stats = TRUE, weights = c(head = 0.5, stats = 0.5)))
)
confirmation <- imap_dfr(candidates, \(args, name) cached_cv(name, args, train$text, train$label, folds, "08_confirmation"))
write_result(confirmation, "08_confirmation_folds")
confirmed <- summarise_cv(confirmation, by = "variant") |>
  left_join(summarise(confirmation, n_features = mean(n_features), .by = variant), by = "variant")
write_result(confirmed, "08_confirmation")
print(select(confirmed, variant, n_features, accuracy_mean, accuracy_se, macro_f1_mean, macro_f1_se, fit_seconds))

p_confirm <- plot_cv_summary(arrange(confirmed, accuracy_mean), by = "variant") +
  labs(title = "Confirmation on the full training split",
       subtitle = "Linear SVM (C = 1), 5-fold stratified CV, same folds as the baselines. Mean +/- 2 SE over folds.")
save_figure(p_confirm, file.path(paths$figures, "08_ablation_confirmation.png"), width = 9.5, height = 3.8)
