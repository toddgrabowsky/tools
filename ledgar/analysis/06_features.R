# 06 -- Feature engineering: what do the candidate feature blocks look like,
# and is there prima facie evidence that each carries class signal?
#
# Blocks (provisionr::feature_spec): word n-gram TF-IDF, character n-gram
# TF-IDF, "head" n-grams (opening words only) and dense surface statistics.
# This script only describes them; whether they earn their keep is decided
# by cross-validated ablation in 08_ablation.R.
#
# Output: results/06_*.csv, figures/06_*.png

source("analysis/_setup.R")
col <- provision_colours()
blue <- col$categorical[["blue"]]

train <- readRDS(file.path(paths$processed, "ledgar_clean.rds")) |> filter(split == "train")

# -- 1. Size and sparsity of each block --------------------------------------
specs <- list(
  "word 1-gram" = feature_spec(word_ngram = c(1, 1)),
  "word 1-2-gram" = feature_spec(word_ngram = c(1, 2)),
  "word 1-3-gram" = feature_spec(word_ngram = c(1, 3)),
  "char 2-5-gram" = feature_spec(word_ngram = NULL, char_ngram = c(2, 5), min_doc_count = 5),
  "head 1-2-gram (first 12 words)" = feature_spec(word_ngram = NULL, head_ngram = c(1, 2)),
  "surface statistics" = feature_spec(word_ngram = NULL, stats = TRUE)
)
set.seed(SEED)
sample_rows <- sample(nrow(train), 10000)
block_shape <- imap_dfr(specs, \(spec, name) {
  secs <- system.time({
    fz <- fit_featurizer(train$text[sample_rows], spec)
    x <- bake_features(fz, train$text[sample_rows])
  })[["elapsed"]]
  tibble(block = name, features = ncol(x), nonzero_per_doc = Matrix::nnzero(x) / nrow(x),
         density = Matrix::nnzero(x) / prod(dim(x)), seconds_per_10k_docs = secs)
})
write_result(block_shape, "06_block_shape")
print(block_shape)

# -- 2. Surface statistics: how much class signal per statistic? -------------
stats_long <- as_tibble(text_stats(train$text)) |>
  mutate(label = train$label) |>
  pivot_longer(-label, names_to = "statistic")

# eta^2 of a one-way ANOVA on ranks = share of variance explained by class
stat_signal <- stats_long |>
  mutate(r = rank(value), .by = statistic) |>
  summarise(eta_sq = summary(lm(r ~ label))$r.squared, .by = statistic) |>
  arrange(desc(eta_sq))
write_result(stat_signal, "06_stat_signal")

p_signal <- stat_signal |>
  mutate(statistic = fct_reorder(statistic, eta_sq)) |>
  ggplot(aes(eta_sq, statistic)) +
  geom_col(fill = blue, width = 0.65) +
  geom_text(aes(label = scales::percent(eta_sq, 1)), hjust = -0.2, size = 3.2, colour = col$ink2) +
  scale_x_continuous(labels = scales::percent, expand = expansion(mult = c(0, 0.12))) +
  labs(title = "Every surface statistic carries some class signal",
       subtitle = "Share of rank variance explained by the class label (eta-squared).",
       x = NULL, y = NULL) +
  theme_provision(grid = "x")

top_stat <- stat_signal$statistic[1]
class_order <- stats_long |> filter(statistic == top_stat) |>
  summarise(m = median(value), .by = label) |> arrange(m)
show <- c(head(class_order$label, 6), tail(class_order$label, 6))
p_top <- stats_long |>
  filter(statistic == top_stat, label %in% show) |>
  mutate(label = factor(label, levels = show)) |>
  ggplot(aes(value, label)) +
  geom_boxplot(fill = alpha(blue, 0.25), colour = blue, outlier.size = 0.3, outlier.alpha = 0.3, width = 0.6) +
  labs(title = glue::glue("'{top_stat}' by class"),
       subtitle = "Six lowest and six highest classes by median.", x = top_stat, y = NULL) +
  theme_provision(grid = "x")
save_figure(patchwork::wrap_plots(p_signal, p_top, ncol = 2),
            file.path(paths$figures, "06_surface_statistics.png"), width = 11, height = 4.8)

# -- 3. Head block: how much does the opening tell us? ------------------------
# Purity of the label given the first k words (after normalisation).
head_purity <- map_dfr(1:8, \(k) {
  tibble(label = train$label,
         head = word(normalize_text(train$text, lowercase = TRUE, mask_numbers = TRUE), 1, k)) |>
    filter(!is.na(head)) |>
    count(head, label) |>
    summarise(top = max(n), n = sum(n), .by = head) |>
    summarise(k = k, distinct_openings = n(),
              # accuracy of "predict the most common label for this opening" (in-sample, optimistic)
              in_sample_accuracy = sum(top) / sum(n),
              # the same, ignoring openings that occur only once
              coverage_repeated = sum(n[n > 1]) / sum(n),
              accuracy_on_repeated = sum(top[n > 1]) / sum(n[n > 1]))
})
write_result(head_purity, "06_head_purity")
print(head_purity)

p_head <- head_purity |>
  select(k, `share of provisions whose opening recurs` = coverage_repeated,
         `label purity among recurring openings` = accuracy_on_repeated) |>
  pivot_longer(-k) |>
  ggplot(aes(k, value, colour = name)) +
  geom_line(linewidth = 0.8) + geom_point(size = 2) +
  geom_text(data = \(d) filter(d, k == 8), aes(label = str_wrap(name, 22)), hjust = 0, nudge_x = 0.15,
            size = 3, lineheight = 0.9, show.legend = FALSE) +
  scale_colour_provision(guide = "none") +
  scale_x_continuous(breaks = 1:8, expand = expansion(mult = c(0.03, 0.32))) +
  scale_y_continuous(labels = scales::percent, limits = c(0, 1)) +
  labs(title = "The opening words are informative, but only when they recur",
       subtitle = "Label purity vs coverage of the first k words of a provision (training split).",
       x = "Opening length k (words)", y = NULL) +
  theme_provision(grid = "xy")
save_figure(p_head, file.path(paths$figures, "06_head_purity.png"), width = 7.5, height = 4.5)

# -- 4. Document-frequency profile of the word block --------------------------
it <- text2vec::itoken(normalize_text(train$text, lowercase = TRUE, mask_numbers = TRUE),
                       tokenizer = text2vec::word_tokenizer, progressbar = FALSE)
v <- text2vec::create_vocabulary(it, ngram = c(1L, 2L))
df_profile <- tibble(min_doc_count = c(1, 2, 3, 5, 10, 20, 50)) |>
  mutate(features = map_int(min_doc_count, \(m) sum(v$doc_count >= m)),
         token_coverage = map_dbl(min_doc_count, \(m) sum(v$term_count[v$doc_count >= m]) / sum(v$term_count)))
write_result(df_profile, "06_min_doc_count_profile")
print(df_profile)

p_df <- ggplot(df_profile, aes(features, token_coverage)) +
  geom_line(colour = blue, linewidth = 0.8) + geom_point(colour = blue, size = 2.2) +
  geom_text(aes(label = paste0("min df ", min_doc_count)), vjust = 1.8, hjust = 0.2, size = 3, colour = col$ink2) +
  scale_x_log10(labels = scales::comma, expand = expansion(mult = c(0.08, 0.12))) +
  scale_y_continuous(labels = scales::percent) +
  labs(title = "Pruning rare n-grams removes most features but few tokens",
       subtitle = "Word 1-2-gram vocabulary size vs share of token occurrences kept, by minimum document frequency.",
       x = "Features kept (log scale)", y = "Token occurrences covered") +
  theme_provision(grid = "xy")
save_figure(p_df, file.path(paths$figures, "06_min_doc_count.png"), width = 7.5, height = 4.5)
