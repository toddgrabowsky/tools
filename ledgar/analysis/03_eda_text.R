# 03 -- EDA: text length, vocabulary, surface artefacts and duplicates.
#
# Text statistics use the training split only. The duplicate / leakage audit
# compares texts across splits but never looks at held-out labels.
#
# Output: figures/03_*.png, results/03_*.csv

source("analysis/_setup.R")
library(tidytext)
col <- provision_colours()
blue <- col$categorical[["blue"]]

ledgar <- ledgar_read(paths$raw)
train <- filter(ledgar, split == "train")

# -- 1. Length ---------------------------------------------------------------
train_len <- train |>
  mutate(n_char = str_length(text), n_word = str_count(text, "\\S+"))

len_summary <- train_len |>
  reframe(stat = c("min", "q05", "q25", "median", "mean", "q75", "q95", "q99", "max"),
          words = c(min(n_word), quantile(n_word, c(.05, .25, .5)), mean(n_word),
                    quantile(n_word, c(.75, .95, .99)), max(n_word)))
write_result(len_summary, "03_length_summary")
print(deframe(len_summary) |> round(1))

p_len <- ggplot(train_len, aes(n_word)) +
  geom_histogram(bins = 60, fill = blue, colour = col$surface, linewidth = 0.2) +
  geom_vline(xintercept = median(train_len$n_word), colour = col$ink2, linetype = "dashed") +
  annotate("text", x = median(train_len$n_word), y = Inf, vjust = 1.5, hjust = -0.08, size = 3.2,
           colour = col$ink2, label = glue::glue("median {median(train_len$n_word)} words")) +
  scale_x_log10(labels = scales::comma) +
  scale_y_continuous(labels = scales::comma, expand = expansion(mult = c(0, 0.05))) +
  labs(title = "Provisions are short, with a long right tail",
       subtitle = "Words per training provision (log scale).",
       x = "Words per provision", y = "Provisions")
save_figure(p_len, file.path(paths$figures, "03_length_distribution.png"), width = 7, height = 4.2)

# Length differs systematically by class -> length is itself a weak feature.
len_by_class <- train_len |>
  summarise(median_words = median(n_word), q25 = quantile(n_word, .25),
            q75 = quantile(n_word, .75), n = n(), .by = label) |>
  arrange(median_words)
write_result(len_by_class, "03_length_by_class")
kw <- kruskal.test(n_word ~ label, data = train_len)
# eta-squared from Kruskal-Wallis H: share of rank variance explained by class
eta2 <- unname((kw$statistic - nlevels(train$label) + 1) / (nrow(train) - nlevels(train$label)))
cli::cli_alert_info("Kruskal-Wallis length ~ class: H = {round(kw$statistic)}, eta^2 = {round(eta2, 3)}")

extremes <- c(head(len_by_class$label, 12), tail(len_by_class$label, 12))
p_len_class <- train_len |>
  filter(label %in% extremes) |>
  mutate(label = factor(label, levels = extremes)) |>
  ggplot(aes(n_word, label)) +
  ggridges::geom_density_ridges(fill = blue, colour = col$surface, alpha = 0.75,
                                scale = 1.6, linewidth = 0.3, quantile_lines = TRUE, quantiles = 2) +
  scale_x_log10(labels = scales::comma) +
  labs(title = "Provision length depends strongly on the class",
       subtitle = glue::glue("The 12 longest and 12 shortest classes by median length (white line).\n",
                             "Class explains {scales::percent(eta2, 1)} of the rank variance in length."),
       x = "Words per provision (log scale)", y = NULL) +
  theme_provision(grid = "x")
save_figure(p_len_class, file.path(paths$figures, "03_length_by_class.png"), width = 8, height = 7)

# -- 2. Vocabulary -----------------------------------------------------------
tokens <- train |>
  select(id, label, text) |>
  unnest_tokens(word, text)

vocab <- count(tokens, word, sort = TRUE) |>
  mutate(rank = row_number(), share = n / sum(n))
doc_freq <- tokens |> distinct(id, word) |> count(word, name = "doc_freq")
vocab <- left_join(vocab, doc_freq, by = "word")

vocab_summary <- tibble(
  tokens = nrow(tokens), types = nrow(vocab),
  hapax = sum(vocab$n == 1), hapax_share = mean(vocab$n == 1),
  types_df5 = sum(vocab$doc_freq >= 5),
  numeric_types = sum(str_detect(vocab$word, "^[0-9.,]+$")),
  top100_token_share = sum(vocab$share[1:100])
)
write_result(vocab_summary, "03_vocab_summary")
print(vocab_summary, width = Inf)

zipf_fit <- lm(log10(n) ~ log10(rank), data = filter(vocab, rank >= 10, rank <= 5000))
p_zipf <- ggplot(vocab, aes(rank, n)) +
  geom_line(colour = blue, linewidth = 0.8) +
  geom_abline(intercept = coef(zipf_fit)[1], slope = coef(zipf_fit)[2],
              colour = col$ink2, linetype = "dashed") +
  ggrepel::geom_text_repel(data = \(d) filter(d, rank %in% c(1, 2, 3, 5, 10, 30, 100, 1000, 10000)),
                           aes(label = word), size = 3, colour = col$ink2, nudge_y = 0.25, seed = 1) +
  scale_x_log10(labels = scales::comma) + scale_y_log10(labels = scales::comma) +
  labs(title = "Word frequencies follow Zipf's law",
       subtitle = glue::glue("Rank-frequency plot of {scales::comma(nrow(vocab))} word types; ",
                             "dashed = power-law fit (slope {round(coef(zipf_fit)[2], 2)}) on ranks 10-5,000."),
       x = "Frequency rank", y = "Occurrences") +
  theme_provision(grid = "xy")
save_figure(p_zipf, file.path(paths$figures, "03_zipf.png"), width = 7, height = 4.5)

# Vocabulary growth (Heaps' law): does the vocabulary saturate with 60k docs?
set.seed(SEED)
doc_order <- sample(unique(tokens$id))
growth <- tokens |>
  mutate(doc_pos = match(id, doc_order)) |>
  summarise(first_doc = min(doc_pos), .by = word) |>
  count(first_doc, name = "new_types") |>
  complete(first_doc = seq_along(doc_order), fill = list(new_types = 0)) |>
  mutate(types = cumsum(new_types))
p_heaps <- growth |>
  filter(first_doc %% 250 == 0 | first_doc == 1) |>
  ggplot(aes(first_doc, types)) +
  geom_line(colour = blue, linewidth = 0.8) +
  scale_x_continuous(labels = scales::comma) + scale_y_continuous(labels = scales::comma) +
  labs(title = "The vocabulary keeps growing, but slowly",
       subtitle = glue::glue("Distinct word types seen after reading n provisions (random order). ",
                             "The last 10,000 provisions add {scales::comma(sum(tail(growth$new_types, 10000)))} types."),
       x = "Provisions read", y = "Distinct word types") +
  theme_provision(grid = "xy")
save_figure(p_heaps, file.path(paths$figures, "03_vocab_growth.png"), width = 7, height = 4.2)

# -- 3. Surface artefacts ----------------------------------------------------
artefacts <- train |>
  transmute(
    `contains digits` = str_detect(text, "\\d"),
    `contains currency amount` = str_detect(text, "\\$\\s?\\d"),
    `starts with enumerator, e.g. (a) or 5.1` = str_detect(text, "^\\s*(\\(?[a-zA-Z0-9]{1,3}\\)|\\d+(\\.\\d+)*\\.?\\s)"),
    `contains non-ASCII characters` = str_detect(text, "[^\\x01-\\x7F]"),
    `contains typographic quotes` = str_detect(text, "[\\u2018\\u2019\\u201c\\u201d]"),
    `contains blank placeholder (___ or [ ])` = str_detect(text, "_{2,}|\\[\\s*\\]"),
    `has an ALL-CAPS run of 5+ words` = str_detect(text, "(\\b[A-Z]{2,}\\b[ ,;]+){5,}"),
    `has repeated whitespace` = str_detect(text, "\\s{2,}"),
    `ends without terminal punctuation` = !str_detect(text, "[.;:)\"\\u201d]\\s*$"),
    `shorter than 10 words` = str_count(text, "\\S+") < 10
  ) |>
  summarise(across(everything(), mean)) |>
  pivot_longer(everything(), names_to = "artefact", values_to = "share") |>
  arrange(desc(share))
write_result(artefacts, "03_text_artefacts")

p_art <- artefacts |>
  mutate(artefact = fct_reorder(artefact, share)) |>
  ggplot(aes(share, artefact)) +
  geom_col(fill = blue, width = 0.65) +
  geom_text(aes(label = scales::percent(share, 0.1)), hjust = -0.15, size = 3.2, colour = col$ink2) +
  scale_x_continuous(labels = scales::percent, expand = expansion(mult = c(0, 0.12))) +
  labs(title = "Surface artefacts the cleaning step has to deal with",
       subtitle = "Share of training provisions showing each pattern.", x = NULL, y = NULL) +
  theme_provision(grid = "x")
save_figure(p_art, file.path(paths$figures, "03_text_artefacts.png"), width = 8, height = 4.2)

# -- 4. Duplicates and cross-split leakage -----------------------------------
dup_exact <- flag_duplicates(ledgar$text, method = "exact")
dup_tmpl <- flag_duplicates(ledgar$text, method = "template")

audit <- ledgar |>
  select(id, split, label) |>
  mutate(exact_group = dup_exact$group, tmpl_group = dup_tmpl$group)

# Within train: repeats and label conflicts among template-identical texts.
train_dups <- audit |>
  filter(split == "train") |>
  mutate(group_size = n(), n_labels = n_distinct(label), is_repeat = row_number() > 1, .by = tmpl_group)
# Held-out rows whose template also occurs in train.
train_groups <- unique(audit$tmpl_group[audit$split == "train"])
train_exact <- unique(audit$exact_group[audit$split == "train"])

dup_summary <- bind_rows(
  tibble(check = "train: rows that repeat an earlier exact text",
         n = sum(duplicated(train$text)), of = nrow(train)),
  tibble(check = "train: rows that repeat an earlier template",
         n = sum(train_dups$is_repeat), of = nrow(train)),
  tibble(check = "train: rows in a template group with conflicting labels",
         n = sum(train_dups$n_labels > 1), of = nrow(train)),
  audit |> filter(split != "train") |>
    summarise(n = sum(exact_group %in% train_exact), of = n(), .by = split) |>
    transmute(check = paste0(split, ": rows whose exact text is in train"), n, of),
  audit |> filter(split != "train") |>
    summarise(n = sum(tmpl_group %in% train_groups), of = n(), .by = split) |>
    transmute(check = paste0(split, ": rows whose template is in train"), n, of)
) |> mutate(share = n / of)
write_result(dup_summary, "03_duplicate_summary")
print(dup_summary, width = Inf)

# Examples of label conflicts: same boilerplate, different annotated type.
conflicts <- train_dups |>
  filter(n_labels > 1) |>
  left_join(select(train, id, text), by = "id") |>
  summarise(labels = paste(sort(unique(as.character(label))), collapse = " | "),
            rows = n(), example = str_trunc(first(text), 160), .by = tmpl_group) |>
  count(labels, wt = rows, name = "rows", sort = TRUE)
write_result(conflicts, "03_label_conflict_pairs")
print(head(conflicts, 10), width = Inf)

# Near-duplicates: cosine similarity of every held-out provision to its
# nearest training provision in TF-IDF space.
nn_similarity <- function(query, reference, chunk = 1000L) {
  fz <- fit_featurizer(reference, feature_spec(word_ngram = c(1, 2)))
  ref <- bake_features(fz, reference)
  qry <- bake_features(fz, query)
  idx <- split(seq_len(nrow(qry)), ceiling(seq_len(nrow(qry)) / chunk))
  unlist(lapply(idx, \(i) {
    sims <- as.matrix(Matrix::tcrossprod(qry[i, , drop = FALSE], ref))
    apply(sims, 1, max)
  }), use.names = FALSE)
}

nn <- bind_rows(
  # reference for train itself: 5k random train rows against the other 55k
  local({
    set.seed(SEED); i <- sample(nrow(train), 5000)
    tibble(split = "train (5k sample vs rest of train)",
           sim = nn_similarity(train$text[i], train$text[-i]))
  }),
  tibble(split = "validation vs train",
         sim = nn_similarity(ledgar$text[ledgar$split == "validation"], train$text)),
  tibble(split = "test vs train",
         sim = nn_similarity(ledgar$text[ledgar$split == "test"], train$text))
) |> mutate(split = fct_inorder(split))
saveRDS(nn, file.path(paths$processed, "03_nn_similarity.rds"))

nn_summary <- nn |>
  summarise(median_sim = median(sim), share_ge_0.95 = mean(sim >= 0.95),
            share_ge_0.8 = mean(sim >= 0.8), share_lt_0.5 = mean(sim < 0.5), .by = split)
write_result(nn_summary, "03_nn_similarity_summary")
print(nn_summary, width = Inf)

p_nn <- ggplot(nn, aes(sim)) +
  geom_histogram(binwidth = 0.025, boundary = 0, fill = blue, colour = col$surface, linewidth = 0.2) +
  facet_wrap(vars(split), ncol = 1, scales = "free_y") +
  scale_x_continuous(limits = c(0, 1.0001), breaks = seq(0, 1, 0.2)) +
  scale_y_continuous(labels = scales::comma, expand = expansion(mult = c(0, 0.05))) +
  labs(title = "About one provision in seven has a near-twin in train -- in every split",
       subtitle = "Cosine similarity (TF-IDF, 1-2 grams) to the nearest training provision. Boilerplate is intrinsic\nto the corpus, not an artefact of the split: held-out rows look like train rows.",
       x = "Similarity to nearest training provision", y = "Provisions")
save_figure(p_nn, file.path(paths$figures, "03_nearest_neighbour_similarity.png"), width = 7, height = 6.5)
