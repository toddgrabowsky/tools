# 05 -- Cleaning: turn the EDA findings into a reproducible cleaning step.
#
# Decisions (see results/03_*.csv for the evidence):
#   * repair typographic quotes / dashes / odd whitespace everywhere;
#   * train only: drop < 4-word fragments, template duplicates with
#     conflicting labels, and repeats of an earlier template;
#   * validation / test: text is repaired but no row is ever dropped. Each
#     row is annotated with whether its template was seen in train, so
#     performance can later be reported separately on novel text.
#
# Output: data/processed/ledgar_clean.rds, results/05_*.csv, figures/05_*.png

source("analysis/_setup.R")
col <- provision_colours()

ledgar <- ledgar_read(paths$raw)

train_raw <- filter(ledgar, split == "train")
train <- clean_corpus(train_raw)
cleaning_log <- attr(train, "cleaning_log")
write_result(cleaning_log, "05_cleaning_log")
print(cleaning_log)

# Which classes lose the most rows?
class_effect <- count(train_raw, label, name = "before") |>
  left_join(count(train, label, name = "after"), by = "label") |>
  mutate(removed = before - after, removed_share = removed / before) |>
  arrange(desc(removed_share))
write_result(class_effect, "05_cleaning_by_class")
stopifnot(all(class_effect$after >= 20))  # no class is cleaned out of existence

held_out <- ledgar |>
  filter(split != "train") |>
  mutate(text = normalize_text(text),
         template_in_train = text_fingerprint(text) %in% text_fingerprint(train_raw$text),
         # template groups for grouped resampling are computed downstream
         n_words = str_count(text, "\\S+"))

clean <- bind_rows(
  train |> mutate(template_in_train = NA, n_words = str_count(text, "\\S+")),
  held_out
)
saveRDS(clean, file.path(paths$processed, "ledgar_clean.rds"))
cli::cli_alert_success("Saved cleaned corpus: {nrow(clean)} rows")

# Effect of text normalisation on the vocabulary the models will see.
vocab_size <- function(text) {
  it <- text2vec::itoken(text, tokenizer = text2vec::word_tokenizer, progressbar = FALSE)
  v <- text2vec::create_vocabulary(it)
  tibble(types = nrow(v), types_in_2plus_docs = sum(v$doc_count >= 2))
}
vocab_effect <- bind_rows(
  `raw text` = vocab_size(train_raw$text),
  `repaired` = vocab_size(normalize_text(train_raw$text)),
  `+ lower-cased` = vocab_size(normalize_text(train_raw$text, lowercase = TRUE)),
  `+ digits masked` = vocab_size(normalize_text(train_raw$text, lowercase = TRUE, mask_numbers = TRUE)),
  .id = "normalisation"
)
write_result(vocab_effect, "05_vocab_effect")
print(vocab_effect)

# -- Figures ------------------------------------------------------------------
p_log <- cleaning_log |>
  filter(step != "input") |>
  mutate(step = fct_rev(fct_inorder(step))) |>
  ggplot(aes(rows_removed, step)) +
  geom_col(fill = col$categorical[["blue"]], width = 0.6) +
  geom_text(aes(label = glue::glue("{scales::comma(rows_removed)} ({scales::percent(rows_removed / nrow(train_raw), 0.01)})")),
            hjust = -0.1, size = 3.3, colour = col$ink2) +
  scale_x_continuous(labels = scales::comma, expand = expansion(mult = c(0, 0.3))) +
  labs(title = glue::glue("Cleaning removes {scales::percent(1 - nrow(train) / nrow(train_raw), 0.1)} of the training rows"),
       subtitle = glue::glue("{scales::comma(nrow(train_raw))} -> {scales::comma(nrow(train))} provisions. ",
                             "Held-out splits are never filtered."),
       x = "Training rows removed", y = NULL) +
  theme_provision(grid = "x")

p_vocab <- vocab_effect |>
  pivot_longer(-normalisation, names_to = "measure", values_to = "types") |>
  mutate(normalisation = fct_inorder(normalisation),
         measure = recode(measure, types = "all word types", types_in_2plus_docs = "types in 2+ provisions")) |>
  ggplot(aes(normalisation, types, fill = measure)) +
  geom_col(position = position_dodge(width = 0.7), width = 0.62) +
  geom_text(aes(label = scales::comma(types)), position = position_dodge(width = 0.7),
            vjust = -0.5, size = 3, colour = col$ink2) +
  scale_fill_provision(name = NULL) +
  scale_y_continuous(labels = scales::comma, expand = expansion(mult = c(0, 0.1))) +
  labs(title = glue::glue("Lower-casing and digit masking shrink the vocabulary by ",
                          "{scales::percent(1 - last(vocab_effect$types) / first(vocab_effect$types), 1)}"),
       subtitle = "Distinct word types in the training split after each cumulative normalisation step.",
       x = NULL, y = "Word types")

save_figure(patchwork::wrap_plots(p_log, p_vocab, ncol = 1, heights = c(1, 1.6)),
            file.path(paths$figures, "05_cleaning_effects.png"), width = 8, height = 7.5)
