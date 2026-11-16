# 04 -- EDA: how separable are the classes, and which ones will be confused?
#
# Training split only.
# Output: figures/04_*.png, results/04_*.csv

source("analysis/_setup.R")
library(tidytext)
col <- provision_colours()

train <- ledgar_read(paths$raw, "train")

# -- 1. Most distinctive terms per class (class-level TF-IDF) ----------------
class_terms <- train |>
  mutate(text = normalize_text(text, lowercase = TRUE, mask_numbers = TRUE)) |>
  unnest_tokens(term, text, token = "ngrams", n = 2) |>
  count(label, term) |>
  filter(n >= 5) |>
  bind_tf_idf(term, label, n)

top_terms <- class_terms |>
  slice_max(tf_idf, n = 6, by = label, with_ties = FALSE)
write_result(select(top_terms, label, term, n, tf_idf), "04_distinctive_bigrams")

showcase <- c("Governing Laws", "Jurisdictions", "Venues", "Submission To Jurisdiction",
              "Indemnifications", "Indemnity", "Amendments", "Modifications",
              "Taxes", "Tax Withholdings", "Withholdings", "Entire Agreements")
p_terms <- top_terms |>
  filter(label %in% showcase) |>
  mutate(label = factor(label, levels = showcase),
         term = reorder_within(term, tf_idf, label)) |>
  ggplot(aes(tf_idf, term)) +
  geom_col(fill = col$categorical[["blue"]], width = 0.65) +
  facet_wrap(vars(label), scales = "free", ncol = 4) +
  scale_y_reordered() +
  scale_x_continuous(expand = expansion(mult = c(0, 0.05))) +
  labs(title = "Related classes share their most distinctive phrases",
       subtitle = "Top bigrams by class-level TF-IDF for twelve classes from four families of near-synonymous labels.",
       x = "Class-level TF-IDF", y = NULL) +
  theme_provision(base_size = 9, grid = "x") +
  theme(axis.text.x = element_blank())
save_figure(p_terms, file.path(paths$figures, "04_distinctive_terms.png"), width = 11, height = 6.5)

# -- 2. Class centroid similarity --------------------------------------------
fz <- fit_featurizer(train$text, feature_spec(word_ngram = c(1, 2), min_doc_count = 3))
x <- bake_features(fz, train$text)
indicator <- Matrix::sparse.model.matrix(~ 0 + label, data = train)
centroids <- Matrix::t(indicator) %*% x
centroids <- as.matrix(centroids / sqrt(Matrix::rowSums(centroids^2)))
rownames(centroids) <- levels(train$label)
sim <- tcrossprod(centroids)

pairs <- as_tibble(sim, rownames = "class_a") |>
  pivot_longer(-class_a, names_to = "class_b", values_to = "similarity") |>
  filter(class_a < class_b) |>
  arrange(desc(similarity))
write_result(pairs, "04_centroid_similarity_pairs")
print(head(pairs, 15))

ord <- hclust(as.dist(1 - sim), method = "average")$order
lvl <- rownames(sim)[ord]
p_heat <- as_tibble(sim, rownames = "a") |>
  pivot_longer(-a, names_to = "b", values_to = "similarity") |>
  mutate(a = factor(a, lvl), b = factor(b, rev(lvl))) |>
  ggplot(aes(a, b, fill = similarity)) +
  geom_raster() +
  scale_fill_gradientn(colours = c(col$surface, col$sequential), limits = c(0, 1),
                       name = "Cosine similarity of class centroids") +
  coord_equal() +
  labs(title = "Most classes are distinct; a few tight clusters are not",
       subtitle = "TF-IDF centroid similarity between all 100 classes, ordered by hierarchical clustering.",
       x = NULL, y = NULL) +
  theme_provision(base_size = 8, grid = "none") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5, size = 4.6),
        axis.text.y = element_text(size = 4.6), axis.line = element_blank(),
        legend.key.width = unit(1.6, "cm"), legend.key.height = unit(0.25, "cm"))
save_figure(p_heat, file.path(paths$figures, "04_centroid_similarity_heatmap.png"), width = 10, height = 10.5)

p_pairs <- pairs |>
  slice_head(n = 20) |>
  mutate(pair = fct_reorder(paste(class_a, class_b, sep = "  /  "), similarity)) |>
  ggplot(aes(similarity, pair)) +
  geom_segment(aes(x = 0.5, xend = similarity, yend = pair), colour = col$categorical[["blue"]], linewidth = 0.6) +
  geom_point(colour = col$categorical[["blue"]], size = 2.2) +
  geom_text(aes(label = sprintf("%.2f", similarity)), hjust = -0.4, size = 3, colour = col$ink2) +
  scale_x_continuous(limits = c(0.5, 1.03), expand = expansion(0)) +
  labs(title = "The label pairs most likely to be confused",
       subtitle = "20 most similar class centroids. Several are near-synonyms in the label scheme itself.",
       x = "Cosine similarity of class centroids", y = NULL) +
  theme_provision(grid = "x")
save_figure(p_pairs, file.path(paths$figures, "04_confusable_pairs.png"), width = 8, height = 5.5)

# How isolated is each class? Similarity to its nearest other class.
nearest <- pairs |>
  pivot_longer(c(class_a, class_b), values_to = "label") |>
  summarise(nearest_similarity = max(similarity), .by = label) |>
  left_join(count(train, label), by = "label")
write_result(nearest, "04_class_isolation")

# -- 3. Low-dimensional map ---------------------------------------------------
set.seed(SEED)
idx <- sample(nrow(train), 15000)
svd <- irlba::irlba(x[idx, ], nv = 50)
lsa <- svd$u %*% diag(svd$d)
emb <- uwot::umap(lsa, n_neighbors = 30, min_dist = 0.15, metric = "cosine", n_threads = 4, verbose = FALSE)
map_df <- tibble(x = emb[, 1], y = emb[, 2], label = train$label[idx])

families <- list(
  "Jurisdiction family" = c("Governing Laws", "Submission To Jurisdiction", "Waiver Of Jury Trials"),
  "Change family" = c("Amendments", "Waivers", "Modifications"),
  "Tax family" = c("Taxes", "Tax Withholdings", "Withholdings"),
  "Well separated" = c("Counterparts", "Notices", "Base Salary")
)
fam_df <- imap_dfr(families, \(labs, fam) {
  map_df |> mutate(family = fam,
                   group = if_else(label %in% labs, as.character(label), NA_character_),
                   slot = match(group, labs))
}) |> mutate(family = fct_inorder(family))
label_pos <- fam_df |> filter(!is.na(group)) |>
  summarise(x = median(x), y = median(y), .by = c(family, group, slot))

p_map <- ggplot(fam_df, aes(x, y)) +
  geom_point(data = \(d) filter(d, is.na(group)), colour = col$grid, size = 0.25) +
  geom_point(data = \(d) filter(d, !is.na(group)), aes(colour = factor(slot)), size = 0.6, alpha = 0.8) +
  ggrepel::geom_label_repel(data = label_pos, aes(label = group, colour = factor(slot)), size = 3,
                            fill = alpha(col$surface, 0.85), label.size = 0, seed = 2, show.legend = FALSE,
                            min.segment.length = 0) +
  facet_wrap(vars(family)) +
  scale_colour_provision(guide = "none") +
  labs(title = "Boilerplate classes form islands; near-synonymous classes overlap",
       subtitle = "UMAP of 15,000 training provisions (TF-IDF -> 50-d LSA). Grey: all other classes.",
       x = NULL, y = NULL) +
  theme_provision(grid = "none") +
  theme(axis.text = element_blank(), axis.line = element_blank())
save_figure(p_map, file.path(paths$figures, "04_umap_families.png"), width = 9, height = 8)
