test_that("feature_spec validates its inputs", {
  expect_s3_class(feature_spec(), "feature_spec")
  expect_error(feature_spec(word_ngram = c(2, 1)), "min <= max")
  expect_error(feature_spec(word_ngram = NULL), "at least one block")
  expect_error(feature_spec(weights = c(bogus = 1)), "weights")
  expect_equal(feature_spec(weights = c(char = 0.5))$weights[["char"]], 0.5)
})

test_that("TF-IDF blocks are unit length and aligned to the training vocabulary", {
  d <- toy_corpus()
  fz <- fit_featurizer(d$text, feature_spec(word_ngram = c(1, 2), min_doc_count = 2))
  x <- bake_features(fz, d$text)
  expect_s4_class(x, "dgCMatrix")
  expect_equal(nrow(x), nrow(d))
  expect_equal(unname(Matrix::rowSums(x^2)), rep(1, nrow(d)), tolerance = 1e-8)
  new <- bake_features(fz, c("notices in writing", "zzz qqq"))
  expect_equal(colnames(new), colnames(x))
  expect_equal(sum(new[2, ]), 0)          # unseen words -> empty row, not an error
})

test_that("the featurizer never learns from the texts it is applied to", {
  d <- toy_corpus()
  fz <- fit_featurizer(d$text[1:60], feature_spec(min_doc_count = 1))
  a <- bake_features(fz, d$text[61:90])
  b <- bake_features(fz, d$text[61:62])
  expect_equal(as.matrix(a[1:2, ]), as.matrix(b))   # row output independent of batch
})

test_that("all blocks combine with weights and prefixes", {
  d <- toy_corpus()
  spec <- feature_spec(word_ngram = c(1, 1), char_ngram = c(2, 3), head_ngram = c(1, 1),
                       head_words = 3, stats = TRUE, min_doc_count = 1,
                       weights = c(char = 0.5))
  x <- bake_features(fit_featurizer(d$text, spec), d$text)
  prefixes <- table(substr(colnames(x), 1, 2))
  expect_setequal(names(prefixes), c("w:", "c:", "h:", "s:"))
  expect_equal(prefixes[["s:"]], ncol(text_stats("a")))
  char_cols <- startsWith(colnames(x), "c:")
  expect_equal(unname(Matrix::rowSums(x[, char_cols]^2)), rep(0.25, nrow(d)), tolerance = 1e-8)
  expect_false(any(grepl("\001", colnames(x), fixed = TRUE)))
})

test_that("text_stats returns finite values for edge cases", {
  st <- text_stats(c("", "A", "100% of $5"))
  expect_true(all(is.finite(st)))
  expect_equal(unname(st[3, "has_currency"]), 1)
  expect_equal(unname(st[3, "has_percent"]), 1)
})
