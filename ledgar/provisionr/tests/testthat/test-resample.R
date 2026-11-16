test_that("folds are stratified even for rare classes", {
  y <- factor(rep(c("a", "b", "c"), c(100, 10, 5)))
  folds <- assign_folds(y, v = 5, seed = 1)
  tab <- table(y, folds)
  expect_equal(ncol(tab), 5)
  expect_true(all(tab["a", ] == 20))
  expect_true(all(tab["b", ] == 2))
  expect_true(all(tab["c", ] == 1))
  expect_identical(folds, assign_folds(y, v = 5, seed = 1))
})

test_that("groups are never split across folds", {
  set.seed(2)
  y <- factor(sample(c("a", "b"), 200, replace = TRUE))
  groups <- sample(1:40, 200, replace = TRUE)
  folds <- assign_folds(y, v = 4, groups = groups, seed = 3)
  expect_true(all(tapply(folds, groups, function(f) length(unique(f))) == 1))
  expect_setequal(unique(folds), 1:4)
})

test_that("cv_classifier returns out-of-fold predictions for every row", {
  d <- toy_corpus()
  folds <- assign_folds(d$label, v = 3, seed = 1)
  cv <- cv_classifier(d$text, d$label, folds, feature_spec(word_ngram = c(1, 1), min_doc_count = 1),
                      models = list(nb = list(engine = "nb"), svm = list(engine = "svm")))
  expect_equal(nrow(cv$metrics), 6)
  expect_equal(sort(cv$predictions$row[cv$predictions$model == "svm"]), seq_len(nrow(d)))
  expect_true(all(cv$metrics$accuracy > 0.9))
  expect_true(all(cv$predictions$margin >= 0))
  s <- summarise_cv(cv$metrics)
  expect_equal(s$folds, c(3, 3))
  expect_named(s, c("model", "folds", "accuracy_mean", "accuracy_se", "macro_f1_mean", "macro_f1_se",
                    "weighted_f1_mean", "weighted_f1_se", "balanced_accuracy_mean",
                    "balanced_accuracy_se", "fit_seconds"))
})

test_that("the prepare hook only touches analysis rows", {
  d <- toy_corpus()
  folds <- assign_folds(d$label, v = 3, seed = 1)
  seen <- integer()
  prep <- function(text, y) {
    seen <<- c(seen, length(text))
    list(text = text[-1], y = y[-1])
  }
  cv <- cv_classifier(d$text, d$label, folds, feature_spec(word_ngram = c(1, 1), min_doc_count = 1),
                      models = list(nb = list(engine = "nb")), prepare = prep)
  expect_equal(seen, as.integer(nrow(d) - table(folds)))
  expect_equal(nrow(cv$predictions), nrow(d))
})

test_that("target shuffling separates signal from chance", {
  d <- toy_corpus()
  folds <- assign_folds(d$label, v = 3, seed = 1)
  ts <- target_shuffle(d$text, d$label, folds, feature_spec(word_ngram = c(1, 1), min_doc_count = 1),
                       model = list(engine = "nb"), n_perm = 19, seed = 1)
  expect_equal(nrow(ts$null), 19)
  expect_gt(ts$observed$accuracy, 0.9)
  expect_lt(mean(ts$null$accuracy), 0.5)
  expect_equal(ts$p_value[["accuracy"]], 1 / 20)
  expect_s3_class(plot_target_shuffle(ts), "ggplot")

  # labels that are pure noise must NOT look significant
  set.seed(5)
  noise <- sample(d$label)
  ts0 <- target_shuffle(d$text, noise, folds, feature_spec(word_ngram = c(1, 1), min_doc_count = 1),
                        model = list(engine = "nb"), n_perm = 19, seed = 1)
  expect_gt(ts0$p_value[["accuracy"]], 0.05)
})
