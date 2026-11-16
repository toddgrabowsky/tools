toy_features <- function() {
  d <- toy_corpus()
  fz <- fit_featurizer(d$text, feature_spec(word_ngram = c(1, 1), min_doc_count = 1))
  list(x = bake_features(fz, d$text), y = d$label)
}

test_that("every engine learns a separable toy problem", {
  f <- toy_features()
  engines <- list(nb = list(engine = "nb"), cnb = list(engine = "nb", complement = TRUE),
                  centroid = list(engine = "centroid"), svm = list(engine = "svm"),
                  svm_cs = list(engine = "svm", multiclass = "cs"),
                  svm_bal = list(engine = "svm", balanced = TRUE),
                  logreg = list(engine = "logreg", cost = 10))
  for (name in names(engines)) {
    fit <- do.call(fit_classifier, c(list(x = f$x, y = f$y), engines[[name]]))
    pred <- predict(fit, f$x)
    expect_identical(levels(pred), levels(f$y), info = name)
    expect_gt(mean(pred == f$y), 0.95, label = name)
    scores <- predict(fit, f$x, type = "score")
    expect_equal(dim(scores), c(length(f$y), nlevels(f$y)), info = name)
    expect_identical(colnames(scores), levels(f$y), info = name)
    expect_identical(colnames(scores)[max.col(scores, ties.method = "first")], as.character(pred), info = name)
  }
})

test_that("majority baseline predicts the modal class", {
  f <- toy_features()
  y <- f$y
  y[1:10] <- "tax"
  fit <- fit_classifier(f$x, y, engine = "majority")
  expect_true(all(predict(fit, f$x) == "tax"))
})

test_that("classes absent from training are kept as levels but never predicted", {
  f <- toy_features()
  keep <- f$y != "tax"
  fit <- fit_classifier(f$x[keep, ], f$y[keep], engine = "svm")
  scores <- predict(fit, f$x, type = "score")
  expect_true(all(scores[, "tax"] == -Inf))
  expect_false(any(predict(fit, f$x) == "tax"))
})

test_that("binary problems work with liblinear engines", {
  f <- toy_features()
  keep <- f$y != "tax"
  y <- droplevels(f$y[keep])
  for (engine in c("svm", "logreg")) {
    fit <- fit_classifier(f$x[keep, ], y, engine = engine, cost = 10)
    expect_gt(mean(predict(fit, f$x[keep, ]) == y), 0.95, label = engine)
  }
})

test_that("naive Bayes rejects negative features and mismatched input", {
  f <- toy_features()
  x <- f$x
  x@x[1] <- -1
  expect_error(fit_classifier(x, f$y, engine = "nb"), "non-negative")
  fit <- fit_classifier(f$x, f$y, engine = "nb")
  expect_error(predict(fit, f$x[, -1]))
})
