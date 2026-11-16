lv <- c("a", "b", "c")
truth <- factor(c("a", "a", "a", "b", "b", "c"), levels = lv)
est <- factor(c("a", "a", "b", "b", "b", "b"), levels = lv)

test_that("per-class metrics match hand calculation", {
  pc <- per_class_metrics(truth, est)
  expect_equal(pc$support, c(3, 2, 1))
  expect_equal(pc$precision, c(1, 0.5, 0))
  expect_equal(pc$recall, c(2 / 3, 1, 0))
  expect_equal(pc$f1, c(0.8, 2 / 3, 0))
})

test_that("headline metrics match hand calculation", {
  m <- classification_metrics(truth, est)
  expect_equal(m$accuracy, 4 / 6)
  expect_equal(m$macro_f1, mean(c(0.8, 2 / 3, 0)))
  expect_equal(m$weighted_f1, (3 * 0.8 + 2 * 2 / 3) / 6)
  expect_equal(m$balanced_accuracy, mean(c(2 / 3, 1, 0)))
})

test_that("classes absent from both truth and estimate do not dilute macro-F1", {
  t2 <- factor(c("a", "b"), levels = c("a", "b", "unused"))
  expect_equal(classification_metrics(t2, t2)$macro_f1, 1)
})

test_that("inputs are validated", {
  expect_error(classification_metrics(truth, factor(est, levels = rev(lv))), "identical levels")
  expect_error(classification_metrics(truth, est[-1]), "same length")
})

test_that("top-k accuracy is monotone and exact at k = 1", {
  scores <- rbind(c(0.7, 0.2, 0.1), c(0.1, 0.5, 0.4), c(0.3, 0.4, 0.2))
  colnames(scores) <- lv
  tk <- top_k_accuracy(factor(c("a", "c", "a"), levels = lv), scores, k = 1:3)
  expect_equal(tk$accuracy, c(1 / 3, 1, 1))
})

test_that("confusion_pairs lists only errors", {
  cp <- confusion_pairs(truth, est)
  expect_equal(sum(cp$n), sum(truth != est))
  expect_equal(cp$share_of_truth[cp$truth == "c"], 1)
})

test_that("bootstrap intervals bracket the point estimate and are reproducible", {
  set.seed(1)
  big_truth <- factor(sample(lv, 400, replace = TRUE), levels = lv)
  big_est <- big_truth
  flip <- sample(400, 80)
  big_est[flip] <- sample(lv, 80, replace = TRUE)
  b1 <- bootstrap_metrics(big_truth, big_est, n_boot = 200, seed = 42)
  b2 <- bootstrap_metrics(big_truth, big_est, n_boot = 200, seed = 42)
  expect_equal(b1, b2, ignore_attr = TRUE)
  expect_true(all(b1$lower <= b1$estimate & b1$estimate <= b1$upper))
  acc <- b1[b1$metric == "accuracy", ]
  expect_equal(acc$std_error, sqrt(acc$estimate * (1 - acc$estimate) / 400), tolerance = 0.25)
})
