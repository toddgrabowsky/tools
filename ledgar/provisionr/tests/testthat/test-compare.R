test_that("compare_predictions counts agreement and tests discordant pairs", {
  truth <- factor(rep(c("x", "y"), each = 50))
  a <- truth
  b <- truth
  a[1:20] <- "y"            # a wrong on 20 rows where b is right
  b[21:25] <- "y"           # b wrong on 5 rows where a is right
  a[26:30] <- "y"; b[26:30] <- "y"   # both wrong on 5 rows
  cmp <- compare_predictions(truth, a, b, n_boot = 300, seed = 1)
  expect_equal(cmp$agreement$n, c(70, 5, 20, 5))
  expect_equal(cmp$mcnemar$p_value, binom.test(20, 25)$p.value)
  acc <- cmp$metrics[cmp$metrics$metric == "accuracy", ]
  expect_equal(acc$difference, 0.15)
  expect_gt(acc$lower, 0)                       # interval excludes zero
  expect_true(acc$lower < acc$difference && acc$difference < acc$upper)
  expect_gt(acc$share_b_better, 0.99)
})

test_that("identical predictions give a zero difference and p = 1", {
  truth <- factor(rep(c("x", "y"), each = 10))
  est <- truth
  est[1:3] <- "y"
  cmp <- compare_predictions(truth, est, est, n_boot = 50, seed = 1)
  expect_true(all(cmp$metrics$difference == 0))
  expect_true(all(cmp$metrics$lower == 0 & cmp$metrics$upper == 0))
  expect_equal(cmp$mcnemar$p_value, 1)
})

test_that("softmax_scores returns probabilities and handles -Inf", {
  s <- rbind(c(2, 1, -Inf), c(0, 0, 5))
  colnames(s) <- c("a", "b", "c")
  p <- softmax_scores(s)
  expect_equal(rowSums(p), c(1, 1))
  expect_equal(p[1, "c"], 0, ignore_attr = TRUE)
  expect_equal(colnames(p), colnames(s))
  # temperature flattens but never reorders
  hot <- softmax_scores(s, temperature = 50)
  expect_equal(max.col(hot), max.col(p))
  expect_lt(max(hot[2, ]), max(p[2, ]))
})

test_that("fit_temperature recovers a known scale", {
  set.seed(3)
  k <- 4; n <- 4000
  logits <- matrix(rnorm(n * k, sd = 2), n, k, dimnames = list(NULL, letters[1:k]))
  truth <- factor(apply(softmax_scores(logits), 1, function(p) sample(letters[1:k], 1, prob = p)),
                  levels = letters[1:k])
  expect_equal(as.numeric(fit_temperature(logits, truth)), 1, tolerance = 0.1)
  expect_equal(as.numeric(fit_temperature(logits * 5, truth)), 5, tolerance = 0.5)
  expect_lt(attr(fit_temperature(logits * 5, truth), "nll"),
            -mean(log(softmax_scores(logits * 5)[cbind(seq_len(n), as.integer(truth))])))
})
