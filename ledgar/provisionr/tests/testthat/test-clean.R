test_that("normalize_text repairs typography and whitespace", {
  x <- "  “Base Salary” –  it’s ____ here  "
  expect_equal(normalize_text(x), "\"Base Salary\" - it's _ here")
  expect_equal(normalize_text("ABC Def", lowercase = TRUE), "abc def")
  expect_equal(normalize_text("pay $1,250.50 on 12/31/2020", mask_numbers = TRUE), "pay $0 on 0/0/0")
  expect_identical(normalize_text(character()), character())
})

test_that("fingerprints ignore numbers, case and punctuation", {
  a <- text_fingerprint("Section 4.1  Governing Law.")
  b <- text_fingerprint("SECTION 12.7 governing law")
  expect_identical(a, b)
  expect_false(identical(a, text_fingerprint("Section 4.1 Governing Laws")))
})

test_that("flag_duplicates finds repeats and label conflicts", {
  out <- flag_duplicates(c("Sec. 1 Notices.", "Sec. 2 notices", "Taxes", "Sec. 3 Notices"),
                         label = c("a", "a", "b", "c"))
  expect_equal(out$group, c(1, 1, 2, 1))
  expect_equal(out$is_repeat, c(FALSE, TRUE, FALSE, TRUE))
  expect_equal(out$label_conflict, c(TRUE, TRUE, FALSE, TRUE))
  exact <- flag_duplicates(c("a", "A", "a"), method = "exact")
  expect_equal(exact$group_size, c(2, 1, 2))
})

test_that("clean_corpus drops the right rows and logs each step", {
  d <- data.frame(
    text = c("Sec. 1 Notices must be given in writing.", "Sec. 9 Notices must be given in writing.",
             "See above.", "Taxes are borne by the Seller alone.",
             "Governing law is New York law.", "Governing law is new york law"),
    label = c("Notices", "Notices", "Taxes", "Taxes", "Law", "Venue")
  )
  out <- clean_corpus(d, min_words = 3)
  log <- attr(out, "cleaning_log")
  expect_equal(out$label, c("Notices", "Taxes"))
  expect_equal(log$rows_removed, c(0, 1, 2, 1))
  expect_equal(log$rows_left[nrow(log)], nrow(out))
  keep_all <- clean_corpus(d, min_words = 1, drop_conflicts = FALSE, drop_repeats = FALSE)
  expect_equal(nrow(keep_all), nrow(d))
})
