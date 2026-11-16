# A tiny synthetic corpus with three clearly separable classes.
toy_corpus <- function(n_per_class = 30, seed = 1) {
  set.seed(seed)
  vocab <- list(
    law = c("governed", "laws", "state", "york", "delaware", "jurisdiction"),
    notice = c("notices", "writing", "delivered", "mail", "address", "courier"),
    tax = c("taxes", "withholding", "payable", "income", "liable", "authority")
  )
  filler <- c("the", "of", "and", "this", "agreement", "shall", "be", "party")
  make <- function(words) {
    paste(sample(c(sample(words, 5, replace = TRUE), sample(filler, 6, replace = TRUE))), collapse = " ")
  }
  tibble::tibble(
    label = factor(rep(names(vocab), each = n_per_class)),
    text = unlist(lapply(vocab, function(w) replicate(n_per_class, make(w))))
  )
}
