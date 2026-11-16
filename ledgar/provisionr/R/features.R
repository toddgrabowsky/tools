#' Specify a sparse text feature set
#'
#' A feature specification is a plain list describing which feature blocks
#' to build. It carries no data: learning the vocabulary and IDF weights is
#' done by [fit_featurizer()] on training texts only, which is what keeps
#' resampling leakage-free.
#'
#' Blocks (each is L2-normalised per document, multiplied by its weight, and
#' the blocks are then column-bound):
#'
#' * **word** -- TF-IDF over word n-grams of the normalised text.
#' * **char** -- TF-IDF over character n-grams; robust to inflection,
#'   typos and tokenisation quirks.
#' * **head** -- TF-IDF over word n-grams of the first `head_words` words
#'   only. Contract provisions tend to announce their type in the opening
#'   clause ("This Agreement shall be governed by ...").
#' * **stats** -- a handful of dense, standardised surface statistics
#'   (see [text_stats()]).
#'
#' @param word_ngram,char_ngram,head_ngram Integer `c(min, max)` n-gram
#'   range for each block, or `NULL` to leave the block out.
#' @param head_words Number of leading words used by the head block.
#' @param stats Include the [text_stats()] block?
#' @param min_doc_count Drop n-grams seen in fewer training documents.
#' @param max_doc_prop Drop n-grams seen in more than this share of
#'   training documents.
#' @param max_features Cap on the vocabulary size of each block (most
#'   frequent kept).
#' @param sublinear_tf Use `1 + log(tf)` instead of raw term frequency?
#' @param use_idf Weight by smoothed inverse document frequency?
#' @param lowercase,mask_numbers Passed to [normalize_text()].
#' @param weights Named numeric vector of block weights.
#' @return An object of class `feature_spec`.
#' @examples
#' feature_spec()
#' feature_spec(char_ngram = c(3, 5), stats = TRUE, weights = c(char = 0.5))
#' @export
feature_spec <- function(word_ngram = c(1L, 2L), char_ngram = NULL, head_ngram = NULL,
                         head_words = 12L, stats = FALSE,
                         min_doc_count = 2L, max_doc_prop = 1, max_features = Inf,
                         sublinear_tf = TRUE, use_idf = TRUE,
                         lowercase = TRUE, mask_numbers = TRUE,
                         weights = c(word = 1, char = 1, head = 1, stats = 1)) {
  default_w <- c(word = 1, char = 1, head = 1, stats = 1)
  if (length(setdiff(names(weights), names(default_w))) > 0) {
    cli::cli_abort("{.arg weights} must be named with {.val {names(default_w)}}.")
  }
  default_w[names(weights)] <- weights
  check_ngram <- function(x, arg) {
    if (is.null(x)) {
      return(NULL)
    }
    if (length(x) != 2 || x[1] < 1 || x[2] < x[1]) {
      cli::cli_abort("{.arg {arg}} must be {.code c(min, max)} with 1 <= min <= max.")
    }
    as.integer(x)
  }
  spec <- list(
    word_ngram = check_ngram(word_ngram, "word_ngram"),
    char_ngram = check_ngram(char_ngram, "char_ngram"),
    head_ngram = check_ngram(head_ngram, "head_ngram"),
    head_words = as.integer(head_words), stats = isTRUE(stats),
    min_doc_count = as.integer(min_doc_count), max_doc_prop = max_doc_prop,
    max_features = max_features, sublinear_tf = sublinear_tf, use_idf = use_idf,
    lowercase = lowercase, mask_numbers = mask_numbers, weights = default_w
  )
  if (is.null(spec$word_ngram) && is.null(spec$char_ngram) && is.null(spec$head_ngram) && !spec$stats) {
    cli::cli_abort("A feature spec needs at least one block.")
  }
  structure(spec, class = "feature_spec")
}

#' @export
print.feature_spec <- function(x, ...) {
  rng <- function(r) if (is.null(r)) "-" else paste(r, collapse = "-")
  cli::cli_text("<feature_spec> word {rng(x$word_ngram)} | char {rng(x$char_ngram)} | ",
                "head {rng(x$head_ngram)} ({x$head_words} words) | stats {x$stats}")
  cli::cli_text("min_doc_count {x$min_doc_count}, max_doc_prop {x$max_doc_prop}, ",
                "max_features {x$max_features}, sublinear_tf {x$sublinear_tf}, idf {x$use_idf}, ",
                "lowercase {x$lowercase}, mask_numbers {x$mask_numbers}")
  invisible(x)
}

#' Surface statistics of a text
#'
#' Dense, hand-crafted descriptors that the bag-of-n-grams cannot see:
#' length, punctuation density, capitalisation and quoting habits.
#'
#' @param x Character vector (raw, un-normalised text).
#' @return A numeric matrix with one row per text and named columns.
#' @examples
#' text_stats(c("\"Affiliate\" means any Person.", "THIS AGREEMENT IS GOVERNED BY NEW YORK LAW"))
#' @export
text_stats <- function(x) {
  n_char <- pmax(nchar(x), 1L)
  n_word <- pmax(stringr::str_count(x, "\\S+"), 1L)
  n_alpha <- pmax(stringr::str_count(x, "[A-Za-z]"), 1L)
  cbind(
    log_words = log(n_word),
    mean_word_len = n_char / n_word,
    digit_share = stringr::str_count(x, "[0-9]") / n_char,
    upper_share = stringr::str_count(x, "[A-Z]") / n_alpha,
    cap_word_share = stringr::str_count(x, "\\b[A-Z][a-z]+") / n_word,
    comma_rate = stringr::str_count(x, ",") / n_word,
    semicolon_rate = stringr::str_count(x, ";") / n_word,
    paren_rate = stringr::str_count(x, "\\(") / n_word,
    quote_rate = stringr::str_count(x, "[\"\\u201c\\u201d]") / n_word,
    sentence_rate = stringr::str_count(x, "[.!?](\\s|$)") / n_word,
    has_currency = as.numeric(stringr::str_detect(x, "\\$")),
    has_percent = as.numeric(stringr::str_detect(x, "%"))
  )
}

# -- internal helpers -------------------------------------------------------

l2_normalize <- function(x) {
  norms <- sqrt(Matrix::rowSums(x^2))
  norms[norms == 0] <- 1
  Matrix::Diagonal(x = 1 / norms) %*% x
}

first_words <- function(x, n) {
  stringr::str_extract(x, sprintf("^(?:\\S+\\s*){1,%d}", n)) |>
    tidyr::replace_na("") |>
    stringr::str_trim()
}

block_iterator <- function(text, block) {
  tokenizer <- if (block == "char") text2vec::char_tokenizer else text2vec::word_tokenizer
  text2vec::itoken(text, tokenizer = tokenizer, progressbar = FALSE)
}

fit_tfidf_block <- function(text, block, ngram, spec) {
  # text2vec needs a one-character separator; for character n-grams use a
  # control character that cannot occur in normalised text and strip it later.
  sep <- if (block == "char") "\001" else "_"
  vocab <- text2vec::create_vocabulary(block_iterator(text, block), ngram = ngram, sep_ngram = sep) |>
    text2vec::prune_vocabulary(doc_count_min = spec$min_doc_count,
                               doc_proportion_max = spec$max_doc_prop,
                               vocab_term_max = spec$max_features)
  n_docs <- length(text)
  idf <- if (spec$use_idf) log((1 + n_docs) / (1 + vocab$doc_count)) + 1 else rep(1, nrow(vocab))
  list(vocab = vocab, idf = idf, ngram = ngram)
}

bake_tfidf_block <- function(fit, text, block, spec) {
  dtm <- text2vec::create_dtm(block_iterator(text, block), text2vec::vocab_vectorizer(fit$vocab))
  if (spec$sublinear_tf) {
    dtm@x <- 1 + log(dtm@x)
  }
  dtm <- dtm %*% Matrix::Diagonal(x = fit$idf)
  dtm <- l2_normalize(dtm)
  colnames(dtm) <- paste0(substr(block, 1, 1), ":", gsub("\001", "", fit$vocab$term, fixed = TRUE))
  dtm
}

prepare_text <- function(text, spec) {
  normalize_text(text, lowercase = spec$lowercase, mask_numbers = spec$mask_numbers)
}

#' Learn a featurizer from training texts
#'
#' Learns everything data-dependent about a [feature_spec()] -- the pruned
#' vocabularies, IDF weights and the centring / scaling of the statistics
#' block -- from `text`. Call it on the analysis (training) portion of a
#' resample only; apply it to any text with [bake_features()].
#'
#' @param text Character vector of training texts.
#' @param spec A [feature_spec()].
#' @return An object of class `featurizer`.
#' @examples
#' txt <- c("This Agreement is governed by New York law.",
#'          "Governed by the laws of Delaware.",
#'          "Notices shall be in writing.", "All notices must be in writing.")
#' fz <- fit_featurizer(txt, feature_spec(word_ngram = c(1, 1), min_doc_count = 1))
#' dim(bake_features(fz, c("Notices in writing", "unseen words only")))
#' @export
fit_featurizer <- function(text, spec = feature_spec()) {
  stopifnot(is.character(text), inherits(spec, "feature_spec"))
  clean <- prepare_text(text, spec)
  blocks <- list()
  if (!is.null(spec$word_ngram)) {
    blocks$word <- fit_tfidf_block(clean, "word", spec$word_ngram, spec)
  }
  if (!is.null(spec$char_ngram)) {
    blocks$char <- fit_tfidf_block(clean, "char", spec$char_ngram, spec)
  }
  if (!is.null(spec$head_ngram)) {
    blocks$head <- fit_tfidf_block(first_words(clean, spec$head_words), "head", spec$head_ngram, spec)
  }
  if (spec$stats) {
    st <- text_stats(text)
    sds <- apply(st, 2, stats::sd)
    blocks$stats <- list(center = colMeans(st), scale = ifelse(sds > 0, sds, 1))
  }
  structure(list(spec = spec, blocks = blocks, n_train = length(text)), class = "featurizer")
}

#' Apply a fitted featurizer
#'
#' @param featurizer Result of [fit_featurizer()].
#' @param text Character vector to featurise.
#' @return A sparse `dgCMatrix` with one row per text. Column names are
#'   prefixed by block: `w:` word, `c:` char, `h:` head, `s:` stats.
#' @export
bake_features <- function(featurizer, text) {
  stopifnot(inherits(featurizer, "featurizer"), is.character(text))
  spec <- featurizer$spec
  clean <- prepare_text(text, spec)
  parts <- purrr::imap(featurizer$blocks, function(fit, block) {
    m <- switch(block,
      word = ,
      char = bake_tfidf_block(fit, clean, block, spec),
      head = bake_tfidf_block(fit, first_words(clean, spec$head_words), block, spec),
      stats = {
        st <- sweep(sweep(text_stats(text), 2, fit$center), 2, fit$scale, "/")
        # squash outliers, then put the block on the same unit scale as TF-IDF
        st <- tanh(st / 3) / sqrt(ncol(st))
        colnames(st) <- paste0("s:", colnames(st))
        methods::as(methods::as(st, "CsparseMatrix"), "generalMatrix")
      }
    )
    m * spec$weights[[block]]
  })
  out <- do.call(cbind, unname(parts))
  methods::as(out, "CsparseMatrix")
}

#' @export
print.featurizer <- function(x, ...) {
  sizes <- purrr::map_int(x$blocks, \(b) if (is.null(b$vocab)) length(b$center) else nrow(b$vocab))
  cli::cli_text("<featurizer> fitted on {x$n_train} texts; ",
                "{sum(sizes)} features ({paste0(names(sizes), ' ', sizes, collapse = ', ')})")
  invisible(x)
}
