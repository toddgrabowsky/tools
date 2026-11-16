#' Fit a classifier on a sparse feature matrix
#'
#' One calling convention for several fast text classifiers, so that
#' resampling code never needs to know which model it is running.
#'
#' Engines:
#'
#' * `"majority"` -- always predicts the most frequent class (the floor).
#' * `"nb"` -- multinomial naive Bayes with Laplace/Lidstone smoothing
#'   `alpha`; `complement = TRUE` gives complement NB (Rennie et al., 2003),
#'   which is more robust to class imbalance.
#' * `"centroid"` -- nearest class centroid by cosine similarity (Rocchio).
#' * `"svm"` -- one-vs-rest linear SVM (L2-regularised, squared hinge loss)
#'   via 'LiblineaR'; `cost` is the usual C; `multiclass = "cs"` switches to
#'   the joint Crammer-Singer formulation.
#' * `"logreg"` -- one-vs-rest L2-regularised logistic regression via
#'   'LiblineaR'; `cost` as above.
#' * `"xgb"` -- gradient-boosted trees ('xgboost', multi-class soft-max),
#'   intended for low-dimensional dense inputs such as LSA components.
#'
#' @param x Feature matrix (`dgCMatrix` or dense), one row per example.
#' @param y Factor of class labels; unused levels are kept so that
#'   predictions always share the full label set.
#' @param engine Which model to fit.
#' @param ... Engine-specific arguments (see above). `balanced = TRUE`
#'   (svm, logreg) weights classes inversely to their frequency.
#' @return An object of class `provision_classifier`; see
#'   [predict.provision_classifier()].
#' @examples
#' set.seed(1)
#' x <- Matrix::rsparsematrix(60, 20, 0.3, rand.x = function(n) abs(rnorm(n)))
#' y <- factor(rep(c("a", "b", "c"), 20))
#' x[y == "a", 1] <- 3; x[y == "b", 2] <- 3; x[y == "c", 3] <- 3
#' fit <- fit_classifier(x, y, engine = "nb")
#' mean(predict(fit, x) == y)
#' @export
fit_classifier <- function(x, y, engine = c("svm", "logreg", "nb", "centroid", "majority", "xgb"), ...) {
  engine <- rlang::arg_match(engine)
  stopifnot(is.factor(y), nrow(x) == length(y))
  fit <- switch(engine,
    majority = fit_majority(x, y, ...),
    nb = fit_nb(x, y, ...),
    centroid = fit_centroid(x, y, ...),
    svm = fit_liblinear(x, y, loss = "svm", ...),
    logreg = fit_liblinear(x, y, loss = "logreg", ...),
    xgb = fit_xgb(x, y, ...)
  )
  structure(c(fit, list(engine = engine, levels = levels(y), n_features = ncol(x))),
            class = c(paste0("clf_", engine), "provision_classifier"))
}

#' Predict with a fitted classifier
#'
#' @param object A `provision_classifier`.
#' @param newdata Feature matrix with the same columns as the training data.
#' @param type `"class"` for a factor of predicted labels, `"score"` for an
#'   n x K matrix of per-class scores (larger = more likely; log-posteriors
#'   for NB, decision values for SVM, probabilities for logreg / xgb,
#'   cosine similarities for centroid). Classes never seen in training get
#'   `-Inf`.
#' @param ... Unused.
#' @return A factor or a numeric matrix with class names as column names.
#' @export
predict.provision_classifier <- function(object, newdata, type = c("class", "score"), ...) {
  type <- rlang::arg_match(type)
  stopifnot(ncol(newdata) == object$n_features)
  scores <- as.matrix(predict_scores(object, newdata))
  # align to the full label set
  full <- matrix(-Inf, nrow(scores), length(object$levels), dimnames = list(NULL, object$levels))
  full[, colnames(scores)] <- scores
  if (type == "score") {
    return(full)
  }
  factor(object$levels[max.col(full, ties.method = "first")], levels = object$levels)
}

#' @export
print.provision_classifier <- function(x, ...) {
  cli::cli_text("<provision_classifier> engine {.val {x$engine}}, ",
                "{length(x$levels)} classes, {x$n_features} features")
  invisible(x)
}

predict_scores <- function(object, newdata) UseMethod("predict_scores")

# -- majority ---------------------------------------------------------------

fit_majority <- function(x, y) {
  list(prior = prop.table(table(y)))
}

#' @export
predict_scores.clf_majority <- function(object, newdata) {
  matrix(as.numeric(object$prior), nrow(newdata), length(object$prior), byrow = TRUE,
         dimnames = list(NULL, names(object$prior)))
}

# -- naive Bayes ------------------------------------------------------------

class_sums <- function(x, y) {
  y <- droplevels(y)
  indicator <- Matrix::sparseMatrix(i = as.integer(y), j = seq_along(y), x = 1,
                                    dims = c(nlevels(y), length(y)))
  sums <- as.matrix(indicator %*% x)
  rownames(sums) <- levels(y)
  sums
}

fit_nb <- function(x, y, alpha = 0.1, complement = FALSE) {
  if (any(x@x < 0)) {
    cli::cli_abort("Naive Bayes needs non-negative features.")
  }
  sums <- class_sums(x, y)
  if (complement) {
    comp <- sweep(-sums, 2, colSums(sums), "+") + alpha
    w <- log(comp / rowSums(comp))
    w <- -w / rowSums(abs(w))            # weight normalisation of Rennie et al.
    prior <- rep(0, nrow(sums))
  } else {
    w <- log((sums + alpha) / rowSums(sums + alpha))
    prior <- log(as.numeric(table(droplevels(y))) / length(y))
  }
  list(weights = w, prior = prior)
}

#' @export
predict_scores.clf_nb <- function(object, newdata) {
  sweep(as.matrix(newdata %*% t(object$weights)), 2, object$prior, "+")
}

# -- nearest centroid -------------------------------------------------------

fit_centroid <- function(x, y) {
  sums <- class_sums(x, y)
  list(centroids = sums / pmax(sqrt(rowSums(sums^2)), 1e-12))
}

#' @export
predict_scores.clf_centroid <- function(object, newdata) {
  as.matrix(l2_normalize(newdata) %*% t(object$centroids))
}

# -- LiblineaR --------------------------------------------------------------

as_csr <- function(x) {
  if (!inherits(x, "sparseMatrix")) {
    x <- methods::as(x, "CsparseMatrix")
  }
  r <- methods::as(methods::as(x, "generalMatrix"), "RsparseMatrix")
  methods::new("matrix.csr", ra = r@x, ja = r@j + 1L, ia = r@p + 1L, dimension = dim(r))
}

fit_liblinear <- function(x, y, loss = c("svm", "logreg"), cost = 1, multiclass = c("ovr", "cs"),
                          balanced = FALSE, bias = 1, epsilon = NULL) {
  loss <- rlang::arg_match(loss)
  multiclass <- rlang::arg_match(multiclass)
  type <- if (loss == "logreg") 0L else if (multiclass == "cs") 4L else 1L
  y <- droplevels(y)
  wi <- NULL
  if (balanced) {
    counts <- table(y)
    wi <- setNames(as.numeric(length(y) / (nlevels(y) * counts)), names(counts))
  }
  args <- list(data = as_csr(x), target = y, type = type, cost = cost, bias = bias, wi = wi, verbose = FALSE)
  if (!is.null(epsilon)) {
    args$epsilon <- epsilon
  }
  list(model = do.call(LiblineaR::LiblineaR, args), loss = loss)
}

predict_scores_liblinear <- function(object, newdata) {
  m <- object$model
  if (object$loss == "logreg") {
    p <- stats::predict(m, as_csr(newdata), proba = TRUE)$probabilities
    return(p[, m$ClassNames, drop = FALSE])
  }
  # Linear decision function computed directly: W is K x (p + bias)
  w <- m$W
  p <- ncol(newdata)
  scores <- as.matrix(newdata %*% t(w[, seq_len(p), drop = FALSE]))
  if (ncol(w) > p) {
    scores <- sweep(scores, 2, w[, p + 1L] * m$Bias, "+")
  }
  colnames(scores) <- if (nrow(w) == 1L) m$ClassNames[1] else as.character(m$ClassNames)
  if (nrow(w) == 1L) {   # binary problem: one hyperplane
    scores <- cbind(scores, -scores)
    colnames(scores) <- as.character(m$ClassNames)
  }
  scores
}

#' @export
predict_scores.clf_svm <- predict_scores_liblinear

#' @export
predict_scores.clf_logreg <- predict_scores_liblinear

# -- xgboost ----------------------------------------------------------------

fit_xgb <- function(x, y, nrounds = 200, eta = 0.2, max_depth = 6, subsample = 0.8,
                    colsample_bytree = 0.8, nthread = 2, ...) {
  rlang::check_installed("xgboost")
  y <- droplevels(y)
  dtrain <- xgboost::xgb.DMatrix(x, label = as.integer(y) - 1L, nthread = nthread)
  params <- list(objective = "multi:softprob", num_class = nlevels(y), eta = eta,
                 max_depth = max_depth, subsample = subsample,
                 colsample_bytree = colsample_bytree, tree_method = "hist", nthread = nthread, ...)
  list(model = xgboost::xgb.train(params, dtrain, nrounds = nrounds, verbose = 0),
       classes = levels(y))
}

#' @export
predict_scores.clf_xgb <- function(object, newdata) {
  p <- stats::predict(object$model, xgboost::xgb.DMatrix(newdata))
  p <- matrix(p, ncol = length(object$classes), byrow = !is.matrix(p))
  colnames(p) <- object$classes
  p
}
