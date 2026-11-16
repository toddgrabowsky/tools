#' provisionr: sparse text classification workflows for contract provisions
#'
#' The package is organised around the steps of a text-classification study:
#'
#' * **Data** -- [ledgar_download()], [ledgar_read()].
#' * **Cleaning** -- [normalize_text()], [text_fingerprint()],
#'   [flag_duplicates()], [clean_corpus()].
#' * **Features** -- [feature_spec()], [fit_featurizer()], [bake_features()].
#' * **Models** -- [fit_classifier()] and its `predict()` method.
#' * **Evaluation** -- [cv_classifier()], [target_shuffle()],
#'   [bootstrap_metrics()], [classification_metrics()], [per_class_metrics()].
#' * **Plots** -- [theme_provision()] and the `plot_*()` family.
#'
#' @keywords internal
#' @importFrom rlang .data .env %||%
#' @importFrom methods as new
#' @importClassesFrom SparseM matrix.csr
#' @importFrom stats predict setNames
"_PACKAGE"
