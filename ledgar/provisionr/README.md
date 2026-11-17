# provisionr

Reusable building blocks for classifying short legal texts (contract
provisions) with sparse linear models in R. Extracted from the LEDGAR study
in this repository so the workflow can be pointed at other corpora.

```r
# install.packages("devtools")
devtools::install("provisionr")   # from the ledgar/ directory
library(provisionr)
```

## What is in it

| Step | Functions |
|---|---|
| Data | `ledgar_download()`, `ledgar_read()`, `parquet_label_names()` |
| Cleaning | `normalize_text()`, `text_fingerprint()`, `flag_duplicates()`, `clean_corpus()` |
| Features | `feature_spec()`, `fit_featurizer()`, `bake_features()`, `text_stats()` |
| Models | `fit_classifier()` + `predict()` — engines `svm`, `logreg`, `nb`, `centroid`, `majority`, `xgb` |
| Resampling | `assign_folds()`, `cv_classifier()`, `summarise_cv()`, `target_shuffle()` |
| Metrics | `classification_metrics()`, `per_class_metrics()`, `top_k_accuracy()`, `confusion_pairs()`, `bootstrap_metrics()` |
| Model comparison | `compare_predictions()` (paired bootstrap + McNemar), `softmax_scores()`, `fit_temperature()` |
| Plots | `theme_provision()`, `plot_cv_summary()`, `plot_target_shuffle()`, `plot_confusion()`, `plot_per_class()` |

## Design rules

* **Nothing data-dependent leaks.** A `feature_spec()` holds no data.
  `fit_featurizer()` learns vocabularies, IDF weights and scaling from the
  texts it is given; `cv_classifier()` and `target_shuffle()` call it on the
  analysis rows of each fold only.
* **One model interface.** Every engine is fitted with `fit_classifier(x, y,
  engine = ...)` and returns class predictions or an n x K score matrix from
  `predict()`, always over the full label set.
* **Stratification that survives rare classes.** `assign_folds()` deals every
  class out across folds (and keeps `groups` together), instead of pooling
  small strata.
* **Metrics follow LexGLUE.** `accuracy` is micro-F1 for single-label
  problems; `macro_f1` averages per-class F1 with undefined ratios set to 0.

## Example

```r
library(provisionr)

corpus <- ledgar_read(ledgar_download("data/raw"), "train")
train  <- clean_corpus(corpus)                 # audit trail in attr(train, "cleaning_log")

spec  <- feature_spec(word_ngram = c(1, 2))
folds <- assign_folds(train$label, v = 5, seed = 1)

cv <- cv_classifier(
  train$text, train$label, folds, spec,
  models = list(svm = list(engine = "svm", cost = 1),
                cnb = list(engine = "nb", complement = TRUE)),
  workers = 4
)
summarise_cv(cv$metrics)

# Does the pipeline find signal where there is none?
ts <- target_shuffle(train$text[1:5000], train$label[1:5000],
                     assign_folds(train$label[1:5000], 5), spec,
                     model = list(engine = "svm"), n_perm = 100)
plot_target_shuffle(ts)

# Final model
fz  <- fit_featurizer(train$text, spec)
fit <- fit_classifier(bake_features(fz, train$text), train$label, engine = "svm")
predict(fit, bake_features(fz, "This Agreement shall be governed by the laws of the State of New York."))
```

## Development

```sh
Rscript -e 'roxygen2::roxygenise("provisionr")'   # regenerate NAMESPACE + man/
Rscript -e 'devtools::test("provisionr")'
R CMD build provisionr && R CMD check --no-manual provisionr_0.1.0.tar.gz
```
