# LEDGAR contract-provision classifier

An end-to-end text-classification study in R on **LEDGAR** (LexGLUE): 80,000
contract provisions from SEC filings, each labelled with one of 100 provision
types. It covers exploration, cleaning, feature engineering, iterative model
selection and a deliberately sceptical evaluation (k-fold cross-validation,
target shuffling, two untouched hold-outs, bootstrap intervals).

**Read the results:** [`reports/report.md`](reports/report.md)

**Reusable code:** the [`provisionr`](provisionr/) R package.

## Layout

```
ledgar/
├── provisionr/        R package: data access, cleaning, features, models, CV,
│                      target shuffling, metrics, plots (+ tests, man pages)
├── analysis/          numbered scripts, one per step of the study
│   ├── _setup.R         shared preamble + the decisions carried between steps
│   ├── 01_acquire.R     download LEDGAR, checksums, label map
│   ├── 02_eda_labels.R  class imbalance, split drift
│   ├── 03_eda_text.R    length, vocabulary, artefacts, duplicates / leakage
│   ├── 04_eda_separability.R  distinctive terms, class similarity, UMAP
│   ├── 05_clean.R       cleaning with an audit log
│   ├── 06_features.R    what each feature block looks like
│   ├── 07_baselines.R   majority / NB / centroid / linear models, 5-fold CV
│   ├── 08_ablation.R    feature ablations (screen -> follow-up -> confirm)
│   ├── 09_tuning.R      model families x hyper-parameters, 5-fold CV
│   ├── 10_select_refit.R  selection, validation hold-out check, refit
│   ├── 11_target_shuffle.R  permutation test of the whole pipeline
│   ├── 12_test_evaluation.R  one-shot test evaluation, bootstrap CIs
│   ├── 13_robustness.R  repeated/grouped CV, drift, learning curve, errors
│   └── 14_transformer_comparison.R  fine-tuned Legal-BERT-small vs the SVM
├── transformer/       Python: finetune.py (PyTorch), pilot.sh, train_full.sh
├── results/           every table the scripts produce (CSV, committed)
├── figures/           every figure the scripts produce (PNG, committed)
├── reports/           report.Rmd -> report.md, built from results/ + figures/
├── docs/epiq/         snapshot of the epiq issue board for this project
├── tools/             epiq.py (ticket CLI), install_deps.R
└── Makefile           `make setup`, `make all`, `make check`, `make board`
```

`data/` (raw parquet files, cached intermediates) and `models/` (fitted
objects) are git-ignored; `make all` recreates them.

## Reproducing

Requires R ≥ 4.1, a C/C++ toolchain, and network access to `huggingface.co`
and its CDN (`us.aws.cdn.hf.co`) for the download step.

```sh
cd ledgar
make setup     # install CRAN dependencies and provisionr
make all       # run steps 01..13 and render the report (about 2 h on 4 cores)
make check     # unit tests + R CMD check of provisionr
```

Single steps run with `make 07` or `Rscript analysis/07_baselines.R` (from this
directory). Expensive steps cache per-configuration results under
`data/processed/`, so an interrupted run resumes.

## Transformer comparison

`transformer/finetune.py` fine-tunes `nlpaueb/legal-bert-small-uncased` on the
train split (epoch chosen on validation, test scored once);
`analysis/14_transformer_comparison.R` compares it with the SVM on the same
test rows. It needs a Python virtualenv and about 3.5 CPU-hours:

```sh
python3 -m venv .venv && .venv/bin/pip install -r transformer/requirements.txt
bash transformer/pilot.sh        # optional: learning-rate / length pilot (~1.2 h)
bash transformer/train_full.sh   # ~3 h on 4 cores, resumable
make 14 && make report
```

## Evaluation protocol

| Data | Used for |
|---|---|
| train (60,000; 57,947 after cleaning) | EDA, cleaning rules, 5-fold CV for every feature and model decision |
| validation (10,000) | first hold-out: checks that CV rankings transfer; settles the de-duplication question; learning curve. Merged into the final fit afterwards |
| test (10,000) | scored only after the model was frozen; no decision depends on it |

Target shuffling re-runs the complete CV with permuted labels to show what the
pipeline scores when there is nothing to learn.

## How the work was tracked

* **Git:** one commit per step or iteration; the subject starts with the ticket
  ref, the body records what was found.
* **epiq:** the board lives in epiq's own event log (`__epiq_state__` branch).
  `tools/epiq.py` drives it non-interactively through the `epiq-mcp` server;
  a readable snapshot is committed in [`docs/epiq/board.md`](docs/epiq/board.md),
  with the raw event log beside it.

```sh
git log --oneline --grep '^XJBFAME'    # commits belonging to one ticket
python3 tools/epiq.py list             # live board (needs `npm i -g epiq`)
```

## Data and citation

LEDGAR: Tuggener, von Däniken, Peetz & Cieliebak (2020), *LEDGAR: A Large-Scale
Multi-label Corpus for Text Classification of Legal Provisions in Contracts*,
LREC. Distributed as part of LexGLUE: Chalkidis et al. (2022), *LexGLUE: A
Benchmark Dataset for Legal Language Understanding in English*, ACL
(`coastalcph/lex_glue` on the Hugging Face hub).
