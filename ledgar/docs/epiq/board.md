# epiq board snapshot

Exported from the [epiq](https://github.com/ljtn/epiq) event log with `tools/epiq.py export`. Tickets are listed in creation order; commits whose subject starts with a ticket ref belong to that ticket (`git log --oneline --grep '^<ref>'`).

## Goals

| Goal | Done / total |
|---|---|
| 0 Setup | 1 / 1 |
| 1 Data | 1 / 1 |
| 2 Explore | 3 / 3 |
| 3 Clean | 1 / 1 |
| 4 Features | 1 / 1 |
| 5 Model | 5 / 5 |
| 6 Validate | 5 / 5 |
| 7 Package | 2 / 2 |
| 8 Transformer | 4 / 4 |

## Tickets

| Ref | Lane | Goal | Tags | Title |
|---|---|---|---|---|
| `5CJRS49` | Done | 0 Setup | setup | Bootstrap project: R toolchain, repo layout, epiq board |
| `JJQMBXW` | Done | 1 Data | data | Acquire LEDGAR (LexGLUE) and document provenance |
| `MESQXFW` | Done | 2 Explore | eda | EDA: label distribution and class imbalance |
| `CZ9GV3J` | Done | 2 Explore | eda | EDA: text length, vocabulary and duplicates |
| `2RV3J66` | Done | 2 Explore | eda | EDA: class separability and confusable label pairs |
| `G40T12E` | Done | 3 Clean | cleaning | Cleaning: normalise text, handle duplicates and leakage |
| `XWH411Z` | Done | 4 Features | features | Feature engineering: TF-IDF word/char n-grams + handcrafted features |
| `CB85EX1` | Done | 5 Model | modelling | Baselines: majority, multinomial NB, linear models |
| `7K7TEH9` | Done | 5 Model | modelling | Iterate: feature ablations under k-fold CV |
| `PP011SR` | Done | 5 Model | modelling | Iterate: model family comparison and hyper-parameter tuning |
| `194Q29G` | Done | 5 Model | modelling | Model selection and refit |
| `3VK9VCE` | Done | 6 Validate | validation | Validation: target shuffling (permutation) test |
| `H1JCGBW` | Done | 6 Validate | validation | Validation: held-out test evaluation with bootstrap CIs |
| `JDRQF0W` | Done | 6 Validate | validation | Validation: learning curve, error analysis, robustness |
| `4FHPWE5` | Done | 7 Package | packaging | Package reusable workflow as an R package with tests |
| `1THT29R` | Done | 7 Package | docs | Final report and README |
| `R1ATBW1` | Done | 6 Validate | validation, finding | Quantify temporal drift: does random CV overstate out-of-time performance? |
| `NW0W1WT` | Done | 6 Validate | validation, finding | Grouped CV: do near-duplicate templates inflate the CV estimate? |
| `XJBFAME` | Done | 5 Model | modelling, finding | Ablation follow-up: are block comparisons confounded by effective regularisation? |
| `ADZ558Q` | Done | 8 Transformer | setup | Fine-tuning setup: choose an open-source pretrained model that is trainable on 4 CPUs |
| `3HSD0VG` | Done | 8 Transformer | modelling | Fine-tune the transformer on LEDGAR train, select on validation |
| `52P1MWY` | Done | 8 Transformer | validation | Compare fine-tuned transformer with the TF-IDF + SVM model on test |
| `E9PHFDQ` | Done | 8 Transformer | docs | Update report and README with the transformer comparison |

### `5CJRS49` Bootstrap project: R toolchain, repo layout, epiq board

*Opened 2026-11-16 02:18 UTC — lane: Done*

Install R + packages, lay out the analysis project and the reusable R package, wire up epiq helper.

> **claude/ada**: R 4.3.3 from apt + CRAN source packages (text2vec, LiblineaR, rsample, yardstick, xgboost, nanoparquet). Layout: `provisionr/` (package), `analysis/` (numbered scripts), `results/`, `figures/`.

### `JJQMBXW` Acquire LEDGAR (LexGLUE) and document provenance

*Opened 2026-11-16 03:19 UTC — lane: Done*

Download the three official parquet splits from coastalcph/lex_glue, record checksums and label map.

> **claude/ada**: 80,000 provisions (60k/10k/10k), 100 labels, no missing values. sha256 per split in results/01_provenance.csv. Note: the HF CDN host had to be allow-listed in the environment.

### `MESQXFW` EDA: label distribution and class imbalance

*Opened 2026-11-16 03:19 UTC — lane: Done*

100 classes; how skewed? Are split proportions consistent?

> **claude/ada**: Imbalance 138x (23..3,167 per class), effective classes 74, majority baseline 5.3%. **Test label mix has drifted from train** (chi-sq p~1e-61) while validation is close - the LexGLUE split is chronological. Implication: CV on train is optimistic; opened a follow-up ticket. Figures: 02_label_counts, 02_label_concentration, 02_label_split_consistency.

### `CZ9GV3J` EDA: text length, vocabulary and duplicates

*Opened 2026-11-16 03:19 UTC — lane: Done*

Token/char lengths per class, Zipf, exact + near duplicates, cross-split leakage.

> **claude/ada**: Median 86 words (long tail to 1,215); class explains 21% of rank variance in length. Vocabulary 21.5k types, 3.7k purely numeric -> mask digits. **No exact duplicates, but 3.4% template repeats in train, 39 rows with conflicting labels, 5-6% of held-out rows share a template with train, ~15% have a >=0.95 cosine twin.** Near-duplication is uniform across splits. Follow-up: grouped CV to measure how much twins inflate CV.

### `2RV3J66` EDA: class separability and confusable label pairs

*Opened 2026-11-16 03:19 UTC — lane: Done*

Distinctive terms per class, class-centroid similarity, low-dim projection.

> **claude/ada**: Most classes are well separated in TF-IDF space; errors will concentrate in near-synonymous label pairs (Applicable/Governing Laws 0.97 centroid cosine, Defined Terms/Definitions 0.93, Tax Withholdings/Withholdings 0.92, Jurisdiction family). Figures 04_*.

### `G40T12E` Cleaning: normalise text, handle duplicates and leakage

*Opened 2026-11-16 03:19 UTC — lane: Done*

Turn EDA findings into a tested cleaning function; quantify what is removed.

> **claude/ada**: clean_corpus(): 60,000 -> 57,947 train rows (2 fragments, 39 label-conflict rows, 2,012 template repeats); held-out splits untouched but tagged `template_in_train`. Vocabulary 27.8k -> 17.5k types after lower-casing + digit masking. Open question carried to modelling: does de-duplication help or hurt held-out accuracy?

### `XWH411Z` Feature engineering: TF-IDF word/char n-grams + handcrafted features

*Opened 2026-11-16 03:19 UTC — lane: Done*

Sparse feature builders that are fit on training folds only.

> **claude/ada**: Feature blocks implemented in provisionr (word / char / head n-gram TF-IDF + 12 surface statistics), all fitted on training folds only. Evidence each block has signal: stats eta^2 7-32%; first 5 words recur for 70% of provisions and identify the label 76% of the time. Ablation under CV decides what stays (7K7TEH9).

### `CB85EX1` Baselines: majority, multinomial NB, linear models

*Opened 2026-11-16 03:19 UTC — lane: Done*

Establish a floor and a first honest CV estimate.

> **claude/ada**: 5-fold CV, word 1-2-gram TF-IDF: majority 5.3% | centroid 76.6% | NB 78.9% | complement NB 80.6% | logreg(C=1) 83.4% | **linear SVM(C=1) 87.3% acc / 81.6% macro-F1**. Fold SE ~0.1-0.2pp, so differences between families are real. LogReg at C=1 is over-shrunk - tune before judging.

### `7K7TEH9` Iterate: feature ablations under k-fold CV

*Opened 2026-11-16 03:19 UTC — lane: Done*

Which feature blocks earn their keep?

> **claude/ada**: Screening (20k subsample, SVM C=1, paired folds): nothing beats plain word 1-2-gram TF-IDF by more than noise. Clear losers: unigrams only (-1.2pp), raw tf (-0.4pp), no idf (-0.4pp), **head block (-0.3pp)**. Neutral: digit masking, min_doc_count=5 (halves features for free), surface stats, char n-grams (+0.07pp at 4x cost). Also: char n-grams with 4 forked workers OOM-killed the run -> per-variant caching + fewer workers.

> **claude/ada**: Screening done; see XJBFAME for the follow-up that overturned the head-block verdict.

### `PP011SR` Iterate: model family comparison and hyper-parameter tuning

*Opened 2026-11-16 03:19 UTC — lane: Done*

Linear SVM / logistic / NB-SVM / boosted trees on LSA; tune with CV.

> **claude/ada**: 24 configs / 7 families, paired 5-fold CV. Winner **linear SVM one-vs-rest C=1: 87.9% acc, 82.6% macro-F1**; Crammer-Singer C=1 is a statistical tie (-0.05pp, paired p=0.27) at ~3x less compute; logistic regression C=100 -0.27pp (p=0.007); NB families 3-6pp behind. Class-balanced weights do not help macro-F1.

### `194Q29G` Model selection and refit

*Opened 2026-11-16 03:19 UTC — lane: Done*

Pick by CV (not test), refit on train+validation.

> **claude/ada**: Selection by CV: linear SVM OvR C=1. Validation hold-out (untouched until now; fit on full train): **88.6% acc / 82.1% macro-F1**; family ranking agrees with CV (Spearman 0.95). Non-linear check on 100-d LSA: xgboost 81.0% vs linear SVM 77.0% on the same LSA features, both far below the sparse linear model (88.6%) - representation, not linearity, is what matters. De-duplicating train: 88.6% vs 88.7% raw -> no accuracy benefit; kept for honest fold assignment. Refit on train+validation: 67,336 rows.

### `3VK9VCE` Validation: target shuffling (permutation) test

*Opened 2026-11-16 03:19 UTC — lane: Done*

Null distribution of the CV metric under shuffled labels.

> **claude/ada**: 100 permutations x full 5-fold CV on 6k provisions: real labels 80.0% acc vs shuffled mean 2.3% (max 3.1%), z ~ 320, permutation p <= 0.0099. Null lies between frequency-guessing (1.8%) and majority rate (4.7%). In-sample accuracy on shuffled labels is 99.6% -> training fit is meaningless for this model class; resampling is essential.

### `H1JCGBW` Validation: held-out test evaluation with bootstrap CIs

*Opened 2026-11-16 03:19 UTC — lane: Done*

One-shot test evaluation; per-class metrics; confusion; calibration.

> **claude/ada**: Test (10k, scored once): **89.3% acc [88.8, 90.0], macro-F1 84.8% [83.4, 85.7]**, top-3 95.8%. Train-only fit for like-for-like with LexGLUE: 88.7 / 83.6 (published TF-IDF+SVM 87.2 / 82.4; RoBERTa-large 88.6 / 83.6). 685 test rows are textual twins of train/val rows (99% acc); novel text 88.6%. Margin gives 97.6% acc at 70% coverage.

### `JDRQF0W` Validation: learning curve, error analysis, robustness

*Opened 2026-11-16 03:19 UTC — lane: Done*

Does more data help? Where and why does it fail?

> **claude/ada**: Repeated CV: 88.0% +/- 0.26pp between folds, 0.11pp between repeats. Learning curve still rising ~2pp per doubling beyond 20k. Most confident errors are label-scheme collisions and catch-all classes (General, Miscellaneous).

### `4FHPWE5` Package reusable workflow as an R package with tests

*Opened 2026-11-16 03:19 UTC — lane: Done*

R CMD check clean, unit tests, roxygen docs.

> **claude/ada**: provisionr 0.1.0: 30 exported functions across data / clean / features / models / resample / metrics / plots, roxygen man pages, testthat suite (111 expectations, all pass), R CMD check clean apart from a container-locale warning. Makefile wraps setup / all / check / board.

### `1THT29R` Final report and README

*Opened 2026-11-16 03:19 UTC — lane: Done*

Rendered report with all figures and a reproducible pipeline entry point.

> **claude/ada**: reports/report.md is generated from results/*.csv and figures/ by reports/report.Rmd, so every number in the text is computed. README documents layout, protocol and how to reproduce (make setup / all / check / board).

### `R1ATBW1` Quantify temporal drift: does random CV overstate out-of-time performance?

*Opened 2026-11-16 03:23 UTC — lane: Done*

Raised by MESQXFW: test label mix differs from train (chronological split). Compare CV estimate vs validation vs test; consider prior-shift correction.

> **claude/ada**: No drift penalty found: train-only pipeline 88.0% CV -> 88.6% validation -> 88.7% test; re-weighting test to train label mix changes accuracy by -0.1pp. The significant label-mix shift (MESQXFW) does not translate into lower accuracy over these two periods.

### `NW0W1WT` Grouped CV: do near-duplicate templates inflate the CV estimate?

*Opened 2026-11-16 03:41 UTC — lane: Done*

Raised by CZ9GV3J. Compare stratified k-fold against group k-fold where template-duplicates stay in one fold.

> **claude/ada**: Grouped 5-fold (template twins kept together) on raw train: 88.1% vs 88.5% plain -> twins inflate naive CV by ~0.4pp; de-duplicating before CV was the right call. Test accuracy by nearest-train similarity: 73% (<0.3) ... 99% (>=0.95).

### `XJBFAME` Ablation follow-up: are block comparisons confounded by effective regularisation?

*Opened 2026-11-16 04:44 UTC — lane: Done*

Raised by 7K7TEH9. Adding a unit-norm block rescales document vectors (norm sqrt(2)), which acts like changing SVM C. Re-test head/stats/char blocks at several weights across C in {0.3,1,3} before discarding them.

> **claude/ada**: Result (20k subsample, 5-fold): the screening verdict on the head block was a confound. Accuracy by C (0.3 / 1 / 3): word only 83.4 / 83.9 / 83.8; +head w=1 83.7 / 83.7 / 83.4; **+head w=0.5 84.0 / 84.5 / 84.3**; +head w=0.25 83.6 / 84.2 / 84.1; +stats ~ no change; +char w=0.5 83.5 / 84.1 / 83.9 (not worth 4x compute). Next: tune head block design (words, n-gram range) then confirm on the full training split.

> **claude/ada**: Confirmed on full train (same folds as baselines): word 1-2 + head(w=0.5, first 20 words) -> **87.9% acc / 82.5% macro-F1** vs 87.3 / 81.6 for words alone; +stats adds nothing. Selected feature set recorded in chosen_spec(). Lesson: compare feature blocks only with regularisation re-tuned.

### `ADZ558Q` Fine-tuning setup: choose an open-source pretrained model that is trainable on 4 CPUs

*Opened 2026-11-17 01:30 UTC — lane: Done*

No GPU in this environment. Candidates: Legal-BERT-small (legal-domain, 35M params), MiniLM, DistilBERT. Benchmark throughput, fix the budget.

> **claude/ada**: No GPU, 4 CPUs. Throughput: Legal-BERT-small 22 rows/s @128 tokens (45 min/epoch), 15 rows/s @256; Legal-BERT-base 6 rows/s (2.6 h/epoch) -> infeasible. Chosen: nlpaueb/legal-bert-small-uncased (35M params, legal-domain pretraining, CC-BY-SA-4.0). Fine-tuning in Python/PyTorch (R torch needs a blocked download host); evaluation stays in R with provisionr.

### `3HSD0VG` Fine-tune the transformer on LEDGAR train, select on validation

*Opened 2026-11-17 01:30 UTC — lane: Done*

Train split only for weights; validation for epoch/hyper-parameter selection; test untouched until frozen.

> **claude/ada**: Pilot (6k train / 2k val, 3 epochs): lr 5e-5 74.1% | 1e-4 79.5% | 2e-4 80.2% | 4e-4 78.9% (all 128 tokens); lr 2e-4 @ 256 tokens 81.1%. Full run launched with lr 2e-4, 256 tokens, 3 epochs.

> **claude/ada**: Full run done in 177 min on 4 CPUs. Validation accuracy 80.9% -> 84.1% -> 85.8% over 3 epochs (macro-F1 68.7 -> 74.8 -> 77.5); epoch 3 selected; still improving, i.e. not converged within the CPU budget.

### `52P1MWY` Compare fine-tuned transformer with the TF-IDF + SVM model on test

*Opened 2026-11-17 01:30 UTC — lane: Done*

Same test rows, same metrics (provisionr), bootstrap CIs, paired test, per-class and agreement analysis, cost.

> **claude/ada**: Test, train-only fits: **SVM 88.7% / 83.6% vs Legal-BERT-small 86.1% / 78.1%** (acc / macro-F1); paired difference -2.56pp [-3.11, -2.01] and -5.47pp [-6.51, -4.41]; McNemar p=2e-20. Tied on unfamiliar text; SVM wins on boilerplate and long provisions. Ensemble +0.22pp. BERT better calibrated (ECE 0.8% vs 4.1%). BERT costs ~68x the training time. Caveat: BERT not converged (CPU budget).

### `E9PHFDQ` Update report and README with the transformer comparison

*Opened 2026-11-17 01:30 UTC — lane: Done*

Add a section to the report.

> **claude/ada**: Report section 7 + summary paragraph added; README documents the transformer workflow (venv, pilot, full run, make 14).
