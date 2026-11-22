# Staged Transfer Learning for Sentence-Level Lexical Bias Classification in News

Supplementary material for the undergraduate research project report (CSC406),
Department of Computer Science, Faculty of Science, University of Jaffna.

- **Author:** D. L. L. Dikkumbura
- **Supervisor:** Dr. E. Y. A. Charles
- **Submission version:** release `v1.0-submission` (November 2026)

## What this project does

The project asks whether staged transfer learning from existing annotated datasets improves
sentence-level lexical-bias detection on the BASIL news corpus. XLM-RoBERTa base is fine-tuned
in stages (Anno-lexical, then BABE, then BASIL) and compared with BASIL-only training and a
word-level (multi-task) variant. Evaluation uses an article-level split, validation-only
epoch and threshold selection, three seeds, three-seed probability ensembles and
story-level bootstrap confidence intervals.

## Repository contents

| Path | Description |
|---|---|
| `split_manifest.csv` | One row per BASIL sentence (7,977): sentence, article and story IDs, split, label, source outlet, fold. The `fold` column is a story-grouped fold index that was **not** used in the reported experiments. |
| `test_predictions.csv` | Saved test-set probabilities for every run (seeds 42, 43, 44) and ensemble. |
| `validation_predictions.csv` | Saved validation probabilities used for epoch and threshold selection. |
| `metrics_per_run.csv` | Per-seed and ensemble metrics (ROC-AUC, PR-AUC, precision, recall, F1, TP/FP/TN/FN), recomputed from the saved probabilities. Source of Appendix A. |
| `bootstrap_results.csv` | Bootstrap confidence intervals for the ensembles. |
| `confusion_matrices.csv` | Confusion matrices at the validation-selected thresholds. |
| `training_history.csv` | Per-epoch training history and selected epochs. |
| `word_level_sentence_summary.csv` | Word-level model summary (pointing accuracy). |
| `experiment_config.json` | Hyperparameters and settings used. |
| `scripts/experiment_source.py` | Experiment code. |
| `scripts/train_span_multitask.py` | Word-level (sentence + word-piece heads) model and training. |
| `scripts/make_figs.py` | Generates the report figures from the saved predictions. |
| `scripts/*.ipynb` | Original Colab notebook. |

## Setup used in the report

| Stage | Epochs | Learning rate | Batch size | Max length |
|---|---|---|---|---|
| Anno-lexical | 2 | 2e-5 | 32 | 128 |
| BABE | 3 | 1e-5 | 16 | 128 |
| BASIL | 8 | 1e-5 | 16 | 256 |

AdamW (weight decay 0.01), linear schedule with 10% warm-up, gradient clipping at 1.0, FP16.
BASIL training uses per-epoch negative sampling (all 307 positive training sentences plus five
newly sampled negatives per positive) with an unweighted loss. The Anno-lexical and BABE stages
were trained once (seed 42); the BASIL stage was repeated with seeds 42, 43 and 44.

## Reproducing the figures

The figures are drawn from the saved predictions; no model is retrained.

```bash
pip install pandas numpy matplotlib scikit-learn
python scripts/make_figs.py . figs
```

`make_figs.py` takes two arguments: the folder holding `test_predictions.csv` and
`split_manifest.csv`, and an output folder. In a notebook, set the two paths directly
(`D = "..."`, `O = "..."`) instead of using `sys.argv`.

## Data sources (not redistributed here)

The raw datasets are not included. Obtain them from their original sources and follow their licences:

- **BASIL:** Fan et al., "In Plain Sight: Media Bias Through the Lens of Factual Reporting", EMNLP-IJCNLP 2019.
- **BABE:** Spinde et al., "Neural Media Bias Detection Using Distant Supervision With BABE", Findings of EMNLP 2021.
- **Anno-lexical:** Horych et al., "The Promises and Pitfalls of LLM Annotations in Dataset Labeling: a Case Study on Media Bias Detection", Findings of NAACL 2025.

The prediction files contain sentence IDs and model probabilities only, not the sentence text.

## Important limitations

- **Story overlap.** Splits are by article, not by story: 66 of the 100 BASIL stories have articles in more than one split, and 47 of the 49 test stories have an article in the training set. Absolute scores may therefore be optimistic.
- **Small test set.** The test split has 92 biased sentences, so PR-AUC and F1 estimates are uncertain. ROC-AUC gains are supported by the paired bootstrap; PR-AUC and F1 gains are not.
- **External stages run once** (seed 42); only the BASIL stage was repeated across seeds.
- **Library versions were not recorded** in the Colab sessions.

## Licence

The code in this repository is released under the MIT licence (see `LICENSE`). The datasets remain
under their original licences.
