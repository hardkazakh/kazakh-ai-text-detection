# Kazakh AI-Text Detection

Detecting **AI-generated** and **AI-obfuscated** text in **Kazakh** — a low-resource, Cyrillic-script language. This repository benchmarks five models on a three-way classification task and includes the full pipeline: exploratory data analysis, leakage-free data curation, and seed-ensembled transformer fine-tuning with statistical significance testing.

## Task

Given a passage of Kazakh text, classify it into one of three classes:

| Label | Class | Meaning |
|:-----:|-------|---------|
| 0 | **Human** | Original human-written text |
| 1 | **AI-Generated** | Text produced by an LLM |
| 2 | **AI-Obfuscated** | An AI paraphrase / obfuscation of existing text |

## Headline result

The best model, **mDeBERTa-v3 (base)**, reaches **0.917 macro-F1** (0.909 accuracy, 0.985 one-vs-rest ROC-AUC) on the held-out test set. XLM-R (base) is statistically indistinguishable from it; both clearly beat the DistilmBERT and TF-IDF baselines.

## Leaderboard (test set, n = 900)

| Model | Accuracy | Macro-F1 | Weighted-F1 | ROC-AUC (OvR) | Macro-F1 (seed mean ± std) |
|-------|:-------:|:-------:|:-----------:|:-------------:|:--------------------------:|
| **mDeBERTa-v3 (base)** | **0.909** | **0.917** | **0.909** | **0.985** | 0.901 ± 0.008 |
| XLM-R (base) | 0.906 | 0.915 | 0.905 | 0.981 | 0.883 ± 0.019 |
| mBERT (cased) | 0.891 | 0.901 | 0.891 | 0.975 | 0.876 ± 0.014 |
| DistilmBERT | 0.857 | 0.869 | 0.856 | 0.961 | 0.843 ± 0.012 |
| TF-IDF + LogReg | 0.771 | 0.776 | 0.771 | 0.917 | — (single run) |

Transformers were fine-tuned across three seeds (42, 1, 2); the leaderboard's headline columns report the best (seed 42) run, with per-seed mean ± std shown in the last column. Full numbers are in [`results/tables/table2_overall_results.csv`](results/tables/table2_overall_results.csv).

### Statistical significance

95% bootstrap confidence intervals ([`table3_bootstrap_ci.csv`](results/tables/table3_bootstrap_ci.csv)) and Holm-corrected McNemar tests ([`table4_mcnemar_holm.csv`](results/tables/table4_mcnemar_holm.csv)) tell a consistent story:

- The top three models (mDeBERTa-v3, XLM-R, mBERT) are **not** significantly different from one another on this test set.
- All three transformers significantly outperform **DistilmBERT** and the **TF-IDF baseline** (Holm-corrected *p* < 0.05).

### Performance by text length

Accuracy scales with input length ([`table5_length_stratified_f1.csv`](results/tables/table5_length_stratified_f1.csv)). Short inputs (≤30 tokens) are the hardest for every model; on long inputs (>120 tokens) the transformers approach or reach perfect macro-F1, while the TF-IDF baseline actually degrades.

## Dataset

Leakage-audited, stratified splits (see [`data/README.md`](data/README.md) for the full data card):

| Split | Human | AI-Generated | AI-Obfuscated | Total |
|-------|:-----:|:------------:|:-------------:|:-----:|
| train | 1609 | 1889 | 700 | 4198 |
| dev | 345 | 405 | 150 | 900 |
| test | 345 | 405 | 150 | 900 |
| **Total** | **2299** | **2699** | **1000** | **5998** |

`data/raw/` holds the original pooled splits; `data/cleaned/` holds the curated, deduplicated, leakage-free splits produced by notebook 02 and used for all reported results.

## Repository layout

```
.
├── notebooks/
│   ├── 01_Kazakh_AI_Text_EDA.ipynb            # Exploratory data analysis
│   ├── 02_Kazakh_AI_Data_Cleaning.ipynb       # Curation, dedup, leakage-free re-split
│   └── 03_Kazakh_AI_Detection_KerasNLP_v5.ipynb  # Fine-tuning + evaluation + stats
├── data/
│   ├── raw/       # Original pooled splits
│   ├── cleaned/   # Curated splits used for reported results
│   └── README.md  # Data card
└── results/
    ├── figures/       # fig1–fig6 (PNG)
    ├── tables/        # table1–table6, per-model classification reports
    ├── metrics/       # all_metrics.json, per_seed_scores.json
    ├── predictions/   # per-model test-set predictions (CSV)
    └── run_manifest.json
```

## Pipeline

Run the notebooks in order:

1. **`01_Kazakh_AI_Text_EDA.ipynb`** — class semantics, length/vocabulary analysis, tokenization, embedding visualization, statistical testing, and outlier detection over the raw splits.
2. **`02_Kazakh_AI_Data_Cleaning.ipynb`** — Unicode-safe cleaning for Kazakh, quality filtering, exact + normalized deduplication, a cross-split leakage audit, and a stratified re-split. Exports `data/cleaned/` plus a removal log and data card.
3. **`03_Kazakh_AI_Detection_KerasNLP_v5.ipynb`** — full fine-tuning of each backbone (seed-ensembled), a TF-IDF + Logistic Regression reference baseline, and the full evaluation suite (bootstrap CIs, Holm-corrected McNemar tests, length-stratified error analysis).

## Training configuration

| Setting | Value |
|---------|-------|
| Framework | KerasNLP 0.29.1 / Keras 3.15.0 (JAX backend) |
| Strategy | Full fine-tuning |
| Seeds | 42, 1, 2 |
| Sequence length | 256 |
| Epochs | 5 |
| Batch size | 16 |
| Learning rate | 2e-5 |

See [`results/run_manifest.json`](results/run_manifest.json) for the exact run metadata.

## Figures

| File | Shows |
|------|-------|
| `fig1_dataset_overview.png` | Dataset composition and class balance |
| `fig2_training_curves.png` | Training / validation curves |
| `fig3_model_comparison.png` | Model comparison across metrics |
| `fig4_confusion_matrices.png` | Per-model confusion matrices |
| `fig5_bootstrap_ci.png` | Macro-F1 with 95% bootstrap CIs |
| `fig6_length_stratified_f1.png` | Macro-F1 by input length band |

## Reproducing

```bash
pip install -r requirements.txt
jupyter lab   # then run notebooks 01 → 02 → 03 in order
```

Heavy EDA cells (transformer tokenizer, sentence-transformer embeddings, t-SNE/UMAP) are gated behind a `RUN_HEAVY` flag in notebook 01's config cell.

## License

Code is released under the MIT License (see [`LICENSE`](LICENSE)). **The dataset may carry its own terms** — confirm the source data's license and usage rights before redistributing (see the note in [`data/README.md`](data/README.md)).
