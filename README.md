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

## Robustness analyses (2026 revision)

Six controlled analyses were added in response to peer review. Three support the benchmark as
constructed; three constrain what it can be used to claim. All of them are reproducible from this
repository — notebooks 04, 05 and 06 need no GPU and no retraining, because the files in
`results/predictions/` are row-aligned with `data/cleaned/test_cleaned.json`.

| Notebook | Question | Result | Tables |
|---|---|---|---|
| `04` | Is the AI-obfuscated class recognised by its shortness? | **No.** On a strictly length-matched subset (390 docs, 130/class, median 39 words in *every* class) macro-F1 falls only 0.026–0.057, while a word-count-only control collapses 0.416 → 0.233 | `results_r2/table_A1*`, `table_A2` |
| `04` | How does Kazakh morphology interact with the 256-token limit? | Fertility 1.96 (XLM-R) / 2.41 (mDeBERTa) / 2.93 (mBERT, DistilmBERT) sub-tokens per word, near-constant across classes. Truncation is **not**: 0% of AI-obfuscated vs 25–30% of human documents | `results_r2/table_A6`, `table_A7` |
| `05` | Is there near-duplicate leakage across splits? | **None.** MinHash LSH (128 perms) + exact Jaccard verification, thresholds 0.5–0.9, word-5-gram and char-8-gram shingles: zero cross-split pairs everywhere, zero contaminated test documents | `results_r2/table_A3`, `table_A4` |
| `06` | How much signal is basic surface statistics? | 31 surface features reach 0.715; TF-IDF 0.775; TF-IDF + surface 0.841 — still 0.076 below the best transformer ensemble | `results_r2/table_A5*` |
| `09` | Does the model ranking survive reseeding? | **No.** Over 30 runs, seed SD is 0.034 (mDeBERTa) and 0.217 (XLM-R, including one degenerate run), larger than every gap between encoders. Wilcoxon signed-rank on paired per-seed scores separates no pair (all *p*<sub>Holm</sub> = 1.00), and the top-two ensemble ordering flips | `results_r2_seeds/*` |
| `08` | Does detection transfer to an unseen generator? | **No.** Against Mistral-7B-Instruct-v0.3, binary macro-F1 drops 0.897 → 0.692 and machine recall falls to **0.475**, while human recall is preserved at 0.916 | `results_r2_crossgen/table_crossgenerator.csv` |
| `07` | What does LLM paraphrasing do to the decision? | A single paraphrasing pass does **not** evade the binary decision (AI arm 82.4% → 85.7%, *p* = 0.374). But only 4.6–6.8% of genuine paraphrases receive the AI-obfuscated label, so that class recognises the corpus's own paraphrasing procedure rather than paraphrasing in general | `results_r2_paired/*` |

### Run order

Run `00_setup_and_check.ipynb` once, then:

| Order | Notebook | Runtime | Hardware |
|:---:|---|---|---|
| 1 | `04_length_control_and_tokenizer_fertility` | ~5 min | CPU |
| 2 | `05_near_duplicate_audit_minhash` | ~4 min | CPU |
| 3 | `06_surface_feature_baseline` | ~2 min | CPU |
| 4 | `09_multiseed_finetuning` | ~6 h (30 runs) | GPU — **run before 07 and 08**; saves the weights they load |
| 5 | `07_paired_paraphrase_design` | ~1 h | GPU |
| 6 | `08_cross_generator_ood_eval` | ~1 h | GPU |

Notebooks 07, 08 and 09 append every finished unit to disk and skip it on re-run.

### Models used

| Role | Model | Notes |
|---|---|---|
| Corpus generator | `Qwen/Qwen2.5-7B-Instruct`, `Qwen/Qwen2.5-32B-Instruct-AWQ` | as in the original benchmark |
| Unseen generator (notebook 08) | `mistralai/Mistral-7B-Instruct-v0.3` | the Kazakh-native candidates (KazLLM, Sherkala) are gated repositories; access was not granted |
| Paraphraser (notebook 07) | `Qwen/Qwen2.5-7B-Instruct` | same family as the corpus generator — isolates paraphrasing with generator family held constant; **not** an unseen-paraphraser test |

### Known limitations of the added analyses

- The original 3,300-prompt suite was unavailable for notebook 08, so prompt distribution changes
  alongside the generator; the measured drop is an **upper bound** on the generator effect.
- No Kazakh POS tagger of characterised accuracy on this text mixture was available, so notebook 06
  omits POS features and exposes a hook instead.
- Sensitivity to the 256-token context limit itself (a longer window, or chunking) is untested.

## Reproducing

```bash
pip install -r requirements.txt
jupyter lab   # then run notebooks 01 → 02 → 03 in order
```

Heavy EDA cells (transformer tokenizer, sentence-transformer embeddings, t-SNE/UMAP) are gated behind a `RUN_HEAVY` flag in notebook 01's config cell.

## License

Code is released under the MIT License (see [`LICENSE`](LICENSE)). **The dataset may carry its own terms** — confirm the source data's license and usage rights before redistributing (see the note in [`data/README.md`](data/README.md)).
