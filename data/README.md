# Data Card

Kazakh three-class text corpus for AI-text detection. Each record is a JSON object:

```json
{ "text": "...", "label": 0, "source": "Conversion_text" }
```

| Field | Type | Description |
|-------|------|-------------|
| `text` | string | The Kazakh passage (Cyrillic script) |
| `label` | int | 0 = Human, 1 = AI-Generated, 2 = AI-Obfuscated |
| `source` | string | Provenance tag the label is derived from |

## Label semantics

The label is inferred from the `source` field and confirmed with the researcher:

| Label | Class | Typical `source` values |
|:-----:|-------|-------------------------|
| 0 | Human | `*_text` (e.g. `Conversion_text`, `Summary_text`, `Reading_text`) |
| 1 | AI-Generated | `*_response` (e.g. `Story_response`, `Reading_response`) |
| 2 | AI-Obfuscated | `Paraphrase_text2` |

In the cleaned splits the `source` values are normalized to short category names (`Story`, `Reading`, `Summary`, `Conversion`, `Paraphrase`).

## Files

### `raw/` — original pooled splits
| File | Records |
|------|:-------:|
| `train_set.json` | 6298 |
| `dev_set.json` | 1350 |
| `test_set.json` | 1350 |

### `cleaned/` — curated splits (used for all reported results)
Produced by `notebooks/02_Kazakh_AI_Data_Cleaning.ipynb`: Unicode-safe cleaning, quality filtering, exact + normalized deduplication, a cross-split leakage audit, and a stratified re-split.

| File | Records | Human | AI-Generated | AI-Obfuscated |
|------|:-------:|:-----:|:------------:|:-------------:|
| `train_cleaned.json` | 4198 | 1609 | 1889 | 700 |
| `dev_cleaned.json` | 900 | 345 | 405 | 150 |
| `test_cleaned.json` | 900 | 345 | 405 | 150 |

## Source composition (cleaned)

| Source | train | dev | test |
|--------|:-----:|:---:|:----:|
| Conversion | 2113 | 449 | 438 |
| Paraphrase | 700 | 150 | 150 |
| Summary | 696 | 135 | 168 |
| Reading | 426 | 95 | 78 |
| Story | 263 | 71 | 66 |

## ⚠️ Licensing note

Confirm the license and usage rights of the underlying source texts before redistributing this dataset publicly. The MIT license in the repository root covers the **code**, not necessarily the **data**.
