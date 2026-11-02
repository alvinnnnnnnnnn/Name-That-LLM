# Name That LLM

My work on the Kaggle competition [Name That LLM](https://www.kaggle.com/competitions/name-that-llm): given a short passage (40–250 words), predict which of 18 large language models wrote it.

- **Task:** 18-class text classification (authorship attribution)
- **Metric:** balanced accuracy (every model counts equally)
- **Key difficulty:** test prompts are unseen, so topic words don't help; the model has to learn writing *style*
- **Deadline:** 13 Jan 2027

## Table of contents

- [Project files](#project-files)
- [EDA findings](#eda-findings)
  - [1. Sentence length (weak signal)](#1-sentence-length-weak-signal)
  - [2. Punctuation (strongest signal for families)](#2-punctuation-strongest-signal-for-families)
  - [3. Overused words (best clue for some twins)](#3-overused-words-best-clue-for-some-twins)
  - [Summary](#summary)
- [Model training](#model-training)
  - [Part 1: N-gram features + linear classifiers](#part-1-n-gram-features--linear-classifiers)
  - [Part 2: Encoder models](#part-2-encoder-models)

## Project files

| File | What it is |
|---|---|
| `data/train.csv` | 2,699 labelled passages: `id`, `prompt_id`, `text`, `model` |
| `eda.ipynb` | eda notebook: EDA, charts |
| `part1.ipynb`| part 1 notebook: model training (N-gram features + linear classifiers) |

150 prompts, each answered by all 18 models (150 passages per model; `claude-fable-5` has 149).

## EDA findings

### 1. Sentence length (weak signal)

- Average sentence length is similar across models (20.7–26.9 words), and so is the average number of sentences per passage (4.3–5.8).
- How much sentence length **varies** is more useful:
  - Steady, even sentences: `gemini-3.1-pro-preview`, `gpt-6-astra` (std 7.3), `qwen3.8-max` (7.4).
  - Mix of very long and very short: `claude-opus-5` (12.7), `claude-opus-4-6` (11.8), `kimi-k3` (11.4), `glm-5.3` (11.0).
- `gpt-6-astra`, `gemini-3.1-pro-preview` and `qwen3.8-max` never write a sentence longer than about 50 words. Every other model except `gpt-5.6-luna` (68) has at least one sentence of 93+ words.
- `gemini-3.1-pro-preview` and `gpt-6-astra` never write a one-sentence passage. `gpt-5.5` does it most often (8 passages).
- Prompt `p076` produced mostly single-sentence answers (1.6 sentences on average vs a median of 4.8), so some prompts force a structure.

### 2. Punctuation (strongest signal for families)

Measured per 100 words, because passage length is random.

- **Apostrophes split the families.** Claude and GLM never use curly `’` (0.00), only straight `'`. GPT-5.6 is the opposite. Kimi is the only model that mixes both.
- **Quotes mirror apostrophes:** curly `“` mainly from GPT and Kimi, straight `"` mainly from Claude and GLM.
- **Em-dashes (`—`):** heavy for Claude and GLM (0.87–1.55), moderate for GPT, almost zero for Gemini and Qwen.
- **Spaced em-dash (` — `):** only `claude-opus-4-6`, `claude-opus-5` and the two GLMs ever use it.
- **Gemini uses the least punctuation:** fewest colons (0.16–0.18) and hyphens. `gemini-3.1-pro-preview` uses the most full stops and fewest commas.
- **Twins differ only slightly:**
  - GPT-5.6: Luna uses the most em-dashes and brackets, Sol the most curly quotes and semicolons, Terra the most colons.
  - GLM-flash uses more semicolons than GLM (0.54 vs 0.38).
  - Claude: Opus 4.6 uses the fewest semicolons, Opus 5 the most; Sonnet uses the most straight apostrophes and hyphens.
  - Gemini flash: 3.6 uses more straight than curly apostrophes; 3.7 and 3.8 use more curly.

### 3. Overused words (best clue for some twins)

Words each model uses much more often than the other 17, keeping only words used across at least 10 prompts.

- **Gemini has the most distinctive vocabulary** (top words used 4–12× more than other models). Claude and `gpt-5.5` have the weakest word habits (mostly 2–3×).
- **Family fingerprints:**
  - Gemini: `profound`, `colossal`, `vibrant`, `abyss` (almost never used by other families).
  - GPT-5.6 and Astra: `therefore`, `familiar`, `seem`.
  - Claude and GLM: `nearly`. Gemini almost never hedges with `perhaps`.
- **Twins:**
  - Gemini: 3.6-flash sounds upbeat (`dynamic`, `vibrant`), 3.7-flash cosmic (`celestial`, `primordial`), 3.8-flash dark (`abyss`, `void`), 3.1-pro violent (`jagged`, `chaotic`).
  - Claude: Fable uses absolutes (`everything`, `nothing`), Opus 4.6 sounds scientific (`roughly`, `molecular`), Opus 4.8 gentle (`quietly`, `you`), Opus 5 tells stories with numbers (`thing`, `ten`, `he`), Sonnet talks about minds (`structure`, `intelligence`).
  - GPT-5.6: the twins share most of their top words (Sol and Terra share 7 of 15). Each has one strong tell: Luna `gathers`, Sol `travel`, Terra `crowded`.
  - GLM: glm-5.3 is about time and scale (`billion`, `instant`), glm-5.3-flash sounds explanatory (`scientists`, `whether`).
- Caveat: a few topic words still slip through (`weather`, `cathedral`, `mathematics`).

### Summary

- **Families are easy to tell apart,** mainly through punctuation.
- **Twins are the hard part.** Word choice helps for the Gemini and Claude models, but the GPT-5.6 twins and the two GLMs share both punctuation and vocabulary.
- Sentence length adds a little, mainly for the "steady sentence" models (`gemini-3.1-pro-preview`, `gpt-6-astra`, `qwen3.8-max`).

## Model training

### Part 1: N-gram features + linear classifiers

Each passage is turned into numbers by a vectorizer, then a linear classifier learns which patterns belong to which model. No text cleaning is done (no lowercasing, punctuation removal, stopword removal or lemmatising), because the EDA showed that punctuation and small words are the signal.

#### Vectorizers

Both vectorizers combine two views with `make_union`:

| View | What it captures | Linked EDA finding |
|---|---|---|
| **Character n-grams** (2–5 characters) | punctuation and typing habits, e.g. `’s`, ` — `, `; t` | punctuation separates the families |
| **Word n-grams** (single words and pairs, punctuation kept as tokens) | vocabulary and phrasing | word choice helps separate some twins |

- **TF-IDF** weights each pattern by how often it appears, scales each passage for its length, and down-weights patterns every passage uses.
- **Count** uses raw counts, with no length scaling or down-weighting.

#### Classifiers

- **LinearSVC** (`C=0.5`): a linear support vector machine, the standard choice for large sparse text features.
- **Logistic regression** (`C=10`, `max_iter=2000`): also linear, and it outputs probabilities, which will be needed later for blending models.
- Both use `class_weight='balanced'`, so all 18 models count equally, matching the balanced-accuracy metric.

#### Validation

- **5-fold GroupKFold on `prompt_id`:** each fold holds out whole prompts, so the model is always tested on questions it has never seen, just like the real test set.
- Every passage gets one out-of-fold prediction, and balanced accuracy is computed over all 2,699 of them.

#### Results

| Vectorizer | Classifier | Balanced accuracy |
|---|---|---|
| TF-IDF | Logistic regression | **0.346** |
| TF-IDF | LinearSVC | 0.338 |
| Count | Logistic regression | 0.307 |
| Count | LinearSVC | 0.293 |

Random guessing scores about 0.056.

Recall per model (share of its passages predicted correctly), sorted by the best combination:

| Model | TF-IDF + LR | TF-IDF + SVM | Count + LR | Count + SVM |
|---|---|---|---|---|
| `claude-opus-5` | 0.65 | 0.67 | 0.45 | 0.39 |
| `gemini-3.1-pro-preview` | 0.65 | 0.74 | 0.67 | 0.64 |
| `gpt-6-astra` | 0.49 | 0.54 | 0.49 | 0.50 |
| `gemini-3.6-flash` | 0.45 | 0.43 | 0.37 | 0.35 |
| `gemini-3.7-flash` | 0.45 | 0.43 | 0.39 | 0.36 |
| `claude-opus-4-6` | 0.41 | 0.42 | 0.37 | 0.33 |
| `gpt-5.6-luna` | 0.37 | 0.28 | 0.22 | 0.17 |
| `qwen3.8-max` | 0.37 | 0.35 | 0.31 | 0.31 |
| `claude-sonnet-5` | 0.36 | 0.35 | 0.27 | 0.30 |
| `gemini-3.8-flash` | 0.33 | 0.31 | 0.32 | 0.28 |
| `gpt-5.5` | 0.31 | 0.28 | 0.26 | 0.27 |
| `kimi-k3` | 0.29 | 0.29 | 0.23 | 0.23 |
| `gpt-5.6-sol` | 0.23 | 0.21 | 0.15 | 0.17 |
| `gpt-5.6-terra` | 0.23 | 0.23 | 0.26 | 0.24 |
| `glm-5.3` | 0.19 | 0.17 | 0.22 | 0.22 |
| `claude-opus-4-8` | 0.18 | 0.14 | 0.23 | 0.20 |
| `glm-5.3-flash` | 0.17 | 0.16 | 0.17 | 0.15 |
| `claude-fable-5` | 0.11 | 0.08 | 0.13 | 0.16 |

#### What the results show

- **TF-IDF beats Count by about 0.04** with both classifiers. Passage length is random (40–250 words), and raw counts let long passages look different just because they are long. TF-IDF's length scaling removes that.
- **Logistic regression and LinearSVC score almost the same** (0.346 vs 0.338). A difference this small is within the noise of a 150-prompt dataset. Logistic regression is kept as the main model because it gives probabilities.
- **The same models are easy or hard in every combination.** `gemini-3.1-pro-preview` (0.64–0.74) and `gpt-6-astra` (0.49–0.54) are always recognised best, matching their distinctive style in the EDA. `claude-fable-5`, `glm-5.3-flash`, `claude-opus-4-8` and the GPT-5.6 twins are always among the worst. The weak classes come from the data, not from the choice of model.
- **`claude-opus-5` is a catch-all with TF-IDF.** Its recall is high (0.65–0.67) but its precision is low (0.35–0.37): many Claude and GLM passages get labelled as Opus 5. With Count features its recall drops to 0.39–0.45 and the other Claude and GLM models rise slightly, but the overall score is lower.

#### What the confusion matrix shows (TF-IDF + SVM)

- **Gemini is easy to recognise as a family.** About 97% of Gemini passages are predicted as some Gemini model. The flash models mostly get mixed up with each other (e.g. 33% of 3.8-flash is predicted as 3.7-flash).
- **Claude and GLM act like one family.** They share straight apostrophes, heavy em-dashes and spaced em-dashes, and the model can't separate them. Both GLMs are predicted as Claude more often than as GLM, and `claude-opus-5` absorbs 25% of `glm-5.3`, 21% of `claude-fable-5` and 15% of `claude-opus-4-8`.
- **The GPT-5.6 twins spread their predictions** across each other and `gpt-6-astra`, consistent with them sharing most of their vocabulary.
- **`gpt-5.5`, `kimi-k3` and `qwen3.8-max` get mixed up.** All three use curly apostrophes and none has strong word habits.

#### Why the scores are limited

- **Little data:** each model has only 150 passages, and about 120 per fold for training.
- **Near-identical twins:** some models share both punctuation and vocabulary, leaving few patterns to separate them.
- **Short passages:** at 40 words, many style clues simply don't appear.
- **Surface patterns only:** n-grams count characters and words, but don't capture sentence structure or tone.
- **Comparison:** public notebooks report about 0.43 for this kind of model, but trained on the official data plus large community-generated datasets.

#### Score on test data (on Kaggle)
TF-IDF + Logistic Regression scored 0.35 on test data

### Part 2: Encoder models