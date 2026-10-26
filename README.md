# Name That LLM

My work on the Kaggle competition [Name That LLM](https://www.kaggle.com/competitions/name-that-llm): given a short passage (40–250 words), predict which of 18 large language models wrote it.

- **Task:** 18-class text classification (authorship attribution)
- **Metric:** balanced accuracy (every model counts equally)
- **Key difficulty:** test prompts are unseen, so topic words don't help; the model has to learn writing *style*
- **Deadline:** 13 Jan 2027

## Project files

| File | What it is |
|---|---|
| `data/train.csv` | 2,699 labelled passages: `id`, `prompt_id`, `text`, `model` |
| `data/test.csv` | unlabelled passages (full set only available when Kaggle scores a submission) |
| `data/sample_submission.parquet` | submission format: `id`, `model` |
| `script.ipynb` | main notebook: loading, EDA, charts |

150 prompts, each answered by all 18 models (150 passages per model; `claude-fable-5` has 149).

## EDA findings