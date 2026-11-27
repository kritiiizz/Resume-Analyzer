# Resume Analyzer

NLP course project (HCMUT, Semester I 2026–2027). The system classifies a resume into one of
24 job categories and compares it with a job description.

Team: Kriti Sharma, Phúc Hà Nguyễn Bảo, Long Phạm Trương Hoàng, Yuvaan Chottani.

## Task and data

- **Task:** text classification, resume text → job category (24 classes).
- **Dataset:** [Kaggle Resume Dataset](https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset)
  (2,482 resumes; columns `ID`, `Resume_str`, `Resume_html`, `Category`). We use `Resume_str` as input
  and `Category` as the label.
- **Split:** duplicates removed, stratified 80 / 10 / 10, `random_state=42`. Every model uses the same split.
- **Label leakage:** most resumes start with their job title, which nearly gives the category away,
  so each model is also trained with the title removed.

## Models

| Family | Models |
|---|---|
| Baselines | Dummy, BoW + Multinomial NB, TF-IDF + Logistic Regression, TF-IDF + LinearSVC |
| RNN / CNN | BiLSTM / BiGRU with attention, TextCNN, CNN + BiLSTM; random vs. GloVe embeddings |
| Transformer | DistilBERT fine-tuned with head + tail truncation |

## Results (test set, 249 resumes)

Mean ± standard deviation over 3 random seeds (the classical baselines have no randomness, so one run each).

| Model | Macro-F1 | Accuracy |
|---|---|---|
| Dummy (most frequent class) | 0.004 | 0.048 |
| BoW + Multinomial NB | 0.519 | 0.582 |
| TF-IDF + Logistic Regression | 0.578 | 0.643 |
| TF-IDF + LinearSVC | 0.604 | 0.671 |
| DistilBERT, head + tail (128 + 382 tokens) | 0.732 ± 0.011 | 0.823 (seed 42) |
| TextCNN + GloVe | 0.757 ± 0.020 | 0.813 |
| BiGRU + attention + GloVe | 0.807 ± 0.026 | 0.853 |
| **CNN + BiLSTM + attention + GloVe** | **0.815 ± 0.006** | **0.861** |

**Job-title leakage.** Most resumes start with the job title. Removing it lowers macro-F1 by only 1–4 points
for the baselines and our RNN/CNN models, but by about 19 points for DistilBERT
(0.732 ± 0.011 → 0.540 ± 0.010), which relied mostly on the title.

**Errors.** About half of the best model's 32 test errors are resumes whose text belongs to a different field
than their label (e.g. a registered-nurse resume labelled FITNESS), so part of the remaining error comes from the
dataset labels. See `reports/error_analysis/`.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Put the data here (not committed to git; shared through the team Google Drive):

```
data/raw/Resume.csv                 Kaggle file
data/splits/train.csv val.csv test.csv   shared split
data/glove/glove.6B.300d.txt        GloVe vectors (https://nlp.stanford.edu/projects/glove/)
```

## How to run

```bash
# Deep learning models (details in docs/DL_MODELS.md)
python -m src.dl.train --data data/splits --model bilstm --embed glove
bash scripts/run_dl_experiments.sh data/splits      # every experiment
python -m src.dl.summarize                          # tables + chart in reports/

# Error analysis for any model's test predictions
python -m src.error_analysis --pred runs/<run>/test_predictions.csv --name bilstm_glove

# Demo
streamlit run app/streamlit_app.py
```

The demo lists every model it finds in `runs/` (RNN/CNN runs), `models/*.joblib` (scikit-learn pipelines)
and `models/<folder>/` (fine-tuned Hugging Face model). `src/export_for_demo.py` shows how to save them.

## Repository layout

```
app/streamlit_app.py     demo
src/dl/                  BiLSTM, TextCNN, CNN+BiLSTM: data, models, training, prediction
src/registry.py          one interface over all trained models (used by the demo)
src/skills.py            rule-based skill matching between resume and job description
src/error_analysis.py    per-class scores, confused pairs, error tags, file for manual labelling
src/export_for_demo.py   how to save baselines / DistilBERT / predictions for the demo
src/llm_baseline/        original LLM-based parser and matcher (optional zero-shot baseline)
scripts/                 experiment runner
docs/DL_MODELS.md        settings and experiments of the RNN/CNN models
reports/                 results.csv, figures, error analysis
```

## Credits

The project started from [kritiiizz/Resume-Analyzer](https://github.com/kritiiizz/Resume-Analyzer) (MIT licence),
an LLM-based resume parser. Its code is kept in `src/llm_baseline/`; the dataset pipeline, trained models,
evaluation and demo were built for this course project.
