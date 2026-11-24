# Deep learning models: BiLSTM, TextCNN, CNN+BiLSTM

Owner: Yuvaan. This part trains the three recurrent and convolutional models for resume
category classification (24 classes) and runs the embedding, architecture and leakage experiments.

## Files

| File | What it does |
|---|---|
| `src/dl/text.py` | Light cleaning (masks emails/phones/URLs), tokeniser that keeps `c++`, `c#`, `node.js`, optional job-title removal |
| `src/dl/data.py` | Loads the shared split (or splits the raw CSV with seed 42), builds the vocabulary from **train only**, pads batches |
| `src/dl/embeddings.py` | Random vs. GloVe embedding matrix, plus GloVe coverage of the vocabulary |
| `src/dl/models.py` | `RNNClassifier` (BiLSTM/BiGRU + attention or max-pool), `TextCNN`, `CNNBiLSTM` |
| `src/dl/train.py` | Trains one model: class-weighted loss, AdamW, early stopping on validation macro-F1, then evaluates on the test set |
| `src/dl/summarize.py` | Builds the comparison tables and chart from `reports/results.csv` |
| `src/dl/predict.py` | Loads a trained run for the demo: top-3 categories and attention words |
| `scripts/run_dl_experiments.sh` | Runs every experiment for the report |

## Setup (Mac)

```bash
pip install -r requirements.txt
# GloVe (about 820 MB zip; we only need the 300d file)
mkdir -p data/glove && cd data/glove
curl -LO https://nlp.stanford.edu/data/glove.6B.zip && unzip glove.6B.zip glove.6B.300d.txt && cd ../..
```

Training uses the Mac GPU (Apple MPS) automatically, then CUDA, then the CPU. Force one with `--device cpu`.

## Data

Use the **shared split from Phúc**: a folder with `train.csv`, `val.csv` and `test.csv` that holds the columns
`Resume_str` and `Category` (otherwise pass `--text-col` / `--label-col`). If the split isn't ready yet,
point `--data` at the raw Kaggle `Resume.csv`. The code then splits it the same way the team agreed:
drop duplicates, stratified 80/10/10, `random_state=42`.

## Run

One model:
```bash
python -m src.dl.train --data data/splits --model bilstm --embed glove
python -m src.dl.train --data data/splits --model textcnn --embed random
python -m src.dl.train --data data/splits --model cnn_bilstm --embed glove --remove-title
```

All experiments, then the tables:
```bash
bash scripts/run_dl_experiments.sh data/splits
python -m src.dl.summarize
```

Useful flags: `--rnn gru`, `--no-attention`, `--freeze`, `--remove-title`, `--max-len 256|512|1000`,
`--truncate head_tail`, `--no-clean` (text already preprocessed), `--seed`, `--epochs`, `--patience`.

## Outputs

Each run writes `runs/<run-name>/`:
- `best.pt`, `config.json`, `vocab.json`: the trained model (used by the demo)
- `history.csv`, `curves.png`: loss and validation curves for the report
- `test_report.txt`: per-class precision / recall / F1
- `confusion_matrix.png`: row-normalised 24×24 test confusion matrix
- `test_predictions.csv`: true, predicted, confidence, top-3 and the start of the text, **for error analysis**

Every run appends one row to `reports/results.csv`, using the same columns as the baseline table
(Model, Representation, Val Macro-F1, Test Macro-F1, Test Accuracy) plus extras: job title kept or removed,
max length, parameters, training time, seed.

## Experiments → report

| # | Experiment | Question it answers |
|---|---|---|
| 1 | BiLSTM, TextCNN, CNN+BiLSTM × random / GloVe | Which architecture and which representation work best? |
| 2 | Same models with the job title removed | How much do the models depend on the title (label leakage)? |
| 3 | LSTM vs. GRU; attention vs. max-pool; frozen vs. fine-tuned GloVe | Which design choices matter? |
| 4 | Max length 256 / 512 / 1000; head vs. head+tail | How much of the resume is needed? |
| 5 | Best model with 3 seeds | Are the differences larger than random variation? |

## Default settings (for the Methodology section)

- Embeddings 300-d (random N(0, 0.1) or GloVe 6B, words GloVe doesn't know stay random), vocabulary from train, min frequency 2, max 30k words
- BiLSTM / BiGRU: 2 layers, hidden size 128 per direction, additive attention (or masked max-pool), dropout 0.4
- TextCNN: filter widths 3/4/5, 100 filters each, max-over-time pooling, dropout 0.5
- CNN+BiLSTM: Conv1d (128 filters, width 3) → 1-layer BiLSTM (128) → attention
- Training: AdamW (lr 1e-3, weight decay 1e-4), class-weighted cross-entropy, gradient clipping 1.0,
  batch size 32, lr halved on plateau, early stopping (patience 5) on validation macro-F1, max 30 epochs, max length 512 tokens

## Demo use (for Kriti)

```python
from src.dl.predict import load_run
clf = load_run("runs/bilstm_lstm_att_glove_title_len512_s42")
clf.predict(text)       # [("INFORMATION-TECHNOLOGY", 0.81), ...]
clf.top_words(text)     # words with the highest attention, for highlighting
```
