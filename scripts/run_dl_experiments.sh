#!/usr/bin/env bash
# All deep learning experiments for the report. Run from the repo root:
#   bash scripts/run_dl_experiments.sh data/splits            (shared team split, preferred)
#   bash scripts/run_dl_experiments.sh data/raw/Resume.csv    (raw Kaggle file, split here with seed 42)
# Then: python -m src.dl.summarize   -> tables + chart in reports/
set -e
DATA=${1:-data/splits}   # use the aligned split folder (Phuc's split + Long's text_no_header column)
GLOVE=${GLOVE:-data/glove/glove.6B.300d.txt}
T="python -m src.dl.train --data $DATA --glove $GLOVE"

echo "== 1. Main comparison: 3 architectures x 2 embeddings (job title kept) =="
for m in bilstm textcnn cnn_bilstm; do
  for e in random glove; do $T --model $m --embed $e; done
done

echo "== 2. Label leakage: same models with the job title removed =="
for m in bilstm textcnn cnn_bilstm; do $T --model $m --embed glove --text-col text_no_header; done   # Long's header removal

echo "== 3. Ablations on BiLSTM + GloVe =="
$T --model bilstm --embed glove --rnn gru            # LSTM vs GRU
$T --model bilstm --embed glove --no-attention       # attention vs max-pooling
$T --model bilstm --embed glove --freeze             # frozen vs fine-tuned GloVe
for L in 256 1000; do $T --model bilstm --embed glove --max-len $L; done   # input length (512 is in part 1)
$T --model bilstm --embed glove --truncate head_tail # head vs head+tail truncation

echo "== 4. Stability: best setting with 2 more seeds (fill in the best model here) =="
for s in 7 123; do $T --model bilstm --embed glove --seed $s; done

python -m src.dl.summarize
